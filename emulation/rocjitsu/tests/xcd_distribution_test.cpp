// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file xcd_distribution_test.cpp
/// @brief How a single AQL dispatch is spread across the XCDs of a multi-XCD SoC.

#include "aql_queue.h"
#include "test_paths.h"

#include "embedded_schema.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"
#include "simdojo/sim/topology.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace {

using namespace rocjitsu;

const std::string CONFIG_PATH = test::config_path("gfx950_mi355x.json");

constexpr uint32_t kTotalXcds = 8;
constexpr uint32_t kCusPerXcd = 36; // 4 SEs x 9 CUs
constexpr uint32_t kTotalCus = kTotalXcds * kCusPerXcd;
constexpr uint64_t kKdAddr = 0x10000;
constexpr uint32_t kWavefrontSize = 64;

/// A loaded gfx950 SoC plus a trivial s_endpgm kernel resident in GPU memory.
struct XcdDistributionFixture {
  config::LoadedConfig loaded;
  std::unique_ptr<simdojo::SimulationEngine> engine;
  SoC *soc = nullptr;
  amdgpu::GpuMemory *memory = nullptr;

  XcdDistributionFixture() : loaded(config::load_config(CONFIG_PATH, rocjitsu::kEmbeddedSchema)) {
    soc = loaded.soc();
    memory = loaded.memory();
    engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
    engine->topology().set_root(loaded.take_root());
    loaded.wire_links(engine->topology());
    engine->create();

    using namespace rocr::llvm::amdhsa;
    kernel_descriptor_t kd{};
    kd.kernel_code_entry_byte_offset = sizeof(kernel_descriptor_t);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                    ((256 / 8) - 1)); // CDNA4 VGPR granularity is 8
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                    ((104 / 8) - 1));
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);
    memory->load_image(reinterpret_cast<const uint8_t *>(&kd), sizeof(kd), kKdAddr);
    memory->write32(kKdAddr + sizeof(kernel_descriptor_t),
                    build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));
  }
};

/// Index of the XCD whose command processor is @p cp.
uint32_t assigned_xcd_index(const SoC &soc, const amdgpu::CommandProcessor *cp) {
  for (uint32_t xi = 0; xi < soc.num_xcds(); ++xi)
    if (soc.xcd(xi)->command_processor() == cp)
      return xi;
  ADD_FAILURE() << "command processor does not belong to this SoC";
  return 0;
}

} // namespace

// Fan-out is a property of the queue, not of the device. A queue registered
// directly against one command processor, as a test that wants that command
// processor's CUs to itself does, keeps the whole grid on that one XCD. The KFD
// path opts in; see FanoutQueueGridSpreadsOverAllXcds for that behavior.
TEST(XcdDistributionTest, QueueWithoutFanoutKeepsGridOnOneXcd) {
  XcdDistributionFixture fx;
  ASSERT_EQ(fx.soc->num_xcds(), kTotalXcds);
  ASSERT_EQ(fx.soc->all_cus().size(), kTotalCus);

  auto *cp = fx.soc->assign_queue_owner_cp(/*queue_ordinal=*/0);
  ASSERT_NE(cp, nullptr);
  test::AqlQueue queue(fx.memory, cp);
  queue.dispatch(kKdAddr, kTotalCus * kWavefrontSize, kWavefrontSize);

  fx.engine->run();

  auto counts = fx.soc->dispatched_workgroups_per_xcd();
  ASSERT_EQ(counts.size(), kTotalXcds);
  EXPECT_EQ(std::accumulate(counts.begin(), counts.end(), uint64_t{0}), kTotalCus);

  size_t xcds_used = 0;
  for (auto count : counts)
    xcds_used += count > 0 ? 1 : 0;
  EXPECT_EQ(xcds_used, 1u) << "expected the whole grid confined to one XCD";

  // ...and it must be the XCD the queue was actually assigned to. Cardinality
  // alone would still pass if the grid ran on the wrong one.
  EXPECT_EQ(counts[assigned_xcd_index(*fx.soc, cp)], kTotalCus);
}

// A queue marked for fan-out spreads each dispatch over every XCD, round-robin
// one workgroup at a time. The permutation is part of the contract: kernels that
// swizzle their workgroup index for cache locality assume workgroup i runs on XCD
// i % num_xcds.
TEST(XcdDistributionTest, FanoutQueueGridSpreadsOverAllXcds) {
  XcdDistributionFixture fx;
  ASSERT_EQ(fx.soc->num_xcds(), kTotalXcds);

  auto *cp = fx.soc->assign_queue_owner_cp(/*queue_ordinal=*/0);
  ASSERT_NE(cp, nullptr);
  test::AqlQueue queue(fx.memory, cp, test::AqlQueue::DEFAULT_RING_ADDR,
                       test::AqlQueue::DEFAULT_RING_SIZE, test::AqlQueue::DEFAULT_READ_PTR_ADDR,
                       test::AqlQueue::DEFAULT_WRITE_PTR_ADDR,
                       test::AqlQueue::DEFAULT_DOORBELL_ADDR, /*xcd_fanout=*/true);
  queue.dispatch(kKdAddr, kTotalCus * kWavefrontSize, kWavefrontSize);

  fx.engine->run();

  auto counts = fx.soc->dispatched_workgroups_per_xcd();
  ASSERT_EQ(counts.size(), kTotalXcds);
  EXPECT_EQ(std::accumulate(counts.begin(), counts.end(), uint64_t{0}), kTotalCus);
  for (uint32_t xi = 0; xi < kTotalXcds; ++xi)
    EXPECT_EQ(counts[xi], kTotalCus / kTotalXcds) << "xcd" << xi;
}

// The split must not depend on which XCD the queue landed on: rank is the XCD's
// own index, so the workgroup-to-XCD mapping is the same for every queue.
TEST(XcdDistributionTest, FanoutIsIndependentOfOwningXcd) {
  XcdDistributionFixture fx;

  // Ask for an ordinal that lands the queue on an XCD other than xcd0.
  auto *cp = fx.soc->assign_queue_owner_cp(/*queue_ordinal=*/3);
  ASSERT_NE(cp, nullptr);
  ASSERT_NE(cp, fx.soc->xcd(0)->command_processor());

  test::AqlQueue queue(fx.memory, cp, test::AqlQueue::DEFAULT_RING_ADDR,
                       test::AqlQueue::DEFAULT_RING_SIZE, test::AqlQueue::DEFAULT_READ_PTR_ADDR,
                       test::AqlQueue::DEFAULT_WRITE_PTR_ADDR,
                       test::AqlQueue::DEFAULT_DOORBELL_ADDR, /*xcd_fanout=*/true);
  queue.dispatch(kKdAddr, kTotalCus * kWavefrontSize, kWavefrontSize);

  fx.engine->run();

  auto counts = fx.soc->dispatched_workgroups_per_xcd();
  for (uint32_t xi = 0; xi < kTotalXcds; ++xi)
    EXPECT_EQ(counts[xi], kTotalCus / kTotalXcds) << "xcd" << xi;
}

// A grid with fewer workgroups than XCDs is not split: an empty share is
// indistinguishable from a barrier packet and would retire the dispatch early.
TEST(XcdDistributionTest, GridSmallerThanXcdCountIsNotSplit) {
  XcdDistributionFixture fx;

  auto *cp = fx.soc->assign_queue_owner_cp(/*queue_ordinal=*/0);
  ASSERT_NE(cp, nullptr);
  test::AqlQueue queue(fx.memory, cp, test::AqlQueue::DEFAULT_RING_ADDR,
                       test::AqlQueue::DEFAULT_RING_SIZE, test::AqlQueue::DEFAULT_READ_PTR_ADDR,
                       test::AqlQueue::DEFAULT_WRITE_PTR_ADDR,
                       test::AqlQueue::DEFAULT_DOORBELL_ADDR, /*xcd_fanout=*/true);
  constexpr uint32_t kWgs = kTotalXcds - 1;
  queue.dispatch(kKdAddr, kWgs * kWavefrontSize, kWavefrontSize);

  fx.engine->run();

  auto counts = fx.soc->dispatched_workgroups_per_xcd();
  EXPECT_EQ(std::accumulate(counts.begin(), counts.end(), uint64_t{0}), kWgs);
  size_t xcds_used = 0;
  for (auto count : counts)
    xcds_used += count > 0 ? 1 : 0;
  EXPECT_EQ(xcds_used, 1u);
}

// Queue ownership still rotates across XCDs. With fan-out that no longer decides
// where the work runs, but it does spread ring reads and completion signalling.
TEST(XcdDistributionTest, QueuesRotateAcrossXcds) {
  XcdDistributionFixture fx;
  ASSERT_EQ(fx.soc->num_xcds(), kTotalXcds);
  ASSERT_EQ(fx.soc->all_cus().size(), kTotalCus);

  constexpr uint32_t kWgsPerQueue = kCusPerXcd;
  std::vector<std::unique_ptr<test::AqlQueue>> queues;
  for (uint32_t qi = 0; qi < kTotalXcds; ++qi) {
    auto *cp = fx.soc->assign_queue_owner_cp(qi);
    ASSERT_NE(cp, nullptr);
    uint64_t ring = 0xF0000000ULL + qi * 0x100000ULL;
    queues.push_back(std::make_unique<test::AqlQueue>(fx.memory, cp, ring, 4096, ring + 0x10000,
                                                      ring + 0x10008, ring + 0x10010));
    queues.back()->dispatch(kKdAddr, kWgsPerQueue * kWavefrontSize, kWavefrontSize);
  }

  fx.engine->run();

  auto counts = fx.soc->dispatched_workgroups_per_xcd();
  ASSERT_EQ(counts.size(), kTotalXcds);
  for (uint32_t xi = 0; xi < kTotalXcds; ++xi)
    EXPECT_EQ(counts[xi], kWgsPerQueue) << "xcd" << xi;
}

// The counter is documented as a lifetime running total, so a second dispatch on
// the same queue must add to it rather than replace it. Both other tests submit
// one packet per command processor, so they would still pass if the counter were
// reset or overwritten per packet; this samples the histogram around the second
// dispatch and checks the delta as well as the accumulated total.
TEST(XcdDistributionTest, CounterAccumulatesAcrossDispatchesOnOneQueue) {
  XcdDistributionFixture fx;
  ASSERT_EQ(fx.soc->num_xcds(), kTotalXcds);
  ASSERT_EQ(fx.soc->all_cus().size(), kTotalCus);

  auto *cp = fx.soc->assign_queue_owner_cp(/*queue_ordinal=*/0);
  ASSERT_NE(cp, nullptr);
  const uint32_t xi = assigned_xcd_index(*fx.soc, cp);

  constexpr uint32_t kFirstWgs = kCusPerXcd;
  constexpr uint32_t kSecondWgs = kCusPerXcd / 2;

  test::AqlQueue queue(fx.memory, cp);
  queue.dispatch(kKdAddr, kFirstWgs * kWavefrontSize, kWavefrontSize);
  queue.dispatch(kKdAddr, kSecondWgs * kWavefrontSize, kWavefrontSize);
  fx.engine->run();

  auto counts = fx.soc->dispatched_workgroups_per_xcd();
  ASSERT_EQ(counts.size(), kTotalXcds);
  EXPECT_EQ(counts[xi], kFirstWgs + kSecondWgs)
      << "counter must accumulate across packets, not restart per packet";
  EXPECT_EQ(std::accumulate(counts.begin(), counts.end(), uint64_t{0}),
            uint64_t{kFirstWgs} + kSecondWgs);
}
