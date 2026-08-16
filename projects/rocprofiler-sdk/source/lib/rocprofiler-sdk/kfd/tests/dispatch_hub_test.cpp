// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Unit tests for the signal-less pending-completion hub, exercised without a GPU,
// the HSA runtime, or the reader thread.

#include "lib/rocprofiler-sdk/kfd/dispatch_hub.hpp"
#include "lib/rocprofiler-sdk/kfd/complete_signal_less_dispatch.hpp"
#include "lib/rocprofiler-sdk/kfd/owner_registry.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include <algorithm>
#include <cstdlib>
#include <map>
#include <random>
#include <set>

#include <sys/wait.h>
#include <unistd.h>

namespace
{
using namespace rocprofiler::kfd;

// Payload that tracks its own lifetime, so a test can prove exactly one owner at
// all times and that cleanup runs exactly once (invariant 6, U17).
struct tracked_payload
{
    static std::atomic<int> live;
    uint64_t                id    = 0;
    bool                    armed = false;  // true while this instance owns the resource

    tracked_payload() = default;
    explicit tracked_payload(uint64_t v)
    : id{v}
    , armed{true}
    {
        ++live;
    }
    tracked_payload(tracked_payload&& rhs) noexcept
    : id{rhs.id}
    , armed{rhs.armed}
    {
        rhs.armed = false;
    }
    tracked_payload& operator=(tracked_payload&& rhs) noexcept
    {
        if(this != &rhs)
        {
            if(armed) --live;
            id        = rhs.id;
            armed     = rhs.armed;
            rhs.armed = false;
        }
        return *this;
    }
    tracked_payload(const tracked_payload&) = delete;
    tracked_payload& operator=(const tracked_payload&) = delete;
    ~tracked_payload()
    {
        if(armed) --live;
    }
};

std::atomic<int> tracked_payload::live = {0};

using hub_t = DispatchHub<tracked_payload>;

correlation_key
key_of(uint32_t slot, uint32_t dispatch_id, uint32_t generation = 0)
{
    return correlation_key{slot, dispatch_id, generation};
}

hub_t::registration
reg_of(correlation_key key, uint64_t corr_id = 1, uint64_t payload_id = 0)
{
    auto r           = hub_t::registration{};
    r.key            = key;
    r.correlation_id = corr_id;
    r.payload        = tracked_payload{payload_id != 0 ? payload_id : key.dispatch_idx_low32};
    return r;
}

// The hub exposes no global live count, so tests sum over the small key space.
size_t
live_entries(const hub_t& hub)
{
    size_t n = 0;
    for(uint32_t gpu = 0; gpu < 4; ++gpu)
        for(uint32_t slot = 0; slot < 64; ++slot)
            n += hub.pending_for_slot(gpu, slot);
    return n;
}

// A slot is quarantined exactly when no key on it is admissible any more.
bool
slot_quarantined(const hub_t& hub, uint32_t gpu, uint32_t slot)
{
    return !hub.can_register_batch({correlation_key{slot, 0xFEEDu, 0u, gpu}});
}

// Register a batch of dispatches (all with the same corr_id); returns acceptance.
bool
register_many(hub_t& hub, std::initializer_list<correlation_key> keys, uint64_t corr_id = 1)
{
    auto batch = std::vector<hub_t::registration>{};
    for(auto k : keys)
        batch.emplace_back(reg_of(k, corr_id));
    return hub.register_batch(std::move(batch));
}

bool
register_one(hub_t& hub, correlation_key key, uint64_t corr_id = 1)
{
    return register_many(hub, {key}, corr_id);
}
}  // namespace

// Registration / admission: the state machine's legal transitions and rejections.
TEST(DispatchHub, registration_and_admission)
{
    {  // U3b/U9: a result before its registration is rejected, not cached.
        auto hub = hub_t{};
        auto key = key_of(4, 9);
        EXPECT_FALSE(hub.record_kernel_end(key, 2000, true).has_value()) << "no-owner EOP";
        ASSERT_TRUE(register_one(hub, key));
        EXPECT_EQ(live_entries(hub), 1u) << "stale result not retained";
        auto p = hub.record_kernel_end(key, 5000, true);
        ASSERT_TRUE(p.has_value());
        EXPECT_EQ(p->end_ticks, 5000u) << "fresh record, not stale";
    }
    {  // U8: START is retained until a terminal transition -- no 5 s eviction.
        auto hub = hub_t{};
        auto key = key_of(4, 11);
        ASSERT_TRUE(register_one(hub, key));
        EXPECT_TRUE(hub.record_kernel_start(key, 42));
        auto p = hub.record_kernel_end(key, 99, true);
        ASSERT_TRUE(p.has_value());
        ASSERT_TRUE(p->start_ticks.has_value());
        EXPECT_EQ(*p->start_ticks, 42u);
    }
    {  // An unmatched START is dropped.
        auto hub = hub_t{};
        EXPECT_FALSE(hub.record_kernel_start(key_of(4, 12), 1));
        EXPECT_EQ(live_entries(hub), 0u);
    }
    {  // Invariant 1/U2: a batch registers all-or-none. A last-entry collision
       // (key_of(4,2) already live) must insert none of the batch.
        auto hub = hub_t{};
        ASSERT_TRUE(register_many(hub, {key_of(4, 1), key_of(4, 2), key_of(4, 3)}));
        EXPECT_EQ(live_entries(hub), 3u);
        EXPECT_FALSE(register_many(hub, {key_of(4, 10), key_of(4, 11), key_of(4, 2)}))
            << "last-entry collision";
        EXPECT_EQ(live_entries(hub), 3u) << "no partial registration";
        EXPECT_FALSE(hub.record_kernel_end(key_of(4, 10), 1, true).has_value());
        EXPECT_FALSE(hub.record_kernel_end(key_of(4, 11), 1, true).has_value());
    }
    {  // Invariant 3: a duplicate key within one batch is rejected, not overwrite.
        auto hub = hub_t{};
        EXPECT_FALSE(register_many(hub, {key_of(4, 1), key_of(4, 1)}));
        EXPECT_EQ(live_entries(hub), 0u);
    }
    {  // can_register_batch: the enqueue-side pre-check for eligibility.
        auto hub = hub_t{};
        EXPECT_FALSE(hub.can_register_batch({key_of(4, 1), key_of(4, 1)})) << "dup in batch";
        ASSERT_TRUE(register_one(hub, key_of(4, 2)));
        EXPECT_FALSE(hub.can_register_batch({key_of(4, 2)})) << "live key";
        hub.quarantine_slot(0, 6);
        EXPECT_FALSE(hub.can_register_batch({key_of(6, 1)})) << "quarantined slot";
        EXPECT_TRUE(hub.can_register_batch({key_of(7, 1)}));
        hub.set_mode(session_mode::stopping);
        EXPECT_FALSE(hub.can_register_batch({key_of(7, 1)})) << "stopping session";
    }
    {  // Same slot/index/generation on two GPUs = two independent dispatches.
        auto hub     = hub_t{};
        auto on_gpu0 = correlation_key{40, 1, 0, /*gpu_id=*/0};
        auto on_gpu1 = correlation_key{40, 1, 0, /*gpu_id=*/1};
        auto batch   = std::vector<hub_t::registration>{};
        batch.emplace_back(reg_of(on_gpu0, /*corr_id=*/10, /*queue_token=*/1));
        batch.emplace_back(reg_of(on_gpu1, /*corr_id=*/20, /*queue_token=*/2));
        ASSERT_TRUE(hub.register_batch(std::move(batch))) << "two GPUs != duplicate key";
        EXPECT_EQ(live_entries(hub), 2u);
        auto p0 = hub.record_kernel_end(on_gpu0, 500, true);
        ASSERT_TRUE(p0.has_value());
        EXPECT_EQ(p0->key.gpu_id, 0u);
        EXPECT_EQ(live_entries(hub), 1u);
        auto p1 = hub.record_kernel_end(on_gpu1, 600, true);
        ASSERT_TRUE(p1.has_value());
        EXPECT_EQ(p1->key.gpu_id, 1u);
        EXPECT_EQ(p1->end_ticks, 600u) << "GPU 1 must get its own record";
    }
}

// Unit 3: EOP shapes.
TEST(DispatchHub, completion_shapes)
{
    {  // Shape (ii): an EOP with a lost START still proves the unique PENDING entry.
        auto hub = hub_t{};
        auto key = key_of(4, 50);
        ASSERT_TRUE(register_one(hub, key));
        auto p = hub.record_kernel_end(key, 900, /*drain_loss_free=*/true);
        ASSERT_TRUE(p.has_value());
        EXPECT_FALSE(p->start_ticks.has_value());
        EXPECT_EQ(p->end_ticks, 900u);
        EXPECT_EQ(live_entries(hub), 0u);
    }
    {  // The same EOP under a lossy drain proves nothing: the record may be torn.
        auto hub = hub_t{};
        auto key = key_of(4, 51);
        ASSERT_TRUE(register_one(hub, key));
        EXPECT_FALSE(hub.record_kernel_end(key, 900, /*drain_loss_free=*/false).has_value());
        EXPECT_EQ(live_entries(hub), 1u) << "still pending, still matchable";
    }
}

// The same result-vs-loss race under real concurrency: a prover racing a thread
// that keeps quarantining the slot. Each key resolves exactly once.
TEST(DispatchHub, concurrent_prove_vs_leak_resolves_each_key_once)
{
    constexpr uint32_t kCount = 512;

    auto hub = hub_t{};
    for(uint32_t i = 0; i < kCount; ++i)
    {
        ASSERT_TRUE(register_one(hub, key_of(4, i)));
    }
    const int live_after_register = tracked_payload::live.load();
    EXPECT_EQ(live_after_register, static_cast<int>(kCount));

    auto proven_n = std::atomic<uint32_t>{0};
    auto leaked_n = std::atomic<uint32_t>{0};

    auto prover = std::thread{[&hub, &proven_n]() {
        for(uint32_t i = 0; i < kCount; ++i)
            if(hub.record_kernel_end(key_of(4, i), 7, true).has_value()) ++proven_n;
    }};
    auto leaker = std::thread{[&hub, &leaked_n]() {
        for(uint32_t i = 0; i < kCount; ++i)
            leaked_n += hub.quarantine_slot(0, 4).size();
    }};
    prover.join();
    leaker.join();

    EXPECT_EQ(proven_n.load() + leaked_n.load(), kCount) << "exactly one winner per key";
    EXPECT_EQ(live_entries(hub), 0u);
    EXPECT_EQ(tracked_payload::live.load(), 0) << "each payload destroyed once";
}

// Quarantine, leak ledger, and child-epoch short-circuit.
TEST(DispatchHub, quarantine_leak_and_ledger)
{
    {  // U12: quarantine leaks pending work and blocks reservation permanently.
        auto hub = hub_t{};
        ASSERT_TRUE(register_one(hub, key_of(4, 1)));
        ASSERT_TRUE(register_one(hub, key_of(4, 2)));
        ASSERT_TRUE(register_one(hub, key_of(9, 1)));  // different slot survives
        auto lost = hub.quarantine_slot(0, 4);
        EXPECT_EQ(lost.size(), 2u);
        EXPECT_TRUE(slot_quarantined(hub, 0, 4));
        EXPECT_EQ(live_entries(hub), 1u);
        EXPECT_FALSE(register_one(hub, key_of(4, 3))) << "permanently unusable";
        EXPECT_TRUE(register_one(hub, key_of(9, 2))) << "unaffected slot works";
    }
    {  // U13: a new generation on a quarantined slot is still refused.
        auto hub = hub_t{};
        ASSERT_TRUE(register_one(hub, key_of(4, 1, /*generation=*/0)));
        hub.quarantine_slot(0, 4);
        EXPECT_FALSE(register_one(hub, key_of(4, 1, /*generation=*/1)));
    }
    {  // The ledger drives the finalize skip: leaked ids excluded, others not.
        auto hub = hub_t{};
        EXPECT_FALSE(hub.is_ledgered(11)) << "empty ledger excludes nothing";
        EXPECT_FALSE(hub.is_ledgered(22));
        ASSERT_TRUE(register_one(hub, key_of(4, 1), /*corr_id=*/11));
        ASSERT_TRUE(register_one(hub, key_of(4, 2), /*corr_id=*/22));
        ASSERT_TRUE(hub.record_kernel_end(key_of(4, 1), 1, true).has_value());
        ASSERT_EQ(hub.quarantine_slot(0, 4).size(), 1u);
        EXPECT_FALSE(hub.is_ledgered(11)) << "completed -> retires normally";
        EXPECT_TRUE(hub.is_ledgered(22)) << "leaked -> finalize skips";
        auto dangling = std::vector<uint64_t>{11, 22, 33};
        auto retired  = std::vector<uint64_t>{};
        for(auto id : dangling)
        {
            if(hub.is_ledgered(id)) continue;
            retired.emplace_back(id);
        }
        EXPECT_EQ(retired, (std::vector<uint64_t>{11, 33}));
    }
    {  // U19: after abandon_in_child every entry point short-circuits.
        auto hub = hub_t{};
        ASSERT_TRUE(register_one(hub, key_of(4, 1)));
        hub.abandon_in_child();
        EXPECT_FALSE(register_one(hub, key_of(4, 2)));
        EXPECT_FALSE(hub.record_kernel_start(key_of(4, 1), 1));
        EXPECT_FALSE(hub.record_kernel_end(key_of(4, 1), 1, true).has_value());
        EXPECT_FALSE(hub.is_ledgered(1));
        EXPECT_TRUE(hub.quarantine_slot(0, 4).empty());
        EXPECT_TRUE(hub.drain_for_teardown().first.empty());
    }
}

// Close / drain lifecycle: closing gate, teardown drain, stranding policy.
TEST(DispatchHub, close_and_drain_lifecycle)
{
    {  // U18: a stopping session still completes an in-flight EOP, not leaks it.
        auto hub = hub_t{};
        auto key = key_of(4, 77);
        ASSERT_TRUE(register_one(hub, key, /*corr_id=*/99));
        hub.set_mode(session_mode::stopping);
        EXPECT_TRUE(hub.record_kernel_start(key, 1000));
        auto p = hub.record_kernel_end(key, 2000, /*drain_loss_free=*/true);
        ASSERT_TRUE(p.has_value());
        ASSERT_TRUE(p->start_ticks.has_value());
        EXPECT_EQ(*p->start_ticks, 1000u);
        EXPECT_EQ(p->end_ticks, 2000u);
        EXPECT_FALSE(hub.is_ledgered(99)) << "completed, not on loss ledger";
    }
    {  // Teardown leaks pending entries; registration is still admitted after
       // eligibility refuses, and a later drain leaks it too.
        auto hub = hub_t{};
        ASSERT_TRUE(register_one(hub, key_of(4, 1), /*corr_id=*/11));
        ASSERT_TRUE(register_one(hub, key_of(4, 2), /*corr_id=*/11));
        auto [lost, stats] = hub.drain_for_teardown();
        EXPECT_EQ(stats.dispatches, 2u);
        EXPECT_EQ(stats.correlation_ids, 1u);
        EXPECT_EQ(live_entries(hub), 0u);
        EXPECT_EQ(hub.mode(), session_mode::stopping);
        EXPECT_TRUE(hub.is_ledgered(11));
        EXPECT_FALSE(hub.can_register_batch({key_of(4, 3)}));
        EXPECT_TRUE(register_one(hub, key_of(4, 3), /*corr_id=*/12));
        EXPECT_EQ(hub.drain_for_teardown().second.dispatches, 1u);
        EXPECT_TRUE(hub.is_ledgered(12));
    }
    {  // U5: destroy strands, quarantines, and makes a reused doorbell signal-only.
        auto reg = OwnerRegistry{};
        auto hub = hub_t{};
        reg.add_queue(/*token=*/1, /*gpu=*/0, uint32_t{40});
        ASSERT_TRUE(register_one(hub, key_of(40, 1), /*corr_id=*/500));
        ASSERT_TRUE(register_one(hub, key_of(40, 2), /*corr_id=*/500));
        hub.mark_slot_closing(0, 40);
        EXPECT_TRUE(hub.is_closing(0, 40));
        auto stranded = hub.quarantine_slot(0, 40);
        EXPECT_EQ(stranded.size(), 2u);
        EXPECT_TRUE(slot_quarantined(hub, 0, 40));
        EXPECT_EQ(live_entries(hub), 0u);
        EXPECT_TRUE(hub.is_ledgered(500)) << "deliberately not retired";
        reg.remove_queue(1);
        reg.add_queue(/*token=*/2, 0, uint32_t{40});
        EXPECT_TRUE(reg.slot_uniquely_owned(0, 40)) << "ownership looks clean";
        EXPECT_FALSE(hub.can_register_batch({key_of(40, 1)}));
        EXPECT_FALSE(register_one(hub, key_of(40, 7), 600));
        EXPECT_FALSE(hub.record_kernel_end(key_of(40, 1), 999, /*loss_free=*/true).has_value());
        EXPECT_FALSE(hub.record_kernel_end(key_of(40, 2), 999, true).has_value());
    }
    {  // Closing is eligibility-only: a batch past it still registers; quarantine refuses.
        auto hub = hub_t{};
        hub.mark_slot_closing(0, 40);
        EXPECT_TRUE(hub.is_closing(0, 40));
        EXPECT_TRUE(hub.can_register_batch({key_of(40, 1)}));
        ASSERT_TRUE(register_one(hub, key_of(40, 1)));
        EXPECT_TRUE(hub.record_kernel_end(key_of(40, 1), 5, true).has_value());
        hub.quarantine_slot(0, 40);
        EXPECT_FALSE(hub.can_register_batch({key_of(40, 2)}));
        EXPECT_FALSE(register_one(hub, key_of(40, 2)));
    }
    {  // A clean destroy strands nothing and stays silent.
        auto hub = hub_t{};
        hub.mark_slot_closing(0, 40);
        auto stranded = hub.quarantine_slot(0, 40);
        EXPECT_TRUE(stranded.empty());
        EXPECT_EQ(live_entries(hub), 0u);
        EXPECT_EQ(hub.mode(), session_mode::running);
    }
    {  // pending_for_slot counts only that slot's live entries; proven leave.
        auto hub = hub_t{};
        EXPECT_EQ(hub.pending_for_slot(0, 40), 0u);
        ASSERT_TRUE(register_one(hub, key_of(40, 1)));
        ASSERT_TRUE(register_one(hub, key_of(40, 2)));
        ASSERT_TRUE(register_one(hub, key_of(41, 1)));
        EXPECT_EQ(hub.pending_for_slot(0, 40), 2u);
        EXPECT_EQ(hub.pending_for_slot(0, 41), 1u);
        ASSERT_TRUE(hub.record_kernel_end(key_of(40, 1), 5, true).has_value());
        EXPECT_EQ(hub.pending_for_slot(0, 40), 1u);
        ASSERT_TRUE(hub.record_kernel_end(key_of(40, 2), 5, true).has_value());
        EXPECT_EQ(hub.pending_for_slot(0, 40), 0u);
        EXPECT_EQ(hub.pending_for_slot(0, 41), 1u);
    }
    {  // An incomplete drain strands exactly the remainder and tombstones them.
        auto hub = hub_t{};
        ASSERT_TRUE(register_one(hub, key_of(40, 1), /*corr_id=*/901));
        ASSERT_TRUE(register_one(hub, key_of(40, 2), /*corr_id=*/902));
        ASSERT_TRUE(register_one(hub, key_of(40, 3), /*corr_id=*/903));
        hub.mark_slot_closing(0, 40);
        EXPECT_TRUE(hub.record_kernel_end(key_of(40, 2), 5, true).has_value());
        EXPECT_EQ(hub.pending_for_slot(0, 40), 2u);
        auto stranded = hub.quarantine_slot(0, 40);
        EXPECT_EQ(stranded.size(), 2u);
        EXPECT_FALSE(hub.is_ledgered(902)) << "paired, retired normally";
        EXPECT_TRUE(hub.is_ledgered(901));
        EXPECT_TRUE(hub.is_ledgered(903));
        EXPECT_FALSE(hub.can_register_batch({key_of(40, 1)}));
        EXPECT_FALSE(hub.can_register_batch({key_of(40, 3)}));
    }
    {  // Records pairable only after HW completion still complete and strand nothing.
        auto hub = hub_t{};
        ASSERT_TRUE(register_one(hub, key_of(40, 1), /*corr_id=*/900));
        ASSERT_TRUE(register_one(hub, key_of(40, 2), /*corr_id=*/900));
        hub.mark_slot_closing(0, 40);
        EXPECT_EQ(hub.pending_for_slot(0, 40), 2u);
        ASSERT_TRUE(hub.record_kernel_end(key_of(40, 1), 500, true).has_value());
        ASSERT_TRUE(hub.record_kernel_end(key_of(40, 2), 600, true).has_value());
        EXPECT_EQ(hub.pending_for_slot(0, 40), 0u);
        auto stranded = hub.quarantine_slot(0, 40);
        EXPECT_TRUE(stranded.empty());
        EXPECT_FALSE(hub.is_ledgered(900));
    }
}

// Destroy runs concurrently with enqueue registration on other slots; run under
// TSan for the ordering. The hub must stay consistent and never deadlock.
TEST(DispatchHub, concurrent_destroy_and_registration_stay_consistent)
{
    constexpr uint32_t kSlots = 64;

    auto hub = hub_t{};
    for(uint32_t s = 0; s < kSlots; ++s)
    {
        ASSERT_TRUE(register_one(hub, key_of(s, 1), /*corr_id=*/s));
    }

    auto destroyer = std::thread{[&hub]() {
        for(uint32_t s = 0; s < kSlots; ++s)
        {
            hub.mark_slot_closing(0, s);
            hub.quarantine_slot(0, s);
        }
    }};

    auto prover = std::atomic<uint32_t>{0};
    auto reader = std::thread{[&hub, &prover]() {
        for(uint32_t s = 0; s < kSlots; ++s)
            if(hub.record_kernel_end(key_of(s, 1), 9, true).has_value()) ++prover;
    }};

    destroyer.join();
    reader.join();

    // Every slot ended quarantined, nothing pending, each entry went one way only.
    for(uint32_t s = 0; s < kSlots; ++s)
    {
        EXPECT_TRUE(slot_quarantined(hub, 0, s));
    }
    EXPECT_EQ(live_entries(hub), 0u);
    EXPECT_LE(prover.load(), kSlots);
    EXPECT_EQ(tracked_payload::live.load(), 0);
}

// The drain observes progress from another thread -- how the real wait terminates
// early: the reader pairs records while the closing thread polls.
TEST(DispatchHub, pending_for_slot_observes_concurrent_pairing)
{
    constexpr uint32_t kCount = 128;

    auto hub = hub_t{};
    for(uint32_t i = 0; i < kCount; ++i)
    {
        ASSERT_TRUE(register_one(hub, key_of(40, i)));
    }
    EXPECT_EQ(hub.pending_for_slot(0, 40), kCount);

    auto reader = std::thread{[&hub]() {
        for(uint32_t i = 0; i < kCount; ++i)
            hub.record_kernel_end(key_of(40, i), 7, true);
    }};

    // The closing thread polls exactly as drain_close_signal_less_queue does.
    size_t pending = kCount;
    for(int spin = 0; spin < 100000 && pending > 0; ++spin)
    {
        pending = hub.pending_for_slot(0, 40);
    }
    reader.join();

    EXPECT_EQ(hub.pending_for_slot(0, 40), 0u);
    auto stranded = hub.quarantine_slot(0, 40);
    EXPECT_TRUE(stranded.empty()) << "a drain that observed completion must strand nothing";
}

// Unit 4: live doorbell-owner registry (requirement 3). Injectivity and counts.
TEST(OwnerRegistry, injectivity_and_quarantine)
{
    {  // An unknown slot is not injective: an unresolved doorbell is never unique.
        auto reg = OwnerRegistry{};
        EXPECT_FALSE(reg.slot_uniquely_owned(/*gpu_id=*/0, /*slot=*/4100));
    }
    {  // A sole owner is unique.
        auto reg = OwnerRegistry{};
        EXPECT_EQ(reg.add_queue(/*token=*/1, /*gpu=*/0, /*slot=*/uint32_t{40}),
                  OwnerRegistry::add_result::sole_owner);
        EXPECT_TRUE(reg.slot_uniquely_owned(0, 40));
        EXPECT_EQ(reg.owners_of(0, 40), 1u);
        EXPECT_EQ(reg.live_queues(), 1u);
    }
    {  // A second live owner collides; the caller quarantines the slot in the hub.
        auto reg = OwnerRegistry{};
        auto hub = hub_t{};
        ASSERT_TRUE(register_one(hub, key_of(40, 1)));
        ASSERT_TRUE(register_one(hub, key_of(40, 2)));
        EXPECT_EQ(reg.add_queue(1, 0, uint32_t{40}), OwnerRegistry::add_result::sole_owner);
        EXPECT_EQ(reg.add_queue(2, 0, uint32_t{40}), OwnerRegistry::add_result::collision);
        EXPECT_FALSE(reg.slot_uniquely_owned(0, 40));
        EXPECT_EQ(reg.owners_of(0, 40), 2u);
        auto stranded = hub.quarantine_slot(0, 40);
        EXPECT_EQ(stranded.size(), 2u);
        EXPECT_TRUE(slot_quarantined(hub, 0, 40));
        EXPECT_FALSE(hub.can_register_batch({key_of(40, 3)}));
    }
    {  // Quarantine outlives the collision: the slot stays refused even after a co-owner dies.
        auto reg = OwnerRegistry{};
        auto hub = hub_t{};
        reg.add_queue(1, 0, uint32_t{40});
        reg.add_queue(2, 0, uint32_t{40});
        hub.quarantine_slot(0, 40);
        reg.remove_queue(2);
        EXPECT_TRUE(reg.slot_uniquely_owned(0, 40)) << "ownership looks clean again";
        EXPECT_TRUE(slot_quarantined(hub, 0, 40)) << "but slot stays unusable";
        EXPECT_FALSE(hub.can_register_batch({key_of(40, 9)}));
    }
    {  // A pre-session queue participates in injectivity, so a later queue collides.
        auto reg = OwnerRegistry{};
        EXPECT_EQ(reg.add_queue(/*pre-session*/ 1, 0, uint32_t{7}),
                  OwnerRegistry::add_result::sole_owner);
        EXPECT_TRUE(reg.slot_uniquely_owned(0, 7));
        EXPECT_EQ(reg.add_queue(/*post-session*/ 2, 0, uint32_t{7}),
                  OwnerRegistry::add_result::collision);
        EXPECT_FALSE(reg.slot_uniquely_owned(0, 7));
    }
}

// Unit 4/5: registry resolution, GPU scoping, and exact reference counting.
TEST(OwnerRegistry, resolution_scope_and_counts)
{
    {  // An unresolved queue disables its whole GPU until it dies; others unaffected.
        auto reg = OwnerRegistry{};
        reg.add_queue(1, 0, uint32_t{40});
        reg.add_queue(2, 1, uint32_t{50});  // a different GPU
        EXPECT_TRUE(reg.slot_uniquely_owned(0, 40));
        EXPECT_EQ(reg.add_queue(3, 0, std::nullopt), OwnerRegistry::add_result::slot_unknown);
        EXPECT_EQ(reg.unresolved_queues(0), 1u);
        EXPECT_FALSE(reg.slot_uniquely_owned(0, 40)) << "this GPU is out";
        EXPECT_FALSE(reg.slot_uniquely_owned(0, 99));
        EXPECT_TRUE(reg.slot_uniquely_owned(1, 50)) << "other GPU unaffected";
        reg.remove_queue(3);
        EXPECT_EQ(reg.unresolved_queues(0), 0u);
        EXPECT_TRUE(reg.slot_uniquely_owned(0, 40));
    }
    {  // Slots are scoped per GPU: the same slot on two GPUs is not a collision.
        auto reg = OwnerRegistry{};
        EXPECT_EQ(reg.add_queue(1, 0, uint32_t{40}), OwnerRegistry::add_result::sole_owner);
        EXPECT_EQ(reg.add_queue(2, 1, uint32_t{40}), OwnerRegistry::add_result::sole_owner);
        EXPECT_TRUE(reg.slot_uniquely_owned(0, 40));
        EXPECT_TRUE(reg.slot_uniquely_owned(1, 40));
    }
    {  // remove releases ownership; re-registering a token replaces, not doubles.
        auto reg = OwnerRegistry{};
        reg.add_queue(1, 0, uint32_t{40});
        reg.add_queue(1, 0, uint32_t{40});  // same token again
        EXPECT_EQ(reg.owners_of(0, 40), 1u);
        EXPECT_EQ(reg.live_queues(), 1u);
        reg.add_queue(1, 0, uint32_t{41});  // same token moves slots
        EXPECT_EQ(reg.owners_of(0, 40), 0u);
        EXPECT_EQ(reg.owners_of(0, 41), 1u);
        reg.remove_queue(1);
        EXPECT_EQ(reg.owners_of(0, 41), 0u);
        EXPECT_EQ(reg.live_queues(), 0u);
        reg.remove_queue(1);  // idempotent
        EXPECT_EQ(reg.live_queues(), 0u);
    }
}

// Unit 7: fork epoch / child abandonment (requirement 8, U19)

namespace
{
// Inherited process-wide state: function-local statics, so a forked child
// inherits them AND runs their destructors at a normal exit().
hub_t&
forked_hub()
{
    static auto _v = hub_t{};
    return _v;
}

OwnerRegistry&
forked_registry()
{
    static auto _v = OwnerRegistry{};
    return _v;
}

// Everything a child must call without touching an inherited mutex; true when
// every entry point reported "disabled/empty".
bool
all_entry_points_short_circuit()
{
    bool ok = true;

    ok = ok && !register_one(forked_hub(), key_of(40, 99));
    ok = ok && !forked_hub().record_kernel_end(key_of(40, 1), 1, true).has_value();
    ok = ok && !forked_hub().record_kernel_start(key_of(40, 1), 1);
    ok = ok && forked_hub().pending_for_slot(0, 40) == 0;
    ok = ok && !forked_hub().is_closing(0, 40);
    ok = ok && !forked_hub().is_ledgered(500);
    ok = ok && forked_hub().mode() == session_mode::child_stale;
    ok = ok && forked_hub().quarantine_slot(0, 40).empty();
    ok = ok && forked_hub().drain_for_teardown().first.empty();

    ok = ok && !forked_registry().slot_uniquely_owned(0, 40);
    ok = ok && !forked_registry().slot_of(1).has_value();
    ok = ok && forked_registry().owners_of(0, 40) == 0;
    ok = ok && forked_registry().unresolved_queues(0) == 0;

    return ok;
}
}  // namespace

// U19: a REAL fork. The child abandons inherited state as the atfork handler
// does, exercises the entry points, and exits normally so the inherited statics'
// destructors run. It must not hang, crash, or double-free; the parent is intact.
TEST(fork_safety, forked_child_short_circuits_and_survives_normal_exit)
{
    auto parent_only = hub_t{};
    ASSERT_TRUE(register_one(parent_only, key_of(9, 1)));

    pid_t pid = fork();
    ASSERT_NE(pid, -1);

    if(pid == 0)
    {
        // CHILD. Exactly the atfork handler's work: atomic stores only.
        forked_hub().abandon_in_child();
        forked_registry().abandon_in_child();

        const bool ok = all_entry_points_short_circuit();

        // Normal exit(): runs static destructors over the abandoned state.
        std::exit(ok ? 0 : 2);
    }

    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status)) << "child did not exit normally";
    if(WIFEXITED(status))
    {
        EXPECT_EQ(WEXITSTATUS(status), 0) << "child entry points did not short-circuit";
    }

    EXPECT_EQ(live_entries(parent_only), 1u);
    EXPECT_TRUE(parent_only.record_kernel_end(key_of(9, 1), 5, true).has_value());
}

// The child must survive a fork taken while another thread hammers the shared
// objects -- the case where an inherited mutex can be left permanently locked.
TEST(fork_safety, child_survives_a_fork_taken_under_contention)
{
    auto hub  = hub_t{};
    auto stop = std::atomic<bool>{false};
    auto busy = std::thread{[&hub, &stop]() {
        for(uint32_t i = 0; !stop.load(); ++i)
        {
            register_one(hub, key_of(1, i % 512));
            hub.record_kernel_end(key_of(1, i % 512), 1, true);
            live_entries(hub);
        }
    }};

    for(int i = 0; i < 8; ++i)
    {
        pid_t pid = fork();
        ASSERT_NE(pid, -1);
        if(pid == 0)
        {
            // Abandoning makes every entry point avoid the possibly-locked mutex.
            hub.abandon_in_child();
            const bool ok = live_entries(hub) == 0 && !register_one(hub, key_of(1, 7)) &&
                            !hub.record_kernel_end(key_of(1, 7), 1, true).has_value();
            // _exit, not exit: this case is about not deadlocking on an inherited
            // locked mutex; the busy thread's allocations are unreachable in the
            // child, so a normal-exit leak check would flag a fork artifact.
            _exit(ok ? 0 : 2);
        }
        int status = 0;
        ASSERT_EQ(waitpid(pid, &status, 0), pid);
        EXPECT_TRUE(WIFEXITED(status));
        if(WIFEXITED(status))
        {
            EXPECT_EQ(WEXITSTATUS(status), 0);
        }
    }

    stop.store(true);
    busy.join();
}

// Unit 8: stateful model / property test. Drives the hub + retry owner +
// finalizer through a SEEDED random event sequence against a reference model,
// asserting the completion invariants after every event.

namespace
{
enum class model_state
{
    absent,
    pending,
    proven,  // ownership handed out; can never become leaked
    leaked,
};

// The reference model, tracked independently of the hub so the two can be compared.
struct reference_model
{
    std::map<std::pair<uint32_t, uint32_t>, model_state> state;    // (slot,id) -> state
    std::map<std::pair<uint32_t, uint32_t>, uint64_t>    corr_of;  // -> correlation id

    std::set<uint32_t>                           quarantined;
    std::set<uint64_t>                           ledger;
    std::map<std::pair<uint32_t, uint32_t>, int> emitted;
    std::map<std::pair<uint32_t, uint32_t>, int> retired;

    using key_t = std::pair<uint32_t, uint32_t>;

    bool stopping = false;

    // Only a still-pending key is inadmissible; mode and permanent quarantine are
    // what stop a leaked key ever being reserved again.
    bool admissible(const key_t& k) const
    {
        return !stopping && at(k) != model_state::pending && quarantined.count(k.first) == 0;
    }

    model_state at(const key_t& k) const
    {
        auto it = state.find(k);
        return (it == state.end()) ? model_state::absent : it->second;
    }
};

correlation_key
to_key(const reference_model::key_t& k)
{
    return key_of(k.first, k.second);
}
}  // namespace

TEST(DispatchHub, stateful_model_matches_the_reference_across_random_events)
{
    constexpr int      kEvents  = 4000;
    constexpr uint32_t kSlots   = 6;
    constexpr uint32_t kPerSlot = 8;

    const int payloads_before = tracked_payload::live.load();

    auto hub   = hub_t{};
    auto model = reference_model{};
    auto rng   = std::mt19937{20260803};

    // Proven completions waiting to be finalized, mirroring the task group.
    auto in_flight = std::vector<hub_t::proven>{};
    // Completions the task group refused, awaiting the teardown-thread drain.
    auto deferred = std::vector<hub_t::proven>{};

    auto random_key = [&rng]() { return reference_model::key_t{rng() % kSlots, rng() % kPerSlot}; };

    // Finalize one proven completion as production does: retire exactly once, emit
    // only when timing was usable.
    auto finalize = [&model](hub_t::proven&& p, bool convert_ok) {
        auto mk      = reference_model::key_t{p.key.doorbell_off, p.key.dispatch_idx_low32};
        int  emits   = 0;
        int  retires = 0;
        auto outcome = run_complete_signal_less_dispatch(
            p.start_ticks,
            p.end_ticks,
            /*enqueue_ts=*/0,
            /*now_ns=*/1'000'000,
            [convert_ok](uint64_t t, uint64_t* out) {
                if(!convert_ok) return false;
                *out = t;
                return true;
            },
            [&emits](uint64_t, uint64_t) { ++emits; },
            [&retires]() { ++retires; });

        EXPECT_EQ(retires, 1);
        EXPECT_LE(emits, 1);
        EXPECT_EQ(emits, outcome == finalize_outcome::result_ready ? 1 : 0);

        model.emitted[mk] += emits;
        model.retired[mk] += retires;
    };

    for(int ev = 0; ev < kEvents; ++ev)
    {
        switch(rng() % 10)
        {
            case 0:  // RegisterBatch
            {
                auto batch = std::vector<hub_t::registration>{};
                auto keys  = std::vector<reference_model::key_t>{};
                auto corr  = uint64_t{100 + (rng() % 7)};
                for(uint32_t i = 0, n = 1 + (rng() % 3); i < n; ++i)
                {
                    auto k = random_key();
                    if(std::find(keys.begin(), keys.end(), k) != keys.end()) continue;
                    keys.emplace_back(k);
                    batch.emplace_back(reg_of(to_key(k), corr));
                }
                if(keys.empty()) break;

                bool expect_ok = true;
                for(const auto& k : keys)
                    expect_ok = expect_ok && model.admissible(k);

                // Production asks eligibility first; a batch it accepted must register.
                auto flat = std::vector<correlation_key>{};
                for(const auto& k : keys)
                    flat.emplace_back(to_key(k));
                const bool eligible = hub.can_register_batch(flat);
                EXPECT_EQ(eligible, expect_ok) << "can_register_batch disagreed with the model";
                if(!eligible) break;

                const bool got = hub.register_batch(std::move(batch));
                EXPECT_TRUE(got) << "an eligible batch failed to register";
                if(got)
                {
                    for(const auto& k : keys)
                    {
                        model.state[k]   = model_state::pending;
                        model.corr_of[k] = corr;
                    }
                }
                break;
            }
            case 1:  // START
            {
                auto k = random_key();
                hub.record_kernel_start(to_key(k), 10 + (rng() % 50));
                break;
            }
            case 2:
            case 3:  // EOP (sometimes under a lossy drain)
            {
                auto       k             = random_key();
                const bool loss_free     = (rng() % 4) != 0;
                const bool expect_proven = loss_free && model.at(k) == model_state::pending;

                auto got = hub.record_kernel_end(to_key(k), 900, loss_free);
                EXPECT_EQ(got.has_value(), expect_proven)
                    << "record_kernel_end disagreed with the model";
                if(got)
                {
                    model.state[k] = model_state::proven;
                    in_flight.emplace_back(std::move(*got));
                }
                break;
            }
            case 4:  // SubmitTask / RejectTask / RunTask
            {
                if(in_flight.empty()) break;
                auto p = std::move(in_flight.back());
                in_flight.pop_back();

                const auto disposition = rng() % 3;
                if(disposition == 0)
                {
                    finalize(std::move(p), /*convert_ok=*/true);  // accepted + ran
                }
                else if(disposition == 1)
                {
                    finalize(std::move(p), /*convert_ok=*/false);  // ConvertFail
                }
                else
                {
                    deferred.emplace_back(std::move(p));  // deferred to teardown thread
                }
                break;
            }
            case 5:  // teardown thread drains the deferred completions
            {
                for(auto& q : deferred)
                    finalize(std::move(q), true);
                deferred.clear();
                break;
            }
            case 6:  // Collision -> quarantine a slot
            {
                const uint32_t slot = rng() % kSlots;
                auto           lost = hub.quarantine_slot(0, slot);
                model.quarantined.insert(slot);
                for(auto& kv : model.state)
                {
                    if(kv.first.first == slot && kv.second == model_state::pending)
                    {
                        kv.second = model_state::leaked;
                        model.ledger.insert(model.corr_of[kv.first]);
                    }
                }
                for(auto& l : lost)
                    EXPECT_EQ(l.key.doorbell_off, slot);
                break;
            }
            case 8:  // Teardown drain (bulk loss) -- rare, and terminal
            {
                if((rng() % 12) != 0) break;
                hub.drain_for_teardown();
                model.stopping = true;
                for(auto& kv : model.state)
                {
                    if(kv.second == model_state::pending)
                    {
                        kv.second = model_state::leaked;
                        model.ledger.insert(model.corr_of[kv.first]);
                    }
                }
                break;
            }
            default: break;
        }

        // --- invariants, after EVERY event ---------------------------------

        for(const auto& kv : model.state)
        {
            const auto& k = kv.first;

            EXPECT_LE(model.emitted[k], 1) << "more than one record for one dispatch";
            EXPECT_LE(model.retired[k], 1) << "more than one cleanup for one dispatch";
            if(model.emitted[k] > 0)
            {
                EXPECT_EQ(kv.second, model_state::proven) << "record emitted without a proven EOP";
            }

            // Retire iff proven; a leaked entry never retires and stays unmatchable.
            if(kv.second == model_state::leaked)
            {
                EXPECT_EQ(model.retired[k], 0) << "a leaked dispatch was retired";
                EXPECT_EQ(model.emitted[k], 0) << "a leaked dispatch emitted a record";
                EXPECT_FALSE(hub.record_kernel_end(to_key(k), 1, true).has_value())
                    << "a leaked entry was still matchable";
                EXPECT_FALSE(hub.can_register_batch({to_key(k)}))
                    << "a leaked key was re-registerable";
            }
        }

        // The loss ledger is exactly the leaked correlation ids.
        for(auto id : model.ledger)
        {
            EXPECT_TRUE(hub.is_ledgered(id)) << "leaked correlation id missing from the ledger";
        }

        // A quarantined slot never accepts a reservation again.
        for(auto slot : model.quarantined)
        {
            EXPECT_TRUE(slot_quarantined(hub, 0, slot));
        }
    }

    // Teardown: everything still pending leaks, nothing proven is lost.
    for(auto& q : deferred)
        finalize(std::move(q), true);
    deferred.clear();
    for(auto& p : in_flight)
        finalize(std::move(p), true);
    in_flight.clear();
    hub.drain_for_teardown();

    EXPECT_EQ(tracked_payload::live.load(), payloads_before);
}
