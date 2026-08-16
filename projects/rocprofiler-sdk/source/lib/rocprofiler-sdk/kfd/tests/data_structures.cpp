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

#include "lib/rocprofiler-sdk/kfd/correlation_types.hpp"
#include "lib/rocprofiler-sdk/kfd/doorbell_map.hpp"
#include "lib/rocprofiler-sdk/kfd/kfd_correlation.hpp"

#include <gtest/gtest.h>

namespace
{
using namespace rocprofiler::kfd;
rocprofiler_queue_id_t
qid(uint64_t h)
{
    return rocprofiler_queue_id_t{h};
}
}  // namespace

TEST(correlation_key, equality_hash_and_per_gpu_scoping)
{
    auto a = correlation_key{7, 100, 0};
    auto c = correlation_key{7, 100, 1};  // different generation
    EXPECT_EQ(a, (correlation_key{7, 100, 0}));
    EXPECT_NE(a, c);
    EXPECT_EQ(correlation_key_hash{}(a), correlation_key_hash{}(correlation_key{7, 100, 0}));
    // GPU id scopes the key; three-field construction still means GPU 0.
    auto on_gpu0 = correlation_key{7, 100, 0, /*gpu_id=*/0};
    auto on_gpu1 = correlation_key{7, 100, 0, /*gpu_id=*/1};
    EXPECT_NE(on_gpu0, on_gpu1) << "identical slot/index/generation on two GPUs must not be equal";
    EXPECT_EQ(on_gpu0, (correlation_key{7, 100, 0, 0}));
    EXPECT_EQ((correlation_key{7, 100, 0}), on_gpu0);
}
// KFD-result-vs-HSA-fallback decision in get_dispatch_time; a converted end may
// land within kKfdFutureSlackNs past now, beyond that is rejected.
TEST(kfd_time_is_sane, dispatch_window_and_conversion_slack)
{
    constexpr uint64_t now = 1'000'000'000;
    struct row
    {
        uint64_t    start, end, enqueue, now;
        bool        expect;
        const char* label;
    };
    const row rows[] = {
        {150, 250, 100, 300, true, "inside window"},
        {100, 300, 100, 300, true, "exactly on both bounds"},
        {now - 5'000'000, now + 2'700'000, 0, now, true, "observed skew past now"},
        {0, now + kKfdFutureSlackNs, 0, now, true, "exactly the slack bound"},
        {0, now + kKfdFutureSlackNs + 1, 0, now, false, "one past slack bound"},
        {0, now + 5'000'000'000, 0, now, false, "5s out"},
        {99, 250, 100, 300, false, "starts before enqueue"},
        {150, 301 + kKfdFutureSlackNs, 100, 300, false, "ends beyond now + slack"},
        {250, 150, 100, 300, false, "inverted interval"},
        {150, 150, 100, 300, false, "zero-length interval"},
    };
    for(const auto& tc : rows)
        EXPECT_EQ(kfd_time_is_sane(tc.start, tc.end, tc.enqueue, tc.now), tc.expect) << tc.label;
}
// Generation bumping on reuse/destroy, degenerate sharing, and page-relative slot helpers.
TEST(DoorbellMap, generation_bind_resolve_and_slot_helpers)
{
    {  // unknown destroy is a no-op; reuse then bumps gen so old records never re-attribute
        auto m = DoorbellMap{};
        m.on_queue_destroyed(qid(123));
        EXPECT_EQ(m.get_generation(0, 7), 0u) << "unknown destroy noop";
        m.bind_and_resolve(0, qid(42), 7);
        m.on_queue_destroyed(qid(42));
        EXPECT_EQ(m.get_generation(0, 7), 1u) << "destroy bumps generation";
        auto e = m.bind_and_resolve(0, qid(43), 7);
        EXPECT_EQ(e.doorbell_off, 7u) << "reuse doorbell_off";
        EXPECT_EQ(e.generation, 1u) << "reuse carries bumped gen";
    }
    {  // two queues share a doorbell without a destroy: generation unchanged
        auto m = DoorbellMap{};
        EXPECT_EQ(m.bind_and_resolve(0, qid(42), 7).doorbell_off, 7u) << "same doorbell a";
        EXPECT_EQ(m.bind_and_resolve(0, qid(43), 7).doorbell_off, 7u) << "same doorbell b";
        EXPECT_EQ(m.get_generation(0, 7), 0u) << "no destroy -> gen unchanged";
    }
    {  // capture (pointer) and reader (record) reduce to the same page slot
        constexpr uint64_t kPage = 4096;
        EXPECT_EQ(doorbell_off_to_page_slot(4100u), 4u) << "reader slot 4";
        EXPECT_EQ(doorbell_off_to_page_slot(4104u), 8u) << "reader slot 8";
        EXPECT_EQ(doorbell_ptr_to_page_slot(0x7f0000004010ull, kPage), 4u) << "capture slot 4";
        EXPECT_EQ(doorbell_ptr_to_page_slot(0x7f0000004020ull, kPage), 8u) << "capture slot 8";
        EXPECT_EQ(doorbell_off_to_page_slot(4100u),
                  doorbell_ptr_to_page_slot(0x7f0000004010ull, kPage))
            << "both sides agree";
    }
    {  // first call binds, later calls cache without changing state
        auto m  = DoorbellMap{};
        auto e1 = m.bind_and_resolve(0, qid(42), 4u);
        EXPECT_EQ(e1.doorbell_off, 4u) << "bind off";
        EXPECT_EQ(e1.generation, 0u) << "bind gen";
        auto e2 = m.bind_and_resolve(0, qid(42), 4u);
        EXPECT_EQ(e2.doorbell_off, 4u) << "cache off";
        EXPECT_EQ(e2.generation, 0u) << "cache gen";
    }
    {  // after reuse, slow (write-lock) then fast (read-lock) resolves report bumped gen 1
        auto m = DoorbellMap{};
        m.bind_and_resolve(0, qid(1), 4u);
        m.on_queue_destroyed(qid(1));
        auto e = m.bind_and_resolve(0, qid(2), 4u);
        EXPECT_EQ(e.doorbell_off, 4u) << "rebind off";
        EXPECT_EQ(e.generation, 1u) << "rebind slow-path gen";
        auto f = m.bind_and_resolve(0, qid(2), 4u);
        EXPECT_EQ(f.doorbell_off, 4u) << "fast off";
        EXPECT_EQ(f.generation, 1u) << "fast-path gen not stale";
    }
    {  // migration is an upsert: same queue -> new doorbell resolves to the new slot
        auto m = DoorbellMap{};
        EXPECT_EQ(m.bind_and_resolve(0, qid(7), 4u).doorbell_off, 4u) << "migrate before";
        EXPECT_EQ(m.bind_and_resolve(0, qid(7), 8u).doorbell_off, 8u) << "migrate after";
    }
    {  // slot numbers repeat across GPUs, so generations are per-GPU
        auto m       = DoorbellMap{};
        auto on_gpu0 = m.bind_and_resolve(/*gpu_id=*/0, qid(1), /*doorbell_off=*/7);
        auto on_gpu1 = m.bind_and_resolve(/*gpu_id=*/1, qid(2), /*doorbell_off=*/7);
        EXPECT_EQ(on_gpu0.generation, 0u) << "gpu0 gen";
        EXPECT_EQ(on_gpu1.generation, 0u) << "gpu1 gen";
        EXPECT_EQ(on_gpu0.gpu_id, 0u) << "gpu0 id";
        EXPECT_EQ(on_gpu1.gpu_id, 1u) << "gpu1 id";
        m.on_queue_destroyed(qid(1));
        EXPECT_EQ(m.get_generation(0, 7), 1u) << "gpu0 slot bumped";
        EXPECT_EQ(m.get_generation(1, 7), 0u) << "another GPU's slot must be untouched";
        EXPECT_EQ(m.bind_and_resolve(1, qid(2), 7).generation, 0u) << "gpu1 keeps own gen";
        EXPECT_EQ(m.bind_and_resolve(0, qid(3), 7).generation, 1u) << "gpu0 reuse bumped";
    }
}
