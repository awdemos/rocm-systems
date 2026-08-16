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

// Unit tests for the no-signal finalizer. Every seam (tick converter S3, record
// emitter, retirement observer S7, executor S2) is injected, so each branch is
// deterministic.

#include "lib/rocprofiler-sdk/kfd/complete_signal_less_dispatch.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>

namespace
{
using namespace rocprofiler::kfd;

// S7: counts retires/emits so "exactly once" is checked directly, not via logs.
struct observer
{
    int      retires = 0;
    int      emits   = 0;
    uint64_t start   = 0;
    uint64_t end     = 0;

    auto retire_fn()
    {
        return [this]() { ++retires; };
    }
    auto emit_fn()
    {
        return [this](uint64_t s, uint64_t e) {
            ++emits;
            start = s;
            end   = e;
        };
    }
};

// S3: adds a fixed epoch so converted values are distinguishable from raw ticks;
// ok=false forces the conversion-failure branch (as a non-GPU agent would).
struct converter
{
    bool     ok    = true;
    uint64_t epoch = 1'000'000;

    auto fn()
    {
        return [this](uint64_t ticks, uint64_t* out) {
            if(!ok) return false;
            *out = ticks + epoch;
            return true;
        };
    }
};
}  // namespace

// start known + conversion + sanity OK -> one record with converted KFD
// timestamps, correlation id retired exactly once.
TEST(complete_signal_less_dispatch, result_ready_emits_once_and_retires_once)
{
    auto obs     = observer{};
    auto conv    = converter{};
    auto outcome = run_complete_signal_less_dispatch(std::optional<uint64_t>{500},
                                                     /*end_ticks=*/900,
                                                     /*enqueue_ts=*/0,
                                                     /*now_ns=*/10'000'000,
                                                     conv.fn(),
                                                     obs.emit_fn(),
                                                     obs.retire_fn());
    EXPECT_EQ(outcome, finalize_outcome::result_ready);
    EXPECT_EQ(obs.emits, 1);
    EXPECT_EQ(obs.retires, 1);
    EXPECT_EQ(obs.start, 500u + conv.epoch);
    EXPECT_EQ(obs.end, 900u + conv.epoch);
}

// Every no-timing shape (lost start, refused conversion, each sanity clause) must
// emit nothing yet still retire exactly once -- completion is proven, not leaked.
TEST(complete_signal_less_dispatch, no_timing_shapes_emit_nothing_but_retire_once)
{
    struct row
    {
        std::optional<uint64_t> start;
        uint64_t                end, enqueue, now;
        bool                    convert_ok;
        uint64_t                epoch;
        const char*             label;
    };
    const row rows[] = {
        {std::nullopt, 900, 0, 10'000'000, true, 1'000'000, "start lost (shape ii)"},
        {std::optional<uint64_t>{500}, 900, 0, 10'000'000, false, 1'000'000, "conversion refused"},
        {std::optional<uint64_t>{1},
         1000 + kKfdFutureSlackNs + 1,
         0,
         1000,
         true,
         0,
         "end beyond now + slack"},
        {std::optional<uint64_t>{500},
         900,
         9'000'000,
         10'000'000,
         true,
         1'000'000,
         "starts before enqueue"},
        {std::optional<uint64_t>{900},
         900,
         0,
         10'000'000,
         true,
         1'000'000,
         "non-positive interval"},
    };
    for(const auto& tc : rows)
    {
        auto obs     = observer{};
        auto conv    = converter{tc.convert_ok, tc.epoch};
        auto outcome = run_complete_signal_less_dispatch(
            tc.start, tc.end, tc.enqueue, tc.now, conv.fn(), obs.emit_fn(), obs.retire_fn());
        EXPECT_EQ(outcome, finalize_outcome::completed_no_timing) << tc.label;
        EXPECT_EQ(obs.emits, 0) << tc.label;
        EXPECT_EQ(obs.retires, 1) << tc.label;
    }
}

// AIPROFSDK-1036: a converted firmware end a few ms past `now` must be kept but
// clamped to `now` (retirement samples a host clock just after now, so an
// unclamped future end lands after retirement). The whole interval shifts down so
// the duration is preserved -- same as tracing::adjust_profiling_time.
TEST(complete_signal_less_dispatch, converted_end_past_now_is_clamped_to_now)
{
    constexpr uint64_t now      = 1'000'000'000;
    constexpr uint64_t skew     = 2'700'000;  // measured 2.0-2.7 ms
    constexpr uint64_t duration = 5'000'000;
    auto               obs      = observer{};
    auto               conv     = converter{true, /*epoch=*/0};  // passthrough -> end at now + skew
    auto outcome = run_complete_signal_less_dispatch(std::optional<uint64_t>{now + skew - duration},
                                                     now + skew,
                                                     /*enqueue_ts=*/0,
                                                     now,
                                                     conv.fn(),
                                                     obs.emit_fn(),
                                                     obs.retire_fn());
    EXPECT_EQ(outcome, finalize_outcome::result_ready);
    EXPECT_EQ(obs.emits, 1);
    EXPECT_EQ(obs.retires, 1);
    EXPECT_EQ(obs.end, now);  // clamped
    EXPECT_EQ(obs.end - obs.start, duration);
}

// AIPROFSDK-1036 ordering regression: retire() runs from the scope destructor,
// strictly after `now`, and samples a host clock (>= now); the emitted end must
// not exceed retired_ts within the 10us async-copy-tracing tolerance. Pre-clamp,
// a verbatim future end landed tens of us after retirement (+75898 ns on gfx1201).
TEST(complete_signal_less_dispatch, emitted_end_never_lands_after_retirement)
{
    constexpr uint64_t now         = 1'000'000'000;
    constexpr uint64_t tolerance   = 10'000;
    constexpr uint64_t future_skew = 75'898;  // the observed i93 offender skew
    auto               obs         = observer{};
    auto               conv        = converter{true, /*epoch=*/0};
    uint64_t           retired_ts  = 0;
    uint64_t           host_clock  = now;
    auto               retire_fn   = [&obs, &retired_ts, &host_clock]() {
        ++obs.retires;
        retired_ts = ++host_clock;  // any value >= now, mirroring a later sample
    };
    auto outcome = run_complete_signal_less_dispatch(std::optional<uint64_t>{now - 5'000'000},
                                                     now + future_skew,
                                                     /*enqueue_ts=*/0,
                                                     now,
                                                     conv.fn(),
                                                     obs.emit_fn(),
                                                     retire_fn);
    ASSERT_EQ(outcome, finalize_outcome::result_ready);
    ASSERT_EQ(obs.emits, 1);
    ASSERT_EQ(obs.retires, 1);
    EXPECT_LE(obs.end, retired_ts + tolerance);
}

// A throwing client callback must not skip cleanup: the scope destructor retires
// exactly once on the way out (AGENTS.md RAII requirement).
TEST(complete_signal_less_dispatch, throwing_emit_still_retires_exactly_once)
{
    auto obs  = observer{};
    auto conv = converter{};
    EXPECT_THROW(run_complete_signal_less_dispatch(
                     std::optional<uint64_t>{500},
                     900,
                     0,
                     10'000'000,
                     conv.fn(),
                     [](uint64_t, uint64_t) { throw std::runtime_error{"callback failed"}; },
                     obs.retire_fn()),
                 std::runtime_error);
    EXPECT_EQ(obs.retires, 1);
}

// Each rejection cause must be attributed correctly: mislabelled counters send the
// next investigation the wrong way. The id retires exactly once regardless.
TEST(complete_signal_less_dispatch, reports_the_exact_rejection_cause)
{
    constexpr uint64_t now = 1'000'000'000;
    auto               run = [&](std::optional<uint64_t> start_ticks,
                   uint64_t                end_ticks,
                   uint64_t                enqueue_ts,
                   bool                    convert_ok) {
        auto obs     = observer{};
        auto conv    = converter{convert_ok, /*epoch=*/0};
        auto detail  = finalize_detail{};
        auto outcome = run_complete_signal_less_dispatch(start_ticks,
                                                         end_ticks,
                                                         enqueue_ts,
                                                         now,
                                                         conv.fn(),
                                                         obs.emit_fn(),
                                                         obs.retire_fn(),
                                                         &detail);
        EXPECT_EQ(obs.retires, 1);
        return std::make_pair(outcome, detail.reason);
    };
    struct row
    {
        std::optional<uint64_t> start;
        uint64_t                end, enqueue;
        bool                    convert_ok;
        finalize_outcome        outcome;
        finalize_reason         reason;
        const char*             label;
    };
    const row rows[] = {
        {now - 5'000'000,
         now - 1'000'000,
         0,
         true,
         finalize_outcome::result_ready,
         finalize_reason::ready,
         "success -> ready + emit"},
        {std::nullopt,
         now - 1'000'000,
         0,
         true,
         finalize_outcome::completed_no_timing,
         finalize_reason::start_unknown,
         "shape ii: start lost"},
        {now - 5'000'000,
         now - 1'000'000,
         0,
         false,
         finalize_outcome::completed_no_timing,
         finalize_reason::convert_failed,
         "conversion refused"},
        {now - 1'000'000,
         now - 5'000'000,
         0,
         true,
         finalize_outcome::completed_no_timing,
         finalize_reason::bad_interval,
         "non-positive interval"},
        {now - 5'000'000,
         now - 1'000'000,
         now - 2'000'000,
         true,
         finalize_outcome::completed_no_timing,
         finalize_reason::before_enqueue,
         "starts before enqueue"},
        {1,
         now + kKfdFutureSlackNs + 1,
         0,
         true,
         finalize_outcome::completed_no_timing,
         finalize_reason::after_now,
         "end beyond now + slack"},
    };
    for(const auto& tc : rows)
    {
        auto [outcome, reason] = run(tc.start, tc.end, tc.enqueue, tc.convert_ok);
        EXPECT_EQ(outcome, tc.outcome) << tc.label;
        EXPECT_EQ(reason, tc.reason) << tc.label;
    }
}

// A few ms past now stays inside the slack: reported ready, NOT counted against
// after_now, else the breakdown blames the guard for the skew it absorbs.
TEST(complete_signal_less_dispatch, conversion_skew_is_not_counted_as_a_rejection)
{
    constexpr uint64_t now  = 1'000'000'000;
    auto               obs  = observer{};
    auto               conv = converter{true, /*epoch=*/0};
    auto               det  = finalize_detail{};
    auto outcome = run_complete_signal_less_dispatch(std::optional<uint64_t>{now - 5'000'000},
                                                     now + 2'700'000,
                                                     0,
                                                     now,
                                                     conv.fn(),
                                                     obs.emit_fn(),
                                                     obs.retire_fn(),
                                                     &det);
    EXPECT_EQ(outcome, finalize_outcome::result_ready);
    EXPECT_EQ(det.reason, finalize_reason::ready);
    EXPECT_EQ(obs.emits, 1);
}
