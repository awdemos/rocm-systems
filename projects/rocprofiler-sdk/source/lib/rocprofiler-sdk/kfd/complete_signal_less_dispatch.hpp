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

#pragma once

#include "lib/common/scope_destructor.hpp"
#include "lib/rocprofiler-sdk/kfd/correlation_types.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

// Completion of a signal-less dispatch, free of the HSA and tracing headers so
// every branch is unit-testable with injected callables. Runs on a task-group
// worker, or on the teardown thread draining deferred completions -- never on
// the reader or processor thread, and never under a hub/queue lock.

namespace rocprofiler
{
namespace kfd
{
// The two terminal outcomes of a proven completion. Both retire the correlation
// id; they differ only in whether a record is emitted.
enum class finalize_outcome
{
    result_ready,         // KFD timestamps emitted
    completed_no_timing,  // no record; start unknown or convert/sanity failed
};

// Why a completion ended up with no timing, reported so a no-timing spike can be
// attributed without a rebuild.
enum class finalize_reason
{
    ready = 0,       // RESULT_READY: record emitted
    start_unknown,   // shape ii -- EOP proved completion but its START was lost
    convert_failed,  // hsa_amd_profiling_convert_tick_to_system_domain said no
    bad_interval,    // converted start >= end
    before_enqueue,  // converted start precedes this dispatch's own enqueue
    after_now,       // converted end is beyond now + the conversion slack
};

// Everything the finalizer learned, for diagnostics.
struct finalize_detail
{
    finalize_reason reason    = finalize_reason::ready;
    uint64_t        start_ns  = 0;
    uint64_t        end_ns    = 0;
    bool            converted = false;
};

inline const char*
finalize_reason_name(finalize_reason r)
{
    switch(r)
    {
        case finalize_reason::start_unknown: return "start-unknown";
        case finalize_reason::convert_failed: return "convert-failed";
        case finalize_reason::bad_interval: return "bad-interval";
        case finalize_reason::before_enqueue: return "before-enqueue";
        case finalize_reason::after_now: return "after-now";
        case finalize_reason::ready: break;
    }
    return "ready";
}

// Decide the outcome and produce system-domain timestamps.
template <typename ConvertFn>
finalize_outcome
resolve_finalize(const std::optional<uint64_t>& start_ticks,
                 uint64_t                       end_ticks,
                 uint64_t                       enqueue_ts,
                 uint64_t                       now_ns,
                 ConvertFn&&                    convert,
                 uint64_t*                      start_ns_out,
                 uint64_t*                      end_ns_out,
                 finalize_detail*               detail = nullptr)
{
    auto _note = [detail](finalize_reason r) {
        if(detail) detail->reason = r;
        return finalize_outcome::completed_no_timing;
    };

    // Shape (ii): the EOP proved completion but its START was lost, so there is no
    // interval to report.
    if(!start_ticks) return _note(finalize_reason::start_unknown);

    if(!convert(*start_ticks, start_ns_out) || !convert(end_ticks, end_ns_out))
        return _note(finalize_reason::convert_failed);

    // Distinct firmware ticks can convert to the same nanosecond for a kernel
    // short enough that the conversion quantizes the interval away. The duration
    // is real and the completion was proven, so widen it to the smallest
    // representable interval rather than discarding it below; consumers require
    // start < end strictly, so equal timestamps cannot simply be accepted.
    // Identical ticks are left alone: firmware reported no duration at all, which
    // is the degenerate case the interval guard exists to catch.
    if(*start_ns_out == *end_ns_out && *start_ticks != end_ticks) *end_ns_out = *start_ns_out + 1;

    // A record whose converted end is seconds past `now` is stale/lapped, not a
    // conversion artifact; reject it before the clamp below can pull it back into
    // range. The correlation key does the real discrimination; this only bounds
    // the arithmetic.
    if(*end_ns_out > now_ns + kKfdFutureSlackNs) return _note(finalize_reason::after_now);

    // Tick-to-system-domain conversion re-syncs periodically, so a just-completed
    // dispatch's converted end can land a small amount AFTER a `now` sampled right
    // behind it. The signal path handles this in tracing::adjust_profiling_time by
    // clamping the interval into [enqueue, now]; mirror that here so a signal-less
    // end never lands after this dispatch's own retirement (sampled just after
    // `now`). Shift the whole interval to preserve the measured duration, matching
    // the signal path's operator-=/operator+=.
    if(*end_ns_out > now_ns)
    {
        const uint64_t _shift = *end_ns_out - now_ns;
        *end_ns_out           = now_ns;
        *start_ns_out         = (*start_ns_out > _shift) ? *start_ns_out - _shift : 0;
    }

    if(detail)
    {
        detail->converted = true;
        detail->start_ns  = *start_ns_out;
        detail->end_ns    = *end_ns_out;
    }

    // Same correlation guard the signal path uses, reported clause by clause.
    if(!(*start_ns_out < *end_ns_out)) return _note(finalize_reason::bad_interval);
    if(*start_ns_out < enqueue_ts) return _note(finalize_reason::before_enqueue);

    return finalize_outcome::result_ready;
}

// Convert, emit on success, and retire EXACTLY ONCE on every path.
template <typename ConvertFn, typename EmitFn, typename RetireFn>
finalize_outcome
run_complete_signal_less_dispatch(const std::optional<uint64_t>& start_ticks,
                                  uint64_t                       end_ticks,
                                  uint64_t                       enqueue_ts,
                                  uint64_t                       now_ns,
                                  ConvertFn&&                    convert,
                                  EmitFn&&                       emit,
                                  RetireFn&&                     retire,
                                  finalize_detail*               detail = nullptr)
{
    auto _cleanup = common::scope_destructor{[&retire]() { retire(); }};

    uint64_t _start_ns = 0;
    uint64_t _end_ns   = 0;
    auto     _outcome  = resolve_finalize(
        start_ticks, end_ticks, enqueue_ts, now_ns, convert, &_start_ns, &_end_ns, detail);

    // Only result_ready has timestamps to emit. completed_no_timing is equally
    // terminal -- the scope destructor above retires the id either way -- it just
    // has no record to produce, so there is deliberately no branch for it.
    if(_outcome == finalize_outcome::result_ready) emit(_start_ns, _end_ns);
    return _outcome;
}

}  // namespace kfd
}  // namespace rocprofiler
