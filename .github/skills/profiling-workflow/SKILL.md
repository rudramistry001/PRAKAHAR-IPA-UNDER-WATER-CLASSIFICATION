---
name: profiling-workflow
description: Use before making any performance/throughput/memory optimization change to a training or inference pipeline — enforces profile-first, change-one-thing, benchmark, validate-accuracy discipline, with concrete PyTorch/CUDA profiling commands and what to report.
---

# Profiling Workflow

Never change something for performance reasons without measuring first.
"This should be faster" is a Hypothesized claim, not a Finding — it needs a
benchmark before it goes in a Recommendation.

## Required sequence

```
PROFILE            -> establish where time/memory actually goes
IDENTIFY BOTTLENECK -> name the specific operation/stage, with numbers
HYPOTHESIZE         -> state what change should help and why
CHANGE ONE FACTOR    -> exactly one variable per benchmark run
BENCHMARK           -> re-measure the same way as the baseline
COMPARE              -> report the delta, not just the new number
VALIDATE ACCURACY    -> confirm the change didn't silently change model behavior
```

Changing multiple factors at once (e.g. batch size + precision + dataloader
workers together) invalidates attribution — do it one factor at a time, or
explicitly note that the report is an unattributed combined effect.

## Concrete tooling

- **CPU/data pipeline**: `py-spy`, `cProfile`, or `torch.utils.data`
  worker-timing logs to find whether the bottleneck is I/O, CPU-side
  augmentation, or the collate function.
- **GPU compute**: `torch.profiler` (`torch.profiler.profile` with
  `ProfilerActivity.CUDA`) for per-op time; `nvidia-smi dmon` or
  `nvidia-smi -l 1` for utilization/memory over time.
- **Memory**: `torch.cuda.max_memory_allocated()` /
  `torch.cuda.memory_summary()` before/after; watch for fragmentation vs.
  genuine peak usage.
- **DataLoader**: check `num_workers`, `pin_memory`, `persistent_workers`,
  and whether GPU is starved (low utilization with high host CPU) vs.
  compute-bound (high utilization, host idle).
- **Mixed precision**: verify actual dtype used at each stage
  (`torch.autocast` scope, whether the loss is scaled via `GradScaler`) —
  don't assume AMP is active just because it's configured.
- **Distributed**: check communication-vs-compute overlap and whether
  gradient sync is the bottleneck (all-reduce time relative to step time).

## What to report

Always report: latency (p50, and p99 if variance matters), throughput
(samples/sec), peak memory, GPU utilization, CPU utilization, and — every
time — accuracy/loss impact on a fixed validation subset. A latency win with
an unchecked accuracy impact is an incomplete result, not a finished one.

Use the `ml-output-contracts` skill's optimization-output extension
(Baseline Benchmark / Optimized Benchmark / Performance Delta / Accuracy
Delta) to report results.
