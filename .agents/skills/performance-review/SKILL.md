---
name: performance-review
description: Diagnoses and improves runtime, database, frontend, build, memory, or training performance. Use when behavior is slow, resource-heavy, timing-sensitive, or needs measurable optimization.
---

# Performance Review

Use this when optimizing or explaining performance.

## Method

1. Define the performance symptom and target metric.
2. Establish a baseline before changing code.
3. Identify likely bottlenecks with measurement, not guesswork.
4. Apply the smallest change expected to move the metric.
5. Re-measure and compare with the baseline.
6. Check for correctness regressions.

## Common Metrics

- Frontend: bundle size, interaction latency, layout shift, FPS.
- Backend: p50/p95 latency, query count, CPU, memory, throughput.
- Database: query plan, index usage, lock time, rows scanned.
- Unity/game: frame time, allocations, update cost, GC spikes.
- ML/training: samples/sec, loss stability, GPU/CPU utilization.

## Output

Report baseline, change, measured result, tradeoffs, and remaining bottlenecks.
