# Phase D experiment plan

## Question

Under an identical region input/output contract and memory budget, can joint
selection of intermediate sharding and M-dimension tiling beat stock XLA and a
locally selected plan for:

```text
Dot1 -> collective -> ReLU -> Dot2
```

## First experiment

- Devices: physical GPUs 4, 5, and 6.
- Global tensors: FP32 `3072x3072`.
- Output contracts, evaluated separately: replicated and column-sharded.
- Untiled candidates: stock XLA, A (RS->AR), B (RS->RS), C (AR before Dot2).
- Tiled candidates: A, B, and C with M-axis tile counts 2 and 4.
- Realizations: serialized `lax.scan` and scheduler-visible fixed unrolling.
- Timing: 10 warmups, 50 interleaved samples, three experiment repetitions.
- Correctness: compare sampled output values and global shape against an
  untiled reference.

Every candidate must finish with the selected output contract inside the timed
compiled region. Candidates with different boundary layouts are not directly
ranked.

## Why this search is restricted

At the exact 36 MiB hidden payload, measured three-GPU amortized costs are
3.495 ms for reduce-scatter and 6.832 ms for all-reduce. Local dot costs are
only about 0.23--0.25 ms. Communication dominates, while each additional tile
pays an approximately 1.3--1.6 ms small-message floor. Tile counts above four
are therefore unlikely to win before overlap is demonstrated.

The current combined HLO uses synchronous, non-pipelined all-reduce. Tiled
candidates are accepted as overlapping realizations only when optimized HLO
and a device trace show asynchronous start/done operations and physical
concurrency. Otherwise they are classified as serialized.

## Outputs

```text
measurements/phase_d/RUN/
  environment.json
  candidates.csv
  samples.csv
  hlo/
  traces/
```

Each candidate row records strategy, output contract, tile count, realization,
median/p10/p90 latency, compilation time, static peak-memory estimate,
collective sequence/count, HLO hash, and correctness.

## Decision rules

1. Deduplicate candidates with identical optimized-HLO hashes.
2. Reject candidates that violate the output contract or memory budget.
3. Rank by end-to-end median within one output contract.
4. Require improvement in all three repetitions; report confidence intervals.
5. Explain the winner with HLO and trace evidence, not isolated-cost sums.
6. Report at least one losing tiled case and attribute the loss to startup,
   matmul efficiency, memory, or missing overlap.

## Follow-up matrix

After the square FP32 experiment works, add BF16 transformer-like shapes one
at a time. Each unseen local matmul shape or collective payload is measured in
Phase C format before the Phase D candidate is ranked; the calibrated model is
not extrapolated beyond its lookup support.
