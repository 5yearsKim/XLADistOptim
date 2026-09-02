# Phase C: GPU measurement and calibration results

## Scope

This calibration targets the current workstation, JAX 0.11.1, and physical
GPUs 4, 5, and 6. It covers isolated FP32/BF16 matmuls, all-reduce,
reduce-scatter, all-gather, relevant back-to-back collective sequences, and an
independent BF16 matmul plus all-reduce region.

Both synchronous host-observed latency and amortized launch throughput are
recorded. The cost tables use amortized medians; raw samples are retained.

## Principal observations

At a 64 MiB input payload per rank, the amortized collective medians are:

| Devices | All-reduce | Reduce-scatter | All-gather |
|---|---:|---:|---:|
| 4,5 | 4.431 ms | 2.396 ms | 4.385 ms |
| 4,6 | 2.494 ms | 1.482 ms | 2.787 ms |
| 4,5,6 | 11.478 ms | 5.691 ms | 21.209 ms |

The measured 4,5 pair is slower than 4,6 at this size despite its `PIX`
topology label. Cost selection must therefore use measured configuration keys,
not infer performance from the coarse topology label.

On three GPUs, the relevant back-to-back sequences cost:

| Input per rank | RS -> AR | RS -> RS |
|---|---:|---:|
| 1 MiB | 2.067 ms | 1.207 ms |
| 16 MiB | 2.917 ms | 2.346 ms |
| 64 MiB | 9.207 ms | 7.385 ms |

For the tested realization, RS -> RS is consistently faster.

BF16 matmul throughput ranges from roughly 14 TFLOP/s for 1024-cubed to
358 TFLOP/s for the large rectangular shapes. FP32 ranges from roughly 23 to
185 TFLOP/s. FLOP count alone is not a sufficient predictor; `(M,K,N,dtype)`
is the lookup key.

The combined overlap experiment uses a per-rank 2048-cubed BF16 matmul. The
optimized HLO marks each all-reduce `is_sync=true` and `is_pipelined=false`.
The supported model for this realization is therefore serialized. After
removing one shared launch baseline, its prediction error is 1.9% median and
15.1% maximum; the small-message case is dominated by subtraction noise.

## Validation

The lookup-backed model deliberately retains model failures:

| Component | Median relative error | Maximum relative error |
|---|---:|---:|
| Collective interpolation | 8.9% | 115.8% |
| Matmul FLOP-only diagnostic | 35.4% | 103.1% |
| Serialized combined model | 1.9% | 15.1% |

The large collective error occurs around backend algorithm/regime transitions.
The large matmul error shows shape-dependent accelerator efficiency. Phase D
must use exact lookup entries when available and measure missing candidate
shapes rather than extrapolate through these discontinuities.

Exact Phase D points were therefore added: the three-GPU FP32 36 MiB payload
costs 6.832 ms for all-reduce, 3.495 ms for reduce-scatter, and 9.030 ms for
all-gather. The two local matmul orientations cost 0.248 ms for
`3072x1024x3072` and 0.231 ms for `3072x3072x1024`.

Memory values are XLA static executable analysis. They are useful for pruning
candidates, but final claims should validate allocator high-water marks in a
device trace.

## Artifacts

- `measurements/phase_c_v2/collectives`: summaries, environments, raw samples.
- `measurements/phase_c_v2/matmul`: FP32/BF16 summaries and raw samples.
- `measurements/phase_c_v2/overlap`: summaries, samples, and optimized HLO.
- `measurements/phase_c_v2/sequences`: back-to-back sequence data and HLO.
- `measurements/phase_c_v2/calibration`: lookup model, validation CSV, report,
  and plots.

## Phase D constraint

The calibrated model is valid only for measured device configurations,
shapes, dtypes, and payload sizes. Phase D should enumerate a restricted search
whose candidates map to these entries. A missing point triggers measurement,
not unrestricted extrapolation.
