# Research Direction: Communication-Aware Distributed Lowering for TPU/XLA

## Research Direction

Develop a **communication-aware distributed lowering optimization for TPU/XLA** that jointly decides:

\[
(	ext{tile size},\ 	ext{sharding},\ 	ext{collective realization},\ 	ext{schedule})
\]

for compute–communication regions such as:

```text
MatMul
  ↓
ReduceScatter
  ↓
RMSNorm
  ↓
AllGather
  ↓
MatMul
```

The goal is to replace coarse execution:

```text
Compute → Collective → Compute
```

with tile-level pipelined execution:

```text
compute(tile i)
communicate(tile i-1)
consume(tile i-2)
```

to minimize end-to-end makespan while respecting TPU-specific constraints such as ICI topology/bandwidth, MXU utilization, HBM capacity, and collective startup overhead.

A possible formulation is:

\[
\min_{b,S,A,P} T_{	ext{makespan}}
\]

where:

- \(b\): tile size
- \(S\): sharding
- \(A\): collective implementation
- \(P\): pipeline schedule

## Research Plan

1. **Characterize existing XLA behavior**  
   Study `Dot → ReduceScatter/AllGather` patterns, GSPMD, async collectives, and windowed einsum. Measure where communication is serialized and where XLA already overlaps it.

2. **Build a cost model**  
   Model compute time, communication startup/bandwidth, topology, MXU efficiency, and additional buffer usage as functions of tile size and sharding.

3. **Develop the optimizer**  
   Given an HLO region, enumerate or search feasible tile sizes, communication realizations, and schedules, then select the lowest-cost configuration.

4. **Implement an XLA prototype**  
   Transform selected HLO regions into tiled compute plus asynchronous/chunked collectives and expose the required scheduling dependencies.

5. **Evaluate on TPU**  
   Test representative LLM operators/blocks and compare against stock XLA/GSPMD/windowed-einsum lowering using:
   - latency
   - communication overlap
   - HBM usage
   - MXU utilization

## Core Research Question

> **Can an XLA lowering pass automatically co-optimize computation tiling and collective communication to create efficient distributed software pipelines on TPU?**

## Key Novelty

The key novelty should be the **joint optimization of lowering decisions**, rather than simply adding communication/computation overlap, because XLA already contains several mechanisms for overlap.
