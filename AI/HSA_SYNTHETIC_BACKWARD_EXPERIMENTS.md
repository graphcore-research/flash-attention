# HSA Synthetic Backward Experiments

## Context

This note summarizes the long-running attempt to make the synthetic micro backward path competitive with the current hybrid path for long-context training in:

- [flash_hsa_synthetic_grid_sm100.py](/workspace/codebases/nanochat/flash-attention/flash_attn/cute/flash_hsa_synthetic_grid_sm100.py)
- [benchmark_hsa.py](/workspace/codebases/nanochat/flash-attention/tests/cute/benchmark_hsa.py)
- [test_hsa.py](/workspace/codebases/nanochat/flash-attention/tests/cute/test_hsa.py)

The forward path was already strong. The open question was whether the same compression idea could be made to translate into backward well enough to beat the hybrid path on ultra-long contexts.

Short answer: not with the designs tried here.


## Problem Statement

### What works well in forward

The direct micro forward turns each qgroup bucket into a tiny dense problem:

- `packed_q = 2`
- `union_k <= 16`
- `head_dim = 64`
- `max_direct_segments = 1`

For each qgroup bucket we can think of:

```text
Q_b   in R^(2 x 64)
K_b   in R^(U x 64)
V_b   in R^(U x 64)
O_b   in R^(2 x 64)
LSE_b in R^(2)
U <= 16
```

Forward is naturally q-row owned:

```text
2 query rows -> tiny compressed union of keys -> one tiny dense attention -> write O/LSE once
```

That maps very well to the row-compact direct metadata.

### Why backward is hard

Backward does not have a single natural owner:

- `dQ` is q-row owned
- `dK/dV` are key-row owned

That means a naive micro backward has to fight an ownership mismatch:

```text
qgroup-local math wants row ownership
global dK/dV accumulation wants key ownership
```

This is the core reason backward kept turning into reduction plumbing.


## Reference Math

The compressed bucket backward we were trying to realize is:

```text
S      = Q_b @ K_b^T
P      = softmax(S)
dProb  = dO_b @ V_b^T
dpsum  = rowwise_sum(O_b * dO_b)
dS     = P ⊙ (dProb - dpsum[:, None]) * softmax_scale
dQ     = dS @ K_b
dK    += dS^T @ Q_b
dV    += P^T @ dO_b
```

The hope was:

- forward compression reduces the effective K dimension from full sequence length to `U <= 16`
- backward should be able to operate directly on that same compressed problem
- only the final `dK/dV` writeback should need global atomics


## Two Competing Backward Shapes

### Hybrid path

```text
micro forward
  -> build packed sparse-mask metadata
  -> run large amortized sparse/block backward kernel
```

This path pays some preprocessing, but the actual hot backward kernel is large, amortized, and regular.

### Pure micro paths tried here

```text
micro forward
  -> run custom synthetic backward directly over row-compact metadata
```

This keeps the irregularity alive in the hot loop. That turned out to be the defining difficulty.


## High-Level Result

The best pure-micro long backward we found was scalar `bucket_dense` with `qgroups_per_cta_bwd=2`.

Representative warm long-context results:

| Case | Hybrid | Scalar `bucket_dense` | Legacy `long4` |
| --- | ---: | ---: | ---: |
| `64k` | `1.815 ms` | `1.990 ms` | `2.034 ms` |
| `128k` | `3.245 ms` | `3.564 ms` | `3.655 ms` |
| `256k` | `6.032 ms` | `6.685 ms` | `6.772 ms` |
| `512k` | `11.534 ms` | `12.747 ms` | `13.041 ms` |
| `1M` | `22.874 ms` | `25.235 ms` | `25.734 ms` |

So we improved pure-micro backward a lot relative to earlier one-kernel designs, but still did not beat hybrid.


## Important Correctness Fixes

### 1. Micro forward saved-state bug

At one point `train-eq` had non-finite `dQ`, but the root cause was not backward. It was the synthetic micro forward saved-state path.

Grounded diagnosis:

- plain sparse-mask HSA: finite
- stable synthetic forward + sparse-mask backward: finite
- micro synthetic forward + sparse-mask backward: non-finite

Fix:

- rewrote micro forward to use a single exact streaming softmax state per row:
  - `running_max`
  - `running_sum`
  - `out_acc`

This restored finite gradients and removed a branch-wide correctness blocker.


## Experiment Timeline

### 1. One-kernel backward writeback rewrite

#### Goal

Reduce obvious backward writeback overhead:

- eliminate per-occurrence global `dQ` atomics
- replace scalar `dK/dV` flushes with grouped `fp32x4` atomics

#### What we tried

- CTA-local unique-row `dQ` accumulator
- grouped `fp32x4` `dK/dV` epilogues
- short/long one-kernel variants

#### What happened

The grouped `dK/dV` epilogue was fine.

The shared `dQ` accumulator was a disaster.

Representative `ncu` fingerprint on `train-eq`:

- short writeback kernel: about `227.6 us`
- legacy one-kernel baseline: about `70.1 us`
- huge shared-memory pressure
- `L1 Wavefronts Shared Excessive = 25,811,168`
- high short-scoreboard stalls

#### Conclusion

Shared `dQ` accumulation was the wrong optimization. It created a shared-memory bottleneck worse than the original global-atomic behavior.


### 2. Long one-kernel scaling by wider CTA / more keys per CTA

#### Goal

Beat hybrid on long contexts by increasing useful work per CTA.

Tried:

- `short`
- `long`
- `warpgroup`
- parametric `keys_per_cta = 4 / 8 / 16`

#### What happened

This helped a little in some narrow regimes, especially compared with older fallback paths, but never changed the main picture.

Representative result:

- `keys_per_cta = 4` remained the best pure-micro long one-kernel
- `8` and `16` usually regressed

#### Conclusion

Launch granularity was not the main remaining problem.


### 3. Two-stage packed-row backward

#### Goal

Pre-pack reusable row state once, then run a key-owned reduction kernel over packed rows.

#### What happened

- correct
- structurally matched the intended design
- slower than the simpler one-kernel long path

#### Why it lost

Prepass + packing + final scatter overhead outweighed the row-state reload savings.


### 4. Persistent long backward

#### Goal

Collapse the giant grid of tiny long-context CTAs into a persistent grid over key tiles.

#### What happened

- correct
- slower than non-persistent `one_kernel_long`

Representative profile at `32k`:

- `one_kernel_long` backward: about `243 us`
- `persistent_long` backward: about `314 us`

#### Why it lost

Persistent scheduling/worklist overhead did not fix the actual inner-loop cost.


### 5. Persistent member-tiled backward

#### Goal

Keep persistent scheduling, but reuse row state across multiple keys inside each persistent tile via a CTA-local member cache.

#### What happened

- functionally correct after fixes
- still slower
- latency/divergence bound

Representative `ncu` shape:

- much worse `One or More Eligible`
- much worse warp cycles per issued instruction
- scoreboard/MIO-style stall pattern from the cache/shared path

#### Why it lost

The member-cache management overhead cost more than the saved reloads.


### 6. Bucket-dense compressed backward

#### Goal

Stop doing key-owned reverse-incidence traversal and instead compute backward directly over the compressed `[2 x U]` bucket.

This was the first design that actually mirrored why the forward pass worked well.

#### Why it was promising

Instead of:

```text
for each unique key occurrence:
  reload row state
  recompute score/prob
  atomically update dQ
  accumulate dK/dV
```

We moved toward:

```text
for each qgroup bucket:
  operate on the tiny compressed [2 x U] dense problem directly
  keep dQ qgroup-owned
  use global atomics only for dK/dV
```

#### Result

This became the best pure-micro long path.

It was the right direction, but still not enough to beat hybrid.


### 7. Source-correlated `ncu` on scalar `bucket_dense`

#### Goal

Find the dominant remaining cost family inside scalar `bucket_dense`.

#### What we learned

At `128k` and `512k`, the dominant excessive-sector instructions were repeated scalar bf16 global loads in the union-loop `K/V` path:

- hot `LDG.E.U16` family
- about `46%` excessive global sectors on `bucket_dense`

Important non-finding:

- grouped `REDG.E.ADD.F32x4` atomics for `dQ` and `dK/dV` were not the issue
- those showed `0` excessive sectors

#### Conclusion

The atomics were not the remaining bottleneck. The scalar global load pattern was.


### 8. Shared-staged scalar `bucket_dense`

#### Goal

Fix the bad scalar `LDG.E.U16` family by staging:

- `Q`
- `O`
- `dO`
- `K`
- `V`

in shared memory using a forward-style cooperative load.

#### What happened

- correctness was restored after fixing a real staging/metadata race
- performance regressed badly

Representative `128k` result:

- staged `bucket_dense` `fwd+bwd`: about `4.96 ms`
- hybrid: about `3.26 ms`
- legacy `long4`: about `3.64 ms`

#### Why it lost

Shared staging removed the bad global-load family, but the staging/barrier/shared-load overhead was larger than the memory win.

This was an important lesson:

```text
memory-shape cleanup alone is not enough;
the remaining problem is also the math/reduction structure
```


### 9. Tensor-core `bucket_dense_tc`

#### Goal

Turn the compressed backward into a tiny dense tensor-core problem:

- padded `16 x 16` tiles
- BF16 inputs
- FP32 accumulation
- one dense microkernel per qgroup bucket

#### What happened

- exact on the bring-up cases
- much slower than scalar `bucket_dense`

Representative `128k` result:

| Variant | `fwd` | `bwd` | `fwd+bwd` |
| --- | ---: | ---: | ---: |
| Hybrid | `2.110 ms` | `1.889 ms` | `3.289 ms` |
| Scalar `bucket_dense` | `2.331 ms` | `2.486 ms` | `3.616 ms` |
| `bucket_dense_tc` | `2.152 ms` | `4.798 ms` | `6.304 ms` |

Profiler result:

- scalar `bucket_dense` backward self-CUDA: about `843 us`
- `bucket_dense_tc` backward self-CUDA: about `3478 us`

#### Why it lost

The TC decomposition made the kernel much heavier than the scalar compressed-bucket baseline.


### 10. Full-warp dual-row scalar `bucket_dense`

#### Goal

Eliminate the duplicated per-row `K/V` load family inside scalar `bucket_dense` by making one full warp own both q rows:

- 32 lanes
- 2 dims per lane
- load `K/V` once per lane
- use the same fragment for both rows

#### What happened

- `out`, `dK`, and `dV` stayed exact
- `dQ` still lived in the same rough experimental band as the rest of the micro backward family
- backward-only time sometimes improved
- end-to-end did not

Representative results:

| Case | Hybrid | Scalar `bucket_dense` | `bucket_dense_dualrow` | `long4` |
| --- | ---: | ---: | ---: | ---: |
| `128k` | `3.268 ms` | `3.615 ms` | `3.794 ms` | `3.614 ms` |
| `256k` | `6.002 ms` | `6.750 ms` | `7.091 ms` | `6.790 ms` |
| `512k` | `11.541 ms` | `12.909 ms` | `13.637 ms` | `13.080 ms` |

#### Why it lost

It reduced some duplicated backward work, but not enough to beat the simpler scalar bucket-dense path end-to-end.


## Why Hybrid Kept Winning

The hybrid path was hard to beat because it was not merely "paying some packing tax."

It was doing something structurally better:

```text
irregular synthetic problem
  -> regularize into packed sparse/block metadata
  -> run a large amortized backward kernel
```

Meanwhile, all pure-micro backward variants kept some form of this inside the hot kernel:

- irregular participation checks
- tiny-bucket probability / `dS` math
- repeated local reductions
- global `dK/dV` accumulation

The preprocessing cost on hybrid was not the real issue. The main win was the better hot backward kernel shape.


## Best Current Mental Model

### Forward

Forward wins because the ownership matches the compression:

```text
qgroup bucket owns output rows
```

### Backward

Backward loses because ownership is split:

```text
qgroup bucket naturally owns dQ
global key rows naturally own dK/dV
```

Pure-micro backward tries to solve both in one irregular hot loop.

Hybrid regularizes the reduction problem before running the expensive kernel.


## What Actually Worked

These changes were useful and worth keeping conceptually:

- exact streaming-softmax micro forward saved-state fix
- grouped `fp32x4` epilogues instead of scalar flushes
- compressed-bucket `bucket_dense` backward
- long profiling mode in `benchmark_hsa.py`
- explicit comparator variants for A/B work


## What Did Not Work

The following were either clear losses or not good enough to promote:

- CTA-local shared `dQ` accumulator
- larger long one-kernel CTAs alone
- persistent long backward
- persistent member-tiled backward
- packed two-stage long backward
- shared-staged scalar `bucket_dense`
- tensor-core `bucket_dense_tc`
- full-warp dual-row scalar `bucket_dense`
- `qgroups_per_cta_bwd=4` as a default


## Recommendations

### Product path

Use hybrid as the real long-context backward path.

### Research / experimental path

Keep scalar `bucket_dense` as the best pure-micro comparator and fallback reference.

### What not to do next

Do not spend more time on:

- persistent scheduling variants
- larger one-kernel CTAs by themselves
- shared `dQ` accumulation
- more local kernel surgery that does not change the structural reduction story

### If pure-micro backward is revisited

The next serious idea should be a new algorithmic decomposition, not another local optimization pass.

The key requirement would be:

```text
preserve forward-style compression
while regularizing the backward reduction problem enough
that the hot kernel stops carrying irregularity directly
```


## Bottom Line

The experiments were not a failure, but they did rule out a large local neighborhood.

We learned that:

1. the forward compression idea is good
2. the backward problem is harder because of ownership mismatch
3. the best pure-micro backward we found is scalar `bucket_dense`
4. hybrid still wins because its backward is structurally more regular and more amortized

That is the current state of play.
