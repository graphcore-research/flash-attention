# HSA Backward Packing Findings

## Summary

This note summarizes the packing and forward-reuse experiments for the synthetic long-context backward path in:

- [flash_hsa_synthetic_grid_sm100.py](/workspace/codebases/nanochat/flash-attention/flash_attn/cute/flash_hsa_synthetic_grid_sm100.py)
- [benchmark_hsa.py](/workspace/codebases/nanochat/flash-attention/tests/cute/benchmark_hsa.py)
- [test_hsa.py](/workspace/codebases/nanochat/flash-attention/tests/cute/test_hsa.py)

Short answer:

- packing is clearly worth it for the synthetic forward path
- it has not been worth carrying extra packed artifacts into backward
- the best pure-micro backward remains scalar `bucket_dense`
- the fastest e2e path remains hybrid: `direct_micro_fwd_sparse_mask_bwd`


## What Was Tested

### 1. Saved compressed `P`

Variant:

- `bucket_dense_saved_prob`

Idea:

- save compressed bucket-local `P` in forward
- skip `score -> prob` rebuild in backward

Versions tried:

- replay-style sidecar forward kernel
- inline row-compact forward save inside the original CTA lifetime
- `FP32` and `BF16` cache variants

Observed result:

- backward could get faster in some versions
- forward save cost outweighed the backward win
- after making the saved-`P` path exact enough, it still lost

Takeaway:

- at the current scalar `bucket_dense` quality level, rebuilding `P` is cheaper than writing and rereading it


### 2. Saved packed forward tensors, segmented family

Variant family:

- old `bucket_dense_saved_packed` segmented-direct-micro path

Idea:

- preserve forward-packed tensors and make backward consume that exact packed representation

Observed result:

- structurally far from the competitive row-compact `bucket_dense` kernel
- initially wrong
- slower even after partial repair

Takeaway:

- segmented packed backward is too far from the kernel family that is actually competitive


### 3. Saved packed union `K/V`, row-compact family

Variant:

- repurposed `bucket_dense_saved_packed`

Idea:

- stay row-compact and bucket-dense-shaped
- save packed union `K/V` from forward
- make backward read those packed union slots instead of gathering global `K/V` rows

Observed result:

- correctness became good
- forward tax became small after inline save
- backward gain was tiny

Representative narrow result:

```text
64k
bucket_dense:      fwd=1.262 ms  bwd=1.031 ms  fb=1.935 ms
saved_packed:      fwd=1.304 ms  bwd=0.961 ms  fb=1.971 ms

128k
bucket_dense:      fwd=4.131 ms  bwd=3.348 ms  fb=7.706 ms
saved_packed:      fwd=4.217 ms  bwd=3.344 ms  fb=7.863 ms
```

Takeaway:

- packed union `K/V` reuse alone is almost neutral on backward
- even a small extra forward save cost is enough to make it lose e2e


### 4. Inline row-compact wombo combo: saved packed union `K/V` + saved compressed `P`

Variant:

- current `bucket_dense_saved_packed`

Idea:

- emit both caches inside the original row-compact forward CTA lifetime
- backward consumes both:
  - packed union `K/V`
  - saved compressed `P`
- skip both:
  - global union `K/V` gathers
  - `score -> prob` rebuild

This is the strongest structural reuse version we tried.

Observed result:

- numerically clean
- still slower e2e

Representative direct smoke:

```text
64k
diffs: out=0 dq=7.5e-09 dk=7.5e-09 dv=7.5e-09
bucket_dense:       fwd=1.189 ms  bwd=1.506 ms  fb=1.811 ms
saved_packed_combo: fwd=1.291 ms  bwd=0.793 ms  fb=1.926 ms

128k
diffs: out=0 dq=3.7e-09 dk=3.7e-09 dv=3.7e-09
bucket_dense:       fwd=2.059 ms  bwd=1.099 ms  fb=3.237 ms
saved_packed_combo: fwd=2.263 ms  bwd=1.080 ms  fb=3.422 ms
```

Takeaway:

- even when both reusable pieces are saved inline, the forward-side write cost still dominates the tiny backward win


## Why Packing Helps Forward But Not Backward

Forward uses packing to regularize the main computation:

```text
packed rows + packed keys/values
  -> regular tiny dense attention
  -> one output O and one LSE
```

Backward has a different cost structure:

```text
dQ = dS @ K
dK += dS^T @ Q
dV += P^T @ dO
```

So even if packed artifacts are reused, backward still must emit three gradient tensors:

- `dQ`
- `dK`
- `dV`

That means saved-state reuse only wins if it removes a very large chunk of real hot work.

What the experiments show is:

- packing/reuse removes some recompute
- but it adds new memory movement:
  - forward-side writes of saved state
  - backward-side reads of saved state
  - extra cache/token plumbing
- once scalar `bucket_dense` got good enough, that memory movement cost more than the saved recompute


## Practical Takeaways

### 1. We are not “rebuilding the synthetic mask and packing twice” in the important path

For the promoted long synthetic path:

- forward already reuses the schedule/metadata
- backward already reuses the same direct row-compact plan
- the expensive remaining work is actual backward math/control, not rebuilding topology

### 2. Gather/scatter-style reuse is the wrong trade for this backward

The cleanest summary is:

```text
forward:
  gather/pack once -> good trade

backward:
  gather/pack/save/reload while still computing dQ/dK/dV -> bad trade
```

So for this problem:

- packing is worth it for forward
- the same packing is not worth carrying into backward

### 3. Scalar `bucket_dense` remains the best pure-micro comparator

Best current pure-micro path:

- scalar `bucket_dense`

Current overall winner:

- hybrid `direct_micro_fwd_sparse_mask_bwd`


## Current Recommendation

Do not spend more time on:

- saved `P`
- saved packed forward tensors
- segmented packed backward
- packed-union reuse branches as the main path

The highest-value next work is:

- optimize the real hybrid backward path directly
- keep scalar `bucket_dense` as the best pure-micro comparison point

