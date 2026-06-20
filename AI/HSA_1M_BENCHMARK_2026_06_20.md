# HSA Benchmarks Through 1M Context, 2026-06-20

Branch: `HSA-FA`, starting commit `cf287d3`.

GPU: `NVIDIA GB200`, capability `(10, 0)`, selected with
`CUDA_VISIBLE_DEVICES=1`.

All commands were run from `/workspace/codebases/nanochat/flash-attention` with
`PYTHONPATH=.` and hard `timeout` wrappers. Dense/all-payload Python paths were
not run at long sequence lengths. Explicit 2D long runs used
`--skip-correctness`; correctness was checked separately at 4K.

Interpretation note: for explicit 2D HSA training use, payloads are assumed to
be cached/precomputed. The `build_s` column is setup cost only. Reported
speedups for explicit 2D HSA versus FA4-packed exclude payload construction and
compare only timed CUDA hot-path forward kernels.

## AR-HSA Walk vs Sliding FA4

Command template:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. timeout 240s python -u tests/cute/benchmark_arhsa_walk.py \
  --n-queries <SEQ> --n-heads 4 --head-dim-v 64 --leaves-per-query 4 \
  --n-iters 3 --graph-mode level_dag --level-range-kernels --incoming-packed-step \
  --query-warp-readout --query-warp-fused-bwd --skip-torch --no-check --no-memory \
  --iters 1 --warmup 1 --compare-fa4 --fa4-seqlen <SEQ> --fa4-n-heads 4 \
  --fa4-head-dim 64 --fa4-iters 1 --fa4-warmup 1 --fa4-window-left 5
```

For 16K the timeout was `180s`; the command first failed without
`--incoming-packed-step`, then passed with the template above.

| seq | AR-HSA full_cute_hot_ms | AR-HSA fwd+bwd prealloc ms | AR-HSA custom fwd+bwd ms | sliding FA4 fwd+bwd ms | FA4 / AR prealloc |
|---:|---:|---:|---:|---:|---:|
| 16K | 0.199 | 0.663 | 1.585 | 0.840 | 1.27x |
| 64K | 0.243 | 0.708 | 1.604 | 0.874 | 1.23x |
| 256K | 0.306 | 0.898 | 1.525 | 1.159 | 1.29x |
| 1M | 0.877 | 2.718 | 2.883 | 3.757 | 1.38x |

Notes:

- This benchmark uses a fixed synthetic graph size: `n_nodes=4096`,
  `n_edges=24576`, `n_iters=3`, `leaves_per_query=4`, and scales query count.
- The hot path scales well to 1M for this AR-HSA walk setup.
- The sliding FA4 comparator is `window_left=5`, not dense causal FA4.

## Cached HSA Primary Long vs FA4/SWA-like Baselines

Command template:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. FLASH_ATTN_HSA_BENCH_MAX_RETRIES=1 \
  FLASH_ATTN_HSA_PRIMARY_LONG_ONLY=1 FLASH_ATTN_HSA_PRIMARY_LONG_SEQLENS=<SEQ> \
  FLASH_ATTN_HSA_PRIMARY_LONG_WARMUP_ITERS=1 FLASH_ATTN_HSA_PRIMARY_LONG_BENCHMARK_ITERS=1 \
  timeout 420s python -u tests/cute/benchmark_hsa.py
```

1M forward-only timeout command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. FLASH_ATTN_HSA_BENCH_MAX_RETRIES=1 \
  FLASH_ATTN_HSA_PRIMARY_LONG_ONLY=1 FLASH_ATTN_HSA_PRIMARY_LONG_FORWARD_ONLY=1 \
  FLASH_ATTN_HSA_PRIMARY_LONG_SEQLENS=1M FLASH_ATTN_HSA_PRIMARY_LONG_WARMUP_ITERS=1 \
  FLASH_ATTN_HSA_PRIMARY_LONG_BENCHMARK_ITERS=1 timeout 240s python -u tests/cute/benchmark_hsa.py
```

| seq | HSA fwd ms | HSA bwd ms | HSA fwd+bwd ms | dense FA4 fwd+bwd ms | sliding logS FA4 fwd+bwd ms | sliding flop-matched FA4 fwd+bwd ms | status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 16K | 0.980 | 6.136 | 6.086 | 1.802 | 0.899 | 0.794 | measured |
| 64K | 1.956 | 350.539 | 342.565 | 9.764 | 1.358 | 1.331 | measured |
| 256K | 5.784 | 1281.212 | 1279.380 | 139.188 | 1.803 | 1.785 | measured |
| 1M | - | - | - | - | - | - | fwd+bwd timed out at 420s; fwd-only timed out at 240s |

Notes:

- This path is not the fast AR-HSA walk path above. It exercises
  `hsa_sparse_mask_plain` / cached-HSA primary long plumbing, and is dominated
  by the old sparse-mask backward stack from 64K onward.
- The 1M timeout happens before a timing line, so setup/runtime construction or
  the first measured step is too slow for this bounded run.

## Explicit 2D Packed, D64

4K correctness command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. timeout 180s python -u tests/cute/benchmark_hsa_2d_sparse.py \
  --case-family disjoint_confetti --seqlen 4096 --heads 8 --head-dim 64 \
  --packed-q 16 --support-k 64 --islands-per-row 4 --island-width 4 \
  --variants direct_2d,direct_2d_compact,fa4_packed --warmup-iters 1 \
  --benchmark-iters 1 --json
```

Long perf command template:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. timeout <TIMEOUT>s python -u tests/cute/benchmark_hsa_2d_sparse.py \
  --case-family disjoint_confetti --seqlen <SEQ> --heads 8 --head-dim 64 \
  --packed-q 16 --support-k 128 --islands-per-row 8 --island-width 4 \
  --variants direct_2d_compact --warmup-iters 1 --benchmark-iters 1 --skip-correctness
```

| seq | build_s | direct_2d ms | direct_2d_compact ms | FA4-packed ms | correctness/status |
|---:|---:|---:|---:|---:|---|
| 4K | 6.004 | 0.734 | 0.540 | 3.527 | maxdiff `9.54e-07`, mean `3.11e-08` |
| 16K | 0.464 | - | 2.546 | 13.710 | cached hot path, `5.39x` faster |
| 64K | 0.570 | - | 9.325 | 48.402 | cached hot path, `5.19x` faster |
| 256K | 0.565 | - | 36.624 | 250.736 | cached hot path, `6.85x` faster |
| 1M | 0.533 | - | 145.843 | 1221.157 | cached hot path, `8.37x` faster |

Notes:

- Post-fix setup time is flat at about `0.5-0.6s` through 1M for the full-span
  compact case, but setup is not included in the cached-kernel speedup. Before
  the fix, 256K took `189.830s` to build and 1M timed out at `300s`.
- The fix vectorizes disjoint-confetti mask construction, skips exact expensive
  geometry only for `--skip-correctness` perf runs, reuses full-span compact
  payload tensors instead of copying Q/K/V, reuses precomputed mask words, and
  skips redundant output scatter for contiguous q rows.
- Kernel time scales roughly linearly and is now decisively faster than
  FA4-packed at 1M for this sparse benchmark.

## Explicit 2D Packed, D128

Enabled correctness command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. timeout 180s python -u tests/cute/benchmark_hsa_2d_sparse.py \
  --case-family disjoint_confetti --seqlen 4096 --heads 8 --head-dim 128 \
  --packed-q 16 --support-k 64 --islands-per-row 4 --island-width 4 \
  --variants direct_2d,direct_2d_compact,fa4_packed --warmup-iters 1 \
  --benchmark-iters 1 --json
```

D128 disabled gate command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. FLASH_ATTN_HSA_CACHED_GATHER_D128=0 \
  timeout 180s python -u tests/cute/benchmark_hsa_2d_sparse.py \
  --case-family disjoint_confetti --seqlen 4096 --heads 8 --head-dim 128 \
  --packed-q 16 --support-k 64 --islands-per-row 4 --island-width 4 \
  --variants direct_2d --warmup-iters 1 --benchmark-iters 1 --json
```

D128 16K perf command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. timeout 240s python -u tests/cute/benchmark_hsa_2d_sparse.py \
  --case-family disjoint_confetti --seqlen 16384 --heads 8 --head-dim 128 \
  --packed-q 16 --support-k 128 --islands-per-row 8 --island-width 4 \
  --variants direct_2d_compact --warmup-iters 1 --benchmark-iters 1 --skip-correctness
```

| seq | build_s | direct_2d ms | direct_2d_compact ms | FA4-packed ms | correctness/status |
|---:|---:|---:|---:|---:|---|
| 4K | 27.936 | 1.022 | 1.010 | 3.038 | maxdiff `7.15e-07`, mean `3.11e-08` |
| 4K, D128 disabled | 28.035 | unsupported_shape | - | - | `FLASH_ATTN_HSA_CACHED_GATHER_D128=0` gate works |
| 16K | 12.218 | - | 5.213 | - | pre-fix perf-only |
| 1M | 0.534 | - | 293.556 | 1216.928 | post-fix perf-only with FA4-packed, `4.15x` faster |

Code-level route/gate evidence:

- D128 direct is gated by `FLASH_ATTN_HSA_CACHED_GATHER_D128` and
  `FLASH_ATTN_HSA_CACHED_GATHER_D128_MAX_PACKED_K`, default max `128`.
- D128 TC gather/scatter exists but requires `rows_per_cta=16` and
  `tile_k=32`; unsupported shapes raise
  `synthetic_2d_masked_gather_scatter_tc_fwd_unsupported_shape`.
- The public explicit 2D benchmark does not print whether the default direct
  route chose scalar or TC internally, so the timing above is the currently
  routed default path plus an explicit disabled-gate check.

## Fixed / Gated / Blocked

Fixed:

- Current AR-HSA walk hot path scales to 1M and beats the sliding-left-5 FA4
  comparator in this synthetic setup.
- Explicit 2D D64 and D128 direct paths remain numerically clean at 4K.
- Explicit 2D D64/D128 full-span compact payload setup no longer wedges at
  long context. Excluding cached setup, 1M D64 now runs in `145.843 ms` vs
  FA4-packed `1221.157 ms`;
  1M D128 runs in `293.556 ms` vs FA4-packed `1216.928 ms`.

Gated:

- D128 direct is default-enabled only inside the D128 packed-k gate; disabling
  `FLASH_ATTN_HSA_CACHED_GATHER_D128` correctly returns `unsupported_shape`.
- D128 TC path remains shape-gated to `rows_per_cta=16`, `tile_k=32`.
- Explicit 2D runtime recommendation remains `do_not_route` for direct-only
  benchmark suites when the full direct/custom/FA4 go/no-go set is not present.

Blocked:

- Cached-HSA primary long fwd+bwd is dominated by old sparse-mask backward and
  times out at 1M. This is not competitive with the AR-HSA walk hot path.
- Explicit 2D compact payload construction is fixed for the benchmark full-span
  passthrough case. Real cached-training payload generation may still need the
  same vectorization/precompute treatment for non-full-span and overlapping
  residual payloads.
- The benchmark surface does not currently expose a clean scalar-vs-TC selector
  for explicit 2D D128, so scalar/TC split timing needs either a script flag or
  a lower-level helper benchmark.

Next optimization target:

1. Carry the same no-copy/full-span passthrough and vectorized mask/payload
   construction pattern into real cached-training payloads, including
   non-full-span and overlapping residual cases.
2. Add explicit benchmark flags for direct 2D scalar vs TC routes, especially
   D128, so the routed default can be decomposed without editing code.
3. Do not spend time on the cached-HSA primary long backward path unless that
   path is still intended; AR-HSA walk is the scaling path that looks healthy.
