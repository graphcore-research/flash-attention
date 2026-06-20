# HSA Fusion Checklist Timing Notes

Date: 2026-06-19
GPU: NVIDIA GB200, CUDA capability (10, 0)
Torch: 2.9.0a0+145a3a7bda.nv25.10

## Forward

Clean fused-tail routing is covered by
`tests/cute/test_hsa_cached_2d_helpers.py::test_cached_2d_monolithic_forward_requires_complete_fused_tail_payload`
and
`tests/cute/test_hsa_cached_2d_helpers.py::test_cached_2d_forward_prefers_monolithic_clean_fused_tail`.

Direct-final residual routing is covered by the existing mixed residual tests:
scatter-only full coverage, exact-dense base coverage, online-combine overlap,
missing residual-row init, packed residual rows, and mixed scatter+packed
residuals. The support predicate still rejects duplicate residual rows inside a
single residual kernel and incomplete mixed residual coverage.

Direct-final residual timing used `_run_cached_direct_final_residual_forward`
with support predicate checks, rows=1024, heads=8, D=64, support rows=32:

| case | ms | repeat max diff |
|---|---:|---:|
| scatter-only full coverage | 0.087682 | 0 |
| mixed scatter+packed safe coverage | 0.142681 | 0 |

Online combine still uses an FP32 work buffer followed by a final cast. The final
cast is not safely fused by changing the combine destination to BF16: a
two-combine probe with BF16 as the work buffer changed final BF16 output by
0.015625 max for D64 and 0.0078125 max for D128 versus FP32 work plus final
cast. Final cast cost for rows=1024, heads=8, packed_k=32 was:

| D | combine ms | final cast ms | cast/combine |
|---|---:|---:|---:|
| 64 | 0.084257 | 0.015812 | 0.188 |
| 128 | 0.161468 | 0.019348 | 0.120 |

## D128 Direct Forward

D128 scalar direct remains capped by default to packed_k <= 128 through
`FLASH_ATTN_HSA_CACHED_GATHER_D128_MAX_PACKED_K` because the wider scalar path
outside the TC route was not shown faster. A D128 TC score gather+scatter path
was added for the existing 16x32 union TC route. It keeps the D64 TC structure:
16 query rows, two 16-key K score halves per 32-key chunk, 128-wide shared Q/K/V,
and scalar P*V/output over 128 value columns.

Targeted gather+scatter timing, groups=64, heads=8:

| D | packed_k | output dtype | scalar ms | TC ms | speedup |
|---|---:|---|---:|---:|---:|
| 64 | 32 | fp32 | 0.084152 | 0.071887 | 1.171x |
| 64 | 64 | fp32 | 0.161572 | 0.123696 | 1.306x |
| 64 | 128 | fp32 | 0.315338 | 0.226754 | 1.391x |
| 128 | 32 | fp32 | 0.158314 | 0.131097 | 1.208x |
| 128 | 64 | fp32 | 0.309277 | 0.225217 | 1.373x |
| 128 | 128 | fp32 | 0.609244 | 0.413432 | 1.474x |
| 64 | 32 | bf16 | 0.084099 | 0.075860 | 1.109x |
| 64 | 64 | bf16 | 0.161464 | 0.127145 | 1.270x |
| 64 | 128 | bf16 | 0.315296 | 0.229406 | 1.374x |
| 128 | 32 | bf16 | 0.158461 | 0.137309 | 1.154x |
| 128 | 64 | bf16 | 0.309298 | 0.231500 | 1.336x |
| 128 | 128 | bf16 | 0.609149 | 0.418463 | 1.456x |

## Backward Helpers

Backward remains structurally split into zero helpers, main DQ/DK/DV kernels,
atomic or key-owned DK/DV paths, and final casts. The existing fused zero/cast
helpers are faster for small row counts but slower for long row counts, so the
default auto gate now uses fused helpers only for <=4096 rows. `on` and `off`
env overrides still force behavior.

Helper timing, heads=8:

| rows | D | cast2 sep | cast2 fused | cast3 sep | cast3 fused | zero2 sep | zero2 fused | zero3 sep | zero3 fused |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 64 | 0.033220 | 0.030778 | 0.049686 | 0.043106 | 0.031630 | 0.026685 | 0.047870 | 0.037963 |
| 65536 | 64 | 0.438684 | 0.448818 | 0.658181 | 0.665872 | 0.344756 | 0.368924 | 0.517127 | 0.550675 |
| 4096 | 128 | 0.057452 | 0.057384 | 0.086161 | 0.084005 | 0.049181 | 0.049206 | 0.073760 | 0.071781 |
| 65536 | 128 | 0.862198 | 0.887454 | 1.292986 | 1.315329 | 0.680028 | 0.733525 | 1.020104 | 1.091007 |

## 2026-06-20 Long 2D Row Validation

The explicit/direct/compact 2D payloads now preserve `q_row_idx`, derive
`total_rows` from the max valid row id, validate scatter bounds, and scatter
compact outputs back to true row ids instead of flattening by valid-row order.

Validation run:

- `python -m py_compile flash_attn/cute/hsa_explicit_2d_sparse_analysis.py tests/cute/test_hsa_cached_2d_helpers.py` passed.
- `git diff --check` passed.
- `python -m pytest tests/cute/test_hsa_cached_2d_helpers.py -q` passed: 31 passed, 9 warnings in 9.03s.
- Small explicit smoke, `seqlen=128`, `heads=2`, `D=64`, `packed_q=8`,
  `support_k=32`: direct and compact both returned `(2, 128, 64)`,
  `max_diff=7.152557e-07`, `total_rows=128`.

Low-level long-index probe used sparse row ids above 65536 without constructing
a dense explicit benchmark:

| rows | q row min | q row max | groups | packed_q | packed_k | heads | D | gather ms | scatter ms | combine ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 70048 | 65547 | 70024 | 16 | 8 | 32 | 8 | 64 | 0.046211 | 0.027021 | 0.026381 |

The full long explicit benchmark construction remains a Python-side blocker:
the previously observed 65K-row explicit case construction was approximately
67s. That slow construction path was not rerun here.

## 2026-06-20 Explicit 2D Selective Payload Construction

`tests/cute/benchmark_hsa_2d_sparse.py` now passes the requested variant set
into `build_explicit_2d_sparse_case`, so a `direct_2d_compact`-only benchmark
does not also build the direct, micro/custom, and shared-support payload
families. Long perf-only runs can use `--skip-correctness` to skip the dense
PyTorch oracle and report payload build time explicitly as `build_s`.

Evidence:

- Previously measured all-payload 16K builder: 25.3s. That slow path was not
  rerun.
- 4K compact-only correctness run:
  `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python -u tests/cute/benchmark_hsa_2d_sparse.py --case-family disjoint_confetti --seqlen 4096 --heads 8 --head-dim 64 --packed-q 16 --support-k 64 --islands-per-row 4 --island-width 4 --variants direct_2d_compact --warmup-iters 1 --benchmark-iters 1 --json`
  produced `build_s=3.338`, `direct2d_compact_ms=0.739`,
  `output_max_diff=9.536743e-07`, `output_mean_diff=3.113932e-08`.
- Perf-only compact runs with `--skip-correctness`, `heads=8`, `D=64`,
  `packed_q=16`, `support_k=128`, `islands_per_row=8`, `island_width=4`:

| seq | build_s | direct2d_compact_ms |
|---:|---:|---:|
| 16K | 13.652 | 2.871 |
| 64K | 50.210 | 9.962 |
| 256K | 207.657 | 38.469 |

The selective construction patch removes avoidable unrelated payload and dense
oracle work, but the remaining compact payload construction is still
Python-side and scales roughly linearly with row count. That is the next
explicit-2D long-context blocker.

## Remaining Fusion Gates

Overlapping residual FP32 online-combine cast-out is blocked. The existing
support gate still returns `direct_final_online_combine_requires_fp32_accum`
when overlapping packed/scatter residual rows would require BF16 online
accumulation. The measured BF16-work-buffer probe changed final BF16 output by
0.015625 for D64 and 0.0078125 for D128 versus FP32 work plus final cast, so the
final cast is not safe to fuse by changing the combine destination dtype.

2D backward split/zero/cast helpers are gated. Fused zero/cast helpers remain
under the auto row threshold, defaulting to <=4096 rows through
`FLASH_ATTN_HSA_CACHED_FUSED_GRAD_HELPER_MAX_ROWS`; the timing table above shows
the fused helpers lose at 65536 rows. Direct-DQ backward also stays behind its
tiny/min-row gates, covered by
`test_cached_backward_direct_dq_auto_gate_uses_row_threshold`.

D128 direct forward is gated. The TC route is used only for the tested
`tc16x32`, `packed_q=16`, `tile_k=32`, `support_rows<=128` envelope, and the
scalar D128 direct path remains capped by
`FLASH_ATTN_HSA_CACHED_GATHER_D128_MAX_PACKED_K`. Coverage:
`test_synthetic_2d_masked_fwd_gate_allows_k2048_d64_and_caps_d128`,
`test_synthetic_2d_masked_gather_scatter_tc_d128_matches_scalar`, and
`test_synthetic_2d_masked_gather_d128_matches_dense_partial_mask`.

AR-HSA DK/DV atomics and payload/schedule construction remain gated. The
key-owned DK/DV path requires a ready `cached_tc8x8_fused` backward payload with
`owned_k_row_idx` and all `owned_occurrence_*` tensors, must overwrite all KV
rows, and auto-enables only up to
`FLASH_ATTN_HSA_CACHED_GENERALIZED_BWD_KEY_OWNED_MAX_ROWS` (default 128). The
helper tests cover missing occurrence payloads, all-KV overwrite gating, and
the small-all-owned auto gate. Broader schedule/payload construction is still a
separate Python-side build problem, so no small safe win was taken here.

## 2026-06-20 Explicit 2D TC Small-Support Selector

The explicit 2D benchmark now exposes `direct_2d_tc` and the
`direct_2d_compact` full-passthrough path auto-routes to that tensor-core
gather/scatter kernel for the tested cached-training envelope:

- full-passthrough compact payload with contiguous query rows
- `packed_q=16`
- BF16/FP16 Q/K/V
- `head_dim in {64, 128}`
- `support_rows <= 128`
- disabled with `FLASH_ATTN_HSA_EXPLICIT_DIRECT_2D_TC=off`

Correctness:

- `python -m py_compile flash_attn/cute/hsa_explicit_2d_sparse_analysis.py tests/cute/benchmark_hsa_2d_sparse.py tests/cute/test_hsa.py`
  passed.
- `git diff --check` passed.
- `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. timeout 180s python -m pytest tests/cute/test_hsa.py::test_hsa_explicit_2d_sparse_variants_match_dense_oracle tests/cute/test_hsa.py::test_hsa_explicit_2d_tc_variant_matches_dense_oracle -q`
  passed: 4 passed in 26.23s.
- Compact routed correctness smoke:
  `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. timeout 180s python -u tests/cute/benchmark_hsa_2d_sparse.py --case-family disjoint_confetti --seqlen 4096 --heads 4 --head-dim 64 --packed-q 16 --support-k 4 --islands-per-row 1 --island-width 4 --variants dense,direct_2d_compact,direct_2d_tc --warmup-iters 1 --benchmark-iters 1 --json`
  produced `direct2d_compact_ms=0.151`, `direct2d_tc_ms=0.120`,
  `output_max_diff=9.536743e-07` for both compact and TC.

Cached hot-path timings excluding payload construction:

| seq | H | D | support_k | live keys/query | route | ms |
|---:|---:|---:|---:|---:|---|---:|
| 1M | 4 | 64 | 4 | 4 | compact auto TC | 8.607 |
| 1M | 4 | 64 | 4 | 4 | explicit TC | 8.595 |
| 1M | 4 | 64 | 4 | 4 | compact with TC off | 19.098 |
| 256K | 4 | 128 | 32 | 8 | compact auto TC | 9.016 |
| 256K | 4 | 128 | 32 | 8 | explicit TC | 9.002 |
| 256K | 8 | 64 | 128 | 32 | compact auto TC | 19.046 |
| 256K | 8 | 64 | 128 | 32 | explicit TC | 19.032 |

This fixes the small-support/low-live-key explicit 2D path where scalar compact
was leaving most of the available tensor-core throughput unused. The selector
is intentionally narrow: non-full-span payloads, overlapping residuals, mixed
packed+scatter residuals, and wider/unsupported shapes still use the existing
safe paths and gates.
