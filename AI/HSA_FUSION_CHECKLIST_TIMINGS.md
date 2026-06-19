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
