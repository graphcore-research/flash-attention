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

## 2026-06-20 Explicit 2D TC High-Support D64 Selector

The explicit compact TC selector was too conservative for D64 full-passthrough
compact payloads. The wrapper can already run the TC gather/scatter kernel at
larger support widths, and bounded probes showed that support 512/1024 are real
wins while support 256 is not. The default selector now routes D64
`direct_2d_compact` through TC when `512 <= support_rows <= 1024`. The upper
bound is configurable with
`FLASH_ATTN_HSA_EXPLICIT_DIRECT_2D_TC_HIGH_SUPPORT_MAX`; D128 remains on the
existing small-support gate.

Correctness and timing probes:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. timeout 180s python -u tests/cute/benchmark_hsa_2d_sparse.py \
  --case-family disjoint_confetti --seqlen 2048 --heads 4 --head-dim 64 \
  --packed-q 16 --support-k 256 --islands-per-row 16 --island-width 4 \
  --variants dense,direct_2d_compact,direct_2d_tc --warmup-iters 1 --benchmark-iters 2 --json

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. timeout 180s python -u tests/cute/benchmark_hsa_2d_sparse.py \
  --case-family disjoint_confetti --seqlen 16384 --heads 4 --head-dim 64 \
  --packed-q 16 --support-k 256 --islands-per-row 16 --island-width 4 \
  --variants direct_2d_compact,direct_2d_tc --warmup-iters 1 --benchmark-iters 2 \
  --skip-correctness --json

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. timeout 240s python -u tests/cute/benchmark_hsa_2d_sparse.py \
  --case-family disjoint_confetti --seqlen 65536 --heads 4 --head-dim 64 \
  --packed-q 16 --support-k 512 --islands-per-row 32 --island-width 4 \
  --variants direct_2d_compact,direct_2d_tc --warmup-iters 1 --benchmark-iters 2 \
  --skip-correctness --json

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. timeout 240s python -u tests/cute/benchmark_hsa_2d_sparse.py \
  --case-family disjoint_confetti --seqlen 16384 --heads 4 --head-dim 64 \
  --packed-q 16 --support-k 1024 --islands-per-row 64 --island-width 4 \
  --variants direct_2d_compact,direct_2d_tc --warmup-iters 1 --benchmark-iters 2 \
  --skip-correctness --json
```

| seq | D | support_k | route state | compact/scalar ms | forced/default TC ms | decision |
|---:|---:|---:|---|---:|---:|---|
| 2K | 64 | 256 | pre-patch probe | 0.614 | 0.330 | TC wins only at tiny shape |
| 16K | 64 | 256 | pre-patch probe | 2.522 | 3.282 | keep scalar |
| 64K | 64 | 512 | pre-patch forced | 18.320 | 9.307 | default TC |
| 16K | 64 | 1024 | pre-patch forced | 9.331 | 5.403 | default TC |
| 16K | 64 | 512 | post-patch compact | old scalar 6.355 | 2.742 | default TC active |
| 16K | 64 | 1024 | post-patch compact | old scalar 9.331 | 5.400 | default TC active |

D128 support512 was also probed with correctness enabled:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. timeout 180s python -u tests/cute/benchmark_hsa_2d_sparse.py \
  --case-family disjoint_confetti --seqlen 1024 --heads 4 --head-dim 128 \
  --packed-q 16 --support-k 512 --islands-per-row 32 --island-width 4 \
  --variants dense,direct_2d_compact,direct_2d_tc --warmup-iters 1 --benchmark-iters 2 --json
```

The D128 TC kernel was correct (`max_diff=8.34e-07`) but slower than dense at
that shape (`1.031 ms` vs `0.491 ms`), and scalar compact is still unsupported
by the D128 packed-k cap. D128 support512 therefore remains gated off by
default rather than being claimed as a win.

## 2026-06-20 Online-Combine Cast-Out

The overlapping-residual direct-final path keeps FP32 online softmax combine
semantics unchanged, then casts the complete FP32 output buffer to the model
dtype. The old cast used the indexed CuTe row-cast kernel even when the row set
was exactly `0..total_rows-1`. A contiguous CuTe cast kernel was added for
diagnostics, but the fastest correct path is PyTorch's contiguous
`out_final_flat.copy_(out_work_flat)`, so `FLASH_ATTN_HSA_CACHED_DIRECT_FINAL_CONTIG_CAST=auto`
defaults to the PyTorch contiguous copy. `cute` forces the new contiguous CuTe
kernel and `off`/`indexed` forces the old indexed CuTe kernel.

Cast microbenchmark command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. timeout 240s python - <<'PY'
import torch
from flash_attn.cute.flash_hsa_synthetic_grid_sm100 import _run_cached_cast_all_rows_kernel, _run_cached_cast_rows_kernel

def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

for rows in (1024, 4096, 65536, 262144):
    for d in (64, 128):
        h = 8
        src = torch.randn((rows, h, d), device="cuda", dtype=torch.float32)
        dst_indexed = torch.empty((rows, h, d), device="cuda", dtype=torch.bfloat16)
        dst_all = torch.empty_like(dst_indexed)
        dst_torch = torch.empty_like(dst_indexed)
        row_idx = torch.arange(rows, device="cuda", dtype=torch.int32)
        _run_cached_cast_rows_kernel(src, row_idx, dst_indexed)
        _run_cached_cast_all_rows_kernel(src, dst_all)
        dst_torch.copy_(src)
        torch.cuda.synchronize()
        print(rows, d, bench(lambda: _run_cached_cast_rows_kernel(src, row_idx, dst_indexed)), bench(lambda: _run_cached_cast_all_rows_kernel(src, dst_all)), bench(lambda: dst_torch.copy_(src)))
PY
```

| rows | H | D | indexed CuTe ms | contiguous CuTe ms | PyTorch contiguous copy ms | contig CuTe / indexed |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 8 | 64 | 0.020232 | 0.019469 | 0.004235 | 1.039x |
| 1024 | 8 | 128 | 0.019303 | 0.018573 | 0.004225 | 1.039x |
| 4096 | 8 | 64 | 0.021951 | 0.017553 | 0.004260 | 1.251x |
| 4096 | 8 | 128 | 0.018764 | 0.017630 | 0.006255 | 1.064x |
| 65536 | 8 | 64 | 0.154131 | 0.117412 | 0.030631 | 1.313x |
| 65536 | 8 | 128 | 0.299721 | 0.231646 | 0.059352 | 1.294x |
| 262144 | 8 | 64 | 0.602175 | 0.457474 | 0.113820 | 1.316x |
| 262144 | 8 | 128 | 1.183752 | 0.910181 | 0.225020 | 1.301x |

Validation:

- `python -m py_compile flash_attn/cute/flash_hsa_synthetic_grid_sm100.py flash_attn/cute/hsa_cached_2d_forward_analysis.py tests/cute/test_hsa_cached_2d_helpers.py`
  passed.
- `git diff --check` passed.
- `PYTHONPATH=. timeout 120s python -m pytest tests/cute/test_hsa_cached_2d_helpers.py::test_cached_2d_direct_final_online_combine_cast_mode_gate -q`
  passed.
- Existing backward gates were rechecked with:
  `PYTHONPATH=. timeout 120s python -m pytest tests/cute/test_hsa_cached_2d_helpers.py::test_cached_fused_grad_helper_auto_gate_uses_row_threshold tests/cute/test_hsa_cached_2d_helpers.py::test_cached_backward_key_owned_dkdv_gate_requires_occurrence_payload tests/cute/test_hsa_cached_2d_helpers.py::test_cached_backward_key_owned_overwrite_gate_requires_all_kv_rows tests/cute/test_hsa_cached_2d_helpers.py::test_cached_backward_key_owned_auto_gate_is_small_all_owned_only tests/cute/test_hsa_cached_2d_helpers.py::test_cached_backward_direct_dq_auto_gate_uses_row_threshold -q`
  and all 5 passed.

## 2026-06-20 Mixed Residual Coverage and Backward Helpers

Mixed packed+scatter / non-full-span residual direct-final was inspected again.
The current support predicate already allows direct-final when the base rows and
all residual row groups have a full `total_rows` union, including serial overlap
across scatter and packed groups. It initializes residual-only missing rows and
uses FP32 online combine when rows overlap across base/residual or across
residual families.

Still blocked:

- Rows missing from the base+residual union cannot be direct-finalized safely.
  The split fallback initializes every row, computes whatever base/residual work
  exists, then runs `_run_cached_finalize_output_rows_kernel` over all rows.
  In direct-final mode there is no computed output/LSE contribution for missing
  rows; initializing them would create zero/invalid rows rather than attention
  results. The existing tests keep this blocked:
  `mixed_residual_incomplete_direct_final_row_coverage` and
  `packed_residual_incomplete_direct_final_row_coverage`.
- Duplicate rows inside one residual kernel remain blocked by
  `_direct_final_has_duplicate_residual_rows_within_kernel`. The current combine
  kernels support serial online-softmax combination across separate base /
  scatter / packed launches, but not unordered duplicate updates within a
  single residual kernel launch. Covered by
  `test_cached_2d_direct_final_blocks_duplicate_rows_inside_residual_kernel`.

Backward helper dispatch was then measured because all-row zero/finalize helper
paths still used indexed CuTe row kernels. PyTorch contiguous `copy_` / `zero_`
is faster for all tested row counts, so
`FLASH_ATTN_HSA_CACHED_TORCH_CONTIG_GRAD_HELPERS=on` is now the default. Set it
to `off` to force the previous CuTe helper behavior.

Command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. timeout 240s python - <<'PY'
import torch
from flash_attn.cute.flash_hsa_synthetic_grid_sm100 import (
    _run_cached_cast_rows_kernel, _run_cached_cast_two_rows_kernel,
    _run_cached_cast_three_rows_kernel, _run_cached_zero_rows_kernel,
    _run_cached_zero_two_rows_kernel, _run_cached_zero_three_rows_kernel,
)

def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

for rows in (4096, 65536, 262144):
    for d in (64, 128):
        h = 8
        row_idx = torch.arange(rows, device="cuda", dtype=torch.int32)
        a = torch.randn((rows, h, d), device="cuda", dtype=torch.float32)
        b = torch.randn_like(a)
        c = torch.randn_like(a)
        da = torch.empty((rows, h, d), device="cuda", dtype=torch.bfloat16)
        db = torch.empty_like(da)
        dc = torch.empty_like(da)
        print(rows, d, bench(lambda: _run_cached_cast_three_rows_kernel(a,b,c,row_idx,da,db,dc)), bench(lambda: (da.copy_(a), db.copy_(b), dc.copy_(c))), bench(lambda: _run_cached_zero_three_rows_kernel(row_idx,a,b,c)), bench(lambda: (a.zero_(), b.zero_(), c.zero_())))
PY
```

Representative timings:

| rows | D | CuTe cast3 ms | torch copy3 ms | CuTe zero3 ms | torch zero3 ms |
|---:|---:|---:|---:|---:|---:|
| 4096 | 64 | 0.025814 | 0.012474 | 0.022774 | 0.016920 |
| 4096 | 128 | 0.049537 | 0.018472 | 0.041364 | 0.017144 |
| 65536 | 64 | 0.450636 | 0.093801 | 0.331100 | 0.061646 |
| 65536 | 128 | 0.864057 | 0.177412 | 0.640476 | 0.116822 |
| 262144 | 64 | 1.786637 | 0.343328 | 1.311463 | 0.217341 |
| 262144 | 128 | 3.441538 | 0.674292 | 2.551896 | 0.424623 |

Validation:

- `python -m py_compile flash_attn/cute/hsa_cached_2d_forward_analysis.py tests/cute/test_hsa_cached_2d_helpers.py`
  passed.
- `git diff --check` passed.
- `PYTHONPATH=. timeout 120s python -m pytest tests/cute/test_hsa_cached_2d_helpers.py::test_cached_torch_contiguous_grad_helper_env_gate tests/cute/test_hsa_cached_2d_helpers.py::test_cached_fused_grad_helper_auto_gate_uses_row_threshold tests/cute/test_hsa_cached_2d_helpers.py::test_cached_backward_key_owned_dkdv_gate_requires_occurrence_payload tests/cute/test_hsa_cached_2d_helpers.py::test_cached_backward_key_owned_overwrite_gate_requires_all_kv_rows tests/cute/test_hsa_cached_2d_helpers.py::test_cached_backward_key_owned_auto_gate_is_small_all_owned_only tests/cute/test_hsa_cached_2d_helpers.py::test_cached_backward_direct_dq_auto_gate_uses_row_threshold -q`
  passed: 6 passed in 2.26s.

## 2026-06-20 AR-HSA Readout Backward and Atomics Headroom

The AR-HSA walk benchmark confirms that the 1M fwd+bwd preallocated hot path is
readout-backward dominated, not walk/softmax dominated. This pass rechecked the
current query-warp fused backward, the auto tensor-core QV selector, forced QV
tensor-core packing, and forward-denominator reuse.

Commands were run with `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.` from this
directory, with Torch/custom comparisons disabled:

```bash
timeout 240s python -u tests/cute/benchmark_arhsa_walk.py \
  --n-queries 262144 --n-heads 4 --head-dim-v 64 --leaves-per-query 4 \
  --n-iters 3 --graph-mode level_dag --level-range-kernels --incoming-packed-step \
  --query-warp-readout --query-warp-fused-bwd \
  --skip-torch --skip-custom-fwd-bwd --no-check --no-memory --iters 3 --warmup 1

timeout 240s python -u tests/cute/benchmark_arhsa_walk.py \
  --n-queries 1048576 --n-heads 4 --head-dim-v 64 --leaves-per-query 4 \
  --n-iters 3 --graph-mode level_dag --level-range-kernels --incoming-packed-step \
  --query-warp-readout --query-warp-fused-bwd \
  --skip-torch --skip-custom-fwd-bwd --no-check --no-memory --iters 3 --warmup 1
```

| queries | mode | readout_fwd ms | readout_bwd ms | full_hot_fwd ms | fwd+bwd_prealloc ms |
|---:|---|---:|---:|---:|---:|
| 262144 | query-warp fused bwd | 0.2994 | 0.5876 | 0.3140 | 0.9265 |
| 1048576 | query-warp fused bwd | 0.8274 | 1.8355 | 0.8997 | 2.7249 |

At 1M queries, readout backward is about 67% of the preallocated fwd+bwd hot
path. The walk/softmax side is already small by comparison.

The auto tensor-core QV path was rechecked on the 262K random-leaf layout:

```bash
timeout 240s python -u tests/cute/benchmark_arhsa_walk.py \
  --n-queries 262144 --n-heads 4 --head-dim-v 64 --leaves-per-query 4 \
  --n-iters 3 --graph-mode level_dag --level-range-kernels --incoming-packed-step \
  --query-warp-readout --auto-readout-bwd \
  --skip-torch --skip-custom-fwd-bwd --no-check --no-memory --iters 3 --warmup 1

timeout 240s python -u tests/cute/benchmark_arhsa_walk.py \
  --n-queries 262144 --n-heads 4 --head-dim-v 64 --leaves-per-query 4 \
  --n-iters 3 --graph-mode level_dag --level-range-kernels --incoming-packed-step \
  --query-warp-readout --auto-readout-bwd --auto-qv-output-util-threshold 0 \
  --skip-torch --skip-custom-fwd-bwd --no-check --no-memory --iters 3 --warmup 1
```

| queries | mode | selected bwd | QV output util | readout_bwd ms | fwd+bwd_prealloc ms |
|---:|---|---|---:|---:|---:|
| 262144 | auto | query_warp_fused | 0.0639 | 0.5604 | 0.9437 |
| 262144 | forced TC QV | tensor_core_query_value_packed_pack_scatter | 0.0639 | 2.5344 | 2.8499 |

The tensor-core QV route remains correctly gated off for this random sparse
layout. It only fills about 6.4% of the output tile space, and forcing it is
roughly 4.3x slower for readout backward than query-warp fused.

Forward-denominator reuse is a small positive hot-path optimization but not a
new default switch here. The benchmark's isolated `readout_bwd_cute_ms` includes
an extra forward readout when `--reuse-forward-denom` is set, so the fair signal
is `fwd_bwd_cute_prealloc_ms`.

```bash
timeout 240s python -u tests/cute/benchmark_arhsa_walk.py \
  --n-queries 262144 --n-heads 4 --head-dim-v 64 --leaves-per-query 4 \
  --n-iters 3 --graph-mode level_dag --level-range-kernels --incoming-packed-step \
  --query-warp-readout --query-warp-fused-bwd --reuse-forward-denom \
  --skip-torch --skip-custom-fwd-bwd --no-check --no-memory --iters 3 --warmup 1

timeout 240s python -u tests/cute/benchmark_arhsa_walk.py \
  --n-queries 1048576 --n-heads 4 --head-dim-v 64 --leaves-per-query 4 \
  --n-iters 3 --graph-mode level_dag --level-range-kernels --incoming-packed-step \
  --query-warp-readout --query-warp-fused-bwd --reuse-forward-denom \
  --skip-torch --skip-custom-fwd-bwd --no-check --no-memory --iters 3 --warmup 1
```

| queries | baseline fwd+bwd_prealloc ms | reuse-denom fwd+bwd_prealloc ms | delta |
|---:|---:|---:|---:|
| 262144 | 0.9265 | 0.8980 | 3.1% faster |
| 1048576 | 2.7249 | 2.6857 | 1.4% faster |

The denominator-precomputed path is already implemented in
`run_arhsa_leaf_readout_backward(..., denom_precomputed=True)` and covered by
the query-warp fused backward test. It should be used when the caller already
retains the forward denominator; forcing it globally would require retaining an
extra forward buffer in every production path for a 1-3% hot-path win.

Cached generalized backward DK/DV atomics were inspected again. The default
route keeps tile-atomic DK/DV in the main cached backward kernel. The key-owned
non-atomic route is only safe when the backward payload has ready owned
occurrence tensors and `owned_k_row_idx.numel() == k_flat.shape[0]`; otherwise
DK/DV rows can be stale or raced. Auto mode also limits this to at most 128 KV
rows. Those gates live in `_can_use_cached_backward_key_owned_dkdv`,
`_cached_backward_key_owned_overwrites_all_kv_rows`, and
`_auto_use_cached_backward_key_owned_dkdv`.

Status:

- Fixed/default: no new code change in this pass.
- Gated with evidence: tensor-core QV readout backward stays gated by output
  utilization; forced path is slower on the target random-leaf layout.
- Gated with evidence: denom reuse remains an explicit path; it is correct and
  mildly faster only when the forward denom is already retained.
- Blocked with code evidence: broad non-atomic DK/DV cannot replace tile
  atomics unless the payload owns and overwrites every KV row. Partial ownership
  needs zero/finalization or atomics to avoid stale DK/DV rows.
