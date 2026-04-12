# FP4 PV Exact-Lane Status

## Summary

This document captures the current stable state of the SA3-inspired FP4 PV rewrite in the FlashAttention CuTe path, the relevant files, the commands used to validate and profile it, the experiments that were kept vs reverted, and the remaining work needed to get the fused PV kernel below the recovered `qkfast` baseline.

Current scope of the active FP4 PV path:

- architecture: `SM100/SM110`
- attention shape: dense fixed-length
- mode: noncausal
- topology: `MHA`
- dimensions:
  - dispatch-enabled: `head_dim=head_dim_v in {64, 128}`
  - stable smoke coverage:
    - `d64`: `S=512` and `S=1024`
    - `d128`: `S=512`
- known unstable row:
  - `d128`, `S=1024` still produces `NaN/Inf` in the fused PV path
- public API remains:
  - `fp4_qk_format="nvfp4"`
  - `use_fp4_pv=True`

Everything beyond dense fixed-length noncausal `MHA` in `{64, 128}` remains intentionally out of scope for the active exact lane until the must-win row beats `qkfast` robustly on clean devices.

## Relevant Files

- [flash_attn/cute/fp4_flash_fwd_sm100_pvfused.py](/workspace/codebases/fp4_matmul/flash-attention/flash_attn/cute/fp4_flash_fwd_sm100_pvfused.py)
  - main FP4 PV kernel
  - current exact-lane split-handoff implementation
  - current stable SA3-inspired front half plus optimized exact back half
- [flash_attn/cute/interface.py](/workspace/codebases/fp4_matmul/flash-attention/flash_attn/cute/interface.py)
  - dispatch narrowing for the exact FP4 PV lane
- [tests/cute/benchmark_fp4_pv.py](/workspace/codebases/fp4_matmul/flash-attention/tests/cute/benchmark_fp4_pv.py)
  - fused-only benchmark harness
  - exact profiling mode
  - bogus-fast `qkfast` rejection logic
- [tests/cute/test_fp4_flash_attn.py](/workspace/codebases/fp4_matmul/flash-attention/tests/cute/test_fp4_flash_attn.py)
  - fake-runtime and controller coverage for exact-lane dispatch/shape/handoffs

## Current Stable Line

The branch is currently centered on a stable exact-lane split-handoff implementation rather than the older generic PV path.

Kept behavior in the current exact lane:

- compact exact CTA layout:
  - `4` softmax warps
  - `2` correction warps
  - `1` MMA warp
  - `1` epilogue warp
  - `1` load warp
  - total `288` threads
- split handoff model:
  - `S-ready`: MMA -> softmax
  - `P/SFP/acc_scale-ready`: softmax -> MMA
  - `O-ready`: MMA -> correction/epilogue
- exact lane does not use the legacy stats pipeline in steady state
- exact correction reads final `row_sum` / `row_max` after `pipeline_o_acc`
- producer-side `SFP` publication is kept
- front-half quantizer improvements are kept:
  - direct use of the live post-rowmax score fragment
  - grouped `P` scale computed from `max(e)` directly
  - packed `P` emitted directly in the 8-value group loop
- back-half exact-lane improvements are kept:
  - TMEM `O` rescale setup hoisted out of the hot loop
  - row-quad TMEM rescale slices and fragment views hoisted out of the hot loop
  - exact-lane `S` fragment / packed-`P` FP32 fragment reuse across the softmax loop

## Current Performance Anchor

Current honest must-win-row anchor on the restored stable line:

- row:
  - noncausal `MHA`
  - `d128`
  - `S=512`
  - `batch=2`
- device `2`:
  - `qkfast_ms = 0.11710399761795998`
  - `pv_fused_ms = 0.1406399980187416`
  - `pv_fused_over_qkfast = 1.2009837484589165`

This number is still above `1.0`, so the exact FP4 PV lane is not yet at the required win condition.

Historically during this push sequence, the stable line has also been observed in better states around the `~1.05x-1.09x` band on clean runs, but it has not held below `1.0` robustly across devices `1` and `2`.

## General-Shape Extension

Recent widening work re-enabled the exact fused lane for dense noncausal `d64` `MHA`.

The key runtime fix was not a new math path. The launch heuristic was overestimating how many KV stages fit in shared memory for the exact `d64` lane:

- raw heuristic result: `kv_stage=22`
- actual launchable cap for the current fused layout: `kv_stage=21`
- measured reason:
  - `kv_stage=21` gives `shared_storage.size_in_bytes() = 228352`
  - `kv_stage=22` gives `shared_storage.size_in_bytes() = 238592`
  - the fused kernel targets a `224 KiB = 229376` shared-memory budget

Current short-smoke data after the d64 cap:

- `d64`, `S=512`, device `2`, default path:
  - `pv_fused_over_qkfast = 0.9884854211718744`
- `d64`, `S=512`, device `1`, `FLASH_ATTN_FP4_PV_EXACT_SFV_DIRECT=1`:
  - `pv_fused_over_qkfast = 0.9793912401494516`
- `d64`, `S=1024`, device `2`, default path:
  - `pv_fused_over_qkfast = 0.8320691502132515`

This widening does not make the broader exact lane generally stable yet. In particular, `d128`, `S=1024` still fails with `NaN/Inf` in the fused PV output and needs separate debugging.

Another general-shape issue was fixed after the d64 widening:

- same-process mixed-seqlen exact-lane calls were reusing stale compile-cache entries
- concrete repro before the fix:
  - call exact FP4 PV once at `S=64`
  - call it again at `S=1024` in the same process
  - CuTe raised a shape mismatch on `mK.shape[1]`
- current fix:
  - the fused-lane compile cache key now includes dense `seqlen_q` and `seqlen_k`
  - a runtime probe now verifies exact-lane recompilation across `S=64 -> 1024` for both default and `FLASH_ATTN_FP4_PV_EXACT_SFV_DIRECT=1`

There is still a deeper runtime correctness question in the exact FP4 PV path:

- the long-standing constant-`V` oracle probes do not currently match the expected all-ones output
- the packed `V`/`SFV` fixture itself dequantizes correctly in Python
- that points at the fused runtime path rather than the test encoding
- this has not been fixed in the current push and should be treated as active follow-up work before using those probes as gating correctness checks

## Validation Commands

### 1. Python compile sanity

```bash
cd /workspace/codebases/fp4_matmul/flash-attention
/workspace/codebases/fp4_matmul/.venv/bin/python -m py_compile \
  flash_attn/cute/fp4_flash_fwd_sm100_pvfused.py \
  tests/cute/benchmark_fp4_pv.py \
  tests/cute/test_fp4_flash_attn.py
```

### 2. Focused exact-lane fake-runtime/controller slice

```bash
cd /workspace/codebases/fp4_matmul/flash-attention
PYTHONPATH=/workspace/codebases/fp4_matmul/flash-attention \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/workspace/codebases/fp4_matmul/.venv/bin/python -m pytest -q \
  tests/cute/test_fp4_flash_attn.py \
  -k 'fp4_pv_fused_fake_compile_dense_forward or \
       fp4_pv_fused_exact_lane_uses_compact_warp_map or \
       fp4_pv_fused_exact_lane_uses_split_handoffs or \
       fp4_pv_fused_exact_lane_skips_legacy_stats_pipeline or \
       fp4_use_fp4_pv_dispatch_selects_split_kernel or \
       fp4_pv_benchmark_controller_aggregates_medians_and_failures'
```

Expected recent result:

```text
6 passed, 126 deselected
```

### 3. Fused-only benchmark, must-win row

Device `2`:

```bash
cd /workspace/codebases/fp4_matmul/flash-attention
timeout 120s /workspace/codebases/fp4_matmul/.venv/bin/python \
  tests/cute/benchmark_fp4_pv.py \
  --head-dims 128 \
  --seqlens 512 \
  --causal-values false \
  --batch-size 2 \
  --num-heads 4 \
  --num-heads-kv 4 \
  --device 2 \
  --compare-mode fused-only \
  --emit-json
```

Device `1`:

```bash
cd /workspace/codebases/fp4_matmul/flash-attention
timeout 120s /workspace/codebases/fp4_matmul/.venv/bin/python \
  tests/cute/benchmark_fp4_pv.py \
  --head-dims 128 \
  --seqlens 512 \
  --causal-values false \
  --batch-size 2 \
  --num-heads 4 \
  --num-heads-kv 4 \
  --device 1 \
  --compare-mode fused-only \
  --emit-json
```

### 4. Exact profiling mode

```bash
cd /workspace/codebases/fp4_matmul/flash-attention
timeout 120s env FLASH_ATTN_FP4_PROFILE_EXACT_SKIP_QKFAST=1 \
  /workspace/codebases/fp4_matmul/.venv/bin/python \
  tests/cute/benchmark_fp4_pv.py \
  --compare-mode profile-exact \
  --device 2 \
  --head-dims 128 \
  --seqlens 512 \
  --causal-values false \
  --batch-size 2 \
  --num-heads 4 \
  --num-heads-kv 4
```

Imported report inspection:

```bash
ncu --import /tmp/fp4_pv_exact_d2.ncu-rep --page details
ncu --import /tmp/fp4_pv_exact_d2.ncu-rep --page source --print-source sass --csv
ncu --import /tmp/fp4_qkfast_exact_d2.ncu-rep --page details
```

### 5. Optional GPU cleanup / sanity

```bash
fuser -k -v /dev/nvidia*
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader
```

## Profiling Findings

The biggest useful shift in understanding came from real `ncu` import-based profiling on device `2`.

### Exact fused PV kernel profile highlights

- `No Eligible`: `81.64%`
- `Issued Warp Per Scheduler`: `0.18`
- dominant stall component:
  - roughly `6.3` cycles of long scoreboard on L1TEX-backed accesses
- excessive memory traffic:
  - `L2 Theoretical Sectors Global Excessive = 89088`
  - `L1 Wavefronts Shared Excessive = 15360`

### Comparison to recovered `qkfast`

Recovered `MHAFast` still has global excess, but essentially zero shared excess in the same comparison setup. That makes the unique remaining FP4 PV tax much more likely to be in the shared scale/operand path than in the front-half quantizer.

### Hot source-side instructions from the exact PV import

These hot spots were especially informative:

- `0xfffbede07c00  STS.U8 [R2+UR25+0x35800], R6`
  - about `9216` excessive shared wavefronts
- `0xfffbede072f0  STS.U8 [R4+UR25+0x35800], R8`
  - about `3072` excessive shared wavefronts
- `0xfffbede0d9f0  STS.U8 [R2+0x35400], R8`
  - about `3072` excessive shared wavefronts
- matching global byte loads:
  - `0xfffbede07b50 LDG.E.U8 ...`
    - about `9216` global excessive sectors
  - `0xfffbede07210 LDG.E.U8 ...+0x600`
    - about `3072` global excessive sectors

Working interpretation from this push sequence:

- the `0x35800` cluster corresponds to the public `SFV` scale-byte load/store path in `load_v_scale_stage_public()`
- the `0x35400` cluster corresponds to exact `SFP` constant-byte fill / stage prep

## Kept Changes

These are the changes that have remained on the stable line and are still believed to be directionally correct:

- exact-lane compact CTA map (`288` threads)
- exact-lane split handoff model
- exact-lane skip of the legacy stats pipeline
- exact-lane correction waits on `pipeline_o_acc` rather than the old stats barrier
- SA3-inspired fused front-half quantizer:
  - live post-rowmax score fragment
  - direct `max(e)/6` grouped scale
  - direct packed-`P` emission
- producer-side `SFP` publication
- exact-lane TMEM `O` rescale setup hoists
- exact row-quad TMEM rescale slice hoists
- exact-lane fragment reuse inside the softmax loop
- benchmark harness upgrades:
  - fused-only exact benchmarking
  - bogus-fast baseline rejection
  - deterministic exact profiling mode
  - `FLASH_ATTN_FP4_PROFILE_EXACT_SKIP_QKFAST`
- exact-lane fake-runtime tests:
  - compact warp map
  - split handoffs
  - no legacy stats pipeline

## Reverted or Failed Experiments

The following ideas were tried and intentionally reverted because they regressed, hung, or failed legalization:

- exact local-`O` SA3-style recurrence using local FP32 storage
  - finite but much slower, around `~2.3x`
- producer-side `SFV` shared->TMEM move
  - hangs or dies on real devices
- separate exact `acc_scale` handoff
  - hangs
- exact `q_stage=1` dedicated MMA path
  - slower than the stable line
- exact single correction warp
  - slower
- exact two-softmax-warp lane
  - slower
- direct / CTA-direct `V` loader experiments
  - slower
- forcing `kv_stage=1` or `2`
  - worse
- `FLASH_ATTN_FP4_PV_CORR_TILE_SIZE=32`
  - not robust; worse or numerically bad
- `FLASH_ATTN_FP4_PV_CORR_TILE_SIZE=64`
  - register allocation failure
- vectorized `SFV` copies:
  - `tiled_copy_1d(..., num_copy_elems=2/4)` failed IR/legalization for this scale layout
- manual `Uint32` recast scale copies
  - slower or numerically bad
- exact `SFP` tail-only initialization
  - regressed or broke numerics
- compact exact `SFP` source alias
  - `cute_nvgpu.make_umma_smem_desc` legalization failure
- moving the exact-lane `S` release earlier
  - row stopped returning / hung
- non-blocking TMEM allocation handshake experiments
  - regressions
- named-barrier replacements in the back half
  - hangs or regressions

## What Still Needs Fixing

The remaining blocker is not the front-half SA3-style softmax/quantizer path anymore.

The remaining blocker is the exact back-half operand and memory path, especially:

- exact `SFV` staging and public scale-byte movement
- residual exact `SFP` scale-byte preparation cost
- TMEM `O` rescale traffic and access efficiency
- the exact-lane shared/global byte traffic pattern that differs from recovered `qkfast`

Right now the best evidence says:

- front-half math is close enough
- the remaining tax is memory-side and access-shape-side
- direct copy-width tricks are not viable on this layout/backend
- the next real win will likely require a structural exact-lane `SFV` staging redesign rather than another small scheduling tweak

## Recommended Next Steps

1. Keep the current stable split-handoff exact lane intact as the baseline.
2. Use the existing exact profiling mode as the authority before the next structural change.
3. Target exact `SFV` staging shape directly:
   - reduce or redesign the current public scale-byte path
   - avoid direct reuse of the current generic swizzled shared-scale contract if possible
   - preserve blockscaled legality
4. Re-profile after each structural `SFV` change using the exact profiling mode and imported `ncu` reports.
5. Keep `d64` dense `MHA` enabled, but do not widen further to `GQA` or causal until the `d128`, `S=512` must-win row is robustly `< 1.0` on devices `1` and `2`, and `d128`, `S=1024` no longer produces `NaN/Inf`.

## New Experimental Knob

A new exact-lane-only experiment is now available behind:

- `FLASH_ATTN_FP4_PV_EXACT_SFV_DIRECT=1`

What it does:

- keeps the current exact-lane `TMA V` path intact
- bypasses the generic public `SFV` staging copy in `load_v_scale_stage_public()`
- uses the manual public-SFV byte loader directly into the exact-lane shared `SFV` stage
- keeps the default stable path unchanged when the env var is unset
- is compile-cache separated from the default path

Validation coverage added for this knob:

- exact-lane fake-runtime property coverage
- compile-cache separation coverage for the exact fused lane

Short smoke results from the first shortened fused-only pass (`--warmup 1 --iters 1`):

- device `2`: `pv_fused_over_qkfast = 0.9918`
- device `1`: `pv_fused_over_qkfast = 1.0298`

These numbers are directionally positive, but they are not yet acceptance-quality because they do not use the full multi-run median workflow.

Recommended measurement workflow for the new experiment:

```bash
cd /workspace/codebases/fp4_matmul/flash-attention
FLASH_ATTN_FP4_PV_EXACT_SFV_DIRECT=1   timeout 120s env FLASH_ATTN_FP4_PROFILE_EXACT_SKIP_QKFAST=1   /workspace/codebases/fp4_matmul/.venv/bin/python   tests/cute/benchmark_fp4_pv.py   --compare-mode profile-exact   --device 2   --head-dims 128   --seqlens 512   --causal-values false   --batch-size 2   --num-heads 4   --num-heads-kv 4
```

## Acceptance Target That Is Still Outstanding

Must-win row:

- noncausal `MHA`
- `d128`
- `S=512`
- `batch=2`

Required:

- device `1`: median `pv_fused_ms / qkfast_ms < 1.0`
- device `2`: median `pv_fused_ms / qkfast_ms < 1.0`
- first launch finite
- repeated launch finite

That target has not yet been met.
