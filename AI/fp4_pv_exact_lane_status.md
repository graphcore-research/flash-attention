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
    - `d128`: `S=512` and `S=1024`
- current caveat:
  - correctness on the dense noncausal `MHA` smoke rows is currently clean, but the widened `d128` lane still needs performance work
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

Another broader-shape correctness bug is now fixed in the exact correction tail:

- before the fix:
  - the compact exact correction lane only published `LSE` for the first `64` rows of each `128`-row tile
  - practical symptom:
    - `d64`, `S=512`: `pv_fused_lse_max = 6.0015`
    - `d64`, `S=1024`: `pv_fused_lse_max = 6.7229`
    - the bad rows were structured: `64..127`, `192..255`, ...
- kept fix:
  - exact correction now performs the second `64`-row `LSE` publication pass for `m_block_size=128`
- current result:
  - `d64`, `S=512`: `pv_fused_lse_max = 0.9424`
  - `d64`, `S=1024`: `pv_fused_lse_max = 0.9222`
  - `d128`, `S=512`: `pv_fused_lse_max = 0.9416`

The remaining `d128` correctness bug is also fixed now:

- root cause:
  - the compact exact lane used only `2` correction warps for both `d64` and `d128`
  - `d64` only exercises the first `64` output channels, but `d128` needs the wider correction/store footprint
  - practical symptom on `d128`, `S=1024`:
    - deterministic rows with `32` `NaN`s in the second half-channel block, while `LSE` stayed finite
- kept fix:
  - keep the compact `2`-warp correction map only for exact `d64`
  - exact `d128` now uses a `4`-warp correction map
- current result:
  - repeated direct `d128`, `S=1024` probes are finite
  - benchmark smokes now report bounded error on all four dense noncausal `MHA` rows:
    - `d64`, `S=512`: `pv_fused_out_max = 0.0786`, `pv_fused_lse_max = 0.9424`
    - `d64`, `S=1024`: `pv_fused_out_max = 0.0366`, `pv_fused_lse_max = 0.9222`
    - `d128`, `S=512`: `pv_fused_out_max = 0.0594`, `pv_fused_lse_max = 0.9416`
    - `d128`, `S=1024`: `pv_fused_out_max = 0.0361`, `pv_fused_lse_max = 0.9192`

One small performance tuning pass was also kept for the widened exact `d128` lane:

- tuning:
  - keep exact `d128` correction warps at `4` for correctness
  - lower the exact `d128` correction register target from the generic `80` to `72`
- measured effect on a standalone `d128`, `S=1024`, fused-only smoke:
  - `FLASH_ATTN_FP4_FORCE_REGS_CORRECTION=80`: `pv_fused_ms = 0.3761`
  - `FLASH_ATTN_FP4_FORCE_REGS_CORRECTION=72`: `pv_fused_ms = 0.3559`
- this is only a modest gain, but it is directionally correct and does not change the correctness boundary

Another general-shape issue was fixed after the d64 widening:

- same-process mixed-seqlen exact-lane calls were reusing stale compile-cache entries
- concrete repro before the fix:
  - call exact FP4 PV once at `S=64`
  - call it again at `S=1024` in the same process
  - CuTe raised a shape mismatch on `mK.shape[1]`
- current fix:
  - the fused-lane compile cache key now includes dense `seqlen_q` and `seqlen_k`
  - the shipped default exact lane now has a runtime probe that verifies recompilation across `S=64 -> 1024`
  - the legacy `FLASH_ATTN_FP4_PV_EXACT_SFV_DIRECT=0` opt-out is still kept for debugging / A-B comparison, but it is no longer treated as a supported runtime path

There is still a deeper runtime correctness question in the exact FP4 PV path, but the state is better than the last pushed baseline:

- the long-standing noncausal constant-`V` exact-lane probe now passes again after fixing the exact softmax `SFP` writer
  - the kept fix was:
    - reduce grouped pre-exp amax by the actual logical `(row, col)` slot pointer
    - add the missing `32`-row-per-softmax-warp offset in the exact lane before writing `sSFP`
  - practical result:
    - `test_fp4_pv_probe_constant_v_populates_all_output_channels` now passes on the exact fused lane
- the exact-lane `SFV` side is materially better now
  - `tile_atom_to_shape_sfv_vt(...)` now follows the K-major public `SFVt` swizzle
  - the exact direct `SFV` loader now writes every `d` lane inside each `16`-wide scale group
  - the exact fused lane defaults `FLASH_ATTN_FP4_PV_EXACT_SFV_DIRECT` to on
  - practical result:
    - `test_fp4_pv_probe_v_scale_axis_is_colwise` now passes on the shipped exact path
- CUTLASS / TVM / quack compatibility needed two local shims on this stack before the runtime probes were meaningful again
  - float8 scale tensors are compiled and launched through `uint8` storage views, with CUTLASS element types set explicitly
  - newer MLIR `SwizzleType` objects are patched back to the legacy `num_bits / num_base / num_shift` interface expected by existing descriptor and copy helpers
- the packed `V` / public `SFVt` fixture still dequantizes correctly in Python
  - that remains a useful guardrail while widening beyond the current validated rows
- current follow-up target:
  - keep optimizing the default exact lane on the now-correct broader dense noncausal rows
  - focus on performance, especially `d128`, `S=1024`, which is still materially slower than the recovered `qkfast` baseline
  - the legacy `FLASH_ATTN_FP4_PV_EXACT_SFV_DIRECT=0` path still goes non-finite on the `S=64 -> 1024` runtime probe and should be treated as diagnostic-only until fixed or removed

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

## Latest Exact `SFV` Loader Tweak

A small exact-lane `SFV` staging change is now in tree:

- `load_v_scale_stage_public_direct()` skips the identity `1.0` fill on full `128`-row tiles
- dense `S=512/1024` noncausal rows hit that full-tile fast path on every block
- partial-tail tiles still keep the old identity-fill behavior

Targeted correctness validation:

- direct constant-`V` probe still passes
  - `out_max_err = 0.0547`
  - `LSE` remained finite

The timing situation is also clearer now:

- the benchmark harness now batches repeated launches per timed iteration
- standard `full` / `fused-only` timing now uses host-synchronized batched timing
- single-launch CUDA-event timing is kept only for the exact-profile mode

Why this was needed:

- single-launch CUDA-event timing was under-reporting short kernels on some devices
- concrete example on device `1`, must-win row `d128, S=512`:
  - event timing claimed `qkfast ~= 0.0795 ms`
  - batched host timing measured `qkfast ~= 0.1472 ms`
  - event timing also badly under-reported `bf16` on that device

Current must-win-row status with the batched host timer:

- row:
  - dense noncausal `MHA`
  - `d128`
  - `S=512`
  - batch `2`
- device `2`:
  - `qkfast_ms ~= 0.1182`
  - `pv_fused_ms ~= 0.1463`
  - ratio `~= 1.238`
- device `1`:
  - `qkfast_ms ~= 0.1316`
  - `pv_fused_ms ~= 0.2124`
  - ratio `~= 1.613`

So the current status is:

- correctness for the shipped exact `SFV` fast path looks clean
- the timing is trustworthy enough again to guide optimization
- device `2` is materially closer to target
- device `1` is still substantially behind

## Exact vs `qkfast` Profile Snapshot

One `ncu` comparison on device `2`, must-win row (`d128`, `S=512`, noncausal, batch `2`) is now available.

Exact fused PV kernel:

- duration: `117.5 us`
- block size: `352`
- registers/thread: `168`
- dynamic shared memory/block: `229.4 KB`
- local spilling requests: `52,153`
- theoretical occupancy: `17.2%`
- achieved occupancy: `17.1%`

Recovered `qkfast` baseline:

- duration: `19.3 us`
- block size: `512`
- registers/thread: `128`
- dynamic shared memory/block: `216.1 KB`
- local spilling requests: `0`
- theoretical occupancy: `25.0%`
- achieved occupancy: `17.2%`

What this changes:

- the exact lane does not look bandwidth-bound
- the clearest delta versus `qkfast` is spill pressure plus a heavier per-thread register footprint
- shared memory is also slightly higher on the exact lane, but the spill difference is much more dramatic
- the next promising knob is register pressure in the exact softmax/back-half path, not another blind `SFV` byte-remap experiment

## Latest Exact Softmax Reduction Change

One exact-lane kernel-side reduction change is now kept in
`online_softmax_with_quant_pv_exact()`:

- the exact lane no longer uses the fully segmented `32`-lane slot-pointer
  reduction on every group
- dense unmasked rows now take the same butterfly reduction path used by the
  stable fast lane
- the exact segmented reduction is still used when masked peers actually share
  a logical `SFP` slot
- a small follow-up cleanup also hoisted the `slot_ptr` calculation into the
  masked-only branch so the dense row does not materialize that `Int64` hot-path
  value unnecessarily

Why this was worth trying:

- the exact must-win row is spill-heavy and register-heavy versus `qkfast`
- the old exact path paid the segmented slot-pointer reduction even on the
  dense noncausal row where it is not needed

Current validation on the kept hybrid reduction:

- direct noncausal constant-`V` probe stayed clean
  - `out_max_err = 0.0547`
  - `LSE` remained finite
- direct dense noncausal smoke remained bounded on all four validated rows
  - `d64`, `S=512`: `pv_fused_out_max = 0.0786`, `pv_fused_lse_max = 0.9424`
  - `d64`, `S=1024`: `pv_fused_out_max = 0.0533`, `pv_fused_lse_max = 0.9222`
  - `d128`, `S=512`: `pv_fused_out_max = 0.0594`, `pv_fused_lse_max = 0.9416`
  - `d128`, `S=1024`: `pv_fused_out_max = 0.0361`, `pv_fused_lse_max = 0.9192`

Current must-win-row timing snapshot after this change:

- device `1`:
  - `qkfast_ms ~= 0.4849`
  - `pv_fused_ms ~= 0.5685`
  - ratio `~= 1.173`
- device `2`:
  - `qkfast_ms ~= 0.1419`
  - `pv_fused_ms ~= 0.1700`
  - ratio `~= 1.198`

Interpretation:

- this is a real improvement versus the earlier hybrid exact baseline from this
  note (`~1.238` on device `2`, `~1.613` on device `1`)
- it is still not close enough to the `< 1.0` target
- the remaining work is still structural register/spill reduction inside the
  exact softmax / back-half path

Recent non-keeper from this sub-pass:

- narrowing the dense fast path all the way down to only the `offset=1`
  butterfly partner
  - correctness stayed clean
  - device `1` was essentially flat, but device `2` regressed back to
    `~1.241`
  - conclusion:
    - reverted
- raising the exact `d128` default `num_regs_other` target from `48` to `64`
  - fresh-process medians were noisy and not consistent enough to keep as a
    default
  - the best samples moved in the right direction, but later repeats did not
    reproduce a clean cross-device win
  - conclusion:
    - keep it as a benchmark-only candidate, not a baked default
- raising the exact softmax register target from `192` to `208`
  - device `1` improved only marginally
  - device `2` regressed
  - conclusion:
    - reverted / not kept
- splitting `online_softmax_with_quant_pv_exact()` into separate
  reduction/publication and pack passes
  - the exact math stayed conceptually unchanged
  - compile time increased sharply enough that the experiment was not practical
    to iterate on here
  - conclusion:
    - reverted before taking it further
- reusing the stable flatview quantizer on a row-offset exact `sSFP` subview for
  the dense unmasked row
  - first single-run timings looked promising
  - fresh-process medians regressed on both devices
    - device `1`: `~1.215`
    - device `2`: `~1.272`
  - output / `LSE` errors also drifted away from the current exact baseline
    (`pv_fused_out_max ~= 0.155`, `pv_fused_lse_max ~= 0.294`)
  - conclusion:
    - reverted
- collapsing the dense butterfly fast-path `(row, col)` pair into one logical
  slot key
  - correctness stayed clean
  - single-run timing on device `1` regressed back to the `~1.214` range
  - conclusion:
    - reverted

Recent non-keepers from the latest pass:

- turning the exact grouped-slot loop into a runtime loop
  - changed `online_softmax_with_quant_pv_exact()` from
    `cutlass.range_constexpr(num_groups)` to `cutlass.range(num_groups, unroll=1)`
  - exact-lane probes stayed clean
    - `test_fp4_pv_probe_constant_v_populates_all_output_channels`
    - `test_fp4_pv_probe_v_scale_axis_is_colwise`
  - the must-win fresh-process median on device `2` did not improve
    - `qkfast_ms ~= 0.13063`
    - `pv_fused_ms ~= 0.15790`
    - ratio `~= 1.209`
  - compile / wall time also got worse
  - conclusion:
    - reverted
- widening the correction/rescale tile from `16` to `32`
  - knob:
    - `FLASH_ATTN_FP4_PV_CORR_TILE_SIZE=32`
  - exact-lane probes stayed clean
  - must-win fresh-process median on device `2` regressed
    - `qkfast_ms ~= 0.12260`
    - `pv_fused_ms ~= 0.15348`
    - ratio `~= 1.252`
  - conclusion:
    - keep the default `corr_tile_size=16`
- short single-process register screens on device `2`
  - tested:
    - `FLASH_ATTN_FP4_FORCE_REGS_SOFTMAX in {176, 184}`
    - `FLASH_ATTN_FP4_FORCE_REGS_OTHER in {32, 64}`
  - all results stayed clustered around `~1.25`
  - no new setting separated enough from noise to justify a fresh-process median
  - conclusion:
    - no new keeper
- `kv_stage` screening on the must-win row
  - short single-process samples:
    - `20`: `~1.274`
    - `19`: `~1.264`
    - `18`: `~1.186`
  - `kv_stage=18` looked promising enough to verify properly
  - exact-lane probes with `FLASH_ATTN_FP4_FORCE_KV_STAGE=18` stayed clean
  - but the must-win fresh-process median on device `2` reverted to:
    - `qkfast_ms ~= 0.11211`
    - `pv_fused_ms ~= 0.13988`
    - ratio `~= 1.248`
  - conclusion:
    - the short-run win was timing noise
- manual direct-loader screening
  - trying the CTA-manual PV direct-loader path triggered an unrelated `nvcc`
    build in the sibling CCE tree during the screen
  - that run was not a meaningful flash-attention-only comparison
  - conclusion:
    - do not use that datapoint for optimization decisions
- exact fused-`exp2` rewrite inside `online_softmax_with_quant_pv_exact()`
  - idea:
    - compute `exp2(v)` once into `p0..p7`
    - accumulate `row_sum` from those registers
    - scale the same registers in place instead of evaluating a second `exp2`
      pass for packed `P`
  - outcome:
    - both exact-lane oracle probes regressed immediately
    - constant-`V`:
      - max abs diff jumped to `1.375`
    - colwise `V`-scale:
      - `87.5%` mismatched
      - max abs diff `5.5`
  - conclusion:
    - reverted
    - the exact lane is sensitive to the current direct `exp2(v - group_max + log2_fp4_max)`
      form, even though the fused rewrite looked algebraically equivalent
- switching the exact helper to the MN-major `tile_atom_to_shape_sf_mn(...)`
  logical `SFP` view
  - rationale:
    - the stable fused path already uses the explicit MN-major helper for its
      `SFP` logical view
  - outcome:
    - both exact-lane oracle probes regressed
    - constant-`V`:
      - max abs diff `0.7705`
    - colwise `V`-scale:
      - `62.7%` mismatched
      - max abs diff `3.0820`
  - conclusion:
    - reverted
    - the exact lane still needs the current generic `bs_layout.tile_atom_to_shape_SF(...)`
      logical view

## Recommended Next Steps

1. Keep the current stable split-handoff exact lane intact as the baseline.
2. Use the batched host timer for routine perf checks and the exact profiling mode for structural investigations.
3. Target exact `SFV` staging shape directly:
   - reduce or redesign the current public scale-byte path
   - avoid direct reuse of the current generic swizzled shared-scale contract if possible
   - preserve blockscaled legality
4. Re-profile after each structural `SFV` change using the exact profiling mode and imported `ncu` reports.
5. Keep `d64` dense `MHA` enabled, but do not widen further to `GQA` or causal until the `d128`, `S=512` must-win row is robustly `< 1.0` on devices `1` and `2`.

Recent non-keepers from this pass:

- exact `d128` correction-reg sweep on device `2`
  - `72` is still best among `{64, 72, 80}`
  - measured ratios:
    - `64`: `1.214`
    - `72`: `1.175`
    - `80`: `1.219`
  - conclusion:
    - keep `72`
- `FLASH_ATTN_FP4_PV_EXACT_SFV_DIRECT=0` on the must-win row
  - still goes non-finite (`LSE` NaN/Inf) on `d128, S=512`
  - conclusion:
    - keep it diagnostic-only
- full-tile `SFV` scale-byte broadcast experiment
  - reused one public `SFV` byte across the `16` rows of each sequence group
  - correctness stayed clean
  - device `2` must-win ratio moved the wrong way, from `~1.238` to `~1.258`
  - conclusion:
    - reverted

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

## Recovery Alignment Check

I re-opened the clean `6be31f4` recovery worktree to explain why it had been benchmarking materially slower than the current main tree, even though the obvious exact-kernel diffs were small.

What actually explained most of that gap:

- the recovery worktree was still on the older float8 TVM-FFI compatibility path in `cute_dsl_utils.py`
- the main tree uses direct `uint8` storage views for float8 tensors and assigns the CUTLASS float8 dtype explicitly
- the recovery `interface.py` also kept routing non-PV-fast float8 scales through the legacy qkfast helper, while the main tree only does that on the old `overlay` stack

After aligning the recovery worktree with the main tree's float8 runtime path:

- exact recovery probes stayed clean:
  - constant-`V`
  - colwise `V` scale
  - exact-lane `S=64 -> 1024` recompilation
- must-win row on device `2` moved from the previous `~1.40x` band down to:
  - `qkfast_ms = 0.11446`
  - `pv_fused_ms = 0.14157`
  - `pv_fused_over_qkfast = 1.237`

Practical conclusion:

- the recovery-vs-main perf gap was mostly **not** explained by the d128 exact CTA/tail shape
- it was mostly explained by float8 runtime marshalling differences
- once that was aligned, the recovery worktree matched the current main-tree baseline closely enough that the recovery branch stopped being an independent performance lead

## Rejected Experiments From This Pass

- Porting the full current main-tree d128 correction-tail shape into the recovered `6be31f4` tree
  - included:
    - wide d128 correction warp map
    - tuned d128 `num_regs_correction = 72`
    - second compact exact-correction `LSE` publication pass
  - result on the must-win row after that port:
    - `qkfast_ms = 0.10810`
    - `pv_fused_ms = 0.15132`
    - `pv_fused_over_qkfast = 1.400`
  - conclusion:
    - by itself, that kernel hunk port did not explain the recovery/main gap
    - reverted in the recovery worktree

- Exact row-sum rewrite in `online_softmax_with_quant_pv_exact()`
  - attempted to replace the second set of `exp2(v)` evaluations with `scale_f32 * sum(p0..p7)`
  - direct constant-`V` probe regressed badly:
    - `out_max_err = 1.375`
    - `LSE` stayed finite
  - conclusion:
    - not numerically safe enough on this path
    - reverted

- Removing the redundant exact-lane `col` shuffles from the dense butterfly reduction
  - rationale:
    - `fp4_pv_exact_group_coord()` currently returns `(row_offset + group_idx, 0)`, so the `col` lane value is constant
  - validation:
    - constant-`V` probe stayed clean
  - timing on the must-win row was not stable enough to justify keeping it:
    - first run: `1.236`
    - immediate rerun: `1.296`
  - conclusion:
    - no defensible win
    - reverted

## Current Baseline After This Pass

- main tree remains the active baseline
- recovery tree is now useful only as a historical checkpoint, not a faster branch
- must-win row is still in the `~1.24x` band on trustworthy device-`2` host-timed fused-only runs
- the next real win still has to come from reducing exact back-half cost, not from recovery plumbing or another micro-screened schedule tweak

## Exact-Path Plumbing Cleanup

I kept one exact-lane source cleanup in the main tree:

- split the coordinate-dump logic out of `online_softmax_with_quant_pv_exact()`
- the hot exact helper no longer takes `tScS`
- the hot exact path no longer constructs or threads the dead `sSFP_logical_u8_flat_exact` view
- the debug path now rebuilds its own logical `tScS` only when `FLASH_ATTN_FP4_PV_DEBUG_DUMP_PCOORDS=1`
- the hot exact helper now uses the proven live mapping directly:
  - `row = row_offset + group_idx`
  - `col = 0`

Validation after the cleanup:

- direct constant-`V` oracle stayed clean:
  - `out_max_err = 0.0546875`
  - finite `LSE`
- debug coordinate dump still worked:
  - emitted `pcoord-match ...`
  - no `pcoord-mismatch ...` output

Must-win row on device `2` after the cleanup stayed in the same noisy band:

- run 1:
  - `qkfast_ms = 0.10390`
  - `pv_fused_ms = 0.13163`
  - `pv_fused_over_qkfast = 1.267`
- run 2:
  - `qkfast_ms = 0.11977`
  - `pv_fused_ms = 0.14589`
  - `pv_fused_over_qkfast = 1.218`
- direct-mapping-only reruns:
  - `1.258`
  - `1.267`

Conclusion:

- this cleanup did not produce a decisive performance win
- it also did not show a correctness regression
- I kept it because it narrows the shipped exact helper and removes dead exact-only plumbing from the hot path

## Rejected Exact-Tail Reorder

I also tried reordering the exact tail so the register-to-TMEM `P` stores ran before the shared-to-TMEM `SFP` copy and proxy fence.

Validation:

- constant-`V` oracle stayed clean

Performance:

- must-win row on device `2` regressed to:
  - `qkfast_ms = 0.12106`
  - `pv_fused_ms = 0.15887`
  - `pv_fused_over_qkfast = 1.312`

Conclusion:

- rejected and reverted

## Mapping-First Pass

Implemented the mapping-first exact-helper cleanup in `fp4_flash_fwd_sm100_pvfused.py`.

- derived exact grouped-slot mapping from the live kernel instead of the unstable host-side CuTe introspection path
- kept a diagnostic-only `FLASH_ATTN_FP4_PV_DEBUG_DUMP_PCOORDS=1` path that rebuilds the old grouped coordinate tensor and prints `pcoord-mismatch ...` if the derived mapping ever diverges
- added a focused runtime harness in `test_fp4_flash_attn.py` that enables that debug path on a full `d128, S=128` exact tile and asserts:
  - no `pcoord-mismatch` output
  - sampled `pcoord-match gi=0 row=0 col=0` output is present

Derived mapping result for the current exact fused softmax fragment:

- `row = row_offset + group_idx`
- `col = 0`

Practical code consequence:

- default exact execution no longer constructs `tScP_conv_groups` in `online_softmax_with_quant_pv_exact()`
- the grouped coordinate tensor now exists only behind the diagnostic env gate

Validation on the landed mapping-only state:

- `python -m py_compile flash_attn/cute/fp4_flash_fwd_sm100_pvfused.py tests/cute/test_fp4_flash_attn.py`
- focused exact probe slice:
  - `test_fp4_pv_probe_constant_v_populates_all_output_channels`
  - `test_fp4_pv_probe_v_scale_axis_is_colwise`
  - `test_fp4_pv_probe_exact_group_coord_matches_live_layout`
  - `test_fp4_pv_probe_exact_lane_recompiles_across_seqlens`
  - result: `4 passed, 146 deselected`

Post-mapping must-win-row timing on device `2`:

- row: noncausal `MHA`, `d128`, `S=512`, `batch=2`
- `qkfast_ms = 0.10878245779116388`
- `pv_fused_ms = 0.136179340910827`
- `pv_fused_over_qkfast = 1.2518501941945321`

Conclusion from this pass:

- the mapping derivation is now proven and the hot exact path is simpler
- this cleanup did **not** move the must-win row below `1.0`
- it also did not produce a clear device-`2` win versus the prior baseline

Rejected follow-up from the same pass:

- replacing the dense exact butterfly path with an unconditional warp-wide max reduction
  - correctness stayed clean
  - device `2` must-win ratio moved the wrong way to about `1.251`
  - reverted

## Exact Softmax Helper Spill Experiments

I spent this pass on the exact softmax helper itself rather than more register/stage screens.

Baseline target row for all measurements:

- device `2`
- noncausal `MHA`
- `d128`
- `S=512`
- batch `2`
- `FLASH_ATTN_FP4_PV_EXACT_SFV_DIRECT=1`
- `--compare-mode fused-only`

### PTXAS readout

I forced fresh PTX dumps with:

- `CUTE_DSL_KEEP_PTX=1`
- `CUTE_DSL_DUMP_DIR=/tmp/fp4_ptx_dump_512`
- `CUTE_DSL_NO_CACHE=1`

and ran:

- `/usr/local/cuda-13.0/bin/ptxas -v -arch=sm_100a -O3 ...`

For the must-win row kernel, the first chunked exact-helper variant compiled to:

- `168` registers
- `344` bytes stack frame
- `532` bytes spill stores
- `680` bytes spill loads

This is worse than the earlier must-win PTXAS snapshot (`224` / `396` / `572`), so the chunking did not reduce spill pressure even though its runtime stayed in the same noisy band.

### Variants tried

1. Chunk the exact grouped loop into four `8`-group helper calls, with one generic helper handling both dense and masked paths via `use_masked_exp_emu`.
   - correctness stayed clean
   - must-win timing:
     - `qkfast_ms = 0.10203`
     - `pv_fused_ms = 0.12713`
     - `pv_fused_over_qkfast = 1.246`
   - later rerun after restoring this exact variant:
     - `qkfast_ms = 0.09208`
     - `pv_fused_ms = 0.11581`
     - `pv_fused_over_qkfast = 1.258`
   - this is the best correctness-clean state from this pass, so I kept it in the tree

2. Split the helper into separate dense and masked implementations so the dense compile dropped the slot-pointer path entirely.
   - constant-`V` oracle stayed clean
   - must-win timing regressed to:
     - `qkfast_ms = 0.10356`
     - `pv_fused_ms = 0.13384`
     - `pv_fused_over_qkfast = 1.292`
   - rejected and reverted

3. Restore the direct monolithic exact loop and try packing directly from `exp2(...)` expressions instead of keeping `p0..p7` locals live.
   - small constant-`V` oracle stayed clean
   - must-win row failed with `FP4 PV LSE contains NaN or Inf`
   - rejected and reverted

4. Restore the direct monolithic exact loop without the pack rewrite.
   - correctness stayed clean
   - must-win row was materially worse than the chunked helper on repeated runs:
     - `1.350`
     - `1.390`
   - rejected in favor of the chunked helper state above

5. Pairwise `e2m1x2` pack rewrite: convert two floats to one byte at a time, then assemble the packed word from four bytes.
   - constant-`V` oracle stayed clean after fixing the helper asm shape
   - must-win timing stayed in the same band:
     - `qkfast_ms = 0.11027`
     - `pv_fused_ms = 0.13899`
     - `pv_fused_over_qkfast = 1.260`
   - PTXAS got materially worse on the must-win row:
     - `168` regs
     - `456` bytes stack frame
     - `632` bytes spill stores
     - `768` bytes spill loads
   - rejected and reverted

6. Mark `pack_float8_to_e2m1_word(...)` itself as a `dsl_user_op`.
   - exact oracle stayed clean
   - repeated must-win timings moved only slightly:
     - `qkfast_ms = 0.10823`
     - `pv_fused_ms = 0.13521`
     - `pv_fused_over_qkfast = 1.249`
     - rerun: `0.10949`, `0.13686`, `1.250`
   - PTXAS still got slightly worse than the kept chunked baseline:
     - `168` regs
     - `352` bytes stack frame
     - `540` bytes spill stores
     - `692` bytes spill loads
   - rejected and reverted

7. Exact-only `4 + dup-tail` pack helper, matching the live PTX pattern (`p0..p3` plus a duplicated tail value).
   - kept the exact row-sum path unchanged and only specialized the pack side
   - exact probes still passed:
     - constant-`V`: `max_err = 0.0546875`, finite `LSE`
     - colwise `V`-scale: `assert_close` passed, raw `max_err = 0.21875`
   - must-win timing did not improve:
     - `qkfast_ms = 0.10068`
     - `pv_fused_ms = 0.12692`
     - `pv_fused_over_qkfast = 1.261`
   - rejected and reverted

8. Exact-only opaque pack helper: same `pack_float8_to_e2m1_word(...)` asm body, but with `has_side_effects=True` to restrict motion around the pack boundary.
   - constant-`V` oracle stayed clean
   - must-win timing regressed:
     - `qkfast_ms = 0.12902`
     - `pv_fused_ms = 0.16501`
     - `pv_fused_over_qkfast = 1.279`
   - rejected and reverted

9. Incremental exact row-sum accumulation: keep the exact math unchanged but accumulate `row_sum_new` directly instead of holding `e0..e7` temporaries live.
   - constant-`V` oracle stayed clean
   - must-win timings landed in the noise band:
     - `qkfast_ms = 0.10177`
     - `pv_fused_ms = 0.12723`
     - `pv_fused_over_qkfast = 1.250`
     - rerun: `0.10243`, `0.12898`, `1.259`
   - PTXAS was not an improvement; it matched the worse `352 / 540 / 692` stack/spill shape rather than the kept baseline `344 / 532 / 680`
   - rejected and reverted

10. Dense exact-path unconditional butterfly reduction in the unmasked case.
   - removed the `(row, col)` partner checks on the dense must-win path and applied the `offset=1,2,4,8,16` max reductions unconditionally
   - constant-`V` oracle stayed clean
   - must-win timing regressed:
     - `qkfast_ms = 0.09507`
     - `pv_fused_ms = 0.12116`
     - `pv_fused_over_qkfast = 1.275`
   - rejected and reverted

11. Per-term `p_i * scale_f32` exact row-sum rewrite.
   - idea:
     - compute only the `p0..p7` values
     - reuse them for row-sum as per-term `p_i * scale_f32`
     - avoid the separate `exp2(v)` row-sum pass without using the previously rejected `scale_f32 * sum(p)` aggregate form
   - outcome:
     - constant-`V` oracle regressed immediately:
       - `max_err = 1.375`
       - finite `LSE`
   - rejected and reverted

12. Dynamic exact grouped loop with `range(...)` / `range_dynamic(...)`.
   - rationale:
     - if the grouped helper could run as a real dynamic loop instead of full constexpr unrolling, it might shrink the hot exact block enough to reduce spills
   - outcome:
     - builtin `range(group_count)` broke the exact oracle with the same `max_err = 1.375`
     - `cutlass.range_dynamic(group_count)` also broke the oracle, even after carrying the loop index as `Int32`
     - `range_dynamic` additionally emits a deprecation warning in the current DSL
   - conclusion:
     - this is not a drop-in runtime-loop recovery on the current stack
     - rejected and reverted

13. Two-pass exact split matching the non-fused FP4 path:
   - first pass:
     - exact grouped reduction
     - exact `SFP` publication
     - write quantized `p0..p7` into the temporary exact quant fragment
   - second pass:
     - pack that fragment into `e2m1` words
   - initial flat-pack version regressed the constant-`V` oracle to `max_err = 0.9140625`
   - packing directly from `tSrP_conv_quant_groups[group_idx, ei]` improved that but was still far off:
     - constant-`V`: `max_err = 0.509765625`
     - finite `LSE`
   - conclusion:
     - the structural split is directionally interesting, but the current exact fragment/group order is still not a drop-in replacement for the one-pass pack path
     - rejected and reverted

14. Debug-only two-pass parity harness on the exact one-pass baseline.
   - landed as an internal-only diagnostic behind the existing `FLASH_ATTN_FP4_PV_DEBUG_DUMP_PCOORDS=1` hook:
     - phase 1 writes exact grouped quantized `p0..p7` values into a separate scratch fragment
     - phase 2 repacks that scratch and compares the result against the existing one-pass packed word
   - exact group-coordinate mapping is still confirmed:
     - `pcoord-match gi=0..3 row=0..3 col=0`
   - packed-word parity still fails systematically on the dense exact tile:
     - example row-0 mismatches:
       - `gi=0`: `ref_word=0x22227777`, `two_pass_word=0x77777777`
       - later groups drift to `0x70707777` / `0x70707070`-style patterns rather than converging
   - two scratch/pack order attempts were screened:
     - grouped scratch write + grouped repack
     - grouped scratch write + flat scratch repack
   - both still mismatch the live one-pass word stream, so the exact two-pass path was not promoted and no parity assertion was kept in tests
   - current code keeps:
     - the debug-only parity harness
     - the one-pass exact production path unchanged
     - only the existing live-layout regression test on the debug hook

15. Corrected flat-scratch two-pass parity and rejected live two-pass promotion.
   - fixed the debug-only two-pass parity harness by replacing the exact-backed scratch with a plain flat scratch of length `num_groups * 8` and explicit `base = group_idx * 8` indexing
   - the corrected parity harness now matches the one-pass packed words on the dense exact tile:
     - `pflat0` reproduces the live `pref0` values
     - `ppack-match gi=0..3` shows the flat scratch repack agrees with the live one-pass word stream
   - added a focused regression:
     - `test_fp4_pv_probe_exact_two_pass_pack_matches_one_pass`
   - then promoted the same two-pass flat scratch into the live exact path as an A/B:
     - exact probe slice stayed correctness-clean after limiting the debug reference work to `row_offset == 0`
     - must-win perf on device `2`, `d128`, `S=512`, `batch=2`, `fused-only`, `FLASH_ATTN_FP4_PV_EXACT_SFV_DIRECT=1` did not improve:
       - `qkfast_ms = 0.11768`
       - `pv_fused_ms = 0.14849`
       - `pv_fused_over_qkfast = 1.262`
   - conclusion:
     - the flat two-pass split is now correctness-proven as a debug parity harness
     - it is not a performance win in the live kernel on the must-win row
     - production was reverted to the one-pass exact path, but the corrected debug parity harness and new regression test were kept
   - restored must-win baseline after reverting production:
     - device `2`, `d128`, `S=512`, `batch=2`, `fused-only`, `FLASH_ATTN_FP4_PV_EXACT_SFV_DIRECT=1`
     - `qkfast_ms = 0.11442`
     - `pv_fused_ms = 0.14278`
     - `pv_fused_over_qkfast = 1.248`

16. Kept exact dense-reduction cleanup: remove dead `col` shuffles from the butterfly max path.
   - change:
     - exact `col` is always `0` on this lane mapping
     - removed the `partner_col = shuffle(col, offset=...)` traffic and the redundant `(col == partner_col)` checks
     - kept the `partner_row` gating, so this is not the previously rejected unconditional butterfly path
   - correctness:
     - focused exact slice stayed clean:
       - `4 passed, 147 deselected`
   - must-win timing on device `2`, isolated fused-only reruns:
     - run 1:
       - `qkfast_ms = 0.10329`
       - `pv_fused_ms = 0.12730`
       - `pv_fused_over_qkfast = 1.233`
     - run 2:
       - `qkfast_ms = 0.11234`
       - `pv_fused_ms = 0.13828`
       - `pv_fused_over_qkfast = 1.231`
   - PTXAS from a fresh forced dump was mixed rather than improved:
     - `168` registers
     - `352` bytes stack frame
     - `544` bytes spill stores
     - `696` bytes spill loads
     - this is slightly worse than the earlier `344 / 532 / 680` readout even though runtime improved on device `2`
   - device `1` was not usable as a confirmation gate during this pass:
     - `nvidia-smi` showed GPU `1` pinned at `100%` with three resident `python3` processes using about `968 MiB` each
     - the single device-`1` benchmark result (`2.675x`) should be treated as contaminated and not compared to the clean device-`2` numbers

### Current conclusion

- The best state from this pass is still well above the target:
  - `pv_fused_over_qkfast ~= 1.23`
- The exact softmax helper remains the right place to work, but the problem is not solved by simple loop chunking or by splitting masked/dense control flow.
- The direct-pack rewrite is not safe on the must-win row.
- The pairwise-byte pack rewrite is also not the answer; it compiles and stays correct, but it increases spill pressure.
- The corrected flat-scratch two-pass split is a valid diagnostic / reference path, but not a shipped speedup.
- Deriving row-sum or packed `P` from reused scaled terms is still not safe:
  - both the per-term `p_i * scale_f32` row-sum rewrite and the `p_i = exp2(v_i) * inv_scale_f32` rewrite reproduced the `max_err = 1.375` constant-`V` failure and were reverted.
- The next credible target is the pack/quant live range itself, likely around the `pack_float8_to_e2m1_word(...)` path and its `MUFU.EX2` / `PRMT` spill cluster, not the correction tail.

17. Recovered the local exact baseline after an accidental kernel-file reset.
   - restored the main-tree kernel from the `6be31f4` recovery worktree, then re-applied:
     - default exact `SFV` direct mode when `FLASH_ATTN_FP4_PV_EXACT_SFV_DIRECT` is unset
     - exact group-coordinate debug dump
     - exact packed-word parity debug harness on a flat scratch, limited to `row_offset == 0`
     - chunked exact helper (`4 x 8` groups on the dense `32`-group tile)
     - exact d128 widened correction tail (`4` correction warps) with `num_regs_correction = 72`
     - dead `partner_col` removal in the exact butterfly path
   - focused exact slice is clean again:
     - `4 passed, 147 deselected`
   - restored must-win baseline on device `2`, `d128`, `S=512`, `batch=2`, noncausal, fused-only:
     - `qkfast_ms = 0.11880`
     - `pv_fused_ms = 0.14467`
     - `pv_fused_over_qkfast = 1.2178`
   - this is back in the pre-reset performance band, though still above the `< 1.0` target

18. Clean single-run screen for `FLASH_ATTN_FP4_FORCE_REGS_OTHER=64`.
   - rationale:
     - after restoring the chunked helper and d128 tail shape, re-check whether the only mildly promising register knob from earlier work helps on the recovered line
   - clean device `2` result:
     - `qkfast_ms = 0.10852`
     - `pv_fused_ms = 0.13526`
     - `pv_fused_over_qkfast = 1.2464`
   - conclusion:
     - `num_regs_other=64` is still not an improvement on the current restored baseline
     - keep the kernel default unchanged

19. Revalidated the restored `4 x 8` exact-helper baseline after rejecting the spill-first subhelper refactor.
   - correctness:
     - focused exact slice stayed clean:
       - `4 passed, 147 deselected`
   - honest device `2` must-win rerun:
     - `qkfast_ms = 0.11196`
     - `pv_fused_ms = 0.14035`
     - `pv_fused_over_qkfast = 1.2536`
   - conclusion:
     - the recovered main-tree anchor is still the same `~1.25x` band
     - no hidden win was left behind by the rejected refactor

20. Hoisted the exact butterfly row-equality checks out of the inner grouped loop, then rejected it.
   - change:
     - computed the `offset=1,2,4,8,16` row-equality masks once per helper call instead of once per group
     - left the debug path unchanged
   - correctness:
     - focused exact slice stayed clean:
       - `4 passed, 147 deselected`
   - must-win timing on device `2`:
     - `qkfast_ms = 0.10330`
     - `pv_fused_ms = 0.12921`
     - `pv_fused_over_qkfast = 1.2508`
   - conclusion:
     - effectively neutral versus the recovered baseline
     - reverted rather than stack a non-winning source change

21. Reduced the dense exact chunk size from `4 x 8` to `8 x 4`, then rejected it.
   - correctness:
     - focused exact slice stayed clean:
       - `4 passed, 147 deselected`
   - must-win timing on device `2` regressed badly:
     - `qkfast_ms = 0.11629`
     - `pv_fused_ms = 0.26117`
     - `pv_fused_over_qkfast = 2.2458`
   - conclusion:
     - the current exact helper does not tolerate further chunk-size fragmentation
     - reverted immediately

22. Screened `FLASH_ATTN_FP4_FORCE_REGS_SOFTMAX=176` on the restored baseline.
   - rationale:
     - this register target is already hinted at in the source comments and is one of the few remaining low-cost screens
   - clean device `2` reruns were inconsistent:
     - run 1:
       - `qkfast_ms = 0.11079`
       - `pv_fused_ms = 0.13376`
       - `pv_fused_over_qkfast = 1.2073`
     - run 2:
       - `qkfast_ms = 0.11444`
       - `pv_fused_ms = 0.14656`
       - `pv_fused_over_qkfast = 1.2807`
   - conclusion:
     - there is not enough stability to bake `num_regs_softmax = 176` into the exact d128 default
     - leave the kernel default unchanged until a tighter median workflow or a source-level win moves the line more clearly

23. Rejected an exact-only inline-asm helper that packs directly from pre-exp logits.
   - change:
     - kept the exact row-sum path unchanged
     - replaced the production exact helper's `p0..p7` SSA temporaries with one inline-asm helper that did:
       - `sub/add`
       - `ex2.approx`
       - `cvt.rn.satfinite.e2m1x2`
       - final packed-word assembly
   - motivation:
     - target the exact pack-side live range directly so the `p0..p7` values never exist in compiler SSA
   - result:
     - exact correctness regressed immediately
     - focused exact slice failed:
       - constant-`V` probe failed with max abs diff `1.375`
       - colwise `V`-scale probe also failed broadly
   - conclusion:
     - reverted immediately
     - the exact failure signature matches the earlier unsafe pack/rowsum rewrites, so this helper is not a correct drop-in for the existing exact lane

24. Additional low-cost register screens on the restored baseline.
   - `FLASH_ATTN_FP4_FORCE_REGS_SOFTMAX=184`
     - `qkfast_ms = 0.11055`
     - `pv_fused_ms = 0.13890`
     - `pv_fused_over_qkfast = 1.2564`
     - rejected
   - `FLASH_ATTN_FP4_FORCE_REGS_SOFTMAX=168`
     - `qkfast_ms = 0.09853`
     - `pv_fused_ms = 0.12576`
     - `pv_fused_over_qkfast = 1.2764`
     - rejected
   - `FLASH_ATTN_FP4_FORCE_REGS_OTHER=32`
     - `qkfast_ms = 0.11540`
     - `pv_fused_ms = 0.15448`
     - `pv_fused_over_qkfast = 1.3386`
     - rejected
   - conclusion:
     - the only vaguely promising softmax-reg screen remains `176`, but it is still too unstable to bake
     - the remaining work is backend-shape reduction, not another simple register-budget sweep

25. Rejected the mathematically exact pack-reuse form `p_i = exp2(v_i) * exp2(log2(6) - group_max_safe)`.
   - change:
     - kept the exact row-sum `e0..e7 = exp2(v_i)` path
     - replaced the second pack-side `exp2(v_i - group_max_safe + log2_fp4_max)` burst with:
       - one `pack_scale_f32 = exp2(log2_fp4_max - group_max_safe)`
       - eight multiplies `p_i = e_i * pack_scale_f32`
   - result:
     - exact correctness regressed immediately with the same signature as the earlier unsafe reuse rewrites:
       - constant-`V` probe failed with max abs diff `1.375`
       - colwise `V`-scale probe also failed broadly
   - conclusion:
     - reverted immediately
     - this closes the "mathematically exact reuse" line as well; the exact pack path is not a drop-in rewrite over the row-sum exponents

26. Screened `2 x 16` chunking for the dense exact `32`-group tile.
   - correctness:
     - focused exact slice stayed clean
   - must-win timing on device `2` regressed:
     - `qkfast_ms = 0.11173`
     - `pv_fused_ms = 0.14235`
     - `pv_fused_over_qkfast = 1.2740`
   - conclusion:
     - reverted in favor of the existing `4 x 8` chunking

27. Settled the only remaining mildly promising register knob with a steadier baseline-vs-`176` comparison.
   - baseline, device `2`, `--warmup 20 --iters 40 --fresh-runs 4`:
     - `qkfast_ms = 0.10007`
     - `pv_fused_ms = 0.12557`
     - `pv_fused_over_qkfast = 1.2548`
   - `FLASH_ATTN_FP4_FORCE_REGS_SOFTMAX=176`, rerun 1:
     - `qkfast_ms = 0.10490`
     - `pv_fused_ms = 0.12907`
     - `pv_fused_over_qkfast = 1.2304`
   - `FLASH_ATTN_FP4_FORCE_REGS_SOFTMAX=176`, rerun 2:
     - `qkfast_ms = 0.10280`
     - `pv_fused_ms = 0.12871`
     - `pv_fused_over_qkfast = 1.2521`
   - conclusion:
     - the apparent ratio win is not a real fused-kernel win; `pv_fused_ms` was higher than the clean baseline on both reruns
     - do not bake `num_regs_softmax = 176` into the exact d128 default

28. Derived the exact grouped value-coordinate pattern with a debug-only dump.
   - using `FLASH_ATTN_FP4_PV_DEBUG_DUMP_PCOORDS=1`, the exact grouped conversion values for the first dense groups printed as:
     - `gi=0`: `(0,0) (32,0) (64,0) (96,0) (128,0) (160,0) (192,0) (224,0)`
     - `gi=1`: `(1,0) (33,0) (65,0) (97,0) (129,0) (161,0) (193,0) (225,0)`
   - conclusion:
     - exact grouped values are not contiguous in the logical conversion view; they stride by `32` rows at fixed column `0`
     - this explains why several pack-side rewrites that assumed a simpler contiguous/flat order were brittle or outright wrong

29. Extended the exact debug dump to inspect actual grouped values, not just coordinates.
   - with a direct random-QK probe under `FLASH_ATTN_FP4_PV_DEBUG_DUMP_PCOORDS=1`, the first exact groups showed:
     - lanes `0..3` saw distinct `v0..v3`
     - but `v4..v7` were all exactly `0.0`
   - example:
     - `lane=0 gi=0`: `v0=-48.93 v1=-43.71 v2=-33.41 v3=-43.01 v4=v5=v6=v7=0.0`
     - `lane=1 gi=0`: `v0=-38.67 v1=-14.51 v2=-59.90 v3=-41.79 v4=v5=v6=v7=0.0`
   - interpretation:
     - on the dense exact `m_block_size=128` tile, each grouped exact pack sees `4` live logits plus `4` zero-padded tail entries
     - this also explains the earlier PTX shape and why the previous `4 + dup-tail` pack helper was correctness-clean

30. Tried a dense exact zero-tail specialization and rejected it.
   - specialization:
     - dense, unmasked exact path only
     - used `4` live logits plus a repeated zero-tail value
     - folded the row-sum tail to `+4.0`
     - packed as `p0 p1 p2 p3 ptail ptail ptail ptail`
   - correctness:
     - focused exact slice stayed clean:
       - `4 passed, 147 deselected`
   - must-win timing on device `2` regressed slightly on the steadier workflow:
     - `qkfast_ms = 0.10766`
     - `pv_fused_ms = 0.13547`
     - `pv_fused_over_qkfast = 1.2583`
   - conclusion:
     - reverted in favor of the restored baseline
     - the structural fact is real, but this source-level specialization is not the speedup

31. Dumped the generated PTX for the current dense exact baseline and closed the pack-side zero-tail line more firmly.
   - setup:
     - ran the exact profile-only compile with:
       - `CUTE_DSL_KEEP_PTX=1`
       - `CUTE_DSL_DUMP_DIR=/tmp/fa_ptx_dump`
       - `FLASH_ATTN_FP4_PV_EXACT_SFV_DIRECT=1`
   - useful PTX result:
     - the emitted dense exact block already recognizes the `4 live + repeated tail` shape on its own
     - representative block from the current baseline:
       - computes only `4` live pack exponents
       - then packs:
         - `byte0 <- (p1, p0)`
         - `byte1 <- (p3, p2)`
         - `byte2 <- (ptail, ptail)`
         - `byte3 <- (ptail, ptail)`
   - implication:
     - the compiler is already folding the high-level exact helper down to the same dense-tail pattern that the recent source rewrites tried to express manually
     - that explains why the source-level zero-tail specialization and later direct-asm dense pack idea did not buy anything
   - rejected follow-up:
     - added a direct dense-exact asm pack helper that took `(v0, v1, v2, v3, group_max_log2)` and emitted the repeated-tail packed word internally
     - kept the same dense exact row-sum form and scale publication
   - correctness:
     - focused exact slice stayed clean:
       - `4 passed, 147 deselected`
   - must-win timing on device `2` was flat-to-worse:
     - `qkfast_ms = 0.11774`
     - `pv_fused_ms = 0.14779`
     - `pv_fused_over_qkfast = 1.2552`
   - conclusion:
     - reverted immediately
     - the remaining work is not another high-level or medium-level "4 live + tail" pack rewrite
     - the next credible target is deeper backend-shape reduction around the exact unrolled group blocks and spill cluster

32. Rechecked the dense exact lane-mapping and screened single-lane `SFP` publication.
   - dense lane-map result:
     - with the debug dump widened to all `32` lanes for `gi=0`, every lane reported the same logical group coords:
       - `c0=(0, 0) c1=(32, 0)`
     - with random `Q/K`, the first dense exact group still had distinct live logits across essentially the full warp
   - implication:
     - the dense exact reduction really is warp-wide, so the max reduction width cannot be shrunk below `32`
     - the `partner_row` checks are semantically dead on the dense path, but the values being reduced are not duplicated across a smaller lane subset
   - screened optimization:
     - changed the production exact helper so only lane `0` computed `scale_u8` and wrote `sSFP_logical_u8[row, col, 0]` after the warp reduction
     - left the packed `P` path unchanged
   - correctness:
     - focused exact slice stayed clean:
       - `4 passed, 147 deselected`
   - must-win timing on device `2` regressed:
     - `qkfast_ms = 0.11088`
     - `pv_fused_ms = 0.14334`
     - `pv_fused_over_qkfast = 1.2928`
   - conclusion:
     - reverted immediately
     - redundant `scale_u8` publication is not the dominant cost center
     - the remaining gap is deeper in the exact group body than the shared-scale-byte store itself

33. Screened lane-0 tail-value broadcast for the dense exact d128 path.
   - change:
     - kept the existing warp-wide exact reduction
     - for dense unmasked d128 only, computed the repeated tail pack value `ptail = exp2(-group_max_safe + log2_fp4_max)` in lane `0`
     - broadcast `ptail` across the warp with `utils.shuffle_sync(..., offset=0)` and reused it for `p4..p7`
     - also kept the lane-0-only `scale_u8` publication inside the same specialized path
   - correctness:
     - focused exact slice stayed clean:
       - `4 passed, 147 deselected`
   - must-win timing on device `2` regressed:
     - `qkfast_ms = 0.10457`
     - `pv_fused_ms = 0.13344`
     - `pv_fused_over_qkfast = 1.2761`
   - conclusion:
     - reverted immediately
     - the warp-uniform tail pack value is also not the dominant source of the exact-lane overhead
     - this leaves the live per-lane `p0..p3` pack exponents and the overall unrolled exact group body as the main remaining suspects

34. Rechecked the runtime-loop line and rejected the monolithic `32`-group version.
   - change:
     - replaced the dense exact `4 x 8` production path with one monolithic `for group_idx in cutlass.range(32, unroll=1)` helper
   - outcome:
     - exact correctness regressed immediately with the known bad signature:
       - constant-`V` probe failed with max abs diff `1.375`
       - colwise `V`-scale probe also failed broadly
   - conclusion:
     - reverted immediately
     - this confirms the old note: a monolithic runtime loop is not a safe drop-in recovery on the current stack

35. Kept a backend-positive dense d128 exact specialization that hides both row-sum delta and packed-word construction inside `dsl_user_op` helpers.
   - specialization:
     - dense, unmasked exact d128 path only
     - `exact_d128_dense_row_sum_delta(v0..v3)` computes the `4` live `exp2` terms plus the `+4.0` zero-tail contribution inside one opaque helper
     - `pack_exact_d128_dense_word(v0..v3, group_max_safe)` computes the packed `P` word, including the repeated tail bytes, inside one opaque helper
     - the warp-wide exact reduction, scale-byte publication, and all masked paths remain unchanged
   - correctness:
     - focused exact slice stayed clean:
       - `4 passed, 147 deselected`
   - backend profile on device `2` improved materially relative to the previous exact baseline:
       - local loads: `173,824` vs `191,360`
       - local stores: `24,192` vs `29,440`
       - registers/thread unchanged at `168`
   - must-win timing:
     - first host-timed run was directionally positive:
       - `qkfast_ms = 0.11197`
       - `pv_fused_ms = 0.13948`
       - `pv_fused_over_qkfast = 1.2457`
     - steadier rerun was effectively baseline:
       - `qkfast_ms = 0.11381`
       - `pv_fused_ms = 0.14290`
       - `pv_fused_over_qkfast = 1.2556`
   - conclusion:
     - keep this variant on disk for now
     - it is the first exact-lane change that clearly reduces local-memory pressure without breaking correctness
     - runtime improvement is not established yet, but this is a better base than the prior source-only rewrites

36. Established a cleaner direct `ptxas -v` baseline for the kept dual-asm exact d128 path.
   - method:
     - dumped PTX for the must-win row with:
       - `CUTE_DSL_KEEP_PTX=1`
       - `CUTE_DSL_DUMP_DIR=/tmp/fa_ptx_baseline`
       - `CUTE_DSL_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas`
     - then ran `ptxas -v` directly on the dumped kernel PTX instead of relying on noisy runtime profiling
   - result for the exact fused kernel:
     - `168` registers
     - `200` bytes stack frame
     - `376` bytes spill stores
     - `472` bytes spill loads
   - conclusion:
     - use this direct `ptxas -v` path as the backend comparison baseline for further exact-lane source screens
     - the machine is too contended right now to treat end-to-end timing as authoritative

37. Screened a dense exact rewrite that skipped `v4..v7` loads and the generic masked-state variables in the d128 specialization.
   - change:
     - restructured the dense exact d128 branch to use only `v0..v3`
     - folded the local max to `max(v0..v3, 0.0)`
     - bypassed the generic `group_is_masked` / `packed_word=0` / `p0..p7` setup in that branch
   - outcome:
     - the rewrite compiled after fixing a CuTeDSL control-flow issue around `partner_row`
     - noisy must-win timing got materially worse:
       - `qkfast_ms = 0.11628`
       - `pv_fused_ms = 0.18364`
       - `pv_fused_over_qkfast = 1.579`
   - conclusion:
     - reverted immediately
     - the current compiler lowering is sensitive to that source reshaping; simply removing the dead tail loads is not a free win

38. Screened opaque scale-byte generation for the same dense exact d128 branch.
   - change:
     - swapped the dense exact branch from `float_to_ue4m3_byte(exp2(...))` to the existing `exact_d128_dense_scale_u8(group_max_safe)` helper
     - left the dual-asm row-sum and packed-word helpers unchanged
   - backend result from direct `ptxas -v`:
     - unchanged versus the kept baseline:
       - `168` registers
       - `200` bytes stack frame
       - `376` bytes spill stores
       - `472` bytes spill loads
   - conclusion:
     - reverted
     - the remaining backend cost is not in the high-level scale-byte publication path

39. Screened a narrower post-reduction dense d128 specialization that only removed the dead masked-state tail.
   - change:
     - kept the existing `v4..v7` loads and warp-wide max reduction exactly as-is
     - moved the dense exact d128 branch to after the reduction so it could skip:
       - `group_is_masked`
       - `group_max_safe`
       - zero-initializing `p0..p7`
     - left the dual-asm row-sum and packed-word helpers unchanged
   - backend result from direct `ptxas -v`:
     - regressed relative to the kept baseline:
       - `216` bytes stack frame vs `200`
       - `392` bytes spill stores vs `376`
       - `504` bytes spill loads vs `472`
   - conclusion:
     - reverted immediately
     - even the narrower post-reduction specialization worsened backend pressure

40. Screened an opaque dense d128 warp-reduction helper.
   - change:
     - added `exact_d128_dense_group_max(row, v0..v3)` as one `dsl_user_op` helper
     - the helper computes:
       - `max(v0..v3, 0.0)`
       - the full `1/2/4/8/16` `shfl.sync.bfly` warp reduction with row-equality gating
     - combined it with the existing dense d128 row-sum and packed-word helpers
   - correctness:
     - direct constant-`V` probe stayed clean:
       - `max_err = 0.0546875`
       - finite `LSE`
   - backend result from direct `ptxas -v`:
     - neutral versus the kept baseline:
       - `168` registers
       - `200` bytes stack frame
       - `376` bytes spill stores
       - `472` bytes spill loads
   - runtime sample:
     - one in-process must-win run on the contended device came back much worse:
       - `qkfast_ms = 0.12734`
       - `pv_fused_ms = 0.33087`
       - `pv_fused_over_qkfast = 2.5982`
   - conclusion:
     - reverted
     - shrinking the Python IR around the warp-reduction ladder is not enough by itself

41. Screened compile-time zero tails for the dense exact d128 branch.
   - change:
     - replaced dense exact `v4..v7` fragment loads with compile-time `Float32(0.0)` constants
     - left the rest of the helper shape unchanged
   - backend result from direct `ptxas -v`:
     - exactly neutral versus the kept baseline:
       - `168` registers
       - `200` bytes stack frame
       - `376` bytes spill stores
       - `472` bytes spill loads
   - conclusion:
     - reverted
     - removing the dense tail loads alone does not move the backend shape

42. Screened the narrowest dense d128 no-mask cleanup.
   - change:
     - left the helper shape otherwise unchanged
     - only replaced the dense exact branch's impossible masked-state handling with:
       - `group_is_masked = False`
       - `group_max_safe = group_max_log2`
   - backend result from direct `ptxas -v`:
     - regressed to the same worse shape seen in the earlier post-reduction screen:
       - `216` bytes stack frame
       - `392` bytes spill stores
       - `504` bytes spill loads
   - conclusion:
     - reverted immediately
     - even the minimal dense no-mask rewrite perturbs lowering enough to hurt the backend

43. Screened a combined dense d128 pack-plus-row-sum helper.
   - change:
     - added one opaque helper that returned:
       - low `32` bits: packed exact dense d128 word
       - high `32` bits: `row_sum_new` delta as raw `f32` bits
     - used it only in the dense unmasked exact `d128` branch
   - correctness:
     - focused exact slice still passed:
       - `4 passed, 147 deselected`
     - direct probes on the screened kernel:
       - constant-`V`: `max_err = 0.0546875`, finite `LSE`
       - colwise `V`-scale: `max_err = 0.21875`, finite `LSE`
   - backend result from direct `ptxas -v`:
     - neutral versus the kept baseline:
       - `168` registers
       - `200` bytes stack frame
       - `376` bytes spill stores
       - `472` bytes spill loads
   - conclusion:
     - reverted
     - merging the dense row-sum and pack helpers did not improve backend shape and made the scale-axis probe looser than the kept baseline

44. Screened routing the live dense exact d128 path through the dedicated specialized helper.
   - change:
     - rewired the `num_groups == 32` path to call `online_softmax_with_quant_pv_exact_group_range_dense_d128(...)`
     - kept the existing `4 x 8` chunking and dense d128 helper trio:
       - `exact_d128_dense_row_sum_delta(...)`
       - `exact_d128_dense_scale_u8(...)`
       - `pack_exact_d128_dense_word(...)`
   - correctness:
     - focused exact slice still passed:
       - `4 passed, 147 deselected`
   - backend result from direct `ptxas -v`:
     - regressed versus the kept baseline:
       - `208` bytes stack frame vs `200`
       - `384` bytes spill stores vs `376`
       - `480` bytes spill loads vs `472`
   - runtime note:
     - the compile-and-run benchmark harness also hit a CUDA-side segfault during kernel lookup on this variant
   - conclusion:
     - reverted immediately
     - rewiring through the dedicated helper does not improve the backend shape and is less stable on this stack

45. Kept a branchless masked butterfly reduction for the dense exact d128 group max.
   - change:
     - added `exact_d128_dense_group_max_branchless(row, v0..v3)`
     - dense exact `d128` now computes the `1/2/4/8/16` butterfly reduction in one opaque helper
     - the helper keeps the same row gating, but replaces the control-flow ladder with:
       - `shfl.sync.bfly`
       - `setp.eq`
       - `selp.f32(..., -inf, ...)`
       - `max.f32`
   - correctness:
     - focused exact slice passed on the screened kernel:
       - `4 passed, 147 deselected`
     - direct must-win path checks in fresh processes stayed clean:
       - exact PV alone: finite output and `LSE`
       - exact PV followed by `qkfast`: finite output and `LSE`
   - backend result from direct `ptxas -v`:
     - improved stack frame with spills unchanged:
       - kept variant: `168` registers, `192` bytes stack frame, `376` bytes spill stores, `472` bytes spill loads
       - previous baseline: `168 / 200 / 376 / 472`
   - same-script host-timed A/B on the contended device:
     - branchless reduction variant:
       - `pv_ms = 0.3580`
       - `qk_ms = 0.1498`
       - ratio `= 2.3892`
     - restored baseline:
       - `pv_ms = 0.3757`
       - `qk_ms = 0.1536`
       - ratio `= 2.4453`
   - caveat:
     - the full `_benchmark_case(...)` harness still hit a CUDA-side segfault during kernel lookup on this variant
     - since the direct exact PV and mixed PV+`qkfast` paths both ran cleanly, the harness crash is being treated as a separate follow-up rather than an immediate revert trigger
   - conclusion:
     - kept for now
     - this is the first recent dense d128 screen with both a direct PTXAS win and a same-setup A/B timing win

46. Screened removing the impossible masked-state handling on top of the kept branchless reduction.
   - change:
     - kept the branchless dense d128 group-max helper
     - only replaced:
       - `group_is_masked = (group_max_log2 == -inf)`
       - `group_max_safe = selp(0, group_max_log2)`
     - with the dense exact assumption:
       - `group_is_masked = False`
       - `group_max_safe = group_max_log2`
   - backend result from direct `ptxas -v`:
     - worse than the kept branchless baseline:
       - screened variant: `168` registers, `200` bytes stack frame, `380` bytes spill stores, `476` bytes spill loads
       - kept branchless baseline: `168 / 192 / 376 / 472`
   - runtime note:
     - this narrower screen also hit the CUDA-side kernel-lookup segfault during direct exact-PV bring-up
   - conclusion:
     - reverted immediately
     - even on top of the branchless reduction, the dense no-mask rewrite still hurts backend lowering

47. Screened hoisting `scale_f32` out of the kept dense exact d128 fast path.
   - change:
     - kept the branchless dense d128 group-max helper
     - changed the dense exact d128 fast path to avoid materializing:
       - `scale_f32 = exp2(group_max_safe - log2_fp4_max)`
     - used `exact_d128_dense_scale_u8(group_max_safe)` directly instead
   - backend result from direct `ptxas -v`:
     - exactly neutral versus the kept branchless baseline:
       - `168` registers, `192` bytes stack frame, `376` bytes spill stores, `472` bytes spill loads
   - runtime note:
     - this screen also hit the CUDA-side kernel-lookup segfault during direct exact-PV bring-up
   - conclusion:
     - reverted
     - removing `scale_f32` from the dense fast path does not improve the backend shape

48. Screened fusing the dense exact d128 packed-word and scale-byte generation.
   - change:
     - kept the branchless dense d128 group-max helper
     - kept the exact row-sum helper separate
     - tried one new helper returning:
       - low `32` bits: packed FP4 word
       - high `32` bits: scale-byte payload
   - result:
     - compile failed during PTX assembly
     - repeated `ptxas` errors:
       - `Arguments mismatch for instruction 'mov'`
   - conclusion:
     - reverted immediately
     - this pack-plus-scale helper shape is not viable in the current DSL/PTX lowering

49. Screened making the dense exact d128 pack helper opaque on top of the kept branchless reduction.
   - change:
     - added `@dsl_user_op` to `pack_exact_d128_dense_word(...)`
   - backend result from direct `ptxas -v`:
     - exactly neutral versus the kept branchless baseline:
       - `168` registers, `192` bytes stack frame, `376` bytes spill stores, `472` bytes spill loads
   - runtime note:
     - direct exact-PV bring-up still hit the CUDA-side kernel-lookup segfault on this screen
   - conclusion:
     - reverted
     - making the pack helper opaque does not improve the backend shape

50. Screened splitting the dense exact d128 fast path out before the generic masked path.
   - change:
     - kept the branchless dense d128 group-max helper
     - moved the dense exact `d128` path to run immediately after the branchless reduction:
       - dense `scale_f32`
       - dense row-sum helper
       - dense packed-word helper
     - left the generic masked path only for the non-dense cases
   - backend result from direct `ptxas -v`:
     - hard regression versus the kept branchless baseline:
       - screened variant: `168` registers, `216` bytes stack frame, `392` bytes spill stores, `496` bytes spill loads
       - kept branchless baseline: `168 / 192 / 376 / 472`
   - runtime note:
     - this screen also hit the CUDA-side kernel-lookup segfault during direct exact-PV bring-up
   - conclusion:
     - reverted immediately
     - even with the branchless reduction in place, early-splitting the dense fast path worsens backend lowering

51. Screened immediate shared-byte publication for the dense exact d128 `SFP` store.
   - change:
     - kept the branchless dense d128 group-max helper
     - added a small `st.shared.u8` helper and used it to publish the dense exact `d128` scale byte immediately
     - goal:
       - stop keeping the shared `SFP` address live across the dense pack work
   - backend result from direct `ptxas -v`:
     - regressed versus the kept branchless baseline:
       - screened variant: `168` registers, `200` bytes stack frame, `376` bytes spill stores, `472` bytes spill loads
       - kept branchless baseline: `168 / 192 / 376 / 472`
   - runtime note:
     - this screen also hit the CUDA-side kernel-lookup segfault during direct exact-PV bring-up
   - conclusion:
     - reverted immediately
     - publishing the scale byte earlier does not reduce the address-spill problem in the generated code

52. Re-screened the combined dense exact d128 row-sum plus packed-word helper on top of the kept branchless reduction.
   - change:
     - kept the branchless dense d128 group-max helper
     - replaced the separate:
       - `exact_d128_dense_row_sum_delta(...)`
       - `pack_exact_d128_dense_word(...)`
     - with one helper returning:
       - low `32` bits: packed FP4 word
       - high `32` bits: exact dense row-sum delta bits
   - backend result from direct `ptxas -v`:
     - regressed versus the kept branchless baseline:
       - screened variant: `168` registers, `200` bytes stack frame, `376` bytes spill stores, `472` bytes spill loads
       - kept branchless baseline: `168 / 192 / 376 / 472`
   - runtime note:
     - this screen also hit the CUDA-side kernel-lookup segfault during direct exact-PV bring-up
   - conclusion:
     - reverted
     - the combined row-sum-plus-pack helper still loses against the kept branchless baseline

53. Re-screened routing the dense exact d128 path through the dedicated `online_softmax_with_quant_pv_exact_group_range_dense_d128(...)` helper.
   - change:
     - kept the branchless dense d128 group-max helper
     - switched the `num_groups == 32`, dense non-masked exact path to call the dedicated dense helper in `4 x 8` chunks
     - updated that dedicated helper to use `exact_d128_dense_group_max_branchless(...)`
   - backend result from direct `ptxas -v`:
     - exact same hard regression as the earlier early fast-path split:
       - screened variant: `168` registers, `216` bytes stack frame, `392` bytes spill stores, `496` bytes spill loads
       - kept branchless baseline: `168 / 192 / 376 / 472`
   - runtime note:
     - direct PTX-dump bring-up again hit the CUDA-side `cuLibraryGetKernel` segfault after emitting PTX
   - conclusion:
     - reverted immediately
     - routing the dense exact path through the dedicated helper is not a viable way to reduce the remaining spill cluster

54. Screened narrowing the dense exact fast-path guard from `head_dim_v_padded >= 128` to `head_dim_v_padded == 128`.
   - change:
     - only changed the two dense exact fast-path guards inside `online_softmax_with_quant_pv_exact_group_range(...)`
     - goal:
       - see whether an exact-`128` guard gives the compiler a cleaner specialization boundary for the must-win row
   - backend result from direct `ptxas -v`:
     - regressed stack back to the older baseline:
       - screened variant: `168` registers, `200` bytes stack frame, `376` bytes spill stores, `472` bytes spill loads
       - kept branchless baseline: `168 / 192 / 376 / 472`
   - PTX result:
     - the full `shfl.sync.idx` reduction sequence was still present in the dumped PTX
   - conclusion:
     - reverted immediately
     - the surviving `shfl.sync.idx` block is not explained by the `>= 128` guard shape

55. Screened splitting the exact group-range helper by mask mode at the parent call site.
   - change:
     - added a separate unmasked exact helper with the masked slot-reduction logic removed from its source body
     - dispatched:
       - masked path -> existing `online_softmax_with_quant_pv_exact_group_range(...)`
       - unmasked path -> new unmasked helper
     - goal:
       - dead-strip the `SHFL.IDX` masked peer-reduction block from the noncausal must-win specialization
   - backend result from direct `ptxas -v`:
     - same regression as the `== 128` screen:
       - screened variant: `168` registers, `200` bytes stack frame, `376` bytes spill stores, `472` bytes spill loads
       - kept branchless baseline: `168 / 192 / 376 / 472`
   - PTX result:
     - the same `shfl.sync.idx` sequence was still present in the dumped PTX
   - conclusion:
     - reverted immediately
     - simple mask-mode helper splitting does not remove the surviving `idx` reduction from the must-win specialization

56. Screened splitting the outer exact softmax wrapper into explicit masked and unmasked entrypoints.
   - change:
     - added a separate `online_softmax_with_quant_pv_exact_unmasked(...)` wrapper
     - re-added an unmasked group-range helper
     - switched the exact producer call site to:
       - masked path -> existing generic exact wrapper
       - unmasked path -> new unmasked exact wrapper
   - backend result from direct `ptxas -v`:
     - same regression as the recent mask-splitting screens:
       - screened variant: `168` registers, `200` bytes stack frame, `376` bytes spill stores, `472` bytes spill loads
       - kept branchless baseline: `168 / 192 / 376 / 472`
   - PTX result:
     - the exact same `shfl.sync.idx` block remained at the must-win reduction site
     - so the masked peer-slot reduction survives even across an explicit outer exact masked/unmasked wrapper split
   - conclusion:
     - reverted immediately
     - the surviving `idx` reduction is coming from a deeper lowering path than either inner-helper or outer-wrapper source splitting

57. Screened splitting the exact producer call site into explicit literal `True` / `False` calls.
   - change:
     - changed the exact producer call site to:
       - masked path -> `use_masked_exp_emu=True`
       - unmasked path -> `use_masked_exp_emu=False`
     - kept the same exact wrapper body otherwise
   - result:
     - failed before PTX emission with:
       - `DSLRuntimeError: range_constexpr should be preprocessed by preprocessor`
   - conclusion:
     - reverted immediately
     - this literal-bool producer split is not viable in the current DSL lowering path

58. Screened a debug-free production exact wrapper.
   - change:
     - added a separate production-only exact wrapper with the debug pack-compare path removed
     - switched the exact producer to call:
       - debug env on -> original exact wrapper
       - debug env off -> new production wrapper
   - result:
     - failed before PTX emission with the same error:
       - `DSLRuntimeError: range_constexpr should be preprocessed by preprocessor`
   - conclusion:
     - reverted immediately
     - the debug-path isolation idea is still plausible conceptually, but this direct wrapper split is not viable in the current DSL form

59. Screened rewriting the masked peer-slot reducer to a `bfly`-only segmented reduction.
   - change:
     - rewrote `reduce_fp4_pv_group_amax_masked(...)` as a five-step `shuffle_sync_bfly(...)` reduction over offsets `1, 2, 4, 8, 16`
     - also tried the same reducer under `@cute.jit`
   - result:
     - all variants failed before PTX emission with the same DSL lowering error:
       - `DSLRuntimeError: range_constexpr should be preprocessed by preprocessor`
   - conclusion:
     - reverted immediately
     - the masked-reduction specialization target is still correct, but this direct `bfly` rewrite is not viable in the current DSL lowering path

60. Restored the exact helper stack to a valid JIT shape and re-screened the masked reducer as a `bfly`-only segmented reduction.
   - prerequisite fix:
     - restored the helper stack so the exact grouped helpers are JIT-compiled again:
       - `online_softmax_with_quant_pv_exact_group_range(...)`
       - `online_softmax_with_quant_pv_exact_group_range_dense_d128(...)`
     - removed a stray duplicate `@cute.jit` on `reduce_fp4_pv_group_amax_masked(...)`
   - kept change:
     - rewrote `reduce_fp4_pv_group_amax_masked(...)` from the full `32`-lane `shuffle_sync(idx)` scan to a five-step segmented `shuffle_sync_bfly(...)` reduction over offsets `1, 2, 4, 8, 16`
   - PTX / backend result:
     - the hot masked peer-slot reduction block collapsed from the large `shfl.sync.idx` ladder to `bfly`
     - direct `ptxas -v` improved substantially:
       - prior restored baseline: `168` registers, `200` byte stack frame, `376` spill stores, `472` spill loads
       - kept `bfly` reducer: `168` registers, `152` byte stack frame, `324` spill stores, `420` spill loads
     - `shfl.sync.idx.b32` count in the dumped PTX dropped to `17`
   - direct runtime validation:
     - constant-`V` exact oracle: finite output and `LSE`, `max_err = 0.0546875`
     - colwise `V`-scale probe: passes the same `assert_close(atol=2e-1, rtol=5e-2)` check as the focused test
   - must-win benchmark:
     - direct non-dump benchmark run on the same visible GPU completed cleanly:
       - `qkfast_ms = 0.12918`
       - `pv_fused_ms = 0.14483`
       - `pv_fused_over_qkfast = 1.121`
   - conclusion:
     - kept
     - this is the first recent change that materially improves both backend pressure and the must-win ratio

61. Re-screened the dedicated dense-unmasked `d128` helper route on top of the kept masked-`bfly` reducer baseline.
   - change:
     - routed the exact producer to `online_softmax_with_quant_pv_exact_group_range_dense_d128(...)` for the noncausal `d128` path
     - updated that helper to use the current branchless group-max and asm row-sum / pack helpers
   - backend result from direct `ptxas -v`:
     - regressed versus the kept masked-`bfly` baseline:
       - screened variant: `168` registers, `176` byte stack frame, `340` spill stores, `444` spill loads
       - kept baseline: `168 / 152 / 324 / 420`
   - conclusion:
     - reverted immediately
     - the dedicated dense-helper split is still not a win, even with the current branchless / asm primitives

62. Re-screened `FLASH_ATTN_FP4_FORCE_REGS_SOFTMAX` on the kept masked-`bfly` reducer baseline.
   - must-win row:
     - `d128`, `S=512`, `batch=2`, noncausal, `compare-mode=fused-only`
   - direct benchmark results:
     - `168`: `pv_fused_over_qkfast = 1.171`
     - `176`: `1.142`
     - `184`: `1.273`
   - conclusion:
     - no forced softmax cap beats the kept baseline run at `1.121`
     - keep the default softmax register target unchanged

63. Re-screened `FLASH_ATTN_FP4_FORCE_REGS_CORRECTION` on the kept masked-`bfly` reducer baseline.
   - must-win row:
     - `d128`, `S=512`, `batch=2`, noncausal, `compare-mode=fused-only`
   - direct benchmark results:
     - `64`: `pv_fused_over_qkfast = 1.119`
     - `72`: `1.119`
     - `80`: `1.272`
   - conclusion:
     - `64` and `72` are effectively tied within noise
     - no compelling reason to change the baked correction-register default from `72`

64. Re-screened `FLASH_ATTN_FP4_FORCE_REGS_OTHER` on the kept masked-`bfly` reducer baseline.
   - must-win row:
     - `d128`, `S=512`, `batch=2`, noncausal, `compare-mode=fused-only`
   - direct benchmark results:
     - `40`: `pv_fused_over_qkfast = 1.215`
     - `48`: `1.165`
     - `56`: `1.320`
     - `64`: `1.292`
   - conclusion:
     - decisively negative sweep
     - keep the baked `num_regs_other` default unchanged

65. Screened a packed-`i64` dense helper that merges `row_sum_delta` and `packed_word`.
   - change:
     - added a single dense `d128` asm helper returning:
       - low 32 bits: packed `P` word
       - high 32 bits: `row_sum_delta` bits
     - switched the generic dense fast path to:
       - unpack the helper result
       - use `exact_d128_dense_scale_u8(group_max_safe)` for the scale byte
   - backend result from direct `ptxas -v`:
     - exactly neutral versus the kept masked-`bfly` baseline:
       - screened variant: `168 / 152 / 324 / 420`
       - kept baseline: `168 / 152 / 324 / 420`
   - conclusion:
     - reverted immediately
     - helper fusion does not change the lowered backend shape here

66. Screened combined softmax/correction register caps on the kept masked-`bfly` reducer baseline.
   - must-win row:
     - `d128`, `S=512`, `batch=2`, noncausal, `compare-mode=fused-only`
   - same-run results:
     - base: `pv_fused_over_qkfast = 1.147`
     - `soft176 + corr64`: `1.282`
     - `soft176 + corr72`: `1.296`
     - `soft168 + corr64`: `1.171`
   - conclusion:
     - no combined cap beat the in-process base
     - this direction is not worth baking into the default heuristics

67. Screened a fused dense `scale+pack` helper returning one `i64`.
   - change:
     - attempted to fuse the dense `d128` fast-path scale-byte conversion and packed-word generation into one helper result
     - wired the generic dense producer branch to unpack:
       - low 32 bits: packed `P` word
       - high 32 bits: scale byte payload
   - result:
     - compile failed repeatedly in `ptxas` with `Arguments mismatch for instruction 'mov'`
     - the helper did not produce a runnable kernel, so there is no backend or timing result to compare
   - conclusion:
     - reverted immediately
     - avoid this `mov.b64`-style fused return shape for the dense exact pack path

68. Screened moving the dead `p0..p7` temporaries into the generic slow path only.
   - change:
     - removed the eager `p0..p7 = 0` setup from the top of the exact group body
     - created those temporaries only inside the masked/generic branch, leaving the dense `d128` fast path to use only:
       - `exact_d128_dense_row_sum_delta(...)`
       - `pack_exact_d128_dense_word(...)`
       - `float_to_ue4m3_byte(scale_f32)`
   - backend result:
     - exactly neutral versus the kept masked-`bfly` baseline after normalizing the dumped PTX:
       - screened variant: `168 / 152 / 324 / 420`
       - kept baseline: `168 / 152 / 324 / 420`
   - runtime result:
     - direct must-win run on device `2`: `pv_fused_over_qkfast = 1.249`
   - conclusion:
     - reverted immediately
     - the remaining late pack-cluster pressure is not coming from those eager temporary initializations

69. Screened dense fast-path reordering as `pack -> row_sum -> scale`.
   - change:
     - reordered the dense `d128` fast path from:
       - `row_sum_new += exact_d128_dense_row_sum_delta(...)`
       - `packed_word = pack_exact_d128_dense_word(...)`
       - `scale_u8 = float_to_ue4m3_byte(scale_f32)`
     - to:
       - `packed_word = ...`
       - `row_sum_new += ...`
       - `scale_u8 = ...`
   - backend result:
     - exactly neutral versus the kept masked-`bfly` baseline:
       - screened variant: `168 / 152 / 324 / 420`
       - kept baseline: `168 / 152 / 324 / 420`
   - runtime result:
     - direct must-win run on device `2`: `pv_fused_over_qkfast = 1.283`
   - conclusion:
     - reverted immediately
     - simple dense fast-path scheduling changes are not enough to move the remaining pack-cluster cost

70. Screened a dense pack-helper rewrite that reuses the tail base constant.
   - change:
     - rewrote `pack_exact_d128_dense_word(...)` so the tail register first holds:
       - `pt = log2_fp4_max - group_max_safe`
     - then uses:
       - `tmp = v_i + pt`
       - `ex2(tmp)` for `p0..p3`
       - `ex2(pt)` for the repeated tail value
     - this removes the repeated `sub + add` sequence against `$5` in source
   - backend result:
     - regressed versus the kept masked-`bfly` baseline:
       - screened variant: `168` registers, `160` byte stack frame, `328` spill stores, `424` spill loads
       - kept baseline: `168 / 152 / 324 / 420`
   - runtime result:
     - direct must-win run on device `2`: `pv_fused_over_qkfast = 1.276`
   - conclusion:
     - reverted immediately
     - the current pack helper already lowers better than this base-constant reuse rewrite

71. Screened a streamed dense row-sum helper.
   - change:
     - rewrote `exact_d128_dense_row_sum_delta(...)` to stream through one temporary:
       - `ex2(v0) -> acc`
       - `ex2(v1..v3) -> tmp`
       - `acc += tmp` after each step
     - this replaces the current helper's parallel `e0..e3` materialization
   - backend result:
     - exactly neutral versus the kept masked-`bfly` baseline:
       - screened variant: `168 / 152 / 324 / 420`
       - kept baseline: `168 / 152 / 324 / 420`
   - runtime result:
     - direct must-win run on device `2`: `pv_fused_over_qkfast = 1.270`
   - conclusion:
     - reverted immediately
     - the remaining late dense back-half cost is not improved by source-level streaming inside the row-sum helper

72. Screened a pairwise dense pack helper.
   - change:
     - rewrote `pack_exact_d128_dense_word(...)` to convert the live dense values as two direct pairs:
       - `v0/v1 -> byte0`
       - `v2/v3 -> byte1`
     - this avoids materializing `p0..p3` simultaneously and reuses only `lo/hi/tmp/pt` inside the asm block
   - backend result:
     - exactly neutral versus the kept masked-`bfly` baseline:
       - screened variant: `168 / 152 / 324 / 420`
       - kept baseline: `168 / 152 / 324 / 420`
   - runtime result:
     - direct must-win run on device `2`: `pv_fused_over_qkfast = 1.263`
   - conclusion:
     - reverted immediately
     - reducing the pack helper's visible float temporaries at source level does not move the remaining backend shape

73. Screened swapping the dense exact store order to `packed_word` before `scale_u8`.
   - change:
     - kept the current dense `d128` math and helpers unchanged
     - changed only the final store order in the exact group body from:
       - `sSFP_logical_u8[...] = scale_u8`
       - `tSrP_r2t_words[group_idx] = packed_word`
     - to:
       - `tSrP_r2t_words[group_idx] = packed_word`
       - `sSFP_logical_u8[...] = scale_u8`
   - backend result:
     - exactly neutral versus the kept masked-`bfly` baseline:
       - screened variant: `168 / 152 / 324 / 420`
       - kept baseline: `168 / 152 / 324 / 420`
   - runtime result:
     - direct must-win run on device `2`: `pv_fused_over_qkfast = 1.262`
   - conclusion:
     - reverted immediately
     - simple store ordering does not improve the remaining dense `d128` pack/scale cluster

74. Screened dense fast-path scheduling as `row_sum -> scale -> pack`.
   - change:
     - kept the dense `d128` helpers unchanged
     - changed only the dense fast-path order from:
       - `row_sum`
       - `pack`
       - `scale`
     - to:
       - `row_sum`
       - `scale`
       - `pack`
   - backend result:
     - regressed versus the kept masked-`bfly` baseline:
       - screened variant: `168` registers, `160` byte stack frame, `328` spill stores, `424` spill loads
       - kept baseline: `168 / 152 / 324 / 420`
   - runtime result:
     - direct must-win run on device `2`: `pv_fused_over_qkfast = 1.279`
   - conclusion:
     - reverted immediately
     - order-only rewrites around the dense `d128` fast path are now exhausted

75. Screened marking the dense pack asm as side-effectful.
   - change:
     - changed `pack_exact_d128_dense_word(...)` from `has_side_effects=False` to `True`
     - intent was to constrain backend motion of the dense pack cluster across neighboring group work
   - backend result:
     - exactly neutral versus the kept masked-`bfly` baseline:
       - screened variant: `168 / 152 / 324 / 420`
       - kept baseline: `168 / 152 / 324 / 420`
   - runtime result:
     - direct must-win run on device `2`: `pv_fused_over_qkfast = 1.284`
   - conclusion:
     - reverted immediately
     - anchoring the pack asm with side effects does not improve the lowered dense back-half shape

76. Screened marking the dense row-sum asm as side-effectful.
   - change:
     - changed `exact_d128_dense_row_sum_delta(...)` from `has_side_effects=False` to `True`
     - intent was to prevent backend motion of the row-sum `MUFU.EX2` cluster across neighboring group work
   - backend result:
     - exactly neutral versus the kept masked-`bfly` baseline:
       - screened variant: `168 / 152 / 324 / 420`
       - kept baseline: `168 / 152 / 324 / 420`
   - runtime result:
     - direct must-win run on device `2`: `pv_fused_over_qkfast = 1.289`
   - conclusion:
     - reverted immediately
     - side-effectful asm barriers are not enough to move the remaining dense `d128` pack/scale cluster

77. Screened a fused dense helper returning `{packed_word, scale_bits}` as separate asm outputs.
   - change:
     - added `exact_d128_dense_pack_and_scale(...)` returning:
       - output 0: dense packed `P` word
       - output 1: dense `E4M3` scale bits
     - used a real multi-output LLVM struct return instead of the earlier failed `mov.b64` / `i64` return path
     - dense fast path order:
       - `row_sum_new += exact_d128_dense_row_sum_delta(...)`
       - `packed_word, scale_u8 = exact_d128_dense_pack_and_scale(...)`
   - note:
     - the first compile failure on this helper was a simple asm operand-numbering bug after adding a second output; fixed before measuring the final variant
   - backend result:
     - exactly neutral versus the kept masked-`bfly` baseline:
       - screened variant: `168 / 152 / 324 / 420`
       - kept baseline: `168 / 152 / 324 / 420`
   - runtime result:
     - direct must-win run on device `2`: `pv_fused_over_qkfast = 1.243`
   - conclusion:
     - reverted immediately
     - fusing dense pack and scale into a struct-return helper does not improve the lowered backend shape in this order

78. Re-screened the same fused dense `{pack, scale}` helper with order flipped to `{pack,scale} -> row_sum`.
   - change:
     - kept the same `exact_d128_dense_pack_and_scale(...)` helper
     - changed only the dense fast-path order to:
       - `packed_word, scale_u8 = ...`
       - `row_sum_new += exact_d128_dense_row_sum_delta(...)`
   - backend result:
     - regressed versus the kept masked-`bfly` baseline:
       - screened variant: `168` registers, `168` byte stack frame, `336` spill stores, `432` spill loads
       - kept baseline: `168 / 152 / 324 / 420`
   - runtime result:
     - direct must-win run on device `2`: `pv_fused_over_qkfast = 1.254`
   - conclusion:
     - reverted immediately
     - this closes out the fused struct-return dense helper path

79. Screened a direct shared-store dense scale helper.
   - change:
     - added `exact_d128_dense_store_scale_u8(...)` using:
       - `utils.elem_pointer(sSFP_logical_u8, (row, col, 0))`
       - inline PTX `st.shared.u8`
     - dense fast path became:
       - `row_sum_new += exact_d128_dense_row_sum_delta(...)`
       - `packed_word = pack_exact_d128_dense_word(...)`
       - direct `st.shared.u8` for the dense `E4M3` scale byte
     - the final generic `sSFP_logical_u8[...] = scale_u8` store was skipped on the nonmasked dense branch
   - backend result:
     - regressed versus the kept masked-`bfly` baseline:
       - screened variant: `168` registers, `160` byte stack frame, `328` spill stores, `424` spill loads
       - kept baseline: `168 / 152 / 324 / 420`
   - runtime result:
     - direct must-win run on device `2`: `pv_fused_over_qkfast = 1.299`
   - conclusion:
     - reverted immediately
     - this closes out the clean direct `st.shared.u8` dense-scale-store route

80. Screened a fused dense helper returning `{row_sum_delta, scale_bits}`.
   - change:
     - added `exact_d128_dense_row_sum_and_scale(...)` returning:
       - output 0: dense `row_sum_delta`
       - output 1: dense `E4M3` scale bits
     - dense fast path became:
       - `row_sum_delta, scale_u8 = exact_d128_dense_row_sum_and_scale(...)`
       - `row_sum_new += row_sum_delta`
       - `packed_word = pack_exact_d128_dense_word(...)`
   - backend result:
     - regressed versus the kept masked-`bfly` baseline:
       - screened variant: `168` registers, `160` byte stack frame, `328` spill stores, `424` spill loads
       - kept baseline: `168 / 152 / 324 / 420`
   - runtime result:
     - direct must-win run on device `2`: `pv_fused_over_qkfast = 1.253`
   - conclusion:
     - reverted immediately
     - this closes out the clean fused `{row_sum, scale}` dense-helper path

81. Screened moving only the generic masked/slow exact branch into a helper.
   - change:
     - added `exact_fp4_pv_group_row_sum_pack_scale_generic(...)` for the non-dense exact path
     - kept the dense `d128` fast path unchanged
     - parent exact group body no longer carried the generic branch's `e0..e7`, `p0..p7`, and pack/scale expressions directly
   - backend result:
     - exactly neutral versus the kept masked-`bfly` baseline:
       - screened variant: `168 / 152 / 324 / 420`
       - kept baseline: `168 / 152 / 324 / 420`
   - runtime result:
     - direct must-win run on device `2`: `pv_fused_over_qkfast = 1.248`
   - conclusion:
     - reverted immediately
     - isolating only the generic branch does not improve the hot dense `d128` path on the current baseline

82. Screened marking the dense branchless group-max asm as side-effectful.
   - change:
     - changed `exact_d128_dense_group_max_branchless(...)` from `has_side_effects=False` to `True`
     - intent was to stop the backend from hoisting multiple dense group-max results ahead of the late pack/scale cluster
   - backend result:
     - exactly neutral versus the kept masked-`bfly` baseline:
       - screened variant: `168 / 152 / 324 / 420`
       - kept baseline: `168 / 152 / 324 / 420`
   - runtime result:
     - direct must-win run on device `2`: `pv_fused_over_qkfast = 1.296`
   - conclusion:
     - reverted immediately
     - anchoring the branchless group-max asm does not improve the dense `d128` back-half lowering

83. Screened returning the dense scale byte as `i8` directly from asm.
   - change:
     - rewrote `exact_d128_dense_scale_u8(...)` to return `T.i8()` directly from inline asm instead of:
       - returning `i16`
       - truncating to `i8` in LLVM
     - switched the dense fast path to use `exact_d128_dense_scale_u8(group_max_safe)`
   - result:
     - compile failed repeatedly in `ptxas` with `Unexpected instruction types specified for 'mov'`
     - no backend or runtime result was produced
   - conclusion:
     - reverted immediately
     - direct `i8` asm return is not a viable route for this dense scale helper on the current toolchain

84. Screened a fused dense helper that returns the packed word and stores the scale byte directly to shared.
   - change:
     - added `exact_d128_dense_pack_store_scale(...)` taking:
       - shared pointer for `sSFP_logical_u8[row, col, 0]`
       - `v0..v3`
       - `group_max_safe`
     - helper:
       - computes the dense packed `P` word
       - computes the dense `E4M3` scale byte
       - performs `st.shared.u8` internally
       - returns only the packed word
     - dense fast path became:
       - `row_sum_new += exact_d128_dense_row_sum_delta(...)`
       - `packed_word = exact_d128_dense_pack_store_scale(...)`
       - no separate dense `scale_u8` value in the fast path
   - backend result:
     - regressed versus the kept masked-`bfly` baseline:
       - screened variant: `168` registers, `160` byte stack frame, `328` spill stores, `424` spill loads
       - kept baseline: `168 / 152 / 324 / 420`
   - runtime result:
     - direct must-win run on device `2`: `pv_fused_over_qkfast = 1.305`
   - conclusion:
     - reverted immediately
     - this closes out the last clean fusion of the dense pack and scale-store paths

85. Screened an all-in-one dense helper that returns `{row_sum_delta, packed_word}` and stores the scale byte directly to shared.
   - change:
     - added `exact_d128_dense_row_sum_pack_store_scale(...)` taking:
       - shared pointer for `sSFP_logical_u8[row, col, 0]`
       - dense fast-path logits `v0..v3`
       - `group_max_safe`
     - helper:
       - computes dense `row_sum_delta`
       - computes the dense packed `P` word
       - computes the dense `E4M3` scale byte
       - performs `st.shared.u8` internally
       - returns only `{row_sum_delta, packed_word}`
     - dense fast path became:
       - `row_sum_delta, packed_word = exact_d128_dense_row_sum_pack_store_scale(...)`
       - `row_sum_new += row_sum_delta`
       - no separate dense `scale_u8`
       - outer `sSFP_logical_u8[...] = scale_u8` ran only for the masked dense case
   - backend result:
     - exactly neutral versus the kept masked-`bfly` baseline:
       - screened variant: `168 / 152 / 324 / 420`
       - kept baseline: `168 / 152 / 324 / 420`
   - runtime result:
     - direct must-win run on device `2`: `pv_fused_over_qkfast = 1.271`
   - conclusion:
     - reverted immediately
     - full dense fusion of `{row_sum, pack, scale-store}` is also a dead end on the current lowering

86. Screened moving the masked slot reducer itself into inline asm.
   - change:
     - added `reduce_fp4_pv_group_amax_masked_branchless(...)` as a `dsl_user_op`
     - kept the same `bfly` offsets `1, 2, 4, 8, 16`
     - compared `slot_ptr` in asm and accumulated `has_masked_peer` with predicates
     - `reduce_fp4_pv_group_amax_masked(...)` became a thin wrapper around the asm helper
   - backend result:
     - regressed badly versus the kept masked-`bfly` baseline:
       - screened variant: `168` registers, `192` byte stack frame, `364` spill stores, `460` spill loads
       - kept baseline: `168 / 152 / 324 / 420`
   - runtime result:
     - direct must-win run on device `2`: `pv_fused_over_qkfast = 1.298`
   - conclusion:
     - reverted immediately
     - the current masked-`bfly` reducer is better left in DSL form; moving it into inline asm makes the backend shape much worse

87. Screened a full dense-group asm helper for the exact `d128` fast path.
   - change:
     - added `exact_d128_dense_group_all_in_one(...)`
     - helper took:
       - shared pointer for `sSFP_logical_u8[row, col, 0]`
       - `row`
       - dense fast-path logits `v0..v3`
     - helper performed inside one asm block:
       - branchless dense group-max reduction
       - dense row-sum delta
       - dense packed `P` word
       - dense `E4M3` scale byte store to shared
     - dense fast path became:
       - `row_sum_delta, packed_word = exact_d128_dense_group_all_in_one(...)`
       - `row_sum_new += row_sum_delta`
       - no separate dense `group_max`, `pack`, or `scale_u8` path outside the helper
   - backend result:
     - improved versus the kept masked-`bfly` baseline:
       - screened variant: `168` registers, `144` byte stack frame, `308` spill stores, `400` spill loads
       - kept baseline: `168 / 152 / 324 / 420`
   - runtime result:
     - direct must-win run on device `2`: `pv_fused_over_qkfast = 2.386`
   - conclusion:
     - reverted immediately
     - this is a real backend/runtime divergence case: PTXAS improved materially, but end-to-end kernel time collapsed, so the helper is not viable

88. Screened a full dense-group asm helper that returns `{row_sum_delta, packed_word, scale_bits}`.
   - change:
     - added `exact_d128_dense_group_row_sum_pack_scale(...)`
     - helper kept dense `group_max` internal, but returned the scale byte bits instead of storing shared memory inside the asm block
     - dense fast path became:
       - `row_sum_delta, packed_word, scale_u8 = exact_d128_dense_group_row_sum_pack_scale(...)`
       - `row_sum_new += row_sum_delta`
       - outer shared scale store stayed in the normal path
   - result:
     - PTX emitted, but direct `ptxas -v` failed with repeated:
       - `Arguments mismatch for instruction 'max'`
       - `Arguments mismatch for instruction 'ex2'`
       - `Arguments mismatch for instruction 'sub'`
     - no backend or runtime result was produced
   - conclusion:
     - reverted immediately
     - multi-output dense all-in-one helper is not viable on the current toolchain

89. Re-screened dense exact chunking as `2 x 16` on the kept masked-`bfly` baseline.
   - change:
     - for the `num_groups == 32` exact path, changed:
       - `0..7, 8..15, 16..23, 24..31`
     - to:
       - `0..15, 16..31`
     - applied in both the main exact path and the producer exact path
   - backend result:
     - exactly neutral versus the kept masked-`bfly` baseline:
       - screened variant: `168 / 152 / 324 / 420`
       - kept baseline: `168 / 152 / 324 / 420`
   - runtime result:
     - direct must-win run on device `2`: `pv_fused_over_qkfast = 1.226`
   - conclusion:
     - reverted immediately
     - repartitioning the exact 32-group tile to `2 x 16` does not improve the current baseline

90. Screened removing `make_warp_uniform(...)` from the block-scaled partial MMA helper.
   - change:
     - in `blackwell_helpers.py`, changed `gemm_ptx_fp4_block_scaled_partial(...)` to pass:
       - `tA_addr`
       - `smem_desc_start_b_lo`
       - `acc_tmem_addr`
       - `tmem_sa_addr`
       - `tmem_sb_addr`
     - directly into the inline asm, instead of wrapping each with `cute.arch.make_warp_uniform(...)`
   - PTX effect:
     - the later descriptor/control path did simplify
     - visible `shfl.sync.idx` sites in the must-win PTX dropped from the earlier broader set down to:
       - `88`
       - `1140`
       - `1157`
       - `1159`
       - `1161`
       - `1254`
       - `2374`
   - backend result:
     - exactly neutral versus the kept masked-`bfly` baseline:
       - screened variant: `168 / 152 / 324 / 420`
       - kept baseline: `168 / 152 / 324 / 420`
   - runtime result:
     - direct must-win run on device `2`: `pv_fused_over_qkfast = 1.277`
   - conclusion:
     - reverted immediately
     - reducing the descriptor-path `shfl.sync.idx` count alone is not enough; without a PTXAS move, the kernel still regresses at runtime

91. Screened removing `make_warp_uniform(...)` from the full FP4 block-scaled MMA helper.
   - change:
     - in `blackwell_helpers.py`, changed `gemm_ptx_fp4_block_scaled(...)` to pass:
       - `smem_desc_start_b_lo`
       - `acc_tmem_addr`
       - `tmem_sa_addr`
       - `tmem_sb_addr`
     - directly into the inline asm, instead of wrapping each with `cute.arch.make_warp_uniform(...)`
   - PTX effect:
     - the later descriptor/control path changed again
     - visible `shfl.sync.idx` sites in the must-win PTX became:
       - `88`
       - `1140`
       - `1157`
       - `1159`
       - `1161`
       - `1163`
       - `1164`
       - `1165`
       - `1166`
       - `1167`
       - `2275`
       - `3395`
   - backend result:
     - exactly neutral versus the kept masked-`bfly` baseline:
       - screened variant: `168 / 152 / 324 / 420`
       - kept baseline: `168 / 152 / 324 / 420`
   - runtime result:
     - direct must-win run on device `2`: `pv_fused_over_qkfast = 1.277`
   - conclusion:
     - reverted immediately
     - reducing the full-helper descriptor broadcasts also does not help unless PTXAS actually moves

92. Screened literal TMEM-offset specialization in the FP4 tcgen05 helpers.
   - change:
     - in `fp4_flash_fwd_sm100_pvfused.py`, changed the PV fused helper construction to pass plain Python ints for:
       - `tmem_sa_addr`
       - `tmem_sb_addr`
     - in `blackwell_helpers.py`, changed both:
       - `gemm_ptx_fp4_block_scaled(...)`
       - `gemm_ptx_fp4_block_scaled_partial(...)`
     - to embed:
       - `tmem_acc`
       - `tmem_sa`
       - `tmem_sb`
     - as literal `mov.b32` immediates inside the tcgen05 asm, instead of register operands
   - PTX effect:
     - this was the strongest descriptor-path cleanup so far
     - visible `shfl.sync.idx` sites in the must-win PTX dropped to:
       - `88`
       - `1140`
       - `1157`
       - `1158`
       - `1251`
       - `2265`
       - `2368`
       - `3384`
   - backend result:
     - exactly neutral versus the kept masked-`bfly` baseline:
       - screened variant: `168 / 152 / 324 / 420`
       - kept baseline: `168 / 152 / 324 / 420`
   - runtime result:
     - direct must-win run on device `2`: `pv_fused_over_qkfast = 1.252`
   - conclusion:
     - reverted immediately
     - even the strongest source-side cleanup of the late descriptor broadcasts does not improve the kernel unless it also changes PTXAS

93. Screened the runtime `2CTA` override on the must-win row.
   - change:
     - kept the kernel source unchanged
     - benchmark env:
       - `FLASH_ATTN_FP4_FORCE_2CTA=1`
       - `FLASH_ATTN_FP4_PV_EXACT_SFV_DIRECT=1`
   - runtime result:
     - benchmark harness never produced a single clean fresh-process sample
     - after `10` attempts, the last failure was:
       - `RuntimeError: CUDA Error: cudaErrorInvalidValue`
   - conclusion:
     - `2CTA=1` is not viable for the exact PV must-win row in the current bringup
     - no code change kept

94. Screened the runtime `q_stage=2` override on the must-win row.
   - change:
     - kept the kernel source unchanged
     - benchmark env:
       - `FLASH_ATTN_FP4_FORCE_Q_STAGE=2`
       - `FLASH_ATTN_FP4_PV_EXACT_SFV_DIRECT=1`
   - runtime result:
     - benchmark harness never produced a single clean fresh-process sample
     - after `10` attempts, the last failure was:
       - `AssertionError` at:
         - `assert self.tmem_total <= self.tmem_alloc_cols`
   - conclusion:
     - `q_stage=2` exceeds the exact PV TMEM budget on this row
     - no code change kept

95. Mapped the remaining visible `shfl.sync.idx` sites in the kept baseline PTX.
   - setup:
     - used the kept-baseline dump in:
       - `/tmp/fa_ptx_kept_baseline/pv_clean.ptx`
       - `/tmp/fa_ptx_kept_baseline/pv.sass`
   - remaining PTX sites:
     - `1140`
     - `1157`
     - `1159`
     - `1161`
     - `1163`
     - `1165`
     - `1167`
     - `1169`
     - `1170`
     - `1171`
     - `1172`
     - `1173`
     - `1266`
     - `2280`
     - `2383`
     - `3399`
   - mapping:
     - `1140`:
       - broadcast of the Q-side shared descriptor low bits before the QK MMA descriptor construction
     - `1157..1173`:
       - broadcast of the compile-time TMEM offsets:
         - `0`
         - `384`
         - `400`
         - `112`
         - `256`
         - `416`
         - `432`
       - these correspond to the fused PV TMEM base offsets used later in the block-scaled MMA/control path
     - `1266`:
       - broadcast of the compact scale-copy shared descriptor before `tcgen05.cp.cta_group::1.32x128b.warpx4`
     - `2280`, `2383`, `3399`:
       - broadcast of shared descriptor low bits immediately ahead of later block-scaled FP4 MMA issue sites
       - these are in the late TMEM/descriptor control path, not the dense exact pack helper
   - conclusion:
     - the remaining visible `shfl.sync.idx` sites are descriptor/control broadcasts
     - the dense exact pack helper is no longer where the visible `idx` traffic lives
     - further source-level work should not revisit local dense pack scheduling unless it also changes the later TMEM/descriptor path

96. Screened replacing the PV `SFV` `s2t` copy with the direct `tcgen05.cp` helper style.
   - change:
     - in `fp4_flash_fwd_sm100_pvfused.py`, replaced the two live:
       - `cute.copy(tiled_copy_s2t_sfv, ...)`
     - call sites with:
       - `sm100_utils.copy_scale_smem_to_tmem(Int32(self.tmem_sfv_offset), ..., cta_group=self.cta_group_size)`
   - first lowering failure:
     - passing the existing `tCsSFV_compact_s2t[(...)]` object directly failed in:
       - `Int64(smem_desc)`
     - error:
       - `DSLRuntimeError: tensor<... cute_nvgpu.smem_desc ...> to integer conversion is not supported`
   - second lowering failure:
     - retrying with:
       - `tCsSFV_compact_s2t[(...)].iterator.toint()`
     - failed because the indexed descriptor view is an MLIR `OpResult`, not a pointer-like iterator object
     - error:
       - `AttributeError: 'cutlass._mlir._mlir_libs._cutlass_ir._mlir.ir.OpResult' object has no attribute 'toint'`
   - third lowering failure:
     - patched `copy_scale_smem_to_tmem(...)` in `blackwell_helpers.py` to accept a non-`int` descriptor input directly and re-ran the same screen
     - inline-asm argument lowering still failed before PTX emission
     - error:
       - `AssertionError` from `get_op_result_or_value`, because the PV `smem_desc_view` object is not accepted as a direct inline-asm operand either
   - conclusion:
     - this route is blocked by the PV `s2t` source descriptor type
     - unlike the qkfast helper path, the PV `tCsSFV_compact_s2t[(...)]` object is a `cute_nvgpu.smem_desc_view`, not a plain `i64` descriptor value
     - reverted immediately

97. Probed the raw PV descriptor operand types directly at the CuTe / MLIR boundary.
   - setup:
     - inspected the live `PV` `s2t` source descriptor objects and their iterator/value forms
     - tested the direct inline-asm route against both:
       - `smem_desc_view`
       - `smem_desc`
   - result:
     - inline asm does not accept either operand type directly
     - the `smem_desc_view` value form fails verification because it is not LLVM-dialect compatible
     - the `smem_desc` iterator form also fails verification because it is not LLVM-dialect compatible either
   - conclusion:
     - the inline-asm `tcgen05.cp` route is blocked at the operand-type level for PV `SFV`
     - this is not just a missing cast; the exposed CuTe PV descriptor objects are not legal inline-asm operands

98. Screened a native `nvvm.tcgen05_cp` route for PV `SFV`.
   - change:
     - added a temporary helper in `blackwell_helpers.py` that called:
       - `nvvm.tcgen05_cp(shape=..., taddr=..., smem_desc=...)`
     - rewired the two live PV `SFV` call sites in `fp4_flash_fwd_sm100_pvfused.py` to use it
   - intermediate failures:
     - passing the raw wrapper objects hit the Python ODS boundary:
       - `AssertionError` from `get_op_result_or_value`
     - after narrowing the call to the raw TMEM pointer value and leaving the source descriptor as the CuTe `smem_desc` value, the op reached IR verification
   - final verifier failure:
     - `'nvvm.tcgen05.cp' op operand #0 must be LLVM pointer in address space 6, but got '!cute.ptr<f8E4M3FN, tmem, align<16>>'`
   - benchmark result:
     - the must-win compile screen produced `0` successful fresh-process samples after `10` attempts because every compile hit the same verifier error
   - conclusion:
     - native `nvvm.tcgen05_cp` accepts the PV `smem_desc` operand, but it still rejects the TMEM destination exposed by CuTe
     - the destination is currently a CuTe `!cute.ptr<..., tmem>` value, not the LLVM AS6 pointer type that `nvvm.tcgen05.cp` requires
     - reverted both files immediately

99. Screened the same native `nvvm.tcgen05_cp` route with `tmem_ptr.to_llvm_ptr()`.
   - change:
     - kept the same temporary NVVM helper, but changed operand `0` from the CuTe TMEM pointer wrapper to:
       - `tmem_ptr.to_llvm_ptr()`
   - result:
     - this fixed the first verifier failure
     - `nvvm.tcgen05.cp` now accepted operand `0` as:
       - `!llvm.ptr<6>`
     - the verifier then failed on operand `1` instead:
       - `'nvvm.tcgen05.cp' op operand #1 must be 64-bit signless integer, but got '!cute_nvgpu.smem_desc'`
   - conclusion:
     - `to_llvm_ptr()` is the right direction for the TMEM destination
     - the remaining typed blocker on the native NVVM path is the PV `smem_desc -> i64` conversion

100. Screened a generic cast of the PV `smem_desc` to `i64` on top of the `to_llvm_ptr()` route.
   - change:
     - inside the temporary NVVM helper, inserted:
       - `builtin.unrealized_conversion_cast([Int64.mlir_type], [smem_desc])`
     - passed that result as operand `1` to `nvvm.tcgen05.cp`
   - result:
     - this moved the failure past IR verification
     - the compile then died in LLVM translation with unresolved conversion-cast state
     - the top-level error was:
       - `LLVM Translation failed for operation: builtin.unrealized_conversion_cast`
     - the dumped failing op showed the TMEM pointer conversion path still contained unresolved casts:
       - `builtin.unrealized_conversion_cast ... (i32) -> !cute.ptr<f8E4M3FN, tmem, align<16>>`
   - benchmark result:
     - the must-win compile screen again produced `0` successful fresh-process samples after `10` attempts
   - conclusion:
     - the native NVVM `cp` route can be made verifier-clean only by introducing generic conversion casts that do not lower cleanly in the current CuTe / LLVM pipeline
     - this is now a toolchain / lowering boundary, not a local kernel-source issue
     - reverted both files immediately

101. Probed the direct `cute_nvgpu.arch.copy.SM100.copy_s2t` op in isolated MLIR.
   - setup:
     - loaded the CuTe MLIR subpackages directly without importing the full `cutlass` Python package
     - this made it possible to inspect the generated op wrappers and builders in isolation
   - concrete findings:
     - the direct op is present:
       - `cute_nvgpu.arch.copy.SM100.copy_s2t`
     - the broadcast attr is registered and parses as:
       - `#cute_nvgpu.copy_s2t_broadcast_mode<x4>`
     - the generated Python wrapper advertises builder names for:
       - `CuteArchCopySM100CopyS2TDpAttr`
       - `CuteArchCopySM100CopyS2TBitAttr`
       - `CuteArchCopySM100CopyS2TCtaAttr`
     - but the runtime builder registry reports all three as absent:
       - `AttrBuilder.contains(...) == False`
     - the direct operand types from the earlier verifier failures parse cleanly:
       - `!cute_nvgpu.smem_desc`
       - `!cute.ptr<ui32, tmem>`
   - op-construction results:
     - building the op programmatically succeeds only if `broadcast` is supplied as the registered enum attr
     - supplying raw Python ints for `dp / bits / cta` fails immediately in the ODS wrapper:
       - `TypeError ... Invalid attribute value ... (std::bad_cast)`
     - supplying standard MLIR integer attrs also fails verifier:
       - `IntegerAttr`
       - `I32Attr`
       - `I64Attr`
       - `UI32Attr`
     - every variant fails on the same field first:
       - `op attribute 'dp' failed to satisfy constraint: arbitrary integer attribute`
   - conclusion:
     - the direct arch-copy op is exposed, but the concrete attr class needed for `dp / bits / cta` is not reachable through the current Python builder layer
     - this blocks a clean source-level swap from the current `cute.copy(...)` path to the direct `cute_nvgpu.arch.copy.SM100.copy_s2t` op
     - the remaining work on this route is in the CuTe dialect bindings / lowering layer, not in the local flash-attention kernel source

102. Added an experimental MXFP4 PV exact-lane entry point.
   - reminder from the local SageAttention3-style path:
     - the active exact lane keeps the online-softmax recurrence fused with P quantization
     - it groups the live softmax fragment, computes one FP4 scale per grouped P slot, writes packed E2M1 P words, and feeds the existing block-scaled PV MMA path
     - the checked local `SageAttention/sageattention3_blackwell` code path allocates E4M3 scale tensors with `D // 16` / `N // 16` scale storage and uses `scale_vec::4X ... ue4m3`, so the MXFP4 work here is an experimental E8M0/vec32 adaptation of that fused-flow pattern rather than a direct copy of SageAttention3 defaults
   - change:
     - added an E8M0 scale-byte helper using `cvt.rp.satfinite.ue8m0x2.f32`
     - routed PV scale-byte stores through `float_to_pv_scale_byte(...)`
     - kept NVFP4 as E4M3/vec16 and selected E8M0/vec32 when the fused lane is constructed with MXFP4
     - relaxed the exact fused lane interface gate so dense fixed-length noncausal MHA can use `fp4_qk_format="mxfp4"` with `use_fp4_pv=True`
     - passed the actual QK format into the fused PV kernel's PV scale config instead of hard-coding NVFP4
   - current scope:
     - only matching QK/PV formats are enabled:
       - NVFP4 QK + NVFP4 PV
       - MXFP4 QK + MXFP4 PV
     - mixed NVFP4/MXFP4 PV scale inputs still fail validation
     - QK-only MXFP4 remains outside this bring-up path
   - validation:
     - `python -m py_compile flash_attn/cute/fp4_flash_fwd_sm100_pvfused.py flash_attn/cute/interface.py tests/cute/test_fp4_flash_attn.py`
     - `/workspace/codebases/fp4_matmul/.venv/bin/python -m pytest -p no:rerunfailures tests/cute/test_fp4_flash_attn.py -k "fp4_pv_mxfp4_fake_compile_dense_forward or fp4_pv_fused_fake_compile_dense_forward or fp4_pv_fused_exact_lane_accepts_mxfp4_scale_config or fp4_pv_validation_errors or fp4_qk_validation_errors" -q`
     - result: `35 passed, 120 deselected`
   - runtime boundary:
     - this shell reports `torch.cuda.is_available() == False` and `device_count == 0`, so no SM100 runtime correctness or performance claim was made for MXFP4 PV in this pass
   - next step:
     - run a real SM100 MXFP4 PV compile/runtime probe and compare the exact-lane MXFP4 result against the current NVFP4 PV baseline and FA4 once a usable CUDA device is available

103. Runtime follow-up for mixed NVFP4-QK / MXFP4-PV on SM100.
   - environment:
     - usable runtime device in this pass was exposed with `CUDA_VISIBLE_DEVICES=1`
     - physical device `0` was still unreliable through CUDA even when `nvidia-smi` reported it idle
   - kept fixes:
     - restored the exact `d64` LSE second-publication pass in `correction_loop_exact_pv(...)`
       - the compact exact `d64` correction map has only `64` correction threads
       - without the second pass, rows `64..127`, `192..255`, ... retained stale LSE values
     - fixed the runtime probe helper so a parent `CUDA_VISIBLE_DEVICES` mask is preserved instead of remapping the subprocess back to physical device `0`
     - made block-scaled PTX helpers accept an explicit MMA kind
       - NVFP4 / vec16 continues to emit `kind::mxf4nvf4.block_scale.scale_vec::4X`
       - MXFP4 / vec32 now emits `kind::mxf4.block_scale.scale_vec::2X`
       - this matches the CUTLASS SM100 distinction between NVFP4-style and MXFP4-style FP4 MMA forms
   - validation:
     - `python -m py_compile flash_attn/cute/blackwell_helpers.py flash_attn/cute/blackwell_helpers_qkfast.py flash_attn/cute/fp4_flash_fwd_sm100_pvfused.py flash_attn/cute/fp4_flash_fwd_sm100_qkfast.py flash_attn/cute/fp4_flash_fwd_sm100.py flash_attn/cute/interface.py tests/cute/benchmark_fp4_pv.py tests/cute/test_fp4_flash_attn.py`
     - `/workspace/codebases/fp4_matmul/.venv/bin/python -m pytest -p no:rerunfailures tests/cute/test_fp4_flash_attn.py -k "fp4_pv_mxfp4_fake_compile_dense_forward or fp4_pv_fused_fake_compile_dense_forward or fp4_pv_fused_exact_lane_accepts_mxfp4_scale_config or fp4_pv_validation_errors or fp4_qk_validation_errors" -q`
       - result: `34 passed, 120 deselected`
     - `CUDA_VISIBLE_DEVICES=1 ... pytest -k "fp4_pv_probe_exact_lane_general_shape_lse_stays_bounded" -q`
       - result: `1 passed, 153 deselected`
   - NVFP4 runtime after restoring the LSE publication pass:
     - `d64`, `S=512`, full correctness smoke:
       - `pv_fused_out_max = 0.07861328125`
       - `pv_fused_lse_max = 0.9423937797546387`
     - this confirms the old stale-LSE row pattern is fixed again
   - current timing snapshot on this host:
     - `d64`, `S=512`, fused-only, NVFP4 PV, `fresh_runs=3`, `warmup=5`, `iters=10`
       - `qkfast_ms ~= 0.10964`
       - `pv_fused_ms ~= 0.14038`
       - `pv_fused_over_qkfast ~= 1.280`
     - this pass did not reproduce the earlier sub-`1.0` d64 timing from the summary section
     - a detached worktree at commit `750603f` was attempted for comparison, but the old benchmark path timed out during compile/runtime before producing a datapoint
   - MXFP4 PV findings:
     - mixed NVFP4-QK / MXFP4-PV now reaches real SM100 runtime with the correct `kind::mxf4` PTX form
     - fused-only timing improved versus the earlier wrong-kind screen, but remains above target:
       - before correct kind: ratio around `1.30`
       - after correct kind: ratio around `1.20`
     - correctness is not stable enough to claim a keeper:
       - some direct runs produce finite LSE with the same `0.9424` error band as NVFP4
       - repeated same-process runs can produce huge finite LSE error or NaN/Inf LSE
       - forcing MXFP4 SFP scale bytes to the E8M0 byte for `1.0` did not fix the LSE corruption
     - conclusion:
       - the remaining MXFP4 issue is not the `ue8m0` conversion instruction itself
       - the suspect boundary is vec32 scale layout / `s2t` scale movement / lower-level JIT cache identity when float8 scale tensors are passed as `uint8` storage views
       - do not treat MXFP4 PV as a performance path yet; use it only as an experimental bring-up route

104. Follow-up cleanup and MXFP4 failure classification.
   - kept cleanup:
     - removed the dead exact-only `sSFP_logical_u8_flat_exact` construction and call-threading
     - removed unused exact fragment-shape and coordinate-tensor arguments from the production `softmax_step -> softmax_step_exact_pv` call chain
     - these were documented as already gone in the status history, but the current dirty tree still had them
   - screened and reverted:
     - extended the dense `4`-live-logit helper to `d64`
     - correctness stayed bounded, but the short timing screen did not improve the ratio:
       - `qkfast_ms ~= 0.11190`
       - `pv_fused_ms ~= 0.13847`
       - `pv_fused_over_qkfast ~= 1.237`
     - reverted back to the existing `head_dim_v_padded >= 128` specialization boundary
   - MXFP4 diagnostics:
     - bypassing MX SFP conversion by forcing generated P-scale bytes to E8M0 `1.0` still produced huge LSE error:
       - representative aggregate: `pv_fused_lse_max ~= 690.7`
     - disabling the direct exact SFV loader was worse for MXFP4:
       - `0` clean fresh samples out of `5`
       - last failure reported `FP4 PV output contains NaN or Inf`
     - conclusion:
       - the MXFP4 corruption is not fixed by SFP conversion bypass
       - the direct SFV loader remains required, but the vec32/MX scale path is still not correctness-stable
   - benchmark harness fix:
     - `--skip-baseline-check` now still rejects obviously corrupted finite PV runs when BF16 is present
     - loose sanity bounds:
       - `PV_SANITY_OUT_MAX = 10.0`
       - `PV_SANITY_LSE_MAX = 10.0`
     - this prevents MXFP4 finite-but-huge LSE screens from being counted as clean successes
   - validation:
     - `python -m py_compile flash_attn/cute/fp4_flash_fwd_sm100_pvfused.py tests/cute/benchmark_fp4_pv.py`
     - `/workspace/codebases/fp4_matmul/.venv/bin/python -m pytest -p no:rerunfailures tests/cute/test_fp4_flash_attn.py -k "fp4_pv_mxfp4_fake_compile_dense_forward or fp4_pv_fused_fake_compile_dense_forward or fp4_pv_fused_exact_lane_accepts_mxfp4_scale_config or fp4_pv_validation_errors or fp4_qk_validation_errors" -q`
       - result: `34 passed, 120 deselected`
   - current performance status:
     - no current speedup versus recovered `qkfast` / FA4 has been reproduced under the batched host timer
     - recent `d64`, `S=512`, NVFP4 screens remain around `1.19x-1.28x` over `qkfast`
     - MXFP4 PV remains experimental and should not be used as the optimization baseline until the row-stat corruption is resolved

105. MXFP4 P-scale consistency pass after rechecking SageAttention3.
   - SageAttention3 reminder:
     - the local `SageAttention/sageattention3_blackwell` PV path is not MXFP4 by default
     - it uses FP4 payloads with E4M3 scale bytes, `D // 16` / `N // 16` scale storage, and `mxf4nvf4.block_scale.scale_vec::4X ... ue4m3`
     - therefore MXFP4 PV here is still an E8M0/vec32 adaptation of the Sage3 fused P-quantization pattern, not a direct Sage3 copy
   - kept code change:
     - added log2-domain E8M0 helpers for generated P scales:
       - `log2_to_ue8m0_byte_rp(...)`
       - `log2_to_ue8m0_scale_log2_rp(...)`
     - the MXFP4 exact P path now quantizes the generated scale to E8M0 first, then packs the FP4 P payload using the rounded power-of-two scale that the MMA will actually consume
     - this removes the previous mismatch where MX wrote an E8M0 scale byte but normalized P with the continuous pre-rounded `amax / 6` scale
     - NVFP4 / E4M3 behavior is intentionally left on the existing path
   - screened and not kept:
     - forcing PV to `kind::mxf4nvf4.block_scale.scale_vec::2X` was worse for the MX path
     - representative result:
       - `FP4 PV LSE contains NaN or Inf`
     - conclusion:
       - keep the current pure `kind::mxf4.block_scale.scale_vec::2X` PTX form for MXFP4 PV
     - a dense-MX constant-scale fast path was also screened and reverted
       - idea:
         - skip per-group E8M0 conversion when dense fragments round `amax / 6` to the constant E8M0 scale `2^-2`
       - result:
         - no useful ratio improvement
         - one `d64`, `S=512` fresh attempt failed in the aggregate screen
       - conclusion:
         - keep the safer per-group rounded-scale path for now
   - validation:
     - `python -m py_compile flash_attn/cute/fp4_flash_fwd_sm100_pvfused.py tests/cute/benchmark_fp4_pv.py tests/cute/test_fp4_flash_attn.py`
     - `PYTHONPATH=/workspace/codebases/fp4_matmul/flash-attention PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 timeout 240s /workspace/codebases/fp4_matmul/.venv/bin/python -m pytest -q tests/cute/test_fp4_flash_attn.py -k "fp4_pv_mxfp4_fake_compile_dense_forward or fp4_pv_fused_fake_compile_dense_forward or fp4_pv_fused_exact_lane_accepts_mxfp4_scale_config or fp4_pv_validation_errors or fp4_qk_validation_errors"`
       - result: `34 passed, 120 deselected`
     - `CUDA_VISIBLE_DEVICES=1 ... benchmark_fp4_pv.py --head-dims 64,128 --seqlens 512 --compare-mode full --fp4-pv-format mxfp4 --skip-baseline-check --fresh-runs 2 --max-attempts 5`
       - `d64`, `S=512`: `success_count=2`, `failure_count=0`, `pv_fused_out_max=0.032959`, `pv_fused_lse_max=0.942394`, `pv_fused_over_qkfast=1.265`
       - `d128`, `S=512`: `success_count=2`, `failure_count=0`, `pv_fused_out_max=0.041016`, `pv_fused_lse_max=0.941578`, `pv_fused_over_qkfast=1.242`
     - `CUDA_VISIBLE_DEVICES=1 ... benchmark_fp4_pv.py --head-dims 64,128 --seqlens 1024 --compare-mode full --fp4-pv-format mxfp4 --skip-baseline-check --fresh-runs 1 --max-attempts 3`
       - `d64`, `S=1024`: `success_count=1`, `failure_count=1`, clean sample had `pv_fused_out_max=0.028198`, `pv_fused_lse_max=0.922174`, `pv_fused_over_qkfast=1.265`
       - `d128`, `S=1024`: `success_count=1`, `failure_count=0`, `pv_fused_out_max=0.027039`, `pv_fused_lse_max=0.919175`, `pv_fused_over_qkfast=1.222`
   - current conclusion:
     - MXFP4 PV is now a usable correctness baseline for the `S=512` dense rows tested here
     - it is still not below the `1.0` ratio target
     - `S=1024`, especially `d64`, still needs a stability pass before treating MXFP4 as fully clean for broader shapes

106. Current perf refresh and failed dense `d128` constant-group-max screen.
   - refreshed the historical sub-`1.0` claim:
     - the top-of-file sub-`1.0` datapoints were on `d64` rows, not the current `d128` must-win row
     - on the current tree they did not reproduce on physical device `1` or `2`
     - current `d64` full-mode refresh:
       - physical device `1`, NVFP4 PV:
         - `S=512`: `success_count=2`, `failure_count=0`, `pv_fused_over_qkfast=1.269`, `pv_fused_lse_max=0.942394`
         - `S=1024`: `success_count=2`, `failure_count=0`, `pv_fused_over_qkfast=1.281`, `pv_fused_lse_max=0.922174`
       - physical device `2`, NVFP4 PV:
         - `S=512`: `success_count=2`, `failure_count=0`, `pv_fused_over_qkfast=1.290`, `pv_fused_lse_max=0.942394`
         - `S=1024`: `success_count=2`, `failure_count=0`, `pv_fused_over_qkfast=1.298`, `pv_fused_lse_max=0.922174`
   - current must-win fused-only refresh on physical device `2`:
     - `d128`, `S=512`, NVFP4 PV:
       - `success_count=3`, `failure_count=0`
       - `qkfast_ms=0.10620`
       - `pv_fused_ms=0.13207`
       - `pv_fused_over_qkfast=1.244`
     - `d128`, `S=512`, MXFP4 PV:
       - `success_count=3`, `failure_count=0`
       - `qkfast_ms=0.10474`
       - `pv_fused_ms=0.13106`
       - `pv_fused_over_qkfast=1.251`
   - screened and reverted:
     - idea:
       - in the dense nonmasked `d128` path, replace `exact_d128_dense_group_max_branchless(...)` with `Float32(0.0)`
       - rationale:
         - the current helper clamps the dense group max with `0.0`, so source-level constant folding might remove shuffle/reduction overhead
     - result:
       - correctness stayed finite for the tested `d128`, `S=512` rows
       - broad NVFP4 replacement regressed the default path:
         - baseline NVFP4 full-mode A/B: `pv_fused_over_qkfast=1.176`
         - constant-max NVFP4 full-mode A/B: `1.310`
       - MXFP4-only gating was noisy and did not reproduce a stable gain:
         - baseline MXFP4 full-mode A/B: `1.225`
         - broad constant-max MXFP4 first run: `1.187`
         - MXFP4-only confirmation: `1.229`
     - conclusion:
       - no code change kept
       - the old assumption that dense group max is a profitable constant is not useful for the current generated code
       - the relevant SageAttention3 pattern is still per-group P scaling from the score maxima plus row max; it is not a blanket constant-scale PV path
   - current interpretation:
     - the current tree has not reproduced any real speedup versus recovered `qkfast` / FA4 under the fresh-process batched timer
     - MXFP4 PV is now a correctness-capable experimental path, but its current speed is essentially tied with NVFP4 PV and still about `1.25x` over qkfast on the must-win row
     - further small source-level pack/reduction tweaks are unlikely to be enough; the strongest evidence still points at exact back-half `SFV` staging / scale-byte movement and the descriptor-control path

107. Re-screened skipping exact-lane `SFP` identity prefill.
   - idea:
     - the exact producer writes generated `P` scale bytes before the S2T copy, so the initial `sSFP` fill might be redundant on dense full tiles
     - this directly targeted the documented exact `SFP` constant-byte fill / stage-prep cost
   - change screened:
     - skipped `fill_scale_stage_constant(...)` for `sSFP` when `use_exact_fp4_pv_lane=True`
     - kept the old fill for the non-exact PV path
   - validation:
     - `python -m py_compile flash_attn/cute/fp4_flash_fwd_sm100_pvfused.py tests/cute/benchmark_fp4_pv.py tests/cute/test_fp4_flash_attn.py`
     - focused fake compile / validation suite:
       - `34 passed, 120 deselected`
   - runtime result:
     - first aggregate on physical device `2`:
       - `d64`, `S=512`, NVFP4 PV:
         - `success_count=2`, `failure_count=0`
         - `pv_fused_over_qkfast=1.301`
         - `pv_fused_out_max=0.032959`, `pv_fused_lse_max=0.942394`
       - `d128`, `S=512`, NVFP4 PV:
         - `success_count=2`, `failure_count=2`
         - clean samples had `pv_fused_over_qkfast=1.251`
         - `pv_fused_out_max=0.033173`, `pv_fused_lse_max=0.941578`
     - repeat `d128`, `S=512`, NVFP4 aggregate:
       - only `1` clean sample after `6` attempts
       - last failure:
         - `FP4 PV output contains NaN or Inf`
   - conclusion:
     - reverted immediately
     - the exact SFP prefill is still required for stable `d128`, likely because the S2T copy covers scale slots not deterministically overwritten by the current per-warp producer
     - even the clean samples did not improve the ratio, so this route is not worth narrowing to a d64-only special case right now

108. Probed direct `tcgen05.cp` for generated `SFP` S2T.
   - idea:
     - item `96` through `101` proved the direct-copy route is blocked for loaded `SFV`
     - generated `SFP` might have exposed a simpler descriptor shape because it is produced locally in shared memory
   - change screened:
     - replaced the exact-lane generated-`SFP`:
       - `cute.copy(tiled_copy_s2t_sfp_exact, ..., tCtSFP_compact_s2t_exact)`
     - with:
       - `sm100_utils.copy_scale_smem_to_tmem(Int32(self.tmem_sfp_offset), tCsSFP_compact_s2t_exact[(..., stage)], cta_group=...)`
   - result:
     - fake compile tests did not exercise the lowering path
     - first real `d128`, `S=512`, NVFP4 compile failed before PTX emission:
       - `DSLRuntimeError: tensor<Value(... !cute_nvgpu.smem_desc_view ...)> to integer conversion is not supported`
   - conclusion:
     - reverted immediately
     - generated `SFP` has the same source-descriptor blocker as loaded `SFV`
     - direct low-level `tcgen05.cp` is not locally usable for either PV scale path until the CuTe descriptor-to-integer / direct `copy_s2t` binding issue is fixed

109. Kept scale-fill dtype cleanup for broader FP4 format support.
   - issue:
     - `load_scale_stage_layout(...)` filled tail identity scale bytes with `self.pv_sf_dtype`
     - that is correct for PV scale fills, but wrong for Q/K scale fills if the Q/K and PV scale dtypes diverge
   - kept change:
     - added an explicit `scale_dtype` argument to `load_scale_stage_layout(...)`
     - Q/K callers pass `self.fp4_sf_dtype`
     - PV fallback callers pass `self.pv_sf_dtype`
   - validation:
     - `python -m py_compile flash_attn/cute/fp4_flash_fwd_sm100_pvfused.py tests/cute/benchmark_fp4_pv.py tests/cute/test_fp4_flash_attn.py`
     - focused fake compile / validation suite:
       - `34 passed, 120 deselected`
     - real runtime smoke on physical device `2`:
       - `d128`, `S=512`, NVFP4 PV, full mode:
         - `pv_fused_over_qkfast=1.262`
         - `pv_fused_out_max=0.070801`, `pv_fused_lse_max=0.941578`
       - `d128`, `S=512`, MXFP4 PV, full mode:
         - `pv_fused_over_qkfast=1.195`
         - `pv_fused_out_max=0.041016`, `pv_fused_lse_max=0.941578`
   - performance:
     - this is a general-shape correctness cleanup, not a speedup
     - the must-win row is still above the `1.0` target

110. Re-screened larger shapes and full-tile `SFV` unique-scale copy.
   - broad fused-only shape refresh on physical device `2`:
     - NVFP4 PV:
       - `d64`, `S=512`: `pv_fused_over_qkfast=1.249`
       - `d64`, `S=1024`: `pv_fused_over_qkfast=1.292`
       - `d64`, `S=2048`: `pv_fused_over_qkfast=1.712`
       - `d128`, `S=512`: `pv_fused_over_qkfast=1.264`
       - `d128`, `S=1024`: `pv_fused_over_qkfast=1.362`
       - `d128`, `S=2048`: `pv_fused_over_qkfast=2.475`
     - MXFP4 PV:
       - `d64`, `S=512`: `pv_fused_over_qkfast=1.270`
       - `d64`, `S=1024`: `pv_fused_over_qkfast=1.272`, with `success_count=1`, `failure_count=1`
       - `d64`, `S=2048`: `pv_fused_over_qkfast=1.607`
       - `d128`, `S=512`: `pv_fused_over_qkfast=1.227`
       - `d128`, `S=1024`: `pv_fused_over_qkfast=1.226`
       - `d128`, `S=2048`: `pv_fused_over_qkfast=2.426`
   - interpretation:
     - larger sequence lengths do not currently recover the old sub-`1.0` behavior
     - `S=2048` is materially worse for both NVFP4 and MXFP4, so the remaining target should stay on the back-half scale movement / descriptor path rather than assuming more work amortizes the overhead
   - optimization screened:
     - changed the full-tile direct `SFV` loader to copy one representative scale byte per `(sequence scale group, V dimension)` instead of writing the same scale across every row in the group
     - partial-tail handling was left unchanged
   - validation:
     - `python -m py_compile flash_attn/cute/fp4_flash_fwd_sm100_pvfused.py tests/cute/benchmark_fp4_pv.py tests/cute/test_fp4_flash_attn.py`
     - focused fake compile / validation suite:
       - `34 passed, 120 deselected`
   - runtime result:
     - NVFP4 PV full mode, physical device `2`:
       - `d64`, `S=512`: `pv_fused_over_qkfast=1.295`, `success_count=1`, `failure_count=0`
       - `d128`, `S=512`: `pv_fused_over_qkfast=1.312`, `success_count=1`, `failure_count=0`
     - MXFP4 PV full mode, physical device `2`:
       - `d64`, `S=512`: `pv_fused_over_qkfast=1.249`, `success_count=1`, `failure_count=0`
       - `d128`, `S=512`: `pv_fused_over_qkfast=1.235`, `success_count=1`, `failure_count=0`
   - conclusion:
     - reverted immediately
     - the reduced-store mapping is semantically valid for the clean rows tested, but the generated code is slower than the existing direct loader
     - this again points away from simple source-level loop count reductions and toward the actual S2T descriptor / scale traffic mechanism

111. Re-screened MXFP4 register knobs and pack-helper rewrites.
   - generic `SFV` path re-check:
     - command shape:
       - `d128`, `S=512`, MXFP4 PV, full mode, physical device `2`
       - `FLASH_ATTN_FP4_PV_EXACT_SFV_DIRECT=0`
     - result:
       - `success_count=1`, `failure_count=1`
       - `pv_fused_over_qkfast=1.212`
       - `pv_fused_out_max=0.048393`, `pv_fused_lse_max=0.941578`
     - conclusion:
       - no change kept
       - the generic staged `SFV` path is still diagnostic-only and does not beat the direct loader
   - correction tile / register knob screens on `d128`, `S=512`, MXFP4 PV:
     - `FLASH_ATTN_FP4_PV_CORR_TILE_SIZE=32`:
       - `pv_fused_over_qkfast=1.248`
       - clean but slower
     - `FLASH_ATTN_FP4_PV_CORR_TILE_SIZE=64`:
       - ptxas failed register allocation at `192` registers
       - no runtime datapoint
     - `FLASH_ATTN_FP4_FORCE_REGS_SOFTMAX=200`:
       - `pv_fused_over_qkfast=1.222`
     - `FLASH_ATTN_FP4_FORCE_REGS_SOFTMAX=184`:
       - `pv_fused_over_qkfast=1.232`
     - `FLASH_ATTN_FP4_FORCE_REGS_OTHER=64`:
       - `pv_fused_over_qkfast=1.233`
     - conclusion:
       - no register or correction-tile change kept
   - MXFP4 folded d128 pack helper screen:
     - idea:
       - fold the E8M0 round-up exponent recovery into the dense d128 pack helper so the path does not materialize `actual_scale_log2` as a separate DSL value before packing
     - validation while screened:
       - `git diff --check`
       - `python -m py_compile flash_attn/cute/fp4_flash_fwd_sm100_pvfused.py tests/cute/benchmark_fp4_pv.py tests/cute/test_fp4_flash_attn.py`
       - focused fake compile / validation suite:
         - `34 passed, 120 deselected`
     - runtime:
       - initial paired fused-only A/B looked promising:
         - folded helper: `success_count=3`, `failure_count=0`, `pv_fused_over_qkfast=1.198`
         - old helper: `success_count=3`, `failure_count=0`, `pv_fused_over_qkfast=1.227`
       - repeat after reverting the broader generic helper did not reproduce:
         - folded helper: `success_count=3`, `failure_count=0`, `pv_fused_over_qkfast=1.237`
         - immediate old-helper re-check: `success_count=3`, `failure_count=0`, `pv_fused_over_qkfast=1.223`
     - conclusion:
       - reverted
       - this is not stable enough to keep as a real MXFP4 speedup
   - generic 8-value log2-scaled pack helper screen:
     - idea:
       - replace the generic d64/masked path's materialized `p0..p7` values with an inline asm helper that exponentiates and packs directly
     - validation while screened:
       - `git diff --check`
       - `python -m py_compile flash_attn/cute/fp4_flash_fwd_sm100_pvfused.py tests/cute/benchmark_fp4_pv.py tests/cute/test_fp4_flash_attn.py`
       - focused fake compile / validation suite:
         - `34 passed, 120 deselected`
     - runtime:
       - `d64`, `S=512`, NVFP4 PV, fused-only:
         - `success_count=2`, `failure_count=0`, `pv_fused_over_qkfast=1.224`
       - `d64`, `S=512`, NVFP4 PV, full mode:
         - `success_count=1`, `failure_count=0`, `pv_fused_over_qkfast=1.286`
         - `pv_fused_out_max=0.072021`, `pv_fused_lse_max=0.942394`
       - `d64`, `S=512`, MXFP4 PV, fused-only:
         - `success_count=2`, `failure_count=0`, `pv_fused_over_qkfast=1.269`
       - `d64`, `S=512`, MXFP4 PV, full mode repeat:
         - `success_count=2`, `failure_count=1`, `pv_fused_over_qkfast=1.258`
         - `pv_fused_out_max=0.053040`, `pv_fused_lse_max=0.942394`
   - conclusion:
     - reverted
     - the helper is not a robust general-shape win and is not worth the added instability risk

112. Screened alternate `SFV` S2T copy atoms.
   - idea:
     - CuTe exposes `64x128b.warpx2` copy atoms in addition to the current `32x128b.warpx4` broadcast atom
     - if the scale-copy cost is dominated by the `warpx4` broadcast pattern, a `warpx2` atom might reduce back-half traffic without needing the blocked direct descriptor route
   - change screened:
     - added an opt-in `SFV`-only selector for:
       - `tcgen05.Cp2x64x128b0213Op`
       - `tcgen05.Cp2x64x128b0123Op`
     - left Q/K scale copies and generated `SFP` on the existing `Cp4x32x128bOp`
   - validation while screened:
     - default path:
       - `git diff --check`
       - `python -m py_compile flash_attn/cute/fp4_flash_fwd_sm100_pvfused.py tests/cute/benchmark_fp4_pv.py tests/cute/test_fp4_flash_attn.py`
       - focused fake compile / validation suite:
         - `34 passed, 120 deselected`
   - runtime result on physical device `2`:
     - `warpx2_0213`, `d128`, `S=512`, NVFP4 PV, full mode:
       - `success_count=1`, `failure_count=0`
       - `pv_fused_over_qkfast=1.504`
       - `pv_fused_out_max=0.072876`, `pv_fused_lse_max=0.941578`
     - `warpx2_0123`, `d128`, `S=512`, NVFP4 PV, full mode:
       - `success_count=1`, `failure_count=0`
       - `pv_fused_over_qkfast=1.274`
       - `pv_fused_out_max=0.070312`, `pv_fused_lse_max=0.941578`
     - `warpx2_0123`, `d128`, `S=512`, MXFP4 PV, full mode:
       - `success_count=1`, `failure_count=2`
       - `pv_fused_over_qkfast=1.578`
       - `pv_fused_out_max=0.041016`, `pv_fused_lse_max=0.941578`
   - conclusion:
     - reverted immediately
     - the alternate CuTe copy atoms compile and can be numerically finite, but they are slower and less stable than the existing `32x128b.warpx4` path
     - this makes the useful descriptor-control target narrower: the win is not available by selecting another stock S2T atom

113. Screened making the benchmark tile override affect FP4 QK/PV.
   - issue:
     - `tests/cute/benchmark_fp4_pv.py` accepts `--pv-tile-m/--pv-tile-n`
     - `interface.py` currently ignores that override when `is_fp4_qk=True`, forcing `FwdConfig(128, 128, ...)`
   - change screened:
     - temporarily gave explicit `tile_mn` precedence over the FP4 default
     - this was only used to test whether nearby exact-lane tile geometry could improve the must-win row
   - validation while screened:
     - `git diff --check`
     - `python -m py_compile flash_attn/cute/interface.py flash_attn/cute/fp4_flash_fwd_sm100_pvfused.py tests/cute/benchmark_fp4_pv.py tests/cute/test_fp4_flash_attn.py`
   - runtime/JIT result on physical device `2`:
     - `128x64`, `d128`, `S=512`, NVFP4 PV:
       - failed during JIT lowering before runtime
       - `load_scale_stage(...)` raised:
         - `ValueError: Expected source and destination tensors to have the same size in mode-1, but got 16 and 32`
       - interpretation:
         - the current Q/K scale staging layout is not compatible with shrinking `N` to `64`
     - `128x256`, `d128`, `S=512`, NVFP4 PV:
       - failed during kernel construction
       - `assert self.tmem_total <= self.tmem_alloc_cols`
       - interpretation:
         - expanding `N` to `256` exceeds the available TMEM allocation once FP4 QK and PV scale columns are included
   - conclusion:
     - reverted immediately
     - the exact FP4 lane is effectively pinned to `128x128` until the scale-layout and TMEM-column accounting are changed together
     - the existing benchmark tile knob should remain ignored for FP4 QK/PV rather than exposing unsupported shapes

114. Screened alternate generated-`SFP` S2T copy atom.
   - idea:
     - item `112` rejected alternate `warpx2` atoms for loaded `SFV`
     - generated `SFP` has a different producer and might respond differently, so screen it separately before closing the stock-copy-atom route
   - change screened:
     - added a temporary `use_warpx2_0123` option to `scale_s2t_copy_and_partition(...)`
     - used it only for exact-lane generated `SFP`
     - left Q/K scales and loaded `SFV` on the existing `Cp4x32x128bOp`
   - validation while screened:
     - `git diff --check`
     - `python -m py_compile flash_attn/cute/fp4_flash_fwd_sm100_pvfused.py tests/cute/benchmark_fp4_pv.py tests/cute/test_fp4_flash_attn.py`
   - runtime result on physical device `2`:
     - `d128`, `S=512`, NVFP4 PV, full mode:
       - `success_count=1`, `failure_count=0`
       - `pv_fused_ms=0.16577`
       - `pv_fused_over_qkfast=1.243`
       - `pv_fused_out_max=0.087891`, `pv_fused_lse_max=0.941578`
   - conclusion:
     - reverted immediately
     - the generated-`SFP` `warpx2_0123` path is finite but slower in absolute fused time than the stable path
     - stock `warpx2` copy atoms are now rejected for both PV scale operands

115. Instance handoff / recreation checkpoint for the pushed branch.
   - checkpoint intent:
     - this note is the durable restart point for the current `fp4-attention` branch state
     - the live code includes the stable exact-lane FP4 PV implementation plus MXFP4 PV support and the scale-fill dtype cleanup from item `109`
     - the later screens in items `110` through `114` are intentionally documented as rejected and reverted
   - recreate the workspace state from a new instance:
     - `cd /workspace/codebases/fp4_matmul/flash-attention`
     - `git fetch origin`
     - `git checkout fp4-attention`
     - `git pull --ff-only origin fp4-attention`
     - read this file from the top-level current-state summary and items `109` through `115`
   - cheap validation commands:
     - `git diff --check`
     - `/workspace/codebases/fp4_matmul/.venv/bin/python -m py_compile flash_attn/cute/interface.py flash_attn/cute/fp4_flash_fwd_sm100_pvfused.py tests/cute/benchmark_fp4_pv.py tests/cute/test_fp4_flash_attn.py`
     - `PYTHONPATH=/workspace/codebases/fp4_matmul/flash-attention PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 timeout 240s /workspace/codebases/fp4_matmul/.venv/bin/python -m pytest -q tests/cute/test_fp4_flash_attn.py -k "fp4_pv_mxfp4_fake_compile_dense_forward or fp4_pv_fused_fake_compile_dense_forward or fp4_pv_fused_exact_lane_accepts_mxfp4_scale_config or fp4_pv_validation_errors or fp4_qk_validation_errors"`
   - expected cheap validation result:
     - focused test suite: `34 passed, 120 deselected`
   - validation caveat from the final checkpoint instance:
     - `git diff --check` passed
     - `python -m py_compile ...` passed for the handoff command listed above
     - the focused pytest command timed out after `240s` with no output in the final instance, despite previously completing as `34 passed, 120 deselected`
     - if recreating progress, rerun the focused pytest command first and check for local environment stalls before trusting runtime benchmark noise
   - primary runtime probes:
     - use a quiet physical GPU; recent screens used physical device `2` exposed as benchmark device `0`
     - NVFP4 PV must-win smoke:
       - `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/workspace/codebases/fp4_matmul/flash-attention PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 timeout 900s /workspace/codebases/fp4_matmul/.venv/bin/python tests/cute/benchmark_fp4_pv.py --device 0 --head-dims 128 --seqlens 512 --compare-mode full --fp4-pv-format nvfp4 --skip-baseline-check --fresh-runs 1 --max-attempts 3`
     - MXFP4 PV must-win smoke:
       - `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/workspace/codebases/fp4_matmul/flash-attention PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 timeout 900s /workspace/codebases/fp4_matmul/.venv/bin/python tests/cute/benchmark_fp4_pv.py --device 0 --head-dims 128 --seqlens 512 --compare-mode full --fp4-pv-format mxfp4 --skip-baseline-check --fresh-runs 1 --max-attempts 3`
     - broader fused-only shape refresh:
       - `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/workspace/codebases/fp4_matmul/flash-attention PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 timeout 1800s /workspace/codebases/fp4_matmul/.venv/bin/python tests/cute/benchmark_fp4_pv.py --device 0 --head-dims 64,128 --seqlens 512,1024,2048 --compare-mode fused-only --fp4-pv-format mxfp4 --skip-baseline-check --fresh-runs 1 --max-attempts 3`
   - current performance boundary:
     - the branch has not reproduced a robust speedup versus qkfast / FA4 under the fresh-process timer
     - current stable `d128`, `S=512` ratios are still generally around `1.2x` over qkfast, with noise between individual fresh-process runs
     - the target remains `pv_fused_over_qkfast < 1.0`
   - next useful direction:
     - do not reopen the reverted screens in items `110` through `114`
     - the strongest remaining evidence still points at a true descriptor/control-path fix for exact `SFP` / `SFV` scale movement, not another source-level pack helper, register cap, stock S2T copy atom, or tile-size override
