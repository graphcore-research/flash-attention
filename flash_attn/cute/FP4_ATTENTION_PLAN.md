# FP4 Flash Attention Plan

## Current Scope
- QK-only FP4 forward on SM100/SM110.
- Dense fixed-length attention only.
- Supported shapes: batched MHA and GQA, causal and noncausal, `num_splits=1`.
- FP4 PV v1 plumbing is in progress, but the runtime kernel is not stable yet.
- Unsupported in the stable landing: varlen, paged KV, block sparsity, local windowing, backward.

## Public API
- `_flash_attn_fwd(...)` and `flash_attn_func(...)` now accept:
  - `fp4_qk_format: Optional[Literal["nvfp4", "mxfp4"]] = None`
  - `q_scale: Optional[torch.Tensor] = None`
  - `k_scale: Optional[torch.Tensor] = None`
  - `use_fp4_pv: bool = False`
  - `v_scale: Optional[torch.Tensor] = None`
- Tensor contract:
  - `q`, `k`: packed `torch.uint8`, shape `(..., head_dim // 2)`
  - `v`: `torch.float16` or `torch.bfloat16` for QK-only FP4, packed `torch.uint8` when `use_fp4_pv=True`
  - `q_scale`, `k_scale`: FP8 with shape `(..., head_dim // 16)` for `nvfp4`, `(..., head_dim // 32)` for `mxfp4`
  - `v_scale`: FP8 with shape `(..., head_dim_v // 16)` for `nvfp4`
- Format mapping:
  - `nvfp4` -> `fp4_sf_dtype="e4m3"`, `fp4_sf_vec_size=16`
  - `mxfp4` -> `fp4_sf_dtype="e8m0"`, `fp4_sf_vec_size=32`

## Kernel Status
- `FP4FlashAttentionForwardSm100` remains the implementation home for FP4 QK.
- The same kernel now contains an experimental native FP4 PV path:
  - packed FP4 `V`
  - on-the-fly FP4 `P`
  - block-scaled FP4 PV MMA
  - BF16 output path unchanged
- Packed FP4 Q/K are reinterpreted as logical `Float4E2M1FN` CuTe tensors in `cute_dsl_utils.py`.
- Scale tensors are reinterpreted with `blockscaled_layout.tile_atom_to_shape_SF(...)` inside the kernel setup.
- Q/K scale staging is manual cooperative GMEM->SMEM in v1.
  - No scale TMA is used in this landing.
  - MMA copies staged scales from SMEM to TMEM with `copy_scale_smem_to_tmem(...)` before each FP4 QK instruction.
- PV v1 currently uses the same manual GMEM->SMEM scale staging approach for `V`.
- The current FP4 PV blocker is runtime liveness:
  - the simplest dense MHA slice now compiles and launches
  - the kernel then wedges during device execution (`torch.cuda.synchronize()` never returns)
  - this is after fixing the earlier bring-up issues in V-scale relayout, V-scale copy padding, and launch-time descriptor/setup errors
- The current `P` quantization path is still a correctness-first bring-up:
  - `P` is converted to FP4 in `softmax_step(...)`
  - `P` scales are currently filled with a constant factor instead of a final per-row/per-block dynamic scale reducer
- FP4 QK scheduling is now shape-gated instead of globally forcing `use_2cta_instrs=False`.
  - d128 MHA uses the faster schedule and is the current QK-only win.
  - The remaining laggard is noncausal `GQA (6q,2kv), d128`, which still defaults to the stable grouped-KV path.
- An experimental CTA-local packed GQA kernel path now exists behind the internal
  `FLASH_ATTN_FP4_Q3_LOCAL_PACK=1` selector.
  - The failing logical-FP4 copy was replaced with a raw packed-byte CTA-local Q gather into an aliased byte view of `sQ`.
  - The current gather uses 32-bit chunk copies, which are faster than the initial byte-at-a-time bring-up.
  - The path compiles and runs for noncausal `GQA (6q,2kv), d128`, but it is not enabled by default.
  - Current measurement with the 32-bit chunk gather:
    - `seqlen=512`: FP4/BF16 `1.185x`
    - `seqlen=2048`: FP4/BF16 `1.651x`
  - Because it misses the promotion gate, the grouped-KV default stays in place.

## Tests
- Root-level smoke script was replaced by pytest coverage in `tests/cute/test_fp4_flash_attn.py`.
- Coverage includes:
  - fake-compile checks for `nvfp4` and `mxfp4`
  - head dims `64` and `128`
  - causal and noncausal dense forward
  - compile-cache separation between BF16, QK-only `nvfp4`, QK+PV `nvfp4`, and `mxfp4`
  - internal selector coverage for the experimental raw-byte local-pack path
  - validation errors for missing scales, wrong dtypes/shapes, and unsupported features
  - runtime GPU comparisons against a dequantized BF16 reference for dense MHA and GQA
  - FP4 PV fake-compile and validation coverage

## Next Step
- Stabilize FP4 PV runtime on the narrowest slice first: dense noncausal MHA, `d64`, `nvfp4`, `num_splits=1`.
  - The remaining blocker is an in-kernel deadlock after launch, not API or ABI plumbing.
  - The next debugging pass should focus on the PV synchronization contract around:
    - `P` scale staging (`sSFP` / `tCtSFP`)
    - `V` scale staging (`sSFV` / `tCtSFV`)
    - the block-scaled PV MMA loop
- Keep the stable QK-only path intact while this PV bring-up continues.
