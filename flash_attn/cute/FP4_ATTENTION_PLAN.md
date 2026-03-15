# FP4 Flash Attention Plan

## Current Scope
- QK-only FP4 forward on SM100/SM110.
- Dense fixed-length attention only.
- Supported shapes: batched MHA and GQA, causal and noncausal, `num_splits=1`.
- Unsupported in this landing: varlen, paged KV, block sparsity, local windowing, backward, FP4 PV.

## Public API
- `_flash_attn_fwd(...)` and `flash_attn_func(...)` now accept:
  - `fp4_qk_format: Optional[Literal["nvfp4", "mxfp4"]] = None`
  - `q_scale: Optional[torch.Tensor] = None`
  - `k_scale: Optional[torch.Tensor] = None`
- Tensor contract:
  - `q`, `k`: packed `torch.uint8`, shape `(..., head_dim // 2)`
  - `v`: `torch.float16` or `torch.bfloat16`
  - `q_scale`, `k_scale`: FP8 with shape `(..., head_dim // 16)` for `nvfp4`, `(..., head_dim // 32)` for `mxfp4`
- Format mapping:
  - `nvfp4` -> `fp4_sf_dtype="e4m3"`, `fp4_sf_vec_size=16`
  - `mxfp4` -> `fp4_sf_dtype="e8m0"`, `fp4_sf_vec_size=32`

## Kernel Status
- `FP4FlashAttentionForwardSm100` remains the implementation home for FP4 QK.
- Packed FP4 Q/K are reinterpreted as logical `Float4E2M1FN` CuTe tensors in `cute_dsl_utils.py`.
- Scale tensors are reinterpreted with `blockscaled_layout.tile_atom_to_shape_SF(...)` inside the kernel setup.
- Q/K scale staging is manual cooperative GMEM->SMEM in v1.
  - No scale TMA is used in this landing.
  - MMA copies staged scales from SMEM to TMEM with `copy_scale_smem_to_tmem(...)` before each FP4 QK instruction.
- `use_2cta_instrs` is forced off for FP4 QK.

## Tests
- Root-level smoke script was replaced by pytest coverage in `tests/cute/test_fp4_flash_attn.py`.
- Coverage includes:
  - fake-compile checks for `nvfp4` and `mxfp4`
  - head dims `64` and `128`
  - causal and noncausal dense forward
  - compile-cache separation between BF16, `nvfp4`, and `mxfp4`
  - validation errors for missing scales, wrong dtypes/shapes, and unsupported features
  - runtime GPU comparisons against a dequantized BF16 reference for dense MHA and GQA

## Next Step
- Add an explicit FP4 PV path.
- Expected follow-up API additions:
  - `use_fp4_pv`
  - `v_scale`
- That work will quantize `P` on the fly at the softmax/PV handoff while keeping the current QK FP4 path intact.
