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
- FP4 QK scheduling is now shape-gated instead of globally forcing `use_2cta_instrs=False`.
  - d128 MHA uses the faster schedule and is the current QK-only win.
  - The remaining laggard is noncausal `GQA (6q,2kv), d128`, which still defaults to the stable grouped-KV path.
- An experimental CTA-local packed GQA kernel path now exists behind the internal
  `FLASH_ATTN_FP4_Q3_LOCAL_PACK=1` selector.
  - The failing logical-FP4 copy was replaced with a raw packed-byte CTA-local Q gather into an aliased byte view of `sQ`.
  - The path compiles and runs for noncausal `GQA (6q,2kv), d128`, but it is not enabled by default.
  - Current measurement with the raw-byte gather:
    - `seqlen=512`: FP4/BF16 `1.307x`
    - `seqlen=2048`: FP4/BF16 `1.787x`
  - Because it misses the promotion gate, the grouped-KV default stays in place.

## Tests
- Root-level smoke script was replaced by pytest coverage in `tests/cute/test_fp4_flash_attn.py`.
- Coverage includes:
  - fake-compile checks for `nvfp4` and `mxfp4`
  - head dims `64` and `128`
  - causal and noncausal dense forward
  - compile-cache separation between BF16, `nvfp4`, and `mxfp4`
  - internal selector coverage for the experimental raw-byte local-pack path
  - validation errors for missing scales, wrong dtypes/shapes, and unsupported features
  - runtime GPU comparisons against a dequantized BF16 reference for dense MHA and GQA

## Next Step
- Finish the last QK-only laggard first: noncausal `GQA (6q,2kv), d128`.
  - Keep the grouped-KV schedule as the default until a faster CTA-local KV-reuse path is ready.
  - The raw-byte local-pack path is now a correctness/bring-up reference, not the next default.
  - The next real optimization target is widening the raw-byte gather back toward an efficient vectorized load path without re-triggering CuTe's verifier failures.
- FP4 PV remains the follow-up after the QK-only d128 GQA path has a real win.
  - Expected later API additions:
    - `use_fp4_pv`
    - `v_scale`
  - That work will quantize `P` on the fly at the softmax/PV handoff while keeping the current QK FP4 path intact.
