# FP4 Flash Attention Plan

## Current Scope
- QK-only FP4 forward on SM100/SM110.
- Dense fixed-length attention only.
- Supported shapes: batched MHA and GQA, causal and noncausal, `num_splits=1`.
- FP4 PV now emits inline PTX for the TS block-scaled MMA transport, but runtime stability still needs to be re-verified on the narrow bring-up slice.
- Unsupported in the stable public landing: varlen, paged KV, block sparsity, local windowing, public autograd backward.
- An experimental internal FP4 backward Q/K helper now exists for dense fixed-length MHA on SM100/SM110.
  - Current behavior: dequantize TK-style rowwise Q/K back to BF16 and reuse the existing backward kernel.
  - Native FP4 `dQ = dS @ K` / `dK = dS^T @ Q` is still the next kernel bring-up step because it requires on-tile `dS` quantization plus TK columnwise operand handling.

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
  - MMA copies staged scales from SMEM to TMEM with `scale_s2t_copy_and_partition(...)` plus `tcgen05.Cp4x32x128bOp(...)`.
- PV v1 currently uses the same manual GMEM->SMEM scale staging approach for `V`.
- FP4 PV now uses a TS-style block-scaled PTX helper (`gemm_ptx_fp4_block_scaled_partial(...)`) instead of the earlier `cute.gemm(...)` block-scaled MMA atom path.
- `P` is quantized to FP4 in `softmax_step(...)` with CTA-local per-block dynamic scales written into `sSFP`.
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
  - experimental FP4 backward-Q/K validation and dequantization-contract coverage for the internal helper

## Next Step
- Land native FP4 backward `dQ`/`dK` by quantizing `dS` on-tile and consuming TK columnwise Q/K metadata directly.
  - The FP4-specific backward kernel work now lives in `fp4_flash_bwd_sm100.py`; keep `flash_bwd_sm100.py` as the plain SM100 reference path.
  - The missing low-level primitive is no longer the PTX transport; it is the `dS` quantization + staging contract.
- Re-verify FP4 PV runtime on the narrowest slice first: dense noncausal MHA, `d64`, `nvfp4`, `num_splits=1`.
- Keep the stable QK-only path intact while PV and backward native FP4 bring-up continue.
