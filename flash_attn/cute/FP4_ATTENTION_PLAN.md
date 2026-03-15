# FP4 Flash Attention — Integration Plan

## Goal
Adapt FA4's CUTE-DSL flash attention to use FP4 (E2M1) block-scaled MMA for the QK product,
with pre-quantized Q and K inputs (NVFP4 or MXFP4 format).

## Architecture Overview

### Current (BF16)
```
Q[bf16] → SMEM → tcgen05.mma.kind::f16 → S[fp32 in TMEM] → softmax → P → PV GEMM → O
K[bf16] → SMEM ↗
```

### Target (FP4)
```
Q[fp4]    → SMEM → tcgen05.mma.kind::mxf4nvf4.block_scale.scale_vec::4X → S[fp32 in TMEM] → softmax → P → PV GEMM → O
K[fp4]    → SMEM ↗
Q_sc[fp8] → SMEM → tcgen05.cp → TMEM (scale A) ↗
K_sc[fp8] → SMEM → tcgen05.cp → TMEM (scale B) ↗
```

## Completed Infrastructure

### 1. Block-Scaled Instruction Descriptor (`mma_sm100_desc.py`)
- `make_block_scaled_instr_desc()` — FP4 idesc with M>>7, scale_factor_id, scale_type
- `mma_op_to_idesc_block_scaled()` — converts BlockScaledMmaOp to idesc

### 2. FP4 PTX Emission (`blackwell_helpers.py`)
- `declare_ptx_idesc_block_scaled()` — PTX register for block-scaled idesc
- `gemm_ptx_fp4_block_scaled()` — emits `kind::mxf4nvf4.block_scale` PTX with TMEM scale operands

### 3. Existing CUTE-DSL Infrastructure
- `make_blockscaled_trivial_tiled_mma()` — creates `MmaMXF4NVF4Op` (NVFP4) or `MmaMXF4Op` (MXFP4)
- `blockscaled_layout.py` — `make_smem_layout_sfa/sfb`, `make_tmem_layout_sfa/sfb`

## Files

| File | Status | Purpose |
|------|--------|---------|
| `mma_sm100_desc.py` | ✅ Done | Block-scaled idesc |
| `blackwell_helpers.py` | ✅ Done | FP4 PTX emission |
| `fp4_flash_fwd_sm100.py` | 🔧 TODO | FP4 forward attention (copy of `flash_fwd_sm100.py`) |
| `fp4_flash_bwd_sm100.py` | 📋 Later | FP4 backward attention (copy of `flash_bwd_sm100.py`) |

## Changes Needed in `fp4_flash_fwd_sm100.py`

### Phase 1: MMA Op & Idesc (minimal, testable)
1. **Add FP4 mode flag** to `__init__` (e.g. `use_fp4_qk=True`)
2. **Swap `tiled_mma_qk`** from `make_trivial_tiled_mma` → `make_blockscaled_trivial_tiled_mma`
   ```python
   # FP4 NVFP4: sf_dtype=Float8E4M3FN, sf_vec_size=16
   tiled_mma_qk = make_blockscaled_trivial_tiled_mma(
       Float4E2M1FN, q_major_mode, k_major_mode,
       Float8E4M3FN, 16,  # sf_dtype, sf_vec_size
       cta_group, self.mma_tiler_qk[:2],
   )
   ```
3. **Update mma_tiler_qk**: K reduction dim is 64 for FP4 (vs 16 for BF16)
4. **Relax type check**: Q/K can be FP4 while V stays BF16

### Phase 2: SMEM Layouts & Scale Factors
5. **SMEM layouts for FP4 Q/K**: Use FP4 element type for sQ/sK layouts
6. **Add scale factor SMEM**: Using `make_smem_layout_sfa/sfb` from `blockscaled_layout.py`
7. **Add to SMEM struct**: sSFA, sSFK fields in SharedStorage

### Phase 3: TMEM & Scale Loading
8. **TMEM allocation**: Add TMEM regions for Q/K scale factors
9. **Scale loading**: Before QK MMA, load scales SMEM→TMEM via `tcgen05.cp.cta_group::X.32x128b.warpx4`
10. **TMA descriptors**: Add TMA for Q_scale, K_scale GMEM→SMEM

### Phase 4: QK GEMM Dispatch
11. **Replace `gemm_Si`**: Use `gemm_ptx_fp4_block_scaled` instead of `gemm_ptx_precomputed_varname`
12. **Replace idesc declaration**: Use `declare_ptx_idesc_block_scaled`

### Phase 5: Input Interface
13. **Accept FP4 tensors**: mQ as fp4x2 packed, mK as fp4x2 packed
14. **Accept scale tensors**: mQ_scale, mK_scale as fp8 
15. **Accept global scales**: Q_sg, K_sg as float32

## Key Constraints
- **K dimension**: FP4 MMA uses K=64 per instruction (vs K=16 for BF16)
  - `head_dim` must be multiple of 64
  - `mma_tiler_qk[2]` = head_dim (padded to multiple of 64)
- **SMEM swizzle**: FP4 data is 4-bit, so 128B swizzle = 256 elements
- **Scale TMEM addressing**: Scales must be at specific TMEM offsets per the hardware spec
- **Output stays FP32 in TMEM**: S matrix remains in TMEM for softmax (no dtype change there)

## Testing Strategy
1. Start with Phase 1 — verify the MmaMXF4NVF4Op atom can be created
2. Add phases incrementally, testing compilation at each step
3. Final correctness test: compare FP4 attention output vs BF16 reference
