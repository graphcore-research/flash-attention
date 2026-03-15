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

## Files

| File | Status | Purpose |
|------|--------|---------|
| `mma_sm100_desc.py` | ✅ Done | Block-scaled idesc |
| `blackwell_helpers.py` | ✅ Done | FP4 PTX emission + scale copy |
| `fp4_flash_fwd_sm100.py` | 🔧 Phase 4 done | FP4 forward attention |
| `fp4_flash_bwd_sm100.py` | 📋 Later | FP4 backward attention |

## Progress

### ✅ Phases 1-4 Complete
- `FP4FlashAttentionForwardSm100` class with `use_fp4_qk` flag
- `make_blockscaled_trivial_tiled_mma` → `MmaMXF4NVF4Op` for QK
- Block-scaled idesc, FP4 GEMM dispatch with TMEM scale addresses
- Scale factor SMEM/TMEM layouts, SharedStorage fields
- `copy_scale_smem_to_tmem` PTX helper

### 🔧 Phase 5: Input Interface (Remaining)
- Accept FP4 Q/K tensors (packed fp4x2) + fp8 scale tensors
- Add TMA for Q_scale, K_scale (GMEM→SMEM)
- Pipeline scale loads alongside Q/K data loads
- Test compilation
