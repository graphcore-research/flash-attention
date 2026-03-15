#!/usr/bin/env python3
"""
Minimal test for FP4 Flash Attention compilation.
Tests that FP4FlashAttentionForwardSm100 can be instantiated
and (attempt to) compile with FP4 Q/K and BF16 V.
"""
import os
import sys
import torch
import math

# Add flash-attention to path
FA_DIR = "/workspace/codebases/fp4_matmul/flash-attention"
sys.path.insert(0, FA_DIR)
os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Float4E2M1FN, Float8E4M3FN, Int32, Int64
from cutlass.cute.runtime import from_dlpack

print("=" * 60)
print("FP4 Flash Attention Compilation Test")
print("=" * 60)

# ── Configuration ──
batch_size = 1
seqlen_q = 128
seqlen_k = 128
num_heads = 4
num_heads_kv = 4
head_dim = 128
head_dim_v = 128
sf_vec_size = 16  # NVFP4: 1 scale per 16 elements
softmax_scale = 1.0 / math.sqrt(head_dim)

print(f"\nConfig: B={batch_size}, Sq={seqlen_q}, Sk={seqlen_k}, "
      f"H={num_heads}, Hkv={num_heads_kv}, D={head_dim}, Dv={head_dim_v}")

# ── Step 1: Create test tensors ──
print("\n[1] Creating test tensors...")
device = "cuda"

# FP4 Q/K are packed as uint8 (2 fp4 values per byte)
# Shape: (B, S, H, D//2) in uint8
q_packed = torch.randint(0, 256, (batch_size, seqlen_q, num_heads, head_dim // 2),
                         dtype=torch.uint8, device=device)
k_packed = torch.randint(0, 256, (batch_size, seqlen_k, num_heads_kv, head_dim // 2),
                         dtype=torch.uint8, device=device)

# V stays BF16
v = torch.randn(batch_size, seqlen_k, num_heads_kv, head_dim_v,
                 dtype=torch.bfloat16, device=device)

# Output is BF16
o = torch.zeros(batch_size, seqlen_q, num_heads, head_dim_v,
                dtype=torch.bfloat16, device=device)

# Scale factors: FP8 E4M3, shape (B, S, H, D//sf_vec_size)
q_scale = torch.ones(batch_size, seqlen_q, num_heads, head_dim // sf_vec_size,
                     dtype=torch.float8_e4m3fn, device=device)
k_scale = torch.ones(batch_size, seqlen_k, num_heads_kv, head_dim // sf_vec_size,
                     dtype=torch.float8_e4m3fn, device=device)

# LSE output
lse = torch.zeros(batch_size, num_heads, seqlen_q, dtype=torch.float32, device=device)

print(f"  Q packed:  {q_packed.shape} {q_packed.dtype}")
print(f"  K packed:  {k_packed.shape} {k_packed.dtype}")
print(f"  V:         {v.shape} {v.dtype}")
print(f"  O:         {o.shape} {o.dtype}")
print(f"  Q_scale:   {q_scale.shape} {q_scale.dtype}")
print(f"  K_scale:   {k_scale.shape} {k_scale.dtype}")
print(f"  LSE:       {lse.shape} {lse.dtype}")

# ── Step 2: Import and instantiate FP4FlashAttentionForwardSm100 ──
print("\n[2] Importing FP4FlashAttentionForwardSm100...")
try:
    from flash_attn.cute.fp4_flash_fwd_sm100 import FP4FlashAttentionForwardSm100
    print("  ✓ Import successful")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

print("\n[3] Instantiating FP4FlashAttentionForwardSm100...")
try:
    fa_fwd = FP4FlashAttentionForwardSm100(
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        qhead_per_kvhead=num_heads // num_heads_kv,
        is_causal=False,
        is_local=False,
        is_split_kv=False,
        pack_gqa=False,
        m_block_size=128,
        n_block_size=128,
        q_stage=1,
        is_persistent=True,
        use_fp4_qk=True,
        fp4_sf_dtype="e4m3",
        fp4_sf_vec_size=sf_vec_size,
    )
    print("  ✓ Instantiation successful")
    print(f"    use_fp4_qk: {fa_fwd.use_fp4_qk}")
    print(f"    head_dim_padded: {fa_fwd.head_dim_padded}")
    print(f"    tmem_sfa_offset: {fa_fwd.tmem_sfa_offset}")
    print(f"    tmem_sfk_offset: {fa_fwd.tmem_sfk_offset}")
    print(f"    tmem_total: {fa_fwd.tmem_total}")
except Exception as e:
    import traceback
    print(f"  ✗ Instantiation failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── Step 3: Try compilation with cute tensors ──
print("\n[4] Attempting cute.compile with real tensors...")
try:
    from flash_attn.cute.interface import to_cute_tensor
    import cuda.bindings.driver as cuda_drv

    # Create cute tensors from torch
    q_tensor = to_cute_tensor(q_packed)
    k_tensor = to_cute_tensor(k_packed)
    v_tensor = to_cute_tensor(v)
    o_tensor = to_cute_tensor(o)
    lse_tensor = to_cute_tensor(lse, assumed_align=4)
    q_scale_tensor = to_cute_tensor(q_scale)
    k_scale_tensor = to_cute_tensor(k_scale)

    current_stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)

    print("  Cute tensors created:")
    print(f"    Q: {q_tensor}")
    print(f"    K: {k_tensor}")
    print(f"    V: {v_tensor}")

    print("  Attempting compile...")
    compiled = cute.compile(
        fa_fwd,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        Float32(softmax_scale),
        current_stream,
        None,  # cu_seqlens_q
        None,  # cu_seqlens_k
        None,  # seqused_q
        None,  # seqused_k
        None,  # page_table
        None,  # window_size_left
        None,  # window_size_right
        None,  # learnable_sink
        None,  # blocksparse
        None,  # aux_tensors
        q_scale_tensor,   # mQ_scale
        k_scale_tensor,   # mK_scale
        options="--enable-tvm-ffi",
    )
    print("  ✓ Compilation successful!")

    # Try running
    print("\n[5] Attempting kernel launch...")
    compiled(
        q_packed, k_packed, v, o, lse,
        softmax_scale, current_stream,
        None, None, None, None, None,
        None, None, None, None, None,
        q_scale, k_scale,
    )
    torch.cuda.synchronize()
    print("  ✓ Kernel launch successful!")
    print(f"  Output stats: mean={o.float().mean():.4f}, std={o.float().std():.4f}")

except Exception as e:
    import traceback
    print(f"  ✗ Failed: {e}")
    traceback.print_exc()
    print("\n  This is expected — we're iteratively working through compilation issues.")

print("\n" + "=" * 60)
print("Test complete")
print("=" * 60)
