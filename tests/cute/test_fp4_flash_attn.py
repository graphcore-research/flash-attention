import os
import subprocess
import sys
import textwrap
import types

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from flash_attn.cute.interface import (
    _adapt_tk_nvfp4_transpose_scale,
    _bwd_postprocess_convert,
    _bwd_preprocess,
    _flash_attn_bwd,
    _flash_attn_bwd_fp4_qk,
    _flash_attn_fwd,
    _unswizzle_tk_nvfp4_transpose_scale_bytes,
)
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
from tests.cute.benchmark_fp4_qk import _benchmark_case
from tests.cute.benchmark_fp4_pv import _aggregate_benchmark_runs


FP4_GRID = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)

FP4_FORMAT_TO_DTYPE = {
    "nvfp4": torch.float8_e4m3fn,
    "mxfp4": torch.float8_e8m0fnu,
}

FP4_FORMAT_TO_VEC = {
    "nvfp4": 16,
    "mxfp4": 32,
}


def _fp4_pv_seqlen_k_padded(seqlen_k: int) -> int:
    return ((seqlen_k + 127) // 128) * 128


def _install_fake_cuda_runtime(monkeypatch):
    compile_calls = []

    def fake_compile(*args, **kwargs):
        compile_calls.append((args, kwargs))

        def _kernel(*_runtime_args, **_runtime_kwargs):
            return None

        return _kernel

    monkeypatch.setattr("flash_attn.cute.interface.cute.compile", fake_compile)
    monkeypatch.setattr(
        "flash_attn.cute.interface.cuda.CUstream",
        lambda stream: stream,
    )
    monkeypatch.setattr(
        "flash_attn.cute.interface.torch.cuda.current_stream",
        lambda: types.SimpleNamespace(cuda_stream=0),
    )
    monkeypatch.setattr(
        "flash_attn.cute.interface.torch.cuda.get_device_properties",
        lambda _device: types.SimpleNamespace(
            multi_processor_count=132, major=10, minor=0, name="Fake SM100"
        ),
    )
    monkeypatch.setattr(
        "torch.cuda.get_device_properties",
        lambda _device: types.SimpleNamespace(
            multi_processor_count=132, major=10, minor=0, name="Fake SM100"
        ),
    )
    # CuTe runtime fake tensors do not expose layout metadata yet, so keep the
    # fake-compile tests focused on dispatch/cache behavior rather than layout recasting.
    monkeypatch.setattr("flash_attn.cute.interface.to_cute_fp4_tensor", lambda tensor, *args, **kwargs: tensor)
    monkeypatch.setattr("flash_attn.cute.interface.to_cute_fp4_vt_tensor", lambda tensor, *args, **kwargs: tensor)
    monkeypatch.setattr(_flash_attn_fwd, "compile_cache", {})
    monkeypatch.setattr(_flash_attn_bwd, "compile_cache", {})
    monkeypatch.setattr(_bwd_preprocess, "compile_cache", {})
    monkeypatch.setattr(_bwd_postprocess_convert, "compile_cache", {})
    return compile_calls


def _make_fake_dense_inputs(
    *,
    head_dim: int,
    head_dim_v: int = 64,
    num_heads: int = 4,
    num_heads_kv: int | None = None,
    batch_size: int = 2,
    seqlen_q: int = 128,
    seqlen_k: int = 128,
):
    num_heads_kv = num_heads if num_heads_kv is None else num_heads_kv
    q = torch.empty(batch_size, seqlen_q, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.empty(batch_size, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.empty(batch_size, seqlen_k, num_heads_kv, head_dim_v, device="cuda", dtype=torch.bfloat16)
    return q, k, v


def _make_fake_fp4_dense_inputs(
    *,
    fp4_qk_format: str,
    head_dim: int,
    head_dim_v: int = 64,
    num_heads: int = 4,
    num_heads_kv: int | None = None,
    batch_size: int = 2,
    seqlen_q: int = 128,
    seqlen_k: int = 128,
):
    num_heads_kv = num_heads if num_heads_kv is None else num_heads_kv
    sf_vec = FP4_FORMAT_TO_VEC[fp4_qk_format]
    sf_dtype = FP4_FORMAT_TO_DTYPE[fp4_qk_format]
    q = torch.empty(batch_size, seqlen_q, num_heads, head_dim // 2, device="cuda", dtype=torch.uint8)
    k = torch.empty(batch_size, seqlen_k, num_heads_kv, head_dim // 2, device="cuda", dtype=torch.uint8)
    v = torch.empty(batch_size, seqlen_k, num_heads_kv, head_dim_v, device="cuda", dtype=torch.bfloat16)
    q_scale = torch.empty(
        batch_size, seqlen_q, num_heads, head_dim // sf_vec, device="cuda", dtype=sf_dtype
    )
    k_scale = torch.empty(
        batch_size, seqlen_k, num_heads_kv, head_dim // sf_vec, device="cuda", dtype=sf_dtype
    )
    return q, k, v, q_scale, k_scale


def _make_fake_fp4_pv_dense_inputs(
    *,
    fp4_qk_format: str,
    fp4_pv_format: str | None = None,
    head_dim: int,
    head_dim_v: int = 64,
    num_heads: int = 4,
    num_heads_kv: int | None = None,
    batch_size: int = 2,
    seqlen_q: int = 128,
    seqlen_k: int = 128,
):
    num_heads_kv = num_heads if num_heads_kv is None else num_heads_kv
    fp4_pv_format = fp4_qk_format if fp4_pv_format is None else fp4_pv_format
    sf_vec = FP4_FORMAT_TO_VEC[fp4_qk_format]
    sf_dtype = FP4_FORMAT_TO_DTYPE[fp4_qk_format]
    pv_sf_vec = FP4_FORMAT_TO_VEC[fp4_pv_format]
    pv_sf_dtype = FP4_FORMAT_TO_DTYPE[fp4_pv_format]
    seqlen_k_padded = _fp4_pv_seqlen_k_padded(seqlen_k)
    q = torch.empty(batch_size, seqlen_q, num_heads, head_dim // 2, device="cuda", dtype=torch.uint8)
    k = torch.empty(batch_size, seqlen_k, num_heads_kv, head_dim // 2, device="cuda", dtype=torch.uint8)
    # FP4 PV consumes pretransposed packed Vt / SFVt:
    # packed Vt  : (B, H_k, D_v, S_k_padded // 2)
    # scale SFVt : (B, H_k, D_v, S_k_padded // 16) in transpose-colwise storage
    v = torch.empty(batch_size, num_heads_kv, head_dim_v, seqlen_k_padded // 2, device="cuda", dtype=torch.uint8)
    q_scale = torch.empty(
        batch_size, seqlen_q, num_heads, head_dim // sf_vec, device="cuda", dtype=sf_dtype
    )
    k_scale = torch.empty(
        batch_size, seqlen_k, num_heads_kv, head_dim // sf_vec, device="cuda", dtype=sf_dtype
    )
    v_scale = torch.empty(
        batch_size, num_heads_kv, head_dim_v, seqlen_k_padded // pv_sf_vec, device="cuda", dtype=pv_sf_dtype
    )
    return q, k, v, q_scale, k_scale, v_scale


def _make_fake_fp4_bwd_inputs(
    *,
    fp4_qk_format: str,
    head_dim: int,
    num_heads: int = 4,
    num_heads_kv: int | None = None,
    batch_size: int = 2,
    seqlen_q: int = 128,
    seqlen_k: int = 128,
    device: str = "cuda",
):
    num_heads_kv = num_heads if num_heads_kv is None else num_heads_kv
    sf_vec = FP4_FORMAT_TO_VEC[fp4_qk_format]
    sf_dtype = FP4_FORMAT_TO_DTYPE[fp4_qk_format]
    q = torch.empty(batch_size, seqlen_q, num_heads, head_dim // 2, device=device, dtype=torch.uint8)
    k = torch.empty(batch_size, seqlen_k, num_heads_kv, head_dim // 2, device=device, dtype=torch.uint8)
    v = torch.empty(batch_size, seqlen_k, num_heads_kv, head_dim, device=device, dtype=torch.bfloat16)
    out = torch.empty(batch_size, seqlen_q, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    dout = torch.empty_like(out)
    lse = torch.empty(batch_size, num_heads, seqlen_q, device=device, dtype=torch.float32)
    q_scale = torch.empty(batch_size, seqlen_q, num_heads, head_dim // sf_vec, device=device, dtype=sf_dtype)
    k_scale = torch.empty(batch_size, seqlen_k, num_heads_kv, head_dim // sf_vec, device=device, dtype=sf_dtype)
    q_col = torch.empty(batch_size, head_dim, num_heads, (seqlen_q + 1) // 2, device=device, dtype=torch.uint8)
    k_col = torch.empty(batch_size, head_dim, num_heads_kv, (seqlen_k + 1) // 2, device=device, dtype=torch.uint8)
    q_col_scale = torch.empty(batch_size, head_dim, num_heads, (seqlen_q + sf_vec - 1) // sf_vec, device=device, dtype=sf_dtype)
    k_col_scale = torch.empty(batch_size, head_dim, num_heads_kv, (seqlen_k + sf_vec - 1) // sf_vec, device=device, dtype=sf_dtype)
    q_sg = torch.empty(batch_size, num_heads, device=device, dtype=torch.float32)
    k_sg = torch.empty(batch_size, num_heads_kv, device=device, dtype=torch.float32)
    return q, k, v, out, dout, lse, q_scale, k_scale, q_col, k_col, q_col_scale, k_col_scale, q_sg, k_sg


def _unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    packed_i32 = packed.to(torch.int32)
    return torch.stack((packed_i32 & 0xF, (packed_i32 >> 4) & 0xF), dim=-1).flatten(start_dim=-2)


def _dequantize_fp4(packed: torch.Tensor, scale: torch.Tensor, sf_vec_size: int) -> torch.Tensor:
    values = FP4_GRID.to(device=packed.device)[_unpack_fp4(packed).long()]
    return (
        values.unflatten(dim=-1, sizes=(-1, sf_vec_size))
        * scale.to(torch.float32).unsqueeze(-1)
    ).flatten(start_dim=-2)


def _fp4_pv_partner_reduce(values: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    out = values.clone()
    for lane in range(values.numel()):
        partner = lane ^ 1
        if partner < values.numel() and rows[lane] == rows[partner] and cols[lane] == cols[partner]:
            out[lane] = torch.maximum(out[lane], values[partner])
    return out


def _fp4_pv_full_warp_reduce(values: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    out = values.clone()
    for offset in (1, 2, 4, 8, 16):
        prev = out.clone()
        for lane in range(values.numel()):
            partner = lane ^ offset
            if partner < values.numel() and rows[lane] == rows[partner] and cols[lane] == cols[partner]:
                out[lane] = torch.maximum(prev[lane], prev[partner])
    return out


def _fp4_pv_exact_slot_reduce(values: torch.Tensor, slot_ids: torch.Tensor) -> torch.Tensor:
    out = values.clone()
    for lane in range(values.numel()):
        out[lane] = values[slot_ids == slot_ids[lane]].max()
    return out


def _fp4_pv_online_softmax_reference(
    score_blocks: list[torch.Tensor],
    value_blocks: list[torch.Tensor],
) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor]:
    row_max = torch.tensor(float("-inf"), dtype=torch.float32)
    row_sum = torch.tensor(0.0, dtype=torch.float32)
    out = torch.zeros(value_blocks[0].shape[-1], dtype=torch.float32)
    states = []
    for scores, values in zip(score_blocks, value_blocks):
        scores_f32 = scores.to(torch.float32)
        values_f32 = values.to(torch.float32)
        block_max = scores_f32.max()
        m_new = torch.maximum(row_max, block_max)
        alpha = (
            torch.tensor(0.0, dtype=torch.float32)
            if torch.isneginf(row_max)
            else torch.exp(row_max - m_new)
        )
        p_block = torch.exp(scores_f32 - m_new)
        l_new = alpha * row_sum + p_block.sum()
        inv_l_new = (
            torch.tensor(1.0, dtype=torch.float32)
            if (not torch.isfinite(l_new)) or l_new == 0
            else torch.tensor(1.0, dtype=torch.float32) / l_new
        )
        out = (alpha * row_sum * out + p_block @ values_f32) * inv_l_new
        states.append(
            {
                "m_new": m_new,
                "alpha": alpha,
                "l_old": row_sum,
                "l_new": l_new,
                "inv_l_new": inv_l_new,
                "scores_shifted": scores_f32 - m_new,
                "o_new": out.clone(),
            }
        )
        row_max = m_new
        row_sum = l_new
    return states, out


def _fp4_pv_online_softmax_decode_reference(
    score_blocks: list[torch.Tensor],
    value_blocks: list[torch.Tensor],
) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor]:
    return _fp4_pv_online_softmax_reference(score_blocks, value_blocks)


def _fp4_pv_online_softmax_encode_reference(
    score_blocks: list[torch.Tensor],
    value_blocks: list[torch.Tensor],
) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor]:
    return _fp4_pv_online_softmax_reference(score_blocks, value_blocks)


def _stage_fp4_vt_public_bytes_for_pv(packed_vt: torch.Tensor) -> torch.Tensor:
    """Repack public seq-packed Vt bytes into the D-packed PV shared-memory byte layout."""
    batch_size, num_heads, head_dim, seqlen_packed = packed_vt.shape
    assert head_dim % 2 == 0
    out = torch.empty(
        batch_size,
        num_heads,
        seqlen_packed * 2,
        head_dim // 2,
        dtype=torch.uint8,
        device=packed_vt.device,
    )
    packed_i32 = packed_vt.to(torch.int32)
    for row in range(seqlen_packed * 2):
        row_pair, row_parity = divmod(row, 2)
        src0 = packed_i32[:, :, 0::2, row_pair]
        src1 = packed_i32[:, :, 1::2, row_pair]
        if row_parity == 0:
            nibble0 = src0 & 0xF
            nibble1 = src1 & 0xF
        else:
            nibble0 = (src0 >> 4) & 0xF
            nibble1 = (src1 >> 4) & 0xF
        out[:, :, row] = (nibble0 | (nibble1 << 4)).to(torch.uint8)
    return out


def _swizzle_fp4_vt_scale(scale_vt: torch.Tensor) -> torch.Tensor:
    batch_size, num_heads, head_dim, seqlen_groups = scale_vt.shape
    out = torch.empty_like(scale_vt)
    flat_in = scale_vt.reshape(batch_size, num_heads, head_dim, seqlen_groups)
    flat_out = out.view(batch_size, num_heads, -1)
    for d in range(head_dim):
        tile_m, row_in_tile = divmod(d, 64)
        quad, row_mod16 = divmod(row_in_tile, 16)
        for seq_group in range(seqlen_groups):
            offset = (
                tile_m * 64 * seqlen_groups
                + (seq_group // 4) * 256
                + (seq_group % 4)
                + quad * 4
                + row_mod16 * 16
            )
            flat_out[:, :, offset] = flat_in[:, :, d, seq_group]
    return out


def _unswizzle_fp4_vt_scale(scale_vt: torch.Tensor) -> torch.Tensor:
    batch_size, num_heads, head_dim, seqlen_groups = scale_vt.shape
    out = torch.empty_like(scale_vt)
    flat_in = scale_vt.view(batch_size, num_heads, -1)
    logical = out.reshape(batch_size, num_heads, head_dim, seqlen_groups)
    for d in range(head_dim):
        tile_m, row_in_tile = divmod(d, 64)
        quad, row_mod16 = divmod(row_in_tile, 16)
        for seq_group in range(seqlen_groups):
            offset = (
                tile_m * 64 * seqlen_groups
                + (seq_group // 4) * 256
                + (seq_group % 4)
                + quad * 4
                + row_mod16 * 16
            )
            logical[:, :, d, seq_group] = flat_in[:, :, offset]
    return out


def _swizzle_tk_nvfp4_transpose_scale(scale: torch.Tensor) -> torch.Tensor:
    batch_size, head_dim, num_heads, seqlen_groups = scale.shape
    if head_dim % 64 != 0:
        raise ValueError("TK transpose-scale swizzle helper expects head_dim divisible by 64.")
    if seqlen_groups % 4 != 0:
        raise ValueError("TK transpose-scale swizzle helper expects seqlen_groups divisible by 4.")
    logical_u8 = scale.view(torch.uint8).permute(0, 2, 1, 3).contiguous()
    flat_out = torch.empty(batch_size, num_heads, head_dim * seqlen_groups, device=scale.device, dtype=torch.uint8)
    for d in range(head_dim):
        tile_m, row_in_tile = divmod(d, 64)
        j = row_in_tile % 32
        grp = row_in_tile // 32
        tile_base = tile_m * 64 * seqlen_groups
        for seq_group in range(seqlen_groups):
            offset = tile_base + (seq_group // 4) * 256 + j * 8 + grp * 4 + (seq_group % 4)
            flat_out[:, :, offset] = logical_u8[:, :, d, seq_group]
    swizzled_u8 = flat_out.view(batch_size, num_heads, head_dim, seqlen_groups).permute(0, 2, 1, 3).contiguous()
    return swizzled_u8.flatten().view(torch.float8_e4m3fn).view_as(scale)


def _dequantize_fp4_vt(packed_vt: torch.Tensor, scale_vt: torch.Tensor, sf_vec_size: int) -> torch.Tensor:
    values = FP4_GRID.to(device=packed_vt.device)[_unpack_fp4(packed_vt).long()]
    logical_scale_vt = _unswizzle_fp4_vt_scale(scale_vt)
    values = values.unflatten(dim=-1, sizes=(-1, sf_vec_size))
    values = values * logical_scale_vt.to(torch.float32).unsqueeze(-1)
    return values.flatten(start_dim=2, end_dim=3).permute(0, 3, 1, 2).contiguous()


@pytest.mark.parametrize(
    "head_dim,head_dim_v,causal",
    [
        (64, 64, False),
        (64, 64, True),
        (128, 128, False),
        (128, 128, True),
    ],
)
def test_fp4_qk_fake_compile_dense_forward(monkeypatch, head_dim, head_dim_v, causal):
    compile_calls = _install_fake_cuda_runtime(monkeypatch)

    with FakeTensorMode():
        q, k, v, q_scale, k_scale = _make_fake_fp4_dense_inputs(
            fp4_qk_format="nvfp4",
            head_dim=head_dim,
            head_dim_v=head_dim_v,
        )
        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            causal=causal,
            return_lse=True,
            _arch=100,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
        )

    assert out.dtype == v.dtype
    assert lse.dtype == torch.float32
    assert len(compile_calls) == 1
    assert len(_flash_attn_fwd.compile_cache) == 1


@pytest.mark.parametrize(
    "num_heads,num_heads_kv,head_dim,head_dim_v,causal",
    [
        (4, 4, 64, 64, False),
        (4, 4, 64, 64, True),
        (6, 2, 64, 64, False),
        (6, 2, 64, 64, True),
        (4, 4, 128, 128, False),
        (4, 4, 128, 128, True),
        (6, 2, 128, 128, False),
        (6, 2, 128, 128, True),
    ],
)
def test_fp4_pv_fake_compile_dense_forward(
    monkeypatch, num_heads, num_heads_kv, head_dim, head_dim_v, causal
):
    compile_calls = _install_fake_cuda_runtime(monkeypatch)
    exact_sage_lane = (
        num_heads == num_heads_kv == 4 and head_dim == 128 and head_dim_v == 128 and not causal
    )
    fp4_pv_format = "nvfp4" if exact_sage_lane else None

    with FakeTensorMode():
        q, k, v, q_scale, k_scale, v_scale = _make_fake_fp4_pv_dense_inputs(
            fp4_qk_format="nvfp4",
            fp4_pv_format=fp4_pv_format,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            num_heads=num_heads,
            num_heads_kv=num_heads_kv,
        )
        if exact_sage_lane:
            out, lse = _flash_attn_fwd(
                q,
                k,
                v,
                causal=causal,
                return_lse=True,
                _arch=100,
                fp4_qk_format="nvfp4",
                q_scale=q_scale,
                k_scale=k_scale,
                use_fp4_pv=True,
                v_scale=v_scale,
            )

            assert out.dtype == torch.bfloat16
            assert lse.dtype == torch.float32
            assert len(compile_calls) == 1
            assert len(_flash_attn_fwd.compile_cache) == 1
        else:
            with pytest.raises(NotImplementedError, match="exact Sage-style FP4 PV rewrite"):
                _flash_attn_fwd(
                    q,
                    k,
                    v,
                    causal=causal,
                    return_lse=True,
                    _arch=100,
                    fp4_qk_format="nvfp4",
                    q_scale=q_scale,
                    k_scale=k_scale,
                    use_fp4_pv=True,
                    v_scale=v_scale,
                )


def test_fp4_pv_fused_fake_compile_dense_forward(monkeypatch):
    _install_fake_cuda_runtime(monkeypatch)
    kernel_kwargs = {}

    def fake_pvfused_kernel(*_args, **kwargs):
        kernel_kwargs.update(kwargs)
        return object()

    def fail_regular_pv_kernel(*_args, **_kwargs):
        raise AssertionError("PV experiment should not instantiate the regular FP4 PV kernel.")

    monkeypatch.setattr("flash_attn.cute.interface.FP4FlashAttentionForwardSm100PVFused", fake_pvfused_kernel)
    monkeypatch.setattr("flash_attn.cute.interface.FP4FlashAttentionForwardSm100", fail_regular_pv_kernel)

    with FakeTensorMode():
        q, k, v, q_scale, k_scale, v_scale = _make_fake_fp4_pv_dense_inputs(
            fp4_qk_format="nvfp4",
            fp4_pv_format="nvfp4",
            head_dim=128,
            head_dim_v=128,
            num_heads=4,
            num_heads_kv=4,
            seqlen_q=512,
            seqlen_k=512,
        )
        _flash_attn_fwd(
            q,
            k,
            v,
            causal=False,
            _arch=100,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
            use_fp4_pv=True,
            v_scale=v_scale,
        )

    assert kernel_kwargs["use_fp4_pv"] is True
    assert kernel_kwargs["fp4_sf_dtype"] == "e4m3"
    assert kernel_kwargs["fp4_sf_vec_size"] == 16
    assert kernel_kwargs["pv_sf_dtype"] == "e4m3"
    assert kernel_kwargs["pv_sf_vec_size"] == 16
    assert kernel_kwargs["qhead_per_kvhead"] == 1
    assert kernel_kwargs["pack_gqa"] is False
    assert kernel_kwargs["use_2cta_instrs"] is False


def _instantiate_fake_exact_fp4_pv_fused_kernel(monkeypatch):
    _install_fake_cuda_runtime(monkeypatch)
    from flash_attn.cute.fp4_flash_fwd_sm100_pvfused import FP4FlashAttentionForwardSm100PVFused

    return FP4FlashAttentionForwardSm100PVFused(
        head_dim=128,
        head_dim_v=128,
        qhead_per_kvhead=1,
        is_causal=False,
        is_local=False,
        is_split_kv=False,
        pack_gqa=False,
        q_subtile_factor=None,
        m_block_size=128,
        n_block_size=128,
        q_stage=1,
        is_persistent=True,
        score_mod=None,
        mask_mod=None,
        has_aux_tensors=False,
        paged_kv_non_tma=False,
        is_varlen_q=False,
        use_2cta_instrs=False,
        use_fp4_qk=True,
        use_fp4_pv=True,
        fp4_sf_dtype="e4m3",
        fp4_sf_vec_size=16,
        pv_sf_dtype="e4m3",
        pv_sf_vec_size=16,
        pack_gqa_local=False,
        group_qheads_by_kv=False,
        fp4_pv_fp32_online_rescale=False,
    )


def test_fp4_pv_fused_exact_lane_uses_compact_warp_map(monkeypatch):
    kernel = _instantiate_fake_exact_fp4_pv_fused_kernel(monkeypatch)
    assert kernel.threads_per_cta == 288
    assert kernel.softmax0_warp_ids == (0, 1, 2, 3)
    assert kernel.softmax1_warp_ids == ()
    assert kernel.correction_warp_ids == (4, 5)
    assert kernel.mma_warp_id == 6
    assert kernel.epilogue_warp_ids == (7,)
    assert kernel.load_warp_ids == (8,)
    assert kernel.empty_warp_ids == ()


def test_fp4_pv_fused_exact_lane_uses_split_handoffs(monkeypatch):
    kernel = _instantiate_fake_exact_fp4_pv_fused_kernel(monkeypatch)
    assert kernel.use_exact_fp4_pv_lane is True
    assert kernel.use_exact_fp4_pv_s_ready_handoff is True
    assert kernel.use_exact_fp4_pv_p_ready_handoff is True


def test_fp4_pv_fused_exact_lane_skips_legacy_stats_pipeline(monkeypatch):
    kernel = _instantiate_fake_exact_fp4_pv_fused_kernel(monkeypatch)
    assert kernel.use_exact_fp4_pv_lane is True
    assert kernel.use_exact_fp4_pv_legacy_stats_pipeline is False


def test_fp4_pv_fused_exact_lane_supports_exact_sfv_direct_opt_in(monkeypatch):
    monkeypatch.setenv("FLASH_ATTN_FP4_PV_EXACT_SFV_DIRECT", "1")
    kernel = _instantiate_fake_exact_fp4_pv_fused_kernel(monkeypatch)
    assert kernel.use_exact_fp4_pv_lane is True
    assert kernel.fp4_pv_exact_sfv_direct is True


def test_fp4_pv_fused_dispatch_ignores_removed_legacy_selector(monkeypatch):
    _install_fake_cuda_runtime(monkeypatch)
    calls = []

    def fake_fused_kernel(*_args, **kwargs):
        calls.append(("fused", kwargs))
        return object()

    monkeypatch.setattr("flash_attn.cute.interface.FP4FlashAttentionForwardSm100PVFused", fake_fused_kernel)

    with FakeTensorMode():
        q, k, v, q_scale, k_scale, v_scale = _make_fake_fp4_pv_dense_inputs(
            fp4_qk_format="nvfp4",
            fp4_pv_format="nvfp4",
            head_dim=128,
            head_dim_v=128,
            num_heads=4,
            num_heads_kv=4,
            seqlen_q=512,
            seqlen_k=512,
        )
        _flash_attn_fwd(
            q,
            k,
            v,
            causal=False,
            _arch=100,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
            use_fp4_pv=True,
            v_scale=v_scale,
        )

    assert not hasattr(__import__("flash_attn.cute.interface", fromlist=["_get_internal_fp4_pv_impl"]), "_get_internal_fp4_pv_impl")
    assert calls and calls[0][0] == "fused"


@pytest.mark.parametrize(
    "head_dim,causal",
    [
        (64, False),
        (64, True),
        (128, False),
        (128, True),
    ],
)
def test_fp4_bwd_qk_fake_compile_dense_native(monkeypatch, head_dim, causal):
    compile_calls = _install_fake_cuda_runtime(monkeypatch)
    monkeypatch.setattr("flash_attn.cute.interface._get_device_arch", lambda: 100)
    monkeypatch.setenv("FLASH_ATTN_FP4_BWD_ENABLE_NATIVE", "1")

    with FakeTensorMode():
        (
            q,
            k,
            v,
            out,
            dout,
            lse,
            q_scale,
            k_scale,
            q_col,
            k_col,
            q_col_scale,
            k_col_scale,
            q_sg,
            k_sg,
        ) = _make_fake_fp4_bwd_inputs(
            fp4_qk_format="nvfp4",
            head_dim=head_dim,
            num_heads=4,
            batch_size=2,
            seqlen_q=64,
            seqlen_k=64,
        )
        dq, dk, dv = _flash_attn_bwd_fp4_qk(
            q,
            k,
            v,
            out,
            dout,
            lse,
            q_scale,
            k_scale,
            q_sg,
            k_sg,
            causal=causal,
            q_col_packed=q_col,
            k_col_packed=k_col,
            q_col_scale=q_col_scale,
            k_col_scale=k_col_scale,
        )

    assert dq.dtype == torch.bfloat16
    assert dk.dtype == torch.bfloat16
    assert dv.dtype == torch.bfloat16
    assert len(compile_calls) == 3
    assert len(_flash_attn_bwd.compile_cache) == 1


def test_fp4_bwd_qk_fake_compile_native_cache_separates_transpose_layouts(monkeypatch):
    _install_fake_cuda_runtime(monkeypatch)
    monkeypatch.setattr("flash_attn.cute.interface._get_device_arch", lambda: 100)
    monkeypatch.setenv("FLASH_ATTN_FP4_BWD_ENABLE_NATIVE", "1")

    with FakeTensorMode():
        inputs_h2 = _make_fake_fp4_bwd_inputs(
            fp4_qk_format="nvfp4",
            head_dim=64,
            num_heads=2,
            batch_size=1,
            seqlen_q=64,
            seqlen_k=64,
        )
        q, k, v, out, dout, lse, q_scale, k_scale, q_col, k_col, q_col_scale, k_col_scale, q_sg, k_sg = inputs_h2
        _flash_attn_bwd_fp4_qk(
            q,
            k,
            v,
            out,
            dout,
            lse,
            q_scale,
            k_scale,
            q_sg,
            k_sg,
            q_col_packed=q_col,
            k_col_packed=k_col,
            q_col_scale=q_col_scale,
            k_col_scale=k_col_scale,
        )
        bwd_cache_after_h2 = len(_flash_attn_bwd.compile_cache)

        inputs_h4 = _make_fake_fp4_bwd_inputs(
            fp4_qk_format="nvfp4",
            head_dim=64,
            num_heads=4,
            batch_size=1,
            seqlen_q=64,
            seqlen_k=64,
        )
        q, k, v, out, dout, lse, q_scale, k_scale, q_col, k_col, q_col_scale, k_col_scale, q_sg, k_sg = inputs_h4
        _flash_attn_bwd_fp4_qk(
            q,
            k,
            v,
            out,
            dout,
            lse,
            q_scale,
            k_scale,
            q_sg,
            k_sg,
            q_col_packed=q_col,
            k_col_packed=k_col,
            q_col_scale=q_col_scale,
            k_col_scale=k_col_scale,
        )

    assert bwd_cache_after_h2 == 1
    assert len(_flash_attn_bwd.compile_cache) == 2


def test_fp4_compile_cache_separates_bf16_and_fp4(monkeypatch):
    compile_calls = _install_fake_cuda_runtime(monkeypatch)

    with FakeTensorMode():
        bf16_q, bf16_k, bf16_v = _make_fake_dense_inputs(head_dim=64, head_dim_v=64)
        _flash_attn_fwd(bf16_q, bf16_k, bf16_v, causal=False, _arch=100)

        q, k, v, q_scale, k_scale = _make_fake_fp4_dense_inputs(
            fp4_qk_format="nvfp4",
            head_dim=64,
            head_dim_v=64,
        )
        _flash_attn_fwd(
            q,
            k,
            v,
            causal=False,
            _arch=100,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
        )
        _flash_attn_fwd(
            q,
            k,
            v,
            causal=False,
            _arch=100,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
        )

    assert len(compile_calls) == 2
    assert len(_flash_attn_fwd.compile_cache) == 2


def test_fp4_compile_cache_separates_qk_only_and_qk_pv(monkeypatch):
    compile_calls = _install_fake_cuda_runtime(monkeypatch)

    with FakeTensorMode():
        q, k, v_bf16, q_scale, k_scale = _make_fake_fp4_dense_inputs(
            fp4_qk_format="nvfp4",
            head_dim=128,
            head_dim_v=128,
        )
        _flash_attn_fwd(
            q,
            k,
            v_bf16,
            causal=False,
            _arch=100,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
        )

        q, k, v_packed, q_scale, k_scale, v_scale = _make_fake_fp4_pv_dense_inputs(
            fp4_qk_format="nvfp4",
            head_dim=128,
            head_dim_v=128,
        )
        _flash_attn_fwd(
            q,
            k,
            v_packed,
            causal=False,
            _arch=100,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
            use_fp4_pv=True,
            v_scale=v_scale,
        )
        _flash_attn_fwd(
            q,
            k,
            v_packed,
            causal=False,
            _arch=100,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
            use_fp4_pv=True,
            v_scale=v_scale,
        )

    assert len(compile_calls) == 2
    assert len(_flash_attn_fwd.compile_cache) == 2


def test_fp4_compile_cache_separates_pv_loader_modes(monkeypatch):
    compile_calls = _install_fake_cuda_runtime(monkeypatch)

    with FakeTensorMode():
        q, k, v_packed, q_scale, k_scale, v_scale = _make_fake_fp4_pv_dense_inputs(
            fp4_qk_format="nvfp4",
            head_dim=64,
            head_dim_v=64,
        )
        monkeypatch.delenv("FLASH_ATTN_FP4_PV_DIRECT_LOADER", raising=False)
        _flash_attn_fwd(
            q,
            k,
            v_packed,
            causal=False,
            _arch=100,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
            use_fp4_pv=True,
            v_scale=v_scale,
        )
        monkeypatch.setenv("FLASH_ATTN_FP4_PV_DIRECT_LOADER", "1")
        _flash_attn_fwd(
            q,
            k,
            v_packed,
            causal=False,
            _arch=100,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
            use_fp4_pv=True,
            v_scale=v_scale,
        )

    assert len(compile_calls) == 2
    assert len(_flash_attn_fwd.compile_cache) == 2


def test_fp4_pv_online_softmax_reference_matches_closed_form():
    score_blocks = [
        torch.tensor([0.0, -0.5, 1.0], dtype=torch.float32),
        torch.tensor([2.0, -1.0], dtype=torch.float32),
    ]
    value_blocks = [
        torch.tensor([[1.0, 0.0], [0.0, 2.0], [2.0, 1.0]], dtype=torch.float32),
        torch.tensor([[3.0, 4.0], [5.0, 6.0]], dtype=torch.float32),
    ]

    states, out = _fp4_pv_online_softmax_reference(score_blocks, value_blocks)

    scores_full = torch.cat(score_blocks)
    values_full = torch.cat(value_blocks, dim=0)
    probs_full = torch.softmax(scores_full, dim=0)
    out_expected = probs_full @ values_full
    lse_expected = torch.logsumexp(scores_full, dim=0)

    torch.testing.assert_close(out, out_expected)
    torch.testing.assert_close(states[-1]["m_new"] + torch.log(states[-1]["l_new"]), lse_expected)


def test_fp4_pv_online_softmax_reference_keeps_running_state_in_float32():
    score_blocks = [torch.tensor([0.0, 1.0], dtype=torch.float32)]
    value_blocks = [torch.tensor([[1.0], [2.0]], dtype=torch.float32)]

    states, _ = _fp4_pv_online_softmax_reference(score_blocks, value_blocks)

    for key in ("m_new", "alpha", "l_old", "l_new", "inv_l_new", "o_new"):
        assert states[0][key].dtype == torch.float32


def test_fp4_pv_online_softmax_decode_and_encode_reference_share_recurrence():
    score_blocks = [
        torch.tensor([0.25, -1.0, 1.5], dtype=torch.float32),
        torch.tensor([-0.75, 0.5], dtype=torch.float32),
    ]
    value_blocks = [
        torch.tensor([[1.0, 2.0], [2.0, 3.0], [4.0, 5.0]], dtype=torch.float32),
        torch.tensor([[6.0, 7.0], [8.0, 9.0]], dtype=torch.float32),
    ]

    decode_states, decode_out = _fp4_pv_online_softmax_decode_reference(score_blocks, value_blocks)
    encode_states, encode_out = _fp4_pv_online_softmax_encode_reference(score_blocks, value_blocks)

    torch.testing.assert_close(decode_out, encode_out)
    assert len(decode_states) == len(encode_states)
    for decode_state, encode_state in zip(decode_states, encode_states):
        for key in ("m_new", "alpha", "l_old", "l_new", "inv_l_new", "o_new"):
            torch.testing.assert_close(decode_state[key], encode_state[key])


def test_fp4_pv_online_softmax_encode_centric_rescale_contract_matches_reference():
    score_blocks = [
        torch.tensor([0.0, -0.5, 1.0], dtype=torch.float32),
        torch.tensor([2.0, -1.0], dtype=torch.float32),
    ]
    value_blocks = [
        torch.tensor([[1.0, 0.0], [0.0, 2.0], [2.0, 1.0]], dtype=torch.float32),
        torch.tensor([[3.0, 4.0], [5.0, 6.0]], dtype=torch.float32),
    ]
    encode_scales = [torch.tensor(3.0, dtype=torch.float32), torch.tensor(5.0, dtype=torch.float32)]

    states, _ = _fp4_pv_online_softmax_reference(score_blocks, value_blocks)
    encoded_acc = None
    prev_scale = torch.tensor(1.0, dtype=torch.float32)
    for state, value_block, encode_scale in zip(states, value_blocks, encode_scales):
        p_block = torch.exp(state["scores_shifted"])
        pv_block = p_block @ value_block
        if encoded_acc is None:
            encoded_acc = encode_scale * pv_block
        else:
            correction_scale = state["alpha"] * encode_scale / prev_scale
            encoded_acc = correction_scale * encoded_acc + encode_scale * pv_block
        recovered = encoded_acc * (state["inv_l_new"] / encode_scale)
        torch.testing.assert_close(recovered, state["o_new"])
        prev_scale = encode_scale


def test_fp4_pv_cta_quant_reference_warp_reduce_matches_full_block_amax():
    values = torch.tensor(
        [
            1.0, 5.0, 3.0, 7.0, 2.0, 4.0, 6.0, 8.0,
            9.0, 12.0, 10.0, 11.0, 13.0, 15.0, 14.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
            25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
        ],
        dtype=torch.float32,
    )
    rows = torch.tensor(
        [0] * 8 + [1] * 8 + list(range(16, 32)),
        dtype=torch.int64,
    )
    cols = torch.zeros(32, dtype=torch.int64)

    legacy = _fp4_pv_partner_reduce(values, rows, cols)
    cta = _fp4_pv_full_warp_reduce(values, rows, cols)

    assert torch.equal(cta[:8], torch.full((8,), 8.0))
    assert torch.equal(cta[8:16], torch.full((8,), 16.0))
    assert torch.equal(cta[16:], values[16:])
    assert not torch.equal(legacy[:8], cta[:8])
    assert not torch.equal(legacy[8:16], cta[8:16])


def test_fp4_pv_cta_quant_reference_masked_slot_reduce_matches_exact_amax():
    values = torch.tensor(
        [
            1.0,
            -torch.inf,
            3.0,
            7.0,
            2.0,
            9.0,
            4.0,
            5.0,
            8.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
            18.0,
            19.0,
            20.0,
            21.0,
            22.0,
            23.0,
            24.0,
            25.0,
            26.0,
            27.0,
            28.0,
            29.0,
            30.0,
            31.0,
            32.0,
        ],
        dtype=torch.float32,
    )
    slot_ids = torch.arange(32, dtype=torch.int64)
    slot_ids[[0, 3, 5]] = 0
    slot_ids[[1, 6]] = 1
    rows = slot_ids.clone()
    cols = torch.zeros_like(slot_ids)

    widened = _fp4_pv_full_warp_reduce(values, rows, cols)
    exact = _fp4_pv_exact_slot_reduce(values, slot_ids)

    assert torch.equal(exact[[0, 3, 5]], torch.full((3,), 9.0))
    assert torch.equal(exact[[1, 6]], torch.full((2,), 4.0))
    assert not torch.equal(widened[[0, 3, 5]], exact[[0, 3, 5]])
    assert not torch.equal(widened[[1, 6]], exact[[1, 6]])


def test_fp4_pv_cta_quant_reference_storage_slot_key_avoids_coarse_alias():
    values = torch.tensor([1.0, 9.0, 4.0, 7.0], dtype=torch.float32)
    coarse_slot_ids = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
    storage_slot_ids = torch.tensor([100, 100, 200, 200], dtype=torch.int64)

    coarse = _fp4_pv_exact_slot_reduce(values, coarse_slot_ids)
    exact = _fp4_pv_exact_slot_reduce(values, storage_slot_ids)

    assert torch.equal(coarse, torch.full((4,), 9.0))
    assert torch.equal(exact[:2], torch.full((2,), 9.0))
    assert torch.equal(exact[2:], torch.full((2,), 7.0))


def test_fp4_pv_direct_loader_byte_repack_matches_logical_vt_values():
    batch_size, num_heads, head_dim, seqlen = 1, 2, 64, 128
    logical_codes = torch.arange(batch_size * num_heads * head_dim * seqlen, dtype=torch.uint8)
    logical_codes = (logical_codes % 16).view(batch_size, num_heads, head_dim, seqlen)
    packed_vt = ((logical_codes[..., 1::2] << 4) | logical_codes[..., 0::2]).contiguous()

    staged = _stage_fp4_vt_public_bytes_for_pv(packed_vt)
    expected = _unpack_fp4(packed_vt).permute(0, 1, 3, 2).contiguous()
    staged_values = _unpack_fp4(staged)

    assert torch.equal(staged_values, expected)


def test_fp4_pv_public_vt_scale_unswizzle_roundtrips():
    logical = torch.linspace(
        0.25,
        8.0,
        steps=1 * 2 * 64 * 8,
        dtype=torch.float32,
    ).reshape(1, 2, 64, 8).to(torch.float8_e4m3fn)
    swizzled = _swizzle_fp4_vt_scale(logical)
    roundtrip = _unswizzle_fp4_vt_scale(swizzled)

    assert torch.equal(roundtrip.view(torch.uint8), logical.view(torch.uint8))


def test_fp4_compile_cache_separates_pv_cta_quant_variants(monkeypatch):
    compile_calls = _install_fake_cuda_runtime(monkeypatch)

    with FakeTensorMode():
        q, k, v_packed, q_scale, k_scale, v_scale = _make_fake_fp4_pv_dense_inputs(
            fp4_qk_format="nvfp4",
            head_dim=64,
            head_dim_v=64,
        )
        monkeypatch.delenv("FLASH_ATTN_FP4_PV_ENABLE_CTA_ENCODE", raising=False)
        _flash_attn_fwd(
            q,
            k,
            v_packed,
            causal=False,
            _arch=100,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
            use_fp4_pv=True,
            v_scale=v_scale,
        )
        monkeypatch.setenv("FLASH_ATTN_FP4_PV_ENABLE_CTA_ENCODE", "1")
        _flash_attn_fwd(
            q,
            k,
            v_packed,
            causal=False,
            _arch=100,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
            use_fp4_pv=True,
            v_scale=v_scale,
        )

    assert len(compile_calls) == 2
    assert len(_flash_attn_fwd.compile_cache) == 2


def test_fp4_compile_cache_separates_pv_cta_decode_and_encode_centric(monkeypatch):
    compile_calls = _install_fake_cuda_runtime(monkeypatch)

    with FakeTensorMode():
        q, k, v_packed, q_scale, k_scale, v_scale = _make_fake_fp4_pv_dense_inputs(
            fp4_qk_format="nvfp4",
            head_dim=64,
            head_dim_v=64,
        )
        monkeypatch.setenv("FLASH_ATTN_FP4_PV_ENABLE_CTA_ENCODE", "1")
        monkeypatch.delenv("FLASH_ATTN_FP4_PV_ENCODE_CENTRIC", raising=False)
        _flash_attn_fwd(
            q,
            k,
            v_packed,
            causal=False,
            _arch=100,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
            use_fp4_pv=True,
            v_scale=v_scale,
        )
        monkeypatch.setenv("FLASH_ATTN_FP4_PV_ENCODE_CENTRIC", "1")
        _flash_attn_fwd(
            q,
            k,
            v_packed,
            causal=False,
            _arch=100,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
            use_fp4_pv=True,
            v_scale=v_scale,
        )

    assert len(compile_calls) == 2
    assert len(_flash_attn_fwd.compile_cache) == 2


def test_fp4_compile_cache_separates_exact_sfv_direct_variants(monkeypatch):
    compile_calls = _install_fake_cuda_runtime(monkeypatch)
    monkeypatch.setattr("torch._subclasses.fake_tensor.init_gpu_context", lambda _device: None)

    with FakeTensorMode():
        q, k, v_packed, q_scale, k_scale, v_scale = _make_fake_fp4_pv_dense_inputs(
            fp4_qk_format="nvfp4",
            head_dim=128,
            head_dim_v=128,
            seqlen_q=512,
            seqlen_k=512,
        )
        monkeypatch.delenv("FLASH_ATTN_FP4_PV_EXACT_SFV_DIRECT", raising=False)
        _flash_attn_fwd(
            q,
            k,
            v_packed,
            causal=False,
            _arch=100,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
            use_fp4_pv=True,
            v_scale=v_scale,
        )
        monkeypatch.setenv("FLASH_ATTN_FP4_PV_EXACT_SFV_DIRECT", "1")
        _flash_attn_fwd(
            q,
            k,
            v_packed,
            causal=False,
            _arch=100,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
            use_fp4_pv=True,
            v_scale=v_scale,
        )

    assert len(compile_calls) == 2
    assert len(_flash_attn_fwd.compile_cache) == 2


def test_fp4_pv_cta_quant_direct_loader_dispatches_cta_quantizer(monkeypatch):
    _install_fake_cuda_runtime(monkeypatch)
    monkeypatch.setenv("FLASH_ATTN_FP4_PV_DIRECT_LOADER", "1")
    monkeypatch.setenv("FLASH_ATTN_FP4_PV_ENABLE_CTA_ENCODE", "1")
    kernel_state = {}
    real_kernel_cls = _flash_attn_fwd.__globals__["FP4FlashAttentionForwardSm100"]

    def wrapped_kernel(*args, **kwargs):
        kernel = real_kernel_cls(*args, **kwargs)
        kernel_state["fp4_pv_direct_loader"] = kernel.fp4_pv_direct_loader
        kernel_state["fp4_pv_cta_quant"] = kernel.fp4_pv_cta_quant
        return kernel

    monkeypatch.setitem(_flash_attn_fwd.__globals__, "FP4FlashAttentionForwardSm100", wrapped_kernel)

    with FakeTensorMode():
        q, k, v_packed, q_scale, k_scale, v_scale = _make_fake_fp4_pv_dense_inputs(
            fp4_qk_format="nvfp4",
            head_dim=64,
            head_dim_v=64,
        )
        _flash_attn_fwd(
            q,
            k,
            v_packed,
            causal=False,
            _arch=100,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
            use_fp4_pv=True,
            v_scale=v_scale,
        )

    assert kernel_state["fp4_pv_direct_loader"] is True
    assert kernel_state["fp4_pv_cta_quant"] is True


def test_fp4_pv_cta_quant_encode_centric_dispatches_kernel_mode(monkeypatch):
    _install_fake_cuda_runtime(monkeypatch)
    monkeypatch.setenv("FLASH_ATTN_FP4_PV_ENABLE_CTA_ENCODE", "1")
    monkeypatch.setenv("FLASH_ATTN_FP4_PV_ENCODE_CENTRIC", "1")
    kernel_state = {}
    real_kernel_cls = _flash_attn_fwd.__globals__["FP4FlashAttentionForwardSm100"]

    def wrapped_kernel(*args, **kwargs):
        kernel = real_kernel_cls(*args, **kwargs)
        kernel_state["fp4_pv_cta_quant"] = kernel.fp4_pv_cta_quant
        kernel_state["fp4_pv_encode_centric_requested"] = kernel.fp4_pv_encode_centric_requested
        kernel_state["fp4_pv_encode_centric"] = kernel.fp4_pv_encode_centric
        return kernel

    monkeypatch.setitem(_flash_attn_fwd.__globals__, "FP4FlashAttentionForwardSm100", wrapped_kernel)

    with FakeTensorMode():
        q, k, v_packed, q_scale, k_scale, v_scale = _make_fake_fp4_pv_dense_inputs(
            fp4_qk_format="nvfp4",
            head_dim=64,
            head_dim_v=64,
        )
        _flash_attn_fwd(
            q,
            k,
            v_packed,
            causal=False,
            _arch=100,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
            use_fp4_pv=True,
            v_scale=v_scale,
        )

    assert kernel_state["fp4_pv_cta_quant"] is True
    assert kernel_state["fp4_pv_encode_centric_requested"] is True
    assert kernel_state["fp4_pv_encode_centric"] is False


def test_fp4_d128_noncausal_uses_2cta_schedule(monkeypatch):
    _install_fake_cuda_runtime(monkeypatch)
    kernel_kwargs = {}

    def fake_fp4_kernel(*_args, **kwargs):
        kernel_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr("flash_attn.cute.interface.FP4FlashAttentionForwardSm100MHAFast", fake_fp4_kernel)

    with FakeTensorMode():
        q, k, v, q_scale, k_scale = _make_fake_fp4_dense_inputs(
            fp4_qk_format="nvfp4",
            head_dim=128,
            head_dim_v=128,
            seqlen_q=512,
            seqlen_k=512,
        )
        _flash_attn_fwd(
            q,
            k,
            v,
            causal=False,
            _arch=100,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
        )

    assert kernel_kwargs["q_stage"] == 1
    assert kernel_kwargs["use_2cta_instrs"] is True


def test_fp4_gqa_d128_noncausal_explicit_pack_gqa_uses_2cta(monkeypatch):
    _install_fake_cuda_runtime(monkeypatch)
    kernel_kwargs = {}

    def fake_fp4_kernel(*_args, **kwargs):
        kernel_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr("flash_attn.cute.interface.FP4FlashAttentionForwardSm100GQAFast", fake_fp4_kernel)

    with FakeTensorMode():
        q, k, v, q_scale, k_scale = _make_fake_fp4_dense_inputs(
            fp4_qk_format="nvfp4",
            head_dim=128,
            head_dim_v=128,
            num_heads=6,
            num_heads_kv=2,
            seqlen_q=512,
            seqlen_k=512,
        )
        _flash_attn_fwd(
            q,
            k,
            v,
            causal=False,
            _arch=100,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
            pack_gqa=True,
        )

    assert kernel_kwargs["pack_gqa"] is True
    assert kernel_kwargs["pack_gqa_local"] is False
    assert kernel_kwargs["q_stage"] == 1
    assert kernel_kwargs["use_2cta_instrs"] is True


def test_fp4_gqa_d128_noncausal_default_uses_grouped_kv_reuse(monkeypatch):
    _install_fake_cuda_runtime(monkeypatch)
    kernel_kwargs = {}

    def fake_fp4_kernel(*_args, **kwargs):
        kernel_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr("flash_attn.cute.interface.FP4FlashAttentionForwardSm100GQAFast", fake_fp4_kernel)

    with FakeTensorMode():
        q, k, v, q_scale, k_scale = _make_fake_fp4_dense_inputs(
            fp4_qk_format="nvfp4",
            head_dim=128,
            head_dim_v=128,
            num_heads=6,
            num_heads_kv=2,
            seqlen_q=512,
            seqlen_k=512,
        )
        _flash_attn_fwd(
            q,
            k,
            v,
            causal=False,
            _arch=100,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
        )

    assert kernel_kwargs["pack_gqa"] is False
    assert kernel_kwargs["pack_gqa_local"] is False
    assert kernel_kwargs["group_qheads_by_kv"] is True
    assert kernel_kwargs["m_block_size"] == 128
    assert kernel_kwargs["n_block_size"] == 128
    assert kernel_kwargs["q_stage"] == 1
    assert kernel_kwargs["use_2cta_instrs"] is False
    assert kernel_kwargs["is_persistent"] is True


def test_fp4_gqa_d128_noncausal_local_pack_experiment_is_internally_selectable(monkeypatch):
    _install_fake_cuda_runtime(monkeypatch)
    monkeypatch.setenv("FLASH_ATTN_FP4_Q3_LOCAL_PACK", "1")
    monkeypatch.setattr("flash_attn.cute.interface.to_cute_tensor", lambda tensor, *args, **kwargs: tensor)
    kernel_kwargs = {}

    def fake_fp4_kernel(*_args, **kwargs):
        kernel_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr("flash_attn.cute.interface.FP4FlashAttentionForwardSm100GQAFast", fake_fp4_kernel)

    with FakeTensorMode():
        q, k, v, q_scale, k_scale = _make_fake_fp4_dense_inputs(
            fp4_qk_format="nvfp4",
            head_dim=128,
            head_dim_v=128,
            num_heads=6,
            num_heads_kv=2,
            seqlen_q=512,
            seqlen_k=512,
        )
        _flash_attn_fwd(
            q,
            k,
            v,
            causal=False,
            _arch=100,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
        )

    assert kernel_kwargs["pack_gqa"] is False
    assert kernel_kwargs["pack_gqa_local"] is True
    assert kernel_kwargs["group_qheads_by_kv"] is False
    assert kernel_kwargs["q_stage"] == 1
    assert kernel_kwargs["use_2cta_instrs"] is False
    assert kernel_kwargs["is_persistent"] is True


def test_fp4_causal_d128_keeps_bringup_schedule(monkeypatch):
    _install_fake_cuda_runtime(monkeypatch)
    kernel_kwargs = {}

    def fake_fp4_kernel(*_args, **kwargs):
        kernel_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr("flash_attn.cute.interface.FP4FlashAttentionForwardSm100MHAFast", fake_fp4_kernel)

    with FakeTensorMode():
        q, k, v, q_scale, k_scale = _make_fake_fp4_dense_inputs(
            fp4_qk_format="nvfp4",
            head_dim=128,
            head_dim_v=128,
            seqlen_q=512,
            seqlen_k=512,
        )
        _flash_attn_fwd(
            q,
            k,
            v,
            causal=True,
            _arch=100,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
        )

    assert kernel_kwargs["q_stage"] == 1
    assert kernel_kwargs["use_2cta_instrs"] is False


def test_fp4_pv_d128_noncausal_keeps_bringup_schedule(monkeypatch):
    _install_fake_cuda_runtime(monkeypatch)
    kernel_kwargs = {}

    def fake_fp4_kernel(*_args, **kwargs):
        kernel_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr("flash_attn.cute.interface.FP4FlashAttentionForwardSm100", fake_fp4_kernel)

    with FakeTensorMode():
        q, k, v, q_scale, k_scale, v_scale = _make_fake_fp4_pv_dense_inputs(
            fp4_qk_format="nvfp4",
            head_dim=128,
            head_dim_v=128,
            num_heads=6,
            num_heads_kv=2,
            seqlen_q=512,
            seqlen_k=512,
        )
        _flash_attn_fwd(
            q,
            k,
            v,
            causal=False,
            _arch=100,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
            use_fp4_pv=True,
            v_scale=v_scale,
        )

    assert kernel_kwargs["use_fp4_pv"] is True
    assert kernel_kwargs["q_stage"] == 1
    assert kernel_kwargs["use_2cta_instrs"] is False
    assert kernel_kwargs["pack_gqa_local"] is False


def test_fp4_use_fp4_pv_dispatch_selects_split_kernel(monkeypatch):
    _install_fake_cuda_runtime(monkeypatch)
    calls = []

    def fake_nonpv_kernel(*_args, **kwargs):
        calls.append(("nonpv", kwargs))
        return object()

    def fake_pv_kernel(*_args, **kwargs):
        calls.append(("pv", kwargs))
        return object()

    def fake_pvfused_kernel(*_args, **kwargs):
        calls.append(("pvfused", kwargs))
        return object()

    monkeypatch.setattr("flash_attn.cute.interface.FP4FlashAttentionForwardSm100MHAFast", fake_nonpv_kernel)
    monkeypatch.setattr("flash_attn.cute.interface.FP4FlashAttentionForwardSm100GQAFast", fake_nonpv_kernel)
    monkeypatch.setattr("flash_attn.cute.interface.FP4FlashAttentionForwardSm100", fake_pv_kernel)
    monkeypatch.setattr("flash_attn.cute.interface.FP4FlashAttentionForwardSm100PVFused", fake_pvfused_kernel)

    with FakeTensorMode():
        q, k, v, q_scale, k_scale = _make_fake_fp4_dense_inputs(
            fp4_qk_format="nvfp4",
            head_dim=128,
            head_dim_v=128,
            seqlen_q=512,
            seqlen_k=512,
        )
        _flash_attn_fwd(
            q,
            k,
            v,
            causal=False,
            _arch=100,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
        )

    assert calls[0][0] == "nonpv"
    assert "use_fp4_pv" not in calls[0][1]
    calls.clear()

    with FakeTensorMode():
        q, k, v, q_scale, k_scale, v_scale = _make_fake_fp4_pv_dense_inputs(
            fp4_qk_format="nvfp4",
            head_dim=64,
            head_dim_v=64,
            seqlen_q=512,
            seqlen_k=512,
        )
        with pytest.raises(NotImplementedError, match="exact Sage-style FP4 PV rewrite"):
            _flash_attn_fwd(
                q,
                k,
                v,
                causal=False,
                _arch=100,
                fp4_qk_format="nvfp4",
                q_scale=q_scale,
                k_scale=k_scale,
                use_fp4_pv=True,
                v_scale=v_scale,
            )
    assert not calls
    calls.clear()

    with FakeTensorMode():
        q, k, v, q_scale, k_scale, v_scale = _make_fake_fp4_pv_dense_inputs(
            fp4_qk_format="nvfp4",
            fp4_pv_format="nvfp4",
            head_dim=128,
            head_dim_v=128,
            seqlen_q=512,
            seqlen_k=512,
        )
        _flash_attn_fwd(
            q,
            k,
            v,
            causal=False,
            _arch=100,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
            use_fp4_pv=True,
            v_scale=v_scale,
        )

    assert calls[0][0] == "pvfused"
    assert calls[0][1]["use_fp4_pv"] is True
    assert calls[0][1]["fp4_sf_dtype"] == "e4m3"
    assert calls[0][1]["pv_sf_dtype"] == "e4m3"


def test_fp4_use_fp4_pv_dispatch_supports_causal_gqa(monkeypatch):
    _install_fake_cuda_runtime(monkeypatch)
    calls = []

    def fake_pvfused_kernel(*_args, **kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr("flash_attn.cute.interface.FP4FlashAttentionForwardSm100PVFused", fake_pvfused_kernel)

    with FakeTensorMode():
        q, k, v, q_scale, k_scale, v_scale = _make_fake_fp4_pv_dense_inputs(
            fp4_qk_format="nvfp4",
            fp4_pv_format="nvfp4",
            head_dim=128,
            head_dim_v=128,
            num_heads=6,
            num_heads_kv=2,
            seqlen_q=512,
            seqlen_k=512,
        )
        with pytest.raises(NotImplementedError, match="exact Sage-style FP4 PV rewrite"):
            _flash_attn_fwd(
                q,
                k,
                v,
                causal=True,
                _arch=100,
                fp4_qk_format="nvfp4",
                q_scale=q_scale,
                k_scale=k_scale,
                use_fp4_pv=True,
                v_scale=v_scale,
            )

    assert not calls


@pytest.mark.parametrize(
    "kwargs,expected_error",
    [
        ({"q_scale": None}, ValueError),
        ({"k_scale": None}, ValueError),
        ({"q_scale_dtype": torch.float16}, TypeError),
        ({"k_scale_shape_delta": 1}, ValueError),
        ({"fp4_qk_format": "mxfp4"}, NotImplementedError),
        ({"head_dim": 96, "head_dim_v": 96}, NotImplementedError),
        ({"head_dim": 128, "head_dim_v": 64}, NotImplementedError),
        ({"v_dtype": torch.float16}, NotImplementedError),
        ({"num_splits": 2}, NotImplementedError),
        ({"window_size_left": 8}, NotImplementedError),
        ({"cu_seqlens_q": True}, NotImplementedError),
        ({"seqused_q": True}, NotImplementedError),
        ({"page_table": True}, NotImplementedError),
        ({"block_sparse": True}, NotImplementedError),
        ({"softcap": 0.5}, NotImplementedError),
        ({"score_mod": True}, NotImplementedError),
        ({"mask_mod": True}, NotImplementedError),
        ({"aux_tensors": True}, NotImplementedError),
        ({"learnable_sink": True}, NotImplementedError),
    ],
)
def test_fp4_qk_validation_errors(monkeypatch, kwargs, expected_error):
    _install_fake_cuda_runtime(monkeypatch)

    with FakeTensorMode():
        q, k, v, q_scale, k_scale = _make_fake_fp4_dense_inputs(
            fp4_qk_format=kwargs.get("fp4_qk_format", "nvfp4"),
            head_dim=kwargs.get("head_dim", 64),
            head_dim_v=kwargs.get("head_dim_v", 64),
        )
        if "v_dtype" in kwargs:
            v = v.to(kwargs["v_dtype"])
        if kwargs.get("q_scale") is None and "q_scale" in kwargs:
            q_scale = None
        if kwargs.get("k_scale") is None and "k_scale" in kwargs:
            k_scale = None
        if "q_scale_dtype" in kwargs:
            q_scale = q_scale.to(kwargs["q_scale_dtype"])
        if "k_scale_shape_delta" in kwargs:
            k_scale = torch.empty(
                *k_scale.shape[:-1],
                k_scale.shape[-1] + kwargs["k_scale_shape_delta"],
                device="cuda",
                dtype=k_scale.dtype,
            )

        cu_seqlens_q = (
            torch.empty(q.shape[0] + 1, device="cuda", dtype=torch.int32)
            if kwargs.get("cu_seqlens_q")
            else None
        )
        seqused_q = (
            torch.empty(q.shape[0], device="cuda", dtype=torch.int32)
            if kwargs.get("seqused_q")
            else None
        )
        page_table = (
            torch.empty(q.shape[0], 1, device="cuda", dtype=torch.int32)
            if kwargs.get("page_table")
            else None
        )
        score_mod = (lambda score, *_args: score) if kwargs.get("score_mod") else None
        mask_mod = (lambda *_args: True) if kwargs.get("mask_mod") else None
        aux_tensors = [torch.empty(1, device="cuda", dtype=torch.float32)] if kwargs.get("aux_tensors") else None
        learnable_sink = (
            torch.empty(q.shape[-2], device="cuda", dtype=torch.bfloat16)
            if kwargs.get("learnable_sink")
            else None
        )
        block_sparse_tensors = None
        if kwargs.get("block_sparse"):
            block_sparse_tensors = BlockSparseTensorsTorch(
                full_block_cnt=torch.empty(1, 1, 1, device="cuda", dtype=torch.int32),
                full_block_idx=torch.empty(1, 1, 1, 1, device="cuda", dtype=torch.int32),
                mask_block_cnt=torch.empty(1, 1, 1, device="cuda", dtype=torch.int32),
                mask_block_idx=torch.empty(1, 1, 1, 1, device="cuda", dtype=torch.int32),
                block_size=(128, 128),
            )

        with pytest.raises(expected_error):
            _flash_attn_fwd(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                seqused_q=seqused_q,
                page_table=page_table,
                causal=kwargs.get("causal", False),
                softcap=kwargs.get("softcap"),
                window_size_left=kwargs.get("window_size_left"),
                num_splits=kwargs.get("num_splits", 1),
                score_mod=score_mod,
                mask_mod=mask_mod,
                block_sparse_tensors=block_sparse_tensors,
                aux_tensors=aux_tensors,
                learnable_sink=learnable_sink,
                _arch=100,
                fp4_qk_format=kwargs.get("fp4_qk_format", "nvfp4"),
                q_scale=q_scale,
                k_scale=k_scale,
            )


@pytest.mark.parametrize(
    "kwargs,expected_error",
    [
        ({"use_fp4_pv": True, "v_scale": None}, ValueError),
        ({"use_fp4_pv": False, "provide_v_scale": True}, ValueError),
        ({"use_fp4_pv": True, "v_dtype": torch.bfloat16}, TypeError),
        ({"use_fp4_pv": True, "v_scale_dtype": torch.float16}, TypeError),
        ({"use_fp4_pv": True, "v_scale_shape_delta": 1}, ValueError),
        ({"use_fp4_pv": True, "fp4_qk_format": None}, ValueError),
        ({"use_fp4_pv": True, "fp4_qk_format": "mxfp4", "head_dim": 64, "head_dim_v": 64}, NotImplementedError),
        ({"use_fp4_pv": True, "head_dim": 128, "head_dim_v": 128, "fp4_pv_format": "mxfp4"}, TypeError),
        ({"use_fp4_pv": True, "head_dim": 128, "head_dim_v": 128, "causal": True, "fp4_pv_format": "mxfp4"}, TypeError),
        ({"use_fp4_pv": True, "head_dim": 128, "head_dim_v": 64}, NotImplementedError),
    ],
)
def test_fp4_pv_validation_errors(monkeypatch, kwargs, expected_error):
    _install_fake_cuda_runtime(monkeypatch)

    with FakeTensorMode():
        if kwargs.get("use_fp4_pv"):
            q, k, v, q_scale, k_scale, v_scale = _make_fake_fp4_pv_dense_inputs(
                fp4_qk_format=kwargs.get("fp4_qk_format", "nvfp4") or "nvfp4",
                fp4_pv_format=kwargs.get("fp4_pv_format"),
                head_dim=kwargs.get("head_dim", 64),
                head_dim_v=kwargs.get("head_dim_v", 64),
            )
        else:
            q, k, v, q_scale, k_scale = _make_fake_fp4_dense_inputs(
                fp4_qk_format=kwargs.get("fp4_qk_format", "nvfp4") or "nvfp4",
                head_dim=kwargs.get("head_dim", 64),
                head_dim_v=kwargs.get("head_dim_v", 64),
            )
            v_scale = None
        if "v_dtype" in kwargs:
            v = v.to(kwargs["v_dtype"])
        if kwargs.get("v_scale") is None and "v_scale" in kwargs:
            v_scale = None
        if kwargs.get("provide_v_scale"):
            v_scale = torch.empty(
                v.shape[0],
                v.shape[1],
                v.shape[-2] // 16,
                v.shape[-1] * 2,
                device="cuda",
                dtype=torch.float8_e4m3fn,
            )
        if "v_scale_dtype" in kwargs and v_scale is not None:
            v_scale = v_scale.to(kwargs["v_scale_dtype"])
        if "v_scale_shape_delta" in kwargs and v_scale is not None:
            v_scale = torch.empty(
                *v_scale.shape[:-1],
                v_scale.shape[-1] + kwargs["v_scale_shape_delta"],
                device="cuda",
                dtype=v_scale.dtype,
            )

        with pytest.raises(expected_error):
            _flash_attn_fwd(
                q,
                k,
                v,
                causal=kwargs.get("causal", False),
                _arch=100,
                fp4_qk_format=kwargs.get("fp4_qk_format", "nvfp4"),
                q_scale=q_scale,
                k_scale=k_scale,
                use_fp4_pv=kwargs.get("use_fp4_pv", False),
                v_scale=v_scale,
            )


def test_fp4_qk_rejects_non_sm100_arch(monkeypatch):
    _install_fake_cuda_runtime(monkeypatch)

    with FakeTensorMode():
        q, k, v, q_scale, k_scale = _make_fake_fp4_dense_inputs(
            fp4_qk_format="nvfp4",
            head_dim=64,
            head_dim_v=64,
        )
        with pytest.raises(NotImplementedError, match="SM100/SM110"):
            _flash_attn_fwd(
                q,
                k,
                v,
                _arch=90,
                fp4_qk_format="nvfp4",
                q_scale=q_scale,
                k_scale=k_scale,
            )


@pytest.mark.parametrize(
    "kwargs,expected_error",
    [
        ({"fp4_qk_format": "mxfp4"}, NotImplementedError),
        ({"num_heads": 4, "num_heads_kv": 2}, NotImplementedError),
        ({"missing_q_col_packed": True}, ValueError),
        ({"missing_k_col_packed": True}, ValueError),
        ({"missing_q_col_scale": True}, ValueError),
        ({"missing_k_col_scale": True}, ValueError),
        ({"q_sg_shape_delta": 1}, ValueError),
        ({"v_dtype": torch.float16}, NotImplementedError),
        ({"softcap": 0.5}, NotImplementedError),
        ({"block_sparse": True}, NotImplementedError),
    ],
)
def test_fp4_bwd_qk_validation_errors(monkeypatch, kwargs, expected_error):
    captured = {}

    def fake_bwd(*args, **bwd_kwargs):
        captured["called"] = True
        raise AssertionError("validation should fail before dispatch")

    monkeypatch.setattr("flash_attn.cute.interface._get_device_arch", lambda: 100)
    monkeypatch.setattr("flash_attn.cute.interface._flash_attn_bwd", fake_bwd)

    q_heads = kwargs.get("num_heads", 4)
    k_heads = kwargs.get("num_heads_kv", q_heads)
    q, k, v, out, dout, lse, q_scale, k_scale, q_col, k_col, q_col_scale, k_col_scale, q_sg, k_sg = _make_fake_fp4_bwd_inputs(
        fp4_qk_format=kwargs.get("fp4_qk_format", "nvfp4") if kwargs.get("fp4_qk_format") else "nvfp4",
        head_dim=64,
        num_heads=q_heads,
        num_heads_kv=k_heads,
        batch_size=2,
        seqlen_q=32,
        seqlen_k=32,
        device="cpu",
    )
    if k_heads != q_heads:
        k = torch.empty(k.shape[0], k.shape[1], k_heads, k.shape[-1], device="cpu", dtype=k.dtype)
        v = torch.empty(v.shape[0], v.shape[1], k_heads, v.shape[-1], device="cpu", dtype=v.dtype)
        k_scale = torch.empty(
            k_scale.shape[0], k_scale.shape[1], k_heads, k_scale.shape[-1], device="cpu", dtype=k_scale.dtype
        )
        k_col = torch.empty(k_col.shape[0], k_col.shape[1], k_heads, k_col.shape[-1], device="cpu", dtype=k_col.dtype)
        k_col_scale = torch.empty(
            k_col_scale.shape[0], k_col_scale.shape[1], k_heads, k_col_scale.shape[-1], device="cpu", dtype=k_col_scale.dtype
        )
        k_sg = torch.empty(k_sg.shape[0], k_heads, device="cpu", dtype=k_sg.dtype)
    if "v_dtype" in kwargs:
        v = v.to(kwargs["v_dtype"])
    if "q_sg_shape_delta" in kwargs:
        q_sg = torch.empty(
            q_sg.shape[0],
            q_sg.shape[1] + kwargs["q_sg_shape_delta"],
            device="cpu",
            dtype=q_sg.dtype,
        )
    block_sparse_tensors = None
    if kwargs.get("block_sparse"):
        block_sparse_tensors = BlockSparseTensorsTorch(
            full_block_cnt=torch.empty(1, 1, 1, device="cpu", dtype=torch.int32),
            full_block_idx=torch.empty(1, 1, 1, 1, device="cpu", dtype=torch.int32),
            mask_block_cnt=torch.empty(1, 1, 1, device="cpu", dtype=torch.int32),
            mask_block_idx=torch.empty(1, 1, 1, 1, device="cpu", dtype=torch.int32),
            block_size=(128, 128),
        )

    with pytest.raises(expected_error):
        _flash_attn_bwd_fp4_qk(
            q,
            k,
            v,
            out,
            dout,
            lse,
            q_scale,
            k_scale,
            q_sg,
            k_sg,
            softcap=kwargs.get("softcap", 0.0),
            fp4_qk_format=kwargs.get("fp4_qk_format", "nvfp4"),
            block_sparse_tensors=block_sparse_tensors,
            q_col_packed=None if kwargs.get("missing_q_col_packed") else q_col,
            k_col_packed=None if kwargs.get("missing_k_col_packed") else k_col,
            q_col_scale=None if kwargs.get("missing_q_col_scale") else q_col_scale,
            k_col_scale=None if kwargs.get("missing_k_col_scale") else k_col_scale,
        )
    assert "called" not in captured


def test_tk_nvfp4_transpose_scale_adapter_unswizzles_exactly():
    batch_size, head_dim, num_heads, seqlen_groups = 2, 64, 3, 8
    logical = torch.linspace(
        0.25,
        8.0,
        steps=batch_size * head_dim * num_heads * seqlen_groups,
        device="cpu",
        dtype=torch.float32,
    ).reshape(batch_size, head_dim, num_heads, seqlen_groups).to(torch.float8_e4m3fn)
    swizzled = _swizzle_tk_nvfp4_transpose_scale(logical)
    unswizzled = _unswizzle_tk_nvfp4_transpose_scale_bytes(swizzled)
    assert torch.equal(unswizzled.view(torch.uint8), logical.view(torch.uint8))

    adapted = _adapt_tk_nvfp4_transpose_scale(
        swizzled,
        torch.ones(batch_size, num_heads, dtype=torch.float32),
        logical,
    )
    assert torch.equal(adapted.view(torch.uint8), logical.view(torch.uint8))


def test_fp4_bwd_qk_dequantizes_rowwise_inputs_before_dispatch(monkeypatch):
    captured = {}

    def fake_bwd(q, k, v, out, dout, lse, **bwd_kwargs):
        captured["q"] = q
        captured["k"] = k
        captured["kwargs"] = bwd_kwargs
        return (
            torch.zeros_like(q),
            torch.zeros_like(k),
            torch.zeros_like(v),
        )

    monkeypatch.setattr("flash_attn.cute.interface._get_device_arch", lambda: 100)
    monkeypatch.setattr("flash_attn.cute.interface._flash_attn_bwd", fake_bwd)

    q, k, v, out, dout, lse, q_scale, k_scale, q_col, k_col, q_col_scale, k_col_scale, q_sg, k_sg = _make_fake_fp4_bwd_inputs(
        fp4_qk_format="nvfp4",
        head_dim=64,
        num_heads=2,
        batch_size=1,
        seqlen_q=4,
        seqlen_k=4,
        device="cpu",
    )
    q.fill_(0x22)
    k.fill_(0x44)
    q_scale.fill_(1.0)
    k_scale.fill_(1.0)
    q_col.fill_(0x22)
    k_col.fill_(0x44)
    q_col_scale.fill_(1.0)
    k_col_scale.fill_(1.0)
    q_sg.fill_(2.0)
    k_sg.fill_(0.5)
    v.zero_()
    out.zero_()
    dout.zero_()
    lse.zero_()

    dq, dk, dv = _flash_attn_bwd_fp4_qk(
        q,
        k,
        v,
        out,
        dout,
        lse,
        q_scale,
        k_scale,
        q_sg,
        k_sg,
        q_col_packed=q_col,
        k_col_packed=k_col,
        q_col_scale=q_col_scale,
        k_col_scale=k_col_scale,
        causal=True,
        deterministic=True,
    )

    assert dq.shape == (1, 4, 2, 64)
    assert dk.shape == (1, 4, 2, 64)
    assert dv.shape == (1, 4, 2, 64)
    torch.testing.assert_close(
        captured["q"].float(),
        torch.full((1, 4, 2, 64), 2.0),
    )
    torch.testing.assert_close(
        captured["k"].float(),
        torch.full((1, 4, 2, 64), 1.0),
    )
    assert captured["kwargs"]["causal"] is True
    assert captured["kwargs"]["deterministic"] is True
    torch.testing.assert_close(captured["kwargs"]["dq_scale_tensor"], torch.ones_like(k_sg))
    torch.testing.assert_close(captured["kwargs"]["dk_scale_tensor"], torch.ones_like(q_sg))
    assert "q_col_packed_fp4" not in captured["kwargs"]
    assert "k_col_packed_fp4" not in captured["kwargs"]


def test_fp4_bwd_qk_native_uses_adapted_tk_scales_by_default(monkeypatch):
    captured = {}

    def fake_bwd(q, k, v, out, dout, lse, **bwd_kwargs):
        captured["q"] = q
        captured["k"] = k
        captured["kwargs"] = bwd_kwargs
        return (
            torch.zeros_like(q),
            torch.zeros_like(k),
            torch.zeros_like(v),
        )

    monkeypatch.setattr("flash_attn.cute.interface._get_device_arch", lambda: 100)
    monkeypatch.setattr("flash_attn.cute.interface._flash_attn_bwd", fake_bwd)
    monkeypatch.setenv("FLASH_ATTN_FP4_BWD_ENABLE_NATIVE", "1")

    q, k, v, out, dout, lse, q_scale, k_scale, q_col, k_col, q_col_scale, k_col_scale, q_sg, k_sg = _make_fake_fp4_bwd_inputs(
        fp4_qk_format="nvfp4",
        head_dim=64,
        num_heads=2,
        batch_size=1,
        seqlen_q=64,
        seqlen_k=64,
        device="cpu",
    )
    q.fill_(0x22)
    k.fill_(0x44)
    q_scale.fill_(1.0)
    k_scale.fill_(1.0)
    q_col.fill_(0x22)
    k_col.fill_(0x44)
    q_sg.fill_(2.0)
    k_sg.fill_(0.5)
    v.zero_()
    out.zero_()
    dout.zero_()
    lse.zero_()
    logical_q_col_scale = torch.full_like(q_col_scale, 0.34375)
    logical_k_col_scale = torch.full_like(k_col_scale, 0.171875)
    q_col_scale.copy_(_swizzle_tk_nvfp4_transpose_scale(torch.full_like(q_col_scale, 0.171875)))
    k_col_scale.copy_(_swizzle_tk_nvfp4_transpose_scale(torch.full_like(k_col_scale, 0.34375)))

    dq, dk, dv = _flash_attn_bwd_fp4_qk(
        q,
        k,
        v,
        out,
        dout,
        lse,
        q_scale,
        k_scale,
        q_sg,
        k_sg,
        q_col_packed=q_col,
        k_col_packed=k_col,
        q_col_scale=q_col_scale,
        k_col_scale=k_col_scale,
    )

    assert dq.shape == (1, 64, 2, 64)
    assert dk.shape == (1, 64, 2, 64)
    assert dv.shape == (1, 64, 2, 64)
    assert captured["kwargs"]["q_col_packed_fp4"] is q_col
    assert captured["kwargs"]["k_col_packed_fp4"] is k_col
    torch.testing.assert_close(captured["kwargs"]["q_col_scale_fp4"].float(), logical_q_col_scale.float())
    torch.testing.assert_close(captured["kwargs"]["k_col_scale_fp4"].float(), logical_k_col_scale.float())
    torch.testing.assert_close(captured["kwargs"]["dq_scale_tensor"], torch.ones_like(k_sg))
    torch.testing.assert_close(captured["kwargs"]["dk_scale_tensor"], torch.ones_like(q_sg))


def test_fp4_bwd_qk_native_d128_uses_bridge_by_default(monkeypatch):
    captured = {}

    def fake_bwd(q, k, v, out, dout, lse, **bwd_kwargs):
        captured["kwargs"] = bwd_kwargs
        return (
            torch.zeros_like(q),
            torch.zeros_like(k),
            torch.zeros_like(v),
        )

    monkeypatch.setattr("flash_attn.cute.interface._get_device_arch", lambda: 100)
    monkeypatch.setattr("flash_attn.cute.interface._flash_attn_bwd", fake_bwd)
    monkeypatch.setenv("FLASH_ATTN_FP4_BWD_ENABLE_NATIVE", "1")

    q, k, v, out, dout, lse, q_scale, k_scale, q_col, k_col, q_col_scale, k_col_scale, q_sg, k_sg = _make_fake_fp4_bwd_inputs(
        fp4_qk_format="nvfp4",
        head_dim=128,
        num_heads=2,
        batch_size=1,
        seqlen_q=64,
        seqlen_k=64,
        device="cpu",
    )

    _flash_attn_bwd_fp4_qk(
        q,
        k,
        v,
        out,
        dout,
        lse,
        q_scale,
        k_scale,
        q_sg,
        k_sg,
        q_col_packed=q_col,
        k_col_packed=k_col,
        q_col_scale=q_col_scale,
        k_col_scale=k_col_scale,
    )

    assert "q_col_packed_fp4" not in captured["kwargs"]
    assert "k_col_packed_fp4" not in captured["kwargs"]
    torch.testing.assert_close(captured["kwargs"]["dq_scale_tensor"], torch.ones_like(k_sg))
    torch.testing.assert_close(captured["kwargs"]["dk_scale_tensor"], torch.ones_like(q_sg))


def test_fp4_bwd_qk_native_d128_can_force_unsafe_native(monkeypatch):
    captured = {}

    def fake_bwd(q, k, v, out, dout, lse, **bwd_kwargs):
        captured["kwargs"] = bwd_kwargs
        return (
            torch.zeros_like(q),
            torch.zeros_like(k),
            torch.zeros_like(v),
        )

    monkeypatch.setattr("flash_attn.cute.interface._get_device_arch", lambda: 100)
    monkeypatch.setattr("flash_attn.cute.interface._flash_attn_bwd", fake_bwd)
    monkeypatch.setenv("FLASH_ATTN_FP4_BWD_ENABLE_NATIVE", "1")
    monkeypatch.setenv("FLASH_ATTN_FP4_BWD_ALLOW_UNSAFE_D128", "1")

    q, k, v, out, dout, lse, q_scale, k_scale, q_col, k_col, q_col_scale, k_col_scale, q_sg, k_sg = _make_fake_fp4_bwd_inputs(
        fp4_qk_format="nvfp4",
        head_dim=128,
        num_heads=2,
        batch_size=1,
        seqlen_q=64,
        seqlen_k=64,
        device="cpu",
    )
    q.fill_(0x22)
    k.fill_(0x44)
    q_scale.fill_(1.0)
    k_scale.fill_(1.0)
    q_col.fill_(0x22)
    k_col.fill_(0x44)
    q_sg.fill_(2.0)
    k_sg.fill_(0.5)

    _flash_attn_bwd_fp4_qk(
        q,
        k,
        v,
        out,
        dout,
        lse,
        q_scale,
        k_scale,
        q_sg,
        k_sg,
        q_col_packed=q_col,
        k_col_packed=k_col,
        q_col_scale=q_col_scale,
        k_col_scale=k_col_scale,
    )

    assert captured["kwargs"]["q_col_packed_fp4"] is q_col
    assert captured["kwargs"]["k_col_packed_fp4"] is k_col


def test_fp4_bwd_qk_native_can_force_synthesized_tk_scales(monkeypatch):
    captured = {}

    def fake_bwd(q, k, v, out, dout, lse, **bwd_kwargs):
        captured["kwargs"] = bwd_kwargs
        return (
            torch.zeros_like(q),
            torch.zeros_like(k),
            torch.zeros_like(v),
        )

    monkeypatch.setattr("flash_attn.cute.interface._get_device_arch", lambda: 100)
    monkeypatch.setattr("flash_attn.cute.interface._flash_attn_bwd", fake_bwd)
    monkeypatch.setenv("FLASH_ATTN_FP4_BWD_ENABLE_NATIVE", "1")
    monkeypatch.setenv("FLASH_ATTN_FP4_BWD_FORCE_SYNTH_COL_SCALE", "1")

    q, k, v, out, dout, lse, q_scale, k_scale, q_col, k_col, q_col_scale, k_col_scale, q_sg, k_sg = _make_fake_fp4_bwd_inputs(
        fp4_qk_format="nvfp4",
        head_dim=64,
        num_heads=2,
        batch_size=1,
        seqlen_q=64,
        seqlen_k=64,
        device="cpu",
    )
    q.fill_(0x22)
    k.fill_(0x44)
    q_scale.fill_(1.0)
    k_scale.fill_(1.0)
    q_col.fill_(0x22)
    k_col.fill_(0x44)
    q_col_scale.fill_(7.0)
    k_col_scale.fill_(7.0)
    q_sg.fill_(2.0)
    k_sg.fill_(0.5)
    v.zero_()
    out.zero_()
    dout.zero_()
    lse.zero_()

    _flash_attn_bwd_fp4_qk(
        q,
        k,
        v,
        out,
        dout,
        lse,
        q_scale,
        k_scale,
        q_sg,
        k_sg,
        q_col_packed=q_col,
        k_col_packed=k_col,
        q_col_scale=q_col_scale,
        k_col_scale=k_col_scale,
    )

    torch.testing.assert_close(captured["kwargs"]["q_col_scale_fp4"].float(), torch.full_like(q_col_scale, 0.34375).float())
    torch.testing.assert_close(captured["kwargs"]["k_col_scale_fp4"].float(), torch.full_like(k_col_scale, 0.171875).float())
    torch.testing.assert_close(captured["kwargs"]["dq_scale_tensor"], torch.ones_like(k_sg))
    torch.testing.assert_close(captured["kwargs"]["dk_scale_tensor"], torch.ones_like(q_sg))


def test_fp4_bwd_qk_native_can_use_external_transpose_metadata(monkeypatch):
    captured = {}

    def fake_bwd(q, k, v, out, dout, lse, **bwd_kwargs):
        captured["kwargs"] = bwd_kwargs
        return (
            torch.zeros_like(q),
            torch.zeros_like(k),
            torch.zeros_like(v),
        )

    monkeypatch.setattr("flash_attn.cute.interface._get_device_arch", lambda: 100)
    monkeypatch.setattr("flash_attn.cute.interface._flash_attn_bwd", fake_bwd)
    monkeypatch.setenv("FLASH_ATTN_FP4_BWD_ENABLE_NATIVE", "1")
    monkeypatch.setenv("FLASH_ATTN_FP4_BWD_USE_EXTERNAL_COL_METADATA", "1")

    q, k, v, out, dout, lse, q_scale, k_scale, q_col, k_col, q_col_scale, k_col_scale, q_sg, k_sg = _make_fake_fp4_bwd_inputs(
        fp4_qk_format="nvfp4",
        head_dim=64,
        num_heads=2,
        batch_size=1,
        seqlen_q=4,
        seqlen_k=4,
        device="cpu",
    )
    q_col.fill_(0x12)
    k_col.fill_(0x34)
    q_col_scale.fill_(3.0)
    k_col_scale.fill_(4.0)
    q_sg.fill_(2.0)
    k_sg.fill_(0.5)

    _flash_attn_bwd_fp4_qk(
        q,
        k,
        v,
        out,
        dout,
        lse,
        q_scale,
        k_scale,
        q_sg,
        k_sg,
        q_col_packed=q_col,
        k_col_packed=k_col,
        q_col_scale=q_col_scale,
        k_col_scale=k_col_scale,
    )

    assert torch.equal(captured["kwargs"]["q_col_packed_fp4"], q_col)
    assert torch.equal(captured["kwargs"]["k_col_packed_fp4"], k_col)
    torch.testing.assert_close(captured["kwargs"]["q_col_scale_fp4"], q_col_scale)
    torch.testing.assert_close(captured["kwargs"]["k_col_scale_fp4"], k_col_scale)
    torch.testing.assert_close(captured["kwargs"]["dq_scale_tensor"], k_sg)
    torch.testing.assert_close(captured["kwargs"]["dk_scale_tensor"], q_sg)


def _require_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for runtime FP4 tests.")
    major, minor = torch.cuda.get_device_capability()
    if major not in (10, 11):
        pytest.skip(f"FP4 runtime tests require SM100/SM110, got {major}.{minor}.")


def _require_tk_quant_v5():
    tk_dir = "/workspace/codebases/fp4_matmul/TK_quantisation/nvfp4_v5"
    if not os.path.isdir(tk_dir):
        pytest.skip("TK NVFP4 quantization directory is unavailable.")
    if not any(name.startswith("_tk_quant_v5") and name.endswith(".so") for name in os.listdir(tk_dir)):
        pytest.skip("TK NVFP4 quantization extension is unavailable.")


def _run_fp4_bwd_native_runtime_case(*, timeout_s: int = 180) -> None:
    script = textwrap.dedent(
        """
        import math
        import os
        import sys
        import torch

        sys.path.insert(0, "/workspace/codebases/fp4_matmul/flash-attention")
        sys.path.insert(0, "/workspace/codebases/fp4_matmul/TK_quantisation/nvfp4_v5")

        import _tk_quant_v5 as tk
        from flash_attn.cute.interface import _flash_attn_bwd_fp4_qk

        torch.manual_seed(0)
        device = "cuda"
        batch_size, seqlen, num_heads, head_dim = 1, 128, 4, 64
        softmax_scale = 1.0 / math.sqrt(head_dim)

        q_bf16 = (torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
        k_bf16 = (torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
        v = (torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
        out = (torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
        dout = (torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
        lse = torch.randn(batch_size, num_heads, seqlen, device=device, dtype=torch.float32).contiguous()

        q_packed = torch.empty(batch_size, seqlen, num_heads, head_dim // 2, device=device, dtype=torch.uint8)
        k_packed = torch.empty_like(q_packed)
        q_scale = torch.empty(batch_size, seqlen, num_heads, head_dim // 16, device=device, dtype=torch.float8_e4m3fn)
        k_scale = torch.empty_like(q_scale)
        q_col_packed = torch.empty(batch_size, head_dim, num_heads, seqlen // 2, device=device, dtype=torch.uint8)
        k_col_packed = torch.empty_like(q_col_packed)
        q_col_scale = torch.empty(batch_size, head_dim, num_heads, seqlen // 16, device=device, dtype=torch.float8_e4m3fn)
        k_col_scale = torch.empty_like(q_col_scale)
        q_sg = torch.empty(batch_size, num_heads, device=device, dtype=torch.float32)
        k_sg = torch.empty_like(q_sg)

        for b in range(batch_size):
            for h in range(num_heads):
                qh = q_bf16[b, :, h, :].contiguous()
                kh = k_bf16[b, :, h, :].contiguous()
                qamax = qh.abs().max().reshape(1).float()
                kamax = kh.abs().max().reshape(1).float()
                qr, qsu8, qc, qcsu8 = tk.tk_quantize_transpose(qh, qamax, qamax, True)
                kr, ksu8, kc, kcsu8 = tk.tk_quantize_transpose(kh, kamax, kamax, True)
                q_packed[b, :, h, :] = qr
                k_packed[b, :, h, :] = kr
                q_scale[b, :, h, :] = qsu8.view(torch.float8_e4m3fn)
                k_scale[b, :, h, :] = ksu8.view(torch.float8_e4m3fn)
                q_col_packed[b, :, h, :] = qc
                k_col_packed[b, :, h, :] = kc
                q_col_scale[b, :, h, :] = qcsu8.view(torch.float8_e4m3fn)
                k_col_scale[b, :, h, :] = kcsu8.view(torch.float8_e4m3fn)
                q_sg[b, h] = qamax.item() / 2688.0
                k_sg[b, h] = kamax.item() / 2688.0

        for key in list(os.environ):
            if key.startswith("FLASH_ATTN_FP4_BWD_"):
                os.environ.pop(key, None)
        os.environ["FLASH_ATTN_FP4_BWD_FORCE_REFERENCE"] = "1"
        dq_ref, dk_ref, dv_ref = _flash_attn_bwd_fp4_qk(
            q_packed,
            k_packed,
            v,
            out,
            dout,
            lse,
            q_scale,
            k_scale,
            q_sg,
            k_sg,
            softmax_scale=softmax_scale,
            q_col_packed=q_col_packed,
            k_col_packed=k_col_packed,
            q_col_scale=q_col_scale,
            k_col_scale=k_col_scale,
        )
        ref_cpu = {
            "dq": dq_ref.float().cpu().clone(),
            "dk": dk_ref.float().cpu().clone(),
            "dv": dv_ref.float().cpu().clone(),
        }
        del os.environ["FLASH_ATTN_FP4_BWD_FORCE_REFERENCE"]
        os.environ["FLASH_ATTN_FP4_BWD_ENABLE_NATIVE"] = "1"
        dq, dk, dv = _flash_attn_bwd_fp4_qk(
            q_packed,
            k_packed,
            v,
            out,
            dout,
            lse,
            q_scale,
            k_scale,
            q_sg,
            k_sg,
            softmax_scale=softmax_scale,
            q_col_packed=q_col_packed,
            k_col_packed=k_col_packed,
            q_col_scale=q_col_scale,
            k_col_scale=k_col_scale,
        )
        got_cpu = {
            "dq": dq.float().cpu(),
            "dk": dk.float().cpu(),
            "dv": dv.float().cpu(),
        }
        for name in ("dq", "dk", "dv"):
            if not torch.isfinite(ref_cpu[name]).all().item():
                raise AssertionError(f"Reference {name} contains NaN or Inf.")
            if not torch.isfinite(got_cpu[name]).all().item():
                raise AssertionError(f"Native {name} contains NaN or Inf.")

        dq_mean_abs = (got_cpu["dq"] - ref_cpu["dq"]).abs().mean().item()
        dk_mean_abs = (got_cpu["dk"] - ref_cpu["dk"]).abs().mean().item()
        dv_max_abs = (got_cpu["dv"] - ref_cpu["dv"]).abs().max().item()
        if dq_mean_abs > 0.1:
            raise AssertionError(f"dQ mean_abs {dq_mean_abs} exceeded 0.1")
        if dk_mean_abs > 0.1:
            raise AssertionError(f"dK mean_abs {dk_mean_abs} exceeded 0.1")
        if dv_max_abs != 0.0:
            raise AssertionError(f"dV max_abs {dv_max_abs} should be exactly 0.0")
        print("runtime-ok")
        """
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd="/workspace/codebases/fp4_matmul/flash-attention",
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=os.environ.copy(),
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"FP4 native backward runtime smoke test timed out after {timeout_s}s.")
        if result.returncode != 0:
            pytest.fail(
                "FP4 native backward runtime smoke test failed.\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )


def _run_fp4_bwd_native_tk_scale_adapter_runtime_case(*, timeout_s: int = 180) -> None:
    script = textwrap.dedent(
        """
        import math
        import os
        import sys
        import torch

        sys.path.insert(0, "/workspace/codebases/fp4_matmul/flash-attention")
        sys.path.insert(0, "/workspace/codebases/fp4_matmul/TK_quantisation/nvfp4_v5")

        import _tk_quant_v5 as tk
        from flash_attn.cute.interface import _flash_attn_bwd_fp4_qk

        torch.manual_seed(0)
        device = "cuda"
        batch_size, seqlen, num_heads, head_dim = 1, 128, 4, 64
        softmax_scale = 1.0 / math.sqrt(head_dim)

        q_bf16 = (torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
        k_bf16 = (torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
        v = (torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
        out = (torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
        dout = (torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
        lse = torch.randn(batch_size, num_heads, seqlen, device=device, dtype=torch.float32).contiguous()

        q_packed = torch.empty(batch_size, seqlen, num_heads, head_dim // 2, device=device, dtype=torch.uint8)
        k_packed = torch.empty_like(q_packed)
        q_scale = torch.empty(batch_size, seqlen, num_heads, head_dim // 16, device=device, dtype=torch.float8_e4m3fn)
        k_scale = torch.empty_like(q_scale)
        q_col_packed = torch.empty(batch_size, head_dim, num_heads, seqlen // 2, device=device, dtype=torch.uint8)
        k_col_packed = torch.empty_like(q_col_packed)
        q_col_scale = torch.empty(batch_size, head_dim, num_heads, seqlen // 16, device=device, dtype=torch.float8_e4m3fn)
        k_col_scale = torch.empty_like(q_col_scale)
        q_sg = torch.empty(batch_size, num_heads, device=device, dtype=torch.float32)
        k_sg = torch.empty_like(q_sg)

        for b in range(batch_size):
            for h in range(num_heads):
                qh = q_bf16[b, :, h, :].contiguous()
                kh = k_bf16[b, :, h, :].contiguous()
                qamax = qh.abs().max().reshape(1).float()
                kamax = kh.abs().max().reshape(1).float()
                qr, qsu8, qc, qcsu8 = tk.tk_quantize_transpose(qh, qamax, qamax, True)
                kr, ksu8, kc, kcsu8 = tk.tk_quantize_transpose(kh, kamax, kamax, True)
                q_packed[b, :, h, :] = qr
                k_packed[b, :, h, :] = kr
                q_scale[b, :, h, :] = qsu8.view(torch.float8_e4m3fn)
                k_scale[b, :, h, :] = ksu8.view(torch.float8_e4m3fn)
                q_col_packed[b, :, h, :] = qc
                k_col_packed[b, :, h, :] = kc
                q_col_scale[b, :, h, :] = qcsu8.view(torch.float8_e4m3fn)
                k_col_scale[b, :, h, :] = kcsu8.view(torch.float8_e4m3fn)
                q_sg[b, h] = qamax.item() / 2688.0
                k_sg[b, h] = kamax.item() / 2688.0

        for key in list(os.environ):
            if key.startswith("FLASH_ATTN_FP4_BWD_"):
                os.environ.pop(key, None)
        os.environ["FLASH_ATTN_FP4_BWD_ENABLE_NATIVE"] = "1"
        dq_adapt, dk_adapt, dv_adapt = _flash_attn_bwd_fp4_qk(
            q_packed,
            k_packed,
            v,
            out,
            dout,
            lse,
            q_scale,
            k_scale,
            q_sg,
            k_sg,
            softmax_scale=softmax_scale,
            q_col_packed=q_col_packed,
            k_col_packed=k_col_packed,
            q_col_scale=q_col_scale,
            k_col_scale=k_col_scale,
        )
        os.environ["FLASH_ATTN_FP4_BWD_FORCE_SYNTH_COL_SCALE"] = "1"
        dq_synth, dk_synth, dv_synth = _flash_attn_bwd_fp4_qk(
            q_packed,
            k_packed,
            v,
            out,
            dout,
            lse,
            q_scale,
            k_scale,
            q_sg,
            k_sg,
            softmax_scale=softmax_scale,
            q_col_packed=q_col_packed,
            k_col_packed=k_col_packed,
            q_col_scale=q_col_scale,
            k_col_scale=k_col_scale,
        )

        adapt_cpu = {
            "dq": dq_adapt.float().cpu(),
            "dk": dk_adapt.float().cpu(),
            "dv": dv_adapt.float().cpu(),
        }
        synth_cpu = {
            "dq": dq_synth.float().cpu(),
            "dk": dk_synth.float().cpu(),
            "dv": dv_synth.float().cpu(),
        }

        for name in ("dq", "dk", "dv"):
            if not torch.isfinite(adapt_cpu[name]).all().item():
                raise AssertionError(f"Adapted {name} contains NaN or Inf.")
            if not torch.isfinite(synth_cpu[name]).all().item():
                raise AssertionError(f"Synthesized fallback {name} contains NaN or Inf.")

        dq_mean_abs = (adapt_cpu["dq"] - synth_cpu["dq"]).abs().mean().item()
        dk_mean_abs = (adapt_cpu["dk"] - synth_cpu["dk"]).abs().mean().item()
        dv_max_abs = (adapt_cpu["dv"] - synth_cpu["dv"]).abs().max().item()
        if dq_mean_abs > 0.02:
            raise AssertionError(f"dQ mean_abs {dq_mean_abs} exceeded 0.02")
        if dk_mean_abs > 0.06:
            raise AssertionError(f"dK mean_abs {dk_mean_abs} exceeded 0.06")
        if dv_max_abs != 0.0:
            raise AssertionError(f"dV max_abs {dv_max_abs} should be exactly 0.0")
        print("runtime-ok")
        """
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd="/workspace/codebases/fp4_matmul/flash-attention",
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=os.environ.copy(),
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"FP4 native TK-scale adapter backward smoke test timed out after {timeout_s}s.")
    if result.returncode != 0:
        pytest.fail(
            "FP4 native TK-scale adapter backward smoke test failed.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def _run_fp4_runtime_case(
    num_heads: int,
    num_heads_kv: int,
    *,
    head_dim: int,
    causal: bool,
    seqlen_q: int = 128,
    seqlen_k: int = 128,
    pack_gqa: bool = False,
    use_fp4_pv: bool = False,
    timeout_s: int = 180,
    extra_env: dict[str, str] | None = None,
) -> None:
    script = textwrap.dedent(
        f"""
        import torch
        from flash_attn.cute import fa_logging
        from flash_attn.cute.interface import _flash_attn_fwd

        fa_logging.set_fa_log_level(2)

        FP4_GRID = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=torch.float32,
            device="cuda",
        )

        def _unpack_fp4(packed):
            packed_i32 = packed.to(torch.int32)
            return torch.stack((packed_i32 & 0xF, (packed_i32 >> 4) & 0xF), dim=-1).flatten(start_dim=-2)

        def _dequantize_fp4(packed, scale, sf_vec_size):
            values = FP4_GRID[_unpack_fp4(packed).long()]
            return (
                values.unflatten(dim=-1, sizes=(-1, sf_vec_size))
                * scale.to(torch.float32).unsqueeze(-1)
            ).flatten(start_dim=-2)

        def _unswizzle_fp4_vt_scale(scale_vt):
            batch_size, num_heads, head_dim, seqlen_groups = scale_vt.shape
            out = torch.empty_like(scale_vt)
            flat_in = scale_vt.view(batch_size, num_heads, -1)
            logical = out.reshape(batch_size, num_heads, head_dim, seqlen_groups)
            for d in range(head_dim):
                tile_m, row_in_tile = divmod(d, 64)
                quad, row_mod16 = divmod(row_in_tile, 16)
                for seq_group in range(seqlen_groups):
                    offset = (
                        tile_m * 64 * seqlen_groups
                        + (seq_group // 4) * 256
                        + (seq_group % 4)
                        + quad * 4
                        + row_mod16 * 16
                    )
                    logical[:, :, d, seq_group] = flat_in[:, :, offset]
            return out

        def _dequantize_fp4_vt(packed_vt, scale_vt, sf_vec_size):
            values = FP4_GRID[_unpack_fp4(packed_vt).long()]
            logical_scale_vt = _unswizzle_fp4_vt_scale(scale_vt)
            values = values.unflatten(dim=-1, sizes=(-1, sf_vec_size))
            values = values * logical_scale_vt.to(torch.float32).unsqueeze(-1)
            return values.flatten(start_dim=3, end_dim=4).permute(0, 3, 1, 2).contiguous()

        torch.manual_seed(0)
        batch_size = 2
        seqlen_q = {seqlen_q}
        seqlen_k = {seqlen_k}
        seqlen_k_padded = ((seqlen_k + 127) // 128) * 128
        head_dim = {head_dim}
        head_dim_v = {head_dim}
        # Keep the synthetic FP4 test inputs in a realistic pre-quantized range so the
        # causal cases remain comparable to the dequantized BF16 reference at the chosen tolerance.
        scale_value = 0.125

        q_packed = torch.randint(
            0,
            256,
            (batch_size, seqlen_q, {num_heads}, head_dim // 2),
            device="cuda",
            dtype=torch.uint8,
        )
        k_packed = torch.randint(
            0,
            256,
            (batch_size, seqlen_k, {num_heads_kv}, head_dim // 2),
            device="cuda",
            dtype=torch.uint8,
        )
        v_packed = torch.randint(
            0,
            256,
            (batch_size, {num_heads_kv}, head_dim_v, seqlen_k_padded // 2),
            device="cuda",
            dtype=torch.uint8,
        )
        v_bf16 = torch.randn(
            batch_size,
            seqlen_k,
            {num_heads_kv},
            head_dim_v,
            device="cuda",
            dtype=torch.bfloat16,
        ) * 0.25
        q_scale = torch.full(
            (batch_size, seqlen_q, {num_heads}, head_dim // 16),
            scale_value,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        k_scale = torch.full(
            (batch_size, seqlen_k, {num_heads_kv}, head_dim // 16),
            scale_value,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        v_scale = torch.full(
            (batch_size, {num_heads_kv}, head_dim_v, seqlen_k_padded // 16),
            scale_value,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        v_scale = _swizzle_fp4_vt_scale(v_scale)

        q_ref = _dequantize_fp4(q_packed, q_scale, 16).to(torch.bfloat16)
        k_ref = _dequantize_fp4(k_packed, k_scale, 16).to(torch.bfloat16)
        v_ref = _dequantize_fp4_vt(v_packed, v_scale, 16)[:, :seqlen_k].to(torch.bfloat16) if {use_fp4_pv} else v_bf16

        v_runtime = v_packed if {use_fp4_pv} else v_bf16

        out_fp4, lse_fp4 = _flash_attn_fwd(
            q_packed,
            k_packed,
            v_runtime,
            causal={causal},
            return_lse=True,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
            use_fp4_pv={use_fp4_pv},
            v_scale=v_scale if {use_fp4_pv} else None,
            pack_gqa={pack_gqa},
        )
        out_ref, lse_ref = _flash_attn_fwd(
            q_ref,
            k_ref,
            v_ref,
            causal={causal},
            return_lse=True,
        )
        torch.cuda.synchronize()

        if not torch.isfinite(out_fp4.float()).all().item():
            raise AssertionError("FP4 output contains NaN or Inf.")
        if not torch.isfinite(lse_fp4.float()).all().item():
            raise AssertionError("FP4 LSE contains NaN or Inf.")

        torch.testing.assert_close(out_fp4.float(), out_ref.float(), atol=2e-1, rtol=5e-2)
        torch.testing.assert_close(lse_fp4.float(), lse_ref.float(), atol=2e-1, rtol=5e-2)
        print("runtime-ok")
        """
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd="/workspace/codebases/fp4_matmul/flash-attention",
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env={**os.environ, **(extra_env or {})},
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            "FP4 runtime case timed out after "
            f"{timeout_s}s for ({num_heads}q,{num_heads_kv}kv,d={head_dim},s_q={seqlen_q},s_k={seqlen_k},causal={causal},use_fp4_pv={use_fp4_pv})."
        )
    if result.returncode != 0:
        pytest.fail(
            "FP4 runtime subprocess failed for "
            f"({num_heads}q,{num_heads_kv}kv,d={head_dim},s_q={seqlen_q},s_k={seqlen_k},causal={causal},use_fp4_pv={use_fp4_pv}).\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def _run_fp4_pv_probe(script: str, *, direct_loader: bool, timeout_s: int = 180) -> None:
    extra_env = {"FLASH_ATTN_FP4_PV_DIRECT_LOADER": "1"} if direct_loader else {}
    try:
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd="/workspace/codebases/fp4_matmul/flash-attention",
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env={**os.environ, **extra_env},
            check=False,
        )
    except subprocess.TimeoutExpired:
        loader_mode = "direct" if direct_loader else "legacy"
        pytest.fail(f"FP4 PV probe timed out after {timeout_s}s for {loader_mode} loader.")
    if result.returncode != 0:
        loader_mode = "direct" if direct_loader else "legacy"
        pytest.fail(
            f"FP4 PV probe failed for {loader_mode} loader.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def _run_fp4_pv_cta_quant_compare_case(
    *,
    head_dim: int = 64,
    causal: bool = False,
    direct_loader: bool = False,
    seeds: tuple[int, ...] = (0,),
    atol: float = 2e-1,
    rtol: float = 5e-2,
    timeout_s: int = 180,
) -> None:
    extra_env = {"FLASH_ATTN_FP4_PV_DIRECT_LOADER": "1"} if direct_loader else {}
    seeds_literal = ", ".join(str(seed) for seed in seeds)
    script = textwrap.dedent(
        f"""
        import os
        import torch
        from flash_attn.cute.interface import _flash_attn_fwd

        FP4_GRID = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=torch.float32,
            device="cuda",
        )

        def _swizzle_fp4_vt_scale(scale_vt):
            batch_size, num_heads, head_dim, seqlen_groups = scale_vt.shape
            out = torch.empty_like(scale_vt)
            flat_in = scale_vt.reshape(batch_size, num_heads, head_dim, seqlen_groups)
            flat_out = out.view(batch_size, num_heads, -1)
            for d in range(head_dim):
                tile_m, row_in_tile = divmod(d, 64)
                quad, row_mod16 = divmod(row_in_tile, 16)
                for seq_group in range(seqlen_groups):
                    offset = (
                        tile_m * 64 * seqlen_groups
                        + (seq_group // 4) * 256
                        + (seq_group % 4)
                        + quad * 4
                        + row_mod16 * 16
                    )
                    flat_out[:, :, offset] = flat_in[:, :, d, seq_group]
            return out

        for seed in ({seeds_literal},):
            torch.manual_seed(seed)
            batch_size, seqlen_q, seqlen_k, num_heads = 1, 128, 128, 4
            seqlen_k_padded = ((seqlen_k + 127) // 128) * 128
            head_dim = {head_dim}
            scale_value = 0.125

            q_packed = torch.randint(0, 256, (batch_size, seqlen_q, num_heads, head_dim // 2), device="cuda", dtype=torch.uint8)
            k_packed = torch.randint(0, 256, (batch_size, seqlen_k, num_heads, head_dim // 2), device="cuda", dtype=torch.uint8)
            v_packed = torch.randint(0, 256, (batch_size, num_heads, head_dim, seqlen_k_padded // 2), device="cuda", dtype=torch.uint8)
            q_scale = torch.full((batch_size, seqlen_q, num_heads, head_dim // 16), scale_value, device="cuda", dtype=torch.float8_e4m3fn)
            k_scale = torch.full((batch_size, seqlen_k, num_heads, head_dim // 16), scale_value, device="cuda", dtype=torch.float8_e4m3fn)
            v_scale = torch.full((batch_size, num_heads, head_dim, seqlen_k_padded // 16), scale_value, device="cuda", dtype=torch.float8_e4m3fn)
            v_scale = _swizzle_fp4_vt_scale(v_scale)

            os.environ.pop("FLASH_ATTN_FP4_PV_ENABLE_CTA_ENCODE", None)
            out_legacy, lse_legacy = _flash_attn_fwd(
                q_packed,
                k_packed,
                v_packed,
                causal={causal},
                return_lse=True,
                fp4_qk_format="nvfp4",
                q_scale=q_scale,
                k_scale=k_scale,
                use_fp4_pv=True,
                v_scale=v_scale,
            )
            os.environ["FLASH_ATTN_FP4_PV_ENABLE_CTA_ENCODE"] = "1"
            out_cta, lse_cta = _flash_attn_fwd(
                q_packed,
                k_packed,
                v_packed,
                causal={causal},
                return_lse=True,
                fp4_qk_format="nvfp4",
                q_scale=q_scale,
                k_scale=k_scale,
                use_fp4_pv=True,
                v_scale=v_scale,
            )
            torch.cuda.synchronize()

            if not torch.isfinite(out_cta.float()).all().item():
                raise AssertionError(f"seed={{seed}}: CTA FP4 PV output contains NaN or Inf.")
            if not torch.isfinite(lse_cta.float()).all().item():
                raise AssertionError(f"seed={{seed}}: CTA FP4 PV LSE contains NaN or Inf.")

            try:
                torch.testing.assert_close(out_cta.float(), out_legacy.float(), atol={atol}, rtol={rtol})
                torch.testing.assert_close(lse_cta.float(), lse_legacy.float(), atol={atol}, rtol={rtol})
            except AssertionError as exc:
                raise AssertionError(f"seed={{seed}}: {{exc}}") from exc
        print("cta-legacy-ok")
        """
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd="/workspace/codebases/fp4_matmul/flash-attention",
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env={**os.environ, **extra_env},
            check=False,
        )
    except subprocess.TimeoutExpired:
        loader_mode = "direct" if direct_loader else "legacy"
        pytest.fail(f"FP4 PV CTA quant compare timed out after {timeout_s}s for {loader_mode} loader.")
    if result.returncode != 0:
        loader_mode = "direct" if direct_loader else "legacy"
        pytest.fail(
            f"FP4 PV CTA quant compare failed for {loader_mode} loader.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def _run_fp4_pv_causal_oracle_case(
    *,
    head_dim: int,
    direct_loader: bool = False,
    timeout_s: int = 180,
    expect_close: bool = True,
    atol: float = 2e-1,
    rtol: float = 5e-2,
) -> None:
    extra_env = {
        "FLASH_ATTN_FP4_PV_ENABLE_CTA_ENCODE": "1",
        **({"FLASH_ATTN_FP4_PV_DIRECT_LOADER": "1"} if direct_loader else {}),
    }
    script = textwrap.dedent(
        f"""
        import torch
        from flash_attn.cute.interface import _flash_attn_fwd

        FP4_GRID = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=torch.float32,
            device="cuda",
        )

        def _unpack_fp4(packed):
            packed_i32 = packed.to(torch.int32)
            return torch.stack((packed_i32 & 0xF, (packed_i32 >> 4) & 0xF), dim=-1).flatten(start_dim=-2)

        def _dequantize_fp4(packed, scale, sf_vec_size):
            values = FP4_GRID[_unpack_fp4(packed).long()]
            return (
                values.unflatten(dim=-1, sizes=(-1, sf_vec_size))
                * scale.to(torch.float32).unsqueeze(-1)
            ).flatten(start_dim=-2)

        def _swizzle_fp4_vt_scale(scale_vt):
            batch_size, num_heads, head_dim, seqlen_groups = scale_vt.shape
            out = torch.empty_like(scale_vt)
            flat_in = scale_vt.reshape(batch_size, num_heads, head_dim, seqlen_groups)
            flat_out = out.view(batch_size, num_heads, -1)
            for d in range(head_dim):
                tile_m, row_in_tile = divmod(d, 64)
                quad, row_mod16 = divmod(row_in_tile, 16)
                for seq_group in range(seqlen_groups):
                    offset = (
                        tile_m * 64 * seqlen_groups
                        + (seq_group // 4) * 256
                        + (seq_group % 4)
                        + quad * 4
                        + row_mod16 * 16
                    )
                    flat_out[:, :, offset] = flat_in[:, :, d, seq_group]
            return out

        def _unswizzle_fp4_vt_scale(scale_vt):
            batch_size, num_heads, head_dim, seqlen_groups = scale_vt.shape
            out = torch.empty_like(scale_vt)
            flat_in = scale_vt.view(batch_size, num_heads, -1)
            logical = out.reshape(batch_size, num_heads, head_dim, seqlen_groups)
            for d in range(head_dim):
                tile_m, row_in_tile = divmod(d, 64)
                quad, row_mod16 = divmod(row_in_tile, 16)
                for seq_group in range(seqlen_groups):
                    offset = (
                        tile_m * 64 * seqlen_groups
                        + (seq_group // 4) * 256
                        + (seq_group % 4)
                        + quad * 4
                        + row_mod16 * 16
                    )
                    logical[:, :, d, seq_group] = flat_in[:, :, offset]
            return out

        def _dequantize_fp4_vt(packed_vt, scale_vt, sf_vec_size):
            values = FP4_GRID[_unpack_fp4(packed_vt).long()]
            logical_scale_vt = _unswizzle_fp4_vt_scale(scale_vt)
            values = values.unflatten(dim=-1, sizes=(-1, sf_vec_size))
            values = values * logical_scale_vt.to(torch.float32).unsqueeze(-1)
            return values.flatten(start_dim=3, end_dim=4).permute(0, 3, 1, 2).contiguous()

        torch.manual_seed(0)
        batch_size, seqlen_q, seqlen_k, num_heads = 1, 128, 128, 4
        head_dim = {head_dim}
        seqlen_k_padded = 128
        scale_value = 0.125

        q_packed = torch.zeros((batch_size, seqlen_q, num_heads, head_dim // 2), device="cuda", dtype=torch.uint8)
        k_packed = torch.zeros((batch_size, seqlen_k, num_heads, head_dim // 2), device="cuda", dtype=torch.uint8)
        v_packed = torch.randint(
            0, 256, (batch_size, num_heads, head_dim, seqlen_k_padded // 2), device="cuda", dtype=torch.uint8
        )
        q_scale = torch.ones((batch_size, seqlen_q, num_heads, head_dim // 16), device="cuda", dtype=torch.float8_e4m3fn)
        k_scale = torch.ones((batch_size, seqlen_k, num_heads, head_dim // 16), device="cuda", dtype=torch.float8_e4m3fn)
        v_scale = torch.full(
            (batch_size, num_heads, head_dim, seqlen_k_padded // 16),
            scale_value,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        v_scale = _swizzle_fp4_vt_scale(v_scale)

        q_ref = _dequantize_fp4(q_packed, q_scale, 16).to(torch.bfloat16)
        k_ref = _dequantize_fp4(k_packed, k_scale, 16).to(torch.bfloat16)
        v_ref = _dequantize_fp4_vt(v_packed, v_scale, 16)[:, :seqlen_k].to(torch.bfloat16)

        out_fp4, lse_fp4 = _flash_attn_fwd(
            q_packed,
            k_packed,
            v_packed,
            causal=True,
            return_lse=True,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
            use_fp4_pv=True,
            v_scale=v_scale,
        )
        out_ref, lse_ref = _flash_attn_fwd(
            q_ref,
            k_ref,
            v_ref,
            causal=True,
            return_lse=True,
        )
        torch.cuda.synchronize()

        if not torch.isfinite(out_fp4.float()).all().item():
            raise AssertionError("FP4 PV causal oracle output contains NaN or Inf.")
        if not torch.isfinite(lse_fp4.float()).all().item():
            raise AssertionError("FP4 PV causal oracle LSE contains NaN or Inf.")

        if {expect_close}:
            torch.testing.assert_close(out_fp4.float(), out_ref.float(), atol={atol}, rtol={rtol})
            torch.testing.assert_close(lse_fp4.float(), lse_ref.float(), atol={atol}, rtol={rtol})
        else:
            diff_out = (out_fp4.float() - out_ref.float()).abs().max().item()
            diff_lse = (lse_fp4.float() - lse_ref.float()).abs().max().item()
            print(f"informational_diff_out={{diff_out}} informational_diff_lse={{diff_lse}}")
        print("pv-causal-oracle-ok")
        """
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd="/workspace/codebases/fp4_matmul/flash-attention",
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env={**os.environ, **extra_env},
            check=False,
        )
    except subprocess.TimeoutExpired:
        loader_mode = "direct" if direct_loader else "legacy"
        pytest.fail(f"FP4 PV causal oracle timed out after {timeout_s}s for {loader_mode} loader.")
    if result.returncode != 0:
        loader_mode = "direct" if direct_loader else "legacy"
        pytest.fail(
            f"FP4 PV causal oracle failed for {loader_mode} loader.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def test_fp4_bwd_qk_native_runtime_smoke():
    _require_sm100()
    _require_tk_quant_v5()
    _run_fp4_bwd_native_runtime_case()


def test_fp4_bwd_qk_native_tk_scale_adapter_runtime_smoke():
    _require_sm100()
    _require_tk_quant_v5()
    _run_fp4_bwd_native_tk_scale_adapter_runtime_case()


@pytest.mark.parametrize(
    "num_heads,num_heads_kv,head_dim,causal",
    [
        (4, 4, 64, False),
        (6, 2, 64, False),
        (4, 4, 64, True),
        (6, 2, 64, True),
        (4, 4, 128, False),
        (6, 2, 128, False),
        (4, 4, 128, True),
        (6, 2, 128, True),
    ],
)
def test_fp4_qk_runtime_matches_bf16_reference(num_heads, num_heads_kv, head_dim, causal):
    _require_sm100()
    _run_fp4_runtime_case(num_heads, num_heads_kv, head_dim=head_dim, causal=causal)


def test_fp4_qk_runtime_matches_bf16_reference_long_seqlen_d128_noncausal_mha():
    _require_sm100()
    _run_fp4_runtime_case(4, 4, head_dim=128, causal=False, seqlen_q=512, seqlen_k=512)


@pytest.mark.parametrize("seqlen", [512, 2048])
def test_fp4_qk_runtime_matches_bf16_reference_long_seqlen_d128_noncausal_gqa(seqlen):
    _require_sm100()
    _run_fp4_runtime_case(6, 2, head_dim=128, causal=False, seqlen_q=seqlen, seqlen_k=seqlen)


def test_fp4_qk_runtime_explicit_pack_gqa_matches_bf16_reference():
    _require_sm100()
    _run_fp4_runtime_case(6, 2, head_dim=128, causal=False, pack_gqa=True)


@pytest.mark.parametrize(
    "num_heads,num_heads_kv,head_dim,causal",
    [
        (4, 4, 64, False),
        (6, 2, 64, False),
        (4, 4, 64, True),
        (6, 2, 64, True),
        (4, 4, 128, False),
        (6, 2, 128, False),
        (4, 4, 128, True),
        (6, 2, 128, True),
    ],
)
def test_fp4_pv_runtime_matches_bf16_reference(num_heads, num_heads_kv, head_dim, causal):
    _require_sm100()
    _run_fp4_runtime_case(
        num_heads,
        num_heads_kv,
        head_dim=head_dim,
        causal=causal,
        use_fp4_pv=True,
    )


def test_fp4_pv_runtime_matches_bf16_reference_masked_edge_causal():
    _require_sm100()
    _run_fp4_runtime_case(
        4,
        4,
        head_dim=64,
        causal=True,
        seqlen_q=64,
        seqlen_k=128,
        use_fp4_pv=True,
    )


def test_fp4_pv_cta_quant_runtime_matches_bf16_reference_masked_edge_causal():
    _require_sm100()
    _run_fp4_runtime_case(
        4,
        4,
        head_dim=64,
        causal=True,
        seqlen_q=64,
        seqlen_k=128,
        use_fp4_pv=True,
        extra_env={"FLASH_ATTN_FP4_PV_ENABLE_CTA_ENCODE": "1"},
    )


@pytest.mark.parametrize("head_dim", [64, 128])
def test_fp4_pv_cta_quant_pv_only_causal_oracle_matches_bf16_regular_loader(head_dim):
    _require_sm100()
    _run_fp4_pv_causal_oracle_case(
        head_dim=head_dim,
        direct_loader=False,
        expect_close=True,
        atol=2.5e-1,
    )


@pytest.mark.parametrize("head_dim", [64, 128])
def test_fp4_pv_cta_quant_pv_only_causal_oracle_matches_bf16_direct_loader(head_dim):
    _require_sm100()
    _run_fp4_pv_causal_oracle_case(
        head_dim=head_dim,
        direct_loader=True,
        expect_close=True,
        atol=2.5e-1,
    )


def test_fp4_pv_cta_quant_runtime_matches_legacy_path():
    _require_sm100()
    _run_fp4_pv_cta_quant_compare_case(
        head_dim=64,
        causal=False,
        direct_loader=False,
    )


def test_fp4_pv_cta_quant_direct_loader_runtime_matches_legacy_path():
    _require_sm100()
    _run_fp4_pv_cta_quant_compare_case(
        head_dim=64,
        causal=False,
        direct_loader=True,
    )


@pytest.mark.parametrize(
    "kind,head_dim",
    [
        ("mha", 64),
        ("mha", 128),
        ("gqa", 128),
    ],
)
def test_fp4_qk_benchmark_noncausal_target_rows_are_numerically_clean(kind, head_dim):
    _require_sm100()
    result = _benchmark_case(
        kind=kind,
        seqlen=512,
        head_dim=head_dim,
        causal=False,
        batch_size=2,
        scale_value=0.125,
        warmup=5,
        iters=10,
    )
    assert result["qk_out_max"] < 0.02
    assert result["qk_lse_max"] < 0.02


def test_fp4_qk_benchmark_causal_mha_d64_row_is_clean():
    _require_sm100()
    result = _benchmark_case(
        kind="mha",
        seqlen=512,
        head_dim=64,
        causal=True,
        batch_size=2,
        scale_value=0.125,
        warmup=5,
        iters=10,
    )
    assert result["qk_out_max"] < 0.02
    assert result["qk_lse_max"] < 0.02


def test_fp4_qk_benchmark_causal_mha_d128_row_regression_watch():
    _require_sm100()
    result = _benchmark_case(
        kind="mha",
        seqlen=512,
        head_dim=128,
        causal=True,
        batch_size=2,
        scale_value=0.125,
        warmup=5,
        iters=10,
    )
    assert result["qk_out_max"] < 0.2
    assert result["qk_lse_max"] < 0.2


def test_fp4_pv_benchmark_controller_aggregates_medians_and_failures():
    samples = [
        {
            "kind": "mha",
            "hq": 4,
            "hkv": 4,
            "d": 128,
            "causal": False,
            "seqlen": 512,
            "qkfast_ms": 10.0,
            "pv_fused_ms": 9.5,
            "bf16_ms": 11.0,
            "qkfast_out_max": 0.01,
            "qkfast_lse_max": 0.02,
            "pv_fused_out_max": 0.03,
            "pv_fused_lse_max": 0.04,
        },
        {
            "kind": "mha",
            "hq": 4,
            "hkv": 4,
            "d": 128,
            "causal": False,
            "seqlen": 512,
            "qkfast_ms": 12.0,
            "pv_fused_ms": 10.5,
            "bf16_ms": 12.0,
            "qkfast_out_max": 0.02,
            "qkfast_lse_max": 0.01,
            "pv_fused_out_max": 0.04,
            "pv_fused_lse_max": 0.02,
        },
        {
            "kind": "mha",
            "hq": 4,
            "hkv": 4,
            "d": 128,
            "causal": False,
            "seqlen": 512,
            "qkfast_ms": 11.0,
            "pv_fused_ms": 10.0,
            "bf16_ms": 10.0,
            "qkfast_out_max": 0.03,
            "qkfast_lse_max": 0.03,
            "pv_fused_out_max": 0.01,
            "pv_fused_lse_max": 0.03,
        },
        {
            "kind": "mha",
            "hq": 4,
            "hkv": 4,
            "d": 128,
            "causal": False,
            "seqlen": 512,
            "qkfast_ms": 9.0,
            "pv_fused_ms": 8.5,
            "bf16_ms": 9.0,
            "qkfast_out_max": 0.02,
            "qkfast_lse_max": 0.05,
            "pv_fused_out_max": 0.05,
            "pv_fused_lse_max": 0.01,
        },
        {
            "kind": "mha",
            "hq": 4,
            "hkv": 4,
            "d": 128,
            "causal": False,
            "seqlen": 512,
            "qkfast_ms": 13.0,
            "pv_fused_ms": 11.5,
            "bf16_ms": 13.0,
            "qkfast_out_max": 0.01,
            "qkfast_lse_max": 0.02,
            "pv_fused_out_max": 0.02,
            "pv_fused_lse_max": 0.05,
        },
    ]
    failures = [{"returncode": 1}, {"returncode": 1}]

    aggregated = _aggregate_benchmark_runs(samples, failures)

    assert aggregated["success_count"] == 5
    assert aggregated["failure_count"] == 2
    assert aggregated["qkfast_ms"] == 11.0
    assert aggregated["pv_fused_ms"] == 10.0
    assert aggregated["bf16_ms"] == 11.0
    assert aggregated["qkfast_over_bf16"] == pytest.approx(1.0)
    assert aggregated["pv_fused_over_qkfast"] == pytest.approx(10.0 / 11.0)
    assert aggregated["pv_fused_over_bf16"] == pytest.approx(10.0 / 11.0)
    assert aggregated["qkfast_out_max"] == 0.03
    assert aggregated["qkfast_lse_max"] == 0.05
    assert aggregated["pv_fused_out_max"] == 0.05
    assert aggregated["pv_fused_lse_max"] == 0.05


@pytest.mark.parametrize("direct_loader", [False])
def test_fp4_pv_probe_causal_row0_identity(direct_loader):
    _require_sm100()
    _run_fp4_pv_probe(
        """
        import torch
        from flash_attn.cute.interface import _flash_attn_fwd

        FP4_GRID = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=torch.float32,
            device="cuda",
        )

        def _unpack_fp4(packed):
            packed_i32 = packed.to(torch.int32)
            return torch.stack((packed_i32 & 0xF, (packed_i32 >> 4) & 0xF), dim=-1).flatten(start_dim=-2)

        def _swizzle_fp4_vt_scale(scale_vt):
            batch_size, num_heads, head_dim, seqlen_groups = scale_vt.shape
            out = torch.empty_like(scale_vt)
            flat_in = scale_vt.reshape(batch_size, num_heads, head_dim, seqlen_groups)
            flat_out = out.view(batch_size, num_heads, -1)
            for d in range(head_dim):
                tile_m, row_in_tile = divmod(d, 64)
                quad, row_mod16 = divmod(row_in_tile, 16)
                for seq_group in range(seqlen_groups):
                    offset = (
                        tile_m * 64 * seqlen_groups
                        + (seq_group // 4) * 256
                        + (seq_group % 4)
                        + quad * 4
                        + row_mod16 * 16
                    )
                    flat_out[:, :, offset] = flat_in[:, :, d, seq_group]
            return out

        batch_size, seqlen_q, seqlen_k, num_heads, head_dim = 1, 64, 128, 4, 64
        seqlen_k_padded = ((seqlen_k + 127) // 128) * 128
        q = torch.zeros(batch_size, seqlen_q, num_heads, head_dim // 2, device="cuda", dtype=torch.uint8)
        k = torch.zeros(batch_size, seqlen_k, num_heads, head_dim // 2, device="cuda", dtype=torch.uint8)
        v = torch.zeros(batch_size, num_heads, head_dim, seqlen_k_padded // 2, device="cuda", dtype=torch.uint8)
        v[:, :, :, 0].fill_(0x22)
        q_scale = torch.ones(batch_size, seqlen_q, num_heads, head_dim // 16, device="cuda", dtype=torch.float8_e4m3fn)
        k_scale = torch.ones(batch_size, seqlen_k, num_heads, head_dim // 16, device="cuda", dtype=torch.float8_e4m3fn)
        v_scale = torch.ones(batch_size, num_heads, head_dim, seqlen_k_padded // 16, device="cuda", dtype=torch.float8_e4m3fn)
        v_scale = _swizzle_fp4_vt_scale(v_scale)

        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            causal=True,
            return_lse=True,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
            use_fp4_pv=True,
            v_scale=v_scale,
        )
        torch.cuda.synchronize()
        expected = torch.ones_like(out[:, 0:1].float())
        torch.testing.assert_close(out[:, 0:1].float(), expected, atol=2e-1, rtol=5e-2)
        assert torch.isfinite(lse.float()).all().item()
        """,
        direct_loader=direct_loader,
    )


@pytest.mark.parametrize("direct_loader", [False])
def test_fp4_pv_probe_constant_v_populates_all_output_channels(direct_loader):
    _require_sm100()
    _run_fp4_pv_probe(
        """
        import torch
        from flash_attn.cute.interface import _flash_attn_fwd

        def _swizzle_fp4_vt_scale(scale_vt):
            batch_size, num_heads, head_dim, seqlen_groups = scale_vt.shape
            out = torch.empty_like(scale_vt)
            flat_in = scale_vt.reshape(batch_size, num_heads, head_dim, seqlen_groups)
            flat_out = out.view(batch_size, num_heads, -1)
            for d in range(head_dim):
                tile_m, row_in_tile = divmod(d, 64)
                quad, row_mod16 = divmod(row_in_tile, 16)
                for seq_group in range(seqlen_groups):
                    offset = (
                        tile_m * 64 * seqlen_groups
                        + (seq_group // 4) * 256
                        + (seq_group % 4)
                        + quad * 4
                        + row_mod16 * 16
                    )
                    flat_out[:, :, offset] = flat_in[:, :, d, seq_group]
            return out

        batch_size, seqlen_q, seqlen_k, num_heads, head_dim = 1, 64, 64, 4, 128
        seqlen_k_padded = ((seqlen_k + 127) // 128) * 128
        q = torch.zeros(batch_size, seqlen_q, num_heads, head_dim // 2, device="cuda", dtype=torch.uint8)
        k = torch.zeros(batch_size, seqlen_k, num_heads, head_dim // 2, device="cuda", dtype=torch.uint8)
        v = torch.full((batch_size, num_heads, head_dim, seqlen_k_padded // 2), 0x22, device="cuda", dtype=torch.uint8)
        q_scale = torch.ones(batch_size, seqlen_q, num_heads, head_dim // 16, device="cuda", dtype=torch.float8_e4m3fn)
        k_scale = torch.ones(batch_size, seqlen_k, num_heads, head_dim // 16, device="cuda", dtype=torch.float8_e4m3fn)
        v_scale = torch.ones(batch_size, num_heads, head_dim, seqlen_k_padded // 16, device="cuda", dtype=torch.float8_e4m3fn)
        v_scale = _swizzle_fp4_vt_scale(v_scale)

        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            causal=False,
            return_lse=True,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
            use_fp4_pv=True,
            v_scale=v_scale,
        )
        torch.cuda.synchronize()
        expected = torch.ones_like(out.float())
        torch.testing.assert_close(out.float(), expected, atol=2e-1, rtol=5e-2)
        assert torch.isfinite(lse.float()).all().item()
        """,
        direct_loader=direct_loader,
    )


@pytest.mark.parametrize("direct_loader", [False])
def test_fp4_pv_probe_v_scale_axis_is_colwise(direct_loader):
    _require_sm100()
    _run_fp4_pv_probe(
        """
        import torch
        from flash_attn.cute.interface import _flash_attn_fwd

        def _swizzle_fp4_vt_scale(scale_vt):
            batch_size, num_heads, head_dim, seqlen_groups = scale_vt.shape
            out = torch.empty_like(scale_vt)
            flat_in = scale_vt.reshape(batch_size, num_heads, head_dim, seqlen_groups)
            flat_out = out.view(batch_size, num_heads, -1)
            for d in range(head_dim):
                tile_m, row_in_tile = divmod(d, 64)
                quad, row_mod16 = divmod(row_in_tile, 16)
                for seq_group in range(seqlen_groups):
                    offset = (
                        tile_m * 64 * seqlen_groups
                        + (seq_group // 4) * 256
                        + (seq_group % 4)
                        + quad * 4
                        + row_mod16 * 16
                    )
                    flat_out[:, :, offset] = flat_in[:, :, d, seq_group]
            return out

        batch_size, seqlen_q, seqlen_k, num_heads, head_dim = 1, 64, 64, 4, 128
        seqlen_k_padded = ((seqlen_k + 127) // 128) * 128
        q = torch.zeros(batch_size, seqlen_q, num_heads, head_dim // 2, device="cuda", dtype=torch.uint8)
        k = torch.zeros(batch_size, seqlen_k, num_heads, head_dim // 2, device="cuda", dtype=torch.uint8)
        v = torch.full((batch_size, num_heads, head_dim, seqlen_k_padded // 2), 0x22, device="cuda", dtype=torch.uint8)
        q_scale = torch.ones(batch_size, seqlen_q, num_heads, head_dim // 16, device="cuda", dtype=torch.float8_e4m3fn)
        k_scale = torch.ones(batch_size, seqlen_k, num_heads, head_dim // 16, device="cuda", dtype=torch.float8_e4m3fn)
        scale_blocks = torch.tensor([0.125, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0], device="cuda", dtype=torch.float32)
        v_scale_logical = scale_blocks.repeat_interleave(16).view(1, 1, head_dim, 1).expand(batch_size, num_heads, head_dim, seqlen_k_padded // 16)
        v_scale = _swizzle_fp4_vt_scale(v_scale_logical.to(torch.float8_e4m3fn))

        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            causal=False,
            return_lse=True,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
            use_fp4_pv=True,
            v_scale=v_scale,
        )
        torch.cuda.synchronize()
        expected = scale_blocks.repeat_interleave(16).view(1, 1, 1, head_dim).expand_as(out.float())
        torch.testing.assert_close(out.float(), expected, atol=2e-1, rtol=5e-2)
        assert torch.isfinite(lse.float()).all().item()
        """,
        direct_loader=direct_loader,
    )
