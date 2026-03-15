import types

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch


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
        lambda _device: types.SimpleNamespace(multi_processor_count=132),
    )
    # CuTe runtime fake tensors do not expose layout metadata yet, so keep the
    # fake-compile tests focused on dispatch/cache behavior rather than layout recasting.
    monkeypatch.setattr("flash_attn.cute.interface.to_cute_fp4_tensor", lambda tensor: tensor)
    monkeypatch.setattr(_flash_attn_fwd, "compile_cache", {})
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


def _unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    packed_i32 = packed.to(torch.int32)
    return torch.stack((packed_i32 & 0xF, (packed_i32 >> 4) & 0xF), dim=-1).flatten(start_dim=-2)


def _dequantize_fp4(packed: torch.Tensor, scale: torch.Tensor, sf_vec_size: int) -> torch.Tensor:
    values = FP4_GRID.to(device=packed.device)[_unpack_fp4(packed).long()]
    return (
        values.unflatten(dim=-1, sizes=(-1, sf_vec_size))
        * scale.to(torch.float32).unsqueeze(-1)
    ).flatten(start_dim=-2)


@pytest.mark.parametrize("fp4_qk_format", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_fp4_qk_fake_compile_dense_forward(monkeypatch, fp4_qk_format, head_dim, causal):
    compile_calls = _install_fake_cuda_runtime(monkeypatch)

    with FakeTensorMode():
        q, k, v, q_scale, k_scale = _make_fake_fp4_dense_inputs(
            fp4_qk_format=fp4_qk_format,
            head_dim=head_dim,
            head_dim_v=head_dim,
        )
        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            causal=causal,
            return_lse=True,
            _arch=100,
            fp4_qk_format=fp4_qk_format,
            q_scale=q_scale,
            k_scale=k_scale,
        )

    assert out.dtype == v.dtype
    assert lse.dtype == torch.float32
    assert len(compile_calls) == 1
    assert len(_flash_attn_fwd.compile_cache) == 1


def test_fp4_compile_cache_separates_formats_and_bf16(monkeypatch):
    compile_calls = _install_fake_cuda_runtime(monkeypatch)

    with FakeTensorMode():
        bf16_q, bf16_k, bf16_v = _make_fake_dense_inputs(head_dim=64, head_dim_v=64)
        _flash_attn_fwd(bf16_q, bf16_k, bf16_v, causal=False, _arch=100)

        for fp4_qk_format in ("nvfp4", "mxfp4"):
            q, k, v, q_scale, k_scale = _make_fake_fp4_dense_inputs(
                fp4_qk_format=fp4_qk_format,
                head_dim=64,
                head_dim_v=64,
            )
            _flash_attn_fwd(
                q,
                k,
                v,
                causal=False,
                _arch=100,
                fp4_qk_format=fp4_qk_format,
                q_scale=q_scale,
                k_scale=k_scale,
            )

    assert len(compile_calls) == 3
    assert len(_flash_attn_fwd.compile_cache) == 3


@pytest.mark.parametrize(
    "kwargs,expected_error",
    [
        ({"q_scale": None}, ValueError),
        ({"k_scale": None}, ValueError),
        ({"q_scale_dtype": torch.float16}, TypeError),
        ({"k_scale_shape_delta": 1}, ValueError),
        ({"num_splits": 2}, NotImplementedError),
        ({"window_size_left": 8}, NotImplementedError),
        ({"cu_seqlens_q": True}, NotImplementedError),
        ({"seqused_q": True}, NotImplementedError),
        ({"page_table": True}, NotImplementedError),
        ({"block_sparse": True}, NotImplementedError),
    ],
)
def test_fp4_qk_validation_errors(monkeypatch, kwargs, expected_error):
    _install_fake_cuda_runtime(monkeypatch)

    with FakeTensorMode():
        q, k, v, q_scale, k_scale = _make_fake_fp4_dense_inputs(
            fp4_qk_format="nvfp4",
            head_dim=64,
            head_dim_v=64,
        )
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
                causal=False,
                window_size_left=kwargs.get("window_size_left"),
                num_splits=kwargs.get("num_splits", 1),
                block_sparse_tensors=block_sparse_tensors,
                _arch=100,
                fp4_qk_format="nvfp4",
                q_scale=q_scale,
                k_scale=k_scale,
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


def _require_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for runtime FP4 tests.")
    major, minor = torch.cuda.get_device_capability()
    if major not in (10, 11):
        pytest.skip(f"FP4 runtime tests require SM100/SM110, got {major}.{minor}.")


@pytest.mark.parametrize(
    "num_heads,num_heads_kv",
    [
        (4, 4),
        (6, 2),
    ],
)
def test_fp4_qk_runtime_matches_bf16_reference(num_heads, num_heads_kv):
    _require_sm100()

    torch.manual_seed(0)
    batch_size = 2
    seqlen_q = 128
    seqlen_k = 128
    head_dim = 64
    head_dim_v = 64
    device = "cuda"
    scale_value = 0.25

    q_packed = torch.randint(
        0,
        256,
        (batch_size, seqlen_q, num_heads, head_dim // 2),
        device=device,
        dtype=torch.uint8,
    )
    k_packed = torch.randint(
        0,
        256,
        (batch_size, seqlen_k, num_heads_kv, head_dim // 2),
        device=device,
        dtype=torch.uint8,
    )
    v = torch.randn(batch_size, seqlen_k, num_heads_kv, head_dim_v, device=device, dtype=torch.bfloat16) * 0.25
    q_scale = torch.full(
        (batch_size, seqlen_q, num_heads, head_dim // 16),
        scale_value,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    k_scale = torch.full(
        (batch_size, seqlen_k, num_heads_kv, head_dim // 16),
        scale_value,
        device=device,
        dtype=torch.float8_e4m3fn,
    )

    q_ref = _dequantize_fp4(q_packed, q_scale, 16).to(torch.bfloat16)
    k_ref = _dequantize_fp4(k_packed, k_scale, 16).to(torch.bfloat16)

    out_fp4, lse_fp4 = _flash_attn_fwd(
        q_packed,
        k_packed,
        v,
        causal=False,
        return_lse=True,
        fp4_qk_format="nvfp4",
        q_scale=q_scale,
        k_scale=k_scale,
    )
    out_ref, lse_ref = _flash_attn_fwd(
        q_ref,
        k_ref,
        v,
        causal=False,
        return_lse=True,
    )

    torch.testing.assert_close(out_fp4.float(), out_ref.float(), atol=2e-1, rtol=5e-2)
    torch.testing.assert_close(lse_fp4.float(), lse_ref.float(), atol=2e-1, rtol=5e-2)
