import importlib.util
import math
from pathlib import Path

import pytest
import torch


def _safe_cuda_capability():
    try:
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_capability()
    except Exception:
        return None


try:
    from flash_attn.cute import (
        analyze_explicit_2d_sparse_forward,
        analyze_hsa_approx_sparse_gemm_forward,
        analyze_hsa_shared_sparse_gemm_forward,
        analyze_hsa_sparse24_feasibility,
        build_explicit_2d_sparse_case,
        hybrid_backward_to_attend_mask,
        backward_packed_masks_to_attend_mask,
        backward_descriptors_to_attend_mask,
        build_hsa_schedule,
        compute_hsa_mask,
        flash_attn_func,
        flash_attn_hsa_func,
        flash_attn_hsa_sparse_func,
        fused_forward_to_attend_mask,
        forward_descriptors_to_attend_mask,
        forward_tile_masks_to_attend_mask,
        get_hsa_mask_mod,
        hsa_reference_attention,
        schedule_to_attend_mask,
        summarize_hsa_approx_sparse_gemm_forward,
        summarize_hsa_shared_sparse_gemm_forward,
        summarize_hsa_sparse24_feasibility,
    )
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - import guard for unsupported envs
    analyze_hsa_approx_sparse_gemm_forward = None
    analyze_explicit_2d_sparse_forward = None
    analyze_hsa_shared_sparse_gemm_forward = None
    analyze_hsa_sparse24_feasibility = None
    build_hsa_schedule = None
    build_explicit_2d_sparse_case = None
    backward_descriptors_to_attend_mask = None
    backward_packed_masks_to_attend_mask = None
    compute_hsa_mask = None
    flash_attn_func = None
    flash_attn_hsa_func = None
    flash_attn_hsa_sparse_func = None
    fused_forward_to_attend_mask = None
    forward_descriptors_to_attend_mask = None
    forward_tile_masks_to_attend_mask = None
    hybrid_backward_to_attend_mask = None
    get_hsa_mask_mod = None
    hsa_reference_attention = None
    schedule_to_attend_mask = None
    summarize_hsa_approx_sparse_gemm_forward = None
    summarize_hsa_shared_sparse_gemm_forward = None
    summarize_hsa_sparse24_feasibility = None
    _IMPORT_ERROR = exc


_CUDA_CAPABILITY = _safe_cuda_capability()
HAS_HSA_FA4 = (
    _IMPORT_ERROR is None
    and _CUDA_CAPABILITY is not None
    and _CUDA_CAPABILITY[0] >= 9
)

pytestmark = pytest.mark.skipif(
    not HAS_HSA_FA4,
    reason=f"HSA FA4 test requires CUDA SM90+ with FA4 available ({_IMPORT_ERROR!r})",
)

HAS_HSA_SPARSE_FA4 = HAS_HSA_FA4 and _CUDA_CAPABILITY[0] >= 10
_BENCHMARK_HSA_MODULE = None


def _unwrap_output(result):
    return result[0] if isinstance(result, tuple) else result


def _load_benchmark_hsa_module():
    global _BENCHMARK_HSA_MODULE
    if _BENCHMARK_HSA_MODULE is None:
        module_path = Path(__file__).with_name("benchmark_hsa.py")
        spec = importlib.util.spec_from_file_location("benchmark_hsa_test_helper", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _BENCHMARK_HSA_MODULE = module
    return _BENCHMARK_HSA_MODULE


def _empty_stream_like(stream):
    stream_type = stream.__class__
    device = stream.query_indices.device
    return stream_type(
        query_indices=torch.empty(0, dtype=torch.int32, device=device),
        key_indices=torch.empty(0, dtype=torch.int32, device=device),
        row_indices=torch.empty(0, dtype=torch.int32, device=device),
        cu_seqlens_q=torch.tensor([0], dtype=torch.int32, device=device),
        cu_seqlens_k=torch.tensor([0], dtype=torch.int32, device=device),
        max_seqlen_q=0,
        max_seqlen_k=0,
    )


def _make_hsa_metadata(batch_size, seqlen, device):
    keep_ids = torch.zeros(batch_size, 3, seqlen, dtype=torch.int32, device=device)
    hash_ids = torch.zeros(batch_size, 3, seqlen, dtype=torch.int32, device=device)

    for batch_idx in range(batch_size):
        cursor = 0
        doc_id = 0
        sec_id = 0
        sent_id = 0
        while cursor < seqlen:
            keep_ids[batch_idx, 2, cursor] = 1
            hash_ids[batch_idx, :, cursor] = torch.tensor(
                [sent_id, sec_id, doc_id], dtype=torch.int32, device=device
            )
            cursor += 1
            if cursor >= seqlen:
                break

            for _ in range(2):
                if cursor >= seqlen:
                    break
                keep_ids[batch_idx, 1, cursor] = 1
                keep_ids[batch_idx, 2, cursor] = 1
                hash_ids[batch_idx, :, cursor] = torch.tensor(
                    [sent_id, sec_id, doc_id], dtype=torch.int32, device=device
                )
                cursor += 1
                if cursor >= seqlen:
                    break

                for _ in range(2):
                    if cursor >= seqlen:
                        break
                    keep_ids[batch_idx, 0, cursor] = 1
                    keep_ids[batch_idx, 1, cursor] = 1
                    hash_ids[batch_idx, :, cursor] = torch.tensor(
                        [sent_id, sec_id, doc_id], dtype=torch.int32, device=device
                    )
                    cursor += 1
                    if cursor >= seqlen:
                        break

                    body_tokens = min(7 + ((sent_id + batch_idx) % 5), seqlen - cursor)
                    keep_ids[batch_idx, 0, cursor:cursor + body_tokens] = 1
                    hash_ids[batch_idx, 0, cursor:cursor + body_tokens] = sent_id
                    hash_ids[batch_idx, 1, cursor:cursor + body_tokens] = sec_id
                    hash_ids[batch_idx, 2, cursor:cursor + body_tokens] = doc_id
                    cursor += body_tokens
                    sent_id += 1
                sec_id += 1
            doc_id += 1

    return keep_ids, hash_ids


def _make_sentence_only_metadata(batch_size, seqlen, device):
    keep_ids = torch.zeros(batch_size, 3, seqlen, dtype=torch.int32, device=device)
    hash_ids = torch.zeros(batch_size, 3, seqlen, dtype=torch.int32, device=device)
    keep_ids[:, 0, :] = 1
    keep_ids[:, 1, 0] = 1
    keep_ids[:, 2, 0] = 1
    return keep_ids, hash_ids


def _dense_causal_out_lse(q, k, v, softmax_scale):
    scores = torch.einsum("bthd,bshd->bhts", q.float(), k.float()) * softmax_scale
    causal = torch.triu(
        torch.ones(q.shape[1], k.shape[1], device=q.device, dtype=torch.bool),
        diagonal=1,
    )
    scores = scores.masked_fill(causal, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhts,bshd->bthd", probs, v.float()).to(dtype=q.dtype)
    return out.contiguous(), lse.contiguous()


def _assert_close(lhs, rhs, name, atol=2e-2, rtol=2e-2):
    max_diff = (lhs - rhs).abs().max().item()
    mean_diff = (lhs - rhs).abs().mean().item()
    assert torch.allclose(lhs, rhs, atol=atol, rtol=rtol), (
        f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
    )


def test_hsa_forward_matches_reference_on_partial_tile():
    device = "cuda"
    dtype = torch.bfloat16
    batch_size, seqlen, nheads, headdim = 1, 137, 4, 64
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)

    out_fa4 = _unwrap_output(flash_attn_hsa_func(q, k, v, keep_ids, hash_ids))
    out_ref = hsa_reference_attention(q, k, v, keep_ids, hash_ids)

    _assert_close(out_fa4.float(), out_ref.float(), "hsa_forward")


def test_hsa_backward_matches_reference():
    device = "cuda"
    dtype = torch.bfloat16
    batch_size, seqlen, nheads, headdim = 1, 65, 2, 64
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)

    q_data = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    k_data = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    v_data = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)

    q_fa4 = q_data.clone().requires_grad_(True)
    k_fa4 = k_data.clone().requires_grad_(True)
    v_fa4 = v_data.clone().requires_grad_(True)
    out_fa4 = _unwrap_output(flash_attn_hsa_func(q_fa4, k_fa4, v_fa4, keep_ids, hash_ids))
    loss_fa4 = out_fa4.float().square().mean()
    loss_fa4.backward()

    q_ref = q_data.clone().float().requires_grad_(True)
    k_ref = k_data.clone().float().requires_grad_(True)
    v_ref = v_data.clone().float().requires_grad_(True)
    out_ref = hsa_reference_attention(
        q_ref.to(dtype=dtype),
        k_ref.to(dtype=dtype),
        v_ref.to(dtype=dtype),
        keep_ids,
        hash_ids,
    ).float()
    loss_ref = out_ref.square().mean()
    loss_ref.backward()

    _assert_close(out_fa4.float(), out_ref, "hsa_backward_output")
    _assert_close(q_fa4.grad.float(), q_ref.grad, "q_grad", atol=6e-2, rtol=6e-2)
    _assert_close(k_fa4.grad.float(), k_ref.grad, "k_grad", atol=6e-2, rtol=6e-2)
    _assert_close(v_fa4.grad.float(), v_ref.grad, "v_grad", atol=6e-2, rtol=6e-2)


def test_fixed_length_public_api_supports_aux_tensors_in_backward():
    device = "cuda"
    dtype = torch.bfloat16
    batch_size, seqlen, nheads, headdim = 1, 33, 2, 64
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)

    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype).requires_grad_(True)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype).requires_grad_(True)
    v = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype).requires_grad_(True)

    out = _unwrap_output(
        flash_attn_func(
            q,
            k,
            v,
            mask_mod=get_hsa_mask_mod(),
            aux_tensors=[keep_ids, hash_ids],
        )
    )
    loss = out.float().square().mean()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()
    assert torch.isfinite(v.grad).all()


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_hsa_schedule_reconstructs_dense_mask():
    device = "cuda"
    keep_ids, hash_ids = _make_hsa_metadata(batch_size=1, seqlen=97, device=device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    dense_mask = schedule_to_attend_mask(schedule)
    ref_mask = torch.isfinite(compute_hsa_mask(keep_ids, hash_ids))

    assert torch.equal(dense_mask, ref_mask)


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_hsa_forward_descriptors_reconstruct_dense_mask():
    device = "cuda"
    keep_ids, hash_ids = _make_hsa_metadata(batch_size=1, seqlen=97, device=device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    dense_mask = forward_descriptors_to_attend_mask(schedule)
    ref_mask = torch.isfinite(compute_hsa_mask(keep_ids, hash_ids))

    assert torch.equal(dense_mask, ref_mask)


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_hsa_fused_forward_schedule_reconstructs_dense_mask():
    import flash_attn.cute.hsa as hsa_module

    device = "cuda"
    keep_ids, hash_ids = _make_hsa_metadata(batch_size=1, seqlen=193, device=device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    fused_schedule = hsa_module._build_hsa_fused_forward_schedule(
        schedule,
        q_block_size=256,
        k_block_size=128,
    )
    dense_mask = fused_forward_to_attend_mask(schedule, fused_schedule)
    ref_mask = torch.isfinite(compute_hsa_mask(keep_ids, hash_ids))

    assert torch.equal(dense_mask, ref_mask)


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_hsa_forward_tile_masks_reconstruct_dense_mask():
    import flash_attn.cute.hsa as hsa_module

    device = "cuda"
    keep_ids, hash_ids = _make_hsa_metadata(batch_size=1, seqlen=193, device=device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    sparse_tensors, tile_masks = hsa_module._build_forward_hsa_tile_masks(
        schedule,
        q_block_size=256,
        k_block_size=128,
    )
    dense_mask = forward_tile_masks_to_attend_mask(schedule, sparse_tensors, tile_masks)
    ref_mask = torch.isfinite(compute_hsa_mask(keep_ids, hash_ids))

    assert torch.equal(dense_mask, ref_mask)


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_hsa_backward_descriptors_reconstruct_dense_mask():
    device = "cuda"
    keep_ids, hash_ids = _make_hsa_metadata(batch_size=1, seqlen=97, device=device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    dense_mask = backward_descriptors_to_attend_mask(schedule)
    ref_mask = torch.isfinite(compute_hsa_mask(keep_ids, hash_ids))

    assert torch.equal(dense_mask, ref_mask)


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_hsa_backward_packed_masks_reconstruct_dense_mask():
    import flash_attn.cute.hsa as hsa_module

    device = "cuda"
    keep_ids, hash_ids = _make_hsa_metadata(batch_size=1, seqlen=193, device=device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    sparse_tensors, packed_masks = hsa_module._build_backward_hsa_packed_masks(
        schedule,
        q_block_size=128,
        k_block_size=128,
    )
    dense_mask = backward_packed_masks_to_attend_mask(schedule, sparse_tensors, packed_masks)
    ref_mask = torch.isfinite(compute_hsa_mask(keep_ids, hash_ids))

    assert torch.equal(dense_mask, ref_mask)


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_hsa_backward_packed_mask_occupancy_matches_dense_mask_allowed_pairs(monkeypatch):
    import flash_attn.cute.hsa as hsa_module

    benchmark_hsa = _load_benchmark_hsa_module()
    device = "cuda"
    batch_size, seqlen, nheads, headdim = 1, 193, 4, 64
    keep_ids, hash_ids = _make_hsa_metadata(batch_size=batch_size, seqlen=seqlen, device=device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)

    monkeypatch.setenv("FLASH_ATTN_HSA_BACKWARD_BLOCK_Q", "64")
    monkeypatch.setenv("FLASH_ATTN_HSA_BACKWARD_BLOCK_K", "128")
    monkeypatch.setenv("FLASH_ATTN_HSA_BACKWARD_SUBTILE_FACTOR", "1")

    runtime = hsa_module._get_hsa_block_sparse_runtime(schedule, q, k)
    summary = benchmark_hsa._summarize_sparse_backward_occupancy(
        runtime.backward_sparse,
        runtime.backward_packed_masks,
        seqlen=seqlen,
    )
    ref_allowed_pairs = int(torch.isfinite(compute_hsa_mask(keep_ids, hash_ids)).sum().item())

    assert summary["allowed_pairs"] == ref_allowed_pairs


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_hsa_hybrid_backward_schedule_reconstructs_dense_mask():
    import flash_attn.cute.hsa as hsa_module

    device = "cuda"
    keep_ids, hash_ids = _make_hsa_metadata(batch_size=1, seqlen=193, device=device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    hybrid_schedule = hsa_module._build_hsa_hybrid_backward_schedule(
        schedule,
        k_block_size=128,
        anchor_row_panel_size=64,
    )
    dense_mask = hybrid_backward_to_attend_mask(schedule, hybrid_schedule)
    ref_mask = torch.isfinite(compute_hsa_mask(keep_ids, hash_ids))

    assert torch.equal(dense_mask, ref_mask)


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_hsa_sparse_forward_matches_reference():
    device = "cuda"
    dtype = torch.bfloat16
    batch_size, seqlen, nheads, headdim = 1, 129, 4, 64
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)

    out_sparse = _unwrap_output(
        flash_attn_hsa_sparse_func(q, k, v, keep_ids=keep_ids, hash_ids=hash_ids, hsa_schedule=schedule)
    )
    out_ref = hsa_reference_attention(q, k, v, keep_ids, hash_ids)

    _assert_close(out_sparse.float(), out_ref.float(), "hsa_sparse_forward")


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_hsa_sparse_runtime_uses_canonical_schedule_only():
    device = "cuda"
    dtype = torch.bfloat16
    batch_size, seqlen, nheads, headdim = 1, 129, 4, 64
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    schedule.sentence_stream = _empty_stream_like(schedule.sentence_stream)
    schedule.section_prefix_stream = _empty_stream_like(schedule.section_prefix_stream)
    schedule.document_prefix_stream = _empty_stream_like(schedule.document_prefix_stream)
    schedule.section_self_indices = torch.empty(0, dtype=torch.int32, device=device)
    schedule.document_self_indices = torch.empty(0, dtype=torch.int32, device=device)
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)

    out_sparse = _unwrap_output(
        flash_attn_hsa_sparse_func(q, k, v, keep_ids=keep_ids, hash_ids=hash_ids, hsa_schedule=schedule)
    )
    out_ref = hsa_reference_attention(q, k, v, keep_ids, hash_ids)

    _assert_close(out_sparse.float(), out_ref.float(), "hsa_sparse_canonical_schedule")


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_hsa_sparse_fused_forward_matches_reference(monkeypatch):
    import flash_attn.cute.hsa as hsa_module

    device = "cuda"
    dtype = torch.bfloat16
    batch_size, seqlen, nheads, headdim = 1, 129, 4, 64
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    schedule.sentence_stream = _empty_stream_like(schedule.sentence_stream)
    schedule.section_prefix_stream = _empty_stream_like(schedule.section_prefix_stream)
    schedule.document_prefix_stream = _empty_stream_like(schedule.document_prefix_stream)
    schedule.section_self_indices = torch.empty(0, dtype=torch.int32, device=device)
    schedule.document_self_indices = torch.empty(0, dtype=torch.int32, device=device)

    monkeypatch.setenv("FLASH_ATTN_HSA_USE_FUSED_FWD", "1")
    monkeypatch.setattr(
        hsa_module,
        "_run_hsa_blocksparse_forward",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("default sparse forward helper used")),
    )
    monkeypatch.setattr(
        hsa_module,
        "_run_hsa_fused_forward",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("python fused forward helper used")),
    )
    monkeypatch.setattr(
        hsa_module,
        "get_hsa_schedule_mask_mod",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("global schedule mask_mod used")),
    )

    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)

    out_sparse = _unwrap_output(
        flash_attn_hsa_sparse_func(q, k, v, keep_ids=keep_ids, hash_ids=hash_ids, hsa_schedule=schedule)
    )
    out_ref = hsa_reference_attention(q, k, v, keep_ids, hash_ids)

    _assert_close(out_sparse.float(), out_ref.float(), "hsa_sparse_fused_forward")


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_hsa_true_fused_equal_head_forward_exports_sentence_cache_without_precompute(monkeypatch):
    import flash_attn.cute.hsa as hsa_module
    import flash_attn.cute.flash_hsa_fwd_sm100 as hsa_fwd_module

    device = "cuda"
    dtype = torch.bfloat16
    batch_size, seqlen, nheads, headdim = 1, 256, 4, 64
    keep_ids, hash_ids = _make_sentence_only_metadata(batch_size, seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)

    monkeypatch.setenv("FLASH_ATTN_HSA_USE_MONOLITHIC_BWD", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_USE_TRUE_FUSED_BWD", "1")
    monkeypatch.delenv("FLASH_ATTN_HSA_USE_FUSED_FWD", raising=False)
    monkeypatch.setattr(
        hsa_fwd_module,
        "_run_hsa_sentence_stream_cache_precompute",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy sentence cache precompute used")
        ),
    )
    monkeypatch.setattr(
        hsa_module,
        "_run_hsa_blocksparse_forward",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("blocksparse forward fallback used")
        ),
    )

    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)

    out_sparse = _unwrap_output(
        flash_attn_hsa_sparse_func(q, k, v, keep_ids=keep_ids, hash_ids=hash_ids, hsa_schedule=schedule)
    )
    out_ref = hsa_reference_attention(q, k, v, keep_ids, hash_ids)

    _assert_close(out_sparse.float(), out_ref.float(), "hsa_true_fused_equal_head_forward_export")


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_hsa_true_fused_gqa_forward_keeps_blocksparse_fallback(monkeypatch):
    import flash_attn.cute.flash_hsa_fwd_sm100 as hsa_fwd_module

    device = "cuda"
    dtype = torch.bfloat16
    batch_size, seqlen, nheads, n_kv_heads, headdim = 1, 129, 4, 2, 64
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)

    monkeypatch.setenv("FLASH_ATTN_HSA_USE_MONOLITHIC_BWD", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_USE_TRUE_FUSED_BWD", "1")
    monkeypatch.delenv("FLASH_ATTN_HSA_USE_FUSED_FWD", raising=False)
    monkeypatch.setattr(
        hsa_fwd_module,
        "run_hsa_fwd_sm100_fused",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("equal-head fused forward used for GQA")),
    )

    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)

    out_sparse = _unwrap_output(
        flash_attn_hsa_sparse_func(q, k, v, keep_ids=keep_ids, hash_ids=hash_ids, hsa_schedule=schedule)
    )
    out_ref = hsa_reference_attention(q, k, v, keep_ids, hash_ids)

    _assert_close(out_sparse.float(), out_ref.float(), "hsa_true_fused_gqa_forward_fallback")


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_hsa_sparse_fast_path_does_not_use_legacy_varlen_helpers(monkeypatch):
    import flash_attn.cute.hsa as hsa_module
    import flash_attn.cute.flash_hsa_bwd_sm100 as hsa_bwd_module

    device = "cuda"
    dtype = torch.bfloat16
    batch_size, seqlen, nheads, headdim = 1, 129, 4, 64
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    schedule.sentence_stream = _empty_stream_like(schedule.sentence_stream)
    schedule.section_prefix_stream = _empty_stream_like(schedule.section_prefix_stream)
    schedule.document_prefix_stream = _empty_stream_like(schedule.document_prefix_stream)
    schedule.section_self_indices = torch.empty(0, dtype=torch.int32, device=device)
    schedule.document_self_indices = torch.empty(0, dtype=torch.int32, device=device)
    monkeypatch.setenv("FLASH_ATTN_HSA_USE_PACKED_BWD", "1")

    monkeypatch.setattr(
        hsa_module,
        "_run_varlen_fa4_stream",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy forward helper used")),
    )
    monkeypatch.setattr(
        hsa_module,
        "_run_varlen_fa4_bwd",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy backward helper used")),
    )
    monkeypatch.setattr(
        hsa_module,
        "_run_hsa_packed_mask_backward",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("packed-mask backward helper used")),
    )
    monkeypatch.setattr(
        hsa_module,
        "_run_hsa_backward_panel_batched",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy batched backward helper used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "run_hsa_bwd_sm100_monolithic",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("monolithic backward scaffold used")),
    )

    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype).requires_grad_(True)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype).requires_grad_(True)
    v = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype).requires_grad_(True)

    out_sparse = _unwrap_output(
        flash_attn_hsa_sparse_func(
            q,
            k,
            v,
            keep_ids=keep_ids,
            hash_ids=hash_ids,
            hsa_schedule=schedule,
        )
    )
    loss_sparse = out_sparse.float().square().mean()
    loss_sparse.backward()

    out_ref = hsa_reference_attention(q.detach(), k.detach(), v.detach(), keep_ids, hash_ids)
    _assert_close(out_sparse.float(), out_ref.float(), "hsa_sparse_no_legacy_helpers")
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_hsa_hybrid_backward_matches_reference(monkeypatch):
    device = "cuda"
    dtype = torch.bfloat16
    batch_size, seqlen, nheads, headdim = 1, 65, 4, 64
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    monkeypatch.setenv("FLASH_ATTN_HSA_USE_PACKED_BWD", "1")

    q_data = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    k_data = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    v_data = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)

    q_sparse = q_data.clone().requires_grad_(True)
    k_sparse = k_data.clone().requires_grad_(True)
    v_sparse = v_data.clone().requires_grad_(True)
    out_sparse = _unwrap_output(
        flash_attn_hsa_sparse_func(
            q_sparse,
            k_sparse,
            v_sparse,
            keep_ids=keep_ids,
            hash_ids=hash_ids,
            hsa_schedule=schedule,
        )
    )
    loss_sparse = out_sparse.float().square().mean()
    loss_sparse.backward()

    q_ref = q_data.clone().float().requires_grad_(True)
    k_ref = k_data.clone().float().requires_grad_(True)
    v_ref = v_data.clone().float().requires_grad_(True)
    out_ref = hsa_reference_attention(
        q_ref.to(dtype=dtype),
        k_ref.to(dtype=dtype),
        v_ref.to(dtype=dtype),
        keep_ids,
        hash_ids,
    ).float()
    loss_ref = out_ref.square().mean()
    loss_ref.backward()

    _assert_close(out_sparse.float(), out_ref, "hsa_hybrid_backward_output")
    _assert_close(q_sparse.grad.float(), q_ref.grad, "hsa_hybrid_q_grad", atol=6e-2, rtol=6e-2)
    _assert_close(k_sparse.grad.float(), k_ref.grad, "hsa_hybrid_k_grad", atol=6e-2, rtol=6e-2)
    _assert_close(v_sparse.grad.float(), v_ref.grad, "hsa_hybrid_v_grad", atol=6e-2, rtol=6e-2)


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_hsa_sparse_backward_matches_reference():
    device = "cuda"
    dtype = torch.bfloat16
    batch_size, seqlen, nheads, headdim = 1, 65, 4, 64
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)

    q_data = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    k_data = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    v_data = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)

    q_sparse = q_data.clone().requires_grad_(True)
    k_sparse = k_data.clone().requires_grad_(True)
    v_sparse = v_data.clone().requires_grad_(True)
    out_sparse = _unwrap_output(
        flash_attn_hsa_sparse_func(
            q_sparse,
            k_sparse,
            v_sparse,
            keep_ids=keep_ids,
            hash_ids=hash_ids,
            hsa_schedule=schedule,
        )
    )
    loss_sparse = out_sparse.float().square().mean()
    loss_sparse.backward()

    q_ref = q_data.clone().float().requires_grad_(True)
    k_ref = k_data.clone().float().requires_grad_(True)
    v_ref = v_data.clone().float().requires_grad_(True)
    out_ref = hsa_reference_attention(
        q_ref.to(dtype=dtype),
        k_ref.to(dtype=dtype),
        v_ref.to(dtype=dtype),
        keep_ids,
        hash_ids,
    ).float()
    loss_ref = out_ref.square().mean()
    loss_ref.backward()

    _assert_close(out_sparse.float(), out_ref, "hsa_sparse_backward_output")
    _assert_close(q_sparse.grad.float(), q_ref.grad, "hsa_sparse_q_grad", atol=6e-2, rtol=6e-2)
    _assert_close(k_sparse.grad.float(), k_ref.grad, "hsa_sparse_k_grad", atol=6e-2, rtol=6e-2)
    _assert_close(v_sparse.grad.float(), v_ref.grad, "hsa_sparse_v_grad", atol=6e-2, rtol=6e-2)


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
@pytest.mark.parametrize("n_kv_heads", [4, 2])
def test_hsa_monolithic_sentence_full_family_uses_mma_and_matches_reference(monkeypatch, n_kv_heads):
    import flash_attn.cute.hsa as hsa_module
    import flash_attn.cute.flash_hsa_bwd_sm100 as hsa_bwd_module

    device = "cuda"
    dtype = torch.bfloat16
    batch_size, seqlen, nheads, headdim = 1, 256, 4, 64
    keep_ids, hash_ids = _make_sentence_only_metadata(batch_size, seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    monkeypatch.setenv("FLASH_ATTN_HSA_USE_MONOLITHIC_BWD", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_USE_KERNEL_SENTENCE_FULL", "1")
    monkeypatch.delenv("FLASH_ATTN_HSA_USE_PACKED_BWD", raising=False)
    monkeypatch.delenv("FLASH_ATTN_HSA_USE_HYBRID_BWD", raising=False)

    monkeypatch.setattr(
        hsa_module,
        "_run_hsa_packed_mask_backward",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("packed-mask backward helper used")),
    )
    monkeypatch.setattr(
        hsa_module,
        "_run_hsa_backward_panel_batched",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy batched backward helper used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_descriptor_mma_batches",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("sentence-family panel batch helper used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "run_hsa_bwd_sm100_packed",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("packed backward prototype used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_panel_batch_cute",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("panel CuTe helper used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_bwd_rowgroup_main_cute",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("row-group backward helper used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_bwd_monolithic_main_torch",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Torch monolithic fallback used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_bwd_sentence_full_kernel_slice",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("sentence_full scalar kernel slice used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_sentence_full_mma_batches",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("sentence_full host-side MMA prepass used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_sentence_full_fa4_fastpath",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("sentence_full host-side FA4 fastpath used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_bwd_sentence_full_scalar",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("sentence_full scalar helper used")),
    )

    q_data = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    k_data = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)
    v_data = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)

    q_sparse = q_data.clone().requires_grad_(True)
    k_sparse = k_data.clone().requires_grad_(True)
    v_sparse = v_data.clone().requires_grad_(True)
    out_sparse = _unwrap_output(
        flash_attn_hsa_sparse_func(
            q_sparse,
            k_sparse,
            v_sparse,
            keep_ids=keep_ids,
            hash_ids=hash_ids,
            hsa_schedule=schedule,
        )
    )
    loss_sparse = out_sparse.float().square().mean()
    loss_sparse.backward()

    q_ref = q_data.clone().float().requires_grad_(True)
    k_ref = k_data.clone().float().requires_grad_(True)
    v_ref = v_data.clone().float().requires_grad_(True)
    out_ref = hsa_reference_attention(
        q_ref.to(dtype=dtype),
        k_ref.to(dtype=dtype),
        v_ref.to(dtype=dtype),
        keep_ids,
        hash_ids,
    ).float()
    loss_ref = out_ref.square().mean()
    loss_ref.backward()

    _assert_close(out_sparse.float(), out_ref, "hsa_monolithic_sentence_full_output")
    _assert_close(q_sparse.grad.float(), q_ref.grad, "hsa_monolithic_sentence_full_q_grad", atol=6e-2, rtol=6e-2)
    _assert_close(k_sparse.grad.float(), k_ref.grad, "hsa_monolithic_sentence_full_k_grad", atol=6e-2, rtol=6e-2)
    _assert_close(v_sparse.grad.float(), v_ref.grad, "hsa_monolithic_sentence_full_v_grad", atol=6e-2, rtol=6e-2)


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
@pytest.mark.parametrize("n_kv_heads", [4, 2])
def test_hsa_sentence_full_direct_kernel_matches_reference(monkeypatch, n_kv_heads):
    import flash_attn.cute.hsa as hsa_module
    import flash_attn.cute.flash_hsa_bwd_sm100 as hsa_bwd_module

    device = "cuda"
    dtype = torch.bfloat16
    batch_size, seqlen, nheads, headdim = 1, 256, 4, 64
    softmax_scale = headdim ** -0.5
    keep_ids, hash_ids = _make_sentence_only_metadata(batch_size, seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    monolithic_schedule = hsa_module._get_hsa_monolithic_backward_schedule(schedule)

    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_bwd_sentence_full_scalar",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("sentence_full scalar helper used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_bwd_sentence_full_kernel_slice",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("sentence_full scalar kernel slice used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_sentence_full_fa4_fastpath",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("sentence_full host-side FA4 fastpath used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_sentence_full_mma_batches",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("sentence_full host-side MMA prepass used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_monolithic_mma_batches",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("host-side monolithic MMA prepass used")),
    )

    q_data = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    k_data = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)
    v_data = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)
    dout = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    out, lse = _dense_causal_out_lse(q_data, k_data, v_data, softmax_scale)

    dq, dk, dv = hsa_bwd_module._run_hsa_bwd_sentence_full_direct(
        q_data,
        k_data,
        v_data,
        out,
        dout,
        lse,
        schedule,
        monolithic_schedule,
        softmax_scale,
        False,
    )

    q_ref = q_data.clone().float().requires_grad_(True)
    k_ref = k_data.clone().float().requires_grad_(True)
    v_ref = v_data.clone().float().requires_grad_(True)
    out_ref = hsa_reference_attention(
        q_ref.to(dtype=dtype),
        k_ref.to(dtype=dtype),
        v_ref.to(dtype=dtype),
        keep_ids,
        hash_ids,
    ).float()
    (out_ref * dout.float()).sum().backward()

    _assert_close(dq.float(), q_ref.grad, "hsa_sentence_full_direct_q_grad", atol=6e-2, rtol=6e-2)
    _assert_close(dk.float(), k_ref.grad, "hsa_sentence_full_direct_k_grad", atol=6e-2, rtol=6e-2)
    _assert_close(dv.float(), v_ref.grad, "hsa_sentence_full_direct_v_grad", atol=6e-2, rtol=6e-2)


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
@pytest.mark.parametrize("n_kv_heads", [4, 2])
def test_hsa_true_fused_sentence_only_matches_reference(monkeypatch, n_kv_heads):
    import flash_attn.cute.hsa as hsa_module
    import flash_attn.cute.flash_hsa_bwd_sm100 as hsa_bwd_module

    device = "cuda"
    dtype = torch.bfloat16
    batch_size, seqlen, nheads, headdim = 1, 256, 4, 64
    keep_ids, hash_ids = _make_sentence_only_metadata(batch_size, seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    monkeypatch.setenv("FLASH_ATTN_HSA_USE_MONOLITHIC_BWD", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_USE_TRUE_FUSED_BWD", "1")
    monkeypatch.delenv("FLASH_ATTN_HSA_USE_KERNEL_SENTENCE_FULL", raising=False)
    monkeypatch.delenv("FLASH_ATTN_HSA_USE_PACKED_BWD", raising=False)
    monkeypatch.delenv("FLASH_ATTN_HSA_USE_HYBRID_BWD", raising=False)
    hsa_bwd_module.run_hsa_bwd_sm100_monolithic.compile_cache.clear()
    hsa_bwd_module.run_hsa_bwd_sm100_monolithic.launch_plan_cache.clear()

    monkeypatch.setattr(
        hsa_module,
        "_run_hsa_packed_mask_backward",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("packed-mask backward helper used")),
    )
    monkeypatch.setattr(
        hsa_module,
        "_run_hsa_backward_panel_batched",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy batched backward helper used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_bwd_sentence_families_direct",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("split sentence direct path used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_bwd_sentence_families_direct_rows",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("split sentence row precompute used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_bwd_anchor_only_main_cute",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("split anchor-only path used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_sentence_scatter_rows_kernel",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy per-tensor sentence scatter used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_pack_rows_kernel",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy sentence row pack used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_sentence_pack_outputs_kernel",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy sentence output pack used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_sentence_scatter_triplet_rows_kernel",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy sentence triplet scatter used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_descriptor_mma_batches",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("sentence-family panel batch helper used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_sentence_full_kernel_fa4_slice",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("sentence_full split kernel fastpath used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_sentence_full_fa4_fastpath",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("sentence_full host fastpath used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_bwd_monolithic_main_cute",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy mixed monolithic kernel used")),
    )
    original_lazy_cute_imports = hsa_bwd_module._lazy_cute_imports

    def _lazy_cute_imports_without_fwd():
        cutlass, cute, utils, fast_sampling, flash_attn_func, _flash_attn_fwd, _flash_attn_bwd = (
            original_lazy_cute_imports()
        )
        return (
            cutlass,
            cute,
            utils,
            fast_sampling,
            flash_attn_func,
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("sentence_lse backward recompute used")
            ),
            _flash_attn_bwd,
        )

    monkeypatch.setattr(
        hsa_bwd_module,
        "_lazy_cute_imports",
        _lazy_cute_imports_without_fwd,
    )

    q_data = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    k_data = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)
    v_data = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)

    q_sparse = q_data.clone().requires_grad_(True)
    k_sparse = k_data.clone().requires_grad_(True)
    v_sparse = v_data.clone().requires_grad_(True)
    out_sparse = _unwrap_output(
        flash_attn_hsa_sparse_func(
            q_sparse,
            k_sparse,
            v_sparse,
            keep_ids=keep_ids,
            hash_ids=hash_ids,
            hsa_schedule=schedule,
        )
    )
    loss_sparse = out_sparse.float().square().mean()
    loss_sparse.backward()

    q_ref = q_data.clone().float().requires_grad_(True)
    k_ref = k_data.clone().float().requires_grad_(True)
    v_ref = v_data.clone().float().requires_grad_(True)
    out_ref = hsa_reference_attention(
        q_ref.to(dtype=dtype),
        k_ref.to(dtype=dtype),
        v_ref.to(dtype=dtype),
        keep_ids,
        hash_ids,
    ).float()
    loss_ref = out_ref.square().mean()
    loss_ref.backward()

    _assert_close(out_sparse.float(), out_ref, "hsa_true_fused_sentence_only_output")
    _assert_close(q_sparse.grad.float(), q_ref.grad, "hsa_true_fused_sentence_only_q_grad", atol=6e-2, rtol=6e-2)
    _assert_close(k_sparse.grad.float(), k_ref.grad, "hsa_true_fused_sentence_only_k_grad", atol=6e-2, rtol=6e-2)
    _assert_close(v_sparse.grad.float(), v_ref.grad, "hsa_true_fused_sentence_only_v_grad", atol=6e-2, rtol=6e-2)


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
@pytest.mark.parametrize(
    ("n_kv_heads", "make_metadata", "enable_2cta", "expected"),
    [
        (4, _make_sentence_only_metadata, False, False),
        (4, _make_sentence_only_metadata, True, True),
        (2, _make_sentence_only_metadata, False, False),
        (2, _make_sentence_only_metadata, True, False),
        (4, _make_hsa_metadata, False, False),
        (4, _make_hsa_metadata, True, False),
        (2, _make_hsa_metadata, False, False),
        (2, _make_hsa_metadata, True, False),
    ],
)
def test_hsa_true_fused_sentence_2cta_policy(n_kv_heads, make_metadata, enable_2cta, expected, monkeypatch):
    import flash_attn.cute.hsa as hsa_module
    import flash_attn.cute.flash_hsa_bwd_sm100 as hsa_bwd_module

    device = "cuda"
    batch_size, seqlen, nheads, headdim = 1, 65, 4, 64
    if enable_2cta:
        monkeypatch.setenv("FLASH_ATTN_HSA_USE_SENTENCE_VARLEN_2CTA", "1")
    else:
        monkeypatch.delenv("FLASH_ATTN_HSA_USE_SENTENCE_VARLEN_2CTA", raising=False)
    keep_ids, hash_ids = make_metadata(batch_size, seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    monolithic_schedule = hsa_module._get_hsa_monolithic_backward_schedule(schedule)
    q = torch.empty(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)
    k = torch.empty(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=torch.bfloat16)

    assert (
        hsa_bwd_module._should_force_hsa_sentence_varlen_2cta(
            q,
            k,
            use_true_fused=True,
            monolithic_schedule=monolithic_schedule,
        )
        is expected
    )


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
@pytest.mark.parametrize("n_kv_heads", [4, 2])
def test_hsa_monolithic_sentence_families_bypass_kernel_when_no_anchors(monkeypatch, n_kv_heads):
    import flash_attn.cute.hsa as hsa_module
    import flash_attn.cute.flash_hsa_bwd_sm100 as hsa_bwd_module

    device = "cuda"
    dtype = torch.bfloat16
    batch_size, seqlen, nheads, headdim = 1, 256, 4, 64
    keep_ids, hash_ids = _make_sentence_only_metadata(batch_size, seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    monkeypatch.setenv("FLASH_ATTN_HSA_USE_MONOLITHIC_BWD", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_USE_KERNEL_SENTENCE_FULL", "1")
    monkeypatch.delenv("FLASH_ATTN_HSA_USE_PACKED_BWD", raising=False)
    monkeypatch.delenv("FLASH_ATTN_HSA_USE_HYBRID_BWD", raising=False)

    monkeypatch.setattr(
        hsa_module,
        "_run_hsa_packed_mask_backward",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("packed-mask backward helper used")),
    )
    monkeypatch.setattr(
        hsa_module,
        "_run_hsa_backward_panel_batched",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy batched backward helper used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_descriptor_mma_batches",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("sentence-family panel batch helper used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_bwd_monolithic_main_cute",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("monolithic mixed-family kernel used")),
    )
    original_lazy_cute_imports = hsa_bwd_module._lazy_cute_imports

    def _lazy_cute_imports_without_fwd():
        cutlass, cute, utils, fast_sampling, flash_attn_func, _flash_attn_fwd, _flash_attn_bwd = (
            original_lazy_cute_imports()
        )
        return (
            cutlass,
            cute,
            utils,
            fast_sampling,
            flash_attn_func,
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("sentence_lse backward recompute used")
            ),
            _flash_attn_bwd,
        )

    monkeypatch.setattr(
        hsa_bwd_module,
        "_lazy_cute_imports",
        _lazy_cute_imports_without_fwd,
    )

    q_data = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    k_data = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)
    v_data = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)

    q_sparse = q_data.clone().requires_grad_(True)
    k_sparse = k_data.clone().requires_grad_(True)
    v_sparse = v_data.clone().requires_grad_(True)
    out_sparse = _unwrap_output(
        flash_attn_hsa_sparse_func(
            q_sparse,
            k_sparse,
            v_sparse,
            keep_ids=keep_ids,
            hash_ids=hash_ids,
            hsa_schedule=schedule,
        )
    )
    loss_sparse = out_sparse.float().square().mean()
    loss_sparse.backward()

    q_ref = q_data.clone().float().requires_grad_(True)
    k_ref = k_data.clone().float().requires_grad_(True)
    v_ref = v_data.clone().float().requires_grad_(True)
    out_ref = hsa_reference_attention(
        q_ref.to(dtype=dtype),
        k_ref.to(dtype=dtype),
        v_ref.to(dtype=dtype),
        keep_ids,
        hash_ids,
    ).float()
    loss_ref = out_ref.square().mean()
    loss_ref.backward()

    _assert_close(out_sparse.float(), out_ref, "hsa_monolithic_sentence_families_output")
    _assert_close(q_sparse.grad.float(), q_ref.grad, "hsa_monolithic_sentence_families_q_grad", atol=6e-2, rtol=6e-2)
    _assert_close(k_sparse.grad.float(), k_ref.grad, "hsa_monolithic_sentence_families_k_grad", atol=6e-2, rtol=6e-2)
    _assert_close(v_sparse.grad.float(), v_ref.grad, "hsa_monolithic_sentence_families_v_grad", atol=6e-2, rtol=6e-2)


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
@pytest.mark.parametrize("n_kv_heads", [4, 2])
def test_hsa_monolithic_sentence_tail_family_uses_mma_and_matches_reference(monkeypatch, n_kv_heads):
    import flash_attn.cute.hsa as hsa_module
    import flash_attn.cute.flash_hsa_bwd_sm100 as hsa_bwd_module

    device = "cuda"
    dtype = torch.bfloat16
    batch_size, seqlen, nheads, headdim = 1, 65, 4, 64
    keep_ids, hash_ids = _make_sentence_only_metadata(batch_size, seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    monkeypatch.setenv("FLASH_ATTN_HSA_USE_MONOLITHIC_BWD", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_USE_KERNEL_SENTENCE_FULL", "1")
    monkeypatch.delenv("FLASH_ATTN_HSA_USE_PACKED_BWD", raising=False)
    monkeypatch.delenv("FLASH_ATTN_HSA_USE_HYBRID_BWD", raising=False)

    monkeypatch.setattr(
        hsa_module,
        "_run_hsa_packed_mask_backward",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("packed-mask backward helper used")),
    )
    monkeypatch.setattr(
        hsa_module,
        "_run_hsa_backward_panel_batched",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy batched backward helper used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_descriptor_mma_batches",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("sentence-family panel batch helper used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "run_hsa_bwd_sm100_packed",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("packed backward prototype used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_panel_batch_cute",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("panel CuTe helper used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_bwd_rowgroup_main_cute",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("row-group backward helper used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_bwd_monolithic_main_torch",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Torch monolithic fallback used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_bwd_sentence_full_kernel_slice",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("sentence_full scalar kernel slice used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_bwd_sentence_full_scalar",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("sentence_full scalar helper used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_bwd_sentence_tail_scalar",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("sentence_tail scalar helper used")),
    )

    q_data = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    k_data = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)
    v_data = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)

    q_sparse = q_data.clone().requires_grad_(True)
    k_sparse = k_data.clone().requires_grad_(True)
    v_sparse = v_data.clone().requires_grad_(True)
    out_sparse = _unwrap_output(
        flash_attn_hsa_sparse_func(
            q_sparse,
            k_sparse,
            v_sparse,
            keep_ids=keep_ids,
            hash_ids=hash_ids,
            hsa_schedule=schedule,
        )
    )
    loss_sparse = out_sparse.float().square().mean()
    loss_sparse.backward()

    q_ref = q_data.clone().float().requires_grad_(True)
    k_ref = k_data.clone().float().requires_grad_(True)
    v_ref = v_data.clone().float().requires_grad_(True)
    out_ref = hsa_reference_attention(
        q_ref.to(dtype=dtype),
        k_ref.to(dtype=dtype),
        v_ref.to(dtype=dtype),
        keep_ids,
        hash_ids,
    ).float()
    loss_ref = out_ref.square().mean()
    loss_ref.backward()

    _assert_close(out_sparse.float(), out_ref, "hsa_monolithic_sentence_tail_output")
    _assert_close(q_sparse.grad.float(), q_ref.grad, "hsa_monolithic_sentence_tail_q_grad", atol=6e-2, rtol=6e-2)
    _assert_close(k_sparse.grad.float(), k_ref.grad, "hsa_monolithic_sentence_tail_k_grad", atol=6e-2, rtol=6e-2)
    _assert_close(v_sparse.grad.float(), v_ref.grad, "hsa_monolithic_sentence_tail_v_grad", atol=6e-2, rtol=6e-2)


def _run_mixed_sparse_mask_case(
    *,
    batch_size: int,
    seqlen: int,
    nheads: int,
    headdim: int,
    n_kv_heads: int,
    keep_ids: torch.Tensor,
    hash_ids: torch.Tensor,
    q_data: torch.Tensor | None = None,
    k_data: torch.Tensor | None = None,
    v_data: torch.Tensor | None = None,
):
    device = "cuda"
    dtype = torch.bfloat16
    if q_data is None:
        q_data = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    if k_data is None:
        k_data = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)
    if v_data is None:
        v_data = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)

    q_sparse = q_data.clone().requires_grad_(True)
    k_sparse = k_data.clone().requires_grad_(True)
    v_sparse = v_data.clone().requires_grad_(True)
    out_sparse = _unwrap_output(
        flash_attn_hsa_sparse_func(
            q_sparse,
            k_sparse,
            v_sparse,
            keep_ids=keep_ids,
            hash_ids=hash_ids,
            hsa_schedule=build_hsa_schedule(keep_ids, hash_ids),
        )
    )
    loss_sparse = out_sparse.float().square().mean()
    loss_sparse.backward()

    q_ref = q_data.clone().float().requires_grad_(True)
    k_ref = k_data.clone().float().requires_grad_(True)
    v_ref = v_data.clone().float().requires_grad_(True)
    out_ref = hsa_reference_attention(
        q_ref.to(dtype=dtype),
        k_ref.to(dtype=dtype),
        v_ref.to(dtype=dtype),
        keep_ids,
        hash_ids,
    ).float()
    loss_ref = out_ref.square().mean()
    loss_ref.backward()
    return out_sparse, q_sparse.grad, k_sparse.grad, v_sparse.grad, out_ref, q_ref.grad, k_ref.grad, v_ref.grad


def _capture_synthetic_forward_saved_tensors(
    *,
    q_data: torch.Tensor,
    k_data: torch.Tensor,
    v_data: torch.Tensor,
    keep_ids: torch.Tensor,
    hash_ids: torch.Tensor,
):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module
    import flash_attn.cute.hsa as hsa_module

    schedule = build_hsa_schedule(keep_ids, hash_ids)
    softmax_scale = 1.0 / (q_data.shape[-1] ** 0.5)
    runtime = hsa_module._get_hsa_block_sparse_runtime(schedule, q_data, k_data)
    hsa_module._ensure_hsa_synthetic_grid_metadata(schedule, runtime)
    out, lse, *_ = synthetic_module.run_hsa_fwd_sm100_synthetic_grid(
        q_data,
        k_data,
        v_data,
        schedule,
        softmax_scale,
        runtime=runtime,
    )
    return out.detach(), lse.detach()


def _run_synthetic_case_with_saved_tensor_override(
    monkeypatch,
    *,
    batch_size: int,
    seqlen: int,
    nheads: int,
    headdim: int,
    n_kv_heads: int,
    keep_ids: torch.Tensor,
    hash_ids: torch.Tensor,
    q_data: torch.Tensor,
    k_data: torch.Tensor,
    v_data: torch.Tensor,
    saved_out: torch.Tensor,
    saved_lse: torch.Tensor,
):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    def _override_fwd(q, k, v, schedule, softmax_scale, *, runtime=None):
        del schedule, softmax_scale, runtime
        sentence_lse, sentence_q_stream, sentence_k_stream, sentence_v_stream, sentence_out_stream = (
            synthetic_module._empty_sentence_cache(q, k, v)
        )
        return (
            saved_out.clone(),
            saved_lse.clone(),
            sentence_lse,
            sentence_q_stream,
            sentence_k_stream,
            sentence_v_stream,
            sentence_out_stream,
        )

    with monkeypatch.context() as local_patch:
        local_patch.setattr(synthetic_module, "run_hsa_fwd_sm100_synthetic_grid", _override_fwd)
        local_patch.setenv("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "1")
        local_patch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_Q", "2")
        local_patch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_K", "2")
        local_patch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MAX_PACKED_K", "128")
        local_patch.delenv("FLASH_ATTN_HSA_SYNTHETIC_MICRO_FWD", raising=False)
        return _run_mixed_sparse_mask_case(
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            headdim=headdim,
            n_kv_heads=n_kv_heads,
            keep_ids=keep_ids,
            hash_ids=hash_ids,
            q_data=q_data,
            k_data=k_data,
            v_data=v_data,
        )


def _run_benchmark_case_env(case_name: str, env_updates: dict[str, str | None]):
    benchmark_hsa = _load_benchmark_hsa_module()
    case = next(candidate for candidate in benchmark_hsa.ALL_CASES if candidate.name == case_name)
    schedule, keep_ids, hash_ids, q_data, k_data, v_data = benchmark_hsa._build_case_tensors(case)

    hsa_forward = lambda q, k, v: benchmark_hsa._run_sparse_attention(
        q, k, v, keep_ids, hash_ids, schedule, env_updates=env_updates
    )
    q = q_data.clone().requires_grad_(True)
    k = k_data.clone().requires_grad_(True)
    v = v_data.clone().requires_grad_(True)
    out = benchmark_hsa._unwrap_output(hsa_forward(q, k, v))
    loss = out.float().square().mean()
    dq, dk, dv = torch.autograd.grad(loss, (q, k, v))
    return out.detach(), dq.detach(), dk.detach(), dv.detach()


def _assert_finite_gradients(name, *grads):
    for grad in grads:
        assert grad is not None, f"{name}: missing gradient"
        assert torch.isfinite(grad).all(), f"{name}: non-finite gradient"


def _assert_gradient_finite_status(name, grad, *, expected_nonfinite: int = 0):
    nonfinite = int((~torch.isfinite(grad)).sum().item())
    assert nonfinite == expected_nonfinite, f"{name}: expected {expected_nonfinite} non-finite elements, got {nonfinite}"


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
@pytest.mark.parametrize("use_true_fused", [False, True])
@pytest.mark.parametrize("n_kv_heads", [4, 2])
def test_hsa_mixed_monolithic_env_routes_to_packed_mask_backward(monkeypatch, use_true_fused, n_kv_heads):
    import flash_attn.cute.hsa as hsa_module
    import flash_attn.cute.flash_hsa_bwd_sm100 as hsa_bwd_module

    device = "cuda"
    dtype = torch.bfloat16
    batch_size, seqlen, nheads, headdim = 1, 65, 4, 64
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    monkeypatch.setenv("FLASH_ATTN_HSA_USE_MONOLITHIC_BWD", "1")
    if use_true_fused:
        monkeypatch.setenv("FLASH_ATTN_HSA_USE_TRUE_FUSED_BWD", "1")
        monkeypatch.delenv("FLASH_ATTN_HSA_USE_KERNEL_SENTENCE_FULL", raising=False)
    else:
        monkeypatch.delenv("FLASH_ATTN_HSA_USE_TRUE_FUSED_BWD", raising=False)
        monkeypatch.setenv("FLASH_ATTN_HSA_USE_KERNEL_SENTENCE_FULL", "1")
    monkeypatch.delenv("FLASH_ATTN_HSA_USE_PACKED_BWD", raising=False)
    monkeypatch.delenv("FLASH_ATTN_HSA_USE_HYBRID_BWD", raising=False)

    packed_mask_calls = 0
    original_packed_mask_backward = hsa_module._run_hsa_packed_mask_backward

    def _tracked_packed_mask_backward(*args, **kwargs):
        nonlocal packed_mask_calls
        packed_mask_calls += 1
        return original_packed_mask_backward(*args, **kwargs)

    monkeypatch.setattr(hsa_module, "_run_hsa_packed_mask_backward", _tracked_packed_mask_backward)
    monkeypatch.setattr(
        hsa_module,
        "_run_hsa_backward_panel_batched",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy graph-traversal backward helper used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "run_hsa_bwd_sm100_monolithic",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("mixed monolithic backward used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "run_hsa_bwd_sm100_packed",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("packed graph-traversal prototype used")),
    )

    out_sparse, q_grad, k_grad, v_grad, out_ref, q_ref_grad, k_ref_grad, v_ref_grad = _run_mixed_sparse_mask_case(
        batch_size=batch_size,
        seqlen=seqlen,
        nheads=nheads,
        headdim=headdim,
        n_kv_heads=n_kv_heads,
        keep_ids=keep_ids,
        hash_ids=hash_ids,
    )

    assert packed_mask_calls == 1
    _assert_close(out_sparse.float(), out_ref, "hsa_mixed_sparse_mask_route_output")
    _assert_finite_gradients("hsa_mixed_sparse_mask_route_grads", q_grad, k_grad, v_grad)


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
@pytest.mark.parametrize(
    "block_q,block_k,subtile_factor",
    [
        (64, 128, 1),
        (64, 64, 1),
        (32, 64, 1),
        (32, 32, 1),
    ],
)
def test_hsa_sparse_mask_runtime_honors_requested_backward_geometry(monkeypatch, block_q, block_k, subtile_factor):
    import flash_attn.cute.hsa as hsa_module

    device = "cuda"
    batch_size, seqlen, nheads, headdim = 1, 65, 4, 64
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)

    monkeypatch.setenv("FLASH_ATTN_HSA_BACKWARD_BLOCK_Q", str(block_q))
    monkeypatch.setenv("FLASH_ATTN_HSA_BACKWARD_BLOCK_K", str(block_k))
    monkeypatch.setenv("FLASH_ATTN_HSA_BACKWARD_SUBTILE_FACTOR", str(subtile_factor))

    runtime = hsa_module._get_hsa_block_sparse_runtime(schedule, q, k)
    sparse_block_q = block_q * subtile_factor
    assert runtime.backward_block_q == block_q
    assert runtime.backward_block_k == block_k
    assert runtime.backward_subtile_factor == subtile_factor
    assert runtime.backward_sparse.block_size == (sparse_block_q, block_k)
    assert runtime.backward_sparse_torch is not None
    assert runtime.backward_sparse_torch.block_size == (sparse_block_q, block_k)
    assert runtime.backward_packed_masks.q_block_size == sparse_block_q
    assert runtime.backward_packed_masks.k_block_size == block_k


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
@pytest.mark.parametrize(
    "label,batch_size,seqlen,nheads,headdim,n_kv_heads",
    [
        ("mixed_small_eq", 1, 65, 4, 64, 4),
        ("mixed_small_gqa", 1, 65, 4, 64, 2),
        ("train_eq", 2, 1024, 8, 64, 8),
        ("train_gqa", 2, 1024, 8, 64, 2),
    ],
)
def test_hsa_mixed_sparse_mask_backward_smoke(monkeypatch, label, batch_size, seqlen, nheads, headdim, n_kv_heads):
    import flash_attn.cute.hsa as hsa_module
    import flash_attn.cute.flash_hsa_bwd_sm100 as hsa_bwd_module

    device = "cuda"
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    monkeypatch.setenv("FLASH_ATTN_HSA_USE_MONOLITHIC_BWD", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_USE_TRUE_FUSED_BWD", "1")
    monkeypatch.delenv("FLASH_ATTN_HSA_USE_KERNEL_SENTENCE_FULL", raising=False)
    monkeypatch.delenv("FLASH_ATTN_HSA_USE_PACKED_BWD", raising=False)
    monkeypatch.delenv("FLASH_ATTN_HSA_USE_HYBRID_BWD", raising=False)

    packed_mask_calls = 0
    original_packed_mask_backward = hsa_module._run_hsa_packed_mask_backward

    def _tracked_packed_mask_backward(*args, **kwargs):
        nonlocal packed_mask_calls
        packed_mask_calls += 1
        return original_packed_mask_backward(*args, **kwargs)

    monkeypatch.setattr(hsa_module, "_run_hsa_packed_mask_backward", _tracked_packed_mask_backward)
    monkeypatch.setattr(
        hsa_module,
        "_run_hsa_backward_panel_batched",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy graph-traversal backward helper used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "run_hsa_bwd_sm100_monolithic",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("mixed monolithic backward used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "run_hsa_bwd_sm100_packed",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("packed graph-traversal prototype used")),
    )

    out_sparse, q_grad, k_grad, v_grad, out_ref, q_ref_grad, k_ref_grad, v_ref_grad = _run_mixed_sparse_mask_case(
        batch_size=batch_size,
        seqlen=seqlen,
        nheads=nheads,
        headdim=headdim,
        n_kv_heads=n_kv_heads,
        keep_ids=keep_ids,
        hash_ids=hash_ids,
    )

    assert packed_mask_calls == 1
    _assert_close(out_sparse.float(), out_ref, f"{label}_output")
    _assert_finite_gradients(f"{label}_grads", q_grad, k_grad, v_grad)


def test_benchmark_hsa_external_hdt_adapter_smoke():
    benchmark_hsa = _load_benchmark_hsa_module()
    attention_fn, status = benchmark_hsa._load_external_hdt_attention()
    assert attention_fn is not None or status.startswith(("missing_", "import_failed_", "missing_loader"))


def test_hsa_sparse24_dense_block_is_ineligible():
    from flash_attn.cute.hsa_sparse24_analysis import _classify_mask_matrix

    result = _classify_mask_matrix(torch.ones((4, 4), dtype=torch.bool))

    assert not result["eligible"]
    assert "dense_segment" in result["failure_reasons"]


def test_hsa_sparse24_shared_column_block_is_b_side_eligible():
    from flash_attn.cute.hsa_sparse24_analysis import _classify_mask_matrix

    mask = torch.tensor(
        [
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [1, 0, 1, 0],
        ],
        dtype=torch.bool,
    )

    result = _classify_mask_matrix(mask)

    assert result["eligible"]
    assert result["operand"] == "B"
    assert result["support_nnz"] == 2


def test_hsa_sparse24_row_prefix_block_requires_output_mask():
    from flash_attn.cute.hsa_sparse24_analysis import _classify_mask_matrix

    mask = torch.tensor(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )

    result = _classify_mask_matrix(mask)

    assert not result["eligible"]
    assert "row_dependent_columns" in result["failure_reasons"]
    assert "requires_output_mask" in result["failure_reasons"]


def test_hsa_sparse24_bitmap_row_variation_is_ineligible():
    from flash_attn.cute.hsa_sparse24_analysis import _classify_mask_matrix

    mask = torch.tensor(
        [
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
        ],
        dtype=torch.bool,
    )

    result = _classify_mask_matrix(mask)

    assert not result["eligible"]
    assert "row_dependent_columns" in result["failure_reasons"]
    assert "requires_output_mask" in result["failure_reasons"]


def test_hsa_approx_sparse_gemm_prunes_top2_of4():
    from flash_attn.cute.hsa_approx_sparse_gemm_analysis import _prune_dense_matrix_topk_per_group

    x = torch.tensor(
        [
            [1.0, -4.0, 3.0, 2.0, -8.0, 7.0, 6.0, -5.0],
            [-1.0, 2.0, -3.0, 4.0, 5.0, -6.0, 7.0, -8.0],
        ],
        dtype=torch.float32,
    )

    pruned = _prune_dense_matrix_topk_per_group(x, group_size=4, nnz_per_group=2)

    assert pruned.shape == x.shape
    for row_idx in range(pruned.shape[0]):
        for group_start in range(0, pruned.shape[1], 4):
            group = pruned[row_idx, group_start:group_start + 4]
            assert int((group != 0).sum().item()) == 2

    assert pruned[0, 1].item() == pytest.approx(-4.0)
    assert pruned[0, 2].item() == pytest.approx(3.0)
    assert pruned[0, 0].item() == pytest.approx(0.0)
    assert pruned[0, 3].item() == pytest.approx(0.0)


def test_hsa_approx_sparse_gemm_summary_unavailable_status():
    summary = summarize_hsa_approx_sparse_gemm_forward(
        {
            "status": "requires_synthetic_grid",
            "group_size": 4,
            "nnz_per_group": 2,
            "sampled_members": 0,
            "available_members": 0,
        }
    )

    assert summary["status"] == "requires_synthetic_grid"
    assert summary["group_size"] == 4
    assert summary["nnz_per_group"] == 2
    assert summary["sampled_members"] == 0
    assert math.isnan(summary["dense_qk_ms"])


def test_hsa_shared_sparse_gemm_exact_support_grouping_is_deterministic():
    from flash_attn.cute.hsa_shared_sparse_gemm_analysis import _group_qgroups_by_exact_support

    entries = [
        {
            "qgroup_idx": 7,
            "packed_q": 2,
            "q_count": 2,
            "union_rows": torch.tensor([4, 6], dtype=torch.int32),
            "mask_words": torch.tensor([0b11, 0b10], dtype=torch.int32),
        },
        {
            "qgroup_idx": 4,
            "packed_q": 2,
            "q_count": 1,
            "union_rows": torch.tensor([9, 11], dtype=torch.int32),
            "mask_words": torch.tensor([0b01, 0b00], dtype=torch.int32),
        },
        {
            "qgroup_idx": 3,
            "packed_q": 2,
            "q_count": 2,
            "union_rows": torch.tensor([4, 6], dtype=torch.int32),
            "mask_words": torch.tensor([0b11, 0b10], dtype=torch.int32),
        },
    ]

    first = _group_qgroups_by_exact_support(entries, max_qgroups_per_bucket=4)
    second = _group_qgroups_by_exact_support(list(reversed(entries)), max_qgroups_per_bucket=4)

    first_ids = [group["qgroup_ids"] for group in first]
    second_ids = [group["qgroup_ids"] for group in second]

    assert first_ids == [[3, 7], [4]]
    assert second_ids == first_ids


def test_hsa_shared_sparse_gemm_greedy_merge_respects_constraints():
    from flash_attn.cute.hsa_shared_sparse_gemm_analysis import (
        _greedy_merge_support_groups,
        _make_support_group,
    )

    def _entry(qgroup_idx: int, support_rows: list[int]):
        return {
            "qgroup_idx": qgroup_idx,
            "packed_q": 2,
            "q_count": 2,
            "union_rows": torch.tensor(support_rows, dtype=torch.int32),
            "mask_words": torch.tensor([0b1111, 0b1111], dtype=torch.int32),
        }

    groups = [
        _make_support_group([_entry(0, [1, 2, 3, 4])]),
        _make_support_group([_entry(1, [1, 2, 3, 5])]),
        _make_support_group([_entry(2, [20, 21, 22, 23])]),
        _make_support_group([_entry(3, [1, 2, 3, 4])]),
    ]

    merged = _greedy_merge_support_groups(
        groups,
        min_jaccard=0.5,
        max_support_rows=5,
        max_qgroups_per_bucket=2,
    )

    merged_ids = [group["qgroup_ids"] for group in merged]

    assert merged_ids == [[0, 1], [2], [3]]
    assert len(merged[0]["support_rows"]) == 5


def test_hsa_shared_sparse_gemm_coarse_window_merge_grows_buckets():
    from flash_attn.cute.hsa_shared_sparse_gemm_analysis import (
        _coarse_window_merge_support_groups,
        _encode_mask_rows_to_words,
        _make_support_group,
    )
    from flash_attn.cute.hsa_shared_sparse_gemm_analysis import _decode_mask_words

    def _entry(qgroup_idx: int, support_rows: list[int]):
        return {
            "qgroup_idx": qgroup_idx,
            "packed_q": 2,
            "q_count": 2,
            "union_rows": torch.tensor(support_rows, dtype=torch.int32),
            "mask_words": torch.tensor([0b1111, 0b1111], dtype=torch.int32),
        }

    groups = [
        _make_support_group([_entry(0, [0, 1, 2, 3])]),
        _make_support_group([_entry(1, [4, 5, 6, 7])]),
        _make_support_group([_entry(2, [8, 9, 10, 11])]),
        _make_support_group([_entry(3, [20, 21, 22, 23])]),
    ]

    merged = _coarse_window_merge_support_groups(
        groups,
        max_support_rows=12,
        max_qgroups_per_bucket=4,
    )

    merged_ids = [group["qgroup_ids"] for group in merged]

    assert merged_ids == [[0, 1, 2], [3]]
    assert len(merged[0]["support_rows"]) == 12

    mask_rows = torch.tensor(
        [
            [True, False, True, True, False, False, True, False],
            [False, True, False, False, True, True, False, True],
        ],
        dtype=torch.bool,
    )
    words = _encode_mask_rows_to_words(mask_rows)
    decoded = _decode_mask_words(words.flatten(), rows=2, cols=8, words_per_row=words.shape[1])
    torch.testing.assert_close(decoded, mask_rows)


def test_hsa_shared_sparse_gemm_main_plus_residual_reconstructs_dense_qk():
    from flash_attn.cute.hsa_shared_sparse_gemm_analysis import _split_dense_matrix_main_and_residual

    q = torch.tensor(
        [
            [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0],
            [0.5, -1.5, 2.5, -3.5, 4.5, -5.5, 6.5, -7.5],
        ],
        dtype=torch.float32,
    )
    k = torch.tensor(
        [
            [1.0, 2.0, -3.0, 4.0, -5.0, 6.0, 7.0, -8.0],
            [-8.0, 7.0, 6.0, -5.0, 4.0, -3.0, 2.0, 1.0],
            [2.0, -1.0, 0.5, -0.25, -4.0, 3.0, -2.0, 1.0],
        ],
        dtype=torch.float32,
    )

    main, residual = _split_dense_matrix_main_and_residual(k, group_size=4, nnz_per_group=2)

    dense_scores = q @ k.transpose(0, 1)
    split_scores = q @ main.transpose(0, 1) + q @ residual.transpose(0, 1)

    torch.testing.assert_close(split_scores, dense_scores, atol=1e-6, rtol=1e-6)


def test_hsa_shared_sparse_gemm_summary_unavailable_status():
    summary = summarize_hsa_shared_sparse_gemm_forward(
        {
            "status": "requires_synthetic_grid",
            "group_size": 4,
            "nnz_per_group": 2,
            "available_qgroups": 0,
            "sampled_qgroups": 0,
        }
    )

    assert summary["status"] == "requires_synthetic_grid"
    assert summary["group_size"] == 4
    assert summary["nnz_per_group"] == 2
    assert summary["sampled_qgroups"] == 0
    assert math.isnan(summary["dense_qk_ms"])


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_hsa_synthetic_grid_runtime_builds_metadata(monkeypatch):
    import flash_attn.cute.hsa as hsa_module

    device = "cuda"
    batch_size, seqlen, nheads, headdim = 1, 65, 4, 64
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)

    monkeypatch.setenv("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "1")
    runtime = hsa_module._get_hsa_block_sparse_runtime(schedule, q, k)

    assert runtime.forward_synthetic_grid is not None
    assert runtime.backward_synthetic_grid is not None
    assert runtime.synthetic_grid is not None
    assert runtime.synthetic_grid.logical_block_q == 32
    assert runtime.synthetic_grid.logical_block_k == 32
    assert runtime.synthetic_grid.num_tiles > 0
    assert runtime.forward_synthetic_grid.forward_execution_plan is not None
    assert runtime.forward_synthetic_grid.forward_execution_plan["bucket_size"]


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_hsa_approx_sparse_gemm_forward_summary_runs_on_runtime_metadata(monkeypatch):
    device = "cuda"
    batch_size, seqlen, nheads, headdim = 1, 65, 4, 64
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)

    monkeypatch.setenv("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_Q", "2")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_K", "2")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MAX_PACKED_K", "32")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MAX_DIRECT_SEGMENTS", "1")

    report = analyze_hsa_approx_sparse_gemm_forward(
        schedule,
        q,
        k,
        v,
        use_synthetic_grid=True,
        warmup_iters=0,
        benchmark_iters=1,
        max_members=8,
    )
    summary = summarize_hsa_approx_sparse_gemm_forward(report)

    assert summary["status"] == "measured"
    assert summary["sampled_members"] > 0
    assert math.isfinite(summary["dense_qk_ms"])
    assert math.isfinite(summary["sparse_qk_precompressed_ms"])
    assert math.isfinite(summary["dense_fwd_ms"])
    assert math.isfinite(summary["sparse_fwd_precompressed_ms"])
    assert summary["output_max_diff"] >= 0.0


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
@pytest.mark.parametrize("case_name", ["mixed-small", "train-eq", "train-gqa"])
def test_hsa_shared_sparse_gemm_forward_summary_runs_on_benchmark_cases(monkeypatch, case_name):
    benchmark_hsa = _load_benchmark_hsa_module()
    case = next(candidate for candidate in benchmark_hsa.ALL_CASES if candidate.name == case_name)
    schedule, _keep_ids, _hash_ids, q, k, v = benchmark_hsa._build_case_tensors(case)

    monkeypatch.setenv("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_Q", "2")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_K", "2")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MAX_PACKED_K", "64")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MAX_DIRECT_SEGMENTS", "1")

    report = analyze_hsa_shared_sparse_gemm_forward(
        schedule,
        q,
        k,
        v,
        use_synthetic_grid=True,
        warmup_iters=0,
        benchmark_iters=1,
        max_buckets=4,
        max_qgroups=16,
    )
    summary = summarize_hsa_shared_sparse_gemm_forward(report)

    assert summary["status"] == "measured"
    assert summary["bucket_count"] > 0
    assert summary["sampled_qgroups"] > 0
    assert math.isfinite(summary["dense_qk_ms"])
    assert math.isfinite(summary["sparse_main_qk_ms"])
    assert math.isfinite(summary["sparse_exact_qk_ms"])
    assert math.isfinite(summary["dense_fwd_ms"])
    assert math.isfinite(summary["sparse_main_fwd_ms"])
    assert math.isfinite(summary["sparse_exact_fwd_ms"])
    assert summary["exact_output_max_diff"] >= 0.0


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_benchmark_hsa_shared_sparse_gemm_summary_is_read_only(monkeypatch):
    benchmark_hsa = _load_benchmark_hsa_module()

    device = "cuda"
    batch_size, seqlen, nheads, headdim = 1, 65, 4, 64
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)

    monkeypatch.setenv("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_Q", "2")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_K", "2")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MAX_PACKED_K", "64")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MAX_DIRECT_SEGMENTS", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_SHARED_SPARSE_GEMM_WARMUP_ITERS", "0")
    monkeypatch.setenv("FLASH_ATTN_HSA_SHARED_SPARSE_GEMM_BENCH_ITERS", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_SHARED_SPARSE_GEMM_MAX_BUCKETS", "4")
    monkeypatch.setenv("FLASH_ATTN_HSA_SHARED_SPARSE_GEMM_MAX_QGROUPS", "16")

    env_updates = {
        "FLASH_ATTN_HSA_USE_SYNTHETIC_GRID": "1",
        "FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_Q": "2",
        "FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_K": "2",
        "FLASH_ATTN_HSA_SYNTHETIC_MAX_PACKED_K": "64",
        "FLASH_ATTN_HSA_SYNTHETIC_MAX_DIRECT_SEGMENTS": "1",
    }

    out_before = benchmark_hsa._run_sparse_attention(
        q,
        k,
        v,
        keep_ids,
        hash_ids,
        schedule,
        env_updates=env_updates,
    )
    summary = benchmark_hsa._get_shared_sparse_gemm_forward_summary(
        schedule,
        q,
        k,
        v,
        use_synthetic_grid=True,
    )
    out_after = benchmark_hsa._run_sparse_attention(
        q,
        k,
        v,
        keep_ids,
        hash_ids,
        schedule,
        env_updates=env_updates,
    )

    assert summary["status"] == "measured"
    assert summary["bucket_count"] > 0
    _assert_close(out_before.float(), out_after.float(), "benchmark_shared_sparse_gemm_summary_read_only")


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_hsa_sparse24_analysis_summary_runs_on_runtime_metadata(monkeypatch):
    device = "cuda"
    batch_size, seqlen, nheads, headdim = 1, 65, 4, 64
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)

    monkeypatch.setenv("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_Q", "2")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_K", "2")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MAX_PACKED_K", "128")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MAX_DIRECT_SEGMENTS", "4")

    report = analyze_hsa_sparse24_feasibility(schedule, q, k, use_synthetic_grid=True)
    summary = summarize_hsa_sparse24_feasibility(report)

    assert report["physical"]["summary"]["segments"] > 0
    assert summary["physical_segments"] > 0
    assert summary["synthetic_segments"] > 0
    assert summary["synthetic_direct_segments"] >= 0
    for value in summary.values():
        if isinstance(value, float):
            assert math.isfinite(value)


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_benchmark_hsa_sparse24_summary_is_read_only(monkeypatch):
    benchmark_hsa = _load_benchmark_hsa_module()

    device = "cuda"
    batch_size, seqlen, nheads, headdim = 1, 65, 4, 64
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)

    env_updates: dict[str, str] = {}
    out_before = benchmark_hsa._run_sparse_attention(
        q,
        k,
        v,
        keep_ids,
        hash_ids,
        schedule,
        env_updates=env_updates,
    )
    summary = benchmark_hsa._get_sparse24_feasibility_summary(schedule, q, k, use_synthetic_grid=False)
    out_after = benchmark_hsa._run_sparse_attention(
        q,
        k,
        v,
        keep_ids,
        hash_ids,
        schedule,
        env_updates=env_updates,
    )

    assert summary["physical_segments"] > 0
    assert summary["synthetic_segments"] == 0
    _assert_close(out_before.float(), out_after.float(), "benchmark_sparse24_summary_read_only")


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_hsa_sparse24_runtime_bitmap_partial_tile_is_ineligible():
    device = "cuda"
    batch_size, seqlen, nheads, headdim = 1, 65, 4, 64
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)

    report = analyze_hsa_sparse24_feasibility(schedule, q, k, use_synthetic_grid=False)
    bitmap_segments = [
        segment
        for segment in report["physical"]["segments"]
        if segment["segment_kind"] == "partial_tile" and segment["mask_kind"] == "bitmap"
    ]

    assert bitmap_segments
    assert any(
        "row_dependent_columns" in segment["failure_reasons"] and not segment["eligible"]
        for segment in bitmap_segments
    )


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_hsa_synthetic_grid_runtime_honors_env_geometry(monkeypatch):
    import flash_attn.cute.hsa as hsa_module

    device = "cuda"
    batch_size, seqlen, nheads, headdim = 1, 65, 4, 64
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)

    monkeypatch.setenv("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_Q", "2")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_K", "128")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MAX_PACKED_K", "128")
    runtime = hsa_module._get_hsa_block_sparse_runtime(schedule, q, k)
    hsa_module._ensure_hsa_synthetic_grid_metadata(schedule, runtime)

    assert runtime.forward_synthetic_grid is not None
    assert runtime.synthetic_grid is runtime.forward_synthetic_grid
    assert runtime.forward_synthetic_grid.logical_block_q == 2
    assert runtime.forward_synthetic_grid.logical_block_k == 128
    assert runtime.forward_synthetic_grid.max_packed_k == 128
    assert runtime.forward_synthetic_grid.num_tiles > 0
    assert runtime.forward_synthetic_grid.forward_execution_plan is not None
    assert runtime.forward_synthetic_grid.forward_execution_plan["qgroup_bucket_packed_q"]


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
@pytest.mark.parametrize(
    "batch_size,seqlen,nheads,headdim",
    [
        (1, 65, 4, 64),
        (2, 1024, 8, 64),
    ],
)
def test_hsa_synthetic_grid_2x2_collapses_to_one_bucket(monkeypatch, batch_size, seqlen, nheads, headdim):
    import flash_attn.cute.hsa as hsa_module

    device = "cuda"
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)

    monkeypatch.setenv("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_Q", "2")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_K", "2")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MAX_PACKED_K", "128")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MAX_DIRECT_SEGMENTS", "1")
    monkeypatch.delenv("FLASH_ATTN_HSA_SYNTHETIC_PACKED_K_BIN", raising=False)
    runtime = hsa_module._get_hsa_block_sparse_runtime(schedule, q, k)
    hsa_module._ensure_hsa_synthetic_grid_metadata(schedule, runtime)

    metadata = runtime.forward_synthetic_grid
    assert metadata is not None
    plan = metadata.forward_execution_plan
    assert plan is not None
    assert len(plan["bucket_packed_k"]) == 1
    assert plan["bucket_use_qgroup_q"] == [True]
    assert plan["bucket_scatter_only"] == [True]
    direct_plan = plan["direct_execution_plan"]
    assert direct_plan is not None
    assert direct_plan["max_direct_segments"] == 1
    assert all(num_segments == 1 for num_segments in direct_plan["qgroup_bucket_num_segments"])
    row_plan = direct_plan["row_compact_plan"]
    assert row_plan is not None
    assert max(row_plan["bucket_row_k_cap"]) <= int(row_plan["row_k_cap_limit"])
    assert row_plan["bucket_row_k_length"].numel() > 0
    assert row_plan["bucket_union_to_row_slot"].numel() > 0


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
@pytest.mark.parametrize(
    "batch_size,seqlen,nheads,headdim",
    [
        (1, 65, 4, 64),
        (2, 1024, 8, 64),
    ],
)
def test_hsa_synthetic_grid_2x2_builds_segmented_direct_plan(monkeypatch, batch_size, seqlen, nheads, headdim):
    import flash_attn.cute.hsa as hsa_module

    device = "cuda"
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)

    monkeypatch.setenv("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_Q", "2")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_K", "2")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MAX_PACKED_K", "128")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MAX_DIRECT_SEGMENTS", "4")
    runtime = hsa_module._get_hsa_block_sparse_runtime(schedule, q, k)
    hsa_module._ensure_hsa_synthetic_grid_metadata(schedule, runtime)

    metadata = runtime.forward_synthetic_grid
    assert metadata is not None
    plan = metadata.forward_execution_plan
    assert plan is not None
    direct_plan = plan["direct_execution_plan"]
    assert direct_plan is not None
    assert direct_plan["max_direct_segments"] == 4
    assert len(direct_plan["bucket_packed_k"]) >= 1
    assert all(1 <= num_segments <= 4 for num_segments in direct_plan["qgroup_bucket_num_segments"])
    assert any(num_segments > 1 for num_segments in direct_plan["qgroup_bucket_num_segments"])
    assert direct_plan["row_compact_plan"] is None


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
@pytest.mark.parametrize("logical_block_k", [2, 4])
def test_hsa_synthetic_grid_segmented_sparse_parse_flag_rebuilds_row_compact_plan(monkeypatch, logical_block_k):
    import flash_attn.cute.hsa as hsa_module

    device = "cuda"
    batch_size, seqlen, nheads, headdim = 1, 65, 4, 64
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)

    monkeypatch.setenv("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_Q", "2")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_K", str(logical_block_k))
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MAX_PACKED_K", "128")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MAX_DIRECT_SEGMENTS", "4")
    monkeypatch.delenv("FLASH_ATTN_HSA_SYNTHETIC_PARSE_SPARSE_FWD", raising=False)

    runtime = hsa_module._get_hsa_block_sparse_runtime(schedule, q, k)
    hsa_module._ensure_hsa_synthetic_grid_metadata(schedule, runtime)

    metadata = runtime.forward_synthetic_grid
    assert metadata is not None
    direct_plan = metadata.forward_execution_plan["direct_execution_plan"]
    assert direct_plan is not None
    assert metadata.sparse_parse_fwd is False
    assert direct_plan["row_compact_plan"] is None

    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_PARSE_SPARSE_FWD", "1")
    hsa_module._ensure_hsa_synthetic_grid_metadata(schedule, runtime)

    metadata = runtime.forward_synthetic_grid
    assert metadata is not None
    direct_plan = metadata.forward_execution_plan["direct_execution_plan"]
    assert direct_plan is not None
    assert metadata.sparse_parse_fwd is True
    row_plan = direct_plan["row_compact_plan"]
    assert row_plan is not None
    if logical_block_k > 2:
        assert direct_plan["union_row_compact_plan"] is not None
    else:
        assert direct_plan["union_row_compact_plan"] is None
    assert max(direct_plan["qgroup_bucket_num_segments"]) > 1
    assert max(row_plan["bucket_row_k_cap"]) <= int(row_plan["row_k_cap_limit"])
    assert row_plan["bucket_row_k_length"].numel() > 0


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
@pytest.mark.parametrize(
    "label,batch_size,seqlen,nheads,headdim,n_kv_heads",
    [
        ("mixed_small_eq", 1, 65, 4, 64, 4),
        ("train_eq", 2, 1024, 8, 64, 8),
    ],
)
def test_hsa_synthetic_grid_matches_sparse_path(
    monkeypatch,
    label,
    batch_size,
    seqlen,
    nheads,
    headdim,
    n_kv_heads,
):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    device = "cuda"
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    dtype = torch.bfloat16
    q_data = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    k_data = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)
    v_data = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)
    monkeypatch.delenv("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", raising=False)
    baseline = _run_mixed_sparse_mask_case(
        batch_size=batch_size,
        seqlen=seqlen,
        nheads=nheads,
        headdim=headdim,
        n_kv_heads=n_kv_heads,
        keep_ids=keep_ids,
        hash_ids=hash_ids,
        q_data=q_data,
        k_data=k_data,
        v_data=v_data,
    )
    out_base, _, _, _, out_ref, _, _, _ = baseline

    calls = {"fwd": 0, "bwd": 0}
    original_fwd = synthetic_module.run_hsa_fwd_sm100_synthetic_grid

    def _tracked_fwd(*args, **kwargs):
        calls["fwd"] += 1
        return original_fwd(*args, **kwargs)

    def _tracked_bwd(*args, **kwargs):
        calls["bwd"] += 1
        raise AssertionError("synthetic-grid backward should stay on sparse-mask fallback")

    monkeypatch.setattr(synthetic_module, "run_hsa_fwd_sm100_synthetic_grid", _tracked_fwd)
    monkeypatch.setattr(synthetic_module, "run_hsa_bwd_sm100_synthetic_grid", _tracked_bwd)
    monkeypatch.setenv("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "1")
    out_synth, q_synth, k_synth, v_synth, _, _, _, _ = _run_mixed_sparse_mask_case(
        batch_size=batch_size,
        seqlen=seqlen,
        nheads=nheads,
        headdim=headdim,
        n_kv_heads=n_kv_heads,
        keep_ids=keep_ids,
        hash_ids=hash_ids,
        q_data=q_data,
        k_data=k_data,
        v_data=v_data,
    )

    assert calls["fwd"] == 1
    assert calls["bwd"] == 0
    _assert_close(out_synth.float(), out_base.float(), f"{label}_synthetic_grid_output_vs_sparse")
    _assert_close(out_synth.float(), out_ref.float(), f"{label}_synthetic_grid_output_vs_ref")
    _assert_finite_gradients(f"{label}_synthetic_grid_grads", q_synth, k_synth, v_synth)


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
@pytest.mark.parametrize("logical_block_k", [2, 4])
def test_hsa_synthetic_grid_segmented_sparse_parse_forward_matches_sparse_path(monkeypatch, logical_block_k):
    import flash_attn.cute.hsa as hsa_module
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    label = f"mixed_small_segmented_sparse_parse_2x{logical_block_k}"
    batch_size, seqlen, nheads, headdim, n_kv_heads = 1, 65, 4, 64, 4
    device = "cuda"
    dtype = torch.bfloat16
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    q_data = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    k_data = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)
    v_data = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)

    monkeypatch.delenv("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", raising=False)
    baseline = _run_mixed_sparse_mask_case(
        batch_size=batch_size,
        seqlen=seqlen,
        nheads=nheads,
        headdim=headdim,
        n_kv_heads=n_kv_heads,
        keep_ids=keep_ids,
        hash_ids=hash_ids,
        q_data=q_data,
        k_data=k_data,
        v_data=v_data,
    )
    out_base, _, _, _, out_ref, _, _, _ = baseline

    calls = {"row_compact": 0, "combine": 0}
    original_row_compact = synthetic_module._run_synthetic_direct_row_micro_fwd_kernel
    original_combine = synthetic_module._run_synthetic_direct_combine_rows_kernel

    def _tracked_row_compact(*args, **kwargs):
        calls["row_compact"] += 1
        return original_row_compact(*args, **kwargs)

    def _tracked_combine(*args, **kwargs):
        calls["combine"] += 1
        return original_combine(*args, **kwargs)

    _tracked_row_compact.compile_cache = original_row_compact.compile_cache
    _tracked_combine.compile_cache = original_combine.compile_cache
    monkeypatch.setattr(synthetic_module, "_run_synthetic_direct_row_micro_fwd_kernel", _tracked_row_compact)
    monkeypatch.setattr(synthetic_module, "_run_synthetic_direct_combine_rows_kernel", _tracked_combine)
    monkeypatch.setenv("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_Q", "2")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_K", str(logical_block_k))
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MAX_PACKED_K", "128")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MAX_DIRECT_SEGMENTS", "4")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_PARSE_SPARSE_FWD", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MICRO_FWD", "1")
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    runtime = hsa_module._get_hsa_block_sparse_runtime(schedule, q_data, k_data)
    hsa_module._ensure_hsa_synthetic_grid_metadata(schedule, runtime)
    metadata = runtime.forward_synthetic_grid
    assert metadata is not None
    direct_plan = metadata.forward_execution_plan["direct_execution_plan"]
    assert direct_plan is not None
    out_sparse_parse, q_parse, k_parse, v_parse, _, _, _, _ = _run_mixed_sparse_mask_case(
        batch_size=batch_size,
        seqlen=seqlen,
        nheads=nheads,
        headdim=headdim,
        n_kv_heads=n_kv_heads,
        keep_ids=keep_ids,
        hash_ids=hash_ids,
        q_data=q_data,
        k_data=k_data,
        v_data=v_data,
    )

    assert calls["row_compact"] > 0
    if logical_block_k > 2:
        assert direct_plan["union_row_compact_plan"] is not None
        assert calls["combine"] == 0
    else:
        assert direct_plan["union_row_compact_plan"] is None
        assert calls["combine"] > 0
    _assert_close(out_sparse_parse.float(), out_base.float(), f"{label}_output_vs_sparse")
    _assert_close(out_sparse_parse.float(), out_ref.float(), f"{label}_output_vs_ref")
    _assert_finite_gradients(f"{label}_grads", q_parse, k_parse, v_parse)


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_hsa_synthetic_grid_segmented_sparse_parse_packed_kv_forward_matches_sparse_path(monkeypatch):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    label = "mixed_small_segmented_sparse_parse_2x4_packed_kv"
    batch_size, seqlen, nheads, headdim, n_kv_heads = 1, 65, 4, 64, 4
    device = "cuda"
    dtype = torch.bfloat16
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    q_data = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    k_data = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)
    v_data = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)

    monkeypatch.delenv("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", raising=False)
    baseline = _run_mixed_sparse_mask_case(
        batch_size=batch_size,
        seqlen=seqlen,
        nheads=nheads,
        headdim=headdim,
        n_kv_heads=n_kv_heads,
        keep_ids=keep_ids,
        hash_ids=hash_ids,
        q_data=q_data,
        k_data=k_data,
        v_data=v_data,
    )
    out_base, _, _, _, out_ref, _, _, _ = baseline

    calls = {"packed_kv": 0, "combine": 0}
    original_packed_kv = synthetic_module._run_synthetic_direct_row_micro_fwd_packed_kv_kernel
    original_combine = synthetic_module._run_synthetic_direct_combine_rows_kernel

    def _tracked_packed_kv(*args, **kwargs):
        calls["packed_kv"] += 1
        return original_packed_kv(*args, **kwargs)

    def _tracked_combine(*args, **kwargs):
        calls["combine"] += 1
        return original_combine(*args, **kwargs)

    _tracked_packed_kv.compile_cache = original_packed_kv.compile_cache
    _tracked_combine.compile_cache = original_combine.compile_cache
    monkeypatch.setattr(synthetic_module, "_run_synthetic_direct_row_micro_fwd_packed_kv_kernel", _tracked_packed_kv)
    monkeypatch.setattr(synthetic_module, "_run_synthetic_direct_combine_rows_kernel", _tracked_combine)
    monkeypatch.setenv("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_Q", "2")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_K", "4")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MAX_PACKED_K", "128")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MAX_DIRECT_SEGMENTS", "4")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_PARSE_SPARSE_FWD", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_PARSE_SPARSE_PACKED_KV_FWD", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MICRO_FWD", "1")
    out_sparse_parse, q_parse, k_parse, v_parse, _, _, _, _ = _run_mixed_sparse_mask_case(
        batch_size=batch_size,
        seqlen=seqlen,
        nheads=nheads,
        headdim=headdim,
        n_kv_heads=n_kv_heads,
        keep_ids=keep_ids,
        hash_ids=hash_ids,
        q_data=q_data,
        k_data=k_data,
        v_data=v_data,
    )

    assert calls["packed_kv"] > 0
    assert calls["combine"] == 0
    _assert_close(out_sparse_parse.float(), out_base.float(), f"{label}_output_vs_sparse")
    _assert_close(out_sparse_parse.float(), out_ref.float(), f"{label}_output_vs_ref")
    _assert_finite_gradients(f"{label}_grads", q_parse, k_parse, v_parse)


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
@pytest.mark.parametrize(
    "label,batch_size,seqlen,nheads,headdim,n_kv_heads",
    [
        ("mixed_small_eq", 1, 65, 4, 64, 4),
        ("train_eq", 2, 1024, 8, 64, 8),
    ],
)
def test_hsa_synthetic_micro_forward_matches_reference_and_has_finite_grads(
    monkeypatch,
    label,
    batch_size,
    seqlen,
    nheads,
    headdim,
    n_kv_heads,
):
    device = "cuda"
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    dtype = torch.bfloat16
    q_data = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    k_data = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)
    v_data = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)

    monkeypatch.setenv("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_Q", "2")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_K", "2")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MAX_PACKED_K", "128")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MICRO_FWD", "1")
    out_micro, q_micro, k_micro, v_micro, out_ref, _, _, _ = _run_mixed_sparse_mask_case(
        batch_size=batch_size,
        seqlen=seqlen,
        nheads=nheads,
        headdim=headdim,
        n_kv_heads=n_kv_heads,
        keep_ids=keep_ids,
        hash_ids=hash_ids,
        q_data=q_data,
        k_data=k_data,
        v_data=v_data,
    )

    _assert_close(out_micro.float(), out_ref.float(), f"{label}_synthetic_micro_output_vs_ref")
    _assert_finite_gradients(f"{label}_synthetic_micro_grads", q_micro, k_micro, v_micro)


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_hsa_synthetic_micro_saved_tensor_mix_has_finite_grads(monkeypatch):
    batch_size, seqlen, nheads, headdim, n_kv_heads = 2, 1024, 8, 64, 8
    device = "cuda"
    dtype = torch.bfloat16
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    q_data = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    k_data = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)
    v_data = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)

    base_env = {
        "FLASH_ATTN_HSA_USE_SYNTHETIC_GRID": "1",
        "FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_Q": "2",
        "FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_K": "2",
        "FLASH_ATTN_HSA_SYNTHETIC_MAX_PACKED_K": "128",
    }

    with monkeypatch.context() as local_patch:
        for key, value in base_env.items():
            local_patch.setenv(key, value)
        local_patch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MICRO_FWD", "0")
        stable_out, stable_lse = _capture_synthetic_forward_saved_tensors(
            q_data=q_data,
            k_data=k_data,
            v_data=v_data,
            keep_ids=keep_ids,
            hash_ids=hash_ids,
        )

    with monkeypatch.context() as local_patch:
        for key, value in base_env.items():
            local_patch.setenv(key, value)
        local_patch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MICRO_FWD", "1")
        micro_out, micro_lse = _capture_synthetic_forward_saved_tensors(
            q_data=q_data,
            k_data=k_data,
            v_data=v_data,
            keep_ids=keep_ids,
            hash_ids=hash_ids,
        )

    combos = {
        "stable_out_stable_lse": (stable_out, stable_lse),
        "micro_out_micro_lse": (micro_out, micro_lse),
        "micro_out_stable_lse": (micro_out, stable_lse),
        "stable_out_micro_lse": (stable_out, micro_lse),
    }
    for name, (saved_out, saved_lse) in combos.items():
        out_mix, q_mix, k_mix, v_mix, _, _, _, _ = _run_synthetic_case_with_saved_tensor_override(
            monkeypatch,
            batch_size=batch_size,
            seqlen=seqlen,
            nheads=nheads,
            headdim=headdim,
            n_kv_heads=n_kv_heads,
            keep_ids=keep_ids,
            hash_ids=hash_ids,
            q_data=q_data,
            k_data=k_data,
            v_data=v_data,
            saved_out=saved_out,
            saved_lse=saved_lse,
        )
        _assert_close(out_mix.float(), saved_out.float(), f"{name}_saved_output")
        _assert_finite_gradients(f"{name}_saved_grads", q_mix, k_mix, v_mix)


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_train_eq_synthetic_micro_forward_finite_gradient_anchors(monkeypatch):
    benchmark_hsa = _load_benchmark_hsa_module()
    case = next(candidate for candidate in benchmark_hsa.ALL_CASES if candidate.name == "train-eq")

    stable_synth_env = dict(benchmark_hsa._benchmark_env_for_case(case))
    stable_synth_env.update(
        {
            "FLASH_ATTN_HSA_USE_SYNTHETIC_GRID": "1",
            "FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_Q": "2",
            "FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_K": "2",
            "FLASH_ATTN_HSA_SYNTHETIC_MAX_PACKED_K": "128",
            "FLASH_ATTN_HSA_SYNTHETIC_MAX_DIRECT_SEGMENTS": "1",
            "FLASH_ATTN_HSA_SYNTHETIC_MICRO_FWD": "0",
            "FLASH_ATTN_HSA_SYNTHETIC_MICRO_BWD": "0",
            "FLASH_ATTN_HSA_SYNTHETIC_SHORT_BWD": "off",
            "FLASH_ATTN_HSA_SYNTHETIC_FUSED_BWD": "off",
            "FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD": "off",
        }
    )

    plain_out, plain_q, plain_k, plain_v = _run_benchmark_case_env(
        "train-eq",
        benchmark_hsa._plain_sparse_mask_baseline_env(case),
    )
    _assert_gradient_finite_status("train_eq_plain_sparse_mask_q_grad", plain_q)
    _assert_finite_gradients("train_eq_plain_sparse_mask_grads", plain_q, plain_k, plain_v)

    stable_out, stable_q, stable_k, stable_v = _run_benchmark_case_env("train-eq", stable_synth_env)
    _assert_gradient_finite_status("train_eq_stable_synthetic_forward_q_grad", stable_q)
    _assert_finite_gradients("train_eq_stable_synthetic_forward_grads", stable_q, stable_k, stable_v)

    micro_out, micro_q, micro_k, micro_v = _run_benchmark_case_env(
        "train-eq",
        benchmark_hsa._sparse_mask_mixed_backward_baseline_env(case),
    )
    _assert_gradient_finite_status("train_eq_micro_synthetic_forward_q_grad", micro_q)
    _assert_finite_gradients("train_eq_micro_synthetic_forward_grads", micro_q, micro_k, micro_v)

    micro_bwd_out, micro_bwd_q, micro_bwd_k, micro_bwd_v = _run_benchmark_case_env(
        "train-eq",
        benchmark_hsa._one_kernel_synthetic_short_env(case),
    )
    _assert_gradient_finite_status("train_eq_micro_synthetic_one_kernel_q_grad", micro_bwd_q)
    _assert_finite_gradients(
        "train_eq_micro_synthetic_one_kernel_grads",
        micro_bwd_q,
        micro_bwd_k,
        micro_bwd_v,
    )

    _assert_close(plain_out.float(), stable_out.float(), "train_eq_stable_synthetic_vs_plain_output", atol=2e-2, rtol=2e-2)
    _assert_close(plain_out.float(), micro_out.float(), "train_eq_micro_synthetic_vs_plain_output", atol=2e-2, rtol=2e-2)
    _assert_close(plain_out.float(), micro_bwd_out.float(), "train_eq_micro_synthetic_one_kernel_vs_plain_output", atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_hsa_synthetic_micro_backward_routes_off_sparse_mask(monkeypatch):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module
    import flash_attn.cute.hsa as hsa_module

    batch_size, seqlen, nheads, headdim, n_kv_heads = 1, 65, 4, 64, 4
    device = "cuda"
    dtype = torch.bfloat16
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    q_data = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    k_data = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)
    v_data = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)

    calls = {"bwd": 0}
    original_bwd = synthetic_module.run_hsa_bwd_sm100_synthetic_grid

    def _tracked_bwd(*args, **kwargs):
        calls["bwd"] += 1
        return original_bwd(*args, **kwargs)

    monkeypatch.setattr(synthetic_module, "run_hsa_bwd_sm100_synthetic_grid", _tracked_bwd)
    monkeypatch.setattr(
        hsa_module,
        "_run_hsa_packed_mask_backward",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("packed-mask backward used")),
    )
    monkeypatch.setenv("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_Q", "2")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_K", "2")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MAX_PACKED_K", "128")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MICRO_FWD", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MICRO_BWD", "1")

    out_micro, q_micro, k_micro, v_micro, out_ref, _, _, _ = _run_mixed_sparse_mask_case(
        batch_size=batch_size,
        seqlen=seqlen,
        nheads=nheads,
        headdim=headdim,
        n_kv_heads=n_kv_heads,
        keep_ids=keep_ids,
        hash_ids=hash_ids,
        q_data=q_data,
        k_data=k_data,
        v_data=v_data,
    )

    assert calls["bwd"] == 1
    _assert_close(out_micro.float(), out_ref.float(), "mixed_small_eq_synthetic_micro_bwd_output_vs_ref")
    _assert_finite_gradients("mixed_small_eq_synthetic_micro_bwd_grads", q_micro, k_micro, v_micro)


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
@pytest.mark.parametrize(
    "label,batch_size,seqlen,nheads,headdim,n_kv_heads",
    [
        ("mixed_small_eq", 1, 65, 4, 64, 4),
        ("train_eq", 2, 1024, 8, 64, 8),
        ("longer_eq", 2, 2048, 8, 64, 8),
    ],
)
def test_hsa_synthetic_micro_forward_backward_matches_reference_and_has_finite_grads(
    monkeypatch,
    label,
    batch_size,
    seqlen,
    nheads,
    headdim,
    n_kv_heads,
):
    device = "cuda"
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    dtype = torch.bfloat16
    q_data = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    k_data = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)
    v_data = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)

    monkeypatch.setenv("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_Q", "2")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_K", "2")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MAX_PACKED_K", "128")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MICRO_FWD", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MICRO_BWD", "1")
    out_micro, q_micro, k_micro, v_micro, out_ref, q_ref_grad, k_ref_grad, v_ref_grad = _run_mixed_sparse_mask_case(
        batch_size=batch_size,
        seqlen=seqlen,
        nheads=nheads,
        headdim=headdim,
        n_kv_heads=n_kv_heads,
        keep_ids=keep_ids,
        hash_ids=hash_ids,
        q_data=q_data,
        k_data=k_data,
        v_data=v_data,
    )

    _assert_close(out_micro.float(), out_ref.float(), f"{label}_synthetic_micro_fwd_bwd_output_vs_ref")
    _assert_finite_gradients(f"{label}_synthetic_micro_fwd_bwd_grads", q_micro, k_micro, v_micro)
    _assert_close(q_micro.float(), q_ref_grad, f"{label}_synthetic_micro_fwd_bwd_q_grad", atol=6e-2, rtol=6e-2)
    _assert_close(k_micro.float(), k_ref_grad, f"{label}_synthetic_micro_fwd_bwd_k_grad", atol=6e-2, rtol=6e-2)
    _assert_close(v_micro.float(), v_ref_grad, f"{label}_synthetic_micro_fwd_bwd_v_grad", atol=6e-2, rtol=6e-2)


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_hsa_synthetic_micro_backward_row_compact_falls_back_when_headdim_unsupported(monkeypatch):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    device = "cuda"
    batch_size, seqlen, nheads, headdim, n_kv_heads = 1, 65, 4, 128, 4
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    dtype = torch.bfloat16
    q_data = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    k_data = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)
    v_data = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)

    calls = {"dense": 0, "masked": 0}
    original_dense = synthetic_module._run_synthetic_direct_micro_bwd_dense_kernel
    original_masked = synthetic_module._run_synthetic_direct_micro_bwd_masked_kernel

    def _tracked_dense(*args, **kwargs):
        calls["dense"] += 1
        return original_dense(*args, **kwargs)

    def _tracked_masked(*args, **kwargs):
        calls["masked"] += 1
        return original_masked(*args, **kwargs)

    monkeypatch.setattr(
        synthetic_module,
        "_run_synthetic_direct_row_micro_bwd_kernel",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("row-compact backward used unexpectedly")),
    )
    monkeypatch.setattr(synthetic_module, "_run_synthetic_direct_micro_bwd_dense_kernel", _tracked_dense)
    monkeypatch.setattr(synthetic_module, "_run_synthetic_direct_micro_bwd_masked_kernel", _tracked_masked)
    monkeypatch.setenv("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_Q", "2")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_K", "2")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MAX_PACKED_K", "128")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MICRO_FWD", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MICRO_BWD", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MAX_DIRECT_SEGMENTS", "1")

    out_micro, q_micro, k_micro, v_micro, out_ref, _, _, _ = _run_mixed_sparse_mask_case(
        batch_size=batch_size,
        seqlen=seqlen,
        nheads=nheads,
        headdim=headdim,
        n_kv_heads=n_kv_heads,
        keep_ids=keep_ids,
        hash_ids=hash_ids,
        q_data=q_data,
        k_data=k_data,
        v_data=v_data,
    )

    assert calls["dense"] + calls["masked"] > 0
    _assert_close(out_micro.float(), out_ref.float(), "mixed_small_headdim128_synthetic_micro_output_vs_ref")
    _assert_finite_gradients("mixed_small_headdim128_synthetic_micro_grads", q_micro, k_micro, v_micro)


def test_hsa_synthetic_micro_backward_selector_honors_accum_mode(monkeypatch):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    calls = []

    def _row_local(*args, **kwargs):
        calls.append("row_local")

    def _union_local(*args, **kwargs):
        calls.append("union_local")

    monkeypatch.setattr(synthetic_module, "_run_synthetic_direct_row_micro_bwd_kernel_row_local", _row_local)
    monkeypatch.setattr(synthetic_module, "_run_synthetic_direct_row_micro_bwd_kernel_union_local", _union_local)

    t = torch.empty(1)
    args = (t,) * 17

    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_FUSED_BWD", "off")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ROW_BWD_ACCUM_MODE", "row_local")
    synthetic_module._run_synthetic_direct_row_micro_bwd_kernel(*args, softmax_scale=1.0)
    assert calls == ["row_local"]

    calls.clear()
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ROW_BWD_ACCUM_MODE", "union_local")
    synthetic_module._run_synthetic_direct_row_micro_bwd_kernel(*args, softmax_scale=1.0)
    assert calls == ["union_local"]


def test_hsa_synthetic_micro_backward_selector_honors_short_mode(monkeypatch):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    calls = []

    def _short(*args, **kwargs):
        calls.append("short")

    def _row_local(*args, **kwargs):
        calls.append("row_local")

    monkeypatch.setattr(synthetic_module, "_run_synthetic_direct_row_micro_bwd_kernel_short", _short)
    monkeypatch.setattr(synthetic_module, "_run_synthetic_direct_row_micro_bwd_kernel_row_local", _row_local)

    t = torch.empty(1)
    args = (t,) * 17

    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_FUSED_BWD", "off")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ROW_BWD_ACCUM_MODE", "row_local")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_SHORT_BWD", "on")
    synthetic_module._run_synthetic_direct_row_micro_bwd_kernel(*args, softmax_scale=1.0)
    assert calls == ["short"]

    calls.clear()
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_SHORT_BWD", "off")
    synthetic_module._run_synthetic_direct_row_micro_bwd_kernel(*args, softmax_scale=1.0)
    assert calls == ["row_local"]


def test_hsa_synthetic_micro_backward_selector_honors_fused_mode(monkeypatch):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    calls = []

    def _fused(*args, **kwargs):
        calls.append("fused")

    def _row_local(*args, **kwargs):
        calls.append("row_local")

    monkeypatch.setattr(synthetic_module, "_run_synthetic_direct_row_micro_bwd_kernel_fused", _fused)
    monkeypatch.setattr(synthetic_module, "_run_synthetic_direct_row_micro_bwd_kernel_row_local", _row_local)

    q_rows = torch.empty((1, 1, 64))
    k_rows = torch.empty((1, 1, 64))
    v_rows = torch.empty((1, 1, 64))
    out_rows = torch.empty((1, 1, 64))
    dout_rows = torch.empty((1, 1, 64))
    lse_rows = torch.empty((1, 1))
    q_row_idx = torch.empty((1, 2), dtype=torch.int32)
    row_k_row_idx = torch.empty((1, 2, 1), dtype=torch.int32)
    union_k_row_idx = torch.empty((1, 8), dtype=torch.int32)
    row_k_to_union_idx = torch.empty((1, 2, 1), dtype=torch.int32)
    union_to_row_slot = torch.empty((1, 2, 8), dtype=torch.int32)
    q_length = torch.empty((1,), dtype=torch.int32)
    row_k_length = torch.empty((1, 2), dtype=torch.int32)
    union_k_length = torch.empty((1,), dtype=torch.int32)
    dq_rows = torch.empty((1, 1, 64))
    dk_rows = torch.empty((1, 1, 64))
    dv_rows = torch.empty((1, 1, 64))
    args = (
        q_rows,
        k_rows,
        v_rows,
        out_rows,
        dout_rows,
        lse_rows,
        q_row_idx,
        row_k_row_idx,
        union_k_row_idx,
        row_k_to_union_idx,
        union_to_row_slot,
        q_length,
        row_k_length,
        union_k_length,
        dq_rows,
        dk_rows,
        dv_rows,
    )

    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_FUSED_BWD", "on")
    synthetic_module._run_synthetic_direct_row_micro_bwd_kernel(*args, softmax_scale=1.0)
    assert calls == ["fused"]

    calls.clear()
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_FUSED_BWD", "off")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ROW_BWD_ACCUM_MODE", "row_local")
    synthetic_module._run_synthetic_direct_row_micro_bwd_kernel(*args, softmax_scale=1.0)
    assert calls == ["row_local"]


def test_hsa_synthetic_row_compact_backward_selector_honors_one_kernel_mode(monkeypatch):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    q_rows = torch.empty((1, 1, 64))
    union_k_row_idx = torch.empty((8, 12), dtype=torch.int32)
    q_row_idx = torch.empty((8, 2), dtype=torch.int32)
    unique_key_row_idx = torch.empty((0,), dtype=torch.int32)

    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD", "on")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT", "auto")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_SPLIT_BWD", "on")
    mode = synthetic_module._select_row_compact_synthetic_bwd_mode(
        q_rows,
        union_k_row_idx,
        q_row_idx,
        unique_key_row_idx,
        4,
    )
    assert mode == "one_kernel"

    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD", "off")
    mode = synthetic_module._select_row_compact_synthetic_bwd_mode(
        q_rows,
        union_k_row_idx,
        q_row_idx,
        unique_key_row_idx,
        4,
    )
    assert mode == "split"


def test_hsa_synthetic_row_compact_backward_selector_falls_back_when_one_kernel_unsupported(monkeypatch):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    q_rows = torch.empty((1, 1, 64))
    q_row_idx = torch.empty((8, 2), dtype=torch.int32)
    unique_key_row_idx = torch.empty((0,), dtype=torch.int32)

    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD", "auto")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT", "auto")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_SPLIT_BWD", "off")
    unsupported_union = torch.empty((8, 17), dtype=torch.int32)
    mode = synthetic_module._select_row_compact_synthetic_bwd_mode(
        q_rows,
        unsupported_union,
        q_row_idx,
        unique_key_row_idx,
        4,
    )
    assert mode == "legacy"


def test_hsa_synthetic_row_compact_backward_selector_bucket_dense_unsupported_skips_split(monkeypatch):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    q_rows = torch.empty((1, 1, 64))
    union_k_row_idx = torch.empty((8, 12), dtype=torch.int32)
    q_row_idx = torch.empty((8, 2), dtype=torch.int32)
    unique_key_row_idx = torch.empty((6,), dtype=torch.int32)

    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD", "on")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT", "bucket_dense")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_QGROUPS_PER_CTA_BWD", "4")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_SPLIT_BWD", "on")
    mode = synthetic_module._select_row_compact_synthetic_bwd_mode(
        q_rows,
        union_k_row_idx,
        q_row_idx,
        unique_key_row_idx,
        4,
    )
    assert mode == "one_kernel"


def test_hsa_synthetic_one_kernel_backward_selector_honors_pingpong_mode(monkeypatch):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    calls = []

    def _base(*args, **kwargs):
        calls.append("base")

    def _pingpong(*args, **kwargs):
        calls.append("pingpong")

    monkeypatch.setattr(synthetic_module, "_run_synthetic_direct_row_micro_bwd_kernel_one_kernel_base", _base)
    monkeypatch.setattr(
        synthetic_module,
        "_run_synthetic_direct_row_micro_bwd_kernel_one_kernel_pingpong",
        _pingpong,
    )

    q_rows = torch.empty((1, 1, 64))
    k_rows = torch.empty((1, 1, 64))
    v_rows = torch.empty((1, 1, 64))
    out_rows = torch.empty((1, 1, 64))
    dout_rows = torch.empty((1, 1, 64))
    lse_rows = torch.empty((1, 1))
    q_row_idx = torch.empty((8, 2), dtype=torch.int32)
    union_to_row_slot = torch.empty((8, 2, 12), dtype=torch.int32)
    unique_key_row_idx = torch.empty((6,), dtype=torch.int32)
    unique_key_member_idx = torch.empty((16,), dtype=torch.int32)
    unique_key_union_idx = torch.empty((16,), dtype=torch.int32)
    unique_key_occurrence_row_ptr = torch.empty((7,), dtype=torch.int32)
    dq_rows = torch.empty((1, 1, 64))
    dk_rows = torch.empty((1, 1, 64))
    dv_rows = torch.empty((1, 1, 64))

    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_PINGPONG", "on")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT", "baseline")
    synthetic_module._run_synthetic_direct_row_micro_bwd_kernel_one_kernel(
        q_rows,
        k_rows,
        v_rows,
        out_rows,
        dout_rows,
        lse_rows,
        q_row_idx,
        union_to_row_slot,
        unique_key_row_idx,
        unique_key_member_idx,
        unique_key_union_idx,
        unique_key_occurrence_row_ptr,
        dq_rows,
        dk_rows,
        dv_rows,
        softmax_scale=1.0,
        max_unique_key_occurrences=4,
    )
    assert calls == ["pingpong"]

    calls.clear()
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_PINGPONG", "off")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT", "baseline")
    synthetic_module._run_synthetic_direct_row_micro_bwd_kernel_one_kernel(
        q_rows,
        k_rows,
        v_rows,
        out_rows,
        dout_rows,
        lse_rows,
        q_row_idx,
        union_to_row_slot,
        unique_key_row_idx,
        unique_key_member_idx,
        unique_key_union_idx,
        unique_key_occurrence_row_ptr,
        dq_rows,
        dk_rows,
        dv_rows,
        softmax_scale=1.0,
        max_unique_key_occurrences=4,
    )
    assert calls == ["base"]


def test_hsa_synthetic_one_kernel_backward_selector_falls_back_when_pingpong_unsupported(monkeypatch):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    calls = []

    def _base(*args, **kwargs):
        calls.append("base")

    def _pingpong(*args, **kwargs):
        calls.append("pingpong")

    monkeypatch.setattr(synthetic_module, "_run_synthetic_direct_row_micro_bwd_kernel_one_kernel_base", _base)
    monkeypatch.setattr(
        synthetic_module,
        "_run_synthetic_direct_row_micro_bwd_kernel_one_kernel_pingpong",
        _pingpong,
    )

    q_rows = torch.empty((1, 1, 64))
    k_rows = torch.empty((1, 1, 64))
    v_rows = torch.empty((1, 1, 64))
    out_rows = torch.empty((1, 1, 64))
    dout_rows = torch.empty((1, 1, 64))
    lse_rows = torch.empty((1, 1))
    q_row_idx = torch.empty((8, 2), dtype=torch.int32)
    union_to_row_slot = torch.empty((8, 2, 13), dtype=torch.int32)
    unique_key_row_idx = torch.empty((6,), dtype=torch.int32)
    unique_key_member_idx = torch.empty((16,), dtype=torch.int32)
    unique_key_union_idx = torch.empty((16,), dtype=torch.int32)
    unique_key_occurrence_row_ptr = torch.empty((7,), dtype=torch.int32)
    dq_rows = torch.empty((1, 1, 64))
    dk_rows = torch.empty((1, 1, 64))
    dv_rows = torch.empty((1, 1, 64))

    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_PINGPONG", "auto")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT", "baseline")
    synthetic_module._run_synthetic_direct_row_micro_bwd_kernel_one_kernel(
        q_rows,
        k_rows,
        v_rows,
        out_rows,
        dout_rows,
        lse_rows,
        q_row_idx,
        union_to_row_slot,
        unique_key_row_idx,
        unique_key_member_idx,
        unique_key_union_idx,
        unique_key_occurrence_row_ptr,
        dq_rows,
        dk_rows,
        dv_rows,
        softmax_scale=1.0,
        max_unique_key_occurrences=4,
    )
    assert calls == ["base"]


def test_hsa_synthetic_one_kernel_backward_selector_honors_variant_mode(monkeypatch):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    calls = []

    def _base(*args, **kwargs):
        calls.append("base")

    def _short(*args, **kwargs):
        calls.append("short")

    def _long(*args, **kwargs):
        calls.append("long")

    monkeypatch.setattr(synthetic_module, "_run_synthetic_direct_row_micro_bwd_kernel_one_kernel_base", _base)
    monkeypatch.setattr(synthetic_module, "_run_synthetic_direct_row_micro_bwd_kernel_one_kernel_short", _short)
    monkeypatch.setattr(synthetic_module, "_run_synthetic_direct_row_micro_bwd_kernel_one_kernel_long", _long)

    q_rows = torch.empty((1, 1, 64))
    k_rows = torch.empty((1, 1, 64))
    v_rows = torch.empty((1, 1, 64))
    out_rows = torch.empty((1, 1, 64))
    dout_rows = torch.empty((1, 1, 64))
    lse_rows = torch.empty((1, 1))
    q_row_idx = torch.empty((8, 2), dtype=torch.int32)
    union_to_row_slot = torch.empty((8, 2, 12), dtype=torch.int32)
    unique_key_row_idx = torch.empty((6,), dtype=torch.int32)
    unique_key_member_idx = torch.empty((16,), dtype=torch.int32)
    unique_key_union_idx = torch.empty((16,), dtype=torch.int32)
    unique_key_occurrence_row_ptr = torch.empty((7,), dtype=torch.int32)
    dq_rows = torch.empty((1, 1, 64))
    dk_rows = torch.empty((1, 1, 64))
    dv_rows = torch.empty((1, 1, 64))

    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_PINGPONG", "off")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT", "baseline")
    synthetic_module._run_synthetic_direct_row_micro_bwd_kernel_one_kernel(
        q_rows,
        k_rows,
        v_rows,
        out_rows,
        dout_rows,
        lse_rows,
        q_row_idx,
        union_to_row_slot,
        unique_key_row_idx,
        unique_key_member_idx,
        unique_key_union_idx,
        unique_key_occurrence_row_ptr,
        dq_rows,
        dk_rows,
        dv_rows,
        softmax_scale=1.0,
        max_unique_key_occurrences=4,
    )
    assert calls == ["base"]

    calls.clear()
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT", "short")
    synthetic_module._run_synthetic_direct_row_micro_bwd_kernel_one_kernel(
        q_rows,
        k_rows,
        v_rows,
        out_rows,
        dout_rows,
        lse_rows,
        q_row_idx,
        union_to_row_slot,
        unique_key_row_idx,
        unique_key_member_idx,
        unique_key_union_idx,
        unique_key_occurrence_row_ptr,
        dq_rows,
        dk_rows,
        dv_rows,
        softmax_scale=1.0,
        max_unique_key_occurrences=4,
    )
    assert calls == ["short"]

    calls.clear()
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT", "long")
    synthetic_module._run_synthetic_direct_row_micro_bwd_kernel_one_kernel(
        q_rows,
        k_rows,
        v_rows,
        out_rows,
        dout_rows,
        lse_rows,
        q_row_idx,
        union_to_row_slot,
        unique_key_row_idx,
        unique_key_member_idx,
        unique_key_union_idx,
        unique_key_occurrence_row_ptr,
        dq_rows,
        dk_rows,
        dv_rows,
        softmax_scale=1.0,
        max_unique_key_occurrences=4,
    )
    assert calls == ["long"]


def test_hsa_synthetic_one_kernel_backward_variant_auto_resolves_bucket_dense(monkeypatch):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT", "auto")
    assert synthetic_module._select_synthetic_one_kernel_bwd_variant(torch.empty((8, 2), dtype=torch.int32)) == "bucket_dense"


def test_hsa_synthetic_one_kernel_backward_variant_explicit_bucket_dense_tc(monkeypatch):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT", "bucket_dense_tc")
    assert synthetic_module._select_synthetic_one_kernel_bwd_variant(torch.empty((8, 2), dtype=torch.int32)) == "bucket_dense_tc"


def test_hsa_synthetic_one_kernel_backward_variant_explicit_bucket_dense_saved_prob(monkeypatch):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT", "bucket_dense_saved_prob")
    assert (
        synthetic_module._select_synthetic_one_kernel_bwd_variant(torch.empty((8, 2), dtype=torch.int32))
        == "bucket_dense_saved_prob"
    )


def test_hsa_synthetic_one_kernel_backward_variant_explicit_bucket_dense_two_pass(monkeypatch):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT", "bucket_dense_two_pass")
    assert (
        synthetic_module._select_synthetic_one_kernel_bwd_variant(torch.empty((8, 2), dtype=torch.int32))
        == "bucket_dense_two_pass"
    )


def test_hsa_synthetic_one_kernel_backward_variant_explicit_bucket_dense_dualrow(monkeypatch):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT", "bucket_dense_dualrow")
    assert (
        synthetic_module._select_synthetic_one_kernel_bwd_variant(torch.empty((8, 2), dtype=torch.int32))
        == "bucket_dense_dualrow"
    )


def test_hsa_synthetic_row_compact_one_kernel_dispatch_routes_bucket_dense(monkeypatch):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    calls = []

    def _bucket_dense(*args, **kwargs):
        calls.append("bucket_dense")

    def _legacy(*args, **kwargs):
        calls.append("legacy")

    monkeypatch.setattr(synthetic_module, "_run_synthetic_direct_row_micro_bwd_kernel_bucket_dense", _bucket_dense)
    monkeypatch.setattr(synthetic_module, "_run_synthetic_direct_row_micro_bwd_kernel_one_kernel", _legacy)
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT", "auto")

    q_rows = torch.empty((1, 1, 64))
    k_rows = torch.empty((1, 1, 64))
    v_rows = torch.empty((1, 1, 64))
    out_rows = torch.empty((1, 1, 64))
    dout_rows = torch.empty((1, 1, 64))
    lse_rows = torch.empty((1, 1))
    q_row_idx = torch.empty((8, 2), dtype=torch.int32)
    row_k_row_idx = torch.empty((8, 2, 12), dtype=torch.int32)
    union_k_row_idx = torch.empty((8, 12), dtype=torch.int32)
    row_k_to_union_idx = torch.empty((8, 2, 12), dtype=torch.int32)
    union_to_row_slot = torch.empty((8, 2, 12), dtype=torch.int32)
    q_length = torch.empty((8,), dtype=torch.int32)
    row_k_length = torch.empty((8, 2), dtype=torch.int32)
    union_k_length = torch.empty((8,), dtype=torch.int32)
    dq_rows = torch.empty((1, 1, 64))
    dk_rows = torch.empty((1, 1, 64))
    dv_rows = torch.empty((1, 1, 64))

    synthetic_module._run_synthetic_direct_row_micro_bwd_kernel_row_compact_one_kernel(
        q_rows,
        k_rows,
        v_rows,
        out_rows,
        dout_rows,
        lse_rows,
        q_row_idx,
        row_k_row_idx,
        union_k_row_idx,
        row_k_to_union_idx,
        union_to_row_slot,
        q_length,
        row_k_length,
        union_k_length,
        None,
        None,
        None,
        None,
        dq_rows,
        dk_rows,
        dv_rows,
        softmax_scale=1.0,
        max_unique_key_occurrences=0,
        workspace={},
    )
    assert calls == ["bucket_dense"]


def test_hsa_synthetic_row_compact_one_kernel_dispatch_routes_bucket_dense_tc(monkeypatch):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    calls = []

    def _bucket_dense_tc(*args, **kwargs):
        calls.append("bucket_dense_tc")

    def _legacy(*args, **kwargs):
        calls.append("legacy")

    monkeypatch.setattr(
        synthetic_module,
        "_run_synthetic_direct_row_micro_bwd_kernel_bucket_dense_tc",
        _bucket_dense_tc,
    )
    monkeypatch.setattr(synthetic_module, "_run_synthetic_direct_row_micro_bwd_kernel_one_kernel", _legacy)
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT", "bucket_dense_tc")

    q_rows = torch.empty((1, 1, 64))
    k_rows = torch.empty((1, 1, 64))
    v_rows = torch.empty((1, 1, 64))
    out_rows = torch.empty((1, 1, 64))
    dout_rows = torch.empty((1, 1, 64))
    lse_rows = torch.empty((1, 1))
    q_row_idx = torch.empty((8, 2), dtype=torch.int32)
    row_k_row_idx = torch.empty((8, 2, 12), dtype=torch.int32)
    union_k_row_idx = torch.empty((8, 12), dtype=torch.int32)
    row_k_to_union_idx = torch.empty((8, 2, 12), dtype=torch.int32)
    union_to_row_slot = torch.empty((8, 2, 12), dtype=torch.int32)
    q_length = torch.empty((8,), dtype=torch.int32)
    row_k_length = torch.empty((8, 2), dtype=torch.int32)
    union_k_length = torch.empty((8,), dtype=torch.int32)
    dq_rows = torch.empty((1, 1, 64))
    dk_rows = torch.empty((1, 1, 64))
    dv_rows = torch.empty((1, 1, 64))

    synthetic_module._run_synthetic_direct_row_micro_bwd_kernel_row_compact_one_kernel(
        q_rows,
        k_rows,
        v_rows,
        out_rows,
        dout_rows,
        lse_rows,
        q_row_idx,
        row_k_row_idx,
        union_k_row_idx,
        row_k_to_union_idx,
        union_to_row_slot,
        q_length,
        row_k_length,
        union_k_length,
        None,
        None,
        None,
        None,
        dq_rows,
        dk_rows,
        dv_rows,
        softmax_scale=1.0,
        max_unique_key_occurrences=0,
        workspace={},
    )
    assert calls == ["bucket_dense_tc"]


def test_hsa_synthetic_row_compact_one_kernel_dispatch_routes_bucket_dense_saved_prob(monkeypatch):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    calls = []

    def _bucket_dense_saved_prob(*args, **kwargs):
        calls.append("bucket_dense_saved_prob")

    def _bucket_dense(*args, **kwargs):
        calls.append("bucket_dense")

    monkeypatch.setattr(
        synthetic_module,
        "_run_synthetic_direct_row_micro_bwd_kernel_bucket_dense_saved_prob",
        _bucket_dense_saved_prob,
    )
    monkeypatch.setattr(synthetic_module, "_run_synthetic_direct_row_micro_bwd_kernel_bucket_dense", _bucket_dense)
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT", "bucket_dense_saved_prob")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_QGROUPS_PER_CTA_BWD", "2")

    q_rows = torch.empty((65536, 1, 64), dtype=torch.bfloat16)
    k_rows = torch.empty((1, 1, 64), dtype=torch.bfloat16)
    v_rows = torch.empty((1, 1, 64), dtype=torch.bfloat16)
    out_rows = torch.empty((65536, 1, 64), dtype=torch.bfloat16)
    dout_rows = torch.empty((65536, 1, 64), dtype=torch.bfloat16)
    lse_rows = torch.empty((65536, 1))
    q_row_idx = torch.empty((8, 2), dtype=torch.int32)
    row_k_row_idx = torch.empty((8, 2, 12), dtype=torch.int32)
    union_k_row_idx = torch.empty((8, 12), dtype=torch.int32)
    row_k_to_union_idx = torch.empty((8, 2, 12), dtype=torch.int32)
    union_to_row_slot = torch.empty((8, 2, 12), dtype=torch.int32)
    q_length = torch.empty((8,), dtype=torch.int32)
    row_k_length = torch.empty((8, 2), dtype=torch.int32)
    union_k_length = torch.empty((8,), dtype=torch.int32)
    saved_prob_rows = torch.empty((8, 2, 1, 16), dtype=torch.bfloat16)
    dq_rows = torch.empty((65536, 1, 64))
    dk_rows = torch.empty((1, 1, 64))
    dv_rows = torch.empty((1, 1, 64))

    synthetic_module._run_synthetic_direct_row_micro_bwd_kernel_row_compact_one_kernel(
        q_rows,
        k_rows,
        v_rows,
        out_rows,
        dout_rows,
        lse_rows,
        q_row_idx,
        row_k_row_idx,
        union_k_row_idx,
        row_k_to_union_idx,
        union_to_row_slot,
        q_length,
        row_k_length,
        union_k_length,
        None,
        None,
        None,
        None,
        dq_rows,
        dk_rows,
        dv_rows,
        saved_prob_rows=saved_prob_rows,
        softmax_scale=1.0,
        max_unique_key_occurrences=0,
        workspace={},
    )
    assert calls == ["bucket_dense_saved_prob"]


def test_hsa_synthetic_row_compact_one_kernel_dispatch_routes_bucket_dense_two_pass(monkeypatch):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    calls = []

    def _bucket_dense_two_pass(*args, **kwargs):
        calls.append("bucket_dense_two_pass")

    def _bucket_dense(*args, **kwargs):
        calls.append("bucket_dense")

    monkeypatch.setattr(
        synthetic_module,
        "_run_synthetic_direct_row_micro_bwd_kernel_bucket_dense_two_pass",
        _bucket_dense_two_pass,
    )
    monkeypatch.setattr(synthetic_module, "_run_synthetic_direct_row_micro_bwd_kernel_bucket_dense", _bucket_dense)
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT", "bucket_dense_two_pass")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_QGROUPS_PER_CTA_BWD", "2")

    q_rows = torch.empty((65536, 1, 64))
    k_rows = torch.empty((1, 1, 64))
    v_rows = torch.empty((1, 1, 64))
    out_rows = torch.empty((65536, 1, 64))
    dout_rows = torch.empty((65536, 1, 64))
    lse_rows = torch.empty((65536, 1))
    q_row_idx = torch.empty((8, 2), dtype=torch.int32)
    row_k_row_idx = torch.empty((8, 2, 12), dtype=torch.int32)
    union_k_row_idx = torch.empty((8, 12), dtype=torch.int32)
    row_k_to_union_idx = torch.empty((8, 2, 12), dtype=torch.int32)
    union_to_row_slot = torch.empty((8, 2, 12), dtype=torch.int32)
    q_length = torch.empty((8,), dtype=torch.int32)
    row_k_length = torch.empty((8, 2), dtype=torch.int32)
    union_k_length = torch.empty((8,), dtype=torch.int32)
    unique_key_row_idx = torch.empty((1,), dtype=torch.int32)
    unique_key_member_idx = torch.empty((1,), dtype=torch.int32)
    unique_key_union_idx = torch.empty((1,), dtype=torch.int32)
    unique_key_occurrence_row_ptr = torch.tensor([0, 1], dtype=torch.int32)
    dq_rows = torch.empty((65536, 1, 64))
    dk_rows = torch.empty((1, 1, 64))
    dv_rows = torch.empty((1, 1, 64))

    synthetic_module._run_synthetic_direct_row_micro_bwd_kernel_row_compact_one_kernel(
        q_rows,
        k_rows,
        v_rows,
        out_rows,
        dout_rows,
        lse_rows,
        q_row_idx,
        row_k_row_idx,
        union_k_row_idx,
        row_k_to_union_idx,
        union_to_row_slot,
        q_length,
        row_k_length,
        union_k_length,
        unique_key_row_idx,
        unique_key_member_idx,
        unique_key_union_idx,
        unique_key_occurrence_row_ptr,
        dq_rows,
        dk_rows,
        dv_rows,
        softmax_scale=1.0,
        max_unique_key_occurrences=1,
        workspace={},
    )
    assert calls == ["bucket_dense_two_pass"]


def test_hsa_synthetic_row_compact_one_kernel_dispatch_bucket_dense_saved_prob_falls_back_without_cache(monkeypatch):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    calls = []

    def _bucket_dense_saved_prob(*args, **kwargs):
        calls.append("bucket_dense_saved_prob")

    def _bucket_dense(*args, **kwargs):
        calls.append("bucket_dense")

    monkeypatch.setattr(
        synthetic_module,
        "_run_synthetic_direct_row_micro_bwd_kernel_bucket_dense_saved_prob",
        _bucket_dense_saved_prob,
    )
    monkeypatch.setattr(synthetic_module, "_run_synthetic_direct_row_micro_bwd_kernel_bucket_dense", _bucket_dense)
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT", "bucket_dense_saved_prob")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_QGROUPS_PER_CTA_BWD", "2")

    q_rows = torch.empty((32768, 1, 64), dtype=torch.bfloat16)
    k_rows = torch.empty((1, 1, 64), dtype=torch.bfloat16)
    v_rows = torch.empty((1, 1, 64), dtype=torch.bfloat16)
    out_rows = torch.empty((32768, 1, 64), dtype=torch.bfloat16)
    dout_rows = torch.empty((32768, 1, 64), dtype=torch.bfloat16)
    lse_rows = torch.empty((32768, 1))
    q_row_idx = torch.empty((8, 2), dtype=torch.int32)
    row_k_row_idx = torch.empty((8, 2, 12), dtype=torch.int32)
    union_k_row_idx = torch.empty((8, 12), dtype=torch.int32)
    row_k_to_union_idx = torch.empty((8, 2, 12), dtype=torch.int32)
    union_to_row_slot = torch.empty((8, 2, 12), dtype=torch.int32)
    q_length = torch.empty((8,), dtype=torch.int32)
    row_k_length = torch.empty((8, 2), dtype=torch.int32)
    union_k_length = torch.empty((8,), dtype=torch.int32)
    dq_rows = torch.empty((32768, 1, 64))
    dk_rows = torch.empty((1, 1, 64))
    dv_rows = torch.empty((1, 1, 64))

    synthetic_module._run_synthetic_direct_row_micro_bwd_kernel_row_compact_one_kernel(
        q_rows,
        k_rows,
        v_rows,
        out_rows,
        dout_rows,
        lse_rows,
        q_row_idx,
        row_k_row_idx,
        union_k_row_idx,
        row_k_to_union_idx,
        union_to_row_slot,
        q_length,
        row_k_length,
        union_k_length,
        None,
        None,
        None,
        None,
        dq_rows,
        dk_rows,
        dv_rows,
        saved_prob_rows=None,
        softmax_scale=1.0,
        max_unique_key_occurrences=0,
        workspace={},
    )
    assert calls == ["bucket_dense"]


def test_hsa_synthetic_row_compact_one_kernel_dispatch_bucket_dense_saved_prob_falls_back_below_long_gate(monkeypatch):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    calls = []

    def _bucket_dense_saved_prob(*args, **kwargs):
        calls.append("bucket_dense_saved_prob")

    def _bucket_dense(*args, **kwargs):
        calls.append("bucket_dense")

    monkeypatch.setattr(
        synthetic_module,
        "_run_synthetic_direct_row_micro_bwd_kernel_bucket_dense_saved_prob",
        _bucket_dense_saved_prob,
    )
    monkeypatch.setattr(synthetic_module, "_run_synthetic_direct_row_micro_bwd_kernel_bucket_dense", _bucket_dense)
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT", "bucket_dense_saved_prob")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_QGROUPS_PER_CTA_BWD", "2")

    q_rows = torch.empty((32768, 1, 64), dtype=torch.bfloat16)
    k_rows = torch.empty((1, 1, 64), dtype=torch.bfloat16)
    v_rows = torch.empty((1, 1, 64), dtype=torch.bfloat16)
    out_rows = torch.empty((32768, 1, 64), dtype=torch.bfloat16)
    dout_rows = torch.empty((32768, 1, 64), dtype=torch.bfloat16)
    lse_rows = torch.empty((32768, 1))
    q_row_idx = torch.empty((8, 2), dtype=torch.int32)
    row_k_row_idx = torch.empty((8, 2, 12), dtype=torch.int32)
    union_k_row_idx = torch.empty((8, 12), dtype=torch.int32)
    row_k_to_union_idx = torch.empty((8, 2, 12), dtype=torch.int32)
    union_to_row_slot = torch.empty((8, 2, 12), dtype=torch.int32)
    q_length = torch.empty((8,), dtype=torch.int32)
    row_k_length = torch.empty((8, 2), dtype=torch.int32)
    union_k_length = torch.empty((8,), dtype=torch.int32)
    saved_prob_rows = torch.empty((8, 2, 1, 16), dtype=torch.bfloat16)
    dq_rows = torch.empty((32768, 1, 64))
    dk_rows = torch.empty((1, 1, 64))
    dv_rows = torch.empty((1, 1, 64))

    synthetic_module._run_synthetic_direct_row_micro_bwd_kernel_row_compact_one_kernel(
        q_rows,
        k_rows,
        v_rows,
        out_rows,
        dout_rows,
        lse_rows,
        q_row_idx,
        row_k_row_idx,
        union_k_row_idx,
        row_k_to_union_idx,
        union_to_row_slot,
        q_length,
        row_k_length,
        union_k_length,
        None,
        None,
        None,
        None,
        dq_rows,
        dk_rows,
        dv_rows,
        saved_prob_rows=saved_prob_rows,
        softmax_scale=1.0,
        max_unique_key_occurrences=0,
        workspace={},
    )
    assert calls == ["bucket_dense"]


def test_hsa_synthetic_row_compact_one_kernel_dispatch_bucket_dense_two_pass_falls_back_below_long_gate(monkeypatch):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    calls = []

    def _bucket_dense_two_pass(*args, **kwargs):
        calls.append("bucket_dense_two_pass")

    def _bucket_dense(*args, **kwargs):
        calls.append("bucket_dense")

    monkeypatch.setattr(
        synthetic_module,
        "_run_synthetic_direct_row_micro_bwd_kernel_bucket_dense_two_pass",
        _bucket_dense_two_pass,
    )
    monkeypatch.setattr(synthetic_module, "_run_synthetic_direct_row_micro_bwd_kernel_bucket_dense", _bucket_dense)
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT", "bucket_dense_two_pass")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_QGROUPS_PER_CTA_BWD", "2")

    q_rows = torch.empty((32768, 1, 64))
    k_rows = torch.empty((1, 1, 64))
    v_rows = torch.empty((1, 1, 64))
    out_rows = torch.empty((32768, 1, 64))
    dout_rows = torch.empty((32768, 1, 64))
    lse_rows = torch.empty((32768, 1))
    q_row_idx = torch.empty((8, 2), dtype=torch.int32)
    row_k_row_idx = torch.empty((8, 2, 12), dtype=torch.int32)
    union_k_row_idx = torch.empty((8, 12), dtype=torch.int32)
    row_k_to_union_idx = torch.empty((8, 2, 12), dtype=torch.int32)
    union_to_row_slot = torch.empty((8, 2, 12), dtype=torch.int32)
    q_length = torch.empty((8,), dtype=torch.int32)
    row_k_length = torch.empty((8, 2), dtype=torch.int32)
    union_k_length = torch.empty((8,), dtype=torch.int32)
    dq_rows = torch.empty((32768, 1, 64))
    dk_rows = torch.empty((1, 1, 64))
    dv_rows = torch.empty((1, 1, 64))

    synthetic_module._run_synthetic_direct_row_micro_bwd_kernel_row_compact_one_kernel(
        q_rows,
        k_rows,
        v_rows,
        out_rows,
        dout_rows,
        lse_rows,
        q_row_idx,
        row_k_row_idx,
        union_k_row_idx,
        row_k_to_union_idx,
        union_to_row_slot,
        q_length,
        row_k_length,
        union_k_length,
        None,
        None,
        None,
        None,
        dq_rows,
        dk_rows,
        dv_rows,
        softmax_scale=1.0,
        max_unique_key_occurrences=0,
        workspace={},
    )
    assert calls == ["bucket_dense"]


def test_hsa_synthetic_row_compact_one_kernel_dispatch_routes_bucket_dense_dualrow(monkeypatch):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    calls = []

    def _bucket_dense_dualrow(*args, **kwargs):
        calls.append("bucket_dense_dualrow")

    def _legacy(*args, **kwargs):
        calls.append("legacy")

    monkeypatch.setattr(
        synthetic_module,
        "_run_synthetic_direct_row_micro_bwd_kernel_bucket_dense_dualrow",
        _bucket_dense_dualrow,
    )
    monkeypatch.setattr(synthetic_module, "_run_synthetic_direct_row_micro_bwd_kernel_one_kernel", _legacy)
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT", "bucket_dense_dualrow")

    q_rows = torch.empty((1, 1, 64))
    k_rows = torch.empty((1, 1, 64))
    v_rows = torch.empty((1, 1, 64))
    out_rows = torch.empty((1, 1, 64))
    dout_rows = torch.empty((1, 1, 64))
    lse_rows = torch.empty((1, 1))
    q_row_idx = torch.empty((8, 2), dtype=torch.int32)
    row_k_row_idx = torch.empty((8, 2, 12), dtype=torch.int32)
    union_k_row_idx = torch.empty((8, 12), dtype=torch.int32)
    row_k_to_union_idx = torch.empty((8, 2, 12), dtype=torch.int32)
    union_to_row_slot = torch.empty((8, 2, 12), dtype=torch.int32)
    q_length = torch.empty((8,), dtype=torch.int32)
    row_k_length = torch.empty((8, 2), dtype=torch.int32)
    union_k_length = torch.empty((8,), dtype=torch.int32)
    dq_rows = torch.empty((1, 1, 64))
    dk_rows = torch.empty((1, 1, 64))
    dv_rows = torch.empty((1, 1, 64))

    synthetic_module._run_synthetic_direct_row_micro_bwd_kernel_row_compact_one_kernel(
        q_rows,
        k_rows,
        v_rows,
        out_rows,
        dout_rows,
        lse_rows,
        q_row_idx,
        row_k_row_idx,
        union_k_row_idx,
        row_k_to_union_idx,
        union_to_row_slot,
        q_length,
        row_k_length,
        union_k_length,
        None,
        None,
        None,
        None,
        dq_rows,
        dk_rows,
        dv_rows,
        softmax_scale=1.0,
        max_unique_key_occurrences=0,
        workspace={},
    )
    assert calls == ["bucket_dense_dualrow"]


def test_hsa_synthetic_one_kernel_long_keys_per_cta_selector(monkeypatch):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    monkeypatch.delenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_LONG_KEYS_PER_CTA", raising=False)
    assert synthetic_module._select_synthetic_one_kernel_long_keys_per_cta() == 4

    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_LONG_KEYS_PER_CTA", "8")
    assert synthetic_module._select_synthetic_one_kernel_long_keys_per_cta() == 8

    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_LONG_KEYS_PER_CTA", "16")
    assert synthetic_module._select_synthetic_one_kernel_long_keys_per_cta() == 16

    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_LONG_KEYS_PER_CTA", "auto")
    assert synthetic_module._select_synthetic_one_kernel_long_keys_per_cta(torch.empty((1, 131072, 4, 64))) == 4


def test_hsa_synthetic_one_kernel_backward_selector_warpgroup_aliases_long_8(monkeypatch):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    calls = []

    def _long(*args, **kwargs):
        calls.append(kwargs["keys_per_cta"])

    monkeypatch.setattr(synthetic_module, "_run_synthetic_direct_row_micro_bwd_kernel_one_kernel_long", _long)

    q_rows = torch.empty((1, 1, 64))
    k_rows = torch.empty((1, 1, 64))
    v_rows = torch.empty((1, 1, 64))
    out_rows = torch.empty((1, 1, 64))
    dout_rows = torch.empty((1, 1, 64))
    lse_rows = torch.empty((1, 1))
    q_row_idx = torch.empty((4096, 2), dtype=torch.int32)
    union_to_row_slot = torch.empty((8, 2, 12), dtype=torch.int32)
    unique_key_row_idx = torch.empty((6,), dtype=torch.int32)
    unique_key_member_idx = torch.empty((16,), dtype=torch.int32)
    unique_key_union_idx = torch.empty((16,), dtype=torch.int32)
    unique_key_occurrence_row_ptr = torch.empty((7,), dtype=torch.int32)
    dq_rows = torch.empty((1, 1, 64))
    dk_rows = torch.empty((1, 1, 64))
    dv_rows = torch.empty((1, 1, 64))

    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_PINGPONG", "off")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT", "warpgroup")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_LONG_KEYS_PER_CTA", "16")
    synthetic_module._run_synthetic_direct_row_micro_bwd_kernel_one_kernel(
        q_rows,
        k_rows,
        v_rows,
        out_rows,
        dout_rows,
        lse_rows,
        q_row_idx,
        union_to_row_slot,
        unique_key_row_idx,
        unique_key_member_idx,
        unique_key_union_idx,
        unique_key_occurrence_row_ptr,
        dq_rows,
        dk_rows,
        dv_rows,
        softmax_scale=1.0,
        max_unique_key_occurrences=4,
    )
    assert calls == [8]


def test_hsa_synthetic_long_bwd_mode_selector_honors_env(monkeypatch):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    q = torch.empty((1, 32768, 4, 64))
    q_rows = torch.empty((1, 1, 64))
    union_k_row_idx = torch.empty((8, 16), dtype=torch.int32)
    q_row_idx = torch.empty((2048, 2), dtype=torch.int32)
    unique_key_row_idx = torch.empty((6,), dtype=torch.int32)

    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LONG_BWD_MODE", "one_kernel")
    assert (
        synthetic_module._select_synthetic_long_bwd_mode(
            q,
            q_rows,
            union_k_row_idx,
            q_row_idx,
            unique_key_row_idx,
            4,
        )
        == "one_kernel"
    )

    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LONG_BWD_MODE", "two_stage")
    assert (
        synthetic_module._select_synthetic_long_bwd_mode(
            q,
            q_rows,
            union_k_row_idx,
            q_row_idx,
            unique_key_row_idx,
            4,
        )
        == "two_stage"
    )

    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LONG_BWD_MODE", "persistent")
    assert (
        synthetic_module._select_synthetic_long_bwd_mode(
            q,
            q_rows,
            union_k_row_idx,
            q_row_idx,
            unique_key_row_idx,
            4,
        )
        == "persistent"
    )

    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LONG_BWD_MODE", "persistent_member_tiled")
    assert (
        synthetic_module._select_synthetic_long_bwd_mode(
            q,
            q_rows,
            union_k_row_idx,
            q_row_idx,
            unique_key_row_idx,
            4,
        )
        == "persistent_member_tiled"
    )

    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LONG_BWD_MODE", "two_stage")
    unsupported_union = torch.empty((8, 17), dtype=torch.int32)
    assert (
        synthetic_module._select_synthetic_long_bwd_mode(
            q,
            q_rows,
            unsupported_union,
            q_row_idx,
            unique_key_row_idx,
            4,
        )
        == "one_kernel"
    )

    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LONG_BWD_MODE", "persistent_member_tiled")
    assert (
        synthetic_module._select_synthetic_long_bwd_mode(
            q,
            q_rows,
            unsupported_union,
            q_row_idx,
            unique_key_row_idx,
            4,
        )
        == "one_kernel"
    )

    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LONG_BWD_MODE", "persistent")
    assert (
        synthetic_module._select_synthetic_long_bwd_mode(
            q,
            q_rows,
            unsupported_union,
            q_row_idx,
            unique_key_row_idx,
            4,
        )
        == "one_kernel"
    )


def test_hsa_synthetic_long_bwd_mode_selector_auto_is_long_only(monkeypatch):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    q_short = torch.empty((1, 16384, 4, 64))
    q_long = torch.empty((1, 32768, 4, 64))
    q_rows = torch.empty((1, 1, 64))
    union_k_row_idx = torch.empty((8, 16), dtype=torch.int32)
    q_row_idx = torch.empty((2048, 2), dtype=torch.int32)
    unique_key_row_idx = torch.empty((6,), dtype=torch.int32)

    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LONG_BWD_MODE", "auto")
    assert (
        synthetic_module._select_synthetic_long_bwd_mode(
            q_short,
            q_rows,
            union_k_row_idx,
            q_row_idx,
            unique_key_row_idx,
            4,
        )
        == "one_kernel"
    )
    assert (
        synthetic_module._select_synthetic_long_bwd_mode(
            q_long,
            q_rows,
            union_k_row_idx,
            q_row_idx,
            unique_key_row_idx,
            4,
        )
        == "one_kernel"
    )

@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_hsa_synthetic_grid_gqa_falls_back(monkeypatch):
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    device = "cuda"
    batch_size, seqlen, nheads, headdim, n_kv_heads = 2, 1024, 8, 64, 2
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    dtype = torch.bfloat16
    q_data = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    k_data = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)
    v_data = torch.randn(batch_size, seqlen, n_kv_heads, headdim, device=device, dtype=dtype)

    monkeypatch.setattr(
        synthetic_module,
        "run_hsa_fwd_sm100_synthetic_grid",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("synthetic-grid forward used for GQA")),
    )
    monkeypatch.setenv("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "1")
    out_sparse, q_grad, k_grad, v_grad, out_ref, _, _, _ = _run_mixed_sparse_mask_case(
        batch_size=batch_size,
        seqlen=seqlen,
        nheads=nheads,
        headdim=headdim,
        n_kv_heads=n_kv_heads,
        keep_ids=keep_ids,
        hash_ids=hash_ids,
        q_data=q_data,
        k_data=k_data,
        v_data=v_data,
    )
    _assert_close(out_sparse.float(), out_ref.float(), "train_gqa_synthetic_grid_fallback_output")
    _assert_finite_gradients("train_gqa_synthetic_grid_fallback_grads", q_grad, k_grad, v_grad)


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_hsa_unpacked_direct_fwd_flag_disables_synthetic_grid(monkeypatch):
    import flash_attn.cute.hsa as hsa_module

    device = "cuda"
    batch_size, seqlen, nheads, headdim = 1, 65, 4, 64
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)

    monkeypatch.setenv("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "1")
    assert hsa_module._can_use_hsa_synthetic_grid_for_inputs(schedule, q, k) is True

    monkeypatch.setenv("FLASH_ATTN_HSA_UNPACKED_DIRECT_FWD", "1")
    assert hsa_module._can_use_hsa_synthetic_grid_for_inputs(schedule, q, k) is False


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_hsa_synthetic_grid_row_compact_guard_rejects_segmented_direct_plan(monkeypatch):
    import flash_attn.cute.hsa as hsa_module

    device = "cuda"
    batch_size, seqlen, nheads, headdim = 2, 1024, 8, 64
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)

    monkeypatch.setenv("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_REQUIRE_ROW_COMPACT", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_Q", "2")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_K", "2")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MAX_PACKED_K", "128")
    monkeypatch.setenv("FLASH_ATTN_HSA_SYNTHETIC_MAX_DIRECT_SEGMENTS", "4")

    runtime = hsa_module._get_hsa_block_sparse_runtime(schedule, q, k)
    hsa_module._ensure_hsa_synthetic_grid_metadata(schedule, runtime)

    metadata = runtime.forward_synthetic_grid
    assert metadata is not None
    direct_plan = metadata.forward_execution_plan["direct_execution_plan"]
    assert direct_plan is not None
    assert direct_plan["row_compact_plan"] is None
    assert hsa_module._can_use_hsa_synthetic_grid_for_inputs(schedule, q, k, runtime=runtime) is False


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_hsa_blocksparse_forward_unpacked_direct_flag_routes_direct_kernel(monkeypatch):
    import flash_attn.cute.flash_hsa_fwd_sm100 as hsa_fwd_module

    device = "cuda"
    batch_size, seqlen, nheads, headdim = 1, 65, 4, 64
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)
    sentinel_out = torch.full_like(v, 0.25)
    sentinel_lse = torch.full((batch_size, nheads, seqlen), -3.0, dtype=torch.float32, device=device)
    calls = {"direct": 0}

    def _tracked_direct(*args, **kwargs):
        calls["direct"] += 1
        return sentinel_out, sentinel_lse

    monkeypatch.setattr(hsa_fwd_module, "_run_hsa_fwd_sm100_direct", _tracked_direct)
    monkeypatch.setenv("FLASH_ATTN_HSA_UNPACKED_DIRECT_FWD", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_USE_MONOLITHIC_BWD", "0")

    out, lse, sentence_lse, sentence_q_stream, sentence_k_stream, sentence_v_stream, sentence_out_stream = (
        hsa_fwd_module.run_hsa_fwd_sm100_blocksparse(q, k, v, schedule, 1.0 / math.sqrt(headdim))
    )

    assert calls["direct"] == 1
    assert torch.equal(out, sentinel_out)
    assert torch.equal(lse, sentinel_lse)
    assert sentence_lse.numel() == 0
    assert sentence_q_stream.numel() == 0
    assert sentence_k_stream.numel() == 0
    assert sentence_v_stream.numel() == 0
    assert sentence_out_stream.numel() == 0


def test_benchmark_hsa_unpacked_direct_env_sets_flag():
    benchmark_hsa = _load_benchmark_hsa_module()
    case = next(candidate for candidate in benchmark_hsa.ALL_CASES if candidate.name == "mixed-small")

    env = benchmark_hsa._unpacked_direct_fwd_env(case)

    assert env["FLASH_ATTN_HSA_UNPACKED_DIRECT_FWD"] == "1"
    assert env["FLASH_ATTN_HSA_USE_SYNTHETIC_GRID"] == "0"
    assert env["FLASH_ATTN_HSA_SYNTHETIC_MICRO_FWD"] == "0"


def test_benchmark_hsa_unpacked_direct_compare_mode(monkeypatch):
    benchmark_hsa = _load_benchmark_hsa_module()

    monkeypatch.setenv("FLASH_ATTN_HSA_UNPACKED_DIRECT_FWD", "compare")
    assert benchmark_hsa._use_unpacked_direct_compare_mode() is True

    monkeypatch.setenv("FLASH_ATTN_HSA_UNPACKED_DIRECT_FWD", "sweep")
    assert benchmark_hsa._use_unpacked_direct_compare_mode() is True

    monkeypatch.setenv("FLASH_ATTN_HSA_UNPACKED_DIRECT_FWD", "1")
    assert benchmark_hsa._use_unpacked_direct_compare_mode() is False


def test_benchmark_hsa_approx_sparse_gemm_summary_helper(monkeypatch):
    benchmark_hsa = _load_benchmark_hsa_module()

    captured = {}

    def _fake_analyze(schedule, q, k, v, **kwargs):
        captured["args"] = (schedule, q, k, v)
        captured["kwargs"] = kwargs
        return {"status": "measured", "dense_qk_ms": 1.0}

    def _fake_summarize(report):
        assert report["dense_qk_ms"] == pytest.approx(1.0)
        return {"status": report["status"], "dense_qk_ms": report["dense_qk_ms"]}

    monkeypatch.setenv("FLASH_ATTN_HSA_APPROX_SPARSE_GEMM_MAX_BUCKETS", "7")
    monkeypatch.setenv("FLASH_ATTN_HSA_APPROX_SPARSE_GEMM_MAX_MEMBERS", "11")
    monkeypatch.setenv("FLASH_ATTN_HSA_APPROX_SPARSE_GEMM_WARMUP_ITERS", "3")
    monkeypatch.setenv("FLASH_ATTN_HSA_APPROX_SPARSE_GEMM_BENCH_ITERS", "4")
    monkeypatch.setenv("FLASH_ATTN_HSA_APPROX_SPARSE_GEMM_GROUP_SIZE", "4")
    monkeypatch.setenv("FLASH_ATTN_HSA_APPROX_SPARSE_GEMM_NNZ_PER_GROUP", "2")

    import flash_attn.cute.hsa_approx_sparse_gemm_analysis as approx_module

    monkeypatch.setattr(approx_module, "analyze_hsa_approx_sparse_gemm_forward", _fake_analyze)
    monkeypatch.setattr(approx_module, "summarize_hsa_approx_sparse_gemm_forward", _fake_summarize)

    q = torch.zeros(1, 1, 1, 64)
    summary = benchmark_hsa._get_approx_sparse_gemm_forward_summary(
        "schedule",
        q,
        q,
        q,
        use_synthetic_grid=True,
    )

    assert summary["status"] == "measured"
    assert summary["dense_qk_ms"] == pytest.approx(1.0)
    assert captured["kwargs"]["use_synthetic_grid"] is True
    assert captured["kwargs"]["max_buckets"] == 7
    assert captured["kwargs"]["max_members"] == 11
    assert captured["kwargs"]["warmup_iters"] == 3
    assert captured["kwargs"]["benchmark_iters"] == 4


def test_benchmark_hsa_shared_sparse_gemm_summary_helper(monkeypatch):
    benchmark_hsa = _load_benchmark_hsa_module()

    captured = {}

    def _fake_analyze(schedule, q, k, v, **kwargs):
        captured["args"] = (schedule, q, k, v)
        captured["kwargs"] = kwargs
        return {"status": "measured", "dense_qk_ms": 1.0}

    def _fake_summarize(report):
        assert report["dense_qk_ms"] == pytest.approx(1.0)
        return {"status": report["status"], "dense_qk_ms": report["dense_qk_ms"]}

    monkeypatch.setenv("FLASH_ATTN_HSA_SHARED_SPARSE_GEMM_GROUP_SIZE", "4")
    monkeypatch.setenv("FLASH_ATTN_HSA_SHARED_SPARSE_GEMM_NNZ_PER_GROUP", "2")
    monkeypatch.setenv("FLASH_ATTN_HSA_SHARED_SPARSE_GEMM_BUCKETIZER", "coarse_window")
    monkeypatch.setenv("FLASH_ATTN_HSA_SHARED_SPARSE_GEMM_MIN_JACCARD", "0.75")
    monkeypatch.setenv("FLASH_ATTN_HSA_SHARED_SPARSE_GEMM_MAX_SUPPORT_ROWS", "96")
    monkeypatch.setenv("FLASH_ATTN_HSA_SHARED_SPARSE_GEMM_MAX_QGROUPS_PER_BUCKET", "9")
    monkeypatch.setenv("FLASH_ATTN_HSA_SHARED_SPARSE_GEMM_MAX_BUCKETS", "7")
    monkeypatch.setenv("FLASH_ATTN_HSA_SHARED_SPARSE_GEMM_MAX_QGROUPS", "11")
    monkeypatch.setenv("FLASH_ATTN_HSA_SHARED_SPARSE_GEMM_WARMUP_ITERS", "3")
    monkeypatch.setenv("FLASH_ATTN_HSA_SHARED_SPARSE_GEMM_BENCH_ITERS", "4")
    monkeypatch.setenv("FLASH_ATTN_HSA_SHARED_SPARSE_GEMM_ENABLE_SEMI_STRUCTURED", "0")
    monkeypatch.setenv("FLASH_ATTN_HSA_SHARED_SPARSE_GEMM_ENABLE_CUSTOM_MASKED", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_SHARED_SPARSE_GEMM_ENABLE_FA4_PACKED", "0")
    monkeypatch.setenv("FLASH_ATTN_HSA_SHARED_SPARSE_GEMM_ENABLE_SHARED_CTA", "1")
    monkeypatch.setenv("FLASH_ATTN_HSA_SHARED_SPARSE_GEMM_SHARED_CTA_QGROUPS_PER_CTA", "8")

    import flash_attn.cute.hsa_shared_sparse_gemm_analysis as shared_module

    monkeypatch.setattr(shared_module, "analyze_hsa_shared_sparse_gemm_forward", _fake_analyze)
    monkeypatch.setattr(shared_module, "summarize_hsa_shared_sparse_gemm_forward", _fake_summarize)

    q = torch.zeros(1, 1, 1, 64)
    summary = benchmark_hsa._get_shared_sparse_gemm_forward_summary(
        "schedule",
        q,
        q,
        q,
        use_synthetic_grid=True,
    )

    assert summary["status"] == "measured"
    assert summary["dense_qk_ms"] == pytest.approx(1.0)
    assert captured["kwargs"]["use_synthetic_grid"] is True
    assert captured["kwargs"]["group_size"] == 4
    assert captured["kwargs"]["nnz_per_group"] == 2
    assert captured["kwargs"]["bucketizer"] == "coarse_window"
    assert captured["kwargs"]["min_jaccard"] == pytest.approx(0.75)
    assert captured["kwargs"]["max_support_rows"] == 96
    assert captured["kwargs"]["max_qgroups_per_bucket"] == 9
    assert captured["kwargs"]["max_buckets"] == 7
    assert captured["kwargs"]["max_qgroups"] == 11
    assert captured["kwargs"]["warmup_iters"] == 3
    assert captured["kwargs"]["benchmark_iters"] == 4
    assert captured["kwargs"]["enable_sparse_payload"] is False
    assert captured["kwargs"]["enable_custom_masked"] is True
    assert captured["kwargs"]["enable_fa4_packed"] is False
    assert captured["kwargs"]["enable_shared_cta"] is True
    assert captured["kwargs"]["shared_cta_qgroups_per_cta"] == 8


def test_benchmark_hsa_unpacked_direct_correctness_policy():
    benchmark_hsa = _load_benchmark_hsa_module()
    mixed_small = next(candidate for candidate in benchmark_hsa.ALL_CASES if candidate.name == "mixed-small")
    long_16k = next(candidate for candidate in benchmark_hsa.ALL_CASES if candidate.name == "long-16k")
    long_32k = next(candidate for candidate in benchmark_hsa.ALL_CASES if candidate.name == "long-32k")
    long_64k = next(candidate for candidate in benchmark_hsa.ALL_CASES if candidate.name == "long-64k")

    assert benchmark_hsa._should_validate_unpacked_direct_outputs(mixed_small) is True
    assert benchmark_hsa._should_validate_unpacked_direct_outputs(long_16k) is True
    assert benchmark_hsa._should_validate_unpacked_direct_outputs(long_32k) is True
    assert benchmark_hsa._should_validate_unpacked_direct_outputs(long_64k) is False


def test_benchmark_hsa_unpacked_direct_case_summary(monkeypatch):
    benchmark_hsa = _load_benchmark_hsa_module()
    case = next(candidate for candidate in benchmark_hsa.ALL_CASES if candidate.name == "mixed-small")
    q_data = torch.zeros(1, 1, 1, 1)
    k_data = torch.zeros(1, 1, 1, 1)
    v_data = torch.zeros(1, 1, 1, 1)

    monkeypatch.setattr(
        benchmark_hsa,
        "_build_case_tensors",
        lambda _: ("schedule", "keep_ids", "hash_ids", q_data, k_data, v_data),
    )

    def _fake_measure(_forward_fn, _q, _k, _v, _warmup_iters, _benchmark_iters, *, env_updates=None):
        env_updates = env_updates or {}
        if env_updates.get("FLASH_ATTN_HSA_UNPACKED_DIRECT_FWD") == "1":
            return {
                "fwd_ms": 8.0,
                "bwd_ms": 14.0,
                "fwd_bwd_ms": 22.0,
                "status": "measured",
            }
        return {
            "fwd_ms": 4.0,
            "bwd_ms": 14.0,
            "fwd_bwd_ms": 18.0,
            "status": "measured",
        }

    monkeypatch.setattr(benchmark_hsa, "_measure_triplet_or_status", _fake_measure)

    def _fake_run_sparse_attention(_q, _k, _v, _keep_ids, _hash_ids, _schedule, *, env_updates=None):
        env_updates = env_updates or {}
        fill_value = 2.0 if env_updates.get("FLASH_ATTN_HSA_UNPACKED_DIRECT_FWD") == "1" else 1.0
        return torch.full_like(v_data, fill_value)

    monkeypatch.setattr(benchmark_hsa, "_run_sparse_attention", _fake_run_sparse_attention)

    result = benchmark_hsa._benchmark_unpacked_direct_case(case)
    line = benchmark_hsa._format_unpacked_direct_case_report(result)

    assert result["case"] == "mixed-small"
    assert result["plain_sparse_fwd_ms"] == pytest.approx(4.0)
    assert result["unpacked_direct_fwd_ms"] == pytest.approx(8.0)
    assert result["speedup"] == pytest.approx(0.5)
    assert result["correctness_status"] == "measured"
    assert result["max_diff"] == pytest.approx(1.0)
    assert result["mean_diff"] == pytest.approx(1.0)
    assert result["status"] == "measured"
    assert "plain_sparse_fwd_ms=4.000" in line
    assert "unpacked_direct_fwd_ms=8.000" in line
    assert "plain_sparse_fwd_vs_unpacked_direct=0.500x" in line


def test_benchmark_hsa_unpacked_direct_long_case_skips_correctness(monkeypatch):
    benchmark_hsa = _load_benchmark_hsa_module()
    case = next(candidate for candidate in benchmark_hsa.ALL_CASES if candidate.name == "long-64k")
    q_data = torch.zeros(1, 1, 1, 1)
    k_data = torch.zeros(1, 1, 1, 1)
    v_data = torch.zeros(1, 1, 1, 1)

    monkeypatch.setattr(
        benchmark_hsa,
        "_build_case_tensors",
        lambda _: ("schedule", "keep_ids", "hash_ids", q_data, k_data, v_data),
    )
    monkeypatch.setattr(
        benchmark_hsa,
        "_measure_triplet_or_status",
        lambda *_args, **_kwargs: {
            "fwd_ms": 3.0,
            "bwd_ms": 5.0,
            "fwd_bwd_ms": 8.0,
            "status": "measured",
        },
    )
    monkeypatch.setattr(
        benchmark_hsa,
        "_run_sparse_attention",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("correctness path should be skipped")),
    )

    result = benchmark_hsa._benchmark_unpacked_direct_case(case)

    assert result["correctness_status"] == "skipped_long_gt_32k"
    assert result["max_diff"] is None
    assert result["mean_diff"] is None


def test_benchmark_hsa_unpacked_direct_case_subprocess_success(monkeypatch):
    benchmark_hsa = _load_benchmark_hsa_module()
    case = next(candidate for candidate in benchmark_hsa.ALL_CASES if candidate.name == "mixed-small")

    class _Completed:
        returncode = 0
        stdout = '{"case":"mixed-small","status":"measured","plain_sparse_status":"measured","unpacked_direct_status":"measured","correctness_status":"measured"}\n'
        stderr = ""

    monkeypatch.setattr(benchmark_hsa.subprocess, "run", lambda *args, **kwargs: _Completed())

    result = benchmark_hsa._run_unpacked_direct_case_subprocess(case)

    assert result["case"] == "mixed-small"
    assert result["status"] == "measured"


def test_benchmark_hsa_unpacked_direct_case_subprocess_failure(monkeypatch):
    benchmark_hsa = _load_benchmark_hsa_module()
    case = next(candidate for candidate in benchmark_hsa.ALL_CASES if candidate.name == "long-2M")

    class _Completed:
        returncode = 137
        stdout = ""
        stderr = "killed\n"

    monkeypatch.setattr(benchmark_hsa.subprocess, "run", lambda *args, **kwargs: _Completed())

    result = benchmark_hsa._run_unpacked_direct_case_subprocess(case)

    assert result["case"] == "long-2M"
    assert result["status"].startswith("child_exit_137")
    assert result["plain_sparse_fwd_ms"] is None
    assert result["unpacked_direct_fwd_ms"] is None


def test_benchmark_hsa_unpacked_direct_case_subprocess_timeout(monkeypatch):
    benchmark_hsa = _load_benchmark_hsa_module()
    case = next(candidate for candidate in benchmark_hsa.ALL_CASES if candidate.name == "long-1M")

    def _raise_timeout(*args, **kwargs):
        raise benchmark_hsa.subprocess.TimeoutExpired(cmd="python", timeout=42.0)

    monkeypatch.setattr(benchmark_hsa.subprocess, "run", _raise_timeout)
    monkeypatch.setenv("FLASH_ATTN_HSA_UNPACKED_DIRECT_CHILD_TIMEOUT_S", "42")

    result = benchmark_hsa._run_unpacked_direct_case_subprocess(case)

    assert result["case"] == "long-1M"
    assert result["status"] == "child_timeout_42s"


def test_benchmark_hsa_unpacked_direct_forward_only_case_summary(monkeypatch):
    benchmark_hsa = _load_benchmark_hsa_module()
    case = next(candidate for candidate in benchmark_hsa.ALL_CASES if candidate.name == "long-64k")
    q_data = torch.zeros(1, 1, 1, 1)
    k_data = torch.zeros(1, 1, 1, 1)
    v_data = torch.zeros(1, 1, 1, 1)

    monkeypatch.setattr(
        benchmark_hsa,
        "_build_case_tensors",
        lambda _: ("schedule", "keep_ids", "hash_ids", q_data, k_data, v_data),
    )

    def _fake_measure_forward(_forward_fn, _q, _k, _v, _warmup_iters, _benchmark_iters, *, env_updates=None):
        env_updates = env_updates or {}
        if env_updates.get("FLASH_ATTN_HSA_UNPACKED_DIRECT_FWD") == "1":
            return 7.5
        return 9.0

    monkeypatch.setattr(benchmark_hsa, "_measure_forward_ms", _fake_measure_forward)
    monkeypatch.setattr(
        benchmark_hsa,
        "_run_sparse_attention",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("correctness should be skipped for long-64k")),
    )

    result = benchmark_hsa._benchmark_unpacked_direct_forward_only_case(case)

    assert result["case"] == "long-64k"
    assert result["plain_sparse_fwd_ms"] == pytest.approx(9.0)
    assert result["unpacked_direct_fwd_ms"] == pytest.approx(7.5)
    assert result["speedup"] == pytest.approx(1.2)
    assert result["plain_sparse_bwd_ms"] is None
    assert result["unpacked_direct_bwd_ms"] is None
    assert result["correctness_status"] == "skipped_long_gt_32k"
    assert result["mode"] == "forward_only"


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
@pytest.mark.parametrize("case_name", ["mixed-small", "train-eq"])
def test_external_hdt_benchmark_subprocess_smoke(case_name):
    benchmark_hsa = _load_benchmark_hsa_module()
    case = next(case for case in benchmark_hsa.ALL_CASES if case.name == case_name)
    result = benchmark_hsa._run_external_hdt_case_subprocess(case)
    status = str(result["status"])
    assert not status.startswith("child_"), status


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
def test_hsa_synthetic_2d_masked_fwd_kernel_smoke():
    import flash_attn.cute.flash_hsa_synthetic_grid_sm100 as synthetic_module

    case = build_explicit_2d_sparse_case(
        case_family="disjoint_confetti",
        seqlen=16,
        heads=4,
        head_dim=64,
        packed_q=8,
        support_k=32,
        islands_per_row=2,
        island_width=2,
        row_shift=5,
        device="cuda",
        dtype=torch.bfloat16,
        seed=0,
    )
    bucket = case["full_bucket"]
    out, lse = synthetic_module._run_synthetic_2d_masked_fwd_kernel(
        bucket["custom_q_buf"],
        bucket["custom_k_buf"],
        bucket["custom_v_buf"],
        bucket["custom_q_length"],
        bucket["custom_k_length"],
        bucket["custom_mask_words"],
        softmax_scale=1.0 / math.sqrt(64.0),
    )

    assert out.shape == (2, 8, 4, 64)
    assert lse.shape == (2, 8, 4)
    assert torch.isfinite(out).all()
    assert torch.isfinite(lse).logical_or(torch.isneginf(lse)).all()


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
@pytest.mark.parametrize("case_family", ["disjoint_confetti", "compact_control"])
def test_hsa_explicit_2d_sparse_variants_match_dense_oracle(case_family):
    report = analyze_explicit_2d_sparse_forward(
        case_family=case_family,
        seqlen=16,
        heads=4,
        head_dim=64,
        packed_q=8,
        support_k=32,
        islands_per_row=2,
        island_width=2,
        row_shift=5,
        warmup_iters=0,
        benchmark_iters=1,
        variants=("dense", "custom_masked", "fa4_packed", "direct_2d"),
        device="cuda",
        dtype=torch.bfloat16,
        seed=0,
    )

    assert report["results"]["custom_masked"]["status"] == "measured"
    assert report["results"]["fa4_packed"]["status"] == "measured"
    assert report["results"]["direct_2d"]["status"] == "measured"
    assert report["results"]["custom_masked"]["output_max_diff"] < 0.1
    assert report["results"]["fa4_packed"]["output_max_diff"] < 0.1
    assert report["results"]["direct_2d"]["output_max_diff"] < 0.1
