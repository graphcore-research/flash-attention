import importlib.util
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
    )
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - import guard for unsupported envs
    build_hsa_schedule = None
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


def _assert_finite_gradients(name, *grads):
    for grad in grads:
        assert grad is not None, f"{name}: missing gradient"
        assert torch.isfinite(grad).all(), f"{name}: non-finite gradient"


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
@pytest.mark.parametrize("case_name", ["mixed-small", "train-eq"])
def test_external_hdt_benchmark_subprocess_smoke(case_name):
    benchmark_hsa = _load_benchmark_hsa_module()
    case = next(case for case in benchmark_hsa.ALL_CASES if case.name == case_name)
    result = benchmark_hsa._run_external_hdt_case_subprocess(case)
    status = str(result["status"])
    assert not status.startswith("child_"), status
