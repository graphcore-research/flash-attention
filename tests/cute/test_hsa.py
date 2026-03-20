import pytest
import torch


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


HAS_HSA_FA4 = (
    _IMPORT_ERROR is None
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] >= 9
)

pytestmark = pytest.mark.skipif(
    not HAS_HSA_FA4,
    reason=f"HSA FA4 test requires CUDA SM90+ with FA4 available ({_IMPORT_ERROR!r})",
)

HAS_HSA_SPARSE_FA4 = HAS_HSA_FA4 and torch.cuda.get_device_capability()[0] >= 10


def _unwrap_output(result):
    return result[0] if isinstance(result, tuple) else result


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
        hsa_module,
        "_run_varlen_fa4_bwd",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy varlen backward helper used")),
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
        hsa_module,
        "_run_varlen_fa4_bwd",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy varlen backward helper used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_bwd_monolithic_main_cute",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("monolithic mixed-family kernel used")),
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
        hsa_module,
        "_run_varlen_fa4_bwd",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy varlen backward helper used")),
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


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
@pytest.mark.parametrize("n_kv_heads", [4, 2])
def test_hsa_monolithic_anchor_full_family_uses_mma_and_matches_reference(monkeypatch, n_kv_heads):
    import flash_attn.cute.hsa as hsa_module
    import flash_attn.cute.flash_hsa_bwd_sm100 as hsa_bwd_module

    device = "cuda"
    dtype = torch.bfloat16
    batch_size, seqlen, nheads, headdim = 1, 65, 4, 64
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
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
        hsa_module,
        "_run_varlen_fa4_bwd",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy varlen backward helper used")),
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
        "_run_hsa_bwd_anchor_full_scalar",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("anchor_full scalar helper used")),
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

    _assert_close(out_sparse.float(), out_ref, "hsa_monolithic_anchor_full_output")
    _assert_close(q_sparse.grad.float(), q_ref.grad, "hsa_monolithic_anchor_full_q_grad", atol=6e-2, rtol=6e-2)
    _assert_close(k_sparse.grad.float(), k_ref.grad, "hsa_monolithic_anchor_full_k_grad", atol=6e-2, rtol=6e-2)
    _assert_close(v_sparse.grad.float(), v_ref.grad, "hsa_monolithic_anchor_full_v_grad", atol=6e-2, rtol=6e-2)


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
@pytest.mark.parametrize("n_kv_heads", [4, 2])
def test_hsa_monolithic_anchor_tail_family_uses_mma_and_matches_reference(monkeypatch, n_kv_heads):
    import flash_attn.cute.hsa as hsa_module
    import flash_attn.cute.flash_hsa_bwd_sm100 as hsa_bwd_module

    device = "cuda"
    dtype = torch.bfloat16
    batch_size, seqlen, nheads, headdim = 1, 65, 4, 64
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
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
        hsa_module,
        "_run_varlen_fa4_bwd",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy varlen backward helper used")),
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
        "_run_hsa_bwd_anchor_tail_scalar",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("anchor_tail scalar helper used")),
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

    _assert_close(out_sparse.float(), out_ref, "hsa_monolithic_anchor_tail_output")
    _assert_close(q_sparse.grad.float(), q_ref.grad, "hsa_monolithic_anchor_tail_q_grad", atol=6e-2, rtol=6e-2)
    _assert_close(k_sparse.grad.float(), k_ref.grad, "hsa_monolithic_anchor_tail_k_grad", atol=6e-2, rtol=6e-2)
    _assert_close(v_sparse.grad.float(), v_ref.grad, "hsa_monolithic_anchor_tail_v_grad", atol=6e-2, rtol=6e-2)


@pytest.mark.skipif(not HAS_HSA_SPARSE_FA4, reason="Scheduled sparse HSA path requires CUDA SM100+")
@pytest.mark.parametrize("n_kv_heads", [4, 2])
def test_hsa_monolithic_backward_matches_reference(monkeypatch, n_kv_heads):
    import flash_attn.cute.hsa as hsa_module
    import flash_attn.cute.flash_hsa_bwd_sm100 as hsa_bwd_module

    device = "cuda"
    dtype = torch.bfloat16
    batch_size, seqlen, nheads, headdim = 1, 65, 4, 64
    keep_ids, hash_ids = _make_hsa_metadata(batch_size, seqlen, device)
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
        hsa_module,
        "_run_varlen_fa4_bwd",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy varlen backward helper used")),
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
        "_run_hsa_monolithic_mma_batches",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("host-side monolithic MMA prepass used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_bwd_anchor_full_scalar",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("anchor_full scalar helper used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_bwd_anchor_tail_scalar",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("anchor_tail scalar helper used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_anchor_full_mma_batches",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("anchor_full family prepass used")),
    )
    monkeypatch.setattr(
        hsa_bwd_module,
        "_run_hsa_anchor_tail_mma_batches",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("anchor_tail family prepass used")),
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

    _assert_close(out_sparse.float(), out_ref, "hsa_monolithic_backward_output")
    _assert_close(q_sparse.grad.float(), q_ref.grad, "hsa_monolithic_q_grad", atol=6e-2, rtol=6e-2)
    _assert_close(k_sparse.grad.float(), k_ref.grad, "hsa_monolithic_k_grad", atol=6e-2, rtol=6e-2)
    _assert_close(v_sparse.grad.float(), v_ref.grad, "hsa_monolithic_v_grad", atol=6e-2, rtol=6e-2)
