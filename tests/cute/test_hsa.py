import pytest
import torch


try:
    from flash_attn.cute import (
        flash_attn_func,
        flash_attn_hsa_func,
        get_hsa_mask_mod,
        hsa_reference_attention,
    )
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - import guard for unsupported envs
    flash_attn_func = None
    flash_attn_hsa_func = None
    get_hsa_mask_mod = None
    hsa_reference_attention = None
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


def _unwrap_output(result):
    return result[0] if isinstance(result, tuple) else result


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
