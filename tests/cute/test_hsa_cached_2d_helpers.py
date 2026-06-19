import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import flash_attn.cute.hsa_cached_2d_forward_analysis as cached_2d


def test_runtime_payload_cache_reuses_runtime_dict():
    class Runtime:
        pass

    runtime = Runtime()
    cache = cached_2d._runtime_payload_cache(runtime, "_payload_cache")
    assert isinstance(cache, dict)
    cache["key"] = {"status": "ready"}
    assert cached_2d._runtime_payload_cache(runtime, "_payload_cache") is cache

    runtime._payload_cache = []
    assert cached_2d._runtime_payload_cache(runtime, "_payload_cache") is None


def test_cached_payload_shape_key_separates_shape_policy_layout_and_plan():
    q = torch.empty((4, 2, 64), dtype=torch.bfloat16)
    k = torch.empty((5, 2, 64), dtype=torch.bfloat16)
    v = torch.empty((5, 2, 64), dtype=torch.bfloat16)
    direct_plan = {}
    row_plan = {}
    policy = (("max_rows_per_group", 16),)

    base = cached_2d._cached_payload_shape_key(
        q_flat=q,
        k_flat=k,
        v_flat=v,
        direct_plan=direct_plan,
        row_plan=row_plan,
        policy_key=policy,
        include_mask_bool=False,
    )

    assert base == cached_2d._cached_payload_shape_key(
        q_flat=q,
        k_flat=k,
        v_flat=v,
        direct_plan=direct_plan,
        row_plan=row_plan,
        policy_key=policy,
        include_mask_bool=False,
    )
    assert base != cached_2d._cached_payload_shape_key(
        q_flat=torch.empty((6, 2, 64), dtype=torch.bfloat16),
        k_flat=k,
        v_flat=v,
        direct_plan=direct_plan,
        row_plan=row_plan,
        policy_key=policy,
        include_mask_bool=False,
    )
    assert base != cached_2d._cached_payload_shape_key(
        q_flat=q,
        k_flat=k,
        v_flat=v,
        direct_plan=direct_plan,
        row_plan=row_plan,
        policy_key=(("max_rows_per_group", 8),),
        include_mask_bool=False,
    )
    assert base != cached_2d._cached_payload_shape_key(
        q_flat=q,
        k_flat=k,
        v_flat=v,
        direct_plan=direct_plan,
        row_plan=row_plan,
        policy_key=policy,
        include_mask_bool=True,
    )
    assert base != cached_2d._cached_payload_shape_key(
        q_flat=q,
        k_flat=k,
        v_flat=v,
        direct_plan={},
        row_plan=row_plan,
        policy_key=policy,
        include_mask_bool=False,
    )


def test_cached_lse_public_to_flat_triton_gate_rejects_cpu():
    q = torch.empty((2, 16, 8, 64), dtype=torch.bfloat16)
    q_flat = q.reshape(-1, 8, 64)
    lse = torch.empty((2, 8, 16), dtype=torch.float32)

    assert not cached_2d._can_use_triton_lse_public_to_flat(q, q_flat, lse)


def test_cached_lse_public_to_flat_triton_matches_pytorch_and_reuses_buffer(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton public-LSE-to-flat test")
    if not cached_2d._HAS_TRITON_LSE_PUBLIC_TO_FLAT:
        pytest.skip("Triton unavailable")

    monkeypatch.delenv("FLASH_ATTN_HSA_CACHED_BWD_TRITON_LSE_PUBLIC_TO_FLAT", raising=False)
    q = torch.empty((2, 128, 8, 64), dtype=torch.bfloat16, device="cuda")
    q_flat = q.reshape(-1, 8, 64)
    lse = torch.randn((2, 8, 128), dtype=torch.float32, device="cuda")
    ref = lse.permute(0, 2, 1).contiguous().view(-1, 8).float()

    payload = {}
    lse_flat = cached_2d._get_cached_lse_flat_for_backward_buffer(payload, q_flat, 8)
    lse_flat_again = cached_2d._get_cached_lse_flat_for_backward_buffer(payload, q_flat, 8)
    assert lse_flat_again.data_ptr() == lse_flat.data_ptr()

    got = cached_2d._flatten_cached_lse_for_backward(q, q_flat, lse, out=lse_flat)
    torch.cuda.synchronize()

    assert got.data_ptr() == lse_flat.data_ptr()
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_cached_packing_policy_allows_wide_2d_union_but_not_wide_direct():
    assert cached_2d.CachedPackingPolicy().max_union_k_2d == 1024

    policy = cached_2d._coerce_cached_packing_policy(
        cached_2d.CachedPackingPolicy(max_union_k_direct=128, max_union_k_2d=1024)
    )
    assert policy.max_union_k_2d == 1024

    try:
        cached_2d._coerce_cached_packing_policy(
            cached_2d.CachedPackingPolicy(max_union_k_direct=256, max_union_k_2d=512)
        )
    except ValueError as exc:
        assert "max_union_k_direct" in str(exc)
    else:
        raise AssertionError("expected max_union_k_direct >128 to be rejected")


def test_synthetic_2d_masked_fwd_gate_allows_k2048_d64_and_d128(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for direct 2D gate test")

    from flash_attn.cute.flash_hsa_synthetic_grid_sm100 import _can_use_synthetic_2d_masked_fwd

    monkeypatch.delenv("FLASH_ATTN_HSA_CACHED_GATHER_MAX_PACKED_K", raising=False)
    monkeypatch.delenv("FLASH_ATTN_HSA_CACHED_GATHER_D128", raising=False)
    q64 = torch.empty((1, 8, 64), dtype=torch.bfloat16, device="cuda")
    assert _can_use_synthetic_2d_masked_fwd(q64, q64, q64, packed_q=16, packed_k=2048)
    assert not _can_use_synthetic_2d_masked_fwd(q64, q64, q64, packed_q=16, packed_k=2049)

    q128 = torch.empty((1, 8, 128), dtype=torch.bfloat16, device="cuda")
    assert _can_use_synthetic_2d_masked_fwd(q128, q128, q128, packed_q=16, packed_k=2048)

    monkeypatch.setenv("FLASH_ATTN_HSA_CACHED_GATHER_D128", "0")
    assert not _can_use_synthetic_2d_masked_fwd(q128, q128, q128, packed_q=16, packed_k=2048)


def test_synthetic_2d_masked_gather_d128_matches_dense_partial_mask(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for D128 direct 2D correctness test")

    from flash_attn.cute.flash_hsa_synthetic_grid_sm100 import _run_synthetic_2d_masked_gather_fwd_kernel

    monkeypatch.delenv("FLASH_ATTN_HSA_CACHED_GATHER_D128", raising=False)
    torch.manual_seed(123)
    groups, packed_q, packed_k, num_heads, head_dim = 1, 4, 65, 2, 128
    softmax_scale = head_dim**-0.5
    q_rows = torch.randn(groups * packed_q, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    k_rows = torch.randn(groups * packed_k, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    v_rows = torch.randn(groups * packed_k, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    q_row_idx = torch.arange(groups * packed_q, dtype=torch.int32, device="cuda").view(groups, packed_q)
    k_row_idx = torch.arange(groups * packed_k, dtype=torch.int32, device="cuda").view(groups, packed_k)
    q_length = torch.full((groups,), packed_q, dtype=torch.int32, device="cuda")
    k_length = torch.full((groups,), packed_k, dtype=torch.int32, device="cuda")
    mask_words_cpu = torch.zeros((groups, packed_q, (packed_k + 31) // 32), dtype=torch.int32)
    for q_idx in range(packed_q):
        for k_idx in range(packed_k):
            if (k_idx + 3 * q_idx) % 5 != 0 and (k_idx < 60 or k_idx % 2 == 1):
                word_idx, bit_idx = divmod(k_idx, 32)
                mask_words_cpu[0, q_idx, word_idx] |= 1 << bit_idx
    mask_words = mask_words_cpu.to("cuda")

    out, lse = _run_synthetic_2d_masked_gather_fwd_kernel(
        q_rows,
        k_rows,
        v_rows,
        q_row_idx,
        k_row_idx,
        q_length,
        k_length,
        mask_words,
        softmax_scale=softmax_scale,
    )
    torch.cuda.synchronize()

    ref_out = torch.zeros_like(out)
    ref_lse = torch.empty_like(lse)
    for q_idx in range(packed_q):
        active_keys = []
        for k_idx in range(packed_k):
            word_idx, bit_idx = divmod(k_idx, 32)
            if int(mask_words_cpu[0, q_idx, word_idx]) & (1 << bit_idx):
                active_keys.append(k_idx)
        for head_idx in range(num_heads):
            scores = (
                q_rows[q_idx, head_idx].float()
                @ k_rows[torch.tensor(active_keys, device="cuda"), head_idx].float().T
            ) * softmax_scale
            probs = torch.softmax(scores, dim=-1)
            ref_out[0, q_idx, head_idx] = probs @ v_rows[torch.tensor(active_keys, device="cuda"), head_idx].float()
            ref_lse[0, q_idx, head_idx] = torch.logsumexp(scores, dim=-1)

    torch.testing.assert_close(out, ref_out, rtol=0, atol=2e-5)
    torch.testing.assert_close(lse, ref_lse, rtol=0, atol=2e-5)


def test_cached_backward_key_owned_dkdv_gate_requires_occurrence_payload():
    assert not cached_2d._can_use_cached_backward_key_owned_dkdv(None)

    payload = {
        "status": "ready",
        "backward_kernel_family": "cached_tc8x8_fused",
        "owned_k_row_idx": torch.tensor([0], dtype=torch.int32),
        "owned_occurrence_ptr": torch.tensor([0, 1], dtype=torch.int32),
        "owned_occurrence_kind": torch.tensor([0], dtype=torch.int32),
        "owned_occurrence_range_idx": torch.tensor([0], dtype=torch.int32),
        "owned_occurrence_tile_idx": torch.tensor([0], dtype=torch.int32),
        "owned_occurrence_col_idx": torch.tensor([0], dtype=torch.int32),
    }

    assert cached_2d._can_use_cached_backward_key_owned_dkdv(payload)
    payload["owned_k_row_idx"] = torch.empty((0,), dtype=torch.int32)
    assert not cached_2d._can_use_cached_backward_key_owned_dkdv(payload)


def test_cached_backward_key_owned_overwrite_gate_requires_all_kv_rows():
    payload = {
        "status": "ready",
        "backward_kernel_family": "cached_tc8x8_fused",
        "owned_k_row_idx": torch.tensor([0, 1, 2], dtype=torch.int32),
        "owned_occurrence_ptr": torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        "owned_occurrence_kind": torch.tensor([0, 0, 1], dtype=torch.int32),
        "owned_occurrence_range_idx": torch.tensor([0, 0, 1], dtype=torch.int32),
        "owned_occurrence_tile_idx": torch.tensor([0, 1, 2], dtype=torch.int32),
        "owned_occurrence_col_idx": torch.tensor([0, 1, 2], dtype=torch.int32),
    }
    k_flat = torch.empty((3, 2, 64), dtype=torch.bfloat16)
    assert cached_2d._cached_backward_key_owned_overwrites_all_kv_rows(payload, k_flat)

    k_flat = torch.empty((4, 2, 64), dtype=torch.bfloat16)
    assert not cached_2d._cached_backward_key_owned_overwrites_all_kv_rows(payload, k_flat)


def test_cached_backward_key_owned_auto_gate_is_small_all_owned_only(monkeypatch):
    payload = {
        "status": "ready",
        "backward_kernel_family": "cached_tc8x8_fused",
        "owned_k_row_idx": torch.tensor([0, 1, 2], dtype=torch.int32),
        "owned_occurrence_ptr": torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        "owned_occurrence_kind": torch.tensor([0, 0, 1], dtype=torch.int32),
        "owned_occurrence_range_idx": torch.tensor([0, 0, 1], dtype=torch.int32),
        "owned_occurrence_tile_idx": torch.tensor([0, 1, 2], dtype=torch.int32),
        "owned_occurrence_col_idx": torch.tensor([0, 1, 2], dtype=torch.int32),
    }
    monkeypatch.setenv("FLASH_ATTN_HSA_CACHED_GENERALIZED_BWD_KEY_OWNED_MAX_ROWS", "3")
    assert cached_2d._auto_use_cached_backward_key_owned_dkdv(
        payload,
        torch.empty((3, 2, 64), dtype=torch.bfloat16),
    )
    assert not cached_2d._auto_use_cached_backward_key_owned_dkdv(
        payload,
        torch.empty((4, 2, 64), dtype=torch.bfloat16),
    )

    monkeypatch.delenv("FLASH_ATTN_HSA_CACHED_GENERALIZED_BWD_KEY_OWNED_MAX_ROWS", raising=False)
    payload["owned_k_row_idx"] = torch.arange(128, dtype=torch.int32)
    payload["owned_occurrence_ptr"] = torch.arange(129, dtype=torch.int32)
    payload["owned_occurrence_kind"] = torch.zeros((128,), dtype=torch.int32)
    payload["owned_occurrence_range_idx"] = torch.zeros((128,), dtype=torch.int32)
    payload["owned_occurrence_tile_idx"] = torch.zeros((128,), dtype=torch.int32)
    payload["owned_occurrence_col_idx"] = torch.zeros((128,), dtype=torch.int32)
    assert cached_2d._auto_use_cached_backward_key_owned_dkdv(
        payload,
        torch.empty((128, 2, 64), dtype=torch.bfloat16),
    )
    assert not cached_2d._auto_use_cached_backward_key_owned_dkdv(
        payload,
        torch.empty((129, 2, 64), dtype=torch.bfloat16),
    )


def test_cached_backward_direct_dq_auto_gate_uses_row_threshold(monkeypatch):
    payload = {
        "total_rows": 4096,
        "residual_mode": "fused_tail",
        "fused_output_row_count": 4096,
        "fused_q_row_idx": torch.empty((512, 8), dtype=torch.int32),
        "fused_q_length": torch.full((512,), 8, dtype=torch.int32),
    }

    class FakeCudaTensor:
        is_cuda = True
        shape = (4096, 8, 64)

    monkeypatch.delenv("FLASH_ATTN_HSA_CACHED_DIRECT_DQ_BWD", raising=False)
    monkeypatch.delenv("FLASH_ATTN_HSA_CACHED_DIRECT_DQ_BWD_TINY_ROWS", raising=False)
    monkeypatch.setenv("FLASH_ATTN_HSA_CACHED_DIRECT_DQ_BWD_MIN_ROWS", "4096")
    assert cached_2d._use_cached_backward_direct_dq(payload, FakeCudaTensor())

    monkeypatch.setenv("FLASH_ATTN_HSA_CACHED_DIRECT_DQ_BWD_MIN_ROWS", "4097")
    assert not cached_2d._use_cached_backward_direct_dq(payload, FakeCudaTensor())

    monkeypatch.setenv("FLASH_ATTN_HSA_CACHED_DIRECT_DQ_BWD", "1")
    assert cached_2d._use_cached_backward_direct_dq(payload, FakeCudaTensor())

    monkeypatch.setenv("FLASH_ATTN_HSA_CACHED_DIRECT_DQ_BWD", "0")
    assert not cached_2d._use_cached_backward_direct_dq(payload, FakeCudaTensor())

    monkeypatch.delenv("FLASH_ATTN_HSA_CACHED_DIRECT_DQ_BWD", raising=False)
    payload["total_rows"] = 65
    payload["fused_output_row_count"] = 65
    payload["fused_q_row_idx"] = torch.arange(65, dtype=torch.int32).view(13, 5)
    payload["fused_q_length"] = torch.full((13,), 5, dtype=torch.int32)

    class FakeTinyCudaTensor:
        is_cuda = True
        shape = (65, 8, 64)

    monkeypatch.setenv("FLASH_ATTN_HSA_CACHED_DIRECT_DQ_BWD_TINY_ROWS", "128")
    assert cached_2d._use_cached_backward_direct_dq(payload, FakeTinyCudaTensor())

    monkeypatch.setenv("FLASH_ATTN_HSA_CACHED_DIRECT_DQ_BWD_TINY_ROWS", "64")
    assert not cached_2d._use_cached_backward_direct_dq(payload, FakeTinyCudaTensor())


def test_cached_backward_payload_empty_exact_local_k_is_rank2():
    payload = {
        "status": "ready",
        "residual_mode": "fused_tail",
        "exact_kernel_family": "tc8x8",
        "exact_dense_rows_per_range": 8,
        "exact_dense_keys_per_tile": 8,
        "fused_q_row_idx": torch.tensor([[0, 1, -1, -1, -1, -1, -1, -1]], dtype=torch.int32),
        "fused_q_length": torch.tensor([2], dtype=torch.int32),
        "fused_exact_tile_ptr": torch.tensor([0, 0], dtype=torch.int32),
        "fused_exact_k_row_idx": torch.empty((0, 8), dtype=torch.int32),
        "fused_tail_tile_ptr": torch.tensor([0, 1], dtype=torch.int32),
        "fused_tail_k_row_idx": torch.tensor([[0, 1, 2, -1, -1, -1, -1, -1]], dtype=torch.int32),
        "fused_tail_mask_words": torch.tensor([[[0b111], [0b011]]], dtype=torch.int32),
    }

    backward_payload = cached_2d.build_cached_generalized_backward_payload(payload)

    assert backward_payload is not None
    assert tuple(backward_payload["exact_tile_local_k_idx"].shape) == (0, 8)
    assert tuple(backward_payload["tail_tile_local_k_idx"].shape) == (1, 8)


def test_cached_masked_payload_force_combine_for_grouped_scatter(monkeypatch):
    calls = []

    def combine(*args, **kwargs):
        calls.append("combine")

    def scatter_tc(*args, **kwargs):
        calls.append("scatter_tc")

    def scatter(*args, **kwargs):
        calls.append("scatter")

    monkeypatch.setattr(cached_2d, "_run_synthetic_2d_masked_gather_combine_fwd_kernel", combine)
    monkeypatch.setattr(cached_2d, "_run_synthetic_2d_masked_gather_scatter_tc_fwd_kernel", scatter_tc)
    monkeypatch.setattr(cached_2d, "_run_synthetic_2d_masked_gather_scatter_fwd_kernel", scatter)

    payload = {
        "packed_q": 2,
        "support_rows": 4,
        "tile_k": 32,
        "q_row_idx": torch.empty((2, 2), dtype=torch.int32),
        "range_tc_scatter_q_row_idx": torch.tensor([[0, 1]], dtype=torch.int32),
        "range_tc_scatter_k_row_idx": torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
        "range_tc_scatter_q_length": torch.tensor([2], dtype=torch.int32),
        "range_tc_scatter_k_length": torch.tensor([4], dtype=torch.int32),
        "range_tc_scatter_mask_words": torch.tensor([[[0b1111], [0b1111]]], dtype=torch.int32),
        "range_tc_scatter_group_count": 1,
        "range_tc_scatter_row_count": 2,
        "range_scatter_q_row_idx": torch.tensor([[2, 3]], dtype=torch.int32),
        "range_scatter_k_row_idx": torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
        "range_scatter_q_length": torch.tensor([2], dtype=torch.int32),
        "range_scatter_k_length": torch.tensor([4], dtype=torch.int32),
        "range_scatter_mask_words": torch.tensor([[[0b1111], [0b1111]]], dtype=torch.int32),
        "range_scatter_union_group_count": 1,
        "range_scatter_union_row_count": 2,
        "range_packed_q_row_idx": torch.empty((0, 2), dtype=torch.int32),
        "range_packed_k_row_idx": torch.empty((0, 4), dtype=torch.int32),
        "range_packed_q_length": torch.empty((0,), dtype=torch.int32),
        "range_packed_k_length": torch.empty((0,), dtype=torch.int32),
        "range_packed_mask_words": torch.empty((0, 2, 1), dtype=torch.int32),
        "range_packed_q_row_idx_flat": torch.empty((0,), dtype=torch.int32),
        "range_packed_k_row_idx_flat": torch.empty((0,), dtype=torch.int32),
    }
    q = torch.empty((4, 1, 64), dtype=torch.bfloat16)
    k = torch.empty((4, 1, 64), dtype=torch.bfloat16)
    v = torch.empty((4, 1, 64), dtype=torch.bfloat16)
    out = torch.empty((4, 1, 64), dtype=torch.float32)
    lse = torch.empty((4, 1), dtype=torch.float32)

    cached_2d._run_cached_masked_payload_forward(
        payload,
        q,
        k,
        v,
        out,
        lse,
        softmax_scale=1.0,
        force_combine_scatter=False,
    )
    assert calls == ["scatter_tc", "scatter"]

    calls.clear()
    cached_2d._run_cached_masked_payload_forward(
        payload,
        q,
        k,
        v,
        out,
        lse,
        softmax_scale=1.0,
        force_combine_scatter=True,
    )
    assert calls == ["combine", "combine"]


def test_cached_2d_range_annotation_materializes_kernel_descriptors(monkeypatch):
    q = torch.empty((4, 2, 64), dtype=torch.bfloat16)
    range_execution = [
        {"group_start": 0, "group_end": 2, "scatter_only": True, "family": "union_2d"},
        {"group_start": 2, "group_end": 3, "scatter_only": True, "family": "k_window"},
        {"group_start": 3, "group_end": 5, "scatter_only": False, "family": "union_2d"},
    ]

    monkeypatch.setattr(cached_2d, "_can_use_synthetic_2d_masked_fwd", lambda *args, **kwargs: True)

    annotated = cached_2d._annotate_range_execution_kernels(
        range_execution,
        rows_per_group=16,
        max_union_k=64,
        tile_k=32,
        union_kernel="tc16x32",
        q_flat=q,
        k_flat=q,
        v_flat=q,
    )
    descriptors = cached_2d._materialize_range_execution_tensors(annotated, device=q.device)

    assert [entry["kernel_kind"] for entry in annotated] == ["tc_scatter", "scatter", "packed"]
    assert descriptors["range_group_start"].tolist() == [0, 2, 3]
    assert descriptors["range_group_end"].tolist() == [2, 3, 5]
    assert descriptors["range_scatter_only"].tolist() == [1, 1, 0]
    assert descriptors["range_kernel_kind"].tolist() == [2, 1, 0]


def test_cached_2d_flat_row_idx_uses_precomputed_payload_slice():
    payload = {
        "q_row_idx": torch.tensor([[0, 1], [2, -1], [3, 4]], dtype=torch.int32),
        "q_row_idx_flat": torch.tensor([0, 1, 2, -1, 3, 4], dtype=torch.int32),
    }

    row_idx = cached_2d._slice_cached_flat_row_idx(
        payload,
        flat_key="q_row_idx_flat",
        matrix_key="q_row_idx",
        group_start=1,
        group_end=3,
        width=2,
    )

    assert row_idx.tolist() == [2, -1, 3, 4]


def test_attach_precomputed_cached_payload_preserves_container_and_clears_caches():
    class Schedule:
        pass

    schedule = Schedule()
    schedule._precomputed_forward_direct_plan_payload = {
        "version": 7,
        "entries": [
            {
                "forward_block_q": 128,
                "logical_block_q": 64,
                "logical_block_k": 64,
                "max_packed_k": 128,
                "max_direct_segments": 4,
                "direct_execution_plan": {"kept": True},
            }
        ],
    }
    schedule._precomputed_cached_generalized_forward_payload_device_cache = {"stale": object()}
    schedule._resolved_cached_generalized_forward_payload_fast_cache = {"stale": object()}
    schedule._resolved_cached_generalized_forward_payload_cache = {"stale": object()}
    cached_payload = {"status": "ready", "reason": "unit"}

    returned = cached_2d.attach_precomputed_cached_generalized_forward_payload(
        schedule,
        cached_payload,
        forward_block_q=128,
        logical_block_q=64,
        logical_block_k=64,
        max_packed_k=128,
        max_direct_segments=4,
    )

    assert returned is schedule
    container = schedule._precomputed_forward_direct_plan_payload
    assert container["version"] == 7
    assert len(container["entries"]) == 1
    assert container["entries"][0]["direct_execution_plan"] == {"kept": True}
    assert container["entries"][0]["cached_generalized_forward_payload"] is cached_payload
    assert not hasattr(schedule, "_precomputed_cached_generalized_forward_payload_device_cache")
    assert not hasattr(schedule, "_resolved_cached_generalized_forward_payload_fast_cache")
    assert not hasattr(schedule, "_resolved_cached_generalized_forward_payload_cache")


def test_cached_2d_monolithic_forward_requires_complete_fused_tail_payload():
    q = torch.empty((4, 2, 64), dtype=torch.bfloat16)
    payload = {
        "total_rows": 4,
        "residual_mode": "fused_tail",
        "exact_kernel_family": "tc8x8",
        "exact_dense_rows_per_range": 8,
        "exact_dense_keys_per_tile": 8,
        "fused_q_row_idx": torch.tensor([[0, 1, 2, 3, 4, 5, -1, -1]], dtype=torch.int32),
        "q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "fused_output_row_count": 3,
        "geometry": {"fused_total_coverage_frac": 1.0},
    }

    assert cached_2d._cached_monolithic_forward_support_reason(payload, q, q, q) == "requires_cuda"

    class FakeCudaTensor:
        is_cuda = True
        dtype = torch.bfloat16
        shape = (4, 2, 64)

    assert (
        cached_2d._cached_monolithic_forward_support_reason(payload, FakeCudaTensor(), FakeCudaTensor(), FakeCudaTensor())
        == "incomplete_fused_output_row_coverage"
    )

    payload["fused_output_row_count"] = 4
    assert cached_2d._cached_monolithic_forward_support_reason(payload, FakeCudaTensor(), FakeCudaTensor(), FakeCudaTensor()) is None


def test_cached_2d_direct_final_residual_allows_scatter_only_full_coverage():
    payload = {
        "total_rows": 6,
        "residual_mode": "fused_tail",
        "exact_kernel_family": "tc8x8",
        "exact_dense_rows_per_range": 8,
        "exact_dense_keys_per_tile": 8,
        "fused_q_row_idx": torch.tensor([[0, 1, 2, 3, 4, 5, -1, -1]], dtype=torch.int32),
        "q_row_idx": torch.empty((1, 16), dtype=torch.int32),
        "fused_output_row_count": 4,
        "range_tc_scatter_row_count": 1,
        "range_scatter_row_count": 1,
        "range_packed_group_count": 0,
        "range_tc_scatter_q_row_idx": torch.empty((1, 16), dtype=torch.int32),
        "range_scatter_q_row_idx": torch.empty((1, 16), dtype=torch.int32),
        "range_packed_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "geometry": {"fused_total_coverage_frac": 0.8},
    }

    class FakeCudaTensor:
        is_cuda = True
        dtype = torch.bfloat16
        shape = (6, 2, 64)

    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        is None
    )


def test_cached_2d_direct_final_residual_allows_exact_dense_base_coverage():
    payload = {
        "total_rows": 6,
        "residual_mode": "fused_tail",
        "exact_kernel_family": "tc8x8",
        "exact_dense_rows_per_range": 8,
        "exact_dense_keys_per_tile": 8,
        "fused_q_row_idx": torch.empty((0, 8), dtype=torch.int32),
        "q_row_idx": torch.empty((1, 16), dtype=torch.int32),
        "fused_output_row_count": 0,
        "exact_dense_output_row_count": 4,
        "range_tc_scatter_row_count": 1,
        "range_scatter_row_count": 1,
        "range_packed_group_count": 0,
        "range_tc_scatter_q_row_idx": torch.empty((1, 16), dtype=torch.int32),
        "range_scatter_q_row_idx": torch.empty((1, 16), dtype=torch.int32),
        "range_packed_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "geometry": {"fused_total_coverage_frac": 0.0},
    }

    class FakeCudaTensor:
        is_cuda = True
        dtype = torch.bfloat16
        shape = (6, 2, 64)

    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        is None
    )


def test_cached_2d_direct_final_residual_gates_packed_overlap_without_fp32_combine(monkeypatch):
    payload = {
        "total_rows": 6,
        "residual_mode": "fused_tail",
        "exact_kernel_family": "tc8x8",
        "exact_dense_rows_per_range": 8,
        "exact_dense_keys_per_tile": 8,
        "fused_q_row_idx": torch.empty((1, 8), dtype=torch.int32),
        "q_row_idx": torch.empty((1, 16), dtype=torch.int32),
        "fused_output_row_count": 6,
        "range_tc_scatter_row_count": 0,
        "range_scatter_row_count": 0,
        "range_packed_group_count": 1,
        "range_tc_scatter_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "range_scatter_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "range_packed_q_row_idx": torch.tensor([[4, 5, -1, -1, -1, -1, -1, -1]], dtype=torch.int32),
        "geometry": {"fused_total_coverage_frac": 0.8},
    }

    class FakeCudaTensor:
        is_cuda = True
        dtype = torch.bfloat16
        shape = (6, 2, 64)

    monkeypatch.delenv("FLASH_ATTN_HSA_CACHED_DIRECT_FINAL_ONLINE_COMBINE", raising=False)
    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        is None
    )

    monkeypatch.setenv("FLASH_ATTN_HSA_CACHED_DIRECT_FINAL_ONLINE_COMBINE", "0")
    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        == "direct_final_online_combine_requires_fp32_accum"
    )

    monkeypatch.setenv("FLASH_ATTN_HSA_CACHED_DIRECT_FINAL_ONLINE_COMBINE", "1")
    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        is None
    )
    monkeypatch.delenv("FLASH_ATTN_HSA_CACHED_DIRECT_FINAL_ONLINE_COMBINE", raising=False)

    payload["fused_output_row_count"] = 4
    payload["fused_q_row_idx"] = torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=torch.int32)
    payload["range_packed_q_row_idx"] = torch.tensor([[4, 5, -1, -1, -1, -1, -1, -1]], dtype=torch.int32)
    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        is None
    )

    payload["fused_output_row_count"] = 6
    payload["fused_q_row_idx"] = torch.tensor([[0, 1, 2, 3, 4, 5, -1, -1]], dtype=torch.int32)
    payload["range_scatter_row_count"] = 1
    payload["range_scatter_q_row_idx"] = torch.tensor([[4, -1, -1, -1, -1, -1, -1, -1]], dtype=torch.int32)
    monkeypatch.setenv("FLASH_ATTN_HSA_CACHED_DIRECT_FINAL_ONLINE_COMBINE", "0")
    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        == "direct_final_online_combine_requires_fp32_accum"
    )

    payload["fused_output_row_count"] = 4
    payload["fused_q_row_idx"] = torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=torch.int32)
    payload["range_packed_q_row_idx"] = torch.tensor([[3, -1, -1, -1, -1, -1, -1, -1]], dtype=torch.int32)
    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        == "mixed_residual_incomplete_direct_final_row_coverage"
    )


def test_cached_2d_direct_final_residual_initializes_missing_packed_rows():
    payload = {
        "total_rows": 6,
        "residual_mode": "fused_tail",
        "exact_kernel_family": "tc8x8",
        "exact_dense_rows_per_range": 8,
        "exact_dense_keys_per_tile": 8,
        "fused_q_row_idx": torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=torch.int32),
        "q_row_idx": torch.empty((1, 16), dtype=torch.int32),
        "fused_output_row_count": 4,
        "range_tc_scatter_row_count": 0,
        "range_scatter_row_count": 0,
        "range_packed_group_count": 1,
        "range_tc_scatter_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "range_scatter_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "range_packed_q_row_idx": torch.tensor([[4, 5, -1, -1, -1, -1, -1, -1]], dtype=torch.int32),
        "geometry": {"fused_total_coverage_frac": 0.8},
    }

    class FakeCudaTensor:
        is_cuda = True
        dtype = torch.bfloat16
        shape = (6, 2, 64)

    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        is None
    )
    missing = cached_2d._get_direct_final_missing_init_row_idx(payload, torch.device("cpu"))
    assert missing.tolist() == [4, 5]

    payload["fused_q_row_idx"] = torch.tensor([[0, 1, 2, 3, 4, 5, -1, -1]], dtype=torch.int32)
    payload.pop("_workspace", None)
    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        is None
    )
    missing = cached_2d._get_direct_final_missing_init_row_idx(payload, torch.device("cpu"))
    assert missing.tolist() == [4, 5]


def test_cached_2d_direct_final_base_rows_trim_in_payload_order():
    payload = {
        "total_rows": 8,
        "residual_mode": "fused_tail",
        "exact_kernel_family": "tc8x8",
        "exact_dense_rows_per_range": 8,
        "exact_dense_keys_per_tile": 8,
        "fused_q_row_idx": torch.tensor([[4, 5, 6, 7, 0, 1, 2, 3]], dtype=torch.int32),
        "q_row_idx": torch.empty((1, 16), dtype=torch.int32),
        "fused_output_row_count": 4,
        "range_tc_scatter_row_count": 0,
        "range_scatter_row_count": 4,
        "range_packed_group_count": 0,
        "range_tc_scatter_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "range_scatter_q_row_idx": torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=torch.int32),
        "range_scatter_q_length": torch.tensor([4], dtype=torch.int32),
        "range_packed_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "geometry": {"fused_total_coverage_frac": 0.8},
    }

    class FakeCudaTensor:
        is_cuda = True
        dtype = torch.bfloat16
        shape = (8, 2, 64)

    base_rows = cached_2d._get_direct_final_base_output_row_idx(payload, torch.device("cpu"), base_source="fused")
    assert base_rows.tolist() == [4, 5, 6, 7]
    assert cached_2d._direct_final_base_residual_union_row_count(payload, torch.device("cpu")) == 8
    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        is None
    )


def test_cached_2d_direct_final_residual_initializes_missing_online_scatter_rows():
    payload = {
        "total_rows": 6,
        "residual_mode": "fused_tail",
        "exact_kernel_family": "tc8x8",
        "exact_dense_rows_per_range": 8,
        "exact_dense_keys_per_tile": 8,
        "fused_q_row_idx": torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=torch.int32),
        "q_row_idx": torch.empty((1, 16), dtype=torch.int32),
        "fused_output_row_count": 4,
        "range_tc_scatter_row_count": 0,
        "range_scatter_row_count": 3,
        "range_packed_group_count": 0,
        "range_tc_scatter_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "range_scatter_q_row_idx": torch.tensor([[3, 4, 5, -1, -1, -1, -1, -1]], dtype=torch.int32),
        "range_scatter_q_length": torch.tensor([3], dtype=torch.int32),
        "range_packed_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "geometry": {"fused_total_coverage_frac": 0.8},
    }

    class FakeCudaTensor:
        is_cuda = True
        dtype = torch.bfloat16
        shape = (6, 2, 64)

    assert cached_2d._direct_final_requires_online_combine(payload, torch.device("cpu"), base_source="fused")
    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        is None
    )
    missing = cached_2d._get_direct_final_missing_init_row_idx(payload, torch.device("cpu"))
    assert missing.tolist() == [4, 5]


def test_cached_2d_direct_final_allows_serial_mixed_residual_overlap():
    payload = {
        "total_rows": 6,
        "residual_mode": "fused_tail",
        "exact_kernel_family": "tc8x8",
        "exact_dense_rows_per_range": 8,
        "exact_dense_keys_per_tile": 8,
        "fused_q_row_idx": torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=torch.int32),
        "q_row_idx": torch.empty((2, 16), dtype=torch.int32),
        "fused_output_row_count": 4,
        "range_tc_scatter_row_count": 0,
        "range_scatter_row_count": 2,
        "range_packed_group_count": 1,
        "range_tc_scatter_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "range_scatter_q_row_idx": torch.tensor([[3, 4, -1, -1, -1, -1, -1, -1]], dtype=torch.int32),
        "range_scatter_q_length": torch.tensor([2], dtype=torch.int32),
        "range_packed_q_row_idx": torch.tensor([[4, 5, -1, -1, -1, -1, -1, -1]], dtype=torch.int32),
        "range_packed_q_length": torch.tensor([2], dtype=torch.int32),
        "geometry": {"fused_total_coverage_frac": 0.8},
    }

    class FakeCudaTensor:
        is_cuda = True
        dtype = torch.bfloat16
        shape = (6, 2, 64)

    assert cached_2d._direct_final_requires_online_combine(payload, torch.device("cpu"), base_source="fused")
    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        is None
    )
    missing = cached_2d._get_direct_final_missing_init_row_idx(payload, torch.device("cpu"))
    assert missing.tolist() == [4, 5]


def test_cached_2d_direct_final_blocks_duplicate_rows_inside_residual_kernel():
    payload = {
        "total_rows": 6,
        "residual_mode": "fused_tail",
        "exact_kernel_family": "tc8x8",
        "exact_dense_rows_per_range": 8,
        "exact_dense_keys_per_tile": 8,
        "fused_q_row_idx": torch.tensor([[0, 1, 2, 3, 5, -1, -1, -1]], dtype=torch.int32),
        "q_row_idx": torch.empty((1, 16), dtype=torch.int32),
        "fused_output_row_count": 5,
        "range_tc_scatter_row_count": 0,
        "range_scatter_row_count": 2,
        "range_packed_group_count": 0,
        "range_tc_scatter_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "range_scatter_q_row_idx": torch.tensor([[4, 4, -1, -1, -1, -1, -1, -1]], dtype=torch.int32),
        "range_scatter_q_length": torch.tensor([2], dtype=torch.int32),
        "range_packed_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "geometry": {"fused_total_coverage_frac": 0.8},
    }

    class FakeCudaTensor:
        is_cuda = True
        dtype = torch.bfloat16
        shape = (6, 2, 64)

    assert cached_2d._direct_final_has_duplicate_residual_rows_within_kernel(payload, torch.device("cpu"))
    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        == "direct_final_duplicate_residual_rows_within_kernel"
    )


def test_cached_2d_direct_final_residual_initializes_missing_mixed_rows():
    payload = {
        "total_rows": 6,
        "residual_mode": "fused_tail",
        "exact_kernel_family": "tc8x8",
        "exact_dense_rows_per_range": 8,
        "exact_dense_keys_per_tile": 8,
        "fused_q_row_idx": torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=torch.int32),
        "q_row_idx": torch.empty((2, 16), dtype=torch.int32),
        "fused_output_row_count": 4,
        "range_tc_scatter_row_count": 1,
        "range_scatter_row_count": 0,
        "range_packed_group_count": 1,
        "range_tc_scatter_q_row_idx": torch.tensor([[4, -1, -1, -1, -1, -1, -1, -1]], dtype=torch.int32),
        "range_scatter_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "range_packed_q_row_idx": torch.tensor([[5, -1, -1, -1, -1, -1, -1, -1]], dtype=torch.int32),
        "geometry": {"fused_total_coverage_frac": 0.8},
    }

    class FakeCudaTensor:
        is_cuda = True
        dtype = torch.bfloat16
        shape = (6, 2, 64)

    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        is None
    )
    missing = cached_2d._get_direct_final_missing_init_row_idx(payload, torch.device("cpu"))
    assert missing.tolist() == [4, 5]

    payload["range_packed_q_row_idx"] = torch.tensor([[3, -1, -1, -1, -1, -1, -1, -1]], dtype=torch.int32)
    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        == "mixed_residual_incomplete_direct_final_row_coverage"
    )
