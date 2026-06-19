import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import flash_attn.cute.hsa_cached_2d_forward_analysis as cached_2d


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


def test_cached_2d_direct_final_residual_allows_packed_residual_with_full_fused_coverage():
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

    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        is None
    )

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
    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        is None
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
