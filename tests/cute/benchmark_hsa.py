import importlib.util
import json
import time
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path

import torch

_FLASH_ATTN_IMPORTS = None
_EXTERNAL_HDT_ATTENTION = None
_EXTERNAL_HDT_STATUS = "uninitialized"


def _lazy_flash_attn_imports():
    global _FLASH_ATTN_IMPORTS
    if _FLASH_ATTN_IMPORTS is None:
        from flash_attn.cute import (
            build_hsa_schedule,
            flash_attn_func,
            flash_attn_hsa_sparse_func,
            hsa_reference_attention,
        )

        _FLASH_ATTN_IMPORTS = (
            build_hsa_schedule,
            flash_attn_func,
            flash_attn_hsa_sparse_func,
            hsa_reference_attention,
        )
    return _FLASH_ATTN_IMPORTS


def _ensure_cuda_ready(retries: int = 5, sleep_s: float = 1.0):
    last_error = None
    for attempt in range(retries):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    torch.cuda.set_device(0)
                    torch.cuda.get_device_name(0)
                    return
        except Exception as exc:
            last_error = exc
        if attempt + 1 < retries:
            time.sleep(sleep_s)
    if last_error is not None:
        raise RuntimeError(f"CUDA initialization failed after {retries} attempts") from last_error
    raise RuntimeError(f"CUDA is unavailable after {retries} attempts")


@dataclass
class BenchmarkCase:
    name: str
    batch_size: int
    seqlen: int
    nheads: int
    headdim: int
    n_kv_heads: int | None = None
    warmup_iters: int = 5
    benchmark_iters: int = 20


CANONICAL_CASES = (
    BenchmarkCase(name="sentence-only", batch_size=1, seqlen=256, nheads=4, headdim=64, n_kv_heads=4),
    BenchmarkCase(name="mixed-small", batch_size=1, seqlen=65, nheads=4, headdim=64, n_kv_heads=4),
    BenchmarkCase(name="train-eq", batch_size=2, seqlen=1024, nheads=8, headdim=64, n_kv_heads=8),
    BenchmarkCase(name="train-gqa", batch_size=2, seqlen=1024, nheads=8, headdim=64, n_kv_heads=2),
    BenchmarkCase(name="longer-eq", batch_size=2, seqlen=2048, nheads=8, headdim=64, n_kv_heads=8),
)

# Keep the stable 64x128 sparse backward geometry for long-context benchmarking.
# 32x* is blocked by the FA4 backward dQ M-mode restriction, and 64x64 is not
# currently stable/correct on the kept sparse backward path.
LONG_CONTEXT_CASES = (
    BenchmarkCase(name="long-16k", batch_size=1, seqlen=16384, nheads=4, headdim=64, n_kv_heads=4, warmup_iters=1, benchmark_iters=1),
    BenchmarkCase(name="long-32k", batch_size=1, seqlen=32768, nheads=4, headdim=64, n_kv_heads=4, warmup_iters=1, benchmark_iters=1),
    BenchmarkCase(name="long-64k", batch_size=1, seqlen=65536, nheads=4, headdim=64, n_kv_heads=4, warmup_iters=1, benchmark_iters=1),
    BenchmarkCase(name="long-100k", batch_size=1, seqlen=100000, nheads=4, headdim=64, n_kv_heads=4, warmup_iters=1, benchmark_iters=1),
    BenchmarkCase(name="long-128k", batch_size=1, seqlen=131072, nheads=4, headdim=64, n_kv_heads=4, warmup_iters=1, benchmark_iters=1),
)

ALL_CASES = CANONICAL_CASES + LONG_CONTEXT_CASES
LONG_CONTEXT_CASE_NAMES = frozenset(case.name for case in LONG_CONTEXT_CASES)


def _unwrap_output(result):
    return result[0] if isinstance(result, tuple) else result


def _load_external_hdt_attention():
    global _EXTERNAL_HDT_ATTENTION, _EXTERNAL_HDT_STATUS
    if _EXTERNAL_HDT_ATTENTION is not None or _EXTERNAL_HDT_STATUS != "uninitialized":
        return _EXTERNAL_HDT_ATTENTION, _EXTERNAL_HDT_STATUS

    module_path = Path(__file__).resolve().parents[2] / "third_party" / "hdt" / "src" / "HDT" / "hsparse_attn.py"
    if not module_path.exists():
        _EXTERNAL_HDT_STATUS = "missing_vendor"
        return None, _EXTERNAL_HDT_STATUS
    try:
        spec = importlib.util.spec_from_file_location("external_hdt_hsparse_attn", module_path)
        if spec is None or spec.loader is None:
            _EXTERNAL_HDT_STATUS = "missing_loader"
            return None, _EXTERNAL_HDT_STATUS
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _EXTERNAL_HDT_ATTENTION = module.attention_fn
        _EXTERNAL_HDT_STATUS = "available"
    except Exception as exc:  # pragma: no cover - benchmark-only import guard
        _EXTERNAL_HDT_STATUS = f"import_failed_{type(exc).__name__}"
        _EXTERNAL_HDT_ATTENTION = None
    return _EXTERNAL_HDT_ATTENTION, _EXTERNAL_HDT_STATUS


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
            hash_ids[batch_idx, 0, cursor] = sent_id
            hash_ids[batch_idx, 1, cursor] = sec_id
            hash_ids[batch_idx, 2, cursor] = doc_id
            cursor += 1
            if cursor >= seqlen:
                break

            for _ in range(2):
                if cursor >= seqlen:
                    break
                keep_ids[batch_idx, 1, cursor] = 1
                keep_ids[batch_idx, 2, cursor] = 1
                hash_ids[batch_idx, 0, cursor] = sent_id
                hash_ids[batch_idx, 1, cursor] = sec_id
                hash_ids[batch_idx, 2, cursor] = doc_id
                cursor += 1
                if cursor >= seqlen:
                    break

                for _ in range(2):
                    if cursor >= seqlen:
                        break
                    keep_ids[batch_idx, 0, cursor] = 1
                    keep_ids[batch_idx, 1, cursor] = 1
                    hash_ids[batch_idx, 0, cursor] = sent_id
                    hash_ids[batch_idx, 1, cursor] = sec_id
                    hash_ids[batch_idx, 2, cursor] = doc_id
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


def _measure_ms(fn, warmup_iters, benchmark_iters):
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(benchmark_iters):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000.0 / benchmark_iters


@contextmanager
def _temporary_env(**updates):
    old_values = {}
    for key, value in updates.items():
        old_values[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _split_monolithic_env():
    return {
        "FLASH_ATTN_HSA_USE_MONOLITHIC_BWD": "1",
        "FLASH_ATTN_HSA_USE_TRUE_FUSED_BWD": None,
        "FLASH_ATTN_HSA_USE_KERNEL_SENTENCE_FULL": "1",
        "FLASH_ATTN_HSA_USE_HYBRID_BWD": None,
        "FLASH_ATTN_HSA_USE_SENTENCE_VARLEN_2CTA": None,
    }


def _true_fused_env():
    return {
        "FLASH_ATTN_HSA_USE_MONOLITHIC_BWD": "1",
        "FLASH_ATTN_HSA_USE_TRUE_FUSED_BWD": "1",
        "FLASH_ATTN_HSA_USE_KERNEL_SENTENCE_FULL": None,
        "FLASH_ATTN_HSA_USE_HYBRID_BWD": None,
        "FLASH_ATTN_HSA_USE_SENTENCE_VARLEN_2CTA": None,
    }


def _benchmark_env_for_case(case: BenchmarkCase):
    del case
    return _true_fused_env()


def _benchmark_mode_label(case: BenchmarkCase) -> str:
    if os.environ.get("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "0") == "1":
        return "sentence_true_fused" if case.name == "sentence-only" else "mixed_sparse_mask_synthetic"
    return "sentence_true_fused" if case.name == "sentence-only" else "mixed_sparse_mask"


def _synthetic_mode_label() -> str:
    if os.environ.get("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "0") != "1":
        return "disabled"
    if os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_MICRO_FWD", "0") == "1":
        if os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_MICRO_BWD", "0") == "1":
            return "direct_micro_fwd_bwd"
        return "direct_micro_fwd"
    return "one_launch_generic_fa"


def _synthetic_bwd_kernel_mode_label() -> str:
    if os.environ.get("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "0") != "1":
        return "sparse_mask_plain"
    if os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_MICRO_BWD", "0") != "1":
        return "sparse_mask_bwd"
    one_kernel_mode = os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD", "off").strip().lower()
    if one_kernel_mode != "off":
        return f"one_kernel_{one_kernel_mode}"
    if os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_FUSED_BWD", "off").strip().lower() in {"1", "true", "yes", "on"}:
        return "legacy_fused"
    if os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_SPLIT_BWD", "off").strip().lower() in {"1", "true", "yes", "on"}:
        return "split"
    short_mode = os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_SHORT_BWD", "off").strip().lower()
    if short_mode != "off":
        return f"short_{short_mode}"
    return os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_ROW_BWD_ACCUM_MODE", "row_local")


def _benchmark_sparse_bwd_config_label() -> str:
    block_q = os.environ.get("FLASH_ATTN_HSA_BACKWARD_BLOCK_Q", "64")
    block_k = os.environ.get("FLASH_ATTN_HSA_BACKWARD_BLOCK_K", "128")
    subtile_factor = os.environ.get("FLASH_ATTN_HSA_BACKWARD_SUBTILE_FACTOR", "1")
    return f"bwd_block_q={block_q} bwd_block_k={block_k} bwd_subtile_factor={subtile_factor}"


def _should_measure_previous_synthetic_baseline(case: BenchmarkCase) -> bool:
    return (
        os.environ.get("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "0") == "1"
        and case.name != "sentence-only"
    )


def _previous_synthetic_baseline_label() -> str:
    return "direct_micro_row_local_baseline"


def _previous_synthetic_baseline_env(case: BenchmarkCase) -> dict[str, str | None]:
    env = dict(_benchmark_env_for_case(case))
    env.update(
        {
            "FLASH_ATTN_HSA_USE_SYNTHETIC_GRID": "1",
            "FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_Q": "2",
            "FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_K": "2",
            "FLASH_ATTN_HSA_SYNTHETIC_MAX_PACKED_K": "128",
            "FLASH_ATTN_HSA_SYNTHETIC_MAX_DIRECT_SEGMENTS": "1",
            "FLASH_ATTN_HSA_SYNTHETIC_MICRO_FWD": "1",
            "FLASH_ATTN_HSA_SYNTHETIC_MICRO_BWD": "1",
            "FLASH_ATTN_HSA_SYNTHETIC_FUSED_BWD": "off",
            "FLASH_ATTN_HSA_SYNTHETIC_QGROUPS_PER_CTA": "2",
            "FLASH_ATTN_HSA_SYNTHETIC_QGROUPS_PER_CTA_BWD": "2",
            "FLASH_ATTN_HSA_SYNTHETIC_ROW_BWD_ACCUM_MODE": "row_local",
            "FLASH_ATTN_HSA_SYNTHETIC_SHORT_BWD": "off",
        }
    )
    return env


def _should_measure_short_synthetic_baseline(case: BenchmarkCase) -> bool:
    return (
        os.environ.get("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "0") == "1"
        and case.name != "sentence-only"
    )


def _short_synthetic_baseline_label() -> str:
    return "direct_micro_short_bwd_baseline"


def _short_synthetic_baseline_env(case: BenchmarkCase) -> dict[str, str | None]:
    env = dict(_previous_synthetic_baseline_env(case))
    env.update(
        {
            "FLASH_ATTN_HSA_SYNTHETIC_SHORT_BWD": "on",
        }
    )
    return env


def _should_measure_plain_sparse_mask_baseline(case: BenchmarkCase) -> bool:
    return (
        os.environ.get("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "0") == "1"
        and case.name != "sentence-only"
    )


def _plain_sparse_mask_baseline_label() -> str:
    return "hsa_sparse_mask_plain"


def _plain_sparse_mask_baseline_env(case: BenchmarkCase) -> dict[str, str | None]:
    env = dict(_benchmark_env_for_case(case))
    env.update(
        {
            "FLASH_ATTN_HSA_USE_SYNTHETIC_GRID": "0",
            "FLASH_ATTN_HSA_SYNTHETIC_MICRO_FWD": "0",
            "FLASH_ATTN_HSA_SYNTHETIC_MICRO_BWD": "0",
            "FLASH_ATTN_HSA_SYNTHETIC_FUSED_BWD": "off",
            "FLASH_ATTN_HSA_SYNTHETIC_SHORT_BWD": "off",
        }
    )
    return env


def _should_measure_sparse_mask_mixed_backward_baseline(case: BenchmarkCase) -> bool:
    return (
        os.environ.get("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "0") == "1"
        and case.name != "sentence-only"
        and (case.n_kv_heads is None or case.n_kv_heads == case.nheads)
    )


def _sparse_mask_mixed_backward_baseline_label() -> str:
    return "direct_micro_fwd_sparse_mask_bwd"


def _sparse_mask_mixed_backward_baseline_env(case: BenchmarkCase) -> dict[str, str | None]:
    env = dict(_benchmark_env_for_case(case))
    env.update(
        {
            "FLASH_ATTN_HSA_USE_SYNTHETIC_GRID": "1",
            "FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_Q": "2",
            "FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_K": "2",
            "FLASH_ATTN_HSA_SYNTHETIC_MAX_PACKED_K": "128",
            "FLASH_ATTN_HSA_SYNTHETIC_MAX_DIRECT_SEGMENTS": "1",
            "FLASH_ATTN_HSA_SYNTHETIC_MICRO_FWD": "1",
            "FLASH_ATTN_HSA_SYNTHETIC_MICRO_BWD": "0",
            "FLASH_ATTN_HSA_SYNTHETIC_FUSED_BWD": "off",
            "FLASH_ATTN_HSA_SYNTHETIC_QGROUPS_PER_CTA": "2",
            "FLASH_ATTN_HSA_SYNTHETIC_QGROUPS_PER_CTA_BWD": "2",
            "FLASH_ATTN_HSA_SYNTHETIC_ROW_BWD_ACCUM_MODE": "row_local",
            "FLASH_ATTN_HSA_SYNTHETIC_SHORT_BWD": "off",
        }
    )
    return env


def _is_long_context_case(case: BenchmarkCase) -> bool:
    return case.name in LONG_CONTEXT_CASE_NAMES


def _get_selected_cases(*, env_var: str, default: tuple[BenchmarkCase, ...]) -> list[BenchmarkCase]:
    selected = [name.strip() for name in os.environ.get(env_var, "").split(",")]
    selected = [name for name in selected if name]
    if not selected:
        return list(default)
    by_name = {case.name: case for case in ALL_CASES}
    missing = [name for name in selected if name not in by_name]
    if missing:
        raise KeyError(f"Unknown {env_var} entries: {', '.join(missing)}")
    return [by_name[name] for name in selected]


def _use_sparse_profile_mode() -> bool:
    return os.environ.get("FLASH_ATTN_HSA_PROFILE_SPARSE_MASK", "0") == "1"


def _use_geometry_sweep_mode() -> bool:
    return os.environ.get("FLASH_ATTN_HSA_GEOMETRY_SWEEP", "0") == "1"


def _geometry_sweep_configs() -> tuple[tuple[int, int, int], ...]:
    return (
        (64, 128, 1),
        (64, 64, 1),
        (32, 64, 1),
        (32, 32, 1),
    )


def _geometry_sweep_env(block_q: int, block_k: int, subtile_factor: int) -> dict[str, str]:
    return {
        "FLASH_ATTN_HSA_BACKWARD_BLOCK_Q": str(block_q),
        "FLASH_ATTN_HSA_BACKWARD_BLOCK_K": str(block_k),
        "FLASH_ATTN_HSA_BACKWARD_SUBTILE_FACTOR": str(subtile_factor),
    }


def _summarize_sparse_backward_occupancy(sparse_tensors, packed_masks, *, seqlen: int) -> dict[str, float | int]:
    mask_block_cnt = sparse_tensors.mask_block_cnt.detach().cpu()
    mask_block_idx = sparse_tensors.mask_block_idx.detach().cpu()
    full_block_cnt = None if sparse_tensors.full_block_cnt is None else sparse_tensors.full_block_cnt.detach().cpu()
    full_block_idx = None if sparse_tensors.full_block_idx is None else sparse_tensors.full_block_idx.detach().cpu()
    block_id_table = packed_masks.block_id_table.detach().cpu()
    mask_words = packed_masks.mask_words.detach().cpu()

    batch_size = mask_block_cnt.shape[0]
    num_k_blocks = mask_block_cnt.shape[2]
    num_q_blocks = mask_block_idx.shape[3]
    q_block_size, k_block_size = sparse_tensors.block_size
    words_per_row = packed_masks.words_per_row

    allowed_pairs = 0
    valid_active_area = 0
    nominal_active_area = 0
    active_blocks = 0

    for batch_idx in range(batch_size):
        for k_block in range(num_k_blocks):
            k_start = k_block * k_block_size
            k_len = min(k_block_size, seqlen - k_start)
            tail_mask = (1 << (k_len % 32)) - 1 if k_len % 32 != 0 else None

            full_cnt = 0 if full_block_cnt is None else int(full_block_cnt[batch_idx, 0, k_block].item())
            active_blocks += full_cnt
            for offset in range(full_cnt):
                q_block = int(full_block_idx[batch_idx, 0, k_block, offset].item())
                q_start = q_block * q_block_size
                q_len = min(q_block_size, seqlen - q_start)
                allowed_pairs += q_len * k_len
                valid_active_area += q_len * k_len
                nominal_active_area += q_block_size * k_block_size

            partial_cnt = int(mask_block_cnt[batch_idx, 0, k_block].item())
            active_blocks += partial_cnt
            for offset in range(partial_cnt):
                q_block = int(mask_block_idx[batch_idx, 0, k_block, offset].item())
                q_start = q_block * q_block_size
                q_len = min(q_block_size, seqlen - q_start)
                block_id = int(block_id_table[batch_idx, k_block, q_block].item())
                valid_active_area += q_len * k_len
                nominal_active_area += q_block_size * k_block_size
                for q_local in range(q_len):
                    for word_idx in range(words_per_row):
                        word = int(mask_words[block_id, q_local, word_idx].item()) & 0xFFFFFFFF
                        if tail_mask is not None and word_idx == words_per_row - 1:
                            word &= tail_mask
                        allowed_pairs += word.bit_count()

    total_blocks = batch_size * num_k_blocks * num_q_blocks
    causal_pairs = batch_size * seqlen * (seqlen + 1) // 2

    return {
        "allowed_pairs": allowed_pairs,
        "total_blocks": total_blocks,
        "active_blocks": active_blocks,
        "token_density": allowed_pairs / causal_pairs if causal_pairs > 0 else 0.0,
        "active_block_density": active_blocks / total_blocks if total_blocks > 0 else 0.0,
        "active_fill_nominal": allowed_pairs / nominal_active_area if nominal_active_area > 0 else 0.0,
        "active_fill_valid": allowed_pairs / valid_active_area if valid_active_area > 0 else 0.0,
    }


def _run_sparse_attention(
    q,
    k,
    v,
    keep_ids,
    hash_ids,
    schedule,
    *,
    env_updates=None,
):
    _, _, flash_attn_hsa_sparse_func, _ = _lazy_flash_attn_imports()
    env_updates = env_updates or {}
    with _temporary_env(**env_updates):
        return _unwrap_output(
            flash_attn_hsa_sparse_func(q, k, v, keep_ids=keep_ids, hash_ids=hash_ids, hsa_schedule=schedule)
        )


def _run_external_hdt_attention(q, k, v, keep_ids, hash_ids):
    attention_fn, status = _load_external_hdt_attention()
    if attention_fn is None:
        raise RuntimeError(status)
    if q.shape[1] > 1024:
        raise RuntimeError("unsupported_seq_gt_1024")
    q_ext = q.transpose(1, 2).contiguous()
    k_ext = k.transpose(1, 2).contiguous()
    v_ext = v.transpose(1, 2).contiguous()
    if k_ext.shape[1] != q_ext.shape[1]:
        raise RuntimeError("unsupported_gqa")
    keep_list = [keep_ids[:, idx, :].contiguous() for idx in range(3)]
    hash_list = [hash_ids[:, idx, :].contiguous() for idx in range(3)]
    out, _ = attention_fn(q_ext, k_ext, v_ext, keep_list, hash_list, causal=True, output_attentions=False)
    return out


def _build_case_tensors(case: BenchmarkCase):
    build_hsa_schedule, _, _, _ = _lazy_flash_attn_imports()
    device = "cuda"
    dtype = torch.bfloat16
    keep_ids, hash_ids = _make_hsa_metadata(case.batch_size, case.seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    n_kv_heads = case.n_kv_heads if case.n_kv_heads is not None else case.nheads
    q_data = torch.randn(case.batch_size, case.seqlen, case.nheads, case.headdim, device=device, dtype=dtype)
    k_data = torch.randn(case.batch_size, case.seqlen, n_kv_heads, case.headdim, device=device, dtype=dtype)
    v_data = torch.randn(case.batch_size, case.seqlen, n_kv_heads, case.headdim, device=device, dtype=dtype)
    return schedule, keep_ids, hash_ids, q_data, k_data, v_data


def _measure_external_hdt_case() -> dict[str, object]:
    by_name = {candidate.name: candidate for candidate in ALL_CASES}
    case_name = os.environ["FLASH_ATTN_HSA_EXTERNAL_HDT_CASE"]
    case = by_name[case_name]
    _, keep_ids, hash_ids, q_data, k_data, v_data = _build_case_tensors(case)
    external_hdt_forward = lambda q, k, v: _run_external_hdt_attention(q, k, v, keep_ids, hash_ids)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Logical operators 'and' and 'or' are deprecated.*")
        warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage.*")
        try:
            result = {
                "status": "measured",
                "fwd_ms": _measure_forward_ms(
                    external_hdt_forward,
                    q_data,
                    k_data,
                    v_data,
                    case.warmup_iters,
                    case.benchmark_iters,
                ),
                "bwd_ms": _measure_backward_ms(
                    external_hdt_forward,
                    q_data,
                    k_data,
                    v_data,
                    case.warmup_iters,
                    case.benchmark_iters,
                ),
                "fwd_bwd_ms": _measure_forward_backward_ms(
                    external_hdt_forward,
                    q_data,
                    k_data,
                    v_data,
                    case.warmup_iters,
                    case.benchmark_iters,
                ),
            }
        except Exception as exc:  # pragma: no cover - GPU benchmark guard
            torch.cuda.empty_cache()
            reason = str(exc)
            if reason.startswith(("unsupported_", "missing_", "import_failed_", "missing_loader")):
                result = {"status": reason}
            else:
                result = {"status": f"failed_{type(exc).__name__}"}
    return result


def _run_external_hdt_case_subprocess(case: BenchmarkCase) -> dict[str, object]:
    child_env = os.environ.copy()
    child_env["FLASH_ATTN_HSA_EXTERNAL_HDT_CHILD"] = "1"
    child_env["FLASH_ATTN_HSA_EXTERNAL_HDT_CASE"] = case.name
    result = subprocess.run(
        [sys.executable, __file__],
        env=child_env,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        status = f"child_exit_{result.returncode}"
        stderr = result.stderr.strip().splitlines()
        if stderr:
            status = f"{status}_{stderr[-1][:120]}"
        return {"status": status}

    stdout_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not stdout_lines:
        return {"status": "child_empty_output"}
    try:
        payload = json.loads(stdout_lines[-1])
    except json.JSONDecodeError:
        return {"status": "child_bad_output"}
    if not isinstance(payload, dict) or "status" not in payload:
        return {"status": "child_bad_payload"}
    return payload


def _get_synthetic_grid_summary(schedule, q, k):
    import flash_attn.cute.hsa as hsa_module
    from flash_attn.cute.flash_hsa_synthetic_grid_sm100 import summarize_synthetic_grid

    runtime = hsa_module._get_hsa_block_sparse_runtime(schedule, q, k)
    hsa_module._ensure_hsa_synthetic_grid_metadata(schedule, runtime)
    return summarize_synthetic_grid(runtime)


def _measure_backward_ms(forward_fn, q_data, k_data, v_data, warmup_iters, benchmark_iters, *, env_updates=None):
    env_updates = env_updates or {}
    with _temporary_env(**env_updates):
        q = q_data.clone().requires_grad_(True)
        k = k_data.clone().requires_grad_(True)
        v = v_data.clone().requires_grad_(True)
        loss = forward_fn(q, k, v).float().square().mean()

    def _run():
        with _temporary_env(**env_updates):
            torch.autograd.grad(loss, (q, k, v), retain_graph=True)

    return _measure_ms(
        _run,
        warmup_iters,
        benchmark_iters,
    )


def _measure_forward_ms(forward_fn, q_data, k_data, v_data, warmup_iters, benchmark_iters, *, env_updates=None):
    env_updates = env_updates or {}

    def _run():
        with _temporary_env(**env_updates):
            forward_fn(q_data, k_data, v_data)

    return _measure_ms(_run, warmup_iters, benchmark_iters)


def _measure_forward_backward_ms(
    forward_fn,
    q_data,
    k_data,
    v_data,
    warmup_iters,
    benchmark_iters,
    *,
    env_updates=None,
):
    env_updates = env_updates or {}

    def _run():
        with _temporary_env(**env_updates):
            q = q_data.clone().requires_grad_(True)
            k = k_data.clone().requires_grad_(True)
            v = v_data.clone().requires_grad_(True)
            loss = forward_fn(q, k, v).float().square().mean()
            torch.autograd.grad(loss, (q, k, v))

    return _measure_ms(_run, warmup_iters, benchmark_iters)


def _event_self_device_us(event) -> float:
    return float(getattr(event, "self_cuda_time_total", getattr(event, "self_device_time_total", 0.0)))


def _top_cuda_rows(prof, *, include_patterns: tuple[str, ...] = (), limit: int = 8) -> list[tuple[str, float]]:
    rows = []
    events = sorted(prof.key_averages(), key=_event_self_device_us, reverse=True)
    for event in events:
        key = getattr(event, "key", getattr(event, "name", ""))
        value_us = _event_self_device_us(event)
        if value_us <= 0.0:
            continue
        if include_patterns and not any(pattern in key for pattern in include_patterns):
            continue
        if key.startswith("aten::") or key.startswith("_") or key.startswith("void at::") or key.startswith("Memcpy"):
            continue
        rows.append((key, value_us))
        if len(rows) >= limit:
            break
    return rows


def _profile_sparse_mask_case(case: BenchmarkCase):
    build_hsa_schedule, flash_attn_func, flash_attn_hsa_sparse_func, _ = _lazy_flash_attn_imports()
    device = "cuda"
    dtype = torch.bfloat16
    keep_ids, hash_ids = _make_hsa_metadata(case.batch_size, case.seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    n_kv_heads = case.n_kv_heads if case.n_kv_heads is not None else case.nheads
    q_data = torch.randn(case.batch_size, case.seqlen, case.nheads, case.headdim, device=device, dtype=dtype)
    k_data = torch.randn(case.batch_size, case.seqlen, n_kv_heads, case.headdim, device=device, dtype=dtype)
    v_data = torch.randn(case.batch_size, case.seqlen, n_kv_heads, case.headdim, device=device, dtype=dtype)
    env_updates = _benchmark_env_for_case(case)

    with _temporary_env(**env_updates):
        q_hsa = q_data.clone().requires_grad_(True)
        k_hsa = k_data.clone().requires_grad_(True)
        v_hsa = v_data.clone().requires_grad_(True)
        out_hsa = flash_attn_hsa_sparse_func(q_hsa, k_hsa, v_hsa, keep_ids=keep_ids, hash_ids=hash_ids, hsa_schedule=schedule)
        hsa_loss = _unwrap_output(out_hsa).float().square().mean()

    q_dense = q_data.clone().requires_grad_(True)
    k_dense = k_data.clone().requires_grad_(True)
    v_dense = v_data.clone().requires_grad_(True)
    dense_loss = _unwrap_output(flash_attn_func(q_dense, k_dense, v_dense, causal=True)).float().square().mean()

    def _run_hsa():
        with _temporary_env(**env_updates):
            torch.autograd.grad(hsa_loss, (q_hsa, k_hsa, v_hsa), retain_graph=True)

    def _run_dense():
        torch.autograd.grad(dense_loss, (q_dense, k_dense, v_dense), retain_graph=True)

    warmups = max(3, min(case.warmup_iters, 5))
    for _ in range(warmups):
        _run_hsa()
        _run_dense()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
    ) as sparse_prof:
        _run_hsa()
        torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
    ) as dense_prof:
        _run_dense()
        torch.cuda.synchronize()

    sparse_rows = _top_cuda_rows(sparse_prof, limit=6)
    dense_rows = _top_cuda_rows(
        dense_prof,
        include_patterns=(
            "FlashAttentionBackwardSm100",
            "flash_attncuteflash_bwd_sm100",
            "FlashAttentionBackwardPreprocess",
            "flash_attncuteflash_bwd_preprocess",
            "FlashAttentionBackwardPostprocess",
            "flash_attncuteflash_bwd_postprocess",
        ),
        limit=6,
    )
    dense_kernel_key, dense_kernel_us = dense_rows[0] if dense_rows else ("", 0.0)

    print(
        f"sparse_mask_profile_case={case.name} mode={_benchmark_mode_label(case)} "
        f"{_benchmark_sparse_bwd_config_label()}"
    )
    if sparse_rows:
        top_key, top_us = sparse_rows[0]
        ratio = top_us / dense_kernel_us if dense_kernel_us > 0.0 else float('inf')
        print(f"sparse_mask_backward_top case={case.name} self_cuda_us={top_us:.1f} vs_fa4_backward={ratio:.2f}x kernel={top_key}")
        for key, value_us in sparse_rows[1:]:
            print(f"sparse_mask_backward_neighbor case={case.name} self_cuda_us={value_us:.1f} kernel={key}")
    if dense_kernel_us > 0.0:
        print(f"dense_fa4_backward_baseline case={case.name} self_cuda_us={dense_kernel_us:.1f} kernel={dense_kernel_key}")
        for key, value_us in dense_rows[1:]:
            print(f"dense_fa4_backward_neighbor case={case.name} self_cuda_us={value_us:.1f} kernel={key}")


def _run_geometry_sweep_case(case: BenchmarkCase, *, block_q: int, block_k: int, subtile_factor: int):
    build_hsa_schedule, flash_attn_func, _, hsa_reference_attention = _lazy_flash_attn_imports()
    device = "cuda"
    dtype = torch.bfloat16
    keep_ids, hash_ids = _make_hsa_metadata(case.batch_size, case.seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    n_kv_heads = case.n_kv_heads if case.n_kv_heads is not None else case.nheads
    q_data = torch.randn(case.batch_size, case.seqlen, case.nheads, case.headdim, device=device, dtype=dtype)
    k_data = torch.randn(case.batch_size, case.seqlen, n_kv_heads, case.headdim, device=device, dtype=dtype)
    v_data = torch.randn(case.batch_size, case.seqlen, n_kv_heads, case.headdim, device=device, dtype=dtype)
    env_updates = {
        **_benchmark_env_for_case(case),
        **_geometry_sweep_env(block_q, block_k, subtile_factor),
    }

    hsa_forward = lambda q, k, v: _run_sparse_attention(q, k, v, keep_ids, hash_ids, schedule, env_updates=env_updates)
    dense_causal_forward = lambda q, k, v: _unwrap_output(flash_attn_func(q, k, v, causal=True))
    dense_ref_forward = lambda q, k, v: hsa_reference_attention(q, k, v, keep_ids, hash_ids)

    try:
        out_hsa = hsa_forward(q_data, k_data, v_data)
        out_ref = dense_ref_forward(q_data, k_data, v_data)
        hsa_max_diff = (out_hsa.float() - out_ref.float()).abs().max().item()
        hsa_mean_diff = (out_hsa.float() - out_ref.float()).abs().mean().item()

        hsa_bwd_ms = _measure_backward_ms(
            hsa_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=env_updates,
        )
        dense_causal_bwd_ms = _measure_backward_ms(
            dense_causal_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
        )
        hsa_bwd_vs_dense = hsa_bwd_ms / dense_causal_bwd_ms if dense_causal_bwd_ms > 0 else float("inf")

        print(
            f"geometry_sweep_case={case.name} mode={_benchmark_mode_label(case)} "
            f"bwd_block_q={block_q} bwd_block_k={block_k} bwd_subtile_factor={subtile_factor} "
            f"shape=(B={case.batch_size}, T={case.seqlen}, H={case.nheads}, KV={n_kv_heads}, D={case.headdim}) "
            f"hsa_max_diff={hsa_max_diff:.6f} hsa_mean_diff={hsa_mean_diff:.6f} "
            f"hsa_bwd_ms={hsa_bwd_ms:.3f} dense_causal_bwd_ms={dense_causal_bwd_ms:.3f} "
            f"hsa_bwd_vs_dense={hsa_bwd_vs_dense:.2f}x"
        )
    except Exception as exc:
        print(
            f"geometry_sweep_failure case={case.name} mode={_benchmark_mode_label(case)} "
            f"bwd_block_q={block_q} bwd_block_k={block_k} bwd_subtile_factor={subtile_factor} "
            f"error={type(exc).__name__}: {exc}"
        )


def _run_geometry_sweep_once():
    default_cases = tuple(case for case in CANONICAL_CASES if case.name != "sentence-only")
    for case in _get_selected_cases(
        env_var="FLASH_ATTN_HSA_GEOMETRY_SWEEP_CASES",
        default=default_cases,
    ):
        for block_q, block_k, subtile_factor in _geometry_sweep_configs():
            _run_geometry_sweep_case(
                case,
                block_q=block_q,
                block_k=block_k,
                subtile_factor=subtile_factor,
            )


def _run_long_case(case: BenchmarkCase):
    import flash_attn.cute.hsa as hsa_module

    build_hsa_schedule, flash_attn_func, _, hsa_reference_attention = _lazy_flash_attn_imports()
    device = "cuda"
    dtype = torch.bfloat16
    keep_ids, hash_ids = _make_hsa_metadata(case.batch_size, case.seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    n_kv_heads = case.n_kv_heads if case.n_kv_heads is not None else case.nheads
    q_data = torch.randn(case.batch_size, case.seqlen, case.nheads, case.headdim, device=device, dtype=dtype)
    k_data = torch.randn(case.batch_size, case.seqlen, n_kv_heads, case.headdim, device=device, dtype=dtype)
    v_data = torch.randn(case.batch_size, case.seqlen, n_kv_heads, case.headdim, device=device, dtype=dtype)
    env_updates = _benchmark_env_for_case(case)

    with _temporary_env(**env_updates):
        runtime = hsa_module._get_hsa_block_sparse_runtime(schedule, q_data, k_data)
    occupancy = _summarize_sparse_backward_occupancy(
        runtime.backward_sparse,
        runtime.backward_packed_masks,
        seqlen=case.seqlen,
    )

    hsa_forward = lambda q, k, v: _run_sparse_attention(q, k, v, keep_ids, hash_ids, schedule, env_updates=env_updates)
    dense_causal_forward = lambda q, k, v: _unwrap_output(flash_attn_func(q, k, v, causal=True))
    hsa_bwd_ms = _measure_backward_ms(
        hsa_forward,
        q_data,
        k_data,
        v_data,
        case.warmup_iters,
        case.benchmark_iters,
        env_updates=env_updates,
    )
    dense_causal_bwd_ms = _measure_backward_ms(
        dense_causal_forward,
        q_data,
        k_data,
        v_data,
        case.warmup_iters,
        case.benchmark_iters,
    )
    hsa_bwd_vs_dense = hsa_bwd_ms / dense_causal_bwd_ms if dense_causal_bwd_ms > 0 else float("inf")
    plain_sparse_mask_bwd_ms = None
    hsa_bwd_vs_plain_sparse_mask = None
    prev_synth_bwd_ms = None
    hsa_bwd_vs_prev = None
    short_synth_bwd_ms = None
    hsa_bwd_vs_short = None
    sparse_mask_mixed_bwd_ms = None
    hsa_bwd_vs_sparse_mask_mixed = None

    hdt_in_repo_dense_bwd_ms = None
    hdt_in_repo_dense_status = "skipped_32k_plus"
    hdt_in_repo_dense_ratio = None
    hdt_vendor_bwd_ms = None
    hdt_vendor_status = "skipped_32k_plus"
    hdt_vendor_ratio = None
    if case.name == "long-16k":
        dense_ref_forward = lambda q, k, v: hsa_reference_attention(q, k, v, keep_ids, hash_ids)
        try:
            hdt_in_repo_dense_bwd_ms = _measure_backward_ms(
                dense_ref_forward,
                q_data,
                k_data,
                v_data,
                case.warmup_iters,
                case.benchmark_iters,
            )
            hdt_in_repo_dense_ratio = (
                hsa_bwd_ms / hdt_in_repo_dense_bwd_ms if hdt_in_repo_dense_bwd_ms > 0 else float("inf")
            )
            hdt_in_repo_dense_status = "measured"
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            torch.cuda.empty_cache()
            hdt_in_repo_dense_status = f"failed_{type(exc).__name__}"
        external_result = _run_external_hdt_case_subprocess(case)
        hdt_vendor_status = str(external_result.get("status", "child_bad_payload"))
        hdt_vendor_bwd_ms = external_result.get("bwd_ms")
        if hdt_vendor_status == "measured" and isinstance(hdt_vendor_bwd_ms, (int, float)):
            hdt_vendor_ratio = hsa_bwd_ms / hdt_vendor_bwd_ms if hdt_vendor_bwd_ms > 0 else float("inf")
    if _should_measure_previous_synthetic_baseline(case):
        plain_env_updates = _plain_sparse_mask_baseline_env(case)
        plain_hsa_forward = lambda q, k, v: _run_sparse_attention(
            q, k, v, keep_ids, hash_ids, schedule, env_updates=plain_env_updates
        )
        plain_sparse_mask_bwd_ms = _measure_backward_ms(
            plain_hsa_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=plain_env_updates,
        )
        hsa_bwd_vs_plain_sparse_mask = (
            hsa_bwd_ms / plain_sparse_mask_bwd_ms if plain_sparse_mask_bwd_ms > 0 else float("inf")
        )
        prev_env_updates = _previous_synthetic_baseline_env(case)
        prev_hsa_forward = lambda q, k, v: _run_sparse_attention(
            q, k, v, keep_ids, hash_ids, schedule, env_updates=prev_env_updates
        )
        prev_synth_bwd_ms = _measure_backward_ms(
            prev_hsa_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=prev_env_updates,
        )
        hsa_bwd_vs_prev = hsa_bwd_ms / prev_synth_bwd_ms if prev_synth_bwd_ms > 0 else float("inf")
    if _should_measure_short_synthetic_baseline(case):
        short_env_updates = _short_synthetic_baseline_env(case)
        short_hsa_forward = lambda q, k, v: _run_sparse_attention(
            q, k, v, keep_ids, hash_ids, schedule, env_updates=short_env_updates
        )
        short_synth_bwd_ms = _measure_backward_ms(
            short_hsa_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=short_env_updates,
        )
        hsa_bwd_vs_short = hsa_bwd_ms / short_synth_bwd_ms if short_synth_bwd_ms > 0 else float("inf")
    if _should_measure_sparse_mask_mixed_backward_baseline(case):
        sparse_mask_env_updates = _sparse_mask_mixed_backward_baseline_env(case)
        sparse_mask_hsa_forward = lambda q, k, v: _run_sparse_attention(
            q, k, v, keep_ids, hash_ids, schedule, env_updates=sparse_mask_env_updates
        )
        sparse_mask_mixed_bwd_ms = _measure_backward_ms(
            sparse_mask_hsa_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=sparse_mask_env_updates,
        )
        hsa_bwd_vs_sparse_mask_mixed = (
            hsa_bwd_ms / sparse_mask_mixed_bwd_ms if sparse_mask_mixed_bwd_ms > 0 else float("inf")
        )

    line = (
        f"{case.name}: mode={_benchmark_mode_label(case)} {_benchmark_sparse_bwd_config_label()} "
        f"shape=(B={case.batch_size}, T={case.seqlen}, H={case.nheads}, KV={n_kv_heads}, D={case.headdim}) "
        f"allowed_pairs={occupancy['allowed_pairs']} token_density={occupancy['token_density']:.6f} "
        f"active_blocks={occupancy['active_blocks']} total_blocks={occupancy['total_blocks']} "
        f"active_block_density={occupancy['active_block_density']:.6f} "
        f"active_fill_nominal={occupancy['active_fill_nominal']:.6f} "
        f"active_fill_valid={occupancy['active_fill_valid']:.6f} "
        f"hsa_bwd_ms={hsa_bwd_ms:.3f} dense_causal_bwd_ms={dense_causal_bwd_ms:.3f} "
        f"hsa_bwd_vs_dense={hsa_bwd_vs_dense:.2f}x"
    )
    if hdt_in_repo_dense_bwd_ms is not None and hdt_in_repo_dense_ratio is not None:
        line += (
            f" hdt_in_repo_dense_bwd_ms={hdt_in_repo_dense_bwd_ms:.3f}"
            f" hsa_bwd_vs_hdt_in_repo={hdt_in_repo_dense_ratio:.2f}x"
        )
    else:
        line += f" hdt_in_repo_dense_bwd_status={hdt_in_repo_dense_status}"
    if hdt_vendor_bwd_ms is not None and hdt_vendor_ratio is not None:
        line += f" hdt_vendor_bwd_ms={hdt_vendor_bwd_ms:.3f} hsa_bwd_vs_hdt_vendor={hdt_vendor_ratio:.2f}x"
    else:
        line += f" hdt_vendor_bwd_status={hdt_vendor_status}"
    if plain_sparse_mask_bwd_ms is not None:
        line += (
            f" sparse_mask_plain_label={_plain_sparse_mask_baseline_label()}"
            f" sparse_mask_plain_bwd_ms={plain_sparse_mask_bwd_ms:.3f}"
            f" hsa_bwd_vs_sparse_mask_plain={hsa_bwd_vs_plain_sparse_mask:.2f}x"
        )
    if prev_synth_bwd_ms is not None:
        line += (
            f" prev_synth_label={_previous_synthetic_baseline_label()}"
            f" prev_synth_bwd_ms={prev_synth_bwd_ms:.3f}"
            f" hsa_bwd_vs_prev={hsa_bwd_vs_prev:.2f}x"
        )
    if short_synth_bwd_ms is not None:
        line += (
            f" short_synth_label={_short_synthetic_baseline_label()}"
            f" short_synth_bwd_ms={short_synth_bwd_ms:.3f}"
            f" hsa_bwd_vs_short={hsa_bwd_vs_short:.2f}x"
        )
    if sparse_mask_mixed_bwd_ms is not None:
        line += (
            f" sparse_mask_mixed_bwd_label={_sparse_mask_mixed_backward_baseline_label()}"
            f" sparse_mask_mixed_bwd_ms={sparse_mask_mixed_bwd_ms:.3f}"
            f" hsa_bwd_vs_sparse_mask_mixed={hsa_bwd_vs_sparse_mask_mixed:.2f}x"
        )
    print(line)


def run_case(case: BenchmarkCase):
    build_hsa_schedule, flash_attn_func, _, hsa_reference_attention = _lazy_flash_attn_imports()
    device = "cuda"
    dtype = torch.bfloat16
    keep_ids, hash_ids = _make_hsa_metadata(case.batch_size, case.seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    n_kv_heads = case.n_kv_heads if case.n_kv_heads is not None else case.nheads
    q_data = torch.randn(case.batch_size, case.seqlen, case.nheads, case.headdim, device=device, dtype=dtype)
    k_data = torch.randn(case.batch_size, case.seqlen, n_kv_heads, case.headdim, device=device, dtype=dtype)
    v_data = torch.randn(case.batch_size, case.seqlen, n_kv_heads, case.headdim, device=device, dtype=dtype)
    env_updates = _benchmark_env_for_case(case)

    hsa_forward = lambda q, k, v: _run_sparse_attention(q, k, v, keep_ids, hash_ids, schedule, env_updates=env_updates)
    dense_causal_forward = lambda q, k, v: _unwrap_output(flash_attn_func(q, k, v, causal=True))
    dense_ref_forward = lambda q, k, v: hsa_reference_attention(q, k, v, keep_ids, hash_ids)
    out_hsa = hsa_forward(q_data, k_data, v_data)
    out_ref = dense_ref_forward(q_data, k_data, v_data)
    hsa_max_diff = (out_hsa.float() - out_ref.float()).abs().max().item()
    hsa_mean_diff = (out_hsa.float() - out_ref.float()).abs().mean().item()
    synthetic_summary = None
    if os.environ.get("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "0") == "1":
        synthetic_summary = _get_synthetic_grid_summary(schedule, q_data, k_data)

    hsa_fwd_ms = _measure_forward_ms(
        hsa_forward,
        q_data,
        k_data,
        v_data,
        case.warmup_iters,
        case.benchmark_iters,
        env_updates=env_updates,
    )
    dense_causal_fwd_ms = _measure_forward_ms(
        dense_causal_forward,
        q_data,
        k_data,
        v_data,
        case.warmup_iters,
        case.benchmark_iters,
    )
    hsa_bwd_ms = _measure_backward_ms(
        hsa_forward,
        q_data,
        k_data,
        v_data,
        case.warmup_iters,
        case.benchmark_iters,
        env_updates=env_updates,
    )
    dense_causal_bwd_ms = _measure_backward_ms(
        dense_causal_forward,
        q_data,
        k_data,
        v_data,
        case.warmup_iters,
        case.benchmark_iters,
    )
    hsa_fwd_bwd_ms = _measure_forward_backward_ms(
        hsa_forward,
        q_data,
        k_data,
        v_data,
        case.warmup_iters,
        case.benchmark_iters,
        env_updates=env_updates,
    )
    dense_causal_fwd_bwd_ms = _measure_forward_backward_ms(
        dense_causal_forward,
        q_data,
        k_data,
        v_data,
        case.warmup_iters,
        case.benchmark_iters,
    )

    hsa_bwd_vs_dense = hsa_bwd_ms / dense_causal_bwd_ms if dense_causal_bwd_ms > 0 else float("inf")
    hsa_fwd_vs_dense = hsa_fwd_ms / dense_causal_fwd_ms if dense_causal_fwd_ms > 0 else float("inf")
    hsa_fwd_bwd_vs_dense = hsa_fwd_bwd_ms / dense_causal_fwd_bwd_ms if dense_causal_fwd_bwd_ms > 0 else float("inf")
    plain_sparse_mask_fwd_ms = None
    plain_sparse_mask_bwd_ms = None
    plain_sparse_mask_fwd_bwd_ms = None
    hsa_fwd_bwd_vs_plain_sparse_mask = None
    prev_synth_fwd_ms = None
    prev_synth_bwd_ms = None
    prev_synth_fwd_bwd_ms = None
    hsa_fwd_bwd_vs_prev = None
    short_synth_fwd_ms = None
    short_synth_bwd_ms = None
    short_synth_fwd_bwd_ms = None
    hsa_fwd_bwd_vs_short = None
    sparse_mask_mixed_fwd_ms = None
    sparse_mask_mixed_bwd_ms = None
    sparse_mask_mixed_fwd_bwd_ms = None
    hsa_fwd_bwd_vs_sparse_mask_mixed = None
    if _should_measure_plain_sparse_mask_baseline(case):
        plain_env_updates = _plain_sparse_mask_baseline_env(case)
        plain_hsa_forward = lambda q, k, v: _run_sparse_attention(
            q, k, v, keep_ids, hash_ids, schedule, env_updates=plain_env_updates
        )
        plain_sparse_mask_fwd_ms = _measure_forward_ms(
            plain_hsa_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=plain_env_updates,
        )
        plain_sparse_mask_bwd_ms = _measure_backward_ms(
            plain_hsa_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=plain_env_updates,
        )
        plain_sparse_mask_fwd_bwd_ms = _measure_forward_backward_ms(
            plain_hsa_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=plain_env_updates,
        )
        hsa_fwd_bwd_vs_plain_sparse_mask = (
            hsa_fwd_bwd_ms / plain_sparse_mask_fwd_bwd_ms if plain_sparse_mask_fwd_bwd_ms > 0 else float("inf")
        )
    if _should_measure_previous_synthetic_baseline(case):
        prev_env_updates = _previous_synthetic_baseline_env(case)
        prev_hsa_forward = lambda q, k, v: _run_sparse_attention(
            q, k, v, keep_ids, hash_ids, schedule, env_updates=prev_env_updates
        )
        prev_synth_fwd_ms = _measure_forward_ms(
            prev_hsa_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=prev_env_updates,
        )
        prev_synth_bwd_ms = _measure_backward_ms(
            prev_hsa_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=prev_env_updates,
        )
        prev_synth_fwd_bwd_ms = _measure_forward_backward_ms(
            prev_hsa_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=prev_env_updates,
        )
        hsa_fwd_bwd_vs_prev = hsa_fwd_bwd_ms / prev_synth_fwd_bwd_ms if prev_synth_fwd_bwd_ms > 0 else float("inf")
    if _should_measure_short_synthetic_baseline(case):
        short_env_updates = _short_synthetic_baseline_env(case)
        short_hsa_forward = lambda q, k, v: _run_sparse_attention(
            q, k, v, keep_ids, hash_ids, schedule, env_updates=short_env_updates
        )
        short_synth_fwd_ms = _measure_forward_ms(
            short_hsa_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=short_env_updates,
        )
        short_synth_bwd_ms = _measure_backward_ms(
            short_hsa_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=short_env_updates,
        )
        short_synth_fwd_bwd_ms = _measure_forward_backward_ms(
            short_hsa_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=short_env_updates,
        )
        hsa_fwd_bwd_vs_short = hsa_fwd_bwd_ms / short_synth_fwd_bwd_ms if short_synth_fwd_bwd_ms > 0 else float("inf")
    if _should_measure_sparse_mask_mixed_backward_baseline(case):
        sparse_mask_env_updates = _sparse_mask_mixed_backward_baseline_env(case)
        sparse_mask_hsa_forward = lambda q, k, v: _run_sparse_attention(
            q, k, v, keep_ids, hash_ids, schedule, env_updates=sparse_mask_env_updates
        )
        sparse_mask_mixed_fwd_ms = _measure_forward_ms(
            sparse_mask_hsa_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=sparse_mask_env_updates,
        )
        sparse_mask_mixed_bwd_ms = _measure_backward_ms(
            sparse_mask_hsa_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=sparse_mask_env_updates,
        )
        sparse_mask_mixed_fwd_bwd_ms = _measure_forward_backward_ms(
            sparse_mask_hsa_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=sparse_mask_env_updates,
        )
        hsa_fwd_bwd_vs_sparse_mask_mixed = (
            hsa_fwd_bwd_ms / sparse_mask_mixed_fwd_bwd_ms if sparse_mask_mixed_fwd_bwd_ms > 0 else float("inf")
        )
    hdt_in_repo_dense_fwd_ms = _measure_forward_ms(
        dense_ref_forward,
        q_data,
        k_data,
        v_data,
        case.warmup_iters,
        case.benchmark_iters,
    )
    hdt_in_repo_dense_bwd_ms = _measure_backward_ms(
        dense_ref_forward,
        q_data,
        k_data,
        v_data,
        case.warmup_iters,
        case.benchmark_iters,
    )
    hdt_in_repo_dense_fwd_bwd_ms = _measure_forward_backward_ms(
        dense_ref_forward,
        q_data,
        k_data,
        v_data,
        case.warmup_iters,
        case.benchmark_iters,
    )
    external_result = _run_external_hdt_case_subprocess(case)
    hdt_vendor_status = str(external_result.get("status", "child_bad_payload"))
    hdt_vendor_fwd_ms = external_result.get("fwd_ms")
    hdt_vendor_bwd_ms = external_result.get("bwd_ms")
    hdt_vendor_fwd_bwd_ms = external_result.get("fwd_bwd_ms")

    line = (
        f"{case.name}: mode={_benchmark_mode_label(case)} {_benchmark_sparse_bwd_config_label()} "
        f"shape=(B={case.batch_size}, T={case.seqlen}, H={case.nheads}, KV={n_kv_heads}, D={case.headdim}) "
        f"hsa_max_diff={hsa_max_diff:.6f} hsa_mean_diff={hsa_mean_diff:.6f} "
        f"hsa_fwd_ms={hsa_fwd_ms:.3f} dense_causal_fwd_ms={dense_causal_fwd_ms:.3f} "
        f"hsa_fwd_vs_dense={hsa_fwd_vs_dense:.2f}x "
        f"hsa_bwd_ms={hsa_bwd_ms:.3f} dense_causal_bwd_ms={dense_causal_bwd_ms:.3f} "
        f"hsa_bwd_vs_dense={hsa_bwd_vs_dense:.2f}x "
        f"hsa_fwd_bwd_ms={hsa_fwd_bwd_ms:.3f} dense_causal_fwd_bwd_ms={dense_causal_fwd_bwd_ms:.3f} "
        f"hsa_fwd_bwd_vs_dense={hsa_fwd_bwd_vs_dense:.2f}x"
    )
    if hdt_vendor_status == "measured":
        line += (
            f" hdt_in_repo_dense_fwd_ms={hdt_in_repo_dense_fwd_ms:.3f}"
            f" hdt_in_repo_dense_bwd_ms={hdt_in_repo_dense_bwd_ms:.3f}"
            f" hdt_in_repo_dense_fwd_bwd_ms={hdt_in_repo_dense_fwd_bwd_ms:.3f}"
            f" hdt_vendor_fwd_ms={hdt_vendor_fwd_ms:.3f}"
            f" hdt_vendor_bwd_ms={hdt_vendor_bwd_ms:.3f}"
            f" hdt_vendor_fwd_bwd_ms={hdt_vendor_fwd_bwd_ms:.3f}"
        )
    else:
        line += (
            f" hdt_in_repo_dense_fwd_ms={hdt_in_repo_dense_fwd_ms:.3f}"
            f" hdt_in_repo_dense_bwd_ms={hdt_in_repo_dense_bwd_ms:.3f}"
            f" hdt_in_repo_dense_fwd_bwd_ms={hdt_in_repo_dense_fwd_bwd_ms:.3f}"
            f" hdt_vendor_status={hdt_vendor_status}"
        )
    if (
        plain_sparse_mask_fwd_bwd_ms is not None
        and plain_sparse_mask_bwd_ms is not None
        and plain_sparse_mask_fwd_ms is not None
    ):
        line += (
            f" sparse_mask_plain_label={_plain_sparse_mask_baseline_label()}"
            f" sparse_mask_plain_fwd_ms={plain_sparse_mask_fwd_ms:.3f}"
            f" sparse_mask_plain_bwd_ms={plain_sparse_mask_bwd_ms:.3f}"
            f" sparse_mask_plain_fwd_bwd_ms={plain_sparse_mask_fwd_bwd_ms:.3f}"
            f" hsa_fwd_bwd_vs_sparse_mask_plain={hsa_fwd_bwd_vs_plain_sparse_mask:.2f}x"
        )
    if prev_synth_fwd_bwd_ms is not None and prev_synth_bwd_ms is not None and prev_synth_fwd_ms is not None:
        line += (
            f" prev_synth_label={_previous_synthetic_baseline_label()}"
            f" prev_synth_fwd_ms={prev_synth_fwd_ms:.3f}"
            f" prev_synth_bwd_ms={prev_synth_bwd_ms:.3f}"
            f" prev_synth_fwd_bwd_ms={prev_synth_fwd_bwd_ms:.3f}"
            f" hsa_fwd_bwd_vs_prev={hsa_fwd_bwd_vs_prev:.2f}x"
        )
    if short_synth_fwd_bwd_ms is not None and short_synth_bwd_ms is not None and short_synth_fwd_ms is not None:
        line += (
            f" short_synth_label={_short_synthetic_baseline_label()}"
            f" short_synth_fwd_ms={short_synth_fwd_ms:.3f}"
            f" short_synth_bwd_ms={short_synth_bwd_ms:.3f}"
            f" short_synth_fwd_bwd_ms={short_synth_fwd_bwd_ms:.3f}"
            f" hsa_fwd_bwd_vs_short={hsa_fwd_bwd_vs_short:.2f}x"
        )
    if (
        sparse_mask_mixed_fwd_bwd_ms is not None
        and sparse_mask_mixed_bwd_ms is not None
        and sparse_mask_mixed_fwd_ms is not None
    ):
        line += (
            f" sparse_mask_mixed_bwd_label={_sparse_mask_mixed_backward_baseline_label()}"
            f" sparse_mask_mixed_fwd_ms={sparse_mask_mixed_fwd_ms:.3f}"
            f" sparse_mask_mixed_bwd_ms={sparse_mask_mixed_bwd_ms:.3f}"
            f" sparse_mask_mixed_fwd_bwd_ms={sparse_mask_mixed_fwd_bwd_ms:.3f}"
            f" hsa_fwd_bwd_vs_sparse_mask_mixed={hsa_fwd_bwd_vs_sparse_mask_mixed:.2f}x"
        )
    if synthetic_summary is not None:
        line += (
            f" synthetic_micro_fwd={1 if os.environ.get('FLASH_ATTN_HSA_SYNTHETIC_MICRO_FWD', '0') == '1' else 0}"
            f" synthetic_micro_bwd={1 if os.environ.get('FLASH_ATTN_HSA_SYNTHETIC_MICRO_BWD', '0') == '1' else 0}"
            f" synthetic_one_kernel_bwd={os.environ.get('FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD', 'off')}"
            f" synthetic_fused_bwd={1 if os.environ.get('FLASH_ATTN_HSA_SYNTHETIC_FUSED_BWD', 'off').strip().lower() in {'1', 'true', 'yes', 'on'} else 0}"
            f" synthetic_mode_label={_synthetic_mode_label()}"
            f" synthetic_bwd_kernel_mode={_synthetic_bwd_kernel_mode_label()}"
            f" synth_qgroups_per_cta={os.environ.get('FLASH_ATTN_HSA_SYNTHETIC_QGROUPS_PER_CTA', '2')}"
            f" synth_qgroups_per_cta_bwd={os.environ.get('FLASH_ATTN_HSA_SYNTHETIC_QGROUPS_PER_CTA_BWD', '2')}"
            f" synth_row_bwd_accum_mode={os.environ.get('FLASH_ATTN_HSA_SYNTHETIC_ROW_BWD_ACCUM_MODE', 'row_local')}"
            f" synthetic_short_bwd_mode={os.environ.get('FLASH_ATTN_HSA_SYNTHETIC_SHORT_BWD', 'off')}"
            f" synth_logical_block={synthetic_summary['logical_block_q']}x{synthetic_summary['logical_block_k']}"
            f" synth_max_packed_k={synthetic_summary['max_packed_k']}"
            f" synth_max_direct_segments={synthetic_summary['max_direct_segments']}"
            f" synth_tiles={synthetic_summary['num_tiles']}"
            f" synth_qgroups={synthetic_summary['forward_qgroups']}"
            f" synth_avg_q_rows={synthetic_summary['avg_q_rows']:.1f}"
            f" synth_avg_k_rows={synthetic_summary['avg_k_rows']:.1f}"
            f" synth_avg_logical_pairs={synthetic_summary['avg_logical_pairs']:.1f}"
            f" synth_fwd_tiles={synthetic_summary['forward_tiles']}"
            f" synth_bwd_tiles={synthetic_summary['backward_tiles']}"
            f" synth_fwd_buckets={synthetic_summary['forward_buckets']}"
            f" synth_bwd_buckets={synthetic_summary['backward_buckets']}"
            f" synth_fwd_avg_packed={synthetic_summary['forward_avg_packed_q']:.1f}x{synthetic_summary['forward_avg_packed_k']:.1f}"
            f" synth_bwd_avg_packed={synthetic_summary['backward_avg_packed_q']:.1f}x{synthetic_summary['backward_avg_packed_k']:.1f}"
            f" synth_fwd_avg_num_splits={synthetic_summary['forward_avg_num_splits']:.2f}"
            f" synth_fwd_avg_fill={synthetic_summary['forward_avg_fill']:.4f}"
            f" synth_fwd_fill_p50={synthetic_summary['forward_fill_p50']:.4f}"
            f" synth_fwd_fill_p90={synthetic_summary['forward_fill_p90']:.4f}"
            f" synth_fwd_avg_direct_segments={synthetic_summary['forward_avg_direct_segments']:.2f}"
            f" synth_fwd_avg_segment_k={synthetic_summary['forward_avg_segment_k_length']:.2f}"
            f" synth_fwd_avg_segment_fill={synthetic_summary['forward_avg_segment_fill']:.4f}"
            f" synth_fwd_segment_fill_p50={synthetic_summary['forward_segment_fill_p50']:.4f}"
            f" synth_fwd_segment_fill_p90={synthetic_summary['forward_segment_fill_p90']:.4f}"
            f" synth_fwd_avg_row_k={synthetic_summary['forward_avg_row_k']:.2f}"
            f" synth_fwd_row_k_p50={synthetic_summary['forward_row_k_p50']:.2f}"
            f" synth_fwd_row_k_p90={synthetic_summary['forward_row_k_p90']:.2f}"
            f" synth_fwd_avg_union_k={synthetic_summary['forward_avg_union_k']:.2f}"
            f" synth_bwd_avg_fill={synthetic_summary['backward_avg_fill']:.4f}"
        )
    print(line)


def _run_cases_once():
    _ensure_cuda_ready()
    print(f"device={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
    if _use_geometry_sweep_mode():
        _run_geometry_sweep_once()
        return
    if _use_sparse_profile_mode():
        for case in _get_selected_cases(
            env_var="FLASH_ATTN_HSA_PROFILE_CASES",
            default=(CANONICAL_CASES[1], CANONICAL_CASES[2], CANONICAL_CASES[3]),
        ):
            _profile_sparse_mask_case(case)
        return
    for case in _get_selected_cases(env_var="FLASH_ATTN_HSA_CASES", default=CANONICAL_CASES):
        if _is_long_context_case(case):
            _run_long_case(case)
        else:
            run_case(case)


def main():
    if os.environ.get("FLASH_ATTN_HSA_EXTERNAL_HDT_CHILD", "0") == "1":
        try:
            payload = _measure_external_hdt_case()
        except Exception as exc:
            payload = {"status": f"failed_{type(exc).__name__}"}
        print(json.dumps(payload), flush=True)
        return

    if os.environ.get("FLASH_ATTN_HSA_BENCH_CHILD", "0") == "1":
        attempt = os.environ.get("FLASH_ATTN_HSA_BENCH_ATTEMPT", "?")
        try:
            _run_cases_once()
            return
        except Exception as exc:
            print(
                f"hsa benchmark bootstrap failure on attempt {attempt}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(2) from None

    max_retries = int(os.environ.get("FLASH_ATTN_HSA_BENCH_MAX_RETRIES", "5"))
    child_env = os.environ.copy()
    child_env["FLASH_ATTN_HSA_BENCH_CHILD"] = "1"
    child_env.pop("FLASH_ATTN_HSA_BENCH_ATTEMPT", None)

    last_rc = 1
    for attempt in range(1, max_retries + 1):
        child_env["FLASH_ATTN_HSA_BENCH_ATTEMPT"] = str(attempt)
        print(f"hsa benchmark attempt {attempt}/{max_retries}", file=sys.stderr, flush=True)
        result = subprocess.run(
            [sys.executable, __file__],
            env=child_env,
            check=False,
        )
        if result.returncode == 0:
            return
        last_rc = result.returncode
    print(
        f"hsa benchmark failed after {max_retries} attempts with exit code {last_rc}",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(last_rc)


if __name__ == "__main__":
    main()
