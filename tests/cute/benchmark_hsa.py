import importlib.util
import json
import time
import os
import math
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
            flash_attn_hsa_func,
            flash_attn_hsa_sparse_func,
            hsa_reference_attention,
            schedule_to_attend_mask,
        )

        _FLASH_ATTN_IMPORTS = (
            build_hsa_schedule,
            flash_attn_func,
            flash_attn_hsa_func,
            flash_attn_hsa_sparse_func,
            hsa_reference_attention,
            schedule_to_attend_mask,
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
    BenchmarkCase(name="long-256k", batch_size=1, seqlen=262144, nheads=4, headdim=64, n_kv_heads=4, warmup_iters=1, benchmark_iters=1),
    BenchmarkCase(name="long-512k", batch_size=1, seqlen=524288, nheads=4, headdim=64, n_kv_heads=4, warmup_iters=1, benchmark_iters=1),
    BenchmarkCase(name="long-1M", batch_size=1, seqlen=1048576, nheads=4, headdim=64, n_kv_heads=4, warmup_iters=1, benchmark_iters=1),
    BenchmarkCase(name="long-2M", batch_size=1, seqlen=2097152, nheads=4, headdim=64, n_kv_heads=4, warmup_iters=1, benchmark_iters=1),
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


def _causal_sliding_window_pairs(seqlen: int, window_tokens: int) -> int:
    w = max(1, min(int(window_tokens), int(seqlen)))
    return w * seqlen - (w * (w - 1)) // 2


def _flop_matched_sliding_window_tokens(seqlen: int, allowed_pairs: int) -> int:
    if seqlen <= 1:
        return 1
    causal_pairs = seqlen * (seqlen + 1) // 2
    target_pairs = max(1, min(int(allowed_pairs), int(causal_pairs)))
    low = 1
    high = seqlen
    while low < high:
        mid = (low + high) // 2
        if _causal_sliding_window_pairs(seqlen, mid) < target_pairs:
            low = mid + 1
        else:
            high = mid
    best = low
    if best > 1:
        prev = best - 1
        if abs(_causal_sliding_window_pairs(seqlen, prev) - target_pairs) <= abs(
            _causal_sliding_window_pairs(seqlen, best) - target_pairs
        ):
            best = prev
    return best


def _flop_matched_sliding_window(seqlen: int, allowed_pairs: int) -> tuple[int, tuple[int, int]]:
    tokens = _flop_matched_sliding_window_tokens(seqlen, allowed_pairs)
    return tokens, (max(0, tokens - 1), 0)


def _should_attempt_dense_fa4_long(case: BenchmarkCase) -> bool:
    return True


def _log_sliding_window_tokens(seqlen: int) -> int:
    if seqlen <= 1:
        return 1
    return max(1, int(math.ceil(math.log2(seqlen))))


def _measure_triplet_or_status(
    forward_fn,
    q_data,
    k_data,
    v_data,
    warmup_iters: int,
    benchmark_iters: int,
    *,
    env_updates: dict[str, str | None] | None = None,
):
    try:
        return {
            "fwd_ms": _measure_forward_ms(
                forward_fn,
                q_data,
                k_data,
                v_data,
                warmup_iters,
                benchmark_iters,
                env_updates=env_updates,
            ),
            "bwd_ms": _measure_backward_ms(
                forward_fn,
                q_data,
                k_data,
                v_data,
                warmup_iters,
                benchmark_iters,
                env_updates=env_updates,
            ),
            "fwd_bwd_ms": _measure_forward_backward_ms(
                forward_fn,
                q_data,
                k_data,
                v_data,
                warmup_iters,
                benchmark_iters,
                env_updates=env_updates,
            ),
            "status": "measured",
        }
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        torch.cuda.empty_cache()
        return {
            "fwd_ms": None,
            "bwd_ms": None,
            "fwd_bwd_ms": None,
            "status": f"unavailable_{type(exc).__name__}",
        }


def _append_labeled_triplet_fields(
    parts: list[str],
    *,
    prefix: str,
    label: str,
    fwd_ms,
    bwd_ms,
    fwd_bwd_ms,
    status: str,
    extra_fields: list[str] | None = None,
):
    parts.append(f"{prefix}_label={label}")
    if extra_fields:
        parts.extend(extra_fields)
    if fwd_ms is not None and bwd_ms is not None and fwd_bwd_ms is not None:
        parts.extend(
            [
                f"{prefix}_fwd_ms={fwd_ms:.3f}",
                f"{prefix}_bwd_ms={bwd_ms:.3f}",
                f"{prefix}_fwd_bwd_ms={fwd_bwd_ms:.3f}",
            ]
        )
    else:
        parts.append(f"{prefix}_status={status}")


def _append_dense_triplet_fields(
    parts: list[str],
    *,
    fwd_ms,
    bwd_ms,
    fwd_bwd_ms,
    status: str,
):
    if fwd_ms is not None and bwd_ms is not None and fwd_bwd_ms is not None:
        parts.extend(
            [
                f"dense_fa4_fwd_ms={fwd_ms:.3f}",
                f"dense_fa4_bwd_ms={bwd_ms:.3f}",
                f"dense_fa4_fwd_bwd_ms={fwd_bwd_ms:.3f}",
            ]
        )
    else:
        parts.append(f"dense_fa4_status={status}")


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
        variant = os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT", "auto").strip().lower()
        long_mode = os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_LONG_BWD_MODE", "one_kernel").strip().lower()
        long_keys_per_cta = os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_LONG_KEYS_PER_CTA", "4").strip().lower()
        if variant == "warpgroup":
            long_keys_per_cta = "8"
        pingpong_mode = os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_PINGPONG", "off").strip().lower()
        if variant in {
            "bucket_dense",
            "bucket_dense_saved_packed",
            "bucket_dense_saved_prob",
            "bucket_dense_two_pass",
            "bucket_dense_dualrow",
            "bucket_dense_tc",
        }:
            return f"one_kernel_{one_kernel_mode}_variant_{variant}_long_{long_mode}_pingpong_{pingpong_mode}"
        return (
            f"one_kernel_{one_kernel_mode}_variant_{variant}_long_{long_mode}"
            f"_keys_{long_keys_per_cta}_pingpong_{pingpong_mode}"
        )
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
    return case.name != "sentence-only"


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
    return case.name != "sentence-only"


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


def _should_measure_one_kernel_synthetic_baseline(case: BenchmarkCase) -> bool:
    return case.name != "sentence-only" and (case.n_kv_heads is None or case.n_kv_heads == case.nheads)


def _one_kernel_synthetic_baseline_label() -> str:
    return "direct_micro_one_kernel_bwd_legacy"


def _one_kernel_synthetic_baseline_env(case: BenchmarkCase) -> dict[str, str | None]:
    env = dict(_previous_synthetic_baseline_env(case))
    env.update(
        {
            "FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD": "on",
            "FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT": "baseline",
            "FLASH_ATTN_HSA_SYNTHETIC_LONG_BWD_MODE": "one_kernel",
            "FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_PINGPONG": "off",
            "FLASH_ATTN_HSA_SYNTHETIC_SPLIT_BWD": "off",
            "FLASH_ATTN_HSA_SYNTHETIC_SHORT_BWD": "off",
            "FLASH_ATTN_HSA_SYNTHETIC_FUSED_BWD": "off",
        }
    )
    return env


def _one_kernel_synthetic_pingpong_label() -> str:
    return "direct_micro_one_kernel_bwd_pingpong"


def _one_kernel_synthetic_pingpong_env(case: BenchmarkCase) -> dict[str, str | None]:
    env = dict(_one_kernel_synthetic_baseline_env(case))
    env.update(
        {
            "FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_PINGPONG": "on",
        }
    )
    return env


def _one_kernel_synthetic_short_label() -> str:
    return "direct_micro_one_kernel_bwd_short"


def _one_kernel_synthetic_short_env(case: BenchmarkCase) -> dict[str, str | None]:
    env = dict(_one_kernel_synthetic_baseline_env(case))
    env.update(
        {
            "FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT": "short",
        }
    )
    return env


def _one_kernel_synthetic_long_label() -> str:
    return "direct_micro_one_kernel_bwd_long_4"


def _one_kernel_synthetic_bucket_dense_label() -> str:
    return "direct_micro_bucket_dense_bwd"


def _one_kernel_synthetic_bucket_dense_two_pass_label() -> str:
    return "direct_micro_bucket_dense_two_pass_bwd"


def _one_kernel_synthetic_bucket_dense_saved_prob_label() -> str:
    return "direct_micro_bucket_dense_saved_prob_bwd"


def _one_kernel_synthetic_bucket_dense_saved_packed_label() -> str:
    return "direct_micro_bucket_dense_saved_packed_bwd"


def _one_kernel_synthetic_bucket_dense_tc_label() -> str:
    return "direct_micro_bucket_dense_tc_bwd"


def _one_kernel_synthetic_bucket_dense_dualrow_label() -> str:
    return "direct_micro_bucket_dense_dualrow_bwd"


def _one_kernel_synthetic_bucket_dense_env(case: BenchmarkCase) -> dict[str, str | None]:
    env = dict(_previous_synthetic_baseline_env(case))
    env.update(
        {
            "FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD": "on",
            "FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT": "bucket_dense",
            "FLASH_ATTN_HSA_SYNTHETIC_LONG_BWD_MODE": "one_kernel",
            "FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_PINGPONG": "off",
            "FLASH_ATTN_HSA_SYNTHETIC_QGROUPS_PER_CTA_BWD": "2",
            "FLASH_ATTN_HSA_SYNTHETIC_SPLIT_BWD": "off",
            "FLASH_ATTN_HSA_SYNTHETIC_SHORT_BWD": "off",
            "FLASH_ATTN_HSA_SYNTHETIC_FUSED_BWD": "off",
        }
    )
    return env


def _one_kernel_synthetic_bucket_dense_tc_env(case: BenchmarkCase) -> dict[str, str | None]:
    env = dict(_one_kernel_synthetic_bucket_dense_env(case))
    env.update({"FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT": "bucket_dense_tc"})
    return env


def _one_kernel_synthetic_bucket_dense_two_pass_env(case: BenchmarkCase) -> dict[str, str | None]:
    env = dict(_one_kernel_synthetic_bucket_dense_env(case))
    env.update({"FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT": "bucket_dense_two_pass"})
    return env


def _one_kernel_synthetic_bucket_dense_saved_prob_env(case: BenchmarkCase) -> dict[str, str | None]:
    env = dict(_one_kernel_synthetic_bucket_dense_env(case))
    env.update({"FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT": "bucket_dense_saved_prob"})
    return env


def _one_kernel_synthetic_bucket_dense_saved_packed_env(case: BenchmarkCase) -> dict[str, str | None]:
    env = dict(_one_kernel_synthetic_bucket_dense_env(case))
    env.update({"FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT": "bucket_dense_saved_packed"})
    return env


def _one_kernel_synthetic_bucket_dense_dualrow_env(case: BenchmarkCase) -> dict[str, str | None]:
    env = dict(_one_kernel_synthetic_bucket_dense_env(case))
    env.update({"FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT": "bucket_dense_dualrow"})
    return env


def _one_kernel_synthetic_long_env(case: BenchmarkCase, *, keys_per_cta: int = 4) -> dict[str, str | None]:
    env = dict(_one_kernel_synthetic_baseline_env(case))
    env.update(
        {
            "FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT": "long",
            "FLASH_ATTN_HSA_SYNTHETIC_LONG_BWD_MODE": "one_kernel",
            "FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_LONG_KEYS_PER_CTA": str(keys_per_cta),
        }
    )
    return env


def _one_kernel_synthetic_long_8_label() -> str:
    return "direct_micro_one_kernel_bwd_long_8"


def _one_kernel_synthetic_long_8_env(case: BenchmarkCase) -> dict[str, str | None]:
    return _one_kernel_synthetic_long_env(case, keys_per_cta=8)


def _one_kernel_synthetic_long_16_label() -> str:
    return "direct_micro_one_kernel_bwd_long_16"


def _one_kernel_synthetic_long_16_env(case: BenchmarkCase) -> dict[str, str | None]:
    return _one_kernel_synthetic_long_env(case, keys_per_cta=16)


def _one_kernel_synthetic_two_stage_label() -> str:
    return "direct_micro_two_stage_bwd_long"


def _one_kernel_synthetic_two_stage_env(case: BenchmarkCase) -> dict[str, str | None]:
    env = dict(_one_kernel_synthetic_long_env(case))
    env.update(
        {
            "FLASH_ATTN_HSA_SYNTHETIC_LONG_BWD_MODE": "two_stage",
        }
    )
    return env


def _one_kernel_synthetic_persistent_label() -> str:
    return "direct_micro_persistent_bwd_long"


def _one_kernel_synthetic_persistent_env(case: BenchmarkCase) -> dict[str, str | None]:
    env = dict(_one_kernel_synthetic_long_env(case))
    env.update(
        {
            "FLASH_ATTN_HSA_SYNTHETIC_LONG_BWD_MODE": "persistent",
        }
    )
    return env


def _one_kernel_synthetic_persistent_member_tiled_label() -> str:
    return "direct_micro_persistent_member_tiled_bwd_long"


def _one_kernel_synthetic_persistent_member_tiled_env(case: BenchmarkCase) -> dict[str, str | None]:
    env = dict(_one_kernel_synthetic_long_env(case))
    env.update(
        {
            "FLASH_ATTN_HSA_SYNTHETIC_LONG_BWD_MODE": "persistent_member_tiled",
        }
    )
    return env


def _should_measure_plain_sparse_mask_baseline(case: BenchmarkCase) -> bool:
    return case.name != "sentence-only"


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
    return case.name != "sentence-only" and (case.n_kv_heads is None or case.n_kv_heads == case.nheads)


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


def _use_bucket_dense_long_profile_mode() -> bool:
    return os.environ.get("FLASH_ATTN_HSA_PROFILE_BUCKET_DENSE_LONG", "0") == "1"


def _get_bucket_dense_long_profile_target_mode() -> str:
    value = os.environ.get("FLASH_ATTN_HSA_PROFILE_TARGET_MODE", "all").strip().lower()
    if value not in {
        "all",
        "bucket_dense",
        "bucket_dense_saved_packed",
        "bucket_dense_saved_prob",
        "bucket_dense_two_pass",
        "bucket_dense_dualrow",
        "bucket_dense_tc",
        "hybrid",
        "long4",
    }:
        value = "all"
    return value


def _use_geometry_sweep_mode() -> bool:
    return os.environ.get("FLASH_ATTN_HSA_GEOMETRY_SWEEP", "0") == "1"


def _include_long_experimental_comparators() -> bool:
    return os.environ.get("FLASH_ATTN_HSA_BENCHMARK_INCLUDE_LONG_EXPERIMENTAL", "0") == "1"


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


def _summarize_long_tile_reuse(runtime, *, keys_per_tile: int = 8) -> dict[str, float | int]:
    metadata = runtime.forward_synthetic_grid
    if metadata is None or metadata.forward_execution_plan is None:
        return {
            "tile_keys": keys_per_tile,
            "tiles": 0,
            "avg_occurrences": 0.0,
            "max_occurrences": 0,
            "avg_unique_members": 0.0,
            "max_unique_members": 0,
        }
    direct_plan = metadata.forward_execution_plan.get("direct_execution_plan")
    if direct_plan is None:
        return {
            "tile_keys": keys_per_tile,
            "tiles": 0,
            "avg_occurrences": 0.0,
            "max_occurrences": 0,
            "avg_unique_members": 0.0,
            "max_unique_members": 0,
        }
    row_plan = direct_plan.get("row_compact_plan")
    if row_plan is None:
        return {
            "tile_keys": keys_per_tile,
            "tiles": 0,
            "avg_occurrences": 0.0,
            "max_occurrences": 0,
            "avg_unique_members": 0.0,
            "max_unique_members": 0,
        }

    occ_counts: list[int] = []
    unique_member_counts: list[int] = []
    bucket_unique_ranges = row_plan["bucket_unique_key_range"]
    bucket_occ_ptr_ranges = row_plan["bucket_unique_key_occurrence_ptr_range"]
    bucket_packed_q = direct_plan["bucket_packed_q"]
    bucket_packed_k = direct_plan["bucket_packed_k"]
    bucket_max_occ = row_plan.get("bucket_max_unique_key_occurrences", [0] * len(bucket_packed_q))
    all_member_idx = row_plan["bucket_unique_key_member_idx"].detach().cpu()
    all_occ_ptr = row_plan["bucket_unique_key_occurrence_row_ptr"].detach().cpu()

    for bucket_idx, packed_q in enumerate(bucket_packed_q):
        if int(packed_q) != 2 or int(bucket_packed_k[bucket_idx]) > 16 or int(bucket_max_occ[bucket_idx]) > 8:
            continue
        unique_key_start, unique_key_end = bucket_unique_ranges[bucket_idx]
        if unique_key_end <= unique_key_start:
            continue
        unique_ptr_start, unique_ptr_end = bucket_occ_ptr_ranges[bucket_idx]
        occ_ptr = all_occ_ptr[unique_ptr_start:unique_ptr_end]
        if occ_ptr.numel() <= 1:
            continue
        member_base = int(occ_ptr[0].item())
        member_end = int(occ_ptr[-1].item())
        member_idx = all_member_idx[member_base:member_end]
        key_count = unique_key_end - unique_key_start
        num_tiles = (key_count + keys_per_tile - 1) // keys_per_tile
        for tile_idx in range(num_tiles):
            key_start_rel = tile_idx * keys_per_tile
            key_end_rel = min(key_start_rel + keys_per_tile, key_count)
            occ_start = int(occ_ptr[key_start_rel].item())
            occ_end = int(occ_ptr[key_end_rel].item())
            tile_occ = occ_end - occ_start
            occ_counts.append(tile_occ)
            if tile_occ > 0:
                tile_members = member_idx[(occ_start - member_base):(occ_end - member_base)]
                unique_member_counts.append(int(tile_members.unique().numel()))
            else:
                unique_member_counts.append(0)

    if not occ_counts:
        return {
            "tile_keys": keys_per_tile,
            "tiles": 0,
            "avg_occurrences": 0.0,
            "max_occurrences": 0,
            "avg_unique_members": 0.0,
            "max_unique_members": 0,
        }

    occ_tensor = torch.tensor(occ_counts, dtype=torch.float32)
    member_tensor = torch.tensor(unique_member_counts, dtype=torch.float32)
    return {
        "tile_keys": keys_per_tile,
        "tiles": len(occ_counts),
        "avg_occurrences": float(occ_tensor.mean().item()),
        "max_occurrences": int(max(occ_counts)),
        "avg_unique_members": float(member_tensor.mean().item()),
        "max_unique_members": int(max(unique_member_counts)),
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
    _, _, _, flash_attn_hsa_sparse_func, _, _ = _lazy_flash_attn_imports()
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
    build_hsa_schedule, _, _, _, _, _ = _lazy_flash_attn_imports()
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


def _measure_grad_finite_status(forward_fn, q_data, k_data, v_data, *, env_updates=None):
    env_updates = env_updates or {}
    with _temporary_env(**env_updates):
        q = q_data.clone().requires_grad_(True)
        k = k_data.clone().requires_grad_(True)
        v = v_data.clone().requires_grad_(True)
        loss = forward_fn(q, k, v).float().square().mean()
        dq, dk, dv = torch.autograd.grad(loss, (q, k, v))
    q_nonfinite = int((~torch.isfinite(dq)).sum().item())
    k_nonfinite = int((~torch.isfinite(dk)).sum().item())
    v_nonfinite = int((~torch.isfinite(dv)).sum().item())
    return {
        "finite": q_nonfinite == 0 and k_nonfinite == 0 and v_nonfinite == 0,
        "q_nonfinite": q_nonfinite,
        "k_nonfinite": k_nonfinite,
        "v_nonfinite": v_nonfinite,
    }


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
    build_hsa_schedule, flash_attn_func, _, flash_attn_hsa_sparse_func, _, _ = _lazy_flash_attn_imports()
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


def _profile_bucket_dense_long_case(case: BenchmarkCase, target_mode: str):
    build_hsa_schedule, _, _, _, _, _ = _lazy_flash_attn_imports()
    device = "cuda"
    dtype = torch.bfloat16
    keep_ids, hash_ids = _make_hsa_metadata(case.batch_size, case.seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    n_kv_heads = case.n_kv_heads if case.n_kv_heads is not None else case.nheads
    q_data = torch.randn(case.batch_size, case.seqlen, case.nheads, case.headdim, device=device, dtype=dtype)
    k_data = torch.randn(case.batch_size, case.seqlen, n_kv_heads, case.headdim, device=device, dtype=dtype)
    v_data = torch.randn(case.batch_size, case.seqlen, n_kv_heads, case.headdim, device=device, dtype=dtype)

    mode_specs = {
        "bucket_dense": {
            "env": _one_kernel_synthetic_bucket_dense_env(case),
            "patterns": ("FlashHSASyntheticDirectRowMicroBwdBucketDenseSm100",),
        },
        "bucket_dense_saved_packed": {
            "env": _one_kernel_synthetic_bucket_dense_saved_packed_env(case),
            "patterns": (
                "FlashHSASyntheticDirectMicroBwdDensePackedInputSm100",
                "FlashHSASyntheticDirectMicroBwdMaskedPackedInputSm100",
            ),
        },
        "bucket_dense_saved_prob": {
            "env": _one_kernel_synthetic_bucket_dense_saved_prob_env(case),
            "patterns": ("FlashHSASyntheticDirectRowMicroBwdBucketDenseSavedProbSm100",),
        },
        "bucket_dense_two_pass": {
            "env": _one_kernel_synthetic_bucket_dense_two_pass_env(case),
            "patterns": (
                "FlashHSASyntheticDirectRowMicroBwdBucketDenseTwoPassStage1Sm100",
                "FlashHSASyntheticDirectRowMicroBwdBucketDenseTwoPassReductionSm100",
            ),
        },
        "bucket_dense_dualrow": {
            "env": _one_kernel_synthetic_bucket_dense_dualrow_env(case),
            "patterns": ("FlashHSASyntheticDirectRowMicroBwdBucketDenseDualrowSm100",),
        },
        "bucket_dense_tc": {
            "env": _one_kernel_synthetic_bucket_dense_tc_env(case),
            "patterns": ("FlashHSASyntheticDirectRowMicroBwdBucketDenseTcSm100",),
        },
        "hybrid": {
            "env": _sparse_mask_mixed_backward_baseline_env(case),
            "patterns": (
                "FlashAttentionBackwardSm100",
                "flash_attncuteflash_bwd_sm100",
                "FlashAttentionBackwardPreprocess",
                "flash_attncuteflash_bwd_preprocess",
                "FlashAttentionBackwardPostprocess",
                "flash_attncuteflash_bwd_postprocess",
            ),
        },
        "long4": {
            "env": _one_kernel_synthetic_long_env(case, keys_per_cta=4),
            "patterns": (
                "FlashHSASyntheticDirectRowMicroBwdOneKernelWritebackSm100",
                "FlashHSASyntheticDirectRowMicroBwdOneKernelSm100",
            ),
        },
    }
    selected_labels = (
        [
            "bucket_dense",
            "bucket_dense_saved_packed",
            "bucket_dense_saved_prob",
            "bucket_dense_two_pass",
            "bucket_dense_dualrow",
            "hybrid",
            "long4",
        ]
        if target_mode == "all"
        else [target_mode]
    )
    warmups = max(1, min(case.warmup_iters, 2))
    benchmark_iters = max(1, min(case.benchmark_iters, 2))

    print(
        f"bucket_dense_long_profile_case={case.name} target_mode={target_mode} "
        f"shape=(B={case.batch_size}, T={case.seqlen}, H={case.nheads}, KV={n_kv_heads}, D={case.headdim})"
    )
    for label in selected_labels:
        env_updates = mode_specs[label]["env"]
        include_patterns = mode_specs[label]["patterns"]
        forward_fn = lambda q, k, v, env_updates=env_updates: _run_sparse_attention(
            q, k, v, keep_ids, hash_ids, schedule, env_updates=env_updates
        )

        fwd_ms = _measure_forward_ms(
            forward_fn,
            q_data,
            k_data,
            v_data,
            warmups,
            benchmark_iters,
            env_updates=env_updates,
        )
        bwd_ms = _measure_backward_ms(
            forward_fn,
            q_data,
            k_data,
            v_data,
            warmups,
            benchmark_iters,
            env_updates=env_updates,
        )
        fwd_bwd_ms = _measure_forward_backward_ms(
            forward_fn,
            q_data,
            k_data,
            v_data,
            warmups,
            benchmark_iters,
            env_updates=env_updates,
        )

        with _temporary_env(**env_updates):
            q = q_data.clone().requires_grad_(True)
            k = k_data.clone().requires_grad_(True)
            v = v_data.clone().requires_grad_(True)
            loss = _unwrap_output(forward_fn(q, k, v)).float().square().mean()

            for _ in range(warmups):
                torch.autograd.grad(loss, (q, k, v), retain_graph=True)
            torch.cuda.synchronize()

            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,
                with_stack=False,
                profile_memory=False,
            ) as prof:
                torch.autograd.grad(loss, (q, k, v), retain_graph=True)
                torch.cuda.synchronize()

        top_rows = _top_cuda_rows(prof, limit=8)
        backward_rows = _top_cuda_rows(prof, include_patterns=include_patterns, limit=6)
        backward_kernel_key, backward_kernel_us = backward_rows[0] if backward_rows else ("", 0.0)
        print(
            f"bucket_dense_long_profile_mode case={case.name} label={label} "
            f"fwd_ms={fwd_ms:.3f} bwd_ms={bwd_ms:.3f} fwd_bwd_ms={fwd_bwd_ms:.3f} "
            f"backward_kernel_key={json.dumps(backward_kernel_key)} "
            f"backward_kernel_self_cuda_us={backward_kernel_us:.1f}"
        )
        for key, value_us in top_rows:
            print(
                f"bucket_dense_long_profile_row case={case.name} label={label} "
                f"self_cuda_us={value_us:.1f} kernel={json.dumps(key)}"
            )


def _run_geometry_sweep_case(case: BenchmarkCase, *, block_q: int, block_k: int, subtile_factor: int):
    build_hsa_schedule, flash_attn_func, _, _, hsa_reference_attention, _ = _lazy_flash_attn_imports()
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

    (
        build_hsa_schedule,
        flash_attn_func,
        flash_attn_hsa_func,
        _,
        hsa_reference_attention,
        _,
    ) = _lazy_flash_attn_imports()
    device = "cuda"
    dtype = torch.bfloat16
    keep_ids, hash_ids = _make_hsa_metadata(case.batch_size, case.seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    n_kv_heads = case.n_kv_heads if case.n_kv_heads is not None else case.nheads
    q_data = torch.randn(case.batch_size, case.seqlen, case.nheads, case.headdim, device=device, dtype=dtype)
    k_data = torch.randn(case.batch_size, case.seqlen, n_kv_heads, case.headdim, device=device, dtype=dtype)
    v_data = torch.randn(case.batch_size, case.seqlen, n_kv_heads, case.headdim, device=device, dtype=dtype)
    env_updates = _benchmark_env_for_case(case)

    with _temporary_env(**_one_kernel_synthetic_long_env(case)):
        runtime = hsa_module._get_hsa_block_sparse_runtime(schedule, q_data, k_data)
    occupancy = _summarize_sparse_backward_occupancy(
        runtime.backward_sparse,
        runtime.backward_packed_masks,
        seqlen=case.seqlen,
    )
    tile_reuse = _summarize_long_tile_reuse(runtime, keys_per_tile=8)

    hsa_forward = lambda q, k, v: _run_sparse_attention(q, k, v, keep_ids, hash_ids, schedule, env_updates=env_updates)
    mask_mod_forward = lambda q, k, v: _unwrap_output(
        flash_attn_hsa_func(q, k, v, keep_ids=keep_ids, hash_ids=hash_ids)
    )
    dense_causal_forward = lambda q, k, v: _unwrap_output(flash_attn_func(q, k, v, causal=True))
    sliding_log_tokens = _log_sliding_window_tokens(case.seqlen)
    sliding_log_window = (max(0, sliding_log_tokens - 1), 0)
    sliding_log_pairs = _causal_sliding_window_pairs(case.seqlen, sliding_log_tokens)
    sliding_flopmatched_tokens, sliding_flopmatched_window = _flop_matched_sliding_window(
        case.seqlen, occupancy["allowed_pairs"]
    )
    sliding_flopmatched_pairs = _causal_sliding_window_pairs(case.seqlen, sliding_flopmatched_tokens)
    sliding_log_forward = lambda q, k, v: _unwrap_output(
        flash_attn_func(q, k, v, causal=True, window_size=sliding_log_window)
    )
    sliding_flopmatched_forward = lambda q, k, v: _unwrap_output(
        flash_attn_func(q, k, v, causal=True, window_size=sliding_flopmatched_window)
    )
    hsa_fwd_ms = _measure_forward_ms(
        hsa_forward,
        q_data,
        k_data,
        v_data,
        case.warmup_iters,
        case.benchmark_iters,
        env_updates=env_updates,
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
    hsa_fwd_bwd_ms = _measure_forward_backward_ms(
        hsa_forward,
        q_data,
        k_data,
        v_data,
        case.warmup_iters,
        case.benchmark_iters,
        env_updates=env_updates,
    )
    dense_causal_fwd_ms = None
    dense_causal_bwd_ms = None
    dense_causal_fwd_bwd_ms = None
    dense_causal_bwd_status = "unmeasured"
    dense_causal_fwd_status = "unmeasured"
    dense_causal_fwd_bwd_status = "unmeasured"
    hsa_fwd_vs_dense = None
    hsa_bwd_vs_dense = None
    hsa_fwd_bwd_vs_dense = None
    mask_mod_result = _measure_triplet_or_status(
        mask_mod_forward,
        q_data,
        k_data,
        v_data,
        case.warmup_iters,
        case.benchmark_iters,
    )
    sliding_log_result = _measure_triplet_or_status(
        sliding_log_forward,
        q_data,
        k_data,
        v_data,
        case.warmup_iters,
        case.benchmark_iters,
    )
    sliding_flopmatched_result = _measure_triplet_or_status(
        sliding_flopmatched_forward,
        q_data,
        k_data,
        v_data,
        case.warmup_iters,
        case.benchmark_iters,
    )
    sliding_fa4_fwd_ms = None
    sliding_fa4_bwd_ms = None
    sliding_fa4_fwd_bwd_ms = None
    sliding_fa4_status = "unmeasured"
    sliding_fa4_vs_hybrid = None
    sliding_fa4_vs_bucket_dense = None
    if _should_attempt_dense_fa4_long(case):
        dense_result = _measure_triplet_or_status(
            dense_causal_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
        )
        dense_causal_fwd_ms = dense_result["fwd_ms"]
        dense_causal_bwd_ms = dense_result["bwd_ms"]
        dense_causal_fwd_bwd_ms = dense_result["fwd_bwd_ms"]
        dense_causal_fwd_status = dense_result["status"]
        dense_causal_bwd_status = dense_result["status"]
        dense_causal_fwd_bwd_status = dense_result["status"]
        if dense_causal_fwd_ms is not None and dense_causal_bwd_ms is not None and dense_causal_fwd_bwd_ms is not None:
            hsa_fwd_vs_dense = hsa_fwd_ms / dense_causal_fwd_ms if dense_causal_fwd_ms > 0 else float("inf")
            hsa_bwd_vs_dense = hsa_bwd_ms / dense_causal_bwd_ms if dense_causal_bwd_ms > 0 else float("inf")
            hsa_fwd_bwd_vs_dense = hsa_fwd_bwd_ms / dense_causal_fwd_bwd_ms if dense_causal_fwd_bwd_ms > 0 else float("inf")
    sliding_fa4_fwd_ms = sliding_flopmatched_result["fwd_ms"]
    sliding_fa4_bwd_ms = sliding_flopmatched_result["bwd_ms"]
    sliding_fa4_fwd_bwd_ms = sliding_flopmatched_result["fwd_bwd_ms"]
    sliding_fa4_status = sliding_flopmatched_result["status"]
    hybrid_fwd_ms = None
    hybrid_bwd_ms = None
    hybrid_fwd_bwd_ms = None
    hybrid_vs_dense = None
    hybrid_status = "unavailable_unsupported_case"
    bucket_dense_fwd_ms = None
    bucket_dense_bwd_ms = None
    bucket_dense_fwd_bwd_ms = None
    bucket_dense_vs_dense = None
    bucket_dense_vs_hybrid = None
    bucket_dense_saved_prob_fwd_ms = None
    bucket_dense_saved_prob_bwd_ms = None
    bucket_dense_saved_prob_fwd_bwd_ms = None
    bucket_dense_saved_prob_vs_dense = None
    bucket_dense_saved_prob_vs_hybrid = None
    bucket_dense_two_pass_fwd_ms = None
    bucket_dense_two_pass_bwd_ms = None
    bucket_dense_two_pass_fwd_bwd_ms = None
    bucket_dense_two_pass_vs_dense = None
    bucket_dense_two_pass_vs_hybrid = None
    bucket_dense_tc_fwd_ms = None
    bucket_dense_tc_bwd_ms = None
    bucket_dense_tc_fwd_bwd_ms = None
    bucket_dense_tc_vs_dense = None
    bucket_dense_tc_vs_hybrid = None
    bucket_dense_dualrow_fwd_ms = None
    bucket_dense_dualrow_bwd_ms = None
    bucket_dense_dualrow_fwd_bwd_ms = None
    bucket_dense_dualrow_vs_dense = None
    bucket_dense_dualrow_vs_hybrid = None
    one_kernel_long_4_fwd_ms = None
    one_kernel_long_4_bwd_ms = None
    one_kernel_long_4_fwd_bwd_ms = None
    one_kernel_long_4_vs_dense = None
    one_kernel_long_4_vs_hybrid = None
    one_kernel_long_8_fwd_ms = None
    one_kernel_long_8_bwd_ms = None
    one_kernel_long_8_fwd_bwd_ms = None
    one_kernel_long_8_vs_dense = None
    one_kernel_long_8_vs_hybrid = None
    one_kernel_long_16_fwd_ms = None
    one_kernel_long_16_bwd_ms = None
    one_kernel_long_16_fwd_bwd_ms = None
    one_kernel_long_16_vs_dense = None
    one_kernel_long_16_vs_hybrid = None
    plain_sparse_mask_bwd_ms = None
    hsa_bwd_vs_plain_sparse_mask = None
    prev_synth_bwd_ms = None
    hsa_bwd_vs_prev = None
    short_synth_bwd_ms = None
    hsa_bwd_vs_short = None
    one_kernel_synth_bwd_ms = None
    hsa_bwd_vs_one_kernel = None
    one_kernel_short_bwd_ms = None
    hsa_bwd_vs_one_kernel_short = None
    one_kernel_long_bwd_ms = None
    hsa_bwd_vs_one_kernel_long = None
    two_stage_long_bwd_ms = None
    hsa_bwd_vs_two_stage_long = None
    persistent_long_bwd_ms = None
    hsa_bwd_vs_persistent_long = None
    persistent_member_tiled_long_bwd_ms = None
    hsa_bwd_vs_persistent_member_tiled_long = None
    one_kernel_pingpong_bwd_ms = None
    hsa_bwd_vs_one_kernel_pingpong = None
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
    if _should_measure_one_kernel_synthetic_baseline(case):
        one_kernel_env_updates = _one_kernel_synthetic_baseline_env(case)
        one_kernel_hsa_forward = lambda q, k, v: _run_sparse_attention(
            q, k, v, keep_ids, hash_ids, schedule, env_updates=one_kernel_env_updates
        )
        one_kernel_synth_bwd_ms = _measure_backward_ms(
            one_kernel_hsa_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=one_kernel_env_updates,
        )
        hsa_bwd_vs_one_kernel = hsa_bwd_ms / one_kernel_synth_bwd_ms if one_kernel_synth_bwd_ms > 0 else float("inf")
        one_kernel_short_env_updates = _one_kernel_synthetic_short_env(case)
        one_kernel_short_forward = lambda q, k, v: _run_sparse_attention(
            q, k, v, keep_ids, hash_ids, schedule, env_updates=one_kernel_short_env_updates
        )
        one_kernel_short_bwd_ms = _measure_backward_ms(
            one_kernel_short_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=one_kernel_short_env_updates,
        )
        hsa_bwd_vs_one_kernel_short = (
            hsa_bwd_ms / one_kernel_short_bwd_ms if one_kernel_short_bwd_ms > 0 else float("inf")
        )
        one_kernel_long_env_updates = _one_kernel_synthetic_long_env(case)
        one_kernel_long_forward = lambda q, k, v: _run_sparse_attention(
            q, k, v, keep_ids, hash_ids, schedule, env_updates=one_kernel_long_env_updates
        )
        one_kernel_long_bwd_ms = _measure_backward_ms(
            one_kernel_long_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=one_kernel_long_env_updates,
        )
        hsa_bwd_vs_one_kernel_long = (
            hsa_bwd_ms / one_kernel_long_bwd_ms if one_kernel_long_bwd_ms > 0 else float("inf")
        )
        if _include_long_experimental_comparators():
            two_stage_long_env_updates = _one_kernel_synthetic_two_stage_env(case)
            two_stage_long_forward = lambda q, k, v: _run_sparse_attention(
                q, k, v, keep_ids, hash_ids, schedule, env_updates=two_stage_long_env_updates
            )
            two_stage_long_bwd_ms = _measure_backward_ms(
                two_stage_long_forward,
                q_data,
                k_data,
                v_data,
                case.warmup_iters,
                case.benchmark_iters,
                env_updates=two_stage_long_env_updates,
            )
            hsa_bwd_vs_two_stage_long = (
                hsa_bwd_ms / two_stage_long_bwd_ms if two_stage_long_bwd_ms > 0 else float("inf")
            )
            persistent_long_env_updates = _one_kernel_synthetic_persistent_env(case)
            persistent_long_forward = lambda q, k, v: _run_sparse_attention(
                q, k, v, keep_ids, hash_ids, schedule, env_updates=persistent_long_env_updates
            )
            persistent_long_bwd_ms = _measure_backward_ms(
                persistent_long_forward,
                q_data,
                k_data,
                v_data,
                case.warmup_iters,
                case.benchmark_iters,
                env_updates=persistent_long_env_updates,
            )
            hsa_bwd_vs_persistent_long = (
                hsa_bwd_ms / persistent_long_bwd_ms if persistent_long_bwd_ms > 0 else float("inf")
            )
            persistent_member_tiled_long_env_updates = _one_kernel_synthetic_persistent_member_tiled_env(case)
            persistent_member_tiled_long_forward = lambda q, k, v: _run_sparse_attention(
                q, k, v, keep_ids, hash_ids, schedule, env_updates=persistent_member_tiled_long_env_updates
            )
            persistent_member_tiled_long_bwd_ms = _measure_backward_ms(
                persistent_member_tiled_long_forward,
                q_data,
                k_data,
                v_data,
                case.warmup_iters,
                case.benchmark_iters,
                env_updates=persistent_member_tiled_long_env_updates,
            )
            hsa_bwd_vs_persistent_member_tiled_long = (
                hsa_bwd_ms / persistent_member_tiled_long_bwd_ms
                if persistent_member_tiled_long_bwd_ms > 0
                else float("inf")
            )
        one_kernel_pingpong_env_updates = _one_kernel_synthetic_pingpong_env(case)
        one_kernel_pingpong_forward = lambda q, k, v: _run_sparse_attention(
            q, k, v, keep_ids, hash_ids, schedule, env_updates=one_kernel_pingpong_env_updates
        )
        one_kernel_pingpong_bwd_ms = _measure_backward_ms(
            one_kernel_pingpong_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=one_kernel_pingpong_env_updates,
        )
        hsa_bwd_vs_one_kernel_pingpong = (
            hsa_bwd_ms / one_kernel_pingpong_bwd_ms if one_kernel_pingpong_bwd_ms > 0 else float("inf")
        )
    if _should_measure_sparse_mask_mixed_backward_baseline(case):
        sparse_mask_env_updates = _sparse_mask_mixed_backward_baseline_env(case)
        sparse_mask_hsa_forward = lambda q, k, v: _run_sparse_attention(
            q, k, v, keep_ids, hash_ids, schedule, env_updates=sparse_mask_env_updates
        )
        hybrid_fwd_ms = _measure_forward_ms(
            sparse_mask_hsa_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=sparse_mask_env_updates,
        )
        hybrid_bwd_ms = _measure_backward_ms(
            sparse_mask_hsa_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=sparse_mask_env_updates,
        )
        hybrid_fwd_bwd_ms = _measure_forward_backward_ms(
            sparse_mask_hsa_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=sparse_mask_env_updates,
        )
        if dense_causal_fwd_bwd_ms is not None and dense_causal_fwd_bwd_ms > 0:
            hybrid_vs_dense = hybrid_fwd_bwd_ms / dense_causal_fwd_bwd_ms
        if sliding_fa4_fwd_bwd_ms is not None and hybrid_fwd_bwd_ms > 0:
            sliding_fa4_vs_hybrid = sliding_fa4_fwd_bwd_ms / hybrid_fwd_bwd_ms
        hybrid_status = "measured"
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
    if _should_measure_one_kernel_synthetic_baseline(case):
        long_family_cases = [
            (_one_kernel_synthetic_bucket_dense_label(), _one_kernel_synthetic_bucket_dense_env(case)),
            (_one_kernel_synthetic_long_label(), _one_kernel_synthetic_long_env(case)),
        ]
        if case.seqlen >= 65536:
            long_family_cases.append(
                (
                    _one_kernel_synthetic_bucket_dense_saved_packed_label(),
                    _one_kernel_synthetic_bucket_dense_saved_packed_env(case),
                )
            )
            long_family_cases.append(
                (
                    _one_kernel_synthetic_bucket_dense_saved_prob_label(),
                    _one_kernel_synthetic_bucket_dense_saved_prob_env(case),
                )
            )
            long_family_cases.append(
                (_one_kernel_synthetic_bucket_dense_two_pass_label(), _one_kernel_synthetic_bucket_dense_two_pass_env(case))
            )
        long_family_cases.append(
            (_one_kernel_synthetic_bucket_dense_dualrow_label(), _one_kernel_synthetic_bucket_dense_dualrow_env(case))
        )
        if _include_long_experimental_comparators():
            long_family_cases.extend(
                [
                    (_one_kernel_synthetic_bucket_dense_tc_label(), _one_kernel_synthetic_bucket_dense_tc_env(case)),
                    (_one_kernel_synthetic_long_8_label(), _one_kernel_synthetic_long_8_env(case)),
                    (_one_kernel_synthetic_long_16_label(), _one_kernel_synthetic_long_16_env(case)),
                ]
            )
        for label, env in long_family_cases:
            long_forward = lambda q, k, v, env=env: _run_sparse_attention(
                q, k, v, keep_ids, hash_ids, schedule, env_updates=env
            )
            fwd_ms = _measure_forward_ms(
                long_forward,
                q_data,
                k_data,
                v_data,
                case.warmup_iters,
                case.benchmark_iters,
                env_updates=env,
            )
            bwd_ms = _measure_backward_ms(
                long_forward,
                q_data,
                k_data,
                v_data,
                case.warmup_iters,
                case.benchmark_iters,
                env_updates=env,
            )
            fwd_bwd_ms = _measure_forward_backward_ms(
                long_forward,
                q_data,
                k_data,
                v_data,
                case.warmup_iters,
                case.benchmark_iters,
                env_updates=env,
            )
            vs_dense = None if dense_causal_fwd_bwd_ms is None or dense_causal_fwd_bwd_ms <= 0 else fwd_bwd_ms / dense_causal_fwd_bwd_ms
            vs_hybrid = None if hybrid_fwd_bwd_ms is None or hybrid_fwd_bwd_ms <= 0 else fwd_bwd_ms / hybrid_fwd_bwd_ms
            if label == _one_kernel_synthetic_bucket_dense_label():
                bucket_dense_fwd_ms = fwd_ms
                bucket_dense_bwd_ms = bwd_ms
                bucket_dense_fwd_bwd_ms = fwd_bwd_ms
                bucket_dense_vs_dense = vs_dense
                bucket_dense_vs_hybrid = vs_hybrid
                if sliding_fa4_fwd_bwd_ms is not None and bucket_dense_fwd_bwd_ms > 0:
                    sliding_fa4_vs_bucket_dense = sliding_fa4_fwd_bwd_ms / bucket_dense_fwd_bwd_ms
            elif label == _one_kernel_synthetic_bucket_dense_saved_prob_label():
                bucket_dense_saved_prob_fwd_ms = fwd_ms
                bucket_dense_saved_prob_bwd_ms = bwd_ms
                bucket_dense_saved_prob_fwd_bwd_ms = fwd_bwd_ms
                bucket_dense_saved_prob_vs_dense = vs_dense
                bucket_dense_saved_prob_vs_hybrid = vs_hybrid
            elif label == _one_kernel_synthetic_bucket_dense_two_pass_label():
                bucket_dense_two_pass_fwd_ms = fwd_ms
                bucket_dense_two_pass_bwd_ms = bwd_ms
                bucket_dense_two_pass_fwd_bwd_ms = fwd_bwd_ms
                bucket_dense_two_pass_vs_dense = vs_dense
                bucket_dense_two_pass_vs_hybrid = vs_hybrid
            elif label == _one_kernel_synthetic_bucket_dense_dualrow_label():
                bucket_dense_dualrow_fwd_ms = fwd_ms
                bucket_dense_dualrow_bwd_ms = bwd_ms
                bucket_dense_dualrow_fwd_bwd_ms = fwd_bwd_ms
                bucket_dense_dualrow_vs_dense = vs_dense
                bucket_dense_dualrow_vs_hybrid = vs_hybrid
            elif label == _one_kernel_synthetic_bucket_dense_tc_label():
                bucket_dense_tc_fwd_ms = fwd_ms
                bucket_dense_tc_bwd_ms = bwd_ms
                bucket_dense_tc_fwd_bwd_ms = fwd_bwd_ms
                bucket_dense_tc_vs_dense = vs_dense
                bucket_dense_tc_vs_hybrid = vs_hybrid
            elif label == _one_kernel_synthetic_long_label():
                one_kernel_long_4_fwd_ms = fwd_ms
                one_kernel_long_4_bwd_ms = bwd_ms
                one_kernel_long_4_fwd_bwd_ms = fwd_bwd_ms
                one_kernel_long_4_vs_dense = vs_dense
                one_kernel_long_4_vs_hybrid = vs_hybrid
            elif label == _one_kernel_synthetic_long_8_label():
                one_kernel_long_8_fwd_ms = fwd_ms
                one_kernel_long_8_bwd_ms = bwd_ms
                one_kernel_long_8_fwd_bwd_ms = fwd_bwd_ms
                one_kernel_long_8_vs_dense = vs_dense
                one_kernel_long_8_vs_hybrid = vs_hybrid
            else:
                one_kernel_long_16_fwd_ms = fwd_ms
                one_kernel_long_16_bwd_ms = bwd_ms
                one_kernel_long_16_fwd_bwd_ms = fwd_bwd_ms
                one_kernel_long_16_vs_dense = vs_dense
                one_kernel_long_16_vs_hybrid = vs_hybrid

    line = (
        f"{case.name}: mode={_benchmark_mode_label(case)} {_benchmark_sparse_bwd_config_label()} "
        f"shape=(B={case.batch_size}, T={case.seqlen}, H={case.nheads}, KV={n_kv_heads}, D={case.headdim}) "
        f"allowed_pairs={occupancy['allowed_pairs']} token_density={occupancy['token_density']:.6f} "
        f"active_blocks={occupancy['active_blocks']} total_blocks={occupancy['total_blocks']} "
        f"active_block_density={occupancy['active_block_density']:.6f} "
        f"active_fill_nominal={occupancy['active_fill_nominal']:.6f} "
        f"active_fill_valid={occupancy['active_fill_valid']:.6f} "
        f"hsa_fwd_ms={hsa_fwd_ms:.3f} "
        f"hsa_bwd_ms={hsa_bwd_ms:.3f} "
        f"hsa_fwd_bwd_ms={hsa_fwd_bwd_ms:.3f}"
    )
    if dense_causal_fwd_ms is not None and dense_causal_bwd_ms is not None and dense_causal_fwd_bwd_ms is not None:
        line += (
            f" dense_causal_fwd_ms={dense_causal_fwd_ms:.3f}"
            f" dense_causal_bwd_ms={dense_causal_bwd_ms:.3f}"
            f" dense_causal_fwd_bwd_ms={dense_causal_fwd_bwd_ms:.3f}"
            f" hsa_fwd_vs_dense={hsa_fwd_vs_dense:.2f}x"
            f" hsa_bwd_vs_dense={hsa_bwd_vs_dense:.2f}x"
            f" hsa_fwd_bwd_vs_dense={hsa_fwd_bwd_vs_dense:.2f}x"
        )
    else:
        line += (
            f" dense_causal_fwd_status={dense_causal_fwd_status}"
            f" dense_causal_bwd_status={dense_causal_bwd_status}"
            f" dense_causal_fwd_bwd_status={dense_causal_fwd_bwd_status}"
        )
    primary_parts: list[str] = []
    _append_labeled_triplet_fields(
        primary_parts,
        prefix="mask_mod_fa4",
        label="plain_fa4_mask_mod_hsa",
        fwd_ms=mask_mod_result["fwd_ms"],
        bwd_ms=mask_mod_result["bwd_ms"],
        fwd_bwd_ms=mask_mod_result["fwd_bwd_ms"],
        status=mask_mod_result["status"],
    )
    _append_labeled_triplet_fields(
        primary_parts,
        prefix="hybrid",
        label=_sparse_mask_mixed_backward_baseline_label(),
        fwd_ms=hybrid_fwd_ms,
        bwd_ms=hybrid_bwd_ms,
        fwd_bwd_ms=hybrid_fwd_bwd_ms,
        status=hybrid_status,
    )
    _append_labeled_triplet_fields(
        primary_parts,
        prefix="sliding_log_fa4",
        label="plain_fa4_sliding_logS_causal",
        fwd_ms=sliding_log_result["fwd_ms"],
        bwd_ms=sliding_log_result["bwd_ms"],
        fwd_bwd_ms=sliding_log_result["fwd_bwd_ms"],
        status=sliding_log_result["status"],
        extra_fields=[
            f"sliding_log_fa4_window_tokens={sliding_log_tokens}",
            f"sliding_log_fa4_window_left={sliding_log_window[0]}",
            f"sliding_log_fa4_pairs={sliding_log_pairs}",
        ],
    )
    _append_labeled_triplet_fields(
        primary_parts,
        prefix="sliding_flopmatched_fa4",
        label="plain_fa4_sliding_flopmatched_causal",
        fwd_ms=sliding_flopmatched_result["fwd_ms"],
        bwd_ms=sliding_flopmatched_result["bwd_ms"],
        fwd_bwd_ms=sliding_flopmatched_result["fwd_bwd_ms"],
        status=sliding_flopmatched_result["status"],
        extra_fields=[
            f"sliding_flopmatched_fa4_window_tokens={sliding_flopmatched_tokens}",
            f"sliding_flopmatched_fa4_window_left={sliding_flopmatched_window[0]}",
            f"sliding_flopmatched_fa4_pairs={sliding_flopmatched_pairs}",
        ],
    )
    _append_dense_triplet_fields(
        primary_parts,
        fwd_ms=dense_causal_fwd_ms,
        bwd_ms=dense_causal_bwd_ms,
        fwd_bwd_ms=dense_causal_fwd_bwd_ms,
        status=dense_causal_fwd_bwd_status,
    )
    line += " " + " ".join(primary_parts)
    if sliding_fa4_fwd_bwd_ms is not None and sliding_fa4_fwd_ms is not None and sliding_fa4_bwd_ms is not None:
        line += (
            f" sliding_fa4_label=plain_fa4_sliding_flopmatched_causal"
            f" sliding_fa4_window_tokens={sliding_flopmatched_tokens}"
            f" sliding_fa4_window_left={sliding_flopmatched_window[0]}"
            f" sliding_fa4_pairs={sliding_flopmatched_pairs}"
            f" sliding_fa4_fwd_ms={sliding_fa4_fwd_ms:.3f}"
            f" sliding_fa4_bwd_ms={sliding_fa4_bwd_ms:.3f}"
            f" sliding_fa4_fwd_bwd_ms={sliding_fa4_fwd_bwd_ms:.3f}"
        )
    else:
        line += (
            f" sliding_fa4_label=plain_fa4_sliding_flopmatched_causal"
            f" sliding_fa4_window_tokens={sliding_flopmatched_tokens}"
            f" sliding_fa4_window_left={sliding_flopmatched_window[0]}"
            f" sliding_fa4_pairs={sliding_flopmatched_pairs}"
            f" sliding_fa4_status={sliding_fa4_status}"
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
    if one_kernel_synth_bwd_ms is not None:
        line += (
            f" one_kernel_bwd_label={_one_kernel_synthetic_baseline_label()}"
            f" one_kernel_bwd_ms={one_kernel_synth_bwd_ms:.3f}"
            f" hsa_bwd_vs_one_kernel={hsa_bwd_vs_one_kernel:.2f}x"
        )
    if one_kernel_short_bwd_ms is not None:
        line += (
            f" one_kernel_short_label={_one_kernel_synthetic_short_label()}"
            f" one_kernel_short_bwd_ms={one_kernel_short_bwd_ms:.3f}"
            f" hsa_bwd_vs_one_kernel_short={hsa_bwd_vs_one_kernel_short:.2f}x"
        )
    if one_kernel_long_bwd_ms is not None:
        line += (
            f" one_kernel_long_label={_one_kernel_synthetic_long_label()}"
            f" one_kernel_long_bwd_ms={one_kernel_long_bwd_ms:.3f}"
            f" hsa_bwd_vs_one_kernel_long={hsa_bwd_vs_one_kernel_long:.2f}x"
        )
    if two_stage_long_bwd_ms is not None:
        line += (
            f" two_stage_long_label={_one_kernel_synthetic_two_stage_label()}"
            f" two_stage_long_bwd_ms={two_stage_long_bwd_ms:.3f}"
            f" hsa_bwd_vs_two_stage_long={hsa_bwd_vs_two_stage_long:.2f}x"
        )
    if persistent_long_bwd_ms is not None:
        line += (
            f" persistent_long_label={_one_kernel_synthetic_persistent_label()}"
            f" persistent_long_bwd_ms={persistent_long_bwd_ms:.3f}"
            f" hsa_bwd_vs_persistent_long={hsa_bwd_vs_persistent_long:.2f}x"
        )
    if persistent_member_tiled_long_bwd_ms is not None:
        line += (
            f" persistent_member_tiled_long_label={_one_kernel_synthetic_persistent_member_tiled_label()}"
            f" persistent_member_tiled_long_bwd_ms={persistent_member_tiled_long_bwd_ms:.3f}"
            f" hsa_bwd_vs_persistent_member_tiled_long={hsa_bwd_vs_persistent_member_tiled_long:.2f}x"
        )
    if one_kernel_pingpong_bwd_ms is not None:
        line += (
            f" one_kernel_pingpong_label={_one_kernel_synthetic_pingpong_label()}"
            f" one_kernel_pingpong_bwd_ms={one_kernel_pingpong_bwd_ms:.3f}"
            f" hsa_bwd_vs_one_kernel_pingpong={hsa_bwd_vs_one_kernel_pingpong:.2f}x"
        )
    if sparse_mask_mixed_bwd_ms is not None:
        line += (
            f" sparse_mask_mixed_bwd_label={_sparse_mask_mixed_backward_baseline_label()}"
            f" sparse_mask_mixed_bwd_ms={sparse_mask_mixed_bwd_ms:.3f}"
            f" hsa_bwd_vs_sparse_mask_mixed={hsa_bwd_vs_sparse_mask_mixed:.2f}x"
        )
    if hybrid_fwd_bwd_ms is not None and hybrid_fwd_ms is not None and hybrid_bwd_ms is not None:
        if hybrid_vs_dense is not None:
            line += f" hybrid_vs_dense={hybrid_vs_dense:.2f}x"
        if sliding_fa4_vs_hybrid is not None:
            line += f" sliding_fa4_vs_hybrid={sliding_fa4_vs_hybrid:.2f}x"
    if bucket_dense_fwd_bwd_ms is not None and bucket_dense_fwd_ms is not None and bucket_dense_bwd_ms is not None:
        line += (
            f" bucket_dense_label={_one_kernel_synthetic_bucket_dense_label()}"
            f" bucket_dense_fwd_ms={bucket_dense_fwd_ms:.3f}"
            f" bucket_dense_bwd_ms={bucket_dense_bwd_ms:.3f}"
            f" bucket_dense_fwd_bwd_ms={bucket_dense_fwd_bwd_ms:.3f}"
        )
        if bucket_dense_vs_dense is not None:
            line += f" bucket_dense_vs_dense={bucket_dense_vs_dense:.2f}x"
        if bucket_dense_vs_hybrid is not None:
            line += f" bucket_dense_vs_hybrid={bucket_dense_vs_hybrid:.2f}x"
        if sliding_fa4_vs_bucket_dense is not None:
            line += f" sliding_fa4_vs_bucket_dense={sliding_fa4_vs_bucket_dense:.2f}x"
    if (
        bucket_dense_saved_prob_fwd_bwd_ms is not None
        and bucket_dense_saved_prob_fwd_ms is not None
        and bucket_dense_saved_prob_bwd_ms is not None
    ):
        line += (
            f" bucket_dense_saved_prob_label={_one_kernel_synthetic_bucket_dense_saved_prob_label()}"
            f" bucket_dense_saved_prob_fwd_ms={bucket_dense_saved_prob_fwd_ms:.3f}"
            f" bucket_dense_saved_prob_bwd_ms={bucket_dense_saved_prob_bwd_ms:.3f}"
            f" bucket_dense_saved_prob_fwd_bwd_ms={bucket_dense_saved_prob_fwd_bwd_ms:.3f}"
        )
        if bucket_dense_saved_prob_vs_dense is not None:
            line += f" bucket_dense_saved_prob_vs_dense={bucket_dense_saved_prob_vs_dense:.2f}x"
        if bucket_dense_saved_prob_vs_hybrid is not None:
            line += f" bucket_dense_saved_prob_vs_hybrid={bucket_dense_saved_prob_vs_hybrid:.2f}x"
    if (
        bucket_dense_two_pass_fwd_bwd_ms is not None
        and bucket_dense_two_pass_fwd_ms is not None
        and bucket_dense_two_pass_bwd_ms is not None
    ):
        line += (
            f" bucket_dense_two_pass_label={_one_kernel_synthetic_bucket_dense_two_pass_label()}"
            f" bucket_dense_two_pass_fwd_ms={bucket_dense_two_pass_fwd_ms:.3f}"
            f" bucket_dense_two_pass_bwd_ms={bucket_dense_two_pass_bwd_ms:.3f}"
            f" bucket_dense_two_pass_fwd_bwd_ms={bucket_dense_two_pass_fwd_bwd_ms:.3f}"
        )
        if bucket_dense_two_pass_vs_dense is not None:
            line += f" bucket_dense_two_pass_vs_dense={bucket_dense_two_pass_vs_dense:.2f}x"
        if bucket_dense_two_pass_vs_hybrid is not None:
            line += f" bucket_dense_two_pass_vs_hybrid={bucket_dense_two_pass_vs_hybrid:.2f}x"
    if (
        bucket_dense_dualrow_fwd_bwd_ms is not None
        and bucket_dense_dualrow_fwd_ms is not None
        and bucket_dense_dualrow_bwd_ms is not None
    ):
        line += (
            f" bucket_dense_dualrow_label={_one_kernel_synthetic_bucket_dense_dualrow_label()}"
            f" bucket_dense_dualrow_fwd_ms={bucket_dense_dualrow_fwd_ms:.3f}"
            f" bucket_dense_dualrow_bwd_ms={bucket_dense_dualrow_bwd_ms:.3f}"
            f" bucket_dense_dualrow_fwd_bwd_ms={bucket_dense_dualrow_fwd_bwd_ms:.3f}"
        )
        if bucket_dense_dualrow_vs_dense is not None:
            line += f" bucket_dense_dualrow_vs_dense={bucket_dense_dualrow_vs_dense:.2f}x"
        if bucket_dense_dualrow_vs_hybrid is not None:
            line += f" bucket_dense_dualrow_vs_hybrid={bucket_dense_dualrow_vs_hybrid:.2f}x"
    if (
        bucket_dense_tc_fwd_bwd_ms is not None
        and bucket_dense_tc_fwd_ms is not None
        and bucket_dense_tc_bwd_ms is not None
    ):
        line += (
            f" bucket_dense_tc_label={_one_kernel_synthetic_bucket_dense_tc_label()}"
            f" bucket_dense_tc_fwd_ms={bucket_dense_tc_fwd_ms:.3f}"
            f" bucket_dense_tc_bwd_ms={bucket_dense_tc_bwd_ms:.3f}"
            f" bucket_dense_tc_fwd_bwd_ms={bucket_dense_tc_fwd_bwd_ms:.3f}"
        )
        if bucket_dense_tc_vs_dense is not None:
            line += f" bucket_dense_tc_vs_dense={bucket_dense_tc_vs_dense:.2f}x"
        if bucket_dense_tc_vs_hybrid is not None:
            line += f" bucket_dense_tc_vs_hybrid={bucket_dense_tc_vs_hybrid:.2f}x"
    if one_kernel_long_4_fwd_bwd_ms is not None and one_kernel_long_4_fwd_ms is not None and one_kernel_long_4_bwd_ms is not None:
        line += (
            f" one_kernel_long_4_label={_one_kernel_synthetic_long_label()}"
            f" one_kernel_long_4_fwd_ms={one_kernel_long_4_fwd_ms:.3f}"
            f" one_kernel_long_4_bwd_ms={one_kernel_long_4_bwd_ms:.3f}"
            f" one_kernel_long_4_fwd_bwd_ms={one_kernel_long_4_fwd_bwd_ms:.3f}"
        )
        if one_kernel_long_4_vs_dense is not None:
            line += f" one_kernel_long_4_vs_dense={one_kernel_long_4_vs_dense:.2f}x"
        if one_kernel_long_4_vs_hybrid is not None:
            line += f" one_kernel_long_4_vs_hybrid={one_kernel_long_4_vs_hybrid:.2f}x"
    if one_kernel_long_8_fwd_bwd_ms is not None and one_kernel_long_8_fwd_ms is not None and one_kernel_long_8_bwd_ms is not None:
        line += (
            f" one_kernel_long_8_label={_one_kernel_synthetic_long_8_label()}"
            f" one_kernel_long_8_fwd_ms={one_kernel_long_8_fwd_ms:.3f}"
            f" one_kernel_long_8_bwd_ms={one_kernel_long_8_bwd_ms:.3f}"
            f" one_kernel_long_8_fwd_bwd_ms={one_kernel_long_8_fwd_bwd_ms:.3f}"
        )
        if one_kernel_long_8_vs_dense is not None:
            line += f" one_kernel_long_8_vs_dense={one_kernel_long_8_vs_dense:.2f}x"
        if one_kernel_long_8_vs_hybrid is not None:
            line += f" one_kernel_long_8_vs_hybrid={one_kernel_long_8_vs_hybrid:.2f}x"
    if one_kernel_long_16_fwd_bwd_ms is not None and one_kernel_long_16_fwd_ms is not None and one_kernel_long_16_bwd_ms is not None:
        line += (
            f" one_kernel_long_16_label={_one_kernel_synthetic_long_16_label()}"
            f" one_kernel_long_16_fwd_ms={one_kernel_long_16_fwd_ms:.3f}"
            f" one_kernel_long_16_bwd_ms={one_kernel_long_16_bwd_ms:.3f}"
            f" one_kernel_long_16_fwd_bwd_ms={one_kernel_long_16_fwd_bwd_ms:.3f}"
        )
        if one_kernel_long_16_vs_dense is not None:
            line += f" one_kernel_long_16_vs_dense={one_kernel_long_16_vs_dense:.2f}x"
        if one_kernel_long_16_vs_hybrid is not None:
            line += f" one_kernel_long_16_vs_hybrid={one_kernel_long_16_vs_hybrid:.2f}x"
    line += (
        f" long_tile_keys={tile_reuse['tile_keys']}"
        f" long_tile_count={tile_reuse['tiles']}"
        f" long_tile_occ_avg={tile_reuse['avg_occurrences']:.2f}"
        f" long_tile_occ_max={tile_reuse['max_occurrences']}"
        f" long_tile_unique_members_avg={tile_reuse['avg_unique_members']:.2f}"
        f" long_tile_unique_members_max={tile_reuse['max_unique_members']}"
        f" synthetic_micro_fwd={1 if os.environ.get('FLASH_ATTN_HSA_SYNTHETIC_MICRO_FWD', '0') == '1' else 0}"
        f" synthetic_micro_bwd={1 if os.environ.get('FLASH_ATTN_HSA_SYNTHETIC_MICRO_BWD', '0') == '1' else 0}"
        f" synthetic_one_kernel_bwd={os.environ.get('FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD', 'off')}"
        f" synthetic_one_kernel_bwd_variant={os.environ.get('FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT', 'auto')}"
        f" synthetic_one_kernel_long_keys_per_cta={os.environ.get('FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_LONG_KEYS_PER_CTA', '4')}"
        f" synthetic_long_bwd_mode={os.environ.get('FLASH_ATTN_HSA_SYNTHETIC_LONG_BWD_MODE', 'one_kernel')}"
        f" synthetic_one_kernel_bwd_pingpong={os.environ.get('FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_PINGPONG', 'off')}"
        f" synthetic_bwd_kernel_mode={_synthetic_bwd_kernel_mode_label()}"
    )
    print(line)


def run_case(case: BenchmarkCase):
    (
        build_hsa_schedule,
        flash_attn_func,
        flash_attn_hsa_func,
        _,
        hsa_reference_attention,
        schedule_to_attend_mask,
    ) = _lazy_flash_attn_imports()
    device = "cuda"
    dtype = torch.bfloat16
    keep_ids, hash_ids = _make_hsa_metadata(case.batch_size, case.seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    allowed_pairs = int(schedule_to_attend_mask(schedule).sum().item())
    n_kv_heads = case.n_kv_heads if case.n_kv_heads is not None else case.nheads
    q_data = torch.randn(case.batch_size, case.seqlen, case.nheads, case.headdim, device=device, dtype=dtype)
    k_data = torch.randn(case.batch_size, case.seqlen, n_kv_heads, case.headdim, device=device, dtype=dtype)
    v_data = torch.randn(case.batch_size, case.seqlen, n_kv_heads, case.headdim, device=device, dtype=dtype)
    env_updates = _benchmark_env_for_case(case)

    hsa_forward = lambda q, k, v: _run_sparse_attention(q, k, v, keep_ids, hash_ids, schedule, env_updates=env_updates)
    mask_mod_forward = lambda q, k, v: _unwrap_output(
        flash_attn_hsa_func(q, k, v, keep_ids=keep_ids, hash_ids=hash_ids)
    )
    dense_causal_forward = lambda q, k, v: _unwrap_output(flash_attn_func(q, k, v, causal=True))
    sliding_log_tokens = _log_sliding_window_tokens(case.seqlen)
    sliding_log_window = (max(0, sliding_log_tokens - 1), 0)
    sliding_log_pairs = _causal_sliding_window_pairs(case.seqlen, sliding_log_tokens)
    sliding_flopmatched_tokens, sliding_flopmatched_window = _flop_matched_sliding_window(case.seqlen, allowed_pairs)
    sliding_flopmatched_pairs = _causal_sliding_window_pairs(case.seqlen, sliding_flopmatched_tokens)
    sliding_log_forward = lambda q, k, v: _unwrap_output(
        flash_attn_func(q, k, v, causal=True, window_size=sliding_log_window)
    )
    sliding_flopmatched_forward = lambda q, k, v: _unwrap_output(
        flash_attn_func(q, k, v, causal=True, window_size=sliding_flopmatched_window)
    )
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
    hsa_grad_status = None
    if case.name == "train-eq" and os.environ.get("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "0") == "1":
        hsa_grad_status = _measure_grad_finite_status(
            hsa_forward,
            q_data,
            k_data,
            v_data,
            env_updates=env_updates,
        )

    hsa_bwd_vs_dense = hsa_bwd_ms / dense_causal_bwd_ms if dense_causal_bwd_ms > 0 else float("inf")
    hsa_fwd_vs_dense = hsa_fwd_ms / dense_causal_fwd_ms if dense_causal_fwd_ms > 0 else float("inf")
    hsa_fwd_bwd_vs_dense = hsa_fwd_bwd_ms / dense_causal_fwd_bwd_ms if dense_causal_fwd_bwd_ms > 0 else float("inf")
    mask_mod_result = _measure_triplet_or_status(
        mask_mod_forward,
        q_data,
        k_data,
        v_data,
        case.warmup_iters,
        case.benchmark_iters,
    )
    sliding_log_result = _measure_triplet_or_status(
        sliding_log_forward,
        q_data,
        k_data,
        v_data,
        case.warmup_iters,
        case.benchmark_iters,
    )
    sliding_flopmatched_result = _measure_triplet_or_status(
        sliding_flopmatched_forward,
        q_data,
        k_data,
        v_data,
        case.warmup_iters,
        case.benchmark_iters,
    )
    dense_fa4_status = "measured"
    hybrid_status = "unavailable_unsupported_case"
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
    one_kernel_synth_fwd_ms = None
    one_kernel_synth_bwd_ms = None
    one_kernel_synth_fwd_bwd_ms = None
    hsa_fwd_bwd_vs_one_kernel = None
    one_kernel_short_fwd_ms = None
    one_kernel_short_bwd_ms = None
    one_kernel_short_fwd_bwd_ms = None
    hsa_fwd_bwd_vs_one_kernel_short = None
    one_kernel_long_fwd_ms = None
    one_kernel_long_bwd_ms = None
    one_kernel_long_fwd_bwd_ms = None
    hsa_fwd_bwd_vs_one_kernel_long = None
    one_kernel_pingpong_fwd_ms = None
    one_kernel_pingpong_bwd_ms = None
    one_kernel_pingpong_fwd_bwd_ms = None
    hsa_fwd_bwd_vs_one_kernel_pingpong = None
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
    if _should_measure_one_kernel_synthetic_baseline(case):
        one_kernel_env_updates = _one_kernel_synthetic_baseline_env(case)
        one_kernel_hsa_forward = lambda q, k, v: _run_sparse_attention(
            q, k, v, keep_ids, hash_ids, schedule, env_updates=one_kernel_env_updates
        )
        one_kernel_synth_fwd_ms = _measure_forward_ms(
            one_kernel_hsa_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=one_kernel_env_updates,
        )
        one_kernel_synth_bwd_ms = _measure_backward_ms(
            one_kernel_hsa_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=one_kernel_env_updates,
        )
        one_kernel_synth_fwd_bwd_ms = _measure_forward_backward_ms(
            one_kernel_hsa_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=one_kernel_env_updates,
        )
        hsa_fwd_bwd_vs_one_kernel = (
            hsa_fwd_bwd_ms / one_kernel_synth_fwd_bwd_ms if one_kernel_synth_fwd_bwd_ms > 0 else float("inf")
        )
        one_kernel_short_env_updates = _one_kernel_synthetic_short_env(case)
        one_kernel_short_forward = lambda q, k, v: _run_sparse_attention(
            q, k, v, keep_ids, hash_ids, schedule, env_updates=one_kernel_short_env_updates
        )
        one_kernel_short_fwd_ms = _measure_forward_ms(
            one_kernel_short_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=one_kernel_short_env_updates,
        )
        one_kernel_short_bwd_ms = _measure_backward_ms(
            one_kernel_short_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=one_kernel_short_env_updates,
        )
        one_kernel_short_fwd_bwd_ms = _measure_forward_backward_ms(
            one_kernel_short_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=one_kernel_short_env_updates,
        )
        hsa_fwd_bwd_vs_one_kernel_short = (
            hsa_fwd_bwd_ms / one_kernel_short_fwd_bwd_ms if one_kernel_short_fwd_bwd_ms > 0 else float("inf")
        )
        one_kernel_long_env_updates = _one_kernel_synthetic_long_env(case)
        one_kernel_long_forward = lambda q, k, v: _run_sparse_attention(
            q, k, v, keep_ids, hash_ids, schedule, env_updates=one_kernel_long_env_updates
        )
        one_kernel_long_fwd_ms = _measure_forward_ms(
            one_kernel_long_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=one_kernel_long_env_updates,
        )
        one_kernel_long_bwd_ms = _measure_backward_ms(
            one_kernel_long_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=one_kernel_long_env_updates,
        )
        one_kernel_long_fwd_bwd_ms = _measure_forward_backward_ms(
            one_kernel_long_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=one_kernel_long_env_updates,
        )
        hsa_fwd_bwd_vs_one_kernel_long = (
            hsa_fwd_bwd_ms / one_kernel_long_fwd_bwd_ms if one_kernel_long_fwd_bwd_ms > 0 else float("inf")
        )
        one_kernel_pingpong_env_updates = _one_kernel_synthetic_pingpong_env(case)
        one_kernel_pingpong_forward = lambda q, k, v: _run_sparse_attention(
            q, k, v, keep_ids, hash_ids, schedule, env_updates=one_kernel_pingpong_env_updates
        )
        one_kernel_pingpong_fwd_ms = _measure_forward_ms(
            one_kernel_pingpong_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=one_kernel_pingpong_env_updates,
        )
        one_kernel_pingpong_bwd_ms = _measure_backward_ms(
            one_kernel_pingpong_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=one_kernel_pingpong_env_updates,
        )
        one_kernel_pingpong_fwd_bwd_ms = _measure_forward_backward_ms(
            one_kernel_pingpong_forward,
            q_data,
            k_data,
            v_data,
            case.warmup_iters,
            case.benchmark_iters,
            env_updates=one_kernel_pingpong_env_updates,
        )
        hsa_fwd_bwd_vs_one_kernel_pingpong = (
            hsa_fwd_bwd_ms / one_kernel_pingpong_fwd_bwd_ms
            if one_kernel_pingpong_fwd_bwd_ms > 0
            else float("inf")
        )
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
        hybrid_status = "measured"
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
    if hsa_grad_status is not None:
        line += (
            f" hsa_grads_finite={1 if hsa_grad_status['finite'] else 0}"
            f" hsa_q_nonfinite={hsa_grad_status['q_nonfinite']}"
            f" hsa_k_nonfinite={hsa_grad_status['k_nonfinite']}"
            f" hsa_v_nonfinite={hsa_grad_status['v_nonfinite']}"
        )
    primary_parts: list[str] = []
    _append_labeled_triplet_fields(
        primary_parts,
        prefix="mask_mod_fa4",
        label="plain_fa4_mask_mod_hsa",
        fwd_ms=mask_mod_result["fwd_ms"],
        bwd_ms=mask_mod_result["bwd_ms"],
        fwd_bwd_ms=mask_mod_result["fwd_bwd_ms"],
        status=mask_mod_result["status"],
    )
    _append_labeled_triplet_fields(
        primary_parts,
        prefix="hybrid",
        label=_sparse_mask_mixed_backward_baseline_label(),
        fwd_ms=sparse_mask_mixed_fwd_ms,
        bwd_ms=sparse_mask_mixed_bwd_ms,
        fwd_bwd_ms=sparse_mask_mixed_fwd_bwd_ms,
        status=hybrid_status,
    )
    _append_labeled_triplet_fields(
        primary_parts,
        prefix="sliding_log_fa4",
        label="plain_fa4_sliding_logS_causal",
        fwd_ms=sliding_log_result["fwd_ms"],
        bwd_ms=sliding_log_result["bwd_ms"],
        fwd_bwd_ms=sliding_log_result["fwd_bwd_ms"],
        status=sliding_log_result["status"],
        extra_fields=[
            f"sliding_log_fa4_window_tokens={sliding_log_tokens}",
            f"sliding_log_fa4_window_left={sliding_log_window[0]}",
            f"sliding_log_fa4_pairs={sliding_log_pairs}",
        ],
    )
    _append_labeled_triplet_fields(
        primary_parts,
        prefix="sliding_flopmatched_fa4",
        label="plain_fa4_sliding_flopmatched_causal",
        fwd_ms=sliding_flopmatched_result["fwd_ms"],
        bwd_ms=sliding_flopmatched_result["bwd_ms"],
        fwd_bwd_ms=sliding_flopmatched_result["fwd_bwd_ms"],
        status=sliding_flopmatched_result["status"],
        extra_fields=[
            f"sliding_flopmatched_fa4_window_tokens={sliding_flopmatched_tokens}",
            f"sliding_flopmatched_fa4_window_left={sliding_flopmatched_window[0]}",
            f"sliding_flopmatched_fa4_pairs={sliding_flopmatched_pairs}",
        ],
    )
    _append_dense_triplet_fields(
        primary_parts,
        fwd_ms=dense_causal_fwd_ms,
        bwd_ms=dense_causal_bwd_ms,
        fwd_bwd_ms=dense_causal_fwd_bwd_ms,
        status=dense_fa4_status,
    )
    line += " " + " ".join(primary_parts)
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
        one_kernel_synth_fwd_bwd_ms is not None
        and one_kernel_synth_bwd_ms is not None
        and one_kernel_synth_fwd_ms is not None
    ):
        line += (
            f" one_kernel_bwd_label={_one_kernel_synthetic_baseline_label()}"
            f" one_kernel_bwd_fwd_ms={one_kernel_synth_fwd_ms:.3f}"
            f" one_kernel_bwd_bwd_ms={one_kernel_synth_bwd_ms:.3f}"
            f" one_kernel_bwd_fwd_bwd_ms={one_kernel_synth_fwd_bwd_ms:.3f}"
            f" hsa_fwd_bwd_vs_one_kernel={hsa_fwd_bwd_vs_one_kernel:.2f}x"
        )
    if (
        one_kernel_short_fwd_bwd_ms is not None
        and one_kernel_short_bwd_ms is not None
        and one_kernel_short_fwd_ms is not None
    ):
        line += (
            f" one_kernel_short_label={_one_kernel_synthetic_short_label()}"
            f" one_kernel_short_fwd_ms={one_kernel_short_fwd_ms:.3f}"
            f" one_kernel_short_bwd_ms={one_kernel_short_bwd_ms:.3f}"
            f" one_kernel_short_fwd_bwd_ms={one_kernel_short_fwd_bwd_ms:.3f}"
            f" hsa_fwd_bwd_vs_one_kernel_short={hsa_fwd_bwd_vs_one_kernel_short:.2f}x"
        )
    if (
        one_kernel_long_fwd_bwd_ms is not None
        and one_kernel_long_bwd_ms is not None
        and one_kernel_long_fwd_ms is not None
    ):
        line += (
            f" one_kernel_long_label={_one_kernel_synthetic_long_label()}"
            f" one_kernel_long_fwd_ms={one_kernel_long_fwd_ms:.3f}"
            f" one_kernel_long_bwd_ms={one_kernel_long_bwd_ms:.3f}"
            f" one_kernel_long_fwd_bwd_ms={one_kernel_long_fwd_bwd_ms:.3f}"
            f" hsa_fwd_bwd_vs_one_kernel_long={hsa_fwd_bwd_vs_one_kernel_long:.2f}x"
        )
    if (
        one_kernel_pingpong_fwd_bwd_ms is not None
        and one_kernel_pingpong_bwd_ms is not None
        and one_kernel_pingpong_fwd_ms is not None
    ):
        line += (
            f" one_kernel_pingpong_label={_one_kernel_synthetic_pingpong_label()}"
            f" one_kernel_pingpong_fwd_ms={one_kernel_pingpong_fwd_ms:.3f}"
            f" one_kernel_pingpong_bwd_ms={one_kernel_pingpong_bwd_ms:.3f}"
            f" one_kernel_pingpong_fwd_bwd_ms={one_kernel_pingpong_fwd_bwd_ms:.3f}"
            f" hsa_fwd_bwd_vs_one_kernel_pingpong={hsa_fwd_bwd_vs_one_kernel_pingpong:.2f}x"
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
            f" synthetic_one_kernel_bwd_variant={os.environ.get('FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_VARIANT', 'auto')}"
            f" synthetic_one_kernel_long_keys_per_cta={os.environ.get('FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_LONG_KEYS_PER_CTA', '4')}"
            f" synthetic_long_bwd_mode={os.environ.get('FLASH_ATTN_HSA_SYNTHETIC_LONG_BWD_MODE', 'one_kernel')}"
            f" synthetic_one_kernel_bwd_pingpong={os.environ.get('FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD_PINGPONG', 'off')}"
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
    if _use_bucket_dense_long_profile_mode():
        for case in _get_selected_cases(
            env_var="FLASH_ATTN_HSA_PROFILE_CASES",
            default=(LONG_CONTEXT_CASES[4], LONG_CONTEXT_CASES[6], LONG_CONTEXT_CASES[7]),
        ):
            _profile_bucket_dense_long_case(case, _get_bucket_dense_long_profile_target_mode())
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
