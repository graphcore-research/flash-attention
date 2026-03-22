import time
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from contextlib import contextmanager

import torch

_FLASH_ATTN_IMPORTS = None


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

    split_forward = lambda q, k, v: _run_sparse_attention(
        q, k, v, keep_ids, hash_ids, schedule, env_updates=_split_monolithic_env()
    )
    true_fused_forward = lambda q, k, v: _run_sparse_attention(
        q, k, v, keep_ids, hash_ids, schedule, env_updates=_true_fused_env()
    )
    dense_causal_forward = lambda q, k, v: _unwrap_output(flash_attn_func(q, k, v, causal=True))
    dense_ref_forward = lambda q, k, v: hsa_reference_attention(q, k, v, keep_ids, hash_ids)

    out_split = split_forward(q_data, k_data, v_data)
    out_true_fused = true_fused_forward(q_data, k_data, v_data)
    out_ref = dense_ref_forward(q_data, k_data, v_data)
    split_max_diff = (out_split.float() - out_ref.float()).abs().max().item()
    split_mean_diff = (out_split.float() - out_ref.float()).abs().mean().item()
    true_fused_max_diff = (out_true_fused.float() - out_ref.float()).abs().max().item()
    true_fused_mean_diff = (out_true_fused.float() - out_ref.float()).abs().mean().item()

    split_fwd_ms = _measure_forward_ms(
        split_forward,
        q_data,
        k_data,
        v_data,
        case.warmup_iters,
        case.benchmark_iters,
        env_updates=_split_monolithic_env(),
    )
    true_fused_fwd_ms = _measure_forward_ms(
        true_fused_forward,
        q_data,
        k_data,
        v_data,
        case.warmup_iters,
        case.benchmark_iters,
        env_updates=_true_fused_env(),
    )
    dense_causal_fwd_ms = _measure_forward_ms(
        dense_causal_forward,
        q_data,
        k_data,
        v_data,
        case.warmup_iters,
        case.benchmark_iters,
    )
    split_bwd_ms = _measure_backward_ms(
        split_forward,
        q_data,
        k_data,
        v_data,
        case.warmup_iters,
        case.benchmark_iters,
        env_updates=_split_monolithic_env(),
    )
    true_fused_bwd_ms = _measure_backward_ms(
        true_fused_forward,
        q_data,
        k_data,
        v_data,
        case.warmup_iters,
        case.benchmark_iters,
        env_updates=_true_fused_env(),
    )
    dense_causal_bwd_ms = _measure_backward_ms(
        dense_causal_forward,
        q_data,
        k_data,
        v_data,
        case.warmup_iters,
        case.benchmark_iters,
    )
    split_fwd_bwd_ms = _measure_forward_backward_ms(
        split_forward,
        q_data,
        k_data,
        v_data,
        case.warmup_iters,
        case.benchmark_iters,
        env_updates=_split_monolithic_env(),
    )
    true_fused_fwd_bwd_ms = _measure_forward_backward_ms(
        true_fused_forward,
        q_data,
        k_data,
        v_data,
        case.warmup_iters,
        case.benchmark_iters,
        env_updates=_true_fused_env(),
    )
    dense_causal_fwd_bwd_ms = _measure_forward_backward_ms(
        dense_causal_forward,
        q_data,
        k_data,
        v_data,
        case.warmup_iters,
        case.benchmark_iters,
    )

    split_vs_dense = split_bwd_ms / dense_causal_bwd_ms if dense_causal_bwd_ms > 0 else float("inf")
    true_fused_vs_dense = true_fused_bwd_ms / dense_causal_bwd_ms if dense_causal_bwd_ms > 0 else float("inf")
    true_fused_vs_split = true_fused_bwd_ms / split_bwd_ms if split_bwd_ms > 0 else float("inf")
    split_fwd_vs_dense = split_fwd_ms / dense_causal_fwd_ms if dense_causal_fwd_ms > 0 else float("inf")
    true_fused_fwd_vs_dense = true_fused_fwd_ms / dense_causal_fwd_ms if dense_causal_fwd_ms > 0 else float("inf")
    true_fused_fwd_vs_split = true_fused_fwd_ms / split_fwd_ms if split_fwd_ms > 0 else float("inf")
    split_fwd_bwd_vs_dense = split_fwd_bwd_ms / dense_causal_fwd_bwd_ms if dense_causal_fwd_bwd_ms > 0 else float("inf")
    true_fused_fwd_bwd_vs_dense = (
        true_fused_fwd_bwd_ms / dense_causal_fwd_bwd_ms if dense_causal_fwd_bwd_ms > 0 else float("inf")
    )
    true_fused_fwd_bwd_vs_split = true_fused_fwd_bwd_ms / split_fwd_bwd_ms if split_fwd_bwd_ms > 0 else float("inf")

    print(
        f"{case.name}: shape=(B={case.batch_size}, T={case.seqlen}, H={case.nheads}, KV={n_kv_heads}, D={case.headdim}) "
        f"split_max_diff={split_max_diff:.6f} split_mean_diff={split_mean_diff:.6f} "
        f"true_fused_max_diff={true_fused_max_diff:.6f} true_fused_mean_diff={true_fused_mean_diff:.6f} "
        f"split_fwd_ms={split_fwd_ms:.3f} true_fused_fwd_ms={true_fused_fwd_ms:.3f} dense_causal_fwd_ms={dense_causal_fwd_ms:.3f} "
        f"split_fwd_vs_dense={split_fwd_vs_dense:.2f}x true_fused_fwd_vs_dense={true_fused_fwd_vs_dense:.2f}x "
        f"true_fused_fwd_vs_split={true_fused_fwd_vs_split:.2f}x "
        f"split_bwd_ms={split_bwd_ms:.3f} true_fused_bwd_ms={true_fused_bwd_ms:.3f} dense_causal_bwd_ms={dense_causal_bwd_ms:.3f} "
        f"split_vs_dense={split_vs_dense:.2f}x true_fused_vs_dense={true_fused_vs_dense:.2f}x "
        f"true_fused_vs_split={true_fused_vs_split:.2f}x "
        f"split_fwd_bwd_ms={split_fwd_bwd_ms:.3f} true_fused_fwd_bwd_ms={true_fused_fwd_bwd_ms:.3f} "
        f"dense_causal_fwd_bwd_ms={dense_causal_fwd_bwd_ms:.3f} split_fwd_bwd_vs_dense={split_fwd_bwd_vs_dense:.2f}x "
        f"true_fused_fwd_bwd_vs_dense={true_fused_fwd_bwd_vs_dense:.2f}x "
        f"true_fused_fwd_bwd_vs_split={true_fused_fwd_bwd_vs_split:.2f}x"
    )


def _run_cases_once():
    _ensure_cuda_ready()
    print(f"device={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
    for case in CANONICAL_CASES:
        run_case(case)


def main():
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
