import time
import os
from dataclasses import dataclass
from contextlib import contextmanager

import torch

from flash_attn.cute import (
    build_hsa_schedule,
    flash_attn_func,
    flash_attn_hsa_func,
    flash_attn_hsa_sparse_func,
    hsa_reference_attention,
)


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


def _run_sparse_attention(q, k, v, keep_ids, hash_ids, schedule, *, use_fused_fwd=False, use_packed_bwd=False):
    with _temporary_env(
        FLASH_ATTN_HSA_USE_FUSED_FWD="1" if use_fused_fwd else None,
        FLASH_ATTN_HSA_USE_PACKED_BWD="1" if use_packed_bwd else None,
        FLASH_ATTN_HSA_USE_HYBRID_BWD=None,
    ):
        return _unwrap_output(
            flash_attn_hsa_sparse_func(q, k, v, keep_ids=keep_ids, hash_ids=hash_ids, hsa_schedule=schedule)
        )


def _measure_backward_ms(forward_fn, q_data, k_data, v_data, warmup_iters, benchmark_iters):
    q = q_data.clone().requires_grad_(True)
    k = k_data.clone().requires_grad_(True)
    v = v_data.clone().requires_grad_(True)
    loss = forward_fn(q, k, v).float().square().mean()

    return _measure_ms(
        lambda: torch.autograd.grad(loss, (q, k, v), retain_graph=True),
        warmup_iters,
        benchmark_iters,
    )


def _measure_fwd_bwd_ms(forward_fn, q_data, k_data, v_data, warmup_iters, benchmark_iters):
    def _run():
        q = q_data.clone().requires_grad_(True)
        k = k_data.clone().requires_grad_(True)
        v = v_data.clone().requires_grad_(True)
        out = forward_fn(q, k, v)
        out.float().square().mean().backward()

    return _measure_ms(_run, warmup_iters, benchmark_iters)


def run_case(case: BenchmarkCase):
    device = "cuda"
    dtype = torch.bfloat16
    keep_ids, hash_ids = _make_hsa_metadata(case.batch_size, case.seqlen, device)
    schedule = build_hsa_schedule(keep_ids, hash_ids)
    n_kv_heads = case.n_kv_heads if case.n_kv_heads is not None else case.nheads
    q_data = torch.randn(case.batch_size, case.seqlen, case.nheads, case.headdim, device=device, dtype=dtype)
    k_data = torch.randn(case.batch_size, case.seqlen, n_kv_heads, case.headdim, device=device, dtype=dtype)
    v_data = torch.randn(case.batch_size, case.seqlen, n_kv_heads, case.headdim, device=device, dtype=dtype)

    maskmod_forward = lambda q, k, v: _unwrap_output(flash_attn_hsa_func(q, k, v, keep_ids, hash_ids))
    sparse_forward = lambda q, k, v: _run_sparse_attention(
        q, k, v, keep_ids, hash_ids, schedule, use_fused_fwd=False, use_packed_bwd=False
    )
    packed_forward = lambda q, k, v: _run_sparse_attention(
        q, k, v, keep_ids, hash_ids, schedule, use_fused_fwd=False, use_packed_bwd=True
    )
    fused_forward = lambda q, k, v: _run_sparse_attention(
        q, k, v, keep_ids, hash_ids, schedule, use_fused_fwd=True, use_packed_bwd=False
    )
    dense_causal_forward = lambda q, k, v: _unwrap_output(flash_attn_func(q, k, v, causal=True))
    dense_full_forward = lambda q, k, v: _unwrap_output(flash_attn_func(q, k, v, causal=False))
    dense_ref_forward = lambda q, k, v: hsa_reference_attention(q, k, v, keep_ids, hash_ids)

    out_fa4 = maskmod_forward(q_data, k_data, v_data)
    out_sparse = sparse_forward(q_data, k_data, v_data)
    out_packed = packed_forward(q_data, k_data, v_data)
    out_fused = fused_forward(q_data, k_data, v_data)
    out_ref = dense_ref_forward(q_data, k_data, v_data)
    max_diff = (out_fa4.float() - out_ref.float()).abs().max().item()
    mean_diff = (out_fa4.float() - out_ref.float()).abs().mean().item()
    sparse_max_diff = (out_sparse.float() - out_ref.float()).abs().max().item()
    sparse_mean_diff = (out_sparse.float() - out_ref.float()).abs().mean().item()
    packed_max_diff = (out_packed.float() - out_ref.float()).abs().max().item()
    packed_mean_diff = (out_packed.float() - out_ref.float()).abs().mean().item()
    fused_max_diff = (out_fused.float() - out_ref.float()).abs().max().item()
    fused_mean_diff = (out_fused.float() - out_ref.float()).abs().mean().item()

    fa4_ms = _measure_ms(
        lambda: maskmod_forward(q_data, k_data, v_data),
        case.warmup_iters,
        case.benchmark_iters,
    )
    sparse_ms = _measure_ms(
        lambda: sparse_forward(q_data, k_data, v_data),
        case.warmup_iters,
        case.benchmark_iters,
    )
    packed_ms = _measure_ms(
        lambda: packed_forward(q_data, k_data, v_data),
        case.warmup_iters,
        case.benchmark_iters,
    )
    fused_ms = _measure_ms(
        lambda: fused_forward(q_data, k_data, v_data),
        case.warmup_iters,
        case.benchmark_iters,
    )
    dense_causal_ms = _measure_ms(
        lambda: dense_causal_forward(q_data, k_data, v_data),
        case.warmup_iters,
        case.benchmark_iters,
    )
    dense_full_ms = _measure_ms(
        lambda: dense_full_forward(q_data, k_data, v_data),
        case.warmup_iters,
        case.benchmark_iters,
    )
    ref_ms = _measure_ms(
        lambda: dense_ref_forward(q_data, k_data, v_data),
        max(1, case.warmup_iters // 2),
        max(5, case.benchmark_iters // 2),
    )

    maskmod_bwd_ms = _measure_backward_ms(maskmod_forward, q_data, k_data, v_data, case.warmup_iters, case.benchmark_iters)
    sparse_bwd_ms = _measure_backward_ms(sparse_forward, q_data, k_data, v_data, case.warmup_iters, case.benchmark_iters)
    packed_bwd_ms = _measure_backward_ms(packed_forward, q_data, k_data, v_data, case.warmup_iters, case.benchmark_iters)
    ref_bwd_ms = _measure_backward_ms(
        dense_ref_forward,
        q_data,
        k_data,
        v_data,
        max(1, case.warmup_iters // 2),
        max(5, case.benchmark_iters // 2),
    )

    maskmod_fwd_bwd_ms = _measure_fwd_bwd_ms(
        maskmod_forward, q_data, k_data, v_data, case.warmup_iters, case.benchmark_iters
    )
    sparse_fwd_bwd_ms = _measure_fwd_bwd_ms(
        sparse_forward, q_data, k_data, v_data, case.warmup_iters, case.benchmark_iters
    )
    packed_fwd_bwd_ms = _measure_fwd_bwd_ms(
        packed_forward, q_data, k_data, v_data, case.warmup_iters, case.benchmark_iters
    )
    ref_fwd_bwd_ms = _measure_fwd_bwd_ms(
        dense_ref_forward,
        q_data,
        k_data,
        v_data,
        max(1, case.warmup_iters // 2),
        max(5, case.benchmark_iters // 2),
    )

    speedup = ref_ms / fa4_ms if fa4_ms > 0 else float("inf")
    sparse_speedup = ref_ms / sparse_ms if sparse_ms > 0 else float("inf")
    packed_speedup = ref_ms / packed_ms if packed_ms > 0 else float("inf")
    fused_speedup = ref_ms / fused_ms if fused_ms > 0 else float("inf")

    print(
        f"{case.name}: shape=(B={case.batch_size}, T={case.seqlen}, H={case.nheads}, KV={n_kv_heads}, D={case.headdim}) "
        f"maskmod_max_diff={max_diff:.6f} maskmod_mean_diff={mean_diff:.6f} "
        f"sparse_max_diff={sparse_max_diff:.6f} sparse_mean_diff={sparse_mean_diff:.6f} "
        f"packed_max_diff={packed_max_diff:.6f} packed_mean_diff={packed_mean_diff:.6f} "
        f"fused_max_diff={fused_max_diff:.6f} fused_mean_diff={fused_mean_diff:.6f} "
        f"maskmod_ms={fa4_ms:.3f} sparse_ms={sparse_ms:.3f} packed_ms={packed_ms:.3f} fused_ms={fused_ms:.3f} "
        f"dense_causal_ms={dense_causal_ms:.3f} dense_full_ms={dense_full_ms:.3f} ref_ms={ref_ms:.3f} "
        f"maskmod_bwd_ms={maskmod_bwd_ms:.3f} sparse_bwd_ms={sparse_bwd_ms:.3f} packed_bwd_ms={packed_bwd_ms:.3f} ref_bwd_ms={ref_bwd_ms:.3f} "
        f"maskmod_fwd_bwd_ms={maskmod_fwd_bwd_ms:.3f} sparse_fwd_bwd_ms={sparse_fwd_bwd_ms:.3f} "
        f"packed_fwd_bwd_ms={packed_fwd_bwd_ms:.3f} ref_fwd_bwd_ms={ref_fwd_bwd_ms:.3f} "
        f"maskmod_speedup={speedup:.2f}x sparse_speedup={sparse_speedup:.2f}x packed_speedup={packed_speedup:.2f}x fused_speedup={fused_speedup:.2f}x"
    )


def main():
    assert torch.cuda.is_available(), "CUDA is required"
    print(f"device={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
    cases = [
        BenchmarkCase(name="small-correctness", batch_size=1, seqlen=137, nheads=4, headdim=64),
        BenchmarkCase(name="train-eq", batch_size=2, seqlen=1024, nheads=8, headdim=64, n_kv_heads=8),
        BenchmarkCase(name="train-gqa", batch_size=2, seqlen=1024, nheads=8, headdim=64, n_kv_heads=2),
        BenchmarkCase(name="longer-eq", batch_size=2, seqlen=2048, nheads=8, headdim=64, n_kv_heads=8),
    ]
    for case in cases:
        run_case(case)


if __name__ == "__main__":
    main()
