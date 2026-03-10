import time
from dataclasses import dataclass

import torch

from flash_attn.cute import flash_attn_hsa_func, hsa_reference_attention


@dataclass
class BenchmarkCase:
    name: str
    batch_size: int
    seqlen: int
    nheads: int
    headdim: int
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


def run_case(case: BenchmarkCase):
    device = "cuda"
    dtype = torch.bfloat16
    keep_ids, hash_ids = _make_hsa_metadata(case.batch_size, case.seqlen, device)
    q = torch.randn(case.batch_size, case.seqlen, case.nheads, case.headdim, device=device, dtype=dtype)
    k = torch.randn(case.batch_size, case.seqlen, case.nheads, case.headdim, device=device, dtype=dtype)
    v = torch.randn(case.batch_size, case.seqlen, case.nheads, case.headdim, device=device, dtype=dtype)

    out_fa4 = _unwrap_output(flash_attn_hsa_func(q, k, v, keep_ids, hash_ids))
    out_ref = hsa_reference_attention(q, k, v, keep_ids, hash_ids)
    max_diff = (out_fa4.float() - out_ref.float()).abs().max().item()
    mean_diff = (out_fa4.float() - out_ref.float()).abs().mean().item()

    fa4_ms = _measure_ms(
        lambda: _unwrap_output(flash_attn_hsa_func(q, k, v, keep_ids, hash_ids)),
        case.warmup_iters,
        case.benchmark_iters,
    )
    ref_ms = _measure_ms(
        lambda: hsa_reference_attention(q, k, v, keep_ids, hash_ids),
        max(1, case.warmup_iters // 2),
        max(5, case.benchmark_iters // 2),
    )
    speedup = ref_ms / fa4_ms if fa4_ms > 0 else float("inf")

    print(
        f"{case.name}: shape=(B={case.batch_size}, T={case.seqlen}, H={case.nheads}, D={case.headdim}) "
        f"max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} "
        f"fa4_ms={fa4_ms:.3f} ref_ms={ref_ms:.3f} speedup={speedup:.2f}x"
    )


def main():
    assert torch.cuda.is_available(), "CUDA is required"
    print(f"device={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
    cases = [
        BenchmarkCase(name="small-correctness", batch_size=1, seqlen=137, nheads=4, headdim=64),
        BenchmarkCase(name="train-shape", batch_size=2, seqlen=1024, nheads=8, headdim=64),
    ]
    for case in cases:
        run_case(case)


if __name__ == "__main__":
    main()
