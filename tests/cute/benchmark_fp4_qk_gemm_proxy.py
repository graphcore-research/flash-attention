import argparse

import torch


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _bench_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _quantize_nvfp4(tk_mod, x: torch.Tensor):
    rows, cols = x.shape
    fp4 = torch.empty((rows, cols // 2), dtype=torch.float4_e2m1fn_x2, device="cuda")
    scale = torch.empty((rows // 128, cols // 64, 512), dtype=torch.float8_e4m3fn, device="cuda")
    scale_global = torch.empty((1,), dtype=torch.float32, device="cuda")
    tk_mod.nvfp4_quantize(x, fp4, scale, scale_global, False)
    return fp4, scale, scale_global


def _make_case_inputs(*, seqlen: int, head_dim: int, seed: int):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    head_dim_padded = 128 if head_dim <= 128 else ((head_dim + 127) // 128) * 128
    q = torch.randn((seqlen, head_dim), dtype=torch.bfloat16, device="cuda", generator=generator) / (head_dim ** 0.5)
    k = torch.randn((seqlen, head_dim), dtype=torch.bfloat16, device="cuda", generator=generator) / (head_dim ** 0.5)
    if head_dim_padded > head_dim:
        q = torch.nn.functional.pad(q, (0, head_dim_padded - head_dim))
        k = torch.nn.functional.pad(k, (0, head_dim_padded - head_dim))
    return q.contiguous(), k.contiguous(), head_dim_padded


def _load_tk_nvfp4():
    import sys

    sys.path.insert(0, "/workspace/codebases/fp4_matmul")
    from fp4_cce_TK.nvfp4_cce_tk import _get_tk_nvfp4

    return _get_tk_nvfp4()


def _benchmark_case(*, tk_mod, seqlen: int, head_dim: int, warmup: int, iters: int):
    q, k, head_dim_padded = _make_case_inputs(
        seqlen=seqlen,
        head_dim=head_dim,
        seed=37_000 + seqlen * 17 + head_dim * 101,
    )
    q_fp4, q_scale, q_scale_global = _quantize_nvfp4(tk_mod, q)
    k_fp4, k_scale, k_scale_global = _quantize_nvfp4(tk_mod, k)
    out_fp4 = torch.empty((seqlen, seqlen), dtype=torch.bfloat16, device="cuda")
    out_bf16 = torch.empty((seqlen, seqlen), dtype=torch.bfloat16, device="cuda")
    k_t = k.transpose(0, 1).contiguous()

    def run_fp4():
        tk_mod.nvfp4_gemm(q_fp4, q_scale, q_scale_global, k_fp4, k_scale, k_scale_global, out_fp4)

    def run_bf16():
        torch.matmul(q, k_t, out=out_bf16)

    fp4_ms = _bench_ms(run_fp4, warmup=warmup, iters=iters)
    bf16_ms = _bench_ms(run_bf16, warmup=warmup, iters=iters)
    return {
        "seqlen": seqlen,
        "head_dim": head_dim,
        "head_dim_padded": head_dim_padded,
        "fp4_gemm_ms": fp4_ms,
        "bf16_matmul_ms": bf16_ms,
        "fp4_over_bf16": fp4_ms / bf16_ms,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Proxy benchmark for pure QK score GEMM using the in-tree TK NVFP4 GEMM path. "
            "This isolates Q @ K^T from FlashAttention softmax/P@V work."
        )
    )
    parser.add_argument("--seqlens", default="512,2048", help="Comma-separated seqlens.")
    parser.add_argument("--head-dims", default="64,128", help="Comma-separated head dims.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    tk_mod = _load_tk_nvfp4()
    seqlens = _parse_csv_ints(args.seqlens)
    head_dims = _parse_csv_ints(args.head_dims)

    print("seqlen,head_dim,head_dim_padded,fp4_gemm_ms,bf16_matmul_ms,fp4_over_bf16")
    for head_dim in head_dims:
        for seqlen in seqlens:
            result = _benchmark_case(
                tk_mod=tk_mod,
                seqlen=seqlen,
                head_dim=head_dim,
                warmup=args.warmup,
                iters=args.iters,
            )
            print(
                f"{result['seqlen']},"
                f"{result['head_dim']},"
                f"{result['head_dim_padded']},"
                f"{result['fp4_gemm_ms']:.6f},"
                f"{result['bf16_matmul_ms']:.6f},"
                f"{result['fp4_over_bf16']:.3f}"
            )


if __name__ == "__main__":
    main()
