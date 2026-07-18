# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

"""Benchmark the sigmoid evaluators used by FA4 FlashSigmoid attention."""

import argparse
import math

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
from cutlass.cute.runtime import from_dlpack

from flash_attn.cute.utils import flash_sigmoid_exp2_poly_2, sigmoid_native_2


THREADS = 256
VALUES_PER_BLOCK = THREADS * 2


@cute.kernel
def _flash_sigmoid_kernel(
    variant: cutlass.Constexpr,
    degree: cutlass.Constexpr,
    sequence_length: cutlass.Constexpr,
    evaluations: cutlass.Constexpr,
    m_scores: cute.Tensor,
    m_output: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    pair_idx = bidx * THREADS + tidx
    element_idx = pair_idx * 2

    if element_idx + 1 < cute.size(m_scores):
        score_x = cutlass.Float32(m_scores[element_idx])
        score_y = cutlass.Float32(m_scores[element_idx + 1])
        acc_x = cutlass.Float32(0.0)
        acc_y = cutlass.Float32(0.0)
        bias = cutlass.Float32(-math.log(sequence_length))
        inv_sequence_length = cutlass.Float32(1.0 / sequence_length)

        # Slightly offset each independent evaluation to prevent common
        # subexpression elimination while preserving the B3 score distribution.
        for iteration in cutlass.range_constexpr(evaluations):
            offset = cutlass.Float32((iteration - evaluations // 2) * 0.03125)
            if cutlass.const_expr(variant == "sfu"):
                value_x, value_y = sigmoid_native_2(
                    score_x + offset + bias,
                    score_y - offset + bias,
                )
            else:
                value_x, value_y = flash_sigmoid_exp2_poly_2(
                    score_x + offset,
                    score_y - offset,
                    inv_sequence_length,
                    degree=degree,
                )
            acc_x += value_x
            acc_y += value_y

        m_output[element_idx] = acc_x
        m_output[element_idx + 1] = acc_y


@cute.jit
def _flash_sigmoid(
    variant: cutlass.Constexpr,
    degree: cutlass.Constexpr,
    sequence_length: cutlass.Constexpr,
    evaluations: cutlass.Constexpr,
    scores: cute.Tensor,
    output: cute.Tensor,
    stream: cuda.CUstream,
):
    _flash_sigmoid_kernel(
        variant,
        degree,
        sequence_length,
        evaluations,
        scores,
        output,
    ).launch(
        grid=[cute.ceil_div(cute.size(scores), VALUES_PER_BLOCK), 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
    )


def _compile(variant, degree, sequence_length, evaluations, scores, output, stream):
    return cute.compile(
        _flash_sigmoid,
        variant,
        degree,
        sequence_length,
        evaluations,
        from_dlpack(scores),
        from_dlpack(output),
        stream,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=1 << 20)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--evaluations", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if args.elements % 2:
        raise ValueError("--elements must be even")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("This benchmark requires an SM100 or newer CUDA GPU")

    torch.manual_seed(args.seed)
    scores = torch.randn(args.elements, device="cuda", dtype=torch.float32)
    output = torch.empty_like(scores)
    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    rows = []
    reference = None
    for variant, degree in (("sfu", 0), ("poly_d2", 2), ("poly_d3", 3)):
        compiled = _compile(
            variant,
            degree,
            args.sequence_length,
            args.evaluations,
            scores,
            output,
            stream,
        )
        compiled(from_dlpack(scores), from_dlpack(output), stream)
        torch.cuda.synchronize()
        actual = output.clone()
        if reference is None:
            reference = actual
            relative_mae = 0.0
        else:
            relative_mae = float(
                (actual - reference).abs().mean()
                / reference.abs().mean().clamp_min(1e-12)
            )

        elapsed_us = testing.benchmark(
            compiled,
            kernel_arguments=testing.JitArguments(
                from_dlpack(scores),
                from_dlpack(output),
                stream,
            ),
            warmup_iterations=args.warmup,
            iterations=args.iterations,
            use_cuda_graphs=True,
            stream=stream,
        )
        evaluations_per_second = (
            args.elements * args.evaluations / (elapsed_us * 1e-6)
        )
        rows.append((variant, elapsed_us, evaluations_per_second, relative_mae))

    sfu_rate = rows[0][2]
    print("variant     time_us   Geval/s   speedup_vs_sfu   relative_mae")
    for variant, elapsed_us, rate, relative_mae in rows:
        print(
            f"{variant:<10} {elapsed_us:8.3f} {rate / 1e9:9.2f} "
            f"{rate / sfu_rate:14.3f}x {relative_mae:14.6f}"
        )


if __name__ == "__main__":
    main()
