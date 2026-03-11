from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

from cuda.bindings import nvrtc


_ROOT = Path(__file__).resolve().parents[3]
_SPLINE_HEADER = (
    _ROOT / "autonumerics_zero" / "spline_ops" / "spline_structs_odd_bf16.cuh"
)
_CUDA_INCLUDE = Path("/usr/local/cuda/include")



_SIGMOID_FWD_D3_CANONICAL_INLINE_ASM = """{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<24>;
	.reg .b32 	%r<2>;
	mov.f32 %f1, $1;
	mov.f32 %f2, $2;
	abs.f32 %f3, %f1;
	abs.f32 %f4, %f2;
	mov.f32 %f5, 0f40C00000;
	min.f32 %f6, %f3, %f5;
	min.f32 %f7, %f4, %f5;
	mov.f32 %f8, 0f3B5E2000;
	mov.f32 %f9, 0fBD5A8000;
	fma.rn.f32 %f10, %f8, %f6, %f9;
	fma.rn.f32 %f11, %f8, %f7, %f9;
	mov.f32 %f12, 0f3E8FE000;
	fma.rn.f32 %f13, %f10, %f6, %f12;
	fma.rn.f32 %f14, %f11, %f7, %f12;
	mov.f32 %f15, 0f3F000000;
	fma.rn.f32 %f16, %f1, %f13, %f15;
	fma.rn.f32 %f17, %f2, %f14, %f15;
	setp.gt.f32 %p1, %f1, 0f00000000;
	setp.gt.f32 %p2, %f2, 0f00000000;
	setp.lt.f32 %p3, %f3, %f5;
	setp.lt.f32 %p4, %f4, %f5;
	selp.f32 %f18, 0f3F800000, 0f00000000, %p1;
	selp.f32 %f19, 0f3F800000, 0f00000000, %p2;
	selp.f32 %f20, %f16, %f18, %p3;
	selp.f32 %f21, %f17, %f19, %p4;
	{ cvt.rn.bf16x2.f32 %r1, %f21, %f20;}
	mov.b32 $0, %r1;
}
"""

_HANDWRITTEN_SYMBOLS = {
    "fa4_spline_tanh_fwd_d3_bf16x2",
    "fa4_spline_tanh_fwd_d4_bf16x2",
    "fa4_spline_tanh_fwd_d5_bf16x2",
    "fa4_spline_tanh_fwd_d6_bf16x2",
    "fa4_spline_sigmoid_fwd_d3_bf16x2",
    "fa4_spline_sigmoid_grad_d5_bf16x2",
}


def _nvrtc_check(result: int, prog, action: str) -> None:
    if result == 0:
        return
    _, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
    log = bytearray(log_size)
    nvrtc.nvrtcGetProgramLog(prog, log)
    raise RuntimeError(f"NVRTC {action} failed with code {result}: {bytes(log).decode(errors='ignore')}")


def _strip_ptx_module_header(ptx: str) -> str:
    ptx = ptx.replace("\x00", "")
    lines = ptx.splitlines()
    start = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(".visible .func") or stripped.startswith(".func"):
            start = idx
            break
    return "\n".join(lines[start:]).rstrip() + "\n"


def require_device_backend_support() -> None:
    if not _SPLINE_HEADER.is_file():
        raise RuntimeError(f"Missing spline header: {_SPLINE_HEADER}")
    if not _CUDA_INCLUDE.is_dir():
        raise RuntimeError(f"Missing CUDA include directory: {_CUDA_INCLUDE}")


def handwritten_symbol_names() -> set[str]:
    return set(_HANDWRITTEN_SYMBOLS)


@lru_cache(maxsize=1)
def get_handwritten_spline_ptx() -> str:
    require_device_backend_support()
    src = f"""
#include <cuda_bf16.h>
#include "{_SPLINE_HEADER.as_posix()}"

static __device__ __forceinline__ __nv_bfloat162 fa4_pack_bf16x2(float x, float y) {{
    return __floats2bfloat162_rn(x, y);
}}

static __device__ __forceinline__ unsigned int fa4_pack_bits(__nv_bfloat162 v) {{
    return *reinterpret_cast<unsigned int*>(&v);
}}

extern "C" __device__ __noinline__ unsigned int fa4_spline_tanh_fwd_d3_bf16x2(float x, float y) {{
    return fa4_pack_bits(TANH_FWD_D3_ODD_BF16::evaluate(fa4_pack_bf16x2(x, y)));
}}

extern "C" __device__ __noinline__ unsigned int fa4_spline_tanh_fwd_d4_bf16x2(float x, float y) {{
    return fa4_pack_bits(TANH_FWD_D4_ODD_BF16::evaluate(fa4_pack_bf16x2(x, y)));
}}

extern "C" __device__ __noinline__ unsigned int fa4_spline_tanh_fwd_d5_bf16x2(float x, float y) {{
    return fa4_pack_bits(TANH_FWD_D5_ODD_BF16::evaluate(fa4_pack_bf16x2(x, y)));
}}

extern "C" __device__ __noinline__ unsigned int fa4_spline_tanh_fwd_d6_bf16x2(float x, float y) {{
    return fa4_pack_bits(TANH_FWD_D6_ODD_BF16::evaluate(fa4_pack_bf16x2(x, y)));
}}

extern "C" __device__ __noinline__ unsigned int fa4_spline_sigmoid_fwd_d3_bf16x2(float x, float y) {{
    constexpr float clamp = 6.0f;
    constexpr float c0 = 0.281005859375f;
    constexpr float c1 = -0.0533447265625f;
    constexpr float c2 = 0.0033893585205078125f;
    float ax = fabsf(x);
    float ay = fabsf(y);
    float tx = fminf(ax, clamp);
    float ty = fminf(ay, clamp);
    float px = fmaf(fmaf(c2, tx, c1), tx, c0);
    float py = fmaf(fmaf(c2, ty, c1), ty, c0);
    float sx = ax < clamp ? fmaf(x, px, 0.5f) : (x > 0.0f ? 1.0f : 0.0f);
    float sy = ay < clamp ? fmaf(y, py, 0.5f) : (y > 0.0f ? 1.0f : 0.0f);
    return fa4_pack_bits(fa4_pack_bf16x2(sx, sy));
}}

extern "C" __device__ __noinline__ unsigned int fa4_spline_sigmoid_grad_d5_bf16x2(float x, float y) {{
    return fa4_pack_bits(SIGMOID_BWD_D5_EVEN_BF16::evaluate(fa4_pack_bf16x2(x, y)));
}}
"""
    err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"fa4_handwritten_spline_wrappers.cu", 0, [], [])
    if err != 0:
        raise RuntimeError(f"NVRTC program creation failed with code {err}")
    opts = [
        b"--gpu-architecture=compute_80",
        b"--std=c++17",
        b"--device-as-default-execution-space",
        b"--relocatable-device-code=true",
        f"-I{_CUDA_INCLUDE.as_posix()}".encode(),
    ]
    compile_result = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0]
    _nvrtc_check(compile_result, prog, "compile")
    _, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
    ptx = bytearray(ptx_size)
    nvrtc.nvrtcGetPTX(prog, ptx)
    return _strip_ptx_module_header(bytes(ptx).decode(errors="ignore"))


def handwritten_spline_ptx_provider(ptx_content: str) -> str | None:
    del ptx_content
    return get_handwritten_spline_ptx()


def contains_handwritten_symbols(ptx_content: str) -> bool:
    return any(re.search(rf"\\b{re.escape(symbol)}\\b", ptx_content) for symbol in _HANDWRITTEN_SYMBOLS)


def _extract_function_body(ptx: str, symbol: str) -> str:
    header = re.search(
        rf"\.visible\s+\.func\s+\(\.param\s+\.b32\s+func_retval0\)\s+{re.escape(symbol)}\([^)]*\)\s*\{{",
        ptx,
        re.MULTILINE | re.DOTALL,
    )
    if header is None:
        raise RuntimeError(f"Failed to locate handwritten PTX body for {symbol}")
    body_start = header.end()
    depth = 1
    idx = body_start
    while idx < len(ptx):
        ch = ptx[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return ptx[body_start:idx]
        idx += 1
    raise RuntimeError(f"Failed to parse handwritten PTX body for {symbol}")


def _translate_body_to_inline_asm(body: str, symbol: str) -> str:
    lines: list[str] = []
    param0 = f"{symbol}_param_0"
    param1 = f"{symbol}_param_1"
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("ld.param.f32") and param0 in line:
            dst = line.split()[1].rstrip(",")
            lines.append(f"mov.f32 {dst}, $1;")
            continue
        if line.startswith("ld.param.f32") and param1 in line:
            dst = line.split()[1].rstrip(",")
            lines.append(f"mov.f32 {dst}, $2;")
            continue
        if line.startswith("st.param.b32") and "[func_retval0+0]" in line:
            src = line.split(",")[-1].strip().rstrip(";")
            lines.append(f"mov.b32 $0, {src};")
            continue
        if line == "ret;":
            continue
        lines.append(line)
    return "{\n\t" + "\n\t".join(lines) + "\n}\n"


@lru_cache(maxsize=None)
def get_handwritten_inline_asm(symbol: str) -> str:
    if symbol not in _HANDWRITTEN_SYMBOLS:
        raise ValueError(f"Unsupported handwritten spline symbol: {symbol}")
    if symbol == "fa4_spline_sigmoid_fwd_d3_bf16x2":
        return _SIGMOID_FWD_D3_CANONICAL_INLINE_ASM
    ptx = get_handwritten_spline_ptx()
    body = _extract_function_body(ptx, symbol)
    return _translate_body_to_inline_asm(body, symbol)
