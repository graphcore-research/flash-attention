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
_SPLINE_SOLLYA_HEADER = (
    _ROOT / "autonumerics_zero" / "spline_ops" / "spline_structs_sollya_bf16.cuh"
)
_CUDA_INCLUDE = Path("/usr/local/cuda/include")



_DEGREES = (3, 4, 5, 6)
_SOURCES = ("current", "sollya")
_HANDWRITTEN_SYMBOLS = {
    *(f"fa4_spline_tanh_fwd_{source}_d{degree}_bf16x2" for source in _SOURCES for degree in _DEGREES),
    *(f"fa4_spline_sigmoid_fwd_{source}_d{degree}_bf16x2" for source in _SOURCES for degree in _DEGREES),
    *(f"fa4_spline_sigmoid_grad_{source}_d{degree}_bf16x2" for source in _SOURCES for degree in _DEGREES),
    # Backward-compatible aliases for the current default rows.
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
    if not _SPLINE_SOLLYA_HEADER.is_file():
        raise RuntimeError(f"Missing Sollya spline header: {_SPLINE_SOLLYA_HEADER}")
    if not _CUDA_INCLUDE.is_dir():
        raise RuntimeError(f"Missing CUDA include directory: {_CUDA_INCLUDE}")


def handwritten_symbol_names() -> set[str]:
    return set(_HANDWRITTEN_SYMBOLS)


@lru_cache(maxsize=1)
def get_handwritten_spline_ptx() -> str:
    require_device_backend_support()
    tanh_wrappers = []
    sigmoid_wrappers = []
    sigmoid_grad_wrappers = []
    for degree in _DEGREES:
        tanh_wrappers.append(
            f"""
extern "C" __device__ __noinline__ unsigned int fa4_spline_tanh_fwd_current_d{degree}_bf16x2(float x, float y) {{
    return fa4_pack_bits(TANH_FWD_D{degree}_ODD_BF16::evaluate(fa4_pack_bf16x2(x, y)));
}}

extern "C" __device__ __noinline__ unsigned int fa4_spline_tanh_fwd_sollya_d{degree}_bf16x2(float x, float y) {{
    return fa4_pack_bits(TANH_FWD_D{degree}_ODD_SOLLYA_BF16::evaluate(fa4_pack_bf16x2(x, y)));
}}
"""
        )
        sigmoid_wrappers.append(
            f"""
extern "C" __device__ __noinline__ unsigned int fa4_spline_sigmoid_fwd_current_d{degree}_bf16x2(float x, float y) {{
    return fa4_pack_bits(SIGMOID_FWD_D{degree}_ODD_BF16::evaluate(fa4_pack_bf16x2(x, y)));
}}

extern "C" __device__ __noinline__ unsigned int fa4_spline_sigmoid_fwd_sollya_d{degree}_bf16x2(float x, float y) {{
    return fa4_pack_bits(SIGMOID_FWD_D{degree}_ODD_SOLLYA_BF16::evaluate(fa4_pack_bf16x2(x, y)));
}}
"""
        )
        sigmoid_grad_wrappers.append(
            f"""
extern "C" __device__ __noinline__ unsigned int fa4_spline_sigmoid_grad_current_d{degree}_bf16x2(float x, float y) {{
    return fa4_pack_bits(SIGMOID_BWD_D{degree}_EVEN_BF16::evaluate(fa4_pack_bf16x2(x, y)));
}}

extern "C" __device__ __noinline__ unsigned int fa4_spline_sigmoid_grad_sollya_d{degree}_bf16x2(float x, float y) {{
    return fa4_pack_bits(SIGMOID_BWD_D{degree}_EVEN_SOLLYA_BF16::evaluate(fa4_pack_bf16x2(x, y)));
}}
"""
        )
    src = f"""
#include <cuda_bf16.h>
#include "{_SPLINE_HEADER.as_posix()}"
#include "{_SPLINE_SOLLYA_HEADER.as_posix()}"

static __device__ __forceinline__ __nv_bfloat162 fa4_pack_bf16x2(float x, float y) {{
    return __floats2bfloat162_rn(x, y);
}}

static __device__ __forceinline__ unsigned int fa4_pack_bits(__nv_bfloat162 v) {{
    return *reinterpret_cast<unsigned int*>(&v);
}}

{''.join(tanh_wrappers)}
{''.join(sigmoid_wrappers)}
{''.join(sigmoid_grad_wrappers)}

// Backward-compatible aliases for the current default symbols.
extern "C" __device__ __noinline__ unsigned int fa4_spline_tanh_fwd_d3_bf16x2(float x, float y) {{
    return fa4_spline_tanh_fwd_current_d3_bf16x2(x, y);
}}
extern "C" __device__ __noinline__ unsigned int fa4_spline_tanh_fwd_d4_bf16x2(float x, float y) {{
    return fa4_spline_tanh_fwd_current_d4_bf16x2(x, y);
}}
extern "C" __device__ __noinline__ unsigned int fa4_spline_tanh_fwd_d5_bf16x2(float x, float y) {{
    return fa4_spline_tanh_fwd_current_d5_bf16x2(x, y);
}}
extern "C" __device__ __noinline__ unsigned int fa4_spline_tanh_fwd_d6_bf16x2(float x, float y) {{
    return fa4_spline_tanh_fwd_current_d6_bf16x2(x, y);
}}
extern "C" __device__ __noinline__ unsigned int fa4_spline_sigmoid_fwd_d3_bf16x2(float x, float y) {{
    return fa4_spline_sigmoid_fwd_current_d3_bf16x2(x, y);
}}
extern "C" __device__ __noinline__ unsigned int fa4_spline_sigmoid_grad_d5_bf16x2(float x, float y) {{
    return fa4_spline_sigmoid_grad_current_d5_bf16x2(x, y);
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
    ptx = get_handwritten_spline_ptx()
    body = _extract_function_body(ptx, symbol)
    return _translate_body_to_inline_asm(body, symbol)
