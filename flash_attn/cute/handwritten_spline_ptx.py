from __future__ import annotations

import os
import re
import struct
from functools import lru_cache
from pathlib import Path

from flash_attn.cute.polynomial_manifest import (
    FLASH_SIGMOID_DIRECT_D3_COEFFS,
    FLASH_SIGMOID_DIRECT_CLAMP,
    FLASH_SIGMOID_DIRECT_GRAD_D4_COEFFS,
    FLASH_SIGMOID_DIRECT_MIDPOINTS,
    FLASH_SIGMOID_DIRECT_SPLIT,
    get_softcap_tanh_analytical_backward_coeffs,
    get_softcap_tanh_forward_spec,
)

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
    *(f"fa4_spline_tanh_grad_analytical_{source}_d{degree}_bf16x2" for source in _SOURCES for degree in _DEGREES),
    *(f"fa4_spline_sigmoid_fwd_{source}_d{degree}_bf16x2" for source in _SOURCES for degree in _DEGREES),
    *(f"fa4_spline_sigmoid_grad_{source}_d{degree}_bf16x2" for source in _SOURCES for degree in _DEGREES),
    # Backward-compatible aliases for the current default rows.
    "fa4_spline_tanh_fwd_d3_bf16x2",
    "fa4_spline_tanh_fwd_d4_bf16x2",
    "fa4_spline_tanh_fwd_d5_bf16x2",
    "fa4_spline_tanh_fwd_d6_bf16x2",
    "fa4_spline_sigmoid_fwd_d3_bf16x2",
    "fa4_spline_sigmoid_grad_d5_bf16x2",
    "fa4_flash_sigmoid_direct_d3_bf16x2",
    "fa4_flash_sigmoid_direct_grad_d4_bf16x2",
    "fa4_flash_sigmoid_direct_d3_grad_d4_bf16x2",
}


def _float_literal(value: float) -> str:
    rendered = f"{value:.17g}"
    if "." not in rendered and "e" not in rendered:
        rendered += ".0"
    return rendered + "f"


def _bf16x2_literal(value: float) -> str:
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    bf16 = (bits >> 16) & 0xFFFF
    return f"fa4_from_bits(0x{bf16:04x}{bf16:04x}U)"


def _bf16x2_bits(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    bf16 = (bits >> 16) & 0xFFFF
    return (bf16 << 16) | bf16


def _f32_ptx_literal(value: float) -> str:
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    return f"0f{bits:08x}"


def _tanh_analytical_wrapper_source(degree: int, source: str) -> str:
    spec = get_softcap_tanh_forward_spec(
        degree=degree,
        backend="device",
        coeff_source=source,
    )
    coeffs = get_softcap_tanh_analytical_backward_coeffs(
        degree=degree,
        backend="device",
        coeff_source=source,
    )
    symbol = f"fa4_spline_tanh_grad_analytical_{source}_d{degree}_bf16x2"
    constants = "\n".join(
        f"    const __nv_bfloat162 c{idx} = __float2bfloat162_rn({_float_literal(value)});"
        for idx, value in enumerate(coeffs)
    )
    horner = f"    __nv_bfloat162 value = c{len(coeffs) - 1};\n"
    for idx in range(len(coeffs) - 2, -1, -1):
        horner += f"    value = __hfma2(t, value, c{idx});\n"
    return f"""
extern "C" __device__ __noinline__ unsigned int {symbol}(float x, float y) {{
    const __nv_bfloat162 input = fa4_pack_bf16x2(x, y);
    const unsigned int input_bits = fa4_pack_bits(input);
    const __nv_bfloat162 abs_input = fa4_from_bits(input_bits & 0x7fff7fffU);
    const __nv_bfloat162 clamp = __float2bfloat162_rn({_float_literal(spec.clamp)});
    const __nv_bfloat162 t = __hmin2(abs_input, clamp);
{constants}
{horner}
    const unsigned int active = __hlt2_mask(abs_input, clamp);
    return fa4_pack_bits(fa4_select_bf16x2(active, value, __float2bfloat162_rn(0.0f)));
}}
"""


def _direct_sigmoid_wrapper_source(*, gradient: bool) -> str:
    coeffs = (
        FLASH_SIGMOID_DIRECT_GRAD_D4_COEFFS
        if gradient
        else FLASH_SIGMOID_DIRECT_D3_COEFFS
    )
    symbol = (
        "fa4_flash_sigmoid_direct_grad_d4_bf16x2"
        if gradient
        else "fa4_flash_sigmoid_direct_d3_bf16x2"
    )
    declarations = [
        "    __nv_bfloat162 midpoint = "
        f"{_bf16x2_literal(FLASH_SIGMOID_DIRECT_MIDPOINTS[0])};"
    ]
    for coefficient, value in enumerate(coeffs[0]):
        declarations.append(
            f"    __nv_bfloat162 c{coefficient} = "
            f"{_bf16x2_literal(value)};"
        )
    selections = [
        "    const unsigned int positive = __hge2_mask(input, "
        f"{_bf16x2_literal(FLASH_SIGMOID_DIRECT_SPLIT)});",
        "    midpoint = fa4_select_bf16x2(positive, "
        f"{_bf16x2_literal(FLASH_SIGMOID_DIRECT_MIDPOINTS[1])}, midpoint);",
    ]
    for coefficient, value in enumerate(coeffs[1]):
        selections.append(
            f"    c{coefficient} = fa4_select_bf16x2(positive, "
            f"{_bf16x2_literal(value)}, c{coefficient});"
        )
    degree = len(coeffs[0]) - 1
    horner = [f"    __nv_bfloat162 value = c{degree};"]
    for coefficient in range(degree - 1, -1, -1):
        horner.append(
            f"    value = __hfma2(z, value, c{coefficient});"
        )
    return f"""
extern "C" __device__ __noinline__ unsigned int {symbol}(float x, float y) {{
    const __nv_bfloat162 input = fa4_pack_bf16x2(x, y);
    const __nv_bfloat162 zero = {_bf16x2_literal(0.0)};
{chr(10).join(declarations)}
{chr(10).join(selections)}
    const __nv_bfloat162 clamped = __hmax2(
        {_bf16x2_literal(-FLASH_SIGMOID_DIRECT_CLAMP)},
        __hmin2(input, {_bf16x2_literal(FLASH_SIGMOID_DIRECT_CLAMP)})
    );
    const __nv_bfloat162 z = __hsub2(clamped, midpoint);
{chr(10).join(horner)}
    return fa4_pack_bits(__hmax2(zero, value));
}}
"""


def _direct_sigmoid_inline_asm(*, gradient: bool) -> str:
    """Minimal packed-BF16 PTX for the production FlashSigmoid fits."""
    rows = (
        FLASH_SIGMOID_DIRECT_GRAD_D4_COEFFS
        if gradient
        else FLASH_SIGMOID_DIRECT_D3_COEFFS
    )
    degree = len(rows[0]) - 1
    registers = (
        "%fa4_input, %fa4_mask_lo, %fa4_mask_hi, %fa4_mask, "
        "%fa4_clamped, %fa4_value"
    )
    lines = [
        "{",
        "\t.reg .pred %fa4_pos_lo, %fa4_pos_hi;",
        f"\t.reg .b32 {registers};",
        "\tcvt.rn.bf16x2.f32 %fa4_input, $2, $1;",
        f"\tsetp.ge.f32 %fa4_pos_lo, $1, {_f32_ptx_literal(FLASH_SIGMOID_DIRECT_SPLIT)};",
        f"\tsetp.ge.f32 %fa4_pos_hi, $2, {_f32_ptx_literal(FLASH_SIGMOID_DIRECT_SPLIT)};",
        "\tselp.b32 %fa4_mask_lo, 0x0000ffff, 0, %fa4_pos_lo;",
        "\tselp.b32 %fa4_mask_hi, 0xffff0000, 0, %fa4_pos_hi;",
        "\tor.b32 %fa4_mask, %fa4_mask_lo, %fa4_mask_hi;",
    ]
    lines.extend(
        [
            f"\tmov.b32 %fa4_mask_lo, 0x{_bf16x2_bits(FLASH_SIGMOID_DIRECT_CLAMP):08x};",
            "\tmin.bf16x2 %fa4_clamped, %fa4_input, %fa4_mask_lo;",
            f"\tmov.b32 %fa4_mask_lo, 0x{_bf16x2_bits(-FLASH_SIGMOID_DIRECT_CLAMP):08x};",
            "\tmax.bf16x2 %fa4_clamped, %fa4_clamped, %fa4_mask_lo;",
            f"\tlop3.b32 %fa4_value, %fa4_mask, "
            f"0x{_bf16x2_bits(rows[1][degree]):08x}, "
            f"0x{_bf16x2_bits(rows[0][degree]):08x}, 0xca;",
        ]
    )
    for idx in range(degree - 1, -1, -1):
        lines.extend(
            [
                f"\tlop3.b32 %fa4_mask_hi, %fa4_mask, "
                f"0x{_bf16x2_bits(rows[1][idx]):08x}, "
                f"0x{_bf16x2_bits(rows[0][idx]):08x}, 0xca;",
                "\tfma.rn.bf16x2 %fa4_value, %fa4_clamped, "
                "%fa4_value, %fa4_mask_hi;",
            ]
        )
    lines.extend(
        [
            "\tmov.b32 %fa4_mask_lo, 0;",
            "\tmax.bf16x2 %fa4_value, %fa4_value, %fa4_mask_lo;",
            "\tmov.b32 $0, %fa4_value;",
            "}",
        ]
    )
    return "\n".join(lines) + "\n"


def _direct_sigmoid_with_grad_inline_asm() -> str:
    """Evaluate production D3 sigmoid and D4 derivative with shared setup."""
    forward_rows = FLASH_SIGMOID_DIRECT_D3_COEFFS
    gradient_rows = FLASH_SIGMOID_DIRECT_GRAD_D4_COEFFS
    lines = [
        "{",
        "\t.reg .pred %fa4_pos_lo, %fa4_pos_hi;",
        "\t.reg .b32 %fa4_input, %fa4_mask_lo, %fa4_mask_hi, %fa4_mask, "
        "%fa4_clamped, %fa4_value, %fa4_grad;",
        "\tcvt.rn.bf16x2.f32 %fa4_input, $3, $2;",
        f"\tsetp.ge.f32 %fa4_pos_lo, $2, {_f32_ptx_literal(FLASH_SIGMOID_DIRECT_SPLIT)};",
        f"\tsetp.ge.f32 %fa4_pos_hi, $3, {_f32_ptx_literal(FLASH_SIGMOID_DIRECT_SPLIT)};",
        "\tselp.b32 %fa4_mask_lo, 0x0000ffff, 0, %fa4_pos_lo;",
        "\tselp.b32 %fa4_mask_hi, 0xffff0000, 0, %fa4_pos_hi;",
        "\tor.b32 %fa4_mask, %fa4_mask_lo, %fa4_mask_hi;",
        f"\tmov.b32 %fa4_mask_lo, 0x{_bf16x2_bits(FLASH_SIGMOID_DIRECT_CLAMP):08x};",
        "\tmin.bf16x2 %fa4_clamped, %fa4_input, %fa4_mask_lo;",
        f"\tmov.b32 %fa4_mask_lo, 0x{_bf16x2_bits(-FLASH_SIGMOID_DIRECT_CLAMP):08x};",
        "\tmax.bf16x2 %fa4_clamped, %fa4_clamped, %fa4_mask_lo;",
    ]

    def append_horner(rows, output: str) -> None:
        degree = len(rows[0]) - 1
        lines.append(
            f"\tlop3.b32 {output}, %fa4_mask, "
            f"0x{_bf16x2_bits(rows[1][degree]):08x}, "
            f"0x{_bf16x2_bits(rows[0][degree]):08x}, 0xca;"
        )
        for idx in range(degree - 1, -1, -1):
            lines.extend(
                [
                    f"\tlop3.b32 %fa4_mask_hi, %fa4_mask, "
                    f"0x{_bf16x2_bits(rows[1][idx]):08x}, "
                    f"0x{_bf16x2_bits(rows[0][idx]):08x}, 0xca;",
                    f"\tfma.rn.bf16x2 {output}, %fa4_clamped, "
                    f"{output}, %fa4_mask_hi;",
                ]
            )

    append_horner(forward_rows, "%fa4_value")
    append_horner(gradient_rows, "%fa4_grad")
    lines.extend(
        [
            "\tmov.b32 %fa4_mask_lo, 0;",
            "\tmax.bf16x2 %fa4_value, %fa4_value, %fa4_mask_lo;",
            "\tmax.bf16x2 %fa4_grad, %fa4_grad, %fa4_mask_lo;",
            "\tmov.b32 $0, %fa4_value;",
            "\tmov.b32 $1, %fa4_grad;",
            "}",
        ]
    )
    return "\n".join(lines) + "\n"


def _compile_device_ptx(source: str, name: str, arch: str) -> str:
    err, prog = nvrtc.nvrtcCreateProgram(source.encode(), name.encode(), 0, [], [])
    if err != 0:
        raise RuntimeError(f"NVRTC program creation failed with code {err}")
    opts = [
        f"--gpu-architecture={arch}".encode(),
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
    tanh_grad_wrappers = []
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
        for source in _SOURCES:
            tanh_grad_wrappers.append(
                _tanh_analytical_wrapper_source(degree, source)
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

static __device__ __forceinline__ unsigned int fa4_pack_bits(__nv_bfloat162 value) {{
    return reinterpret_cast<const unsigned int &>(value);
}}

static __device__ __forceinline__ __nv_bfloat162 fa4_from_bits(unsigned int value) {{
    return reinterpret_cast<const __nv_bfloat162 &>(value);
}}

static __device__ __forceinline__ __nv_bfloat162 fa4_select_bf16x2(
    unsigned int mask,
    __nv_bfloat162 true_value,
    __nv_bfloat162 false_value
) {{
    const unsigned int true_bits = fa4_pack_bits(true_value);
    const unsigned int false_bits = fa4_pack_bits(false_value);
    unsigned int selected;
    asm("lop3.b32 %0, %1, %2, %3, 0xca;"
        : "=r"(selected)
        : "r"(mask), "r"(true_bits), "r"(false_bits));
    return fa4_from_bits(selected);
}}

{''.join(tanh_wrappers)}
{''.join(tanh_grad_wrappers)}
{''.join(sigmoid_wrappers)}
{''.join(sigmoid_grad_wrappers)}
{_direct_sigmoid_wrapper_source(gradient=False)}
{_direct_sigmoid_wrapper_source(gradient=True)}

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
    return _compile_device_ptx(
        src,
        "fa4_handwritten_spline_wrappers.cu",
        "compute_80",
    )


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
        if line.startswith("st.param.b32") and "func_retval0" in line:
            src = line.split(",")[-1].strip().rstrip(";")
            lines.append(f"mov.b32 $0, {src};")
            continue
        if line == "ret;":
            continue
        # The caller's generated PTX also uses names such as %f1 and %r1.
        # Keeping NVRTC's function-local names here lets the inline block
        # shadow input registers after LLVM substitutes $1/$2, silently
        # corrupting the evaluator.  Prefix every extracted register family;
        # the surrounding braces make reuse across expansions safe.
        line = re.sub(r"%(rd|rs|r|f|p)(?=<|\d)", r"%fa4_\1", line)
        lines.append(line)
    # Parameter/result moves are synthesized above and skip the per-line
    # rewrite via ``continue``. Apply the idempotent namespace rewrite once to
    # the complete block so those registers cannot collide with FA4's caller.
    lines = [
        re.sub(r"%(rd|rs|r|f|p)(?=<|\d)", r"%fa4_\1", line)
        for line in lines
    ]
    return "{\n\t" + "\n\t".join(lines) + "\n}\n"


@lru_cache(maxsize=None)
def get_handwritten_inline_asm(symbol: str) -> str:
    if symbol not in _HANDWRITTEN_SYMBOLS:
        raise ValueError(f"Unsupported handwritten spline symbol: {symbol}")
    if symbol == "fa4_flash_sigmoid_direct_d3_bf16x2":
        return _direct_sigmoid_inline_asm(gradient=False)
    if symbol == "fa4_flash_sigmoid_direct_grad_d4_bf16x2":
        return _direct_sigmoid_inline_asm(gradient=True)
    if symbol == "fa4_flash_sigmoid_direct_d3_grad_d4_bf16x2":
        return _direct_sigmoid_with_grad_inline_asm()
    ptx = get_handwritten_spline_ptx()
    body = _extract_function_body(ptx, symbol)
    return _translate_body_to_inline_asm(body, symbol)


def audit_handwritten_fast_path(symbol: str, max_instructions: int = 72) -> str:
    """Fail closed if a supposedly compact polynomial wrapper regresses."""
    asm = get_handwritten_inline_asm(symbol)
    instructions = []
    for raw_line in asm.splitlines():
        line = raw_line.strip()
        if not line or line in ("{", "}") or line.startswith("."):
            continue
        if line.endswith(";"):
            instructions.append(line)

    forbidden = ("ex2.", "rcp.", "lg2.", "tanh")
    found_forbidden = sorted(
        operation for operation in forbidden if any(operation in line for line in instructions)
    )
    if found_forbidden:
        raise RuntimeError(
            f"Handwritten polynomial {symbol} unexpectedly uses {found_forbidden}"
        )
    if len(instructions) > max_instructions:
        raise RuntimeError(
            f"Handwritten polynomial {symbol} expanded to {len(instructions)} PTX "
            f"instructions (budget {max_instructions})"
        )
    fma_count = sum("fma.rn.bf16x2" in line for line in instructions)
    if fma_count == 0:
        raise RuntimeError(f"Handwritten polynomial {symbol} contains no packed BF16 FMA")
    return f"{symbol}[{len(instructions)} PTX instructions,{fma_count} packed FMA]"
