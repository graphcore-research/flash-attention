# Copyright (c) 2025, Tri Dao.

import math
import hashlib
import inspect
import re
from typing import Type, Callable, Optional, Tuple, overload
from functools import partial

import cutlass
import cutlass.cute as cute

from cutlass import Float32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op, select_
from cutlass._mlir.dialects import nvvm, llvm
from cutlass.cute.runtime import from_dlpack


# cute.arch.{fma,mul,add}_packed_f32x2 uses RZ rounding mode by default
fma_packed_f32x2 = partial(cute.arch.fma_packed_f32x2, rnd=nvvm.RoundingModeKind.RN)
mul_packed_f32x2 = partial(cute.arch.mul_packed_f32x2, rnd=nvvm.RoundingModeKind.RN)
add_packed_f32x2 = partial(cute.arch.add_packed_f32x2, rnd=nvvm.RoundingModeKind.RN)
sub_packed_f32x2 = partial(
    cute.arch.calc_packed_f32x2_op,
    src_c=None,
    calc_func=nvvm.sub_packed_f32x2,
    rnd=nvvm.RoundingModeKind.RN,
)


def hash_callable(func: Callable, set_cute_hash=True) -> str:
    """Hash a callable based on the source code or bytecode and closure values.

    Fast-path: if the callable (or its __wrapped__ base) has a ``__cute_hash__``
    attribute, that value is returned immediately. Code-generation backends such
    as Inductor can set this attribute to avoid expensive runtime hashing.

    set_cute_hash: whether or not to set func.__cute_hash__ if not present
    """
    if hasattr(func, "__cute_hash__"):
        return func.__cute_hash__

    # Unwrap decorated functions (e.g., cute.jit wrappers).
    if hasattr(func, "__wrapped__"):
        base_func = func.__wrapped__
        if hasattr(base_func, "__cute_hash__"):
            return base_func.__cute_hash__
        func = base_func

    try:
        data = inspect.getsource(func).encode()
    except (OSError, TypeError):
        if hasattr(func, "__code__") and func.__code__ is not None:
            data = func.__code__.co_code
        else:
            data = repr(func).encode()

    hasher = hashlib.sha256(data)

    if hasattr(func, "__closure__") and func.__closure__ is not None:
        for idx, cell in enumerate(func.__closure__):
            cell_value = cell.cell_contents
            hasher.update(repr(cell_value).encode())

    hash = hasher.hexdigest()

    if set_cute_hash:
        func.__cute_hash__ = hash

    return hash


def create_softcap_scoremod(softcap_val):
    inv_softcap = 1.0 / softcap_val

    @cute.jit
    def scoremod_premask_fn(acc_S_SSA, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        scores = acc_S_SSA * inv_softcap
        return softcap_val * cute.math.tanh(scores, fastmath=True)

    return scoremod_premask_fn


def create_softcap_scoremod_bwd_native(softcap_val):
    """Backward for native softcap using SFU tanh: grad * (1 - tanh²(x))."""
    inv_softcap = 1.0 / softcap_val

    @cute.jit
    def scoremod_bwd_fn(grad, acc_S_SSA, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        scores = acc_S_SSA * inv_softcap
        t = cute.math.tanh(scores, fastmath=True)
        # tanh'(x) = 1 - tanh²(x)
        derivative = Float32(1.0) - t * t
        return grad * derivative

    return scoremod_bwd_fn


def create_softcap_scoremod_bwd_ste(softcap_val=None):
    """STE backward for softcap — passes grad through unchanged.

    Straight-Through Estimator: forward applies softcap * tanh(x/softcap),
    backward ignores the tanh derivative and just returns grad.
    This eliminates the expensive backward tanh computation (~300µs savings).
    """

    @cute.jit
    def scoremod_bwd_fn(grad, acc_S_SSA, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        return grad

    return scoremod_bwd_fn


# =============================================================================
# Spline tanh emulation — FMA-only, no SFU
# ODD polynomial: tanh(x) = sign(x) * |x| * poly(|x|)
# D5 fit on [0, 3.0], max abs error < 0.00053
# =============================================================================

@dsl_user_op
def fmin(a: float | Float32, b: float | Float32, *, loc=None, ip=None) -> Float32:
    """f32 min using PTX min.ftz.f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "min.ftz.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fabs_f32(a: float | Float32, *, loc=None, ip=None) -> Float32:
    """f32 absolute value using PTX abs.ftz.f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "abs.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def copysign_f32(mag: float | Float32, sign_val: float | Float32, *, loc=None, ip=None) -> Float32:
    """f32 copysign: returns mag with the sign of sign_val, using PTX."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(mag).ir_value(loc=loc, ip=ip),
                Float32(sign_val).ir_value(loc=loc, ip=ip),
            ],
            "copysign.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def tanh_emulation(x: Float32, *, loc=None, ip=None) -> Float32:
    """Scalar f32 spline tanh: tanh(x) ≈ x*poly(|x|) for |x|<3, ±1 otherwise."""
    poly_tanh_d6 = (
        0.9999999999996861,
        0.018184368627828466,
        -0.44364107914224865,
        0.21709437130752046,
        -0.023897373260303105,
        -0.0071969181772045645,
        0.0015005805508435293,
    )
    clamp = 3.0
    abs_x = fabs_f32(x, loc=loc, ip=ip)
    t = fmin(abs_x, clamp, loc=loc, ip=ip)
    h = evaluate_polynomial(t, poly_tanh_d6, loc=loc, ip=ip)
    poly_result = x * h
    # For |x| >= clamp, return ±tanh(3.0) ~ 0.995 instead of ±1 to keep gradients alive
    saturated = copysign_f32(0.9950547536867305, x, loc=loc, ip=ip)
    return select_(abs_x < clamp, poly_result, saturated)


@dsl_user_op
def tanh_emulation_2(
    x: Float32, y: Float32, *, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    """Paired f32x2 spline tanh for ILP — processes 2 values simultaneously."""
    poly_tanh_d6 = (
        0.9999999999996861,
        0.018184368627828466,
        -0.44364107914224865,
        0.21709437130752046,
        -0.023897373260303105,
        -0.0071969181772045645,
        0.0015005805508435293,
    )
    clamp = 3.0
    abs_x = fabs_f32(x, loc=loc, ip=ip)
    abs_y = fabs_f32(y, loc=loc, ip=ip)
    tx = fmin(abs_x, clamp, loc=loc, ip=ip)
    ty = fmin(abs_y, clamp, loc=loc, ip=ip)
    hx, hy = evaluate_polynomial_2(tx, ty, poly_tanh_d6, loc=loc, ip=ip)
    poly_x = x * hx
    poly_y = y * hy
    poly_y = y * hy
    sat_x = copysign_f32(0.9950547536867305, x, loc=loc, ip=ip)
    sat_y = copysign_f32(0.9950547536867305, y, loc=loc, ip=ip)
    return select_(abs_x < clamp, poly_x, sat_x), select_(abs_y < clamp, poly_y, sat_y)


@cute.jit
def tanh_emulationf(x: cute.TensorSSA | Float32) -> cute.TensorSSA | Float32:
    """tanh emulation for both vector and scalar, matching exp2f API."""
    if const_expr(isinstance(x, cute.TensorSSA)):
        res = cute.make_fragment(x.shape, Float32)
        res.store(x)
        for i in cutlass.range_constexpr(0, cute.size(x.shape), 2):
            res[i], res[i + 1] = tanh_emulation_2(res[i], res[i + 1])
        return res.load()
    else:
        return tanh_emulation(x)


# =============================================================================
# D3 Tanh — LIGHTWEIGHT (2 FMA + 1 mul), matches SPLINE_TANH_FWD_D3 CUDA kernel
# ODD polynomial: tanh(x) = sign(x) * |x| * poly(|x|), domain [0,3]
# Coefficients: c1=1.124, c2=-0.427, c3=0.054, max_abs_error ≈ 0.011
# =============================================================================

@dsl_user_op
def tanh_emulation_d3(x: Float32, *, loc=None, ip=None) -> Float32:
    """Scalar f32 D3 spline tanh: tanh(x) ≈ x*poly(|x|) for |x|<3, ±1 otherwise.
    Only 2 FMAs + 1 mul vs D6's 6 FMAs + 1 mul."""
    c1 = 1.1240234375
    c2 = -0.4267578125
    c3 = 0.054229736328125
    clamp = 3.0
    abs_x = fabs_f32(x, loc=loc, ip=ip)
    t = fmin(abs_x, clamp, loc=loc, ip=ip)
    # Horner: h(t) = t * (c1 + t*(c2 + c3*t))  — just 2 FMA + 1 mul
    h = t * Float32(c3) + Float32(c2)       # c3*t + c2
    h = t * h + Float32(c1)                 # t*(c3*t+c2) + c1
    poly_result = t * h                     # t * (c1 + t*(c2 + c3*t))
    # Restore sign via copysign on the result (tanh is odd)
    poly_result = copysign_f32(poly_result, x, loc=loc, ip=ip)
    # Saturate: |x| >= 3 → ±tanh(3) ≈ ±0.995
    saturated = copysign_f32(0.9950547536867305, x, loc=loc, ip=ip)
    return select_(abs_x < clamp, poly_result, saturated)


@dsl_user_op
def tanh_emulation_d3_2(
    x: Float32, y: Float32, *, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    """Paired f32x2 D3 spline tanh — 2 FMA + 1 mul per element for ILP."""
    c1 = 1.1240234375
    c2 = -0.4267578125
    c3 = 0.054229736328125
    clamp = 3.0
    abs_x = fabs_f32(x, loc=loc, ip=ip)
    abs_y = fabs_f32(y, loc=loc, ip=ip)
    tx = fmin(abs_x, clamp, loc=loc, ip=ip)
    ty = fmin(abs_y, clamp, loc=loc, ip=ip)
    # Horner 2-wide: h(t) = t*(c1 + t*(c2 + c3*t))
    hx, hy = fma_packed_f32x2((Float32(c3), Float32(c3)), (tx, ty), (Float32(c2), Float32(c2)), loc=loc, ip=ip)
    hx, hy = fma_packed_f32x2((tx, ty), (hx, hy), (Float32(c1), Float32(c1)), loc=loc, ip=ip)
    poly_x = tx * hx
    poly_y = ty * hy
    # Restore sign (tanh is odd)
    poly_x = copysign_f32(poly_x, x, loc=loc, ip=ip)
    poly_y = copysign_f32(poly_y, y, loc=loc, ip=ip)
    sat_x = copysign_f32(0.9950547536867305, x, loc=loc, ip=ip)
    sat_y = copysign_f32(0.9950547536867305, y, loc=loc, ip=ip)
    return select_(abs_x < clamp, poly_x, sat_x), select_(abs_y < clamp, poly_y, sat_y)


@cute.jit
def tanh_emulationf_d3(x: cute.TensorSSA | Float32) -> cute.TensorSSA | Float32:
    """D3 tanh emulation for both vector and scalar."""
    if const_expr(isinstance(x, cute.TensorSSA)):
        res = cute.make_fragment(x.shape, Float32)
        res.store(x)
        for i in cutlass.range_constexpr(0, cute.size(x.shape), 2):
            res[i], res[i + 1] = tanh_emulation_d3_2(res[i], res[i + 1])
        return res.load()
    else:
        return tanh_emulation_d3(x)


# =============================================================================
# D4 Tanh gradient — for backward pass of D3 tanh softcap
# tanh'(x) = 1 - tanh(x)^2, symmetric (even function on |x|)
# Coefficients from SPLINE_TANH_GRAD_D4: c0=1.0, c1=-0.164, c2=-0.770, c3=0.445, c4=-0.069
# Domain [0,3], 4 FMA Horner
# =============================================================================

@dsl_user_op
def tanh_grad_d4_2(
    x: Float32, y: Float32, *, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    """Paired f32x2 D4 tanh gradient — even function, no sign restore needed."""
    c0 = 1.0
    c1 = -0.163723089
    c2 = -0.769918975
    c3 = 0.444966726
    c4 = -0.068935747
    clamp = 3.0
    abs_x = fabs_f32(x, loc=loc, ip=ip)
    abs_y = fabs_f32(y, loc=loc, ip=ip)
    tx = fmin(abs_x, clamp, loc=loc, ip=ip)
    ty = fmin(abs_y, clamp, loc=loc, ip=ip)
    # Horner: ((((c4*t + c3)*t + c2)*t + c1)*t + c0
    hx, hy = fma_packed_f32x2((Float32(c4), Float32(c4)), (tx, ty), (Float32(c3), Float32(c3)), loc=loc, ip=ip)
    hx, hy = fma_packed_f32x2((tx, ty), (hx, hy), (Float32(c2), Float32(c2)), loc=loc, ip=ip)
    hx, hy = fma_packed_f32x2((tx, ty), (hx, hy), (Float32(c1), Float32(c1)), loc=loc, ip=ip)
    hx, hy = fma_packed_f32x2((tx, ty), (hx, hy), (Float32(c0), Float32(c0)), loc=loc, ip=ip)
    # Saturate: |x| >= 3 → tanh'(3) ≈ 0 (use small epsilon to keep alive)
    sat = Float32(0.0)
    return select_(abs_x < clamp, hx, sat), select_(abs_y < clamp, hy, sat)


@cute.jit
def tanh_gradf_d4(x: cute.TensorSSA | Float32) -> cute.TensorSSA | Float32:
    """D4 tanh gradient for both vector and scalar."""
    if const_expr(isinstance(x, cute.TensorSSA)):
        res = cute.make_fragment(x.shape, Float32)
        res.store(x)
        for i in cutlass.range_constexpr(0, cute.size(x.shape), 2):
            res[i], res[i + 1] = tanh_grad_d4_2(res[i], res[i + 1])
        return res.load()
    else:
        # Scalar fallback: just use 1 - tanh_d3(x)^2
        t = tanh_emulation_d3(x)
        return Float32(1.0) - t * t


# D3 softcap score_mod — lightweight, matches CUDA kernel performance
def create_softcap_scoremod_spline_d3(softcap_val):
    """Softcapping scoremod using D3 spline tanh (2 FMA+1 mul, no SFU)."""
    inv_softcap = 1.0 / softcap_val

    @cute.jit
    def scoremod_premask_fn(acc_S_SSA, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        scores = acc_S_SSA * inv_softcap
        return softcap_val * tanh_emulationf_d3(scores)

    return scoremod_premask_fn


def create_softcap_scoremod_bwd_spline_d3(softcap_val):
    """Softcapping scoremod gradient using D4 tanh gradient spline."""
    inv_softcap = 1.0 / softcap_val

    @cute.jit
    def scoremod_bwd_fn(grad, acc_S_SSA, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        scores = acc_S_SSA * inv_softcap
        # tanh'(x) directly via D4 polynomial — no need to compute tanh then square
        derivative = tanh_gradf_d4(scores)
        return grad * derivative

    return scoremod_bwd_fn


def create_softcap_scoremod_bwd_spline_d3_1mt2(softcap_val):
    """Softcapping scoremod gradient using D3 tanh + 1-tanh² (same pattern as D6)."""
    inv_softcap = 1.0 / softcap_val

    @cute.jit
    def scoremod_bwd_fn(grad, acc_S_SSA, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        scores = acc_S_SSA * inv_softcap
        t = tanh_emulationf_d3(scores)
        derivative = Float32(1.0) - t * t
        return grad * derivative

    return scoremod_bwd_fn



@dsl_user_op
def sigmoid_emulation_2(
    x: Float32, y: Float32, *, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    """Paired f32x2 spline sigmoid for ILP — processes 2 values simultaneously.

    D3 minimax polynomial on [0,6] with odd symmetry:
      h(t) = t * (c1 + t*(c2 + c3*t))
      sigmoid(x) = 0.5 + copysign(h(min(|x|,6)), x)
    """
    # D3 minimax polynomial for h(x) = sigmoid(x) - 0.5 (odd function).
    # h(x) = x * poly(|x|) — sign comes from x, like tanh (avoids broken copysign).
    c0 = 0.281005859375
    c1 = -0.0533447265625
    c2 = 0.0033893585205078125
    clamp = 6.0
    abs_x = fabs_f32(x, loc=loc, ip=ip)
    abs_y = fabs_f32(y, loc=loc, ip=ip)
    tx = fmin(abs_x, clamp, loc=loc, ip=ip)
    ty = fmin(abs_y, clamp, loc=loc, ip=ip)
    # Horner: poly(t) = (c2*t + c1)*t + c0
    px, py = fma_packed_f32x2((c2, c2), (tx, ty), (c1, c1), loc=loc, ip=ip)
    px, py = fma_packed_f32x2((px, py), (tx, ty), (c0, c0), loc=loc, ip=ip)
    # h(x) = x * poly(|x|) — sign from x, like tanh
    poly_x = Float32(0.5) + x * px
    poly_y = Float32(0.5) + y * py
    # Saturate: for |x| >= clamp, sigmoid → 0 or 1
    sat_x = select_(x > Float32(0.0), Float32(1.0), Float32(0.0))
    sat_y = select_(y > Float32(0.0), Float32(1.0), Float32(0.0))
    return select_(abs_x < clamp, poly_x, sat_x), select_(abs_y < clamp, poly_y, sat_y)


@dsl_user_op
def sigmoid_fast_2(
    x: Float32, y: Float32, *, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    """Lean paired sigmoid — 5 ops per pair.

    Uses 2-sided clamp on signed input (like SIGMOID_N2_D3_BITWISE):
      t = clamp(x, -6, 6), then sigmoid(t) ≈ c0 + t*(c1 + |t|*(c2 + c3*|t|))

    But simpler: since h(t) is odd and we need |t| for the even-degree terms,
    we use: sigmoid(x) = 0.5 + t * poly(|t|) where t = clamp(x, -6, 6).
    The clamp ensures t stays bounded, so t * poly(|t|) can't diverge.

    Ops: 2 fmax + 2 fmin + 2 fabs + 3 FMA pairs = 9 paired ops
    (fabs is free - just clears sign bit)
    """
    c1 = 0.281005859375
    c2 = -0.0533447265625
    c3 = 0.0033893585205078125
    pos_clamp = Float32(6.0)
    neg_clamp = Float32(-6.0)
    # 2-sided clamp: t = max(min(x, 6), -6)
    tx = fmax(fmin(x, pos_clamp, loc=loc, ip=ip), neg_clamp, loc=loc, ip=ip)
    ty = fmax(fmin(y, pos_clamp, loc=loc, ip=ip), neg_clamp, loc=loc, ip=ip)
    # |t| for even-degree terms
    atx = fabs_f32(tx, loc=loc, ip=ip)
    aty = fabs_f32(ty, loc=loc, ip=ip)
    # Horner on |t|: poly(|t|) = (c3*|t| + c2)*|t| + c1
    px, py = fma_packed_f32x2((c3, c3), (atx, aty), (c2, c2), loc=loc, ip=ip)
    px, py = fma_packed_f32x2((px, py), (atx, aty), (c1, c1), loc=loc, ip=ip)
    # sigmoid = 0.5 + t * poly(|t|) — t is clamped, so this is bounded
    rx, ry = fma_packed_f32x2((tx, ty), (px, py), (Float32(0.5), Float32(0.5)), loc=loc, ip=ip)
    return rx, ry


def sigmoid_native_2(x, y):
    """SFU-based sigmoid: sigmoid(x) = rcp(1 + exp2(-x * log2(e))).

    Uses hardware SFU units (exp2 + rcp_approx). Exact but causes
    SFU contention when used inside attention (competes with softmax exp2).
    """
    LOG2_E = 1.4426950408889634
    neg_x_log2e = x * Float32(-LOG2_E)
    neg_y_log2e = y * Float32(-LOG2_E)
    exp_x = cute.arch.exp2(neg_x_log2e)
    exp_y = cute.arch.exp2(neg_y_log2e)
    sx = cute.arch.rcp_approx(Float32(1.0) + exp_x)
    sy = cute.arch.rcp_approx(Float32(1.0) + exp_y)
    return sx, sy


def tanh_native_2(x, y):
    """SFU-based tanh: tanh(x) = 2*sigmoid(2x) - 1.

    Uses hardware SFU units (exp2 + rcp_approx): 2 SFU ops + 3 ALU ops per element.
    """
    LOG2_E = 1.4426950408889634
    # tanh(x) = 2*sigmoid(2x) - 1 = 2*rcp(1 + exp2(-2x*log2e)) - 1
    neg_2x_log2e = x * Float32(-2.0 * LOG2_E)
    neg_2y_log2e = y * Float32(-2.0 * LOG2_E)
    exp_x = cute.arch.exp2(neg_2x_log2e)
    exp_y = cute.arch.exp2(neg_2y_log2e)
    rx = cute.arch.rcp_approx(Float32(1.0) + exp_x)
    ry = cute.arch.rcp_approx(Float32(1.0) + exp_y)
    tx = Float32(2.0) * rx - Float32(1.0)
    ty = Float32(2.0) * ry - Float32(1.0)
    return tx, ty


@cute.jit
def tanh_emulationf_hybrid(
    x: cute.TensorSSA | Float32,
    sfu_freq: cutlass.Constexpr[int] = 4,
    sfu_res: cutlass.Constexpr[int] = 1,
) -> cute.TensorSSA | Float32:
    """Hybrid tanh: routes pairs between D3 polynomial (ALU) and SFU tanh.

    sfu_freq / sfu_res control the routing (same convention as sigmoid/exp2):
      pair_idx % sfu_freq < sfu_freq - sfu_res → ALU (D3 polynomial)
      pair_idx % sfu_freq >= sfu_freq - sfu_res → SFU (exp2 + rcp)
    Default freq=4, res=1 → 75% ALU, 25% SFU.
    """
    if const_expr(isinstance(x, cute.TensorSSA)):
        res = cute.make_fragment(x.shape, Float32)
        res.store(x)
        for i in cutlass.range_constexpr(0, cute.size(x.shape), 2):
            pair_idx = cutlass.const_expr(i // 2)
            if cutlass.const_expr(pair_idx % sfu_freq < sfu_freq - sfu_res):
                # ALU path: D3 polynomial
                res[i], res[i + 1] = tanh_emulation_d3_2(res[i], res[i + 1])
            else:
                # SFU path: exp2 + rcp hardware
                res[i], res[i + 1] = tanh_native_2(res[i], res[i + 1])
        return res.load()
    else:
        return tanh_emulation_d3(x)


# ---- Hybrid softcap score_mods ----

def create_softcap_scoremod_hybrid(softcap_val, sfu_freq=4, sfu_res=1):
    """Softcapping scoremod using hybrid ALU/SFU tanh routing."""
    inv_softcap = 1.0 / softcap_val

    @cute.jit
    def scoremod_premask_fn(acc_S_SSA, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        scores = acc_S_SSA * inv_softcap
        return softcap_val * tanh_emulationf_hybrid(scores, sfu_freq=sfu_freq, sfu_res=sfu_res)

    return scoremod_premask_fn


def create_softcap_scoremod_bwd_hybrid(softcap_val, sfu_freq=4, sfu_res=1):
    """Softcapping scoremod backward using hybrid tanh + 1-t² routing."""
    inv_softcap = 1.0 / softcap_val

    @cute.jit
    def scoremod_bwd_fn(grad, acc_S_SSA, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        scores = acc_S_SSA * inv_softcap
        t = tanh_emulationf_hybrid(scores, sfu_freq=sfu_freq, sfu_res=sfu_res)
        derivative = Float32(1.0) - t * t
        return grad * derivative

    return scoremod_bwd_fn

def create_softcap_scoremod_spline(softcap_val):
    """Softcapping scoremod using spline tanh (FMA-only, no SFU)."""
    inv_softcap = 1.0 / softcap_val

    @cute.jit
    def scoremod_premask_fn(acc_S_SSA, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        scores = acc_S_SSA * inv_softcap
        return softcap_val * tanh_emulationf(scores)

    return scoremod_premask_fn


def create_softcap_scoremod_bwd_spline(softcap_val):
    """Softcapping scoremod gradient using spline tanh."""
    inv_softcap = 1.0 / softcap_val

    @cute.jit
    def scoremod_bwd_fn(grad, acc_S_SSA, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        scores = acc_S_SSA * inv_softcap
        t = tanh_emulationf(scores)
        derivative = Float32(1.0) - t * t
        return grad * derivative

    return scoremod_bwd_fn


def convert_from_dlpack(x, leading_dim, alignment=16, divisibility=1) -> cute.Tensor:
    return (
        from_dlpack(x, assumed_align=alignment)
        .mark_layout_dynamic(leading_dim=leading_dim)
        .mark_compact_shape_dynamic(
            mode=leading_dim, stride_order=x.dim_order(), divisibility=divisibility
        )
    )


def convert_from_dlpack_leading_static(
    x, leading_dim, alignment=16, static_modes=None, stride_order=None
) -> cute.Tensor:
    if stride_order is None:
        stride_order = x.dim_order()
    x_ = from_dlpack(x, assumed_align=alignment)
    for i in range(x.ndim):
        if i != leading_dim and (static_modes is None or i not in static_modes):
            x_ = x_.mark_compact_shape_dynamic(mode=i, stride_order=stride_order)
    return x_


def make_tiled_copy_A(
    copy_atom: cute.CopyAtom, tiled_mma: cute.TiledMma, swapAB: cutlass.Constexpr[bool] = False
) -> cute.TiledCopy:
    if const_expr(swapAB):
        return cute.make_tiled_copy_B(copy_atom, tiled_mma)
    else:
        return cute.make_tiled_copy_A(copy_atom, tiled_mma)


def make_tiled_copy_B(
    copy_atom: cute.CopyAtom, tiled_mma: cute.TiledMma, swapAB: cutlass.Constexpr[bool] = False
) -> cute.TiledCopy:
    if const_expr(swapAB):
        return cute.make_tiled_copy_A(copy_atom, tiled_mma)
    else:
        return cute.make_tiled_copy_B(copy_atom, tiled_mma)


def mma_make_fragment_A(
    smem: cute.Tensor, thr_mma: cute.core.ThrMma, swapAB: cutlass.Constexpr[bool] = False
) -> cute.Tensor:
    if const_expr(swapAB):
        return mma_make_fragment_B(smem, thr_mma)
    else:
        return thr_mma.make_fragment_A(thr_mma.partition_A(smem))


def mma_make_fragment_B(
    smem: cute.Tensor, thr_mma: cute.core.ThrMma, swapAB: cutlass.Constexpr[bool] = False
) -> cute.Tensor:
    if const_expr(swapAB):
        return mma_make_fragment_A(smem, thr_mma)
    else:
        return thr_mma.make_fragment_B(thr_mma.partition_B(smem))


def get_smem_store_atom(
    arch: cutlass.Constexpr[int], element_type: Type[cute.Numeric], transpose: bool = False
) -> cute.CopyAtom:
    if const_expr(arch < 90 or element_type.width != 16):
        return cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            element_type,
            num_bits_per_copy=2 * element_type.width,
        )
    else:
        return cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=transpose, num_matrices=4),
            element_type,
        )


@cute.jit
def warp_reduce(
    val: cute.TensorSSA | cute.Numeric,
    op: Callable,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.TensorSSA | cute.Numeric:
    if const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_fragment(val.shape, val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    else:
        for i in cutlass.range_constexpr(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


def convert_layout_acc_mn(acc_layout: cute.Layout, transpose: bool = False) -> cute.Layout:
    """
    For Sm80, convert ((2, 2), MMA_M, MMA_N, ...) to ((2, MMA_M), (2, MMA_N), ...).
    For Sm90, convert ((2, 2, V), MMA_M, MMA_N, ...) to ((2, MMA_M), (2, V, MMA_N), ...).
    """
    acc_layout_col_major = cute.make_layout(acc_layout.shape)
    shape = (
        (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),  # MMA_M
        (
            acc_layout_col_major.shape[0][0],
            *acc_layout_col_major.shape[0][2:],
            acc_layout_col_major.shape[2],
        ),  # MMA_N
        *acc_layout_col_major.shape[3:],
    )
    stride = (
        (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),  # MMA_M
        (
            acc_layout_col_major.stride[0][0],
            *acc_layout_col_major.stride[0][2:],
            acc_layout_col_major.stride[2],
        ),  # MMA_N
        *acc_layout_col_major.stride[3:],
    )
    if const_expr(transpose):
        shape = (shape[1], shape[0], *shape[2:])
        stride = (stride[1], stride[0], *stride[2:])
    acc_layout_mn = cute.make_layout(shape, stride=stride)
    return cute.composition(acc_layout, acc_layout_mn)


def make_acc_tensor_mn_view(acc: cute.Tensor, transpose: bool = False) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, convert_layout_acc_mn(acc.layout, transpose=transpose))


@cute.jit
def convert_layout_acc_frgA(acc_layout: cute.Layout) -> cute.Layout:
    # For back to back gemm, convert layout of acc0 to gemm 1 accept layout.
    # For Sm80, as the mma instruction shape is 16x8x16, we need to convert from (4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
    # For Sm90, FP16/BF16, convert acc_layout from ((2, 2, N / 8), MMA_M, MMA_N) to ((2, 2, 2), MMA_M, (N / 16, MMA_N))
    # TODO: Sm90 FP8
    if const_expr(cute.rank(acc_layout.shape[0]) == 3):  # Sm90
        l = cute.logical_divide(
            acc_layout, ((None, None, 2), None, None)
        )  # ((2, 2, (2, N / 16)), MMA_M, MMA_N)
        rA_mma_view = cute.make_layout(
            (
                (l.shape[0][0], l.shape[0][1], l.shape[0][2][0]),
                l.shape[1],
                (l.shape[0][2][1], l.shape[2]),
            ),
            stride=(
                (l.stride[0][0], l.stride[0][1], l.stride[0][2][0]),
                l.stride[1],
                (l.stride[0][2][1], l.stride[2]),
            ),
        )
    else:  # Sm80
        # (4, MMA_M, MMA_N) -> (4, MMA_M, (2, MMA_N / 2))
        l = cute.logical_divide(acc_layout, (None, None, 2))
        rA_mma_view = cute.make_layout(
            (
                (l.shape[0], l.shape[2][0]),
                l.shape[1],
                l.shape[2][1],
            ),
            stride=(
                (l.stride[0], l.stride[2][0]),
                l.stride[1],
                l.stride[2][1],
            ),
        )
    return rA_mma_view


def make_acc_tensor_frgA_view(acc: cute.Tensor) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, convert_layout_acc_frgA(acc.layout))


def select(a: cute.Tensor, mode: list[int]) -> cute.Tensor:
    return cute.make_tensor(a.iterator, cute.select(a.layout, mode))


def transpose_view(a: cute.Tensor) -> cute.Tensor:
    """Transpose the first two dimensions of a tensor on smem."""
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))
    # stride = (a.layout.stride[1], a.layout.stride[0], *a.layout.stride[2:])
    # return cute.make_tensor(a.iterator, cute.make_layout(shape, stride=stride))


def parse_swizzle_from_pointer(ptr: cute.Pointer) -> cute.Swizzle:
    """Extract swizzle parameters from a pointer's swizzle_type.

    The swizzle_type string has the form '!cute.swizzle<"S<b,m,s>">' where
    b, m, s are the swizzle parameters (bits, base, shift).

    Returns:
        A cute.Swizzle object constructed from the extracted parameters

    Raises:
        ValueError: If the swizzle_type string cannot be parsed
    """
    # Ideally there should be a better API to get swizzle parameters, but we'll just parse
    # the string here.
    swizzle_str = str(ptr.type.swizzle_type)
    # Extract the inner part "S<b,m,s>"
    match = re.search(r"S<(\d+),(\d+),(\d+)>", swizzle_str)
    if match:
        b, m, s = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return cute.make_swizzle(b, m, s)
    else:
        raise ValueError(f"Could not parse swizzle_type: {swizzle_str}")


@cute.jit
def exp2f(x: cute.TensorSSA | Float32) -> cute.TensorSSA | Float32:
    """exp2f calculation for both vector and scalar.
    :param x: input value
    :type x: cute.TensorSSA or Float32
    :return: exp2 value
    :rtype: cute.TensorSSA or Float32
    """
    if const_expr(isinstance(x, cute.TensorSSA)):
        res = cute.make_fragment(x.shape, Float32)
        res.store(x)
        for i in cutlass.range_constexpr(cute.size(x.shape)):
            res[i] = cute.arch.exp2(res[i])
        return res.load()
    else:
        return cute.arch.exp2(x)


@dsl_user_op
def log2f(a: float | Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "lg2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def logf(a: float | Float32, *, loc=None, ip=None) -> Float32:
    return log2f(a, loc=loc, ip=ip) * math.log(2.0)


@dsl_user_op
def fmax(
    a: float | Float32, b: float | Float32, c: float | Float32 | None = None, *, loc=None, ip=None
) -> Float32:
    return Float32(
        nvvm.fmax(
            T.f32(),
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            c=Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def fmax_reduce(
    x: cute.TensorSSA, init_val: float | Float32 | None = None, arch: cutlass.Constexpr[int] = 80
) -> Float32:
    if const_expr(arch < 100 or cute.size(x.shape) % 8 != 0):
        # if const_expr(init_val is None):
        #     init_val = -cutlass.Float32.if
        # return x.reduce(cute.ReductionOp.MAX, init_val, 0)
        res = cute.make_fragment(x.shape, Float32)
        res.store(x)
        # local_max = [res[0], res[1]]
        # for i in cutlass.range_constexpr(2, cute.size(x.shape), 2):
        #     local_max[0] = fmax(local_max[0], res[i + 0])
        #     local_max[1] = fmax(local_max[1], res[i + 1])
        # local_max[0] = fmax(local_max[0], local_max[1])
        # return local_max[0] if const_expr(init_val is None) else fmax(local_max[0], init_val)
        local_max = [res[0], res[1], res[2], res[3]]
        for i in cutlass.range_constexpr(4, cute.size(x.shape), 4):
            local_max[0] = fmax(local_max[0], res[i + 0])
            local_max[1] = fmax(local_max[1], res[i + 1])
            local_max[2] = fmax(local_max[2], res[i + 2])
            local_max[3] = fmax(local_max[3], res[i + 3])
        local_max[0] = fmax(local_max[0], local_max[1])
        local_max[2] = fmax(local_max[2], local_max[3])
        local_max[0] = fmax(local_max[0], local_max[2])
        return local_max[0] if const_expr(init_val is None) else fmax(local_max[0], init_val)
    else:
        # [2025-06-15] x.reduce only seems to use 50% 3-input max and 50% 2-input max
        # We instead force the 3-input max.
        res = cute.make_fragment(x.shape, Float32)
        res.store(x)
        local_max_0 = (
            fmax(init_val, res[0], res[1])
            if const_expr(init_val is not None)
            else fmax(res[0], res[1])
        )
        local_max = [
            local_max_0,
            fmax(res[2], res[3]),
            fmax(res[4], res[5]),
            fmax(res[6], res[7]),
        ]
        for i in cutlass.range_constexpr(8, cute.size(x.shape), 8):
            local_max[0] = fmax(local_max[0], res[i], res[i + 1])
            local_max[1] = fmax(local_max[1], res[i + 2], res[i + 3])
            local_max[2] = fmax(local_max[2], res[i + 4], res[i + 5])
            local_max[3] = fmax(local_max[3], res[i + 6], res[i + 7])
        local_max[0] = fmax(local_max[0], local_max[1])
        return fmax(local_max[0], local_max[2], local_max[3])


@cute.jit
def fadd_reduce(
    x: cute.TensorSSA, init_val: float | Float32 | None = None, arch: cutlass.Constexpr[int] = 80
) -> Float32:
    if const_expr(arch < 100 or cute.size(x.shape) % 8 != 0):
        if const_expr(init_val is None):
            init_val = Float32.zero
        return x.reduce(cute.ReductionOp.ADD, init_val, 0)
        # res = cute.make_fragment(x.shape, Float32)
        # res.store(x)
        # local_sum = [res[0], res[1], res[2], res[3]]
        # for i in cutlass.range_constexpr(4, cute.size(x.shape), 4):
        #     local_sum[0] += res[i + 0]
        #     local_sum[1] += res[i + 1]
        #     local_sum[2] += res[i + 2]
        #     local_sum[3] += res[i + 3]
        # local_sum[0] += local_sum[1]
        # local_sum[2] += local_sum[3]
        # local_sum[0] += local_sum[2]
        # return local_sum[0] if const_expr(init_val is None) else local_sum[0] + init_val
    else:
        res = cute.make_fragment(x.shape, Float32)
        res.store(x)
        local_sum_0 = (
            add_packed_f32x2((init_val, 0.0), (res[0], res[1]))
            # add_packed_f32x2((init_val / 2, init_val / 2), (res[0], res[1]))
            if const_expr(init_val is not None)
            else (res[0], res[1])
        )
        local_sum = [local_sum_0, (res[2], res[3]), (res[4], res[5]), (res[6], res[7])]
        for i in cutlass.range_constexpr(8, cute.size(x.shape), 8):
            local_sum[0] = add_packed_f32x2(local_sum[0], (res[i + 0], res[i + 1]))
            local_sum[1] = add_packed_f32x2(local_sum[1], (res[i + 2], res[i + 3]))
            local_sum[2] = add_packed_f32x2(local_sum[2], (res[i + 4], res[i + 5]))
            local_sum[3] = add_packed_f32x2(local_sum[3], (res[i + 6], res[i + 7]))
        local_sum[0] = add_packed_f32x2(local_sum[0], local_sum[1])
        local_sum[2] = add_packed_f32x2(local_sum[2], local_sum[3])
        local_sum[0] = add_packed_f32x2(local_sum[0], local_sum[2])
        return local_sum[0][0] + local_sum[0][1]


@dsl_user_op
def atomic_add_fp32(a: float | Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    # gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    # # cache_hint = cutlass.Int64(0x12F0000000000000)
    # llvm.inline_asm(
    #     None,
    #     [gmem_ptr_i64, Float32(a).ir_value(loc=loc, ip=ip)],
    #     # [gmem_ptr_i64, Float32(a).ir_value(loc=loc, ip=ip), cache_hint.ir_value()],
    #     "red.global.add.f32 [$0], $1;",
    #     # "red.global.add.L2::cache_hint.f32 [$0], $1, 0x12F0000000000000;",
    #     # "red.global.add.L2::cache_hint.f32 [$0], $1, $2;",
    #     "l,f",
    #     # "l,f,l",
    #     has_side_effects=True,
    #     is_align_stack=False,
    #     asm_dialect=llvm.AsmDialect.AD_ATT,
    # )
    nvvm.atomicrmw(
        res=T.f32(), op=nvvm.AtomicOpKind.FADD, ptr=gmem_ptr.llvm_ptr, a=Float32(a).ir_value()
    )


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@dsl_user_op
def elem_pointer_i64(x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
    flat_coord_i64 = tuple(cutlass.Int64(c) for c in cute.flatten(coord))
    flat_stride = cute.flatten_to_tuple(x.stride)
    assert len(flat_coord_i64) == len(flat_stride), (
        "Coordinate and stride must have the same length"
    )
    offset = sum(c * s for c, s in zip(flat_coord_i64, flat_stride))
    # HACK: we assume that applying the offset does not change the pointer alignment
    byte_offset = offset * x.element_type.width // 8
    return cute.make_ptr(
        x.element_type,
        x.iterator.toint() + byte_offset,
        x.memspace,
        assumed_align=x.iterator.alignment,
    )


@cute.jit
def predicate_k(tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor:
    # Only compute predicates for the "k" dimension. For the mn dimension, we will use "if"
    tApA = cute.make_fragment(
        cute.make_layout(
            (cute.size(tAcA, mode=[0, 1]), cute.size(tAcA, mode=[1]), cute.size(tAcA, mode=[2])),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(tAcA[(0, rest_v), 0, rest_k][1], limit)
    return tApA


def canonical_warp_group_idx(sync: bool = True) -> cutlass.Int32:
    warp_group_idx = cute.arch.thread_idx()[0] // 128
    if const_expr(sync):
        warp_group_idx = cute.arch.make_warp_uniform(warp_group_idx)
    return warp_group_idx


# @dsl_user_op
# def warp_vote_any_lt(a: float | Float32, b: float | Float32, *, loc=None, ip=None) -> cutlass.Boolean:
#     mask = cutlass.Int32(-1)
#     return cutlass.Boolean(
#         llvm.inline_asm(
#             T.i32(),
#             [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip), mask.ir_value(loc=loc, ip=ip)],
#             ".pred p1, p2;\n"
#             "setp.lt.f32 p1, $1, $2;\n"
#             "vote.sync.any.pred p2, p1, $3;\n"
#             "selp.u32 $0, 1, 0, p2;",
#             # "selp.u32 $0, 1, 0, p1;",
#             "=r,f,f,r",
#             has_side_effects=False,
#             is_align_stack=False,
#             asm_dialect=llvm.AsmDialect.AD_ATT,
#         )
#     )


@cute.jit
def shuffle_sync(
    value: cute.Numeric,
    offset: cute.typing.Int,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.Numeric:
    assert value.width % 32 == 0, "value type must be a multiple of 32 bits"
    # 1 -> 0b11111, 2 -> 0b11110, 4 -> 0b11100, 8 -> 0b11000, 16 -> 0b10000, 32 -> 0b00000
    mask = cute.arch.WARP_SIZE - width
    clamp = cute.arch.WARP_SIZE - 1
    mask_and_clamp = mask << 8 | clamp
    # important: need stride 1 and not 0 for recast_tensor to work
    val = cute.make_rmem_tensor(cute.make_layout((1,), stride=(1,)), type(value))
    val[0] = value
    val_i32 = cute.recast_tensor(val, cutlass.Int32)
    for i in cutlass.range_constexpr(cute.size(val_i32)):
        val_i32[i] = cute.arch.shuffle_sync(val_i32[i], offset, mask_and_clamp=mask_and_clamp)
    return val[0]


@dsl_user_op
def shr_u32(val: cutlass.Uint32, shift: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Uint32:
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Uint32(val).ir_value(loc=loc, ip=ip),
                cutlass.Uint32(shift).ir_value(loc=loc, ip=ip),
            ],
            "shr.s32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def warp_prefix_sum(val: cutlass.Int32, lane: Optional[cutlass.Int32] = None) -> cutlass.Int32:
    if const_expr(lane is None):
        lane = cute.arch.lane_idx()
    # if cute.arch.thread_idx()[0] >= 128 and cute.arch.thread_idx()[0] < 128 + 32 and cute.arch.block_idx()[0] == 0: cute.printf("tidx = %d, val = %d", cute.arch.thread_idx()[0] % 32, val)
    for i in cutlass.range_constexpr(int(math.log2(cute.arch.WARP_SIZE))):
        offset = 1 << i
        # Very important that we set mask_and_clamp to 0
        partial_sum = cute.arch.shuffle_sync_up(val, offset=offset, mask_and_clamp=0)
        if lane >= offset:
            val += partial_sum
        # if cute.arch.thread_idx()[0] >= 128 and cute.arch.thread_idx()[0] < 128 + 32 and cute.arch.block_idx()[0] == 0: cute.printf("tidx = %d, partial_sum = %d, val = %d", cute.arch.thread_idx()[0] % 32, partial_sum, val)
    return val


@dsl_user_op
def cvt_f16x2_f32(
    a: float | Float32, b: float | Float32, to_dtype: Type, *, loc=None, ip=None
) -> cutlass.Int32:
    assert to_dtype in [cutlass.BFloat16, cutlass.Float16], "to_dtype must be BFloat16 or Float16"
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            f"cvt.rn.{'bf16x2' if to_dtype is cutlass.BFloat16 else 'f16x2'}.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@overload
def cvt_f16(src: cute.Tensor, dst: cute.Tensor) -> None: ...


@overload
def cvt_f16(src: cute.Tensor, dtype: Type[cute.Numeric]) -> cute.Tensor: ...


@cute.jit
def cvt_f16(src: cute.Tensor, dst_or_dtype):
    """Convert Float32 tensor to Float16/BFloat16.

    Args:
        src: Source tensor with Float32 element type
        dst_or_dtype: Either a destination tensor or a dtype (Float16/BFloat16)

    Returns:
        None if dst is a tensor, or a new tensor if dtype is provided
    """
    if const_expr(isinstance(dst_or_dtype, type)):
        # dtype variant: create new tensor and call the tensor variant
        dtype = dst_or_dtype
        dst = cute.make_fragment(src.shape, dtype)
        cvt_f16(src, dst)
        return dst
    else:
        # tensor variant: write to dst
        dst = dst_or_dtype
        assert cute.size(dst.shape) == cute.size(src.shape), "dst and src must have the same size"
        assert cute.size(src.shape) % 2 == 0, "src must have an even number of elements"
        assert dst.element_type in [cutlass.BFloat16, cutlass.Float16], (
            "dst must be BFloat16 or Float16"
        )
        assert src.element_type is Float32, "src must be Float32"
        dst_i32 = cute.recast_tensor(dst, cutlass.Int32)
        assert cute.size(dst_i32.shape) * 2 == cute.size(src.shape)
        for i in cutlass.range_constexpr(cute.size(dst_i32)):
            dst_i32[i] = cvt_f16x2_f32(src[2 * i], src[2 * i + 1], dst.element_type)


@dsl_user_op
@cute.jit
def evaluate_polynomial(x: Float32, poly: Tuple[Float32, ...], *, loc=None, ip=None) -> Float32:
    deg = len(poly) - 1
    out = poly[deg]
    for i in cutlass.range_constexpr(deg - 1, -1, -1):
        out = out * x + poly[i]
    return out


@dsl_user_op
@cute.jit
def evaluate_polynomial_2(
    x: Float32, y: Float32, poly: Tuple[Float32, ...], *, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    deg = len(poly) - 1
    out = (poly[deg], poly[deg])
    for i in cutlass.range_constexpr(deg - 1, -1, -1):
        out = fma_packed_f32x2(out, (x, y), (poly[i], poly[i]))
    return out


@dsl_user_op
def add_round_down(x: float | Float32, y: float | Float32, *, loc=None, ip=None) -> Float32:
    # There's probably a way to call llvm or nvvm to do this instead of ptx
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip), Float32(y).ir_value(loc=loc, ip=ip)],
            "add.rm.ftz.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def combine_int_frac_ex2(x_rounded: Float32, frac_ex2: Float32, *, loc=None, ip=None) -> Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(x_rounded).ir_value(loc=loc, ip=ip),
                Float32(frac_ex2).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t"
            ".reg .s32 x_rounded_i, frac_ex_i, x_rounded_e, out_i;\n\t"
            "mov.b32 x_rounded_i, $1;\n\t"
            "mov.b32 frac_ex_i, $2;\n\t"
            "shl.b32 x_rounded_e, x_rounded_i, 23;\n\t"
            # add.u32 generates IMAD instruction and add.s32 generates LEA instruction
            # IMAD uses the FMA pipeline and LEA uses the ALU pipeline, afaik
            "add.s32 out_i, x_rounded_e, frac_ex_i;\n\t"
            "mov.b32 $0, out_i;\n\t"
            "}\n",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def ex2_emulation(x: Float32, *, loc=None, ip=None) -> Float32:
    # We assume x <= 127.0
    poly_ex2_deg3 = (
        1.0,
        0.695146143436431884765625,
        0.227564394474029541015625,
        0.077119089663028717041015625,
    )
    fp32_round_int = float(2**23 + 2**22)
    x_clamped = cute.arch.fmax(x, -127.0)
    # We want to round down here, so that the fractional part is in [0, 1)
    x_rounded = add_round_down(x_clamped, fp32_round_int, loc=loc, ip=ip)
    # The integer floor of x is now in the last 8 bits of x_rounded
    # We assume the next 2 ops round to nearest even. The rounding mode is important.
    x_rounded_back = x_rounded - fp32_round_int
    x_frac = x_clamped - x_rounded_back
    x_frac_ex2 = evaluate_polynomial(x_frac, poly_ex2_deg3, loc=loc, ip=ip)
    return combine_int_frac_ex2(x_rounded, x_frac_ex2, loc=loc, ip=ip)


# TODO: check that the ex2_emulation_2 produces the same SASS as the ptx version
@dsl_user_op
def ex2_emulation_2(x: Float32, y: Float32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    # We assume x <= 127.0 and y <= 127.0
    poly_ex2_deg3 = (
        1.0,
        0.695146143436431884765625,
        0.227564394474029541015625,
        0.077119089663028717041015625,
    )
    fp32_round_int = float(2**23 + 2**22)
    xy_clamped = (cute.arch.fmax(x, -127.0), cute.arch.fmax(y, -127.0))
    # We want to round down here, so that the fractional part is in [0, 1)
    xy_rounded = cute.arch.add_packed_f32x2(
        xy_clamped, (fp32_round_int, fp32_round_int), rnd=nvvm.RoundingModeKind.RM
    )
    # The integer floor of x & y are now in the last 8 bits of xy_rounded
    # We want the next 2 ops to round to nearest even. The rounding mode is important.
    xy_rounded_back = sub_packed_f32x2(xy_rounded, (fp32_round_int, fp32_round_int))
    xy_frac = sub_packed_f32x2(xy_clamped, xy_rounded_back)
    xy_frac_ex2 = evaluate_polynomial_2(*xy_frac, poly_ex2_deg3, loc=loc, ip=ip)
    x_out = combine_int_frac_ex2(xy_rounded[0], xy_frac_ex2[0], loc=loc, ip=ip)
    y_out = combine_int_frac_ex2(xy_rounded[1], xy_frac_ex2[1], loc=loc, ip=ip)
    return x_out, y_out


@dsl_user_op
def ex2_emulation_2_linear(x: Float32, y: Float32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    """exp2 via 2-interval linear spline — single FMA per value.

    Same range reduction as ex2_emulation_2 but replaces the deg-3 Horner
    chain (3 dependent FMAs) with one FMA + one select per value.
    Max relative error ~1.15% vs 0.019% for deg-3.
    """
    # Linear spline coefficients for 2^frac, fitted on f32:
    #   interval [0.0, 0.5):  c1 = 0.826759108576946,  c0 = 0.988477885884221
    #   interval [0.5, 1.0):  c1 = 1.169213944165023,  c0 = 0.813311860240843
    c1_lo = 0.826759108576946
    c0_lo = 0.988477885884221
    c1_hi = 1.169213944165023
    c0_hi = 0.813311860240843

    fp32_round_int = float(2**23 + 2**22)
    xy_clamped = (cute.arch.fmax(x, -127.0), cute.arch.fmax(y, -127.0))
    xy_rounded = cute.arch.add_packed_f32x2(
        xy_clamped, (fp32_round_int, fp32_round_int), rnd=nvvm.RoundingModeKind.RM
    )
    xy_rounded_back = sub_packed_f32x2(xy_rounded, (fp32_round_int, fp32_round_int))
    xy_frac = sub_packed_f32x2(xy_clamped, xy_rounded_back)

    # Select coefficients based on interval: frac >= 0.5
    fx, fy = xy_frac
    mask_x = fx >= 0.5
    mask_y = fy >= 0.5
    c1_x = select_(mask_x, c1_hi, c1_lo)
    c0_x = select_(mask_x, c0_hi, c0_lo)
    c1_y = select_(mask_y, c1_hi, c1_lo)
    c0_y = select_(mask_y, c0_hi, c0_lo)

    # Single FMA per value: 2^frac ≈ c1*frac + c0
    frac_ex2_x = fx * c1_x + c0_x
    frac_ex2_y = fy * c1_y + c0_y

    x_out = combine_int_frac_ex2(xy_rounded[0], frac_ex2_x, loc=loc, ip=ip)
    y_out = combine_int_frac_ex2(xy_rounded[1], frac_ex2_y, loc=loc, ip=ip)
    return x_out, y_out


@dsl_user_op
def exp2f_identity_2(x: Float32, y: Float32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    return x, y


@dsl_user_op
def e2e_asm2(x: Float32, y: Float32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    out_f32x2 = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Float32(x).ir_value(loc=loc, ip=ip), Float32(y, loc=loc, ip=ip).ir_value()],
        "{\n\t"
        ".reg .f32 f1, f2, f3, f4, f5, f6, f7;\n\t"
        ".reg .b64 l1, l2, l3, l4, l5, l6, l7, l8, l9, l10;\n\t"
        ".reg .s32 r1, r2, r3, r4, r5, r6, r7, r8;\n\t"
        "max.ftz.f32 f1, $2, 0fC2FE0000;\n\t"
        "max.ftz.f32 f2, $3, 0fC2FE0000;\n\t"
        "mov.b64 l1, {f1, f2};\n\t"
        "mov.f32 f3, 0f4B400000;\n\t"
        "mov.b64 l2, {f3, f3};\n\t"
        "add.rm.ftz.f32x2 l7, l1, l2;\n\t"
        "sub.rn.ftz.f32x2 l8, l7, l2;\n\t"
        "sub.rn.ftz.f32x2 l9, l1, l8;\n\t"
        "mov.f32 f7, 0f3D9DF09D;\n\t"
        "mov.b64 l6, {f7, f7};\n\t"
        "mov.f32 f6, 0f3E6906A4;\n\t"
        "mov.b64 l5, {f6, f6};\n\t"
        "mov.f32 f5, 0f3F31F519;\n\t"
        "mov.b64 l4, {f5, f5};\n\t"
        "mov.f32 f4, 0f3F800000;\n\t"
        "mov.b64 l3, {f4, f4};\n\t"
        "fma.rn.ftz.f32x2 l10, l9, l6, l5;\n\t"
        "fma.rn.ftz.f32x2 l10, l10, l9, l4;\n\t"
        "fma.rn.ftz.f32x2 l10, l10, l9, l3;\n\t"
        "mov.b64 {r1, r2}, l7;\n\t"
        "mov.b64 {r3, r4}, l10;\n\t"
        "shl.b32 r5, r1, 23;\n\t"
        "add.s32 r7, r5, r3;\n\t"
        "shl.b32 r6, r2, 23;\n\t"
        "add.s32 r8, r6, r4;\n\t"
        "mov.b32 $0, r7;\n\t"
        "mov.b32 $1, r8;\n\t"
        "}\n",
        "=r,=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    out0 = Float32(llvm.extractvalue(T.f32(), out_f32x2, [0], loc=loc, ip=ip))
    out1 = Float32(llvm.extractvalue(T.f32(), out_f32x2, [1], loc=loc, ip=ip))
    return out0, out1


@dsl_user_op
def exp2_spline_n2_op(x: Float32, y: Float32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    # Implementation of Spline N=2 Deg=2
    # Coefficients from EVOLVED_H2_HARDCODED_2_DEG2:
    # Degree 2: 0x3BFF (0.99951171875)
    # Degree 1: 0x39CA (0.7236328125)
    # Degree 0: 0x3BFF (0.99951171875)

    # 1. Floor (Range reduction)
    fp32_round_int = float(2**23 + 2**22)
    xy_clamped = (cute.arch.fmax(x, -127.0), cute.arch.fmax(y, -127.0))
    xy_rounded = cute.arch.add_packed_f32x2(
        xy_clamped, (fp32_round_int, fp32_round_int), rnd=nvvm.RoundingModeKind.RM
    )
    xy_rounded_back = sub_packed_f32x2(xy_rounded, (fp32_round_int, fp32_round_int))
    xy_frac = sub_packed_f32x2(xy_clamped, xy_rounded_back)

    # 2. Polynomial Evaluation (N=2 Deg=2)
    # Coeffs are constant across intervals for this specific evolved function.
    c2 = 0.99951171875
    c1 = 0.7236328125
    c0 = 0.99951171875
    
    poly_coeffs = (c0, c1, c2) 
    
    xy_frac_ex2 = evaluate_polynomial_2(*xy_frac, poly_coeffs, loc=loc, ip=ip)
    
    # 3. Reconstruction
    x_out = combine_int_frac_ex2(xy_rounded[0], xy_frac_ex2[0], loc=loc, ip=ip)
    y_out = combine_int_frac_ex2(xy_rounded[1], xy_frac_ex2[1], loc=loc, ip=ip)
    return x_out, y_out


@dsl_user_op
def exp2_spline_n2_custom(x: Float32, y: Float32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    # Port of evolved_pow2_h2_hardcoded_2 (Linear Spline N=2) from C++ benchmark.
    # Coefficients:
    # Degree 1: 0x39CA (0.7236328125)
    # Degree 0: 0x3BFF (0.99951171875)

    # 1. Floor (Range reduction)
    fp32_round_int = float(2**23 + 2**22)
    xy_clamped = (cute.arch.fmax(x, -127.0), cute.arch.fmax(y, -127.0))
    xy_rounded = cute.arch.add_packed_f32x2(
        xy_clamped, (fp32_round_int, fp32_round_int), rnd=nvvm.RoundingModeKind.RM
    )
    xy_rounded_back = sub_packed_f32x2(xy_rounded, (fp32_round_int, fp32_round_int))
    xy_frac = sub_packed_f32x2(xy_clamped, xy_rounded_back)

    # 2. Interval Selection
    # Interval 0: [0, 0.5) -> xy_frac < 0.5
    # Interval 1: [0.5, 1.0) -> xy_frac >= 0.5
    
    # Unpack frac to scalar for selection
    # Using DSL indexing which should be supported for packed types
    frac_0 = xy_frac[0]
    frac_1 = xy_frac[1]
    
    # Define Coefficients (Identical for now, but distinct paths)
    c1_0 = 0.7236328125
    c0_0 = 0.99951171875
    c1_1 = 0.7236328125
    c0_1 = 0.99951171875
    
    # Select Coeffs for 0
    # Note: DSL comparison returns i1 (bool), select_ handles it.
    mask_0 = frac_0 >= 0.5
    c1_val_0 = select_(mask_0, c1_1, c1_0)
    c0_val_0 = select_(mask_0, c0_1, c0_0)
    
    # Select Coeffs for 1
    mask_1 = frac_1 >= 0.5
    c1_val_1 = select_(mask_1, c1_1, c1_0)
    c0_val_1 = select_(mask_1, c0_1, c0_0)
    
    # Scalar FMA via operators (DSL should lower to FMA)
    res_0 = frac_0 * c1_val_0 + c0_val_0
    res_1 = frac_1 * c1_val_1 + c0_val_1
    
    # 3. Reconstruction
    x_out = combine_int_frac_ex2(xy_rounded[0], res_0, loc=loc, ip=ip)
    y_out = combine_int_frac_ex2(xy_rounded[1], res_1, loc=loc, ip=ip)
    return x_out, y_out
    


@dsl_user_op
def domain_offset_aligned(
    coord: cute.Coord, tensor: cute.Tensor, *, loc=None, ip=None
) -> cute.Tensor:
    assert isinstance(tensor.iterator, cute.Pointer)
    # We assume that applying the offset does not change the pointer alignment
    new_ptr = cute.make_ptr(
        tensor.element_type,
        elem_pointer(tensor, coord).toint(),
        tensor.memspace,
        assumed_align=tensor.iterator.alignment,
    )
    return cute.make_tensor(new_ptr, tensor.layout)


@dsl_user_op
def domain_offset_i64(coord: cute.Coord, tensor: cute.Tensor, *, loc=None, ip=None) -> cute.Tensor:
    flat_coord_i64 = tuple(cutlass.Int64(c) for c in cute.flatten(coord))
    flat_stride = cute.flatten_to_tuple(tensor.stride)
    assert len(flat_coord_i64) == len(flat_stride), (
        "Coordinate and stride must have the same length"
    )
    offset = sum(c * s for c, s in zip(flat_coord_i64, flat_stride))
    assert isinstance(tensor.iterator, cute.Pointer)
    # HACK: we assume that applying the offset does not change the pointer alignment
    new_ptr = cute.make_ptr(
        tensor.element_type,
        tensor.iterator.toint() + offset * tensor.element_type.width // 8,
        tensor.memspace,
        assumed_align=tensor.iterator.max_alignment,
    )
    return cute.make_tensor(new_ptr, tensor.layout)


@dsl_user_op
def coord_offset_i64(
    tensor: cute.Tensor, idx: cute.typing.Int, dim: int, *, loc=None, ip=None
) -> cute.Tensor:
    offset = cutlass.Int64(idx) * cute.size(tensor.stride[dim])
    assert isinstance(tensor.iterator, cute.Pointer)
    # HACK: we assume that applying the offset does not change the pointer alignment
    new_ptr = cute.make_ptr(
        tensor.element_type,
        tensor.iterator.toint() + offset * tensor.element_type.width // 8,
        tensor.memspace,
        assumed_align=tensor.iterator.max_alignment,
    )
    new_layout = cute.slice_(
        tensor.layout, (*[None] * dim, 0, *[None] * (cute.rank(tensor) - dim - 1))
    )
    return cute.make_tensor(new_ptr, new_layout)


@cute.jit
def scalar_to_ssa(a: cute.Numeric, dtype) -> cute.TensorSSA:
    """Convert a scalar to a cute TensorSSA of shape (1,) and given dtype"""
    vec = cute.make_fragment(1, dtype)
    vec[0] = a
    return vec.load()


def ssa_to_scalar(val):
    """Could inline but nice for reflecting the above api"""
    return val[0]
