import importlib
import os
import sys
from functools import lru_cache
from typing import Optional, Sequence, Tuple

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False


MAX_POLY_COEFFS = 8
DEFAULT_SIGMOID_SPLINE_POLY_COEFFS = (
    0.281005859375,
    -0.0533447265625,
    0.0033893585205078125,
)


def normalize_poly_coeffs(
    coeffs: Optional[Sequence[float]], name: str = "coeffs"
) -> Optional[Tuple[float, ...]]:
    if coeffs is None:
        return None
    vals = tuple(float(c) for c in coeffs)
    if len(vals) == 0:
        raise ValueError(f"{name} must contain at least one coefficient")
    if len(vals) > MAX_POLY_COEFFS:
        raise ValueError(f"{name} length {len(vals)} exceeds MAX_POLY_COEFFS={MAX_POLY_COEFFS}")
    return vals


def derive_grad_poly_coeffs(coeffs: Tuple[float, ...]) -> Tuple[float, ...]:
    return tuple((i + 1) * coeff for i, coeff in enumerate(coeffs))


def use_spline_sigmoid_gate(
    poly_coeffs: Optional[Tuple[float, ...]],
    grad_poly_coeffs: Optional[Tuple[float, ...]],
) -> bool:
    return (
        poly_coeffs == DEFAULT_SIGMOID_SPLINE_POLY_COEFFS
        and grad_poly_coeffs is None
        and load_spline_ops() is not None
    )


@lru_cache(maxsize=1)
def load_spline_ops():
    # torch must be imported before the extension so its shared libraries are loaded.
    import torch  # noqa: F401

    candidates = [
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "autonumerics_zero",
                "spline_ops",
            )
        )
    ]
    for candidate in (None, *candidates):
        if candidate is not None and os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)
        try:
            return importlib.import_module("spline_ops")
        except ImportError:
            continue
    return None


def use_fused_output_gate(
    poly_coeffs: Optional[Tuple[float, ...]],
    grad_poly_coeffs: Optional[Tuple[float, ...]],
) -> bool:
    return poly_coeffs is None or use_spline_sigmoid_gate(poly_coeffs, grad_poly_coeffs)


def _validate_clamp(clamp: float) -> float:
    clamp = float(clamp)
    if clamp <= 0.0:
        raise ValueError("Polynomial clamp must be > 0")
    return clamp


def _pad_coeffs(coeffs: Tuple[float, ...]) -> Tuple[float, ...]:
    return coeffs + (0.0,) * (MAX_POLY_COEFFS - len(coeffs))


def _prepare_gate_tensor(gate: torch.Tensor, out_shape: Tuple[int, ...]) -> torch.Tensor:
    if tuple(gate.shape) != out_shape:
        gate = gate.expand(out_shape)
    return gate.contiguous()


def _eval_poly_horner_torch(x: torch.Tensor, coeffs: Tuple[float, ...]) -> torch.Tensor:
    out = torch.full_like(x, coeffs[-1])
    for coeff in reversed(coeffs[:-1]):
        out = out * x + coeff
    return out


def output_gate_forward_torch(
    out: torch.Tensor,
    gate: torch.Tensor,
    poly_coeffs: Tuple[float, ...],
    clamp: float,
) -> torch.Tensor:
    clamp = _validate_clamp(clamp)
    gate = _prepare_gate_tensor(gate, tuple(out.shape))
    t = gate.clamp(min=-clamp, max=clamp)
    poly = _eval_poly_horner_torch(t.abs(), poly_coeffs)
    act = 0.5 + t * poly
    return out * act.to(dtype=out.dtype)


def output_gate_forward_spline(
    out: torch.Tensor,
    gate: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    spline_ops = load_spline_ops()
    if spline_ops is None:
        raise RuntimeError("spline_ops is not available")
    gate = _prepare_gate_tensor(gate, tuple(out.shape))
    gate_act = spline_ops.sigmoid_fwd(gate)
    return out * gate_act.to(dtype=out.dtype), gate_act


def output_gate_spline_activation(gate: torch.Tensor) -> torch.Tensor:
    spline_ops = load_spline_ops()
    if spline_ops is None:
        raise RuntimeError("spline_ops is not available")
    return spline_ops.sigmoid_fwd(gate.contiguous())


def output_gate_backward_torch(
    dout: torch.Tensor,
    out: torch.Tensor,
    gate: torch.Tensor,
    poly_coeffs: Tuple[float, ...],
    grad_poly_coeffs: Optional[Tuple[float, ...]],
    clamp: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    clamp = _validate_clamp(clamp)
    if grad_poly_coeffs is None:
        grad_poly_coeffs = derive_grad_poly_coeffs(poly_coeffs)
    gate = _prepare_gate_tensor(gate, tuple(out.shape))
    t = gate.clamp(min=-clamp, max=clamp)
    abs_t = t.abs()
    act = 0.5 + t * _eval_poly_horner_torch(abs_t, poly_coeffs)
    dact = _eval_poly_horner_torch(abs_t, grad_poly_coeffs)
    dact = dact * (gate.abs() < clamp).to(dtype=gate.dtype)
    return dout * act.to(dtype=dout.dtype), (dout * out).to(dtype=gate.dtype) * dact


def output_gate_backward_spline(
    dout: torch.Tensor,
    out: torch.Tensor,
    gate_act: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    spline_ops = load_spline_ops()
    if spline_ops is None:
        raise RuntimeError("spline_ops is not available")
    gate_act = gate_act.contiguous()
    dout_attn = dout * gate_act.to(dtype=dout.dtype)
    grad_gate = spline_ops.sigmoid_bwd_alg((dout * out).to(dtype=gate_act.dtype).contiguous(), gate_act)
    return dout_attn, grad_gate


if HAS_TRITON:
    @triton.jit
    def _eval_poly_fma(
        x,
        c0,
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
        num_coeffs: tl.constexpr,
    ):
        if num_coeffs == 1:
            return x * 0.0 + c0
        if num_coeffs == 2:
            return tl.fma(x, c1, c0)
        if num_coeffs == 3:
            p = tl.fma(x, c2, c1)
            return tl.fma(x, p, c0)
        if num_coeffs == 4:
            p = tl.fma(x, c3, c2)
            p = tl.fma(x, p, c1)
            return tl.fma(x, p, c0)
        if num_coeffs == 5:
            p = tl.fma(x, c4, c3)
            p = tl.fma(x, p, c2)
            p = tl.fma(x, p, c1)
            return tl.fma(x, p, c0)
        if num_coeffs == 6:
            p = tl.fma(x, c5, c4)
            p = tl.fma(x, p, c3)
            p = tl.fma(x, p, c2)
            p = tl.fma(x, p, c1)
            return tl.fma(x, p, c0)
        if num_coeffs == 7:
            p = tl.fma(x, c6, c5)
            p = tl.fma(x, p, c4)
            p = tl.fma(x, p, c3)
            p = tl.fma(x, p, c2)
            p = tl.fma(x, p, c1)
            return tl.fma(x, p, c0)
        p = tl.fma(x, c7, c6)
        p = tl.fma(x, p, c5)
        p = tl.fma(x, p, c4)
        p = tl.fma(x, p, c3)
        p = tl.fma(x, p, c2)
        p = tl.fma(x, p, c1)
        return tl.fma(x, p, c0)


    @triton.jit
    def _output_gate_forward_poly_kernel(
        out_ptr,
        gate_ptr,
        out_gated_ptr,
        n_elements,
        clamp,
        c0,
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
        num_coeffs: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        out = tl.load(out_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        t = tl.maximum(tl.minimum(gate, clamp), -clamp)
        poly = _eval_poly_fma(tl.abs(t), c0, c1, c2, c3, c4, c5, c6, c7, num_coeffs=num_coeffs)
        act = tl.fma(t, poly, 0.5)
        tl.store(out_gated_ptr + offs, out * act, mask=mask)


    @triton.jit
    def _output_gate_backward_poly_kernel(
        dout_ptr,
        out_ptr,
        gate_ptr,
        dout_attn_ptr,
        dgate_ptr,
        n_elements,
        clamp,
        act_c0,
        act_c1,
        act_c2,
        act_c3,
        act_c4,
        act_c5,
        act_c6,
        act_c7,
        grad_c0,
        grad_c1,
        grad_c2,
        grad_c3,
        grad_c4,
        grad_c5,
        grad_c6,
        grad_c7,
        num_act_coeffs: tl.constexpr,
        num_grad_coeffs: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        dout = tl.load(dout_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        out = tl.load(out_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        t = tl.maximum(tl.minimum(gate, clamp), -clamp)
        abs_t = tl.abs(t)
        act_poly = _eval_poly_fma(
            abs_t,
            act_c0,
            act_c1,
            act_c2,
            act_c3,
            act_c4,
            act_c5,
            act_c6,
            act_c7,
            num_coeffs=num_act_coeffs,
        )
        act = tl.fma(t, act_poly, 0.5)
        grad_poly = _eval_poly_fma(
            abs_t,
            grad_c0,
            grad_c1,
            grad_c2,
            grad_c3,
            grad_c4,
            grad_c5,
            grad_c6,
            grad_c7,
            num_coeffs=num_grad_coeffs,
        )
        dact = tl.where(tl.abs(gate) < clamp, grad_poly, 0.0)
        tl.store(dout_attn_ptr + offs, dout * act, mask=mask)
        tl.store(dgate_ptr + offs, dout * out * dact, mask=mask)


def output_gate_forward(
    out: torch.Tensor,
    gate: torch.Tensor,
    poly_coeffs: Tuple[float, ...],
    clamp: float,
) -> torch.Tensor:
    clamp = _validate_clamp(clamp)
    gate = _prepare_gate_tensor(gate, tuple(out.shape))
    if (not HAS_TRITON) or (not out.is_cuda):
        return output_gate_forward_torch(out, gate, poly_coeffs, clamp)

    coeffs = _pad_coeffs(poly_coeffs)
    out_gated = torch.empty_like(out)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]),)
    _output_gate_forward_poly_kernel[grid](
        out.contiguous().view(-1),
        gate.view(-1),
        out_gated.view(-1),
        n_elements,
        float(clamp),
        *coeffs,
        num_coeffs=len(poly_coeffs),
        BLOCK=1024,
        num_warps=4,
    )
    return out_gated


def output_gate_backward(
    dout: torch.Tensor,
    out: torch.Tensor,
    gate: torch.Tensor,
    poly_coeffs: Tuple[float, ...],
    grad_poly_coeffs: Optional[Tuple[float, ...]],
    clamp: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    clamp = _validate_clamp(clamp)
    if grad_poly_coeffs is None:
        grad_poly_coeffs = derive_grad_poly_coeffs(poly_coeffs)
    gate = _prepare_gate_tensor(gate, tuple(out.shape))
    if (not HAS_TRITON) or (not dout.is_cuda):
        return output_gate_backward_torch(dout, out, gate, poly_coeffs, grad_poly_coeffs, clamp)

    act_coeffs = _pad_coeffs(poly_coeffs)
    grad_coeffs = _pad_coeffs(grad_poly_coeffs)
    dout_attn = torch.empty_like(dout)
    dgate = torch.empty_like(gate)
    n_elements = dout.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]),)
    _output_gate_backward_poly_kernel[grid](
        dout.contiguous().view(-1),
        out.contiguous().view(-1),
        gate.view(-1),
        dout_attn.view(-1),
        dgate.view(-1),
        n_elements,
        float(clamp),
        *act_coeffs,
        *grad_coeffs,
        num_act_coeffs=len(poly_coeffs),
        num_grad_coeffs=len(grad_poly_coeffs),
        BLOCK=1024,
        num_warps=4,
    )
    return dout_attn, dgate
