from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
from typing import Iterable


@dataclass(frozen=True)
class PolynomialSpec:
    name: str
    target: str
    symmetry: str
    degree: int
    fit_interval: float
    clamp: float
    objective: str
    coeffs: tuple[float, ...]
    backend_targets: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class ComposedPolynomialSpec:
    name: str
    target: str
    composed_from: str
    backend_targets: tuple[str, ...]
    notes: str = ""


SOFTCAP_TANH_D4 = PolynomialSpec(
    name="softcap_tanh_d4",
    target="tanh(x)",
    symmetry="odd-factorized",
    degree=4,
    fit_interval=2.50,
    clamp=2.75,
    objective="constrained_least_squares_bf16_runtime",
    coeffs=(
        1.0859375000,
        -0.3222656250,
        -0.0230712891,
        0.0164794922,
    ),
    backend_targets=("cute", "device"),
    notes="Canonical D4 softcap fit used for the longer-run FA4 port.",
)

SIGMOID_D3 = PolynomialSpec(
    name="sigmoid_d3",
    target="sigmoid(x)",
    symmetry="odd-centered",
    degree=3,
    fit_interval=6.00,
    clamp=6.00,
    objective="constrained_least_squares_bf16_runtime",
    coeffs=(
        0.281005859375,
        -0.0533447265625,
        0.0033893585205078125,
    ),
    backend_targets=("cute", "device", "output_gate"),
    notes="Shared D3 sigmoid fit for FA4 sigmoid attention, output gate, and device-side activations.",
)

SIGMOID_GRAD_D5_BF16 = PolynomialSpec(
    name="sigmoid_grad_d5_bf16",
    target="sigmoid'(x)",
    symmetry="even",
    degree=5,
    fit_interval=7.50,
    clamp=4.75,
    objective="constrained_least_squares_bf16_runtime",
    coeffs=(
        0.2500000000,
        -0.0046386719,
        -0.0722656250,
        0.0257568359,
        -0.0034027100,
        0.0001573563,
    ),
    backend_targets=("cute", "device"),
    notes="Direct polynomial sigmoid-gradient fit kept for debug-only backward comparisons.",
)

SWISH_D3_COMPOSED = ComposedPolynomialSpec(
    name="swish_d3_composed",
    target="swish(x)",
    composed_from="SIGMOID_FWD_D3_ODD_BF16",
    backend_targets=("spline_ops", "device"),
    notes="Default BF16 swish forward path composes x with the canonical D3 sigmoid fit.",
)

SWISH_GRAD_D4_BF16 = PolynomialSpec(
    name="swish_grad_d4_bf16",
    target="swish'(x)",
    symmetry="odd-centered",
    degree=4,
    fit_interval=7.25,
    clamp=5.25,
    objective="constrained_least_squares_bf16_runtime",
    coeffs=(
        0.6093750000,
        -0.2128906250,
        0.0296630859,
        -0.0014572144,
    ),
    backend_targets=("spline_ops",),
    notes="Default BF16 swish backward path used by spline_ops.",
)

GELU_FWD_D5_BF16 = PolynomialSpec(
    name="gelu_fwd_d5_bf16",
    target="gelu(x)",
    symmetry="odd-centered",
    degree=5,
    fit_interval=3.0,
    clamp=3.0,
    objective="fp16_seeded_bf16_runtime",
    coeffs=(
        0.3962402344,
        0.0178222656,
        -0.1052856445,
        0.0362548828,
        -0.0038852692,
    ),
    backend_targets=("spline_ops",),
    notes="Default BF16 GeLU forward fit used by spline_ops and spline_compile.",
)

GELU_BWD_D5_BF16 = PolynomialSpec(
    name="gelu_bwd_d5_bf16",
    target="gelu'(x)",
    symmetry="odd-centered",
    degree=5,
    fit_interval=3.75,
    clamp=3.0,
    objective="fp16_seeded_bf16_runtime",
    coeffs=(
        0.8706054688,
        -0.2055664062,
        -0.1563720703,
        0.0755004883,
        -0.0088424683,
    ),
    backend_targets=("spline_ops",),
    notes="Default BF16 GeLU backward fit used by spline_ops and spline_compile.",
)

ACTIVE_POLYNOMIALS: tuple[PolynomialSpec, ...] = (
    SOFTCAP_TANH_D4,
    SIGMOID_D3,
    SIGMOID_GRAD_D5_BF16,
    SWISH_GRAD_D4_BF16,
    GELU_FWD_D5_BF16,
    GELU_BWD_D5_BF16,
)
ACTIVE_COMPOSED_POLYNOMIALS: tuple[ComposedPolynomialSpec, ...] = (SWISH_D3_COMPOSED,)

STRUCT_HEADER = Path(__file__).resolve().parents[3] / "autonumerics_zero" / "spline_ops" / "spline_structs_odd_bf16.cuh"


def get_softcap_tanh_forward_spec(degree: int = 4, backend: str = "device") -> PolynomialSpec:
    if degree != 4:
        raise ValueError(f"Only D4 is canonicalized for the current softcap port, got D{degree}")
    if backend not in SOFTCAP_TANH_D4.backend_targets:
        raise ValueError(f"Unsupported backend '{backend}' for canonical softcap D4")
    return SOFTCAP_TANH_D4


def get_softcap_tanh_analytical_backward_coeffs(
    degree: int = 4,
    backend: str = "device",
) -> tuple[float, ...]:
    spec = get_softcap_tanh_forward_spec(degree=degree, backend=backend)
    return tuple((i + 1) * coeff for i, coeff in enumerate(spec.coeffs))


def get_sigmoid_forward_spec() -> PolynomialSpec:
    return SIGMOID_D3


def get_sigmoid_gradient_spec(dtype: str = "bf16") -> PolynomialSpec:
    if dtype != "bf16":
        raise ValueError(f"Only BF16 sigmoid-gradient coefficients are canonicalized, got {dtype!r}")
    return SIGMOID_GRAD_D5_BF16


def get_default_output_gate_coeffs() -> tuple[float, ...]:
    return SIGMOID_D3.coeffs


def get_default_swish_specs() -> tuple[ComposedPolynomialSpec, PolynomialSpec]:
    return SWISH_D3_COMPOSED, SWISH_GRAD_D4_BF16


def get_default_gelu_specs() -> tuple[PolynomialSpec, PolynomialSpec]:
    return GELU_FWD_D5_BF16, GELU_BWD_D5_BF16


def coeffs_to_cli_arg(coeffs: Iterable[float]) -> str:
    return ",".join(f"{coeff:.17g}" for coeff in coeffs)


def evaluate_odd_factorized_forward(x: float, coeffs: tuple[float, ...], clamp: float) -> float:
    clamped_x = min(max(x, -clamp), clamp)
    t = abs(clamped_x)
    poly = 0.0
    for coeff in reversed(coeffs):
        poly = poly * t + coeff
    y = t * poly
    y = min(y, 1.0)
    return math.copysign(y, clamped_x)


def evaluate_odd_factorized_derivative(x: float, coeffs: tuple[float, ...], clamp: float) -> float:
    if abs(x) >= clamp:
        return 0.0
    t = abs(x)
    deriv_coeffs = tuple((i + 1) * coeff for i, coeff in enumerate(coeffs))
    poly = 0.0
    for coeff in reversed(deriv_coeffs):
        poly = poly * t + coeff
    return poly


def evaluate_centered_sigmoid_forward(x: float, coeffs: tuple[float, ...], clamp: float) -> float:
    t = max(min(x, clamp), -clamp)
    poly = 0.0
    for coeff in reversed(coeffs):
        poly = poly * abs(t) + coeff
    return 0.5 + t * poly


def evaluate_even_polynomial(x: float, coeffs: tuple[float, ...], clamp: float) -> float:
    t = min(abs(x), clamp)
    poly = 0.0
    for coeff in reversed(coeffs):
        poly = poly * t + coeff
    return poly


def evaluate_sigmoid_backward_from_probability(
    x: float,
    forward_coeffs: tuple[float, ...] = SIGMOID_D3.coeffs,
    forward_clamp: float = SIGMOID_D3.clamp,
) -> float:
    p = evaluate_centered_sigmoid_forward(x, forward_coeffs, forward_clamp)
    return p * (1.0 - p)


def evaluate_sigmoid_backward_poly(
    x: float,
    coeffs: tuple[float, ...] = SIGMOID_GRAD_D5_BF16.coeffs,
    clamp: float = SIGMOID_GRAD_D5_BF16.clamp,
) -> float:
    return evaluate_even_polynomial(x, coeffs, clamp)


def _extract_bf16_struct(header_text: str, struct_name: str) -> tuple[float, tuple[float, ...]]:
    struct_re = re.compile(rf"struct {struct_name} \{{(.*?)\n\}};", re.S)
    match = struct_re.search(header_text)
    if match is None:
        raise ValueError(f"Could not find struct {struct_name}")
    body = match.group(1)
    clamp_match = re.search(r"__hmin2\(abs_val, __float2bfloat162_rn\(([-+0-9.eE]+)f\)\)", body)
    if clamp_match is None:
        raise ValueError(f"Could not find clamp for {struct_name}")
    coeff_matches = re.findall(r"c(\d+)\s*=\s*__float2bfloat162_rn\(([-+0-9.eE]+)f\);", body)
    if not coeff_matches:
        raise ValueError(f"Could not find coefficients for {struct_name}")
    coeffs = tuple(float(val) for _, val in sorted(((int(idx), val) for idx, val in coeff_matches)))
    return float(clamp_match.group(1)), coeffs


def _check_struct_against_spec(errors: list[str], header_text: str, struct_name: str, spec: PolynomialSpec) -> None:
    clamp, coeffs = _extract_bf16_struct(header_text, struct_name)
    if not math.isclose(clamp, spec.clamp, rel_tol=0.0, abs_tol=1e-9):
        errors.append(f"{struct_name}: clamp {clamp} != manifest {spec.clamp}")
    if len(coeffs) != len(spec.coeffs):
        errors.append(f"{struct_name}: coeff length {len(coeffs)} != manifest {len(spec.coeffs)}")
        return
    for idx, (got, want) in enumerate(zip(coeffs, spec.coeffs)):
        if not math.isclose(got, want, rel_tol=0.0, abs_tol=1e-9):
            errors.append(f"{struct_name}: c{idx} {got} != manifest {want}")


def _check_composed_struct(errors: list[str], header_text: str, struct_name: str, spec: ComposedPolynomialSpec) -> None:
    struct_re = re.compile(rf"struct {struct_name} \{{(.*?)\n\}};", re.S)
    match = struct_re.search(header_text)
    if match is None:
        errors.append(f"Could not find composed struct {struct_name}")
        return
    body = match.group(1)
    if spec.composed_from not in body:
        errors.append(f"{struct_name}: expected composed source {spec.composed_from}")


def audit_active_polynomials() -> list[str]:
    header_text = STRUCT_HEADER.read_text()
    errors: list[str] = []

    checks = (
        ("TANH_FWD_D4_ODD_BF16", SOFTCAP_TANH_D4),
        ("SIGMOID_FWD_D3_ODD_BF16", SIGMOID_D3),
        ("SIGMOID_BWD_D5_EVEN_BF16", SIGMOID_GRAD_D5_BF16),
        ("SWISH_BWD_D4_ODD_BF16", SWISH_GRAD_D4_BF16),
        ("GELU_FWD_D5_ODD_BF16", GELU_FWD_D5_BF16),
        ("GELU_BWD_D5_ODD_BF16", GELU_BWD_D5_BF16),
    )

    for struct_name, spec in checks:
        _check_struct_against_spec(errors, header_text, struct_name, spec)
    for struct_name, spec in (("SWISH_FWD_D3_ODD_BF16", SWISH_D3_COMPOSED),):
        _check_composed_struct(errors, header_text, struct_name, spec)
    return errors


def assert_active_polynomials_in_sync() -> None:
    errors = audit_active_polynomials()
    if errors:
        joined = "\n".join(f"- {err}" for err in errors)
        raise RuntimeError(f"Active polynomial coefficient audit failed:\n{joined}")


def run_polynomial_coefficient_audit() -> tuple[str, ...]:
    assert_active_polynomials_in_sync()
    return tuple(spec.name for spec in (*ACTIVE_POLYNOMIALS, *ACTIVE_COMPOSED_POLYNOMIALS))
