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
    source: str = "current"
    family: str = ""


@dataclass(frozen=True)
class ComposedPolynomialSpec:
    name: str
    target: str
    composed_from: str
    backend_targets: tuple[str, ...]
    notes: str = ""
    source: str = "current"
    degree: int | None = None


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
SOLLYA_STRUCT_HEADER = Path(__file__).resolve().parents[3] / "autonumerics_zero" / "spline_ops" / "spline_structs_sollya_bf16.cuh"
SOLLYA_SWEEP_JSON = (
    Path(__file__).resolve().parents[3]
    / "autonumerics_zero"
    / "cuda_benchmarks"
    / "analysis_results"
    / "sollya_device_bf16.json"
)

_SWEEP_META = {
    "tanh_fwd": {
        "name_prefix": "softcap_tanh",
        "target": "tanh(x)",
        "symmetry": "odd-factorized",
        "backend_targets": ("cute", "device"),
        "notes": "BF16 softcap tanh sweep row.",
    },
    "sigmoid_fwd": {
        "name_prefix": "sigmoid",
        "target": "sigmoid(x)",
        "symmetry": "odd-centered",
        "backend_targets": ("cute", "device", "output_gate"),
        "notes": "BF16 sigmoid sweep row.",
    },
    "sigmoid_bwd": {
        "name_prefix": "sigmoid_grad",
        "target": "sigmoid'(x)",
        "symmetry": "even",
        "backend_targets": ("cute", "device"),
        "notes": "BF16 direct sigmoid-gradient sweep row.",
    },
    "swish_bwd": {
        "name_prefix": "swish_grad",
        "target": "swish'(x)",
        "symmetry": "odd-centered",
        "backend_targets": ("spline_ops",),
        "notes": "BF16 swish-gradient sweep row.",
    },
    "gelu_fwd": {
        "name_prefix": "gelu_fwd",
        "target": "gelu(x)",
        "symmetry": "odd-centered",
        "backend_targets": ("spline_ops",),
        "notes": "BF16 GeLU forward sweep row.",
    },
    "gelu_bwd": {
        "name_prefix": "gelu_bwd",
        "target": "gelu'(x)",
        "symmetry": "odd-centered",
        "backend_targets": ("spline_ops",),
        "notes": "BF16 GeLU backward sweep row.",
    },
}


def _load_sollya_sweep_data() -> dict[str, object]:
    if not SOLLYA_SWEEP_JSON.is_file():
        raise FileNotFoundError(
            f"Missing Sollya sweep JSON at {SOLLYA_SWEEP_JSON}. "
            "Run generate_sollya_structs_bf16.py first."
        )
    import json

    return json.loads(SOLLYA_SWEEP_JSON.read_text())


def _row_for_family_degree(family: str, degree: int) -> dict[str, object]:
    data = _load_sollya_sweep_data()
    degree_key = f"D{degree}"
    try:
        return data["families"][family][degree_key]
    except KeyError as exc:
        raise ValueError(f"Unsupported family/degree pair: {family} D{degree}") from exc


def _generic_spec_from_sweep(
    family: str,
    degree: int,
    coeff_source: str = "current",
) -> PolynomialSpec:
    if coeff_source not in ("current", "sollya"):
        raise ValueError(f"Unsupported coeff_source={coeff_source!r}")
    meta = _SWEEP_META[family]
    row = _row_for_family_degree(family, degree)
    coeff_key = "current_coeffs" if coeff_source == "current" else "sollya_coeffs"
    objective = (
        "current_bf16_runtime"
        if coeff_source == "current"
        else "sollya_fpminimax_bf16_runtime"
    )
    return PolynomialSpec(
        name=f"{meta['name_prefix']}_d{degree}_{coeff_source}",
        target=meta["target"],
        symmetry=meta["symmetry"],
        degree=degree,
        fit_interval=float(row["clamp"]),
        clamp=float(row["clamp"]),
        objective=objective,
        coeffs=tuple(float(coeff) for coeff in row[coeff_key]),
        backend_targets=tuple(meta["backend_targets"]),
        notes=meta["notes"],
        source=coeff_source,
        family=family,
    )


def _swish_forward_composed_spec(
    degree: int,
    coeff_source: str = "current",
) -> ComposedPolynomialSpec:
    sigmoid_row = _row_for_family_degree("sigmoid_fwd", degree)
    struct_key = "current_struct" if coeff_source == "current" else "sollya_struct"
    return ComposedPolynomialSpec(
        name=f"swish_d{degree}_composed_{coeff_source}",
        target="swish(x)",
        composed_from=str(sigmoid_row[struct_key]),
        backend_targets=("spline_ops", "device"),
        notes="Swish forward composes x with the matching sigmoid sweep row.",
        source=coeff_source,
        degree=degree,
    )


def get_softcap_tanh_forward_spec(
    degree: int = 4,
    backend: str = "device",
    coeff_source: str = "current",
) -> PolynomialSpec:
    spec = (
        SOFTCAP_TANH_D4
        if degree == 4 and coeff_source == "current"
        else _generic_spec_from_sweep("tanh_fwd", degree, coeff_source=coeff_source)
    )
    if backend not in spec.backend_targets:
        raise ValueError(f"Unsupported backend '{backend}' for softcap D{degree} ({coeff_source})")
    return spec


def get_softcap_tanh_analytical_backward_coeffs(
    degree: int = 4,
    backend: str = "device",
    coeff_source: str = "current",
) -> tuple[float, ...]:
    spec = get_softcap_tanh_forward_spec(
        degree=degree,
        backend=backend,
        coeff_source=coeff_source,
    )
    return tuple((i + 1) * coeff for i, coeff in enumerate(spec.coeffs))


def get_sigmoid_forward_spec(
    degree: int = 3,
    coeff_source: str = "current",
) -> PolynomialSpec:
    if degree == 3 and coeff_source == "current":
        return SIGMOID_D3
    return _generic_spec_from_sweep("sigmoid_fwd", degree, coeff_source=coeff_source)


def get_sigmoid_gradient_spec(
    dtype: str = "bf16",
    degree: int = 5,
    coeff_source: str = "current",
) -> PolynomialSpec:
    if dtype != "bf16":
        raise ValueError(f"Only BF16 sigmoid-gradient coefficients are canonicalized, got {dtype!r}")
    if degree == 5 and coeff_source == "current":
        return SIGMOID_GRAD_D5_BF16
    return _generic_spec_from_sweep("sigmoid_bwd", degree, coeff_source=coeff_source)


def get_default_output_gate_coeffs(
    coeff_source: str = "current",
    degree: int = 3,
) -> tuple[float, ...]:
    return get_sigmoid_forward_spec(degree=degree, coeff_source=coeff_source).coeffs


def get_default_swish_specs() -> tuple[ComposedPolynomialSpec, PolynomialSpec]:
    return SWISH_D3_COMPOSED, SWISH_GRAD_D4_BF16


def get_default_gelu_specs() -> tuple[PolynomialSpec, PolynomialSpec]:
    return GELU_FWD_D5_BF16, GELU_BWD_D5_BF16


def get_swish_forward_spec(
    degree: int = 3,
    coeff_source: str = "current",
) -> ComposedPolynomialSpec:
    if degree == 3 and coeff_source == "current":
        return SWISH_D3_COMPOSED
    return _swish_forward_composed_spec(degree, coeff_source=coeff_source)


def get_swish_backward_spec(
    degree: int = 4,
    coeff_source: str = "current",
) -> PolynomialSpec:
    if degree == 4 and coeff_source == "current":
        return SWISH_GRAD_D4_BF16
    return _generic_spec_from_sweep("swish_bwd", degree, coeff_source=coeff_source)


def get_gelu_forward_spec(
    degree: int = 5,
    coeff_source: str = "current",
) -> PolynomialSpec:
    if degree == 5 and coeff_source == "current":
        return GELU_FWD_D5_BF16
    return _generic_spec_from_sweep("gelu_fwd", degree, coeff_source=coeff_source)


def get_gelu_backward_spec(
    degree: int = 5,
    coeff_source: str = "current",
) -> PolynomialSpec:
    if degree == 5 and coeff_source == "current":
        return GELU_BWD_D5_BF16
    return _generic_spec_from_sweep("gelu_bwd", degree, coeff_source=coeff_source)


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


def audit_polynomial_selection(selections: Iterable[tuple[str, int, str]]) -> tuple[str, ...]:
    current_header_text = STRUCT_HEADER.read_text()
    sollya_header_text = SOLLYA_STRUCT_HEADER.read_text()
    sweep_data = _load_sollya_sweep_data()
    errors: list[str] = []
    audited: list[str] = []
    for family, degree, coeff_source in selections:
        row = sweep_data["families"][family][f"D{degree}"]
        header_text = current_header_text if coeff_source == "current" else sollya_header_text
        struct_key = "current_struct" if coeff_source == "current" else "sollya_struct"
        struct_name = row[struct_key]
        if family == "swish_fwd":
            _check_composed_struct(
                errors,
                header_text,
                struct_name,
                get_swish_forward_spec(degree=degree, coeff_source=coeff_source),
            )
            audited.append(f"swish_fwd_d{degree}_{coeff_source}")
            continue
        spec = {
            "tanh_fwd": get_softcap_tanh_forward_spec,
            "sigmoid_fwd": get_sigmoid_forward_spec,
            "sigmoid_bwd": get_sigmoid_gradient_spec,
            "swish_bwd": get_swish_backward_spec,
            "gelu_fwd": get_gelu_forward_spec,
            "gelu_bwd": get_gelu_backward_spec,
        }[family](degree=degree, coeff_source=coeff_source)
        _check_struct_against_spec(errors, header_text, struct_name, spec)
        audited.append(spec.name)
    if errors:
        joined = "\n".join(f"- {err}" for err in errors)
        raise RuntimeError(f"Polynomial selection audit failed:\n{joined}")
    return tuple(audited)


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
