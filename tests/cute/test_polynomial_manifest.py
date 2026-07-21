import math
import struct
from pathlib import Path

import pytest

import flash_attn.cute.polynomial_manifest as polynomial_manifest
from flash_attn.cute.handwritten_spline_ptx import (
    audit_handwritten_fast_path,
    get_handwritten_inline_asm,
)
from flash_attn.cute.polynomial_manifest import (
    GELU_BWD_D5_BF16,
    GELU_FWD_D5_BF16,
    SOFTCAP_TANH_D4,
    SIGMOID_D3,
    SWISH_D3_COMPOSED,
    SWISH_GRAD_D4_BF16,
    audit_polynomial_selection,
    audit_active_polynomials,
    assert_active_polynomials_in_sync,
    evaluate_centered_sigmoid_forward,
    evaluate_flash_sigmoid_direct_d3,
    evaluate_flash_sigmoid_direct_d4_gradient,
    evaluate_flash_sigmoid_exp2_poly,
    evaluate_sigmoid_attention_tail_safe,
    evaluate_odd_factorized_derivative,
    evaluate_odd_factorized_forward,
    evaluate_sigmoid_backward_from_probability,
    evaluate_sigmoid_backward_poly,
    get_default_output_gate_coeffs,
    get_default_gelu_specs,
    get_default_swish_specs,
    get_gelu_forward_spec,
    get_sigmoid_forward_spec,
    get_sigmoid_gradient_spec,
    get_softcap_tanh_forward_spec,
    get_swish_forward_spec,
    get_softcap_tanh_analytical_backward_coeffs,
    run_polynomial_coefficient_audit,
)


def test_active_polynomials_match_bf16_header():
    assert_active_polynomials_in_sync()


def test_current_bf16_swish_fused_form_matches_sigmoid_coefficients():
    errors = audit_active_polynomials()
    assert not errors


def test_output_gate_defaults_share_sigmoid_manifest():
    assert get_default_output_gate_coeffs() == SIGMOID_D3.coeffs


def test_polynomial_coefficient_audit_reports_all_active_defaults():
    assert run_polynomial_coefficient_audit() == (
        "softcap_tanh_d4",
        "sigmoid_d3",
        "sigmoid_grad_d5_bf16",
        "swish_grad_d4_bf16",
        "gelu_fwd_d5_bf16",
        "gelu_bwd_d5_bf16",
        "swish_d3_composed",
    )


def test_exp2_pwl2_hinge_keeps_compact_f16x2_instruction_shape():
    symbol = "fa4_exp2_fractional_pwl2_hinge_f16x2"
    asm = get_handwritten_inline_asm(symbol)

    assert asm.count("fma.rn.f16x2") == 2
    assert "max.f16x2" in asm
    assert not any(op in asm for op in ("ex2.", "rcp.", "lg2.", "tanh"))
    assert audit_handwritten_fast_path(
        symbol,
        max_instructions=11,
        max_packed_fma=2,
    ) == f"{symbol}[11 PTX instructions,2 packed FMA]"


def test_exp2_pwl1_safe_keeps_single_fma_f16x2_instruction_shape():
    symbol = "fa4_exp2_fractional_pwl1_safe_f16x2"
    asm = get_handwritten_inline_asm(symbol)

    assert asm.count("fma.rn.f16x2") == 1
    assert "0x3c003c00" in asm
    assert not any(op in asm for op in ("max.", "ex2.", "rcp.", "lg2.", "tanh"))
    assert audit_handwritten_fast_path(
        symbol,
        max_instructions=4,
        max_packed_fma=1,
    ) == f"{symbol}[4 PTX instructions,1 packed FMA]"


def test_exp2_pwl2_safe_keeps_compact_f16x2_instruction_shape():
    symbol = "fa4_exp2_fractional_pwl2_safe_f16x2"
    asm = get_handwritten_inline_asm(symbol)

    assert asm.count("fma.rn.f16x2") == 2
    assert "max.f16x2" in asm
    assert "0x3a623a62" in asm
    assert "0x36763676" in asm
    assert "0x3c003c00" in asm
    assert not any(op in asm for op in ("ex2.", "rcp.", "lg2.", "tanh"))
    assert audit_handwritten_fast_path(
        symbol,
        max_instructions=11,
        max_packed_fma=2,
    ) == f"{symbol}[11 PTX instructions,2 packed FMA]"


def test_exp2_pwl2_safe_keeps_compact_bf16x2_instruction_shape():
    symbol = "fa4_exp2_fractional_pwl2_safe_bf16x2"
    asm = get_handwritten_inline_asm(symbol)

    assert asm.count("fma.rn.bf16x2") == 2
    assert "max.bf16x2" in asm
    assert "0x3f4c3f4c" in asm
    assert "0x3ecf3ecf" in asm
    assert "0x3f803f80" in asm
    assert not any(op in asm for op in ("ex2.", "rcp.", "lg2.", "tanh"))
    assert audit_handwritten_fast_path(
        symbol,
        max_instructions=11,
        max_packed_fma=2,
    ) == f"{symbol}[11 PTX instructions,2 packed FMA]"


def test_exp2_d2_safe_keeps_compact_f16x2_instruction_shape():
    symbol = "fa4_exp2_fractional_d2_safe_f16x2"
    asm = get_handwritten_inline_asm(symbol)

    assert asm.count("fma.rn.f16x2") == 2
    assert "0x35703570" in asm
    assert "0x39483948" in asm
    assert "0x3c003c00" in asm
    assert "max.f16x2" not in asm
    assert not any(op in asm for op in ("ex2.", "rcp.", "lg2.", "tanh"))
    assert audit_handwritten_fast_path(
        symbol,
        max_instructions=7,
        max_packed_fma=2,
    ) == f"{symbol}[7 PTX instructions,2 packed FMA]"


def test_exp2_d2_safe_keeps_compact_bf16x2_instruction_shape():
    symbol = "fa4_exp2_fractional_d2_safe_bf16x2"
    asm = get_handwritten_inline_asm(symbol)

    assert asm.count("fma.rn.bf16x2") == 2
    assert "0x3eae3eae" in asm
    assert "0x3f293f29" in asm
    assert "0x3f803f80" in asm
    assert not any(op in asm for op in ("ex2.", "rcp.", "lg2.", "tanh"))
    assert audit_handwritten_fast_path(
        symbol,
        max_instructions=7,
        max_packed_fma=2,
    ) == f"{symbol}[7 PTX instructions,2 packed FMA]"


def test_source_aware_lookup_defaults_and_sollya_rows():
    assert get_softcap_tanh_forward_spec() == SOFTCAP_TANH_D4
    assert get_sigmoid_forward_spec() == SIGMOID_D3
    assert get_sigmoid_gradient_spec() == get_sigmoid_gradient_spec(
        degree=5,
        coeff_source="current",
    )

    softcap_sollya = get_softcap_tanh_forward_spec(degree=6, coeff_source="sollya")
    sigmoid_sollya = get_sigmoid_forward_spec(degree=4, coeff_source="sollya")
    gelu_sollya = get_gelu_forward_spec(degree=3, coeff_source="sollya")
    swish_sollya = get_swish_forward_spec(degree=5, coeff_source="sollya")

    assert softcap_sollya.source == "sollya"
    assert sigmoid_sollya.source == "sollya"
    assert gelu_sollya.source == "sollya"
    assert swish_sollya.source == "sollya"
    assert softcap_sollya.degree == 6
    assert sigmoid_sollya.degree == 4
    assert gelu_sollya.degree == 3
    assert swish_sollya.degree == 5


def test_selection_audit_checks_current_and_sollya_headers():
    audited = audit_polynomial_selection(
        (
            ("tanh_fwd", 4, "current"),
            ("sigmoid_fwd", 3, "current"),
            ("sigmoid_fwd", 4, "sollya"),
            ("sigmoid_bwd", 5, "sollya"),
            ("swish_fwd", 6, "sollya"),
            ("gelu_fwd", 3, "sollya"),
        )
    )
    assert audited == (
        "softcap_tanh_d4",
        "sigmoid_d3",
        "sigmoid_d4_sollya",
        "sigmoid_grad_d5_sollya",
        "swish_fwd_d6_sollya",
        "gelu_fwd_d3_sollya",
    )


def test_selection_audit_current_defaults_does_not_require_sweep_json(tmp_path, monkeypatch):
    missing_sweep = tmp_path / "missing_sollya_device_bf16.json"
    monkeypatch.setattr(polynomial_manifest, "SOLLYA_SWEEP_JSON", missing_sweep)

    audited = audit_polynomial_selection(
        (
            ("tanh_fwd", 4, "current"),
            ("sigmoid_fwd", 3, "current"),
            ("sigmoid_bwd", 3, "current"),
        )
    )
    assert audited == ("softcap_tanh_d4", "sigmoid_d3", "sigmoid_grad_d3_current")


def test_selection_audit_recognizes_bias_aware_flash_sigmoid_exp2_d3():
    assert audit_polynomial_selection((("flash_sigmoid_exp2", 3, "current"),)) == (
        "flash_sigmoid_exp2_d3_current",
    )


def test_selection_audit_recognizes_bias_aware_flash_sigmoid_exp2_d2():
    assert audit_polynomial_selection((("flash_sigmoid_exp2", 2, "current"),)) == (
        "flash_sigmoid_exp2_d2_current",
    )


def test_flash_sigmoid_exp2_d2_manifest_matches_device_ptx():
    utils_source = Path(polynomial_manifest.__file__).with_name("utils.py").read_text()
    for coefficient in polynomial_manifest.FLASH_SIGMOID_EXP2_D2_COEFFS:
        bits = struct.unpack("<I", struct.pack("<f", coefficient))[0]
        assert f"0f{bits:08X}" in utils_source


def test_default_swish_and_gelu_specs_match_manifest():
    swish_fwd, swish_bwd = get_default_swish_specs()
    gelu_fwd, gelu_bwd = get_default_gelu_specs()
    assert swish_fwd == SWISH_D3_COMPOSED
    assert swish_bwd == SWISH_GRAD_D4_BF16
    assert gelu_fwd == GELU_FWD_D5_BF16
    assert gelu_bwd == GELU_BWD_D5_BF16


def test_softcap_d4_analytical_backward_matches_finite_difference():
    eps = 1e-5
    for x in (-2.0, -1.0, -0.25, 0.25, 1.0, 2.0):
        fd = (
            evaluate_odd_factorized_forward(x + eps, SOFTCAP_TANH_D4.coeffs, SOFTCAP_TANH_D4.clamp)
            - evaluate_odd_factorized_forward(x - eps, SOFTCAP_TANH_D4.coeffs, SOFTCAP_TANH_D4.clamp)
        ) / (2 * eps)
        analytic = evaluate_odd_factorized_derivative(x, SOFTCAP_TANH_D4.coeffs, SOFTCAP_TANH_D4.clamp)
        assert math.isclose(fd, analytic, rel_tol=0.0, abs_tol=5e-4)


def test_softcap_d4_analytical_backward_zeroes_outside_clamp():
    coeffs = get_softcap_tanh_analytical_backward_coeffs()
    assert coeffs == tuple((i + 1) * coeff for i, coeff in enumerate(SOFTCAP_TANH_D4.coeffs))
    assert evaluate_odd_factorized_derivative(10.0, SOFTCAP_TANH_D4.coeffs, SOFTCAP_TANH_D4.clamp) == 0.0


def test_sigmoid_backward_from_probability_matches_explicit_p_times_one_minus_p():
    for x in (-8.0, -3.0, -0.5, 0.0, 0.5, 3.0, 8.0):
        p = evaluate_centered_sigmoid_forward(x, SIGMOID_D3.coeffs, SIGMOID_D3.clamp)
        grad = evaluate_sigmoid_backward_from_probability(x)
        assert math.isclose(grad, p * (1.0 - p), rel_tol=0.0, abs_tol=1e-12)
        assert 0.0 <= grad <= 0.25


def test_sigmoid_attention_tail_safe_preserves_flashsigmoid_negative_tail():
    sequence_length = 4096
    bias = -math.log(sequence_length)
    for qk_score in (-3.0, -1.0, 0.0, 1.0, 3.0):
        x = qk_score + bias
        expected = 1.0 / (1.0 + math.exp(-x))
        actual = evaluate_sigmoid_attention_tail_safe(x)
        assert actual > 0.0
        assert math.isclose(actual, expected, rel_tol=0.015, abs_tol=2e-6)


def test_sigmoid_attention_tail_safe_is_continuous_at_core_boundary():
    eps = 1e-6
    for boundary in (-2.75, 2.75):
        below = evaluate_sigmoid_attention_tail_safe(boundary - eps)
        above = evaluate_sigmoid_attention_tail_safe(boundary + eps)
        assert abs(above - below) < 0.01


def test_flash_sigmoid_bias_aware_exp2_d3_matches_score_distribution():
    sequence_length = 4096
    scores = tuple(-6.0 + 0.025 * index for index in range(481))
    weights = tuple(math.exp(-0.5 * score * score) for score in scores)
    expected = tuple(
        1.0 / (1.0 + math.exp(-(score - math.log(sequence_length))))
        for score in scores
    )
    actual = tuple(
        evaluate_flash_sigmoid_exp2_poly(score, sequence_length, degree=3)
        for score in scores
    )
    relative_l1 = sum(
        weight * abs(got - want)
        for weight, got, want in zip(weights, actual, expected)
    ) / sum(weight * want for weight, want in zip(weights, expected))
    gradient_relative_l1 = sum(
        weight * abs(got * (1.0 - got) - want * (1.0 - want))
        for weight, got, want in zip(weights, actual, expected)
    ) / sum(weight * want * (1.0 - want) for weight, want in zip(weights, expected))
    assert relative_l1 < 0.0001
    assert gradient_relative_l1 < 0.0001


def test_flash_sigmoid_bias_aware_exp2_d2_matches_score_distribution():
    sequence_length = 4096
    scores = tuple(-6.0 + 0.025 * index for index in range(481))
    weights = tuple(math.exp(-0.5 * score * score) for score in scores)
    expected = tuple(
        1.0 / (1.0 + math.exp(-(score - math.log(sequence_length))))
        for score in scores
    )
    actual = tuple(
        evaluate_flash_sigmoid_exp2_poly(score, sequence_length, degree=2)
        for score in scores
    )
    relative_l1 = sum(
        weight * abs(got - want)
        for weight, got, want in zip(weights, actual, expected)
    ) / sum(weight * want for weight, want in zip(weights, expected))
    gradient_relative_l1 = sum(
        weight * abs(got * (1.0 - got) - want * (1.0 - want))
        for weight, got, want in zip(weights, actual, expected)
    ) / sum(weight * want * (1.0 - want) for weight, want in zip(weights, expected))
    assert relative_l1 < 0.0014
    assert gradient_relative_l1 < 0.0014


def test_flash_sigmoid_direct_d3_d4_matches_score_distribution():
    sequence_length = polynomial_manifest.FLASH_SIGMOID_DIRECT_SEQUENCE_LENGTH
    scores = tuple(-6.0 + 0.025 * index for index in range(481))
    weights = tuple(math.exp(-0.5 * score * score) for score in scores)
    expected = tuple(
        1.0 / (1.0 + math.exp(-(score - math.log(sequence_length))))
        for score in scores
    )
    expected_gradient = tuple(value * (1.0 - value) for value in expected)
    actual = tuple(evaluate_flash_sigmoid_direct_d3(score) for score in scores)
    actual_gradient = tuple(
        evaluate_flash_sigmoid_direct_d4_gradient(score) for score in scores
    )

    relative_l1 = sum(
        weight * abs(got - want)
        for weight, got, want in zip(weights, actual, expected)
    ) / sum(weight * want for weight, want in zip(weights, expected))
    gradient_relative_l1 = sum(
        weight * abs(got - want)
        for weight, got, want in zip(weights, actual_gradient, expected_gradient)
    ) / sum(
        weight * want for weight, want in zip(weights, expected_gradient)
    )

    assert relative_l1 < 0.014
    assert gradient_relative_l1 < 0.014
    assert min(actual) >= 0.0
    assert min(actual_gradient) >= 0.0


def test_flash_sigmoid_direct_fit_rejects_other_sequence_lengths():
    with pytest.raises(ValueError, match="only supports sequence length 4096"):
        evaluate_flash_sigmoid_direct_d3(0.0, sequence_length=2048)


def test_sigmoid_poly_backward_stays_close_to_algebraic_reference_in_core_region():
    xs = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    for x in xs:
        alg = evaluate_sigmoid_backward_from_probability(x)
        poly = evaluate_sigmoid_backward_poly(x)
        assert abs(alg - poly) < 0.02
