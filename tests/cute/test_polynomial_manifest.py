import math

from flash_attn.cute.polynomial_manifest import (
    GELU_BWD_D5_BF16,
    GELU_FWD_D5_BF16,
    SOFTCAP_TANH_D4,
    SIGMOID_D3,
    SWISH_D3_COMPOSED,
    SWISH_GRAD_D4_BF16,
    assert_active_polynomials_in_sync,
    evaluate_centered_sigmoid_forward,
    evaluate_odd_factorized_derivative,
    evaluate_odd_factorized_forward,
    evaluate_sigmoid_backward_from_probability,
    evaluate_sigmoid_backward_poly,
    get_default_output_gate_coeffs,
    get_default_gelu_specs,
    get_default_swish_specs,
    get_softcap_tanh_analytical_backward_coeffs,
    run_polynomial_coefficient_audit,
)


def test_active_polynomials_match_bf16_header():
    assert_active_polynomials_in_sync()


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


def test_sigmoid_poly_backward_stays_close_to_algebraic_reference_in_core_region():
    xs = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    for x in xs:
        alg = evaluate_sigmoid_backward_from_probability(x)
        poly = evaluate_sigmoid_backward_poly(x)
        assert abs(alg - poly) < 0.02
