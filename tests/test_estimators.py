import numpy as np
import pytest

from clawdia import estimators


def test_elementwise_error_metrics_have_known_values():
    x = np.array([1.0, 2.0, 3.0])
    y = np.ones(3)

    assert estimators.mse(x, y) == pytest.approx(5 / 3)
    assert estimators.medse(x, y) == pytest.approx(1.0)
    assert estimators.residual(x, y) == pytest.approx(np.sqrt(5))


def test_structural_similarity_identity_symmetry_and_derived_metrics():
    x = np.linspace(-0.5, 0.5, 32)
    y = x**3

    assert estimators.ssim(x, x) == pytest.approx(1.0, abs=1e-12)
    assert estimators.ssim(x, y) == pytest.approx(
        estimators.ssim(y, x), abs=1e-12
    )
    assert estimators.dssim(x, y) == pytest.approx(
        (1 - estimators.ssim(x, y)) / 2, abs=1e-12
    )
    assert estimators.issim(x, y) == pytest.approx(
        -estimators.ssim(x, y), abs=1e-12
    )


def test_softmax_is_stable_and_respects_axis():
    values = np.array([[1000.0, 1001.0], [-1000.0, -999.0]])
    probabilities = estimators.softmax(values, axis=1)
    expected_row = np.array([1, np.e]) / (1 + np.e)

    np.testing.assert_allclose(
        probabilities,
        np.vstack([expected_row, expected_row]),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        probabilities.sum(axis=1), 1.0, rtol=1e-12, atol=1e-12
    )

