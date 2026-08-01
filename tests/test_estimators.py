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


@pytest.fixture
def bin_centred_sinusoid():
    n_samples = 128
    sample_rate = 128.0
    amplitude = 0.25
    time = np.arange(n_samples) / sample_rate
    signal = amplitude * np.sin(2 * np.pi * 8 * time)
    return signal, sample_rate, amplitude


def test_weighted_inner_product_matches_bin_centred_fft_result(
    bin_centred_sinusoid,
):
    signal, sample_rate, amplitude = bin_centred_sinusoid
    dt = 1 / sample_rate
    expected = sample_rate * len(signal) * amplitude**2

    actual = estimators.inner_product_weighted(
        signal, signal, at=dt, window="boxcar"
    )
    assert actual == pytest.approx(expected, rel=5e-12, abs=5e-12)

    frequencies = np.fft.rfftfreq(len(signal), dt)
    flat_psd = np.vstack([frequencies, np.full_like(frequencies, 2.0)])
    weighted = estimators.inner_product_weighted(
        signal, signal, at=dt, psd=flat_psd, window="boxcar"
    )
    assert weighted == pytest.approx(expected / 2, rel=5e-12, abs=5e-12)


def test_overlap_has_expected_scale_sign_and_orthogonality(
    bin_centred_sinusoid,
):
    signal, sample_rate, _ = bin_centred_sinusoid
    dt = 1 / sample_rate
    time = np.arange(len(signal)) / sample_rate
    orthogonal = np.sin(2 * np.pi * 9 * time)

    assert estimators.overlap(
        signal, signal, at=dt, window="boxcar"
    ) == pytest.approx(1.0, abs=5e-12)
    assert estimators.overlap(
        signal, -signal, at=dt, window="boxcar"
    ) == pytest.approx(-1.0, abs=5e-12)
    assert estimators.overlap(
        signal, orthogonal, at=dt, window="boxcar"
    ) == pytest.approx(0.0, abs=5e-12)
    assert estimators.doverlap(
        signal, signal, at=dt, window="boxcar"
    ) == pytest.approx(0.0, abs=5e-12)




def test_match_zero_signal_is_zero():
    zeros = np.zeros(32)
    assert estimators.match(zeros, zeros, window="boxcar") == 0.0


def test_find_merger_returns_largest_absolute_sample():
    assert estimators.find_merger(np.array([1.0, -4.0, 3.0])) == 1
