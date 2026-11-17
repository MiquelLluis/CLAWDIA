import numpy as np
import pytest

from clawdia import estimators

# Conservative roundoff tolerance for short float64 FFT-based cases.
# This is a practical test tolerance, not a general error bound.
FFT_TOL = 5e-12


@pytest.fixture
def bin_centred_8hz_sinusoid():
    """Centered sinusoidal tone of 8 Hz with sampling rate of 128 Hz."""
    n_samples = 128
    sample_rate = 128.0
    amplitude = 0.25
    time = np.arange(n_samples) / sample_rate
    signal = amplitude * np.sin(2 * np.pi * 8 * time)
    return signal, sample_rate, amplitude


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


class TestInnerProductWeighted:
    def test_matches_bin_centred_fft_result(self, bin_centred_8hz_sinusoid):
        """Check the raw-FFT self-inner-product with unit and constant PSDs."""
        signal, sample_rate, amplitude = bin_centred_8hz_sinusoid
        dt = 1 / sample_rate

        # Let N be the sample count, A the amplitude, and fs the sample rate.
        # This bin-centred sine contains whole cycles, so
        #   sum(x**2) = N/2 * A**2.
        #
        # Parseval's identity for NumPy's unscaled forward FFT gives
        #   sum(x**2) = 1/N * sum(|X[k]|**2),
        # where X is the full FFT.
        #
        # Only the positive/negative tone bins are nonzero, with equal magnitude:
        #   N/2 * A**2 = 1/N * 2*|X[k0]|**2   =>   |X[k0]|**2 = N**2/4 * A**2.
        #
        # The rFFT retains just the positive tone bin. With unit PSD, the
        # implemented self-inner-product is
        #   <X|X> = 4*df * |X[k0]|**2.
        # Since df = fs/N, this becomes
        #   <X|X> = fs * N * A**2.
        expected = sample_rate * len(signal) * amplitude**2

        # A boxcar preserves the tone; omitting the PSD gives unit weighting.
        actual = estimators.inner_product_weighted(
            signal, signal, at=dt, window="boxcar"
        )
        assert actual == pytest.approx(expected, rel=FFT_TOL, abs=FFT_TOL)

        # A full-band PSD of 2 retains all bins and halves the inner product.
        frequencies = np.fft.rfftfreq(len(signal), dt)
        flat_psd = np.vstack([frequencies, np.full_like(frequencies, 2.0)])
        weighted = estimators.inner_product_weighted(
            signal, signal, at=dt, psd=flat_psd, window="boxcar"
        )
        assert weighted == pytest.approx(expected / 2, rel=FFT_TOL, abs=FFT_TOL)


    @pytest.mark.parametrize(
        "x, y",
        [
            (np.ones(4), np.ones(5)),
            (np.ones(4, dtype=complex), np.ones(4)),
        ],
    )
    def test_rejects_invalid_inputs(self, x, y):
        """Reject unequal lengths and complex inputs."""
        with pytest.raises(ValueError):
            estimators.inner_product_weighted(x, y, at=1, window="boxcar")


def test_overlap_has_expected_scale_sign_and_orthogonality(
        bin_centred_8hz_sinusoid):
    signal, sample_rate, _ = bin_centred_8hz_sinusoid
    dt = 1 / sample_rate
    time = np.arange(len(signal)) / sample_rate
    orthogonal = np.sin(2 * np.pi * 9 * time)

    assert estimators.overlap(
        signal, signal, at=dt, window="boxcar"
    ) == pytest.approx(1.0, rel=0, abs=FFT_TOL)
    assert estimators.overlap(
        signal, -signal, at=dt, window="boxcar"
    ) == pytest.approx(-1.0, rel=0, abs=FFT_TOL)
    assert estimators.overlap(
        signal, orthogonal, at=dt, window="boxcar"
    ) == pytest.approx(0.0, rel=0, abs=FFT_TOL)
    assert estimators.doverlap(
        signal, signal, at=dt, window="boxcar"
    ) == pytest.approx(0.0, rel=0, abs=FFT_TOL)


class TestMatch:
    """Check the match AND the corresponding imatch loss."""

    def test_recovers_cyclic_lag_and_imatch_preserves_it(self):
        """Recover a seven-sample delay and preserve it in the loss wrapper.

        The pulse [1, 0, -1] has 0 Hz and Nyquist components, keeping this
        check independent of the unresolved weighting of those special bins.
        With a 0.25-second sample interval, the aligning lag is -1.75 seconds.
        """
        signal = np.zeros(64)
        signal[10] = 1.0
        signal[12] = -1.0
        shifted = np.roll(signal, 7)  # cyclic lag in samples

        match, lag = estimators.match(
            signal, shifted, at=0.25, window="boxcar", return_lag=True
        )
        mismatch, mismatch_lag = estimators.imatch(
            signal, shifted, at=0.25, window="boxcar", return_lag=True
        )

        assert match == pytest.approx(1.0, rel=0, abs=FFT_TOL)
        assert lag == (-7, -1.75)
        assert mismatch == pytest.approx(0.0, rel=0, abs=FFT_TOL)
        assert mismatch_lag == lag

    @pytest.mark.parametrize(
        "phase",
        [
            pytest.param(np.pi / 2, id="quadrature"),
            pytest.param(0.37, id="nonquadrature"),
        ],
    )
    def test_single_tone_is_invariant_to_phase(self, phase):
        """Recognise real sinusoids different only by a global phase.

        Ten samples per period make a 90-degree rotation a 2.5-sample shift,
        so an integer-lag search alone cannot align it exactly. A second phase
        also checks that maximisation is not restricted to quadrature rotations.
        """
        theta = 2 * np.pi * np.arange(10) / 10
        signal = np.sin(theta)
        rotated = np.sin(theta + phase)

        match = estimators.match(signal, rotated, window="boxcar")

        assert match == pytest.approx(1.0, rel=0, abs=FFT_TOL)

    @pytest.mark.parametrize("n_samples", [127, 128], ids=["odd", "even"])
    def test_global_phase_and_cyclic_lag_with_weighted_tones(self, n_samples):
        """Recover a delayed, phase-rotated two-tone signal under PSD weighting.

        Both tones receive the same phase rotation, not the frequency-dependent
        rotation of a time shift. Their bins (3 and 8) have a difference coprime
        to either signal length, giving a unique aligning integer lag of -7.
        The PSD gives the tones different positive weights, but they still
        represent the same waveform up to phase, delay, and amplitude scaling.
        Both odd and even lengths yield real-FFT spectrums.
        """
        dt = 1 / n_samples
        theta = 2 * np.pi * np.arange(n_samples) / n_samples
        phase = 0.7
        signal = np.sin(3 * theta) + 0.5 * np.sin(8 * theta)
        rotated = np.sin(3 * theta + phase) + 0.5 * np.sin(8 * theta + phase)
        shifted = 2.5 * np.roll(rotated, 7)
        psd = np.array([[1.0, 10.0], [2.0, 11.0]])

        match, lag = estimators.match(
            signal, shifted, at=dt, psd=psd, window="boxcar", return_lag=True
        )
        mismatch, mismatch_lag = estimators.imatch(
            signal, shifted, at=dt, psd=psd, window="boxcar", return_lag=True
        )

        assert match == pytest.approx(1.0, rel=0, abs=FFT_TOL)
        assert lag[0] == -7
        assert lag[1] == pytest.approx(-7 * dt, rel=0, abs=1e-12)
        assert mismatch == pytest.approx(0.0, rel=0, abs=FFT_TOL)
        assert mismatch_lag == lag

    @pytest.mark.parametrize("phase", [0.0, 0.7], ids=["aligned", "rotated"])
    def test_phase_invariance_does_not_hide_missing_frequency_content(self, phase):
        """Preserve the known imperfect match when one of two tones is missing.

        Equal-amplitude tones at 3 and 8 Hz have weighted powers proportional
        to 1/4 and 1/9 under this PSD (when linearly interpolated).
        A template containing only the 3 Hz tone must therefore match by
            sqrt((1/4) / (1/4 + 1/9)) = sqrt(9/13),
        whatever its global phase is. This also checks normalisation after phase
        maximisation: removing phase disagreement must not hide any missing
        spectral content.
        """
        theta = 2 * np.pi * np.arange(128) / 128
        signal = np.sin(3 * theta) + np.sin(8 * theta)
        template = np.sin(3 * theta + phase)
        psd = np.array(
            [ 
                [1.0, 10.0],  # frequencies
                [2.0, 11.0]   # PSD values
            ]
        )  # lin. interp:  S(3) = 4,  S(8) = 9

        match = estimators.match(
            signal, template, at=1 / 128, psd=psd, window="boxcar"
        )

        assert match == pytest.approx(np.sqrt(9 / 13), rel=0, abs=FFT_TOL)

    def test_zero_signal_is_zero(self):
        zeros = np.zeros(32)
        assert estimators.match(zeros, zeros, window="boxcar") == 0.0

    @pytest.mark.parametrize(
        "x, y",
        [
            pytest.param(np.ones(4), np.ones(5), id="unequal-lengths"),
            pytest.param(np.ones(4, dtype=complex), np.ones(4), id="complex-input"),
        ],
    )
    def test_rejects_invalid_inputs(self, x, y):
        """Reject unequal lengths and complex waveforms under the real-input API."""
        with pytest.raises(ValueError):
            estimators.match(x, y, at=1, window="boxcar")


def test_snr_matches_known_bin_centred_signal(bin_centred_8hz_sinusoid):
    signal, sample_rate, amplitude = bin_centred_8hz_sinusoid
    dt = 1 / sample_rate
    frequencies = np.fft.rfftfreq(len(signal), dt)
    psd_level = 2.0
    psd = np.vstack([frequencies, np.full_like(frequencies, psd_level)])
    duration = len(signal) / sample_rate
    expected = amplitude * np.sqrt(duration / psd_level)

    actual = estimators.snr(
        signal, psd=psd, at=dt, window=np.ones(len(signal))
    )
    assert actual == pytest.approx(expected, rel=FFT_TOL, abs=FFT_TOL)


def test_find_merger_returns_largest_absolute_sample():
    assert estimators.find_merger(np.array([1.0, -4.0, 3.0])) == 1
