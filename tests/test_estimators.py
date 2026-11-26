import numpy as np
import pytest
from scipy.signal.windows import get_window

from clawdia import estimators

# Roughly 45 float64 epsilons for the small direct calculations.
FLOAT64_RTOL = 1e-14

# The observed relative error and absolute error from zero were below 1e-15
# in all current FFT-related tests. This tolerance value is well below the
# error from a missing or misweighted tone.
FFT_TOL = 1e-13


#------------------------------------------------------------------------------
# FIXTURES
#------------------------------------------------------------------------------

@pytest.fixture
def bin_centred_8hz_sinusoid():
    """Centered sinusoidal tone of 8 Hz with 128 Hz of sampling rate."""
    n_samples = 128
    sample_rate = 128.0
    amplitude = 0.25
    time = np.arange(n_samples) / sample_rate
    signal = amplitude * np.sin(2 * np.pi * 8 * time)
    return signal, sample_rate, amplitude


@pytest.fixture(params=["constant", "nyquist"])
def excluded_component(request):
    """A 32-sample signal with power only at 0 Hz or Nyquist.

    - constant: [1.0,  1.0, 1.0,  1.0, ...]
    - nyquist:  [1.0, -1.0, 1.0, -1.0, ...]
    """
    if request.param == "constant":
        return np.ones(32)
    return (-1.0) ** np.arange(32)


@pytest.fixture
def contaminated_signals(excluded_component):
    """Two 32 Hz signals containing the same 8 Hz tone at amplitudes 1 and 2.

    All combinations, accounting for both from `excluded_component`:
    - x constant: [ 8,  9,  8,  7, ...]
    - x nyquist:  [ 8, -7,  8, -9, ...]
    - y constant: [-3, -1, -3, -5, ...]
    - y nyquist:  [-3,  5, -3,  1, ...]
    """
    tone = np.tile([0.0, 1.0, 0.0, -1.0], 8)  # 32 samples
    x = tone + 8 * excluded_component
    y = 2 * tone - 3 * excluded_component
    return x, y


@pytest.fixture
def flat_noise_psd():
    """A flat PSD exceeding the Nyquist limit for the 32 Hz test signals."""
    return np.array([[0.0, 24.0], [2.0, 2.0]])


@pytest.fixture(params=[9, 10], ids=["odd", "even"])
def highest_bin_tone(request):
    """The highest tone below Nyquist, alone and mixed with an equal-power tone.

    The sampling rate is determined from the signal's length; the duration of
    the signal is always 1 second.
    """
    n_samples = request.param
    dt = 1 / n_samples
    amplitude = 0.25
    theta = 2 * np.pi * np.arange(n_samples) / n_samples  # [0, ..., 2π)
    highest_bin = (n_samples - 1) // 2
    tone = amplitude * np.sin(highest_bin * theta)
    mixture = tone + amplitude * np.sin(theta)
    # With 1 Hz bin spacing, end one bin past the highest usable tone so
    # the exclusive upper index retains it for both signal lengths.
    psd = np.array([[0.0, highest_bin + 1], [2.0, 2.0]])
    return tone, mixture, dt, amplitude, psd


@pytest.fixture(
    params=[(0.0, 0.5), (20.0, 24.0)],
    ids=["zero-frequency-only", "above-nyquist"],
)
def empty_band_psd(request):
    """A PSD band containing no usable bins for 32 samples at 32 Hz."""
    return np.array([request.param, [2.0, 2.0]])


#------------------------------------------------------------------------------
# TESTS
#------------------------------------------------------------------------------


def test_elementwise_error_metrics_have_known_values():
    x = np.array([1.0, 2.0, 3.0])
    y = np.ones(3)

    # The squared errors are [0, 1, 4]. Their median is exactly representable.
    assert estimators.mse(x, y) == pytest.approx(5/3, rel=FLOAT64_RTOL, abs=0)
    assert estimators.medse(x, y) == 1.0
    assert estimators.residual_l2_norm(x, y) == pytest.approx(
        np.sqrt(5), rel=FLOAT64_RTOL, abs=0
    )


def test_structural_similarity_has_known_value_and_is_symmetric():
    x = np.array([-0.02, 0.0, 0.02])
    y = np.array([0.04, 0.01, 0.01])

    # By hand, the means are 0 and 0.02, the population variances are
    # 0.0008/3 and 0.0002, and the covariance is -0.0002.
    # With C1 = 0.0001 and C2 = 0.0009 for unit data range, SSIM is
    # (0.0001 / 0.0005) * (0.0005 / (0.0008/3 + 0.0002 + 0.0009))
    # = (1/5) * (15/41) = 3/41.
    expected = 3 / 41

    assert estimators.ssim(x, x) == pytest.approx(1.0, rel=FLOAT64_RTOL, abs=0)
    assert estimators.ssim(x, y) == pytest.approx(expected, rel=FLOAT64_RTOL, abs=0)
    assert estimators.ssim(y, x) == pytest.approx(expected, rel=FLOAT64_RTOL, abs=0)
    assert estimators.dssim(x, y) == pytest.approx(
        (1 - expected) / 2, rel=FLOAT64_RTOL, abs=0
    )
    assert estimators.issim(x, y) == pytest.approx(-expected, rel=FLOAT64_RTOL, abs=0)


class TestInnerProductWeighted:
    def test_matches_bin_centred_fft_result(self, bin_centred_8hz_sinusoid):
        """Check the physical self-inner-product with unit and constant PSDs."""
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
        # The rFFT retains just the positive tone bin. With unit PSD:
        #   (x|x) = 4*df * dt**2 * |X[k0]|**2.
        # Since df = fs/N and dt = 1/fs, this becomes
        #   (x|x) = (N/fs) * A**2 = duration * A**2.
        expected = len(signal) / sample_rate * amplitude**2

        # A boxcar preserves the tone; omitting the PSD gives unit weighting.
        actual = estimators.inner_product_weighted(
            signal, signal, at=dt, window="boxcar"
        )
        assert actual == pytest.approx(expected, rel=FFT_TOL, abs=0)

        # A flat PSD of 2 retains the tone and halves the inner product.
        flat_psd = np.array([[0.0, sample_rate / 2], [2.0, 2.0]])
        weighted = estimators.inner_product_weighted(
            signal, signal, at=dt, psd=flat_psd, window="boxcar"
        )
        assert weighted == pytest.approx(expected / 2, rel=FFT_TOL, abs=0)

    @pytest.mark.parametrize(
        # At the sampling rate of 128 Hz, these lengths give frequency spacings
        # of 1 and 2 Hz. Both representations contain an integer number of
        # periods of the original 8 Hz tone.
        "n_samples", [128, 64], ids=["1-Hz-spacing", "2-Hz-spacing"]
    )
    def test_cropping_preserves_frequency_spacing(self, bin_centred_8hz_sinusoid, n_samples):
        """Preserve the inner product when restricting the PSD range.

        Regression: the frequency step was previously taken from ff[1] after
        cropping, mistaking the second retained frequency for the FFT bin
        spacing. This incorrectly changed the inner product when the PSD range
        changed, even though all signal power remained inside both ranges.
        The full and restricted ranges must give the same inner product using
        the original bin spacing.
        """
        signal, sample_rate, amplitude = bin_centred_8hz_sinusoid
        signal = signal[:n_samples]
        dt = 1 / sample_rate

        full_psd = np.array([[0.0, sample_rate / 2], [2.0, 2.0]])
        cropped_psd = np.array([[6.0, 10.0], [2.0, 2.0]])

        full = estimators.inner_product_weighted(
            signal, signal, at=dt, psd=full_psd, window="boxcar"
        )
        cropped = estimators.inner_product_weighted(
            signal, signal, at=dt, psd=cropped_psd, window="boxcar"
        )

        # The tone is strictly inside both bands, so its weighted power is equal.
        expected = n_samples / sample_rate * amplitude**2 / 2
        assert full == pytest.approx(expected, rel=FFT_TOL, abs=0)
        assert cropped == pytest.approx(expected, rel=FFT_TOL, abs=0)

    @pytest.mark.parametrize(
        "x, y, error",
        [
            pytest.param(np.ones(4), np.ones(5), "same length", id="unequal-lengths"),
            pytest.param(
                np.ones(4, dtype=complex), np.ones(4), "cannot be complex",
                id="complex-input",
            ),
        ],
    )
    def test_rejects_invalid_signals(self, x, y, error):
        """Check the ValueError for unequal signal lengths and complex inputs."""
        with pytest.raises(ValueError, match=error):
            estimators.inner_product_weighted(x, y, at=1.0, window="boxcar")

    def test_excluded_components_do_not_affect_interior_tone(
        self, contaminated_signals, flat_noise_psd
    ):
        """Ignore 0 Hz and Nyquist power in weighted and unweighted cross-products."""
        x, y = contaminated_signals
        psd = flat_noise_psd
        # One second of a unit-amplitude tone has unit-PSD norm 1. The second
        # tone has twice the amplitude; a PSD of 2 halves the cross-product.
        assert estimators.inner_product_weighted(
            x, y, at=1 / 32, window="boxcar"
        ) == pytest.approx(2.0, rel=FFT_TOL, abs=0)
        assert estimators.inner_product_weighted(
            x, y, at=1 / 32, psd=psd, window="boxcar"
        ) == pytest.approx(1.0, rel=FFT_TOL, abs=0)

    def test_signal_with_only_excluded_power_has_zero_norm(self, excluded_component):
        """Return zero when all signal power lies at 0 Hz or Nyquist."""
        assert estimators.inner_product_weighted(
            excluded_component, excluded_component, at=1 / 32, window="boxcar"
        ) == 0.0

    def test_highest_bin_below_nyquist_is_retained(self, highest_bin_tone):
        """Keep the highest usable bin for both odd and even signal lengths."""
        tone, _, dt, amplitude, psd = highest_bin_tone
        # One second of a bin-centred tone:
        #   (h|h) = A**2 / PSD
        for weighting, psd_level in ((None, 1.0), (psd, 2.0)):
            expected = amplitude**2 / psd_level
            assert estimators.inner_product_weighted(
                tone, tone, at=dt, psd=weighting, window="boxcar"
            ) == pytest.approx(expected, rel=FFT_TOL, abs=0)

    def test_zero_psd_at_excluded_bins_is_not_used(self, contaminated_signals):
        """Avoid dividing by PSD zeros at the excluded frequencies."""
        signal, _ = contaminated_signals
        psd = np.array([[0.0, 8.0, 16.0], [0.0, 2.0, 0.0]])
        assert estimators.inner_product_weighted(
            signal, signal, at=1 / 32, psd=psd, window="boxcar"
        ) == pytest.approx(0.5, rel=FFT_TOL, abs=0)

    def test_band_without_usable_bins_is_zero(self, empty_band_psd):
        """Return zero when the PSD band contains no usable FFT bins."""
        signal = np.arange(32, dtype=float)
        assert estimators.inner_product_weighted(
            signal, signal, at=1 / 32, psd=empty_band_psd, window="boxcar"
        ) == 0.0

    def test_exclusion_is_applied_after_windowing(self):
        """Verify that the 0 Hz bin is excluded after windowing.

        A Hann window turns a constant input into a constant part and a
        cosine. The expected value counts only the cosine. Keeping the 0 Hz
        bin would count both parts, and subtracting the mean before windowing
        would erase the input and give zero.
        """
        signal = np.ones(8)
        psd = np.array([[0.0, 4.0], [2.0, 2.0]])
        # Hann = 0.5 - 0.5*cos(2*pi*j/N). The remaining tone has amplitude
        # 0.5 and duration 1, so its weighted norm is 0.5**2 / 2.
        assert estimators.inner_product_weighted(
            signal, signal, at=1 / 8, psd=psd, window="hann"
        ) == pytest.approx(0.5**2 / 2, rel=FFT_TOL, abs=0)


class TestOverlap:
    @pytest.mark.parametrize(
        "signal_scale, orthogonal_scale, expected",
        [
            pytest.param(1.0, 0.0, 1.0, id="identical"),
            pytest.param(3.0, 0.0, 1.0, id="positive-scaling"),
            pytest.param(-1.0, 0.0, -1.0, id="sign-reversal"),
            pytest.param(0.0, 1.0, 0.0, id="orthogonal"),
        ],
    )
    def test_scale_sign_and_orthogonality(
        self, bin_centred_8hz_sinusoid, signal_scale, orthogonal_scale, expected
    ):
        """Check identity, amplitude and sign changes, and orthogonality."""
        signal, sample_rate, amplitude = bin_centred_8hz_sinusoid
        time = np.arange(len(signal)) / sample_rate
        orthogonal = amplitude * np.sin(2 * np.pi * 9 * time)
        other = signal_scale * signal + orthogonal_scale * orthogonal

        actual = estimators.overlap(signal, other, at=1/sample_rate, window="boxcar")
        assert actual == pytest.approx(expected, rel=0, abs=FFT_TOL)

    def test_weighted_mixture_and_its_pseudo_distance(self, bin_centred_8hz_sinusoid):
        """Check signed overlap and nonzero pseudo-distance with unequal PSD weights."""
        signal, sample_rate, amplitude = bin_centred_8hz_sinusoid
        dt = 1 / sample_rate
        time = np.arange(len(signal)) * dt
        mixture = -signal + amplitude * np.sin(2 * np.pi * 9 * time)  # add 9 Hz
        psd = np.array([[7.0, 10.0], [1.0, 4.0]])

        # Interpolation gives PSDs of 2 and 3 at 8 Hz and 9 Hz. The mixture
        # shares only the reversed 8 Hz tone, so its overlap is
        #   -sqrt((1/2) / (1/2 + 1/3)).
        expected = -np.sqrt(3 / 5)
        overlap = estimators.overlap(signal, mixture, at=dt, psd=psd, window="boxcar")
        distance = estimators.doverlap(signal, mixture, at=dt, psd=psd, window="boxcar")

        assert overlap == pytest.approx(expected, rel=FFT_TOL, abs=0)
        assert distance == pytest.approx((1 - expected) / 2, rel=FFT_TOL, abs=0)

    def test_excluded_components_do_not_affect_interior_tone(
        self, contaminated_signals, flat_noise_psd
    ):
        """Ignore 0 Hz and Nyquist power when normalising the overlap."""
        x, y = contaminated_signals
        psd = flat_noise_psd
        for weighting in (None, psd):
            assert estimators.overlap(
                x, y, at=1 / 32, psd=weighting, window="boxcar"
            ) == pytest.approx(1.0, rel=FFT_TOL, abs=0)

    def test_signal_with_only_excluded_power_has_zero_norm(self, excluded_component):
        """Return zero by convention when excluded power leaves no signal norm."""
        assert estimators.overlap(
            excluded_component, excluded_component, window="boxcar"
        ) == 0.0

    def test_highest_bin_below_nyquist_is_retained(self, highest_bin_tone):
        """Include the highest usable tone in the known equal-power mixture overlap."""
        tone, mixture, dt, _, psd = highest_bin_tone
        for weighting in (None, psd):
            assert estimators.overlap(
                mixture, tone, at=dt, psd=weighting, window="boxcar"
            ) == pytest.approx(1 / np.sqrt(2), rel=FFT_TOL, abs=0)

    def test_band_without_usable_bins_is_zero(self, empty_band_psd):
        """Return zero by convention when the PSD band contains no usable bins."""
        signal = np.arange(32, dtype=float)
        assert estimators.overlap(
            signal, signal, at=1 / 32, psd=empty_band_psd, window="boxcar"
        ) == 0.0


class TestMatch:
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

        assert match == pytest.approx(1.0, rel=FFT_TOL, abs=0)

    def test_phase_maximised_match_and_mismatch_with_missing_tone(self):
        """Preserve the known imperfect match when one of two tones is missing.

        Equal-amplitude tones at 3 and 8 Hz have weighted powers proportional
        to 1/4 and 1/9 under this PSD (when linearly interpolated).
        A template containing only the 3 Hz tone must therefore match by
            sqrt((1/4) / (1/4 + 1/9)) = sqrt(9/13),
        for any global phase of the template. The mismatch is its complement.
        """
        theta = 2 * np.pi * np.arange(128) / 128
        signal = np.sin(3 * theta) + np.sin(8 * theta)
        template = np.sin(3 * theta + 0.7)
        psd = np.array([[1.0, 10.0], [2.0, 11.0]])

        match = estimators.match(
            signal, template, at=1 / 128, psd=psd, window="boxcar"
        )
        mismatch = estimators.imatch(
            signal, template, at=1 / 128, psd=psd, window="boxcar"
        )
        expected = np.sqrt(9 / 13)
        assert match == pytest.approx(expected, rel=FFT_TOL, abs=0)
        assert mismatch == pytest.approx(1 - expected, rel=FFT_TOL, abs=0)

    @pytest.mark.parametrize(
        "x_has_power, y_has_power",
        [
            pytest.param(False, False, id="both-zero"),
            pytest.param(False, True, id="first-zero"),
            pytest.param(True, False, id="second-zero"),
        ],
    )
    def test_zero_norm_returns_zero_match_and_lag(self, x_has_power, y_has_power):
        """Return the documented value and lag when normalisation is undefined."""
        pulse = np.array([0.0, 1.0, 0.0, -1.0])
        x = x_has_power * pulse
        y = y_has_power * pulse

        assert estimators.match(x, y, window="boxcar") == 0.0
        assert estimators.match(
            x, y, window="boxcar", return_lag=True
        ) == (0.0, (0, 0.0))

    @pytest.mark.parametrize(
        "x, y, error",
        [
            pytest.param(np.ones(4), np.ones(5), "same length", id="unequal-lengths"),
            pytest.param(
                np.ones(4, dtype=complex), np.ones(4), "must be real-valued",
                id="complex-input",
            ),
        ],
    )
    def test_rejects_invalid_signals(self, x, y, error):
        """Check the ValueError for unequal signal lengths and complex inputs."""
        with pytest.raises(ValueError, match=error):
            estimators.match(x, y, at=1.0, window="boxcar")

    @pytest.mark.parametrize(
        "dt, psd, error",
        [
            pytest.param(0.0, None, "positive finite", id="zero-sampling-interval"),
            pytest.param(
                1 / 4, np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0]]),
                "strictly positive", id="zero-in-band-psd",
            ),
        ],
    )
    def test_rejects_invalid_sampling_or_in_band_psd(self, dt, psd, error):
        signal = np.array([0.0, 1.0, 0.0, -1.0])
        with pytest.raises(ValueError, match=error):
            estimators.match(signal, signal, at=dt, psd=psd, window="boxcar")

    def test_excluded_components_do_not_affect_interior_tone(
        self, contaminated_signals, flat_noise_psd
    ):
        """Ignore 0 Hz and Nyquist power in the correlation and normalisation."""
        x, y = contaminated_signals
        psd = flat_noise_psd
        for weighting in (None, psd):
            assert estimators.match(
                x, y, at=1 / 32, psd=weighting, window="boxcar"
            ) == pytest.approx(1.0, rel=FFT_TOL, abs=0)

    def test_signal_with_only_excluded_power_has_zero_norm(self, excluded_component):
        """Return zero match and lag when all signal power is excluded."""
        assert estimators.match(
            excluded_component, excluded_component, window="boxcar", return_lag=True
        ) == (0.0, (0, 0.0))

    def test_highest_bin_below_nyquist_is_retained(self, highest_bin_tone):
        """Include the highest usable tone in the known equal-power mixture match."""
        tone, mixture, dt, _, psd = highest_bin_tone
        for weighting in (None, psd):
            assert estimators.match(
                mixture, tone, at=dt, psd=weighting, window="boxcar"
            ) == pytest.approx(1 / np.sqrt(2), rel=FFT_TOL, abs=0)

    def test_band_without_usable_bins_is_zero(self, empty_band_psd):
        """Return zero match and lag when the PSD band contains no usable bins."""
        signal = np.arange(32, dtype=float)
        assert estimators.match(
            signal, signal, at=1 / 32, psd=empty_band_psd, window="boxcar", return_lag=True
        ) == (0.0, (0, 0.0))


class TestMatchAndImatch:
    """Check that imatch complements match and preserves its reported lag."""

    def test_recovers_cyclic_lag(self):
        """Recover a seven-sample delay and preserve it in the loss wrapper.

        The pulse [1, 0, -1] has zero Fourier coefficients at 0 Hz and Nyquist,
        so both excluded bins are already absent from the signal.
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

        assert match == pytest.approx(1.0, rel=FFT_TOL, abs=0)
        assert lag == (-7, -1.75)
        assert mismatch == pytest.approx(0.0, rel=0, abs=FFT_TOL)
        assert mismatch_lag == lag

    @pytest.mark.parametrize(
        "n_samples, shift, extra_amplitude",
        [
            pytest.param(127, 7, 0.0, id="delayed-perfect-match"),
            pytest.param(128, -7, 1.0, id="advanced-imperfect-match"),
        ],
    )
    def test_recovers_weighted_phase_and_cyclic_lag(
        self, n_samples, shift, extra_amplitude
    ):
        """Recover the signed lag despite global phase and missing content."""
        dt = 1 / n_samples
        theta = 2 * np.pi * np.arange(n_samples) / n_samples
        phase = 0.7

        signal = np.sin(3 * theta) + 0.5 * np.sin(8 * theta)
        signal += extra_amplitude * np.sin(5 * theta)
        rotated = np.sin(3 * theta + phase) + 0.5 * np.sin(8 * theta + phase)
        template = 2.5 * np.roll(rotated, shift)
        psd = np.array([[1.0, 10.0], [2.0, 11.0]])

        # Each tone contributes amplitude**2 / PSD. The shared 3 Hz and 8 Hz
        # tones have amplitudes 1 and 0.5, and interpolated PSD values 4 and 9.
        # The extra 5 Hz tone has PSD 6.
        shared_power = 1 / 4 + 0.5**2 / 9
        extra_power = extra_amplitude**2 / 6
        expected = np.sqrt(shared_power / (shared_power + extra_power))

        match, lag = estimators.match(
            signal, template, at=dt, psd=psd, window="boxcar", return_lag=True
        )
        mismatch, mismatch_lag = estimators.imatch(
            signal, template, at=dt, psd=psd, window="boxcar", return_lag=True
        )

        assert match == pytest.approx(expected, rel=FFT_TOL, abs=0)
        assert lag[0] == -shift
        assert lag[1] == pytest.approx(-shift / n_samples, rel=FLOAT64_RTOL, abs=0)
        assert mismatch == pytest.approx(1 - expected, rel=0, abs=FFT_TOL)
        assert mismatch_lag == lag


class TestSnr:
    def test_matches_known_bin_centred_signal(self, bin_centred_8hz_sinusoid):
        """Recover the analytical SNR from tone amplitude, duration, and PSD."""
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
        assert actual == pytest.approx(expected, rel=FFT_TOL, abs=0)

    def test_excluded_components_do_not_affect_interior_tone(
        self, contaminated_signals, flat_noise_psd
    ):
        """Ignore 0 Hz and Nyquist power in the signal norm."""
        x, _ = contaminated_signals
        psd = flat_noise_psd
        # One second of a unit-amplitude tone with PSD 2 gives SNR squared 1/2.
        assert estimators.snr(
            x, at=1 / 32, psd=psd, window=np.ones(32)
        )**2 == pytest.approx(0.5, rel=FFT_TOL, abs=0)

    def test_signal_with_only_excluded_power_has_zero_norm(self, excluded_component):
        """Return zero SNR when all signal power lies at 0 Hz or Nyquist."""
        psd = np.array([[0.0, 16.0], [2.0, 2.0]])
        assert estimators.snr(
            excluded_component, at=1 / 32, psd=psd, window=np.ones(32)
        ) == 0.0

    def test_highest_bin_below_nyquist_is_retained(self, highest_bin_tone):
        """Keep the highest usable tone's analytical SNR for odd and even lengths."""
        tone, _, dt, amplitude, psd = highest_bin_tone
        assert estimators.snr(
            tone, at=dt, psd=psd, window=np.ones(len(tone))
        )**2 == pytest.approx(amplitude**2 / 2, rel=FFT_TOL, abs=0)

    def test_zero_psd_at_excluded_bins_is_not_used(self, contaminated_signals):
        """Avoid dividing by PSD zeros at the excluded frequencies."""
        signal, _ = contaminated_signals
        psd = np.array([[0.0, 8.0, 16.0], [0.0, 2.0, 0.0]])
        assert estimators.snr(
            signal, at=1 / 32, psd=psd, window=np.ones(32)
        )**2 == pytest.approx(0.5, rel=FFT_TOL, abs=0)

    def test_band_without_usable_bins_is_zero(self, empty_band_psd):
        """Return zero SNR when the PSD band contains no usable FFT bins."""
        signal = np.arange(32, dtype=float)
        assert estimators.snr(
            signal, at=1 / 32, psd=empty_band_psd, window=np.ones(32)
        ) == 0.0

    def test_exclusion_is_applied_after_windowing(self):
        """Verify that the 0 Hz bin is excluded after windowing.

        A Hann window turns a constant input into a constant part and a
        cosine. The expected value counts only the cosine. Keeping the 0 Hz
        bin would count both parts, and subtracting the mean before windowing
        would erase the input and give zero.
        """
        signal = np.ones(8)
        psd = np.array([[0.0, 4.0], [2.0, 2.0]])
        # The periodic Hann window creates a tone of amplitude 0.5 over one
        # second; its squared SNR with PSD 2 is 0.5**2 / 2.
        assert estimators.snr(
            signal, at=1 / 8, psd=psd, window=get_window("hann", len(signal))
        )**2 == pytest.approx(0.5**2 / 2, rel=FFT_TOL, abs=0)


@pytest.mark.parametrize("sample_rate", [64.0, 128.0], ids=["64-Hz", "128-Hz"])
@pytest.mark.parametrize(
    "window", ["boxcar", ("tukey", 0.5)], ids=["boxcar", "tukey"]
)
def test_self_inner_product_equals_snr_squared(sample_rate, window):
    """Check (h|h) = SNR squared across sample rates and windows."""
    dt = 1 / sample_rate
    time = np.arange(128) * dt
    signal = 0.25 * np.sin(2 * np.pi * 8 * time)
    # The PSD is interpolated onto the FFT grid at both sample rates.
    psd = np.array([[5.75, 10.25], [2.0, 6.0]])

    inner = estimators.inner_product_weighted(
        signal, signal, at=dt, psd=psd, window=window
    )
    snr = estimators.snr(
        signal, at=dt, psd=psd, window=get_window(window, len(signal))
    )

    assert inner == pytest.approx(snr**2, rel=FFT_TOL, abs=0)


@pytest.mark.parametrize(
    "upper_cutoff, upper_tone_retained",
    [
        pytest.param(8.0, False, id="on-tone-bin"),
        pytest.param(8.5, False, id="between-bins"),
        pytest.param(9.0, True, id="on-next-bin"),
    ],
)
def test_estimators_follow_pycbc_upper_cutoff(
    bin_centred_8hz_sinusoid, upper_cutoff, upper_tone_retained
):
    """Apply the same exclusive upper bin index to all frequency estimators."""
    upper_tone, sample_rate, amplitude = bin_centred_8hz_sinusoid
    dt = 1 / sample_rate
    time = np.arange(len(upper_tone)) * dt
    lower_tone = amplitude * np.sin(2 * np.pi * 4 * time)
    mixture = lower_tone + upper_tone
    psd = np.array([[0.0, upper_cutoff], [2.0, 2.0]])

    # Each tone contributes A**2 / 2 over this one-second window.
    n_retained = 1 + upper_tone_retained
    expected_power = n_retained * amplitude**2 / 2
    expected_similarity = 1 / np.sqrt(n_retained)

    inner = estimators.inner_product_weighted(
        mixture, mixture, at=dt, psd=psd, window="boxcar"
    )
    snr = estimators.snr(mixture, at=dt, psd=psd, window=np.ones(len(mixture)))
    overlap = estimators.overlap(mixture, lower_tone, at=dt, psd=psd, window="boxcar")
    match = estimators.match(mixture, lower_tone, at=dt, psd=psd, window="boxcar")

    assert inner == pytest.approx(expected_power, rel=FFT_TOL, abs=0)
    assert snr == pytest.approx(np.sqrt(expected_power), rel=FFT_TOL, abs=0)
    assert overlap == pytest.approx(expected_similarity, rel=FFT_TOL, abs=0)
    assert match == pytest.approx(expected_similarity, rel=FFT_TOL, abs=0)


def test_find_merger_returns_largest_absolute_sample():
    assert estimators.find_merger(np.array([1.0, -4.0, 3.0])) == 1
