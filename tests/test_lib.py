import numpy as np
import pytest

from clawdia import lib



class TestExtractPatches:
    """Test class for the `extract_patches` function."""

    # Fixtures for common test signals
    @pytest.fixture(scope="class")
    def basic_2d_signal(self):
        """A 2D signal with two 6-element signals."""
        return np.array([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]], dtype=np.float64)

    @pytest.fixture(scope="class")
    def basic_1d_signal(self):
        """A 1D signal converted to 2D internally."""
        return np.array([0, 1, 2, 3, 4, 5], dtype=np.float64)

    @pytest.fixture(scope="class")
    def signal_with_limits(self):
        """A 2D signal with limits restricting patch extraction."""
        signals = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.float64)
        limits = np.array([[1, 4]])  # Valid region: indices 1 to 3 (elements 1, 2, 3)
        return signals, limits

    # Test input validation
    def test_3d_input_raises_error(self):
        """Test that 3D input raises ValueError."""
        signals = np.random.rand(2, 2, 2)
        with pytest.raises(ValueError):
            lib.extract_patches(signals, patch_size=2)

    def test_patch_min_exceeds_limits(self, signal_with_limits):
        """Test error when patch_min exceeds allowed by limits."""
        signals, limits = signal_with_limits
        with pytest.raises(ValueError):
            lib.extract_patches(signals, patch_size=3, limits=limits, patch_min=4)

    def test_n_patches_exceeds_max(self):
        """Test error when n_patches exceeds maximum possible."""
        signals = np.array([[0, 1, 2, 3]], dtype=np.float64)
        with pytest.raises(ValueError):
            lib.extract_patches(signals, patch_size=2, n_patches=4)

    # Test extraction of all possible patches
    def test_extract_all_patches_2d(self, basic_2d_signal):
        """Test extracting all patches from 2D signal without limits."""
        signals = basic_2d_signal
        patch_size = 3
        patches = lib.extract_patches(signals, patch_size=patch_size)
        expected_shape = (8, patch_size)  # 4 patches per signal * 2 signals
        assert patches.shape == expected_shape

    def test_extract_all_patches_with_limits(self, signal_with_limits):
        """Test extraction with limits returns correct patches."""
        signals, limits = signal_with_limits
        patch_size = 3
        patches = lib.extract_patches(signals, patch_size=patch_size, limits=limits)
        expected = [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
        assert np.array_equal(patches, expected)

    # Test random extraction
    def test_random_extraction_reproducible(self, basic_1d_signal):
        """Test random extraction with fixed random_state is reproducible."""
        patch_size = 3
        n_patches = 2
        patches1 = lib.extract_patches(basic_1d_signal, patch_size=patch_size, n_patches=n_patches, random_state=42)
        patches2 = lib.extract_patches(basic_1d_signal, patch_size=patch_size, n_patches=n_patches, random_state=42)
        assert np.array_equal(patches1, patches2)

    # Test step parameter
    def test_step_parameter(self, basic_1d_signal):
        """Test step parameter reduces number of patches."""
        patch_size = 3
        step = 2
        patches = lib.extract_patches(basic_1d_signal, patch_size=patch_size, step=step)
        assert patches.shape == (2, patch_size)

    # Test L2 normalization
    def test_l2_normalization(self, basic_1d_signal):
        """Test patches are L2 normalized and coefs are correct."""
        patch_size = 3
        patches, coefs = lib.extract_patches(basic_1d_signal, patch_size=patch_size, l2_normed=True, return_norm_coefs=True)
        norms = np.linalg.norm(patches, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6) | np.allclose(norms, 0.0, atol=1e-6)
        original_patches = lib.extract_patches(basic_1d_signal, patch_size=patch_size)
        expected_coefs = np.linalg.norm(original_patches, axis=1)
        assert np.allclose(coefs, expected_coefs)

    def test_zero_patch_normalization(self):
        """Test all-zero patches are handled correctly."""
        signals = np.zeros((1, 6))
        patch_size = 3
        patches, coefs = lib.extract_patches(signals, patch_size=patch_size, l2_normed=True, return_norm_coefs=True)
        assert np.all(patches == 0) and np.all(coefs == 0)

    # Test allow_allzeros
    def test_allow_allzeros_false(self):
        """Test allow_allzeros=False skips zero patches."""
        signals = np.array([[0, 0, 0, 1, 1, 1]])
        patches = lib.extract_patches(signals, patch_size=3, n_patches=1, allow_allzeros=False, random_state=42)
        assert np.any(patches != 0)

    # Test return_norm_coefs
    def test_return_norm_coefs_without_l2(self, basic_1d_signal):
        """Test return_norm_coefs returns ones without L2."""
        patches, coefs = lib.extract_patches(basic_1d_signal, patch_size=3, return_norm_coefs=True)
        assert np.all(coefs == 1.0)

    # Test warnings
    def test_warning_signal_not_fully_divided(self):
        """Test warning when signal can't be fully divided."""
        signal = np.array([0, 1, 2, 3, 4])
        with pytest.warns(RuntimeWarning):
            lib.extract_patches(signal, patch_size=2, step=2)

    # Test limits and patch_min interaction
    def test_limits_and_patch_min(self):
        """Test patch_min correctly restricts patch starts."""
        signals = np.array([[0, 1, 2, 3, 4, 5]])
        limits = np.array([[2, 5]])
        patches = lib.extract_patches(signals, patch_size=3, limits=limits, patch_min=2)
        for patch in patches:
            count = sum((2 <= x) & (x < 5) for x in patch)
            assert count >= 2

    def test_extract_patches_with_step_1(self, basic_1d_signal):
        """Test step=1 extracts maximum patches (no skipping)."""
        patches = lib.extract_patches(basic_1d_signal, patch_size=3, step=1)
        assert patches.shape == (4, 3)  # 6-3+1=4 patches
        np.testing.assert_array_equal(patches[0], [0, 1, 2])
        np.testing.assert_array_equal(patches[-1], [3, 4, 5])

    def test_extract_patches_with_large_step(self, basic_1d_signal):
        """Test step=4 skips intermediate patches."""
        patches = lib.extract_patches(basic_1d_signal, patch_size=3, step=4)
        assert patches.shape == (1, 3)  # Only first patch [0,1,2]

    def test_over_extraction_attempt_raises_error(self):
        """Test n_patches > max available raises ValueError with exact message."""
        signals = np.array([[0, 1, 2, 3]])
        patch_size = 2
        n_patches = 10
        max_patches = 3  # (4 - 2 + 1)

        with pytest.raises(ValueError) as exc_info:
            lib.extract_patches(signals, patch_size=patch_size, n_patches=n_patches)
        
        expected_msg = (
            f"the keyword argument 'n_patches' ({n_patches}) exceeds "
            f"the maximum number of patches that can be extracted ({max_patches})."
        )
        assert str(exc_info.value) == expected_msg

    def test_min_patch_size_with_limits(self):
        """Test patch_min=3 requires 3 valid elements within limits."""
        signals = np.array([[0, 1, 2, 3, 4, 5, 6]])
        limits = np.array([[2, 6]])  # Valid indices 2-5 (values 2,3,4,5)
        patches = lib.extract_patches(signals, patch_size=4, limits=limits, patch_min=3)
        # Valid patches must have ≥3 elements within 2-5:
        expected = np.array([[1, 2, 3, 4], [2, 3, 4, 5]])
        np.testing.assert_array_equal(patches, expected)

    def test_odd_length_signal_splitting(self):
        """Test splitting signal with odd length and no step."""
        signal = np.array([0, 1, 2, 3, 4])  # Length 5
        patches = lib.extract_patches(signal, patch_size=2)
        assert patches.shape == (4, 2)  # 5-2+1=4 patches

    def test_l2_normalization_without_coefs(self, basic_1d_signal):
        """Test L2 normalization without returning coefficients."""
        patches = lib.extract_patches(basic_1d_signal, patch_size=3, l2_normed=True)
        norms = np.linalg.norm(patches, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6) | np.allclose(norms, 0.0, atol=1e-6)

    def test_extract_all_patches_implicitly(self, basic_1d_signal):
        """Test n_patches=None extracts all possible patches (default)."""
        patches = lib.extract_patches(basic_1d_signal, patch_size=3, n_patches=None)
        assert patches.shape == (4, 3)  # Same as step=1
