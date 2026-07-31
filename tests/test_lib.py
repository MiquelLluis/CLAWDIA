import numpy as np
import pytest

from clawdia import lib





def test_extract_patches_returns_exact_multisignal_windows():
    signals = np.arange(12, dtype=np.float64).reshape(2, 6)
    patches = lib.extract_patches(signals, patch_size=3, step=2)
    expected = np.array(
        [[0, 1, 2], [2, 3, 4], [6, 7, 8], [8, 9, 10]], dtype=float
    )
    np.testing.assert_array_equal(patches, expected)
    assert patches.dtype == signals.dtype


def test_extract_patches_limits_select_windows_with_required_overlap():
    signal = np.arange(7, dtype=float)[None, :]
    patches = lib.extract_patches(
        signal,
        patch_size=4,
        limits=np.array([[2, 6]]),
        patch_min=3,
    )
    np.testing.assert_array_equal(patches, [[1, 2, 3, 4], [2, 3, 4, 5]])


def test_extract_patches_truncates_with_warning():
    signal = np.arange(8, dtype=float)
    with pytest.warns(RuntimeWarning, match="last 1 bins"):
        patches = lib.extract_patches(signal, patch_size=3, step=2)
    np.testing.assert_array_equal(
        patches, [[0, 1, 2], [2, 3, 4], [4, 5, 6]]
    )


def test_extract_patches_padding_covers_last_sample_exactly():
    signal = np.arange(8, dtype=float)
    patches = lib.extract_patches(
        signal, patch_size=3, step=2, allow_padding=True
    )
    np.testing.assert_array_equal(
        patches, [[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 0]]
    )


def test_extract_patches_padding_supports_short_signals():
    patches = lib.extract_patches(
        np.array([2.0, 3.0]), patch_size=4, allow_padding=True
    )
    np.testing.assert_array_equal(patches, [[2.0, 3.0, 0.0, 0.0]])


def test_random_patch_extraction_is_seeded():
    signal = np.arange(20, dtype=float)
    first = lib.extract_patches(
        signal, patch_size=4, n_patches=5, random_state=42
    )
    second = lib.extract_patches(
        signal, patch_size=4, n_patches=5, random_state=42
    )
    np.testing.assert_array_equal(first, second)


def test_patch_l2_normalisation_and_coefficients_reconstruct_input():
    signal = np.arange(6, dtype=float)
    patches, norms = lib.extract_patches(
        signal,
        patch_size=3,
        l2_normed=True,
        return_norm_coefs=True,
    )
    raw = lib.extract_patches(signal, patch_size=3)

    assert np.issubdtype(patches.dtype, np.floating)
    np.testing.assert_allclose(
        np.linalg.norm(patches[1:], axis=1), 1.0, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        patches * norms[:, None], raw, rtol=1e-12, atol=1e-12
    )


def test_all_zero_patch_handling():
    zeros = np.zeros(6)
    patches, norms = lib.extract_patches(
        zeros,
        patch_size=3,
        l2_normed=True,
        return_norm_coefs=True,
    )
    np.testing.assert_array_equal(patches, 0)
    np.testing.assert_array_equal(norms, 0)

    with pytest.raises(ValueError, match="contains only zeros"):
        lib.extract_patches(
            zeros,
            patch_size=3,
            n_patches=1,
            allow_allzeros=False,
        )


@pytest.mark.parametrize(
    "signal, kwargs, error",
    [
        (np.zeros((2, 2, 2)), {"patch_size": 2}, ValueError),
        (np.arange(4), {"patch_size": 0}, ValueError),
        (np.arange(4), {"patch_size": 5}, ValueError),
        (np.arange(4), {"patch_size": 2, "step": 0}, ValueError),
        (np.arange(4), {"patch_size": 2, "n_patches": -1}, ValueError),
        (
            np.arange(4),
            {"patch_size": 2, "limits": np.array([[0, 2], [2, 4]])},
            ValueError,
        ),
    ],
)
def test_extract_patches_rejects_invalid_geometry(signal, kwargs, error):
    with pytest.raises(error):
        lib.extract_patches(signal, **kwargs)


def test_extract_patches_rejects_over_extraction():
    with pytest.raises(ValueError, match="maximum number of patches"):
        lib.extract_patches(
            np.arange(4, dtype=float), patch_size=2, n_patches=4
        )



