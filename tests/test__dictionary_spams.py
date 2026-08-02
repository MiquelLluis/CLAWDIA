import sys

import numpy as np
import pytest

from clawdia import lib
from clawdia.dictionaries import DictionarySpams


@pytest.fixture(scope="module")
def clean_data(data_dir):
    with np.load(data_dir / "strains_clean.npz") as data:
        return data["strains"], data["wave_pos"]


@pytest.fixture(scope="module")
def initial_components(data_dir):
    return np.load(data_dir / "_dictionary_spams" / "dico_spams_initial.npy")


@pytest.fixture(scope="module")
def trained_components(data_dir):
    return np.load(data_dir / "_dictionary_spams" / "dico_spams_trained.npy")


@pytest.fixture(scope="module")
def initial_dictionary(clean_data):
    signals, limits = clean_data
    return DictionarySpams(
        signal_pool=signals,
        wave_pos=limits,
        a_length=64,
        d_size=80,
        lambda1=0.1,
        batch_size=1,
        identifier="Test dictionary",
        l2_normed=True,
        allow_allzeros=False,
        patch_min=16,
        random_state=42,
    )


@pytest.fixture(scope="module")
def trained_dictionary(initial_components, trained_components):
    model = DictionarySpams(
        dict_init=initial_components.copy(),
        lambda1=0.1,
        batch_size=1,
        identifier="Test dictionary",
        trained=True,
        n_iter=1000,
        n_train=100,
    )
    model.components = trained_components.copy()
    return model


@pytest.mark.regression
def test_seeded_initial_dictionary_matches_reference(
    initial_dictionary, initial_components
):
    np.testing.assert_allclose(
        initial_dictionary.components,
        initial_components,
        rtol=1e-9,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        np.linalg.norm(initial_dictionary.components, axis=1),
        1.0,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.integration
@pytest.mark.regression
@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="the committed SPAMS training reference is Linux-specific",
)
def test_training_and_warm_start_match_reference(
    initial_dictionary, clean_data, trained_components, data_dir
):
    signals, limits = clean_data
    patches = lib.extract_patches(
        signals,
        patch_size=64,
        limits=limits,
        n_patches=100,
        random_state=84,
        l2_normed=True,
        allow_allzeros=False,
    )
    model = initial_dictionary.copy()
    model.train(patches, n_iter=1000, verbose=False, threads=1)

    assert model.trained
    assert model.n_train == 100
    assert model.n_iter == 1000
    assert model.t_train > 0
    np.testing.assert_allclose(
        model.components, trained_components, rtol=1e-9, atol=1e-11
    )

    model.train(
        patches,
        warm_start=True,
        n_iter=500,
        verbose=False,
        threads=1,
    )
    warm_reference = np.load(
        data_dir / "_dictionary_spams" / "dico_spams_trained_warm.npy"
    )
    assert model.n_iter == 1500
    np.testing.assert_allclose(
        model.components, warm_reference, rtol=1e-9, atol=1e-11
    )


def test_identity_dictionary_has_analytical_soft_threshold_solution(
    identity_dictionary,
):
    """Verify exact LASSO soft-thresholding with an identity dictionary."""
    signal = np.array([1.0, 0.0, 0.0, 0.0])
    reconstruction = identity_dictionary.reconstruct(
        signal, sc_lambda=0.25, normed=False
    )
    expected = np.array([0.75, 0.0, 0.0, 0.0])
    nrmse = np.linalg.norm(reconstruction - expected) / np.linalg.norm(expected)

    assert nrmse < 1e-9
    np.testing.assert_allclose(
        reconstruction, expected, rtol=1e-9, atol=1e-11
    )


def test_reconstruction_preserves_nondivisible_signal_length(identity_dictionary):
    signal = np.array([1.0, 0.0, 0.5, 0.0, -0.25, 0.0, 0.0])
    reconstruction = identity_dictionary.reconstruct(
        signal, sc_lambda=0.0, step=2, normed=False
    )

    assert reconstruction.shape == signal.shape
    np.testing.assert_allclose(
        reconstruction, signal, rtol=1e-9, atol=1e-11
    )


def test_reconstruction_normalisation_and_zero_signal(identity_dictionary):
    signal = np.array([2.0, 0.0, 0.0, 0.0])
    reconstruction = identity_dictionary.reconstruct(
        signal, sc_lambda=0.25, normed=True
    )
    np.testing.assert_allclose(
        reconstruction, [1.0, 0.0, 0.0, 0.0], rtol=1e-9, atol=1e-11
    )

    zeros = identity_dictionary.reconstruct(
        np.zeros(4), sc_lambda=0.25, normed=True
    )
    np.testing.assert_array_equal(zeros, 0.0)


def test_batch_and_partial_minibatch_equal_individual_reconstruction(
    identity_dictionary,
):
    signals = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0, 0.0],
        ]
    )
    expected = np.vstack(
        [
            identity_dictionary.reconstruct(x, 0.2, normed=False)
            for x in signals
        ]
    )
    batch = identity_dictionary.reconstruct_batch(
        signals, 0.2, normed=False, verbose=False
    )
    minibatch = identity_dictionary.reconstruct_minibatch(
        signals,
        sc_lambda=0.2,
        batchsize=2,
        normed=False,
        verbose=False,
    )

    np.testing.assert_allclose(batch, expected, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(minibatch, expected, rtol=1e-9, atol=1e-11)


def test_partial_minibatch_forwards_window_normalisation(identity_dictionary):
    signals = np.array(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
        ]
    )
    expected = identity_dictionary.reconstruct_batch(
        signals,
        0.25,
        normed=False,
        normed_windows=False,
        verbose=False,
    )
    actual = identity_dictionary.reconstruct_minibatch(
        signals,
        sc_lambda=0.25,
        batchsize=2,
        normed=False,
        normed_windows=False,
        verbose=False,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-11)


def test_iterative_reconstruction_conserves_signal(identity_dictionary):
    signals = np.array(
        [[1.0, 0.5, 0.0, 0.0], [0.25, -0.75, 0.0, 0.0]]
    )
    reconstruction, residual, iterations = (
        identity_dictionary.reconstruct_iterative(
            signals,
            sc_lambda=0.2,
            batchsize=1,
            max_iter=20,
            threshold=1e-10,
            normed=False,
            full_output=True,
            verbose=False,
        )
    )

    np.testing.assert_allclose(
        reconstruction + residual, signals, rtol=1e-9, atol=1e-11
    )
    assert np.all(iterations <= 20)
    assert np.linalg.norm(residual) <= np.linalg.norm(signals)


def test_margin_constrained_reconstruction_finds_suppression_boundary(
    identity_dictionary,
):
    signal = np.array([1.0, 0.0, 0.0, 0.0])
    reconstruction, code, result = (
        identity_dictionary.reconstruct_margin_constrained(
            signal,
            margin=4,
            lambda_lims=(0.1, 2.0),
            normed=False,
            full_output=True,
        )
    )

    assert reconstruction.shape == signal.shape
    assert code is not None
    assert result["converged"]
    assert result["x"] == pytest.approx(1.0, abs=2e-10)
    assert np.linalg.norm(reconstruction) < 1e-9


@pytest.mark.parametrize('dico', ['dico_initial', 'dico_trained'])
def test_copy(dico, request):
    dico = request.getfixturevalue(dico)
    dico_copy = dico.copy()
    np.testing.assert_array_equal(dico.components, dico_copy.components)
    np.testing.assert_array_equal(dico.dict_init, dico_copy.dict_init)


def test_reconstruct(dico_trained, reconstructions_input,
                     reconstructions_target, reconstructions_code_target):
    reconstructions = []
    codes = []

    for i, x in enumerate(reconstructions_input):
        rec, code = dico_trained.reconstruct(
            x,
            sc_lambda=0.5,
            step=2,
            normed=True,
            with_code=True
        )
        reconstructions.append(rec)
        codes.append(code.toarray())

    reconstructions = np.array(reconstructions)
    codes = np.array(codes)

    np.testing.assert_array_almost_equal(reconstructions, reconstructions_target, decimal=9)
    np.testing.assert_array_almost_equal(codes, reconstructions_code_target, decimal=9)


def test_optimum_lambda(dico_trained, target_optimum_lambda):
    strain_input = target_optimum_lambda['input']
    strain_ref = target_optimum_lambda['reference']
    strain_limits = target_optimum_lambda['limits']

    rec, l_opt, loss = dico_trained.reconstruct_optimum_lambda(
        strain_input,
        reference=strain_ref,
        kwargs_minimize={},
        kwargs_lasso={},
        step=2,
        limits=strain_limits,
        normed=True
    )

    np.testing.assert_array_almost_equal(rec, target_optimum_lambda['reconstruction'], decimal=9)
    assert l_opt == pytest.approx(target_optimum_lambda['l_opt'].item())
    assert loss == pytest.approx(target_optimum_lambda['loss'].item())


def test_reset(dico_initial, dico_trained):
    dico = dico_trained.copy()
    dico.reset()

    np.testing.assert_array_equal(dico.components, dico_initial.components)
    np.testing.assert_array_equal(dico.dict_init, dico_initial.dict_init)
    assert not dico.trained
    assert dico.n_train is None
    assert dico.t_train is None