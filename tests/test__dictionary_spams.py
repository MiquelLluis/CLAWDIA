import sys

import numpy as np
import pytest

import clawdia._dictionary_spams as dictionary_spams_module
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


def test_loss_optimised_reconstruction_has_known_lambda(identity_dictionary):
    signal = np.array([1.0, 0.0, 0.0, 0.0])
    reference = np.array([0.6, 0.0, 0.0, 0.0])

    def squared_error(x, y):
        return float(np.mean((x - y) ** 2))

    reconstruction, optimum, loss = (
        identity_dictionary.reconstruct_loss_optimised(
            signal,
            reference=reference,
            loss_func=squared_error,
            normed=False,
            kwargs_minimize={
                "method": "bounded",
                "bounds": (np.log10(0.05), np.log10(0.9)),
                "options": {"xatol": 1e-7},
            },
        )
    )

    assert optimum == pytest.approx(0.4, rel=1e-5)
    np.testing.assert_allclose(
        reconstruction, reference, rtol=1e-5, atol=1e-7
    )
    assert loss == pytest.approx(squared_error(reconstruction, reference))


def test_loss_optimised_reuses_best_evaluation_when_last_is_worse(
    identity_dictionary, monkeypatch
):
    signal = np.array([1.0, 0.0, 0.0, 0.0])
    reference = np.array([0.6, 0.0, 0.0, 0.0])
    optimum_log = np.log10(0.4)
    calls = []
    reconstruct = identity_dictionary.reconstruct

    def recording_reconstruct(signal_, sc_lambda, **kwargs):
        calls.append(sc_lambda)
        return reconstruct(signal_, sc_lambda, **kwargs)

    def minimise_with_worse_last_evaluation(cost_function, **kwargs):
        optimum_loss = cost_function(optimum_log)
        cost_function(np.log10(0.8))
        return {"x": optimum_log, "fun": optimum_loss, "success": True}

    monkeypatch.setattr(identity_dictionary, "reconstruct", recording_reconstruct)
    monkeypatch.setattr(
        dictionary_spams_module.scipy.optimize,
        "minimize_scalar",
        minimise_with_worse_last_evaluation,
    )

    reconstruction, optimum, loss = (
        identity_dictionary.reconstruct_loss_optimised(
            signal,
            reference=reference,
            loss_func=lambda x, y: float(np.mean((x - y) ** 2)),
            normed=False,
        )
    )

    np.testing.assert_allclose(
        reconstruction, reference, rtol=1e-9, atol=1e-11
    )
    assert optimum == pytest.approx(0.4)
    assert loss == pytest.approx(0.0, abs=1e-20)
    np.testing.assert_allclose(calls, [0.4, 0.8], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("loss_function", ["match", "overlap", "ssim"])
def test_loss_optimised_documented_metrics_recover_identical_shape(
    identity_dictionary, loss_function
):
    signal = np.array([0.0, 1.0, 0.0, 0.0])
    reconstruction, optimum, loss = (
        identity_dictionary.reconstruct_loss_optimised(
            signal,
            reference=signal,
            loss_func=loss_function,
            normed=True,
            kwargs_minimize={
                "method": "bounded",
                "bounds": (np.log10(0.01), np.log10(0.5)),
                "options": {"xatol": 1e-5},
            },
        )
    )

    assert reconstruction.shape == signal.shape
    assert 0.01 <= optimum <= 0.5
    assert loss == pytest.approx(0.0, abs=1e-10)


def test_loss_optimised_handles_nondivisible_cropped_interval(
    identity_dictionary,
):
    signal = np.array([0.0, 1.0, 0.0, 0.5, 0.0, -0.25, 0.0])
    reference = 0.7 * signal
    limits = (1, 6)

    def squared_error(x, y):
        return float(np.mean((x - y) ** 2))

    reconstruction, optimum, loss = (
        identity_dictionary.reconstruct_loss_optimised(
            signal,
            reference=reference,
            limits=limits,
            step=2,
            loss_func=squared_error,
            normed=False,
            kwargs_minimize={
                "method": "bounded",
                "bounds": (np.log10(0.01), np.log10(0.9)),
                "options": {"xatol": 1e-6},
            },
        )
    )
    cropped = identity_dictionary.reconstruct(
        signal[slice(*limits)], optimum, step=2, normed=False
    )

    assert reconstruction.shape == signal.shape
    assert cropped.shape == reference[slice(*limits)].shape
    assert loss == pytest.approx(
        squared_error(cropped, reference[slice(*limits)]), rel=1e-12, abs=1e-12
    )


@pytest.mark.regression
def test_realistic_single_batch_and_minibatch_reconstruction(
    trained_dictionary, data_dir
):
    with np.load(
        data_dir / "_dictionary_spams" / "reconstructions_A.npz"
    ) as data:
        signals = data["input"]
        expected = data["target_reconstructions"]
        expected_codes = data["target_codes"]

    reconstructions = []
    codes = []
    for signal in signals:
        reconstruction, code = trained_dictionary.reconstruct(
            signal,
            sc_lambda=0.5,
            step=2,
            normed=True,
            with_code=True,
        )
        reconstructions.append(reconstruction)
        codes.append(code.toarray())

    np.testing.assert_allclose(
        reconstructions, expected, rtol=1e-9, atol=1e-11
    )
    np.testing.assert_allclose(codes, expected_codes, rtol=1e-9, atol=1e-11)

    batch = trained_dictionary.reconstruct_batch(
        signals, 0.5, step=2, normed=True, verbose=False
    )
    minibatch = trained_dictionary.reconstruct_minibatch(
        signals,
        sc_lambda=0.5,
        step=2,
        batchsize=2,
        normed=True,
        verbose=False,
    )
    np.testing.assert_allclose(batch, expected, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(minibatch, expected, rtol=1e-9, atol=1e-11)


@pytest.mark.regression
def test_realistic_iterative_reconstruction(trained_dictionary, data_dir):
    with np.load(
        data_dir / "_dictionary_spams" / "reconstructions_iterative.npz"
    ) as data:
        signals = data["input"]
        expected_reconstruction = data["target_reconstructions"]
        expected_residual = data["target_residuals"]
        expected_iterations = data["target_iters"]

    reconstruction, residual, iterations = trained_dictionary.reconstruct_iterative(
        signals,
        sc_lambda=0.7,
        step=2,
        batchsize=2,
        max_iter=1000,
        threshold=0.01,
        normed=True,
        full_output=True,
        verbose=False,
    )
    # The first full minibatch remains a valid dependency regression. The final
    # row in the committed baseline encoded the old remainder-batch bug, which
    # silently re-enabled window normalisation.
    np.testing.assert_allclose(
        reconstruction[:2],
        expected_reconstruction[:2],
        rtol=1e-9,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        residual[:2], expected_residual[:2], rtol=1e-9, atol=1e-11
    )
    np.testing.assert_array_equal(iterations[:2], expected_iterations[:2])
    assert reconstruction.shape == signals.shape
    assert residual.shape == signals.shape
    assert np.all(np.isfinite(reconstruction[2]))
    assert np.all(np.isfinite(residual[2]))
    assert iterations[2] <= 1000
    assert np.linalg.norm(residual[2]) <= np.linalg.norm(signals[2])


@pytest.mark.regression
def test_realistic_margin_boundary_retains_legacy_prefix(
    trained_dictionary, data_dir
):
    with np.load(
        data_dir / "_dictionary_spams" / "reconstruct_auto.npz",
        allow_pickle=True,
    ) as data:
        signal = data["input"]
        legacy_reconstruction = data["reconstruction"]
        legacy_code = data["code"]
        legacy_result = data["result"].item()

    reconstruction, code, result = (
        trained_dictionary.reconstruct_margin_constrained(
            signal,
            margin=100,
            lambda_lims=(0.01, 10),
            step=4,
            normed=True,
            full_output=True,
        )
    )

    assert reconstruction.shape == signal.shape
    np.testing.assert_allclose(
        reconstruction[:3208],
        legacy_reconstruction[:3208],
        rtol=1e-9,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        code.toarray()[:, : legacy_code.shape[1]],
        legacy_code,
        rtol=1e-9,
        atol=1e-11,
    )
    assert result["x"] == pytest.approx(legacy_result["x"], abs=1e-12)


@pytest.mark.parametrize('dico', ['dico_initial', 'dico_trained'])
def test_copy(dico, request):
    dico = request.getfixturevalue(dico)
    dico_copy = dico.copy()
    np.testing.assert_array_equal(dico.components, dico_copy.components)
    np.testing.assert_array_equal(dico.dict_init, dico_copy.dict_init)


def test_reset(dico_initial, dico_trained):
    dico = dico_trained.copy()
    dico.reset()

    np.testing.assert_array_equal(dico.components, dico_initial.components)
    np.testing.assert_array_equal(dico.dict_init, dico_initial.dict_init)
    assert not dico.trained
    assert dico.n_train is None
    assert dico.t_train is None