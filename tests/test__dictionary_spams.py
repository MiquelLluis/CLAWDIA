import sys

import numpy as np
import pytest

import clawdia
from clawdia._dictionary_spams import DictionarySpams
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


#------------------------------------------------------------------------------
# TESTS

def test___init__(dico_initial, components_init):
    np.testing.assert_array_almost_equal(dico_initial.components, components_init, decimal=9)


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='random reproducibility of SPAM only guaranteed on Linux')
def test_train(dico_initial, strains_clean, wave_pos_clean, components_trained):
    training_patches = clawdia.lib.extract_patches(
        strains_clean,
        patch_size=64,
        limits=wave_pos_clean,
        n_patches=100,
        random_state=84,
        l2_normed=True,
        allow_allzeros=False
    )
    dico = dico_initial.copy()
    dico.train(
        training_patches,
        n_iter=1000,
        verbose=False,
        threads=1
    )

    np.testing.assert_array_almost_equal(dico.components, components_trained, decimal=9)


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='random reproducibility of SPAM only guaranteed on Linux')
def test_train_warm(dico_initial, strains_clean, wave_pos_clean):
    training_patches = clawdia.lib.extract_patches(
        strains_clean,
        patch_size=64,
        limits=wave_pos_clean,
        n_patches=100,
        random_state=84,
        l2_normed=True,
        allow_allzeros=False
    )
    dico = dico_initial.copy()
    dico.train(
        training_patches,
        n_iter=1000,
        verbose=False,
        threads=1
    )
    dico.train(
        training_patches,
        warm_start=True,
        n_iter=500,
        verbose=False,
        threads=1
    )

    target_components = np.load('tests/data/_dictionary_spams/dico_spams_trained_warm.npy')

    np.testing.assert_array_almost_equal(dico.components, target_components, decimal=9)


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


def test_reconstruct_batch(dico_trained, reconstructions_input, reconstructions_target):
    reconstructions = dico_trained.reconstruct_batch(
        reconstructions_input,
        sc_lambda=0.5,
        step=2,
        normed=True,
        verbose=False
    )
    
    np.testing.assert_array_almost_equal(reconstructions, reconstructions_target, decimal=9)


def test_reconstruct_minibatch(dico_trained, reconstructions_input, reconstructions_target):
    reconstructions = dico_trained.reconstruct_minibatch(
        reconstructions_input,
        sc_lambda=0.5,
        step=2,
        batchsize=2,
        normed=True,
        normed_windows=True,
        verbose=False
    )
    
    np.testing.assert_array_almost_equal(reconstructions, reconstructions_target, decimal=9)


def test_reconstruct_iterative_minibatch(dico_trained, reconstructions_iterative_input,
                                         reconstructions_iterative_target,
                                         reconstructions_iterative_residuals_target,
                                         reconstructions_iterative_iters_target):
    reconstructions, residuals, iters = dico_trained.reconstruct_iterative_minibatch(
        reconstructions_iterative_input,
        sc_lambda=0.7,
        step=2,
        batchsize=2,
        max_iter=1000,
        threshold=0.01,
        normed=True,
        full_output=True,
        verbose=False
    )
    
    np.testing.assert_array_almost_equal(
        reconstructions, reconstructions_iterative_target, decimal=9
    )
    np.testing.assert_array_almost_equal(
        residuals, reconstructions_iterative_residuals_target, decimal=9
    )
    np.testing.assert_array_almost_equal(
        iters, reconstructions_iterative_iters_target, decimal=9
    )


def test_reconstruct_margin_constrained(dico_trained, target_reconstruct_margin_constrained):
    strain_input = target_reconstruct_margin_constrained['input']
    reconstruction, code, result = dico_trained.reconstruct_margin_constrained(
        strain_input,
        zero_marg=100,
        lambda_lims=(0.01, 10),
        step=4,
        normed=True,
        full_output=True
    )
    code = code.toarray()

    np.testing.assert_array_almost_equal(reconstruction, target_reconstruct_margin_constrained['reconstruction'], decimal=9)
    np.testing.assert_array_almost_equal(code, target_reconstruct_margin_constrained['code'], decimal=9)
    assert result == pytest.approx(target_reconstruct_margin_constrained['result'].item())


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