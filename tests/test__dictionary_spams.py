import numpy as np
import pytest

import clawdia
from clawdia._dictionary_spams import DictionarySpams


#------------------------------------------------------------------------------
# LOAD DATA

@pytest.fixture(scope='module')
def file_clean():
    return np.load('tests/data/strains_clean.npz')
@pytest.fixture(scope='module')
def strains_clean(file_clean):
    return file_clean['strains']
@pytest.fixture(scope='module')
def wave_pos_clean(file_clean):
    return file_clean['wave_pos']


@pytest.fixture(scope='module')
def components_init():
    return np.load('tests/data/_dictionary_spams/dico_spams_initial.npy')
@pytest.fixture(scope='module')
def components_trained():
    return np.load('tests/data/_dictionary_spams/dico_spams_trained.npy')


@pytest.fixture(scope='module')
def reconstructions_basic():
    return np.load('tests/data/_dictionary_spams/reconstructions_A.npz')
@pytest.fixture(scope='module')
def reconstructions_input(reconstructions_basic):
    return reconstructions_basic['input']
@pytest.fixture(scope='module')
def reconstructions_target(reconstructions_basic):
    return reconstructions_basic['target_reconstructions']
@pytest.fixture(scope='module')
def reconstructions_code_target(reconstructions_basic):
    return reconstructions_basic['target_codes']
    

@pytest.fixture
def reconstructions_iterative():
    return np.load('tests/data/_dictionary_spams/reconstructions_iterative.npz')
@pytest.fixture
def reconstructions_iterative_input(reconstructions_iterative):
    return reconstructions_iterative['input']
@pytest.fixture
def reconstructions_iterative_target(reconstructions_iterative):
    return reconstructions_iterative['target_reconstructions']
@pytest.fixture
def reconstructions_iterative_residuals_target(reconstructions_iterative):
    return reconstructions_iterative['target_residuals']
@pytest.fixture
def reconstructions_iterative_iters_target(reconstructions_iterative):
    return reconstructions_iterative['target_iters']


@pytest.fixture
def target_reconstruct_auto():
    return np.load('tests/data/_dictionary_spams/reconstruct_auto.npz', allow_pickle=True)



#------------------------------------------------------------------------------
# COMPUTE DICTIONARIES 1 TIME

@pytest.fixture
def dico_initial(strains_clean, wave_pos_clean):
    return DictionarySpams(
        signal_pool=strains_clean.T,
        wave_pos=wave_pos_clean,
        a_length=64,
        d_size=80,
        lambda1=0.1,
        batch_size=1,
        identifier='Test dictionary',
        l2_normed=True,
        allow_allzeros=False,
        patch_min=16,
        random_state=42
    )


@pytest.fixture
def dico_trained(dico_initial, strains_clean, wave_pos_clean):
    training_patches = clawdia.lib.extract_patches(
        strains_clean.T,
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
    return dico



#------------------------------------------------------------------------------
# TESTS

def test___init__(dico_initial, components_init):
    np.testing.assert_array_almost_equal(dico_initial.components, components_init, decimal=9)


def test_train(dico_trained, components_trained):
    np.testing.assert_array_almost_equal(dico_trained.components, components_trained, decimal=9)


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
        reconstructions_input.T,  # Current version still uses Fortran order.
        sc_lambda=0.5,
        step=2,
        normed=True,
        verbose=False
    )
    reconstructions = reconstructions.T
    
    np.testing.assert_array_almost_equal(reconstructions, reconstructions_target, decimal=9)


def test_reconstruct_minibatch(dico_trained, reconstructions_input, reconstructions_target):
    reconstructions = dico_trained.reconstruct_minibatch(
        reconstructions_input.T,  # Current version still uses Fortran order.
        sc_lambda=0.5,
        step=2,
        batchsize=2,
        normed=True,
        normed_windows=True,
        verbose=False
    )
    reconstructions = reconstructions.T
    
    np.testing.assert_array_almost_equal(reconstructions, reconstructions_target, decimal=9)


def test_reconstruct_iterative_minibatch(dico_trained, reconstructions_iterative_input,
                                         reconstructions_iterative_target,
                                         reconstructions_iterative_residuals_target,
                                         reconstructions_iterative_iters_target):
    reconstructions, residuals, iters = dico_trained.reconstruct_iterative_minibatch(
        reconstructions_iterative_input.T,
        sc_lambda=0.7,
        step=2,
        batchsize=2,
        max_iter=1000,
        threshold=0.01,
        normed=True,
        full_output=True,
        verbose=False
    )
    reconstructions = reconstructions.T
    residuals = residuals.T
    
    np.testing.assert_array_almost_equal(
        reconstructions, reconstructions_iterative_target, decimal=9
    )
    np.testing.assert_array_almost_equal(
        residuals, reconstructions_iterative_residuals_target, decimal=9
    )
    np.testing.assert_array_almost_equal(
        iters, reconstructions_iterative_iters_target, decimal=9
    )


def test_reconstruct_auto(dico_trained, target_reconstruct_auto):
    strain_input = target_reconstruct_auto['input']
    reconstruction, code, result = dico_trained.reconstruct_auto(
        strain_input,
        zero_marg=100,
        lambda_lims=(0.01, 10),
        step=4,
        normed=True,
        full_output=True
    )
    code = code.toarray()

    np.testing.assert_array_almost_equal(reconstruction, target_reconstruct_auto['reconstruction'], decimal=9)
    np.testing.assert_array_almost_equal(code, target_reconstruct_auto['code'], decimal=9)
    assert result == pytest.approx(target_reconstruct_auto['result'].item())
