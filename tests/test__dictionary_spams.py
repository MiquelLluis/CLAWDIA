import numpy as np
import pytest

import clawdia
from clawdia._dictionary_spams import DictionarySpams


#------------------------------------------------------------------------------
# LOAD DATA

@pytest.fixture(scope='module')
def strains_clean():
    return np.load('tests/data/strains_clean.npy')


@pytest.fixture(scope='module')
def wave_pos_clean():
    return np.load('tests/data/wave_pos_clean.npy')


# @pytest.fixture
# def strains_injected():
#     return np.load('tests/data/strains_injected.npy')


@pytest.fixture(scope='module')
def components_init():
    return np.load('tests/data/_dictionary_spams/dico_spams_initial.npy')


@pytest.fixture(scope='module')
def components_trained():
    return np.load('tests/data/_dictionary_spams/dico_spams_trained.npy')


@pytest.fixture(scope='module')
def reconstructions_test_A():
    return np.load('tests/data/_dictionary_spams/reconstructions_A.npz')


@pytest.fixture(scope='module')
def reconstructions_target(reconstructions_test_A):
    return reconstructions_test_A['target']
    

@pytest.fixture(scope='module')
def reconstructions_input(reconstructions_test_A):
    return reconstructions_test_A['input']



#------------------------------------------------------------------------------
# COMPUTE DICTIONARIES 1 TIME

@pytest.fixture(scope='module')
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


@pytest.fixture(scope='module')
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
    np.testing.assert_array_equal(dico_initial.components, components_init)


def test_train(dico_trained, components_trained):
    np.testing.assert_array_equal(dico_trained.components, components_trained)


@pytest.mark.parametrize('dico', ['dico_initial', 'dico_trained'])
def test_copy(dico, request):
    dico = request.getfixturevalue(dico)
    dico_copy = dico.copy()
    np.testing.assert_array_equal(dico.components, dico_copy.components)
    np.testing.assert_array_equal(dico.dict_init, dico_copy.dict_init)


def test_reconstruct(dico_trained, reconstructions_input, reconstructions_target):
    reconstructions = np.zeros_like(reconstructions_input)
    for i, x in enumerate(reconstructions_input):
        rec = dico_trained.reconstruct(
            x,
            sc_lambda=0.5,
            step=2,
            normed=False,
            with_code=False
        )
        reconstructions[i,:len(rec)] = rec
    
    np.testing.assert_array_equal(reconstructions, reconstructions_target)


def test_reconstruct_batch(dico_trained, reconstructions_input, reconstructions_target):
    reconstructions = np.zeros_like(reconstructions_input)

    recs = dico_trained.reconstruct_batch(
        reconstructions_input.T,  # Current version still uses Fortran order.
        sc_lambda=0.5,
        step=2,
        normed=False,
        verbose=False
    )
    reconstructions[:, :recs.shape[0]] = recs.T
    
    np.testing.assert_array_equal(reconstructions, reconstructions_target)


def test_reconstruct_minibatch(dico_trained, reconstructions_input, reconstructions_target):
    reconstructions = np.zeros_like(reconstructions_input)

    recs = dico_trained.reconstruct_minibatch(
        reconstructions_input.T,  # Current version still uses Fortran order.
        sc_lambda=0.5,
        step=2,
        batchsize=2,
        normed=False,
        normed_windows=True,
        verbose=False
    )
    reconstructions[:, :recs.shape[0]] = recs.T
    
    np.testing.assert_array_equal(reconstructions, reconstructions_target)
