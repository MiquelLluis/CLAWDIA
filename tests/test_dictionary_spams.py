import pickle

import numpy as np
from numpy.testing import assert_array_equal
import yaml

import clawdia
from dataset import Dataset


with open('parameters.yaml') as f:
    CFG = yaml.safe_load(f)
with open('data/dataset.pkl', 'rb') as f:
    DATASET = pickle.load(f)
DICO = clawdia.dictionaries.load('data/denoising_dictionary.npz')


def test_DictionarySpams_reconstruct_iterative_minibatch():
    cfg = CFG['denoising']
    ds = DATASET
    dico = DICO
    with open('data/denoised_iteratively.pkl', 'rb') as f:
        target_reconstructions, target_residuals, target_iters = pickle.load(f)

    signals = ds.dataset.copy().T.copy()
    signals /= np.max(np.abs(signals), axis=0)
    rng = np.random.default_rng(cfg['random_seed'])
    noise = np.float32(rng.uniform(-1, 1, size=signals.shape[0]))
    injections = signals + (cfg['max_amplitude_noise']
                            * np.tile(noise[:,None], signals.shape[1]))
    injections /= np.max(np.abs(injections), axis=0)

    reconstructions, residuals, iters = dico.reconstruct_iterative_minibatch(
        injections, sc_lambda=cfg['sc_lambda'], threshold=cfg['threshold'],
        step=cfg['step'], batchsize=cfg['batchsize'], normed=True,
        max_iter=cfg['max_iter'], full_output=True, verbose=True
    )

    assert_array_equal(reconstructions, reconstructions)
    assert_array_equal(residuals, target_residuals)
    assert_array_equal(iters, target_iters)
