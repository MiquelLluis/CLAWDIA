import pickle
import yaml

import clawdia
from dataset import Dataset


def main(cfg):
    with open('data/dataset.pkl', 'rb') as f:
        ds = pickle.load(f)

    dico = clawdia.dictionaries.DictionarySpams(
        signal_pool=ds.Xtrain.T,
        wave_pos=None,  # TODO: Segurament desprès els tinga que especificar.
        a_length=cfg['a_length'],
        d_size=cfg['d_size'],
        lambda1=cfg['lambda1'],
        batch_size=cfg['batch_size'],
        l2_normed=cfg['l2_normed'],
        allow_allzeros=cfg['allow_allzeros'],
        n_iter=cfg['n_iter'],
        patch_min=None,  # TODO: Només cal si wave_pos s'especifica.
        random_state=cfg['random_seed'],
        ignore_completeness=cfg['ignore_completeness'],
        mode_traindl=cfg['mode_traindl'],
        mode_lasso=cfg['mode_lasso']
    )

    train_patches = clawdia.lib.extract_patches(
        ds.Xtrain.T,
        patch_size=cfg['a_length'],
        limits=None,  # TODO: Segurament desprès els tinga que especificar.
        n_patches=cfg['train_patches'],
        l2_normed=cfg['l2_normed'],
        allow_allzeros=cfg['allow_allzeros'],
        patch_min=None,  # TODO: Només cal si wave_pos s'especifica.
        random_state=cfg['random_seed']
    )

    dico.train(train_patches, verbose=True, threads=cfg['threads'])

    return dico


if __name__ == '__main__':
    with open('parameters.yaml') as f:
        cfg = yaml.safe_load(f)['denoising_dictionary']

    dico = main(cfg)

    dico.save('data/denoising_dictionary.npz')

    print("DONETE!")
