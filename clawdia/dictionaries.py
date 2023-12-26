import numpy as np

from ._dictionary_spams import DictionarySpams
from ._dictionary_lrsdl import DictionaryLRSDL


def load(file):
    dico_raw = dict(np.load(file))

    if 'lambd2' in dico_raw:
        # Initialize LRSDL instance
        dico = DictionaryLRSDL(
            lambd=dico_raw.pop('lambd'), lambd2=dico_raw.pop('lambd2'), eta=dico_raw.pop('eta'),
            k=dico_raw.pop('k'), k0=dico_raw.pop('k0'), updateX_iters=dico_raw.pop('updateX_iters'),
            updateD_iters=dico_raw.pop('updateD_iters')
        )
    else:
        # Initialize DictionarySpams instance
        dico = DictionarySpams(dict_init=dico_raw.pop('dict_init'))

    # Restore the state of the dictionary
    for key, value in dico_raw.items():
        # Restore all 0d-array to their former types
        if value.ndim == 0:
            value = value.item()
        setattr(dico, key, value)

    return dico


def save(file, dico):
    """Call to the dictionary's save method."""

    dico.save(file)
