import numpy as np

from ._dictionary_spams import DictionarySpams
from ._dictionary_lrsdl import DictionaryLRSDL


__all__ = ['load', 'save', 'DictionarySpams', 'DictionaryLRSDL']


def load(file):
    dico_raw = dict(np.load(file, allow_pickle=True))

    if 'lambd2' in dico_raw:
        # Initialize LRSDL instance
        dico = DictionaryLRSDL(
            lambd=dico_raw.pop('lambd'), lambd2=dico_raw.pop('lambd2'), eta=dico_raw.pop('eta'),
            k=dico_raw.pop('k'), k0=dico_raw.pop('k0'), updateX_iters=dico_raw.pop('updateX_iters'),
            updateD_iters=dico_raw.pop('updateD_iters')
        )
    
    else:
        dict_init = dico_raw.pop('dict_init')
        
        # For backwards compatibility with versions previous to v0.4,
        # transpose it from Fortran to C order.
        if dict_init.flags.f_contiguous:
            print('dict_init')
            dict_init = dict_init.T
        
        # Initialize DictionarySpams instance
        dico = DictionarySpams(dict_init=dict_init)


    # Restore the state of the dictionary
    for key, value in dico_raw.items():
        # Restore all 0d-array to their former types
        if value.ndim == 0:
            value = value.item()
        
        # For backwards compatibility with versions previous to v0.4,
        # transpose all dictionary components from Fortran to C order.
        elif value.ndim == 2 and value.flags.f_contiguous:
            print(key)
            value = value.T

        setattr(dico, key, value)

    return dico


def save(file, dico):
    """Call to the dictionary's save method."""

    dico.save(file)
