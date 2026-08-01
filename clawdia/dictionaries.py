"""Main module for managing all SDL models.

This module serves as the central interface for handling dictionary models 
included in the CLAWDIA pipeline. It provides classes and functions to load, 
save, and manage different types of dictionary models used in Sparse Dictionary 
Learning (SDL). Support is included for both SPAMS-based dictionaries and 
Low-Rank Sparse Dictionary Learning (LRSDL) models, ensuring compatibility and 
ease of use.

"""
import numpy as np

from ._dictionary_spams import DictionarySpams
from ._dictionary_lrsdl import DictionaryLRSDL


__all__ = ['DictionarySpams', 'DictionaryLRSDL', 'load', 'save']


def load(file):
    """Load a saved CLAWDIA dictionary and restore its public state."""
    with np.load(file, allow_pickle=True) as archive:
        dico_raw = {
            key: value.item()
            if isinstance(value, np.ndarray) and value.ndim == 0
            else value
            for key, value in archive.items()
        }

    format_version = dico_raw.pop('_clawdia_format_version', None)
    is_legacy = format_version is None

    # Initialise the correct dictionary instance.
    if 'lambd2' in dico_raw:  # LRSDL
        dico = DictionaryLRSDL(
            lambd=dico_raw.pop('lambd'), lambd2=dico_raw.pop('lambd2'), eta=dico_raw.pop('eta'),
            k=dico_raw.pop('k'), k0=dico_raw.pop('k0'), updateX_iters=dico_raw.pop('updateX_iters'),
            updateD_iters=dico_raw.pop('updateD_iters')
        )
    
    else:  # SPAMS
        dict_init = dico_raw.pop('dict_init')
        
        # For backwards compatibility with versions previous to v0.4,
        # transpose it from Fortran to C order.
        if is_legacy and dict_init.flags.f_contiguous:
            dict_init = dict_init.T
        
        dico = DictionarySpams(dict_init=dict_init)


    # Restore the state of the dictionary
    for key, value in dico_raw.items():
        # For backwards compatibility with versions previous to v0.4,
        # transpose all dictionary components from Fortran to C order.
        if (
            is_legacy
            and isinstance(value, np.ndarray)
            and value.ndim == 2
            and value.flags.f_contiguous
        ):
            value = value.T

        if (
            isinstance(dico, DictionaryLRSDL)
            and key in {'D_range', 'Y_range'}
            and isinstance(value, np.ndarray)
        ):
            value = value.tolist()

        setattr(dico, key, value)
    
    return dico


def save(file, dico):
    """Same as using the dictionary's save method."""
    dico.save(file)
