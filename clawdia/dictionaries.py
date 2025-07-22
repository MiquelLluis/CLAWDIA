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
    dico_raw = dict(np.load(file, allow_pickle=True))

    # Initialize the correct dictionary instance.
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
        if dict_init.flags.f_contiguous:
            print('dict_init')
            dict_init = dict_init.T
        
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
    
    # In case of loading older instances in which this attribute
    # didn't exist, it is set to the default of spams.trainDL.
    # This way it should produce the same results as before.
    if isinstance(dico, DictionarySpams) and not hasattr(dico, 'modeD_traindl'):
        dico.modeD_traindl = 0

    return dico


def save(file, dico):
    """Same as using the dictionary's save method."""
    dico.save(file)
