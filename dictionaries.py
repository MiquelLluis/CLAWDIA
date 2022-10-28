"""Try to import all available dictionaries."""
import warnings

import numpy as np


try:
    from ._dictionary_spams import DictionarySpams
except ImportError:
    warnings.warn("spams-python not found, 'DictionarySpams' won't be available", ImportWarning)

# try:
#     from ._dictionary_sklearn import DictionarySklearn
# except ImportError:
#     warnings.warn("scikit-learn not installed, 'DictionarySklearn' won't be available", ImportWarning)


#______________________________________________________________________________
# Load and save functions.

def load(file):
    """TODO

    Only tested with DictionarySpams!

    """
    dico_raw = dict(np.load(file))
    # Initialize DictionarySpams instance
    dico = DictionarySpams(dict_init=dico_raw.pop('dict_init'))
    # Restore the state of the dictionary
    for key, value in dico_raw.items():
        # Restore all 0d-array to their former types
        if value.ndim == 0:
            dico_raw[key] = value.item()
        setattr(dico, key, value)

    return dico


def save(file, dico):
    np.savez(file, **vars(dico))
