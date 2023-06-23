import numpy as np

from ._dictionary_spams import DictionarySpams
from ._dictionary_dictol import train_lrsdl, predict_lrsdl


def load(file):
    """TODO

    WARNING: Only tested with DictionarySpams!

    """
    dico_raw = dict(np.load(file))
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
    """TODO

    Just call to the dictionary's save method.

    WARNING: Only works with DictionarySpams!

    """
    dico.save(file)
