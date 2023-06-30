import yaml

from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
import numpy as np

from . import lib


# Current steps to leave outside Claudia (or for future added steps):
#------------------------------------------------------------------------------

# INPUT
# - Extract windows from frame
# - Whiten(asd) + crop + norm

# OUTPUT
# - Denoised strains (optional)
# - Prediction vector
# - F1 score
# - Confusion matrix


# Current steps to include:
#------------------------------------------------------------------------------

# INPUT:
# - Strains: ndarray
#       > Length = atoms' length!!!
# - Dictionaries

# Preprocessing:
# - Denoise + norm

# Classification:
# - DictionaryLRSDL.predict labels

#------------------------------------------------------------------------------



class Pipeline:
    def __init__(self):
        # Load settings and dictionaries.
        pass

    def __call__(self, strains):
        # Complete list of steps of the pipeline.
        pass

    def _preprocess(self):
        # Can be extended by inheriting Pipeline.
        pass

    def _predict(self):
        pass
