import yaml

from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
import numpy as np

from . import lib


class Pipeline:
    def __init__(self, *, dico_den, dico_den_params, dico_clas, dico_clas_params):
        if dico_den.components.shape[0] > dico_clas.D.shape[0]:
            raise ValueError(
                "the length of the atoms of the denoising dictionary must be shorter"
                " or equal to the length of the atoms of the classification dictionary"
            )

        # Load settings and dictionaries.
        self.dico_den = dico_den
        self.dico_den_params = dico_den_params
        self.dico_clas = dico_clas
        self.dico_clas_params = dico_clas_params

    def __call__(self, strains, with_losses=False):
        # if strains.shape[1] != dico_clas.D.shape[0]:
        #     raise ValueError(
        #         "the length of the strains must be equal to the length of the atoms"
        #         " of the classification dictionary"
        #     )

        chewed = self._preprocess(strains)
        results = self._predict(chewed, with_losses=with_losses)
        return results

    def _preprocess(self, strains):
        # Denoise + norm
        pps = self.dico_den.reconstruct_minibatch(strains.T, normed=True, **self.dico_den_params).T
        # assert pps.shape[1] == self.dico_clas.D.shape[0], "issue when windowing signal with the given length and step"
        return pps

    def _predict(self, strains, with_losses=False):
        results = self.dico_clas.predict(strains, with_losses=with_losses, **self.dico_clas_params)
        return results
