"""Basic implementation of the pipeline model.

The `Pipeline` class provides a minimal example of how to use CLAWDIA as a 
classification pipeline. This implementation assumes that the dictionaries 
have already been trained and that all necessary hyperparameters and 
post-training parameters are provided.

.. warning::

    This module is still under development and has not been thoroughly tested. 
    API changes or unexpected behavior may occur in future updates.

"""


class Pipeline:
    """
    Implements a basic classification pipeline that preprocesses input 
    gravitational-wave strain data using a denoising dictionary and 
    subsequently classifies it using a classification dictionary.
    
    """
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

    def __call__(self, strains, with_losses=False, with_preprocessed=False):
        chewed = self._preprocess(strains)
        results = self._predict(chewed, with_losses=with_losses)
        if with_preprocessed:
            results += (chewed,)
        return results

    def _preprocess(self, strains):
        # Denoise + norm
        pps = self.dico_den.reconstruct_minibatch(strains.T, normed=True, **self.dico_den_params).T
        return pps

    def _predict(self, strains, with_losses=False):
        results = self.dico_clas.predict(strains, with_losses=with_losses, **self.dico_clas_params)
        return results
