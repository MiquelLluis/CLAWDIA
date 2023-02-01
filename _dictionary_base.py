from abc import ABC, abstractmethod
import time

import numpy as np
import spams

from . import patches_1d
from . import util


# Remove warning from OpenMP, present in older versions of python-spams.
if not '__version__' in dir(spams) or spams.__version__ <= '2.6.5.4':
    import os
    os.environ['KMP_WARNINGS'] = 'FALSE'


class _DictionaryBase(ABC):
    """Base class of the dictionaries.

    TODO

    """
    @abstractmethod 
    def _reconstruct(self, sc_lambda, step, **kwargs):
        pass

    def reconstruct(self, signal, sc_lambda, step=1, normed=True, with_code=False, **kwargs):
        """Reconstruct a signal as a sparse combination of dictionary atoms.

        Parameters
        ----------
        signal : ndarray
            Sample to be reconstructed.

        sc_lambda : float
            Regularization parameter of the sparse coding transformation.

        step : int, 1 by default
            Sample interval between each patch extracted from signal.
            Determines the number of patches to be extracted. 1 by default.

        normed : boolean, True by default
            Normalize the result to the maximum absolute value.

        with_code : boolean, False by default.
            If True, also returns the coefficients array.

        **kwargs
            Passed directly to the external learning function.

        Returns
        -------
        signal_rec : array
            Reconstructed signal.

        code : array(p_size, d_size), optional
            Transformed data, encoded as a sparse combination of atoms.
            Returned when 'with_code' is True.

        """
        if not isinstance(signal, np.ndarray):
            raise TypeError("'signal' must be a numpy array")

        signal_rec, code = self._reconstruct(signal, sc_lambda, step, **kwargs)

        if normed and signal_rec.any():
            norm = np.max(np.abs(signal_rec))
            signal_rec /= norm
            code /= norm

        return (signal_rec, code) if with_code else signal_rec

    def reconstruct_batch(self, signals, sc_lambda, out=None, step=1, normed=True, **kwargs):
        """TODO

        Reconstruct multiple signals, each one as a sparse combination of
        dictionary atoms.

        """
        if out is None:
            out = np.empty_like(signals)
        n_signals = signals.shape[1]

        for i in range(n_signals):
            out[:,i] = self.reconstruct(
                signals[:,i],
                sc_lambda, 
                step=step,
                normed=normed,
                with_code=False,
                **kwargs
            )

        return out

    def reconstruct_auto(self, signal, *, zero_marg, lambda_lims, step=1, normed=True,
                         full_output=False, kwargs_bisect={}, kwargs_rec={}):
        """TODO

        Reconstrueix un únic senyal buscant per bisecció la lambda que
        minimitza el senyal reconstruit al marge esquerre del senyal, la mida
        dels quals ve determinada per 'zero_marg'.

        """
        # Margins of the signals to be zeroed
        margin = signal[:zero_marg]
        # Function to be bisected.
        def fun(sc_lambda):
            rec, _ = self._reconstruct(margin, sc_lambda, step, **kwargs_rec)
            return np.sum(np.abs(rec))

        try:
            result = util.semibool_bisect(fun, *lambda_lims, **kwargs_bisect)
        except util.BoundaryError:
            rec = np.zeros_like(signal)
            code = None
            result = {'x': np.min(lambda_lims), 'f': 0., 'converged': False, 'niters': 0, 'funcalls': 2}
        else:
            rec, code = self._reconstruct(signal, result['x'], step, **kwargs_rec)
            if normed and rec.any():
                norm = np.max(np.abs(rec))
                rec /= norm
                code /= norm

        return (rec, code, result) if full_output else rec

    @abstractmethod
    def save(self, file):
        pass
