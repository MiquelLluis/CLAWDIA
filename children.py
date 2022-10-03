import warnings

import numpy as np
import sklearn


def omp_singlematch_batch(parents, dictionary, **kwargs):
	"""TODO
	Troba els indexs `i_atoms` amb els seus coeficients `c_atoms` dels àtoms
	de `dictionary` més pareguts cada pare en `parents` gastant l'OMP.
	Aquells pares per als que l'OMP no convergisca tindràn coeficient
	`c_atoms[i] = 0`.

	"""
	# Ignore convergence warnings from sklearn's sparse_encode.
	with warnings.catch_warnings():
	    warnings.simplefilter('ignore', category=RuntimeWarning)
	    codes = sklearn.decomposition.sparse_encode(
	        parents.T,
	        dictionary.T,
	        algorithm='omp',
	        n_nonzero_coefs=1,
	        **kwargs
	    ).T

	i_atoms = np.argmax(np.abs(codes), axis=0)
	c_atoms = np.ravel(codes[i_atoms,np.indices(i_atoms.shape)])

	return i_atoms, c_atoms