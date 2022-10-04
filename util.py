import numpy as np


def abs_normalize(array, axis=0):
	"""TODO
	Normalitza inplace un array ignorant els errors de divissió entre 0 i
	canviant els nan a 0.

	"""
	with np.errstate(divide='ignore', invalid='ignore'):
	    array /= np.max(np.abs(array), axis=axis, keepdims=True)
	    np.nan_to_num(array, copy=False)
