import numpy as np


_xtol = 2e-12
_rtol = 8.881784197001252e-16


def abs_normalize(array, axis=0):
	"""TODO
	Normalitza inplace un array ignorant els errors de divissió entre 0 i
	canviant els nan a 0.

	"""
	with np.errstate(divide='ignore', invalid='ignore'):
	    array /= np.max(np.abs(array), axis=axis, keepdims=True)
	    np.nan_to_num(array, copy=False)


def semibool_bisect(f, a, b, args=(), xtol=_xtol, rtol=_rtol, maxiter=100):
	"""TODO

	Mètode de bisecció adaptat a una funció f(x) tal que
		f(x) != 0    	x <= x0  i
		f(x) == 0       x  > x0,
	o viceversa. Un dels dos extrems del límite [a, b] ha de ser f(x) = 0.
	Algorisme basat en la funció de bisecció `scipy.optimize.bisect`.

	"""
	fa = f(a, *args)
	fb = f(b, *args)
	solver_stats = {'funcalls': 2}
	if fa*fb != 0:
		raise ValueError("There isn't a boundary point yielding f(x)=0")
	if fa < fb:
		a, b = b, a
		fa, fb = fb, fa
	
	dm = b - a
	for i in range(maxiter):
		dm *= 0.5
		xm = a + dm
		fm = f(xm, *args)
		solver_stats['funcalls'] += 1
		if fm != 0:
			a = xm
		if abs(dm) < xtol + rtol*abs(xm):
			solver_stats['converged'] = True
			solver_stats['niters'] = i+1
			solver_stats['x'] = xm
			return solver_stats

	solver_stats['converged'] = False
	solver_stats['niters'] = i+1
	solver_stats['x'] = a
	
	return solver_stats
