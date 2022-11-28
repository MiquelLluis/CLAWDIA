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


def l2_normalize(array, axis=0):
    """TODO
    Normalitza inplace un array amb la norma L2 ignorant els errors de divissió
    entre 0 i canviant els nan a 0.

    """
    with np.errstate(divide='ignore', invalid='ignore'):
        array /= np.linalg.norm(array, axis=axis, keepdims=True)
        np.nan_to_num(array, copy=False)


def semibool_bisect(f, a, b, args=(), xtol=_xtol, rtol=_rtol, maxiter=100, verbose=False):
    """TODO

    Troba x0 pel mètode de bisecció adaptat a una funció f(x) tal que
        f(x)  > 0       x <= x0,
        f(x) == 0       x  > x0,
    o viceversa. Un dels dos extrems del límite [a, b] ha de ser f(x) = 0.
    Algorisme basat en la funció de bisecció `scipy.optimize.bisect`.

    Nota: 'rtol' controla l'ordre de precissió respecte 'x'.

    Result
    ------
    solver_stats: dict
        'x': Solution.
        'f': Value of f(x).
        'converged': bool.
        'niters': Number of iterations performed.
        'funcalls': Number of times `f` was evaluated.

    """
    fa = f(a, *args)
    fb = f(b, *args)
    solver_stats = {'funcalls': 2}
    if fa*fb != 0:
        raise ValueError("There isn't a boundary point in the 0 zone")
    if fa == 0:
        a, b = b, a
        fa, fb = fb, fa
    
    dm = b - a
    for i in range(maxiter):
        dm *= 0.5
        xm = a + dm
        if verbose:
            print(f" iteration {i}, evaluating f({xm}) ...")
        fm = f(xm, *args)
        solver_stats['funcalls'] += 1
        if fm != 0:
            a = xm
            fa = fm
        if abs(dm) < xtol + rtol*abs(xm):
            solver_stats['converged'] = True
            solver_stats['niters'] = i+1
            solver_stats['x'] = a  # Last point where f(x) != 0
            solver_stats['f'] = fa
            return solver_stats

    # Not converged
    solver_stats['converged'] = False
    solver_stats['niters'] = i+1
    solver_stats['x'] = a
    solver_stats['f'] = fa
    
    return solver_stats
