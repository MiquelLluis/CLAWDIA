import warnings

import numpy as np
import sklearn


def _omp_singlematch_batch(signals, dictionary, **kwargs):
    """TODO
    Troba els indexs `i_atoms` amb els seus coeficients `c_atoms` dels àtoms
    de `dictionary` més pareguts a cada senyal en `signals` gastant l'OMP.
    Compte: Aquesta versió força un únic àtom per senyal i diccionari.
    Aquells senyals per als que l'OMP no convergisca tindràn coeficient
    `c_atoms[i] = 0`.

    """
    codes = sklearn.decomposition.sparse_encode(
        signals.T,
        dictionary.T,
        algorithm='omp',
        n_nonzero_coefs=1,
        **kwargs
    ).T
    i_atoms = np.argmax(np.abs(codes), axis=0)
    c_atoms = np.ravel(codes[i_atoms,np.indices(i_atoms.shape)])

    return i_atoms, c_atoms


def omp_singlematch_batch(signals, dictionary, **kwargs):
    """TODO
    Troba els indexs `i_atoms` amb els seus coeficients `c_atoms` dels àtoms
    de `dictionary` més pareguts a cada senyal en `signals` gastant l'OMP.
    Compte: Aquesta versió requereix un únic àtom per senyal i diccionari.
    Aquells senyals per als que l'OMP no convergisca tindràn coeficient
    `c_atoms[i] = 0`.

    """
    n_signals = len(signals)
    mask_null = np.all(signals==0, axis=0, keepdims=True)
    n_null = np.sum(mask_null)

    # Ommit null signals if any.
    if n_null > 0:
        i_null = mask_null.nonzero()
        mask_nonull = ~mask_null
        signals_ = signals[mask_nonull]
    else:
        signals_ = signals

    if n_null < n_signals:
        i_atoms, c_atoms = _omp_singlematch_batch(signals_, dictionary, **kwargs)

        if n_null > 0:
            i_atoms_partial = i_atoms
            c_atoms_partial = c_atoms
            i_atoms = np.zeros(n_signals)
            c_atoms = np.zeros_like(i_atoms)
            i_atoms[mask_nonull] = i_atoms_partial
            c_atoms[mask_nonull] = c_atoms_partial
    
    # All null (zeros) signals.
    else:
        i_atoms = np.zeros(n_signals)
        c_atoms = np.zeros_like(i_atoms)

    return i_atoms, c_atoms


def gen_index_children_batch(parents, dictionaries, dict_order=None, out=None, verbose=False, **kwargs):
    """TODO
    Troba els indexs `i_atoms` amb els seus coeficients `c_atoms` dels àtoms
    de `dictionary` més pareguts a cada pare en `parents` gastant l'OMP.
    Aquells pares per als que l'OMP no convergisca tindràn coeficient
    `c_atoms[i] = 0`.

    parents: 3d-array (l_window, n_dictionaries, n_signals)
    dictionaries: dict( key: 2d-array(l_window, n_atoms) )
    dict_order: dict( key: int )
        Si s'introdueix, es gastarà per mapejar l'ordre en que s'introdueixen
        els children de cada diccionari dins `i_children`. Si no,
        s'introdueixen amb l'ordre que l'interpret de python esculla.
    **kwargs:
        Passat a la funció `sklearn.decomposition.sparse_encode`.

    """
    l_window, n_dicos, n_signals = parents.shape
    assert n_dicos == len(dictionaries)

    parents_flat = parents.reshape(l_window, -1, order='F')

    if dict_order is None:
        dict_order = {key: i for i, key in enumerate(dictionaries)}

    if out is None:
        i_children = np.empty((2, n_dicos, n_signals), order='F')
    else:
        i_children = out

    for kdc, dico in dictionaries.items():
        idc = dict_order[kdc]
        i_atoms, c_atoms = omp_singlematch_batch(parents_flat, dico, **kwargs)
        i_children[:,idc] = [
            i_atoms.reshape(n_dicos, -1, order='F'),
            c_atoms.reshape(n_dicos, -1, order='F')
        ]
        if verbose:
            _lost = np.sum(c_atoms==0)
            print(f"Children generated with '{kdc}': {len(c_atoms)} (lost: {_lost})")

    return i_children