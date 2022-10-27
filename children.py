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
    n_signals = signals.shape[1]
    mask_null = np.all(signals==0, axis=0)
    n_null = np.sum(mask_null)

    # Ommit null signals if any.
    if n_null > 0:
        mask_nonull = ~mask_null
        signals_ = signals[:,mask_nonull]
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


def pick_children_batch(parents, dictionaries, labels=None, out=None, verbose=False, **kwargs):
    """TODO
    Troba els indexs `i_atoms` amb els seus coeficients `c_atoms` dels àtoms
    de `dictionary` més pareguts a cada pare en `parents` gastant l'OMP.
    Aquells pares per als que l'OMP no convergisca tindràn coeficient
    `c_atoms[i] = 0`.

    parents: 3d-array (l_window, n_dictionaries, n_signals)
    dictionaries: dict( key: 2d-array(l_window, n_atoms) )
    labels: dict( key: int )
        Si s'introdueix, es gastarà per mapejar l'ordre en que s'introdueixen
        els children de cada diccionari dins `i_children`. Si no,
        s'introdueixen amb l'ordre que l'interpret de python esculla.
    out: 4d-array (2, n_dictionaries, n_dictionaries, n_signals), optional
        Eixida inplace opcional.
    **kwargs:
        Passat a la funció `sklearn.decomposition.sparse_encode`.

    """
    l_window, n_dicos, n_signals = parents.shape
    assert n_dicos == len(dictionaries)

    parents_flat = parents.reshape(l_window, -1, order='F')

    if labels is None:
        labels = {key: i for i, key in enumerate(dictionaries)}

    if out is None:
        i_children = np.empty((2, n_dicos, n_dicos, n_signals), order='F')
    else:
        i_children = out

    for kdc, dico in dictionaries.items():
        idc = labels[kdc]
        i_atoms, c_atoms = omp_singlematch_batch(parents_flat, dico, **kwargs)
        i_children[:,idc] = [
            i_atoms.reshape(n_dicos, -1, order='F'),
            c_atoms.reshape(n_dicos, -1, order='F')
        ]
        if verbose:
            _lost = np.sum(c_atoms==0)
            print(f"Children generated with '{kdc}': {len(c_atoms)} (lost: {_lost})")

    return i_children


def pick_children_autolambda_batch(parents, lambdas, dictionaries_set, labels=None, verbose=False, **kwargs):
    """TODO

    Torna els índex de cada senyal dins `dictionaries` que més s'assembla a
    cadascún dels `parents` tenint en compte la lambda (dins `lambdas`) a les
    que s'han reconstruit.

    PARAMETERS
    ----------
    parents: array, shape (length, no. labels, no. signals)

    lambdas: array, shape (no. labels, no. signals)

    dictionaries_set: dict
        'lambdas': array-like
            List of lambdas at with each dictionary in 'dictionaries' was
            reconstructed.
        'dictionaries': array-like
            List of `dict()` of dictionaries, so that each value is a
            dict(label: 2d-array(length, no. atoms)).

    labels: dict( label: int )
        If given, will be the order of the dictionaries (within each lambda)
        in which each children's coordinate will be assigned.
        If None, the order will be the default when accessing each 
        dictionaries[lambda].

    **kwargs:
        Passed to `sklearn.decomposition.sparse_encode`.

    RETURNS
    -------
    i_children: 2d-array, shape (2, no. labels, no. labels, no. signals)
        Each pair of values in `index=0` is the coordinates of the selected
        atom, (labels[wf], ind. atom)
    
    """
    l_window, n_labels, n_signals = parents.shape
    assert n_labels == len(labels)

    parents_flat = parents.reshape(l_window, -1, order='F')

    if labels is None:
        labels = {key: i for i, key in enumerate(dictionaries)}

    if out is None:
        i_children = np.empty((2, n_labels, n_labels, n_signals), order='F')
    else:
        i_children = out

    for kdc, dico in dictionaries.items():
        idc = labels[kdc]
        i_atoms, c_atoms = omp_singlematch_batch(parents_flat, dico, **kwargs)
        i_children[:,idc] = [
            i_atoms.reshape(n_labels, -1, order='F'),
            c_atoms.reshape(n_labels, -1, order='F')
        ]
        if verbose:
            _lost = np.sum(c_atoms==0)
            print(f"Children generated with '{kdc}': {len(c_atoms)} (lost: {_lost})")

    return i_children
