import itertools as it

import numpy as np
import sklearn


def _omp_singlematch(signal, dictionary, kwargs_omp):
    """TODO
    Torna l'índex `i_atom` amb el seu coeficient `c_atom` de l'àtom
    de `dictionary` més paregut al senyal `signal` gastant l'OMP.
    Compte: Aquesta versió força un únic àtom.
    Aquells senyals per als que l'OMP no convergisca tindràn coeficient
    `c_atoms[i] = 0`.

    """
    signal = signal[None,:]
    code = sklearn.decomposition.sparse_encode(
        signal,
        dictionary.T,
        algorithm='omp',
        n_nonzero_coefs=1,
        **kwargs_omp
    ).ravel()
    i_atom = np.argmax(np.abs(code))
    c_atom = code[i_atom]

    return i_atom, c_atom


def _omp_singlematch_batch(signals, dictionary, kwargs_omp):
    """TODO

    """
    codes = sklearn.decomposition.sparse_encode(
        signals.T,
        dictionary.T,
        algorithm='omp',
        n_nonzero_coefs=1,
        **kwargs_omp
    ).T
    i_atoms = np.argmax(np.abs(codes), axis=0)
    c_atoms = np.ravel(codes[i_atoms,np.indices(i_atoms.shape)])

    return i_atoms, c_atoms


def omp_singlematch_batch(signals, dictionary, **kwargs_omp):
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
        i_atoms, c_atoms = _omp_singlematch_batch(signals_, dictionary, kwargs_omp)

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


def _pick_children_from_parent(parent, dictionaries, kwargs_omp):
    """TODO

    """
    pass


def pick_children_batch(parents, dictionaries, labels=None, out=None, verbose=False, **kwargs_omp):
    """TODO
    
    Torna els index de l'arbre children corresponent a cada parent en `parents`
    i diccionari en `dictionaries`. Cada index apunta a un àtom del diccionari
    corresponent.

    parents: 3d-array (l_window, n_dictionaries, n_signals)
    dictionaries: dict( key: 2d-array(l_window, n_atoms) )
    labels: dict( key: int )
        Si s'introdueix, es gastarà per mapejar l'ordre en que s'introdueixen
        els children de cada diccionari dins `i_children`. Si no,
        s'introdueixen amb l'ordre que l'interpret de python esculla.
    out: 4d-array (2, n_dictionaries, n_dictionaries, n_signals), optional
        Eixida inplace opcional.
    **kwargs_omp:
        Passed to OMP's method of `sklearn.decomposition.sparse_encode`.

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
        i_atoms, c_atoms = omp_singlematch_batch(parents_flat, dico, **kwargs_omp)
        i_children[:,idc] = [
            i_atoms.reshape(n_dicos, -1, order='F'),
            c_atoms.reshape(n_dicos, -1, order='F')
        ]
        if verbose:
            _lost = np.sum(c_atoms==0)
            print(f"Children generated with '{kdc}': {len(c_atoms)} (lost: {_lost})")

    return i_children


def pick_children_autolambda_batch(parents, lambdas, dictionaries_set, **kwargs_omp):
    """TODO

    Torna els índex de cada senyal dins `dictionaries` que més s'assembla a
    cadascún dels `parents` tenint en compte la lambda (dins `lambdas`) a les
    que s'han reconstruit.

    PARAMETERS
    ----------
    parents: array, shape (length, no. labels, no. signals)

    lambdas: array, shape (no. labels, no. signals)

    dictionaries_set: dict
        'dictionaries': array-like
            List of `dict()` of dictionaries, so that each value is a
            dict(label: 2d-array(length, no. atoms)).
        'lambdas': array-like
            List of lambdas at with each dictionary set in 'dictionaries' was
            reconstructed.

    labels: dict( label: int )
        If given, will be the order of the dictionaries (within each lambda)
        in which each children's coordinate will be assigned.
        If None, the order will be the default when accessing each 
        dictionaries[lambda].

    **kwargs_omp:
        Passed to OMP's method of `sklearn.decomposition.sparse_encode`.

    RETURNS
    -------
    i_children: 3d-array, shape (no. labels, no. labels, no. signals)
        Each value is the index of the selected atom inside its corresponding
        dictionary, indicated by the lambda of its parent in `i_dicset`.
    i_dicset: 2d-array, shape (no.labels, no. signals)
        The index of the set of dictionaries inside `dictionaries` produced
        with the closest lambda of reconstruction to each parent in `parents`,
        which is indicated in `lambdas`.
    
    """
    l_window, n_labels, n_signals = parents.shape
    # if labels is None:
    #     labels = {key: i for i, key in enumerate(dictionaries[0])}
    # assert n_labels == len(labels)

    dictionaries = dictionaries_set['dictionaries']
    lambdas_dict = dictionaries_set['lambdas']

    parents_flat = parents.reshape(l_window, -1, order='F')
    lambdas_flat = lambdas.ravel(order='F')
    n_parents = parents_flat.shape[1]

    # Will be reshaped afterwards
    i_children = np.empty((n_labels, n_parents), order='F')
    i_dicset = np.empty(n_parents)

    for ip in range(n_parents):
        parent = parents_flat[ip]
        # Map each parent to its corresponding set of dictionaries which were made
        # with the closest lambda of reconstruction.
        lambda_parent = lambdas_flat[ip]
        i_dicos = np.argmin(np.abs(lambdas_dict - lambda_parent))
        # Current set of dictionaries to use whose lambda of reconstruction is
        # closest to the parent.
        dicos_clas = dictionaries[i_dicos]

        i_children[:,ip] = _pick_children_from_parent(parent, dicos_clas, kwargs_omp)
        i_dicset[ip] = i_dicos

    return i_children
