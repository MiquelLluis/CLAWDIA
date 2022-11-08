import numpy as np

from . import estimators


def classificate_tree(parents, children, labels=None, nc_val=-1):
    """TODO

    Quan tinga clar el procediment comentar-lo detalladament.

    PARAMETERS
    ----------
    parents: array_like, (samples, n_parents)
        Parent waveforms whose indices coincide to their respective
        morphological families (labels). Each parent waveform will have
        associated `n_parents` children.
    children: array_like, (samples, n_children, n_parents)
        Reconstructions associated with parent waveforms.
    labels: array_like, optional
		Label names sorted. If None, will default to `range(n_parents)`.
    nc_val: int, -1 by default
        Value assigned to the non classified signals.

    RETURNS
    -------
    label: int
        Index of the most fitting morphological family (label).

    """
    n_labels = parents.shape[1]
    products = np.ones(n_labels)
    discards = np.zeros(n_labels)

    for ip in range(n_labels):
        p = parents[:,ip]
        # If the denoising dictionary 'wf' was not able to reconstruct it
        # we assume impossible for the signal to belong to its morphology.
        if not p.any():
            discards[ip] = 1
            continue
        for ic in range(n_labels):
            c = children[:,ic,ip]
            products[ic] *= estimators.issim(p, c)

    losses = products + discards
    i_label = np.argmin(losses)

    # If all losses are >= 1 then all are discarded.
    if losses[i_label] >= 1:
        i_label = nc_val
    # If there are multiple minimums, mark it as not classified.
    elif np.sum(losses == losses[i_label]) > 1:
        i_label = nc_val

    if labels is not None:
    	label = labels[i_label]
    else:
    	label = i_label
    
    return label


def classificate_batch_indexed(parents, indices, dictionaries, labels, nc_val=-1):
    """TODO

    Classifica un conjunt de senyals (conjunt d'arbres parents-children) donat
    els seus parents i els índexs dels children apuntant als diccionaris i tal.

    parents: 3d-array(l_signals, n_labels, n_signals)
    indices: dict
        'atoms':  3d-array (n_labels, n_labels, n_signals)
            I.e. un índex referent al dataset per cada label de children per cada
            label de parents per cada senyal a classificar.
        'dictionaries': 2d-array (n_labels, n_signals)
            I.e. un índex referent a l'àtom dins el seu corresponent dataset
            indicat a `indices['dictionaries']` per cada label de parent per
            cada senyal a classificar.
    labels: iterable
        Iterable-like list of labels.
    nc_val: int, -1 by default
        Value assigned to the non classified signals.

    """
    l_signals, n_labels, n_signals = parents.shape
    children_i = np.empty((l_signals, n_labels, n_labels), order='F')  # For each signal
    i_dicset = indices['i_dicset']
    i_children = indices['i_children']

    y_pred = np.empty(n_signals, dtype=int)
    for isi in range(n_signals):
        parents_i = parents[...,isi]
        # dataset = _build_dataset_from_dictionaries(i_dicset[:,isi], dictionaries, labels)
        # _rebuild_children_tree_inplace(i_children[...,isi], dataset, out=children_i)
        for ip, p_lab in enumerate(labels):
            pp = parents_i[:,ip]
            i_dic_set_p = i_dicset[ip,isi]
            dicos = dictionaries[i_dic_set_p]
            i_atoms_children = i_children[:,ip,isi]
            for ic, c_lab in enumerate(labels):
                dico = dicos[c_lab]
                i_child = i_atoms_children[ic]
                children_i[:,ic,ip] = dico[:,i_child]


        y_pred[i] = classificate_tree(parents_i, children_i, nc_val=nc_val)

    return y_pred


# def _build_dataset_from_dictionaries(indices, dictionaries, labels):
#     dataset = {}
#     # For each parent
#     for ilab, lab in enumerate(labels):
#         i_dicset = indices[ilab]
#         dataset[lab] = dictionaries[i_dicset][lab]

#     return dataset


# def _rebuild_children_tree_inplace(indices, dataset, out=None):
#     if out is None:
#         l, n = next(iter(dataset.values())).shape
#         out = np.empty((l, n, n), order='F')

#     # For each parent
#     for ip, p_lab in enumerate(dataset):
#         # For each child
#         for ic, c_lab in enumerate(dataset):
#             ic_dataset = indices[ic,ip]
#             out[:,ic,ip] = dataset[p_lab][:,ic_dataset]

#     return out