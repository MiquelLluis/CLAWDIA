import numpy as np

from . import estimators


def classificate_tree(parents, children, labels=None, nc_val=-1):
    """TODO

    Quan tinga clar el procediment comentar-lo detalladament.

    PARAMETERS
    ----------
    parents: array_like, (samples, n_parents)
        Parent waveforms whose indices coincide to their respective morphological
        families (labels). Each parent waveform will have associated
        `n_parents` children.
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
            products[ic] *= grawadile.estimators.issim(p, c)

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


def classificate_batch_indexed(parents, i_children, dataset, labels=None, **kwargs):
    """TODO

    Classifica un conjunt de senyals (conjunt d'arbres parents-children) donat
    els seus parents i els índexs dels children apuntant al dataset.

    parents: 3d-array(l_signals, n_labels, n_signals)
    i_children: 3d-array(n_labels, n_labels, n_signals)
        I.e. un índex referent al dataset, per cada label de children per cada
        label de parents per cada senyal.
    dataset: dict( key: 2d-array(l_signals, n_signals) )
        Nota: Les key no tenen per què ser igual a les labels, però en
        aquest cas s'assumirà que l'ordre de `dataset.values()` es correspòn
        amb el de `labels`.
    labels: array_like, optional
        Label names sorted. If None, `dataset.keys()` will be used instead.
        S'ha d'acomplir `len(labels) == dataset.shape[1]`.
    **kwargs: arguments de `classificate_tree`.

    """
    l_signals, n_labels, n_signals = parents.shape
    if labels is None:
        labels = np.asarray(dataset.keys())

    # Children tree for each signal.
    children_isi = np.empty((l_signals, n_labels, n_labels), order='F')
    # Classification results.
    y_labels = np.empty(n_signals, dtype=labels.dtype)

    for isi in range(n_signals):
        parents_isi = parents[...,isi]
        # Reconstruct selected children's tree:
        for i, data in enumerate(dataset.values()):
            mask = i_children[i,:,isi]
            children_isi[:,i,:] = data[:,mask]

        y_labels[isi] = classificate_tree(parents_isi, children_isi, labels=labels, **kwargs)

    return y_labels
