import numpy as np


def classificate_signal(parents, children, labels=None, nc_val=-1):
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
            products[ic] *= 1 - grawadile.estimators.ssim(p, c)

    losses = products + discards
    label = np.argmin(losses)

    # If all losses are >= 1 then all are discarded.
    if losses[label] >= 1:
        print('lost by all lossers')
        label = nc_val
    # If there are multiple minimums, mark it as not classified.
    elif np.sum(losses == losses[label]) > 1:
        print('lost by multiple minimums:', losses)
        label = nc_val

    if labels is not None:
    	label = labels[label]
    
    return label