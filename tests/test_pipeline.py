"""Check pipeline argument forwarding and return packaging with recording stand-ins.

These tests do not verify numerical denoising or classification with real models.
"""

import numpy as np
import pytest

from clawdia.pipeline import Pipeline


class DenoiserLog:
    """Log denoising calls and add a visible offset instead of denoising.

    Four dummy atoms, length two by default, follow DictionarySpams' row-atom
    layout. Their count exceeds the classifier's default atom length, exposing
    confusion between atom count and length. Only their shape is used; adding
    10 makes preprocessing visible without simulating denoising.
    """

    def __init__(self, atom_length=2):
        self.components = np.zeros((4, atom_length))
        self.calls = []

    def reconstruct_minibatch(self, signals, **kwargs):
        self.calls.append((signals.copy(), kwargs))
        return signals + 10


class ClassifierLog:
    """Log classification calls and return fixed labels and optional losses.

    Two dummy atoms, length three by default, follow LRSDL's column-atom
    layout; only their shape is used. The chosen arbitrary outputs identify
    what the pipeline should return unchanged.
    """

    def __init__(self, atom_length=3):
        self.D = np.zeros((atom_length, 2))
        self.calls = []

    def predict(self, signals, with_losses=False, **kwargs):
        self.calls.append((signals.copy(), with_losses, kwargs))
        predictions = np.arange(signals.shape[0])
        if with_losses:
            return predictions, np.linspace(0.1, 0.2, signals.shape[0])
        return predictions


@pytest.mark.parametrize("with_losses", [False, True])
@pytest.mark.parametrize("with_preprocessed", [False, True])
def test_pipeline_forwards_orientation_parameters_and_return_options(
    with_losses, with_preprocessed
):
    """Check forwarded arguments and all four combinations of optional outputs."""
    denoiser = DenoiserLog()
    classifier = ClassifierLog()
    pipeline = Pipeline(
        dico_den=denoiser,
        dico_den_params={"sc_lambda": 0.2},
        dico_clas=classifier,
        dico_clas_params={"threshold": 0.4},
    )
    # Four signals with three samples each make the two axes distinguishable.
    strains = np.arange(12, dtype=float).reshape(4, 3)

    result = pipeline(
        strains,
        with_losses=with_losses,
        with_preprocessed=with_preprocessed,
    )

    denoiser_input, denoiser_kwargs = denoiser.calls[0]
    # Both public model methods expect one signal per row.
    np.testing.assert_array_equal(denoiser_input, strains)
    assert denoiser_kwargs == {"normed": True, "sc_lambda": 0.2}

    # Adding 10 makes preprocessing visible.
    expected_preprocessed = strains + 10
    classifier_input, recorded_losses, classifier_kwargs = classifier.calls[0]
    np.testing.assert_array_equal(classifier_input, expected_preprocessed)
    assert recorded_losses is with_losses
    assert classifier_kwargs == {"threshold": 0.4}

    if not with_losses and not with_preprocessed:
        np.testing.assert_array_equal(result, np.arange(4))
    else:
        assert isinstance(result, tuple)
        np.testing.assert_array_equal(result[0], np.arange(4))
        if with_losses:
            np.testing.assert_allclose(
                result[1], np.linspace(0.1, 0.2, 4), rtol=0, atol=0
            )
        if with_preprocessed:
            np.testing.assert_array_equal(result[-1], expected_preprocessed)


def test_pipeline_rejects_longer_denoising_atoms():
    """Reject denoising atoms of length four against classifier atoms of length three."""
    with pytest.raises(ValueError, match="must be shorter"):
        Pipeline(
            dico_den=DenoiserLog(atom_length=4),
            dico_den_params={},
            dico_clas=ClassifierLog(atom_length=3),
            dico_clas_params={},
        )
