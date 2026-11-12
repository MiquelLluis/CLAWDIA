"""Check pipeline argument forwarding and return packaging with recording stand-ins.

These tests do not verify numerical denoising or classification with real models.
"""

import numpy as np
import pytest

from clawdia.pipeline import Pipeline


class RecordingDenoiser:
    """Record denoising calls and add a visible offset instead of denoising."""

    def __init__(self, atom_length=2):
        self.components = np.zeros((4, atom_length))
        self.calls = []

    def reconstruct_minibatch(self, signals, **kwargs):
        self.calls.append((signals.copy(), kwargs))
        return signals + 10


class RecordingClassifier:
    """Record classification calls and return fixed labels and optional losses."""

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
    denoiser = RecordingDenoiser()
    classifier = RecordingClassifier()
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
    # This preserves the current transposition, not a verified model contract.
    np.testing.assert_array_equal(denoiser_input, strains)
    assert denoiser_kwargs == {"normed": True, "sc_lambda": 0.2}

    # Adding 10 reveals whether preprocessing reaches the classifier, but is
    # insensitive to signal orientation and therefore misses the row-axis bug.
    expected_preprocessed = strains + 10
    classifier_input, recorded_losses, classifier_kwargs = classifier.calls[0]
    np.testing.assert_array_equal(classifier_input, expected_preprocessed)
    assert recorded_losses is with_losses
    assert classifier_kwargs == {"threshold": 0.4}

    # Predictions alone are an array; optional losses and preprocessed signals
    # are returned alongside them in a tuple, in that order.
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
    """Check constructor rejection with a nominally longer denoising atom."""
    # This shares the stand-in's incorrect SPAMS layout, so it does not yet
    # establish that the real dictionaries' atom lengths are compared correctly.
    with pytest.raises(ValueError, match="must be shorter"):
        Pipeline(
            dico_den=RecordingDenoiser(atom_length=4),
            dico_den_params={},
            dico_clas=RecordingClassifier(atom_length=3),
            dico_clas_params={},
        )
