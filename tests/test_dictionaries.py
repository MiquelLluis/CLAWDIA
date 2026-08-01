import copy

import numpy as np
import pytest

from clawdia import dictionaries


SPAMS_INITIAL = np.array(
    [[1.0, 0.0], [0.0, 1.0], [2**-0.5, 2**-0.5]]
)
SPAMS_SIGNAL_POOL = np.array(
    [
        [1.0, 0.2, 0.8, 0.4, 0.6, 0.3],
        [0.1, 1.0, 0.3, 0.9, 0.5, 0.7],
    ]
)
SPAMS_TRAINING_PATCHES = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.6, 0.4],
        [0.4, 0.6],
    ]
)
LRSDL_TRAINING_SIGNALS = np.array(
    [
        [1.0, 0.0, 0.5, 0.0, 0.25, 0.0],
        [0.8, 0.0, 0.4, 0.0, 0.2, 0.0],
        [0.0, 1.0, 0.0, 0.5, 0.0, 0.25],
        [0.0, 0.8, 0.0, 0.4, 0.0, 0.2],
    ]
)
LRSDL_TRAINING_LABELS = np.array([1, 1, 2, 2])


def _make_spams(initialisation):
    kwargs = {
        "lambda1": 0.1,
        "batch_size": 2,
        "n_iter": 2,
        "identifier": f"round-trip-{initialisation}",
    }
    if initialisation == "explicit":
        return dictionaries.DictionarySpams(
            dict_init=SPAMS_INITIAL.copy(),
            **kwargs,
        )
    return dictionaries.DictionarySpams(
        signal_pool=SPAMS_SIGNAL_POOL.copy(),
        a_length=2,
        d_size=3,
        random_state=17,
        **kwargs,
    )


def _make_lrsdl(k0):
    return dictionaries.DictionaryLRSDL(
        lambd=0.02,
        lambd2=0.03,
        eta=0.0002,
        k=1,
        k0=k0,
        updateX_iters=2,
        updateD_iters=2,
    )


def _assert_state_value_equal(expected, actual):
    assert type(actual) is type(expected)
    if isinstance(expected, np.ndarray):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert actual.flags.c_contiguous == expected.flags.c_contiguous
        assert actual.flags.f_contiguous == expected.flags.f_contiguous
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_state_value_equal(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for expected_item, actual_item in zip(expected, actual):
            _assert_state_value_equal(expected_item, actual_item)
    else:
        assert actual == expected


def _assert_attribute_state_equal(expected, actual):
    assert actual.keys() == expected.keys()
    for attribute in expected:
        _assert_state_value_equal(expected[attribute], actual[attribute])


def _round_trip(model, destination):
    before_save = copy.deepcopy(vars(model))
    dictionaries.save(destination, model)
    _assert_attribute_state_equal(before_save, vars(model))

    restored = dictionaries.load(destination)
    assert type(restored) is type(model)
    _assert_attribute_state_equal(vars(model), vars(restored))
    return restored


@pytest.mark.parametrize("initialisation", ["explicit", "signal_pool"])
def test_untrained_spams_round_trip_preserves_initialised_state(
    tmp_path, initialisation
):
    """Preserve untrained SPAMS state from both supported initialisation paths."""
    model = _make_spams(initialisation)
    restored = _round_trip(model, tmp_path / f"spams-{initialisation}.npz")

    assert restored.trained is False
    assert restored.model is None
    assert restored.lambda1 == 0.1
    assert restored.n_iter == 2
    assert restored.n_train is None
    assert restored.t_train is None


@pytest.mark.integration
@pytest.mark.parametrize("initialisation", ["explicit", "signal_pool"])
def test_trained_spams_round_trip_preserves_usable_continuation_state(
    tmp_path, initialisation
):
    """Preserve real SPAMS training state and use it for a warm start."""
    model = _make_spams(initialisation)
    model.train(SPAMS_TRAINING_PATCHES, verbose=False, threads=1)

    restored = _round_trip(
        model, tmp_path / f"spams-trained-{initialisation}.npz"
    )

    assert restored.trained is True
    assert restored.n_train == len(SPAMS_TRAINING_PATCHES)
    assert restored.t_train > 0
    assert restored.model.keys() == {"A", "B", "iter"}

    previous_iterations = restored.n_iter
    restored.train(
        SPAMS_TRAINING_PATCHES,
        n_iter=1,
        warm_start=True,
        verbose=False,
        threads=1,
    )
    assert restored.n_iter == previous_iterations + 1
    assert restored.model["iter"] == previous_iterations + 1


@pytest.mark.parametrize("k0", [0, 1])
def test_untrained_lrsdl_round_trip_preserves_initialised_state(tmp_path, k0):
    """Preserve LRSDL parameters before fitting, with and without shared atoms."""
    model = _make_lrsdl(k0)
    restored = _round_trip(model, tmp_path / f"lrsdl-k0-{k0}.npz")

    assert restored.t_train is None
    for attribute in ("D", "D0", "X", "X0", "Y", "D_range", "Y_range"):
        assert getattr(restored, attribute) is None


@pytest.mark.integration
@pytest.mark.parametrize("k0", [0, 1])
def test_trained_lrsdl_round_trip_preserves_fitted_state_and_predictions(
    tmp_path, k0
):
    """Preserve fitted LRSDL state and predictions for both dictionary forms."""
    model = _make_lrsdl(k0)
    model.fit(
        LRSDL_TRAINING_SIGNALS,
        y_true=LRSDL_TRAINING_LABELS,
        l_atoms=2,
        iterations=1,
        step=2,
        random_seed=7,
    )
    predictions, losses = model.predict(
        LRSDL_TRAINING_SIGNALS, with_losses=True
    )

    restored = _round_trip(model, tmp_path / f"lrsdl-trained-k0-{k0}.npz")
    restored_predictions, restored_losses = restored.predict(
        LRSDL_TRAINING_SIGNALS, with_losses=True
    )

    assert restored.t_train > 0
    assert isinstance(restored.D_range, list)
    assert isinstance(restored.Y_range, list)
    np.testing.assert_array_equal(restored_predictions, predictions)
    np.testing.assert_allclose(
        restored_losses, losses, rtol=0, atol=1e-12
    )
