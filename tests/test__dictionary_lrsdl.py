import numpy as np
import pytest

from clawdia import dictionaries
from clawdia.dictionaries import DictionaryLRSDL


@pytest.fixture(scope="module")
def reference_population():
    n_samples = 100
    n_features = 20
    samples_per_class = n_samples // 2
    rng = np.random.default_rng(1048596)
    times = np.linspace(0, 1, n_features)

    signals = np.ones((n_samples, n_features), dtype=float)
    for i in range(samples_per_class):
        frequency = rng.uniform(2, 5)
        signals[i] *= np.sin(frequency * 2 * np.pi * times)
    for i in range(samples_per_class, n_samples):
        frequency = rng.uniform(5, 8)
        signals[i] *= np.sin(frequency * 2 * np.pi * times)
    labels = np.repeat([1, 2], samples_per_class)
    return signals, labels


@pytest.fixture(scope="module")
def reference_model(data_dir):
    return dictionaries.load(
        data_dir / "_dictionary_lrsdl" / "LRSDL_reference_model.npz"
    )


@pytest.fixture
def controlled_model():
    model = DictionaryLRSDL(lambd=0.01, k=1, k0=0)
    model.D = np.eye(2)
    model.D0 = np.empty((2, 0))
    model.D_range = np.array([0, 1, 2])
    model.num_classes = 2
    return model


def small_training_data():
    signals = np.array(
        [
            [1.0, 0.0, 0.5, 0.0, 0.25, 0.0],
            [0.8, 0.0, 0.4, 0.0, 0.2, 0.0],
            [0.0, 1.0, 0.0, 0.5, 0.0, 0.25],
            [0.0, 0.8, 0.0, 0.4, 0.0, 0.2],
        ]
    )
    labels = np.array([1, 1, 2, 2])
    return signals, labels


# -----------------------------------------------------------------------------
# Parameterized input validation
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("invalid_y_true", [
    [1, 1, 2],  # List instead of ndarray
    np.array([1.5, 2.0, 1.0]),  # Float labels
    np.array([0, 0, 1])  # Labels starting at 0
], ids=["list_labels", "float_labels", "zero_based_labels"])
def test_fit_input_validation(invalid_y_true, default_model):
    """Verify proper error handling for invalid y_true configurations."""
    X = np.random.randn(len(invalid_y_true), 20)
    
    with pytest.raises((TypeError, ValueError)):
        default_model.fit(X, y_true=invalid_y_true, l_atoms=20, iterations=10)


# -----------------------------------------------------------------------------
# Windowing & filtering tests
# -----------------------------------------------------------------------------

def test_window_generation(reproducibility_data, default_model):
    """Verify window extraction logic with different step sizes."""
    X, y_true = reproducibility_data
    l_atoms = 15
    
    # Test non-overlapping windows
    default_model.fit(X, y_true=y_true, l_atoms=l_atoms, step=l_atoms, iterations=1)
    expected_windows = (X.shape[1] - l_atoms) // l_atoms + 1
    assert default_model.Y.shape[1] == expected_windows * X.shape[0]

    # Test overlapping windows
    default_model.fit(X, y_true=y_true, l_atoms=l_atoms, step=5, iterations=1)
    expected_windows = (X.shape[1] - l_atoms) // 5 + 1
    assert default_model.Y.shape[1] == expected_windows * X.shape[0]


def test_threshold_filtering(reproducibility_data, default_model, training_config):
    """Verify threshold-based filtering removes low-energy windows."""
    X, y_true = reproducibility_data

    # Set threshold to maximum to trigger the exception
    training_config_copy = training_config.copy()
    training_config_copy['threshold'] = 1
    
    with pytest.raises(ValueError) as excinfo:
        default_model.fit(X, y_true=y_true, **training_config_copy)
    
    assert "not enough training samples" in str(excinfo.value)


# -----------------------------------------------------------------------------
# Post-training state checks
# -----------------------------------------------------------------------------

def test_post_training_attributes(trained_model):
    """Verify essential attributes are set after training."""
    assert trained_model.t_train > 0
    n_classes = len(trained_model.D_range) - 1
    assert trained_model.D.shape[1] == trained_model.k * n_classes
    assert trained_model.D0.shape[1] == trained_model.k0
    assert hasattr(trained_model, 'X') and trained_model.X is not None


@pytest.mark.integration
@pytest.mark.regression
def test_seeded_training_matches_reference_model(
    reference_population, reference_model
):
    signals, labels = reference_population
    model = DictionaryLRSDL(
        lambd=0.01,
        lambd2=0.01,
        eta=0.0001,
        k=4,
        k0=4,
        updateX_iters=100,
        updateD_iters=100,
    )
    with pytest.warns(RuntimeWarning, match="last 5 bins"):
        model.fit(
            signals,
            y_true=labels,
            l_atoms=15,
            iterations=100,
            random_seed=1048596,
            step=20,
            threshold=0,
        )

    assert model.t_train > 0
    np.testing.assert_allclose(model.D, reference_model.D, rtol=0, atol=1e-6)
    np.testing.assert_allclose(model.D0, reference_model.D0, rtol=0, atol=1e-6)


@pytest.mark.integration
@pytest.mark.regression
def test_reference_model_classifies_separated_frequency_population(
    reference_model, data_dir
):
    """Reproduce the reference script's third seeded population and predictions."""
    rng = np.random.default_rng(1048596)
    n_samples = 100
    n_features = 20
    samples_per_class = 50
    times = np.linspace(0, 1, n_features)

    def generate():
        signals = np.ones((n_samples, n_features), dtype=float)
        for i in range(samples_per_class):
            frequency = rng.uniform(2, 5)
            signals[i] *= np.sin(frequency * 2 * np.pi * times)
        for i in range(samples_per_class, n_samples):
            frequency = rng.uniform(5, 8)
            signals[i] *= np.sin(frequency * 2 * np.pi * times)
        return signals

    # The reference script generated training and validation populations before
    # recording predictions for the third population from the same RNG instance.
    generate()
    generate()
    test_signals = generate()
    original = test_signals.copy()
    labels = np.repeat([1, 2], samples_per_class)
    expected = np.loadtxt(
        data_dir / "_dictionary_lrsdl" / "LRSDL_reference_model_test.txt"
    )

    predictions, losses = reference_model.predict(
        test_signals, with_losses=True
    )

    assert np.mean(predictions == labels) >= 0.95
    np.testing.assert_array_equal(predictions, expected[:, 0])
    np.testing.assert_allclose(losses, expected[:, 1], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(test_signals, original)
