import numpy as np
import pytest

import clawdia
from clawdia.dictionaries import DictionaryLRSDL


# -----------------------------------------------------------------------------
# Fixtures with module scope
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def default_model_params():
    """Shared parameters for all tests using default model configuration."""
    return {
        'lambd': 0.01,
        'lambd2': 0.01,
        'eta': 0.0001,
        'k': 4,
        'k0': 4,
        'updateX_iters': 100,
        'updateD_iters': 100
    }


@pytest.fixture(scope="module")
def reproducibility_data():
    """Fixed dataset for reproducibility testing (module-scoped)."""
    ns = 100  # samples (signals)
    nf = 20   # features
    spc = ns // 2  # samples per class
    rng = np.random.default_rng(1048596)  # Fixed seed for reproducibility

    X = np.ones((ns, nf), dtype=float)
    for i in range(spc):
        f = rng.uniform(2, 5)
        X[i] *= np.sin(f * 2*np.pi * np.linspace(0, 1, nf))
    for i in range(spc, ns):
        f = rng.uniform(5, 8)
        X[i] *= np.sin(5 * 2*np.pi * np.linspace(0, 1, nf))
    y_true = np.array([1]*spc + [2]*spc)

    return X, y_true


@pytest.fixture(scope="module")
def training_config():
    """Fixed training parameters for reproducibility tests."""
    return {
        'l_atoms': 15,
        'iterations': 100,
        'random_seed': 1048596,
        'step': 20,
        'threshold': 0
    }


@pytest.fixture(scope="module")
def reference_model():
    """Pre-trained reference model for comparison (module-scoped)."""
    dico = clawdia.dictionaries.load('tests/data/_dictionary_lrsdl/LRSDL_reference_model.npz')
    
    return dico


@pytest.fixture(scope="module")
def trained_model(default_model_params, reproducibility_data, training_config):
    """Fixture to initialize and train a dictionary model for reuse in tests."""
    X, y_true = reproducibility_data
    model = DictionaryLRSDL(**default_model_params)
    model.fit(X, y_true=y_true, **training_config)
    
    return model  # Return the trained model


# -----------------------------------------------------------------------------
# Reusable model fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def default_model(default_model_params):
    """Fresh model instance for each test (function-scoped)."""
    return DictionaryLRSDL(**default_model_params)


# -----------------------------------------------------------------------------
# Core functionality tests
# -----------------------------------------------------------------------------

def test_reproducibility(reference_model, trained_model):
    """Verify training produces identical results to precomputed reference."""
    # Compare dictionary atoms
    assert np.allclose(trained_model.D, reference_model.D, atol=1e-6), (
        "Class-specific dictionary differs from reference")
    assert np.allclose(trained_model.D0, reference_model.D0, atol=1e-6), (
        "Shared dictionary differs from reference")


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
