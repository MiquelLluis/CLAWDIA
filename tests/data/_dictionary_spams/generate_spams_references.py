"""Generate a coupled set of SPAMS dictionary and reconstruction references.

These references reproduce the numerical behavior expected from the SPAMS
dictionary for the reference (golden-master) tests. Their inputs are synthetic waveforms,
Gaussian, sine-Gaussian, and ring-down waveforms, originally generated in
``Data generation (outdated API calls).ipynb``.

This script only regenerates the SPAMS results, not the source waveforms nor
their noise.

Command-line usage
------------------
python generate_spams_references.py [--output-dir DIRECTORY]

--output-dir DIRECTORY
    Destination for the six reference files; defaults to this script's
    directory. Missing directories are created and existing reference files
    are OVERWRITTEN.
-h, --help
    Show this documentation and the available options, then exit without
    generating files.

NOTE: Generation settings are supplied by the JSON file, not command-line flags.

Inputs
------
Inputs are resolved relative to this script, independently of the working
directory. No existing SPAMS reference is read.

generate_spams_references.json
    Generation cfg and input-row selections.
../strains_clean.npz
    ``strains`` contains the clean waveform pool; ``wave_pos`` gives its
    waveform limits within the array.
../strains_injected.npz
    ``strains`` contains the noise-injected waveforms. The configuration selects
    the reconstruction inputs and the margin-reconstruction input separately.
    Signals are used at their full length (``lengths`` is ignored).

Outputs
-------
Six numerical reference files are written:

dico_spams_initial.npy
    Initial atoms, the starting state of the fixed reference dictionary.
    They are sampled from the clean waveform pool.
dico_spams_trained.npy
    Atoms after initial training. Used as the training test's expected
    result and as the dictionary for the reconstruction reference tests.
dico_spams_trained_warm.npy
    Atoms after warm-start training. Used to check the continuation
    from the first trained state.
reconstructions_A.npz
    Three input signals, their single-pass reconstructions, and sparse codes.
    These are the expected results for single, batch, and minibatch reconstruction.
reconstructions_iterative.npz
    Three input signals, iterative reconstructions, residuals, and per-signal
    iteration counts.
reconstruct_auto.npz
    One input signal, its margin-constrained reconstruction, sparse code, and
    boundary-search result. The old filename is retained; the current
    public method (as of v. 0.5.1) is ``reconstruct_margin_constrained``.

Workflow
--------
Read the configuration, initialize the dictionary, select training patches,
perform initial training, and compute the three reconstruction references.
Then continue training to record the warm-start result. Validate the numerical
arrays before saving them.

LEGACY NOTE: the old notebook ran reconstructions after warm start. Now
reconstruction references deliberately use the state before warm start,
matching the reference fixture. The obsolete ``optimum_reconstruct.npz``
example is not generated: the current optimized-reconstruction method has
analytical tests instead.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from clawdia import lib
from clawdia.dictionaries import DictionarySpams


REFERENCE_DIR = Path(__file__).resolve().parent
DATA_DIR = REFERENCE_DIR.parent
CONFIG_PATH = Path(__file__).resolve().with_suffix(".json")


def generate_references(output_dir):
    """Build and validate the complete reference set before writing any files."""

    with CONFIG_PATH.open(encoding="utf-8") as config:
        cfg = json.load(config)
    clean_path = DATA_DIR / "strains_clean.npz"
    injected_path = DATA_DIR / "strains_injected.npz"
    with np.load(clean_path) as data:
        clean_signals, limits = data["strains"], data["wave_pos"]
    with np.load(injected_path) as data:
        signals = data["strains"][cfg["injected_signal_indices"]]
        margin_signal = data["strains"][cfg["margin_signal_index"]]

    # Initialise and train the model.
    model = DictionarySpams(
        signal_pool=clean_signals, wave_pos=limits,
        **cfg["initialisation"],
    )
    initial_components = model.components.copy()
    patches = lib.extract_patches(
        clean_signals, limits=limits, **cfg["training_patches"]
    )
    model.train(patches, **cfg["training"])
    trained_components = model.components.copy()
    np.testing.assert_array_equal(model.dict_init, initial_components)
    np.testing.assert_allclose(
        np.linalg.norm(trained_components, axis=1), 1.0, rtol=1e-12, atol=1e-12
    )

    # Reconstruction references.
    # These use the initially trained state, before warm start.
    #
    # 1) SIMPLE RECONSTRUCTION
    reconstructions, codes = [], []
    for signal in signals:
        reconstruction, code = model.reconstruct(
            signal, **cfg["reconstruction"]
        )
        reconstructions.append(reconstruction)
        codes.append(code.toarray())
    reconstructions = np.array(reconstructions)
    codes = np.array(codes)
    assert reconstructions.shape == signals.shape
    assert np.all(np.max(np.abs(reconstructions), axis=1) > 0)

    # 2) ITERATIVE RECONSTRUCTION
    iterative, residuals, iterations = model.reconstruct_iterative(
        signals, **cfg["iterative_reconstruction"]
    )
    assert iterative.shape == residuals.shape == signals.shape
    assert np.all(
        (iterations >= 1)
        & (iterations < cfg["iterative_reconstruction"]["max_iter"])
    )
    # Residuals are never normalised, therefore we can directly subtract them
    # from the original signals.
    raw_reconstruction = signals - residuals
    peaks = np.max(np.abs(raw_reconstruction), axis=1, keepdims=True)
    assert np.all(peaks > 0)
    expected_iterative = raw_reconstruction
    if cfg["iterative_reconstruction"]["normed"]:
        expected_iterative = raw_reconstruction / peaks
    np.testing.assert_allclose(iterative, expected_iterative, rtol=1e-10, atol=1e-12)

    # 3) MARGIN-CONSTRAINED RECONSTRUCTION
    margin_reconstruction, margin_code, margin_result = (
        model.reconstruct_margin_constrained(margin_signal, **cfg["margin_reconstruction"])
    )
    assert margin_reconstruction.shape == margin_signal.shape
    assert margin_result["converged"] and margin_code is not None
    assert np.max(np.abs(margin_reconstruction)) > 0

    # "Warm" start: keep trainingn the already-trained dictionary.
    model.train(patches, **cfg["warm_start"])
    total_training_iterations = cfg["training"]["n_iter"] + cfg["warm_start"]["n_iter"]
    assert model.n_iter == model.model["iter"] == total_training_iterations

    # Prepare all data to be saved to disk with NumPy.
    arrays = {
        "dico_spams_initial.npy": initial_components,
        "dico_spams_trained.npy": trained_components,
        "dico_spams_trained_warm.npy": model.components.copy(),
    }
    archives = {
        "reconstructions_A.npz": {
            "input": signals,
            "target_reconstructions": reconstructions,
            "target_codes": codes,
        },
        "reconstructions_iterative.npz": {
            "input": signals,
            "target_reconstructions": iterative,
            "target_residuals": residuals,
            "target_iters": iterations,
        },
        "reconstruct_auto.npz": {
            "input": margin_signal,
            "reconstruction": margin_reconstruction,
            "code": margin_code.toarray(),
            "result": margin_result,
        },
    }
    for array in arrays.values():
        assert np.all(np.isfinite(array))
    for archive in archives.values():
        for value in archive.values():
            if isinstance(value, np.ndarray):
                assert np.all(np.isfinite(value))

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        np.save(output_dir / name, array)
    for name, archive in archives.items():
        np.savez_compressed(output_dir / name, **archive)

    print(f"Generated six SPAMS references in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REFERENCE_DIR,
        help="Destination for reference files (default: this script's directory).",
    )
    output_dir = parser.parse_args().output_dir
    generate_references(output_dir)
