from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from clawdia.dictionaries import DictionaryLRSDL


# Fixed parameters and data generation
REFERENCE_DIR = Path("tests/data/_dictionary_lrsdl/")
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)


def generate_reference_model():
    ns, nf = 100, 20  # samples, features
    spc = ns // 2  # samples per class
    l_atoms = 15
    step = 20
    rng = np.random.default_rng(1048596)

    X, y_true = gen_population(ns, nf, spc, rng)

    # Train model
    dico = DictionaryLRSDL(
        lambd=0.01, lambd2=0.01, eta=0.0001,
        k=4, k0=4, updateX_iters=100, updateD_iters=100
    )
    dico.fit(
        X, y_true=y_true, l_atoms=l_atoms, iterations=100, step=step,
        threshold=0, random_seed=1048596, verbose=True, show_after=10
    )
    print(dico)

    # Train model
    X, y_true = gen_population(ns, nf, spc, rng)
    y_pred = dico.predict(X, threshold=0, offset=0, with_losses=False)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='binary')

    print(f"F1 score TRAIN: {f1:.3f}")

    # Test model
    X, y_true = gen_population(ns, nf, spc, rng)
    y_pred, losses = dico.predict(X, threshold=0, offset=0, with_losses=True)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='binary')

    print(f"F1 score TEST:  {f1:.3f}")
    
    
    f_model = REFERENCE_DIR / "LRSDL_reference_model.npz"
    dico.save(f_model)
    f_results = REFERENCE_DIR/"LRSDL_reference_model_test.txt"
    np.savetxt(f_results, np.array([y_pred, losses]).T)

    print(f"Model saved to: {f_model}")
    print(f"Test results saved to: {f_results}")

def gen_population(ns, nf, spc, rng):
    X = np.ones((ns, nf), dtype=float)
    for i in range(spc):
        f = rng.uniform(2, 5)
        X[i] *= np.sin(f * 2*np.pi * np.linspace(0, 1, nf))
    for i in range(spc, ns):
        f = rng.uniform(5, 8)
        X[i] *= np.sin(5 * 2*np.pi * np.linspace(0, 1, nf))
    
    y_true = np.array([1]*spc + [2]*spc)
    return X, y_true


if __name__ == "__main__":
    generate_reference_model()