# SPAMS Numerical References

These six files are a reference set for the training/warm-start workflow and the single/batched, iterative, and margin-constrained reconstruction reference tests. They preserve accepted numerical behavior using the golden-master approach; runtime synthetic tests provide independent correctness checks.

> [!CAUTION]
> The committed files are reviewed and accepted baselines. Newly generated outputs are candidates, not automatically trusted replacements. Investigate any differences and independently validate the numerical behaviour before accepting a new baseline. Running the script without `--output-dir` will overwrite the accepted files.

To generate a candidate set from the repository root without overwriting the accepted baselines, run:

```bash
python tests/data/_dictionary_spams/generate_spams_references.py --output-dir /tmp/clawdia-spams-reference-check
```

Generation is explicit and never runs as a test fixture. The script uses the committed `strains_clean.npz` and `strains_injected.npz` inputs, so it requires no waveform simulator or external service. These contain synthetic Gaussian, sine-Gaussian, and ring-down waveforms, with and without injected noise. Their generation through GWADAMA, including noise injection using `noise_psd.txt`, is currently documented in the original notebook [Data generation (outdated API calls).ipynb](../../Data%20generation%20(outdated%20API%20calls).ipynb) and is not implemented by this script.

Generation parameters and input-row selections are read from `generate_spams_references.json`, alongside the identically named Python script.

All reconstruction references use the initially trained dictionary, **before the warm start**. The separate warm-start snapshot records the continued training state. This set was generated on Linux x86_64 with Python 3.11.9, CLAWDIA 0.5.1, NumPy 1.26.4, SciPy 1.16.0, spams-bin 2.6.12, and Dictol 0.1.1.
