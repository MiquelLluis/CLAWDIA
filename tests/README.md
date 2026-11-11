# CLAWDIA test suite

The test suite aims to protect scientifically meaningful behaviour of CLAWDIA. It is a work in progress, not a guarantee of numerical correctness or complete coverage. The principles below guide its ongoing review and development.

The suite uses [pytest](https://docs.pytest.org/en/stable/) as its testing framework.


## Test categories

| Category | Marker | Purpose |
| --- | --- | --- |
| Unit (default) | none | Checks focused behaviour using controlled inputs and an independently known answer, invariant, or expected error. |
| Integration | `@pytest.mark.integration` | Checks interactions between components or workflows. |
| Reference (golden-master) | `@pytest.mark.reference` | Compares numerical outputs with stored, reviewed and accepted baselines. |

A test may be both integration and reference: integration describes what runs together, while reference describes where the expected result comes from. The unmarked suite mostly contains focused unit/component tests.

Reference tests follow the established [golden-master approach](https://en.wikipedia.org/wiki/Characterization_test), also used to characterize existing behaviour. They detect changes from accepted outputs, but do not independently establish scientific correctness.


## Running the tests

With CLAWDIA and its normal dependencies installed, running the tests only requires `pytest` in the same Python environment, and optionally `pytest-cov` to get coverage information. Run these commands from the repository root:

```bash
# Entire suite
python -m pytest -q

# Default suite
python -m pytest -m "not integration and not reference" -q

# Integration and reference suite
python -m pytest -m "integration or reference" -q

# Coverage (branch coverage is enabled by default in pyproject.toml)
python -m pytest --cov=clawdia --cov-report=term-missing
```


## Suite layout

Test modules mirror the corresponding CLAWDIA modules and use module-level test functions by default. Classes may be used to group related tests, for example those testing the same function.


## Scientific assertions

These are a collection of guiding points based on those documented in [NumPy](https://numpy.org/devdocs/reference/testing.html), [SciPy](https://docs.scipy.org/doc/scipy/dev/contributor/contributor_toc.html), and [Astropy](https://docs.astropy.org/en/latest/development/testguide.html). *Existing tests are being reviewed against them; they are not all applied uniformly yet.*

- Prioritise important behaviour and known risks over coverage. Avoid trivial assertions and exhaustive parameter combinations without a clear reason. Briefly explain a test's motivation and expected result where they are not obvious.
- Prefer small deterministic signals with an analytically known or independently defined answer.
- For reconstructions, check more than shape and type: compare with an expected signal or verify a meaningful numerical invariant, such as residual conservation. For reference cases, verify it matches the accepted numerical output up to a justified tolerance.
- Use `pytest.approx` for approximate Python scalars, `numpy.testing.assert_array_equal` for exact array values, and `numpy.testing.assert_allclose` for approximate arrays.
- Prefer explicit tolerances. Choose them from the algorithm, input scale, floating-point precision, and expected accumulation error, not to simply make a test pass.
- Verify relevant shape, dtype, labels, model state, and any sampling metadata exposed by the interface.
- Cover boundary cases, invalid inputs, and expected warnings or exceptions where relevant.
- Keep tests reasonably fast, CPU-only, and independent of external services when possible. Unexpected CLAWDIA `RuntimeWarning`s are treated as errors by the pytest configuration; assert expected ones with `pytest.warns`.
- Specify a seed for stochastic algorithms. If an API must temporarily use NumPy's global random state, verify that the caller's state is restored after the call.
- Prioritise public behaviour, while also testing critical private functions directly where useful.
- When the scientifically expected result is unclear, document the uncertainty; never assume the current output to be correct.


## Fixtures and reference data

- Shared fixtures are defined in `conftest.py`.
- Module-specific fixtures stay in their test module, preferably after imports and before the tests.
- Paths must be derived from `tests/`, not from the process working directory.
- Prefer synthetic signals and models created at runtime. Use stored numerical baselines in explicitly marked reference tests, sparingly for complete workflows or critical numerical behaviour.
- For training and persistence workflows, use the intended initialisation and training methods rather than fabricate a trained state. Small explicit dictionaries remain useful for analytical reconstruction tests.
- Keep the origin, inputs, configuration, and relevant dependency settings of reference data documented. Retain generation scripts or notebooks, including upstream data recipes not yet replaced by a script.

> [!CAUTION]
> Generating new reference data does not validate it. Investigate differences before changing an accepted baseline, and generate candidates separately for review. Never replace references simply to make a failing test pass; changes may come from defects, intentional algorithm changes, dependencies, inputs, configuration, or unsuitable tolerances.

Files under `tests/data/` are not necessarily scientific ground truth. The [SPAMS reference README](data/_dictionary_spams/README.md) describes its accepted baseline set and candidate-generation procedure.
