# CLAWDIA test suite

The test suite checks the scientific behaviour of CLAWDIA's public interfaces.
Coverage is useful, but it is not the acceptance criterion; tests should use controlled inputs and verify a meaningful numerical result, physical invariant, error condition, or preservation of state.

## Running the tests

Install the test dependencies with `pip install -e '.[test]'`, then run pytest
through the active Python environment:

```bash
# Entire suite
python -m pytest -q

# Fast unit suite
python -m pytest -m "not integration and not regression" -q

# Integration and regression suite
python -m pytest -m "integration or regression" -q

# Coverage, including branches
python -m pytest --cov=clawdia --cov-branch --cov-report=term-missing
```


## Test categories

Tests without a marker form the **unit suite**. They:
- exercise one public function or a small, isolated behaviour using deterministic synthetic data.
- should be fast.
- should not require a trained model, external service, or large reference file.

The remaining categories are declared as custom pytest markers in `pyproject.toml`:

| Category | Marker | Purpose |
| --- | --- | --- |
| Unit | none | Checks an isolated behaviour against an independently known result or invariant. |
| Integration | `@pytest.mark.integration` | Check multiple components, an external backend, persistence, or iterative model training together. |
| Regression | `@pytest.mark.regression` | Compares behaviour with a trusted reference result, often dependency- or platform-sensitive. |

These categories are not mutually exclusive:
- **Integration** describes the scope of the execution;
- **Regression** describes the source of the expected result.


## Suite layout

Test modules mirror the corresponding CLAWDIA modules, and use module-level test functions by default.

Test classes may be introduced where they provide useful grouping or shared setup.


## Scientific assertions

The test suite is a work in progress. The following conventions guide new and revised tests; they follow the testing guidance of [NumPy](https://numpy.org/devdocs/reference/testing.html), [SciPy](https://docs.scipy.org/doc/scipy/dev/contributor/contributor_toc.html), and [Astropy](https://docs.astropy.org/en/latest/development/testguide.html), adapted to CLAWDIA's numerical signal-processing requirements.

- Prefer small deterministic signals with an analytically known or independently defined answer.
- For reconstructions, verify the output shape as well as an independent accuracy metric or invariant, such as residual conservation.
- Use `pytest.approx` for approximate Python scalars, `numpy.testing.assert_array_equal` for exact array values, and `numpy.testing.assert_allclose` for approximate arrays.
- Prefer explicit tolerances. Choose them from the algorithm, input scale, floating-point precision, and expected accumulation error, not to simply make a test pass.
- Verify relevant shape, dtype, labels, model state, and any sampling metadata exposed by the interface.
- Cover boundary cases, invalid inputs, and expected warnings or exceptions where relevant.
- Supply a seed for stochastic algorithms. Where an API must temporarily use NumPy's global random state, verify that the caller's state is preserved.
- Test through public interfaces; access private internals only when the relevant behaviour cannot be observed through a public result.


## Fixtures and reference data

- Shared fixtures are defined in `conftest.py`.
- Module-specific fixtures should stay beside the tests that use them.
- Paths must be derived from `tests/`, not from the process working directory.

Files under `tests/data/` are not necessarily scientific ground truth:
- Each regression fixture should
  - have a known origin, and
  - represent an intentional behaviour.
- Do not regenerate a reference only because a test fails; determine whether
  - the API changed,
  - the test is obsolete,
  - a dependency changed numerically, or
  - the implementation regressed.

Generation scripts or notebooks should be kept when they document how a trusted fixture was produced, even if they require updating before they can be run again.


## Adding a test

1. Identify the public behaviour and define the expected result independently of the implementation where possible.
2. Put the test in the module corresponding to that public interface.
3. Use synthetic data unless committed reference data adds genuine scientific value.
4. Mark the test only when it meets the integration or regression definition above.
5. Run the relevant test module, both marker selections, and the complete suite.
6. Check coverage for untested behaviour, not as a substitute for meaningful assertions.
