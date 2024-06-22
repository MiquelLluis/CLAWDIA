import numpy as np
import pytest

from clawdia import lib


#------------------------------------------------------------------------------
# LOAD DATA

@pytest.fixture
def data_extract_patches():
    inputs = []
    parameters = []
    targets = []

    parameters_buffer = np.load('tests/data/lib/extract_patches/parameters_all_tests.npz')

    for i_test in range(16):
        inputs.append(np.load(f'tests/data/lib/extract_patches/input_test-{i_test}.npz')['input'])

        parameters_i = {}
        for k, v in parameters_buffer.items():
            v = v[i_test]
            if isinstance(v, (int, np.int64)) and (v == -1):
                v = None
            elif k == 'limits' and -1 in v:
                v = None
            parameters_i[k] = v
        parameters.append(parameters_i)

        f_target = np.load(f'tests/data/lib/extract_patches/target_test-{i_test}.npz')
        if 'target_coefs' in f_target:
            targets.append((
                f_target['target_Y'],
                f_target['target_coefs']
            ))
        else:
            targets.append(f_target['target_Y'])

    return inputs, parameters, targets




#------------------------------------------------------------------------------
# TESTS


@pytest.mark.parametrize('i_test', range(16))
def test_extract_patches(i_test, data_extract_patches):
    """
    - Split (single) signal with:
      > steps 1, 4, nostep.
      > with and without l2 normalization.
      > with normalization coefficients.
      > with odd and even lengths (pending to implement).
    - Random patches from 2d-array (signal pool):
      > Extract N patches.
      > Extract All patches.
      > Try to extract more patches than available.
      > with and without l2 normalization.
      > allow and do not allow allzeros.
      > with and without limits.
      > with and without patch_min (case with limits).

    """
    input_ = data_extract_patches[0][i_test]
    parameters = data_extract_patches[1][i_test]
    targets = data_extract_patches[2][i_test]

    out = lib.extract_patches(
        input_,
        **parameters
    )
    if isinstance(out, tuple):
        np.testing.assert_array_almost_equal(out[0], targets[0], decimal=9)
        np.testing.assert_array_almost_equal(out[1], targets[1], decimal=9)
    else:
        np.testing.assert_array_almost_equal(out, targets, decimal=9)
