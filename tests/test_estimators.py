import numpy as np
import pytest

from clawdia import estimators


def test_elementwise_error_metrics_have_known_values():
    x = np.array([1.0, 2.0, 3.0])
    y = np.ones(3)

    assert estimators.mse(x, y) == pytest.approx(5 / 3)
    assert estimators.medse(x, y) == pytest.approx(1.0)
    assert estimators.residual(x, y) == pytest.approx(np.sqrt(5))

