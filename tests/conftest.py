import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "clawdia-matplotlib")
)

import numpy as np
import pytest

from clawdia.dictionaries import DictionarySpams


DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def data_dir():
    return DATA_DIR


@pytest.fixture
def identity_dictionary():
    return DictionarySpams(
        dict_init=np.eye(4, dtype=float),
        ignore_completeness=True,
    )
