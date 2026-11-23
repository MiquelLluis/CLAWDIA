import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "clawdia-matplotlib")
)

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def data_dir():
    return DATA_DIR
