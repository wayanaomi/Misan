"""Point Misan's SQLite DB, data dir and model dir at an isolated temp location for
the whole test session (see master prompt: "keep test fixtures isolated from the
actual application database"). Must run before any test module imports misan.config.
"""
import os
import tempfile
from pathlib import Path

_tmp_dir = Path(tempfile.mkdtemp(prefix="misan_test_"))
os.environ["MISAN_DB_PATH"] = str(_tmp_dir / "misan_test.db")
os.environ["MISAN_DATA_DIR"] = str(_tmp_dir / "data")
os.environ["MISAN_MODELS_DIR"] = str(_tmp_dir / "data" / "models")

import pytest  # noqa: E402

from misan import simulate, training  # noqa: E402
from misan.config import SETTINGS  # noqa: E402
from misan.persistence import init_db  # noqa: E402

init_db()

FAULT_SUBSET = ["NORMAL", "CMP-DEG", "REF-LEAK"]


@pytest.fixture(scope="session")
def trained_version():
    df = simulate.generate_dataset(fault_classes=FAULT_SUBSET, runs_per_condition=1, duration_minutes=30, seed=1)
    path = SETTINGS.data_dir / "raw" / "test_dataset.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    result = training.train(str(path), seed=1)
    return result, df
