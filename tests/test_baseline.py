import numpy as np
import pandas as pd

from misan.baseline import RuleBasedBaseline
from misan.features import FEATURE_COLUMNS


def test_rule_based_baseline_flags_deviation():
    rng = np.random.default_rng(0)
    healthy = pd.DataFrame(
        {col: rng.normal(0, 1, 200) for col in FEATURE_COLUMNS}
    )
    baseline = RuleBasedBaseline(z_threshold=2.5)
    baseline.fit(healthy)

    normal_row = healthy.iloc[0].copy()
    assert baseline.predict_one(normal_row) == "NORMAL"

    anomalous_row = healthy.iloc[0].copy()
    anomalous_row["discharge_pressure_mean"] += 10  # ~10 std devs away
    anomalous_row["compressor_power_mean"] += 10
    anomalous_row["compression_ratio_mean"] += 10
    assert baseline.predict_one(anomalous_row) == "CMP-DEG"
