import pandas as pd

from misan.physics_validation import check_signature, resolve_with_fallback


def _baseline():
    return pd.Series({"discharge_pressure_mean": 2200.0, "compressor_power_mean": 5.0, "compression_ratio_mean": 2.4})


def test_check_signature_confirms_when_directions_match():
    row = pd.Series({"discharge_pressure_mean": 2500.0, "compressor_power_mean": 6.5, "compression_ratio_mean": 2.8})
    status, note = check_signature(row, "CMP-DEG", _baseline())
    assert status == "confirmed"
    assert "matched" in note


def test_check_signature_rejects_when_directions_contradict():
    row = pd.Series({"discharge_pressure_mean": 1800.0, "compressor_power_mean": 3.0, "compression_ratio_mean": 1.9})
    status, _ = check_signature(row, "CMP-DEG", _baseline())
    assert status == "rejected"


def test_check_signature_unchecked_without_rules():
    row = pd.Series({"discharge_pressure_mean": 2500.0})
    status, note = check_signature(row, "NORMAL", _baseline())
    assert status == "unchecked"
    assert "no physics signature" in note


def test_check_signature_unchecked_without_baseline():
    row = pd.Series({"discharge_pressure_mean": 2500.0})
    status, note = check_signature(row, "CMP-DEG", None)
    assert status == "unchecked"


def test_resolve_with_fallback_switches_class():
    # top class contradicts its signature, but the alternative confirms
    row = pd.Series({"discharge_pressure_mean": 1800.0, "compressor_power_mean": 3.0, "compression_ratio_mean": 1.9})
    baseline = _baseline()
    probs = {"CMP-DEG": 0.6, "NORMAL": 0.4}
    final_class, status, note, fallback_used = resolve_with_fallback(probs, baseline, row, top_class="CMP-DEG")
    assert status == "rejected"
    assert fallback_used is False  # NORMAL has no signature so it can't "confirm"
    assert final_class == "CMP-DEG"
