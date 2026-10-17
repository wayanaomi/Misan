from misan.detection import PersistenceTracker


def test_alert_requires_three_consecutive_abnormal_windows():
    tracker = PersistenceTracker(n=3)

    assert tracker.update("HVAC-01", True) is False
    assert tracker.update("HVAC-01", True) is False
    assert tracker.update("HVAC-01", True) is True


def test_normal_window_resets_consecutive_alert_sequence():
    tracker = PersistenceTracker(n=3)

    assert tracker.update("HVAC-01", True) is False
    assert tracker.update("HVAC-01", True) is False
    assert tracker.update("HVAC-01", False) is False
    assert tracker.update("HVAC-01", True) is False
    assert tracker.update("HVAC-01", True) is False
    assert tracker.update("HVAC-01", True) is True


def test_alert_persistence_is_tracked_per_asset():
    tracker = PersistenceTracker(n=3)

    assert tracker.update("HVAC-01", True) is False
    assert tracker.update("HVAC-02", True) is False
    assert tracker.update("HVAC-01", True) is False
    assert tracker.update("HVAC-02", True) is False
    assert tracker.update("HVAC-01", True) is True
    assert tracker.update("HVAC-02", True) is True
