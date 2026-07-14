"""Tests for fair operating-point selection in the report generator."""
from step4_analysis import first_meeting_target


def test_first_meeting_target_never_selects_below_floor_when_available():
    key, row = first_meeting_target({
        "low": {"recall": 0.94},
        "meeting": {"recall": 0.96},
        "high": {"recall": 0.99},
    }, 0.95)
    assert key == "meeting"
    assert row["recall"] >= 0.95


def test_first_meeting_target_falls_back_to_best_available_recall():
    key, _ = first_meeting_target({
        "low": {"recall": 0.80},
        "best": {"recall": 0.90},
    }, 0.95)
    assert key == "best"
