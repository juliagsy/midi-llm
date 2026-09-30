from midi_llm.train.estimate import detect_hardware, estimate_training


def test_detect_hardware():
    profile = detect_hardware()
    assert profile.machine
    assert profile.cpu


def test_estimate_training_heuristic():
    report = estimate_training(repr_name="remi", steps=50, run_benchmark=False)
    assert report["estimate"]["steps"] == 50
    assert report["estimate"]["estimated_hours"] > 0
    assert "hardware" in report
    assert len(report["scenarios"]) == 2
