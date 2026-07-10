from pathlib import Path


PATCH = (
    Path(__file__).parents[1]
    / "patches"
    / "gaussian-splatting"
    / "visible-monitored-training.patch"
)


def test_patch_contains_required_monitor_hooks():
    assert PATCH.exists(), f"missing tracked patch: {PATCH}"
    text = PATCH.read_text(encoding="utf-8")
    assert "from visible_training_monitor import VisibleTrainingMonitor" in text
    assert "monitor.update_progress" in text
    assert "monitor.evaluate" in text
    assert "(scene.gaussians, pipe, background" in text
    assert 'monitor.finalize("early_stopped"' in text
    assert 'monitor.finalize("completed"' in text
