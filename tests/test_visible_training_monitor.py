import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


MODULE = Path(__file__).parents[1] / "scripts" / "visible_training_monitor.py"
monitor = None
if MODULE.exists():
    SPEC = importlib.util.spec_from_file_location("visible_training_monitor", MODULE)
    monitor = importlib.util.module_from_spec(SPEC)
    assert SPEC.loader is not None
    sys.modules[SPEC.name] = monitor
    SPEC.loader.exec_module(monitor)


def test_monitor_module_exists():
    assert MODULE.exists(), f"missing production module: {MODULE}"


def rec(iteration, psnr, ssim, lpips):
    return monitor.MetricRecord(iteration, psnr, ssim, lpips)


@pytest.mark.skipif(monitor is None, reason="monitor module is not implemented")
def test_quality_drop_requires_psnr_and_one_secondary_metric():
    previous = rec(5000, 25.00, 0.8000, 0.2000)
    assert monitor.quality_dropped(previous, rec(6000, 24.94, 0.7989, 0.2000))
    assert monitor.quality_dropped(previous, rec(6000, 24.94, 0.8000, 0.2011))
    assert not monitor.quality_dropped(previous, rec(6000, 24.96, 0.7980, 0.2020))
    assert not monitor.quality_dropped(previous, rec(6000, 24.94, 0.7995, 0.2005))


@pytest.mark.skipif(monitor is None, reason="monitor module is not implemented")
def test_threshold_boundaries_do_not_stop():
    previous = rec(5000, 25.00, 0.8000, 0.2000)
    current = rec(6000, 24.95, 0.7990, 0.2010)
    assert not monitor.quality_dropped(previous, current)


@pytest.mark.skipif(monitor is None, reason="monitor module is not implemented")
def test_select_best_orders_psnr_then_ssim_then_lpips():
    records = [
        rec(5000, 25.0, 0.80, 0.20),
        rec(6000, 25.1, 0.79, 0.22),
        rec(7000, 25.1, 0.81, 0.23),
        rec(8000, 25.1, 0.81, 0.19),
    ]
    assert monitor.select_best(records).iteration == 8000


@pytest.mark.skipif(monitor is None, reason="monitor module is not implemented")
def test_atomic_json_and_metric_csv(tmp_path):
    files = monitor.RunFiles(tmp_path)
    payload = {"state": "training", "iteration": 10}
    monitor.atomic_write_json(files.status, payload)
    assert json.loads(files.status.read_text(encoding="utf-8")) == payload

    files.append_metric(rec(5000, 25.0, 0.80, 0.20), False)
    files.append_metric(rec(6000, 24.9, 0.79, 0.22), True)
    with files.metrics.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["iteration"]) for row in rows] == [5000, 6000]
    assert [int(row["quality_dropped"]) for row in rows] == [0, 1]


def test_lifecycle_records_drop_and_best_iteration(tmp_path, monkeypatch):
    monkeypatch.setenv("MYGS_MONITOR_DIR", str(tmp_path))
    monkeypatch.setenv("MYGS_EVAL_ITERATIONS", "5000,6000")
    monkeypatch.setenv("MYGS_TRAIN_PID", "12345")
    instance = monitor.VisibleTrainingMonitor.from_environment(tmp_path, 15000)
    instance.update_progress(10, 0.5)
    assert instance.record_metric(rec(5000, 25.0, 0.800, 0.200)) is False
    assert instance.record_metric(rec(6000, 24.9, 0.798, 0.202)) is True
    instance.finalize("early_stopped", 6000, "quality_drop")
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["best_iteration"] == 5000
    assert summary["last_iteration"] == 6000
    assert summary["state"] == "early_stopped"


def test_finalize_exit_preserves_terminal_state(tmp_path, monkeypatch):
    monkeypatch.setenv("MYGS_MONITOR_DIR", str(tmp_path))
    monkeypatch.setenv("MYGS_EVAL_ITERATIONS", "5000")
    instance = monitor.VisibleTrainingMonitor.from_environment(tmp_path, 5000)
    instance.record_metric(rec(5000, 25.0, 0.800, 0.200))
    instance.finalize("completed", 5000, "max_iterations")
    monitor.finalize_process_exit(tmp_path, 1)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["state"] == "completed"
    assert summary["reason"] == "max_iterations"


def test_evaluate_records_backend_result(tmp_path, monkeypatch):
    monkeypatch.setenv("MYGS_MONITOR_DIR", str(tmp_path))
    monkeypatch.setenv("MYGS_EVAL_ITERATIONS", "5000,6000")
    instance = monitor.VisibleTrainingMonitor.from_environment(tmp_path, 6000)
    expected = rec(5000, 25.0, 0.800, 0.200)
    monkeypatch.setattr(instance, "_evaluate_cameras", lambda *args: expected)

    stopped = instance.evaluate(5000, [object()], object(), (), False)

    assert stopped is False
    assert instance.records == [expected]
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["phase"] == "evaluating"
    assert status["iteration"] == 5000
