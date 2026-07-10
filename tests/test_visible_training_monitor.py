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
