#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


PSNR_DROP = 0.05
SSIM_DROP = 0.001
LPIPS_RISE = 0.001
COMPARISON_EPSILON = 1e-12


@dataclass(frozen=True)
class MetricRecord:
    iteration: int
    psnr: float
    ssim: float
    lpips: float


def quality_dropped(previous: MetricRecord, current: MetricRecord) -> bool:
    psnr_bad = previous.psnr - current.psnr > PSNR_DROP + COMPARISON_EPSILON
    ssim_bad = previous.ssim - current.ssim > SSIM_DROP + COMPARISON_EPSILON
    lpips_bad = current.lpips - previous.lpips > LPIPS_RISE + COMPARISON_EPSILON
    return psnr_bad and (ssim_bad or lpips_bad)


def select_best(records: Iterable[MetricRecord]) -> MetricRecord:
    values = list(records)
    if not values:
        raise ValueError("at least one metric record is required")
    return max(values, key=lambda item: (item.psnr, item.ssim, -item.lpips))


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


@dataclass(frozen=True)
class RunFiles:
    root: Path

    @property
    def status(self) -> Path:
        return self.root / "status.json"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics.csv"

    @property
    def summary(self) -> Path:
        return self.root / "summary.json"

    def append_metric(self, record: MetricRecord, dropped: bool) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        new_file = not self.metrics.exists()
        with self.metrics.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("iteration", "psnr", "ssim", "lpips", "quality_dropped"),
            )
            if new_file:
                writer.writeheader()
            writer.writerow({**asdict(record), "quality_dropped": int(dropped)})
