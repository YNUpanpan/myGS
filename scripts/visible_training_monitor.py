#!/usr/bin/env python3
from __future__ import annotations

import csv
import argparse
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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


TERMINAL_STATES = {"completed", "early_stopped", "failed", "preflight_failed"}
DEFAULT_EVAL_ITERATIONS = tuple(range(5000, 15001, 1000))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_eval_iterations(raw: str | None) -> tuple[int, ...]:
    if raw is None or not raw.strip():
        return DEFAULT_EVAL_ITERATIONS
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError("evaluation iterations must be positive")
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError("evaluation iterations must be strictly increasing")
    return values


class VisibleTrainingMonitor:
    def __init__(
        self,
        files: RunFiles,
        total_iterations: int,
        eval_iterations: tuple[int, ...],
        pid: int,
    ) -> None:
        self.files = files
        self.total_iterations = total_iterations
        self.eval_iterations = eval_iterations
        self.pid = pid
        self.started = time.monotonic()
        self.records: list[MetricRecord] = []
        self._lpips_model = None

    @classmethod
    def from_environment(
        cls, model_path: str | os.PathLike[str], total_iterations: int
    ) -> "VisibleTrainingMonitor":
        model_root = Path(model_path).resolve()
        monitor_root = Path(os.environ.get("MYGS_MONITOR_DIR", model_root)).resolve()
        if monitor_root != model_root:
            raise ValueError(
                f"MYGS_MONITOR_DIR must match model path: {monitor_root} != {model_root}"
            )
        if total_iterations <= 0:
            raise ValueError("total_iterations must be positive")
        eval_iterations = _parse_eval_iterations(os.environ.get("MYGS_EVAL_ITERATIONS"))
        if eval_iterations[-1] > total_iterations:
            raise ValueError("evaluation iteration exceeds total iterations")
        pid = int(os.environ.get("MYGS_TRAIN_PID", os.getpid()))
        instance = cls(RunFiles(monitor_root), total_iterations, eval_iterations, pid)
        instance._write_status("training", 0, phase="initializing")
        return instance

    def _write_status(
        self,
        state: str,
        iteration: int,
        *,
        phase: str,
        ema_loss: float | None = None,
        reason: str | None = None,
    ) -> None:
        elapsed = max(0.0, time.monotonic() - self.started)
        eta = None
        if iteration > 0 and iteration < self.total_iterations:
            eta = elapsed * (self.total_iterations - iteration) / iteration
        payload = {
            "state": state,
            "phase": phase,
            "iteration": iteration,
            "total_iterations": self.total_iterations,
            "progress_percent": round(iteration * 100.0 / self.total_iterations, 4),
            "ema_loss": ema_loss,
            "elapsed_seconds": round(elapsed, 3),
            "eta_seconds": None if eta is None else round(eta, 3),
            "pid": self.pid,
            "reason": reason,
            "updated_at": _utc_now(),
        }
        atomic_write_json(self.files.status, payload)

    def update_progress(self, iteration: int, ema_loss: float) -> None:
        self._write_status("training", iteration, phase="training", ema_loss=ema_loss)

    def record_metric(self, record: MetricRecord) -> bool:
        if self.records and record.iteration <= self.records[-1].iteration:
            raise ValueError("metric iterations must be strictly increasing")
        dropped = bool(self.records) and quality_dropped(self.records[-1], record)
        self.records.append(record)
        self.files.append_metric(record, dropped)
        return dropped

    def evaluate(
        self,
        iteration: int,
        cameras,
        render_func,
        render_args: tuple,
        train_test_exp: bool,
    ) -> bool:
        if iteration not in self.eval_iterations:
            raise ValueError(f"unexpected evaluation iteration: {iteration}")
        self._write_status("training", iteration, phase="evaluating")
        record = self._evaluate_cameras(
            iteration, cameras, render_func, render_args, train_test_exp
        )
        return self.record_metric(record)

    def _evaluate_cameras(
        self,
        iteration: int,
        cameras,
        render_func,
        render_args: tuple,
        train_test_exp: bool,
    ) -> MetricRecord:
        import torch
        from lpipsPyTorch.modules.lpips import LPIPS
        from torchvision.utils import save_image
        from utils.image_utils import psnr
        from utils.loss_utils import ssim

        camera_list = list(cameras)
        if not camera_list:
            raise ValueError("evaluation camera list is empty")
        if self._lpips_model is None:
            self._lpips_model = LPIPS(net_type="vgg").to("cuda").eval()

        evaluation_root = self.files.root / "evaluation" / f"iteration_{iteration}"
        renders_root = evaluation_root / "renders"
        gt_root = evaluation_root / "gt"
        renders_root.mkdir(parents=True, exist_ok=True)
        gt_root.mkdir(parents=True, exist_ok=True)

        psnr_values: list[float] = []
        ssim_values: list[float] = []
        lpips_values: list[float] = []
        with torch.no_grad():
            for index, camera in enumerate(camera_list):
                image = torch.clamp(render_func(camera, *render_args)["render"], 0.0, 1.0)
                ground_truth = torch.clamp(camera.original_image.to("cuda"), 0.0, 1.0)
                if train_test_exp:
                    image = image[..., image.shape[-1] // 2 :]
                    ground_truth = ground_truth[..., ground_truth.shape[-1] // 2 :]

                psnr_values.append(float(psnr(image, ground_truth).mean().item()))
                ssim_values.append(float(ssim(image, ground_truth).mean().item()))
                lpips_value = self._lpips_model(
                    image.unsqueeze(0), ground_truth.unsqueeze(0)
                )
                lpips_values.append(float(lpips_value.mean().item()))

                if index < 5:
                    image_name = Path(str(camera.image_name)).stem
                    filename = f"{index:02d}_{image_name}.png"
                    save_image(image, renders_root / filename)
                    save_image(ground_truth, gt_root / filename)

                del image, ground_truth

        torch.cuda.empty_cache()
        count = float(len(camera_list))
        return MetricRecord(
            iteration=iteration,
            psnr=sum(psnr_values) / count,
            ssim=sum(ssim_values) / count,
            lpips=sum(lpips_values) / count,
        )

    def finalize(self, state: str, iteration: int, reason: str) -> None:
        if state not in TERMINAL_STATES:
            raise ValueError(f"invalid terminal state: {state}")
        best = select_best(self.records) if self.records else None
        summary = {
            "state": state,
            "reason": reason,
            "last_iteration": iteration,
            "best_iteration": None if best is None else best.iteration,
            "best_metrics": None if best is None else asdict(best),
            "updated_at": _utc_now(),
        }
        atomic_write_json(self.files.summary, summary)
        self._write_status(state, iteration, phase="finished", reason=reason)


def finalize_process_exit(run_dir: str | os.PathLike[str], exit_code: int) -> None:
    files = RunFiles(Path(run_dir).resolve())
    status = {}
    if files.status.exists():
        status = json.loads(files.status.read_text(encoding="utf-8"))
    if status.get("state") in TERMINAL_STATES:
        return
    iteration = int(status.get("iteration", 0))
    reason = f"process_exit_{exit_code}"
    summary = {
        "state": "failed",
        "reason": reason,
        "last_iteration": iteration,
        "best_iteration": None,
        "best_metrics": None,
        "updated_at": _utc_now(),
    }
    atomic_write_json(files.summary, summary)
    status.update(
        {
            "state": "failed",
            "phase": "finished",
            "reason": reason,
            "updated_at": _utc_now(),
        }
    )
    atomic_write_json(files.status, status)


def _main() -> int:
    parser = argparse.ArgumentParser(description="Visible 3DGS training monitor")
    subparsers = parser.add_subparsers(dest="command", required=True)
    finalize_parser = subparsers.add_parser("finalize-exit")
    finalize_parser.add_argument("--run-dir", required=True)
    finalize_parser.add_argument("--exit-code", required=True, type=int)
    args = parser.parse_args()
    if args.command == "finalize-exit":
        finalize_process_exit(args.run_dir, args.exit_code)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
