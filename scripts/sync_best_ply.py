#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_REMOTE_HOST = "pch-5090"
DEFAULT_REMOTE_ROOT = "/home/pch/myGS"
DEFAULT_LOCAL_ROOT = Path(__file__).resolve().parents[1] / "supersplat_ply"


@dataclass(frozen=True)
class BestPlyArtifact:
    scene: str
    run_id: str
    best_iteration: int
    remote_summary_path: str
    remote_ply_path: str
    local_path: Path


def _clean_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError("filename component cannot be empty")
    return cleaned


def _metric(summary: dict[str, Any], name: str) -> float:
    metrics = summary.get("best_metrics")
    if not isinstance(metrics, dict) or name not in metrics:
        raise ValueError(f"summary.best_metrics.{name} is required")
    return float(metrics[name])


def _best_iteration(summary: dict[str, Any]) -> int:
    if "best_iteration" not in summary:
        raise ValueError("summary.best_iteration is required")
    value = int(summary["best_iteration"])
    if value <= 0:
        raise ValueError("summary.best_iteration must be positive")
    return value


def _join_remote(*parts: str) -> str:
    result = "/".join(part.strip("/") for part in parts if part)
    return "/" + result if parts and parts[0].startswith("/") else result


def build_artifact(
    *,
    scene: str,
    run_id: str,
    summary: dict[str, Any],
    remote_root: str,
    local_root: Path,
) -> BestPlyArtifact:
    best_iteration = _best_iteration(summary)
    clean_scene = _clean_component(scene)
    clean_run_id = _clean_component(run_id)
    psnr = _metric(summary, "psnr")
    ssim = _metric(summary, "ssim")
    lpips = _metric(summary, "lpips")
    filename = (
        f"{clean_scene}_{clean_run_id}_best_iter{best_iteration}_"
        f"psnr{psnr:.4f}_ssim{ssim:.4f}_lpips{lpips:.4f}.ply"
    )
    remote_run_dir = _join_remote(remote_root, "outputs", scene, run_id)
    return BestPlyArtifact(
        scene=scene,
        run_id=run_id,
        best_iteration=best_iteration,
        remote_summary_path=_join_remote(remote_run_dir, "summary.json"),
        remote_ply_path=_join_remote(
            remote_run_dir,
            "point_cloud",
            f"iteration_{best_iteration}",
            "point_cloud.ply",
        ),
        local_path=local_root / filename,
    )


def _run_text(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout


def load_remote_summary(remote_host: str, remote_summary_path: str) -> dict[str, Any]:
    raw = _run_text(
        ["ssh", "-o", "BatchMode=yes", remote_host, "cat", remote_summary_path]
    )
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("remote summary must be a JSON object")
    return payload


def remote_file_exists(remote_host: str, remote_path: str) -> bool:
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", remote_host, "test", "-f", remote_path],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def copy_remote_ply(remote_host: str, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["scp", "-o", "BatchMode=yes", f"{remote_host}:{remote_path}", str(local_path)],
        check=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync the best 3DGS point_cloud.ply for SuperSplat validation."
    )
    parser.add_argument("--scene", default="visible")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary_path = _join_remote(
        args.remote_root, "outputs", args.scene, args.run_id, "summary.json"
    )
    summary = load_remote_summary(args.remote_host, summary_path)
    artifact = build_artifact(
        scene=args.scene,
        run_id=args.run_id,
        summary=summary,
        remote_root=args.remote_root,
        local_root=args.local_root,
    )
    if not remote_file_exists(args.remote_host, artifact.remote_ply_path):
        raise FileNotFoundError(
            f"remote best PLY does not exist: {artifact.remote_ply_path}"
        )
    if not args.dry_run:
        copy_remote_ply(args.remote_host, artifact.remote_ply_path, artifact.local_path)
    print(
        json.dumps(
            {
                "scene": artifact.scene,
                "run_id": artifact.run_id,
                "best_iteration": artifact.best_iteration,
                "remote_ply_path": artifact.remote_ply_path,
                "local_path": str(artifact.local_path),
                "dry_run": bool(args.dry_run),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
