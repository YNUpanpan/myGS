# Visible 3DGS Monitored Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train only the visible scene to at most 15000 iterations on GPU 0, synchronously evaluate a fixed 43-image holdout every 1000 iterations from 5000, expose live progress, and stop exactly at a checkpoint when the approved quality-decline rule fires.

**Architecture:** Keep the official 3DGS checkout fixed at `54c035f`. Put reusable monitoring, metric history, atomic status output, and early-stop logic in a tracked Python module, then apply a small tracked patch that calls this module from the official continuous training loop. A Bash launcher performs all preflight checks and starts a persistent worker; a separate read-only monitor reports status without controlling or deleting the run.

**Tech Stack:** Bash, Python 3, dataclasses, JSON/CSV, PyTorch cu128, torchvision, official 3DGS renderer, LPIPS VGG, pytest/unittest, Git, Ubuntu, NVIDIA RTX 5090.

## Global Constraints

- Server: `pch-5090`; project root: `/home/pch/myGS`.
- Source scene: `/home/pch/myGS/datasets/uav_3dgs/processed/visible`.
- Use only GPU 0 for this run; do not start thermal training.
- Use project Conda env `mygs-3dgs-cu128`; do not change system CUDA.
- Require official 3DGS checkout HEAD `54c035f` before applying the monitoring patch.
- Run with `--eval`, producing the fixed 296-train/43-test split from 339 visible images.
- Formal checkpoints: `5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000`.
- Early stop when PSNR drops by more than `0.05 dB` and either SSIM drops by more than `0.001` or LPIPS rises by more than `0.001` relative to the previous checkpoint.
- Raw data stays read-only. Never use recursive deletion or batch deletion. Keep smoke and formal outputs.
- Git tracks scripts, patch, tests, docs, and summaries only; it does not track data, tools, logs, renders, point clouds, or checkpoints.
- Implement directly on server branch `main`, as previously authorized for this project.

---

## File Structure

- Create `scripts/visible_training_monitor.py`: pure threshold logic, best-checkpoint selection, atomic state files, CSV history, GPU evaluation, fixed-view render saving, and CLI finalization.
- Create `tests/test_visible_training_monitor.py`: deterministic unit tests for thresholds, best selection, atomic status, CSV history, and terminal-state preservation.
- Create `patches/gaussian-splatting/visible-monitored-training.patch`: minimal hooks for progress updates, synchronous evaluation, early stop, and terminal finalization.
- Create `scripts/run_visible_training.sh`: preflight, patch verification/application, LPIPS cache check, formal/smoke launch, persistence, and resume validation.
- Create `scripts/monitor_visible_training.sh`: read-only latest-run status, metrics, PID, GPU, and log display.
- Modify `.gitignore`: ignore all project-local tools and generated runtime state without deleting them.
- Modify `AGENTS.md`: append implementation, smoke, live-run, and final-result evidence under conversation 4.

---

### Task 1: Pure Monitoring State and Early-Stop Logic

**Files:**
- Create: `/home/pch/myGS/scripts/visible_training_monitor.py`
- Create: `/home/pch/myGS/tests/test_visible_training_monitor.py`

**Interfaces:**
- Produces: `MetricRecord`, `quality_dropped(previous, current) -> bool`, `select_best(records) -> MetricRecord`, `atomic_write_json(path, payload)`, and `RunFiles`.
- Consumes: standard-library `csv`, `json`, `os`, `tempfile`, `dataclasses`, `pathlib`, and `typing` only for this task.

- [ ] **Step 1: Write failing threshold and best-selection tests**

Create `tests/test_visible_training_monitor.py` with these initial tests:

```python
from pathlib import Path
import importlib.util
import sys

MODULE = Path(__file__).parents[1] / "scripts" / "visible_training_monitor.py"
SPEC = importlib.util.spec_from_file_location("visible_training_monitor", MODULE)
monitor = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = monitor
SPEC.loader.exec_module(monitor)


def rec(iteration, psnr, ssim, lpips):
    return monitor.MetricRecord(iteration, psnr, ssim, lpips)


def test_quality_drop_requires_psnr_and_one_secondary_metric():
    previous = rec(5000, 25.00, 0.8000, 0.2000)
    assert monitor.quality_dropped(previous, rec(6000, 24.94, 0.7989, 0.2000))
    assert monitor.quality_dropped(previous, rec(6000, 24.94, 0.8000, 0.2011))
    assert not monitor.quality_dropped(previous, rec(6000, 24.96, 0.7980, 0.2020))
    assert not monitor.quality_dropped(previous, rec(6000, 24.94, 0.7995, 0.2005))


def test_threshold_boundaries_do_not_stop():
    previous = rec(5000, 25.00, 0.8000, 0.2000)
    current = rec(6000, 24.95, 0.7990, 0.2010)
    assert not monitor.quality_dropped(previous, current)


def test_select_best_orders_psnr_then_ssim_then_lpips():
    records = [
        rec(5000, 25.0, 0.80, 0.20),
        rec(6000, 25.1, 0.79, 0.22),
        rec(7000, 25.1, 0.81, 0.23),
        rec(8000, 25.1, 0.81, 0.19),
    ]
    assert monitor.select_best(records).iteration == 8000
```

- [ ] **Step 2: Run tests and verify the expected failure**

Run:

```bash
cd /home/pch/myGS
tools/miniconda3/bin/conda run -n mygs-3dgs-cu128 python -m pytest tests/test_visible_training_monitor.py -q
```

Expected: collection fails because `scripts/visible_training_monitor.py` does not exist.

- [ ] **Step 3: Implement the minimal pure module**

Create `scripts/visible_training_monitor.py` with:

```python
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
```

- [ ] **Step 4: Run unit tests and verify they pass**

Run the Step 2 command again.

Expected: `3 passed`.

- [ ] **Step 5: Add atomic-file and CSV tests**

Add `import csv` and `import json` at the top of the test file, then append:

```python
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
```

Do not delete the temporary directory manually; pytest owns its lifecycle.

- [ ] **Step 6: Run the complete Task 1 test file**

Expected: all tests pass and `python -m py_compile scripts/visible_training_monitor.py` exits 0.

- [ ] **Step 7: Commit Task 1**

```bash
git add scripts/visible_training_monitor.py tests/test_visible_training_monitor.py
git commit -m "Add visible training monitor core"
git push
```

---

### Task 2: Runtime Progress, Test Metrics, and Terminal State

**Files:**
- Modify: `/home/pch/myGS/scripts/visible_training_monitor.py`
- Modify: `/home/pch/myGS/tests/test_visible_training_monitor.py`

**Interfaces:**
- Produces: `VisibleTrainingMonitor.from_environment(model_path, total_iterations)`, `update_progress(iteration, ema_loss)`, `evaluate(iteration, cameras, render_func, render_args, train_test_exp) -> bool`, and `finalize(state, iteration, reason)`.
- Consumes: `MYGS_MONITOR_DIR`, `MYGS_EVAL_ITERATIONS`, `MYGS_TRAIN_PID`, PyTorch tensors, 3DGS `render_func`, and fixed test cameras.

- [ ] **Step 1: Write failing lifecycle tests**

Add `import json` if it is not already present, then append this exact lifecycle test:

```python
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
```

- [ ] **Step 2: Verify the new lifecycle tests fail**

Expected: failure because `VisibleTrainingMonitor` is not defined.

- [ ] **Step 3: Implement lifecycle methods and environment validation**

Add `VisibleTrainingMonitor` with these exact invariants:

- `MYGS_MONITOR_DIR` must resolve to the same directory as `model_path`.
- Formal eval iterations default to the approved list and must be strictly increasing.
- `update_progress` atomically writes state `training` every 10 iterations and computes ETA from monotonic elapsed time.
- `record_metric` compares only with the immediately previous record, appends CSV, updates best metrics, and returns the early-stop decision.
- `finalize` writes terminal `status.json` and `summary.json`; a later shell finalizer must not overwrite `completed` or `early_stopped` with `failed`.

- [ ] **Step 4: Implement GPU evaluation once per checkpoint**

In `evaluate`:

1. Set status phase to `evaluating`.
2. Lazily instantiate one `lpipsPyTorch.modules.lpips.LPIPS(net_type="vgg")` model on CUDA and reuse it.
3. Render each of the 43 cameras under `torch.no_grad()`.
4. Compute mean PSNR with `utils.image_utils.psnr`, SSIM with `utils.loss_utils.ssim`, and LPIPS with the cached VGG model.
5. Save renders and ground truths for the first five test cameras to `evaluation/iteration_<N>/renders` and `gt` using `torchvision.utils.save_image`.
6. Call `record_metric` and return its Boolean result.
7. Release per-view tensors and call `torch.cuda.empty_cache()` after evaluation.

- [ ] **Step 5: Add a CLI finalizer**

Support:

```bash
python scripts/visible_training_monitor.py finalize-exit \
  --run-dir /home/pch/myGS/outputs/visible/<run_id> \
  --exit-code 1
```

If current state is nonterminal, set state `failed`, preserve last iteration, and record `process_exit_<code>`; otherwise leave the terminal files unchanged.

- [ ] **Step 6: Run unit tests and static compilation**

Expected: all monitor tests pass; `py_compile` exits 0.

- [ ] **Step 7: Commit Task 2**

```bash
git add scripts/visible_training_monitor.py tests/test_visible_training_monitor.py
git commit -m "Add visible quality evaluation lifecycle"
git push
```

---

### Task 3: Minimal Official 3DGS Training Hook Patch

**Files:**
- Create: `/home/pch/myGS/patches/gaussian-splatting/visible-monitored-training.patch`
- Modify only in ignored checkout: `/home/pch/myGS/tools/gaussian-splatting/train.py`

**Interfaces:**
- Consumes: `VisibleTrainingMonitor` on `PYTHONPATH` and the existing official `training()` loop.
- Produces: progress updates every 10 iterations, synchronous checkpoint evaluations, exact early-stop break, and terminal status.

- [ ] **Step 1: Verify the patch baseline before editing**

Run:

```bash
test "$(git -C tools/gaussian-splatting rev-parse --short HEAD)" = "54c035f"
git -C tools/gaussian-splatting diff --exit-code -- train.py
```

Expected: both commands exit 0.

- [ ] **Step 2: Add the monitor import and initialize it after `Scene` creation**

Apply an edit equivalent to:

```python
from visible_training_monitor import VisibleTrainingMonitor

# inside training(), after scene = Scene(...)
monitor = VisibleTrainingMonitor.from_environment(scene.model_path, opt.iterations)
early_stopped = False
last_iteration = first_iter
```

- [ ] **Step 3: Add progress and synchronous evaluation hooks**

Inside the no-grad block:

```python
last_iteration = iteration
if iteration % 10 == 0:
    monitor.update_progress(iteration, ema_loss_for_log)

stop_requested = False
if iteration in monitor.eval_iterations:
    stop_requested = monitor.evaluate(
        iteration,
        scene.getTestCameras(),
        render,
        (pipe, background, 1.0, SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
        dataset.train_test_exp,
    )
```

Run the official save and checkpoint blocks for the iteration before honoring `stop_requested`. If true, set `early_stopped = True`, close the progress bar, and `break` before the next iteration. After the loop:

```python
if early_stopped:
    monitor.finalize("early_stopped", last_iteration, "quality_drop")
else:
    monitor.finalize("completed", last_iteration, "max_iterations")
```

- [ ] **Step 4: Disable duplicate official test reporting in the launcher contract**

The later launcher must pass `--test_iterations 15001`; monitor evaluations remain governed by `MYGS_EVAL_ITERATIONS`. Saving and checkpoint arguments still contain all formal evaluation iterations.

- [ ] **Step 5: Generate and reset the tracked patch safely**

Run:

```bash
mkdir -p patches/gaussian-splatting
git -C tools/gaussian-splatting diff -- train.py > patches/gaussian-splatting/visible-monitored-training.patch
git -C tools/gaussian-splatting apply --reverse --check /home/pch/myGS/patches/gaussian-splatting/visible-monitored-training.patch
git -C tools/gaussian-splatting apply --reverse /home/pch/myGS/patches/gaussian-splatting/visible-monitored-training.patch
git -C tools/gaussian-splatting apply --check /home/pch/myGS/patches/gaussian-splatting/visible-monitored-training.patch
```

Expected: reverse-check succeeds before the reverse application; forward-check succeeds after it. This changes only the explicit `train.py` file and does not delete any file.

- [ ] **Step 6: Apply the patch and verify Python compilation**

```bash
git -C tools/gaussian-splatting apply /home/pch/myGS/patches/gaussian-splatting/visible-monitored-training.patch
tools/miniconda3/bin/conda run -n mygs-3dgs-cu128 python -m py_compile tools/gaussian-splatting/train.py scripts/visible_training_monitor.py
```

Expected: both exit 0. If forward-check fails but reverse-check succeeds, the patch is already applied and must not be applied a second time.

- [ ] **Step 7: Commit Task 3**

```bash
git add patches/gaussian-splatting/visible-monitored-training.patch
git commit -m "Add monitored 3DGS training patch"
git push
```

---

### Task 4: Persistent Launcher and Read-Only Monitor

**Files:**
- Create: `/home/pch/myGS/scripts/run_visible_training.sh`
- Create: `/home/pch/myGS/scripts/monitor_visible_training.sh`
- Modify: `/home/pch/myGS/.gitignore`

**Interfaces:**
- `bash scripts/run_visible_training.sh smoke` starts a retained short validation run.
- `bash scripts/run_visible_training.sh start` starts the formal 15000-step persistent run and prints `run_id`, output path, log path, and PID.
- `bash scripts/run_visible_training.sh resume <run_dir>` validates and resumes from the last checkpoint in that same directory.
- `bash scripts/monitor_visible_training.sh [run_dir]` performs read-only reporting.

- [ ] **Step 1: Add launcher preflight checks**

Start `scripts/run_visible_training.sh` with this preflight function and call it before creating a run directory:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

preflight() {
  local source_dir checkout_head prepared
  source_dir="$(scene_processed_dir visible)"
  checkout_head="$(git -C "${GAUSSIAN_SPLATTING_DIR}" rev-parse --short HEAD)"
  prepared="$(find "${source_dir}/images" -maxdepth 1 \( -type f -o -type l \) -name "$(scene_pattern visible)" | wc -l)"

test -x "${CONDA_ROOT}/bin/conda"
test -f "${GAUSSIAN_SPLATTING_DIR}/train.py"
  test "${checkout_head}" = "54c035f"
  test -d "${source_dir}/images"
  test -f "${source_dir}/sparse/0/cameras.bin"
  test -f "${source_dir}/sparse/0/images.bin"
  test -f "${source_dir}/sparse/0/points3D.bin"
  test "${prepared}" = "339"
  CUDA_VISIBLE_DEVICES=0 "${CONDA_ROOT}/bin/conda" run -n "${CONDA_ENV_NAME}" \
    python -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1"
}
```

It must also count 339 prepared images, verify GPU 0 is visible, verify the monitor patch is applied or safely apply it once, and reject launch if an existing recorded visible PID is still alive.

- [ ] **Step 2: Add formal and smoke parameter sets**

Formal values:

```bash
iterations=15000
eval_iterations=(5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000)
```

Smoke values retained under a distinct `smoke-<timestamp>` run:

```bash
iterations=20
eval_iterations=(10 20)
```

Both modes pass `--eval`, `--disable_viewer`, `--test_iterations $((iterations + 1))`, and identical save/checkpoint iteration arrays. Smoke thresholds and output must not alter formal defaults.

- [ ] **Step 3: Add persistent execution and exit finalization**

Start the worker with `nohup setsid`, `CUDA_VISIBLE_DEVICES=0`, `PYTHONPATH=/home/pch/myGS/scripts`, `MYGS_MONITOR_DIR`, `MYGS_EVAL_ITERATIONS`, and `MYGS_TRAIN_PID`. Redirect stdout/stderr to one log. Save only the exact launcher PID to `train.pid`. After Python exits, invoke `finalize-exit` with its exit code.

- [ ] **Step 4: Add safe resume validation**

Resume must require an explicit existing run directory, terminal/non-running PID, matching visible source path in `cfg_args`, and the numerically greatest `chkpnt<N>.pth`. It must preserve `metrics.csv` and pass `--start_checkpoint` for that exact file. It must never search outside the explicit run directory and never delete partial output.

- [ ] **Step 5: Implement read-only monitoring**

`monitor_visible_training.sh` must print:

- resolved run directory;
- parsed `status.json`;
- complete `metrics.csv` if present;
- whether the exact PID in `train.pid` is alive;
- GPU 0 memory/utilization from `nvidia-smi`;
- last 30 lines of the log path recorded in status.

It must not send signals, mutate status, or create output.

Use this shell structure:

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/pch/myGS"
run_dir="${1:-}"
if [[ -z "${run_dir}" ]]; then
  run_dir="$(find "${PROJECT_ROOT}/outputs/visible" -mindepth 1 -maxdepth 1 -type d -printf '%p\n' | sort | tail -1)"
fi
test -n "${run_dir}"
test -d "${run_dir}"
echo "run_dir=${run_dir}"

if [[ -f "${run_dir}/status.json" ]]; then
  "${PROJECT_ROOT}/tools/miniconda3/bin/conda" run -n mygs-3dgs-cu128 \
    python -m json.tool "${run_dir}/status.json"
fi
if [[ -f "${run_dir}/metrics.csv" ]]; then
  sed -n '1,200p' "${run_dir}/metrics.csv"
fi
if [[ -f "${run_dir}/train.pid" ]]; then
  pid="$(<"${run_dir}/train.pid")"
  if kill -0 "${pid}" 2>/dev/null; then
    echo "pid=${pid} alive=true"
  else
    echo "pid=${pid} alive=false"
  fi
fi
nvidia-smi --id=0 --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
log_file="$(find "${PROJECT_ROOT}/logs" -maxdepth 1 -type f -name '*-train-visible.log' -printf '%p\n' | sort | tail -1)"
if [[ -n "${log_file}" ]]; then
  echo "log=${log_file}"
  tail -n 30 "${log_file}"
fi
```

- [ ] **Step 6: Extend `.gitignore` without deleting files**

Add:

```gitignore
tools/
outputs/
logs/*.log
```

Keep the existing specific ignore rules. Do not remove or clean the existing `tools/` contents.

- [ ] **Step 7: Verify shell scripts**

```bash
bash -n scripts/run_visible_training.sh
bash -n scripts/monitor_visible_training.sh
shellcheck scripts/run_visible_training.sh scripts/monitor_visible_training.sh
```

Expected: syntax checks pass. If `shellcheck` is unavailable, record that fact and use `bash -n` plus smoke execution; do not install system packages without approval.

- [ ] **Step 8: Commit Task 4**

```bash
git add .gitignore scripts/run_visible_training.sh scripts/monitor_visible_training.sh
git commit -m "Add persistent visible training runner"
git push
```

---

### Task 5: LPIPS Preflight and Smoke Run

**Files:**
- Modify if required by evidence: `/home/pch/myGS/scripts/run_visible_training.sh`
- Modify if required by evidence: `/home/pch/myGS/scripts/visible_training_monitor.py`
- Modify: `/home/pch/myGS/AGENTS.md`

- [ ] **Step 1: Run all offline verification**

```bash
cd /home/pch/myGS
tools/miniconda3/bin/conda run -n mygs-3dgs-cu128 python -m pytest tests/test_visible_training_monitor.py -q
tools/miniconda3/bin/conda run -n mygs-3dgs-cu128 python -m py_compile scripts/visible_training_monitor.py tools/gaussian-splatting/train.py
bash -n scripts/run_visible_training.sh
bash -n scripts/monitor_visible_training.sh
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 2: Cache and verify LPIPS VGG weights before training**

From `/home/pch/myGS/tools/gaussian-splatting`, instantiate `LPIPS(net_type="vgg")` once in the project Conda environment on GPU 0 and evaluate two small equal tensors. Expected: finite LPIPS value and cached VGG/LPIPS weights. If download fails because of restricted network, retry only through the approved server access path; do not start smoke or formal training without a successful load.

- [ ] **Step 3: Start the retained smoke run**

```bash
cd /home/pch/myGS
bash scripts/run_visible_training.sh smoke
```

Expected: prints a unique smoke run directory, log path, and PID.

- [ ] **Step 4: Monitor smoke to a terminal state**

Poll at short intervals using the read-only monitor. Expected within the smoke run:

- status progresses through `training` and `evaluating`;
- `metrics.csv` contains iterations 10 and 20;
- `point_cloud/iteration_10` and `point_cloud/iteration_20` exist;
- `chkpnt10.pth` and `chkpnt20.pth` exist;
- final state is `completed` unless the intentionally tiny smoke metrics trigger the real early-stop rule at 20, in which case `early_stopped` is acceptable.

- [ ] **Step 5: Diagnose any failure systematically**

On failure, stop formal execution, preserve the smoke directory and log, identify the root cause, add a failing automated test when applicable, make one minimal fix, and rerun all Task 5 verification. Do not delete failed smoke output.

- [ ] **Step 6: Record smoke evidence in conversation 4 and commit**

Append exact run directory, log, metrics, terminal state, GPU, and verification commands to `AGENTS.md` under conversation 4.

```bash
git add AGENTS.md
git commit -m "Record visible training smoke verification"
git push
```

---

### Task 6: Formal Visible Training and Continuous Monitoring

**Files:**
- Generated only: `/home/pch/myGS/outputs/visible/<run_id>/`
- Generated only: `/home/pch/myGS/logs/<timestamp>-train-visible.log`
- Modify after terminal result: `/home/pch/myGS/AGENTS.md`

- [ ] **Step 1: Recheck formal preconditions immediately before launch**

Verify GPU 0 is idle, no recorded visible training PID is alive, smoke state is terminal, source count remains 339, sparse files exist, tests pass, patch is applied, and Git tracked files are clean. Existing unrelated ignored/untracked tool files do not authorize deletion.

- [ ] **Step 2: Start exactly one formal visible run**

```bash
cd /home/pch/myGS
bash scripts/run_visible_training.sh start
```

Capture printed `run_id`, output directory, log path, and PID. Confirm no thermal process was started.

- [ ] **Step 3: Monitor live progress until the first quality baseline**

Use `scripts/monitor_visible_training.sh <run_dir>` repeatedly. Report meaningful progress to the user without gaps longer than 60 seconds while actively attached. At iteration 5000, verify the first complete PSNR/SSIM/LPIPS row, checkpoint, point cloud, and five fixed-view render/GT pairs.

- [ ] **Step 4: Enforce each later early-stop decision**

For every reached checkpoint from 6000 onward, verify the current row compares with the immediately previous row using the approved thresholds. If quality drops, verify state becomes `early_stopped`, last iteration equals that checkpoint, and the recommended checkpoint is the preceding non-degraded checkpoint. Otherwise continue monitoring to the next checkpoint, at most 15000.

- [ ] **Step 5: Verify terminal outputs**

Run read-only checks confirming:

- state is `completed` or `early_stopped`;
- summary stop reason, last iteration, and best iteration agree with metrics;
- best PLY and checkpoint exist;
- every reached checkpoint has a complete metric row and fixed-view outputs;
- no thermal training output was created by this task;
- raw data count and names remain unchanged.

- [ ] **Step 6: Update conversation 4 with final evidence**

Record run directory, log, terminal state, last/best iteration, complete metrics table, stop decision, best PLY/checkpoint, and raw-data verification. Mark only this independent visible task complete; do not mark the earlier visible+thermal Task 6 complete.

- [ ] **Step 7: Commit and push the final engineering record**

```bash
git add AGENTS.md
git commit -m "Record monitored visible training result"
git push
```

Expected: push succeeds; generated outputs remain ignored.

---

## Self-Review

### Spec coverage

- Visible-only, GPU 0, 15000 maximum, `--eval`, 296/43 split, and fixed checkpoints are covered in Global Constraints and Tasks 3–6.
- PSNR/SSIM/LPIPS, fixed renders, exact early stop, best checkpoint, and retained outputs are covered in Tasks 1–3 and 6.
- Live status, persistence, PID, logs, failure state, and safe resume are covered in Tasks 2 and 4.
- LPIPS preflight and smoke-before-formal requirements are covered in Task 5.
- Git/data/deletion boundaries and independent conversation 4 logging are covered throughout and in Tasks 4–6.

### Placeholder scan

- The plan contains no deferred implementation markers or unspecified error-handling steps.
- Formal values and smoke-only overrides are explicit and separate.

### Interface consistency

- `MetricRecord`, `quality_dropped`, `select_best`, `VisibleTrainingMonitor`, `record_metric`, `evaluate`, and `finalize` use the same names across tasks.
- `MYGS_MONITOR_DIR`, `MYGS_EVAL_ITERATIONS`, and `MYGS_TRAIN_PID` are the only monitoring environment interfaces.
- Terminal states are consistently `completed`, `early_stopped`, `failed`, and `preflight_failed`.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-10-visible-3dgs-monitored-training-implementation.md`.

Two execution options:

1. **Subagent-Driven (recommended by the generic workflow)** — dispatch a fresh subagent per task with review gates.
2. **Inline Execution** — execute this plan in the current session with `superpowers:executing-plans` and checkpoints.

For this project, the previously selected working style is Inline Execution, so Inline Execution is the default recommendation unless the user explicitly chooses delegation.
