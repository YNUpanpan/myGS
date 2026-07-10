# Visible 3DGS 监控训练实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 仅在 GPU 0 上训练 visible 场景，最多训练 15000 步；从 5000 步开始每 1000 步同步评估固定的 43 张测试图，提供实时进度，并在已确认的质量下降判据触发时精确停止在检查点。

**架构：** 官方 3DGS checkout 固定为 `54c035f`。将可复用的进度监控、指标历史、原子状态输出和早停逻辑放入受 Git 跟踪的 Python 模块，再通过一个受跟踪的最小补丁从官方连续训练循环调用该模块。Bash 启动脚本负责全部预检并启动持久工作进程，独立的只读监控脚本负责报告状态，不控制或删除运行结果。

**技术栈：** Bash、Python 3、dataclasses、JSON/CSV、PyTorch cu128、torchvision、官方 3DGS 渲染器、LPIPS VGG、pytest/unittest、Git、Ubuntu、NVIDIA RTX 5090。

## 全局约束

- 服务器为 `pch-5090`，项目根目录为 `/home/pch/myGS`。
- 场景输入为 `/home/pch/myGS/datasets/uav_3dgs/processed/visible`。
- 本次只使用 GPU 0，不启动 thermal 训练。
- 使用项目 Conda 环境 `mygs-3dgs-cu128`，不修改系统 CUDA。
- 应用监控补丁前，必须确认官方 3DGS checkout HEAD 为 `54c035f`。
- 使用 `--eval`，将 339 张 visible 图像固定划分为 296 张训练图和 43 张测试图。
- 正式检查点为 `5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000`。
- 相对上一检查点，PSNR 下降超过 `0.05 dB`，且 SSIM 下降超过 `0.001` 或 LPIPS 上升超过 `0.001` 时早停。
- 原始数据保持只读；禁止递归删除或批量删除；保留 smoke run 和正式运行输出。
- Git 只跟踪脚本、补丁、测试、文档和摘要，不跟踪数据、工具、日志、渲染图、点云或 checkpoint。
- 按本项目此前授权，直接在服务器 `main` 分支实施。

---

## 文件结构

- 创建 `scripts/visible_training_monitor.py`：负责纯阈值逻辑、最佳检查点选择、原子状态文件、CSV 历史、GPU 评估、固定视角渲染保存和 CLI 收尾。
- 创建 `tests/test_visible_training_monitor.py`：为阈值、最佳结果选择、原子状态、CSV 历史和终止状态保留提供确定性单元测试。
- 创建 `patches/gaussian-splatting/visible-monitored-training.patch`：为进度更新、同步评估、早停和终止状态增加最小钩子。
- 创建 `scripts/run_visible_training.sh`：负责预检、补丁验证/应用、LPIPS 缓存检查、正式/smoke 启动、持久运行和续训校验。
- 创建 `scripts/monitor_visible_training.sh`：只读显示最近运行的状态、指标、PID、GPU 和日志。
- 修改 `.gitignore`：忽略全部项目本地工具和生成的运行状态，但不删除它们。
- 修改 `AGENTS.md`：在对话 4 下追加实施、smoke run、实时运行和最终结果证据。

---

### 任务 1：纯监控状态与早停逻辑

**文件：**
- 创建：`/home/pch/myGS/scripts/visible_training_monitor.py`
- 创建：`/home/pch/myGS/tests/test_visible_training_monitor.py`

**接口：**
- 产出：`MetricRecord`、`quality_dropped(previous, current) -> bool`、`select_best(records) -> MetricRecord`、`atomic_write_json(path, payload)` 和 `RunFiles`。
- 依赖：本任务只使用标准库 `csv`、`json`、`os`、`tempfile`、`dataclasses`、`pathlib` 和 `typing`。

- [ ] **步骤 1：编写阈值与最佳检查点选择的失败测试**

创建 `tests/test_visible_training_monitor.py`，写入以下初始测试：

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

- [ ] **步骤 2：运行测试并确认按预期失败**

运行：

```bash
cd /home/pch/myGS
tools/miniconda3/bin/conda run -n mygs-3dgs-cu128 python -m pytest tests/test_visible_training_monitor.py -q
```

预期：由于 `scripts/visible_training_monitor.py` 尚不存在，测试收集失败。

- [ ] **步骤 3：实现最小纯逻辑模块**

创建 `scripts/visible_training_monitor.py`，内容如下：

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

- [ ] **步骤 4：运行单元测试并确认通过**

Run the Step 2 command again.

预期：输出 `3 passed`。

- [ ] **步骤 5：增加原子文件与 CSV 测试**

在测试文件顶部增加 `import csv` 和 `import json`，然后追加：

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

不要手动删除临时目录；其生命周期由 pytest 管理。

- [ ] **步骤 6：运行任务 1 的完整测试文件**

预期：全部测试通过，且 `python -m py_compile scripts/visible_training_monitor.py` 退出码为 0。

- [ ] **步骤 7：提交任务 1**

```bash
git add scripts/visible_training_monitor.py tests/test_visible_training_monitor.py
git commit -m "Add visible training monitor core"
git push
```

---

### 任务 2：运行时进度、测试指标与终止状态

**文件：**
- 修改：`/home/pch/myGS/scripts/visible_training_monitor.py`
- 修改：`/home/pch/myGS/tests/test_visible_training_monitor.py`

**接口：**
- 产出：`VisibleTrainingMonitor.from_environment(model_path, total_iterations)`、`update_progress(iteration, ema_loss)`、`evaluate(iteration, cameras, render_func, render_args, train_test_exp) -> bool` 和 `finalize(state, iteration, reason)`。
- 依赖：`MYGS_MONITOR_DIR`、`MYGS_EVAL_ITERATIONS`、`MYGS_TRAIN_PID`、PyTorch 张量、3DGS `render_func` 和固定测试相机。

- [ ] **步骤 1：编写生命周期失败测试**

若文件尚未导入 `json`，先增加 `import json`，再追加以下生命周期测试：

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

- [ ] **步骤 2：确认新的生命周期测试失败**

预期：由于 `VisibleTrainingMonitor` 尚未定义，测试失败。

- [ ] **步骤 3：实现生命周期方法与环境校验**

增加 `VisibleTrainingMonitor`，严格满足以下不变量：

- `MYGS_MONITOR_DIR` must resolve to the same directory as `model_path`.
- 正式评估迭代默认使用已批准列表，并且必须严格递增。
- `update_progress` atomically writes state `training` every 10 iterations and computes ETA from monotonic elapsed time.
- `record_metric` compares only with the immediately previous record, appends CSV, updates best metrics, and returns the early-stop decision.
- `finalize` writes terminal `status.json` and `summary.json`; a later shell finalizer must not overwrite `completed` or `early_stopped` with `failed`.

- [ ] **步骤 4：实现每个检查点一次的 GPU 评估**

在 `evaluate` 中：

1. Set status phase to `evaluating`.
2. Lazily instantiate one `lpipsPyTorch.modules.lpips.LPIPS(net_type="vgg")` model on CUDA and reuse it.
3. Render each of the 43 cameras under `torch.no_grad()`.
4. Compute mean PSNR with `utils.image_utils.psnr`, SSIM with `utils.loss_utils.ssim`, and LPIPS with the cached VGG model.
5. Save renders and ground truths for the first five test cameras to `evaluation/iteration_<N>/renders` and `gt` using `torchvision.utils.save_image`.
6. Call `record_metric` and return its Boolean result.
7. Release per-view tensors and call `torch.cuda.empty_cache()` after evaluation.

- [ ] **步骤 5：增加 CLI 退出状态收尾器**

支持：

```bash
python scripts/visible_training_monitor.py finalize-exit \
  --run-dir /home/pch/myGS/outputs/visible/<run_id> \
  --exit-code 1
```

If current state is nonterminal, set state `failed`, preserve last iteration, and record `process_exit_<code>`; otherwise leave the terminal files unchanged.

- [ ] **步骤 6：运行单元测试与静态编译检查**

预期：全部监控测试通过，`py_compile` 退出码为 0。

- [ ] **步骤 7：提交任务 2**

```bash
git add scripts/visible_training_monitor.py tests/test_visible_training_monitor.py
git commit -m "Add visible quality evaluation lifecycle"
git push
```

---

### 任务 3：官方 3DGS 训练循环的最小钩子补丁

**文件：**
- 创建：`/home/pch/myGS/patches/gaussian-splatting/visible-monitored-training.patch`
- 仅在已忽略的 checkout 中修改：`/home/pch/myGS/tools/gaussian-splatting/train.py`

**接口：**
- 依赖：`PYTHONPATH` 中的 `VisibleTrainingMonitor` 和现有官方 `training()` 循环。
- 产出：每 10 步进度更新、同步检查点评估、精确早停退出和终止状态。

- [ ] **步骤 1：编辑前验证补丁基线**

运行：

```bash
test "$(git -C tools/gaussian-splatting rev-parse --short HEAD)" = "54c035f"
git -C tools/gaussian-splatting diff --exit-code -- train.py
```

预期：两个命令的退出码均为 0。

- [ ] **步骤 2：导入监控模块并在创建 `Scene` 后初始化**

应用等价于以下内容的编辑：

```python
from visible_training_monitor import VisibleTrainingMonitor

# inside training(), after scene = Scene(...)
monitor = VisibleTrainingMonitor.from_environment(scene.model_path, opt.iterations)
early_stopped = False
last_iteration = first_iter
```

- [ ] **步骤 3：增加进度与同步评估钩子**

在 no-grad 代码块中：

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

先执行当前迭代的官方保存和 checkpoint 代码块，再处理 `stop_requested`。若为真，设置 `early_stopped = True`，关闭进度条，并在进入下一迭代前 `break`。循环结束后：

```python
if early_stopped:
    monitor.finalize("early_stopped", last_iteration, "quality_drop")
else:
    monitor.finalize("completed", last_iteration, "max_iterations")
```

- [ ] **步骤 4：在启动参数中禁用重复的官方测试报告**

后续启动脚本必须传入 `--test_iterations 15001`；监控评估仍由 `MYGS_EVAL_ITERATIONS` 控制。保存与 checkpoint 参数继续包含全部正式评估迭代。

- [ ] **步骤 5：安全生成并反向应用受跟踪补丁**

运行：

```bash
mkdir -p patches/gaussian-splatting
git -C tools/gaussian-splatting diff -- train.py > patches/gaussian-splatting/visible-monitored-training.patch
git -C tools/gaussian-splatting apply --reverse --check /home/pch/myGS/patches/gaussian-splatting/visible-monitored-training.patch
git -C tools/gaussian-splatting apply --reverse /home/pch/myGS/patches/gaussian-splatting/visible-monitored-training.patch
git -C tools/gaussian-splatting apply --check /home/pch/myGS/patches/gaussian-splatting/visible-monitored-training.patch
```

预期：反向应用前的反向检查成功，反向应用后的正向检查成功。该操作只修改明确的 `train.py`，不删除任何文件。

- [ ] **步骤 6：应用补丁并验证 Python 编译**

```bash
git -C tools/gaussian-splatting apply /home/pch/myGS/patches/gaussian-splatting/visible-monitored-training.patch
tools/miniconda3/bin/conda run -n mygs-3dgs-cu128 python -m py_compile tools/gaussian-splatting/train.py scripts/visible_training_monitor.py
```

预期：两个命令退出码均为 0。若正向检查失败而反向检查成功，说明补丁已经应用，不得重复应用。

- [ ] **步骤 7：提交任务 3**

```bash
git add patches/gaussian-splatting/visible-monitored-training.patch
git commit -m "Add monitored 3DGS training patch"
git push
```

---

### 任务 4：持久启动脚本与只读监控脚本

**文件：**
- 创建：`/home/pch/myGS/scripts/run_visible_training.sh`
- 创建：`/home/pch/myGS/scripts/monitor_visible_training.sh`
- 修改：`/home/pch/myGS/.gitignore`

**接口：**
- `bash scripts/run_visible_training.sh smoke` starts a retained short validation run.
- `bash scripts/run_visible_training.sh start` starts the formal 15000-step persistent run and prints `run_id`, output path, log path, and PID.
- `bash scripts/run_visible_training.sh resume <run_dir>` validates and resumes from the last checkpoint in that same directory.
- `bash scripts/monitor_visible_training.sh [run_dir]` performs read-only reporting.

- [ ] **步骤 1：增加启动前预检**

在 `scripts/run_visible_training.sh` 开头加入以下预检函数，并在创建运行目录前调用：

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

脚本还必须确认已准备图像为 339 张、GPU 0 可见、监控补丁已应用或可以安全应用一次；如果已有记录的 visible PID 仍存活，则拒绝启动。

- [ ] **步骤 2：增加正式运行和 smoke run 参数集**

正式运行参数：

```bash
iterations=15000
eval_iterations=(5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000)
```

smoke run 使用独立的 `smoke-<timestamp>` 运行目录，并采用：

```bash
iterations=20
eval_iterations=(10 20)
```

两种模式都传入 `--eval`、`--disable_viewer`、`--test_iterations $((iterations + 1))`，并使用相同的保存与 checkpoint 迭代数组。smoke run 的阈值和输出不得修改正式运行默认值。

- [ ] **步骤 3：增加持久执行与退出状态收尾**

使用 `nohup setsid` 启动工作进程，并设置 `CUDA_VISIBLE_DEVICES=0`、`PYTHONPATH=/home/pch/myGS/scripts`、`MYGS_MONITOR_DIR`、`MYGS_EVAL_ITERATIONS` 和 `MYGS_TRAIN_PID`。将 stdout/stderr 重定向到同一个日志，只把准确的启动 PID 写入 `train.pid`。Python 退出后，用其退出码调用 `finalize-exit`。

- [ ] **步骤 4：增加安全续训校验**

续训必须要求明确且已存在的运行目录、已经终止或不再运行的 PID、`cfg_args` 中匹配的 visible 源路径，以及数字序号最大的 `chkpnt<N>.pth`。必须保留 `metrics.csv`，并把该明确文件传给 `--start_checkpoint`。不得在指定运行目录之外搜索，也不得删除不完整输出。

- [ ] **步骤 5：实现只读监控**

`monitor_visible_training.sh` must print:

- resolved run directory;
- parsed `status.json`;
- complete `metrics.csv` if present;
- whether the exact PID in `train.pid` is alive;
- GPU 0 memory/utilization from `nvidia-smi`;
- last 30 lines of the log path recorded in status.

脚本不得发送信号、修改状态或创建输出。

采用以下 shell 结构：

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

- [ ] **步骤 6：扩展 `.gitignore`，但不删除文件**

增加：

```gitignore
tools/
outputs/
logs/*.log
```

保留现有的细分忽略规则，不移除或清理已有 `tools/` 内容。

- [ ] **步骤 7：验证 shell 脚本**

```bash
bash -n scripts/run_visible_training.sh
bash -n scripts/monitor_visible_training.sh
shellcheck scripts/run_visible_training.sh scripts/monitor_visible_training.sh
```

预期：语法检查通过。如果 `shellcheck` 不可用，记录这一事实，并使用 `bash -n` 加 smoke run 验证；未经批准不得安装系统软件包。

- [ ] **步骤 8：提交任务 4**

```bash
git add .gitignore scripts/run_visible_training.sh scripts/monitor_visible_training.sh
git commit -m "Add persistent visible training runner"
git push
```

---

### 任务 5：LPIPS 预检与 smoke run

**文件：**
- 若证据表明需要则修改：`/home/pch/myGS/scripts/run_visible_training.sh`
- 若证据表明需要则修改：`/home/pch/myGS/scripts/visible_training_monitor.py`
- 修改：`/home/pch/myGS/AGENTS.md`

- [ ] **步骤 1：运行全部离线验证**

```bash
cd /home/pch/myGS
tools/miniconda3/bin/conda run -n mygs-3dgs-cu128 python -m pytest tests/test_visible_training_monitor.py -q
tools/miniconda3/bin/conda run -n mygs-3dgs-cu128 python -m py_compile scripts/visible_training_monitor.py tools/gaussian-splatting/train.py
bash -n scripts/run_visible_training.sh
bash -n scripts/monitor_visible_training.sh
git diff --check
```

预期：全部命令退出码均为 0。

- [ ] **步骤 2：训练前缓存并验证 LPIPS VGG 权重**

在 `/home/pch/myGS/tools/gaussian-splatting` 中，使用项目 Conda 环境在 GPU 0 上实例化一次 `LPIPS(net_type="vgg")`，并评估两个相同的小张量。预期获得有限的 LPIPS 数值，同时 VGG/LPIPS 权重已缓存。如果因网络限制下载失败，只能通过已批准的服务器访问路径重试；权重加载成功前不得启动 smoke run 或正式训练。

- [ ] **步骤 3：启动保留结果的 smoke run**

```bash
cd /home/pch/myGS
bash scripts/run_visible_training.sh smoke
```

预期：输出唯一的 smoke run 目录、日志路径和 PID。

- [ ] **步骤 4：监控 smoke run 直到终止状态**

使用只读监控脚本短间隔轮询。smoke run 中预期：

- status progresses through `training` and `evaluating`;
- `metrics.csv` contains iterations 10 and 20;
- `point_cloud/iteration_10` and `point_cloud/iteration_20` exist;
- `chkpnt10.pth` and `chkpnt20.pth` exist;
- final state is `completed` unless the intentionally tiny smoke metrics trigger the real early-stop rule at 20, in which case `early_stopped` is acceptable.

- [ ] **步骤 5：系统化诊断任何失败**

若失败，停止进入正式训练，保留 smoke run 目录和日志，定位根因；适用时增加一个失败的自动化测试，只实施一个最小修复，然后重新运行任务 5 的全部验证。不要删除失败的 smoke run 输出。

- [ ] **步骤 6：在对话 4 中记录 smoke run 证据并提交**

Append exact run directory, log, metrics, terminal state, GPU, and verification commands to `AGENTS.md` under conversation 4.

```bash
git add AGENTS.md
git commit -m "Record visible training smoke verification"
git push
```

---

### 任务 6：Visible 正式训练与持续监控

**文件：**
- 仅生成：`/home/pch/myGS/outputs/visible/<run_id>/`
- 仅生成：`/home/pch/myGS/logs/<timestamp>-train-visible.log`
- 得到终止结果后修改：`/home/pch/myGS/AGENTS.md`

- [ ] **步骤 1：正式启动前再次检查前置条件**

确认 GPU 0 空闲、没有已记录且仍存活的 visible 训练 PID、smoke run 已进入终止状态、源图数量仍为 339、sparse 文件存在、测试通过、补丁已应用且 Git 已跟踪文件干净。现有无关的忽略或未跟踪工具文件不构成删除授权。

- [ ] **步骤 2：只启动一个 visible 正式运行**

```bash
cd /home/pch/myGS
bash scripts/run_visible_training.sh start
```

记录输出的 `run_id`、输出目录、日志路径和 PID，并确认没有启动 thermal 进程。

- [ ] **步骤 3：实时监控到首个质量基准点**

重复运行 `scripts/monitor_visible_training.sh <run_dir>`。主动监控期间，向用户报告有意义的进度，间隔不超过 60 秒。到达 5000 步时，验证首行完整 PSNR/SSIM/LPIPS、checkpoint、点云以及 5 组固定视角 render/GT。

- [ ] **步骤 4：执行后续每个检查点的早停判断**

从 6000 步开始，对每个已到达检查点，确认当前指标行按已批准阈值与紧邻的上一行比较。若质量下降，确认状态变为 `early_stopped`、最后迭代等于该检查点，并推荐下降前的检查点；否则继续监控下一个检查点，最多到 15000 步。

- [ ] **步骤 5：验证终止输出**

运行只读检查，确认：

- state is `completed` or `early_stopped`;
- summary stop reason, last iteration, and best iteration agree with metrics;
- best PLY and checkpoint exist;
- every reached checkpoint has a complete metric row and fixed-view outputs;
- no thermal training output was created by this task;
- raw data count and names remain unchanged.

- [ ] **步骤 6：用最终证据更新对话 4**

记录运行目录、日志、终止状态、最后/最佳迭代、完整指标表、停止判断、最佳 PLY/checkpoint 和原始数据验证。只将本次独立 visible 任务标记为完成，不要把先前的 visible+thermal Task 6 标记为完成。

- [ ] **步骤 7：提交并推送最终工程记录**

```bash
git add AGENTS.md
git commit -m "Record monitored visible training result"
git push
```

预期：推送成功，生成输出继续保持忽略状态。

---

## 自审

### 规格覆盖

- 仅 visible、GPU 0、15000 步上限、`--eval`、296/43 划分和固定检查点由全局约束及任务 3–6 覆盖。
- PSNR/SSIM/LPIPS、固定渲染图、精确早停、最佳 checkpoint 和保留输出由任务 1–3、6 覆盖。
- 实时状态、持久运行、PID、日志、失败状态和安全续训由任务 2、4 覆盖。
- LPIPS 预检以及正式训练前必须 smoke run 的要求由任务 5 覆盖。
- Git/数据/删除边界和独立对话 4 日志贯穿全文，并由任务 4–6 落实。

### 占位符扫描

- 计划不含延后实现标记或未明确的错误处理步骤。
- 正式参数和仅用于 smoke run 的覆盖参数均已明确分开。

### 接口一致性

- `MetricRecord`, `quality_dropped`, `select_best`, `VisibleTrainingMonitor`, `record_metric`, `evaluate`, and `finalize` use the same names across tasks.
- `MYGS_MONITOR_DIR`, `MYGS_EVAL_ITERATIONS`, and `MYGS_TRAIN_PID` are the only monitoring environment interfaces.
- 终止状态统一使用 `completed`、`early_stopped`、`failed` 和 `preflight_failed`。

## 执行交接

计划已完成并保存到 `docs/superpowers/plans/2026-07-10-visible-3dgs-monitored-training-implementation.md`。

两种执行方式：

1. **子代理驱动（通用工作流推荐）**：每个任务分派新的子代理，并设置审查关口。
2. **当前会话内执行**：在当前会话中使用 `superpowers:executing-plans` 和检查点执行计划。

本项目此前已选择当前会话内执行，因此除非用户明确选择委派，否则默认推荐继续在当前会话内执行。
