# AGENTS.md

## 操作约束

- 禁止批量删除文件或目录。
- 不要使用 `del /s`、`rd /s`、`rmdir /s`、`Remove-Item -Recurse`、`rm -rf`。
- 需要删除文件时，只能一次删除一个明确路径的文件。
- 如果需要批量删除文件，应停止操作，并请求用户手动删除。
- 原始数据目录保持只读，不移动、不删除、不改名。

## 项目目标

- 使用 3D Gaussian Splatting 分别重建可见光和红外无人机环绕场景。
- 可见光数据：339 张，服务器路径 `/home/pch/myGS/datasets/uav_3dgs/raw/visible`。
- 红外数据：339 张，服务器路径 `/home/pch/myGS/datasets/uav_3dgs/raw/thermal`。
- 服务器：`pch-5090`，GPU 为两张 NVIDIA GeForce RTX 5090。
- 训练目标：项目 Conda 环境使用 CUDA 12.8 / PyTorch cu128，不切换系统 CUDA。
- GitHub 远端：`git@github.com:YNUpanpan/myGS.git`。

## 工程边界

- `/home/pch/myGS/datasets/uav_3dgs/raw/`：原始数据，只读保留。
- `/home/pch/myGS/datasets/uav_3dgs/processed/`：COLMAP/3DGS 输入数据。
- `/home/pch/myGS/outputs/`：3DGS 训练输出。
- `/home/pch/myGS/logs/`：运行日志。
- `/home/pch/myGS/scripts/`：可复现脚本。
- `/home/pch/myGS/docs/`：设计、计划与结果记录。

## 任务日志

### 2026-07-09 对话 1：需求澄清与方案设计

#### 目标

- 明确可见光与红外 3DGS 重建的整体流程。
- 在正式实现前确认服务器、数据、环境、代码基线、COLMAP 策略、训练目标和 GitHub 上传范围。
- 将本次对话的任务记录写入 `AGENTS.md`。

#### 已确认决策

- 主执行环境：直接操作 Ubuntu 5090 服务器 `pch-5090`。
- 项目路径：`/home/pch/myGS`。
- CUDA 策略：项目 Conda 环境使用 CUDA 12.8 / PyTorch cu128，不改系统 CUDA。
- Conda 策略：在 `/home/pch/myGS/tools/miniconda3` 安装项目专用 Miniconda。
- 3DGS 基线：官方 `graphdeco-inria/gaussian-splatting`。
- 位姿策略：可见光和红外分别独立运行 COLMAP。
- 产出目标：完整可复现工程，并跑通可见光和红外两套基线训练。
- GitHub 上传范围：只上传工程文件、脚本、配置、文档和日志摘要；不上传原始数据、COLMAP 中间文件、训练模型或大文件。

#### 只读探查结果

- `/home/pch/myGS` 初始不是 git 仓库，且没有 `AGENTS.md`。
- 服务器可连接，主机名为 `ubuntu`。
- GPU：2 x NVIDIA GeForce RTX 5090，每张约 32607 MiB 显存，驱动版本 570.144。
- 系统 `nvcc --version` 显示 CUDA 11.5，因此训练将依赖项目 Conda 环境中的 CUDA 12.8/PyTorch cu128。
- 系统已有 `/usr/bin/colmap` 和 `/usr/bin/git`。
- 数据目录结构：
  - `/home/pch/myGS/datasets/uav_3dgs/raw/visible`
  - `/home/pch/myGS/datasets/uav_3dgs/raw/thermal`
- 数据数量：
  - visible: 339 张 `*_V.JPG`
  - thermal: 339 张 `*_T.JPG`
  - `.MRK`: 未在服务器数据目录中找到

#### 已完成

- 确认路线 1：标准可复现基线。
- 确认工程目录、数据流、脚本设计、验证标准和 GitHub 排除规则。
- 创建规格文档：`docs/superpowers/specs/2026-07-09-uav-3dgs-baseline-design.md`。

#### 待执行

- 用户审阅规格文档。
- 审阅通过后进入实施计划。
- 编写环境安装、数据准备、COLMAP、训练和汇总脚本。
- 初始化并推送 GitHub 仓库。

### 2026-07-09 对话 2：规格确认与实施计划

#### 目标

- 用户确认规格文档后，编写可执行实施计划。
- 继续保证本次对话任务记录写入 `AGENTS.md`。

#### 已确认

- 用户已确认 `/home/pch/myGS/docs/superpowers/specs/2026-07-09-uav-3dgs-baseline-design.md` 可以进入下一步。

#### 已完成

- 创建实施计划：`docs/superpowers/plans/2026-07-09-uav-3dgs-baseline-implementation.md`。
- 计划覆盖环境检查、项目 Conda、PyTorch cu128、官方 3DGS、数据准备、COLMAP、训练、汇总和 GitHub 推送。

#### 待执行

- 用户选择执行方式：Subagent-Driven 或 Inline Execution。
- 按任务逐步实现脚本并验证。

#### Implementation Progress

- 用户明确选择 Inline Execution，并同意直接在 `/home/pch/myGS` 的 `main` 上实现。
- Created shared scene configuration and environment check scripts.
- Verified `scripts/check_environment.sh`: visible `339/339`, thermal `339/339`, two RTX 5090 GPUs visible, `/usr/bin/colmap` and `/usr/bin/git` available.
