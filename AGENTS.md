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

#### 下一步

- 用户审阅规格文档。
- 审阅通过后编写实施计划。

### 2026-07-09 对话 2：规格确认与实施计划

#### 目标

- 用户确认规格文档后，编写可执行实施计划。
- 继续保证本次对话任务记录写入 `AGENTS.md`。

#### 已确认决策

- 用户已确认 `/home/pch/myGS/docs/superpowers/specs/2026-07-09-uav-3dgs-baseline-design.md` 可以进入下一步。
- 用户明确选择 Inline Execution。
- 用户同意直接在服务器 `/home/pch/myGS` 的 `main` 分支上实现。

#### 只读探查结果

- `scripts/check_environment.sh` 验证通过：
  - visible 原始图片数量为 `339/339`。
  - thermal 原始图片数量为 `339/339`。
  - 服务器可见 2 张 NVIDIA GeForce RTX 5090。
  - `/usr/bin/colmap` 和 `/usr/bin/git` 可用。
- 项目 Conda 环境 `mygs-3dgs-cu128` 验证通过：
  - PyTorch：`2.11.0+cu128`。
  - PyTorch CUDA：`12.8`。
  - `torch.cuda.is_available()` 为 true。
  - 可见 GPU 数量为 2。
- 官方 3DGS 训练相关源码 checkout HEAD 与官方 GitHub SSH HEAD 校验一致：`54c035f`。
- 3DGS CUDA 扩展 import 验证通过：
  - `diff_gaussian_rasterization`
  - `simple_knn._C`
  - `fused_ssim`

#### 已完成

- 创建实施计划：`docs/superpowers/plans/2026-07-09-uav-3dgs-baseline-implementation.md`。
- 计划覆盖环境检查、项目 Conda、PyTorch cu128、官方 3DGS、数据准备、COLMAP、训练、汇总和 GitHub 推送。
- 创建 `configs/scenes.env`，统一记录项目根目录、数据目录、输出目录、日志目录、工具目录、场景名称、文件匹配规则和期望图片数量。
- 创建 `scripts/common.sh`，提供场景校验、路径解析、图片计数、日志路径和命令检查等公共函数。
- 创建 `scripts/check_environment.sh`，用于只读检查服务器、GPU、CUDA、COLMAP、Git、数据数量和仓库状态。
- 已提交并推送：`2498076 Add environment check scripts`。
- 创建 `scripts/install_miniconda.sh`，在 `/home/pch/myGS/tools/miniconda3` 安装项目专用 Miniconda。
- 创建 `scripts/setup_3dgs_env.sh`，创建 Conda 环境 `mygs-3dgs-cu128`，安装 Python、PyTorch cu128 和基础依赖。
- 为解决 Conda 非交互 Terms of Service 阻塞，环境创建改为使用 `conda-forge` 并加 `--override-channels`。
- 为解决 PyTorch CDN 访问时优先走不可达 IPv6 的问题，新增 `scripts/pip_ipv4.py`，强制 pip 下载走 IPv4 DNS 解析。
- 新增 `scripts/install_tmp_torch_wheels.py`，作为已下载 PyTorch/CUDA wheel 的恢复安装助手。
- 已提交并推送：`8dc4f40 Add project CUDA environment setup`。
- 创建 `scripts/fetch_3dgs.sh`，用于拉取官方 `graphdeco-inria/gaussian-splatting` 到 `tools/gaussian-splatting`。
- 服务器直连 GitHub HTTPS 超时，GitHub SSH 可用但完整 clone 速度很慢。
- 采用镜像时只把镜像作为传输通道，不把镜像作为可信源：
  - 主仓库和 `diff-gaussian-rasterization` 使用 `gitclone.com` 加速。
  - `fused-ssim` 在 `gitclone.com` 返回 502，改用 GitHub SSH。
  - `simple-knn` 使用 Inria GitLab 官方地址。
  - 最终 checkout HEAD 与官方 GitHub SSH HEAD 校验一致：`54c035f`。
- 因 `SIBR_viewers` 是 Viewer 子模块，训练基线不依赖它，且 Inria 侧下载慢，所以当前只拉取训练相关子模块：
  - `submodules/diff-gaussian-rasterization`
  - `submodules/fused-ssim`
  - `submodules/simple-knn`
- 编译 3DGS CUDA 扩展时，PyTorch 最初检测到系统默认 `nvcc 11.5`，与 PyTorch cu128 不匹配。
- 未切换系统 CUDA；编译命令临时设置 `CUDA_HOME=/usr/local/cuda-12.8` 和 PATH，使扩展使用服务器已有 CUDA 12.8 编译器。
- 已成功构建并安装：
  - `diff_gaussian_rasterization`
  - `simple_knn`
  - `fused_ssim`
- 已提交并推送：`0fe832f Add 3DGS source fetch script`。
- 创建 `scripts/prepare_dataset.sh`，用于为指定场景准备 3DGS/COLMAP 输入图片目录。
- 脚本执行前会校验原始数据目录存在，并校验原始图片数量是否等于期望数量。
- 脚本只读取原始数据目录，不移动、不删除、不改名原始图片。
- 输出目录为：
  - `/home/pch/myGS/datasets/uav_3dgs/processed/visible/images`
  - `/home/pch/myGS/datasets/uav_3dgs/processed/thermal/images`
- 脚本优先创建符号链接；如符号链接不可用，才复制图片到 processed 目录。
- 已运行 visible 数据准备：
  - 输出：`Prepared visible: 339/339`
  - manifest：`/home/pch/myGS/datasets/uav_3dgs/processed/visible/manifest.txt`
- 已运行 thermal 数据准备：
  - 输出：`Prepared thermal: 339/339`
  - manifest：`/home/pch/myGS/datasets/uav_3dgs/processed/thermal/manifest.txt`
- 已验证 manifest：
  - visible：`scene=visible`，`prepared_count=339`
  - thermal：`scene=thermal`，`prepared_count=339`
- 已验证 processed images 实际数量：
  - visible：339 张 `*_V.JPG`
  - thermal：339 张 `*_T.JPG`
- 原始数据目录未移动、未删除、未改名。

#### 待执行

- 创建并运行 `scripts/run_colmap.sh`。
- 对 visible 和 thermal 分别独立运行 COLMAP。
- 验证两个场景均产生 `sparse/0/cameras.bin`、`images.bin` 和 `points3D.bin`。

#### 下一步

- 进入 COLMAP 阶段，优先确认可使用 GPU 版本 COLMAP。
- COLMAP 完成后再写入 `AGENTS.md`。

### 2026-07-10 对话 3：GPU COLMAP 执行与任务日志修正

#### 目标

- 阅读 `AGENTS.md` 和前面对话记录，确认任务后继续执行。
- 检查并使用 GPU 版本 COLMAP。
- 对 visible 和 thermal 分别独立运行 COLMAP。
- 执行完成后再写入 `AGENTS.md`。
- 修正 `AGENTS.md` 任务日志结构，恢复每次任务固定字段格式。

#### 已确认决策

- Task 5 使用项目内 GPU COLMAP，不使用系统 `/usr/bin/colmap`。
- COLMAP 脚本默认启用 GPU SIFT：
  - `SiftExtraction.use_gpu=1`
  - `SiftMatching.use_gpu=1`
- 原始数据目录保持只读，不移动、不删除、不改名。
- `AGENTS.md` 按每次任务固定字段记录：
  - 目标
  - 已确认决策
  - 只读探查结果
  - 已完成
  - 待执行
  - 下一步

#### 只读探查结果

- 系统 `/usr/bin/colmap` 为无 CUDA 版本，不适合作为本任务默认 COLMAP。
- 项目 GPU COLMAP 路径为 `/home/pch/myGS/tools/colmap-3.9.1-cuda/bin/colmap`。
- 项目 GPU COLMAP 验证结果：
  - 版本：`COLMAP 3.9.1`
  - 提交：`e990364`
  - 构建：`with CUDA`
- visible COLMAP 模型统计：
  - 注册图像：`339/339`
  - 点数：`273725`
  - 平均重投影误差：`0.869689px`
- thermal COLMAP 模型统计：
  - 注册图像：`339/339`
  - 点数：`151216`
  - 平均重投影误差：`0.436450px`
- 已验证两个场景均存在：
  - `sparse/0/cameras.bin`
  - `sparse/0/images.bin`
  - `sparse/0/points3D.bin`
- `scripts/run_colmap.sh` 通过 `bash -n`。

#### 已完成

- 创建 `scripts/run_colmap.sh`，用于对指定场景独立运行 COLMAP：
  - `feature_extractor`
  - `exhaustive_matcher`
  - `mapper`
  - `image_undistorter`
- 更新 `configs/scenes.env`：
  - `COLMAP_CUDA_DIR=/home/pch/myGS/tools/colmap-3.9.1-cuda`
  - `COLMAP_BIN=/home/pch/myGS/tools/colmap-3.9.1-cuda/bin/colmap`
- 已确认系统 `/usr/bin/colmap` 为无 CUDA 版本，因此 Task 5 使用项目内 GPU COLMAP。
- 已确认项目 COLMAP：
  - 版本：`COLMAP 3.9.1`
  - 提交：`e990364`
  - 构建：`with CUDA`
- `scripts/run_colmap.sh` 默认启用 GPU SIFT：
  - `SiftExtraction.use_gpu=1`
  - `SiftMatching.use_gpu=1`
- 首次 visible 运行后发现 `image_undistorter` 输出 sparse 模型到 `sparse/` 根目录，而 3DGS 需要 `sparse/0/`。
- 已修正脚本：在运行前后统一检查并规范化 sparse 结构，将 `cameras.bin`、`images.bin`、`points3D.bin` 放入 `sparse/0/`。
- 已运行 visible COLMAP：
  - 输入：`/home/pch/myGS/datasets/uav_3dgs/processed/visible/images`
  - 输出：`/home/pch/myGS/datasets/uav_3dgs/processed/visible/sparse/0`
  - 注册图像：`339/339`
  - 点数：`273725`
  - 平均重投影误差：`0.869689px`
  - 最新成功日志：`/home/pch/myGS/logs/20260709-172538-colmap-visible.log`
  - 完整重建日志：`/home/pch/myGS/logs/20260709-162256-colmap-visible.log`
- 已运行 thermal COLMAP：
  - 输入：`/home/pch/myGS/datasets/uav_3dgs/processed/thermal/images`
  - 输出：`/home/pch/myGS/datasets/uav_3dgs/processed/thermal/sparse/0`
  - 注册图像：`339/339`
  - 点数：`151216`
  - 平均重投影误差：`0.436450px`
  - 日志：`/home/pch/myGS/logs/20260709-172559-colmap-thermal.log`
- 已验证两个场景均存在：
  - `sparse/0/cameras.bin`
  - `sparse/0/images.bin`
  - `sparse/0/points3D.bin`
- 已验证 `scripts/run_colmap.sh` 通过 `bash -n`。
- 原始数据目录未移动、未删除、未改名。
- 已提交并推送：`7dd34eb Add CUDA COLMAP pipeline script`。
- 已修正 `AGENTS.md` 结构：将实施计划 Task 条目收拢为对话日志中的执行摘要，并新增本次对话固定字段记录。

#### 待执行

- 提交并推送本次 `AGENTS.md` 日志结构修正。
- 进入 3DGS 训练脚本阶段。

#### 下一步

- 进入 Task 6：创建并运行 `scripts/run_training.sh`。
- 对 visible 和 thermal 分别运行 3DGS 训练。
- 验证两个场景均产生训练输出和可复现实验日志。

### 2026-07-10 对话 4：Visible 3DGS 监控训练与质量早停设计

#### 目标

- 将本次工作作为新的独立任务记录，不并入既有 Task 6。
- 只训练 visible 场景，不启动 thermal 训练。
- 实时记录训练进度，并从 5000 步开始每 1000 步记录质量和固定视角效果。
- 质量明确下降时停止训练，最大训练到 15000 步。

#### 已确认决策

- 使用单张 GPU，指定 GPU 0；GPU 1 不参与本次训练。
- 使用官方 `--eval` 固定划分：296 张训练图、43 张测试图。
- 评估检查点为 5000、6000、7000、8000、9000、10000、11000、12000、13000、14000、15000。
- 每个检查点记录 PSNR、SSIM、LPIPS 和固定测试视角渲染图。
- 早停判据：相对上一个检查点，PSNR 下降超过 `0.05 dB`，且 SSIM 下降超过 `0.001` 或 LPIPS 上升超过 `0.001`。
- 采用单训练进程内同步评估，保证在检查点精确停止。
- 不删除已生成结果，原始数据继续保持只读。

#### 只读探查结果

- `951d4c3 Restructure task log entries` 已提交并推送到 `origin/main`；对话 3 中“待提交”是提交前的历史状态，不是当前遗漏提交。
- 官方 3DGS 当前 checkout 提交为 `54c035f`。
- 官方 `train.py` 支持 `test_iterations`、`save_iterations`、`checkpoint_iterations` 和 `start_checkpoint`。
- 官方 COLMAP 数据读取器在 `--eval` 下按排序图像名每 8 张取 1 张测试图。
- visible 的 339 张图像会固定划分为 296 张训练图和 43 张测试图。
- 官方指标代码支持 PSNR、SSIM 和 LPIPS；LPIPS VGG 权重需要在正式训练前预检和缓存。
- 服务器两张 RTX 5090 当前探查时均为空闲。
- 服务器会在 SSH 断开时清理整个会话 cgroup；`nohup`、`setsid`、`tmux` 均无法跨断线保留训练进程。
- 服务器没有可用的 user systemd、无交互 sudo 或 atd，因此本次训练使用受控前台 SSH 会话，并通过独立只读连接监控。
- LPIPS VGG 主干权重已通过 IPv4 从 PyTorch 官方地址下载并校验 SHA-256 前缀 `397923af`。
- LPIPS 线性权重来自官方 GitHub SSH 仓库 `richzhang/PerceptualSimilarity`，checkout 提交为 `082bb24f84c091ea94de2867d34c4544f68e0963`。
- 仓库根目录存在一个名称为单个空格的历史未跟踪条目；它不是待提交日志，本次不删除、不修改、不纳入提交。

#### 已完成

- 澄清并确认 visible 单场景、单 GPU、15000 步上限和同步质量早停设计。
- 创建规格文档：`docs/superpowers/specs/2026-07-10-visible-3dgs-monitored-training-design.md`。
- 创建实施计划：`docs/superpowers/plans/2026-07-10-visible-3dgs-monitored-training-implementation.md`。
- 按测试驱动方式实现：
  - `scripts/visible_training_monitor.py`
  - `patches/gaussian-splatting/visible-monitored-training.patch`
  - `scripts/run_visible_training.sh`
  - `scripts/monitor_visible_training.sh`
  - 对应自动化测试
- 自动化回归验证：`15 passed`；Python 编译和 Bash 语法检查通过。
- 已完成成功 smoke run：
  - 运行目录：`/home/pch/myGS/outputs/visible/smoke-20260710-081201`
  - 日志：`/home/pch/myGS/logs/20260710-081201-train-visible.log`
  - 终止状态：`completed`
  - 10 步：PSNR `12.8568142`，SSIM `0.3359472`，LPIPS `0.6690347`
  - 20 步：PSNR `13.2950412`，SSIM `0.3419230`，LPIPS `0.6621497`
  - 最佳迭代：20
  - 10/20 步均产生 checkpoint、point cloud 和 5 组固定视角 render/GT。
- visible 原始图数量复核仍为 `339`，原始数据未移动、删除、改名或修改。
- 已完成 visible 正式训练：
  - 运行编号：`20260710-082237`
  - 输出目录：`/home/pch/myGS/outputs/visible/20260710-082237`
  - 日志：`/home/pch/myGS/logs/20260710-082237-train-visible.log`
  - 使用 GPU：GPU 0，单卡训练
  - 数据划分：296 张训练图、43 张固定测试图
  - 最终状态：`completed`
  - 终止原因：`max_iterations`
  - 最终迭代：15000
  - 最佳迭代：15000
  - 训练与同步评估耗时：`1140.832` 秒
- 正式训练质量记录如下，`quality_dropped` 均为 `0`：

  | 迭代 | PSNR (dB) | SSIM | LPIPS |
  | ---: | ---: | ---: | ---: |
  | 5000 | 26.115900 | 0.864166 | 0.180623 |
  | 6000 | 26.467526 | 0.876354 | 0.164718 |
  | 7000 | 27.264930 | 0.893812 | 0.150487 |
  | 8000 | 27.788346 | 0.902541 | 0.139094 |
  | 9000 | 28.322533 | 0.911493 | 0.130411 |
  | 10000 | 28.419137 | 0.913659 | 0.127566 |
  | 11000 | 28.843647 | 0.919990 | 0.121546 |
  | 12000 | 28.867639 | 0.921922 | 0.118977 |
  | 13000 | 29.119158 | 0.923907 | 0.116298 |
  | 14000 | 29.262401 | 0.926552 | 0.114541 |
  | 15000 | 29.488692 | 0.928610 | 0.111143 |

- 已验证正式运行产生：
  - 11 个 checkpoint
  - 11 组 point cloud
  - 11 个评估目录
  - 每个评估目录包含 5 张固定测试视角 render 和对应 5 张 GT，共 10 个图像文件
- 所有检查点相对前一个检查点均未满足质量下降组合条件，因此未提前停止，并按约定训练至 15000 步。

#### 待执行

- 提交并推送本次 visible 正式训练结果记录。
- thermal 训练不属于本次任务，保持未启动状态。

#### 下一步

- 本次 visible 独立训练任务完成后，由用户决定是否另开任务评估渲染效果或启动 thermal 训练。

### 2026-07-10 对话 5：最佳 PLY 本地同步用于 SuperSplat 验证

#### 目标

- 每次 3DGS 训练结束后，将质量最高迭代对应的 `point_cloud.ply` 同步到本地目录，方便用户在 SuperSplat 中手动验证效果。
- 本地交付目录固定为 `F:\PCH\myGSproj\supersplat_ply\`。
- 本次先同步 visible 正式训练 `20260710-082237` 的最佳 PLY。

#### 已确认决策

- 质量最高迭代以训练输出目录中的 `summary.json` 为准，读取其中的 `best_iteration` 和 `best_metrics`。
- 本地文件名包含场景、运行编号、最佳迭代和核心指标，便于区分多次训练结果。
- `.ply` 文件属于大体积验证交付物，不纳入 Git；`supersplat_ply/` 保持为本地目录。
- 不删除、不移动、不改名远端训练输出和原始数据。

#### 只读探查结果

- visible 正式训练 `summary.json` 显示：
  - `best_iteration=15000`
  - PSNR `29.488691729168558`
  - SSIM `0.9286100656487221`
  - LPIPS `0.11114280185727186`
- 最佳远端 PLY 路径为：
  - `/home/pch/myGS/outputs/visible/20260710-082237/point_cloud/iteration_15000/point_cloud.ply`

#### 已完成

- 创建同步脚本：`scripts/sync_best_ply.py`。
- 创建自动化测试：`tests/test_sync_best_ply.py`。
- 更新 `.gitignore`，明确忽略本地 `supersplat_ply/` 目录。
- 已将本次 visible 最佳 PLY 同步到本地：
  - `F:\PCH\myGSproj\supersplat_ply\visible_20260710-082237_best_iter15000_psnr29.4887_ssim0.9286_lpips0.1111.ply`
- 已验证本地文件：
  - 文件大小：`1004521316` 字节
  - 与远端文件大小一致
  - 文件头为 `ply`

#### 待执行

- 将同步脚本、测试、`.gitignore` 和本次中文任务日志同步到服务器仓库并提交推送。

#### 下一步

- 以后每次训练结束后，运行 `scripts/sync_best_ply.py --run-id <运行编号>`，把最佳 PLY 拉到本地 `supersplat_ply/` 后再进行 SuperSplat 验证。
