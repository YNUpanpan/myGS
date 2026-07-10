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
