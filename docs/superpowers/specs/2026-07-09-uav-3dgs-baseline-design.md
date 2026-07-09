# UAV Visible/Thermal 3DGS Baseline Design

Date: 2026-07-09

## Purpose

Build a reproducible baseline project on server `pch-5090` for reconstructing two UAV circular-flight scenes with 3D Gaussian Splatting:

- visible-light scene: 339 images
- thermal scene: 339 images

The baseline should run end to end: environment setup, data preparation, COLMAP, 3DGS training, logging, and GitHub upload of engineering files.

## Confirmed Decisions

- Work directly on the Ubuntu server through SSH entry `pch-5090`.
- Use `/home/pch/myGS` as the project root.
- Use project-local Conda at `/home/pch/myGS/tools/miniconda3`.
- Use CUDA 12.8 through the project Conda/PyTorch environment instead of changing system CUDA.
- Use official `graphdeco-inria/gaussian-splatting` as the 3DGS baseline.
- Run COLMAP independently for visible and thermal data.
- Produce a complete reproducible engineering baseline and run both trainings.
- Upload only engineering files to `git@github.com:YNUpanpan/myGS.git`.
- Do not upload raw images, processed COLMAP data, training outputs, models, or other large artifacts.

## Current Server Findings

- `/home/pch/myGS` initially contained `datasets/` only and was not a git repository.
- No `AGENTS.md` existed initially.
- GPU: 2 x NVIDIA GeForce RTX 5090, each about 32607 MiB VRAM.
- Driver: 570.144.
- System `nvcc`: CUDA 11.5.
- System tools available: `/usr/bin/colmap`, `/usr/bin/git`.
- Raw data:
  - `/home/pch/myGS/datasets/uav_3dgs/raw/visible`: 339 `*_V.JPG` files
  - `/home/pch/myGS/datasets/uav_3dgs/raw/thermal`: 339 `*_T.JPG` files
  - `.MRK` files were not found on the server under the dataset path.

## Directory Design

```text
/home/pch/myGS/
  AGENTS.md
  README.md
  .gitignore
  scripts/
  configs/
  docs/
    superpowers/
      specs/
  logs/
  tools/
    miniconda3/
    gaussian-splatting/
  datasets/
    uav_3dgs/
      raw/
        visible/
        thermal/
      processed/
        visible/
        thermal/
  outputs/
    visible/
    thermal/
```

Raw data stays read-only. Generated files go under `processed/`, `outputs/`, and `logs/`.

## Data Flow

Visible and thermal data are processed independently:

```text
raw/visible -> processed/visible -> COLMAP visible -> train visible -> outputs/visible
raw/thermal -> processed/thermal -> COLMAP thermal -> train thermal -> outputs/thermal
```

Each processed scene follows the 3DGS/COLMAP layout:

```text
datasets/uav_3dgs/processed/<scene>/
  images/
  distorted/
  sparse/
  stereo/
  colmap.log
  manifest.txt
```

Data preparation should prefer symbolic links from raw images into `processed/<scene>/images` to avoid copying. If symlinks are unavailable, copying can be used as a fallback. No raw image should be modified.

## Script Design

```text
scripts/check_environment.sh
scripts/install_miniconda.sh
scripts/setup_3dgs_env.sh
scripts/fetch_3dgs.sh
scripts/prepare_dataset.sh
scripts/run_colmap.sh
scripts/run_train.sh
scripts/summarize_runs.sh
```

Expected use:

```bash
bash scripts/check_environment.sh
bash scripts/install_miniconda.sh
bash scripts/setup_3dgs_env.sh
bash scripts/fetch_3dgs.sh
bash scripts/prepare_dataset.sh visible
bash scripts/run_colmap.sh visible
bash scripts/run_train.sh visible
bash scripts/prepare_dataset.sh thermal
bash scripts/run_colmap.sh thermal
bash scripts/run_train.sh thermal
bash scripts/summarize_runs.sh
```

Scripts should be single-purpose and safe to inspect. They should fail early on missing prerequisites and write logs to `logs/`.

## Training Design

- Use the official 3DGS baseline parameters first.
- Run visible and thermal sequentially by default to simplify debugging.
- Allow explicit GPU selection through `CUDA_VISIBLE_DEVICES`.
- Write outputs to `outputs/<scene>/<run_id>/`.
- Record log files as `logs/<timestamp>-<step>-<scene>.log`.

## Validation Criteria

Environment:

- Project Python can import PyTorch.
- `torch.cuda.is_available()` is true.
- PyTorch reports a CUDA 12.8/cu128 runtime.
- RTX 5090 GPUs are visible.

Data:

- `raw/visible` has 339 visible images.
- `raw/thermal` has 339 thermal images.
- `processed/<scene>/images` count matches raw count.

COLMAP:

- `processed/<scene>/sparse/0` exists.
- `cameras.bin`, `images.bin`, and `points3D.bin` exist.
- Registered image count is recorded.

Training:

- `outputs/<scene>/<run_id>` exists.
- Training logs exist.
- A 3DGS point cloud such as `point_cloud/iteration_*/point_cloud.ply` is produced.

GitHub:

- Git repository is initialized in `/home/pch/myGS`.
- Remote is `git@github.com:YNUpanpan/myGS.git`.
- Git tracks only engineering files.
- Raw data, processed data, logs, tools, and outputs are ignored.

## Git Ignore Policy

The repository should ignore:

```gitignore
datasets/**/raw/
datasets/**/processed/
outputs/
logs/*.log
tools/miniconda3/
tools/gaussian-splatting/
*.pth
*.pt
*.ply
*.bin
```

Small documentation files, scripts, configs, and summaries remain trackable.

## AGENTS.md Policy

`AGENTS.md` is the authoritative task log and operation constraint file. Each conversation should append a clear task entry with:

- date and conversation number
- goal
- confirmed decisions
- commands or actions summarized
- results
- failures or risks
- next steps

The file also records the deletion constraint: no batch deletion commands or recursive directory deletion. If cleanup is required, stop and ask the user to handle bulk deletion manually, or delete only one explicit file at a time after confirmation.

## Risks

- Thermal COLMAP may be weaker than visible COLMAP because thermal imagery can have lower texture.
- The system CUDA is 11.5, so any build process that accidentally uses system `nvcc` may conflict with the cu128 target.
- Official 3DGS dependencies may need careful version pinning for RTX 5090 and CUDA 12.8.
- GitHub SSH push requires the server to have the correct SSH key access.

## Next Step

After user review and approval of this design, create an implementation plan, then implement the scripts and repository setup in controlled steps.
