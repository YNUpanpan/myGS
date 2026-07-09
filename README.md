# UAV 3DGS Baseline

This project prepares a reproducible baseline for reconstructing UAV visible-light and thermal scenes with 3D Gaussian Splatting.

## Server Layout

- Server: `pch-5090`
- Project root: `/home/pch/myGS`
- Visible raw data: `/home/pch/myGS/datasets/uav_3dgs/raw/visible`
- Thermal raw data: `/home/pch/myGS/datasets/uav_3dgs/raw/thermal`

## Scope

- Use the official `graphdeco-inria/gaussian-splatting` implementation.
- Use a project-local Miniconda installation and a CUDA 12.8 / PyTorch cu128 environment.
- Run COLMAP independently for visible and thermal data.
- Keep raw data read-only.
- Track scripts, configs, docs, and summaries in git.
- Exclude raw images, processed COLMAP data, training outputs, models, and large artifacts from git.

See `AGENTS.md` and `docs/superpowers/specs/2026-07-09-uav-3dgs-baseline-design.md` for the confirmed design and task log.
