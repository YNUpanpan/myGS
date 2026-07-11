# Night 3DGS Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconstruct the nighttime visible-light and thermal UAV scenes with the same reproducible COLMAP and monitored 3DGS pipeline used for the daytime baseline.

**Architecture:** Keep the existing daytime `visible` and `thermal` scenes unchanged. Add independent `night_visible` and `night_thermal` scene names, each with its own raw, processed, output, log, and SuperSplat PLY artifacts.

**Tech Stack:** Bash, Python 3, pytest, CUDA COLMAP 3.9.1, official 3D Gaussian Splatting checkout `54c035f`, PyTorch cu128 Conda environment `mygs-3dgs-cu128`.

## Global Constraints

- Do not delete, move, rename, or modify raw image data.
- Do not use bulk deletion commands.
- Do not overwrite existing daytime processed data or training outputs.
- Night local source data is under `F:\PCH\myGSproj\night`.
- Night visible images match `*_V.JPG`; night thermal images match `*_T.JPG`.
- Expected night counts are 317 visible images and 317 thermal images.
- Use independent server raw directories: `/home/pch/myGS/datasets/uav_3dgs/raw/night_visible` and `/home/pch/myGS/datasets/uav_3dgs/raw/night_thermal`.
- Use independent server output directories: `/home/pch/myGS/outputs/night_visible` and `/home/pch/myGS/outputs/night_thermal`.

---

### Task 1: Add Night Scene Support

**Files:**
- Modify: `configs/scenes.env`
- Modify: `scripts/common.sh`
- Modify: `tests/test_visible_training_scripts.py`

**Interfaces:**
- Consumes: existing `require_scene`, `scene_pattern`, `expected_count`, `scene_raw_dir`, `scene_processed_dir`, `scene_output_root`.
- Produces: support for `night_visible` and `night_thermal` across existing dataset, COLMAP, training, and sync scripts.

- [ ] Write failing tests that require `night_visible` and `night_thermal` scene names, expected counts, and raw/output paths.
- [ ] Run the targeted pytest file and verify the new tests fail before implementation.
- [ ] Add `NIGHT_VISIBLE_SCENE`, `NIGHT_THERMAL_SCENE`, patterns, and expected counts to `configs/scenes.env`.
- [ ] Update `scripts/common.sh` scene helpers to accept all four scenes.
- [ ] Run the targeted pytest file and Bash syntax checks.

### Task 2: Add Generic Night Training Entry Points

**Files:**
- Create: `scripts/run_night_visible_training.sh`
- Create: `scripts/run_night_thermal_training.sh`
- Create: `scripts/monitor_night_visible_training.sh`
- Create: `scripts/monitor_night_thermal_training.sh`
- Modify: `tests/test_visible_training_scripts.py`

**Interfaces:**
- Consumes: `scene_processed_dir night_visible`, `scene_processed_dir night_thermal`, existing monitor patch and monitor module.
- Produces: standard 15000-step monitored training commands for both night scenes.

- [ ] Write failing tests for the four new scripts.
- [ ] Run the targeted pytest file and verify failures are for missing scripts.
- [ ] Create the scripts by following the existing visible and thermal patterns, replacing only the scene names and labels.
- [ ] Run pytest and `bash -n` checks.

### Task 3: Upload and Prepare Night Data

**Files:**
- Remote raw directories only; no local code edits.

**Interfaces:**
- Consumes: local `F:\PCH\myGSproj\night`.
- Produces: remote raw scene directories with 317 visible and 317 thermal images.

- [ ] Create remote raw directories without deleting or modifying existing data.
- [ ] Upload `*_V.JPG` files to `raw/night_visible`.
- [ ] Upload `*_T.JPG` files to `raw/night_thermal`.
- [ ] Verify remote counts are 317 and 317.
- [ ] Run `scripts/prepare_dataset.sh night_visible`.
- [ ] Run `scripts/prepare_dataset.sh night_thermal`.

### Task 4: Run COLMAP

**Files:**
- Remote processed directories only; no local code edits.

**Interfaces:**
- Consumes: prepared night image directories.
- Produces: `sparse/0/cameras.bin`, `images.bin`, and `points3D.bin` for both night scenes.

- [ ] Run `scripts/run_colmap.sh night_visible`.
- [ ] Verify `night_visible/sparse/0` files and registration summary.
- [ ] Run `scripts/run_colmap.sh night_thermal`.
- [ ] Verify `night_thermal/sparse/0` files and registration summary.

### Task 5: Train and Sync Best PLY

**Files:**
- Remote output directories.
- Local `supersplat_ply/`.
- Modify: `AGENTS.md`.

**Interfaces:**
- Consumes: night COLMAP outputs.
- Produces: monitored 3DGS outputs, best PLY files for SuperSplat, and a task log entry.

- [ ] Run night visible smoke training.
- [ ] Run night visible formal 15000-step training.
- [ ] Sync best night visible PLY to local `supersplat_ply/`.
- [ ] Run night thermal smoke training.
- [ ] Run night thermal formal 15000-step training.
- [ ] Sync best night thermal PLY to local `supersplat_ply/`.
- [ ] Update `AGENTS.md` with decisions, probes, completed work, metrics, and remaining manual SuperSplat checks.
- [ ] Run final tests and syntax checks.
