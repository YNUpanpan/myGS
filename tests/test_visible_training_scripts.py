import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "scripts" / "run_visible_training.sh"
HIGHRES_RUNNER = ROOT / "scripts" / "run_visible_highres_training.sh"
MONITOR = ROOT / "scripts" / "monitor_visible_training.sh"
THERMAL_RUNNER = ROOT / "scripts" / "run_thermal_training.sh"
THERMAL_MONITOR = ROOT / "scripts" / "monitor_thermal_training.sh"
NIGHT_VISIBLE_RUNNER = ROOT / "scripts" / "run_night_visible_training.sh"
NIGHT_THERMAL_RUNNER = ROOT / "scripts" / "run_night_thermal_training.sh"
NIGHT_VISIBLE_MONITOR = ROOT / "scripts" / "monitor_night_visible_training.sh"
NIGHT_THERMAL_MONITOR = ROOT / "scripts" / "monitor_night_thermal_training.sh"
GITIGNORE = ROOT / ".gitignore"
SETUP_ENV = ROOT / "scripts" / "setup_3dgs_env.sh"
SCENES_ENV = ROOT / "configs" / "scenes.env"
COMMON = ROOT / "scripts" / "common.sh"
COLMAP_RUNNER = ROOT / "scripts" / "run_colmap.sh"
DIMENSION_FILTER_PREP = ROOT / "scripts" / "prepare_dimension_filtered_dataset.sh"
DIMENSION_FILTER_PREP_PY = ROOT / "scripts" / "prepare_dimension_filtered_dataset.py"


def test_runner_contract_and_syntax():
    assert RUNNER.exists(), f"missing runner: {RUNNER}"
    text = RUNNER.read_text(encoding="utf-8")
    for required in (
        "CUDA_VISIBLE_DEVICES=0",
        "MYGS_MONITOR_DIR",
        "MYGS_EVAL_ITERATIONS",
        "visible-monitored-training.patch",
        "--eval",
        "--disable_viewer",
        "--checkpoint_iterations",
        "foreground-ssh",
        "train.session",
        "finalize-exit",
        'mode="${1:-}"',
    ):
        assert required in text
    assert "printf -v worker_command" not in text
    assert "tmux new-session" not in text
    assert "nohup" not in text
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)


def test_highres_runner_contract_and_syntax():
    assert HIGHRES_RUNNER.exists(), f"missing runner: {HIGHRES_RUNNER}"
    text = HIGHRES_RUNNER.read_text(encoding="utf-8")
    for required in (
        "CUDA_VISIBLE_DEVICES=0",
        "MYGS_MONITOR_DIR",
        "MYGS_EVAL_ITERATIONS",
        "--resolution",
        '"1"',
        "--iterations",
        '"30000"',
        "MYGS_VISIBLE_RESOLUTION",
        "DEFAULT_RESOLUTION",
        "resolution-${FULL_RESOLUTION}",
        "5000,10000,15000,20000,25000,30000",
        "highres-",
        "smoke-highres-",
        "--disable_viewer",
        "--checkpoint_iterations",
        "foreground-ssh",
        "finalize-exit",
    ):
        assert required in text
    assert "[ INFO ] Encountered quite large input images" not in text
    assert "tmux new-session" not in text
    assert "nohup" not in text
    subprocess.run(["bash", "-n", str(HIGHRES_RUNNER)], check=True)


def test_thermal_runner_contract_and_syntax():
    assert THERMAL_RUNNER.exists(), f"missing runner: {THERMAL_RUNNER}"
    text = THERMAL_RUNNER.read_text(encoding="utf-8")
    for required in (
        "CUDA_VISIBLE_DEVICES=0",
        "MYGS_MONITOR_DIR",
        "MYGS_EVAL_ITERATIONS",
        "visible-monitored-training.patch",
        "scene_processed_dir thermal",
        "scene_output_root thermal",
        "scene_pattern thermal",
        "--eval",
        "--disable_viewer",
        "--checkpoint_iterations",
        "5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000",
        "foreground-ssh",
        "train.session",
        "finalize-exit",
        'mode="${1:-}"',
    ):
        assert required in text
    assert "scene_processed_dir visible" not in text
    assert "scene_output_root visible" not in text
    assert "tmux new-session" not in text
    assert "nohup" not in text
    subprocess.run(["bash", "-n", str(THERMAL_RUNNER)], check=True)


def test_night_scene_configuration_contract():
    config = SCENES_ENV.read_text(encoding="utf-8")
    common = COMMON.read_text(encoding="utf-8")
    for required in (
        'NIGHT_VISIBLE_SCENE="night_visible"',
        'NIGHT_THERMAL_SCENE="night_thermal"',
        'NIGHT_VISIBLE_EXHAUSTIVE_SCENE="night_visible_exhaustive"',
        'NIGHT_VISIBLE_EXHAUSTIVE_MAIN_SCENE="night_visible_exhaustive_main"',
        'NIGHT_VISIBLE_MULTICAMERA_EXHAUSTIVE_SCENE="night_visible_multicamera_exhaustive"',
        'NIGHT_VISIBLE_4032_EXHAUSTIVE_SCENE="night_visible_4032_exhaustive"',
        'NIGHT_THERMAL_EXHAUSTIVE_SCENE="night_thermal_exhaustive"',
        'NIGHT_VISIBLE_PATTERN="*_V.JPG"',
        'NIGHT_THERMAL_PATTERN="*_T.JPG"',
        'NIGHT_VISIBLE_EXHAUSTIVE_PATTERN="*_V.JPG"',
        'NIGHT_VISIBLE_EXHAUSTIVE_MAIN_PATTERN="*_V.JPG"',
        'NIGHT_VISIBLE_MULTICAMERA_EXHAUSTIVE_PATTERN="*_V.JPG"',
        'NIGHT_VISIBLE_4032_EXHAUSTIVE_PATTERN="*_V.JPG"',
        'NIGHT_THERMAL_EXHAUSTIVE_PATTERN="*_T.JPG"',
        'NIGHT_VISIBLE_EXPECTED_COUNT="317"',
        'NIGHT_THERMAL_EXPECTED_COUNT="317"',
        'NIGHT_VISIBLE_EXHAUSTIVE_EXPECTED_COUNT="317"',
        'NIGHT_VISIBLE_EXHAUSTIVE_MAIN_EXPECTED_COUNT="317"',
        'NIGHT_VISIBLE_MULTICAMERA_EXHAUSTIVE_EXPECTED_COUNT="317"',
        'NIGHT_VISIBLE_4032_EXHAUSTIVE_EXPECTED_COUNT="312"',
        'NIGHT_THERMAL_EXHAUSTIVE_EXPECTED_COUNT="317"',
    ):
        assert required in config
    for scene in (
        "NIGHT_VISIBLE_SCENE",
        "NIGHT_THERMAL_SCENE",
        "NIGHT_VISIBLE_EXHAUSTIVE_SCENE",
        "NIGHT_VISIBLE_EXHAUSTIVE_MAIN_SCENE",
        "NIGHT_VISIBLE_MULTICAMERA_EXHAUSTIVE_SCENE",
        "NIGHT_VISIBLE_4032_EXHAUSTIVE_SCENE",
        "NIGHT_THERMAL_EXHAUSTIVE_SCENE",
    ):
        assert f'"${{{scene}}}"' in common
    assert "Invalid scene '${scene}'. Expected one of:" in common
    assert 'echo "${NIGHT_VISIBLE_PATTERN}"' in common
    assert 'echo "${NIGHT_THERMAL_PATTERN}"' in common
    assert 'echo "${NIGHT_VISIBLE_EXHAUSTIVE_PATTERN}"' in common
    assert 'echo "${NIGHT_VISIBLE_EXHAUSTIVE_MAIN_PATTERN}"' in common
    assert 'echo "${NIGHT_VISIBLE_MULTICAMERA_EXHAUSTIVE_PATTERN}"' in common
    assert 'echo "${NIGHT_VISIBLE_4032_EXHAUSTIVE_PATTERN}"' in common
    assert 'echo "${NIGHT_THERMAL_EXHAUSTIVE_PATTERN}"' in common
    assert 'echo "${NIGHT_VISIBLE_EXPECTED_COUNT}"' in common
    assert 'echo "${NIGHT_THERMAL_EXPECTED_COUNT}"' in common
    assert 'echo "${NIGHT_VISIBLE_EXHAUSTIVE_EXPECTED_COUNT}"' in common
    assert 'echo "${NIGHT_VISIBLE_EXHAUSTIVE_MAIN_EXPECTED_COUNT}"' in common
    assert 'echo "${NIGHT_VISIBLE_MULTICAMERA_EXHAUSTIVE_EXPECTED_COUNT}"' in common
    assert 'echo "${NIGHT_VISIBLE_4032_EXHAUSTIVE_EXPECTED_COUNT}"' in common
    assert 'echo "${NIGHT_THERMAL_EXHAUSTIVE_EXPECTED_COUNT}"' in common
    assert '"${NIGHT_VISIBLE_EXHAUSTIVE_SCENE}"|"${NIGHT_VISIBLE_EXHAUSTIVE_MAIN_SCENE}"|"${NIGHT_VISIBLE_MULTICAMERA_EXHAUSTIVE_SCENE}"|"${NIGHT_VISIBLE_4032_EXHAUSTIVE_SCENE}") echo "${RAW_ROOT}/${NIGHT_VISIBLE_SCENE}"' in common
    assert '"${NIGHT_THERMAL_EXHAUSTIVE_SCENE}") echo "${RAW_ROOT}/${NIGHT_THERMAL_SCENE}"' in common


def test_night_training_runner_contract_and_syntax():
    cases = (
        (NIGHT_VISIBLE_RUNNER, "night_visible", "Night visible"),
        (NIGHT_THERMAL_RUNNER, "night_thermal", "Night thermal"),
    )
    for path, scene, label in cases:
        assert path.exists(), f"missing runner: {path}"
        text = path.read_text(encoding="utf-8")
        for required in (
            "CUDA_VISIBLE_DEVICES=0",
            "MYGS_MONITOR_DIR",
            "MYGS_EVAL_ITERATIONS",
            "visible-monitored-training.patch",
            f'DEFAULT_SCENE="{scene}"',
            "TRAINING_SCENE=",
            "scene_processed_dir",
            "scene_output_root",
            "scene_pattern",
            "--eval",
            "--disable_viewer",
            "--checkpoint_iterations",
            "5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000",
            "foreground-ssh",
            "train.session",
            "finalize-exit",
            label,
        ):
            assert required in text
        assert "tmux new-session" not in text
        assert "nohup" not in text
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_monitor_contract_and_syntax():
    assert MONITOR.exists(), f"missing monitor: {MONITOR}"
    text = MONITOR.read_text(encoding="utf-8")
    for required in ("status.json", "metrics.csv", "train.pid", "nvidia-smi", "tail -n 30"):
        assert required in text
    assert "kill -0" in text
    assert "kill -9" not in text
    subprocess.run(["bash", "-n", str(MONITOR)], check=True)


def test_thermal_monitor_contract_and_syntax():
    assert THERMAL_MONITOR.exists(), f"missing monitor: {THERMAL_MONITOR}"
    text = THERMAL_MONITOR.read_text(encoding="utf-8")
    for required in ("outputs/thermal", "status.json", "metrics.csv", "train.pid", "nvidia-smi", "tail -n 30"):
        assert required in text
    assert "outputs/visible" not in text
    assert "kill -0" in text
    assert "kill -9" not in text
    subprocess.run(["bash", "-n", str(THERMAL_MONITOR)], check=True)


def test_colmap_runner_supports_multicamera_and_largest_model_selection():
    text = COLMAP_RUNNER.read_text(encoding="utf-8")
    for required in (
        "COLMAP_IMAGE_READER_SINGLE_CAMERA",
        '--ImageReader.single_camera "${COLMAP_IMAGE_READER_SINGLE_CAMERA}"',
        "select_largest_sparse_model",
        "registered_image_count",
        "selected_sparse_model=",
        "selected_registered_images=",
    ):
        assert required in text
    assert '--input_path "${distorted_dir}/sparse/0"' not in text
    subprocess.run(["bash", "-n", str(COLMAP_RUNNER)], check=True)


def test_dimension_filtered_prepare_contract_and_syntax():
    shell_text = DIMENSION_FILTER_PREP.read_text(encoding="utf-8")
    py_text = DIMENSION_FILTER_PREP_PY.read_text(encoding="utf-8")
    assert "prepare_dimension_filtered_dataset.py" in shell_text
    assert "--target_width" not in shell_text
    for required in (
        "--scene",
        "--raw-dir",
        "--processed-dir",
        "--pattern",
        "--expected",
        "--width",
        "--height",
    ):
        assert required in shell_text
    for required in (
        "from PIL import Image",
        "Filtered image count mismatch",
        "Rejected mismatched images",
        "Prepared directory contains files outside the filtered set",
        "target_width",
        "target_height",
    ):
        assert required in py_text
    assert "shutil.rmtree" not in py_text
    assert "unlink()" not in py_text
    subprocess.run(["bash", "-n", str(DIMENSION_FILTER_PREP)], check=True)


def test_night_monitor_contract_and_syntax():
    cases = (
        (NIGHT_VISIBLE_MONITOR, "outputs/night_visible"),
        (NIGHT_THERMAL_MONITOR, "outputs/night_thermal"),
    )
    for path, output_root in cases:
        assert path.exists(), f"missing monitor: {path}"
        text = path.read_text(encoding="utf-8")
        for required in (
            output_root,
            "status.json",
            "metrics.csv",
            "train.pid",
            "nvidia-smi",
            "tail -n 30",
        ):
            assert required in text
        assert "kill -0" in text
        assert "kill -9" not in text
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_tools_directory_is_ignored_as_a_whole():
    patterns = GITIGNORE.read_text(encoding="utf-8").splitlines()
    assert "tools/" in patterns


def test_project_environment_installs_pytest():
    assert "pytest" in SETUP_ENV.read_text(encoding="utf-8")
