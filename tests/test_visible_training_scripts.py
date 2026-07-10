import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "scripts" / "run_visible_training.sh"
MONITOR = ROOT / "scripts" / "monitor_visible_training.sh"
GITIGNORE = ROOT / ".gitignore"
SETUP_ENV = ROOT / "scripts" / "setup_3dgs_env.sh"


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


def test_monitor_contract_and_syntax():
    assert MONITOR.exists(), f"missing monitor: {MONITOR}"
    text = MONITOR.read_text(encoding="utf-8")
    for required in ("status.json", "metrics.csv", "train.pid", "nvidia-smi", "tail -n 30"):
        assert required in text
    assert "kill -0" in text
    assert "kill -9" not in text
    subprocess.run(["bash", "-n", str(MONITOR)], check=True)


def test_tools_directory_is_ignored_as_a_whole():
    patterns = GITIGNORE.read_text(encoding="utf-8").splitlines()
    assert "tools/" in patterns


def test_project_environment_installs_pytest():
    assert "pytest" in SETUP_ENV.read_text(encoding="utf-8")
