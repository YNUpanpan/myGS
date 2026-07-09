#!/usr/bin/env python
"""Install already-downloaded PyTorch CUDA wheels from /tmp/pip-unpack-*.

This is a recovery helper for slow or unstable PyTorch wheel downloads. It
deduplicates wheel files by filename and installs them without hitting indexes.
"""

from pathlib import Path
import subprocess
import sys


PREFIX_ORDER = (
    "cuda_toolkit-",
    "nvidia_",
    "triton-",
    "torch-",
    "torchvision-",
    "torchaudio-",
)


def wheel_priority(path: Path) -> tuple[int, str]:
    name = path.name
    for idx, prefix in enumerate(PREFIX_ORDER):
        if name.startswith(prefix):
            return idx, name
    return len(PREFIX_ORDER), name


def main() -> None:
    wheels_by_name: dict[str, Path] = {}
    for path in Path("/tmp").glob("pip-unpack-*/*.whl"):
        if path.name not in wheels_by_name or path.stat().st_size > wheels_by_name[path.name].stat().st_size:
            wheels_by_name[path.name] = path

    wheels = sorted(wheels_by_name.values(), key=wheel_priority)
    selected = [p for p in wheels if any(p.name.startswith(prefix) for prefix in PREFIX_ORDER)]
    if not selected:
        raise SystemExit("No PyTorch/CUDA wheels found under /tmp/pip-unpack-*")

    print("Installing local wheels:")
    for path in selected:
        print(f"  {path}")

    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-index", *map(str, selected)])


if __name__ == "__main__":
    main()
