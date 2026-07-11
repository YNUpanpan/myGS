#!/usr/bin/env python3
"""Prepare a processed dataset by linking only images with a target size."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    return parser.parse_args()


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def main() -> int:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    images_dir = processed_dir / "images"
    manifest = processed_dir / "manifest.txt"

    if not raw_dir.is_dir():
        raise SystemExit(f"Missing raw directory: {raw_dir}")

    images_dir.mkdir(parents=True, exist_ok=True)

    selected: list[Path] = []
    rejected: list[tuple[Path, tuple[int, int]]] = []
    for src in sorted(raw_dir.glob(args.pattern)):
        size = image_size(src)
        if size == (args.width, args.height):
            selected.append(src)
        else:
            rejected.append((src, size))

    if len(selected) != args.expected:
        details = ", ".join(f"{path.name}:{size[0]}x{size[1]}" for path, size in rejected)
        raise SystemExit(
            f"Filtered image count mismatch for {args.scene}: "
            f"{len(selected)}/{args.expected}; rejected={details}"
        )

    for src in selected:
        link_or_copy(src, images_dir / src.name)

    prepared_files = sorted(
        path
        for path in images_dir.glob(args.pattern)
        if path.is_file() or path.is_symlink()
    )
    prepared_names = {path.name for path in prepared_files}
    selected_names = {path.name for path in selected}
    extra_names = sorted(prepared_names - selected_names)
    if extra_names:
        raise SystemExit(
            "Prepared directory contains files outside the filtered set; "
            "manual cleanup is required: " + ", ".join(extra_names)
        )
    if len(prepared_files) != args.expected:
        raise SystemExit(
            f"Prepared image count mismatch for {args.scene}: "
            f"{len(prepared_files)}/{args.expected}"
        )

    with manifest.open("w", encoding="utf-8") as handle:
        handle.write(f"scene={args.scene}\n")
        handle.write(f"raw_dir={raw_dir}\n")
        handle.write(f"processed_dir={processed_dir}\n")
        handle.write(f"images_dir={images_dir}\n")
        handle.write(f"pattern={args.pattern}\n")
        handle.write(f"expected_count={args.expected}\n")
        handle.write(f"target_width={args.width}\n")
        handle.write(f"target_height={args.height}\n")
        handle.write(f"prepared_count={len(prepared_files)}\n")
        handle.write("rejected_count={}\n".format(len(rejected)))
        for path, size in rejected:
            handle.write(f"rejected={path.name},{size[0]}x{size[1]}\n")
        handle.write("\n")
        for src in selected:
            handle.write(f"{src.name}\n")

    print(f"Prepared {args.scene}: {len(prepared_files)}/{args.expected}")
    print(f"Rejected mismatched images: {len(rejected)}")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
