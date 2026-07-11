#!/usr/bin/env python3
"""Summarize image dimensions for a directory/pattern."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--pattern", default="*.JPG")
    parser.add_argument("--expected-width", type=int)
    parser.add_argument("--expected-height", type=int)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    sizes: list[tuple[str, tuple[int, int]]] = []
    for image_path in sorted(args.image_dir.glob(args.pattern)):
        with Image.open(image_path) as image:
            sizes.append((image_path.name, image.size))

    counts = Counter(size for _, size in sizes)
    print(f"count={len(sizes)}")
    for size, count in sorted(counts.items()):
        print(f"{size[0]}x{size[1]} {count}")

    if args.expected_width and args.expected_height:
        expected = (args.expected_width, args.expected_height)
        mismatches = [(name, size) for name, size in sizes if size != expected]
        print(f"mismatch_count={len(mismatches)}")
        for name, size in mismatches[: args.limit]:
            print(f"mismatch {name} {size[0]}x{size[1]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
