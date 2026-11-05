"""扫描 Wall Security Dataset ALEE，生成 Normal/Wall crossing 二分类清单。

数据目录：
    D:\\SceneDetection\\datasets\\Wall Security Dataset ALEE

官方 train/test 已经分好；本脚本只把 train 再按类别分层切出 15% 验证集，
官方 test 完全不参与训练。

用法：
    python adapters/prepare_climb.py
    python adapters/prepare_climb.py --check

输出：
    D:\\SceneDetection\\features\\climb\\labels_climb.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from collections import Counter

import cv2

DEFAULT_ROOT = r"D:\SceneDetection\datasets\Wall Security Dataset ALEE"
DEFAULT_OUT = r"D:\SceneDetection\features\climb\labels_climb.csv"
CLASS_MAP = {
    "Normal Class": ("normal", 0),
    "Wall crossing": ("wall_crossing", 1),
}


def probe(path: str, check_frame: bool) -> dict:
    capture = cv2.VideoCapture(path)
    opened = capture.isOpened()
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    readable = True
    if check_frame and opened:
        ok, _ = capture.read()
        readable = bool(ok)
    capture.release()
    return {
        "opened": opened,
        "readable": readable,
        "fps": round(fps, 3),
        "frames": frames,
        "width": width,
        "height": height,
        "duration_sec": round(frames / fps, 2) if fps else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="准备攀爬动作数据清单")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--limit-per-class", type=int, default=0)
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        print(f"找不到数据目录：{args.root}")
        return 1
    random.seed(args.seed)
    rows = []
    corrupt = []
    for official_split in ("train", "test"):
        for folder_name, (class_name, label) in CLASS_MAP.items():
            folder = os.path.join(args.root, official_split, folder_name)
            if not os.path.isdir(folder):
                print(f"缺少目录：{folder}")
                continue
            files = sorted(
                name
                for name in os.listdir(folder)
                if os.path.splitext(name)[1].lower() in {".mp4", ".avi", ".mov", ".mkv"}
            )
            if args.limit_per_class:
                files = files[: args.limit_per_class]
            if not files:
                continue

            if official_split == "train":
                indices = list(range(len(files)))
                random.shuffle(indices)
                val_count = (
                    min(len(files) - 1, int(math.ceil(len(files) * args.val_ratio)))
                    if len(files) > 1
                    else 0
                )
                val_indices = set(indices[:val_count])
                roles = [
                    ("val" if index in val_indices else "train")
                    for index in range(len(files))
                ]
            else:
                roles = ["test"] * len(files)

            for name, role in zip(files, roles):
                path = os.path.join(folder, name)
                info = probe(path, args.check)
                if not info["opened"] or not info["readable"] or info["frames"] <= 0:
                    corrupt.append(path)
                stem = os.path.splitext(name)[0]
                rows.append(
                    {
                        "path": path,
                        "dataset": "wall_security_alee",
                        "official_split": official_split,
                        "split": role,
                        "class_name": class_name,
                        "label": label,
                        "source": stem,
                        "group": stem,
                        **info,
                        "size_mb": round(os.path.getsize(path) / 1024**2, 2),
                    }
                )

    if not rows:
        print("没有找到视频")
        return 1
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fieldnames = [
        "path",
        "dataset",
        "official_split",
        "split",
        "class_name",
        "label",
        "source",
        "group",
        "opened",
        "readable",
        "fps",
        "frames",
        "width",
        "height",
        "duration_sec",
        "size_mb",
    ]
    with open(args.out, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 64)
    print(f"清单：{len(rows)} 段 -> {args.out}")
    counts = Counter((row["split"], row["class_name"]) for row in rows)
    for split in ("train", "val", "test"):
        print(
            f"  {split:<5} normal={counts.get((split, 'normal'), 0):<4} "
            f"wall_crossing={counts.get((split, 'wall_crossing'), 0)}"
        )
    if corrupt:
        print(f"\n!! 无法读取 {len(corrupt)} 段：")
        for path in corrupt[:20]:
            print("  ", path)
    else:
        print("\n完整性检查：所有视频都可打开")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
