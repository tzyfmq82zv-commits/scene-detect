"""挖掘攀爬模型的非围墙场景难负样本。

当前攀爬模型在混合视频里会把走路、奔跑、摔倒等场景误判成 Wall crossing。
本脚本从非围墙数据源中采样 2 秒窗口，挑出当前模型高分的窗口，缓存为
Normal 训练样本，并生成可继续训练的新 manifest。

用法：
    python adapters/mine_climb_hard_negatives.py
    python adapters/mine_climb_hard_negatives.py --threshold 0.55 --max-per-source 60

输出：
    D:\\SceneDetection\\features\\climb\\hardneg_frames\\...
    D:\\SceneDetection\\features\\climb\\labels_climb_hardneg.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from feature_extract import apply_crop  # noqa: E402
from train_climb import CLASS_NAMES, ClimbClassifier  # noqa: E402

DATASETS_ROOT = r"D:\SceneDetection\datasets"
FEATURES_ROOT = r"D:\SceneDetection\features\climb"
DEFAULT_MANIFEST = os.path.join(FEATURES_ROOT, "labels_climb.csv")
DEFAULT_CACHE = os.path.join(FEATURES_ROOT, "frames_k16")
DEFAULT_MODEL = os.path.join(CODE_ROOT, "runs_climb", "best.pt")
DEFAULT_OUT = os.path.join(FEATURES_ROOT, "labels_climb_hardneg.csv")

SOURCES = [
    (os.path.join(DATASETS_ROOT, "hmdb51_org", "walk"), "hmdb_walk", "none"),
    (os.path.join(DATASETS_ROOT, "hmdb51_org", "run"), "hmdb_run", "none"),
    (os.path.join(DATASETS_ROOT, "hmdb51_org", "fall_floor"), "hmdb_fall", "none"),
    (
        os.path.join(DATASETS_ROOT, "fight", "bus_violence", "NoViolence"),
        "bus_normal",
        "none",
    ),
    (os.path.join(DATASETS_ROOT, "fall", "urfall"), "ur_adl", "right"),
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def sample_window(
    capture, start: int, window: int, total: int, frames: int, size: int, crop: str
) -> np.ndarray:
    picks = np.linspace(start, min(start + window - 1, total - 1), frames).astype(int)
    sampled = []
    for index in picks:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            break
        if crop != "none":
            frame = apply_crop(frame, crop)
        frame = cv2.resize(frame, (size, size))
        sampled.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not sampled:
        return np.zeros((frames, size, size, 3), dtype=np.uint8)
    while len(sampled) < frames:
        sampled.append(sampled[-1])
    return np.stack(sampled[:frames]).astype(np.uint8)


def to_batch(frames: np.ndarray) -> torch.Tensor:
    array = frames.astype(np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(array.transpose(0, 3, 1, 2)).unsqueeze(0).float()


def main() -> int:
    parser = argparse.ArgumentParser(description="挖掘攀爬难负样本")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--max-per-source", type=int, default=60)
    parser.add_argument("--max-files-per-source", type=int, default=80)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=1.5)
    parser.add_argument("--copies", type=int, default=2)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"找不到模型：{args.model}")
        return 1
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = int(checkpoint.get("frames", 16))
    size = int(checkpoint.get("image_size", 224))
    model = ClimbClassifier(dropout=checkpoint.get("dropout", 0.35)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with open(args.manifest, newline="", encoding="utf-8-sig") as handle:
        original_rows = list(csv.DictReader(handle))
    fieldnames = list(original_rows[0].keys())
    for extra in ("hard_negative", "hard_negative_prob"):
        if extra not in fieldnames:
            fieldnames.append(extra)

    hard_rows = []
    cache_dir = os.path.join(args.cache, "train", "normal")
    os.makedirs(cache_dir, exist_ok=True)
    started = time.time()
    for folder, source_name, crop in SOURCES:
        if not os.path.isdir(folder):
            print(f"跳过不存在的数据源：{folder}")
            continue
        videos = sorted(
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if os.path.splitext(name)[1].lower() in {".mp4", ".avi", ".mov", ".mkv"}
        )
        videos = videos[: args.max_files_per_source]
        selected = 0
        print(f"[{source_name}] {len(videos)} 个视频")
        for video in videos:
            if selected >= args.max_per_source:
                break
            capture = cv2.VideoCapture(video)
            if not capture.isOpened():
                continue
            fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total < int(frames * 1.1):
                capture.release()
                continue
            window = max(1, int(round(args.window_seconds * fps)))
            stride = max(1, int(round(args.stride_seconds * fps)))
            stem = os.path.splitext(os.path.basename(video))[0]
            for window_index, start in enumerate(range(0, max(1, total - window + 1), stride)):
                if selected >= args.max_per_source:
                    break
                sampled = sample_window(capture, start, window, total, frames, size, crop)
                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                        logits = model(to_batch(sampled).to(device))
                    probability = float(torch.softmax(logits, dim=1)[0, 1].cpu())
                if probability < args.threshold:
                    continue
                unique = f"hn_{source_name}_{stem}_w{window_index:03d}"
                target = os.path.join(cache_dir, f"{unique}.npy")
                np.save(target, sampled)
                hard_rows.append(
                    {
                        "path": f"{unique}.mp4",
                        "dataset": "hard_negative",
                        "official_split": "train",
                        "split": "train",
                        "class_name": "normal",
                        "label": 0,
                        "source": stem,
                        "group": f"hn_{source_name}_{stem}",
                        "opened": True,
                        "readable": True,
                        "fps": round(fps, 3),
                        "frames": frames,
                        "width": size,
                        "height": size,
                        "duration_sec": round(window / fps, 2),
                        "size_mb": 0,
                        "hard_negative": 1,
                        "hard_negative_prob": round(probability, 4),
                    }
                )
                selected += 1
            capture.release()
        print(f"  选中 {selected} 个高分窗口")

    if not hard_rows:
        print("没有挖到难负样本，降低 --threshold 或增加数据源")
        return 1

    output_rows = []
    for row in original_rows:
        row = dict(row)
        row.setdefault("hard_negative", "0")
        row.setdefault("hard_negative_prob", "")
        output_rows.append(row)
    for row in hard_rows:
        output_rows.append(row)
        for copy_index in range(1, max(1, args.copies)):
            duplicate = dict(row)
            duplicate["source"] = f"{row['source']}_copy{copy_index}"
            output_rows.append(duplicate)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    print("\n" + "=" * 64)
    print(f"难负样本 {len(hard_rows)} 个，输出总行数 {len(output_rows)}")
    print(f"用时 {(time.time()-started)/60:.1f} 分钟")
    print(f"清单：{args.out}")
    print(f"缓存：{cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
