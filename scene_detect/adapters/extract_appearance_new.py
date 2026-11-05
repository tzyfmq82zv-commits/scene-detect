"""给新数据集的每个骨架片段补 RGB 外观特征（双流模型的第二条支路）。

旧数据已经有 appearance.npz；新数据只有骨架，必须补这一步，否则训练时
外观支路的输入会全是 0，打架和摔倒都会退化。

做法和 extract_appearance.py 一致：
    按片段里存的 bbox 裁人 -> 8 帧 -> ResNet18 -> 8x512 特征
区别是按 video_uid 找源视频，并兼容 new_datasets 的多层目录。

用法：
    python adapters/extract_appearance_new.py --limit-videos 3   # 试跑
    python adapters/extract_appearance_new.py
    python adapters/extract_appearance_new.py --overwrite

输出：每个片段目录下一份 appearance.npz（键是片段文件名去扩展名）。
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time

import cv2
import numpy as np

FEATURES_ROOT = r"D:\SceneDetection\features"
DEFAULT_LABELS = os.path.join(FEATURES_ROOT, "labels_six.csv")
DEFAULT_VIDEOS = os.path.join(FEATURES_ROOT, "new_datasets", "videos.csv")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CROP_PADDING = 0.18


def read_csv(path: str) -> list[dict]:
    import csv

    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def absolute_clip(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(FEATURES_ROOT, path)


def crop_with_padding(frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    box_width = max(x2 - x1, 1.0)
    box_height = max(y2 - y1, 1.0)
    pad_x = box_width * CROP_PADDING
    pad_y = box_height * CROP_PADDING
    x1 = int(max(0, np.floor(x1 - pad_x)))
    y1 = int(max(0, np.floor(y1 - pad_y)))
    x2 = int(min(width, np.ceil(x2 + pad_x)))
    y2 = int(min(height, np.ceil(y2 + pad_y)))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return frame[y1:y2, x1:x2]


def to_batch(crops: list[np.ndarray], size: int = 224):
    import torch

    tensors = []
    for crop in crops:
        image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        tensors.append(torch.from_numpy(image.transpose(2, 0, 1)))
    return torch.stack(tensors)


def read_needed_frames(video_path: str, needed: set[int]) -> dict[int, np.ndarray]:
    """顺序解码一次，只保留需要的帧，避免把整段 1080p 读进内存。"""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return {}
    kept: dict[int, np.ndarray] = {}
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index in needed:
            kept[index] = frame
        index += 1
    capture.release()
    return kept


def process_folder(
    folder_rows: list[dict],
    uid_to_path: dict[str, str],
    backbone,
    device: str,
    frames_per_clip: int,
    batch_size: int,
    overwrite: bool,
) -> int:
    import torch

    folder = os.path.dirname(absolute_clip(folder_rows[0]["clip"]))
    output_path = os.path.join(folder, "appearance.npz")
    existing: dict[str, np.ndarray] = {}
    if os.path.exists(output_path) and not overwrite:
        with np.load(output_path) as data:
            existing = {key: data[key] for key in data.files}

    todo = [
        row
        for row in folder_rows
        if os.path.splitext(os.path.basename(row["clip"]))[0] not in existing
    ]
    if not todo:
        return 0

    by_uid: dict[str, list[dict]] = collections.defaultdict(list)
    for row in todo:
        uid = str(row.get("video_uid", "")).strip()
        if not uid:
            print(f"    !! 旧片段没有 video_uid，跳过：{row['clip']}")
            continue
        by_uid[uid].append(row)

    done = 0
    for uid, rows in by_uid.items():
        video_path = uid_to_path.get(uid)
        if not video_path or not os.path.exists(video_path):
            print(f"    !! 找不到源视频：{uid}")
            continue

        clip_data = []
        needed: set[int] = set()
        for row in rows:
            stem = os.path.splitext(os.path.basename(row["clip"]))[0]
            with np.load(absolute_clip(row["clip"])) as data:
                if "bbox" not in data:
                    continue
                boxes = data["bbox"].astype(np.float32)
                start = int(data["start_frame"])
            picks = np.linspace(0, len(boxes) - 1, frames_per_clip).astype(int)
            indices = [start + int(pick) for pick in picks]
            clip_data.append((stem, boxes, picks, indices))
            needed.update(indices)

        if not clip_data:
            continue
        frames = read_needed_frames(video_path, needed)
        if not frames:
            print(f"    !! 解码失败：{video_path}")
            continue

        crops: list[np.ndarray] = []
        owners: list[int] = []
        for clip_index, (_, boxes, picks, indices) in enumerate(clip_data):
            for pick, index in zip(picks, indices):
                frame = frames.get(index)
                if frame is None:
                    continue
                crop = crop_with_padding(frame, boxes[pick])
                if crop is None:
                    continue
                crops.append(crop)
                owners.append(clip_index)

        if not crops:
            continue
        features: list[np.ndarray] = []
        for start in range(0, len(crops), batch_size):
            batch = crops[start : start + batch_size]
            with torch.no_grad():
                output = backbone(to_batch(batch).to(device)).cpu().numpy()
            features.extend(list(output))

        grouped: dict[int, list[np.ndarray]] = collections.defaultdict(list)
        for owner, feature in zip(owners, features):
            grouped[owner].append(feature)
        for clip_index, (stem, _, _, _) in enumerate(clip_data):
            items = grouped.get(clip_index, [])
            if not items:
                continue
            feature = np.stack(items)
            if feature.shape[0] < frames_per_clip:
                pad = np.repeat(feature[-1:], frames_per_clip - feature.shape[0], axis=0)
                feature = np.concatenate([feature, pad], axis=0)
            existing[stem] = feature[:frames_per_clip].astype(np.float16)
            done += 1

    if done:
        np.savez_compressed(output_path, **existing)
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description="补新数据集的 RGB 外观特征")
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--videos", default=DEFAULT_VIDEOS)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--datasets", default="hmdb51,gmdcsa24")
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.labels):
        print(f"找不到标注：{args.labels}\n先跑 python adapters/merge_training_labels.py")
        return 1
    if not os.path.exists(args.videos):
        print(f"找不到视频清单：{args.videos}")
        return 1

    wanted = {item.strip() for item in args.datasets.split(",") if item.strip()}
    rows = [
        row for row in read_csv(args.labels) if row.get("dataset") in wanted
    ]
    if not rows:
        print("筛选后没有片段")
        return 1

    uid_to_path = {
        row["video_uid"]: row["path"]
        for row in read_csv(args.videos)
        if row.get("video_uid") and row.get("path")
    }
    if args.limit_videos:
        allowed = set(sorted(uid_to_path)[: args.limit_videos])
        rows = [row for row in rows if row.get("video_uid") in allowed]

    import torch
    import torchvision

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("加载 ResNet18（ImageNet 预训练）……")
    backbone = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    )
    backbone.fc = torch.nn.Identity()
    backbone.eval().to(device)

    by_folder: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_folder[os.path.dirname(row["clip"])].append(row)

    started = time.time()
    total = 0
    folders = sorted(by_folder)
    for index, folder in enumerate(folders, start=1):
        count = process_folder(
            by_folder[folder],
            uid_to_path,
            backbone,
            device,
            args.frames,
            args.batch_size,
            args.overwrite,
        )
        total += count
        print(f"  [{index}/{len(folders)}] {folder}  新增 {count}")

    print(
        f"\n完成：新增 {total} 个片段的外观特征，"
        f"用时 {(time.time()-started)/60:.1f} 分钟"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
