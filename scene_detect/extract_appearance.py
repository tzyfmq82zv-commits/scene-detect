"""给每个骨架片段算一份 RGB 外观特征，供双流模型的第二条支路使用。

可行性验证已经证明外观里有信号（AUROC 0.81，骨架只有 0.55 左右），
这一步是把「人所在区域的画面长什么样」变成可以喂给网络的向量。

关键设计：**不重新跑 YOLO 和跟踪**。片段文件里已经存了每一帧的检测框，
直接按那些框裁剪就行。重跑一遍检测加跟踪要几十分钟，而这里只解码视频
再裁剪，快得多，而且结果和训练时的框完全一致。

每个片段存 8 帧的 ResNet18 特征（8x512 的 float16），约 8 KB，
全部 4000 多个片段加起来才 30 MB 左右，比存裁好的图片小几十倍。

用法：
    python extract_appearance.py              # 全部数据集
    python extract_appearance.py --only busviolence_fight
    python extract_appearance.py --frames 4   # 少采几帧更快
"""

from __future__ import annotations

import argparse
import os
import re
import time
from collections import defaultdict

import cv2
import numpy as np

from feature_extract import apply_crop

FEATURES_ROOT = r"D:\SceneDetection\features"
DATASETS_ROOT = r"D:\SceneDetection\datasets"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 检测框往外扩一点，把躯干和手臂的接触区域也纳进来——
# 打架的关键外观线索常常在两个人的交界处，卡着框裁会切掉
CROP_PADDING = 0.18

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}


def resolve_crop(dataset: str, video_path: str) -> str:
    """决定读帧时要不要先裁掉画面的一半。

    UR Fall 的原始 mp4 是「深度图|彩色图」左右拼接的，抽骨架时裁掉了
    左半边的深度图，所以片段里存的检测框是**裁完之后**的坐标系。

    抽外观特征时如果直接读未裁剪的原图，按那些框裁出来的就是深度图那一块，
    完全是垃圾——实测就是这样：摔倒这一类的训练输入全成了深度图，
    训练出来的摔倒分支在真实视频上输出恒等于 0。

    这里刻意不 import run_video（会导致循环依赖），按数据集名自行判断。
    """
    text = f"{dataset} {video_path}".lower()
    if "urfall" in text:
        return "right"
    return "none"


def find_source_video(source_name: str, cache: dict) -> str | None:
    """按片段里记的 source 名（不带扩展名）找到原始视频路径。"""
    if not cache:
        for root, _, files in os.walk(DATASETS_ROOT):
            for name in files:
                if os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS:
                    cache[os.path.splitext(name)[0]] = os.path.join(root, name)
    return cache.get(source_name)


def crop_with_padding(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    """按检测框裁剪并外扩，越界部分用边缘像素补齐。"""
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
    """把裁剪出来的小块统一缩放、归一化成网络输入。"""
    import torch

    tensors = []
    for crop in crops:
        image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        tensors.append(torch.from_numpy(image.transpose(2, 0, 1)))
    return torch.stack(tensors)


def process_dataset(
    dataset: str,
    frames_per_clip: int,
    backbone,
    device: str,
    video_cache: dict,
    overwrite: bool,
) -> int:
    """处理一个数据集，把每个片段的外观特征写进 appearance.npz。"""
    import torch

    folder = os.path.join(FEATURES_ROOT, dataset)
    if not os.path.isdir(folder):
        print(f"  跳过（目录不存在）：{folder}")
        return 0
    output_path = os.path.join(folder, "appearance.npz")

    existing: dict[str, np.ndarray] = {}
    if os.path.exists(output_path) and not overwrite:
        with np.load(output_path) as data:
            existing = {key: data[key] for key in data.files}
        print(f"  已有 {len(existing)} 个片段的外观特征，将增量补齐")

    # 按源视频分组：同一段视频只解码一次，它的所有片段共用这些帧
    by_video: dict[str, list[tuple[str, np.ndarray, int]]] = defaultdict(list)
    clip_files = sorted(f for f in os.listdir(folder) if f.endswith(".npz"))
    for filename in clip_files:
        stem = os.path.splitext(filename)[0]
        if stem in existing:
            continue
        with np.load(os.path.join(folder, filename)) as data:
            if "bbox" not in data:
                continue
            source = str(data["source"]) if "source" in data else stem.split("__")[0]
            start = int(data["start_frame"]) if "start_frame" in data else 0
            boxes = data["bbox"].astype(np.float32)
        by_video[source].append((stem, boxes, start))

    todo = sum(len(v) for v in by_video.values())
    if not todo:
        print("  没有需要处理的片段")
        return 0
    print(f"  待处理 {todo} 个片段，来自 {len(by_video)} 段视频")

    results = dict(existing)
    done = 0
    started = time.time()
    for source, items in by_video.items():
        video_path = find_source_video(source, video_cache)
        if video_path is None:
            print(f"    !! 找不到源视频，跳过：{source}")
            continue
        crop = resolve_crop(dataset, video_path)
        capture = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            # 必须和抽骨架时用同样的裁剪，否则检测框和画面不在一个坐标系里
            if crop != "none":
                frame = apply_crop(frame, crop)
            frames.append(frame)
        capture.release()
        if not frames:
            continue
        total = len(frames)

        for stem, boxes, start in items:
            count = boxes.shape[0]
            picks = np.linspace(0, count - 1, frames_per_clip).astype(int)
            crops = []
            for pick in picks:
                frame_index = min(start + int(pick), total - 1)
                crop = crop_with_padding(frames[frame_index], boxes[pick])
                if crop is not None:
                    crops.append(crop)
            if not crops:
                continue
            with torch.no_grad():
                feature = backbone(to_batch(crops).to(device)).cpu().numpy()
            # 帧数不足时用重复补齐，保证每条支路的输入长度一致
            if feature.shape[0] < frames_per_clip:
                pad = np.repeat(feature[-1:], frames_per_clip - feature.shape[0], axis=0)
                feature = np.concatenate([feature, pad], axis=0)
            results[stem] = feature.astype(np.float16)
            done += 1
        if done % 500 < len(items):
            elapsed = time.time() - started
            print(f"    {done}/{todo}  用时 {elapsed:.0f}s")

    np.savez_compressed(output_path, **results)
    print(f"  完成 {done} 个，累计 {len(results)} 个 -> {output_path}")
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description="抽取每段片段的 RGB 外观特征")
    parser.add_argument("--only", default=None, help="只处理指定数据集，逗号分隔")
    parser.add_argument("--frames", type=int, default=8, help="每个片段采样几帧")
    parser.add_argument("--overwrite", action="store_true", help="丢弃已有结果重新算")
    args = parser.parse_args()

    import torch
    import torchvision

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("加载 ResNet18（ImageNet 预训练）……")
    backbone = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    )
    backbone.fc = torch.nn.Identity()
    backbone.eval().to(device)

    datasets = [
        d
        for d in sorted(os.listdir(FEATURES_ROOT))
        if os.path.isdir(os.path.join(FEATURES_ROOT, d))
        and os.path.exists(os.path.join(FEATURES_ROOT, d, "manifest.csv"))
    ]
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        datasets = [d for d in datasets if d in wanted]
    if not datasets:
        print("没有找到可处理的数据集")
        return 1

    video_cache: dict[str, str] = {}
    total = 0
    for dataset in datasets:
        print(f"\n[{dataset}]")
        total += process_dataset(
            dataset, args.frames, backbone, device, video_cache, args.overwrite
        )
    print(f"\n全部完成，共生成 {total} 个片段的外观特征")
    print("下一步：python train.py --model dual")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
