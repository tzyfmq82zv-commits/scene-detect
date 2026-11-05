"""用 YOLO11-Pose + 多目标跟踪从视频里提取骨架序列，产出训练用的 .npz 片段。

这是整条链路里最关键的一步预处理：
    视频 -> 逐帧人体检测与关键点 -> 按轨迹分配 ID -> 每条轨迹切成定长片段 -> .npz

为什么要先做这一步，而不是训练时现算：
  1. 姿态估计很慢，训练时要反复读同一段数据几十个 epoch，
     现算的话一个 epoch 要跑几个小时；先抽好特征就只剩读文件的开销；
  2. 抽取过程能人工检查——跟踪 ID 跳没跳、骨架崩没崩，一眼就能看出来；
  3. 之后换模型结构（ST-GCN / TCN / Transformer）时不用重新跑 YOLO。

用法示例：
    python feature_extract.py --source "D:/SceneDetection/datasets/fall/urfall" --out "D:/SceneDetection/features/urfall"
    python feature_extract.py --source video.mp4 --out out_dir --save-vis
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

# YOLO 输出的人体关键点数量（COCO-17）
NUM_KEYPOINTS = 17


@dataclass
class TrackBuffer:
    """一条轨迹的原始序列。轨迹可能长时间消失再出现，所以按帧号稀疏存。"""

    frames: list[int] = field(default_factory=list)
    keypoints: list[np.ndarray] = field(default_factory=list)
    boxes: list[np.ndarray] = field(default_factory=list)

    def add(self, frame_index: int, keypoints: np.ndarray, box: np.ndarray) -> None:
        self.frames.append(frame_index)
        self.keypoints.append(keypoints)
        self.boxes.append(box)

    def __len__(self) -> int:
        return len(self.frames)


def resample_sequence(
    keypoints: np.ndarray, boxes: np.ndarray, length: int
) -> tuple[np.ndarray, np.ndarray]:
    """把一段轨迹线性重采样到固定帧数，顺带让时间间隔变得均匀。

    轨迹里可能有丢帧（人短暂被遮挡时那一帧没检出），直接当连续帧用会让
    动作的时间节奏变形。这里统一重采样到固定长度，顺便把这个洞补上。
    """
    current = keypoints.shape[0]
    if current == length:
        return keypoints, boxes
    if current < 2:
        return np.repeat(keypoints, length, axis=0), np.repeat(boxes, length, axis=0)
    source = np.linspace(0.0, 1.0, current)
    target = np.linspace(0.0, 1.0, length)
    keypoint_flat = keypoints.reshape(current, -1)
    box_flat = boxes.reshape(current, -1)
    out_kpts = np.stack(
        [
            np.interp(target, source, keypoint_flat[:, col])
            for col in range(keypoint_flat.shape[1])
        ],
        axis=1,
    ).reshape(length, *keypoints.shape[1:])
    out_boxes = np.stack(
        [
            np.interp(target, source, box_flat[:, col])
            for col in range(box_flat.shape[1])
        ],
        axis=1,
    ).reshape(length, *boxes.shape[1:])
    return out_kpts.astype(np.float32), out_boxes.astype(np.float32)


def slice_track(
    track: TrackBuffer,
    clip_length: int,
    stride: int,
    max_clips: int,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """把一条轨迹切成若干定长片段，返回 (关键点, 检测框, 起始帧号) 列表。"""
    keypoints = np.stack(track.keypoints).astype(np.float32)
    boxes = np.stack(track.boxes).astype(np.float32)
    total = keypoints.shape[0]

    if total < clip_length:
        # 轨迹太短：整段放大到目标长度，当作一个片段用
        out_kpts, out_boxes = resample_sequence(keypoints, boxes, clip_length)
        return [(out_kpts, out_boxes, track.frames[0])]

    starts = list(range(0, total - clip_length + 1, stride))
    # 尾巴上如果还剩不少帧，也补一个片段，避免丢掉动作的后半段
    if starts and total - clip_length > starts[-1] + stride // 2:
        starts.append(total - clip_length)

    clips: list[tuple[np.ndarray, np.ndarray, int]] = []
    for start in starts[:max_clips]:
        window_kpts = keypoints[start:start + clip_length]
        window_boxes = boxes[start:start + clip_length]
        clips.append((window_kpts, window_boxes, track.frames[start]))
    return clips


def _write_visualisation(result, writer, out_dir: str, video_name: str):
    """把带骨架和 ID 的帧写进一个视频，用于人工检查跟踪是否稳定。

    这一步不是花哨的可选项——跟踪 ID 一旦频繁跳变，
    后面所有时序模型的输入都是垃圾，而这个问题只看数字指标是发现不了的。
    """
    import cv2

    frame = result.plot()
    if writer is None:
        vis_dir = os.path.join(out_dir, "_vis")
        os.makedirs(vis_dir, exist_ok=True)
        path = os.path.join(vis_dir, f"{video_name}.mp4")
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (width, height)
        )
    writer.write(frame)
    return writer


def extract_from_video(
    video_path: str,
    out_dir: str,
    model,
    tracker: str,
    clip_length: int,
    stride: int,
    min_confidence: float,
    min_track_length: int,
    max_clips: int,
    imgsz: int,
    save_vis: bool,
) -> list[dict]:
    """处理单个视频，返回生成的片段清单。"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    tracks: dict[int, TrackBuffer] = defaultdict(TrackBuffer)

    results = model.track(
        source=video_path,
        stream=True,
        # 必须是 False。persist=True 的本意是「在多次逐帧调用之间保留跟踪器状态」，
        # 我们是「一个视频调一次 track()」，用 True 会把上一段视频的跟踪器（包括
        # 里面的 GMC 全局运动补偿）带进下一段视频。不同视频分辨率不一样时，
        # GMC 会拿上一段视频的灰度图去算光流，金字塔尺寸对不上，就报
        # "GMC failed, falling back to identity"；同时上一段视频的轨迹 ID 也会
        # 残留下来，污染新视频开头一秒左右的 ID 分配。
        persist=False,
        tracker=tracker,
        classes=[0],            # 只要人这一类，其他检测结果直接丢掉
        conf=min_confidence,
        imgsz=imgsz,
        verbose=False,
    )

    writer = None
    for frame_index, result in enumerate(results):
        if save_vis:
            writer = _write_visualisation(result, writer, out_dir, video_name)

        boxes = result.boxes
        keypoints = result.keypoints
        if boxes is None or boxes.id is None or keypoints is None:
            continue

        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        kpts = keypoints.data.cpu().numpy()          # (人数, 17, 3)
        if kpts.shape[1] != NUM_KEYPOINTS:
            continue

        for person_index, track_id in enumerate(ids):
            tracks[track_id].add(frame_index, kpts[person_index], xyxy[person_index])

    if writer is not None:
        writer.release()

    manifest: list[dict] = []
    for track_id, track in tracks.items():
        if len(track) < min_track_length:
            continue
        for clip_index, (clip_kpts, clip_boxes, start_frame) in enumerate(
            slice_track(track, clip_length, stride, max_clips)
        ):
            name = f"{video_name}__track{track_id:04d}__t{clip_index:04d}.npz"
            path = os.path.join(out_dir, name)
            np.savez_compressed(
                path,
                keypoints=clip_kpts,          # (T, 17, 3)  最后一维是 x, y, 置信度
                bbox=clip_boxes,              # (T, 4)      xyxy 像素坐标
                source=video_name,
                track_id=track_id,
                start_frame=start_frame,
            )
            manifest.append(
                {
                    "clip": name,
                    "source": video_name,
                    "track_id": track_id,
                    "start_frame": start_frame,
                    "frames": int(clip_kpts.shape[0]),
                }
            )
    return manifest


def collect_videos(source: str) -> list[str]:
    """source 可以是单个视频，也可以是一个目录（会递归找常见后缀）。"""
    if os.path.isfile(source):
        return [source]
    extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}
    videos: list[str] = []
    for root, _, files in os.walk(source):
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() in extensions:
                videos.append(os.path.join(root, name))
    return videos


def apply_crop(frame: np.ndarray, crop: str) -> np.ndarray:
    """按方位裁掉画面的一半。"""
    height, width = frame.shape[:2]
    if crop == "left":
        return frame[:, : width // 2]
    if crop == "right":
        return frame[:, width // 2 :]
    if crop == "top":
        return frame[: height // 2, :]
    if crop == "bottom":
        return frame[height // 2 :, :]
    return frame


def crop_video(video_path: str, crop: str, out_path: str) -> str:
    """把左右（或上下）拼接的视频裁掉一半，写成新文件后返回路径。

    为什么需要这一步：UR Fall 的 mp4 是「深度图 | 彩色图」左右拼接的 640x240。
    左半边是深度图，YOLO-Pose 在它上面基本检不出人，会连带把整帧的检测结果带偏。
    必须先裁出右半边的彩色图，才能正常跑姿态估计。
    """
    import cv2

    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    writer = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            cropped = apply_crop(frame, crop)
            if writer is None:
                height, width = cropped.shape[:2]
                writer = cv2.VideoWriter(
                    out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
            writer.write(cropped)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="用 YOLO11-Pose + 多目标跟踪提取骨架序列"
    )
    parser.add_argument("--source", required=True, help="视频文件或视频目录")
    parser.add_argument("--out", required=True, help="输出 .npz 的目录")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="YOLO 姿态模型权重")
    parser.add_argument(
        "--tracker",
        default="botsort.yaml",
        help="跟踪器配置：botsort.yaml 抗遮挡更好，bytetrack.yaml 更快",
    )
    parser.add_argument("--clip-length", type=int, default=48, help="每个片段的帧数")
    parser.add_argument("--stride", type=int, default=24, help="滑窗步长，越小片段越多")
    parser.add_argument(
        "--min-confidence", type=float, default=0.3, help="人体检测置信度阈值"
    )
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=16,
        help="短于这个帧数的轨迹直接丢弃，避免把噪声当动作",
    )
    parser.add_argument(
        "--max-clips-per-track",
        type=int,
        default=8,
        help="单条轨迹最多产出的片段数，防止某个长镜头淹没整个数据集",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="推理分辨率")
    parser.add_argument("--save-vis", action="store_true", help="额外输出带骨架的可视化视频")
    parser.add_argument(
        "--crop",
        choices=["none", "left", "right", "top", "bottom"],
        default="none",
        help=(
            "先裁掉画面的一半再检测。UR Fall 的 mp4 是深度图|彩色图左右拼接，"
            "需要选 right；其余数据集保持 none"
        ),
    )

    # 不带参数直接运行时，多半是想跑下一步却拿错了文件，
    # 这里直接指明应该跑哪个脚本，省得对着 --source/--out 的报错发懵
    if len(sys.argv) == 1:
        print(
            "feature_extract.py 是单数据集工具，必须用 --source 和 --out 指定输入输出。\n"
            "\n"
            "如果你想一次跑完所有数据集（下一步该做的），请运行：\n"
            "    python extract_all.py\n"
            "\n"
            "--- 单数据集用法的参数说明 ---"
        )
        parser.print_help()
        return 1

    args = parser.parse_args()

    from ultralytics import YOLO

    os.makedirs(args.out, exist_ok=True)
    videos = collect_videos(args.source)
    if not videos:
        print(f"在 {args.source} 下没找到视频文件")
        return 1

    # 需要裁剪时先写一份裁好的临时视频；这样能复用 ultralytics 原生的视频读取，
    # 不必往 tracker 内部塞自己造的帧序列，省掉一类难查的跟踪器状态问题
    temp_dir = None
    if args.crop != "none":
        temp_dir = os.path.join(args.out, "_cropped")
        os.makedirs(temp_dir, exist_ok=True)
        print(f"预处理：裁出画面的 {args.crop} 半边 -> {temp_dir}")
        cropped_videos: list[str] = []
        for index, video in enumerate(videos, start=1):
            target = os.path.join(
                temp_dir, os.path.splitext(os.path.basename(video))[0] + ".mp4"
            )
            crop_video(video, args.crop, target)
            cropped_videos.append(target)
            print(f"  [{index}/{len(videos)}] {os.path.basename(video)}")
        videos = cropped_videos

    model = YOLO(args.model)
    all_rows: list[dict] = []
    for index, video in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] {os.path.basename(video)}")
        rows = extract_from_video(
            video_path=video,
            out_dir=args.out,
            model=model,
            tracker=args.tracker,
            clip_length=args.clip_length,
            stride=args.stride,
            min_confidence=args.min_confidence,
            min_track_length=args.min_track_length,
            max_clips=args.max_clips_per_track,
            imgsz=args.imgsz,
            save_vis=args.save_vis,
        )
        print(f"    产出 {len(rows)} 个片段")
        all_rows.extend(rows)

    manifest_path = os.path.join(args.out, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["clip", "source", "track_id", "start_frame", "frames"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n共生成 {len(all_rows)} 个骨架片段")
    print(f"清单文件：{manifest_path}")
    if temp_dir is not None:
        print(f"裁剪后的临时视频保留在 {temp_dir}，确认结果无误后可以删掉")
    print(
        "下一步：把 manifest.csv 复制成 labels.csv，为每个片段补上\n"
        "  running / falling / fighting 三列 0/1 标签（全 0 表示正常），\n"
        "  然后运行 train.py。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
