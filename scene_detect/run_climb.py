"""用攀爬二分类模型对视频做滑窗推理。

输出：
    D:\\SceneDetection\\output\\<视频名>_climb.mp4
    D:\\SceneDetection\\output\\<视频名>_climb_events.json

用法：
    python run_climb.py --source "D:\\path\\video.mp4" --show
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, deque

import cv2
import numpy as np
import torch

from train_climb import CLASS_NAMES, ClimbClassifier

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
OUTPUT_ROOT = r"D:\SceneDetection\output"
DEFAULT_SOURCE = (
    r"D:\SceneDetection\datasets\Wall Security Dataset ALEE\test\Wall crossing\1.mp4"
)


def default_source() -> str:
    """优先用 1.mp4；不存在时取 Wall crossing 目录里的第一个视频。"""
    if os.path.exists(DEFAULT_SOURCE):
        return DEFAULT_SOURCE
    folder = os.path.dirname(DEFAULT_SOURCE)
    if os.path.isdir(folder):
        for name in sorted(os.listdir(folder)):
            if os.path.splitext(name)[1].lower() in {".mp4", ".avi", ".mov", ".mkv"}:
                return os.path.join(folder, name)
    return DEFAULT_SOURCE


def sample_frames(
    capture,
    start_frame: int,
    count: int,
    total: int,
    frames: int,
    image_size: int,
) -> np.ndarray:
    picks = np.linspace(
        start_frame, min(start_frame + count - 1, total - 1), frames
    ).astype(int)
    sampled = []
    for index in picks:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.resize(frame, (image_size, image_size))
        sampled.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not sampled:
        return np.zeros((frames, image_size, image_size, 3), dtype=np.uint8)
    while len(sampled) < frames:
        sampled.append(sampled[-1])
    return np.stack(sampled[:frames]).astype(np.uint8)


def prepare_batch(frames: np.ndarray) -> torch.Tensor:
    array = frames.astype(np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(array.transpose(0, 3, 1, 2)).unsqueeze(0).float()


def main() -> int:
    parser = argparse.ArgumentParser(description="Wall-crossing sliding-window inference")
    parser.add_argument(
        "--source",
        default=default_source(),
        help="Input video; defaults to test Wall crossing/1.mp4",
    )
    parser.add_argument("--checkpoint", default=os.path.join("runs_climb", "best.pt"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--events", default=None)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument(
        "--cooldown",
        type=float,
        default=5.0,
        help="Cooldown in seconds; default 5 to avoid duplicate alarms",
    )
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"Video not found: {args.source}")
        return 1
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClimbClassifier(dropout=checkpoint.get("dropout", 0.35)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    frames_per_clip = int(checkpoint.get("frames", 16))
    image_size = int(checkpoint.get("image_size", 224))

    capture = cv2.VideoCapture(args.source)
    if not capture.isOpened():
        print(f"Cannot open video: {args.source}")
        return 1
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stem = os.path.splitext(os.path.basename(args.source))[0]
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    output_path = args.output or os.path.join(OUTPUT_ROOT, f"{stem}_climb.mp4")
    events_path = args.events or os.path.join(OUTPUT_ROOT, f"{stem}_climb_events.json")
    writer = None
    if not args.no_save:
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

    window = max(1, int(round(args.window_seconds * fps)))
    stride = max(1, int(round(args.stride_seconds * fps)))
    history = deque(maxlen=3)
    events = []
    last_alarm = -1e9
    frame_index = 0
    try:
        while frame_index < total:
            frames = sample_frames(
                capture, frame_index, window, total, frames_per_clip, image_size
            )
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits = model(prepare_batch(frames).to(device))
                probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
            best = int(np.argmax(probabilities))
            best_name = CLASS_NAMES[best]
            confidence = float(probabilities[best])
            history.append(best_name)
            candidate = Counter(history).most_common(1)[0]
            stable_name = candidate[0] if candidate[1] >= 2 else best_name
            stable_confidence = float(probabilities[CLASS_NAMES.index(stable_name)])
            timestamp = frame_index / fps

            if (
                stable_name == "wall_crossing"
                and stable_confidence >= args.threshold
                and timestamp - last_alarm >= args.cooldown
            ):
                last_alarm = timestamp
                events.append(
                    {
                        "time_sec": round(timestamp, 2),
                        "action": "wall_crossing",
                        "confidence": round(stable_confidence, 4),
                    }
                )
                print(
                    f"  [{timestamp:6.2f}s] wall_crossing "
                    f"{stable_confidence:.3f}"
                )

            if writer is not None or args.show:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    break
                color = (0, 0, 255) if stable_name == "wall_crossing" else (180, 180, 180)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), color, -1)
                cv2.putText(
                    frame,
                    f"{stable_name}  {stable_confidence:.2f}",
                    (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if writer is not None:
                    writer.write(frame)
                if args.show:
                    cv2.imshow("climb", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            frame_index += stride
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    with open(events_path, "w", encoding="utf-8") as handle:
        json.dump(events, handle, ensure_ascii=False, indent=2)
    print(f"\nEvents: {len(events)}")
    print(f"Annotated video: {output_path}")
    print(f"Event JSON: {events_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
