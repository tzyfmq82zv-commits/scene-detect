"""对单个视频跑完整的检测流水线，输出带标注的画面和报警事件。

这是里程碑 3（集成 + 报警逻辑）的验证入口：跑通它说明
检测 → 跟踪 → 动作判定 → 投票报警 → 事件记录 这条链路是通的，
接下来 Qt 界面只是把同一套逻辑套上个界面。

用法：
    python run_video.py                      # 跑默认的样例视频
    python run_video.py --source 你的视频.mp4
    python run_video.py --source xxx.mp4 --no-save --show   # 直接弹窗看，不存文件
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import cv2
import numpy as np

from feature_extract import apply_crop
from pipeline import PipelineConfig, SceneDetector

DATASETS_ROOT = r"D:\SceneDetection\datasets"
OUTPUT_ROOT = r"D:\SceneDetection\output"
TEST_VIDEO_DIR = r"D:\SceneDetection\test_video"
TEST_VIDEO = os.path.join(TEST_VIDEO_DIR, "scene_test.mp4")

# 默认样例用 UR Fall：它是我们最可靠的一类（摔倒 F1 0.80、零误报），
# 拿来验证整条链路最直观。注意需要裁掉左半边的深度图。
URFALL_SAMPLE = os.path.join(DATASETS_ROOT, "fall", "urfall", "fall-01-cam0.mp4")
DEFAULT_SOURCE = TEST_VIDEO if os.path.exists(TEST_VIDEO) else URFALL_SAMPLE


def resolve_crop(source: str, requested: str) -> str:
    """决定要不要先裁掉画面的一半。

    UR Fall 的原始 mp4 是「深度图|彩色图」左右拼接的，不裁掉左半边的
    深度图就检不到人；但合成出来的测试视频已经裁好了，再裁一次会把
    人切成两半。所以默认交给这里按文件名自动判断。
    """
    if requested != "auto":
        return requested
    return "right" if "urfall" in source.lower() else "none"


def open_video(path: str, crop: str):
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    return capture, fps


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real-time scene detection")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Input video")
    parser.add_argument("--output", default=None, help="Annotated video output path")
    parser.add_argument("--events", default=None, help="Alert event JSON output path")
    parser.add_argument(
        "--crop",
        choices=["auto", "none", "left", "right", "top", "bottom"],
        default="auto",
        help="Crop half of the frame. auto: UR Fall uses the right color half",
    )
    parser.add_argument(
        "--classifier",
        default=None,
        help="Model weights; default runs_six_transfer/best.pt",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N frames")
    parser.add_argument("--no-save", action="store_true", help="Do not write annotated video")
    parser.add_argument("--show", action="store_true", help="Preview while processing; press q to quit")
    parser.add_argument(
        "--alarm-fighting",
        action="store_true",
        help="Enable fighting alarms (off by default due to false positives)",
    )
    parser.add_argument("--alarm-running", action="store_true", help="Enable running alarms")
    parser.add_argument(
        "--temporal-confirmations",
        type=int,
        default=3,
        help="Consecutive classifications required before switching action",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Pose inference resolution; lower is faster",
    )
    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"Input video not found: {args.source}")
        print("Use --source to specify a video.")
        return 1

    config = PipelineConfig()
    if args.classifier:
        config.classifier_path = args.classifier
    config.alarm_fighting = args.alarm_fighting
    config.alarm_running = args.alarm_running
    config.action_switch_confirmations = max(1, args.temporal_confirmations)
    config.imgsz = max(320, args.imgsz)

    crop = resolve_crop(args.source, args.crop)

    print("Loading models...")
    detector = SceneDetector(config)
    print(
        f"  Action model : {os.path.basename(config.classifier_path)}"
        f"  label columns {list(detector.label_columns)}"
    )
    print(f"  Thresholds : " + "  ".join(
        f"{k}={v:.3f}" for k, v in detector.thresholds.items()
    ))
    print(f"  Running threshold : {config.running_threshold} body-heights/s (rule)")
    print(
        f"  Alarms : Falling=on  Fighting={'on' if config.alarm_fighting else 'off'}  "
        f"Running={'on' if config.alarm_running else 'off'}"
    )

    if crop != "none":
        print(f"  Crop : keep {crop} half")
    capture, fps = open_video(args.source, crop)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if crop in ("left", "right"):
        width //= 2
    elif crop in ("top", "bottom"):
        height //= 2
    print(f"\nInput: {os.path.basename(args.source)}  {width}x{height}  {fps:.1f}fps  {total_frames} frames")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.source))[0]
    output_path = args.output or os.path.join(OUTPUT_ROOT, f"{stem}_annotated.mp4")
    events_path = args.events or os.path.join(OUTPUT_ROOT, f"{stem}_events.json")

    writer = None
    if not args.no_save:
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

    all_events = []
    frame_index = 0
    started = time.time()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if crop != "none":
                frame = apply_crop(frame, crop)
            annotated, events = detector.process(frame, frame_index, fps)
            if writer is not None:
                writer.write(annotated)
            if args.show:
                cv2.imshow("scene detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            for event in events:
                all_events.append(event.to_dict())
                print(
                    f"  [{event.to_dict()['time_sec']:6.2f}s]  Track {event.track_id}  "
                    f"{event.action.replace('_', ' ').title()}  confidence {event.confidence:.3f}  "
                    f"{event.detail}"
                )
            frame_index += 1
            if args.limit and frame_index >= args.limit:
                break
            if frame_index % 100 == 0:
                elapsed = time.time() - started
                print(
                    f"    {frame_index}/{total_frames or '?'} frames  "
                    f"{frame_index/elapsed:.1f} fps  events {len(all_events)}"
                )
    except KeyboardInterrupt:
        print("\nInterrupted; processed output is still saved")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.time() - started
    with open(events_path, "w", encoding="utf-8") as handle:
        json.dump(all_events, handle, ensure_ascii=False, indent=2)

    print("\n" + "=" * 62)
    print(f"Processed {frame_index} frames in {elapsed:.1f}s, {frame_index/max(elapsed,1):.1f} fps")
    print(f"Alert events: {len(all_events)}")
    if all_events:
        counts: dict[str, int] = {}
        for item in all_events:
            name = item["action"].replace("_", " ").title()
            counts[name] = counts.get(name, 0) + 1
        for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {name}: {count}")
    print(f"Annotated video: {output_path}" + (" (not saved)" if args.no_save else ""))
    print(f"Event JSON: {events_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
