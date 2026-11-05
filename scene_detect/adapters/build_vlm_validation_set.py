"""Build a small frozen manifest for VLM validation.

Contains:
    - the 14 segments of scene_test.mp4
    - 5 Normal and 5 Wall crossing clips from the wall-security test set

The VLM must not receive the filename or ground truth in the prompt.

Output:
    D:\\SceneDetection\\features\\vlm_validation\\manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import cv2

SCENE_VIDEO = r"D:\SceneDetection\test_video\scene_test.mp4"
TIMELINE = r"D:\SceneDetection\test_video\scene_test_timeline.json"
WALL_ROOT = r"D:\SceneDetection\datasets\Wall Security Dataset ALEE\test"
DEFAULT_OUT = r"D:\SceneDetection\features\vlm_validation\manifest.csv"


def duration(path: str) -> float:
    capture = cv2.VideoCapture(path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    return frames / fps if fps else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build VLM validation manifest")
    parser.add_argument("--scene-video", default=SCENE_VIDEO)
    parser.add_argument("--timeline", default=TIMELINE)
    parser.add_argument("--wall-root", default=WALL_ROOT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    timeline = json.loads(Path(args.timeline).read_text(encoding="utf-8"))
    rows = []
    for index, segment in enumerate(timeline):
        rows.append(
            {
                "id": f"scene_{index:02d}",
                "source": args.scene_video,
                "start_sec": segment["start_sec"],
                "duration_sec": round(segment["end_sec"] - segment["start_sec"], 2),
                "ground_truth": segment["action"],
                "label": segment["label"],
                "crop": "0.05,0.05,0.62,0.95" if segment["action"] == "wall_crossing" else "",
                "note": "scene_test timeline",
            }
        )

    for class_name, ground_truth, count in (
        ("Normal Class", "normal", 5),
        ("Wall crossing", "wall_crossing", 5),
    ):
        folder = Path(args.wall_root) / class_name
        files = sorted(
            (
                path
                for path in folder.iterdir()
                if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
            ),
            key=lambda path: path.name,
        )
        if ground_truth == "wall_crossing":
            files = [path for path in files if path.name != "10.mp4"]
        for path in files[:count]:
            rows.append(
                {
                    "id": f"wall_{ground_truth}_{path.stem}",
                    "source": str(path),
                    "start_sec": 0.0,
                    "duration_sec": round(min(5.0, duration(str(path))), 2),
                    "ground_truth": ground_truth,
                    "label": class_name,
                    "crop": "",
                    "note": "wall-security test",
                }
            )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for row in rows:
        counts[row["ground_truth"]] = counts.get(row["ground_truth"], 0) + 1
    print(f"Wrote {len(rows)} clips to {output}")
    print("Ground truth distribution:", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
