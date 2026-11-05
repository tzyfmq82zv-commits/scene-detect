"""用本地 Qwen2.5-VL 对一个视频窗口做多帧动作验证。

脚本默认从 scene_test.mp4 的 52.8–57.6 秒翻墙段抽 8 帧，然后让 VLM
在不看文件名、不看 ground truth 的情况下选择一个动作。

用法：
    python vlm_check.py
    python vlm_check.py --source "D:\path\video.mp4" --start 10 --duration 4
    python vlm_check.py --cpu

输出：
    frames_8/*.jpg
    VLM 原始回答（终端）以及 vlm_result.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

DEFAULT_MODEL = r"D:\Models\Qwen2.5-VL-3B-Instruct"
DEFAULT_SOURCE = r"D:\SceneDetection\test_video\scene_test.mp4"
CLASSES = [
    "normal",
    "running",
    "falling",
    "fighting",
    "sitting",
    "walking",
    "kicking",
    "wall_crossing",
]


def extract_frames(
    source: str,
    start_sec: float,
    duration: float,
    frame_count: int,
    output_dir: Path,
    crop: tuple[float, float, float, float] | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start = max(0, int(round(start_sec * fps)))
    end = min(total - 1, start + max(1, int(round(duration * fps))) - 1)
    picks = np.linspace(start, end, frame_count).astype(int)
    paths = []
    for index, frame_index in enumerate(picks):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            continue
        if crop is not None:
            height, width = frame.shape[:2]
            x1, y1, x2, y2 = crop
            frame = frame[
                int(y1 * height) : int(y2 * height),
                int(x1 * width) : int(x2 * width),
            ]
        path = output_dir / f"frame_{index:02d}.jpg"
        success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if success:
            path.write_bytes(encoded.tobytes())
            paths.append(path)
    capture.release()
    if not paths:
        raise SystemExit("No frames extracted")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen2.5-VL multi-frame action check")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--start", type=float, default=52.8)
    parser.add_argument("--duration", type=float, default=4.8)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument(
        "--crop",
        default="",
        help="Optional normalized crop x1,y1,x2,y2, e.g. 0.05,0.05,0.60,0.95",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--cpu", action="store_true", help="Force CPU (slow)")
    args = parser.parse_args()

    model_dir = Path(args.model)
    if not (model_dir / "config.json").exists():
        print(f"Model directory is incomplete: {model_dir}")
        return 1
    output_dir = Path(args.output) if args.output else model_dir / "frames_8"
    crop = None
    if args.crop:
        values = [float(item) for item in args.crop.split(",")]
        if len(values) != 4:
            raise SystemExit("--crop must contain four values")
        crop = tuple(values)
    frames = extract_frames(
        args.source, args.start, args.duration, args.frames, output_dir, crop
    )
    print(f"Extracted {len(frames)} frames to {output_dir}")

    print("Loading Qwen2.5-VL...")
    processor = AutoProcessor.from_pretrained(
        str(model_dir), trust_remote_code=True
    )
    dtype = torch.float32 if args.cpu else torch.bfloat16
    load_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if args.cpu:
        load_kwargs["device_map"] = "cpu"
    else:
        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = {0: "7GiB", "cpu": "16GiB"}
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_dir), **load_kwargs
    )
    model.eval()

    prompt = (
        "These images are chronological frames from one surveillance video clip. "
        "Classify the main human action. Choose exactly one label from: "
        + ", ".join(CLASSES)
        + ". `normal` means none of the target actions is present. "
        "Pay special attention to wall/fence interaction: wall_crossing means "
        "the person is climbing over, straddling, standing on, or moving from "
        "one side of a wall/fence to the other. Do not choose normal merely "
        "because the person is upright or walking; if the body is on/over the "
        "fence, choose wall_crossing. "
        "Use `uncertain` only if the visual evidence is insufficient. "
        "Do not guess from filenames. Return only JSON with keys "
        '`action` (one label), `confidence` (0 to 1), and `evidence` (short).'
    )
    content = [{"type": "image", "image": str(path)} for path in frames]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )
    trimmed = generated[:, inputs.input_ids.shape[1] :]
    answer = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    print("\nVLM answer:\n" + answer)
    result_path = model_dir / "vlm_result.json"
    result_path.write_text(
        json.dumps(
            {
                "source": args.source,
                "start_sec": args.start,
                "duration_sec": args.duration,
                "frames": len(frames),
                "answer": answer,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
