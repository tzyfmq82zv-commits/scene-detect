"""Run Qwen2.5-VL over a frozen VLM validation manifest.

Model is loaded once and reused for all clips.

Usage:
    python adapters/build_vlm_validation_set.py
    python adapters/vlm_batch_validate.py --limit 4   # smoke test
    python adapters/vlm_batch_validate.py             # all clips

Outputs:
    D:\\Models\\Qwen2.5-VL-3B-Instruct\\vlm_validation\\
      frames\\<clip_id>\\*.jpg
      results.json
      results.csv
      summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from vlm_check import CLASSES, extract_frames

DEFAULT_MANIFEST = r"D:\SceneDetection\features\vlm_validation\manifest.csv"
DEFAULT_MODEL = r"D:\Models\Qwen2.5-VL-3B-Instruct"


def build_prompt() -> str:
    return (
        "These images are chronological frames from one surveillance video clip. "
        "Classify the main human action. Choose exactly one label from: "
        + ", ".join(CLASSES)
        + ". `normal` means none of the target actions is present. "
        "Pay special attention to wall/fence interaction: wall_crossing means "
        "the person is climbing over, straddling, standing on, or moving from "
        "one side of a wall/fence to the other. Seeing a wall in the background "
        "is not enough. If the person is on/over a fence, choose wall_crossing "
        "even if the posture resembles walking. "
        "Use `uncertain` only if evidence is insufficient. "
        "Do not infer from filenames. Return only JSON with keys "
        "`action`, `confidence`, and `evidence`."
    )


def parse_answer(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            action = str(parsed.get("action", "parse_error")).strip().lower()
            if action not in CLASSES and action != "uncertain":
                action = "parse_error"
            return {
                "action": action,
                "confidence": float(parsed.get("confidence", 0) or 0),
                "evidence": str(parsed.get("evidence", "")),
            }
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    return {"action": "parse_error", "confidence": 0.0, "evidence": text[:300]}


def metrics(rows: list[dict]) -> dict:
    classes = list(CLASSES)
    confusion = [[0 for _ in classes] for _ in classes]
    errors = 0
    for row in rows:
        truth = row["ground_truth"]
        predicted = row["prediction"]
        if truth not in classes:
            continue
        if predicted in classes:
            confusion[classes.index(truth)][classes.index(predicted)] += 1
        else:
            errors += 1
    per_class = {}
    for index, name in enumerate(classes):
        tp = confusion[index][index]
        fp = sum(confusion[row][index] for row in range(len(classes))) - tp
        fn = sum(confusion[index]) - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[index]),
        }
    active = [name for name in classes if per_class[name]["support"] > 0]
    correct = sum(confusion[i][i] for i in range(len(classes)))
    total = sum(sum(row) for row in confusion)
    return {
        "accuracy": correct / max(total, 1),
        "macro_f1": sum(per_class[name]["f1"] for name in active) / max(len(active), 1),
        "classes": per_class,
        "confusion": confusion,
        "parse_errors": errors,
    }


def backup_invalid_result(path: Path) -> Path:
    backup = path.with_name(f"results.corrupt-{int(time.time())}.json")
    os.replace(path, backup)
    return backup


def durable_text_write(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Flush file data to disk before atomically replacing the target."""
    temporary_path = path.with_name(path.name + ".tmp")
    with open(temporary_path, "w", encoding=encoding, newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def durable_csv_write(path: Path, rows: list[dict]) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    with open(temporary_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def save_outputs(results: list[dict], output_dir: Path) -> dict:
    """Durably save checkpoints after each clip."""
    result_path = output_dir / "results.json"
    durable_text_write(
        result_path, json.dumps(results, ensure_ascii=False, indent=2)
    )
    durable_csv_write(output_dir / "results.csv", results)

    summary = metrics(results)
    summary_path = output_dir / "summary.json"
    durable_text_write(
        summary_path, json.dumps(summary, ensure_ascii=False, indent=2)
    )
    return summary


def print_summary(summary: dict, completed: int, total: int) -> None:
    print("\n" + "=" * 64)
    print(f"completed {completed}/{total} clips")
    print(f"accuracy {summary['accuracy']:.4f}  macro F1 {summary['macro_f1']:.4f}")
    for name, item in summary["classes"].items():
        if item["support"]:
            print(
                f"  {name:<14} P={item['precision']:.3f} "
                f"R={item['recall']:.3f} F1={item['f1']:.3f} "
                f"n={item['support']}"
            )
    print(f"parse errors: {summary['parse_errors']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch Qwen2.5-VL validation")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard an existing results.json and start from the first clip",
    )
    args = parser.parse_args()

    model_dir = Path(args.model)
    output_dir = Path(args.output) if args.output else model_dir / "vlm_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("Empty manifest")
        return 1

    result_path = output_dir / "results.json"
    if args.fresh and result_path.exists():
        backup = backup_invalid_result(result_path)
        print(f"Previous result backed up to: {backup}")

    results_by_id: dict[str, dict] = {}
    if result_path.exists():
        try:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            results_by_id = {row["id"]: row for row in existing}
            print(f"Resuming: found {len(results_by_id)} completed clips")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            backup = backup_invalid_result(result_path)
            print(f"Invalid checkpoint moved to: {backup}")
            print(f"Reason: {exc}")
            print("Starting from the first clip.")

    row_ids = [row["id"] for row in rows]
    pending_rows = [row for row in rows if row["id"] not in results_by_id]
    completed_results = [
        results_by_id[row_id] for row_id in row_ids if row_id in results_by_id
    ]
    if not pending_rows:
        summary = save_outputs(completed_results, output_dir)
        print_summary(summary, len(completed_results), len(rows))
        print(f"Results: {result_path}")
        return 0

    print(
        f"Pending {len(pending_rows)}/{len(rows)} clips; "
        f"{args.frames} frames per clip"
    )
    print("Loading Qwen2.5-VL...")
    processor = AutoProcessor.from_pretrained(
        str(model_dir), trust_remote_code=True
    )
    dtype = torch.float32 if args.cpu else torch.bfloat16
    kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "device_map": "cpu" if args.cpu else "auto",
    }
    if not args.cpu:
        kwargs["max_memory"] = {0: "7GiB", "cpu": "16GiB"}
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_dir), **kwargs
    )
    model.eval()
    prompt = build_prompt()

    started = time.time()
    for index, row in enumerate(pending_rows, start=1):
        crop = None
        if row.get("crop"):
            crop = tuple(float(value) for value in row["crop"].split(","))
        frame_dir = output_dir / "frames" / row["id"]
        frames = extract_frames(
            row["source"],
            float(row["start_sec"]),
            float(row["duration_sec"]),
            args.frames,
            frame_dir,
            crop,
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
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        answer = processor.batch_decode(
            generated[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        parsed = parse_answer(answer)
        result = {
            **row,
            "prediction": parsed["action"],
            "confidence": parsed["confidence"],
            "evidence": parsed["evidence"],
            "raw_answer": answer,
            "correct": parsed["action"] == row["ground_truth"],
        }
        results_by_id[row["id"]] = result
        completed_results = [
            results_by_id[row_id] for row_id in row_ids if row_id in results_by_id
        ]
        summary = save_outputs(completed_results, output_dir)
        elapsed = time.time() - started
        eta = elapsed / index * (len(pending_rows) - index)
        print(
            f"[{index:02d}/{len(pending_rows)}] {row['id']:<24} "
            f"truth={row['ground_truth']:<14} pred={parsed['action']:<14} "
            f"conf={parsed['confidence']:.2f}  ETA {eta/60:.1f} min",
            flush=True,
        )

    print_summary(summary, len(completed_results), len(rows))
    print(f"Results: {result_path}")
    print(f"Summary: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
