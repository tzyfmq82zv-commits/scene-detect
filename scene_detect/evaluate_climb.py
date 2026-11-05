"""在官方 test 上评估攀爬二分类模型。

用法：
    python evaluate_climb.py
    python evaluate_climb.py --split val
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_climb import (
    CACHE_ROOT,
    CLASS_NAMES,
    MANIFEST,
    ClimbClassifier,
    ClimbDataset,
    metrics,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="评估攀爬二分类模型")
    parser.add_argument("--checkpoint", default=os.path.join("runs_climb", "best.pt"))
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--cache-root", default=CACHE_ROOT)
    parser.add_argument("--split", choices=["test", "val", "train"], default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    with open(args.manifest, newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == args.split]
    if not rows:
        print(f"{args.split} 集为空")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClimbClassifier(dropout=checkpoint.get("dropout", 0.35)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    dataset = ClimbDataset(
        rows,
        args.cache_root,
        int(checkpoint.get("frames", 16)),
        int(checkpoint.get("image_size", 224)),
        False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    targets, predictions = [], []
    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(frames)
            predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            targets.extend(labels.numpy().tolist())

    report = metrics(np.asarray(targets), np.asarray(predictions))
    print(f"Climb {args.split}: {len(rows)} 段")
    print(f"accuracy {report['accuracy']:.4f}  macro F1 {report['macro_f1']:.4f}")
    for name in CLASS_NAMES:
        item = report[name]
        print(
            f"  {name:<14} P {item['precision']:.3f} "
            f"R {item['recall']:.3f} F1 {item['f1']:.3f} "
            f"正样本 {item['positives']}"
        )
    print("confusion matrix（行=真实，列=预测）")
    for row in report["confusion"]:
        print("  ", row)

    output = os.path.join(
        os.path.dirname(args.checkpoint), f"{args.split}_report.json"
    )
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"已写入：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
