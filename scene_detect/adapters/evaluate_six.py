"""按数据集拆开评估六头模型，定位域偏移。

用法：
    python adapters/evaluate_six.py --checkpoint runs_six/best.pt

它会复现训练时的 balanced split，然后分别输出 urfall / busviolence /
weizmann / hmdb51 / gmdcsa24 上每一类的精确率、召回率和 F1。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from dataset import Clip, read_label_csv, split_by_group_balanced  # noqa: E402
from model import build_model  # noqa: E402
from train import evaluate_at, make_loader, predict  # noqa: E402


def dataset_of(path: str) -> str:
    lowered = path.lower().replace("\\", "/")
    for name in ("urfall", "busviolence", "weizmann", "hmdb51", "gmdcsa24"):
        if name in lowered:
            return name
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser(description="六头模型分数据集评估")
    parser.add_argument("--checkpoint", default=r"runs_six\best.pt")
    parser.add_argument("--labels", default=r"D:\SceneDetection\features\labels_six.csv")
    parser.add_argument("--features-root", default=r"D:\SceneDetection\features")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--compare", default=None, help="在同一验证片上额外评估旧权重")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    columns = tuple(checkpoint["label_columns"])
    thresholds = np.asarray(checkpoint["thresholds"], dtype=np.float32)
    clip_length = int(checkpoint.get("clip_length", 32))

    clips = read_label_csv(args.labels, root=args.features_root, columns=columns)
    _, val_clips = split_by_group_balanced(
        clips, columns=columns, val_ratio=args.val_ratio, seed=args.seed
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        checkpoint.get("model", "dual"),
        num_joints=17,
        in_channels=3,
        num_classes=len(columns),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    loader = make_loader(
        val_clips, clip_length, args.batch_size, 0, shuffle=False, augment=False
    )
    probabilities, targets = predict(model, loader, device, device.type == "cuda")

    buckets: dict[str, list[int]] = {}
    for index, clip in enumerate(val_clips):
        buckets.setdefault(dataset_of(clip.path), []).append(index)

    print(f"checkpoint epoch={checkpoint.get('epoch')}  macro={checkpoint.get('macro_f1'):.4f}")
    print(f"{'dataset':<14}{'n':>6}  " + "  ".join(f"{c[:4]:>10}" for c in columns))
    for name in sorted(buckets):
        indices = np.asarray(buckets[name], dtype=np.int64)
        report = evaluate_at(
            probabilities[indices], targets[indices], thresholds, columns
        )
        values = "  ".join(f"{report[c]['f1']:>10.3f}" for c in columns)
        print(f"{name:<14}{len(indices):>6}  {values}")

    lines = []
    for name in sorted(buckets):
        indices = np.asarray(buckets[name], dtype=np.int64)
        report = evaluate_at(
            probabilities[indices], targets[indices], thresholds, columns
        )
        lines.append(f"\n[{name}]")
        for column in columns:
            item = report[column]
            lines.append(
                f"  {column:<9} P={item['precision']:.3f} "
                f"R={item['recall']:.3f} F1={item['f1']:.3f} "
                f"正={item['positives']}"
            )
    print("\n".join(lines))

    if args.compare:
        old = torch.load(args.compare, map_location="cpu", weights_only=False)
        old_columns = tuple(old["label_columns"])
        indices = [columns.index(column) for column in old_columns]
        old_clips = [
            Clip(
                path=clip.path,
                labels=np.asarray(clip.labels)[indices],
                group=clip.group,
            )
            for clip in val_clips
        ]
        old_model = build_model(
            old.get("model", "dual"),
            num_joints=17,
            in_channels=3,
            num_classes=len(old_columns),
        ).to(device)
        old_model.load_state_dict(old["state_dict"])
        old_model.eval()
        old_loader = make_loader(
            old_clips, clip_length, args.batch_size, 0, shuffle=False, augment=False
        )
        old_prob, old_target = predict(
            old_model, old_loader, device, device.type == "cuda"
        )
        old_thresholds = np.asarray(old["thresholds"], dtype=np.float32)
        print(
            f"\n[compare] {args.compare} epoch={old.get('epoch')} "
            f"macro={old.get('macro_f1'):.4f}"
        )
        print(f"{'dataset':<14}{'n':>6}  " + "  ".join(
            f"{c[:4]:>10}" for c in old_columns
        ))
        for name in sorted(buckets):
            idx = np.asarray(buckets[name], dtype=np.int64)
            report = evaluate_at(
                old_prob[idx], old_target[idx], old_thresholds, old_columns
            )
            values = "  ".join(f"{report[c]['f1']:>10.3f}" for c in old_columns)
            print(f"{name:<14}{len(idx):>6}  {values}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
