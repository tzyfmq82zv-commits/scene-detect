"""训练双流动作识别模型（跑 / 摔 / 打 / 坐 / 走 / 踢，多标签）。

用法：
    python train.py --labels labels.csv --features-root ./features --epochs 60
    python train.py --labels labels.csv --features-root ./features --model tcn   # 跑基线对比

几个刻意的设计选择：
  * 按视频划分训练/验证集：同一视频的相邻片段相似度极高，按片段随机划分
    会让指标虚高十几点，到现场直接崩；
  * 每个类别各用独立阈值：摔倒要的是高召回（漏报代价大），打架要的是
    低误报（误报会让人麻木），统一阈值没法同时满足；
  * 用 BCEWithLogitsLoss + pos_weight 而不是交叉熵：正负样本比例悬殊，
    且允许「边跑边打」这种多标签情况。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
    LABEL_COLUMNS,
    LABEL_NAMES_ZH,
    SkeletonDataset,
    compute_pos_weight,
    read_label_csv,
    split_by_group,
    split_by_group_balanced,
    summarize,
)
from model import build_model


DEFAULT_SIX = r"D:\SceneDetection\features\labels_six.csv"
DEFAULT_LABELS = (
    DEFAULT_SIX
    if os.path.exists(DEFAULT_SIX)
    else r"D:\SceneDetection\features\labels.csv"
)
DEFAULT_FEATURES_ROOT = r"D:\SceneDetection\features"
DEFAULT_LABELS_COLS = "running,falling,fighting,sitting,walking,kicking"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(clips, clip_length, batch_size, workers, shuffle, augment):
    dataset = SkeletonDataset(clips, clip_length=clip_length, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle and len(dataset) >= batch_size,
        persistent_workers=workers > 0,
    )


def train_one_epoch(
    model, loader, criterion, optimiser, scaler, device, amp, grad_clip,
    frozen_backbone=False,
):
    model.train()
    if frozen_backbone:
        # requires_grad=False 不会冻结 BatchNorm 的滑动统计，必须切回 eval
        model.skeleton_backbone.eval()
        model.appearance_encoder.eval()
    total_loss = 0.0
    seen = 0
    for primary, neighbor, pair, appearance, labels in loader:
        primary = primary.to(device, non_blocking=True)
        neighbor = neighbor.to(device, non_blocking=True)
        pair = pair.to(device, non_blocking=True)
        appearance = appearance.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimiser.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(primary, neighbor, pair, appearance)
            loss = criterion(logits, labels)
        if amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimiser)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()
        total_loss += loss.item() * primary.size(0)
        seen += primary.size(0)
    return total_loss / max(seen, 1)


@torch.no_grad()
def predict(model, loader, device, amp):
    """跑一遍验证集，返回概率和标签，阈值扫描在 CPU 侧做。"""
    model.eval()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for primary, neighbor, pair, appearance, labels in loader:
        primary = primary.to(device, non_blocking=True)
        neighbor = neighbor.to(device, non_blocking=True)
        pair = pair.to(device, non_blocking=True)
        appearance = appearance.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(primary, neighbor, pair, appearance)
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
        targets.append(labels.numpy())
    return np.concatenate(probabilities), np.concatenate(targets)


def sweep_thresholds(
    probabilities: np.ndarray, targets: np.ndarray, columns: tuple[str, ...]
):
    """为每个类别单独找 F1 最高的阈值。

    固定 0.5 是在假设正负样本均衡，而我们的数据远不均衡，
    0.5 往往让召回率惨不忍睹。这里直接扫一遍，选出各类最优阈值。
    """
    grid = np.arange(0.05, 0.951, 0.025)
    best_threshold = np.full(len(columns), 0.5, dtype=np.float32)
    best_f1 = np.zeros(len(columns), dtype=np.float32)
    for index in range(len(columns)):
        truth = targets[:, index] > 0.5
        if truth.sum() == 0:
            continue
        scores = probabilities[:, index]
        for threshold in grid:
            predicted = scores >= threshold
            true_positive = np.logical_and(predicted, truth).sum()
            if true_positive == 0:
                continue
            precision = true_positive / predicted.sum()
            recall = true_positive / truth.sum()
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > best_f1[index]:
                best_f1[index] = f1
                best_threshold[index] = threshold
    return best_threshold, best_f1


def evaluate_at(
    probabilities: np.ndarray,
    targets: np.ndarray,
    thresholds: np.ndarray,
    columns: tuple[str, ...],
):
    """按给定阈值算每个类别的 precision / recall / F1 / 样本数。"""
    report = {}
    for index, column in enumerate(columns):
        predicted = probabilities[:, index] >= thresholds[index]
        truth = targets[:, index] > 0.5
        true_positive = int(np.logical_and(predicted, truth).sum())
        false_positive = int(np.logical_and(predicted, ~truth).sum())
        false_negative = int(np.logical_and(~predicted, truth).sum())
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        report[column] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "threshold": float(thresholds[index]),
            "positives": int(truth.sum()),
            "predicted": int(predicted.sum()),
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
        }
    report["macro_f1"] = float(np.mean([report[c]["f1"] for c in columns]))
    return report


def format_report(report: dict, columns: tuple[str, ...]) -> str:
    lines = ["  类别     阈值   精确率   召回率    F1     正样本   实际触发"]
    for column in columns:
        item = report[column]
        lines.append(
            f"  {LABEL_NAMES_ZH[column]:<5}  {item['threshold']:.2f}   "
            f"{item['precision']:6.3f}   {item['recall']:6.3f}   {item['f1']:6.3f}"
            f"   {item['positives']:6d}   {item['predicted']:6d}"
        )
    lines.append(f"  宏平均 F1 = {report['macro_f1']:.4f}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="训练骨架动作识别模型")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="标注 CSV 路径")
    parser.add_argument(
        "--features-root", default=DEFAULT_FEATURES_ROOT, help="片段相对路径的根目录"
    )
    parser.add_argument(
        "--labels-cols",
        default=DEFAULT_LABELS_COLS,
        help=(
            "训练哪几列标签，逗号分隔。六类完整模型传 "
            "running,falling,fighting,sitting,walking,kicking"
        ),
    )
    parser.add_argument(
        "--split-mode",
        choices=["balanced", "random"],
        default="balanced",
        help="balanced：多次尝试按组划分，让每类验证比例接近 val-ratio",
    )
    parser.add_argument("--out", default="./runs", help="模型和日志输出目录")
    parser.add_argument(
        "--model",
        default="dual",
        choices=["dual", "stgcn_interaction", "stgcn", "tcn"],
        help=(
            "默认 dual（双流：骨架 + RGB 外观）。外观可行性验证里 AUROC 0.81，"
            "而纯骨架只有 0.55 左右，所以打架这一类得靠外观支路。"
            "先跑 extract_appearance.py 生成外观特征，否则外观输入会是全零"
        ),
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-length", type=int, default=48)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--no-pos-weight",
        action="store_true",
        help="关掉类别加权。默认开启，因为正样本比例极不均衡",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help=(
            "验证集 F1 连续多少轮没提升就停止。实测相邻轮次之间会有 ±0.02 的"
            "正常波动，耐心值设太小会在还没收敛时就提前截断"
        ),
    )
    parser.add_argument(
        "--init-from",
        default=None,
        help="从旧权重初始化（只加载形状匹配的层，分类头重新随机）",
    )
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=0,
        help="前 N 轮冻结骨架和外观骨干，只训练分类头",
    )
    parser.add_argument(
        "--max-pos-weight",
        type=float,
        default=0.0,
        help="类别权重上限；0 表示不限制。建议六类模型设 6",
    )

    args = parser.parse_args()

    columns = tuple(item.strip() for item in args.labels_cols.split(",") if item.strip())
    unknown = [c for c in columns if c not in LABEL_COLUMNS]
    if unknown or not columns:
        print(f"无效的 --labels-cols：{unknown or args.labels_cols}")
        return 1

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    os.makedirs(args.out, exist_ok=True)

    clips = read_label_csv(args.labels, root=args.features_root, columns=columns)
    if not clips:
        print("标注文件里没有任何片段")
        return 1

    if args.split_mode == "balanced":
        train_clips, val_clips = split_by_group_balanced(
            clips, columns=columns, val_ratio=args.val_ratio, seed=args.seed
        )
    else:
        train_clips, val_clips = split_by_group(
            clips, val_ratio=args.val_ratio, seed=args.seed
        )
    if not train_clips or not val_clips:
        print("训练集或验证集为空，检查一下 --val-ratio 和数据量")
        return 1

    print("=" * 68)
    print(f"训练标签列：{'、'.join(LABEL_NAMES_ZH[c] for c in columns)}")
    print("训练集分布：")
    print(summarize(train_clips, columns))
    print("验证集分布：")
    print(summarize(val_clips, columns))
    print("=" * 68)

    train_loader = make_loader(
        train_clips, args.clip_length, args.batch_size, args.workers, True, True
    )
    val_loader = make_loader(
        val_clips, args.clip_length, args.batch_size, args.workers, False, False
    )

    model = build_model(
        args.model, num_joints=17, in_channels=3, num_classes=len(columns),
        dropout=args.dropout,
    ).to(device)

    if args.init_from:
        blob = torch.load(args.init_from, map_location="cpu", weights_only=False)
        source = blob.get("state_dict", {})
        target = model.state_dict()
        compatible = {
            key: value
            for key, value in source.items()
            if key in target and target[key].shape == value.shape
        }
        skipped = [key for key in source if key not in compatible]
        model.load_state_dict(compatible, strict=False)
        print(
            f"迁移初始化：加载 {len(compatible)}/{len(source)} 层"
            f"（跳过 {len(skipped)} 层，主要是分类头）"
        )

    def set_backbone_trainable(trainable: bool) -> None:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = trainable or name.startswith("head.")

    if args.freeze_epochs > 0:
        set_backbone_trainable(False)
        print(f"前 {args.freeze_epochs} 轮冻结骨干，只训练分类头")
    parameters = sum(p.numel() for p in model.parameters())
    print(f"模型 {args.model}，参数量 {parameters/1e6:.2f}M，设备 {device}，混合精度 {amp}")

    if args.no_pos_weight:
        pos_weight = None
    else:
        pos_weight = compute_pos_weight(train_clips, num_labels=len(columns)).to(device)
        if args.max_pos_weight > 0:
            pos_weight = torch.clamp(pos_weight, max=args.max_pos_weight)
        print(
            "类别权重 (负/正)：",
            [
                f"{LABEL_NAMES_ZH[c]}:{w:.1f}"
                for c, w in zip(columns, pos_weight.cpu().numpy())
            ],
        )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(args.out, "tb"))
    except Exception:  # noqa: BLE001
        writer = None

    best_score = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    last_report: dict = {}

    for epoch in range(1, args.epochs + 1):
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            set_backbone_trainable(True)
            print("冻结结束，解冻骨干继续微调")
        started = time.time()
        loss = train_one_epoch(
            model, train_loader, criterion, optimiser, scaler, device, amp,
            args.grad_clip,
            frozen_backbone=args.freeze_epochs > 0 and epoch <= args.freeze_epochs,
        )
        scheduler.step()

        probabilities, targets = predict(model, val_loader, device, amp)
        tuned_thresholds, _ = sweep_thresholds(probabilities, targets, columns)
        report = evaluate_at(probabilities, targets, tuned_thresholds, columns)
        report["loss"] = loss
        last_report = report

        if writer is not None:
            writer.add_scalar("train/loss", loss, epoch)
            writer.add_scalar("val/macro_f1", report["macro_f1"], epoch)
            for column in columns:
                writer.add_scalar(f"val/{column}_f1", report[column]["f1"], epoch)
                writer.add_scalar(f"val/{column}_recall", report[column]["recall"], epoch)
                writer.add_scalar(f"val/{column}_precision", report[column]["precision"], epoch)

        improved = report["macro_f1"] > best_score
        print(
            f"\n[epoch {epoch:3d}/{args.epochs}]  损失 {loss:.4f}  "
            f"验证宏F1 {report['macro_f1']:.4f}  "
            f"用时 {time.time()-started:.1f}s" + ("   <- 最佳" if improved else "")
        )
        print(format_report(report, columns))

        if improved:
            best_score = report["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": args.model,
                    "state_dict": model.state_dict(),
                    "thresholds": tuned_thresholds.tolist(),
                    "label_columns": list(columns),
                    "clip_length": args.clip_length,
                    "epoch": epoch,
                    "macro_f1": report["macro_f1"],
                    "report": report,
                },
                os.path.join(args.out, "best.pt"),
            )
        else:
            epochs_without_improvement += 1

        torch.save(
            {
                "model": args.model,
                "state_dict": model.state_dict(),
                "epoch": epoch,
            },
            os.path.join(args.out, "last.pt"),
        )

        if epochs_without_improvement >= args.patience:
            print(f"\n验证集 {args.patience} 轮没有提升，提前停止")
            break

    if writer is not None:
        writer.close()

    print("\n" + "=" * 68)
    print(f"训练结束。最佳轮次 {best_epoch}，验证宏平均 F1 = {best_score:.4f}")
    print("最后一轮的详细指标：")
    if last_report:
        print(format_report(last_report, columns))
    print(f"\n模型保存在 {os.path.join(args.out, 'best.pt')}")
    print(
        "该文件里已经带上了每类的最优阈值（thresholds 字段），"
        "接入实时推理时直接用它，不要用固定的 0.5。"
    )

    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(last_report, handle, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
