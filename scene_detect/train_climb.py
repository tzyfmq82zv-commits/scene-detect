"""训练 Normal / Wall crossing 二分类视频模型。

输入为整段视频均匀采样的 16 帧，ResNet18 提特征后做时序注意力池化。

前置：
    python adapters/prepare_climb.py
    python adapters/cache_climb_frames.py

用法：
    python train_climb.py
    python train_climb.py --epochs 20 --batch-size 8

输出：
    runs_climb/best.pt
    runs_climb/report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

MANIFEST = r"D:\SceneDetection\features\climb\labels_climb.csv"
CACHE_ROOT = r"D:\SceneDetection\features\climb\frames_k16"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs_climb")
CLASS_NAMES = ["normal", "wall_crossing"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cache_path(row: dict, cache_root: str) -> str:
    stem = os.path.splitext(os.path.basename(row["path"]))[0]
    return os.path.join(cache_root, row["split"], row["class_name"], f"{stem}.npy")


class ClimbDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        cache_root: str,
        frames: int,
        image_size: int,
        train: bool,
    ) -> None:
        self.rows = rows
        self.cache_root = cache_root
        self.frames = frames
        self.image_size = image_size
        self.train = train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        frames = np.load(cache_path(row, self.cache_root))
        if frames.shape[1] != self.image_size:
            frames = np.stack(
                [
                    cv2.resize(
                        frame,
                        (self.image_size, self.image_size),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    for frame in frames
                ]
            )
        if frames.shape[0] != self.frames:
            picks = np.linspace(0, frames.shape[0] - 1, self.frames).astype(int)
            frames = frames[picks]
        frames = frames.astype(np.float32)
        if self.train:
            if random.random() < 0.5:
                frames = frames[:, :, ::-1]
            factor = random.uniform(0.82, 1.18)
            frames = np.clip(frames * factor, 0, 255)
            if random.random() < 0.3:
                frames = np.clip(frames + random.randint(-10, 10), 0, 255)
        frames = frames / 255.0
        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(frames.transpose(0, 3, 1, 2)).float()
        return tensor, int(row["label"])


class ClimbClassifier(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.35) -> None:
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, count, channels, height, width = frames.shape
        features = self.backbone(frames.reshape(batch * count, channels, height, width))
        features = features.reshape(batch, count, -1)
        weights = torch.softmax(self.attention(features).squeeze(-1), dim=1)
        pooled = (features * weights.unsqueeze(-1)).sum(dim=1)
        maximum = features.max(dim=1).values
        return self.head(torch.cat([pooled, maximum], dim=1))


def metrics(targets: np.ndarray, predictions: np.ndarray) -> dict:
    result = {}
    for index, name in enumerate(CLASS_NAMES):
        tp = int(((predictions == index) & (targets == index)).sum())
        fp = int(((predictions == index) & (targets != index)).sum())
        fn = int(((predictions != index) & (targets == index)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        result[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "positives": int((targets == index).sum()),
        }
    result["accuracy"] = float((predictions == targets).mean())
    result["macro_f1"] = float(np.mean([result[name]["f1"] for name in CLASS_NAMES]))
    result["confusion"] = np.zeros((2, 2), dtype=int).tolist()
    for target, predicted in zip(targets, predictions):
        result["confusion"][int(target)][int(predicted)] += 1
    return result


def evaluate(model, loader, criterion, device, amp) -> tuple[dict, float]:
    model.eval()
    targets, predictions = [], []
    loss_sum = 0.0
    seen = 0
    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(frames)
                loss = criterion(logits, labels)
            loss_sum += loss.item() * labels.size(0)
            seen += labels.size(0)
            predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            targets.extend(labels.cpu().numpy().tolist())
    return metrics(np.asarray(targets), np.asarray(predictions)), loss_sum / max(seen, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="训练攀爬二分类模型")
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--cache-root", default=CACHE_ROOT)
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--init-from",
        default=None,
        help="从已有攀爬权重初始化，例如 runs_climb/best.pt",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    if not os.path.exists(args.manifest):
        print(f"找不到清单：{args.manifest}")
        return 1
    with open(args.manifest, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    hard_count = sum(str(row.get("hard_negative", "0")) == "1" for row in rows)
    print(
        f"清单：{args.manifest}  总行数 {len(rows)}  "
        f"难负样本行 {hard_count}"
    )
    if hard_count and os.path.abspath(args.out) == os.path.abspath(OUT_DIR):
        print(
            "警告：这次训练包含难负样本，但输出仍是默认 runs_climb，"
            "会覆盖原基线；建议加 --out runs_climb_hn"
        )
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    if not train_rows or not val_rows:
        print("训练集或验证集为空")
        return 1
    missing = [
        row["path"]
        for row in train_rows + val_rows
        if not os.path.exists(cache_path(row, args.cache_root))
    ]
    if missing:
        print(f"缺少 {len(missing)} 个帧缓存，先运行 cache_climb_frames.py")
        print("示例：", missing[0])
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    train_loader = DataLoader(
        ClimbDataset(train_rows, args.cache_root, args.frames, args.image_size, True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        ClimbDataset(val_rows, args.cache_root, args.frames, args.image_size, False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    counts = Counter(int(row["label"]) for row in train_rows)
    weights = torch.tensor(
        [len(train_rows) / max(counts.get(index, 0), 1) for index in range(2)],
        dtype=torch.float32,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    model = ClimbClassifier(dropout=args.dropout).to(device)
    if args.init_from:
        source = torch.load(args.init_from, map_location="cpu", weights_only=False)
        target = model.state_dict()
        compatible = {
            key: value
            for key, value in source["state_dict"].items()
            if key in target and target[key].shape == value.shape
        }
        model.load_state_dict(compatible, strict=False)
        print(f"迁移初始化：加载 {len(compatible)}/{len(source['state_dict'])} 层")
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.02
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    os.makedirs(args.out, exist_ok=True)
    best_score = -1.0
    best_epoch = -1
    best_report = {}
    stale = 0

    for epoch in range(1, args.epochs + 1):
        if epoch == 1 and args.freeze_epochs > 0:
            for name, parameter in model.named_parameters():
                parameter.requires_grad = not name.startswith("backbone.")
            model.backbone.eval()
            print(f"前 {args.freeze_epochs} 轮冻结 ResNet18 骨干")
        if epoch == args.freeze_epochs + 1:
            for parameter in model.parameters():
                parameter.requires_grad = True
            print("解冻骨干，开始全模型微调")

        model.train()
        if epoch <= args.freeze_epochs:
            model.backbone.eval()
        started = time.time()
        loss_sum = 0.0
        seen = 0
        for frames, labels in train_loader:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(frames)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()
            loss_sum += loss.item() * labels.size(0)
            seen += labels.size(0)
        scheduler.step()

        report, val_loss = evaluate(model, val_loader, criterion, device, amp)
        score = report["macro_f1"]
        marker = ""
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_report = report
            stale = 0
            torch.save(
                {
                    "model": "resnet18_attention",
                    "state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "frames": args.frames,
                    "image_size": args.image_size,
                    "dropout": args.dropout,
                    "epoch": epoch,
                    "macro_f1": score,
                    "report": report,
                },
                os.path.join(args.out, "best.pt"),
            )
            marker = "  <- 最佳"
        else:
            stale += 1
        print(
            f"[{epoch:02d}/{args.epochs}] loss {loss_sum/max(seen,1):.4f} "
            f"val {val_loss:.4f} macroF1 {score:.4f}{marker} "
            f"用时 {time.time()-started:.1f}s"
        )
        for name in CLASS_NAMES:
            item = report[name]
            print(
                f"    {name:<14} P {item['precision']:.3f} "
                f"R {item['recall']:.3f} F1 {item['f1']:.3f}"
            )
        if stale >= args.patience:
            print(f"验证集 {args.patience} 轮未提升，提前停止")
            break

    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_macro_f1": best_score,
                "class_names": CLASS_NAMES,
                "report": best_report,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n最佳轮次 {best_epoch}，宏 F1 {best_score:.4f}")
    print(f"模型：{os.path.join(args.out, 'best.pt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
