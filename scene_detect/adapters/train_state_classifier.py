"""训练坐/走/踢/其他四分类状态分类器。

输入不是原始视频，而是六类主模型融合后的 384 维特征。
它只负责界面里的坐/走/踢状态，不参与摔倒/打架报警。

用法：
    python adapters/train_state_classifier.py

输出：
    runs_six_transfer/state_classifier.joblib
"""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from dataset import read_label_csv  # noqa: E402
from model import build_model  # noqa: E402
from train import make_loader  # noqa: E402

CHECKPOINT = os.path.join(CODE_ROOT, "runs_six_transfer", "best.pt")
OUTPUT = os.path.join(CODE_ROOT, "runs_six_transfer", "state_classifier.joblib")
LABELS = r"D:\SceneDetection\features\labels_six.csv"
FEATURES_ROOT = r"D:\SceneDetection\features"
STATE_COLUMNS = ["sitting", "walking", "kicking"]
CLASS_NAMES = STATE_COLUMNS + ["other"]


def macro_f1(target: np.ndarray, prediction: np.ndarray, classes: int) -> float:
    values = []
    for index in range(classes):
        tp = int(((prediction == index) & (target == index)).sum())
        fp = int(((prediction == index) & (target != index)).sum())
        fn = int(((prediction != index) & (target == index)).sum())
        denominator = 2 * tp + fp + fn
        values.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(values))


def main() -> int:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    columns = tuple(checkpoint["label_columns"])
    clips = read_label_csv(LABELS, root=FEATURES_ROOT, columns=columns)

    selected = []
    targets = []
    for clip in clips:
        if "hmdb51" not in clip.path.replace("\\", "/"):
            continue
        values = [int(clip.labels[columns.index(name)] > 0.5) for name in STATE_COLUMNS]
        if sum(values) != 1:
            continue
        selected.append(clip)
        targets.append(values.index(1))

    other = []
    for clip in clips:
        values = [int(clip.labels[columns.index(name)] > 0.5) for name in STATE_COLUMNS]
        if sum(values) == 0:
            other.append(clip)
    sampler = np.random.default_rng(7)
    sampler.shuffle(other)
    other = other[: min(6000, len(other))]
    selected.extend(other)
    targets.extend([3] * len(other))
    print(
        "训练样本：",
        {name: targets.count(index) for index, name in enumerate(CLASS_NAMES)},
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        "dual", num_joints=17, in_channels=3, num_classes=len(columns)
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    features = []
    loader = make_loader(selected, 32, 64, 0, shuffle=False, augment=False)
    with torch.no_grad():
        for primary, _, _, appearance, _ in loader:
            primary = primary.to(device)
            appearance = appearance.to(device)
            skeleton = model.skeleton_backbone(primary)
            visual = model.appearance_encoder(appearance)
            features.append(torch.cat([skeleton, visual], dim=1).cpu().numpy())
    features = np.concatenate(features)
    targets = np.asarray(targets)

    groups = np.asarray([clip.group for clip in selected])
    unique = sorted(set(groups))
    rng = np.random.default_rng(42)
    rng.shuffle(unique)
    val_groups = set(unique[: max(1, int(len(unique) * 0.2))])
    val_mask = np.asarray([group in val_groups for group in groups])
    train_mask = ~val_mask

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced"),
    )
    classifier.fit(features[train_mask], targets[train_mask])
    probabilities = classifier.predict_proba(features[val_mask])
    y_val = targets[val_mask]

    print("四分类基线")
    prediction = classifier.predict(features[val_mask])
    print("accuracy", round(accuracy_score(y_val, prediction), 4))
    print(
        classification_report(
            y_val, prediction, target_names=CLASS_NAMES, digits=3, zero_division=0
        )
    )

    best = None
    for threshold in np.arange(0.30, 0.76, 0.05):
        prediction = np.argmax(probabilities, axis=1).astype(np.int64)
        prediction[probabilities.max(axis=1) < threshold] = 3
        score = macro_f1(y_val, prediction, 4)
        if best is None or score > best[0]:
            best = (score, float(threshold), prediction)
    score, threshold, prediction = best
    print(f"最佳置信阈值 {threshold:.2f}，宏 F1 {score:.3f}")
    print(
        classification_report(
            y_val, prediction, target_names=CLASS_NAMES, digits=3, zero_division=0
        )
    )
    print("confusion\n", confusion_matrix(y_val, prediction))

    payload = {
        "classifier": classifier,
        "class_names": CLASS_NAMES,
        "confidence_threshold": threshold,
        "source_checkpoint": CHECKPOINT,
        "source_columns": list(columns),
        "validation_macro_f1": score,
        "validation_accuracy": float(accuracy_score(y_val, prediction)),
    }
    joblib.dump(payload, OUTPUT)
    print(f"已保存：{OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
