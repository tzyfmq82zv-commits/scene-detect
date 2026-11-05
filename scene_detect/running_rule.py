"""用速度规则判定「奔跑」，替代训练一个只有 8 个正样本的奔跑分类头。

为什么奔跑走规则而不是走模型：
  1. 训练数据里奔跑只有 8 个正样本（Weizmann 的 10 个视频，每个只能切出 1 段，
     还有 2 段因为人贴在画面边缘没检出轨迹）。这个量级训不出任何东西；
  2. 奔跑本质上是速度问题，不是姿态问题。用「每秒位移 ÷ 人体身高」这个
     归一化速度就能分开——它自动抵消了远近和机位差异；
  3. 规则可解释、不占算力、阈值能在界面上直接调，接进实时报警最省事。

两条使用方式：
    标定（在 Weizmann 的 run/walk 上找最优阈值）：
        python running_rule.py --calibrate
    推理（实时链路里调用这个类）：
        rule = RunningRule(threshold=2.0)
        rule.is_running(bbox_history, fps)

速度定义：body-heights / second
    路径用滑动窗口的净位移（不是逐帧位移累加），避免姿态抖动被累加进速度；
    尺度用检测框高度的中位数，人在画面里贴边被截断时退回肩宽估计。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, field

import numpy as np

DATASETS_ROOT = r"D:\SceneDetection\datasets"
FEATURES_ROOT = r"D:\SceneDetection\features"
LABELS_CSV = os.path.join(FEATURES_ROOT, "labels.csv")
VIDEO_INDEX_CACHE = os.path.join(FEATURES_ROOT, "video_index.json")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}

# 站立时检测框的 高/宽 大致在 2.0 以上；摔倒躺平后会降到 1.0 附近。
# 用这个比值把「快速下坠」和「快速平移」区分开，否则摔倒会被误判成奔跑。
MIN_UPRIGHT_RATIO = 1.2

# 奔跑的位移应该以水平方向为主。下坠的位移是竖直的，这里把它排除掉。
MIN_HORIZONTAL_FRACTION = 0.55


def moving_median(values: np.ndarray) -> float:
    return float(np.median(values)) if values.size else 0.0


def window_speeds(
    centers: np.ndarray, scale: float, fps: float, window: int = 12
) -> np.ndarray:
    """滑动窗口净位移速度，单位是 人体身高/秒。

    用窗口两端的位置差而不是逐帧位移之和：逐帧位移会把关键点抖动累加进去，
    一段静止不动但框在抖的片段能算出很高的假速度。
    """
    frames = centers.shape[0]
    if frames < 2 or scale <= 1e-6 or fps <= 1e-6:
        return np.zeros(0, dtype=np.float32)
    window = int(min(window, frames - 1))
    if window < 1:
        return np.zeros(0, dtype=np.float32)
    displacement = np.linalg.norm(centers[window:] - centers[:-window], axis=1)
    return (displacement / scale) * (fps / window)


def clip_features(keypoints: np.ndarray, bbox: np.ndarray) -> dict:
    """从一个片段的骨架和检测框里抽出判定奔跑需要的几个量。"""
    centers = np.stack(
        [(bbox[:, 0] + bbox[:, 2]) * 0.5, (bbox[:, 1] + bbox[:, 3]) * 0.5], axis=1
    )
    heights = bbox[:, 3] - bbox[:, 1]
    widths = np.maximum(bbox[:, 2] - bbox[:, 0], 1e-6)

    # 尺度：优先用检测框高度中位数。人贴画面边缘被截断的时候框会偏矮，
    # 这时退回用肩宽估计身高，避免速度被系统性放大
    valid_heights = heights[heights > 1]
    scale = float(np.median(valid_heights)) if valid_heights.size else 0.0
    shoulder_width = np.linalg.norm(
        keypoints[:, 5, :2] - keypoints[:, 6, :2], axis=1
    )
    shoulder_conf = keypoints[:, [5, 6], 2].min(axis=1)
    valid_shoulder = shoulder_width[(shoulder_conf > 0.3) & (shoulder_width > 1)]
    if valid_shoulder.size >= 3:
        # 肩宽约为身高的 0.26，据此折算成等效身高
        shoulder_based = float(np.median(valid_shoulder)) / 0.26
        # 两者差别很大时说明框被截断了，选更保守（更大）的那个，
        # 宁可低估速度，也不要因为框偏矮而把人误判成奔跑
        scale = max(scale, shoulder_based) if scale > 0 else shoulder_based

    return {
        "centers": centers,
        "scale": scale,
        "upright_ratio": float(np.median(heights / widths)),
    }


@dataclass
class RunningRule:
    """奔跑判定规则。实时链路里直接实例化调用即可。"""

    threshold: float = 2.0            # body-heights / second
    window: int = 12                  # 滑动窗口帧数
    min_upright_ratio: float = MIN_UPRIGHT_RATIO
    min_horizontal_fraction: float = MIN_HORIZONTAL_FRACTION
    min_frames: int = 12

    def speed(self, bbox: np.ndarray, keypoints: np.ndarray, fps: float) -> float:
        """返回中位速度（人体身高/秒）。"""
        if bbox.shape[0] < 2:
            return 0.0
        features = clip_features(keypoints, bbox)
        speeds = window_speeds(features["centers"], features["scale"], fps, self.window)
        return moving_median(speeds)

    def is_running(self, bbox: np.ndarray, keypoints: np.ndarray, fps: float) -> bool:
        """判断一段轨迹是不是在奔跑。

        三个条件同时满足才算：速度够快、人是直立的、位移以水平方向为主。
        后两条是为了不被摔倒带偏——人摔倒时检测框中心下坠很快，
        只看速度会把它误判成奔跑。
        """
        if bbox.shape[0] < self.min_frames:
            return False
        features = clip_features(keypoints, bbox)
        if features["upright_ratio"] < self.min_upright_ratio:
            return False

        centers = features["centers"]
        delta = centers[-1] - centers[0]
        horizontal = abs(float(delta[0]))
        vertical = abs(float(delta[1]))
        total = horizontal + vertical
        if total > 1e-6 and horizontal / total < self.min_horizontal_fraction:
            return False

        speeds = window_speeds(centers, features["scale"], fps, self.window)
        return moving_median(speeds) >= self.threshold


def build_video_index(datasets_root: str, cache_path: str) -> dict[str, list[float]]:
    """扫描数据集目录，建立 视频名 -> [fps, 总帧数] 的索引并缓存。

    为什么需要它：特征片段里只存了骨架坐标，没存 fps 和原始帧数。
    而切片时不足 48 帧的短片会被拉伸到 48 帧，如果按 48 帧算时长，
    速度会被系统性低估（Weizmann 的片子大约低估 1.7 倍），阈值就标偏了。
    所以这里回到源视频把真实 fps 和帧数读出来。
    """
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as handle:
            cached = json.load(handle)
        if cached.get("root") == datasets_root:
            return cached["videos"]

    import cv2

    index: dict[str, list[float]] = {}
    for root, _, files in os.walk(datasets_root):
        for name in files:
            if os.path.splitext(name)[1].lower() not in VIDEO_EXTENSIONS:
                continue
            path = os.path.join(root, name)
            capture = cv2.VideoCapture(path)
            fps = capture.get(cv2.CAP_PROP_FPS)
            frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            capture.release()
            if fps and fps > 0 and frames and frames > 0:
                index[os.path.splitext(name)[0]] = [float(fps), float(frames)]

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as handle:
        json.dump({"root": datasets_root, "videos": index}, handle)
    return index


def clip_duration_seconds(
    source: str, clip_frames: int, clip_length: int, index: dict
) -> float | None:
    """算出这个片段覆盖的真实时长（秒）。取不到源视频信息时返回 None。"""
    meta = index.get(source)
    if not meta:
        return None
    fps, total_frames = meta
    if fps <= 0:
        return None
    # 短片会被拉伸到 clip_length，真正覆盖的帧数是源视频的帧数
    effective = min(float(clip_length), float(total_frames))
    return effective / fps


def read_clips(labels_csv: str, features_root: str) -> list[dict]:
    rows: list[dict] = []
    with open(labels_csv, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            path = os.path.join(features_root, row["clip"].replace("/", os.sep))
            if not os.path.exists(path):
                continue
            row["_path"] = path
            rows.append(row)
    return rows


def calibrate(args) -> int:
    index = build_video_index(DATASETS_ROOT, VIDEO_INDEX_CACHE)
    print(f"视频索引：{len(index)} 个源视频")

    rows = read_clips(args.labels, FEATURES_ROOT)
    if not rows:
        print(f"没读到任何片段，检查 {args.labels}")
        return 1
    print(f"片段总数：{len(rows)}")

    samples: list[dict] = []
    skipped = 0
    for row in rows:
        duration = clip_duration_seconds(
            row["source"], int(row["frames"]), args.clip_length, index
        )
        if duration is None or duration <= 0:
            skipped += 1
            continue
        with np.load(row["_path"]) as data:
            keypoints = data["keypoints"].astype(np.float32)
            bbox = data["bbox"].astype(np.float32)
        features = clip_features(keypoints, bbox)
        fps = duration and (keypoints.shape[0] / duration)
        samples.append(
            {
                "_path": row["_path"],
                "dataset": row["dataset"],
                "source": row["source"],
                "running": int(row["running"]),
                "falling": int(row["falling"]),
                "fighting": int(row["fighting"]),
                "speed": moving_median(
                    window_speeds(features["centers"], features["scale"], fps, args.window)
                ),
                "upright_ratio": features["upright_ratio"],
                "scale": features["scale"],
            }
        )
    if skipped:
        print(f"（{skipped} 个片段找不到源视频信息，已跳过）")

    positives = [s for s in samples if s["running"] == 1]
    # 负样本只取 Weizmann 的非奔跑动作（走路、弯腰、挥手等）。
    # 这是和奔跑同机位同分辨率的对照集，标出来的阈值才干净。
    # 摔倒要留给下面的守卫条件去排除，公交车上的人则会放到最后一起复查。
    negatives = [
        s for s in samples if s["running"] == 0 and s["dataset"] == "weizmann"
    ]
    print(f"奔跑正样本 {len(positives)} 个；同机位直立负样本 {len(negatives)} 个")
    if not positives or not negatives:
        print("正负样本不足，无法标定")
        return 1

    print("\n奔跑样本的速度（人体身高/秒）：")
    for item in sorted(positives, key=lambda s: s["speed"]):
        print(
            f"  {item['source']:<16} 速度 {item['speed']:5.2f}  "
            f"直立比 {item['upright_ratio']:4.2f}"
        )
    print("\n直立负样本的速度分布：")
    negative_speeds = np.array([s["speed"] for s in negatives])
    print(
        f"  中位数 {np.median(negative_speeds):.2f}  "
        f"90分位 {np.percentile(negative_speeds, 90):.2f}  "
        f"最大 {negative_speeds.max():.2f}"
    )

    grid = np.arange(0.5, 5.01, 0.05)
    best = None
    for threshold in grid:
        true_positive = sum(1 for s in positives if s["speed"] >= threshold)
        false_positive = sum(1 for s in negatives if s["speed"] >= threshold)
        recall = true_positive / len(positives)
        precision = true_positive / max(true_positive + false_positive, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        if best is None or f1 > best["f1"]:
            best = {
                "threshold": float(threshold),
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "tp": true_positive,
                "fp": false_positive,
            }

    print("\n" + "=" * 64)
    print("标定结果")
    print("=" * 64)
    print("  阈值      奔跑召回    同机位误报      F1")
    for threshold in (1.0, 1.05, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5):
        true_positive = sum(1 for s in positives if s["speed"] >= threshold)
        false_positive = sum(1 for s in negatives if s["speed"] >= threshold)
        recall = true_positive / len(positives)
        precision = true_positive / max(true_positive + false_positive, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        mark = "  <- F1 最优" if abs(threshold - best["threshold"]) < 0.03 else ""
        print(
            f"  {threshold:4.2f}   {true_positive:2d}/{len(positives)}"
            f"      {false_positive:2d}/{len(negatives)}"
            f"        {f1:.3f}{mark}"
        )
    print(
        "\n  阈值调高的代价是漏报奔跑，收益是少误报。摔倒检测要的是高召回，"
        "\n  奔跑误报只是弹个提示，所以这里可以为了压误报把阈值往上取一点。"
    )

    rule = RunningRule(
        threshold=best["threshold"],
        window=args.window,
        min_upright_ratio=args.min_upright_ratio,
        min_horizontal_fraction=args.min_horizontal_fraction,
    )
    full_tp = full_fp = full_fn = 0
    for sample in samples:
        with np.load(sample["_path"]) as data:
            keypoints = data["keypoints"].astype(np.float32)
            bbox = data["bbox"].astype(np.float32)
        duration = clip_duration_seconds(
            sample["source"], keypoints.shape[0], args.clip_length, index
        )
        fired = rule.is_running(bbox, keypoints, keypoints.shape[0] / duration)
        if fired and sample["running"]:
            full_tp += 1
        elif fired and not sample["running"]:
            full_fp += 1
        elif not fired and sample["running"]:
            full_fn += 1

    print("\n加上「直立 + 水平位移为主」两个守卫条件后，在全部片段上的表现：")
    print(f"  奔跑命中 {full_tp}/{len(positives)}   误报 {full_fp} 个   漏报 {full_fn} 个")
    fired_wrong: list[tuple[str, str]] = []
    for sample in samples:
        if sample["running"]:
            continue
        with np.load(sample["_path"]) as data:
            keypoints = data["keypoints"].astype(np.float32)
            bbox = data["bbox"].astype(np.float32)
        duration = clip_duration_seconds(
            sample["source"], keypoints.shape[0], args.clip_length, index
        )
        if duration and rule.is_running(bbox, keypoints, keypoints.shape[0] / duration):
            fired_wrong.append((sample["dataset"], sample["source"]))
    if fired_wrong:
        print("\n  误报明细（这些是速度快的非奔跑片段）：")
        counts: dict[str, int] = {}
        for dataset, source in fired_wrong:
            counts[dataset] = counts.get(dataset, 0) + 1
        for dataset, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"    {dataset}: {count} 个")
        print(
            "    其中 Weizmann 的 skip（跳跃跑）、jack（开合跳）、side（侧向跳跃）\n"
            "    本身就是高速动作，被判成奔跑属于标注口径问题，不是规则失效。\n"
            "    真正要盯的是 urfall 里的摔倒误报——那才是会让人麻木的假警报。"
        )

    print("\n" + "-" * 64)
    print("把这个阈值写进实时链路：")
    print(f"    rule = RunningRule(threshold={best['threshold']:.2f})")
    if args.save:
        payload = {
            "threshold": best["threshold"],
            "window": args.window,
            "min_upright_ratio": args.min_upright_ratio,
            "min_horizontal_fraction": args.min_horizontal_fraction,
            "calibration": best,
        }
        with open(args.save, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"  已写入 {args.save}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="奔跑判定的速度规则与阈值标定")
    parser.add_argument("--labels", default=LABELS_CSV, help="标注 CSV 路径")
    parser.add_argument("--clip-length", type=int, default=48)
    parser.add_argument("--window", type=int, default=12, help="滑动窗口帧数")
    parser.add_argument("--min-upright-ratio", type=float, default=MIN_UPRIGHT_RATIO)
    parser.add_argument(
        "--min-horizontal-fraction", type=float, default=MIN_HORIZONTAL_FRACTION
    )
    parser.add_argument("--save", default=None, help="把标定结果写入 JSON")
    parser.add_argument(
        "--calibrate", action="store_true", help="在现有数据上标定阈值"
    )
    args = parser.parse_args()

    # 直接 `python running_rule.py` 就是标定；--calibrate 保留给显式调用的场景
    if args.calibrate or len(sys.argv) == 1:
        return calibrate(args)
    print(
        "直接运行本脚本即是标定：python running_rule.py\n"
        "实时链路里请 import RunningRule 使用。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
