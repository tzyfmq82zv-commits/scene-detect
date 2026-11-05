"""按统一清单抽取 HMDB51 / GMDCSA24 的骨架片段，并生成新标注 CSV。

它复用现有的 feature_extract.extract_from_video，不复制流水线逻辑。

标签规则：
    HMDB51     整段视频一个动作。sit→sitting，walk→walking，
               kick/kick_ball→kicking，run→running，fall_floor→falling。
    GMDCSA24   Fall 视频只把「窗口与摔倒过程相交」的片段标 falling=1，
               摔倒开始前的片段标 0，落地后再保留 post_fall_seconds；
               ADL 视频全标 0。这样不会重犯整段标成正样本的提前报警问题。

用法：
    python adapters/extract_new.py --limit 10          # 先试跑
    python adapters/extract_new.py --clip-length 32
    python adapters/extract_new.py --datasets gmdcsa24
    python adapters/extract_new.py --force             # 重抽已有片段

前置条件：
    1. python adapters/build_manifest.py
    2. python adapters/detect_fall_onsets.py --finalize

输出：
    D:\\SceneDetection\\features\\new_datasets\\clips\\...
    D:\\SceneDetection\\features\\new_datasets\\labels_new.csv
    D:\\SceneDetection\\features\\new_datasets\\manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE)
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from feature_extract import extract_from_video  # noqa: E402

FEATURES_ROOT = r"D:\SceneDetection\features"
DEFAULT_OUT = os.path.join(FEATURES_ROOT, "new_datasets")
LABEL_COLUMNS = (
    "running",
    "falling",
    "fighting",
    "sitting",
    "walking",
    "kicking",
)


def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_existing_labels(path: str) -> list[dict]:
    """重跑或只跑一部分数据集时，保留没被本次处理覆盖的旧行。"""
    if not os.path.exists(path):
        return []
    try:
        return read_csv(path)
    except (OSError, csv.Error):
        return []


def load_onsets(path: str) -> dict[tuple[str, str], dict]:
    """最终标注按 (subject, stem) 建索引。"""
    result: dict[tuple[str, str], dict] = {}
    if not path or not os.path.exists(path):
        return result
    for row in read_csv(path):
        try:
            onset = int(float(row["onset_frame"]))
        except (KeyError, TypeError, ValueError):
            continue
        result[(row["subject"].strip(), row["stem"].strip())] = {
            "onset": onset,
            "landed": row.get("landed_frame", ""),
            "fps": row.get("fps", ""),
            "frames": row.get("frames", ""),
        }
    return result


def window_intersects(
    start: int, length: int, onset: int, fall_end: int
) -> bool:
    return start <= fall_end and start + length - 1 >= onset


def output_dir_for(out_root: str, row: dict) -> str:
    dataset = row["dataset"]
    if dataset == "hmdb51":
        return os.path.join(out_root, "clips", "hmdb51", row["action"])
    if dataset == "gmdcsa24":
        part = "Fall" if row["action"] == "fall" else "ADL"
        return os.path.join(out_root, "clips", "gmdcsa24", f"{row['subject']}_{part}")
    return os.path.join(out_root, "clips", dataset)


def clip_reference(path: str) -> str:
    """默认写相对 features 的路径；输出目录在别的盘时退回绝对路径。"""
    absolute = os.path.abspath(path)
    try:
        return os.path.relpath(absolute, FEATURES_ROOT).replace("\\", "/")
    except ValueError:
        return absolute.replace("\\", "/")


def remove_stale(out_dir: str, uid: str) -> int:
    """只删除以该视频 uid 开头的 npz，重跑时避免新旧片段混在一起。"""
    removed = 0
    if not os.path.isdir(out_dir):
        return 0
    for name in os.listdir(out_dir):
        if name.startswith(uid + "__") and name.lower().endswith(".npz"):
            try:
                os.remove(os.path.join(out_dir, name))
                removed += 1
            except OSError:
                pass
    return removed


def rename_with_uid(out_dir: str, uid: str, produced: list[dict]) -> None:
    """把 01__track... 改成 gmdcsa_S1__Fall_01__track...

    dataset.py 用文件名双下划线前的部分当 group，这样同一受试者
    的所有片段会整体落在训练集或验证集，不会跨集合泄漏。
    """
    for item in produced:
        current = item["clip"]
        suffix = current.split("__", 1)[1] if "__" in current else current
        new_name = f"{uid}__{suffix}"
        old_path = os.path.join(out_dir, current)
        new_path = os.path.join(out_dir, new_name)
        if old_path != new_path and os.path.exists(old_path):
            if os.path.exists(new_path):
                os.remove(new_path)
            os.replace(old_path, new_path)
        item["clip"] = new_name


def save_progress(path: str, progress: dict) -> None:
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(progress, handle, ensure_ascii=False, indent=1)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="抽取 HMDB51 / GMDCSA24 骨架片段")
    parser.add_argument("--videos", default=os.path.join(DEFAULT_OUT, "videos.csv"))
    parser.add_argument(
        "--onsets",
        default=os.path.join(DEFAULT_OUT, "fall_onsets", "fall_onsets_final.csv"),
    )
    parser.add_argument("--out-root", default=DEFAULT_OUT)
    parser.add_argument("--labels", default=None, help="默认 <out-root>/labels_new.csv")
    parser.add_argument("--datasets", default="hmdb51,gmdcsa24")
    parser.add_argument("--actions", default="", help="只跑指定动作，逗号分隔")
    parser.add_argument("--clip-length", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--min-track-length", type=int, default=8)
    parser.add_argument("--max-clips", type=int, default=8)
    parser.add_argument("--min-confidence", type=float, default=0.30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--post-fall-seconds", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=0, help="总共只跑前 N 段")
    parser.add_argument("--limit-per-action", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="忽略进度并重抽")
    args = parser.parse_args()

    if not os.path.exists(args.videos):
        print(f"找不到视频清单：{args.videos}\n先运行 python adapters/build_manifest.py")
        return 1

    videos = read_csv(args.videos)
    wanted_datasets = {
        item.strip().lower() for item in args.datasets.split(",") if item.strip()
    }
    wanted_actions = {
        item.strip().lower() for item in args.actions.split(",") if item.strip()
    }
    selected = [
        row
        for row in videos
        if row["dataset"].lower() in wanted_datasets
        and (not wanted_actions or row["action"].lower() in wanted_actions)
    ]

    if args.limit_per_action:
        limited: list[dict] = []
        counts: dict[tuple[str, str], int] = {}
        for row in selected:
            key = (row["dataset"], row["action"])
            if counts.get(key, 0) >= args.limit_per_action:
                continue
            counts[key] = counts.get(key, 0) + 1
            limited.append(row)
        selected = limited
    if args.limit:
        selected = selected[: args.limit]

    if not selected:
        print("筛选后没有视频")
        return 1

    onsets = load_onsets(args.onsets)
    if any(
        row["dataset"] == "gmdcsa24" and row["action"] == "fall" for row in selected
    ) and not onsets:
        print(
            f"找不到摔倒起始帧标注：{args.onsets}\n"
            "先跑 detect_fall_onsets.py 并 --finalize；否则无法按窗口打标签。"
        )
        return 1

    os.makedirs(args.out_root, exist_ok=True)
    labels_path = args.labels or os.path.join(args.out_root, "labels_new.csv")
    existing_rows = [] if args.force else read_existing_labels(labels_path)
    progress_path = os.path.join(args.out_root, "progress.json")
    progress: dict = {}
    if os.path.exists(progress_path) and not args.force:
        with open(progress_path, encoding="utf-8") as handle:
            progress = json.load(handle)

    os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.environ.get("TEMP", "."), "scene_detect_yolo"))
    from ultralytics import YOLO

    model = YOLO(args.model)
    all_rows: list[dict] = []
    processed_uids: set[str] = set()
    skipped_fall: list[str] = []
    started = time.time()
    done = 0

    for index, row in enumerate(selected, start=1):
        uid = row["video_uid"]
        if not args.force and uid in progress:
            continue

        onset = None
        fall_end = None
        if row["dataset"] == "gmdcsa24" and row["action"] == "fall":
            item = onsets.get((row["subject"], row["source"]))
            if item is None:
                skipped_fall.append(uid)
                continue
            onset = item["onset"]
            try:
                fps = float(row["fps"])
            except (TypeError, ValueError):
                fps = 30.0
            landed = item.get("landed")
            try:
                landed_frame = int(float(landed)) if landed not in ("", None) else onset
            except (TypeError, ValueError):
                landed_frame = onset
            fall_end = min(
                int(row["frames"]) - 1,
                max(onset, landed_frame) + int(args.post_fall_seconds * fps),
            )

        out_dir = output_dir_for(args.out_root, row)
        os.makedirs(out_dir, exist_ok=True)
        if args.force:
            remove_stale(out_dir, uid)

        produced = extract_from_video(
            video_path=row["path"],
            out_dir=out_dir,
            model=model,
            tracker=args.tracker,
            clip_length=args.clip_length,
            stride=args.stride,
            min_confidence=args.min_confidence,
            min_track_length=args.min_track_length,
            max_clips=args.max_clips,
            imgsz=args.imgsz,
            save_vis=False,
        )
        rename_with_uid(out_dir, uid, produced)

        base_labels = {
            column: int(float(row.get(f"label_{column}", 0) or 0))
            for column in LABEL_COLUMNS
        }
        positive = 0
        for item in produced:
            labels = dict(base_labels)
            fall_start = ""
            if row["dataset"] == "gmdcsa24" and row["action"] == "fall":
                labels = {column: 0 for column in LABEL_COLUMNS}
                start = int(item["start_frame"])
                length = int(item["frames"])
                if window_intersects(start, length, onset, fall_end):
                    labels["falling"] = 1
                    fall_start = onset
            if labels.get("falling"):
                positive += 1
            all_rows.append(
                {
                    "clip": clip_reference(os.path.join(out_dir, item["clip"])),
                    "dataset": row["dataset"],
                    "source": row["source"],
                    "track_id": item["track_id"],
                    "start_frame": item["start_frame"],
                    "frames": item["frames"],
                    **labels,
                    "fall_start_frame": fall_start,
                    "group": uid.split("__", 1)[0],
                    "action": row["action"],
                    "video_uid": uid,
                }
            )

        progress[uid] = {
            "clips": len(produced),
            "positive_falling": positive,
            "seconds": round(float(row["duration_sec"]), 2),
        }
        processed_uids.add(uid)
        save_progress(progress_path, progress)
        done += 1

        if index % 20 == 0 or index == len(selected):
            elapsed = time.time() - started
            rate = done / max(elapsed, 1e-6)
            print(
                f"  {index}/{len(selected)} 已处理 {done} 个视频  "
                f"{rate:.1f} 视频/秒  片段 {len(all_rows)}"
            )

    # 只替换本次真正重抽的视频，其他数据集的旧行原样保留。
    final_rows = [
        row for row in existing_rows if row.get("video_uid") not in processed_uids
    ] + all_rows
    fieldnames = [
        "clip",
        "dataset",
        "source",
        "track_id",
        "start_frame",
        "frames",
        *LABEL_COLUMNS,
        "fall_start_frame",
        "group",
        "action",
        "video_uid",
    ]
    with open(labels_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final_rows)

    manifest_path = os.path.join(args.out_root, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final_rows)

    print("\n" + "=" * 68)
    print(
        f"本次处理 {done} 个视频，产出 {len(all_rows)} 个片段，"
        f"清单共 {len(final_rows)} 个片段，"
        f"用时 {(time.time()-started)/60:.1f} 分钟"
    )
    if skipped_fall:
        print(f"!! {len(skipped_fall)} 段摔倒视频没有起始帧标注，已跳过：")
        for uid in skipped_fall[:10]:
            print("   ", uid)

    if final_rows:
        print("\n标签分布（整个 labels_new.csv）：")
        for column in LABEL_COLUMNS:
            count = sum(int(row[column]) for row in final_rows)
            print(
                f"  {column:<8} {count:>6}  "
                f"({count/len(final_rows)*100:5.1f}%)"
            )
        normal = sum(
            1 for row in final_rows if not any(int(row[c]) for c in LABEL_COLUMNS)
        )
        print(f"  {'normal':<8} {normal:>6}  ({normal/len(final_rows)*100:5.1f}%)")

    print(f"\n片段目录：{os.path.join(args.out_root, 'clips')}")
    print(f"标注文件：{labels_path}")
    print(
        "\n注意：现有 train.py 的 LABEL_COLUMNS 只有 running/falling/fighting。\n"
        "要用 sitting/walking/kicking 训练，先把 dataset.py 的 LABEL_COLUMNS\n"
        "和 LABEL_NAMES_ZH 补上这三列，再用 --labels-cols 指定。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
