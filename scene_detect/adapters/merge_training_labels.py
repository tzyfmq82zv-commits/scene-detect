"""把旧的 labels.csv 和新的 labels_new.csv 合并成 6 列统一标注。

旧数据：running / falling / fighting
新数据：sitting / walking / kicking，同时带 running / falling

输出 labels_six.csv，列固定为：
    running, falling, fighting, sitting, walking, kicking
全 0 表示正常。

用法：
    python adapters/merge_training_labels.py
    python adapters/merge_training_labels.py --out D:\\SceneDetection\\features\\labels_six.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import os

FEATURES_ROOT = r"D:\SceneDetection\features"
OLD_LABELS = os.path.join(FEATURES_ROOT, "labels.csv")
NEW_LABELS = os.path.join(FEATURES_ROOT, "new_datasets", "labels_new.csv")
DEFAULT_OUT = os.path.join(FEATURES_ROOT, "labels_six.csv")

COLUMNS = ("running", "falling", "fighting", "sitting", "walking", "kicking")
FIELDNAMES = [
    "clip",
    "dataset",
    "source",
    "track_id",
    "start_frame",
    "frames",
    *COLUMNS,
    "fall_start_frame",
    "group",
    "action",
    "video_uid",
]


def read_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def normalize_clip(path: str) -> str:
    absolute = path if os.path.isabs(path) else os.path.join(FEATURES_ROOT, path)
    absolute = os.path.abspath(absolute)
    try:
        return os.path.relpath(absolute, FEATURES_ROOT).replace("\\", "/")
    except ValueError:
        return absolute.replace("\\", "/")


def to_int(value, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def normalize_row(row: dict) -> dict | None:
    clip = normalize_clip(str(row.get("clip", "")).strip())
    if not clip:
        return None
    absolute = clip if os.path.isabs(clip) else os.path.join(FEATURES_ROOT, clip)
    if not os.path.exists(absolute):
        return None
    stem = os.path.splitext(os.path.basename(clip))[0]
    group = str(row.get("group", "")).strip() or stem.split("__")[0]
    result = {
        "clip": clip,
        "dataset": row.get("dataset", ""),
        "source": row.get("source", ""),
        "track_id": row.get("track_id", ""),
        "start_frame": row.get("start_frame", 0),
        "frames": row.get("frames", 0),
        "fall_start_frame": row.get("fall_start_frame", ""),
        "group": group,
        "action": row.get("action", ""),
        "video_uid": row.get("video_uid", ""),
    }
    for column in COLUMNS:
        result[column] = to_int(row.get(column, 0))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="合并旧标注和新标注")
    parser.add_argument("--old", default=OLD_LABELS)
    parser.add_argument("--new", default=NEW_LABELS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    merged: dict[str, dict] = {}
    missing = 0
    for path, name in ((args.old, "旧数据"), (args.new, "新数据")):
        if not os.path.exists(path):
            print(f"!! 找不到{name}标注：{path}")
            return 1
        rows = read_rows(path)
        kept = 0
        for row in rows:
            normalized = normalize_row(row)
            if normalized is None:
                missing += 1
                continue
            merged[normalized["clip"]] = normalized
            kept += 1
        print(f"{name}: 读入 {len(rows)} 行，有效 {kept} 行")

    rows = sorted(merged.values(), key=lambda r: (r["dataset"], r["group"], r["clip"]))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 68)
    print(f"合并后 {len(rows)} 个片段，跳过缺失文件 {missing} 个")
    for column in COLUMNS:
        count = sum(row[column] for row in rows)
        print(f"  {column:<9} {count:>6}  ({count/len(rows)*100:5.1f}%)")
    normal = sum(1 for row in rows if not any(row[c] for c in COLUMNS))
    print(f"  {'normal':<9} {normal:>6}  ({normal/len(rows)*100:5.1f}%)")
    print("\n按数据集：")
    by_dataset = collections.Counter(row["dataset"] for row in rows)
    for name, count in sorted(by_dataset.items()):
        print(f"  {name:<20} {count:>6}")
    print(f"\n分组数：{len(set(row['group'] for row in rows))}")
    print(f"标注文件：{args.out}")
    print("下一步：python adapters/extract_appearance_new.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
