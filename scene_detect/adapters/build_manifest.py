"""扫描 HMDB51 和 GMDCSA24，生成统一的视频清单。

输出一行一个视频，后续的摔倒起始帧标注和特征抽取都读这份清单。

用法（在 PLACE 环境里）：
    python adapters/build_manifest.py
    python adapters/build_manifest.py --limit 20
    python adapters/build_manifest.py --hmdb-classes sit,walk,kick,run,fall_floor

默认只收录和当前项目有关的 HMDB51 类别，避免把 6766 段全部跑一遍。
想全收就传 --hmdb-classes all。
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import cv2

DATASETS_ROOT = r"D:\SceneDetection\datasets"
DEFAULT_HMDB = os.path.join(DATASETS_ROOT, "hmdb51_org")
DEFAULT_OUT = r"D:\SceneDetection\features\new_datasets"

# 目录名 -> 标签。没列出的类别被视为「正常」，但仍然保留动作名方便筛选。
HMDB_LABELS = {
    "sit": {"sitting": 1},
    "walk": {"walking": 1},
    "kick": {"kicking": 1},
    "kick_ball": {"kicking": 1},
    "run": {"running": 1},
    "fall_floor": {"falling": 1},
    "punch": {"fighting": 1},
    "push": {},
    "stand": {},
}

LABEL_COLUMNS = (
    "running",
    "falling",
    "fighting",
    "sitting",
    "walking",
    "kicking",
)

DEFAULT_HMDB_CLASSES = "sit,walk,kick,kick_ball,run,fall_floor"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}


def find_gmdcsa_root() -> str | None:
    """GMDCSA24 解压后的目录名带提交哈希，按前缀找。"""
    pattern = os.path.join(
        DATASETS_ROOT, "fall", "ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-*"
    )
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def probe_video(path: str) -> dict:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        return {"fps": 0.0, "frames": 0, "width": 0, "height": 0, "duration_sec": 0.0}
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return {
        "fps": round(fps, 3),
        "frames": frames,
        "width": width,
        "height": height,
        "duration_sec": round(frames / fps, 2) if fps > 0 else 0.0,
    }


def hmdb_identity(stem: str, action: str) -> tuple[str, str]:
    """从 HMDB 文件名里拆出演员名，用来做按人分组。

    文件名形如 sit_Anja_sit_u_nm_np1_fr_med_0。
    分组键写进文件名前缀（双下划线之前），dataset.py 的 split_by_group
    会自动把它当作 group，这样同一个演员不会同时出现在训练和验证里。
    """
    prefix = action + "_"
    rest = stem[len(prefix) :] if stem.startswith(prefix) else stem
    actor = rest.split("_", 1)[0].strip() or "unknown"
    group = f"hmdb_{actor}"
    uid = f"{group}__{action}_{stem}"
    return group, uid


def make_row(
    path: str,
    dataset: str,
    source: str,
    group: str,
    uid: str,
    action: str,
    subject: str,
    labels: dict[str, int],
) -> dict:
    info = probe_video(path)
    row = {
        "video_uid": uid,
        "video": os.path.basename(path),
        "path": path,
        "dataset": dataset,
        "source": source,
        "group": group,
        "action": action,
        "subject": subject,
    }
    row.update(info)
    for column in LABEL_COLUMNS:
        row[f"label_{column}"] = int(labels.get(column, 0))
    return row


def collect_hmdb(root: str, classes: list[str], limit: int) -> list[dict]:
    rows: list[dict] = []
    available = sorted(
        name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))
    )
    wanted = available if classes == ["all"] else classes
    for action in wanted:
        folder = os.path.join(root, action)
        if not os.path.isdir(folder):
            print(f"  !! HMDB 缺少类别目录：{action}")
            continue
        videos = sorted(
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS
        )
        if limit:
            videos = videos[:limit]
        for path in videos:
            stem = os.path.splitext(os.path.basename(path))[0]
            group, uid = hmdb_identity(stem, action)
            rows.append(
                make_row(
                    path=path,
                    dataset="hmdb51",
                    source=stem,
                    group=group,
                    uid=uid,
                    action=action,
                    subject="",
                    labels=HMDB_LABELS.get(action, {}),
                )
            )
        print(f"  HMDB {action:<12} {len(videos)} 段")
    return rows


def collect_gmdcsa(root: str, limit: int) -> list[dict]:
    rows: list[dict] = []
    subjects = sorted(
        name
        for name in os.listdir(root)
        if name.lower().startswith("subject")
        and os.path.isdir(os.path.join(root, name))
    )
    for subject_dir in subjects:
        digits = "".join(ch for ch in subject_dir if ch.isdigit()) or subject_dir
        for part in ("ADL", "Fall"):
            folder = os.path.join(root, subject_dir, part)
            if not os.path.isdir(folder):
                continue
            videos = sorted(
                os.path.join(folder, name)
                for name in os.listdir(folder)
                if os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS
            )
            if limit:
                videos = videos[:limit]
            group = f"gmdcsa_S{digits}"
            for path in videos:
                stem = os.path.splitext(os.path.basename(path))[0]
                uid = f"{group}__{part}_{stem}"
                labels = {"falling": 1} if part == "Fall" else {}
                rows.append(
                    make_row(
                        path=path,
                        dataset="gmdcsa24",
                        source=stem,
                        group=group,
                        uid=uid,
                        action="fall" if part == "Fall" else "adl",
                        subject=f"S{digits}",
                        labels=labels,
                    )
                )
            print(f"  GMDCSA {subject_dir:<10} {part:<4} {len(videos)} 段")
    return rows


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "video_uid",
        "video",
        "path",
        "dataset",
        "source",
        "group",
        "action",
        "subject",
        "fps",
        "frames",
        "width",
        "height",
        "duration_sec",
        *[f"label_{column}" for column in LABEL_COLUMNS],
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 68)
    print(f"共 {len(rows)} 段视频，{sum(r['duration_sec'] for r in rows)/60:.1f} 分钟")
    by_dataset: dict[str, dict] = {}
    for row in rows:
        item = by_dataset.setdefault(
            row["dataset"], {"videos": 0, "seconds": 0.0, "actions": {}}
        )
        item["videos"] += 1
        item["seconds"] += row["duration_sec"]
        item["actions"][row["action"]] = item["actions"].get(row["action"], 0) + 1
    for name, item in sorted(by_dataset.items()):
        print(
            f"  {name:<10} {item['videos']:>5} 段  "
            f"{item['seconds']/60:>6.1f} 分钟  {item['actions']}"
        )
    broken = [r for r in rows if r["frames"] <= 0 or r["fps"] <= 0]
    if broken:
        print(f"\n  !! {len(broken)} 段打不开或元数据异常：")
        for row in broken[:10]:
            print("    ", row["path"])


def main() -> int:
    parser = argparse.ArgumentParser(description="生成新数据集的统一视频清单")
    parser.add_argument("--hmdb-root", default=DEFAULT_HMDB)
    parser.add_argument(
        "--gmdcsa-root",
        default=None,
        help="默认自动查找 datasets/fall/ekramalam-GMDCSA24-*",
    )
    parser.add_argument("--out", default=os.path.join(DEFAULT_OUT, "videos.csv"))
    parser.add_argument(
        "--datasets",
        default="hmdb,gmdcsa24",
        help="要扫描的数据集，逗号分隔：hmdb,gmdcsa24",
    )
    parser.add_argument(
        "--hmdb-classes",
        default=DEFAULT_HMDB_CLASSES,
        help="HMDB 类别，逗号分隔；all 表示全部 51 类",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="每个类别/子目录只取前 N 段，用于快速试跑",
    )
    args = parser.parse_args()

    wanted = {item.strip().lower() for item in args.datasets.split(",") if item.strip()}
    hmdb_classes = [
        item.strip() for item in args.hmdb_classes.split(",") if item.strip()
    ]
    rows: list[dict] = []

    if "hmdb" in wanted or "hmdb51" in wanted:
        if not os.path.isdir(args.hmdb_root):
            print(f"找不到 HMDB51：{args.hmdb_root}")
            return 1
        print(f"[HMDB51] {args.hmdb_root}")
        rows.extend(collect_hmdb(args.hmdb_root, hmdb_classes, args.limit))

    if "gmdcsa24" in wanted or "gmdcsa" in wanted:
        root = args.gmdcsa_root or find_gmdcsa_root()
        if not root or not os.path.isdir(root):
            print("找不到 GMDCSA24，检查 --gmdcsa-root")
            return 1
        print(f"[GMDCSA24] {root}")
        rows.extend(collect_gmdcsa(root, args.limit))

    if not rows:
        print("没有找到任何视频")
        return 1

    rows.sort(key=lambda r: (r["dataset"], r["group"], r["action"], r["source"]))
    write_csv(args.out, rows)
    print_summary(rows)
    print(f"\n清单已写入：{args.out}")
    print("下一步：python adapters/detect_fall_onsets.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
