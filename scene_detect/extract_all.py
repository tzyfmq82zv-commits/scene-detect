"""下一步要跑的脚本：对全部数据集批量提取骨架特征，并自动生成标注文件。

一条命令跑完里程碑 2 的数据准备部分：
    视频 -> YOLO11-Pose + BoT-SORT -> 骨架片段 .npz -> labels.csv

用法：
    python extract_all.py                  # 全量提取（几千个视频，耗时较长）
    python extract_all.py --limit 3        # 每个数据集只跑 3 个视频，先验证流程
    python extract_all.py --save-vis       # 额外输出带骨架的可视化视频，人工检查跟踪

跑完会得到：
    D:\\SceneDetection\\features\\<数据集名>\\*.npz    骨架片段
    D:\\SceneDetection\\features\\<数据集名>\\manifest.csv
    D:\\SceneDetection\\features\\labels.csv           汇总标注，直接喂给 train.py
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time

from feature_extract import collect_videos, crop_video, extract_from_video

DATASETS_ROOT = r"D:\SceneDetection\datasets"
FEATURES_ROOT = r"D:\SceneDetection\features"

# 每个数据集怎么标，规则按顺序匹配视频文件名，第一个命中的生效。
# 标签列的顺序固定是 running / falling / fighting，没写到的列默认 0。
# 三列全 0 就是「正常」，这是我们的第四个语义类别，不需要单独建类。
DATASETS: list[dict] = [
    {
        "name": "urfall",
        "source": os.path.join(DATASETS_ROOT, "fall", "urfall"),
        "crop": "right",
        "note": "mp4 是深度图|彩色图左右拼接，必须裁右半边，否则 YOLO 检不到人",
        "rules": [
            (r"^fall-", {"falling": 1}),
            (r"^adl-", {}),
        ],
    },
    {
        "name": "busviolence_fight",
        "source": os.path.join(DATASETS_ROOT, "fight", "bus_violence", "Violence"),
        "crop": "none",
        "note": "标成打架；但其中一部分其实是偷窃等暴力行为，详见结尾提示",
        "rules": [(r".*", {"fighting": 1})],
    },
    {
        "name": "busviolence_normal",
        "source": os.path.join(DATASETS_ROOT, "fight", "bus_violence", "NoViolence"),
        "crop": "none",
        "note": "正常类负样本",
        "rules": [(r".*", {})],
    },
    {
        "name": "weizmann",
        "source": os.path.join(DATASETS_ROOT, "run", "weizmann"),
        "crop": "none",
        "note": "文件名形如 eli_run / daria_walk，动作在最后一个下划线之后",
        "rules": [
            (r"_run$", {"running": 1}),
            (r".*", {}),
        ],
    },
]

LABEL_COLUMNS = ("running", "falling", "fighting")
LABEL_NAMES_ZH = {"running": "奔跑", "falling": "摔倒", "fighting": "打架"}


def labels_for(video_name: str, rules: list[tuple[str, dict]]) -> dict[str, int]:
    """按规则表决定一个视频的标签。"""
    for pattern, labels in rules:
        if re.search(pattern, video_name, flags=re.IGNORECASE):
            return labels
    return {}


def run_dataset(entry: dict, args, model) -> list[dict]:
    """处理一个数据集，返回带标签的行列表。"""
    name = entry["name"]
    out_dir = os.path.join(FEATURES_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)

    videos = collect_videos(entry["source"])
    if not videos:
        print(f"  !! 没找到视频，跳过：{entry['source']}")
        return []

    if args.limit:
        videos = videos[: args.limit]

    print(f"  视频 {len(videos)} 个   {entry.get('note', '')}")

    # 需要裁剪时先落一份裁好的临时文件，复用 ultralytics 原生的视频读取路径
    temp_dir = None
    if entry["crop"] != "none":
        temp_dir = os.path.join(out_dir, "_cropped")
        os.makedirs(temp_dir, exist_ok=True)
        cropped: list[str] = []
        for index, video in enumerate(videos, start=1):
            target = os.path.join(
                temp_dir, os.path.splitext(os.path.basename(video))[0] + ".mp4"
            )
            crop_video(video, entry["crop"], target)
            cropped.append(target)
            if index % 20 == 0 or index == len(videos):
                print(f"      裁剪 {index}/{len(videos)}")
        videos = cropped

    rows: list[dict] = []
    started = time.time()
    for index, video in enumerate(videos, start=1):
        video_name = os.path.splitext(os.path.basename(video))[0]
        produced = extract_from_video(
            video_path=video,
            out_dir=out_dir,
            model=model,
            tracker=args.tracker,
            clip_length=args.clip_length,
            stride=args.stride,
            min_confidence=args.min_confidence,
            min_track_length=args.min_track_length,
            max_clips=args.max_clips_per_track,
            imgsz=args.imgsz,
            save_vis=args.save_vis,
        )
        labels = labels_for(video_name, entry["rules"])
        for row in produced:
            row["dataset"] = name
            for column in LABEL_COLUMNS:
                row[column] = int(labels.get(column, 0))
            # 片段在 labels.csv 里用相对路径，方便整个 features 目录搬走
            row["clip"] = os.path.relpath(
                os.path.join(out_dir, row["clip"]), FEATURES_ROOT
            ).replace("\\", "/")
            rows.append(row)

        if index % 25 == 0 or index == len(videos):
            elapsed = time.time() - started
            rate = index / max(elapsed, 1e-6)
            remaining = (len(videos) - index) / max(rate, 1e-6)
            print(
                f"      {index}/{len(videos)}  已产出 {len(rows)} 个片段  "
                f"{rate:.1f} 视频/秒  剩余约 {remaining/60:.1f} 分钟"
            )

    manifest_path = os.path.join(out_dir, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"  完成：{len(rows)} 个片段")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="批量提取骨架特征并生成标注")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="YOLO 姿态模型权重")
    parser.add_argument(
        "--tracker",
        default="botsort.yaml",
        help="botsort.yaml 抗遮挡更好；调不出速度时换 bytetrack.yaml",
    )
    parser.add_argument("--clip-length", type=int, default=48, help="片段帧数")
    parser.add_argument("--stride", type=int, default=24, help="滑窗步长")
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--min-track-length", type=int, default=16)
    parser.add_argument("--max-clips-per-track", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--save-vis", action="store_true", help="输出骨架可视化视频")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="每个数据集只处理前 N 个视频，用于快速验证流程",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="只跑指定的数据集（逗号分隔，名字见脚本里的 DATASETS）",
    )
    args = parser.parse_args()

    from ultralytics import YOLO

    selected = DATASETS
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        selected = [entry for entry in DATASETS if entry["name"] in wanted]
        if not selected:
            print(f"没有匹配的数据集：{args.only}")
            return 1

    os.makedirs(FEATURES_ROOT, exist_ok=True)
    print(f"输出目录：{FEATURES_ROOT}")
    if args.limit:
        print(f"注意：--limit {args.limit}，每个数据集只跑前 {args.limit} 个视频")
    print("=" * 70)

    model = YOLO(args.model)
    all_rows: list[dict] = []
    started = time.time()
    for entry in selected:
        print(f"\n[{entry['name']}]  {entry['source']}")
        try:
            all_rows.extend(run_dataset(entry, args, model))
        except KeyboardInterrupt:
            print("\n被中断。已完成的片段仍然保留，重跑会跳过已存在的 npz。")
            break

    if not all_rows:
        print("\n没有产出任何片段，检查一下视频路径和 --min-track-length 是否合适")
        return 1

    labels_path = os.path.join(FEATURES_ROOT, "labels.csv")
    fieldnames = [
        "clip",
        "dataset",
        "source",
        "track_id",
        "start_frame",
        "frames",
        *LABEL_COLUMNS,
    ]
    with open(labels_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    total = len(all_rows)
    print("\n" + "=" * 70)
    print(f"全部完成，用时 {(time.time()-started)/60:.1f} 分钟，共 {total} 个骨架片段")
    print(f"标注文件：{labels_path}")
    print("\n片段分布：")
    for column in LABEL_COLUMNS:
        count = sum(row[column] for row in all_rows)
        print(f"  {LABEL_NAMES_ZH[column]:<4} {count:>6}  ({count/total*100:5.1f}%)")
    normal = sum(1 for row in all_rows if not any(row[c] for c in LABEL_COLUMNS))
    print(f"  正常   {normal:>6}  ({normal/total*100:5.1f}%)")

    per_dataset: dict[str, int] = {}
    for row in all_rows:
        per_dataset[row["dataset"]] = per_dataset.get(row["dataset"], 0) + 1
    print("\n按数据集：")
    for name, count in sorted(per_dataset.items()):
        print(f"  {name:<22} {count:>6}")

    print(
        "\n下一步训练：\n"
        f'  python train.py --labels "{labels_path}" '
        f'--features-root "{FEATURES_ROOT}" --epochs 60'
    )
    print(
        "\n两点提醒：\n"
        "  1. busviolence_fight 里混了一部分「偷窃」而不是「打架」的片段，\n"
        "     标签噪声会拉低打架的精确率。想干净点就抽看几十个片段手工剔一遍。\n"
        "  2. labels.csv 是自动生成的，busviolence 和 weizmann 的标签来自目录和文件名，\n"
        "     跑完建议先看几个 --save-vis 的输出确认视频内容和标签对得上。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
