"""按 UR Fall 的逐帧标注重新标注摔倒片段。

之前的问题：一段摔倒视频会被切成好几个 48 帧的片段，我把**全部**片段都
标成了 falling=1，包括最开头那段人还在直立走动、离倒地还有好几秒的。

这么标的后果很严重，实测 fall-03 的视频里：
    人真正倒地      5.70s
    模型越过阈值    1.40s   ← 人还站着走动
概率从 1.1s 起就稳定在 0.9 以上，也就是说模型学到的根本不是摔倒动作，
而是「这段视频属于摔倒视频」这个整体判断。离线指标里摔倒 F1 0.80 看着
不错，但它是靠「认出这是哪段视频」刷出来的，不是靠识别动作。

UR Fall 官方给了逐帧标注 urfall-cam0-falls.csv，每帧一个标记：
    -1  还没摔（直立活动）
     0  正在倒地
     1  已经倒地

据此改成按「片段窗口里有没有真的包含倒地过程」来打标签：
窗口完全落在 -1 区间的一律标 0。这样模型必须等到窗口里出现倒地动作才能
输出摔倒，提前量自然就消掉了。

用法：
    python relabel_falls.py                  # 就地更新 labels.csv（先自动备份）
    python relabel_falls.py --dry-run        # 只看会改多少个，不写文件
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil

DATASETS_ROOT = r"D:\SceneDetection\datasets"
FEATURES_ROOT = r"D:\SceneDetection\features"
LABELS_CSV = os.path.join(FEATURES_ROOT, "labels.csv")

# UR Fall 的官方标注，第三列是逐帧状态
URFALL_FALLS_CSV = os.path.join(
    DATASETS_ROOT, "fall", "urfall", "urfall-cam0-falls.csv"
)

# 片段长度，必须和抽取时用的一致，否则算出来的窗口范围是错的
CLIP_LENGTH = 48


def load_frame_flags(csv_path: str) -> dict[str, dict[int, int]]:
    """读逐帧标注，返回 {视频名: {帧号: 标记}}。帧号用文件里的 1 起始编号。"""
    if not os.path.exists(csv_path):
        raise SystemExit(
            f"找不到逐帧标注文件：{csv_path}\n"
            "可以从 UR Fall 官网下载 urfall-cam0-falls.csv 放到该目录"
        )
    flags: dict[str, dict[int, int]] = {}
    with open(csv_path, newline="", encoding="utf-8", errors="ignore") as handle:
        for row in csv.reader(handle):
            if len(row) < 3:
                continue
            try:
                frame = int(row[1])
                flag = int(row[2])
            except ValueError:
                continue
            flags.setdefault(row[0].strip(), {})[frame] = flag
    if not flags:
        raise SystemExit(f"标注文件是空的或格式不对：{csv_path}")
    return flags


def normalize_source(name: str) -> str:
    """把片段里的视频名转成标注文件里的写法。

    我们的片段里 source 存的是 mp4 的文件名（fall-01-cam0），
    而官方标注用的是序列号（fall-01），中间差一个 -cam0 / -cam1 后缀，
    不处理的话查表永远查不上，脚本会假装「没有需要修改的片段」。
    """
    for suffix in ("-cam0", "-cam1"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def window_contains_fall(
    flags: dict[int, int], start_frame: int, clip_length: int = CLIP_LENGTH
) -> tuple[bool, int | None]:
    """判断这个片段窗口里有没有倒地过程。

    npz 里存的 start_frame 是从 0 开始的帧序号，而标注文件用的是从 1 开始的
    编号，所以两边要差一个 1。
    返回 (是否含倒地, 窗口内最早出现倒地的帧号)。
    """
    first = start_frame + 1
    last = start_frame + clip_length
    earliest = None
    for frame in range(first, last + 1):
        flag = flags.get(frame)
        if flag is not None and flag >= 0:
            earliest = frame
            break
    return (earliest is not None), earliest


def main() -> int:
    parser = argparse.ArgumentParser(description="按逐帧标注重新标注摔倒片段")
    parser.add_argument("--labels", default=LABELS_CSV, help="标注 CSV 路径")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    args = parser.parse_args()

    annotations = load_frame_flags(URFALL_FALLS_CSV)
    print(f"逐帧标注：{len(annotations)} 段视频")

    with open(args.labels, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    fieldnames = list(rows[0].keys())
    print(f"标注文件：{len(rows)} 个片段")

    changed_to_zero = 0
    kept_positive = 0
    untouched = 0
    for row in rows:
        if row.get("dataset") != "urfall" or str(row.get("falling", "0")) != "1":
            untouched += 1
            continue
        source = row.get("source", "")
        flags = annotations.get(normalize_source(source))
        if not flags:
            # 这段视频没有逐帧标注，保守起见保持原样，避免误改成 0
            print(f"  !! 找不到逐帧标注，跳过：{source}")
            untouched += 1
            continue
        try:
            start_frame = int(float(row.get("start_frame", 0)))
        except ValueError:
            untouched += 1
            continue

        contains_fall, earliest = window_contains_fall(flags, start_frame)
        if contains_fall:
            kept_positive += 1
            row["fall_start_frame"] = str(earliest)
        else:
            row["falling"] = "0"
            changed_to_zero += 1

    print(
        f"\n摔倒片段：保留为正样本 {kept_positive} 个，"
        f"改成 0 {changed_to_zero} 个，其余不动 {untouched} 个"
    )
    if changed_to_zero == 0:
        print("没有需要修改的片段——可能已经标注过了，或者片段长度参数对不上")
        return 0

    if args.dry_run:
        print("\n这是 dry-run，没有写任何文件。确认无误后去掉 --dry-run 重跑。")
        return 0

    if "fall_start_frame" not in fieldnames:
        fieldnames.append("fall_start_frame")
    backup = args.labels + ".bak"
    shutil.copyfile(args.labels, backup)
    with open(args.labels, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_falling = sum(1 for row in rows if str(row.get("falling", "0")) == "1")
    print(f"\n原文件已备份到 {backup}")
    print(f"更新完成，现在摔倒正样本 {total_falling} 个（原来 98 个）")
    print("\n下一步重训：python train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
