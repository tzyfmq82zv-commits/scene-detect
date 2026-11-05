"""把攀爬数据集每段视频均匀抽帧并缓存为 .npy。

用法：
    python adapters/cache_climb_frames.py
    python adapters/cache_climb_frames.py --frames 16 --size 224 --workers 8

输出：
    D:\\SceneDetection\\features\\climb\\frames_k16\\<split>\\<class>\\<stem>.npy
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

DEFAULT_MANIFEST = r"D:\SceneDetection\features\climb\labels_climb.csv"
DEFAULT_CACHE = r"D:\SceneDetection\features\climb\frames_k16"


def cache_path(cache_root: str, row: dict) -> str:
    stem = os.path.splitext(os.path.basename(row["path"]))[0]
    return os.path.join(cache_root, row["split"], row["class_name"], f"{stem}.npy")


def extract_one(task: tuple[str, str, int, int, bool]) -> tuple[str, bool, str]:
    source, target, frames_per_clip, size, overwrite = task
    if os.path.exists(target) and not overwrite:
        return target, True, "cached"
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        return target, False, "打不开"
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        capture.release()
        return target, False, "帧数为 0"
    picks = np.linspace(0, total - 1, frames_per_clip).astype(int)
    frames = []
    for index in picks:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            continue
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        return target, False, "读不到帧"
    while len(frames) < frames_per_clip:
        frames.append(frames[-1])
    os.makedirs(os.path.dirname(target), exist_ok=True)
    np.save(target, np.stack(frames[:frames_per_clip]).astype(np.uint8))
    return target, True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description="缓存攀爬视频采样帧")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"找不到清单：{args.manifest}\n先运行 prepare_climb.py")
        return 1
    with open(args.manifest, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[: args.limit]

    tasks = [
        (
            row["path"],
            cache_path(args.cache, row),
            args.frames,
            args.size,
            args.overwrite,
        )
        for row in rows
    ]
    started = time.time()
    ok_count = 0
    failures = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(extract_one, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            target, ok, message = future.result()
            if ok:
                ok_count += 1
            else:
                failures.append((target, message))
            if done % 100 == 0 or done == len(futures):
                print(
                    f"  {done}/{len(futures)} 成功 {ok_count} "
                    f"用时 {time.time()-started:.0f}s"
                )

    print(f"\n完成：{ok_count}/{len(tasks)}，缓存目录：{args.cache}")
    if failures:
        print(f"失败 {len(failures)} 个：")
        for target, message in failures[:20]:
            print("  ", message, target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
