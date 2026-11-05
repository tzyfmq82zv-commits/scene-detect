"""多线程分片下载器（底层用 curl 传输）。

为什么不用 Python 的 requests：部分站点（如 Zenodo）会拒绝 Python 默认的
TLS 握手，报 SSL: UNEXPECTED_EOF_WHILE_READING；而 curl 的握手能被正常接受。
所以这里让 curl 负责真正的字节传输，Python 只做任务调度、断点续传和合并。

用法：
    python multidownload.py <url> <输出路径> [--threads 16] [--referer URL]

支持断点续传：中断后直接重新运行同一条命令即可。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def run_curl(args: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["curl.exe", *args],
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def probe(url: str, referer: str | None, timeout: int) -> tuple[int, bool]:
    """返回 (总大小, 是否支持分片)。"""
    args = ["-sL", "-r", "0-0", "-D", "-", "-o", os.devnull, "--max-time", "40", "-A", UA]
    if referer:
        args += ["-e", referer]
    args.append(url)
    result = run_curl(args, timeout)
    headers = result.stdout.decode("utf-8", "ignore")
    total = 0
    supports_range = False
    for line in headers.splitlines():
        low = line.lower().strip()
        if low.startswith("content-range:") and "/" in line:
            supports_range = True
            try:
                total = int(line.rsplit("/", 1)[1].strip())
            except ValueError:
                pass
        elif low.startswith("content-length:") and not total:
            try:
                total = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return total, supports_range


class Progress:
    def __init__(self, total: int, parts: list[str]) -> None:
        self.total = total
        self.parts = parts
        self.start = time.time()
        self._stop = threading.Event()

    def done(self) -> int:
        return sum(os.path.getsize(p) for p in self.parts if os.path.exists(p))

    def render(self) -> str:
        got = self.done()
        elapsed = max(time.time() - self.start, 0.001)
        speed = got / 1024 / elapsed
        if self.total:
            pct = got / self.total * 100
            eta = (self.total - got) / max(got / elapsed, 1)
            return (
                f"{pct:5.1f}%  {got/1024**2:7.1f}/{self.total/1024**2:.1f} MB  "
                f"{speed:7.0f} KB/s  ETA {eta/60:5.1f} min"
            )
        return f"{got/1024**2:7.1f} MB  {speed:7.0f} KB/s"

    def ticker(self) -> None:
        while not self._stop.wait(15):
            sys.stderr.write("\r  " + self.render())
            sys.stderr.flush()

    def start_ticker(self) -> None:
        threading.Thread(target=self.ticker, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()


def fetch_part(
    url: str,
    referer: str | None,
    start: int,
    end: int,
    path: str,
    timeout: int,
    retries: int = 60,
) -> None:
    target = end - start + 1
    for attempt in range(1, retries + 1):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have >= target:
            return
        chunk = f"{start + have}-{end}"
        tmp = f"{path}.tmp"
        args = [
            "-sL",
            "--max-time",
            str(timeout),
            "--retry",
            "0",
            "-A",
            UA,
            "-r",
            chunk,
            "-o",
            tmp,
        ]
        if referer:
            args += ["-e", referer]
        args.append(url)
        try:
            run_curl(args, timeout + 30)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # 无论 curl 是否被中断，都先落盘已收到的字节，避免限速时进度被丢弃
            if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
                try:
                    with open(path, "ab") as dst, open(tmp, "rb") as src:
                        shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
                    os.remove(tmp)
                except OSError:
                    pass
        if os.path.getsize(path) >= target:
            return
        delay = min(2 * attempt, 20)
        sys.stderr.write(
            f"\n  分片 {start} 第 {attempt}/{retries} 轮结束（已收 "
            f"{os.path.getsize(path)/1024**2:.1f}/{target/1024**2:.1f} MB），{delay}s 后继续...\n"
        )
        time.sleep(delay)
    raise RuntimeError(f"分片 {start}-{end} 重试 {retries} 次仍未完成")


def download(url: str, out: str, threads: int, referer: str | None, timeout: int) -> None:
    total, supports_range = probe(url, referer, timeout)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    if os.path.exists(out) and total and os.path.getsize(out) >= total:
        print(f"已完成，跳过：{out}（{os.path.getsize(out)/1024**2:.1f} MB）")
        return

    use_split = supports_range and total > 0 and threads > 1
    print(
        f"总大小 {total/1024**2:.1f} MB，" + ("分片模式 " + str(threads) + " 线程" if use_split else "单线程")
    )

    if not use_split:
        parts = [out + ".part0"]
        total = total or 0
    else:
        span = (total + threads - 1) // threads
        ranges = [
            (i * span, min((i + 1) * span - 1, total - 1))
            for i in range(threads)
            if i * span < total
        ]
        parts = [f"{out}.part{i}" for i in range(len(ranges))]

    progress = Progress(total, parts)
    progress.start_ticker()
    errors: list[Exception] = []

    def worker(index: int) -> None:
        try:
            if use_split:
                start, end = ranges[index]
            else:
                start, end = 0, max(total - 1, 0)
            fetch_part(url, referer, start, end, parts[index], timeout)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    workers = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(len(parts))]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    progress.stop()
    sys.stderr.write("\r  " + progress.render() + "  \n")

    if errors:
        raise RuntimeError(f"{len(errors)} 个分片失败：{errors[0]}")

    if use_split and len(parts) > 1:
        with open(out, "wb") as dst:
            for part in parts:
                with open(part, "rb") as src:
                    while True:
                        block = src.read(8 * 1024 * 1024)
                        if not block:
                            break
                        dst.write(block)
        for part in parts:
            os.remove(part)
    else:
        os.replace(parts[0], out)


def main() -> int:
    parser = argparse.ArgumentParser(description="多线程分片下载器（curl 传输）")
    parser.add_argument("url")
    parser.add_argument("out")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--referer", default=None)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    start = time.time()
    download(args.url, args.out, args.threads, args.referer, args.timeout)
    if os.path.exists(args.out):
        size = os.path.getsize(args.out)
        elapsed = max(time.time() - start, 0.001)
        print(f"完成：{args.out}")
        print(
            f"  {size/1024**2:.1f} MB，用时 {elapsed/60:.1f} 分钟，"
            f"均速 {size/1024/elapsed:.0f} KB/s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
