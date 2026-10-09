#!/usr/bin/env python3
"""量磁碟的持續讀寫頻寬。

## 為什麼需要這個

模擬器的成本模型只有「把 block 讀回來」的價格，**寫下去是免費的**。
但真實硬體不是：toolagent 若把每個新 block 都寫一份到磁碟，
需要持續 7,172 MiB/s。所以「這個策略在真機上寫得下去嗎」是一個
獨立於延遲的可行性判準，而它需要一個**實測的**裝置頻寬，不是規格書數字。

## 做法

O_DIRECT 循序寫入 / 讀取，繞過 page cache（不然量到的是 DRAM）。
每次量測都記整機爭用狀態——磁碟頻寬跟 PCIe 一樣是全機共用的。

用法：
  python code/disk_bw.py --paths /ssd7/hungwei /home/hungwei --size-mib 2048
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import mmap
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_guard import host_contention  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results/m2_harness"
CHUNK = 4 * 1024 * 1024      # 4 MiB，對齊 O_DIRECT 的區塊需求


def device_of(path: str) -> str:
    try:
        out = subprocess.run(["df", "--output=source", path],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[-1]
    except Exception:  # noqa: BLE001
        return "?"


def measure(path: Path, size_mib: int) -> dict:
    path.mkdir(parents=True, exist_ok=True)
    f = path / f".disk_bw_probe_{os.getpid()}.bin"
    # 🔴 O_DIRECT 要求緩衝區**頁對齊**。bytearray 是 malloc 出來的，
    #    對齊與否看運氣——同一支程式在 /（nvme）成功、在 /ssd7（sata）
    #    回 EINVAL，原因就是這個，不是檔案系統不支援。
    #    mmap(-1, n) 拿到的是匿名映射，保證頁對齊。
    wbuf = mmap.mmap(-1, CHUNK)
    wbuf.write(os.urandom(CHUNK))
    wbuf.seek(0)
    rbuf = mmap.mmap(-1, CHUNK)
    n = max(1, size_mib * 1024 * 1024 // CHUNK)
    res: dict = {"path": str(path), "device": device_of(str(path)),
                 "size_mib": n * CHUNK // 1024**2}
    # O_DIRECT 需要對齊的緩衝區，且不是每種檔案系統都支援
    # （/ssd7 的 fuseblk/exfat 之類會回 EINVAL）。
    # 退路：不用 O_DIRECT，改以 fsync 強制落盤（寫入頻寬照樣量得到），
    # 讀取則在 fsync 後用 posix_fadvise(DONTNEED) 把 page cache 丟掉。
    direct = True
    try:
        probe = os.open(f, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT)
        os.close(probe)
    except OSError:
        direct = False
    res["o_direct"] = direct
    wflags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | (os.O_DIRECT if direct else 0)
    rflags = os.O_RDONLY | (os.O_DIRECT if direct else 0)
    try:
        fd = os.open(f, wflags)
        try:
            mv = memoryview(wbuf)
            t0 = time.perf_counter()
            for _ in range(n):
                os.write(fd, mv)
            os.fsync(fd)
            dt = time.perf_counter() - t0
        finally:
            mv.release()
            os.close(fd)
        res["write_mibps"] = round(res["size_mib"] / dt, 1)
        res["write_s"] = round(dt, 3)

        # 讀。沒有 O_DIRECT 時先把這個檔案的 page cache 丟掉，
        # 否則量到的是 DRAM 不是磁碟。
        fd = os.open(f, rflags)
        try:
            if not direct:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            mv = memoryview(rbuf)
            t0 = time.perf_counter()
            while os.readv(fd, [mv]):
                pass
            dt = time.perf_counter() - t0
        finally:
            mv.release()
            os.close(fd)
        res["read_mibps"] = round(res["size_mib"] / dt, 1)
        res["read_s"] = round(dt, 3)
    except OSError as e:
        res["error"] = f"{type(e).__name__}: {e}"
    finally:
        f.unlink(missing_ok=True)
        wbuf.close()
        rbuf.close()
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+",
                    default=["/ssd7/hungwei", "/home/hungwei"])
    ap.add_argument("--size-mib", type=int, default=2048)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=str(OUT / "disk_bw.csv"))
    a = ap.parse_args()

    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-disk-bw"
    hc = host_contention()
    print(f"整機爭用：{hc['level']}（外來 process {hc['foreign_procs']} 個）")
    print("⚠️ 磁碟頻寬跟 PCIe 一樣是全機共用的；HEAVY 下量到的是下界。")
    rows = []
    for p in a.paths:
        for i in range(a.repeats):
            r = measure(Path(p), a.size_mib)
            r.update({"run_id": run_id, "repeat": i,
                      "ts": datetime.now().astimezone().isoformat(),
                      "host_contention": hc["level"],
                      "foreign_procs": hc["foreign_procs"]})
            rows.append(r)
            if "error" in r:
                print(f"  {p:22s} #{i} 🔴 {r['error']}")
            else:
                print(f"  {p:22s} #{i} 寫 {r['write_mibps']:>8,.1f} MiB/s"
                      f"　讀 {r['read_mibps']:>8,.1f} MiB/s  ({r['device']})")
    o = Path(a.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with o.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {o}")
    ok = [r for r in rows if "error" not in r]
    if ok:
        from statistics import median
        for p in a.paths:
            v = [r for r in ok if r["path"] == p]
            if v:
                print(f"  {p:22s} 中位：寫 "
                      f"{median(x['write_mibps'] for x in v):,.0f} MiB/s、讀 "
                      f"{median(x['read_mibps'] for x in v):,.0f} MiB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
