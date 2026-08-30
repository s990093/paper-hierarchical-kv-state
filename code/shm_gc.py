#!/usr/bin/env python3
"""清掉自己洩漏在 /dev/shm 的 vLLM 卸載 mmap 檔。

## 為什麼需要這個

vLLM 的 `CPUOffloadingSpec` 把 CPU 階配置成 `/dev/shm` 上的 mmap 檔
（啟動 log：`Created mmap file /dev/shm/vllm_offload_<uuid>.mmap (32.00 GB)`）。
**server 被 SIGKILL 時這個檔不會被回收。**

2026-08-30 實測：連續跑幾輪 baseline 之後 `/dev/shm` 就 100% 滿
（221 GB 全部是自己的洩漏檔），接著所有帶卸載的 baseline 都在啟動時死掉。
這跟 GPU 插隊是同一類問題——**共用資源被靜默耗盡**，而且錯誤訊息完全不指向真因。

## 安全性

* **只動自己的檔案**（uid 相符）。`/dev/shm` 是全機共用的，
  上面有其他使用者的 `__KMP_REGISTERED_LIB_*` 等檔案，一律不碰。
* **只刪沒有行程持有的**。用 `/proc/*/maps` 逐一比對，有人開著就跳過。
* 預設 dry-run。要真的刪必須加 `--apply`。

用法:
    python code/shm_gc.py            # 只看，不刪
    python code/shm_gc.py --apply    # 真的刪孤兒檔
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SHM = Path("/dev/shm")
PATTERNS = ("vllm_offload_",)


def in_use() -> set[str]:
    """目前被任何行程 mmap 或開啟的 /dev/shm 路徑。"""
    used: set[str] = set()
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/maps") as f:
                for line in f:
                    if "/dev/shm/" in line:
                        used.add(line.rstrip().split(" ", 5)[-1].strip())
        except OSError:
            pass
        fd = f"/proc/{d}/fd"
        try:
            for e in os.listdir(fd):
                try:
                    t = os.readlink(f"{fd}/{e}")
                except OSError:
                    continue
                if t.startswith("/dev/shm/"):
                    used.add(t)
        except OSError:
            pass
    return used


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真的刪除（預設只列出）")
    a = ap.parse_args()

    me = os.getuid()
    used = in_use()
    mine, orphan, held, foreign = [], [], [], 0
    for f in sorted(SHM.iterdir()):
        try:
            st = f.stat()
        except OSError:
            continue
        if st.st_uid != me:
            foreign += 1
            continue
        if not any(f.name.startswith(p) for p in PATTERNS):
            continue
        mine.append((f, st.st_size))
        (held if str(f) in used else orphan).append((f, st.st_size))

    gb = lambda b: b / 1024**3  # noqa: E731
    total = sum(s for _, s in mine)
    o_total = sum(s for _, s in orphan)
    st = os.statvfs(SHM)
    print(f"/dev/shm: {gb(st.f_blocks * st.f_frsize):.0f} GB total, "
          f"{gb(st.f_bavail * st.f_frsize):.1f} GB free")
    print(f"其他使用者的檔案 {foreign} 個（不碰）")
    print(f"我的 vllm_offload 檔 {len(mine)} 個，共 {gb(total):.1f} GB")
    print(f"  仍被行程持有 : {len(held):>3} 個 ({gb(sum(s for _, s in held)):.1f} GB) — 跳過")
    print(f"  孤兒（可刪） : {len(orphan):>3} 個 ({gb(o_total):.1f} GB)")
    for f, s in orphan:
        print(f"      {f.name}  {gb(s):.1f} GB")

    if not orphan:
        return 0
    if not a.apply:
        print("\n(dry-run。加 --apply 才會真的刪)")
        return 0

    freed = 0
    for f, s in orphan:
        try:
            f.unlink()
            freed += s
        except OSError as e:
            print(f"  刪不掉 {f.name}: {e}")
    print(f"\n釋放 {gb(freed):.1f} GB")
    st = os.statvfs(SHM)
    print(f"/dev/shm 現在剩 {gb(st.f_bavail * st.f_frsize):.1f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
