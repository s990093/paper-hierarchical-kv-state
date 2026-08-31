#!/usr/bin/env python3
"""GPU 獨佔性守衛 —— 這是一台共用機器。

## 為什麼需要這個

`/ssd7` 底下有二十幾個使用者的目錄。**隨時可能有人插隊佔用 GPU。**
如果在量 TTFT / TPOT / peak VRAM 的時候有別人的 process 進來：

* 時間數字被 SM 爭用污染 → TTFT/TPOT 全部偏高，而且偏多少無法事後修正
* `peak_vram` 讀到的是兩個 process 的總和 → 容量結論直接錯
* 最糟的是**它會靜默發生**，跑出來的數字看起來完全正常

`EXPERIMENT_PLAN.md` §0 禁令 1 說「不准編造數字」。**被污染的數字比沒有數字更糟**，
因為它看起來像是量到的。所以這支工具做三件事：

1. **開跑前**：目標 GPU 必須是乾淨的（無其他 compute process）。不乾淨就不要開始。
2. **跑的時候**：背景取樣，記下任何外來 PID 出現的時刻與它用了多少記憶體。
3. **跑完後**：只要中途出現過外來 process，該次 run 標成 `CONTAMINATED`，
   **結果不得寫進 results/，必須重量。**

## 用法

```bash
python code/gpu_guard.py --check 0                 # 0 = 乾淨可用，非 0 = 有人在用
python code/gpu_guard.py --idle-gpus               # 印出目前乾淨的 GPU index
python code/gpu_guard.py --watch 0 --out w.jsonl   # 前景監看（Ctrl-C 停）
```

程式內使用：

```python
from gpu_guard import GpuWatcher
with GpuWatcher(gpu=0) as w:
    ...跑實驗...
if w.contaminated:
    print(w.verdict())   # 這次不算數，重跑
```
"""

from __future__ import annotations

import argparse
import json
import os
import json
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

POLL_S = 3.0


# ─────────────────────── 廠商抽象（NVIDIA / AMD） ───────────────────────
# 目前在 RTX 3090 上打磨方法；平台 B 是 AMD MI300X（論文的 κ 第二個點）。
# 只有這一層碰得到廠商工具，其餘程式碼一律走 compute_apps() / gpu_util()
# / free_mib() 這三個介面。搬到 ROCm 只需要這裡能跑。
#
# 🔴 查詢失敗**不可以**回傳空清單。空清單會被 host_contention() 讀成
#    「整機沒有外來負載」= QUIET，等於把「沒量到」寫成「沒人用」，
#    然後把污染的數字標成乾淨的——這正是 EXPERIMENT_PLAN §0 禁令 1
#    要防的失敗模式。查不到就要讓上層知道。
class SmiUnavailable(RuntimeError):
    """找不到（或無法執行）GPU 查詢工具。呼叫端必須顯式處理，不可當成乾淨。"""


def _which(*names: str) -> str | None:
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None


def vendor() -> str:
    """'nvidia' / 'amd'；都找不到就丟 SmiUnavailable。"""
    if _which("nvidia-smi"):
        return "nvidia"
    if _which("amd-smi", "rocm-smi"):
        return "amd"
    raise SmiUnavailable(
        "找不到 nvidia-smi 也找不到 amd-smi/rocm-smi。\n"
        "爭用偵測無法運作 → 不得進行任何計時量測（無法判斷機器是否乾淨）。")


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SmiUnavailable(f"{cmd[0]} 執行失敗：{e}") from e
    if out.returncode != 0:
        raise SmiUnavailable(
            f"{cmd[0]} 回傳 {out.returncode}：{out.stderr.strip()[:200]}")
    return out.stdout


def _smi(query: str, extra: list[str] | None = None) -> list[list[str]]:
    """NVIDIA 專用的原始查詢。AMD 路徑不走這裡。"""
    out = _run(["nvidia-smi", f"--query-{query}",
                "--format=csv,noheader,nounits", *(extra or [])])
    return [[c.strip() for c in line.split(",")]
            for line in out.strip().splitlines() if line.strip()]


def _amd_json(args: list[str]) -> dict | list:
    """amd-smi / rocm-smi 的 JSON 輸出。兩者格式不同，各自解析。"""
    exe = _which("amd-smi", "rocm-smi")
    if exe is None:
        raise SmiUnavailable("amd-smi/rocm-smi 不存在")
    return json.loads(_run([exe, *args, "--json"]))


def uuid_to_index() -> dict[str, int]:
    return {r[1]: int(r[0]) for r in _smi("gpu=index,uuid") if len(r) >= 2}


def compute_apps() -> list[dict]:
    """目前所有 GPU 上的 compute process。查不到就丟 SmiUnavailable。"""
    if vendor() == "amd":
        return _amd_compute_apps()
    idx = uuid_to_index()
    rows = []
    for r in _smi("compute-apps=gpu_uuid,pid,used_memory"):
        if len(r) < 3:
            continue
        try:
            rows.append({"gpu": idx.get(r[0], -1), "pid": int(r[1]),
                         "used_mib": int(r[2])})
        except ValueError:
            continue
    return rows


def _amd_compute_apps() -> list[dict]:
    """AMD 後端。amd-smi（ROCm 6+）與 rocm-smi（舊版）的欄位名不同，兩種都試。

    ⚠️ 這條路徑**尚未在真機驗證過**（本機只有 NVIDIA）。
       第一次在 MI300X 上跑時，必須先用
       `python code/gpu_guard.py --selftest` 對照 `amd-smi process` 的輸出，
       確認 pid 與 gpu index 都對得上，再開始任何計時量測。
    """
    data = _amd_json(["process"])
    rows: list[dict] = []
    items = data if isinstance(data, list) else list(data.values())
    for i, entry in enumerate(items):
        gpu = entry.get("gpu", entry.get("gpu_id", entry.get("card", i)))
        procs = entry.get("process_list", entry.get("process_info", []))
        if isinstance(procs, dict):
            procs = list(procs.values())
        for pr in procs or []:
            info = pr.get("process_info", pr)
            pid = info.get("pid", info.get("PID"))
            mem = (info.get("memory_usage", {}) or {}).get("vram_mem",
                                                           info.get("VRAM", 0))
            if pid is None:
                continue
            try:
                rows.append({"gpu": int(gpu), "pid": int(pid),
                             "used_mib": int(mem) // (1024 * 1024)
                             if int(mem) > 1 << 20 else int(mem)})
            except (TypeError, ValueError):
                continue
    return rows


def gpu_util() -> dict[int, int]:
    """每張卡目前的使用率（%）。"""
    if vendor() == "amd":
        data = _amd_json(["metric", "-u"])
        items = data if isinstance(data, list) else list(data.values())
        out = {}
        for i, e in enumerate(items):
            g = e.get("gpu", e.get("gpu_id", i))
            u = (e.get("usage", {}) or {}).get("gfx_activity",
                                               e.get("GPU use (%)", 0))
            if isinstance(u, dict):
                u = u.get("value", 0)
            try:
                out[int(g)] = int(u)
            except (TypeError, ValueError):
                continue
        return out
    return {int(r[0]): int(r[1]) for r in _smi("gpu=index,utilization.gpu")
            if len(r) >= 2 and r[1].isdigit()}


def _descendants(root: int) -> set[int]:
    """root 的所有子孫 PID（含自己）。用 /proc 走，不依賴 psutil。"""
    children: dict[int, list[int]] = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                parts = f.read().rsplit(")", 1)[1].split()
            children.setdefault(int(parts[1]), []).append(int(d))
        except (OSError, IndexError, ValueError):
            continue
    seen, stack = {root}, [root]
    while stack:
        for c in children.get(stack.pop(), []):
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return seen


def foreign_on(gpu: int, own_root: int | None = None) -> list[dict]:
    """gpu 上不屬於我們的 compute process。own_root 預設為本行程。"""
    own = _descendants(own_root if own_root is not None else os.getpid())
    return [a for a in compute_apps() if a["gpu"] == gpu and a["pid"] not in own]


@dataclass
class GpuWatcher:
    """量測期間持續監看某張卡，記錄任何外來 process。

    contaminated == True 代表這次 run 的數字不可用，必須重量。
    """

    gpu: int
    poll_s: float = POLL_S
    own_root: int | None = None
    out_path: str | None = None

    samples: list[dict] = field(default_factory=list)
    intruders: dict[int, dict] = field(default_factory=dict)
    started_clean: bool | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _t: threading.Thread | None = None

    @property
    def contaminated(self) -> bool:
        return bool(self.intruders) or self.started_clean is False

    def _sample(self) -> None:
        f = foreign_on(self.gpu, self.own_root)
        now = datetime.now().astimezone().isoformat()
        for a in f:
            rec = self.intruders.setdefault(
                a["pid"], {"pid": a["pid"], "first_seen": now, "peak_mib": 0, "n": 0})
            rec["last_seen"] = now
            rec["peak_mib"] = max(rec["peak_mib"], a["used_mib"])
            rec["n"] += 1
        if f:
            self.samples.append({"ts": now, "foreign": f})

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_s):
            self._sample()

    def __enter__(self) -> GpuWatcher:
        pre = foreign_on(self.gpu, self.own_root)
        self.started_clean = not pre
        for a in pre:
            self.intruders[a["pid"]] = {
                "pid": a["pid"], "first_seen": "BEFORE_START",
                "last_seen": "BEFORE_START", "peak_mib": a["used_mib"], "n": 1}
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._t:
            self._t.join(timeout=self.poll_s + 2)
        self._sample()
        if self.out_path:
            Path(self.out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.out_path).write_text(json.dumps(self.report(), indent=2) + "\n")

    def report(self) -> dict:
        return {
            "gpu": self.gpu,
            "started_clean": self.started_clean,
            "contaminated": self.contaminated,
            "verdict": self.verdict(),
            "intruders": list(self.intruders.values()),
            "n_dirty_samples": len(self.samples),
        }

    def verdict(self) -> str:
        if self.started_clean is False:
            return "CONTAMINATED_AT_START"
        if self.intruders:
            return "CONTAMINATED_DURING_RUN"
        return "CLEAN"


def host_contention(exclude_gpu: int | None = None,
                    own_root: int | None = None) -> dict:
    """**整台機器**上有多少外來負載——不只自己那張卡。

    為什麼需要這個：`GpuWatcher` 只看目標 GPU 上有沒有別人的 process。
    但 GPU 的 SM 是各卡獨占的，**PCIe、host RAM 頻寬、/dev/shm 卻是全機共用**。
    別人在 GPU 1–6 上跑，不會出現在 GPU 0 的 compute-apps 裡，
    卻會實實在在地拖慢 GPU 0 上「把 KV 從 CPU 搬回來」的量測。

    實測（本專案 RUNLOG 發現 5）：五個 server 同時搶 PCIe 時，
    卸載 baseline 的 warm TTFT 被灌水 26–52%，而完全不碰 PCIe 的
    full_gpu 只差 ±2%。

    所以**任何量搬運成本的 run 都要把這個數字記進結果**，
    否則事後無法判斷該次量測可不可信。
    """
    own = _descendants(own_root if own_root is not None else os.getpid())
    apps = [a for a in compute_apps()
            if a["pid"] not in own and a["gpu"] != exclude_gpu]
    util = gpu_util()
    busy = sorted({a["gpu"] for a in apps})
    return {
        "foreign_procs": len(apps),
        "foreign_gpus": busy,
        "foreign_gpu_count": len(busy),
        "foreign_total_mib": sum(a["used_mib"] for a in apps),
        "foreign_max_util": max((util.get(g, 0) for g in busy), default=0),
        "mean_util_excl_self": (
            round(sum(v for k, v in util.items() if k != exclude_gpu)
                  / max(1, len(util) - (1 if exclude_gpu in util else 0)), 1)),
        "level": ("QUIET" if not busy
                  else "LIGHT" if max((util.get(g, 0) for g in busy), default=0) < 30
                  else "HEAVY"),
    }


def free_mib(gpu: int) -> int | None:
    """這張卡目前的可用記憶體（MiB）。"""
    for r in _smi("gpu=index,memory.free"):
        if len(r) >= 2 and int(r[0]) == gpu:
            return int(r[1])
    return None


def wait_until_free(gpu: int, need_mib: int, timeout_s: float = 300.0,
                    poll_s: float = 5.0, consecutive: int = 3
                    ) -> tuple[bool, int | None]:
    """等到這張卡**連續** `consecutive` 次取樣都有 need_mib 可用為止。

    為什麼不能只看 compute-apps：行程結束到 driver 把記憶體還回去之間有延遲。
    實測踩到兩次——`idle_gpus()` 說卡是空的，但 vLLM 啟動時看到
    `Free memory on device cuda:0 (8.51/23.68 GiB)` 而直接失敗。

    為什麼不能只取樣一次：釋放過程中的瞬間值會忽高忽低。2026-08-30 第二次踩到
    ——單次取樣通過了，等 vLLM 真的載完模型要配置 KV pool 時只剩 12.32 GiB。
    所以要連續數次都達標才算數。
    """
    t0 = time.time()
    hits = 0
    last = None
    while time.time() - t0 < timeout_s:
        last = free_mib(gpu)
        if last is not None and last >= need_mib:
            hits += 1
            if hits >= consecutive:
                return True, last
        else:
            hits = 0
        time.sleep(poll_s)
    return False, last


def idle_gpus(own_root: int | None = None) -> list[int]:
    busy = {a["gpu"] for a in compute_apps()
            if a["pid"] not in _descendants(own_root if own_root is not None else os.getpid())}
    n = len(_smi("gpu=index"))
    return [i for i in range(n) if i not in busy]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=int, help="檢查這張卡是否乾淨；乾淨回 0")
    ap.add_argument("--idle-gpus", action="store_true", help="印出乾淨的 GPU index")
    ap.add_argument("--watch", type=int, help="前景監看這張卡直到 Ctrl-C")
    ap.add_argument("--out")
    ap.add_argument("--interval", type=float, default=POLL_S)
    a = ap.parse_args()

    if a.idle_gpus:
        g = idle_gpus()
        print(" ".join(map(str, g)))
        return 0 if g else 1

    if a.check is not None:
        f = foreign_on(a.check)
        if f:
            print(f"GPU {a.check}: BUSY — {len(f)} foreign process(es)")
            for x in f:
                print(f"  pid={x['pid']} using {x['used_mib']} MiB")
            return 1
        print(f"GPU {a.check}: CLEAN")
        return 0

    if a.watch is not None:
        print(f"watching GPU {a.watch} every {a.interval}s — Ctrl-C to stop")
        with GpuWatcher(gpu=a.watch, poll_s=a.interval, out_path=a.out) as w:
            try:
                while True:
                    time.sleep(a.interval)
                    print(f"  {datetime.now():%H:%M:%S} verdict={w.verdict()} "
                          f"intruders={len(w.intruders)}")
            except KeyboardInterrupt:
                pass
        print(json.dumps(w.report(), indent=2))
        return 0 if not w.contaminated else 1

    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
