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


def smi_index(logical: int = 0) -> int:
    """torch 的 cuda:<logical> 在 nvidia-smi / amd-smi 裡是第幾張卡（GpuWatcher 要的是後者）。

    為什麼需要（2026-10-15，H200 Slurm 準備）：Slurm 用 CUDA_VISIBLE_DEVICES 指定 job 的卡。
    若節點沒有用 cgroup 限制裝置，nvidia-smi 會列出整台的卡，job 的 cuda:0 可能是 nvidia-smi 的 3 號；
    這時 GpuWatcher(gpu=0) 會去看別人的卡、對自己那張視而不見 → 污染被標成乾淨（規則 7）。
    對不上就丟 SmiUnavailable，不猜。AMD（MI300X 單卡容器）與沒設 CUDA_VISIBLE_DEVICES 時行為不變。
    """
    if vendor() == "amd":
        # 10/19 起容器有兩張 MI300X：HIP_VISIBLE_DEVICES 指定實體卡，amd-smi 照實體編號列全部的卡。
        # 實測 HIP 編號和 amd-smi 編號都依 PCI bus 排序、一一對應（0＝86:00.0、1＝a6:00.0；RUNLOG 15.1）。
        hv = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("ROCR_VISIBLE_DEVICES")
        if not hv:
            return logical
        ents = [e.strip() for e in hv.split(",") if e.strip()]
        if logical >= len(ents) or not ents[logical].isdigit():
            raise SmiUnavailable(f"HIP_VISIBLE_DEVICES={hv!r} 的第 {logical} 張卡對不到 amd-smi 的編號")
        return int(ents[logical])
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        return logical
    ents = [e.strip() for e in cvd.split(",") if e.strip()]
    if logical >= len(ents):
        raise SmiUnavailable(f"CUDA_VISIBLE_DEVICES={cvd!r} 沒有第 {logical} 張卡")
    rows = [r for r in _smi("gpu=index,uuid") if len(r) >= 2]
    e = ents[logical]
    if e.startswith(("GPU-", "MIG-")):
        hit = [int(r[0]) for r in rows if r[1].startswith(e)]
        if len(hit) == 1:
            return hit[0]
    elif e.isdigit():
        if any(int(r[0]) == int(e) for r in rows):
            return int(e)
        if len(rows) == len(ents):
            # cgroup 只露出 job 自己的卡，而兩邊的編號不同（一邊是節點上的編號、一邊是 job 內的）：
            # nvidia-smi 的第 logical 列就是它（多卡時兩邊都依 PCI 匯流排排序，env_h200.sh 設 CUDA_DEVICE_ORDER=PCI_BUS_ID）
            return int(rows[logical][0])
    raise SmiUnavailable(f"CUDA_VISIBLE_DEVICES={cvd!r} 的第 {logical} 張卡對不到 nvidia-smi 的卡：{rows}")


def _slurm_node_gpu_count() -> int | None:
    """這個 job 所在節點上總共幾張 GPU（scontrol 的 Gres）。查不到回 None（呼叫端必須當成「看不到」）。"""
    import re
    node = os.environ.get("SLURMD_NODENAME") or os.environ.get("SLURM_JOB_NODELIST") or os.uname().nodename
    try:
        out = _run(["scontrol", "show", "node", node])
    except SmiUnavailable:
        return None
    m = re.search(r"\bGres=(\S+)", out)
    if not m:
        return None
    # 例：Gres=gpu:h200:8(S:0-1)、Gres=gpu:8、Gres=gpu:a:2,gpu:b:2
    counts = [int(x) for x in re.findall(r"gpu(?::[^:,(]+)?:(\d+)", m.group(1))]
    return sum(counts) if counts else None


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


def _amd_val(x) -> float:
    """amd-smi JSON 的量是 {"value": v, "unit": u}。解析不了就丟例外，不可靜默略過。"""
    if isinstance(x, dict):
        v, u = x.get("value"), str(x.get("unit", "")).upper()
        if v in (None, "N/A"):
            raise SmiUnavailable(f"amd-smi 欄位無值：{x}")
        mult = {"B": 1, "KB": 1e3, "KIB": 1024, "MB": 1e6, "MIB": 2**20, "GB": 1e9, "GIB": 2**30}.get(u, 1)
        return float(v) * mult if u in ("B", "KB", "KIB", "MB", "MIB", "GB", "GIB") else float(v)
    return float(x)


def _amd_compute_apps() -> list[dict]:
    """AMD 後端（2026-09-15 於 MI300X 真機驗證並改寫）。

    真機上發現的三件事（舊版全部答錯）：
    1. amd-smi 的記憶體欄位是 {"value","unit"} 物件；舊版 int(dict) 丟例外後被
       `continue` 靜默略過 → 有行程卻回傳 [] → 被讀成「乾淨」（CLAUDE.md 規則 7）。
    2. **回報的 pid 是 host PID namespace 的**（實測容器內 pid 46730 ↔ amd-smi 3365442），
       所以不能用 pid 判斷「是不是自己的」。見 foreign_on() 的 AMD 分支。
    3. 容器內的監控工具（amdtop）也會出現，VRAM=0。
    """
    data = _amd_json(["process"])
    if not isinstance(data, list):
        raise SmiUnavailable(f"amd-smi process 輸出格式非預期：{str(data)[:200]}")
    rows: list[dict] = []
    for entry in data:
        gpu = int(entry["gpu"])
        for pr in entry.get("process_list") or []:
            info = pr.get("process_info", pr)
            if not isinstance(info, dict) or "pid" not in info:
                raise SmiUnavailable(f"amd-smi process 條目格式非預期：{str(pr)[:200]}")
            vram = _amd_val(info["memory_usage"]["vram_mem"])
            rows.append({"gpu": gpu, "pid": int(info["pid"]), "host_pid": True,
                         "used_mib": int(vram // 2**20)})
    return rows


def _container_kfd_holders() -> dict[int, str]:
    """本容器內開著 /dev/kfd 的行程 {pid: cmdline}。ROCm 的每個 GPU 行程都會開 kfd。"""
    out = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            fds = os.listdir(f"/proc/{d}/fd")
        except OSError:
            continue
        for fd in fds:
            try:
                if os.readlink(f"/proc/{d}/fd/{fd}") == "/dev/kfd":
                    with open(f"/proc/{d}/cmdline", "rb") as f:
                        out[int(d)] = f.read().replace(b"\0", b" ").decode(errors="replace").strip()[:120]
                    break
            except OSError:
                continue
    return out


# 容器內允許並存的監控工具（開 kfd 但不配置 VRAM）。出現時仍逐次記錄，只是不判為污染。
AMD_MONITOR_CMDS = ("amdtop", "rocm-smi", "amd-smi", "nvtop", "radeontop", "rocmtop", "amdgpu_top")


def _amd_gpu_data(flag: str) -> list[dict]:
    data = _amd_json(["metric", flag])
    items = data.get("gpu_data") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise SmiUnavailable(f"amd-smi metric {flag} 輸出格式非預期：{str(data)[:200]}")
    return items


def gpu_util() -> dict[int, int]:
    """每張卡目前的使用率（%）。"""
    if vendor() == "amd":
        return {int(e["gpu"]): int(_amd_val(e["usage"]["gfx_activity"])) for e in _amd_gpu_data("-u")}
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


def _holder_gpu(pid: int) -> int | None:
    """容器行程用哪張實體卡：ROCR_VISIBLE_DEVICES、HIP_VISIBLE_DEVICES 的第一個（實體編號，和 amd-smi 一致，RUNLOG 15.1）。
    讀不到或沒設 → None。HIP 編號在只有 ROCR 時是邏輯編號，所以 ROCR 優先。"""
    try:
        env = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
    except OSError:
        return None
    for key in (b"ROCR_VISIBLE_DEVICES=", b"HIP_VISIBLE_DEVICES="):
        for e in env:
            if e.startswith(key):
                v = e[len(key):].decode(errors="replace").split(",")[0].strip()
                return int(v) if v.isdigit() else None
    return None


def foreign_on(gpu: int, own_root: int | None = None) -> list[dict]:
    """gpu 上不屬於我們的 compute process。own_root 預設為本行程。"""
    own = _descendants(own_root if own_root is not None else os.getpid())
    if vendor() != "amd":
        return [a for a in compute_apps() if a["gpu"] == gpu and a["pid"] not in own]
    # AMD：amd-smi 給 host pid，對不上容器 pid → 改用「數量」判斷。
    #   有配置 VRAM 的 amd-smi 行程數  −  本容器內開 /dev/kfd 的行程數  =  容器外的行程數（下界）
    # VRAM=0 的條目（真機上的 host pid 8110，早於本容器任何 GPU 程式就存在、gfx 用量 0）
    # 不佔記憶體也不佔算力，只記錄不判污染。
    # ⚠️ 已知盲點：若本容器某行程已開 kfd 但尚未配置 VRAM，同時外面有一個配置了 VRAM 的行程，
    #    兩者相抵會漏報。開跑前（本 run 尚無 GPU 行程）的檢查不受此影響，是精確的。
    apps = [a for a in compute_apps() if a["gpu"] == gpu]
    holders = _container_kfd_holders()
    # 10/19 兩張卡：容器裡在「別張卡」的行程（環境變數 ROCR／HIP_VISIBLE_DEVICES 指到別張）不是這張卡的入侵者，
    # 也不能拿來抵這張卡上的外部行程。看不出用哪張卡的（沒設環境變數）照舊算在這張（保守）。
    holders = {pid: cmd for pid, cmd in holders.items() if _holder_gpu(pid) in (None, gpu)}
    out = []
    n_external = sum(1 for a in apps if a["used_mib"] > 0) - len(holders)
    if n_external > 0:
        out.append({"pid": -1, "kind": "external_namespace", "count": n_external,
                    "used_mib": sum(a["used_mib"] for a in apps)})
    for pid, cmd in holders.items():
        if pid in own or any(m in cmd for m in AMD_MONITOR_CMDS):
            continue
        out.append({"pid": pid, "kind": "same_container", "cmd": cmd, "used_mib": -1})
    return out


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
    paused_intervals: list[dict] = field(default_factory=list)
    discarded_samples: int = 0
    _pause_gen: int = 0
    _paused: threading.Event = field(default_factory=threading.Event)
    _stop: threading.Event = field(default_factory=threading.Event)
    _t: threading.Thread | None = None

    @property
    def contaminated(self) -> bool:
        return bool(self.intruders) or self.started_clean is False

    def pause(self, reason: str):
        """自己的 server 關閉期間暫停取樣。

        2026-09-15 MI300X 實測：vLLM 行程結束、容器內 kfd 已關之後，amd-smi（host 端）仍回報
        那 180 GB 約數秒 → 數量判斷把「自己剛關掉的 server」算成容器外行程 → 假污染。
        暫停區間逐筆記進報告，不是靜默略過。
        """
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            t0 = datetime.now().astimezone().isoformat()
            self._pause_gen += 1
            self._paused.set()
            try:
                yield
            finally:
                self._paused.clear()
                self.paused_intervals.append({"from": t0, "to": datetime.now().astimezone().isoformat(),
                                              "reason": reason})
        return _cm()

    def _sample(self) -> None:
        if self._paused.is_set():
            return
        gen = self._pause_gen
        f = foreign_on(self.gpu, self.own_root)
        # 2026-09-15 競態：取樣開始後 12 ms 才進入暫停，這一筆讀到正在關閉的自家 server。
        # 取樣期間只要碰到暫停（旗標仍在，或暫停次數變了），整筆丟棄並記錄。
        if self._paused.is_set() or gen != self._pause_gen:
            self.discarded_samples += 1
            return
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
            "paused_intervals": self.paused_intervals,
            "discarded_samples_overlapping_pause": self.discarded_samples,
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
    if vendor() == "amd":
        # 單一 VF 容器：看得到的只有這張卡。同一台 host 上其他 pod 對 PCIe root complex /
        # host 記憶體頻寬的負載**觀測不到** → 不能回報 QUIET（規則 7）。
        util = gpu_util()
        return {"foreign_procs": 0, "foreign_gpus": [], "foreign_gpu_count": 0,
                "foreign_total_mib": 0, "foreign_max_util": 0,
                "mean_util_excl_self": None, "visible_gpus": sorted(util),
                "level": "UNOBSERVABLE_HOST"}
    if os.environ.get("SLURM_JOB_ID"):
        # Slurm job（平台 C，H200）：裝置被 cgroup 限制時，nvidia-smi 只列得出 job 自己的卡，
        # 同節點別人的 job 完全看不到 → 和 AMD 容器同一個處理，不能回 QUIET（規則 7，H200_SLURM §5.3）。
        # 節點卡數查不到（scontrol 失敗、沒有 Gres）也當成看不到。
        visible = len(_smi("gpu=index"))
        node_gpus = _slurm_node_gpu_count()
        if node_gpus is None or visible < node_gpus:
            return {"foreign_procs": 0, "foreign_gpus": [], "foreign_gpu_count": 0,
                    "foreign_total_mib": 0, "foreign_max_util": 0,
                    "mean_util_excl_self": None, "visible_gpus": visible, "node_gpus": node_gpus,
                    "level": "UNOBSERVABLE_HOST"}
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
    if vendor() == "amd":
        for e in _amd_gpu_data("-m"):
            if int(e["gpu"]) == gpu:
                return int(_amd_val(e["mem_usage"]["free_vram"]) // 2**20)
        return None
    for r in _smi("gpu=index,memory.free"):
        if len(r) >= 2 and int(r[0]) == gpu:
            return int(r[1])
    return None


def wait_until_released(gpu: int, timeout_s: float = 300.0, poll_s: float = 2.0,
                        consecutive: int = 3, idle_used_mib: int = 2048) -> tuple[bool, int | None]:
    """等到 GPU 上**沒有任何行程持有 VRAM**（AMD：amd-smi 裡 VRAM>0 的條目為 0），連續數次。
    用在自己的 server 關掉之後，確認 host 端也真的收回了記憶體。"""
    t0, hits, last = time.time(), 0, None
    while time.time() - t0 < timeout_s:
        if vendor() == "amd":
            last = sum(a["used_mib"] for a in compute_apps() if a["gpu"] == gpu)
        else:
            last = sum(a["used_mib"] for a in compute_apps() if a["gpu"] == gpu)
        if last <= idle_used_mib:
            hits += 1
            if hits >= consecutive:
                return True, last
        else:
            hits = 0
        time.sleep(poll_s)
    return False, last


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
    if vendor() == "amd":
        gpus = sorted(int(e["gpu"]) for e in _amd_gpu_data("-m"))
        return [g for g in gpus if not foreign_on(g, own_root)]
    busy = {a["gpu"] for a in compute_apps()
            if a["pid"] not in _descendants(own_root if own_root is not None else os.getpid())}
    n = len(_smi("gpu=index"))
    return [i for i in range(n) if i not in busy]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=int, help="檢查這張卡是否乾淨；乾淨回 0")
    ap.add_argument("--idle-gpus", action="store_true", help="印出乾淨的 GPU index")
    ap.add_argument("--watch", type=int, help="前景監看這張卡直到 Ctrl-C")
    ap.add_argument("--host", action="store_true", help="印出整機的 host_contention（JSON）")
    ap.add_argument("--out")
    ap.add_argument("--interval", type=float, default=POLL_S)
    a = ap.parse_args()

    if a.idle_gpus:
        g = idle_gpus()
        print(" ".join(map(str, g)))
        return 0 if g else 1

    if a.host:
        print(json.dumps(host_contention(), indent=2))
        return 0

    if a.check is not None:
        f = foreign_on(a.check)
        if f:
            print(f"GPU {a.check}: BUSY — {len(f)} foreign process(es)")
            for x in f:
                print(f"  pid={x['pid']} using {x['used_mib']} MiB")
            return 1
        print(f"GPU {a.check}: CLEAN")
        if os.environ.get("SLURM_JOB_ID"):   # Slurm 裡「這張卡乾淨」不代表整機乾淨（H200_SLURM §5.3）
            print(f"host: {host_contention(exclude_gpu=a.check)['level']}")
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
