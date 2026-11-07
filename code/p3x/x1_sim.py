"""x1_sim.py — X1：延後設計下，把 oracle 拆成四個元件，一次開一個，對上「機制相同、只差資訊」的對手。

判準、模型、對手清單寫死在 docs/phase3_20261017/explore/X1_oracle_decomposition.md §2（開跑前，2026-10-17T09:52:26Z）。
不碰 GPU（CUDA_VISIBLE_DEVICES=""）。不改任何既有檔：import m8_trace_oracle.load_f／f_of（f(i)），
自檢時 import m7_sim.sim_restore 與 m7_restore_harness.Tier 對拍。

子命令：
  selftest  §2.8 的不變量 1–4（任一條不過就 exit 1）
  sweep     §2.5 的格子 × §2.4 的策略，多行程；每個 job 的逐請求 TTFT 存到 $RUN_DIR/*.npz
  analyze   §2.6–2.7 的 headroom、判定、Cake／只載入比值、oracle 是否不慢於對手

模型摘要（細節見 §2.2）：
  - trace 時間、一次一個請求、GPU 不排隊；CPU 層與慢層各一條 I/O 線（讀寫分開排隊，寫入中讀取 ×k_read）。
  - 回合結束：session 的完整 block 全部 BF16 進 CPU；CPU 滿就依策略降級（寫 INT4）或丟。refcount 共用前綴。
  - 回來時：Cake（或只載入）還原有存的前綴，其餘重算；INT4 載入＝位元組÷頻寬＋反量化 d。
"""
from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import heapq
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict, deque
from datetime import datetime

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
REPO = os.path.dirname(CODE)
sys.path.insert(0, HERE)
sys.path.insert(0, CODE)

import numpy as np  # noqa: E402

import x1_trace as XT  # noqa: E402

# P3_CONST（code/p3_platform.py）：沒設＝MI300X（原本）；h200＝讀 results/*_h200 的同一批實測、寫 results/p3_explore_h200/x1
import p3_platform as PF  # noqa: E402
RES7 = PF.rdir("m7_write_policy_mi300x")
OUTDIR = PF.explore_out("x1")
MiB, GiB = 1 << 20, 1 << 30
BF = 64 * MiB                    # Llama-3.1-8B BF16 block
INF = XT.INF
CPU_GIB = [128, 256, 512, 1024]
SLOWS = ["ssd", "b32", "nfs"]
GPUS = ["idle", "busy"]
MODES = ["cake", "load"]
ALL_VIEWS = [(s, g, m) for s in SLOWS for g in GPUS for m in MODES]
TRACES = ["conversation", "toolagent"]

S_NONE, S_CPU, S_SLOW, S_DROP = 0, 1, 2, 3
# ICPE T3（docs/h200/icpe/T3_planner_sensitivity.md）：X1_PLANNER=stair → mode＝cake 的還原改用 STAIR（照 CacheFlow 的想法自己寫的簡化版）；
# 沒設＝原本的 Cake（逐位元不變）。
PLANNER = os.environ.get("X1_PLANNER", "cake")
if PLANNER not in ("cake", "stair"):
    raise SystemExit(f"X1_PLANNER={PLANNER!r}：只接受 cake 或 stair")
STAIR_NLAY = 32                  # Llama-3.1-8B 的層數：會合點的粒度＝1/32 塊

# ---- ICPE next（icpe-sim，2026-11-04；docs/icpe_brief_20261030/next/SIM_CHANGES.md）---------------------------------
# GPU 快取層（GPU→CPU→慢層）。GPU 預算＝0（預設）時完全走原本的程式路徑、輸出欄位不變（逐位元相同）。
#   寫入方式 wb＝回寫、互斥：回合結束只留在 GPU；GPU 滿才把被選中的 session 以 BF16 寫進 CPU；session 回到 GPU 時 CPU 那份釋放。
#            wt＝直寫、包含：CPU 層完全照原本（回合結束照樣寫 CPU）；GPU 另外留一份，逐出不用寫。
#   GPU 規則 LRU；BEL（完美預測：先擠出下次最晚回來的）；TTLA:<秒>（釘住，到期就逐出）；TTLB:<秒>（釘住，到期後照 LRU）。
#   粒度＝session（共用前綴照 refcount 只存一份）。TTL 從回合結束（＝這個 view 的第一個 token）起算 → 一次只能跑一種時間設定。
S_GPU = 4
GPU_WRITES = ("wb", "wt")
EXTRA_SLOWS = []                 # sweep 的 --slows 裡不在 SLOWS 的等級；constants() 只加這裡列的（預設空＝原本的常數）
GFIO_CSV = os.path.join(REPO, "results", "s3_h200", "gfio", "gfio_20261026-104426-s3-gfio.csv")


def parse_gpol(spec):
    """GPU 規則字串 → (順序, TTL 秒或 None)。"""
    if spec in ("LRU", "BEL"):
        return spec.lower(), None
    head, _, val = spec.partition(":")
    if head in ("TTLA", "TTLB") and val:
        ttl = float(val)
        assert ttl >= 0, spec
        return head.lower(), ttl
    raise SystemExit(f"GPU 規則 {spec!r}：只接受 LRU、BEL、TTLA:<秒>、TTLB:<秒>")


def extra_slow(name, tp):
    """可選的慢層等級（不改 ssd／b32／nfs）。nvme807＝H200 計算節點本地 NVMe 實測（G-fio：/tmp、64 MiB 循序讀、8 執行緒、6 次中位數）；
    寫＝讀 × SSD 級的寫讀比（NEXT_PLAN §6b 草案）；k_read 沿用 SSD 級。"""
    if name != "nvme807":
        raise SystemExit(f"慢層 {name!r}：只接受 {SLOWS} 與 nvme807")
    rows = [x for x in csv.DictReader(open(GFIO_CSV)) if x["dir"].startswith("/tmp/") and x["pattern"] == "seq"
            and int(x["io_bytes"]) == 64 * MiB and int(x["threads"]) == 8]
    assert len(rows) == 1, f"{GFIO_CSV}：/tmp 64 MiB 循序 8 執行緒的列有 {len(rows)} 列"
    x = rows[0]
    reps = sorted(float(v) for v in x["gibps_reps"].split(";"))
    med = (reps[(len(reps) - 1) // 2] + reps[len(reps) // 2]) / 2
    r = float(x["gibps"])
    assert len(reps) == int(x["n_rep"]) and abs(med - r) < 1e-3, f"gibps {r} 不是 gibps_reps 的中位數 {med}"   # 用 B 欄驗 A 欄
    loc = tp["local"]
    return dict(r=r, w=r * loc["write_GiBps"] / loc["read_GiBps"], k=loc["k_read"], src_run=x["run_id"])

# 策略：(順序, 被選中時的動作, 回合結束的 hook, 預取)
POLICIES = {
    "LRU": ("lru", "demote", None, None),
    "PRED": ("pred", "demote", None, None),
    "LRU2": ("lru2", "demote", None, None),
    "ARC": ("arc", "demote", None, None),
    "S3FIFO": ("s3", "demote", None, None),
    "GDSF": ("gdsf", "demote", None, None),
    "PENS": ("pens", "demote", None, None),
    "LRU+NONE": ("lru", "none", None, None),
    "LRU+2HW": ("lru", "2hw", None, None),
    "LRU+2HA": ("lru", "demote", "2ha", None),
    "LRU+RPF": ("lru", "demote", None, "rpf"),
    "O-BEL": ("bel", "demote", None, None),
    "O-BELS": ("bels", "demote", None, None),
    "O-DROP": ("lru", "demote", "odrop", None),
    "O-NOW": ("lru", "onow", None, None),
    "O-PF": ("lru", "demote", None, "opf"),
    "O-ALL": ("bel", "onow", "odrop", "opf"),
}
# 事後穩健性（X9 提醒，2026-10-17 晚）：「之後再也沒人讀」只用在離 trace 結束還有 ≥H 秒的決定；
# 最後 H 秒內的決定一律當「會回來」（照 LRU 機制保留／照寫）。不改原登記判定。
# 事後加入、只報告的對手（X2 提醒，2026-10-17 晚）：session 版 2Q 式規則＝先比到目前為止的請求數、再比閒置時間（LRU）
POLICIES["FREQ2Q"] = ("freq", "demote", None, None)
for _h in (1800, 3600):
    for _p in ("O-DROP", "O-NOW", "O-ALL"):
        POLICIES[f"{_p}-H{_h}"] = POLICIES[_p]
# §2.4 的對稱對手集合
RIVALS = {
    "O-ORD": ["LRU", "LRU2", "ARC", "S3FIFO", "GDSF", "PENS"],
    "O-DROP": ["LRU", "LRU+NONE", "LRU+2HW", "LRU+2HA"],
    "O-NOW": ["LRU", "LRU+NONE", "LRU+2HW"],
    "O-PF": ["LRU", "LRU+RPF"],
}
COMPONENT_POLICIES = {"O-ORD": ["O-BEL", "O-BELS"], "O-DROP": ["O-DROP"], "O-NOW": ["O-NOW"], "O-PF": ["O-PF"]}
NON_ORACLE = ["LRU", "LRU2", "ARC", "S3FIFO", "GDSF", "PENS", "LRU+NONE", "LRU+2HW", "LRU+2HA", "LRU+RPF"]


def run_id():
    return os.environ.get("RUN_ID", "adhoc")


def now_ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def code_sha():
    h = hashlib.sha256()
    for p in (__file__, XT.__file__):
        h.update(open(p, "rb").read())
    return h.hexdigest()[:12]


# ================================================================== 常數〔MI300X〕
def constants():
    sys.path.insert(0, CODE)
    PF.check_m7_res()                            # load_f 讀 $M7_RES/calib_c1.csv：要和其他常數同一台
    from m8_trace_oracle import load_f
    fmeas, slope, icpt, frun = load_f()          # ms，C1 中位數，i ≥ 80 線性外插
    busy = json.load(open(os.path.join(RES7, "busy_levels.json")))
    bfac = float(busy["_measured_slowdown"]["busy2"])
    tp = json.load(open(PF.TIER_PARAMS))         # 限速層：兩台都用 MI300X 的（03 §R2）；mi300x 時就是 RES7 那份
    bw = json.load(open(PF.rdir("m10_precision_mi300x", "speed_llama", "bw.json")))
    iow = json.load(open(PF.IOWALL))             # 每 block 位元組＝格式定義，和 GPU 無關（下一行斷言 BF16）
    i4 = int(iow["llama"]["bytes_per_block"]["int4"])
    assert abs(iow["llama"]["bytes_per_block"]["bf16"] - BF) < 1
    import pandas as pd
    d = pd.read_csv(PF.rdir("m10_precision_mi300x", "speed_llama", "restores.csv.gz"))
    rid4 = PF.run("20261016-034902-m10-grid-int4-llama")
    g = d[(d.run_id == rid4) & (d.config == "E1") & (d["mode"] == "load_only") & (d.bw_name == "cpu") & (d.fp8_read > 0)]
    dq = (g.t_dequant / g.fp8_read).groupby(g.gpu_state).median()
    b32 = float(bw["targets"]["b32"]["GiBps"])
    K = dict(
        f_ms=[float(x) for x in fmeas], f_slope=float(slope), f_icpt=float(icpt), f_runs=list(frun), busy_factor=bfac,
        int4_bytes=i4, dq_idle=float(dq["idle"]), dq_busy=float(dq["busy"]), dq_run=rid4,
        cpu=dict(r=tp["cpu"]["read_GiBps"], w=tp["cpu"]["write_GiBps"], k=tp["cpu"]["k_read"]),
        slow=dict(ssd=dict(r=tp["local"]["read_GiBps"], w=tp["local"]["write_GiBps"], k=tp["local"]["k_read"]),
                  b32=dict(r=b32, w=b32, k=1.0),
                  nfs=dict(r=tp["nfs"]["read_GiBps"], w=tp["nfs"]["write_GiBps"], k=tp["nfs"]["k_read"])))
    for nm in EXTRA_SLOWS:                       # ICPE next：可選的慢層等級（預設沒有）
        K["slow"][nm] = extra_slow(nm, tp)
    return K


def f_array(K, nmax, gpu):
    """位置 0..nmax-1 的重算時間（秒）。"""
    fm = K["f_ms"]
    out = []
    for i in range(nmax):
        ms = fm[i] if i < len(fm) else K["f_icpt"] + K["f_slope"] * i
        out.append(ms / 1e3 * (K["busy_factor"] if gpu == "busy" else 1.0))
    return out


# ================================================================== I/O 線（＝m7_restore_harness.Tier 的 share 模式，無鎖、無 c_s）
class Chan:
    __slots__ = ("name", "r", "w", "k", "t_free", "t_free_w", "busy_r", "busy_w", "bytes_r", "bytes_w")

    def __init__(self, name, r_GiBps, w_GiBps, k):
        self.name = name
        self.r = r_GiBps * GiB
        self.w = w_GiBps * GiB
        self.k = k
        self.t_free = 0.0
        self.t_free_w = 0.0
        self.busy_r = self.busy_w = 0.0
        self.bytes_r = self.bytes_w = 0

    def read(self, nbytes, now):
        st = now if now > self.t_free else self.t_free
        dur = nbytes / self.r
        if self.t_free_w > st:
            dur *= self.k
        self.t_free = st + dur
        self.busy_r += dur
        self.bytes_r += nbytes
        return self.t_free

    def write(self, nbytes, now):
        if nbytes <= 0:
            return self.t_free_w
        st = now if now > self.t_free_w else self.t_free_w
        dur = nbytes / self.w
        self.t_free_w = st + dur
        self.busy_w += dur
        self.bytes_w += nbytes
        return self.t_free_w


class View:
    """一種時間設定：(慢層, GPU 狀態, 還原方式)。各自有自己的 I/O 線與逐請求結果。"""

    def __init__(self, K, slow, gpu, mode, nmax, nreq, slow_bytes=None, dq=None):
        self.key = (slow, gpu, mode)
        self.slow_name, self.gpu, self.mode = slow, gpu, mode
        self.planner = PLANNER
        self.f = f_array(K, nmax, gpu)
        self.Fc = [0.0]
        for x in self.f:
            self.Fc.append(self.Fc[-1] + x)
        self.i4 = K["int4_bytes"] if slow_bytes is None else slow_bytes
        self.d = (K["dq_idle"] if gpu == "idle" else K["dq_busy"]) if dq is None else dq
        c = K["cpu"]
        s = K["slow"][slow]
        self.cpu = Chan("cpu", c["r"], c["w"], c["k"])
        self.slow = Chan("slow_" + slow, s["r"], s["w"], s["k"])
        self.ttft = np.zeros(nreq)
        self.rest = np.zeros(nreq)
        self.nC = np.zeros(nreq, np.int32)
        self.nS = np.zeros(nreq, np.int32)
        self.nR = np.zeros(nreq, np.int32)
        self.pf_bytes = 0
        self._mcpu = {}

    def m_cpu(self, n):
        """n 個 block 全部以 INT4 放在 CPU（沒有爭用）時 Cake 的會合點（預取讀到這裡為止）。"""
        m = self._mcpu.get(n)
        if m is None:
            l = self.i4 / self.cpu.r + self.d
            p, q, rt, lt = 0, n - 1, 0.0, 0.0
            while p <= q:
                if rt <= lt:
                    rt += self.f[p]
                    p += 1
                else:
                    lt += l
                    q -= 1
            m = self._mcpu[n] = p
        return m

    def _stair(self, lay, blocks, L, t0, pf_done):
        """T3 §1 的 STAIR：會合點可以落在一塊的中間（1/STAIR_NLAY 塊）、CPU 線與慢層線同時載入（合計受 GPU 輸入頻寬＝CPU 線讀取頻寬限制）。
        用名目成本選會合點，再用原本的 Chan.read 執行。回傳 (結束時間, 整塊重算的塊數, nC, nS)。"""
        f, cpu, slow, d, i4, Fc = self.f, self.cpu, self.slow, self.d, self.i4, self.Fc
        sc0, ss0 = max(t0, cpu.t_free), max(t0, slow.t_free)
        mc = cpu.k if cpu.t_free_w > sc0 else 1.0
        ms = slow.k if slow.t_free_w > ss0 else 1.0
        lc_blk, lc_i4, ls_blk = BF / cpu.r * mc, i4 / cpu.r * mc + d, i4 / slow.r * ms + d
        # 從最後一塊往前載入到第 m 塊時，各線的名目完成時間（相對 t0；CPU 線要等預取完成 pf_done）與位元組
        bc, bs, rin = sc0 - t0, ss0 - t0, cpu.r
        Tc, Ts, Sb = [bc] * (L + 1), [bs] * (L + 1), [0.0] * (L + 1)
        hc, hs = [False] * (L + 1), [False] * (L + 1)
        rd = [0.0] * L
        for i in range(L - 1, -1, -1):
            k = lay[i]
            Tc[i], Ts[i], Sb[i], hc[i], hs[i] = Tc[i + 1], Ts[i + 1], Sb[i + 1], hc[i + 1], hs[i + 1]
            if k == 1:
                Tc[i] += lc_blk; Sb[i] += BF; hc[i] = True                      # noqa: E702
            elif k == 2:
                rd[i] = pf_done[blocks[i]] - t0
                Tc[i] = max(Tc[i], rd[i]) + lc_i4; Sb[i] += i4; hc[i] = True    # noqa: E702
            else:
                Ts[i] += ls_blk; Sb[i] += i4; hs[i] = True                      # noqa: E702

        def load(m):
            return max(Tc[m] if hc[m] else 0.0, Ts[m] if hs[m] else 0.0, Sb[m] / rin)
        m = 0
        while m <= L and Fc[m] < load(m):
            m += 1
        if m == 0:
            j, x = -1, 0.0                             # 全部載入
        elif m > L:
            j, x = L - 1, 1.0                          # 全部重算
        else:
            j = m - 1                                  # 第 j 塊裡的某一層交會
            k = lay[j]
            a = (lc_blk if k == 1 else lc_i4 if k == 2 else ls_blk)
            c0 = (max(Tc[j + 1], rd[j]) if k == 2 else Tc[j + 1]) if (hc[j + 1] or k in (1, 2)) else 0.0
            comps = [(c0, a if k in (1, 2) else 0.0),
                     (Ts[j + 1] if (hs[j + 1] or k == 3) else 0.0, a if k == 3 else 0.0),
                     (Sb[j + 1] / rin, (BF if k == 1 else i4) / rin)]
            xs = [(A0 + aa - Fc[j]) / (f[j] + aa) for A0, aa in comps if f[j] + aa > 0]
            xr = min(1.0, max(0.0, max(xs) if xs else 0.0))

            def obj(xx):
                return max(Fc[j] + xx * f[j], *(A0 + (1 - xx) * aa for A0, aa in comps))
            lo = math.floor(xr * STAIR_NLAY) / STAIR_NLAY
            hi = math.ceil(xr * STAIR_NLAY) / STAIR_NLAY
            x = lo if obj(lo) <= obj(hi) else hi
        rt = t0 + (Fc[j] + x * f[j] if j >= 0 else 0.0)
        lc = ls = t0
        nbytes = 0.0
        nC = nS = 0
        for i in range(L - 1, j - 1 if j >= 0 else -1, -1):
            w = 1.0 if i > j else 1.0 - x
            if w <= 0:
                continue
            k = lay[i]
            if k == 1:
                lc = cpu.read(BF * w, lc)
                nbytes += BF * w
                nC += 1
            elif k == 2:
                pd = pf_done[blocks[i]]
                lc = cpu.read(i4 * w, lc if lc > pd else pd) + d * w
                nbytes += i4 * w
                nC += 1
            else:
                ls = slow.read(i4 * w, ls) + d * w
                nbytes += i4 * w
                nS += 1
        end = max(rt, lc, ls, t0 + nbytes / rin)
        return end, (j + 1 if x >= 1.0 else j) if j >= 0 else 0, nC, nS

    def restore(self, lay, blocks, L, n_all, t0, old_n, pf_done, g=0, gpf_done=None):
        """lay[i]∈{1: CPU BF16, 2: CPU INT4（預取）, 3: 慢層 INT4}，i<L。回傳 (TTFT, 還原舊 context, nC, nS, 重算數)。
        ICPE next：lay[0:g]＝0（已在 GPU，不用還原）；Cake 從第 g 塊開始重算、從最後一塊往前載入。g＝0 時和原本逐位元相同。
        4＝預取進 GPU 的塊（已反量化成 BF16）：不用載入，只等預取完成（gpf_done）。
        加速（ICPE next；結果逐位元相同，自檢 T10 對 restore_ref）：Chan.read 內聯成區域變數，同樣的浮點運算、同樣的順序；
        還原期間沒有寫入，所以 t_free_w 固定；每塊的 BF／r、(BF／r)×k 先算好（同一個除法、同一個值）。"""
        f, cpu, slow, d, i4 = self.f, self.cpu, self.slow, self.d, self.i4
        nC = nS = 0
        if self.mode == "cake" and self.planner == "stair":
            assert g == 0 and gpf_done is None, "STAIR 不支援 GPU 層"
            end, meet, nC, nS = self._stair(lay, blocks, L, t0, pf_done)
        else:
            cf, cwf, cb, cn = cpu.t_free, cpu.t_free_w, cpu.busy_r, cpu.bytes_r
            sf, swf, sb, sn = slow.t_free, slow.t_free_w, slow.busy_r, slow.bytes_r
            cB = BF / cpu.r
            cBk = cB * cpu.k
            c4 = i4 / cpu.r
            c4k = c4 * cpu.k
            s4 = i4 / slow.r
            s4k = s4 * slow.k
            if self.mode == "cake":
                p, q, rt, lt = g, L - 1, t0, t0
                while p <= q:
                    if rt <= lt:
                        rt += f[p]
                        p += 1
                    else:
                        k = lay[q]
                        if k == 1:
                            st = lt if lt > cf else cf
                            dur = cBk if cwf > st else cB
                            cf = st + dur
                            cb += dur
                            cn += BF
                            lt = cf
                            nC += 1
                        elif k == 3:
                            st = lt if lt > sf else sf
                            dur = s4k if swf > st else s4
                            sf = st + dur
                            sb += dur
                            sn += i4
                            lt = sf + d
                            nS += 1
                        elif k == 2:
                            pd = pf_done[blocks[q]]
                            nw = lt if lt > pd else pd
                            st = nw if nw > cf else cf
                            dur = c4k if cwf > st else c4
                            cf = st + dur
                            cb += dur
                            cn += i4
                            lt = cf + d
                            nC += 1
                        else:              # 4
                            pd = gpf_done[blocks[q]]
                            lt = lt if lt > pd else pd
                        q -= 1
                end = rt if rt > lt else lt
                meet = p
            else:
                lt = t0
                for i in range(g, L):
                    k = lay[i]
                    if k == 1:
                        st = lt if lt > cf else cf
                        dur = cBk if cwf > st else cB
                        cf = st + dur
                        cb += dur
                        cn += BF
                        lt = cf
                        nC += 1
                    elif k == 3:
                        st = lt if lt > sf else sf
                        dur = s4k if swf > st else s4
                        sf = st + dur
                        sb += dur
                        sn += i4
                        lt = sf + d
                        nS += 1
                    elif k == 2:
                        pd = pf_done[blocks[i]]
                        nw = lt if lt > pd else pd
                        st = nw if nw > cf else cf
                        dur = c4k if cwf > st else c4
                        cf = st + dur
                        cb += dur
                        cn += i4
                        lt = cf + d
                        nC += 1
                    else:                  # 4
                        pd = gpf_done[blocks[i]]
                        lt = lt if lt > pd else pd
                end = lt
                meet = g
            cpu.t_free, cpu.busy_r, cpu.bytes_r = cf, cb, cn
            slow.t_free, slow.busy_r, slow.bytes_r = sf, sb, sn
        self.last_end = end
        Fc = self.Fc
        base = end - t0
        ttft = base + Fc[n_all] - Fc[L]
        rest = base + (Fc[old_n] - Fc[L] if old_n > L else 0.0)
        return ttft, rest, nC, nS, meet - g + (n_all - L)

    def restore_ref(self, lay, blocks, L, n_all, t0, old_n, pf_done, g=0, gpf_done=None):
        """（參考實作：ICPE next 加速前的版本，只給自檢對拍用；正式路徑用 restore。）
        lay[i]∈{1: CPU BF16, 2: CPU INT4（預取）, 3: 慢層 INT4}，i<L。回傳 (TTFT, 還原舊 context, nC, nS, 重算數)。
        ICPE next：lay[0:g]＝0（已在 GPU，不用還原）；Cake 從第 g 塊開始重算、從最後一塊往前載入。g＝0 時和原本逐位元相同。
        4＝預取進 GPU 的塊（已反量化成 BF16）：不用載入，只等預取完成（gpf_done）。"""
        f, cpu, slow, d, i4 = self.f, self.cpu, self.slow, self.d, self.i4
        nC = nS = 0
        if self.mode == "cake" and self.planner == "stair":
            assert g == 0 and gpf_done is None, "STAIR 不支援 GPU 層"
            end, meet, nC, nS = self._stair(lay, blocks, L, t0, pf_done)
        elif self.mode == "cake":
            p, q, rt, lt = g, L - 1, t0, t0
            while p <= q:
                if rt <= lt:
                    rt += f[p]
                    p += 1
                else:
                    k = lay[q]
                    if k == 1:
                        lt = cpu.read(BF, lt)
                        nC += 1
                    elif k == 2:
                        pd = pf_done[blocks[q]]
                        lt = cpu.read(i4, lt if lt > pd else pd) + d
                        nC += 1
                    elif k == 4:
                        pd = gpf_done[blocks[q]]
                        lt = lt if lt > pd else pd
                    else:
                        lt = slow.read(i4, lt) + d
                        nS += 1
                    q -= 1
            end = rt if rt > lt else lt
            meet = p
        else:
            lt = t0
            for i in range(g, L):
                k = lay[i]
                if k == 1:
                    lt = cpu.read(BF, lt)
                    nC += 1
                elif k == 2:
                    pd = pf_done[blocks[i]]
                    lt = cpu.read(i4, lt if lt > pd else pd) + d
                    nC += 1
                elif k == 4:
                    pd = gpf_done[blocks[i]]
                    lt = lt if lt > pd else pd
                else:
                    lt = slow.read(i4, lt) + d
                    nS += 1
            end = lt
            meet = g
        self.last_end = end
        Fc = self.Fc
        base = end - t0
        ttft = base + Fc[n_all] - Fc[L]
        rest = base + (Fc[old_n] - Fc[L] if old_n > L else 0.0)
        return ttft, rest, nC, nS, meet - g + (n_all - L)


# ================================================================== 慢層容量上限的淘汰規則（ICPE next 第二批；PREREG §2 第 5 項）
SLOW_POLICIES = ["LRU", "LRU2", "ARC", "S3FIFO", "GDSF", "PENS", "BEL"]


class SlowOrder:
    """慢層狀態（S_SLOW）的 session 的淘汰順序。「進入慢層」（被 CPU 降級）＝一次存取；session 回來（離開慢層）＝一次命中
    （ARC、S3FIFO 用它當頻率訊號）；LRU2、PENS、GDSF、BEL 用真實的請求紀錄／成本（同 CPU 層的同名規則）。"""

    def __init__(self, eng, rule, cap_bytes, gdsf_view=None):
        assert rule in SLOW_POLICIES, rule
        self.e, self.rule, self.C = eng, rule, float(cap_bytes)
        self.mem = OrderedDict()          # LRU 順序（也當成員集合）
        self.hits = {}                    # session → 回來（命中）次數
        self.T1, self.T2, self.B1, self.B2 = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
        self.t1b = self.t2b = self.b1b = self.b2b = 0
        self.p = 0.0
        self.S, self.M, self.Gh = OrderedDict(), OrderedDict(), OrderedDict()
        self.f, self.sz = {}, {}
        self.sb_ = self.mb_ = 0
        self.gH, self.gver, self.gheap, self.gL = {}, {}, [], 0.0
        if rule == "GDSF":
            v = gdsf_view
            K = eng.K
            ls = eng.i4 / (K["slow"][v.slow_name]["r"] * GiB) + v.d     # 留在慢層 vs 丟掉重算：每塊省下的時間
            self.g_c = [max(fi - ls, 0.0) for fi in v.f]
            self.g_cc = [0.0]
            for x in self.g_c:
                self.g_cc.append(self.g_cc[-1] + x)
        if rule == "PENS":
            fi = f_array(eng.K, eng.P["nmax"], "idle")
            self.pens_fc = [0.0]
            for x in fi:
                self.pens_fc.append(self.pens_fc[-1] + x)

    def size(self, s):
        return len(self.e.sb[s]) * self.e.i4

    def add(self, s):
        e, r = self.e, self.rule
        self.mem[s] = None
        self.mem.move_to_end(s)
        if r == "ARC":
            size = self.size(s)
            if s in self.B1:
                self.b1b -= self.B1.pop(s)
                self.p = min(self.C, self.p + max(1.0, self.b2b / max(self.b1b, 1)) * size)
                self.T2[s] = size
                self.t2b += size
            elif s in self.B2:
                self.b2b -= self.B2.pop(s)
                self.p = max(0.0, self.p - max(1.0, self.b1b / max(self.b2b, 1)) * size)
                self.T2[s] = size
                self.t2b += size
            elif self.hits.get(s, 0) > 0:
                self.T2[s] = size
                self.t2b += size
            else:
                self.T1[s] = size
                self.t1b += size
            while self.t1b + self.b1b > self.C and self.B1:
                self.b1b -= self.B1.popitem(last=False)[1]
            while self.t1b + self.t2b + self.b1b + self.b2b > 2 * self.C and self.B2:
                self.b2b -= self.B2.popitem(last=False)[1]
        elif r == "S3FIFO":
            size = self.size(s)
            self.sz[s] = size
            self.f[s] = min(self.hits.get(s, 0), 3)
            if s in self.Gh:
                del self.Gh[s]
                self.M[s] = None
                self.mb_ += size
            else:
                self.S[s] = None
                self.sb_ += size
        elif r == "GDSF":
            bl = e.sb[s]
            n = len(bl)
            ex = e.excl_start(s, e.ref_live)
            cps = (self.g_cc[n] - self.g_cc[ex]) / ((n - ex) * e.i4) if ex < n else self.g_c[n - 1] / e.i4
            H = self.gL + e.nreq[s] * cps * e.i4
            v = self.gver.get(s, 0) + 1
            self.gver[s] = v
            heapq.heappush(self.gheap, (H, e.last[s], v, s))

    def remove(self, s, hit):
        """s 離開慢層（hit＝回來了；False＝被丟掉）。不在裡面就什麼都不做。"""
        if s not in self.mem:
            return
        del self.mem[s]
        if hit:
            self.hits[s] = self.hits.get(s, 0) + 1
        r = self.rule
        if r == "ARC":
            if s in self.T1:
                self.t1b -= self.T1.pop(s)
            elif s in self.T2:
                self.t2b -= self.T2.pop(s)
        elif r == "S3FIFO":
            if s in self.S:
                del self.S[s]
                self.sb_ -= self.sz[s]
            elif s in self.M:
                del self.M[s]
                self.mb_ -= self.sz[s]
            self.f.pop(s, None)
        elif r == "GDSF":
            self.gver[s] = self.gver.get(s, 0) + 1

    def select(self):
        """選一個慢層狀態的 session 丟掉（呼叫端負責 drop）；沒有就回 None。選中的從結構裡拿掉（ARC／S3FIFO 進鬼表）。"""
        if not self.mem:
            return None
        e, r = self.e, self.rule
        if r == "LRU":
            s = next(iter(self.mem))
        elif r == "LRU2":
            s = min(self.mem, key=lambda x: (e.prev[x] if e.nreq[x] >= 2 else -1, e.last[x]))
        elif r == "PENS":
            cur, fc = e.cur, self.pens_fc

            def V(x):
                n = len(e.sb[x])
                ex = e.excl_start(x, e.ref_live)
                cost = fc[n] - fc[ex] if ex < n else fc[n] - fc[n - 1]
                return (cost / max(1, cur - e.last[x] + 1), e.last[x])
            s = min(self.mem, key=V)
        elif r == "BEL":
            s = max(self.mem, key=lambda x: (e.next_req(x, e.ref_live)[0], -e.last[x]))
        elif r == "GDSF":
            s = None
            while self.gheap:
                H, _, v, x = heapq.heappop(self.gheap)
                if self.gver.get(x) == v and x in self.mem:
                    self.gL = H
                    self.gver[x] = v + 1
                    s = x
                    break
            if s is None:
                s = next(iter(self.mem))
        elif r == "ARC":
            if self.T1 and (self.t1b > self.p or not self.T2):
                s, sz = self.T1.popitem(last=False)
                self.t1b -= sz
                self.B1[s] = sz
                self.b1b += sz
            else:
                s, sz = self.T2.popitem(last=False)
                self.t2b -= sz
                self.B2[s] = sz
                self.b2b += sz
        else:   # S3FIFO
            Sc = 0.1 * self.C
            while True:
                if self.S and (self.sb_ > Sc or not self.M):
                    x, _ = self.S.popitem(last=False)
                    self.sb_ -= self.sz[x]
                    if self.f[x] >= 1:
                        self.f[x] = 0
                        self.M[x] = None
                        self.mb_ += self.sz[x]
                        continue
                    self.Gh[x] = None
                    while len(self.Gh) > max(len(self.M), 1):
                        self.Gh.popitem(last=False)
                    del self.f[x]
                    s = x
                    break
                x, _ = self.M.popitem(last=False)
                if self.f[x] > 0:
                    self.f[x] -= 1
                    self.M[x] = None
                    continue
                self.mb_ -= self.sz[x]
                del self.f[x]
                s = x
                break
        self.mem.pop(s, None)
        return s


# ================================================================== 引擎
class Engine:
    def __init__(self, P, policy, cap_bytes, views, K, gdsf_view=None, check=False, gpu_cap=0, gpu_policy="LRU", gpu_write="wb",
                 pf_target="cpu", noise=None, slow_cap=None, slow_policy="LRU", pf_expire=False, noise_overdue="clamp"):
        self.P = P
        self.full, self.nall, self.acc, self.t = P["full"], P["nall"], P["acc"], P["tl"]
        self.kind, self.sess, self.parent = P["kindc"], P["sessl"], P["parentl"]
        self.order, self.act, self.hook, self.pf = POLICIES[policy]
        self.policy = policy
        self.dead_cut = float("inf")
        if "-H" in policy:
            self.dead_cut = P["tl"][-1] - float(policy.split("-H")[1])
        self.C = cap_bytes
        self.views = views
        self.K = K
        self.check = check
        self.i4 = K["int4_bytes"]
        B, S = P["B"], P["R"]["n_sess"]
        self.ref_cpu = [0] * B
        self.ref_live = [0] * B
        self.on_slow = [False] * B
        self.fmt = [0] * B            # 0 不在 CPU、1 BF16、2 INT4（預取）
        self.pf_done = [0.0] * B
        self.pf_owner = [-1] * B
        self.sb = [None] * S          # session 的 block（最新一輪的完整 block）
        self.st = [S_NONE] * S
        self.last = [-1] * S
        self.prev = [-1] * S
        self.nreq = [0] * S
        self.spf = [0] * S            # 預取進 CPU 的 block 數
        self.sT = [INF] * S           # 降級當下算的「最後一個 block 下次被讀」（預取候選用）
        self.pfdone = [False] * S
        self.cpu_used = 0
        self.cur = -1
        self.cpu_set = set()
        self.lru = OrderedDict()
        self.n_demote = self.n_drop = self.n_unpf = 0
        self.n_pfexp = 0              # 偏離 14 (i) 丟掉的次數（只在診斷時看；不進 CSV）
        self.res_sum = 0.0
        self.w_slow_blocks = 0
        # ARC
        self.T1, self.T2, self.B1, self.B2 = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
        self.t1b = self.t2b = self.b1b = self.b2b = 0
        self.arc_p = 0.0
        # S3-FIFO
        self.S3S, self.S3M, self.S3G = OrderedDict(), OrderedDict(), OrderedDict()
        self.s3f = {}
        self.s3sb = self.s3mb = 0
        self.s3size = {}
        # GDSF
        self.gH, self.gver, self.gheap, self.gL = {}, {}, [], 0.0
        if self.order == "gdsf":
            v = gdsf_view
            l_cpu = BF / (K["cpu"]["r"] * GiB)
            l_sl = self.i4 / (K["slow"][v.slow_name]["r"] * GiB) + v.d
            self.g_c = [max(min(fi, l_sl) - l_cpu, 0.0) for fi in v.f]
            self.g_cc = [0.0]
            for x in self.g_c:
                self.g_cc.append(self.g_cc[-1] + x)
        # Pensieve：f(i) 的前綴和（GPU 閒；GPU 忙是同一個倍數，不改順序）
        if self.order == "pens":
            fi = f_array(K, P["nmax"], "idle")
            self.pens_fc = [0.0]
            for x in fi:
                self.pens_fc.append(self.pens_fc[-1] + x)
        # 預取（單一時間設定）
        if self.pf:
            assert len(views) == 1, "預取策略的狀態和時間有關，一次只能跑一種時間設定"
        self.pfheap = []
        self.pf_sessions = set()
        # ICPE next：GPU 快取層（gpu_cap＝0＝關閉，下面都不建、step 走原本的路徑）
        self._fix_fast = True         # 加速：_fix 沒有副作用時可以跳過（慢層容量上限會包一層有副作用的 _fix → False）
        self.noise = None             # W4 雜訊（PREREG §2 第 6 項、§3 B）；None＝完美預知（原本的路徑）
        # PREREG 偏離 14（SIM_CHANGES 22）：(i) O-PF 預取的副本過期就丟；(a) 模型 P 抽到已經過去的時間就重抽。預設關＝原本的路徑
        assert noise_overdue in ("clamp", "resample"), noise_overdue
        self.pfx = bool(pf_expire) and self.pf == "opf"
        self.nzo_rs = noise_overdue == "resample"
        self.G = gpu_cap
        self.pf_target = pf_target
        self.gpf = None               # 預取進 GPU 的塊的主人（只有 pf_target＝gpu 時才建）
        if self.G > 0:
            self._gpu_init(gpu_policy, gpu_write, B, S, len(P["full"]))
        else:
            assert pf_target == "cpu", "預取搬回 GPU 要開 GPU 層"
        if noise is not None:
            self._noise_init(noise)
        self.sc = None                # 慢層容量上限（第二批；PREREG §2 第 5 項）；None＝沒有上限（原本的路徑）
        if slow_cap is not None:
            self._sc_init(slow_cap, slow_policy, gdsf_view)

    # ---------------------------------------------------------- GPU 快取層（ICPE next）
    def _gpu_init(self, gpu_policy, gpu_write, B, S, nreq):
        assert gpu_write in GPU_WRITES, gpu_write
        self.gorder, self.gttl = parse_gpol(gpu_policy)
        self.gpol, self.gwrite = gpu_policy, gpu_write
        if self.gttl is not None:
            assert len(self.views) == 1, "TTL 從回合結束（第一個 token）起算，和時間設定有關：一次只能跑一種時間設定"
        self.ref_gpu = [0] * B
        self.gpu_used = 0
        self.gin = [False] * S        # session 在 GPU 上
        self.gsb = [None] * S         # wt：session 在 GPU 上的 block（wb 用 sb）
        self.glru = OrderedDict()     # GPU 上的 session，最近回合結束的在最後
        self.gexp = [0.0] * S         # TTL：釘到這個時間
        self.gpver = [0] * S          # （GDSF 已經用了 gver）
        self.gttlheap = []            # TTLA：(到期, 版本, session)
        self.n_gevict = self.n_gexpire = 0
        self.g_wb_blocks = 0          # wb：GPU 逐出時寫進 CPU 的 block 數
        self.sstart = [float("inf")] * S   # 對話開始時間＝這個 session 第一個請求的到達時間（TTL 的防死結規則用）
        for r, s in enumerate(self.sess):
            if self.sstart[s] == float("inf"):
                self.sstart[s] = self.t[r]
        if gpu_write == "wb":
            self._fix = self._fix_g   # GPU 上的持有者也算活著（見 _fix_g）
        for v in self.views:
            v.nG = np.zeros(nreq, np.int32)
        assert self.pf_target in ("cpu", "gpu"), self.pf_target
        if self.pf_target == "gpu" and self.pf:     # PREREG §2 第 2 項的敏感度：預取直接搬回 GPU
            self.gpf = [-1] * B
            self.gpf_done = [0.0] * B
            self.gspf = [0] * S
            self.gpf_sessions = set()

    def _gbl(self, s):
        return self.sb[s] if self.gwrite == "wb" else self.gsb[s]

    def _next_req_bl(self, bl, ref):
        """同 next_req，但給定 block 清單與 refcount（GPU 層用 ref_gpu）。"""
        lo, hi = 0, len(bl)
        while lo < hi:
            mid = (lo + hi) >> 1
            if ref[bl[mid]] <= 1:
                hi = mid
            else:
                lo = mid + 1
        b = bl[lo] if lo < len(bl) else bl[-1]
        return XT.next_access(self.acc, b, self.cur)

    def _g_next(self, x):
        """GPU 層的完美預知：x 獨有的第一塊（照 ref_gpu）下次被讀的請求 index（有雜訊時加雜訊）。"""
        nx = self._next_req_bl(self._gbl(x), self.ref_gpu)
        return nx if self.noise is None else self._nz(x, nx)

    def _g_add(self, s, te):
        """回合結束：s 進 GPU（或更新順序）；TTL 從 te 起算。"""
        self.gin[s] = True
        self.glru[s] = None
        self.glru.move_to_end(s)
        if self.gttl is not None:
            self.gexp[s] = te + self.gttl
            if self.gorder == "ttla":
                self.gpver[s] += 1
                heapq.heappush(self.gttlheap, (self.gexp[s], self.gpver[s], s))

    def _g_remove(self, s):
        self.gin[s] = False
        self.glru.pop(s, None)
        self.gpver[s] += 1

    def _g_select(self, s_new, now):
        """GPU 超量時選一個 session 逐出。s_new＝剛回合結束的 session。"""
        o = self.gorder
        if o == "lru":
            return next(iter(self.glru))
        if o == "bel":
            rg = self.ref_gpu
            return max(self.glru, key=lambda x: (self._g_next(x), -self.last[x]))
        for x in self.glru:            # TTL：沒在釘住期的，照 LRU
            if self.gexp[x] <= now:
                return x
        # 全部都在釘住期 → 先放掉「對話開始時間最晚」的那個（PREREG §2 第 3 項；Continuum §5.2 的防死結規則）
        st0 = self.sstart
        return max(self.glru, key=lambda x: (st0[x], x))

    def _g_release(self, bl):
        rg = self.ref_gpu
        for b in bl:
            rg[b] -= 1
            if rg[b] == 0:
                self.gpu_used -= BF

    def _g_evict_wb(self, x):
        """wb：x 從 GPU 寫進 CPU（BF16；CPU 已有 BF16 的 block 不用寫）。回傳寫進 CPU 的 block 數。CPU 層把這次進入當成一次存取。"""
        self._g_remove(x)
        self.st[x] = S_CPU
        rc, fmt = self.ref_cpu, self.fmt
        cw = 0
        for b in self.sb[x]:
            rc[b] += 1
            fm = fmt[b]
            if fm == 0:
                fmt[b] = 1
                self.cpu_used += BF
                cw += 1
            elif fm == 2:
                fmt[b] = 1
                self.cpu_used += BF - self.i4
                cw += 1
                o = self.pf_owner[b]
                if o >= 0:
                    self.spf[o] -= 1
                    self.pf_owner[b] = -1
        self._g_release(self.sb[x])
        self._order_add(x)
        self.n_gevict += 1
        self.g_wb_blocks += cw
        if self.gpf is not None:
            self._pf_push(x)
        return cw

    def _g_free_wt(self, x):
        """wt：丟掉 GPU 上那份（CPU 層本來就有自己的狀態）。"""
        self._g_remove(x)
        self._g_release(self.gsb[x])
        self.gsb[x] = None
        self.n_gevict += 1
        if self.gpf is not None and self.st[x] in (S_CPU, S_SLOW):
            self._pf_push(x)

    def _g_enforce(self, s_new, now):
        """GPU 已用 > 預算時逐出。回傳寫進 CPU 的 block 數（wt 永遠 0）。"""
        cw = 0
        while self.gpu_used > self.G:
            if self.gpf is not None and self.gpf_sessions and (self.pf == "rpf" or not self.glru):
                if self.pf == "rpf":      # 同 CPU 預取：近期版預取進來的比 GPU 上的都舊，先清
                    u = min(self.gpf_sessions, key=lambda x: self.last[x])
                else:
                    u = max(self.gpf_sessions, key=lambda x: (self.sT[x], -self.last[x]))
                self._g_unprefetch(u, self.sb[u])
                continue
            x = self._g_select(s_new, now)
            if self.gwrite == "wb":
                cw += self._g_evict_wb(x)
            else:
                self._g_free_wt(x)
        return cw

    def _g_expire(self, t_now):
        """TTLA：到期的 session 在到期時刻逐出（wb：寫 CPU、CPU 超量就照 CPU 規則降級；寫入排在到期時刻）。"""
        h = self.gttlheap
        v = self.views[0]
        while h and h[0][0] <= t_now:
            exp, ver, x = heapq.heappop(h)
            if not self.gin[x] or self.gpver[x] != ver:
                continue
            self.n_gexpire += 1
            if self.gwrite == "wb":
                cw = self._g_evict_wb(x)
                w = self.enforce()
                self.w_slow_blocks += w
                v.cpu.write(cw * BF, exp)
                v.slow.write(w * self.i4, exp)
            else:
                self._g_free_wt(x)

    # ---------------------------------------------------------- 慢層容量上限（第二批；PREREG §2 第 5 項；使用者 11/4 的兩個決定）
    def _sc_init(self, frac, rule, gdsf_view):
        """上限＝⌊frac × n_blocks_full⌋ 個 INT4 塊；慢層上所有 INT4 塊都算。超過 → 先丟乾淨的副本（沒有慢層狀態的持有者；
        照變乾淨的先後），不夠再照 rule 丟整個慢層狀態的 session（回來全部重算）。"""
        assert 0 < frac, frac
        if rule == "GDSF":
            assert gdsf_view is not None, "慢層 GDSF 要指定時間設定（同 CPU 的 GDSF：一個 job 一組 (慢層, GPU 狀態)）"
        self.sc = dict(frac=float(frac), rule=rule)
        self.sc_cap = int(math.floor(frac * self.P["meta"]["n_blocks_full"]))
        self.n_slow = sum(self.on_slow)
        self.clean_q = deque()
        self.n_sc_clean = self.n_sc_drop = 0
        self.sorder = SlowOrder(self, rule, self.sc_cap * self.i4, gdsf_view)
        self._fix_inner = self._fix
        self._fix = self._fix_sc
        self._fix_fast = False
        self._demote_inner = self.demote
        self.demote = self._demote_sc
        self._enforce_inner = self.enforce
        self.enforce = self._enforce_sc

    def _slow_holders(self, b):
        n = self.ref_live[b] - self.ref_cpu[b]
        if self.G > 0 and self.gwrite == "wb":
            n -= self.ref_gpu[b]
        return n

    def _fix_sc(self, b):
        was = self.on_slow[b]
        w = self._fix_inner(b)
        now = self.on_slow[b]
        if now != was:
            self.n_slow += 1 if now else -1
        if now and self._slow_holders(b) == 0:
            self.clean_q.append(b)            # 可能變乾淨了（丟的時候再驗一次）
        return w

    def _demote_sc(self, s, now_by_view=None):
        w = self._demote_inner(s, now_by_view)
        self.sorder.add(s)
        return w

    def _enforce_sc(self):
        w = self._enforce_inner()
        self._slow_enforce()
        return w

    def _slow_enforce(self):
        while self.n_slow > self.sc_cap:
            if self.clean_q:
                b = self.clean_q.popleft()
                if self.on_slow[b] and self._slow_holders(b) == 0 and self.fmt[b] != 2:
                    self.on_slow[b] = False
                    self.n_slow -= 1
                    self.n_sc_clean += 1
                continue
            s = self.sorder.select()
            if s is None:
                break
            assert self.st[s] == S_SLOW, (s, self.st[s])
            self.drop(s)
            self.n_sc_drop += 1

    # ---------------------------------------------------------- W4 雜訊（PREREG §2 第 6 項、§3 B）
    def _noise_init(self, nz):
        """nz＝dict(model＝P|M, lam, draw＝B1|B2, seed)。完美預知的每一種用途（O-BEL／O-BELS 的順序、O-DROP／O-NOW 的生死、
        O-PF 的預取順序與騰空間、GPU 層的 BEL、GPU 預取）都經過 _nz。λ＝0 時 _nz 直接回傳真值（自檢：和沒有雜訊逐值相同）。"""
        m, lam, dr, seed = nz["model"], float(nz["lam"]), nz["draw"], int(nz["seed"])
        assert m in ("P", "M") and dr in ("B1", "B2") and 0.0 <= lam < 1.0, nz
        self.noise = dict(model=m, lam=lam, draw=dr, seed=seed)
        self.nzm, self.nzl, self.nz_b1 = m, lam, dr == "B1"
        self.nzrng = random.Random(seed)
        # 模型 P 的經驗分佈：每個請求（回合結束）到同一個 session 下一次請求的時間；之後不再回來＝INF（照它的經驗比例）
        nxt = {}
        gaps = []
        tl, sl = self.t, self.sess
        for r in range(len(tl) - 1, -1, -1):
            s = sl[r]
            gaps.append(tl[nxt[s]] - tl[r] if s in nxt else INF)
            nxt[s] = r
        self.nz_gaps = gaps[::-1]
        if self.nzo_rs:                        # 偏離 14 (a)：重抽用（照值排序；INF 排在最後）
            self.nz_sorted = sorted(self.nz_gaps)
        self.nzd = [None] * self.P["R"]["n_sess"]
        self.next_req = self._next_req_nz      # select（O-BEL／O-BELS）、dead（O-DROP／O-NOW）、make_room（O-PF）都走這個

    def _nz_sample(self):
        rng = self.nzrng
        return rng.random(), rng.choice(self.nz_gaps), rng.uniform(-self.nzl, self.nzl)

    def _next_req_nz(self, s, ref):
        nx, e = Engine.next_req(self, s, ref)
        return self._nz(s, nx), e

    def _nz(self, s, idx):
        """真的「下次回來」（請求 index）→ 有雜訊的 index。B1：用這個 session 上次回合結束時抽的；B2：每次問都重抽。
        P：以機率 λ 換成「上次回合結束＋經驗分佈裡抽的間隔」（可能變成不再回來、或不再回來變成會回來）；
        M：剩餘時間（從現在算）×(1＋ε)、ε～U[−λ, λ]；不再回來維持不再回來。時間換回 index：第一個到達時間 ≥ 它的請求，至少是下一個請求。"""
        if self.nzl == 0.0:
            return idx
        if self.nz_b1:
            d = self.nzd[s]
            if d is None:
                d = self.nzd[s] = self._nz_sample()
        else:
            d = self._nz_sample()
        u, g, eps = d
        tl = self.t
        if self.nzm == "P":
            if u >= self.nzl:
                return idx
            if g >= INF:
                return INF
            Tq = tl[self.last[s]] + g
            if self.nzo_rs:
                # 偏離 14 (a)：抽到的時間已經過去（換算後會落在已處理的請求）→ 從經驗分佈裡「間隔 > 已經過的時間」那部分均勻重抽
                # （含 INF＝之後不再回來），同一個亂數串流多抽一次；B1 取代存著的那組。
                tq = tl[self.cur] if self.cur >= 0 else tl[0]
                if Tq <= tq:
                    sg = self.nz_sorted
                    k = bisect.bisect_right(sg, tq - tl[self.last[s]])
                    g = sg[k + self.nzrng.randrange(len(sg) - k)]
                    if self.nz_b1:
                        self.nzd[s] = (u, g, eps)
                    if g >= INF:
                        return INF
                    Tq = tl[self.last[s]] + g
        else:
            if idx >= len(tl) or eps == 0.0:      # 含 INF（不再回來）；超過紀錄長度（只在截短的自檢裡會發生）照原值
                return idx
            tq = tl[self.cur] if self.cur >= 0 else tl[0]
            Tq = tq + (tl[idx] - tq) * (1.0 + eps)
        j = bisect.bisect_left(tl, Tq)
        c = self.cur + 1
        return j if j > c else c

    # ---------------------------------------------------------- PREREG 偏離 14 (i)：O-PF 預取的副本過期就丟
    def _pf_expire(self):
        """請求 r 開始時（cur＝r−1；GPU 層 TTLA 到期、推進預取之前）：還留著預取副本、預測被讀的請求 sT ≤ cur 的 session
        → 丟掉它全部的預取副本（CPU：INT4；GPU：預取進 GPU 的那份；資料還在慢層），不再放回預取佇列（pfdone）。
        每個 session 的處理互不影響（只減 cpu_used／gpu_used、改它自己的塊），所以和走訪順序無關。"""
        cur = self.cur
        if self.gpf is not None:
            for s in [x for x in self.gpf_sessions if self.gspf[x] > 0 and self.sT[x] <= cur]:
                self._g_unprefetch(s, self.sb[s], push=False)
                self.pfdone[s] = True
                self.n_pfexp += 1
        else:
            for s in [x for x in self.pf_sessions if self.spf[x] > 0 and self.sT[x] <= cur]:
                for b in self.sb[s]:
                    if self.pf_owner[b] == s:
                        self.cpu_used -= self.i4
                        self.fmt[b] = 0
                        self.pf_owner[b] = -1
                self.spf[s] = 0
                self.pf_sessions.discard(s)
                self.pfdone[s] = True
                self.n_pfexp += 1

    # ---------------------------------------------------------- 預取直接搬回 GPU（PREREG §2 第 2 項的敏感度）
    def _pf_push(self, s):
        """s 離開 GPU、資料在 CPU 或慢層：變成預取候選（同 demote 裡的做法）。"""
        if not self.sb[s]:
            return
        T = XT.next_access(self.acc, self.sb[s][-1], self.cur)
        if self.noise is not None:
            T = self._nz(s, T)
        self.sT[s] = T
        self.pfdone[s] = False
        if self.pf == "opf" and T < INF:
            heapq.heappush(self.pfheap, (T, s))
        elif self.pf == "rpf":
            heapq.heappush(self.pfheap, (-self.last[s], s))

    def _gpf_claim(self, b):
        """預取進 GPU 的塊被某個 session 的回合結束拿去當常駐（同一份資料，GPU 位元組不變）。"""
        o = self.gpf[b]
        self.gpf[b] = -1
        self.gspf[o] -= 1
        if self.gspf[o] <= 0:
            self.gpf_sessions.discard(o)

    def _g_unprefetch(self, s, bl, push=True):
        """取消 s 預取進 GPU 的塊（bl＝要找的清單）。"""
        gpf, rg = self.gpf, self.ref_gpu
        for b in bl:
            if gpf[b] == s:
                gpf[b] = -1
                if rg[b] == 0:
                    self.gpu_used -= BF
                self.gspf[s] -= 1
        assert self.gspf[s] == 0, (s, self.gspf[s])
        self.gpf_sessions.discard(s)
        self.pfdone[s] = False
        if push:
            self.n_unpf += 1
            if self.pf == "opf" and self.st[s] in (S_CPU, S_SLOW) and self.sT[s] < INF:
                heapq.heappush(self.pfheap, (self.sT[s], s))

    def _g_make_room(self, cand, need, now, v):
        """預取 cand 進 GPU 需要 need 位元組：只准逐出預取規則認為「比 cand 冷」的 GPU session（同 make_room）。"""
        if self.G - self.gpu_used >= need:
            return True
        rg = self.ref_gpu
        if self.pf == "opf":
            Tc = self.sT[cand]
            el = []
            for x in self.glru:
                nx = self._g_next(x)
                if nx > Tc:
                    el.append((nx, -self.last[x], x))
            el.sort(reverse=True)
            order = [x for _, _, x in el]
        else:
            lc = self.last[cand]
            order = sorted((x for x in self.glru if self.last[x] < lc), key=lambda x: self.last[x])
        cw = 0
        for x in order:
            if self.G - self.gpu_used >= need:
                break
            if self.gwrite == "wb":
                cw += self._g_evict_wb(x)
            else:
                self._g_free_wt(x)
        if cw:
            v.cpu.write(cw * BF, now)
            w = self.enforce()
            self.w_slow_blocks += w
            v.slow.write(w * self.i4, now)
        return self.G - self.gpu_used >= BF

    def _advance_prefetch_gpu(self, t_from, t_to):
        """同 advance_prefetch，但目的地是 GPU：候選＝不在 GPU 上、資料在 CPU 或慢層的 session；它存著、不在 GPU 上的塊
        全部從最後一塊往前搬（GPU 上的塊載入成本＝0，Cake 的會合點退到最前面）。CPU 上的 BF16 走 CPU 線；慢層的 INT4 走慢層線
        ＋反量化 d；放進 GPU 是 BF16（64 MiB／塊）。"""
        v = self.views[0]
        cpu, slow = v.cpu, v.slow
        rg, gpf, fmt = self.ref_gpu, self.gpf, self.fmt
        # 先 pop 再處理（提早結束就推回去）：wb 騰空間時 CPU 層可能降級別的 session、把新的候選推進 heap，heap 頂端會變
        # （原本的 CPU 預取不會：它推進去的鍵一定比頂端大；這裡的 CPU 降級不經過「比候選者冷」的篩選）。
        while self.pfheap:
            key, s = heapq.heappop(self.pfheap)
            if self.st[s] not in (S_CPU, S_SLOW) or self.gin[s] or self.pfdone[s] or \
                    (self.pf == "opf" and key != self.sT[s]) or (self.pf == "rpf" and key != -self.last[s]):
                continue
            todo = [b for b in reversed(self.sb[s]) if rg[b] == 0 and gpf[b] < 0]
            if not todo:
                self.pfdone[s] = True
                continue
            sc = cpu.t_free if cpu.t_free > t_from else t_from
            ss = slow.t_free if slow.t_free > t_from else t_from
            st0 = sc if sc < ss else ss
            if st0 >= t_to or not self._g_make_room(s, len(todo) * BF, st0, v):
                heapq.heappush(self.pfheap, (key, s))
                return
            for b in todo:
                fm = fmt[b]
                ch = cpu if fm else slow
                st = ch.t_free if ch.t_free > t_from else t_from
                if st >= t_to or self.G - self.gpu_used < BF:
                    heapq.heappush(self.pfheap, (key, s))
                    return
                if fm == 1:
                    done = cpu.read(BF, st)
                    v.pf_bytes += BF
                elif fm == 2:
                    pd = self.pf_done[b]
                    done = cpu.read(self.i4, st if st > pd else pd) + v.d
                    v.pf_bytes += self.i4
                else:
                    done = slow.read(self.i4, st) + v.d
                    v.pf_bytes += self.i4
                gpf[b] = s
                self.gpf_done[b] = done
                self.gpu_used += BF
                self.gspf[s] += 1
                self.gpf_sessions.add(s)
            self.pfdone[s] = True

    def _fix_g(self, b):
        """wb 版的 _fix：ref_live 也算 GPU 上的持有者。CPU 那份沒有 CPU 狀態的持有者時釋放；只有還有慢層狀態的持有者
        （ref_live − ref_cpu − ref_gpu > 0）而且慢層還沒有時才寫慢層（session 搬上 GPU 不寫慢層）。"""
        if self.ref_live[b] == 0:
            fm = self.fmt[b]
            if fm:
                self.cpu_used -= BF if fm == 1 else self.i4
                self.fmt[b] = 0
                o = self.pf_owner[b]
                if o >= 0:
                    self.spf[o] -= 1
                    self.pf_owner[b] = -1
            self.on_slow[b] = False
            return 0
        if self.ref_cpu[b] == 0 and self.fmt[b] == 1:
            self.cpu_used -= BF
            self.fmt[b] = 0
            if not self.on_slow[b] and self.ref_live[b] - self.ref_gpu[b] > 0:
                self.on_slow[b] = True
                return 1
        return 0

    # ---------------------------------------------------------- 小工具
    def excl_start(self, s, ref):
        bl = self.sb[s]
        lo, hi = 0, len(bl)
        while lo < hi:
            mid = (lo + hi) >> 1
            if ref[bl[mid]] <= 1:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def next_req(self, s, ref):
        """session 只有它持有的第一個 block（沒有就用最後一個 block）在目前請求之後第一次被讀的請求 index。"""
        bl = self.sb[s]
        e = self.excl_start(s, ref)
        b = bl[e] if e < len(bl) else bl[-1]
        return XT.next_access(self.acc, b, self.cur), e

    def dead(self, s):
        if self.t[self.cur if self.cur >= 0 else 0] > self.dead_cut:
            return False
        if not self.sb[s]:
            return True
        return self.next_req(s, self.ref_live)[0] >= INF

    def _fix(self, b):
        if self.ref_live[b] == 0:
            fm = self.fmt[b]
            if fm:
                self.cpu_used -= BF if fm == 1 else self.i4
                self.fmt[b] = 0
                o = self.pf_owner[b]
                if o >= 0:
                    self.spf[o] -= 1
                    self.pf_owner[b] = -1
            self.on_slow[b] = False
            return 0
        if self.ref_cpu[b] == 0 and self.fmt[b] == 1:
            self.cpu_used -= BF
            self.fmt[b] = 0
            if not self.on_slow[b]:
                self.on_slow[b] = True
                return 1
        return 0

    # ---------------------------------------------------------- 順序結構
    def _order_add(self, s):
        """回合結束（session 被存取）時更新順序結構。"""
        o = self.order
        self.cpu_set.add(s)
        if o == "lru":
            self.lru[s] = None
            self.lru.move_to_end(s)
        elif o == "arc":
            size = len(self.sb[s]) * BF
            if s in self.T1:
                self.t1b -= self.T1.pop(s)
                self.T2[s] = size
                self.t2b += size
            elif s in self.T2:
                self.t2b -= self.T2.pop(s)
                self.T2[s] = size
                self.t2b += size
            elif s in self.B1:
                sz = self.B1.pop(s)
                self.b1b -= sz
                delta = max(1.0, self.b2b / max(self.b1b, 1)) * size
                self.arc_p = min(float(self.C), self.arc_p + delta)
                self.T2[s] = size
                self.t2b += size
            elif s in self.B2:
                sz = self.B2.pop(s)
                self.b2b -= sz
                delta = max(1.0, self.b1b / max(self.b2b, 1)) * size
                self.arc_p = max(0.0, self.arc_p - delta)
                self.T2[s] = size
                self.t2b += size
            else:
                self.T1[s] = size
                self.t1b += size
            while self.t1b + self.b1b > self.C and self.B1:
                _, sz = self.B1.popitem(last=False)
                self.b1b -= sz
            while self.t1b + self.t2b + self.b1b + self.b2b > 2 * self.C and self.B2:
                _, sz = self.B2.popitem(last=False)
                self.b2b -= sz
        elif o == "s3":
            size = len(self.sb[s]) * BF
            if s in self.S3S or s in self.S3M:
                self.s3f[s] = min(self.s3f[s] + 1, 3)
                old = self.s3size[s]
                if s in self.S3S:
                    self.s3sb += size - old
                else:
                    self.s3mb += size - old
                self.s3size[s] = size
            elif s in self.S3G:
                del self.S3G[s]
                self.S3M[s] = None
                self.s3mb += size
                self.s3f[s] = 0
                self.s3size[s] = size
            else:
                self.S3S[s] = None
                self.s3sb += size
                self.s3f[s] = 0
                self.s3size[s] = size
        elif o == "gdsf":
            bl = self.sb[s]
            n = len(bl)
            e = self.excl_start(s, self.ref_cpu)
            if e < n:
                cps = (self.g_cc[n] - self.g_cc[e]) / ((n - e) * BF)
            else:
                cps = self.g_c[n - 1] / BF
            H = self.gL + self.nreq[s] * cps * BF   # 每 block 的 cost×freq（size 的單位取 block，不影響排序）
            v = self.gver.get(s, 0) + 1
            self.gver[s] = v
            self.gH[s] = H
            heapq.heappush(self.gheap, (H, self.last[s], v, s))

    def _order_remove(self, s):
        """s 離開 CPU（不是由 select 選中時，例如回合結束的 hook、預取騰空間）。"""
        self.cpu_set.discard(s)
        o = self.order
        if o == "lru":
            self.lru.pop(s, None)
        elif o == "arc":
            if s in self.T1:
                self.t1b -= self.T1.pop(s)
            elif s in self.T2:
                self.t2b -= self.T2.pop(s)
        elif o == "s3":
            if s in self.S3S:
                del self.S3S[s]
                self.s3sb -= self.s3size[s]
            elif s in self.S3M:
                del self.S3M[s]
                self.s3mb -= self.s3size[s]
            self.s3f.pop(s, None)
        elif o == "gdsf":
            self.gver[s] = self.gver.get(s, 0) + 1   # 讓 heap 裡的舊項失效

    def select(self):
        """依順序策略選一個 CPU 狀態的 session（呼叫端負責降級／丟）。沒有就回 None。"""
        o = self.order
        if not self.cpu_set:
            return None
        if o == "lru":
            s = next(iter(self.lru))
            return s
        if o == "pred":
            # 通用預測器介面：預測下次到達＝−(上次存取的請求序號)；降級預測最晚的
            return max(self.cpu_set, key=lambda x: -self.last[x])
        if o == "lru2":
            return min(self.cpu_set, key=lambda x: (self.prev[x] if self.nreq[x] >= 2 else -1, self.last[x]))
        if o == "freq":
            return min(self.cpu_set, key=lambda x: (self.nreq[x], self.last[x]))
        if o == "pens":
            cur = self.cur
            fc = self.pens_fc

            def V(x):
                n = len(self.sb[x])
                e = self.excl_start(x, self.ref_cpu)
                cost = fc[n] - fc[e] if e < n else fc[n] - fc[n - 1]
                return (cost / max(1, cur - self.last[x] + 1), self.last[x])
            return min(self.cpu_set, key=V)
        if o == "bel":
            def key(x):
                nx, _ = self.next_req(x, self.ref_cpu)
                return (nx, -self.last[x])
            return max(self.cpu_set, key=key)
        if o == "bels":
            cur = self.cur

            def key(x):
                nx, e = self.next_req(x, self.ref_cpu)
                dist = 1e12 if nx >= INF else float(nx - cur)
                freed = (len(self.sb[x]) - e) * BF
                return (dist * max(freed, 1), -self.last[x])
            return max(self.cpu_set, key=key)
        if o == "arc":
            if self.T1 and (self.t1b > self.arc_p or not self.T2):
                s, sz = self.T1.popitem(last=False)
                self.t1b -= sz
                self.B1[s] = sz
                self.b1b += sz
            else:
                s, sz = self.T2.popitem(last=False)
                self.t2b -= sz
                self.B2[s] = sz
                self.b2b += sz
            self.cpu_set.discard(s)
            return s
        if o == "s3":
            Sc = 0.1 * self.C
            while True:
                if self.S3S and (self.s3sb > Sc or not self.S3M):
                    s, _ = self.S3S.popitem(last=False)
                    self.s3sb -= self.s3size[s]
                    if self.s3f[s] >= 1:
                        self.s3f[s] = 0
                        self.S3M[s] = None
                        self.s3mb += self.s3size[s]
                        continue
                    self.S3G[s] = None
                    while len(self.S3G) > max(len(self.S3M), 1):
                        self.S3G.popitem(last=False)
                    del self.s3f[s]
                    self.cpu_set.discard(s)
                    return s
                s, _ = self.S3M.popitem(last=False)
                if self.s3f[s] > 0:
                    self.s3f[s] -= 1
                    self.S3M[s] = None
                    continue
                self.s3mb -= self.s3size[s]
                del self.s3f[s]
                self.cpu_set.discard(s)
                return s
        if o == "gdsf":
            while self.gheap:
                H, _, v, s = heapq.heappop(self.gheap)
                if self.gver.get(s) == v and s in self.cpu_set:
                    self.gL = H
                    self.gver[s] = v + 1
                    self.cpu_set.discard(s)
                    return s
            return None
        raise ValueError(o)

    # ---------------------------------------------------------- 動作
    def demote(self, s, now_by_view=None):
        """CPU → 慢層。回傳要寫的 INT4 block 數。"""
        assert self.st[s] == S_CPU
        self.st[s] = S_SLOW
        self.cpu_set.discard(s)
        if self.order == "lru":
            self.lru.pop(s, None)
        w = 0
        for b in self.sb[s]:
            self.ref_cpu[b] -= 1
            w += self._fix(b)
        self.n_demote += 1
        self.res_sum += self.t[self.cur] - self.t[self.last[s]]   # 這次在 CPU 待了多久（上次回合到被降級）
        if self.pf:
            T = XT.next_access(self.acc, self.sb[s][-1], self.cur)
            if self.noise is not None:
                T = self._nz(s, T)
            self.sT[s] = T
            self.pfdone[s] = False
            if self.pf == "opf" and T < INF:
                heapq.heappush(self.pfheap, (T, s))
            elif self.pf == "rpf":
                heapq.heappush(self.pfheap, (-self.last[s], s))
        return w

    def drop(self, s):
        st = self.st[s]
        w = 0
        if self.sc is not None and st == S_SLOW:
            self.sorder.remove(s, hit=False)
        if self.gpf is not None and self.gspf[s] > 0:
            self._g_unprefetch(s, self.sb[s], push=False)
        if st in (S_CPU, S_SLOW):
            if st == S_CPU:
                self.cpu_set.discard(s)
                if self.order == "lru":
                    self.lru.pop(s, None)
            for b in self.sb[s]:
                if self.pf_owner[b] == s:
                    self.cpu_used -= self.i4
                    self.fmt[b] = 0
                    self.pf_owner[b] = -1
                    self.spf[s] -= 1
                self.ref_live[b] -= 1
                if st == S_CPU:
                    self.ref_cpu[b] -= 1
                w += self._fix(b)
            self.pf_sessions.discard(s)
        elif st == S_GPU:               # ICPE next（wb）：從 GPU 直接丟，不用寫
            self._g_remove(s)
            for b in self.sb[s]:
                self.ref_live[b] -= 1
            self._g_release(self.sb[s])
            for b in self.sb[s]:
                w += self._fix(b)
        self.st[s] = S_DROP
        self.sb[s] = []
        self.n_drop += 1
        return w

    def unprefetch(self, s):
        for b in self.sb[s]:
            if self.pf_owner[b] == s:
                self.cpu_used -= self.i4
                self.fmt[b] = 0
                self.pf_owner[b] = -1
        self.spf[s] = 0
        self.pf_sessions.discard(s)
        self.pfdone[s] = False
        self.n_unpf += 1
        if self.pf == "opf" and self.st[s] == S_SLOW and self.sT[s] < INF:
            heapq.heappush(self.pfheap, (self.sT[s], s))

    def victim_action(self, s):
        a = self.act
        if a == "demote":
            return self.demote(s)
        if a == "none":
            return self.drop(s)
        if a == "2hw":
            return self.drop(s) if self.nreq[s] < 2 else self.demote(s)
        if a == "onow":
            return self.drop(s) if self.dead(s) else self.demote(s)
        raise ValueError(a)

    def enforce(self):
        """CPU 已用 > 容量時依策略降級／丟。回傳要寫慢層的 block 數。"""
        w = 0
        while self.cpu_used > self.C:
            if self.pf == "rpf" and self.pf_sessions:
                # 近期版預取：預取進來的 session 比所有 CPU session 都舊（LRU 先把它們降下去的），照 LRU 先清
                s = min(self.pf_sessions, key=lambda x: self.last[x])
                self.unprefetch(s)
                continue
            s = self.select()
            if s is None:
                if self.pf_sessions:
                    if self.pf == "opf":
                        s = max(self.pf_sessions, key=lambda x: (self.sT[x], -self.last[x]))
                    else:
                        s = min(self.pf_sessions, key=lambda x: self.last[x])
                    self.unprefetch(s)
                    continue
                raise RuntimeError(f"CPU 超量 {self.cpu_used} > {self.C}，但沒有可以降級的")
            w += self.victim_action(s)
        return w

    # ---------------------------------------------------------- 預取（單一時間設定）
    def make_room(self, cand, need, now, v):
        """預取 cand 需要 need 位元組：只准降級「預取策略認為比 cand 冷」的 CPU session。回傳是否騰得出。"""
        if self.C - self.cpu_used >= need:
            return True
        if self.pf == "opf":
            Tc = self.sT[cand]
            el = []
            for x in self.cpu_set:
                nx, _ = self.next_req(x, self.ref_cpu)
                if nx > Tc:
                    el.append((nx, -self.last[x], x))
            el.sort(reverse=True)
            order = [x for _, _, x in el]
        else:
            lc = self.last[cand]
            order = sorted((x for x in self.cpu_set if self.last[x] < lc), key=lambda x: self.last[x])
        w = 0
        for x in order:
            if self.C - self.cpu_used >= need:
                break
            self._order_remove(x)
            self.cpu_set.add(x)   # demote 會再拿掉
            if self.act == "onow" and self.dead(x):
                w += self.drop(x)
            else:
                w += self.demote(x)
        if w:
            v.slow.write(w * self.i4, now)
            self.w_slow_blocks += w
        if self.sc is not None:
            self._slow_enforce()
        return self.C - self.cpu_used >= self.i4

    def advance_prefetch(self, t_from, t_to):
        v = self.views[0]
        slow = v.slow
        while self.pfheap:
            st = slow.t_free if slow.t_free > t_from else t_from
            if st >= t_to:
                return
            key, s = self.pfheap[0]
            if self.st[s] != S_SLOW or self.pfdone[s] or (self.pf == "opf" and key != self.sT[s]) or \
                    (self.pf == "rpf" and key != -self.last[s]):
                heapq.heappop(self.pfheap)
                continue
            bl = self.sb[s]
            m = v.m_cpu(len(bl))
            if self.G > 0:      # ICPE next：已在 GPU 上的不用預取
                todo = [b for b in reversed(bl[m:]) if self.fmt[b] == 0 and self.on_slow[b] and self.ref_gpu[b] == 0]
            else:
                todo = [b for b in reversed(bl[m:]) if self.fmt[b] == 0 and self.on_slow[b]]
            if not todo:
                self.pfdone[s] = True
                heapq.heappop(self.pfheap)
                continue
            if not self.make_room(s, len(todo) * self.i4, st, v):
                return     # Aggressive：最早回來的都騰不出空間，之後的更不可能；等下一個事件
            for b in todo:
                st = slow.t_free if slow.t_free > t_from else t_from
                if st >= t_to:
                    return
                if self.C - self.cpu_used < self.i4:
                    return
                done = slow.read(self.i4, st)
                self.fmt[b] = 2
                self.pf_done[b] = done
                self.pf_owner[b] = s
                self.cpu_used += self.i4
                self.spf[s] += 1
                v.pf_bytes += self.i4
                self.pf_sessions.add(s)
            self.pfdone[s] = True
            heapq.heappop(self.pfheap)

    # ---------------------------------------------------------- 一個請求
    def step(self, r):
        t = self.t[r]
        if self.pfx:              # 偏離 14 (i)；沒開＝原本的路徑
            self._pf_expire()
        if self.G > 0 and self.gorder == "ttla":
            self._g_expire(t)
        if self.pf:
            if self.gpf is not None:
                self._advance_prefetch_gpu(self.t[r - 1] if r > 0 else t, t)
            else:
                self.advance_prefetch(self.t[r - 1] if r > 0 else t, t)
        self.cur = r
        if self.noise is not None and self.nz_b1:
            self.nzd[self.sess[r]] = self._nz_sample()
        if self.sc is not None and self.st[self.sess[r]] == S_SLOW:
            self.sorder.remove(self.sess[r], hit=True)       # 回來了：離開慢層
        blocks = self.full[r]
        fmt, on_slow = self.fmt, self.on_slow
        if self.G > 0:      # ICPE next：GPU 上的 block 一定是前綴（session 粒度＋鏈式前綴）
            return self._step_gpu(r, t, blocks)
        lay = []
        for b in blocks:
            fm = fmt[b]
            if fm == 1:
                lay.append(1)
            elif fm == 2:
                lay.append(2)
            elif on_slow[b]:
                lay.append(3)
            else:
                break
        L = len(lay)
        if self.check:   # 每步：有存的 block 必須是前綴
            for b in blocks[L:]:
                assert fmt[b] == 0 and not on_slow[b], "有存的 block 不是前綴（refcount 錯）"
        kd = self.kind[r]
        old_n = len(self.full[self.parent[r]]) if kd == 1 else 0
        tend = []
        for v in self.views:
            tt, rs, nC, nS, nR = v.restore(lay, blocks, L, self.nall[r], t, old_n, self.pf_done)
            v.ttft[r], v.rest[r], v.nC[r], v.nS[r], v.nR[r] = tt, rs, nC, nS, nR
            tend.append(t + tt)
        # ---- 回合結束：全部 BF16 進 CPU
        s = self.sess[r]
        old = self.sb[s] if self.sb[s] else []
        old_state = self.st[s]
        cw = 0
        rc, rl = self.ref_cpu, self.ref_live
        # 加速（逐位元相同）：session 還在 CPU、舊清單是新清單的前綴 → 前綴的 +1／−1 互相抵銷（CPU 狀態的持有者的塊一定是
        # BF16，所以 fmt、cpu_used、cw 都不變；遞減後 ref_cpu ≥ 1，_fix 不會做事）→ 只處理新加的尾巴、不用釋放舊的
        k0 = 0
        if old_state == S_CPU and self._fix_fast and old and len(old) <= len(blocks) and blocks[len(old) - 1] == old[-1] \
                and blocks[:len(old)] == old:
            k0 = len(old)
        for b in (blocks[k0:] if k0 else blocks):
            rl[b] += 1
            rc[b] += 1
            fm = fmt[b]
            if fm == 0:
                fmt[b] = 1
                self.cpu_used += BF
                cw += 1
            elif fm == 2:
                fmt[b] = 1
                self.cpu_used += BF - self.i4
                cw += 1
                o = self.pf_owner[b]
                if o >= 0:
                    self.spf[o] -= 1
                    self.pf_owner[b] = -1
        w = 0
        if old_state in (S_CPU, S_SLOW) and not k0:
            if self._fix_fast:            # 加速（逐位元相同）：_fix 只有在「沒人持有」或「CPU 沒有持有者但有 BF16」時才做事
                fix = self._fix
                isc = old_state == S_CPU
                for b in old:
                    rl[b] -= 1
                    if isc:
                        rc[b] -= 1
                    if rl[b] == 0 or (rc[b] == 0 and fmt[b] == 1):
                        w += fix(b)
            else:
                for b in old:
                    rl[b] -= 1
                    if old_state == S_CPU:
                        rc[b] -= 1
                    w += self._fix(b)
        if self.spf[s] <= 0:
            self.pf_sessions.discard(s)
        self.sb[s] = list(blocks)
        self.st[s] = S_CPU
        self.prev[s] = self.last[s]
        self.last[s] = r
        self.nreq[s] += 1
        if blocks:
            self._order_add(s)
            hk = self.hook
            if hk == "2ha" and self.nreq[s] < 2:
                self._order_remove(s)
                self.cpu_set.add(s)
                w += self.drop(s)
            elif hk == "odrop" and self.dead(s):
                self._order_remove(s)
                self.cpu_set.add(s)
                w += self.drop(s)
        else:
            self._order_remove(s)          # 沒有完整 block：離開 CPU 的順序結構，不只 cpu_set（否則 LRU／ARC／S3FIFO 留下過期項目；SIM_CHANGES 19，使用者 11/6 選 A）
        w += self.enforce()
        self.w_slow_blocks += w
        for v, te in zip(self.views, tend):
            v.cpu.write(cw * BF, te)
            v.slow.write(w * self.i4, te)
        if self.check and (r % self.check == 0 or r == len(self.full) - 1):
            self.audit()

    # ---------------------------------------------------------- 一個請求（GPU 層開啟時；ICPE next）
    def _step_gpu(self, r, t, blocks):
        fmt, on_slow, rg = self.fmt, self.on_slow, self.ref_gpu
        n = len(blocks)
        g = 0
        while g < n and rg[blocks[g]] > 0:
            g += 1
        lay = [0] * g
        gpf = self.gpf
        for b in blocks[g:]:
            fm = fmt[b]
            if gpf is not None and gpf[b] >= 0:
                lay.append(4)
            elif fm == 1:
                lay.append(1)
            elif fm == 2:
                lay.append(2)
            elif on_slow[b]:
                lay.append(3)
            else:
                break
        L = len(lay)
        if self.check:
            for b in blocks[g:]:
                assert rg[b] == 0, "GPU 上的 block 不是前綴"
            for b in blocks[L:]:
                assert fmt[b] == 0 and not on_slow[b], "有存的 block 不是前綴（refcount 錯）"
                assert gpf is None or gpf[b] < 0, "預取進 GPU 的 block 不在已存的前綴裡"
        kd = self.kind[r]
        old_n = len(self.full[self.parent[r]]) if kd == 1 else 0
        tend = []
        g4 = g + (lay.count(4) if gpf is not None else 0)
        gdone = self.gpf_done if gpf is not None else None
        for v in self.views:
            tt, rs, nC, nS, nR = v.restore(lay, blocks, L, self.nall[r], t, old_n, self.pf_done, g, gdone)
            v.ttft[r], v.rest[r], v.nC[r], v.nS[r], v.nR[r] = tt, rs, nC, nS, nR
            v.nG[r] = g4
            tend.append(t + tt)
        if self.gwrite == "wb":
            self._turn_end_wb(r, blocks, tend)
        else:
            self._turn_end_wt(r, blocks, tend)
        if self.check and (r % self.check == 0 or r == len(self.full) - 1):
            self.audit()

    def _turn_end_wb(self, r, blocks, tend):
        """回寫、互斥：回合結束整個 session 留在 GPU（不寫 CPU）；舊的那份釋放（CPU 上的 BF16 沒有別的 CPU 持有者就丟，
        自己預取進 CPU 的 INT4 也丟；慢層那份照原本留著）；hook；GPU 超量 → 照 GPU 規則寫進 CPU；CPU 超量 → 照 CPU 規則。"""
        s = self.sess[r]
        old = self.sb[s] if self.sb[s] else []
        old_state = self.st[s]
        rl, rc, rg = self.ref_live, self.ref_cpu, self.ref_gpu
        gpf = self.gpf
        # 加速（逐位元相同）：session 還在 GPU、舊清單是新清單的前綴 → 前綴的 ref_live／ref_gpu +1／−1 互相抵銷
        # （ref_gpu ≥ 1，GPU 位元組不變；ref_cpu 不動，_fix 不會做事）→ 只處理新加的尾巴、不用釋放舊的
        k0 = 0
        if old_state == S_GPU and gpf is None and self._fix_fast and old and len(old) <= len(blocks) \
                and blocks[len(old) - 1] == old[-1] and blocks[:len(old)] == old:
            k0 = len(old)
        if gpf is None:                  # 加速（逐位元相同）：GPU 位元組是整數，累加順序不影響
            nn = 0
            for b in (blocks[k0:] if k0 else blocks):
                rl[b] += 1
                x = rg[b] + 1
                rg[b] = x
                if x == 1:
                    nn += 1
            self.gpu_used += nn * BF
        else:
            for b in blocks:
                rl[b] += 1
                rg[b] += 1
                if rg[b] == 1:
                    if gpf[b] >= 0:
                        self._gpf_claim(b)
                    else:
                        self.gpu_used += BF
            if self.gspf[s] > 0:
                self._g_unprefetch(s, old, push=False)   # 沒被這一輪用到的（通常沒有）
        w = 0
        if old_state == S_CPU:
            self._order_remove(s)
        fmt = self.fmt
        if old_state in (S_CPU, S_SLOW, S_GPU) and not k0:
            if self._fix_fast:
                # 加速（逐位元相同）：每塊的遞減與 _fix 合成一個迴圈——_fix(b) 只讀寫 b 自己的狀態（和整數計數），
                # 一個 session 的清單裡每塊只出現一次；_fix 只有在「沒人持有」或「CPU 沒有持有者卻有 BF16」時才做事
                fix = self._fix
                nf = 0
                isc, isg = old_state == S_CPU, old_state == S_GPU
                for b in old:
                    rl[b] -= 1
                    if isc:
                        rc[b] -= 1
                    elif isg:
                        x = rg[b] - 1
                        rg[b] = x
                        if x == 0:
                            nf += 1
                    if rl[b] == 0 or (rc[b] == 0 and fmt[b] == 1):
                        if nf:                    # （_fix 不讀 gpu_used；先補齊只是讓狀態在呼叫時一致）
                            self.gpu_used -= nf * BF
                            nf = 0
                        w += fix(b)
                self.gpu_used -= nf * BF
            else:
                for b in old:
                    rl[b] -= 1
                    if old_state == S_CPU:
                        rc[b] -= 1
                if old_state == S_GPU:
                    self._g_release(old)
                for b in old:
                    w += self._fix(b)
        if self.pf:                      # 自己預取進 CPU 的 INT4：已經搬上 GPU，不再需要（沒有預取的規則不會有 fmt＝2）
            for b in blocks:
                if fmt[b] == 2 and self.pf_owner[b] == s:
                    self.cpu_used -= self.i4
                    fmt[b] = 0
                    self.pf_owner[b] = -1
                    self.spf[s] -= 1
        if self.spf[s] <= 0:
            self.pf_sessions.discard(s)
        self.sb[s] = list(blocks)
        self.st[s] = S_GPU
        self.prev[s] = self.last[s]
        self.last[s] = r
        self.nreq[s] += 1
        if blocks:
            self._g_add(s, tend[0])
            hk = self.hook
            if (hk == "2ha" and self.nreq[s] < 2) or (hk == "odrop" and self.dead(s)):
                w += self.drop(s)
        else:
            self._g_remove(s)
        cw = self._g_enforce(s, tend[0])
        w += self.enforce()
        self.w_slow_blocks += w
        for v, te in zip(self.views, tend):
            v.cpu.write(cw * BF, te)
            v.slow.write(w * self.i4, te)

    def _turn_end_wt(self, r, blocks, tend):
        """直寫、包含：CPU 層和原本完全一樣（回合結束全部 BF16 進 CPU）；GPU 另外留一份。CPU 層的 hook 把 session 丟掉時，
        GPU 那份也丟（完美預測＝之後不會再用）；CPU 層之後因為容量把它降級／丟掉時，GPU 那份不動。"""
        fmt = self.fmt
        s = self.sess[r]
        old = self.sb[s] if self.sb[s] else []
        old_state = self.st[s]
        cw = 0
        rc, rl = self.ref_cpu, self.ref_live
        k0 = 0                            # 加速（逐位元相同），同原本路徑的前綴說明
        if old_state == S_CPU and self._fix_fast and old and len(old) <= len(blocks) and blocks[len(old) - 1] == old[-1] \
                and blocks[:len(old)] == old:
            k0 = len(old)
        for b in (blocks[k0:] if k0 else blocks):
            rl[b] += 1
            rc[b] += 1
            fm = fmt[b]
            if fm == 0:
                fmt[b] = 1
                self.cpu_used += BF
                cw += 1
            elif fm == 2:
                fmt[b] = 1
                self.cpu_used += BF - self.i4
                cw += 1
                o = self.pf_owner[b]
                if o >= 0:
                    self.spf[o] -= 1
                    self.pf_owner[b] = -1
        w = 0
        if old_state in (S_CPU, S_SLOW) and not k0:
            if self._fix_fast:            # 加速（逐位元相同），同原本路徑的說明
                fix = self._fix
                isc = old_state == S_CPU
                for b in old:
                    rl[b] -= 1
                    if isc:
                        rc[b] -= 1
                    if rl[b] == 0 or (rc[b] == 0 and fmt[b] == 1):
                        w += fix(b)
            else:
                for b in old:
                    rl[b] -= 1
                    if old_state == S_CPU:
                        rc[b] -= 1
                    w += self._fix(b)
        if self.spf[s] <= 0:
            self.pf_sessions.discard(s)
        self.sb[s] = list(blocks)
        self.st[s] = S_CPU
        self.prev[s] = self.last[s]
        self.last[s] = r
        self.nreq[s] += 1
        if blocks:
            self._order_add(s)
            hk = self.hook
            if hk == "2ha" and self.nreq[s] < 2:
                self._order_remove(s)
                self.cpu_set.add(s)
                w += self.drop(s)
            elif hk == "odrop" and self.dead(s):
                self._order_remove(s)
                self.cpu_set.add(s)
                w += self.drop(s)
        else:
            self._order_remove(s)          # 沒有完整 block：離開 CPU 的順序結構，不只 cpu_set（否則 LRU／ARC／S3FIFO 留下過期項目；SIM_CHANGES 19，使用者 11/6 選 A）
        # ---- GPU 那份
        rg = self.ref_gpu
        oldg = self.gsb[s] if self.gin[s] else None
        keep = bool(blocks) and self.st[s] != S_DROP
        gpf = self.gpf
        g0 = 0                            # 加速（逐位元相同）：GPU 上舊的那份是新清單的前綴 → 前綴的 +1／−1 互相抵銷
        if keep and oldg and gpf is None and len(oldg) <= len(blocks) and blocks[len(oldg) - 1] == oldg[-1] \
                and blocks[:len(oldg)] == oldg:
            g0 = len(oldg)
        if keep:
            for b in (blocks[g0:] if g0 else blocks):
                rg[b] += 1
                if rg[b] == 1:
                    if gpf is not None and gpf[b] >= 0:
                        self._gpf_claim(b)
                    else:
                        self.gpu_used += BF
        if gpf is not None and self.gspf[s] > 0:
            self._g_unprefetch(s, old, push=False)
        if oldg is not None and not g0:
            self._g_release(oldg)
        if keep:
            self.gsb[s] = list(blocks)
            self._g_add(s, tend[0])
        else:
            self.gsb[s] = None
            self._g_remove(s)
        self._g_enforce(s, tend[0])
        w += self.enforce()
        self.w_slow_blocks += w
        for v, te in zip(self.views, tend):
            v.cpu.write(cw * BF, te)
            v.slow.write(w * self.i4, te)

    def audit(self):
        used = 0
        for b, fm in enumerate(self.fmt):
            if fm == 1:
                used += BF
                assert self.ref_cpu[b] > 0
            elif fm == 2:
                used += self.i4
                assert self.on_slow[b]
            if self.ref_live[b] > 0:
                assert fm or self.on_slow[b] or (self.G > 0 and self.ref_gpu[b] > 0), "活著的 block 不在任何層"
        assert used == self.cpu_used, (used, self.cpu_used)
        assert self.cpu_used <= self.C or not self.cpu_set, (self.cpu_used, self.C)
        if self.G > 0:
            self.audit_gpu()
        if self.sc is not None:
            assert self.n_slow == sum(self.on_slow), (self.n_slow, sum(self.on_slow))
            assert self.n_slow <= self.sc_cap or not self.sorder.mem, (self.n_slow, self.sc_cap)
            ss = {x for x in range(self.P["R"]["n_sess"]) if self.st[x] == S_SLOW}
            assert set(self.sorder.mem) == ss, "慢層的淘汰結構和慢層狀態的 session 對不上"

    def audit_gpu(self):
        """ICPE next：GPU 層的 refcount 守恆、已用位元組、wb 的各層持有者計數。"""
        B, S = self.P["B"], self.P["R"]["n_sess"]
        cg, cc, cl = [0] * B, [0] * B, [0] * B
        for s in range(S):
            if self.gin[s]:
                assert s in self.glru
                for b in self._gbl(s):
                    cg[b] += 1
            if self.gwrite == "wb":
                assert self.gin[s] == (self.st[s] == S_GPU and bool(self.sb[s])), (s, self.st[s])
                if self.st[s] in (S_CPU, S_SLOW, S_GPU):
                    for b in self.sb[s]:
                        cl[b] += 1
                        cc[b] += self.st[s] == S_CPU
        assert len(self.glru) == sum(self.gin)
        assert cg == self.ref_gpu, "ref_gpu 和 GPU 上 session 的 block 對不上"
        gpf = self.gpf
        used = sum(BF for b in range(B) if self.ref_gpu[b] > 0 or (gpf is not None and gpf[b] >= 0))
        assert used == self.gpu_used, (used, self.gpu_used)
        if gpf is not None:
            cnt = [0] * S
            for b in range(B):
                if gpf[b] >= 0:
                    assert self.ref_gpu[b] == 0, "預取進 GPU 的塊同時是常駐"
                    assert b in self.sb[gpf[b]], "預取進 GPU 的塊不屬於它的主人"
                    cnt[gpf[b]] += 1
            assert cnt == self.gspf, "gspf 對不上"
            assert self.gpf_sessions == {x for x in range(S) if cnt[x] > 0}
        assert self.gpu_used <= self.G, (self.gpu_used, self.G)
        if self.gwrite == "wb":
            assert cl == self.ref_live and cc == self.ref_cpu, "wb：ref_live／ref_cpu 和 session 狀態對不上"
            for b in range(B):
                nslow = self.ref_live[b] - self.ref_cpu[b] - self.ref_gpu[b]
                if self.ref_cpu[b] > 0:
                    assert self.fmt[b] == 1, "CPU 狀態的持有者，但 CPU 沒有 BF16"
                if nslow > 0:
                    assert self.on_slow[b] or self.fmt[b] == 1, "慢層狀態的持有者，但慢層和 CPU 都沒有"

    def run(self):
        for r in range(len(self.full)):
            self.step(r)
        return self


# ================================================================== 載入 trace（給引擎用的形式）
_PCACHE = {}


def prep(name, rule="R1"):
    if (name, rule) in _PCACHE:
        return _PCACHE[(name, rule)]
    if name.startswith("fi:"):      # FreeInference（事後加入，x1_fi.py）；session 用資料自帶的 session_id
        import x1_fi
        P = x1_fi.prepare_fi(name[3:])
    elif name.startswith("nb:"):    # ICPE next：icpe-data 轉好的新紀錄（x1_nb.py；$X1_NB_DIR/<name>.npz）
        import x1_nb
        P = x1_nb.prepare_nb(name[3:])
    else:
        P = XT.prepare(name, rule=rule)
    R = P["R"]
    km = {"new": 0, "cont": 1, "fork": 2}
    P["kindc"] = [km[k] for k in R["kind"]]
    P["sessl"] = [int(x) for x in R["sess"]]
    P["parentl"] = [int(x) for x in R["parent"]]
    P["tl"] = [float(x) for x in P["t"]]
    P["nmax"] = max(P["nall"]) + 1
    P["ws_GiB"] = P["meta"]["n_blocks_full"] * BF / GiB
    _PCACHE[(name, rule)] = P
    return P


def make_views(K, P, keys):
    return [View(K, s, g, m, P["nmax"], len(P["full"])) for (s, g, m) in keys]


def summarize(P, v, eng, gcols=False):
    kd = np.array(P["kindc"])
    ret = kd == 1
    x = v.ttft[ret]
    span = P["tl"][-1] - P["tl"][0]
    ext = {}
    if "user_trigger" in P:
        u = ret & P["user_trigger"]
        ext = dict(n_ret_user=int(u.sum()), sum_ret_user=float(v.ttft[u].sum()),
                   mean_ret_user=float(v.ttft[u].mean()) if u.any() else float("nan"))
    return dict(**ext, sum_ret=float(x.sum()), n_req=len(kd), n_ret=int(ret.sum()), mean_ret=float(x.mean()), median_ret=float(np.median(x)),
                p90_ret=float(np.percentile(x, 90)), mean_rest_ret=float(v.rest[ret].mean()),
                total_all=float(v.ttft.sum()), mean_all=float(v.ttft.mean()),
                ret_cpu_loads=int(v.nC[ret].sum()), ret_slow_loads=int(v.nS[ret].sum()),
                ret_recompute=int(v.nR[ret].sum()),
                slow_write_GiB=v.slow.bytes_w / GiB, rho_w=v.slow.busy_w / span,
                slow_read_GiB=v.slow.bytes_r / GiB, rho_r=v.slow.busy_r / span, prefetch_GiB=v.pf_bytes / GiB,
                n_demote=eng.n_demote, n_drop=eng.n_drop, n_unprefetch=eng.n_unpf,
                cpu_residence_s_mean=(eng.res_sum / eng.n_demote) if eng.n_demote else float("nan"),
                **(_gpu_summary(v, eng, ret) if gcols else {}))


def _gpu_summary(v, eng, ret):
    """ICPE next：只在這次 sweep 有開 GPU 層時才加的欄位（GPU 層＝0 的列填 0）。"""
    on = eng.G > 0
    return dict(ret_gpu_hits=int(v.nG[ret].sum()) if on else 0, n_gpu_evict=eng.n_gevict if on else 0,
                n_gpu_expire=eng.n_gexpire if on else 0,
                gpu_wb_GiB=(eng.g_wb_blocks * BF / GiB) if on else 0.0)


# ================================================================== sweep
_K = None
OUTDIR_A = "x1"


_GPU_COLS = False     # ICPE next：這次 sweep 有沒有開 GPU 層（有＝每列多 gpu_GiB／gpu_write／gpu_policy 與 _gpu_summary 的欄位）
_NZ_COLS = False      # ICPE next：這次 sweep 有沒有雜訊（有＝每列多 noise_model／noise_lambda／noise_draw／noise_seed）
_PFX = False          # PREREG 偏離 14 (i)：--pf-expire（開＝每列多 pf_expire）
_NZO = "clamp"        # PREREG 偏離 14 (a)：--noise-overdue resample（開＝每列多 noise_overdue）
_SC_COLS = False      # ICPE next：這次 sweep 有沒有慢層容量上限（有＝每列多 slow_cap_frac／slow_policy／n_slow_clean_evict／n_slow_sess_drop）
_ENGINE = "py"        # ICPE next：py＝Engine（原本）；rust＝x1_rust 的常駐 x1fast（sweep --engine rust；_rust_sweep）
_XR = None            # x1_rust 模組（只有 --engine rust 才 import）


def uses_oracle(policy, gs, sc=None):
    """這個組合有沒有用到完美預知（雜訊只對這些有意義）。"""
    o, a, h, pf = POLICIES[policy]
    return o in ("bel", "bels") or a == "onow" or h == "odrop" or pf == "opf" or (gs is not None and gs["pol"] == "BEL") or \
        (sc is not None and sc["pol"] == "BEL")


def slow_specs(fracs, policies):
    """第二批：慢層容量上限（佔 n_blocks_full 的比例）× 慢層規則。沒給比例 → [None]（沒有上限＝原本的路徑）。"""
    if not fracs:
        return [None]
    out = []
    for fr in fracs:
        for sp in policies:
            assert sp in SLOW_POLICIES, sp
            out.append(dict(frac=float(fr), pol=sp))
    return out


def noise_specs(models, lams, draws, seeds):
    """W4（PREREG §3 B）：λ＞0 → 每個模型 × 抽法 × seed；λ＝0 只跑一次（自檢：要和沒有雜訊的 oracle 逐值相同）。"""
    out = []
    for m in models:
        for lam in lams:
            if lam == 0:
                out.append(dict(model=m, lam=0.0, draw=draws[0], seed=seeds[0]))
                continue
            for d in draws:
                for sd in seeds:
                    out.append(dict(model=m, lam=float(lam), draw=d, seed=int(sd)))
    return out


def _job(job):
    trace, rule, cpu_gib, policy, keys = job[:5]
    gs = job[5] if len(job) > 5 else None          # ICPE next：None＝GPU 層關閉（原本的路徑）
    nz = job[6] if len(job) > 6 else None          # ICPE next：None＝完美預知
    sc = job[7] if len(job) > 7 else None          # ICPE next：None＝慢層沒有上限
    K = _K
    P = prep(trace, rule)
    t0 = time.time()
    views = make_views(K, P, keys)
    gv = views[0] if (POLICIES[policy][0] == "gdsf" or (sc is not None and sc["pol"] == "GDSF")) else None
    gkw = _gpu_kw(gs)
    if nz is not None:
        gkw["noise"] = nz
    if sc is not None:
        gkw.update(slow_cap=sc["frac"], slow_policy=sc["pol"])
    if _PFX:                 # 偏離 14；沒開＝原本的呼叫
        gkw["pf_expire"] = True
    if _NZO != "clamp":
        gkw["noise_overdue"] = _NZO
    eng = Engine(P, policy, cpu_gib * GiB, views, K, gdsf_view=gv, **gkw).run()
    return _job_rows(job, P, views, eng, t0)


def _gpu_kw(gs):
    return {} if gs is None else dict(gpu_cap=int(gs["G"] * GiB), gpu_policy=gs["pol"], gpu_write=gs["write"], pf_target=gs.get("pf", "cpu"))


def _rust_post(job, jd, t0):
    """--engine rust：Rust 跑完一個 job（jd＝它的目錄）→ 讀回 views → 和 Python 引擎同一段的摘要、存 npz（_job_rows）。"""
    trace, rule, cpu_gib, policy, keys = job[:5]
    P = prep(trace, rule)
    views = make_views(_K, P, keys)
    try:
        eng = _XR.read_outputs(jd, P, views, _gpu_kw(job[5]), policy)
    finally:
        _XR.shutil.rmtree(jd, ignore_errors=True)
    return _job_rows(job, P, views, eng, t0)


def _job_rows(job, P, views, eng, t0):
    """一個 job 跑完後：每個時間設定一列（summarize）＋逐請求 TTFT 的 npz（$RUN_DIR）。Python 與 Rust 引擎共用。"""
    trace, rule, cpu_gib, policy, keys = job[:5]
    gs = job[5] if len(job) > 5 else None
    nz = job[6] if len(job) > 6 else None
    sc = job[7] if len(job) > 7 else None
    out = []
    rd = os.environ.get("RUN_DIR", "")
    arrs = {}
    for v in views:
        sm = summarize(P, v, eng, gcols=_GPU_COLS)
        row = dict(trace=trace, rule=rule, cpu_GiB=cpu_gib, cpu_frac_ws=cpu_gib / P["ws_GiB"], policy=policy,
                   slow=v.slow_name, gpu=v.gpu, restore=v.mode)
        if _GPU_COLS:
            row.update(gpu_GiB=gs["G"] if gs else 0, gpu_write=gs["write"] if gs else "none", gpu_policy=gs["pol"] if gs else "none",
                       pf_target=gs.get("pf", "cpu") if gs else "cpu")
        if _NZ_COLS:
            row.update(noise_model=nz["model"] if nz else "none", noise_lambda=nz["lam"] if nz else 0.0,
                       noise_draw=nz["draw"] if nz else "none", noise_seed=nz["seed"] if nz else -1)
        if _SC_COLS:
            row.update(slow_cap_frac=sc["frac"] if sc else 0.0, slow_policy=sc["pol"] if sc else "none")
        if _PFX:                 # 偏離 14：開了才多欄（沒開＝原本的欄位）
            row.update(pf_expire=True)
        if _NZO != "clamp":
            row.update(noise_overdue=_NZO)
        if _SC_COLS:
            sm.update(slow_cap_blocks=eng.sc_cap if sc else -1, n_slow_clean_evict=eng.n_sc_clean if sc else 0,
                      n_slow_sess_drop=eng.n_sc_drop if sc else 0)
        out.append(dict(**row, **sm, elapsed_s=time.time() - t0))
        tag = f"{v.slow_name}_{v.gpu}_{v.mode}"
        arrs[f"ttft_{tag}"] = v.ttft
        arrs[f"rest_{tag}"] = v.rest
    if rd:
        gtag = "" if gs is None else f"_G{gs['G']:g}-{gs['write']}-{gs['pol'].replace(':', '')}-pf{gs.get('pf', 'cpu')}"
        if nz is not None:
            gtag += f"_nz{nz['model']}{nz['lam']:g}{nz['draw']}s{nz['seed']}"
        if sc is not None:
            gtag += f"_sc{sc['frac']:g}{sc['pol']}"
        if _PFX:
            gtag += "_pfx"
        if _NZO != "clamp":
            gtag += f"_nzo{_NZO}"
        np.savez_compressed(os.path.join(rd, f"req_{trace}_{rule}_{cpu_gib}_{policy}{gtag}_{'-'.join(keys[0]) if len(keys) <= 2 else 'n' + str(len(keys))}.npz"),
                            **arrs)
    return out


def gpu_specs(gpu_gib, gpu_writes, gpu_policies, pf_targets=("cpu",)):
    """ICPE next：GPU 層的設定組合。只有 0 → [None]（原本的路徑）；0 和其他值並列時 0 只出現一次。
    pf＝預取目的地（cpu＝主結果；gpu＝PREREG §2 第 2 項的敏感度；只對有預取的 CPU 規則產生 job，見 jobs_for）。"""
    out = []
    for G in gpu_gib:
        if G == 0:
            out.append(None)
            continue
        for wm in gpu_writes:
            assert wm in GPU_WRITES, wm
            for gp in gpu_policies:
                parse_gpol(gp)
                for pt in pf_targets:
                    assert pt in ("cpu", "gpu"), pt
                    out.append(dict(G=G, write=wm, pol=gp, pf=pt))
    return out


def jobs_for(traces, rules, cpus, policies, modes=MODES, slows=None, gspecs=None, nspecs=None, sspecs=None):
    J = []
    slows = SLOWS if slows is None else slows
    allv = [(s, g, m) for s in slows for g in GPUS for m in MODES]    # slows＝SLOWS 時就是 ALL_VIEWS
    views = [k for k in allv if k[2] in modes]
    gspecs = [None] if not gspecs else gspecs
    nspecs = [None] if not nspecs else nspecs
    sspecs = [None] if not sspecs else sspecs
    for tr, ru in [(t_, r_) for t_ in traces for r_ in rules]:
        for c in cpus:
            for p in policies:
                for gs in gspecs:
                    o, a, h, pf = POLICIES[p]
                    if gs is not None and gs.get("pf", "cpu") == "gpu" and not pf:
                        continue          # 沒有預取的規則：預取目的地沒有差別，只跑 pf＝cpu 那份
                    for sc in sspecs:
                        for nz in nspecs:
                            if nz is not None and not uses_oracle(p, gs, sc):
                                continue      # 沒用到完美預知：雜訊沒有差別
                            tdep = pf or (gs is not None and parse_gpol(gs["pol"])[1] is not None)   # TTL 也和時間設定有關
                            if tdep:
                                for k in views:
                                    J.append((tr, ru, c, p, [k], gs, nz, sc))
                            elif o == "gdsf" or (sc is not None and sc["pol"] == "GDSF"):
                                for s in slows:
                                    for g in GPUS:
                                        J.append((tr, ru, c, p, [(s, g, m) for m in modes], gs, nz, sc))
                            else:
                                J.append((tr, ru, c, p, list(views), gs, nz, sc))
    return J


JOBS_FROM_COLS = ("trace", "rule", "cpu_GiB", "policy", "gpu_policy", "slow", "gpu")


def cell_key(tr, ru, c, p, gs, k):
    """--jobs-from：一個 view 的格子鍵。寫法和 x1_sweep.csv 相同（cpu_GiB 當浮點數；GPU 層關閉＝gpu_policy "none"）。"""
    return (tr, ru, float(c), p, gs["pol"] if gs else "none", k[0], k[1])


def filter_jobs(J, path, traces, rules):
    """--jobs-from <csv>（ICPE next，icpe-data 11/6 的 P4 請求）：jobs_for 照常展開之後，只留下「某一個 view 的
    (trace, rule, cpu_GiB, policy, gpu_policy, slow, gpu) 在清單裡」的 job；留下的 job 原封不動（整個 job 照算，
    包括清單沒列到的其他 view），順序不變。只看一個 view 的 job（有預取或 TTL）＝它自己的 (slow, gpu) 要在清單裡。
    restore、gpu_GiB、gpu_write、pf_target、雜訊、慢層上限不在鍵裡，照命令列參數。
    清單裡屬於這次 traces×rules、但這次的參數產生不出 job 的格子 → 印出來（只警告，不停）。"""
    with open(path, "rb") as fh:
        raw = fh.read()
    rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    miss = [c for c in JOBS_FROM_COLS if not rows or c not in rows[0]]
    if miss:
        raise SystemExit(f"--jobs-from {path}：沒有資料列或缺欄位 {miss}（要有 {', '.join(JOBS_FROM_COLS)}）")
    cells = set()
    for r in rows:
        v = [r[c].strip() for c in JOBS_FROM_COLS]
        cells.add((v[0], v[1], float(v[2]), v[3], v[4], v[5], v[6]))
    mine = {x for x in cells if x[0] in traces and x[1] in rules}
    keep, can = [], set()
    for j in J:
        ks = [cell_key(j[0], j[1], j[2], j[3], j[5], k) for k in j[4]]
        can.update(ks)
        if any(x in mine for x in ks):
            keep.append(j)
    un = sorted(mine - can)
    byp = {}
    for x in un:
        byp[x[3]] = byp.get(x[3], 0) + 1
    print(f"jobs_from={path} sha256={hashlib.sha256(raw).hexdigest()} rows={len(rows)} cells={len(cells)} "
          f"cells_this_sweep={len(mine)} kept_jobs={len(keep)} dropped_jobs={len(J) - len(keep)} "
          f"unmatched_cells={len(un)} {byp}", flush=True)
    for x in un[:10]:
        print(f"jobs_from unmatched（這次的參數產生不出 job）: {x}", flush=True)
    return keep


def cmd_sweep(a):
    global _K, _GPU_COLS, _NZ_COLS, _SC_COLS, _ENGINE, _XR, _PFX, _NZO
    import multiprocessing as mp
    for nm in a.slows:
        if nm not in SLOWS and nm not in EXTRA_SLOWS:
            EXTRA_SLOWS.append(nm)
    _K = constants()
    gspecs = gpu_specs(a.gpu_gib, a.gpu_write, a.gpu_policies, a.pf_target)
    _GPU_COLS = any(g is not None for g in gspecs)
    nspecs = noise_specs(a.noise_models, a.noise_lambdas, a.noise_draws, a.noise_seeds) if a.noise_models else [None]
    _NZ_COLS = any(n is not None for n in nspecs)
    sspecs = slow_specs(a.slow_cap_frac, a.slow_policies)
    _SC_COLS = any(x is not None for x in sspecs)
    _PFX = bool(getattr(a, "pf_expire", False))              # 偏離 14 (i)
    _NZO = getattr(a, "noise_overdue", "clamp") or "clamp"   # 偏離 14 (a)
    if _PFX or _NZO != "clamp":
        print(f"deviation14 pf_expire={_PFX} noise_overdue={_NZO}", flush=True)
    if getattr(a, "engine", "py") != "rust":      # --engine rust：父行程的準備在下面（x1_rust.prepare_cached）
        for tr in a.traces:
            for ru in a.rules:
                prep(tr, ru)
    J = jobs_for(a.traces, a.rules, a.cpu, a.policies, a.modes, slows=a.slows, gspecs=gspecs, nspecs=nspecs, sspecs=sspecs)
    if getattr(a, "jobs_from", None):      # ICPE next（icpe-data 11/6，P4）：只跑清單裡的格子；不給＝原本的路徑
        J = filter_jobs(J, a.jobs_from, a.traces, a.rules)
    # 重的（預取、scan）先跑
    J.sort(key=lambda j: (POLICIES[j[3]][3] is None, POLICIES[j[3]][0] not in ("bel", "bels", "pens", "pred", "lru2")))
    out_path = os.path.join(OUTDIR, a.out)
    fields = None
    new = not os.path.exists(out_path)
    fh = open(out_path, "a", newline="")
    w = None
    sha = code_sha()
    _ENGINE = getattr(a, "engine", "py")
    if _ENGINE == "rust":         # ICPE next：Rust 引擎（x1_rust.py）。code_sha 加上 Rust 原始碼的 sha，CSV 看得出是哪個引擎
        import x1_rust as _xr
        _XR = _xr
        rsha = _XR.src_sha()
        b = _XR.binary()
        fb_tr = {(j[0], j[1]) for j in J if _XR.unsupported(PLANNER, j[6], j[7], _NZO)}    # 有 job 退回 Python Engine 的紀錄（要完整的 P）
        for tr in a.traces:
            for ru in a.rules:
                lite = (tr, ru) not in fb_tr
                td, tsha, how, dt = _XR.prepare_cached(sys.modules[__name__], tr, ru, lite)
                print(f"rust trace {tr} {ru} -> {td} prep={how} {dt:.1f}s lite={lite}", flush=True)
        nfb = {}
        for j in J:
            why = _XR.unsupported(PLANNER, j[6], j[7], _NZO)
            if why:
                nfb[why] = nfb.get(why, 0) + 1
        sha = f"{sha}+rs{rsha[:8]}"
        print(f"engine=rust src_sha256={rsha} bin={b} bin_sha256={_XR.file_sha(b)} py_fallback_jobs={sum(nfb.values())} {nfb}", flush=True)
    print(f"jobs={len(J)} workers={a.workers} code_sha={sha}", flush=True)
    # ICPE next：worker 跑完 maxtasks 個 job 就換一個新的（從父行程重新 fork），不讓記憶體跨 job 一直累積
    # （TRX40 量到：每個 job 留下一些、只升不降；STATUS_TRX 10:05）。不改任何 job 的計算：job 之間本來就不共用狀態
    # （父行程先 prep 好紀錄，worker 只讀）；imap_unordered 的 chunksize＝1，所以一個 task＝一個 job。0＝不換（原本的行為）。
    mt = getattr(a, "maxtasks", 4) or None
    print(f"maxtasksperchild={mt}", flush=True)
    t0 = time.time()
    done = 0

    def put(res):
        nonlocal w, fields, done
        for row in res:
            row = dict(run_id=run_id(), ts=now_ts(), **row, code_sha=sha)
            if w is None:
                fields = list(row.keys())
                w = csv.DictWriter(fh, fieldnames=fields)
                if new:
                    w.writeheader()
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})
        fh.flush()
        done += 1
        r0 = res[0]
        print(f"[{done}/{len(J)} {time.time()-t0:.0f}s] {r0['trace']} {r0['rule']} {r0['cpu_GiB']} {r0['policy']} "
              f"views={len(res)} job={r0['elapsed_s']:.1f}s mean_ret[0]={r0['mean_ret']:.4f}", flush=True)
    rust_ok = [_ENGINE == "rust" and _XR.unsupported(PLANNER, j[6], j[7], _NZO) is None for j in J]
    JR = [j for j, ok in zip(J, rust_ok) if ok]
    JP = [j for j, ok in zip(J, rust_ok) if not ok]          # --engine py（預設）：JP＝J（同樣的順序）＝原本的路徑
    if JR:
        _rust_sweep(JR, a.workers, put)
    if JP:
        with mp.get_context("fork").Pool(a.workers, maxtasksperchild=mt) as pool:
            for res in pool.imap_unordered(_job, JP):
                put(res)
    fh.close()
    print("sweep done", time.time() - t0, flush=True)


def _rust_sweep(JR, workers, put):
    """--engine rust（ICPE next）：每份紀錄一個常駐的 x1fast（紀錄只讀一次、workers 個執行緒），同時最多 workers 個 job 在跑；
    父行程寫每個 job 的輸入目錄（x1_rust.write_job），Rust 跑完 → 小的 Pool（workers//4 個行程）讀回結果、做和 Python 引擎
    同一段的摘要與存 npz（_rust_post → _job_rows）→ put 寫 CSV。任何錯誤（Rust 行程提早結束、讀回失敗）→ 整個 sweep 停下。"""
    import multiprocessing as mp
    import queue
    events = queue.Queue()
    me = sys.modules[__name__]
    tlog = os.environ.get("X1_RUST_TIMING_LOG", "")
    srv, meta = {}, {}
    pend = deque(JR)
    inflight = nrows = 0
    # 摘要用的 worker 先 fork（之後才開讀取執行緒，不在有執行緒時 fork）；它們不跑引擎、沒有循環參照 → 不用定期換新
    with mp.get_context("fork").Pool(max(1, workers // 4)) as pool:
        try:
            while nrows < len(JR):
                while pend and inflight < workers:
                    j = pend.popleft()
                    key = (j[0], j[1])
                    if key not in srv:
                        srv[key] = _XR.Server(_XR._TDIR[key], workers, events)
                    P = prep(*key)
                    t0 = time.time()
                    views = make_views(_K, P, j[4])
                    jd = _XR.write_job(me, P, j[3], j[2] * GiB, views, _K, _gpu_kw(j[5]), nz=j[6], pfx=_PFX, nzo=_NZO)
                    meta[jd] = (j, t0)
                    srv[key].submit(jd)
                    inflight += 1
                ev = events.get()
                if ev[0] == "done":
                    inflight -= 1
                    j, t0 = meta.pop(ev[1])
                    if tlog:
                        with open(tlog, "a") as fh_:
                            fh_.write(json.dumps(dict(trace=j[0], rule=j[1], cpu=j[2], policy=j[3], n_views=len(j[4]),
                                                      gpu=j[5]["pol"] if j[5] else "off", run_s=ev[2], rust_wall_s=ev[3],
                                                      since_dispatch_s=round(time.time() - t0, 3))) + "\n")
                    pool.apply_async(_rust_post, (j, ev[1], t0), callback=lambda r: events.put(("rows", r)),
                                     error_callback=lambda e: events.put(("err", e)))
                elif ev[0] == "rows":
                    put(ev[1])
                    nrows += 1
                elif ev[0] == "err":
                    raise ev[1]
                elif ev[0] == "eof" and not ev[1].closing:
                    raise RuntimeError(f"x1fast（常駐，{ev[1].tdir}）提早結束，exit {ev[1].p.wait()}（錯誤訊息在 stderr）")
        except BaseException:
            for s_ in srv.values():
                s_.kill()
            raise
        for s_ in srv.values():
            rc = s_.close()
            if rc != 0:
                raise RuntimeError(f"x1fast（常駐，{s_.tdir}）exit {rc}")


# ================================================================== selftest（§2.8 不變量）
def cmd_selftest(a):
    global _K
    K = _K = constants()
    ok = True
    rows = []

    def rec(name, passed, detail):
        nonlocal ok
        ok &= bool(passed)
        rows.append(dict(run_id=run_id(), ts=now_ts(), invariant=name, passed=bool(passed), detail=detail))
        print(("PASS " if passed else "FAIL ") + name + " :: " + detail, flush=True)

    # --- 2a：Chan 對 m7_restore_harness.Tier（share）
    from m7_restore_harness import Tier
    from m7_sim import CB, sim_restore
    rng = random.Random(7)
    for trial in range(200):
        r, w, k = rng.uniform(0.2, 40), rng.uniform(0.2, 40), rng.choice([1.0, 1.161, 3.306])
        A = Chan("x", r, w, k)
        T = Tier("x", read_Bps=r * GiB, write_Bps=w * GiB, mode="share", k_read=k)
        tnow = 0.0
        for _ in range(50):
            tnow += rng.expovariate(20.0)
            nb = rng.choice([CB, 20971520])
            if rng.random() < 0.4:
                x1, x2 = A.write(nb, tnow), T.reserve_write(nb, tnow)
            else:
                x1, x2 = A.read(nb, tnow), T.reserve_read(nb, tnow)
            if x1 != x2:
                rec("chan_eq_Tier", False, f"trial {trial}: {x1} != {x2}")
                break
    else:
        rec("chan_eq_Tier", True, "200 個隨機讀寫序列，完成時間逐位元相同")
    # --- 2b：Cake 還原對 m7_sim.sim_restore（BF16、無反量化、CPU＋慢層混合 layout，還原前隨機的寫入背景）
    f0 = f_array(K, 300, "idle")
    bad = 0
    for trial in range(300):
        n = rng.randint(1, 120)
        slow = rng.choice(SLOWS)
        v = View(K, slow, "idle", rng.choice(["cake", "load"]), 300, 1, slow_bytes=CB, dq=0.0)
        tc = Tier("cpu", read_Bps=v.cpu.r, write_Bps=v.cpu.w, mode="share", k_read=v.cpu.k)
        ts_ = Tier("ssd", read_Bps=v.slow.r, write_Bps=v.slow.w, mode="share", k_read=v.slow.k)
        t0 = rng.uniform(0, 5)
        for _ in range(rng.randint(0, 3)):
            tw = rng.uniform(0, 5)
            nb = CB * rng.randint(1, 20)
            v.slow.write(nb, tw)
            ts_.reserve_write(nb, tw)
        lay = [rng.choice([1, 3]) for _ in range(n)]
        loc = [tc if x == 1 else ts_ for x in lay]
        tt, rs, nC, nS, nR = v.restore(lay, list(range(n)), n, n, t0, n, [0.0] * n)
        mode = "cake" if v.mode == "cake" else "load_only"
        te, meet, nc2, ns2 = sim_restore(n, loc, f0, t0, mode)
        if not (v.last_end == te and nC == nc2 and nS == ns2):
            bad += 1
            if bad <= 3:
                print("  mismatch", trial, v.last_end, te, nC, nc2, nS, ns2, flush=True)
    rec("restore_eq_m7_sim", bad == 0, f"300 個隨機 layout（BF16、d=0、cake／load_only），不一致 {bad}")
    # --- 1、3、4：PRED≡LRU（全部 12 種時間設定）、多設定一次跑≡逐一跑、refcount 守恆
    for tr in a.traces:
        P = prep(tr)
        for c in a.cpu:
            V1 = make_views(K, P, ALL_VIEWS)
            Engine(P, "LRU", c * GiB, V1, K).run()
            V2 = make_views(K, P, ALL_VIEWS)
            Engine(P, "PRED", c * GiB, V2, K).run()
            eq = all(np.array_equal(x.ttft, y.ttft) and np.array_equal(x.rest, y.rest) for x, y in zip(V1, V2))
            rec(f"pred_recency_eq_LRU[{tr},{c}GiB]", eq, "12 種時間設定的逐請求 TTFT／還原時間逐位元相同" if eq else "不同")
            k1 = ("nfs", "busy", "cake")
            V3 = make_views(K, P, [k1])
            Engine(P, "LRU", c * GiB, V3, K).run()
            i = ALL_VIEWS.index(k1)
            eq3 = np.array_equal(V3[0].ttft, V1[i].ttft)
            rec(f"multiview_eq_single[{tr},{c}GiB]", eq3, f"LRU {k1} 單獨跑 vs 12 種一次跑")
        # refcount 守恆：每一步 audit（慢，只跑前 N 個請求、最小 CPU、每種策略）
        sub = dict(P)
        nsub = a.audit_n
        sub["full"], sub["nall"], sub["tl"] = P["full"][:nsub], P["nall"][:nsub], P["tl"][:nsub]
        sub["kindc"], sub["sessl"], sub["parentl"] = P["kindc"][:nsub], P["sessl"][:nsub], P["parentl"][:nsub]
        for pol in POLICIES:
            keys = [("nfs", "idle", "cake")] if (POLICIES[pol][3] or POLICIES[pol][0] == "gdsf") else ALL_VIEWS[:2]
            vv = make_views(K, sub, keys)
            try:
                Engine(sub, pol, a.audit_cap * GiB, vv, K, gdsf_view=vv[0], check=25).run()
                rec(f"refcount_audit[{tr},{pol}]", True, f"前 {nsub} 個請求、CPU {a.audit_cap} GiB、每步重算已用位元組")
            except AssertionError as e:
                rec(f"refcount_audit[{tr},{pol}]", False, repr(e))
    p = os.path.join(OUTDIR, "x1_invariants.csv")
    new = not os.path.exists(p)
    with open(p, "a", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if new:
            wr.writeheader()
        wr.writerows(rows)
    print("selftest", "OK" if ok else "FAILED", flush=True)
    sys.exit(0 if ok else 1)


# ================================================================== analyze
def cmd_analyze(a):
    import pandas as pd
    d = pd.concat([pd.read_csv(os.path.join(OUTDIR, x)) for x in a.sweep], ignore_index=True)
    if a.src_run:
        d = d[d.run_id.isin(a.src_run)]
    assert len(d), "沒有資料"
    gax = [c for c in ("gpu_GiB", "gpu_write", "gpu_policy", "pf_target") if c in d.columns]
    if gax and len(d[gax].drop_duplicates()) > 1:   # ICPE next：這裡的樞紐表只看 policy，GPU 軸會被 aggfunc="first" 靜默蓋掉
        raise SystemExit("這份 sweep 有 GPU 層的軸（" + ", ".join(gax) + "）：x1_sim analyze 不處理，請用 code/icpe/next_sim_analyze.py")
    if a.traces:
        d = d[d.trace.isin(a.traces) | (d.trace.str.startswith("fi:") & ("freeinference" in a.traces))]
    fi = d.trace.str.startswith("fi:")
    if fi.any():
        # FreeInference：每週一個獨立重播；回來請求的平均＝Σ(每週總和) ÷ Σ(每週個數)；ρ_w 取各週最大
        g = d[fi].copy()
        if a.metric == "mean_ret_user":
            g["sum_ret"], g["n_ret"] = g["sum_ret_user"], g["n_ret_user"]
        agg = g.groupby(["rule", "cpu_GiB", "policy", "slow", "gpu", "restore"]).agg(
            sum_ret=("sum_ret", "sum"), n_ret=("n_ret", "sum"), rho_w=("rho_w", "max"),
            run_id=("run_id", "first"), n_weeks=("trace", "nunique")).reset_index()
        agg["mean_ret"] = agg.sum_ret / agg.n_ret
        agg["trace"] = "freeinference"
        print("FreeInference weeks per cell:", sorted(agg.n_weeks.unique()), " returning requests:",
              int(agg.n_ret.iloc[0]))
        d = pd.concat([d[~fi], agg], ignore_index=True)
    if a.metric not in ("mean_ret", "mean_ret_user"):
        d["mean_ret"] = d[a.metric]
    global OUTDIR_A
    OUTDIR_A = a.out_prefix
    if a.dead_h:
        for c in ("O-DROP", "O-NOW"):
            COMPONENT_POLICIES[c] = [f"{c}-H{a.dead_h}"]
        d = d[~d.policy.isin(["O-DROP", "O-NOW", "O-ALL"])].copy()
        d.loc[d.policy.str.endswith(f"-H{a.dead_h}"), "policy"] = d.policy.str.replace(f"-H{a.dead_h}", "", regex=False)
        d = d[~d.policy.str.contains("-H")]
        for c in ("O-DROP", "O-NOW"):
            COMPONENT_POLICIES[c] = [c]
    cell = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore"]
    piv = d.pivot_table(index=cell, columns="policy", values="mean_ret", aggfunc="first")
    rho = d.pivot_table(index=cell, columns="policy", values="rho_w", aggfunc="first")
    src = ",".join(sorted(d.run_id.unique()))
    rows = []
    viol = []
    for idx, r in piv.iterrows():
        for comp, rivals in RIVALS.items():
            ops = COMPONENT_POLICIES[comp]
            if any(p not in r or pd.isna(r[p]) for p in ops + rivals):
                continue
            o_best = min(ops, key=lambda p: r[p])
            rv_best = min(rivals, key=lambda p: r[p])
            To, Tr = float(r[o_best]), float(r[rv_best])
            h = (Tr - To) / Tr
            sat = bool(rho.loc[idx, o_best] >= 1 or rho.loc[idx, rv_best] >= 1)
            row = dict(zip(cell, idx))
            row.update(component=comp, oracle_policy=o_best, T_oracle=To, best_rival=rv_best, T_rival=Tr,
                       headroom=h, write_saturated=sat, rho_w_oracle=float(rho.loc[idx, o_best]),
                       rho_w_rival=float(rho.loc[idx, rv_best]), src_runs=src)
            rows.append(row)
            for rv in rivals:
                if float(r[rv]) < To - 1e-12:
                    viol.append(dict(zip(cell, idx), component=comp, oracle_policy=o_best, T_oracle=To, rival=rv,
                                     T_rival=float(r[rv]), oracle_slower_by=(To - float(r[rv])) / float(r[rv])))
        if all(p in r for p in ["O-ALL"] + NON_ORACLE):
            nb = min(NON_ORACLE, key=lambda p: r[p])
            rows.append(dict(zip(cell, idx), component="O-ALL(report)", oracle_policy="O-ALL", T_oracle=float(r["O-ALL"]),
                             best_rival=nb, T_rival=float(r[nb]), headroom=(float(r[nb]) - float(r["O-ALL"])) / float(r[nb]),
                             write_saturated=bool(rho.loc[idx, "O-ALL"] >= 1 or rho.loc[idx, nb] >= 1),
                             rho_w_oracle=float(rho.loc[idx, "O-ALL"]), rho_w_rival=float(rho.loc[idx, nb]), src_runs=src))
            for p in NON_ORACLE:
                if float(r[p]) < float(r["O-ALL"]) - 1e-12:
                    viol.append(dict(zip(cell, idx), component="O-ALL(report)", oracle_policy="O-ALL",
                                     T_oracle=float(r["O-ALL"]), rival=p, T_rival=float(r[p]),
                                     oracle_slower_by=(float(r["O-ALL"]) - float(r[p])) / float(r[p])))
    H = pd.DataFrame(rows)
    rid, tsx = run_id(), now_ts()
    H.insert(0, "ts", tsx)
    H.insert(0, "run_id", rid)
    H.to_csv(os.path.join(OUTDIR, OUTDIR_A + "_headroom.csv"), index=False)
    V = pd.DataFrame(viol)
    if len(V):
        V.insert(0, "ts", tsx)
        V.insert(0, "run_id", rid)
    V.to_csv(os.path.join(OUTDIR, OUTDIR_A + "_violations.csv"), index=False)
    # ---- 判定（§2.7）。explore/README §4-2：每種重建規則各判一次，正式判定用「每格取兩種規則中較小的 headroom」
    comps = list(RIVALS)
    HC0 = H[H.component.isin(comps)].copy()
    ck = ["trace", "cpu_GiB", "slow", "gpu", "restore", "component"]
    if HC0.rule.nunique() > 1:
        idxmin = HC0.groupby(ck).headroom.idxmin()
        HM = HC0.loc[idxmin.values].copy()
        sat = HC0.groupby(ck).write_saturated.any()
        HM["write_saturated"] = sat.loc[[tuple(x) for x in HM[ck].values]].values
        HM["rule_of_min"] = HM["rule"]
        HM["rule"] = "min"
        HC0 = pd.concat([HC0, HM], ignore_index=True)
    HC0.to_csv(os.path.join(OUTDIR, OUTDIR_A + "_headroom_components.csv"), index=False)
    verdict_rows, hot = [], []
    verdicts = {}
    for ru, HC in HC0.groupby("rule"):
        hot_r = []
        for comp in comps:
            g = HC[(HC.component == comp) & (HC.restore == "cake")]
            for (tr, sl, gp), gg in g.groupby(["trace", "slow", "gpu"]):
                ncpu = int((gg.headroom >= 0.15).sum())
                ncpu_ns = int(((gg.headroom >= 0.15) & (~gg.write_saturated)).sum())
                if ncpu >= 2:
                    hot_r.append(dict(rule=ru, component=comp, trace=tr, slow=sl, gpu=gp, n_cpu_ge15=ncpu,
                                      n_cpu_ge15_unsat=ncpu_ns,
                                      cpus_ge15=" ".join(str(int(c)) for c in gg[gg.headroom >= 0.15].cpu_GiB)))
        all_lt5 = bool((HC.headroom < 0.05).all())
        verdict = "有看頭" if hot_r else ("死路" if all_lt5 else "可能")
        verdicts[ru] = verdict
        hot += hot_r
        for comp in comps:
            g = HC[HC.component == comp]
            gc = g[g.restore == "cake"]
            verdict_rows.append(dict(run_id=rid, ts=tsx, rule=ru, component=comp, n_cells=len(g),
                                     max_headroom_cake=float(gc.headroom.max()),
                                     median_headroom_cake=float(gc.headroom.median()),
                                     n_cells_cake_ge15=int((gc.headroom >= 0.15).sum()),
                                     n_cells_cake_ge5=int((gc.headroom >= 0.05).sum()),
                                     min_headroom_all=float(g.headroom.min()),
                                     max_headroom_all=float(g.headroom.max()),
                                     n_cells_all_ge5=int((g.headroom >= 0.05).sum()),
                                     n_groups_hot=sum(1 for h in hot_r if h["component"] == comp),
                                     n_groups_hot_unsat=sum(1 for h in hot_r if h["component"] == comp and h["n_cpu_ge15_unsat"] >= 2),
                                     verdict_rule=verdict, src_runs=src))
    VR = pd.DataFrame(verdict_rows)
    VR.to_csv(os.path.join(OUTDIR, OUTDIR_A + "_verdict.csv"), index=False)
    HOT = pd.DataFrame(hot)
    if len(HOT):
        HOT.insert(0, "ts", tsx)
        HOT.insert(0, "run_id", rid)
    HOT.to_csv(os.path.join(OUTDIR, OUTDIR_A + "_hot_groups.csv"), index=False)
    # ---- Cake／只載入比值（只報告；用正式判定那一版）
    fin = "min" if "min" in verdicts else list(verdicts)[0]
    HC = HC0[HC0.rule == fin]
    ratio_rows = []
    key = ["trace", "cpu_GiB", "slow", "gpu", "component"]
    hc = HC[HC.restore == "cake"].set_index(key)
    hl = HC[HC.restore == "load"].set_index(key)
    for k in hc.index:
        if k in hl.index:
            lo = float(hl.loc[k, "headroom"])
            ca = float(hc.loc[k, "headroom"])
            ratio_rows.append(dict(zip(key, k), rule=fin, headroom_cake=ca, headroom_load=lo,
                                   ratio=(ca / lo) if lo > 0.01 else float("nan"), load_gt1pct=lo > 0.01))
    RR = pd.DataFrame(ratio_rows)
    RR.insert(0, "ts", tsx)
    RR.insert(0, "run_id", rid)
    RR.to_csv(os.path.join(OUTDIR, OUTDIR_A + "_cake_vs_load.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 1000)
    print("=== verdicts by rule:", verdicts, " 正式（每格取較小）:", verdicts.get(fin))
    print(VR.drop(columns=["run_id", "ts", "src_runs"]).to_string())
    print("=== hot groups (≥15% on ≥2 CPU sizes, Cake)")
    print(HOT.drop(columns=["run_id", "ts"]).to_string() if len(HOT) else "(none)")
    for ru in sorted(verdicts):
        HX = HC0[HC0.rule == ru]
        for mode in ("cake", "load"):
            print(f"=== headroom % [{ru}] ({mode})")
            t = HX[HX.restore == mode].pivot_table(index=["trace", "slow", "gpu", "component"], columns="cpu_GiB",
                                                   values="headroom")
            print((t * 100).round(2).to_string())
    print(f"=== best rival / oracle policy [{fin}] (Cake)")
    print(HC[HC.restore == "cake"].pivot_table(index=["trace", "slow", "gpu", "component"], columns="cpu_GiB",
                                                values="best_rival", aggfunc="first").to_string())
    vr = RR[RR.load_gt1pct] if "load_gt1pct" in RR else RR.iloc[0:0]   # FI 只跑 Cake：沒有只載入格（2026-10-17 analyze-fi 的 AttributeError）
    print("=== Cake/load ratio: cells with load headroom > 1%:", len(vr), " ratio<=0.5:",
          int((vr.ratio <= 0.5).sum()) if "ratio" in vr else 0)
    if len(vr):
        print(vr.groupby("component").ratio.describe().to_string())
    print("=== violations (oracle slower than a symmetric rival):", len(V))
    if len(V):
        print(V.groupby(["rule", "component", "rival"]).size().to_string())
        print(V.sort_values("oracle_slower_by", ascending=False).head(30).to_string())
    print("=== O-ALL(report) headroom % (Cake)")
    t = H[(H.component == "O-ALL(report)") & (H.restore == "cake")].pivot_table(index=["trace", "rule", "slow", "gpu"],
                                                                                columns="cpu_GiB", values="headroom")
    print((t * 100).round(2).to_string())


def gib_arg(x):
    """容量參數：整數照舊用 int（輸出逐位元不變）；非整數（例：111.0625＝1,777 塊）用 float。"""
    v = float(x)
    return int(v) if v.is_integer() else v


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("selftest")
    s.add_argument("--traces", nargs="+", default=TRACES)
    s.add_argument("--cpu", nargs="+", type=int, default=[128, 1024])
    s.add_argument("--audit-n", type=int, default=1500)
    s.add_argument("--audit-cap", type=int, default=64)
    s = sp.add_parser("sweep")
    s.add_argument("--traces", nargs="+", default=TRACES)
    s.add_argument("--rules", nargs="+", default=["R1", "R2"])
    s.add_argument("--modes", nargs="+", default=MODES)
    s.add_argument("--cpu", nargs="+", type=gib_arg, default=CPU_GIB)
    s.add_argument("--policies", nargs="+", default=[p for p in POLICIES if p != "PRED"])
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--maxtasks", type=int, default=int(os.environ.get("X1_MAXTASKS", "4")),
                   help="每個 worker 跑幾個 job 就換新的（重新 fork；避免記憶體累積；預設 4，可用 $X1_MAXTASKS 改）；0＝不換（原本的行為）")
    s.add_argument("--out", default="x1_sweep.csv")
    s.add_argument("--engine", choices=["py", "rust"], default=os.environ.get("X1_ENGINE", "py"),
                   help="py＝原本的 Python Engine（預設）；rust＝code/p3x/x1fast（x1_rust.py；不支援的功能那個 job 自動用 py）")
    # ICPE next（預設＝原本的行為）
    s.add_argument("--slows", nargs="+", default=SLOWS, help="慢層等級；可加 nvme807（H200 本地 NVMe 實測）")
    s.add_argument("--gpu-gib", nargs="+", type=gib_arg, default=[0], help="GPU 快取層的 KV 預算（GiB）；0＝關閉；PREREG＝111.0625（1,777 塊）")
    s.add_argument("--gpu-write", nargs="+", default=["wb"], help="wb＝回寫互斥、wt＝直寫包含")
    s.add_argument("--gpu-policies", nargs="+", default=["LRU"], help="LRU、BEL、TTLA:<秒>、TTLB:<秒>")
    s.add_argument("--pf-target", nargs="+", default=["cpu"], help="預取目的地：cpu（主結果）、gpu（敏感度；要開 GPU 層）")
    s.add_argument("--noise-models", nargs="*", default=[], help="W4：P（照 PBKV 的混合）、M（乘法誤差）；不給＝沒有雜訊")
    s.add_argument("--noise-lambdas", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3])
    s.add_argument("--noise-draws", nargs="+", default=["B1", "B2"])
    s.add_argument("--noise-seeds", nargs="+", type=int, default=list(range(10)))
    s.add_argument("--slow-cap-frac", nargs="*", type=float, default=[], help="慢層容量上限＝比例 × n_blocks_full（INT4 塊）；不給＝沒有上限")
    s.add_argument("--slow-policies", nargs="+", default=SLOW_POLICIES, help="慢層滿時的規則：LRU LRU2 ARC S3FIFO GDSF PENS BEL")
    s.add_argument("--pf-expire", action="store_true",
                   help="PREREG 偏離 14 (i)：O-PF／O-ALL 預取的副本，預測被讀的請求過了就丟（不給＝原本）")
    s.add_argument("--noise-overdue", choices=["clamp", "resample"], default="clamp",
                   help="PREREG 偏離 14 (a)：模型 P 抽到已經過去的時間 → clamp＝下一個請求（原本）、resample＝從更長的那部分重抽")
    s.add_argument("--jobs-from", default=None,
                   help="CSV（欄位 trace,rule,cpu_GiB,policy,gpu_policy,slow,gpu）：只跑「某一個 view 在清單裡」的 job（見 filter_jobs）；不給＝全部")
    s = sp.add_parser("analyze")
    s.add_argument("--sweep", nargs="+", default=["x1_sweep.csv"])
    s.add_argument("--src-run", nargs="*", default=[])
    s.add_argument("--traces", nargs="*", default=[])
    s.add_argument("--metric", default="mean_ret", help="mean_ret（主指標）／mean_ret_user（FI 只算使用者發起）／median_ret 等")
    s.add_argument("--out-prefix", default="x1")
    s.add_argument("--dead-h", type=int, default=0, help="事後穩健性：O-DROP／O-NOW／O-ALL 換成 -H<秒> 版")
    a = ap.parse_args()
    {"selftest": cmd_selftest, "sweep": cmd_sweep, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
