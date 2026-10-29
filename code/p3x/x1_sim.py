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
from collections import OrderedDict
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

    def restore(self, lay, blocks, L, n_all, t0, old_n, pf_done):
        """lay[i]∈{1: CPU BF16, 2: CPU INT4（預取）, 3: 慢層 INT4}，i<L。回傳 (TTFT, 還原舊 context, nC, nS, 重算數)。"""
        f, cpu, slow, d, i4 = self.f, self.cpu, self.slow, self.d, self.i4
        nC = nS = 0
        if self.mode == "cake" and self.planner == "stair":
            end, meet, nC, nS = self._stair(lay, blocks, L, t0, pf_done)
        elif self.mode == "cake":
            p, q, rt, lt = 0, L - 1, t0, t0
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
                    else:
                        lt = slow.read(i4, lt) + d
                        nS += 1
                    q -= 1
            end = rt if rt > lt else lt
            meet = p
        else:
            lt = t0
            for i in range(L):
                k = lay[i]
                if k == 1:
                    lt = cpu.read(BF, lt)
                    nC += 1
                elif k == 2:
                    pd = pf_done[blocks[i]]
                    lt = cpu.read(i4, lt if lt > pd else pd) + d
                    nC += 1
                else:
                    lt = slow.read(i4, lt) + d
                    nS += 1
            end = lt
            meet = 0
        self.last_end = end
        Fc = self.Fc
        base = end - t0
        ttft = base + Fc[n_all] - Fc[L]
        rest = base + (Fc[old_n] - Fc[L] if old_n > L else 0.0)
        return ttft, rest, nC, nS, meet + (n_all - L)


# ================================================================== 引擎
class Engine:
    def __init__(self, P, policy, cap_bytes, views, K, gdsf_view=None, check=False):
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
        if self.pf:
            self.advance_prefetch(self.t[r - 1] if r > 0 else t, t)
        self.cur = r
        blocks = self.full[r]
        fmt, on_slow = self.fmt, self.on_slow
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
        for b in blocks:
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
        if old_state in (S_CPU, S_SLOW):
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
            self.cpu_set.discard(s)
        w += self.enforce()
        self.w_slow_blocks += w
        for v, te in zip(self.views, tend):
            v.cpu.write(cw * BF, te)
            v.slow.write(w * self.i4, te)
        if self.check and (r % self.check == 0 or r == len(self.full) - 1):
            self.audit()

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
                assert fm or self.on_slow[b], "活著的 block 不在任何層"
        assert used == self.cpu_used, (used, self.cpu_used)
        assert self.cpu_used <= self.C or not self.cpu_set, (self.cpu_used, self.C)

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


def summarize(P, v, eng):
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
                cpu_residence_s_mean=(eng.res_sum / eng.n_demote) if eng.n_demote else float("nan"))


# ================================================================== sweep
_K = None
OUTDIR_A = "x1"


def _job(job):
    trace, rule, cpu_gib, policy, keys = job
    K = _K
    P = prep(trace, rule)
    t0 = time.time()
    views = make_views(K, P, keys)
    gv = views[0] if POLICIES[policy][0] == "gdsf" else None
    eng = Engine(P, policy, cpu_gib * GiB, views, K, gdsf_view=gv).run()
    out = []
    rd = os.environ.get("RUN_DIR", "")
    arrs = {}
    for v in views:
        sm = summarize(P, v, eng)
        out.append(dict(trace=trace, rule=rule, cpu_GiB=cpu_gib, cpu_frac_ws=cpu_gib / P["ws_GiB"], policy=policy,
                        slow=v.slow_name, gpu=v.gpu, restore=v.mode, **sm, elapsed_s=time.time() - t0))
        tag = f"{v.slow_name}_{v.gpu}_{v.mode}"
        arrs[f"ttft_{tag}"] = v.ttft
        arrs[f"rest_{tag}"] = v.rest
    if rd:
        np.savez_compressed(os.path.join(rd, f"req_{trace}_{rule}_{cpu_gib}_{policy}_{'-'.join(keys[0]) if len(keys) <= 2 else 'n' + str(len(keys))}.npz"),
                            **arrs)
    return out


def jobs_for(traces, rules, cpus, policies, modes=MODES):
    J = []
    views = [k for k in ALL_VIEWS if k[2] in modes]
    for tr, ru in [(t_, r_) for t_ in traces for r_ in rules]:
        for c in cpus:
            for p in policies:
                o, a, h, pf = POLICIES[p]
                if pf:
                    for k in views:
                        J.append((tr, ru, c, p, [k]))
                elif o == "gdsf":
                    for s in SLOWS:
                        for g in GPUS:
                            J.append((tr, ru, c, p, [(s, g, m) for m in modes]))
                else:
                    J.append((tr, ru, c, p, list(views)))
    return J


def cmd_sweep(a):
    global _K
    import multiprocessing as mp
    _K = constants()
    for tr in a.traces:
        for ru in a.rules:
            prep(tr, ru)
    J = jobs_for(a.traces, a.rules, a.cpu, a.policies, a.modes)
    # 重的（預取、scan）先跑
    J.sort(key=lambda j: (POLICIES[j[3]][3] is None, POLICIES[j[3]][0] not in ("bel", "bels", "pens", "pred", "lru2")))
    out_path = os.path.join(OUTDIR, a.out)
    fields = None
    new = not os.path.exists(out_path)
    fh = open(out_path, "a", newline="")
    w = None
    sha = code_sha()
    print(f"jobs={len(J)} workers={a.workers} code_sha={sha}", flush=True)
    t0 = time.time()
    done = 0
    with mp.get_context("fork").Pool(a.workers) as pool:
        for res in pool.imap_unordered(_job, J):
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
    fh.close()
    print("sweep done", time.time() - t0, flush=True)


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
    s.add_argument("--cpu", nargs="+", type=int, default=CPU_GIB)
    s.add_argument("--policies", nargs="+", default=[p for p in POLICIES if p != "PRED"])
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--out", default="x1_sweep.csv")
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
