#!/usr/bin/env python3
"""m16_pause_sim.py — 第三階段 C-I3：agent 工具暫停期間 KV「留 GPU／卸 CPU／卸 SSD 級／丟掉重算」的模擬器（只用 CPU）

問題（docs/stage3_20261019/lit/L3_reuse_serving_agentic.md §5 缺口 1、§7 X3；experiment_matrix.md I3 卡）：
  工具暫停期間，KV 留在 GPU／卸到 CPU／卸到 SSD 級／丟掉重算，用實測成本比 κ 做決定，能否比現成規則好？
  決策會不會隨硬體翻轉？

🔴 硬性限制（另一個 session 正在 MI300X 上計時）：
  * 這支程式**不 import torch／vllm／任何會碰 GPU 的東西**；只用 numpy 與標準庫。程式結尾有 assert。
  * 執行時外層一律 `nice -n 19 taskset -c 0-3`。
🔴 CLAUDE.md §1 規則 1：所有成本常數都從 repo 的實測檔讀出（`load_costs()`），每一個都記錄檔案路徑與 run_id；
  找不到的標 NOT_MEASURED，該平台／動作就不能用（`Costs.require()` 會拒跑）。
🔴 規則 6：Mooncake 的 hash_ids 是 512-token block；載入時斷言 (len-1)*512 < input_length <= len*512。

子命令：
  costs       印出平台×模型的成本常數與來源（CSV），含 NOT_MEASURED 清單
  trace       Mooncake trace 的 session 推法與自我檢查（JSON）
  decision-map  靜態 κ 表：各平台／路徑在各 context 長度的 CPU／SSD／重算還原成本與排名（只看還原端）
  one         跑一格（所有策略）
  scan        探索性掃描（multiprocessing 4 個行程；外層請 nice -n 19 taskset -c 0-3）
  analyze     讀 scan.csv、decisions.csv，產生 summary.json 與 tables.md
  --self-test 自測（印 ALL PASS）

模擬模型（細節見 docs/stage3_20261019/i3_pause_sim.md）：
  * C 個 slot，每個 slot 依序跑 Mooncake toolagent 推得的 agent session（多輪；context 隨輪數成長）。
  * 一輪＝(還原歷史 KV) → prefill 新 token（含第一個 token）；TTFT＝到達→prefill 完成。decode 不模擬（NOT_MEASURED）。
  * 一輪結束進入暫停（間隔分佈），策略在暫停開始時做一次決定（KEEP／CPU／SSD／DROP；TTL 類記 KEEP）。
  * 資源：GPU 計算（FIFO）、H2D、D2H、SSD 讀、SSD 寫（各自 FIFO）；HBM 與 CPU 容量以 byte 計。
  * GPU 忙：endo＝H2D 開始時 GPU 正在算 → H2D 乘實測放大倍數；exo＝外加背景負載（duty 0.7），
    計算與 H2D 都乘實測放大倍數（第二階段的 busy 校準）；off＝關掉做對照。
"""
from __future__ import annotations

import argparse
import bisect
import csv
import heapq
import io
import itertools
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

# 保險：子行程也不要看到任何 GPU（本程式本來就不會 import GPU 函式庫）
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("ROCR_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(REPO, "results", "m16_stage3_mi300x", "i3_pause_sim")
TRACE_DIR = "/mlsteam/data/tiara/datasets/traces"
HF_HUB = "/mlsteam/data/tiara/hf-cache/hub"
GIT_OLD = "52803bc^"   # 平台 A（3090）的舊結果在這個 commit 被刪（「推倒重來」）；只讀，不還原到工作樹

GiB = float(1 << 30)
MiB = float(1 << 20)
MOONCAKE_BLOCK = 512    # hash_ids 一個 = 512 token（CLAUDE.md 規則 6；載入時斷言）
CHUNK_TOK = 512         # harness（m10）的重算 chunk = 512 token
NEWTURN_TOK = 256       # m10 速度格子的新一輪 token 數（spec_grid new_tokens=256）
INF = float("inf")

RUN_ID = os.environ.get("RUN_ID", "adhoc")


def now_ts():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(p):
    return os.path.relpath(p, REPO) if p.startswith(REPO) else p


# ============================================================================ 小工具
def interp(x, xs, ys):
    """分段線性；超出兩端時取端點值（常數延伸）。呼叫端負責把「延伸」記在來源表。"""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_right(xs, x) - 1
    t = (x - xs[i]) / (xs[i + 1] - xs[i])
    return ys[i] + t * (ys[i + 1] - ys[i])


def q(v, p):
    if len(v) == 0:
        return float("nan")
    return float(np.quantile(np.asarray(v, float), p))


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def git_csv(path):
    """讀 git 歷史裡的 CSV（只讀；不動工作樹）。"""
    out = subprocess.run(["git", "-C", REPO, "show", f"{GIT_OLD}:{path}"], capture_output=True, text=True)
    if out.returncode != 0:
        raise FileNotFoundError(f"git show {GIT_OLD}:{path}: {out.stderr.strip()}")
    return list(csv.DictReader(io.StringIO(out.stdout)))


# ============================================================================ 成本層
class NotMeasured(RuntimeError):
    pass


@dataclass
class Costs:
    """一個（平台, 模型, 路徑）的成本模型。單位：秒、byte、token。所有常數都來自 sources 裡列的檔案。"""
    platform: str
    model: str
    path: str                     # harness（第二階段 Restorer 路徑）或 vllm（引擎路徑）
    bpt: int                      # KV bytes / token（BF16）
    max_tok: int                  # 成本模型的有效上限（量過的最長 context）
    # 重算：f(i) = alpha + beta*i（ms，第 i 個 512-token chunk，GPU 閒）
    alpha_ms: float = float("nan")
    beta_ms: float = float("nan")
    # H2D 還原（CPU 層、BF16、全部載入）：t = c + bytes*s（GPU 閒）
    h2d_c: float = float("nan")
    h2d_s: float = float("nan")
    h2d_busy_x: tuple = ()        # (token 長度...)
    h2d_busy_y: tuple = ()        # (忙/閒 倍數...)
    comp_busy_x: tuple = ()
    comp_busy_y: tuple = ()
    nt_idle_x: tuple = ()         # 新一輪（256 token＋第一個 token）的實測時間
    nt_idle_y: tuple = ()
    nt_busy_x: tuple = ()
    nt_busy_y: tuple = ()
    d2h_Bps: float = float("nan")
    k_duplex: float = 1.0         # H2D 與 D2H 同時進行時 H2D 變慢的倍數
    ssd_dev: str = ""
    ssd_r_Bps: float = float("nan")
    ssd_w_Bps: float = float("nan")
    ssd_k_read: float = 1.0       # 寫入進行中時讀取變慢的倍數
    hbm_real_tok: float = float("nan")
    sources: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    # 靜態（只看還原端）的 per-token 常數（vllm 路徑用）
    static: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- 取用檢查
    def require(self, *names):
        miss = [n for n in names if n in self.missing]
        if miss:
            raise NotMeasured(f"{self.platform}/{self.model}/{self.path}: NOT_MEASURED {miss} — 模擬器拒跑（不估）")

    def src(self, const, value, unit, file, run_id, how, flag=""):
        self.sources.append(dict(platform=self.platform, model=self.model, path=self.path, const=const,
                                 value=value, unit=unit, file=file, src_run_id=run_id, how=how, flag=flag))

    # ---------------------------------------------------------------- 成本函數
    def R(self, tok):
        """從位置 0 重算 tok 個 token 的時間（秒，GPU 閒）。f 線性 → 累積是二次。"""
        if tok <= 0:
            return 0.0
        x = tok / CHUNK_TOK
        return max(0.0, self.alpha_ms * x + self.beta_ms * x * (x - 1.0) / 2.0) / 1000.0

    def comp_f(self, tok, busy):
        return interp(tok, self.comp_busy_x, self.comp_busy_y) if busy else 1.0

    def h2d_f(self, tok, busy):
        return interp(tok, self.h2d_busy_x, self.h2d_busy_y) if busy else 1.0

    def recompute(self, tok_from, tok_to, busy):
        """重算 [tok_from, tok_to) 的時間（前綴已在 GPU）。"""
        if tok_to <= tok_from:
            return 0.0
        return (self.R(tok_to) - self.R(tok_from)) * self.comp_f(tok_to, busy)

    def ovh(self, tok, busy):
        """新一輪的固定開銷（第一個 token＋小 prefill 的低效率）＝實測新一輪時間 − 同長度的 chunk 化 prefill 成本。"""
        d = self.R(tok + NEWTURN_TOK) - self.R(tok)
        if not busy:
            return max(0.0, interp(tok, self.nt_idle_x, self.nt_idle_y) - d)
        if tok >= self.nt_busy_x[0]:
            return max(0.0, interp(tok, self.nt_busy_x, self.nt_busy_y) - d * self.comp_f(tok, True))
        # 忙碌的新一輪只量過 ≥32K；更短的用 32K 的「忙/閒」比例延伸（記在來源表 flag）
        r0 = self.nt_busy_y[0] / interp(self.nt_busy_x[0], self.nt_idle_x, self.nt_idle_y)
        return max(0.0, interp(tok, self.nt_idle_x, self.nt_idle_y) * r0 - d * self.comp_f(tok, True))

    def prefill(self, n_hist, delta, busy):
        """新一輪：prefill delta 個新 token（前面 n_hist 已在 GPU）＋第一個 token。"""
        delta = max(delta, 0)
        return self.recompute(n_hist, n_hist + delta, busy) + self.ovh(n_hist + delta, busy)

    def h2d(self, nbytes, tok, busy):
        if nbytes <= 0:
            return 0.0
        return (self.h2d_c + nbytes * self.h2d_s) * self.h2d_f(tok, busy)

    def d2h(self, nbytes):
        return 0.0 if nbytes <= 0 else nbytes / self.d2h_Bps

    def ssd_read(self, nbytes):
        return 0.0 if nbytes <= 0 else nbytes / self.ssd_r_Bps

    def ssd_write(self, nbytes):
        return 0.0 if nbytes <= 0 else nbytes / self.ssd_w_Bps

    def load_time(self, tier, tok, busy):
        b = tok * self.bpt
        if tier == "cpu":
            return self.h2d(b, tok, busy)
        if tier == "ssd":
            return self.ssd_read(b)
        raise ValueError(tier)

    def cake_split(self, tier, m_tok, busy):
        """Cake 式：GPU 從前面重算 [0,p)、I/O 從後面載入 [p,m)。回傳 (p, max(兩條線))，p 以 512 token 為步。"""
        best = (0, self.load_time(tier, m_tok, busy))
        p = CHUNK_TOK
        while p <= m_tok:
            t = max(self.recompute(0, p, busy), self.load_time(tier, m_tok - p, busy))
            if t < best[1]:
                best = (p, t)
            p += CHUNK_TOK
        return best

    def restore_est(self, tier, n_tok, have_tok, busy, mode):
        """還原 n_tok 歷史的預估時間（不含排隊）。tier∈cpu/ssd/drop；have_tok＝該層已有的前綴。"""
        if n_tok <= 0:
            return 0.0
        if tier == "drop":
            return self.recompute(0, n_tok, busy)
        m = min(have_tok, n_tok)
        if mode == "cake" and m > 0:
            _, t = self.cake_split(tier, m, busy)
        else:
            t = self.load_time(tier, m, busy)
        return t + self.recompute(m, n_tok, busy)


# ---------------------------------------------------------------------------- 模型設定（KV bytes／token）
MODELS = {
    "llama": dict(hub="models--unsloth--Llama-3.1-8B-Instruct", lc_fmt="fp8_token", vllm_key="b-llama8b"),
    "qwen": dict(hub="models--Qwen--Qwen2.5-7B-Instruct-1M", lc_fmt="fp8_head", vllm_key="b-qwen7b-1m"),
}


def kv_bytes_per_token(model):
    """由 HF config.json 算 BF16 KV bytes/token＝層數×KV head×head_dim×2(K,V)×2 bytes。"""
    import glob
    cfgs = sorted(glob.glob(os.path.join(HF_HUB, MODELS[model]["hub"], "snapshots", "*", "config.json")))
    if not cfgs:
        raise NotMeasured(f"找不到 {model} 的 config.json")
    c = json.load(open(cfgs[0]))
    hd = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
    return int(c["num_hidden_layers"] * c["num_key_value_heads"] * hd * 2 * 2), cfgs[0]


def _row1(rows, **kw):
    hit = [r for r in rows if all(str(r.get(k)) == str(v) for k, v in kw.items())]
    if len(hit) != 1:
        raise NotMeasured(f"期望 1 列，找到 {len(hit)} 列：{kw}")
    return hit[0]


def load_costs(platform, model, ssd_device="local"):
    """回傳 Costs（含 sources 與 missing）。platform ∈ {mi300x, mi300x_vllm, 3090}。"""
    if platform == "mi300x":
        return _load_mi300x(model, ssd_device)
    if platform == "mi300x_vllm":
        return _load_vllm_mi300x(model)
    if platform == "3090":
        return _load_3090(model)
    raise ValueError(platform)


def _load_mi300x(model, ssd_device):
    bpt, cfgp = kv_bytes_per_token(model)
    cm = Costs(platform="mi300x", model=model, path="harness", bpt=bpt, max_tok=0)
    cm.src("kv_bytes_per_token", bpt, "B/token", cfgp, "-", "layers×kv_heads×head_dim×2×2（BF16）")
    sp = os.path.join(REPO, "results", "m10_precision_mi300x", f"speed_{model}")
    # ---- 1. 每 chunk 重算 f(i)（GPU 閒）：bw.json 的中位數，並用 calib_f.csv 逐 rep 重算中位數交叉驗證
    bw = json.load(open(os.path.join(sp, "bw.json")))
    f = np.asarray(bw["f_ms_median_by_chunk"], float)
    assert int(bw["chunk_bytes"]) == bpt * CHUNK_TOK, ("單位檢查失敗：chunk_bytes != bpt*512", bw["chunk_bytes"], bpt)
    cal = read_csv(os.path.join(sp, "calib_f.csv"))
    byc = defaultdict(list)
    for r in cal:
        assert r["gpu_state"] == "idle"
        assert int(r["pos_end"]) == (int(r["chunk_idx"]) + 1) * CHUNK_TOK, "calib_f 的 pos_end 與 chunk_idx×512 不符"
        byc[int(r["chunk_idx"])].append(float(r["ms"]))
    med = np.array([np.median(byc[i]) for i in range(len(f))])
    assert np.max(np.abs(med - f)) < 0.01, ("calib_f.csv 中位數與 bw.json 不符", np.max(np.abs(med - f)))
    i = np.arange(len(f), dtype=float)
    A = np.vstack([np.ones_like(i), i]).T
    (alpha, beta), *_ = np.linalg.lstsq(A, f, rcond=None)
    cm.alpha_ms, cm.beta_ms = float(alpha), float(beta)
    resid = float(np.max(np.abs(A @ np.array([alpha, beta]) - f) / f))
    cm.src("f_alpha", round(alpha, 4), "ms/chunk(512tok)", rel(os.path.join(sp, "calib_f.csv")), bw["run_id"],
           f"f(i)=α+βi 最小平方擬合 64 個 chunk 的中位數（{len(cal)} 列）；逐 chunk 最大相對殘差 {resid:.3%}")
    cm.src("f_beta", round(beta, 5), "ms/chunk per chunk", rel(os.path.join(sp, "bw.json")), bw["run_id"], "同上")
    # ---- 2. 32K／64K／128K 的參考線（compute_only、load_only E0 CPU 層），GPU 閒／忙
    fmt = MODELS[model]["lc_fmt"]
    files = {32768: os.path.join(sp, "summary_fp8.csv"),
             65536: os.path.join(REPO, "results", "m10_longctx_mi300x", f"speed_{model}_65536_{fmt}", "summary.csv"),
             129024: os.path.join(REPO, "results", "m10_longctx_mi300x", f"speed_{model}_129024_{fmt}", "summary.csv")}
    ref = {}
    for L, p in files.items():
        rows = read_csv(p)
        for st in ("idle", "busy"):
            co = _row1(rows, length=L, bw_name="cpu", gpu_state=st, config="-", mode="compute_only")
            lo = _row1(rows, length=L, bw_name="cpu", gpu_state=st, config="E0", mode="load_only")
            for r in (co, lo):
                assert r["contaminated_rows"] == "0", ("污染列不得使用", p, L, st)
            ref[(L, st)] = dict(comp=float(co["median_t_restore"]), load=float(lo["median_t_restore"]),
                                nt=float(lo["median_ttft"]) - float(lo["median_t_restore"]),
                                run=co["run_ids"], file=rel(p), n=int(co["n"]))
    Ls = sorted(files)
    # 驗證重算擬合：用 64 個 chunk 擬的 f 去預測 64K／128K 的 compute_only（樣本外）
    cm.max_tok = max(Ls)
    for L in Ls:
        pred, meas = cm.R(L), ref[(L, "idle")]["comp"]
        err = pred / meas - 1
        cm.src(f"check_R({L})", round(pred, 4), "s", ref[(L, "idle")]["file"], ref[(L, "idle")]["run"],
               f"擬合預測 vs 實測 compute_only 閒 {meas:.4f} s（誤差 {err:+.2%}）", flag="validation")
        assert abs(err) < 0.03, ("重算擬合對實測 compute_only 誤差 >3%", L, pred, meas)
    # H2D（CPU 層全部載入，BF16 E0）：t = c + bytes*s
    xs = np.array([L * bpt for L in Ls], float)
    ys = np.array([ref[(L, "idle")]["load"] for L in Ls])
    s, c = np.polyfit(xs, ys, 1)
    cm.h2d_c, cm.h2d_s = float(max(c, 0.0)), float(s)
    for L in Ls:
        cm.src(f"load_only_cpu_idle({L})", ref[(L, "idle")]["load"], "s", ref[(L, "idle")]["file"],
               ref[(L, "idle")]["run"], f"E0 load_only cpu idle median（n={ref[(L, 'idle')]['n']}）")
    cm.src("h2d_GiBps_fit", round(1 / s / GiB, 3), "GiB/s", "(上三列)", "-",
           f"線性擬合 t=c+bytes/B；c={c*1e3:.3f} ms（<0 時取 0）；bw.json real_pinned_h2d={bw['real_pinned_h2d_GiBps']} GiB/s 交叉參考")
    cm.h2d_busy_x = tuple(Ls)
    cm.h2d_busy_y = tuple(ref[(L, "busy")]["load"] / ref[(L, "idle")]["load"] for L in Ls)
    cm.comp_busy_x = tuple(Ls)
    cm.comp_busy_y = tuple(ref[(L, "busy")]["comp"] / ref[(L, "idle")]["comp"] for L in Ls)
    for L, hy, cy in zip(Ls, cm.h2d_busy_y, cm.comp_busy_y):
        cm.src(f"busy_x_h2d({L})", round(hy, 4), "×", ref[(L, "busy")]["file"], ref[(L, "busy")]["run"],
               "load_only cpu 忙/閒（忙＝m7_busy duty 0.7）；<32K 用 32K 值（常數延伸）", flag="const-extend<32K")
        cm.src(f"busy_x_compute({L})", round(cy, 4), "×", ref[(L, "busy")]["file"], ref[(L, "busy")]["run"],
               "compute_only 忙/閒；<32K 用 32K 值（常數延伸）", flag="const-extend<32K")
    # 第一階段的 busy 校準（只當交叉參考）
    bl = json.load(open(os.path.join(REPO, "results", "m7_write_policy_mi300x", "busy_levels.json")))
    cm.src("busy2_phase1_recompute_x", bl["_measured_slowdown"]["busy2"], "×",
           "results/m7_write_policy_mi300x/busy_levels.json", ";".join(bl["_source_runs"]),
           "第一階段 C5（duty 0.7，32K 的第 32 個 chunk）；只當交叉參考", flag="cross-check")
    # ---- 3. 新一輪（256 token＋第一個 token）：ttft − t_restore
    ntx, nty = [], []
    rows32 = read_csv(files[32768])
    for L in (8192, 16384):          # 8K、16K 只有 cake 列（同一個 new_turn 操作）；有污染就不用
        try:
            r = _row1(rows32, length=L, bw_name="cpu", gpu_state="idle", config="E0", mode="cake")
        except NotMeasured:
            continue
        if r["contaminated_rows"] != "0":
            cm.notes.append(f"新一輪 {L} 的列 contaminated_rows={r['contaminated_rows']}，不用")
            continue
        ntx.append(L)
        nty.append(float(r["median_ttft"]) - float(r["median_t_restore"]))
        cm.src(f"newturn_idle({L})", round(nty[-1], 5), "s", rel(files[32768]), r["run_ids"],
               "E0 cake cpu idle：ttft−t_restore（新一輪 256 token）")
    for L in Ls:
        ntx.append(L)
        nty.append(ref[(L, "idle")]["nt"])
        cm.src(f"newturn_idle({L})", round(nty[-1], 5), "s", ref[(L, "idle")]["file"], ref[(L, "idle")]["run"],
               "E0 load_only cpu idle：ttft−t_restore")
    cm.nt_idle_x, cm.nt_idle_y = tuple(ntx), tuple(nty)
    cm.notes.append(f"新一輪開銷在 <{ntx[0]} token 用 {ntx[0]} 的值（常數延伸）")
    cm.nt_busy_x = tuple(Ls)
    cm.nt_busy_y = tuple(ref[(L, "busy")]["nt"] for L in Ls)
    for L, y in zip(Ls, cm.nt_busy_y):
        cm.src(f"newturn_busy({L})", round(y, 5), "s", ref[(L, "busy")]["file"], ref[(L, "busy")]["run"],
               "E0 load_only cpu busy：ttft−t_restore；<32K 用 32K 的忙/閒比例延伸", flag="ratio-extend<32K")
    # ---- 4. D2H、雙向干擾（第一階段 C0，同一台、Llama 的 64 MiB chunk 形狀）
    p0 = os.path.join(REPO, "results", "m7_write_policy_mi300x", "calib_c0.csv")
    c0 = read_csv(p0)
    v = [float(r["value"]) for r in c0 if r["item"] == "D2H_pinned_chunk_from_kv_view" and r["size_mib"] == "64"
         and int(r["rep"]) >= 1]
    assert all(r["unit"] == "GiB/s" for r in c0 if r["item"] == "D2H_pinned_chunk_from_kv_view")
    cm.d2h_Bps = float(np.median(v)) * GiB
    cm.src("d2h_GiBps", round(float(np.median(v)), 3), "GiB/s", rel(p0), c0[0]["run_id"],
           f"D2H_pinned_chunk_from_kv_view 64 MiB，rep≥1 的中位數（n={len(v)}；rep0 暖機不算，同 tier_params 規則）",
           flag="llama-shape" if model != "llama" else "")
    if model != "llama":
        cm.notes.append("D2H 只量過 Llama 的 64 MiB chunk 形狀；Qwen 沿用同一條鏈路的值（H2D 兩模型差 2.5%）")
    pd_ = os.path.join(REPO, "results", "m7_write_policy_mi300x", "calib_c0_duplex.csv")
    dup = read_csv(pd_)
    alone = np.median([float(r["h2d_GiBps"]) for r in dup if r["case"] == "alone"])
    withd = np.median([float(r["h2d_GiBps"]) for r in dup if r["case"] == "with_d2h"])
    cm.k_duplex = float(alone / withd)
    cm.src("k_duplex_h2d", round(cm.k_duplex, 4), "×", rel(pd_), dup[0]["run_id"],
           "H2D 單獨／同時有 D2H 的中位數比")
    # ---- 5. SSD 級（真實裝置，第一階段 C2 的持續讀寫）
    p2 = os.path.join(REPO, "results", "m7_write_policy_mi300x", "calib_c2.csv")
    c2 = read_csv(p2)
    rd = [float(r["MiBps"]) for r in c2 if r["device"] == ssd_device and r["op"] == "read_sustained"]
    wr = [float(r["MiBps"]) for r in c2 if r["device"] == ssd_device and r["op"] == "write_sustained"]
    if not rd or not wr:
        cm.missing += ["ssd_r_Bps", "ssd_w_Bps"]
    else:
        cm.ssd_dev = ssd_device
        cm.ssd_r_Bps, cm.ssd_w_Bps = float(np.median(rd)) * MiB, float(np.median(wr)) * MiB
        cm.src(f"ssd_read_{ssd_device}", float(np.median(rd)), "MiB/s", rel(p2), c2[0]["run_id"],
               f"read_sustained（64 個 64 MiB 檔、O_DIRECT）中位數，n={len(rd)}，範圍 {min(rd)}–{max(rd)}")
        cm.src(f"ssd_write_{ssd_device}", float(np.median(wr)), "MiB/s", rel(p2), c2[0]["run_id"],
               f"write_sustained 中位數，n={len(wr)}，範圍 {min(wr)}–{max(wr)}")
        tp = json.load(open(os.path.join(REPO, "results", "m7_write_policy_mi300x", "tier_params.json")))
        cm.ssd_k_read = float(tp[ssd_device]["k_read"])
        cm.src(f"k_read_{ssd_device}", cm.ssd_k_read, "×", "results/m7_write_policy_mi300x/tier_params.json",
               "20261008-132038-m7-a2share（a2 real）", "寫入進行中時讀取變慢倍數（第一階段 A2 真實裝置）")
        cm.src(f"limiter_param_{ssd_device}_read", tp[ssd_device]["read_GiBps"], "GiB/s",
               "results/m7_write_policy_mi300x/tier_params.json", ";".join(tp["source_runs"]),
               "第二階段「SSD 級／NFS 級」用的限速參數（實測最慢×0.9）；本模擬預設用實測中位數，這列只當對照",
               flag="limiter-param")
    # ---- 6. 真實 HBM 容量（vLLM 0.19 在 MI300X 的 KV token 數，BF16）
    pc = os.path.join(REPO, "results", "audit_20261010", "m5_server_paperwork_mi300x.csv")
    vk = MODELS[model]["vllm_key"]
    caps = [float(r["value"]) for r in read_csv(pc) if r["model_key"] == vk and r["metric"] ==
            "server_gpu_kv_cache_tokens_log" and r["setting"] in ("bf16", "full_gpu")]
    if caps:
        cm.hbm_real_tok = float(np.median(caps))
        cm.src("hbm_real_kv_tokens", cm.hbm_real_tok, "token", rel(pc), "m5（多個 run；見檔內 run_id 欄）",
               f"vLLM server log 的 GPU KV cache tokens（BF16／full_gpu 設定，n={len(caps)} 的中位數）")
    else:
        cm.missing.append("hbm_real_tok")
    return cm


def _load_vllm_mi300x(model):
    """MI300X 的 vLLM 引擎路徑（第一輪 M2，audit_20261010 重建）：只有還原端，D2H（store）NOT_MEASURED。"""
    bpt, cfgp = kv_bytes_per_token(model)
    cm = Costs(platform="mi300x_vllm", model=model, path="vllm", bpt=bpt, max_tok=0)
    p = os.path.join(REPO, "results", "audit_20261010", "cost_constants_mi300x_rebuilt.csv")
    vk = MODELS[model]["vllm_key"]
    rows = [r for r in read_csv(p) if r["model_key"] == vk]
    st = {}
    for t in ("cpu", "ssd", "drop"):
        r = [x for x in rows if x["tier"] == t]
        if r and r[0]["us_per_token"] not in ("", "NOT_MEASURED"):
            st[t] = float(r[0]["us_per_token"])
            cm.src(f"{t}_us_per_token", st[t], "µs/token", rel(p), r[0]["run_id"],
                   f"warm TTFT − gpu_resident，ctx={r[0]['ctx']}（vLLM OffloadingConnector 路徑）")
            st["ctx"] = int(r[0]["ctx"])
    cm.static = st
    cm.max_tok = st.get("ctx", 0)
    cm.missing += ["d2h_Bps(vllm store)", "busy_x", "position-resolved recompute(此處只用整段平均)"]
    cm.notes.append("vLLM 路徑只有還原端（ctx 固定一點）；store／D2H、GPU 忙放大 NOT_MEASURED → 只進靜態 κ 表，不跑動態模擬")
    return cm


def _load_3090(model):
    """平台 A（3090，第一輪 M2，vLLM 路徑）。舊檔在 git 52803bc^（工作樹已刪）；只讀。"""
    if model != "llama":
        cm = Costs(platform="3090", model=model, path="vllm", bpt=kv_bytes_per_token(model)[0], max_tok=0)
        cm.missing += ["all(只有 AWQ 權重的 qwen-awq 量測，與 MI300X 的 BF16 權重不可通約)"]
        return cm
    bpt, _ = kv_bytes_per_token(model)
    cm = Costs(platform="3090", model=model, path="vllm", bpt=bpt, max_tok=0)
    st = {}
    for dev in ("sata", "nvme"):
        path = f"results/m2_harness/retrieval_cost_{dev}.csv"
        rows = git_csv(path)
        assert {r["model_key"] for r in rows} == {"llama"}, "3090 retrieval 檔的 model_key 不是 llama"

        def warm(t):
            v = [float(r["ttft_ms"]) for r in rows if r["tier"] == t and r["round"] == "warm" and r["ttft_ms"]]
            return float(np.median(v)), len(v)
        ctx = max(int(r["ctx"]) for r in rows)
        g, ng = warm("gpu_resident")
        hc = sorted({r.get("host_contention", "") for r in rows})
        for t in ("cpu", "ssd", "drop"):
            w, n = warm(t)
            st[f"{t}_{dev}"] = (w - g) / ctx * 1000.0   # µs/token
            cm.src(f"{t}_us_per_token_{dev}", round(st[f'{t}_{dev}'], 3), "µs/token", f"git:{GIT_OLD}:{path}",
                   rows[0]["run_id"], f"(warm {t} 中位 {w:.1f} − gpu_resident {g:.1f}) ms ÷ ctx {ctx}（n={n}）",
                   flag="host_contention=" + "/".join(hc))
        st["ctx"] = ctx
    rp = "results/m2_harness/recompute_position.csv"
    rr = git_csv(rp)
    pts = sorted({int(r["cached_prefix_tokens"]) for r in rr if r["ttft_ms"]})
    chunk = int(rr[0]["recomputed_tokens"])
    at = {P: float(np.median([float(r["ttft_ms"]) for r in rr if int(r["cached_prefix_tokens"]) == P and r["ttft_ms"]]))
          for P in pts}
    st["recompute_pos"] = at
    st["recompute_chunk"] = chunk
    cm.src("recompute_position_ms", json.dumps(at), "ms per chunk", f"git:{GIT_OLD}:{rp}", rr[0]["run_id"],
           f"vLLM 重算 {chunk} token 的 TTFT 中位數，依已快取前綴位置；位置上限 {max(pts)}",
           flag="host_contention=" + "/".join(sorted({r['host_contention'] for r in rr})))
    cm.max_tok = max(pts) + chunk
    dp = "results/m2_harness/disk_bw_sustained.csv"
    try:
        dr = git_csv(dp)
        st["ssd_w_MiBps_sata"] = float(np.median([float(r["write_mibps"]) for r in dr]))
        cm.src("ssd_write_sata", st["ssd_w_MiBps_sata"], "MiB/s", f"git:{GIT_OLD}:{dp}", dr[0]["run_id"],
               "16 GiB 持續寫（/dev/sdh）", flag="host_contention=HEAVY")
    except FileNotFoundError:
        cm.missing.append("ssd_w_Bps")
    cm.static = st
    cm.missing += ["d2h_Bps", "busy_x_h2d", "busy_x_compute", "raw_h2d(只量過 vLLM connector 路徑)"]
    cm.notes.append("3090：D2H 與 GPU 忙放大 NOT_MEASURED → 動態模擬不能用卸載動作；只進靜態 κ 表。"
                    "所有列 host_contention=HEAVY（舊 RUNLOG 發現 5：PCIe 路徑被平行 job 灌水 26–52%）")
    return cm


# ============================================================================ Mooncake trace → session
def load_mooncake(name):
    """載入 Mooncake trace，並斷言 hash_ids 是 512-token block：(len-1)*512 < input_length <= len*512。"""
    p = os.path.join(TRACE_DIR, f"{name}_trace.jsonl")
    recs = [json.loads(l) for l in open(p)]
    bad = [i for i, r in enumerate(recs)
           if not ((len(r["hash_ids"]) - 1) * MOONCAKE_BLOCK < r["input_length"] <= len(r["hash_ids"]) * MOONCAKE_BLOCK)]
    if bad:
        raise AssertionError(f"{name}: {len(bad)}/{len(recs)} 列不滿足 512-token block 關係（例：第 {bad[:5]} 列）→ 停下回報")
    ts = [r["timestamp"] for r in recs]
    assert all(b >= a for a, b in zip(ts, ts[1:])), "timestamp 不是非遞減"
    return recs, p


def infer_sessions(recs):
    """用 hash 前綴鏈推 session（同 code/m8_trace_oracle.py 的 D4 規則，加上鏈化）。

    1. parent(B)＝更早的請求 A，A 的最後一個完整 block（A 至少 2 個完整 block，不靠共用的 block 0）
       出現在 B.hash_ids 的同一位置（Mooncake 的 hash 是前綴鏈：同位置同 hash ⇒ 整段前綴相同）；取最深的那個。
    2. 一個 parent 可能有多個 child（平行工具呼叫／分支）：最早的 child 當「下一輪」，其餘 child 另起 session（冷啟動）。
    3. 自我檢查：每一對 (A, B) 都必須 B.hash_ids[:nfull_A] == A.hash_ids[:nfull_A]、B.ts ≥ A.ts（斷言）；
       另外回報 B.input ≥ A.input + A.output 的比例（用另一欄位交叉驗證「B 含 A 的輸出」）。
    4. 每一輪可重用的歷史：若 B.in ≥ A.in + A.out → A.in + A.out；否則只保證 A 的完整 block（nfull_A×512）。
    """
    lastfull = {}
    parent = [None] * len(recs)
    for i, r in enumerate(recs):
        h = r["hash_ids"]
        for k in range(len(h) - 1, 0, -1):
            j = lastfull.get(h[k])
            if j is not None:
                parent[i] = j
                break
        nfull = r["input_length"] // MOONCAKE_BLOCK
        if nfull >= 2:
            lastfull[h[nfull - 1]] = i
    kids = defaultdict(list)
    for i, p in enumerate(parent):
        if p is not None:
            A, B = recs[p], recs[i]
            nf = A["input_length"] // MOONCAKE_BLOCK
            assert B["hash_ids"][:nf] == A["hash_ids"][:nf], ("前綴檢查失敗", p, i)
            assert B["timestamp"] >= A["timestamp"], ("時間倒流", p, i)
            kids[p].append(i)
    nxt = {p: min(v, key=lambda j: (recs[j]["timestamp"], j)) for p, v in kids.items()}
    is_cont = set(nxt.values())
    sessions = []
    n_in_out_ok = n_pairs = 0
    for h in range(len(recs)):
        if h in is_cont:
            continue
        chain = [h]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        turns = []
        for t, ri in enumerate(chain):
            r = recs[ri]
            if t == 0:
                reuse = 0
            else:
                A = recs[chain[t - 1]]
                n_pairs += 1
                if r["input_length"] >= A["input_length"] + A["output_length"]:
                    n_in_out_ok += 1
                    reuse = A["input_length"] + A["output_length"]
                else:
                    reuse = min((A["input_length"] // MOONCAKE_BLOCK) * MOONCAKE_BLOCK, r["input_length"])
                # 自我檢查：同 session 下一輪的前綴（完整 block）以上一輪為前綴
                nf = A["input_length"] // MOONCAKE_BLOCK
                assert r["hash_ids"][:nf] == A["hash_ids"][:nf]
            turns.append(dict(inp=int(r["input_length"]), out=int(r["output_length"]), reuse=int(reuse),
                              ts=r["timestamp"] / 1000.0, row=ri))
        for t in range(len(turns) - 1):
            turns[t]["gap"] = max(0.0, turns[t + 1]["ts"] - turns[t]["ts"])
        turns[-1]["gap"] = INF
        sessions.append(turns)
    fan = Counter(len(v) for v in kids.values())
    stats = dict(n_requests=len(recs), n_parent_child_pairs=sum(len(v) for v in kids.values()),
                 n_parents_with_multiple_children=sum(1 for v in kids.values() if len(v) > 1),
                 fanout_hist=dict(sorted(fan.items())[:10]),
                 n_sessions=len(sessions), n_multi_turn=sum(1 for s in sessions if len(s) > 1),
                 n_chain_pairs=n_pairs, frac_in_ge_prev_in_plus_out=(n_in_out_ok / n_pairs if n_pairs else float("nan")))
    return sessions, stats


def mooncake_pool(name="toolagent", max_tok=None):
    recs, p = load_mooncake(name)
    sessions, stats = infer_sessions(recs)
    pool = [s for s in sessions if len(s) >= 2]
    excl = 0
    if max_tok:
        keep = [s for s in pool if max(t["inp"] + t["out"] for t in s) <= max_tok]
        excl = len(pool) - len(keep)
        pool = keep
    gaps = np.array([t["gap"] for s in pool for t in s[:-1]], float)
    stats.update(dict(trace_file=p, pool_sessions=len(pool), pool_excluded_over_max_tok=excl,
                      pool_pauses=int(len(gaps)), pool_turns=int(sum(len(s) for s in pool)),
                      gap_quantiles_s={str(k): q(gaps, k) for k in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)},
                      gap_frac_zero=float(np.mean(gaps == 0)),
                      p_end=len(pool) / (len(gaps) + len(pool)),
                      max_ctx_quantiles={str(k): q([max(t["inp"] + t["out"] for t in s) for s in pool], k)
                                         for k in (0.1, 0.5, 0.9, 0.99, 1.0)}))
    return pool, gaps, stats


# ============================================================================ 間隔分佈
class GapDist:
    """已知的邊際分佈（給 Continuum 與 κ 感知策略用；Tokencake、InferCept 用自己的預測器）。
    一律以排序後的樣本／分位點陣列表示，條件期望用前綴和算。"""

    def __init__(self, kind, pts, label, params=None):
        self.kind, self.label, self.params = kind, label, params or {}
        self.g = np.sort(np.asarray(pts, float))
        self.cs = np.concatenate([[0.0], np.cumsum(self.g)])
        self.n = len(self.g)

    @staticmethod
    def empirical(samples, label):
        return GapDist("empirical", samples, label)

    @staticmethod
    def lognormal(median, sigma, label=None, K=4000):
        u = (np.arange(K) + 0.5) / K
        z = np.sqrt(2) * _erfinv(2 * u - 1)
        return GapDist("lognormal", median * np.exp(sigma * z), label or f"LN(med={median},σ={sigma})",
                       dict(median=median, sigma=sigma))

    @staticmethod
    def const(v, label=None):
        return GapDist("const", [v], label or f"const({v})", dict(v=v))

    @staticmethod
    def quantiles(pq, label=None, K=4000):
        """pq：[(p, value)...]；在 log(value) 上對 p 線性內插（兩端以端點延伸）。"""
        ps = np.array([p for p, _ in pq])
        lv = np.log(np.maximum([v for _, v in pq], 1e-6))
        u = (np.arange(K) + 0.5) / K
        return GapDist("quantiles", np.exp(np.interp(u, ps, lv)), label or f"quantiles{pq}", dict(pq=pq))

    def sample(self, rng, size):
        if self.kind == "lognormal":
            return self.params["median"] * np.exp(self.params["sigma"] * rng.standard_normal(size))
        if self.kind == "const":
            return np.full(size, self.params["v"])
        return rng.choice(self.g, size=size, replace=True)

    def cdf(self, x):
        return np.searchsorted(self.g, x, side="right") / self.n

    def sf(self, x):
        return 1.0 - self.cdf(x)

    def cond_rem(self, e, H=INF):
        """E[min(G−e, H) | G > e]。沒有 G > e 的樣本時回傳 max(樣本最大值−e, 0)＋1（已超過分佈的支撐：視為「隨時」）。"""
        i = int(np.searchsorted(self.g, e, side="right"))
        m = self.n - i
        if m <= 0:
            return 1.0
        if H == INF:
            return (self.cs[self.n] - self.cs[i]) / m - e
        j = int(np.searchsorted(self.g, e + H, side="right"))
        s = (self.cs[j] - self.cs[i]) + (self.n - j) * (e + H) - m * e
        return s / m

    def mean(self):
        return float(self.g.mean())


def _erfinv(y):
    """numpy 沒有 erfinv：用 Giles (2010) 的單精度近似再做兩步牛頓修正（|誤差| < 1e-12）。"""
    y = np.clip(np.asarray(y, float), -1 + 1e-15, 1 - 1e-15)
    w = -np.log((1.0 - y) * (1.0 + y))
    x = np.where(w < 5.0,
                 _poly(w - 2.5, [2.81022636e-08, 3.43273939e-07, -3.5233877e-06, -4.39150654e-06, 0.00021858087,
                                 -0.00125372503, -0.00417768164, 0.246640727, 1.50140941]),
                 _poly(np.sqrt(w) - 3.0, [-0.000200214257, 0.000100950558, 0.00134934322, -0.00367342844,
                                          0.00573950773, -0.0076224613, 0.00943887047, 1.00167406, 2.83297682])) * y
    erf = np.vectorize(math.erf)
    for _ in range(2):
        x = x - (erf(x) - y) / (2 / math.sqrt(math.pi) * np.exp(-x * x))
    return x


def _poly(t, cs):
    r = np.zeros_like(t) + cs[0]
    for c in cs[1:]:
        r = r * t + c
    return r


# ============================================================================ 事件模擬
PAUSED = ("paused", "done")
ACTIVE = ("waiting", "restoring", "prefill", "wait_pf")
SERVICE = ("restoring", "prefill")
ACTIONS = ("KEEP", "CPU", "SSD", "DROP")


class Job:
    __slots__ = ("kind", "s", "nbytes", "tok", "dur_fn", "on_done", "cancelled", "start", "end", "move", "copy_tok",
                 "keep_release", "cascade", "epoch")

    def __init__(self, kind, s, dur_fn, on_done, nbytes=0, tok=0):
        self.kind, self.s, self.dur_fn, self.on_done = kind, s, dur_fn, on_done
        self.nbytes, self.tok = nbytes, tok
        self.cancelled = False
        self.start = self.end = None
        self.move = False
        self.copy_tok = 0
        self.keep_release = True
        self.cascade = False
        self.epoch = 0


class Res:
    """單一伺服器 FIFO 資源（GPU 計算、H2D、D2H、SSD 讀、SSD 寫）。進行中的工作不能中途停，只能讓結果作廢。"""

    def __init__(self, sim, name):
        self.sim, self.name = sim, name
        self.q = deque()
        self.cur = None
        self.busy_time = 0.0

    def busy(self):
        return self.cur is not None

    def submit(self, job):
        self.q.append(job)
        self._kick()

    def _kick(self):
        while self.cur is None and self.q:
            job = self.q.popleft()
            if job.cancelled:
                continue
            d = job.dur_fn()
            job.start, job.end = self.sim.now, self.sim.now + d
            self.cur = job
            self.busy_time += d
            self.sim.at(job.end, self._fin, job)

    def _fin(self, job):
        self.cur = None
        job.on_done(job)
        self._kick()


class Sess:
    __slots__ = ("sid", "turns", "k", "state", "slot", "kv_tok", "on_gpu", "hbm", "cpu_tok", "ssd_tok", "cpu_res",
                 "move", "wb", "pf", "last_use", "pause_start", "ret", "dead", "t_arr", "t_adm", "t_rest",
                 "need", "n_hist", "delta", "loc_arr", "pinned_until", "last_gap", "start_t", "evicted_flag",
                 "decision", "t_start_session", "dec_index", "infer_choice", "act_tok", "t_out", "b_out", "epoch", "truncs", "jobs")

    def __init__(self, sid, turns):
        self.sid, self.turns = sid, turns
        self.k = 0
        self.state = "new"
        self.slot = None
        self.kv_tok = 0
        self.on_gpu = False
        self.hbm = 0.0
        self.cpu_tok = 0
        self.ssd_tok = 0
        self.cpu_res = 0.0
        self.move = None
        self.wb = None
        self.pf = None
        self.last_use = 0.0
        self.pause_start = 0.0
        self.ret = INF
        self.dead = False
        self.t_arr = self.t_adm = self.t_rest = 0.0
        self.need = 0.0
        self.n_hist = self.delta = 0
        self.loc_arr = ""
        self.pinned_until = -1.0
        self.last_gap = None
        self.start_t = 0.0
        self.evicted_flag = False
        self.decision = None
        self.t_start_session = 0.0
        self.dec_index = -1
        self.infer_choice = None
        self.act_tok = 0
        self.t_out = -1.0
        self.b_out = 0.0
        self.epoch = 0         # 每次「歷史被截斷」加 1；舊 epoch 的副本只有共同前綴有效
        self.truncs = []       # [(結束的 epoch, 截斷後的長度)]
        self.jobs = []         # 進行中的副本寫入（d2h／ssdw）


@dataclass
class Cfg:
    C: int = 8
    hbm_bytes: float = 0.0
    cpu_bytes: float = 0.0
    busy: str = "endo"          # off / endo / exo / both
    restore: str = "load"       # load / cake
    pressure_frac: float = 0.10  # （保留欄位；目前「壓力」＝有請求在等 HBM）
    horizon_H: float = INF      # 條件期望不截斷（「不會再回來」另由 p_end 處理）
    tokencake_alpha: float = 0.5  # L3 未給 α；取 0.5（記在報告）
    continuum_eta: float = 1.0    # Continuum 冷啟動的 η＝1（L3 §H5）


class Sim:
    def __init__(self, cm: Costs, cfg: Cfg, workload, policy, gd_known: GapDist, p_end: float):
        self.cm, self.cfg = cm, cfg
        self.now = 0.0
        self.ev = []
        self.seq = itertools.count()
        self.gpu = Res(self, "gpu")
        self.h2d_r = Res(self, "h2d")
        self.d2h_r = Res(self, "d2h")
        self.ssdw = Res(self, "ssdw")
        self.ssdr = Res(self, "ssdr")
        self.hbm_cap = cfg.hbm_bytes
        self.cpu_cap = cfg.cpu_bytes
        self.hbm_used = 0.0
        self.paused_bytes = 0.0
        self.cpu_used = 0.0
        self.pending_free = 0.0
        self.wq = deque()
        self.gd = gd_known
        self.p_end = p_end
        self.sessions = [Sess(i, t) for i, t in enumerate(workload["sessions"])]
        self.slot_gaps = list(workload["slot_gaps"])
        self.next_sess = 0
        self.n_done = 0
        self.policy = policy
        policy.sim = self
        self.busy_exo = cfg.busy in ("exo", "both")
        self.busy_endo = cfg.busy in ("endo", "both")
        # 指標
        self.turn_rows = []
        self.decisions = []
        self.jct = []
        self.acc_hbm = 0.0
        self.acc_paused = 0.0
        self.bytes = Counter()
        self.gpu_recompute_s = 0.0
        self.n_preempt = 0
        self.t_last = 0.0
        self.hist_gaps_sum = 0.0
        self.hist_gaps_n = 0
        self.T_bar_sum = 0.0
        self.T_bar_n = 0
        self.active_tok = 0
        self.n_active_service = 0
        self.n_pf = 0
        self._est_cache = {}
        self.mu_pen = 0.0      # 被趕出去的 session 回來時實際付的等待＋還原（秒）
        self.mu_bs = 0.0       # 它們不在 HBM 的 byte·秒
        self.w_sum = 0.0       # 被趕出去的 session 回來時等 HBM 的時間（秒）
        self.w_n = 0
        self.n_evict = 0       # 因為有人要空間而被趕出 HBM 的次數
        self.ev_age_sum = 0.0  # 被趕時已經暫停多久（秒）的總和
        self.evictions = []    # (kv_tok, 目的地, 真實剩餘暫停, 已結束)

    # ---------------------------------------------------------------- 事件
    def at(self, t, fn, *a):
        heapq.heappush(self.ev, (t, next(self.seq), fn, a))

    def run(self):
        for slot in range(self.cfg.C):
            self.at(self._slot_gap(), self.start_session, slot)
        total = len(self.sessions)
        while self.ev and self.n_done < total:
            t, _, fn, a = heapq.heappop(self.ev)
            dt = t - self.now
            if dt > 0:
                self.acc_hbm += self.hbm_used * dt
                self.acc_paused += self.paused_bytes * dt
            self.now = t
            fn(*a)
        self.t_last = self.now
        self.check_books()
        return self

    def check_books(self):
        """帳目一致性（每一格跑完都檢查）。"""
        S = self.sessions
        assert abs(self.hbm_used - sum(x.hbm for x in S)) < 1, "HBM 帳"
        assert abs(self.cpu_used - sum(x.cpu_res for x in S)) < 1, "CPU 帳"
        assert abs(self.paused_bytes - sum(x.hbm for x in S if x.state in PAUSED)) < 1, "暫停中 HBM 帳"
        assert abs(self.pending_free - sum(x.hbm for x in S if x.move is not None)) < 1, "搬移中 HBM 帳"
        assert self.active_tok == sum(x.act_tok for x in S if x.state in ACTIVE), "服務中 token 帳"
        assert self.n_active_service == sum(1 for x in S if x.state in SERVICE), "服務中計數"
        assert self.hbm_used <= self.hbm_cap * (1 + 1e-9) + 1 and self.cpu_used <= self.cpu_cap * (1 + 1e-9) + 1
        assert all(x.cpu_tok <= x.kv_tok and x.ssd_tok <= x.kv_tok for x in S), "副本不可能比 KV 長"
        for x in S:
            t = max([x.cpu_tok] + [j.copy_tok for j in self._inflight_d2h(x)])
            assert abs(x.cpu_res - t * self.cm.bpt) < 1, "CPU 保留量≠max(副本, 進行中寫入)"

    def _slot_gap(self):
        return self.slot_gaps.pop() if self.slot_gaps else 0.0

    # ---------------------------------------------------------------- 帳
    def set_hbm(self, s, b):
        d = b - s.hbm
        self.hbm_used += d
        if s.state in PAUSED:
            self.paused_bytes += d
        s.hbm = b
        if self.hbm_used > self.hbm_cap * (1 + 1e-9) + 1:
            raise AssertionError(f"HBM 超用 {self.hbm_used / GiB:.3f} > {self.hbm_cap / GiB:.3f} GiB")

    def set_state(self, s, st):
        was, now_p = s.state in PAUSED, st in PAUSED
        if was and not now_p:
            self.paused_bytes -= s.hbm
        elif now_p and not was:
            self.paused_bytes += s.hbm
        wa, na = s.state in ACTIVE, st in ACTIVE
        if wa and not na:
            self.active_tok -= s.act_tok
            s.act_tok = 0
        elif na and not wa:
            s.act_tok = s.turns[s.k]["inp"] + s.turns[s.k]["out"]
            self.active_tok += s.act_tok
        ws, ns = s.state in SERVICE, st in SERVICE
        if ws and not ns:
            self.n_active_service -= 1
        elif ns and not ws:
            self.n_active_service += 1
        s.state = st

    def set_cpu(self, s, b):
        self.cpu_used += b - s.cpu_res
        s.cpu_res = b
        if self.cpu_used > self.cpu_cap * (1 + 1e-9) + 1:
            raise AssertionError("CPU 超用")

    def hbm_free(self):
        return self.hbm_cap - self.hbm_used

    def pressure(self):
        """瞬間的容量壓力＝有請求在等 HBM（Tokencake 原文「有等待請求」）。只有自己佔滿 HBM 不算壓力。"""
        return bool(self.wq)

    def gpu_busy_for(self):
        return self.busy_exo or (self.busy_endo and self.gpu.busy())

    def victims(self):
        return [s for s in self.sessions if s.state in PAUSED and s.on_gpu and s.move is None and s.pf is None
                and s.hbm > 0]

    # ---------------------------------------------------------------- session 生命週期
    def start_session(self, slot):
        if self.next_sess >= len(self.sessions):
            return
        s = self.sessions[self.next_sess]
        self.next_sess += 1
        s.slot = slot
        s.t_start_session = self.now
        self.arrive(s)

    def arrive(self, s):
        turn = s.turns[s.k]
        s.t_arr = self.now
        s.n_hist = min(turn["reuse"], s.kv_tok) if s.k > 0 else 0
        s.delta = turn["inp"] - s.n_hist
        # 歷史被截斷（下一輪沒有包含上一輪的全部輸出）：超出 n_hist 的副本作廢；
        # 進行中的寫入照寫，但完成時只認「和新內容共同的前綴」（epoch）
        if s.k > 0 and s.n_hist < s.kv_tok:
            s.truncs.append((s.epoch, s.n_hist))
            s.epoch += 1
        s.cpu_tok = min(s.cpu_tok, s.n_hist)
        s.ssd_tok = min(s.ssd_tok, s.n_hist)
        self._recalc_cpu(s)
        s.need = (turn["inp"] + turn["out"]) * self.cm.bpt
        if s.k == 0:
            s.loc_arr = "cold"
        elif s.on_gpu:
            s.loc_arr = "gpu"
        elif s.pf is not None:
            s.loc_arr = "prefetching"
        elif s.cpu_tok >= s.n_hist > 0:
            s.loc_arr = "cpu"
        elif s.ssd_tok >= s.n_hist > 0:
            s.loc_arr = "ssd"
        elif max(s.cpu_tok, s.ssd_tok) > 0:
            s.loc_arr = "partial"
        else:
            s.loc_arr = "none"
        s.evicted_flag = s.k > 0 and s.loc_arr != "gpu"
        if s.evicted_flag and s.t_out >= 0:
            self.mu_bs += s.b_out * (self.now - s.t_out)
        if s.k > 0:
            self.policy.on_return(s)
        if s.move is not None:            # 還在搬出：KV 仍在 HBM → 取消釋放（副本照寫完）
            s.move.keep_release = False
            self.pending_free -= s.hbm
            s.move = None
            s.loc_arr = "gpu"
            s.evicted_flag = False
        self.set_state(s, "waiting")
        if s.pf is not None:              # 預取進行中：等它做完
            self.set_state(s, "wait_pf")
            return
        self.wq.append(s)
        self.pump()

    def pump(self):
        guard = 0
        while self.wq:
            guard += 1
            if guard > 10000:
                raise AssertionError("pump 迴圈")
            s = self.wq[0]
            extra = s.need - s.hbm
            if extra <= self.hbm_free() + 1e-6:
                self.wq.popleft()
                self.admit(s)
                continue
            deficit = extra - self.hbm_free() - self.pending_free
            if deficit > 1e-6:
                self.policy.make_room(deficit)
            if extra <= self.hbm_free() + 1e-6:
                continue
            if self.pending_free > 1e-6:
                break                     # 等搬出完成
            # 沒有可以趕的，也沒有正在搬出的：活躍中的 session 結束後會變成可趕；都沒有 → 搶占（vLLM 的作法：丟最晚到的）
            if self.n_active_service > 0 or self.n_pf > 0:
                break                     # 服務中的 session 結束（或預取完成）後會變成可趕
            cand = [x for x in list(self.wq)[1:] if x.on_gpu and x.hbm > 0]
            if not cand:
                if s.on_gpu and s.hbm > 0 and len(self.wq) == 1:
                    raise AssertionError("容量不足以放下單一 session（容量下限設定錯誤）")
                break
            v = cand[-1]
            self.n_preempt += 1
            self._drop_gpu(v)
        return

    def admit(self, s):
        s.t_adm = self.now
        self.set_hbm(s, max(s.need, 0.0))
        self.set_state(s, "restoring")
        if s.k == 0 or s.on_gpu or s.n_hist <= 0:
            s.on_gpu = True
            self.after_restore(s)
        else:
            self.restore(s, prefetch=False)

    # ---------------------------------------------------------------- 還原
    def _src(self, s):
        c, d = min(s.cpu_tok, s.n_hist), min(s.ssd_tok, s.n_hist)
        if c >= d and c > 0:
            return "cpu", c
        if d > 0:
            return "ssd", d
        return None, 0

    def _load_job(self, s, tier, m_tok, done):
        nbytes = m_tok * self.cm.bpt
        cm = self.cm
        if tier == "cpu":
            def dur():
                f = cm.h2d(nbytes, m_tok, self.gpu_busy_for())
                if self.d2h_r.busy():
                    f *= cm.k_duplex
                return f
            self.bytes["h2d"] += nbytes
            self.h2d_r.submit(Job("h2d", s, dur, done, nbytes, m_tok))
        else:
            def dur():
                f = cm.ssd_read(nbytes)
                if self.ssdw.busy():
                    f *= cm.ssd_k_read
                return f
            self.bytes["ssd_read"] += nbytes
            self.ssdr.submit(Job("ssdr", s, dur, done, nbytes, m_tok))

    def _gpu_job(self, s, seconds_fn, done, kind="gpu"):
        self.gpu.submit(Job(kind, s, seconds_fn, done))

    def restore(self, s, prefetch):
        """把 s 的歷史（n_hist token）還原到 HBM。prefetch=True 時還原完不接 prefill。"""
        cm = self.cm
        n = s.n_hist if not prefetch else s.kv_tok
        tier, m = self._src_for(s, n)
        busy = self.busy_exo

        def finish(_=None):
            s.on_gpu = True
            if prefetch:
                s.pf = None
                self.n_pf -= 1
                if s.state == "wait_pf":
                    self.set_state(s, "waiting")
                    self.wq.append(s)
                self.pump()
                return
            self.after_restore(s)

        tail = (m, n)

        def tail_job(_=None):
            if tail[1] > tail[0]:
                def dur():
                    t = cm.recompute(tail[0], tail[1], self.busy_exo)
                    self.gpu_recompute_s += t
                    return t
                self._gpu_job(s, dur, finish, "recompute")
            else:
                finish()

        if m <= 0:
            tail_job()
            return
        if self.cfg.restore == "cake":
            p, _ = cm.cake_split(tier, m, busy)
        else:
            p = 0
        if p <= 0:
            self._load_job(s, tier, m, tail_job)
            return
        st = {"n": 2}

        def both(_=None):
            st["n"] -= 1
            if st["n"] == 0:
                tail_job()

        def dur_front():
            t = cm.recompute(0, p, self.busy_exo)
            self.gpu_recompute_s += t
            return t
        self._gpu_job(s, dur_front, both, "recompute")
        self._load_job(s, tier, m - p, both)

    def _src_for(self, s, n):
        c, d = min(s.cpu_tok, n), min(s.ssd_tok, n)
        if c >= d and c > 0:
            return "cpu", c
        if d > 0:
            return "ssd", d
        return None, 0

    def after_restore(self, s):
        s.t_rest = self.now
        s.on_gpu = True
        self.set_state(s, "prefill")
        cm = self.cm
        n, dl = s.n_hist, s.delta

        def dur():
            return cm.prefill(n, dl, self.busy_exo)
        self._gpu_job(s, dur, self.end_turn, "prefill")

    def end_turn(self, job):
        s = job.s
        turn = s.turns[s.k]
        ttft = self.now - s.t_arr
        self.turn_rows.append((s.sid, s.k, s.n_hist, s.delta, ttft, s.t_adm - s.t_arr, s.t_rest - s.t_adm,
                               self.now - s.t_rest, s.loc_arr, s.turns[s.k - 1]["gap"] if s.k > 0 else -1.0))
        if s.evicted_flag:
            self.T_bar_sum += s.t_rest - s.t_arr
            self.T_bar_n += 1
            self.mu_pen += s.t_rest - s.t_arr
            self.w_sum += s.t_adm - s.t_arr
            self.w_n += 1
        s.kv_tok = turn["inp"] + turn["out"]
        s.on_gpu = True
        s.last_use = self.now
        gap = turn["gap"]
        s.k += 1
        s.pause_start = self.now
        if s.k < len(s.turns):
            s.ret = self.now + gap
            s.dead = False
            self.set_state(s, "paused")
            self.at(s.ret, self.arrive, s)
            if s.last_gap is not None or True:
                pass
        else:
            s.ret = INF
            s.dead = True
            self.set_state(s, "done")
            self.jct.append(self.now - s.t_start_session)
            self.n_done += 1
            self.at(self.now + self._slot_gap(), self.start_session, s.slot)
        self.policy.on_turn_end(s)
        d0 = len(self.decisions)
        self.policy.on_pause(s)
        assert len(self.decisions) == d0 + 1, f"{self.policy.name}: 一次暫停必須恰好記一個決定"
        self.pump()

    def on_return_gap(self, s):
        """session 回來時，把真實的暫停長度加進線上歷史（Tokencake 用）。"""
        g = self.now - s.pause_start
        s.last_gap = g
        self.hist_gaps_sum += g
        self.hist_gaps_n += 1

    # ---------------------------------------------------------------- 決定與搬移
    def log_decision(self, s, action, eager=False):
        """暫停開始的決定。KEEP＝只留在 HBM（不寫下層副本）；CPU／SSD＝寫一份到該層（eager＝同時立刻釋放 HBM，
        否則 HBM 當快取留著、有人要空間時才放）；DROP＝立刻釋放、不寫副本。"""
        assert action in ACTIONS
        if action == "DROP":
            eager = True
        self.decisions.append((s.sid, s.k, action, s.kv_tok, (s.ret - self.now) if not s.dead else INF, s.dead,
                               self.pressure(), eager))

    def _drop_gpu(self, v):
        """立刻釋放 v 在 HBM 的 KV（DROP）。進行中的寫出作廢（物理上中止）。"""
        if v.wb is not None:
            v.wb.cancelled = True
            v.wb = None
        if v.move is not None:
            v.move.keep_release = False
            v.move.cancelled = True
            self.pending_free -= v.hbm
            v.move = None
        self._recalc_cpu(v)
        self._left(v)
        self.set_hbm(v, 0.0)

    # ---------------------------------------------------------------- 副本帳（epoch 與 CPU 保留量）
    def _valid(self, s, j):
        """j 完成時真正有效的副本長度：舊 epoch 的寫入只認和現在內容共同的前綴。"""
        L = j.copy_tok
        for e_end, ln in s.truncs:
            if e_end >= j.epoch:
                L = min(L, ln)
        return min(L, s.kv_tok)

    def _inflight_d2h(self, s):
        return [j for j in s.jobs if j.kind == "d2h" and not j.cancelled]

    def _recalc_cpu(self, s):
        """CPU 佔用＝max(已完成的副本, 進行中寫入的目標長度) × bytes/token。"""
        t = max([s.cpu_tok] + [j.copy_tok for j in self._inflight_d2h(s)])
        self.set_cpu(s, t * self.cm.bpt)

    def _submit_copy(self, s, kind, nbytes, copy_tok, done, move=False, cascade=False):
        cm = self.cm
        dur = (lambda: cm.d2h(nbytes)) if kind == "d2h" else (lambda: cm.ssd_write(nbytes))
        j = Job(kind, s, dur, done, nbytes, copy_tok)
        j.copy_tok, j.move, j.cascade, j.epoch = copy_tok, move, cascade, s.epoch
        s.jobs.append(j)
        if kind == "d2h":
            self.bytes["d2h"] += nbytes
            self._recalc_cpu(s)
            self.d2h_r.submit(j)
        else:
            self.bytes["ssd_write"] += nbytes
            self.ssdw.submit(j)
        return j

    def _left(self, v):
        if v.on_gpu:
            v.t_out = self.now
            v.b_out = v.kv_tok * self.cm.bpt
        v.on_gpu = False

    def t_ev_bar(self):
        """暫停中的 session 平均撐多久才被（有人要空間而）趕出 HBM；沒有紀錄時 inf。"""
        return self.ev_age_sum / self.n_evict if self.n_evict else INF

    def w_bar(self):
        """被趕出去的 session 回來時平均要等多久才拿到 HBM（線上累計）。"""
        return self.w_sum / self.w_n if self.w_n else 0.0

    def mu_bar(self):
        """HBM 的影子價格（秒／(byte·秒)）：被趕出去的 session 實際付的代價 ÷ 它們不在 HBM 的 byte·秒（線上累計）。"""
        return self.mu_pen / self.mu_bs if self.mu_bs > 0 else 0.0

    def cpu_make_room(self, nbytes, exclude):
        """CPU 層 LRU 淘汰（所有策略共用的機制）。回傳是否有足夠空間。"""
        if self.cpu_cap - self.cpu_used >= nbytes:
            return True
        cands = sorted((x for x in self.sessions if x is not exclude and x.cpu_tok > 0 and not self._inflight_d2h(x)
                        and x.state not in ("restoring",) and x.pf is None), key=lambda x: x.last_use)
        for x in cands:
            if self.cpu_cap - self.cpu_used >= nbytes:
                break
            x.cpu_tok = 0
            self._recalc_cpu(x)
        return self.cpu_cap - self.cpu_used >= nbytes

    def cpu_room_possible(self, nbytes, exclude):
        if self.cpu_cap - self.cpu_used >= nbytes:
            return True
        ev = sum(x.cpu_res for x in self.sessions if x is not exclude and x.cpu_tok > 0 and not self._inflight_d2h(x)
                 and x.state != "restoring" and x.pf is None)
        return self.cpu_cap - self.cpu_used + ev >= nbytes

    def write_behind(self, s, cascade=False):
        """把 s 的 KV 複製一份到 CPU（不釋放 HBM）。只寫還沒有副本的部分。
        cascade＝寫完 CPU 再串寫到 SSD 級（vLLM 0.28 TieringOffloadingSpec：「Always offload to all tiers」）。"""
        if s.wb is not None or s.move is not None or s.cpu_tok >= s.kv_tok:
            return False
        delta = (s.kv_tok - s.cpu_tok) * self.cm.bpt
        if not self.cpu_make_room(delta, s):
            return False
        s.wb = self._submit_copy(s, "d2h", delta, s.kv_tok, self._wb_done, cascade=cascade)
        return True

    def _wb_done(self, j):
        s = j.s
        if j in s.jobs:
            s.jobs.remove(j)
        if s.wb is j:
            s.wb = None
        if j.cancelled:
            self._recalc_cpu(s)
            return
        got = self._valid(s, j)
        s.cpu_tok = max(s.cpu_tok, got)
        self._recalc_cpu(s)
        if j.cascade and s.ssd_tok < got and not math.isnan(self.cm.ssd_w_Bps):
            self._submit_copy(s, "ssdw", (got - s.ssd_tok) * self.cm.bpt, got, self._ssd_done)
        if j.move and j.keep_release:
            self._release_after_move(s, j)

    def evict(self, v, dest, demand=True):
        """把 v 從 HBM 趕出到 dest。回傳 'instant'／'pending'／'fail'。demand＝因為有人要空間（計入 n_evict）。"""
        cm = self.cm
        if demand and v.on_gpu:
            self.n_evict += 1
            self.ev_age_sum += self.now - v.pause_start
            self.evictions.append((v.kv_tok, dest, (v.ret - self.now) if not v.dead else INF, v.dead))
        if dest == "drop":
            self._drop_gpu(v)
            self.pump_later()
            return "instant"
        if dest == "cpu":
            if v.cpu_tok >= v.kv_tok:
                self._left(v)
                self.set_hbm(v, 0.0)
                return "instant"
            if v.wb is not None:          # 寫入中的副本 → 改成搬移，寫完才放 HBM
                v.wb.move = True
                v.move, v.wb = v.wb, None
                self.pending_free += v.hbm
                return "pending"
            delta = (v.kv_tok - v.cpu_tok) * cm.bpt
            if not self.cpu_make_room(delta, v):
                return "fail"
            v.move = self._submit_copy(v, "d2h", delta, v.kv_tok, self._wb_done, move=True)
            self.pending_free += v.hbm
            return "pending"
        if dest == "ssd":
            if v.ssd_tok >= v.kv_tok:
                self._left(v)
                self.set_hbm(v, 0.0)
                return "instant"
            if math.isnan(cm.ssd_w_Bps):
                return "fail"
            delta = (v.kv_tok - v.ssd_tok) * cm.bpt
            v.move = self._submit_copy(v, "ssdw", delta, v.kv_tok, self._ssd_done, move=True)
            self.pending_free += v.hbm
            return "pending"
        raise ValueError(dest)

    def _ssd_done(self, j):
        s = j.s
        if j in s.jobs:
            s.jobs.remove(j)
        if j.cancelled:
            return
        s.ssd_tok = max(s.ssd_tok, self._valid(s, j))
        if j.move and j.keep_release:
            self._release_after_move(s, j)
        elif s.move is j:
            s.move = None

    def _release_after_move(self, s, j):
        if s.move is j:
            s.move = None
            self.pending_free -= s.hbm
        if s.state in PAUSED and s.on_gpu:
            self._left(s)
            self.set_hbm(s, 0.0)
        self.pump()

    def pump_later(self):
        pass

    def prefetch(self, s, k=None):
        """主動載回（Tokencake、oracle）：只在 HBM 有空位時做，不為它趕別人。k＝排程當時的輪次（過期就不做）。"""
        if k is not None and s.k != k:
            return
        if s.state not in PAUSED or s.on_gpu or s.pf is not None or s.dead or s.move is not None:
            return
        if s.cpu_tok <= 0 and s.ssd_tok <= 0:
            return
        b = s.kv_tok * self.cm.bpt
        if self.hbm_free() < b or self.wq:
            return
        self.set_hbm(s, b)
        s.pf = True
        self.n_pf += 1
        self.restore(s, prefetch=True)

    # ---------------------------------------------------------------- 成本估計（給策略用；用設定檔，不看佇列）
    def est(self, tier, s, n=None):
        n = s.kv_tok if n is None else n
        if tier == "drop":
            # 有部分副本時，丟掉＝載入副本＋重算其餘
            src, m = self._src_for(s, n) if (s.cpu_tok or s.ssd_tok) else (None, 0)
            key = ("drop", src, n, m)
        else:
            key = (tier, n)
        v = self._est_cache.get(key)
        if v is not None:
            return v
        b = self.busy_exo
        if tier in ("cpu", "ssd"):
            v = self.cm.restore_est(tier, n, n, b, self.cfg.restore)
        elif m > 0:
            v = self.cm.restore_est(src, n, m, b, self.cfg.restore)
        else:
            v = self.cm.recompute(0, n, b)
        self._est_cache[key] = v
        return v

    # ---------------------------------------------------------------- 指標
    def summary(self):
        tt = np.array([r[4] for r in self.turn_rows])
        rw = np.array([r[5] + r[6] for r in self.turn_rows])
        later = [r for r in self.turn_rows if r[1] > 0]
        tt2 = np.array([r[4] for r in later]) if later else np.array([np.nan])
        dec = Counter(d[2] for d in self.decisions)
        nd = max(1, len(self.decisions))
        loc = Counter(r[8] for r in later)
        nl = max(1, len(later))
        out = dict(n_turns=len(self.turn_rows), n_sessions=len(self.jct), n_decisions=len(self.decisions),
                   ttft_p50=q(tt, .5), ttft_p95=q(tt, .95), ttft_p99=q(tt, .99), ttft_mean=float(tt.mean()),
                   ttft_ret_p50=q(tt2, .5), ttft_ret_p95=q(tt2, .95), ttft_ret_mean=float(np.nanmean(tt2)),
                   wait_p50=q(rw, .5), wait_p95=q(rw, .95), wait_mean=float(rw.mean()),
                   jct_p50=q(self.jct, .5), jct_p95=q(self.jct, .95), jct_mean=float(np.mean(self.jct)),
                   sim_time_s=self.t_last, hbm_paused_GiBs=self.acc_paused / GiB, hbm_total_GiBs=self.acc_hbm / GiB,
                   ssd_write_GiB=self.bytes["ssd_write"] / GiB, d2h_GiB=self.bytes["d2h"] / GiB,
                   h2d_GiB=self.bytes["h2d"] / GiB, ssd_read_GiB=self.bytes["ssd_read"] / GiB,
                   gpu_recompute_s=self.gpu_recompute_s, gpu_busy_frac=self.gpu.busy_time / max(self.t_last, 1e-9),
                   n_preempt=self.n_preempt)
        for a in ACTIONS:
            out[f"dec_{a}"] = dec[a] / nd
        out["dec_eager_release"] = sum(1 for d in self.decisions if d[7]) / nd
        out["n_evict"] = self.n_evict
        for a in ("gpu", "cpu", "ssd", "partial", "none", "prefetching"):
            out[f"ret_{a}"] = loc[a] / nl
        return out


# ============================================================================ 策略
class Policy:
    name = "base"
    uses_cpu = False

    def __init__(self):
        self.sim = None

    def on_turn_end(self, s):
        pass

    def on_return(self, s):
        self.sim.on_return_gap(s)

    def on_pause(self, s):
        self.sim.log_decision(s, "KEEP")

    def order(self, cands):
        return sorted(cands, key=lambda x: x.last_use)       # LRU

    def dest(self, v):
        return "drop"

    def make_room(self, deficit):
        sim = self.sim
        freed = 0.0
        for v in self.order(sim.victims()):
            if freed >= deficit:
                break
            b = v.hbm
            r = sim.evict(v, self.dest(v))
            if r == "fail":
                r = sim.evict(v, "drop")
            freed += b


class VLLM(Policy):
    """vLLM 預設（APC）：暫停時 KV 留著；放不下時 LRU 淘汰＝丟掉，回來重算。"""
    name = "vllm"


class VLLMCPU(Policy):
    """vLLM APC＋CPU 卸載（OffloadingConnector／LMCache 式）：每輪算完就把新 block 寫一份到 CPU（write-through，
    CPU 層 LRU）；HBM 淘汰時若 CPU 副本完整就立即釋放，否則等寫完；回來從 CPU 載入，副本被 CPU 淘汰則重算。"""
    name = "vllm_cpu"
    uses_cpu = True

    def on_turn_end(self, s):
        self.sim.write_behind(s)

    def on_pause(self, s):
        self.sim.log_decision(s, "CPU" if (s.wb is not None or s.cpu_tok >= s.kv_tok) else "KEEP")

    def dest(self, v):
        if v.cpu_tok >= v.kv_tok or v.wb is not None:
            return "cpu"
        return "drop"


class VLLMTier(VLLMCPU):
    """vLLM 0.28 的 TieringOffloadingSpec（CPU 主層＋檔案系統次層）：寫進 CPU 的 block 一律串寫到所有次層
    （原始碼 v1/kv_offload/tiering/manager.py 檔頭的設計原則 1）；CPU 層 LRU 淘汰（SSD 已有副本）；
    讀次層要先升到 CPU（本模擬以 SSD 讀取時間計，未另加 CPU→GPU 那一段）。"""
    name = "vllm_tier"

    def on_turn_end(self, s):
        self.sim.write_behind(s, cascade=True)

    def on_pause(self, s):
        self.sim.log_decision(s, "SSD" if (s.wb is not None or s.cpu_tok >= s.kv_tok) else "KEEP")

    def dest(self, v):
        if v.cpu_tok >= v.kv_tok or v.wb is not None:
            return "cpu"
        if v.ssd_tok >= v.kv_tok:
            return "ssd"
        return "drop"


class Continuum(Policy):
    """Continuum（arXiv 2511.02230）的 TTL：τ* = argmax_τ F(τ)·(T̄·η + PR(n)) − τ。
    公式照 L3 §H5 對原文 §4.1–4.2 Eq.1–2 的轉述實作（本 agent 沒有讀原文 PDF）。
    F＝已知的間隔分佈；T̄＝被趕出的請求回來時的平均等待（線上估，初值 0）；η＝1（冷啟動）；
    PR(n)：有 CPU 卸載時取載回時間，否則取重算時間。TTL 內釘住（不可被 LRU 趕），過期後回到 vLLM 的 LRU。
    死鎖時從最晚開始的 program 先解釘（原文 §4.3）。"""
    name = "continuum"

    def __init__(self, cpu=False):
        super().__init__()
        self.cpu = cpu
        self.uses_cpu = cpu
        if cpu:
            self.name = "continuum_cpu"

    def on_turn_end(self, s):
        if self.cpu:
            self.sim.write_behind(s)

    def tau(self, s):
        sim = self.sim
        T_bar = sim.T_bar_sum / sim.T_bar_n if sim.T_bar_n else 0.0
        PR = sim.est("cpu", s) if self.cpu else sim.est("drop", s)
        val = sim.cfg.continuum_eta * T_bar + PR
        if getattr(self, "_grid", None) is None:
            g = sim.gd.g
            cand = np.unique(np.concatenate([[0.0], np.quantile(g, np.linspace(0.005, 0.995, 199))]))
            self._grid = (cand, np.searchsorted(g, cand, side="right") / len(g))
        cand, F = self._grid
        obj = F * val - cand
        return float(cand[int(np.argmax(obj))])

    def on_pause(self, s):
        t = self.tau(s)
        s.pinned_until = self.sim.now + t
        if t > 0:
            self.sim.at(s.pinned_until + 1e-9, self._expire, s)
        cp = self.cpu and (s.wb is not None or s.cpu_tok >= s.kv_tok)
        self.sim.log_decision(s, "CPU" if cp else "KEEP")

    def _expire(self, s):
        self.sim.pump()

    def dest(self, v):
        if self.cpu and (v.cpu_tok >= v.kv_tok or v.wb is not None):
            return "cpu"
        return "drop"

    def make_room(self, deficit):
        sim = self.sim
        freed = 0.0
        cands = sim.victims()
        free_c = [v for v in cands if v.pinned_until <= sim.now]
        pinned = sorted([v for v in cands if v.pinned_until > sim.now], key=lambda x: -x.t_start_session)
        for v in self.order(free_c) + pinned:     # 先 LRU 趕沒釘住的；不夠才解釘（最晚開始的 program 先）
            if freed >= deficit:
                break
            v.pinned_until = -1.0
            b = v.hbm
            if sim.evict(v, self.dest(v)) == "fail":
                sim.evict(v, "drop")
            freed += b


class Tokencake(Policy):
    """Tokencake（arXiv 2510.18586）的 Time Scheduler：預測工具時長 T_fc（EWMA：α·t_req + (1−α)·t_hist；
    t_req＝這個 session 上一次的暫停，t_hist＝全域歷史平均；α 原文未給 → 0.5）；
    T_transfer＝T_offload＋T_upload；T_fc ≤ T_transfer 不卸；否則有壓力（有人在等或 HBM 快滿）才卸到 CPU，
    並在預測回來前 T_upload 主動載回。重算不是選項（原文）；放不下時 LRU 卸到 CPU，CPU 也放不下才丟。"""
    name = "tokencake"
    uses_cpu = True

    def on_pause(self, s):
        sim = self.sim
        a = sim.cfg.tokencake_alpha
        hist = sim.hist_gaps_sum / sim.hist_gaps_n if sim.hist_gaps_n else None
        if hist is None:
            sim.log_decision(s, "KEEP")
            return
        treq = s.last_gap if s.last_gap is not None else hist
        tfc = a * treq + (1 - a) * hist
        b = s.kv_tok * sim.cm.bpt
        t_up = sim.est("cpu", s)
        t_tr = sim.cm.d2h(max(0, s.kv_tok - s.cpu_tok) * sim.cm.bpt) + t_up
        if tfc <= t_tr or not sim.pressure():
            sim.log_decision(s, "KEEP")
            return
        if not sim.cpu_room_possible(b, s):
            sim.log_decision(s, "KEEP")
            return
        sim.log_decision(s, "CPU", eager=True)
        if sim.evict(s, "cpu", demand=False) == "fail":
            sim.evict(s, "drop", demand=False)
        sim.at(sim.now + max(0.0, tfc - t_up), sim.prefetch, s, s.k)

    def dest(self, v):
        if self.sim.cpu_room_possible((v.kv_tok - v.cpu_tok) * self.sim.cm.bpt, v):
            return "cpu"
        return "drop"


class InferCept(Policy):
    """InferCept（ICML'24）：三種做法的 GPU 記憶體浪費（記憶體×時間），取最小。照 L3 §H1 對原文 Eq.1–3 的轉述：
      Discard＝T_rec(n)·(n + batch_ctx)；Preserve＝T_INT·n；Swap＝2·T_swap(n)·batch_ctx（M 約掉）。
    T_INT 用「已經暫停多久」估（原文 §4.4）→ 保留到 e*·n = min(Discard, Swap)，e* 時切換成 Swap（CPU）或 Discard。
    batch_ctx＝暫停開始當下正在服務的 session 的 token 數。swap budget 與分層 pipeline 不模擬。"""
    name = "infercept"
    uses_cpu = True

    def plan(self, s):
        sim = self.sim
        n = max(s.kv_tok, 1)
        batch = sim.active_tok - (s.act_tok if s.state in ACTIVE else 0)
        W_d = sim.est("drop", s) * (n + batch)
        W_s = 2 * sim.est("cpu", s) * batch
        cpu_ok = sim.cpu_room_possible((s.kv_tok - s.cpu_tok) * sim.cm.bpt, s)
        choice = "cpu" if (W_s < W_d and cpu_ok) else "drop"
        return min(W_d, W_s if cpu_ok else W_d) / n, choice

    def on_pause(self, s):
        t_sw, choice = self.plan(s)
        s.infer_choice = choice
        if t_sw <= 0:
            self.sim.log_decision(s, "CPU" if choice == "cpu" else "DROP", eager=True)
            if self.sim.evict(s, choice, demand=False) == "fail":
                self.sim.evict(s, "drop", demand=False)
            return
        self.sim.log_decision(s, "KEEP")
        self.sim.at(self.sim.now + t_sw, self._switch, s, s.k)

    def _switch(self, s, k):
        sim = self.sim
        if s.k != k or s.state not in PAUSED or not s.on_gpu or s.move is not None or s.pf is not None:
            return
        _, choice = self.plan(s)
        if sim.evict(s, choice, demand=False) == "fail":
            sim.evict(s, "drop", demand=False)
        sim.pump()

    def dest(self, v):
        return self.plan(v)[1]


class Kappa(Policy):
    """κ 感知四選一（本研究的候選；v4，掃描前凍結。設計演進見 docs/stage3_20261019/i3_pause_sim.md §2.4）。

    每個暫停中的 session v（已暫停 e 秒、KV b bytes）：
      C_X(v)＝用實測成本算的還原時間，X∈{CPU（要有空間）、SSD、DROP（重算；有部分副本時＝載入副本＋重算其餘）}
      X*＝argmin_X C_X（κ 在這裡起作用：哪一層最便宜取決於重算／搬運的實測比）
      R̄(v)＝E[G−e | G>e]（已知的間隔分佈）；P(回來)＝由間隔分佈與「最後一輪」比例 p_end 算的條件機率
      W̄＝被趕出去的 session 回來時平均等 HBM 的時間（線上累計）
      score(v)＝P(回來)·(C_X* ＋ W̄) ÷ (b·R̄)＝「趕它出去，每釋放 1 byte·秒 HBM 的預期 TTFT 代價」
    * 暫停開始（四選一）：
        - 若 X*＝CPU、而且預期剩餘暫停 R̄ > max(寫副本的時間, T̄_ev) → 背景寫 CPU 副本（記 CPU；HBM 先當快取留著，
          有人要空間時瞬間釋放）。T̄_ev＝這一格裡暫停中的 session 平均撐多久才被趕（線上；沒有趕過＝∞＝不寫）：
          預期比「通常被趕的時間」更早回來的，副本多半白寫。
        - 否則 KEEP（只留在 HBM）。
      不在暫停開始就釋放 HBM：有副本時「等有人要再放」弱支配「現在就放」（現在放＝回來時白付還原與重新排隊）。
    * 有人要空間：照 score 由小到大趕；目的地＝argmin{搬移時間＋C_X（X∈CPU/SSD）, C_DROP}（搬移時間由等待者付）。
    * 不預取（線上不知道確切回來時間）。"""
    name = "kappa"
    uses_cpu = True
    know_end = False
    exact = False
    prefetch = False

    def tiers(self, s):
        sim = self.sim
        cm = sim.cm
        out = {}
        need_cpu = (s.kv_tok - s.cpu_tok) * cm.bpt
        if s.cpu_tok >= s.kv_tok or s.wb is not None or sim.cpu_room_possible(need_cpu, s):
            out["cpu"] = (sim.est("cpu", s), 0.0 if s.cpu_tok >= s.kv_tok else cm.d2h(need_cpu))
        if not math.isnan(cm.ssd_r_Bps):
            need_ssd = (s.kv_tok - s.ssd_tok) * cm.bpt
            out["ssd"] = (sim.est("ssd", s), 0.0 if s.ssd_tok >= s.kv_tok else cm.ssd_write(need_ssd))
        out["drop"] = (sim.est("drop", s), 0.0)
        return out

    def best(self, s):
        t = self.tiers(s)
        x = min(t, key=lambda k: t[k][0])
        return x, t[x][0], t[x][1], t

    def p_ret(self, e):
        S = float(self.sim.gd.sf(e)) if e > 0 else 1.0
        pe = self.sim.p_end
        den = (1 - pe) * S + pe
        return (1 - pe) * S / den if den > 0 else 0.0

    def rem(self, s):
        """(剩餘暫停的估計, P(回來))。"""
        if s.dead and self.know_end:
            return INF, 0.0
        if self.exact:
            return max(s.ret - self.sim.now, 0.0), 1.0
        e = self.sim.now - s.pause_start
        R = self.sim.gd.cond_rem(e, self.sim.cfg.horizon_H)
        return R, (1.0 if self.know_end else self.p_ret(e))

    def score(self, s):
        x, c, _, _ = self.best(s)
        R, pr = self.rem(s)
        return pr * (c + self.sim.w_bar()) / (max(s.hbm, 1.0) * max(R, 1e-9))

    def order(self, cands):
        return sorted(cands, key=self.score)

    def dest(self, v):
        x, c, mv, t = self.best(v)
        opts = {k: (val[1] + val[0]) for k, val in t.items()}   # 搬移時間（被等的人付）＋之後的還原
        opts["drop"] = t["drop"][0]
        return min(opts, key=opts.get)

    @staticmethod
    def _end(sim, s):
        sim.log_decision(s, "DROP")
        sim._drop_gpu(s)
        for j in s.jobs:
            j.cancelled = True
        s.cpu_tok = 0
        s.ssd_tok = 0
        sim._recalc_cpu(s)

    def on_pause(self, s):
        sim = self.sim
        if s.dead and self.know_end:
            self._end(sim, s)
            return
        x, c, mv, t = self.best(s)
        R, pr = self.rem(s)
        logged = False
        if x == "cpu" and R > max(mv, sim.t_ev_bar()):
            if s.cpu_tok >= s.kv_tok or sim.write_behind(s):
                sim.log_decision(s, "CPU")
                logged = True
        if not logged:
            sim.log_decision(s, "KEEP")
        if self.prefetch and not s.dead:
            sim.at(max(sim.now, s.ret - c), sim.prefetch, s, s.k)


class KappaTier(Kappa):
    """【事後加入的第二候選，看過 vllm_tier 的結果之後才加；報告裡另標】κ 感知＋κ 把關的串寫：
    同 κ 感知（v4）的暫停決定與趕出順序；寫 CPU 副本時，只有「從 SSD 級還原 < 重算」（用實測成本、依這個 session 的長度）
    才串寫到 SSD 級。本地 SSD（8 GiB/s）時等於 vllm_tier 的串寫；NFS（0.38 GiB/s）時不串寫。"""
    name = "kappa_tier"

    def on_pause(self, s):
        sim = self.sim
        x, c, mv, t = self.best(s)
        R, pr = self.rem(s)
        gate = ("ssd" in t) and t["ssd"][0] < sim.est("drop", s)
        if x == "cpu" and R > max(mv, sim.t_ev_bar()):
            if s.cpu_tok >= s.kv_tok or sim.write_behind(s, cascade=gate):
                sim.log_decision(s, "SSD" if gate else "CPU")
                return
        sim.log_decision(s, "KEEP")


class KappaEnd(Kappa):
    """歸因用（不是候選）：κ 感知＋知道 session 已經結束（最後一輪立刻丟掉、清副本）。"""
    name = "kappa_end"
    know_end = True


class OracleNoPF(Kappa):
    """歸因用：知道每個 session 確切回來時間（含不會再回來），結構同 κ 感知，不預取。"""
    name = "oracle_nopf"
    know_end = True
    exact = True


class Oracle(OracleNoPF):
    """完美資訊＋在確切回來前 C_X* 秒預取（HBM 有空位時）。
    注意：共用 GPU／鏈路時這仍是貪婪規則，不是下界（自測以靜態版與 C=1 動態版檢查最佳性）。"""
    name = "oracle"
    prefetch = True


POLICIES = {"vllm": lambda: VLLM(), "vllm_cpu": lambda: VLLMCPU(), "vllm_tier": lambda: VLLMTier(),
            "continuum": lambda: Continuum(False),
            "continuum_cpu": lambda: Continuum(True), "tokencake": lambda: Tokencake(),
            "infercept": lambda: InferCept(), "kappa": lambda: Kappa(), "kappa_tier": lambda: KappaTier(),
            "kappa_end": lambda: KappaEnd(),
            "oracle_nopf": lambda: OracleNoPF(), "oracle": lambda: Oracle()}
BASELINES = ["vllm", "vllm_cpu", "vllm_tier", "continuum", "continuum_cpu", "tokencake", "infercept"]


# ============================================================================ 工作負載與容量
def make_workload(pool, gd: GapDist | None, n_sess, seed, C):
    """抽 n_sess 個 session（不重複）；gd=None 時用 trace 的真實間隔，否則把間隔換成從 gd 抽的。
    slot 的開場與兩個 session 之間的空檔也從同一分佈抽（gd=None 時從 trace 的間隔抽）。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pool))[:n_sess] if n_sess <= len(pool) else rng.integers(0, len(pool), n_sess)
    sess = []
    all_gaps = np.array([t["gap"] for s in pool for t in s[:-1]], float)
    for i in idx:
        turns = [dict(t) for t in pool[int(i)]]
        if gd is not None:
            g = gd.sample(rng, len(turns) - 1)
            for t, x in zip(turns[:-1], g):
                t["gap"] = float(max(0.0, x))
        sess.append(turns)
    if gd is None:
        sg = rng.choice(all_gaps, size=C + n_sess)
    else:
        sg = gd.sample(rng, C + n_sess)
    return dict(sessions=sess, slot_gaps=[float(max(0.0, x)) for x in sg])


CGROUP_MEM = "/sys/fs/cgroup/memory.max"


def pod_mem_bytes():
    """容器的記憶體上限（cgroup memory.max；experiment_matrix §0 記為 435 GiB）。設定值，不是量測。"""
    try:
        return float(open(CGROUP_MEM).read().strip())
    except (OSError, ValueError):
        return float("nan")


def capacity(cm: Costs, wl, C, level, cpu_mult=None):
    """HBM 與 CPU 容量（byte）。
    W_ref＝C × 本次抽到的 session 最大 context 的平均 × bytes/token；
    HBM：緊＝0.25 W_ref、中＝0.5 W_ref、鬆＝1.0 W_ref、real＝vLLM 在 MI300X 實測的 KV token 數；
         下限＝本次最大的單輪需求（否則單一 session 放不下）。
    CPU：預設＝HBM × (容器記憶體上限 ÷ 實測 HBM KV 容量)，也就是照真實機器的 CPU:HBM 比例縮放；
         real 時＝容器記憶體上限；cpu_mult 給定時＝cpu_mult × HBM（敏感度）。"""
    mx = [max(t["inp"] + t["out"] for t in s) for s in wl["sessions"]]
    W = C * float(np.mean(mx)) * cm.bpt
    floor = max(mx) * cm.bpt * 1.0001
    ratio = {"tight": 0.25, "mid": 0.5, "loose": 1.0}
    pod = pod_mem_bytes()
    cm.require("hbm_real_tok")
    real = cm.hbm_real_tok * cm.bpt
    if level == "real":
        h = real
    elif level == "inf":
        h = 1e18
    else:
        h = ratio[level] * W
    h = max(h, floor)
    if level == "inf":
        c = 1e18
    elif cpu_mult is not None:
        c = cpu_mult * h
    elif level == "real":
        c = pod
    else:
        c = h * pod / real
    return h, c, W, floor


def run_one(cm, pool, gd_work, gd_known, p_end, C, cap_level, policy, seed, busy="endo", restore="load",
            n_sess=None, cpu_mult=None):
    n_sess = n_sess or max(200, 12 * C)
    wl = make_workload(pool, gd_work, n_sess, seed, C)
    hbm, cpu, W, floor = capacity(cm, wl, C, cap_level, cpu_mult)
    cfg = Cfg(C=C, hbm_bytes=hbm, cpu_bytes=cpu, busy=busy, restore=restore)
    pol = POLICIES[policy]()
    if cm.path != "harness":
        raise NotMeasured(f"{cm.platform}/{cm.model}: 動態模擬需要的常數 NOT_MEASURED：{cm.missing}")
    sim = Sim(cm, cfg, wl, pol, gd_known, p_end).run()
    out = sim.summary()
    out.update(hbm_GiB=hbm / GiB, cpu_GiB=cpu / GiB, W_ref_GiB=W / GiB, floor_GiB=floor / GiB,
               hbm_over_W=hbm / W)
    return out, sim


# ============================================================================ 決策分帶（看翻轉）
CTX_BANDS = [(0, 8192, "<8K"), (8192, 32768, "8–32K"), (32768, 65536, "32–64K"), (65536, 10 ** 9, "≥64K")]
GAP_BANDS = [(0, 1, "<1s"), (1, 10, "1–10s"), (10, 60, "10–60s"), (60, 300, "1–5m"), (300, INF, "≥5m")]


def band(v, bands):
    for lo, hi, name in bands:
        if lo <= v < hi:
            return name
    return bands[-1][2]


def decision_bands(sim):
    """三種紀錄，依 (context 帶, 真實暫停帶) 分：pause＝暫停開始的決定；evict＝有人要空間時的目的地；
    return＝回來時 KV 在哪。gap 帶用真實值（只供分析，策略本身看不到）。"""
    c = Counter()
    for d in sim.decisions:
        sid, k, a, kv, gap, dead, pres, eager = d
        gb = "end" if dead else band(gap, GAP_BANDS)
        c[("pause", band(kv, CTX_BANDS), gb, a + ("!" if eager and a != "DROP" else ""))] += 1
    for kv, dest, rem, dead in sim.evictions:
        gb = "end" if dead else band(rem, GAP_BANDS)
        c[("evict", band(kv, CTX_BANDS), gb, dest.upper())] += 1
    for r in sim.turn_rows:
        if r[1] > 0:
            c[("return", band(r[2], CTX_BANDS), band(r[9], GAP_BANDS), r[8])] += 1
    return c


# ============================================================================ 掃描
GAPS_MAIN = ["mooncake", "ln5", "ln30", "ln120"]


def gap_dists(name, pool_gaps, sigma):
    if name == "mooncake":
        return None, GapDist.empirical(pool_gaps, "mooncake-toolagent")
    med = float(name[2:])
    g = GapDist.lognormal(med, sigma)
    return g, g


_POOL_CACHE = {}


def _pool(name, max_tok):
    key = (name, max_tok)
    if key not in _POOL_CACHE:
        _POOL_CACHE[key] = mooncake_pool(name, max_tok)
    return _POOL_CACHE[key]


def _job(args):
    (platform, model, ssd_dev, C, cap, gapname, seed, busy, restore, sigma, policies, tag, cpu_mult) = args
    cm = load_costs(platform, model, ssd_dev)
    pool, pgaps, st = _pool("toolagent", cm.max_tok)
    gd_work, gd_known = gap_dists(gapname, pgaps, sigma)
    rows, dec_rows = [], []
    for pol in policies:
        t0 = time.time()
        try:
            out, sim = run_one(cm, pool, gd_work, gd_known, st["p_end"], C, cap, pol, seed, busy, restore,
                               cpu_mult=cpu_mult)
        except NotMeasured as e:
            rows.append(dict(platform=platform, model=model, ssd_dev=ssd_dev, C=C, cap=cap, gap=gapname, seed=seed,
                             busy=busy, restore=restore, sigma=sigma, policy=pol, tag=tag, cpu_mult=cpu_mult,
                             status=f"NOT_MEASURED:{e}"))
            continue
        out.update(platform=platform, model=model, ssd_dev=ssd_dev, C=C, cap=cap, gap=gapname, seed=seed, busy=busy,
                   restore=restore, sigma=sigma, policy=pol, tag=tag, cpu_mult=cpu_mult, status="ok",
                   wall_s=round(time.time() - t0, 3))
        rows.append(out)
        if pol in ("kappa", "kappa_tier", "oracle", "vllm_cpu", "vllm_tier", "tokencake", "infercept", "vllm"):
            for (kind, cb, gb, a), n in decision_bands(sim).items():
                dec_rows.append(dict(platform=platform, model=model, ssd_dev=ssd_dev, C=C, cap=cap, gap=gapname,
                                     seed=seed, busy=busy, restore=restore, sigma=sigma, policy=pol, tag=tag,
                                     cpu_mult=cpu_mult, kind=kind, ctx_band=cb, gap_band=gb, action=a, n=n))
    return rows, dec_rows


def build_grid(a):
    """主掃描：平台 mi300x（3090 的動態模擬 NOT_MEASURED）× 模型 × C × 容量 × 間隔 × seed。
    敏感度（C∈{8,32}、容量∈{緊,中}）：GPU 外加背景忙（exo）、GPU 忙放大關掉（off）、SSD 級換成 NFS（真實裝置）、
    Cake 還原、lognormal σ=2、CPU 容量縮成 0.25×HBM（逼出 SSD／重算的選擇），以及後者再配 NFS。"""
    grid = []
    pols = list(POLICIES)
    for model in a.models:
        for C in a.C:
            for cap in a.caps:
                for gap in a.gaps:
                    for seed in range(a.seeds):
                        grid.append(("mi300x", model, "local", C, cap, gap, seed, "endo", "load", 1.0, pols, "main", None))
    if a.sens:
        for model in a.models:
            for C in (8, 32):
                for cap in ("tight", "mid"):
                    for gap in a.gaps:
                        for seed in range(a.seeds):
                            g = lambda dev, busy, rs, sg, tag, cmul=None: grid.append(
                                ("mi300x", model, dev, C, cap, gap, seed, busy, rs, sg, pols, tag, cmul))
                            g("local", "exo", "load", 1.0, "busy_exo")
                            g("local", "off", "load", 1.0, "busy_off")
                            g("nfs", "endo", "load", 1.0, "ssd_nfs")
                            g("local", "endo", "cake", 1.0, "cake")
                            g("local", "endo", "load", 1.0, "cpu_small", 0.25)
                            g("nfs", "endo", "load", 1.0, "cpu_small_nfs", 0.25)
                            if gap != "mooncake":
                                g("local", "endo", "load", 2.0, "sigma2")
    return grid


def cmd_scan(a):
    import multiprocessing as mp
    os.makedirs(a.out, exist_ok=True)
    grid = build_grid(a)
    print(f"[scan] run_id={RUN_ID} 格數={len(grid)} 行程={a.procs}", flush=True)
    rows, decs = [], []
    t0 = time.time()
    with mp.get_context("fork").Pool(a.procs) as P:
        for i, (r, d) in enumerate(P.imap_unordered(_job, grid, chunksize=1)):
            rows += r
            decs += d
            if (i + 1) % 20 == 0 or i + 1 == len(grid):
                print(f"[scan] {i + 1}/{len(grid)} {time.time() - t0:.0f}s", flush=True)
    ts = now_ts()
    for r in rows + decs:
        r["run_id"] = RUN_ID
        r["ts"] = ts
    raw = a.raw or a.out
    os.makedirs(raw, exist_ok=True)
    write_csv(os.path.join(raw, "scan.csv"), rows)
    write_csv(os.path.join(raw, "decisions.csv"), decs)
    print(f"[scan] 寫出 {len(rows)} 列 scan.csv、{len(decs)} 列 decisions.csv 到 {raw}；{time.time() - t0:.0f}s")


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    front = ["run_id", "ts"]
    keys = front + [k for k in keys if k not in front]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


# ============================================================================ 分析
METRICS = ["ttft_ret_p95", "ttft_p95", "jct_p95", "ttft_ret_mean", "ttft_mean"]
PERFECT = ["kappa_end", "oracle_nopf", "oracle"]
CANDS = ["kappa", "kappa_tier"]          # kappa＝事前凍結的候選；kappa_tier＝事後加入（看過 vllm_tier 才加）
PAIRS = [("main", "ssd_nfs", "下層：本地 SSD → NFS"), ("cpu_small", "cpu_small_nfs", "CPU 小；下層：本地 SSD → NFS"),
         ("busy_off", "busy_exo", "GPU：閒 → 外加忙（duty 0.7）")]
KEYF = ("tag", "platform", "model", "ssd_dev", "busy", "restore", "sigma", "cpu_mult", "C", "cap", "gap")


def cmd_analyze(a):
    raw = a.raw or a.out
    allrows = read_csv(os.path.join(raw, "scan.csv"))
    rows = [r for r in allrows if r["status"] == "ok"]
    not_ok = [r for r in allrows if r["status"] != "ok"]
    decs = read_csv(os.path.join(raw, "decisions.csv"))
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cells[tuple(r[k] for k in KEYF)][r["policy"]].append(r)
    out_cells = []
    for key, pols in sorted(cells.items()):
        d = dict(zip(KEYF, key))
        seeds = sorted({r["seed"] for rs in pols.values() for r in rs})
        d["n_seeds"] = len(seeds)
        d["hbm_over_W"] = float(np.mean([float(r["hbm_over_W"]) for r in pols["vllm"]]))
        d["hbm_GiB"] = float(np.mean([float(r["hbm_GiB"]) for r in pols["vllm"]]))
        for m in METRICS:
            avg = {p: float(np.mean([float(r[m]) for r in rs])) for p, rs in pols.items()}
            base = {p: v for p, v in avg.items() if p in BASELINES}
            bp = min(base, key=base.get)
            d[f"{m}_best_baseline"] = bp
            d[f"{m}_best_baseline_val"] = base[bp]
            for p in BASELINES:
                d[f"{m}__{p}"] = avg[p]
            pf = min((avg[p] for p in PERFECT if p in avg), default=float("nan"))
            d[f"{m}_best_perfect"] = pf
            d[f"{m}_kappa_end"] = avg.get("kappa_end", float("nan"))
            d[f"{m}_gain_kappa_end"] = 1 - avg["kappa_end"] / base[bp] if base[bp] > 0 else 0.0
            d[f"{m}_headroom_perfect"] = 1 - pf / base[bp] if base[bp] > 0 else 0.0
            for cand in CANDS:
                d[f"{m}_{cand}"] = avg[cand]
                d[f"{m}_gain_{cand}"] = 1 - avg[cand] / base[bp] if base[bp] > 0 else 0.0
                ps = []
                for sd in seeds:
                    bv = min(float(r[m]) for p, rs in pols.items() if p in BASELINES for r in rs if r["seed"] == sd)
                    kv = [float(r[m]) for r in pols[cand] if r["seed"] == sd]
                    if kv and bv > 0:
                        ps.append(1 - kv[0] / bv)
                d[f"{m}_gain_{cand}_seedmin"] = min(ps) if ps else float("nan")
                d[f"{m}_gain_{cand}_seedmax"] = max(ps) if ps else float("nan")
        for p in ("kappa", "kappa_tier", "vllm_cpu", "vllm_tier", "tokencake", "infercept", "oracle"):
            for x in ACTIONS:
                d[f"{p}_dec_{x}"] = float(np.mean([float(r[f"dec_{x}"]) for r in pols[p]]))
            for x in ("gpu", "cpu", "ssd", "none", "partial"):
                d[f"{p}_ret_{x}"] = float(np.mean([float(r[f"ret_{x}"]) for r in pols[p]]))
        for p in ("kappa", "kappa_tier", "vllm_cpu", "vllm_tier", "tokencake", "infercept"):
            d[f"{p}_ssd_write_GiB"] = float(np.mean([float(r["ssd_write_GiB"]) for r in pols[p]]))
            d[f"{p}_hbm_paused_GiBs"] = float(np.mean([float(r["hbm_paused_GiBs"]) for r in pols[p]]))
        out_cells.append(d)
    ts = now_ts()
    for d in out_cells:
        d["run_id"] = RUN_ID
        d["ts"] = ts
    write_csv(os.path.join(a.out, "cells.csv"), out_cells)

    def stat(cs, m, key="gain_kappa"):
        g = [d[f"{m}_{key}"] for d in cs]
        if not g:
            return {}
        return dict(n_cells=len(g), max=max(g), median=float(np.median(g)), min=min(g),
                    n_ge_10pct=sum(1 for x in g if x >= 0.10), n_ge_5pct=sum(1 for x in g if x >= 0.05),
                    n_le_minus_5pct=sum(1 for x in g if x <= -0.05))
    tags = sorted({d["tag"] for d in out_cells})
    summ = dict(run_id=RUN_ID, ts=ts, scan_csv=os.path.join(raw, "scan.csv"), n_runs_ok=len(rows),
                n_runs_not_ok=len(not_ok), not_ok_status=dict(Counter(r["status"][:80] for r in not_ok)),
                metrics_note="gain＝1 − κ／最好的現成規則（>0＝κ 較好）；headroom_perfect＝完美資訊三個變體的最小值對最好的現成規則",
                gain_kappa={t: {m: stat([d for d in out_cells if d["tag"] == t], m) for m in METRICS} for t in tags},
                gain_kappa_tier={t: {m: stat([d for d in out_cells if d["tag"] == t], m, "gain_kappa_tier")
                                     for m in METRICS} for t in tags},
                gain_kappa_end={t: {m: stat([d for d in out_cells if d["tag"] == t], m, "gain_kappa_end")
                                    for m in METRICS} for t in tags},
                headroom_perfect={t: {m: stat([d for d in out_cells if d["tag"] == t], m, "headroom_perfect")
                                      for m in METRICS} for t in tags},
                best_baseline_counts={t: dict(Counter(d["ttft_ret_p95_best_baseline"] for d in out_cells
                                                      if d["tag"] == t)) for t in tags})
    # ---- 跨硬體的「最佳現成規則」翻轉與遺憾：A 設定下最好的現成規則搬到 B 設定，比 B 的最好規則差多少；κ 的遺憾
    byk = {tuple(d[k] for k in KEYF): d for d in out_cells}
    m = "ttft_ret_p95"
    regret = []
    for ta, tb, what in PAIRS:
        for d in out_cells:
            if d["tag"] != ta:
                continue
            kb = list(tuple(d[k] for k in KEYF))
            kb[0] = tb
            if ta == "busy_off":
                kb[KEYF.index("busy")] = "exo"
            if tb.endswith("nfs"):
                kb[KEYF.index("ssd_dev")] = "nfs"
            db = byk.get(tuple(kb))
            if db is None:
                continue
            for X, Y, dir_ in ((d, db, "A→B"), (db, d, "B→A")):
                best_x = X[f"{m}_best_baseline"]
                regret.append(dict(what=what, dir=dir_, model=d["model"], C=d["C"], cap=d["cap"], gap=d["gap"],
                                   best_src=best_x, best_dst=Y[f"{m}_best_baseline"],
                                   flip=best_x != Y[f"{m}_best_baseline"],
                                   regret_transplant=Y[f"{m}__{best_x}"] / Y[f"{m}_best_baseline_val"] - 1,
                                   regret_kappa=Y[f"{m}_kappa"] / Y[f"{m}_best_baseline_val"] - 1,
                                   regret_kappa_tier=Y[f"{m}_kappa_tier"] / Y[f"{m}_best_baseline_val"] - 1))
    for r in regret:
        r["run_id"] = RUN_ID
        r["ts"] = ts
    write_csv(os.path.join(a.out, "regret.csv"), regret)

    def rstat(rs, k):
        v = [r[k] for r in rs]
        return dict(n=len(v), median=float(np.median(v)) if v else None, max=max(v) if v else None,
                    n_ge_10pct=sum(1 for x in v if x >= 0.10))
    summ["cross_hw"] = {f"{w} {dr}": dict(n_flip=sum(1 for r in regret if r["what"] == w and r["dir"] == dr and r["flip"]),
                                          transplant=rstat([r for r in regret if r["what"] == w and r["dir"] == dr], "regret_transplant"),
                                          kappa=rstat([r for r in regret if r["what"] == w and r["dir"] == dr], "regret_kappa"),
                                          kappa_tier=rstat([r for r in regret if r["what"] == w and r["dir"] == dr], "regret_kappa_tier"))
                        for w in {r["what"] for r in regret} for dr in ("A→B", "B→A")}
    # ---- 決策翻轉：同一個 (C, 容量, 間隔, ctx 帶, gap 帶) 下，κ 感知（與 oracle）多數的「趕出目的地」／「回來時位置」
    agg = defaultdict(Counter)
    for r in decs:
        if r["policy"] not in ("kappa", "kappa_tier", "oracle") or r["kind"] not in ("evict", "return"):
            continue
        k = (r["policy"], r["kind"], r["tag"], r["model"], r["ssd_dev"], r["busy"], r["cpu_mult"], r["C"], r["cap"],
             r["gap"], r["ctx_band"], r["gap_band"])
        agg[k][r["action"]] += int(r["n"])
    modal = {k: max(v, key=v.get) for k, v in agg.items() if sum(v.values()) >= 5}
    flips, compared = [], Counter()
    for k, a1 in modal.items():
        pol, kind, tag, model, dev, busy, cmul, C, cap, gap, cb, gb = k
        cmp = []
        if tag == "main" and model == "llama":
            cmp.append(("model llama↔qwen", (pol, kind, tag, "qwen", dev, busy, cmul, C, cap, gap, cb, gb)))
        if tag == "busy_exo":
            cmp.append(("GPU 忙 exo↔off", (pol, kind, "busy_off", model, dev, "off", cmul, C, cap, gap, cb, gb)))
        if tag == "ssd_nfs":
            cmp.append(("下層 NFS↔本地 SSD", (pol, kind, "main", model, "local", "endo", "", C, cap, gap, cb, gb)))
        if tag == "cpu_small_nfs":
            cmp.append(("CPU 小＋下層 NFS↔本地 SSD", (pol, kind, "cpu_small", model, "local", "endo", cmul, C, cap, gap, cb, gb)))
        for what, k2 in cmp:
            if k2 in modal:
                compared[(what, pol, kind)] += 1
                if modal[k2] != a1:
                    flips.append(dict(what=what, policy=pol, kind=kind, model=model, C=C, cap=cap, gap=gap, ctx_band=cb,
                                      gap_band=gb, a=a1, b=modal[k2], n_a=sum(agg[k].values()), n_b=sum(agg[k2].values())))
    summ["flips"] = dict(n=len(flips), compared={" | ".join(k): v for k, v in compared.items()},
                         by={" | ".join(k): v for k, v in Counter((f["what"], f["policy"], f["kind"]) for f in flips).items()})
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1, ensure_ascii=False, default=str)
    for f in flips:
        f["run_id"] = RUN_ID
        f["ts"] = ts
    write_csv(os.path.join(a.out, "flips.csv"), flips)
    print(json.dumps({t: (summ["gain_kappa"][t]["ttft_ret_p95"], summ["gain_kappa_tier"][t]["ttft_ret_p95"]) for t in tags},
                     indent=1, ensure_ascii=False))
    print(json.dumps(summ["cross_hw"], indent=1, ensure_ascii=False))
    print("flips:", summ["flips"]["by"])


# ============================================================================ costs／trace／decision-map
def cmd_costs(a):
    rows, miss = [], []
    for platform in ("mi300x", "mi300x_vllm", "3090"):
        for model in ("llama", "qwen"):
            for dev in (("local", "nfs") if platform == "mi300x" else ("-",)):
                try:
                    cm = load_costs(platform, model, dev if dev != "-" else "local")
                except NotMeasured as e:
                    miss.append(dict(platform=platform, model=model, missing=str(e)))
                    continue
                if dev == "nfs":
                    rows += [r for r in cm.sources if "ssd" in r["const"] or "k_read" in r["const"]]
                else:
                    rows += cm.sources
                miss.append(dict(platform=platform, model=model, ssd_dev=dev, missing=";".join(cm.missing),
                                 notes=" | ".join(cm.notes), max_tok=cm.max_tok))
    ts = now_ts()
    for r in rows + miss:
        r["run_id"] = RUN_ID
        r["ts"] = ts
    os.makedirs(a.out, exist_ok=True)
    write_csv(os.path.join(a.out, "costs_sources.csv"), rows)
    write_csv(os.path.join(a.out, "costs_missing.csv"), miss)
    for m in miss:
        print(m)
    print(f"[costs] {len(rows)} 個常數 → {rel(os.path.join(a.out, 'costs_sources.csv'))}")


def cmd_trace(a):
    out = {}
    for name in ("toolagent", "conversation"):
        pool, gaps, st = mooncake_pool(name, 129024)
        out[name] = st
    out["run_id"] = RUN_ID
    out["ts"] = now_ts()
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "trace_sessions.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


def cmd_decision_map(a):
    """靜態 κ 表（只看還原端、GPU 閒）：每個 token 的還原成本（µs/token）與 CPU/SSD/重算的排名。"""
    rows = []
    for model in ("llama", "qwen"):
        for dev in ("local", "nfs"):
            cm = load_costs("mi300x", model, dev)
            for L in (8192, 16384, 32768, 65536, 129024):
                c = cm.restore_est("cpu", L, L, False, "load") / L * 1e6
                s = cm.restore_est("ssd", L, L, False, "load") / L * 1e6
                d = cm.R(L) / L * 1e6
                rank = ">".join(sorted({"CPU": c, "SSD": s, "DROP": d}, key={"CPU": c, "SSD": s, "DROP": d}.get))
                rows.append(dict(platform="mi300x", path="harness", model=model, ssd_dev=dev, ctx=L,
                                 cpu_us_tok=round(c, 3), ssd_us_tok=round(s, 3), drop_us_tok=round(d, 3),
                                 kappa_cpu=round(d / c, 2), kappa_ssd=round(d / s, 3), order_cheapest_first=rank,
                                 note="GPU 閒；harness 重算比預設 vLLM 慢 1.7–2 倍（toolbox §4）"))
        cmv = load_costs("mi300x_vllm", model)
        st = cmv.static
        if {"cpu", "ssd", "drop"} <= set(st):
            c, s, d = st["cpu"], st["ssd"], st["drop"]
            rank = ">".join(sorted({"CPU": c, "SSD": s, "DROP": d}, key={"CPU": c, "SSD": s, "DROP": d}.get))
            rows.append(dict(platform="mi300x", path="vllm", model=model, ssd_dev="local(vLLM fs tier)", ctx=st["ctx"],
                             cpu_us_tok=c, ssd_us_tok=s, drop_us_tok=d, kappa_cpu=round(d / c, 2),
                             kappa_ssd=round(d / s, 3), order_cheapest_first=rank, note="vLLM OffloadingConnector 路徑；整段平均"))
    cm3 = load_costs("3090", "llama")
    st = cm3.static
    for dev in ("sata", "nvme"):
        c, s, d = st[f"cpu_{dev}"], st[f"ssd_{dev}"], st[f"drop_{dev}"]
        rank = ">".join(sorted({"CPU": c, "SSD": s, "DROP": d}, key={"CPU": c, "SSD": s, "DROP": d}.get))
        rows.append(dict(platform="3090", path="vllm", model="llama", ssd_dev=dev, ctx=st["ctx"], cpu_us_tok=round(c, 3),
                         ssd_us_tok=round(s, 3), drop_us_tok=round(d, 3), kappa_cpu=round(d / c, 2),
                         kappa_ssd=round(d / s, 3), order_cheapest_first=rank,
                         note="vLLM 路徑；host_contention=HEAVY；動態模擬 NOT_MEASURED（D2H、GPU 忙放大）"))
    ts = now_ts()
    for r in rows:
        r["run_id"] = RUN_ID
        r["ts"] = ts
    os.makedirs(a.out, exist_ok=True)
    write_csv(os.path.join(a.out, "decision_map_static.csv"), rows)
    for r in rows:
        print({k: r[k] for k in ("platform", "path", "model", "ssd_dev", "ctx", "cpu_us_tok", "ssd_us_tok",
                                 "drop_us_tok", "order_cheapest_first")})


def cmd_one(a):
    cm = load_costs(a.platform, a.model, a.ssd_dev)
    pool, pgaps, st = mooncake_pool("toolagent", cm.max_tok)
    gw, gk = gap_dists(a.gap, pgaps, a.sigma)
    for pol in a.policies:
        t0 = time.time()
        out, sim = run_one(cm, pool, gw, gk, st["p_end"], a.C, a.cap, pol, a.seed, a.busy, a.restore,
                           cpu_mult=a.cpu_mult)
        print(f"{pol:14s} ttft p50 {out['ttft_p50']:.3f} p95 {out['ttft_p95']:.3f} ret_p95 {out['ttft_ret_p95']:.3f} "
              f"jct p95 {out['jct_p95']:.1f} dec K/C/S/D {out['dec_KEEP']:.2f}/{out['dec_CPU']:.2f}/"
              f"{out['dec_SSD']:.2f}/{out['dec_DROP']:.2f} ret gpu/cpu/ssd/none {out['ret_gpu']:.2f}/{out['ret_cpu']:.2f}/"
              f"{out['ret_ssd']:.2f}/{out['ret_none']:.2f} ssdW {out['ssd_write_GiB']:.1f} hbmP {out['hbm_paused_GiBs']:.0f} "
              f"pre {out['n_preempt']} hbm {out['hbm_GiB']:.1f}G ({time.time() - t0:.2f}s)")


# ============================================================================ 自測
def self_test():
    ok = []

    def check(cond, msg):
        if not cond:
            raise AssertionError("FAIL: " + msg)
        ok.append(msg)
        print("  PASS", msg)

    print("[self-test] 1. 單位")
    check(GiB == 1073741824 and MiB == 1048576, "GiB＝2^30 byte、MiB＝2^20 byte")
    for m in ("llama", "qwen"):
        bpt, _ = kv_bytes_per_token(m)
        bw = json.load(open(os.path.join(REPO, "results", "m10_precision_mi300x", f"speed_{m}", "bw.json")))
        check(bw["chunk_bytes"] == bpt * CHUNK_TOK, f"{m}: chunk_bytes（bw.json）＝ bytes/token × 512（{bpt} B/token）")
    check(kv_bytes_per_token("llama")[0] == 131072, "Llama-3.1-8B KV＝128 KiB/token")
    recs, _ = load_mooncake("toolagent")
    check(all((len(r["hash_ids"]) - 1) * 512 < r["input_length"] <= len(r["hash_ids"]) * 512 for r in recs),
          "Mooncake toolagent：(len(hash_ids)−1)×512 < input_length ≤ len×512 全部成立")
    cm = load_costs("mi300x", "llama")
    check(abs(cm.R(32768) / 3.65884 - 1) < 0.03, "重算擬合在 32K 對 compute_only 實測 <3%")
    check(abs(cm.h2d(4 * GiB, 32768, False) - 0.08213) / 0.08213 < 0.05, "H2D 擬合在 32K（4 GiB）對 load_only 實測 <5%")
    check(abs(cm.d2h(GiB) - 1 / 42.88) < 0.002, "D2H：1 GiB ÷ 42.9 GiB/s")
    check(cm.ssd_read(8053.8 * MiB) > 0.999 and cm.ssd_read(8053.8 * MiB) < 1.001, "SSD 讀：8053.8 MiB ÷ 8053.8 MiB/s＝1 s")
    for p in ("3090",):
        c3 = load_costs(p, "llama")
        try:
            c3.require("d2h_Bps")
            check(False, "3090 D2H 應為 NOT_MEASURED")
        except NotMeasured:
            check(True, "3090 缺 D2H → require() 拒跑（NOT_MEASURED 不估）")
    g = GapDist.lognormal(30.0, 1.0)
    check(abs(np.median(g.g) / 30.0 - 1) < 0.01, "lognormal 分位點的中位數＝30 s")
    check(abs(g.cond_rem(0.0, 1e9) / (30 * math.exp(0.5)) - 1) < 0.02, "lognormal 平均＝med·e^{σ²/2}（條件期望在 e=0）")
    ge = GapDist.empirical([1, 2, 3, 4], "t")
    check(abs(ge.cond_rem(2.5, 100) - 1.0) < 1e-9 and abs(ge.cond_rem(2.5) - 1.0) < 1e-9 and abs(ge.cond_rem(0.5) - 2.0) < 1e-9,
          "經驗分佈 E[G−e | G>e]（[1,2,3,4]：e=2.5 → 1.0；e=0.5 → 2.0）")

    print("[self-test] 2. trace 推 session")
    sessions, st = infer_sessions(recs)
    check(st["n_chain_pairs"] > 1000, f"推得 {st['n_chain_pairs']} 對前後輪（前綴與時間斷言都通過）")
    check(st["frac_in_ge_prev_in_plus_out"] > 0.5,
          f"下一輪 input ≥ 上一輪 input＋output 的比例 {st['frac_in_ge_prev_in_plus_out']:.3f}（交叉欄位檢查，>0.5）")
    for s in sessions[:2000]:
        for t in s:
            assert t["reuse"] <= t["inp"]
    check(True, "每一輪可重用的歷史 ≤ 這一輪 input")
    pool, pgaps, pst = mooncake_pool("toolagent", cm.max_tok)
    gd = GapDist.empirical(pgaps, "mooncake")

    print("[self-test] 3. 守恆、容量無限、間隔極端")
    for pol in POLICIES:
        out, sim = run_one(cm, pool, None, gd, pst["p_end"], 8, "tight", pol, 0, n_sess=60)
        n_turns = sum(len(x.turns) for x in sim.sessions)
        check(out["n_turns"] == n_turns and out["n_decisions"] == n_turns,
              f"{pol}: 每輪恰一個 TTFT、每次暫停恰一個決定（{n_turns} 輪）")
        cnt = Counter((r[0], r[1]) for r in sim.turn_rows)
        check(max(cnt.values()) == 1, f"{pol}: 沒有重複的 (session, 輪)")
        check(abs(sim.hbm_used - sum(s.hbm for s in sim.sessions)) < 1, f"{pol}: HBM 帳＝各 session 加總")
    base = {}
    for pol in POLICIES:
        out, sim = run_one(cm, pool, None, gd, pst["p_end"], 8, "inf", pol, 1, n_sess=60)
        base[pol] = out
    check(all(base["vllm"]["ttft_mean"] <= base[p]["ttft_mean"] + 1e-9 for p in POLICIES if p != "oracle"),
          "容量無限：KEEP（vllm）的平均 TTFT ≤ 每個線上策略")
    check(base["kappa"]["dec_KEEP"] == 1.0, "容量無限：κ 感知 100% KEEP（從沒發生過趕人 → 不寫副本）")
    check(base["oracle"]["ttft_mean"] <= base["vllm"]["ttft_mean"] + 1e-9, "容量無限：oracle ≤ KEEP")
    g0 = GapDist.const(1e-3)
    out, _ = run_one(cm, pool, g0, g0, pst["p_end"], 8, "tight", "kappa", 2, n_sess=60)
    keep_live = out["dec_KEEP"]
    check(keep_live >= 0.95, f"間隔→0（1 ms）：κ 感知 KEEP 比例 {keep_live:.3f} ≥ 0.95")
    for pol in ("kappa", "oracle"):
        out, sim = run_one(cm, pool, g0, g0, pst["p_end"], 8, "tight", pol, 2, n_sess=60)
        bad = [d for d in sim.decisions if d[7] and not (d[5] and d[2] == "DROP")]
        check(not bad, f"間隔→0：{pol} 不在暫停開始釋放 HBM（只有知道結束的 oracle 會丟最後一輪）；"
                       f"非最後一輪的決定 {dict(Counter(d[2] for d in sim.decisions if not d[5]))}")
    # 間隔→∞ 且容量緊：合成工作負載（每個 session 3 輪、每輪 8K token、暫停 1e6 s、session 之間不留空），
    # HBM 只放得下 3 個 session、C=8 → 壓力持續存在
    syn = dict(sessions=[[dict(inp=8192 * (k + 1), out=0, reuse=8192 * k, gap=(1e6 if k < 2 else INF)) for k in range(3)]
                         for _ in range(40)], slot_gaps=[0.0] * 48)
    for pol in ("kappa", "oracle"):
        cfg = Cfg(C=8, hbm_bytes=3 * 3 * 8192 * cm.bpt, cpu_bytes=1e18)
        sim = Sim(cm, cfg, syn, POLICIES[pol](), GapDist.const(1e6), 1 / 3).run()
        dec = Counter(d[2] for d in sim.decisions if not d[5])      # 不算最後一輪
        n = sum(dec.values())
        check(dec["KEEP"] <= 0.5 * n, f"間隔→∞ 且容量緊（合成、壓力持續）：{pol} 在非最後一輪的 KEEP {dec['KEEP']}/{n} ≤ 50%（{dict(dec)}）")
    # 同一個情境、間隔→0：應該全部 KEEP
    syn0 = dict(sessions=[[dict(inp=8192 * (k + 1), out=0, reuse=8192 * k, gap=(1e-3 if k < 2 else INF)) for k in range(3)]
                          for _ in range(40)], slot_gaps=[0.0] * 48)
    sim = Sim(cm, Cfg(C=8, hbm_bytes=3 * 3 * 8192 * cm.bpt, cpu_bytes=1e18), syn0, Kappa(), GapDist.const(1e-3), 1 / 3).run()
    dec = Counter(d[2] for d in sim.decisions if not d[5])
    check(dec["KEEP"] == sum(dec.values()), f"同一個合成情境、間隔→0：κ 感知非最後一輪全部 KEEP（{dict(dec)}）")

    # 回歸測試（run 20261019-141044-m16-pausesim 的失敗）：下一輪截斷歷史時，舊內容的副本寫入還在進行（NFS 寫很慢）
    cmn = load_costs("mi300x", "llama", "nfs")
    trunc = dict(sessions=[[dict(inp=8192, out=4096, reuse=0, gap=0.01),
                            dict(inp=9000, out=100, reuse=8192, gap=0.01),
                            dict(inp=20000, out=10, reuse=9100, gap=INF)] for _ in range(12)], slot_gaps=[0.0] * 20)
    for pol in ("vllm_tier", "kappa_tier", "vllm_cpu", "tokencake", "infercept", "oracle"):
        sim = Sim(cmn, Cfg(C=4, hbm_bytes=2 * 20010 * cmn.bpt, cpu_bytes=1.5 * 20010 * cmn.bpt), trunc, POLICIES[pol](),
                  GapDist.const(0.01), 1 / 3).run()       # run() 結尾的 check_books 會檢查副本長度與 CPU 保留量
        check(len(sim.turn_rows) == 36, f"截斷＋寫入進行中（NFS）：{pol} 跑完且帳目一致（36 輪）")

    print("[self-test] 4. oracle ≤ 線上策略")
    # 4a 靜態（嚴格）：每次暫停獨立評估同一個目標函數 J(a)=還原成本 + μ·bytes·HBM 佔用時間，oracle 用真實間隔取最小
    rng = np.random.default_rng(7)
    paus = [(t["inp"] + t["out"], t["gap"]) for s in pool[:400] for t in s[:-1]]
    for mu in (0.0, 1e-12, 1e-11, 1e-10):
        tot = Counter()
        for n, gp in paus:
            b = n * cm.bpt
            Jc = {"KEEP": mu * b * gp,
                  "CPU": cm.restore_est("cpu", n, n, False, "load") + mu * b * min(gp, cm.d2h(b)),
                  "SSD": cm.restore_est("ssd", n, n, False, "load") + mu * b * min(gp, cm.ssd_write(b)),
                  "DROP": cm.R(n)}
            tot["oracle"] += min(Jc.values())
            tot["keep"] += Jc["KEEP"]
            tot["drop"] += Jc["DROP"]
            tot["cpu"] += Jc["CPU"]
            Eg = gd.cond_rem(0.0)
            Jk = {"KEEP": mu * b * Eg, "CPU": Jc["CPU"], "SSD": Jc["SSD"], "DROP": Jc["DROP"]}
            tot["kappa_static"] += Jc[min(Jk, key=Jk.get)]
            tot["random"] += Jc[ACTIONS[int(rng.integers(4))]]
        check(all(tot["oracle"] <= v + 1e-9 for k, v in tot.items()),
              f"靜態（μ={mu:g} s/(B·s)）：oracle 總成本 {tot['oracle']:.1f} ≤ 每個規則（KEEP {tot['keep']:.1f}、"
              f"κ {tot['kappa_static']:.1f}、隨機 {tot['random']:.1f}）")
    # 4b 動態、C=1：只有一個活著的 session，oracle 丟掉已結束的 KV → 每一輪都不用還原（＝只剩 prefill 的下限）
    for seed in (0, 1, 2):
        res, ex = {}, {}
        for pol in POLICIES:
            out, sim = run_one(cm, pool, None, gd, pst["p_end"], 1, "tight", pol, seed, n_sess=80)
            res[pol] = out["ttft_mean"]
            ex[pol] = sum(r[5] + r[6] for r in sim.turn_rows)
        check(ex["oracle"] < 1e-9 and all(res["oracle"] <= res[p] + 1e-9 for p in POLICIES),
              f"動態 C=1（seed {seed}）：oracle 等待＋還原總和 {ex['oracle']:.3g} s＝0，平均 TTFT ≤ 所有策略")
    # 4c 動態、C>1：共用 GPU／鏈路的排隊讓「完美資訊的貪婪規則」不保證最佳 → 只報告，不當成通過條件
    viol, tot = [], 0
    for C, cap, seed in ((8, "tight", 0), (8, "mid", 1), (32, "tight", 2), (32, "mid", 0)):
        res = {}
        for pol in POLICIES:
            out, _ = run_one(cm, pool, None, gd, pst["p_end"], C, cap, pol, seed, n_sess=120)
            res[pol] = out["ttft_mean"]
        for p in BASELINES + ["kappa"]:
            tot += 1
            if res["oracle"] > res[p] * 1.001:
                viol.append((C, cap, seed, p, round(res["oracle"], 4), round(res[p], 4)))
    print(f"  INFO 動態 C>1：oracle 平均 TTFT > 線上策略的比較 {len(viol)}/{tot}：{viol}")
    print(f"[self-test] ALL PASS（{len(ok)} 項）")
    return 0


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("cmd", nargs="?", choices=["costs", "trace", "decision-map", "one", "scan", "analyze"])
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--raw", default=None, help="scan.csv／decisions.csv 的位置（大檔放 run 目錄；預設＝--out）")
    ap.add_argument("--platform", default="mi300x")
    ap.add_argument("--model", default="llama")
    ap.add_argument("--models", nargs="+", default=["llama", "qwen"])
    ap.add_argument("--ssd-dev", default="local")
    ap.add_argument("--C", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--caps", nargs="+", default=["tight", "mid", "loose", "real"])
    ap.add_argument("--gaps", nargs="+", default=GAPS_MAIN)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sens", action="store_true")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--cap", default="tight")
    ap.add_argument("--gap", default="mooncake")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--busy", default="endo")
    ap.add_argument("--restore", default="load")
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--cpu-mult", type=float, default=None)
    ap.add_argument("--policies", nargs="+", default=list(POLICIES))
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.cmd == "one":
        a.C = a.C[0]
    return {"costs": cmd_costs, "trace": cmd_trace, "decision-map": cmd_decision_map, "one": cmd_one,
            "scan": cmd_scan, "analyze": cmd_analyze}[a.cmd](a)


assert "torch" not in sys.modules and "vllm" not in sys.modules, "不准 import torch／vllm（會開 /dev/kfd）"


def assert_no_kfd():
    """確認本行程沒有開 /dev/kfd 或 /dev/dri（gpu_guard 會把開 kfd 的行程當入侵者）。"""
    bad = []
    try:
        for fd in os.listdir("/proc/self/fd"):
            try:
                t = os.readlink(f"/proc/self/fd/{fd}")
            except OSError:
                continue
            if t.startswith("/dev/kfd") or t.startswith("/dev/dri"):
                bad.append(t)
    except OSError:
        pass
    assert not bad, f"本行程開了 GPU 裝置：{bad}"
    return True

if __name__ == "__main__":
    rc = main()
    assert "torch" not in sys.modules and "vllm" not in sys.modules
    assert_no_kfd()
    sys.exit(rc or 0)
