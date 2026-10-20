"""x5_costs.py — X5：每個 block 的還原成本（第二階段 MI300X 實測）→ Cake 兩條線 → 還原時間表。

來源（全部是 repo 裡的實測檔；run_id 寫在 SOURCES）：
  - f(i)：results/m10_precision_mi300x/speed_<m>/bw.json 的 f_ms_median_by_chunk（GPU 閒，64 個 block）；
          i ≥ 64 用最小平方線性外插〔推算〕。GPU 忙＝閒 × 同一個 INT4 grid 的 compute_only（32K）忙÷閒。
  - CPU-BF16 載入：E0 load_only 的 t_h2d ÷ 64（32K，閒／忙）。
  - CPU-INT4 載入：E1 load_only 的 t_h2d ÷ 64 ＋ E1 cake（CPU 層）的 t_dequant ÷ n_load_int4（32K，閒／忙）。
    原寫法 K0：主線 INT4 grid；融合版：stage 42 grid（只量 32K CPU 層）。
  - 慢層：max(位元組 ÷ B, h)＋d（同 m10_iowall），B＝bw.json 的 b16／b32／b48。
  - 每個 block 的位元組：restores 的 bytes_stored ÷ n_blocks（E0＝BF16、E1＝INT4）。
校正（X5 §2.2）：兩條線推算 vs 實測 cake 中位數（8K／16K／32K 閒、32K 忙）。
  CPU 層：乘「實測÷推算」比值（在 16／32／64 個 block 之間線性內插、範圍外取最近的；忙只有 64 → 全部用 64 的）。
  慢層：推算誤差中位數 ≤ 5% 才用（gate）；不過 → 呼叫端要停下來。
"""
from __future__ import annotations

import functools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE)) if os.path.basename(os.path.dirname(HERE)) == "code" else os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "code"))
import m10_sweep_sim as SIM  # noqa: E402  （import，不改）

import p3_platform as PF  # noqa: E402  （P3_CONST：沒設＝MI300X；h200＝results/*_h200 與對應的 run）
R10 = PF.rdir("m10_precision_mi300x")
R10S = PF.rdir("m10_sweep_mi300x")
MiB = 1 << 20
GiB = 1 << 30

SOURCES = {
    "llama": dict(
        bw=os.path.join(R10, "speed_llama", "bw.json"),
        k0=(os.path.join(R10, "speed_llama", "restores.csv.gz"), PF.run("20261016-034902-m10-grid-int4-llama")),
        fused=(os.path.join(R10S, "speed_dqfast_int4_llama", "restores.csv"), PF.run("20261016-235027-m10-dqfast-grid-int4-llama"))),
    "qwen": dict(
        bw=os.path.join(R10, "speed_qwen", "bw.json"),
        k0=(os.path.join(R10, "speed_qwen", "restores.csv.gz"), PF.run("20261016-052143-m10-grid-int4-qwen")),
        fused=(os.path.join(R10S, "speed_dqfast_int4_qwen", "restores.csv"), PF.run("20261017-000209-m10-dqfast-grid-int4-qwen"))),
}
SLOW = ("b16", "b32", "b48")
GPUS = ("idle", "busy")
DQS = ("k0", "fused")
STATES = ("BF16", "INT4", "SLOW", "DROP")   # CPU-BF16、CPU-INT4、慢層 INT4、丟掉
GATE_SLOW_MEDIAN = 0.05
NMAX = 260                                  # trace 的 k_eff 最大 240


def _med(d, **kw):
    q = d
    for k, v in kw.items():
        q = q[q[k] == v]
    assert len(q) > 0, f"沒有列：{kw}"
    return q


class Costs:
    """variant：
      "reg"    ＝事先登記（CPU 層乘比值，比值在 16／32／64 內插、範圍外取最近；慢層不校正）
      "excess" ＝敏感度（只報告）：CPU 層改成「實測−推算」的差，在 k＝1（差＝0）到 16／32／64 之間線性內插、
                 64 以上照比例放大；慢層同樣用差的內插校正（推算對慢層 INT4 一律偏快）。
    """
    def __init__(self, model: str, variant: str = "reg"):
        import json
        assert variant in ("reg", "excess", "loadonly")
        self.variant = variant
        self.model = model
        src = SOURCES[model]
        bw = json.load(open(src["bw"]))
        self.bw_run = bw["run_id"]
        f = np.array(bw["f_ms_median_by_chunk"], dtype=float) / 1000.0
        x = np.arange(len(f))
        slope, icpt = np.polyfit(x, f, 1)
        self.f_slope, self.f_icpt = float(slope), float(icpt)
        xi = np.arange(NMAX + 1)
        self.f_idle = np.where(xi < len(f), np.r_[f, np.zeros(NMAX + 1 - len(f))], icpt + slope * xi)
        self.B = {k: float(bw["targets"][k]["GiBps"]) for k in SLOW}
        d0 = pd.read_csv(src["k0"][0])
        d0 = d0[d0.run_id == src["k0"][1]]
        df = pd.read_csv(src["fused"][0])
        df = df[df.run_id == src["fused"][1]]
        for d in (d0, df):
            assert not d.contaminated_gpu.astype(str).str.lower().eq("true").any(), "有被標污染的列"
            assert d.kv_bad.fillna(0).sum() == 0, "kv_bad"
        self.runs = dict(bw=self.bw_run, k0=src["k0"][1], fused=src["fused"][1])
        g32 = d0[d0.length == 32768]
        e0 = _med(g32, config="E0", mode="load_only", bw_name="cpu", gpu_state="idle")
        e1 = _med(g32, config="E1", mode="load_only", bw_name="cpu", gpu_state="idle")
        nbk = int(e0.n_blocks.iloc[0])
        self.bytes = dict(BF16=float(e0.bytes_stored.median()) / nbk, INT4=float(e1.bytes_stored.median()) / nbk)
        co = {s: _med(g32, mode="compute_only", bw_name="cpu", gpu_state=s).t_restore.median() for s in GPUS}
        self.busy_factor = float(co["busy"] / co["idle"])
        self.f = dict(idle=self.f_idle, busy=self.f_idle * self.busy_factor)
        self.h = {}
        self.d = {}
        for s in GPUS:
            e0 = _med(g32, config="E0", mode="load_only", bw_name="cpu", gpu_state=s)
            e1 = _med(g32, config="E1", mode="load_only", bw_name="cpu", gpu_state=s)
            self.h[("BF16", s)] = float(e0.t_h2d.median()) / nbk
            self.h[("INT4", s)] = float(e1.t_h2d.median()) / nbk
            for dq, dd in (("k0", d0), ("fused", df)):
                c = _med(dd[dd.length == 32768], config="E1", mode="cake", bw_name="cpu", gpu_state=s)
                self.d[(dq, s)] = float((c.t_dequant / c.n_load_int4).median())
        # 實測 cake 中位數（校正、驗證用）
        self.meas = {}
        for dq, dd in (("k0", d0), ("fused", df)):
            cake = dd[(dd["mode"] == "cake") & dd.config.isin(["E0", "E1"])]
            for (L, bwn, s, cfg), q in cake.groupby(["length", "bw_name", "gpu_state", "config"]):
                self.meas[(dq, int(L) // 512, bwn, s, "BF16" if cfg == "E0" else "INT4")] = float(q.t_restore.median())
        self.meas_lo = {}
        for dq, dd in (("k0", d0), ("fused", df)):
            lo = dd[(dd["mode"] == "load_only") & (dd.length == 32768) & (dd.bw_name == "cpu") & dd.config.isin(["E0", "E1"])]
            for (s_, cfg), q in lo.groupby(["gpu_state", "config"]):
                self.meas_lo[(dq, "BF16" if cfg == "E0" else "INT4", s_)] = float(q.t_restore.median())
        self._ratio_cache = {}
        self.validation = self._validate()

    # ---------------------------------------------------------------- 每個 block 的載入時間
    def load_block(self, loc: str, gpu: str, slow: str, dq: str) -> float:
        if loc == "BF16":                      # CPU-BF16
            return self.h[("BF16", gpu)]
        if loc == "INT4":                      # CPU-INT4
            return self.h[("INT4", gpu)] + self.d[(dq, gpu)]
        if loc == "SLOW":                      # 慢層 INT4
            return max(self.bytes["INT4"] / (self.B[slow] * GiB), self.h[("INT4", gpu)]) + self.d[(dq, gpu)]
        if loc == "SLOWBF16":                  # 慢層 BF16（只用在驗證）
            return max(self.bytes["BF16"] / (self.B[slow] * GiB), self.h[("BF16", gpu)])
        raise ValueError(loc)

    def _raw(self, state: str, s: int, k: int, gpu: str, slow: str, dq: str) -> float:
        """兩條線推算（未校正）：block 0..s-1 在 CPU-BF16（共用前綴），s..k-1 在 state。"""
        f = self.f[gpu]
        if k <= 0:
            return 0.0
        if state == "DROP":
            t_sh = 0.0
            if s > 0:
                _, t_sh, _, _ = SIM.simulate(list(f[:s]), [self.load_block("BF16", gpu, slow, dq)] * s)
            return t_sh + float(f[s:k].sum())
        lb = self.load_block("BF16", gpu, slow, dq)
        lx = self.load_block(state, gpu, slow, dq)
        load = [lb] * min(s, k) + [lx] * max(k - s, 0)
        _, t, _, _ = SIM.simulate(list(f[:k]), load)
        return t

    def _ratio(self, state: str, k: int, gpu: str, dq: str) -> float:
        """CPU 層的「實測÷推算」比值，在 16／32／64 個 block 之間線性內插（範圍外取最近）。"""
        key = (state, gpu, dq)
        if key not in self._ratio_cache:
            pts = []
            for n in (16, 32, 64):
                m = self.meas.get((dq, n, "cpu", gpu, state))
                if m is None and dq == "fused" and state == "BF16":
                    m = self.meas.get(("k0", n, "cpu", gpu, state))
                if m is None:
                    continue
                pts.append((n, m / self._raw(state, 0, n, gpu, "b16", dq)))
            if dq == "fused" and state == "INT4":
                # 融合版只量了 32K：其他長度沿用 32K 的比值〔推算〕
                pts = [p for p in pts if p[0] == 64]
            assert pts, f"沒有校正點 {key}"
            self._ratio_cache[key] = pts
        pts = self._ratio_cache[key]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return float(np.interp(k, xs, ys))

    def _excess(self, state: str, k: int, gpu: str, slow: str, dq: str) -> float:
        """敏感度用：實測−推算的差，k＝1 時 0，在有實測的 block 數之間線性內插，最大的實測點以上照推算比例放大。"""
        key = ("ex", state, gpu, slow, dq)
        if key not in self._ratio_cache:
            pts = [(1, 0.0)]
            tier = "cpu" if state in ("BF16", "INT4") else slow
            mst = state if state in ("BF16", "INT4") else "INT4"
            for n in (16, 32, 64):
                if dq == "fused" and mst == "INT4" and n != 64:
                    continue
                m = self.meas.get((dq, n, tier, gpu, mst))
                if m is None and dq == "fused":
                    m = self.meas.get(("k0", n, tier, gpu, mst)) if mst == "BF16" else None
                if m is None and dq == "fused" and tier != "cpu":
                    # 慢層的融合版沒量：用原寫法的差〔推算〕
                    m0 = self.meas.get(("k0", n, tier, gpu, mst))
                    if m0 is None:
                        continue
                    pts.append((n, m0 - self._raw(state, 0, n, gpu, slow, "k0")))
                    continue
                if m is None:
                    continue
                pts.append((n, m - self._raw(state, 0, n, gpu, slow, dq)))
            self._ratio_cache[key] = pts
        pts = self._ratio_cache[key]
        if k <= pts[-1][0]:
            return float(np.interp(k, [p[0] for p in pts], [p[1] for p in pts]))
        nmax, emax = pts[-1]
        return emax * self._raw(state, 0, k, gpu, slow, dq) / self._raw(state, 0, nmax, gpu, slow, dq)

    @functools.lru_cache(maxsize=None)
    def restore(self, state: str, s: int, k: int, gpu: str, slow: str, dq: str) -> float:
        if self.variant == "loadonly":
            # 只報告（X6 補充）：真實系統只認最長前綴、不做 Cake 的「前段重算＋後段載入」→ 有存的就整段載入；
            # 丟掉的照舊重算。CPU 層乘「實測只載÷推算只載」（32K），慢層不校正。
            if k <= 0:
                return 0.0
            if state == "DROP":
                return self._raw("DROP", s, k, gpu, slow, dq)
            lb = self.load_block("BF16", gpu, slow, dq)
            lx = self.load_block(state, gpu, slow, dq)
            t = lb * min(s, k) + lx * max(k - s, 0)
            if state in ("BF16", "INT4"):
                t *= self._lo_ratio(state, gpu, dq)
            return t
        t = self._raw(state, s, k, gpu, slow, dq)
        if self.variant == "reg":
            if state in ("BF16", "INT4"):
                t *= self._ratio(state, k, gpu, dq)
        else:
            if state in ("BF16", "INT4", "SLOW") and k > 0:
                t = max(t + self._excess(state, k, gpu, slow, dq), 1e-6)
        return t

    def _lo_ratio(self, state, gpu, dq):
        key = ("lo", state, gpu, dq)
        if key not in self._ratio_cache:
            m = self.meas_lo.get((dq, state, gpu)) or self.meas_lo.get(("k0", state, gpu))
            self._ratio_cache[key] = m / (64 * self.load_block(state, gpu, "b16", dq))
        return self._ratio_cache[key]

    def table(self, state: str, s: int, gpu: str, slow: str, dq: str) -> np.ndarray:
        if NMAX > 300:
            return self.table_fast(state, s, gpu, slow, dq)
        return np.array([self.restore(state, s, k, gpu, slow, dq) for k in range(NMAX + 1)])

    def raw_vec(self, state: str, s: int, gpu: str, slow: str, dq: str) -> np.ndarray:
        """FreeInference 用（k 到幾千）：兩條線的 min–max 形式（重算線至少拿 block 0，b ≥ 1），和 simulate 在一個 block 內。"""
        f = self.f[gpu][:NMAX + 1]
        F = np.r_[0.0, np.cumsum(f)]
        K = np.arange(NMAX + 1)
        if state == "DROP":
            t_sh = self._raw("BF16", 0, s, gpu, slow, dq) if s > 0 else 0.0
            out = t_sh + (F[np.maximum(K, s)] - F[s])
            for k in range(min(s, NMAX + 1)):
                out[k] = self._raw("BF16", 0, k, gpu, slow, dq)
            return out
        lb = self.load_block("BF16", gpu, slow, dq)
        lx = self.load_block(state, gpu, slow, dq)
        l = np.where(np.arange(NMAX + 1) < s, lb, lx)
        Cl = np.r_[0.0, np.cumsum(l)]
        out = np.zeros(NMAX + 1)
        for k in range(1, NMAX + 1):
            # 貪婪兩條線（同 m10_sweep_sim.simulate）：重算線在 F[b] 空下來時，載入線已經開始的載入數
            # j_b＝#{j: Lb[j] < F[b]}（同時空下來重算線先拿）；重算數 b*＝min{b ≥ 1: b + j_b ≥ k}
            Lb = Cl[k] - Cl[k - np.arange(k + 1)]          # 從後面載 j 個 block 的時間
            bs = np.arange(1, k + 1)
            jb = np.searchsorted(Lb, F[1:k + 1], "left")
            ok = np.nonzero(bs + jb >= k)[0]
            b = int(bs[ok[0]]) if len(ok) else k
            out[k] = max(F[b], Lb[k - b])
        return out

    def table_fast(self, state: str, s: int, gpu: str, slow: str, dq: str) -> np.ndarray:
        assert self.variant == "reg", "FreeInference 只跑事先登記的校正"
        t = self.raw_vec(state, s, gpu, slow, dq)
        if state in ("BF16", "INT4"):
            K = np.arange(NMAX + 1)
            pts = self._ratio_cache.get((state, gpu, dq))
            if pts is None:
                self._ratio(state, 64, gpu, dq)
                pts = self._ratio_cache[(state, gpu, dq)]
            t = t * np.interp(K, [p[0] for p in pts], [p[1] for p in pts])
        return t

    def prefill(self, k_from: int, L_tokens: int, gpu: str) -> float:
        """新一輪的 prefill：從 block k_from 開始到 L_tokens，每個 block f(i) 依 token 比例。"""
        f = self.f[gpu]
        nb = -(-L_tokens // 512)
        if nb <= k_from:
            return 0.0
        t = float(f[k_from:nb].sum())
        last = L_tokens - 512 * (nb - 1)
        t -= f[nb - 1] * (1 - last / 512.0)
        return t

    # ---------------------------------------------------------------- 驗證
    def _validate(self) -> list[dict]:
        out = []
        for (dq, n, bwn, s, st), m in self.meas.items():
            if bwn == "cpu":
                p_raw = self._raw(st, 0, n, s, "b16", dq)
                p = p_raw * self._ratio(st, n, s, dq)
                kind = "cpu"
            else:
                loc = "SLOW" if st == "INT4" else "SLOWBF16"
                p_raw = self._raw_slow(loc, n, s, bwn, dq)
                p = p_raw
                kind = "slow"
            out.append(dict(model=self.model, dq=dq, n_blocks=n, tier=bwn, gpu=s, state=st, kind=kind,
                            meas_s=m, pred_raw_s=p_raw, pred_used_s=p, err_raw=(p_raw - m) / m, err_used=(p - m) / m))
        return out

    def _raw_slow(self, loc, n, gpu, slow, dq):
        f = self.f[gpu]
        _, t, _, _ = SIM.simulate(list(f[:n]), [self.load_block(loc, gpu, slow, dq)] * n)
        return t

    def slow_gate(self) -> dict:
        v = [abs(r["err_raw"]) for r in self.validation if r["kind"] == "slow" and r["dq"] == "k0"]
        med = float(np.median(v))
        return dict(model=self.model, n=len(v), median_abs_err=med, max_abs_err=float(np.max(v)), passed=med <= GATE_SLOW_MEDIAN)

    def constants(self) -> dict:
        return dict(model=self.model, runs=self.runs, bytes_per_block=self.bytes, busy_factor=self.busy_factor,
                    f0_ms=1000 * self.f_idle[0], f63_ms=1000 * self.f_idle[63], f_slope_ms=1000 * self.f_slope,
                    h_ms={f"{a}_{b}": 1000 * v for (a, b), v in self.h.items()},
                    dequant_ms={f"{a}_{b}": 1000 * v for (a, b), v in self.d.items()},
                    slow_GiBps=self.B)


if __name__ == "__main__":
    import json
    for m in ("llama", "qwen"):
        c = Costs(m)
        print(json.dumps(c.constants(), ensure_ascii=False, indent=1))
        print(c.slow_gate())
        for r in sorted(c.validation, key=lambda r: (r["dq"], r["kind"], r["tier"], r["gpu"], r["state"], r["n_blocks"])):
            print(f"{r['dq']:5s} {r['tier']:4s} {r['gpu']:4s} {r['state']:4s} n={r['n_blocks']:3d} meas={r['meas_s']:.4f} raw={r['pred_raw_s']:.4f} ({r['err_raw']:+.1%}) used={r['pred_used_s']:.4f} ({r['err_used']:+.1%})")
        for st in STATES:
            print(st, [round(c.restore(st, 1, k, "idle", "b32", "k0"), 4) for k in (2, 8, 16, 32, 64, 128, 240)])
