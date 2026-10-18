"""analyze_map.py — 地圖的判定與報告（04_prereg_map.md v4 §4、§6、§10；06_simulator.md §21）。

輸入：run_map.py 的 map_cells.csv（一列＝一個格子）與每個工作的 per-session npz。
主指標（每個格子、每個做法）＝ssd_H_write_issued_GiB ＋ ssd_H_loadonly_read_GiB（npz 裡每個 session 的值；2H 版另報）。
等待時間護欄：回來請求的 ttftq 平均與 p99 都 ≤ 1.05 × C0（主視角：Llama、MI300X、GPU 閒、CPU 閒、SSD 1 張專用）；NOT_MEASURED → 當成通過、標出。
最好的對手＝所有非 oracle 的做法（含參數）裡通過護欄、主指標位元組最少的那個（完整樣本上選、bootstrap 固定）。
省多少＝(最好的對手 − O) ÷ 最好的對手。session 配對 bootstrap 1,000 次（固定 seed；FreeInference 各週當一份資料、週內分層重抽）；
15 個主格（E2、E3、E4、E5(30 分)、E6 × 63／170／435 GiB）做 studentized max-T 同時 CI：
  每格：點估計 θ̂、bootstrap 值 θ*、SE＝sd(θ*)；T_b＝max_c |θ*_cb − θ̂_c| / SE_c；q＝T_b 的 95 百分位；CI＝θ̂ ± q·SE。
0/0：最好的對手＝0 → 省多少定為 0（旗標 best_zero；bootstrap 裡 best*＝0 也當 0）；SE＝0 的格子不進 max-T（CI 退化成點）。
判定（§6）：GO＝至少一個主格，某份資料的 CI 下界 ≥15%，而且不同來源的另一份資料在同一格的 CI 下界 ≥5%；
  NO-GO＝所有進判定的真實資料、所有主格 CI 上界 <5%（而且每份都完整）；其他＝ASK。
  有效 session 數（1／Σp²）<100 的資料不進判定（FreeInference 用全部週合起來）。來源：{Bailian coder, traceA}、{Mooncake}、{FreeInference}、{WildChat}。
  §10 的 A1 提早 GO：不含 FreeInference 的資料已滿足 GO → 定案 GO；否則 PENDING_B（等 B 的 A1 範圍；用 --mode early／full）。
只報告（永遠不讓判定變 GO）：E1、O-loose、拆解（O−O-len、O−O-reuse、O-len 對 C4r、O-reuse 對 D-idle）、2H、敏感度（CPU 忙、3090、Qwen、70B、疊加×2／×4、E2 block）、合成。
用法：python code/p3_sim/analyze_map.py --cells <out>/map_cells.csv --mode early|full --out-prefix <path 前綴>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

GiB = 1 << 30
NOT = "NOT_MEASURED"
SEED = 20261018
N_BOOT = 1000
GUARD = 1.05
GUARD_SENS = [("1.05", 1.05), ("1.10", 1.10), ("1.20", 1.20), ("none", None)]       # 04 §4：護欄敏感度（只報告）；1.05 是登記的、判定用的那個
GO_STRONG, GO_WEAK, NOGO = 0.15, 0.05, 0.05
MIN_EFF_SESSIONS = 100.0
TRACES = ["Bailian coder", "Bailian traceA", "Mooncake", "WildChat", "FreeInference"]
SOURCE = {"Bailian coder": "alibaba", "Bailian traceA": "alibaba", "Mooncake": "mooncake", "WildChat": "wildchat", "FreeInference": "freeinference"}
MEMS = (63.0, 170.0, 435.0)
PRIMARY_OUTERS = [("E2", "session"), ("E3", "session"), ("E4", "session"), ("E5-T1800", "session"), ("E6", "block")]
PRIMARY_CELLS = [(o, m) for o, _ in PRIMARY_OUTERS for m in MEMS]            # 15 個主格
ENGINE_OF = dict(PRIMARY_OUTERS)
USECOLS = ["run_id", "stage", "comp", "slice", "trace_group", "source_group", "synthetic", "semi_synthetic", "overlay", "engine", "outer_code",
           "T_s", "mem_GiB", "model", "policy_label", "policy_family", "oracle", "h_tag", "view_group", "view_idx", "is_primary", "k_gpus", "job_id",
           "npz_path", "npz_sfx_H", "npz_sfx_2H", "prim_H_GiB", "prim_2H_GiB", "ttftq_mean_s", "ttftq_p99_s", "n_ret", "surprise_events",
           "surprise_blocks", "reads_fail", "reads_unverified", "ret_recompute_prefix_blocks", "recompute_prefix_blocks", "ssd_write_GiB",
           "ssd_loadonly_read_GiB", "n_req", "n_sess"]


def olabel(code, T):
    """外層做法的標籤：E5 帶 T（秒）。"""
    if code == "E5":
        return f"E5-T{int(float(T))}"
    return code


# ====================================================================== 讀資料
def load_cells(path):
    head = pd.read_csv(path, nrows=0).columns
    cols = [c for c in USECOLS if c in head]
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    df["olabel"] = [olabel(c, t) if str(t) not in ("", "nan") else c for c, t in zip(df.outer_code, df["T_s"])]
    for c in ("is_primary", "oracle", "synthetic", "semi_synthetic"):
        if c in df:
            df[c] = df[c].astype(str).str.lower().isin(("true", "1"))
    return df


class Index:
    """(trace, slice, engine, olabel, mem, model, overlay, view_group, policy_label, tag) → 一列（主視角那一列）。"""

    def __init__(self, df):
        d = df[(df.view_group != "main") | (df.is_primary)]
        self.rows = {}
        for r in d.to_dict("records"):
            key = (r["trace_group"], r["slice"], r["engine"], r["olabel"], float(r["mem_GiB"]), r["model"], int(r["overlay"]), r["view_group"],
                   r["policy_label"], r["h_tag"])
            self.rows.setdefault(key, r)
        self.slices = {}
        for r in d.to_dict("records"):
            self.slices.setdefault(r["trace_group"], set()).add(r["slice"])

    def row(self, trace, slc, engine, ol, mem, label, tag, model="llama", overlay=1, vg="main"):
        """非 oracle＝tag 'both'；oracle＝自己的 H／2H 工作。"""
        for t in ((tag,) if label in ORACLE_LABELS else ("both",)):
            r = self.rows.get((trace, slc, engine, ol, float(mem), model, overlay, vg, label, t))
            if r is not None:
                return r
        return None


ORACLE_LABELS = ("O", "O-len", "O-reuse", "O-loose")
AM_NM = NOT


def hbytes(r, h):
    """這一列在 H（'H'）或 2H 版本的主指標（GiB）。"""
    x = r["prim_H_GiB"] if h == "H" else r["prim_2H_GiB"]
    return float(x)


def sess_bytes(r, h):
    """per-session 的主指標位元組（依原始 session 編號排序）；並用 CSV 的列驗算總和。回傳 (sess_orig, x float 位元組)。"""
    z = np.load(r["npz_path"])
    sfx = r["npz_sfx_H"] if h == "H" else r["npz_sfx_2H"]
    assert isinstance(sfx, str) and sfx, f"{r['job_id']}: 沒有 {h} 版的 npz 欄位"
    x = (z["ssd_write_bytes" + sfx].astype(np.int64) + z["loadonly_ssd_read_bytes" + sfx].astype(np.int64))
    so = z["sess_orig"]
    order = np.argsort(so, kind="stable")
    tot = float(x.sum()) / GiB
    want = hbytes(r, h)
    assert abs(tot - want) <= 1e-6 * max(1.0, abs(want)), f"{r['job_id']}: npz 總和 {tot} GiB 和 CSV 的 {want} 不符"
    return so[order], x[order].astype(np.float64)


# ====================================================================== 護欄
def hist_pct(hist, edges, q):
    """直方圖的百分位（格內線性內插）。"""
    n = hist.sum()
    if n == 0:
        return float("nan")
    cum = np.cumsum(hist)
    target = q / 100.0 * n
    i = int(np.searchsorted(cum, target, side="left"))
    i = min(i, len(hist) - 1)
    prev = cum[i - 1] if i > 0 else 0
    frac = (target - prev) / max(hist[i], 1)
    return float(edges[i] * (edges[i + 1] / edges[i]) ** min(max(frac, 0.0), 1.0))


def _num(x):
    if isinstance(x, str):
        return None if x == NOT else float(x)
    x = float(x)
    return x


_TTFT_POOL_CACHE = {}


def _ttft_npz(path):
    z = np.load(path)
    if "ttft_vals" not in z.files:
        raise SystemExit(f"{path}: 沒有逐請求的 ttft_vals，無法精確合併幾週的護欄 p99（04 §4）；請用新版 run_map.py 重跑這個工作")
    return z["ttft_vals"].astype(np.float64), int(z["ttft_n"]), float(z["ttft_sum"]), int(z["ttft_nm"])


def trace_ttft(rows):
    """一份資料（一個格子、一個做法）的回來請求等待時間：(mean, p99, nm)。
    一個切片＝用列的精確值；FreeInference 幾週＝用 npz 的逐請求 ttft 串起來，p99＝np.percentile（精確，和直接對全部請求算一樣），
    平均＝總和÷個數（精確）。nm＝True：有 NOT_MEASURED。（只快取結果，不快取陣列。）"""
    if len(rows) == 1:
        r = rows[0]
        m, p = _num(r["ttftq_mean_s"]), _num(r["ttftq_p99_s"])
        return (None, None, True) if (m is None or p is None) else (m, p, False)
    key = tuple(r["npz_path"] for r in rows)
    if key in _TTFT_POOL_CACHE:
        return _TTFT_POOL_CACHE[key]
    xs = []
    n, s, nm = 0, 0.0, 0
    for r in rows:
        v_, nn_, ss_, nm_ = _ttft_npz(r["npz_path"])
        assert len(v_) == nn_, f"{r['npz_path']}: ttft_vals 個數 {len(v_)} ≠ ttft_n {nn_}"
        xs.append(v_)
        n += nn_
        s += ss_
        nm += nm_
    if nm:
        out = (None, None, True)
    elif n == 0:
        out = (float("nan"), float("nan"), False)
    else:
        out = (s / n, float(np.percentile(np.concatenate(xs), 99)), False)
    _TTFT_POOL_CACHE[key] = out
    return out


def guard_status(pol, c0, thr=GUARD):
    """回傳 (通過?, NM 旗標)。pol／c0＝(mean, p99, nm)。NOT_MEASURED（做法的）→ 通過並標出；C0 自己 NM（不該發生）也當通過並標出；
    平均或 p99 是 NaN（沒有回來的請求）→ 通過（沒有東西可比）。"""
    if pol[2] or c0[2]:
        return True, True
    for a, b in ((pol[0], c0[0]), (pol[1], c0[1])):
        if a is None or b is None or math.isnan(a) or math.isnan(b):
            continue
        if a > thr * b + 1e-12:
            return False, False
    return True, False


# ====================================================================== 一個格子（一份資料）
def trace_cell(ix, trace, slices, engine, ol, mem, h, model="llama", overlay=1, vg="main", labels=None):
    """一份資料（可能是好幾個切片）在一個格子裡，每個做法的主指標總和（GiB）與等待時間。缺任何一個切片的某個做法 → 該做法不在 res 裡。"""
    labels = labels or [p for p in LABELS]
    res = {}
    for lab in labels:
        rows = [ix.row(trace, s, engine, ol, mem, lab, h, model, overlay, vg) for s in slices]
        if any(r is None for r in rows):
            continue
        res[lab] = dict(rows=rows, GiB=sum(hbytes(r, h) for r in rows), ttft=trace_ttft(rows))
    return res


LABELS = ["C0", "C1", "C1-8", "C2", "C3", "C6", "C4-8192", "C4-16384", "C4-32768", "C4r-8192", "C4r-16384", "C4r-32768", "D-len-8192",
          "D-len-16384", "D-len-32768", "D-idle-300", "D-idle-1800", "D-idle-3600", "D-idle-14400", "O", "O-len", "O-reuse", "O-loose"]
NON_ORACLE = [x for x in LABELS if x not in ORACLE_LABELS]


def best_of(res, candidates=None, guard=True, thr=GUARD):
    """最好的對手：非 oracle（或給定的 candidates）裡通過護欄、主指標最少的。回傳 dict(label, GiB, excluded, nm, passed)；沒有任何一個 → label None。"""
    cands = [c for c in (candidates or NON_ORACLE) if c in res]
    out = dict(label=None, GiB=float("nan"), excluded=[], nm=[], passed=[], n_candidates=len(cands))
    if "C0" not in res or not cands:
        return out
    c0 = res["C0"]["ttft"]
    for c in cands:
        ok, nmf = guard_status(res[c]["ttft"], c0, thr) if guard else (True, False)
        if not ok:
            out["excluded"].append(c)
            continue
        if nmf:
            out["nm"].append(c)
        out["passed"].append(c)
    if out["passed"]:
        lab = min(out["passed"], key=lambda c: (res[c]["GiB"], LABELS.index(c)))
        out.update(label=lab, GiB=res[lab]["GiB"])
    return out


def savings_point(best, o):
    """(best − O)/best；best＝0 → 0（旗標）。"""
    if best is None or o is None or math.isnan(best) or math.isnan(o):
        return float("nan"), False
    if best == 0.0:
        return 0.0, True
    return (best - o) / best, False


# ====================================================================== bootstrap（paired、stratified、max-T）
def effective_sessions(n_req):
    n = np.asarray(n_req, float)
    p = n / n.sum()
    return float(1.0 / np.sum(p * p))


def bootstrap_maxT(strata, cells, n_boot=N_BOOT, seed=None):
    """strata：[(名字, 這個層的 session 數 S, {cell: (x_best (S,), x_O (S,) 或 None)})]，每層用同一組重抽（所有格子、所有做法 paired）。
    θ_c＝(Σ best − Σ O)/Σ best（Σ best＝0 → 0）。回傳 dict(theta_hat, theta_star(B,C), se, q, lo, hi)。"""
    C = len(cells)
    names = [s[0] for s in strata]
    mats = []
    for (_, S, d) in strata:
        X = np.zeros((S, 2 * C))
        for j, c in enumerate(cells):
            xb, xo = d[c]
            X[:, 2 * j] = xb
            X[:, 2 * j + 1] = xo
        mats.append(X)
    full = sum(X.sum(axis=0) for X in mats)

    def theta(tot):
        b, o = tot[0::2], tot[1::2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(b > 0, (b - o) / b, 0.0)
        return t
    th_hat = theta(full)
    rng = np.random.default_rng(seed)
    order = sorted(range(len(strata)), key=lambda i: names[i])
    th_star = np.zeros((n_boot, C))
    for b in range(n_boot):
        tot = np.zeros(2 * C)
        for i in order:
            S = strata[i][1]
            w = np.bincount(rng.integers(0, S, S), minlength=S).astype(np.float64)
            tot += w @ mats[i]
        th_star[b] = theta(tot)
    se = th_star.std(axis=0, ddof=1)
    ok = se > 0
    if ok.any():
        T = np.max(np.abs(th_star[:, ok] - th_hat[ok]) / se[ok], axis=1)
        q = float(np.quantile(T, 0.95))
    else:
        q = 0.0
    return dict(theta_hat=th_hat, theta_star=th_star, se=se, q=q, lo=th_hat - q * se, hi=th_hat + q * se)


# ====================================================================== 判定（§6、§10）
def decide(traces, mode="full"):
    """traces：{名字: dict(eligible, complete, synthetic, source, cells={cell: dict(lo, hi, point)})}；cell＝(外層, 記憶體)。
    mode＝'full'（§6 照用全部資料）｜'early'（§10：只看不含 FreeInference 的資料；GO→定案 GO，否則 PENDING_B）。
    只有 eligible、complete、非合成的資料進判定。回傳 dict(verdict, reason, go_cells, used)。"""
    use = {t: v for t, v in traces.items() if v["eligible"] and v["complete"] and not v.get("synthetic")
           and not (mode == "early" and v["source"] == "freeinference")}
    why_not = {t: ("合成" if v.get("synthetic") else "有效 session <100" if not v["eligible"] else "不完整" if not v["complete"] else "early 模式不含 FreeInference")
               for t, v in traces.items() if t not in use}
    go_cells = []
    cells = sorted({c for v in use.values() for c in v["cells"]})
    for c in cells:
        for t1, v1 in use.items():
            c1 = v1["cells"].get(c)
            if c1 is None or not c1["lo"] >= GO_STRONG:
                continue
            for t2, v2 in use.items():
                if v2["source"] == v1["source"]:
                    continue
                c2 = v2["cells"].get(c)
                if c2 is not None and c2["lo"] >= GO_WEAK:
                    go_cells.append(dict(cell=c, strong=t1, weak=t2, strong_lo=c1["lo"], weak_lo=c2["lo"]))
    res = dict(used=sorted(use), not_used=why_not, go_cells=go_cells, mode=mode)
    if go_cells:
        g = go_cells[0]
        res.update(verdict="GO" if mode == "full" else "GO（早期，定案）",
                   reason=f"{g['cell'][0]}@{g['cell'][1]:g} GiB：{g['strong']} 的 CI 下界 {g['strong_lo']:.1%} ≥15%，不同來源的 {g['weak']} 的 CI 下界 {g['weak_lo']:.1%} ≥5%"
                          f"（共 {len(go_cells)} 組）")
        return res
    if mode == "early":
        res.update(verdict="PENDING_B", reason="不含 FreeInference 的資料沒有滿足 GO；判定等 B 的 A1 範圍跑完（04 §10）")
        return res
    if not use:
        res.update(verdict="ASK", reason="沒有任何進判定的真實資料（有效 session <100、不完整或只有合成資料）")
        return res
    incomplete = [t for t, v in traces.items() if v["eligible"] and not v["complete"] and not v.get("synthetic")]
    all_low = all(c["hi"] < NOGO for v in use.values() for c in v["cells"].values()) and \
        all(len(v["cells"]) == len(PRIMARY_CELLS) for v in use.values())
    if all_low and not incomplete:
        res.update(verdict="NO-GO", reason=f"所有進判定的資料（{', '.join(sorted(use))}）、所有 15 個主格，同時 CI 上界都 <5%")
    else:
        top = max((c["lo"] for v in use.values() for c in v["cells"].values()), default=float("nan"))
        res.update(verdict="ASK", reason="沒有滿足 GO，也不是全部上界 <5%" + (f"；不完整的資料：{incomplete}" if incomplete else "")
                   + f"（最高的 CI 下界 {top:.1%}）")
    return res


# ====================================================================== 主分析
def analyze(df, mode="full", n_boot=N_BOOT, h_main="H", verbose=True):
    ix = Index(df)
    out_rows, boot_info, traces = [], {}, {}
    tr_present = [t for t in ix.slices if t in TRACES or t == "synth"]
    for trace in sorted(tr_present, key=lambda t: (TRACES.index(t) if t in TRACES else 99)):
        slices = sorted(ix.slices[trace])
        info = dict(slices=slices, source=SOURCE.get(trace, "synthetic"), synthetic=(trace == "synth"), cells={}, eligible=False, complete=True)
        # 有效 session 數：用任一個主格 C0 的 npz 裡每個 session 的請求數（所有切片合起來）
        nreq_all = []
        for s in slices:
            r0 = ix.row(trace, s, "session", "E2", MEMS[0], "C0", "both")
            if r0 is None:
                info["complete"] = False
                continue
            nreq_all.append(np.load(r0["npz_path"])["n_req"])
        info["eff_sessions"] = effective_sessions(np.concatenate(nreq_all)) if nreq_all else float("nan")
        info["eligible"] = bool(info["eff_sessions"] >= MIN_EFF_SESSIONS)
        for hh in ("H", "2H"):
            strata, cells_ok = [], []
            tab = {}
            for (ol, mem) in PRIMARY_CELLS:
                eng = ENGINE_OF[ol]
                res = trace_cell(ix, trace, slices, eng, ol, mem, hh)
                if any(l not in res for l in NON_ORACLE + ["O"]):
                    info["complete"] = info["complete"] and hh != h_main      # 2H 版缺格只影響 2H 的報告，不擋判定
                    continue
                best = best_of(res)
                if best["label"] is None:
                    info["complete"] = info["complete"] and hh != h_main
                    continue
                pt, bz = savings_point(best["GiB"], res["O"]["GiB"])
                tab[(ol, mem)] = dict(res=res, best=best, point=pt, best_zero=bz)
            # bootstrap：需要每個切片的 per-session 位元組（best 與 O 兩個做法）
            cells = [c for c in PRIMARY_CELLS if c in tab]
            if cells:
                per_slice = {}
                for si, s in enumerate(slices):
                    d, ref = {}, None
                    for c in cells:
                        eng = ENGINE_OF[c[0]]
                        rb = ix.row(trace, s, eng, c[0], c[1], tab[c]["best"]["label"], hh)
                        ro = ix.row(trace, s, eng, c[0], c[1], "O", hh)
                        sb, xb = sess_bytes(rb, hh)
                        so, xo = sess_bytes(ro, hh)
                        assert np.array_equal(sb, so), f"{trace}/{s}/{c}: best 與 O 的 session 對不上"
                        if ref is None:
                            ref = sb
                        assert np.array_equal(ref, sb), f"{trace}/{s}/{c}: 不同格子的 session 編號對不上"
                        d[c] = (xb, xo)
                    per_slice[s] = (len(ref), d)
                strata = [(s, per_slice[s][0], per_slice[s][1]) for s in slices]
                rng_seed = [SEED, TRACES.index(trace) if trace in TRACES else 99]
                bs = bootstrap_maxT(strata, cells, n_boot, rng_seed)
                # 驗算：bootstrap 的全樣本點估計和 CSV 算出來的點估計一致
                for j, c in enumerate(cells):
                    if not tab[c]["best_zero"]:
                        assert abs(bs["theta_hat"][j] - tab[c]["point"]) < 1e-9 * max(1, abs(tab[c]["point"])) + 1e-9, (trace, c, bs["theta_hat"][j], tab[c]["point"])
                for j, c in enumerate(cells):
                    t = tab[c]
                    row = dict(kind="primary", trace=trace, source_group=info["source"], synthetic=info["synthetic"], outer=c[0], mem_GiB=c[1],
                               h=hh, best_label=t["best"]["label"], best_GiB=t["best"]["GiB"], O_GiB=t["res"]["O"]["GiB"], savings=t["point"],
                               se=float(bs["se"][j]), q=bs["q"], ci_lo=float(bs["lo"][j]), ci_hi=float(bs["hi"][j]), best_zero=t["best_zero"],
                               guard_excluded=";".join(t["best"]["excluded"]), guard_nm=";".join(t["best"]["nm"]), n_boot=n_boot,
                               eligible=info["eligible"], eff_sessions=info["eff_sessions"])
                    out_rows.append(row)
                    if hh == h_main:
                        info["cells"][c] = dict(lo=row["ci_lo"], hi=row["ci_hi"], point=row["savings"])
                boot_info[(trace, hh)] = dict(q=bs["q"], cells=len(cells))
            if hh == h_main and len(info["cells"]) < len(PRIMARY_CELLS):
                info["complete"] = False
        traces[trace] = info
        if verbose:
            print(f"[{trace}] 切片 {len(slices)}；有效 session {info['eff_sessions']:.1f}；進判定 {info['eligible']}；完整 {info['complete']}", flush=True)
    verdict = decide(traces, mode)
    # 合成資料「如果它是真實的」會是什麼（只標示，不是判定）
    syn = {t: v for t, v in traces.items() if v.get("synthetic")}
    pseudo = None
    if syn:
        fake = {t: dict(v, synthetic=False, source="synthetic") for t, v in syn.items()}
        pseudo = decide(fake, "full")
    reports = report_only(ix, traces)
    return dict(rows=out_rows, traces=traces, verdict=verdict, synthetic_pseudo_verdict=pseudo, reports=reports, boot=boot_info)


# ====================================================================== 只報告的項目
def report_only(ix, traces):
    """只算點估計（不做 bootstrap、不進判定）：每份資料 × 每個外層做法 × 每個記憶體（A2 的整張地圖，含 E1、E5 的另兩個 T、E6 與 E2 的 block 版），
    加上 O-loose、拆解、敏感度（Qwen、70B、疊加、CPU 忙、3090）。"""
    rows = []
    seen = sorted({(k[0], k[2], k[3], k[4], k[5], k[6], k[7]) for k in ix.rows})      # (trace, engine, olabel, mem, model, overlay, view_group)
    for (trace, eng, ol, mem, model, ov, vg) in seen:
        if trace not in traces:
            continue
        slices = sorted(ix.slices[trace])
        if ol == "E1":      # E1 沒有 SSD，不進判定、也沒有「省多少」：只報等待時間與重算（04 §6）
            if model == "llama" and ov == 1 and vg == "main":
                for lab in LABELS:
                    rs = [ix.row(trace, s_, eng, ol, mem, lab, "H", model, ov, vg) for s_ in slices]
                    if any(r is None for r in rs):
                        continue
                    t = trace_ttft(rs)
                    rows.append(dict(kind="report_E1", trace=trace, source_group=SOURCE.get(trace, "synthetic"), synthetic=(trace == "synth"),
                                     outer=f"{ol}@{eng}", mem_GiB=mem, h="H", item=lab, model=model, overlay=ov, view_group=vg,
                                     ttftq_mean_s=t[0] if not t[2] else AM_NM, ttftq_p99_s=t[1] if not t[2] else AM_NM,
                                     ret_recompute_prefix_blocks=int(sum(r["ret_recompute_prefix_blocks"] for r in rs)),
                                     surprise_blocks=int(sum(r["surprise_blocks"] for r in rs))))
            continue
        for hh in ("H", "2H"):
            res = trace_cell(ix, trace, slices, eng, ol, mem, hh, model, ov, vg)
            if "O" not in res or "C0" not in res:
                continue
            best = best_of(res)
            pt, bz = savings_point(best["GiB"], res["O"]["GiB"])
            parts = ([vg[5:] if vg.startswith("sens_") else vg] if vg != "main" else []) + ([model] if model != "llama" else []) + \
                ([f"ov{ov}"] if ov > 1 else [])
            kind = ("sens_" + "_".join(parts)) if parts else ("primary_point" if ((ol, mem) in PRIMARY_CELLS and eng == ENGINE_OF.get(ol)) else "map")
            if eng == "block" and ol == "E2" and not parts:
                kind = "sens_E2_block"
            base = dict(trace=trace, source_group=SOURCE.get(trace, "synthetic"), synthetic=(trace == "synth"), outer=f"{ol}@{eng}", mem_GiB=mem,
                        h=hh, model=model, overlay=ov, view_group=vg, best_label=best["label"], best_GiB=best["GiB"],
                        guard_excluded=";".join(best["excluded"]), guard_nm=";".join(best["nm"]), best_zero=bz)
            rows.append(dict(base, kind=kind, item="O", O_GiB=res["O"]["GiB"], savings=pt))
            if hh != "H" or vg != "main" or model != "llama" or ov != 1:
                continue
            # 拆解與 O-loose（只在主視角、Llama、H）
            for gl, gthr in GUARD_SENS:          # 護欄的敏感度（04 §4，只報告）：同一批做法、換護欄的鬆緊
                bg = best_of(res, guard=gthr is not None, thr=gthr if gthr is not None else GUARD)
                sg, bzg = savings_point(bg["GiB"], res["O"]["GiB"])
                rows.append(dict(base, kind="report_guard_sens", item=f"guard_{gl}", guard_thr=gl, best_label=bg["label"], best_GiB=bg["GiB"],
                                 guard_excluded=";".join(bg["excluded"]), guard_nm=";".join(bg["nm"]), best_zero=bzg, O_GiB=res["O"]["GiB"], savings=sg))
            for lab in ("O-loose", "O-len", "O-reuse"):
                if lab in res and not math.isnan(best["GiB"]):
                    s2, _ = savings_point(best["GiB"], res[lab]["GiB"])
                    rows.append(dict(base, kind="report_oracle_variant", item=lab, O_GiB=res[lab]["GiB"], savings=s2))
            if "O" in res and "O-len" in res and best["GiB"] > 0:
                rows.append(dict(base, kind="report_decomp", item="O_minus_Olen", O_GiB=res["O"]["GiB"], savings=(res["O-len"]["GiB"] - res["O"]["GiB"]) / best["GiB"]))
            if "O" in res and "O-reuse" in res and best["GiB"] > 0:
                rows.append(dict(base, kind="report_decomp", item="O_minus_Oreuse", O_GiB=res["O"]["GiB"], savings=(res["O-reuse"]["GiB"] - res["O"]["GiB"]) / best["GiB"]))
            for lab, fam in (("O-len", "C4r"), ("O-reuse", "D-idle")):
                b2 = best_of(res, [c for c in NON_ORACLE if c.startswith(fam)])
                if lab in res and b2["label"]:
                    s2, _ = savings_point(b2["GiB"], res[lab]["GiB"])
                    rows.append(dict(base, kind="report_decomp", item=f"{lab}_vs_best_{fam}", best_label=b2["label"], best_GiB=b2["GiB"],
                                     guard_excluded=";".join(b2["excluded"]), O_GiB=res[lab]["GiB"], savings=s2))
    return rows


# ====================================================================== 輸出
def fmt_pct(x):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.1f}%"


def write_outputs(result, prefix, mode, cells_path):
    os.makedirs(os.path.dirname(os.path.abspath(prefix)) or ".", exist_ok=True)
    rows = result["rows"]
    # summary CSV：主格（有 CI）＋ 只報告
    allrows = rows + result["reports"]
    keys = []
    for r in allrows:
        for k in r:
            if k not in keys:
                keys.append(k)
    rid = os.environ.get("RUN_ID", "adhoc")
    ts = pd.Timestamp.now(tz="Asia/Taipei").isoformat(timespec="seconds")
    with open(prefix + "_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["run_id", "ts"] + keys, restval="")
        w.writeheader()
        for r in allrows:
            w.writerow(dict(run_id=rid, ts=ts, **r))
    v = result["verdict"]
    js = dict(run_id=rid, ts=ts, mode=mode, cells_csv=cells_path, verdict=v["verdict"], reason=v["reason"], go_cells=v["go_cells"], used=v["used"],
              not_used=v["not_used"], n_boot=N_BOOT, seed=SEED, guard=GUARD,
              traces={t: dict(slices=i["slices"], eligible=i["eligible"], complete=i["complete"], eff_sessions=i["eff_sessions"], source=i["source"],
                              synthetic=i["synthetic"]) for t, i in result["traces"].items()},
              guard_sensitivity=[{k: (None if isinstance(x, float) and math.isnan(x) else x) for k, x in r.items()} for r in result["reports"] if r["kind"] == "report_guard_sens"],
              synthetic_pseudo_verdict=result["synthetic_pseudo_verdict"], boot_q={f"{k[0]}|{k[1]}": b for k, b in result["boot"].items()})
    json.dump(js, open(prefix + "_verdict.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    # markdown：整張地圖（主格有 CI；其他只有點）
    L = [f"# 地圖判定（{mode}）", "", f"- 判定：**{v['verdict']}**", f"- 理由：{v['reason']}", f"- 進判定的資料：{', '.join(v['used']) or '（無）'}；沒進的：{json.dumps(v['not_used'], ensure_ascii=False)}",
         f"- bootstrap {N_BOOT} 次、seed {SEED}、max-T 同時 95% CI；等待時間護欄 {GUARD}×C0；run `{rid}`", ""]
    if result["synthetic_pseudo_verdict"]:
        L += [f"- **合成資料**（不是判定）：如果當成真實資料，規則會給 {result['synthetic_pseudo_verdict']['verdict']}（{result['synthetic_pseudo_verdict']['reason']}）", ""]
    for hh, title in (("H", "截尾 H（判定用）"), ("2H", "截尾 2H（只報告）")):
        L += [f"## 15 個主格：省多少＝(最好的對手 − O) ÷ 最好的對手；{title}；點估計［同時 CI 下界, 上界］", ""]
        trs = [t for t in result["traces"]]
        L += ["| 外層做法 | 記憶體 GiB | " + " | ".join(f"{t}{'（合成）' if result['traces'][t]['synthetic'] else ''}" for t in trs) + " |",
              "|:--|--:|" + "|".join(":--" for _ in trs) + "|"]
        for (ol, mem) in PRIMARY_CELLS:
            cells = []
            for t in trs:
                r = next((x for x in rows if x["trace"] == t and x["outer"] == ol and x["mem_GiB"] == mem and x["h"] == hh), None)
                cells.append("—" if r is None else f"{fmt_pct(r['savings'])} [{fmt_pct(r['ci_lo'])}, {fmt_pct(r['ci_hi'])}]" +
                             (f"<br>最好：{r['best_label']}" if hh == "H" else "") + ("<br>best=0" if r["best_zero"] else ""))
            L.append(f"| {ol} | {mem:g} | " + " | ".join(cells) + " |")
        L.append("")
    gs = [r for r in result["reports"] if r["kind"] == "report_guard_sens" and (r["outer"].split("@")[0], r["mem_GiB"]) in PRIMARY_CELLS
          and r["outer"].split("@")[1] == ENGINE_OF.get(r["outer"].split("@")[0])]
    if gs:
        trs = [t for t in result["traces"]]
        L += ["## 護欄敏感度（只報告；點估計；H；主格）：護欄 1.05 是登記的（判定用），其他只用來看判定有多依賴 5%", "",
              "| 外層做法 | 記憶體 GiB | 護欄 | " + " | ".join(trs) + " |", "|:--|--:|:--|" + "|".join(":--" for _ in trs) + "|"]
        for (ol, mem) in PRIMARY_CELLS:
            for gl, _ in GUARD_SENS:
                cells = []
                for t in trs:
                    r = next((x for x in gs if x["trace"] == t and x["outer"].split("@")[0] == ol and x["mem_GiB"] == mem and x["h"] == "H"
                              and x["guard_thr"] == gl), None)
                    cells.append("—" if r is None else f"{fmt_pct(r['savings'])}（{r['best_label']}）")
                L.append(f"| {ol} | {mem:g} | {gl if gl != 'none' else '不設'} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## 整張地圖（只有點估計，H；只報告）", ""]
    rep = [r for r in result["reports"] if r["kind"] in ("map", "primary_point", "sens_E2_block") and r["h"] == "H" and r["item"] == "O"]
    if rep:
        d = pd.DataFrame(rep)
        d["col"] = d.trace
        for (eng_ol), g in d.groupby("outer"):
            L += [f"### {eng_ol}", "", "| 記憶體 GiB | " + " | ".join(sorted(g.col.unique())) + " |", "|--:|" + "|".join("--:" for _ in g.col.unique()) + "|"]
            for mem, g2 in g.groupby("mem_GiB"):
                L.append(f"| {mem:g} | " + " | ".join(fmt_pct(float(g2[g2.col == c].savings.iloc[0])) if (g2.col == c).any() else "—" for c in sorted(g.col.unique())) + " |")
            L.append("")
    sens = [r for r in result["reports"] if str(r["kind"]).startswith("sens_") and r["kind"] != "sens_E2_block" and r["h"] == "H"]
    if sens:
        L += ["## 敏感度（只報告；點估計；半合成標在 kind）", "", "| kind | 資料 | 外層 | 記憶體 | 最好的對手 | 省多少 |", "|:--|:--|:--|--:|:--|--:|"]
        for r in sens:
            L.append(f"| {r['kind']} | {r['trace']} | {r['outer']} | {r['mem_GiB']:g} | {r['best_label']} | {fmt_pct(r['savings'])} |")
        L.append("")
    open(prefix + "_map.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    return prefix


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--mode", choices=["early", "full"], default="full")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    df = load_cells(a.cells)
    res = analyze(df, a.mode, a.boot)
    write_outputs(res, a.out_prefix, a.mode, a.cells)
    print(json.dumps(dict(verdict=res["verdict"]["verdict"], reason=res["verdict"]["reason"]), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
