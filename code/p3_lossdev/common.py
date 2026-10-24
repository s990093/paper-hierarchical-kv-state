"""common.py — 格式表、固定的決策介面（分數 → 動作）、成本、oracle、門檻挑選、診斷指標、計時。
**合成資料、方法開發，不是結論**（docs/phase3_20261017/07_lossdev_synth.md §2.2–2.3）。
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import os
import pickle
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(REPO, "results", "p3_lossdev")
PLACEHOLDER_TABLE = os.path.join(RES, "format_table_PLACEHOLDER.json")
REAL_TABLE = os.path.join(REPO, "results", "p3_formats", "format_table.json")

FMT = ["bf16", "int8", "int4", "int2"]
BF16, INT8, INT4, INT2 = 0, 1, 2, 3
GUARD_S = 6 * 3600.0          # valid 上「沒回來、而且觀察窗還剩 ≥ 6 小時」才當成不回來（§2.2）
LAMBDAS = (1.0, 4.0, 16.0)
LAMBDA_MAIN = 4.0

F0 = ["a_log"]
F1 = ["ctx_log", "out_log", "trig_tool", "meta_cat", "meta_flag", "meta_score", "hour"]
F2T = ["prev_gap_log", "mean_gap_log", "min_gap_log", "last_human_gap_log", "sess_age_log"]
F2C = ["turn_idx", "n_tool_ret", "n_human_ret"]
FAMILIES = {"F0": F0, "F1": F1, "F2t": F2T, "F2c": F2C}
ALL_FEATS = F0 + F1 + F2T + F2C


def now_iso():
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def code_sha():
    h = hashlib.sha256()
    for fn in sorted(os.listdir(HERE)):
        if fn.endswith(".py"):
            h.update(fn.encode())
            h.update(open(os.path.join(HERE, fn), "rb").read())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- 格式表
def write_placeholder():
    os.makedirs(RES, exist_ok=True)
    tab = dict(
        label="PLACEHOLDER：大小由位元數算（不是量測）；品質判準照 04_prereg_map §3（BF16、8-bit 一律過；INT4 只在 context ≤ 32K 過；"
              "INT2 只能給不會再被讀的）。results/p3_formats/format_table.json 出來後換掉重跑。",
        model="llama-3.1-8b", bytes_per_token_bf16=131072,
        formats={
            "bf16": dict(ratio=1.0, pass_max_ctx=None, derivation="16 bit"),
            "int8": dict(ratio=(128 * 8 + 16) / (128 * 16), pass_max_ctx=None, derivation="8 bit＋每 token 每 head 一個 fp16 scale（128 維）"),
            "int4": dict(ratio=(32 * 4 + 32) / (32 * 16), pass_max_ctx=32768, derivation="KIVI：4 bit＋每 32 個一組 fp16 scale＋zero"),
            "int2": dict(ratio=(32 * 2 + 32) / (32 * 16), pass_max_ctx=0, derivation="KIVI：2 bit＋每 32 個一組 fp16 scale＋zero；讀取一律算不過"),
        })
    with open(PLACEHOLDER_TABLE, "w", encoding="utf-8") as f:
        json.dump(tab, f, ensure_ascii=False, indent=1)
    return tab


def load_table(path=None):
    path = path or PLACEHOLDER_TABLE
    if path == PLACEHOLDER_TABLE and not os.path.exists(path):
        write_placeholder()
    tab = json.load(open(path, encoding="utf-8"))
    if "formats" not in tab or not all(k in tab["formats"] for k in FMT):
        raise ValueError(f"{path}：格式表沒有 formats.{{{','.join(FMT)}}}（需要轉接，見 adapt_table）")
    R = np.array([float(tab["formats"][k]["ratio"]) for k in FMT])
    pmax = [tab["formats"][k]["pass_max_ctx"] for k in FMT]
    if not (np.all(np.diff(R) < 0) and R[0] == 1.0):
        raise ValueError(f"格式大小必須 BF16＝1 且嚴格遞減：{R}")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    return dict(R=R, pass_max=pmax, path=path, sha=sha, label=tab.get("label", ""))


def passes(tab, f, ctx):
    """f：格式代碼陣列；ctx：token 數陣列 → 讀取時品質過不過（bool 陣列）。"""
    f = np.asarray(f)
    out = np.zeros(len(f), dtype=bool)
    for code, pm in enumerate(tab["pass_max"]):
        m = f == code
        out[m] = True if pm is None else (np.asarray(ctx)[m] <= pm)
    return out


def pass_min(tab, ctx):
    """每列「品質過關裡最小的格式」。"""
    ctx = np.asarray(ctx)
    best = np.full(len(ctx), BF16)
    for code in range(len(FMT)):
        ok = passes(tab, np.full(len(ctx), code), ctx)
        best = np.where(ok, code, best)       # 代碼越大越小；最後一個過的就是最小的
    return best


def pclass(tab, ctx):
    """門檻依「最小過關格式」分類（INT4 類、8-bit 類…）。"""
    return pass_min(tab, ctx)


# ---------------------------------------------------------------- 成本
def cost(tab, S, tau, M, ctx, f1, f2, lam):
    """每列（一個 D1）的成本，單位位元組。回傳 (total, bytes_part, viol_part, n_viol)。
    C = r(f1)S(1+[τ<M]) + [τ≥M] r(f2)S(1+[τ<∞]) + [τ<∞] λS·1[讀到的格式不過]"""
    R = tab["R"]
    S = np.asarray(S, dtype=float)
    read = np.isfinite(tau)
    before = tau < M
    moved = ~before
    f2 = np.where(moved, f2, f1)
    b = R[f1] * S * (1 + before) + moved * R[f2] * S * (1 + read)
    fread = np.where(before, f1, f2)
    bad = read & ~passes(tab, fread, ctx)
    v = bad * lam * S
    return b + v, b, v, bad


def oracle_actions(tab, tau, ctx):
    pm = pass_min(tab, ctx)
    f = np.where(np.isfinite(tau), pm, INT2)
    return f, f


def apply_policy(tab, ctx, s1, s2, has2, th1, th2):
    """固定介面：D1 s1 ≥ θ1[k] → 最小過關格式，否則 INT2；D2（has2）s2 < θ2[k] → INT2，否則維持 f1。"""
    pm = pass_min(tab, ctx)
    k = pm
    t1 = np.array([th1.get(int(c), -np.inf) for c in k])
    t2 = np.array([th2.get(int(c), -np.inf) for c in k])
    f1 = np.where(s1 >= t1, pm, INT2)
    s2 = np.where(has2, s2, np.inf)
    f2 = np.where(has2 & (s2 < t2), INT2, f1)
    return f1, f2


def _best_split(s, c_low, c_high):
    """分數 < θ 的取 c_low、≥ θ 的取 c_high；回傳讓總和最小的 θ（在相鄰的不同分數之間取中點；±inf 也是候選）。"""
    if len(s) == 0:
        return -np.inf
    o = np.argsort(s, kind="stable")
    ss, lo, hi = s[o], c_low[o], c_high[o]
    cl = np.r_[0.0, np.cumsum(lo)]
    ch = np.r_[0.0, np.cumsum(hi)]
    tot = cl + (ch[-1] - ch)                 # 前 j 個取 low
    n = len(ss)
    ok = np.ones(n + 1, dtype=bool)
    ok[1:n] = ss[1:] > ss[:-1]               # 只能切在不同分數之間
    j = int(np.argmin(np.where(ok, tot, np.inf)))
    if j == 0:
        return -np.inf
    if j == n:
        return np.inf
    return 0.5 * (ss[j - 1] + ss[j])


def tune(tab, S, tau, M, ctx, s1, s2, has2, lam, rounds=2):
    """在 valid（tau 是觀察到的近似答案）上挑 θ1、θ2（座標下降）。"""
    R = tab["R"]
    S = np.asarray(S, dtype=float)
    read = np.isfinite(tau)
    pm = pass_min(tab, ctx)
    classes = sorted(set(int(x) for x in pm))
    th1 = {c: -np.inf for c in classes}       # 先假設 D1 全部用最小過關格式
    th2 = {}
    for _ in range(rounds):
        f1, _ = apply_policy(tab, ctx, s1, s2, has2, th1, {c: -np.inf for c in classes})
        # D2：只在 f1 不是 INT2、且有 D2 的列上挑
        for c in classes:
            m = has2 & (pm == c) & (f1 != INT2)
            ra = read[m]
            c_int2 = R[INT2] * S[m] * (1 + ra) + ra * lam * S[m]
            c_keep = R[f1[m]] * S[m] * (1 + ra) + ra * lam * S[m] * ~passes(tab, f1[m], ctx[m])
            th2[c] = _best_split(s2[m], c_int2, c_keep)
        # D1：給定 θ2，比「最小過關」與「INT2」兩種的整列成本
        fp = pm
        f2p = np.where(has2 & (np.where(has2, s2, np.inf) < np.array([th2.get(int(c), -np.inf) for c in pm])), INT2, fp)
        c_pass = cost(tab, S, tau, M, ctx, fp, f2p, lam)[0]
        c_low = cost(tab, S, tau, M, ctx, np.full(len(S), INT2), np.full(len(S), INT2), lam)[0]
        for c in classes:
            m = pm == c
            th1[c] = _best_split(s1[m], c_low[m], c_pass[m])
    return th1, th2


# ---------------------------------------------------------------- 診斷
def auc(y, s):
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cindex(tau, s):
    """Harrell C：分數越大越早回來；不回來＝設限在很大的時間。"""
    from lifelines.utils import concordance_index
    read = np.isfinite(tau)
    t = np.where(read, tau, 1e12)
    if read.sum() < 2:
        return np.nan
    return float(concordance_index(t, -np.asarray(s, dtype=float), read))


# ---------------------------------------------------------------- 計時與大小
def time_predict(fn, X, n1=300, nb=20, bs=1024):
    """fn(X2d) → 預測。回傳 (batch=1 的中位數 µs, batch=bs 的每列中位數 µs)。單執行緒由呼叫者設定。"""
    X = np.ascontiguousarray(X, dtype=np.float32 if X.dtype == np.float32 else np.float64)
    x1 = X[:1]
    for _ in range(20):
        fn(x1)
    ts = []
    for i in range(n1):
        xi = X[i % len(X):i % len(X) + 1]
        t0 = time.perf_counter_ns()
        fn(xi)
        ts.append(time.perf_counter_ns() - t0)
    xb = X[:bs] if len(X) >= bs else np.resize(X, (bs, X.shape[1]))
    fn(xb)
    tb = []
    for _ in range(nb):
        t0 = time.perf_counter_ns()
        fn(xb)
        tb.append((time.perf_counter_ns() - t0) / len(xb))
    return float(np.median(ts) / 1000.0), float(np.median(tb) / 1000.0)


def nbytes_pickle(obj):
    try:
        return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        return -1


def nbytes_torch(model):
    import torch
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell()
