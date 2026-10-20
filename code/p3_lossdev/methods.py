"""methods.py — 各種 loss／模型族／大小：在 train 上訓練，輸出 valid、test 上 D1、D2 的「會被讀」分數（越大越可能被讀）。
**合成資料、方法開發，不是結論。** 介面固定：分數 → common.tune 挑門檻 → common.apply_policy（07 §2.2）。
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.special import ndtr

import common as C
from synth_ext import generate, observed

warnings.filterwarnings("ignore")
NT = int(os.environ.get("P3LD_THREADS", "2"))
H6 = 6 * 3600.0
DAY = 86400.0
EDGES = np.array([0, 10, 60, 300, 1800, 7200, 21600, 86400], dtype=float)   # 離散 hazard 的格（秒）
LAVA_AGES = np.array([0, 60, 300, 1800, 7200, 14400], dtype=float)


# ============================================================ 資料
class Cell:
    def __init__(self, rho, pi, seed):
        self.rho, self.pi, self.seed = rho, pi, seed
        kw = {"w": float(os.environ["P3LD_W"])} if os.environ.get("P3LD_W") else {}   # 事後加的 w 軸（07 §2.10）
        d, meta = generate(rho, pi, seed, **kw)
        self.d, self.meta = d, meta
        o1, o2 = observed(d, "D1"), observed(d, "D2")
        self.o1, self.o2 = o1, o2
        self.train_window = 0.6 * meta["params"]["horizon_s"]
        self.tr1 = o1[(o1["split"] == "train") & o1["usable"]]
        self.tr2 = o2[(o2["split"] == "train") & o2["usable"]]
        v1 = o1[o1["split"] == "valid"]
        res = (v1["e_obs"] == 1) | (v1["t_obs"] >= C.GUARD_S)
        self.va1 = v1[res].copy()
        self.va1["tau_hat"] = np.where(self.va1["e_obs"] == 1, self.va1["t_obs"], np.inf)
        self.va2 = o2.loc[o2.index.isin(self.va1.index[self.va1["tau_hat"] > self.va1["M"]])]
        self.te1 = o1[o1["split"] == "test"]
        self.te2 = o2[o2["split"] == "test"]
        h = pd.util.hash_pandas_object(d[["sess", "turn_idx", "t_end", "tau", "M", "ctx", "meta_score"]], index=False).values
        import hashlib
        self.data_sha = hashlib.sha256(h.tobytes()).hexdigest()[:16]
        self._seq = None

    def align2(self, d1, d2, s2):
        """把 D2 的分數對齊到 D1 的列（沒有 D2 的列是 NaN）。"""
        out = pd.Series(np.nan, index=d1.index)
        out.loc[d2.index] = s2
        return out.to_numpy()

    # GRU 用：每個 D1 列「之前的輪」的序列（最多 L 輪）
    def seq(self, L=16):
        if self._seq is not None:
            return self._seq
        d = self.d
        g = d.sort_values(["sess", "turn_idx"])
        gap = np.log10(np.maximum(g["tau"].to_numpy(), 1e-3))
        cls = g["cls"].to_numpy()
        per = np.stack([np.where(cls == 2, 0.0, gap) / 3.0, (cls == 0).astype(float), (cls == 1).astype(float),
                        g["meta_score"].to_numpy(), g["meta_flag"].to_numpy().astype(float), g["meta_cat"].to_numpy() / 7.0,
                        g["ctx_log"].to_numpy() / 5.0, g["out_log"].to_numpy() / 3.0], axis=1).astype(np.float32)
        X = np.zeros((len(d), L, per.shape[1]), dtype=np.float32)
        pos = {ix: i for i, ix in enumerate(d.index)}
        sess = g["sess"].to_numpy()
        idx = g.index.to_numpy()
        start = 0
        n = len(g)
        while start < n:
            end = start
            while end < n and sess[end] == sess[start]:
                end += 1
            for k in range(start + 1, end):
                lo = max(start, k - L)
                X[pos[idx[k]], L - (k - lo):] = per[lo:k]
            start = end
        self._seq = pd.Series(list(range(len(d))), index=d.index), X
        return self._seq


def Xof(df, feats):
    return df[feats].to_numpy(dtype=float)


class Prep:
    """LR／MLP／DeepSurv 用：標準化、NaN→0、加缺值指標。"""
    def __init__(self, X):
        self.nanc = np.where(np.isnan(X).any(axis=0))[0]
        self.mu = np.nanmean(X, axis=0)
        self.sd = np.nanstd(X, axis=0)
        self.mu = np.where(np.isnan(self.mu), 0.0, self.mu)
        self.sd = np.where((self.sd > 0) & ~np.isnan(self.sd), self.sd, 1.0)

    def __call__(self, X):
        Z = (X - self.mu) / self.sd
        ind = np.isnan(X[:, self.nanc]).astype(float)
        Z = np.where(np.isnan(Z), 0.0, Z)
        return np.hstack([Z, ind]).astype(np.float32)


# ============================================================ 標籤
def lab_bce(df, Hs):
    t, e = df["t_obs"].to_numpy(), df["e_obs"].to_numpy()
    y = ((e == 1) & (t <= Hs)).astype(int)
    keep = (y == 1) | (t >= Hs)
    return y, keep


def lab_logt_imp(df, window):
    t, e = df["t_obs"].to_numpy(), df["e_obs"].to_numpy()
    return np.where(e == 1, np.log10(np.maximum(t, 1e-3)), np.log10(2 * window))   # LRB：視窗外一律 2×視窗（刻意同分）


# ============================================================ 模型族
def lgbm_params(seed, n=200, leaves=63, depth=-1, lr=0.05):
    return dict(n_estimators=n, num_leaves=leaves, max_depth=depth, learning_rate=lr, min_child_samples=50,
                n_jobs=NT, random_state=seed, verbose=-1)


def fit_lgbm_clf(X, y, seed, w=None, **kw):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(**lgbm_params(seed, **kw))
    m.fit(X, y, sample_weight=w)
    return m, (lambda Z: m.booster_.predict(Z, num_threads=NT))


def fit_lgbm_reg(X, y, seed, objective="regression", alpha=None, **kw):
    import lightgbm as lgb
    p = lgbm_params(seed, **kw)
    p["objective"] = objective
    if alpha is not None:
        p["alpha"] = alpha
    m = lgb.LGBMRegressor(**p)
    m.fit(X, y)
    return m, (lambda Z: m.booster_.predict(Z, num_threads=NT))


def fit_xgb_clf(X, y, seed, n=200, depth=6, lr=0.05, w=None):
    import xgboost as xgb
    dm = xgb.DMatrix(X, label=y, weight=w)
    b = xgb.train(dict(objective="binary:logistic", max_depth=depth, eta=lr, tree_method="hist", nthread=NT, seed=seed,
                       min_child_weight=5, verbosity=0), dm, num_boost_round=n)
    return b, (lambda Z: b.inplace_predict(Z))


def fit_xgb_surv(X, t, e, seed, kind, n=200, depth=6, lr=0.05):
    import xgboost as xgb
    t = np.maximum(t, 1e-3)
    if kind == "cox":
        dm = xgb.DMatrix(X, label=np.where(e == 1, t, -t))
        p = dict(objective="survival:cox", eval_metric="cox-nloglik")
    else:
        dm = xgb.DMatrix(X)
        dm.set_float_info("label_lower_bound", t)
        dm.set_float_info("label_upper_bound", np.where(e == 1, t, np.inf))
        p = dict(objective="survival:aft", eval_metric="aft-nloglik", aft_loss_distribution="normal",
                 aft_loss_distribution_scale=1.0)
    p.update(max_depth=depth, eta=lr, tree_method="hist", nthread=NT, seed=seed, min_child_weight=5, verbosity=0)
    b = xgb.train(p, dm, num_boost_round=n)
    return b


def fit_lr(X, y, seed):
    from sklearn.linear_model import LogisticRegression
    pr = Prep(X)
    m = LogisticRegression(C=1.0, max_iter=1000)
    m.fit(pr(X), y)
    return (pr, m), (lambda Z: m.predict_proba(pr(Z))[:, 1])


def _torch():
    import torch
    torch.set_num_threads(NT)
    return torch


def fit_mlp(X, y, seed, hidden=(64, 64), epochs=15, w=None):
    torch = _torch()
    torch.manual_seed(seed)
    pr = Prep(X)
    Z = torch.tensor(pr(X))
    Y = torch.tensor(y, dtype=torch.float32)
    W = torch.tensor(w if w is not None else np.ones(len(y)), dtype=torch.float32)
    layers, d = [], Z.shape[1]
    for h in hidden:
        layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]
        d = h
    layers += [torch.nn.Linear(d, 1)]
    net = torch.nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(Z), generator=g)
        for i in range(0, len(Z), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            lo = torch.nn.functional.binary_cross_entropy_with_logits(net(Z[b]).squeeze(1), Y[b], weight=W[b])
            lo.backward()
            opt.step()
    net.eval()

    def pred(Q):
        with torch.no_grad():
            return torch.sigmoid(net(torch.from_numpy(pr(Q))).squeeze(1)).numpy()
    return (pr, net), pred


def fit_gru(seqX, curX, y, seed, hidden=32, epochs=10):
    torch = _torch()
    torch.manual_seed(seed)
    pr = Prep(curX)

    class Net(torch.nn.Module):
        def __init__(s, din, dcur):
            super().__init__()
            s.gru = torch.nn.GRU(din, hidden, batch_first=True)
            s.head = torch.nn.Sequential(torch.nn.Linear(hidden + dcur, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))

        def forward(s, q, c):
            _, h = s.gru(q)
            return s.head(torch.cat([h[-1], c], dim=1)).squeeze(1)
    Q = torch.tensor(seqX)
    Cc = torch.tensor(pr(curX))
    Y = torch.tensor(y, dtype=torch.float32)
    net = Net(Q.shape[2], Cc.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(Q), generator=g)
        for i in range(0, len(Q), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            lo = torch.nn.functional.binary_cross_entropy_with_logits(net(Q[b], Cc[b]), Y[b])
            lo.backward()
            opt.step()
    net.eval()

    def pred(sq, cu):
        with torch.no_grad():
            return torch.sigmoid(net(torch.from_numpy(sq), torch.from_numpy(pr(cu)))).numpy()
    return net, pred


def fit_rsf(X, t, e, seed, n=100, max_rows=6000):
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    rng = np.random.default_rng(seed)
    if len(X) > max_rows:
        ix = rng.choice(len(X), max_rows, replace=False)
        X, t, e = X[ix], t[ix], e[ix]
    Xf = np.where(np.isnan(X), -1.0, X)
    m = RandomSurvivalForest(n_estimators=n, min_samples_leaf=30, max_features="sqrt", n_jobs=NT, random_state=seed)
    m.fit(Xf, Surv.from_arrays(event=e.astype(bool), time=np.maximum(t, 1e-3)))
    times = m.unique_times_
    j = int(np.searchsorted(times, DAY, side="right") - 1)
    j = max(0, min(j, len(times) - 1))

    def pred(Z):
        sf = m.predict_survival_function(np.where(np.isnan(Z), -1.0, Z), return_array=True)
        return 1.0 - sf[:, j]
    return m, pred


def fit_deepsurv(X, t, e, seed, hidden=(64, 64), epochs=20):
    torch = _torch()
    import torchtuples as tt
    from pycox.models import CoxPH
    np.random.seed(seed)
    torch.manual_seed(seed)
    pr = Prep(X)
    Z = pr(X)
    net = tt.practical.MLPVanilla(Z.shape[1], list(hidden), 1, batch_norm=False, dropout=0.0, output_bias=False)
    model = CoxPH(net, tt.optim.Adam(1e-3))
    model.fit(Z, (np.maximum(t, 1e-3).astype(np.float32), e.astype(np.float32)), batch_size=256, epochs=epochs, verbose=False)

    def pred(Q):
        return model.predict(pr(Q)).reshape(-1)
    return net, pred


# ============================================================ 由各 loss 產生分數
def _per_dp(cell, feats, fit_fn):
    """D1、D2 各訓一個模型；fit_fn(train_df, dp) → predict(df)。回傳 (s1_va, s2_va, s1_te, s2_te, models)。"""
    p1 = fit_fn(cell.tr1, "D1")
    p2 = fit_fn(cell.tr2, "D2")
    s1v, s1t = p1(cell.va1), p1(cell.te1)
    s2v = cell.align2(cell.va1, cell.va2, p2(cell.va2)) if len(cell.va2) else np.full(len(cell.va1), np.nan)
    s2t = cell.align2(cell.te1, cell.te2, p2(cell.te2)) if len(cell.te2) else np.full(len(cell.te1), np.nan)
    return s1v, s2v, s1t, s2t


def hazard_expand(df, feats):
    t, e = df["t_obs"].to_numpy(), df["e_obs"].to_numpy()
    X = Xof(df, feats)
    nb = len(EDGES) - 1
    rows, bins, ys = [], [], []
    for j in range(nb):
        lo, hi = EDGES[j], EDGES[j + 1]
        ent = t > lo                                    # 進入這一格（還沒回來）
        ev = ent & (e == 1) & (t <= hi)
        surv_full = ent & (t > hi)                      # 整格都觀察到、沒回來（含 e=1 但在後面回來）
        m = ev | surv_full
        rows.append(np.nonzero(m)[0])
        bins.append(np.full(int(m.sum()), j))
        ys.append(ev[m].astype(int))
    r = np.concatenate(rows)
    return np.hstack([X[r], np.concatenate(bins)[:, None]]), np.concatenate(ys)


def hazard_predict(pred, X, a=None):
    """P(在 24 小時內回來 | 已經過 a)。a=None：從 0 算。"""
    nb = len(EDGES) - 1
    H = np.stack([pred(np.hstack([X, np.full((len(X), 1), j)])) for j in range(nb)], axis=1)
    H = np.clip(H, 1e-9, 1 - 1e-9)
    if a is None:
        return 1 - np.prod(1 - H, axis=1)
    a = np.asarray(a, dtype=float)
    S = np.ones(len(X))
    for j in range(nb):
        lo, hi = EDGES[j], EDGES[j + 1]
        frac = np.clip((hi - np.maximum(a, lo)) / (hi - lo), 0, 1)
        S *= (1 - H[:, j]) ** frac
    return 1 - S


def scores_for(cell, spec):
    """spec：dict(loss=..., model=..., size=..., feats=[...])。回傳 dict(s1v, s2v, s1t, s2t, prob, fixed_thr, models_d1)。"""
    loss = spec["loss"]
    feats = spec.get("feats", C.ALL_FEATS)
    model = spec.get("model", "lgbm")
    sz = spec.get("size", {})
    seed = cell.seed
    out = dict(prob=False, fixed_thr=None, timing_model=None)
    t0 = time.time()

    if loss.startswith("bce") and model in ("lgbm", "xgb", "lr", "mlp"):
        Hs = spec.get("H", H6)
        lam = spec.get("lam_w", C.LAMBDA_MAIN)
        weighted = loss == "bce_w"
        tm = {}

        def fit(df, dp):
            y, keep = lab_bce(df, Hs)
            X = Xof(df, feats)[keep]
            y = y[keep]
            w = None
            if weighted:
                tab = cell.table
                pm = C.pass_min(tab, df["ctx"].to_numpy()[keep])
                S = df["S"].to_numpy()[keep]
                dr = tab["R"][pm] - tab["R"][C.INT2]
                w = np.where(y == 1, lam * S, (2.0 if dp == "D1" else 1.0) * dr * S)
                w = w / w.mean()
            if model == "lgbm":
                m, p = fit_lgbm_clf(X, y, seed, w=w, **sz)
            elif model == "xgb":
                m, p = fit_xgb_clf(X, y, seed, w=w, **sz)
            elif model == "lr":
                m, p = fit_lr(X, y, seed)
            else:
                m, p = fit_mlp(X, y, seed, w=w, **sz)
            tm[dp] = (m, p)
            return lambda q: p(Xof(q, feats))
        out["s1v"], out["s2v"], out["s1t"], out["s2t"] = _per_dp(cell, feats, fit)
        out["prob"] = True
        out["timing_model"] = tm["D1"]
        if spec.get("fixed_thr") is not None:
            out["fixed_thr"] = spec["fixed_thr"]

    elif loss == "logt_imp":
        def fit(df, dp):
            m, p = fit_lgbm_reg(Xof(df, feats), lab_logt_imp(df, cell.train_window), seed, **sz)
            return lambda q: -p(Xof(q, feats))
        out["s1v"], out["s2v"], out["s1t"], out["s2t"] = _per_dp(cell, feats, fit)

    elif loss == "quantile":
        qs = np.arange(1, 10) / 10.0
        ystar = np.log10(DAY)

        def fit(df, dp):
            X, y = Xof(df, feats), lab_logt_imp(df, cell.train_window)
            ps = [fit_lgbm_reg(X, y, seed, objective="quantile", alpha=q, n=100)[1] for q in qs]

            def pr(q):
                Z = Xof(q, feats)
                P = np.sort(np.stack([p(Z) for p in ps], axis=1), axis=1)
                F = np.array([np.interp(ystar, P[i], qs, left=0.05, right=0.95) for i in range(len(P))])
                return F - 1e-4 * P[:, 4]
            return pr
        out["s1v"], out["s2v"], out["s1t"], out["s2t"] = _per_dp(cell, feats, fit)

    elif loss in ("hazard", "hazard_cond"):
        def fit(df, dp):
            X, y = hazard_expand(df, feats)
            m, p = fit_lgbm_clf(X, y, seed, **sz)
            return p
        p1 = fit(cell.tr1, "D1")
        s1v = hazard_predict(p1, Xof(cell.va1, feats))
        s1t = hazard_predict(p1, Xof(cell.te1, feats))
        if loss == "hazard":
            p2 = fit(cell.tr2, "D2")
            s2v_ = hazard_predict(p2, Xof(cell.va2, feats))
            s2t_ = hazard_predict(p2, Xof(cell.te2, feats))
        else:   # D1 訓練一次，D2 用條件化：已閒置 a 的列，把 F0 設成 0（和 D1 訓練時一樣），從 a 之後的格算
            def cond(df):
                Xq = Xof(df, feats)
                if "a_log" in feats:
                    Xq[:, feats.index("a_log")] = 0.0
                return hazard_predict(p1, Xq, a=df["a"].to_numpy())
            s2v_, s2t_ = cond(cell.va2), cond(cell.te2)
        out.update(s1v=s1v, s1t=s1t, s2v=cell.align2(cell.va1, cell.va2, s2v_), s2t=cell.align2(cell.te1, cell.te2, s2t_), prob=True)

    elif loss in ("cox", "aft", "aft_cond"):
        import xgboost as xgb
        kind = "cox" if loss == "cox" else "aft"

        def fit(df):
            return fit_xgb_surv(Xof(df, feats), df["t_obs"].to_numpy(), df["e_obs"].to_numpy(), seed, kind, **sz)

        def score(b, df, a=None):
            Xq = Xof(df, feats)
            if a is not None and "a_log" in feats:
                Xq[:, feats.index("a_log")] = 0.0
            pr = b.predict(xgb.DMatrix(Xq), output_margin=True)
            if kind == "cox":
                return pr                                   # log-risk：越大越早
            mu = pr                                         # AFT：log 時間的位置參數
            z24 = (np.log(DAY) - mu) / 1.0
            if a is None:
                return ndtr(z24)
            za = (np.log(np.maximum(a, 1e-3)) - mu) / 1.0
            return (ndtr(z24) - ndtr(za)) / np.maximum(1 - ndtr(za), 1e-12)
        b1 = fit(cell.tr1)
        s1v, s1t = score(b1, cell.va1), score(b1, cell.te1)
        if loss == "aft_cond":
            s2v_, s2t_ = score(b1, cell.va2, a=cell.va2["a"].to_numpy()), score(b1, cell.te2, a=cell.te2["a"].to_numpy())
        else:
            b2 = fit(cell.tr2)
            s2v_, s2t_ = score(b2, cell.va2), score(b2, cell.te2)
        out.update(s1v=s1v, s1t=s1t, s2v=cell.align2(cell.va1, cell.va2, s2v_), s2t=cell.align2(cell.te1, cell.te2, s2t_),
                   prob=(kind == "aft"))

    elif loss == "lava":
        df = cell.tr1
        t, e = df["t_obs"].to_numpy(), df["e_obs"].to_numpy()
        X0 = Xof(df, feats)
        Xs, ys = [], []
        for age in LAVA_AGES:
            alive = t > age
            y = ((e == 1) & (t - age <= H6)).astype(int)
            keep = alive & ((y == 1) | (t - age >= H6))
            Xa = X0[keep].copy()
            if "a_log" in feats:
                Xa[:, feats.index("a_log")] = np.log10(1 + age)
            Xs.append(Xa)
            ys.append(y[keep])
        m, p = fit_lgbm_clf(np.vstack(Xs), np.concatenate(ys), seed, **sz)
        pr = lambda q: p(Xof(q, feats))
        out.update(s1v=pr(cell.va1), s1t=pr(cell.te1), s2v=cell.align2(cell.va1, cell.va2, pr(cell.va2)),
                   s2t=cell.align2(cell.te1, cell.te2, pr(cell.te2)), prob=True, timing_model=(m, p))

    elif loss == "lambdarank":
        import lightgbm as lgb

        def fit(df, dp):
            t, e = df["t_obs"].to_numpy(), df["e_obs"].to_numpy()
            rel = np.where((e == 1) & (t <= 60), 3, np.where((e == 1) & (t <= 3600), 2, np.where((e == 1) & (t <= H6), 1, 0)))
            keep = (e == 1) & (t <= H6) | (t >= H6)
            sub = df[keep].assign(rel=rel[keep], grp=(df.loc[keep, "t_dec"] // 3600).astype(np.int64)).sort_values("grp", kind="stable")
            grp = sub.groupby("grp", sort=False).size().to_numpy()
            p = lgbm_params(seed, **sz)
            m = lgb.LGBMRanker(objective="lambdarank", **p)
            m.fit(Xof(sub, feats), sub["rel"].to_numpy(), group=grp)
            return lambda q: m.booster_.predict(Xof(q, feats), num_threads=NT)
        out["s1v"], out["s2v"], out["s1t"], out["s2t"] = _per_dp(cell, feats, fit)

    elif loss == "bce" and model == "gru":
        pos, SQ = cell.seq()
        Hs = H6
        cur = [f for f in feats if f in C.F0 + C.F1]

        def fit(df, dp):
            y, keep = lab_bce(df, Hs)
            ii = pos.loc[df.index[keep]].to_numpy()
            net, p = fit_gru(SQ[ii], Xof(df, cur)[keep], y[keep], seed)
            return lambda q: p(SQ[pos.loc[q.index].to_numpy()], Xof(q, cur))
        out["s1v"], out["s2v"], out["s1t"], out["s2t"] = _per_dp(cell, feats, fit)
        out["prob"] = True

    elif loss == "surv" and model in ("rsf", "deepsurv"):
        def fit(df, dp):
            X, t, e = Xof(df, feats), df["t_obs"].to_numpy(), df["e_obs"].to_numpy()
            if model == "rsf":
                m, p = fit_rsf(X, t, e, seed)
            else:
                m, p = fit_deepsurv(X, t, e, seed)
            return lambda q: p(Xof(q, feats))
        out["s1v"], out["s2v"], out["s1t"], out["s2t"] = _per_dp(cell, feats, fit)
        out["prob"] = model == "rsf"
    else:
        raise ValueError(f"未知的 spec：{spec}")
    out["fit_s"] = time.time() - t0
    return out
