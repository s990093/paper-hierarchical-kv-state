#!/usr/bin/env python3
"""t1.py — ICPE T1（docs/h200/icpe/T1_h200_cpu_tier.md）：CPU 層換成 H200 自己量的常數。

  python code/icpe/t1.py build      §1：H200 的 C0＋duplex 照 code/m7_params.py 的規則 → results/icpe_h200/t1/tier_params_h200cpu.json
  python code/icpe/t1.py selftest   §2 的四項自檢（各開一個子行程：沒設／設了 P3_TIER=h200cpu），不過 → exit 1
  python code/icpe/t1.py dump       （selftest 內部用）印出這個環境下各模擬讀到的常數與輸出路徑

local（SSD 級）、nfs（NFS 級）逐位元照抄 MI300X 的 tier_params.json（使用者選擇）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
REPO = os.path.dirname(CODE)
T1 = os.path.join(REPO, "results", "icpe_h200", "t1")
NEW = os.path.join(T1, "tier_params_h200cpu.json")
OLD = os.path.join(REPO, "results", "m7_write_policy_mi300x", "tier_params.json")
H7 = os.path.join(REPO, "results", "m7_write_policy_h200")
GiB, MiB = 1 << 30, 1 << 20
# 登記 §1 寫下的值（build 要重現它們，對不上就停）
PREREG = {"read_GiBps": 43.5888, "write_GiBps": 41.1687, "k_read": 1.002}


def run_id():
    return os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ============================================================================ build（§1）
def cmd_build(a):
    import pandas as pd
    old = json.load(open(OLD))
    c0 = pd.read_csv(os.path.join(H7, "calib_c0.csv"))
    c0 = c0[c0.kind == "bw"].copy()
    c0["v"] = c0.value.astype(float)
    c0 = c0[c0.rep.astype(float) > 0]                       # m7_params.py：C0 排除每組第 0 次暖機
    runs = sorted(set(c0.run_id))
    assert runs == ["20261016-061254-m7h-c0"], runs

    def mn(item):
        x = c0[c0.item == item].v
        assert len(x) == 14, (item, len(x))
        return float(x.min())
    dp = pd.read_csv(os.path.join(H7, "calib_c0_duplex.csv"))
    assert sorted(set(dp.run_id)) == ["20261016-061303-m7h-duplex"], set(dp.run_id)
    al, wd = dp[dp.case == "alone"].h2d_GiBps, dp[dp.case == "with_d2h"].h2d_GiBps
    assert len(al) == 5 and len(wd) == 5, (len(al), len(wd))
    k = round(float(al.median()) / float(wd.median()), 3)  # MI300X 檔的 k_read 也是小數 3 位
    r, w = 0.9 * mn("H2D_pinned_chunk_into_kv_view"), 0.9 * mn("D2H_pinned_chunk_from_kv_view")
    cpu = {"read_GiBps": r, "write_GiBps": w, "c_ms": 0.0, "ell_chunk_ms": 64 * MiB / (r * GiB) * 1e3, "k_read": k}
    for key, want in PREREG.items():
        assert abs(cpu[key] - want) < 5e-5, (key, cpu[key], want)
    P = json.loads(json.dumps(old))                          # 深拷貝；local、nfs 逐位元照抄
    P["cpu"] = cpu
    P["source_runs"] = sorted(set(old["source_runs"]) | set(runs) | {"20261016-061303-m7h-duplex"})
    P["rule"] = old["rule"] + "；cpu＝H200（T1）"
    P["k_read_source"] = ("cpu: H200 results/m7_write_policy_h200/calib_c0_duplex.csv median alone/with_d2h (run 20261016-061303-m7h-duplex), "
                          "rounded to 3 decimals; local/nfs: copied from the MI300X tier_params.json (" + old["k_read_source"] + ")")
    P["t1"] = dict(prereg="docs/h200/icpe/T1_h200_cpu_tier.md", run_id=run_id(), ts=ts(), derived_from=os.path.relpath(OLD, REPO),
                   cpu_source=dict(c0="20261016-061254-m7h-c0", duplex="20261016-061303-m7h-duplex",
                                   h2d_min_GiBps=mn("H2D_pinned_chunk_into_kv_view"), d2h_min_GiBps=mn("D2H_pinned_chunk_from_kv_view"),
                                   duplex_alone_median=float(al.median()), duplex_with_d2h_median=float(wd.median())),
                   unchanged=["local", "nfs", "derate"], mi300x_cpu=old["cpu"])
    os.makedirs(T1, exist_ok=True)
    json.dump(P, open(NEW, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(dict(cpu=cpu, out=os.path.relpath(NEW, REPO)), ensure_ascii=False))


# ============================================================================ dump（selftest 用）
def _canon(x):
    import numpy as np
    if isinstance(x, dict):
        return {str(k): _canon(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_canon(v) for v in x.tolist()]
    if isinstance(x, (np.floating, float)):
        return repr(float(x))
    if isinstance(x, (np.integer, int, bool, str)) or x is None:
        return x if not isinstance(x, np.integer) else int(x)
    return f"<{type(x).__name__}>"


def _h(x):
    return hashlib.sha256(json.dumps(_canon(x), sort_keys=True).encode()).hexdigest()[:16]


def cmd_dump(a):
    for p in (CODE, os.path.join(CODE, "p3x"), os.path.join(CODE, "icpe"), os.path.join(CODE, "s3")):
        sys.path.insert(0, p)
    import p3_platform as PF
    # x4_sim 要第一個 import（同實際執行：x4_sim 是主程式）：它在 import m7_* 之前把 M7_TIER_PARAMS 設成 PF.TIER_PARAMS；
    # m7_write_policy 的 TIER_PARAMS 在它自己 import 時就定了，x3_restore／x1_sim 先 import 的話會拿到 env_h200.sh 的值
    import x4_sim as X4
    import m7_write_policy as MW
    import x1_sim as XS
    import x3_restore as X3
    import x5_costs as XC
    import e2e as E
    K = XS.constants()
    out = dict(tier=PF.TIER, tier_params=os.path.relpath(PF.TIER_PARAMS, REPO))
    out["x1"] = dict(cpu=_canon(K["cpu"]), rest=_h({k: v for k, v in K.items() if k != "cpu"}))
    x3 = {m: X3.consts(m) for m in ("llama", "qwen")}
    out["x3"] = dict(cpu_Bps={m: repr(x3[m]["cpu_Bps"]) for m in x3},
                     rest=_h({m: {k: v for k, v in x3[m].items() if k != "cpu_Bps"} for m in x3}), tiers=_h(X3.tiers_gibps()))
    P4 = MW.load_params()
    out["x4"] = dict(cpu=_canon(P4["cpu"]), local=_h(P4["local"]), nfs=_h(P4["nfs"]),
                     tier_env=os.path.relpath(os.environ["M7_TIER_PARAMS"], REPO))
    x5 = {}
    for m in ("llama", "qwen"):
        c = XC.Costs(m)
        for st in ("BF16", "INT4"):
            for g in XC.GPUS:
                for dq in XC.DQS:
                    try:
                        c._ratio(st, 64, g, dq)
                    except AssertionError:
                        pass
        x5[m] = _h({k: v for k, v in vars(c).items()})
    out["x5"] = x5
    EK = {m: E.constants(m) for m in ("llama", "qwen")}
    out["e2e"] = dict(cpu={m: _canon(EK[m]["cpu"]) for m in EK}, rest=_h({m: {k: v for k, v in EK[m].items() if k != "cpu"} for m in EK}))
    out["paths"] = {k: os.path.relpath(v, REPO) for k, v in
                    dict(x1=XS.OUTDIR, x3=X3.OUT, x4=X4.OUT, e2e=E.OUT, explore_x5=PF.explore_out("x5")).items()}
    print(json.dumps(out, ensure_ascii=False, sort_keys=True))


# ============================================================================ selftest（§2）
def cmd_selftest(a):
    def dump(tier):
        env = dict(os.environ)
        env.pop("P3_TIER", None)
        if tier:
            env["P3_TIER"] = tier
        env.setdefault("P3_CONST", "h200")
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "dump"], env=env, capture_output=True, text=True, cwd=REPO)
        if r.returncode != 0:
            print(r.stderr[-3000:], file=sys.stderr)
            raise SystemExit(f"dump（P3_TIER={tier!r}）失敗")
        return json.loads(r.stdout.strip().splitlines()[-1])
    A, B = dump(""), dump("h200cpu")
    old, new = json.load(open(OLD)), json.load(open(NEW))
    res = []

    def chk(name, cond, info=None):
        res.append(dict(check=name, ok=bool(cond), info=info))
        print(("PASS " if cond else "FAIL ") + name + ("" if info is None else f"  {info}"), flush=True)
    rw = lambda c: [repr(float(c["read_GiBps"])), repr(float(c["write_GiBps"])), repr(float(c["k_read"]))]   # noqa: E731
    x1cpu = lambda d: [d["x1"]["cpu"][k] for k in ("r", "w", "k")]                                       # noqa: E731
    # 1. 沒設開關：讀 MI300X 的檔、輸出在原本的位置
    chk("1a 沒設：tier_params＝MI300X 的檔", A["tier_params"] == os.path.relpath(OLD, REPO), A["tier_params"])
    chk("1b 沒設：X1 的 cpu＝MI300X 的值", x1cpu(A) == rw(old["cpu"]), x1cpu(A))
    chk("1c 沒設：X4 的 cpu＝MI300X 的值", A["x4"]["cpu"]["read_GiBps"] == repr(float(old["cpu"]["read_GiBps"])), A["x4"]["tier_env"])
    chk("1d 沒設：輸出在原本的位置", A["paths"]["x1"] == "results/p3_explore_h200/x1" and A["paths"]["x3"] == "results/p3_explore_h200/x3"
        and A["paths"]["x4"] == "results/p3_explore_h200/x4" and A["paths"]["e2e"] == "results/s3_h200/e2e", A["paths"])
    # 2. 設了開關：只有 cpu 改變
    chk("2a 設了：X1 的 cpu＝新值", x1cpu(B) == rw(new["cpu"]), x1cpu(B))
    chk("2b 設了：X1 其他常數不變", A["x1"]["rest"] == B["x1"]["rest"])
    chk("2c 設了：X3 的 cpu＝新值、其他不變", B["x3"]["cpu_Bps"]["llama"] == repr(new["cpu"]["read_GiBps"] * GiB)
        and A["x3"]["rest"] == B["x3"]["rest"] and A["x3"]["tiers"] == B["x3"]["tiers"], B["x3"]["cpu_Bps"])
    chk("2d 設了：X4 的 cpu＝新值、local／nfs 不變", B["x4"]["cpu"]["read_GiBps"] == repr(float(new["cpu"]["read_GiBps"]))
        and A["x4"]["local"] == B["x4"]["local"] and A["x4"]["nfs"] == B["x4"]["nfs"], B["x4"]["tier_env"])
    chk("2e 設了：E2E 的 cpu＝新值、其他不變", all([B["e2e"]["cpu"][m][k] for k in ("r", "w", "k")] == rw(new["cpu"]) for m in ("llama", "qwen"))
        and A["e2e"]["rest"] == B["e2e"]["rest"])
    chk("2f 新檔的 local、nfs、derate 和 MI300X 的逐位元相同", all(old[k] == new[k] for k in ("local", "nfs", "derate")))
    # 3. X5 不受影響
    chk("3 X5 的成本常數設不設開關逐值相同", A["x5"] == B["x5"], B["x5"])
    # 4. 輸出都在 t1 底下
    chk("4 設了：輸出都在 results/icpe_h200/t1/ 底下", all(v.startswith("results/icpe_h200/t1/") for v in B["paths"].values()), B["paths"])
    ok = all(r["ok"] for r in res)
    os.makedirs(T1, exist_ok=True)
    json.dump(dict(run_id=run_id(), ts=ts(), ok=ok, checks=res, dump_off=A, dump_on=B), open(os.path.join(T1, "selftest.json"), "w"),
              indent=1, ensure_ascii=False)
    print("SELFTEST", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


# ============================================================================ report（§5）
R = lambda *p: os.path.join(REPO, "results", *p)                     # noqa: E731
META = {"run_id", "ts", "src_runs", "src", "src_run", "src_sweep_run", "src_cfg"}
K_X1 = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore", "component"]
K_X4 = ["trace", "rule", "p", "cpu_GiB", "slow", "fscale"]
K_FS = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore"]
# （舊檔, 新檔, 鍵）：舊＝MI300X 的 CPU 層（ICPE_EVIDENCE 的值），新＝T1
PAIRS = []
for _V in ("V0", "V1", "V2"):
    for _p in ("x1", "x1h1800", "x1h3600"):
        PAIRS += [(R("icpe_h200", "p01", f"sim_{_V}", "x1", f"{_p}_verdict.csv"), os.path.join(T1, "p01", f"sim_{_V}", "x1", f"{_p}_verdict.csv"),
                   ["rule", "component"]),
                  (R("icpe_h200", "p01", f"sim_{_V}", "x1", f"{_p}_headroom_components.csv"),
                   os.path.join(T1, "p01", f"sim_{_V}", "x1", f"{_p}_headroom_components.csv"), K_X1)]
for _p in ("x1fi", "x1fiuser"):
    PAIRS += [(R("icpe_h200", "p12", "x1", f"{_p}_verdict.csv"), os.path.join(T1, "p12", "x1", f"{_p}_verdict.csv"), ["rule", "component"]),
              (R("icpe_h200", "p12", "x1", f"{_p}_headroom_components.csv"), os.path.join(T1, "p12", "x1", f"{_p}_headroom_components.csv"), K_X1)]
PAIRS += [(R("p3_explore_h200", "x4", "x4_verdict_cfg.csv"), os.path.join(T1, "explore", "x4", "x4_verdict_cfg.csv"), K_X4),
          (R("p3_explore_h200", "x4", "x4_verdict_verdicts.csv"), os.path.join(T1, "explore", "x4", "x4_verdict_verdicts.csv"), ["metric", "group"]),
          (R("icpe_h200", "p12", "x4", "p12_queue_cells.csv"), os.path.join(T1, "p12", "x4", "p12_queue_cells.csv"), K_X4),
          (R("icpe_h200", "p02", "fullsim_hint.csv"), os.path.join(T1, "p02", "fullsim_hint.csv"), K_FS),
          (R("p3_explore_h200", "x3", "x3_restore_validate_summary.csv"), os.path.join(T1, "explore", "x3", "x3_restore_validate_summary.csv"),
           ["variant", "scope"])]
JPAIRS = [(R("p3_explore_h200", "x3", "x3_verdict.json"), os.path.join(T1, "explore", "x3", "x3_verdict.json")),
          (R("icpe_h200", "p12", "x4", "p12_queue_verdict.json"), os.path.join(T1, "p12", "x4", "p12_queue_verdict.json")),
          (R("s3_h200", "e2e", "e2e_report.json"), os.path.join(T1, "e2e", "e2e_report.json"))]


def _isnum(s):
    import pandas as pd
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def cmp_csv(old, new, keys):
    import pandas as pd
    rel = lambda p: os.path.relpath(p, REPO)                          # noqa: E731
    if not (os.path.exists(old) and os.path.exists(new)):
        return dict(old=rel(old), new=rel(new), status="NOT_MEASURED", missing=[rel(p) for p in (old, new) if not os.path.exists(p)])
    a, b = pd.read_csv(old), pd.read_csv(new)
    cols = [c for c in a.columns if c not in keys and c not in META and c in b.columns]
    m = a.merge(b, on=keys, how="outer", suffixes=("_old", "_new"), indicator=True).copy()
    out = dict(old=rel(old), new=rel(new), status="ok", keys=keys, n_old=len(a), n_new=len(b), n_both=int((m._merge == "both").sum()),
               only_old=int((m._merge == "left_only").sum()), only_new=int((m._merge == "right_only").sum()),
               src_old=sorted({str(x) for x in a.get("run_id", [])}), src_new=sorted({str(x) for x in b.get("run_id", [])}), cols={})
    m = m[m._merge == "both"]
    for c in cols:
        x, y = m[c + "_old"], m[c + "_new"]
        if _isnum(x) and _isnum(y):
            d = (y.astype(float) - x.astype(float))
            out["cols"][c] = dict(kind="num", max_abs_diff=float(d.abs().max()) if len(d) else 0.0,
                                  n_changed=int((d.abs() > 1e-12).sum()))
        else:
            ch = m[~((x.isna() & y.isna()) | (x.astype(str) == y.astype(str)))]          # 兩邊都空＝沒變
            out["cols"][c] = dict(kind="str", n_changed=len(ch),
                                  changes=[dict({k: (r[k].item() if hasattr(r[k], "item") else r[k]) for k in keys},
                                                old=str(r[c + "_old"]), new=str(r[c + "_new"])) for _, r in ch.iterrows()])
    if "headroom" in cols:                                          # 跨門檻的格子（README 規則 6）：rule＝min（判定用）或沒有 rule＝min 的檔就全部
        mm = m[m.rule == "min"] if "rule" in m.columns and (m.rule == "min").any() else m
        cr = []
        for thr in (0.15, 0.05):
            x = mm[(mm.headroom_old >= thr) != (mm.headroom_new >= thr)]
            cr += [dict({k: (r[k].item() if hasattr(r[k], "item") else r[k]) for k in keys}, threshold=thr,
                        headroom_old=float(r.headroom_old), headroom_new=float(r.headroom_new)) for _, r in x.iterrows()]
        out["crossings"] = cr
    return out


def _leaves(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if k in META:
                continue
            yield from _leaves(v, f"{p}.{k}" if p else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _leaves(v, f"{p}[{i}]")
    else:
        yield p, x


def cmp_json(old, new):
    rel = lambda p: os.path.relpath(p, REPO)                          # noqa: E731
    if not (os.path.exists(old) and os.path.exists(new)):
        return dict(old=rel(old), new=rel(new), status="NOT_MEASURED", missing=[rel(p) for p in (old, new) if not os.path.exists(p)])
    A, B = dict(_leaves(json.load(open(old)))), dict(_leaves(json.load(open(new))))
    ch = []
    for k in sorted(set(A) | set(B)):
        a, b = A.get(k, "<無>"), B.get(k, "<無>")
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
            if a != b:
                ch.append(dict(path=k, old=a, new=b, diff=b - a))
        elif a != b:
            ch.append(dict(path=k, old=a, new=b))
    return dict(old=rel(old), new=rel(new), status="ok", n_leaves=len(set(A) | set(B)), n_changed=len(ch), changes=ch)


def summary(root_p01, root_p12, root_x3, root_q, root_fs, root_e2e):
    """ICPE_EVIDENCE B、C 各列的那幾個數，用同一個定義從一組目錄算（舊目錄算出來要等於 ICPE_EVIDENCE 寫的值）。"""
    import pandas as pd
    S = {}

    def hc(d, p):
        f = os.path.join(d, f"{p}_headroom_components.csv")
        return pd.read_csv(f) if os.path.exists(f) else None
    for V in ("V0", "V1", "V2"):
        d = os.path.join(root_p01, f"sim_{V}", "x1")
        for p in ("x1", "x1h1800", "x1h3600"):
            q = hc(d, p)
            if q is None:
                S[f"{V}.{p}"] = "NOT_MEASURED"
                continue
            q = q[q.rule == "min"]
            S[f"{V}.{p}"] = dict(n_cells=len(q), max_by_component={k: float(v) for k, v in q.groupby("component").headroom.max().items()},
                                 n_ge15_all=int((q.headroom >= 0.15).sum()), n_ge15_odrop=int((q[q.component == "O-DROP"].headroom >= 0.15).sum()))
            vf = os.path.join(d, f"{p}_verdict.csv")
            if os.path.exists(vf):
                v = pd.read_csv(vf)
                w = v[v.rule == "min"].verdict_rule.unique()
                S[f"{V}.{p}"]["word"] = str(w[0]) if len(w) == 1 else f"<{list(w)}>"
            if V == "V0" and p == "x1":
                b1 = q[(q.trace == "conversation") & (q.slow == "ssd") & (q.gpu == "idle") & (q.restore == "cake") & q.cpu_GiB.isin([128, 256])]
                S["B1"] = {f"{r.component}@{int(r.cpu_GiB)}": float(r.headroom) for r in b1.itertuples()}
    for p in ("x1fi", "x1fiuser"):
        q = hc(os.path.join(root_p12, "x1"), p)
        S[f"B3.{p}"] = "NOT_MEASURED" if q is None else dict(n_cells=len(q), max=float(q.headroom.max()))
    xv = os.path.join(root_x3, "x3_verdict.json")
    if os.path.exists(xv):
        x = json.load(open(xv))
        S["C1"] = {k: dict(verdict=x[k].get("verdict"), interesting_cells=x[k].get("interesting_cells"), main_cells=x[k].get("main_cells"),
                           best_time_weighted_hidden=max((c.get("time_weighted_hidden", float("nan")) for c in x[k].get("main_best", [])), default=None),
                           nfs_best=max(((c.get("time_weighted_hidden"), c.get("trace"), c.get("signal")) for c in x[k].get("main_best", [])
                                         if c.get("tier") == "nfs"), default=None))
                   for k in ("v1_original_registration", "v2_with_freeinference") if k in x}
    else:
        S["C1"] = "NOT_MEASURED"
    qv = os.path.join(root_q, "p12_queue_verdict.json")
    S["B4"] = json.load(open(qv)) if os.path.exists(qv) else "NOT_MEASURED"
    if isinstance(S["B4"], dict):
        S["B4"] = {k: S["B4"][k] for k in ("D1", "D2", "report_only_QP") if k in S["B4"]}
    fs = os.path.join(root_fs, "fullsim_hint.csv")
    if os.path.exists(fs):
        f = pd.read_csv(fs)
        S["C2"] = dict(hint_vs_lru=[float(f.hint_vs_lru.min()), float(f.hint_vs_lru.max())], opf_vs_lru=[float(f.opf_vs_lru.min()), float(f.opf_vs_lru.max())],
                       share_median_by_slow={k: float(v) for k, v in f.groupby("slow").hint_share_of_opf.median().items()},
                       n_share_gt1=int((f.hint_share_of_opf > 1).sum()))
    else:
        S["C2"] = "NOT_MEASURED"
    er = os.path.join(root_e2e, "e2e_report.json")
    if os.path.exists(er):
        e = json.load(open(er))
        S["E2E"] = {k: e.get(k) for k in ("verdict", "n_pass", "n_ask_level", "n_not_measured", "pass_per_model", "m_star")}
    else:
        S["E2E"] = "NOT_MEASURED"
    # B7（P0-1 §3.3 的規則）：V1、V2 的 X1 原登記、X1 截尾 H=3,600 的字和 V0 相同；X5 不重跑（字沿用舊的 p01c_verdict.json）
    w = {V: (S[f"{V}.x1"].get("word") if isinstance(S[f"{V}.x1"], dict) else None,
             S[f"{V}.x1h3600"].get("word") if isinstance(S[f"{V}.x1h3600"], dict) else None) for V in ("V0", "V1", "V2")}
    S["B7"] = dict(words=w, same_as_V0={V: (None if None in w[V] or None in w["V0"] else w[V] == w["V0"]) for V in ("V1", "V2")})
    return S


def cmd_report(a):
    import pandas as pd
    comps = [cmp_csv(o, n, k) for o, n, k in PAIRS] + [cmp_json(o, n) for o, n in JPAIRS]
    old = summary(R("icpe_h200", "p01"), R("icpe_h200", "p12"), R("p3_explore_h200", "x3"), R("icpe_h200", "p12", "x4"), R("icpe_h200", "p02"),
                  R("s3_h200", "e2e"))
    new = summary(os.path.join(T1, "p01"), os.path.join(T1, "p12"), os.path.join(T1, "explore", "x3"), os.path.join(T1, "p12", "x4"),
                  os.path.join(T1, "p02"), os.path.join(T1, "e2e"))
    p01c_old = json.load(open(R("icpe_h200", "p01", "p01c_verdict.json")))
    new["B7"]["x5_words_reused"] = {V: p01c_old["words"][V]["x5_registered"] for V in ("V0", "V1", "V2")}
    old["B7"]["x5_words_reused"] = new["B7"]["x5_words_reused"]
    rows = []
    for k in sorted(set(old) | set(new)):
        for path, ov in (dict(_leaves(old[k])) if isinstance(old.get(k), (dict, list)) else {"": old.get(k)}).items():
            nv = (dict(_leaves(new[k])) if isinstance(new.get(k), (dict, list)) else {"": new.get(k)}).get(path, "NOT_MEASURED")
            d = (nv - ov) if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (ov, nv)) else None
            rows.append(dict(run_id=run_id(), ts=ts(), item=k + (f".{path}" if path else ""), old_mi300x_cpu=ov, new_h200_cpu=nv, diff=d,
                             changed=(ov != nv)))
    word_changes = [dict(file=c["new"], col=col, changes=v["changes"]) for c in comps if c.get("status") == "ok" and "cols" in c
                    for col, v in c["cols"].items() if v["kind"] == "str" and v["n_changed"]]
    crossings = [dict(file=c["new"], **x) for c in comps if c.get("crossings") for x in c["crossings"]]
    nm = [c["new"] for c in comps if c.get("status") == "NOT_MEASURED"]
    doc = dict(run_id=run_id(), ts=ts(), prereg="docs/h200/icpe/T1_h200_cpu_tier.md", tier_params=os.path.relpath(NEW, REPO),
               summary_old=old, summary_new=new, n_word_changes=sum(len(w["changes"]) for w in word_changes), word_changes=word_changes,
               n_crossings=len(crossings), crossings=crossings, not_measured=nm, comparisons=comps)
    json.dump(doc, open(os.path.join(T1, "t1_compare.json"), "w"), indent=1, ensure_ascii=False, default=str)
    pd.DataFrame(rows).to_csv(os.path.join(T1, "t1_compare.csv"), index=False)
    print(json.dumps(dict(n_word_changes=doc["n_word_changes"], n_crossings=len(crossings), not_measured=nm,
                          E2E_old=old.get("E2E"), E2E_new=new.get("E2E"), B7=new["B7"]), ensure_ascii=False, default=str))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("build")
    sp.add_parser("selftest")
    sp.add_parser("dump")
    sp.add_parser("report")
    a = ap.parse_args()
    dict(build=cmd_build, selftest=cmd_selftest, dump=cmd_dump, report=cmd_report)[a.cmd](a)


if __name__ == "__main__":
    main()
