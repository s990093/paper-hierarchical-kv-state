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


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("build")
    sp.add_parser("selftest")
    sp.add_parser("dump")
    a = ap.parse_args()
    dict(build=cmd_build, selftest=cmd_selftest, dump=cmd_dump)[a.cmd](a)


if __name__ == "__main__":
    main()
