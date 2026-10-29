#!/usr/bin/env python3
"""t3.py — ICPE T3（docs/h200/icpe/T3_planner_sensitivity.md）：還原規劃器換成 STAIR（照 CacheFlow 的想法自己寫的簡化版）的敏感度。

  python code/icpe/t3.py selftest   §1：沒設 X1_PLANNER 時 x1_sim 的四條不變量照過；隨機 2,000 個還原 STAIR ≤ Cake
  python code/icpe/t3.py ratio      §2（只報告）：conversation 一個重播，同一份放置與 I/O 線狀態下，回來請求的 STAIR ÷ Cake 還原時間
  python code/icpe/t3.py report     §3：T1（Cake）對 T3（STAIR）的判定字與數字 → results/icpe_h200/t3/t3_compare.{json,csv}

環境：P3_CONST=h200、P3_TIER=h200cpu（T1 的常數）；X1_PLANNER 由這支自己控制（selftest／ratio 在同一個 process 裡逐個 View 設）。
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
REPO = os.path.dirname(CODE)
sys.path.insert(0, os.path.join(CODE, "p3x"))
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)
T1 = os.path.join(REPO, "results", "icpe_h200", "t1")
T3 = os.path.join(REPO, "results", "icpe_h200", "t3")


def run_id():
    import t1
    return t1.run_id()


def ts():
    import t1
    return t1.ts()


def _views(XS, K, slow):
    a = XS.View(K, slow, "idle", "cake", 300, 1)
    b = XS.View(K, slow, "idle", "cake", 300, 1)
    a.planner, b.planner = "cake", "stair"
    return a, b


def _fresh(v, K, slow):
    c, s = K["cpu"], K["slow"][slow]
    v.cpu = type(v.cpu)("cpu", c["r"], c["w"], c["k"])
    v.slow = type(v.slow)("slow_" + slow, s["r"], s["w"], s["k"])


def cmd_selftest(a):
    import subprocess
    import tempfile
    assert os.environ.get("X1_PLANNER", "cake") == "cake", "selftest 要在沒設 X1_PLANNER 的環境跑"
    import x1_sim as XS
    ok_all = True
    # (1) 沒設 X1_PLANNER：x1_sim 原本的不變量（輸出寫到暫存目錄，不碰 T1 的 x1_invariants.csv）
    with tempfile.TemporaryDirectory() as td:
        code = ("import sys, types; sys.path[:0]=[%r, %r]; import x1_sim as XS; XS.OUTDIR=%r; "
                "rc = XS.cmd_selftest(types.SimpleNamespace(traces=XS.TRACES, cpu=[128, 1024], audit_n=1500, audit_cap=64)); "
                "sys.exit(rc if isinstance(rc, int) else 0)") % (os.path.join(CODE, "p3x"), CODE, td)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        print(r.stdout[-1500:], r.stderr[-800:], flush=True)
        ok1 = r.returncode == 0
    print(f"[t3 selftest] (1) 沒設 X1_PLANNER 時 x1_sim selftest：{'PASS' if ok1 else 'FAIL'}（exit {r.returncode}）", flush=True)
    ok_all &= ok1
    # (2) 隨機 2,000 個還原：新的 I/O 線、隨機排列，STAIR ≤ Cake
    K = XS.constants()
    rng = random.Random(20261029)
    worst, n = 0.0, 0
    for slow in ("ssd", "nfs", "b32"):
        vc, vs = _views(XS, K, slow)
        for _ in range(2000 // 3 + 1):
            L = rng.randint(1, 250)
            pat = rng.choice(["cpu", "slow", "mix", "pf", "all"])
            lay = []
            for i in range(L):
                if pat == "cpu":
                    lay.append(1)
                elif pat == "slow":
                    lay.append(3)
                elif pat == "pf":
                    lay.append(rng.choice([1, 2]))
                elif pat == "mix":
                    lay.append(1 if i < L * rng.random() else 3)
                else:
                    lay.append(rng.choice([1, 2, 3]))
            blocks = list(range(L))
            t0 = 100.0
            pf = [t0 - 1.0 if rng.random() < 0.8 else t0 + rng.random() * 0.5 for _ in range(L)]
            res = []
            for v in (vc, vs):
                _fresh(v, K, slow)
                res.append(v.restore(lay, blocks, L, L, t0, L, pf)[0])
            n += 1
            worst = max(worst, res[1] - res[0])
    ok2 = worst <= 1e-9
    print(f"[t3 selftest] (2) {n} 個隨機還原，max(STAIR − Cake)＝{worst:.3e} s：{'PASS' if ok2 else 'FAIL'}", flush=True)
    ok_all &= ok2
    return 0 if ok_all else 1


class DualView:
    """包兩個 View（Cake、STAIR）：每次還原都從同一份 I/O 線狀態各算一次，只把 Cake 的結果交給引擎（軌跡和 Cake 的 run 相同）。"""

    def __new__(cls, XS, K, slow, nmax, nreq):
        base = XS.View

        class _V(base):
            def __init__(self):
                super().__init__(K, slow, "idle", "cake", nmax, nreq)
                self.planner = "cake"
                self.pairs = []

            def restore(self, lay, blocks, L, n_all, t0, old_n, pf_done):
                c0, s0 = copy.copy(self.cpu), copy.copy(self.slow)
                self.planner = "stair"
                r_s = base.restore(self, lay, blocks, L, n_all, t0, old_n, pf_done)
                self.cpu, self.slow = c0, s0
                self.planner = "cake"
                r_c = base.restore(self, lay, blocks, L, n_all, t0, old_n, pf_done)
                self.pairs.append((L, r_c[1], r_s[1]))
                return r_c
        return _V()


def cmd_ratio(a):
    import numpy as np
    import pandas as pd
    import x1_sim as XS
    K = XS.constants()
    P = XS.prep("conversation", "R1")
    rows = []
    kind = np.asarray(P["kindc"])
    for slow in ("ssd", "nfs"):
        for cg in (128, 256):
            v = DualView(XS, K, slow, P["nmax"], len(P["full"]))
            XS.Engine(P, "LRU", cg * XS.GiB, [v], K).run()
            pr = pd.DataFrame(v.pairs, columns=["L", "rest_cake", "rest_stair"])
            pr = pr[(pr.L > 0) & (pr.rest_cake > 0)]
            q = pr.rest_stair / pr.rest_cake
            rows.append(dict(run_id=run_id(), ts=ts(), trace="conversation", rule="R1", policy="LRU", slow=slow, cpu_GiB=cg, n=len(pr),
                             ratio_median=float(q.median()), ratio_p10=float(q.quantile(.1)), ratio_p90=float(q.quantile(.9)),
                             ratio_min=float(q.min()), ratio_max=float(q.max()), frac_le_0_9=float((q <= 0.9).mean()),
                             n_ret=int((kind == 1).sum())))
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    os.makedirs(T3, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(T3, "t3_ratio.csv"), index=False)
    return 0


def cmd_report(a):
    import pandas as pd
    import t1
    pairs = []
    for V in ("V0", "V1", "V2"):
        for p in ("x1", "x1h1800", "x1h3600"):
            for f, k in ((f"{p}_verdict.csv", ["rule", "component"]), (f"{p}_headroom_components.csv", t1.K_X1)):
                pairs.append((os.path.join(T1, "p01", f"sim_{V}", "x1", f), os.path.join(T3, "p01", f"sim_{V}", "x1", f), k))
    for p in ("x1fi", "x1fiuser"):
        for f, k in ((f"{p}_verdict.csv", ["rule", "component"]), (f"{p}_headroom_components.csv", t1.K_X1)):
            pairs.append((os.path.join(T1, "p12", "x1", f), os.path.join(T3, "p12", "x1", f), k))
    pairs.append((os.path.join(T1, "p02", "fullsim_hint.csv"), os.path.join(T3, "p02", "fullsim_hint.csv"), t1.K_FS))
    comps = [t1.cmp_csv(o, n, k) for o, n, k in pairs]
    S = lambda r: t1.summary(os.path.join(r, "p01"), os.path.join(r, "p12"), os.path.join(r, "explore", "x3"),     # noqa: E731
                             os.path.join(r, "p12", "x4"), os.path.join(r, "p02"), os.path.join(r, "e2e"))
    old, new = S(T1), S(T3)
    keep = [k for k in old if k.startswith(("V0.", "V1.", "V2.", "B1", "B3", "B7", "C2"))]
    rows = []
    for k in keep:
        for path, ov in (dict(t1._leaves(old[k])) if isinstance(old.get(k), (dict, list)) else {"": old.get(k)}).items():
            nv = (dict(t1._leaves(new[k])) if isinstance(new.get(k), (dict, list)) else {"": new.get(k)}).get(path, "NOT_MEASURED")
            d = (nv - ov) if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (ov, nv)) else None
            rows.append(dict(run_id=run_id(), ts=ts(), item=k + (f".{path}" if path else ""), t1_cake=ov, t3_stair=nv, diff=d, changed=(ov != nv)))
    words = [dict(file=c["new"], col=col, changes=v["changes"]) for c in comps if c.get("status") == "ok" and "cols" in c
             for col, v in c["cols"].items() if v["kind"] == "str" and v["n_changed"]]
    crossings = [dict(file=c["new"], **x) for c in comps if c.get("crossings") for x in c["crossings"]]
    nm = [c["new"] for c in comps if c.get("status") == "NOT_MEASURED"]
    doc = dict(run_id=run_id(), ts=ts(), prereg="docs/h200/icpe/T3_planner_sensitivity.md", planner="stair",
               summary_t1=dict((k, old[k]) for k in keep), summary_t3=dict((k, new[k]) for k in keep),
               n_word_changes=sum(len(w["changes"]) for w in words), word_changes=words, n_crossings=len(crossings), crossings=crossings,
               not_measured=nm, comparisons=comps)
    os.makedirs(T3, exist_ok=True)
    json.dump(doc, open(os.path.join(T3, "t3_compare.json"), "w"), indent=1, ensure_ascii=False, default=str)
    pd.DataFrame(rows).to_csv(os.path.join(T3, "t3_compare.csv"), index=False)
    print(json.dumps(dict(n_word_changes=doc["n_word_changes"], n_crossings=len(crossings), not_measured=nm, B7=new.get("B7")),
                     ensure_ascii=False, default=str))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for c in ("selftest", "ratio", "report"):
        sp.add_parser(c)
    a = ap.parse_args()
    sys.exit({"selftest": cmd_selftest, "ratio": cmd_ratio, "report": cmd_report}[a.cmd](a))


if __name__ == "__main__":
    main()
