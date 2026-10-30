"""e2e_diag.py — E2E-C 判定之後的診斷（只報告，不改判定；docs/stage3_design_20261024/runs/E2E_combo.md「FreeInference 的異常」）。

FreeInference 第 9–16 週，預取（HINT-PF、O-PF）比 LRU 慢。看之前先寫下的假設：
  H1 慢層通道飽和：預取的讀佔住慢層通道，需要的還原排在後面（rho_r 接近 1）
  H2 CPU 層打轉：預取把較快要用的塊擠出去，回來的請求更常讀慢層或重算（ret_slow_loads、ret_recompute、n_demote 變多）
  H3 放大：§3.3 把還原那段（含等通道的時間）整段乘上 r_e2e；只會放大，不會讓正負號翻過來 → 沒有背景（IDLE）時異常應該還在
  H4 程式的錯
做法：同一週、同一格，跑 LRU／HINT-PF／O-PF × {cprefill 背景（stream）, 沒有背景（IDLE）}，每次記 E2E 的回來請求平均 TTFT（測試集、開發集）
與 X1 的 summarize（rho_r、rho_w、slow_read_GiB、prefetch_GiB、n_demote、ret_*）。

  python code/s3/e2e_diag.py --model llama --weeks 2026-05-17_2026-05-23 2026-07-12_2026-07-18 2026-08-30_2026-09-05 \
      --slows ssd nfs --cpu 128 --bg cprefill:2048:65536 --out results/s3_h200/e2e/diag/fi_anomaly.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import e2e as E  # noqa: E402

XS, XG = E.XS, E.XG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama", choices=E.MODELS)
    ap.add_argument("--weeks", nargs="+", required=True)
    ap.add_argument("--slows", nargs="+", default=list(E.SLOWS))
    ap.add_argument("--cpu", type=int, nargs="+", default=[128])
    ap.add_argument("--bg", default="cprefill:2048:65536")
    ap.add_argument("--policies", nargs="+", default=["LRU", "HINT-PF", "O-PF"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    E.use_model(a.model)
    K = E.constants(a.model)
    dc = json.load(open(os.path.join(E.OUT, "dcheck.json")))
    fac_bg = E.factors(E.load_b0(a.model), a.bg, "stream", dc[a.model]["path"])
    views = {"bg": fac_bg, "idle": E.IDLE}
    rid, ts = E.run_id(), E.ts()
    want = {f"fi:{w}" for w in a.weeks}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    new = not os.path.exists(a.out)
    fh = open(a.out, "a", newline="")
    w = None
    for name, week in E.replays("freeinference"):
        if name not in want:
            continue
        P = XS.prep(name)
        lead, lcnt = E.leads_fi(P, week, a.model)
        umask = __import__("numpy").asarray(P["user_trigger"], bool)
        nmax, nreq = P["nmax"], len(P["full"])
        for slow in a.slows:
            for cg in a.cpu:
                for vname, fac in views.items():
                    for pol in a.policies:
                        v = E.BgView(K, slow, fac, nmax, nreq)
                        if pol == "LRU":
                            eng = XS.Engine(P, "LRU", cg * XS.GiB, [v], K)
                        else:
                            eng = XG.make_engine(P, pol, cg * XS.GiB, [v], K, lead)
                        eng.run()
                        ws = E.window_stats(P, v, umask)
                        sm = XS.summarize(P, v, eng)
                        row = dict(run_id=rid, ts=ts, model=a.model, replay=name, slow=slow, cpu_GiB=cg, view=vname,
                                   bg=fac["bg"], policy=pol,
                                   ttft_test=ws["test"]["sum"] / ws["test"]["n"], ttft_dev=ws["dev"]["sum"] / ws["dev"]["n"],
                                   n_test=ws["test"]["n"], n_dev=ws["dev"]["n"], leads=json.dumps(lcnt, sort_keys=True),
                                   **{k: sm[k] for k in ("n_ret", "mean_ret", "median_ret", "p90_ret", "mean_rest_ret", "ret_cpu_loads",
                                                         "ret_slow_loads", "ret_recompute", "slow_read_GiB", "rho_r", "slow_write_GiB",
                                                         "rho_w", "prefetch_GiB", "n_demote", "n_drop", "n_unprefetch")})
                        if w is None:
                            w = csv.DictWriter(fh, fieldnames=list(row.keys()))
                            if new:
                                w.writeheader()
                        w.writerow(row)
                        fh.flush()
                        print(json.dumps(dict(replay=name, slow=slow, cpu=cg, view=vname, policy=pol), ensure_ascii=False), flush=True)
        XS._PCACHE.clear()
    fh.close()
    print("→", a.out)


if __name__ == "__main__":
    main()
