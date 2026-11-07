"""排隊一階近似的誤差指標：服務順序和紀錄順序差多少（每個請求的排名位移），以及「下一個回來的 session」的判斷被改變的比例。
用法：python -I qpost_reorder.py --trace nb:semianalysis_N16_s1 --rule nb --cpu 111.0625 --out <目錄>
跑一個 job（CPU LRU × GPU LRU wb、6 個時間設定、qpost=1），讀 v{k}_qps（開始時刻）算：
  rank_disp＝|服務排名 − 請求序號|；
  pair_flip＝隨機抽 20 萬對「不同 session、序號相差 ≤ W 的請求」，服務先後和紀錄先後相反的比例（W＝50、500、5,000）。"""
import argparse, json, os, shutil, subprocess, sys
ap = argparse.ArgumentParser()
ap.add_argument("--trace", required=True); ap.add_argument("--rule", default="nb"); ap.add_argument("--cpu", type=float, default=111.0625)
ap.add_argument("--out", required=True)
a = ap.parse_args()
REPO = os.environ["REPO"]
for p in ("code", "code/icpe", "code/p3x", "code/icpe/rux_x1fast/v6_slowcap"):
    sys.path.insert(0, os.path.join(REPO, p))
import numpy as np
import x1_rust_v6 as XR
sys.modules["x1_rust"] = XR
import p3_fvar
os.makedirs(a.out, exist_ok=True)
XS = p3_fvar.patch_x1("V1", os.path.join(REPO, "results/icpe_h200/p01/calib"), a.out, {})
K = XS.constants(); XS._K = K
P = XS.prep(a.trace, a.rule)
TDIR, _ = XR.prepare_trace(P, a.trace, a.rule)
XR.SRC, XR._BIN = os.path.join(REPO, "code/icpe/rux_x1fast/v8_features"), None
BIN = XR.binary()
V6 = [(s, g, "cake") for s in ("ssd", "b32", "nfs") for g in ("idle", "busy")]
views = XS.make_views(K, P, V6)
jd = XR.write_job(XS, P, "LRU", a.cpu * XS.GiB, views, K, dict(gpu_cap=int(111.0625 * XS.GiB), gpu_policy="LRU", gpu_write="wb", pf_target="cpu"))
_ln = open(os.path.join(jd, "jobs.txt")).read().strip()
open(os.path.join(jd, "jobs.txt"), "w").write(_ln + " qpost=1\n")
env = dict(os.environ, X1FAST_THREADS="1", X1FAST_FORK="0", X1FAST_SUMMARY_ONLY="0"); env.pop("X1FAST_SERVE", None)
cp = subprocess.run([BIN, TDIR, jd], env=env, capture_output=True, text=True); assert cp.returncode == 0, cp.stderr[-2000:]
sess = np.fromfile(os.path.join(TDIR, "sess.i64"), np.int64)
n = len(sess)
rng = np.random.default_rng(0)
rows = []
for k, (sl, g, _) in enumerate(V6):
    S = np.fromfile(os.path.join(jd, "out", "j0", f"v{k}_qps.f64"), np.float64)
    W = np.fromfile(os.path.join(jd, "out", "j0", f"v{k}_qpw.f64"), np.float64)
    order = np.lexsort((np.arange(n), S))
    rank = np.empty(n, np.int64); rank[order] = np.arange(n)
    disp = np.abs(rank - np.arange(n))
    r = dict(trace=a.trace, cpu=a.cpu, slow=sl, gpu=g, mean_wait=float(W.mean()), rank_disp_median=float(np.median(disp)),
             rank_disp_p90=float(np.percentile(disp, 90)), rank_disp_p99=float(np.percentile(disp, 99)), rank_disp_max=int(disp.max()))
    for Wn in (50, 500, 5000):
        i = rng.integers(0, n - 1, 400000); j = np.minimum(i + rng.integers(1, Wn + 1, len(i)), n - 1)
        m = (sess[i] != sess[j]) & (i != j)
        i, j = i[m][:200000], j[m][:200000]
        r[f"pair_flip_w{Wn}"] = float(np.mean(rank[i] > rank[j]))
    rows.append(r); print(json.dumps(r), flush=True)
shutil.rmtree(jd, ignore_errors=True)
json.dump(rows, open(os.path.join(a.out, "qpost_reorder.json"), "w"), indent=1)
