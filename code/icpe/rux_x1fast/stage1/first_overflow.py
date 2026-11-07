"""icpe-rux 第 1 段（d）：慢層第一次超過上限的請求位置 r*（在 r* 之前，7 種慢層規則與沒有上限的 job 狀態完全相同）。
不計時；只包一層 _slow_enforce 記下「第一次真的丟東西」的請求序號，不改語意。
用法：python -I first_overflow.py --trace <name> [--policy LRU --gpu-policy LRU --cpu 111.0625 ...] --out <dir>"""
import argparse, json, os, socket, sys
from datetime import datetime
ap = argparse.ArgumentParser()
ap.add_argument("--trace", required=True)
ap.add_argument("--rule", default="R1")
ap.add_argument("--cpus", nargs="+", type=float, default=[8, 32, 111.0625, 256, 1024])
ap.add_argument("--policy", default="LRU")
ap.add_argument("--gpu-policy", default="LRU")
ap.add_argument("--out", required=True)
a = ap.parse_args()
REPO = os.environ["REPO"]
for p in ("code/p3x", "code/icpe", "code"):
    sys.path.insert(0, os.path.join(REPO, p))
import p3_fvar
os.makedirs(a.out, exist_ok=True)
XS = p3_fvar.patch_x1("V1", os.path.join(REPO, "results/icpe_h200/p01/calib"), a.out, {})
K = XS.constants()
XS._K = K
P = XS.prep(a.trace, a.rule)
n = len(P["full"])
fh = open(os.path.join(a.out, "first_overflow.jsonl"), "a")
for cpu in a.cpus:
    views = XS.make_views(K, P, [("ssd", "idle", "cake")])
    e = XS.Engine(P, a.policy, cpu * XS.GiB, views, K, gpu_cap=int(111.0625 * XS.GiB), gpu_policy=a.gpu_policy, gpu_write="wb",
                  slow_cap=0.5, slow_policy="LRU")
    first = {"clean": None, "drop": None}
    inner = e._slow_enforce

    def wrapped():
        c0, d0 = e.n_sc_clean, e.n_sc_drop
        inner()
        if first["clean"] is None and e.n_sc_clean > c0:
            first["clean"] = e.cur
        if first["drop"] is None and e.n_sc_drop > d0:
            first["drop"] = e.cur
    e._slow_enforce = wrapped
    e.run()
    rs = [x for x in (first["clean"], first["drop"]) if x is not None]
    rstar = min(rs) if rs else None
    r = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"),
             host=socket.gethostname(), trace=a.trace, policy=a.policy, gpu_policy=a.gpu_policy, cpu_GiB=cpu, n_req=n,
             sc_cap=e.sc_cap, first_clean_r=first["clean"], first_drop_r=first["drop"], rstar=rstar,
             rstar_frac=None if rstar is None else round(rstar / n, 4), n_sc_clean=e.n_sc_clean, n_sc_drop=e.n_sc_drop)
    fh.write(json.dumps(r) + "\n")
    fh.flush()
    print(json.dumps(r), flush=True)
