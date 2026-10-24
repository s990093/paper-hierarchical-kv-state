"""x4_check_qp.py — 對拍：增量版 queue_prefetch vs 參考版（每次重建保護集合），摘要必須逐欄完全相同。"""
import itertools
import os
import sys
import multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x4_sim as X  # noqa: E402


def one(c):
    tr, rule, p, cg, slow = c
    P = X.params()
    f, _, _, _ = X.f_ext()
    reqs, ns, ws = X._get_subset(tr, p, 0, rule)
    cap = int(round(cg * X.GiB / X.CB))
    bad = []
    for pol in ("QP", "RP", "GP", "OP"):
        qp, ip = X.POLICIES[pol]
        out = []
        for ref in (True, False):
            sim = X.Sim(reqs, cap, X.mk_tiers(P, slow), f, qp=qp, ip=ip, promote=True)
            sim.qp_ref = ref
            rows, busy = sim.run_events()
            out.append(X.summarize(rows, busy, sim, P, slow, cap, ws))
        diff = [k for k in out[0] if not (out[0][k] == out[1][k] or (out[0][k] != out[0][k] and out[1][k] != out[1][k]))]
        if diff:
            bad.append((pol, diff[:5]))
    return c, bad


if __name__ == "__main__":
    cells = list(itertools.product(["conversation", "toolagent"], ["loose", "strict"], [0.1, 0.2], [32, 128], ["nfs"]))
    with mp.Pool(8) as pool:
        res = pool.map(one, cells)
    nbad = 0
    for c, bad in res:
        print(c, "OK" if not bad else f"DIFF {bad}")
        nbad += bool(bad)
    print(f"QP_EQUIV {'PASS' if nbad == 0 else 'FAIL'}: {len(res) - nbad}/{len(res)} cells identical (4 policies each)")
    sys.exit(0 if nbad == 0 else 1)
