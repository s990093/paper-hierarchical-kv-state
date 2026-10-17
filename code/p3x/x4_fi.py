"""x4_fi.py — X4 的 FreeInference agentic trace 對照（登記之後才加；README §4.1，2026-10-17T10:01Z）〔模擬，CPU，MI300X 常數〕

資料：/ssd8/hungwei/paper-hkv/datasets/freeinference_agentic_trace/（HF harvardMadsys/freeinference_agentic_trace @ 16d3c04d）。
用 x4_sim 的同一個 Sim／summarize／Cake／LRU；只換載入：
  - session 由 session_id 給定（不用重建）；parent＝同一 session 的上一個 step。
  - block_ids 是 16-token 的鏈式 hash（o200k_base）。512-token block 的 id＝第 32k+31 個 16-token id（鏈式 → 唯一代表整段前綴）；
    最後不滿 512 token 的部分當一個這個請求獨有的 block。o200k token 當成 Llama token 1:1〔假設〕。
  - 到達＝start_ms。

規則 6（載入時斷言）：
  (1) len(block_ids)==floor(Σrole_tokens/16) 的比例 ≥0.95（16-token 單位；README.txt 的建議）
  (2) ttft_ms ≤ end_ms−start_ms 的比例 ≥0.95，且 (end_ms−start_ms)/completion_tokens 的中位數在 1–2000 ms/token（毫秒）
  (3) 512-token id 的前綴鏈：同一個 id 永遠在同一位置
  (4) session_id 對拍 block id：同一 session 前後兩個 step，第 0 個 512-block id 相同的比例 ≥0.8（同一段開頭）

子命令：
  desc   每週的描述（只讀 parquet metadata＋選的週）
  sweep  主掃描（同 x4_sim 的欄位，trace＝fi_<week>，rule＝session_id）
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import x4_sim as X  # noqa: E402

FI_DIR = os.environ.get("X4_FI_DIR", "/ssd8/hungwei/paper-hkv/datasets/freeinference_agentic_trace/release")
FI_REV = "16d3c04dae6ccaa0c48a9064e204e3a8f1eb9ade"


def load_week(week):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    d = os.path.join(FI_DIR, week, "processed")
    f = pq.ParquetFile(os.path.join(d, "requests.parquet"))
    cols = ["session_id", "step", "start_ms", "end_ms", "ttft_ms", "completion_tokens", "role_tokens", "step_trigger",
            "block_ids"]
    recs = []
    n_unit_ok = n = 0
    for b in f.iter_batches(batch_size=4000, columns=cols):
        ln = pc.list_value_length(b.column("block_ids")).to_numpy()
        rt = b.column("role_tokens").to_pylist()
        tok = np.array([sum(v for v in x.values() if v is not None) for x in rt])
        n_unit_ok += int((ln == tok // 16).sum())
        n += len(b)
        ids = pc.list_slice(b.column("block_ids"), 31, None, 32).to_pylist()
        sid = b.column("session_id").to_pylist()
        step = b.column("step").to_numpy()
        st = b.column("start_ms").to_numpy()
        en = b.column("end_ms").to_numpy()
        tt = b.column("ttft_ms").to_numpy(zero_copy_only=False)
        ct = b.column("completion_tokens").to_numpy(zero_copy_only=False)
        trig = b.column("step_trigger").to_pylist()
        for i in range(len(b)):
            recs.append((int(st[i]), sid[i], int(step[i]), int(en[i]), tt[i], ct[i], trig[i], int(tok[i]), ids[i]))
    unit = n_unit_ok / n
    assert unit >= 0.95, f"{week}: len(block_ids)==floor(Σrole_tokens/16) 只有 {unit:.4f}"
    dur = np.array([r[3] - r[0] for r in recs], float)
    tt = np.array([r[4] if r[4] is not None else np.nan for r in recs], float)
    ok_t = np.nanmean(tt <= dur + 1e-9)
    ct = np.array([r[5] if r[5] is not None else 0 for r in recs], float)
    m = ct >= 50
    ms_per_tok = float(np.median(dur[m] / ct[m]))
    assert ok_t >= 0.95, f"{week}: ttft_ms ≤ end−start 只有 {ok_t:.3f}"
    assert 1 <= ms_per_tok <= 2000, f"{week}: (end−start)/completion 中位數 {ms_per_tok:.1f} ms/token，單位不是毫秒？"
    recs.sort(key=lambda r: (r[0], r[1], r[2]))
    pos_of = {}
    bad = 0
    for r in recs:
        for i, b_ in enumerate(r[8]):
            if pos_of.setdefault(b_, i) != i:
                bad += 1
    assert bad == 0, f"{week}: 512-token id 前綴鏈違反 {bad} 次"
    last = {}
    reqs = []
    same0 = npair = 0
    gaps_user = []
    for k, r in enumerate(recs):
        st_, sid, step, en, _, _, trig, tok, ids = r
        h = list(ids)
        if tok - 512 * len(h) > 0:
            h.append(-(k + 1))        # 這個請求獨有的不滿 512 token 尾段
        pk = last.get(sid, -1)
        if pk >= 0:
            npair += 1
            if reqs[pk]["h"] and h and reqs[pk]["h"][0] == h[0]:
                same0 += 1
            if trig == "user":
                gaps_user.append((st_ - recs[pk][3]) / 1000.0)
        reqs.append(dict(t=st_ / 1000.0, h=h, ret=pk >= 0, ret_in_unit=pk >= 0 and trig == "user", par=pk,
                         out=r[5], sid=sid))
        last[sid] = k
    f0 = same0 / max(1, npair)
    assert f0 >= 0.8, f"{week}: 同 session 前後兩步第 0 個 block 相同只有 {f0:.3f}（session_id 和 block id 對不上？）"
    meta = dict(week=week, hf_rev=FI_REV, n_req=len(reqs), n_sessions=len(last), unit_match=unit,
                frac_ttft_le_dur=float(ok_t), med_ms_per_out_tok=ms_per_tok, chain_bad=bad, frac_same_block0=f0,
                n_ret=sum(q["ret"] for q in reqs), n_ret_user=sum(q["ret_in_unit"] for q in reqs),
                med_blocks=float(np.median([len(q["h"]) for q in reqs])), max_blocks=max(len(q["h"]) for q in reqs),
                span_s=reqs[-1]["t"] - reqs[0]["t"],
                user_return_gap_s_p50=float(np.median(gaps_user)) if gaps_user else float("nan"))
    return reqs, meta


def subset_fi(reqs_all, p, seed):
    import random
    sids = sorted({q["sid"] for q in reqs_all})
    rng = random.Random(1_000_003 * (seed + 1) + 29)
    u = {s: rng.random() for s in sids}
    keep = [i for i, q in enumerate(reqs_all) if u[q["sid"]] < p]
    kidx = {i: k for k, i in enumerate(keep)}
    out = []
    for i in keep:
        q = reqs_all[i]
        pk = kidx.get(q["par"], -1) if q["par"] >= 0 else -1
        out.append(dict(t=q["t"], h=q["h"], ret=pk >= 0, ret_in_unit=pk >= 0 and q["ret_in_unit"], par=pk,
                        out=q["out"]))
    return X.finish_reqs(out), len({reqs_all[i]["sid"] for i in keep})


_C = {}


def _get(week, p, seed):
    if week not in _C:
        _C[week] = load_week(week)
    k = (week, p, seed)
    if k not in _C:
        reqs, ns = subset_fi(_C[week][0], p, seed)
        _C[k] = (reqs, ns, len({b for q in reqs for b in q["h"]}))
    return _C[k]


def _job(cfg):
    P = X.params()
    f, _, _, _ = X.f_ext(4000)
    if cfg.get("fscale", 1.0) != 1.0:
        f = [x * cfg["fscale"] for x in f]
    reqs, ns, ws = _get(cfg["week"], cfg["p"], cfg["seed"])
    cap = int(round(cfg["cpu_GiB"] * X.GiB / X.CB))
    out = []
    for pol in cfg["policies"]:
        qp, ip = X.POLICIES[pol]
        sim = X.Sim(reqs, cap, X.mk_tiers(P, cfg["slow"]), f, qp=qp, ip=ip, promote=True)
        rows, busy = sim.run_events()
        sm = X.summarize(rows, busy, sim, P, cfg["slow"], cap, ws)
        c = {k: v for k, v in cfg.items() if k not in ("policies", "week")}
        out.append(dict(c, trace=f"fi_{cfg['week']}", rule="session_id", policy=pol, n_units=ns, **sm))
    return out


def cmd_desc(a):
    import pandas as pd
    rows = []
    for w in a.weeks:
        reqs, meta = load_week(w)
        npos = sum(len(q["h"]) for q in reqs)
        meta["frac_block_positions_extrapolated"] = sum(max(0, len(q["h"]) - 80) for q in reqs) / npos
        f, _, _, _ = X.f_ext(4000)
        meta["gpu_new_prefill_s"] = 0.0
        rr = X.finish_reqs([dict(t=q["t"], h=q["h"], ret=q["ret"], ret_in_unit=q["ret_in_unit"], par=q["par"], out=0)
                            for q in reqs])
        meta["gpu_new_prefill_s"] = sum(sum(f[q["n_old"]:len(q["h"])]) for q in rr)
        meta["ws_GiB"] = len({b for q in reqs for b in q["h"]}) * X.CB / X.GiB
        rows.append(dict(run_id=X.RID, ts=X.now_ts(), **meta))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(X.OUT, "x4_fi_desc.csv"), index=False)


def cmd_sweep(a):
    import multiprocessing as mp
    import pandas as pd
    cfgs = [dict(week=w, p=p, seed=s, cpu_GiB=c, slow=sl, fscale=a.fscale, policies=a.policies)
            for w in a.weeks for p in a.p for s in (a.seeds if p < 1 else [0]) for c in a.cpu_GiB for sl in a.slow]
    print("jobs", len(cfgs), flush=True)
    t0 = datetime.now()
    res = []
    with mp.Pool(a.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, cfgs, chunksize=1)):
            res.extend(r)
            print(i, "/", len(cfgs), datetime.now() - t0, flush=True)
    d = pd.DataFrame(res)
    d.insert(0, "ts", X.now_ts())
    d.insert(0, "run_id", X.RID)
    fn = os.path.join(X.run_dir(), a.out)
    d.to_csv(fn, index=False)
    print(fn, len(d), datetime.now() - t0, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["desc", "sweep"])
    ap.add_argument("--weeks", nargs="*", default=["2026-07-26_2026-08-01"])
    ap.add_argument("--p", type=float, nargs="*", default=[1.0])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--cpu-GiB", dest="cpu_GiB", type=float, nargs="*", default=[32, 128, 512])
    ap.add_argument("--slow", nargs="*", default=["local", "b32", "nfs"])
    ap.add_argument("--policies", nargs="*", default=list(X.POLICIES))
    ap.add_argument("--fscale", type=float, default=1.0)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default="x4_fi_sweep.csv")
    a = ap.parse_args()
    os.makedirs(X.OUT, exist_ok=True)
    {"desc": cmd_desc, "sweep": cmd_sweep}[a.cmd](a)


if __name__ == "__main__":
    main()
