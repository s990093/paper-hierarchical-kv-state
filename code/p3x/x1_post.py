"""x1_post.py — X1 事後檢查（只報告，不改原登記判定）。

  freq2q   O-ORD 的對手集合加上事後加入的 FREQ2Q（session 版 2Q：先比請求數、再比閒置），重算 headroom（每格取 R1／R2 較小）
  minutes  「CPU 裝得下幾分鐘的歷史」：LRU 在該容量下，session 從回合結束到被降級的平均時間（秒→分鐘）
  dead     截尾穩健版：O-DROP／O-NOW 換成 -H1800／-H3600 版（「之後沒人讀」只用在離 trace 結束 ≥H 秒的決定），重算判定
另外檢查：x1_sweep_post.csv 的 LRU 和主 sweep 的 LRU 逐格相同（同一份程式路徑）。
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "..", "..", "results", "p3_explore", "x1")
CELL = ["trace", "cpu_GiB", "slow", "gpu", "restore"]


def piv(d):
    return d.pivot_table(index=["trace", "rule"] + CELL[1:], columns="policy", values="mean_ret", aggfunc="first")


def headroom_min(P, oracles, rivals):
    rows = []
    for idx, r in P.iterrows():
        To = min(r[o] for o in oracles)
        rv = min(rivals, key=lambda p: r[p])
        rows.append(dict(zip(P.index.names, idx), T_oracle=To, best_rival=rv, T_rival=r[rv], headroom=(r[rv] - To) / r[rv]))
    H = pd.DataFrame(rows)
    return H.loc[H.groupby(CELL).headroom.idxmin()]


def verdict(H):
    hot = []
    for (tr, sl, gp), g in H[H.restore == "cake"].groupby(["trace", "slow", "gpu"]):
        if (g.headroom >= 0.15).sum() >= 2:
            hot.append((tr, sl, gp))
    return ("有看頭" if hot else ("死路" if (H.headroom < 0.05).all() else "可能")), hot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all"])
    a = ap.parse_args()
    rid, ts = os.environ.get("RUN_ID", "adhoc"), datetime.now().astimezone().isoformat(timespec="seconds")
    d = pd.read_csv(os.path.join(OUT, "x1_sweep.csv"))
    post = pd.read_csv(os.path.join(OUT, "x1_sweep_post.csv"))
    hs = pd.read_csv(os.path.join(OUT, "x1_sweep_h.csv"))
    # ---- LRU 一致性
    k = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore"]
    m = d[d.policy == "LRU"].merge(post[post.policy == "LRU"], on=k, suffixes=("", "_post"))
    same = bool((m.mean_ret == m.mean_ret_post).all()) and len(m) == len(d[d.policy == "LRU"])
    print(f"LRU 主 sweep vs post sweep：{len(m)} 格，mean_ret 逐格相同＝{same}")
    out = []
    # ---- FREQ2Q
    P = piv(pd.concat([d, post[post.policy == "FREQ2Q"]], ignore_index=True))
    reg = ["LRU", "LRU2", "ARC", "S3FIFO", "GDSF", "PENS"]
    H0 = headroom_min(P, ["O-BEL", "O-BELS"], reg)
    H1 = headroom_min(P, ["O-BEL", "O-BELS"], reg + ["FREQ2Q"])
    v0, hot0 = verdict(H0)
    v1, hot1 = verdict(H1)
    print("O-ORD 原對手：", v0, len(hot0), "組；加 FREQ2Q：", v1, len(hot1), "組")
    for nm, H in (("O-ORD_registered", H0), ("O-ORD_plus_FREQ2Q", H1)):
        for _, r in H.iterrows():
            out.append(dict(run_id=rid, ts=ts, check=nm, **r.to_dict()))
    fq = P.reset_index()
    fq["FREQ2Q_vs_LRU"] = (fq["LRU"] - fq["FREQ2Q"]) / fq["LRU"]
    fq["FREQ2Q_vs_bestreg"] = (fq[reg].min(axis=1) - fq["FREQ2Q"]) / fq[reg].min(axis=1)
    t = fq[(fq.restore == "cake") & (fq.gpu == "idle")].pivot_table(index=["trace", "rule", "slow"], columns="cpu_GiB",
                                                                    values="FREQ2Q_vs_bestreg")
    print("FREQ2Q 相對原登記最好順序對手的改善 %（Cake、GPU 閒；正＝FREQ2Q 更快）")
    print((t * 100).round(1).to_string())
    c = H1[H1.restore == "cake"].pivot_table(index=["trace", "slow", "gpu"], columns="cpu_GiB", values="headroom")
    print("O-ORD headroom %（對手含 FREQ2Q，每格取 R1／R2 較小，Cake）")
    print((c * 100).round(1).to_string())
    print("加 FREQ2Q 後最好對手是 FREQ2Q 的格數：", int((H1.best_rival == "FREQ2Q").sum()), "/", len(H1))
    # ---- 容量軸：LRU 的平均 CPU 停留時間
    mins = post[(post.policy == "LRU") & (post.restore == "cake") & (post.slow == "ssd") & (post.gpu == "idle")]
    mt = mins.pivot_table(index=["trace", "rule"], columns="cpu_GiB", values="cpu_residence_s_mean") / 60
    print("LRU：session 從回合結束到被降級的平均時間（分鐘）＝CPU 裝得下幾分鐘的歷史")
    print(mt.round(2).to_string())
    for (tr, ru), row in mt.iterrows():
        for cg, v in row.items():
            out.append(dict(run_id=rid, ts=ts, check="lru_residence_min", trace=tr, rule=ru, cpu_GiB=cg, value=float(v),
                            src_run=post.run_id.iloc[0]))
    # ---- 截尾穩健版
    for h in (1800, 3600):
        x = hs[hs.policy.isin([f"O-DROP-H{h}", f"O-NOW-H{h}", f"O-ALL-H{h}"])].copy()
        x["policy"] = x.policy.str.replace(f"-H{h}", "", regex=False)
        dd = pd.concat([d[~d.policy.isin(["O-DROP", "O-NOW", "O-ALL"])], x], ignore_index=True)
        P2 = piv(dd)
        res = {}
        for comp, ors, rv in (("O-DROP", ["O-DROP"], ["LRU", "LRU+NONE", "LRU+2HW", "LRU+2HA"]),
                              ("O-NOW", ["O-NOW"], ["LRU", "LRU+NONE", "LRU+2HW"]),
                              ("O-ORD", ["O-BEL", "O-BELS"], reg),
                              ("O-PF", ["O-PF"], ["LRU", "LRU+RPF"])):
            if not set(ors + rv) <= set(P2.columns) or P2[ors + rv].isna().any().any():
                print(f"H{h} {comp}: 缺資料 NOT_MEASURED")
                continue
            H = headroom_min(P2, ors, rv)
            H["component"] = comp
            res[comp] = H
            for _, r in H.iterrows():
                out.append(dict(run_id=rid, ts=ts, check=f"dead_H{h}", **r.to_dict()))
        allH = pd.concat(res.values(), ignore_index=True)
        hot = []
        for comp, H in res.items():
            for (tr, sl, gp), g in H[H.restore == "cake"].groupby(["trace", "slow", "gpu"]):
                if (g.headroom >= 0.15).sum() >= 2:
                    hot.append((comp, tr, sl, gp))
        vv = "有看頭" if hot else ("死路" if (allH.headroom < 0.05).all() else "可能")
        print(f"=== 截尾穩健版 H={h}s：判定 {vv}；過門檻的組：", sorted(set(h_[0] for h_ in hot)), len(hot))
        for comp in ("O-DROP", "O-NOW"):
            if comp in res:
                c = res[comp][res[comp].restore == "cake"].pivot_table(index=["trace", "slow", "gpu"], columns="cpu_GiB",
                                                                     values="headroom")
                print(f"{comp}（H={h}）headroom %，Cake")
                print((c * 100).round(1).to_string())
    # ---- 兩份 Mooncake trace 是否重疊（X5 提醒）：conversation 的請求在 toolagent 有相同（timestamp, input, output）的比例
    import x1_trace as XT
    rc, _ = XT.load_trace("conversation")
    rt, _ = XT.load_trace("toolagent")
    from collections import Counter
    key = lambda r: (r["timestamp"], r["input_length"], r["output_length"])
    ct = Counter(key(r) for r in rt)
    cc = Counter(key(r) for r in rc)
    n_match = sum(min(v, ct[k]) for k, v in cc.items())
    same_h = 0
    tmap = {}
    for r in rt:
        tmap.setdefault(key(r), []).append(len(r["hash_ids"]))
    for r in rc:
        if key(r) in tmap and len(r["hash_ids"]) in tmap[key(r)]:
            same_h += 1
    print(f"conversation 請求 {len(rc)} 個中，toolagent 有相同 (timestamp, input, output) 的：{n_match}；其中 block 數也相同：{same_h}")
    out.append(dict(run_id=rid, ts=ts, check="trace_overlap", trace="conversation_in_toolagent", value=n_match,
                    n_conversation=len(rc), n_toolagent=len(rt), same_nblocks=same_h))
    pd.DataFrame(out).to_csv(os.path.join(OUT, "x1_posthoc_checks.csv"), index=False)


if __name__ == "__main__":
    main()
