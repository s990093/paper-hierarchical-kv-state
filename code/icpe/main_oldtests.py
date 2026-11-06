"""main_oldtests.py — icpe-main：論文裡 Mooncake 的兩個舊測試，照原定義在新、舊上限表上重算並列（使用者 2026-11-06）。

  python -I code/icpe/main_oldtests.py --src <label>=<headroom_components.csv> ... \
      [--src-h3600 <label>=<截尾 H3600 的 headroom_components.csv> ...] --out-dir <dir> --out-prefix <p> [--check-v0 <label>]

兩個舊測試（定義取自 docs/icpe_brief_20261030/make_ceiling_table.py 與 docs/icpe_paper_20261030/EVIDENCE.md 的 B9）：
  T1「各容量的上限」（論文的 50.5%＝conversation、O-PF、SSD 級、CPU 128 GiB）：Cake、rule＝min、GPU 閒／忙取大。
     這裡三個慢層與全部 CPU 容量都輸出；另列 rule＝R1、R2（偏離 7：Mooncake 各自判定）與 O-GPU（新模擬器才有）。
  T2「192 格裡 ≥15% 的格數」（論文的 95 格、截尾 H3600＋只算不飽和的 36 格）：conversation＋toolagent、Cake、rule＝min、
     CPU 128／256／512／1,024、慢層 ssd／b32／nfs、GPU 閒／忙、4 個元件（O-ORD、O-DROP、O-NOW、O-PF）＝192 格；
     「不飽和」＝write_saturated 為 False。另列 R1、R2 與 O-GPU（只報告）。
新模擬器的表只取主時間設定（gpu_write＝wb、gpu_GiB＝111.0625、slow_cap_frac＝0；Cake）。
--check-v0 <label>：那個 label 必須是舊 V0 的表（論文數字的來源），程式核對能重現論文的值，不能就停。
"""
from __future__ import annotations

import argparse
import os
import socket
from datetime import datetime

import pandas as pd

COMP4 = ["O-ORD", "O-DROP", "O-NOW", "O-PF"]
CAP4 = [128.0, 256.0, 512.0, 1024.0]
TRACES = ["conversation", "toolagent"]
THR = 0.15


def load(path):
    d = pd.read_csv(path)
    d["cpu_GiB"] = d.cpu_GiB.astype(float)
    if "gpu_write" in d.columns:                      # 新模擬器：只取主時間設定
        d = d[(d.gpu_write == "wb") & (d.gpu_GiB.astype(float) == 111.0625)]
        if "slow_cap_frac" in d.columns:
            d = d[d.slow_cap_frac.astype(float) == 0.0]
    d = d[(d.restore == "cake") & d.trace.isin(TRACES)].copy()
    d["write_saturated"] = d.write_saturated.astype(str).str.lower() == "true"
    return d


def t1(d, label, trunc):
    comps = COMP4 + (["O-GPU"] if (d.component == "O-GPU").any() else [])
    g = d[d.component.isin(comps)].groupby(["trace", "rule", "component", "slow", "cpu_GiB"])
    out = g.agg(headroom_max_idle_busy=("headroom", "max"), n_gpu=("gpu", "size"),
                any_write_saturated=("write_saturated", "any")).reset_index()
    assert (out.n_gpu == 2).all(), f"{label}：有格子不是剛好 GPU 閒／忙兩列"
    out.insert(0, "trunc", trunc)
    out.insert(0, "source", label)
    out["definition"] = out.apply(lambda r: "old (rule=min, 4 comp)" if r.rule == "min" and r.component in COMP4
                                  else "extra (report only)", axis=1)
    return out


def t2(d, label, trunc):
    rows = []
    sub = d[d.cpu_GiB.isin(CAP4)]
    for rule in ("min", "R1", "R2"):
        for cset, comps in (("4comp", COMP4), ("O-GPU", ["O-GPU"])):
            x = sub[(sub.rule == rule) & sub.component.isin(comps)]
            if not len(x):
                continue
            for filt, y in (("all", x), ("nonsat", x[~x.write_saturated])):
                ge = y[y.headroom >= THR]
                base = dict(source=label, trunc=trunc, rule=rule, comp_set=cset, filter=filt,
                            definition="old" if (rule == "min" and cset == "4comp") else "extra (report only)")
                rows.append(dict(base, group_by="total", group="all", n_cells_universe=len(x), n_cells=len(y), n_ge15=len(ge)))
                for gb in ("cpu_GiB", "slow", "trace", "component"):
                    for k, v in ge.groupby(gb).size().items():
                        rows.append(dict(base, group_by=gb, group=str(k), n_cells_universe=len(x), n_cells=len(y), n_ge15=int(v)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, metavar="LABEL=PATH")
    ap.add_argument("--src-h3600", nargs="*", default=[], metavar="LABEL=PATH")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-prefix", default="oldtests")
    ap.add_argument("--check-v0", default="")
    a = ap.parse_args()
    meta = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"),
                host=socket.gethostname())
    T1, T2, srcs = [], [], []
    for trunc, specs in (("none", a.src), ("h3600", a.src_h3600)):
        for spec in specs:
            lab, path = spec.split("=", 1)
            d = load(path)
            srcs.append(dict(source=lab, trunc=trunc, path=path, src_analysis_run_id=";".join(sorted(map(str, d.run_id.unique())))))
            T1.append(t1(d, lab, trunc))
            T2.append(t2(d, lab, trunc))
    T1, T2, S = pd.concat(T1, ignore_index=True), pd.concat(T2, ignore_index=True), pd.DataFrame(srcs)

    if a.check_v0:     # 論文數字：STORY／make_ceiling_table（50.5%）與 EVIDENCE B9（95／192、36）
        v = T1[(T1.source == a.check_v0) & (T1.trunc == "none") & (T1.trace == "conversation") & (T1.rule == "min")
               & (T1.component == "O-PF") & (T1.slow == "ssd") & (T1.cpu_GiB == 128.0)].headroom_max_idle_busy
        assert len(v) == 1 and round(float(v.iloc[0]) * 100, 1) == 50.5, f"V0 T1 沒有重現 50.5%：{list(v)}"

        def n(trunc, filt, gb="total", grp="all"):
            q = T2[(T2.source == a.check_v0) & (T2.trunc == trunc) & (T2.rule == "min") & (T2.comp_set == "4comp")
                   & (T2["filter"] == filt) & (T2.group_by == gb) & (T2.group == grp)]
            return int(q.n_ge15.iloc[0]) if len(q) else 0
        u = T2[(T2.source == a.check_v0) & (T2.rule == "min") & (T2.comp_set == "4comp") & (T2.group_by == "total") & (T2["filter"] == "all")]
        assert set(u.n_cells_universe) == {192}, f"V0 T2 的格子不是 192：{set(u.n_cells_universe)}"
        exp = {("none", "all", "total", "all"): 95, ("none", "all", "cpu_GiB", "128.0"): 28, ("none", "all", "cpu_GiB", "256.0"): 31,
               ("none", "all", "cpu_GiB", "512.0"): 24, ("none", "all", "cpu_GiB", "1024.0"): 12,
               ("none", "all", "slow", "nfs"): 42, ("none", "all", "slow", "ssd"): 29, ("none", "all", "slow", "b32"): 24,
               ("h3600", "nonsat", "total", "all"): 36, ("h3600", "nonsat", "cpu_GiB", "128.0"): 16,
               ("h3600", "nonsat", "cpu_GiB", "256.0"): 12, ("h3600", "nonsat", "cpu_GiB", "512.0"): 8,
               ("h3600", "nonsat", "cpu_GiB", "1024.0"): 0, ("h3600", "nonsat", "slow", "ssd"): 19,
               ("h3600", "nonsat", "slow", "b32"): 17, ("h3600", "nonsat", "slow", "nfs"): 0}
        bad = {k: (n(*k), want) for k, want in exp.items() if n(*k) != want}
        assert not bad, f"V0 T2 沒有重現 EVIDENCE B9：{bad}"
        print(f"--check-v0 {a.check_v0} PASS：重現 50.5%、95／192（128:28 256:31 512:24 1024:12；nfs 42 ssd 29 b32 24）、"
              f"H3600＋不飽和 36（128:16 256:12 512:8 1024:0；ssd 19 b32 17 nfs 0）")

    os.makedirs(a.out_dir, exist_ok=True)
    for name, X in (("t1_capacity", T1), ("t2_count192", T2), ("sources", S)):
        X = X.copy()
        for i, (k, v) in enumerate(meta.items()):
            X.insert(i, k, v)
        X.to_csv(os.path.join(a.out_dir, f"{a.out_prefix}_{name}.csv"), index=False)
    pd.set_option("display.width", 250)
    h = T1[(T1.trace == "conversation") & (T1.rule == "min") & (T1.component == "O-PF") & (T1.slow == "ssd") & (T1.cpu_GiB == 128.0)]
    print("T1 conversation O-PF SSD 128 GiB rule=min（GPU 閒／忙取大）：")
    print(h[["source", "trunc", "headroom_max_idle_busy", "any_write_saturated"]].to_string(index=False))
    k = T2[(T2.rule == "min") & (T2.comp_set == "4comp") & (T2.group_by == "total")]
    print("T2 192 格（rule=min、4 元件）≥15% 的格數：")
    print(k.pivot_table(index=["source", "trunc"], columns="filter", values="n_ge15", aggfunc="first").to_string())
    print(f"→ {a.out_dir}/{a.out_prefix}_{{t1_capacity,t2_count192,sources}}.csv")


if __name__ == "__main__":
    main()
