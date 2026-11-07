"""next_sim_dev14_check.py — icpe-sim：兩份 x1_sweep 逐列比（偏離 14＋SIM_CHANGES 23 的驗證、xref_dev14 的跨機器比對）。原本在 run 20261107-140640 的 scripts/dev14_check.py，內容不變。"""
"""偏離 14＋SIM_CHANGES 23 的驗證（icpe-sim）：兩份 x1_sweep 逐列比對，按 (規則類別, 雜訊, GPU 開關) 分組數不同的列。
python -I dev14_check.py <ref.csv> <new.csv> <out.csv> <run_id> <tag> <mode>
mode＝all：全部列都要相同；nonopf：不含 opf 的規則、而且（沒有雜訊、模型 M、或 λ＝0）的列要相同，其他只報數字；
pgr（SIM_CHANGES 23，旗標關 vs 只開 --pf-gpu-recheck）：除了「O-ALL、預取到 GPU」以外都要相同（cls 細分成 oall／opf／other）。
比的欄＝兩邊共有、去掉 run_id／ts／code_sha／elapsed_s／pf_expire／noise_overdue／pf_gpu_recheck。"""
import datetime
import sys
import pandas as pd

ref, new, out, rid, tag, mode = sys.argv[1:7]
OPF = {"O-PF", "O-ALL"}
a = pd.read_csv(ref, dtype=str)
b = pd.read_csv(new, dtype=str)
K = [c for c in ["trace", "rule", "cpu_GiB", "policy", "slow", "gpu", "restore", "gpu_GiB", "gpu_write", "gpu_policy", "pf_target",
                 "noise_model", "noise_lambda", "noise_draw", "noise_seed"] if c in a.columns and c in b.columns]
skip = {"run_id", "ts", "code_sha", "elapsed_s", "pf_expire", "noise_overdue", "pf_gpu_recheck", "host"}
cols = [c for c in a.columns if c in b.columns and c not in skip and c not in K]
for d in (a, b):
    d["cpu_GiB"] = d["cpu_GiB"].astype(float).map(repr)
assert not a.duplicated(K).any() and not b.duplicated(K).any(), "鍵重複"
m = a.merge(b, on=K, how="outer", suffixes=("_r", "_n"), indicator=True)
only = int((m["_merge"] != "both").sum())
m = m[m["_merge"] == "both"].copy()
diff = pd.Series(False, index=m.index)
for c in cols:
    diff |= m[c + "_r"].fillna("NA") != m[c + "_n"].fillna("NA")
m["diff"] = diff
m["cls"] = m["policy"].map(lambda p: ("oall" if p == "O-ALL" else "opf") if (mode == "pgr" and p in OPF) else ("opf" if p in OPF else "other"))
nm = m["noise_model"] if "noise_model" in m else pd.Series("none", index=m.index)
lam = m["noise_lambda"].astype(float) if "noise_lambda" in m else pd.Series(0.0, index=m.index)
m["noise"] = [("none" if x in ("none", "NA") or pd.isna(x) else ("lam0" if l == 0 else x)) for x, l in zip(nm, lam)]
m["gpu_on"] = (m["gpu_GiB"].astype(float) > 0) if "gpu_GiB" in m else False
m["pft"] = m["pf_target"] if "pf_target" in m else "cpu"
g = m.groupby(["cls", "noise", "gpu_on", "pft"]).agg(rows=("diff", "size"), diff_rows=("diff", "sum")).reset_index()
if mode == "all":
    g["expect_identical"] = True
elif mode == "pgr":
    g["expect_identical"] = ~((g["cls"] == "oall") & (g["pft"] == "gpu"))
else:
    g["expect_identical"] = (g["cls"] == "other") & g["noise"].isin(["none", "M", "lam0"])
g["ok"] = ~g["expect_identical"] | (g["diff_rows"] == 0)
ts = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
g.insert(0, "tag", tag); g.insert(0, "host", "acvlab"); g.insert(0, "ts", ts); g.insert(0, "run_id", rid)
g["ref"] = ref; g["new"] = new; g["rows_only_one_side"] = only; g["n_cols_compared"] = len(cols)
import os
g.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
print(g.drop(columns=["ref", "new"]).to_string())
ok = bool(g["ok"].all()) and only == 0
print("DEV14", tag, "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
