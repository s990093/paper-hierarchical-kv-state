"""w7_map.py — W7 地圖的彙總（PREREG 偏離 12）：只讀每份合成紀錄的上限表（analyze_v8 的 *_headroom_components.csv）與 meta，不模擬。
用法：python -I w7_map.py --design <design.csv> --nb-dir <合成 nb 目錄> --runs-glob '<每份紀錄的上限表 glob>' --out <目錄>
  上限表的 trace 欄＝nb:w7_pPPP_sS；CPU 容量依該紀錄的 WS_ref 換回水準 k（cpu＝max(1, round(WS_ref·2^k·16))/16 GiB，k＝−6..2）。
判定（偏離 12）：
  - 單位＝(點, 水準 k, 慢層, GPU 閒／忙, 元件)，只用 cake；上限取 3 個 seed 的中位數（另報最小、最大）；中位數 ≥ 0.15 → 在區域內。
  - 算判定的元件＝O-ORD、O-DROP、O-NOW、O-PF、O-GPU（同偏離 7）；O-ALL(report) 只報告。
  - 每個元件：區域內的比例；比例在 [0.05, 0.95] → 「可以加密一輪（要先再登記）」。
  - 各維度和上限的關係只報告：點層級（每點取各單位上限中位數的最大值）和 9 個維度＋WS_ref／GPU 的 Spearman 等級相關。
輸出 w7_cells.csv（每份紀錄每格）、w7_map.csv（跨 seed）、w7_summary.csv、w7_dims.csv（每列有 run_id、ts、host）；最後一行 W7_MAP_RESULT。"""
import argparse, glob, json, os, socket
from datetime import datetime

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--design", required=True)
ap.add_argument("--nb-dir", required=True)
ap.add_argument("--runs-glob", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
a = ap.parse_args()
GPU_GIB = 111.0625
LEVELS = list(range(-6, 3))
JUDGE = ["O-ORD", "O-DROP", "O-NOW", "O-PF", "O-GPU"]
D = pd.read_csv(a.design)
H = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(a.runs_glob))], ignore_index=True)
H = H[(H.restore == "cake")].copy()
H = H.rename(columns={k: f"analyze_{k}" for k in ("run_id", "ts", "host") if k in H.columns})   # analyze_v8 自己的出處欄
H["name"] = H.trace.str.replace("nb:", "", regex=False)
H["point"] = H.name.str.extract(r"w7_p(\d+)_s\d+")[0].astype(int)
H["seed"] = H.name.str.extract(r"w7_p\d+_s(\d+)")[0].astype(int)
ws = {}
for nm in H.name.unique():
    ws[nm] = json.load(open(os.path.join(a.nb_dir, f"{nm}.meta.json")))["WS_ref_GiB"]
H["WS_ref_GiB"] = H.name.map(ws)
H["cpu_GiB"] = H.cpu_GiB.astype(float)


def level_of(r):
    for k in LEVELS:
        if max(1, round(r.WS_ref_GiB * 2.0 ** k * 16)) / 16 == r.cpu_GiB:
            return k
    return np.nan


H["level"] = H.apply(level_of, axis=1)
# 不同水準可能因為取整撞到同一個容量（WS 很小時）→ 那格算在每個對到的水準都不對；先檢查
assert H.level.notna().all(), "有容量對不回水準"
H["ws_over_gpu"] = H.WS_ref_GiB / GPU_GIB
meta = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"), host=socket.gethostname())
os.makedirs(a.out, exist_ok=True)


def put(X, nm):
    X = X.copy()
    for i, (k, v) in enumerate(meta.items()):
        X.insert(i, k, v)
    X.to_csv(os.path.join(a.out, nm), index=False)


DIMS = ["turns_mean", "gap_family", "gap_mean_s", "tool_share", "init_ctx_tokens", "inc_tokens", "sys_tokens", "fork_frac", "N"]
H = H.merge(D[["point"] + DIMS], on="point", how="left")
put(H, "w7_cells.csv")
U = ["point", "level", "slow", "gpu", "component"]
M = H.groupby(U).agg(n_seeds=("headroom", "size"), hr_med=("headroom", "median"), hr_min=("headroom", "min"), hr_max=("headroom", "max"),
                     ws_over_gpu_med=("ws_over_gpu", "median")).reset_index()
M["complete"] = M.n_seeds == len(a.seeds)
M["in_region"] = M.complete & (M.hr_med >= 0.15)
M = M.merge(D[["point"] + DIMS], on="point", how="left")
put(M, "w7_map.csv")
S = []
for c, g in M[M.complete].groupby("component"):
    share = float(g.in_region.mean())
    S.append(dict(component=c, judged=c in JUDGE, n_units=len(g), n_in_region=int(g.in_region.sum()), share_in_region=share,
                  densify_allowed=bool(c in JUDGE and 0.05 <= share <= 0.95), hr_med_max=float(g.hr_med.max()),
                  seed_range_med=float((g.hr_max - g.hr_min).median())))
S = pd.DataFrame(S)
put(S, "w7_summary.csv")
L = M[M.complete].groupby(["component", "level"]).in_region.mean().unstack()


def spearman(x, y):
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    return float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) > 0 and np.std(ry) > 0 else float("nan")


P = M[M.complete].groupby(["point", "component"]).hr_med.max().reset_index()
P = P.merge(D[["point"] + DIMS], on="point").merge(M.groupby("point").ws_over_gpu_med.first().reset_index(), on="point")
P["gap_family_code"] = P.gap_family.map({"exp": 0, "lognormal": 1, "lomax": 2})
R = []
for c, g in P.groupby("component"):
    for d in [x for x in DIMS if x != "gap_family"] + ["gap_family_code", "ws_over_gpu_med"]:
        R.append(dict(component=c, dim=d, spearman_point_max=spearman(g[d], g.hr_med)))
R = pd.DataFrame(R)
put(R, "w7_dims.csv")
pd.set_option("display.width", 250)
print(S.round(4).to_string(index=False))
print("各水準在區域內的比例：")
print(L.round(3).to_string())
print("Spearman（點層級最大上限 vs 維度）：")
print(R.pivot(index="dim", columns="component", values="spearman_point_max").round(2).to_string())
print("W7_MAP_RESULT", f"traces={H.name.nunique()} units={len(M)} complete={int(M.complete.sum())}")
