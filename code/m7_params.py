"""m7_params.py — 由 C0、C2 實測導出限速器參數（phase1/05 §2 第 2 步，規則開跑前寫死）

規則：每層每方向取「實測最慢」（C0 排除每組第 0 次暖機），再乘 0.9。
SSD 層用「每個 chunk 一個檔」的持續讀寫（已含開檔與每次 I/O 開銷），所以 c 設 0，
迴歸出的截距只記錄、不再加（避免重複計入）。
"""
import json, sys
import pandas as pd
R = sys.argv[1]
c0 = pd.read_csv(f"{R}/calib_c0.csv"); c0 = c0[c0.kind == "bw"].copy(); c0["v"] = c0.value.astype(float)
c0 = c0[c0.rep.astype(float) > 0]
c2 = pd.read_csv(f"{R}/calib_c2.csv")
GiB = 1 << 30; MiB = 1 << 20
def mn(item): return float(c0[c0.item == item].v.min())
P = {"derate": 0.9, "rule": "min(measured) * 0.9; C0 excludes rep 0 (warm-up); SSD from sustained per-chunk-file I/O, c=0",
     "source_runs": sorted(set(c0.run_id) | set(c2.run_id))}
P["cpu"] = {"read_GiBps": 0.9 * mn("H2D_pinned_chunk_into_kv_view"), "write_GiBps": 0.9 * mn("D2H_pinned_chunk_from_kv_view"), "c_ms": 0.0}
for dev in ("local", "nfs"):
    x = c2[c2.device == dev]
    r = x[x.op == "read_sustained"].MiBps.min(); w = x[x.op == "write_sustained"].MiBps.min()
    import numpy as np
    s = x[x.op == "read1"]; A = np.polyfit(s.size_mib, s.secs, 1)
    P[dev] = {"read_GiBps": 0.9 * r * MiB / GiB, "write_GiBps": 0.9 * w * MiB / GiB, "c_ms": 0.0,
              "measured_min_read_MiBps": r, "measured_min_write_MiBps": w, "regression_intercept_ms_info": A[1] * 1e3}
for k in ("cpu", "local", "nfs"):
    P[k]["ell_chunk_ms"] = 64 * MiB / (P[k]["read_GiBps"] * GiB) * 1e3
json.dump(P, open(f"{R}/tier_params.json", "w"), indent=1)
print(json.dumps(P, indent=1))
