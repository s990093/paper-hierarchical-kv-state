"""p01_tools.py — ICPE P0-1（docs/h200/icpe/P0-1_recompute_speed.md）的小工具與分析。

  bw-nfs --out-dir D        §3.2：H200 的 speed_<m>/bw.json 複本，只加 targets.nfs（tier_params 的 nfs 讀頻寬）→ D/bw_<m>.json

每個輸出都有 run_id、ts。
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TIER_PARAMS = os.environ.get("M7_TIER_PARAMS") or os.path.join(REPO, "results", "m7_write_policy_mi300x", "tier_params.json")
PREC_H200 = os.path.join(REPO, "results", "m10_precision_h200")


def run_id():
    return os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def cmd_bw_nfs(a):
    tp = json.load(open(TIER_PARAMS))
    os.makedirs(a.out_dir, exist_ok=True)
    for m in ("llama", "qwen"):
        src = os.path.join(PREC_H200, f"speed_{m}", "bw.json")
        d = json.load(open(src))
        assert "nfs" not in d["targets"], src
        d["targets"]["nfs"] = {"b_star": None, "GiBps": float(tp["nfs"]["read_GiBps"]),
                               "source": f"{os.path.relpath(TIER_PARAMS, REPO)} 的 nfs.read_GiBps（03 §R2：限速層兩台都用 MI300X 的）"}
        d["_icpe_p01"] = {"copied_from": os.path.relpath(src, REPO), "run_id": run_id(), "ts": ts(),
                          "note": "只加 targets.nfs；f 與 b16／b32／b48 不動（P0-1 §3.2：限速層頻寬沿用舊值）"}
        out = os.path.join(a.out_dir, f"bw_{m}.json")
        json.dump(d, open(out, "w"), indent=1, ensure_ascii=False)
        print(m, {k: v["GiBps"] for k, v in d["targets"].items()}, "→", out)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("bw-nfs")
    b.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    {"bw-nfs": cmd_bw_nfs}[a.cmd](a)


if __name__ == "__main__":
    main()
