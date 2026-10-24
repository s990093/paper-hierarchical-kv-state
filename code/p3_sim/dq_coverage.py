"""dq_coverage.py — 正式表的解壓常數覆蓋表（04 v4 §9、06 §19）：每個 (欄組, 平台, 模型, 格式, 閒／忙) 是數字（ms／block）還是 NOT_MEASURED。
只讀正式表，不跑模擬。輸出 $RUN_DIR/dq_coverage.csv 與 results/p3_sim/dq_coverage.csv。
  TIARA_NO_GPU=1 bash code/run3090.sh p3sim-dqcov python code/p3_sim/dq_coverage.py
"""
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import formats as FM  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "results", "p3_sim")


def main():
    rows = []
    for dqs in FM.DQ_SETS:
        for plat in FM.DQ_SETS[dqs]:
            for model in ("llama", "qwen"):
                mf = FM.load_format_table(model, plat, "official" if dqs == FM.DEFAULT_DQ_SET else f"official:{dqs}")
                for fid in (1, 2, 3, 4):
                    for g in ("idle", "busy"):
                        v = mf.dq[g][fid]
                        rows.append(dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"), dq_set=dqs, platform=plat,
                                         model=model, fid=fid, fmt=FM.FMT_NAMES[fid], concrete=mf.src["concrete"][FM.FMT_NAMES[fid]], gpu=g,
                                         dq_ms=("NOT_MEASURED" if v is None else round(v * 1e3, 6)), column=mf.src["dq_cols"][0 if g == "idle" else 1],
                                         format_table_sha256=mf.src["sha256"]))
    for d in (os.environ.get("RUN_DIR"), RESULTS):
        if not d:
            continue
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "dq_coverage.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    for dqs in FM.DQ_SETS:
        for plat in FM.DQ_SETS[dqs]:
            for model in ("llama", "qwen"):
                sel = [r for r in rows if r["dq_set"] == dqs and r["platform"] == plat and r["model"] == model]
                nm = [f"{r['fmt']}/{r['gpu']}" for r in sel if r["dq_ms"] == "NOT_MEASURED"]
                print(f"{dqs:12s} {plat:7s} {model:6s} 有值 {len(sel) - len(nm)}/{len(sel)}  NOT_MEASURED：{', '.join(nm) or '無'}")


if __name__ == "__main__":
    main()
