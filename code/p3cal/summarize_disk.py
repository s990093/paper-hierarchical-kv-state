"""summarize_disk.py — /ssd8 讀寫頻寬（disk_bw.py 的輸出＋逐秒 diskstats）整理成 results/p3_calib_3090/ssd8_bw.csv（2026-10-18）。

  python code/p3cal/summarize_disk.py --run /ssd8/hungwei/paper-hkv/runs/<run_id> --out results/p3_calib_3090

輸出：
  ssd8_bw.csv          每次重複一列（disk_bw 的寫／讀 MiB/s）＋這次重複期間裝置總量、扣掉我們之後別人的量、逐秒速度的分位數
  ssd8_bw_persec.csv   逐秒裝置讀寫速度（看有沒有 SLC 懸崖）
  ssd8_bw_summary.json 中位數、GiB/s 換算、別人的 I/O、整段 loadavg
每列有 run_id（記錄殼的目錄名）、disk_bw_run_id（disk_bw.py 自己產生的 id）、ts。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from datetime import datetime

MiB = 1 << 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rid = os.path.basename(a.run.rstrip("/"))
    os.makedirs(a.out, exist_ok=True)
    bw = list(csv.DictReader(open(os.path.join(a.run, "disk_bw.csv"))))
    ds = list(csv.DictReader(open(os.path.join(a.run, "diskstats.csv"))))
    t = [datetime.fromisoformat(x["ts"]).timestamp() for x in ds]
    per = []
    for i in range(1, len(ds)):
        dt = float(ds[i]["t_mono"]) - float(ds[i - 1]["t_mono"])
        per.append({"run_id": rid, "ts": ds[i]["ts"], "t_epoch": t[i],
                    "write_MiBps": round((int(ds[i]["wr_sectors"]) - int(ds[i - 1]["wr_sectors"])) * 512 / MiB / dt, 1),
                    "read_MiBps": round((int(ds[i]["rd_sectors"]) - int(ds[i - 1]["rd_sectors"])) * 512 / MiB / dt, 1),
                    "in_flight": ds[i]["in_flight"], "loadavg1": ds[i]["loadavg1"]})
    with open(os.path.join(a.out, "ssd8_bw_persec.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0].keys()))
        w.writeheader()
        w.writerows(per)
    rows = []
    for r in bw:
        if "error" in r and r["error"]:
            rows.append({"run_id": rid, "disk_bw_run_id": r["run_id"], "ts": r["ts"], "repeat": r["repeat"],
                         "error": r["error"]})
            continue
        te = datetime.fromisoformat(r["ts"]).timestamp()        # disk_bw 在這次重複寫＋讀完之後記 ts
        ws, rs_ = float(r["write_s"]), float(r["read_s"])
        tw0, tw1 = te - rs_ - ws, te - rs_
        tr0, tr1 = te - rs_, te
        # 這段期間裝置的總量（取 diskstats 樣本最接近的兩端，取樣間隔 1 s → 兩端各有 ≤1 s 的誤差）

        def dev_bytes(t0, t1, col):
            i0 = min(range(len(t)), key=lambda i: abs(t[i] - t0))
            i1 = min(range(len(t)), key=lambda i: abs(t[i] - t1))
            return (int(ds[i1][col]) - int(ds[i0][col])) * 512, t[i1] - t[i0]

        own = int(r["size_mib"]) * MiB
        wdev, wspan = dev_bytes(tw0, tw1, "wr_sectors")
        rdev_w, _ = dev_bytes(tw0, tw1, "rd_sectors")
        rdev, rspan = dev_bytes(tr0, tr1, "rd_sectors")
        wdev_r, _ = dev_bytes(tr0, tr1, "wr_sectors")
        pw = [p["write_MiBps"] for p in per if tw0 + 1 <= p["t_epoch"] <= tw1]
        pr = [p["read_MiBps"] for p in per if tr0 + 1 <= p["t_epoch"] <= tr1]
        q = lambda v, f: round(statistics.quantiles(v, n=20)[f], 1) if len(v) >= 20 else ""  # noqa: E731
        rows.append({
            "run_id": rid, "disk_bw_run_id": r["run_id"], "ts": r["ts"], "repeat": r["repeat"], "path": r["path"],
            "device": r["device"], "o_direct": r["o_direct"], "size_mib": r["size_mib"],
            "write_MiBps": r["write_mibps"], "read_MiBps": r["read_mibps"], "write_s": r["write_s"], "read_s": r["read_s"],
            "write_GiBps": round(float(r["write_mibps"]) / 1024, 4), "read_GiBps": round(float(r["read_mibps"]) / 1024, 4),
            "dev_write_MiB_during_write": round(wdev / MiB, 1), "foreign_write_MiB_during_write": round((wdev - own) / MiB, 1),
            "dev_read_MiB_during_write": round(rdev_w / MiB, 1),
            "dev_read_MiB_during_read": round(rdev / MiB, 1), "foreign_read_MiB_during_read": round((rdev - own) / MiB, 1),
            "dev_write_MiB_during_read": round(wdev_r / MiB, 1),
            "persec_write_p05": q(pw, 0), "persec_write_p50": round(statistics.median(pw), 1) if pw else "",
            "persec_write_p95": q(pw, 18), "persec_write_max": max(pw) if pw else "",
            "persec_read_p05": q(pr, 0), "persec_read_p50": round(statistics.median(pr), 1) if pr else "",
            "persec_read_p95": q(pr, 18),
            "host_contention": r["host_contention"], "foreign_gpu_procs": r["foreign_procs"]})
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(a.out, "ssd8_bw.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if "error" not in r]
    summ = {"run_id": rid, "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "device": ok[0]["device"] if ok else None, "size_mib_each": ok[0]["size_mib"] if ok else None,
            "n_ok": len(ok), "n_err": len(rows) - len(ok),
            "write_MiBps": [float(r["write_MiBps"]) for r in ok], "read_MiBps": [float(r["read_MiBps"]) for r in ok],
            "write_MiBps_median": statistics.median(float(r["write_MiBps"]) for r in ok) if ok else None,
            "read_MiBps_median": statistics.median(float(r["read_MiBps"]) for r in ok) if ok else None,
            "foreign_write_MiB_during_write": [r["foreign_write_MiB_during_write"] for r in ok],
            "foreign_read_MiB_during_read": [r["foreign_read_MiB_during_read"] for r in ok],
            "loadavg1_range": [min(float(p["loadavg1"]) for p in per), max(float(p["loadavg1"]) for p in per)],
            "note": "foreign＝裝置總量−我們自己的量（含 ext4 journal/metadata；取樣 1 s，兩端各 ≤1 s 誤差）"}
    if ok:
        summ["write_GiBps_median"] = round(summ["write_MiBps_median"] / 1024, 4)
        summ["read_GiBps_median"] = round(summ["read_MiBps_median"] / 1024, 4)
    json.dump(summ, open(os.path.join(a.out, "ssd8_bw_summary.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
