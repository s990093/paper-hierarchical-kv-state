"""h200_nsys_memcpy.py — 把 nsys 的 cuda_gpu_trace（CSV）整理成「H2D 拷貝：極小 vs 大塊」的摘要（13_h200_repro §14 A3；只報告）。

  nsys stats -r cuda_gpu_trace -f csv -o <prefix> <report>.nsys-rep   → <prefix>_cuda_gpu_trace.csv
  python code/h200_nsys_memcpy.py <prefix>_cuda_gpu_trace.csv --label busy --out x.json

每一筆 H2D：大小、在 GPU 上的時間（Duration），和同一個 stream 上一筆 H2D 結束到這一筆開始的間隔（gap）。
極小＝< 64 KiB（Qwen fp8_head 的 scale 896 B）。另數 trace 裡 kernel 的筆數（依名稱），看搬的路上有沒有我們自己的 kernel。
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict

TINY = 64 * 1024


def _col(hdr, *keys):
    for i, h in enumerate(hdr):
        if all(k.lower() in h.lower() for k in keys):
            return i
    raise KeyError(f"沒有欄位 {keys}：{hdr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = list(csv.reader(open(a.csv)))
    hdr, body = rows[0], rows[1:]
    i_st, i_du, i_nm = _col(hdr, "start"), _col(hdr, "duration"), _col(hdr, "name")
    i_by = _col(hdr, "bytes")
    i_sm = _col(hdr, "strm")
    mb = "(mb)" in hdr[i_by].lower()
    h2d = defaultdict(list)
    kern = Counter()
    for r in body:
        nm = r[i_nm]
        if "Host-to-Device" in nm or "HtoD" in nm:
            b = float(r[i_by] or 0) * (1e6 if mb else 1)
            h2d[r[i_sm]].append((float(r[i_st]), float(r[i_du]), b))
        elif "memcpy" not in nm.lower() and "memset" not in nm.lower():
            kern[nm[:80]] += 1
    out = {"label": a.label, "csv": a.csv, "tiny_bytes": TINY, "streams": {}}
    for s, v in h2d.items():
        v.sort()
        tiny = [x for x in v if x[2] < TINY]
        big = [x for x in v if x[2] >= TINY]
        gaps_t = [v[k][0] - (v[k - 1][0] + v[k - 1][1]) for k in range(1, len(v)) if v[k][2] < TINY]
        gaps_b = [v[k][0] - (v[k - 1][0] + v[k - 1][1]) for k in range(1, len(v)) if v[k][2] >= TINY]
        med = lambda x: statistics.median(x) / 1e6 if x else None   # noqa: E731  ns → ms
        out["streams"][s] = dict(n_tiny=len(tiny), n_big=len(big),
                                 tiny_dur_ms_med=med([x[1] for x in tiny]), big_dur_ms_med=med([x[1] for x in big]),
                                 gap_before_tiny_ms_med=med(gaps_t), gap_before_big_ms_med=med(gaps_b),
                                 tiny_sizes=sorted({int(x[2]) for x in tiny})[:8])
    out["kernels_top"] = kern.most_common(15)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
