"""w7_ws.py — 紀錄的「活工作集」WS_ref（W7 的 CPU 容量用它的比例；真實紀錄也可以量，用來對照）。
定義（icpe-rux，W7 偏離 12）：
  - 一個 session 在 [第一個請求的時刻, 最後一個請求的時刻] 之間是「活的」；
  - 時刻 τ 的工作集 WS(τ)＝所有活 session 在 τ 之前（含）最新那個請求的完整 chunk 鍵的聯集大小（共用前綴只算一次）；
    session 的最後一個請求之後它的鍵就不算（不會再回來用）。
  - WS_ref＝WS(τ) 在 [t_0, t_last] 上的時間加權平均 × 64 MiB（Llama-3.1-8B BF16，一塊 512 token＝64 MiB，同主結果）。
  也報 WS 的最大值、時間加權的 p50／p90。只讀 nb 檔，不模擬。
用法：python -I w7_ws.py --nb-dir <dir> --names <name> ... [--out <csv>]"""
import argparse, json, os, socket, sys
from datetime import datetime

import numpy as np

GIB_PER_BLOCK = 64 / 1024


def ws_profile(t, sess, full_off, full_key):
    """回傳 dict（WS_ref_GiB 等）。full_key 是原始鍵（任意 int64），只比相等。"""
    n = len(t)
    last = {}
    for i, s in enumerate(sess.tolist()):
        last[s] = i
    is_last = np.zeros(n, bool)
    is_last[list(last.values())] = True
    ref = {}
    cur = {}
    ws = 0
    acc = 0.0
    segs = []                                  # (時長, WS) 給分位數
    wmax = 0
    for i in range(n):
        if i:
            dt = float(t[i] - t[i - 1])
            if dt > 0:
                acc += dt * ws
                segs.append((dt, ws))
        s = int(sess[i])
        new = full_key[full_off[i]:full_off[i + 1]].tolist()
        old = cur.get(s, ())
        if is_last[i]:
            new_keep = ()
        else:
            new_keep = new
        so, sn = set(old), set(new_keep)
        for k in sn - so:
            c = ref.get(k, 0)
            if c == 0:
                ws += 1
            ref[k] = c + 1
        for k in so - sn:
            c = ref[k] - 1
            if c == 0:
                ws -= 1
                del ref[k]
            else:
                ref[k] = c
        if new_keep:
            cur[s] = new_keep
        else:
            cur.pop(s, None)
        wmax = max(wmax, ws)
    span = float(t[-1] - t[0]) if n else 0.0
    mean = acc / span if span > 0 else float(ws)
    if segs:
        d = np.array(segs, dtype=np.float64)
        o = np.argsort(d[:, 1], kind="stable")
        cw = np.cumsum(d[o, 0]) / d[:, 0].sum()
        p50 = float(d[o, 1][np.searchsorted(cw, 0.5)])
        p90 = float(d[o, 1][np.searchsorted(cw, 0.9)])
    else:
        p50 = p90 = float(ws)
    assert ws == 0 and not ref, "最後應該沒有活的 session"
    return dict(WS_ref_GiB=mean * GIB_PER_BLOCK, WS_p50_GiB=p50 * GIB_PER_BLOCK, WS_p90_GiB=p90 * GIB_PER_BLOCK,
                WS_max_GiB=wmax * GIB_PER_BLOCK, span_s=span, n_req=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb-dir", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows = []
    for nm in a.names:
        z = np.load(os.path.join(a.nb_dir, f"{nm}.npz"), allow_pickle=False)
        r = ws_profile(z["t"], z["sess"], z["full_off"], z["full_key"])
        r = dict(name=nm, **r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        rows.append(r)
    if a.out:
        import csv
        meta = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"), host=socket.gethostname())
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(meta) + list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow(dict(meta, **r))
    print("W7_WS_RESULT", f"n={len(rows)}")


if __name__ == "__main__":
    main()
