"""next_d14_patch.py — icpe-data：PREREG 偏離 14 的「第一段」補丁（STATUS_DATA 2026-11-07 16:34）。

同一個 block：原本的 sweep（results/icpe_next/b/<plan>/<block>/x1_sweep.csv.xz）拿掉被重跑的列，換上 d14 的列
（results/icpe_next/b/d14_<plan>/<block>/x1_sweep.csv.xz），寫成一份完整的 sweep，給 next_sim_analyze／main_rowcheck 用。
只讀 CSV，不跑模擬。比對全部用字串（不轉浮點數），所以沿用的列逐位元不動。

規則（任何一條不成立就停，exit 1）：
  * 鍵＝KEY_COLS 裡 header 有的欄；原本與 d14 各自沒有重複的鍵。
  * d14 的鍵 ⊆ 原本的鍵（一對一換掉，不多出任何格子）。
  * 原本裡 policy ∈ 清單 rerun_policies 的列，鍵都要在 d14 裡（該重跑的都重跑了）。
  * 換掉的列數＝d14 的列數；補丁後列數＝原本的列數。
  * 只有清單 action=rerun 的 block 可以補。
輸出的欄＝原本的欄＋d14 多出來的旗標欄（pf_expire、noise_overdue、pf_gpu_recheck）；沿用的列這幾欄留空（＝沒開，也和這些列無關）。
每列保留自己的 run_id、ts、host、code_sha，所以任何一列都能反查是哪個 run 算的。

用法：
  python -I code/icpe/next_d14_patch.py --b-root results/icpe_next/b --list results/icpe_next/sim/dev14_rerun_list.csv \
      --blocks sa_main/sa_N16_s1 [plan/block ...] --out-dir <dir> [--summary <csv>]
  （<dir>/<plan>/<block>/x1_sweep.csv.xz；--summary 每個 block 一列，含 run_id、ts、host）
"""
from __future__ import annotations

import argparse
import csv
import io
import lzma
import os
import socket
import sys
from datetime import datetime

KEY_COLS = ["trace", "rule", "cpu_GiB", "policy", "slow", "gpu", "restore", "gpu_GiB", "gpu_write", "gpu_policy", "pf_target",
            "noise_model", "noise_lambda", "noise_draw", "noise_seed"]
FLAG_COLS = ["pf_expire", "noise_overdue", "pf_gpu_recheck"]


def read(path):
    op = lzma.open if path.endswith(".xz") else open
    with op(path, "rt", newline="") as fh:
        rd = csv.reader(fh)
        hdr = next(rd)
        return hdr, list(rd)


def keyed(hdr, rows, kc, what):
    ix = [hdr.index(c) for c in kc]
    out = {}
    for r in rows:
        k = tuple(r[i] for i in ix)
        if k in out:
            raise SystemExit(f"{what}：重複的鍵 {dict(zip(kc, k))}")
        out[k] = r
    return out


def patch(base_path, d14_path, rerun_policies):
    bh, br = read(base_path)
    dh, dr = read(d14_path)
    kc = [c for c in KEY_COLS if c in bh]
    miss = [c for c in kc if c not in dh]
    if miss:
        raise SystemExit(f"d14 缺鍵欄 {miss}：{d14_path}")
    extra = [c for c in dh if c not in bh]
    if any(c not in FLAG_COLS for c in extra):
        raise SystemExit(f"d14 多出不認得的欄 {extra}：{d14_path}")
    lost = [c for c in bh if c not in dh and c != "host"]
    if lost:
        raise SystemExit(f"d14 少了原本有的欄 {lost}：{d14_path}")
    B = keyed(bh, br, kc, f"原本 {base_path}")
    D = keyed(dh, dr, kc, f"d14 {d14_path}")
    notin = [k for k in D if k not in B]
    if notin:
        raise SystemExit(f"d14 有 {len(notin)} 個鍵不在原本裡，例 {dict(zip(kc, notin[0]))}")
    pi = bh.index("policy")
    need = [k for k, r in B.items() if r[pi] in rerun_policies and k not in D]
    if need:
        raise SystemExit(f"原本有 {len(need)} 列是 rerun_policies 卻沒有 d14 的列，例 {dict(zip(kc, need[0]))}")
    oh = bh + [c for c in FLAG_COLS if c in extra]
    out = []
    n_rep = 0
    for k, r in B.items():
        if k in D:
            d = dict(zip(dh, D[k]))
            if "host" in bh and "host" not in dh:
                d["host"] = ""           # 收進 git 時才加 host；d14 若是直接讀 run 目錄的 CSV 就沒有
            out.append([d.get(c, "") for c in oh])
            n_rep += 1
        else:
            out.append(r + [""] * (len(oh) - len(bh)))
    assert n_rep == len(D) and len(out) == len(B), (n_rep, len(D), len(out), len(B))
    return oh, out, dict(base_rows=len(B), d14_rows=len(D), replaced=n_rep, kept=len(B) - n_rep, key_cols=";".join(kc),
                         flag_cols=";".join(c for c in FLAG_COLS if c in extra))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-root", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--blocks", nargs="+", required=True, help="plan/block（原本的 plan 名稱，不加 d14_）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--summary")
    a = ap.parse_args()
    L = {(r["plan"], r["block"]): r for r in csv.DictReader(open(a.list, newline=""))}
    rid = os.environ.get("RUN_ID", "")
    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    host = socket.gethostname()
    summ = []
    for pb in a.blocks:
        plan, block = pb.split("/", 1)
        r = L.get((plan, block))
        if r is None or r["action"] != "rerun":
            raise SystemExit(f"{pb}：清單裡不是 action=rerun（{r and r['action']}）")
        base = f"{a.b_root}/{plan}/{block}/x1_sweep.csv.xz"
        d14 = f"{a.b_root}/d14_{plan}/{block}/x1_sweep.csv.xz"
        oh, out, st = patch(base, d14, set(r["rerun_policies"].split()))
        dst = f"{a.out_dir}/{plan}/{block}"
        os.makedirs(dst, exist_ok=True)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(oh)
        w.writerows(out)
        with lzma.open(f"{dst}/x1_sweep.csv.xz", "wt", preset=6) as fh:
            fh.write(buf.getvalue())
        print(f"PATCHED {pb} base={st['base_rows']} d14={st['d14_rows']} replaced={st['replaced']} kept={st['kept']} → {dst}", flush=True)
        summ.append(dict(run_id=rid, ts=ts, host=host, plan=plan, block=block, base=base, d14=d14, **st))
    if a.summary:
        new = not os.path.exists(a.summary)
        with open(a.summary, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summ[0]))
            if new:
                w.writeheader()
            w.writerows(summ)
    print(f"D14_PATCH_OK blocks={len(summ)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
