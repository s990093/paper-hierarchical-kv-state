"""main_d14patch.py — icpe-main：把偏離 14 重跑的列（d14 block）換進原本的 sweep，給正式上限表用（icpe-data 的 next_d14_patch.py 的獨立實作，
兩邊的輸出由 icpe-main 逐值比對）。不解析數值，csv 模組逐字處理。

  python -I code/icpe/main_d14patch.py --list results/icpe_next/sim/dev14_rerun_list.csv --root results/icpe_next/b \
      --plan <plan> [<plan> ...] [--block <block> ...] --out-dir <dir>
  → <dir>/<plan>/<block>/x1_sweep.csv（補好的）＋ <dir>/d14patch_manifest.csv

規則（icpe-data 2026-11-07 16:4x 的訊息，資料都在 git）：
  - 清單裡 action＝rerun 的列才補；用 rerun_policies（空白分隔）。
  - 原本：<root>/<plan>/<block>/x1_sweep.csv.xz；重跑：<root>/d14_<plan>/<block>/x1_sweep.csv.xz（同名 block，plan 前加 d14_）。
  - 鍵＝[trace, rule, cpu_GiB, policy, slow, gpu, restore, gpu_GiB, gpu_write, gpu_policy, pf_target, noise_model, noise_lambda,
    noise_draw, noise_seed] 裡原本表頭有的欄；字串比較，不轉成數字。
  - 原本的列，鍵出現在 d14 的就換成 d14 那一列；其他列逐字保留。
  - 欄：原本的欄＋d14 檔有的 pf_expire、noise_overdue、pf_gpu_recheck（保留的列這幾欄留空）。每列保留自己的 run_id、ts、host、code_sha。
檢查（不過就 exit 1）：兩邊都沒有重複鍵；d14 的鍵 ⊆ 原本的鍵；原本 policy ∈ rerun_policies 的列都有 d14 列；換掉的列數＝d14 列數；
輸出列數＝原本列數；d14 的每一欄（除了新增的旗標欄）都在原本表頭裡；重讀輸出逐列核對（保留的列＝原本、換掉的列＝d14）。
清單裡 action≠rerun 的 block 不碰（只記在 manifest）。
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

KEY_ALL = ["trace", "rule", "cpu_GiB", "policy", "slow", "gpu", "restore", "gpu_GiB", "gpu_write", "gpu_policy", "pf_target",
           "noise_model", "noise_lambda", "noise_draw", "noise_seed"]
NEW_FLAGS = ["pf_expire", "noise_overdue", "pf_gpu_recheck"]


def opentext(p):
    return io.TextIOWrapper(lzma.open(p), encoding="utf-8", newline="") if p.endswith(".xz") else open(p, encoding="utf-8", newline="")


def readall(p):
    with opentext(p) as f:
        r = csv.reader(f)
        h = next(r)
        rows = list(r)
    for i, x in enumerate(rows):
        if len(x) != len(h):
            die(f"{p} 第 {i + 2} 列 {len(x)} 欄、表頭 {len(h)} 欄")
    return h, rows


def die(msg):
    print(f"停：{msg}", file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--plan", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--block", nargs="*", default=[], help="只補這些 block（同一個 plan 裡有別的家族、還沒跑完時用）；沒給＝plan 裡全部")
    a = ap.parse_args()
    meta = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"),
                host=socket.gethostname())
    with open(a.list, encoding="utf-8", newline="") as f:
        L = list(csv.DictReader(f))
    for c in ("plan", "block", "action", "rerun_policies"):
        if c not in L[0]:
            die(f"{a.list} 沒有 {c} 欄（欄：{list(L[0])}）")
    man = []
    for plan in a.plan:
        todo = [r for r in L if r["plan"] == plan and (not a.block or r["block"] in a.block)]
        if not todo:
            die(f"清單裡沒有 plan={plan}" + (f" 的 block {a.block}" if a.block else ""))
        for r in todo:
            blk = r["block"]
            if r["action"] != "rerun":
                man.append(dict(meta, plan=plan, block=blk, action=r["action"], base_rows="", d14_rows="", replaced="", out=""))
                continue
            pol = set(r["rerun_policies"].split())
            bp = os.path.join(a.root, plan, blk, "x1_sweep.csv.xz")
            dp = os.path.join(a.root, f"d14_{plan}", blk, "x1_sweep.csv.xz")
            for p in (bp, dp):
                if not os.path.exists(p):
                    die(f"找不到 {p}")
            bh, br = readall(bp)
            dh, dr = readall(dp)
            key = [c for c in KEY_ALL if c in bh]
            miss = [c for c in key if c not in dh]
            if miss:
                die(f"{dp} 沒有鍵欄 {miss}")
            extra = [c for c in dh if c not in bh and c not in NEW_FLAGS]
            if extra:
                die(f"{dp} 有原本表頭沒有的欄 {extra}（不是 {NEW_FLAGS}）")
            add = [c for c in NEW_FLAGS if c in dh and c not in bh]
            oh = bh + add
            bk = [tuple(x[bh.index(c)] for c in key) for x in br]
            dk = [tuple(x[dh.index(c)] for c in key) for x in dr]
            if len(set(bk)) != len(bk):
                die(f"{bp} 有重複鍵")
            if len(set(dk)) != len(dk):
                die(f"{dp} 有重複鍵")
            dmap = dict(zip(dk, dr))
            bset = set(bk)
            notin = [k for k in dk if k not in bset]
            if notin:
                die(f"{dp} 有 {len(notin)} 個鍵不在原本裡；例 {notin[:2]}")
            ip = bh.index("policy")
            need = [k for k, x in zip(bk, br) if x[ip] in pol and k not in dmap]
            if need:
                die(f"{bp} 有 {len(need)} 列 policy ∈ rerun_policies 但 d14 沒有；例 {need[:2]}")
            out_rows, nrep = [], 0
            for k, x in zip(bk, br):
                if k in dmap:
                    y = dmap[k]
                    out_rows.append([y[dh.index(c)] for c in oh])
                    nrep += 1
                else:
                    out_rows.append(x + [""] * len(add))
            if nrep != len(dr) or len(out_rows) != len(br):
                die(f"{blk}：換掉 {nrep}、d14 {len(dr)}、輸出 {len(out_rows)}、原本 {len(br)}")
            od = os.path.join(a.out_dir, plan, blk)
            os.makedirs(od, exist_ok=True)
            op = os.path.join(od, "x1_sweep.csv")
            with open(op, "w", encoding="utf-8", newline="") as g:
                w = csv.writer(g, lineterminator="\n")
                w.writerow(oh)
                w.writerows(out_rows)
            h2, r2 = readall(op)                                   # 重讀核對
            assert h2 == oh and len(r2) == len(br)
            for k, x, y in zip(bk, br, r2):
                want = [dmap[k][dh.index(c)] for c in oh] if k in dmap else x + [""] * len(add)
                assert y == want, f"{op}：鍵 {k} 重讀和預期不同"
            man.append(dict(meta, plan=plan, block=blk, action="rerun", base_rows=len(br), d14_rows=len(dr), replaced=nrep, out=op))
            print(f"{plan}/{blk}：原本 {len(br)} 列，換掉 {nrep}（policy ∈ {sorted(pol)}），新增欄 {add}")
    os.makedirs(a.out_dir, exist_ok=True)
    mp = os.path.join(a.out_dir, "d14patch_manifest.csv")
    with open(mp, "w", encoding="utf-8", newline="") as g:
        dw = csv.DictWriter(g, fieldnames=list(man[0]), lineterminator="\n")
        dw.writeheader()
        dw.writerows(man)
    n = sum(1 for m in man if m["action"] == "rerun")
    print(f"檢查通過：{n} 個 block 補好（另 {len(man) - n} 個清單上不是 rerun）→ {mp}")


if __name__ == "__main__":
    main()
