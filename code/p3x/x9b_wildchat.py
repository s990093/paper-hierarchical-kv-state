"""x9b_wildchat.py — X9b 的 T4（I15：結束語規則）與 T8（I2／I19：重新生成／分岔的代理）。WildChat-1M。

判準在 X9 文件 §2.5（2026-10-17T13:41:37Z 寫定）。固定規則與計數，不訓練。
用法： python -I code/p3x/x9b_wildchat.py --out results/p3_explore/x9 --rundir $RUN_DIR --procs 4
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

DATA = "/ssd8/hungwei/paper-hkv/datasets/wildchat-1m/data/train-*.parquet"
RUN_ID = os.environ.get("RUN_ID", "")
# §2.5 T4 寫死的關鍵字清單
KW = ["thank", "thx", "bye", "that's all", "that is all", "no more question",
      "谢谢", "謝謝", "感谢", "感謝", "多谢", "多謝", "再见", "再見", "拜拜",
      "спасибо", "до свидания", "gracias", "merci", "danke", "obrigad", "ありがとう", "감사"]
RX = re.compile("|".join(re.escape(k) for k in KW), re.IGNORECASE)
SHORT = 50
DUPLEN = 30
US = 1_000_000


def ts():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def one_file(path):
    tb = pq.read_table(path, columns=["conversation_hash", "hashed_ip", "conversation"])
    n = tb.num_rows
    conv = tb.column("conversation").combine_chunks()
    parent = pc.list_parent_indices(conv).to_numpy()
    flat = pc.list_flatten(conv)
    role = pc.struct_field(flat, "role").to_pylist()
    content = pc.struct_field(flat, "content").to_pylist()
    mts = pc.struct_field(flat, "timestamp")
    tnull = pc.is_null(mts).to_numpy(zero_copy_only=False)
    tus = pc.cast(pc.fill_null(mts, 0), "int64").to_numpy()
    ip = tb.column("hashed_ip").to_pylist()
    users = defaultdict(list)
    a_first = np.full(n, -1, dtype=np.int64)
    a_last = np.full(n, -1, dtype=np.int64)
    for k in range(len(role)):
        c = int(parent[k])
        if role[k] == "user":
            users[c].append(content[k] or "")
        elif role[k] == "assistant" and not tnull[k]:
            t = int(tus[k])
            if a_first[c] < 0:
                a_first[c] = t
            a_last[c] = t
    # T4 計數
    cnt = dict(n_final=0, n_nonfinal=0, r1_final=0, r1_nonfinal=0, r2_final=0, r2_nonfinal=0)
    fin_r1 = np.zeros(n, dtype=bool)
    fin_r2 = np.zeros(n, dtype=bool)
    dup_a = np.zeros(n, dtype=bool)
    multi = np.zeros(n, dtype=bool)
    first_hash = [""] * n
    for c in range(n):
        us = users.get(c, [])
        if not us:
            continue
        multi[c] = len(us) >= 2
        for j, m in enumerate(us):
            hit = RX.search(m) is not None
            h2 = hit and len(m) <= SHORT
            if j == len(us) - 1:
                cnt["n_final"] += 1
                cnt["r1_final"] += hit
                cnt["r2_final"] += h2
                fin_r1[c] = hit
                fin_r2[c] = h2
            else:
                cnt["n_nonfinal"] += 1
                cnt["r1_nonfinal"] += hit
                cnt["r2_nonfinal"] += h2
        for j in range(1, len(us)):
            if len(us[j]) >= DUPLEN and us[j] == us[j - 1]:
                dup_a[c] = True
                break
        if len(us[0]) >= DUPLEN:
            first_hash[c] = hashlib.sha1(us[0].encode("utf-8", "surrogatepass")).hexdigest()
    has_user = np.array([c in users for c in range(n)])
    return dict(path=os.path.basename(path), n=n, cnt=cnt, ip=ip, a_first=a_first, a_last=a_last,
                fin_r1=fin_r1, fin_r2=fin_r2, dup_a=dup_a, multi=multi, first_hash=first_hash, has_user=has_user)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    assert RUN_ID
    files = sorted(glob.glob(DATA))
    assert len(files) == 14, len(files)
    with ProcessPoolExecutor(a.procs) as ex:
        R = list(ex.map(one_file, files))
    now = ts()
    cnt = defaultdict(int)
    for r in R:
        for k, v in r["cnt"].items():
            cnt[k] += v
    ip = sum((r["ip"] for r in R), [])
    a_first = np.concatenate([r["a_first"] for r in R])
    a_last = np.concatenate([r["a_last"] for r in R])
    fin_r1 = np.concatenate([r["fin_r1"] for r in R])
    fin_r2 = np.concatenate([r["fin_r2"] for r in R])
    dup_a = np.concatenate([r["dup_a"] for r in R])
    multi = np.concatenate([r["multi"] for r in R])
    has_user = np.concatenate([r["has_user"] for r in R])
    fh = sum((r["first_hash"] for r in R), [])
    N = len(ip)
    # 同一個 IP 的對話依第一個 assistant 時間排序
    by_ip = defaultdict(list)
    for c in range(N):
        if ip[c] is not None and a_first[c] >= 0:
            by_ip[ip[c]].append(c)
    came_back = np.zeros(N, dtype=bool)   # 這個對話結束後 1 h 內同 IP 開了新對話
    fork_b = np.zeros(N, dtype=bool)      # 24 h 內同 IP、第一則訊息相同（≥30 字元）的較晚對話
    for _, cs in by_ip.items():
        cs.sort(key=lambda c: a_first[c])
        firsts = np.array([a_first[c] for c in cs])
        for idx, c in enumerate(cs):
            # 換對話再回來：之後第一個開始時間 > a_last[c] 的對話，在 1 h 內
            k = np.searchsorted(firsts, a_last[c], side="right")
            if k < len(cs) and firsts[k] - a_last[c] <= 3600 * US:
                came_back[c] = True
        # 滑動視窗：較早 24 h 內同 IP 有第一則訊息相同的對話 → 這個對話標 fork_b
        win = defaultdict(int)
        left = 0
        for idx, c in enumerate(cs):
            while firsts[left] < a_first[c] - 86400 * US:
                if fh[cs[left]]:
                    win[fh[cs[left]]] -= 1
                left += 1
            if fh[c]:
                if win.get(fh[c], 0) > 0:
                    fork_b[c] = True
                win[fh[c]] += 1
    def prf(fire_final, fire_nonfinal, n_final, n_nonfinal):
        prec = fire_final / max(1, fire_final + fire_nonfinal)
        rec = fire_final / max(1, n_final)
        return prec, rec
    base = cnt["n_final"] / (cnt["n_final"] + cnt["n_nonfinal"])
    rows = []
    for rule in ("R1", "R2"):
        k = rule.lower()
        p, r_ = prf(cnt[f"{k}_final"], cnt[f"{k}_nonfinal"], cnt["n_final"], cnt["n_nonfinal"])
        rows.append(dict(run_id=RUN_ID, ts=now, dataset="wildchat-1m", test="T4", variant="main", rule=rule,
                         n_final=cnt["n_final"], n_nonfinal=cnt["n_nonfinal"], fire_final=cnt[f"{k}_final"],
                         fire_nonfinal=cnt[f"{k}_nonfinal"], base_rate_final=base, precision=p, recall=r_,
                         hit_rate_final=cnt[f"{k}_final"] / cnt["n_final"],
                         hit_rate_nonfinal=cnt[f"{k}_nonfinal"] / cnt["n_nonfinal"]))
        # hashed_ip 版：拿掉「換對話再回來」的最後一輪
        keep = has_user & ~came_back
        fin = fin_r1 if rule == "R1" else fin_r2
        n_final_v = int(keep.sum())
        ff = int((fin & keep).sum())
        p2 = ff / max(1, ff + cnt[f"{k}_nonfinal"])
        rows.append(dict(run_id=RUN_ID, ts=now, dataset="wildchat-1m", test="T4", variant="exclude_same_ip_new_conv_1h",
                         rule=rule, n_final=n_final_v, n_nonfinal=cnt["n_nonfinal"], fire_final=ff,
                         fire_nonfinal=cnt[f"{k}_nonfinal"],
                         base_rate_final=n_final_v / (n_final_v + cnt["n_nonfinal"]), precision=p2,
                         recall=ff / max(1, n_final_v), n_final_removed=int((has_user & came_back).sum())))
    t8 = dict(run_id=RUN_ID, ts=now, dataset="wildchat-1m", test="T8",
              n_conv=N, n_conv_multi_user=int(multi.sum()),
              a_dup_consecutive_user_ge30=int(dup_a.sum()), a_frac_of_multi=float(dup_a.sum() / max(1, multi.sum())),
              b_same_ip_same_first_msg_24h=int(fork_b.sum()), b_frac_of_all=float(fork_b.sum() / N),
              n_conv_with_ip=int(sum(1 for x in ip if x is not None)))
    os.makedirs(a.out, exist_ok=True)
    for fn, rs in (("x9b_t4_closing_rule.csv", rows), ("x9b_t8_regen_proxy.csv", [t8])):
        keys = []
        for r in rs:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(os.path.join(a.out, fn), "w", newline="") as fh_:
            w = csv.DictWriter(fh_, fieldnames=keys)
            w.writeheader()
            w.writerows(rs)
        print(fn, json.dumps(rs, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
