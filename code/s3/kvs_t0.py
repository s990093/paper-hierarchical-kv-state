#!/usr/bin/env python3
"""kvs_t0.py — T0：共享者多久來一個？（Mooncake trace 統計，CPU，只報告）

登記：docs/kv_sharing_hiding_20261026/runs/T0_arrivals.md（判準、定義、單位檢查都在那裡；這支只照做）。

  python code/s3/kvs_t0.py run --traces conversation,toolagent,synthetic
  python code/s3/kvs_t0.py --self-test

單位（CLAUDE.md §1 規則 6）：
  - hash_ids 一個＝512 token：沿用 m8_trace_oracle.load_trace 的斷言（len(hash_ids)==ceil(input_length/512) 吻合率 ≥0.95、前綴鏈）。
  - timestamp 當毫秒；用登記 §3 的單位檢查（重現 Strata 的 38% 量級）確認，不過就 exit 4、不寫 summary。
輸出：results/s3_h200/kvs_t0/t0_summary.csv（run_id、ts）；F0 用的群到達清單放 run 目錄。
"""
from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
REPO = os.path.dirname(CODE)
sys.path.insert(0, CODE)

OUT = os.path.join(REPO, "results", "s3_h200", "kvs_t0")
BLOCK_TOK = 512
MiB = 1 << 20
GiB = 1 << 30
TS_PER_S = 1000.0                                  # timestamp 當毫秒（§3 檢查）
SYS_FRAC = 0.5                                     # §2.1：≥50% 的請求共用的前 d 塊＝系統 prompt
S_TOKENS = (2048, 6144, 16384)                     # d_S＝4、12、32 塊
WINDOWS = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 60.0)
UNIT_CHECK = dict(trace="toolagent", S=6144, w=1.0, lo=0.19, hi=0.76)   # Strata 38% 的 0.5–2 倍
LLAMA_CHUNK = 64 * MiB                             # Llama-3.1-8B 每 512 token
FIELDS = ["run_id", "ts", "trace", "metric", "S_tokens", "sys_mode", "key", "value", "unit", "note"]


def run_id() -> str:
    return os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-kvs-t0-adhoc"


def ts() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def run_dir() -> str:
    d = os.environ.get("RUN_DIR") or os.path.join(os.environ.get("TIARA_RUNS", os.path.join(REPO, "_big", "runs")), run_id())
    os.makedirs(d, exist_ok=True)
    return d


# ============================================================================ 頻寬（登記 §2.3）
def bw_table() -> dict:
    """{(平台, tier): GiB/s}。全部從檔案讀；找不到就丟例外（不猜）。"""
    out = {}
    for plat in ("h200", "mi300x"):
        bw = json.load(open(os.path.join(REPO, f"results/m10_precision_{plat}/speed_llama/bw.json")))
        out[(plat, "cpu")] = float(bw["real_pinned_h2d_GiBps"])
    tp = json.load(open(os.path.join(REPO, "results/m7_write_policy_mi300x/tier_params.json")))
    out[("mi300x", "nvme")] = float(tp["local"]["read_GiBps"])
    out[("h200", "nfs")] = out[("mi300x", "nfs")] = float(tp["nfs"]["read_GiBps"])
    rows = [r for r in csv.DictReader(open(os.path.join(REPO, "results/s3_h200/gfio/gfio_20261026-104426-s3-gfio.csv")))
            if r["dir"] == "/tmp/hungweilai-s3/gfio" and r["pattern"] == "seq" and r["io_bytes"] == str(64 * MiB) and r["threads"] == "8"]
    if len(rows) != 1:
        raise RuntimeError(f"G-fio 的 /tmp seq 64MiB x8 應該剛好一列，找到 {len(rows)} 列")
    out[("h200", "nvme")] = float(rows[0]["gibps"])
    return out


# ============================================================================ trace 結構
def sys_blocks(rows: list, frac: float = SYS_FRAC) -> int:
    """最大的 d：≥frac 的請求前 d 個 hash 完全相同（前綴鏈已驗證 → 看第 d−1 個 hash 就夠）。"""
    n = len(rows)
    d = 0
    while True:
        c = Counter(r["hash_ids"][d] for r in rows if len(r["hash_ids"]) > d)
        if not c or c.most_common(1)[0][1] < frac * n:
            return d
        d += 1


def groups_at(rows: list, depth: int) -> dict:
    """深度 depth（塊）的群：第 depth−1 個 hash 相同的請求 → [(t 秒, 總塊數, 列號)]，依時間排序。"""
    g = defaultdict(list)
    for k, r in enumerate(rows):
        h = r["hash_ids"]
        if len(h) >= depth:
            g[h[depth - 1]].append((r["timestamp"] / TS_PER_S, len(h), k))
    for v in g.values():
        v.sort()
    return g


def frac_neighbor(groups: dict, n_req: int, w: float) -> tuple[float, float]:
    """（±w 內有同群的請求比例，只看之後 w 內有同群的比例）；分母＝trace 的全部請求。"""
    both = after = 0
    for v in groups.values():
        t = [x[0] for x in v]
        for i, ti in enumerate(t):
            prev_ok = i > 0 and ti - t[i - 1] <= w
            next_ok = i + 1 < len(t) and t[i + 1] - ti <= w
            both += prev_ok or next_ok
            after += next_ok
    return both / n_req, after / n_req


def gaps(groups: dict) -> tuple[list, list]:
    g, sizes = [], []
    for v in groups.values():
        sizes.append(len(v))
        g += [v[i + 1][0] - v[i][0] for i in range(len(v) - 1)]
    return g, sizes


def episodes(groups: dict, W_of) -> tuple[list, int, int]:
    """登記 T0-c：還原一完成就丟（最保守的留存）。在途時到的＝跟隨者；否則是領頭、開一次還原（W_of(塊數) 秒）。
    回傳（每個領頭的跟隨者數 N_W、跟隨者總數、群內請求總數）。"""
    nw, foll, tot = [], 0, 0
    for v in groups.values():
        until = -math.inf
        for t, nb, _ in v:
            tot += 1
            if t <= until:
                nw[-1] += 1
                foll += 1
            else:
                until = t + W_of(nb)
                nw.append(0)
    return nw, foll, tot


def rates(groups: dict, W_of) -> tuple[list, list]:
    """每群（≥2 個請求、時間跨度 >0）的到達率 λ 與 N＝λ·W（W 用該群塊數的中位數）。"""
    lam, nn = [], []
    for v in groups.values():
        if len(v) < 2 or v[-1][0] <= v[0][0]:
            continue
        lam_g = (len(v) - 1) / (v[-1][0] - v[0][0])
        nb = sorted(x[1] for x in v)[len(v) // 2]
        lam.append(lam_g)
        nn.append(lam_g * W_of(nb))
    return lam, nn


def pct(v: list, ps=(10, 25, 50, 75, 90, 99)) -> dict:
    if not v:
        return {f"p{p}": "NOT_MEASURED" for p in ps} | {"n": 0, "mean": "NOT_MEASURED", "max": "NOT_MEASURED"}
    s = sorted(v)
    n = len(s)

    def q(p):
        x = p / 100 * (n - 1)
        lo = int(x)
        hi = min(lo + 1, n - 1)
        return s[lo] + (s[hi] - s[lo]) * (x - lo)
    return {f"p{p}": round(q(p), 6) for p in ps} | {"n": n, "mean": round(sum(s) / n, 6), "max": round(s[-1], 6)}


# ============================================================================ 一份 trace
def analyze(name: str, rows: list, meta: dict, bw: dict, emit, dump_dir: str | None) -> dict:
    n = len(rows)
    sysb = sys_blocks(rows)
    emit(trace=name, metric="n_req", value=n, unit="requests", note="load_trace 的驗證見 note2", key="",
         S_tokens="", sys_mode="")
    emit(trace=name, metric="sys_blocks", value=sysb, unit="blocks(512 tok)", note=f"≥{SYS_FRAC:.0%} 的請求共用", key="",
         S_tokens="", sys_mode="")
    emit(trace=name, metric="load_meta", value=json.dumps(meta, ensure_ascii=False), unit="", note="m8_trace_oracle.load_trace",
         key="", S_tokens="", sys_mode="")
    check = {}
    for S in S_TOKENS:
        dS = math.ceil(S / BLOCK_TOK)
        for mode, depth in (("excl_sys", sysb + dS), ("raw", dS)):
            G = groups_at(rows, depth)
            for w in WINDOWS:
                both, after = frac_neighbor(G, n, w)
                emit(trace=name, metric="T0a_frac_neighbor_pm_w", S_tokens=S, sys_mode=mode, key=f"w={w}", value=both, unit="frac of all req")
                emit(trace=name, metric="T0a_frac_neighbor_after_w", S_tokens=S, sys_mode=mode, key=f"w={w}", value=after, unit="frac of all req")
                if name == UNIT_CHECK["trace"] and S == UNIT_CHECK["S"] and w == UNIT_CHECK["w"]:
                    check[(mode, "pm")] = both
                    check[(mode, "after")] = after
            if mode != "excl_sys":
                continue
            g, sizes = gaps(G)
            for k, v in pct(g).items():
                emit(trace=name, metric="T0b_gap_s", S_tokens=S, sys_mode=mode, key=k, value=v, unit="s")
            for k, v in pct(sizes).items():
                emit(trace=name, metric="T0b_group_size", S_tokens=S, sys_mode=mode, key=k, value=v, unit="requests")
            for (plat, tier), gibps in sorted(bw.items()):
                W_of = (lambda nb, b=gibps: nb * LLAMA_CHUNK / (b * GiB))
                nw, foll, tot = episodes(G, W_of)
                for k, v in pct(nw).items():
                    emit(trace=name, metric="T0c_followers_per_restore", S_tokens=S, sys_mode=mode, key=f"{plat}/{tier}/{k}", value=v,
                         unit="requests", note=f"W＝塊數×64MiB÷{gibps} GiB/s")
                emit(trace=name, metric="T0c_frac_followers", S_tokens=S, sys_mode=mode, key=f"{plat}/{tier}",
                     value=foll / n, unit="frac of all req", note=f"群內 {tot} 個請求")
                lam, nn = rates(G, W_of)
                for k, v in pct(nn).items():
                    emit(trace=name, metric="T0d_lambdaW", S_tokens=S, sys_mode=mode, key=f"{plat}/{tier}/{k}", value=v, unit="requests")
            for k, v in pct(rates(G, lambda nb: 0.0)[0]).items():
                emit(trace=name, metric="T0d_lambda", S_tokens=S, sys_mode=mode, key=k, value=v, unit="req/s")
            if dump_dir:
                keep = [dict(times=[x[0] for x in v], nblocks=[x[1] for x in v]) for v in G.values() if len(v) >= 2]
                p = os.path.join(dump_dir, f"t0_groups_{name}_S{S}.json.gz")
                with gzip.open(p, "wt") as fh:
                    json.dump(dict(trace=name, S_tokens=S, sys_blocks=sysb, depth=sysb + dS, ts_per_s=TS_PER_S, groups=keep), fh)
                emit(trace=name, metric="groups_file", S_tokens=S, sys_mode=mode, key="", value=p, unit="path",
                     note=f"{len(keep)} 個 ≥2 的群（F0 trace 版的輸入）")
    return dict(sys_blocks=sysb, check=check)


def unit_check(results: dict) -> tuple[bool, dict]:
    c = results.get(UNIT_CHECK["trace"], {}).get("check", {})
    if not c:
        return False, {"reason": f"沒有跑 {UNIT_CHECK['trace']}"}
    inside = {f"{m}/{k}": (UNIT_CHECK["lo"] <= v <= UNIT_CHECK["hi"]) for (m, k), v in c.items()}
    vals = {f"{m}/{k}": round(v, 4) for (m, k), v in c.items()}
    return any(inside.values()), dict(values=vals, inside=inside, rule=UNIT_CHECK)


# ============================================================================ 指令
def cmd_run(a):
    import m8_trace_oracle as M8
    if a.trace_dir:
        M8.TRACE_DIR = a.trace_dir
    rid, rd = run_id(), run_dir()
    bw = bw_table()
    rows_out = []

    def emit(**kw):
        kw.update(run_id=rid, ts=ts())
        rows_out.append({f: kw.get(f, "") for f in FIELDS})
    for (plat, tier), v in sorted(bw.items()):
        emit(trace="-", metric="bandwidth", key=f"{plat}/{tier}", value=v, unit="GiB/s", note="登記 §2.3 的檔案")
    res = {}
    for name in a.traces.split(","):
        rows, meta = M8.load_trace(name)
        print(f"[t0] {name}: {meta}", flush=True)
        res[name] = analyze(name, rows, meta, bw, emit, rd)
    ok, info = unit_check(res)
    emit(trace=UNIT_CHECK["trace"], metric="unit_check", value="PASS" if ok else "FAIL", unit="",
         note=json.dumps(info, ensure_ascii=False))
    print(f"[t0] 單位檢查：{'PASS' if ok else 'FAIL'} {info}", flush=True)
    raw = os.path.join(rd, "t0_summary.csv")
    with open(raw, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows_out)
    if not ok:
        print(f"[t0] 單位檢查不過 → 停（規則 6）；完整輸出只在 {raw}，不寫 results", file=sys.stderr)
        sys.exit(4)
    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "t0_summary.csv")
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows_out)
    print(f"[t0] → {dst}（{len(rows_out)} 列）", flush=True)


# ============================================================================ 自測（合成 trace，只驗邏輯）
def self_test() -> bool:
    ok = True

    def check(name, cond):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name)
        ok &= bool(cond)

    # 合成：系統 prompt 2 塊（hash 1、2）全部共用；群 A＝之後 12 塊相同（hash 100..111），群 B 另一組
    def req(t_ms, tail, n_extra=0, base=(1, 2)):
        h = list(base) + list(tail) + [10_000 + t_ms * 10 + i for i in range(n_extra)]
        return dict(timestamp=t_ms, input_length=len(h) * BLOCK_TOK - 7, output_length=10, hash_ids=h)
    A = list(range(100, 112))
    B = list(range(200, 212))
    rows = [req(0, A, 3), req(400, A, 2), req(3000, A, 1), req(500, B, 0), req(9000, B, 4), req(20000, [300 + i for i in range(12)], 0),
            req(30000, [400 + i for i in range(12)], 0)]     # A＝3/7 < 50%：不能被當成系統 prompt
    rows.sort(key=lambda r: r["timestamp"])
    check("sys_blocks＝2", sys_blocks(rows) == 2)
    half = rows[:6]                                   # A＝3/6＝50%：照登記「≥50%」會被算進系統 prompt（邊界照定義）
    check("sys_blocks 的邊界：剛好 50% 也算（≥）", sys_blocks(half) == 2 + 12)
    G = groups_at(rows, 2 + 12)
    sizes = sorted(len(v) for v in G.values())
    check("群大小＝[1, 1, 2, 3]", sizes == [1, 1, 2, 3])
    both, after = frac_neighbor(G, len(rows), 1.0)
    check("±1 s：A 的前兩個互為鄰居 → 2/7", abs(both - 2 / 7) < 1e-12 and abs(after - 1 / 7) < 1e-12)
    both10, _ = frac_neighbor(G, len(rows), 10.0)
    check("±10 s：A 三個＋B 兩個 → 5/7", abs(both10 - 5 / 7) < 1e-12)
    raw = groups_at(rows, 12)                       # 不扣系統 prompt：深度 12 → hash_ids[11]
    check("raw 深度 12 的群和扣掉系統 prompt 的群不同（深度不同）", raw.keys() != G.keys())
    g, _ = gaps(G)
    check("群內間隔＝{0.4, 2.6, 8.5}", sorted(round(x, 6) for x in g) == [0.4, 2.6, 8.5])
    # 還原 1 秒：A 在 t=0 開，0.4 跟上；3.0 另開；B 0.5 開、9.0 另開；C、D 各一個 → N_W＝[1,0,0,0,0,0]
    nw, foll, tot = episodes(G, lambda nb: 1.0)
    check("episodes：N_W 排序後＝[0,0,0,0,0,1]、跟隨者 1、群內 7", sorted(nw) == [0, 0, 0, 0, 0, 1] and foll == 1 and tot == 7)
    nw5, foll5, _ = episodes(G, lambda nb: 5.0)
    check("還原 5 秒：A 的 0.4、3.0 都跟上 → 跟隨者 2", foll5 == 2)
    lam, nn = rates(G, lambda nb: 2.0)
    check("到達率：A＝2/3 req/s、B＝1/8.5 req/s；N＝λ·2", sorted(round(x, 6) for x in lam) == sorted([round(2 / 3, 6), round(1 / 8.5, 6)])
          and sorted(round(x, 6) for x in nn) == sorted([round(4 / 3, 6), round(2 / 8.5, 6)]))
    p = pct([1, 2, 3, 4])
    check("pct：p50＝2.5", p["p50"] == 2.5 and p["n"] == 4)
    check("pct：空 → NOT_MEASURED", pct([])["p50"] == "NOT_MEASURED")
    # 單位檢查
    good = {"toolagent": {"check": {("excl_sys", "pm"): 0.30, ("raw", "pm"): 0.9}}}
    bad = {"toolagent": {"check": {("excl_sys", "pm"): 0.05, ("raw", "pm"): 0.9}}}
    check("單位檢查：一個在 [0.19, 0.76] 就過", unit_check(good)[0] and not unit_check(bad)[0])
    check("單位檢查：沒跑 toolagent → 不過", not unit_check({})[0])
    # 頻寬表讀得到、而且每一格都有
    try:
        bw = bw_table()
        check("頻寬表：2 平台 × 3 tier 都有", set(bw) == {(p, t) for p in ("h200", "mi300x") for t in ("cpu", "nvme", "nfs")})
    except Exception as e:  # noqa: BLE001
        check(f"頻寬表讀得到（{e}）", False)
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--traces", default="conversation,toolagent,synthetic")
    r.add_argument("--trace-dir", default="", help="預設＝m8_trace_oracle.TRACE_DIR（$TIARA_DATASETS/traces）")
    r.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if a.cmd == "run":
        cmd_run(a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
