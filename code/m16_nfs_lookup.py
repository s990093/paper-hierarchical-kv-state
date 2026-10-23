#!/usr/bin/env python3
"""m16_nfs_lookup.py — I3g d（檔案層在 NFS）為什麼每個請求要等 41 秒「查下層」：拆成「NFS 單次存在檢查的延遲」×「一條執行緒依序查」。
只用 CPU、標準函式庫（不 import torch）。設計與事先寫死的判定：experiment_log.md「NFS 等待拆解（67）」（跑之前寫）。

vLLM 0.28 的檔案層查找（讀過原始碼）：每個 tier 一條背景執行緒（tiering/async_lookup.py），每個排程步把要查的 key 打成一批，
逐個檔案做存在檢查（fs/manager.py:76-83：有 C 擴充時 faccessat，否則 os.path.exists）；路徑 <base>_r0/<hhh>/<hh>_g0/<hash>.bin（file_mapper.py:118-126）。
寫入（fs/io.py）：先檢查存在、寫暫存檔、rename；有 O_DIRECT 就用（64 的 c、d 的 server log 都沒有「不支援 O_DIRECT」的警告）。

量法：在目標目錄照同樣的路徑結構先建 --n-files 個小檔（「已存在」的 key），然後一條執行緒做 --n-lookups 次 os.access(F_OK)
（命中比例 --hit-frac＝64 d 的 tiering_chunk_hits／chunk_queries ≈ 0.55），量每次的延遲；三個階段：
  Q   安靜
  W   同時有寫入：--writers 條執行緒寫 2 MiB 檔（暫存檔＋rename，O_DIRECT 可用就用），總速率限在 --write-mbps（64 d 平均約 100 MB/s）
  WR  寫入＋讀取：再加 --readers 條執行緒讀 W 寫好的檔（O_DIRECT），總速率限在 --read-mbps（64 d 平均約 180 MB/s）
推回 64：查找執行緒的忙碌度 ρ＝chunk_queries × 平均延遲 ÷ makespan（d 用 NFS 的數字，c 用本地的數字）。
自己的檔案跑完一律刪掉（--keep 才留）。

  python3 -I code/m16_nfs_lookup.py run --dir /mlsteam/data/tiara/stage3/nfsbench --label nfs --out results/.../i3g_agentic/nfs_lookup
  python3 -I code/m16_nfs_lookup.py report --out results/.../i3g_agentic/nfs_lookup --i3g results/m16_stage3_mi300x/i3g_agentic
  python3 -I code/m16_nfs_lookup.py --self-test
"""
from __future__ import annotations

import argparse
import glob
import json
import mmap
import os
import random
import shutil
import statistics
import sys
import threading
import time
from datetime import datetime, timezone

MiB = 1 << 20


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def key_path(base: str, h: str) -> str:
    return f"{base}_r0/{h[:3]}/{h[3:5]}_g0/{h}.bin"


def rand_hash(rng: random.Random) -> str:
    return "%064x" % rng.getrandbits(256)


def pct(v, q):
    s = sorted(v)
    k = (len(s) - 1) * q
    f = int(k)
    return s[f] + (s[min(f + 1, len(s) - 1)] - s[f]) * (k - f)


def summ(v):
    return dict(n=len(v), mean=statistics.fmean(v), p50=pct(v, 0.5), p95=pct(v, 0.95), p99=pct(v, 0.99), max=max(v)) if v else dict(n=0)


class Rate:
    """總速率限制（token bucket，所有執行緒共用）。"""

    def __init__(self, mbps: float):
        self.bps, self.lock, self.t, self.tokens = mbps * 1e6, threading.Lock(), time.monotonic(), 0.0

    def take(self, nbytes: int, stop: threading.Event):
        while not stop.is_set():
            with self.lock:
                t = time.monotonic()
                self.tokens = min(self.tokens + (t - self.t) * self.bps, 4 * nbytes)
                self.t = t
                if self.tokens >= nbytes:
                    self.tokens -= nbytes
                    return True
            time.sleep(0.002)
        return False


def odirect_ok(d: str) -> bool:
    p = os.path.join(d, ".odirect_probe")
    try:
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_DIRECT", 0), 0o644)
        buf = mmap.mmap(-1, 4096)
        os.write(fd, buf)
        os.close(fd)
        return True
    except OSError:
        return False
    finally:
        try:
            os.unlink(p)
        except OSError:
            pass


def write_file(path: str, buf: mmap.mmap, direct: bool):
    """vLLM fs/io.py 的作法：已存在就跳過；寫暫存檔再 rename。"""
    if os.path.exists(path):
        return 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{threading.get_ident()}"
    fl = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | (getattr(os, "O_DIRECT", 0) if direct else 0)
    fd = os.open(tmp, fl, 0o644)
    try:
        os.write(fd, buf)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    return len(buf)


def read_file(path: str, buf: mmap.mmap, direct: bool):
    fd = os.open(path, os.O_RDONLY | (getattr(os, "O_DIRECT", 0) if direct else 0))
    try:
        return os.preadv(fd, [buf], 0)
    finally:
        os.close(fd)


def lookups(base: str, have: list, n: int, hit_frac: float, rng: random.Random):
    hit, miss = [], []
    for _ in range(n):
        if rng.random() < hit_frac:
            p, out = key_path(base, rng.choice(have)), hit
        else:
            p, out = key_path(base, rand_hash(rng)), miss
        t0 = time.perf_counter_ns()
        os.access(p, os.F_OK)
        out.append((time.perf_counter_ns() - t0) / 1e9)
    return hit, miss


def cmd_run(a):
    rng = random.Random(a.seed)
    rid = os.environ.get("RUN_ID", "adhoc")
    root = os.path.join(a.dir, f"m16_nfsbench_{rid}")
    base = os.path.join(root, "model_x")
    os.makedirs(root, exist_ok=False)
    os.makedirs(a.out, exist_ok=True)
    direct = odirect_ok(root)
    res = dict(run_id=rid, ts=now_iso(), label=a.label, dir=a.dir, root=root, args=vars(a), o_direct=direct, phases={})
    try:
        # 0. 建「已存在」的小檔（多執行緒；只為了有東西可以查）
        have = [rand_hash(rng) for _ in range(a.n_files)]
        small = mmap.mmap(-1, 4096)
        t0 = time.monotonic()

        def mk(chunk):
            for h in chunk:
                write_file(key_path(base, h), small, direct)
        th = [threading.Thread(target=mk, args=(have[i::16],)) for i in range(16)]
        [t.start() for t in th]
        [t.join() for t in th]
        res["create_s"] = time.monotonic() - t0
        big = mmap.mmap(-1, a.file_mib * MiB)
        written, wlock = [], threading.Lock()
        for ph in ("Q", "W", "WR"):
            stop = threading.Event()
            bg, stats = [], dict(w_bytes=0, r_bytes=0, w_err=0, r_err=0)
            if ph in ("W", "WR"):
                wr = Rate(a.write_mbps)

                def writer(i):
                    r_ = random.Random(a.seed * 1000 + i)
                    while not stop.is_set():
                        if not wr.take(len(big), stop):
                            break
                        h = rand_hash(r_)
                        try:
                            stats["w_bytes"] += write_file(key_path(base + "_big", h), big, direct)
                            with wlock:
                                written.append(h)
                        except OSError:
                            stats["w_err"] += 1
                bg += [threading.Thread(target=writer, args=(i,), daemon=True) for i in range(a.writers)]
            if ph == "WR":
                rr = Rate(a.read_mbps)

                def reader(i):
                    r_ = random.Random(a.seed * 2000 + i)
                    rb = mmap.mmap(-1, a.file_mib * MiB)
                    while not stop.is_set():
                        if not rr.take(len(rb), stop):
                            break
                        with wlock:
                            h = r_.choice(written) if written else None
                        if h is None:
                            time.sleep(0.01)
                            continue
                        try:
                            stats["r_bytes"] += read_file(key_path(base + "_big", h), rb, direct)
                        except OSError:
                            stats["r_err"] += 1
                bg += [threading.Thread(target=reader, args=(i,), daemon=True) for i in range(a.readers)]
            [t.start() for t in bg]
            if bg:
                time.sleep(a.warm_s)               # 讓背景 I/O 先到穩態
            t0 = time.monotonic()
            hit, miss = lookups(base, have, a.n_lookups, a.hit_frac, rng)
            dur = time.monotonic() - t0
            stop.set()
            [t.join(timeout=60) for t in bg]
            allv = hit + miss
            res["phases"][ph] = dict(all=summ(allv), hit=summ(hit), miss=summ(miss), lookup_wall_s=dur,
                                     bg_write_mbps=stats["w_bytes"] / 1e6 / (dur + (a.warm_s if bg else 0)),
                                     bg_read_mbps=stats["r_bytes"] / 1e6 / (dur + (a.warm_s if bg else 0)), **stats)
            print(ph, json.dumps(res["phases"][ph]["all"]), f"w {res['phases'][ph]['bg_write_mbps']:.0f} MB/s r {res['phases'][ph]['bg_read_mbps']:.0f} MB/s",
                  flush=True)
        res["bytes_written_total"] = len(written) * a.file_mib * MiB + a.n_files * 4096
    finally:
        t0 = time.monotonic()
        if not a.keep:
            shutil.rmtree(root, ignore_errors=True)
        res["cleanup_s"], res["removed"] = time.monotonic() - t0, (not a.keep) and not os.path.exists(root)
    p = os.path.join(a.out, f"lookup_{a.label}_{rid}.json")
    json.dump(res, open(p, "w"), indent=1)
    print("寫好", p, "刪乾淨" if res["removed"] else "（留著或沒刪掉）")


def cmd_report(a):
    rows = {}
    for f in sorted(glob.glob(os.path.join(a.out, "lookup_*.json"))):
        d = json.load(open(f))
        rows[d["label"]] = d                        # 同一個 label 取最後一個
    rep = json.load(open(os.path.join(a.i3g, "i3g_report.json")))
    io = {}
    for k in ("c", "d"):
        c = rep["configs"][k]
        s = json.load(open(os.path.join(a.i3g, f"summary_{c['run_id']}.json")))
        q = c["metrics"]["series"]["tiering_chunk_queries"]["delta"]
        lk = c["metrics"]["series"]["offload_lookup_async_delay_sum_s"]["delta"]
        io[k] = dict(run_id=c["run_id"], chunk_queries=q, makespan_s=s["throughput"]["makespan_s"], lookup_delay_mean_s=lk / 364)
    rho = {}
    for k, lab in (("d", "nfs"), ("c", "local")):
        r = rows.get(lab)
        if not r:
            rho[k] = "NOT_MEASURED"
            continue
        rho[k] = {ph: io[k]["chunk_queries"] * r["phases"][ph]["all"]["mean"] / io[k]["makespan_s"] for ph in r["phases"]}
    if isinstance(rho.get("d"), dict) and "WR" in rho["d"]:
        x = rho["d"]["WR"]
        v = "LOOKUP_THREAD_SATURATED" if x >= 0.8 else ("NOT_EXPLAINED_BY_LOOKUP_LATENCY" if x < 0.3 else "PARTIAL")
        n = rows["nfs"]["phases"]
        infl = n["WR"]["all"]["mean"] / n["Q"]["all"]["mean"]
        v2 = "IO_INFLATES_LOOKUP" if infl >= 2 else "INTRINSIC_NFS_LATENCY"
    else:
        v, v2, infl = "INCOMPLETE", "INCOMPLETE", None
    out = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=now_iso(), verdict=v, verdict_cause=v2, wr_over_q=infl, rho=rho, i3g=io,
               runs={k: dict(run_id=r["run_id"], dir=r["dir"], o_direct=r["o_direct"], removed=r["removed"], create_s=r["create_s"],
                             phases=r["phases"]) for k, r in rows.items()},
               rule="ρ_d（WR 階段 NFS 的平均延遲 × d 的 chunk_queries ÷ d 的 makespan）≥ 0.8 → LOOKUP_THREAD_SATURATED；< 0.3 → "
                    "NOT_EXPLAINED_BY_LOOKUP_LATENCY；其他 PARTIAL。NFS 的 WR／Q 平均延遲比 ≥ 2 → IO_INFLATES_LOOKUP，否則 INTRINSIC_NFS_LATENCY。")
    json.dump(out, open(os.path.join(a.out, "nfs_lookup_report.json"), "w"), indent=1, ensure_ascii=False)
    ms = lambda x: f"{1000 * x:.3f}"  # noqa: E731
    L = [f"# NFS 等待拆解（run {out['run_id']}）", "", f"判定：**{v}**；**{v2}**（NFS 的 WR／Q 平均延遲比 {infl if infl is None else round(infl, 2)}）", "",
         "| 目錄 | 階段 | 背景寫 MB/s | 背景讀 MB/s | 次數 | 平均（ms） | P50 | P95 | P99 | 命中平均 | 未命中平均 | ρ（推回 64） |",
         "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for lab, k in (("local", "c"), ("nfs", "d")):
        r = rows.get(lab)
        if not r:
            L.append(f"| {lab} | | | | NOT_MEASURED | | | | | | | |")
            continue
        for ph, p in r["phases"].items():
            al = p["all"]
            L.append(f"| {lab}（{r['dir']}） | {ph} | {p['bg_write_mbps']:.0f} | {p['bg_read_mbps']:.0f} | {al['n']} | {ms(al['mean'])} | {ms(al['p50'])} | "
                     f"{ms(al['p95'])} | {ms(al['p99'])} | {ms(p['hit']['mean'])} | {ms(p['miss']['mean'])} | {rho[k][ph]:.2f} |")
    L += ["", f"64 的對照：d chunk_queries {io['d']['chunk_queries']:.0f}、makespan {io['d']['makespan_s']:.0f} s、每請求查找等待 {io['d']['lookup_delay_mean_s']:.2f} s；"
          f"c {io['c']['chunk_queries']:.0f}、{io['c']['makespan_s']:.0f} s、{io['c']['lookup_delay_mean_s']:.2f} s。",
          "ρ＝查找執行緒的忙碌度（一條執行緒依序查）；ρ 接近或超過 1 時排隊會爆。量測和 64 不同時間，NFS 是共用的，延遲會隨別人的負載變。",
          "", "規則：" + out["rule"]]
    open(os.path.join(a.out, "nfs_lookup_report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def self_test():
    import tempfile
    ok = True
    with tempfile.TemporaryDirectory() as td:
        h = "ab" * 32
        p = key_path(os.path.join(td, "m"), h)
        ok &= p.endswith(f"m_r0/aba/ba_g0/{h}.bin")
        buf = mmap.mmap(-1, MiB)
        n = write_file(p, buf, False)
        ok &= n == MiB and os.path.getsize(p) == MiB and write_file(p, buf, False) == 0
        rb = mmap.mmap(-1, MiB)
        ok &= read_file(p, rb, False) == MiB
        hit, miss = lookups(os.path.join(td, "m"), [h], 200, 0.5, random.Random(0))
        ok &= len(hit) + len(miss) == 200 and 60 < len(hit) < 140
        r = Rate(50.0)
        stop = threading.Event()
        t0 = time.monotonic()
        for _ in range(10):
            r.take(MiB, stop)
        dt = time.monotonic() - t0
        ok &= 0.1 < dt < 0.4                      # 10 MiB @ 50 MB/s ≈ 0.21 s
        ok &= abs(pct([1, 2, 3, 4], 0.5) - 2.5) < 1e-9
    print("self-test", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--dir", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--n-files", type=int, default=20000)
    r.add_argument("--n-lookups", type=int, default=5000)
    r.add_argument("--hit-frac", type=float, default=0.55)
    r.add_argument("--file-mib", type=int, default=2)
    r.add_argument("--writers", type=int, default=16)
    r.add_argument("--readers", type=int, default=16)
    r.add_argument("--write-mbps", type=float, default=100.0)
    r.add_argument("--read-mbps", type=float, default=180.0)
    r.add_argument("--warm-s", type=float, default=5.0)
    r.add_argument("--seed", type=int, default=20261023)
    r.add_argument("--keep", action="store_true")
    q = sp.add_parser("report")
    q.add_argument("--out", required=True)
    q.add_argument("--i3g", default="results/m16_stage3_mi300x/i3g_agentic")
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_report(a)


if __name__ == "__main__":
    main()
