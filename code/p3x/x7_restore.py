"""x7_restore.py — X7：3090 上真的跑 Cake／只算／只載，量會合點 b 與還原時間（CPU 層真實 H2D＋限速層）。

為什麼另寫（不改 m10_speed）：
  1. m10_speed.Bench 在 GPU 上另存一份參考 KV（ref＝kv.clone()）。Llama 32K 時 KV 4 GiB×2＋權重 15 GiB 會超過 3090 的 24 GiB；
     這裡改用主機上的 BF16 副本當參考（它就是 prefill 後 D2H 的那份，逐位元相同），一次搬一個 block 回 GPU 比對。
  2. m10_speed 的 Busy 只傳 --duty；這裡把 m7_busy.py 的 --duty --period-ms --n 都開出來（x7_busy.py 量過 3090 上的實際 duty）。
  3. 只跑 X7 要的格子：config E0（全 BF16）／E1:<fmt>（全低精度），mode cake／compute_only／load_only，任意頻寬清單。
還原本身完全用 m7_restore_harness.Restorer（不改它）；預設反量化＝主線的 m10_codec.decode，--fused 才換 m10_dequant_fast。

  python code/p3x/x7_restore.py --model qwen --length 32768 --out-dir $RUN_DIR \
      --tiers cpu x_b16=0.5907 mi_local=6.98 --jobs E0:cake E0:load_only E1:fp8_head:cake R:compute_only \
      --gpu-states idle busy --busy-cfg 0.7:20:8192 --reps 5

--tiers：cpu（不限速、真的 H2D）或 名字=GiB/s（限速層，m7_restore_harness.Tier，讀寫對稱）。
--jobs：<config>:<mode>，config＝E0 或 E1:<fmt>；compute_only 寫成 R:compute_only（和頻寬無關，每個 GPU 狀態每個 rep 只跑一次）。
--job-tiers：限定某個 job 只在哪些頻寬跑（例 E0:load_only=cpu,mi_local）；沒列的 job 在所有頻寬跑。
每列：run_id、ts、模型、長度、config、fmt、mode、tier、GiB/s、GPU 狀態、背景設定、rep、順序、b、各來源 block 數、bytes_read、
      t_restore、t_recompute、t_load、t_h2d、t_dequant、t_wait、waiting_line、t_new_turn、ttft、copy_bound、kv_bad（每格第一次）、
      contaminated_gpu（本程式自己的 GpuWatcher；背景 m7_busy 是子 process，白名單）、host_level（gpu_guard.host_contention）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)

import torch  # noqa: E402

GiB = 1 << 30
FIELDS = ["run_id", "ts", "model", "length", "n_blocks", "config", "fmt", "mode", "tier", "tier_GiBps", "gpu_state",
          "busy_cfg", "rep", "order_idx", "b", "n_recompute", "n_load_bf16", "n_load_fp8", "n_load_int4", "n_load_cpu",
          "n_load_ssd", "bytes_read", "t_restore", "t_recompute", "t_load", "t_h2d", "t_dequant", "t_wait", "waiting_line",
          "t_new_turn", "ttft", "copy_bound", "kv_bad", "dequant_impl", "contaminated_gpu", "host_level"]


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


class Busy:
    def __init__(self, cfg: str | None):
        self.cfg, self.p, self.info = cfg, None, {}

    def __enter__(self):
        if self.cfg:
            duty, period, n = self.cfg.split(":")
            self.p = subprocess.Popen([sys.executable, os.path.join(CODE, "m7_busy.py"), "--duty", duty, "--period-ms",
                                       period, "--n", n], stdout=subprocess.PIPE, text=True)
            line = self.p.stdout.readline()
            if not line.startswith("READY"):
                self.p.kill()
                raise RuntimeError(f"m7_busy 沒有啟動：{line!r}")
            self.info["ready"] = line.strip()
            time.sleep(1.0)
        return self

    def __exit__(self, *a):
        if self.p:
            self.p.terminate()
            out, _ = self.p.communicate(timeout=60)
            self.info["done"] = [x for x in out.splitlines() if x.startswith("DONE")]
        return False


def parse_tiers(specs):
    out = []
    for s in specs:
        if s == "cpu":
            out.append({"name": "cpu", "GiBps": "real"})
        else:
            n, v = s.split("=")
            out.append({"name": n, "GiBps": float(v)})
    return out


def parse_job(j):
    parts = j.split(":")
    mode = parts[-1]
    cfg = ":".join(parts[:-1])
    if cfg in ("-", "R"):                 # R:compute_only（argparse 會把開頭是 - 的值當成選項）
        return "-", "bf16", mode
    if cfg == "E0":
        return "E0", "bf16", mode
    if cfg.startswith("E1:"):
        return "E1", cfg[3:], mode
    raise ValueError(j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["llama", "qwen"], required=True)
    ap.add_argument("--length", type=int, default=32768)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tiers", nargs="+", required=True)
    ap.add_argument("--jobs", nargs="+", required=True)
    ap.add_argument("--job-tiers", nargs="*", default=[])
    ap.add_argument("--gpu-states", nargs="+", default=["idle"])
    ap.add_argument("--busy-cfg", default="0.7:20:8192")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--fused", action="store_true", help="反量化改用 m10_dequant_fast（Triton 融合 kernel）")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.environ["M7_MODEL_GLOB"] = os.environ[f"M7_{a.model.upper()}_GLOB"]
    if a.fused:
        import m10_dequant_fast
        m10_dequant_fast.install()
    from m10_codec import decode, encode
    from m7_model import CHUNK, KVModel
    from m7_restore_harness import Restorer, Tier
    from gpu_guard import GpuWatcher, host_contention, smi_index

    rid = os.environ.get("RUN_ID", "NO_RUN_ID")
    os.makedirs(a.out_dir, exist_ok=True)
    n = a.length // CHUNK
    tiers = parse_tiers(a.tiers)
    jobs = [parse_job(j) for j in a.jobs]
    jt = {}
    for s in a.job_tiers:
        k, v = s.split("=")
        jt[k] = set(v.split(","))
    fmts = sorted({f for c, f, m in jobs if c == "E1"})

    km = KVModel(max_len=n * CHUNK + max(512, a.new_tokens))
    rs = Restorer(km)
    rs.time_h2d = True
    g = torch.Generator().manual_seed(a.seed)
    ids = torch.randint(1000, min(120000, km.cfg.vocab_size), (n * CHUNK + a.new_tokens,), generator=g).to(km.device)
    # 寫入：prefill → 每個 block 一份 BF16 pinned 副本＋每種低精度格式一份 pinned 副本
    km.prefill_chunked(ids[:n * CHUNK])
    torch.cuda.synchronize()
    bf, lp = [], {f: [] for f in fmts}
    for i in range(n):
        v = km.chunk_view(i)
        h = km.host_chunk()
        h.copy_(v)
        bf.append(h)
        for f in fmts:
            e = encode(v, f)
            hp = e.map(lambda t: torch.empty(t.shape, dtype=t.dtype, pin_memory=True))
            hp.copy_from(e)
            lp[f].append(hp)
            del e
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    def expected(i, cfg, fmt, src):
        if src in ("recompute", "load_bf16"):
            return bf[i].to(km.device)
        return decode(lp[fmt][i].to(km.device))

    def restore(cfg, fmt, mode, tier):
        if tier["GiBps"] == "real":
            t = Tier("cpu_real", read_Bps=0, write_Bps=0)
        else:
            Bps = tier["GiBps"] * GiB
            t = Tier(f"ssd_{tier['name']}", read_Bps=Bps, write_Bps=Bps)
        host = lp[fmt] if cfg == "E1" else bf
        km.kv[:, :, :, :n * CHUNK].zero_()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rec = rs.restore(ids, n, host, [t] * n, mode)
        t_new, _ = rs.new_turn(ids[n * CHUNK:n * CHUNK + a.new_tokens], n * CHUNK)
        rec["ttft"] = time.perf_counter() - t0
        rec["t_new_turn"] = t_new
        return rec

    def kv_bad(rec, cfg, fmt):
        bad = 0
        for i, src, _, _ in rec["blocks"]:
            bad += int(not torch.equal(km.chunk_view(i), expected(i, cfg, fmt, src)))
        return bad

    try:
        watch = GpuWatcher(gpu=smi_index(0), own_root=os.getpid(), out_path=os.path.join(a.out_dir, f"gpu_guard_inner_{rid}.json"))
        watch.__enter__()
    except Exception as ex:  # noqa: BLE001  查不到不等於乾淨（規則 7）
        print("GpuWatcher unavailable:", repr(ex), flush=True)
        watch = None
    try:
        level = host_contention(exclude_gpu=smi_index(0))
    except Exception as ex:  # noqa: BLE001
        level = {"level": f"UNKNOWN({type(ex).__name__})"}
    rf = open(os.path.join(a.out_dir, "restores.csv"), "a", newline="")
    w = csv.DictWriter(rf, fieldnames=FIELDS, extrasaction="ignore")
    if rf.tell() == 0:
        w.writeheader()
    rng = random.Random(f"x7|{rid}|{a.seed}")
    # 暖機：每種模式各一次（不記）
    for c, f, m in jobs:
        restore(c if c != "-" else "E0", f, m, tiers[0])
    verified = set()
    busy_info = []
    for st in a.gpu_states:
        with Busy(a.busy_cfg if st == "busy" else None) as bz:
            for rep in range(a.reps):
                todo = []
                for c, f, m in jobs:
                    if m == "compute_only":
                        todo.append((c, f, m, {"name": "-", "GiBps": ""}))
                        continue
                    key = f"{c}:{f}:{m}" if c == "E1" else f"{c}:{m}"
                    for t in tiers:
                        if key in jt and t["name"] not in jt[key]:
                            continue
                        todo.append((c, f, m, t))
                rng.shuffle(todo)
                for k, (c, f, m, t) in enumerate(todo):
                    cfg = "E0" if c == "-" else c
                    tier = tiers[0] if m == "compute_only" else t
                    rec = restore(cfg, f, m, tier)
                    vk = (cfg, f, m, t["name"], st)
                    kb = ""
                    if not a.no_verify and vk not in verified:
                        kb = kv_bad(rec, cfg, f)
                        verified.add(vk)
                    cont = "NOT_CHECKED" if watch is None else (bool(watch.intruders) or watch.started_clean is False)
                    w.writerow(dict(
                        run_id=rid, ts=ts(), model=a.model, length=a.length, n_blocks=n, config=cfg if c != "-" else "-",
                        fmt=f, mode=m, tier=t["name"], tier_GiBps=t["GiBps"], gpu_state=st,
                        busy_cfg=a.busy_cfg if st == "busy" else "", rep=rep, order_idx=k, b=rec["meet"],
                        n_recompute=rec["n_recompute"], n_load_bf16=rec["n_load_bf16"], n_load_fp8=rec["n_load_fp8"],
                        n_load_int4=rec["n_load_int4"], n_load_cpu=rec["n_load_cpu"], n_load_ssd=rec["n_load_ssd"],
                        bytes_read=rec["bytes_read"], t_restore=round(rec["t_restore"], 6),
                        t_recompute=round(rec["t_recompute"], 6), t_load=round(rec.get("t_load", 0.0), 6),
                        t_h2d=round(rec["t_h2d"], 6), t_dequant=round(rec["t_dequant"], 6),
                        t_wait=round(rec.get("t_wait", 0.0), 6) if m == "cake" else "",
                        waiting_line=rec.get("waiting_line", ""), t_new_turn=round(rec["t_new_turn"], 6),
                        ttft=round(rec["ttft"], 6), copy_bound=rec["n_copy_bound"], kv_bad=kb,
                        dequant_impl="fused_triton" if a.fused else "torch(m10_codec)", contaminated_gpu=cont,
                        host_level=level.get("level")))
                rf.flush()
                print(f"[{st} rep {rep}] {len(todo)} restores ok", flush=True)
        busy_info.append({"state": st, **bz.info})
    rf.close()
    try:
        level_end = host_contention(exclude_gpu=smi_index(0))
    except Exception as ex:  # noqa: BLE001
        level_end = {"level": f"UNKNOWN({type(ex).__name__})"}
    if watch is not None:
        watch.__exit__(None, None, None)
    json.dump({"run_id": rid, "argv": sys.argv, "host_contention_start": level, "host_contention_end": level_end,
               "busy": busy_info, "contaminated_inner": None if watch is None else watch.contaminated,
               "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / GiB, 3)},
              open(os.path.join(a.out_dir, f"meta_{rid}.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
