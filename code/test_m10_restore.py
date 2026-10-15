"""test_m10_restore.py — 還原器加了低精度路徑之後的兩個檢查（03_plan_h200.md §6 的 m7_restore_harness 那一列）

mixed   低精度載入對不對：同一份 KV，依配置把 block 存成 BF16 或低精度（E0、E1、F25、B25），
        用 compute_only／load_only（讀法 A）／cake 還原，逐 block 比對：
          recompute 的 block ＝ 原本 prefill 的 KV（逐位元）
          load_bf16 的 block ＝ 原本的 KV（逐位元）
          load_<低精度> 的 block ＝ decode(encode(原本的 KV))（逐位元）
        另查：來源計數、bytes_read＝實際載入的位元組總和、有載低精度時 t_dequant > 0。
regress 全 BF16 時，新舊還原器（舊＝git HEAD 的 m7_restore_harness.py，由 --old 指定檔案）
        在同一個 process、同一份 KV 上交錯跑，比 KV 是否逐位元相同、會合點、還原時間中位數。
        ⚠️ 2026-10-15 診斷：同一個 process 裡「先建立」的 Restorer 比「後建立」的快（MI300X，local:cake 0.225 vs 0.260 s，
        同一份程式互比也一樣），所以同 process 比新舊會混進建立順序。乾淨的比法：--impls new 或 --impls old，
        每個 process 只建一個，交替開 process 再比（regress_xproc）。正式腳本一個 process 只建一個 Restorer。
        判準（跑之前寫死，2026-10-15）：KV 全部逐位元相同；每個（層, 模式）的 t_restore 中位數比值
        新/舊 在 0.95–1.05 之間（比第一階段的雜訊 δ＝6.2% 嚴）。

這是 MI300X 上的彩排（results/m10_pilot_mi300x/），數字不進任何圖或判定。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m10_codec import Enc, decode, encode  # noqa: E402
from m7_model import CHUNK, KVModel  # noqa: E402
from m7_restore_harness import GiB, Restorer, Tier  # noqa: E402

CONFIGS = {   # (block, n) → 格式；照 03 §2.1（E1s、M25、R25 的機制和這幾個相同，這裡不重複）
    # F25／B25：前、後各 k＝n//4 個 block（不含 block 0）；n＝64 時 k＝16 → block 1–16、48–63，和 03 §2.1 相同
    "E0": lambda i, n: "bf16",
    "E1": lambda i, n: "LP",
    "F25": lambda i, n: "LP" if 1 <= i <= n // 4 else "bf16",
    "B25": lambda i, n: "LP" if n - n // 4 <= i <= n - 1 else "bf16",
}


def setup(n, seed):
    km = KVModel(max_len=n * CHUNK + 512)
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1000, min(120000, km.cfg.vocab_size), (n * CHUNK + 512,), generator=g).to(km.device)
    km.prefill_chunked(ids[:n * CHUNK])
    torch.cuda.synchronize()
    ref = [km.chunk_view(i).clone() for i in range(n)]        # GPU 上的參考 KV
    return km, ids, ref


def mixed(a):
    n = a.n                                                     # 預設 64（32K）；24 GB 的卡用 --n 32 或 16（m10_verify.sh）
    assert n >= 8, n
    km, ids, ref = setup(n, a.seed)
    rs = Restorer(km)
    bf16_host = [km.host_chunk().copy_(r) for r in ref]
    lp_host = [encode(r, a.fmt).to("cpu").pin() for r in ref]
    lp_expect = [decode(encode(r, a.fmt)) for r in ref]
    # 限速：讓 BF16 一個 block 的載入約 50 ms，會合點大約落在中間（只為了讓三種來源都出現，不是量速度）
    bw = km.chunk_bytes / 0.05
    res = {"model": km.cfg._name_or_path, "fmt": a.fmt, "n": n, "chunk_bytes": km.chunk_bytes,
           "lp_block_bytes": lp_host[0].nbytes, "tier_read_GiBps": bw / GiB, "torch": torch.__version__,
           "hip": torch.version.hip, "cuda": torch.version.cuda, "runs": []}
    ok = True
    for cfg, f in CONFIGS.items():
        host = [lp_host[i] if f(i, n) == "LP" else bf16_host[i] for i in range(n)]
        for mode in ("compute_only", "load_only", "cake"):
            tier = Tier("ssd_test", read_Bps=bw, write_Bps=bw)
            km.kv[:, :, :, :n * CHUNK].zero_()
            torch.cuda.synchronize()
            rec = rs.restore(ids, n, host, [tier] * n, mode)
            bad, srcs = [], {}
            for i, src, _, _ in rec["blocks"]:
                srcs[src] = srcs.get(src, 0) + 1
                want = lp_expect[i] if src.startswith("load_") and src != "load_bf16" else ref[i]
                if not torch.equal(km.chunk_view(i), want):
                    bad.append((i, src))
            seen = sorted(b[0] for b in rec["blocks"])
            exp_bytes = sum(host[i].nbytes if isinstance(host[i], Enc) else km.chunk_bytes
                            for i, src, _, _ in rec["blocks"] if src != "recompute")
            n_lp = sum(v for k, v in srcs.items() if k not in ("recompute", "load_bf16"))
            checks = {
                "kv_bitwise_ok": not bad,
                "every_block_once": seen == list(range(n)),
                "bytes_read_ok": rec["bytes_read"] == exp_bytes,
                "dequant_timed": (rec["t_dequant"] > 0) == (n_lp > 0),
                "counts_ok": rec["n_recompute"] == srcs.get("recompute", 0) and
                             rec["n_load_bf16"] == srcs.get("load_bf16", 0) and
                             sum(rec[f"n_load_{k}"] for k in ("fp8", "int8", "int4")) == n_lp,
            }
            ok &= all(checks.values())
            row = {"config": cfg, "mode": mode, "meet": rec["meet"], "sources": srcs, "bytes_read": rec["bytes_read"],
                   "t_restore": round(rec["t_restore"], 4), "t_dequant": round(rec["t_dequant"], 5),
                   "n_copy_bound": rec["n_copy_bound"], "bad_blocks": bad[:8], **checks}
            res["runs"].append(row)
            print(json.dumps(row), flush=True)
    res["all_pass"] = bool(ok)
    print("ALL PASS" if ok else "SOME FAILED", flush=True)
    return res, ok


def load_old(path):
    spec = importlib.util.spec_from_file_location("m7_restore_harness_old", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod        # dataclass 在 exec 時要查得到自己的模組
    spec.loader.exec_module(mod)
    return mod


def regress(a):
    n = 32                                                      # 16K，和第一階段 C4 同長度
    km, ids, ref = setup(n, a.seed)
    old = load_old(a.old)
    # 建立順序照 --impls（stream 的建立順序可能影響 GPU 硬體佇列的分配；new2／old2＝同一份程式的第二個實例）
    mk = {"new": lambda: Restorer(km), "new2": lambda: Restorer(km),
          "old": lambda: old.Restorer(km), "old2": lambda: old.Restorer(km)}
    impls = {k: mk[k]() for k in a.impls.split(",")}
    host = [km.host_chunk().copy_(r) for r in ref]
    P = json.load(open(os.path.join(HERE, "..", "results", "m7_write_policy_mi300x", "tier_params.json")))
    tiers = {"local": P["local"]["read_GiBps"], "nfs": P["nfs"]["read_GiBps"]}
    rng = random.Random(a.seed)
    rows, kv_ok = [], True
    for _ in range(2):                                          # 暖機
        for name, r in impls.items():
            r.restore(ids, n, host, [Tier("w", read_Bps=tiers["local"] * GiB, write_Bps=1e12)] * n, "cake")
    cells = [(t, m) for t in tiers for m in ("cake", "load_only", "compute_only")]
    if a.cells:
        cells = [tuple(c.split(":")) for c in a.cells.split(",")]
    for tname, mode in cells:
            gibps = tiers[tname]
            for rep in range(a.reps):
                order = list(impls)
                rng.shuffle(order)                              # 每次重複隨機先後，抵消漂移
                for name in order:
                    tier = Tier(f"ssd_{tname}", read_Bps=gibps * GiB, write_Bps=gibps * GiB)
                    km.kv[:, :, :, :n * CHUNK].zero_()
                    torch.cuda.synchronize()
                    rec = impls[name].restore(ids, n, host, [tier] * n, mode)
                    same = all(torch.equal(km.chunk_view(i), ref[i]) for i in range(n))
                    kv_ok &= same
                    nl = rec["n_load_cpu"] + rec["n_load_ssd"]
                    rows.append({"tier": tname, "mode": mode, "rep": rep, "impl": name, "meet": rec["meet"],
                                 "t_restore": rec["t_restore"], "kv_bitwise_ok": same,
                                 "ms_per_load": 1e3 * (rec["t_load_cpu"] + rec["t_load_ssd"]) / nl if nl else None,
                                 "ms_per_recompute": 1e3 * rec["t_recompute"] / rec["n_recompute"] if rec["n_recompute"] else None})
            print(tname, mode, "done", flush=True)
    summ, ok = [], kv_ok
    names = list(impls)
    a_, b_ = ("new", "old") if {"new", "old"} <= set(names) else (names + names)[:2]   # 只有一個時比值＝1（跨 process 另外比）
    for tname, mode in cells:
            sel = {k: [r for r in rows if r["tier"] == tname and r["mode"] == mode and r["impl"] == k] for k in impls}
            med = {k: statistics.median(r["t_restore"] for r in v) for k, v in sel.items()}
            ratio = med[a_] / med[b_]
            meets = {k: sorted({r["meet"] for r in v}) for k, v in sel.items()}
            per = {k: {"ms_per_load": statistics.median([r["ms_per_load"] for r in v if r["ms_per_load"]] or [0]),
                       "ms_per_recompute": statistics.median([r["ms_per_recompute"] for r in v if r["ms_per_recompute"]] or [0])}
                   for k, v in sel.items()}
            good = 0.95 <= ratio <= 1.05
            ok &= good
            summ.append({"tier": tname, "mode": mode, "pair": f"{a_}/{b_}", "median_s": {k: round(v, 4) for k, v in med.items()},
                         "ratio": round(ratio, 4), "meets": meets, "per_op_ms": per, "pass": good})
            print(summ[-1], flush=True)
    res = {"model": km.cfg._name_or_path, "n": n, "reps": a.reps, "old_harness": a.old, "kv_bitwise_all": kv_ok,
           "criterion": "KV 全部逐位元相同；t_restore 中位數比值 新/舊 ∈ [0.95, 1.05]（跑之前寫死）",
           "summary": summ, "rows": rows, "all_pass": bool(ok), "torch": torch.__version__}
    print("ALL PASS" if ok else "SOME FAILED", flush=True)
    return res, ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["mixed", "regress"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--fmt", default="fp8_head")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=64, help="mixed：block 數（預設 64＝32K；小卡用 32 或 16）")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--old", help="舊版 m7_restore_harness.py 的路徑（regress 用）")
    ap.add_argument("--impls", default="new,old", help="regress：建立順序（new,old,new2,old2）；比值＝前兩個（有 new、old 時固定 new/old）")
    ap.add_argument("--cells", help="regress：只跑這些格，例 local:cake,nfs:cake")
    a = ap.parse_args()
    t0 = time.time()
    res, ok = mixed(a) if a.what == "mixed" else regress(a)
    res["run_id"] = os.environ.get("RUN_ID", "adhoc")
    res["wall_s"] = round(time.time() - t0, 1)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    sys.exit(0 if ok else 1)
