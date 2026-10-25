#!/usr/bin/env python3
"""w0.py — 第三階段 W0（05 §8）：降級時才壓縮，來得及嗎。判準：docs/stage3_design_20261024/runs/W0_write_cost.md（登記，不改）。

量：一個 32K 請求的 KV（fa3 prefill，64 塊）壓成「粗底（校準基底 xKV＋量化係數；B16、B32）＋精修（每塊 INT4 KIVI）」要多久。
只量編碼（存下來的東西），不量解碼。
  (i)  CPU：torch.set_num_threads(16)（job 拿到的 16 核），KV 先 D2H 到主機
  (ii) GPU：同一段編碼在 GPU 上；閒時、和背景 decode:32:8192（bgengine，另一條 stream）一起跑時；另記背景吞吐比
量時間：獨佔（runner 檢查）。

  run --model M --out CSV
  report     CPU：對降級需求速率（登記 §2）判 GW0
  --self-test CPU：編碼函式在小張量上和 m16_xstack.t_xl_half_cal 的編碼部分逐位元相同
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
REPO = os.path.dirname(CODE)
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)

CHUNK = 512
OUT = os.path.join(REPO, "results", "s3_h200", "w0")
SPECS = {"B16": (20, 30, "int4_kivi"), "B32": (13, 20, "kv-int4_kivi-int2_kivi")}      # A0 登記 §1.3
FIELDS = ["run_id", "ts", "model", "where", "bg", "part", "rep", "t_s", "bytes_in", "gibps_in", "bytes_out", "threads",
          "bg_ratio", "bg_solo_rate", "host_level"]


def run_id():
    return os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def encode_xl(src, cos, sin, U, rk_he: int, W: int, H: int, f: str, apply_rope=None):
    """xKV 校準基底的編碼（只有存的部分）：每個窗口 Z＝[T, W·H·D]（K 先逆 RoPE、FP32），係數＝Z·U[:, :r]，再用半邊格式 f 量化。
    ＝m16_xstack.t_xl_half_cal 前半段（到 t_halves_enc 為止），回傳 (量化後的 parts, 位元組)。"""
    import torch
    import m16_xstack as XS
    import m10_codec_x as X10
    L, Hh, T, D = src.shape
    r = rk_he * W * H
    As = []
    for w in range(L // W):
        Z = torch.cat([(XS.t_rope_inv(src[l], cos, sin) if apply_rope == "k" else src[l].float()).permute(1, 0, 2).reshape(T, H * D)
                       for l in range(w * W, (w + 1) * W)], dim=1)
        As.append(Z @ U[w, :, :r])
        del Z
    A4 = torch.stack(As).unsqueeze(1).contiguous()
    ff = XS.ALIAS.get(f, f)
    if ff in (None, "bf16"):
        b = A4.to(torch.bfloat16)
        return {"x": b}, b.numel() * 2
    p = X10.enc_half(A4, ff, "k")
    return p, sum(v.numel() * v.element_size() for v in p.values())


def compress(X, cos, sin, calib, H: int, which: str):
    """X：區內 KV [L, 2, H, T, D] BF16（CPU 或 GPU）。which＝B16／B32（粗底）或 INT4（每塊精修）。回傳輸出位元組。"""
    import m16_xstack as XS
    import m10_codec_x as X10
    if which == "INT4":
        nb = 0
        for c0 in range(0, X.shape[3], CHUNK):
            e = X10.encode(X[:, :, :, c0:c0 + CHUNK].contiguous(), "int4_kivi")
            nb += e.nbytes
        return nb
    rk, rv, fmt = SPECS[which]
    hk, hv = XS.halves(fmt)
    _, bk = encode_xl(X[:, 0], cos, sin, calib["k_U"], rk, 4, H, hk, apply_rope="k")
    _, bv = encode_xl(X[:, 1], cos, sin, calib["v_U"], rv, 4, H, hv, apply_rope=None)
    return bk + bv


def cmd_run(a):
    import torch
    os.environ["M7_MODEL_GLOB"] = os.environ.get(f"M7_{a.model.upper()}_GLOB", "")
    from m7_model import KVModel
    from bgengine import Background
    from m16_c1a_predict import host_level
    import m16_xstack as XS
    km = KVModel(max_len=64 * CHUNK + 1024)
    g = torch.Generator().manual_seed(11)
    ids = torch.randint(1000, min(120000, km.cfg.vocab_size), (64 * CHUNK,), generator=g).to(km.device)
    km.set_attn("fa3")
    km.prefill_chunked(ids)
    torch.cuda.synchronize()
    km.set_attn("sdpa")
    s0, s1 = CHUNK, 64 * CHUNK                                   # 區內＝block 1..63（和 A0 同一個口徑）
    Xg = km.kv[:, :, :, s0:s1].contiguous()
    bytes_in = Xg.numel() * 2
    cx = XS.TCtx(km, 64 * CHUNK)
    P = torch.arange(s0, s1, device=km.device)
    cos_g, sin_g = cx.cs(P)
    cal = torch.load(a.calib_xl, map_location=km.device)
    if cal["meta"]["model"] != a.model:
        raise SystemExit(f"校準檔 {a.calib_xl} 是 {cal['meta']['model']}")
    rid = run_id()
    rows = []
    lvl = host_level()

    def rec(where, bg, part, rep, t, bout, threads, br="", solo=""):
        rows.append(dict(run_id=rid, ts=ts(), model=a.model, where=where, bg=bg, part=part, rep=rep, t_s=round(t, 5),
                         bytes_in=bytes_in, gibps_in=round(bytes_in / t / 2**30, 3), bytes_out=bout, threads=threads,
                         bg_ratio=br, bg_solo_rate=solo, host_level=lvl))
        print(json.dumps(rows[-1]), flush=True)
    # (i) CPU
    torch.set_num_threads(a.threads)
    Xc = Xg.cpu()
    cos_c, sin_c = cos_g.cpu(), sin_g.cpu()
    calc = {k: (v.cpu() if hasattr(v, "cpu") else v) for k, v in cal.items()}
    for part in ("B16", "B32", "INT4"):
        compress(Xc, cos_c, sin_c, calc, km.H, part)                  # 暖機
        for rep in range(a.reps):
            t0 = time.perf_counter()
            b = compress(Xc, cos_c, sin_c, calc, km.H, part)
            rec("cpu", "idle", part, rep, time.perf_counter() - t0, b, a.threads)
    # (ii) GPU：閒、和 decode:32:8192 一起
    side = torch.cuda.Stream(device=km.device)
    for bgspec in ("idle", "decode:32:8192"):
        bg = Background(km, bgspec)
        solo = bg.solo_rate(2.0)
        bg.start()
        time.sleep(0.5)
        for part in ("B16", "B32", "INT4"):
            with torch.cuda.stream(side):
                compress(Xg, cos_g, sin_g, cal, km.H, part)
            side.synchronize()
            for rep in range(a.reps):
                t0 = time.perf_counter()
                with torch.cuda.stream(side):
                    b = compress(Xg, cos_g, sin_g, cal, km.H, part)
                side.synchronize()
                t1 = time.perf_counter()
                n = sum(1 for t in bg._t if t0 <= t <= t1) if bgspec != "idle" else 0
                br = round(n / (t1 - t0) / solo, 4) if bgspec != "idle" and solo else ""
                rec("gpu", bgspec, part, rep, t1 - t0, b, "", br, round(solo, 3) if solo else "")
        bg.stop()
        del bg
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


DEMAND = {   # 登記 §2：HBM 滿時每秒要降多少 GiB＝prefill 產生 KV 的速度（tokens/s × 每 token 的 KV 位元組）
    "llama": dict(tok_s=32768 / 1.224, kv_per_token=131072, src="RUNLOG_H200.md R2-C6：vLLM 0.28 冷 prefill 32K 1.224 s"),
    "qwen": dict(tok_s=32768 / 1.3934, kv_per_token=57344, src="ICPE P0-1 S1：harness fa3 calib-f Σ64＝1.3934 s（vLLM 的 Qwen prefill 在 H200 沒量過）"),
}


def cmd_report(a):
    import glob
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, "raw", "*.csv"))):
        rows += list(csv.DictReader(open(p)))
    out = dict(run_id=run_id(), ts=ts(), cells={}, verdict={})
    for m, d in DEMAND.items():
        dem = d["tok_s"] * d["kv_per_token"] / 2**30
        cpu = {}
        for part in ("B16", "B32", "INT4"):
            t = [float(r["t_s"]) for r in rows if r["model"] == m and r["where"] == "cpu" and r["part"] == part]
            cpu[part] = statistics.median(t) if t else None
        if any(v is None for v in cpu.values()):
            out["verdict"][m] = "NOT_MEASURED"
            continue
        bytes_in = float(next(r["bytes_in"] for r in rows if r["model"] == m))
        tot = {b: cpu[b] + cpu["INT4"] for b in ("B16", "B32")}                 # 粗底＋精修都要做
        gibps = {b: bytes_in / t / 2**30 for b, t in tot.items()}
        out["cells"][m] = dict(demand_gibps=dem, demand_src=d["src"], cpu_s=cpu, cpu_total_s=tot, cpu_gibps=gibps)
        out["verdict"][m] = "GW0_PASS" if min(gibps.values()) >= dem else "GW0_FAIL"
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "w0_report.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out["verdict"], ensure_ascii=False))


def self_test() -> bool:
    import torch
    import m16_xstack as XS
    torch.manual_seed(0)
    L, H, T, D, W = 4, 2, 64, 16, 4
    src = torch.randn(L, H, T, D).to(torch.bfloat16)
    U = torch.linalg.qr(torch.randn(L // W, W * H * D, W * H * D))[0]
    cos = torch.ones(1, T, D, dtype=torch.bfloat16)
    sin = torch.zeros(1, T, D, dtype=torch.bfloat16)

    class Cx:
        xlcal = {"k_U": U, "v_U": U}

        def cs(self, P):
            return cos, sin
        apply_rope = None
    p, nb = encode_xl(src, cos, sin, U, 2, W, H, "kivi4", apply_rope=None)
    # 參考：t_xl_half_cal 裡同一段（V 那一半，不做 RoPE）的位元組
    _, nb_ref, _ = XS.t_xl_half_cal(src, torch.arange(T), 1, 2, W, "kivi4", type("C", (), {"xlcal": {"v_U": U}, "cs": lambda s, P: (cos, sin),
                                                                                        "apply_rope": None})())
    ok = nb == nb_ref
    print("self-test", "PASS" if ok else f"FAIL（{nb} vs {nb_ref}）")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("run")
    s.add_argument("--model", choices=["llama", "qwen"], required=True)
    s.add_argument("--threads", type=int, default=16)
    s.add_argument("--reps", type=int, default=6)
    s.add_argument("--calib-xl", default="")
    s.add_argument("--out", required=True)
    sub.add_parser("report")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if a.cmd == "run" and not a.calib_xl:
        a.calib_xl = os.path.join(os.environ.get("S3_LOCAL", "/tmp/hungweilai-s3"), "xstack", f"calib_xl_{a.model}.pt")
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
