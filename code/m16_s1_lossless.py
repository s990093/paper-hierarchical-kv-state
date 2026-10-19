#!/usr/bin/env python3
"""m16_s1_lossless.py — 第三階段 S1：無損壓縮 KV 的壓縮比、編解碼速度與盈虧點（只用 numpy＋zstandard，不碰 torch／GPU）。

計畫：docs/stage3_20261019/experiment_matrix.md 的 S1。資料由 m16_s1_kvdump.py 產生（GPU 窗口跑）。

  python code/m16_s1_lossless.py --self-test
  python code/m16_s1_lossless.py analyze --dump <dir> [<dir> ...] --out <outdir> [--codecs zstd:-3 zstd:1 zlib:1] [--threads 1 4] [--max-blocks 16]
  python code/m16_s1_lossless.py breakeven --summary <outdir>/summary.json --tier cpu=49.9 --tier nvme=7.76 --tier nfs=0.37 ...

## 量什麼
BF16（每個值 16 bit：1 sign、8 exponent、7 mantissa）：
  raw      整個 block 的位元組直接 zstd
  bytes2   高位元組（sign＋7 bit exponent）、低位元組（1 bit exponent＋mantissa）分開 zstd（ZipNN 式的位元組分組）
  planes   exponent（8 bit）一條、sign＋mantissa（8 bit）一條，分開 zstd
  熵       order-0 熵：H(exponent)、H(sign+mantissa)、H(16-bit 符號) → 理想的熵編碼壓縮比上限（逐符號、不看上下文）
低精度格式（m10_codec 的實際 part 位元組）：每個 part 直接 zstd；INT4 的 q 另算 nibble 的熵。
每個組合：K、V 分開＋全部；逐層的壓縮比也記（CSV）。
**每一次壓縮都解回來逐位元比對**（不相等就丟 AssertionError，不寫結果）。

速度（另一組 block 樣本）：把一個 block 切成 frame（預設 4 MiB），k 個 thread 各自壓／解；
時間含 planes 的拆與合（numpy），MB/s 一律以「未壓縮的位元組」計。

## 盈虧點（breakeven，〔計算〕，不是量測）
一段 B 位元組（未壓縮）、鏈路頻寬 BW、壓縮比 r、k 個 thread 的解壓速度 D：
  不壓縮：B/BW；邊讀邊解（pipelined）：max(B/(r·BW), B/D)；讀完才解（serial）：B/(r·BW) + B/D。
只算「讀＋解」；SSD／NFS 層讀進主機記憶體之後還要 H2D（兩邊都一樣，所以不影響比較，但絕對值要另加）。
GPU 端解壓（若 ROCm 有可用的庫）不在這一支的範圍。
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import json
import os
import sys
import threading
import time
import zlib

import numpy as np
import zstandard as zstd


# ---------------------------------------------------------------- 基本工具
def entropy_bits(sym: np.ndarray, nbins: int) -> float:
    c = np.bincount(sym.ravel(), minlength=nbins).astype(np.float64)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())


def bf16_planes(u: np.ndarray):
    """u：uint16（BF16 位元）。回傳 (exp8, sm8)：exponent 8 bit；sign 放 sm8 的最高位、mantissa 7 bit 放低位。"""
    exp8 = ((u >> 7) & 0xFF).astype(np.uint8)
    sm8 = (((u >> 8) & 0x80) | (u & 0x7F)).astype(np.uint8)
    return exp8, sm8


def bf16_merge(exp8: np.ndarray, sm8: np.ndarray) -> np.ndarray:
    e = exp8.astype(np.uint16)
    s = sm8.astype(np.uint16)
    return ((s & 0x80) << 8) | (e << 7) | (s & 0x7F)


def bytes2(u: np.ndarray):
    b = u.view(np.uint8).reshape(-1, 2)          # little-endian：[:,0] 低位元組、[:,1] 高位元組
    return np.ascontiguousarray(b[:, 1]), np.ascontiguousarray(b[:, 0])


def bytes2_merge(hi: np.ndarray, lo: np.ndarray) -> np.ndarray:
    out = np.empty((hi.size, 2), dtype=np.uint8)
    out[:, 0], out[:, 1] = lo, hi
    return out.view(np.uint16).ravel()


def parse_codec(spec: str) -> tuple[str, int]:
    """'zstd:3'、'zstd:-3'、'zlib:1'（zlib＝DEFLATE，和 QAT／IAA 同一種格式）。"""
    name, lv = spec.split(":")
    assert name in ("zstd", "zlib"), spec
    return name, int(lv)


def make_codec(spec: str):
    """回傳 (compress, decompress)；每次呼叫各自建 context（zstd 的 context 不是 thread-safe）。"""
    name, lv = parse_codec(spec)
    if name == "zstd":
        c, d = zstd.ZstdCompressor(level=lv), zstd.ZstdDecompressor()
        return c.compress, d.decompress
    return (lambda b: zlib.compress(b, lv)), zlib.decompress


def roundtrip_len(raw: bytes | np.ndarray, spec: str) -> int:
    """壓縮後的位元組數；解回來逐位元比對。"""
    b = raw.tobytes() if isinstance(raw, np.ndarray) else bytes(raw)
    comp, dec = make_codec(spec)
    c = comp(b)
    assert dec(c) == b, f"{spec} 解回來不相同"
    return len(c)


# BF16 的變體：每個變體＝一組 (stream, 要不要壓)。不壓的 stream 原樣存（位元組照算）。
#   raw       整段壓
#   bytes2    高位元組（sign＋7 bit exponent）、低位元組（1 bit exponent＋7 bit mantissa）分開壓（ZipNN 式）
#   bytes2hi  只壓高位元組、低位元組原樣存（Intel intel-accel-for-llm 的 IAXL_KV_DATA_SHUFFLE＋IAXL_ZIP_EXPONENT_ONLY，見 lit/L0）
#   planes    exponent（8 bit）、sign＋mantissa（8 bit）分開壓
#   planes_e  只壓 exponent，sign＋mantissa 原樣存
VARIANTS = ("raw", "bytes2", "bytes2hi", "planes", "planes_e")


def split_variant(u: np.ndarray, var: str) -> list[tuple[np.ndarray, bool]]:
    if var == "raw":
        return [(u, True)]
    if var in ("bytes2", "bytes2hi"):
        hi, lo = bytes2(u)
        return [(hi, True), (lo, var == "bytes2")]
    if var in ("planes", "planes_e"):
        e8, s8 = bf16_planes(u)
        return [(e8, True), (s8, var == "planes")]
    raise ValueError(var)


def merge_variant(streams: list[np.ndarray], var: str) -> np.ndarray:
    if var == "raw":
        return streams[0]
    if var in ("bytes2", "bytes2hi"):
        return bytes2_merge(streams[0], streams[1])
    return bf16_merge(streams[0], streams[1])


# ---------------------------------------------------------------- 讀 dump
class Dump:
    def __init__(self, d: str):
        self.dir = d
        self.meta = json.load(open(os.path.join(d, "meta.json")))
        m = self.meta
        self.nb = m["n_blocks"]
        self.shape = tuple(m["bf16_block_shape"])            # [L',2,H,512,D]
        self.vals = int(np.prod(self.shape))
        path = os.path.join(d, "kv_bf16.u16")
        assert os.path.getsize(path) == self.nb * self.vals * 2, ("kv_bf16 大小不符 meta", path)
        self.bf16 = np.memmap(path, dtype=np.uint16, mode="r", shape=(self.nb,) + self.shape)
        self.parts = {}
        for key, pm in m["parts"].items():
            p = os.path.join(d, f"{key}.bin")
            assert os.path.getsize(p) == self.nb * pm["bytes_per_block"], ("part 大小不符 meta", key)
            self.parts[key] = np.memmap(p, dtype=np.uint8, mode="r", shape=(self.nb, pm["bytes_per_block"]))

    def fmt_parts(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for key in self.parts:
            f, part = key.split(".", 1)
            out.setdefault(f, []).append(part)
        return out


def sample_blocks(nb: int, k: int | None) -> list[int]:
    if not k or k >= nb:
        return list(range(nb))
    return sorted(set(int(round(x)) for x in np.linspace(0, nb - 1, k)))


# ---------------------------------------------------------------- 壓縮比
def ratios_bf16(dp: Dump, blocks: list[int], codecs: list[str], run_id: str, ts: str, rows: list):
    """逐 block、逐 (K/V/all)、逐變體；另記逐層的 planes 熵（CSV），總和放 summary。"""
    tot = {}          # (kv, variant, codec) -> [raw_bytes, comp_bytes]
    ent = {}          # kv -> {exp, sm, u16} 的符號計數（合併全部 block 再算熵）
    for bi in blocks:
        blk = np.asarray(dp.bf16[bi])                     # [L',2,H,T,D]
        for kv, sel in (("K", blk[:, 0]), ("V", blk[:, 1]), ("all", blk)):
            u = np.ascontiguousarray(sel).ravel()
            e8, s8 = bf16_planes(u)
            assert np.array_equal(bf16_merge(e8, s8), u), "planes 拆合不相同"
            hi, lo = bytes2(u)
            assert np.array_equal(bytes2_merge(hi, lo), u), "bytes2 拆合不相同"
            cnt = ent.setdefault(kv, dict(exp=np.zeros(256, np.int64), sm=np.zeros(256, np.int64),
                                          u16=np.zeros(65536, np.int64)))
            cnt["exp"] += np.bincount(e8, minlength=256)
            cnt["sm"] += np.bincount(s8, minlength=256)
            cnt["u16"] += np.bincount(u, minlength=65536)
            if kv == "all":
                continue                                   # all 的壓縮比＝K＋V 合計（下面加總），不重複壓
            n = u.nbytes
            for var in VARIANTS:
                parts = split_variant(u, var)
                assert np.array_equal(merge_variant([p for p, _ in parts], var), u), f"{var} 拆合不相同"
                for cs in codecs:
                    c = sum(roundtrip_len(p, cs) if z else p.nbytes for p, z in parts)
                    for k2 in (kv, "all"):
                        t = tot.setdefault((k2, var, cs), [0, 0])
                        t[0] += n
                        t[1] += c
        # 逐層熵（只記 planes，給 CSV；這是診斷用）
        for li in range(blk.shape[0]):
            for kvi, kv in ((0, "K"), (1, "V")):
                u = np.ascontiguousarray(blk[li, kvi]).ravel()
                e8, s8 = bf16_planes(u)
                rows.append(dict(run_id=run_id, ts=ts, dump=os.path.basename(dp.dir), model=dp.meta["model"],
                                 task=dp.meta["task"], fmt="bf16", block=bi, layer=dp.meta["layers"][li], kv=kv,
                                 variant="planes_entropy", level="", raw_bytes=u.nbytes, comp_bytes="",
                                 ratio=round(16.0 / (entropy_bits(e8, 256) + entropy_bits(s8, 256)), 4),
                                 h_exp=round(entropy_bits(e8, 256), 4), h_sm=round(entropy_bits(s8, 256), 4)))
    out = {}
    for (kv, var, cs), (n, c) in sorted(tot.items(), key=lambda x: str(x[0])):
        out[f"{kv}|{var}|{cs}"] = dict(raw_bytes=n, comp_bytes=c, ratio=round(n / c, 4))
    for kv, cnt in ent.items():
        def h(c):
            p = c[c > 0] / c.sum()
            return float(-(p * np.log2(p)).sum())
        he, hs, hu = h(cnt["exp"]), h(cnt["sm"]), h(cnt["u16"])
        out[f"{kv}|entropy"] = dict(h_exp=round(he, 4), h_sm=round(hs, 4), h_u16=round(hu, 4),
                                    ideal_ratio_planes=round(16 / (he + hs), 4), ideal_ratio_u16=round(16 / hu, 4))
    return out


def ratios_fmt(dp: Dump, blocks: list[int], codecs: list[str]):
    """每個低精度格式：每個 part 直接 zstd；回傳 fmt -> {stored_bytes, comp_bytes@lv, ratio_vs_fmt, ratio_vs_bf16, nibble 熵}"""
    out = {}
    bf16_per_block = dp.meta["bf16_bytes_per_block"]
    for f, parts in dp.fmt_parts().items():
        res = dict(parts=parts, stored_bytes=0, bf16_bytes=bf16_per_block * len(blocks))
        comp = {cs: 0 for cs in codecs}
        nib = {}
        for bi in blocks:
            for p in parts:
                raw = np.asarray(dp.parts[f"{f}.{p}"][bi])
                res["stored_bytes"] += raw.nbytes
                for cs in codecs:
                    comp[cs] += roundtrip_len(raw, cs)
                if p.endswith("q") and f.startswith("int4"):
                    lo, hi = raw & 0x0F, raw >> 4
                    c = nib.setdefault(p, np.zeros(16, np.int64))
                    c += np.bincount(lo, minlength=16) + np.bincount(hi, minlength=16)
                elif p == "q":
                    c = nib.setdefault(p + "_byte", np.zeros(256, np.int64))
                    c += np.bincount(raw, minlength=256)
        for cs in codecs:
            res[cs] = dict(comp_bytes=comp[cs], ratio_vs_fmt=round(res["stored_bytes"] / comp[cs], 4),
                           ratio_vs_bf16=round(res["bf16_bytes"] / comp[cs], 4))
        res["fmt_ratio_vs_bf16"] = round(res["bf16_bytes"] / res["stored_bytes"], 4)
        for k, c in nib.items():
            p = c[c > 0] / c.sum()
            res[f"entropy_{k}_bits"] = round(float(-(p * np.log2(p)).sum()), 4)
        out[f] = res
    return out


# ---------------------------------------------------------------- 速度
def _frames(a: np.ndarray, frame_bytes: int) -> list[np.ndarray]:
    step = max(1, frame_bytes // a.itemsize)
    return [a[i:i + step] for i in range(0, a.size, step)]


def throughput(dp: Dump, blocks: list[int], variant: str, codec: str, threads: int, frame_bytes: int, reps: int):
    """一個 block 一次：拆 stream、切 frame、k 個 thread 壓、再 k 個 thread 解＋合回。
    回傳中位數 MB/s（未壓縮位元組）與壓縮比。不壓的 stream 只算拆／合的時間。"""
    pool = cf.ThreadPoolExecutor(max_workers=threads)
    # codec 的 context 不是 thread-safe：每個 thread 自己一份（用 job 編號 i % threads 分配會讓兩個 thread 共用同一份，
    # 2026-10-19 自測抓到 zstd 的 "Data corruption detected"）
    tls = threading.local()

    def _ctx():
        if not hasattr(tls, "c"):
            tls.c = make_codec(codec)
        return tls.c

    def comp_job(fr):
        return _ctx()[0](fr.tobytes())

    def dec_job(c):
        return _ctx()[1](c)

    tc, td, ratios = [], [], []
    for _ in range(reps):
        for bi in blocks:
            u = np.ascontiguousarray(dp.bf16[bi]).ravel()
            n = u.nbytes
            t0 = time.perf_counter()
            parts = split_variant(u, variant)
            frs = [(si, f) for si, (s, z) in enumerate(parts) if z for f in _frames(s, frame_bytes)]
            comp = list(pool.map(comp_job, [f for _, f in frs]))
            t1 = time.perf_counter()
            dec = list(pool.map(dec_job, comp))
            outs = []
            for si, (s, z) in enumerate(parts):
                if z:
                    outs.append(np.frombuffer(b"".join(d for (sj, _), d in zip(frs, dec) if sj == si), dtype=s.dtype))
                else:
                    outs.append(s)
            back = merge_variant(outs, variant)
            t2 = time.perf_counter()
            assert np.array_equal(back, u), "速度測試解回來不相同"
            tc.append(n / (t1 - t0) / 1e6)
            td.append(n / (t2 - t1) / 1e6)
            ratios.append(n / (sum(len(c) for c in comp) + sum(s.nbytes for s, z in parts if not z)))
    pool.shutdown()
    return dict(comp_MBps=round(float(np.median(tc)), 1), decomp_MBps=round(float(np.median(td)), 1),
                comp_MBps_iqr=[round(float(np.percentile(tc, 25)), 1), round(float(np.percentile(tc, 75)), 1)],
                decomp_MBps_iqr=[round(float(np.percentile(td, 25)), 1), round(float(np.percentile(td, 75)), 1)],
                ratio_framed=round(float(np.median(ratios)), 4), n=len(tc))


# ---------------------------------------------------------------- 子命令
def cmd_analyze(a):
    run_id = os.environ.get("RUN_ID", "NOT_SET")
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    os.makedirs(a.out, exist_ok=True)
    rows: list[dict] = []
    summary = dict(run_id=run_id, ts=ts, codecs=a.codecs, threads=a.threads, frame_bytes=a.frame_bytes,
                   affinity=sorted(os.sched_getaffinity(0)), loadavg=os.getloadavg(), dumps={})
    for d in a.dump:
        dp = Dump(d)
        rb = sample_blocks(dp.nb, a.max_blocks)
        tb = sample_blocks(dp.nb, a.speed_blocks)
        s = dict(meta={k: dp.meta[k] for k in ("model", "task", "n_tokens", "layers", "run_id", "item_id")},
                 ratio_blocks=rb, speed_blocks=tb)
        t0 = time.perf_counter()
        s["bf16"] = ratios_bf16(dp, rb, a.codecs, run_id, ts, rows)
        s["fmts"] = ratios_fmt(dp, rb, a.codecs)
        s["t_ratio_s"] = round(time.perf_counter() - t0, 1)
        sp = {}
        for var in a.variants:
            for cs in a.speed_codecs or a.codecs:
                for th in a.threads:
                    k = f"{var}|{cs}|t{th}"
                    sp[k] = throughput(dp, tb, var, cs, th, a.frame_bytes, a.reps)
                    print(f"  {os.path.basename(d)} {k}: {sp[k]}", flush=True)
        s["speed"] = sp
        summary["dumps"][os.path.basename(d.rstrip('/'))] = s
        print(json.dumps({k: v for k, v in s["bf16"].items() if k.startswith("all|")}, ensure_ascii=False),
              flush=True)
    with open(os.path.join(a.out, "per_layer.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, ensure_ascii=False)
    print("寫好", a.out)


def cmd_breakeven(a):
    """〔計算〕各層級的「讀＋解」時間：不壓縮 vs 各無損變體 vs 低精度格式（＋zstd）。"""
    S = json.load(open(a.summary))
    tiers = {}
    for t in a.tier:
        k, v = t.split("=")
        tiers[k] = float(v) * 2 ** 30            # GiB/s → B/s
    rows = []
    for dn, s in S["dumps"].items():
        for tn, bw in tiers.items():
            base = 1.0 / bw                      # 每個未壓縮位元組的秒數
            for key, sp in s["speed"].items():
                var, cs, th = key.split("|")
                r = sp["ratio_framed"]             # 和速度同一種切 frame 的方式
                lvs = cs
                D = sp["decomp_MBps"] * 1e6
                pipe = max(1 / (r * bw), 1 / D)
                ser = 1 / (r * bw) + 1 / D
                rows.append(dict(dump=dn, tier=tn, GiBps=bw / 2 ** 30, method=f"{var}|{lvs}|{th}", ratio=r,
                                 gain_pipelined=round(1 - pipe / base, 4), gain_serial=round(1 - ser / base, 4)))
            for f, fr in s["fmts"].items():
                rows.append(dict(dump=dn, tier=tn, GiBps=bw / 2 ** 30, method=f"{f}（有損，不含反量化）",
                                 ratio=fr["fmt_ratio_vs_bf16"], gain_pipelined=round(1 - 1 / fr["fmt_ratio_vs_bf16"], 4),
                                 gain_serial=""))
                for cs in S["codecs"]:
                    rr = fr[cs]["ratio_vs_bf16"]
                    rows.append(dict(dump=dn, tier=tn, GiBps=bw / 2 ** 30, method=f"{f}+{cs}（只算位元組）",
                                     ratio=rr, gain_pipelined=round(1 - 1 / rr, 4), gain_serial=""))
    out = a.out or os.path.join(os.path.dirname(a.summary), "breakeven.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("寫好", out, "（〔計算〕：只算讀＋解；SSD／NFS 另有 H2D）")


# ---------------------------------------------------------------- 自測（合成資料，只驗程式，不產生任何結果）
def self_test() -> bool:
    import tempfile
    ok = True

    def check(name, cond):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name)
        ok &= bool(cond)

    rng = np.random.default_rng(0)
    # 合成 BF16：常態分佈 float32 → 截成 BF16 位元
    x = rng.standard_normal(2 * 2 * 2 * 512 * 64).astype(np.float32)
    u = (x.view(np.uint32) >> 16).astype(np.uint16)
    e8, s8 = bf16_planes(u)
    check("planes 拆合逐位元相同", np.array_equal(bf16_merge(e8, s8), u))
    hi, lo = bytes2(u)
    check("bytes2 拆合逐位元相同", np.array_equal(bytes2_merge(hi, lo), u))
    # exponent 的值要等於 float 的指數欄位
    check("exponent 欄位正確", np.array_equal(e8, ((x.view(np.uint32) >> 23) & 0xFF).astype(np.uint8)))
    check("sign 放在 sm8 最高位", np.array_equal(s8 >> 7, (x < 0).astype(np.uint8)))
    check("常數陣列熵＝0", entropy_bits(np.zeros(100, np.uint8), 256) == 0.0)
    check("均勻 0..255 熵＝8", abs(entropy_bits(np.arange(256 * 40, dtype=np.int64) % 256, 256) - 8.0) < 1e-9)
    # 假 dump：2 個 block，[L=2,2,H=2,512,D=64]
    with tempfile.TemporaryDirectory() as d:
        shape = [2, 2, 2, 512, 64]
        nb = 2
        uu = (rng.standard_normal(nb * int(np.prod(shape))).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
        uu.tofile(os.path.join(d, "kv_bf16.u16"))
        q = rng.integers(0, 256, size=(nb, 1000), dtype=np.uint8)
        q.tofile(os.path.join(d, "int4_kivi.k_q.bin"))
        meta = dict(run_id="SELFTEST", model="synthetic", task="none", n_tokens=1024, layers=[0, 1], item_id=None,
                    n_blocks=nb, bf16_block_shape=shape, bf16_bytes_per_block=int(np.prod(shape)) * 2,
                    parts={"int4_kivi.k_q": dict(dtype="torch.uint8", shape=[1000], bytes_per_block=1000)})
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"))
        dp = Dump(d)
        rows: list[dict] = []
        cs = ["zstd:1", "zlib:1"]
        r = ratios_bf16(dp, [0, 1], cs, "SELFTEST", "t", rows)
        check("raw 壓縮比 > 0", r["all|raw|zstd:1"]["ratio"] > 0)
        check("all 的原始位元組＝K＋V", r["all|raw|zstd:1"]["raw_bytes"] == r["K|raw|zstd:1"]["raw_bytes"] + r["V|raw|zstd:1"]["raw_bytes"])
        # 常態分佈的 BF16：exponent 有結構，planes 應該比 raw 好壓（這是程式邏輯檢查，不是結果）
        check("合成常態資料 planes ≥ raw", r["all|planes|zstd:1"]["ratio"] >= r["all|raw|zstd:1"]["ratio"])
        # 只壓一半（hi／exponent）：另一半原樣存 → 壓縮比 ≤ 2
        check("bytes2hi、planes_e 壓縮比 ≤ 2", r["all|bytes2hi|zlib:1"]["ratio"] <= 2 and r["all|planes_e|zstd:1"]["ratio"] <= 2)
        check("理想壓縮比 ≥ 1", r["all|entropy"]["ideal_ratio_planes"] >= 1.0)
        f = ratios_fmt(dp, [0, 1], cs)
        check("均勻隨機位元組 nibble 熵≈4", abs(f["int4_kivi"]["entropy_k_q_bits"] - 4.0) < 0.05)
        for var in VARIANTS:
            for c in cs:
                sp = throughput(dp, [0, 1], var, c, 2, 64 * 1024, 1)
                check(f"速度測試 {var} {c}：跑完、逐位元相同", sp["n"] == 2 and sp["decomp_MBps"] > 0)
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("analyze")
    p.add_argument("--dump", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--codecs", nargs="+", default=["zstd:-3", "zstd:1", "zstd:3", "zlib:1", "zlib:6"])
    p.add_argument("--speed-codecs", nargs="*", default=["zstd:-3", "zstd:1", "zlib:1"])
    p.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--frame-bytes", type=int, default=4 << 20)
    p.add_argument("--max-blocks", type=int, default=16, help="算壓縮比的 block 數（平均取樣）；0＝全部")
    p.add_argument("--speed-blocks", type=int, default=4)
    p.add_argument("--reps", type=int, default=3)
    p = sub.add_parser("breakeven")
    p.add_argument("--summary", required=True)
    p.add_argument("--tier", action="append", required=True, help="名稱=GiB/s，例如 nfs=0.37（出處寫在呼叫的 job 腳本）")
    p.add_argument("--out")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if a.cmd == "analyze":
        cmd_analyze(a)
    elif a.cmd == "breakeven":
        cmd_breakeven(a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
