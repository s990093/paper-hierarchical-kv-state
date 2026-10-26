#!/usr/bin/env python3
"""b0.py — 第三階段 B0（05 §5）＋G-M08（mechanisms/06 §2）：像真的服務的背景下，還原的各種基本動作被拖慢多少、
分 SM／改 stream 優先權能換到多少。判準：docs/stage3_design_20261024/runs/B0_interference.md（登記，不改）。

量時間：獨佔（節點上不能有我們自己的別的 job；runner 檢查）。背景＝bgengine.py（vLLM 式的分頁 KV＋fa3，CUDA graph）。

  env        GPU（P0e）：stream 優先權範圍、green context 能不能用、MPS 控制程式在不在 → JSON
  bgcheck    GPU（P0b 自檢）：每種背景單獨跑的每秒步數、每秒 token、HBM 佔用 → CSV
  run        GPU：一個 (模型, setup, 共用方式) 的整張格子：背景 × 還原動作 × ≥6 次 → CSV（每列有 run_id、ts）
  bgproc     GPU：MPS 模式的背景（另一個 process）：每 0.25 s 印一行「perf_counter 步數」
  report     CPU：干擾矩陣、GB0-FLIP／SHARE／STATIC、G-M08 的判定
  --self-test CPU：格子、隨機順序、判定邏輯（合成資料）

setup：
  b0   回來的 session 32K（64 塊），背景＝閒、decode B＝8／32／64（Llama）或 128（Qwen）×8K、混合批次、prefill 節點、matmul、HBM 複製
  m08  回來的 session 128K（Llama）／256K（Qwen），背景＝長 context decode（Llama 128K B＝4；Qwen 256K B＝4、6）、新文件的 chunked prefill
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
REPO = os.path.dirname(CODE)
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)
import numa_probe as NP                                           # noqa: E402（只記錄 NUMA 位置，不進判定）
import guard as GD                                                # noqa: E402（量時間的 run 的 GPU 監看，CLAUDE.md §3）

OUT = os.path.join(REPO, "results", "s3_h200", "b0")
CHUNK = 512
INFO = {"llama": dict(L=32, H=8, Hq=32, D=128), "qwen": dict(L=28, H=4, Hq=28, D=128)}
BGS = {
    ("b0", "llama"): ["idle", "decode:8:8192", "decode:32:8192", "decode:64:8192", "mixed:512:32:8192", "cprefill:2048:65536",
                      "matmul", "hbm"],
    ("b0", "qwen"): ["idle", "decode:8:8192", "decode:32:8192", "decode:128:8192", "mixed:512:32:8192", "cprefill:2048:65536",
                     "matmul", "hbm"],
    ("m08", "llama"): ["idle", "decode:4:129024", "cprefill:2048:65536"],
    ("m08", "qwen"): ["idle", "decode:4:258048", "decode:6:258048", "cprefill:2048:65536"],
}
REALISTIC = ("decode:", "mixed:", "cprefill:")                 # 05 §5.3「像真實服務的背景」
CTX = {("b0", "llama"): 64, ("b0", "qwen"): 64, ("m08", "llama"): 252, ("m08", "qwen"): 504}
PRIMS_B0 = ["h2d_contig", "h2d_perblock", "h2d_batch", "sm_gather_12", "sm_gather_32", "sm_gather_132", "dq_int4", "dq_int4_torch",
            "dq_fp8", "dq_nvfp4", "dq_vq", "lr_recon", "recompute1", "e2e_cake_bf16_cpu", "e2e_load_bf16_cpu", "e2e_cake_int4_cpu",
            "e2e_load_int4_cpu", "e2e_cake_bf16_b32", "e2e_load_bf16_b32", "e2e_cake_int4_b32", "e2e_load_int4_b32"]
PRIMS_M08 = ["recompute1", "qprefill", "dq_int4", "dq_vq", "lr_recon", "h2d_batch"]
PRODUCTION = ("h2d_contig", "h2d_perblock", "h2d_batch", "sm_gather_12", "sm_gather_32", "sm_gather_132", "dq_int4", "dq_int4_torch",
              "dq_fp8", "dq_nvfp4", "recompute1", "qprefill")        # 05 §5.2 / 08 §3：只用生產格式（沒有低秩重建）
COMPUTE_PRIMS = ("recompute1", "qprefill", "lr_recon")            # G-M08 的「計算型」還原
UNIT_BLOCKS = {"b0": 64, "m08": 16}                              # 每個動作處理幾塊（m08 用 16 塊當一個單位）
MPS_PCTS = (100, 87, 75, 50)
MPS_MARGIN = 6 << 30                                              # 背景 process 的 CUDA context、activation、graph、MPS server 的餘量
WINDOW_S = 1.0                                                    # 一次「量」的最短時間窗（背景吞吐要夠多步才準）
FIELDS = ["run_id", "ts", "model", "setup", "ctx_tokens", "bg", "bg_tokens_per_step", "mode", "prim", "production", "rep",
          "order_idx", "n_exec", "t_med_s", "t_p95_s", "t_min_s", "t_window_s", "bg_steps", "bg_rate", "bg_solo_rate", "bg_ratio",
          "n_rep", "sigma_rel", "delta_star", "p95_sigma_rel", "p95_delta_star", "bg_solo_sigma_rel", "bg_delta_star",
          "unit_blocks", "bytes", "seed", "gpu_mem_gib", "host_level", "own_gpu_intruders", "note"] + NP.ROW_FIELDS
NM = "NOT_MEASURED"
MIN_NOISE_REPS = 6                                                # 05 §2.1：σ 要 ≥6 次
N_BOOT = 10000


def noise(vals: list) -> tuple:
    """05 §2.1：(sigma_rel, delta_star)；少於 6 次 → NOT_MEASURED。"""
    v = [float(x) for x in vals]
    if len(v) < MIN_NOISE_REPS or statistics.median(v) == 0:
        return NM, NM
    sr = statistics.stdev(v) / statistics.median(v)
    return round(sr, 6), round(max(0.05, 2 * sr), 6)


def boot_se_median(vals: list, seed: int = 0) -> float | None:
    """median 的 bootstrap 標準誤（對 rep 重抽 N_BOOT 次）。"""
    import numpy as np
    v = np.asarray([float(x) for x in vals])
    if v.size < MIN_NOISE_REPS:
        return None
    rng = np.random.default_rng(seed)
    return float(np.median(v[rng.integers(0, v.size, size=(N_BOOT, v.size))], axis=1).std(ddof=1))


def p95_noise(p95s: list) -> tuple:
    """P95 的雜訊：P95 的估計＝各 rep 的 P95 的中位數；σ＝它的 bootstrap 標準誤 ÷ 估計（05 §2.1 第 2 點）。"""
    se = boot_se_median(p95s)
    if se is None:
        return NM, NM
    m = statistics.median(float(x) for x in p95s)
    sr = se / m if m else None
    return (NM, NM) if sr is None else (round(sr, 6), round(max(0.05, 2 * sr), 6))


def ratio_ci(a: list, b: list, seed: int = 0) -> tuple:
    """median(a)／median(b) 的 95% CI（兩臂各自對 rep 重抽 N_BOOT 次）。"""
    import numpy as np
    A, B = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    ma = np.median(A[rng.integers(0, A.size, size=(N_BOOT, A.size))], axis=1)
    mb = np.median(B[rng.integers(0, B.size, size=(N_BOOT, B.size))], axis=1)
    r = ma / mb
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def median_ci(a: list, seed: int = 0) -> tuple:
    import numpy as np
    A = np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    m = np.median(A[rng.integers(0, A.size, size=(N_BOOT, A.size))], axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def run_id() -> str:
    return os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"


def ts() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def plan_order(prims: list, seed: int) -> list:
    """每個 (背景, 共用方式) 區塊裡，動作的順序隨機（05 §2：配置順序隨機；種子記在每列）。"""
    r = random.Random(seed)
    p = list(prims)
    r.shuffle(p)
    return p


# ============================================================================ GPU：目標 session 與還原動作
class Target:
    """回來的 session：KV（prefill，fa3）、主機上的 BF16 副本、各格式的編碼（放 GPU，只量反量化）、xKV 係數。"""

    def __init__(self, model: str, setup: str, calib_xl: str):
        import torch
        os.environ["M7_MODEL_GLOB"] = os.environ.get(f"M7_{model.upper()}_GLOB", "")
        if not os.environ["M7_MODEL_GLOB"]:
            raise SystemExit(f"沒有 M7_{model.upper()}_GLOB（runner 會設）")
        from m7_model import KVModel
        self.torch, self.model, self.setup = torch, model, setup
        self.nb = CTX[(setup, model)]
        self.unit = UNIT_BLOCKS[setup]
        self.km = km = KVModel(max_len=self.nb * CHUNK + 1024)
        g = torch.Generator().manual_seed(7)
        self.ids = torch.randint(1000, min(120000, km.cfg.vocab_size), (self.nb * CHUNK + 512,), generator=g).to(km.device)
        km.set_attn("fa3")
        km.prefill_chunked(self.ids[:self.nb * CHUNK])
        torch.cuda.synchronize()
        u = self.unit
        # 動作作用的塊：m08＝block 1..u（block 0 不動）；b0 的 u＝64＝整個 session → block 0..63（不能超出 context：
        # 10/25 冒煙 22238 用了 1..64，lr_recon 取 RoPE 的 cos／sin 越界）
        self.blocks = list(range(1, 1 + u)) if 1 + u <= self.nb else list(range(0, u))
        assert self.blocks[-1] < self.nb, (self.blocks[-1], self.nb)
        self.ref = [km.chunk_view(i).clone() for i in self.blocks]
        self.host = [km.host_chunk() for _ in self.blocks]
        for h, r in zip(self.host, self.ref):
            h.copy_(r)
        self.contig_host = torch.empty((km.L, 2, km.H, u * CHUNK, km.D), dtype=torch.bfloat16, pin_memory=True)
        self.contig_dev = torch.empty((km.L, 2, km.H, u * CHUNK, km.D), dtype=torch.bfloat16, device=km.device)
        import m10_codec as C
        import m10_codec_x as X10
        self.C, self.X10 = C, X10
        self.enc = {f: [X10.encode(r, f) for r in self.ref] for f in ("int4_kivi", "fp8_token", "nvfp4")}
        self.bytes_block = km.chunk_bytes
        # 搬運的描述子：每個（block, 層, K/V, KV head）一段連續的 512×D BF16（128 KiB）
        src, dst = [], []
        for bi, i in enumerate(self.blocks):
            for l in range(km.L):
                for kv in range(2):
                    for h in range(km.H):
                        src.append(self.host[bi][l, kv, h].data_ptr())
                        dst.append(km.kv[l, kv, h, i * CHUNK:(i + 1) * CHUNK].data_ptr())
        self.d_src = torch.tensor(src, dtype=torch.int64)
        self.d_dst = torch.tensor(dst, dtype=torch.int64)
        self.d_size = torch.full((len(src),), CHUNK * km.D * 2, dtype=torch.int64)
        self.d_src_dev, self.d_dst_dev, self.d_size_dev = (t.to(km.device) for t in (self.d_src, self.d_dst, self.d_size))
        self._xl = None
        self.calib_xl = calib_xl
        self.bench = None
        self.numa, self.numa_log = None, {}

    # ---- NUMA（只記錄）：主機緩衝的頁在哪個節點；e2e 的 Bench 第一次用到才建，建好再查一次 ----
    def numa_buffers(self) -> dict:
        b = {"contig_host": self.contig_host, "host_blocks": self.host}
        if self.bench is not None:
            b.update(bench_bf=self.bench.bf, bench_lp=self.bench.lp)
        return b

    def probe_numa(self, when: str) -> dict:
        self.numa = NP.probe(self.torch.cuda.current_device(), self.numa_buffers(),
                             smi=self.numa["gpus_visible"] if self.numa else None)
        self.numa_log[when] = self.numa
        return self.numa

    # ---- G-M11：查表式 VQ（product quantization；runs/G_M11_vq.md）----
    def vq(self, d_sub: int = 4, k: int = 256, iters: int = 6, sample: int = 65536):
        """每個（層, K/V, 子空間）一本 k 項的碼本（d_sub 維）：從這個 session 的 KV 抽 sample 個向量跑 iters 次 Lloyd；
        碼＝uint8 [nblk, L, 2, H, 512, D/d_sub]。解碼＝查表（gather），不用 tensor core。每個值 8/d_sub bit。"""
        if getattr(self, "_vq", None) is None:
            torch, km = self.torch, self.km
            L, H, D = km.L, km.H, km.D
            S = D // d_sub
            X = torch.stack(self.ref)                                            # [nblk, L, 2, H, 512, D]
            g = torch.Generator(device=km.device).manual_seed(5)
            cb = torch.empty(L, 2, S, k, d_sub, dtype=torch.bfloat16, device=km.device)
            codes = torch.empty(X.shape[:-1] + (S,), dtype=torch.uint8, device=km.device)
            for l in range(L):
                for kv in range(2):
                    V = X[:, l, kv].reshape(-1, S, d_sub).float()                # [N, S, d]
                    idx = torch.randint(0, V.shape[0], (min(sample, V.shape[0]),), generator=g, device=km.device)
                    P = V[idx].transpose(0, 1).contiguous()                      # [S, n, d]
                    C = P[:, torch.randperm(P.shape[1], generator=g, device=km.device)[:k]].clone()   # [S, k, d]
                    for _ in range(iters):
                        a = torch.cdist(P, C).argmin(-1)                         # [S, n]
                        for si in range(S):
                            sums = torch.zeros(k, d_sub, device=km.device).index_add_(0, a[si], P[si])
                            cnt = torch.bincount(a[si], minlength=k).clamp_min(1).unsqueeze(1).float()
                            C[si] = torch.where(torch.bincount(a[si], minlength=k).unsqueeze(1) > 0, sums / cnt, C[si])
                    cb[l, kv] = C.to(torch.bfloat16)
                    Cb = cb[l, kv].float()                                       # 用存下來的 BF16 碼本編碼（和解碼一致）
                    cc_all = torch.empty(V.shape[0], S, dtype=torch.uint8, device=km.device)
                    for c0 in range(0, V.shape[0], 1 << 18):
                        part = V[c0:c0 + (1 << 18)].transpose(0, 1)               # [S, m, d]
                        cc_all[c0:c0 + (1 << 18)] = torch.cdist(part, Cb).argmin(-1).to(torch.uint8).T
                    codes[:, l, kv] = cc_all.view(codes.shape[0], H, CHUNK, S)
            self._vq = dict(cb=cb, codes=codes, S=S, d=d_sub, k=k,
                            nbytes=int(codes.numel() + cb.numel() * 2 // len(self.blocks)))
        return self._vq

    def dq_vq(self):
        torch, km, V = self.torch, self.km, self.vq()
        cb, codes, S, d = V["cb"], V["codes"], V["S"], V["d"]
        ar = torch.arange(S, device=km.device)
        for bi, i in enumerate(self.blocks):
            blk = km.chunk_view(i)                                                # [L, 2, H, 512, D]
            for l in range(km.L):
                for kv in range(2):
                    c = codes[bi, l, kv].long()                                   # [H, 512, S]
                    blk[l, kv].copy_(cb[l, kv][ar, c].reshape(km.H, CHUNK, km.D))

    # ---- xKV（B16）的係數：事先編碼，只量解碼（係數反量化＋乘基底＋RoPE）----
    def xl(self):
        if self._xl is None:
            torch = self.torch
            import m16_xstack as XS
            km = self.km
            d = torch.load(self.calib_xl, map_location=km.device)
            cx = XS.TCtx(km, self.nb * CHUNK)
            cx.xlcal = d
            W, rk, rv = 4, 20, 30                                      # B16＝xl:k20:v30:cal+q:int4_kivi（A0 登記 §1.3）
            s0, s1 = self.blocks[0] * CHUNK, (self.blocks[-1] + 1) * CHUNK
            P = torch.arange(s0, s1, device=km.device)
            cos, sin = cx.cs(P)
            parts = {}
            for hi, r in ((0, rk), (1, rv)):
                src = km.kv[:, hi, :, s0:s1]
                L, H, T, D = src.shape
                R = r * W * H
                U = d[f"{'kv'[hi]}_U"]
                As, Bs = [], []
                for w in range(L // W):
                    Z = torch.cat([(XS.t_rope_inv(src[l], cos, sin) if hi == 0 else src[l].float()).permute(1, 0, 2).reshape(T, H * D)
                                   for l in range(w * W, (w + 1) * W)], dim=1)
                    As.append(Z @ U[w, :, :R])
                    Bs.append(U[w, :, :R].T.contiguous().to(torch.bfloat16))
                A4 = torch.stack(As).unsqueeze(1).contiguous()
                parts[hi] = (self.X10.enc_half(A4, "kivi4", "k"), tuple(A4.shape), Bs)
            self._xl = dict(parts=parts, cos=cos, sin=sin, W=W, s0=s0, s1=s1, cx=cx)
        return self._xl

    def lr_recon(self):
        torch, km, X = self.torch, self.km, self.xl()
        import m16_xstack as XS
        W, s0, s1 = X["W"], X["s0"], X["s1"]
        T = s1 - s0
        for hi in (0, 1):
            p, shape, Bs = X["parts"][hi]
            Ad = self.X10.dec_half(p, "kivi4", "k", shape)
            for w in range(km.L // W):
                Zh = Ad[w, 0] @ Bs[w].float()
                for j in range(W):
                    x = Zh[:, j * km.H * km.D:(j + 1) * km.H * km.D].reshape(T, km.H, km.D).permute(1, 0, 2)
                    y = XS.t_rope_fwd(x, X["cos"], X["sin"], km.apply_rope) if hi == 0 else x.to(torch.bfloat16)
                    km.kv[w * W + j, hi, :, s0:s1].copy_(y)

    # ---- 端到端（m10_speed.Bench：Cake／全部載入；CPU 層真的搬、b32 照 MI300X tier_params 限速）----
    def e2e(self, plan: str):
        if self.bench is None:
            import m10_speed as SP
            self.bench = SP.Bench(self.model, self.nb, "int4_kivi", 512)
            self.bench.km.set_attn("fa3")
            self.bench.capture(self.nb)
            # 限速層 b32＝會合點 b*＝32 反推的頻寬（m10_speed calib-f）；H200 fa3 的那一份（ICPE P0-1 S1）
            bw = json.load(open(os.path.join(REPO, "results", "icpe_h200", "p01", "calib", f"speed_{self.model}", "bw.json")))
            self.tiers = {"cpu": dict(kind="cpu", name="cpu"),
                          "b32": dict(kind="throttle", name="b32", GiBps=float(bw["targets"]["b32"]["GiBps"]))}
            self.bw_run_id = bw["run_id"]
            self.probe_numa("bench")                                # 在暖機那一次裡（不計時）
        _, mode, fmt, tier = plan.split("_")
        set_rs_priority(self.bench.rs, high=getattr(self, "mode", "stream") == "stream_hi")
        lp = [fmt == "int4"] * self.nb
        lp[0] = False
        return self.bench.restore(lp, self.tiers[tier], "claim" if mode == "cake" else "loadall", self.nb)

    def run(self, prim: str):
        """在目前的 stream 上做一次動作（呼叫端同步、計時）。"""
        torch, km = self.torch, self.km
        if prim == "h2d_contig":
            self.contig_dev.copy_(self.contig_host, non_blocking=True)
        elif prim == "h2d_perblock":
            for i, h in zip(self.blocks, self.host):
                km.chunk_view(i).copy_(h, non_blocking=True)
        elif prim == "h2d_batch":
            from vllm import _custom_ops as ops
            ops.swap_blocks_batch(self.d_src, self.d_dst, self.d_size)
        elif prim.startswith("sm_gather_"):
            from vllm.v1.kv_offload.cpu.swap_blocks_triton import _swap_blocks_kernel
            n = len(self.d_src)
            _swap_blocks_kernel[(min(int(prim.split("_")[-1]), n),)](self.d_src_dev, self.d_dst_dev, self.d_size_dev, n,
                                                                       BYTES_PER_CHUNK=8192)
        elif prim == "dq_int4":
            from m10_dequant_fast import decode_fast
            for i, e in zip(self.blocks, self.enc["int4_kivi"]):
                decode_fast(e, out=km.chunk_view(i))
        elif prim == "dq_int4_torch":
            for i, e in zip(self.blocks, self.enc["int4_kivi"]):
                self.C.decode(e, out=km.chunk_view(i))
        elif prim == "dq_fp8":
            for i, e in zip(self.blocks, self.enc["fp8_token"]):
                self.C.decode(e, out=km.chunk_view(i))
        elif prim == "dq_nvfp4":
            for i, e in zip(self.blocks, self.enc["nvfp4"]):
                self.X10.decode(e, out=km.chunk_view(i))
        elif prim == "dq_vq":
            self.dq_vq()
        elif prim == "lr_recon":
            self.lr_recon()
        elif prim == "recompute1":
            i = self.nb // 2                                       # 中間那一塊（看 nb/2 塊的 key）
            km.forward_span(self.ids[i * CHUNK:(i + 1) * CHUNK], i * CHUNK)
        elif prim == "qprefill":
            n = self.nb * CHUNK
            km.forward_span(self.ids[n:n + 512], n)
        elif prim.startswith("e2e_"):
            self.e2e(prim)
        else:
            raise KeyError(prim)

    def bytes_of(self, prim: str) -> int:
        u = self.unit * self.bytes_block
        if prim.startswith(("h2d", "sm_gather")):
            return u
        if prim == "dq_vq":
            return self.vq()["codes"].numel() // 1 + 0
        if prim.startswith("dq_"):
            f = {"dq_int4": "int4_kivi", "dq_int4_torch": "int4_kivi", "dq_fp8": "fp8_token", "dq_nvfp4": "nvfp4"}[prim]
            return sum(e.nbytes for e in self.enc[f])
        return 0

    def check(self) -> dict:
        """動作做完的 KV 要對：搬運＝原值逐位元；反量化＝m10_codec 的解碼逐位元（fused 和 torch 版相同）。"""
        torch, km = self.torch, self.km
        out = {}
        for prim in ("h2d_perblock", "h2d_batch", "sm_gather_12", "sm_gather_132"):
            for i in self.blocks:
                km.chunk_view(i).zero_()
            self.run(prim)
            torch.cuda.synchronize()
            out[prim] = all(torch.equal(km.chunk_view(i), r) for i, r in zip(self.blocks, self.ref))
        ref4 = []
        for i, e in zip(self.blocks, self.enc["int4_kivi"]):
            self.C.decode(e, out=km.chunk_view(i))
            ref4.append(km.chunk_view(i).clone())
        self.run("dq_int4")
        torch.cuda.synchronize()
        out["dq_int4_fused_equal_torch"] = all(torch.equal(km.chunk_view(i), r) for i, r in zip(self.blocks, ref4))
        V = self.vq()
        self.run("dq_vq")
        torch.cuda.synchronize()
        bi, l, kv = 0, 0, 1
        c = V["codes"][bi, l, kv].long()
        want = V["cb"][l, kv][torch.arange(V["S"], device=km.device), c].reshape(km.H, CHUNK, km.D)
        out["dq_vq_equal_lookup"] = bool(torch.equal(km.chunk_view(self.blocks[bi])[l, kv], want))
        out["vq_relerr_block0"] = round(float((km.chunk_view(self.blocks[0]).float() - self.ref[0].float()).norm()
                                              / self.ref[0].float().norm()), 4)
        for i, r in zip(self.blocks, self.ref):
            km.chunk_view(i).copy_(r)
        return out


def set_rs_priority(rs, high: bool):
    """Restorer 的重算、搬運、拆 stream 換成高優先權（stream_hi）或一般（其他）。Restorer.set_options 只會往高的方向切，這裡兩個方向都要。"""
    import torch
    key = "hi" if high else "normal"
    if getattr(rs, "_s3_prio", None) == key:
        return
    lo, hi = torch.cuda.Stream.priority_range()
    pr = hi if high else 0
    dev = rs.km.device
    rs.cstream = torch.cuda.Stream(device=dev, priority=pr)
    rs.iostream = torch.cuda.Stream(device=dev, priority=pr)
    rs._h2d_s = torch.cuda.Stream(device=dev, priority=pr)
    rs._s3_prio = key


# ============================================================================ GPU：量
def measure_block(tgt, bg, prims, reps, seed, rate_fn, solo, stream, w, base_row):
    """一個 (背景, 共用方式) 區塊：動作照隨機順序，每個做 1 次暖機＋reps 次「時間窗」；每個時間窗至少 WINDOW_S 秒、
    內含 n_exec 次背靠背執行（每次之間同步、計時）。背景吞吐用時間窗的起訖算。"""
    torch = tgt.torch
    order = plan_order(prims, seed)
    for oi, prim in enumerate(order):
        with torch.cuda.stream(stream):
            tgt.run(prim)
        stream.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.stream(stream):
            tgt.run(prim)
        stream.synchronize()
        t_est = max(1e-5, time.perf_counter() - t0)
        n_exec = max(1, math.ceil(WINDOW_S / t_est))
        rows = []
        for rep in range(reps):
            ts_ = []
            w0 = time.perf_counter()
            for _ in range(n_exec):
                a = time.perf_counter()
                with torch.cuda.stream(stream):
                    tgt.run(prim)
                stream.synchronize()
                ts_.append(time.perf_counter() - a)
            w1 = time.perf_counter()
            steps, rate = rate_fn(w0, w1)
            ts_.sort()
            rows.append(dict(base_row, prim=prim, production=int(prim in PRODUCTION), rep=rep, order_idx=oi, n_exec=n_exec,
                             t_med_s=statistics.median(ts_), t_p95_s=ts_[min(len(ts_) - 1, math.ceil(0.95 * len(ts_)) - 1)],
                             t_min_s=ts_[0], t_window_s=w1 - w0, bg_steps=steps, bg_rate=rate, bg_solo_rate=solo,
                             bg_ratio=(rate / solo if solo else ""), unit_blocks=tgt.unit, bytes=tgt.bytes_of(prim), seed=seed,
                             gpu_mem_gib=round(torch.cuda.max_memory_allocated() / 2**30, 1), ts=ts(), **NP.row_fields(tgt.numa)))
        sr, ds = noise([r["t_med_s"] for r in rows])
        psr, pds = p95_noise([r["t_p95_s"] for r in rows])
        for r in rows:
            r.update(n_rep=len(rows), sigma_rel=sr, delta_star=ds, p95_sigma_rel=psr, p95_delta_star=pds)
            w.writerow(r)


def cmd_run(a):
    import torch
    from bgengine import Background
    from m16_c1a_predict import host_level
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tgt = Target(a.model, a.setup, a.calib_xl)
    print("[b0] 正確性：", json.dumps(tgt.check()), flush=True)
    print("[b0] NUMA：", json.dumps(tgt.probe_numa("start")["summary"], ensure_ascii=False), flush=True)
    prims = PRIMS_B0 if a.setup == "b0" else PRIMS_M08
    if a.prims:
        prims = [p for p in prims if p in a.prims.split(",")]
    bgs = BGS[(a.setup, a.model)] if not a.bgs else a.bgs.split(",")
    lo, hi = torch.cuda.Stream.priority_range()
    rid = run_id()
    gw = GD.Timing(a.out)                                          # 開不起來 → 丟錯（量時間的 run 不能在看不到的情況下跑）
    new = not os.path.exists(a.out)
    fo = open(a.out, "a", newline="")
    w = csv.DictWriter(fo, fieldnames=FIELDS)
    if new:
        w.writeheader()
    # MPS：背景在另一個 process、自己載一份權重 → 先把每個動作跑一次，拿到主程式的峰值，之後每個區塊開背景前檢查放不放得下
    # （10/26 22275：llama decode:64:8192 的背景 process 80.12 GiB＋主程式 59.27 GiB > 139.79 GiB → OOM）
    mps_fit = None
    if any(m.startswith("mps") for m in a.modes.split(",")):
        with torch.cuda.stream(torch.cuda.Stream(device=tgt.km.device)):
            for prim in prims:
                tgt.run(prim)
        torch.cuda.synchronize()
        mps_fit = dict(main_peak=torch.cuda.max_memory_allocated(),
                       weights=sum(p.numel() * p.element_size() for p in tgt.km.hf.parameters()))
        print(f"[b0] MPS：主程式峰值 {mps_fit['main_peak'] / 2**30:.1f} GiB、背景要另載的權重 {mps_fit['weights'] / 2**30:.1f} GiB", flush=True)
    try:
        modes = a.modes.split(",")
        for bi, bgspec in enumerate(plan_order(bgs, a.seed)):            # 背景的順序也隨機
            for mode in modes:
                seed = a.seed * 1000 + bi * 10 + modes.index(mode)
                if mode.startswith("mps"):
                    from bgengine import step_bytes
                    need = mps_fit["weights"] + step_bytes(tgt.km, bgspec) + MPS_MARGIN
                    grow = max(0, mps_fit["main_peak"] - torch.cuda.memory_allocated())
                    free = torch.cuda.mem_get_info()[0]
                    if need + grow > free:
                        note = (f"MPS 背景放不下：背景 process 約 {need / 2**30:.1f} GiB（權重 {mps_fit['weights'] / 2**30:.1f}＋背景 "
                                f"{step_bytes(tgt.km, bgspec) / 2**30:.1f}＋餘量 {MPS_MARGIN / 2**30:.0f}）＋主程式還會長 {grow / 2**30:.1f} GiB "
                                f"> 可用 {free / 2**30:.1f} GiB")
                        print(f"[b0] {bgspec} × {mode}：NOT_MEASURED（{note}）", flush=True)
                        for prim in prims:
                            w.writerow({f: v for f, v in dict(run_id=rid, ts=ts(), model=a.model, setup=a.setup, ctx_tokens=tgt.nb * CHUNK,
                                                               bg=bgspec, mode=mode, prim=prim, t_med_s=NM, n_rep=0, sigma_rel=NM,
                                                               delta_star=NM, seed=seed, note=note, host_level=host_level(),
                                                               own_gpu_intruders=gw.status()).items()})
                        fo.flush()
                        continue
                    pct = int(mode[3:] or 100)
                    proc, rate_fn, solo, tps, bsr = start_bgproc(a, bgspec, pct)
                    bds = round(max(0.05, 2 * bsr), 6) if isinstance(bsr, float) else NM
                    stream = torch.cuda.Stream(device=tgt.km.device)
                else:
                    bg = Background(tgt.km, bgspec, priority=(lo if mode == "stream_bglow" else 0))
                    solo_list = bg.solo_rates(MIN_NOISE_REPS, 1.0)
                    solo = statistics.median(solo_list)
                    bsr, bds = noise(solo_list) if bgspec != "idle" else (NM, NM)
                    bg.start()
                    time.sleep(0.5)
                    tps = getattr(bg, "tokens_per_step", 0)

                    def rate_fn(t0, t1, bg=bg):
                        n = sum(1 for t in bg._t if t0 <= t <= t1)
                        return n, n / (t1 - t0)
                    stream = torch.cuda.Stream(device=tgt.km.device, priority=(hi if mode == "stream_hi" else 0))
                tgt.mode = mode
                base = dict(run_id=rid, model=a.model, setup=a.setup, ctx_tokens=tgt.nb * CHUNK, bg=bgspec, bg_tokens_per_step=tps,
                            mode=mode, host_level=host_level(), own_gpu_intruders=gw.status(), bg_solo_sigma_rel=bsr,
                            bg_delta_star=bds)
                try:
                    measure_block(tgt, None, prims, a.reps, seed, rate_fn, solo, stream, w, base)
                finally:
                    if mode.startswith("mps"):
                        torch.cuda.synchronize()                    # 自己的工作先做完，再請背景優雅地停（cmd_bgproc 的 SIGTERM 處理）
                        proc.terminate()
                        proc.wait(120)
                    else:
                        bg.stop()
                        # rate_fn 的預設參數握著 bg：不清掉的話，上一個背景的 KV／CUDA graph 要等下一個背景建好才釋放
                        # （10/26 22271：decode:64:8192 要 64 GiB 時，這個 process 已經用了 123.76 GiB → OOM）
                        rate_fn = None
                        del bg
                        gc.collect()
                        torch.cuda.empty_cache()
                        print(f"[b0] 背景釋放後 GPU 已配置 {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
                fo.flush()
                print(f"[b0] {bgspec} × {mode} 做完（solo {solo:.2f} 步/s）", flush=True)
    finally:
        fo.close()
        gw.finish()
        try:
            tgt.probe_numa("end")
        except Exception as ex:          # noqa: BLE001  結束時查不到：照實印，start／bench 那幾次照存
            print(f"[b0] NUMA 結束時的查詢失敗：{ex!r}", flush=True)
        side = NP.save(a.out, rid, **tgt.numa_log)
        print("[b0] NUMA 存在", side, flush=True)
        GD.evict(gw, [a.out, side if isinstance(side, str) else ""])     # 污染 → 這個 run 的 CSV 和 sidecar 搬出 results


def start_bgproc(a, spec: str, pct: int):
    """MPS：背景在另一個 process（CUDA_MPS_ACTIVE_THREAD_PERCENTAGE＝pct；100＝不限）。先量單獨的速度，再開始共跑。"""
    env = dict(os.environ)
    env.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)
    if pct < 100:
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(pct)
    p = subprocess.Popen([sys.executable, __file__, "bgproc", "--model", a.model, "--spec", spec], stdout=subprocess.PIPE,
                         text=True, env=env, bufsize=1)
    marks = []
    import threading

    def reader():
        for line in p.stdout:
            if line.startswith("T "):
                _, t, n = line.split()
                marks.append((float(t), int(n)))
            elif line.startswith("READY") or line.startswith("SOLO"):
                marks.append(line.strip())
    th = threading.Thread(target=reader, daemon=True)
    th.start()
    t0 = time.time()
    while not any(isinstance(m, str) and m.startswith("SOLO") for m in marks):
        if p.poll() is not None or time.time() - t0 > 600:
            raise SystemExit(f"bgproc {spec} 沒有啟動（exit {p.poll()}）")
        time.sleep(0.2)
    solo_line = next(m for m in marks if isinstance(m, str) and m.startswith("SOLO"))
    solo = float(solo_line.split()[1])
    tps = int(solo_line.split()[2])
    bsr = solo_line.split()[3] if len(solo_line.split()) > 3 else NM
    bsr = float(bsr) if bsr not in (NM, "") else NM

    def rate_fn(t0, t1):
        pts = [m for m in marks if isinstance(m, tuple)]
        inside = [n for t, n in pts if t0 <= t <= t1]
        if len(inside) < 2:
            return 0, 0.0
        tt = [t for t, n in pts if t0 <= t <= t1]
        return inside[-1] - inside[0], (inside[-1] - inside[0]) / (tt[-1] - tt[0])
    time.sleep(0.5)
    return p, rate_fn, solo, tps, bsr


def cmd_bgproc(a):
    """MPS 的背景 process：載模型（只要權重）→ 背景 → 先單獨跑 2 s 量速度（印 SOLO）→ 一直跑，每 0.25 s 印「T perf_counter 步數」。"""
    import torch
    os.environ["M7_MODEL_GLOB"] = os.environ.get(f"M7_{a.model.upper()}_GLOB", "")
    from m7_model import KVModel
    from bgengine import Background
    km = KVModel(max_len=1024)
    bg = Background(km, a.spec)
    sl = bg.solo_rates(MIN_NOISE_REPS, 1.0)
    sr, _ = noise(sl)
    print(f"SOLO {statistics.median(sl)} {getattr(bg, 'tokens_per_step', 0)} {sr}", flush=True)
    # SIGTERM → 只設旗標：跳出迴圈、等在途的步驟做完（bg.stop 會 join、等 event）、synchronize，再自己結束。
    # 10/26 22275／22276：主程式在 SIGTERM 直接殺掉背景 process 之後 0.2–0.3 s 出現 fatal GPU error（MPS）→ 不在 graph 還在跑時被殺。
    import signal
    stop = {"f": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("f", True))
    bg.start()
    t_end = time.time() + a.seconds
    while time.time() < t_end and not stop["f"]:
        time.sleep(0.25)
        print(f"T {time.perf_counter()} {bg.steps}", flush=True)
    bg.stop()
    torch.cuda.synchronize()
    print("STOPPED", flush=True)


def cmd_bgcheck(a):
    """P0b 自檢：每種背景單獨跑的每秒步數與 token 數、KV 佔用；graph 和 eager 的速度。"""
    import torch
    from bgengine import Background
    os.environ["M7_MODEL_GLOB"] = os.environ.get(f"M7_{a.model.upper()}_GLOB", "")
    from m7_model import KVModel
    km = KVModel(max_len=1024)
    rid, rows = run_id(), []
    specs = a.specs.split(",") if a.specs else sorted({s for (st, m), v in BGS.items() if m == a.model for s in v if s != "idle"})
    for spec in specs:
        for graph in (True, False):
            torch.cuda.reset_peak_memory_stats()
            bg = Background(km, spec, use_graph=graph)
            r = bg.solo_rate(3.0)
            rows.append(dict(run_id=rid, ts=ts(), model=a.model, spec=spec, graph=int(graph), steps_per_s=round(r, 3),
                             tokens_per_s=round(r * bg.tokens_per_step, 1), step_ms=round(1000 / r, 3) if r else "",
                             kv_gib=round(getattr(bg.step, "kv_bytes", 0) / 2**30, 2),
                             peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 1)))
            print(json.dumps(rows[-1]), flush=True)
            del bg
            torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def cmd_env(a):
    """P0e：stream 優先權、green context、MPS。查不到就照實寫 NOT_AVAILABLE（附錯誤），不當成可以用。"""
    import shutil
    import torch
    out = dict(run_id=run_id(), ts=ts(), torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(0),
               sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
               stream_priority_range=list(torch.cuda.Stream.priority_range()))
    try:
        from torch.cuda import green_contexts as G
        out["green_context_supported_flag"] = bool(getattr(G, "SUPPORTED", False))
        try:
            gc = G.GreenContext.create(num_sms=32, device_id=0)
            gc.set_context()
            x = torch.randn(1024, 1024, device="cuda")
            y = x @ x
            torch.cuda.synchronize()
            gc.pop_context()
            out["green_context"] = "AVAILABLE (create num_sms=32 + matmul ran)"
        except Exception as ex:          # noqa: BLE001
            out["green_context"] = f"NOT_AVAILABLE: {ex!r}"
    except Exception as ex:              # noqa: BLE001
        out["green_context"] = f"NOT_AVAILABLE: {ex!r}"
    out["mps_control_binary"] = shutil.which("nvidia-cuda-mps-control") or "NOT_FOUND"
    out["mps_pipe_dir"] = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "")
    pin = torch.empty(64 << 20, dtype=torch.uint8, pin_memory=True).fill_(1)          # 這個 process 配的 pinned 落在哪一邊
    out["numa"] = NP.probe(torch.cuda.current_device(), {"pinned_64MiB": pin})
    print(json.dumps(out, ensure_ascii=False, indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)


# ============================================================================ 判定（CPU）
def _rows(paths):
    out = []
    for p in paths:
        with open(p, newline="") as f:
            out += [r for r in csv.DictReader(f) if r.get("run_id") not in (None, "run_id")]
    return out


def cell_stats(rows) -> dict:
    """(model, setup, bg, mode, prim) → 每個 rep 的時間（t_med_s）、中位數、σ、δ*、背景吞吐比的每個 rep、背景的 δ*。"""
    g = {}
    for r in rows:
        if r.get("t_med_s") == NM:                                # 量不到的格子（例：MPS 放不下）→ judge 另外列進 not_measured
            continue
        k = (r["model"], r["setup"], r["bg"], r["mode"], r["prim"])
        g.setdefault(k, []).append(r)
    out = {}
    for k, rs in g.items():
        reps = [float(r["t_med_s"]) for r in rs]
        br = [float(r["bg_ratio"]) for r in rs if r.get("bg_ratio") not in ("", None)]
        sr, ds = noise(reps)
        bds = rs[0].get("bg_delta_star", NM)
        bds = float(bds) if bds not in (NM, "", None) else NM
        out[k] = dict(t=statistics.median(reps), reps=reps, n=len(reps), sr=sr, ds=ds,
                      noisy=isinstance(sr, float) and 2 * sr > 0.05,
                      bg_ratio=statistics.median(br) if br else None, bg_ratios=br, bg_ds=bds,
                      bg_noisy=isinstance(bds, float) and bds > 0.05)
    return out


def judge(rows) -> dict:
    """登記 §2 的判準＋偏離 1（05 §2.1 的雜訊規則）。2σ > 5% 或 σ 量不到的格子不判定、列出來。"""
    st = cell_stats(rows)
    res = dict(flip=[], share=[], static={}, m08=[], skipped=[],
               noisy=[list(k) for k, v in st.items() if v["noisy"] or v["bg_noisy"]],
               not_measured=[list(k) for k, v in st.items() if v["ds"] == NM]
               + sorted({(r["model"], r["setup"], r["bg"], r["mode"], r["prim"]) for r in rows if r.get("t_med_s") == NM}))

    def usable(*cells):
        return all(c is not None and c["ds"] != NM and not c["noisy"] for c in cells)
    # ---- FLIP：baseline＝那個背景下「閒時最快的計畫」
    lines = {}
    for (m, s, bg, mode, prim), v in st.items():
        if s == "b0" and prim.startswith("e2e_") and mode == "stream":
            lines.setdefault((m, prim.split("_")[-1]), {}).setdefault(bg, {})[prim] = v
    for (m, tier), byb in lines.items():
        idle = byb.get("idle")
        if not idle:
            continue
        best_idle = min(idle, key=lambda p: idle[p]["t"])
        for bg, plans in byb.items():
            if bg == "idle" or not bg.startswith(REALISTIC):
                continue
            best = min(plans, key=lambda p: plans[p]["t"])
            if best == best_idle or best_idle not in plans:
                continue
            base, cand = plans[best_idle], plans[best]
            if not usable(base, cand):
                res["skipped"].append(dict(test="FLIP", model=m, tier=tier, bg=bg, why="noisy_or_not_measured"))
                continue
            gain = base["t"] / cand["t"] - 1
            thr = max(0.10, base["ds"])
            lo, hi = ratio_ci(base["reps"], cand["reps"])
            if gain >= thr and lo > 1:
                res["flip"].append(dict(model=m, tier=tier, bg=bg, idle_best=best_idle, best=best, gain=gain, threshold=thr,
                                        ratio_ci=[lo, hi]))
    res["GB0_FLIP"] = any(len({(x["model"], x["tier"]) for x in res["flip"] if x["bg"] == bg}) >= 2 for bg in {x["bg"] for x in res["flip"]})
    # ---- SHARE：baseline＝同一格的 stream
    for (m, s, bg, mode, prim), v in st.items():
        if s != "b0" or mode == "stream" or not bg.startswith(REALISTIC):
            continue
        ref = st.get((m, s, bg, "stream", prim))
        if ref is None or not v["bg_ratios"]:
            continue
        if not usable(ref, v) or v["bg_noisy"]:
            res["skipped"].append(dict(test="SHARE", model=m, bg=bg, prim=prim, mode=mode, why="noisy_or_not_measured"))
            continue
        sp = ref["t"] / v["t"] - 1
        thr = max(0.30, ref["ds"])
        lo, hi = ratio_ci(ref["reps"], v["reps"])
        blo, _ = median_ci(v["bg_ratios"]) if len(v["bg_ratios"]) >= 2 else (None, None)
        bg_ok = v["bg_ratio"] >= 0.90 and blo is not None and blo >= 0.90
        if sp >= thr and lo > 1 and bg_ok:
            res["share"].append(dict(model=m, bg=bg, prim=prim, mode=mode, speedup=sp, threshold=thr, ratio_ci=[lo, hi],
                                     bg_ratio=v["bg_ratio"], bg_ratio_ci_lo=blo))
        elif sp >= thr and lo > 1 and v["bg_ratio"] >= 0.90:             # 點估計過、CI 沒過 → INCONCLUSIVE（要加重複次數）
            res.setdefault("share_inconclusive", []).append(dict(model=m, bg=bg, prim=prim, mode=mode, speedup=sp,
                                                                 bg_ratio=v["bg_ratio"], bg_ratio_ci_lo=blo))
    res["GB0_SHARE"] = True if res["share"] else ("INCONCLUSIVE" if res.get("share_inconclusive") else False)
    # ---- STATIC：並列最好（和最快的差 ≤ δ*）的等級；存在一個等級，每個背景都有並列最好的在它相鄰一級內
    lv = {"mps": 0, "mps87": 1, "mps75": 2, "mps50": 3}
    ties = {}
    for (m, s, bg, mode, prim), v in st.items():
        if s != "b0" or mode not in lv or not bg.startswith(REALISTIC) or v["bg_ratio"] is None or v["bg_ratio"] < 0.90 or not usable(v):
            continue
        ties.setdefault((m, prim), {}).setdefault(bg, {})[mode] = v
    for (m, prim), byb in ties.items():
        sets = {}
        for bg, cells in byb.items():
            bm = min(cells, key=lambda k: cells[k]["t"])
            thr = cells[bm]["t"] * (1 + cells[bm]["ds"])
            sets[bg] = sorted(lv[k] for k, c in cells.items() if c["t"] <= thr)
        static = any(all(any(abs(l - L) <= 1 for l in ls) for ls in sets.values()) for L in range(4)) if sets else None
        res["static"][f"{m}|{prim}"] = dict(tie_levels=sets, static=static)
    vals = [v["static"] for v in res["static"].values() if v["static"] is not None]
    res["GB0_STATIC"] = bool(vals) and all(vals)
    # ---- G-M08：速度 ≥ 閒時 60%（門檻不變）＋背景「沒有變慢」：變慢 ≤ δ*（背景單獨跑的 σ）且 CI 上界 ≤ δ*
    for (m, s, bg, mode, prim), v in st.items():
        if s != "m08" or not bg.startswith("decode:") or prim not in COMPUTE_PRIMS or not v["bg_ratios"]:
            continue
        idle = st.get((m, s, "idle", "stream", prim))
        if idle is None:
            continue
        if not usable(idle, v) or v["bg_ds"] == NM or v["bg_noisy"]:
            res["skipped"].append(dict(test="G-M08", model=m, bg=bg, prim=prim, mode=mode, why="noisy_or_not_measured"))
            continue
        speed = idle["t"] / v["t"]
        slo, shi = ratio_ci(idle["reps"], v["reps"])
        blo, bhi = median_ci(v["bg_ratios"]) if len(v["bg_ratios"]) >= 2 else (None, None)
        slow = 1 - v["bg_ratio"]
        point_ok = speed >= 0.60 and slow <= v["bg_ds"]
        ci_ok = blo is not None and (1 - blo) <= v["bg_ds"]
        status = "ok" if (point_ok and ci_ok) else ("inconclusive" if point_ok else "fail")
        res["m08"].append(dict(model=m, bg=bg, mode=mode, prim=prim, speed_vs_idle=speed, speed_ci=[slo, shi], bg_slowdown=slow,
                               bg_slowdown_ci_hi=(1 - blo) if blo is not None else None, bg_delta_star=v["bg_ds"],
                               ok=status == "ok", status=status))
    st_m08 = {x["status"] for x in res["m08"]}
    res["G_M08"] = ("PASS" if "ok" in st_m08 else "INCONCLUSIVE" if "inconclusive" in st_m08 else "STOP" if res["m08"] else "NOT_MEASURED")
    res["provisional_because_noisy"] = bool(res["noisy"])
    res["G_M11"] = judge_gm11(st)
    return res


def judge_gm11(st: dict) -> dict:
    """runs/G_M11_vq.md：慢幾倍＝t(背景, stream)／t(閒, stream)。VQ 在 matmul 背景下比 INT4（融合 kernel）少慢：
    INT4 的慢幾倍 ÷ VQ 的慢幾倍 −1 > δ*（δ*＝INT4 在 matmul 背景那一格的），且這個比的 95% CI（四格各自對 rep 重抽）下界 > 1。"""
    import numpy as np
    out = {}
    for m in sorted({k[0] for k in st}):
        cells = {(bg, prim): st.get((m, "b0", bg, "stream", prim)) for bg in ("idle", "matmul", "decode:32:8192")
                 for prim in ("dq_vq", "dq_int4")}
        if any(cells[(bg, p)] is None for bg in ("idle", "matmul") for p in ("dq_vq", "dq_int4")):
            out[m] = dict(verdict="NOT_MEASURED")
            continue
        use = [cells[(bg, p)] for bg in ("idle", "matmul") for p in ("dq_vq", "dq_int4")]
        if any(c["ds"] == NM or c["noisy"] for c in use):
            out[m] = dict(verdict="NOISY_OR_NOT_MEASURED")
            continue
        sv = cells[("matmul", "dq_vq")]["t"] / cells[("idle", "dq_vq")]["t"]
        si = cells[("matmul", "dq_int4")]["t"] / cells[("idle", "dq_int4")]["t"]
        rng = np.random.default_rng(0)

        def bm(c):
            A = np.asarray(c["reps"])
            return np.median(A[rng.integers(0, A.size, size=(N_BOOT, A.size))], axis=1)
        r = (bm(cells[("matmul", "dq_int4")]) / bm(cells[("idle", "dq_int4")])) / (bm(cells[("matmul", "dq_vq")]) / bm(cells[("idle", "dq_vq")]))
        lo, hi = float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))
        thr = cells[("matmul", "dq_int4")]["ds"]
        ok = si / sv - 1 > thr and lo > 1
        dec = {}
        if cells[("decode:32:8192", "dq_vq")] and cells[("decode:32:8192", "dq_int4")]:
            dec = dict(vq=cells[("decode:32:8192", "dq_vq")]["t"] / cells[("idle", "dq_vq")]["t"],
                       int4=cells[("decode:32:8192", "dq_int4")]["t"] / cells[("idle", "dq_int4")]["t"])
        out[m] = dict(slowdown_vq_matmul=sv, slowdown_int4_matmul=si, ratio=si / sv, ratio_ci=[lo, hi], delta_star=thr,
                      verdict="PASS" if ok else "STOP", decode_background=dec)
    v = [x["verdict"] for x in out.values()]
    out["verdict"] = ("PASS" if v and all(x == "PASS" for x in v) else "PARTIAL" if "PASS" in v
                      else "STOP" if v and all(x == "STOP" for x in v) else "NOT_DETERMINED")
    return out


def cmd_report(a):
    import glob
    paths = sorted(glob.glob(os.path.join(OUT, "raw", "*.csv")))
    rows, bad = GD.drop_contaminated(_rows(paths))
    res = judge(rows)
    doc = dict(run_id=run_id(), ts=ts(), inputs=[os.path.relpath(p, REPO) for p in paths], n_rows=len(rows),
               excluded_contaminated_runs=bad, **res)
    os.makedirs(OUT, exist_ok=True)
    json.dump(doc, open(os.path.join(OUT, "b0_report.json"), "w"), indent=1, ensure_ascii=False, default=str)
    print(json.dumps({k: doc[k] for k in ("GB0_FLIP", "GB0_SHARE", "GB0_STATIC", "G_M08", "n_rows", "provisional_because_noisy")}
                     | {"G_M11": doc["G_M11"].get("verdict")},
                     ensure_ascii=False), f"noisy={len(doc['noisy'])} not_measured={len(doc['not_measured'])} skipped={len(doc['skipped'])}")
    if doc["noisy"]:
        print("⚠ 有 2σ > 5% 的格子（05 §2.1 第 5 點：先回報、不拿來判定）：", doc["noisy"][:10])


def self_test() -> bool:
    ok = True

    def chk(name, c):
        nonlocal ok
        print(("PASS " if c else "FAIL ") + name)
        ok &= bool(c)
    chk("隨機順序可重現、是排列", plan_order(PRIMS_B0, 3) == plan_order(PRIMS_B0, 3) and sorted(plan_order(PRIMS_B0, 3)) == sorted(PRIMS_B0))
    chk("noise：6 次才有 σ、δ* 下限 5%", noise([1.0] * 5) == (NM, NM) and noise([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])[1] == 0.05
        and noise([1.0, 1.2, 0.8, 1.1, 0.9, 1.0])[1] > 0.05)
    rows = []

    def add(m, s, bg, mode, prim, t, br, jitter=0.002, bds=0.05):
        for rep in range(6):
            rows.append(dict(model=m, setup=s, bg=bg, mode=mode, prim=prim, t_med_s=str(t * (1 + jitter * (rep - 2.5))),
                             bg_ratio=(str(br + 0.001 * (rep - 2.5)) if br != "" else ""), bg_delta_star=str(bds)))
    for m in ("llama", "qwen"):
        for tier in ("cpu", "b32"):
            add(m, "b0", "idle", "stream", f"e2e_cake_bf16_{tier}", 1.0, "")
            add(m, "b0", "idle", "stream", f"e2e_load_int4_{tier}", 1.2, "")
            add(m, "b0", "cprefill:2048:65536", "stream", f"e2e_cake_bf16_{tier}", 3.0, 0.8)
            add(m, "b0", "cprefill:2048:65536", "stream", f"e2e_load_int4_{tier}", 1.5, 0.9)
        add(m, "b0", "decode:32:8192", "stream", "recompute1", 1.0, 0.7)
        add(m, "b0", "decode:32:8192", "mps87", "recompute1", 0.6, 0.95)
        add(m, "b0", "decode:32:8192", "mps", "recompute1", 0.9, 0.85)
        add(m, "m08", "idle", "stream", "recompute1", 1.0, "")
        add(m, "m08", "decode:4:129024", "stream", "recompute1", 1.4, 0.98)
    r = judge(rows)
    chk(f"FLIP（兩個模型 × 兩層都翻）→ {r['GB0_FLIP']}", r["GB0_FLIP"] is True)
    chk(f"SHARE（mps87 快 67%、背景 0.95）→ {r['GB0_SHARE']}", r["GB0_SHARE"] is True)
    chk(f"G-M08（速度 0.71、背景慢 2%、δ* 5%）→ {r['G_M08']}", r["G_M08"] == "PASS")
    rows2 = [x for x in rows if x["bg"] != "cprefill:2048:65536"]
    chk("沒有翻轉的資料 → FLIP False", judge(rows2)["GB0_FLIP"] is False)
    rows3 = [dict(x, t_med_s=str(float(x["t_med_s"]) * (1 + 0.2 * ((hash(x["prim"] + x["mode"] + str(i)) % 7) - 3) / 3)))
             for i, x in enumerate(rows)]
    r3 = judge(rows3)
    chk(f"很晃（2σ > 5%）的格子 → 列出 {len(r3['noisy'])} 格、不判定", len(r3["noisy"]) > 0 and r3["provisional_because_noisy"])
    rows4 = [x for x in rows if x["bg"] != "decode:4:129024"] + [
        dict(model="llama", setup="m08", bg="decode:4:129024", mode="stream", prim="recompute1", t_med_s=str(1.4), bg_ratio="0.90",
             bg_delta_star="0.05") for _ in range(6)]
    chk(f"G-M08 背景慢 10% > δ* 5% → {judge(rows4)['G_M08']}", judge(rows4)["G_M08"] in ("STOP", "PASS") and
        not any(x["ok"] for x in judge(rows4)["m08"] if x["model"] == "llama"))
    rows5 = [x for x in rows if x["bg"] != "decode:4:129024"]
    for rep, br in enumerate((0.99, 0.90, 0.99, 0.90, 0.99, 0.99)):          # 中位數 0.99（慢 1%），但很分散 → CI 上界 > 5%
        for m in ("llama", "qwen"):
            rows5.append(dict(model=m, setup="m08", bg="decode:4:129024", mode="stream", prim="recompute1", t_med_s=str(1.4),
                              bg_ratio=str(br), bg_delta_star="0.05"))
    chk(f"G-M08 點估計過、CI 上界超過 δ* → {judge(rows5)['G_M08']}", judge(rows5)["G_M08"] == "INCONCLUSIVE")
    nm_rows = rows + [dict(model="llama", setup="b0", bg="decode:64:8192", mode="mps", prim=pr, t_med_s=NM, bg_ratio="", bg_delta_star=NM)
                      for pr in ("recompute1", "h2d_batch")]
    try:
        rn = judge(nm_rows)
        chk("NOT_MEASURED 的列（MPS 放不下）：report 不出錯、列進 not_measured",
            ["llama", "b0", "decode:64:8192", "mps", "recompute1"] in [list(x) for x in rn["not_measured"]])
    except Exception as ex:                                       # noqa: BLE001
        chk(f"NOT_MEASURED 的列：report 出錯 {ex!r}", False)
    print("self-test", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("env")
    s.add_argument("--out", required=True)
    s = sub.add_parser("bgcheck")
    s.add_argument("--model", choices=list(INFO), required=True)
    s.add_argument("--specs", default="")
    s.add_argument("--out", required=True)
    s = sub.add_parser("run")
    s.add_argument("--model", choices=list(INFO), required=True)
    s.add_argument("--setup", choices=["b0", "m08"], required=True)
    s.add_argument("--modes", default="stream,stream_hi")
    s.add_argument("--bgs", default="")
    s.add_argument("--prims", default="")
    s.add_argument("--reps", type=int, default=6)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--calib-xl", default=os.path.join(os.environ.get("S3_LOCAL", "/tmp/hungweilai-s3"), "xstack", "calib_xl_{m}.pt"))
    s.add_argument("--out", required=True)
    s = sub.add_parser("bgproc")
    s.add_argument("--model", choices=list(INFO), required=True)
    s.add_argument("--spec", required=True)
    s.add_argument("--seconds", type=float, default=3600)
    sub.add_parser("report")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if a.cmd == "run":
        a.calib_xl = a.calib_xl.format(m=a.model)
    {"env": cmd_env, "bgcheck": cmd_bgcheck, "run": cmd_run, "bgproc": cmd_bgproc, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
