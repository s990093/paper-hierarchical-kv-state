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
            "dq_fp8", "dq_nvfp4", "lr_recon", "recompute1", "e2e_cake_bf16_cpu", "e2e_load_bf16_cpu", "e2e_cake_int4_cpu",
            "e2e_load_int4_cpu", "e2e_cake_bf16_b32", "e2e_load_bf16_b32", "e2e_cake_int4_b32", "e2e_load_int4_b32"]
PRIMS_M08 = ["recompute1", "qprefill", "dq_int4", "lr_recon", "h2d_batch"]
PRODUCTION = ("h2d_contig", "h2d_perblock", "h2d_batch", "sm_gather_12", "sm_gather_32", "sm_gather_132", "dq_int4", "dq_int4_torch",
              "dq_fp8", "dq_nvfp4", "recompute1", "qprefill")        # 05 §5.2 / 08 §3：只用生產格式（沒有低秩重建）
COMPUTE_PRIMS = ("recompute1", "qprefill", "lr_recon")            # G-M08 的「計算型」還原
UNIT_BLOCKS = {"b0": 64, "m08": 16}                              # 每個動作處理幾塊（m08 用 16 塊當一個單位）
MPS_PCTS = (100, 87, 75, 50)
WINDOW_S = 1.0                                                    # 一次「量」的最短時間窗（背景吞吐要夠多步才準）
FIELDS = ["run_id", "ts", "model", "setup", "ctx_tokens", "bg", "bg_tokens_per_step", "mode", "prim", "production", "rep",
          "order_idx", "n_exec", "t_med_s", "t_p95_s", "t_min_s", "t_window_s", "bg_steps", "bg_rate", "bg_solo_rate", "bg_ratio",
          "unit_blocks", "bytes", "seed", "gpu_mem_gib", "host_level", "own_gpu_intruders"]


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
        self.blocks = list(range(1, 1 + u))                       # 動作作用在 block 1..u（block 0 不動）
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
            w.writerow(dict(base_row, prim=prim, production=int(prim in PRODUCTION), rep=rep, order_idx=oi, n_exec=n_exec,
                            t_med_s=statistics.median(ts_), t_p95_s=ts_[min(len(ts_) - 1, math.ceil(0.95 * len(ts_)) - 1)],
                            t_min_s=ts_[0], t_window_s=w1 - w0, bg_steps=steps, bg_rate=rate, bg_solo_rate=solo,
                            bg_ratio=(rate / solo if solo else ""), unit_blocks=tgt.unit, bytes=tgt.bytes_of(prim), seed=seed,
                            gpu_mem_gib=round(torch.cuda.max_memory_allocated() / 2**30, 1), ts=ts()))


def cmd_run(a):
    import torch
    from bgengine import Background
    from m16_c1a_predict import finish_watch, host_level, intruders, start_watch
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tgt = Target(a.model, a.setup, a.calib_xl)
    print("[b0] 正確性：", json.dumps(tgt.check()), flush=True)
    prims = PRIMS_B0 if a.setup == "b0" else PRIMS_M08
    if a.prims:
        prims = [p for p in prims if p in a.prims.split(",")]
    bgs = BGS[(a.setup, a.model)] if not a.bgs else a.bgs.split(",")
    lo, hi = torch.cuda.Stream.priority_range()
    rid = run_id()
    watch = start_watch(a.out)
    new = not os.path.exists(a.out)
    fo = open(a.out, "a", newline="")
    w = csv.DictWriter(fo, fieldnames=FIELDS)
    if new:
        w.writeheader()
    try:
        modes = a.modes.split(",")
        for bi, bgspec in enumerate(plan_order(bgs, a.seed)):            # 背景的順序也隨機
            for mode in modes:
                seed = a.seed * 1000 + bi * 10 + modes.index(mode)
                if mode.startswith("mps"):
                    pct = int(mode[3:] or 100)
                    proc, rate_fn, solo, tps = start_bgproc(a, bgspec, pct)
                    stream = torch.cuda.Stream(device=tgt.km.device)
                else:
                    bg = Background(tgt.km, bgspec, priority=(lo if mode == "stream_bglow" else 0))
                    solo = bg.solo_rate(2.0)
                    bg.start()
                    time.sleep(0.5)
                    tps = getattr(bg, "tokens_per_step", 0)

                    def rate_fn(t0, t1, bg=bg):
                        n = sum(1 for t in bg._t if t0 <= t <= t1)
                        return n, n / (t1 - t0)
                    stream = torch.cuda.Stream(device=tgt.km.device, priority=(hi if mode == "stream_hi" else 0))
                tgt.mode = mode
                base = dict(run_id=rid, model=a.model, setup=a.setup, ctx_tokens=tgt.nb * CHUNK, bg=bgspec, bg_tokens_per_step=tps,
                            mode=mode, host_level=host_level(), own_gpu_intruders=intruders(watch))
                try:
                    measure_block(tgt, None, prims, a.reps, seed, rate_fn, solo, stream, w, base)
                finally:
                    if mode.startswith("mps"):
                        proc.terminate()
                        proc.wait(60)
                    else:
                        bg.stop()
                        del bg
                        torch.cuda.empty_cache()
                fo.flush()
                print(f"[b0] {bgspec} × {mode} 做完（solo {solo:.2f} 步/s）", flush=True)
    finally:
        fo.close()
        finish_watch(watch, a.out)


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

    def rate_fn(t0, t1):
        pts = [m for m in marks if isinstance(m, tuple)]
        inside = [n for t, n in pts if t0 <= t <= t1]
        if len(inside) < 2:
            return 0, 0.0
        tt = [t for t, n in pts if t0 <= t <= t1]
        return inside[-1] - inside[0], (inside[-1] - inside[0]) / (tt[-1] - tt[0])
    time.sleep(0.5)
    return p, rate_fn, solo, tps


def cmd_bgproc(a):
    """MPS 的背景 process：載模型（只要權重）→ 背景 → 先單獨跑 2 s 量速度（印 SOLO）→ 一直跑，每 0.25 s 印「T perf_counter 步數」。"""
    import torch
    os.environ["M7_MODEL_GLOB"] = os.environ.get(f"M7_{a.model.upper()}_GLOB", "")
    from m7_model import KVModel
    from bgengine import Background
    km = KVModel(max_len=1024)
    bg = Background(km, a.spec)
    solo = bg.solo_rate(2.0)
    print(f"SOLO {solo} {getattr(bg, 'tokens_per_step', 0)}", flush=True)
    bg.start()
    t_end = time.time() + a.seconds
    while time.time() < t_end:
        time.sleep(0.25)
        print(f"T {time.perf_counter()} {bg.steps}", flush=True)
    bg.stop()


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
    """(model, setup, bg, mode, prim) → 中位數（各次的 t_med_s 的中位數）、P95、背景吞吐比的中位數、n。"""
    g = {}
    for r in rows:
        k = (r["model"], r["setup"], r["bg"], r["mode"], r["prim"])
        g.setdefault(k, []).append(r)
    out = {}
    for k, rs in g.items():
        t = sorted(float(r["t_med_s"]) for r in rs)
        br = [float(r["bg_ratio"]) for r in rs if r["bg_ratio"] not in ("", None)]
        out[k] = dict(t=statistics.median(t), p95=t[min(len(t) - 1, math.ceil(0.95 * len(t)) - 1)], n=len(t),
                      bg_ratio=statistics.median(br) if br else None)
    return out


def judge(rows, delta: float = 0.05) -> dict:
    """登記 §2 的 GB0-FLIP／SHARE／STATIC 與 G-M08。"""
    st = cell_stats(rows)
    res = dict(flip=[], share=[], static={}, m08=[])
    lines = {}
    for (m, s, bg, mode, prim), v in st.items():
        if s == "b0" and prim.startswith("e2e_") and mode == "stream":
            tier = prim.split("_")[-1]
            lines.setdefault((m, tier), {}).setdefault(bg, {})[prim] = v["t"]
    for (m, tier), byb in lines.items():
        idle = byb.get("idle")
        if not idle:
            continue
        best_idle = min(idle, key=idle.get)
        for bg, plans in byb.items():
            if bg == "idle" or not bg.startswith(REALISTIC):
                continue
            best = min(plans, key=plans.get)
            if best != best_idle and plans.get(best_idle) and plans[best_idle] / plans[best] - 1 >= 0.10:
                res["flip"].append(dict(model=m, tier=tier, bg=bg, idle_best=best_idle, best=best,
                                        gain=plans[best_idle] / plans[best] - 1))
    flip_lines = {(x["model"], x["tier"]) for x in res["flip"]}
    res["GB0_FLIP"] = any(sum(1 for x in res["flip"] if x["bg"] == bg) >= 2 and len({(x["model"], x["tier"]) for x in res["flip"]
                                                                                      if x["bg"] == bg}) >= 2
                          for bg in {x["bg"] for x in res["flip"]})
    res["flip_lines"] = sorted(flip_lines)
    # SHARE：同一個（模型, 背景, 動作），某個共用方式比 stream（預設）快 ≥30%，而且背景吞吐 ≥ 單獨跑的 0.90
    best_mode = {}
    for (m, s, bg, mode, prim), v in st.items():
        if s != "b0" or mode == "stream" or not bg.startswith(REALISTIC):
            continue
        ref = st.get((m, s, bg, "stream", prim))
        if not ref or v["bg_ratio"] is None:
            continue
        sp = ref["t"] / v["t"] - 1
        if sp >= 0.30 and v["bg_ratio"] >= 0.90:
            res["share"].append(dict(model=m, bg=bg, prim=prim, mode=mode, speedup=sp, bg_ratio=v["bg_ratio"]))
    res["GB0_SHARE"] = bool(res["share"])
    # STATIC：每個（模型, 背景, 動作），在背景吞吐 ≥0.90 的 MPS 比例裡最快的那一級；所有像真實的背景都在相鄰一級內 → 固定預留
    lv = {"mps": 0, "mps87": 1, "mps75": 2, "mps50": 3}
    for (m, s, bg, mode, prim), v in st.items():
        if s != "b0" or mode not in lv or not bg.startswith(REALISTIC) or v["bg_ratio"] is None or v["bg_ratio"] < 0.90:
            continue
        k = (m, prim)
        cur = best_mode.setdefault(k, {}).get(bg)
        if cur is None or v["t"] < st[(m, s, bg, cur, prim)]["t"]:
            best_mode[k][bg] = mode
    for (m, prim), byb in best_mode.items():
        levels = [lv[x] for x in byb.values()]
        res["static"][f"{m}|{prim}"] = dict(best=byb, static=(max(levels) - min(levels) <= 1) if levels else None)
    vals = [v["static"] for v in res["static"].values() if v["static"] is not None]
    res["GB0_STATIC"] = bool(vals) and all(vals)
    # G-M08：長 context decode 背景下，計算型還原拿到單獨（閒）速度的 ≥60%，背景吞吐 ≥0.95（TPOT 慢 ≤5%）
    for (m, s, bg, mode, prim), v in st.items():
        if s != "m08" or not bg.startswith("decode:") or prim not in COMPUTE_PRIMS or v["bg_ratio"] is None:
            continue
        idle = st.get((m, s, "idle", "stream", prim))
        if not idle:
            continue
        speed = idle["t"] / v["t"]
        ok = speed >= 0.60 and v["bg_ratio"] >= 0.95
        res["m08"].append(dict(model=m, bg=bg, mode=mode, prim=prim, speed_vs_idle=speed, bg_ratio=v["bg_ratio"], ok=ok))
    res["G_M08"] = "PASS" if any(x["ok"] for x in res["m08"]) else ("STOP" if res["m08"] else "NOT_MEASURED")
    return res


def cmd_report(a):
    import glob
    paths = sorted(glob.glob(os.path.join(OUT, "raw", "*.csv")))
    rows = _rows(paths)
    res = judge(rows)
    doc = dict(run_id=run_id(), ts=ts(), inputs=[os.path.relpath(p, REPO) for p in paths], n_rows=len(rows), **res)
    os.makedirs(OUT, exist_ok=True)
    json.dump(doc, open(os.path.join(OUT, "b0_report.json"), "w"), indent=1, ensure_ascii=False, default=str)
    print(json.dumps({k: doc[k] for k in ("GB0_FLIP", "GB0_SHARE", "GB0_STATIC", "G_M08", "n_rows")}, ensure_ascii=False))


def self_test() -> bool:
    ok = True

    def chk(name, c):
        nonlocal ok
        print(("PASS " if c else "FAIL ") + name)
        ok &= bool(c)
    chk("隨機順序可重現、是排列", plan_order(PRIMS_B0, 3) == plan_order(PRIMS_B0, 3) and sorted(plan_order(PRIMS_B0, 3)) == sorted(PRIMS_B0))
    rows = []

    def add(m, s, bg, mode, prim, t, br):
        for rep in range(6):
            rows.append(dict(model=m, setup=s, bg=bg, mode=mode, prim=prim, t_med_s=str(t * (1 + 0.01 * rep)), bg_ratio=str(br)))
    for m in ("llama", "qwen"):
        for tier in ("cpu", "b32"):
            add(m, "b0", "idle", "stream", f"e2e_cake_bf16_{tier}", 1.0, "")
            add(m, "b0", "idle", "stream", f"e2e_load_int4_{tier}", 1.2, "")
            add(m, "b0", "cprefill:2048:65536", "stream", f"e2e_cake_bf16_{tier}", 3.0, 0.8)
            add(m, "b0", "cprefill:2048:65536", "stream", f"e2e_load_int4_{tier}", 1.5, 0.9)
        add(m, "b0", "decode:32:8192", "stream", "recompute1", 1.0, 0.7)
        add(m, "b0", "decode:32:8192", "mps87", "recompute1", 0.6, 0.92)
        add(m, "b0", "decode:32:8192", "mps", "recompute1", 0.9, 0.85)
        add(m, "m08", "idle", "stream", "recompute1", 1.0, "")
        add(m, "m08", "decode:4:129024", "stream", "recompute1", 1.4, 0.97)
    r = judge(rows)
    chk(f"FLIP（兩個模型 × 兩層都翻）→ {r['GB0_FLIP']}", r["GB0_FLIP"] is True)
    chk(f"SHARE（mps87 快 67%、背景 0.92）→ {r['GB0_SHARE']}", r["GB0_SHARE"] is True)
    chk(f"G-M08（速度 0.71、背景 0.97）→ {r['G_M08']}", r["G_M08"] == "PASS")
    rows2 = [x for x in rows if x["bg"] != "cprefill:2048:65536"]
    chk("沒有翻轉的資料 → FLIP False", judge(rows2)["GB0_FLIP"] is False)
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
