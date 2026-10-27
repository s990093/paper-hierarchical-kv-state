"""bgengine.py — 第三階段 B0／G-M08 的「像真的服務」背景負載（05 §3 P0b、mechanisms/06 §1）。

和 vLLM 的 FlashAttention 後端同一條路：分頁 KV（page 16 token）、`flash_attn_varlen_func`（seqused_k＋block_table、fa3）、
一次 forward 同時處理一批 decode token 與（可選的）一段 chunked prefill。權重直接用 KVModel 載入的 HF 模型（和還原的 session 共用，
像一個引擎同時服務別人與回來的人）。每一步錄成一個 CUDA graph，背景執行緒在自己的 stream 上一直 replay：
Python 的 GIL 只在送出 replay 時用到，量還原時不會因為背景的 Python 迴圈而失真。

一步（step）的內容（全部是固定形狀；位置固定在 S，所以每一步覆寫同一個 KV 位置：算的量、讀的位元組和真的 decode 每一步相同）：
  decode(B, S)        B 條序列各 1 個 token，每條看 S 個 key（B0：S＝8K；G-M08：S＝128K／256K）
  cprefill(C, S0)     一段 C 個 token 的 chunked prefill，接在 S0 個 key 之後（causal）
  mixed(P, B, S)      一段 P 個 token 的 prefill（接在 S 之後）＋ B 條 decode，同一個 varlen 呼叫（vLLM 的混合批次）
另外兩個舊的極端（不走模型）：matmul（m7_busy 的 4096³ BF16）、hbm（大塊 D2D 複製，m16_busy_mem）。

KV 的值用常態亂數填（attention 的計算量、讀的位元組和值無關）。
"""
from __future__ import annotations

import math
import threading
import time

import torch

PAGE = 16


def _fa():
    from vllm.vllm_flash_attn import flash_attn_varlen_func, is_fa_version_supported
    if not is_fa_version_supported(3):
        raise RuntimeError("fa3 不支援（背景引擎照 vLLM 在 H200 的預設用 fa3）")
    return flash_attn_varlen_func


class ModelStep:
    """一個「decode 序列＋可選 prefill 段」的固定形狀 forward（HF 權重、分頁 KV、fa3 varlen）。"""

    def __init__(self, km, n_dec: int, S: int, n_pre: int = 0, S0: int = 0, seed: int = 0):
        self.km, self.fa = km, _fa()
        dev, L, H, Hq, D = km.device, km.L, km.H, km.Hq, km.D
        self.n_dec, self.S, self.n_pre, self.S0 = n_dec, S, n_pre, S0
        seqs = [(1, S + 1)] * n_dec + ([(n_pre, S0 + n_pre)] if n_pre else [])        # (q 長度, k 長度)
        self.T = sum(q for q, _ in seqs)
        pages = [math.ceil(k / PAGE) for _, k in seqs]
        self.n_pages = sum(pages)
        g = torch.Generator(device=dev).manual_seed(seed)
        # KV：[L, 2, 頁, 16, H, D]
        self.kv = torch.randn((L, 2, self.n_pages, PAGE, H, D), generator=g, device=dev, dtype=torch.bfloat16)
        self.kv.mul_(0.5)
        bt = torch.zeros((len(seqs), max(pages)), dtype=torch.int32, device=dev)
        slots, pos, p0 = [], [], 0
        for i, ((q, k), np_) in enumerate(zip(seqs, pages)):
            bt[i, :np_] = torch.arange(p0, p0 + np_, dtype=torch.int32, device=dev)
            for t in range(k - q, k):                                     # 這一步新 token 的位置與 slot
                pos.append(t)
                slots.append((p0 + t // PAGE) * PAGE + t % PAGE)
            p0 += np_
        self.block_table = bt
        self.slots = torch.tensor(slots, dtype=torch.long, device=dev)
        self.pos = torch.tensor(pos, dtype=torch.long, device=dev).view(1, -1)
        self.cu_q = torch.tensor([0] + list(_cumsum([q for q, _ in seqs])), dtype=torch.int32, device=dev)
        self.seqused_k = torch.tensor([k for _, k in seqs], dtype=torch.int32, device=dev)
        self.max_q, self.max_k = max(q for q, _ in seqs), max(k for _, k in seqs)
        self.last = (self.cu_q[1:] - 1).long()                            # 每條序列最後一個 token（lm_head 只算這些）
        self.ids = torch.randint(100, 20000, (1, self.T), generator=g, device=dev)
        self.tokens_per_step = n_dec + n_pre
        self.kv_bytes = self.kv.numel() * 2

    @torch.inference_mode()
    def forward(self):
        km = self.km
        m = km.hf.model
        H, Hq, D, T = km.H, km.Hq, km.D, self.T
        x = m.embed_tokens(self.ids)
        cos, sin = m.rotary_emb(x, self.pos)
        for li, layer in enumerate(m.layers):
            a = layer.self_attn
            h = layer.input_layernorm(x)
            q = a.q_proj(h).view(1, T, Hq, D)
            k = a.k_proj(h).view(1, T, H, D)
            if hasattr(a, "q_norm"):
                q, k = a.q_norm(q), a.k_norm(k)
            q, k = q.transpose(1, 2), k.transpose(1, 2)
            v = a.v_proj(h).view(1, T, H, D)
            q, k = km.apply_rope(q, k, cos, sin)
            kc = self.kv[li, 0].view(-1, H, D)
            vc = self.kv[li, 1].view(-1, H, D)
            kc.index_copy_(0, self.slots, k[0].transpose(0, 1))
            vc.index_copy_(0, self.slots, v[0])
            o = self.fa(q[0].transpose(0, 1).contiguous(), self.kv[li, 0], self.kv[li, 1], max_seqlen_q=self.max_q,
                        cu_seqlens_q=self.cu_q, max_seqlen_k=self.max_k, seqused_k=self.seqused_k, softmax_scale=D ** -0.5,
                        causal=True, block_table=self.block_table, fa_version=3)
            x = x + a.o_proj(o.reshape(1, T, Hq * D))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return km.hf.lm_head(m.norm(x[:, self.last]))


def _cumsum(xs):
    s = 0
    for v in xs:
        s += v
        yield s


class MatmulStep:
    """m7_busy 的背景：4096³ BF16 matmul（純算力）。"""

    def __init__(self, km, n: int = 4096, reps: int = 8):
        g = torch.Generator(device=km.device).manual_seed(1)
        self.a = torch.randn(n, n, device=km.device, dtype=torch.bfloat16, generator=g)
        self.b = torch.randn(n, n, device=km.device, dtype=torch.bfloat16, generator=g)
        self.c = torch.empty_like(self.a)
        self.reps, self.tokens_per_step, self.kv_bytes = reps, reps, 0

    def forward(self):
        for _ in range(self.reps):
            torch.mm(self.a, self.b, out=self.c)
        return self.c


class HbmStep:
    """m16_busy_mem 的背景：大塊 D2D 複製（純 HBM 頻寬）。"""

    def __init__(self, km, gib: float = 2.0):
        n = int(gib * (1 << 30)) // 2
        self.a = torch.empty(n, device=km.device, dtype=torch.bfloat16).normal_()
        self.b = torch.empty_like(self.a)
        self.tokens_per_step, self.kv_bytes = 1, 0

    def forward(self):
        self.b.copy_(self.a)
        return self.b


def step_bytes(km, spec: str) -> int:
    """這個背景常駐的 GPU 位元組（KV／緩衝；不含 activation、CUDA graph、CUDA context）。和 ModelStep／MatmulStep／HbmStep
    同一個算法，只算不配置——B0 的 MPS 模式開背景 process 之前，用來判斷放不放得下（10/26 22275）。"""
    p = spec.split(":")
    if p[0] == "idle":
        return 0
    if p[0] == "matmul":
        return 3 * 4096 * 4096 * 2
    if p[0] == "hbm":
        return 2 * (int(2.0 * (1 << 30)) // 2) * 2
    if p[0] == "decode":
        n_dec, S, n_pre, S0 = int(p[1]), int(p[2]), 0, 0
    elif p[0] == "cprefill":
        n_dec, S, n_pre, S0 = 0, 0, int(p[1]), int(p[2])
    elif p[0] == "mixed":
        n_dec, S, n_pre, S0 = int(p[2]), int(p[3]), int(p[1]), int(p[3])
    else:
        raise ValueError(spec)
    seqs = [(1, S + 1)] * n_dec + ([(n_pre, S0 + n_pre)] if n_pre else [])
    pages = sum(math.ceil(k / PAGE) for _, k in seqs)
    return km.L * 2 * pages * PAGE * km.H * km.D * 2


def make_step(km, spec: str):
    """spec：idle｜decode:B:S｜cprefill:C:S0｜mixed:P:B:S｜matmul｜hbm。"""
    p = spec.split(":")
    if p[0] == "decode":
        return ModelStep(km, int(p[1]), int(p[2]))
    if p[0] == "cprefill":
        return ModelStep(km, 0, 0, n_pre=int(p[1]), S0=int(p[2]))
    if p[0] == "mixed":
        return ModelStep(km, int(p[2]), int(p[3]), n_pre=int(p[1]), S0=int(p[3]))
    if p[0] == "matmul":
        return MatmulStep(km)
    if p[0] == "hbm":
        return HbmStep(km)
    raise ValueError(spec)


class Background:
    """在自己的 stream 上一直 replay 一步的 CUDA graph。start()／stop()；rate() 回傳最近一段時間的每秒步數。
    priority：CUDA stream 優先權（數字小＝優先；torch 的範圍通常是 −5..0 或 −1..0）。"""

    def __init__(self, km, spec: str, priority: int = 0, use_graph: bool = True, inflight: int = 4):
        self.spec, self.km = spec, km
        self.stream = torch.cuda.Stream(device=km.device, priority=priority)
        self.idle = spec == "idle"
        self.inflight = inflight
        self.steps, self._stop, self._t = 0, threading.Event(), []
        self.thread = None
        if self.idle:
            return
        self.step = make_step(km, spec)
        self.graph = None
        with torch.cuda.stream(self.stream):
            for _ in range(3):                                          # 暖機（含 fa3 的 JIT／autotune）
                self.step.forward()
        torch.cuda.synchronize()
        if use_graph:
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=self.stream):
                self.step.forward()
            torch.cuda.synchronize()
        self.tokens_per_step = self.step.tokens_per_step

    def _run(self):
        evs = []
        with torch.cuda.stream(self.stream):
            while not self._stop.is_set():
                if self.graph is not None:
                    self.graph.replay()
                else:
                    self.step.forward()
                e = torch.cuda.Event()
                e.record(self.stream)
                evs.append(e)
                if len(evs) >= self.inflight:                           # 佇列裡最多 inflight 步（GIL 在 synchronize 時放掉）
                    evs.pop(0).synchronize()
                    self.steps += 1
                    self._t.append(time.perf_counter())
            for e in evs:
                e.synchronize()
                self.steps += 1
                self._t.append(time.perf_counter())

    def start(self):
        if self.idle:
            return self
        self._stop.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        if self.idle or self.thread is None:
            return
        self._stop.set()
        self.thread.join()
        self.thread = None

    def rate(self, t0: float, t1: float) -> float:
        """[t0, t1] 之間完成的步數 ÷ 秒（用完成時間戳記；區間太短、沒有步完成時回 0）。"""
        if self.idle or t1 <= t0:
            return 0.0
        n = sum(1 for t in self._t if t0 <= t <= t1)
        return n / (t1 - t0)

    def solo_rates(self, n: int = 6, seconds: float = 1.0) -> list:
        """背景單獨跑：n 個 seconds 秒的時間窗各自的每秒步數（05 §2.1：σ 從這裡算）。"""
        if self.idle:
            return [0.0] * n
        self.start()
        time.sleep(0.5)
        out = []
        for _ in range(n):
            t0 = time.perf_counter()
            time.sleep(seconds)
            t1 = time.perf_counter()
            out.append(self.rate(t0, t1))
        self.stop()
        return out

    def solo_rate(self, seconds: float = 2.0) -> float:
        """背景單獨跑的每秒步數（自檢：P0b 的「背景單獨跑時的吞吐」）。"""
        if self.idle:
            return 0.0
        self.start()
        time.sleep(0.5)
        t0 = time.perf_counter()
        time.sleep(seconds)
        t1 = time.perf_counter()
        r = self.rate(t0, t1)
        self.stop()
        return r
