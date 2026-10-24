"""m7_model.py — 自管 KV 的 Llama 前向（第一階段 harness 的計算核心）

為什麼不用 HF 的 generate / DynamicCache：
  Cake 的還原需要「前段由 GPU 依序重算、後段由 I/O 直接放進同一塊 KV 緩衝」，
  而且要能逐 chunk 量時間。HF 的 cache 物件每步 torch.cat，會把複製成本混進 f(i)。
  所以這裡重用 HF 的權重與 RoPE 模組，但注意力與 KV 緩衝自己管：
    KV 緩衝 kv[layer, 0/1(K/V), kv_head, pos, head_dim]，預先配置，零複製。
  正確性由 test_m7_correctness.py 對照 HF 整段 prefill 的 logits 與 KV 驗證。

chunk = 512 token；Llama-3.1-8B 每 token KV = 32 層 × 2 × 8 頭 × 128 × 2 B = 128 KiB，
一個 chunk = 64 MiB。主機端每個 chunk 存成連續的 [L, 2, H, 512, D] 區塊。
"""
from __future__ import annotations

import glob
import os

import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right

CHUNK = 512
# ICPE P0-1（2026-10-22，docs/h200/icpe/P0-1_recompute_speed.md §2.1）：attention 的實作。沒設＝sdpa＝原本的路徑（逐位元相同）。
#   fa2／fa3／fa4：vLLM 內附的 FlashAttention（varlen；K、V 用 KV 緩衝的轉置視圖，不複製）；
#   sdpa_cudnn：PyTorch SDPA 的 cuDNN 後端＋實體化的 lower-right 布林遮罩。key_keep（14 號）的路徑一律用 SDPA。
ATTN_CHOICES = ("sdpa", "fa2", "fa3", "fa4", "sdpa_cudnn")
# 預設 Llama-3.1-8B；第 1 輪破解計劃（11）用 M7_MODEL_GLOB 換模型（LongAlpaca-7B、Qwen3-30B-A3B），不設時行為不變
DEFAULT_GLOB = "/mlsteam/data/tiara/hf-cache/hub/models--unsloth--Llama-3.1-8B-Instruct/snapshots/*"
MODEL_GLOB = os.environ.get("M7_MODEL_GLOB", DEFAULT_GLOB)     # 只留著相容；model_path() 不再用它


def model_path() -> str:
    """要載入時才讀 M7_MODEL_GLOB（2026-10-15 改）。原本在 import 時就讀成模組常數：若 m7_model 在呼叫端設好
    M7_MODEL_GLOB 之前就被 import（例如先 import m7_restore_harness），會靜默載入預設的 Llama，配上別的模型的
    tokenizer（m10_quality selfcheck：Qwen 的 token id 超出 Llama 的詞表 → GPU 上 gather 越界）。"""
    g = os.environ.get("M7_MODEL_GLOB", DEFAULT_GLOB)
    ps = sorted(glob.glob(g))
    if not ps:
        raise FileNotFoundError(g)
    return ps[-1]


class KVModel:
    def __init__(self, max_len: int = 40 * 1024, device: str = "cuda:0", load_hf_ref: bool = False):
        from transformers import AutoModelForCausalLM
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        self.apply_rope = apply_rotary_pos_emb
        self.device = torch.device(device)
        kw = {}
        if os.environ.get("M7_EXPERTS_IMPL"):      # MoE：eager（逐專家迴圈）或 grouped_mm
            kw["experts_implementation"] = os.environ["M7_EXPERTS_IMPL"]
        self.hf = AutoModelForCausalLM.from_pretrained(
            model_path(), dtype=torch.bfloat16, attn_implementation="sdpa", **kw).to(self.device).eval()
        cfg = self.hf.config
        self.cfg = cfg
        self.L = cfg.num_hidden_layers
        self.H = cfg.num_key_value_heads
        self.Hq = cfg.num_attention_heads
        # Qwen2 的 config 沒有 head_dim（2026-10-15 run 20261015-022121-m10-corr-qwen-8k 失敗）；照 HF Qwen2Attention 的預設
        self.D = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.max_len = max_len
        self.kv = torch.empty((self.L, 2, self.H, max_len, self.D), dtype=torch.bfloat16, device=self.device)
        self.chunk_bytes = self.L * 2 * self.H * CHUNK * self.D * 2
        # 依 q 長度與 kv 長度快取遮罩物件
        self._masks: dict[tuple[int, int], object] = {}
        # 14 號 D／DS（10/19）：丟掉 token。None＝原本的路徑（逐字相同）。
        #   [max_len] bool：所有層、head 共用（page／block 粒度）；[L, H, max_len] bool：每個（層, KV head）各自（per-head token）。
        #   False 的位置 attention 看不到＝把那些 K／V 從 cache 拿掉（softmax 只在留下的位置上做；RoPE 位置不變）。
        self.key_keep = None
        self.set_attn(os.environ.get("M7_ATTN", "sdpa"))

    def set_attn(self, name: str):
        """換 attention 的實作（ICPE P0-1）。載入失敗就丟錯（規則 2：不退回 sdpa）。"""
        if name not in ATTN_CHOICES:
            raise ValueError(f"M7_ATTN={name!r}：只接受 {ATTN_CHOICES}")
        self.attn = name
        self._fa = None
        if name.startswith("fa"):
            from vllm.vllm_flash_attn import flash_attn_varlen_func, fa_version_unsupported_reason, is_fa_version_supported
            ver = int(name[2:])
            if not is_fa_version_supported(ver):
                raise RuntimeError(f"{name} 在這張卡上不支援：{fa_version_unsupported_reason(ver)}")
            self._fa = (flash_attn_varlen_func, ver)
        self._cu: dict[tuple[int, int], tuple] = {}
        self._bmask: dict[tuple[int, int], torch.Tensor] = {}

    def _attn_alt(self, q: torch.Tensor, li: int, n: int, end: int) -> torch.Tensor:
        """sdpa 以外的實作；q：[1, Hq, n, D]（RoPE 之後）。回傳 [1, n, Hq*D]。"""
        if self._fa is not None:
            fn, ver = self._fa
            cu = self._cu.get((n, end))
            if cu is None:
                cu = self._cu[(n, end)] = (torch.tensor([0, n], dtype=torch.int32, device=self.device),
                                           torch.tensor([0, end], dtype=torch.int32, device=self.device))
            qq = q[0].transpose(0, 1).contiguous()                       # [n, Hq, D]
            k = self.kv[li, 0, :, :end].transpose(0, 1)                   # [end, H, D]（列距 D、head 距 max_len·D）
            v = self.kv[li, 1, :, :end].transpose(0, 1)
            o = fn(qq, k, v, max_seqlen_q=n, cu_seqlens_q=cu[0], max_seqlen_k=end, cu_seqlens_k=cu[1],
                   softmax_scale=self.D ** -0.5, causal=True, fa_version=ver)
            return o.reshape(1, n, self.Hq * self.D)
        # sdpa_cudnn：lower-right causal（第 t 個 query 看得到 ≤ end−n+t 的 key）
        from torch.nn.attention import SDPBackend, sdpa_kernel
        m = self._bmask.get((n, end))
        if m is None:
            pos_k = torch.arange(end, device=self.device)
            m = self._bmask[(n, end)] = pos_k.view(1, end) <= (end - n + torch.arange(n, device=self.device)).view(n, 1)
        K = self.kv[li, 0, :, :end].unsqueeze(0)
        V = self.kv[li, 1, :, :end].unsqueeze(0)
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            o = F.scaled_dot_product_attention(q, K, V, attn_mask=m, enable_gqa=True)
        return o.transpose(1, 2).reshape(1, n, self.Hq * self.D)

    # ---- KV 緩衝的 chunk 視圖 ------------------------------------------------
    def chunk_view(self, i: int) -> torch.Tensor:
        """第 i 個 chunk 在 GPU 緩衝的（非連續）視圖 [L,2,H,512,D]。"""
        return self.kv[:, :, :, i * CHUNK:(i + 1) * CHUNK, :]

    def host_chunk(self) -> torch.Tensor:
        return torch.empty((self.L, 2, self.H, CHUNK, self.D), dtype=torch.bfloat16, pin_memory=True)

    # ---- 前向 ---------------------------------------------------------------
    @torch.inference_mode()
    def forward_span(self, ids: torch.Tensor, start: int, want_logits: bool = False, n_logits: int = 1):
        """計算 ids（位置 start..start+n-1）的 KV 寫進緩衝；需要 [0,start) 的 KV 已在緩衝。

        回傳最後一個位置的 logits（want_logits=True 時）。n_logits>1：回傳最後 n_logits 個位置的 logits
        [n_logits, vocab]（m10 的 teacher forcing 用；n_logits=1 的路徑和原本完全相同）。
        """
        n = ids.shape[-1]
        end = start + n
        assert end <= self.max_len
        m = self.hf.model
        x = m.embed_tokens(ids.view(1, n))
        pos = torch.arange(start, end, device=self.device).view(1, n)
        cos, sin = m.rotary_emb(x, pos)
        key = (n, end)
        mask = self._masks.get(key)
        if mask is None:
            mask = causal_lower_right(n, end)
            self._masks[key] = mask
        for li, layer in enumerate(m.layers):
            a = layer.self_attn
            h = layer.input_layernorm(x)
            q = a.q_proj(h).view(1, n, self.Hq, self.D)
            k = a.k_proj(h).view(1, n, self.H, self.D)
            if hasattr(a, "q_norm"):                  # Qwen3：每個 head 先做 RMSNorm 再 RoPE
                q, k = a.q_norm(q), a.k_norm(k)
            q, k = q.transpose(1, 2), k.transpose(1, 2)
            v = a.v_proj(h).view(1, n, self.H, self.D).transpose(1, 2)
            q, k = self.apply_rope(q, k, cos, sin)
            self.kv[li, 0, :, start:end].copy_(k[0])
            self.kv[li, 1, :, start:end].copy_(v[0])
            if self.key_keep is None and self.attn != "sdpa":
                x = x + a.o_proj(self._attn_alt(q, li, n, end))
                x = x + layer.mlp(layer.post_attention_layernorm(x))
                continue
            K = self.kv[li, 0, :, :end].unsqueeze(0)
            V = self.kv[li, 1, :, :end].unsqueeze(0)
            if self.key_keep is None:
                o = F.scaled_dot_product_attention(q, K, V, attn_mask=mask, enable_gqa=True)
            else:
                o = self._attn_keep(q, K, V, li, start, n, end)
            o = o.transpose(1, 2).reshape(1, n, self.Hq * self.D)
            x = x + a.o_proj(o)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        if want_logits:
            if n_logits > 1:
                return self.hf.lm_head(m.norm(x[:, -n_logits:]))[0]
            x = m.norm(x[:, -1:])
            return self.hf.lm_head(x)[0, -1]
        return None

    def _attn_keep(self, q, K, V, li: int, start: int, n: int, end: int, sub: int = 64):
        """14 號：有 key_keep 時的 attention。遮罩＝causal（第 t 個 query 看到 ≤ start+t 的 key）且 key_keep。
        per-head 的遮罩是 [Hq, n, end]，太大，所以 query 每 sub 個一段算（每段的結果和整段一起算相同，attention 對 query 獨立）。"""
        kk = self.key_keep
        rep = self.Hq // self.H
        pos_k = torch.arange(end, device=self.device)
        outs = []
        step = n if kk.dim() == 1 else sub
        for s0 in range(0, n, step):
            s1 = min(n, s0 + step)
            causal = pos_k.view(1, end) <= (start + torch.arange(s0, s1, device=self.device)).view(-1, 1)   # [m, end]
            if kk.dim() == 1:
                m = (causal & kk[:end].view(1, end)).view(1, 1, s1 - s0, end)
            else:
                hk = kk[li, :, :end].repeat_interleave(rep, dim=0)                                        # [Hq, end]
                m = (causal.unsqueeze(0) & hk.unsqueeze(1)).unsqueeze(0)                                    # [1, Hq, m, end]
            outs.append(F.scaled_dot_product_attention(q[:, :, s0:s1], K, V, attn_mask=m, enable_gqa=True))
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)

    @torch.inference_mode()
    def prefill_chunked(self, ids: torch.Tensor, start: int = 0):
        """以 512 為單位依序 prefill（和還原時的重算切法相同）。"""
        n = ids.shape[-1]
        assert n % CHUNK == 0
        for s in range(0, n, CHUNK):
            self.forward_span(ids[s:s + CHUNK], start + s)
