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
# 預設 Llama-3.1-8B；第 1 輪破解計劃（11）用 M7_MODEL_GLOB 換模型（LongAlpaca-7B、Qwen3-30B-A3B），不設時行為不變
MODEL_GLOB = os.environ.get("M7_MODEL_GLOB", "/mlsteam/data/tiara/hf-cache/hub/models--unsloth--Llama-3.1-8B-Instruct/snapshots/*")


def model_path() -> str:
    ps = sorted(glob.glob(MODEL_GLOB))
    if not ps:
        raise FileNotFoundError(MODEL_GLOB)
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
        self.D = cfg.head_dim
        self.max_len = max_len
        self.kv = torch.empty((self.L, 2, self.H, max_len, self.D), dtype=torch.bfloat16, device=self.device)
        self.chunk_bytes = self.L * 2 * self.H * CHUNK * self.D * 2
        # 依 q 長度與 kv 長度快取遮罩物件
        self._masks: dict[tuple[int, int], object] = {}

    # ---- KV 緩衝的 chunk 視圖 ------------------------------------------------
    def chunk_view(self, i: int) -> torch.Tensor:
        """第 i 個 chunk 在 GPU 緩衝的（非連續）視圖 [L,2,H,512,D]。"""
        return self.kv[:, :, :, i * CHUNK:(i + 1) * CHUNK, :]

    def host_chunk(self) -> torch.Tensor:
        return torch.empty((self.L, 2, self.H, CHUNK, self.D), dtype=torch.bfloat16, pin_memory=True)

    # ---- 前向 ---------------------------------------------------------------
    @torch.inference_mode()
    def forward_span(self, ids: torch.Tensor, start: int, want_logits: bool = False):
        """計算 ids（位置 start..start+n-1）的 KV 寫進緩衝；需要 [0,start) 的 KV 已在緩衝。

        回傳最後一個位置的 logits（want_logits=True 時）。
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
            K = self.kv[li, 0, :, :end].unsqueeze(0)
            V = self.kv[li, 1, :, :end].unsqueeze(0)
            o = F.scaled_dot_product_attention(q, K, V, attn_mask=mask, enable_gqa=True)
            o = o.transpose(1, 2).reshape(1, n, self.Hq * self.D)
            x = x + a.o_proj(o)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        if want_logits:
            x = m.norm(x[:, -1:])
            return self.hf.lm_head(x)[0, -1]
        return None

    @torch.inference_mode()
    def prefill_chunked(self, ids: torch.Tensor, start: int = 0):
        """以 512 為單位依序 prefill（和還原時的重算切法相同）。"""
        n = ids.shape[-1]
        assert n % CHUNK == 0
        for s in range(0, n, CHUNK):
            self.forward_span(ids[s:s + CHUNK], start + s)
