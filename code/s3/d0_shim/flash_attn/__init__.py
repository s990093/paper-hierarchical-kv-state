"""flash_attn 轉接層（只給 D0 用；不是 flash-attn 套件）。

MiniCPM4.1-8B 的 modeling_minicpm.py（revision 3a8dfed9）的 dense 路徑 import `flash_attn`（flash_attn_func、
flash_attn_varlen_func、bert_padding），稀疏路徑 import `infllm_v2`。這台的 conda 環境沒有 flash-attn（torch 2.13＋CUDA 13，
沒有現成的 wheel）。infllm_v2（OpenBMB/infllmv2_cuda_impl，FA2 分出來的）的 `infllmv2_attn_varlen_func` 介面和 FA2 相同，
`topk_idx=None` 時傳給核心的 block mask 是 None（＝一般的 dense attention）→ dense 也走同一套核心。
d0.py smoke 會拿 torch SDPA 對答案（`shim_check`），對不上就停。版本號 2.6.3：modeling 程式的 `unpad_input` 拆 4 個值（FA < 2.7 的介面）。
"""
import torch

__version__ = "2.6.3+infllmv2shim"


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, softmax_scale=None,
                           causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, deterministic=False,
                           return_attn_probs=False, block_table=None):
    import infllm_v2                                       # 呼叫時才取（d0.py 包的計數器才看得到 dense 的呼叫）
    return infllm_v2.infllmv2_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=dropout_p,
                                               softmax_scale=softmax_scale, causal=causal, window_size=window_size, softcap=softcap,
                                               alibi_slopes=alibi_slopes, deterministic=deterministic,
                                               return_attn_probs=return_attn_probs, block_table=block_table, topk_idx=None)


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                    deterministic=False, return_attn_probs=False):
    """(B, S, H, D) → 攤成 varlen（每條序列等長）。"""
    B, Sq, Hq, D = q.shape
    Sk, Hk = k.shape[1], k.shape[2]
    cq = torch.arange(0, (B + 1) * Sq, Sq, dtype=torch.int32, device=q.device)
    ck = torch.arange(0, (B + 1) * Sk, Sk, dtype=torch.int32, device=q.device)
    o = flash_attn_varlen_func(q.reshape(B * Sq, Hq, D), k.reshape(B * Sk, Hk, D), v.reshape(B * Sk, Hk, v.shape[-1]), cq, ck, Sq, Sk,
                               dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal, window_size=window_size,
                               softcap=softcap, alibi_slopes=alibi_slopes, deterministic=deterministic)
    return o.reshape(B, Sq, Hq, -1)
