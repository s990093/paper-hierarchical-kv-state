"""flash_attn.bert_padding 的三個函式（推論用；FA < 2.7 的介面：unpad_input 回傳 4 個值）。"""
import torch
import torch.nn.functional as F


def index_first_axis(x, indices):
    """呼叫端傳進來的已經是攤平的 (b·s, ...)：沿第一軸取。"""
    return x[indices]


def unpad_input(hidden_states, attention_mask):
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen = int(seqlens.max().item())
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    return hidden_states.reshape(-1, *hidden_states.shape[2:])[indices], indices, cu_seqlens, max_seqlen


def pad_input(hidden_states, indices, batch, seqlen):
    out = torch.zeros((batch * seqlen, *hidden_states.shape[1:]), device=hidden_states.device, dtype=hidden_states.dtype)
    out[indices] = hidden_states
    return out.reshape(batch, seqlen, *hidden_states.shape[1:])
