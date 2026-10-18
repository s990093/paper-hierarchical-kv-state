# Copied verbatim (function bodies unchanged) from vLLM-KIVI, kernels/kivi_attention.py
#   https://github.com/LLAA178/vllm-kivi  commit cd0f8e9f2a1d760a4bbdfb59a7d577c564f7368c (2026-03-01)
#   License: Apache License 2.0 (per the project README). Copyright the vLLM-KIVI authors.
# Used only as a timing-only external reference in code/m10_attn_q.py (13_restore2_plan §8b): its 4-bit layout differs
# from ours (K: per channel per 16-token block; V: one scale per token) and it is decode-only, tuned for A100.
import torch
import triton
import triton.language as tl

@triton.jit
def _kivi_paged_attention_kernel(
    Q, K_cache, V_cache, K_scale_ptr, K_min_cache, V_scale_cache, V_min_cache, Out,
    stride_q_bs, stride_q_head, stride_q_d,
    stride_kc_slot, stride_kc_head, stride_kc_d,
    stride_vc_slot, stride_vc_head, stride_vc_d,
    block_tables_ptr, stride_bt_bs, stride_bt_block,
    context_lens_ptr, sm_scale, num_seqs, num_heads, head_dim, block_size,
    BLOCK_SIZE_D: tl.constexpr, BLOCK_SIZE_SEQ: tl.constexpr,
):
    """
    KIVI: A100 Peak Performance Version.
    - Contiguous Block Reads (16 tokens) for maximum bandwidth.
    - Tensor Core dots for zero-latency compute.
    """
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    
    off_d_half = tl.arange(0, BLOCK_SIZE_D // 2)
    off_d0 = off_d_half * 2
    off_d1 = off_d_half * 2 + 1
    
    # Load Query into registers once
    q0 = tl.load(Q + cur_seq * stride_q_bs + cur_head * stride_q_head + off_d0)
    q1 = tl.load(Q + cur_seq * stride_q_bs + cur_head * stride_q_head + off_d1)
    q0_2d = tl.view(q0, (1, BLOCK_SIZE_D // 2))
    q1_2d = tl.view(q1, (1, BLOCK_SIZE_D // 2))
    
    m_i, l_i = -float('inf'), 0.0
    acc0 = tl.zeros([1, BLOCK_SIZE_D // 2], dtype=tl.float32)
    acc1 = tl.zeros([1, BLOCK_SIZE_D // 2], dtype=tl.float32)
    
    context_len = tl.load(context_lens_ptr + cur_seq)
    
    # Iterate block by block (block_size=16) to ensure memory coalescing
    for start_n in range(0, context_len, BLOCK_SIZE_SEQ):
        # Load ONE physical block ID
        physical_block_id = tl.load(block_tables_ptr + cur_seq * stride_bt_bs + (start_n // block_size) * stride_bt_block)
        
        # Metadata is shared for the entire 16-token block in KIVI
        first_slot = physical_block_id * block_size
        sk0 = tl.load(K_scale_ptr + first_slot * stride_kc_slot + cur_head * stride_kc_head + off_d0)
        sk1 = tl.load(K_scale_ptr + first_slot * stride_kc_slot + cur_head * stride_kc_head + off_d1)
        mk0 = tl.load(K_min_cache + first_slot * stride_kc_slot + cur_head * stride_kc_head + off_d0)
        mk1 = tl.load(K_min_cache + first_slot * stride_kc_slot + cur_head * stride_kc_head + off_d1)
        
        # Load 16 tokens of K (Contiguous in memory)
        off_n = tl.arange(0, BLOCK_SIZE_SEQ)
        slot_ids = physical_block_id * block_size + off_n
        mask_n = (start_n + off_n) < context_len
        
        pk = tl.load(K_cache + slot_ids[:, None] * stride_kc_slot + cur_head * stride_kc_head + off_d_half[None, :], mask=mask_n[:, None], other=0)
        
        # Tensor Core Dequant + Dot
        dk0 = ((pk & 0x0F).to(tl.float32) * sk0[None, :] + mk0[None, :]).to(tl.float16)
        dk1 = ((pk >> 4).to(tl.float32) * sk1[None, :] + mk1[None, :]).to(tl.float16)
        
        qk = (tl.dot(q0_2d.to(tl.float16), tl.trans(dk0)) + tl.dot(q1_2d.to(tl.float16), tl.trans(dk1))) * sm_scale
        qk_1d = tl.view(qk, (BLOCK_SIZE_SEQ,))
        qk_1d = tl.where(mask_n, qk_1d, -float('inf'))
        
        # Softmax update
        m_ij = tl.max(qk_1d, axis=0)
        m_next = tl.maximum(m_i, m_ij)
        alpha, p_1d = tl.exp(m_i - m_next), tl.exp(qk_1d - m_next)
        l_i = l_i * alpha + tl.sum(p_1d, axis=0)
        
        # Load 16 tokens of V (Contiguous)
        pv = tl.load(V_cache + slot_ids[:, None] * stride_vc_slot + cur_head * stride_vc_head + off_d_half[None, :], mask=mask_n[:, None], other=0)
        sv = tl.load(V_scale_cache + slot_ids * num_heads + cur_head, mask=mask_n, other=1.0)
        mv = tl.load(V_min_cache + slot_ids * num_heads + cur_head, mask=mask_n, other=0.0)
        
        dv0 = ((pv & 0x0F).to(tl.float32) * sv[:, None] + mv[:, None]).to(tl.float16)
        dv1 = ((pv >> 4).to(tl.float32) * sv[:, None] + mv[:, None]).to(tl.float16)
        
        # Tensor Core Update Acc
        p_2d = tl.view(p_1d, (1, BLOCK_SIZE_SEQ)).to(tl.float16)
        acc0 = acc0 * alpha + tl.dot(p_2d, dv0)
        acc1 = acc1 * alpha + tl.dot(p_2d, dv1)
        m_i = m_next

    # Final store
    acc0, acc1 = acc0 / l_i, acc1 / l_i
    tl.store(Out + cur_seq * stride_q_bs + cur_head * stride_q_head + off_d0, tl.view(acc0, (BLOCK_SIZE_D // 2,)).to(tl.float16))
    tl.store(Out + cur_seq * stride_q_bs + cur_head * stride_q_head + off_d1, tl.view(acc1, (BLOCK_SIZE_D // 2,)).to(tl.float16))

def kivi_paged_attention(q, k_cache, v_cache, k_scale, k_min, v_scale, v_min, block_tables, context_lens, sm_scale, block_size):
    num_seqs, num_heads, head_dim = q.shape
    out = torch.empty_like(q)
    _kivi_paged_attention_kernel[(num_seqs, num_heads)](
        q, k_cache, v_cache, k_scale, k_min, v_scale, v_min, out,
        q.stride(0), q.stride(1), q.stride(2), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), block_tables, block_tables.stride(0), block_tables.stride(1),
        context_lens, sm_scale, num_seqs, num_heads, head_dim, block_size,
        BLOCK_SIZE_D=head_dim, BLOCK_SIZE_SEQ=block_size, num_warps=4, num_stages=4
    )
    return out
