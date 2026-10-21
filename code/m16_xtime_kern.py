"""m16_xtime_kern.py — m16_xtime.py（第三階段 X3）用的 GPU kernel：低秩重建之後「加平均＋（K 才有）套 RoPE＋寫進 KV 緩衝的視圖」
融合成一個 Triton kernel（讀一次、寫一次）。

為什麼要另寫：低秩重建＝一次 GEMM（rocBLAS／hipBLASLt）＋一段逐元素的收尾。收尾如果用 PyTorch 拼（加平均、rotate_half、
乘 cos、乘 sin、相加、copy 進視圖），同一份資料要來回 5 趟以上；拿它和第二階段的融合反量化 kernel（讀一次寫一次）比，
等於拿沒最佳化的寫法比最佳化過的寫法，「重建比反量化貴」會被高估。所以收尾也融合成一個 kernel，和 m10_dequant_fast 同一個等級。
PyTorch 的寫法（store_torch）留著當對照與數值參考，不是退路：Triton 失敗就照規則 2 停下來，由人決定要不要改用 --store-impl torch。

**這支在模組層就 import torch／triton**，只有 m16_xtime.py 的 GPU 路徑（micro／validate／--smoke）才 import 它；
--self-test 與 map 不碰它。

RoPE 照 HF 的 apply_rotary_pos_emb：x·cos＋rotate_half(x)·sin，rotate_half 把第 j 維和第 j＋D/2 維配成一對
（不是相鄰兩維；k1_kvstruct_howto.md §1）。這裡在 FP32 算、最後四捨五入成 BF16；HF 是每一步都在 BF16 算，
所以兩者不是逐位元相同（差 BF16 捨入的量級）——X3 只量時間，不拿這個值去做品質。

10/21 補缺口：加 coef_dequant（xKV 量化過的 token 係數 → BF16，再交給 GEMM）。為什麼不直接用 m10_dequant_fast_x._half：
它的一列是 head_dim 個值、tl.arange(0, D) 要 2 的次方；xKV 的係數一列是聯合秩 r＝r_head×4×H 個值（例 Llama k22＝704、
Qwen k11＝176），不是 2 的次方。所以另寫一個「列 × 主成分區塊（BR 個，用 mask）」的版本，數值運算和 m10_codec_x.dec_half
同一串（KIVI 是 FP32 先乘 scale 再加 zero、關掉 FMA 合併；int8_token 是 FP32 相乘），最後四捨五入成 BF16
→ 和 dec_half(...).to(bfloat16) 逐位元相同（m16_xtime 建 segment 時在 GPU 上驗，不過就停）。
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _store_kernel(x_ptr, m_ptr, c_ptr, s_ptr, o_ptr, R, T, H,
                  sx_l, sx_h, sx_t, sm_l, sm_h, so_l, so_h, so_t,
                  D: tl.constexpr, HALF: tl.constexpr, ROPE: tl.constexpr, ROWS: tl.constexpr):
    """每個 program 處理 ROWS 列；一列＝(層, head, token) 的 D 個值。
    x：[Lx,H,T,D]（stride 任意，最後一維連續）；m：[Lx,H,D]（平均）；c／s：[T,D] FP32 連續（這一段位置的 cos／sin）；
    o：[Lx,H,T,D] BF16 視圖（KV 緩衝的一段）。列號用 int64（KV 緩衝的偏移會超過 2^31，同 m10_dequant_fast 的修正）。"""
    pid = tl.program_id(0)
    r = (pid * ROWS + tl.arange(0, ROWS)).to(tl.int64)
    mask = r < R
    t = r % T
    h = (r // T) % H
    l = r // (T * H)
    d = tl.arange(0, HALF)
    m2 = mask[:, None]
    xo = l * sx_l + h * sx_h + t * sx_t
    mo = l * sm_l + h * sm_h
    x1 = tl.load(x_ptr + xo[:, None] + d[None, :], mask=m2, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + xo[:, None] + (d + HALF)[None, :], mask=m2, other=0.0).to(tl.float32)
    x1 = x1 + tl.load(m_ptr + mo[:, None] + d[None, :], mask=m2, other=0.0).to(tl.float32)
    x2 = x2 + tl.load(m_ptr + mo[:, None] + (d + HALF)[None, :], mask=m2, other=0.0).to(tl.float32)
    oo = l * so_l + h * so_h + t * so_t
    if ROPE:
        co = t * D
        c1 = tl.load(c_ptr + co[:, None] + d[None, :], mask=m2, other=0.0)
        c2 = tl.load(c_ptr + co[:, None] + (d + HALF)[None, :], mask=m2, other=0.0)
        s1 = tl.load(s_ptr + co[:, None] + d[None, :], mask=m2, other=0.0)
        s2 = tl.load(s_ptr + co[:, None] + (d + HALF)[None, :], mask=m2, other=0.0)
        y1 = x1 * c1 - x2 * s1
        y2 = x2 * c2 + x1 * s2
    else:
        y1 = x1
        y2 = x2
    tl.store(o_ptr + oo[:, None] + d[None, :], y1.to(tl.bfloat16), mask=m2)
    tl.store(o_ptr + oo[:, None] + (d + HALF)[None, :], y2.to(tl.bfloat16), mask=m2)


def store(x: torch.Tensor, mean: torch.Tensor, cos: torch.Tensor | None, sin: torch.Tensor | None,
          out: torch.Tensor, rope: bool, impl: str = "triton", rows: int = 16) -> torch.Tensor:
    """out ← (x＋mean) 套（或不套）RoPE。x、out：[Lx,H,T,D]；mean：[Lx,H,D]；cos、sin：[T,D] FP32。"""
    if impl == "torch":
        return store_torch(x, mean, cos, sin, out, rope)
    assert impl == "triton", impl
    Lx, H, T, D = x.shape
    assert out.shape == x.shape and mean.shape == (Lx, H, D), (tuple(x.shape), tuple(out.shape), tuple(mean.shape))
    assert x.stride(-1) == 1 and out.stride(-1) == 1 and mean.stride(-1) == 1 and D % 2 == 0
    assert out.dtype == torch.bfloat16
    if rope:
        assert cos.shape == (T, D) and sin.shape == (T, D) and cos.is_contiguous() and sin.is_contiguous()
        assert cos.dtype == torch.float32 and sin.dtype == torch.float32
        c, s = cos, sin
    else:
        c = s = mean                                   # 不會被讀（ROPE 是 constexpr，整段被編譯掉）
    R = Lx * H * T
    _store_kernel[(triton.cdiv(R, rows),)](x, mean, c, s, out, R, T, H,
                                           x.stride(0), x.stride(1), x.stride(2), mean.stride(0), mean.stride(1),
                                           out.stride(0), out.stride(1), out.stride(2),
                                           D=D, HALF=D // 2, ROPE=bool(rope), ROWS=rows)
    return out


@triton.jit
def _coef_kernel(q_ptr, s_ptr, z_ptr, o_ptr, NROW, T, R,
                 sq_a, sq_t, ss_a, ss_t, ss_r, so_a, so_t,
                 KIND: tl.constexpr, TDIV: tl.constexpr, ROWS: tl.constexpr, BR: tl.constexpr):
    """xKV 係數 [A, 1, T, R] 的反量化 → BF16 [A, T, R]。列＝(a, t)，每個 program 處理 ROWS 列 × BR 個主成分（R 用 mask，不必是 2 的次方）。
    KIND 0＝int8_token：y＝q×s（s 每列一個 FP32，[A,1,T,1]）。
    KIND 1＝kivi4 的 K 角色（m10_codec_x.enc_half(…, "kivi4", "k")：每個主成分各自 scale、沿 token 每 TDIV＝32 個一組）：
            y＝u×s＋z（s、z 是 FP16，[A,1,T/32,1,R]；u 是 4-bit，偶數 index 在低 4 位＝m10_codec._pack4）。
    列號用 int64（同 m10_dequant_fast 的修正）。"""
    pid = tl.program_id(0)
    pc = tl.program_id(1)
    r = (pid * ROWS + tl.arange(0, ROWS)).to(tl.int64)
    c = pc * BR + tl.arange(0, BR)
    mr = r < NROW
    m2 = mr[:, None] & (c < R)[None, :]
    t = r % T
    a = r // T
    qo = a * sq_a + t * sq_t
    if KIND == 0:
        u = tl.load(q_ptr + qo[:, None] + c[None, :], mask=m2, other=0).to(tl.float32)     # q 是有號的 int8
        s = tl.load(s_ptr + a * ss_a + t * ss_t, mask=mr, other=1.0).to(tl.float32)
        y = u * s[:, None]
    else:
        byte = tl.load(q_ptr + qo[:, None] + (c // 2)[None, :], mask=m2, other=0).to(tl.int32)
        u = tl.where((c % 2 == 0)[None, :], byte & 0xF, byte >> 4).to(tl.float32)
        so = (a * ss_a + (t // TDIV) * ss_t)[:, None] + (c * ss_r)[None, :]
        y = u * tl.load(s_ptr + so, mask=m2, other=1.0).to(tl.float32)
        y = y + tl.load(z_ptr + so, mask=m2, other=0.0).to(tl.float32)
    oo = a * so_a + t * so_t
    tl.store(o_ptr + oo[:, None] + c[None, :], y.to(tl.bfloat16), mask=m2)


def coef_dequant(p: dict, f: str, out: torch.Tensor, rows: int = 16, br: int = 128) -> torch.Tensor:
    """p＝m10_codec_x.enc_half(coef, f, "k") 的 parts（或它沿 token 軸＝第 2 維的切片），coef＝[A, 1, T, R]；
    f∈{kivi4, int8_token}；out＝BF16 [A, T, R]（最後一維連續，可以是大緩衝沿 token 的切片）。"""
    q = p["q"]
    A, B, T, Rq = q.shape
    assert B == 1 and q.stride(-1) == 1 and out.stride(-1) == 1 and out.dtype == torch.bfloat16, (q.shape, q.stride())
    if f == "int8_token":
        R, s, z, kind, tdiv, ss_r = Rq, p["scale"], p["scale"], 0, 1, 0
        assert q.dtype == torch.int8 and tuple(s.shape) == (A, 1, T, 1), (q.dtype, tuple(s.shape))
    elif f == "kivi4":
        R, s, z, kind, tdiv = 2 * Rq, p["scale"], p["zero"], 1, 32
        assert T % tdiv == 0 and tuple(s.shape) == (A, 1, T // tdiv, 1, R) and s.stride() == z.stride(), \
            (tuple(s.shape), s.stride(), z.stride())
        ss_r = s.stride(4)
    else:
        raise ValueError(f"coef_dequant 不支援 {f}")
    assert out.dim() == 3 and tuple(out.shape) == (A, T, R), (tuple(out.shape), (A, T, R))
    NROW = A * T
    _coef_kernel[(triton.cdiv(NROW, rows), triton.cdiv(R, br))](
        q, s, z, out, NROW, T, R, q.stride(0), q.stride(2), s.stride(0), s.stride(2), ss_r, out.stride(0), out.stride(1),
        KIND=kind, TDIV=tdiv, ROWS=rows, BR=br, enable_fp_fusion=False)
    return out


def store_torch(x, mean, cos, sin, out, rope: bool):
    """PyTorch 拼的寫法（FP32 算、最後轉 BF16）：數值參考與對照組。"""
    y = x.float() + mean.float()[:, :, None, :]
    if rope:
        h = y.shape[-1] // 2
        c, s = cos[None, None], sin[None, None]
        y1, y2 = y[..., :h], y[..., h:]
        y = torch.cat((y1 * c[..., :h] - y2 * s[..., :h], y2 * c[..., h:] + y1 * s[..., h:]), dim=-1)
    out.copy_(y.to(torch.bfloat16))
    return out
