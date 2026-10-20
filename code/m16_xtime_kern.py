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
