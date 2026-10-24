import triton
import triton.language as tl

@triton.jit
def _dq_rowscale(q_ptr, s_ptr, o_ptr, n_rows, G: tl.constexpr, BR: tl.constexpr):
    """每列 G 個值共用一個 FP32 scale；q 可以是 int8 或 fp8。"""
    pid = tl.program_id(0)
    r = pid * BR + tl.arange(0, BR)
    c = tl.arange(0, G)
    m = r[:, None] < n_rows
    off = r[:, None].to(tl.int64) * G + c[None, :]
    q = tl.load(q_ptr + off, mask=m, other=0).to(tl.float32)
    s = tl.load(s_ptr + r, mask=r < n_rows, other=0.0)
    tl.store(o_ptr + off, (q * s[:, None]).to(tl.bfloat16), mask=m)

@triton.jit
def _dq_tensorscale(q_ptr, o_ptr, scale, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    q = tl.load(q_ptr + off, mask=m, other=0).to(tl.float32)
    tl.store(o_ptr + off, (q * scale).to(tl.bfloat16), mask=m)
