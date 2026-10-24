# 用法：CUDA_VISIBLE_DEVICES= python -I code/tests_h200/aot_triton_sm90.py code（head node，CPU；2026-10-17）
"""m10_dequant_fast 的 Triton kernel 在 head node（CPU，沒有 GPU）上 AOT 編譯成 sm_90（H200）的 cubin。
只驗「NVIDIA 後端編得過」（例：MI300X 上沒事、NVIDIA 上 `cannot cast ... to fp8e4nv` 那種前端／型別錯誤），不驗數值。"""
import inspect, sys
sys.path.insert(0, sys.argv[1])
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
import m10_dequant_fast as K

TARGET = GPUTarget("cuda", 90, 32)
def sig_of(fn, ptrs, consts):
    names = list(inspect.signature(fn.fn).parameters)
    sig = {}
    for n in names:
        if n in consts: sig[n] = "constexpr"
        elif n in ptrs: sig[n] = ptrs[n]
        else: sig[n] = "i64" if n == "R" else "i32"
    return sig

def compile_one(label, fn, ptrs, consts, fp_fusion=True):
    src = ASTSource(fn=fn, signature=sig_of(fn, ptrs, consts), constexprs=consts)
    try:
        k = triton.compile(src, target=TARGET, options={"num_warps": 4, "enable_fp_fusion": fp_fusion})
        n = len(k.asm.get("cubin", b""))
        print(f"PASS {label}: cubin {n} B")
        return True
    except Exception as ex:
        msg = str(ex).strip().splitlines()
        print(f"FAIL {label}: {type(ex).__name__}: {' | '.join(msg[-6:])[:600]}")
        return False

ok = []
F32, F16, U8, I8, BF16, I32 = "*fp32", "*fp16", "*u8", "*i8", "*bf16", "*i32"
# _dq_kernel：FP8（uint8 碼＋查表）、INT8（int8）
for is_fp8, q in ((True, U8), (False, I8)):
    ok.append(compile_one(f"_dq_kernel IS_FP8={is_fp8} q={q}", K._dq_kernel,
                          {"q_ptr": q, "s_ptr": F32, "lut_ptr": F32, "o_ptr": BF16}, {"D": 128, "IS_FP8": is_fp8, "ROWS": 16}))
# _dq4_kernel：int4_token（scale fp32、沒有 zero、OFFSET8）、KIVI 的 K（TDIV 32）與 V（DDIV 32），scale／zero fp16、關 FMA
ok.append(compile_one("_dq4_kernel int4_token", K._dq4_kernel, {"q_ptr": U8, "s_ptr": F32, "z_ptr": F32, "o_ptr": BF16},
                      {"D": 128, "TDIV": 1, "DDIV": 1, "HAS_ZERO": False, "OFFSET8": True, "ROWS": 16}, fp_fusion=False))
for nm, td, dd in (("K", 32, 1), ("V", 1, 32)):
    ok.append(compile_one(f"_dq4_kernel kivi-{nm}", K._dq4_kernel, {"q_ptr": U8, "s_ptr": F16, "z_ptr": F16, "o_ptr": BF16},
                          {"D": 128, "TDIV": td, "DDIV": dd, "HAS_ZERO": True, "OFFSET8": False, "ROWS": 16}, fp_fusion=False))
# _dqm_kernel（多 block）：FP8 查表、FP8 位元運算（BITS，09 §6.4 的 K4）、INT8
for is_fp8, bits, q in ((True, False, U8), (True, True, U8), (False, False, I8)):
    ok.append(compile_one(f"_dqm_kernel IS_FP8={is_fp8} BITS={bits}", K._dqm_kernel,
                          {"q_ptr": q, "s_ptr": F32, "lut_ptr": F32, "o_ptr": BF16, "bid_ptr": I32},
                          {"D": 128, "IS_FP8": is_fp8, "BITS": bits, "ROWS": 16}))
# _dq4m_kernel（多 block）
ok.append(compile_one("_dq4m_kernel int4_token", K._dq4m_kernel, {"q_ptr": U8, "s_ptr": F32, "z_ptr": F32, "o_ptr": BF16, "bid_ptr": I32},
                      {"D": 128, "TDIV": 1, "DDIV": 1, "HAS_ZERO": False, "OFFSET8": True, "ROWS": 16}, fp_fusion=False))
for nm, td, dd in (("K", 32, 1), ("V", 1, 32)):
    ok.append(compile_one(f"_dq4m_kernel kivi-{nm}", K._dq4m_kernel, {"q_ptr": U8, "s_ptr": F16, "z_ptr": F16, "o_ptr": BF16, "bid_ptr": I32},
                          {"D": 128, "TDIV": td, "DDIV": dd, "HAS_ZERO": True, "OFFSET8": False, "ROWS": 16}, fp_fusion=False))
print(f"{sum(ok)}/{len(ok)} PASS（triton {triton.__version__}，target {TARGET}）")
sys.exit(0 if all(ok) else 1)
