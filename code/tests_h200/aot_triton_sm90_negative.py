# 用法：CUDA_VISIBLE_DEVICES= python -I code/tests_h200/aot_triton_sm90_negative.py code code/tests_h200（f3k_extract.py＝m9_f3_dequant 的 _dq_rowscale、_dq_tensorscale 原文抽出；m9_f3 一 import 就建 /mlsteam 的目錄）
"""反向對照：m9_f3_dequant 的 FP8 Triton 寫法（H200 job 20614 實測 `cannot cast int32 ... to fp8e4nv`）用同樣的 AOT 方法編，應該 FAIL。"""
import sys
code = sys.argv[1]; NEG = sys.argv[2]
sys.path.insert(0, code)

import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
sys.path.insert(0, NEG); import f3k_extract as M
cases = [("_dq_rowscale", {"q_ptr": "*fp8e4nv", "s_ptr": "*fp32", "o_ptr": "*bf16", "n_rows": "i32", "G": "constexpr", "BR": "constexpr"}, {"G": 128, "BR": 16}),
         ("_dq_tensorscale", {"q_ptr": "*fp8e4nv", "o_ptr": "*bf16", "scale": "fp32", "n": "i32", "BLOCK": "constexpr"}, {"BLOCK": 1024})]
for name, sig, consts in cases:
    try:
        triton.compile(ASTSource(fn=getattr(M, name), signature=sig, constexprs=consts), target=GPUTarget("cuda", 90, 32))
        print(f"{name}: compiled（和 H200 實測不符）")
    except Exception as ex:
        lines = [l.strip() for l in str(ex).splitlines() if l.strip()]
        hit = [l for l in lines if "cast" in l.lower()]
        print(f"{name}: FAIL（預期）→ {(hit or lines)[-1][:160]}")
