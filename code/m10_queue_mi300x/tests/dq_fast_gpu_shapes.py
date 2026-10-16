"""融合反量化 vs 原寫法（m10_codec.decode），GPU 上、真實 block 形狀的逐位元比對（隨機資料＋一個離群通道，兩種幅度）。
self-test 用的是小的合成 tensor；這支補 Llama／Qwen 每個 block 的實際形狀。int4_kivi 的 encode 要求 T、D 被組大小整除，奇數形狀跳過。
用法：python dq_fast_gpu_shapes.py [--out x.json]
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "code"))
sys.path.insert(0, "/mlsteam/workspace/paper-hierarchical-kv-state/code")
import m10_codec as C                       # noqa: E402
from m10_dequant_fast import decode_fast    # noqa: E402

torch.manual_seed(0)
res, ok = {}, True
for name, shape in (("llama_block", (32, 2, 8, 512, 128)), ("qwen_block", (28, 2, 4, 512, 128)), ("odd", (3, 2, 2, 37, 128))):
    for scale in (1.0, 30.0):
        x = (torch.randn(shape, device="cuda") * scale).to(torch.bfloat16)
        x[..., 0] *= 20
        for fmt in ("fp8_head", "fp8_token", "int8_token", "int4_token", "int4_kivi"):
            k = f"{name}|x{scale}|{fmt}"
            try:
                e = C.encode(x, fmt)
            except AssertionError:
                res[k] = "SKIP（encode 不支援這個形狀）"
                print(k, res[k], flush=True)
                continue
            eq = torch.equal(decode_fast(e), C.decode(e))
            ok &= eq
            res[k] = "BITWISE" if eq else "DIFF"
            print(k, res[k], flush=True)
res["all_pass"] = bool(ok)
if "--out" in sys.argv:
    json.dump(res, open(sys.argv[sys.argv.index("--out") + 1], "w"), indent=1, ensure_ascii=False)
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
