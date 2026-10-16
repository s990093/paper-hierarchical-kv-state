"""融合反量化寫進「整個 KV 緩衝的視圖」時，位址偏移會不會超過 int32（CPU，Triton 直譯模式，不碰 GPU）。
還原器的 out＝km.kv[:, :, :, i*512:(i+1)*512]：Llama 32K 的緩衝 [32,2,8,33280,128]，最遠的元素偏移約 2.18e9 > 2^31−1。
做法：開一塊大 tensor，前面留 >4.4 GB 的護欄、後面接一個 Llama 大小的 KV 緩衝；把一個 block 解進第 63 個 chunk 的視圖，
檢查 (1) 視圖的值＝FP32 參考值截斷成 BF16（直譯器轉 BF16 是截斷）；(2) 護欄全為 0（溢位的寫入會往前繞到護欄裡）。
用法：python dq_fast_offset_cpu_interp.py [fmt ...]
"""
import os
import sys
os.environ["TRITON_INTERPRET"] = "1"
for k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
    os.environ[k] = ""
sys.path.insert(0, "/mlsteam/workspace/paper-hierarchical-kv-state/code")
import torch                                   # noqa: E402
import m10_codec as C                          # noqa: E402
import m10_dequant_fast as DF                  # noqa: E402


def ref32(e):                                  # 和 dq_fast_cpu_interp.py 相同（那支 import 時會直接跑測試，所以抄過來）
    p, (L_, _, H_, T_, D_) = e.parts, e.shape
    if e.fmt.startswith("fp8_"):
        return p["q"].view(C.FP8).float() * p["scale"]
    if e.fmt == "int8_token":
        return p["q"].float() * p["scale"]
    if e.fmt == "int4_token":
        return (C._unpack4(p["q"]).float() - 8) * p["scale"]
    G = C.KIVI_GROUP
    k = C._unpack4(p["k_q"]).float().view(L_, H_, T_ // G, G, D_) * p["k_scale"].float() + p["k_zero"].float()
    v = C._unpack4(p["v_q"]).float().view(L_, H_, T_, D_ // G, G) * p["v_scale"].float() + p["v_zero"].float()
    return torch.stack((k.reshape(L_, H_, T_, D_), v.reshape(L_, H_, T_, D_)), dim=1)


def trunc(y):
    return (y.contiguous().view(torch.int32) & -65536).view(torch.float32).to(torch.bfloat16)


L, H, D, T = 32, 8, 128, 512
TMAX = 64 * 512 + 512
GUARD = 2_300_000_000                          # 元素（BF16）＝4.6 GB，大於最大可能的負向繞回 4.3 GB
NKV = L * 2 * H * TMAX * D
fmts = sys.argv[1:] or ["fp8_token", "int4_token", "int4_kivi"]
torch.manual_seed(0)
big = torch.zeros(GUARD + NKV, dtype=torch.bfloat16)
kv = big[GUARD:].view(L, 2, H, TMAX, D)
x = (torch.randn((L, 2, H, T, D)) * 2).to(torch.bfloat16)
ok = True
for fmt in fmts:
    big.zero_()
    e = C.encode(x, fmt)
    view = kv[:, :, :, 63 * T:64 * T, :]
    DF.decode_fast(e, out=view)
    good_val = torch.equal(view, trunc(ref32(e)))
    guard_clean = not bool(big[:GUARD].any())
    rest_clean = not bool(kv[:, :, :, :63 * T].any()) and not bool(kv[:, :, :, 64 * T:].any())
    print(f"{fmt}: 值正確={good_val}  護欄乾淨={guard_clean}  其他 chunk 乾淨={rest_clean}", flush=True)
    ok &= good_val and guard_clean and rest_clean
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
