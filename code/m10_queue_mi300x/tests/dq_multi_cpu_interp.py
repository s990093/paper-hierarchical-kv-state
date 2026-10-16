"""09 §6.4：decode_multi（一次拆多個 block、寫到各自位置）與 K4（FP8 不查表）的索引驗證（CPU，Triton 直譯模式）。
直譯器轉 BF16 是截斷，所以比的是「FP32 參考值截斷成 BF16」（同 dq_fast_cpu_interp.py）。另驗：沒寫到的段保持 0。
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


def ref32(e):
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


torch.manual_seed(0)
ok = True
for shape in ((2, 2, 2, 64, 128), (3, 2, 4, 96, 64)):
    L, _, H, T, D = shape
    NOUT = 7
    for fmt in ("fp8_head", "fp8_channel", "fp8_token", "int8_token", "int4_token", "int4_kivi"):
        if fmt == "int4_kivi" and T % C.KIVI_GROUP:
            continue
        for nb, bids_l in ((1, [3]), (3, [6, 1, 4]), (4, [0, 1, 2, 3])):
            encs = []
            for _ in range(nb):
                x = (torch.randn(shape) * 2).to(torch.bfloat16)
                x[:, 0, :, :, 5] += 40
                x[:, 0, 0, 0, :] = 0                           # 一些 0 與次正規數附近的值
                x[:, 1, 0, 1, :8] = torch.tensor([1e-3, -1e-3, 2e-3, 5e-4, -5e-4, 0.0, -0.0, 3e-3])
                encs.append(C.encode(x, fmt))
            gp = DF.group_parts(encs[0], nb, "cpu")
            for j, e in enumerate(encs):
                DF.sub_enc(e, gp, j).copy_from(e)
            buf = torch.zeros((L, 2, H, NOUT * T, D), dtype=torch.bfloat16)
            for bits in ((False, True) if fmt.startswith("fp8") else (False,)):
                buf.zero_()
                DF.decode_multi(fmt, shape, gp, nb, buf, torch.tensor(bids_l, dtype=torch.int64), bits=bits)
                good = True
                for j, b in enumerate(bids_l):
                    good &= torch.equal(buf[:, :, :, b * T:(b + 1) * T], trunc(ref32(encs[j])))
                untouched = [b for b in range(NOUT) if b not in bids_l]
                clean = all(not bool(buf[:, :, :, b * T:(b + 1) * T].any()) for b in untouched)
                print(("PASS" if good and clean else "FAIL"), shape, fmt, f"nb={nb}", f"bits={bits}", f"值={good} 其他段乾淨={clean}", flush=True)
                ok &= good and clean
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
