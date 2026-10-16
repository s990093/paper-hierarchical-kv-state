# 直譯模式轉 BF16 的捨入方式和 GPU 不同（連 GPU 上逐位元相同的 FP8 kernel 在這裡也不同）→ 先確認是不是「截斷」，
# 再用「FP32 參考值截斷成 BF16」逐位元比對，驗 kernel 的索引（哪個 byte、哪半個、哪個 scale／zero）。
import os, sys
os.environ["TRITON_INTERPRET"] = "1"
sys.path.insert(0, "/mlsteam/workspace/paper-hierarchical-kv-state/code")
import torch
import m10_codec as C
import m10_dequant_fast as DF

def ref32(e):
    p, (L, _, H, T, D) = e.parts, e.shape
    if e.fmt.startswith("fp8_"):
        return p["q"].view(C.FP8).float() * p["scale"]
    if e.fmt == "int8_token":
        return p["q"].float() * p["scale"]
    if e.fmt == "int4_token":
        return (C._unpack4(p["q"]).float() - 8) * p["scale"]
    G = C.KIVI_GROUP
    k = C._unpack4(p["k_q"]).float().view(L, H, T // G, G, D) * p["k_scale"].float() + p["k_zero"].float()
    v = C._unpack4(p["v_q"]).float().view(L, H, T, D // G, G) * p["v_scale"].float() + p["v_zero"].float()
    return torch.stack((k.reshape(L, H, T, D), v.reshape(L, H, T, D)), dim=1)

def trunc(y):
    return (y.contiguous().view(torch.int32) & -65536).view(torch.float32).to(torch.bfloat16)

torch.manual_seed(0)
ok = True
for shape in ((2, 2, 2, 64, 128), (3, 2, 4, 96, 64)):
    L, _, H, T, D = shape
    x = (torch.randn(shape) * 2).to(torch.bfloat16)
    x[:, 0, :, :, 5] += 40
    buf = torch.zeros((L, 2, H, 3 * T, D), dtype=torch.bfloat16)
    view = buf[:, :, :, T:2 * T, :]
    for fmt in ("fp8_head", "fp8_channel", "fp8_token", "int8_token", "int4_token", "int4_kivi"):
        e = C.encode(x, fmt)
        r = ref32(e)
        assert torch.equal(r.to(torch.bfloat16), C.decode(e)), "ref32 和 m10_codec.decode 不一致"
        got = DF.decode_fast(e)
        a = torch.equal(got, trunc(r))
        buf.zero_(); DF.decode_fast(e, out=view)
        b = torch.equal(view, trunc(r)) and not bool(buf[:, :, :, :T].any()) and not bool(buf[:, :, :, 2 * T:].any())
        print(("PASS" if a and b else "FAIL"), shape, fmt, "eq_trunc", a, "into_view", b)
        ok &= a and b
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
