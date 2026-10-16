"""GPU 版：Llama 32K 大小的整個 KV 緩衝（[32,2,8,33280,128]，4.4 GB）上，融合 kernel（decode_fast、decode_variant 的各 K、
decode_multi）寫進第 0、31、63 個 chunk 的視圖，和原寫法逐位元相同、其他 chunk 保持 0。10/16 修過 int32 溢位（CPU 直譯模式重現：
dq_fast_offset_cpu_interp.py）。stage 39 跑（GPU 不在計時）。"""
import sys
sys.path.insert(0, "/mlsteam/workspace/paper-hierarchical-kv-state/code")
import torch                                    # noqa: E402
import m10_codec as C                           # noqa: E402
import m10_dequant_fast as DF                   # noqa: E402

L, H, D, T = 32, 8, 128, 512
kv = torch.zeros((L, 2, H, 64 * T + T, D), dtype=torch.bfloat16, device="cuda")
torch.manual_seed(0)
x = (torch.randn((L, 2, H, T, D), device="cuda") * 2).to(torch.bfloat16)
x[:, 0, :, :, 5] += 40
ok = True
for fmt in ("fp8_token", "fp8_head", "int8_token", "int4_token", "int4_kivi"):
    e = C.encode(x, fmt)
    want = C.decode(e)
    for kern in ("fast", "k4", "k3:32:4", "multi"):
        if kern == "k4" and not fmt.startswith("fp8"):
            continue
        for i in (0, 31, 63):
            kv.zero_()
            v = kv[:, :, :, i * T:(i + 1) * T]
            if kern == "fast":
                DF.decode_fast(e, out=v)
            elif kern == "multi":
                DF.decode_multi(fmt, e.shape, {k: t.unsqueeze(0) for k, t in e.parts.items()}, 1, kv,
                                torch.tensor([i], dtype=torch.int64, device="cuda"))
            else:
                DF.decode_variant(e, v, kern)
            torch.cuda.synchronize()
            good = torch.equal(v, want) and not bool(kv[:, :, :, :i * T].any()) and not bool(kv[:, :, :, (i + 1) * T:].any())
            ok &= good
            print(("PASS" if good else "FAIL"), fmt, kern, f"chunk {i}", flush=True)
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
