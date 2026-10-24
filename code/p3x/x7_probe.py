"""x7_probe.py — X7（3090 成本常數）開跑前的探測：harness 的注意力在 CUDA 上走哪個 SDPA kernel、PCIe 在負載下的世代。

為什麼：m7_model.forward_span 用 F.scaled_dot_product_attention(q, K, V, attn_mask=causal_lower_right, enable_gqa=True)。
MI300X 上它走 flash kernel（07 §3.2）。CUDA sm_86 上若 flash／mem-efficient 不接受這組參數，會退回 math（把注意力矩陣整個做出來），
f(i) 會慢很多而且和 MI300X 不可比。所以先把每個 backend 強制試一次，記成功或完整錯誤，再和預設的時間比。

  python code/p3x/x7_probe.py --out $RUN_DIR/probe.json

不改任何既有程式；只 import torch。輸出：probe.json（run_id、ts、每個 backend 的結果）。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right

SHAPES = {"llama": dict(Hq=32, H=8, D=128), "qwen": dict(Hq=28, H=4, D=128)}


def smi(q):
    try:
        return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader", "-i",
                               os.environ.get("CUDA_VISIBLE_DEVICES", "0")],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"ERR {e!r}"


def time_it(fn, n=5):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--kv-len", type=int, nargs="+", default=[2048, 32768])
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    res = {"run_id": os.environ.get("RUN_ID", "NO_RUN_ID"), "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
           "torch": torch.__version__, "cuda": torch.version.cuda, "hip": torch.version.hip,
           "device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
           "cudnn": torch.backends.cudnn.version(), "sdpa": {}, "pcie": {}}
    res["pcie"]["idle"] = smi("pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max")
    for model, s in SHAPES.items():
        for kv_len in a.kv_len:
            q = torch.randn(1, s["Hq"], 512, s["D"], device=dev, dtype=torch.bfloat16)
            k = torch.randn(1, s["H"], kv_len, s["D"], device=dev, dtype=torch.bfloat16)
            v = torch.randn_like(k)
            mask = causal_lower_right(512, kv_len)
            ref = None
            for name, be in (("default", None), ("flash", SDPBackend.FLASH_ATTENTION),
                             ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION), ("cudnn", SDPBackend.CUDNN_ATTENTION),
                             ("math", SDPBackend.MATH)):
                key = f"{model}|kv{kv_len}|{name}"

                def call():
                    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
                try:
                    if be is None:
                        o = call()
                        t = time_it(call)
                    else:
                        with sdpa_kernel([be]):
                            o = call()
                            t = time_it(call)
                    torch.cuda.synchronize()
                    if ref is None:
                        ref = o.float()
                    err = float((o.float() - ref).abs().max())
                    res["sdpa"][key] = {"ok": True, "median_ms": round(t * 1e3, 4), "max_abs_diff_vs_default": err}
                except Exception as ex:  # noqa: BLE001  照實記完整錯誤（規則 2：這是探測，不是量測主體）
                    res["sdpa"][key] = {"ok": False, "error": repr(ex), "traceback": traceback.format_exc()[-3000:]}
                print(key, res["sdpa"][key] if res["sdpa"][key]["ok"] else res["sdpa"][key]["error"][:300], flush=True)
            del q, k, v
            torch.cuda.empty_cache()
    # PCIe 世代：連續 H2D 時讀 nvidia-smi（閒置時卡會降到 Gen1 省電）
    n = 1 << 30
    g = torch.empty(n, dtype=torch.uint8, device=dev)
    h = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        for _ in range(40):
            g.copy_(h, non_blocking=True)
    time.sleep(0.5)
    res["pcie"]["during_h2d"] = smi("pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max")
    st.synchronize()
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "sdpa"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
