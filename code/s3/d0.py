#!/usr/bin/env python3
"""d0.py — 第三階段 D0（05 §7、P0f）：MiniCPM4.1-8B 的原生稀疏索引（InfLLM-v2）。

  smoke --out J     P0f：HF transformers（trust_remote_code）載入 MiniCPM4.1-8B、用它自己的 sparse_config，跑一個 32K 的問題；
                    包住 infllmv2_cuda_impl 的兩個函式計數，確認稀疏路徑真的有跑；記下 modeling_minicpm.py 的位置與 sha256
                    → 能／不能（不計分、不進 results 的判定）
D0 的量測（聯集佔比、答案塊 recall）在讀過 modeling 程式、登記 runs/D0_native_index.md 之後才寫。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


def cmd_smoke(a):
    out = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"), run_id=os.environ.get("RUN_ID", "adhoc"))
    try:
        import infllmv2_cuda_impl as I
        out["infllmv2_module"] = getattr(I, "__file__", "?")
        names = [n for n in dir(I) if n.startswith("infllmv2")]
        out["infllmv2_functions"] = names
        calls = {n: 0 for n in names}
        for n in names:
            f = getattr(I, n)
            if callable(f):
                def wrap(*x, _f=f, _n=n, **k):
                    calls[_n] += 1
                    return _f(*x, **k)
                setattr(I, n, wrap)
    except Exception as ex:                                       # noqa: BLE001
        out["infllmv2_import"] = f"FAILED: {ex!r}"
        print(json.dumps(out, indent=1))
        json.dump(out, open(a.out, "w"), indent=1)
        raise SystemExit(2)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = os.environ["M7_MINICPM_GLOB"]
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    out["t_load_s"] = round(time.time() - t0, 1)
    cfg = model.config
    out["sparse_config"] = getattr(cfg, "sparse_config", None)
    out["max_position_embeddings"] = getattr(cfg, "max_position_embeddings", None)
    mod = sys.modules[type(model).__module__]
    out["modeling_file"] = mod.__file__
    out["modeling_sha256"] = hashlib.sha256(open(mod.__file__, "rb").read()).hexdigest()
    import m10_tasks as T
    it = T.make_item("T2", "front", "back", 71000, tok, None, n_blocks=64)
    ids = torch.tensor([it["context_ids"] + it["turns"][0]["query_ids"]], device="cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        o = model(ids, use_cache=True)
    torch.cuda.synchronize()
    out["t_forward_32k_s"] = round(time.time() - t0, 2)
    out["n_tokens"] = ids.shape[1]
    out["logits_finite"] = bool(torch.isfinite(o.logits[0, -1]).all())
    out["infllmv2_calls"] = dict(calls)
    out["sparse_path_used"] = any(v > 0 for v in calls.values())
    out["gpu_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
    print(json.dumps(out, indent=1, default=str), flush=True)
    json.dump(out, open(a.out, "w"), indent=1, default=str)
    if not (out["sparse_path_used"] and out["logits_finite"]):
        raise SystemExit("D0 冒煙：稀疏路徑沒有被呼叫，或 logits 不是有限值（規則 2：停）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("smoke")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    {"smoke": cmd_smoke}[a.cmd](a)
