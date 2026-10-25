#!/usr/bin/env python3
"""d0.py — 第三階段 D0（05 §7、P0f）：MiniCPM4.1-8B 的原生稀疏索引（InfLLM-v2）。

  smoke --out J     P0f：HF transformers 4.56.1（pylibs 側裝）＋trust_remote_code 載入 MiniCPM4.1-8B，sparse_config 照 README 補上，
                    attn_implementation=flash_attention_2（dense 走 d0_shim/flash_attn → infllm_v2 的 varlen、topk_idx=None，先拿 SDPA 對答案）；
                    跑一個 32K 的問題，包住 infllm_v2 的函式計數，確認稀疏路徑真的有跑；記下 modeling_minicpm.py 的位置與 sha256、
                    第一次 sparse 呼叫的 topk_idx 形狀 → 能／不能（不計分、不進 results 的判定）
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


# MiniCPM4.1-8B 的 README（revision 3a8dfed9）「To enable InfLLM v2, you need to add the sparse_config field」原樣；
# config.json 本身沒有這個欄位（沒有就整個模型走 dense）。modeling 程式裡 .get() 的預設值和這組相同。
SPARSE_CONFIG = dict(kernel_size=32, kernel_stride=16, init_blocks=1, block_size=64, window_size=2048, topk=64, use_nope=False,
                     dense_len=8192)
TRANSFORMERS_PIN = "4.56.1"            # config.json 的 transformers_version；5.x 沒有 is_torch_fx_available，cache 的 layer API 也不同
WRAP = ["infllmv2_attn_stage1", "infllmv2_attn_varlen_func", "infllmv2_attn_with_kvcache", "max_pooling_1d", "max_pooling_1d_varlen"]


def wrap_infllm(I, calls: dict, seen: dict):
    """包住 infllm_v2 的函式計數。varlen 分成 dense（topk_idx=None，flash_attn 轉接層走這裡）與 sparse；記第一次 sparse 的 topk_idx 形狀。"""
    for n in WRAP:
        f = getattr(I, n, None)
        if not callable(f):
            continue

        def w(*x, _f=f, _n=n, **k):
            key = _n
            if _n == "infllmv2_attn_varlen_func":
                t = k.get("topk_idx")
                key = _n + (":dense" if t is None else ":sparse")
                if t is not None and "topk_idx" not in seen:
                    seen["topk_idx"] = dict(shape=list(t.shape), dtype=str(t.dtype), q_shape=list(x[0].shape), k_shape=list(x[1].shape),
                                            n_neg1=int((t < 0).sum()), max=int(t.max()))
            calls[key] = calls.get(key, 0) + 1
            return _f(*x, **k)
        setattr(I, n, w)


def shim_check(torch) -> dict:
    """flash_attn 轉接層（dense＝infllm_v2 的 varlen、topk_idx=None）對 torch SDPA：GQA 32/2、causal，Sq＝Sk 與 Sq<Sk（右下對齊）。"""
    import torch.nn.functional as F
    from flash_attn import flash_attn_func
    g = torch.Generator(device="cuda").manual_seed(0)
    out = {}
    for name, sq, sk in (("square", 1000, 1000), ("bottom_right", 100, 1000)):
        q = torch.randn(1, sq, 32, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        k = torch.randn(1, sk, 2, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        v = torch.randn(1, sk, 2, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        o = flash_attn_func(q, k, v, causal=True).float()
        mask = torch.ones(sq, sk, dtype=torch.bool, device="cuda").tril(diagonal=sk - sq)
        ref = F.scaled_dot_product_attention(q.transpose(1, 2).float(), k.transpose(1, 2).float().repeat_interleave(16, 1),
                                             v.transpose(1, 2).float().repeat_interleave(16, 1), attn_mask=mask).transpose(1, 2)
        err = (o - ref).abs()
        out[name] = dict(max_abs=round(float(err.max()), 5), rel_l2=round(float(err.norm() / ref.norm()), 6))
    out["ok"] = all(out[n]["rel_l2"] < 1e-2 and out[n]["max_abs"] < 5e-2 for n in ("square", "bottom_right"))
    return out


def cmd_smoke(a):
    out = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"), run_id=os.environ.get("RUN_ID", "adhoc"), sparse_config=SPARSE_CONFIG)

    def done(code=0, why=""):
        if why:
            out["stop_reason"] = why
        print(json.dumps(out, indent=1, default=str), flush=True)
        json.dump(out, open(a.out, "w"), indent=1, default=str)
        if code:
            raise SystemExit(f"D0 冒煙：{why}（規則 2：停）")
    calls, seen = {}, {}
    try:
        import infllm_v2 as I
        out["infllm_v2_module"] = I.__file__
        wrap_infllm(I, calls, seen)
        import flash_attn
        out["flash_attn_module"] = flash_attn.__file__
        out["flash_attn_version"] = flash_attn.__version__
        import transformers
        out["transformers"] = dict(version=transformers.__version__, file=transformers.__file__)
    except Exception as ex:                                       # noqa: BLE001
        out["import"] = f"FAILED: {ex!r}"
        done(2, "import 失敗")
    if "infllmv2shim" not in out["flash_attn_version"]:
        done(2, f"flash_attn 不是 D0 的轉接層：{out['flash_attn_module']}")
    if out["transformers"]["version"] != TRANSFORMERS_PIN:
        done(2, f"transformers {out['transformers']['version']} ≠ {TRANSFORMERS_PIN}（pylibs 沒有排在 PYTHONPATH 前面？）")
    import torch
    out["shim_check"] = shim_check(torch)
    if not out["shim_check"]["ok"]:
        done(2, "flash_attn 轉接層和 SDPA 對不上")
    calls.clear()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    path = os.environ["M7_MINICPM_GLOB"]
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    out["config_sparse_config_before"] = getattr(cfg, "sparse_config", None)
    cfg.sparse_config = dict(SPARSE_CONFIG)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(path, config=cfg, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                                 attn_implementation="flash_attention_2").cuda().eval()
    out["t_load_s"] = round(time.time() - t0, 1)
    out["attn_class"] = sorted({type(l.self_attn).__name__ for l in model.model.layers})
    out["max_position_embeddings"] = getattr(model.config, "max_position_embeddings", None)
    mod = sys.modules[type(model).__module__]
    out["modeling_file"] = mod.__file__
    out["modeling_sha256"] = hashlib.sha256(open(mod.__file__, "rb").read()).hexdigest()
    if out["attn_class"] != ["MiniCPMInfLLMv2Attention"]:
        done(2, f"注意力類別是 {out['attn_class']}，不是 InfLLM-v2")
    import m10_tasks as T
    it = T.make_item("T2", "front", "back", 71000, tok, None, n_blocks=64)
    ids = torch.tensor([it["context_ids"] + it["turns"][0]["query_ids"]], device="cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        o = model(ids, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    out["t_forward_s"] = round(time.time() - t0, 2)
    out["n_tokens"] = ids.shape[1]
    out["logits_finite"] = bool(torch.isfinite(o.logits[0, -1]).all())
    out["infllm_v2_calls"] = dict(calls)
    out["first_sparse_topk_idx"] = seen.get("topk_idx")
    out["sparse_path_used"] = calls.get("infllmv2_attn_stage1", 0) > 0 and calls.get("infllmv2_attn_varlen_func:sparse", 0) > 0
    out["gpu_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
    if not (out["sparse_path_used"] and out["logits_finite"]):
        done(2, "稀疏路徑沒有被呼叫，或 logits 不是有限值")
    done()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("smoke")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    {"smoke": cmd_smoke}[a.cmd](a)
