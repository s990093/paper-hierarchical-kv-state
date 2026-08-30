#!/usr/bin/env python3
"""比對「論文/計畫書宣稱的模型參數」與「實際下載到的 config.json」。

EXPERIMENT_PLAN.md §2 的表格宣稱了每個模型的層數、KV head 數、KV/token、原生 ctx，
並據此算出容量懸崖與 κ。**那些是算出來的，不是量出來的。**
這支腳本做的是最基本的一步：確認算術的**輸入**沒錯。

KV footprint 公式（每 token，單位 bytes）:

    kv_bytes_per_token = 2 × L × H_kv × d_head × bytes_per_elem
                          ↑   ↑     ↑       ↑
                          |   |     |       └─ hidden_size / num_attention_heads
                          |   |     └───────── num_key_value_heads (GQA)
                          |   └─────────────── num_hidden_layers
                          └─────────────────── K 與 V 各一份

⚠️ 前導的 2 就是 OPEN_ISSUES.md B1 說的「K 與 V 綁在同一個動作」——
   本文把兩者視為一體，這是已知且刻意的簡化。

用法:
    python code/verify_model_config.py                       # 跑預設清單
    python code/verify_model_config.py --json results/m1_capacity/model_configs.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# (別名, HF repo id, 計畫書宣稱值)
# 宣稱值來自 EXPERIMENT_PLAN.md §2 的表格。欄位: layers, kv_heads, kv_kib_per_token, native_ctx
CLAIMS = {
    "qwen2.5-7b-1m": (
        "Qwen/Qwen2.5-7B-Instruct-1M",
        {"layers": 28, "kv_heads": 4, "kv_kib_per_token": 56, "native_ctx": 1_010_000},
    ),
    "llama-3.1-8b": (
        "NousResearch/Meta-Llama-3.1-8B-Instruct",  # meta-llama 是 gated，用無門檻鏡像
        {"layers": 32, "kv_heads": 8, "kv_kib_per_token": 128, "native_ctx": 131_072},
    ),
}

VRAM_GIB = 24.0  # RTX 3090


def load_cfg(repo: str) -> dict:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(repo, trust_remote_code=False)
    return cfg.to_dict()


def analyse(repo: str, claim: dict) -> dict:
    cfg = load_cfg(repo)
    L = cfg["num_hidden_layers"]
    n_heads = cfg["num_attention_heads"]
    h_kv = cfg.get("num_key_value_heads", n_heads)
    d_head = cfg.get("head_dim") or cfg["hidden_size"] // n_heads
    dtype = str(cfg.get("torch_dtype") or cfg.get("dtype") or "unknown")
    native_ctx = cfg.get("max_position_embeddings")
    rope = cfg.get("rope_scaling")

    bytes_per_elem = 2  # BF16/FP16 都是 2 bytes
    kv_b_per_tok = 2 * L * h_kv * d_head * bytes_per_elem
    kv_kib = kv_b_per_tok / 1024

    # 論文 §2.5 風格的算術：整張卡都給 KV 時能放幾個 token
    kv_only_tokens = int(VRAM_GIB * 2**30 / kv_b_per_tok)

    # κ = 每 GiB 能裝的 token 數 / 1000（計畫書表格的那一欄）
    checks = {
        "layers": (L, claim["layers"]),
        "kv_heads": (h_kv, claim["kv_heads"]),
        "kv_kib_per_token": (round(kv_kib, 3), claim["kv_kib_per_token"]),
        "native_ctx": (native_ctx, claim["native_ctx"]),
    }
    mismatches = {
        k: {"actual": a, "claimed": c}
        for k, (a, c) in checks.items()
        if (abs(a - c) > 1e-6 if isinstance(a, (int, float)) and isinstance(c, (int, float)) else a != c)
    }

    return {
        "repo": repo,
        "dtype": dtype,
        "num_hidden_layers": L,
        "num_attention_heads": n_heads,
        "num_key_value_heads": h_kv,
        "head_dim": d_head,
        "hidden_size": cfg.get("hidden_size"),
        "max_position_embeddings": native_ctx,
        "rope_scaling": rope,
        "kv_bytes_per_token": kv_b_per_tok,
        "kv_kib_per_token": round(kv_kib, 3),
        "kv_only_token_capacity_24gib": kv_only_tokens,
        "claimed": claim,
        "mismatches": mismatches,
        "ok": not mismatches,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/m1_capacity/model_configs.json")
    ap.add_argument("--models", nargs="*", default=list(CLAIMS))
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/ssd7/hungwei/paper-hkv/hf-cache/huggingface")

    out: dict = {}
    rc = 0
    for alias in args.models:
        repo, claim = CLAIMS[alias]
        try:
            r = analyse(repo, claim)
        except Exception as e:  # noqa: BLE001
            r = {"repo": repo, "error": f"{type(e).__name__}: {e}", "ok": False}
            rc = 1
        out[alias] = r

        print(f"\n=== {alias}  ({r['repo']}) ===")
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue
        print(f"  dtype                 : {r['dtype']}")
        print(f"  layers / kv_heads     : {r['num_hidden_layers']} / {r['num_key_value_heads']}"
              f"  (attn heads {r['num_attention_heads']}, head_dim {r['head_dim']})")
        print(f"  KV per token          : {r['kv_kib_per_token']} KiB"
              f"   (claimed {claim['kv_kib_per_token']})")
        print(f"  native ctx            : {r['max_position_embeddings']:,}"
              f"   (claimed {claim['native_ctx']:,})")
        print(f"  rope_scaling          : {r['rope_scaling']}")
        print(f"  24 GiB 全給 KV 可放    : {r['kv_only_token_capacity_24gib']:,} tokens"
              "  ← 上界，未扣權重與 activation")
        if r["mismatches"]:
            rc = 1
            print("  🔴 MISMATCH vs EXPERIMENT_PLAN.md:")
            for k, v in r["mismatches"].items():
                print(f"     {k}: actual={v['actual']}  claimed={v['claimed']}")
        else:
            print("  ✅ 與計畫書宣稱值一致")

    p = Path(args.json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {p}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
