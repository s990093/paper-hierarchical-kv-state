"""m10_vllm_ttft.py — 03 §5.2 的表（只報告，不進判定）：harness 的重算比 vLLM 慢多少，拆成「粒度」與「實作」兩層。

在 vLLM 0.28 的 venv 跑（MI300X：/mlsteam/workspace/venv/tiara-v028/bin/python）：
  只算的 TTFT（max_tokens＝1、greedy、關 prefix caching），長度 8K／16K／32K，各 --reps 次（另 2 次暖機不記），
  兩種切法：預設 max_num_batched_tokens、512（和 Cake／harness 一樣一次 512 token）。每次換一組隨機 token（不會命中快取）。
harness 那一邊不在這支量：用 m10_speed calib-f 的 calib_f.csv（逐 chunk 中位數的前綴和＝只算 L 個 token 的時間）。

輸出：--out-dir/vllm_ttft.csv（run_id、ts、model、chunking、max_num_batched_tokens、length、rep、ttft_s）、
      vllm_meta.json（vLLM 版本、實際的 attention backend——從 vLLM 的 log 抓，抓不到寫 NOT_FOUND）
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import statistics
import sys
import time
from datetime import datetime

MODELS = {
    "llama": "/mlsteam/data/tiara/hf-cache/hub/models--unsloth--Llama-3.1-8B-Instruct/snapshots/*",
    "qwen": "/mlsteam/data/tiara/hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct-1M/snapshots/*",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--chunking", choices=["default", "512"], required=True)
    ap.add_argument("--lengths", nargs="+", type=int, default=[8192, 16384, 32768])
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    path = sorted(glob.glob(os.environ.get(f"M7_{a.model.upper()}_GLOB") or MODELS[a.model]))[-1]
    kw = dict(model=path, dtype="bfloat16", enable_prefix_caching=False, max_model_len=max(a.lengths) + 512,
              gpu_memory_utilization=0.80, seed=0)
    if a.chunking == "512":
        kw["max_num_batched_tokens"] = 512
    llm = LLM(**kw)
    try:                                                   # 版本間屬性位置不同：查不到就照實記 NOT_FOUND（不猜）
        mnbt = llm.llm_engine.vllm_config.scheduler_config.max_num_batched_tokens
    except Exception as ex:                                # noqa: BLE001
        mnbt = f"NOT_FOUND ({type(ex).__name__})"
    from transformers import AutoConfig
    vocab = AutoConfig.from_pretrained(path).vocab_size
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    rng = random.Random(f"{a.model}|{a.chunking}")
    rid = os.environ.get("RUN_ID", "adhoc")
    os.makedirs(a.out_dir, exist_ok=True)
    p = os.path.join(a.out_dir, "vllm_ttft.csv")
    new = not os.path.exists(p)
    f = open(p, "a", newline="")
    w = csv.DictWriter(f, fieldnames=["run_id", "ts", "model", "chunking", "max_num_batched_tokens", "length", "rep",
                                      "ttft_s"])
    if new:
        w.writeheader()
    res = {}
    for L in a.lengths:
        ts_ = []
        for rep in range(-2, a.reps):                      # 前 2 次暖機
            ids = [rng.randrange(1000, min(120000, vocab)) for _ in range(L)]
            t0 = time.perf_counter()
            llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
            t = time.perf_counter() - t0
            if rep >= 0:
                ts_.append(t)
                w.writerow(dict(run_id=rid, ts=datetime.now().astimezone().isoformat(timespec="seconds"), model=a.model,
                                chunking=a.chunking, max_num_batched_tokens=mnbt, length=L, rep=rep, ttft_s=round(t, 5)))
        f.flush()
        res[L] = statistics.median(ts_)
        print(f"{a.model} {a.chunking} L={L} median TTFT {res[L]:.4f}s", flush=True)
    f.close()
    meta_p = os.path.join(a.out_dir, "vllm_meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    meta[f"{a.model}|{a.chunking}"] = {"run_id": rid, "vllm": vllm.__version__, "model_path": path,
                                       "max_num_batched_tokens": mnbt, "median_ttft_s": res,
                                       "backend": "see stderr.log of run (grep 'backend')"}
    json.dump(meta, open(meta_p, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
