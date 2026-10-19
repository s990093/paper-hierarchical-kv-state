"""m10_vllm_kvdecode.py — 13 號計畫 §6 (d)（§8b 改成正式要跑）：業界現成的「拆在 attention 裡」——vLLM 的 FP8 KV。只報告。

vLLM 0.28（ROCm）`--kv-cache-dtype fp8`：KV 存 FP8（MI300X 上是 fnuz 版、vLLM 自己的 scale），attention kernel 讀 FP8、在 kernel 裡拆；
`auto`＝BF16。這不是我們的 codec，只當「真實系統讀壓縮 KV 每個 token 要多久」的參考。
量法：同一個 vLLM 實例，每次換一組隨機 token（關 prefix caching、不會命中快取）；
  t1＝max_tokens 1（prefill＋第一個 token）、t65＝max_tokens 65（ignore_eos），各 --reps 次（另 1 次暖機不記）；
  每個 token 的 decode ≈ (median t65 − median t1) ÷ 64（prefill 的變動會進來，所以兩個都報）。
在 vLLM 的 venv 跑：/mlsteam/workspace/venv/tiara-v028/bin/python code/m10_vllm_kvdecode.py --model llama --kv-dtype fp8 --out-dir D
輸出：D/vllm_kvdecode.csv（每次一列）、D/vllm_kvdecode_meta.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import statistics
import time
from datetime import datetime

MODELS = {
    "llama": "/mlsteam/data/tiara/hf-cache/hub/models--unsloth--Llama-3.1-8B-Instruct/snapshots/*",
    # 10/18：vLLM 0.28 的 ROCm attention 不支援 DCA（RocmAttentionImpl 收不到 layer_idx → 43e 的 (d) Qwen 啟動失敗）。
    # 改用同一份權重（symlink 到 HF cache）、只拿掉 dual_chunk_attention_config 的 noDCA 目錄（CLAUDE.md §3 第 3 點；
    # 我們的 HF harness 本來就沒有 DCA，所以兩邊是同一個模型）。
    "qwen": "/mlsteam/data/tiara/models/Qwen2.5-7B-Instruct-1M-noDCA",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--kv-dtype", choices=["auto", "fp8"], required=True)
    ap.add_argument("--lengths", nargs="+", type=int, default=[32768, 129024])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    path = sorted(glob.glob(os.environ.get(f"M7_{a.model.upper()}_GLOB") or MODELS[a.model]))[-1]
    llm = LLM(model=path, dtype="bfloat16", kv_cache_dtype=a.kv_dtype, enable_prefix_caching=False,
              max_model_len=max(a.lengths) + 512, gpu_memory_utilization=0.80, seed=0)
    from transformers import AutoConfig
    vocab = AutoConfig.from_pretrained(path).vocab_size
    sps = {1: SamplingParams(max_tokens=1, temperature=0.0),
           65: SamplingParams(max_tokens=65, min_tokens=65, ignore_eos=True, temperature=0.0)}
    rng = random.Random(f"{a.model}|{a.kv_dtype}")
    rid = os.environ.get("RUN_ID", "adhoc")
    os.makedirs(a.out_dir, exist_ok=True)
    p = os.path.join(a.out_dir, "vllm_kvdecode.csv")
    new = not os.path.exists(p)
    f = open(p, "a", newline="")
    w = csv.DictWriter(f, fieldnames=["run_id", "ts", "model", "kv_dtype", "length", "max_tokens", "rep", "n_out", "t_s"])
    if new:
        w.writeheader()
    res = {}
    for L in a.lengths:
        t = {1: [], 65: []}
        for rep in range(-1, a.reps):                      # 第 1 次暖機
            for mt in (1, 65):
                ids = [rng.randrange(1000, min(120000, vocab)) for _ in range(L)]
                t0 = time.perf_counter()
                out = llm.generate([TokensPrompt(prompt_token_ids=ids)], sps[mt], use_tqdm=False)
                dt = time.perf_counter() - t0
                n_out = len(out[0].outputs[0].token_ids)
                if rep >= 0:
                    t[mt].append(dt)
                    w.writerow(dict(run_id=rid, ts=datetime.now().astimezone().isoformat(timespec="seconds"), model=a.model,
                                    kv_dtype=a.kv_dtype, length=L, max_tokens=mt, rep=rep, n_out=n_out, t_s=round(dt, 5)))
        f.flush()
        m1, m65 = statistics.median(t[1]), statistics.median(t[65])
        res[L] = dict(median_t1_s=round(m1, 5), median_t65_s=round(m65, 5), decode_per_token_s=round((m65 - m1) / 64, 6))
        print(f"{a.model} kv={a.kv_dtype} L={L}: {res[L]}", flush=True)
    f.close()
    mp = os.path.join(a.out_dir, "vllm_kvdecode_meta.json")
    meta = json.load(open(mp)) if os.path.exists(mp) else {}
    meta[f"{a.model}|{a.kv_dtype}"] = {"run_id": rid, "vllm": vllm.__version__, "model_path": path, "result": res,
                                      "backend": "見這個 run 的 stderr.log（grep -i backend）"}
    json.dump(meta, open(mp, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
