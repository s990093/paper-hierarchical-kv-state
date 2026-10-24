"""m10_ruler_cwe_vllm.py — T5（cwe）那個 ❌（cards/V2 第 23 列）的下一步：同一份上游 RULER jsonl，改用 vLLM 0.28 跑。

背景：上游 RULER 的 cwe 題（Llama-3.1-8B、32K、30 題）在我們的還原器上只有 1.0%（開頭兩個 BOS）／6.0%（一個 BOS），
HF 原生 generate 前 10 題和還原器逐字相同（m10_ruler_cwe_check.py）。RULER 原作用 vLLM 跑。
這支只換推論框架：同一份 jsonl（input＋answer_prefix）、greedy、最多 120 token、string_match_all（上游 evaluate 相同）。
兩組 token：up_raw＝tokenizer 預設（add_special_tokens=True，模板已有 <|begin_of_text|> → 兩個 BOS），
          up_nobos＝add_special_tokens=False（一個 BOS）。只跑 BF16（E0），不碰低精度。

在 vLLM 的 venv 跑：/mlsteam/workspace/venv/tiara-v028/bin/python code/m10_ruler_cwe_vllm.py --jsonl … --out <csv>
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from datetime import datetime

LLAMA = os.environ.get("M7_LLAMA_GLOB") or "/mlsteam/data/tiara/hf-cache/hub/models--unsloth--Llama-3.1-8B-Instruct/snapshots/*"   # H200：env_h200.sh 設（10/17 加，沒設時不變）


def string_match_all(pred, refs):
    return sum(1.0 if r.lower() in pred.lower() else 0.0 for r in refs) / len(refs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    path = sorted(glob.glob(LLAMA))[-1]
    tok = AutoTokenizer.from_pretrained(path)
    items = [json.loads(x) for x in open(a.jsonl)]
    llm = LLM(model=path, dtype="bfloat16", enable_prefix_caching=False, max_model_len=33792, gpu_memory_utilization=0.80,
              seed=0)
    sp = SamplingParams(max_tokens=120, temperature=0.0)
    rid = os.environ.get("RUN_ID", "adhoc")
    rows, summ = [], {}
    for var, special in (("up_raw", True), ("up_nobos", False)):
        prompts, nbos = [], []
        for it in items:
            ids = tok(it["input"] + it["answer_prefix"], add_special_tokens=special)["input_ids"]
            nbos.append(sum(1 for t in ids[:3] if t == tok.bos_token_id))
            prompts.append(TokensPrompt(prompt_token_ids=ids))
        outs = llm.generate(prompts, sp, use_tqdm=False)
        sc = []
        for it, o, nb in zip(items, outs, nbos):
            pred = o.outputs[0].text
            s = string_match_all(pred, it["outputs"])
            sc.append(s)
            rows.append(dict(run_id=rid, ts=datetime.now().astimezone().isoformat(timespec="seconds"), variant=var,
                             index=it["index"], n_prompt=len(o.prompt_token_ids), n_bos=nb, n_gen=len(o.outputs[0].token_ids),
                             score=s, pred=pred, gold=json.dumps(it["outputs"])))
        summ[var] = {"n": len(sc), "mean": sum(sc) / len(sc)}
        print(var, summ[var], flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump({"run_id": rid, "vllm": vllm.__version__, "model_path": path, "jsonl": a.jsonl, "summary": summ},
              open(os.path.splitext(a.out)[0] + ".summary.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
