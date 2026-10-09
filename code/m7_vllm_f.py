"""m7_vllm_f.py — 用 vLLM（fused MoE、正式的 attention kernel）量每個 512-token chunk 的重算時間 f(i)。

為什麼（11_round1_plan.md H2）：HF transformers 的 MoE 實作（grouped_mm／eager）比 vLLM 慢很多，會讓 MoE 的重算看起來太貴。
做法：開 prefix caching。第 i 步送 ids[:(i+1)*512]（前 i*512 已在快取），量 generate(max_tokens=1) 的時間＝算第 i 個 chunk＋固定開銷；
緊接著再送一次同樣的 prompt（全部命中，只算最後 1 個 token），當作固定開銷。f(i) ≈ 兩者相減。
每個 rep 用新的隨機 token，避免跨 rep 命中。用 num_cached_tokens 驗證快取確實命中 i*512 個 token（CLAUDE.md 規則 6 的精神：用另一條路徑驗算）。
同一支程式也量 Llama-3.1-8B，和 harness 的 C1（m7_calib.py）對照，確認這個方法本身可信。

用法：python code/m7_vllm_f.py <model glob> <out csv> [--max-chunks 72] [--reps 3]
"""
import argparse
import csv
import datetime
import glob
import json
import os
import random
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_glob")
    ap.add_argument("out")
    ap.add_argument("--max-chunks", type=int, default=72)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    path = sorted(glob.glob(a.model_glob))[-1]
    vocab = json.load(open(os.path.join(path, "config.json"))).get("vocab_size")
    n_tok = a.max_chunks * 512
    llm = LLM(model=path, dtype="bfloat16", enable_prefix_caching=True, max_model_len=n_tok + 1024,
              gpu_memory_utilization=0.85, seed=0, disable_log_stats=False)
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    rid = os.environ.get("RUN_ID", "no-run-id")
    new = not os.path.exists(a.out)
    f = open(a.out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=["run_id", "ts", "item", "chunk_idx", "pos_end", "rep", "ms", "cached_tokens",
                                      "gpu_state", "engine"])
    if new:
        w.writeheader()

    def gen(ids):
        t0 = time.perf_counter()
        out = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
        m = getattr(out[0], "metrics", None)     # 11 追加 3：引擎內部時間戳（排進 GPU → 第一個 token），不含前端開銷
        eng = (m.first_token_ts - m.scheduled_ts) * 1e3 if m is not None and m.scheduled_ts else float("nan")
        return (time.perf_counter() - t0) * 1e3, getattr(out[0], "num_cached_tokens", None), eng

    # 暖機（不記錄）
    rng = random.Random(12345)
    warm = [rng.randrange(1000, min(120000, vocab)) for _ in range(4 * 512)]
    for i in range(4):
        gen(warm[:(i + 1) * 512])
    for rep in range(a.reps):
        rng = random.Random(rep)
        ids = [rng.randrange(1000, min(120000, vocab)) for _ in range(n_tok)]
        for i in range(a.max_chunks):
            p = ids[:(i + 1) * 512]
            t_inc, c_inc, e_inc = gen(p)   # 算第 i 個 chunk（前 i*512 命中）
            t_ov, c_ov, e_ov = gen(p)      # 全部命中，只算最後一個 block：固定開銷
            ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
            for item, ms, ct in (("raw_incr", t_inc, c_inc), ("overhead", t_ov, c_ov), ("f_chunk", t_inc - t_ov, c_inc),
                                 ("engine_incr", e_inc, c_inc), ("engine_ov", e_ov, c_ov), ("f_engine", e_inc - e_ov, c_inc)):
                w.writerow(dict(run_id=rid, ts=ts, item=item, chunk_idx=i, pos_end=(i + 1) * 512, rep=rep,
                                ms=round(ms, 3), cached_tokens=ct, gpu_state="idle", engine="vllm"))
            f.flush()
            print(json.dumps(dict(rep=rep, i=i, inc=round(t_inc, 2), ov=round(t_ov, 2), e_inc=round(e_inc, 2), e_ov=round(e_ov, 2),
                                  c_inc=c_inc, c_ov=c_ov)), flush=True)


if __name__ == "__main__":
    main()
