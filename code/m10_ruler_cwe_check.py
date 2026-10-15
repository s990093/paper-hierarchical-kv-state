"""m10_ruler_cwe_check.py — T5（cwe）的 BF16 正確率：上游 RULER 出題 vs m10_tasks 出題（Llama-3.1-8B、32K、只跑 E0）

為什麼（cards/V2_numbers_vs_papers_20261015.md 第 23 列）：m10_tasks T5（freq_cw 30）上 Llama 只有 47.7%，
30 題裡 12 題答成「the、and、a…」。Double-P 報 cwe 95.17，RULER 原作只報 CWE＋FWE 平均 54.1。
要分清楚：48% 是我們出題的差異，還是這個模型在 32K 本來就這樣。

做法
  gen  直接呼叫上游 scripts/data/synthetic/common_words_extraction.py（RULER commit c3f5e3b），參數和 prepare.py
       對 `cwe` 組出來的一樣：
         template ＝ Templates['meta-llama3'].format(task_template=TASKS['common_words_extraction']['template'])
                    ＋ answer_prefix（prepare.py：「Add answer prefix for all models」）
         --tokenizer_type hf --max_seq_length 32768 --tokens_to_generate 120 --num_samples 30 --random_seed 42
         --freq_cw 30 --freq_ucw 3 --num_cw 10（synthetic.yaml 的 cwe）；num_fewshot 用預設 1
       沒有經過 prepare.py：它一開頭 import nltk（venv 沒裝），cwe 用不到 nltk，指令照它組出來的樣子直接下。
       wonderwords、tenacity 用 --shim 目錄（只放這兩個套件；pylibs 裡另有 numpy，整包放進 PYTHONPATH 會蓋掉 venv 的）。
       config_models.sh：llama3.1-8b-chat 的 MODEL_TEMPLATE_TYPE＝meta-llama3。
       ⚠️ MLSteam 上的 RULER clone 沒拉 Git LFS：synthetic/json/english_words.json 只是 132 B 的指標檔，上游產生器一開頭
       就讀它（只有需要的不同詞數 > 8,166 個 wonderwords 時才用到，32K 用不到），直接跑會 JSONDecodeError
       （run 20261015-054607-m10-ruler-cwe-llama）。做法：把 scripts/data 複製一份（--ruler 指到副本），只把這個檔
       換成從 GitHub LFS 下載的真檔（media.githubusercontent.com/media/NVIDIA/RULER/c3f5e3b…/…/english_words.json，
       sha256 affcd6d4…2eca、8,564,991 B，和指標檔記的一樣）；`diff -rq` 確認其餘檔案和原 clone 相同。
  run  同一個 process、同一個模型，三組：
         up_raw    上游的 input ＋ answer_prefix（pred/call_api.py:305），tokenizer 預設 add_special_tokens=True
                   （pred/model_wrappers.py 的 HF 路徑：self.tokenizer(prompts, ...)）。模板本身已有 <|begin_of_text|>，
                   所以開頭會有兩個 BOS——照上游原樣跑，並記下 BOS 數。
         up_nobos  同一份文字，add_special_tokens=False（只有模板裡的一個 BOS）
         m10       m10_tasks T5 預設參數、seed 1000–1029（和 p1_default_llama 同一批；結果應該和那一輪逐題相同）
       生成：greedy、最多 120 token（上游 tokens_to_generate）、遇到 EOS 停。計分：string_match_all（上游 evaluate 相同）。
       上游 prompt 長度不是 512 的倍數：整段的 512 用 prefill_chunked，尾巴用 Answerer.feed 接著餵。
只看 BF16 的答對率（E0），不碰任何低精度。

  hf   （10/15 加，看完上面三組之後）上游 prompt 在我們的還原器上只有 1–6%，比我們自己的出題還低。
       為了分清楚是還原器（chunk 化 prefill、自己的 attention 迴圈）的問題，還是模型對 prompt 的反應：
       拿同一份上游 jsonl 的前 --hf-check 題，改用 HF 原生 model.generate（整段 prefill、SDPA、greedy、120 token），
       tokenizer 同 up_raw（預設加 BOS）。只跑 BF16。輸出 <out>_hfgen.csv。

用法：python code/m10_ruler_cwe_check.py --ruler <RULER 根目錄> --shim <shim 目錄> --raw <放上游 jsonl 的目錄> --out <csv>
      python code/m10_ruler_cwe_check.py --hf-check 10 --reuse-jsonl <上一次的 validation.jsonl> --out <csv>
"""
from __future__ import annotations

import argparse
import collections
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m10_quality as MQ  # noqa: E402
import m10_tasks as T  # noqa: E402

N, SEED, LEN, GEN = 30, 42, 32768, 120
GENERIC = {"the", "and", "a", "an", "to", "of", "in", "that", "is", "for", "with", "it", "on", "as", "was", "be",
           "by", "at", "this", "i", "you", "he", "she", "they", "we", "or", "not", "but", "from", "have"}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def upstream_template(ruler):
    tmpl = _load(os.path.join(ruler, "scripts/data/template.py"), "ruler_template").Templates
    tasks = _load(os.path.join(ruler, "scripts/data/synthetic/constants.py"), "ruler_constants").TASKS
    cfg = tasks["common_words_extraction"]
    return tmpl["meta-llama3"].format(task_template=cfg["template"]) + cfg["answer_prefix"]


def gen_upstream(a, llama_path):
    template = upstream_template(a.ruler)
    script = os.path.join(a.ruler, "scripts/data/synthetic/common_words_extraction.py")
    cmd = [sys.executable, script, "--save_dir", a.raw, "--save_name", "cwe", "--subset", "validation",
           "--tokenizer_path", llama_path, "--tokenizer_type", "hf", "--max_seq_length", str(LEN),
           "--tokens_to_generate", str(GEN), "--num_samples", str(N), "--random_seed", str(SEED),
           "--freq_cw", "30", "--freq_ucw", "3", "--num_cw", "10", "--template", template]
    env = dict(os.environ, PYTHONPATH=a.shim)
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    open(os.path.join(a.raw, "gen_stderr.log"), "w").write(r.stderr)
    open(os.path.join(a.raw, "gen_cmd.json"), "w").write(json.dumps(cmd, ensure_ascii=False, indent=1))
    if r.returncode != 0:
        raise RuntimeError(f"上游產生器失敗（rc={r.returncode}）：{r.stderr[-2000:]}")
    path = os.path.join(a.raw, "cwe", "validation.jsonl")
    rows = [json.loads(x) for x in open(path)]
    assert len(rows) == N, len(rows)
    print(f"上游產生 {N} 題，{time.time() - t0:.0f}s：{path}", flush=True)
    return rows, template


def list_stats(text, gold):
    """最後一份清單（few-shot 範例之後）的統計：項目數、不同詞數、每個答案詞出現幾次。"""
    head = "Memorize the ones that appear most often.\n"
    s = text.rfind(head)
    e = text.find("\nQuestion:", s)
    body = text[s + len(head):e if e > 0 else None]
    items = re.findall(r"(?:^|\s)\d+\. (\S+)", body)
    c = collections.Counter(items)
    gc = [c.get(g, 0) for g in gold]
    return dict(n_list_items=len(items), n_distinct=len(c), gold_count_min=min(gc), gold_count_max=max(gc),
                top_nongold_count=max([v for k, v in c.items() if k not in set(gold)] or [0]))


def generic_hits(pred):
    ws = [w.lower() for w in re.findall(r"\d+\.\s*\**\s*([A-Za-z][A-Za-z'\-]*)", pred)]
    return sum(w in GENERIC for w in ws)


def answer_ids(ans, ids):
    n = len(ids)
    nfull = (n // 512) * 512
    if nfull == n:
        nfull -= 512
    t = torch.tensor(ids[:nfull], device=ans.km.device)
    ans.km.prefill_chunked(t)
    logits = ans.feed(ids[nfull:], nfull)
    return ans.generate(logits, n, GEN)


def hf_check(a):
    """同一份上游 prompt，用 HF 原生 generate（不經過還原器）。"""
    rid = os.environ.get("RUN_ID", "adhoc")
    up = [json.loads(x) for x in open(a.reuse_jsonl)][:a.hf_check]
    ans = MQ.Answerer("llama")
    tok, hf = ans.tok, ans.km.hf
    rows = []
    for d in up:
        prompt = d["input"] + d.get("answer_prefix", "")
        ids = tok(prompt, return_tensors="pt")["input_ids"].to(ans.km.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = hf.generate(ids, max_new_tokens=GEN, do_sample=False, temperature=None, top_p=None,
                              eos_token_id=sorted(ans.eos), pad_token_id=tok.eos_token_id)
        gen = out[0, ids.shape[1]:].tolist()
        pred = tok.decode(gen, skip_special_tokens=True)
        rows.append(dict(run_id=rid, ts=datetime.now().astimezone().isoformat(timespec="seconds"), source="up_raw_hfgen",
                         idx=d["index"], n_prompt_tokens=ids.shape[1], score=T.score("string_match_all", pred, d["outputs"]),
                         n_gen=len(gen), generic_hits=generic_hits(pred), t_s=round(time.perf_counter() - t0, 2),
                         pred=pred, gold=json.dumps(d["outputs"])))
        print(rows[-1]["idx"], rows[-1]["score"], repr(pred[:100]), flush=True)
    path = os.path.splitext(a.out)[0] + "_hfgen.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"up_raw_hfgen: {sum(r['score'] for r in rows) / len(rows):.3f}（{len(rows)} 題）→ {path}", flush=True)


def main(a):
    if a.hf_check:
        return hf_check(a)
    rid = os.environ.get("RUN_ID", "adhoc")
    os.makedirs(a.raw, exist_ok=True)
    llama_path = sorted(__import__("glob").glob(T.MODELS["llama"]))[-1]
    up, template = gen_upstream(a, llama_path)
    ans = MQ.Answerer("llama")
    tok = ans.tok
    out_rows = []

    def emit(**kw):
        kw.update(run_id=rid, ts=datetime.now().astimezone().isoformat(timespec="seconds"))
        out_rows.append(kw)

    bos = tok.convert_tokens_to_ids("<|begin_of_text|>")
    for mode in ("up_raw", "up_nobos"):
        for d in up:
            prompt = d["input"] + d.get("answer_prefix", "")
            ids = tok(prompt, add_special_tokens=(mode == "up_raw"))["input_ids"]
            nb = 0
            while nb < len(ids) and ids[nb] == bos:
                nb += 1
            t0 = time.perf_counter()
            gen, stop = answer_ids(ans, ids)
            pred = ans.decode(gen)
            gold = d["outputs"]
            emit(source=mode, idx=d["index"], seed=SEED, n_prompt_tokens=len(ids), n_bos_prefix=nb,
                 n_bos_total=ids.count(bos), **list_stats(d["input"], gold),
                 score=T.score("string_match_all", pred, gold), n_gen=len(gen), stop=stop,
                 generic_hits=generic_hits(pred), t_s=round(time.perf_counter() - t0, 2),
                 pred=pred, gold=json.dumps(gold))
        sc = [r["score"] for r in out_rows if r["source"] == mode]
        print(f"{mode}: {sum(sc) / len(sc):.3f}", flush=True)
    for k in range(N):
        seed = 1000 + k
        it = T.make_item("T5", None, None, seed, tok, {})
        t0 = time.perf_counter()
        r = MQ.run_item_e0(ans, it)[0]
        full = tok.decode(MQ.T.full_prompt_ids(it, 1), skip_special_tokens=False)
        ids = MQ.T.full_prompt_ids(it, 1)
        emit(source="m10", idx=k, seed=seed, n_prompt_tokens=len(ids), n_bos_prefix=int(ids[0] == bos),
             n_bos_total=ids.count(bos), **list_stats(full, it["turns"][0]["gold"]),
             score=float(r["score_up"]), n_gen=r["n_gen"], stop=r["stop"], generic_hits=generic_hits(r["pred"]),
             t_s=round(time.perf_counter() - t0, 2), pred=r["pred"], gold=r["gold"])
    sc = [r["score"] for r in out_rows if r["source"] == "m10"]
    print(f"m10: {sum(sc) / len(sc):.3f}", flush=True)

    fields = ["run_id", "ts", "source", "idx", "seed", "n_prompt_tokens", "n_bos_prefix", "n_bos_total",
              "n_list_items", "n_distinct", "gold_count_min", "gold_count_max", "top_nongold_count",
              "score", "n_gen", "stop", "generic_hits", "t_s", "pred", "gold"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    summ = {"run_id": rid, "ruler_commit": "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a", "model": llama_path,
            "upstream_template": template, "n": N, "max_seq_length": LEN, "tokens_to_generate": GEN,
            "by_source": {}}
    for s in ("up_raw", "up_nobos", "m10"):
        v = [r for r in out_rows if r["source"] == s]
        summ["by_source"][s] = {
            "acc": round(sum(r["score"] for r in v) / len(v), 4),
            "n_zero": sum(r["score"] == 0 for r in v), "n_full": sum(r["score"] == 1 for r in v),
            "n_generic_answer": sum(r["generic_hits"] >= 3 for r in v),
            "median_prompt_tokens": sorted(r["n_prompt_tokens"] for r in v)[len(v) // 2],
            "median_list_items": sorted(r["n_list_items"] for r in v)[len(v) // 2],
            "median_distinct": sorted(r["n_distinct"] for r in v)[len(v) // 2],
            "gold_count_min_median": sorted(r["gold_count_min"] for r in v)[len(v) // 2],
            "bos_prefix": sorted({r["n_bos_prefix"] for r in v}), "bos_total": sorted({r["n_bos_total"] for r in v}),
        }
    # m10 這一組應該和 p1_default_llama 的 T5 逐題相同（同 seed、同程式）
    p1 = os.path.join(os.path.dirname(a.out), "p1_default_llama.csv")
    if os.path.exists(p1):
        old = {int(r["seed"]): r for r in csv.DictReader(open(p1)) if r["task"] == "T5"}
        same = [old[r["seed"]]["pred"] == r["pred"] for r in out_rows if r["source"] == "m10" and r["seed"] in old]
        summ["m10_matches_p1_default_llama"] = f"{sum(same)}/{len(same)} 題的輸出逐字相同"
    json.dump(summ, open(os.path.splitext(a.out)[0] + ".summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summ["by_source"], ensure_ascii=False, indent=1))
    print(summ.get("m10_matches_p1_default_llama"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ruler")
    ap.add_argument("--shim")
    ap.add_argument("--raw")
    ap.add_argument("--hf-check", type=int, default=0, help="只跑 HF 原生 generate 的前 N 題（需要 --reuse-jsonl）")
    ap.add_argument("--reuse-jsonl")
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
