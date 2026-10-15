"""m10_longbench.py — 端點的外部效度：LongBench v1 子集（03_plan_h200.md §3.2「外部效度（不進判準）」）

只給端點 E0、E1、E1s、E1sr 用，不進位置判定。產生的題目和 m10_tasks.make_item 同形狀，直接餵 m10_quality。

## 任務（寫死）
narrativeqa、qasper、multifieldqa_en、hotpotqa、2wikimqa、musique：LongBench 的 6 個英文 QA，官方計分都是 qa_f1_score。
（T4 也用 HotpotQA，但那是我們自己重組的兩跳題；這裡的 hotpotqa 是 LongBench 官方的版本，報告要分開講。）

## 選題（可重現；在看任何模型答對率之前寫死）
每個任務：把官方 jsonl 依 `_id` 排序，用 random.Random(f"m10|longbench|{dataset}").sample(range(N), 50) 抽 50 個位置，
再依位置排序。idx 0–49 就是這個順序。字串 seed 走 sha512，跨 Python 版本相同。

## prompt 與截斷（照官方 pred.py）
- prompt＝config/dataset2prompt.json 的模板 .format(**題目)。
- 截斷：`tokenizer(prompt, truncation=False)`（預設會加 special token：Llama 有 BOS，和 pred.py 相同）超過 MAX_LENGTH，
  就保留前一半＋後一半：decode(ids[:half], skip_special_tokens=True) + decode(ids[-half:], ...)，half＝int(MAX_LENGTH/2)。
- MAX_LENGTH＝31,500：官方 config/model2maxlen.json 裡 32K 模型（longchat-v1.5-7b-32k、chatglm2/3-6b-32k）的值。
  加上 chat template（Llama 約 26 個、Qwen 約 20 個 token）＋生成（≤128）之後，整段 ≤ 32,768＋4,096（m10_quality 的 KVModel）。
- 解碼：transformers 5.x 對 BPE tokenizer 忽略 clean_up_tokenization_spaces（會印警告），所以截斷時的 decode 不做
  「" ." → "."」；舊版 transformers 跑官方 pred.py 時 Llama 會做。差別只在被截斷的題目（narrativeqa 22/50），只影響空白。
- chat template：這 6 個任務官方 pred.py 都套 chat（不套的只有 trec、triviaqa、samsum、lsht、lcc、repobench-p）。
  官方 v1 的 build_chat 沒有 Llama 3.1、Qwen2.5 的分支，我們用 tokenizer 自己的 chat template（m10_tasks 的膠水，
  Llama 3.1 日期固定 26 Jul 2024；Qwen 用預設 system prompt），和 T1–T5 相同。
- 生成長度：官方 dataset2maxlen（narrativeqa、qasper 128；multifieldqa_en 64；hotpotqa、2wikimqa、musique 32）放在
  max_new_tokens_upstream；m10_quality 生成到 max(64, 這個值)，score_up 就是官方長度的分數。

## 切塊（我們的做法，不是 LongBench 官方的）
整段＝chat 開頭＋prompt＋user→assistant 膠水。模板在 {context} 之後的那段（結尾的說明＋Question: {input}＋Answer:）
叫「後綴」。context＝「後綴開始之前」最大的 512 倍數；其餘（文件不足一個 block 的尾巴＋後綴＋assistant 開頭）放進
query_ids。這樣：
- context 的每個 block 都能存成 BF16 或低精度；問題和結尾說明一律在 query（BF16），和 T1–T5 相同（問題不會被存成低精度）。
- 後綴的位置用 tokenizer 的 offset mapping 從字元位置換成 token 位置（整段一次 tokenize，不分段，不改官方的切法）。
- hotpotqa 等的說明文字在開頭也有一次，那一份在 block 0（E1s、E1sr 的 block 0 是 BF16）。
context 的長度隨題目變（512 的倍數，≤ 62 個 block），不像 T1–T5 固定 64 個 block。

## 計分
m10_tasks.qa_f1（LongBench 的 qa_f1_score：normalize＝小寫、去標點、去冠詞、壓空白；多個答案取最大）。
自測會用上游 metrics.py 的原始碼（只抽 normalize_answer、f1_score、qa_f1_score，避開它 import jieba）對拍。

用法：
  python -I code/m10_longbench.py --self-test        # CPU：兩個 tokenizer × 6 任務 × 50 題
  python code/m10_longbench.py --e0-smoke --n 5 --out <csv>   # GPU：Llama 只跑全 BF16，每任務 5 題（彩排，不進 results）
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m10_tasks as T  # noqa: E402

DATASETS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique"]
N_PER = 50
LB_DIR = os.environ.get("M10_LONGBENCH", "/mlsteam/data/tiara/datasets/longbench")
MAX_LENGTH = 31500
SELECT_SEED = "m10|longbench|{dataset}"
KV_MAX_LEN = 32768 + 4096           # m10_quality.Answerer 的 KVModel max_len

_ROWS: dict[str, list[dict]] = {}
_CFG: dict[str, dict] = {}


def _cfg(name: str) -> dict:
    if name not in _CFG:
        _CFG[name] = json.load(open(os.path.join(LB_DIR, "config", f"{name}.json")))
    return _CFG[name]


def selected(dataset: str) -> list[dict]:
    """這個任務選中的 50 題（依 _id 排序後用固定 seed 抽；見檔頭）。"""
    if dataset not in DATASETS:
        raise KeyError(f"{dataset} 不在子集裡：{DATASETS}")
    if dataset not in _ROWS:
        rows = [json.loads(l) for l in open(os.path.join(LB_DIR, "data", f"{dataset}.jsonl"), encoding="utf-8")]
        rows.sort(key=lambda r: r["_id"])
        if len({r["_id"] for r in rows}) != len(rows):
            raise RuntimeError(f"{dataset}：_id 有重複，排序不唯一")
        pick = sorted(random.Random(SELECT_SEED.format(dataset=dataset)).sample(range(len(rows)), N_PER))
        _ROWS[dataset] = [rows[i] for i in pick]
    return _ROWS[dataset]


def official_prompt(dataset: str, obj: dict, tok) -> tuple[str, int, bool]:
    """官方 pred.py 的 prompt 與中間截斷。回傳（截斷後的 prompt, 截斷前 token 數, 有沒有截斷）。"""
    prompt = _cfg("dataset2prompt")[dataset].format(**obj)
    ids = tok(prompt, truncation=False)["input_ids"]
    if len(ids) > MAX_LENGTH:
        half = int(MAX_LENGTH / 2)
        prompt = tok.decode(ids[:half], skip_special_tokens=True) + tok.decode(ids[-half:], skip_special_tokens=True)
        return prompt, len(ids), True
    return prompt, len(ids), False


def suffix_char(dataset: str, prompt: str, tok) -> int:
    """後綴（模板 {context} 之後、{input} 之前的固定說明）在 prompt 裡的字元位置。
    截斷時官方會 decode 後半段；Llama 的 tokenizer 會做 clean_up_tokenization（" ." → "."），所以兩種寫法都找。"""
    fixed = _cfg("dataset2prompt")[dataset].split("{context}", 1)[1].split("{input}", 1)[0]
    for f in (fixed, tok.clean_up_tokenization(fixed)):
        i = prompt.rfind(f)
        if i >= 0:
            return i
    raise RuntimeError(f"{dataset}：prompt 裡找不到模板後綴 {fixed[:40]!r}")


def make_item(dataset: str, idx: int, tok, max_ctx: int = 32768, block: int = 512) -> dict:
    obj = selected(dataset)[idx]
    enc = T._enc(tok)
    prompt, n_raw, trunc = official_prompt(dataset, obj, tok)
    o = enc.tok(prompt, add_special_tokens=False, return_offsets_mapping=True)   # 整段一次 tokenize（不分段）
    body = o["input_ids"]
    c = suffix_char(dataset, prompt, tok)
    t_suf = next(t for t, (a, b) in enumerate(o["offset_mapping"]) if b > c)    # 含後綴第一個字元的 token
    full = enc.head + body + enc.u2a
    n_ctx = min(((len(enc.head) + t_suf) // block) * block, max_ctx)
    if n_ctx < block:
        raise RuntimeError(f"{dataset}[{idx}]：整段只有 {len(full)} token，不滿一個 block")
    up = _cfg("dataset2maxlen")[dataset]
    if len(full) + max(64, up) > KV_MAX_LEN:
        raise RuntimeError(f"{dataset}[{idx}]：{len(full)}＋生成 {max(64, up)} 超過 KVModel 的 {KV_MAX_LEN}")
    answers = [a for a in obj["answers"]]
    if not answers or not all(isinstance(a, str) and a.strip() for a in answers):
        raise RuntimeError(f"{dataset}[{idx}]：答案是空的：{answers!r}")
    return dict(
        item_id=f"LB-{dataset}_{idx}", task=f"LB-{dataset}", seed=idx, params={}, model_family=enc.family,
        tokenizer=getattr(tok, "name_or_path", ""), n_blocks=n_ctx // block, block=block,
        max_new_tokens_upstream=up, context_ids=full[:n_ctx],
        turns=[dict(turn=1, band="global", query_ids=full[n_ctx:], gold=answers, gold_text=answers[0],
                    metric="qa_f1", needle_blocks=[], question=obj["input"])],
        lb_id=obj["_id"], length_words=obj["length"], raw_prompt_tokens=n_raw, truncated=trunc,
        suffix_token=len(enc.head) + t_suf,
    )


def all_items(tok, datasets=DATASETS, n=N_PER):
    for d in datasets:
        for i in range(n):
            yield make_item(d, i, tok)


# ── 自測 ───────────────────────────────────────────────────────────────────

def _upstream_qa_f1():
    """從上游 metrics.py 抽出 normalize_answer、f1_score、qa_f1_score 的原始碼執行（不 import 整支：它 import jieba）。"""
    src = open(os.path.join(LB_DIR, "metrics.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name in ("normalize_answer", "f1_score", "qa_f1_score")]
    if len(keep) != 3:
        raise RuntimeError("上游 metrics.py 找不到 normalize_answer／f1_score／qa_f1_score")
    ns = {}
    exec("import re\nimport string\nfrom collections import Counter\n"
         + "\n\n".join(ast.get_source_segment(src, n) for n in keep), ns)
    return ns["qa_f1_score"]


def _ref_trunc(dataset, obj, tok):
    """官方 pred.py 那幾行逐字照抄（return_tensors="pt"），和 official_prompt 對拍。"""
    prompt = _cfg("dataset2prompt")[dataset].format(**obj)
    tokenized_prompt = tok(prompt, truncation=False, return_tensors="pt").input_ids[0]
    max_length = MAX_LENGTH
    if len(tokenized_prompt) > max_length:
        half = int(max_length / 2)
        prompt = tok.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
            tok.decode(tokenized_prompt[-half:], skip_special_tokens=True)
    return prompt


def self_test(models=("llama", "qwen")) -> int:
    up_f1 = _upstream_qa_f1()
    n_pass = n_fail = 0
    fails = []

    def check(name, ok, info=""):
        nonlocal n_pass, n_fail
        if ok:
            n_pass += 1
        else:
            n_fail += 1
            fails.append(f"{name} {info}")

    # 選題：和 _id 排序、seed 有關，和 tokenizer 無關
    for d in DATASETS:
        ids = [r["_id"] for r in selected(d)]
        check(f"select[{d}]", len(ids) == N_PER == len(set(ids)))
    for m in models:
        tok = T.load_tokenizer(m)
        enc = T._enc(tok)
        # tokenizer 自己的 normalizer（Qwen 是 NFC：Ω U+2126 → Ω U+03A9），解碼回來的是正規化後的文字，官方流程也一樣
        nz = getattr(getattr(tok, "backend_tokenizer", None), "normalizer", None)
        norm = nz.normalize_str if nz is not None else (lambda x: x)
        print(f"== {m}：normalizer {nz}；chat 開頭 {len(enc.head)} token、user→assistant 膠水 {len(enc.u2a)} token")
        for d in DATASETS:
            raw, nblk, n_trunc, n_dec_mismatch = [], [], 0, 0
            for i in range(N_PER):
                it = make_item(d, i, tok)
                obj = selected(d)[i]
                ctx, q = it["context_ids"], it["turns"][0]["query_ids"]
                check(f"{m}/{d}[{i}] ctx_len", len(ctx) % 512 == 0 and 512 <= len(ctx) <= 32768, len(ctx))
                check(f"{m}/{d}[{i}] total_fits", len(ctx) + len(q) + max(64, it["max_new_tokens_upstream"]) <= KV_MAX_LEN)
                # context＋query 解碼回去＝chat 開頭＋官方截斷後的 prompt＋膠水（官方截斷用逐字照抄的參考實作）
                ref = _ref_trunc(d, obj, tok)
                prompt, _, _ = official_prompt(d, obj, tok)
                check(f"{m}/{d}[{i}] trunc_eq_ref", prompt == ref)
                dec = tok.decode(ctx + q, clean_up_tokenization_spaces=False)
                want = norm(enc.head_text + ref + enc.u2a_text)
                if dec != want:
                    n_dec_mismatch += 1
                check(f"{m}/{d}[{i}] decode_eq", dec == want, f"len {len(dec)} vs {len(want)}")
                # 後綴（結尾說明＋問題）整段都在 query；query 解碼的結尾＝prompt 的後綴＋膠水
                check(f"{m}/{d}[{i}] suffix_in_query", it["suffix_token"] >= len(ctx), (it["suffix_token"], len(ctx)))
                c0 = suffix_char(d, prompt, tok)
                check(f"{m}/{d}[{i}] query_tail", tok.decode(q, clean_up_tokenization_spaces=False).endswith(
                    norm(prompt[c0:] + enc.u2a_text)))
                check(f"{m}/{d}[{i}] gold", bool(it["turns"][0]["gold"]) and all(g.strip() for g in it["turns"][0]["gold"]))
                again = make_item(d, i, tok)
                check(f"{m}/{d}[{i}] deterministic", again == it)
                raw.append(it["raw_prompt_tokens"])
                nblk.append(it["n_blocks"])
                n_trunc += it["truncated"]
                # 計分對拍：上游 qa_f1_score（多答案取最大，和上游 eval.py 相同）vs m10_tasks.qa_f1
                golds = it["turns"][0]["gold"]
                other = selected(DATASETS[(DATASETS.index(d) + 1) % len(DATASETS)])[i]["answers"][0]
                for pred in (golds[0], "The answer is " + golds[0] + ".", " ".join(golds[0].split()[:3]),
                             golds[0].upper() + "!!", other, "", "an " + golds[-1]):
                    a = max(up_f1(pred, g) for g in golds)
                    b = T.qa_f1(pred, golds)
                    check(f"{m}/{d}[{i}] f1_eq", abs(a - b) < 1e-12, f"{pred!r}: {a} vs {b}")
            raw.sort()
            print(f"  {d:16s} 原始 prompt token：min {raw[0]:6d} 中位 {int(statistics.median(raw)):6d} max {raw[-1]:6d}"
                  f"｜被截斷 {n_trunc}/{N_PER}｜context block 數：min {min(nblk)} 中位 {int(statistics.median(nblk))} max {max(nblk)}"
                  f"｜解碼不同 {n_dec_mismatch}")
        # 一題的摘要給人看
        it = make_item("hotpotqa", 0, tok)
        print(f"  例：{it['item_id']} {it['n_blocks']} blocks；query 尾巴：{tok.decode(it['turns'][0]['query_ids'])[-300:]!r}")
        print(f"      gold：{it['turns'][0]['gold']}")
    for f in fails[:20]:
        print("FAIL", f)
    print(f"自測：通過 {n_pass}、失敗 {n_fail}")
    return 0 if n_fail == 0 else 1


# ── E0 小測（GPU；彩排，只跑全 BF16）─────────────────────────────────────

def e0_smoke(a):
    """每任務 n 題，Llama，只跑 E0。m10_quality.run_item_e0 目前 assert context 剛好 64 個 block，
    LongBench 的 context 長度會變，所以這裡用一個本地的小函式（流程照 run_item_e0：chunk 化 prefill→接 query→greedy）。"""
    import csv
    import time
    from datetime import datetime

    import torch
    import m10_quality as Q

    ans = Q.Answerer(a.model)
    rid = os.environ.get("RUN_ID", "adhoc")
    rows = []
    for d in DATASETS:
        for i in range(a.n):
            it = make_item(d, i, ans.tok)
            ctx = it["context_ids"]
            t0 = time.perf_counter()
            ans.km.prefill_chunked(torch.tensor(ctx, device=ans.km.device))
            tr = it["turns"][0]
            up = it["max_new_tokens_upstream"]
            max_new = max(Q.CUT, up)
            logits = ans.feed(tr["query_ids"], len(ctx))
            gen, stop = ans.generate(logits, len(ctx) + len(tr["query_ids"]), max_new)
            torch.cuda.synchronize()
            p64, pup = ans.decode(gen[:Q.CUT]), ans.decode(gen[:up])
            r = dict(run_id=rid, ts=datetime.now().astimezone().isoformat(timespec="seconds"), model=a.model,
                     item_id=it["item_id"], task=it["task"], lb_id=it["lb_id"], n_blocks=it["n_blocks"],
                     raw_prompt_tokens=it["raw_prompt_tokens"], truncated=it["truncated"], n_gen=len(gen), stop=stop,
                     upstream_len=up, score_64=T.score("qa_f1", p64, tr["gold"]),
                     score_up=T.score("qa_f1", pup, tr["gold"]), gold=json.dumps(tr["gold"], ensure_ascii=False),
                     pred=ans.decode(gen), t_s=round(time.perf_counter() - t0, 2))
            rows.append(r)
            print(f"{it['item_id']:22s} blk={it['n_blocks']:2d} f1_up={r['score_up']:.2f} f1_64={r['score_64']:.2f} "
                  f"gold={tr['gold'][0][:40]!r} pred={r['pred'][:60]!r}", flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("== 每任務平均 qa_f1×100（官方長度 score_up；n＝%d）" % a.n)
    for d in DATASETS:
        v = [r for r in rows if r["task"] == f"LB-{d}"]
        print(f"  {d:16s} {100 * sum(r['score_up'] for r in v) / len(v):6.1f}  （截 64：{100 * sum(r['score_64'] for r in v) / len(v):6.1f}）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--e0-smoke", action="store_true")
    ap.add_argument("--model", default="llama", choices=list(T.MODELS))
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(self_test())
    if a.e0_smoke:
        e0_smoke(a)
        sys.exit(0)
    ap.print_help()
