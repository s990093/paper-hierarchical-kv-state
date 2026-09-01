#!/usr/bin/env python3
"""RULER 風格的合成長上下文任務——產生器與計分器。

## 為什麼需要這一組任務

M5 已經量到兩個端點：**GSM8K（推理）** 上四個 KV 精度階的差異與零無法區分，
而**大海撈針（單鍵檢索）**上同一組設定由 100% 掉到 0%。
兩者之間缺一塊：**長上下文的「理解／整合」**——不是把一個字串找回來，
而是要跨越整份上下文做多跳追蹤與聚合。

RULER（Hsieh et al., NVIDIA, 2024）的合成任務族正好張開這條軸：

| 任務 | 要做的事 | 與大海撈針的差別 |
|---|---|---|
| `niah_multikey_2/3` | 在**全是干擾針**的海裡找一根 | 海本身就是同構的針，靠的是鍵的比對而非「異常偵測」 |
| `niah_multivalue` | 一個鍵、**四個值**，要全數回收 | 單點檢索 → 多點聚合 |
| `niah_multiquery` | **四個鍵**各查一次 | 同上，且查詢分散 |
| `vt` | 變數指派鏈的**多跳追蹤** | 答案不在任何單一位置，要串起 5 個位置 |
| `cwe` / `fwe` | 全文**詞頻聚合** | 答案是整份上下文的統計量，沒有「針」可撈 |

計分沿用 RULER 的 `string_match_all`（參考答案有幾成出現在輸出裡），
`qa` 類用 `string_match_part`。見上游 `scripts/eval/synthetic/constants.py`。

## 與上游的兩處刻意差異（**必須寫進論文**）

1. **haystack 一律用 `noise`**（"The grass is green. ..." 重複），
   不用上游的 Paul Graham essay。原因：該 JSON 需要從 paulgraham.com 爬，
   且非可再散布資料；`noise` 是上游本身就提供的選項之一
   （`niah_single_1`、`vt` 的預設）。受影響的是 `niah_multivalue` 與
   `niah_multiquery`（上游用 essay）——任務語意不變，難度略降。
2. **不含 `qa_1/qa_2`**（需 SQuAD / HotpotQA 原始檔）。
   真實文件上的理解由 LongBench 那一半負責（見 `m5_understanding.py`）。

其餘（模板、答案前綴、複雜度參數、二分搜尋對齊長度、計分）皆照上游。

本模組**不碰 GPU、不連網**，可單獨測試：

    python code/ruler_tasks.py --self-test
"""

from __future__ import annotations

import os
import random
import string
import sys
import uuid
from typing import Callable

import numpy as np

# wonderwords 的詞表（上游用同一份）安裝在側裝目錄，避免污染 vLLM venv
sys.path.insert(0, os.environ.get("PAPER_HKV_PYLIBS", "/ssd7/hungwei/paper-hkv/pylibs"))

# ── 上游 scripts/data/synthetic/constants.py 的模板（逐字） ──────────────
TEMPLATES = {
    "niah": {
        "tokens_to_generate": 128,
        "template": "Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text?",
        "answer_prefix": " The special magic {type_needle_v} for {query} mentioned in the provided text are",
    },
    "variable_tracking": {
        "tokens_to_generate": 30,
        "template": "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{context}\nQuestion: Find all variables that are assigned the value {query} in the text above.",
        "answer_prefix": " Answer: According to the chain(s) of variable assignment in the text above, {num_v} variables are assigned the value {query}, they are: ",
    },
    "common_words_extraction": {
        "tokens_to_generate": 120,
        "template": "Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.\n{context}\nQuestion: What are the 10 most common words in the above list?",
        "answer_prefix": " Answer: The top 10 words that appear most often in the list are:",
    },
    "freq_words_extraction": {
        "tokens_to_generate": 50,
        "template": "Read the following coded text and track the frequency of each coded word. Find the three most frequently appeared coded words. {context}\nQuestion: Do not provide any explanation. Please ignore the dots '....'. What are the three most frequently appeared words in the above coded text?",
        "answer_prefix": " Answer: According to the coded text above, the three most frequently appeared words are:",
    },
}

NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
NEEDLE_FMT = "One of the special magic {type_needle_v} for {key} is: {value}."

# ── 上游 scripts/synthetic.yaml 的複雜度參數 ────────────────────────────
# haystack 一律改為 noise / needle（見檔頭差異 1）。
RULER_TASKS = {
    "niah_multikey_2": dict(kind="niah", haystack="needle", k_type="words",
                            v_type="numbers", n_k=1, n_v=1, n_q=1,
                            zh="多鍵檢索（海本身即干擾針）"),
    "niah_multikey_3": dict(kind="niah", haystack="needle", k_type="uuids",
                            v_type="uuids", n_k=1, n_v=1, n_q=1,
                            zh="多鍵檢索（UUID，無語意線索）"),
    "niah_multivalue": dict(kind="niah", haystack="noise", k_type="words",
                            v_type="numbers", n_k=1, n_v=4, n_q=1,
                            zh="一鍵四值，需全數回收"),
    "niah_multiquery": dict(kind="niah", haystack="noise", k_type="words",
                            v_type="numbers", n_k=4, n_v=1, n_q=4,
                            zh="四鍵各查一次"),
    "vt":  dict(kind="variable_tracking", haystack="noise", n_chains=1, n_hops=4,
                zh="變數指派鏈的多跳追蹤"),
    "cwe": dict(kind="common_words_extraction", freq_cw=30, freq_ucw=3, num_cw=10,
                zh="全文詞頻聚合（前 10 高頻詞）"),
    "fwe": dict(kind="freq_words_extraction", alpha=2.0,
                zh="全文詞頻聚合（Zipf 前 3 高頻詞）"),
}


def _wonderwords(name: str) -> list[str]:
    import wonderwords.random_word as rw
    return rw._get_words_from_text_file(name)


_ADJ_NOUN: list[str] | None = None
_PLAIN: list[str] | None = None


def _adj_noun_words() -> list[str]:
    """上游 niah.py 的 `words`：adjective-noun 兩兩相接後去重排序。"""
    global _ADJ_NOUN
    if _ADJ_NOUN is None:
        nouns, adjs = _wonderwords("nounlist.txt"), _wonderwords("adjectivelist.txt")
        _ADJ_NOUN = sorted({f"{a}-{n}" for a in adjs for n in nouns})
    return _ADJ_NOUN


def _plain_words(seed: int) -> list[str]:
    """上游 common_words_extraction.py 的 `words`：名詞+形容詞+動詞去重後洗牌。"""
    global _PLAIN
    if _PLAIN is None:
        w = sorted(set(_wonderwords("nounlist.txt") + _wonderwords("adjectivelist.txt")
                       + _wonderwords("verblist.txt")))
        random.Random(seed).shuffle(w)
        _PLAIN = w
    return _PLAIN


# ── 隨機基元（上游 niah.py） ────────────────────────────────────────────

def _rand_number(rng: random.Random, digits: int = 7) -> str:
    return str(rng.randint(10 ** (digits - 1), 10 ** digits - 1))


def _rand_uuid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def _rand(rng: random.Random, kind: str) -> str:
    if kind == "numbers":
        return _rand_number(rng)
    if kind == "words":
        return rng.choice(_adj_noun_words())
    if kind == "uuids":
        return _rand_uuid(rng)
    raise NotImplementedError(kind)


# ── 產生器 ─────────────────────────────────────────────────────────────

def _gen_niah(cfg: dict, n_hay: int, rng: random.Random, seed: int) -> tuple[str, list[str]]:
    n_k = max(cfg["n_k"], cfg["n_q"])
    keys, values, needles = [], [], []
    for _ in range(n_k):
        keys.append(_rand(rng, cfg["k_type"]))
        vs = []
        for _ in range(cfg["n_v"]):
            vs.append(_rand(rng, cfg["v_type"]))
            needles.append(NEEDLE_FMT.format(type_needle_v=cfg["v_type"],
                                             key=keys[-1], value=vs[-1]))
        values.append(vs)
    random.Random(seed).shuffle(needles)

    if cfg["haystack"] == "noise":
        sentences = [NOISE] * n_hay
    else:                                    # 'needle'：海本身就是同構的干擾針
        sentences = [NEEDLE_FMT.format(type_needle_v=cfg["v_type"],
                                       key=_rand(rng, cfg["k_type"]),
                                       value=_rand(rng, cfg["v_type"]))
                     for _ in range(n_hay)]
    for index, element in zip(sorted(rng.sample(range(n_hay), len(needles)), reverse=True),
                              needles):
        sentences.insert(index, element)
    context = "\n".join(sentences)

    idx = rng.sample(range(n_k), cfg["n_q"])
    queries = [keys[i] for i in idx]
    answers = [a for i in idx for a in values[i]]
    query = (", ".join(queries[:-1]) + ", and " + queries[-1]) if len(queries) > 1 else queries[0]

    tmpl = TEMPLATES["niah"]["template"] + TEMPLATES["niah"]["answer_prefix"]
    v_type = cfg["v_type"]
    if cfg["n_q"] * cfg["n_v"] == 1:          # 上游的單數化處理
        tmpl = (tmpl.replace("Some", "A").replace("are all", "is")
                .replace("are", "is").replace("answers", "answer"))
        v_type = v_type[:-1]
    return tmpl.format(type_needle_v=v_type, context=context, query=query), answers


def _gen_vt(cfg: dict, n_hay: int, rng: random.Random, seed: int) -> tuple[str, list[str]]:
    n_chains, n_hops = cfg["n_chains"], cfg["n_hops"]
    need = n_chains * (n_hops + 1)
    names: list[str] = []
    while len(set(names)) < need:
        names.append("".join(rng.choices(string.ascii_uppercase, k=5)).upper())
    names = list(dict.fromkeys(names))[:need]

    chains, first_vars = [], []
    for i in range(0, len(names), n_hops + 1):
        this = names[i:i + n_hops + 1]
        first_vars.append(this)
        chain = [f"VAR {this[0]} = {rng.randint(10000, 99999)}"]
        for j in range(n_hops):
            chain.append(f"VAR {this[j + 1]} = VAR {this[j]} ")
        chains.append(chain)
    value = chains[0][0].split("=")[-1].strip()

    sentences = [NOISE] * n_hay
    for chain in chains:
        for insert_pi, j in zip(sorted(rng.sample(range(len(sentences)), len(chain))),
                                range(len(chain))):
            sentences.insert(insert_pi + j, chain[j])
    context = "\n".join(sentences).replace(". \n", ".\n")

    tmpl = (TEMPLATES["variable_tracking"]["template"]
            + TEMPLATES["variable_tracking"]["answer_prefix"])
    return tmpl.format(context=context, query=value, num_v=n_hops + 1), first_vars[0]


def _gen_cwe(cfg: dict, n_words: int, rng: random.Random, seed: int) -> tuple[str, list[str]]:
    words = _plain_words(seed)
    num_cw = cfg["num_cw"]

    def example(n: int, rep_c: int, rep_u: int) -> tuple[str, list[str]]:
        full = rng.sample(words, min(n, len(words)))
        common, uncommon = full[:num_cw], full[num_cw:]
        lst = common * rep_c + uncommon * rep_u
        random.Random(seed).shuffle(lst)
        return " ".join(f"{i + 1}. {w}" for i, w in enumerate(lst)), common

    tmpl = (TEMPLATES["common_words_extraction"]["template"]
            + TEMPLATES["common_words_extraction"]["answer_prefix"])
    fs_ctx, fs_ans = example(40, 10, 3)                       # 上游的 1-shot 範例
    shot = tmpl.format(context=fs_ctx, query="") + " " + \
        " ".join(f"{i + 1}. {w}" for i, w in enumerate(fs_ans))
    ctx, ans = example(n_words, cfg["freq_cw"], cfg["freq_ucw"])
    return shot + "\n" + tmpl.format(context=ctx, query=""), ans


def _zeta(a: float, n: int = 24) -> float:
    """Riemann zeta（a > 1），Euler-Maclaurin 尾項修正。取代上游的 scipy.special.zeta，
    只為了不把 scipy 拉進這個 venv；n=24 時相對誤差 < 1e-13。"""
    s = sum(k ** -a for k in range(1, n))
    return (s + n ** (1 - a) / (a - 1) + 0.5 * n ** -a
            + a / 12 * n ** (-a - 1) - a * (a + 1) * (a + 2) / 720 * n ** (-a - 3))


def _gen_fwe(cfg: dict, n_words: int, rng: random.Random, seed: int,
             vocab_size: int, coded_len: int = 6) -> tuple[str, list[str]]:
    vocab: list[str] = []
    while len(set(vocab)) < vocab_size:
        vocab.append("".join(rng.choices(string.ascii_lowercase, k=coded_len)))
    vocab = sorted(set(vocab))
    random.Random(seed).shuffle(vocab)
    vocab[0] = "..."                                # 上游：把最高頻的那個當雜訊

    k = np.arange(1, len(vocab) + 1)
    cnt = (n_words * (k ** -cfg["alpha"]) / _zeta(cfg["alpha"])).astype(int)
    sampled = [w for w, c in zip(vocab, cnt) for _ in range(c)]
    random.Random(seed).shuffle(sampled)
    tmpl = (TEMPLATES["freq_words_extraction"]["template"]
            + TEMPLATES["freq_words_extraction"]["answer_prefix"])
    return tmpl.format(context=" ".join(sampled), query=""), vocab[1:4]


# ── 對齊到目標長度：二分搜尋 haystack 大小（上游 generate_samples 的做法） ──

def _fit(make: Callable[[int], tuple[str, list[str]]], n_tok: Callable[[str], int],
         budget: int, lo: int, hi: int) -> int:
    """找最大的 n 使 token 數 <= budget。假設 token 數對 n 單調遞增。"""
    while n_tok(make(hi)[0]) <= budget:
        lo, hi = hi, hi * 2
        if hi > 10_000_000:
            break
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        if mid <= 0:
            break
        if n_tok(make(mid)[0]) <= budget:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return max(best, 1)


def build(task: str, ctx: int, n_samples: int, tok, seed: int = 42) -> list[dict]:
    """產生 `n_samples` 筆 `task` 的測資，prompt 長度貼齊 `ctx`（含生成預算）。

    回傳每筆含 prompt / answers / prompt_tokens / max_new_tokens。
    """
    if task not in RULER_TASKS:
        raise KeyError(f"未知的 RULER 任務 {task}；可用：{sorted(RULER_TASKS)}")
    cfg = RULER_TASKS[task]
    kind = cfg["kind"]
    gen_budget = TEMPLATES[kind]["tokens_to_generate"]
    budget = ctx - gen_budget

    def n_tok(s: str) -> int:
        return len(tok(s, add_special_tokens=False)["input_ids"])

    rng = random.Random(seed)
    if kind == "niah":
        size = _fit(lambda n: _gen_niah(cfg, n, random.Random(seed), seed),
                    n_tok, budget, 25, 200)
    elif kind == "variable_tracking":
        size = _fit(lambda n: _gen_vt(cfg, n, random.Random(seed), seed),
                    n_tok, budget, 25, 200)
    elif kind == "common_words_extraction":
        size = _fit(lambda n: _gen_cwe(cfg, n, random.Random(seed), seed),
                    n_tok, budget, 40, 200)
    else:                                          # fwe
        vocab_size = max(4, budget // 50)          # 上游：max_seq_length // 50
        size = _fit(lambda n: _gen_fwe(cfg, n, random.Random(seed), seed, vocab_size),
                    n_tok, budget, 100, 1000)

    out = []
    for i in range(n_samples):
        r = random.Random(seed + 1000 * i)
        if kind == "niah":
            prompt, ans = _gen_niah(cfg, size, r, seed + i)
        elif kind == "variable_tracking":
            prompt, ans = _gen_vt(cfg, size, r, seed + i)
        elif kind == "common_words_extraction":
            prompt, ans = _gen_cwe(cfg, size, r, seed + i)
        else:
            prompt, ans = _gen_fwe(cfg, size, r, seed + i, max(4, budget // 50))
        out.append({"task": task, "prompt": prompt, "answers": ans,
                    "prompt_tokens": n_tok(prompt), "max_new_tokens": gen_budget,
                    "haystack_units": size})
    return out


# ── 計分（上游 scripts/eval/synthetic/constants.py 逐字） ────────────────

def string_match_all(pred: str, refs: list[str]) -> float:
    """參考答案有幾成（不分大小寫）出現在輸出裡。RULER 全部合成任務的預設。"""
    if not refs:
        return 0.0
    return sum(1.0 if r.lower() in pred.lower() else 0.0 for r in refs) / len(refs)


def string_match_part(pred: str, refs: list[str]) -> float:
    return max((1.0 if r.lower() in pred.lower() else 0.0) for r in refs) if refs else 0.0


def score(task: str, pred: str, refs: list[str]) -> float:
    return string_match_all(pred, refs)


def _self_test() -> int:
    """不需 GPU：檢查每個任務都能產生、長度貼齊、且金標可被自身計分為 1.0。"""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--model", default="/ssd7/hungwei/paper-hkv/models/Qwen2.5-7B-Instruct-1M-AWQ-noDCA")
    a = ap.parse_args()
    os.environ.setdefault("HF_HOME", "/ssd7/hungwei/paper-hkv/hf-cache/huggingface")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    bad = 0
    for t in RULER_TASKS:
        s = build(t, a.ctx, 2, tok)
        lens = [x["prompt_tokens"] for x in s]
        gold = " ".join(s[0]["answers"])
        sc = score(t, gold, s[0]["answers"])
        ok = all(l <= a.ctx for l in lens) and sc == 1.0
        bad += not ok
        print(f"{t:18s} tok={lens} gen={s[0]['max_new_tokens']:4d} "
              f"units={s[0]['haystack_units']:6d} ans={s[0]['answers']} "
              f"self-score={sc:.2f} {'OK' if ok else '🔴'}")
    return bad


if __name__ == "__main__":
    sys.exit(_self_test())
