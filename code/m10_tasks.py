#!/usr/bin/env python3
"""m10_tasks.py — 精度這條線（m10）的品質題目：T1–T5，需要的資訊放在指定的位置帶。

依據：docs/precision_20261013/03_plan_h200.md §3.1（任務）、§3.2（難度參數）、§3.4（第 2 輪）、§6（這一支）。
本模組**不碰 GPU、不跑模型**，只用 tokenizer；`python code/m10_tasks.py --self-test` 自測。

## 任務

| 代號 | 上游名稱 | 需要的資訊 | 計分 |
|:--|:--|:--|:--|
| T1 | niah_multikey_3 | 一根 UUID 鍵值針（整片海都是同格式的干擾針）；可加難：n_queries、n_similar（見下） | string_match_all |
| T2 | niah_multivalue | 一個鍵的 4 根針（4 個值） | string_match_all |
| T3 | vt | 一條 4 跳的變數指派鏈（5 行） | string_match_all |
| T4 | qa（HotpotQA） | 兩段 gold 段落 | string_match_part |
| T5 | cwe | 全文詞頻（沒有位置帶，當「全域」對照） | string_match_all |

模板、答案前綴、needle 句型、noise haystack、計分都逐字取自 `git show 52803bc^:code/ruler_tasks.py`
（它又逐字取自 RULER 上游 `scripts/data/synthetic/constants.py`）；qa 的模板與 `Document {i}:` 格式
取自上游 `scripts/data/synthetic/qa.py`（本機副本 /mlsteam/data/tiara/datasets/ruler_ref/RULER）。

## 位置帶與 block 0

context 固定 n_blocks × block ＝ 64 × 512 ＝ 32,768 token。BANDS 是含頭尾的 block 範圍：
前＝1–16、中＝24–39、後＝48–63。block 0 放 chat template 開頭、任務說明（T3 另有上游的 1-shot 範例），
不放任何需要的資訊（混合配置裡 block 0 一律 BF16）。「需要的資訊」（needle）的 token 位置在組裝時就記下，
自測再用「解碼後搜尋答案字串 → 換回 token 位置」的獨立路徑驗算一次（CLAUDE.md 規則 6 的精神）。

## 為什麼「各段分開 tokenize 再接 token id」

1. **長度要剛好 32,768**：每個 block 都是之後真的存、真的量化的單位，必須剛好 64 個滿的 block。
   整串文字一次 tokenize 時，BPE 會跨段合併，長度和 needle 位置都要事後反推，無法精確控制。
2. **切點都選在 pre-tokenizer 本來就會切開的地方**：每段（一句 noise、一根針、一份文件）都以換行結尾，
   下一段從字母開始。Llama-3 與 Qwen2 的 pre-tokenizer 在這些地方本來就切開，所以分開 tokenize 再接起來，
   和整串一次 tokenize 得到相同的 token。自測會實際比對「解碼後重新 tokenize」的切點：
   **接縫以外必須逐 token 相同**（不同就算失敗）；接縫處（context 結尾接問題、答案前綴接第 1 輪答案）可能差一個切點，
   例如 context 截在 '.'、問題以 '\\n' 開頭 → '.' '\\n' 而不是 '.\\n'；vt 的答案前綴照上游以空白結尾，
   答案 token 接在那個空白 token 之後（和模型實際生成時看到的一樣）。所有配置用同一串 token，不影響比較。
3. **最後一段 filler 在 token 邊界截斷**，補到剛好 32,768。所以 context 的結尾可能是一句被截斷的 filler
   （只會是 filler，自測保證 needle 都完整在截斷點之前）。

## 第 2 輪（03 §3.4）

context 同時放第 1、2 輪要問的資訊，分別在 band1、band2（必須不同）。第 2 輪的序列＝
context＋turns[0].query_ids＋turns[1].history_ids＋turns[1].query_ids，其中 history_ids 是
「第 1 輪的**標準答案文字**＋結束 assistant＋user 開頭」（歷史固定，所有配置相同）。
**T5 第 2 輪：待決定**（T5 沒有位置帶，現在只做單輪）。

## 和上游 RULER 的差異（要寫進報告）

- haystack 一律 `noise`（T2 上游用 essay），同 ruler_tasks.py 的差異 1。
- T2 的鍵：形容詞與名詞**各自**均勻抽再接起來，不先展開 912×6782 的組合表再抽（分布相同，只差去重後極少數重複組合；省 ~400 MB 記憶體）。
- T3 的 1-shot 範例：上游讓範例本身 ≤ 500 token；這裡改成「chat 開頭＋範例＋任務說明 ≤ 512」，讓整段任務說明剛好在 block 0。
  變數名額外排除「小寫後是模板／noise 文字的子字串」的組合（例如 GREEN、ABOVE），否則 string_match 會被模板文字誤判為答對。
- T4：其餘位置填**其他題**的段落（03 §3.1），預設不放本題自己的 8 段干擾段落（`n_hard_distractors` 可加）；
  filler 段落若含任一題的答案字串就不用；答案是 yes/no、出現在問題裡、或是 1–3 位數字（會撞到 `Document N:`）的題目不用。
- T5：1-shot 範例照 ruler_tasks.py 的寫法（不把 chat template 包進範例，上游那樣會重複 chat template）；
  詞表排除「小寫後是模板文字子字串」的詞（例如 list、words），避免答案前綴本身就讓計分命中。
- T1 的加難參數（10/15 加，預設＝不變）：`n_queries`＝一次問幾個鍵（照上游 `niah_multiquery`：每個鍵一個值、
  問句用 "k1, k2, and k3"、答案要全部答出、用複數模板），所有被問的鍵都放在同一個帶內；
  `n_similar`／`similar_prefix`＝每個被問的鍵配幾根「相似鍵」干擾針，鍵和被問的鍵前 similar_prefix 個字元相同、
  其餘是新的隨機 UUID 字元、值不同，插在任何位置（可能在答案帶內）。**相似鍵是我們加的，和上游不同**（RULER 沒有）。
- 答案前綴照上游保留開頭的空白（接在 assistant 開頭之後）。
- Llama-3.1 的 chat template 固定 `date_string="26 Jul 2024"`（模板的預設值），可重現。
"""

from __future__ import annotations

import argparse
import bisect
import glob
import json
import os
import random
import re
import string
import sys
import uuid
from collections import Counter

BLOCK = 512
N_BLOCKS = 64
BANDS = {"front": (1, 16), "mid": (24, 39), "back": (48, 63)}   # 含頭尾的 block 範圍


def bands(n_blocks: int = N_BLOCKS) -> dict:
    """位置帶跟著 block 數等比例縮放（10/16 長 context 實驗加，docs/precision_20261013/10_longctx_plan.md）：
    前＝1..n/4、中＝3n/8..5n/8−1、後＝3n/4..n−1（都取整數部分）。n＝64 時和 BANDS 完全相同（主線題目不變）。"""
    if n_blocks == N_BLOCKS:
        return BANDS
    q = n_blocks / 8
    return {"front": (1, int(2 * q)), "mid": (int(3 * q), int(5 * q) - 1), "back": (int(6 * q), n_blocks - 1)}


assert bands(64) == BANDS

TASKS = {"T1": "niah_multikey_3", "T2": "niah_multivalue", "T3": "vt", "T4": "hotpotqa", "T5": "cwe"}
_ALIAS = {v: k for k, v in TASKS.items()}

DEFAULT_PARAMS = {
    # haystack="needle"：整片海都是同格式的干擾針（上游 multikey_3）；"noise" 時改放 n_distractors 根干擾針
    "T1": {"haystack": "needle", "n_distractors": 0,
           # 10/15 加（使用者選「T1 另做更難的版本」；預設＝不變）：n_queries＝一次問幾個鍵（RULER niah_multiquery 的做法）；
           # n_similar＝每個被問的鍵各配幾根「相似鍵」干擾針（鍵的前 similar_prefix 個字元相同、值不同）——相似鍵是我們加的，上游沒有
           "n_queries": 1, "n_similar": 0, "similar_prefix": 0},
    "T2": {"n_values": 4, "n_distractors": 0},                       # 干擾＝其他鍵的單值針，放在任何位置
    "T3": {"n_hops": 4, "n_distractor_chains": 0, "icl": True},      # 干擾鏈放在任何位置
    "T4": {"n_hard_distractors": 0,                                   # 每題自己的干擾段落（HotpotQA distractor）放幾段
           "max_answer_words": 0},                                    # >0：只出答案 ≤ 這麼多個字的題（10/15 加；0＝不限，行為不變）
    "T5": {"freq_cw": 30, "freq_ucw": 3, "num_cw": 10},
}
TOKENS_TO_GENERATE = {"T1": 128, "T2": 128, "T3": 30, "T4": 32, "T5": 120}   # 上游值；03 §1 是每題最多 64

PYLIBS = os.environ.get("PAPER_HKV_PYLIBS", "/mlsteam/data/tiara/pylibs")
HOTPOT = os.environ.get("M10_HOTPOT",
                        "/mlsteam/data/tiara/datasets/hotpotqa/hotpot_qa_distractor_validation.parquet")
MODELS = {
    "llama": "/mlsteam/data/tiara/hf-cache/hub/models--unsloth--Llama-3.1-8B-Instruct/snapshots/*",
    "qwen": "/mlsteam/data/tiara/hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct-1M/snapshots/*",
}
LLAMA_DATE = "26 Jul 2024"

# ── 上游模板（逐字） ───────────────────────────────────────────────────
NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
NEEDLE_FMT = "One of the special magic {type_needle_v} for {key} is: {value}."
NIAH_INSTR = ("Some special magic {type_needle_v} are hidden within the following text. "
              "Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n")
NIAH_Q = "What are all the special magic {type_needle_v} for {query} mentioned in the provided text?"
NIAH_PREFIX = " The special magic {type_needle_v} for {query} mentioned in the provided text are"
VT_INSTR = "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n"
VT_Q = "Question: Find all variables that are assigned the value {query} in the text above."
VT_PREFIX = (" Answer: According to the chain(s) of variable assignment in the text above, "
             "{num_v} variables are assigned the value {query}, they are: ")
CWE_INSTR = ("Below is a numbered list of words. In these words, some appear more often than others. "
             "Memorize the ones that appear most often.\n")
CWE_Q = "Question: What are the 10 most common words in the above list?"
CWE_PREFIX = " Answer: The top 10 words that appear most often in the list are:"
QA_INSTR = ("Answer the question based on the given documents. Only give me the answer and do not output "
            "any other words.\n\nThe following are given documents.\n\n")
QA_Q = ("Answer the question based on the given documents. Only give me the answer and do not output "
        "any other words.\n\nQuestion: {query}")
QA_PREFIX = " Answer:"
DOC_FMT = "Document {i}:\n{document}"
_FIXED_TEXT = "\n".join([NOISE, NEEDLE_FMT, NIAH_INSTR, NIAH_Q, NIAH_PREFIX, VT_INSTR, VT_Q, VT_PREFIX,
                         CWE_INSTR, CWE_Q, CWE_PREFIX, QA_INSTR, QA_Q, QA_PREFIX]).lower()


def _singular(s: str) -> str:
    """上游 niah.py 在 num_needle_q × num_needle_v == 1 時的單數化（作用在格式化之前的模板）。"""
    return s.replace("Some", "A").replace("are all", "is").replace("are", "is").replace("answers", "answer")


# ── 計分（上游 scripts/eval/synthetic/constants.py，單題版，回傳 0–1） ─────────

def string_match_all(pred: str, refs: list[str]) -> float:
    if not refs:
        return 0.0
    return sum(1.0 if r.lower() in pred.lower() else 0.0 for r in refs) / len(refs)


def string_match_part(pred: str, refs: list[str]) -> float:
    return max((1.0 if r.lower() in pred.lower() else 0.0) for r in refs) if refs else 0.0


def _norm_answer(s: str) -> str:
    """LongBench metrics.py 的 normalize_answer（＝HotpotQA 官方 eval）：小寫、去標點、去冠詞、壓空白。"""
    import re
    import string as _s
    s = "".join(ch for ch in s.lower() if ch not in set(_s.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_f1(pred: str, gold: list[str]) -> float:
    """LongBench 的 qa_f1_score（hotpotqa 用的計分），多個 gold 取最大。10/15 加：T4 的 string_match_part 太嚴
    （"Steve Coogan" 對 "Stephen John Coogan" 算錯），先兩種都記，之後決定主判準用哪個。"""
    from collections import Counter
    best = 0.0
    p = _norm_answer(pred).split()
    for g in gold:
        gt = _norm_answer(g).split()
        common = Counter(p) & Counter(gt)
        ns = sum(common.values())
        if ns:
            pr, rc = ns / len(p), ns / len(gt)
            best = max(best, 2 * pr * rc / (pr + rc))
    return best


def score(metric: str, pred: str, gold: list[str]) -> float:
    if metric == "string_match_all":
        return string_match_all(pred, gold)
    if metric == "string_match_part":
        return string_match_part(pred, gold)
    if metric == "qa_f1":
        return qa_f1(pred, gold)
    raise ValueError(metric)


# ── 詞表（直接讀 wonderwords 的文字檔，不 import：pylibs 裡另有一份 numpy，放進 sys.path 會蓋掉 venv 的） ──

_WORDS: dict[str, list[str]] = {}


def _wordfile(name: str) -> list[str]:
    if name not in _WORDS:
        p = os.path.join(PYLIBS, "wonderwords", "assets", name)
        with open(p, encoding="utf-8") as f:
            _WORDS[name] = [w.rstrip() for w in f.readlines()]   # 同 wonderwords._get_words_from_text_file
    return _WORDS[name]


def _adjs_nouns() -> tuple[list[str], list[str]]:
    return sorted(set(_wordfile("adjectivelist.txt"))), sorted(set(_wordfile("nounlist.txt")))


_CWE_VOCAB: list[str] | None = None


def _cwe_vocab() -> list[str]:
    """上游 common_words_extraction.py：名詞＋形容詞＋動詞，去重排序後用固定 seed 42 洗牌。
    額外排除小寫後是模板文字子字串的詞（見檔頭差異）。"""
    global _CWE_VOCAB
    if _CWE_VOCAB is None:
        w = sorted(set(_wordfile("nounlist.txt") + _wordfile("adjectivelist.txt") + _wordfile("verblist.txt")))
        random.Random(42).shuffle(w)
        _CWE_VOCAB = [x for x in w if x and x.lower() not in _FIXED_TEXT]
    return _CWE_VOCAB


# ── tokenizer 與 chat template 的膠水 ────────────────────────────────────

def load_tokenizer(model_glob_or_path: str):
    """接受模型路徑、glob（取排序最後一個 snapshot），或 'llama'／'qwen' 簡寫。"""
    from transformers import AutoTokenizer
    p = MODELS.get(model_glob_or_path, model_glob_or_path)
    if any(c in p for c in "*?["):
        ps = sorted(glob.glob(p))
        if not ps:
            raise FileNotFoundError(p)
        p = ps[-1]
    return AutoTokenizer.from_pretrained(p)


def _family(tok) -> str:
    n = (getattr(tok, "name_or_path", "") or "").lower()
    if "llama" in n:
        return "llama"
    if "qwen" in n:
        return "qwen"
    return "unknown"


class _Enc:
    """每個 tokenizer 一份：分段 tokenize（不加 special token）＋快取，以及 chat template 的膠水。"""

    def __init__(self, tok):
        self.tok = tok
        self.family = _family(tok)
        self.cache: dict[str, list[int]] = {}
        sent = ["@@M10U1@@", "@@M10A1@@", "@@M10U2@@"]
        msgs = [{"role": "user", "content": sent[0]}, {"role": "assistant", "content": sent[1]},
                {"role": "user", "content": sent[2]}]
        kw = {"date_string": LLAMA_DATE} if self.family == "llama" else {}
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        head, rest = s.split(sent[0])
        u2a, rest = rest.split(sent[1])
        a2u, u2a_again = rest.split(sent[2])
        if u2a != u2a_again:
            raise RuntimeError(f"chat template 的 user→assistant 膠水兩次不同：{u2a!r} vs {u2a_again!r}")
        self.head_text, self.u2a_text, self.a2u_text = head, u2a, a2u
        self.head, self.u2a, self.a2u = self(head), self(u2a), self(a2u)

    def __call__(self, text: str) -> list[int]:
        r = self.cache.get(text)
        if r is None:
            r = self.tok(text, add_special_tokens=False)["input_ids"]
            if len(text) < 4096:
                self.cache[text] = r
        return r

    def many(self, texts: list[str]) -> list[list[int]]:
        todo = sorted({t for t in texts if t not in self.cache})
        if todo:
            for t, ids in zip(todo, self.tok(todo, add_special_tokens=False)["input_ids"]):
                self.cache[t] = ids
        return [self.cache[t] for t in texts]


_ENCS: dict[int, _Enc] = {}


def _enc(tok) -> _Enc:
    e = _ENCS.get(id(tok))
    if e is None or e.tok is not tok:
        e = _ENCS[id(tok)] = _Enc(tok)
    return e


# ── 組裝：把 needle 放在指定 token 位置，filler 補滿，最後截到剛好 total ──────────

class _Unit:
    __slots__ = ("payload", "kind", "tag")

    def __init__(self, payload, kind="filler", tag=None):
        self.payload, self.kind, self.tag = payload, kind, tag


def _assemble(enc: _Enc, pre_text: str, fillers: list[_Unit], needles: dict[int, list[_Unit]],
              turn_band: dict[int, str], extras: list[list[_Unit]], render, total: int,
              block: int, rng: random.Random) -> tuple[list[int], list[tuple]]:
    """回傳 (context_ids, spans)。spans：(kind, tag, start, end) —— needle 與 distractor 的實際 token 範圍。

    needle 的規則：在 band 內先抽目標位置 t_j（stars and bars，彼此至少隔一個最長 filler），
    組裝時「目前長度 ≥ t_j」就放下一根，所以實際起點落在 [t_j, t_j + 最長 filler)。
    extras（干擾）在組裝前插進 filler 序列的隨機位置（同一組內保持順序，例如干擾鏈）。
    """
    pre = enc.head + enc(pre_text)
    pos0 = len(pre)
    flen = [len(enc(render(u.payload, 999))) for u in fillers]
    gap = max(flen) + 4
    nl = {t: [len(enc(render(u.payload, 999))) + 4 for u in us] for t, us in needles.items()}
    xl = sum(len(enc(render(u.payload, 999))) + 4 for g in extras for u in g)

    # extras 只插在「一定會被用到」的 filler 前綴裡
    if extras:
        budget = total - pos0 - sum(sum(v) for v in nl.values()) - xl - 2 * gap
        acc, m = 0, 0
        while m < len(flen) and acc + flen[m] <= budget:
            acc += flen[m]
            m += 1
        seq = list(fillers)
        for g in extras:
            if m + 1 < len(g):
                raise RuntimeError("干擾太多，filler 不夠插")
            idx = sorted(rng.sample(range(m + 1), len(g)))
            for j, (pi, u) in enumerate(zip(idx, g)):
                seq.insert(pi + j, u)
            m += len(g)
        fillers = seq

    targets = []
    for t, us in needles.items():
        lo_b, hi_b = bands(total // block)[turn_band[t]]
        s, e = max(lo_b * block, pos0), (hi_b + 1) * block
        need = sum(L + gap for L in nl[t])
        avail = e - s - need
        if avail < 0:
            raise RuntimeError(f"band {turn_band[t]} 放不下：需要 {need} token、只有 {e - s}")
        us_ = sorted(rng.randint(0, avail) for _ in us)
        off = 0
        for j, (u, d) in enumerate(zip(us, us_)):
            targets.append((s + d + off, t, j, u))
            off += nl[t][j] + gap
    targets.sort(key=lambda x: x[0])

    out = list(pre)
    spans = []
    ti, fi, doc_i = 0, 0, 1
    while True:
        while ti < len(targets) and len(out) >= targets[ti][0]:
            _, t, j, u = targets[ti]
            ids = enc(render(u.payload, doc_i))
            spans.append(("needle", (t, j), len(out), len(out) + len(ids)))
            out += ids
            doc_i += 1
            ti += 1
        if ti == len(targets) and len(out) >= total:
            break
        if fi >= len(fillers):
            raise RuntimeError("filler 用完了，還沒補滿")
        u = fillers[fi]
        ids = enc(render(u.payload, doc_i))
        if u.kind != "filler":
            spans.append((u.kind, u.tag, len(out), len(out) + len(ids)))
        out += ids
        doc_i += 1
        fi += 1
    out = out[:total]
    for kind, tag, s, e in spans:
        if e > total and kind != "filler":
            raise RuntimeError(f"{kind} {tag} 被截斷（結束於 {e} > {total}）")
    return out, spans


def _draw_fillers(enc: _Enc, make, render, total: int) -> list[_Unit]:
    """一直抽 filler，直到總長 ≥ total（每次批次 tokenize）。"""
    out, acc = [], 0
    while acc < total + 4 * BLOCK:
        batch = [make() for _ in range(64)]
        lens = [len(x) for x in enc.many([render(u.payload, 999) for u in batch])]
        out += batch
        acc += sum(lens)
    return out


def _blocks(s: int, e: int, block: int) -> set[int]:
    return set(range(s // block, (e - 1) // block + 1))


# ── 各任務的內容 ──────────────────────────────────────────────────────────

def _uuid(rng):
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def _num7(rng):
    return str(rng.randint(10 ** 6, 10 ** 7 - 1))


def _ident(p, i):
    return p


def _niah(task, turn_bands, seed, p, enc, n_blocks, block, attempt):
    crng = random.Random(f"m10|{task}|content|{seed}")
    frng = random.Random(f"m10|{task}|filler|{seed}")
    prng = random.Random(f"m10|{task}|pos|{'-'.join(turn_bands)}|{seed}|{attempt}")
    total = n_blocks * block
    adjs, nouns = _adjs_nouns()
    if task == "T1":
        k_type, v_type, n_v = "uuids", "uuids", 1
        mk_k, mk_v = (lambda r: _uuid(r)), (lambda r: _uuid(r))
    else:
        k_type, v_type, n_v = "words", "numbers", int(p["n_values"])
        mk_k, mk_v = (lambda r: f"{r.choice(adjs)}-{r.choice(nouns)}"), _num7
    used = set()

    def fresh(fn, r):
        while True:
            x = fn(r)
            if x not in used and x.lower() not in _FIXED_TEXT:
                used.add(x)
                return x

    nq = int(p.get("n_queries", 1)) if task == "T1" else 1
    if nq < 1:
        raise ValueError("n_queries 至少 1")
    # 抽取順序和加參數之前相同（nq＝1 時：每輪先一個鍵，再那一輪的值），預設輸出逐一不變
    keys = [[fresh(mk_k, crng) for _ in range(nq)] for _ in turn_bands]
    vals = [[fresh(mk_v, crng) for _ in range(nq * n_v)] for _ in turn_bands]
    needles = {}
    for t, (ks, vs) in enumerate(zip(keys, vals)):
        pairs = list(zip(ks, vs)) if nq > 1 else [(ks[0], v) for v in vs]   # 多鍵：每個鍵一個值；單鍵：一個鍵 n_v 個值
        us = [_Unit(NEEDLE_FMT.format(type_needle_v=v_type, key=k, value=v) + "\n", "needle", (t, j))
              for j, (k, v) in enumerate(pairs)]
        random.Random(f"m10|{task}|order|{seed}|{t}").shuffle(us)   # 上游：needle 順序打散
        needles[t] = us

    haystack = p.get("haystack", "noise") if task == "T1" else "noise"
    if haystack == "needle":
        def make():
            return _Unit(NEEDLE_FMT.format(type_needle_v=v_type, key=fresh(mk_k, frng), value=fresh(mk_v, frng)) + "\n")
    elif haystack == "noise":
        def make():
            return _Unit(NOISE + "\n")
    else:
        raise ValueError(f"haystack={haystack}")
    fillers = _draw_fillers(enc, make, _ident, total)
    extras = [[_Unit(NEEDLE_FMT.format(type_needle_v=v_type, key=fresh(mk_k, crng), value=fresh(mk_v, crng)) + "\n",
                     "distractor", ("d", i))] for i in range(int(p.get("n_distractors", 0)))]
    # 相似鍵干擾針（10/15 加，上游沒有）：每個被問的鍵配 n_similar 根，鍵＝被問的鍵的前 similar_prefix 個字元＋
    # 一個新 UUID 的其餘字元（UUID 格式不變），值是新的 UUID；和一般干擾一樣插在 filler 序列的隨機位置（可能在答案帶內）
    n_sim, pref = int(p.get("n_similar", 0)), int(p.get("similar_prefix", 0))
    similar = [[] for _ in turn_bands]
    if task == "T1" and n_sim:
        if not 0 < pref < 36:
            raise ValueError("similar_prefix 要在 1–35（UUID 36 個字元）")
        srng = random.Random(f"m10|{task}|similar|{seed}")
        for t, ks in enumerate(keys):
            for qi, k in enumerate(ks):
                for i in range(n_sim):
                    sk = fresh(lambda r: k[:pref] + _uuid(r)[pref:], srng)
                    similar[t].append(sk)
                    extras.append([_Unit(NEEDLE_FMT.format(type_needle_v=v_type, key=sk, value=fresh(mk_v, srng)) + "\n",
                                         "distractor", ("sim", t, qi, i))])

    single = n_v * nq == 1                        # 上游：num_needle_q × num_needle_v == 1 才用單數模板
    tv = v_type[:-1] if single else v_type
    S = _singular if single else (lambda s: s)
    instr = S(NIAH_INSTR).format(type_needle_v=tv)
    ctx, spans = _assemble(enc, instr, fillers, needles, dict(enumerate(turn_bands)), extras, _ident,
                           total, block, prng)
    turns = []
    for t, (ks, vs) in enumerate(zip(keys, vals)):
        k = ks[0] if nq == 1 else ", ".join(ks[:-1]) + ", and " + ks[-1]   # 上游 multiquery 的問法
        q = S(NIAH_Q).format(type_needle_v=tv, query=k)
        gold_text = " " + (vs[0] if single else ", ".join(vs[:-1]) + ", and " + vs[-1]) + "."
        d = dict(question=q, prefix=S(NIAH_PREFIX).format(type_needle_v=tv, query=k), gold=list(vs),
                 gold_text=gold_text, metric="string_match_all", key=k)
        if nq > 1:
            d["keys"] = list(ks)
        if similar[t]:
            d["similar_keys"] = list(similar[t])
        turns.append(d)
    return ctx, spans, turns, "\n", instr


def _vt_chain(rng, n_hops, k, used, value=None):
    names = []
    while len(names) < n_hops + 1:
        x = "".join(rng.choices(string.ascii_uppercase, k=k))
        if x not in used and x.lower() not in _FIXED_TEXT:
            used.add(x)
            names.append(x)
    if value is None:
        while True:
            value = str(rng.randint(10000, 99999))
            if value not in used:
                used.add(value)
                break
    lines = [f"VAR {names[0]} = {value}"] + [f"VAR {names[j + 1]} = VAR {names[j]} " for j in range(n_hops)]
    return names, value, lines


def _vt(task, turn_bands, seed, p, enc, n_blocks, block, attempt):
    crng = random.Random(f"m10|{task}|content|{seed}")
    prng = random.Random(f"m10|{task}|pos|{'-'.join(turn_bands)}|{seed}|{attempt}")
    total = n_blocks * block
    n_hops = int(p["n_hops"])
    used: set[str] = set()
    chains = [_vt_chain(crng, n_hops, 5, used) for _ in turn_bands]
    dchains = [_vt_chain(crng, n_hops, 5, used) for _ in range(int(p.get("n_distractor_chains", 0)))]

    pre = VT_INSTR
    if p.get("icl", True):
        # 上游的 1-shot：3 個字母的變數、min(10, n_hops) 跳、noise；大小調到「chat 開頭＋範例＋說明 ≤ block」
        irng = random.Random(f"m10|{task}|icl|{seed}")
        inames, ival, ilines = _vt_chain(irng, min(10, n_hops), 3, set(used))
        pos_seed = irng.random()

        def icl_text(n_noise):
            sents = [NOISE] * n_noise
            r = random.Random(pos_seed)
            for pi, j in zip(sorted(r.sample(range(len(sents)), len(ilines))), range(len(ilines))):
                sents.insert(pi + j, ilines[j])
            ctx = "\n".join(sents).replace(". \n", ".\n")
            # 上游：icl_example = input + " " + " ".join(outputs) + "\n"，再接 "\n" 和正式的模板
            return (VT_INSTR + ctx + "\n" + VT_Q.format(query=ival)
                    + VT_PREFIX.format(num_v=len(inames), query=ival) + " " + " ".join(inames) + "\n\n")

        best = None
        for n_noise in range(len(ilines), 200):
            if len(enc.head) + len(enc(icl_text(n_noise) + VT_INSTR)) > block:
                break
            best = n_noise
        if best is None:
            raise RuntimeError("vt 的 1-shot 範例連最小的都放不進 block 0")
        pre = icl_text(best) + VT_INSTR

    fillers = _draw_fillers(enc, lambda: _Unit(NOISE + "\n"), _ident, total)
    needles = {t: [_Unit(line + "\n", "needle", (t, j)) for j, line in enumerate(c[2])]
               for t, c in enumerate(chains)}
    extras = [[_Unit(line + "\n", "distractor", ("dchain", i, j)) for j, line in enumerate(c[2])]
              for i, c in enumerate(dchains)]
    ctx, spans = _assemble(enc, pre, fillers, needles, dict(enumerate(turn_bands)), extras, _ident,
                           total, block, prng)
    turns = []
    for names, value, _ in chains:
        turns.append(dict(question=VT_Q.format(query=value), prefix=VT_PREFIX.format(num_v=n_hops + 1, query=value),
                          gold=list(names), gold_text=" ".join(names), metric="string_match_all", value=value))
    return ctx, spans, turns, "\n", pre


_HQ: dict | None = None


def _hotpot() -> dict:
    """讀 HotpotQA dev distractor（7,405 題），建段落池與可用題目（篩選見檔頭差異）。"""
    global _HQ
    if _HQ is None:
        import pandas as pd
        d = pd.read_parquet(HOTPOT)
        if len(d) != 7405:
            raise RuntimeError(f"HotpotQA dev distractor 應有 7,405 題，讀到 {len(d)}：{HOTPOT}")
        docs, qas = set(), []
        for r in d.itertuples(index=False):
            ctx = {t: f"{t}\n{''.join(s)}" for t, s in zip(r.context["title"], r.context["sentences"])}
            docs.update(ctx.values())
            qas.append(dict(id=r.id, q=r.question, a=r.answer, type=r.type, level=r.level, ctx=ctx,
                            gold_titles=list(dict.fromkeys(r.supporting_facts["title"]))))
        elig = []
        for i, x in enumerate(qas):
            a = x["a"].strip()
            g = [x["ctx"].get(t) for t in x["gold_titles"]]
            if len(g) != 2 or None in g or not a:
                continue
            al = a.lower()
            if al in ("yes", "no") or re.fullmatch(r"\d{1,3}", a) or al in _FIXED_TEXT or al in x["q"].lower():
                continue
            if al not in (g[0] + "\n" + g[1]).lower():
                continue
            elig.append(i)
        _HQ = dict(docs=sorted(docs), qas=qas, elig=elig)
    return _HQ


def _qa(task, turn_bands, seed, p, enc, n_blocks, block, attempt):
    hq = _hotpot()
    crng = random.Random(f"m10|{task}|content|{seed}")
    frng = random.Random(f"m10|{task}|filler|{seed}")
    prng = random.Random(f"m10|{task}|pos|{'-'.join(turn_bands)}|{seed}|{attempt}")
    total = n_blocks * block
    qs = []
    while len(qs) < len(turn_bands):
        x = hq["qas"][crng.choice(hq["elig"])]
        # 10/15：E0 調難度時發現答案 ≥3 個字的題目多半是被 string_match_part 判錯（"Steve Coogan" 對 "Stephen John Coogan"），
        # 不是模型不會；限制答案長度讓計分可靠（m10_pilot_mi300x/e0_tune，見 RUNLOG Milestone 10-pilot）
        if p["max_answer_words"] and len(x["a"].split()) > p["max_answer_words"]:
            continue
        ok = True
        for y in qs:
            gx = "\n".join(x["ctx"][t] for t in x["gold_titles"]).lower()
            gy = "\n".join(y["ctx"][t] for t in y["gold_titles"]).lower()
            if (x["id"] == y["id"] or set(x["gold_titles"]) & set(y["gold_titles"])
                    or x["a"].lower() in gy + y["q"].lower() or y["a"].lower() in gx + x["q"].lower()
                    or x["a"].lower() in y["a"].lower() or y["a"].lower() in x["a"].lower()):
                ok = False
        if ok:
            qs.append(x)
    gold_titles = {t for x in qs for t in x["gold_titles"]}
    gold_docs = {x["ctx"][t] for x in qs for t in x["gold_titles"]}
    answers = [x["a"].lower() for x in qs]

    def usable(doc):
        return (doc.split("\n", 1)[0] not in gold_titles and doc not in gold_docs
                and not any(a in doc.lower() for a in answers))

    seen: set[str] = set()

    def make():
        while True:
            doc = frng.choice(hq["docs"])
            if doc not in seen and usable(doc):
                seen.add(doc)
                return _Unit(doc)

    def render(doc, i):
        return DOC_FMT.format(i=i, document=doc) + "\n\n"

    fillers = _draw_fillers(enc, make, render, total)
    needles = {}
    for t, x in enumerate(qs):
        g = [x["ctx"][tt] for tt in x["gold_titles"]]
        random.Random(f"m10|{task}|order|{seed}|{t}").shuffle(g)
        needles[t] = [_Unit(doc, "needle", (t, j)) for j, doc in enumerate(g)]
    extras = []
    nh = int(p.get("n_hard_distractors", 0))
    for t, x in enumerate(qs):
        own = [d for tt, d in sorted(x["ctx"].items()) if tt not in x["gold_titles"] and usable(d)]
        random.Random(f"m10|{task}|hard|{seed}|{t}").shuffle(own)
        extras += [[_Unit(d, "distractor", ("hard", t, j))] for j, d in enumerate(own[:nh])]
    ctx, spans = _assemble(enc, QA_INSTR, fillers, needles, dict(enumerate(turn_bands)), extras, render,
                           total, block, prng)
    turns = []
    for x in qs:
        turns.append(dict(question=QA_Q.format(query=x["q"]), prefix=QA_PREFIX, gold=[x["a"]],
                          gold_text=" " + x["a"], metric="string_match_part", hotpot_id=x["id"],
                          hotpot_type=x["type"], hotpot_level=x["level"],
                          gold_docs=[x["ctx"][tt] for tt in x["gold_titles"]]))
    return ctx, spans, turns, "\n\n", QA_INSTR


def _cwe(task, seed, p, enc, n_blocks, block):
    total = n_blocks * block
    words = _cwe_vocab()
    num_cw, fcw, fucw = int(p["num_cw"]), int(p["freq_cw"]), int(p["freq_ucw"])
    crng = random.Random(f"m10|{task}|content|{seed}")
    q_text = CWE_Q.replace("10", str(num_cw))
    pf_text = CWE_PREFIX.replace("10", str(num_cw))

    def example(r, srng, n, rep_c, rep_u):
        full = r.sample(words, min(n, len(words)))
        common, uncommon = full[:num_cw], full[num_cw:]
        lst = common * rep_c + uncommon * rep_u
        srng.shuffle(lst)
        return lst, common

    sl, sc = example(crng, random.Random(f"m10|{task}|shotshuf|{seed}"), 40, 10, 3)   # 上游 ≥4K 的 1-shot
    shot = (CWE_INSTR.replace("10", str(num_cw)) + " ".join(f"{i + 1}. {w}" for i, w in enumerate(sl)) + "\n"
            + q_text + pf_text + " " + " ".join(f"{i + 1}. {w}" for i, w in enumerate(sc)))
    pre = shot + "\n" + CWE_INSTR
    pre_ids = enc.head + enc(pre)

    def main_list(n):
        r = random.Random(f"m10|{task}|main|{seed}|{n}")
        lst, common = example(r, random.Random(f"m10|{task}|mainshuf|{seed}|{n}"), n, fcw, fucw)
        units = [f"{i + 1}. {w}" if i == 0 else f" {i + 1}. {w}" for i, w in enumerate(lst)]
        lens = [len(x) for x in enc.many(units)]
        return lst, common, units, lens

    # 找最小的 n（不同詞數）使清單補得滿 total，之後只截掉尾巴幾個項目
    lo, hi = num_cw + 1, len(words)
    if len(pre_ids) + sum(main_list(hi)[3]) < total:
        raise RuntimeError("cwe 詞表不夠長，補不滿")
    while lo < hi:
        mid = (lo + hi) // 2
        if len(pre_ids) + sum(main_list(mid)[3]) >= total:
            hi = mid
        else:
            lo = mid + 1
    lst, common, units, lens = main_list(lo)
    out = list(pre_ids)
    spans = []
    cset = set(common)
    for w, u in zip(lst, units):
        if len(out) >= total:
            break
        ids = enc(u)
        if w in cset:
            spans.append(("needle", (0, w), len(out), len(out) + len(ids)))
        out += ids
    out = out[:total]
    spans = [(k, t, s, min(e, total)) for k, t, s, e in spans if s < total]
    turns = [dict(question=q_text, prefix=pf_text, gold=list(common),
                  gold_text=" " + " ".join(f"{i + 1}. {w}" for i, w in enumerate(common)),
                  metric="string_match_all", n_distinct_words=lo)]
    return out, spans, turns, "\n", pre


# ── 對外介面 ──────────────────────────────────────────────────────────────

def make_item(task: str, band1: str | None, band2: str | None, seed: int, tok, params: dict | None = None,
              n_blocks: int = N_BLOCKS, block: int = BLOCK) -> dict:
    """產生一題。task：'T1'–'T5' 或上游名稱。band：'front'/'mid'/'back'；T5 兩個都給 None。

    回傳欄位見檔頭；context_ids 長度剛好 n_blocks*block。第 2 輪（band2 不是 None）的完整序列：
    context_ids + turns[0]['query_ids'] + turns[1]['history_ids'] + turns[1]['query_ids']（見 full_prompt_ids）。
    """
    task = _ALIAS.get(task, task)
    if task not in TASKS:
        raise KeyError(f"未知任務 {task}；可用：{list(TASKS)}")
    p = dict(DEFAULT_PARAMS[task])
    for k, v in (params or {}).items():
        if k not in p:
            raise KeyError(f"{task} 沒有參數 {k}；可用：{sorted(p)}")
        p[k] = v
    enc = _enc(tok)
    if task == "T5":
        if band1 is not None or band2 is not None:
            raise ValueError("T5 是全域任務，沒有位置帶（band1、band2 都給 None）")
        ctx, spans, turns, sep, pre = _cwe(task, seed, p, enc, n_blocks, block)
        tb = ["global"]
    else:
        if band1 not in BANDS or (band2 is not None and band2 not in BANDS):
            raise ValueError(f"band 必須是 {list(BANDS)}")
        if band2 == band1:
            raise ValueError("第 1、2 輪的資訊要在不同的位置帶")
        tb = [band1] + ([band2] if band2 else [])
        fn = {"T1": _niah, "T2": _niah, "T3": _vt, "T4": _qa}[task]
        err = None
        for attempt in range(10):     # 位置抽到放不下時換一組位置（內容不變），可重現
            try:
                ctx, spans, turns, sep, pre = fn(task, tb, seed, p, enc, n_blocks, block, attempt)
                for kind, (t, *_), s, e in [x for x in spans if x[0] == "needle"]:
                    lo, hi = bands(n_blocks)[tb[t]]
                    if not _blocks(s, e, block) <= set(range(lo, hi + 1)):
                        raise RuntimeError(f"needle {t} 落在 block {sorted(_blocks(s, e, block))}，不在 {tb[t]}")
                break
            except RuntimeError as ex:
                err = ex
        else:
            raise RuntimeError(f"{task} {tb} seed={seed}：10 次都放不好：{err}")

    total = n_blocks * block
    assert len(ctx) == total, (len(ctx), total)
    out_turns = []
    for t, tr in enumerate(turns):
        nb = sorted(set().union(*[_blocks(s, e, block) for k, tag, s, e in spans
                                  if k == "needle" and tag[0] == t]))
        q_ids = (enc((sep if t == 0 else "") + tr["question"]) + enc.u2a + enc(tr["prefix"]))
        d = dict(turn=t + 1, band=tb[t], query_ids=q_ids, gold=tr["gold"], gold_text=tr["gold_text"],
                 metric=tr["metric"], needle_blocks=nb, question=tr["question"])
        if t > 0:
            d["history_ids"] = enc(turns[t - 1]["gold_text"]) + enc.a2u
        for k in ("key", "keys", "similar_keys", "value", "hotpot_id", "hotpot_type", "hotpot_level", "gold_docs",
                  "n_distinct_words"):
            if k in tr:
                d[k] = tr[k]
        out_turns.append(d)
    item_id = f"{task}_{'-'.join(tb)}_s{seed}"
    return dict(item_id=item_id, task=task, task_name=TASKS[task], seed=seed, params=p,
                model_family=enc.family, tokenizer=getattr(tok, "name_or_path", ""),
                n_blocks=n_blocks, block=block, context_ids=ctx, turns=out_turns,
                max_new_tokens_upstream=TOKENS_TO_GENERATE[task],
                pre_tokens=len(enc.head) + len(enc(pre)),
                spans=[[k, list(tag) if isinstance(tag, tuple) else tag, s, e] for k, tag, s, e in spans
                       if k != "filler"])


def full_prompt_ids(item: dict, turn: int = 1) -> list[int]:
    """第 turn 輪（1 起算）作答前的完整 token 序列。"""
    ids = list(item["context_ids"])
    for t in range(turn):
        tr = item["turns"][t]
        ids += tr.get("history_ids", []) + tr["query_ids"]
    return ids


# ── 自測 ───────────────────────────────────────────────────────────────────

def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 的 byte-level 字母表（Llama-3、Qwen2 的 tokenizer 都用 ByteLevel）。"""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


_BYTE_DEC = {v: k for k, v in _bytes_to_unicode().items()}


class _Locator:
    """token → 原始位元組 → 文字；把「答案字串在解碼文字裡的位置」換回 token 位置（獨立於組裝時的記錄）。"""

    def __init__(self, tok, ids):
        added = {v: k for k, v in tok.get_added_vocab().items()}
        bs = []
        for i, t in zip(ids, tok.convert_ids_to_tokens(ids)):
            bs.append(added[i].encode() if i in added else bytes(_BYTE_DEC[c] for c in t))
        self.cum = [0]
        for b in bs:
            self.cum.append(self.cum[-1] + len(b))
        self.raw = b"".join(bs)
        self.text = self.raw.decode("utf-8", errors="replace")
        self.ok_vs_decode = self.text == tok.decode(ids, skip_special_tokens=False,
                                                    clean_up_tokenization_spaces=False)
        self._c2b = None

    def find(self, s: str, ignore_case: bool = True) -> list[tuple[int, int]]:
        """回傳每個出現位置的 (起 token, 迄 token)（迄不含）。"""
        out = []
        flags = re.IGNORECASE if ignore_case else 0
        for m in re.finditer(re.escape(s), self.text, flags):
            b0 = len(self.text[:m.start()].encode("utf-8", errors="replace"))
            b1 = b0 + len(m.group(0).encode("utf-8", errors="replace"))
            t0 = bisect.bisect_right(self.cum, b0) - 1
            t1 = bisect.bisect_left(self.cum, b1)
            out.append((t0, t1))
        return out


def _check_item(it: dict, tok, again: dict | None) -> list[str]:
    """回傳失敗訊息（空＝全部通過）。"""
    errs = []
    B = it["block"]
    total = it["n_blocks"] * B
    ctx = it["context_ids"]
    if len(ctx) != total:
        errs.append(f"len(context_ids)={len(ctx)} != {total}")
    loc = _Locator(tok, ctx)
    if not loc.ok_vs_decode:
        errs.append("token→bytes 的對照和 tok.decode 不同（定位不可信）")
    if again is not None and json.dumps(it, sort_keys=True) != json.dumps(again, sort_keys=True):
        errs.append("同 seed 兩次產生不同")
    task = it["task"]
    for tr in it["turns"]:
        band = tr["band"]
        bb = bands(it["n_blocks"])
        allowed = set(range(1, it["n_blocks"])) if band == "global" else set(range(bb[band][0], bb[band][1] + 1))
        if not tr["needle_blocks"] or not set(tr["needle_blocks"]) <= allowed:
            errs.append(f"turn{tr['turn']} needle_blocks {tr['needle_blocks']} 不在 {band}")
        if 0 in tr["needle_blocks"]:
            errs.append(f"turn{tr['turn']} needle 在 block 0")
        qtext = tok.decode(tr["query_ids"], skip_special_tokens=False)
        if score(tr["metric"], tr["gold_text"], tr["gold"]) != 1.0 or score(tr["metric"], "", tr["gold"]) != 0.0:
            errs.append(f"turn{tr['turn']} 標準答案自己計分不是 1（或空字串不是 0）")
        if (tr["turn"] == 1) != ("history_ids" not in tr):
            errs.append(f"turn{tr['turn']} history_ids 的有無不對")
        for g in tr["gold"]:
            if g.lower() in qtext.lower():
                errs.append(f"turn{tr['turn']} 答案 {g!r} 出現在問題裡")
            if task == "T5":
                continue
            occ = loc.find(g)
            if not occ:
                errs.append(f"turn{tr['turn']} 答案 {g!r} 不在 context 裡")
            for t0, t1 in occ:
                bl = _blocks(t0, t1, B)
                if not bl <= allowed:
                    errs.append(f"turn{tr['turn']} 答案 {g!r} 出現在 block {sorted(bl)}（帶 {band} 之外）")
        if task == "T4":
            for doc in tr["gold_docs"]:
                occ = loc.find(doc, ignore_case=False)
                if len(occ) != 1:
                    errs.append(f"turn{tr['turn']} gold 段落出現 {len(occ)} 次（應為 1）：{doc[:40]!r}")
                for t0, t1 in occ:
                    if not _blocks(t0, t1, B) <= allowed:
                        errs.append(f"turn{tr['turn']} gold 段落在 block {sorted(_blocks(t0, t1, B))}，不在 {band}")
        if task in ("T1", "T2"):
            for key in tr.get("keys") or [tr["key"]]:
                if not loc.find(key):
                    errs.append(f"turn{tr['turn']} 鍵 {key!r} 不在 context 裡")
                for t0, t1 in loc.find(key):
                    if not _blocks(t0, t1, B) <= allowed:
                        errs.append(f"turn{tr['turn']} 鍵出現在帶外 block {sorted(_blocks(t0, t1, B))}")
        if task == "T1":
            for g in tr["gold"]:                  # 每個答案（值）在 context 裡只出現一次
                if len(loc.find(g)) != 1:
                    errs.append(f"turn{tr['turn']} 答案 {g!r} 出現 {len(loc.find(g))} 次（應為 1）")
            pref = it["params"].get("similar_prefix", 0)
            qkeys = tr.get("keys") or [tr["key"]]
            for sk in tr.get("similar_keys", []):  # 相似鍵：和某個被問的鍵共用前綴、但不相同、確實在 context 裡
                if sk in qkeys or not any(sk[:pref] == k[:pref] for k in qkeys):
                    errs.append(f"turn{tr['turn']} 相似鍵 {sk!r} 沒有共用前綴或等於被問的鍵")
                if len(loc.find(sk)) != 1:
                    errs.append(f"turn{tr['turn']} 相似鍵 {sk!r} 出現 {len(loc.find(sk))} 次（應為 1）")
    if task == "T5":
        tr = it["turns"][0]
        i0 = loc.text.rfind(CWE_INSTR)
        lst_text = loc.text[i0 + len(CWE_INSTR):]
        ents = [e.strip() for e in re.split(r"(?:^|\s)\d+\. ", lst_text) if e.strip()]
        cnt = Counter(ents)
        gold = set(tr["gold"])
        top = {w for w, _ in cnt.most_common(len(gold))}
        mg = min(cnt[w] for w in gold)
        mo = max((c for w, c in cnt.items() if w not in gold), default=0)
        if top != gold or mg <= mo:
            errs.append(f"T5 詞頻前 {len(gold)} 名和答案不同（答案最少 {mg} 次、其他最多 {mo} 次）")
        start_tok = bisect.bisect_right(loc.cum, len(loc.text[:i0 + len(CWE_INSTR)].encode())) - 1
        if start_tok < B:
            errs.append(f"T5 的主清單從 token {start_tok} 開始（在 block 0）")
    # needle span（組裝時的記錄）解碼後真的包含答案
    if task in ("T1", "T2"):
        for tr in it["turns"]:
            t = tr["turn"] - 1
            sp = [x for x in it["spans"] if x[0] == "needle" and x[1][0] == t]
            txt = "".join(tok.decode(ctx[s:e]) for _, _, s, e in sp)
            for g in tr["gold"]:
                if g not in txt:
                    errs.append(f"turn{tr['turn']} needle span 解碼後找不到 {g!r}")
    nd = sum(1 for x in it["spans"] if x[0] == "distractor")
    n_sim_total = (it["params"].get("n_similar", 0) * it["params"].get("n_queries", 1) * len(it["turns"])
                   if task == "T1" else 0)
    want = {"T1": (it["params"].get("n_distractors", 0) if it["params"].get("haystack") == "noise" else 0) + n_sim_total,
            "T2": it["params"].get("n_distractors", 0),
            "T3": it["params"].get("n_distractor_chains", 0) * (it["params"].get("n_hops", 0) + 1),
            "T4": None, "T5": 0}[task]
    if want is not None and nd != want:
        errs.append(f"干擾 span 數 {nd} != {want}")
    return errs


def _retok_report(tok, it: dict) -> tuple[int, int]:
    """完整序列（最後一輪）解碼後重新 tokenize，和原 token 比切點。
    回傳（接縫處的不一致數, 接縫以外的不一致數）。接縫＝context 結尾、各輪 query／history 的起訖：
    這些地方是分開 tokenize 的（例如 context 截在 '.'、問題以 '\n' 開頭 → '.' '\n' 而不是 '.\n'），
    所有配置用的是同一串 token，所以不影響比較；接縫以外不准有。"""
    ids = full_prompt_ids(it, len(it["turns"]))
    a = _Locator(tok, ids)
    re_ids = tok(a.text, add_special_tokens=False)["input_ids"]
    if re_ids == ids:
        return 0, 0
    b = _Locator(tok, re_ids)
    seams = [len(it["context_ids"])]
    for tr in it["turns"]:
        if "history_ids" in tr:
            seams.append(seams[-1] + len(tr["history_ids"]))
        seams.append(seams[-1] + len(tr["query_ids"]))
    sb = [a.cum[x] for x in seams]
    diff = set(a.cum) ^ set(b.cum)
    near = [x for x in diff if min(abs(x - y) for y in sb) <= 8]
    return len(near), len(diff) - len(near)


def _summary(it: dict, tok) -> str:
    ctx = it["context_ids"]
    B = it["block"]
    lines = [f"── {it['item_id']} [{it['model_family']}] pre_tokens={it['pre_tokens']} params={it['params']}"]
    b0 = tok.decode(ctx[:B], skip_special_tokens=False)
    lines.append("  block0[:200]: " + repr(b0[:200]))
    lines.append("  block0[-120:]: " + repr(b0[-120:]))
    for tr in it["turns"]:
        t = tr["turn"] - 1
        sp = [x for x in it["spans"] if x[0] == "needle" and x[1][0] == t]
        if sp:
            s, e = sp[0][2], sp[0][3]
            near = tok.decode(ctx[max(0, s - 12):min(len(ctx), e + 12)])
            lines.append(f"  turn{tr['turn']} band={tr['band']} needle_blocks={tr['needle_blocks']} "
                         f"first needle tok {s}–{e}: {near[:300]!r}")
        if "history_ids" in tr:
            lines.append("  history: " + repr(tok.decode(tr["history_ids"], skip_special_tokens=False)))
        lines.append("  query: " + repr(tok.decode(tr["query_ids"], skip_special_tokens=False)[-300:]))
        lines.append(f"  gold: {tr['gold']}  gold_text: {tr['gold_text']!r}")
    lines.append("  context tail: " + repr(tok.decode(ctx[-25:])))
    return "\n".join(lines)


def _self_test(seeds: int, models: list[str], show: bool) -> int:
    import time
    n_ok = n_bad = 0
    retok = []
    for m in models:
        tok = load_tokenizer(m)
        print(f"=== {m}: {tok.name_or_path}  family={_family(tok)}  vocab={len(tok)}", flush=True)
        e = _enc(tok)
        print(f"  head={e.head_text!r}\n  u2a={e.u2a_text!r}\n  a2u={e.a2u_text!r}")
        combos = [(b, None) for b in BANDS] + [(a, b) for a in BANDS for b in BANDS if a != b]
        for task in TASKS:
            t0 = time.time()
            cs = [(None, None)] if task == "T5" else combos
            shown = False
            for b1, b2 in cs:
                for seed in range(seeds):
                    it = make_item(task, b1, b2, seed, tok)
                    again = make_item(task, b1, b2, seed, tok)
                    errs = _check_item(it, tok, again)
                    seam, off = _retok_report(tok, it)
                    retok.append((m, task, seam, off))
                    if off:
                        errs.append(f"重新 tokenize 在接縫以外有 {off} 個切點不同")
                    if errs:
                        n_bad += 1
                        print(f"  🔴 {it['item_id']}: " + " | ".join(errs[:6]))
                    else:
                        n_ok += 1
                    if show and not shown and (b2 is not None or task == "T5"):
                        print(_summary(it, tok))
                        shown = True
            print(f"  {task} ({TASKS[task]}): {len(cs) * seeds} 題  {time.time() - t0:.1f}s", flush=True)
        # 難度參數的非預設值也要能產生、也要過檢查
        for task, prm in [("T1", {"haystack": "noise", "n_distractors": 20}), ("T1", {"n_queries": 4}),
                          ("T1", {"n_similar": 20, "similar_prefix": 24}),
                          ("T1", {"n_queries": 2, "n_similar": 10, "similar_prefix": 30}),
                          ("T2", {"n_values": 1, "n_distractors": 8}),
                          ("T2", {"n_values": 6}), ("T3", {"n_hops": 6, "n_distractor_chains": 2}),
                          ("T3", {"icl": False}), ("T4", {"n_hard_distractors": 8}), ("T5", {"freq_cw": 20})]:
            b1, b2 = (None, None) if task == "T5" else ("mid", "front")
            it = make_item(task, b1, b2, 7, tok, prm)
            errs = _check_item(it, tok, make_item(task, b1, b2, 7, tok, prm))
            if errs:
                n_bad += 1
                print(f"  🔴 {it['item_id']} {prm}: " + " | ".join(errs[:6]))
            else:
                n_ok += 1
                print(f"  params {task} {prm}: OK")
    print(f"\nHotpotQA 可用題目：{len(_hotpot()['elig'])} / {len(_hotpot()['qas'])}；段落池 {len(_hotpot()['docs'])} 段")
    seam = Counter((m, t) for m, t, x, _ in retok if x)
    tot = Counter((m, t) for m, t, _, _ in retok)
    print("重新 tokenize：接縫處有不同切點的題數（不影響比較，見 _retok_report）："
          + (", ".join(f"{m}/{t}={seam[(m, t)]}/{c}" for (m, t), c in sorted(tot.items()))))
    print(f"  接縫以外不同的題數：{sum(1 for *_, o in retok if o)}（應為 0，已計入失敗）")
    print(f"\n自測：通過 {n_ok}、失敗 {n_bad}")
    return 1 if n_bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--models", default="llama,qwen")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--show", nargs="+", metavar=("TASK", "BAND"), help="印一題：TASK [BAND1 [BAND2]]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="llama")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(_self_test(a.seeds, a.models.split(","), not a.no_show))
    if a.show:
        tok = load_tokenizer(a.model)
        task, bands = a.show[0], a.show[1:] + [None, None]
        it = make_item(task, bands[0], bands[1], a.seed, tok)
        print(_summary(it, tok))
        errs = _check_item(it, tok, None)
        print("檢查：" + ("通過" if not errs else " | ".join(errs)))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
