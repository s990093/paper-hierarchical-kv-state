#!/usr/bin/env python3
"""M5-(c) 長上下文**理解**：LongBench 與 RULER 上的 KV 精度階 ε。

## 這支腳本補的是哪一塊

`m5_quality.py` 已經量到 ε 的兩個端點：

| | 任務 | BF16 → INT4 |
|---|---|---|
| (a) 推理 | GSM8K many-shot（n=1,000） | 77.9% → 75.1%（**與 0 無法區分**） |
| (b) 檢索 | 大海撈針（單鍵、32K） | 100% → 0%（**全失**） |

兩者之間缺的是 **(c) 理解／整合**：上下文很長，但答案不是某個可以「撈」出來的
字串，而要跨全文做多跳追蹤或聚合。這支腳本量這一塊，用兩個互補的來源：

* **LongBench**（THUDM，真實文件）——單文件 QA、多跳 QA、摘要、few-shot 分類、
  合成檢索。優點是真實文本與公認的計分；缺點是上下文長度由資料決定（中位 5K–15K）。
* **RULER**（NVIDIA，合成）——上下文長度可控且對齊 (b) 的掃描，
  任務族從「多鍵檢索」一路張到「全文詞頻聚合」。見 `ruler_tasks.py`。

兩者的**協定與 (a)(b) 完全相同**：同一個模型、同一張卡、同一組 KV 精度、
`temperature=0`、`seed` 固定。**唯一變動的是 `--kv-cache-dtype`。**
因此跨精度的比較是乾淨的；與公開榜單的絕對值比較則不是（見 §「與公開數字的差異」）。

## 與公開數字的差異（**必須寫進論文，不得省略**）

1. 模型是 `Qwen2.5-7B-Instruct-1M` 的 **AWQ-INT4 權重、且移除 DCA** 的變體
   （vLLM 0.28 的 V1 engine 沒有可用的 DCA 路徑），不是原始 BF16 權重。
2. 每個任務取**前 n 筆**（預設 50），不是 LongBench 的全量 150–200 筆。
3. `max_model_len = 32,768`，超過者照 LongBench 的做法**從中間截斷**。
4. RULER 的 haystack 一律用 `noise`，不用需要爬網的 Paul Graham essay
   （`ruler_tasks.py` 檔頭有完整差異清單）。

**這些差異對「跨精度的相對變化」沒有影響**——四個設定吃的是同一批 prompt。

## 用法

    python code/m5_understanding.py --suite longbench --gpu 0
    python code/m5_understanding.py --suite ruler     --gpu 1 --ctx 16384
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import string
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m5_quality as q                                            # noqa: E402
from gpu_guard import GpuWatcher, host_contention, wait_until_free  # noqa: E402

BIG = Path(os.environ.get("PAPER_HKV_BIG", "/ssd7/hungwei/paper-hkv"))
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results/m5_quality"
LB = BIG / "datasets/longbench"
sys.path.insert(0, os.environ.get("PAPER_HKV_PYLIBS", str(BIG / "pylibs")))

# 平台 A 可用的精度階。`fp8_per_token_head` 需 Triton backend，
# 該後端於 sm_86 拒絕 FP8 KV cache——不是本腳本的限制，是硬體的（見 CLAUDE.md §3）。
CONFIGS = {
    "bf16": ("auto", "無損基準"),
    "fp8": ("fp8", "靜態縮放、未校正"),
    "int8": ("int8_per_token_head", "per-token-head 動態縮放"),
    "int4": ("int4_per_token_head", "per-token-head 動態縮放，最低精度階"),
}

# LongBench 的英文子集。選法：涵蓋「單文件 QA / 多跳 QA / 摘要 / few-shot 分類 /
# 合成檢索」五類，且 BF16 基準不在地板上（`passage_count` 的基準只有個位數，
# 量不出退化，故排除）。中文任務排除——模型的中文能力會混進 ε。
LB_TASKS = {
    "multifieldqa_en":      ("單文件 QA", "qa_f1"),
    "qasper":               ("單文件 QA（科學論文）", "qa_f1"),
    "hotpotqa":             ("多跳 QA", "qa_f1"),
    "2wikimqa":             ("多跳 QA", "qa_f1"),
    "gov_report":           ("摘要", "rouge_l"),
    "trec":                 ("few-shot 分類", "classification"),
    "passage_retrieval_en": ("合成檢索", "retrieval"),
}
# 上游 pred.py：這幾個是 few-shot 補完任務，chat 模型**不套** chat template
LB_NO_CHAT = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
# 上游 eval.py：這幾個只取輸出的第一行
LB_FIRST_LINE = {"trec", "triviaqa", "samsum", "lsht"}


# ─────────────────── LongBench 的計分（逐字移植上游 metrics.py） ───────────────────

def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_f1(pred: str, gt: str, **_) -> float:
    p, g = _normalize(pred).split(), _normalize(gt).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def rouge_l(pred: str, gt: str, **_) -> float:
    from rouge import Rouge                       # 上游用的就是這個套件
    try:
        return Rouge().get_scores([pred], [gt], avg=True)["rouge-l"]["f"]
    except Exception:                             # noqa: BLE001  上游也是整段 try/except
        return 0.0


def classification(pred: str, gt: str, all_classes=None, **_) -> float:
    hits = [c for c in (all_classes or []) if c in pred]
    for m in list(hits):
        if m in gt and m != gt:
            hits.remove(m)
    return (1.0 / len(hits)) if gt in hits else 0.0


def retrieval(pred: str, gt: str, **_) -> float:
    m = re.findall(r"Paragraph (\d+)", gt)
    if not m:
        return 0.0
    nums = re.findall(r"\d+", pred)
    return 0.0 if not nums else sum(n == m[0] for n in nums) / len(nums)


LB_METRIC = {"qa_f1": qa_f1, "rouge_l": rouge_l,
             "classification": classification, "retrieval": retrieval}


# ─────────────────── 測資組裝 ───────────────────

def build_longbench(tok, n_per_task: int, max_model_len: int) -> list[dict]:
    prompts = json.loads((LB / "config/dataset2prompt.json").read_text())
    maxlen = json.loads((LB / "config/dataset2maxlen.json").read_text())
    cases = []
    for t in LB_TASKS:
        rows = [json.loads(l) for l in (LB / f"data/{t}.jsonl").open()][:n_per_task]
        gen = maxlen[t]
        budget = max_model_len - gen
        for i, r in enumerate(rows):
            text = prompts[t].format(**r)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            trunc = len(ids) > budget
            if trunc:                              # 上游 pred.py：從**中間**截斷
                half = budget // 2
                text = (tok.decode(ids[:half], skip_special_tokens=True)
                        + tok.decode(ids[-half:], skip_special_tokens=True))
            if t not in LB_NO_CHAT:
                text = tok.apply_chat_template([{"role": "user", "content": text}],
                                               tokenize=False, add_generation_prompt=True)
            cases.append({
                "task": t, "idx": i, "prompt": text, "answers": r["answers"],
                "all_classes": r.get("all_classes"), "metric": LB_TASKS[t][1],
                "max_new_tokens": gen, "truncated": int(trunc),
                "prompt_tokens": len(tok(text, add_special_tokens=False)["input_ids"]),
            })
    return cases


def build_ruler(tok, n_per_task: int, ctx: int, seed: int) -> list[dict]:
    import ruler_tasks as rt
    cases = []
    for t in rt.RULER_TASKS:
        for i, s in enumerate(rt.build(t, ctx, n_per_task, tok, seed=seed)):
            cases.append({"task": t, "idx": i, "prompt": s["prompt"],
                          "answers": s["answers"], "all_classes": None,
                          "metric": "string_match_all",
                          "max_new_tokens": s["max_new_tokens"], "truncated": 0,
                          "prompt_tokens": s["prompt_tokens"]})
    return cases


def score_case(c: dict, pred: str) -> float:
    if c["metric"] == "string_match_all":
        import ruler_tasks as rt
        return rt.string_match_all(pred, c["answers"])
    if c["task"] in LB_FIRST_LINE:                 # 上游 eval.py 的後處理
        pred = pred.lstrip("\n").split("\n")[0]
    fn = LB_METRIC[c["metric"]]
    return max((fn(pred, g, all_classes=c["all_classes"]) for g in c["answers"]),
               default=0.0)


# ─────────────────── 執行 ───────────────────

def ask(port: int, model: str, prompt: str, max_tokens: int) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0.0, "seed": 12345}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)["choices"][0]["text"]


def run_config(name: str, kv_dtype: str, desc: str, gpu: int, cases: list[dict],
               max_len: int, root: Path, run_id: str, suite: str,
               watcher=None) -> list[dict]:
    rows: list[dict] = []
    out = root / name
    print(f"\n[m5c] === {name}（{desc}）===", flush=True)
    try:
        with q.Server(gpu, max_len, out, kv_dtype=kv_dtype) as s:
            print(f"[m5c]   server up, GPU KV = {s.kv_tokens:,} tokens" if s.kv_tokens
                  else "[m5c]   server up", flush=True)
            tot: dict[str, list[float]] = defaultdict(list)
            for j, c in enumerate(cases):
                t0 = time.perf_counter()
                text = ask(s.port, q.MODEL, c["prompt"], c["max_new_tokens"])
                dt = (time.perf_counter() - t0) * 1000
                sc = score_case(c, text or "")
                tot[c["task"]].append(sc)
                rows.append({
                    "run_id": run_id, "ts": datetime.now().astimezone().isoformat(),
                    "suite": suite, "config": name, "kv_dtype": kv_dtype,
                    "model_key": q.MODEL_KEY, "gpu": gpu, "task": c["task"],
                    "idx": c["idx"], "metric": c["metric"], "score": round(sc, 6),
                    "gold": " | ".join(map(str, c["answers"]))[:300],
                    "pred": (text or "").replace("\n", " ")[:300],
                    "latency_ms": round(dt, 1),
                    "out_sha1": hashlib.sha1((text or "").encode()).hexdigest()[:16],
                    "out_len": len(text or ""),
                    "prompt_tokens": c["prompt_tokens"],
                    "max_new_tokens": c["max_new_tokens"],
                    "truncated": c["truncated"],
                    "gpu_kv_cache_tokens": s.kv_tokens, "desc": desc,
                    # 本卡上截至此列為止看到的外來 process 數。品質分數與爭用無關
                    # （同一批 prompt、temperature=0），但仍逐列留痕以便事後排除。
                    "own_gpu_intruders": len(watcher.intruders) if watcher else "",
                    **{k: v for k, v in host_contention(exclude_gpu=gpu).items()
                       if k in ("level", "foreign_gpu_count", "foreign_max_util")},
                    "log": str(out / "server.log"),
                })
                if (j + 1) % 50 == 0:
                    done = sum(len(v) for v in tot.values())
                    mean = sum(sum(v) for v in tot.values()) / max(1, done)
                    print(f"[m5c]   {j + 1}/{len(cases)}  平均分 {100 * mean:.1f}",
                          flush=True)
            for t, v in tot.items():
                print(f"[m5c]   {t:22s} {100 * sum(v) / len(v):6.2f}  (n={len(v)})")
        # 每個設定跑完就先落地一份到 run 目錄。四個設定要三小時，
        # 中途掛掉不該讓前面的白跑——`results/` 的那份仍在全部跑完才寫。
        if rows:
            with (out / "rows.csv").open("w", newline="") as f:
                wtr = csv.DictWriter(f, fieldnames=list(rows[0]))
                wtr.writeheader()
                wtr.writerows(rows)
    except Exception as e:                          # noqa: BLE001
        print(f"[m5c]   🔴 {type(e).__name__}: {e}")
        out.mkdir(parents=True, exist_ok=True)
        (out / "error.txt").write_text(f"{type(e).__name__}: {e}\n")
    return rows


def summarise(rows: list[dict], suite: str) -> None:
    if not rows:
        print("[m5c] 沒有資料")
        return
    cfgs, tasks = [], []
    for r in rows:
        if r["config"] not in cfgs:
            cfgs.append(r["config"])
        if r["task"] not in tasks:
            tasks.append(r["task"])
    agg: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        agg[(r["config"], r["task"])].append(float(r["score"]))

    base = cfgs[0]
    w = max(len(t) for t in tasks) + 2
    print(f"\n{'=' * (w + 12 * len(cfgs))}")
    print(f"{suite} × KV 精度（分數 = 該任務的官方指標 × 100）")
    print("=" * (w + 12 * len(cfgs)))
    print(f"{'任務':<{w}}" + "".join(f"{c:>12}" for c in cfgs))
    for t in tasks:
        line = f"{t:<{w}}"
        for c in cfgs:
            v = agg.get((c, t))
            line += f"{(100 * sum(v) / len(v)):>12.2f}" if v else f"{'—':>12}"
        print(line)
    print("-" * (w + 12 * len(cfgs)))
    means = {}
    for c in cfgs:
        per = [100 * sum(agg[(c, t)]) / len(agg[(c, t)]) for t in tasks if agg.get((c, t))]
        means[c] = sum(per) / len(per) if per else float("nan")
    print(f"{'巨觀平均':<{w}}" + "".join(f"{means[c]:>12.2f}" for c in cfgs))
    # 單一設定跑（四張卡各跑一個精度）時 base 就是自己；若該設定全 0
    # （int4 在 RULER 上就是），除法會炸。保留率在那種情形沒有意義，印 "—"。
    ref = means[base]
    print(f"{'相對 ' + base + ' 保留':<{w}}"
          + "".join(f"{(f'{100 * means[c] / ref:.1f}%' if ref else '—'):>12}"
                    for c in cfgs))


def write_rows(path: Path, rows: list[dict]) -> None:
    q.write_rows(path, rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=["longbench", "ruler"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model", default="qwen-awq", choices=list(q.MODEL_CHOICES))
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--n-per-task", type=int, default=50)
    ap.add_argument("--ctx", type=int, default=16384, help="RULER 的上下文長度")
    ap.add_argument("--max-model-len", type=int, default=32768,
                    help="LongBench 的視窗；超過者從中間截斷（同上游 pred.py）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="只組 prompt 並印長度統計，不開 server")
    a = ap.parse_args()

    q.MODEL, q.MODEL_KEY = q.MODEL_CHOICES[a.model], a.model
    os.environ.setdefault("HF_HOME", str(BIG / "hf-cache/huggingface"))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(q.MODEL)
    print(f"[m5c] 模型 {a.model} -> {q.MODEL}")

    if a.suite == "longbench":
        cases = build_longbench(tok, a.n_per_task, a.max_model_len)
        max_len = a.max_model_len
    else:
        cases = build_ruler(tok, a.n_per_task, a.ctx, a.seed)
        # 每筆的 vocab 是各自抽的，token 數會在目標長度上下漂移（fwe 最明顯），
        # 所以視窗照**實際最長的那一筆**開，不用 ctx + 固定餘裕——
        # 開太小的話那幾筆會被 server 拒絕，靜默少樣本。
        max_len = max(c["prompt_tokens"] + c["max_new_tokens"] for c in cases) + 64

    per: dict[str, list[int]] = defaultdict(list)
    for c in cases:
        per[c["task"]].append(c["prompt_tokens"])
    print(f"[m5c] {a.suite}：{len(cases)} 筆 / 設定，{len(per)} 個任務")
    for t, v in per.items():
        tr = sum(c["truncated"] for c in cases if c["task"] == t)
        print(f"[m5c]   {t:22s} n={len(v):3d} tok 中位 {sorted(v)[len(v) // 2]:6d} "
              f"max {max(v):6d} 截斷 {tr}")
    print(f"[m5c] 全部 prompt 合計 {sum(sum(v) for v in per.values()):,} tokens／設定")
    print(f"[m5c] max_model_len = {max_len:,}")
    if a.dry_run:
        return 0

    h = host_contention(exclude_gpu=a.gpu)
    print(f"[m5c] 整機爭用：{h['level']}（外來 process {h['foreign_procs']} 個）")
    print("[m5c] ℹ️  品質是分數不是時間——整機爭用不影響分數，只影響 latency_ms。")
    ok, got = wait_until_free(a.gpu, need_mib=22 * 1024, timeout_s=900)
    if not ok:
        print(f"[m5c] 🔴 GPU {a.gpu} 只有 {got} MiB 可用，不開跑。")
        return 5

    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-m5-{a.suite}"
    root = BIG / "runs" / run_id
    rows: list[dict] = []
    # 四個精度設定會被拆到四張卡上平行跑（見 RUNLOG），監看檔要帶卡號，
    # 否則四個 process 互相覆寫，只留下最後一個的紀錄。
    with GpuWatcher(gpu=a.gpu,
                    out_path=str(OUT / f"gpu_guard_{a.suite}_gpu{a.gpu}.json")) as g:
        if not g.started_clean:
            print(f"[m5c] 🔴 GPU {a.gpu} 開跑前就不乾淨：{g.intruders}")
            return 2
        for name in a.configs:
            dtype, desc = CONFIGS[name]
            rows += run_config(name, dtype, desc, a.gpu, cases, max_len,
                               root, run_id, a.suite, watcher=g)

    # 🔴 CLAUDE.md §3 的「污染就作廢」是為了保護**時間**數字：SM 爭用會把延遲
    #    拉高且事後無法修正。這支腳本量的是**分數**——同一批 prompt、
    #    `temperature=0`、固定 seed，外來 process 不改變輸出。因此照
    #    `m5_quality.py` 的既有做法保留結果，但**逐列記錄**爭用狀況
    #    （`own_gpu_intruders`、`level`、`foreign_*`），讓事後能整批排除。
    #    latency_ms 欄在污染下不可用，這一點對本腳本無影響（不用來下結論）。
    if g.contaminated:
        print(f"[m5c] ⚠️  量測期間本卡出現外來 process：{g.intruders}")
        print("[m5c]     分數保留（品質與爭用無關），但 latency_ms 欄作廢。")
        (root / "CONTAMINATED").write_text(json.dumps(g.intruders, default=str))

    path = Path(a.csv) if a.csv else OUT / f"{a.suite}_precision.csv"
    write_rows(path, rows)
    summarise(rows, a.suite)
    print(f"\n[m5c] run_id = {run_id}\n[m5c] 原始 log = {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
