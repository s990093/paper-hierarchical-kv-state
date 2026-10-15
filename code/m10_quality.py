"""m10_quality.py — 精度這條線的品質實驗（03_plan_h200.md §3.3、§3.4）。目前只有 E0（全 BF16）：D1 用來調題目難度

每題（§3.3）：
  1. m10_tasks 產生 prompt：context 剛好 64 個 block（含 chat template 開頭與說明）
  2. chunk 化 prefill（和還原器的重算同一條路徑：m7_model.forward_span，每次 512 token）
  3. 第 1 輪：接問題，greedy 生成
  4. 第 2 輪（§3.4）：歷史固定成標準答案——第 1 輪生成的 KV 直接被覆寫（KV 緩衝是依位置的，
     attention 只看 [0, end)），接 history_ids＋第 2 輪問題，greedy 生成
生成長度（03 §1 寫最多 64，上游是 niah 128、vt 30、qa 32、cwe 120；10/15 還沒決定）：
  生成到 max(64, 上游) 個 token，兩種截法都計分（score_64、score_up），之後決定用哪個不必重跑。

子命令：
  tune   只跑 E0，調難度（§3.2：每個模型、任務族、位置帶的 BF16 正確率要在 60–95%）。
         低精度配置（E1、F25…）這支還沒有：事先登記凍結之前不跑任何低精度的答對率。

輸出：--out 的 CSV（每列有 run_id、ts），另寫 <out>.summary.csv（每個任務、輪、帶的正確率與 CI）。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m10_tasks as T  # noqa: E402

ROTATIONS = [("front", "back"), ("mid", "front"), ("back", "mid")]   # 每個帶當第 1 輪、第 2 輪各一次
CUT = 64                                                              # 03 §1 的生成上限


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


class Answerer:
    """包住 KVModel：prefill context、接 token、greedy 生成。一個 process 一個（見 test_m10_restore 的建立順序問題）。"""

    def __init__(self, model: str, extra: int = 4096):
        os.environ["M7_MODEL_GLOB"] = os.environ.get(f"M7_{model.upper()}_GLOB") or T.MODELS[model]
        from transformers import GenerationConfig
        from m7_model import CHUNK, KVModel, model_path
        self.CHUNK = CHUNK
        self.n_ctx = T.N_BLOCKS * T.BLOCK
        self.km = KVModel(max_len=self.n_ctx + extra)
        self.tok = T.load_tokenizer(os.environ["M7_MODEL_GLOB"])
        gc = GenerationConfig.from_pretrained(model_path())
        eos = gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]
        self.eos = set(e for e in eos + [self.tok.eos_token_id] if e is not None)
        self.model = model

    @torch.inference_mode()
    def feed(self, ids: list[int], start: int):
        """把 ids 接在 start 之後（每段 ≤512），回傳最後一個位置的 logits。"""
        t = torch.tensor(ids, device=self.km.device)
        logits, s = None, 0
        while s < len(ids):
            e = min(len(ids), s + self.CHUNK)
            logits = self.km.forward_span(t[s:e], start + s, want_logits=(e == len(ids)))
            s = e
        return logits

    @torch.inference_mode()
    def generate(self, logits, start: int, max_new: int) -> tuple[list[int], str]:
        out, pos = [], start
        while len(out) < max_new:
            nxt = int(logits.argmax())
            if nxt in self.eos:
                return out, "eos"
            out.append(nxt)
            if len(out) == max_new:
                break
            logits = self.km.forward_span(torch.tensor([nxt], device=self.km.device), pos, want_logits=True)
            pos += 1
        return out, "len"

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)


def run_item_e0(ans: Answerer, it: dict) -> list[dict]:
    rows = []
    ctx = it["context_ids"]
    assert len(ctx) == ans.n_ctx
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ans.km.prefill_chunked(torch.tensor(ctx, device=ans.km.device))
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0
    pos = len(ctx)
    up = it["max_new_tokens_upstream"]
    max_new = max(CUT, up)
    for t, tr in enumerate(it["turns"]):
        t1 = time.perf_counter()
        seq = tr.get("history_ids", []) + tr["query_ids"]
        assert pos + len(seq) + max_new <= ans.km.max_len, (pos, len(seq), max_new, ans.km.max_len)
        logits = ans.feed(seq, pos)
        start_gen = pos + len(seq)
        gen, stop = ans.generate(logits, start_gen, max_new)
        torch.cuda.synchronize()
        p64, pup = ans.decode(gen[:CUT]), ans.decode(gen[:up])
        rows.append(dict(model=ans.model, item_id=it["item_id"], task=it["task"], seed=it["seed"], band=tr["band"],
                         turn=tr["turn"], params=json.dumps(it["params"], sort_keys=True), metric=tr["metric"],
                         gold=json.dumps(tr["gold"], ensure_ascii=False), needle_blocks=json.dumps(tr["needle_blocks"]),
                         n_gen=len(gen), stop=stop, max_new=max_new, upstream_len=up,
                         score_64=T.score(tr["metric"], p64, tr["gold"]),
                         score_up=T.score(tr["metric"], pup, tr["gold"]),
                         f1_64=T.qa_f1(p64, tr["gold"]) if it["task"] == "T4" else "",
                         f1_up=T.qa_f1(pup, tr["gold"]) if it["task"] == "T4" else "",
                         pred=ans.decode(gen), t_prefill=round(t_prefill, 3) if t == 0 else "",
                         t_turn=round(time.perf_counter() - t1, 3)))
        # 下一輪：歷史固定成標準答案。這一輪生成的 KV 留在 [start_gen, …)，下一輪從 start_gen 開始覆寫
        pos = start_gen
    return rows


def plan_items(tasks, n_per_task, seed0):
    """(task, band1, band2, seed) 的清單：T1–T4 依 ROTATIONS 輪流（每個帶當第 1 輪的題數相同），T5 單輪。"""
    out = []
    for task in tasks:
        for k in range(n_per_task):
            if task == "T5":
                out.append((task, None, None, seed0 + k))
            else:
                b1, b2 = ROTATIONS[k % 3]
                out.append((task, b1, b2, seed0 + k))
    return out


def summarize(rows):
    cells = {}
    for r in rows:
        cells.setdefault((r["task"], int(r["turn"]), r["band"]), []).append(r)
    out = []
    for (task, turn, band), v in sorted(cells.items()):
        d = {"task": task, "turn": turn, "band": band, "n": len(v)}
        for k in ("score_64", "score_up", "f1_up"):
            if v[0][k] in ("", None):
                continue
            xs = [float(r[k]) for r in v]
            m = sum(xs) / len(xs)
            sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1))
            d[k] = round(m, 4)
            d[k + "_ci95"] = round(1.96 * sd / math.sqrt(len(xs)), 4)
            d[k + "_in_60_95"] = 0.60 <= m <= 0.95
        d["frac_hit_len_limit"] = round(sum(r["stop"] == "len" for r in v) / len(v), 3)
        out.append(d)
    return out


def tune(a):
    params = json.loads(a.params) if a.params else {}
    ans = Answerer(a.model)
    rid = os.environ.get("RUN_ID", "adhoc")
    watch = None
    try:
        from gpu_guard import GpuWatcher, smi_index
        watch = GpuWatcher(gpu=smi_index(0), own_root=os.getpid(),
                           out_path=os.path.splitext(a.out)[0] + ".gpu_guard.json")
        watch.__enter__()
    except Exception as ex:                     # 品質 run：查不到爭用照樣跑，但要記下來（規則 7：不當成乾淨）
        print("GpuWatcher unavailable:", repr(ex), flush=True)
    items = plan_items(a.tasks, a.n, a.seed0)
    fields = ["run_id", "ts", "model", "item_id", "task", "seed", "band", "turn", "params", "metric", "gold",
              "needle_blocks", "n_gen", "stop", "max_new", "upstream_len", "score_64", "score_up", "f1_64", "f1_up", "pred",
              "t_prefill", "t_turn", "own_gpu_intruders"]
    new = not os.path.exists(a.out)
    f = open(a.out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    if new:
        w.writeheader()
    all_rows = []
    t_start = time.time()
    for k, (task, b1, b2, seed) in enumerate(items):
        it = T.make_item(task, b1, b2, seed, ans.tok, params.get(task))
        rows = run_item_e0(ans, it)
        intr = "NOT_CHECKED" if watch is None else len(watch.intruders)
        for r in rows:
            r.update(run_id=rid, ts=ts(), own_gpu_intruders=intr)
            w.writerow(r)
        f.flush()
        all_rows += rows
        if k % 10 == 0 or k == len(items) - 1:
            el = time.time() - t_start
            print(f"[{k + 1}/{len(items)}] {it['item_id']} "
                  + " ".join(f"t{r['turn']}:{r['score_64']:.2f}/{r['score_up']:.2f}({r['n_gen']},{r['stop']})" for r in rows)
                  + f"  {el / (k + 1):.1f}s/題", flush=True)
    f.close()
    if watch is not None:
        watch.__exit__(None, None, None)
    write_summary(all_rows, os.path.splitext(a.out)[0] + ".summary.csv", rid, a.model, params)


def write_summary(rows, path, rid, model, params):
    summ = summarize(rows)
    keys = []
    for d in summ:                           # 欄位取聯集：T4 才有 f1_*（10/15 修：原本只取第一列的欄位，T3＋T4 同一個 run 時寫摘要失敗）
        keys += [k for k in d if k not in keys]
    with open(path, "w", newline="") as g:
        sw = csv.DictWriter(g, fieldnames=["run_id", "model", "params"] + keys)
        sw.writeheader()
        for d in summ:
            sw.writerow({"run_id": rid, "model": model, "params": json.dumps(params, sort_keys=True), **d})
    for d in summ:
        print(d, flush=True)


def resummarize(a):
    """從已寫好的逐題 CSV 重算摘要（不用 GPU）。"""
    rows = list(csv.DictReader(open(a.csv)))
    for r in rows:
        r["turn"] = int(r["turn"])
    rid = ",".join(sorted({r["run_id"] for r in rows}))
    params = json.loads(rows[0]["params"]) if rows else {}
    write_summary(rows, os.path.splitext(a.csv)[0] + ".summary.csv", rid, rows[0]["model"], {"per_task_params": "see csv"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tune", help="只跑 E0（全 BF16），調難度")
    p.add_argument("--model", choices=list(T.MODELS), required=True)
    p.add_argument("--tasks", nargs="+", default=["T1", "T2", "T3", "T4", "T5"])
    p.add_argument("--n", type=int, default=30, help="每個任務幾題（T1–T4 依三種輪替平均分配）")
    p.add_argument("--seed0", type=int, default=1000, help="調難度用的 seed 從這裡開始；正式題目用另一段 seed")
    p.add_argument("--params", help='難度參數覆寫，JSON：{"T1": {"n_distractors": 20}, ...}')
    p.add_argument("--out", required=True)
    p = sub.add_parser("summary", help="從逐題 CSV 重算摘要")
    p.add_argument("csv")
    a = ap.parse_args()
    tune(a) if a.cmd == "tune" else resummarize(a)
