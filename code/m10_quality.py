"""m10_quality.py — 精度這條線的品質實驗（03_plan_h200.md §2.1、§2.2、§3.3、§3.4）

每題（§3.3）：
  1. m10_tasks 產生 prompt：context 剛好 64 個 block（含 chat template 開頭與說明）
  2. chunk 化 prefill（和還原器的重算同一條路徑：m7_model.forward_span，每次 512 token）
  3. 第 1 輪：接問題，greedy 生成
  4. 第 2 輪（§3.4）：歷史固定成標準答案——第 1 輪生成的 KV 直接被覆寫（KV 緩衝是依位置的，
     attention 只看 [0, end)），接 history_ids＋第 2 輪問題，greedy 生成
生成長度（03 §1 寫最多 64，上游是 niah 128、vt 30、qa 32、cwe 120；10/15 還沒決定）：
  生成到 max(64, 上游) 個 token，兩種截法都計分（score_64、score_up），之後決定用哪個不必重跑。

子命令：
  tune       只跑 E0，調難度（§3.2：每個模型、任務族、位置帶的 BF16 正確率要在 60–95%）。
  run        正式實驗：配置（E0、E1、E1s、E1sr、F25、M25、B25、R25，可加 @格式，例 E1s@fp8k0_head）×
             讀法 A（強制全載入）／B（照 m10_speed 的 meets.json 重放會合點）× 兩輪。**事先登記凍結之後才跑**（03 §1–§2）。
  selfcheck  結構檢查（不計分、不印任何低精度的答對率）：E0 兩條路徑一致、真的跑 Cake 和重放逐位元相同、
             沒讀到低精度時和 E0 逐位元相同、沒有 NaN。H200 第一天和 MI300X 彩排都用它。
  summary    從逐題 CSV 重算 tune 的摘要。

生成長度（10/15 使用者決定）：run／selfcheck 用 T5 120、其他 64、LongBench 用官方 dataset2maxlen；
tune 照舊生成到 max(64, 上游)、兩種截法都計分。

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
        # 模型和 tokenizer 必須是同一個 snapshot（10/15：m7_model 曾在設好 M7_MODEL_GLOB 前被 import，靜默載成 Llama）
        mp = os.path.realpath(self.km.cfg._name_or_path)
        tp = os.path.realpath(getattr(self.tok, "name_or_path", "") or "")
        if mp != tp or len(self.tok) > self.km.cfg.vocab_size:
            raise RuntimeError(f"模型和 tokenizer 不一致：model={mp} tokenizer={tp} "
                               f"len(tok)={len(self.tok)} vocab={self.km.cfg.vocab_size}")
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
    assert len(ctx) % ans.CHUNK == 0 and len(ctx) <= ans.n_ctx, len(ctx)   # T1–T5 剛好 32K；LongBench 是 512 的倍數
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


# ======================================================================== 低精度配置（03 §2.1、§2.2、§3.3、§3.4）
GEN_LEN = {"T5": 120}           # 10/15 使用者決定：T5 120，其他 64；LongBench 用官方 dataset2maxlen
ALIASES = {"E1-K0": "E1s@fp8k0_head", "E1-INT8": "E1s@int8_token"}     # Step 2b 的兩個換 codec 診斷（block 0 BF16）
CONFIGS = ["E0", "E1", "E1s", "E1sr", "F25", "M25", "B25", "R25"]
BW_LABELS = ["b16", "b32", "b48", "cpu"]                               # m10_speed 的 meets.json 用同樣的名字


def placement(name: str, n: int, seed: int) -> set:
    """低精度 block 的集合（03 §2.1）。n＝64 時和計劃逐字相同（F25＝1–16、M25＝24–39、B25＝48–63、
    R25＝從 1–63 用 seed＝題號抽 16 個）；n 較小時（LongBench）照比例。"""
    q = n // 4
    if name == "E0":
        return set()
    if name == "E1":
        return set(range(n))
    if name == "E1s":
        return set(range(1, n))
    if name == "E1sr":
        return set(range(1, n - 1))
    if name == "F25":
        return set(range(1, 1 + q))
    if name == "M25":
        s0 = (3 * n) // 8
        return set(range(s0, s0 + q))
    if name == "B25":
        return set(range(n - q, n))
    if name == "R25":
        import random
        return set(random.Random(seed).sample(range(1, n), q))
    raise ValueError(name)


def parse_config(spec: str, default_fmt: str) -> tuple[str, str, str]:
    """'F25' → ('F25','F25',fmt)；'E1s@fp8k0_head' → (名稱, 放法, 格式)；E1-K0、E1-INT8 是別名。"""
    full = ALIASES.get(spec, spec)
    name, _, fmt = full.partition("@")
    return spec, name, (fmt or default_fmt)


def gen_len(item: dict) -> int:
    if item["task"].startswith("LB-"):
        return item["max_new_tokens_upstream"]
    return GEN_LEN.get(item["task"], CUT)


class ItemState:
    """一題：prefill 一次，留下每個 block 的 BF16（GPU），各格式的解碼版本（用到才算），依配置把 KV 放進緩衝。"""

    def __init__(self, ans: Answerer, item: dict):
        from m10_codec import decode, encode, rel_err, rel_err_centered
        self._codec = (encode, decode, rel_err, rel_err_centered)
        self.ans, self.it = ans, item
        ctx = item["context_ids"]
        assert len(ctx) % ans.CHUNK == 0 and len(ctx) <= ans.n_ctx, len(ctx)
        self.n, self.ctx_len = len(ctx) // ans.CHUNK, len(ctx)
        self.ids = torch.tensor(ctx, device=ans.km.device)
        ans.km.prefill_chunked(self.ids)
        torch.cuda.synchronize()
        self.bf16 = [ans.km.chunk_view(i).clone() for i in range(self.n)]
        self.lp, self.err = {}, {}
        self.cur = ("bf16", frozenset())

    def lp_blocks(self, fmt: str):
        if fmt not in self.lp:
            encode, decode, rel_err, rel_err_centered = self._codec
            dec, err = [], []
            for x in self.bf16:
                y = decode(encode(x, fmt))
                r, c = rel_err(x, y), rel_err_centered(x, y)
                dec.append(y)
                err.append((r["K"], r["V"], c["K"]))
            self.lp[fmt], self.err[fmt] = dec, err
        return self.lp[fmt]

    def place(self, fmt: str, eff: frozenset):
        """eff＝實際讀到低精度的 block（低精度集合 ∩ [b, n)）。其餘放原本的 BF16：重算出來的 KV 和原本 prefill 的
        逐位元相同（test_m7_correctness），所以 block < b 用原本的 BF16 就是「重放」（§2.2；selfcheck 驗證）。"""
        key = (fmt if eff else "bf16", eff)
        if self.cur == key:
            return
        lp = self.lp_blocks(fmt) if eff else None
        for i in range(self.n):
            self.ans.km.chunk_view(i).copy_(lp[i] if i in eff else self.bf16[i])
        self.cur = key


@torch.inference_mode()
def teacher_force(ans: Answerer, logits0, gold_ids: list[int], start: int):
    """標準答案每個 token 的 log 機率分布 [m, vocab]（float32）。logits0＝問題最後一個位置的 logits（預測 gold[0]）。"""
    rows = [logits0.float().unsqueeze(0)]
    m = len(gold_ids)
    assert 0 < m <= ans.CHUNK + 1, m
    if m > 1:
        lg = ans.km.forward_span(torch.tensor(gold_ids[:-1], device=ans.km.device), start, want_logits=True,
                                 n_logits=m - 1)
        rows.append((lg if lg.dim() == 2 else lg.unsqueeze(0)).float())
    return torch.cat(rows).log_softmax(-1)


def run_turns(st: ItemState, fmt: str, effs: list[frozenset]) -> list[dict]:
    """依序跑每一輪；effs[t]＝第 t 輪時 context 的低精度集合（第 2 輪 (b) 會和第 1 輪不同）。
    第 2 輪的歷史固定成標準答案；第 1 輪問題的 KV 留在原位（存成 BF16，§3.4），只有 context 依 effs[1] 重放。"""
    ans, it = st.ans, st.it
    enc = T._enc(ans.tok)
    pos, out = st.ctx_len, []
    for t, tr in enumerate(it["turns"]):
        st.place(fmt, effs[t])
        seq = tr.get("history_ids", []) + tr["query_ids"]
        assert pos + len(seq) + max(gen_len(it), len(T._enc(ans.tok)(tr["gold_text"]))) + 1 <= ans.km.max_len
        logits0 = ans.feed(seq, pos)
        start_gen = pos + len(seq)
        gen, stop = ans.generate(logits0, start_gen, gen_len(it))
        gold_ids = enc(tr["gold_text"])
        tf = teacher_force(ans, logits0, gold_ids, start_gen)
        out.append(dict(gen=gen, stop=stop, tf=tf, gold_ids=gold_ids, first_logits=logits0.float()))
        pos = start_gen                      # 下一輪從這裡覆寫（生成與 teacher forcing 的 KV 都被蓋掉）
    return out


def _metrics(res: dict, ref: dict | None) -> dict:
    tf, g = res["tf"], torch.tensor(res["gold_ids"], device=res["tf"].device)
    lp = tf[torch.arange(len(g), device=tf.device), g]
    d = {"gold_logprob": float(lp.sum()), "gold_len": len(g)}
    if ref is not None:
        p0 = ref["tf"]
        d["kl_vs_e0"] = float((p0.exp() * (p0 - tf)).sum(-1).mean())
        d["same_as_e0"] = int(res["gen"] == ref["gen"])
    return d


def item_rows(st: ItemState, cfgs: list[tuple], modes: list[str], meets: dict | None, cache: dict) -> list[dict]:
    """一題的所有 (配置, 讀法, 頻寬, GPU 狀態, 輪, 第 2 輪變體)。實際讀到的低精度集合相同的格子只算一次（cache）。"""
    it, n = st.it, st.n
    rows = []

    def get(fmt, effs):
        key = (fmt if any(effs) else "bf16", tuple(effs))
        if key not in cache:
            cache[key] = run_turns(st, fmt, list(effs))
        return cache[key]

    e0 = get("bf16", [frozenset()] * len(it["turns"]))
    for spec, name, fmt in cfgs:
        lp_set = placement(name, n, it["seed"])
        cells = []                                   # (mode, bw, state, b1, b2_turn2b)
        if "A" in modes:
            cells.append(("A", "", "", 0, None))
        if "B" in modes and not it["task"].startswith("LB-"):
            mm = (meets or {}).get(spec) or (meets or {}).get(name)
            if mm is None:
                raise KeyError(f"meets.json 沒有 {spec} 的會合點")
            for bw, d in mm.items():
                cells.append(("B", bw, "idle", d["idle"], d["busy"]))
                cells.append(("B", bw, "busy", d["busy"], None))
        for mode, bw, state, b1, b2 in cells:
            eff1 = frozenset(i for i in lp_set if i >= b1)
            variants = [("a", eff1, b1)] + ([("b", frozenset(i for i in lp_set if i >= b2), b2)] if b2 is not None else [])
            for var, eff2, bb in variants:
                effs = [eff1] + [eff2] * (len(it["turns"]) - 1)
                res = get(fmt, effs)
                for t, tr in enumerate(it["turns"]):
                    if var == "b" and t == 0:
                        continue                     # 第 1 輪和 (a) 相同，只寫一次
                    eff = effs[t]
                    errs = st.err.get(fmt) if eff else None
                    pred = st.ans.decode(res[t]["gen"])
                    row = dict(model=st.ans.model, item_id=it["item_id"], task=it["task"], seed=it["seed"],
                               band=tr["band"], turn=tr["turn"], turn2_variant=var if t > 0 else "",
                               config=spec, fmt=fmt if lp_set else "", mode=mode, bw=bw, gpu_state=state,
                               b=(b1 if t == 0 else bb), n_blocks=n, lp_stored=len(lp_set), lp_read=len(eff),
                               lp_read_blocks=json.dumps(sorted(eff)),
                               relK_read=round(sum(errs[i][0] for i in eff) / len(eff), 6) if eff else "",
                               cenK_read=round(sum(errs[i][2] for i in eff) / len(eff), 6) if eff else "",
                               metric=tr["metric"], gold=json.dumps(tr["gold"], ensure_ascii=False),
                               needle_blocks=json.dumps(tr["needle_blocks"]), n_gen=len(res[t]["gen"]),
                               stop=res[t]["stop"], pred=pred, score=T.score(tr["metric"], pred, tr["gold"]),
                               f1=T.qa_f1(pred, tr["gold"]) if it["task"] == "T4" else "",
                               params=json.dumps(it["params"], sort_keys=True),
                               **_metrics(res[t], e0[t]))
                    rows.append(row)
    return rows


RUN_FIELDS = ["run_id", "ts", "model", "item_id", "task", "seed", "band", "turn", "turn2_variant", "config", "fmt",
              "mode", "bw", "gpu_state", "b", "n_blocks", "lp_stored", "lp_read", "lp_read_blocks", "relK_read",
              "cenK_read", "metric", "gold", "needle_blocks", "n_gen", "stop", "pred", "score", "f1", "gold_logprob",
              "gold_len", "kl_vs_e0", "same_as_e0", "params", "own_gpu_intruders"]


def make_items(a, tok):
    """正式題目：T1–T5 用 CANDIDATE_PARAMS（--params-file）的參數、seed 從 --seed0 起；LongBench 用 m10_longbench。"""
    P = json.load(open(a.params_file))["params"][a.model] if a.params_file else {}
    for task, b1, b2, seed in plan_items([t for t in a.tasks if not t.startswith("LB-")], a.n, a.seed0):
        yield lambda task=task, b1=b1, b2=b2, seed=seed: T.make_item(task, b1, b2, seed, tok, P.get(task))
    lbs = [t[3:] for t in a.tasks if t.startswith("LB-")]
    if lbs:
        import m10_longbench as LB
        for ds in lbs:
            for idx in range(a.lb_n):
                yield lambda ds=ds, idx=idx: LB.make_item(ds, idx, tok)


def _watch(out_path):
    try:
        from gpu_guard import GpuWatcher, smi_index
        w = GpuWatcher(gpu=smi_index(0), own_root=os.getpid(), out_path=os.path.splitext(out_path)[0] + ".gpu_guard.json")
        w.__enter__()
        return w
    except Exception as ex:                  # 品質 run：查不到爭用照樣跑，但記成 NOT_CHECKED（規則 7：不當成乾淨）
        print("GpuWatcher unavailable:", repr(ex), flush=True)
        return None


def run(a):
    ans = Answerer(a.model)
    rid = os.environ.get("RUN_ID", "adhoc")
    cfgs = [parse_config(c, a.fmt) for c in a.configs]
    meets = json.load(open(a.meets))["meets"] if a.meets else None
    if meets is not None and a.bws:          # 讀法 B 只用指定的頻寬（10/15 晚：只用中間頻寬 b32，03 §7 D4）
        meets = {c: {bw: d for bw, d in x.items() if bw in a.bws} for c, x in meets.items()}
    si, sk = (int(x) for x in a.shard.split("/"))
    done = set()
    if os.path.exists(a.out):                # 48 小時上限：重跑時跳過已完成的題（每題寫完才 flush，03 §9）
        done = {r["item_id"] for r in csv.DictReader(open(a.out))}
    f = open(a.out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=RUN_FIELDS, extrasaction="ignore")
    if not done:
        w.writeheader()
    watch = _watch(a.out)
    t0 = time.time()
    n_new = 0
    for k, mk in enumerate(make_items(a, ans.tok)):
        if k % sk != si:                     # --shard i/k：第 k 題只給 k mod 份數＝i 的 process（題目本身不變）
            continue
        it = mk()
        if it["item_id"] in done:
            continue
        st = ItemState(ans, it)
        rows = item_rows(st, cfgs, a.modes, meets, cache={})
        if a.blind:                          # 彩排／凍結前：低精度配置的品質欄一律遮掉（只留結構欄位）
            for r in rows:
                if r["lp_read"]:
                    for col in ("pred", "score", "f1", "gold_logprob", "kl_vs_e0", "same_as_e0"):
                        r[col] = "BLIND"
        intr = "NOT_CHECKED" if watch is None else len(watch.intruders)
        for r in rows:
            r.update(run_id=rid, ts=ts(), own_gpu_intruders=intr)
            w.writerow(r)
        f.flush()
        del st
        n_new += 1
        print(f"[{k + 1}] {it['item_id']} rows={len(rows)} {(time.time() - t0) / n_new:.1f}s/題", flush=True)
    f.close()
    if watch is not None:
        watch.__exit__(None, None, None)


def selfcheck(a):
    """結構檢查，**不計分**：只比位元與形狀（03 §2.2「真的 Cake vs 重放」、§9 的風險列）。輸出 JSON；任何一項不過 → exit 1。"""
    from m10_codec import encode
    from m7_restore_harness import Restorer, Tier
    ans = Answerer(a.model)
    rs = Restorer(ans.km)                    # 一個 process 只建一個
    P = json.load(open(a.params_file))["params"][a.model] if a.params_file else {}
    checks, ok = [], True

    def chk(name, cond, **info):
        nonlocal ok
        ok &= bool(cond)
        checks.append({"check": name, "pass": bool(cond), **info})
        print(("PASS " if cond else "FAIL ") + name, info if info else "", flush=True)

    items = [("T1", "front", "back"), ("T2", "mid", "front"), ("T3", "back", "mid"), ("T4", "front", "mid"),
             ("T5", None, None)][:a.n]
    for task, b1, b2 in items:
        it = T.make_item(task, b1, b2, a.seed0, ans.tok, P.get(task))
        iid = it["item_id"]
        # 1. E0：run_turns 和 tune 用的 run_item_e0（另一條路徑，重新 prefill）生成的開頭一致
        st = ItemState(ans, it)
        e0 = run_turns(st, "bf16", [frozenset()] * len(it["turns"]))
        ref_rows = run_item_e0(ans, it)
        st.cur = None
        chk(f"e0_two_paths[{iid}]", all(r["pred"].startswith(ans.decode(x["gen"])) for r, x in zip(ref_rows, e0)))
        st = ItemState(ans, it)
        lpfmt = a.fmt
        for name in ("E1", "F25", "B25", "R25"):
            lp_set = placement(name, st.n, it["seed"])
            # 2. 讀法 A：沒有 NaN、低精度 block 確實放進緩衝
            resA = run_turns(st, lpfmt, [frozenset(lp_set)] * len(it["turns"]))
            fin = all(bool(torch.isfinite(x["first_logits"]).all()) and bool(torch.isfinite(x["tf"]).all()) for x in resA)
            placed = all(torch.equal(ans.km.chunk_view(i), st.lp_blocks(lpfmt)[i] if i in lp_set else st.bf16[i])
                         for i in range(st.n))
            chk(f"readA_finite_and_placed[{iid}/{name}]", fin and placed)
            # 3. 真的跑 Cake vs 重放：KV 與第 1 輪問題的 logits 逐位元相同
            host = [encode(st.bf16[i], lpfmt).to("cpu").pin() if i in lp_set else st.bf16[i].to("cpu").pin_memory()
                    for i in range(st.n)]
            bw = ans.km.chunk_bytes / a.load_ms * 1e3
            tier = Tier("ssd_selfcheck", read_Bps=bw, write_Bps=bw)
            ans.km.kv[:, :, :, :st.ctx_len].zero_()
            torch.cuda.synchronize()
            rec = rs.restore(st.ids, st.n, host, [tier] * st.n, "cake")
            b = rec["meet"]
            kv_real = [ans.km.chunk_view(i).clone() for i in range(st.n)]
            seq = it["turns"][0]["query_ids"]
            lg_real = ans.feed(seq, st.ctx_len).float()
            st.cur = None
            eff = frozenset(i for i in lp_set if i >= b)
            st.place(lpfmt, eff)
            same_kv = all(torch.equal(kv_real[i], ans.km.chunk_view(i)) for i in range(st.n))
            lg_rep = ans.feed(seq, st.ctx_len).float()
            srcs = {}
            for _, src, _, _ in rec["blocks"]:
                srcs[src] = srcs.get(src, 0) + 1
            chk(f"cake_vs_replay[{iid}/{name}]", same_kv and torch.equal(lg_real, lg_rep), b=b, lp_read=len(eff),
                sources=srcs, max_logit_diff=float((lg_real - lg_rep).abs().max()))
            # 4. 沒讀到任何低精度 → 和 E0 逐位元相同（例：GPU 閒時的 F25）
            if not eff:
                chk(f"no_lp_read_equals_e0[{iid}/{name}]", torch.equal(lg_rep, e0[0]["first_logits"]))
        del st
    res = {"model": a.model, "fmt": a.fmt, "run_id": os.environ.get("RUN_ID", "adhoc"), "all_pass": bool(ok),
           "torch": torch.__version__, "hip": torch.version.hip, "cuda": torch.version.cuda,
           "note": "結構檢查，不計分、沒有任何答對率", "checks": checks}
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("ALL PASS" if ok else "SOME FAILED", flush=True)
    sys.exit(0 if ok else 1)



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
    for name in ("run", "selfcheck"):
        p = sub.add_parser(name)
        p.add_argument("--model", choices=list(T.MODELS), required=True)
        p.add_argument("--fmt", default="fp8_head", help="低精度格式（m10_codec.FORMATS）；D1 挑好的粒度")
        p.add_argument("--params-file", help="results/.../CANDIDATE_PARAMS.json（凍結後的難度參數）")
        p.add_argument("--seed0", type=int, default=10000 if name == "run" else 3000,
                       help="正式題目 10000 起（避開調難度用過的 1000–、2000–）")
        p.add_argument("--out", required=True)
        if name == "run":
            p.add_argument("--configs", nargs="+", default=CONFIGS)
            p.add_argument("--modes", nargs="+", default=["A"], choices=["A", "B"])
            p.add_argument("--meets", help="m10_speed 的 meets.json（讀法 B 必填）")
            p.add_argument("--bws", nargs="*", help="讀法 B 只用這些頻寬（meets.json 的名字，例 b32）；不給＝全部")
            p.add_argument("--shard", default="0/1", help="i/k：只跑第 i 份（題目依順序輪流分成 k 份），平行用")
            p.add_argument("--tasks", nargs="+", default=["T1", "T2", "T3", "T4", "T5"],
                           help="T1–T5；LongBench 寫 LB-<dataset>（例 LB-qasper）")
            p.add_argument("--n", type=int, default=60, help="T1–T5 每個任務幾題")
            p.add_argument("--lb-n", type=int, default=50)
            p.add_argument("--blind", action="store_true",
                           help="低精度配置（實際讀到低精度的列）的回答、分數、log-prob、KL 寫成 BLIND（凍結前試跑用）")
        else:
            p.add_argument("--n", type=int, default=5, help="檢查幾題（T1–T5 各一題）")
            p.add_argument("--load-ms", type=float, default=50.0, help="限速：BF16 一個 block 載入幾毫秒（讓會合點落在中間）")
    a = ap.parse_args()
    {"tune": tune, "summary": resummarize, "run": run, "selfcheck": selfcheck}[a.cmd](a)
