#!/usr/bin/env python3
"""d0.py — 第三階段 D0（05 §7、P0f）：MiniCPM4.1-8B 的原生稀疏索引（InfLLM-v2）。

  smoke --out J     P0f：HF transformers 4.56.1（pylibs 側裝）＋trust_remote_code 載入 MiniCPM4.1-8B，sparse_config 照 README 補上，
                    attn_implementation=flash_attention_2（dense 走 d0_shim/flash_attn → infllm_v2 的 varlen、topk_idx=None，先拿 SDPA 對答案）；
                    跑一個 32K 的問題，包住 infllm_v2 的函式計數，確認稀疏路徑真的有跑；記下 modeling_minicpm.py 的位置與 sha256、
                    第一次 sparse 呼叫的 topk_idx 形狀 → 能／不能（不計分、不進 results 的判定）
D0 的量測（聯集佔比、答案塊 recall）在讀過 modeling 程式、登記 runs/D0_native_index.md 之後才寫。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


# MiniCPM4.1-8B 的 README（revision 3a8dfed9）「To enable InfLLM v2, you need to add the sparse_config field」原樣；
# config.json 本身沒有這個欄位（沒有就整個模型走 dense）。modeling 程式裡 .get() 的預設值和這組相同。
SPARSE_CONFIG = dict(kernel_size=32, kernel_stride=16, init_blocks=1, block_size=64, window_size=2048, topk=64, use_nope=False,
                     dense_len=8192)
TRANSFORMERS_PIN = "4.56.1"            # config.json 的 transformers_version；5.x 沒有 is_torch_fx_available，cache 的 layer API 也不同
WRAP = ["infllmv2_attn_stage1", "infllmv2_attn_varlen_func", "infllmv2_attn_with_kvcache", "max_pooling_1d", "max_pooling_1d_varlen"]


def wrap_infllm(I, calls: dict, seen: dict):
    """包住 infllm_v2 的函式計數。varlen 分成 dense（topk_idx=None，flash_attn 轉接層走這裡）與 sparse；記第一次 sparse 的 topk_idx 形狀。"""
    for n in WRAP:
        f = getattr(I, n, None)
        if not callable(f):
            continue

        def w(*x, _f=f, _n=n, **k):
            key = _n
            if _n == "infllmv2_attn_varlen_func":
                t = k.get("topk_idx")
                key = _n + (":dense" if t is None else ":sparse")
                if t is not None and "topk_idx" not in seen:
                    seen["topk_idx"] = dict(shape=list(t.shape), dtype=str(t.dtype), q_shape=list(x[0].shape), k_shape=list(x[1].shape),
                                            n_neg1=int((t < 0).sum()), max=int(t.max()))
            calls[key] = calls.get(key, 0) + 1
            return _f(*x, **k)
        setattr(I, n, w)


def shim_check(torch) -> dict:
    """flash_attn 轉接層（dense＝infllm_v2 的 varlen、topk_idx=None）對 torch SDPA：GQA 32/2、causal，Sq＝Sk 與 Sq<Sk（右下對齊）。"""
    import torch.nn.functional as F
    from flash_attn import flash_attn_func
    g = torch.Generator(device="cuda").manual_seed(0)
    out = {}
    for name, sq, sk in (("square", 1000, 1000), ("bottom_right", 100, 1000)):
        q = torch.randn(1, sq, 32, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        k = torch.randn(1, sk, 2, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        v = torch.randn(1, sk, 2, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        o = flash_attn_func(q, k, v, causal=True).float()
        mask = torch.ones(sq, sk, dtype=torch.bool, device="cuda").tril(diagonal=sk - sq)
        ref = F.scaled_dot_product_attention(q.transpose(1, 2).float(), k.transpose(1, 2).float().repeat_interleave(16, 1),
                                             v.transpose(1, 2).float().repeat_interleave(16, 1), attn_mask=mask).transpose(1, 2)
        err = (o - ref).abs()
        out[name] = dict(max_abs=round(float(err.max()), 5), rel_l2=round(float(err.norm() / ref.norm()), 6))
    out["ok"] = all(out[n]["rel_l2"] < 1e-2 and out[n]["max_abs"] < 5e-2 for n in ("square", "bottom_right"))
    return out


def cmd_smoke(a):
    out = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"), run_id=os.environ.get("RUN_ID", "adhoc"), sparse_config=SPARSE_CONFIG)

    def done(code=0, why=""):
        if why:
            out["stop_reason"] = why
        print(json.dumps(out, indent=1, default=str), flush=True)
        json.dump(out, open(a.out, "w"), indent=1, default=str)
        if code:
            raise SystemExit(f"D0 冒煙：{why}（規則 2：停）")
    calls, seen = {}, {}
    try:
        import infllm_v2 as I
        out["infllm_v2_module"] = I.__file__
        wrap_infllm(I, calls, seen)
        import flash_attn
        out["flash_attn_module"] = flash_attn.__file__
        out["flash_attn_version"] = flash_attn.__version__
        import transformers
        out["transformers"] = dict(version=transformers.__version__, file=transformers.__file__)
    except Exception as ex:                                       # noqa: BLE001
        out["import"] = f"FAILED: {ex!r}"
        done(2, "import 失敗")
    if "infllmv2shim" not in out["flash_attn_version"]:
        done(2, f"flash_attn 不是 D0 的轉接層：{out['flash_attn_module']}")
    if out["transformers"]["version"] != TRANSFORMERS_PIN:
        done(2, f"transformers {out['transformers']['version']} ≠ {TRANSFORMERS_PIN}（pylibs 沒有排在 PYTHONPATH 前面？）")
    import torch
    out["shim_check"] = shim_check(torch)
    if not out["shim_check"]["ok"]:
        done(2, "flash_attn 轉接層和 SDPA 對不上")
    calls.clear()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    path = os.environ["M7_MINICPM_GLOB"]
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    out["config_sparse_config_before"] = getattr(cfg, "sparse_config", None)
    cfg.sparse_config = dict(SPARSE_CONFIG)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(path, config=cfg, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                                 attn_implementation="flash_attention_2").cuda().eval()
    out["t_load_s"] = round(time.time() - t0, 1)
    out["attn_class"] = sorted({type(l.self_attn).__name__ for l in model.model.layers})
    out["max_position_embeddings"] = getattr(model.config, "max_position_embeddings", None)
    mod = sys.modules[type(model).__module__]
    out["modeling_file"] = mod.__file__
    out["modeling_sha256"] = hashlib.sha256(open(mod.__file__, "rb").read()).hexdigest()
    if out["attn_class"] != ["MiniCPMInfLLMv2Attention"]:
        done(2, f"注意力類別是 {out['attn_class']}，不是 InfLLM-v2")
    import m10_tasks as T
    it = T.make_item("T2", "front", "back", 71000, tok, None, n_blocks=64)
    ids = torch.tensor([it["context_ids"] + it["turns"][0]["query_ids"]], device="cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        o = model(ids, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    out["t_forward_s"] = round(time.time() - t0, 2)
    out["n_tokens"] = ids.shape[1]
    out["logits_finite"] = bool(torch.isfinite(o.logits[0, -1]).all())
    out["infllm_v2_calls"] = dict(calls)
    out["first_sparse_topk_idx"] = seen.get("topk_idx")
    out["sparse_path_used"] = calls.get("infllmv2_attn_stage1", 0) > 0 and calls.get("infllmv2_attn_varlen_func:sparse", 0) > 0
    out["gpu_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
    if not (out["sparse_path_used"] and out["logits_finite"]):
        done(2, "稀疏路徑沒有被呼叫，或 logits 不是有限值")
    done()


# ============================================================================ D0 量測（runs/D0_native_index.md；登記 2026-10-25T14:32:50Z）
import csv          # noqa: E402
import random       # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "results", "s3_h200", "d0")
SEED0, N_PER_TASK, TASKS = 76000, 64, ("T2", "T3", "T4")
NB, BLK, SUB = 64, 512, 8                        # 32K 情境＝64 個 512-token 塊；一個 512 塊＝8 個 64-token 塊
CANDS = list(range(1, NB))                       # 塊 0（sink）永遠保留
KS, K_DECIDE, SKIP_LAYERS = (4, 8, 16, 32), 8, 2
METHODS = ["NATIVE-HITS", "NATIVE-HITS-L", "FULLK", "QUEST-512", "RND0", "RND1", "RND2"]
FIELDS = (["run_id", "ts", "item_id", "task", "seed", "method", "ans_blocks", "top8"] + [f"recall{k}" for k in KS]
          + ["jaccard8_vs_fullk", "jaccard16_vs_fullk", "union_frac", "union_frac_per_layer", "union64_frac", "n_local_blocks",
             "n_stage1_calls", "n_sparse_calls", "n_dense_calls", "topk_shape", "fullk_sum_maxerr", "seq_len", "n_q2",
             "own_gpu_intruders", "host_level"])


def plan() -> list:
    import m16_xstack as XS
    return [dict(task=t, b1=b1, b2=b2, seed=sd, item_id=f"{t}_{b1}-{b2}_s{sd}") for t, b1, b2, sd in
            XS.plan_interleaved_cpu(list(TASKS), N_PER_TASK, SEED0)]


class Capture:
    """包住 infllm_v2.infllmv2_attn_varlen_func：稀疏模式記每層 Q2 的 topk_idx；dense 模式每層算 FULLK、QUEST-512（只算 Q2 那幾列）。"""

    def __init__(self):
        self.mode, self.calls = None, {}

    def reset(self, mode, S, Lq):
        self.mode, self.S, self.Lq = mode, S, Lq
        self.calls = {}
        self.topk, self.fullk, self.quest, self.sumerr = [], [], [], 0.0

    def on_varlen(self, q, k, topk_idx):
        import torch
        key = "sparse" if topk_idx is not None else "dense"
        self.calls[key] = self.calls.get(key, 0) + 1
        if self.mode is None:
            return
        if topk_idx is not None:
            if self.mode != "sparse":
                raise RuntimeError("dense 模式出現帶 topk_idx 的呼叫")
            if topk_idx.dim() != 3 or topk_idx.shape[1] != self.S:
                raise RuntimeError(f"topk_idx 形狀 {tuple(topk_idx.shape)}，預期 [head, {self.S}, topk]")
            self.topk.append(topk_idx[:, -self.Lq:, :].to(torch.int64).cpu())
            self.topk_shape = list(topk_idx.shape)
            return
        if self.mode != "dense":
            return
        if q.shape[0] != self.S or k.shape[0] != self.S:
            raise RuntimeError(f"dense 呼叫的 q {tuple(q.shape)}／k {tuple(k.shape)}，預期第一維 {self.S}")
        with torch.no_grad():
            Hq, D, Hk = q.shape[1], q.shape[2], k.shape[1]
            g = Hq // Hk
            q2 = q[-self.Lq:].float()                                              # [Lq, Hq, D]
            kf = k.float()                                                         # [S, Hk, D]
            lg = torch.einsum("thd,shd->ths", q2, kf.repeat_interleave(g, dim=1)) / math.sqrt(D)
            pos = torch.arange(self.S - self.Lq, self.S, device=q.device)
            lg.masked_fill_(torch.arange(self.S, device=q.device)[None, None, :] > pos[:, None, None], float("-inf"))
            pr = torch.softmax(lg, dim=-1)
            self.sumerr = max(self.sumerr, float((pr.sum(-1) - 1).abs().max()))
            self.fullk.append(pr[..., :NB * BLK].reshape(self.Lq, Hq, NB, BLK).sum(-1).mean(0).cpu())        # [Hq, 64]
            kb = kf[:NB * BLK].reshape(NB, BLK, Hk, D)
            kmin, kmax = kb.amin(1).repeat_interleave(g, dim=1), kb.amax(1).repeat_interleave(g, dim=1)   # [64, Hq, D]
            U = torch.maximum(q2[:, :, None, :] * kmin.permute(1, 0, 2)[None], q2[:, :, None, :] * kmax.permute(1, 0, 2)[None]).sum(-1) / math.sqrt(D)
            self.quest.append(torch.softmax(U, dim=-1).mean(0).cpu())                                    # [Hq, 64]


def install_capture(I, cap: Capture):
    f = I.infllmv2_attn_varlen_func

    def w(*x, **k):
        cap.on_varlen(x[0], x[1], k.get("topk_idx"))
        return f(*x, **k)
    I.infllmv2_attn_varlen_func = w
    g = I.infllmv2_attn_stage1

    def w1(*x, **k):
        cap.calls["stage1"] = cap.calls.get("stage1", 0) + 1
        return g(*x, **k)
    I.infllmv2_attn_stage1 = w1


def ranked(score: dict, seed: int) -> list:
    perm = CANDS[:]
    random.Random(seed).shuffle(perm)
    tie = {b: i for i, b in enumerate(perm)}
    return sorted(CANDS, key=lambda b: (-score[b], tie[b]))


def native_stats(cap: Capture, S: int, Lq: int) -> dict:
    import torch
    n_layers = len(cap.topk)
    hits_all = {b: 0 for b in CANDS}
    hits_late = {b: 0 for b in CANDS}
    U, U64, per_layer = set(), set(), []
    for li, t in enumerate(cap.topk):
        c = t[t >= 0]
        if c.numel() and int(c.max()) >= math.ceil(S / 64):
            raise RuntimeError(f"第 {li} 層 topk_idx 的值 {int(c.max())} 超出 64-token 塊數 {math.ceil(S / 64)}")
        b512 = c // SUB
        cnt = torch.bincount(b512[b512 < NB], minlength=NB)
        for b in CANDS:
            hits_all[b] += int(cnt[b])
            if li >= SKIP_LAYERS:
                hits_late[b] += int(cnt[b])
        layer_set = {b for b in CANDS if int(cnt[b]) > 0}
        per_layer.append(len(layer_set) / len(CANDS))
        U |= layer_set
        U64 |= {int(x) for x in torch.unique(c) if SUB <= int(x) < NB * SUB}
    p0 = S - Lq
    local = {b for b in CANDS if b * BLK < p0 and (b + 1) * BLK > p0 - 2048}
    return dict(hits_all=hits_all, hits_late=hits_late, union_frac=len(U) / len(CANDS), union_frac_per_layer=sum(per_layer) / n_layers,
                union64_frac=len(U64) / ((NB - 1) * SUB), n_local_blocks=len(local))


def dense_scores(lst: list) -> dict:
    import torch
    m = torch.stack(lst[SKIP_LAYERS:]).mean(dim=(0, 1))                  # 略過前 2 層 → 對 (層, q head) 平均 → [64]
    return {b: float(m[b]) for b in CANDS}


def rank_row(order, A, full_order):
    out = {f"recall{k}": round(len(A & set(order[:k])) / len(A), 6) for k in KS}
    for k in (8, 16):
        a, b = set(order[:k]), set(full_order[:k])
        out[f"jaccard{k}_vs_fullk"] = round(len(a & b) / len(a | b), 6)
    out["top8"] = " ".join(map(str, order[:8]))
    return out


def cmd_run(a):
    import torch
    import m10_tasks as T
    from m16_c1a_predict import finish_watch, host_level, intruders, start_watch, verify_answer_blocks
    import infllm_v2 as I
    cap = Capture()
    install_capture(I, cap)
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    import transformers
    if transformers.__version__ != TRANSFORMERS_PIN:
        raise SystemExit(f"transformers {transformers.__version__} ≠ {TRANSFORMERS_PIN}")
    path = os.environ["M7_MINICPM_GLOB"]
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    cfg.sparse_config = dict(SPARSE_CONFIG)
    model = AutoModelForCausalLM.from_pretrained(path, config=cfg, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                                 attn_implementation="flash_attention_2").cuda().eval()
    attn = [l.self_attn for l in model.model.layers]
    if {type(x).__name__ for x in attn} != {"MiniCPMInfLLMv2Attention"}:
        raise SystemExit("不是 InfLLM-v2 的注意力")
    n_layers = len(attn)
    si, sk = (int(x) for x in a.shard.split("/"))
    out = a.out or os.path.join(OUT, "raw", f"d0_s{si}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    done = set()
    if os.path.exists(out):
        done = {r["item_id"] for r in csv.DictReader(open(out)) if r["method"] == "RND2"}
    todo = [r for k, r in enumerate(plan()) if k % sk == si and r["item_id"] not in done]
    if a.max_items:
        todo = todo[:a.max_items]
    print(f"[d0] {len(todo)} 題 → {out}", flush=True)
    rid = os.environ.get("RUN_ID", "adhoc")
    watch = start_watch(out)
    new = not os.path.exists(out)
    fo = open(out, "a", newline="")
    w = csv.DictWriter(fo, fieldnames=FIELDS)
    if new:
        w.writeheader()
    t_start = time.time()
    try:
        for n, r in enumerate(todo):
            if a.budget_min and (time.time() - t_start) / 60 > a.budget_min:
                print(f"[d0] 時間預算到了，做完 {n} 題", flush=True)
                break
            it = T.make_item(r["task"], r["b1"], r["b2"], r["seed"], tok, None, n_blocks=NB)
            ans, _, _ = verify_answer_blocks(tok, it, 1)                        # 不過 → RuntimeError（停）
            A = set(ans)
            seq = T.full_prompt_ids(it, 2)
            q2 = it["turns"][1]["query_ids"]
            S, Lq = len(seq), len(q2)
            if seq[-Lq:] != list(q2) or len(it["context_ids"]) != NB * BLK:
                raise RuntimeError("序列的最後一段不是第 2 輪問題，或情境不是 32K")
            ids = torch.tensor([seq], device="cuda")
            res = {}
            for mode in ("sparse", "dense"):
                for x in attn:
                    x.dense_len = SPARSE_CONFIG["dense_len"] if mode == "sparse" else 10 ** 9
                cap.reset(mode, S, Lq)
                with torch.inference_mode():
                    model(ids, use_cache=True, logits_to_keep=1)
                torch.cuda.synchronize()
                res[mode] = dict(calls=dict(cap.calls))
                if mode == "sparse":
                    if cap.calls.get("stage1", 0) != n_layers or cap.calls.get("sparse", 0) != n_layers or len(cap.topk) != n_layers:
                        raise RuntimeError(f"稀疏模式的呼叫次數 {cap.calls}，預期各 {n_layers}")
                    res["native"] = native_stats(cap, S, Lq)
                    res["topk_shape"] = cap.topk_shape
                else:
                    if cap.calls.get("sparse", 0) or len(cap.fullk) != n_layers:
                        raise RuntimeError(f"dense 模式的呼叫 {cap.calls}、FULLK 層數 {len(cap.fullk)}")
                    if cap.sumerr > 1e-3:
                        raise RuntimeError(f"FULLK 機率加總誤差 {cap.sumerr}")
                    res["fullk"], res["quest"], res["sumerr"] = dense_scores(cap.fullk), dense_scores(cap.quest), cap.sumerr
            cap.mode = None
            nat = res["native"]
            orders = {"NATIVE-HITS": ranked(nat["hits_all"], r["seed"]), "NATIVE-HITS-L": ranked(nat["hits_late"], r["seed"]),
                      "FULLK": ranked(res["fullk"], r["seed"]), "QUEST-512": ranked(res["quest"], r["seed"])}
            for j in range(3):
                o = CANDS[:]
                random.Random(r["seed"] * 10 + j).shuffle(o)
                orders[f"RND{j}"] = o
            lvl, intr = host_level(), intruders(watch)
            for m in METHODS:
                row = dict(run_id=rid, ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"), item_id=r["item_id"], task=r["task"], seed=r["seed"],
                           method=m, ans_blocks=" ".join(map(str, sorted(A))), seq_len=S, n_q2=Lq,
                           fullk_sum_maxerr=f"{res['sumerr']:.2e}", own_gpu_intruders=intr, host_level=lvl,
                           **rank_row(orders[m], A, orders["FULLK"]))
                if m.startswith("NATIVE"):
                    row.update(union_frac=round(nat["union_frac"], 6), union_frac_per_layer=round(nat["union_frac_per_layer"], 6),
                               union64_frac=round(nat["union64_frac"], 6), n_local_blocks=nat["n_local_blocks"],
                               n_stage1_calls=res["sparse"]["calls"].get("stage1", 0), n_sparse_calls=res["sparse"]["calls"].get("sparse", 0),
                               n_dense_calls=res["dense"]["calls"].get("dense", 0), topk_shape="x".join(map(str, res["topk_shape"])))
                w.writerow(row)
            fo.flush()
            print(f"[{n + 1}/{len(todo)}] {r['item_id']} S={S} Lq={Lq}", flush=True)
            torch.cuda.empty_cache()
    finally:
        fo.close()
        finish_watch(watch, out)


def cmd_report(a):
    import glob
    sys.path.insert(0, HERE)
    import a0 as A0
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, "raw", "d0_*.csv"))):
        rows += list(csv.DictReader(open(p)))
    by = {}
    for r in rows:
        by.setdefault(r["method"], {}).setdefault(r["item_id"], r)          # 重複的列取第一個
    items = sorted(set(by.get("NATIVE-HITS", {})) & set(by.get("FULLK", {})))
    res = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"), n_items=len(items),
               inputs=sorted(glob.glob(os.path.join(OUT, "raw", "d0_*.csv"))), recall={}, union={})
    if not items:
        res["verdict"] = "NOT_MEASURED"
    else:
        for m in METHODS:
            res["recall"][m] = {k: A0.boot(float(by[m][i][f"recall{k}"]) for i in items if i in by.get(m, {})) for k in KS}
        nat = by["NATIVE-HITS"]
        for f in ("union_frac", "union_frac_per_layer", "union64_frac", "n_local_blocks"):
            res["union"][f] = A0.boot(float(nat[i][f]) for i in items)
        gaps = {k: A0.boot(float(by["FULLK"][i][f"recall{k}"]) - float(nat[i][f"recall{k}"]) for i in items) for k in KS}
        res["gap_fullk_minus_native"] = gaps
        u, g = res["union"]["union_frac"], gaps[K_DECIDE]
        res["union_verdict"] = "UNION-NO-SAVINGS" if u["mean"] > 0.80 else "UNION-SAVES"
        res["union_borderline"] = u["lo"] <= 0.80 <= u["hi"]
        res["recall_verdict"] = "STOP" if g["mean"] > 0.10 else "SELECTOR-OK"
        res["recall_borderline"] = g["lo"] <= 0.10 <= g["hi"]
        res["verdict"] = ("GD0-STOP" if res["recall_verdict"] == "STOP" else
                          "GD0-SELECTOR-ONLY" if res["union_verdict"] == "UNION-NO-SAVINGS" else "GD0-CONTINUE")
        res["topk_shape"] = sorted({nat[i]["topk_shape"] for i in items})
        res["rows_with_intruders"] = sum(1 for i in items if str(nat[i]["own_gpu_intruders"]) not in ("0", "NOT_CHECKED"))
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "d0_report.json"), "w"), indent=1, ensure_ascii=False, default=str)
    print(json.dumps({k: res.get(k) for k in ("n_items", "verdict", "union_verdict", "union_borderline", "recall_verdict",
                                             "recall_borderline")}, ensure_ascii=False))


def self_test() -> bool:
    ok = True

    def chk(name, c):
        nonlocal ok
        print(("PASS " if c else "FAIL ") + name)
        ok &= bool(c)
    p = plan()
    chk("題數 192、item_id 唯一、種子 76000–76063", len(p) == 192 and len({r["item_id"] for r in p}) == 192
        and min(r["seed"] for r in p) == 76000 and max(r["seed"] for r in p) == 76063)
    s = {b: 0 for b in CANDS}
    s[5] = 3
    s[9] = 3
    o = ranked(s, 1)
    chk("排名：高分在前、同分用固定的隨機排列", set(o[:2]) == {5, 9} and o == ranked(s, 1) and sorted(o) == CANDS)
    chk("recall／jaccard", rank_row(o, {5, 40}, o)["recall8"] == 0.5 and rank_row(o, {5}, o)["jaccard8_vs_fullk"] == 1.0)
    try:
        import torch
    except ImportError:
        print("SKIP torch 的部分")
        return ok
    cap = Capture()
    S, Lq = NB * BLK + 100, 20
    cap.reset("sparse", S, Lq)
    t = torch.full((2, S, 96), -1, dtype=torch.int32)
    t[:, -Lq:, 0] = 8 * 7 + 3            # 64-token 塊 59 → 512 塊 7
    t[0, -Lq:, 1] = 8 * 40               # 512 塊 40（只有 head 0）
    for _ in range(4):
        cap.on_varlen(torch.zeros(S, 32, 128), torch.zeros(S, 2, 128), t)
    ns = native_stats(cap, S, Lq)
    chk("native：命中次數、聯集", ns["hits_all"][7] == 4 * 2 * Lq and ns["hits_all"][40] == 4 * Lq and abs(ns["union_frac"] - 2 / 63) < 1e-9
        and abs(ns["union64_frac"] - 2 / 504) < 1e-9 and ns["hits_late"][7] == 2 * 2 * Lq)
    cap.reset("dense", S, Lq)
    g = torch.Generator().manual_seed(0)
    q = torch.randn(S, 4, 16, generator=g)
    k = torch.randn(S, 2, 16, generator=g)
    k[3 * BLK + 10] = q[-1, 0] * 4                 # 最後一個 Q2 token 的 head 0 很會看塊 3
    for _ in range(3):
        cap.on_varlen(q, k, None)
    fk = dense_scores(cap.fullk)
    chk("dense：FULLK 機率加總＝1、強 key 所在的塊排前面", cap.sumerr < 1e-4 and max(fk, key=fk.get) == 3)
    qs = dense_scores(cap.quest)
    chk("dense：QUEST 分數在 (0,1)、塊數 63", all(0 < v < 1 for v in qs.values()) and len(qs) == 63)
    print("self-test", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("smoke")
    s.add_argument("--out", required=True)
    s = sub.add_parser("run")
    s.add_argument("--shard", default="0/1")
    s.add_argument("--max-items", type=int, default=0)
    s.add_argument("--budget-min", type=float, default=0)
    s.add_argument("--out", default="")
    sub.add_parser("report")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    {"smoke": cmd_smoke, "run": cmd_run, "report": cmd_report}[a.cmd](a)
