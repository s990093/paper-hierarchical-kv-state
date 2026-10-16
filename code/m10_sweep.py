"""m10_sweep.py — 第二階段的大掃描（docs/precision_20261013/09_sweep_plan.md）：格式 × 低精度佔比 × 放法。

這是**探索**（海選），不是事先登記的判定；判定只看 09 §4 的複賽（新題目、生成答案、真的量速度）。主線（07）不受影響。

子命令：
  parts   每種格式的「零件」：每個 block 的位元組、編碼、D2H、H2D、反量化（原寫法 m10_codec／融合 kernel）時間，
          GPU 閒／忙（m7_busy duty 0.7），gpu_guard 監控。→ 記憶體與載入速度的紀錄，也是 m10_sweep_sim 推算的輸入。
  screen  海選：每題 prefill 一次、只用第 1 輪；每個（格式, 放法, 佔比, 讀法）把 KV 放好，問題＋標準答案一次餵進去
          （teacher forcing，不生成），量對 E0 的 KL、標準答案 log 機率、逐位置 top-1 一致率、第一個 token 對不對。
          讀法：A＝64 個 block 全載入；B_idle／B_busy＝b32 頻寬下用 m10_sweep_sim 推算的會合點 b（block < b 用 BF16）。
  policies 印出某題的低精度集合（檢查用，不碰 GPU）。

放法（block 0 一律 BF16；k＝round(佔比×63) 個 block 用低精度，從 1–63 裡挑）：
  F／M／B   前／中／後連續 k 個
  R0–R4     隨機（每題每個 seed 一個固定排列，取前 k 個 → 不同佔比是巢狀的，曲線比較平滑）
  ERR       寫入當下就知道：這個格式在這個 block 的 K 中心化誤差最小的 k 個（誤差大的留 BF16）
  ATT       prefill 結束時看得到：最後一個 block（512 個 query）對每個 block 的 attention 量最少的 k 個（SnapKV 式觀察窗）
  AXE       ATT × ERR 最小的 k 個（PatchKV 式「attention × 重建誤差」風險）
  ORC       作弊上限：答案所在的 block 最後才給低精度（其餘照 R0 的順序）；只用來看位置最多能幫多少
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import statistics
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

N_BLOCKS = 64
FRACS = [round(0.10 + 0.05 * i, 2) for i in range(17)]                 # 10%–90% 每 5%（使用者 10/15 決定）
GEN_POS_POLICIES, GEN_POS_FRAC = ("F", "M", "B", "R0"), 0.25           # --gen-pos：這些格式另外生成 F／M／B／R0 在 25%（10_longctx_plan）
POLICIES = ["F", "M", "B", "R0", "R1", "R2", "R3", "R4", "ERR", "ATT", "AXE", "ORC"]
FORMATS = {
    "llama": ["fp8_token", "fp8e5_head", "fp8k0_token", "int8_token", "int4_token", "int4_kivi", "int4h_token",
              "mxfp4", "nvfp4", "int2_kivi", "int2h_g32", "kv-int8_token-int4_kivi", "kv-int4_kivi-int2_kivi"],
    "qwen": ["fp8_head", "fp8e5_head", "fp8k0_head", "int8_token", "int4_token", "int4_kivi", "int4h_token",
             "mxfp4", "nvfp4", "int2_kivi", "int2h_g32", "kv-int8_token-int4_kivi", "kv-int4_kivi-int2_kivi"],
}


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def k_of(frac: float) -> int:
    return int(round(frac * (N_BLOCKS - 1)))


# ============================================================================ 放法
def order(policy: str, ctx: dict) -> list[int]:
    """1..63 的「先給低精度」順序；低精度集合＝前 k 個。ctx：seed、needle（第 1 輪答案所在 block）、err、att。"""
    cand = list(range(1, N_BLOCKS))
    if policy == "F":
        return cand
    if policy == "B":
        return cand[::-1]
    if policy == "M":                               # 從中間往兩邊長：k 個時是置中的連續段
        mid = (1 + N_BLOCKS - 1) / 2
        return sorted(cand, key=lambda i: (abs(i - mid), i))
    if policy.startswith("R"):
        return random.Random(f"{ctx['seed']}|{policy}").sample(cand, len(cand))
    if policy == "ERR":
        return sorted(cand, key=lambda i: (ctx["err"][i], i))
    if policy == "ATT":
        return sorted(cand, key=lambda i: (ctx["att"][i], i))
    if policy == "AXE":
        return sorted(cand, key=lambda i: (ctx["att"][i] * ctx["err"][i], i))
    if policy == "ORC":
        r0 = order("R0", ctx)
        nd = set(ctx["needle"])
        return [i for i in r0 if i not in nd] + [i for i in r0 if i in nd]
    raise ValueError(policy)


def lp_set(policy: str, k: int, ctx: dict) -> frozenset:
    return frozenset(order(policy, ctx)[:k])


def check_policies():
    """形狀檢查：F25／M25／B25 在 k＝16 時和主線的 placement 逐字相同；R 是巢狀的；每個放法剛好 k 個、不含 block 0。"""
    from m10_quality import placement
    ctx = {"seed": 123, "needle": [5, 6], "err": [0.0] + [((i * 37) % 63) / 63 for i in range(1, 64)],
           "att": [0.0] + [((i * 11) % 63) / 63 for i in range(1, 64)]}
    ok = True
    ok &= lp_set("F", 16, ctx) == frozenset(placement("F25", 64, 0))
    ok &= lp_set("B", 16, ctx) == frozenset(placement("B25", 64, 0))
    for p in POLICIES:
        for f in FRACS:
            s = lp_set(p, k_of(f), ctx)
            ok &= len(s) == k_of(f) and 0 not in s
    ok &= all(lp_set("R2", k_of(a), ctx) <= lp_set("R2", k_of(b), ctx) for a, b in zip(FRACS, FRACS[1:]))
    ok &= not (lp_set("ORC", 30, ctx) & {5, 6})
    m16 = sorted(lp_set("M", 16, ctx))
    ok &= m16 == list(range(m16[0], m16[0] + 16)) and abs((m16[0] + m16[-1]) / 2 - 32) <= 1
    return ok


# ============================================================================ 共用：模型、KV 放置、teacher forcing
class Item:
    def __init__(self, ans, it: dict):
        import torch
        self.ans, self.it, self.torch = ans, it, torch
        km = ans.km
        ctx = it["context_ids"]
        assert len(ctx) == N_BLOCKS * ans.CHUNK, len(ctx)
        self.ctx_len = len(ctx)
        self.ids = torch.tensor(ctx, device=km.device)
        km.prefill_chunked(self.ids)
        torch.cuda.synchronize()
        self.bf16 = [km.chunk_view(i).clone() for i in range(N_BLOCKS)]
        self.src = ["bf16"] * N_BLOCKS              # 緩衝裡每個 block 現在放的是什麼
        self.dec, self.dec_fmt, self.err = None, None, None
        tr = it["turns"][0]
        import m10_tasks as T
        self.gold_ids = T._enc(ans.tok)(tr["gold_text"])
        self.seq = tr["query_ids"] + self.gold_ids[:-1]
        self.needle = tr["needle_blocks"]

    def load_fmt(self, fmt: str):
        """這個格式的 64 個解碼後 block（GPU）與每個 block 的 K 中心化誤差。一次只留一種格式。"""
        from m10_codec import rel_err_centered
        import m10_codec_x as X
        self.dec = None
        self.torch.cuda.empty_cache()
        dec, err, nbytes = [], [], 0
        for x in self.bf16:
            e = X.encode(x, fmt)
            nbytes = e.nbytes
            y = X.decode(e)
            dec.append(y)
            err.append(rel_err_centered(x, y)["K"])
        self.dec, self.dec_fmt, self.err, self.lp_nbytes = dec, fmt, err, nbytes
        self.src = [s if s == "bf16" else "stale" for s in self.src]

    def place(self, eff: frozenset):
        km = self.ans.km
        for i in range(N_BLOCKS):
            want = self.dec_fmt if i in eff else "bf16"
            if self.src[i] != want:
                km.chunk_view(i).copy_(self.dec[i] if i in eff else self.bf16[i])
                self.src[i] = want

    def score(self):
        """問題＋標準答案（去掉最後一個）一次餵進去；回傳標準答案每個位置的 log 機率分布 [m, vocab]。"""
        torch, km = self.torch, self.ans.km
        m = len(self.gold_ids)
        ids = torch.tensor(self.seq, device=km.device)
        with torch.inference_mode():
            if len(self.seq) <= self.ans.CHUNK:
                lg = km.forward_span(ids, self.ctx_len, want_logits=True, n_logits=m)
            else:                                    # 很長的問題：前面先接上，最後一段（含答案）再算
                cut = len(self.seq) - self.ans.CHUNK
                self.ans.feed(self.seq[:cut], self.ctx_len)
                lg = km.forward_span(ids[cut:], self.ctx_len + cut, want_logits=True, n_logits=m)
        lg = lg if lg.dim() == 2 else lg.unsqueeze(0)
        return lg.float().log_softmax(-1)

    def gen(self):
        """照主線的生成長度（T5 120、其他 64）、greedy，生成第 1 輪的答案並計分（m10_tasks.score，和主線同一個計分）。
        KV 就是現在緩衝裡放的；生成寫到 ctx_len 之後的位置，下一次 score／gen 從 ctx_len 重新覆寫（causal，不會讀到舊的）。"""
        import m10_tasks as T
        from m10_quality import gen_len
        ans = self.ans
        tr = self.it["turns"][0]
        with self.torch.inference_mode():
            logits = ans.feed(tr["query_ids"], self.ctx_len)
            out, stop = ans.generate(logits, self.ctx_len + len(tr["query_ids"]), gen_len(self.it))
        return T.score(tr["metric"], ans.decode(out), tr["gold"]), stop

    @staticmethod
    def metrics(lp, lp0, gold):
        import torch
        g = torch.tensor(gold, device=lp.device)
        ar = torch.arange(len(gold), device=lp.device)
        a, a0 = lp.argmax(-1), lp0.argmax(-1)
        return dict(kl=float((lp0.exp() * (lp0 - lp)).sum(-1).mean()), gold_lp=float(lp[ar, g].sum()),
                    top1=float((a == a0).float().mean()), first_ok=int(a[0] == g[0]))

    def attention_mass(self) -> list[float]:
        """prefill 結束時的觀察窗：重算最後一個 block（和 prefill 逐位元相同），每層記下這 512 個 query 對每個 key 的
        attention 機率，加總所有層、head、query → 每個 block 的量。最後一個 block 自己設成 +inf（觀察窗本身一律保護）。"""
        import torch
        import torch.nn.functional as F
        km = self.ans.km
        n, start = self.ans.CHUNK, (N_BLOCKS - 1) * self.ans.CHUNK
        end = start + n
        ids = self.ids[start:end]
        m = km.hf.model
        mass = torch.zeros(end, device=km.device, dtype=torch.float64)
        with torch.inference_mode():
            x = m.embed_tokens(ids.view(1, n))
            pos = torch.arange(start, end, device=km.device).view(1, n)
            cos, sin = m.rotary_emb(x, pos)
            from torch.nn.attention.bias import causal_lower_right
            mask = causal_lower_right(n, end)
            causal = torch.ones(n, n, device=km.device, dtype=torch.bool).tril()
            rep = km.Hq // km.H
            for li, layer in enumerate(m.layers):
                a = layer.self_attn
                h = layer.input_layernorm(x)
                q = a.q_proj(h).view(1, n, km.Hq, km.D)
                k = a.k_proj(h).view(1, n, km.H, km.D)
                if hasattr(a, "q_norm"):
                    q, k = a.q_norm(q), a.k_norm(k)
                q, k = q.transpose(1, 2), k.transpose(1, 2)
                v = a.v_proj(h).view(1, n, km.H, km.D).transpose(1, 2)
                q, k = km.apply_rope(q, k, cos, sin)
                km.kv[li, 0, :, start:end].copy_(k[0])
                km.kv[li, 1, :, start:end].copy_(v[0])
                K = km.kv[li, 0, :, :end]
                for hk in range(km.H):               # 每個 KV head 一組，省記憶體
                    qs = q[0, hk * rep:(hk + 1) * rep].float()                 # [rep, n, D]
                    s = qs @ K[hk].float().T / math.sqrt(km.D)                 # [rep, n, end]
                    s[:, :, start:].masked_fill_(~causal, float("-inf"))
                    mass += s.softmax(-1).sum(dim=(0, 1)).double()
                o = F.scaled_dot_product_attention(q, km.kv[li, 0, :, :end].unsqueeze(0),
                                                   km.kv[li, 1, :, :end].unsqueeze(0), attn_mask=mask, enable_gqa=True)
                o = o.transpose(1, 2).reshape(1, n, km.Hq * km.D)
                x = x + a.o_proj(o)
                x = x + layer.mlp(layer.post_attention_layernorm(x))
        torch.cuda.synchronize()
        km.chunk_view(N_BLOCKS - 1).copy_(self.bf16[N_BLOCKS - 1])          # 保險：放回原本的 BF16
        self.src[N_BLOCKS - 1] = "bf16"
        per = mass.view(N_BLOCKS, n).sum(-1).tolist()
        per[N_BLOCKS - 1] = float("inf")
        return per


# ============================================================================ parts：零件量測（記憶體、載入、反量化）
PARTS_FIELDS = ["run_id", "ts", "model", "fmt", "gpu_state", "block", "bytes", "bits_per_value", "t_encode_ms", "t_d2h_ms",
                "t_h2d_ms", "t_dequant_torch_ms", "t_dequant_fused_ms", "contaminated_gpu", "host_level"]


def cmd_parts(a):
    import torch
    from m10_quality import Answerer
    from m10_speed import Busy, _watcher
    import m10_codec as C
    import m10_codec_x as X
    import m10_dequant_fast as DF
    ans = Answerer(a.model)
    km = ans.km
    rid = os.environ.get("RUN_ID", "adhoc")
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1000, min(120000, km.cfg.vocab_size), (N_BLOCKS * ans.CHUNK,), generator=g).to(km.device)
    km.prefill_chunked(ids)
    torch.cuda.synchronize()
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"parts_{a.model}.csv")
    f = open(out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=PARTS_FIELDS)
    if f.tell() == 0:
        w.writeheader()
    watch, level = _watcher(os.path.join(a.out_dir, f"gpu_guard_{rid}_parts_{a.model}.json"))
    fmts = ["bf16"] + (a.formats or FORMATS[a.model])
    scratch = torch.empty_like(km.chunk_view(0))

    def ev():
        return torch.cuda.Event(enable_timing=True)

    def timed(fn):
        e0, e1 = ev(), ev()
        e0.record()
        r = fn()
        e1.record()
        e1.synchronize()
        return r, e0.elapsed_time(e1)

    summary, n_bad_fused, n_chk_fused = {}, {}, {}
    for state in a.states:
        busy = Busy(0.7, watch) if state == "busy" else None
        if busy:
            busy.__enter__()
        try:
            for fmt in fmts:
                rows = []
                for rep in range(a.reps):
                    for i in range(N_BLOCKS):
                        x = km.chunk_view(i)
                        if fmt == "bf16":
                            host = torch.empty(x.shape, dtype=x.dtype, pin_memory=True)
                            _, t_d2h = timed(lambda: host.copy_(x, non_blocking=True))
                            _, t_h2d = timed(lambda: scratch.copy_(host, non_blocking=True))
                            nb, t_enc, t_dq, t_dqf = x.numel() * 2, 0.0, 0.0, ""
                        else:
                            e, t_enc = timed(lambda: X.encode(x, fmt))
                            hp = e.map(lambda t: torch.empty(t.shape, dtype=t.dtype, pin_memory=True))
                            _, t_d2h = timed(lambda: hp.copy_from(e, non_blocking=True))
                            st = e.empty_like(km.device)
                            _, t_h2d = timed(lambda: st.copy_from(hp, non_blocking=True))
                            _, t_dq = timed(lambda: X.decode(st, out=scratch))
                            ref = scratch.clone() if DF.supported(fmt) and rep == 0 else None
                            t_dqf = timed(lambda: DF.decode_fast(st, out=scratch))[1] if DF.supported(fmt) else ""
                            if ref is not None:          # 真實 KV 上逐位元比對：融合 kernel 對原寫法（每個 block 第一次）
                                same = bool(torch.equal(scratch, ref))
                                n_bad_fused[fmt] = n_bad_fused.get(fmt, 0) + int(not same)
                                n_chk_fused[fmt] = n_chk_fused.get(fmt, 0) + 1
                            nb = e.nbytes
                        rows.append(dict(run_id=rid, ts=ts(), model=a.model, fmt=fmt, gpu_state=state, block=i, bytes=nb,
                                         bits_per_value=round(nb * 8 / x.numel(), 4), t_encode_ms=round(t_enc, 4),
                                         t_d2h_ms=round(t_d2h, 4), t_h2d_ms=round(t_h2d, 4),
                                         t_dequant_torch_ms=round(t_dq, 4),
                                         t_dequant_fused_ms=round(t_dqf, 4) if t_dqf != "" else "",
                                         contaminated_gpu="NOT_CHECKED" if watch is None else bool(watch.intruders) or
                                         watch.started_clean is False, host_level=level))
                for r in rows:
                    w.writerow(r)
                f.flush()
                med = lambda col: round(statistics.median(float(r[col]) for r in rows), 4) if rows[0][col] != "" else None  # noqa: E731
                summary[f"{fmt}|{state}"] = {k: med(k) for k in ("bytes", "bits_per_value", "t_encode_ms", "t_d2h_ms",
                                                                   "t_h2d_ms", "t_dequant_torch_ms", "t_dequant_fused_ms")}
                print(fmt, state, summary[f"{fmt}|{state}"], flush=True)
        finally:
            if busy:
                busy.__exit__(None, None, None)
    f.close()
    if watch is not None:
        watch.__exit__(None, None, None)
    js = os.path.join(a.out_dir, f"parts_{a.model}.json")
    old = json.load(open(js)) if os.path.exists(js) else {}
    old.update({"run_id": rid, "model": a.model, "chunk_bytes_bf16": km.chunk_bytes,
                "contaminated": None if watch is None else bool(watch.intruders) or watch.started_clean is False,
                "fused_vs_torch_real_kv": {f: {"blocks_checked": n_chk_fused[f], "blocks_not_bitwise": n_bad_fused.get(f, 0)}
                                           for f in n_chk_fused},
                "median_per_block": {**old.get("median_per_block", {}), **summary}})
    json.dump(old, open(js, "w"), indent=1, ensure_ascii=False)
    bad = {f: n for f, n in n_bad_fused.items() if n}
    if bad:                                              # 不一樣就是 bug：照規則 2 停下來
        raise SystemExit(f"融合反量化在真實 KV 上和原寫法不一樣（block 數）：{bad}")
    print("fused_vs_torch_real_kv:", {f: n_chk_fused[f] for f in n_chk_fused}, "blocks 全部逐位元相同", flush=True)


# ============================================================================ screen：海選
SCREEN_FIELDS = ["run_id", "ts", "model", "item_id", "task", "band", "seed", "needle_blocks", "fmt", "bits_per_value",
                 "policy", "frac", "k", "readout", "b", "lp_stored", "lp_read", "bytes_stored", "bytes_read", "kl",
                 "gold_lp", "gold_lp_e0", "top1", "first_ok", "first_ok_e0", "cenK_read", "own_gpu_intruders",
                 "n_blocks", "gen_score", "gen_stop"]     # 後三欄 10/16 加（長 context，10_longctx_plan.md）；32K 海選 gen 欄留空


class BModel:
    """讀法 B 的會合點：m10_sweep_sim 的雙線推算。重算線＝grid run 的「只算」參考線；載入線＝BF16 參考線，
    低精度 block 照位元組比例縮短再加上 parts 量到的反量化時間（原寫法，和主線的還原器相同）。"""

    def __init__(self, model: str, grid_blocks: str, parts_json: str, bw: str = "b32"):
        import m10_sweep_sim as S
        p = S.load_parts(grid_blocks, length=N_BLOCKS * 512)
        self.S, self.f, self.bw = S, p["f"], bw
        self.lb = {s: p["l"][(bw, s, "bf16")] for s in ("idle", "busy")}
        pj = json.load(open(parts_json))
        self.bf_bytes = pj["chunk_bytes_bf16"]
        self.parts = pj["median_per_block"]

    def b(self, fmt: str, lp: frozenset, state: str, nbytes: int) -> int:
        dq = self.parts.get(f"{fmt}|{state}", {}).get("t_dequant_torch_ms")
        if dq is None:
            raise KeyError(f"parts 沒有 {fmt}|{state}")
        lb = self.lb[state]
        load = [lb[i] * (nbytes / self.bf_bytes) + dq / 1e3 if i in lp else lb[i] for i in range(N_BLOCKS)]
        return self.S.simulate(self.f[state], load)[0]


def cmd_screen(a):
    global N_BLOCKS
    import torch
    import m10_codec_x as X
    import m10_tasks as T
    from m10_quality import Answerer, make_items, plan_items
    N_BLOCKS = a.n_blocks                          # 10/16：長 context（64K＝128、128K＝252 個 block）；預設 64＝32K
    X.install()
    ans = Answerer(a.model, extra=(N_BLOCKS - 64) * 512 + 4096)
    if N_BLOCKS == 64:
        items = make_items(a, ans.tok)
    else:                                          # 同樣的題目 seed 與內容，只把 haystack 拉長（位置帶照比例，m10_tasks.bands）
        P = json.load(open(a.params_file))["params"][a.model]
        items = (lambda task=task, b1=b1, b2=b2, seed=seed: T.make_item(task, b1, b2, seed, ans.tok, P.get(task),
                                                                          n_blocks=N_BLOCKS)
                 for task, b1, b2, seed in plan_items(a.tasks, a.n, a.seed0))
    gen_cells = ([("ALL", 1.0)] if a.gen_all else [])
    gen_pos = set(a.gen_pos or [])
    rid = os.environ.get("RUN_ID", "adhoc")
    bm = BModel(a.model, a.grid_blocks, a.parts) if a.parts else None
    fmts = a.formats or FORMATS[a.model]
    policies = a.policies or POLICIES
    fracs = a.fracs or FRACS
    if gen_pos:                                    # 要生成的格子必須在海選的格子裡（不然會靜默少做）
        assert all(p in policies for p in GEN_POS_POLICIES) and GEN_POS_FRAC in fracs, (policies, fracs)
    si, sk = (int(x) for x in a.shard.split("/"))
    done = set()
    if os.path.exists(a.out):
        with (gzip.open(a.out, "rt") if a.out.endswith(".gz") else open(a.out)) as fin:
            done = {r["item_id"] for r in csv.DictReader(fin)}
    fo = gzip.open(a.out, "at", newline="") if a.out.endswith(".gz") else open(a.out, "a", newline="")
    w = csv.DictWriter(fo, fieldnames=SCREEN_FIELDS)
    if not done:
        w.writeheader()
    from m10_quality import _watch
    watch = _watch(a.out)
    a.lb_n = 0                                     # 海選不含 LongBench
    t_all, n_items = time.time(), 0
    for kk, mk in enumerate(items):
        if kk % sk != si:
            continue
        it = mk()
        if it["item_id"] in done:
            continue
        t0 = time.time()
        st = Item(ans, it)
        tr = it["turns"][0]
        base = dict(model=a.model, item_id=it["item_id"], task=it["task"], band=tr["band"], seed=it["seed"],
                    needle_blocks=json.dumps(st.needle))
        att = st.attention_mass() if any(p in ("ATT", "AXE") for p in policies) else None
        st.place(frozenset())
        lp0 = st.score()
        m0 = Item.metrics(lp0, lp0, st.gold_ids)
        g0 = st.gen() if (gen_cells or gen_pos) else ("", "")
        rows = [dict(base, fmt="", bits_per_value=16.0, policy="E0", frac=0.0, k=0, readout="A", b="", lp_stored=0,
                     lp_read=0, bytes_stored=N_BLOCKS * ans.km.chunk_bytes, bytes_read=N_BLOCKS * ans.km.chunk_bytes,
                     gold_lp_e0=m0["gold_lp"], first_ok_e0=m0["first_ok"], cenK_read="", gen_score=g0[0], gen_stop=g0[1],
                     **m0)]
        n_fwd, n_gen = 1, int(bool(gen_cells or gen_pos))
        for fmt in fmts:
            st.load_fmt(fmt)
            bpv = st.lp_nbytes * 16 / ans.km.chunk_bytes          # 位元／值＝位元組×8 ÷（BF16 位元組÷2）
            ctx = {"seed": it["seed"], "needle": st.needle, "err": st.err, "att": att}
            cache = {frozenset(): m0}
            cells = [("ALL", 1.0, frozenset(range(1, N_BLOCKS)))]
            for p in policies:
                for fr in fracs:
                    cells.append((p, fr, lp_set(p, k_of(fr), ctx)))
            for p, fr, lp in cells:
                reads = [("A", "", lp)]
                if bm is not None:
                    for s in ("idle", "busy"):
                        b = bm.b(fmt, lp, s, st.lp_nbytes)
                        reads.append((f"B_{s}", b, frozenset(i for i in lp if i >= b)))
                for ro, b, eff in reads:
                    if eff not in cache:
                        st.place(eff)
                        cache[eff] = Item.metrics(st.score(), lp0, st.gold_ids)
                        n_fwd += 1
                    mm = cache[eff]
                    gs = ("", "")
                    if ro == "A" and ((p, fr) in gen_cells or (fmt in gen_pos and p in GEN_POS_POLICIES and fr == GEN_POS_FRAC)):
                        st.place(eff)
                        gs = st.gen()
                        n_gen += 1
                    rows.append(dict(base, gen_score=gs[0], gen_stop=gs[1], fmt=fmt, bits_per_value=round(bpv, 4), policy=p, frac=fr, k=len(lp),
                                     readout=ro, b=b, lp_stored=len(lp), lp_read=len(eff),
                                     bytes_stored=(N_BLOCKS - len(lp)) * ans.km.chunk_bytes + len(lp) * st.lp_nbytes,
                                     bytes_read=sum(st.lp_nbytes if i in eff else ans.km.chunk_bytes
                                                    for i in range(b or 0, N_BLOCKS)),
                                     gold_lp_e0=m0["gold_lp"], first_ok_e0=m0["first_ok"],
                                     cenK_read=round(sum(st.err[i] for i in eff) / len(eff), 6) if eff else "", **mm))
        intr = "NOT_CHECKED" if watch is None else len(watch.intruders)
        for r in rows:
            r.update(run_id=rid, ts=ts(), own_gpu_intruders=intr, n_blocks=N_BLOCKS)
            w.writerow(r)
        fo.flush()
        n_items += 1
        dt = time.time() - t0
        print(f"[{kk + 1}] {it['item_id']} n_blocks={N_BLOCKS} rows={len(rows)} forwards={n_fwd} gens={n_gen} {dt:.1f}s "
              f"({dt / n_fwd * 1e3:.1f} ms/forward) avg {(time.time() - t_all) / n_items:.1f}s/題 "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        del st
        torch.cuda.empty_cache()
    fo.close()
    if watch is not None:
        watch.__exit__(None, None, None)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("parts")
    p.add_argument("--model", choices=list(FORMATS), required=True)
    p.add_argument("--formats", nargs="+")
    p.add_argument("--states", nargs="+", default=["idle", "busy"])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--out-dir", required=True)
    s = sp.add_parser("screen")
    s.add_argument("--model", choices=list(FORMATS), required=True)
    s.add_argument("--params-file", required=True)
    s.add_argument("--tasks", nargs="+", default=["T2", "T3", "T4", "T5"])
    s.add_argument("--n", type=int, default=30, help="每個任務幾題（第 1 輪；T2–T4 三個位置帶輪流）")
    s.add_argument("--seed0", type=int, default=40000, help="海選的題目 seed（不和主線 10000、加題 20000 重疊）")
    s.add_argument("--formats", nargs="+")
    s.add_argument("--policies", nargs="+")
    s.add_argument("--fracs", nargs="+", type=float)
    s.add_argument("--parts", help="parts_<model>.json（有給才算讀法 B）")
    s.add_argument("--grid-blocks", help="主線 FP8 grid run 的 blocks.csv（重算線、BF16 載入線）")
    s.add_argument("--shard", default="0/1")
    s.add_argument("--out", required=True, help="逐列 CSV（可 .gz；很大，放 runs 目錄，不進 git）")
    s.add_argument("--n-blocks", type=int, default=64, help="context 的 block 數（64＝32K、128＝64K、252＝128K；10_longctx_plan）")
    s.add_argument("--gen-all", action="store_true", help="E0 和每個格式的 ALL（讀法 A）另外真的生成答案、計分")
    s.add_argument("--gen-pos", nargs="*", help="這些格式另外生成 F／M／B／R0 在 25%%（讀法 A）")
    c = sp.add_parser("policies")
    a = ap.parse_args()
    if a.cmd == "policies":
        ok = check_policies()
        print("policies OK" if ok else "policies FAIL")
        sys.exit(0 if ok else 1)
    {"parts": cmd_parts, "screen": cmd_screen}[a.cmd](a)


if __name__ == "__main__":
    main()
