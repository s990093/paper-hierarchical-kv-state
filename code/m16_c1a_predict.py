"""m16_c1a_predict.py — 第三階段 C1a：分數能不能預測第 2 輪要用的 block（只 forward、只打分，不生成）。

計畫：docs/stage3_20261019/experiment_matrix.md §3「C1a」。設計與停止條件（寫死，不改）：
docs/stage3_20261019/lit/L1_eviction_selective_loading.md §4 G2／G4、§5 X2（＋X4）。怎麼跑：docs/stage3_20261019/c1a_s2_howto.md。

每題（DS2 的兩輪題目：seed 70000 起，T2–T4 各 45、T5 15；14 號 §8.5；參數 CANDIDATE_PARAMS.json）：
  1. 32K prefill 一次（m10_quality.ItemState：和 DS1／DS2 同一條 chunk 化 prefill）。
  2. 每個 512-token block 一個「保留順序」（block 1–63 排名；block 0 一律保留、不參加排名，所以隨機期望＝k/63）：
     (a) 寫入時：SNAP（無問題版）、EXPATT、KNORM、KEYDIFF、H2O 近似＝m10_drop.scores；STREAM、NEW、RND0–2＝位置。
         token 分數 → block 順序一律用 m10_drop.unit_order(cx, "B", 方法)（14 §8.3：每個（層, KV head）上 block 內 token 分數平均
         的百分位排名，再對所有（層, head）平均）；unit_order 是「先丟」的順序，反過來就是保留順序。
     (b) QRY1＝第 1 輪問題、QRY2＝第 2 輪問題，都是 m10_drop 的 QRY 定義（score_qry：問題 token 當觀察窗、接在 context 後面，
         每個問題 token 對 context 每個 token 的 attention 平均、avg-pool 5、GQA 組內平均）。
         「第 2 輪問題放哪」（--q2-placement）：direct＝m10_drop 的定義原樣（第 2 輪問題直接接在 context 後面，預設）；
         insitu＝實際的第 2 輪序列（context＋第 1 輪問題＋歷史＋第 2 輪問題），只取第 2 輪問題那幾個 token 的 attention。
         另一種放法一律另記一列 QRY2_ALT（敏感度：放法本身會不會改變排名）。
     (c) Quest 式摘要（X4）：每個 block、每個（層, KV head）的 post-RoPE K 逐維 min／max（快取裡存的 BF16）。
         目標問題（兩輪題＝第 2 輪；T5＝唯一一輪）的每個 token、每層、每個 q head 的 post-RoPE query q
         （GQA：q head h 對到 KV head h // (Hq/H)，和 HF repeat_kv、m10_drop 同一個對應）：
             U[b] = Σ_d max(q_d·min_d[b], q_d·max_d[b]) / √D        （＝q⁺·max ＋ q⁻·min，Quest 的上界）
         彙總（寫死）：每個（層, q head, 問題 token）在 64 個 block 上對 U 做 softmax（把一個 block 當一個 key、上界當 logit）
         → 對問題 token 平均 → [層, q head, block] → 略過前 --quest-skip-layers 層（預設 2：Quest 原文前兩層不套用）
         → 對（層, q head）取平均＝QUEST_MEAN、取最大＝QUEST_MAX。
         FULLK_MEAN／FULLK_MAX＝同一個 forward 的「完整 K」對照：真的 attention 機率在每個 block 的總和、對問題 token 平均，
         同樣略過前幾層、對（層, q head）取平均／最大。和 QUEST_* 只差「K 換成 min／max 摘要」這一件事。
  3. 指標（每題 × 每個分數一列）：
     - recall@k（k＝16、32）＝第 2 輪答案所在的 block 有幾成落在保留順序前 k 名；隨機期望 k/63。
       另記 hit_all／hit_any（全部／至少一個在前 k）和隨機排列下的精確期望（超幾何）。
       「答案所在的 block」＝m10_tasks 組裝時記的 needle token 位置 → pos // 512；用兩條獨立路徑驗算（verify_answer_blocks）。
       T5（cwe，全域）沒有位置帶：不報 recall，只報 Spearman 與重疊率（目標＝它唯一一輪的問題）。
     - spearman_vs_target：和目標 QRY（兩輪題＝QRY2；T5＝QRY1）的 63 個 block 名次的 Spearman ρ。
     - overlap@k_vs_target：前 k 名和目標前 k 名重疊的比例；隨機期望 k/63。
     - recall_t1@k：對第 1 輪答案 block 的 recall（只當健全性檢查：QRY1 在這裡應該很高）。

子命令：
  run        正式（--shard i/n、逐題續跑；--smoke＝1 題、印每個分數前 5 名 block 和答案 block，供人工檢查）
  selfcheck  GPU 窗口的第一個檢查：同一題（DS1 的第一題）走 m10_drop screen 的路徑（m10_sweep.Item → m10_drop.Ctx → scores）
             和本程式的路徑（ItemState → compute_item），每個分數逐值比對（torch.equal）、block 順序比對、prefill 逐位元比對
  report     CPU：讀逐列 CSV → c1a_summary.json、tables_c1a.md（bootstrap 10,000 次、以題目為單位；L1 停止條件的各個成分）

GPU 監控照 m10_quality（_watch＝GpuWatcher）：品質實驗，污染時分數保留、逐列記 own_gpu_intruders／host_level，
run 目錄（$TIARA_RUNS/$RUN_ID）留 CONTAMINATED；t_score_s 等時間欄在污染的 run 作廢（CLAUDE.md §3）。

模組頂層只有標準函式庫的 import 與常數；torch、m10_* 一律在函式裡才 import（S2 會 import 這裡的小工具，report 不碰 torch）。
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PARAMS = os.path.join(REPO, "results", "m10_pilot_mi300x", "e0_tune", "CANDIDATE_PARAMS.json")
OUT_DIR = os.path.join(REPO, "results", "m16_stage3_mi300x", "c1a_predictive")

CHUNK = 512                         # m7_model.CHUNK＝m10_tasks.BLOCK（執行時 assert）
N_BLOCKS = 64                       # 32K
CAND = tuple(range(1, N_BLOCKS))    # 參加排名的 block（block 0 一律保留）
N_CAND = len(CAND)                  # 63
KS = (16, 32)                       # L1 X2：約 25%、50%
N_BOOT = 10000
WRITE_SCORED = ("SNAP", "EXPATT", "KNORM", "KEYDIFF", "H2O")      # L1 X2 (1)；全部是 m10_drop.SCORED 裡的
POSITION = ("STREAM", "NEW", "RND0", "RND1", "RND2")
SUMMARIES = ("QUEST_MEAN", "QUEST_MAX", "FULLK_MEAN", "FULLK_MAX")


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def runs_root() -> str:
    return os.environ.get("TIARA_RUNS", "/mlsteam/data/tiara/runs")


def parse_shard(s: str) -> tuple[int, int]:
    i, n = (int(x) for x in s.split("/"))
    if not 0 <= i < n:
        raise ValueError(f"--shard {s}：要 0 ≤ i < n")
    return i, n


# ============================================================================ CSV（續跑、讀取）：S2 也用
def _reader(path):
    return gzip.open(path, "rt", newline="") if path.endswith(".gz") else open(path, newline="")


def read_rows(paths) -> list[dict]:
    """讀逐列 CSV（可 .gz）。跳過重複的表頭、寫到一半被砍的殘列（欄位數不對）。"""
    rows = []
    for p in paths:
        try:
            with _reader(p) as f:
                for r in csv.DictReader(f):
                    if None in r or None in r.values() or r.get("run_id") in (None, "run_id"):
                        continue
                    rows.append(r)
        except EOFError as ex:                      # .gz 最後一段寫到一半：讀得到的照用，並且說出來
            print(f"[read_rows] {p}: {ex!r}（只用讀得到的列）", flush=True)
    return rows


def done_items(out: str) -> set:
    """續跑：一題算完＝該題的列數 ≥ 列上記的 n_rows_item（寫到一半被砍的題會整題重算；報告時去重）。"""
    if not os.path.exists(out):
        return set()
    cnt, want = {}, {}
    for r in read_rows([out]):
        i = r["item_id"]
        cnt[i] = cnt.get(i, 0) + 1
        if r.get("n_rows_item", "").isdigit():
            want[i] = int(r["n_rows_item"])
    return {i for i, c in cnt.items() if i in want and c >= want[i]}


def open_out(out: str, fields: list):
    d = os.path.dirname(os.path.abspath(out))
    os.makedirs(d, exist_ok=True)
    new = not os.path.exists(out) or os.path.getsize(out) == 0
    fo = gzip.open(out, "at", newline="") if out.endswith(".gz") else open(out, "a", newline="")
    w = csv.DictWriter(fo, fieldnames=fields)
    if new:
        w.writeheader()
        fo.flush()
    return fo, w


def side_path(out: str, suffix: str) -> str:
    base = out[:-3] if out.endswith(".gz") else out
    return os.path.splitext(base)[0] + suffix


# ============================================================================ GPU 監控（照 m10_quality）：S2 也用
def start_watch(out: str):
    """m10_quality._watch：GpuWatcher(gpu=smi_index(0)) 包住整段；開不起來 → None，每列記 NOT_CHECKED（規則 7：不當成乾淨）。"""
    from m10_quality import _watch
    return _watch(out)


def host_level() -> str:
    try:
        from gpu_guard import host_contention
        return host_contention(exclude_gpu=None)["level"]
    except Exception as ex:                          # 查不到 ≠ 沒有（規則 7）
        return f"UNKNOWN({type(ex).__name__})"


def intruders(watch):
    if watch is None:
        return "NOT_CHECKED"
    if getattr(watch, "smi_failures", None):          # 監看有一段時間查不到（gpu_guard 10/25）→ 不能報 0
        return f"UNOBSERVED:{len(watch.intruders)}"
    return len(watch.intruders)


def finish_watch(watch, out: str):
    """結束監看；污染 → run 目錄留 CONTAMINATED（沒有 RUN_ID 時放在輸出旁邊）。品質規則：分數保留、時間欄作廢。"""
    if watch is None:
        print("GpuWatcher 沒有開成：每列 own_gpu_intruders＝NOT_CHECKED（不當成乾淨，規則 7）", flush=True)
        return None
    watch.__exit__(None, None, None)
    if watch.contaminated:
        rid = os.environ.get("RUN_ID")
        d = os.path.join(runs_root(), rid) if rid else os.path.dirname(os.path.abspath(out))
        p = os.path.join(d, "CONTAMINATED" if rid else os.path.basename(out) + ".CONTAMINATED")
        os.makedirs(d, exist_ok=True)
        with open(p, "w") as f:
            json.dump(dict(verdict=watch.verdict(), intruders=list(watch.intruders.values()), out=out,
                           note="品質實驗：分數保留、逐列記爭用；t_score_s 等時間欄作廢（CLAUDE.md §3）"),
                      f, indent=1, default=str, ensure_ascii=False)
        print(f"CONTAMINATED（{watch.verdict()}）→ {p}", flush=True)
    return watch.contaminated


# ============================================================================ 統計（CPU）：S2 也用
def boot(values, n_boot: int = N_BOOT, seed: int = 0):
    """以題目為單位的 bootstrap（每個值＝一題）：平均與 95% 百分位 CI。"""
    import numpy as np
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return None
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(1)
    return dict(mean=float(v.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n=int(v.size))


def ci_has(b, x: float = 0.0):
    """CI 包含 x（＝和 x 分不開）。b 是 None（沒有資料）時回 None。"""
    return None if b is None else bool(b["lo"] <= x <= b["hi"])


def exp_hits(m: int, k: int, n: int = N_CAND):
    """m 個答案 block、從 n 個裡隨機排列取前 k 個：全部在裡面、至少一個在裡面的機率（超幾何，精確）。"""
    tot = math.comb(n, k)
    p_all = math.comb(n - m, k - m) / tot if k >= m else 0.0
    p_any = 1.0 - math.comb(n - m, k) / tot
    return p_all, p_any


# ============================================================================ 答案 block（規則 6：兩條獨立路徑驗算）
def needle_blocks_from_spans(it: dict, t: int) -> list:
    """第 t 輪（0 起）的 needle token 位置（m10_tasks 組裝時記的 spans）→ block＝pos // 512。"""
    bs = set()
    for kind, tag, s, e in it["spans"]:
        if kind == "needle" and tag[0] == t:
            bs.update(range(s // CHUNK, (e - 1) // CHUNK + 1))
    return sorted(bs)


def verify_answer_blocks(tok, it: dict, t: int):
    """回傳 (答案 block, 解碼搜尋找到的答案字串 block, 說明)。任何一條不過 → RuntimeError（規則 2：停）。
    (1) spans 的 token 位置 // 512，必須等於 m10_tasks 記的 needle_blocks，且都在 1..63；
    (2) 不看組裝記錄：m10_tasks._Locator（token → 原始位元組 → 文字）在 context 解碼文字裡找每個答案字串
        → 每個答案至少出現一次，而且每次出現都落在 (1) 的 block 裡；
    (3) (1) 的每一段連續 block 直接 tok.decode，每個答案字串要出現在其中一段裡。
    T5（全域，needle＝常見詞的每一次出現）不驗算，回傳說明。"""
    import m10_tasks as T
    tr = it["turns"][t]
    nb = needle_blocks_from_spans(it, t)
    if it["task"] == "T5":
        return nb, [], {"status": "T5_GLOBAL_NOT_CHECKED"}
    errs = []
    if it["block"] != CHUNK or it["n_blocks"] != N_BLOCKS:
        errs.append(f"block={it['block']} n_blocks={it['n_blocks']}（要 {CHUNK}、{N_BLOCKS}）")
    if nb != sorted(tr["needle_blocks"]):
        errs.append(f"spans→block {nb} ≠ needle_blocks {sorted(tr['needle_blocks'])}")
    if not nb or nb[0] < 1 or nb[-1] >= N_BLOCKS:
        errs.append(f"答案 block {nb} 不在 1..{N_BLOCKS - 1}")
    ctx = it["context_ids"]
    loc = T._Locator(tok, ctx)
    if not loc.ok_vs_decode:
        errs.append("token→位元組的對照和 tok.decode 不同（定位不可信）")
    found = {}
    for g in tr["gold"]:
        occ = loc.find(g)
        found[g] = sorted({b for t0, t1 in occ for b in range(t0 // CHUNK, (t1 - 1) // CHUNK + 1)})
        if not occ:
            errs.append(f"答案 {g!r} 在 context 解碼文字裡找不到")
        out = [b for b in found[g] if b not in nb]
        if out:
            errs.append(f"答案 {g!r} 出現在答案 block 以外的 block {out}")
    runs = []
    for b in nb:
        if runs and b == runs[-1][-1] + 1:
            runs[-1].append(b)
        else:
            runs.append([b])
    texts = [tok.decode(ctx[r[0] * CHUNK:(r[-1] + 1) * CHUNK]).lower() for r in runs]
    for g in tr["gold"]:
        if not any(g.lower() in x for x in texts):
            errs.append(f"答案 {g!r} 不在答案 block 的解碼文字裡")
    if errs:
        raise RuntimeError(f"{it['item_id']} 第 {t + 1} 輪答案 block 驗算失敗：" + "；".join(errs))
    loc_blocks = sorted(set().union(*[set(v) for v in found.values()]))
    return nb, loc_blocks, {"status": "OK", "runs": runs, "gold_blocks": found}


# ============================================================================ GPU：打分
def probe(cx, ids_win, start: int, tail_from: int):
    """問題當觀察窗跑過每一層。算式逐行照 m10_drop._window_probs（key_end＝ctx_len），只多兩件事：
    (1) attention 機率只加總第 tail_from 個之後的 query（tail_from＝0 時和 _window_probs 完全相同 → 自我檢查逐位元比對）；
    (2) 留下那些 query 的 post-RoPE q（Quest 上界用）。
    回傳 (p [L, Hq, ctx_len] float32：尾段 query 對每個 context key 的機率加總, q [L, Hq, w−tail_from, D] float32)。
    觀察窗寫進緩衝的位置 ≥ ctx_len（context 不動；之後的 forward 從 ctx_len 起覆寫）。"""
    import torch
    import torch.nn.functional as F
    from torch.nn.attention.bias import causal_lower_right
    km = cx.km
    assert km.key_keep is None
    key_end = cx.ctx_len
    w = len(ids_win)
    end = start + w
    assert start >= cx.ctx_len and end <= km.max_len and 0 <= tail_from < w, (start, end, km.max_len, tail_from, w)
    m = km.hf.model
    rep = km.Hq // km.H
    out = torch.zeros(km.L, km.Hq, key_end, device=km.device, dtype=torch.float32)
    qcap = torch.empty(km.L, km.Hq, w - tail_from, km.D, device=km.device, dtype=torch.float32)
    ids = torch.as_tensor(ids_win, device=km.device).view(1, w)
    with torch.inference_mode():
        x = m.embed_tokens(ids)
        pos = torch.arange(start, end, device=km.device).view(1, w)
        cos, sin = m.rotary_emb(x, pos)
        mask = causal_lower_right(w, end)
        causal = torch.ones(w, w, device=km.device, dtype=torch.bool).tril()
        for li, layer in enumerate(m.layers):
            a = layer.self_attn
            h = layer.input_layernorm(x)
            q = a.q_proj(h).view(1, w, km.Hq, km.D)
            k = a.k_proj(h).view(1, w, km.H, km.D)
            if hasattr(a, "q_norm"):
                q, k = a.q_norm(q), a.k_norm(k)
            q, k = q.transpose(1, 2), k.transpose(1, 2)
            v = a.v_proj(h).view(1, w, km.H, km.D).transpose(1, 2)
            q, k = km.apply_rope(q, k, cos, sin)
            qcap[li] = q[0, :, tail_from:].float()
            km.kv[li, 0, :, start:end].copy_(k[0])
            km.kv[li, 1, :, start:end].copy_(v[0])
            K = km.kv[li, 0, :, :end]
            for hk in range(km.H):
                qs = q[0, hk * rep:(hk + 1) * rep].float()                      # [rep, w, D]
                s = qs @ K[hk].float().T / math.sqrt(km.D)                      # [rep, w, end]
                s[:, :, start:].masked_fill_(~causal, float("-inf"))
                out[li, hk * rep:(hk + 1) * rep] += s.softmax(-1)[:, tail_from:, :key_end].sum(1)
            o = F.scaled_dot_product_attention(q, km.kv[li, 0, :, :end].unsqueeze(0), km.kv[li, 1, :, :end].unsqueeze(0),
                                               attn_mask=mask, enable_gqa=True)
            x = x + a.o_proj(o.transpose(1, 2).reshape(1, w, km.Hq * km.D))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
    return out, qcap


def block_minmax(bf16_blocks):
    """Quest 的摘要：每個 block、每個（層, KV head）的 post-RoPE K 逐維 min／max（BF16 取極值是精確的，再轉 float32）。
    bf16_blocks[b]＝[L, 2, H, 512, D] → 回傳 (kmin, kmax)，各 [L, H, nb, D]。"""
    import torch
    kmin = torch.stack([b[:, 0].amin(dim=2) for b in bf16_blocks], dim=2).float()
    kmax = torch.stack([b[:, 0].amax(dim=2) for b in bf16_blocks], dim=2).float()
    return kmin, kmax


def quest_scores(km, kmin, kmax, qcap):
    """U[l, hq, t, b] = Σ_d max(q·min, q·max)/√D；每個（層, q head, 問題 token）對 block 做 softmax，再對 token 平均 → [L, Hq, nb]。"""
    rep = km.Hq // km.H
    kmx = kmax.repeat_interleave(rep, dim=1)                                   # [L, Hq, nb, D]（q head h → KV head h // rep）
    kmn = kmin.repeat_interleave(rep, dim=1)
    U = (qcap.clamp(min=0) @ kmx.transpose(-1, -2) + qcap.clamp(max=0) @ kmn.transpose(-1, -2)) / math.sqrt(km.D)
    return U.softmax(-1).mean(2)


def fullk_scores(p, n_tail: int, nb: int):
    """完整 K 的對照：真的 attention 機率在每個 block 的總和、對問題 token 平均 → [L, Hq, nb]。"""
    L, Hq, _ = p.shape
    return p[..., :nb * CHUNK].reshape(L, Hq, nb, CHUNK).sum(-1) / n_tail


def agg_blocks(P, skip: int, how: str) -> list:
    """略過前 skip 層，對（層, q head）取平均或最大 → 每個 block 一個分數。"""
    assert 0 <= skip < P.shape[0], (skip, P.shape)
    X = P[skip:].reshape(-1, P.shape[-1])
    return (X.mean(0) if how == "mean" else X.amax(0)).tolist()


def keep_order_from_blocks(s: list) -> list:
    """block 分數 → 保留順序（1..63；和 m10_drop.unit_order 反過來同一個平手規則）。"""
    return sorted(CAND, key=lambda b: (s[b], b))[::-1]


def keep_order_from_tensor(DR, km, st, seed: int, tensor) -> list:
    """自己算的 token 分數 [L, H, n] → 保留順序：塞進一個 m10_drop.Ctx 的快取，用 unit_order 同一套彙總。"""
    c = DR.Ctx(km, st.ids, st.ctx_len, st.bf16, CHUNK, seed, [], [])
    c._scores["QRY"] = tensor
    return DR.unit_order(c, "B", "QRY")[::-1]


def compute_item(ans, it: dict, a, tgt_turn: int | None = None, keep: bool = False) -> dict:
    """一題：驗算答案 block → prefill → 每個分數的保留順序。tgt_turn＝目標輪（0 起；預設＝最後一輪：兩輪題＝第 2 輪、T5＝第 1 輪）。"""
    import torch
    import m10_drop as DR
    from m10_quality import ItemState
    km = ans.km
    km.key_keep = None
    nt = len(it["turns"])
    ti = nt - 1 if tgt_turn is None else tgt_turn
    tgt = f"QRY{ti + 1}"
    nbs, ver = {}, {}
    for t in range(nt):                                          # 先驗算（不過就停），免得白跑 prefill
        nb, lb, info = verify_answer_blocks(ans.tok, it, t)
        nbs[t] = nb
        ver[f"turn{t + 1}"] = dict(needle_blocks=nb, loc_blocks=lb, **info)
    timing = {}

    def timed(name, fn):
        torch.cuda.synchronize()
        t0 = time.time()
        r = fn()
        torch.cuda.synchronize()
        timing[name] = round(time.time() - t0, 3)
        return r

    st = timed("PREFILL", lambda: ItemState(ans, it))
    assert ans.CHUNK == CHUNK and st.n == N_BLOCKS and st.ctx_len == N_BLOCKS * CHUNK, (ans.CHUNK, st.n, st.ctx_len)
    trt = it["turns"][ti]
    q_t = trt["query_ids"]
    cx = DR.Ctx(km, st.ids, st.ctx_len, st.bf16, CHUNK, it["seed"], trt["needle_blocks"], q_t)
    kmin, kmax = timed("QUEST_MINMAX", lambda: block_minmax(st.bf16))
    orders, tens = {}, {}
    for meth in a.methods:
        tens[meth] = timed(meth, lambda meth=meth: DR.scores(cx, meth))
        orders[meth] = DR.unit_order(cx, "B", meth)[::-1]
    for meth in POSITION:
        orders[meth] = DR.unit_order(cx, "B", meth)[::-1]
    # 目標輪的 QRY：m10_drop 的定義原樣（問題直接接在 context 後面）
    tens["QRY_DIRECT"] = timed(f"{tgt}_DIRECT", lambda: DR.scores(cx, "QRY"))
    p_dir, q_dir = timed("PROBE_DIRECT", lambda: probe(cx, q_t, st.ctx_len, 0))
    mine = DR._pool_group(p_dir / len(q_t), km)
    checks = dict(probe_vs_drop_qry_bitwise=bool(torch.equal(mine, tens["QRY_DIRECT"])),
                  probe_vs_drop_qry_maxabs=float((mine.float() - tens["QRY_DIRECT"].float()).abs().max()))
    del mine
    placement = a.q2_placement if ti > 0 else "direct"           # 第 1 輪：接在 context 後面本來就是實際的位置
    if ti > 0:
        pre = []
        for t in range(ti):                                      # 實際序列（m10_tasks.full_prompt_ids）
            pre += it["turns"][t].get("history_ids", []) + it["turns"][t]["query_ids"]
        pre += trt.get("history_ids", [])
        p_in, q_in = timed("PROBE_INSITU", lambda: probe(cx, pre + q_t, st.ctx_len, len(pre)))
        tens["QRY_INSITU"] = DR._pool_group(p_in / len(q_t), km)
        if placement == "direct":
            prim, alt, p_sum, q_sum = "QRY_DIRECT", "QRY_INSITU", p_dir, q_dir
            del p_in, q_in
        else:
            prim, alt, p_sum, q_sum = "QRY_INSITU", "QRY_DIRECT", p_in, q_in
            del p_dir, q_dir
        orders[tgt] = keep_order_from_tensor(DR, km, st, it["seed"], tens[prim])
        orders[tgt + "_ALT"] = keep_order_from_tensor(DR, km, st, it["seed"], tens[alt])
        tr1 = it["turns"][0]
        cx1 = DR.Ctx(km, st.ids, st.ctx_len, st.bf16, CHUNK, it["seed"], tr1["needle_blocks"], tr1["query_ids"])
        tens["QRY1"] = timed("QRY1", lambda: DR.scores(cx1, "QRY"))
        orders["QRY1"] = DR.unit_order(cx1, "B", "QRY")[::-1]
        del cx1
    else:
        orders[tgt] = DR.unit_order(cx, "B", "QRY")[::-1]
        p_sum, q_sum = p_dir, q_dir
    Pq = timed("QUEST", lambda: quest_scores(km, kmin, kmax, q_sum))
    Pf = fullk_scores(p_sum, q_sum.shape[2], N_BLOCKS)
    checks["quest_finite"] = bool(torch.isfinite(Pq).all())
    checks["fullk_finite"] = bool(torch.isfinite(Pf).all())
    for name, P in (("QUEST", Pq), ("FULLK", Pf)):
        for how in ("mean", "max"):
            orders[f"{name}_{how.upper()}"] = keep_order_from_blocks(agg_blocks(P, a.quest_skip_layers, how))
    res = dict(orders=orders, target=tgt, ti=ti, placement=placement, timing=timing, checks=checks, verify=ver, needle=nbs)
    if keep:                                                     # selfcheck 用；一般 run 回傳後這些張量就放掉（呼叫端 empty_cache）
        res.update(tensors=tens, st=st, quest=Pq, fullk=Pf)
    return res


# ============================================================================ 每列的指標
def rank_metrics(order: list, target: list, ans_blocks, ans_t1) -> dict:
    assert sorted(order) == list(CAND) and sorted(target) == list(CAND), "保留順序必須是 1..63 的排列"
    pa = {b: i for i, b in enumerate(order)}
    pt = {b: i for i, b in enumerate(target)}
    n = N_CAND
    d2 = sum((pa[b] - pt[b]) ** 2 for b in CAND)
    out = {"spearman_vs_target": round(1 - 6 * d2 / (n * (n * n - 1)), 6), "top5": " ".join(map(str, order[:5])),
           "order": " ".join(map(str, order))}
    for k in KS:
        top = set(order[:k])
        out[f"overlap{k}_vs_target"] = round(len(top & set(target[:k])) / k, 6)
        if ans_blocks:
            A = set(ans_blocks)
            p_all, p_any = exp_hits(len(A), k)
            out.update({f"recall{k}": round(len(A & top) / len(A), 6), f"hit_all{k}": int(A <= top),
                        f"exp_hit_all{k}": round(p_all, 6), f"hit_any{k}": int(bool(A & top)),
                        f"exp_hit_any{k}": round(p_any, 6)})
        if ans_t1:
            A1 = set(ans_t1)
            out[f"recall_t1_{k}"] = round(len(A1 & top) / len(A1), 6)
    return out


def kind_of(m: str) -> str:
    if m in WRITE_SCORED:
        return "write"
    if m.startswith("RND"):
        return "random"
    if m in ("STREAM", "NEW"):
        return "position"
    if m.startswith("QRY"):
        return "qry"
    if m.startswith("QUEST"):
        return "quest"
    if m.startswith("FULLK"):
        return "fullk"
    return "other"


FIELDS = (["run_id", "ts", "shard", "model", "item_id", "task", "seed", "band1", "band2", "target", "q2_placement",
           "quest_skip_layers", "method", "kind", "needle_t1", "needle_tgt", "ans_loc_tgt"]
          + [f"{p}{k}" for k in KS for p in ("recall", "hit_all", "exp_hit_all", "hit_any", "exp_hit_any")]
          + [f"recall_t1_{k}" for k in KS] + ["spearman_vs_target"] + [f"overlap{k}_vs_target" for k in KS]
          + ["top5", "order", "t_score_s", "n_rows_item", "own_gpu_intruders", "host_level"])


def item_rows(it: dict, res: dict, a) -> list[dict]:
    ti, tgt = res["ti"], res["target"]
    ms = list(a.methods) + list(POSITION)
    ms += (["QRY1", tgt, tgt + "_ALT"] if ti > 0 else [tgt]) + list(SUMMARIES)
    T5 = it["task"] == "T5"
    ans_tgt = None if T5 else res["needle"][ti]
    ans_t1 = res["needle"][0] if (ti > 0 and not T5) else None
    turns = it["turns"]
    tm = res["timing"]
    t_dir, t_in = tm.get(f"{tgt}_DIRECT", ""), tm.get("PROBE_INSITU", "")
    tsc = {m: tm.get(m, "") for m in a.methods}
    tsc.update({"QRY1": tm.get("QRY1", ""), tgt: t_dir if res["placement"] == "direct" else t_in,
                tgt + "_ALT": t_in if res["placement"] == "direct" else t_dir,
                "QUEST_MEAN": tm.get("QUEST", ""), "QUEST_MAX": tm.get("QUEST", "")})   # ti＝0 時 tgt＝QRY1，後寫的蓋過前面
    base = dict(model=a.model, item_id=it["item_id"], task=it["task"], seed=it["seed"], band1=turns[0]["band"],
                band2=turns[1]["band"] if len(turns) > 1 else "", target=tgt, q2_placement=res["placement"],
                quest_skip_layers=a.quest_skip_layers, needle_t1=json.dumps(res["needle"][0]),
                needle_tgt=json.dumps(res["needle"][ti]),
                ans_loc_tgt=json.dumps(res["verify"][f"turn{ti + 1}"]["loc_blocks"]))
    rows = []
    for m in ms:
        r = dict(base, method=m, kind=kind_of(m), t_score_s=tsc.get(m, ""))
        r.update(rank_metrics(res["orders"][m], res["orders"][tgt], ans_tgt, ans_t1))
        rows.append(r)
    return rows


def _print_smoke(it: dict, res: dict, rows: list):
    print(f"\n=== smoke：{it['item_id']}（目標＝{res['target']}，第 2 輪問題放法＝{res['placement']}）", flush=True)
    for t, nb in res["needle"].items():
        v = res["verify"][f"turn{t + 1}"]
        print(f"  第 {t + 1} 輪答案 block（spans→pos//512）：{nb}；解碼搜尋找到答案字串的 block：{v['loc_blocks']}；驗算：{v['status']}")
    print(f"  {'方法':<12} {'前 5 名 block':<22} recall@16 recall@32 recall_t1@16 ρ(對目標) 重疊@16  秒")
    for r in rows:
        print(f"  {r['method']:<12} {r['top5']:<22} {str(r.get('recall16', '')):>9} {str(r.get('recall32', '')):>9} "
              f"{str(r.get('recall_t1_16', '')):>12} {r['spearman_vs_target']:>9} {r['overlap16_vs_target']:>7}  {r['t_score_s']}")
    print("  檢查：" + json.dumps(res["checks"], ensure_ascii=False))
    print("  時間（秒）：" + json.dumps(res["timing"], ensure_ascii=False), flush=True)


# ============================================================================ run
def plan(a) -> list:
    """(task, band1, band2, seed)：T2–T4 各 --n 題、T5 --n-t5 題，seed 從 --seed0 起（和 DS2 同一套：m10_quality.plan_items）。"""
    from m10_quality import plan_items
    t24 = [t for t in a.tasks if t != "T5"]
    return plan_items(t24, a.n, a.seed0) + (plan_items(["T5"], a.n_t5, a.seed0) if "T5" in a.tasks else [])


def item_id_of(task, b1, b2, seed) -> str:
    """和 m10_tasks.make_item 同一個寫法（續跑時不必先出題就能跳過；出題後再 assert 一次）。"""
    tb = ["global"] if task == "T5" else [b1] + ([b2] if b2 else [])
    return f"{task}_{'-'.join(tb)}_s{seed}"


def cmd_run(a):
    import torch
    import m10_tasks as T
    from m10_quality import Answerer
    if a.smoke:
        a.tasks, a.n, a.n_t5, a.shard = ["T2"], 1, 0, "0/1"
    unknown = [m for m in a.methods if m not in WRITE_SCORED]
    if unknown:
        raise SystemExit(f"--methods 只能是 {WRITE_SCORED}：{unknown}")
    ans = Answerer(a.model)
    P = json.load(open(a.params_file))["params"][a.model]
    si, sk = parse_shard(a.shard)
    done = done_items(a.out)
    fo, w = open_out(a.out, FIELDS)
    chk = open(side_path(a.out, ".checks.jsonl"), "a")
    watch = start_watch(a.out)
    rid = os.environ.get("RUN_ID", "adhoc")
    t_all, n_new, bad = time.time(), 0, []
    try:
        for k, (task, b1, b2, seed) in enumerate(plan(a)):
            if k % sk != si or item_id_of(task, b1, b2, seed) in done:
                continue
            t0 = time.time()
            it = T.make_item(task, b1, b2, seed, ans.tok, P.get(task))
            assert it["item_id"] == item_id_of(task, b1, b2, seed), it["item_id"]
            res = compute_item(ans, it, a)
            rows = item_rows(it, res, a)
            hl, intr = host_level(), intruders(watch)
            for r in rows:
                r.update(run_id=rid, ts=ts(), shard=a.shard, n_rows_item=len(rows), own_gpu_intruders=intr, host_level=hl)
                w.writerow(r)
            fo.flush()
            chk.write(json.dumps(dict(run_id=rid, ts=ts(), model=a.model, item_id=it["item_id"], checks=res["checks"],
                                      verify=res["verify"], timing=res["timing"], own_gpu_intruders=intr,
                                      host_level=hl), ensure_ascii=False, default=str) + "\n")
            chk.flush()
            if not all(v for kk, v in res["checks"].items() if isinstance(v, bool)):
                bad.append(it["item_id"])
            n_new += 1
            print(f"[{k + 1}] {it['item_id']} rows={len(rows)} {time.time() - t0:.1f}s avg {(time.time() - t_all) / n_new:.1f}s/題 "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB checks={res['checks']}", flush=True)
            if a.smoke:
                _print_smoke(it, res, rows)
            del res
            torch.cuda.empty_cache()
    finally:
        fo.close()
        chk.close()
        finish_watch(watch, a.out)
    if bad:
        print(f"檢查沒過的題（{len(bad)}）：{bad[:10]}（看 {side_path(a.out, '.checks.jsonl')}）", flush=True)
        if a.smoke:
            sys.exit(1)


# ============================================================================ selfcheck
def cmd_selfcheck(a):
    """同一題（預設 DS1 的第一題 T2_front-back_s60000）：m10_drop screen 的路徑 vs 本程式的路徑，逐值比對。不計分、不寫 results。"""
    import torch
    import m10_drop as DR
    import m10_sweep as SW
    import m10_tasks as T
    from m10_quality import Answerer
    ans = Answerer(a.model)
    P = json.load(open(a.params_file))["params"][a.model]
    it = T.make_item("T2", "front", "back", a.seed0, ans.tok, P.get("T2"))
    km = ans.km
    km.key_keep = None
    out = a.out or os.path.join(runs_root(), os.environ.get("RUN_ID", "adhoc"), f"c1a_selfcheck_{a.model}.json")
    checks, ok = [], True

    def chk(name, cond, **info):
        nonlocal ok
        ok &= bool(cond)
        checks.append({"check": name, "pass": bool(cond), **info})
        print(("PASS " if cond else "FAIL ") + name, info if info else "", flush=True)

    # A：m10_drop.cmd_screen 的路徑（第 418–432 行：SW.Item → DR.Ctx(…, st.needle, 第 1 輪 query_ids) → DR.scores）
    stA = SW.Item(ans, it)
    trA = it["turns"][0]
    cxA = DR.Ctx(km, stA.ids, stA.ctx_len, stA.bf16, ans.CHUNK, it["seed"], stA.needle, trA["query_ids"])
    A = {m: DR.scores(cxA, m).clone() for m in list(a.methods) + ["QRY"]}
    OA = {m: DR.unit_order(cxA, "B", m) for m in list(a.methods) + list(POSITION) + ["QRY"]}
    bfA = stA.bf16
    del cxA, stA
    # B：本程式（重新 prefill；目標輪設成第 1 輪 → 目標 QRY＝DS1 的 QRY）
    res = compute_item(ans, it, a, tgt_turn=0, keep=True)
    chk("prefill_bitwise（重新 prefill 的 64 個 BF16 block 和 A 逐位元相同）",
        all(torch.equal(x, y) for x, y in zip(bfA, res["st"].bf16)))
    for m in a.methods:
        B = res["tensors"][m]
        chk(f"score[{m}] 逐值相同", torch.equal(A[m], B), maxabs=float((A[m].float() - B.float()).abs().max()))
    B = res["tensors"]["QRY_DIRECT"]
    chk("score[QRY] 逐值相同", torch.equal(A["QRY"], B), maxabs=float((A["QRY"].float() - B.float()).abs().max()))
    for m in list(a.methods) + list(POSITION):
        chk(f"order[{m}] 相同", OA[m][::-1] == res["orders"][m])
    chk("order[QRY] 相同", OA["QRY"][::-1] == res["orders"]["QRY1"])
    chk("probe(tail_from=0) 和 m10_drop QRY 逐位元相同", res["checks"]["probe_vs_drop_qry_bitwise"],
        maxabs=res["checks"]["probe_vs_drop_qry_maxabs"])
    chk("Quest／FULLK 沒有 NaN／Inf", res["checks"]["quest_finite"] and res["checks"]["fullk_finite"])
    del res, A, bfA
    torch.cuda.empty_cache()
    # C：兩輪的路徑（目標＝第 2 輪；QRY2、QRY2_ALT、insitu 探針、Quest）跑得完、順序是 1..63 的排列
    res2 = compute_item(ans, it, a)
    rows = item_rows(it, res2, a)
    chk("two_turn_path（第 2 輪目標的每一列都算得出來）", len(rows) == len(a.methods) + len(POSITION) + 3 + len(SUMMARIES)
        and res2["checks"]["quest_finite"] and res2["checks"]["fullk_finite"], n_rows=len(rows),
        top5={r["method"]: r["top5"] for r in rows}, verify=res2["verify"])
    rep = {"model": a.model, "item_id": it["item_id"], "run_id": os.environ.get("RUN_ID", "adhoc"), "all_pass": bool(ok),
           "torch": torch.__version__, "hip": torch.version.hip, "note": "結構檢查，不計分", "checks": checks}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(rep, open(out, "w"), indent=1, ensure_ascii=False, default=str)
    print(("ALL PASS" if ok else "SOME FAILED") + f" → {out}", flush=True)
    sys.exit(0 if ok else 1)


# ============================================================================ report（CPU；不碰 torch）
GROUPS = {"T2-T4": ("T2", "T3", "T4"), "T2": ("T2",), "T3": ("T3",), "T4": ("T4",), "T5": ("T5",)}


def load_complete(paths) -> tuple[dict, dict]:
    """(model, item_id) → {method: row}；只留列數齊全的題（同一格重複時取最後一列）。另回傳不齊全的題。"""
    by = {}
    for r in read_rows(paths):
        by.setdefault((r["model"], r["item_id"]), {})[r["method"]] = r
    ok, partial = {}, {}
    for key, d in by.items():
        want = max(int(r["n_rows_item"]) for r in d.values())
        (ok if len(d) >= want else partial)[key] = d
    return ok, partial


def _f(r, col):
    return float(r[col]) if r.get(col, "") != "" else None


def _topk(r, k):
    return set(int(x) for x in r["order"].split()[:k])


def summarize(items: dict) -> dict:
    """items：item_id → {method: row}（同一個模型、同一組任務）。"""
    methods = []
    for d in items.values():
        methods += [m for m in d if m not in methods]
    out = {}
    for m in methods:
        rs = {i: d[m] for i, d in items.items() if m in d}
        e = dict(n=len(rs), spearman_vs_target=boot(_f(r, "spearman_vs_target") for r in rs.values()))
        for k in KS:
            rec = {i: _f(r, f"recall{k}") for i, r in rs.items() if _f(r, f"recall{k}") is not None}
            rnd = {}
            for i in rec:
                v = [_f(items[i][x], f"recall{k}") for x in ("RND0", "RND1", "RND2") if x in items[i]]
                if v and all(x is not None for x in v):
                    rnd[i] = statistics.mean(v)
            e[f"k{k}"] = dict(
                recall=boot(rec.values()),
                recall_minus_random=boot(v - k / N_CAND for v in rec.values()),
                recall_minus_rnd_paired=boot(rec[i] - rnd[i] for i in rec if i in rnd) if not m.startswith("RND") else None,
                hit_all_minus_exp=boot(_f(r, f"hit_all{k}") - _f(r, f"exp_hit_all{k}") for r in rs.values()
                                       if _f(r, f"hit_all{k}") is not None),
                hit_any_minus_exp=boot(_f(r, f"hit_any{k}") - _f(r, f"exp_hit_any{k}") for r in rs.values()
                                       if _f(r, f"hit_any{k}") is not None),
                recall_t1=boot(_f(r, f"recall_t1_{k}") for r in rs.values() if _f(r, f"recall_t1_{k}") is not None),
                overlap_vs_target=boot(_f(r, f"overlap{k}_vs_target") for r in rs.values()),
                overlap_minus_random=boot(_f(r, f"overlap{k}_vs_target") - k / N_CAND for r in rs.values()))
        out[m] = e
    # X4：Quest 摘要 vs 同一個 forward 的完整 K（同樣的彙總），前 k 名重疊；以及「完整 K 對目標的重疊 − Quest 對目標的重疊」
    for how in ("MEAN", "MAX"):
        qn, fn = f"QUEST_{how}", f"FULLK_{how}"
        both = [d for d in items.values() if qn in d and fn in d]
        for k in KS:
            out.setdefault(f"X4_{how}", {})[f"k{k}"] = dict(
                quest_vs_fullk_overlap=boot(len(_topk(d[qn], k) & _topk(d[fn], k)) / k for d in both),
                fullk_minus_quest_overlap_vs_target=boot(_f(d[fn], f"overlap{k}_vs_target") - _f(d[qn], f"overlap{k}_vs_target")
                                                         for d in both))
    return out


def stop_rules(s: dict) -> dict:
    """L1 §4 G2、G4 的成分（『≈隨機』＝95% CI 包含隨機期望；只對 T2–T4、每個 k 分開）。判讀照 L1 原文，由人決定。"""
    out = {}
    for k in KS:
        kk = f"k{k}"
        wr = {m: ci_has(s[m][kk]["recall_minus_random"]) for m in WRITE_SCORED if m in s}
        wr_rnd = {m: ci_has(s[m][kk]["recall_minus_rnd_paired"]) for m in WRITE_SCORED if m in s}
        quest = {m: ci_has(s[m][kk]["overlap_minus_random"]) for m in ("QUEST_MEAN", "QUEST_MAX") if m in s}
        qry1 = ci_has(s["QRY1"][kk]["recall_minus_random"]) if "QRY1" in s else None
        allw = all(wr.values()) if wr else None
        out[kk] = dict(
            write_time_recall_approx_random=wr, write_time_recall_approx_rnd_paired=wr_rnd,
            quest_overlap_with_target_approx_random=quest, qry1_recall_approx_random=qry1,
            G2_stop_with_QUEST_MEAN=(allw and quest.get("QUEST_MEAN")) if quest else None,
            G2_stop_with_QUEST_MAX=(allw and quest.get("QUEST_MAX")) if quest else None,
            G4_stop=(allw and qry1) if qry1 is not None else None)
    return out


L1_STOP_TEXT = {
    "G2": "寫入時的分數（SNAP、EXPATT、KNORM、KEYDIFF、H2O 近似）對第 2 輪答案所在 block 的召回率，和隨機的 CI 重疊；**而且** "
          "Quest min/max 摘要在 512-token block 上對 QRY 前 k 名的召回也≈隨機。兩者都成立 → 還原時能用的只剩位置（就是現在的 Cake 規則），G2 停止。",
    "G4": "寫入時的分數預測第 2 輪的能力≈隨機（同 G2）；**而且** 第 1 輪問題的 QRY 對第 2 輪也≈隨機。這表示在我們的任務上，"
          "跨輪的重要度不可預測，唯一的策略是無損保留＋問題來了再選（RocketKV-MT 式）。",
    "X2": "停止條件：見 G2、G4。H-X2、H-X4 都成立 → 選擇性還原只能靠位置與下一輪的問題；第三階段不做「寫入時分數」的放置。",
    "C1a": "寫入時分數≈隨機、且摘要也≈隨機 → 選擇性還原只能靠位置與下一輪的問題，C1b 不做（寫成負面結果）。（experiment_matrix.md §3）",
}


def fmt_ci(b, scale: float = 1.0, nd: int = 3) -> str:
    """平均 [CI] 的字串；scale＝100、nd＝1 → 百分點。"""
    if not b:
        return "—"
    return f"{scale * b['mean']:+.{nd}f} [{scale * b['lo']:+.{nd}f}, {scale * b['hi']:+.{nd}f}]"


def cmd_report(a):
    paths = sorted({p for g in a.raw for p in glob.glob(g)})
    items, partial = load_complete(paths)
    rows = [r for d in items.values() for r in d.values()]
    models = sorted({m for m, _ in items})
    rep = {"rule_text_L1": L1_STOP_TEXT, "approx_random": "95% bootstrap CI（以題目為單位、10,000 次）包含隨機期望",
           "random_expectation": {f"k{k}": k / N_CAND for k in KS}, "files": paths,
           "run_ids": sorted({r["run_id"] for r in rows}), "incomplete_items": sorted(f"{m}/{i}" for m, i in partial),
           "contention": {"rows_with_intruders": sum(1 for r in rows if r["own_gpu_intruders"] not in ("0", "NOT_CHECKED")),
                          "rows_not_checked": sum(1 for r in rows if r["own_gpu_intruders"] == "NOT_CHECKED"),
                          "host_levels": sorted({r["host_level"] for r in rows})},
           "q2_placement": sorted({r["q2_placement"] for r in rows}), "models": {}}
    L = ["# C1a：分數能不能預測第 2 輪要用的 block（L1 X2＋X4；只報告）", "",
         f"run：{'、'.join(rep['run_ids'])}；第 2 輪問題放法：{'、'.join(rep['q2_placement'])}；"
         f"不齊全的題（沒用）：{len(partial)}", "",
         f"recall@k 對隨機（k/63：{', '.join(f'{k}→{k / N_CAND:.3f}' for k in KS)}）的差，95% CI（題目為單位 bootstrap 10,000 次）。"
         "ρ＝和目標 QRY 的 Spearman。重疊＝前 k 名和目標前 k 名的重疊率對 k/63 的差。", ""]
    for m in models:
        rep["models"][m] = {}
        for g, tasks in GROUPS.items():
            its = {i: d for (mm, i), d in items.items() if mm == m and next(iter(d.values()))["task"] in tasks}
            if not its:
                continue
            s = summarize(its)
            rep["models"][m][g] = dict(n_items=len(its), methods=s)
            if g == "T2-T4":
                rep["models"][m][g]["stop_rule_components"] = stop_rules(s)
                bands = {}
                for i, d in its.items():
                    bands.setdefault(next(iter(d.values()))["band2"], {})[i] = d
                rep["models"][m][g]["by_band2"] = {b: {mm: {f"k{k}": boot(_f(d[mm], f"recall{k}") for d in bd.values()
                                                                          if mm in d) for k in KS}
                                                        for mm in s if not mm.startswith("X4_")}
                                                    for b, bd in sorted(bands.items())}
            L += [f"## {m}・{g}（{len(its)} 題）", "",
                  "| 方法 | " + " | ".join(f"recall@{k} − k/63" for k in KS) + " | ρ | "
                  + " | ".join(f"重疊@{k} − k/63" for k in KS) + " | recall_t1@16 |",
                  "|:--|" + "--:|" * (2 * len(KS) + 2)]
            for meth, e in s.items():
                if meth.startswith("X4_"):
                    continue
                L.append(f"| {meth} | " + " | ".join(fmt_ci(e[f'k{k}']['recall_minus_random'], 100, 1) for k in KS)
                         + f" | {fmt_ci(e['spearman_vs_target'])} | "
                         + " | ".join(fmt_ci(e[f'k{k}']['overlap_minus_random'], 100, 1) for k in KS)
                         + f" | {fmt_ci(e['k16']['recall_t1'], 100, 1)} |")
            for how in ("MEAN", "MAX"):
                x = s.get(f"X4_{how}")
                if x:
                    L.append(f"\nX4（{how}）：" + "；".join(
                        f"k={k} Quest∩完整K 前 k 重疊 {fmt_ci(x[f'k{k}']['quest_vs_fullk_overlap'])}、"
                        f"完整K−Quest 對目標的重疊 {fmt_ci(x[f'k{k}']['fullk_minus_quest_overlap_vs_target'])}" for k in KS))
            if g == "T2-T4":
                L.append("\n停止條件的成分（L1 G2、G4；≈隨機＝CI 包含隨機期望；判讀由人）：")
                L.append("```\n" + json.dumps(rep["models"][m][g]["stop_rule_components"], ensure_ascii=False, indent=1) + "\n```")
            L.append("")
    L += ["## L1 停止條件原文", ""] + [f"- **{k}**：{v}" for k, v in L1_STOP_TEXT.items()]
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(rep, open(os.path.join(a.out_dir, "c1a_summary.json"), "w"), indent=1, ensure_ascii=False)
    open(os.path.join(a.out_dir, "tables_c1a.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:60]))


# ============================================================================ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "selfcheck"):
        p = sp.add_parser(name)
        p.add_argument("--model", choices=["llama", "qwen"], required=True)
        p.add_argument("--params-file", default=PARAMS)
        p.add_argument("--methods", nargs="+", default=list(WRITE_SCORED), help="寫入時的分數（m10_drop）")
        p.add_argument("--q2-placement", choices=["direct", "insitu"], default="direct",
                       help="第 2 輪問題放哪：direct＝m10_drop QRY 的定義原樣（預設）；insitu＝實際的第 2 輪序列")
        p.add_argument("--quest-skip-layers", type=int, default=2, help="Quest／FULLK 彙總時略過前幾層（Quest 原文：2）")
        if name == "run":
            p.add_argument("--tasks", nargs="+", default=["T2", "T3", "T4", "T5"])
            p.add_argument("--n", type=int, default=45, help="T2–T4 每個任務幾題（L1 X2／14 §8.5：45）")
            p.add_argument("--n-t5", type=int, default=15, help="T5 幾題（15）")
            p.add_argument("--seed0", type=int, default=70000, help="DS2 的題目 seed（14 §8.5）")
            p.add_argument("--shard", default="0/1")
            p.add_argument("--out", required=True, help="逐列 CSV（每題 × 每分數一列）")
            p.add_argument("--smoke", action="store_true", help="1 題（T2 第一題）、印每個分數前 5 名 block 與答案 block")
        else:
            p.add_argument("--seed0", type=int, default=60000, help="DS1 的第一題（T2 front-back）")
            p.add_argument("--out", help="預設 $TIARA_RUNS/$RUN_ID/c1a_selfcheck_<model>.json")
    r = sp.add_parser("report")
    r.add_argument("--raw", nargs="+", required=True, help="逐列 CSV 的萬用字元（可多個）")
    r.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args()
    {"run": cmd_run, "selfcheck": cmd_selfcheck, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
