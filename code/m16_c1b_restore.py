"""m16_c1b_restore.py — 第三階段 C1b：選擇性還原（回來的問題只需要少數 block，只搬需要的部分）。

計畫：docs/stage3_20261019/experiment_matrix.md §3「C1b　選擇性還原：不載入 ≠ 丟掉」。怎麼跑、和設計卡的差異、
事先寫死的成功／停止條件：docs/stage3_20261019/c1b_howto.md。閘門 C1a 已過（results/m16_stage3_mi300x/c1a_predictive/）。

情境：一段對話做完第 1 輪後被卸載到下層（CPU／SSD／NFS），使用者帶著第 2 輪問題回來。完整還原把 64 個 block 全部搬回；
選擇性還原只把挑中的 block（block 1–63 裡挑 m＝round(k×63) 個；block 0 一律載入）搬回 GPU，其餘「留在下層、沒載入」
＝用 KVModel.key_keep 遮掉（RoPE 位置不變；下層的 KV 是無損的 BF16，補載＝把那個 block 的遮罩打開）。

每題（C1a 的題組與種子：DS2 兩輪題，seed 70000 起，T2–T4 各 45、T5 15；32K）：
  1. 打分：m16_c1a_predict.compute_item（和 C1a 同一段程式；寫入時分數只算本程式要用的那一種）→ 每個選法的保留順序，
     逐題和 C1a 的 CSV 逐字比對（order_eq_c1a）。
  2. 暫停前的狀態（完整 KV、sdpa 路徑）：第 1 輪問題＋greedy 生成（計分＝score_t1_full）；再把歷史（第 1 輪標準答案＋換手 token，
     m10_tasks 的 history_ids）也用完整 KV 接上。到這裡的 KV＝卸載前的快取。context 之後這一小段（tail）還原時一律載入。
     T5 是單輪題：暫停點＝context 剛 prefill 完，「回來」的就是它唯一的問題。
  3. 回來（每個格子各做一次；每格只寫第 2 輪問題起的位置，格子之間互不影響）：設好 key_keep → 接第 2 輪問題 → greedy 生成 →
     m10_tasks.score（第二階段同一個計分；T5 生成 120、其他 64 個 token）。
選法（保留順序都是 1..63 的排列；前 m 個載入）：
  QUEST_MEAN   C1a 原樣：第 2 輪問題直接接在 context 後面、**看完整 context** 算出每層的 post-RoPE q，打每個 block 的 K min／max 摘要。
               ⚠ 這個 q 要看過完整 context 才算得出來，還原前其實拿不到 → 偏樂觀。設計卡指定的選法，設計卡的判定照它。
  QUEST_BLIND  可部署版（本程式新增）：q 用「只看得到 block 0＋tail」的 forward 算（還原當下 GPU 上真的有的東西），
               第 2 輪問題放在它真正的位置；摘要與彙總和 QUEST_MEAN 相同。
  FULLK_MEAN   上界（C1a）：完整 K 的真實 attention 機率。選法本身要完整 K → oracle，位元組不含選法的成本。
  WRITE        寫入時最好的一種，照 C1a T2–T4 的 recall@16／@32：Llama＝KNORM、Qwen＝KEYDIFF（同一批題目上選的，偏差寫在 howto）。
  QRY1         第 1 輪問題的 attention（C1a；T5 只有一輪 → 就是當下的問題，kind＝qry_target）。
  POS          位置：只載前 k%（block 1..m；＝C1a 的 NEW 順序）。
  RND          隨機（C1a 的 RND0，每題固定種子）。
  基準：FULL（64 個 block 全載入，走和各格相同的遮罩路徑）；E0_SDPA（key_keep＝None＝第二階段的 sdpa 路徑；FULL−E0_SDPA＝換路徑的雜訊）。
變體：(a) 不補載（＝丟）；(b) 補載：每 REFILL_EVERY＝4 個生成 token 用當下的 q 重打 Quest 分數（第一次在第 2 輪問題 prefill 完、
  用問題自己的 q），門檻規則（寫死）：當下 Quest 分數（QUEST_MEAN 的彙總）前 m 名裡還沒載入的 block 全部補載
  （m＝初始載入的 block 數；載入的不再丟回去）。k ∈ {10, 25, 50}%（m＝6、16、32）。

位元組（照實算，bytes_row）：初始＝(1＋m) 個 block＋tail；摘要（用到 Quest 分數的格子）＝64 個 block × 每個 block 2×L×H×D 個 BF16
（K 的 min、max；常駐 GPU，保守地也算進搬運）；補載＝補載的 block 數 × block 位元組。分母＝完整還原（64 個 block＋tail）。
還原時間：report 用第二階段的 H2D／重算常數（bw.json、tier_params.json）與 I4 的每片成本（逐塊 vs 合併）算；
Quest 打分本身的時間由 qtime 在閒的 GPU 上實測。

子命令：
  selfcheck   GPU：2 題（T2_front-back_s70000、T5_global_s70000）：選法和 C1a 的 CSV 逐字相同、本程式的 forward 和 KVModel 逐位元相同、
              k＝100% 和完整還原逐值相同、補載的遮罩帳對得上、格子之間互不影響。不計分、不寫 results。
  qtime       GPU（閒）：Quest 選 block、可部署版的 q forward、摘要建立、每次補載檢查、單塊 H2D 的時間。污染 → 不寫 results、exit 3。
  run         GPU：--shard i/n、逐題續跑（同目錄所有 c1b_<模型>_s*.csv.gz 的聯集）、--budget-min；GpuWatcher 品質規則
              （污染時分數保留、逐列記爭用、t_gen_s 作廢）。
  report      CPU：表格＋事先寫死的判定（不 import torch）。
  pool-claim／pool-status  CPU：兩張卡共用的工作池（固定的 (模型, shard i/n) 清單、mkdir 當領取鎖）。
  --self-test CPU、不 import torch：出題與切片、工作池、位元組帳、補載規則、判定邏輯、時間模型、report 端到端（合成資料）。

模組頂層只有標準函式庫與 m16_c1a_predict（同樣沒有副作用）的 import；torch、m10_* 一律在函式裡才 import。
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from types import SimpleNamespace

from m16_c1a_predict import (CAND, CHUNK, N_BLOCKS, N_CAND, PARAMS, REPO, boot, done_items, finish_watch, fmt_ci, host_level,
                             intruders, item_id_of, keep_order_from_blocks, open_out, parse_shard, read_rows, runs_root,
                             side_path, start_watch, ts)

OUT_DIR = os.path.join(REPO, "results", "m16_stage3_mi300x", "c1_selective_restore")
RAW_DIR = os.environ.get("C1B_RAW", "/mlsteam/data/tiara/stage3/c1b")          # 逐列原始輸出（> 1 MB，不進 git）
C1A_DIR = os.path.join(REPO, "results", "m16_stage3_mi300x", "c1a_predictive")
BW_JSON = os.path.join(REPO, "results", "m10_precision_mi300x", "speed_{m}", "bw.json")
TIER_PARAMS = os.path.join(REPO, "results", "m7_write_policy_mi300x", "tier_params.json")
I4_CSV = os.path.join(REPO, "results", "m16_stage3_mi300x", "i4_copy_batching", "copybench_sdma1.csv")   # SDMA 開＝ROCm 預設

MODELS = ("llama", "qwen")
SHAPE = {"llama": dict(L=32, H=8, Hq=32, D=128), "qwen": dict(L=28, H=4, Hq=28, D=128)}   # run 時和 km 核對
TASKS_2T = ("T2", "T3", "T4")
ROTATIONS = [("front", "back"), ("mid", "front"), ("back", "mid")]   # ＝m10_quality.ROTATIONS（self-test 用 C1a 的題目集合核對）
SEED0, N_PER_TASK, N_T5 = 70000, 45, 15                              # C1a 的題組（DS2：14 §8.5）
KS_PCT = (10, 25, 50)
REFILL_EVERY = 4
WRITE_BEST = {"llama": "KNORM", "qwen": "KEYDIFF"}                   # C1a T2–T4：Llama KNORM +23.6／+26.4、Qwen KEYDIFF +19.2／+19.3
METHODS_A = ("QUEST_MEAN", "QUEST_BLIND", "FULLK_MEAN", "WRITE", "QRY1", "POS", "RND")
METHODS_B = ("QUEST_MEAN", "QUEST_BLIND", "WRITE", "QRY1", "POS", "RND")
QUEST_USERS = ("QUEST_MEAN", "QUEST_BLIND")                           # 選法本身用到摘要
QA_SET = ("QUEST_MEAN", "QUEST_BLIND")                               # query-aware、可比較的兩種（停止條件看這兩種）
SELFCHECK_ITEMS = (("T2", "front", "back", SEED0), ("T5", None, None, SEED0))

# 判定（事先寫死；c1b_howto.md §3）
EPS = 0.02               # 第二階段的 ε：答對率對完整還原的配對差 95% CI 下界 ≥ −2 個百分點（和 X2 的 M1 同一個寫法）
BYTES_MAX = 0.30         # 搬運位元組（含摘要、補載）÷ 完整還原 ≤ 30%（題目平均）
K_MAIN, K_STOP = 25, 50
MIN_ITEMS = 48           # 設計卡：主格 ≥ 48 題（每個模型 T2–T4）


def c1a_key(model: str) -> dict:
    """本程式的選法 → C1a CSV 的 method 名稱（order 欄逐字比對用）。QUEST_BLIND 是新的，C1a 沒有。"""
    return {"QUEST_MEAN": "QUEST_MEAN", "FULLK_MEAN": "FULLK_MEAN", "WRITE": WRITE_BEST[model], "QRY1": "QRY1",
            "POS": "NEW", "RND": "RND0"}


def kind_of(method: str, task: str) -> str:
    return {"E0_SDPA": "ref", "FULL": "ref", "QUEST_MEAN": "quest_fullctx_q", "QUEST_BLIND": "quest_deployable",
            "FULLK_MEAN": "oracle", "WRITE": "write", "QRY1": "qry_target" if task == "T5" else "qry1", "POS": "position",
            "RND": "random"}[method]


def m_of(k_pct: int) -> int:
    """k%（佔候選 63 個 block）→ 載入幾個候選 block。10→6、25→16、50→32（和 C1a 的 k＝16、32 對齊）。"""
    return int(round(k_pct / 100 * N_CAND))


def cell_list() -> list:
    """(格子名, 選法, k%, 變體)。E0_SDPA、FULL 一定排最前面（same_as_full 要用 FULL 的輸出）。"""
    cells = [("E0_SDPA", "E0_SDPA", 100, "ref"), ("FULL", "FULL", 100, "ref")]
    for v, ms in (("a", METHODS_A), ("b", METHODS_B)):
        for M in ms:
            for k in KS_PCT:
                cells.append((f"{M}@{k}{v}", M, k, v))
    return cells


N_CELLS = len(cell_list())


# ============================================================================ 出題、切片（純 Python）
def plan_c1b(n: int = N_PER_TASK, n_t5: int = N_T5, seed0: int = SEED0, tasks=TASKS_2T + ("T5",)) -> list:
    """C1a 的同一組 (task, band1, band2, seed)，只換處理順序：k 由小到大、每個 k 的三個任務輪替起點（交錯，任何一段前綴都涵蓋
    三個任务與三種位置帶），T5 放最後（時間不夠時先犧牲它）。第 j 題給 j mod n_shards 號 shard。"""
    t24 = [t for t in TASKS_2T if t in tasks]
    out = []
    for k in range(n):
        b1, b2 = ROTATIONS[k % 3]
        for t in range(len(t24)):
            out.append((t24[(t + k) % len(t24)], b1, b2, seed0 + k))
    if "T5" in tasks:
        out += [("T5", None, None, seed0 + k) for k in range(n_t5)]
    return out


def shard_items(plan: list, i: int, n: int) -> list:
    return [x for j, x in enumerate(plan) if j % n == i]


# ============================================================================ 工作池（兩張卡共用；純 Python＋檔案系統）
def pool_units(models=MODELS, n: int = 12) -> list:
    """固定的工作清單：(模型, shard)；兩個模型交錯（llama s0、qwen s0、llama s1、…），兩張卡的 worker 依序領。"""
    return [(m, i) for i in range(n) for m in models]


def unit_dir(pool: str, m: str, i: int, n: int) -> str:
    return os.path.join(pool, f"{m}_s{i}of{n}")


def raw_path(raw_dir: str, m: str, i: int) -> str:
    return os.path.join(raw_dir, f"c1b_{m}_s{i}.csv.gz")


def done_union(raw_dir: str, model: str) -> set:
    """同目錄所有 c1b_<模型>_s*.csv.gz 裡已經寫完的題（任何一個檔案寫完就算；切片數改了也不會重做）。"""
    out = set()
    for p in sorted(glob.glob(os.path.join(raw_dir, f"c1b_{model}_s*.csv.gz"))):
        out |= done_items(p)
    return out


def unit_remaining(raw_dir: str, m: str, i: int, n: int, plan=None, done=None) -> list:
    plan = plan if plan is not None else plan_c1b()
    done = done if done is not None else done_union(raw_dir, m)
    return [item_id_of(*x) for x in shard_items(plan, i, n) if item_id_of(*x) not in done]


def pool_claim(pool: str, units: list, n: int, worker: str, skip_fn=None):
    """依清單順序找第一個還沒被領的工作：os.mkdir 成功＝領到（原子操作；同一個工作只會有一個 worker 拿到）。
    skip_fn(m, i) 為真（例：該 shard 的題已全部寫完）→ 標成 SKIPPED、繼續找下一個。清單領完 → None。"""
    os.makedirs(pool, exist_ok=True)
    for m, i in units:
        d = unit_dir(pool, m, i, n)
        try:
            os.mkdir(d)
        except FileExistsError:
            continue
        info = dict(worker=worker, pid=os.getpid(), ts=ts(), model=m, shard=f"{i}/{n}")
        if skip_fn is not None and skip_fn(m, i):
            info["skipped"] = "該 shard 的題已全部寫完"
            json.dump(info, open(os.path.join(d, "SKIPPED"), "w"), ensure_ascii=False)
            continue
        json.dump(info, open(os.path.join(d, "CLAIM"), "w"), ensure_ascii=False)
        return m, i
    return None


def cmd_pool_claim(a):
    plan = plan_c1b()
    cache = {}

    def skip(m, i):
        if m not in cache:
            cache[m] = done_union(a.raw_dir, m)
        return not unit_remaining(a.raw_dir, m, i, a.n, plan, cache[m])

    u = pool_claim(a.pool, pool_units(a.models, a.n), a.n, a.worker, skip)
    if u is not None:
        print(f"{u[0]} {u[1]}")
        print(f"[pool-claim] {a.worker} 領到 {u[0]} shard {u[1]}/{a.n}", file=sys.stderr)
    else:
        print(f"[pool-claim] {a.worker}：工作池已領完", file=sys.stderr)


def cmd_pool_status(a):
    """第一行：ALL_COMPLETE 或 REMAINING <題數>。其後：每個模型、每個 shard 剩幾題；有給 --pool 時加上領取狀態。"""
    plan = plan_c1b()
    lines, rem_tot = [], 0
    for m in a.models:
        done = done_union(a.raw_dir, m)
        rem = {i: len(unit_remaining(a.raw_dir, m, i, a.n, plan, done)) for i in range(a.n)}
        rem_tot += sum(rem.values())
        lines.append(f"{m}: 已寫完 {len(done)}／{len(plan)} 題；各 shard 剩 " + " ".join(f"s{i}:{r}" for i, r in rem.items()))
        if a.pool:
            st = []
            for i in range(a.n):
                d = unit_dir(a.pool, m, i, a.n)
                st.append(f"s{i}:" + ("skip" if os.path.exists(os.path.join(d, "SKIPPED")) else
                                      "done" if os.path.exists(os.path.join(d, "DONE")) else
                                      "claimed" if os.path.isdir(d) else "-"))
            lines.append(f"{m} 工作池：" + " ".join(st))
    print("ALL_COMPLETE" if rem_tot == 0 else f"REMAINING {rem_tot}")
    print("\n".join(lines))


def cmd_pool_done(a):
    m, i = a.unit.split()
    d = unit_dir(a.pool, m, int(i), a.n)
    json.dump(dict(ts=ts(), rc=a.rc), open(os.path.join(d, "DONE"), "w"))


# ============================================================================ 位元組帳、補載規則（純 Python）
def shape_bytes(model: str) -> dict:
    s = SHAPE[model]
    chunk = s["L"] * 2 * s["H"] * CHUNK * s["D"] * 2                 # ＝KVModel.chunk_bytes（BF16）
    return dict(chunk=chunk, per_token=chunk // CHUNK, summary_block=2 * s["L"] * s["H"] * s["D"] * 2,
                frags_per_block=(CHUNK // 16) * s["L"] * 2, frag_bytes=16 * s["H"] * s["D"] * 2)


def bytes_row(model: str, n_init_blocks: int, n_refill: int, tail_tokens: int, uses_summary: bool) -> dict:
    """搬運位元組（BF16 無損）。n_init_blocks 含 block 0。摘要＝64 個 block 的 K min／max（BF16），用到 Quest 分數才算。"""
    b = shape_bytes(model)
    tail = tail_tokens * b["per_token"]
    full = N_BLOCKS * b["chunk"] + tail
    init = n_init_blocks * b["chunk"] + tail
    summ = N_BLOCKS * b["summary_block"] if uses_summary else 0
    refill = n_refill * b["chunk"]
    moved = init + summ + refill
    return dict(bytes_full=full, bytes_init=init, bytes_summary=summ, bytes_refill=refill, bytes_moved=moved,
                ratio_moved=round(moved / full, 6), ratio_moved_nosummary=round((init + refill) / full, 6),
                ratio_init=round((init + summ) / full, 6))


def refill_pick(block_scores: list, loaded: set, m: int) -> list:
    """補載的門檻規則（寫死）：當下分數的前 m 名（只在 1..63 裡排；同分照 keep_order_from_blocks 的規則）裡還沒載入的。"""
    assert len(block_scores) == N_BLOCKS and 0 < m <= N_CAND, (len(block_scores), m)
    return sorted(b for b in keep_order_from_blocks(block_scores)[:m] if b not in loaded)


def coverage(needle: list, loaded) -> tuple:
    if not needle:
        return "", ""
    s = set(needle)
    return round(len(s & set(loaded)) / len(s), 6), int(s <= set(loaded))


# ============================================================================ GPU：forward（KVModel.forward_span 的逐行複本，多留 q）
def fwd(km, ids, start: int, capture: bool = False):
    """m7_model.KVModel.forward_span（want_logits＝True、n_logits＝1）逐行照抄，只多一件事：capture 時留下每層的 post-RoPE q
    [L, Hq, n, D]（float32；補載的 Quest 打分用）。selfcheck 逐位元比對它和 KVModel 的 logits 與生成。"""
    import torch
    import torch.nn.functional as F
    from torch.nn.attention.bias import causal_lower_right
    n = ids.shape[-1]
    end = start + n
    assert end <= km.max_len, (end, km.max_len)
    m = km.hf.model
    with torch.inference_mode():
        qcap = torch.empty(km.L, km.Hq, n, km.D, device=km.device, dtype=torch.float32) if capture else None
        x = m.embed_tokens(ids.view(1, n))
        pos = torch.arange(start, end, device=km.device).view(1, n)
        cos, sin = m.rotary_emb(x, pos)
        key = (n, end)
        mask = km._masks.get(key)
        if mask is None:
            mask = causal_lower_right(n, end)
            km._masks[key] = mask
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
            if capture:
                qcap[li] = q[0].float()
            km.kv[li, 0, :, start:end].copy_(k[0])
            km.kv[li, 1, :, start:end].copy_(v[0])
            K = km.kv[li, 0, :, :end].unsqueeze(0)
            V = km.kv[li, 1, :, :end].unsqueeze(0)
            if km.key_keep is None:
                o = F.scaled_dot_product_attention(q, K, V, attn_mask=mask, enable_gqa=True)
            else:
                o = km._attn_keep(q, K, V, li, start, n, end)
            o = o.transpose(1, 2).reshape(1, n, km.Hq * km.D)
            x = x + a.o_proj(o)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        logits = km.hf.lm_head(m.norm(x[:, -1:]))[0, -1]
    return logits, qcap


def feed(km, ids: list, start: int, capture: bool = False):
    """m10_quality.Answerer.feed 的切法（每段 ≤ 512）；回傳 (最後一個位置的 logits, 所有 token 的 q 或 None)。"""
    import torch
    t = torch.tensor(ids, device=km.device)
    logits, caps, s = None, [], 0
    while s < len(ids):
        e = min(len(ids), s + CHUNK)
        logits, qc = fwd(km, t[s:e], start + s, capture)
        if capture:
            caps.append(qc)
        s = e
    return logits, (torch.cat(caps, dim=2) if capture else None)


class Refill:
    """(b) 補載。kk＝這一格的 key_keep（就地打開）；loaded＝目前已載入的 block（含 0）；m＝初始載入的候選 block 數。"""

    def __init__(self, km, kk, loaded, m: int, kmin, kmax, skip: int, every: int = REFILL_EVERY, enabled: bool = True):
        self.km, self.kk, self.loaded, self.m = km, kk, set(loaded), m
        self.kmin, self.kmax, self.skip, self.every, self.enabled = kmin, kmax, skip, every, enabled
        self.events, self.n_checks = [], 0

    def check(self, qcap, step: int):
        if not self.enabled:
            return
        from m16_c1a_predict import agg_blocks, quest_scores
        s = agg_blocks(quest_scores(self.km, self.kmin, self.kmax, qcap), self.skip, "mean")
        self.n_checks += 1
        new = refill_pick(s, self.loaded, self.m)
        if new:
            for b in new:
                self.kk[b * CHUNK:(b + 1) * CHUNK] = True
            self.loaded.update(new)
            self.events.append([step, new])

    @property
    def n_refill(self) -> int:
        return sum(len(b) for _, b in self.events)


def gen_turn(ans, q_ids: list, start: int, max_new: int, refill: Refill | None = None, keep_first: bool = False) -> dict:
    """接問題、greedy 生成：和 Answerer.feed＋Answerer.generate 同一個迴圈（停在 EOS 或 max_new）。
    補載：問題 prefill 完先檢查一次（用問題自己的 q）；之後每生成 every 個 token、用這 every 個 token 的 q 檢查一次。"""
    import torch
    km = ans.km
    cap = refill is not None
    logits, qc = feed(km, q_ids, start, capture=cap)
    first = logits.clone() if keep_first else None
    if cap:
        refill.check(qc, 0)
    out, pos, pend, stop = [], start + len(q_ids), [], "len"
    while len(out) < max_new:
        nxt = int(logits.argmax())
        if nxt in ans.eos:
            stop = "eos"
            break
        out.append(nxt)
        if len(out) == max_new:
            break
        logits, qc = fwd(km, torch.tensor([nxt], device=km.device), pos, capture=cap)
        pos += 1
        if cap:
            pend.append(qc)
            if len(out) % refill.every == 0:
                refill.check(torch.cat(pend, dim=2), len(out))
                pend = []
    return dict(out=out, stop=stop, first=first)


def blind_order(km, q_ids: list, p2: int, kmin, kmax, skip: int):
    """QUEST_BLIND：只看得到 block 0＋tail（context 之後全部）時算出的 q → Quest 摘要 → 保留順序。回傳 (順序, 每 block 分數)。"""
    from m16_c1a_predict import agg_blocks, quest_scores
    from m16_s2_equalbytes import keep_tensor
    km.key_keep = keep_tensor(km, list(CAND))
    try:
        _, qc = feed(km, q_ids, p2, capture=True)
    finally:
        km.key_keep = None
    s = agg_blocks(quest_scores(km, kmin, kmax, qc), skip, "mean")
    return keep_order_from_blocks(s), s


# ============================================================================ GPU：一題
def load_c1a_orders(c1a_dir: str, model: str) -> dict:
    """C1a 逐列 CSV → {(item_id, method): order 字串}。"""
    out = {}
    for r in read_rows(sorted(glob.glob(os.path.join(c1a_dir, f"c1a_{model}_s*.csv")))):
        out[(r["item_id"], r["method"])] = r["order"]
    return out


def prepare(ans, it: dict, a, c1a: dict | None) -> dict:
    """一題的前半：打分（C1a）→ 暫停前的狀態 → 每個選法的保留順序。"""
    import torch
    import m10_tasks as T
    from m10_quality import gen_len
    from m16_c1a_predict import block_minmax, compute_item
    km = ans.km
    km.key_keep = None
    for k, v in SHAPE[a.model].items():
        assert getattr(km, k) == v, (a.model, k, getattr(km, k), v)
    assert km.chunk_bytes == shape_bytes(a.model)["chunk"], (km.chunk_bytes, shape_bytes(a.model))
    tm = {}
    t0 = time.time()
    ns = SimpleNamespace(methods=[WRITE_BEST[a.model]], q2_placement="direct", quest_skip_layers=a.quest_skip_layers)
    res = compute_item(ans, it, ns, keep=True)
    st = res.pop("st")
    for k in ("tensors", "quest", "fullk"):
        res.pop(k, None)
    torch.cuda.empty_cache()
    tm["score_s"] = round(time.time() - t0, 3)
    ctx_len = st.ctx_len
    checks = dict(res["checks"])
    ctx_ok = all(bool(torch.equal(km.chunk_view(i), st.bf16[i])) for i in range(N_BLOCKS))
    checks["ctx_intact_after_scoring"] = ctx_ok
    if not ctx_ok:                                           # 打分的觀察窗應該都放回去了；沒有的話照 BF16 備份放回、記下來
        for i in range(N_BLOCKS):
            km.chunk_view(i).copy_(st.bf16[i])
    kmin, kmax = block_minmax(st.bf16)
    checks["summary_bytes_ok"] = bool(kmin.numel() * 2 * 2 == N_BLOCKS * shape_bytes(a.model)["summary_block"])
    turns = it["turns"]
    gl = gen_len(it)
    t1 = time.time()
    if len(turns) > 1:                                      # 暫停前：第 1 輪照常（完整 KV、sdpa）＋歷史
        tr1 = turns[0]
        g1 = gen_turn(ans, tr1["query_ids"], ctx_len, gl)
        score_t1 = T.score(tr1["metric"], ans.decode(g1["out"]), tr1["gold"])
        p_hist = ctx_len + len(tr1["query_ids"])
        feed(km, turns[1]["history_ids"], p_hist)
        p2 = p_hist + len(turns[1]["history_ids"])
        tre, ti = turns[1], 1
    else:
        score_t1, p2, tre, ti = "", ctx_len, turns[0], 0
    q = tre["query_ids"]
    assert p2 + len(q) + gl + 1 <= km.max_len, (p2, len(q), gl, km.max_len)
    tm["turn1_s"] = round(time.time() - t1, 3)
    tail_snap = km.kv[:, :, :, ctx_len:p2].clone()
    t2 = time.time()
    bo, bs = blind_order(km, q, p2, kmin, kmax, a.quest_skip_layers)
    tm["blind_s"] = round(time.time() - t2, 3)
    checks["quest_blind_finite"] = bool(all(math.isfinite(x) for x in bs))
    O = res["orders"]
    orders = {"QUEST_MEAN": O["QUEST_MEAN"], "QUEST_BLIND": bo, "FULLK_MEAN": O["FULLK_MEAN"], "WRITE": O[WRITE_BEST[a.model]],
              "QRY1": O["QRY1"], "POS": O["NEW"], "RND": O["RND0"]}
    for M, o in orders.items():
        if sorted(o) != list(CAND):
            raise RuntimeError(f"{it['item_id']} {M} 的保留順序不是 1..63 的排列（規則 2：停）")
    if orders["POS"] != list(CAND):
        raise RuntimeError(f"{it['item_id']} POS（C1a NEW）不是 1..63 的前段順序：{orders['POS'][:5]}")
    eq = {}
    for M, key in c1a_key(a.model).items():
        ref = (c1a or {}).get((it["item_id"], key))
        eq[M] = None if ref is None else (" ".join(map(str, orders[M])) == ref)
    checks["order_eq_c1a_all"] = None if any(v is None for v in eq.values()) else all(eq.values())
    return dict(st=st, res=res, kmin=kmin, kmax=kmax, ctx_len=ctx_len, p2=p2, tail=p2 - ctx_len, q=q, tre=tre, ti=ti,
                needle=res["needle"][ti], score_t1=score_t1, orders=orders, order_eq=eq, checks=checks,
                tail_snap=tail_snap, gl=gl, timing=tm)


def loaded_init(S: dict, M: str, k: int) -> tuple:
    if M in ("E0_SDPA", "FULL"):
        return set(range(N_BLOCKS)), N_CAND
    m = m_of(k)
    return {0} | set(S["orders"][M][:m]), m


def run_cell(ans, S: dict, M: str, k: int, v: str, a, enabled: bool = True, keep_first: bool = False) -> dict:
    """一個格子：設遮罩 → 接問題 → 生成（(b) 時邊生成邊補載）。回傳生成結果與補載紀錄。"""
    import torch
    from m16_s2_equalbytes import keep_tensor
    km = ans.km
    loaded, m = loaded_init(S, M, k)
    kk = None if M == "E0_SDPA" else keep_tensor(km, sorted(set(range(N_BLOCKS)) - loaded))
    rf = Refill(km, kk, loaded, m, S["kmin"], S["kmax"], a.quest_skip_layers, a.refill_every, enabled) if v == "b" else None
    km.key_keep = kk
    torch.cuda.synchronize()
    t0 = time.time()
    try:
        g = gen_turn(ans, S["q"], S["p2"], S["gl"], rf, keep_first=keep_first)
    finally:
        km.key_keep = None
    torch.cuda.synchronize()
    g.update(t_gen_s=round(time.time() - t0, 3), loaded0=loaded, m=m, kk=kk,
             loaded_final=(rf.loaded if rf else loaded), events=(rf.events if rf else []),
             n_checks=(rf.n_checks if rf else 0), n_refill=(rf.n_refill if rf else 0))
    return g


def prepause_intact(km, S: dict) -> bool:
    import torch
    st = S["st"]
    return bool(all(torch.equal(km.chunk_view(i), st.bf16[i]) for i in range(N_BLOCKS))
                and torch.equal(km.kv[:, :, :, S["ctx_len"]:S["p2"]], S["tail_snap"]))


FIELDS = ["run_id", "ts", "shard", "model", "item_id", "task", "seed", "band1", "band2", "turn_eval", "metric", "cell", "method",
          "kind", "k_pct", "m_blocks", "variant", "refill_every", "write_method", "n_init_blocks", "n_refill_blocks",
          "n_refill_events", "n_refill_checks", "refill_log", "loaded_final", "needle_tgt", "needle_cov_init",
          "needle_cov_final", "needle_all_init", "needle_all_final", "tail_tokens", "bytes_full", "bytes_init", "bytes_summary",
          "bytes_refill", "bytes_moved", "ratio_moved", "ratio_moved_nosummary", "ratio_init", "oracle_selection", "gen_len",
          "n_gen", "stop", "pred", "score", "f1", "same_as_full", "score_t1_full", "order_eq_c1a", "t_gen_s", "n_rows_item",
          "own_gpu_intruders", "host_level"]


def make_row(model: str, it: dict, S: dict, cell: str, M: str, k: int, v: str, g: dict, pred: str, score: float, f1,
             refill_every: int) -> dict:
    """一列（run_id、ts、shard、n_rows_item、爭用欄由呼叫端補）。純 Python（self-test 用合成的 g、S 核對欄位）。"""
    turns = it["turns"]
    nd = list(S["needle"])                       # T5（全域）：m10_tasks 記的是常見詞每一次出現所在的 block，覆蓋率只當參考
    ci, ai = coverage(nd, g["loaded0"])
    cf, af = coverage(nd, g["loaded_final"])
    uses_summary = (M in QUEST_USERS) or v == "b"
    by = bytes_row(model, len(g["loaded0"]), g["n_refill"], S["tail"], uses_summary)
    eq = S["order_eq"].get(M) if M in S["order_eq"] else None
    return dict(model=model, item_id=it["item_id"], task=it["task"], seed=it["seed"], band1=turns[0]["band"],
                band2=turns[1]["band"] if len(turns) > 1 else "", turn_eval=S["ti"] + 1, metric=S["tre"]["metric"],
                cell=cell, method=M, kind=kind_of(M, it["task"]), k_pct=k, m_blocks=g["m"], variant=v,
                refill_every=refill_every if v == "b" else "", write_method=WRITE_BEST[model] if M == "WRITE" else "",
                n_init_blocks=len(g["loaded0"]), n_refill_blocks=g["n_refill"], n_refill_events=len(g["events"]),
                n_refill_checks=g["n_checks"], refill_log=json.dumps(g["events"]), loaded_final=json.dumps(sorted(g["loaded_final"])),
                needle_tgt=json.dumps(nd), needle_cov_init=ci, needle_cov_final=cf, needle_all_init=ai, needle_all_final=af,
                tail_tokens=S["tail"], **by, oracle_selection=int(M == "FULLK_MEAN"), gen_len=S["gl"], n_gen=len(g["out"]),
                stop=g["stop"], pred=pred, score=score, f1=f1, same_as_full="", score_t1_full=S["score_t1"],
                order_eq_c1a="" if eq is None else int(eq), t_gen_s=g["t_gen_s"])


def c1b_item(ans, it: dict, a, c1a: dict | None) -> tuple:
    """一題的全部格子。回傳 (rows, checks)。"""
    import torch
    import m10_tasks as T
    km = ans.km
    S = prepare(ans, it, a, c1a)
    rows, outs = [], {}
    for cell, M, k, v in cell_list():
        g = run_cell(ans, S, M, k, v, a)
        pred = ans.decode(g["out"])
        sc = T.score(S["tre"]["metric"], pred, S["tre"]["gold"])
        f1 = T.qa_f1(pred, S["tre"]["gold"]) if it["task"] == "T4" else ""
        outs[cell] = g["out"]
        rows.append(make_row(a.model, it, S, cell, M, k, v, g, pred, sc, f1, a.refill_every))
        del g
    for r in rows:
        r["same_as_full"] = int(outs[r["cell"]] == outs["FULL"])
    ok = prepause_intact(km, S)
    S["checks"]["prepause_intact_after_cells"] = ok
    if not ok:                                              # 格子之間互相污染＝程式錯誤（規則 2：停）
        raise RuntimeError(f"{it['item_id']}：跑完各格後，暫停前的 KV（context 或 tail）被改動了")
    checks = dict(S["checks"], order_eq_c1a=S["order_eq"], timing=S["timing"], score_t1_full=S["score_t1"],
                  needle=S["needle"], p2=S["p2"], tail=S["tail"])
    del S
    torch.cuda.empty_cache()
    return rows, checks


# ============================================================================ run
def cmd_run(a):
    import torch
    import m10_tasks as T
    from m10_quality import Answerer
    ans = Answerer(a.model)
    P = json.load(open(a.params_file))["params"][a.model]
    c1a = load_c1a_orders(a.c1a_dir, a.model)
    print(f"C1a 的保留順序 {len(c1a)} 列（{a.c1a_dir}）", flush=True)
    si, sk = parse_shard(a.shard)
    done = done_union(os.path.dirname(os.path.abspath(a.out)), a.model)
    todo = [x for x in shard_items(plan_c1b(a.n, a.n_t5, a.seed0, a.tasks), si, sk) if item_id_of(*x) not in done]
    print(f"shard {a.shard}：這個 shard 還有 {len(todo)} 題（同目錄已寫完 {len(done)} 題）", flush=True)
    if not todo:
        return
    fo, w = open_out(a.out, FIELDS)
    chk = open(side_path(a.out, ".checks.jsonl"), "a")
    watch = start_watch(a.out)
    rid = os.environ.get("RUN_ID", "adhoc")
    t_all, n_new, bad = time.time(), 0, []
    try:
        for task, b1, b2, seed in todo:
            if a.budget_min and (time.time() - t_all) / 60 > a.budget_min:
                print(f"時間預算 {a.budget_min} 分鐘到了：剩下的題留給下一個 job（逐題續跑）", flush=True)
                break
            t0 = time.time()
            it = T.make_item(task, b1, b2, seed, ans.tok, P.get(task))
            assert it["item_id"] == item_id_of(task, b1, b2, seed), it["item_id"]
            rows, checks = c1b_item(ans, it, a, c1a)
            hl, intr = host_level(), intruders(watch)
            for r in rows:
                r.update(run_id=rid, ts=ts(), shard=a.shard, n_rows_item=len(rows), own_gpu_intruders=intr, host_level=hl)
                w.writerow(r)
            fo.flush()
            chk.write(json.dumps(dict(run_id=rid, ts=ts(), model=a.model, item_id=it["item_id"], checks=checks,
                                      own_gpu_intruders=intr, host_level=hl), ensure_ascii=False, default=str) + "\n")
            chk.flush()
            flat = [k for k, v in checks.items() if v is False] + [f"order_eq_c1a:{k}" for k, v in checks["order_eq_c1a"].items()
                                                                  if v is False]
            if flat:
                bad.append((it["item_id"], flat))
            n_new += 1
            main = {r["cell"]: r["score"] for r in rows if r["cell"] in ("FULL", "QUEST_MEAN@25a", "QUEST_MEAN@25b",
                                                                           "QUEST_BLIND@25a", "QUEST_BLIND@25b")}
            print(f"[{n_new}/{len(todo)}] {it['item_id']} rows={len(rows)} {time.time() - t0:.1f}s "
                  f"avg {(time.time() - t_all) / n_new:.1f}s/題 peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB "
                  f"t1={checks['score_t1_full']} {main} 檢查不過={flat}", flush=True)
    finally:
        fo.close()
        chk.close()
        finish_watch(watch, a.out)
    if bad:
        print(f"檢查有 false 的題（{len(bad)}）：{bad[:10]}（看 {side_path(a.out, '.checks.jsonl')}；report 會列出）", flush=True)


# ============================================================================ selfcheck
def cmd_selfcheck(a):
    """2 題（T2_front-back_s70000、T5_global_s70000，都在 C1a 的 CSV 裡）。不計分、不寫 results。"""
    import torch
    import m10_tasks as T
    from m10_quality import Answerer, run_turns
    from m16_s2_equalbytes import keep_tensor
    ans = Answerer(a.model)
    km = ans.km
    P = json.load(open(a.params_file))["params"][a.model]
    c1a = load_c1a_orders(a.c1a_dir, a.model)
    out = a.out or os.path.join(runs_root(), os.environ.get("RUN_ID", "adhoc"), f"c1b_selfcheck_{a.model}.json")
    checks, ok, info = [], True, {}

    def chk(name, cond, **kw):
        nonlocal ok
        ok &= bool(cond)
        checks.append({"check": name, "pass": bool(cond), **kw})
        print(("PASS " if cond else "FAIL ") + name, kw if kw else "", flush=True)

    for task, b1, b2, seed in SELFCHECK_ITEMS:
        it = T.make_item(task, b1, b2, seed, ans.tok, P.get(task))
        iid = it["item_id"]
        S = prepare(ans, it, a, c1a)
        chk(f"{iid} 選法和 C1a 的 CSV 逐字相同", S["checks"]["order_eq_c1a_all"] is True, per_method=S["order_eq"])
        chk(f"{iid} 打分後 context 的 KV 沒被改動", S["checks"]["ctx_intact_after_scoring"])
        chk(f"{iid} C1a 的逐題檢查（probe 對 m10_drop 逐位元、Quest／FULLK 有限）",
            all(S["checks"][k] for k in ("probe_vs_drop_qry_bitwise", "quest_finite", "fullk_finite")))
        chk(f"{iid} QUEST_BLIND 的分數有限、順序是 1..63 的排列",
            S["checks"]["quest_blind_finite"] and sorted(S["orders"]["QUEST_BLIND"]) == list(CAND),
            top5=S["orders"]["QUEST_BLIND"][:5], quest_mean_top5=S["orders"]["QUEST_MEAN"][:5])
        chk(f"{iid} 摘要位元組＝64×2×L×H×D×2", S["checks"]["summary_bytes_ok"])
        q, p2, gl = S["q"], S["p2"], S["gl"]
        # C：本程式的 forward／生成迴圈 vs KVModel（Answerer.feed＋generate），三種遮罩
        masks = {"none(sdpa)": None, "all_true": keep_tensor(km, []),
                 "QUEST_MEAN@25": keep_tensor(km, sorted(set(range(N_BLOCKS)) - loaded_init(S, "QUEST_MEAN", 25)[0]))}
        for name, kk in masks.items():
            km.key_keep = kk
            try:
                lg = ans.feed(q, p2)
                first1 = lg.clone()
                out1, stop1 = ans.generate(lg, p2 + len(q), gl)
                g2 = gen_turn(ans, q, p2, gl, keep_first=True)
                rf = Refill(km, kk, set(range(N_BLOCKS)), N_CAND, S["kmin"], S["kmax"], a.quest_skip_layers, a.refill_every,
                            enabled=False)
                g3 = gen_turn(ans, q, p2, gl, rf, keep_first=True)
            finally:
                km.key_keep = None
            chk(f"{iid} forward 複本 vs KVModel［{name}］：第一個 logits 逐位元、生成 token 相同",
                bool(torch.equal(first1, g2["first"])) and out1 == g2["out"] and stop1 == g2["stop"],
                maxabs=float((first1.float() - g2["first"].float()).abs().max()), n_gen=len(out1))
            chk(f"{iid} 留 q（補載關掉）不改變結果［{name}］",
                bool(torch.equal(g2["first"], g3["first"])) and g2["out"] == g3["out"])
        # D：k＝100% 和完整還原逐值相同
        gF = run_cell(ans, S, "FULL", 100, "ref", a, keep_first=True)
        assert m_of(100) == N_CAND
        for M in METHODS_A:
            g = run_cell(ans, S, M, 100, "a", a, keep_first=True)          # m＝round(1.00×63)＝63 → 前 63 名＝全部
            chk(f"{iid} k＝100%［{M}］：載入全部 64 個 block、遮罩和 FULL 相同、logits 與生成逐值相同",
                g["loaded0"] == set(range(N_BLOCKS)) and bool(torch.equal(g["kk"], gF["kk"]))
                and bool(torch.equal(g["first"], gF["first"])) and g["out"] == gF["out"])
            del g
        gB = run_cell(ans, S, "QUEST_BLIND", 100, "b", a, keep_first=True)
        chk(f"{iid} k＝100%（b）：沒有可補的、結果和 FULL 相同", gB["n_refill"] == 0 and gB["out"] == gF["out"]
            and bool(torch.equal(gB["first"], gF["first"])), n_checks=gB["n_checks"])
        # E：補載的遮罩帳
        gR = run_cell(ans, S, "RND", 10, "b", a)
        nm = int((~gR["kk"][:S["ctx_len"]]).sum())
        ev_blocks = [b for _, bs in gR["events"] for b in bs]
        chk(f"{iid} 補載（RND@10b）：遮罩裡沒載入的 token 數＝(64−已載入)×512、補的 block 在 1..63 且原本沒載入",
            nm == (N_BLOCKS - len(gR["loaded_final"])) * CHUNK and gR["loaded_final"] >= gR["loaded0"]
            and all(1 <= b <= 63 for b in ev_blocks) and not (set(ev_blocks) & gR["loaded0"])
            and len(ev_blocks) == len(set(ev_blocks)) == gR["n_refill"]
            and bytes_row(a.model, len(gR["loaded0"]), gR["n_refill"], S["tail"], True)["bytes_refill"]
            == gR["n_refill"] * km.chunk_bytes,
            n_refill=gR["n_refill"], events=gR["events"], n_checks=gR["n_checks"])
        gA = run_cell(ans, S, "RND", 10, "a", a, keep_first=True)
        gN = run_cell(ans, S, "RND", 10, "b", a, enabled=False, keep_first=True)
        chk(f"{iid} (b) 把補載關掉＝(a)（逐位元）", bool(torch.equal(gA["first"], gN["first"])) and gA["out"] == gN["out"])
        # F：格子之間互不影響
        chk(f"{iid} 跑完各格後，暫停前的 KV（context＋tail）逐位元沒變", prepause_intact(km, S), tail=S["tail"])
        # 給人看（不計入 pass）：和第二階段 run_turns 的 E0（歷史和第 2 輪問題一起接）比，最後一輪的生成是否相同
        gE = run_cell(ans, S, "E0_SDPA", 100, "ref", a)
        rt = run_turns(S["st"], "bf16", [frozenset()] * len(it["turns"]))
        info[iid] = dict(e0_same_tokens_as_stage2_run_turns=(rt[-1]["gen"] == gE["out"]),
                         e0_pred=ans.decode(gE["out"])[:200], full_pred=ans.decode(gF["out"])[:200],
                         gold=S["tre"]["gold"], score_t1_full=S["score_t1"], tail=S["tail"], timing=S["timing"],
                         refill_rnd10=dict(n_refill=gR["n_refill"], events=gR["events"]))
        print(f"給人看：{iid} {json.dumps(info[iid], ensure_ascii=False, default=str)}", flush=True)
        del S, gF, gB, gR, gA, gN, gE, rt
        torch.cuda.empty_cache()
    rep = {"model": a.model, "run_id": os.environ.get("RUN_ID", "adhoc"), "all_pass": bool(ok),
           "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
           "torch": torch.__version__, "hip": torch.version.hip, "note": "結構檢查，不計分", "checks": checks, "info": info}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(rep, open(out, "w"), indent=1, ensure_ascii=False, default=str)
    print(("ALL PASS" if ok else "SOME FAILED") + f" peak {rep['peak_reserved_gib']} GiB（reserved） → {out}", flush=True)
    sys.exit(0 if ok else 1)


# ============================================================================ qtime（GPU 閒）
QTIME_FIELDS = ["run_id", "ts", "model", "op", "n_tokens", "bytes", "reps", "t_med_ms", "t_q1_ms", "t_q3_ms", "t_min_ms", "GiBps",
                "note", "gpu_verdict", "own_gpu_intruders", "host_level", "torch", "hip"]


def cmd_qtime(a):
    """Quest 選 block 本身的時間（閒的 GPU、同步計時、暖身 3 次、量 --reps 次取中位數）。污染或監看開不起來 → 不寫 results、exit 3。"""
    import torch
    import m10_tasks as T
    from m10_quality import Answerer
    from m16_c1a_predict import agg_blocks, block_minmax, quest_scores
    from m16_s2_equalbytes import keep_tensor
    ans = Answerer(a.model)
    km = ans.km
    P = json.load(open(a.params_file))["params"][a.model]
    rid = os.environ.get("RUN_ID", "adhoc")
    rdir = os.path.join(runs_root(), rid)
    os.makedirs(rdir, exist_ok=True)
    tmp = os.path.join(rdir, os.path.basename(a.out))
    watch = start_watch(tmp)
    it = T.make_item("T2", "front", "back", SEED0, ans.tok, P.get("T2"))
    S = prepare(ans, it, a, None)
    st, q, p2, skip = S["st"], S["q"], S["p2"], a.quest_skip_layers
    ld25, m25 = loaded_init(S, "QUEST_MEAN", 25)
    kk25 = keep_tensor(km, sorted(set(range(N_BLOCKS)) - ld25))
    kkb = keep_tensor(km, list(CAND))

    def timeit(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        xs = []
        for _ in range(a.reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            xs.append((time.perf_counter() - t0) * 1e3)
        xs.sort()
        qq = statistics.quantiles(xs, n=4) if len(xs) >= 4 else [xs[0], statistics.median(xs), xs[-1]]
        return dict(t_med_ms=round(statistics.median(xs), 4), t_q1_ms=round(qq[0], 4), t_q3_ms=round(qq[2], 4),
                    t_min_ms=round(xs[0], 4))

    def with_keep(kk, fn):
        def g():
            km.key_keep = kk
            try:
                return fn()
            finally:
                km.key_keep = None
        return g

    _, qc_q2 = with_keep(kkb, lambda: feed(km, q, p2, capture=True))()
    qc_step = qc_q2[:, :, -REFILL_EVERY:].contiguous()
    host = km.host_chunk()
    host.copy_(st.bf16[5])
    rows = []

    def add(op, n_tok, nbytes, r, note):
        gib = round(nbytes / 2**30 / (r["t_med_ms"] / 1e3), 3) if nbytes else ""
        rows.append(dict(model=a.model, op=op, n_tokens=n_tok, bytes=nbytes, reps=a.reps, GiBps=gib, note=note, **r))
        print(op, r, flush=True)

    add("minmax_64blocks", N_BLOCKS * CHUNK, 0, timeit(lambda: block_minmax(st.bf16)),
        "寫入（卸載）時建摘要：64 個 block 的 K 逐維 min／max（不在還原的關鍵路徑上）")
    add("blind_q_forward", len(q), 0, timeit(with_keep(kkb, lambda: feed(km, q, p2, capture=True))),
        "QUEST_BLIND 要的 q：第 2 輪問題只看 block 0＋tail 的 forward（遮罩路徑仍對全部 key 算，會高估）")
    add("quest_select_q2", len(q), 0,
        timeit(lambda: keep_order_from_blocks(agg_blocks(quest_scores(km, S["kmin"], S["kmax"], qc_q2), skip, "mean"))),
        "Quest 選 block：問題所有 token 的 q 打 64 個 block 的摘要 → 彙總 → 排序（含 .tolist() 同步）")
    add("quest_refill_check", REFILL_EVERY, 0,
        timeit(lambda: refill_pick(agg_blocks(quest_scores(km, S["kmin"], S["kmax"], qc_step), skip, "mean"), ld25, m25)),
        f"(b) 每次補載檢查：{REFILL_EVERY} 個 token 的 q（每 {REFILL_EVERY} 個生成 token 一次）")
    add("prefill_q2_sdpa", len(q), 0, timeit(lambda: feed(km, q, p2)), "參考：第 2 輪問題的 prefill（完整 KV、sdpa）")
    add("prefill_q2_mask_k25", len(q), 0, timeit(with_keep(kk25, lambda: feed(km, q, p2))),
        "參考：同上、k＝25% 的遮罩路徑（仍對全部 key 算，不代表真的少算）")
    tok = torch.tensor([int(ans.tok.eos_token_id or 0)], device=km.device)
    add("decode_step_sdpa", 1, 0, timeit(lambda: fwd(km, tok, p2 + len(q))), "參考：一個 decode step（完整 KV、sdpa）")
    add("h2d_block_pinned", CHUNK, km.chunk_bytes, timeit(lambda: km.chunk_view(5).copy_(host, non_blocking=True)),
        "尺準：一個 BF16 block 從 pinned host 到 GPU 的分段視圖（和第二階段 bw.json 的 real_pinned_h2d 同一種 copy）")
    if not torch.equal(km.chunk_view(5), st.bf16[5]):
        raise RuntimeError("h2d 量測後 block 5 和 BF16 備份不同（規則 2：停）")
    verdict = "NOT_CHECKED" if watch is None else None
    contaminated = finish_watch(watch, tmp)
    if verdict is None:
        verdict = watch.verdict()
    hl, intr = host_level(), intruders(watch)
    for r in rows:
        r.update(run_id=rid, ts=ts(), gpu_verdict=verdict, own_gpu_intruders=intr, host_level=hl, torch=torch.__version__,
                 hip=torch.version.hip)
    for p in (tmp,) + (() if (watch is None or contaminated) else (a.out,)):
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=QTIME_FIELDS)
            w.writeheader()
            w.writerows(rows)
    if watch is None or contaminated:
        print(f"qtime：GPU {verdict} → 不寫 results（只留在 {tmp}）；Quest 打分時間記 NOT_MEASURED", flush=True)
        sys.exit(3)
    print(f"qtime：GPU CLEAN → {a.out}", flush=True)


# ============================================================================ report（CPU；不碰 torch）
def load_items(paths) -> tuple:
    """(model, item_id) → {cell: row}。同一題有好幾次嘗試（被砍後重做）時，取最後一個列數齊全的 run_id，不混用。
    另回傳不齊全的題。"""
    by = {}
    for r in read_rows(paths):
        by.setdefault((r["model"], r["item_id"]), {}).setdefault(r["run_id"], {})[r["cell"]] = r
    ok, partial = {}, []
    for key, att in by.items():
        good = [(max(x["ts"] for x in d.values()), rid) for rid, d in att.items()
                if len(d) >= max(int(x["n_rows_item"]) for x in d.values())]
        if good:
            ok[key] = att[max(good)[1]]
        else:
            partial.append(f"{key[0]}/{key[1]}")
    return ok, sorted(partial)


def _f(x):
    return float(x) if x not in ("", None) else None


def summarize_cells(items: dict) -> dict:
    """items：item_id → {cell: row}（同一個模型、同一組任務）。每格：答對率、對 FULL／E0_SDPA 的配對差、位元組、覆蓋、補載。"""
    out = {}
    cells = [c for c, *_ in cell_list()]
    full = {i: float(d["FULL"]["score"]) for i, d in items.items() if "FULL" in d}
    e0 = {i: float(d["E0_SDPA"]["score"]) for i, d in items.items() if "E0_SDPA" in d}
    for c in cells:
        rs = {i: d[c] for i, d in items.items() if c in d}
        if not rs:
            continue
        r0 = next(iter(rs.values()))
        sc = {i: float(r["score"]) for i, r in rs.items()}
        mean = lambda col: (statistics.mean(v for v in (_f(r[col]) for r in rs.values()) if v is not None)  # noqa: E731
                            if any(_f(r[col]) is not None for r in rs.values()) else None)
        out[c] = dict(method=r0["method"], k_pct=int(r0["k_pct"]), variant=r0["variant"], kind=r0["kind"], n=len(rs),
                      acc=statistics.mean(sc.values()),
                      diff_vs_full=boot(sc[i] - full[i] for i in sc if i in full),
                      diff_vs_e0_sdpa=boot(sc[i] - e0[i] for i in sc if i in e0),
                      ratio_moved=mean("ratio_moved"), ratio_moved_nosummary=mean("ratio_moved_nosummary"),
                      ratio_init=mean("ratio_init"), needle_cov_init=mean("needle_cov_init"),
                      needle_cov_final=mean("needle_cov_final"), needle_all_init=mean("needle_all_init"),
                      needle_all_final=mean("needle_all_final"), n_refill_blocks=mean("n_refill_blocks"),
                      n_refill_events=mean("n_refill_events"), same_as_full=mean("same_as_full"),
                      oracle_selection=bool(int(r0["oracle_selection"])))
    bma = {}
    for M in METHODS_B:
        for k in KS_PCT:
            ca, cb = f"{M}@{k}a", f"{M}@{k}b"
            pr = [float(d[cb]["score"]) - float(d[ca]["score"]) for d in items.values() if ca in d and cb in d]
            bma[f"{M}@{k}"] = boot(pr)
    return dict(cells=out, b_minus_a=bma, n_items=len(items), acc_full=statistics.mean(full.values()) if full else None,
                acc_e0_sdpa=statistics.mean(e0.values()) if e0 else None,
                full_minus_e0_sdpa=boot(full[i] - e0[i] for i in full if i in e0))


def ok_eps(d) -> bool:
    return d is not None and d["lo"] >= -EPS


def drop_gt_eps(d) -> bool:
    return d is not None and d["mean"] < -EPS


def judge(s: dict) -> dict:
    """事先寫死的判定（c1b_howto.md §3）。s＝summarize_cells 的輸出（T2–T4、一個模型）。
    每個 query-aware 選法 M（QUEST_MEAN＝設計卡指定；QUEST_BLIND＝可部署版）各判一次：
      A＝(a) k＝25%：對完整還原的配對差 CI 下界 ≥ −ε，而且搬運比（含摘要）平均 ≤ 30%
      B＝(b) k＝25%：同上（搬運含補載）
      R＝k＝25% 的 (b)−(a) 配對差 CI 下界 > 0（(b) 明顯好過 (a)）
      STOP＝k＝50% 時 QA_SET 的兩種選法 × (a)、(b) 四格，配對差的平均全部 < −ε（設計卡：k＝50% 都掉超過 ε）
    類別（依序）：題數 < MIN_ITEMS → INSUFFICIENT_N；STOP → STOP；B 且 R → SUCCESS（設計卡的成功）；
      A → SUCCESS_NO_REFILL_NEEDED（不補載就過；「(b) 明顯好過 (a)」不成立或不需要）；B → PARTIAL（補載後過、但 (b)−(a) 不顯著）；
      其他 → INCONCLUSIVE（停下來問人）。"""
    C, bma = s["cells"], s["b_minus_a"]
    stop_cells = [C.get(f"{M}@{K_STOP}{v}") for M in QA_SET for v in ("a", "b")]
    stop = None if any(c is None for c in stop_cells) else all(drop_gt_eps(c["diff_vs_full"]) for c in stop_cells)
    out = {"n_items": s["n_items"], "min_items": MIN_ITEMS, "eps": EPS, "bytes_max": BYTES_MAX, "k_main": K_MAIN,
           "k_stop": K_STOP, "STOP": stop,
           "stop_cells": {f"{M}@{K_STOP}{v}": (C.get(f"{M}@{K_STOP}{v}") or {}).get("diff_vs_full") for M in QA_SET
                          for v in ("a", "b")}, "by_method": {}}
    for M in QA_SET:
        ca, cb = C.get(f"{M}@{K_MAIN}a"), C.get(f"{M}@{K_MAIN}b")
        A = bool(ca and ok_eps(ca["diff_vs_full"]) and ca["ratio_moved"] is not None and ca["ratio_moved"] <= BYTES_MAX)
        B = bool(cb and ok_eps(cb["diff_vs_full"]) and cb["ratio_moved"] is not None and cb["ratio_moved"] <= BYTES_MAX)
        d = bma.get(f"{M}@{K_MAIN}")
        R = bool(d is not None and d["lo"] > 0)
        if s["n_items"] < MIN_ITEMS:
            v = "INSUFFICIENT_N"
        elif stop:
            v = "STOP"
        elif B and R:
            v = "SUCCESS"
        elif A:
            v = "SUCCESS_NO_REFILL_NEEDED"
        elif B:
            v = "PARTIAL"
        else:
            v = "INCONCLUSIVE"
        out["by_method"][M] = dict(verdict=v, A=A, B=B, R=R,
                                   a=dict(diff=ca and ca["diff_vs_full"], ratio_moved=ca and ca["ratio_moved"]),
                                   b=dict(diff=cb and cb["diff_vs_full"], ratio_moved=cb and cb["ratio_moved"]),
                                   b_minus_a=d)
    out["verdict_design_card"] = out["by_method"]["QUEST_MEAN"]["verdict"]
    out["verdict_deployable"] = out["by_method"]["QUEST_BLIND"]["verdict"]
    return out


VERDICT_TEXT = {
    "SUCCESS": "設計卡的成功條件成立：k＝25% 補載後答對率在 ε 內、搬運 ≤ 30%，而且 (b) 明顯好過 (a)。",
    "SUCCESS_NO_REFILL_NEEDED": "k＝25% 不補載就在 ε 內、搬運 ≤ 30%；設計卡的「(b) 明顯好過 (a)」不成立（或補載不需要）→ 「不載入≠丟掉」這半句沒有證據。",
    "PARTIAL": "補載後 k＝25% 在 ε 內、搬運 ≤ 30%，但 (b)−(a) 不顯著 → 停下來問人。",
    "INCONCLUSIVE": "k＝25% 不在 ε 內（或搬運 > 30%），但 k＝50% 也沒有到停止條件 → 停下來問人。",
    "STOP": "k＝50% 時兩種 query-aware 選法、補不補載，答對率都掉超過 ε → query-aware 選擇在這些任務不可靠（設計卡的停止條件）。",
    "INSUFFICIENT_N": f"T2–T4 齊全的題 < {MIN_ITEMS} → 不下判定。",
}


# ---------------------------------------------------------------------------- 還原時間（常數＋公式；純 Python）
def load_consts(model: str, bw_path: str | None = None, tier_path: str = TIER_PARAMS, i4_path: str = I4_CSV) -> dict:
    """L5c §5.2 的常數（每個都記檔名與 run_id）。I4：SDMA 開（ROCm 預設）、GPU 閒、scatter 版面的每片 µs。"""
    bw = json.load(open(bw_path or BW_JSON.format(m=model)))
    tp = json.load(open(tier_path))
    i4 = {}
    i4_run = set()
    for r in csv.DictReader(open(i4_path)):
        if r["layout"] == "scatter" and r["gpu_state"] == "idle" and r["status"] == "ok":
            i4[r["method"]] = float(r["us_per_frag"])
            i4_run.add(r["run_id"])
    tiers = {"cpu_h2d": dict(GiBps=bw["real_pinned_h2d_GiBps"], src=f"bw.json real_pinned_h2d（{bw['run_id']}）"),
             "b32": dict(GiBps=bw["targets"]["b32"]["GiBps"], src=f"bw.json targets.b32（{bw['run_id']}）"),
             "ssd_param": dict(GiBps=tp["local"]["read_GiBps"], src=f"tier_params local（{';'.join(tp['source_runs'])}）"),
             "nfs_param": dict(GiBps=tp["nfs"]["read_GiBps"], src=f"tier_params nfs（{';'.join(tp['source_runs'])}）")}
    return dict(model=model, chunk=bw["chunk_bytes"], f_ms=bw["f_ms_median_by_chunk"], bw_run=bw["run_id"], tiers=tiers,
                i4_us_per_frag=i4, i4_run=sorted(i4_run), files=[bw_path or BW_JSON.format(m=model), tier_path, i4_path])


def load_time_s(n_blocks: float, chunk: int, GiBps: float) -> float:
    return n_blocks * chunk / (GiBps * 2**30)


def cake_time_s(f_ms: list, chunk: int, GiBps: float) -> tuple:
    """第二階段的離散會合：min_b max(Σ_{i<b} f_i, (64−b)·chunk／B)。回傳 (時間, b*)。"""
    best = None
    for b in range(N_BLOCKS + 1):
        t = max(sum(f_ms[:b]) / 1e3, load_time_s(N_BLOCKS - b, chunk, GiBps))
        if best is None or t < best[0]:
            best = (t, b)
    return best


def time_table(model: str, C: dict, qt: dict) -> list:
    """每個（層級／搬法, k）：只搬 k% 的時間 vs 全部搬、vs Cake。Quest 的額外時間（qtime 實測）另列。
    harness 版面＝每個 512-token block 在 host 是連續的一塊（第二階段的 Restorer）；vLLM 版面＝16-token 的頁、每層 K／V 各一片
    （I4 量的是 Llama 的 32 KiB 片；Qwen 的片是 16 KiB，沿用同一個每片 µs＝推算）。"""
    sb = shape_bytes(model)
    assert sb["chunk"] == C["chunk"], (sb, C["chunk"])
    q_over = None
    if qt.get("blind_q_forward") is not None and qt.get("quest_select_q2") is not None:
        q_over = qt["blind_q_forward"] + qt["quest_select_q2"]
    rows = []
    for tier, T in C["tiers"].items():
        full = load_time_s(N_BLOCKS, C["chunk"], T["GiBps"])
        cake, bstar = cake_time_s(C["f_ms"], C["chunk"], T["GiBps"])
        summ = load_time_s(N_BLOCKS * sb["summary_block"] / C["chunk"], C["chunk"], T["GiBps"])
        for k in KS_PCT:
            n = 1 + m_of(k)
            t = load_time_s(n, C["chunk"], T["GiBps"])
            rows.append(dict(model=model, layout="harness", tier=tier, GiBps=T["GiBps"], src=T["src"], k_pct=k, n_blocks=n,
                             t_sel_s=t, t_summary_s=summ, t_full_load_s=full, t_cake_s=cake, cake_bstar=bstar,
                             ratio_vs_full=t / full, ratio_vs_cake=t / cake, quest_overhead_ms=q_over))
    for meth, us in sorted(C["i4_us_per_frag"].items()):
        full = N_BLOCKS * sb["frags_per_block"] * us * 1e-6
        for k in KS_PCT:
            n = 1 + m_of(k)
            t = n * sb["frags_per_block"] * us * 1e-6
            rows.append(dict(model=model, layout="vllm_pages", tier=f"cpu_i4_{meth}", GiBps=round(sb["frag_bytes"] / (us * 1e-6) / 2**30, 3),
                             src=f"I4 copybench_sdma1 scatter idle（{';'.join(C['i4_run'])}）" + ("；Qwen 片大小不同＝推算" if model == "qwen" else ""),
                             k_pct=k, n_blocks=n, t_sel_s=t, t_summary_s=None, t_full_load_s=full, t_cake_s=None, cake_bstar=None,
                             ratio_vs_full=t / full, ratio_vs_cake=None, quest_overhead_ms=q_over))
    return rows


def load_qtime(paths) -> dict:
    """qtime CSV → {model: {op: t_med_ms}}；只收 gpu_verdict＝CLEAN 的列（其他＝NOT_MEASURED）。"""
    out, meta = {}, {}
    for p in paths:
        for r in csv.DictReader(open(p)):
            if r["gpu_verdict"] != "CLEAN":
                continue
            out.setdefault(r["model"], {})[r["op"]] = float(r["t_med_ms"])
            meta.setdefault(r["model"], set()).add(r["run_id"])
    return out, {m: sorted(v) for m, v in meta.items()}


def sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


GROUPS = {"T2-T4": TASKS_2T, "T2": ("T2",), "T3": ("T3",), "T4": ("T4",), "T5": ("T5",)}


def pp(d, nd: int = 1) -> str:
    return fmt_ci(d, 100, nd)


def build_report(paths: list, qtime_paths: list, checks_paths: list) -> tuple:
    """回傳 (rep JSON, markdown 行, cells CSV 列)。"""
    items, partial = load_items(paths)
    rows = [r for d in items.values() for r in d.values()]
    qt, qt_runs = load_qtime(qtime_paths)
    bad_checks = {}
    for p in checks_paths:
        for line in open(p):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            c = d.get("checks", {})
            f = [k for k, v in c.items() if v is False] + [f"order_eq_c1a:{k}" for k, v in (c.get("order_eq_c1a") or {}).items()
                                                          if v is False]
            if f:
                bad_checks[f"{d['model']}/{d['item_id']}"] = f
    run_ids = sorted({r["run_id"] for r in rows})
    rep = {"rule": {"eps": EPS, "bytes_max": BYTES_MAX, "k_main": K_MAIN, "k_stop": K_STOP, "min_items": MIN_ITEMS,
                    "text": VERDICT_TEXT, "judge_doc": judge.__doc__},
           "files": [dict(path=p, bytes=os.path.getsize(p), sha256=sha256(p)) for p in paths], "run_ids": run_ids,
           "incomplete_items": partial, "checks_false": bad_checks, "qtime_run_ids": qt_runs,
           "contention": {"rows_with_intruders": sum(1 for r in rows if r["own_gpu_intruders"] not in ("0", "NOT_CHECKED")),
                          "rows_not_checked": sum(1 for r in rows if r["own_gpu_intruders"] == "NOT_CHECKED"),
                          "host_levels": sorted({r["host_level"] for r in rows}),
                          "note": "品質實驗：分數照用；t_gen_s 在有入侵者的 run 作廢（同一張卡上自己的其他 shard 也算入侵者）"},
           "order_eq_c1a": {"rows_false": sum(1 for r in rows if r["order_eq_c1a"] == "0"),
                            "rows_missing": sum(1 for r in rows if r["order_eq_c1a"] == "" and r["method"] in c1a_key("llama"))},
           "models": {}}
    L = ["# C1b：選擇性還原（只搬需要的 block；不載入 vs 丟掉）", "",
         f"run：{len(run_ids)} 個（見 c1b_summary.json）；不齊全的題（沒用）：{len(partial)}；逐題檢查有 false 的題：{len(bad_checks)}",
         f"配對差＝格子 − 完整還原（FULL，同一條遮罩路徑），百分點，95% CI＝以題目為單位 bootstrap 10,000 次。ε＝{EPS * 100:.0f} 個百分點；"
         f"搬運比＝(初始 block＋tail＋摘要＋補載)／(64 個 block＋tail)。", ""]
    csv_rows = []
    now = ts()
    for m in sorted({mm for mm, _ in items}):
        rep["models"][m] = {}
        for g, tasks in GROUPS.items():
            its = {i: d for (mm, i), d in items.items() if mm == m and next(iter(d.values()))["task"] in tasks}
            if not its:
                continue
            s = summarize_cells(its)
            e = rep["models"][m][g] = dict(s)
            if g == "T2-T4":
                e["judge"] = judge(s)
                bands = {}
                for i, d in its.items():
                    bands.setdefault(next(iter(d.values()))["band2"], {})[i] = d
                e["by_band2"] = {b: {c: summarize_cells(bd)["cells"].get(c) for c in
                                     ("QUEST_MEAN@25a", "QUEST_MEAN@25b", "QUEST_BLIND@25a", "QUEST_BLIND@25b", "POS@25a", "RND@25a")}
                                 for b, bd in sorted(bands.items())}
            for c, x in s["cells"].items():
                csv_rows.append(dict(run_id=";".join(sorted({d[c]["run_id"] for d in its.values() if c in d})), ts=now,
                                     model=m, group=g, cell=c, method=x["method"], k_pct=x["k_pct"], variant=x["variant"],
                                     kind=x["kind"], n=x["n"], acc=round(x["acc"], 6),
                                     diff_mean=x["diff_vs_full"] and round(x["diff_vs_full"]["mean"], 6),
                                     diff_lo=x["diff_vs_full"] and round(x["diff_vs_full"]["lo"], 6),
                                     diff_hi=x["diff_vs_full"] and round(x["diff_vs_full"]["hi"], 6),
                                     ratio_moved=x["ratio_moved"], needle_cov_init=x["needle_cov_init"],
                                     needle_cov_final=x["needle_cov_final"], n_refill_blocks=x["n_refill_blocks"],
                                     oracle_selection=int(x["oracle_selection"])))
            L += [f"## {m}・{g}（{s['n_items']} 題）", "",
                  f"完整還原 FULL 答對率 {100 * s['acc_full']:.1f}%；E0_SDPA（第二階段的路徑）{100 * s['acc_e0_sdpa']:.1f}%；"
                  f"FULL − E0_SDPA（換路徑的雜訊）{pp(s['full_minus_e0_sdpa'])}", "",
                  "| 格子 | 答對率 | 對 FULL 的差 | 搬運比（含摘要） | 答案 block 全蓋到（初始→最終） | 補載 block | 補載次數 | 和 FULL 同字 |",
                  "|:--|--:|--:|--:|--:|--:|--:|--:|"]
            for c, x in s["cells"].items():
                rm = "oracle" if x["oracle_selection"] else (f"{x['ratio_moved']:.3f}" if x["ratio_moved"] is not None else "—")
                cov = (f"{x['needle_all_init']:.2f}→{x['needle_all_final']:.2f}" if x["needle_all_init"] is not None else "—")
                L.append(f"| {c} | {100 * x['acc']:.1f}% | {pp(x['diff_vs_full'])} | {rm} | {cov} | "
                         f"{x['n_refill_blocks']:.2f} | {x['n_refill_events']:.2f} | {x['same_as_full']:.2f} |")
            L += ["", "(b) − (a)（同一題配對，百分點）：" + "；".join(f"{k} {pp(v)}" for k, v in s["b_minus_a"].items()), ""]
            if g == "T2-T4":
                j = e["judge"]
                L += [f"**判定（事先寫死）**：設計卡（QUEST_MEAN）＝**{j['verdict_design_card']}**；可部署版（QUEST_BLIND）＝"
                      f"**{j['verdict_deployable']}**；STOP 成分＝{j['STOP']}", "",
                      f"- {j['verdict_design_card']}：{VERDICT_TEXT[j['verdict_design_card']]}",
                      f"- {j['verdict_deployable']}（可部署版）：{VERDICT_TEXT[j['verdict_deployable']]}", "",
                      "```", json.dumps(j["by_method"], ensure_ascii=False, indent=1, default=str), "```", ""]
        C = load_consts(m)
        tt = time_table(m, C, qt.get(m, {}))
        rep["models"][m]["restore_time"] = dict(constants={k: v for k, v in C.items() if k != "f_ms"}, f_sum_s=sum(C["f_ms"]) / 1e3,
                                                quest_ms=qt.get(m, "NOT_MEASURED"), rows=tt)
        qo = tt[0]["quest_overhead_ms"]
        rc = qt.get(m, {}).get("quest_refill_check")
        L += [f"### {m}：還原時間〔計算：第二階段常數＋I4 每片成本；Quest 額外時間＝qtime 實測〕", "",
              f"Quest 額外時間（可部署版：只看 block 0＋tail 的 q forward＋選 block）：{'NOT_MEASURED' if qo is None else f'{qo:.2f} ms'}；"
              f"補載每次檢查：{'NOT_MEASURED' if rc is None else f'{rc:.3f} ms'}（qtime 中位數；GPU 不乾淨的不收）", "",
              "| 版面／層級 | GiB/s | k | 載入 block | 只搬 k%（s） | 全部搬（s） | Cake（s） | 對全部搬 | 對 Cake |", "|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for r in tt:
            cake = "—" if r["t_cake_s"] is None else f"{r['t_cake_s']:.4f}（b*={r['cake_bstar']}）"
            rvc = "—" if r["ratio_vs_cake"] is None else f"{r['ratio_vs_cake']:.3f}"
            L.append(f"| {r['layout']}／{r['tier']} | {r['GiBps']:.3f} | {r['k_pct']}% | {r['n_blocks']} | {r['t_sel_s']:.4f} | "
                     f"{r['t_full_load_s']:.4f} | {cake} | {r['ratio_vs_full']:.3f} | {rvc} |")
        L += ["", "註：選擇性還原只能「載入」（被選中的 block 不連續，不能用前段重算）；Cake 會重算前段，所以在慢的層級"
              "『只搬 k%』可能比 Cake 慢。補載的 block 在 decode 期間才搬，不在首 token 前；第 2 輪問題的 prefill 不含在內"
              "（每格相同；本 harness 的遮罩路徑仍對全部 key 計算，量不出少算的好處）。", ""]
    return rep, L, csv_rows


CELL_CSV_FIELDS = ["run_id", "ts", "model", "group", "cell", "method", "k_pct", "variant", "kind", "n", "acc", "diff_mean",
                   "diff_lo", "diff_hi", "ratio_moved", "needle_cov_init", "needle_cov_final", "n_refill_blocks",
                   "oracle_selection"]


def cmd_report(a):
    paths = sorted({p for g in a.raw for p in glob.glob(g)})
    qps = sorted({p for g in (a.qtime or []) for p in glob.glob(g)})
    cps = sorted({side_path(p, ".checks.jsonl") for p in paths if os.path.exists(side_path(p, ".checks.jsonl"))})
    rep, L, crow = build_report(paths, qps, cps)
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(rep, open(os.path.join(a.out_dir, "c1b_summary.json"), "w"), indent=1, ensure_ascii=False, default=str)
    open(os.path.join(a.out_dir, "tables_c1b.md"), "w").write("\n".join(L) + "\n")
    with open(os.path.join(a.out_dir, "c1b_cells.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CELL_CSV_FIELDS)
        w.writeheader()
        w.writerows(crow)
    print("\n".join(L[:120]))


# ============================================================================ --self-test（CPU、不 import torch）
def _synth_rows(model: str, n_items: int, scen: str, seed: int = 0) -> list:
    """合成的逐列資料（給 self-test）。scen：success＝k25 (b) 和 FULL 一樣、(a) 掉 30 個百分點；stop＝k50 全掉 50 個百分點。"""
    rng = random.Random(f"{seed}|{model}|{scen}")
    plan = [x for x in plan_c1b() if x[0] != "T5"][:n_items]
    out = []
    for task, b1, b2, sd in plan:
        iid = item_id_of(task, b1, b2, sd)
        full = 1.0 if rng.random() < 0.8 else 0.0
        for cell, M, k, v in cell_list():
            if M in ("E0_SDPA", "FULL"):
                s = full
            elif scen == "success":
                s = full if (v == "b" or k == 50) else (full if rng.random() < 0.6 else 0.0)
            else:
                s = full if k == 10 and rng.random() < 0.1 else (full if k != 50 and rng.random() < 0.5 else 0.0)
            ld = N_BLOCKS if M in ("E0_SDPA", "FULL") else 1 + m_of(k)
            by = bytes_row(model, ld, 1 if v == "b" else 0, 40, M in QUEST_USERS or v == "b")
            out.append(dict(run_id="SYN", ts="2026-10-21T00:00:00+00:00", shard="0/1", model=model, item_id=iid, task=task,
                            seed=sd, band1=b1, band2=b2, turn_eval=2, metric="m", cell=cell, method=M, kind=kind_of(M, task),
                            k_pct=k, m_blocks=m_of(k), variant=v, refill_every=REFILL_EVERY if v == "b" else "",
                            write_method="", n_init_blocks=ld, n_refill_blocks=1 if v == "b" else 0,
                            n_refill_events=1 if v == "b" else 0, n_refill_checks=3 if v == "b" else 0, refill_log="[]",
                            loaded_final="[]", needle_tgt="[5]", needle_cov_init=1, needle_cov_final=1, needle_all_init=1,
                            needle_all_final=1, tail_tokens=40, **by, oracle_selection=int(M == "FULLK_MEAN"), gen_len=64,
                            n_gen=3, stop="eos", pred="x", score=s, f1="", same_as_full=int(s == full), score_t1_full=1.0,
                            order_eq_c1a=1, t_gen_s=1.0, n_rows_item=N_CELLS, own_gpu_intruders=2, host_level="QUIET"))
    return out


def _pool_worker(args):
    pool, units, n, wid, skip_set = args
    got = []
    while True:
        u = pool_claim(pool, units, n, wid, (lambda m, i: (m, i) in skip_set))
        if u is None:
            return got
        got.append(u)
        time.sleep(random.random() * 0.002)


def self_test() -> int:
    import multiprocessing
    import tempfile
    res, ok = [], True

    def t(name, cond, **kw):
        nonlocal ok
        ok &= bool(cond)
        res.append(name)
        print(("PASS " if cond else "FAIL ") + name + (f" {kw}" if kw else ""), flush=True)

    # 1. 出題與切片
    plan = plan_c1b()
    ids = [item_id_of(*x) for x in plan]
    t("出題：150 題、沒有重複", len(ids) == 150 and len(set(ids)) == 150)
    for mdl in MODELS:
        c1a_ids = {r["item_id"] for r in read_rows(sorted(glob.glob(os.path.join(C1A_DIR, f"c1a_{mdl}_s*.csv"))))}
        t(f"出題：和 C1a（{mdl}）的題目集合相同", set(ids) == c1a_ids, n_c1a=len(c1a_ids))
    t("出題：前 3 題涵蓋 T2、T3、T4；T5 在最後", {x[0] for x in plan[:3]} == set(TASKS_2T) and all(x[0] == "T5" for x in plan[135:]))
    t("出題：前 9 題涵蓋三種位置帶組合、每個任務各 3 題",
      {(x[1], x[2]) for x in plan[:9]} == set(ROTATIONS) and all(sum(x[0] == tk for x in plan[:9]) == 3 for tk in TASKS_2T))
    for n in (3, 12):
        parts = [shard_items(plan, i, n) for i in range(n)]
        flat = [item_id_of(*x) for p in parts for x in p]
        t(f"切片 n={n}：每題恰好在一個 shard", sorted(flat) == sorted(ids))
    t("selfcheck 的兩題在出題裡", all(item_id_of(*x) in ids for x in SELFCHECK_ITEMS))
    # 2. 工作池
    U = pool_units(MODELS, 12)
    t("工作池：24 個工作、兩個模型交錯", len(U) == 24 and [u[0] for u in U[:4]] == ["llama", "qwen", "llama", "qwen"]
      and len(set(U)) == 24)
    skip_set = {("llama", 2), ("qwen", 7)}
    with tempfile.TemporaryDirectory() as td:
        pool = os.path.join(td, "pool")
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(6) as pp_:
            got = pp_.map(_pool_worker, [(pool, U, 12, f"c{c}s{s}", skip_set) for c in (0, 1) for s in (1, 2, 3)])
        allg = [u for g in got for u in g]
        t("工作池（6 個 worker 同時搶）：每個 (模型, shard) 恰好被領一次、已完成的不被領",
          sorted(allg) == sorted(set(U) - skip_set) and len(allg) == len(set(allg)),
          n_claimed=len(allg), per_worker=[len(g) for g in got])
        t("工作池：每個工作都有 CLAIM 或 SKIPPED 標記", all(
            os.path.exists(os.path.join(unit_dir(pool, m, i, 12), "SKIPPED" if (m, i) in skip_set else "CLAIM")) for m, i in U))
        t("工作池：領完之後再領 → None", pool_claim(pool, U, 12, "late") is None)
        # unit_remaining／done_union：寫一個假的 CSV（一題齊全、一題不齊）
        rd = os.path.join(td, "raw")
        os.makedirs(rd)
        sh = shard_items(plan, 0, 12)
        fo, w = open_out(raw_path(rd, "llama", 0), FIELDS)
        for j, x in enumerate(sh[:2]):
            for c in range(N_CELLS if j == 0 else 3):
                w.writerow({f: "" for f in FIELDS} | dict(run_id="R", item_id=item_id_of(*x), n_rows_item=N_CELLS, cell=str(c)))
        fo.close()
        rem = unit_remaining(rd, "llama", 0, 12, plan)
        t("續跑：齊全的題算寫完、不齊的題要重做", item_id_of(*sh[0]) not in rem and item_id_of(*sh[1]) in rem
          and len(rem) == len(sh) - 1)
    # 3. k、位元組
    t("k → m：10→6、25→16、50→32", [m_of(k) for k in KS_PCT] == [6, 16, 32])
    t(f"格子數 {N_CELLS}＝2＋7×3＋6×3", N_CELLS == 2 + len(METHODS_A) * 3 + len(METHODS_B) * 3)
    for mdl in MODELS:
        bw = json.load(open(BW_JSON.format(m=mdl)))
        sb = shape_bytes(mdl)
        t(f"位元組（{mdl}）：block 位元組＝bw.json 的 chunk_bytes", sb["chunk"] == bw["chunk_bytes"], chunk=sb["chunk"])
        b = bytes_row(mdl, 17, 0, 100, True)
        exp = (17 * sb["chunk"] + 100 * sb["per_token"] + 64 * sb["summary_block"]) / (64 * sb["chunk"] + 100 * sb["per_token"])
        t(f"位元組（{mdl}）：k＝25% Quest (a) 的搬運比＝{exp:.4f}（≤ 0.30）", abs(b["ratio_moved"] - exp) < 1e-6 and exp <= BYTES_MAX)
        t(f"位元組（{mdl}）：完整還原＝1；摘要＝64×2×L×H×D×2", bytes_row(mdl, 64, 0, 100, False)["ratio_moved"] == 1.0
          and sb["summary_block"] == 2 * SHAPE[mdl]["L"] * SHAPE[mdl]["H"] * SHAPE[mdl]["D"] * 2)
        t(f"位元組（{mdl}）：補載 3 個 block＋3×block 位元組", bytes_row(mdl, 7, 3, 0, True)["bytes_refill"] == 3 * sb["chunk"])
    # 4. 補載規則
    sc = [0.0] * N_BLOCKS
    for b, v in ((0, 9.0), (5, 5.0), (9, 4.0), (12, 3.0), (20, 2.0), (33, 1.0)):
        sc[b] = v
    t("補載規則：前 m 名裡沒載入的（block 0 不排名）", refill_pick(sc, {0, 5, 12}, 3) == [9] and refill_pick(sc, {0}, 2) == [5, 9]
      and refill_pick(sc, {0, 5, 9, 12}, 3) == [])
    t("補載規則：同分時照 keep_order_from_blocks（block 編號大的先）", refill_pick([0.0] * N_BLOCKS, {0}, 2) == [62, 63])
    t("覆蓋：答案 block 全蓋到／部分", coverage([3, 4], {0, 3, 4}) == (1.0, 1) and coverage([3, 4], {0, 3}) == (0.5, 0)
      and coverage([], {0}) == ("", ""))
    # 5. 判定
    for mdl, scen in (("llama", "success"), ("qwen", "stop")):
        its = {}
        for r in _synth_rows(mdl, 60, scen):
            its.setdefault(r["item_id"], {})[r["cell"]] = r
        j = judge(summarize_cells(its))
        want = "SUCCESS" if scen == "success" else "STOP"
        t(f"判定（合成 {scen}）→ {want}", j["verdict_design_card"] == want and j["verdict_deployable"] == want,
          got=(j["verdict_design_card"], j["verdict_deployable"]))
    base = {"n_items": 60, "b_minus_a": {}, "cells": {}}

    def cellx(mean, lo, ratio=0.27):
        return dict(diff_vs_full=dict(mean=mean, lo=lo, hi=mean + 0.01, n=60), ratio_moved=ratio)

    def jj(a25, b25, bma, a50=(-0.01, -0.03), b50=(-0.01, -0.03), n=60, rb=0.27):
        s = dict(base, n_items=n, cells={}, b_minus_a={})
        for M in QA_SET:
            s["cells"].update({f"{M}@25a": cellx(*a25), f"{M}@25b": cellx(*b25, ratio=rb), f"{M}@50a": cellx(*a50),
                               f"{M}@50b": cellx(*b50)})
            s["b_minus_a"][f"{M}@25"] = dict(mean=bma[0], lo=bma[1], hi=bma[0] + 0.05, n=n)
        return judge(s)["verdict_design_card"]

    t("判定：(a) 已在 ε 內、(b)−(a) 不顯著 → SUCCESS_NO_REFILL_NEEDED",
      jj((-0.005, -0.015), (0.0, -0.01), (0.005, -0.01)) == "SUCCESS_NO_REFILL_NEEDED")
    t("判定：(b) 在 ε 內但搬運 > 30% 且 (a) 不在 → INCONCLUSIVE",
      jj((-0.10, -0.15), (0.0, -0.01), (0.10, 0.05), rb=0.35) == "INCONCLUSIVE")
    t("判定：(b) 在 ε 內、(b)−(a) 不顯著、(a) 不在 → PARTIAL", jj((-0.03, -0.06), (-0.005, -0.015), (0.02, -0.01)) == "PARTIAL")
    t("判定：k50 四格平均都 < −ε → STOP（優先）", jj((-0.2, -0.3), (-0.1, -0.2), (0.1, 0.05), (-0.05, -0.1), (-0.03, -0.08)) == "STOP")
    t("判定：k50 有一格沒掉超過 ε → 不 STOP", jj((-0.2, -0.3), (-0.1, -0.2), (0.1, 0.05), (-0.05, -0.1), (-0.01, -0.08))
      == "INCONCLUSIVE")
    t("判定：題數 < 48 → INSUFFICIENT_N", jj((0, 0), (0, 0), (0, 0), n=47) == "INSUFFICIENT_N")
    # 6. 時間模型
    f = [10.0] * 64
    tc, bs = cake_time_s(f, 2**30, 1.0)
    t("Cake 公式：f＝10 ms、每塊 1 s → 全部重算 0.64 s（b*＝64）", abs(tc - 0.64) < 1e-9 and bs == 64)
    tc2, bs2 = cake_time_s([1000.0] * 64, 2**30, 1.0)
    t("Cake 公式：重算比載入慢 → 會合點在中間（b*＝32、32 s）", abs(tc2 - 32.0) < 1e-9 and bs2 == 32)
    for mdl in MODELS:
        C = load_consts(mdl)
        tt = time_table(mdl, C, {})
        h = [r for r in tt if r["tier"] == "cpu_h2d" and r["k_pct"] == 25][0]
        exp_full = 64 * C["chunk"] / (C["tiers"]["cpu_h2d"]["GiBps"] * 2**30)
        t(f"時間（{mdl}）：cpu 全部搬＝64×block／{C['tiers']['cpu_h2d']['GiBps']} GiB/s＝{exp_full * 1e3:.1f} ms；k25 搬 17 塊",
          abs(h["t_full_load_s"] - exp_full) < 1e-12 and h["n_blocks"] == 17 and abs(h["ratio_vs_full"] - 17 / 64) < 1e-12)
        t(f"時間（{mdl}）：I4 的五種搬法都讀到", sorted(C["i4_us_per_frag"]) == ["coalesce", "foreach", "py_loop", "stage", "zerocopy"],
          us=C["i4_us_per_frag"], run=C["i4_run"])
    # 7. report 端到端（合成資料、暫存目錄）＋ load_items 取最後一個齊全的 run
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "c1b_llama_s0.csv.gz")
        fo, w = open_out(p, FIELDS)
        rows = _synth_rows("llama", 50, "success") + _synth_rows("qwen", 50, "stop")
        first = rows[0]["item_id"]
        for r in rows:
            w.writerow(r)
        for r in rows[:5]:                                   # 同一題另一次嘗試（不齊）：不能混進來
            w.writerow(dict(r, run_id="SYN2", ts="2026-10-21T01:00:00+00:00", score=0.0))
        fo.close()
        items, partial = load_items([p])
        t("load_items：不齊全的嘗試不混用", items[("llama", first)]["FULL"]["run_id"] == "SYN" and not partial)
        qp = os.path.join(td, "qtime_llama.csv")
        with open(qp, "w", newline="") as f_:
            w2 = csv.DictWriter(f_, fieldnames=QTIME_FIELDS)
            w2.writeheader()
            for op, v in (("blind_q_forward", 12.0), ("quest_select_q2", 1.5), ("quest_refill_check", 0.4)):
                w2.writerow({k: "" for k in QTIME_FIELDS} | dict(model="llama", op=op, t_med_ms=v, gpu_verdict="CLEAN", run_id="Q"))
        rep, L, crow = build_report([p], [qp], [])
        jl, jq = rep["models"]["llama"]["T2-T4"]["judge"], rep["models"]["qwen"]["T2-T4"]["judge"]
        t("report 端到端：合成 success／stop → SUCCESS／STOP；Quest 額外時間＝13.5 ms",
          jl["verdict_design_card"] == "SUCCESS" and jq["verdict_design_card"] == "STOP"
          and rep["models"]["llama"]["restore_time"]["rows"][0]["quest_overhead_ms"] == 13.5
          and rep["models"]["qwen"]["restore_time"]["rows"][0]["quest_overhead_ms"] is None)
        t("report：cells CSV 每列有 run_id（這一格用到的 run，分號分隔）與 ts", all(r["run_id"] and r["ts"] for r in crow) and len(crow) > 0)
    # 8. 欄位
    S = dict(needle=[5, 6], ti=1, tre=dict(metric="string_match_all"), tail=37, gl=64, score_t1=1.0,
             order_eq={k: True for k in c1a_key("llama")})
    it = dict(item_id="T2_front-back_s70000", task="T2", seed=70000, turns=[dict(band="front"), dict(band="back")])
    g = dict(loaded0={0, 5}, loaded_final={0, 5, 6}, m=6, n_refill=1, events=[[4, [6]]], n_checks=3, out=[1, 2], stop="eos",
             t_gen_s=0.1)
    r = make_row("llama", it, S, "QUEST_BLIND@10b", "QUEST_BLIND", 10, "b", g, "x", 1.0, "", REFILL_EVERY)
    extra = {"run_id", "ts", "shard", "n_rows_item", "own_gpu_intruders", "host_level"}
    t("make_row：欄位＝FIELDS（呼叫端補的 6 欄除外）", set(r) | extra == set(FIELDS) and not (set(r) & extra))
    t("make_row：覆蓋 0.5→1、補載 1 個 block", r["needle_cov_init"] == 0.5 and r["needle_cov_final"] == 1.0
      and r["needle_all_final"] == 1 and r["bytes_refill"] == shape_bytes("llama")["chunk"])
    t("沒有 import torch", "torch" not in sys.modules)
    print(("SELF-TEST ALL PASS" if ok else "SELF-TEST FAILED") + f"（{len(res)} 項）", flush=True)
    return 0 if ok else 1


# ============================================================================ CLI
def main():
    if "--self-test" in sys.argv[1:]:
        sys.exit(self_test())
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "selfcheck", "qtime"):
        p = sp.add_parser(name)
        p.add_argument("--model", choices=list(MODELS), required=True)
        p.add_argument("--params-file", default=PARAMS)
        p.add_argument("--c1a-dir", default=C1A_DIR, help="C1a 的逐列 CSV（保留順序逐字比對）")
        p.add_argument("--quest-skip-layers", type=int, default=2, help="Quest 彙總略過前幾層（C1a：2）")
        p.add_argument("--refill-every", type=int, default=REFILL_EVERY, help="(b) 每幾個生成 token 檢查一次")
        if name == "run":
            p.add_argument("--tasks", nargs="+", default=list(TASKS_2T) + ["T5"])
            p.add_argument("--n", type=int, default=N_PER_TASK)
            p.add_argument("--n-t5", type=int, default=N_T5)
            p.add_argument("--seed0", type=int, default=SEED0)
            p.add_argument("--shard", default="0/1")
            p.add_argument("--budget-min", type=float, default=0, help="超過就不再開新題（逐題續跑；0＝不限）")
            p.add_argument("--out", required=True, help="逐列 CSV（.csv.gz；放 RAW_DIR，不進 git）")
        elif name == "selfcheck":
            p.add_argument("--out", help="預設 $TIARA_RUNS/$RUN_ID/c1b_selfcheck_<model>.json")
        else:
            p.add_argument("--reps", type=int, default=20)
            p.add_argument("--out", required=True, help="乾淨時才寫（results/…/qtime_<model>.csv）；污染時只留在 run 目錄")
    r = sp.add_parser("report")
    r.add_argument("--raw", nargs="+", required=True, help="逐列 CSV 的萬用字元")
    r.add_argument("--qtime", nargs="*", help="qtime CSV 的萬用字元")
    r.add_argument("--out-dir", default=OUT_DIR)
    for name in ("pool-claim", "pool-status", "pool-done"):
        p = sp.add_parser(name)
        p.add_argument("--n", type=int, required=True, help="每個模型切幾份（兩個模型共 2n 個工作）")
        p.add_argument("--models", nargs="+", default=list(MODELS))
        p.add_argument("--raw-dir", default=RAW_DIR)
        p.add_argument("--pool", required=(name != "pool-status"))
        if name == "pool-claim":
            p.add_argument("--worker", required=True)
        if name == "pool-done":
            p.add_argument("--unit", required=True, help="'<模型> <shard>'")
            p.add_argument("--rc", type=int, default=0)
    a = ap.parse_args()
    {"run": cmd_run, "selfcheck": cmd_selfcheck, "qtime": cmd_qtime, "report": cmd_report, "pool-claim": cmd_pool_claim,
     "pool-status": cmd_pool_status, "pool-done": cmd_pool_done}[a.cmd](a)


if __name__ == "__main__":
    main()
