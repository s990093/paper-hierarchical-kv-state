#!/usr/bin/env python3
"""m16_workload.py — 第三階段 F1／I3 的 serving 工作負載產生器（純 Python：不 import torch／vllm）。

為什麼不能 import torch／vllm：AMD 的 gpu_guard 用「本容器內開 /dev/kfd 的行程」判斷入侵者，
產生工作負載的行程只要碰到 GPU runtime，正在跑的計時量測就會被判污染（2026-10-19 第二階段 session 的要求）。
設 M16_FORBID_GPU_IMPORTS=1 時，本檔一載入就裝上 import 阻擋器，任何 torch／vllm 的 import 都會直接失敗。

三種工作負載（輸出 JSON lines；第一行是 {"_meta": {...}}，之後每行一個請求）：

1. prefix_overlap（F1）：n_req 個請求、每個 prompt_len 個 token；請求 i 屬於第 i mod n_groups 組；
   同組共享前 shared_len＝round(overlap_frac×prompt_len) 個 token，其餘每個請求獨有。
   每個請求「獨有部分的第一個 token」強制互不相同（vocab_lo＋i×stride mod span，stride 與 span 互質），
   所以同組兩個請求的共同前綴**恰好**是 shared_len（不會因為亂數碰巧相同而多 1 個）。
   overlap_frac＝1 時同組請求完全相同。
   到達：poisson（第一個在 t＝0，之後間隔 ~ Exp(rate)）、fixed（間隔 1/rate）、burst（全部 t＝0）。

2. mooncake_replay（F1 外部效度／I3 的長度分佈）：Mooncake trace 的 hash_ids 一個＝512 token（CLAUDE.md 規則 6）。
   - 載入時**逐列**斷言 (len(hash_ids)−1)×512 < input_length ≤ len(hash_ids)×512（整個檔案，不只取用的列）。
     2026-10-19 在 toolagent（23,608 列）、conversation（12,031 列）上實測 0 列違反。
   - 同一個 hash_id → 同一段 512 token（由 seed 與 hash_id 決定）；最後一個 block 截到 input_length。
   - 斷言 trace 的前綴鏈性質：同一個 hash_id 永遠在同一個位置、前一個 hash_id 也永遠相同（實測兩個 trace 都 0 例外）。
   - 交叉驗證（第二條獨立路徑）：用「產生出來的 token」逐 512 個算前綴鏈摘要，數每個請求和更早請求共享的完整 block 數；
     和「直接數 hash_ids 前綴重疊」的結果逐請求比對，必須完全相同；另外對前 300 個請求做 hash_ids 的兩兩 LCP 暴力比對。
   - timestamp 是毫秒（D4 §2：99.6% 是 1000 的倍數、範圍 0–3,536,999、和 Mooncake 原文一致）→ t_arrive＝(ts−ts0)/1000×time_scale。
     time_scale＜1＝壓縮時間（到達更密）。

3. agentic_sessions（I3）：n_sess 個 session × n_turns 輪。
   第 0 輪 prompt＝base_ctx 個 token；第 t 輪 prompt＝第 t−1 輪的完整 prompt＋第 t−1 輪輸出的替代 token（out_len 個，
   由 seed／session／turn 決定的固定序列，不是 server 真的生成的 token）＋第 t 輪的工具輸出（長度 ~ tool_out_len_dist）。
   t_ready：第 0 輪＝session 開始時間（相對 run 開始，~ start_dist）；第 t≥1 輪＝相對上一輪「結束」的暫停秒數（~ pause_dist）。
   warm_prefix_len＝len(上一輪 prompt)＋out_len：上一輪結束時就已經知道的前綴（I3 的 warm-up 預取就送這一段）。
   分佈字串：const:x、uniform:a:b、lognormal:mu:sigma（random.lognormvariate）、empirical:<檔案>[:<欄名>]。

4. shared_prefix_burst（F2，重現 vLLM issue #44294）：phase＝seed（1 個請求寫入前綴）→ evict（填充請求擠出 GPU）
   → burst（N 個共用同一前綴的請求同時送）。三段式排程由 m16_vllm_bench.py run 執行。

token id 範圍（vocab_lo, vocab_hi＝[1000, 120000)，左閉右開）：
  - Llama-3.1-8B：vocab_size 128,256，added/special token 是 128000–128255（tokenizer_config.json 的 added_tokens_decoder）。
  - Qwen2.5-7B-1M-noDCA：vocab_size 152,064，special token 是 151643–151664。
  - [1000, 120000) 在兩個模型的 special token 之下；而且和既有的 vLLM 量測（m7_c6_vllm.py、m9_f1_run.py、
    m10_vllm_ttft.py 都用 randrange(1000, 120000)）同一個範圍，數字可以互相對照。
    下限 1000 只是沿用既有腳本（避開最前面的單字元 token），沒有另外驗證必要性。

用法：
  python code/m16_workload.py gen-prefix  --n-req 64 --prompt-len 8192 --overlap 0.5 --n-groups 4 --out-len 128 \
         --arrival poisson --rate 0.5 --seed 0 --out W.jsonl
  python code/m16_workload.py gen-mooncake --trace toolagent --max-req 200 --time-scale 1.0 --max-input-len 32000 --out W.jsonl
  python code/m16_workload.py gen-agentic --n-sess 8 --n-turns 6 --base-ctx 16384 --tool-len uniform:256:2048 \
         --pause uniform:1:120 --out-len 128 --seed 0 --out W.jsonl
  python code/m16_workload.py mooncake-deltas --trace toolagent --out deltas.csv   # I3 的工具輸出長度經驗分佈
  python code/m16_workload.py --self-test
"""
from __future__ import annotations

import argparse
import array
import collections
import hashlib
import importlib.abc
import json
import math
import os
import random
import sys
from datetime import datetime, timezone

# ───────────────────────── 不准碰 GPU runtime ─────────────────────────
GPU_MODULES = ("torch", "vllm", "triton", "amdsmi", "pynvml")


class _ForbidGpuImports(importlib.abc.MetaPathFinder):
    """M16_FORBID_GPU_IMPORTS=1 時裝上：torch／vllm 等一 import 就 ImportError（測試與 client 端用）。"""

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in GPU_MODULES:
            raise ImportError(f"M16_FORBID_GPU_IMPORTS: 不准 import {name}（client／測試路徑不能打開 /dev/kfd）")
        return None


def install_gpu_import_blocker() -> None:
    if not any(isinstance(f, _ForbidGpuImports) for f in sys.meta_path):
        sys.meta_path.insert(0, _ForbidGpuImports())
    loaded = [m for m in sys.modules if m.split(".")[0] in GPU_MODULES]
    if loaded:
        raise RuntimeError(f"M16_FORBID_GPU_IMPORTS: 這些模組已經被載入：{loaded[:5]}")


if os.environ.get("M16_FORBID_GPU_IMPORTS") == "1":
    install_gpu_import_blocker()

# ───────────────────────── 常數 ─────────────────────────
MOONCAKE_BLOCK = 512           # Mooncake hash_ids 的單位（規則 6；載入時逐列驗算）
WS_BLOCK = 16                  # 只用來算「工作集有幾個 16-token block」；vLLM 實際 block size 以 server log 為準
VOCAB_LO_DEFAULT = 1000
VOCAB_HI_DEFAULT = 120000      # 不含
TRACE_DIR = "/mlsteam/data/tiara/datasets/traces"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _rng(seed, *parts) -> random.Random:
    return random.Random("|".join(str(x) for x in (seed,) + parts))


def _toks(rng: random.Random, n: int, lo: int, hi: int) -> list[int]:
    if n <= 0:
        return []
    return rng.choices(range(lo, hi), k=n)


def _check_vocab(lo: int, hi: int) -> None:
    if not (0 <= lo < hi):
        raise ValueError(f"vocab 範圍不合法：[{lo}, {hi})")


def _coprime_stride(span: int) -> int:
    for s in (7919, 104729, 1299709, 15485863):
        if math.gcd(s, span) == 1:
            return s
    s = 2
    while math.gcd(s, span) != 1:
        s += 1
    return s


# ───────────────────────── 分佈 ─────────────────────────
def _load_empirical(spec_rest: str) -> list[float]:
    """empirical:<檔案>[:<欄名>]。檔案：每行一個數，或 CSV（有表頭時用欄名選欄）。# 開頭的行略過。"""
    path, col = spec_rest, None
    if not os.path.exists(path) and ":" in spec_rest:
        path, col = spec_rest.rsplit(":", 1)
    lines = [ln.strip() for ln in open(path) if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"empirical 檔案沒有資料：{path}")
    vals: list[float] = []
    first = lines[0].split(",")
    try:
        float(first[0])
        has_header = False
    except ValueError:
        has_header = True
    if has_header:
        if col is None:
            raise ValueError(f"{path} 有表頭，要用 empirical:<檔案>:<欄名> 指定欄位（可選：{first}）")
        if col not in first:
            raise ValueError(f"{path} 沒有欄位 {col}（可選：{first}）")
        ci = first.index(col)
        for ln in lines[1:]:
            vals.append(float(ln.split(",")[ci]))
    else:
        if col is not None:
            raise ValueError(f"{path} 沒有表頭，不能指定欄名 {col}")
        for ln in lines:
            vals.append(float(ln.split(",")[0]))
    return vals


def parse_dist(spec: str):
    """回傳 sample(rng) -> float。"""
    kind, _, rest = spec.partition(":")
    if kind == "const":
        x = float(rest)
        return lambda rng: x
    if kind == "uniform":
        a, b = (float(v) for v in rest.split(":"))
        if b < a:
            raise ValueError(f"uniform:a:b 需要 a ≤ b：{spec}")
        return lambda rng: rng.uniform(a, b)
    if kind == "lognormal":
        mu, sigma = (float(v) for v in rest.split(":"))
        return lambda rng: rng.lognormvariate(mu, sigma)
    if kind == "empirical":
        vals = _load_empirical(rest)
        return lambda rng: rng.choice(vals)
    raise ValueError(f"不認得的分佈：{spec}（const:x、uniform:a:b、lognormal:mu:sigma、empirical:<檔案>[:<欄名>]）")


# ───────────────────────── 工作集統計（給 F1 設 KV 容量用） ─────────────────────────
def _chain_digests(tokens: list[int], block: int) -> list[bytes]:
    """每個完整 block 的前綴鏈摘要（內容＋前面所有 block）。不含最後不滿一個 block 的部分。"""
    out, prev = [], b""
    for i in range(len(tokens) // block):
        h = hashlib.blake2b(prev, digest_size=16)
        h.update(array.array("I", tokens[i * block:(i + 1) * block]).tobytes())
        prev = h.digest()
        out.append(prev)
    return out


def workset_stats(reqs: list[dict], block: int = WS_BLOCK) -> dict:
    """依送出順序（t_arrive，沒有就照列表順序）：
    ws_blocks：不同的完整 block（前綴鏈）數；ideal_hit_tokens：無限容量、依序送時可命中的 token 數
    （每個請求前面連續、且更早的請求已經出現過的完整 block；上限 prompt_len−1 對齊 block，同 vLLM）。"""
    order = sorted(range(len(reqs)), key=lambda i: (reqs[i].get("t_arrive") or 0.0, i))
    seen: set = set()
    ideal = total = 0
    out_tok = 0
    for i in order:
        p = reqs[i]["prompt_token_ids"]
        ds = _chain_digests(p, block)
        k = 0
        while k < len(ds) and ds[k] in seen:
            k += 1
        cap = ((len(p) - 1) // block) * block
        ideal += min(k * block, cap)
        total += len(p)
        out_tok += int(reqs[i].get("max_tokens") or 0)
        seen.update(ds)
    return dict(block=block, ws_blocks=len(seen), ws_prompt_tokens_in_full_blocks=len(seen) * block,
                total_prompt_tokens=total, total_max_tokens=out_tok,
                ws_blocks_incl_output=len(seen) + math.ceil(out_tok / block) + len(reqs),
                ideal_hit_tokens=ideal, ideal_hit_frac=(ideal / total) if total else None,
                max_prompt_len=max((len(r["prompt_token_ids"]) for r in reqs), default=0))


# ───────────────────────── 1. prefix_overlap ─────────────────────────
def prefix_overlap(n_req: int, prompt_len: int, overlap_frac: float, n_groups: int, out_len: int,
                   arrival: str = "poisson", rate_qps: float = 1.0, seed: int = 0,
                   vocab_lo: int = VOCAB_LO_DEFAULT, vocab_hi: int = VOCAB_HI_DEFAULT) -> list[dict]:
    _check_vocab(vocab_lo, vocab_hi)
    if not (0.0 <= overlap_frac <= 1.0):
        raise ValueError(f"overlap_frac 要在 [0,1]：{overlap_frac}")
    if n_req < 1 or prompt_len < 1 or n_groups < 1 or out_len < 1:
        raise ValueError("n_req、prompt_len、n_groups、out_len 都要 ≥ 1")
    span = vocab_hi - vocab_lo
    if n_req > span:
        raise ValueError(f"n_req={n_req} 超過 vocab 範圍 {span}，無法保證每個請求的獨有 token 互不相同")
    shared_len = int(round(overlap_frac * prompt_len))
    uniq_len = prompt_len - shared_len
    stride = _coprime_stride(span)
    prefixes = [_toks(_rng(seed, "po", "group", g), shared_len, vocab_lo, vocab_hi) for g in range(n_groups)]
    ra = _rng(seed, "po", "arrival")
    if arrival == "burst":
        ts = [0.0] * n_req
    elif arrival in ("poisson", "fixed"):
        if rate_qps <= 0:
            raise ValueError("poisson／fixed 需要 rate_qps > 0")
        ts, t = [], 0.0
        for i in range(n_req):
            if i > 0:
                t += ra.expovariate(rate_qps) if arrival == "poisson" else 1.0 / rate_qps
            ts.append(t)
    else:
        raise ValueError(f"arrival 只能是 poisson、fixed、burst：{arrival}")
    seen_group: set = set()
    out = []
    for i in range(n_req):
        g = i % n_groups
        u = _toks(_rng(seed, "po", "req", i), uniq_len, vocab_lo, vocab_hi)
        if u:
            u[0] = vocab_lo + (i * stride) % span
        out.append(dict(req_id=f"po{i:05d}", kind="prefix_overlap", t_arrive=ts[i],
                        prompt_token_ids=prefixes[g] + u, max_tokens=out_len, group=g, shared_len=shared_len,
                        first_in_group=g not in seen_group, session=None, turn=None))
        seen_group.add(g)
    return out


# ───────────────────────── 2. Mooncake ─────────────────────────
def resolve_trace(name_or_path: str) -> str:
    if os.path.exists(name_or_path):
        return name_or_path
    p = os.path.join(TRACE_DIR, f"{name_or_path}_trace.jsonl")
    if not os.path.exists(p):
        raise FileNotFoundError(f"找不到 trace：{name_or_path}（也不是 {p}）")
    return p


def load_mooncake(path: str) -> tuple[list[dict], dict]:
    """讀整個 trace 並驗證單位（規則 6）。任何一列不符就丟 AssertionError（不改斷言硬過）。"""
    rows: list[dict] = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            n, L = len(r["hash_ids"]), int(r["input_length"])
            assert n >= 1, f"{path}:{ln} hash_ids 是空的"
            assert (n - 1) * MOONCAKE_BLOCK < L <= n * MOONCAKE_BLOCK, (
                f"{path}:{ln} 單位不符：len(hash_ids)={n}、input_length={L}，"
                f"不滿足 ({n}-1)×512 < {L} ≤ {n}×512 → hash_ids 不是 512-token block？（規則 6，停下來回報）")
            rows.append(r)
    assert rows, f"{path} 沒有資料"
    ts = [int(r["timestamp"]) for r in rows]
    assert ts == sorted(ts), f"{path}: timestamp 沒有排序"
    pos_of: dict = {}
    prev_of: dict = {}
    bad_pos = bad_prev = 0
    for r in rows:
        h = r["hash_ids"]
        for i, b in enumerate(h):
            p = h[i - 1] if i else None
            if b in pos_of:
                bad_pos += pos_of[b] != i
                bad_prev += prev_of[b] != p
            else:
                pos_of[b], prev_of[b] = i, p
    assert bad_pos == 0 and bad_prev == 0, (
        f"{path}: 前綴鏈性質不成立（同一 hash 位置不同 {bad_pos} 次、前一個 hash 不同 {bad_prev} 次）"
        "→「同一 hash_id＝同一段內容」的建構不再等於 APC 的前綴共享，停下來回報")
    mult1000 = sum(1 for t in ts if min(t % 1000, 1000 - t % 1000) <= 1) / len(ts)
    meta = dict(path=path, n_rows=len(rows), unit_check="(n-1)*512 < input_length <= n*512：全部列通過",
                chain_check="同 hash 同位置、同前一個 hash：0 例外", ts_min_ms=ts[0], ts_max_ms=ts[-1],
                ts_frac_multiple_of_1000ms=round(mult1000, 4), n_distinct_hash=len(pos_of))
    return rows, meta


def _n_full(r: dict) -> int:
    n = len(r["hash_ids"])
    return n if int(r["input_length"]) == n * MOONCAKE_BLOCK else n - 1


def mooncake_replay(path: str, max_req: int | None = None, time_scale: float = 1.0, seed: int = 0,
                    vocab_lo: int = VOCAB_LO_DEFAULT, vocab_hi: int = VOCAB_HI_DEFAULT,
                    max_input_len: int | None = None, start: int = 0, max_out_len: int | None = None,
                    meta: dict | None = None) -> list[dict]:
    _check_vocab(vocab_lo, vocab_hi)
    if time_scale <= 0:
        raise ValueError("time_scale 要 > 0")
    path = resolve_trace(path)
    rows, lmeta = load_mooncake(path)
    sel: list[tuple[int, dict]] = []
    skipped_long = 0
    for idx in range(start, len(rows)):
        r = rows[idx]
        if max_input_len is not None and int(r["input_length"]) > max_input_len:
            skipped_long += 1
            continue
        sel.append((idx, r))
        if max_req is not None and len(sel) >= max_req:
            break
    if not sel:
        raise ValueError("沒有選到任何請求")
    block_cache: dict[int, list[int]] = {}

    def block(hid: int) -> list[int]:
        b = block_cache.get(hid)
        if b is None:
            b = _toks(_rng(seed, "mc", hid), MOONCAKE_BLOCK, vocab_lo, vocab_hi)
            block_cache[hid] = b
        return b

    ts0 = int(sel[0][1]["timestamp"])
    seen_hash: set = set()
    out = []
    for idx, r in sel:
        h, L = r["hash_ids"], int(r["input_length"])
        toks: list[int] = []
        for hid in h:
            toks.extend(block(hid))
        toks = toks[:L]
        assert len(toks) == L
        k = 0
        nf = _n_full(r)
        while k < nf and h[k] in seen_hash:
            k += 1
        seen_hash.update(h[:nf])
        mt = int(r["output_length"])
        if max_out_len is not None:
            mt = min(mt, max_out_len)
        out.append(dict(req_id=f"mc{idx:06d}", kind="mooncake",
                        t_arrive=(int(r["timestamp"]) - ts0) / 1000.0 * time_scale,
                        prompt_token_ids=toks, max_tokens=max(1, mt), group=h[0],
                        shared_len=k * MOONCAKE_BLOCK, shared_blocks_hash=k, hash_ids=list(h),
                        input_length=L, output_length=int(r["output_length"]), trace_row=idx,
                        timestamp_ms=int(r["timestamp"]), session=None, turn=None))
    cc = mooncake_crosscheck(out)
    if meta is not None:
        meta.update(trace=lmeta, selected=len(out), skipped_longer_than_max_input_len=skipped_long,
                    start=start, time_scale=time_scale, crosscheck=cc)
    return out


def mooncake_crosscheck(reqs: list[dict], brute_n: int = 300) -> dict:
    """第二條獨立路徑：用產生出來的 token（不看 hash_ids）算共享的完整 512-block 數，逐請求和 hash 路徑比對。"""
    seen_dig: set = set()
    hist_tok: collections.Counter = collections.Counter()
    hist_hash: collections.Counter = collections.Counter()
    mism = []
    for r in reqs:
        ds = _chain_digests(r["prompt_token_ids"], MOONCAKE_BLOCK)
        k = 0
        while k < len(ds) and ds[k] in seen_dig:
            k += 1
        seen_dig.update(ds)
        hist_tok[k] += 1
        hist_hash[r["shared_blocks_hash"]] += 1
        if k != r["shared_blocks_hash"]:
            mism.append((r["req_id"], k, r["shared_blocks_hash"]))
    assert not mism, f"token 路徑與 hash_ids 路徑的共享 block 數不一致（前 5 筆：{mism[:5]}）→ 停下來回報"
    # 暴力：前 brute_n 個請求，hash_ids（只看完整 block）兩兩 LCP 的最大值，應等於「連續已出現」的捷徑
    full = [r["hash_ids"][:len(r["prompt_token_ids"]) // MOONCAKE_BLOCK] for r in reqs[:brute_n]]
    bad = 0
    for j in range(len(full)):
        best = 0
        for i in range(j):
            a, b = full[i], full[j]
            m = 0
            while m < len(a) and m < len(b) and a[m] == b[m]:
                m += 1
            best = max(best, m)
        bad += best != reqs[j]["shared_blocks_hash"]
    assert bad == 0, f"hash_ids 兩兩 LCP 與捷徑不一致 {bad} 例 → 停下來回報"
    n = len(reqs)
    return dict(n_req=n, per_request_equal=n, brute_force_pairs_checked_first_n=min(n, brute_n),
                hist_shared_blocks={str(k): v for k, v in sorted(hist_hash.items())},
                hist_identical_between_paths=hist_tok == hist_hash,
                frac_req_with_any_shared_block=round(sum(v for k, v in hist_hash.items() if k > 0) / n, 4))


def mooncake_deltas(path: str) -> tuple[list[dict], dict]:
    """I3 的工具輸出長度／暫停的經驗分佈（〔判讀〕：把 trace 裡「前一輪的延續」當成同一個 agent session 的下一輪）。

    父請求 A 的定義：更早的請求，A 的全部完整 block 是 B 的 hash_ids 的前綴，且是所有這種 A 裡共享最長的。
    只保留「A 只有這一個子請求」的配對（排除共用 system prompt 那種一對多的前綴）。
    delta_in＝B.in − A.in；tool_est＝delta_in − A.out（假設 B 的 prompt 包含 A 的完整輸出，這是假設、不是 trace 給的）；
    pause_est_s＝(B.ts − A.ts)/1000（含 A 自己的生成時間；trace 的時間解析度約 1 s）。
    單位交叉驗證：(nB−nA−1)×512 < delta_in < (nB−nA+1)×512（用 hash 數驗 input_length 差）。"""
    path = resolve_trace(path)
    rows, lmeta = load_mooncake(path)
    key_to_idx: dict = {}
    parent_of: dict[int, int] = {}
    for j, r in enumerate(rows):
        h = r["hash_ids"]
        for k in range(len(h), 0, -1):
            i = key_to_idx.get(tuple(h[:k]))
            if i is not None and int(rows[i]["input_length"]) < int(r["input_length"]):
                parent_of[j] = i
                break
        nf = _n_full(r)
        if nf >= 1:
            key_to_idx[tuple(h[:nf])] = j
    n_children = collections.Counter(parent_of.values())
    out = []
    for j, i in parent_of.items():
        if n_children[i] != 1:
            continue
        A, B = rows[i], rows[j]
        nA, nB = len(A["hash_ids"]), len(B["hash_ids"])
        d = int(B["input_length"]) - int(A["input_length"])
        assert (nB - nA - 1) * MOONCAKE_BLOCK < d < (nB - nA + 1) * MOONCAKE_BLOCK, (i, j, nA, nB, d)
        assert B["hash_ids"][:_n_full(A)] == A["hash_ids"][:_n_full(A)]
        out.append(dict(parent_row=i, child_row=j, delta_in=d, parent_out=int(A["output_length"]),
                        tool_est=d - int(A["output_length"]),
                        pause_est_s=(int(B["timestamp"]) - int(A["timestamp"])) / 1000.0))
    out.sort(key=lambda x: x["child_row"])
    meta = dict(trace=lmeta, n_with_parent=len(parent_of), n_parents=len(n_children),
                n_one_child_pairs=len(out), n_multi_child_parents=sum(1 for v in n_children.values() if v > 1),
                frac_tool_est_nonpositive=(round(sum(1 for o in out if o["tool_est"] <= 0) / len(out), 4)
                                           if out else None))
    return out, meta


# ───────────────────────── 3. agentic ─────────────────────────
def agentic_sessions(n_sess: int, n_turns: int, base_ctx: int, tool_out_len_dist: str, pause_dist: str,
                     out_len: int, seed: int = 0, vocab_lo: int = VOCAB_LO_DEFAULT,
                     vocab_hi: int = VOCAB_HI_DEFAULT, start_dist: str = "const:0",
                     max_ctx: int | None = None, meta: dict | None = None) -> list[dict]:
    _check_vocab(vocab_lo, vocab_hi)
    if n_sess < 1 or n_turns < 1 or base_ctx < 1 or out_len < 1:
        raise ValueError("n_sess、n_turns、base_ctx、out_len 都要 ≥ 1")
    tool_d, pause_d, start_d = parse_dist(tool_out_len_dist), parse_dist(pause_dist), parse_dist(start_dist)
    out = []
    truncated = 0
    for s in range(n_sess):
        rs = _rng(seed, "ag", "sess", s)
        start = max(0.0, float(start_d(rs)))
        prompt = _toks(_rng(seed, "ag", "base", s), base_ctx, vocab_lo, vocab_hi)
        for t in range(n_turns):
            if t == 0:
                warm, shared, tool_len, pause = 0, 0, 0, start
                p = prompt
            else:
                prev = prompt
                outp = _toks(_rng(seed, "ag", "out", s, t - 1), out_len, vocab_lo, vocab_hi)
                tool_len = max(0, int(round(tool_d(_rng(seed, "ag", "toollen", s, t)))))
                tool = _toks(_rng(seed, "ag", "tool", s, t), tool_len, vocab_lo, vocab_hi)
                pause = max(0.0, float(pause_d(_rng(seed, "ag", "pause", s, t))))
                p = prev + outp + tool
                warm, shared = len(prev) + out_len, len(prev)
            if max_ctx is not None and len(p) + out_len > max_ctx:
                truncated += 1
                break
            out.append(dict(req_id=f"ag-s{s:03d}-t{t:02d}", kind="agentic", t_arrive=None, t_ready=pause,
                            prompt_token_ids=p, max_tokens=out_len, group=s, session=s, turn=t,
                            shared_len=shared, warm_prefix_len=warm, tool_len=tool_len))
            prompt = p
    if meta is not None:
        meta.update(sessions_truncated_by_max_ctx=truncated)
    return out


# ───────────────────────── 4. shared_prefix_burst（F2：重現 vLLM issue #44294） ─────────────────────────
def shared_prefix_burst(prefix_len: int, suffix_len: int, n_concurrent: int, seed: int = 0, out_len: int = 64,
                        n_fill: int = 0, fill_len: int = 8192, seed_out_len: int = 1,
                        vocab_lo: int = VOCAB_LO_DEFAULT, vocab_hi: int = VOCAB_HI_DEFAULT) -> list[dict]:
    """三段式（由 m16_vllm_bench.py run 照 phase 依序送）：
      phase=seed  ：1 個請求＝前綴＋獨有尾巴，把前綴寫進 GPU（開卸載時也寫進 CPU）。
      phase=evict ：n_fill 個互不相干的填充請求（各 fill_len 個 token），把前綴擠出 GPU；
                    n_fill＝0 時 bench 端可改用 --evict reset（/reset_prefix_cache 只清 GPU）或靠 --gpu-blocks 放不下。
      phase=burst ：n_concurrent 個請求＝同一個前綴＋各自獨有的尾巴（suffix_len 個），同一時刻送出。
    vLLM 0.28 的 OffloadingConnectorScheduler 在「命中的 chunk 正被別的請求載入」時會延後請求
    （vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:835-857；0.19.1 是 :126-140 的 _blocks_being_loaded）。
    每個請求獨有尾巴的第一個 token 互不相同（同 prefix_overlap 的作法），所以共享長度恰好是 prefix_len。"""
    _check_vocab(vocab_lo, vocab_hi)
    if prefix_len < 1 or suffix_len < 1 or n_concurrent < 1 or out_len < 1 or n_fill < 0:
        raise ValueError("prefix_len、suffix_len、n_concurrent、out_len 要 ≥ 1；n_fill ≥ 0")
    span = vocab_hi - vocab_lo
    if n_concurrent + 1 > span:
        raise ValueError("n_concurrent 太大")
    stride = _coprime_stride(span)
    prefix = _toks(_rng(seed, "sp", "prefix"), prefix_len, vocab_lo, vocab_hi)

    def suffix(i: int) -> list[int]:
        u = _toks(_rng(seed, "sp", "suffix", i), suffix_len, vocab_lo, vocab_hi)
        u[0] = vocab_lo + (i * stride) % span
        return u

    out = [dict(req_id="sp-seed", kind="shared_prefix_burst", phase="seed", t_arrive=0.0,
                prompt_token_ids=prefix + suffix(0), max_tokens=max(1, seed_out_len), group=0,
                shared_len=prefix_len, session=None, turn=None)]
    for j in range(n_fill):
        out.append(dict(req_id=f"sp-fill{j:04d}", kind="shared_prefix_burst", phase="evict", t_arrive=0.0,
                        prompt_token_ids=_toks(_rng(seed, "sp", "fill", j), fill_len, vocab_lo, vocab_hi),
                        max_tokens=1, group=None, shared_len=0, session=None, turn=None))
    for i in range(n_concurrent):
        out.append(dict(req_id=f"sp-b{i:04d}", kind="shared_prefix_burst", phase="burst", t_arrive=0.0,
                        prompt_token_ids=prefix + suffix(i + 1), max_tokens=out_len, group=0,
                        shared_len=prefix_len, session=None, turn=None))
    return out


# ───────────────────────── 輸出 ─────────────────────────
def write_jsonl(reqs: list[dict], path: str, gen_meta: dict) -> dict:
    lines = [json.dumps(r, separators=(",", ":")) for r in reqs]
    sha = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    meta = dict(gen_meta, n=len(reqs), sha256=sha, created=now_iso(), run_id=os.environ.get("RUN_ID", "NOT_SET"),
                workset=workset_stats(reqs))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps({"_meta": meta}, separators=(",", ":")) + "\n")
        for ln in lines:
            f.write(ln + "\n")
    return meta


def read_jsonl(path: str) -> tuple[list[dict], dict]:
    meta: dict = {}
    reqs = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            if "_meta" in o:
                meta = o["_meta"]
            else:
                reqs.append(o)
    return reqs, meta


# ───────────────────────── self-test ─────────────────────────
def _lcp(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def self_test() -> int:
    install_gpu_import_blocker()   # 自我測試一律禁止 torch／vllm
    import tempfile
    fails: list[str] = []

    def check(cond, msg):
        print(("PASS " if cond else "FAIL ") + msg, flush=True)
        if not cond:
            fails.append(msg)

    # 1. prefix_overlap：重疊比例、範圍、確定性、到達
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        rs = prefix_overlap(12, 1000, frac, 3, 8, "burst", seed=7)
        sl = int(round(frac * 1000))
        within = [_lcp(rs[i]["prompt_token_ids"], rs[j]["prompt_token_ids"])
                  for i in range(12) for j in range(i + 1, 12) if rs[i]["group"] == rs[j]["group"]]
        across = [_lcp(rs[i]["prompt_token_ids"], rs[j]["prompt_token_ids"])
                  for i in range(12) for j in range(i + 1, 12) if rs[i]["group"] != rs[j]["group"]]
        check(all(len(r["prompt_token_ids"]) == 1000 for r in rs), f"prefix_overlap frac={frac}: 長度都是 1000")
        check(all(x == sl for x in within), f"prefix_overlap frac={frac}: 同組共同前綴恰好 {sl}（實得 {sorted(set(within))}）")
        check(max(across) < WS_BLOCK, f"prefix_overlap frac={frac}: 跨組共同前綴 < 16（實得最大 {max(across)}）")
        check(all(r["shared_len"] == sl for r in rs), f"prefix_overlap frac={frac}: shared_len 欄位＝{sl}")
        check(all(VOCAB_LO_DEFAULT <= t < VOCAB_HI_DEFAULT for r in rs for t in r["prompt_token_ids"]),
              f"prefix_overlap frac={frac}: token 都在 [1000,120000)")
        check(all(r["t_arrive"] == 0.0 for r in rs), f"prefix_overlap frac={frac}: burst 全部 t=0")
    a = prefix_overlap(20, 512, 0.5, 4, 16, "poisson", 2.0, seed=3)
    b = prefix_overlap(20, 512, 0.5, 4, 16, "poisson", 2.0, seed=3)
    c = prefix_overlap(20, 512, 0.5, 4, 16, "poisson", 2.0, seed=4)
    check(json.dumps(a) == json.dumps(b), "prefix_overlap 同 seed 輸出完全相同")
    check(json.dumps(a) != json.dumps(c), "prefix_overlap 不同 seed 輸出不同")
    big = prefix_overlap(4000, 4, 0.5, 2, 1, "poisson", 5.0, seed=11)
    gaps = [big[i + 1]["t_arrive"] - big[i]["t_arrive"] for i in range(len(big) - 1)]
    mg = sum(gaps) / len(gaps)
    check(abs(mg - 0.2) / 0.2 < 0.05, f"poisson rate=5：平均間隔 {mg:.4f} s（期望 0.2，±5%）")
    fx = prefix_overlap(5, 4, 0.0, 1, 1, "fixed", 4.0, seed=0)
    check([r["t_arrive"] for r in fx] == [0.0, 0.25, 0.5, 0.75, 1.0], "fixed rate=4：間隔 0.25 s")
    ws = workset_stats(prefix_overlap(8, 64, 0.5, 2, 4, "burst", seed=1))
    # 2 組 × 32 token 共享（2 block）＋ 8 × 32 token 獨有（2 block）＝ 4＋16＝20 個不同的 16-token block
    check(ws["ws_blocks"] == 20, f"workset：8 請求×64 token、2 組 50% → 20 個不同 block（實得 {ws['ws_blocks']}）")
    check(ws["ideal_hit_tokens"] == 6 * 32, f"workset：理想命中＝6 個非首請求 × 32（實得 {ws['ideal_hit_tokens']}）")

    # 2. Mooncake：單位斷言、交叉驗證、確定性、負對照
    for name in ("toolagent", "conversation"):
        try:
            m: dict = {}
            r1 = mooncake_replay(name, max_req=400, seed=5, meta=m)
            r2 = mooncake_replay(name, max_req=400, seed=5)
            check(True, f"mooncake {name}: 整個檔案 {m['trace']['n_rows']} 列通過 (n−1)×512 < input_length ≤ n×512")
            check(all(len(r["prompt_token_ids"]) == r["input_length"] for r in r1), f"mooncake {name}: prompt 長度＝input_length")
            check(m["crosscheck"]["hist_identical_between_paths"], f"mooncake {name}: token 路徑與 hash 路徑的共享 block 分佈相同 "
                  f"{m['crosscheck']['hist_shared_blocks'] if len(m['crosscheck']['hist_shared_blocks']) < 8 else '(略)'}")
            check(m["crosscheck"]["frac_req_with_any_shared_block"] > 0, f"mooncake {name}: 有請求共享前綴"
                  f"（{m['crosscheck']['frac_req_with_any_shared_block']}）")
            check(json.dumps(r1) == json.dumps(r2), f"mooncake {name}: 同 seed 輸出完全相同")
            # 同 hash 同內容
            pos: dict = {}
            ok = True
            for r in r1:
                for bi, hid in enumerate(r["hash_ids"][:r["input_length"] // 512]):
                    seg = tuple(r["prompt_token_ids"][bi * 512:(bi + 1) * 512])
                    ok &= pos.setdefault(hid, seg) == seg
            check(ok, f"mooncake {name}: 同一個 hash_id 在所有請求裡是同一段 512 token")
            check(r1[0]["t_arrive"] == 0.0 and all(r1[i]["t_arrive"] <= r1[i + 1]["t_arrive"] for i in range(len(r1) - 1)),
                  f"mooncake {name}: t_arrive 從 0 開始、不遞減")
        except (AssertionError, FileNotFoundError) as e:
            check(False, f"mooncake {name}: {e}")
    with tempfile.TemporaryDirectory() as td:
        bad = os.path.join(td, "bad_trace.jsonl")
        with open(bad, "w") as f:   # 故意用 16-token 的單位：input_length＝len×16
            for i in range(5):
                f.write(json.dumps(dict(timestamp=i, input_length=16 * (i + 3), output_length=4,
                                        hash_ids=list(range(i * 10, i * 10 + i + 3)))) + "\n")
        try:
            load_mooncake(bad)
            check(False, "負對照：16-token 單位的假 trace 應該被斷言擋下")
        except AssertionError as e:
            check("單位不符" in str(e), "負對照：16-token 單位的假 trace 被斷言擋下")
    try:
        dl, dm = mooncake_deltas("toolagent")
        check(dm["n_one_child_pairs"] > 0, f"mooncake-deltas toolagent：{dm['n_one_child_pairs']} 個一對一配對"
              f"（tool_est ≤ 0 的比例 {dm['frac_tool_est_nonpositive']}；有父請求 {dm['n_with_parent']}）")
    except AssertionError as e:
        check(False, f"mooncake-deltas: {e}")

    # 3. agentic：嚴格前綴延伸、長度、暫停、確定性
    with tempfile.TemporaryDirectory() as td:
        emp = os.path.join(td, "emp.txt")
        open(emp, "w").write("# test\n3\n5\n7\n")
        for tool_spec, pause_spec in (("const:100", "const:2.5"), ("uniform:10:50", "uniform:1:3"),
                                      ("lognormal:4:0.5", "lognormal:0:1"), (f"empirical:{emp}", f"empirical:{emp}")):
            ag = agentic_sessions(3, 4, 300, tool_spec, pause_spec, 20, seed=9)
            by = collections.defaultdict(list)
            for r in ag:
                by[r["session"]].append(r)
            ok_prefix = ok_len = ok_warm = True
            for s, turns in by.items():
                for t in range(1, len(turns)):
                    pp, cp = turns[t - 1]["prompt_token_ids"], turns[t]["prompt_token_ids"]
                    ok_prefix &= cp[:len(pp)] == pp and len(cp) > len(pp)
                    ok_len &= len(cp) == len(pp) + 20 + turns[t]["tool_len"]
                    ok_warm &= turns[t]["warm_prefix_len"] == len(pp) + 20 and turns[t]["shared_len"] == len(pp)
            check(len(ag) == 12, f"agentic {tool_spec}/{pause_spec}: 3×4＝12 輪")
            check(ok_prefix, f"agentic {tool_spec}: 每輪 prompt 是上一輪的嚴格前綴延伸")
            check(ok_len, f"agentic {tool_spec}: 長度＝上一輪＋out_len＋tool_len")
            check(ok_warm, f"agentic {tool_spec}: warm_prefix_len＝上一輪＋out_len、shared_len＝上一輪")
            pauses = [r["t_ready"] for r in ag if r["turn"] > 0]
            tls = [r["tool_len"] for r in ag if r["turn"] > 0]
            if pause_spec.startswith("const"):
                check(all(p == 2.5 for p in pauses), "agentic const:2.5：暫停都是 2.5")
                check(all(x == 100 for x in tls), "agentic const:100：工具輸出都是 100")
            elif pause_spec.startswith("uniform"):
                check(all(1 <= p <= 3 for p in pauses) and all(10 <= x <= 50 for x in tls), "agentic uniform：都在範圍內")
            elif pause_spec.startswith("lognormal"):
                check(all(p > 0 for p in pauses), "agentic lognormal：暫停 > 0")
            else:
                check(set(pauses) <= {3.0, 5.0, 7.0} and set(tls) <= {3, 5, 7}, "agentic empirical：都取自檔案的值")
            ag2 = agentic_sessions(3, 4, 300, tool_spec, pause_spec, 20, seed=9)
            check(json.dumps(ag) == json.dumps(ag2), f"agentic {tool_spec}: 同 seed 輸出完全相同")
        agc = agentic_sessions(2, 10, 300, "const:100", "const:1", 20, seed=0, max_ctx=700)
        check(all(len(r["prompt_token_ids"]) + 20 <= 700 for r in agc), "agentic max_ctx：沒有超過上限")
        # JSONL 來回
        p = os.path.join(td, "w.jsonl")
        meta = write_jsonl(ag, p, dict(generator="agentic"))
        back, bm = read_jsonl(p)
        check(json.dumps(back) == json.dumps(ag) and bm["sha256"] == meta["sha256"], "JSONL 寫出再讀回完全相同")
    # 4. shared_prefix_burst
    sp = shared_prefix_burst(1000, 37, 8, seed=2, out_len=5, n_fill=3, fill_len=300)
    ph = collections.Counter(r["phase"] for r in sp)
    check(ph == {"seed": 1, "evict": 3, "burst": 8} and [r["phase"] for r in sp][:1] == ["seed"],
          f"shared_prefix_burst：1 seed＋3 evict＋8 burst，seed 在最前（實得 {dict(ph)}）")
    pre = [r["prompt_token_ids"] for r in sp if r["phase"] in ("seed", "burst")]
    lcps = [_lcp(pre[i], pre[j]) for i in range(len(pre)) for j in range(i + 1, len(pre))]
    check(set(lcps) == {1000}, f"shared_prefix_burst：seed 與 burst 兩兩共同前綴恰好 1000（實得 {sorted(set(lcps))}）")
    fills = [r["prompt_token_ids"] for r in sp if r["phase"] == "evict"]
    check(all(_lcp(f, pre[0]) < WS_BLOCK for f in fills) and all(len(f) == 300 for f in fills),
          "shared_prefix_burst：填充請求和前綴不共享（< 16 token）、長度 300")
    check(json.dumps(sp) == json.dumps(shared_prefix_burst(1000, 37, 8, seed=2, out_len=5, n_fill=3, fill_len=300)),
          "shared_prefix_burst：同 seed 輸出完全相同")
    check(not [m for m in sys.modules if m.split(".")[0] in GPU_MODULES], "整個 self-test 沒有載入 torch／vllm")
    if fails:
        print(f"FAILED {len(fails)} 項：", *fails, sep="\n  ")
        return 1
    print("ALL PASS")
    return 0


# ───────────────────────── CLI ─────────────────────────
def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "--self-test":
        return self_test()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--vocab-lo", type=int, default=VOCAB_LO_DEFAULT)
        p.add_argument("--vocab-hi", type=int, default=VOCAB_HI_DEFAULT)
        p.add_argument("--out", required=True)

    p = sub.add_parser("gen-prefix")
    p.add_argument("--n-req", type=int, required=True)
    p.add_argument("--prompt-len", type=int, required=True)
    p.add_argument("--overlap", type=float, required=True)
    p.add_argument("--n-groups", type=int, default=1)
    p.add_argument("--out-len", type=int, default=128)
    p.add_argument("--arrival", choices=["poisson", "fixed", "burst"], default="poisson")
    p.add_argument("--rate", type=float, default=1.0)
    common(p)
    p = sub.add_parser("gen-mooncake")
    p.add_argument("--trace", required=True, help="toolagent／conversation 或檔案路徑")
    p.add_argument("--max-req", type=int, default=None)
    p.add_argument("--time-scale", type=float, default=1.0)
    p.add_argument("--max-input-len", type=int, default=None, help="比這長的列跳過（計數寫進 meta）")
    p.add_argument("--max-out-len", type=int, default=None)
    p.add_argument("--start", type=int, default=0)
    common(p)
    p = sub.add_parser("gen-agentic")
    p.add_argument("--n-sess", type=int, required=True)
    p.add_argument("--n-turns", type=int, required=True)
    p.add_argument("--base-ctx", type=int, required=True)
    p.add_argument("--tool-len", required=True, help="const:x｜uniform:a:b｜lognormal:mu:sigma｜empirical:<檔>[:<欄>]")
    p.add_argument("--pause", required=True, help="同上（秒）")
    p.add_argument("--out-len", type=int, default=128)
    p.add_argument("--start-dist", default="const:0")
    p.add_argument("--max-ctx", type=int, default=None)
    common(p)
    p = sub.add_parser("gen-burst", help="F2：seed → evict → burst（重現 vLLM #44294）")
    p.add_argument("--prefix-len", type=int, required=True)
    p.add_argument("--suffix-len", type=int, default=256)
    p.add_argument("--n-concurrent", type=int, required=True)
    p.add_argument("--out-len", type=int, default=64)
    p.add_argument("--n-fill", type=int, default=0, help="填充請求數（0＝bench 端用 --evict reset 或靠容量）")
    p.add_argument("--fill-len", type=int, default=8192)
    p.add_argument("--seed-out-len", type=int, default=1)
    common(p)
    p = sub.add_parser("mooncake-deltas")
    p.add_argument("--trace", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "mooncake-deltas":
        rows, meta = mooncake_deltas(a.trace)
        with open(a.out, "w") as f:
            f.write(f"# m16_workload.py mooncake-deltas {now_iso()} run_id={os.environ.get('RUN_ID', 'NOT_SET')}\n")
            f.write("# " + json.dumps(meta, ensure_ascii=False) + "\n")
            f.write("# tool_est＝delta_in−parent_out 是〔判讀〕（假設下一輪 prompt 含上一輪完整輸出）\n")
            cols = ["parent_row", "child_row", "delta_in", "parent_out", "tool_est", "pause_est_s"]
            f.write(",".join(cols) + "\n")
            for r in rows:
                f.write(",".join(str(r[c]) for c in cols) + "\n")
        print(json.dumps(meta, ensure_ascii=False))
        return 0
    gm: dict = dict(generator=a.cmd, args=vars(a))
    if a.cmd == "gen-prefix":
        reqs = prefix_overlap(a.n_req, a.prompt_len, a.overlap, a.n_groups, a.out_len, a.arrival, a.rate, a.seed,
                              a.vocab_lo, a.vocab_hi)
    elif a.cmd == "gen-mooncake":
        mm: dict = {}
        reqs = mooncake_replay(a.trace, a.max_req, a.time_scale, a.seed, a.vocab_lo, a.vocab_hi, a.max_input_len,
                               a.start, a.max_out_len, meta=mm)
        gm["mooncake"] = mm
    elif a.cmd == "gen-burst":
        reqs = shared_prefix_burst(a.prefix_len, a.suffix_len, a.n_concurrent, a.seed, a.out_len, a.n_fill,
                                   a.fill_len, a.seed_out_len, a.vocab_lo, a.vocab_hi)
    else:
        mm = {}
        reqs = agentic_sessions(a.n_sess, a.n_turns, a.base_ctx, a.tool_len, a.pause, a.out_len, a.seed,
                                a.vocab_lo, a.vocab_hi, a.start_dist, a.max_ctx, meta=mm)
        gm["agentic"] = mm
    meta = write_jsonl(reqs, a.out, gm)
    print(json.dumps({k: meta[k] for k in ("n", "sha256", "workset")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
