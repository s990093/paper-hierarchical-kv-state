#!/usr/bin/env python3
"""M5-C：注意力重要度 ↔ 重用預測 的對齊分析。

## 這支腳本回答什麼

`m5_predictor.py` 的說明裡明白寫著：論文 §5.2 的六族特徵，trace 驅動的 PoC
只做得到三族，另外兩族標 **NOT_AVAILABLE**（pooled key/value 統計、`attn_mass`），
理由是「要跑模型才有」，並註記「表 15 的 (C) 區塊（特徵集消融）做不到」。

**這支腳本就是去把 `attn_mass` 那一族量出來。**

三個問題，缺一不可（只做第 3 個等於沒有證據）：

1. **注意力重要度**能不能預測「這個 block 之後會被重用」？（對真值的 AUC）
2. **存取歷史特徵**能不能？（對同一個真值的 AUC）
3. 兩者彼此相關嗎？（Spearman）

🔴 **「對齊」不等於「正確」**：兩個訊號一致，可能是都對，也可能是錯得一樣。
只有在**兩者各自都能預測真值**的前提下，彼此相關才構成收斂效度。
所以本檔一定同時報三個數字，不單獨報相關係數。

## 怎麼量注意力重要度（為什麼不是 n×n）

文獻（H2O 累積注意力、SnapKV 觀察窗、Quest 的 block 選擇）都需要注意力分佈，
而具體化 n×n 的矩陣與 FlashAttention 這類融合 kernel 不相容——那正是生產推論在用的。
這裡用 **SnapKV 式的觀察窗**：只算最後 w 個 query 對所有 key 的分數，
成本 O(w·n) 而不是 O(n²)。w=32、n=16,384 時每層只有 32×32×16,384 個分數。

做法：用 transformers 的 `AttentionInterface` 註冊一個自訂 attention，
它照常用 SDPA 算出正確輸出（不影響模型行為），**另外**用後 w 個 query 算一次分數，
在 GPU 上就地聚合成 per-block 的重要度後丟棄，不留下大張量。

## 工作負載：同一批 block 要同時有「真實注意力」與「真實重用」

* N 份真實長文件（LongBench 原文）當共享前綴
* 每份文件配多個查詢，文件依 Zipf 分佈被重複挑中 → 前綴 block 有真實的重用歷史
* 每個請求跑一次前向 → 得到該請求對每個 block 的注意力重要度
* 真值：該 block 在其後 H 個請求內有沒有再被存取

用法：
    python code/m5_attention_importance.py --model b-llama8b --docs 24 --requests 240
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parent.parent
PLATFORM = os.environ.get("TIARA_PLATFORM") or ("B" if Path("/opt/rocm").exists() else "A")
BIG = Path(os.environ.get("TIARA_DATA", "/mlsteam/data/tiara") if PLATFORM == "B"
           else os.environ.get("PAPER_HKV_BIG", "/ssd7/hungwei/paper-hkv"))
OUT = REPO / ("results/m5_attention_mi300x" if PLATFORM == "B" else "results/m5_attention")
RUNS = Path(os.environ.get("TIARA_RUNS", str(BIG / "runs")))
LB = BIG / "datasets/longbench"

BLOCK = 16          # 與 vLLM 的 block size 一致，才和 M2–M4 的成本模型可通約

MODELS = {
    "b-llama8b": "unsloth/Llama-3.1-8B-Instruct",
    "b-ultralong8b-1m": "nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct",
    "b-qwen7b-1m": str(BIG / "models/Qwen2.5-7B-Instruct-1M-noDCA"),
    "b-qwen14b-1m": str(BIG / "models/Qwen2.5-14B-Instruct-1M-noDCA"),
    "b-mistral-nemo12b": "mistralai/Mistral-Nemo-Instruct-2407",
    "b-qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "b-seedoss36b": "ByteDance-Seed/Seed-OSS-36B-Instruct",
}

# 觀察窗大小（最後幾個 query 參與計分）。SnapKV 用 32；這裡可調並記進 CSV。
OBS_WINDOW = 32

# 每層聚合後的 per-block 重要度會累加到這裡（由自訂 attention 寫入）
_ACC: dict = {"block_sum": None, "n_layers": 0, "seq_len": 0}


def _make_attention_fn(sdpa_fn):
    """回傳一個 attention 實作：照常算輸出，另外累積觀察窗的 per-block 注意力質量。"""
    import torch

    def fn(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        out = sdpa_fn(module, query, key, value, attention_mask, scaling, dropout, **kwargs)

        if _ACC["block_sum"] is not None:
            q_len = query.shape[-2]
            k_len = key.shape[-2]
            w = min(OBS_WINDOW, q_len)
            if w > 0 and k_len >= BLOCK:
                # GQA：key 的 head 數可能少於 query，重複展開以對齊
                n_rep = query.shape[1] // key.shape[1]
                k = key.repeat_interleave(n_rep, dim=1) if n_rep > 1 else key
                q = query[:, :, -w:, :].float()
                scores = torch.matmul(q, k.float().transpose(-1, -2)) * scaling   # [b,h,w,k_len]
                # 因果遮罩：第 j 個觀察 query 的絕對位置是 q_len-w+j
                pos = torch.arange(k_len, device=scores.device)
                qpos = torch.arange(q_len - w, q_len, device=scores.device).unsqueeze(-1)
                scores = scores.masked_fill(pos.unsqueeze(0) > qpos, float("-inf"))
                p = torch.softmax(scores, dim=-1)                                  # 每個 query 加總為 1
                mass = p.sum(dim=(0, 1, 2))                                        # [k_len]
                nb = k_len // BLOCK
                if nb:
                    blk = mass[: nb * BLOCK].view(nb, BLOCK).sum(-1)
                    _ACC["block_sum"][:nb] += blk.detach().to(_ACC["block_sum"].dtype)
                    _ACC["n_layers"] += 1
                    _ACC["seq_len"] = k_len
        return out

    return fn


def load_docs(n_docs: int, doc_tokens: int, tok, seed: int) -> list[str]:
    """從 LongBench 取真實長文件當共享前綴（不是隨機 token —— 注意力分佈會完全不同）。"""
    src = []
    for task in ("gov_report", "hotpotqa", "multifieldqa_en", "qasper"):
        f = LB / f"data/{task}.jsonl"
        if not f.exists():
            continue
        for line in f.open():
            d = json.loads(line)
            t = d.get("context") or ""
            if t:
                src.append(t)
    if not src:
        raise SystemExit(f"🔴 找不到 LongBench 原文（{LB}/data/*.jsonl）——"
                         f"注意力重要度必須用真實文本量，隨機 token 沒有意義。")
    rng = random.Random(seed)
    rng.shuffle(src)
    docs = []
    for t in src:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        if len(ids) < doc_tokens:
            continue
        docs.append(tok.decode(ids[:doc_tokens], skip_special_tokens=True))
        if len(docs) == n_docs:
            break
    if len(docs) < n_docs:
        raise SystemExit(f"🔴 只湊到 {len(docs)} 份 ≥{doc_tokens} token 的文件，需要 {n_docs}")
    return docs


def zipf_pick(n: int, alpha: float, rng: random.Random) -> int:
    w = [1.0 / ((i + 1) ** alpha) for i in range(n)]
    s = sum(w)
    r = rng.random() * s
    acc = 0.0
    for i, x in enumerate(w):
        acc += x
        if r <= acc:
            return i
    return n - 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="b-llama8b", choices=list(MODELS))
    ap.add_argument("--docs", type=int, default=24)
    ap.add_argument("--doc-tokens", type=int, default=4096)
    ap.add_argument("--requests", type=int, default=240)
    ap.add_argument("--alpha", type=float, default=0.9, help="Zipf 偏斜，與 M4 一致")
    ap.add_argument("--horizon", type=int, default=32, help="真值：其後 H 個請求內是否再被存取")
    ap.add_argument("--obs-window", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    global OBS_WINDOW
    OBS_WINDOW = a.obs_window

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from gpu_guard import GpuWatcher, host_contention

    os.environ.setdefault("HF_HOME", str(BIG / ("hf-cache" if PLATFORM == "B"
                                                else "hf-cache/huggingface")))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(a.gpu))
    os.environ.setdefault("HIP_VISIBLE_DEVICES", str(a.gpu))

    model_path = MODELS[a.model]
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-m5c-attn-{a.model}"
    root = RUNS / run_id
    root.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[m5c] run_id={run_id} model={model_path}")

    tok = AutoTokenizer.from_pretrained(model_path)
    docs = load_docs(a.docs, a.doc_tokens, tok, a.seed)
    print(f"[m5c] {len(docs)} 份文件 × {a.doc_tokens} token（LongBench 原文）")

    sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]
    ALL_ATTENTION_FUNCTIONS.register("tiara_obs", _make_attention_fn(sdpa))

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="tiara_obs")
    model.eval()
    print(f"[m5c] 模型載入完成，層數 {model.config.num_hidden_layers}")

    doc_blocks = a.doc_tokens // BLOCK
    rng = random.Random(a.seed)
    queries = ["\n\nQuestion: What is the main finding described above? Answer:",
               "\n\nQuestion: Summarise the key entity mentioned. Answer:",
               "\n\nQuestion: Which year is most relevant here? Answer:"]

    # 1) 先產生請求序列（文件 id），真值才有得算
    seq = [zipf_pick(len(docs), a.alpha, rng) for _ in range(a.requests)]

    # 2) 每個 (請求, block) 的存取歷史特徵（與 m5_predictor 的 deltas/EDC 同構，但自足）
    last_seen: dict[int, list[int]] = defaultdict(list)     # global block id -> 存取過的請求序號
    rows = []
    guard_path = OUT / f"gpu_guard_attn_{a.model}.json"

    with GpuWatcher(gpu=a.gpu, out_path=str(guard_path)) as g:
        if not g.started_clean:
            print(f"[m5c] 🔴 GPU {a.gpu} 開跑前就不乾淨：{g.intruders}")
            return 2
        # 注意：本腳本不啟動 server，所以沒有 teardown 假污染的問題；
        # 若 guard 標示污染，那是真的有別的行程在用同一張卡（例如另一個佇列），
        # 此時前向傳播的數值不受影響，但仍逐列記錄，並在報告中標明。
        for t, d in enumerate(seq):
            prompt = docs[d] + queries[t % len(queries)]
            ids = tok(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].cuda()
            n_blk = ids.shape[-1] // BLOCK
            _ACC["block_sum"] = torch.zeros(n_blk, dtype=torch.float32, device="cuda")
            _ACC["n_layers"] = 0
            with torch.no_grad():
                model(ids, use_cache=False)
            imp = (_ACC["block_sum"] / max(1, _ACC["n_layers"])).cpu().tolist()
            _ACC["block_sum"] = None

            # 只取文件本體的 block（尾端的查詢句不屬於共享前綴）
            for b in range(min(doc_blocks, len(imp))):
                gid = d * doc_blocks + b
                hist = last_seen[gid]
                deltas = [t - hist[-i] for i in range(1, min(4, len(hist)) + 1)] if hist else []
                rows.append({
                    "run_id": run_id, "ts": datetime.now().astimezone().isoformat(),
                    "model_key": a.model, "req_idx": t, "doc_id": d,
                    "block_in_doc": b, "global_block": gid,
                    "attn_mass": round(imp[b], 8),
                    "attn_rank_in_req": 0,          # 稍後填
                    "n_prev_access": len(hist),
                    "age_since_last": (t - hist[-1]) if hist else -1,
                    "delta_1": deltas[0] if len(deltas) > 0 else -1,
                    "delta_2": deltas[1] if len(deltas) > 1 else -1,
                    "delta_3": deltas[2] if len(deltas) > 2 else -1,
                    "reused_within_horizon": 0,     # 稍後填
                    "obs_window": a.obs_window, "block_tokens": BLOCK,
                    "doc_tokens": a.doc_tokens, "alpha": a.alpha, "horizon": a.horizon,
                    "n_layers_aggregated": _ACC["n_layers"],
                    "seq_len": _ACC["seq_len"],
                    "platform": PLATFORM,
                    **{k: v for k, v in host_contention(exclude_gpu=a.gpu).items()
                       if k in ("level", "foreign_gpu_count", "foreign_max_util")},
                    "log": str(root),
                })
            for b in range(min(doc_blocks, len(imp))):
                last_seen[d * doc_blocks + b].append(t)
            if (t + 1) % 10 == 0:
                print(f"[m5c]   {t + 1}/{len(seq)} 個請求", flush=True)

    # 3) 真值：其後 horizon 個請求內是否再被存取
    access = defaultdict(list)
    for t, d in enumerate(seq):
        for b in range(doc_blocks):
            access[d * doc_blocks + b].append(t)
    for r in rows:
        later = [x for x in access[r["global_block"]] if r["req_idx"] < x <= r["req_idx"] + a.horizon]
        r["reused_within_horizon"] = int(bool(later))

    # 每個請求內的注意力排名（百分位；跨請求可比）
    by_req = defaultdict(list)
    for r in rows:
        by_req[r["req_idx"]].append(r)
    for _, rs in by_req.items():
        order = sorted(rs, key=lambda x: -x["attn_mass"])
        for i, r in enumerate(order):
            r["attn_rank_in_req"] = round(i / max(1, len(order) - 1), 5)

    path = Path(a.csv) if a.csv else OUT / f"attn_importance_{a.model}.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[m5c] wrote {len(rows)} rows -> {path}")

    # 4) 三個數字：兩個訊號各自的 AUC + 彼此的 Spearman
    import statistics as st

    def auc(score: list[float], label: list[int]) -> float:
        pairs = sorted(zip(score, label))
        ranks = {}
        i = 0
        while i < len(pairs):
            j = i
            while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
                j += 1
            r = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[k] = r
            i = j + 1
        pos = [ranks[k] for k, (_, y) in enumerate(pairs) if y == 1]
        n1, n0 = len(pos), len(pairs) - len(pos)
        if not n1 or not n0:
            return float("nan")
        return (sum(pos) - n1 * (n1 + 1) / 2) / (n1 * n0)

    def spearman(x: list[float], y: list[float]) -> float:
        def rank(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            r = [0.0] * len(v)
            i = 0
            while i < len(order):
                j = i
                while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                    j += 1
                avg = (i + j) / 2 + 1
                for k in range(i, j + 1):
                    r[order[k]] = avg
                i = j + 1
            return r
        rx, ry = rank(x), rank(y)
        mx, my = st.mean(rx), st.mean(ry)
        num = sum((a1 - mx) * (b1 - my) for a1, b1 in zip(rx, ry))
        den = math.sqrt(sum((a1 - mx) ** 2 for a1 in rx) * sum((b1 - my) ** 2 for b1 in ry))
        return num / den if den else float("nan")

    # 只用「有存取歷史」的樣本比較，否則歷史特徵沒有定義
    ev = [r for r in rows if r["n_prev_access"] > 0]
    y = [r["reused_within_horizon"] for r in ev]
    a_sig = [r["attn_mass"] for r in ev]
    # 歷史訊號：越近期被用過、用過越多次 → 越可能重用（負的 age + 次數，與 LRU/LFU 同構）
    h_sig = [r["n_prev_access"] - 0.01 * r["age_since_last"] for r in ev]
    res = {
        "run_id": run_id, "model_key": a.model, "n_eval": len(ev),
        "positive_rate": round(sum(y) / max(1, len(y)), 4),
        "auc_attn_mass": round(auc(a_sig, y), 4),
        "auc_access_history": round(auc(h_sig, y), 4),
        "spearman_attn_vs_history": round(spearman(a_sig, h_sig), 4),
        "obs_window": a.obs_window, "docs": a.docs, "doc_tokens": a.doc_tokens,
        "requests": a.requests, "alpha": a.alpha, "horizon": a.horizon,
        "csv": str(path),
    }
    (OUT / f"alignment_{a.model}.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    print("\n[m5c] === 對齊分析（三個數字缺一不可）===")
    print(f"  樣本 {res['n_eval']:,}，正例率 {res['positive_rate']:.3f}")
    print(f"  1) 注意力重要度 → 預測重用 AUC = {res['auc_attn_mass']}")
    print(f"  2) 存取歷史     → 預測重用 AUC = {res['auc_access_history']}")
    print(f"  3) 兩者 Spearman 相關            = {res['spearman_attn_vs_history']}")
    print("  🔴 只有 1 與 2 都明顯 > 0.5 時，3 才構成收斂效度；否則相關只是共同的偏誤。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
