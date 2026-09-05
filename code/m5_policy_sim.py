#!/usr/bin/env python3
"""Milestone 5 第二階段（下半）— 把訓練好的預測器接回模擬器。

`m5_predictor.py` 回答的是「預測器準不準」，這支回答的是**唯一重要的問題**：

> 一個真的**線上**、只看得到過去的策略，能拿到 oracle headroom 的多少？

這正是 M4 判定書列的三處缺口之一（`PRIMER.md` §7.8：
「(i) 線上策略能取得 oracle 的多少比例」）。

## 設計：把 oracle 的「知道未來」換成「預測未來」

`m4_oracle.Sim.run_oracle` 的兩個決策各有一個未來量：

| 決策 | oracle 用什麼 | 本策略用什麼 |
|---|---|---|
| 逐出誰 | 真實的下次使用時刻（Bélády） | $t + \\hat\\tau$，$\\hat\\tau$ 為 GBDT 的預測 |
| 逐出到哪 | 真實會不會再被用到 | 校準後的 $\\hat p$ 與式 (9) 的門檻 $p^{*}(pos)$ |

**其餘一切完全相同**——成本記帳、前綴語意、預取重疊、decode 都直接沿用
`Sim` 的方法，不是重寫。這一點是可驗證的，不是宣稱：

    python code/m5_policy_sim.py --check-shim

`lru-shim` 模式把預測換成「上次存取時刻」（＝LRU 的排序），並走 cascade
的目的地規則——此時本檔的迴圈**必須逐位元重現** `Sim.run_online(policy="lru")`。
對不上就代表本檔的記帳與 M4 的不同，那麼「線上策略 vs oracle」的比較
就是在比兩把不同的尺。

## 🔴 評估只在測試段做

預測器是用 trace 前 70% 訓練的。在整條 trace 上跑線上策略，等於讓它在
自己的訓練資料上表演。預設 `--segment test` 只跑後 30%，
**baseline 與 oracle 也跑同一段**，起始快取同樣是空的。

## 🔴 評分時機與論文演算法 1 的差異

演算法 1 是在**驅逐時**對取樣到的 64 個候選評分；本檔是在**每次存取時**
評分一次，之後沿用到該 block 下次被存取為止。
差別在於：驅逐時評分看得到「已經多久沒被用了」，存取時評分看不到。
這個近似讓 12.7M 次存取只要一次批次推論（否則每次驅逐都要呼叫一次模型，
Python 端的呼叫開銷就要數小時），代價是預測略偏「保留」。
**這是本檔的近似，不是論文的設計**，引用時必須一起寫。
"""
from __future__ import annotations

import argparse
import csv
import json
import heapq
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, os.environ.get("PAPER_HKV_PYLIBS",
                                  "/ssd7/hungwei/paper-hkv/pylibs"))

from m4_invariants import (check_results, check_trace_units,       # noqa: E402
                           preflight)
from m4_oracle import (BLOCK, DEVICE_FS_ROOT, DEVICE_WRITE_MIBPS,   # noqa: E402
                       MODEL_PROFILES, SIM_VERSION, Sim,
                       check_decode_bandwidth, load_cost_model,
                       load_decode_model, mooncake_outputs, mooncake_trace,
                       profile, trace_duration_s)
from m5_predictor import (OUT, drop_cost, isotonic_predict,         # noqa: E402
                          load_index, p_star, write_csv)

BIG = Path(os.environ.get("PAPER_HKV_BIG", "/ssd7/hungwei/paper-hkv"))
POLICIES = {"full_gpu": ("lru", False, False), "cpu_lru": ("lru", True, False),
            "cpu_arc": ("arc", True, False), "tier_fs": ("lru", True, True)}


class PolicySim(Sim):
    """線上策略：逐出順序與目的地都由預測器決定。

    繼承 `Sim` 而不是改它——`m4_oracle.sim_version()` 會雜湊 `Sim` 的原始碼，
    改一個字元就會讓既有 results/ 裡所有 CSV 的 `sim_version` 對不上，
    等於把 M4 的結果全部與新結果切開。
    """

    def run_learned(self, trace, next_use, p_hat, t_offset: int,
                    use_cpu: bool, use_ssd: bool, prefix_semantics: bool = True,
                    prefetch: bool = False, outputs=None, per_request=False,
                    dest: str = "cost-aware", mode: str = "learned",
                    drop_cost: str = "tail") -> dict:
        """`next_use[t]`／`p_hat[t]`：第 t 次存取（全域序號）當下對該 block 的預測。

        mode="lru-shim"：忽略預測，改用「上次存取時刻」排序（＝LRU）並走
        cascade 目的地規則。此模式必須逐位元重現 `Sim.run_online("lru")`。
        """
        cm = self.cm
        cascade = dest == "cascade" or mode == "lru-shim"
        gpu: dict[int, float] = {}          # block -> 排序鍵（大＝預測晚用到＝先逐出）
        heap: list[tuple[float, int]] = []
        # cascade 用 OrderedDict（與 Sim.run_online 同一種語意），
        # cost-aware 用 dict + heap（與 Sim.run_oracle 同一種語意）
        cpu_od: OrderedDict[int, None] = OrderedDict()
        ssd_od: OrderedDict[int, None] = OrderedDict()
        cpu: dict[int, float] = {}
        ssd: dict[int, float] = {}
        cpu_h: list[tuple[float, int]] = []
        ssd_h: list[tuple[float, int]] = []
        cpu_set = cpu_od if cascade else cpu
        ssd_set = ssd_od if cascade else ssd
        # blk -> (排序鍵, p̂, 丟掉的邊際成本)。只保留還在某一階的 block，
        # 否則這個 dict 會長到跟工作集一樣大（5.5M 筆）。
        st: dict[int, tuple[float, float, float]] = {}

        total = decode_total = prev_compute = 0.0
        per_req: list[float] = []
        hits = {"gpu": 0, "cpu": 0, "ssd": 0, "drop": 0}
        writes = {"cpu": 0, "ssd": 0}
        evict = {"free": 0, "to_cpu": 0, "to_ssd": 0, "swap_cpu": 0, "lost": 0,
                 "drop_by_choice": 0, "swap_ssd": 0}
        cap = self.cap

        def pop_victim():
            while heap:
                negk, b = heap[0]
                if b not in gpu or -negk != gpu[b]:
                    heapq.heappop(heap)
                    continue
                heapq.heappop(heap)
                return b
            return next(iter(gpu)) if gpu else None

        def worst(h, d):
            """堆頂（排序鍵最大＝預測最晚才用到）；lazy deletion。"""
            while h:
                negk, b = h[0]
                if b not in d or -negk != d[b]:
                    heapq.heappop(h)
                    continue
                return b
            return None

        def demote(b: int) -> None:
            key, p, dc = st[b]
            if cascade:
                if not use_cpu:
                    evict["lost"] += 1
                    st.pop(b, None)
                    return
                cpu_od[b] = None
                cpu_od.move_to_end(b)
                writes["cpu"] += 1
                evict["to_cpu"] += 1
                while len(cpu_od) > cap["cpu"]:
                    ev, _ = cpu_od.popitem(last=False)
                    if use_ssd:
                        ssd_od[ev] = None
                        writes["ssd"] += 1
                        evict["to_ssd"] += 1
                        while len(ssd_od) > cap["ssd"]:
                            lost, _ = ssd_od.popitem(last=False)
                            st.pop(lost, None)
                            evict["lost"] += 1
                    else:
                        st.pop(ev, None)
                        evict["lost"] += 1
                return
            # ---- 成本感知：門檻由式 (9) 給出，逐 block 依其絕對位置算 ----
            if use_cpu and p >= cm.cpu / dc:
                if len(cpu) < cap["cpu"]:
                    cpu[b] = key
                    heapq.heappush(cpu_h, (-key, b))
                    writes["cpu"] += 1
                    evict["to_cpu"] += 1
                    return
                far = worst(cpu_h, cpu)
                if far is not None and cpu[far] > key:
                    heapq.heappop(cpu_h)
                    cpu.pop(far, None)
                    cpu[b] = key
                    heapq.heappush(cpu_h, (-key, b))
                    writes["cpu"] += 1
                    evict["swap_cpu"] += 1
                    b = far                       # 被擠下來的往下一階試
                    key, p, dc = st[b]
            if use_ssd and p >= cm.ssd / dc:
                if len(ssd) < cap["ssd"]:
                    ssd[b] = key
                    heapq.heappush(ssd_h, (-key, b))
                    writes["ssd"] += 1
                    evict["to_ssd"] += 1
                    return
                far = worst(ssd_h, ssd)
                if far is not None and ssd[far] > key:
                    heapq.heappop(ssd_h)
                    ssd.pop(far, None)
                    st.pop(far, None)
                    ssd[b] = key
                    heapq.heappush(ssd_h, (-key, b))
                    writes["ssd"] += 1
                    evict["swap_ssd"] += 1
                    return
            evict["drop_by_choice" if (use_cpu or use_ssd) else "lost"] += 1
            st.pop(b, None)

        t = t_offset
        for ri, req in enumerate(trace):
            gap = self._gap_index(req, gpu, cpu_set, ssd_set, prefix_semantics)
            req_compute = req_transfer = 0.0
            n = len(req)
            # 前綴語意下，丟掉一個 block 的邊際成本是「它到請求結尾的整條尾巴」
            tails = [0.0] * n
            acc = 0.0
            for k in range(n - 1, -1, -1):
                acc += cm.cost("drop", k * BLOCK)
                tails[k] = acc
            for pi, blk in enumerate(req):
                pos = pi * BLOCK
                if pi > gap:
                    hits["drop"] += 1
                    req_compute += cm.cost("drop", pos)
                    cpu_set.pop(blk, None)
                    ssd_set.pop(blk, None)
                elif blk in gpu:
                    hits["gpu"] += 1
                    req_compute += cm.cost("gpu", pos)
                elif blk in cpu_set:
                    hits["cpu"] += 1
                    req_transfer += cm.cost("cpu", pos)
                    cpu_set.pop(blk, None)
                elif blk in ssd_set:
                    hits["ssd"] += 1
                    req_transfer += cm.cost("ssd", pos)
                    ssd_set.pop(blk, None)
                else:
                    hits["drop"] += 1
                    req_compute += cm.cost("drop", pos)
                if mode == "lru-shim":
                    key, p = -float(t), 1.0
                else:
                    key, p = float(next_use[t]), float(p_hat[t])
                gpu[blk] = key
                heapq.heappush(heap, (-key, blk))
                # 🔴 丟掉一個 block 的邊際成本怎麼算，是一個**決定結果的**選擇：
                #   tail  = 它到請求結尾的整條尾巴（與 run_oracle 的規則相同）
                #   block = 只算它自己那一塊
                # 前綴語意下，缺一塊會使其後全部重算，所以 tail 是**上界**——
                # 但那條尾巴是該請求所有缺塊**共用**的，把它整條算給每一個 block，
                # 等於把同一筆成本重複計價，門檻因而被壓得太低（什麼都想留）。
                # 這一項在 p̂ 呈雙峰時看不出來（W=1×），在 p̂ 有中間質量時
                # （W=35×）會把 block 大量推進 SSD 階，而 SSD 讀回比重算還貴。
                st[blk] = (key, p, tails[pi] if drop_cost == "tail"
                           else cm.cost("drop", pos))
                while len(gpu) > cap["gpu"]:
                    v = pop_victim()
                    if v is None:
                        break
                    gpu.pop(v, None)
                    demote(v)
                t += 1
            c_req, prev_compute = self._flush(req_compute, req_transfer,
                                              prev_compute, prefetch)
            d_req = self.decode_ms(len(req), outputs[ri]) if outputs else 0.0
            decode_total += d_req
            c_req += d_req
            total += c_req
            if per_request:
                per_req.append(c_req)
        out = {"total_ms": total, "hits": hits, "writes": writes,
               "evict": evict, "decode_ms": decode_total,
               "prefill_ms": total - decode_total,
               "warm_ms": 0.0, "warm_hits": hits}
        if per_request:
            out["per_request_ms"] = per_req
        return out


# ────────────────────────── 預測 ──────────────────────────

def score_all_accesses(model_path: Path, calib_path: Path, X, chunk=1_000_000):
    """對每一次存取算 (預測的下次使用時刻, 校準後機率)。

    一次批次推論算完整條 trace，模擬迴圈裡只查表。
    見檔頭〈評分時機與論文演算法 1 的差異〉。
    """
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(model_path))
    cal = np.load(calib_path)
    n = len(X)
    yhat = np.empty(n, dtype=np.float32)
    t0 = time.time()
    for i in range(0, n, chunk):
        yhat[i:i + chunk] = booster.predict(np.asarray(X[i:i + chunk]),
                                            num_threads=8)
    p = isotonic_predict(cal["xk"], cal["yk"], -yhat.astype(np.float64))
    tau = np.exp(np.clip(yhat, 0, 30)).astype(np.float64)
    nxt = np.arange(n, dtype=np.float64) + tau
    print(f"[評分] {n:,} 次存取，{time.time() - t0:.1f}s；"
          f"p̂ 中位數 {np.median(p):.4f}、>0.5 的比例 {100 * (p > 0.5).mean():.2f}%")
    return nxt, p


def true_signals(trace, window: int):
    """真實的「下次使用時刻」與「W 內會不會再被用到」。

    這**不是可部署的策略**，是診斷：把預測換成真值再跑一次，就能把
    「預測不夠準」與「策略本身不對」分開。少了這一步，
    「線上策略只拿到 10% headroom」這句話無法歸因——
    可能是預測器爛，也可能是就算預測完美這個策略也只值 10%。
    """
    n = sum(len(r) for r in trace)
    nxt = np.full(n, np.inf, dtype=np.float64)
    last: dict[int, int] = {}
    t = n - 1
    for req in reversed(trace):
        for b in reversed(req):
            nxt[t] = last.get(b, np.inf)
            last[b] = t
            t -= 1
    p = ((nxt - np.arange(n)) <= window).astype(np.float64)
    print(f"[真值] {n:,} 次存取，W 內會被再用到的比例 {100 * p.mean():.1f}%")
    return nxt, p


def segment_duration_s(name: str, r0: int, r1: int) -> float | None:
    p = Path(os.environ.get("PAPER_HKV_TRACES",
                            "/ssd7/hungwei/paper-hkv/datasets/traces")) / \
        f"{name}_trace.jsonl"
    if not p.exists():
        return None
    ts = [json.loads(l)["timestamp"] for l in p.open()][r0:r1]
    return (max(ts) - min(ts)) / 1000.0 if len(ts) > 1 else None


# ────────────────────────── 主流程 ──────────────────────────

def latest_run(pattern: str) -> Path | None:
    runs = sorted((BIG / "runs").glob(pattern))
    return runs[-1] if runs else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", default="toolagent")
    ap.add_argument("--model", default="qwen-awq", choices=list(MODEL_PROFILES))
    ap.add_argument("--device", default="nvme", choices=["sata", "nvme"])
    ap.add_argument("--cpu-gib", type=float, default=24.0)
    ap.add_argument("--ssd-gib", type=float, default=512.0)
    ap.add_argument("--gpu-tokens", type=int, default=None)
    ap.add_argument("--lookup", choices=["prefix", "per-block"], default="prefix")
    ap.add_argument("--prefetch", action="store_true", default=True)
    ap.add_argument("--no-prefetch", dest="prefetch", action="store_false")
    ap.add_argument("--decode", action="store_true", default=True,
                    help="含 decode（端到端）。--no-decode 改成只算 prefill")
    ap.add_argument("--no-decode", dest="decode", action="store_false")
    ap.add_argument("--segment", choices=["test", "full"], default="test")
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--train-run", default=None)
    ap.add_argument("--window-mult", type=float, default=1.0)
    ap.add_argument("--deltas", type=int, default=16)
    ap.add_argument("--losses", nargs="*", default=["sym_l2", "cost_l2"])
    ap.add_argument("--dests", nargs="*", default=["cost-aware"],
                    choices=["cost-aware", "cascade"],
                    help="cascade = 只用預測器決定逐出順序，目的地無條件往下推。"
                         "兩個一起跑就能把『預測器』與『成本模型』的貢獻分開")
    ap.add_argument("--drop-cost", choices=["tail", "block"], default="tail",
                    help="門檻裡「丟掉的成本」怎麼算，見 run_learned 的說明")
    ap.add_argument("--oracle-signal", nargs="*", default=[],
                    choices=["ordering", "both"],
                    help="診斷：把預測換成真值。ordering = 只有逐出順序用真值，"
                         "目的地仍用預測的 p̂；both = 兩者都用真值")
    ap.add_argument("--check-shim", action="store_true",
                    help="只跑等價性檢查：lru-shim 必須逐位元等於 Sim.run_online('lru')")
    ap.add_argument("--shim-requests", type=int, default=800)
    ap.add_argument("--out-dir", default=str(OUT))
    a = ap.parse_args()
    out_dir = Path(a.out_dir)

    prof = profile(a.model)
    cm = load_cost_model(a.device, require_model_key=prof["cost_model_key"])
    bpb = prof["kv_bytes_per_token"] * BLOCK
    gpu_blocks = (a.gpu_tokens or prof["gpu_kv_tokens"]) // BLOCK
    cpu_blocks = int(a.cpu_gib * 1024**3) // bpb
    ssd_blocks = int(a.ssd_gib * 1024**3) // bpb
    dev_write = DEVICE_WRITE_MIBPS[a.device]
    fs_root = DEVICE_FS_ROOT[a.device]
    decode = None
    if a.decode:
        decode = load_decode_model(prof["cost_model_key"])
        check_decode_bandwidth(decode, bpb)

    # ---- 等價性檢查（不需要模型） ----
    if a.check_shim:
        trace = mooncake_trace(a.trace, limit=a.shim_requests)
        sem = {"prefix_semantics": a.lookup == "prefix", "prefetch": a.prefetch}
        base = Sim(cm, gpu_blocks // 64, cpu_blocks // 64, ssd_blocks // 64)
        shim = PolicySim(cm, gpu_blocks // 64, cpu_blocks // 64, ssd_blocks // 64)
        ok = True
        for use_cpu, use_ssd, tag in ((False, False, "full_gpu"),
                                      (True, False, "cpu_lru"),
                                      (True, True, "tier_fs")):
            r1 = base.run_online(trace, "lru", use_cpu, use_ssd, **sem)
            r2 = shim.run_learned(trace, None, None, 0, use_cpu, use_ssd,
                                  mode="lru-shim", **sem)
            same = (abs(r1["total_ms"] - r2["total_ms"]) < 1e-9
                    and r1["hits"] == r2["hits"] and r1["writes"] == r2["writes"])
            ok &= same
            print(f"  {'✅' if same else '🔴'} {tag}：base {r1['total_ms']:,.4f} ms "
                  f"vs shim {r2['total_ms']:,.4f} ms　hits {r1['hits']} / {r2['hits']}"
                  f"　writes {r1['writes']} / {r2['writes']}")
        print("✅ 記帳一致，兩者可比" if ok else
              "🔴 記帳不一致：線上策略與 M4 的數字不可並列")
        return 0 if ok else 1

    # ---- 找特徵與模型 ----
    key = f"{a.trace}:{a.model}:w{a.window_mult}:k{a.deltas}"
    fdir = Path(a.features_dir) if a.features_dir else Path(load_index()[key])
    tdir = (Path(a.train_run) if a.train_run else
            latest_run(f"*-m5p-train-{a.trace}-{a.model}"))
    if tdir is None:
        raise SystemExit("🔴 找不到訓練好的模型，先跑 m5_predictor.py train")
    fm = json.loads((fdir / "features_meta.json").read_text())
    tm = json.loads((tdir / "train_meta.json").read_text())
    print(f"[來源] 特徵 {fdir.name}　模型 {tdir.name}")

    trace_all = mooncake_trace(a.trace)
    outs_all = mooncake_outputs(a.trace)
    # 🔴 單位檢查要對**整條** trace 做。拿切片去比整個檔案的長度中位數會誤報：
    #    conversation 的後 30% 中位數 6,328 vs 全檔 6,909（差 8%），
    #    那是「這一段的請求比較短」，不是 hash_id 粒度解錯。
    check_trace_units(a.trace, trace_all)
    X = np.load(fdir / "X.npy", mmap_mode="r")
    meta = np.load(fdir / "meta.npy", mmap_mode="r")
    if len(X) != sum(len(r) for r in trace_all):
        raise SystemExit("🔴 特徵矩陣的列數與 trace 的存取數對不上——"
                         "特徵是用不同的 --limit-requests 產生的，不可混用")

    # ---- 切出評估段（模型沒看過的那 30%） ----
    if a.segment == "test":
        t_split = int(tm["split"]["t_split"])
        r0 = int(meta[t_split, 2]) + 1
    else:
        r0 = 0
    t_off = sum(len(r) for r in trace_all[:r0])
    trace = trace_all[r0:]
    outs = outs_all[r0:] if decode else None
    dur = segment_duration_s(a.trace, r0, len(trace_all))
    print(f"[評估段] {a.segment}：請求 {r0:,}–{len(trace_all):,}"
          f"（{len(trace):,} 筆、{sum(len(r) for r in trace):,} 次存取、"
          f"{dur:,.0f}s）")

    sem = {"prefix_semantics": a.lookup == "prefix", "prefetch": a.prefetch}
    preflight(cm, trace, None, gpu_blocks, cpu_blocks, ssd_blocks, bpb, fs_root)
    sim = Sim(cm, gpu_blocks, cpu_blocks, ssd_blocks, decode=decode)
    psim = PolicySim(cm, gpu_blocks, cpu_blocks, ssd_blocks, decode=decode)
    kw = dict(sem)
    if decode:
        kw["outputs"] = outs

    res = {}
    for name, (pol, uc, us) in POLICIES.items():
        t0 = time.time()
        res[name] = sim.run_online(trace, pol, uc, us, **kw)
        print(f"  {name:<10} {res[name]['total_ms']:>16,.0f} ms"
              f"　({time.time() - t0:.0f}s)")
    for loss in a.losses:
        nxt, p = score_all_accesses(tdir / f"model_{loss}.txt",
                                    tdir / f"calib_{loss}.npz", X)
        for dest in a.dests:
            name = (f"tiara_{loss}" + ("" if dest == "cost-aware" else "_cascade")
                    + ("" if a.drop_cost == "tail" else "_blockcost"))
            t0 = time.time()
            res[name] = psim.run_learned(trace, nxt, p, t_off, True, True,
                                         dest=dest, drop_cost=a.drop_cost, **kw)
            print(f"  {name:<18} {res[name]['total_ms']:>16,.0f} ms"
                  f"　({time.time() - t0:.0f}s)")
        del nxt, p
    if a.oracle_signal:
        tn, tp = true_signals(trace_all, fm["window_accesses"])
        loss0 = a.losses[0]
        _, phat = score_all_accesses(tdir / f"model_{loss0}.txt",
                                     tdir / f"calib_{loss0}.npz", X)
        for kind in a.oracle_signal:
            t0 = time.time()
            name = f"diag_true_{kind}"
            res[name] = psim.run_learned(trace, tn, tp if kind == "both" else phat,
                                         t_off, True, True, dest="cost-aware",
                                         drop_cost=a.drop_cost, **kw)
            print(f"  {name:<18} {res[name]['total_ms']:>16,.0f} ms"
                  f"　({time.time() - t0:.0f}s)")
        del tn, tp, phat
    t0 = time.time()
    res["oracle"] = sim.run_oracle(trace, True, True, dest="best", **kw)
    print(f"  {'oracle':<10} {res['oracle']['total_ms']:>16,.0f} ms"
          f"　({time.time() - t0:.0f}s)")

    baselines = [k for k in res if k not in ("oracle",)
                 and not k.startswith(("tiara", "diag_"))]
    best = min(baselines, key=lambda k: res[k]["total_ms"])
    # 🔴 `diag_*` 不參與「oracle 必須支配一切」這條檢查，理由要寫清楚，
    #    因為放寬一條安全檢查是很容易自欺的事：
    #
    #    那條檢查的用意是「**線上**策略不可能贏過知道未來的離線解，贏了就是記帳有錯」。
    #    `diag_*` 不是線上策略——它吃的是從整條 trace 算出來的真實下次使用時刻，
    #    也就是**另一個離線解**。而 `m4_oracle.run_oracle` 的 docstring 自己寫著
    #    它是**貪婪**構造、「不保證全域最優」、「是下界的 Oracle」。
    #    兩個離線解互相比較，輸贏都不是錯誤，而是「哪一個構造比較好」的量測。
    #
    #    所以：tiara_*（線上）仍然要通過這條檢查；diag_*（離線）改成明確比較並回報。
    check_results({k: v for k, v in res.items() if not k.startswith("diag_")},
                  trace, best)
    b_ms, o_ms = res[best]["total_ms"], res["oracle"]["total_ms"]
    head = 100 * (b_ms - o_ms) / b_ms
    print(f"[headroom] 最佳 baseline = {best}；oracle 比它好 {head:.2f}%")
    for name in [k for k in res if k.startswith("diag_")]:
        d = res[name]["total_ms"]
        if d < o_ms - 1e-6:
            print(f"🟡 {name}（完美資訊 + 本檔的門檻式目的地規則）比 M4 的貪婪 oracle "
                  f"再好 {100 * (o_ms - d) / o_ms:.2f}%"
                  f"　→ M4 回報的 headroom {head:.2f}% 是**下界**，"
                  f"真值至少 {100 * (b_ms - d) / b_ms:.2f}%")

    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-m5p-policy-{a.trace}-{a.model}"
    rows = []
    for name, r in res.items():
        w = r["writes"]["ssd"] * bpb / 1024**2 / dur if dur else None
        rows.append({
            "run_id": run_id, "ts": datetime.now().astimezone().isoformat(),
            "sim_version": SIM_VERSION, "trace": a.trace,
            "model_profile": a.model, "device": a.device, "segment": a.segment,
            "first_request": r0, "requests": len(trace),
            "accesses": sum(len(x) for x in trace),
            "gpu_blocks": gpu_blocks, "cpu_gib": a.cpu_gib, "ssd_gib": a.ssd_gib,
            "lookup": a.lookup, "prefetch": int(a.prefetch),
            "drop_cost_rule": a.drop_cost,
            "decode": int(bool(decode)), "window_accesses": fm["window_accesses"],
            "features_run": fdir.name, "train_run": tdir.name,
            "policy": name, "total_ms": round(r["total_ms"], 2),
            "prefill_ms": round(r["prefill_ms"], 2),
            "decode_ms": round(r["decode_ms"], 2),
            "gpu_hits": r["hits"]["gpu"], "cpu_hits": r["hits"]["cpu"],
            "ssd_hits": r["hits"]["ssd"], "recompute": r["hits"]["drop"],
            "cpu_writes": r["writes"]["cpu"], "ssd_writes": r["writes"]["ssd"],
            "ssd_write_mibps": round(w, 1) if w is not None else "",
            "write_feasible": ("" if w is None else int(w <= dev_write)),
            "device_write_mibps": dev_write,
            "best_baseline": best,
            "vs_best_baseline_pct": round(100 * (b_ms - r["total_ms"]) / b_ms, 3),
            "oracle_headroom_pct": round(head, 3),
            "headroom_captured_pct": ("" if name in ("oracle",) or b_ms == o_ms
                                      else round(100 * (b_ms - r["total_ms"])
                                                 / (b_ms - o_ms), 2)),
        })
    write_csv(out_dir / "policy_sim.csv", rows)
    print(f"[輸出] {out_dir / 'policy_sim.csv'}")
    for r in rows:
        if r["policy"].startswith("tiara"):
            print(f"  ▶ {r['policy']}：比最佳 baseline 好 "
                  f"{r['vs_best_baseline_pct']:.2f}%，"
                  f"拿到 oracle headroom 的 {r['headroom_captured_pct']}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
