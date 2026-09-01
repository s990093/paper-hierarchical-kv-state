#!/usr/bin/env python3
"""Milestone 4 的掃描驅動程式。**一支涵蓋四個自變數軸。**

先前這是四支各自獨立的腳本（`m4_ssd_sweep` / `m4_budget_sweep` /
`m4_by_length` / `m4_semantics_ablation`），彼此有約七成的重複程式碼，
而且參數預設值曾經不一致——2026-08-31 的模型混用錯誤就是這樣來的：
一支腳本預設 `qwen-awq` 的預算，另一支預設 `llama-bf16` 的成本常數。

四個軸：

| `--axis` | 自變數 | 回答的問題 |
|---|---|---|
| `ssd` | SSD 階容量 | 磁碟階要多大才有價值？寫得下去嗎？ |
| `budget` | GPU KV 預算 | 記憶體壓力多大時階層才開始有用？ |
| `length` | 請求長度（分箱） | 節省集中在長請求嗎？（回答長上下文的價值） |
| `semantics` | 模擬器的系統語意假設 | 前綴語意與預取各自把結果推動多少？ |
| `prefix` | （診斷） | 前綴語意為什麼幾乎不影響結果？ |

用法：
    python code/m4_sweep.py --axis ssd
    python code/m4_sweep.py --axis all
"""
from __future__ import annotations
import argparse
import csv
import math
import shutil
from datetime import datetime
from pathlib import Path

from m4_invariants import check_results, preflight
from m4_oracle import (BLOCK, DEVICE_FS_ROOT, DEVICE_WRITE_MIBPS,
                       check_decode_bandwidth, load_decode_model,
                       mooncake_outputs,
                       MODEL_PROFILES, OUT, SIM_VERSION, Sim, load_cost_model,
                       longctx_trace, mooncake_trace, profile, reuse_rate,
                       trace_duration_s, zipf_trace)

POLICIES = {
    "full_gpu": ("lru", False, False),
    "cpu_lru": ("lru", True, False),
    "cpu_arc": ("arc", True, False),
    "tier_fs": ("lru", True, True),
}
LENGTH_BINS = [0, 4096, 8192, 16384, 32768, 65536, 131072, 10**9]
SEMANTICS = [
    ("per-block/no-prefetch", False, False),   # 修正前的模型
    ("prefix/no-prefetch", True, False),       # 只修 lookup
    ("per-block/prefetch", False, True),       # 只修預取
    ("prefix/prefetch", True, True),           # 兩個都修（最接近 vLLM）
]


# ───────────────────────── 共用 ─────────────────────────

class Ctx:
    """一次掃描的全部設定。把「必須一起切換」的參數綁在一起。"""

    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.prof = profile(a.model)
        self.cm = load_cost_model(a.device,
                                  require_model_key=self.prof["cost_model_key"])
        self.bpb = self.prof["kv_bytes_per_token"] * BLOCK   # 每 block 幾個 byte
        # --gpu-tokens 覆寫：讓「壓力固定、長度變化」的掃描成為可能。
        # 🔴 覆寫成剖面實測值以外的數字時，那個設定**在這張卡上不可部署**，
        #    必須在結果裡標明（欄位 gpu_tokens_override）。
        #    它回答的是「若預算跟著工作負載一起變大（換卡或降 KV 精度），
        #    長度本身是好是壞」——與「同一張卡塞更大的東西」是不同的問題。
        self.gpu_tokens = a.gpu_tokens or self.prof["gpu_kv_tokens"]
        self.gpu_tokens_override = bool(a.gpu_tokens)
        self.gpu_blocks = self.gpu_tokens // BLOCK
        self.cpu_blocks = int(a.cpu_gib * 1024**3) // self.bpb
        self.dev_write = (a.device_write_mibps
                          if a.device_write_mibps is not None
                          else DEVICE_WRITE_MIBPS[a.device])
        self.fs_root = a.fs_root or DEVICE_FS_ROOT[a.device]
        self.sem = {"prefix_semantics": a.lookup == "prefix",
                    "prefetch": a.prefetch}
        # 🔴 decode。放置決策只能優化 prefill；decode 期間該請求的 KV 必須
        #    整份在 GPU 裡，沒有自由度。不模它，回報的 headroom 會是端到端
        #    值的約兩倍（真實負載裡 decode 佔 48.8%–53.8% 的時間）。
        self.decode = None
        if a.decode:
            dm = load_decode_model(self.prof["cost_model_key"])
            check_decode_bandwidth(dm, self.bpb)
            self.decode = dm
            print(f"[decode] 每步 = {dm['decode_base_ms']:.3f} ms + "
                  f"{dm['decode_ms_per_block']:.6f} × blocks"
                  f"（R²={dm['r2']:.4f}，擬合自 {dm['n_points']} 個 ctx）")
        du = shutil.disk_usage(self.fs_root)
        print(f"[剖面] {a.model}：GPU {self.gpu_tokens:,} token = "
              f"{self.gpu_blocks:,} blocks；每 block "
              f"{self.bpb / 1024**2:.1f} MiB"
              + ("　⚠️ 已覆寫（剖面實測值 "
                 f"{self.prof['gpu_kv_tokens']:,}），此設定不可部署"
                 if self.gpu_tokens_override else ""))
        print(f"[裝置] {a.device}：成本常數與持續寫入上限 "
              f"{self.dev_write:,.0f} MiB/s 同時來自這顆碟")
        print(f"[實體] {self.fs_root}：裝置 {du.total / 1024**4:.1f} TiB、"
              f"可用 {du.free / 1024**3:.0f} GiB")
        print(f"[語意] lookup={a.lookup}　prefetch={a.prefetch}　"
              f"oracle-dest={a.oracle_dest}　sim={SIM_VERSION}")
        self.device_total_tib = du.total / 1024**4
        self.device_free_gib = du.free / 1024**3

    def run(self, trace, tname, ssd_blocks, gpu_blocks=None, per_request=False,
            outputs=None):
        """跑一輪五個策略，回傳 (res, best, headroom)。"""
        gb = gpu_blocks or self.gpu_blocks
        preflight(self.cm, trace, tname, gb, self.cpu_blocks, ssd_blocks,
                  self.bpb, self.fs_root)
        sim = Sim(self.cm, gb, self.cpu_blocks, ssd_blocks=ssd_blocks,
                  decode=self.decode)
        kw = dict(self.sem, per_request=per_request)
        if self.decode is not None:
            # 🔴 合成 trace 沒有 output_length 欄，若不明確給就等於沒有 decode，
            #    整張含 decode 的地圖會跟只算 prefill 的一模一樣。
            #    2026-08-31 實測踩到：B 半段輸出與 A 完全相同。
            if outputs is not None:
                kw["outputs"] = outputs
            elif tname:
                kw["outputs"] = mooncake_outputs(tname)
            else:
                raise SystemExit(
                    "🔴 開了 --decode 但這個工作負載沒有輸出長度。"
                    "合成 trace 必須明確傳入 outputs，否則 decode 成本會是 0，"
                    "而結果看起來完全正常。")
        res = {k: sim.run_online(trace, *v, **kw)
               for k, v in POLICIES.items() if not (v[2] and ssd_blocks == 0)}
        res["oracle"] = sim.run_oracle(trace, True, ssd_blocks > 0,
                                       dest=self.a.oracle_dest, **kw)
        best = min((k for k in res if k != "oracle"),
                   key=lambda k: res[k]["total_ms"])
        check_results(res, trace, best)
        head = 100 * (res[best]["total_ms"] - res["oracle"]["total_ms"]) \
            / res[best]["total_ms"]
        return res, best, head

    def base_row(self, tname: str) -> dict:
        return {
            "ts": datetime.now().astimezone().isoformat(),
            "sim_version": SIM_VERSION, "trace": tname,
            "model_profile": self.a.model, "device": self.a.device,
            "gpu_budget_tokens": self.gpu_tokens,
            "gpu_tokens_override": int(self.gpu_tokens_override),
            "profile_gpu_tokens": self.prof["gpu_kv_tokens"],
            "cpu_budget_gib": self.a.cpu_gib,
            "lookup": self.a.lookup, "prefetch": int(self.a.prefetch),
            "oracle_dest": self.a.oracle_dest,
            "device_write_mibps_sustained": self.dev_write,
            "fs_root": self.fs_root,
            "cost_model": str(OUT / "cost_model.json"),
        }


def verdict(h: float) -> str:
    return "GO" if h > 15 else "MARGINAL" if h >= 5 else "NO_GO"


def policy_rows(ctx: Ctx, res: dict, best: str, head: float,
                tname: str, extra: dict) -> list[dict]:
    out = []
    for pol, v in res.items():
        e = v.get("evict", {})
        w = v.get("writes", {})
        out.append({**ctx.base_row(tname), **extra, "policy": pol,
                    "total_ms": round(v["total_ms"], 2),
                    "gpu_hits": v["hits"]["gpu"], "cpu_hits": v["hits"]["cpu"],
                    "ssd_hits": v["hits"]["ssd"], "recompute": v["hits"]["drop"],
                    "ssd_writes": w.get("ssd", ""), "cpu_writes": w.get("cpu", ""),
                    "decode_ms": round(v.get("decode_ms", 0), 2),
                    "prefill_ms": round(v.get("prefill_ms", v["total_ms"]), 2),
                    "evict_free": e.get("free", ""),
                    "evict_to_cpu": e.get("to_cpu", ""),
                    "evict_to_ssd": e.get("to_ssd", ""),
                    "evict_free_pct": round(
                        100 * e["free"] / max(1, sum(e.values())), 3)
                    if e else "",
                    "best_baseline": best,
                    "oracle_headroom_pct": round(head, 3) if pol == "oracle" else "",
                    "verdict": verdict(head) if pol == "oracle" else ""})
    return out


def write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}  ({len(rows)} rows)")


# ───────────────────────── 各軸 ─────────────────────────

def axis_ssd(ctx: Ctx) -> list[dict]:
    """SSD 階容量 vs headroom vs 寫入可行性。"""
    a, rows = ctx.a, []
    for tname in a.trace:
        trace = mooncake_trace(tname, a.trace_limit)
        dur = trace_duration_s(tname)
        uniq = len({b for r in trace for b in r})
        need = uniq * ctx.bpb / 1024**4
        print(f"\n{'=' * 104}\ntrace「{tname}」：{len(trace):,} 請求、"
              f"{sum(len(r) for r in trace):,} 次存取、{uniq:,} 不重複 block")
        print(f"  工作集全放磁碟需要 {need:.1f} TiB"
              f"（裝置 {ctx.device_total_tib:.1f} TiB）；"
              f"trace 時長 {dur / 60:.1f} 分鐘")
        print(f"{'SSD 容量':>11s}{'覆蓋':>7s}{'best':>9s}{'headroom':>10s}"
              f"{'判定':>9s}{'best 寫':>12s}{'頻寬':>11s}{'可行?':>7s}"
              f"{'oracle 寫':>12s}{'頻寬':>11s}")
        for g in a.ssd_gib:
            sb = 10**9 if g < 0 else int(g * 1024**3) // ctx.bpb
            res, best, head = ctx.run(trace, tname, sb)

            def bw(p_):
                w = res.get(p_, {}).get("writes", {}).get("ssd", 0)
                return w * ctx.bpb / 1024**2 / dur if dur else float("nan")
            feas = "✅" if bw(best) <= ctx.dev_write else "🔴"
            label = "無限" if g < 0 else f"{g:,.0f} GiB"
            print(f"{label:>11s}{100 * min(1, sb / uniq):>6.1f}%{best:>9s}"
                  f"{head:>9.2f}%{verdict(head):>9s}"
                  f"{res[best]['writes']['ssd']:>12,}{bw(best):>8,.0f}MiB/s"
                  f"{feas:>6s}{res['oracle']['writes']['ssd']:>12,}"
                  f"{bw('oracle'):>8,.0f}MiB/s")
            rows += policy_rows(ctx, res, best, head, tname, {
                "axis": "ssd",
                "ssd_gib": g if g >= 0 else "unlimited", "ssd_blocks": sb,
                "unique_blocks": uniq, "requests": len(trace),
                "accesses": sum(len(r) for r in trace),
                "ssd_covers_working_set_pct": round(100 * min(1, sb / uniq), 2),
                "working_set_tib": round(need, 2),
                "device_total_tib": round(ctx.device_total_tib, 2),
                "trace_duration_s": round(dur, 1) if dur else "",
            })
            for r in rows[-len(res):]:
                w = r["ssd_writes"]
                r["ssd_write_mibps"] = (round(w * ctx.bpb / 1024**2 / dur, 1)
                                        if dur and w != "" else "")
    return rows


def axis_budget(ctx: Ctx) -> list[dict]:
    """GPU 預算 vs headroom。免費逐出的存量是機制。"""
    a, rows = ctx.a, []
    full = ctx.prof["gpu_kv_tokens"]
    budgets = a.budgets or [full // (2 ** k) for k in range(5)]
    sb = int(a.ssd_gib_fixed * 1024**3) // ctx.bpb
    for tname in a.trace:
        trace = mooncake_trace(tname, a.trace_limit)
        uniq = len({b for r in trace for b in r})
        print(f"\n{'=' * 96}\ntrace「{tname}」（SSD 階固定 "
              f"{a.ssd_gib_fixed:,.0f} GiB）")
        print(f"{'GPU 預算':>12s}{'blocks':>9s}{'壓力':>9s}{'免費逐出':>10s}"
              f"{'oracle CPU 命中':>16s}{'best':>10s}{'headroom':>10s}{'判定':>9s}")
        for bt in budgets:
            gb = bt // BLOCK
            res, best, head = ctx.run(trace, tname, sb, gpu_blocks=gb)
            e = res["oracle"]["evict"]
            fp = 100 * e["free"] / max(1, sum(e.values()))
            print(f"{bt:>12,}{gb:>9,}{uniq / gb:>8.0f}×{fp:>9.1f}%"
                  f"{res['oracle']['hits']['cpu']:>16,}{best:>10s}"
                  f"{head:>9.2f}%{verdict(head):>9s}")
            rows += policy_rows(ctx, res, best, head, tname, {
                "axis": "budget", "gpu_budget_tokens": bt, "gpu_blocks": gb,
                "ssd_gib": a.ssd_gib_fixed, "ssd_blocks": sb,
                "unique_blocks": uniq, "requests": len(trace),
                "pressure_x": round(uniq / gb, 2),
            })
    return rows


def axis_length(ctx: Ctx) -> list[dict]:
    """節省按請求長度分箱。真實資料，零假設。"""
    a, rows = ctx.a, []
    sb = int(a.ssd_gib_fixed * 1024**3) // ctx.bpb
    for tname in a.trace:
        trace = mooncake_trace(tname, a.trace_limit)
        res, best, head = ctx.run(trace, tname, sb, per_request=True)
        bl, ol = res[best]["per_request_ms"], res["oracle"]["per_request_ms"]
        lens = [len(r) * BLOCK for r in trace]
        tot_ms, tot_save = sum(bl), sum(bl) - sum(ol)
        print(f"\n{'=' * 96}\ntrace「{tname}」，最佳 baseline = {best}"
              f"（整體 {head:.2f}%）")
        print(f"{'請求長度':>12s}{'筆數':>8s}{'佔時間':>9s}{'節省':>9s}"
              f"{'佔總節省':>10s}")
        for lo, hi in zip(LENGTH_BINS[:-1], LENGTH_BINS[1:]):
            idx = [i for i, x in enumerate(lens) if lo <= x < hi]
            if not idx:
                continue
            b = sum(bl[i] for i in idx)
            o = sum(ol[i] for i in idx)
            f = lambda x: "∞" if x >= 10**9 else (f"{x // 1024}K" if x >= 1024
                                                  else str(x))
            lab = f"{f(lo)}–{f(hi)}"
            print(f"{lab:>12s}{len(idx):>8,}{100 * b / tot_ms:>8.1f}%"
                  f"{100 * (b - o) / b:>8.2f}%"
                  f"{100 * (b - o) / tot_save:>9.1f}%")
            rows.append({**ctx.base_row(tname), "axis": "length",
                         "bin": lab, "bin_lo_tokens": lo, "bin_hi_tokens": hi,
                         "requests": len(idx),
                         "share_of_total_time_pct": round(100 * b / tot_ms, 3),
                         "baseline_ms": round(b, 2), "oracle_ms": round(o, 2),
                         "best_baseline": best,
                         "saving_pct_within_bin": round(100 * (b - o) / b, 3),
                         "share_of_total_saving_pct":
                             round(100 * (b - o) / tot_save, 2),
                         "overall_headroom_pct": round(head, 3),
                         "ssd_gib": a.ssd_gib_fixed})
    return rows


def axis_semantics(ctx: Ctx) -> list[dict]:
    """模擬器的系統語意假設各自把 headroom 推動多少。"""
    a, rows = ctx.a, []
    sb = int(a.ssd_gib_fixed * 1024**3) // ctx.bpb
    cases: list[tuple[str, list]] = [(f"trace:{t}", mooncake_trace(t, a.trace_limit))
                                     for t in a.trace]
    for r_ in a.pressure:
        n_docs = max(2, round(r_ * ctx.gpu_blocks / (a.doc_tokens // BLOCK)))
        tr = zipf_trace(n_docs, a.doc_tokens // BLOCK,
                        max(400, 10 * n_docs), 0.9, a.seed)
        real = len({b for q in tr for b in q}) / ctx.gpu_blocks
        cases.append((f"pressure:{real:.1f}x(nom {r_:g}x)", tr))
    for label, trace in cases:
        print(f"\n{'=' * 84}\n{label}")
        print(f"{'語意':24s}{'best':>10s}{'headroom':>10s}{'差':>9s}")
        base = None
        for name, pfx, pf in SEMANTICS:
            saved = ctx.sem
            ctx.sem = {"prefix_semantics": pfx, "prefetch": pf}
            try:
                res, best, head = ctx.run(trace, None, sb)
            finally:
                ctx.sem = saved
            base = head if base is None else base
            print(f"{name:24s}{best:>10s}{head:>9.2f}%"
                  f"{head - base:>+8.2f}")
            rows += policy_rows(ctx, res, best, head, label, {
                "axis": "semantics", "semantics": name,
                "lookup": "prefix" if pfx else "per-block",
                "prefetch": int(pf), "ssd_gib": a.ssd_gib_fixed,
                "unique_blocks": len({b for r_ in trace for b in r_}),
                "requests": len(trace),
                "delta_vs_baseline_pp": round(head - base, 3),
            })
    return rows


def axis_prefix(ctx: Ctx) -> list[dict]:
    """診斷：前綴語意為什麼幾乎不影響結果。"""
    a, rows = ctx.a, []
    sb = int(a.ssd_gib_fixed * 1024**3) // ctx.bpb
    print(f"{'trace/策略':26s}{'缺口後':>12s}{'仍在某階':>10s}{'佔比':>9s}"
          f"{'首次出現':>10s}")
    for tname in a.trace:
        trace = mooncake_trace(tname, a.trace_limit)
        for pol, args in (("full_gpu", POLICIES["full_gpu"]),
                          ("cpu_lru", POLICIES["cpu_lru"]),
                          ("tier_fs", POLICIES["tier_fs"])):
            orig = Sim._gap_index
            st = {"post": 0, "res": 0, "first": 0}
            seen: set[int] = set()

            def spy(req, gpu, cpu, ssd, enabled, _o=orig, _st=st, _s=seen):
                g = _o(req, gpu, cpu, ssd, True)
                for b in req[g + 1:]:
                    _st["post"] += 1
                    if b in gpu or b in cpu or b in ssd:
                        _st["res"] += 1
                    if b not in _s:
                        _st["first"] += 1
                _s.update(req)
                return _o(req, gpu, cpu, ssd, enabled)

            Sim._gap_index = staticmethod(spy)
            try:
                Sim(ctx.cm, ctx.gpu_blocks, ctx.cpu_blocks, sb).run_online(
                    trace, *args, prefix_semantics=True, prefetch=True)
            finally:
                Sim._gap_index = orig
            n = st["post"]
            print(f"{tname + '/' + pol:26s}{n:>12,}{st['res']:>10,}"
                  f"{100 * st['res'] / n:>8.3f}%{100 * st['first'] / n:>9.1f}%")
            rows.append({**ctx.base_row(tname), "axis": "prefix",
                         "policy": pol, "post_gap_blocks": n,
                         "post_gap_still_resident": st["res"],
                         "post_gap_resident_pct": round(100 * st["res"] / n, 4),
                         "post_gap_first_ever": st["first"],
                         "post_gap_first_ever_pct": round(100 * st["first"] / n, 2),
                         "ssd_gib": a.ssd_gib_fixed})
    return rows


def axis_surface(ctx: Ctx) -> list[dict]:
    """**headroom 的地圖**：掃（請求長度 × 重用率），找方法適用的區間。

    ## 為什麼要這張圖

    今天量到的都是單點，而且都在 Mooncake 上（中位 6.3K、重用 37–57%）。
    論文的目標是 512K，那裡沒有任何量測。與其挑一個好看的設定，
    不如把整個空間掃出來，**並把真實資料的位置標在圖上**。

    ## 兩個自變數

    * **長度**：決定「放錯地方的代價」。重算成本隨絕對位置線性成長，
      所以 6K 時重算只比 CPU 貴 9.8 倍，512K 時貴 210 倍。
    * **重用率**：決定「天花板」。低重用 -> 強制未命中多 -> 誰都省不掉。
      由「每份文件被查幾次」控制（reuse = 1 − 1/次數，再扣掉尾巴）。

    ## 真實資料的位置（會標在輸出裡）

        Mooncake toolagent      6,346 token　重用 57.0%
        Mooncake conversation   6,909 token　重用 37.3%
        SCBench qa_eng        745,586 token　重用 80.0%

    ## ⚠️ 兩個必須跟著數字走的標記

    1. `synthetic=longctx`——這是合成流量，不是真實 trace。
    2. 單一請求的 KV 必須整份放得進 GPU（vLLM 的啟動檢查）。
       請求長度超過 GPU 預算的點標為 `single_request_fits=False`，
       那是**不可部署**的設定，只能當上界參考。
    """
    a, rows = ctx.a, []
    lengths = [int(x) for x in a.surface_lengths]
    reqs_per_doc = [float(x) for x in a.surface_requeries]
    gpu_tok = ctx.prof["gpu_kv_tokens"]
    print(f"\n{'=' * 100}")
    print(f"headroom 地圖：長度 × 重用率（GPU 預算 {gpu_tok:,} token）")
    print(f"真實資料的位置：Mooncake 6.3K/57%、6.9K/37%　SCBench 746K/80%")
    print(f"{'請求長度':>10s}{'每份被查':>10s}{'重用率':>9s}{'壓力':>9s}"
          f"{'單請求塞得下?':>14s}{'best':>10s}{'headroom':>10s}{'判定':>9s}")
    sb = int(a.ssd_gib_fixed * 1024**3) // ctx.bpb
    for L in lengths:
        doc_b = max(1, L // BLOCK)
        for rq in reqs_per_doc:
            n_req = a.surface_requests
            n_docs = max(1, round(n_req / rq))
            tail_b = max(1, int(doc_b * a.surface_tail_frac))
            tr = longctx_trace(n_docs, doc_b, tail_b, n_req, 0.9, a.seed)
            ru = reuse_rate(tr)
            # 合成請求的輸出長度：從真實 Mooncake 的分佈抽樣，
            # 這樣「decode 佔多少」不是我編的
            outs = None
            if ctx.decode is not None:
                import random as _r
                pool = mooncake_outputs(a.surface_output_from)
                rr_ = _r.Random(a.seed)
                outs = [pool[rr_.randrange(len(pool))] for _ in range(len(tr))]
            uniq = len({b for r in tr for b in r})
            fits = L <= gpu_tok
            res, best, head = ctx.run(tr, None, sb, outputs=outs)
            print(f"{L:>10,}{rq:>10.1f}{100 * ru:>8.1f}%"
                  f"{uniq / ctx.gpu_blocks:>8.0f}×{'✅' if fits else '🔴':>12s}"
                  f"{best:>10s}{head:>9.2f}%{verdict(head):>9s}")
            rows += policy_rows(ctx, res, best, head,
                                f"surface:L{L}:rq{rq:g}", {
                "axis": "surface", "synthetic": "longctx",
                "request_tokens": L, "requeries_per_doc": rq,
                "reuse_pct": round(100 * ru, 2),
                "tail_frac": a.surface_tail_frac,
                "n_docs": n_docs, "requests": n_req,
                "unique_blocks": uniq,
                "pressure_x": round(uniq / ctx.gpu_blocks, 2),
                "single_request_fits": int(fits),
                "ssd_gib": a.ssd_gib_fixed,
            })
    return rows


AXES = {"ssd": (axis_ssd, "ssd_sweep.csv"),
        "budget": (axis_budget, "budget_sweep.csv"),
        "length": (axis_length, "by_length.csv"),
        "semantics": (axis_semantics, "semantics_ablation.csv"),
        "prefix": (axis_prefix, "prefix_gap_probe.csv"),
        "surface": (axis_surface, "headroom_surface.csv")}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axis", nargs="+", default=["ssd"],
                    choices=list(AXES) + ["all"])
    ap.add_argument("--model", default="llama-bf16", choices=list(MODEL_PROFILES),
                    help="模型剖面：一次鎖定 GPU 預算、KV 每 token 位元組、"
                         "成本常數來源三者")
    ap.add_argument("--device", default="nvme", choices=["sata", "nvme"],
                    help="磁碟階：成本常數、持續寫入上限、掛載點三者一起切換")
    ap.add_argument("--trace", nargs="*", default=["toolagent", "conversation"])
    ap.add_argument("--ssd-gib", type=float, nargs="*",
                    default=[0, 32, 128, 512, 2048, -1],
                    help="axis=ssd 用。-1 = 無限（物理上不可能，只作上界參考）")
    ap.add_argument("--ssd-gib-fixed", type=float, default=512.0,
                    help="其餘各軸固定的 SSD 階容量")
    ap.add_argument("--budgets", type=int, nargs="*", default=None)
    ap.add_argument("--pressure", type=float, nargs="*", default=[1, 2, 4, 8])
    ap.add_argument("--doc-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cpu-gib", type=float, default=24.0)
    ap.add_argument("--lookup", choices=["prefix", "per-block"], default="prefix")
    ap.add_argument("--prefetch", action="store_true", default=True)
    ap.add_argument("--no-prefetch", dest="prefetch", action="store_false")
    ap.add_argument("--gpu-tokens", type=int, default=None,
                    help="覆寫剖面的 GPU KV 預算。用於「壓力固定、長度變化」"
                         "的掃描——回答「若預算跟著工作負載變大（換卡或降 KV "
                         "精度），長度本身是好是壞」。⚠️ 覆寫的設定在這張卡上"
                         "不可部署，結果會標記 gpu_tokens_override=1")
    ap.add_argument("--surface-lengths", nargs="*",
                    default=[8192, 32768, 131072, 262144, 524288],
                    help="axis=surface 的請求長度（token）")
    ap.add_argument("--surface-requeries", nargs="*",
                    default=[1.2, 2, 4, 10, 25],
                    help="每份文件被查幾次。決定重用率："
                         "1.2 次≈55%、2 次≈66%、25 次≈95%")
    ap.add_argument("--surface-requests", type=int, default=200)
    ap.add_argument("--surface-output-from", default="conversation",
                    choices=["toolagent", "conversation"],
                    help="合成請求的輸出長度從哪條真實 trace 抽樣")
    ap.add_argument("--surface-tail-frac", type=float, default=0.02,
                    help="每個請求各自不同的尾巴佔多少（2% = 512K 裡的 10K 問題）")
    ap.add_argument("--decode", action="store_true",
                    help="把 decode 的成本也算進去（用 trace 裡真實的 "
                         "output_length）。放置只能優化 prefill，所以開了之後 "
                         "headroom 會降到端到端的真值")
    ap.add_argument("--oracle-dest", default="best",
                    choices=["best", "cost-aware", "cascade"])
    ap.add_argument("--device-write-mibps", type=float, default=None)
    ap.add_argument("--fs-root", default=None)
    ap.add_argument("--trace-limit", type=int, default=None,
                    help="只取前 N 個請求。**僅供煙霧測試**——截短會改變"
                         "工作集與預算的比例，也就是改變了問題本身"
                         "（2026-08-31 實測：2,000 筆給 11.0%，整條給 18.7%）")
    ap.add_argument("--out-dir", default=str(OUT))
    a = ap.parse_args()

    axes = list(AXES) if "all" in a.axis else a.axis
    ctx = Ctx(a)
    for ax in axes:
        fn, name = AXES[ax]
        print(f"\n{'#' * 84}\n# axis = {ax}\n{'#' * 84}")
        rows = fn(ctx)
        if rows:
            write(rows, Path(a.out_dir) / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
