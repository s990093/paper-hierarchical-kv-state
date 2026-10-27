#!/usr/bin/env python3
"""kvs_s0.py — S0-2：兩個使用者問同一份長 context，各自挑前 m 塊，聯集 u(2) 有多大（候選 A；品質型，不量時間）。

登記：docs/kv_sharing_hiding_20261026/runs/S0_union.md（題組、放法、指標、判準都在那裡；這支只照做）。

  python code/s3/kvs_s0.py --self-test                               # CPU：集合運算、欄位、判定邏輯
  python code/s3/kvs_s0.py run --cell L128 --shard 0/1 --budget-min 210   # GPU（runner part kvs-s0）
  python code/s3/kvs_s0.py report                                    # CPU：raw → 判定表

重用 A0（不改 A0、不寫 A0 的輸出）：
  題目＝a0.plan_set("gen2", cell)（seed0＝73000、T2–T4 各 32、每格 96 題）；出題＝Session.T.make_item（和 A0 相同）。
  prefill＝ItemState 的同一段（fa3 → sdpa）；打分＝a0_gpu.forward_hooked＋ctx_mass（每層的 [nreg] 分數）。
  兩個問題（第 1 輪、第 2 輪的 query_ids）都「direct」接在 context 後面（各自獨立的使用者）。
本檔頂層只有標準函式庫（a0.py 同樣）；GPU 的子命令才 import torch。
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
REPO = os.path.dirname(CODE)
sys.path.insert(0, HERE)
sys.path.insert(0, CODE)
import a0 as A  # noqa: E402

OUT = os.path.join(REPO, "results", "s3_h200", "kvs_s0")
RAW = os.path.join(OUT, "raw")
CHUNK = A.CHUNK
JUDGED = ("L128", "Q128", "Q256", "Q14-256")
DECISIVE_CELL = "L128"
RHO_MAIN = "1/4"
EARLY_STOP = 0.85            # 登記 §2：L128 兩個中位數都 > 0.85 → STOP-S（當場）
PRESCREEN = 0.75             # 登記 §2：≥2 個判定格兩個中位數都 ≤ 0.75
FIELDS = ["run_id", "ts", "cell", "model", "n_blocks", "item_id", "task", "seed", "band1", "band2", "turn", "rho", "m",
          "u2_global", "ov_global", "u2_layer", "u2_layer_p90", "rec_own_2", "rec_cross_2", "rec_union_2", "rec_own_1", "rec_cross_1",
          "n_needle_1", "n_needle_2", "nq_1", "nq_2", "t_prefill_s", "t_score_s", "scores_file", "own_gpu_intruders", "host_level",
          "n_rows_item"]


# ============================================================================ 集合運算（純 Python，自測可測）
def topm(score: list, m: int) -> list:
    """分數由大到小取前 m 個（平手取編號小的）＝a0_gpu.topm。"""
    return sorted(range(len(score)), key=lambda j: (-score[j], j))[:m]


def _p90(v: list) -> float:
    s = sorted(v)
    x = 0.9 * (len(s) - 1)
    lo = int(x)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (x - lo)


def _recall(needle: list, sel: set):
    return "NOT_MEASURED" if not needle else sum(b in sel for b in needle) / len(needle)


def union_metrics(mass1: list, mass2: list, needle1: list, needle2: list, rho: str) -> dict:
    """mass1／mass2：[L][nreg] 的每層分數（第 1、2 個使用者）；needle：區內編號（block−1）。登記 §1.3 的欄位。"""
    nreg = len(mass1[0])
    m = A.m_of(rho, nreg)
    g1 = [sum(col) for col in zip(*mass1)]
    g2 = [sum(col) for col in zip(*mass2)]
    S1, S2 = set(topm(g1, m)), set(topm(g2, m))
    per = [len(set(topm(a, m)) | set(topm(b, m))) / (2 * m) for a, b in zip(mass1, mass2)]
    return dict(rho=rho, m=m, u2_global=len(S1 | S2) / (2 * m), ov_global=len(S1 & S2) / m,
                u2_layer=sum(per) / len(per), u2_layer_p90=_p90(per),
                rec_own_2=_recall(needle2, S2), rec_cross_2=_recall(needle2, S1), rec_union_2=_recall(needle2, S1 | S2),
                rec_own_1=_recall(needle1, S1), rec_cross_1=_recall(needle1, S2), n_needle_1=len(needle1), n_needle_2=len(needle2))


# ============================================================================ GPU
def score_question(G, km, q_ids: list, ctx_end: int, nreg: int):
    """一個問題 direct 接在 context 後面（位置 ctx_end 起）：每層的 ctx_mass → [L, nreg]（float64，GPU）。
    ＝a0_gpu.ItemState.question_pass 的打分部分（hook 裡用 [0, ctx_end) 的 key；問題自己的 key 不算）。"""
    import torch
    mass = torch.zeros(km.L, nreg, dtype=torch.float64, device=km.device)
    nq = len(q_ids)

    def hook(li, q):
        mass[li] = G.ctx_mass(km, q, km.kv[li, 0, :, :ctx_end], nq, nreg)
    with torch.inference_mode():
        G.forward_hooked(km, torch.tensor(q_ids, device=km.device), ctx_end, hook=hook)
    return mass


def cmd_run(a):
    import numpy as np
    import torch
    import a0_gpu as G
    from m16_c1a_predict import finish_watch, host_level, intruders, start_watch
    if a.cell not in A.CELLS:
        raise SystemExit(f"不認得的 --cell {a.cell}")
    model, nb = A.CELLS[a.cell]["model"], A.CELLS[a.cell]["nb"]
    nreg = nb - 1
    if a.smoke and not a.out:
        raise SystemExit("--smoke 要給 --out（冒煙的輸出不進 results）")
    si, _ = A.parse_shard(a.shard)
    out = a.out or os.path.join(RAW, f"s0_{a.cell}_s{si}.csv.gz")
    paths = [out] if a.smoke else sorted(set(glob.glob(os.path.join(RAW, f"s0_{a.cell}_s*.csv.gz"))) | {out})
    todo = A.pending(["gen2"], a.shard, [p for p in paths if os.path.exists(p)], a.cell)
    if a.max_items:
        todo = todo[:a.max_items]
    print(f"[s0] {a.cell}（{model}、{nb} 塊）：這個分片還有 {len(todo)} 題 → {out}", flush=True)
    if not todo:
        return
    sess = G.Session(model, nb)                      # 不載 xKV 校準基底（S0 用不到）
    km = sess.km
    rid = A.run_id()
    rd = os.environ.get("RUN_DIR") or os.path.join(os.environ.get("TIARA_RUNS", os.path.join(REPO, "_big", "runs")), rid)
    os.makedirs(rd, exist_ok=True)
    watch = start_watch(out)
    fo, w = A.open_out(out, FIELDS)
    t_start = time.time()
    try:
        for k, r in enumerate(todo):
            if a.budget_min and (time.time() - t_start) / 60 > a.budget_min:
                print(f"[s0] 時間預算 {a.budget_min} 分鐘到了 → 停在題與題之間（做完 {k} 題）", flush=True)
                break
            it = sess.T.make_item(r["task"], r["b1"], r["b2"], r["seed"], sess.ans.tok, sess.P.get(r["task"]), n_blocks=nb)
            if it["item_id"] != r["item_id"]:
                raise RuntimeError(f"item_id {it['item_id']} ≠ 計畫的 {r['item_id']}")
            if len(it["turns"]) < 2:
                raise RuntimeError(f"{it['item_id']}：gen2 的題應該有兩輪，只有 {len(it['turns'])}")
            ids = torch.tensor(it["context_ids"], device=km.device)
            if ids.numel() != nb * CHUNK:
                raise RuntimeError(f"{it['item_id']}：context {ids.numel()} token ≠ {nb}×{CHUNK}")
            km.key_keep = None
            torch.cuda.synchronize()
            t0 = time.time()
            km.set_attn("fa3")
            try:
                km.prefill_chunked(ids)
            finally:
                km.set_attn("sdpa")
            torch.cuda.synchronize()
            t_pre = time.time() - t0
            ctx_end = ids.numel()
            t1 = time.time()
            q = [list(it["turns"][t]["query_ids"]) for t in (0, 1)]
            mass = [score_question(G, km, q[t], ctx_end, nreg) for t in (0, 1)]
            torch.cuda.synchronize()
            t_sc = time.time() - t1
            m1, m2 = (x.float().cpu() for x in mass)
            needle = [[b - 1 for b in it["turns"][t]["needle_blocks"] if 1 <= b < nb] for t in (0, 1)]
            sf = os.path.join(rd, f"scores_{a.cell}_{it['item_id']}.npz")
            np.savez_compressed(sf, mass1=m1.numpy(), mass2=m2.numpy(), needle1=np.array(needle[0]), needle2=np.array(needle[1]))
            L1, L2 = m1.tolist(), m2.tolist()
            lvl, intr = host_level(), intruders(watch)
            for rho in A.RHOS:
                row = union_metrics(L1, L2, needle[0], needle[1], rho)
                row.update(run_id=rid, ts=A.ts(), cell=a.cell, model=model, n_blocks=nb, item_id=it["item_id"], task=r["task"],
                           seed=r["seed"], band1=r["b1"], band2=r["b2"], turn=r["turn"], nq_1=len(q[0]), nq_2=len(q[1]),
                           t_prefill_s=round(t_pre, 2), t_score_s=round(t_sc, 2), scores_file=sf, own_gpu_intruders=intr,
                           host_level=lvl, n_rows_item=len(A.RHOS))
                w.writerow({f: (round(v, 6) if isinstance(v, float) else v) for f, v in ((f, row.get(f, "")) for f in FIELDS)})
            fo.flush()
            # 進度只印時間（分數一律看 report；冒煙不看分數）
            print(f"[{'smoke ' if a.smoke else ''}{k + 1}/{len(todo)}] {it['item_id']} prefill {t_pre:.1f}s 打分 {t_sc:.1f}s "
                  f"需要塊 {len(needle[0])}/{len(needle[1])}", flush=True)
            del mass, m1, m2, L1, L2, ids
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        fo.close()
        finish_watch(watch, out)


# ============================================================================ 報告與判定（登記 §2）
def medians(rows: list) -> dict:
    """每格、ρ＝1/4：u2_global 與 u2_layer 的中位數（每題一個值；同一題重複的列取第一個）。"""
    seen, by = set(), {}
    for r in rows:
        if r["rho"] != RHO_MAIN:
            continue
        k = (r["cell"], r["item_id"])
        if k in seen:
            continue
        seen.add(k)
        by.setdefault(r["cell"], []).append((float(r["u2_global"]), float(r["u2_layer"])))
    return {c: dict(n_items=len(v), u2_global=statistics.median(x[0] for x in v), u2_layer=statistics.median(x[1] for x in v))
            for c, v in by.items()}


def verdict(med: dict, a0_verdict: str | None) -> dict:
    out = dict(medians=med, a0=a0_verdict or "NOT_MEASURED")
    d = med.get(DECISIVE_CELL)
    if d and d["n_items"] >= 96:
        out["decisive"] = "STOP-S" if (d["u2_global"] > EARLY_STOP and d["u2_layer"] > EARLY_STOP) else "CONTINUE"
    else:
        out["decisive"] = "NOT_MEASURED"
    full = [c for c in JUDGED if c in med and med[c]["n_items"] >= 96]
    if len(full) < len(JUDGED):
        out["prescreen"] = "NOT_MEASURED" if out["decisive"] != "STOP-S" else "STOP-S"
        out["incomplete"] = [c for c in JUDGED if c not in full]
        return out
    ok = [c for c in JUDGED if med[c]["u2_global"] <= PRESCREEN and med[c]["u2_layer"] <= PRESCREEN]
    out["cells_le_075"] = ok
    if a0_verdict == "STOP" or not ok:
        out["prescreen"] = "STOP-S"
    elif a0_verdict == "GO" and len(ok) >= 2:
        out["prescreen"] = "PASS-S0-2"
    elif a0_verdict in (None, "NOT_MEASURED"):
        out["prescreen"] = "PENDING_A0"
    else:
        out["prescreen"] = "ASK"                    # 恰好 1 格；或 A0 是 WEAK（登記沒寫到 → 問人，見 S0 偏離 1）
    return out


def a0_status() -> str | None:
    p = os.path.join(A.OUT, "a0_report.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    if d.get("incomplete"):
        return "NOT_MEASURED"
    return d["ga0"]["verdict"]


def cmd_report(a):
    rows = A.read_rows(sorted(glob.glob(os.path.join(RAW, "s0_*_s*.csv.gz"))))
    v = verdict(medians(rows), a0_status())
    v.update(run_id=A.run_id(), ts=A.ts(), n_rows=len(rows))
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "s0_verdict.json")
    json.dump(v, open(p, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(v, ensure_ascii=False, indent=1), "→", p)


# ============================================================================ 自測（CPU）
def self_test() -> bool:
    ok = True

    def check(name, cond):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name)
        ok &= bool(cond)

    for c in JUDGED:
        items = A.plan_set("gen2", c)
        check(f"{c}：gen2 有 96 題、都是第 2 輪、T2–T4 各 32", len(items) == 96 and all(r["turn"] == 1 for r in items)
              and all(sum(r["task"] == t for r in items) == 32 for t in ("T2", "T3", "T4")))
    check("ρ＝1/4 時 m＝63（252 塊）、126（504 塊）", A.m_of("1/4", 251) == 63 and A.m_of("1/4", 503) == 126)
    # 合成：2 層、8 塊。使用者 1 偏好 0,1,2,3；使用者 2 偏好 0,1,4,5（第 2 層相反）
    m1 = [[8, 7, 6, 5, 1, 1, 1, 1], [8, 7, 6, 5, 1, 1, 1, 1]]
    m2 = [[8, 7, 1, 1, 6, 5, 1, 1], [1, 1, 1, 1, 8, 7, 6, 5]]
    r = union_metrics(m1, m2, needle1=[2], needle2=[5], rho="1/2")
    check("m＝round(ρ·nreg)＝4", r["m"] == 4)
    # 全域：g1＝[16,14,12,10,2,2,2,2] → {0,1,2,3}；g2＝[9,8,2,2,14,12,7,6] → {4,5,0,1} → 聯集 6 → 6/8
    check("全域 u2＝6/8、重疊＝2/4", abs(r["u2_global"] - 0.75) < 1e-12 and abs(r["ov_global"] - 0.5) < 1e-12)
    # 逐層：第 1 層 {0,1,2,3}∪{0,1,4,5}＝6 → 0.75；第 2 層 {0,1,2,3}∪{4,5,6,7}＝8 → 1.0；平均 0.875
    check("逐層 u2 平均＝0.875、p90＝0.975", abs(r["u2_layer"] - 0.875) < 1e-12 and abs(r["u2_layer_p90"] - 0.975) < 1e-12)
    check("recall：第 2 輪答案塊 5 在 S2、不在 S1", r["rec_own_2"] == 1.0 and r["rec_cross_2"] == 0.0 and r["rec_union_2"] == 1.0)
    check("recall：第 1 輪答案塊 2 在 S1、不在 S2", r["rec_own_1"] == 1.0 and r["rec_cross_1"] == 0.0)
    check("沒有答案塊 → NOT_MEASURED", union_metrics(m1, m2, [], [], "1/2")["rec_own_2"] == "NOT_MEASURED")
    same = union_metrics(m1, m1, [], [], "1/2")
    check("兩人相同 → u2＝0.5", same["u2_global"] == 0.5 and same["u2_layer"] == 0.5)
    check("topm 平手取編號小的", topm([1, 3, 3, 0], 2) == [1, 2])
    # 判定
    def med(**kw):
        return {c: dict(n_items=96, u2_global=v[0], u2_layer=v[1]) for c, v in kw.items()}
    check("決定性：L128 兩個都 > 0.85 → STOP-S", verdict(med(L128=(0.9, 0.95)), None)["decisive"] == "STOP-S")
    check("決定性：只有一個 > 0.85 → CONTINUE", verdict(med(L128=(0.9, 0.7)), None)["decisive"] == "CONTINUE")
    allc = dict(L128=(0.6, 0.7), Q128=(0.7, 0.74), Q256=(0.8, 0.7), **{"Q14-256": (0.9, 0.9)})
    m4 = {c: dict(n_items=96, u2_global=v[0], u2_layer=v[1]) for c, v in allc.items()}
    check("預篩：2 格 ≤0.75、A0 GO → PASS", verdict(m4, "GO")["prescreen"] == "PASS-S0-2")
    check("預篩：A0 STOP → STOP-S", verdict(m4, "STOP")["prescreen"] == "STOP-S")
    check("預篩：A0 還沒判 → PENDING_A0", verdict(m4, None)["prescreen"] == "PENDING_A0")
    check("預篩：A0 WEAK → ASK", verdict(m4, "WEAK")["prescreen"] == "ASK")
    one = dict(m4)
    one["Q128"] = dict(n_items=96, u2_global=0.8, u2_layer=0.8)
    check("預篩：恰好 1 格 → ASK", verdict(one, "GO")["prescreen"] == "ASK")
    none_ = {c: dict(n_items=96, u2_global=0.8, u2_layer=0.8) for c in JUDGED}
    check("預篩：0 格 → STOP-S", verdict(none_, "GO")["prescreen"] == "STOP-S")
    check("預篩：格子沒跑齊 → NOT_MEASURED", verdict(med(L128=(0.6, 0.6)), "GO")["prescreen"] == "NOT_MEASURED")
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--cell", required=True)
    r.add_argument("--shard", default="0/1")
    r.add_argument("--budget-min", type=float, default=0)
    r.add_argument("--max-items", type=int, default=0)
    r.add_argument("--smoke", action="store_true")
    r.add_argument("--out", default="")
    sub.add_parser("report")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    {"run": cmd_run, "report": cmd_report}.get(a.cmd, lambda _: ap.print_help())(a)


if __name__ == "__main__":
    main()
