#!/usr/bin/env python3
"""m16_x4w.py — X4a 補跑：xKV 每 8 層（Llama）／7 層（Qwen）一組（「W 大」），10/22 使用者要求補回。

為什麼：X4a 為了時間把 W 大縮掉（x_stack_howto.md §15.6、m16_x4.py 的 ALLOCS 註解），W 軸只剩 2 vs 4。
K1b（m16_k1_tucker.py）問「跨更多層一起分解有沒有多省」，這裡用 teacher forcing 直接看答對率的代理（Δgold_lp）。
設計與事先寫死的判定：docs/stage3_20261019/experiment_log.md「10/22 15:3xZ 起：補跑」的 X4a W 大一條（不改）。
  - 題目：X4a 同一批 96 題（seed 64000 起、T2–T5 各 24 題），GPU 路徑＝m16_x4.py tf（共用 SVD、E0 每題都算）。
  - 配置：T∈WBIG_T（20、24、32）× 係數位元 {INT4, K4V2}：W 大照 m16_x4 的 wbig 規則（K 秩上限 16、倍數落在 T±10%），
    同一個 T、同係數的 W＝4（m16_x4 的 best_rank，和 X4a 的 alloc:int4／alloc:k4v2 同一個規則）一起跑，逐題配對。
  - 判定：W 大 − W4 的 gold_lp 逐題差，bootstrap 10,000 次 95% CI；某模型至少一個 (T, 位元) CI 下界 > 0 → W_BIG_BETTER；
    全部的 CI 上界 < 0 → W_BIG_WORSE；其他 → UNDETERMINED。TF 只是篩選（X1 的規矩），不是生成答對率。
不改 m16_x4.py（它的 sha 是 X4 各段冒煙的依據）；配置檔另存 code/m16_xstack_combos_x4w.json。

  python code/m16_x4w.py grid [--out code/m16_xstack_combos_x4w.json]       只用 CPU
  python code/m16_x4w.py report --raw "<x4aw_*.csv.gz>" --out-dir <dir>      只用 CPU
  python code/m16_x4w.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import os
import random
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m16_x4 as X4  # noqa: E402

COMBOS_X4W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "m16_xstack_combos_x4w.json")
BITS_X4W = ("int4", "k4v2")
N_BOOT = 10000


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build(model: str) -> dict:
    cfg, pairs, skipped = {}, [], []

    def add(spec, role, why, T):
        e = cfg.setdefault(spec, dict(spec=spec, roles=[], why=[], targets=[]))
        for k, v in (("roles", role), ("why", why)):
            if v not in e[k]:
                e[k].append(v)
        if T is not None and T not in e["targets"]:
            e["targets"].append(T)
    wb = X4.wval(model, "wbig")
    for T in X4.WBIG_T:
        for bits in BITS_X4W:
            b_big = X4.best_rank(model, T, bits, wb, "diag", None, rk_cap=16)
            b_w4 = X4.best_rank(model, T, bits, 4, "diag", None)
            if b_big is None or b_w4 is None:
                skipped.append(dict(T=T, bits=bits, why="W 大或 W4 沒有落在 ±10% 內的秩", big=bool(b_big), w4=bool(b_w4)))
                continue
            add(b_big[1], f"wbig:{bits}", f"W＝{wb}、{bits}、落在 {T} 倍", T)
            add(b_w4[1], f"w4:{bits}", f"W＝4、{bits}、落在 {T} 倍（配對）", T)
            pairs.append(dict(T=T, bits=bits, wbig=b_big[1], w4=b_w4[1], mult_wbig=round(b_big[2], 3), mult_w4=round(b_w4[2], 3)))
    add("none", "control", "E0 對照（tf 每題本來就算 E0）", None)
    out = []
    for spec, e in cfg.items():
        st = X4.validate(X4.parse_spec(spec), X4.MODEL_INFO[model], X4.NREG)
        r = X4.predict_ratio(spec, model)
        out.append(dict(spec=spec, ratio=r, multiple=round(1 / r, 3), bands=[T for T in X4.TARGETS if X4.in_band(1 / r, T)],
                        targets=sorted(e["targets"]), **X4.spec_meta(spec), roles=e["roles"], why="；".join(e["why"]),
                        deployable=X4.spec_attrs(st)["deployable"]))
    out.sort(key=lambda c: (c["ratio"], c["spec"]))
    return dict(configs=out, pairs=pairs, skipped=skipped)


def grid_doc() -> dict:
    return {"_doc": ["m16_x4w.py 的 X4a 補跑配置（10/22；在跑任何 W 大之前產生，之後不改；--self-test 核對＝程式重算）。",
                     f"T∈{list(X4.WBIG_T)} × 係數 {list(BITS_X4W)}：W 大（Llama 8、Qwen 7；K 秩 ≤16）與 W＝4 各挑倍數最接近 T、在 ±10% 內的秩"
                     "（m16_x4.best_rank，和 X4a 同一個規則）；pairs 是逐題配對比較的對。"],
            "version": "2026-10-22 x4w v1", "targets": list(X4.WBIG_T), "bits": list(BITS_X4W),
            "models": {m: build(m) for m in X4.MODEL_INFO}}


def cmd_grid(a):
    g = grid_doc()
    if os.path.exists(a.out):
        old = json.load(open(a.out))
        if old.get("models") == json.loads(json.dumps(g["models"])):
            print(f"{a.out} 已有、內容相同 → 不重寫")
            return 0
        raise SystemExit(f"{a.out} 已存在但內容不同 → 不默默改事先寫死的配置")
    json.dump(g, open(a.out, "w"), indent=1, ensure_ascii=False)
    for m, v in g["models"].items():
        print(f"{m}：{len(v['configs'])} 個配置、{len(v['pairs'])} 對；跳過 {v['skipped']}")
        for p in v["pairs"]:
            print(f"  T={p['T']} {p['bits']}: {p['wbig']}（{p['mult_wbig']}×） vs {p['w4']}（{p['mult_w4']}×）")
    return 0


def boot(v: list, seed: int = 20261022) -> dict:
    rng = random.Random(seed)
    n = len(v)
    if n == 0:
        return dict(mean=None, lo=None, hi=None, n=0)
    ms = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    return dict(mean=sum(v) / n, lo=ms[int(0.025 * N_BOOT)], hi=ms[int(0.975 * N_BOOT) - 1], p_le0=sum(m <= 0 for m in ms) / N_BOOT, n=n)


def read_rows(pattern: str) -> list:
    rows = []
    for f in sorted(glob.glob(pattern)):
        with gzip.open(f, "rt") as fh:
            rows += list(csv.DictReader(fh))
    return rows


def cmd_report(a):
    rid = os.environ.get("RUN_ID", "adhoc")
    g = json.load(open(a.combos))
    rows = read_rows(a.raw)
    if not rows:
        raise SystemExit(f"沒有列：{a.raw}")
    by = {}
    for r in rows:
        by.setdefault((r["model"], r["item_id"]), {})[r["spec"]] = r
    out = dict(run_id=rid, ts=now(), raw=a.raw, combos=a.combos, src_run_ids=sorted({r["run_id"] for r in rows}),
               contended_rows=sum(1 for r in rows if r.get("own_gpu_intruders") not in ("", "0", None)), models={})
    lines = ["# X4a 補跑：W 大（每 8／7 層一組）vs W＝4（TF，逐題配對）", "", f"run：{rid}；來源 run：{', '.join(out['src_run_ids'])}", "",
             "| 模型 | T | 係數 | W 大（倍數） | W4（倍數） | 題數 | gold_lp 差（W大−W4）[95% CI] | first_ok 差 | relerr_k W大／W4 | relerr_v W大／W4 |",
             "|:--|--:|:--|:--|:--|--:|:--|--:|:--|:--|"]
    for m, v in g["models"].items():
        items = [k for k in by if k[0] == m]
        res = []
        for p in v["pairs"]:
            d, fo, rk, rv = [], [], [[], []], [[], []]
            for k in items:
                e = by[k]
                if p["wbig"] in e and p["w4"] in e:
                    A, B = e[p["wbig"]], e[p["w4"]]
                    d.append(float(A["gold_lp"]) - float(B["gold_lp"]))
                    fo.append(float(A["first_ok"]) - float(B["first_ok"]))
                    for j, x in enumerate((A, B)):
                        rk[j].append(float(x["relerr_k"]))
                        rv[j].append(float(x["relerr_v"]))
            ci = boot(d)
            mean = lambda z: sum(z) / len(z) if z else None  # noqa: E731
            res.append(dict(p, gold_lp_diff=ci, first_ok_diff=mean(fo), relerr_k=[mean(rk[0]), mean(rk[1])], relerr_v=[mean(rv[0]), mean(rv[1])]))
            f = lambda x, n=3: "—" if x is None else f"{x:+.{n}f}"  # noqa: E731
            lines.append(f"| {m} | {p['T']} | {p['bits']} | `{p['wbig']}`（{p['mult_wbig']}） | `{p['w4']}`（{p['mult_w4']}） | {ci['n']} | "
                         f"{f(ci['mean'])} [{f(ci['lo'])}, {f(ci['hi'])}] | {f(mean(fo))} | "
                         f"{f(mean(rk[0]))}／{f(mean(rk[1]))} | {f(mean(rv[0]))}／{f(mean(rv[1]))} |")
        ok = [r for r in res if r["gold_lp_diff"]["n"] > 0]
        if not ok or min(r["gold_lp_diff"]["n"] for r in ok) < 48:
            verdict = "INCOMPLETE"
        elif any(r["gold_lp_diff"]["lo"] > 0 for r in ok):
            verdict = "W_BIG_BETTER"
        elif all(r["gold_lp_diff"]["hi"] < 0 for r in ok):
            verdict = "W_BIG_WORSE"
        else:
            verdict = "UNDETERMINED"
        out["models"][m] = dict(pairs=res, verdict=verdict, n_items=len(items))
    out["rule"] = ("W 大 − W4 的 gold_lp 逐題差（bootstrap 95% CI）：至少一個 (T, 位元) 下界 > 0 → W_BIG_BETTER；全部上界 < 0 → W_BIG_WORSE；"
                   "其他 → UNDETERMINED；最少的一對 < 48 題 → INCOMPLETE。TF 只是篩選。")
    lines += ["", "判定：" + "；".join(f"{m} **{d['verdict']}**（{d['n_items']} 題）" for m, d in out["models"].items()), "", "規則：" + out["rule"],
              f"有 GPU 爭用紀錄的列：{out['contended_rows']}（TF 的分數不受影響；時間欄不用）。"]
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(a.out_dir, "x4w_report.json"), "w"), indent=1, ensure_ascii=False)
    open(os.path.join(a.out_dir, "x4w_report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def self_test() -> bool:
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name + (f"  {info}" if info else ""))
    g = grid_doc()
    for m, v in g["models"].items():
        check(f"{m}：有配對", len(v["pairs"]) >= 1, str(len(v["pairs"])))
        for p in v["pairs"]:
            check(f"{m} T={p['T']} {p['bits']}：W 大的 W＝{X4.wval(m, 'wbig')}、W4 的 W＝4",
                  X4.spec_meta(p["wbig"])["W"] == X4.wval(m, "wbig") and X4.spec_meta(p["w4"])["W"] == 4)
            check(f"{m} T={p['T']} {p['bits']}：兩個都在 T±10%", X4.in_band(p["mult_wbig"], p["T"]) and X4.in_band(p["mult_w4"], p["T"]))
        w4_x4a = {c["spec"] for c in json.load(open(X4.COMBOS_X4))["models"][m]["configs"]}
        same = [p["w4"] in w4_x4a for p in v["pairs"]]
        check(f"{m}：W4 那一邊和 X4a 的 alloc 同一個規則（{sum(same)}／{len(same)} 個本來就在 X4a 的格點裡）", True)
    if os.path.exists(COMBOS_X4W):
        check("combos_x4w.json＝程式重算", json.load(open(COMBOS_X4W)).get("models") == json.loads(json.dumps(g["models"])))
    b = boot([1.0] * 10)
    check("bootstrap：常數 → CI 縮成一點", b["lo"] == b["hi"] == 1.0)
    print("self-test：" + ("ALL PASS" if ok else "有 FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    sp = ap.add_subparsers(dest="cmd")
    g = sp.add_parser("grid")
    g.add_argument("--out", default=COMBOS_X4W)
    r = sp.add_parser("report")
    r.add_argument("--raw", required=True)
    r.add_argument("--combos", default=COMBOS_X4W)
    r.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if a.cmd == "grid":
        sys.exit(cmd_grid(a))
    if a.cmd == "report":
        sys.exit(cmd_report(a))
    ap.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
