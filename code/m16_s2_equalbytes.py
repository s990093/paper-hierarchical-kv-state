"""m16_s2_equalbytes.py — 第三階段 S2：載入區的有損刪除 vs 降精度，同位元組（teacher forcing，只量品質）。

計畫：docs/stage3_20261019/experiment_matrix.md §3「S2」。設計與停止條件（寫死，不改）：
docs/stage3_20261019/lit/L1_eviction_selective_loading.md §4 G1、§5 X1。怎麼跑：docs/stage3_20261019/c1a_s2_howto.md。

題目與度量和 DS1（14 號 §8.4、m10_drop screen）同一套：seed 60000 起、T2–T5 各 --n 題（預設 24）、32K；
m10_sweep.Item prefill 一次，第 1 輪問題＋標準答案一次餵進去（teacher forcing），對 E0 量 KL、gold_lp、top1、first_ok。

只動「載入區」block 32–63（Cake 還原時前段反正會被精確重算；L1 G1）；block 0 一律 BF16、block 1–31 照 E0（BF16、不遮）。
  刪除組（group＝drop）：B 粒度，被丟的 block 用 KVModel.key_keep 遮掉（＝從 cache 拿掉，RoPE 位置不變；14 §4.1）。
    丟哪些：m10_drop.unit_order(cx, "B", 方法)（DS1 的同一個「先丟」順序，全段 1–63 算的）只留區內的 block、保持原順序，取前 k 個。
    方法：DS1 選出的前 2 名寫入時方法（--write-methods，預設 SNAP KEYDIFF；或 --ds2-plan 讀 DS1 pick 的 B 粒度 top2）、
          QRY（知道第 1 輪問題）、RND0；ORC（第 1 輪答案 block 最後才丟，其餘照 RND0）只做 50%。
    丟幾個：名目預算 1/2、1/4、1/8 → 丟區內 50%、75%、87.5%（32 個 block → 16、24、28）；
          另加「照實等位元組」（--no-matched 關掉）：每個量化格式的實際位元組比 r（含 scale／zero）→ 丟 round(32×(1−r)) 個。
  低精度組（group＝quant）：區內 32 個 block 全部換成 encode→decode 後的值（m10_sweep.Item.load_fmt／place＝09 海選同一段）：
    FP8（照 results/m10_precision_mi300x/fmt_select/fmt_choice.json 每個模型選定的）、INT8（int8_token）、INT4 KIVI、
    INT2 KIVI、K4V2（kv-int4_kivi-int2_kivi）。attention 路徑兩種都量：mask（全留的 key_keep，和刪除組同一條路徑＝配對比較用）、
    sdpa（不設 key_keep，和第二階段的量化數字同一條路徑）。
  上下界：E0（全 BF16、sdpa）、MASK0（全 BF16、全留的遮罩路徑；和 E0 的差＝換路徑的雜訊）、ORC 50%。
  全段版（--full-range）：同樣的東西改成動 block 1–63（補充；也讓「全段、丟 50%／75%」和 DS1 的 B 格子逐一對得上 → DS1 重現檢查）。

位元組照實算：每個 block 的 Enc.nbytes（q＋scale＋zero＋…，m10_codec／m10_codec_x 的定義）÷ BF16 的 chunk_bytes；
刪除＝留下的 block 數 × BF16 位元組（丟掉的 block 不存；遮罩本身不算位元組）。每列記 ratio_region（區內）與 ratio_total（整段 64 個 block）。

子命令：
  run     正式（--shard i/n、逐題續跑、GpuWatcher 品質規則）；--smoke＝1 題（DS1 第一題）、全段版、印每個配置、跑 DS1 重現檢查
  report  CPU：逐列 CSV → s2_report.json、tables_s2.md（配對 bootstrap 10,000 次、以題目為單位；L1 G1 停止條件的成分）

模組頂層只有標準函式庫與 m16_c1a_predict（同樣沒有副作用）的 import；torch、m10_* 一律在函式裡才 import。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time

from m16_c1a_predict import (CHUNK, N_BLOCKS, PARAMS, REPO, boot, ci_has, done_items, finish_watch, fmt_ci, host_level,
                             intruders, item_id_of, open_out, parse_shard, read_rows, side_path, start_watch, ts)

OUT_DIR = os.path.join(REPO, "results", "m16_stage3_mi300x", "s2_equal_bytes")
FMT_CHOICE = os.path.join(REPO, "results", "m10_precision_mi300x", "fmt_select", "fmt_choice.json")
DS2_PLAN = os.path.join(REPO, "results", "m10_drop_mi300x", "ds2", "ds2_plan.json")
DS1_RAW = "/mlsteam/data/tiara/drop_m10/ds1_{model}_s*.csv.gz"           # 44e1 的 DS1 逐列輸出

REGION = {"region": (32, 63), "full": (1, 63)}
NOMINAL = (("1/2", 0.5), ("1/4", 0.75), ("1/8", 0.875))                   # 名目預算 → 丟區內的佔比（L1 X1 的表）
NOMINAL_QUANT = {"1/2": ("FP8", "int8_token"), "1/4": ("int4_kivi",), "1/8": ("int2_kivi", "kv-int4_kivi-int2_kivi")}
DEFAULT_FORMATS = ("FP8", "int8_token", "int4_kivi", "int2_kivi", "kv-int4_kivi-int2_kivi")
EPS_FIRST_OK = 0.02                    # 第二階段的 ε（−2 個百分點）用在 first_ok 上（見 howto 的假設清單）
METRICS = ("gold_lp", "first_ok", "kl")                                  # gold_lp、first_ok 越高越好；kl 越低越好

FIELDS = ["run_id", "ts", "shard", "model", "item_id", "task", "band", "seed", "needle_blocks", "variant", "region",
          "needle_in_region", "group", "role", "method", "fmt", "attn_path", "k_blocks", "frac_of_region", "budget_tags",
          "bytes_block_fmt", "bytes_block_bf16", "bytes_region", "ratio_region", "bytes_total", "ratio_total",
          "bits_per_value_region", "kept_frac", "needle_kept", "needle_lowprec", "kl", "gold_lp", "gold_lp_e0", "top1",
          "first_ok", "first_ok_e0", "t_score_s", "write_methods_source", "n_rows_item", "own_gpu_intruders", "host_level"]


def resolve_formats(model: str, names, path: str) -> list:
    """['FP8', 'int8_token', …] → [(名稱, 實際格式)]；FP8＝fmt_choice.json 這個模型選定的。"""
    ch = json.load(open(path))["choice"][model]
    return [(n, ch["fp8"] if n == "FP8" else n) for n in names]


def keep_tensor(km, drop):
    """B 粒度的 key_keep（[max_len] bool；context 之後全 True）：drop 裡的 block 遮掉。和 m10_drop.keep_mask(B) 同一個形狀。"""
    import torch
    kk = torch.ones(km.max_len, dtype=torch.bool, device=km.device)
    for b in drop:
        kk[b * CHUNK:(b + 1) * CHUNK] = False
    return kk


def kept_stats(kk, n: int, nd):
    """留下的 token 比例、答案 block 的 token 留下的比例（DS1 的 kept_frac、needle_kept，同樣四捨五入到 5 位）。"""
    kept = round(float(kk[:n].float().mean()), 5)
    ndk = round(float(kk[nd].float().mean()), 5) if nd.numel() else ""
    return kept, ndk


def tf(st, lp0, kk):
    """teacher forcing（m10_sweep.Item.score）在 key_keep＝kk 下對 E0 的度量；kk＝None＝原本的 sdpa 路徑。"""
    import m10_sweep as SW
    km = st.ans.km
    km.key_keep = kk
    try:
        return SW.Item.metrics(st.score(), lp0, st.gold_ids)
    finally:
        km.key_keep = None


def load_ds1(pattern: str, model: str):
    paths = sorted(glob.glob(pattern))
    rows = read_rows(paths)
    return {(r["item_id"], r["gran"], r["method"], round(float(r["frac"]), 4)): r for r in rows if r["model"] == model}, paths


def compare_ds1(ds1: dict, item_id: str, rows: list) -> list:
    """E0、MASK0、全段版丟 50%／75%（k＝32、47，和 DS1 的 B 格子同一個 k＝round(佔比×63)）對 DS1 的逐列輸出。"""
    out = []

    def cmp(mine, key):
        r = ds1.get(key)
        if r is None:
            return
        d = {}
        for col in ("kl", "gold_lp", "top1", "first_ok", "kept_frac", "needle_kept"):
            if r.get(col, "") != "" and mine.get(col, "") != "":
                d[col] = abs(float(mine[col]) - float(r[col]))
        out.append(dict(key=list(key), run_id_ds1=r["run_id"], exact=all(v == 0 for v in d.values()), absdiff=d))

    for mine in rows:
        if mine["group"] == "E0":
            cmp(mine, (item_id, "", "E0", 0.0))
        elif mine["group"] == "MASK0":
            cmp(mine, (item_id, "B", "MASK0", 0.0))
        elif mine["group"] == "drop" and mine["variant"] == "full":
            for frac in (0.5, 0.75):
                if int(mine["k_blocks"]) == int(round(frac * (N_BLOCKS - 1))):
                    cmp(mine, (item_id, "B", mine["method"], frac))
    return out


def run_item(ans, it: dict, a, drop_methods: list, roles: dict, fmts: list, src: str, mask_check: bool):
    """一題的所有格子。回傳 (rows, checks)。"""
    import torch
    import m10_codec_x as X
    import m10_drop as DR
    import m10_sweep as SW
    km = ans.km
    km.key_keep = None
    st = SW.Item(ans, it)                                          # DS1 同一個 Item（prefill、BF16 備份、第 1 輪 seq）
    tr = it["turns"][0]
    n = st.ctx_len
    assert ans.CHUNK == CHUNK and n == N_BLOCKS * CHUNK, (ans.CHUNK, n)
    lp0 = st.score()
    m0 = SW.Item.metrics(lp0, lp0, st.gold_ids)
    cx = DR.Ctx(km, st.ids, n, st.bf16, ans.CHUNK, it["seed"], st.needle, tr["query_ids"])   # 和 m10_drop.cmd_screen 相同
    tsc = {}
    for meth in (m for m in drop_methods if m in DR.SCORED):
        torch.cuda.synchronize()
        t1 = time.time()
        DR.scores(cx, meth)
        torch.cuda.synchronize()
        tsc[meth] = round(time.time() - t1, 3)
    orders = {m: DR.unit_order(cx, "B", m) for m in list(drop_methods) + ["ORC"]}
    cb = km.chunk_bytes
    nbytes = {alias: X.encode(st.bf16[1], f).nbytes for alias, f in fmts}          # 形狀決定，每個 block 一樣；load_fmt 再核對
    needle = list(st.needle)
    nd = torch.tensor([t for b in needle for t in range(b * CHUNK, (b + 1) * CHUNK)], dtype=torch.long, device=km.device)
    base = dict(model=a.model, item_id=it["item_id"], task=it["task"], band=tr["band"], seed=it["seed"],
                needle_blocks=json.dumps(needle), gold_lp_e0=m0["gold_lp"], first_ok_e0=m0["first_ok"],
                write_methods_source=src, bytes_block_bf16=cb)
    rows, checks = [], {"mask_vs_m10_drop_keep_mask": [], "nbytes": nbytes, "chunk_bytes_bf16": cb, "t_score_s": tsc}

    def add(**kw):
        rows.append(dict(base, **kw))

    ref = dict(variant="", region="", needle_in_region="", role="ref", fmt="bf16", k_blocks=0, frac_of_region=0.0,
               budget_tags="", bytes_block_fmt=cb, bytes_region="", ratio_region=1.0, bytes_total=N_BLOCKS * cb,
               ratio_total=1.0, bits_per_value_region=16.0, kept_frac=1.0, needle_kept=1.0, needle_lowprec=0.0, t_score_s="")
    add(group="E0", method="E0", attn_path="sdpa", **ref, **m0)
    kk_all = keep_tensor(km, [])
    add(group="MASK0", method="MASK0", attn_path="mask", **ref, **tf(st, lp0, kk_all))
    # ---- 刪除組
    for var in a.variants:
        lo, hi = REGION[var]
        nreg = hi - lo + 1
        ks = {}
        for tag, fr in NOMINAL:
            ks.setdefault(int(round(fr * nreg)), []).append(f"nominal:{tag}")
        if a.matched:
            for alias, _ in fmts:
                ks.setdefault(int(round(nreg * (1 - nbytes[alias] / cb))), []).append(f"matched:{alias}")
        nin = sum(lo <= b <= hi for b in needle)
        for meth in list(drop_methods) + ["ORC"]:
            reg = [b for b in orders[meth] if lo <= b <= hi]
            assert len(reg) == nreg, (meth, len(reg), nreg)
            kset = [int(round(0.5 * nreg))] if meth == "ORC" else sorted(ks)
            for k in kset:
                drop = reg[:k]
                kk = keep_tensor(km, drop)
                mm = tf(st, lp0, kk)
                kept, ndk = kept_stats(kk, n, nd)
                tags = ["nominal:1/2"] if meth == "ORC" else ks[k]
                add(variant=var, region=f"{lo}-{hi}", needle_in_region=nin, group="drop", role=roles.get(meth, "orc"),
                    method=meth, fmt="drop", attn_path="mask", k_blocks=k, frac_of_region=round(k / nreg, 6),
                    budget_tags=" ".join(tags), bytes_block_fmt=0, bytes_region=(nreg - k) * cb,
                    ratio_region=round((nreg - k) / nreg, 6), bytes_total=(N_BLOCKS - k) * cb,
                    ratio_total=round((N_BLOCKS - k) / N_BLOCKS, 6), bits_per_value_region=round(16 * (nreg - k) / nreg, 4),
                    kept_frac=kept, needle_kept=ndk, needle_lowprec=0.0, t_score_s=tsc.get(meth, ""), **mm)
                if mask_check and var == "full":
                    for frac in (0.5, 0.75):
                        if k == int(round(frac * (N_BLOCKS - 1))):
                            kd = DR.keep_mask(cx, "B", meth, frac)[0]
                            checks["mask_vs_m10_drop_keep_mask"].append(
                                dict(method=meth, frac=frac, k=k, equal=bool(torch.equal(kd, kk))))
    # ---- 低精度組
    for alias, f in fmts:
        st.load_fmt(f)
        if st.lp_nbytes != nbytes[alias]:
            raise RuntimeError(f"{f}：load_fmt 的位元組 {st.lp_nbytes} ≠ 單一 block 編碼的 {nbytes[alias]}")
        r = nbytes[alias] / cb
        tags = [f"nominal:{t}" for t, qs in NOMINAL_QUANT.items() if alias in qs] + ([f"matched:{alias}"] if a.matched else [])
        for var in a.variants:
            lo, hi = REGION[var]
            nreg = hi - lo + 1
            eff = frozenset(range(lo, hi + 1))
            st.place(eff)
            nin = sum(lo <= b <= hi for b in needle)
            tot = (N_BLOCKS - nreg) * cb + nreg * nbytes[alias]
            for path in a.quant_paths:
                mm = tf(st, lp0, kk_all if path == "mask" else None)
                add(variant=var, region=f"{lo}-{hi}", needle_in_region=nin, group="quant", role="quant", method=alias, fmt=f,
                    attn_path=path, k_blocks=nreg, frac_of_region=1.0, budget_tags=" ".join(tags),
                    bytes_block_fmt=nbytes[alias], bytes_region=nreg * nbytes[alias], ratio_region=round(r, 6),
                    bytes_total=tot, ratio_total=round(tot / (N_BLOCKS * cb), 6), bits_per_value_region=round(16 * r, 4),
                    kept_frac=1.0, needle_kept=1.0, needle_lowprec=round(nin / len(needle), 5) if needle else "",
                    t_score_s="", **mm)
    st.place(frozenset())
    del st, cx
    return rows, checks


def _print_smoke(rows: list, checks: dict, ds1cmp):
    print(f"\n=== smoke：{rows[0]['item_id']}（E0 gold_lp {rows[0]['gold_lp']:.3f}、first_ok {rows[0]['first_ok']}）", flush=True)
    print(f"  {'組':<6} {'方法':<24} {'段':<6} {'路徑':<5} {'k':>3} {'區內位元組比':>11} {'整段比':>8} {'Δgold_lp':>9} "
          f"{'KL':>8} {'first_ok':>8} {'答案留下':>8}")
    for r in rows:
        print(f"  {r['group']:<6} {r['method']:<24} {r['variant']:<6} {r['attn_path']:<5} {r['k_blocks']:>3} "
              f"{r['ratio_region']:>11} {r['ratio_total']:>8} {r['gold_lp'] - r['gold_lp_e0']:>+9.3f} {r['kl']:>8.4f} "
              f"{r['first_ok']:>8} {str(r['needle_kept']):>8}")
    print("  位元組／block：" + json.dumps(checks["nbytes"]) + f"（BF16 {checks['chunk_bytes_bf16']}）")
    print("  遮罩 vs m10_drop.keep_mask：" + json.dumps(checks["mask_vs_m10_drop_keep_mask"], ensure_ascii=False))
    if ds1cmp is None:
        print("  DS1 重現檢查：沒做（沒有 DS1 逐列檔）")
    else:
        mx = {}
        for c in ds1cmp:
            for k, v in c["absdiff"].items():
                mx[k] = max(mx.get(k, 0.0), v)
        print(f"  DS1 重現檢查：比了 {len(ds1cmp)} 格、逐值相同 {sum(c['exact'] for c in ds1cmp)} 格；最大絕對差 {mx}", flush=True)


def cmd_run(a):
    import torch
    import m10_drop as DR
    import m10_tasks as T
    from m10_quality import Answerer, plan_items
    if a.smoke:
        a.tasks, a.n, a.shard, a.full_range = ["T2"], 1, "0/1", True
        if a.check_ds1 is None:
            a.check_ds1 = DS1_RAW.format(model=a.model)
    wm, src = list(a.write_methods), "cli"
    if a.ds2_plan:
        wm = json.load(open(a.ds2_plan))["models"][a.model]["rank"]["B"]["top2"]
        src = f"ds2_plan:{os.path.relpath(a.ds2_plan, REPO)}"
    allowed_write = [m for m in DR.WRITE_TIME if m in DR.METHODS["B"]]
    if any(m not in allowed_write for m in wm):
        raise SystemExit(f"寫入時方法只能是 {allowed_write}：{wm}")
    extra = [m for m in a.extra_methods if m not in wm]
    if any(m not in DR.METHODS["B"] or m == "ORC" for m in extra):
        raise SystemExit(f"--extra-methods 只能是 m10_drop B 粒度的方法（ORC 另外固定做 50%）：{extra}")
    drop_methods = wm + extra
    roles = {m: "write" for m in wm}
    roles.update({m: ("qry" if m == "QRY" else "random" if m.startswith("RND") else "write" if m in DR.WRITE_TIME
                      else "position") for m in extra})
    fmts = resolve_formats(a.model, a.formats, a.fmt_choice)
    a.variants = ["region"] + (["full"] if a.full_range else [])
    ans = Answerer(a.model)
    P = json.load(open(a.params_file))["params"][a.model]
    ds1, ds1_paths = load_ds1(a.check_ds1, a.model) if a.check_ds1 else (None, [])
    if a.check_ds1:
        print(f"DS1 逐列檔 {len(ds1_paths)} 個、{len(ds1)} 列（{a.check_ds1}）", flush=True)
    si, sk = parse_shard(a.shard)
    done = done_items(a.out)
    fo, w = open_out(a.out, FIELDS)
    chk = open(side_path(a.out, ".checks.jsonl"), "a")
    watch = start_watch(a.out)
    rid = os.environ.get("RUN_ID", "adhoc")
    t_all, n_new, bad_mask = time.time(), 0, []
    try:
        for k, (task, b1, b2, seed) in enumerate(plan_items(a.tasks, a.n, a.seed0)):   # ＝m10_quality.make_items（DS1）的順序
            if k % sk != si or item_id_of(task, b1, b2, seed) in done:
                continue
            t0 = time.time()
            it = T.make_item(task, b1, b2, seed, ans.tok, P.get(task))
            assert it["item_id"] == item_id_of(task, b1, b2, seed), it["item_id"]
            rows, checks = run_item(ans, it, a, drop_methods, roles, fmts, src, mask_check=a.smoke or bool(a.check_ds1))
            hl, intr = host_level(), intruders(watch)
            for r in rows:
                r.update(run_id=rid, ts=ts(), shard=a.shard, n_rows_item=len(rows), own_gpu_intruders=intr, host_level=hl)
                w.writerow(r)
            fo.flush()
            ds1cmp = compare_ds1(ds1, it["item_id"], rows) if ds1 is not None else None
            chk.write(json.dumps(dict(run_id=rid, ts=ts(), model=a.model, item_id=it["item_id"], checks=checks, ds1=ds1cmp,
                                      own_gpu_intruders=intr, host_level=hl), ensure_ascii=False, default=str) + "\n")
            chk.flush()
            if not all(c["equal"] for c in checks["mask_vs_m10_drop_keep_mask"]):
                bad_mask.append(it["item_id"])
            n_new += 1
            print(f"[{k + 1}] {it['item_id']} rows={len(rows)} {time.time() - t0:.1f}s avg {(time.time() - t_all) / n_new:.1f}s/題 "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            if a.smoke:
                _print_smoke(rows, checks, ds1cmp)
            torch.cuda.empty_cache()
    finally:
        fo.close()
        chk.close()
        finish_watch(watch, a.out)
    if bad_mask:
        print(f"遮罩和 m10_drop.keep_mask 不同的題：{bad_mask[:10]}（程式錯誤，規則 2：停下來查）", flush=True)
        sys.exit(1)


# ============================================================================ report（CPU；不碰 torch）
L1_STOP_TEXT = {
    "G1": "三個位元組預算（約 1/2、1/4、1/8）下，最好的寫入時刪除法（DS1 選出的前 2 名）都沒有贏過同位元組的均勻低精度："
          "配對差的 CI 上界 ≤ 0，或贏不到 ε。這時寫「在本設定下，有損刪除被量化支配」，第三階段不再做刪除的放置策略。"
          "只有在 ORC／QRY 以外的方法贏過量化時才繼續。",
    "S2": "若 INT4 均勻在每個預算都支配丟棄 → 丟棄只在 < 25% 的預算才值得討論；反之則 drop 進入第三階段的動作空間。"
          "（experiment_matrix.md §3 S2 的成功／停止）",
}


def _complete(rows: list) -> tuple[list, list]:
    """同一題同一格重複時取最後一列；只留列數齊全的題。回傳 (列, 不齊全的題)。"""
    last = {}
    for r in rows:
        last[(r["model"], r["item_id"], r["variant"], r["group"], r["method"], r["attn_path"], r["k_blocks"])] = r
    rows = list(last.values())
    cnt, want = {}, {}
    for r in rows:
        key = (r["model"], r["item_id"])
        cnt[key] = cnt.get(key, 0) + 1
        want[key] = max(want.get(key, 0), int(r["n_rows_item"]))
    ok = {k for k in cnt if cnt[k] >= want[k]}
    return [r for r in rows if (r["model"], r["item_id"]) in ok], sorted(f"{m}/{i}" for m, i in set(cnt) - ok)


def _pair(dc: dict, qc: dict) -> dict:
    """以題目為單位的配對差（刪除 − 量化）。gold_lp、first_ok：正＝刪除比較好；kl：負＝刪除比較好。"""
    common = sorted(set(dc) & set(qc))
    return {m: boot(float(dc[i][m]) - float(qc[i][m]) for i in common) for m in METRICS}


def cmd_report(a):
    paths = sorted({p for g in a.raw for p in glob.glob(g)})
    rows, partial = _complete(read_rows(paths))
    rep = {"rule_text_L1": L1_STOP_TEXT, "eps_first_ok": EPS_FIRST_OK, "files": paths,
           "run_ids": sorted({r["run_id"] for r in rows}), "incomplete_items": partial,
           "contention": {"rows_with_intruders": sum(1 for r in rows if r["own_gpu_intruders"] not in ("0", "NOT_CHECKED")),
                          "rows_not_checked": sum(1 for r in rows if r["own_gpu_intruders"] == "NOT_CHECKED"),
                          "host_levels": sorted({r["host_level"] for r in rows})},
           "write_methods_source": sorted({r["write_methods_source"] for r in rows}),
           "pair_sign": "刪除 − 量化（同一題配對）：gold_lp、first_ok 正＝刪除比較好；kl 負＝刪除比較好",
           "G1_reading": "not_win＝(gold_lp 配對差 CI 上界 ≤ 0) 或 (first_ok 配對差平均 < ε)；某預算『被量化支配』＝該預算的量化格式裡"
                         "至少有一個，兩個寫入時方法都 not_win。只用名目配對（L1 X1 的表：對齊 BF16 比例、位元組照實列出）",
           "models": {}}
    L = ["# S2：載入區的有損刪除 vs 降精度，同位元組（L1 X1；teacher forcing、第 1 輪；只報告）", "",
         f"run：{'、'.join(rep['run_ids'])}；不齊全的題（沒用）：{len(partial)}；寫入時方法的來源：{'、'.join(rep['write_methods_source'])}",
         "", rep["pair_sign"] + "。95% CI＝以題目為單位 bootstrap 10,000 次。", ""]
    for m in sorted({r["model"] for r in rows}):
        R = [r for r in rows if r["model"] == m]
        M = rep["models"][m] = {"n_items": len({r["item_id"] for r in R})}

        def cells(variant, group, method, path, k=None):
            return {r["item_id"]: r for r in R if r["variant"] == variant and r["group"] == group and r["method"] == method
                    and r["attn_path"] == path and (k is None or int(r["k_blocks"]) == k)}

        # 1. 每個配置：對 E0、實際位元組
        cfgs = sorted({(r["variant"], r["group"], r["role"], r["method"], r["fmt"], r["attn_path"], int(r["k_blocks"]))
                       for r in R}, key=lambda c: (c[0], ["E0", "MASK0", "drop", "quant"].index(c[1]), c[3], c[5], c[6]))
        tab = []
        for var, grp, role, meth, fmt, path, k in cfgs:
            c = cells(var, grp, meth, path, k)
            tab.append(dict(variant=var, group=grp, role=role, method=meth, fmt=fmt, attn_path=path, k_blocks=k, n=len(c),
                            ratio_region=sorted({r["ratio_region"] for r in c.values()}),
                            ratio_total=sorted({r["ratio_total"] for r in c.values()}),
                            bits_per_value_region=sorted({r["bits_per_value_region"] for r in c.values()}),
                            budget_tags=sorted({t for r in c.values() for t in r["budget_tags"].split()}),
                            d_gold_lp=boot(float(r["gold_lp"]) - float(r["gold_lp_e0"]) for r in c.values()),
                            d_first_ok=boot(int(r["first_ok"]) - int(r["first_ok_e0"]) for r in c.values()),
                            kl_median=statistics.median(float(r["kl"]) for r in c.values()),
                            needle_kept=statistics.mean(float(r["needle_kept"]) for r in c.values()
                                                        if r["needle_kept"] != "") if grp == "drop" else None))
        M["configs"] = tab
        L += [f"## {m}（{M['n_items']} 題）", "", "### 每個配置（對 E0；位元組照實算）", "",
              "| 段 | 組 | 方法 | 路徑 | k | 區內位元組比 | 整段比 | 預算標籤 | Δgold_lp | Δfirst_ok（百分點） | KL 中位數 | 答案留下 |",
              "|:--|:--|:--|:--|--:|--:|--:|:--|--:|--:|--:|--:|"]
        for t in tab:
            nk = "" if t["needle_kept"] is None else f"{t['needle_kept']:.3f}"
            L.append(f"| {t['variant'] or '—'} | {t['group']} | {t['method']} | {t['attn_path']} | {t['k_blocks']} | "
                     f"{'／'.join(t['ratio_region'])} | {'／'.join(t['ratio_total'])} | {' '.join(t['budget_tags'])} | "
                     f"{fmt_ci(t['d_gold_lp'])} | {fmt_ci(t['d_first_ok'], 100, 1)} | {t['kl_median']:.4f} | {nk} |")
        # 2. 配對：刪除 − 量化
        M["pairs"] = {}
        for var in sorted({r["variant"] for r in R if r["variant"]}):
            lo, hi = REGION[var]
            nreg = hi - lo + 1
            write = sorted({r["method"] for r in R if r["variant"] == var and r["group"] == "drop" and r["role"] == "write"})
            others = [x for x in ("QRY", "RND0", "ORC") if any(r["method"] == x and r["group"] == "drop" for r in R)]
            quants = sorted({r["method"] for r in R if r["variant"] == var and r["group"] == "quant"})
            P = M["pairs"][var] = {"nominal": {}, "matched": {}}
            for tag, fr in NOMINAL:                                                         # L1 X1 的表（名目）
                k = int(round(fr * nreg))
                for q in NOMINAL_QUANT[tag]:
                    qc = cells(var, "quant", q, "mask", nreg)
                    if not qc:
                        continue
                    e = P["nominal"][f"{tag}|{q}"] = {"k_drop": k, "ratio_drop": round((nreg - k) / nreg, 6),
                                                      "ratio_quant": sorted({r["ratio_region"] for r in qc.values()}),
                                                      "by_method": {}}
                    for dm in write + others:
                        if dm == "ORC" and tag != "1/2":
                            continue
                        dc = cells(var, "drop", dm, "mask", k)
                        if dc:
                            e["by_method"][dm] = _pair(dc, qc)
                    if write:
                        means = {dm: e["by_method"][dm]["gold_lp"]["mean"] for dm in write
                                 if dm in e["by_method"] and e["by_method"][dm]["gold_lp"] is not None}
                        best = max(means, key=means.get) if means else None
                        e["best_write_time"] = dict(method=best, post_hoc_selection=True,
                                                    pair=e["by_method"].get(best) if best else None)
            for q in quants:                                                                # 照實等位元組（補充）
                ks = {int(r["k_blocks"]) for r in R if r["variant"] == var and r["group"] == "drop"
                      and f"matched:{q}" in r["budget_tags"].split()}
                qc = cells(var, "quant", q, "mask", nreg)
                if len(ks) != 1 or not qc:
                    continue
                k = ks.pop()
                e = P["matched"][q] = {"k_drop": k, "ratio_drop": round((nreg - k) / nreg, 6),
                                       "ratio_quant": sorted({r["ratio_region"] for r in qc.values()}), "by_method": {}}
                for dm in write + [x for x in others if x != "ORC"]:
                    dc = cells(var, "drop", dm, "mask", k)
                    if dc:
                        e["by_method"][dm] = _pair(dc, qc)
            L += ["", f"### 配對：刪除 − 量化（{var}：block {lo}–{hi}；量化走 mask 路徑，和刪除同一條）", "",
                  "| 配對 | 刪除 k（區內比） | 量化區內比 | 方法 | Δgold_lp | Δfirst_ok（百分點） | ΔKL |", "|:--|:--|:--|:--|--:|--:|--:|"]
            for kind in ("nominal", "matched"):
                for name, e in P[kind].items():
                    for dm, pr in e["by_method"].items():
                        L.append(f"| {kind}:{name} | {e['k_drop']}（{e['ratio_drop']}） | {'／'.join(e['ratio_quant'])} | {dm} | "
                                 f"{fmt_ci(pr['gold_lp'])} | {fmt_ci(pr['first_ok'], 100, 1)} | {fmt_ci(pr['kl'])} |")
        # 3. L1 G1 的成分（只用載入區、名目配對）
        G = {}
        pr = M["pairs"].get("region", {}).get("nominal", {})
        for tag, _ in NOMINAL:
            per_q = {}
            for q in NOMINAL_QUANT[tag]:
                e = pr.get(f"{tag}|{q}")
                if not e:
                    continue
                nw = {}
                for dm, p in e["by_method"].items():
                    if p["gold_lp"] is None or p["first_ok"] is None:
                        continue
                    nw[dm] = dict(not_win=bool(p["gold_lp"]["hi"] <= 0 or p["first_ok"]["mean"] < EPS_FIRST_OK),
                                  gold_lp_ci_hi_le_0=bool(p["gold_lp"]["hi"] <= 0),
                                  first_ok_mean_lt_eps=bool(p["first_ok"]["mean"] < EPS_FIRST_OK),
                                  drop_significantly_better_gold_lp=bool(p["gold_lp"]["lo"] > 0))
                wr = [dm for dm in nw if any(r["method"] == dm and r["role"] == "write" for r in R)]
                per_q[q] = dict(methods=nw, all_write_time_not_win=all(nw[dm]["not_win"] for dm in wr) if wr else None)
            G[tag] = dict(per_quant=per_q, dominated_by_quant=any(v["all_write_time_not_win"] for v in per_q.values())
                          if per_q else None)
        doms = [v["dominated_by_quant"] for v in G.values()]
        M["G1_components"] = dict(by_budget=G, all_budgets_dominated=(all(doms) if doms and None not in doms else None),
                                  mask0_minus_e0=next((t for t in tab if t["group"] == "MASK0"), None))
        L += ["", "### L1 G1 停止條件的成分（載入區、名目配對；判讀由人）", "", "```",
              json.dumps(M["G1_components"], ensure_ascii=False, indent=1, default=str), "```", ""]
    L += ["## L1／矩陣的停止條件原文", ""] + [f"- **{k}**：{v}" for k, v in L1_STOP_TEXT.items()]
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(rep, open(os.path.join(a.out_dir, "s2_report.json"), "w"), indent=1, ensure_ascii=False, default=str)
    open(os.path.join(a.out_dir, "tables_s2.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:80]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("run")
    p.add_argument("--model", choices=["llama", "qwen"], required=True)
    p.add_argument("--params-file", default=PARAMS)
    p.add_argument("--tasks", nargs="+", default=["T2", "T3", "T4", "T5"])
    p.add_argument("--n", type=int, default=24, help="每個任務幾題（DS1：24）")
    p.add_argument("--seed0", type=int, default=60000, help="DS1 的題目 seed（14 §8.4）")
    p.add_argument("--write-methods", nargs="+", default=["SNAP", "KEYDIFF"],
                   help="DS1 選出的前 2 名寫入時方法（DS1 還沒跑完時的預設：SNAP KEYDIFF）")
    p.add_argument("--ds2-plan", help=f"讀 DS1 pick 的 B 粒度 top2（例 {os.path.relpath(DS2_PLAN, REPO)}）；給了就蓋過 --write-methods")
    p.add_argument("--extra-methods", nargs="+", default=["QRY", "RND0"])
    p.add_argument("--formats", nargs="+", default=list(DEFAULT_FORMATS), help="FP8＝fmt_choice.json 這個模型選定的")
    p.add_argument("--fmt-choice", default=FMT_CHOICE)
    p.add_argument("--quant-paths", nargs="+", default=["mask", "sdpa"], choices=["mask", "sdpa"])
    p.add_argument("--no-matched", dest="matched", action="store_false", help="不做照實等位元組的刪除 k")
    p.add_argument("--full-range", action="store_true", help="另做全段版（block 1–63）")
    p.add_argument("--check-ds1", help="DS1 逐列檔的萬用字元：E0、MASK0、全段丟 50%%／75%% 對 DS1 逐值比對（寫進 .checks.jsonl）")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--out", required=True)
    p.add_argument("--smoke", action="store_true", help="1 題（DS1 第一題）、全段版、印每個配置、DS1 重現檢查")
    r = sp.add_parser("report")
    r.add_argument("--raw", nargs="+", required=True)
    r.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args()
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
