#!/usr/bin/env python3
"""a0.py — 第三階段 A0（＝G-M01/02）：縮圖還原的上界。判準與量法：docs/stage3_design_20261024/runs/A0_progressive_restore.md（登記，不改）。

情境：一段很長的對話暫停、KV 被卸到慢層；使用者回來問新問題。讀回時每個區內 block 都讀 16–32 倍的「粗底」（xKV＋量化係數，
事先校準的基底），只有新問題挑中的 m 個 block 升級成 INT4 或 BF16。和「每塊同樣壓縮」在同樣的讀取位元組下比答對率。

子命令（GPU 的才 import torch；本檔頂層只有標準函式庫與 m16_xstack（同樣沒有 torch））：
  plan        CPU：題組（三組 seed）與每個判定格的 TF 配置清單 → results/s3_h200/a0/plan_<cell>.json；檢查 seed 不和既有題組重疊
  calib       GPU：xKV 校準基底（m16_xstack.cmd_calib_xl 的同一段程式；qwen14 另外補模型資訊）
  tf          GPU：TF 篩選（每題一個 prefill、約 188 個配置）
  select-gen  CPU：TF → 生成配置（登記 §1.6，寫死）→ gen_selection_<cell>.json
  gen         GPU：生成答對率（單輪 192＋第 2 輪 96）
  report      CPU：表格與判定（登記 §2）
  gpu-selftest GPU：組 KV 與 E0 逐位元相同、逐層 forward 與 KVModel 逐位元相同、位元組帳、挑法的形狀
  --self-test CPU：配置、位元組帳、同位元組配對、生成配置的挑法、判定邏輯（合成資料）
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
from datetime import datetime
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
REPO = os.path.dirname(CODE)
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)
import m16_xstack as XS  # noqa: E402

CHUNK = 512
OUT = os.path.join(REPO, "results", "s3_h200", "a0")
RAW = os.path.join(OUT, "raw")
LOCAL = os.environ.get("S3_LOCAL", "/tmp/hungweilai-s3")          # 校準基底等大檔（hgx-cn01 本地，不進 /work 配額）
PARAMS = XS.PARAMS if hasattr(XS, "PARAMS") else os.path.join(REPO, "results", "m10_pilot_mi300x", "e0_tune", "CANDIDATE_PARAMS.json")

INFO = {"llama": dict(L=32, H=8, Hq=32, D=128), "qwen": dict(L=28, H=4, Hq=28, D=128), "qwen14": dict(L=48, H=8, Hq=40, D=128)}
PARAMS_OF = {"llama": "llama", "qwen": "qwen", "qwen14": "qwen"}                 # 登記 §1.1：14B 沿用 7B 的題目參數
WRITE_METHOD = {"llama": "KNORM", "qwen": "KEYDIFF", "qwen14": "KEYDIFF"}       # 登記 §1.4（C1b 的選擇）
CELLS = {   # 判定格（登記 §1.1）；ctl＝32K 對照列（只跑 TF、不判定）
    "L128": dict(model="llama", nb=252, judged=True, pair="Q128"),
    "Q128": dict(model="qwen", nb=252, judged=True, pair="L128"),
    "Q256": dict(model="qwen", nb=504, judged=True, pair="Q14-256"),
    "Q14-256": dict(model="qwen14", nb=504, judged=True, pair="Q256"),
    "L32": dict(model="llama", nb=64, judged=False, pair=None),
    "Q32": dict(model="qwen", nb=64, judged=False, pair=None),
}
SETS = {"tf": dict(seed0=71000, n=24, tasks=("T2", "T3", "T4", "T5"), turn=0),
        "gen1": dict(seed0=72000, n=48, tasks=("T2", "T3", "T4", "T5"), turn=0),
        "gen2": dict(seed0=73000, n=32, tasks=("T2", "T3", "T4"), turn=1)}
USED_SEEDS = {"x1": (60000, 24), "calib": (61000, 2), "x2": (62000, 24), "x2b": (63000, 48), "x4a": (64000, 24), "x4b": (65000, 48),
              "xlcal3": (66000, 48), "p11": (67000, 48), "c1a": (70000, 45), "lc128k": (40000, 12)}

BASES = {"B16": "xl:k20:v30:cal+q:int4_kivi", "B24": "xl:k17:v26:cal+q:kv-int4_kivi-int2_kivi",
         "B32": "xl:k13:v20:cal+q:kv-int4_kivi-int2_kivi"}
GRID = ("q:int4_kivi", "q:int2_kivi", "xl:k40:v60:cal+q:int4_kivi", "xl:k27:v40:cal+q:int4_kivi", BASES["B16"],
        "xl:k16:v24:cal+q:int4_kivi", BASES["B24"], BASES["B32"])
RHOS = ("1/16", "1/8", "1/4")
TARGETS = ("int4", "bf16")
CORE_SEL = ("ORACLE", "ORACLE-L", "EXACT-Q", "BASE-ATTN", "BASE-ATTN-L", "ONE-PASS")
EXTRA_SEL = ("QUEST-512", "KSCAN-2b", "WRITE-TIME", "RANDOM", "RECENT", "SEQ")
DIAG_SEL = ("ORACLE-NEEDLE",)             # 偏離 2：知道答案位置的對照（只報告、不參與 GA0 的判定）
EXTRA_BASES = ("B16", "B32")
DROP_SEL = ("ORACLE", "ORACLE-L", "BASE-ATTN")
DROP_BASE = "B32"                         # 丟塊對照的 BASE-ATTN 用 B32 第一趟的挑法
GRAN = {"ORACLE": "global", "ORACLE-L": "layer", "EXACT-Q": "global", "BASE-ATTN": "global", "BASE-ATTN-L": "layer",
        "ONE-PASS": "layer", "QUEST-512": "global", "KSCAN-2b": "global", "WRITE-TIME": "global", "RANDOM": "global",
        "RECENT": "global", "SEQ": "global", "ORACLE-NEEDLE": "global"}
ORACLE_SEL = ("ORACLE", "ORACLE-L", "EXACT-Q", "ORACLE-NEEDLE")                                   # 不可部署
GO_F = 1 / 12
BYTES_TOL = 1.05
EPS = 2.0                                     # 百分點
N_BOOT = 10000
MATCH_RK = range(4, 65)
MATCH_FMTS = ("int4_kivi", "kv-int4_kivi-int2_kivi")
MATCH_PLAIN = ("q:int4_kivi", "q:int2_kivi", "q:int8_token")


def run_id() -> str:
    return os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"


def ts() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ============================================================================ 題目
def item_id_of(task, b1, b2, seed) -> str:
    """＝m10_tasks.make_item 的 item_id（出題後再 assert）。"""
    tb = ["global"] if task == "T5" else [b1] + ([b2] if b2 else [])
    return f"{task}_{'-'.join(tb)}_s{seed}"


def plan_set(name: str) -> list:
    s = SETS[name]
    return [dict(task=t, b1=b1, b2=b2, seed=seed, item_id=item_id_of(t, b1, b2, seed), turn=s["turn"], set=name)
            for t, b1, b2, seed in XS.plan_interleaved_cpu(list(s["tasks"]), s["n"], s["seed0"])]


def seed_overlap() -> dict:
    ours = {(r["task"], r["seed"]) for k in SETS for r in plan_set(k)}
    out = {}
    for k, (s0, n) in USED_SEEDS.items():
        used = {(t, s) for t, _, _, s in XS.plan_interleaved_cpu(["T2", "T3", "T4", "T5"], n, s0)}
        out[k] = len(ours & used)
    sets = {k: {(r["task"], r["seed"]) for r in plan_set(k)} for k in SETS}
    out["tf∩gen1"] = len(sets["tf"] & sets["gen1"])
    out["tf∩gen2"] = len(sets["tf"] & sets["gen2"])
    out["gen1∩gen2"] = len(sets["gen1"] & sets["gen2"])
    return out


# ============================================================================ 位元組
def info_of(model: str) -> dict:
    return INFO[model]


def block_bytes(info: dict, target: str) -> int:
    """一個 block（所有層、K＋V）用精修格式存的位元組。"""
    shape = (info["L"], 2, info["H"], CHUNK, info["D"])
    if target == "bf16":
        return info["L"] * 2 * info["H"] * CHUNK * info["D"] * 2
    if target == "int4":
        return XS.full_fmt_bytes("int4_kivi", shape)
    raise ValueError(target)


def spec_bytes(spec: str, info: dict, nreg: int) -> dict:
    """均勻配置的位元組（predict_bytes；shared.* 不算進請求）。"""
    st = XS.validate(XS.parse_spec(spec), info, nreg)
    by = XS.predict_bytes(st, info, nreg)
    req = sum(v for k, v in by.items() if not k.startswith("shared."))
    return dict(detail=by, request=req, shared=sum(v for k, v in by.items() if k.startswith("shared.")))


def index_bytes(selector: str, info: dict, nreg: int) -> int:
    L, H, D = info["L"], info["H"], info["D"]
    if selector == "QUEST-512":
        return nreg * 2 * L * H * D * 2                        # 每塊、每（層, KV head）的逐維 min、max（BF16）
    if selector == "KSCAN-2b":
        return XS.half_bytes("int2h_g32", (L, H, nreg * CHUNK, D), "k")
    if selector == "WRITE-TIME":
        return (nreg + 7) // 8
    return 0


def m_of(rho: str, nreg: int) -> int:
    return int(round(float(Fraction(rho)) * nreg))


def region_bf16(info: dict, nreg: int) -> int:
    return nreg * block_bytes(info, "bf16")


# ============================================================================ 配置
def cfg_uniform(spec: str, info: dict, nreg: int, role: str) -> dict:
    b = spec_bytes(spec, info, nreg)
    return dict(cid=f"U|{spec}", kind="uniform", spec=spec, roles=[role], deployable=True,
                bytes_base=b["request"], bytes_refine=0, bytes_index=0, bytes_shared=b["shared"],
                f=b["request"] / region_bf16(info, nreg))


def cfg_prog(base: str, rho: str, target: str, sel: str, info: dict, nreg: int, role: str) -> dict:
    m = m_of(rho, nreg)
    bb = spec_bytes(BASES[base], info, nreg)
    br = m * block_bytes(info, target)
    bi = index_bytes(sel, info, nreg)
    return dict(cid=f"P|{base}|{rho}|{target}|{sel}", kind="prog", base=base, spec=BASES[base], rho=rho, m=m, target=target,
                selector=sel, gran=GRAN[sel], roles=[role], deployable=sel not in ORACLE_SEL,
                bytes_base=bb["request"], bytes_refine=br, bytes_index=bi, bytes_shared=bb["shared"],
                f=(bb["request"] + br + bi) / region_bf16(info, nreg))


def cfg_drop(rho: str, target: str, sel: str, info: dict, nreg: int) -> dict:
    m = m_of(rho, nreg)
    br = m * block_bytes(info, target)
    bi = index_bytes(sel, info, nreg) + (nreg + 7) // 8
    return dict(cid=f"D|{rho}|{target}|{sel}", kind="drop", base=DROP_BASE if sel == "BASE-ATTN" else "", rho=rho, m=m,
                target=target, selector=sel, gran=GRAN[sel], roles=["drop"], deployable=sel not in ORACLE_SEL,
                bytes_base=0, bytes_refine=br, bytes_index=bi, bytes_shared=0, f=(br + bi) / region_bf16(info, nreg))


def match_candidates(info: dict, nreg: int) -> list:
    """同位元組配對的候選（登記 §1.5 第 3 點）：(f, spec)，f 由小到大、平手照字串。"""
    out = []
    for fmt in MATCH_FMTS:
        for rk in MATCH_RK:
            spec = f"xl:k{rk}:v{round(1.5 * rk)}:cal+q:{fmt}"
            try:
                out.append((spec_bytes(spec, info, nreg)["request"] / region_bf16(info, nreg), spec))
            except XS.SpecError:
                continue
    for spec in MATCH_PLAIN:
        out.append((spec_bytes(spec, info, nreg)["request"] / region_bf16(info, nreg), spec))
    return sorted(out)


def matched_uniform(fP: float, cands: list):
    """位元組 ≤ f_P 的最大者（沒有就 None）。"""
    best = None
    for f, spec in cands:
        if f <= fP + 1e-12:
            best = (f, spec)
    return best


# ---- M05（runs/M05_duo_heads.md）：DuoAttention 的頭表（只有 Llama-3.1-8B-Instruct 有釋出）；只加進 L128 的生成
M05_P = (0.25, 0.5, 0.75)
M05_DIR = os.path.join(OUT.replace(os.path.join("s3_h200", "a0"), os.path.join("s3_h200", "m05")), "duo_patterns",
                       "Meta-Llama-3.1-8B-Instruct")
M05_SINK, M05_RECENT = 128, 256                        # 頭表的 config.json（sink_size、recent_size）


def m05_pattern() -> list:
    """[L][H] 的分數（越大越像檢索頭）。"""
    rows = [[float(x) for x in l.split()] for l in open(os.path.join(M05_DIR, "full_attention_heads.tsv")) if l.strip()]
    return rows


def m05_streaming(p: float) -> list:
    """分數最低的 round(p·L·H) 個（層, KV head）＝串流頭（只看 sink＋最近 recent 個 token）；平手照（層, head）順序。"""
    pat = m05_pattern()
    flat = sorted(((v, l, h) for l, r in enumerate(pat) for h, v in enumerate(r)))
    k = int(round(p * len(flat)))
    return sorted((l, h) for _, l, h in flat[:k])


def m05_configs(model: str, nreg: int) -> list:
    if model != "llama":
        return []
    info = info_of(model)
    out = []
    for p in M05_P:
        st = m05_streaming(p)
        frac = len(st) / (info["L"] * info["H"])
        keep_tok = min(nreg * CHUNK, M05_RECENT)                    # 串流頭在區內只讀最近 recent 個 token（sink 在 block 0 裡）
        f = (1 - frac) + frac * keep_tok / (nreg * CHUNK)
        out.append(dict(cid=f"H|duo|{p}", kind="duo", p=p, n_streaming=len(st), roles=["M05"], deployable=True,
                        bytes_base=int(round(f * region_bf16(info, nreg))), bytes_refine=0, bytes_index=0, bytes_shared=0, f=f))
    return out


def tf_configs(model: str, nreg: int) -> list:
    """一題的 TF 配置（登記 §1.5），順序＝執行順序（同一個粗底的放在一起）。"""
    info = info_of(model)
    cfgs = [dict(cid="E0", kind="E0", roles=["ref"], deployable=True, bytes_base=0, bytes_refine=0, bytes_index=0,
                 bytes_shared=0, f=1.0)]
    seen = {"E0"}

    def add(c):
        if c["cid"] in seen:
            next(x for x in cfgs if x["cid"] == c["cid"])["roles"].extend(r for r in c["roles"] if r not in
                                                                            next(x for x in cfgs if x["cid"] == c["cid"])["roles"])
            return
        seen.add(c["cid"])
        cfgs.append(c)
    for spec in GRID:
        add(cfg_uniform(spec, info, nreg, "grid"))
    cands = match_candidates(info, nreg)
    for base in BASES:
        for rho in RHOS:
            for tgt in TARGETS:
                fP = cfg_prog(base, rho, tgt, "ORACLE", info, nreg, "core")["f"]
                mu = matched_uniform(fP, cands)
                if mu is not None:
                    add(cfg_uniform(mu[1], info, nreg, f"match:{base}|{rho}|{tgt}"))
    for base in BASES:                                    # 同一個粗底的漸進配置連在一起（GPU 端一次只留一個粗底）
        for rho in RHOS:
            for tgt in TARGETS:
                for sel in CORE_SEL:
                    add(cfg_prog(base, rho, tgt, sel, info, nreg, "core"))
                for sel in DIAG_SEL:
                    add(cfg_prog(base, rho, tgt, sel, info, nreg, "diag"))
        if base in EXTRA_BASES:
            for rho in RHOS:
                for sel in EXTRA_SEL:
                    add(cfg_prog(base, rho, "int4", sel, info, nreg, "extra"))
    for rho in RHOS:
        for tgt in TARGETS:
            for sel in DROP_SEL:
                add(cfg_drop(rho, tgt, sel, info, nreg))
    return cfgs


def cmd_plan(a):
    ov = seed_overlap()
    if any(ov.values()):
        raise SystemExit(f"題組 seed 重疊：{ov}（規則 2：停）")
    os.makedirs(OUT, exist_ok=True)
    for cell, c in CELLS.items():
        nreg = c["nb"] - 1
        cfgs = tf_configs(c["model"], nreg)
        doc = dict(run_id=run_id(), ts=ts(), cell=cell, model=c["model"], n_blocks=c["nb"], nreg=nreg, n_configs=len(cfgs),
                   sets={k: dict(SETS[k], tasks=list(SETS[k]["tasks"]), n_items=len(plan_set(k))) for k in SETS},
                   seed_overlap=ov, configs=cfgs)
        p = os.path.join(OUT, f"plan_{cell}.json")
        json.dump(doc, open(p, "w"), indent=1, ensure_ascii=False)
        kinds = {}
        for x in cfgs:
            kinds[x["kind"]] = kinds.get(x["kind"], 0) + 1
        go = sum(1 for x in cfgs if x["kind"] == "prog" and x["selector"] == "ORACLE" and x["f"] <= GO_F)
        print(f"{cell}: {len(cfgs)} 個配置 {kinds}；ORACLE 且 f ≤ 1/12：{go} → {p}")
    print("seed 重疊：", ov)


# ============================================================================ CSV
TF_FIELDS = ["run_id", "ts", "calib_run_id", "cell", "model", "n_blocks", "item_id", "task", "seed", "band", "turn", "cid", "kind", "spec", "base",
             "rho", "m", "target", "selector", "gran", "deployable", "bytes_base", "bytes_refine", "bytes_index", "bytes_shared",
             "bytes_region_bf16", "f", "multiple", "gold_lp", "gold_lp_e0", "d_gold_lp", "kl", "top1", "first_ok", "needle_cover",
             "sel_hash", "t_cfg_s", "t_prefill_s", "own_gpu_intruders", "host_level", "n_rows_item"]
GEN_FIELDS = ["run_id", "ts", "calib_run_id", "cell", "model", "n_blocks", "item_id", "task", "seed", "band", "turn", "set", "cid", "kind", "spec",
              "base", "rho", "m", "target", "selector", "gran", "deployable", "bytes_base", "bytes_refine", "bytes_index",
              "bytes_shared", "bytes_region_bf16", "f", "multiple", "score", "score_e0", "stop", "n_gen", "gen_sha", "same_as_e0",
              "needle_cover", "sel_hash", "t_cfg_s", "t_prefill_s", "own_gpu_intruders", "host_level", "n_rows_item"]


def _reader(path):
    return gzip.open(path, "rt", newline="") if path.endswith(".gz") else open(path, newline="")


def read_rows(paths) -> list:
    rows = []
    for p in paths:
        try:
            with _reader(p) as f:
                for r in csv.DictReader(f):
                    if None in r or None in r.values() or r.get("run_id") in (None, "run_id"):
                        continue
                    rows.append(r)
        except EOFError as ex:                       # 寫到一半被砍的 .gz：讀得到的照用，並說出來
            print(f"[read_rows] {p}: {ex!r}（只用讀得到的列）", file=sys.stderr, flush=True)
    return rows


def done_items(paths) -> set:
    """續跑：一題算完＝那一題的列數 ≥ n_rows_item（寫到一半被砍的題整題重做；report 去重）。"""
    cnt, want = {}, {}
    for r in read_rows(paths):
        k = (r["item_id"], r.get("turn", ""))
        cnt[k] = cnt.get(k, 0) + 1
        if str(r.get("n_rows_item", "")).isdigit():
            want[k] = int(r["n_rows_item"])
    return {k for k, c in cnt.items() if k in want and c >= want[k]}


def out_glob(kind: str, cell: str) -> str:
    return os.path.join(RAW, f"a0_{kind}_{cell}_s*.csv.gz")


def open_out(path: str, fields: list):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    fo = gzip.open(path, "at", newline="")
    w = csv.DictWriter(fo, fieldnames=fields)
    if new:
        w.writeheader()
        fo.flush()
    return fo, w


def parse_shard(s: str):
    i, n = (int(x) for x in s.split("/"))
    if not 0 <= i < n:
        raise ValueError(s)
    return i, n


def pending(set_names, shard: str, paths) -> list:
    si, sk = parse_shard(shard)
    done = done_items(paths)
    items = [r for name in set_names for r in plan_set(name)]
    return [r for k, r in enumerate(items) if k % sk == si and (r["item_id"], str(r["turn"])) not in done]


def cmd_pending(a):
    paths = sorted(glob.glob(out_glob(a.kind, a.cell)))
    sets = ["tf"] if a.kind == "tf" else ["gen1", "gen2"]
    print(len(pending(sets, a.shard, paths)))


# ============================================================================ 統計
def boot(values, n_boot: int = N_BOOT, seed: int = 0):
    import numpy as np
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return None
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(1)
    return dict(mean=float(v.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n=int(v.size))


def table(rows: list, value: str) -> dict:
    """cid → {item_key: value}（重複的列取第一個；report 前去重）。"""
    out = {}
    for r in rows:
        k = (r["item_id"], r["turn"])
        out.setdefault(r["cid"], {}).setdefault(k, float(r[value]))
    return out


def paired(tab: dict, a: str, b: str, scale: float = 1.0):
    ka, kb = tab.get(a, {}), tab.get(b, {})
    keys = sorted(set(ka) & set(kb))
    return boot([(ka[k] - kb[k]) * scale for k in keys]) if keys else None


# ============================================================================ select-gen（登記 §1.6）
def g1_keys(tf_rows: list, cfgs: list) -> list:
    """登記 §1.6 G1：ORACLE／ORACLE-L、f ≤ 1/12，依 TF 平均 Δgold_lp 取前 3 個不同的 (粗底, ρ, 精修)（平手取 f 小的）。"""
    tab = table(tf_rows, "d_gold_lp")
    mean = {cid: statistics.fmean(v.values()) for cid, v in tab.items() if v}
    cand = []
    for c in cfgs:
        if c["kind"] == "prog" and c["selector"] in ("ORACLE", "ORACLE-L") and c["f"] <= GO_F + 1e-12 and c["cid"] in mean:
            cand.append((mean[c["cid"]], -c["f"], (c["base"], c["rho"], c["target"])))
    cand.sort(reverse=True)
    g1 = []
    for _, _, key in cand:
        if key not in g1:
            g1.append(key)
        if len(g1) == 3:
            break
    return g1


def select_gen(tf_rows: list, cfgs: list, pair_g1: list | None = None) -> dict:
    """TF → 生成配置。回傳 dict(configs=[cid...], why={cid: [理由]}, g1=[(base, rho, target)], g4=[...])。
    pair_g1：同長度另一個判定格的 G1（登記 §1.6 G4；GO 的「另一個模型方向相同」要用同一個配置）。"""
    by = {c["cid"]: c for c in cfgs}
    tab = table(tf_rows, "d_gold_lp")
    mean = {cid: statistics.fmean(v.values()) for cid, v in tab.items() if v}
    g1 = g1_keys(tf_rows, cfgs)
    out, why = [], {}

    def put(cid, reason):
        if cid not in by:
            raise KeyError(cid)
        if cid not in out:
            out.append(cid)
        why.setdefault(cid, []).append(reason)
    put("E0", "G3 錨點")
    put("U|q:int4_kivi", "G3 錨點")
    uni = [(mean[c["cid"]], -c["f"], c["cid"]) for c in cfgs if c["kind"] == "uniform" and c["f"] <= GO_F + 1e-12 and c["cid"] in mean]
    if uni:
        put(max(uni)[2], "G3 TF 最好的均勻（f ≤ 1/12）")
    for i, (base, rho, tgt) in enumerate(g1):
        for sel in ("ORACLE", "ORACLE-L", "BASE-ATTN", "BASE-ATTN-L", "ONE-PASS"):
            put(f"P|{base}|{rho}|{tgt}|{sel}", f"G2 #{i + 1}")
        mu = next((c["cid"] for c in cfgs if c["kind"] == "uniform" and f"match:{base}|{rho}|{tgt}" in c["roles"]), None)
        if mu:
            put(mu, f"G2 #{i + 1} 同位元組配對")
        if i == 0:
            put(f"D|{rho}|{tgt}|ORACLE", "G2 #1 丟塊對照")
    g4 = [tuple(k) for k in (pair_g1 or []) if tuple(k) not in g1]
    for base, rho, tgt in g4:
        put(f"P|{base}|{rho}|{tgt}|ORACLE", "G4 另一格的 G1")
        mu = next((c["cid"] for c in cfgs if c["kind"] == "uniform" and f"match:{base}|{rho}|{tgt}" in c["roles"]), None)
        if mu:
            put(mu, "G4 同位元組配對")
    return dict(configs=out, why=why, g1=[list(k) for k in g1], g4=[list(k) for k in g4])


def cmd_select_gen(a):
    plan = json.load(open(os.path.join(OUT, f"plan_{a.cell}.json")))
    rows = read_rows(sorted(glob.glob(out_glob("tf", a.cell))))
    n_items = len({(r["item_id"], r["turn"]) for r in rows if r["cid"] == "E0"})
    want = len(plan_set("tf"))
    if n_items < want and not a.allow_partial:
        raise SystemExit(f"{a.cell}：TF 只有 {n_items}／{want} 題做完 → 不挑（要挑就加 --allow-partial，會寫進檔案）")
    pair, pair_g1, pair_note = CELLS[a.cell]["pair"], None, ""
    if pair:
        prows = read_rows(sorted(glob.glob(out_glob("tf", pair))))
        pn = len({(r["item_id"], r["turn"]) for r in prows if r["cid"] == "E0"})
        if pn < want and not a.pair_missing:
            raise SystemExit(f"{a.cell}：同長度的另一格 {pair} 的 TF 只有 {pn}／{want} 題 → 等它做完（登記 §1.6 G4）；"
                             f"另一格量不到時加 --pair-missing（GO 在這個長度就不能成立）")
        if pn >= want:
            pair_g1 = g1_keys(prows, json.load(open(os.path.join(OUT, f"plan_{pair}.json")))["configs"])
        else:
            pair_note = f"{pair} 的 TF 只有 {pn}／{want} 題（--pair-missing）"
    sel = select_gen(rows, plan["configs"], pair_g1)
    doc = dict(run_id=run_id(), ts=ts(), cell=a.cell, tf_items=n_items, tf_items_planned=want, partial=n_items < want,
               pair=pair, pair_note=pair_note, rule="登記 §1.6（G1–G4）", **sel)
    p = os.path.join(OUT, f"gen_selection_{a.cell}.json")
    if os.path.exists(p) and not a.force:
        old = json.load(open(p))
        if old.get("configs") != doc["configs"]:
            raise SystemExit(f"{p} 已存在而內容不同 → 停（要重挑加 --force）")
    json.dump(doc, open(p, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(dict(cell=a.cell, g1=doc["g1"], n=len(doc["configs"])), ensure_ascii=False), "→", p)


# ============================================================================ 判定（登記 §2）
def judge_cell(gen_rows: list, cfgs: list, sel: dict) -> dict:
    """一個判定格：每個 ORACLE／ORACLE-L 配置的條件 1–3、SEL、DROP、G-M02 的成分。答對率差以百分點表示。"""
    by = {c["cid"]: c for c in cfgs}
    tab = table(gen_rows, "score")
    acc = {cid: 100 * statistics.fmean(v.values()) for cid, v in tab.items() if v}
    uni = [cid for cid in sel["configs"] if by[cid]["kind"] == "uniform" and cid in tab]
    res = {}
    for cid in sel["configs"]:
        c = by[cid]
        if c["kind"] != "prog" or c["selector"] not in ("ORACLE", "ORACLE-L") or cid not in tab:
            continue
        ok_u = [u for u in uni if by[u]["f"] <= BYTES_TOL * c["f"] + 1e-12]
        ustar = max(ok_u, key=lambda u: (acc[u], -by[u]["f"])) if ok_u else None
        d_e0 = paired(tab, cid, "E0", 100)
        d_u = paired(tab, cid, ustar, 100) if ustar else None
        r = dict(f=c["f"], acc=acc[cid], ustar=ustar, acc_ustar=acc.get(ustar), d_e0=d_e0, d_ustar=d_u,
                 c1=c["f"] <= GO_F + 1e-12, c2=bool(d_e0 and d_e0["lo"] >= -EPS),
                 c3=bool(d_u and d_u["mean"] >= EPS and d_u["lo"] > 0), c3_point=bool(d_u and d_u["mean"] >= EPS))
        key = (c["base"], c["rho"], c["target"])
        dep = {}
        for s in ("BASE-ATTN", "BASE-ATTN-L", "ONE-PASS"):
            dc = f"P|{key[0]}|{key[1]}|{key[2]}|{s}"
            if dc in acc and ustar and acc[cid] != acc[ustar]:
                dep[s] = dict(acc=acc[dc], share=(acc[dc] - acc[ustar]) / (acc[cid] - acc[ustar]), d_oracle=paired(tab, dc, cid, 100))
        r["deployable"] = dep
        dcid = f"D|{key[1]}|{key[2]}|{c['selector']}"
        r["drop"] = paired(tab, cid, dcid, 100) if dcid in tab else None
        res[cid] = r
    return dict(acc=acc, configs=res)


def gm02(gen_rows: list, sel: dict) -> dict:
    tab = table(gen_rows, "score")
    out = []
    for base, rho, tgt in sel["g1"]:
        o, one, b = (f"P|{base}|{rho}|{tgt}|{s}" for s in ("ORACLE-L", "ONE-PASS", "BASE-ATTN-L"))
        if not all(x in tab for x in (o, one, b)):
            continue
        a = {x: 100 * statistics.fmean(tab[x].values()) for x in (o, one, b)}
        gap_one, gap_base = a[o] - a[one], a[o] - a[b]
        out.append(dict(key=[base, rho, tgt], gap_one=gap_one, gap_base=gap_base, one_better=gap_one < gap_base,
                        d_one_minus_base=paired(tab, one, b, 100)))
    n_better = sum(1 for x in out if x["one_better"])
    return dict(rows=out, n_one_better=n_better, verdict=("SUPPORTED" if n_better >= 2 else "NOT_SUPPORTED") if out else "NOT_MEASURED")


def verdict(cells: dict) -> dict:
    """cells：cell → judge_cell(...) 的結果。GO／WEAK／STOP（登記 §2）。"""
    go, weak, any_c3 = [], [], False
    for cell, jc in cells.items():
        pair = CELLS[cell]["pair"]
        for cid, r in jc["configs"].items():
            any_c3 |= r["c3_point"]
            if r["c1"] and r["c2"] and r["c3"]:
                other = cells.get(pair, {}).get("configs", {}).get(cid)
                ok_other = bool(other and other["d_ustar"] and other["d_ustar"]["mean"] > 0)
                (go if ok_other else weak).append(dict(cell=cell, cid=cid, pair=pair, pair_d_ustar=other and other["d_ustar"]))
            elif r["c3_point"]:
                weak.append(dict(cell=cell, cid=cid, c1=r["c1"], c2=r["c2"], c3=r["c3"]))
    v = "GO" if go else ("STOP" if not any_c3 else "WEAK")
    return dict(verdict=v, go=go, weak=weak)


def sel_share(cells: dict, go: list) -> dict:
    """GA0-SEL：使 GO 成立、生成答對率最高的 P；D＝三個可部署挑法中最高的。"""
    if not go:
        return dict(verdict="NOT_APPLICABLE")
    best = max(go, key=lambda g: cells[g["cell"]]["configs"][g["cid"]]["acc"])
    r = cells[best["cell"]]["configs"][best["cid"]]
    if not r["deployable"]:
        return dict(verdict="NOT_MEASURED", at=best)
    d = max(r["deployable"].items(), key=lambda kv: kv[1]["acc"])
    s = d[1]["share"]
    v = "NO_LEARNED_SELECTOR" if s >= 0.7 else ("DO_A_DL1" if s >= 0.5 else "TRY_D0_THEN_STOP")
    return dict(verdict=v, at=best, best_deployable=d[0], share=s, all={k: x["share"] for k, x in r["deployable"].items()})


def drop_verdict(cells: dict, go: list) -> dict:
    out = []
    for g in go:
        d = cells[g["cell"]]["configs"][g["cid"]]["drop"]
        if d:
            out.append(dict(g, d_blur_minus_drop=d, contributes=d["mean"] >= 5.0))
    return dict(rows=out, verdict=("CONTRIBUTES" if any(x["contributes"] for x in out) else "NO") if out else "NOT_MEASURED")


def cmd_report(a):
    res, tfsum = {}, {}
    for cell in CELLS:
        pp = os.path.join(OUT, f"plan_{cell}.json")
        if not os.path.exists(pp):
            continue
        cfgs = json.load(open(pp))["configs"]
        tf_rows = read_rows(sorted(glob.glob(out_glob("tf", cell))))
        if tf_rows:
            tab = table(tf_rows, "d_gold_lp")
            tfsum[cell] = {cid: boot(v.values()) for cid, v in tab.items()}
        sp = os.path.join(OUT, f"gen_selection_{cell}.json")
        if CELLS[cell]["judged"] and os.path.exists(sp):
            gen_rows = read_rows(sorted(glob.glob(out_glob("gen", cell))))
            if gen_rows:
                sel = json.load(open(sp))
                res[cell] = dict(judge=judge_cell(gen_rows, cfgs, sel), gm02=gm02(gen_rows, sel),
                                 n_items=len({(r["item_id"], r["turn"]) for r in gen_rows if r["cid"] == "E0"}))
    v = verdict({c: r["judge"] for c, r in res.items()})
    doc = dict(run_id=run_id(), ts=ts(), ga0=v, ga0_sel=sel_share({c: r["judge"] for c, r in res.items()}, v["go"]),
               ga0_drop=drop_verdict({c: r["judge"] for c, r in res.items()}, v["go"]),
               gm02={c: r["gm02"] for c, r in res.items()}, cells=res, tf=tfsum,
               incomplete=[c for c in CELLS if CELLS[c]["judged"] and c not in res])
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "a0_report.json")
    json.dump(doc, open(p, "w"), indent=1, ensure_ascii=False, default=str)
    print(json.dumps(dict(ga0=v["verdict"], incomplete=doc["incomplete"]), ensure_ascii=False), "→", p)


# ============================================================================ CPU 自測
def self_test() -> bool:
    ok = True

    def chk(name, cond):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name)
        ok &= bool(cond)
    ov = seed_overlap()
    chk(f"seed 不重疊 {ov}", not any(ov.values()))
    chk("題數 96／192／96", [len(plan_set(k)) for k in ("tf", "gen1", "gen2")] == [96, 192, 96])
    chk("item_id 唯一", len({r["item_id"] for k in SETS for r in plan_set(k)}) == 384)
    for m in INFO:
        info = info_of(m)
        for spec, mult in ((BASES["B16"], 16.38), (BASES["B24"], 25.13), (BASES["B32"], 32.77)):
            f = spec_bytes(spec, info, 251)["request"] / region_bf16(info, 251)
            chk(f"{m} {spec} 倍數 {1 / f:.2f}", abs(1 / f - mult) < 0.01)
        chk(f"{m} INT4 精修 3.2×", abs(block_bytes(info, "bf16") / block_bytes(info, "int4") - 3.2) < 1e-9)
    info = info_of("llama")
    cfgs = tf_configs("llama", 251)
    kinds = {}
    for c in cfgs:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    chk(f"配置數 {len(cfgs)} {kinds}", kinds.get("prog") == 162 and kinds.get("drop") == 18 and kinds.get("E0") == 1
        and 8 <= kinds.get("uniform", 0) <= 26)
    chk("cid 唯一", len({c["cid"] for c in cfgs}) == len(cfgs))
    p = next(c for c in cfgs if c["cid"] == "P|B32|1/8|int4|ORACLE")
    want = (spec_bytes(BASES["B32"], info, 251)["request"] + 31 * block_bytes(info, "int4")) / region_bf16(info, 251)
    chk(f"漸進位元組 B32＋1/8 INT4 f＝{p['f']:.4f}", abs(p["f"] - want) < 1e-12 and p["m"] == 31)
    cands = match_candidates(info, 251)
    for c in cfgs:
        if c["kind"] == "prog" and c["selector"] == "ORACLE":
            mu = matched_uniform(c["f"], cands)
            if mu is None or mu[0] > c["f"] + 1e-12:
                chk(f"同位元組配對 {c['cid']}", False)
    chk("同位元組配對都 ≤ f_P", True)
    chk("m＝round(ρ·nreg)", [m_of(r, 251) for r in RHOS] == [16, 31, 63] and [m_of(r, 503) for r in RHOS] == [31, 63, 126]
        and [m_of(r, 63) for r in RHOS] == [4, 8, 16])
    go_cids = [c["cid"] for c in cfgs if c["kind"] == "prog" and c["selector"] == "ORACLE" and c["f"] <= GO_F]
    chk(f"f ≤ 1/12 的 ORACLE 配置 {len(go_cids)} 個（B16 1/16 INT4、B24 1/16·1/8 INT4、B32 1/16·1/8 INT4）", len(go_cids) == 5)
    # 合成 TF → select-gen
    rng = random.Random(0)
    rows = []
    for i in range(8):
        for c in cfgs:
            v = 0.0 if c["kind"] == "E0" else (-1.0 + (0.5 if c.get("selector") in ("ORACLE", "ORACLE-L") else 0) + rng.random() * 0.1)
            rows.append(dict(item_id=f"x{i}", turn="0", cid=c["cid"], d_gold_lp=v))
    sel = select_gen(rows, cfgs)
    chk(f"select-gen {len(sel['configs'])} 個、G1 {sel['g1']}", 10 <= len(sel["configs"]) <= 22 and len(sel["g1"]) == 3
        and sel["configs"][:2] == ["E0", "U|q:int4_kivi"])
    sel4 = select_gen(rows, cfgs, [("B24", "1/16", "int4"), tuple(sel["g1"][0])])
    chk(f"G4 加另一格的 G1：{sel4['g4']}", sel4["g4"] == [["B24", "1/16", "int4"]] and "P|B24|1/16|int4|ORACLE" in sel4["configs"])
    # 合成生成 → 判定：ORACLE 比均勻好 10 分 → GO；換成 0 分 → STOP
    by = {c["cid"]: c for c in cfgs}

    def fake_gen(gain, gain_other):
        cells = {}
        for cell, g in (("L128", gain), ("Q128", gain_other)):
            rr = []
            for i in range(200):
                u = rng.random()                                   # 每題一個潛在難度：配置之間的差＝門檻的差（配對、沒有雜訊）
                for cid in sel["configs"]:
                    c = by[cid]
                    base = 0.6
                    if c["kind"] == "uniform":
                        base = 0.5
                    if c["kind"] == "prog" and c["selector"] in ("ORACLE", "ORACLE-L"):
                        base = 0.5 + g
                    if c["kind"] == "prog" and c["selector"] in ("BASE-ATTN", "BASE-ATTN-L", "ONE-PASS"):
                        base = 0.5 + 0.8 * g
                    if c["kind"] == "drop":
                        base = 0.3
                    rr.append(dict(item_id=f"g{i}", turn="0", cid=cid, score=1.0 if u < base else 0.0))
            cells[cell] = judge_cell(rr, cfgs, sel)
        return cells
    cells = fake_gen(0.1, 0.05)
    v = verdict(cells)
    chk(f"合成：ORACLE＋10 分 → {v['verdict']}", v["verdict"] == "GO")
    s = sel_share(cells, v["go"])
    chk(f"合成：可部署拿到 80% → {s.get('verdict')} share={s.get('share')}", s.get("verdict") in ("NO_LEARNED_SELECTOR", "DO_A_DL1"))
    chk(f"合成：丟塊 → {drop_verdict(cells, v['go'])['verdict']}", drop_verdict(cells, v["go"])["verdict"] == "CONTRIBUTES")
    v0 = verdict(fake_gen(0.0, 0.0))
    chk(f"合成：沒有差 → {v0['verdict']}", v0["verdict"] == "STOP")
    vneg = verdict(fake_gen(0.1, -0.1))
    chk(f"合成：另一個模型方向相反 → {vneg['verdict']}", vneg["verdict"] == "WEAK")
    print("self-test", "PASS" if ok else "FAIL")
    return ok


# ============================================================================ CLI
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("plan")
    s = sub.add_parser("pending")
    s.add_argument("--kind", choices=["tf", "gen"], required=True)
    s.add_argument("--cell", choices=list(CELLS), required=True)
    s.add_argument("--shard", default="0/1")
    s = sub.add_parser("select-gen")
    s.add_argument("--cell", choices=list(CELLS), required=True)
    s.add_argument("--allow-partial", action="store_true")
    s.add_argument("--pair-missing", action="store_true")
    s.add_argument("--force", action="store_true")
    sub.add_parser("report")
    for name in ("calib", "tf", "gen", "gpu-selftest"):
        s = sub.add_parser(name)
        s.add_argument("--model", choices=list(INFO))
        s.add_argument("--cell", choices=list(CELLS))
        s.add_argument("--shard", default="0/1")
        s.add_argument("--budget-min", type=float, default=0)
        s.add_argument("--max-items", type=int, default=0, help="只做前幾題（冒煙）")
        s.add_argument("--smoke", action="store_true", help="冒煙：輸出寫到 --out（不進 results）、每題只跑部分配置")
        s.add_argument("--out", default="")
        s.add_argument("--calib-xl", default="")
        s.add_argument("--store", default="auto", help="gpu|host|auto：原值、粗底放哪（14B 256K 放 host）")
        s.add_argument("--seg-blocks", type=int, default=16, help="粗底分段編碼的 block 數")
        s.add_argument("--params-file", default=PARAMS)
        s.add_argument("--force", action="store_true")
        s.add_argument("--gm04", action="store_true", help="TF 時順便量 G-M04（問題前綴的挑塊）")
    return ap


def main():
    a = build_parser().parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if a.cmd in ("calib", "tf", "gen", "gpu-selftest"):
        import a0_gpu
        return {"calib": a0_gpu.cmd_calib, "tf": a0_gpu.cmd_tf, "gen": a0_gpu.cmd_gen, "gpu-selftest": a0_gpu.cmd_gpu_selftest}[a.cmd](a)
    {"plan": cmd_plan, "pending": cmd_pending, "select-gen": cmd_select_gen, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
