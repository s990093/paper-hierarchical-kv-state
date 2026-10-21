"""m16_x4.py — 第三階段 X4：把「xKV 式跨層低秩＋係數量化」推到極限（10/21）。

設計與事先寫死的判定：docs/stage3_20261019/x4_limit_plan.md（不改）；怎麼跑、時間估計、和計畫不同的地方：x_stack_howto.md §15。
主問題：Q-A 五種方法裡哪幾種該疊（邊際代價）；Q-B 每個目標倍數（8／12／16／20／24／32）下最好的配比、最佳配比掉出 ε 的倍數＝極限；
另外 128K（44c 題組）、Qwen 256K 切片、多輪（C1a 的兩輪題組）。

和 X1／X2（m16_xstack.py）不同、要知道的：
  共用 SVD   每題、每（K／V、窗口 W）只做一次 torch.svd_lowrank，q＝這題所有配置要的最大聯合秩＋16（niter、seed 照 m16_xstack），
             每個配置取前 r 欄（U·S 的前 r 欄＝係數、V 的前 r 欄＝基底）。X1／X2 是每個配置各做一次 q＝r＋16 的隨機 SVD；
             q 越大前 r 個奇異向量越準（越接近精確 SVD），所以這裡的重建誤差 ≤ X1／X2 的同一個配置（gpu-selftest 核對：
             和精確 SVD 的誤差比在 [0.999, 1.05]）。不這樣做的話 X4a 約 80 個 xKV 配置 × 96 題要約 16 卡小時〔估〕。
  丟的順序   X4 的輕度丟一律 xl+drop（基底在全部 token 上估、再丟 block 的係數；m16_xstack 的合法順序 ("xl","drop")），
             才能共用 SVD；X2 的 Qwen 兩種順序 −14.4／−15.4（50% 丟）差不多。位元組照 predict_bytes（丟掉的 block 不存係數、基底照存）。
  其他（位元組帳、量化器、RoPE、遮罩路徑、計分）全部直接用 m16_xstack／m10_*／m16_s2_equalbytes 的函式，不重寫。

子命令（GPU 的才 import torch；模組頂層只有標準函式庫與 m16_xstack（同樣沒有 torch））：
  grid          CPU：產生 code/m16_xstack_combos_x4.json（配比格點；位元組照 predict_bytes 實算）
  tf            GPU：X4a teacher forcing（每題 prefill 一次、共用 SVD、每個配置 TF）
  select-b      CPU：X4a → X4b 的配置（每個倍數帶 TF 最好的＋「只降秩」對照；事先寫死）
  gen           GPU：生成答對率（X4b 32K、X4c 128K／256K、X4d 兩輪 --turns 2）
  select-cd     CPU：X4b → X4c／X4d 的前緣 3 個配置（事先寫死）
  dump          GPU：前緣配置的實際係數位元組存檔（本地盤），給 lossless
  lossless      CPU：用 m16_s1_lossless 的 zstd／planes 量真實壓縮比與解壓速度
  report        CPU：Q-A／Q-B／極限／128K／多輪的表、Pareto 圖、x4_report.json
  gpu-selftest  GPU：共用 SVD vs 精確 SVD、xl+drop、W 2／4／7／8、兩輪生成＝m10_quality.run_turns、dump 的位元組
  --self-test   CPU（不 import torch）
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import json
import math
import os
import statistics
import sys
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m16_xstack as XS  # noqa: E402
from m16_xstack import (EPS, MIN_ITEMS, MLPERF, MODEL_INFO, SpecError, boot2, bytes_summary, canonical,  # noqa: E402
                        parse_spec, predict_bytes, predict_ratio, spec_attrs, validate)

CHUNK, REPO = XS.CHUNK, XS.REPO
OUT = os.path.join(XS.OUT_DIR, "x4")                       # 進 git 的摘要
DATA4 = os.path.join(XS.DATA, "x4")                        # 逐列原始輸出（不進 git）
COMBOS_X4 = os.path.join(HERE, "m16_xstack_combos_x4.json")
NREG = XS.N_BLOCKS - 1                                     # 32K 的區內 block 數（block 0 一律 BF16）

TARGETS = (8, 12, 16, 20, 24, 32)                          # 目標倍數（對 BF16 的區內位元組）
BAND_TOL = 0.10                                            # 「落在 T」＝實際倍數在 T 的 ±10%（計畫 §2）
STACK_TOL = 0.05                                           # 「疊加優於單降秩」的同倍數帶 ±5%（計畫 §3）
MAX_PER_T = 10                                             # 每個 T 最多 10 種分法（計畫 §2：約 6–10）
W_SET = {"llama": (2, 4, 8), "qwen": (2, 4, 7)}
WBIG_T = (20, 24, 32)                                      # （W 大已從 ALLOCS 縮掉；留著給另開版本：W＝8／7 只放在這幾個 T）
DROP_STUDY_T = (16, 24)                                    # 輕度丟的完整掃描（KEYDIFF／KNORM × 6.25／12.5／25%）只在懸崖附近的兩個 T
DROP_FRACS = (0.0625, 0.125, 0.25)
DROP_METHODS = ("KEYDIFF", "KNORM")
BITS = {"bf16": None, "int8": "int8_token", "int4": "int4_kivi", "k4v2": "kv-int4_kivi-int2_kivi",
        "k2v4": "kv-int2_kivi-int4_kivi", "int2": "int2_kivi"}
KV_SPLIT = {"diag": 1.5, "kless": 1.85}                    # V 秩 ≈ 比例 × K 秩（diag＝xKV 的 1:1.5；kless＝K 少 V 多，計畫的 1:1.75–2）
RK_RANGE = range(2, 49)
# 計畫 §2 的秩細格點（每個都用 int4_kivi、W＝4、不丟）：對角（1:1.5）6 組、K 少 V 多 5 組、K 多一點的對照 1 組
FINE_PAIRS = ((6, 9), (8, 12), (9, 14), (11, 17), (13, 20), (16, 24), (6, 12), (8, 14), (9, 17), (11, 20), (13, 24), (11, 14))
CONTROLS = ("none", "q:int4_kivi", "xl:k11:v17+q:int4_kivi", "xl:k22:v34+q:int4_kivi")
# Q-B 每個 T 的分法（照優先順序取，最多 MAX_PER_T 個）：(名稱, 係數位元, W 的種類, K／V 分配, 丟)
ALLOCS = (("int4", "int4", "w4", "diag", None),            # 底：xKV＋INT4
          ("rank_only", "bf16", "w4", "diag", None),       # 只降秩（BF16 係數）
          ("int8", "int8", "w4", "diag", None),
          ("k4v2", "k4v2", "w4", "diag", None),
          ("k2v4", "k2v4", "w4", "diag", None),
          ("int2", "int2", "w4", "diag", None),
          # K／V 秩分配（K 少 V 多）由 FINE_PAIRS 的 5 組 kless 涵蓋（INT4、W＝4、15–39 倍），不在每個 T 另放（省時間）
          ("int4_w2", "int4", "w2", "diag", None),
          # W 大（Llama 8、Qwen 7）縮掉（時間）：它的 SVD 是 8192／3584 維，每題約 24／15 卡秒〔估〕＝X4a 約 1 卡小時（x_stack_howto.md §15.6）
          ("int4_drop_keydiff12", "int4", "w4", "diag", ("KEYDIFF", 0.125)))
ALLOC_WHY = {"int4": "底：xKV＋INT4 KIVI 係數（X1／X2 最好的組合）",
             "rank_only": "只降秩（係數 BF16）：「疊加優於單降秩」的對照",
             "int8": "係數 INT8（逐 token）：位元多、秩少",
             "k4v2": "K 係數 INT4、V 係數 INT2（K4V2）",
             "k2v4": "K 係數 INT2、V 係數 INT4（K2V4）",
             "int2": "係數 INT2 KIVI：位元最少、秩最多",
             "int4_w2": "INT4、窗口 W＝2：基底小、跨層共用少",
             "int4_drop_keydiff12": "INT4＋輕度丟 KEYDIFF 12.5%（xl+drop）：把位元組省在丟 block 上"}

X4A_SEED0, X4A_N = 64000, 24                               # X4a：T2–T5 各 24 題＝96 題（新種子）
X4B_SEED0, X4B_N = 65000, 48                               # X4b：T2–T5 各 48 題＝192 題（新種子）
X4D_SEED0, X4D_N, X4D_TASKS = 70000, 32, ("T2", "T3", "T4")  # X4d：C1a 的兩輪題組（seed 70000 起、T2–T4；這裡各 32 題＝96 題）
X4C_NB, X4C_SEED0, X4C_N = XS.LC_BLOCKS, XS.LC_SEED0, XS.LC_N   # X4c：44c 的 128K 題組（252 block、seed 40000、每任務 12 題）
Q256_NB, Q256_N = 504, 6                                   # Qwen 256K 切片：504 block＝258,048 token（noDCA 上限 262,144；留問題＋生成）
DUMP_N = 1                                                 # lossless：X4a 的題、每個任務 1 題＝4 題
CTX_NAME = {64: "32K", XS.LC_BLOCKS: "128K", Q256_NB: "256K"}


# ============================================================================ 配比格點（純 Python）
def xl_spec(rk: int, rv: int, W: int, bits: str, drop=None) -> str:
    s = f"xl:k{rk}:v{rv}" + (f":w{W}" if W != 4 else "")
    if drop:
        s += f"+drop:{drop[0]}:{XS._g(drop[1])}"
    if BITS[bits]:
        s += f"+q:{BITS[bits]}"
    return canonical(parse_spec(s))


def wval(model: str, kind: str) -> int:
    return {"w2": 2, "w4": 4, "wbig": W_SET[model][2]}[kind]


def spec_meta(spec: str) -> dict:
    """配置的軸：W、K／V 秩（每 head 每層等效）、係數位元、丟的方法與比例；非 xKV 的配置秩為 0。"""
    st = {s["op"]: s for s in parse_spec(spec)}
    xl = st.get("xl")
    q = st["q"]["fmt"] if "q" in st else None
    bits = next((k for k, v in BITS.items() if v == q), q or "bf16") if xl else (q or "none")
    return dict(W=xl["w"] if xl else 0, rk=xl["rk"] if xl else 0, rv=xl["rv"] if xl else 0, bits=bits,
                drop_method=st["drop"]["method"] if "drop" in st else "", drop_frac=st["drop"]["frac"] if "drop" in st else 0.0,
                xl=int(bool(xl)))


def in_band(mult: float, T: float, tol: float = BAND_TOL) -> bool:
    return (1 - tol) * T <= mult <= (1 + tol) * T


def best_rank(model: str, T: int, bits: str, W: int, split: str, drop=None, rk_cap: int | None = None):
    """這個分法要落在 T：照 K／V 分配的比例掃 K 秩，挑倍數最接近 T（對數距離）而且在 ±10% 內的；沒有 → None。"""
    best = None
    for rk in RK_RANGE:
        if rk_cap and rk > rk_cap:
            break
        rv = max(1, int(rk * KV_SPLIT[split] + 0.5))
        try:
            spec = xl_spec(rk, rv, W, bits, drop)
            validate(parse_spec(spec), MODEL_INFO[model], NREG)
        except SpecError:
            continue
        mult = 1 / predict_ratio(spec, model)
        d = abs(math.log(mult / T))
        if in_band(mult, T) and (best is None or d < best[0]):
            best = (d, spec, mult)
    return best


def build_grid(model: str) -> dict:
    """X4a 的配置（事先寫死的產生規則；self-test 核對 JSON＝這裡重算）。"""
    cfg, skipped = {}, []

    def add(spec, role, why, T=None):
        e = cfg.setdefault(spec, dict(spec=spec, roles=[], why=[], targets=[]))
        if role not in e["roles"]:
            e["roles"].append(role)
        if why not in e["why"]:
            e["why"].append(why)
        if T is not None and T not in e["targets"]:
            e["targets"].append(T)
    for T in TARGETS:
        n = 0
        for name, bits, wk, split, drop in ALLOCS:
            if n >= MAX_PER_T:
                break
            if wk == "wbig" and T not in WBIG_T:
                skipped.append(dict(T=T, alloc=name, why=f"W 大的只在 T∈{WBIG_T}（SVD 是 W·H·128 維，最貴）"))
                continue
            b = best_rank(model, T, bits, wval(model, wk), split, drop, rk_cap=16 if wk == "wbig" else None)
            if b is None:
                skipped.append(dict(T=T, alloc=name, why="秩的格點（K 秩 2–48）裡沒有落在 ±10% 內的"))
                continue
            add(b[1], f"alloc:{name}", ALLOC_WHY[name], T)
            n += 1
        if T in DROP_STUDY_T:
            for m in DROP_METHODS:
                for fr in DROP_FRACS:
                    b = best_rank(model, T, "int4", 4, "diag", (m, fr))
                    if b is None:
                        skipped.append(dict(T=T, alloc=f"drop_{m}_{fr}", why="沒有落在 ±10% 內的秩"))
                        continue
                    add(b[1], "drop_study", f"輕度丟的邊際代價：{m} {fr * 100:g}%（xl+drop、INT4、W＝4）", T)
    for rk, rv in FINE_PAIRS:
        add(xl_spec(rk, rv, 4, "int4"), "fine_grid", "計畫 §2 的秩細格點（INT4、W＝4、不丟）")
    for s in CONTROLS:
        add(canonical(parse_spec(s)), "control", "對照（X1／X2 的配置與錨點）")
    out = []
    for spec, e in cfg.items():
        st = validate(parse_spec(spec), MODEL_INFO[model], NREG)
        r = predict_ratio(spec, model)
        mult = 1 / r
        bands = [T for T in TARGETS if in_band(mult, T)]
        out.append(dict(spec=spec, ratio=r, multiple=round(mult, 3), bands=bands, targets=sorted(e["targets"]), **spec_meta(spec), roles=e["roles"],
                        why="；".join(e["why"]), deployable=spec_attrs(st)["deployable"]))
    out.sort(key=lambda c: (c["ratio"], c["spec"]))
    return dict(configs=out, skipped=skipped)


def grid_doc() -> dict:
    return {"_doc": ["m16_x4.py 的 X4a 配置（10/21；**在看到任何 X4 數字之前**由 m16_x4.py grid 產生，之後不改；--self-test 核對＝程式重算）。",
                     "設計：docs/stage3_20261019/x4_limit_plan.md §2；位元組比照 m16_xstack.predict_bytes 實算（含每題基底、scale、遮罩），"
                     "倍數＝1／區內位元組比（32K、block 1–63）。",
                     f"每個目標倍數 T∈{list(TARGETS)}：照 ALLOCS 的優先順序，每個分法掃 K 秩（V 秩＝比例×K 秩）挑倍數最接近 T 且在 ±10% 內的；"
                     f"最多 {MAX_PER_T} 個。W 只掃 2、4（W 大＝Llama 8、Qwen 7 因時間縮掉，見 x_stack_howto.md §15.6）。"
                     f"輕度丟的完整掃描只在 T∈{list(DROP_STUDY_T)}。",
                     "輕度丟一律 xl+drop（基底在全部 token 上估，才能共用 SVD）。roles：alloc:<分法>／drop_study／fine_grid／control。"],
            "version": "2026-10-21 x4 v1", "targets": list(TARGETS), "band_tol": BAND_TOL, "max_per_t": MAX_PER_T,
            "allocs": [dict(name=a[0], bits=a[1], w=a[2], split=a[3], drop=a[4], why=ALLOC_WHY[a[0]]) for a in ALLOCS],
            "kv_split": KV_SPLIT, "fine_pairs": [list(p) for p in FINE_PAIRS], "controls": list(CONTROLS),
            "models": {m: build_grid(m) for m in MODEL_INFO}}


def cmd_grid(a):
    g = grid_doc()
    if os.path.exists(a.out) and not a.force:
        old = json.load(open(a.out))
        if old.get("models") == json.loads(json.dumps(g["models"])):
            print(f"{a.out} 已有、內容相同 → 不重寫")
            return
        raise SystemExit(f"{a.out} 已存在但內容不同 → 不默默改事先寫死的格點（要改另開版本、加 --force）")
    json.dump(g, open(a.out, "w"), indent=1, ensure_ascii=False)
    for m, v in g["models"].items():
        bt = {T: sum(1 for c in v["configs"] if T in c["targets"]) for T in TARGETS}
        print(f"{m}：{len(v['configs'])} 個配置；每個 T 的分法數 {bt}；跳過 {len(v['skipped'])} 個")


def load_x4_configs(path: str, model: str, smoke: bool = False) -> list:
    g = json.load(open(path))
    cs = g["models"][model]["configs"]
    if smoke:
        cs = [c for c in cs if c["spec"] in smoke_specs(model)]
    out = []
    for c in cs:
        st = validate(parse_spec(c["spec"]), MODEL_INFO[model], NREG)
        out.append(dict(c, stages=st))
    return out


def smoke_specs(model: str) -> list:
    """冒煙用：W＝2、xl+drop、INT2、只降秩、錨點（W 大已縮掉，不在冒煙裡算 8192 維的 SVD）。"""
    return [xl_spec(8, 12, 2, "int4"), xl_spec(8, 12, 4, "int4", ("KEYDIFF", 0.125)), xl_spec(11, 17, 4, "int2"),
            xl_spec(5, 8, 4, "bf16"), "q:int4_kivi"]


def smoke_configs(model: str) -> list:
    return [dict(spec=s, roles=["smoke"], stages=validate(parse_spec(s), MODEL_INFO[model], NREG)) for s in smoke_specs(model)]


# ============================================================================ 共用 SVD 的 pipeline（GPU；torch 只在函式裡 import）
class SvdStore:
    """一題、一個區域：每個 (K／V, W) 的 svd_lowrank 只做一次，q＝這題的配置裡最大的聯合秩（每 head 每層等效秩×W×H）＋16，
    seed、niter 照 m16_xstack（XL_SEED＋窗口、XL_NITER）；存 U·S 與 V 的前 R 欄，配置取前 r 欄。只用在「原始 KV」上（每個配置前都還原）。"""

    def __init__(self, cfgs, H: int):
        self.H, self.need, self.store, self.t = H, {}, {}, {}
        for c in cfgs:
            st = {s["op"]: s for s in c["stages"]}
            low = st.get("xl")
            if low is None:
                continue
            if low.get("cal"):
                raise SpecError("X4 不用 :cal（校準基底另判，見 xlcal）")
            for hi, r in ((0, low["rk"]), (1, low["rv"])):
                if r:
                    k = (hi, low["w"])
                    self.need[k] = max(self.need.get(k, 0), r)

    def get(self, src, srcpos, hi: int, W: int, rhe: int, cx):
        import torch
        key = (hi, W)
        if rhe > self.need.get(key, 0):
            raise RuntimeError(f"SvdStore：{key} 要秩 {rhe}，計畫只到 {self.need.get(key)}")
        if key not in self.store:
            L, H, T, D = src.shape
            R = self.need[key] * W * H
            c0, s0 = cx.cs(srcpos)
            torch.cuda.synchronize()
            t0 = time.time()
            US, Vs = [], []
            for w in range(L // W):
                Z = torch.cat([(XS.t_rope_inv(src[l], c0, s0) if hi == 0 else src[l].float()).permute(1, 0, 2).reshape(T, H * D)
                               for l in range(w * W, (w + 1) * W)], dim=1)
                torch.manual_seed(XS.XL_SEED + w)
                U, S, V = torch.svd_lowrank(Z, q=min(R + XS.XL_OVERSAMPLE, min(Z.shape)), niter=XS.XL_NITER)
                if U.shape[1] < R:
                    raise RuntimeError(f"svd_lowrank 只給 {U.shape[1]} 欄，要 {R}（Z {tuple(Z.shape)}）")
                US.append((U[:, :R] * S[:R]).contiguous())
                Vs.append(V[:, :R].contiguous())
                del Z, U, S, V
            torch.cuda.synchronize()
            self.t[f"{'kv'[hi]}_w{W}_R{R}"] = round(time.time() - t0, 3)
            self.store[key] = (US, Vs)
        return self.store[key]


def t_xl_half_shared(src, srcpos, P, hi, rhe, W, f, cx, keep_idx, store):
    """t_xl_half（m16_xstack）的共用 SVD 版：係數＝U·S 的前 r 欄（xl+drop 時只留下的 token）、基底＝V 的前 r 欄（BF16）；
    量化與重建逐行照 t_xl_half。回傳 (解碼 BF16, 係數位元組, 基底位元組)。"""
    import torch
    L, H, T, D = src.shape
    r = rhe * W * H
    US, Vs = store.get(src, srcpos, hi, W, rhe, cx)
    As, Bs = [], []
    for w in range(L // W):
        A = US[w][:, :r]
        if keep_idx is not None:
            A = A.index_select(0, keep_idx)
        As.append(A)
        Bs.append(Vs[w][:, :r].T.contiguous().to(torch.bfloat16))
    A4 = torch.stack(As).unsqueeze(1)
    Ad, nb = XS.t_halves_enc(A4, f, "k")
    Tk = A4.shape[2]
    cos, sin = cx.cs(P)
    outs = []
    for w in range(L // W):
        Zh = Ad[w, 0] @ Bs[w].float()
        for j in range(W):
            x = Zh[:, j * H * D:(j + 1) * H * D].reshape(Tk, H, D).permute(1, 0, 2)
            outs.append(XS.t_rope_fwd(x, cos, sin, cx.apply_rope) if hi == 0 else x.to(torch.bfloat16))
    return torch.stack(outs), nb, (L // W) * r * W * H * D * 2


def x4_drop(st: dict, X, cx):
    """丟的 block（區內編號）與留下的 token 索引；和 m16_xstack.t_pipeline 同一個寫法。"""
    import torch
    C = cx.chunk
    nreg = X.shape[3] // C
    if "drop" not in st:
        return None, []
    k = XS.drop_k(st["drop"]["frac"], nreg)
    dropped = list(cx.orders[st["drop"]["method"]][:k])
    kept = [b for b in range(nreg) if b not in set(dropped)]
    return torch.cat([torch.arange(b * C, (b + 1) * C, device=X.device) for b in kept]), dropped


def x4_pipeline(stages, X, pos, cx, info, store):
    """沒有 xl 的配置 → 直接用 m16_xstack.t_pipeline；有 xl → 共用 SVD（xl 或 xl+drop、q 最後）。回傳同 t_pipeline。"""
    import torch
    st = {s["op"]: s for s in stages}
    ops = [s["op"] for s in stages]
    low = st.get("xl")
    if low is None:
        return XS.t_pipeline(stages, X, pos, cx, info)
    if low.get("cal") or "merge" in st or "lr" in st:
        raise SpecError("X4 的 xl 只接受 xl[+drop][+q]")
    if "drop" in st and ops[:2] != ["xl", "drop"]:
        raise SpecError("X4 的丟一律 xl+drop（基底在全部 token 上估，才能共用 SVD）")
    C = cx.chunk
    nreg = X.shape[3] // C
    by = {}
    idx, dropped = x4_drop(st, X, cx)
    if "drop" in st:
        by["mask"] = (nreg + 7) // 8
    P = pos if idx is None else pos.index_select(0, idx)
    fmt = st["q"]["fmt"] if "q" in st else None
    hv = XS.halves(fmt) if fmt else (None, None)
    outs = []
    for hi, h in enumerate("kv"):
        r = low["rk"] if hi == 0 else low["rv"]
        f = hv[hi]
        if r == 0:
            S = X if idx is None else X.index_select(3, idx)
            d, by[f"{h}.raw"] = XS.t_raw_half(S[:, hi], f, h, C)
        else:
            d, by[f"{h}.coef"], by[f"{h}.basis"] = t_xl_half_shared(X[:, hi], pos, P, hi, r, low["w"], f, cx, idx, store)
        outs.append(d)
    return torch.stack(outs, dim=1), idx, by, dropped


def x4_encode_parts(stages, X, pos, cx, info, store) -> dict:
    """dump 用：xl 配置實際存下來的每一個 part（numpy；係數的量化 part、BF16 的基底、遮罩），位元組＝predict_bytes（呼叫端核對）。"""
    import numpy as np
    import torch
    import m10_codec_x as X10
    st = {s["op"]: s for s in stages}
    low = st["xl"]
    C = cx.chunk
    nreg = X.shape[3] // C
    idx, dropped = x4_drop(st, X, cx)
    parts = {}
    if "drop" in st:
        keep = np.ones(nreg, dtype=bool)
        keep[dropped] = False
        parts["mask"] = (np.packbits(keep), "raw")
    fmt = st["q"]["fmt"] if "q" in st else None
    hv = XS.halves(fmt) if fmt else (None, None)

    def npy(t):
        if t.dtype == torch.bfloat16:
            return t.contiguous().view(torch.int16).cpu().numpy().view(np.uint16), "bf16"
        return t.contiguous().cpu().numpy(), "raw"
    for hi, h in enumerate("kv"):
        r = low["rk"] if hi == 0 else low["rv"]
        W = low["w"]
        US, Vs = store.get(X[:, hi], pos, hi, W, r, cx)
        R = r * W * X.shape[2]
        As = [(US[w][:, :R] if idx is None else US[w][:, :R].index_select(0, idx)) for w in range(len(US))]
        A4 = torch.stack(As).unsqueeze(1).contiguous()
        f = XS.ALIAS.get(hv[hi], hv[hi])
        if f in (None, "bf16"):
            parts[f"{h}.coef.bf16"] = npy(A4.to(torch.bfloat16))
        else:
            for k, v in X10.enc_half(A4, f, "k").items():
                parts[f"{h}.coef.{k}"] = npy(v)
        parts[f"{h}.basis"] = npy(torch.stack([Vs[w][:, :R].T.contiguous().to(torch.bfloat16) for w in range(len(Vs))]))
    return parts


# ============================================================================ 一題（GPU）
def run_turns(ans, it: dict, ctx_len: int, turns: int) -> list:
    """生成 turns 輪（照 m10_quality.run_turns：第 t 輪＝history_ids＋query_ids 接在上一輪的起點之後；第 2 輪的歷史固定成第 1 輪的
    標準答案；生成長度 gen_len）。第 1 輪＝m10_sweep.Item.gen()（gpu-selftest 核對）。KV 就是現在緩衝裡放的（含 key_keep）。"""
    import m10_tasks as T
    from m10_quality import gen_len
    if len(it["turns"]) < turns:
        raise RuntimeError(f"{it['item_id']} 只有 {len(it['turns'])} 輪，要 {turns}")
    pos, out = ctx_len, []
    for t in range(turns):
        tr = it["turns"][t]
        seq = tr.get("history_ids", []) + tr["query_ids"]
        logits = ans.feed(seq, pos)
        gen, stop = ans.generate(logits, pos + len(seq), gen_len(it))
        out.append(dict(score=T.score(tr["metric"], ans.decode(gen), tr["gold"]), stop=stop, gen=gen))
        pos = pos + len(seq)
    return out


def _region(km, n: int):
    nb = n // CHUNK
    return 1, nb - 1, nb


def _orders(km, st, it, cfgs, n):
    import m10_drop as DR
    need = sorted({s["method"] for c in cfgs for s in c["stages"] if s["op"] == "drop"})
    if not need:
        return {}
    cx = DR.Ctx(km, st.ids, n, st.bf16, CHUNK, it["seed"], st.needle, it["turns"][0]["query_ids"])
    out = {m: DR.unit_order(cx, "B", m) for m in need}
    del cx
    return out


def _needle_tok(km, blocks):
    import torch
    return torch.tensor([t for b in blocks for t in range(b * CHUNK, (b + 1) * CHUNK)], dtype=torch.long, device=km.device)


def x4a_item(ans, it, cfgs, info):
    """X4a 一題：prefill → E0 的 teacher forcing → 每個配置：還原區內 → 共用 SVD 的 pipeline → 位元組核對 → 重建誤差 → 放回 → TF。"""
    import torch
    import m10_sweep as SW
    import m16_s2_equalbytes as S2
    km = ans.km
    km.key_keep = None
    st = SW.Item(ans, it)
    n = st.ctx_len
    lo, hi, nb = _region(km, n)
    nreg = hi - lo + 1
    s0, s1 = lo * CHUNK, (hi + 1) * CHUNK
    lp0 = st.score()
    m0 = SW.Item.metrics(lp0, lp0, st.gold_ids)
    orders = _orders(km, st, it, cfgs, n)
    tctx = XS.TCtx(km, n, orders={m: [b - lo for b in o if lo <= b <= hi] for m, o in orders.items()}, deep=XS.deep_layers(km.L))
    pos = torch.arange(s0, s1, device=km.device)
    nd = _needle_tok(km, st.needle)
    store = SvdStore(cfgs, km.H)
    tr = it["turns"][0]
    base = dict(item_id=it["item_id"], task=it["task"], band=tr["band"], seed=it["seed"], gold_lp_e0=m0["gold_lp"],
                first_ok_e0=m0["first_ok"])
    rows = [dict(base, spec="E0", roles="ref", targets="", W=0, rk=0, rv=0, bits="bf16", drop_method="", drop_frac=0.0, deployable=1,
                 k_drop=0, kept_frac=1.0, needle_kept=1.0, bytes_region="", ratio_region=1.0, multiple=1.0, bytes_detail="",
                 relerr_k=0.0, relerr_v=0.0, cenerr_k=0.0, t_pipe_s="", t_tf_s="", **m0)]
    dirty = False
    for c in cfgs:
        if dirty:
            XS.restore_blocks(km, st, range(lo, hi + 1))
        X = km.kv[:, :, :, s0:s1]
        torch.cuda.synchronize()
        ta = time.time()
        with torch.inference_mode():
            dec, idx, by, dropped = x4_pipeline(c["stages"], X, pos, tctx, info, store)
            pred = predict_bytes(c["stages"], info, nreg)
            if by != pred:
                raise RuntimeError(f"{c['spec']}：實際位元組 {by} ≠ 預測 {pred}（規則 2：停）")
            rel = XS.t_relerr(X, dec, idx)
            if dec.data_ptr() != X.data_ptr():
                if idx is None:
                    X.copy_(dec)
                else:
                    X.index_copy_(3, idx, dec)
            del dec
        dirty = True
        torch.cuda.synchronize()
        tb = time.time()
        kk = S2.keep_tensor(km, [lo + b for b in dropped])
        mm = S2.tf(st, lp0, kk)
        kept, ndk = S2.kept_stats(kk, n, nd)
        bs = bytes_summary(by, info, nreg, nb)
        rows.append(dict(base, spec=c["spec"], roles="|".join(c["roles"]), targets="|".join(str(t) for t in c.get("targets", [])),
                         **spec_meta(c["spec"]), deployable=spec_attrs(c["stages"])["deployable"], k_drop=len(dropped),
                         kept_frac=kept, needle_kept=ndk, bytes_region=bs["bytes_region"], ratio_region=bs["ratio_region"],
                         multiple=round(1 / bs["ratio_region"], 4), bytes_detail=bs["bytes_detail"], **rel,
                         t_pipe_s=round(tb - ta, 3), t_tf_s=round(time.time() - tb, 3), **mm))
    if dirty:
        XS.restore_blocks(km, st, range(lo, hi + 1))
    del st
    return rows, dict(t_svd=store.t, need={f"{'kv'[h]}_w{w}": r for (h, w), r in store.need.items()})


def x4_gen_item(ans, it, cfgs, info, turns: int):
    """生成一題（32K／128K／256K；turns＝1 或 2）：E0（sdpa）→ 每個配置：還原區內 → pipeline → 位元組核對 → 放回 → key_keep → 生成 turns 輪。"""
    import torch
    import m10_sweep as SW
    import m16_s2_equalbytes as S2
    km = ans.km
    km.key_keep = None
    st = SW.Item(ans, it)
    n = st.ctx_len
    lo, hi, nb = _region(km, n)
    nreg = hi - lo + 1
    s0, s1 = lo * CHUNK, (hi + 1) * CHUNK
    t0 = time.time()
    e0 = run_turns(ans, it, n, turns)
    t_e0 = round(time.time() - t0, 3)
    orders = _orders(km, st, it, cfgs, n)
    tctx = XS.TCtx(km, n, orders={m: [b - lo for b in o if lo <= b <= hi] for m, o in orders.items()}, deep=XS.deep_layers(km.L))
    pos = torch.arange(s0, s1, device=km.device)
    nds = [_needle_tok(km, it["turns"][t]["needle_blocks"]) for t in range(turns)]
    store = SvdStore(cfgs, km.H)
    rows = []
    for t in range(turns):
        rows.append(dict(turn=t + 1, spec="E0", roles="ref", W=0, rk=0, rv=0, bits="bf16", drop_method="", drop_frac=0.0, deployable=1,
                         k_drop=0, kept_frac=1.0, needle_kept=1.0, bytes_region="", ratio_region=1.0, multiple=1.0,
                         gen_score=e0[t]["score"], gen_stop=e0[t]["stop"], score_e0=e0[t]["score"], t_gen_s=t_e0, t_pipe_s=""))
    dirty = False
    for c in cfgs:
        if dirty:
            XS.restore_blocks(km, st, range(lo, hi + 1))
        X = km.kv[:, :, :, s0:s1]
        torch.cuda.synchronize()
        ta = time.time()
        with torch.inference_mode():
            dec, idx, by, dropped = x4_pipeline(c["stages"], X, pos, tctx, info, store)
            pred = predict_bytes(c["stages"], info, nreg)
            if by != pred:
                raise RuntimeError(f"{c['spec']}：實際位元組 {by} ≠ 預測 {pred}（規則 2：停）")
            if dec.data_ptr() != X.data_ptr():
                if idx is None:
                    X.copy_(dec)
                else:
                    X.index_copy_(3, idx, dec)
            del dec
        dirty = True
        torch.cuda.synchronize()
        tb = time.time()
        kk = S2.keep_tensor(km, [lo + b for b in dropped])
        km.key_keep = kk
        try:
            res = run_turns(ans, it, n, turns)
        finally:
            km.key_keep = None
        tg = round(time.time() - tb, 3)
        bs = bytes_summary(by, info, nreg, nb)
        for t in range(turns):
            kept, ndk = S2.kept_stats(kk, n, nds[t])
            rows.append(dict(turn=t + 1, spec=c["spec"], roles="|".join(c["roles"]), **spec_meta(c["spec"]),
                             deployable=spec_attrs(c["stages"])["deployable"], k_drop=len(dropped), kept_frac=kept, needle_kept=ndk,
                             bytes_region=bs["bytes_region"], ratio_region=bs["ratio_region"], multiple=round(1 / bs["ratio_region"], 4),
                             gen_score=res[t]["score"], gen_stop=res[t]["stop"], score_e0=e0[t]["score"], t_gen_s=tg,
                             t_pipe_s=round(tb - ta, 3)))
    if dirty:
        XS.restore_blocks(km, st, range(lo, hi + 1))
    tr = it["turns"]
    out = [dict(r, item_id=it["item_id"], task=it["task"], band=tr[r["turn"] - 1]["band"], seed=it["seed"]) for r in rows]
    del st
    return out, dict(t_svd=store.t, need={f"{'kv'[h]}_w{w}": r for (h, w), r in store.need.items()})


FIELDS_X4A = ["run_id", "ts", "shard", "model", "item_id", "task", "band", "seed", "spec", "roles", "targets", "W", "rk", "rv", "bits",
              "drop_method", "drop_frac", "xl", "deployable", "k_drop", "kept_frac", "needle_kept", "bytes_region", "ratio_region", "multiple",
              "bytes_detail", "kl", "gold_lp", "gold_lp_e0", "top1", "first_ok", "first_ok_e0", "relerr_k", "relerr_v", "cenerr_k",
              "t_pipe_s", "t_tf_s", "n_rows_item", "own_gpu_intruders", "host_level"]
FIELDS_X4G = ["run_id", "ts", "shard", "model", "ctx", "n_blocks", "item_id", "task", "band", "seed", "turn", "spec", "roles", "W", "rk",
              "rv", "bits", "drop_method", "drop_frac", "xl", "deployable", "k_drop", "kept_frac", "needle_kept", "bytes_region",
              "ratio_region", "multiple", "gen_score", "gen_stop", "score_e0", "t_gen_s", "t_pipe_s", "n_rows_item", "own_gpu_intruders",
              "host_level"]


def _loop(a, plan, make_rows, fields, extra_cols, nb=64):
    """逐題續跑、shard、時間預算、GpuWatcher（品質規則：污染時分數照用、逐列記）——和 m16_xstack.cmd_screen／cmd_gen 同一個寫法。"""
    import torch
    import m10_tasks as T
    si, sk = XS.parse_shard(a.shard)
    done = XS.done_union(a.out)
    fo, w = XS.open_out(a.out, fields)
    chk = open(XS.side_path(a.out, ".checks.jsonl"), "a")
    watch = XS.start_watch(a.out)
    rid = os.environ.get("RUN_ID", "adhoc")
    P = json.load(open(a.params_file))["params"][a.model]
    t_all, n_new = time.time(), 0
    try:
        for k, (task, b1, b2, seed) in enumerate(plan):
            if k % sk != si or XS.item_id_of(task, b1, b2, seed) in done:
                continue
            if a.budget_min and (time.time() - t_all) / 60 > a.budget_min:
                print(f"時間預算 {a.budget_min} 分鐘到了：剩下的題留給下一個 job（逐題續跑）", flush=True)
                break
            t0 = time.time()
            it = T.make_item(task, b1, b2, seed, a._ans.tok, P.get(task), n_blocks=nb)
            assert it["item_id"] == XS.item_id_of(task, b1, b2, seed), it["item_id"]
            rows, info_item = make_rows(it)
            hl, intr = XS.host_level(), XS.intruders(watch)
            for r in rows:
                r.update(run_id=rid, ts=XS.ts(), shard=a.shard, model=a.model, n_rows_item=len(rows), own_gpu_intruders=intr,
                         host_level=hl, **extra_cols)
                w.writerow(r)
            fo.flush()
            chk.write(json.dumps(dict(run_id=rid, ts=XS.ts(), model=a.model, item_id=it["item_id"], **info_item,
                                      peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2), own_gpu_intruders=intr,
                                      host_level=hl), ensure_ascii=False, default=str) + "\n")
            chk.flush()
            n_new += 1
            print(f"[{k + 1}] {it['item_id']} rows={len(rows)} {time.time() - t0:.1f}s avg {(time.time() - t_all) / n_new:.1f}s/題 "
                  f"svd {info_item.get('t_svd')} peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            torch.cuda.empty_cache()
    finally:
        fo.close()
        chk.close()
        XS.finish_watch(watch, a.out)


def cmd_tf(a):
    import m10_sweep as SW
    from m10_quality import Answerer
    if a.smoke:
        a.tasks, a.n, a.shard = ["T2"], 1, "0/1"
    info = MODEL_INFO[a.model]
    cfgs = smoke_configs(a.model) if a.smoke else load_x4_configs(a.combos, a.model)
    print(f"X4a {a.model}：{len(cfgs)} 個配置、題目 seed0={a.seed0} n={a.n}／任務", flush=True)
    plan = XS.plan_interleaved(a.tasks, a.n, a.seed0)
    SW.N_BLOCKS = 64
    a._ans = Answerer(a.model)
    XS.check_info(a._ans.km, info)
    _loop(a, plan, lambda it: x4a_item(a._ans, it, cfgs, info), FIELDS_X4A, {})


def load_selection(path: str, model: str, nreg: int) -> list:
    sel = json.load(open(path))
    if not sel.get("go"):
        raise SystemExit(f"{path} 的 go＝false（{sel.get('reason')}）→ 不跑（規則 2：寫 NEEDS_HUMAN 的是 job）")
    return [dict(e, stages=validate(parse_spec(e["spec"]), MODEL_INFO[model], nreg)) for e in sel["models"][model]["configs"]]


def cmd_gen(a):
    import m10_sweep as SW
    from m10_quality import Answerer
    nb = a.n_blocks
    if a.smoke:
        a.tasks, a.n, a.shard = [a.tasks[0]], 1, "0/1"
    info = MODEL_INFO[a.model]
    cfgs = load_selection(a.selection, a.model, nb - 1)
    print(f"X4 gen {a.model} {CTX_NAME.get(nb, nb)}：{len(cfgs)} 個配置、{a.turns} 輪、題目 seed0={a.seed0} n={a.n}／任務", flush=True)
    plan = XS.plan_interleaved(a.tasks, a.n, a.seed0)
    SW.N_BLOCKS = nb
    a._ans = Answerer(a.model, extra=(nb - 64) * CHUNK + 4096)
    XS.check_info(a._ans.km, info)
    _loop(a, plan, lambda it: x4_gen_item(a._ans, it, cfgs, info, a.turns), FIELDS_X4G,
          dict(ctx=CTX_NAME.get(nb, str(nb)), n_blocks=nb), nb=nb)


def cmd_dump(a):
    """前緣配置（X4b 的計畫檔裡的 xl 配置）在 X4a 的題上的實際係數位元組 → 本地盤；每個（題, 配置）一個目錄（meta.json＋每個 part 一個 .bin）。"""
    import numpy as np
    import torch
    import m10_sweep as SW
    import m10_tasks as T
    from m10_quality import Answerer
    info = MODEL_INFO[a.model]
    cfgs = [c for c in load_selection(a.selection, a.model, NREG) if any(s["op"] == "xl" for s in c["stages"])]
    plan = XS.plan_interleaved(a.tasks, a.n, a.seed0)
    SW.N_BLOCKS = 64
    ans = Answerer(a.model)
    XS.check_info(ans.km, info)
    km = ans.km
    P = json.load(open(a.params_file))["params"][a.model]
    rid = os.environ.get("RUN_ID", "adhoc")
    for task, b1, b2, seed in plan:
        it = T.make_item(task, b1, b2, seed, ans.tok, P.get(task))
        km.key_keep = None
        st = SW.Item(ans, it)
        n = st.ctx_len
        lo, hi, nb = _region(km, n)
        nreg = hi - lo + 1
        s0, s1 = lo * CHUNK, (hi + 1) * CHUNK
        orders = _orders(km, st, it, cfgs, n)
        tctx = XS.TCtx(km, n, orders={m: [b - lo for b in o if lo <= b <= hi] for m, o in orders.items()}, deep=XS.deep_layers(km.L))
        pos = torch.arange(s0, s1, device=km.device)
        store = SvdStore(cfgs, km.H)
        X = km.kv[:, :, :, s0:s1]
        for c in cfgs:
            with torch.inference_mode():
                parts = x4_encode_parts(c["stages"], X, pos, tctx, info, store)
            pred = predict_bytes(c["stages"], info, nreg)
            tot = sum(v.nbytes for v, _ in parts.values())
            want = sum(v for k, v in pred.items() if not k.startswith("shared."))
            if tot != want:
                raise RuntimeError(f"{c['spec']}：dump 的位元組 {tot} ≠ predict_bytes {want}（規則 2：停）")
            d = os.path.join(a.out_dir, a.model, it["item_id"], c["spec"].replace(":", "_").replace("+", "__"))
            os.makedirs(d, exist_ok=True)
            meta = dict(run_id=rid, ts=XS.ts(), model=a.model, item_id=it["item_id"], spec=c["spec"], bytes_total=tot,
                        ratio_region=bytes_summary(pred, info, nreg, nb)["ratio_region"], parts={})
            for k, (v, kind) in parts.items():
                np.ascontiguousarray(v).tofile(os.path.join(d, f"{k}.bin"))
                meta["parts"][k] = dict(kind=kind, dtype=str(v.dtype), shape=list(v.shape), bytes=int(v.nbytes))
            json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1, ensure_ascii=False)
            print(f"dump {it['item_id']} {c['spec']}：{tot / 1e6:.1f} MB、{len(parts)} 個 part", flush=True)
        del st, store
        torch.cuda.empty_cache()


# ============================================================================ 表（CPU；不碰 torch）
X4_TEXT = {
    "Q-A": "（計畫 §3）某一軸加在「xKV＋INT4」上，同倍數帶（±10%）內答對率（TF 的 Δgold_lp；生成的配對差）沒有比「把同樣的位元組省在"
           "降秩或係數位元上」差 → 這一軸值得疊；差 → 不值得。計畫沒寫統計門檻，這裡寫死成：帶內「有這一軸的最好配置」−「沒有這一軸的最好配置」"
           "的逐題配對差（bootstrap 10,000 次）95% CI 上界 < 0 → 差（不值得疊）；其他 → 沒有比較差（值得疊）。無損：真實壓縮比（zstd:3）"
           "≥ 1.05 倍 → 值得疊（在慢連線上；速度由 X3 決定）。",
    "Q-B": "（計畫 §3）每個目標倍數下 TF 最好的分法進 X4b 生成；生成的最佳分法（帶內 X4b 配置裡答對率最高的）就是那個倍數的「最佳配比」。"
           "最佳配比在 ε 內的最大倍數＝極限。",
    "limit": "（計畫 §3）極限倍數 B*_ε（每模型、每上下文）＝ε 內（生成答對率對 E0 的配對差 95% CI 下界 ≥ −2 個百分點）最小的位元組比；"
             "另報 B*_99、B*_99.9（MLPerf 式：答對率 ≥ E0 的 99%／99.9%，點估計）。題數 < 48 的格子寫 INCOMPLETE，不報倍數。",
    "stack": "（計畫 §3）「疊加優於單降秩」：同一個倍數帶（位元組比相差 ±5% 內）最好的「秩＋量化」配置對最好的「只降秩」配置的生成配對差"
             " CI 下界 > 0 → 成立；沒有可比的一對 → NOT_COMPARABLE。",
    "long": "（計畫 §3）128K（256K）時前緣配置對 E0 的差在 ε 內 → 撐住；超出 → 寫出在哪個倍數開始掉。",
    "multiturn": "（計畫 §3）第 2 輪答對率對 E0 的差在 ε 內 → 多輪不壞；超出 → 「壓縮傷多輪」，並和 drop:KEYDIFF:0.5+q:int4_kivi 對照比"
                 "（配對差）。",
    "both_refs": "（計畫 §3）所有比較同時報「對 E0」與「對 q:int4_kivi」（同一題配對）。",
    "select_b": "（X4b 挑法，事先寫死）每個 T：X4a 裡倍數在 T 的 ±10% 內的 xKV 配置，平均 Δgold_lp 最高的（同分取位元組少的）＝帶內最好；"
                "另外為「疊加優於單降秩」放最多 2 個「只降秩」對照：依序看 T＝8、16、12、20、24、32，帶內最好的不是只降秩、而且有只降秩"
                "配置和它的位元組比相差 ±5% 內 → 放那裡面 Δgold_lp 最高的。加上 q:int4_kivi（錨點）。每個模型 X4a 至少 48 題才挑。",
    "select_cd": "（X4c／X4d 挑法，事先寫死）X4b 每個 T 的生成最佳配比；極限 T*＝最佳配比在 ε 內的最大 T（沒有 → 用最小的 T）。前緣 3 個＝"
                 "T* 的最佳配比、T* 的下一個較大 T 的最佳配比（T*＝最大 → 用下一個較小的）、T＝8 的最佳配比（和前兩個重複 → 依序 T＝12、16、… 第一個沒重複的）；"
                 "X4c＝前緣 3 個＋q:int4_kivi；X4d＝前緣前 2 個（時間）＋q:int4_kivi＋drop:KEYDIFF:0.5+q:int4_kivi。X4b 至少 48 題才挑。",
    "deviations": "和計畫不同的地方：共用 SVD（每題每（K／V、W）一次，q＝最大秩＋16，取前 r 欄；誤差 ≤ X1／X2 的每配置 SVD）；"
                  "輕度丟一律 xl+drop；**W 大（Llama 8、Qwen 7）縮掉**、W 只掃 {2, 4}（時間；計畫的 W 軸只剩「W 小」）；"
                  "輕度丟完整掃描只在 T∈{16,24}；X4d 的前緣只放 2 個（T* 與下一個較大的 T；時間）；X4d 用 C1a 題組的 T2–T4 各 32 題（96 題）。",
}


def _num(x):
    return None if x in ("", None) else float(x)


def tf_table(rows: list) -> dict:
    """X4a 一個模型的逐列 → {spec: 統計}；品質＝Δgold_lp（對 E0），題目為單位 bootstrap。per_item 留給配對比較。"""
    by = {}
    for r in rows:
        if r["spec"] != "E0":
            by.setdefault(r["spec"], {})[r["item_id"]] = r
    tab = {}
    for spec, items in by.items():
        rs = list(items.values())
        dl = {i: float(r["gold_lp"]) - float(r["gold_lp_e0"]) for i, r in items.items()}
        fo = {i: int(r["first_ok"]) - int(r["first_ok_e0"]) for i, r in items.items()}
        ratio = statistics.mean(float(r["ratio_region"]) for r in rs)
        tab[spec] = dict(spec=spec, n=len(rs), ratio=ratio, multiple=1 / ratio, **spec_meta(spec),
                         roles=sorted({x for r in rs for x in r["roles"].split("|") if x}),
                         deployable=int(rs[0]["deployable"]), d_gold_lp=boot2(dl.values()), d_first_ok=boot2(fo.values()),
                         kl_median=statistics.median(float(r["kl"]) for r in rs),
                         relerr_k=statistics.mean(float(r["relerr_k"]) for r in rs), relerr_v=statistics.mean(float(r["relerr_v"]) for r in rs),
                         first_ok=statistics.mean(int(r["first_ok"]) for r in rs), first_ok_e0=statistics.mean(int(r["first_ok_e0"]) for r in rs),
                         per_item=dl)
    return tab


def gen_tab(rows: list, anchor: str = "q:int4_kivi") -> dict:
    """生成逐列（一個模型、一個長度、一輪）→ {spec: 統計}：答對率、對 E0 與對錨點的逐題配對差、ε、MLPerf。"""
    e0 = {r["item_id"]: float(r["gen_score"]) for r in rows if r["spec"] == "E0"}
    by = {}
    for r in rows:
        if r["spec"] != "E0" and r["item_id"] in e0:
            by.setdefault(r["spec"], {})[r["item_id"]] = r
    tab = {}
    for spec, items in by.items():
        sc = {i: float(r["gen_score"]) for i, r in items.items()}
        rs = list(items.values())
        ratio = statistics.mean(float(r["ratio_region"]) for r in rs)
        acc, acc0 = statistics.mean(sc.values()), statistics.mean(e0[i] for i in sc)
        tab[spec] = dict(spec=spec, n=len(sc), ratio=ratio, multiple=1 / ratio, **spec_meta(spec),
                         roles=sorted({x for r in rs for x in r["roles"].split("|") if x}), deployable=int(rs[0]["deployable"]),
                         acc=acc, acc_e0=acc0, diff=boot2([sc[i] - e0[i] for i in sc]), per_item=sc,
                         **{f"mlperf{th}": bool(acc >= th * acc0) for th in MLPERF})
        tab[spec]["eps_pass"] = bool(tab[spec]["diff"]["lo"] >= -EPS)
    if anchor in tab:
        A = tab[anchor]["per_item"]
        for c in tab.values():
            com = sorted(set(c["per_item"]) & set(A))
            c["diff_vs_anchor"] = boot2([c["per_item"][i] - A[i] for i in com]) if com and c["spec"] != anchor else None
    return tab


def paired(a: dict, b: dict):
    com = sorted(set(a) & set(b))
    return boot2([a[i] - b[i] for i in com]) if com else None


def qual_tf(c) -> float:
    return c["d_gold_lp"]["mean"] if c.get("d_gold_lp") else -math.inf


def band_of(c, T, tol=BAND_TOL) -> bool:
    return bool(c.get("xl")) and in_band(c["multiple"], T, tol)


def is_rank_only(c) -> bool:
    return bool(c.get("xl")) and c["bits"] == "bf16" and not c["drop_frac"]


def select_b(tab: dict, model: str) -> list:
    """X4_TEXT['select_b']。tab＝tf_table。回傳 X4b 的配置清單（含 q:int4_kivi 錨點）。"""
    sel = {}

    def add(c, role, T=None):
        e = sel.setdefault(c["spec"], dict(spec=c["spec"], roles=[], tiers=["core"], budgets=[], targets=[],
                                           x4a=dict(multiple=round(c["multiple"], 3), d_gold_lp=c["d_gold_lp"], d_first_ok=c["d_first_ok"],
                                                    n=c["n"])))
        if role not in e["roles"]:
            e["roles"].append(role)
        if T is not None and T not in e["targets"]:
            e["targets"].append(T)
    cs = [c for c in tab.values() if c["deployable"]]
    best = {}
    for T in TARGETS:
        cand = [c for c in cs if band_of(c, T)]
        if cand:
            best[T] = max(cand, key=lambda c: (qual_tf(c), -c["ratio"], c["spec"]))
            add(best[T], f"band_best:{T}", T)
    n_ro = 0
    for T in (8, 16, 12, 20, 24, 32):
        if n_ro >= 2 or T not in best or is_rank_only(best[T]):
            continue
        ro = [c for c in cs if is_rank_only(c) and abs(c["ratio"] / best[T]["ratio"] - 1) <= STACK_TOL]
        if ro:
            add(max(ro, key=lambda c: (qual_tf(c), -c["ratio"], c["spec"])), f"rank_only_ref:{T}", T)
            n_ro += 1
    if "q:int4_kivi" in tab:
        add(tab["q:int4_kivi"], "anchor_int4")
    return sorted(sel.values(), key=lambda e: (e["x4a"]["multiple"], e["spec"]))


def gen_best_by_T(gtab: dict, sel_cfgs: list) -> dict:
    """X4b：每個 T 帶內（±10%）X4b 配置裡答對率最高的（同分取位元組少的）。"""
    out = {}
    for T in TARGETS:
        cand = [c for c in gtab.values() if band_of(c, T)]
        if cand:
            out[T] = max(cand, key=lambda c: (c["acc"], -c["ratio"], c["spec"]))
    return out


def select_cd(gtab: dict, n_items: int) -> dict:
    """X4_TEXT['select_cd']。回傳 {frontier: [spec…], tstar, why, x4c: [...], x4d: [...]}。"""
    bb = gen_best_by_T(gtab, [])
    if n_items < MIN_ITEMS or not bb:
        return dict(go=False, why=f"X4b 題數 {n_items}（< {MIN_ITEMS}）或沒有帶內配置")
    ok = [T for T, c in bb.items() if c["eps_pass"]]
    tstar = max(ok) if ok else min(bb)
    Ts = sorted(bb)
    nxt = next((T for T in Ts if T > tstar), None) or (max(T for T in Ts if T < tstar) if any(T < tstar for T in Ts) else None)
    picks, why = [], []
    for T, w in ((tstar, "T*（ε 內最大的 T）的最佳配比" if ok else "沒有 T 在 ε 內 → 用最小的 T"), (nxt, "T* 的下一個較大的 T（T*＝最大 → 下一個較小）")):
        if T is not None and T in bb and bb[T]["spec"] not in picks:
            picks.append(bb[T]["spec"])
            why.append(dict(T=T, spec=bb[T]["spec"], why=w, acc=bb[T]["acc"], diff=bb[T]["diff"], multiple=bb[T]["multiple"]))
    T3 = next((T for T in Ts if bb[T]["spec"] not in picks), None)          # T＝8；重複 → 依序 12、16、… 第一個沒重複的
    if T3 is not None:
        picks.append(bb[T3]["spec"])
        why.append(dict(T=T3, spec=bb[T3]["spec"], why="T＝8 的最佳配比" if T3 == 8 else f"T＝8 重複 → 依序取第一個沒重複的（T＝{T3}）",
                        acc=bb[T3]["acc"], diff=bb[T3]["diff"], multiple=bb[T3]["multiple"]))
    return dict(go=True, tstar=tstar, frontier=picks, why=why, x4c=picks + ["q:int4_kivi"],
                x4d=picks[:2] + ["q:int4_kivi", "drop:KEYDIFF:0.5+q:int4_kivi"])


def _sel_doc(which, rule, models, src):
    return dict(go=all(v.get("go", True) for v in models.values()), which=which, rule=rule, src_run_ids=src,
                run_id=os.environ.get("RUN_ID", "adhoc"), ts=XS.ts(),
                models={m: dict(configs=v["configs"]) for m, v in models.items() if v.get("configs")},
                detail={m: {k: v for k, v in d.items() if k != "configs"} for m, d in models.items()},
                reason="；".join(f"{m}：{v.get('why')}" for m, v in models.items() if not v.get("go", True)) or "ok")


def _read(globs, keys):
    paths = sorted({p for g in globs for p in glob.glob(g)})
    with contextlib.redirect_stdout(sys.stderr):
        rows, partial = XS.dedupe_complete(XS.read_rows(paths), keys)
    return rows, partial, paths


def _write_sel(path, doc):
    """計畫檔：已存在而內容（configs）不同 → 停（不默默改）；相同 → 不重寫。"""
    if os.path.exists(path):
        old = json.load(open(path))
        if old.get("models") == json.loads(json.dumps(doc["models"], default=str)):
            print(f"{path} 已有、內容相同（run {old.get('run_id')}）→ 不重寫", flush=True)
            return
        raise SystemExit(f"{path} 已存在但內容不同 → 不默默改（X4a／X4b 的資料在挑完之後變了？要人看）")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json.dump(doc, open(path, "w"), indent=1, ensure_ascii=False, default=str)
    print(f"寫好 {path}：" + "；".join(f"{m} {len(v['configs'])} 個" for m, v in doc["models"].items()), flush=True)


def cmd_select_b(a):
    rows, partial, paths = _read(a.raw, ("model", "item_id", "spec"))
    models = {}
    for m in MODEL_INFO:
        R = [r for r in rows if r["model"] == m]
        n = len({r["item_id"] for r in R if r["spec"] == "E0"})
        if n < MIN_ITEMS:
            models[m] = dict(go=False, why=f"X4a {n} 題 < {MIN_ITEMS}")
            continue
        cf = select_b(tf_table(R), m)
        models[m] = dict(go=True, n_items=n, configs=[{k: v for k, v in e.items()} for e in cf])
    doc = _sel_doc("x4b", X4_TEXT["select_b"], models, sorted({r["run_id"] for r in rows}))
    doc.update(seed0=a.seed0, n=a.n, tasks=["T2", "T3", "T4", "T5"], incomplete_items=partial)
    _write_sel(a.out, doc)
    if not doc["go"]:
        raise SystemExit(f"X4b 不能挑：{doc['reason']}")


def cmd_select_cd(a):
    rows, partial, paths = _read(a.raw, ("model", "ctx", "item_id", "spec", "turn"))
    rows = [r for r in rows if r.get("ctx") == "32K" and r.get("turn", "1") in ("1", "")]
    docs = {"x4c": {}, "x4d": {}}
    for m in MODEL_INFO:
        R = [r for r in rows if r["model"] == m]
        n = len({r["item_id"] for r in R if r["spec"] == "E0"})
        s = select_cd(gen_tab(R), n)
        for k in docs:
            if not s.get("go"):
                docs[k][m] = dict(go=False, why=s["why"])
                continue
            docs[k][m] = dict(go=True, n_items=n, tstar=s["tstar"], why_frontier=s["why"],
                              configs=[dict(spec=x, roles=["frontier" if x in s["frontier"] else "control"], tiers=["core"], budgets=[])
                                       for x in s[k]])
    for k, path in (("x4c", a.out_c), ("x4d", a.out_d)):
        doc = _sel_doc(k, X4_TEXT["select_cd"], docs[k], sorted({r["run_id"] for r in rows}))
        _write_sel(path, doc)
    if not all(v.get("go") for v in docs["x4c"].values()):
        raise SystemExit("X4c／X4d 不能挑：" + "；".join(f"{m} {v.get('why')}" for m, v in docs["x4c"].items() if not v.get("go")))


# ============================================================================ 無損（CPU；m16_s1_lossless 的函式）
LL_CODECS = ("zstd:-3", "zstd:1", "zstd:3")
LL_HEAD = "zstd:3"                                   # 判定用（和 X1／X2 的無損乘數同一個 codec）
LL_WORTH = 1.05


def lossless_dir(d: str, codecs=LL_CODECS, threads=(1, 4), frame_bytes=4 << 20, reps=3) -> dict:
    """一個 dump 目錄（一題、一個配置）：每個 part 壓縮後的位元組（BF16 的 part 用 S1 的 planes 變體：exponent／sign＋mantissa 分開壓；
    其他整段壓），每次都解回來逐位元比對（S1.roundtrip_len）；解壓速度＝把全部 part 的 stream 切 4 MiB frame、k 個 thread（未壓縮 MB/s）。"""
    import numpy as np
    import m16_s1_lossless as S1
    meta = json.load(open(os.path.join(d, "meta.json")))
    streams, per_part, stored = [], {}, 0
    for k, pm in meta["parts"].items():
        raw = np.fromfile(os.path.join(d, f"{k}.bin"), dtype=np.uint8)
        assert raw.nbytes == pm["bytes"], (d, k)
        stored += raw.nbytes
        if pm["kind"] == "bf16":
            ss = [s for s, z in S1.split_variant(raw.view(np.uint16), "planes")]
        else:
            ss = [raw]
        streams += ss
        per_part[k] = {cs: sum(S1.roundtrip_len(s, cs) for s in ss) for cs in codecs}
        per_part[k]["stored"] = raw.nbytes
    comp = {cs: sum(v[cs] for v in per_part.values()) for cs in codecs}
    out = dict(spec=meta["spec"], item_id=meta["item_id"], model=meta["model"], dump_run_id=meta["run_id"], stored=stored,
               comp=comp, ratio={cs: round(stored / comp[cs], 4) for cs in codecs},
               part_ratio={k: round(v["stored"] / v[LL_HEAD], 4) for k, v in per_part.items()}, speed={})
    import concurrent.futures as cf
    import threading
    for cs in (LL_HEAD, "zstd:-3"):
        for th in threads:
            tls = threading.local()

            def ctx():
                if not hasattr(tls, "c"):
                    tls.c = S1.make_codec(cs)
                return tls.c
            frs = [f for s in streams for f in S1._frames(np.ascontiguousarray(s), frame_bytes)]
            n = sum(f.nbytes for f in frs)
            tc, td = [], []
            with cf.ThreadPoolExecutor(max_workers=th) as pool:
                for _ in range(reps):
                    t0 = time.perf_counter()
                    comp_fr = list(pool.map(lambda f: ctx()[0](f.tobytes()), frs))
                    t1 = time.perf_counter()
                    dec = list(pool.map(lambda c: ctx()[1](c), comp_fr))
                    t2 = time.perf_counter()
                    assert all(x == f.tobytes() for x, f in zip(dec, frs)), "解回來不相同"
                    tc.append(n / (t1 - t0) / 1e6)
                    td.append(n / (t2 - t1) / 1e6)
            out["speed"][f"{cs}|t{th}"] = dict(comp_MBps=round(statistics.median(tc), 1), decomp_MBps=round(statistics.median(td), 1))
    return out


def cmd_lossless(a):
    dirs = sorted({os.path.dirname(p) for g in a.dumps for p in glob.glob(os.path.join(g, "**", "meta.json"), recursive=True)})
    if not dirs:
        raise SystemExit(f"找不到 dump（{a.dumps}）：38f1 的 dump 沒跑、或本地盤被清掉了（容器重啟）→ 要重跑 dump")
    rid = os.environ.get("RUN_ID", "adhoc")
    res = [lossless_dir(d, threads=tuple(a.threads), reps=a.reps) for d in dirs]
    summ = {}
    for r in res:
        e = summ.setdefault(r["model"], {}).setdefault(r["spec"], dict(stored=0, comp={cs: 0 for cs in LL_CODECS}, items=[], speed=[]))
        e["stored"] += r["stored"]
        for cs in LL_CODECS:
            e["comp"][cs] += r["comp"][cs]
        e["items"].append(r["item_id"])
        e["speed"].append(r["speed"])
    for m, d in summ.items():
        for s, e in d.items():
            e["ratio"] = {cs: round(e["stored"] / e["comp"][cs], 4) for cs in LL_CODECS}
            e["worth_stacking"] = bool(e["ratio"][LL_HEAD] >= LL_WORTH)
            e["decomp_MBps_median"] = {k: statistics.median(x[k]["decomp_MBps"] for x in e["speed"]) for k in e["speed"][0]}
            del e["speed"]
    out = dict(run_id=rid, ts=XS.ts(), rule=X4_TEXT["Q-A"], head_codec=LL_HEAD, worth=LL_WORTH, n_dumps=len(dirs),
               affinity=sorted(os.sched_getaffinity(0)), summary=summ, per_dump=res)
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(a.out_dir, "lossless_x4.json"), "w"), indent=1, ensure_ascii=False)
    for m, d in summ.items():
        for s, e in sorted(d.items()):
            print(f"{m} {s}：zstd:3 {e['ratio'][LL_HEAD]}×（{len(e['items'])} 題）解壓 {e['decomp_MBps_median']}", flush=True)


# ============================================================================ report（CPU）
QA_AXES = {
    "drop": ("輕度丟（任何 KEYDIFF／KNORM 比例）", lambda c: c["drop_frac"] > 0, lambda c: not c["drop_frac"]),
    "quant": ("量化係數（任何位元）vs 只降秩（BF16 係數）", lambda c: c["bits"] != "bf16" and not c["drop_frac"], is_rank_only),
    "int4_vs_int8": ("INT4 vs INT8 係數", lambda c: c["bits"] == "int4" and not c["drop_frac"],
                     lambda c: c["bits"] == "int8" and not c["drop_frac"]),
    "lt4_vs_int4": ("低於 4 bit（INT2、K4V2、K2V4）vs INT4", lambda c: c["bits"] in ("int2", "k4v2", "k2v4") and not c["drop_frac"],
                    lambda c: c["bits"] == "int4" and not c["drop_frac"]),
    "w_small": ("W＝2 vs W＝4（INT4）", lambda c: c["W"] == 2 and c["bits"] == "int4" and not c["drop_frac"],
                lambda c: c["W"] == 4 and c["bits"] == "int4" and not c["drop_frac"]),
    "w_big": ("W 大（8／7）vs W＝4（INT4）", lambda c: c["W"] > 4 and c["bits"] == "int4" and not c["drop_frac"],
              lambda c: c["W"] == 4 and c["bits"] == "int4" and not c["drop_frac"]),
    "kv_split": ("K 少 V 多（V/K ≥ 1.7）vs 1:1.5（INT4、W＝4）", lambda c: c["bits"] == "int4" and c["W"] == 4 and not c["drop_frac"]
                 and c["rk"] and c["rv"] / c["rk"] >= 1.7,
                 lambda c: c["bits"] == "int4" and c["W"] == 4 and not c["drop_frac"] and c["rk"] and 1.4 <= c["rv"] / c["rk"] <= 1.6),
}


def qa_axes(tab: dict, quality, per_item_key: str) -> dict:
    """每一軸、每個 T：帶內「有這一軸的最好」vs「沒有這一軸的最好」的逐題配對差與判定（X4_TEXT['Q-A']）。quality(c)＝挑最好用的分數。"""
    out = {}
    for ax, (desc, has, hasnt) in QA_AXES.items():
        e = out[ax] = dict(desc=desc, by_T={})
        for T in TARGETS:
            band = [c for c in tab.values() if band_of(c, T) and c["deployable"]]
            w = [c for c in band if has(c)]
            wo = [c for c in band if hasnt(c)]
            if not w or not wo:
                e["by_T"][str(T)] = dict(status="NOT_COMPARABLE（帶內缺一邊）", n_with=len(w), n_without=len(wo))
                continue
            bw, bwo = max(w, key=quality), max(wo, key=quality)
            d = paired(bw[per_item_key], bwo[per_item_key])
            e["by_T"][str(T)] = dict(best_with=bw["spec"], best_without=bwo["spec"], multiple_with=round(bw["multiple"], 2),
                                     multiple_without=round(bwo["multiple"], 2), diff_with_minus_without=d,
                                     verdict=None if d is None else ("不值得疊（差）" if d["hi"] < 0 else "值得疊（沒有比較差）"))
        vs = [x.get("verdict") for x in e["by_T"].values() if x.get("verdict")]
        e["summary"] = dict(n_compared=len(vs), worth=sum(1 for v in vs if v.startswith("值得")), not_worth=sum(1 for v in vs if v.startswith("不值得")))
    return out


def base_curve(tab: dict) -> list:
    """底（alloc:int4＝xKV＋INT4、W＝4、1:1.5、不丟）沿 T 的 TF 品質與每多壓一倍（log2）掉多少 Δgold_lp（邊際代價）。"""
    pts = sorted([c for c in tab.values() if "alloc:int4" in c["roles"]], key=lambda c: c["multiple"])
    out = []
    for i, c in enumerate(pts):
        e = dict(spec=c["spec"], multiple=round(c["multiple"], 2), d_gold_lp=c["d_gold_lp"], d_first_ok=c["d_first_ok"])
        if i:
            p = pts[i - 1]
            e["slope_per_doubling"] = round((qual_tf(c) - qual_tf(p)) / math.log2(c["multiple"] / p["multiple"]), 4)
        out.append(e)
    return out


def limits(gtab: dict, n_items: int) -> dict:
    """B*_ε、B*_99、B*_99.9（X4_TEXT['limit']）；題數不夠 → INCOMPLETE。只看可部署的有損配置（不含 E0／none）。"""
    if n_items < MIN_ITEMS:
        return dict(status=f"INCOMPLETE（{n_items} 題 < {MIN_ITEMS}）")
    cs = [c for c in gtab.values() if c["deployable"] and c["spec"] != "none"]
    out = {}
    for name, ok in (("eps", lambda c: c["eps_pass"]), ("mlperf99", lambda c: c["mlperf0.99"]), ("mlperf999", lambda c: c["mlperf0.999"])):
        p = [c for c in cs if ok(c)]
        b = min(p, key=lambda c: (c["ratio"], c["spec"])) if p else None
        out[name] = None if b is None else dict(spec=b["spec"], ratio=round(b["ratio"], 6), multiple=round(b["multiple"], 2),
                                                acc=b["acc"], diff=b["diff"])
    return out


def stack_vs_rank_only(gtab: dict) -> list:
    """X4_TEXT['stack']：每個只降秩的配置，找位元組比 ±5% 內最好的「秩＋量化」配置（不丟）。"""
    out = []
    for ro in sorted((c for c in gtab.values() if is_rank_only(c)), key=lambda c: c["ratio"]):
        st = [c for c in gtab.values() if c.get("xl") and c["bits"] != "bf16" and not c["drop_frac"]
              and abs(c["ratio"] / ro["ratio"] - 1) <= STACK_TOL]
        if not st:
            out.append(dict(rank_only=ro["spec"], status="NOT_COMPARABLE（±5% 內沒有秩＋量化）"))
            continue
        b = max(st, key=lambda c: (c["acc"], -c["ratio"]))
        d = paired(b["per_item"], ro["per_item"])
        out.append(dict(rank_only=ro["spec"], stacked=b["spec"], multiple_rank_only=round(ro["multiple"], 2),
                        multiple_stacked=round(b["multiple"], 2), diff_stacked_minus_rank_only=d,
                        holds=None if d is None else bool(d["lo"] > 0)))
    return out


def _strip(tab):
    return {s: {k: v for k, v in c.items() if k != "per_item"} for s, c in tab.items()}


def plot_tf(tab, model, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as ex:
        return f"NOT_PLOTTED（{ex}）"
    col = {"bf16": "#4C78A8", "int8": "#72B7B2", "int4": "#E45756", "k4v2": "#F58518", "k2v4": "#B279A2", "int2": "#9D755D"}
    mk = {2: "s", 4: "o", 7: "^", 8: "^"}
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    xl = [c for c in tab.values() if c.get("xl")]
    for c in xl:
        ax.scatter(c["multiple"], qual_tf(c), color=col.get(c["bits"], "#888888"), marker=mk.get(c["W"], "o"), s=22,
                   facecolors="none" if c["drop_frac"] else None, alpha=0.85)
    front = XS.pareto_front([(c["ratio"], qual_tf(c), c["spec"]) for c in xl])
    fp = sorted((tab[s]["multiple"], qual_tf(tab[s])) for s in front)
    ax.plot([x for x, _ in fp], [y for _, y in fp], "-", color="#222222", lw=1)
    for T in TARGETS:
        ax.axvline(T, color="#BBBBBB", lw=0.7, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("compression multiple vs BF16 (region bytes, predict_bytes)")
    ax.set_ylabel("mean Δ gold log-prob vs E0 (teacher forcing)")
    ax.set_title(f"X4a {model}: color=coef bits (blue bf16, teal int8, red int4, orange K4V2, purple K2V4, brown int2); "
                 "marker=W (s 2, o 4, ^ 7/8); hollow=light drop", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_gen(tabs: dict, model, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as ex:
        return f"NOT_PLOTTED（{ex}）"
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    sty = {"32K": ("o", "#E45756"), "128K": ("s", "#4C78A8"), "256K": ("^", "#54A24B")}
    for ctx, tab in tabs.items():
        m, col = sty.get(ctx, ("o", "#888888"))
        cs = sorted((c for c in tab.values() if c["spec"] != "none"), key=lambda c: c["multiple"])
        for c in cs:
            d = c["diff"]
            ax.errorbar(c["multiple"], 100 * d["mean"], yerr=[[100 * (d["mean"] - d["lo"])], [100 * (d["hi"] - d["mean"])]],
                        fmt=m, color=col, ms=4, capsize=2, label=ctx if c is cs[0] else None)
    ax.axhline(-100 * EPS, color="#222222", lw=0.8, ls="--")
    ax.axhline(0, color="#BBBBBB", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("compression multiple vs BF16 (region bytes)")
    ax.set_ylabel("generation accuracy − E0 (pp, 95% CI)")
    ax.set_title(f"X4 generation {model} (dashed = −ε)", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _ci(d, s=100, nd=1):
    return XS._ci(d, s, nd)


def cmd_report(a):
    if os.path.abspath(a.out_dir) in (os.path.abspath(XS.OUT_DIR), os.path.abspath(XS.OUT_DIR_V2)):
        raise SystemExit("X4 的報告只寫到 x_stack/x4/（不覆蓋第一版與 v2 的檔）")
    os.makedirs(a.out_dir, exist_ok=True)
    rid = os.environ.get("RUN_ID", "adhoc")
    ra, pa, fa = _read(a.raw_tf, ("model", "item_id", "spec"))
    rg, pg, fg = _read(a.raw_gen, ("model", "ctx", "item_id", "spec", "turn"))
    rd, pd, fd = _read(a.raw_turns, ("model", "ctx", "item_id", "spec", "turn"))
    grid = json.load(open(a.combos)) if os.path.exists(a.combos) else {}
    ll = json.load(open(a.lossless)) if a.lossless and os.path.exists(a.lossless) else None
    rep = dict(run_id=rid, ts=XS.ts(), rule_text=X4_TEXT, eps=EPS, targets=TARGETS, band_tol=BAND_TOL,
               files=dict(tf=fa, gen=fg, turns=fd, lossless=a.lossless if ll else "NOT_FOUND"),
               src_run_ids=sorted({r["run_id"] for r in ra + rg + rd}), incomplete_items=dict(tf=pa, gen=pg, turns=pd),
               contention=dict(rows_with_intruders=sum(1 for r in ra + rg + rd if r.get("own_gpu_intruders") not in ("0", "NOT_CHECKED", None)),
                               host_levels=sorted({r.get("host_level", "") for r in ra + rg + rd})), models={})
    L = ["# X4：把「xKV 式跨層低秩＋量化」推到極限（10/21；判定事先寫死，見最後）", "",
         f"run：{len(rep['src_run_ids'])} 個；不齊全的題（沒用）：TF {len(pa)}、生成 {len(pg)}、兩輪 {len(pd)}", ""]
    best_rows = []
    for m in MODEL_INFO:
        M = rep["models"][m] = {}
        A = [r for r in ra if r["model"] == m]
        na = len({r["item_id"] for r in A if r["spec"] == "E0"})
        tab = tf_table(A)
        M["x4a"] = dict(n_items=na, configs=_strip(tab), skipped=grid.get("models", {}).get(m, {}).get("skipped"))
        tf_best = {}
        for T in TARGETS:
            cand = [c for c in tab.values() if band_of(c, T) and c["deployable"]]
            if cand:
                tf_best[T] = max(cand, key=lambda c: (qual_tf(c), -c["ratio"], c["spec"]))
        M["x4a"]["best_by_T"] = {str(T): dict(spec=c["spec"], multiple=round(c["multiple"], 2), d_gold_lp=c["d_gold_lp"],
                                              d_first_ok=c["d_first_ok"]) for T, c in tf_best.items()}
        M["x4a"]["qa"] = qa_axes(tab, lambda c: (qual_tf(c), -c["ratio"]), "per_item")
        M["x4a"]["base_curve"] = base_curve(tab)
        M["x4a"]["pareto_png"] = plot_tf(tab, m, os.path.join(a.out_dir, f"pareto_x4a_{m}.png")) if tab else "NO_DATA"
        # ---- 生成：32K（X4b）、128K／256K（X4c）、兩輪（X4d）
        gt, gsum = {}, {}
        for ctx in ("32K", "128K", "256K"):
            G = [r for r in rg if r["model"] == m and r.get("ctx") == ctx and r.get("turn", "1") in ("1", "")]
            if not G:
                continue
            n = len({r["item_id"] for r in G if r["spec"] == "E0"})
            t = gen_tab(G)
            gt[ctx] = t
            gsum[ctx] = dict(n_items=n, configs=_strip(t), limits=limits(t, n))
            if ctx == "32K":
                bb = gen_best_by_T(t, [])
                gsum[ctx]["best_by_T"] = {str(T): dict(spec=c["spec"], multiple=round(c["multiple"], 2), acc=c["acc"], diff=c["diff"],
                                                       diff_vs_int4=c.get("diff_vs_anchor"), eps_pass=c["eps_pass"],
                                                       mlperf99=c["mlperf0.99"], mlperf999=c["mlperf0.999"]) for T, c in bb.items()}
                fail = sorted(T for T, c in bb.items() if not c["eps_pass"])
                gsum[ctx]["first_T_out_of_eps"] = fail[0] if fail else None
                gsum[ctx]["stack_vs_rank_only"] = stack_vs_rank_only(t)
                gsum[ctx]["qa"] = qa_axes(t, lambda c: (c["acc"], -c["ratio"]), "per_item")
            else:
                gsum[ctx]["holds"] = {s: dict(multiple=round(c["multiple"], 2), eps_pass=c["eps_pass"], diff=c["diff"],
                                              diff_vs_int4=c.get("diff_vs_anchor")) for s, c in t.items()}
                if n < MIN_ITEMS:
                    gsum[ctx]["status"] = f"INCOMPLETE（{n} 題 < {MIN_ITEMS}；只供參考）"
        M["gen"] = gsum
        M["pareto_gen_png"] = plot_gen(gt, m, os.path.join(a.out_dir, f"pareto_x4b_{m}.png")) if gt else "NO_DATA"
        D = [r for r in rd if r["model"] == m]
        if D:
            md = M["x4d"] = {}
            for turn in ("1", "2"):
                R = [r for r in D if r["turn"] == turn]
                n = len({r["item_id"] for r in R if r["spec"] == "E0"})
                t = gen_tab(R)
                drop = t.get("drop:KEYDIFF:0.5+q:int4_kivi")
                md[turn] = dict(n_items=n, configs=_strip(t),
                                vs_drop={s: paired(c["per_item"], drop["per_item"]) for s, c in t.items() if drop and s != drop["spec"]})
                if turn == "2":
                    md["verdict_turn2"] = {s: ("INCOMPLETE" if n < MIN_ITEMS else "多輪不壞" if c["eps_pass"] else "壓縮傷多輪")
                                           for s, c in t.items()}
        if ll:
            M["lossless"] = ll["summary"].get(m, "NOT_MEASURED")
        # ---- 目標倍數 → 最佳配比
        g32 = gsum.get("32K", {})
        for T in TARGETS:
            tb = M["x4a"]["best_by_T"].get(str(T))
            gb = g32.get("best_by_T", {}).get(str(T))
            lm = (ll or {}).get("summary", {}).get(m, {}).get(gb["spec"]) if gb else None
            best_rows.append(dict(model=m, T=T, tf_best=tb and tb["spec"], tf_multiple=tb and tb["multiple"],
                                  tf_d_gold_lp=tb and tb["d_gold_lp"]["mean"], gen_best=gb and gb["spec"], gen_multiple=gb and gb["multiple"],
                                  gen_acc=gb and gb["acc"], gen_diff=gb and gb["diff"]["mean"], gen_diff_lo=gb and gb["diff"]["lo"],
                                  gen_diff_hi=gb and gb["diff"]["hi"], eps=gb and gb["eps_pass"], mlperf99=gb and gb["mlperf99"],
                                  mlperf999=gb and gb["mlperf999"], lossless_ratio=lm and lm["ratio"][LL_HEAD],
                                  multiple_with_lossless=round(gb["multiple"] * lm["ratio"][LL_HEAD], 2) if (gb and lm) else None))
    # ---- 寫檔
    with open(os.path.join(a.out_dir, "x4_best_allocation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "ts"] + list(best_rows[0]) if best_rows else ["run_id", "ts"])
        w.writeheader()
        for r in best_rows:
            w.writerow(dict(r, run_id=rid, ts=rep["ts"]))
    for m in MODEL_INFO:
        cs = rep["models"][m]["x4a"]["configs"]
        if not cs:
            continue
        with open(os.path.join(a.out_dir, f"x4a_configs_{m}.csv"), "w", newline="") as f:
            cols = ["run_id", "ts", "model", "spec", "roles", "W", "rk", "rv", "bits", "drop_method", "drop_frac", "n", "ratio", "multiple",
                    "d_gold_lp", "d_gold_lp_lo", "d_gold_lp_hi", "d_first_ok", "d_first_ok_lo", "d_first_ok_hi", "kl_median", "relerr_k", "relerr_v"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for s, c in sorted(cs.items(), key=lambda x: x[1]["ratio"]):
                w.writerow(dict(run_id=rid, ts=rep["ts"], model=m, spec=s, roles="|".join(c["roles"]), W=c["W"], rk=c["rk"], rv=c["rv"],
                                bits=c["bits"], drop_method=c["drop_method"], drop_frac=c["drop_frac"], n=c["n"], ratio=round(c["ratio"], 6),
                                multiple=round(c["multiple"], 3), d_gold_lp=c["d_gold_lp"]["mean"], d_gold_lp_lo=c["d_gold_lp"]["lo"],
                                d_gold_lp_hi=c["d_gold_lp"]["hi"], d_first_ok=c["d_first_ok"]["mean"], d_first_ok_lo=c["d_first_ok"]["lo"],
                                d_first_ok_hi=c["d_first_ok"]["hi"], kl_median=c["kl_median"], relerr_k=c["relerr_k"], relerr_v=c["relerr_v"]))
    json.dump(rep, open(os.path.join(a.out_dir, "x4_report.json"), "w"), indent=1, ensure_ascii=False, default=str)
    # ---- tables_x4.md
    for m in MODEL_INFO:
        M = rep["models"][m]
        L += [f"## {m}", "", f"X4a：{M['x4a']['n_items']} 題、{len(M['x4a']['configs'])} 個配置；"
              + "；".join(f"{ctx} 生成 {v['n_items']} 題" for ctx, v in M["gen"].items()) + (f"；兩輪 {M['x4d']['2']['n_items']} 題" if "x4d" in M else ""), "",
              "### 目標倍數 → 最佳配比", "",
              "| T | TF 最好（X4a） | 倍數 | Δgold_lp [CI] | 生成最佳配比（X4b） | 倍數 | 對 E0（百分點） | ε | 對 INT4（百分點） | ≥99% | ≥99.9% | ×無損（zstd:3） |",
              "|--:|:--|--:|--:|:--|--:|--:|:--|--:|:--|:--|--:|"]
        for r in [x for x in best_rows if x["model"] == m]:
            tb = M["x4a"]["best_by_T"].get(str(r["T"]))
            gb = M["gen"].get("32K", {}).get("best_by_T", {}).get(str(r["T"]))
            L.append(f"| {r['T']} | {r['tf_best'] or '—'} | {r['tf_multiple'] or '—'} | {XS._ci(tb['d_gold_lp']) if tb else '—'} | "
                     f"{r['gen_best'] or '—'} | {r['gen_multiple'] or '—'} | {_ci(gb['diff']) if gb else '—'} | "
                     f"{'—' if not gb else ('過' if gb['eps_pass'] else '沒過')} | {_ci(gb['diff_vs_int4']) if gb and gb['diff_vs_int4'] else '—'} | "
                     f"{'—' if not gb else ('過' if gb['mlperf99'] else '沒過')} | {'—' if not gb else ('過' if gb['mlperf999'] else '沒過')} | "
                     f"{r['multiple_with_lossless'] or ('NOT_MEASURED' if gb else '—')} |")
        L += ["", "### 極限（ε 內最小位元組比；MLPerf 99%／99.9% 另報）", "", "| 長度 | B*_ε（倍數） | B*_99 | B*_99.9 | 最佳配比第一個掉出 ε 的 T |", "|:--|:--|:--|:--|--:|"]
        for ctx, v in M["gen"].items():
            lm = v["limits"]
            if "status" in lm:
                L.append(f"| {ctx} | {lm['status']} | | | |")
                continue

            def fm(x):
                return "沒有" if not x else f"{x['spec']}（{x['multiple']}×）"
            L.append(f"| {ctx} | {fm(lm['eps'])} | {fm(lm['mlperf99'])} | {fm(lm['mlperf999'])} | {v.get('first_T_out_of_eps', '—')} |")
        L += ["", "### Q-A：哪一軸值得疊在 xKV＋INT4 上（TF，帶內最好的配對；判定見最後）", "",
              "| 軸 | " + " | ".join(f"T={T}" for T in TARGETS) + " |", "|:--|" + "--:|" * len(TARGETS)]
        for ax, e in M["x4a"]["qa"].items():
            cells = []
            for T in TARGETS:
                x = e["by_T"][str(T)]
                cells.append("—" if "verdict" not in x or x["verdict"] is None else
                             f"{'值得' if x['verdict'].startswith('值得') else '不值得'} {XS._ci(x['diff_with_minus_without'], 1, 2)}")
            L.append(f"| {e['desc']} | " + " | ".join(cells) + " |")
        if ll and isinstance(M.get("lossless"), dict):
            L += ["", "無損（實際係數＋基底位元組，zstd:3；≥1.05× → 值得疊）：" + "；".join(
                f"{s} {e['ratio'][LL_HEAD]}×（{'值得' if e['worth_stacking'] else '不值得'}）" for s, e in sorted(M["lossless"].items()))]
        else:
            L += ["", "無損：NOT_MEASURED（38fy 還沒跑或 dump 不在）"]
        L += ["", "底（xKV＋INT4、W＝4、1:1.5）沿倍數的 TF 品質（邊際代價＝每多壓一倍掉多少 Δgold_lp）：" + "；".join(
            f"{x['multiple']}× {x['d_gold_lp']['mean']:+.3f}" + (f"（{x['slope_per_doubling']:+.3f}／倍）" if "slope_per_doubling" in x else "")
            for x in M["x4a"]["base_curve"])]
        sv = M["gen"].get("32K", {}).get("stack_vs_rank_only", [])
        L += ["", "### 疊加優於單降秩（生成，±5%）", ""] + ([f"- {x['stacked']} vs {x['rank_only']}：{_ci(x['diff_stacked_minus_rank_only'])}"
                                                      f" → {'成立' if x['holds'] else '不成立'}" if "stacked" in x else
                                                      f"- {x['rank_only']}：{x['status']}" for x in sv] or ["- 沒有資料"])
        for ctx in ("128K", "256K"):
            v = M["gen"].get(ctx)
            if not v:
                L += ["", f"### {ctx}：沒有資料（沒排、沒跑完或寫明不做）"]
                continue
            L += ["", f"### {ctx}（{v['n_items']} 題{'；' + v['status'] if 'status' in v else ''}）", "", "| 配置 | 倍數 | 對 E0 | ε（撐住？） | 對 INT4 |", "|:--|--:|--:|:--|--:|"]
            for s, x in sorted(v["holds"].items(), key=lambda kv: kv[1]["multiple"]):
                L.append(f"| {s} | {x['multiple']} | {_ci(x['diff'])} | {'撐住' if x['eps_pass'] else '掉'} | {_ci(x['diff_vs_int4']) if x['diff_vs_int4'] else '—'} |")
        if "x4d" in M:
            L += ["", f"### 多輪（C1a 兩輪題組；{M['x4d']['2']['n_items']} 題）", "", "| 配置 | 倍數 | 第 1 輪 對 E0 | 第 2 輪 對 E0 | 第 2 輪判定 | 第 2 輪 對 drop 對照 | 第 2 輪 對 INT4 |",
                  "|:--|--:|--:|--:|:--|--:|--:|"]
            t1, t2 = M["x4d"]["1"]["configs"], M["x4d"]["2"]["configs"]
            for s in sorted(t2, key=lambda s: t2[s]["multiple"]):
                L.append(f"| {s} | {t2[s]['multiple']:.1f} | {_ci(t1[s]['diff']) if s in t1 else '—'} | {_ci(t2[s]['diff'])} | "
                         f"{M['x4d']['verdict_turn2'][s]} | {_ci(M['x4d']['2']['vs_drop'].get(s))} | "
                         f"{_ci(t2[s]['diff_vs_anchor']) if t2[s].get('diff_vs_anchor') else '—'} |")
        L += ["", f"Pareto：`pareto_x4a_{m}.png`（TF）、`pareto_x4b_{m}.png`（生成，32K／128K／256K）", ""]
    L += ["## 事先寫死的判定與和計畫不同的地方", ""] + [f"- **{k}**：{v}" for k, v in X4_TEXT.items()]
    open(os.path.join(a.out_dir, "tables_x4.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:70]))


# ============================================================================ gpu-selftest（GPU）
def cmd_gpu_selftest(a):
    """38e／38f／38g／38h 開頭的冒煙：(1) 每個 W 的小切片（2 個窗口、block 1–4）：共用 SVD 的重建誤差對 numpy 精確 SVD 在 [0.999, 1.05]
    （量化的配置 [0.98, 1.05]），位元組＝預測、xl+drop 的索引＝numpy；另記和 m16_xstack 每配置 SVD 的誤差比；(2) 兩輪生成：E0 的
    run_turns＝m10_quality.run_turns（逐 token）、第 1 輪分數＝Item.gen()；(3) 整個 32K 區：X4a 格點要的每個 (K／V, W) 的 SVD 時間、
    最大秩的配置位元組＝預測、有限值；(4) dump 的 part 位元組＝predict_bytes。"""
    import numpy as np
    import torch
    import m10_sweep as SW
    import m10_tasks as T
    from m10_quality import Answerer, ItemState
    import m10_quality as Q
    res, ok = {"checks": {}, "model": a.model, "run_id": os.environ.get("RUN_ID", "adhoc"), "ts": XS.ts()}, True

    def check(name, cond, **kw):
        nonlocal ok
        ok &= bool(cond)
        res["checks"][name] = {"pass": bool(cond), **kw}
        print(("PASS " if cond else "FAIL ") + name, json.dumps(kw, default=str)[:400], flush=True)
    info = MODEL_INFO[a.model]
    SW.N_BLOCKS = 64
    ans = Answerer(a.model)
    km = ans.km
    XS.check_info(km, info)
    P = json.load(open(a.params_file))["params"][a.model]
    it = T.make_item("T2", "front", "back", X4A_SEED0, ans.tok, P.get("T2"))
    st = SW.Item(ans, it)
    n = st.ctx_len
    tc0 = XS.TCtx(km, n)
    c_np = tc0.cos[0, :, :km.D // 2].double().cpu().numpy()
    s_np = tc0.sin[0, :, :km.D // 2].double().cpu().numpy()
    # 1. 小切片（W＝2、4；W 大已從格點縮掉）
    for W in (2, 4):
        Ls = 2 * W
        Xs = km.kv[:Ls, :, :, CHUNK:5 * CHUNK].clone()
        pos = torch.arange(CHUNK, 5 * CHUNK, device=km.device)
        Xn = Xs.float().cpu().numpy()
        posn = np.arange(CHUNK, 5 * CHUNK)
        info_s = dict(info, L=Ls)
        orders = {"KEYDIFF": [2, 0, 3, 1]}
        tcs = XS.TCtx(km, n, orders=orders, deep=list(range(Ls // 2, Ls)))
        ncx = XS.NCtx(c_np, s_np, orders=orders, deep=tcs.deep)
        specs = [xl_spec(8, 12, W, "bf16"), xl_spec(8, 12, W, "int4"), xl_spec(6, 9, W, "int4", ("KEYDIFF", 0.5)), xl_spec(5, 8, W, "k4v2")]
        cfgs = [dict(spec=s, stages=validate(parse_spec(s), info_s, 4)) for s in specs]
        store = SvdStore(cfgs, km.H)
        for c in cfgs:
            with torch.inference_mode():
                td, tidx, tby, _ = x4_pipeline(c["stages"], Xs, pos, tcs, info_s, store)
                tp, _, tpb, _ = XS.t_pipeline(c["stages"], Xs, pos, tcs, info_s)
            nd_, nidx, nby, _ = XS.np_pipeline(c["stages"], Xn, posn, ncx, info_s)
            ref = Xn if nidx is None else Xn[:, :, :, nidx]
            et = float(np.linalg.norm(td.float().cpu().numpy() - ref) / np.linalg.norm(ref))
            en = float(np.linalg.norm(nd_ - ref) / np.linalg.norm(ref))
            ep = float(np.linalg.norm(tp.float().cpu().numpy() - ref) / np.linalg.norm(ref))
            same_idx = (tidx is None and nidx is None) or (tidx is not None and nidx is not None and np.array_equal(tidx.cpu().numpy(), nidx))
            lo = 0.999 if spec_meta(c["spec"])["bits"] == "bf16" else 0.98
            check(f"shared_svd_vs_exact[W{W}|{c['spec']}]", lo * en <= et <= 1.05 * en and same_idx and tby == nby == tpb ==
                  predict_bytes(c["stages"], info_s, 4), err_shared=et, err_exact=en, err_per_config_svd=ep,
                  shared_over_per_config=et / ep if ep else None, bytes=tby)
    # 2. 兩輪
    e2 = run_turns(ans, it, n, 2)
    g1 = st.gen()
    ist = ItemState(ans, it)
    ref = Q.run_turns(ist, "bf16", [frozenset()] * 2)
    same = all(e2[t]["gen"] == ref[t]["gen"] for t in range(2))
    check("two_turn_eq_m10_quality_run_turns", same and e2[0]["score"] == g1[0] and e2[0]["stop"] == g1[1],
          scores=[x["score"] for x in e2], item_gen=g1, gen_len=[len(x["gen"]) for x in e2])
    del ist
    st = SW.Item(ans, it)                                        # 重新 prefill（ItemState 也 prefill 過；確保 bf16 副本＝緩衝）
    # 3. 整個 32K 區：格點要的 SVD
    grid = load_x4_configs(a.combos, a.model)
    store = SvdStore(grid, km.H)
    lo_, hi_, nb = _region(km, n)
    nreg = hi_ - lo_ + 1
    pos = torch.arange(lo_ * CHUNK, (hi_ + 1) * CHUNK, device=km.device)
    X = km.kv[:, :, :, lo_ * CHUNK:(hi_ + 1) * CHUNK]
    m10o = _orders(km, st, it, grid, n)
    tcf = XS.TCtx(km, n, orders={m: [b - lo_ for b in o if lo_ <= b <= hi_] for m, o in m10o.items()}, deep=XS.deep_layers(km.L))
    big = {}
    for c in grid:
        mt = spec_meta(c["spec"])
        if mt["xl"]:
            k = mt["W"]
            if k not in big or mt["rk"] + mt["rv"] > big[k]["rk"] + big[k]["rv"]:
                big[k] = dict(c, **mt)
    for W, c in sorted(big.items()):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.inference_mode():
            td, _, tby, _ = x4_pipeline(c["stages"], X, pos, tcf, info, store)
            rel = XS.t_relerr(X, td, None) if "drop" not in c["spec"] else {}
        check(f"full_region_maxrank[W{W}|{c['spec']}]", tby == predict_bytes(c["stages"], info, nreg) and bool(torch.isfinite(td.float()).all()),
              t_total_s=round(time.time() - t0, 2), t_svd=dict(store.t), peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2), **rel)
        del td
    # 4. dump 的位元組
    for c in [x for x in grid if "alloc:int4" in x["roles"]][:1] + [x for x in grid if x["drop_frac"]][:1]:
        with torch.inference_mode():
            parts = x4_encode_parts(c["stages"], X, pos, tcf, info, store)
        pred = predict_bytes(c["stages"], info, nreg)
        tot = sum(v.nbytes for v, _ in parts.values())
        check(f"dump_bytes_eq_predict[{c['spec']}]", tot == sum(v for k, v in pred.items() if not k.startswith("shared.")),
              parts={k: v.nbytes for k, (v, _) in parts.items()})
    res["all_pass"] = ok
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False, default=str)
    print("ALL PASS" if ok else "SOME FAILED", f"（{len(res['checks'])} 項）", flush=True)
    sys.exit(0 if ok else 1)


# ============================================================================ --self-test（CPU；不准 import torch）
def self_test() -> bool:
    import gzip
    import random
    import tempfile
    np = XS._np()
    res, ok = {}, True

    def check(name, cond, **kw):
        nonlocal ok
        ok &= bool(cond)
        res[name] = {"pass": bool(cond), **kw}
        print(("PASS " if cond else "FAIL ") + name, json.dumps(kw, default=str, ensure_ascii=False)[:300], flush=True)
    # 1. 格點：JSON＝程式重算；每個 T 6–10 種分法、都在 ±10% 內；順序、W、丟的掃描、細格點、對照
    g = json.load(open(COMBOS_X4)) if os.path.exists(COMBOS_X4) else None
    check("grid_json_eq_regenerated", g is not None and g["models"] == json.loads(json.dumps({m: build_grid(m) for m in MODEL_INFO})),
          path=COMBOS_X4)
    for m in MODEL_INFO:
        G = build_grid(m)
        cs = G["configs"]
        per_T = {T: [c for c in cs if T in c["targets"] and any(r.startswith("alloc:") for r in c["roles"])] for T in TARGETS}
        okb = all(in_band(c["multiple"], T) for T in TARGETS for c in cs if T in c["targets"])
        okv = True
        for c in cs:
            try:
                validate(parse_spec(c["spec"]), MODEL_INFO[m], NREG)
            except SpecError:
                okv = False
            okv &= abs(c["ratio"] - predict_ratio(c["spec"], m)) < 1e-12
        order = all(parse_spec(c["spec"])[1]["op"] == "drop" for c in cs if c["xl"] and c["drop_frac"])
        wok = all(c["W"] in (2, 4) for c in cs if c["xl"])
        ds = {T: sorted((c["drop_method"], c["drop_frac"]) for c in cs if "drop_study" in c["roles"] and T in c["targets"]) for T in DROP_STUDY_T}
        check(f"grid_shape[{m}]", all(6 <= len(v) <= MAX_PER_T for v in per_T.values()) and okb and okv and order and wok
              and all(len(v) == 6 for v in ds.values()) and sum("fine_grid" in c["roles"] for c in cs) == len(FINE_PAIRS)
              and all(any(c["spec"] == canonical(parse_spec(s)) for c in cs) for s in CONTROLS) and 60 <= len(cs) <= 100,
              n=len(cs), per_T={T: len(v) for T, v in per_T.items()}, skipped=[(s["T"], s["alloc"]) for s in G["skipped"]])
        need = SvdStore([dict(stages=parse_spec(c["spec"])) for c in cs], MODEL_INFO[m]["H"]).need
        mx = {}
        for c in cs:
            if c["xl"]:
                mx[(0, c["W"])] = max(mx.get((0, c["W"]), 0), c["rk"])
                mx[(1, c["W"])] = max(mx.get((1, c["W"]), 0), c["rv"])
        check(f"svd_store_need[{m}]", need == mx, need={f"{'kv'[h]}_w{w}": r for (h, w), r in need.items()})
    # 2. 題目不重疊
    T4 = ["T2", "T3", "T4", "T5"]
    keys = {k: {(t, s) for t, _, _, s in XS.plan_interleaved_cpu(ts_, n_, s0)} for k, ts_, n_, s0 in
            (("x1", T4, 24, XS.X1_SEED0), ("calib", T4, 2, XS.CALIB_SEED0), ("x2", T4, 24, XS.X2_SEED0), ("x2b", T4, XS.X2B_N, XS.X2B_SEED0),
             ("x4a", T4, X4A_N, X4A_SEED0), ("x4b", T4, X4B_N, X4B_SEED0), ("c1a", ["T2", "T3", "T4"], 45, 70000), ("c1a_t5", ["T5"], 15, 70000))}
    x4d = {(t, s) for t, _, _, s in XS.plan_interleaved_cpu(list(X4D_TASKS), X4D_N, X4D_SEED0)}
    disj = all(not (keys[a_] & keys[b_]) for a_ in ("x4a", "x4b") for b_ in keys if b_ != a_)
    check("seeds_disjoint_x4", disj and x4d <= keys["c1a"] and len(keys["x4a"]) == 96 and len(keys["x4b"]) == 192 and len(x4d) == 96,
          sizes={k: len(v) for k, v in keys.items()})
    # 3. numpy：xl+drop、W 2／4／8 的位元組＝預測、形狀；精確 SVD 的前 r 欄＝秩 r 的精確 SVD（共用 SVD 的數學前提）
    info_s = dict(L=8, H=2, Hq=4, D=32)
    Ch, nreg = 64, 4
    Tr = Ch * nreg
    rng = np.random.default_rng(4)
    X = XS.np_bf16((rng.normal(size=(8, 2, 2, Tr, 6)) @ rng.normal(size=(6, 32)) + 0.3 * rng.normal(size=(8, 2, 2, Tr, 32))).astype(np.float32))
    inv = 1.0 / (10000 ** (np.arange(0, 32, 2) / 32))
    ang = np.arange(Ch + Tr)[:, None] * inv[None, :]
    cf, sf = XS.np_bf16(np.cos(ang)).astype(np.float64), XS.np_bf16(np.sin(ang)).astype(np.float64)
    posn = np.arange(Ch, Ch + Tr)
    ncx = XS.NCtx(cf, sf, orders={"KEYDIFF": [2, 0, 3, 1]}, deep=[4, 5, 6, 7], chunk=Ch)
    for W in (2, 4, 8):
        for sp in (f"xl:k2:v3{':w' + str(W) if W != 4 else ''}+q:int4_kivi", f"xl:k2:v3{':w' + str(W) if W != 4 else ''}+drop:KEYDIFF:0.5+q:int4_kivi"):
            stg = validate(parse_spec(sp), info_s, nreg)
            dd, idx, by, ex = XS.np_pipeline(stg, X, posn, ncx, info_s)
            check(f"np_x4_bytes_shape[{sp}]", by == predict_bytes(stg, info_s, nreg, Ch) and dd.shape[3] == (Tr if idx is None else len(idx))
                  and np.isfinite(dd).all(), bytes=by)
    Z = rng.normal(size=(200, 40)) @ np.diag(np.linspace(3, 0.1, 40))
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    r = 7
    check("truncated_svd_is_rank_r_optimal", abs(np.linalg.norm(Z - (U[:, :r] * S[:r]) @ Vt[:r]) - np.sqrt((S[r:] ** 2).sum())) < 1e-9)
    # 4. 挑選（合成）
    def tfc(spec, mult, q, n=96):
        mt = spec_meta(spec)
        per = {f"i{k}": q + (0.01 if k % 2 else -0.01) for k in range(n)}
        return dict(spec=spec, n=n, ratio=1 / mult, multiple=mult, **mt, roles=[], deployable=1, d_gold_lp=boot2(per.values(), n_boot=200),
                    d_first_ok=dict(mean=0, lo=0, hi=0), per_item=per)
    tab = {c["spec"]: c for c in [tfc("xl:k29:v44+q:int4_kivi", 8.0, -0.1), tfc("xl:k11:v17", 8.1, -0.3), tfc("xl:k20:v30+q:int8_token", 8.1, -0.05),
                                  tfc("xl:k17:v26:w2+q:int4_kivi", 16.2, -0.4), tfc("xl:k6:v9", 16.0, -0.9), tfc("xl:k14:v21+q:int4_kivi", 16.6, -0.5),
                                  tfc("xl:k8:v12+q:int4_kivi", 29.1, -1.5), tfc("q:int4_kivi", 3.2, -0.01)]}
    sb = {e["spec"]: e for e in select_b(tab, "llama")}
    check("select_b_rule", "band_best:8" in sb["xl:k20:v30+q:int8_token"]["roles"] and "band_best:16" in sb["xl:k17:v26:w2+q:int4_kivi"]["roles"]
          and "band_best:32" in sb["xl:k8:v12+q:int4_kivi"]["roles"] and "rank_only_ref:8" in sb["xl:k11:v17"]["roles"]
          and "rank_only_ref:16" in sb["xl:k6:v9"]["roles"] and "anchor_int4" in sb["q:int4_kivi"]["roles"]
          and "xl:k29:v44+q:int4_kivi" not in sb and "xl:k14:v21+q:int4_kivi" not in sb, picked={k: v["roles"] for k, v in sb.items()})

    def gc(spec, mult, acc, lo, n=96):
        mt = spec_meta(spec)
        per = {f"i{k}": float(k % 100 < acc * 100) for k in range(n)}
        d = dict(mean=acc - 0.8, lo=lo, hi=acc - 0.8 + 0.03, p_le0=0.5, n=n)
        return dict(spec=spec, n=n, ratio=1 / mult, multiple=mult, **mt, roles=[], deployable=1, acc=acc, acc_e0=0.8, diff=d, per_item=per,
                    eps_pass=lo >= -EPS, **{"mlperf0.99": acc >= 0.792, "mlperf0.999": acc >= 0.7992})
    gtab = {c["spec"]: c for c in [gc("xl:k20:v30+q:int8_token", 8.1, 0.80, -0.01), gc("xl:k13:v20+q:int8_token", 12.3, 0.80, -0.015),
                                   gc("xl:k17:v26:w2+q:int4_kivi", 16.2, 0.79, -0.019), gc("xl:k12:v18+q:int4_kivi", 19.4, 0.76, -0.06),
                                   gc("xl:k6:v9", 16.0, 0.5, -0.4), gc("q:int4_kivi", 3.2, 0.8, -0.01)]}
    sc = select_cd(gtab, 96)
    check("select_cd_rule", sc["tstar"] == 16 and sc["frontier"] == ["xl:k17:v26:w2+q:int4_kivi", "xl:k12:v18+q:int4_kivi", "xl:k20:v30+q:int8_token"]
          and sc["x4c"] == sc["frontier"] + ["q:int4_kivi"] and sc["x4d"] == sc["frontier"][:2] + ["q:int4_kivi", "drop:KEYDIFF:0.5+q:int4_kivi"]
          and select_cd(gtab, 20)["go"] is False, sel=sc["frontier"])
    lm = limits(gtab, 96)
    check("limits_rule", lm["eps"]["spec"] == "xl:k17:v26:w2+q:int4_kivi" and limits(gtab, 30)["status"].startswith("INCOMPLETE"), lim=lm["eps"])
    sv = stack_vs_rank_only(gtab)
    check("stack_vs_rank_only_rule", sv and sv[0]["rank_only"] == "xl:k6:v9" and sv[0]["stacked"] == "xl:k17:v26:w2+q:int4_kivi"
          and sv[0]["holds"] is True, sv=sv)
    qa = qa_axes(tab, lambda c: (qual_tf(c), -c["ratio"]), "per_item")
    check("qa_axes_rule", qa["quant"]["by_T"]["8"]["verdict"].startswith("值得") and qa["quant"]["by_T"]["8"]["best_with"] == "xl:k20:v30+q:int8_token"
          and qa["int4_vs_int8"]["by_T"]["8"]["verdict"].startswith("不值得") and qa["drop"]["by_T"]["8"]["status"].startswith("NOT_COMPARABLE"),
          q8=qa["quant"]["by_T"]["8"], i48=qa["int4_vs_int8"]["by_T"]["8"].get("verdict"))
    # 5. 端到端（合成逐列）：select-b → select-cd → report；無損；不准寫第一版／v2 的目錄
    with tempfile.TemporaryDirectory() as d:
        rnd = random.Random(3)
        pa, pg, pt = (os.path.join(d, f"{k}_s0.csv.gz") for k in ("tf", "gen", "turns"))
        with gzip.open(pa, "wt", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS_X4A)
            w.writeheader()
            for m in MODEL_INFO:
                cs = build_grid(m)["configs"]
                for k in range(50):
                    e0 = -2 + rnd.random()
                    base = dict(run_id="SYNA", ts="t", shard="0/1", model=m, item_id=f"T2_s{X4A_SEED0 + k}", task="T2", band="front", seed=k,
                                gold_lp_e0=e0, first_ok_e0=1, n_rows_item=len(cs) + 1, own_gpu_intruders=0, host_level="QUIET")
                    w.writerow(dict(base, spec="E0", roles="ref", ratio_region=1.0, multiple=1.0, deployable=1, kl=0, gold_lp=e0, top1=1,
                                    first_ok=1, relerr_k=0, relerr_v=0))
                    for c in cs:
                        eff = -0.02 * c["multiple"] * (0.5 if c["bits"] == "int4" else 1.0) + rnd.gauss(0, 0.05)
                        w.writerow(dict(base, spec=c["spec"], roles="|".join(c["roles"]), ratio_region=c["ratio"], multiple=c["multiple"],
                                        deployable=c["deployable"], kl=-eff, gold_lp=e0 + eff, top1=0.9, first_ok=int(eff > -0.6),
                                        relerr_k=0.1, relerr_v=0.2, **{k_: c[k_] for k_ in ("W", "rk", "rv", "bits", "drop_method", "drop_frac")}))
        sb_path = os.path.join(d, "x4b_selection.json")
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            cmd_select_b(SimpleNamespace(raw=[pa], out=sb_path, seed0=X4B_SEED0, n=X4B_N))
            cmd_select_b(SimpleNamespace(raw=[pa], out=sb_path, seed0=X4B_SEED0, n=X4B_N))      # 內容相同 → 不重寫
        sel = json.load(open(sb_path))
        n_sel = {m: len(v["configs"]) for m, v in sel["models"].items()}
        check("e2e_select_b", sel["go"] and all(6 <= v <= 9 for v in n_sel.values()), n=n_sel)
        with gzip.open(pg, "wt", newline="") as f, gzip.open(pt, "wt", newline="") as f2:
            w, w2 = csv.DictWriter(f, fieldnames=FIELDS_X4G), csv.DictWriter(f2, fieldnames=FIELDS_X4G)
            w.writeheader()
            w2.writeheader()
            for m in MODEL_INFO:
                specs = [e["spec"] for e in sel["models"][m]["configs"]]
                for ctx, nb_, nn, ww, turns in (("32K", 64, 60, w, 1), ("128K", 252, 50, w, 1), ("32K", 64, 50, w2, 2)):
                    sp = specs if ctx == "32K" and turns == 1 else specs[:3] + ["drop:KEYDIFF:0.5+q:int4_kivi"]
                    for k in range(nn):
                        rows = []
                        for t in range(1, turns + 1):
                            s0 = int(rnd.random() < 0.8)
                            rows.append(dict(turn=t, spec="E0", gen_score=s0, ratio_region=1.0, multiple=1.0, deployable=1))
                            for s in sp:
                                r_ = predict_ratio(s, m, nb_ - 1)
                                mt = spec_meta(s)
                                rows.append(dict(turn=t, spec=s, gen_score=int(s0 and rnd.random() > 0.01 * (1 / r_) ** 0.8), ratio_region=r_,
                                                 multiple=1 / r_, deployable=1, roles="x", **mt))
                        for r_ in rows:
                            ww.writerow(dict(r_, run_id=f"SYN-{ctx}-{turns}", ts="t", model=m, ctx=ctx, n_blocks=nb_, item_id=f"T2_g_s{k}",
                                             n_rows_item=len(rows)))
        pc, pd_ = os.path.join(d, "x4c.json"), os.path.join(d, "x4d.json")
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            try:
                cmd_select_cd(SimpleNamespace(raw=[pg], out_c=pc, out_d=pd_))
                okcd = True
            except SystemExit as ex:
                okcd = str(ex)
        scd = json.load(open(pc)) if os.path.exists(pc) else {}
        check("e2e_select_cd", okcd is True and all(len(v["configs"]) == 4 for v in scd.get("models", {}).values())
              and all(len(v["configs"]) == 4 for v in json.load(open(pd_))["models"].values()),
              n={m: len(v["configs"]) for m, v in scd.get("models", {}).items()}, err=okcd)
        dump = os.path.join(d, "dump", "llama", "T2_x", "cfg")
        os.makedirs(dump)
        q = rng.integers(0, 16, size=4096, dtype=np.uint8)
        b16 = XS.np_bf16(rng.normal(size=2048).astype(np.float32)).view(np.uint32) >> 16
        b16 = b16.astype(np.uint16)
        q.tofile(os.path.join(dump, "k.coef.q.bin"))
        b16.tofile(os.path.join(dump, "k.basis.bin"))
        json.dump(dict(run_id="SYND", spec="xl:k8:v12+q:int4_kivi", item_id="T2_x", model="llama",
                       parts={"k.coef.q": dict(kind="raw", bytes=q.nbytes), "k.basis": dict(kind="bf16", bytes=b16.nbytes)}),
                  open(os.path.join(dump, "meta.json"), "w"))
        llo = os.path.join(d, "ll")
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            cmd_lossless(SimpleNamespace(dumps=[os.path.join(d, "dump")], out_dir=llo, threads=[1, 2], reps=1))
        L = json.load(open(os.path.join(llo, "lossless_x4.json")))
        e = L["summary"]["llama"]["xl:k8:v12+q:int4_kivi"]
        check("e2e_lossless", e["stored"] == q.nbytes + b16.nbytes and e["ratio"][LL_HEAD] > 1.0 and "zstd:3|t2" in e["decomp_MBps_median"],
              ratio=e["ratio"], speed=e["decomp_MBps_median"])
        out = os.path.join(d, "x4")
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            cmd_report(SimpleNamespace(raw_tf=[pa], raw_gen=[pg], raw_turns=[pt], combos=COMBOS_X4, lossless=os.path.join(llo, "lossless_x4.json"),
                                       out_dir=out))
        rep = json.load(open(os.path.join(out, "x4_report.json")))
        Mx = rep["models"]["llama"]
        files = all(os.path.exists(os.path.join(out, f)) for f in ("tables_x4.md", "x4_best_allocation.csv", "x4a_configs_llama.csv"))
        check("e2e_report", files and "32K" in Mx["gen"] and "128K" in Mx["gen"] and "verdict_turn2" in Mx["x4d"]
              and set(Mx["x4a"]["qa"]) == set(QA_AXES) and len(list(csv.DictReader(open(os.path.join(out, "x4_best_allocation.csv"))))) == 12,
              limits=Mx["gen"]["32K"]["limits"].get("eps"), png=Mx["x4a"]["pareto_png"])
        try:
            existed = os.path.isdir(XS.OUT_DIR_V2)
            cmd_report(SimpleNamespace(raw_tf=[], raw_gen=[], raw_turns=[], combos=COMBOS_X4, lossless=None, out_dir=XS.OUT_DIR_V2))
            g2 = False
        except SystemExit:
            g2 = existed or not os.path.isdir(XS.OUT_DIR_V2)          # 擋下來，而且沒有順手建目錄
        check("report_refuses_v1_v2_dirs", g2)
    # 6. job 用的指令解析得了
    ap = build_parser()
    cmds = [["tf", "--model", "llama", "--n", "24", "--seed0", "64000", "--shard", "0/8", "--budget-min", "90", "--out", "/x/tf_llama_s0.csv.gz"],
            ["gen", "--model", "qwen", "--selection", "/x/s.json", "--n-blocks", "252", "--n", "12", "--seed0", "40000", "--shard", "0/2",
             "--budget-min", "90", "--out", "/x/g.csv.gz"],
            ["gen", "--model", "qwen", "--selection", "/x/s.json", "--turns", "2", "--tasks", "T2", "T3", "T4", "--n", "32", "--seed0", "70000",
             "--out", "/x/t.csv.gz"], ["gpu-selftest", "--model", "qwen", "--out", "/x/g.json"],
            ["dump", "--model", "llama", "--selection", "/x/s.json", "--out-dir", "/x/d"], ["lossless", "--dumps", "/x/d", "--out-dir", "/x/o"],
            ["select-b", "--raw", "/x/*.gz", "--out", "/x/s.json"], ["select-cd", "--raw", "/x/*.gz", "--out-c", "/x/c.json", "--out-d", "/x/d.json"],
            ["report", "--raw-tf", "/x/a", "--raw-gen", "/x/b", "--raw-turns", "/x/c"], ["grid", "--out", "/x/g.json"]]
    pa_ = [ap.parse_args(c) for c in cmds]
    check("cli_parse", all(COMMANDS[p.cmd] in globals() for p in pa_) and pa_[1].n_blocks == 252 and pa_[2].turns == 2)
    bad = [x for x in ("torch", "vllm", "transformers") if x in sys.modules]
    check("no_torch_imported", not bad, imported=bad)
    print(f"{'ALL PASS' if ok else 'SOME FAILED'}（{len(res)} 項）", flush=True)
    return ok


# ============================================================================ main
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("grid")
    g.add_argument("--out", default=COMBOS_X4)
    g.add_argument("--force", action="store_true")
    t = sp.add_parser("tf")
    t.add_argument("--model", choices=list(MODEL_INFO), required=True)
    t.add_argument("--combos", default=COMBOS_X4)
    t.add_argument("--tasks", nargs="+", default=["T2", "T3", "T4", "T5"])
    t.add_argument("--n", type=int, default=X4A_N)
    t.add_argument("--seed0", type=int, default=X4A_SEED0)
    t.add_argument("--params-file", default=XS.PARAMS)
    t.add_argument("--budget-min", type=float, default=0)
    t.add_argument("--shard", default="0/1")
    t.add_argument("--out", required=True)
    t.add_argument("--smoke", action="store_true", help="1 題（T2）、冒煙用的 6 個配置")
    sb = sp.add_parser("select-b")
    sb.add_argument("--raw", nargs="+", required=True)
    sb.add_argument("--out", default=os.path.join(OUT, "x4b_selection.json"))
    sb.add_argument("--seed0", type=int, default=X4B_SEED0)
    sb.add_argument("--n", type=int, default=X4B_N)
    gn = sp.add_parser("gen")
    gn.add_argument("--model", choices=list(MODEL_INFO), required=True)
    gn.add_argument("--selection", required=True)
    gn.add_argument("--n-blocks", type=int, default=64, help="64＝32K、252＝128K（44c）、504＝256K（Qwen noDCA）")
    gn.add_argument("--turns", type=int, default=1, choices=[1, 2])
    gn.add_argument("--tasks", nargs="+", default=["T2", "T3", "T4", "T5"])
    gn.add_argument("--n", type=int, default=X4B_N)
    gn.add_argument("--seed0", type=int, default=X4B_SEED0)
    gn.add_argument("--params-file", default=XS.PARAMS)
    gn.add_argument("--budget-min", type=float, default=0)
    gn.add_argument("--shard", default="0/1")
    gn.add_argument("--out", required=True)
    gn.add_argument("--smoke", action="store_true", help="1 題（第一個任務）")
    sc = sp.add_parser("select-cd")
    sc.add_argument("--raw", nargs="+", required=True)
    sc.add_argument("--out-c", default=os.path.join(OUT, "x4c_selection.json"))
    sc.add_argument("--out-d", default=os.path.join(OUT, "x4d_selection.json"))
    du = sp.add_parser("dump")
    du.add_argument("--model", choices=list(MODEL_INFO), required=True)
    du.add_argument("--selection", required=True)
    du.add_argument("--tasks", nargs="+", default=["T2", "T3", "T4", "T5"])
    du.add_argument("--n", type=int, default=DUMP_N)
    du.add_argument("--seed0", type=int, default=X4A_SEED0)
    du.add_argument("--params-file", default=XS.PARAMS)
    du.add_argument("--out-dir", required=True)
    ll = sp.add_parser("lossless")
    ll.add_argument("--dumps", nargs="+", required=True)
    ll.add_argument("--out-dir", default=OUT)
    ll.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    ll.add_argument("--reps", type=int, default=3)
    rp = sp.add_parser("report")
    rp.add_argument("--raw-tf", nargs="+", required=True)
    rp.add_argument("--raw-gen", nargs="+", required=True)
    rp.add_argument("--raw-turns", nargs="+", required=True)
    rp.add_argument("--combos", default=COMBOS_X4)
    rp.add_argument("--lossless", default=os.path.join(OUT, "lossless_x4.json"))
    rp.add_argument("--out-dir", default=OUT)
    gs = sp.add_parser("gpu-selftest")
    gs.add_argument("--model", choices=list(MODEL_INFO), required=True)
    gs.add_argument("--combos", default=COMBOS_X4)
    gs.add_argument("--params-file", default=XS.PARAMS)
    gs.add_argument("--out", required=True)
    return ap


COMMANDS = {"grid": "cmd_grid", "tf": "cmd_tf", "select-b": "cmd_select_b", "gen": "cmd_gen", "select-cd": "cmd_select_cd",
            "dump": "cmd_dump", "lossless": "cmd_lossless", "report": "cmd_report", "gpu-selftest": "cmd_gpu_selftest"}


def main():
    if "--self-test" in sys.argv[1:]:
        sys.exit(0 if self_test() else 1)
    a = build_parser().parse_args()
    globals()[COMMANDS[a.cmd]](a)


if __name__ == "__main__":
    main()
