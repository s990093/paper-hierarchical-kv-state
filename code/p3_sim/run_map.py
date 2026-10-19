"""run_map.py — 地圖實驗的執行器（04_prereg_map.md v4 §5、§6、§9、§10；06_simulator.md §20）。

一個「工作」(job)＝一次放置模擬（一個切片 × 外層做法 × 記憶體 × 壓縮做法［× H］）：
  * session 引擎（E1–E5）：一次放置有多個時間設定（8 個主視角＋敏感度的 4 個），每個時間設定一列。
  * block 引擎（E6 與它的 E2 對照）：一個時間設定一個工作；session 依 sha1(session_id) mod k 分到 k 張 GPU，
    每張各跑一次、各有自己的 CPU 記憶體，位元組加總（k 見 results/p3_sim/kgpu_kmin_all.csv）。
每個工作寫兩個檔到 --out：jobs/<切片>/<id>.json（所有列＋meta）、sess/<切片>/<id>.npz（主視角的每個 session 的帳，給 bootstrap）。
CSV：collect 把所有 json 合成 map_cells.csv（一列＝一個格子＝一個放置×一個時間設定）；這個檔很大，放 /ssd8，不進 git。

用法：
  python code/p3_sim/run_map.py --stage A1 --dry-run                      # 只數格子與估計時間（真實資料也可以）
  TIARA_NO_GPU=1 bash code/run3090.sh p3sim-mapsyn python code/p3_sim/run_map.py --stage A1 --slices synth/small-seed1 --out <dir>
  ...                                  python code/p3_sim/run_map.py --stage A1 --frozen <commit> --out <dir>   # 真實資料：必須給凍結 commit
凍結（第二輪審查 #5；lockfile.py）：真實資料的工作一律要 --frozen <commit>，而且：04 在那個 commit 不含「未凍結」、§9 的鎖定表列的輸入檔
sha256 全部相符、程式 commit 等於 HEAD 或 code/p3_sim/ 無變動的後代、已追蹤檔案無未提交修改、RUN_ID 明確（不是 adhoc）、格式表＝official。
合成資料不需要。
不設 CUDA_VISIBLE_DEVICES（run3090 的 TIARA_NO_GPU=1 已處理）。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import ledger  # noqa: E402
import lockfile  # noqa: E402
import sim  # noqa: E402
import trace as TR  # noqa: E402
from formats import NotMeasured, load_costs  # noqa: E402

GiB = 1 << 30
PREREG = "docs/phase3_20261017/04_prereg_map.md"
RETURN_GAPS = os.path.join(REPO, "results", "p3_sim", "return_gaps.csv")
KALL = os.path.join(REPO, "results", "p3_sim", "kgpu_kmin_all.csv")
TIMING_STDOUT = "/ssd8/hungwei/paper-hkv/runs/20261018-102610-p3sim-timing3/stdout.log"
DEFAULT_OUT = os.environ.get("P3_MAP_OUT", "/ssd8/hungwei/paper-hkv/p3_map")   # H200：code/r_h200_p3.sh 指到 compute node 的本地 /tmp
NOT = "NOT_MEASURED"

# ------------------------------------------------------------------ 寫死的參數（04 v4 §5、§9、§10）
A1_MEM = (63.0, 170.0, 435.0)
A2_MEM = (32.0, 64.0, 128.0, 256.0, 512.0)
INT2_ON_READ = "recompute_unverified"       # 04 §3.3：讀到沒驗證的就重算
UNVERIFIED_LEN = "request"                  # 04 §3.3：讀取長度＝請求的輸入長度（整個上下文）
# P3_MAP_PLATFORM（2026-10-19，H200 移植）：沒設＝mi300x（原本）。h200＝主視角換 costs_h200.json 與 H200 的正式格式表（official_h200）；
# 慢層 profile 照舊（mi300x_local＝限速參數，H200 沿用）；敏感度沒有 3090 列（H200 是唯一平台），CPU 忙等 H200 自己的校準
MAP_PLAT = os.environ.get("P3_MAP_PLATFORM", "mi300x")
if MAP_PLAT not in ("mi300x", "h200"):
    raise SystemExit(f"P3_MAP_PLATFORM={MAP_PLAT!r}：只接受 mi300x、h200")
MAP_FMT = "official" if MAP_PLAT == "mi300x" else f"official_{MAP_PLAT}"     # 真實資料只准用這個格式表
MAIN_SPECS = [(MAP_PLAT, g, "idle", sh, "cake", "mi300x_local") for g in ("idle", "busy") for sh in (1, 2, 4, 8)]
PRIMARY_SPEC = MAIN_SPECS[0]                # 主視角：Llama、MI300X、GPU 閒、CPU 閒、SSD 1 張 GPU 專用
RECOMPUTE_CHARGES = ("none", "pass_size")   # 重算計價（第三輪審查 #1 選項 a）；真實資料的值只從凍結的 04 §9 讀（`重算計價：…`）
CPUBUSY_CSV = os.path.join(REPO, "results", "p3_calib_3090", "cpubusy_slowdown.csv")
CPUBUSY_CSV_H200 = os.path.join(REPO, "results", "p3_calib_h200", "cpubusy_slowdown.csv")   # code/r_h200_p3.sh --part cal


def load_cpu_busy(path=CPUBUSY_CSV):
    """CPU 忙的係數（04 §9；第二輪審查 #13）：從 3090 實測的 cpubusy_slowdown.csv 讀，不抄進程式。
    列＝kind=time、metric=load_only:t_restore；N=8＋place=same（低）、N=48＋place=other（高）的 slowdown_x。
    登記的值是 1.35 與 1.81，四捨五入到小數 2 位必須等於它們（不等＝資料變了，停下來）。回傳 (低, 高, run_id)。"""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["kind"] == "time" and r["metric"] == "load_only:t_restore"]

    def pick(n, place):
        m = [r for r in rows if int(r["N"]) == n and r["place"] == place]
        assert len(m) == 1, f"{path}: load_only:t_restore 的 N={n}、place={place} 應該剛好 1 列，實際 {len(m)}"
        return float(m[0]["slowdown_x"]), m[0]["run_id"]
    (lo, rid_lo), (hi, rid_hi) = pick(8, "same"), pick(48, "other")
    assert rid_lo == rid_hi, (rid_lo, rid_hi)
    assert round(lo, 2) == 1.35 and round(hi, 2) == 1.81, f"CPU 忙的係數 {lo}／{hi} 四捨五入後不是登記的 1.35／1.81"
    return round(lo, 2), round(hi, 2), rid_lo


def load_cpu_busy_h200(path=CPUBUSY_CSV_H200):
    """H200 的 CPU 忙係數：同一個定義（load_only:t_restore；N=8 同 NUMA＝低、N=48 另一個 NUMA＝高）、同一支 summarize_cpubusy 的輸出。
    04 登記的 1.35／1.81 是 3090 的值，不套用到 H200（不斷言）。還沒量（檔不在）→ (None, None, None)：敏感度沒有 CPU 忙的列（照實 NOT_MEASURED）。"""
    if not os.path.exists(path):
        return None, None, None
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["kind"] == "time" and r["metric"] == "load_only:t_restore"]
    pick = {(int(r["N"]), r["place"]): r for r in rows}
    lo, hi = pick[(8, "same")], pick[(48, "other")]
    assert lo["run_id"] == hi["run_id"], (lo["run_id"], hi["run_id"])
    return round(float(lo["slowdown_x"]), 2), round(float(hi["slowdown_x"]), 2), lo["run_id"]


if MAP_PLAT == "mi300x":
    CPU_BUSY_LO, CPU_BUSY_HI, CPU_BUSY_RUN = load_cpu_busy()
    SENS_SPECS = [("mi300x", "idle", CPU_BUSY_LO, 1, "cake", "mi300x_local"), ("mi300x", "idle", CPU_BUSY_HI, 1, "cake", "mi300x_local"),
                  ("3090", "idle", "idle", 1, "cake", "mi300x_local"), ("3090", "idle", "idle", 1, "cake", "ssd8_3090_meas")]
    SENS_GROUPS = [f"sens_cpu{CPU_BUSY_LO:.2f}", f"sens_cpu{CPU_BUSY_HI:.2f}", "sens_3090", "sens_3090_ssd8"]
else:
    CPU_BUSY_LO, CPU_BUSY_HI, CPU_BUSY_RUN = load_cpu_busy_h200()
    SENS_SPECS, SENS_GROUPS = [], []
    if CPU_BUSY_LO is not None:
        SENS_SPECS = [(MAP_PLAT, "idle", CPU_BUSY_LO, 1, "cake", "mi300x_local"), (MAP_PLAT, "idle", CPU_BUSY_HI, 1, "cake", "mi300x_local")]
        SENS_GROUPS = [f"sens_cpu{CPU_BUSY_LO:.2f}", f"sens_cpu{CPU_BUSY_HI:.2f}"]
SENS_LITE_SPECS = [MAIN_SPECS[0], MAIN_SPECS[4]]        # 敏感度的放置（Qwen、70B、疊加）：主視角＋GPU 忙
BLOCK_VIEWS = {"idle": (MAP_PLAT, "idle", "idle", 1, "cake", "mi300x_local"), "busy": (MAP_PLAT, "busy", "idle", 1, "cake", "mi300x_local")}
OUTERS_MAIN = [("E2", None), ("E3", None), ("E4", None), ("E5", 1800)]          # 主格的 session 引擎外層做法（E5＝30 分）
OUTERS_ALL = [("E1", None), ("E2", None), ("E3", None), ("E4", None), ("E5", 300), ("E5", 1800), ("E5", 3600)]
OUTERS_FILL = [("E1", None), ("E5", 300), ("E5", 3600)]                          # --fill：三個記憶體點上補 E1、E5 的另兩個 T
OVERLAYS = (2, 4)

# slice 名 → (source, variant, 判定群組, 來源群組, 階段)
SLICES = {
    "bailian/coder": ("bailian", "coder", "Bailian coder", "alibaba", "A"),
    "bailian/traceA": ("bailian", "traceA", "Bailian traceA", "alibaba", "A"),
    "wildchat/1m": ("wildchat", "1m", "WildChat", "wildchat", "A"),
    "mooncake/conv-rule-strict": ("mooncake", "conv-rule-strict", "Mooncake", "mooncake", "A"),
}
FI_STAGE_A = (1, 5, 9, 13)
for _w in range(1, 17):
    SLICES[f"freeinference/week{_w:02d}"] = ("freeinference", f"week{_w:02d}", "FreeInference", "freeinference", "A" if _w in FI_STAGE_A else "B")
SLICES["synth/small-seed1"] = ("synth", "small-seed1", "synth", "synth", "S")        # 合成：測試與端到端用
STAGE_SLICES = {"A": [s for s, v in SLICES.items() if v[4] == "A"], "B": [s for s, v in SLICES.items() if v[4] == "B"],
                "S": ["synth/small-seed1"]}
STAGES = {"A1": ("A", "A1"), "A2": ("A", "A2"), "B-A1": ("B", "A1"), "B-A2": ("B", "A2")}
# a1_loose／a1_loose_blk＝寬鬆版雙胞胎（04 §4，只報告）：O 與所有對手都用寬鬆品質規則；只跑主視角、O 只跑 H；放在 A1（B-A1 跟著）
C0UTIL_OUTERS = [("E2", None), ("E3", None), ("E4", None), ("E5", 1800)]       # 第三輪審查 #5 的例外：凍結前只跑 C0，算每格的 SSD 使用率
COMPONENTS = {"A1": ["a1_sess", "a1_blk", "a1_loose", "a1_loose_blk"], "A2": ["a2_mem", "a2_blk", "sens_qwen", "sens_70b", "sens_ov2", "sens_ov4"]}


def is_synth(slice_name):
    return SLICES[slice_name][0] == "synth"


# ------------------------------------------------------------------ 壓縮做法（04 §3）：標籤、名字、參數、是不是 oracle
def policy_table():
    """04 §3.1。長度門檻一律用 max(存的長度, 請求輸入長度)（len_rule="max"；第二輪審查 #8）。"""
    pols = []
    for n in ("C0", "C1", "C1-8", "C2", "C3", "C6", "C1-2"):
        pols.append(dict(label=n, name=n, kw={}, oracle=False))
    for t in (8192, 16384, 32768):
        pols.append(dict(label=f"C4-{t}", name="C4", kw={"c4_threshold_tokens": t, "len_rule": "max"}, oracle=False))
    for t in (8192, 16384, 32768):
        pols.append(dict(label=f"C4r-{t}", name="C4r", kw={"c4_threshold_tokens": t, "len_rule": "max"}, oracle=False))
    for t in (8192, 16384, 32768):
        pols.append(dict(label=f"D-len-{t}", name="D-len", kw={"d_threshold_tokens": t, "len_rule": "max"}, oracle=False))
    for tau in (300, 1800, 3600, 14400):
        pols.append(dict(label=f"D-idle-{tau}", name="D-idle", kw={"d_threshold_tokens": 32768, "d_idle_tau_s": float(tau), "len_rule": "max"},
                         oracle=False))
    pols.append(dict(label="R2", name="R2", kw={"d_threshold_tokens": 32768, "len_rule": "max"}, oracle=False))
    for n in ("O", "O-tier", "O-len", "O-reuse", "O-loose"):        # O-tier（第三輪審查 #1 選項 b）：兩趟；O／O-tier 哪個當判定用的「O」由 04 §9 的 `完美預測：` 決定
        pols.append(dict(label=n, name=n, kw={}, oracle=True))
    return pols


POLICIES = policy_table()
NON_ORACLE = [p["label"] for p in POLICIES if not p["oracle"]]
ORACLE = [p["label"] for p in POLICIES if p["oracle"]]
assert len(NON_ORACLE) == 21 and len(ORACLE) == 5
LOOSE_POLS = [p for p in POLICIES if (not p["oracle"]) or p["label"] in ("O", "O-tier")]       # 寬鬆版雙胞胎：21 個對手＋O＋O-tier（哪個當 O 由 04 決定，雙胞胎兩個都跑）


# ------------------------------------------------------------------ H、k、凍結檢查
def load_H(path=RETURN_GAPS):
    """H＝max(1800, p99)（04 §5）；從 return_gaps.csv 讀，順便檢查 H_v4_s＝max(1800, p99_s)。"""
    out, n_req = {}, {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            h = max(1800.0, float(r["p99_s"]))
            assert abs(h - float(r["H_v4_s"])) < 1e-6, (r["slice"], h, r["H_v4_s"])
            out[r["slice"]] = h
            n_req[r["slice"]] = int(r["n_req"])
    return out, n_req


DOC_H_NAMES = {"Bailian coder": "bailian/coder", "Bailian traceA": "bailian/traceA", "WildChat（整份）": "wildchat/1m",
               "Mooncake conversation strict": "mooncake/conv-rule-strict", "FreeInference 第 1 週": "freeinference/week01",
               "FreeInference 第 5 週": "freeinference/week05", "FreeInference 第 9 週": "freeinference/week09",
               "FreeInference 第 13 週": "freeinference/week13"}


def prereg_text(frozen=None):
    if frozen:
        return subprocess.run(["git", "-C", REPO, "show", f"{frozen}:{PREREG}"], check=True, capture_output=True, text=True).stdout
    return open(os.path.join(REPO, PREREG), encoding="utf-8").read()


def check_H_against_doc(Hs, text):
    """04 §9 的表寫的 H（第 3 欄）必須和 return_gaps.csv 算出來的差 ≤ 0.1 s；8 個有寫的切片都要找得到。回傳逐切片的對照列。"""
    found = {}
    for line in text.split("\n"):
        m = re.match(r"^\|\s*(.+?)\s*\|\s*([\d,\.]+)\s*\|\s*([\d,\.]+)\s*\|", line)
        if m and m.group(1) in DOC_H_NAMES:
            found[DOC_H_NAMES[m.group(1)]] = float(m.group(3).replace(",", ""))
    missing = [s for s in DOC_H_NAMES.values() if s not in found]
    assert not missing, f"04 §9 的表找不到這些切片的 H：{missing}"
    rows = []
    for s, h_doc in found.items():
        assert abs(Hs[s] - h_doc) <= 0.1, f"{s}: return_gaps.csv 算出的 H={Hs[s]} 和 04 §9 寫的 {h_doc} 差超過 0.1 s"
        rows.append((s, Hs[s], h_doc))
    return rows


def check_frozen(commit):
    """04 在 <commit> 的版本不得含「未凍結」。回傳 (commit 全名, 內容 sha256)。"""
    try:
        text = prereg_text(commit)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"--frozen {commit}: git show 失敗：{e.stderr}")
    if "未凍結" in text:
        raise SystemExit(f"--frozen {commit}: {PREREG} 還含「未凍結」，拒絕跑真實資料的格子")
    full = subprocess.run(["git", "-C", REPO, "rev-parse", commit], check=True, capture_output=True, text=True).stdout.strip()
    return full, hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_kall(path=KALL):
    """(slice, cap_GiB, view) → dict(k, hourmax_at_k, overload)。view＝idle｜busy（E6 的 k；04 §9）。"""
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[(r["slice"], float(r["cap_GiB"]), r["view"])] = dict(k=int(r["k"]), hourmax=float(r["hourmax_busy_at_k"]),
                                                                       overload=r["overload_hours"] == "True")
    return out


# ------------------------------------------------------------------ 工作的列舉
def _view_name(spec):
    return f"{spec[0]}|{spec[4]}|gpu={spec[1]}|cpu={spec[2]}|ssd/{spec[3]}|{spec[5]}"


def mk_job(stage, comp, slc, engine, outer, T, mem, pol, tag, model="llama", overlay=1, view=None, k=None, rule="strict"):
    """tag：'H'｜'2H'（oracle 各一個工作）或 'both'（非 oracle：H 當放置與主截尾、2H 當第二截尾，一次模擬兩個版本）。"""
    oc = outer + (f"-T{int(T)}" if T else "")
    jid = f"{engine}__{oc}__m{int(mem)}__{model}{'_ov%d' % overlay if overlay > 1 else ''}__{pol['label']}" + \
          (f"__v{view}" if view else "") + f"__{tag}" + ("__loose" if rule == "loose" else "")
    return dict(stage=stage, comp=comp, slice=slc, engine=engine, outer=outer, T=T, mem=mem, pol=pol, tag=tag, model=model,
                overlay=overlay, view=view, k=k, id=jid, rule=rule)


def enumerate_jobs(stage, slices=None, fill=False, skip=(), kall=None):
    """stage＝A1｜A2｜B-A1｜B-A2（04 §10）。slices＝限制切片；skip＝略過的元件名；回傳 jobs（還沒排序）。"""
    grp, kind = STAGES[stage]
    sl = [s for s in STAGE_SLICES[grp] if (not slices or s in slices)]
    if slices:        # --slices 可以直接指定階段以外的切片（合成資料測試用）
        sl = [s for s in slices if s in SLICES]
    comps = [c for c in COMPONENTS[kind] if c not in skip]
    jobs = []
    for s in sl:
        for comp in comps:
            if comp == "a1_sess":
                jobs += _sess_jobs(stage, comp, s, OUTERS_MAIN, A1_MEM)
            elif comp == "a2_mem":
                jobs += _sess_jobs(stage, comp, s, OUTERS_ALL, A2_MEM)
                if fill:
                    jobs += _sess_jobs(stage, "a2_fill", s, OUTERS_FILL, A1_MEM)
            elif comp == "a1_blk":
                jobs += _blk_jobs(stage, comp, s, A1_MEM, ("idle",), kall)
            elif comp == "a1_loose":
                jobs += _sess_jobs(stage, comp, s, OUTERS_MAIN, A1_MEM, rule="loose")
            elif comp == "a1_loose_blk":
                jobs += _blk_jobs(stage, comp, s, A1_MEM, ("idle",), kall, rule="loose", outers=("E6",))
            elif comp == "a2_blk":
                jobs += _blk_jobs(stage, comp, s, A2_MEM, ("idle", "busy"), kall)
            elif comp == "sens_qwen":
                jobs += _sess_jobs(stage, comp, s, OUTERS_MAIN, A1_MEM, model="qwen")
            elif comp == "sens_70b":
                jobs += _sess_jobs(stage, comp, s, OUTERS_MAIN, A1_MEM, model="llama70b_scaled")
            elif comp.startswith("sens_ov"):
                jobs += _sess_jobs(stage, comp, s, OUTERS_MAIN, A1_MEM, overlay=int(comp[-1]))
    return jobs


def _pol_tags(p, rule="strict"):
    if rule == "loose":          # 寬鬆版雙胞胎只報告 H
        return ("H",) if p["oracle"] else ("both",)
    return ("H", "2H") if p["oracle"] else ("both",)


def _sess_jobs(stage, comp, s, outers, mems, model="llama", overlay=1, rule="strict"):
    out = []
    for (oc, T) in outers:
        for m in mems:
            for p in (LOOSE_POLS if rule == "loose" else POLICIES):
                for tag in _pol_tags(p, rule):
                    out.append(mk_job(stage, comp, s, "session", oc, T, m, p, tag, model, overlay, rule=rule))
    return out


def _blk_jobs(stage, comp, s, mems, views, kall, rule="strict", outers=("E6", "E2")):
    out = []
    for oc in outers:
        for m in mems:
            for v in views:
                k = None
                if kall is not None:
                    key = (s, float(m), v)
                    if key not in kall:
                        raise SystemExit(f"{KALL} 沒有 {key} 的 k")
                    k = kall[key]["k"]
                for p in (LOOSE_POLS if rule == "loose" else POLICIES):
                    for tag in _pol_tags(p, rule):
                        out.append(mk_job(stage, comp, s, "block", oc, None, m, p, tag, "llama", 1, v, k, rule))
    return out


def n_rows_of(j):
    if j["engine"] == "block":
        return 1
    if j["comp"] == "a1_sess":
        return len(MAIN_SPECS) + len(SENS_SPECS)
    if j["comp"] in ("a1_loose", "c0util"):
        return 1
    if j["model"] == "llama70b_scaled":
        return 1
    if j["comp"].startswith("sens_"):
        return len(SENS_LITE_SPECS)
    return len(MAIN_SPECS)


# ------------------------------------------------------------------ 時間估計〔估計〕（沿用 estimate_runtime.py 的做法與數字）
def load_timing(path=None):
    """3090 的計時 run（06 §16.1）。只用來排工作的先後與估時間，不影響任何數字。
    檔不在（例如 H200 上沒有 /ssd8）→ None：est_cost_s 改用請求數排序（2026-10-19）。"""
    import ast
    path = path or os.environ.get("P3_TIMING_STDOUT", TIMING_STDOUT)
    if not os.path.exists(path):
        print(f"[run_map] 沒有計時檔 {path}：工作順序改用請求數估（只影響先後，不影響結果）", flush=True)
        return None
    rows = [ast.literal_eval(l.strip()) for l in open(path) if l.strip().startswith("{")]
    cost = {}
    for r in rows:
        cost.setdefault((r["source"] + "/" + r["variant"], r["policy"]), []).append((r["session_s_per_req"], r["block_s_per_req"]))
    return {k: (sum(x[0] for x in v) / len(v), sum(x[1] for x in v) / len(v)) for k, v in cost.items()}


def est_cost_s(j, timing, n_req):
    """〔估計〕一個工作的 CPU 秒。每請求秒數用計時 run（06 §16.1）：FreeInference 各週與 synth 用 week13 的；
    非 oracle 用 C1 的、oracle 用 O 的；E6 分 k 張 GPU 時總工作量不變；疊加×k 的請求數×k。沒有量過每個視角數的差別，一律用量到的（8 個時間設定）。"""
    s = j["slice"]
    if timing is None:                 # 沒有計時檔：只按工作量的比例排（請求數 × 疊加 × O-tier 兩趟）
        return (2.0 if j["pol"]["label"] == "O-tier" else 1.0) * n_req[s] * j["overlay"]
    ref = s if (s, "C1") in timing else "freeinference/week13"      # FreeInference 各週、synth（每請求約 90 個 block，和 week13 同量級）都用 week13 的每請求秒數
    sc, bc = timing[(ref, "O" if j["pol"]["oracle"] else "C1")]
    mult = 2.0 if j["pol"]["label"] == "O-tier" else 1.0         # O-tier＝兩趟（第 1 趟＝一般的 O、不需要時間設定）：上界按 2 倍算
    return mult * n_req[s] * j["overlay"] * (sc if j["engine"] == "session" else bc)


# ------------------------------------------------------------------ 子集（E6：session 分到 k 張 GPU）
def sess_group(sess_key, k):
    return np.array([int(hashlib.sha1(str(x).encode("utf-8")).hexdigest(), 16) % k for x in sess_key], dtype=np.int64)


def sub_P(P, idx):
    """P 的子集（idx 遞增，同一個 session 的請求一定都在）；session 重新編號。"""
    inv = {int(r): i for i, r in enumerate(idx)}
    par = []
    for r in idx:
        p = P["parentl"][r]
        if p >= 0:
            assert p in inv, "父請求不在同一張 GPU（session 被拆開了）"
            par.append(inv[p])
        else:
            par.append(-1)
    sl = np.asarray(P["sessl"])[idx]
    uniq, new = np.unique(sl, return_inverse=True)
    kd = [P["kindc"][r] for r in idx]
    meta = dict(P["meta"], n_req=len(idx), n_sess=len(uniq), n_ret=int(sum(kd)))
    out = dict(full=[P["full"][r] for r in idx], nall=[P["nall"][r] for r in idx], tl=[P["tl"][r] for r in idx], kindc=kd,
               parentl=par, sessl=new.tolist(), obs_end=[P["obs_end"][r] for r in idx], win=P["win"], meta=meta,
               R=dict(n_sess=len(uniq)), nmax=P["nmax"], B=P["B"], sess_key=[P["sess_key"][u] for u in uniq],
               orig_sess=uniq.astype(np.int64))
    if P.get("feats"):
        out["feats"] = [P["feats"][r] for r in idx]
    return out


# ------------------------------------------------------------------ 工作的執行（在 worker 裡）
_PC = {}


def get_P(slc, overlay):
    k = (slc, overlay)
    if k not in _PC:
        _PC.clear()               # 一個 worker 只留一份 P（記憶體）
        src, var = SLICES[slc][0], SLICES[slc][1]
        _PC[k] = TR.prepare_cached(src, var, overlay=overlay, seed=0)
    return _PC[k]


# 位元組、次數類的欄：k 張 GPU 加總；p99／最大／比例：取各 GPU 的最大；其他（時間類）→ NaN（沒有定義「合併」）
def _pool_kind(col):
    if col.endswith("_GiB") or col.startswith(("n_", "ret_", "reads", "surprise_", "wblocks_", "recompute_prefix", "ttft_nm", "blk_", "recompute_charge_blocks", "otier_pass", "otier_ssd", "otier_cpu_event", "otier_tier",
                                               "n_ret")) or col in ("n_demote", "n_drop", "n_lost_blocks"):
        return "sum"
    if "_p99" in col or "_max" in col or col.endswith("_frac") or col in ("gpu_busy_frac",):
        return "max"
    return "other"


def _pool_rows(rows):
    """k 張 GPU 的列合成一列（只合併加總型與最大型的欄；時間類另外用逐請求的值重算）。"""
    out = dict(rows[0])
    for c in rows[0]:
        v0 = rows[0][c]
        if isinstance(v0, bool) or isinstance(v0, str) or v0 is None:
            continue
        if not isinstance(v0, (int, float, np.integer, np.floating)):
            continue
        kd = _pool_kind(c)
        vals = [r[c] for r in rows if isinstance(r.get(c), (int, float, np.integer, np.floating)) and not (isinstance(r.get(c), float) and math.isnan(r[c]))]
        if kd == "sum":
            out[c] = float(sum(vals)) if isinstance(v0, float) else int(sum(vals))
        elif kd == "max":
            out[c] = max(vals) if vals else float("nan")
        else:
            allv = [r.get(c) for r in rows]
            out[c] = v0 if all(x == v0 or (isinstance(x, float) and isinstance(v0, float) and math.isnan(x) and math.isnan(v0)) for x in allv) else float("nan")
    for c in rows[0]:
        if isinstance(rows[0][c], str) and c.startswith(("ttft", "restore")) and rows[0][c] == NOT:
            out[c] = NOT
    return out


TTFT_EDGES = np.logspace(-4, 7, 2401)      # 回來請求的等待時間直方圖的邊界（每格 ×1.0107；FreeInference 幾週合起來的 p99 用，analyze_map.hist_pct）


def ttft_arrays(x, nm, keep_vals=False):
    """回來請求的 ttft（秒）→ 直方圖與加總。nm＝這些請求裡 NOT_MEASURED 的個數（>0 時整個 ttft 不能用）。
    keep_vals＝True（FreeInference 各週）：另存逐請求的 ttft（float64），幾週合起來的護欄 p99 用它精確算（04 §4；不用直方圖）。"""
    x = np.asarray(x, float)
    h = np.histogram(np.clip(x, TTFT_EDGES[0], TTFT_EDGES[-1]), TTFT_EDGES)[0].astype(np.int64)
    d = dict(ttft_hist=h, ttft_n=np.asarray(len(x), np.int64), ttft_sum=np.asarray(float(x.sum())), ttft_nm=np.asarray(int(nm), np.int64))
    if keep_vals:
        d["ttft_vals"] = x.copy()
    return d


def keeps_ttft_vals(slice_name):
    """同一個判定群組有好幾個切片（FreeInference 的週）→ 護欄要把它們合起來 → 存逐請求的 ttft。"""
    g = SLICES[slice_name][2]
    return sum(1 for v in SLICES.values() if v[2] == g) > 1


def cut_ttft(P, view, cut):
    """回來的請求裡到達 < cut 的：(ttft 陣列, NOT_MEASURED 的個數)。cut＝終點 − H（截尾；04 §4 的護欄用截尾版 ttftq_H_*）。"""
    tl = np.asarray(P["tl"], float)
    m = (np.asarray(P["kindc"]) == 1) & (tl < cut)
    return np.asarray(view.ttft)[m], int(np.asarray(view.nmreq, bool)[m].sum())


def _slim_arrays(P, eng, view, orig_sess, keep_vals=False):
    arr = ledger.per_session(P, eng, [view])
    arr["sess_orig"] = np.asarray(orig_sess, np.int64)
    ret = np.asarray(P["kindc"]) == 1
    arr.update(ttft_arrays(np.asarray(view.ttft)[ret], int(np.asarray(view.nmreq, bool)[ret].sum()), keep_vals))
    # 截尾版護欄：tH＝H 的截尾（oracle 的 H／2H 工作＝它自己的 H），tH2＝第二個截尾（非 oracle 的 2H）。逐請求的值只在需要合併幾個切片時存
    for pfx, cut in (("tH", getattr(eng, "cutH", None)), ("tH2", getattr(eng, "cutH2", None))):
        if cut is None:
            continue
        x, nm = cut_ttft(P, view, cut)
        arr[f"{pfx}_n"], arr[f"{pfx}_sum"], arr[f"{pfx}_nm"] = np.asarray(len(x), np.int64), np.asarray(float(x.sum())), np.asarray(nm, np.int64)
        if keep_vals:
            arr[f"{pfx}_vals"] = x.astype(np.float64)
    return arr


def _concat_arrays(parts):
    out = {}
    for key in parts[0]:
        xs = [p[key] for p in parts]
        a0 = np.asarray(xs[0])
        if a0.ndim == 0:
            out[key] = a0
        elif a0.ndim == 1:
            out[key] = np.concatenate([np.asarray(x) for x in xs])
        else:
            out[key] = np.concatenate([np.asarray(x) for x in xs], axis=1)
    for key in ("ttft_hist", "ttft_n", "ttft_sum", "ttft_nm", "tH_n", "tH_sum", "tH_nm", "tH2_n", "tH2_sum", "tH2_nm"):   # 直方圖與加總是加起來，不是接起來
        if key in parts[0]:
            out[key] = sum(np.asarray(p[key]) for p in parts)
    return out


def _atomic_savez(path, **arrs):
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


def k_util_of(ctx, slc, mem):
    """SSD 使用率的 k（第三輪審查 #5）：這份資料片段、這個記憶體的 E6 的 k（kgpu_kmin_all.csv，idle 視角）；所有引擎都用同一個 k。查不到就丟錯（規則 7）。"""
    kall = ctx.get("kall")
    key = (slc, float(mem), "idle")
    if kall is None or key not in kall:
        raise KeyError(f"{KALL} 沒有 {key} 的 k（SSD 使用率要用）")
    return int(kall[key]["k"])


def common_cols(jd, ctx):
    pol = jd["pol"]
    H = ctx["H"][jd["slice"]]
    return dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"), stage=jd["stage"], comp=jd["comp"],
                slice=jd["slice"], trace_group=SLICES[jd["slice"]][2], source_group=SLICES[jd["slice"]][3],
                synthetic=is_synth(jd["slice"]), semi_synthetic=jd["overlay"] > 1, overlay=jd["overlay"], job_id=jd["id"],
                engine=jd["engine"], outer_code=jd["outer"], T_s=jd["T"] or "", mem_GiB=jd["mem"], model=jd["model"],
                policy_label=pol["label"], policy_family=pol["name"], policy_kw=json.dumps(pol["kw"], sort_keys=True),
                oracle=pol["oracle"], h_tag=jd["tag"], H_base_s=H, int2_on_read=INT2_ON_READ, unverified_len_mode=UNVERIFIED_LEN,
                frozen_commit=ctx.get("frozen_commit", ""), code_commit=ctx.get("code_commit", ""),
                quality_rule=jd.get("rule", "strict"), cpu_busy_run_id=CPU_BUSY_RUN, cpu_busy_lo=CPU_BUSY_LO, cpu_busy_hi=CPU_BUSY_HI,
                recompute_charge=ctx.get("recompute_charge", "none"), k_util=k_util_of(ctx, jd["slice"], jd["mem"]))


def prim_cols(jd, row, H, P=None, costs=None):
    """主指標（04 §4）＝SSD 寫入（決定時刻 < 終點 − H）＋ SSD 讀取（到達 < 終點 − H 的請求）。
    讀取（第三輪審查 #4；06 §26，兩個引擎同一個定義）＝只載入的還原讀取（lo_bytes；和 Cake 的會合點無關）＋全部預取讀取（含浪費；session 引擎沒有預取）
    ＝ssd_H_primread_GiB；recompute_charge＝pass_size 時再加重算計價 ssd_H_charged_recompute_GiB（第三輪審查 #1 選項 a）。
    prim_rbase_*／prim_rc_*＝兩個分量（分析程式依凍結的 04 自己決定要不要加 rc，不依賴這裡的開關）。`prim_read_col` 記用了哪些欄。
    非 oracle：H 與 2H 兩個版本都在這一列；oracle：只有自己的 H 那一個。
    另記每個版本的寫入／讀取分量（prim_w_*／prim_r_*）、截尾後的窗口長度（T_*_s）與 SSD 頻寬（主視角 SSD 1 張專用的 mi300x_local 常數），
    給 SSD 使用率＝寫入量÷(寫入頻寬×T)＋讀取量÷(讀取頻寬×T) 用（§7；block 引擎 k 張 GPU 各有一顆專用 SSD → 分母再乘 k）。"""
    w, r = row.get("ssd_H_write_issued_GiB"), row.get("ssd_H_primary_read_GiB")
    w2, r2 = row.get("ssd_H2_write_issued_GiB"), row.get("ssd_H2_primary_read_GiB")
    nan = float("nan")
    chg = row.get("recompute_charge", "none")
    out = dict(prim_H_GiB=nan, prim_2H_GiB=nan, npz_sfx_H="", npz_sfx_2H="",
               prim_read_col="ssd_H_primread_GiB" + ("+ssd_H_charged_recompute_GiB" if chg == "pass_size" else ""),
               prim_rbase_H_GiB=nan, prim_rc_H_GiB=nan, prim_rbase_2H_GiB=nan, prim_rc_2H_GiB=nan,
               prim_w_H_GiB=nan, prim_r_H_GiB=nan, prim_w_2H_GiB=nan, prim_r_2H_GiB=nan, T_H_s=nan, T_2H_s=nan,
               ssd_bw_r_GiBps=costs.ssd["r"] if costs else nan, ssd_bw_w_GiBps=costs.ssd["w"] if costs else nan)
    if P is not None:
        lo, _ = ledger.win_range(P)
        oe = float(max(P["obs_end"])) if P.get("obs_end") else float(P["tl"][-1])
    else:
        lo, oe = 0.0, nan

    def T(h):
        return max(oe - h - lo, 0.0)
    if jd["tag"] == "both":
        H0 = H             # 'both'：H＝這份資料的 H，H_extra＝2H
        out.update(prim_H_GiB=w + r, prim_2H_GiB=w2 + r2, npz_sfx_H="_H", npz_sfx_2H="_H2",
                   ssd_H_write_issued_GiB_2H=w2, ssd_H_loadonly_read_GiB_2H=row.get("ssd_H2_loadonly_read_GiB"),
                   prim_w_H_GiB=w, prim_r_H_GiB=r, prim_w_2H_GiB=w2, prim_r_2H_GiB=r2, T_H_s=T(H0), T_2H_s=T(2 * H0),
                   prim_rbase_H_GiB=row["ssd_H_primread_GiB"], prim_rc_H_GiB=row["ssd_H_charged_recompute_GiB"],
                   prim_rbase_2H_GiB=row["ssd_H2_primread_GiB"], prim_rc_2H_GiB=row["ssd_H2_charged_recompute_GiB"])
    elif jd["tag"] == "H":
        out.update(prim_H_GiB=w + r, npz_sfx_H="_H", prim_w_H_GiB=w, prim_r_H_GiB=r, T_H_s=T(H),
                   prim_rbase_H_GiB=row["ssd_H_primread_GiB"], prim_rc_H_GiB=row["ssd_H_charged_recompute_GiB"])
    else:
        out.update(prim_2H_GiB=w + r, npz_sfx_2H="_H", prim_w_2H_GiB=w, prim_r_2H_GiB=r, T_2H_s=T(H),
                   prim_rbase_2H_GiB=row["ssd_H_primread_GiB"], prim_rc_2H_GiB=row["ssd_H_charged_recompute_GiB"])
    return out


def run_placements(jd, P, ctx):
    """一個工作的模擬：回傳 (rows, npz_arrays, meta)。session 引擎＝一次放置、多個時間設定；block 引擎＝k 張 GPU 各一次、加總。"""
    pol = jd["pol"]
    H0 = ctx["H"][jd["slice"]]
    if jd["tag"] == "both":
        H, Hx = H0, 2 * H0
    else:
        H, Hx = (H0 if jd["tag"] == "H" else 2 * H0), 0.0
    kw = dict(pol_kw=dict(pol["kw"]), int2_on_read=INT2_ON_READ, unverified_len=UNVERIFIED_LEN, H=H, H_extra=Hx, fmt_path=ctx["fmt"],
              quality_rule=jd.get("rule", "strict"), recompute_charge=ctx.get("recompute_charge", "none"))
    base = common_cols(jd, ctx)
    costs_p = load_costs(MAP_PLAT, jd["model"], "mi300x_local")        # 主視角的 SSD 頻寬（使用率用）
    if jd["engine"] == "session":
        if jd["comp"] in ("a1_loose", "c0util"):    # 寬鬆版雙胞胎、C0 使用率：只要主視角
            specs, groups = [PRIMARY_SPEC], ["main"]
        elif jd["comp"] == "a1_sess":
            specs = MAIN_SPECS + (SENS_SPECS if ctx.get("sens_views", True) else [])
            groups = ["main"] * len(MAIN_SPECS) + (SENS_GROUPS if ctx.get("sens_views", True) else [])
        elif jd["model"] == "llama70b_scaled":       # 沒有 f(i)：只能 load-only、不算 TTFT（06 §1.1）
            specs, groups = [(MAP_PLAT, "idle", "idle", 1, "load", "mi300x_local")], ["main"]
        elif jd["comp"].startswith("sens_"):
            specs, groups = SENS_LITE_SPECS, ["main"] * len(SENS_LITE_SPECS)
        else:
            specs, groups = MAIN_SPECS, ["main"] * len(MAIN_SPECS)
        rows, _, ids, eng, views = sim.run_cell(P, jd["outer"], pol["name"], jd["mem"], jd["model"], specs, T=jd["T"], **kw)
        assert all(i["ok"] for i in ids), [i for i in ids if not i["ok"]][:3]
        arr = _slim_arrays(P, eng, views[0], np.arange(P["R"]["n_sess"]), keeps_ttft_vals(jd["slice"]))
        out_rows = []
        for i, (r, g) in enumerate(zip(rows, groups)):
            c = dict(base, view_group=g, view_idx=i, is_primary=(g == "main" and i == 0), k_gpus=1, view_name=_view_name(specs[i]), **r)
            c.update(prim_cols(jd, r, H, P, costs_p))
            out_rows.append(c)
        meta = dict(k=1, n_groups_nonempty=1)
        return out_rows, arr, meta
    # ---- block 引擎
    spec = BLOCK_VIEWS[jd["view"]]
    k = jd["k"]
    gid = sess_group(P["sess_key"], k)
    sl = np.asarray(P["sessl"])
    parts, grows, xs, nmc, nretc, hasf, per_n = [], [], [], 0, 0, True, []
    cutx = {"H": ([], [0]), "H2": ([], [0])}          # 截尾版護欄（ttftq_H_*、ttftq_H2_*）：k 張 GPU 的逐請求值接起來
    for g in range(k):
        idx = np.nonzero(gid[sl] == g)[0]
        per_n.append(int(len(idx)))
        if not len(idx):
            continue
        Pg = P if k == 1 else sub_P(P, idx)
        if k == 1:
            Pg = dict(P, orig_sess=np.arange(P["R"]["n_sess"]))
        rows, _, ids, eng, vs = sim.run_cell_fifo1(Pg, jd["outer"], pol["name"], jd["mem"], jd["model"], spec, **kw)
        assert all(i["ok"] for i in ids), [i for i in ids if not i["ok"]][:3]
        v = vs[0]
        ret = np.asarray(Pg["kindc"]) == 1
        xs.append(np.asarray(v.ttft)[ret])
        nmc += int(np.asarray(v.nmreq, bool)[ret].sum())
        nretc += int(ret.sum())
        hasf = hasf and bool(v.has_f)
        for nm_, cut_ in (("H", getattr(eng, "cutH", None)), ("H2", getattr(eng, "cutH2", None))):
            if cut_ is not None:
                x_, n_ = cut_ttft(Pg, v, cut_)
                cutx[nm_][0].append(x_)
                cutx[nm_][1][0] += n_
        grows.append(rows[0])
        parts.append(_slim_arrays(Pg, eng, v, Pg["orig_sess"], keeps_ttft_vals(jd["slice"])))
        del eng, vs
    row = _pool_rows(grows)
    x = np.concatenate(xs) if xs else np.zeros(0)
    ok = hasf and nmc == 0
    row.update(n_ret=nretc, ttft_nm_ret_requests=nmc, ttftq_valid=ok,
               ttftq_mean_s=float(x.mean()) if (ok and len(x)) else (NOT if not ok else float("nan")),
               ttftq_p99_s=float(np.percentile(x, 99)) if (ok and len(x)) else (NOT if not ok else float("nan")),
               ttftq_p50_s=float(np.percentile(x, 50)) if (ok and len(x)) else (NOT if not ok else float("nan")),
               gpu_busy_frac_max=max(r["gpu_busy_frac"] for r in grows))
    for nm_ in ("H", "H2"):
        if cutx[nm_][0]:
            xc = np.concatenate(cutx[nm_][0])
            okc = hasf and cutx[nm_][1][0] == 0
            row[f"ttftq_{nm_}_mean_s"] = (float(xc.mean()) if len(xc) else float("nan")) if okc else NOT
            row[f"ttftq_{nm_}_p99_s"] = (float(np.percentile(xc, 99)) if len(xc) else float("nan")) if okc else NOT
    c = dict(base, view_group="main", view_idx=0, is_primary=(jd["view"] == "idle"), k_gpus=k, k_group_n_req=json.dumps(per_n),
             view_name=_view_name(spec), **row)
    c.update(prim_cols(jd, row, H, P, costs_p))
    return [c], _concat_arrays(parts), dict(k=k, n_groups_nonempty=len(parts))


def c0util_jobs(slices=None, kall=None):
    """凍結前的例外（第三輪審查 #5、06 §26）：只跑 C0、階段 A 的切片、主視角、H：session 引擎 E2／E3／E4／E5(T=1800)，block 引擎 E6（那個切片的 k）；
    三個記憶體點。用來決定哪些格子 SSD 夠忙（C0 的使用率 ≥5%）才能進判定；不產生任何其他做法或等待時間的輸出。"""
    pol = next(p for p in POLICIES if p["label"] == "C0")
    sl = list(slices) if slices else list(STAGE_SLICES["A"])
    jobs = []
    for s in sl:
        for (oc, T) in C0UTIL_OUTERS:
            for m in A1_MEM:
                jobs.append(mk_job("C0UTIL", "c0util", s, "session", oc, T, m, pol, "H"))
        for m in A1_MEM:
            if kall is None or (s, float(m), "idle") not in kall:
                raise SystemExit(f"{KALL} 沒有 {(s, float(m), 'idle')} 的 k")
            jobs.append(mk_job("C0UTIL", "c0util", s, "block", "E6", None, m, pol, "H", "llama", 1, "idle", kall[(s, float(m), "idle")]["k"]))
    return jobs


def c0util_row(r):
    """C0 的使用率列（只留這些欄；沒有等待時間、沒有別的做法）。使用率＝寫入量÷(寫入頻寬×T×k)＋讀取量÷(讀取頻寬×T×k)，T＝切片窗口截到終點−H，
    k＝這個切片這個記憶體的 E6 的 k（kgpu_kmin_all.csv；所有引擎同一個）。GiB／小時＝(寫入＋讀取)÷T×3600（全部 GPU 合計）。"""
    w, rd, T, k = float(r["prim_w_H_GiB"]), float(r["prim_r_H_GiB"]), float(r["T_H_s"]), int(r["k_util"])
    ol = r["outer_code"] + (f"-T{int(float(r['T_s']))}" if r["T_s"] not in ("", None) else "")
    util = (w / float(r["ssd_bw_w_GiBps"]) + rd / float(r["ssd_bw_r_GiBps"])) / (T * k) if T > 0 else float("nan")
    return dict(slice=r["slice"], outer=ol, engine=r["engine"], mem_GiB=r["mem_GiB"], k=k, c0_write_GiB=w, c0_read_GiB=rd, gib_per_hour=(w + rd) / (T / 3600.0) if T > 0 else float("nan"),
                utilization=util, H_s=r["H_base_s"], T_H_s=T, ssd_bw_w_GiBps=r["ssd_bw_w_GiBps"], ssd_bw_r_GiBps=r["ssd_bw_r_GiBps"], read_def=r["prim_read_col"],
                job_id=r["job_id"], run_id=r["run_id"], ts=r["ts"], code_commit=r["code_commit"], frozen_commit=r["frozen_commit"] or "NOT_FROZEN")


def collect_c0util(out, csv_path):
    """jobs/*/*.json（comp=c0util）→ csv_path（results/p3_sim/c0_ssd_util.csv）。回傳列數。"""
    rows = []
    for root, _, files in os.walk(os.path.join(out, "jobs")):
        for fn in sorted(files):
            if fn.endswith(".json"):
                d = json.load(open(os.path.join(root, fn), encoding="utf-8"))
                if d["comp"] != "c0util":
                    continue
                if d["status"] != "ok":
                    raise SystemExit(f"{fn}: status={d['status']}：{(d.get('error') or '')[-400:]}")
                rows += d["rows"]
    order = {"E2": 0, "E3": 1, "E4": 2, "E5-T1800": 3, "E6": 4}
    rows.sort(key=lambda r: (r["slice"], order[r["outer"]], r["mem_GiB"]))
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return len(rows)


def run_job(jd, ctx):
    """worker：跑一個工作，寫 npz 再寫 json（json 在＝一切都在）。例外都記在 json（FAILED／NOT_MEASURED），不讓 worker 掛掉。"""
    t0 = time.time()
    jpath, npath = jd["json_path"], jd["npz_path"]
    res = dict(job_id=jd["id"], stage=jd["stage"], comp=jd["comp"], slice=jd["slice"], run_id=os.environ.get("RUN_ID", "adhoc"),
               ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"), frozen_commit=ctx.get("frozen_commit", ""), code_commit=ctx.get("code_commit", ""))
    try:
        P = get_P(jd["slice"], jd["overlay"])
        rows, arr, meta = run_placements(jd, P, ctx)
        pol = jd["pol"]
        if jd["comp"] == "c0util":     # 凍結前的例外（第三輪審查 #5）：只留 C0 的使用率欄位，不留任何等待時間或其他做法
            rows = [c0util_row(r) for r in rows]
            res.update(status="ok", rows=rows, meta=meta, elapsed_s=time.time() - t0)
            tmp = jpath + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(res, fh, ensure_ascii=False, default=str)
            os.replace(tmp, jpath)
            return dict(job_id=jd["id"], slice=jd["slice"], status="ok", elapsed_s=res["elapsed_s"], error="")
        # ---- 不變量（04 §3.3、06 §18）：嚴格讀取規則下，所有做法的品質帳（fail＋unverified）都是 0；O、O-len 沒有意外重算
        for r in rows:
            assert r["reads_fail"] == 0 and r["reads_unverified"] == 0, f"品質帳不是 0：{r['reads_fail']} {r['reads_unverified']}"
            if pol["label"] in ("O", "O-len", "O-tier"):
                assert r["surprise_events"] == 0, f"{pol['label']} 有意外重算 {r['surprise_events']}"
        arr["meta_json"] = np.asarray(json.dumps(dict(job_id=jd["id"], slice=jd["slice"], outer=jd["outer"], T=jd["T"], mem_GiB=jd["mem"],
                                                      policy=pol["label"], tag=jd["tag"], k=meta["k"], run_id=res["run_id"]), default=str))
        _atomic_savez(npath, **arr)
        for r in rows:
            r["npz_path"] = npath
            r["json_path"] = jpath
            r["elapsed_s"] = time.time() - t0
        res.update(status="ok", rows=rows, meta=meta, elapsed_s=time.time() - t0)
    except NotMeasured as e:
        res.update(status=NOT, error=str(e), rows=[], elapsed_s=time.time() - t0)
    except Exception:  # noqa: BLE001
        res.update(status="FAILED", error=traceback.format_exc(), rows=[], elapsed_s=time.time() - t0)
    tmp = jpath + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, default=str)
    os.replace(tmp, jpath)
    return dict(job_id=jd["id"], slice=jd["slice"], status=res["status"], elapsed_s=res["elapsed_s"],
                error=res.get("error", "")[-600:] if res["status"] != "ok" else "")


# ------------------------------------------------------------------ 排程、續跑、記憶體守衛
def mem_avail_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1048576.0
    raise RuntimeError("讀不到 /proc/meminfo 的 MemAvailable")


def job_paths(out, j):
    d = os.path.join(out, "jobs", j["slice"].replace("/", "_"))
    n = os.path.join(out, "sess", j["slice"].replace("/", "_"))
    return os.path.join(d, j["id"] + ".json"), os.path.join(n, j["id"] + ".npz")


def classify(out, jobs, retry_failed=True, ctx=None):
    """已完成的跳過；只有一半的（沒有 json 的 npz、.tmp）刪掉重跑；FAILED 重跑（retry_failed）。
    ctx（第三輪審查 #2）：給了就比對舊 json 的 frozen_commit／code_commit：真實資料（ctx['frozen_commit'] 非空）不一致就丟錯（不默默沿用、也不默默覆蓋）；
    合成資料不一致就當作沒做過、重跑。"""
    todo, done, cleaned = [], [], []
    for j in jobs:
        jp, npz = job_paths(out, j)
        j["json_path"], j["npz_path"] = jp, npz
        os.makedirs(os.path.dirname(jp), exist_ok=True)
        os.makedirs(os.path.dirname(npz), exist_ok=True)
        for p in (jp + ".tmp", npz + ".tmp.npz"):
            if os.path.exists(p):
                os.remove(p)
                cleaned.append(p)
        ok = False
        if os.path.exists(jp):
            try:
                d = json.load(open(jp, encoding="utf-8"))
                ok = d.get("status") in ("ok", NOT) or (d.get("status") == "FAILED" and not retry_failed)
                if ok and ctx is not None and d.get("status") in ("ok", NOT):
                    if (d.get("frozen_commit"), d.get("code_commit")) != (ctx.get("frozen_commit", ""), ctx.get("code_commit", "")):
                        if ctx.get("frozen_commit"):
                            raise SystemExit(f"續跑檢查失敗：{jp} 是在 frozen={d.get('frozen_commit')!r}、code={d.get('code_commit')!r} 跑的，"
                                             f"現在是 frozen={ctx.get('frozen_commit')!r}、code={ctx.get('code_commit')!r}；不同 commit 的結果不能混在同一個輸出夾，換一個 --out")
                        ok = False
                if ok and d.get("status") == "ok" and not os.path.exists(npz) and j["comp"] != "c0util":
                    ok = False
                    cleaned.append(jp)
            except (json.JSONDecodeError, OSError):
                ok = False
                cleaned.append(jp)
        if not ok:
            for p in (jp, npz):
                if os.path.exists(p):
                    os.remove(p)
                    cleaned.append(p)
        (done if ok else todo).append(j)
    return todo, done, cleaned


def _warm_one(key):
    """worker：只為了讓 P 快取存在（命中就載入後丟掉；沒有就建）。回傳 (slice, overlay, 命中?, 秒數, 本 process 的最大 RSS GB)。"""
    import resource
    t0 = time.time()
    slc, ov = key
    P = TR.prepare_cached(SLICES[slc][0], SLICES[slc][1], overlay=ov, seed=0)
    return slc, ov, bool(P["meta"]["cache"]["hit"]), time.time() - t0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0


def warm_caches(todo, warm_procs, out):
    """第一次建 P 快取的峰值記憶體很大（FreeInference week13 實測 33.4 GB／次、574 s，run 20261019-001812-p3sim-c0util-prepprobe2）；
    工作一次開 24 個 worker、前 24 個又都是同一兩個切片時，每個 worker 都在建同一份快取，總記憶體超過 440 GB 被系統殺掉
    （run 20261018-234402-p3sim-c0util 的 BrokenProcessPool）。所以先用 --warm-procs 個 process 把需要的 (切片, 疊加) 快取一個一個建好，
    再開 worker（它們只會命中快取）。"""
    keys = sorted({(j["slice"], j["overlay"]) for j in todo})
    if not keys:
        return
    log = open(os.path.join(out, "runner.log"), "a", encoding="utf-8")
    with ProcessPoolExecutor(max_workers=max(1, warm_procs), mp_context=mp.get_context("fork")) as ex:
        for slc, ov, hit, sec, rss in ex.map(_warm_one, keys):
            msg = f"[{time.strftime('%H:%M:%S')}] P 快取 {slc} overlay={ov}：{'命中' if hit else '新建'}，{sec:.0f}s，最大 RSS {rss:.1f} GB"
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()


def dispatch(todo, ctx, procs, out, mem_min_gb, max_fail):
    log = open(os.path.join(out, "runner.log"), "a", encoding="utf-8")

    def say(msg):
        s = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()
    ex = ProcessPoolExecutor(max_workers=procs, mp_context=mp.get_context("fork"))
    pend, it, nfail, nd, t0, stall = {}, iter(todo), 0, 0, time.time(), False
    exhausted = False
    while True:
        while not exhausted and len(pend) < procs and nfail < max_fail:
            m = mem_avail_gb()
            if m < mem_min_gb:
                if not stall:
                    say(f"記憶體守衛：MemAvailable {m:.1f} GB < {mem_min_gb} GB，暫停開新工作（執行中 {len(pend)}）")
                    stall = True
                break
            if stall:
                say(f"記憶體守衛：MemAvailable {m:.1f} GB，恢復開新工作")
                stall = False
            try:
                j = next(it)
            except StopIteration:
                exhausted = True
                break
            pend[ex.submit(run_job, j, ctx)] = j
        if not pend:
            break
        done, _ = wait(list(pend), timeout=20, return_when=FIRST_COMPLETED)
        for f in done:
            j = pend.pop(f)
            try:
                r = f.result()
            except Exception:  # noqa: BLE001   worker 被殺掉（例如記憶體不足）：整個 pool 壞了，記下來、停止（已完成的工作都已寫檔，可以續跑）
                say(f"worker 例外（{j['id']}）：\n{traceback.format_exc()}")
                nfail = max_fail
                pend.clear()
                break
            nd += 1
            if r["status"] == "FAILED":
                nfail += 1
                say(f"FAILED {r['job_id']}（{r['slice']}）\n{r['error']}")
            elif r["status"] == NOT:
                say(f"NOT_MEASURED {r['job_id']}: {r['error']}")
            elif nd % 25 == 0 or len(todo) < 50:
                say(f"完成 {nd}/{len(todo)}  最近：{r['job_id']} {r['elapsed_s']:.0f}s  MemAvailable {mem_avail_gb():.0f} GB")
        if nfail >= max_fail and not pend:
            say(f"失敗 {nfail} 個，不再開新工作（--max-failures {max_fail}）")
            break
    ex.shutdown()
    say(f"結束：本次跑了 {nd} 個工作，失敗 {nfail}，用時 {time.time() - t0:.0f}s")
    return nfail


def collect(out, jobs_glob=None):
    """把 jobs/*/*.json 合成 out/map_cells.csv（一列＝一個格子）；FAILED／NOT_MEASURED 另列在 map_jobs_status.csv。"""
    rows, status = [], []
    for root, _, files in os.walk(os.path.join(out, "jobs")):
        for fn in sorted(files):
            if not fn.endswith(".json"):
                continue
            d = json.load(open(os.path.join(root, fn), encoding="utf-8"))
            status.append(dict(job_id=d["job_id"], slice=d["slice"], stage=d["stage"], comp=d["comp"], status=d["status"],
                               run_id=d["run_id"], ts=d["ts"], elapsed_s=d.get("elapsed_s"), error=(d.get("error") or "")[-2000:]))
            rows += d.get("rows", [])
    keys = []
    seen = set()
    for r in rows:
        for c in r:
            if c not in seen:
                seen.add(c)
                keys.append(c)
    front = ["run_id", "ts", "stage", "comp", "slice", "trace_group", "source_group", "synthetic", "semi_synthetic", "overlay", "engine",
             "outer_code", "T_s", "mem_GiB", "model", "policy_label", "h_tag", "view_group", "is_primary", "k_gpus", "job_id", "npz_path"]
    keys = [c for c in front if c in seen] + [c for c in keys if c not in front]
    p = os.path.join(out, "map_cells.csv")
    with open(p + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    os.replace(p + ".tmp", p)
    with open(os.path.join(out, "map_jobs_status.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(status[0]) if status else ["job_id"])
        w.writeheader()
        w.writerows(status)
    return len(rows), len(status)


# ------------------------------------------------------------------ dry-run
def dry_run(stages, slices, fill, skip, procs, Hs, n_req, kall):
    timing = load_timing()
    print(f"〔估計〕每請求秒數來自計時 run 20261018-102610-p3sim-timing3（06 §16.1）；{procs} 個 process；沒有實際跑任何模擬")
    grand = 0.0
    out = []
    for st in stages:
        jobs = enumerate_jobs(st, slices, fill, skip, kall)
        byc = {}
        for j in jobs:
            c = byc.setdefault(j["comp"], dict(jobs=0, rows=0, cpu_s=0.0))
            c["jobs"] += 1
            c["rows"] += n_rows_of(j)
            c["cpu_s"] += est_cost_s(j, timing, n_req)
        tot = sum(c["cpu_s"] for c in byc.values())
        grand += tot
        print(f"== 階段 {st}（切片 {len(set(j['slice'] for j in jobs))} 個：{sorted(set(j['slice'] for j in jobs))[:20]}）")
        for comp, c in byc.items():
            print(f"   {comp:10s} 工作 {c['jobs']:6d}  格子（列）{c['rows']:7d}  〔估計〕{c['cpu_s'] / 3600:9.1f} CPU 小時")
            out.append(dict(stage=st, comp=comp, jobs=c["jobs"], rows=c["rows"], est_cpu_hours=c["cpu_s"] / 3600))
        print(f"   合計 工作 {len(jobs)}  格子 {sum(c['rows'] for c in byc.values())}  〔估計〕{tot / 3600:.1f} CPU 小時 ≈ {tot / 3600 / procs:.1f} 小時（{procs} 個 process）")
        byl = {}
        for j in jobs:
            byl[j["slice"]] = byl.get(j["slice"], 0.0) + est_cost_s(j, timing, n_req) / 3600
        print("   逐切片 CPU 小時：" + "；".join(f"{s.split('/')[-1]} {h:.1f}" for s, h in sorted(byl.items(), key=lambda x: -x[1])[:20]))
    print(f"全部階段合計〔估計〕{grand / 3600:.1f} CPU 小時 ≈ {grand / 3600 / procs:.1f} 小時")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", choices=list(STAGES), default=None)
    ap.add_argument("--c0-util", action="store_true", help="凍結前的例外（第三輪審查 #5）：只跑 C0（階段 A 的切片、主視角、H），輸出 SSD 使用率 CSV；不需要 --frozen，也不產生任何其他輸出")
    ap.add_argument("--c0-util-csv", default=os.path.join(REPO, "results", "p3_sim", "c0_ssd_util.csv"))
    ap.add_argument("--recompute-charge", choices=list(RECOMPUTE_CHARGES), default=None,
                    help="合成資料／測試用；真實資料一律只從凍結的 04 §9 讀（`重算計價：…`），給了就拒絕")
    ap.add_argument("--slices", nargs="*", default=None, help="限制切片（例如 synth/small-seed1；真實切片需要 --frozen）")
    ap.add_argument("--out", default=None, help=f"輸出資料夾（預設 {DEFAULT_OUT}/<frozen 前 8 碼｜synthetic>）")
    ap.add_argument("--frozen", default=None, help="04 凍結的 commit；沒給就拒跑真實資料")
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--warm-procs", type=int, default=2, help="開 worker 之前，同時建 P 快取的 process 數（每個建置峰值最高約 34 GB，FreeInference week13 實測）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="工作 id 含任一個子字串才跑（測試用）")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 個工作（列舉順序；測試用）")
    ap.add_argument("--skip", nargs="*", default=[], help="略過的元件（a1_sess、a1_blk、a2_mem、a2_blk、sens_qwen、sens_70b、sens_ov2、sens_ov4）")
    ap.add_argument("--fill", action="store_true", help="A2 另補：三個記憶體點上的 E1、E5（5 分、1 小時）")
    ap.add_argument("--no-sens-views", action="store_true", help="A1 的 session 工作不加 §5 敏感度的 4 個時間設定")
    ap.add_argument("--min-free-gb", type=float, default=30.0, help="MemAvailable 低於這個值就不開新工作")
    ap.add_argument("--max-failures", type=int, default=3)
    ap.add_argument("--no-retry-failed", action="store_true")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--fmt", default=MAP_FMT, help="格式表：official｜official:loadonly_k0｜official_h200｜檔案路徑（地圖用預設；預設跟著 P3_MAP_PLATFORM）")
    a = ap.parse_args(argv)

    if not a.stage and not a.c0_util:
        ap.error("要給 --stage 或 --c0-util")
    Hs, n_req = load_H()
    # 凍結前的 H 檢查：讀目前的 04（或凍結的版本）
    frozen_full, frozen_sha = (None, None)
    if a.frozen:
        frozen_full, frozen_sha = check_frozen(a.frozen)
    text = prereg_text(a.frozen)
    check_H_against_doc(Hs, text)
    for s in SLICES:
        assert s in Hs, f"return_gaps.csv 沒有 {s} 的 H"
    kall = load_kall() if os.path.exists(KALL) else None
    if kall is None and not a.dry_run:
        raise SystemExit(f"{KALL} 不存在（先用 stats_v4.py kall 產生）")
    if a.dry_run:
        dry_run(a.stage or [], a.slices, a.fill, a.skip, a.procs, Hs, n_req, None)       # 個數與估計不需要 k
        return 0

    jobs = []
    if a.c0_util:
        if a.stage:
            ap.error("--c0-util 不能和 --stage 一起用")
        jobs = c0util_jobs(a.slices, kall)
    for st in (a.stage or []):
        jobs += enumerate_jobs(st, a.slices, a.fill, a.skip, kall)
    real = sorted({j["slice"] for j in jobs if not is_synth(j["slice"])})
    if real and not a.frozen and not a.c0_util:
        raise SystemExit(f"拒絕：這些真實資料的切片需要 --frozen <04 凍結的 commit>：{real[:6]}…（合成資料不需要）")
    if a.only:
        jobs = [j for j in jobs if any(o in j["id"] or o in j["slice"] for o in a.only)]
    if a.limit:
        jobs = jobs[:a.limit]
    lock = None
    rc_param = a.recompute_charge or "none"
    if a.c0_util and (not a.out or os.environ.get("RUN_ID", "adhoc") in ("", "adhoc")):
        raise SystemExit("--c0-util 要明確的 --out 與 run_id（用 `bash code/run3090.sh <短名> …` 跑）")
    if real and a.frozen and not a.c0_util:
        # 真實資料的判定參數只從凍結的 04 讀（第三輪審查 #2）；命令列不准覆蓋
        if a.recompute_charge is not None:
            raise SystemExit("拒絕：真實資料的重算計價只從凍結的 04 §9 讀（`重算計價：…`），不准用 --recompute-charge 覆蓋")
    if real and not a.collect_only and not a.c0_util:
        # 凍結閘門（第二輪審查 #5）：輸入檔的 sha256、程式 commit、乾淨的工作目錄、明確的 run_id；任何一項不成立就拒絕
        lock = lockfile.verify_freeze(a.frozen, REPO)
        if os.environ.get("RUN_ID", "adhoc") in ("", "adhoc"):
            raise SystemExit("拒絕：真實資料必須有明確的 run_id（用 `bash code/run3090.sh <短名> …` 跑，它會設 RUN_ID；不准用預設的 \"adhoc\"）")
        if a.fmt != MAP_FMT:
            raise SystemExit(f"拒絕：真實資料只准用正式格式表（--fmt {MAP_FMT}），不是 {a.fmt!r}")
        rc_param = lock["recompute_charge"]            # verify_freeze 保證三個判定參數都在凍結的 04 裡
    out = a.out or os.path.join(DEFAULT_OUT, (frozen_full[:8] if frozen_full else "synthetic"))
    os.makedirs(out, exist_ok=True)
    code_commit = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
    ctx = dict(H=Hs, fmt=a.fmt, frozen_commit=frozen_full or "", code_commit=code_commit + ("+dirty" if dirty else ""),
               sens_views=not a.no_sens_views, kall=kall, recompute_charge=rc_param)
    json.dump(dict(stages=a.stage, slices=a.slices, frozen=frozen_full, prereg_sha256=frozen_sha, fmt=a.fmt, procs=a.procs, lock=lock,
                   recompute_charge=rc_param, c0_util=a.c0_util,
                   code_commit=ctx["code_commit"], run_id=os.environ.get("RUN_ID", "adhoc"), ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   args=vars(a), n_jobs=len(jobs)), open(os.path.join(out, f"invocation_{time.strftime('%Y%m%d-%H%M%S')}.json"), "w"),
              default=str, indent=1)
    if a.collect_only:
        print("collect:", collect(out))
        return 0
    timing = load_timing()
    todo, done, cleaned = classify(out, jobs, not a.no_retry_failed, ctx)
    print(f"工作 {len(jobs)}：已完成 {len(done)}、要跑 {len(todo)}、清掉的殘檔 {len(cleaned)}；輸出 {out}", flush=True)
    slc_cost = {}
    for j in todo:
        slc_cost[j["slice"]] = slc_cost.get(j["slice"], 0.0) + est_cost_s(j, timing, n_req)
    todo.sort(key=lambda j: (-slc_cost[j["slice"]], j["slice"], j["overlay"], -est_cost_s(j, timing, n_req)))   # 同切片（同疊加）排在一起（worker 的 P 快取），大的先
    warm_caches(todo, a.warm_procs, out)
    nfail = dispatch(todo, ctx, a.procs, out, a.min_free_gb, a.max_failures)
    if a.c0_util:
        if nfail:
            return 1
        n = collect_c0util(out, a.c0_util_csv)
        print(f"c0-util：{n} 列 → {a.c0_util_csv}", flush=True)
        return 0
    nrow, nj = collect(out)
    print(f"collect：{nrow} 列、{nj} 個工作 → {out}/map_cells.csv", flush=True)
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
