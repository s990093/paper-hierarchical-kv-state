"""analyze_map.py — 地圖的判定與報告（04_prereg_map.md v4 §4、§6、§10；06_simulator.md §21）。

輸入：run_map.py 的 map_cells.csv（一列＝一個格子）與每個工作的 per-session npz。
主指標（每個格子、每個做法）＝SSD 寫入（決定時刻 < 終點 − H）＋ SSD 讀取（到達 < 終點 − H 的請求）。讀取（第三輪審查 #4，兩個引擎同一個定義）＝只載入的還原讀取
（lo_bytes）＋全部預取讀取（含浪費；session 引擎沒有預取）＝ssd_H_primread_GiB（npz：primread_bytes_H[0]）；`重算計價：pass_size`（04 §9）時再加
重算計價 ssd_H_charged_recompute_GiB（npz：recompute_charge_bytes_H；歸屬於讀它的 session）。2H 版另報。
判定用的 O＝04 §9 的 `完美預測：greedy|tier`（一般的 O｜O-tier）；另一個以 report_oracle_alt 報告。
真實資料的凍結閘門（第三輪審查 #2）：一律要 --frozen、過 lockfile.verify_freeze；護欄的參考、完美預測、重算計價只從凍結的 04 讀（缺任何一行就拒絕）；
n_boot 固定 1000；每一列的 frozen_commit／code_commit 唯一、等於鎖定值（或其無程式變動的後代）、不帶 +dirty；run_id 不能是 adhoc；重複的格子丟 DuplicateCell。
等待時間護欄（第二輪審查 #6、#10、#12）：用截尾版 ttftq_H_*（2H 版：非 oracle 用 ttftq_H2_*）；一個做法「合格」＝它的回來請求平均與 p99 都 ≤ 1.05 × ref；
  ref＝C0（guard_ref="C0"）或 C0 與 O 逐項取小（guard_ref="min_C0_O"）。判定用 04 §9 凍結的那個（未凍結＝C0，輸出裡寫明）；兩種都報告。NOT_MEASURED → 當成通過、標出。
  O 自己也要過護欄（≤ 1.05 × C0）；沒過的格子不能支持 GO。
最好的對手＝所有非 oracle 的做法（含參數）裡合格、主指標位元組最少的那個（完整樣本上選、bootstrap 固定）。
省多少＝(最好的對手 − O) ÷ 最好的對手；最好的對手＝0 而 O > 0 → NaN（旗標 best_zero_o_pos），不是 0%；0÷0 → 0（旗標 best_zero）。
SSD 使用率＝寫入量÷(寫入頻寬×T×k)＋讀取量÷(讀取頻寬×T×k)（主視角、SSD 1 張專用、T＝切片窗口截到終點 − H、k＝這個切片這個記憶體的 E6 的 k，所有引擎同一個；
  第三輪審查 #5）。**閘門看 C0 的使用率**：C0 <5% 的格子是「SSD 不忙」，GO、NO-GO 都不算；護欄輸入（C0、O、所有對手的等待時間）有 NOT_MEASURED 的格子也一樣不算（#7）。
  每格仍報告最好的對手與 O 的 GiB／小時與使用率、兩者的重算 token、最好的對手是不是 C0。session 配對 bootstrap 1,000 次（固定 seed；FreeInference 各週當一份資料、週內分層重抽）；
15 個主格（E2、E3、E4、E5(30 分)、E6 × 63／170／435 GiB）做 studentized max-T 同時 CI：
  每格：點估計 θ̂、bootstrap 值 θ*、SE＝sd(θ*)；T_b＝max_c |θ*_cb − θ̂_c| / SE_c；q＝T_b 的 95 百分位；CI＝θ̂ ± q·SE。
0/0：最好的對手＝0 且 O＝0 → 省多少定為 0（旗標 best_zero；bootstrap 裡 best*＝0 也當 0）；SE＝0 的格子不進 max-T（CI 退化成點）。
切片齊全（#4）：每種模式寫死應有的切片（A1／early＝A1 的 8 個切片；full＝再加 FreeInference 其餘 12 週；synthetic＝合成切片）與每個切片的 15 個主格 × 所有做法；
  少一個、多一個、或做法缺格 → IncompleteMap，不給判定。
判定（§6）：GO＝至少一個主格，某份資料的 CI 下界 ≥15%，而且不同來源的另一份資料在同一格的 CI 下界 ≥5%；
  NO-GO＝所有進判定的真實資料、所有主格 CI 上界 <5%（而且每份都完整）；其他＝ASK。
  有效 session 數（1／Σp²）<100 的資料不進判定（FreeInference 用全部週合起來）。來源：{Bailian coder, traceA}、{Mooncake}、{FreeInference}、{WildChat}。
  §10 的 A1 提早 GO：不含 FreeInference 的資料已滿足 GO → 定案 GO；否則 PENDING_B（等 B 的 A1 範圍；用 --mode early／full）。
只報告（永遠不讓判定變 GO）：E1、O-loose、拆解（O−O-len、O−O-reuse、O-len 對 C4r、O-reuse 對 D-idle）、2H、敏感度（CPU 忙、3090、Qwen、70B、疊加×2／×4、E2 block）、合成。
寬鬆版雙胞胎（#3a，只報告）：O 與所有對手都用寬鬆品質規則（rule=loose 的工作）重算最好的對手與省多少。每份資料報告 >128K 的請求比例＝stats_v4.py long128k（資料統計）。
用法：python code/p3_sim/analyze_map.py --cells <out>/map_cells.csv --mode early|full [--synthetic] [--guard-ref auto|C0|min_C0_O] [--frozen <commit>] --out-prefix <path 前綴>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import lockfile  # noqa: E402

GiB = 1 << 30
NOT = "NOT_MEASURED"
SEED = 20261018
N_BOOT = 1000
GUARD = 1.05
GUARD_SENS = [("1.05", 1.05), ("1.10", 1.10), ("1.20", 1.20), ("none", None)]       # 04 §4：護欄敏感度（只報告）；1.05 是登記的、判定用的那個
GUARD_REFS = ("C0", "min_C0_O")
GO_STRONG, GO_WEAK, NOGO = 0.15, 0.05, 0.05
UTIL_MIN = 0.05                                                                      # SSD 忙的門檻：C0 的使用率 <5% 的格子，GO、NO-GO 都不算（第三輪審查 #5）
BLOCK_TOKENS = 512
MIN_EFF_SESSIONS = 100.0
TRACES = ["Bailian coder", "Bailian traceA", "Mooncake", "WildChat", "FreeInference"]
SOURCE = {"Bailian coder": "alibaba", "Bailian traceA": "alibaba", "Mooncake": "mooncake", "WildChat": "wildchat", "FreeInference": "freeinference"}
MEMS = (63.0, 170.0, 435.0)
PRIMARY_OUTERS = [("E2", "session"), ("E3", "session"), ("E4", "session"), ("E5-T1800", "session"), ("E6", "block")]
PRIMARY_CELLS = [(o, m) for o, _ in PRIMARY_OUTERS for m in MEMS]            # 15 個主格
ENGINE_OF = dict(PRIMARY_OUTERS)
PRIMARY_VIEW = "mi300x|cake|gpu=idle|cpu=idle|ssd/1|mi300x_local"            # 主視角（run_map._view_name(PRIMARY_SPEC)）
if os.environ.get("P3_MAP_PLATFORM", "mi300x") != "mi300x":                    # 2026-10-19：H200 的地圖，主視角換成那個平台（run_map 的 MAP_PLAT）
    PRIMARY_VIEW = PRIMARY_VIEW.replace("mi300x|", os.environ["P3_MAP_PLATFORM"] + "|", 1)
# 每種模式應有的切片（04 §10；第二輪審查 #4）：寫死在這裡，不從資料推；少一個、多一個都丟錯
FI_A = [f"freeinference/week{w:02d}" for w in (1, 5, 9, 13)]
FI_ALL = [f"freeinference/week{w:02d}" for w in range(1, 17)]
A1_EXPECTED = {"Bailian coder": ["bailian/coder"], "Bailian traceA": ["bailian/traceA"], "WildChat": ["wildchat/1m"],
               "Mooncake": ["mooncake/conv-rule-strict"], "FreeInference": FI_A}
EXPECTED = {"early": A1_EXPECTED, "full": dict(A1_EXPECTED, FreeInference=FI_ALL), "synthetic": {"synth": ["synth/small-seed1"]}}
USECOLS = ["run_id", "stage", "comp", "slice", "trace_group", "source_group", "synthetic", "semi_synthetic", "overlay", "engine", "outer_code",
           "T_s", "mem_GiB", "model", "policy_label", "policy_family", "oracle", "h_tag", "view_group", "view_idx", "is_primary", "k_gpus", "job_id",
           "view_name", "quality_rule", "npz_path", "npz_sfx_H", "npz_sfx_2H", "prim_H_GiB", "prim_2H_GiB", "prim_w_H_GiB", "prim_r_H_GiB",
           "prim_w_2H_GiB", "prim_r_2H_GiB", "T_H_s", "T_2H_s", "ssd_bw_r_GiBps", "ssd_bw_w_GiBps", "prim_read_col",
           "k_util", "prim_rbase_H_GiB", "prim_rc_H_GiB", "prim_rbase_2H_GiB", "prim_rc_2H_GiB", "recompute_charge", "frozen_commit", "code_commit",
           "otier_cpu_event_mismatch_requests", "otier_approx",
           "ttftq_mean_s", "ttftq_p99_s", "ttftq_H_mean_s", "ttftq_H_p99_s", "ttftq_H2_mean_s", "ttftq_H2_p99_s", "n_ret", "surprise_events",
           "surprise_blocks", "reads_fail", "reads_unverified", "ret_recompute_prefix_blocks", "recompute_prefix_blocks", "ssd_write_GiB",
           "ssd_loadonly_read_GiB", "n_req", "n_sess"]
REQUIRED_COLS = ["view_name", "quality_rule", "ttftq_H_mean_s", "ttftq_H_p99_s", "prim_w_H_GiB", "prim_r_H_GiB", "T_H_s", "ssd_bw_r_GiBps",
                 "ssd_bw_w_GiBps", "recompute_prefix_blocks", "k_gpus", "k_util", "prim_rbase_H_GiB", "prim_rc_H_GiB"]


class IncompleteMap(Exception):
    """地圖不齊（少切片、多切片、缺格）：不給判定（第二輪審查 #4）。"""


def olabel(code, T):
    """外層做法的標籤：E5 帶 T（秒）。"""
    if code == "E5":
        return f"E5-T{int(float(T))}"
    return code


# ====================================================================== 讀資料
def load_cells(path):
    head = pd.read_csv(path, nrows=0).columns
    miss = [c for c in REQUIRED_COLS if c not in head]
    if miss:
        raise SystemExit(f"{path}: 缺必要的欄 {miss}（用新版 run_map.py 重跑）")
    cols = [c for c in USECOLS if c in head]
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    df["olabel"] = [olabel(c, t) if str(t) not in ("", "nan") else c for c, t in zip(df.outer_code, df["T_s"])]
    for c in ("is_primary", "oracle", "synthetic", "semi_synthetic"):
        if c in df:
            df[c] = df[c].astype(str).str.lower().isin(("true", "1"))
    return df


ORACLE_LABELS = ("O", "O-tier", "O-len", "O-reuse", "O-loose")
ORACLE_VARIANT_LABEL = {"greedy": "O", "tier": "O-tier"}       # 04 §9 的 `完美預測：` → 判定用的「O」是哪個做法；另一個（O_alt）只報告
RECOMPUTE_CHARGES = ("none", "pass_size")
ORACLE_VARIANTS = ("greedy", "tier")
AM_NM = NOT


class DuplicateCell(Exception):
    """同一個格子（切片×外層×記憶體×做法×視角×H）在 map_cells.csv 出現兩列：不准默默取第一列（第三輪審查 #2）。"""


class Index:
    """(trace, slice, engine, olabel, mem, model, overlay, view_group, 視角名稱, 品質規則, policy_label, tag) → 一列。
    視角名稱在鍵裡（第二輪審查 #11）：非主視角的列（例如 E6 的 GPU 忙、session 引擎的 7 個其他時間設定）不再被丟掉。"""

    def __init__(self, df):
        self.rows = {}
        self.slices = {}
        for r in df.to_dict("records"):
            key = (r["trace_group"], r["slice"], r["engine"], r["olabel"], float(r["mem_GiB"]), r["model"], int(r["overlay"]), r["view_group"],
                   r["view_name"], r["quality_rule"], r["policy_label"], r["h_tag"])
            if key in self.rows:
                raise DuplicateCell(f"重複的格子（不准默默取第一列）：{key}；兩列的 job_id：{self.rows[key]['job_id']}、{r['job_id']}")
            self.rows[key] = r
            if r["quality_rule"] == "strict":
                self.slices.setdefault(r["trace_group"], set()).add(r["slice"])
        self.views = sorted({(k[0], k[2], k[3], k[4], k[5], k[6], k[7], k[8]) for k in self.rows if k[9] == "strict"})

    def row(self, trace, slc, engine, ol, mem, label, tag, model="llama", overlay=1, vg="main", view=PRIMARY_VIEW, rule="strict"):
        """非 oracle＝tag 'both'；oracle＝自己的 H／2H 工作。"""
        for t in ((tag,) if label in ORACLE_LABELS else ("both",)):
            r = self.rows.get((trace, slc, engine, ol, float(mem), model, overlay, vg, view, rule, label, t))
            if r is not None:
                return r
        return None


def hcols(r, h):
    """這一列在 H（'H'）或 2H 版本的欄位名：(位元組欄尾巴, 護欄的 ttft 欄前綴, npz 的 ttft 前綴)。
    oracle 的 H／2H 是各自的工作（自己的 H 都叫 ttftq_H_*、tH）；非 oracle 是一次模擬兩個截尾（2H＝ttftq_H2_*、tH2）。"""
    if h == "H":
        return "H", "ttftq_H", "tH"
    return ("2H", "ttftq_H", "tH") if bool(r["oracle"]) else ("2H", "ttftq_H2", "tH2")


def hbytes(r, h, charge="none"):
    """這一列在 H（'H'）或 2H 版本的主指標（GiB）＝寫入＋只載入的還原讀取＋全部預取讀取（＋重算計價，charge＝pass_size 時）。
    從分量算（prim_w／prim_rbase／prim_rc），不依賴執行器當時的開關；並驗算「列上的 prim_*_GiB」在執行器的開關下是同一個數。"""
    w, rb, rc = (float(r[f"prim_w_{h}_GiB"]), float(r[f"prim_rbase_{h}_GiB"]), float(r[f"prim_rc_{h}_GiB"]))
    run_on = str(r.get("recompute_charge", "none")) == "pass_size"
    got = float(r["prim_H_GiB"] if h == "H" else r["prim_2H_GiB"])
    assert abs(got - (w + rb + (rc if run_on else 0.0))) <= 1e-6 * max(1.0, abs(got)), f"{r['job_id']}: prim_{h}_GiB 不等於分量加總"
    return w + rb + (rc if charge == "pass_size" else 0.0)


def sess_bytes(r, h, charge="none"):
    """per-session 的主指標位元組（依原始 session 編號排序）；並用 CSV 的列驗算總和。回傳 (sess_orig, x float 位元組)。
    讀取分量（兩個引擎同一個定義，第三輪審查 #4）：primread_bytes＝只載入的還原讀取＋全部預取讀取；charge＝pass_size 時加 recompute_charge_bytes（歸屬於讀它的 session）。"""
    z = np.load(r["npz_path"])
    sfx = r["npz_sfx_H"] if h == "H" else r["npz_sfx_2H"]
    assert isinstance(sfx, str) and sfx, f"{r['job_id']}: 沒有 {h} 版的 npz 欄位"
    rd = z["primread_bytes" + sfx][0].astype(np.int64)
    if charge == "pass_size":
        rd = rd + z["recompute_charge_bytes" + sfx].astype(np.int64)
    x = (z["ssd_write_bytes" + sfx].astype(np.int64) + rd)
    so = z["sess_orig"]
    order = np.argsort(so, kind="stable")
    tot = float(x.sum()) / GiB
    want = hbytes(r, h, charge)
    assert abs(tot - want) <= 1e-6 * max(1.0, abs(want)), f"{r['job_id']}: npz 總和 {tot} GiB 和 CSV 的 {want} 不符"
    return so[order], x[order].astype(np.float64)


# ====================================================================== 護欄
def hist_pct(hist, edges, q):
    """直方圖的百分位（格內線性內插）。"""
    n = hist.sum()
    if n == 0:
        return float("nan")
    cum = np.cumsum(hist)
    target = q / 100.0 * n
    i = int(np.searchsorted(cum, target, side="left"))
    i = min(i, len(hist) - 1)
    prev = cum[i - 1] if i > 0 else 0
    frac = (target - prev) / max(hist[i], 1)
    return float(edges[i] * (edges[i + 1] / edges[i]) ** min(max(frac, 0.0), 1.0))


def _num(x):
    if isinstance(x, str):
        return None if x == NOT else float(x)
    x = float(x)
    return x


_TTFT_POOL_CACHE = {}


class NoPooledTtft(Exception):
    """幾個切片合起來的護欄需要逐請求的 ttft，但這一列不是 npz 存的那個視角（npz 只存工作的第 0 個視角）→ 這個視角不報告。"""


def _ttft_npz(path, pfx):
    z = np.load(path)
    if f"{pfx}_vals" not in z.files:
        raise SystemExit(f"{path}: 沒有逐請求的 {pfx}_vals，無法精確合併幾週的護欄 p99（04 §4）；請用新版 run_map.py 重跑這個工作")
    return z[f"{pfx}_vals"].astype(np.float64), int(z[f"{pfx}_n"]), float(z[f"{pfx}_sum"]), int(z[f"{pfx}_nm"])


def trace_ttft(rows, h="H"):
    """一份資料（一個格子、一個做法）的回來請求等待時間（截尾版）：(mean, p99, nm)。
    一個切片＝用列的精確值（ttftq_H_*／ttftq_H2_*）；FreeInference 幾週＝用 npz 的逐請求值串起來（tH_vals／tH2_vals），p99＝np.percentile（精確），
    平均＝總和÷個數（精確）。nm＝True：有 NOT_MEASURED。只有工作的第 0 個視角存了 npz → 幾週合併時其他視角丟 NoPooledTtft。"""
    _, ttc, pfx = hcols(rows[0], h)
    if len(rows) == 1:
        r = rows[0]
        m, p = _num(r[ttc + "_mean_s"]), _num(r[ttc + "_p99_s"])
        return (None, None, True) if (m is None or p is None) else (m, p, False)
    if any(int(r["view_idx"]) != 0 for r in rows):
        raise NoPooledTtft(rows[0]["job_id"])
    key = (pfx,) + tuple(r["npz_path"] for r in rows)
    if key in _TTFT_POOL_CACHE:
        return _TTFT_POOL_CACHE[key]
    xs = []
    n, s, nm = 0, 0.0, 0
    for r in rows:
        v_, nn_, ss_, nm_ = _ttft_npz(r["npz_path"], pfx)
        assert len(v_) == nn_, f"{r['npz_path']}: {pfx}_vals 個數 {len(v_)} ≠ {pfx}_n {nn_}"
        xs.append(v_)
        n += nn_
        s += ss_
        nm += nm_
    if nm:
        out = (None, None, True)
    elif n == 0:
        out = (float("nan"), float("nan"), False)
    else:
        out = (s / n, float(np.percentile(np.concatenate(xs), 99)), False)
    _TTFT_POOL_CACHE[key] = out
    return out


def guard_status(pol, ref, thr=GUARD):
    """回傳 (通過?, NM 旗標)。pol／ref＝(mean, p99, nm)。NOT_MEASURED（做法的或 ref 的）→ 通過並標出；
    平均或 p99 是 NaN（沒有回來的請求）→ 通過（沒有東西可比）。"""
    if pol[2] or ref[2]:
        return True, True
    for a, b in ((pol[0], ref[0]), (pol[1], ref[1])):
        if a is None or b is None or math.isnan(a) or math.isnan(b):
            continue
        if a > thr * b + 1e-12:
            return False, False
    return True, False


def guard_ref_value(res, ref):
    """護欄的參考（第二輪審查 #6 之後使用者決定的兩種）：'C0'＝C0 的 (mean, p99, nm)；'min_C0_O'＝C0 與 O 逐項取小（NaN 當作沒有）；任一個是 NM → NM。"""
    c0 = res["C0"]["ttft"]
    if ref == "C0":
        return c0
    assert ref == "min_C0_O", ref
    o = res["O"]["ttft"]
    if c0[2] or o[2]:
        return (None, None, True)

    def mn(a, b):
        if a is None or math.isnan(a):
            return b
        if b is None or math.isnan(b):
            return a
        return min(a, b)
    return (mn(c0[0], o[0]), mn(c0[1], o[1]), False)


# ====================================================================== 一個格子（一份資料）
LABELS = ["C0", "C1", "C1-8", "C2", "C3", "C6", "C1-2", "C4-8192", "C4-16384", "C4-32768", "C4r-8192", "C4r-16384", "C4r-32768", "D-len-8192",
          "D-len-16384", "D-len-32768", "D-idle-300", "D-idle-1800", "D-idle-3600", "D-idle-14400", "R2", "O", "O-tier", "O-len", "O-reuse", "O-loose"]
NON_ORACLE = [x for x in LABELS if x not in ORACLE_LABELS]
NEEDED = NON_ORACLE + ["O", "O-tier"]            # 資料裡必須有的做法（兩個 O 都要：判定用哪個由 04 決定，另一個報告）
NEEDED_ROLE = NON_ORACLE + ["O", "O_alt"]       # trace_cell 回傳的 res 的鍵：O＝判定用的、O_alt＝另一個（只報告）


def trace_cell(ix, trace, slices, engine, ol, mem, h, model="llama", overlay=1, vg="main", labels=None, view=PRIMARY_VIEW, rule="strict",
               ovar="greedy", charge="none"):
    """一份資料（可能是好幾個切片）在一個格子裡，每個做法的主指標總和（GiB）、等待時間、SSD 使用率、重算 token。
    缺任何一個切片的某個做法 → 該做法不在 res 裡。
    SSD 使用率＝Σ_切片(寫入量÷寫入頻寬＋讀取量÷讀取頻寬) ÷ Σ_切片(T×k)（T＝窗口截到終點 − H；k＝這個切片這個記憶體的 E6 的 k，kgpu_kmin_all.csv；
    所有引擎同一個 k，第三輪審查 #5）；GiB／小時＝位元組÷ΣT×3600。主指標含重算計價與否由 charge 決定（04 §9 的 `重算計價：`）。
    ovar（04 §9 的 `完美預測：`）：res["O"]＝判定用的 O（greedy→一般的 O；tier→O-tier），res["O_alt"]＝另一個（只報告）；各帶 label。判定用的那個缺 → res 沒有 "O"。"""
    labels = labels or [p for p in LABELS]
    res = {}
    for lab in labels:
        rows = [ix.row(trace, s, engine, ol, mem, lab, h, model, overlay, vg, view, rule) for s in slices]
        if any(r is None for r in rows):
            continue
        try:
            tt = trace_ttft(rows, h)
        except NoPooledTtft:
            continue
        sfx = hcols(rows[0], h)[0]
        num = den = Tsum = 0.0
        for r in rows:
            w, T = float(r[f"prim_w_{sfx}_GiB"]), float(r[f"T_{sfx}_s"])
            rd = float(r[f"prim_rbase_{sfx}_GiB"]) + (float(r[f"prim_rc_{sfx}_GiB"]) if charge == "pass_size" else 0.0)
            k = float(r["k_util"])
            assert k >= 1 and k == int(k), f"{r['job_id']}: k_util={k}"
            num += w / float(r["ssd_bw_w_GiBps"]) + rd / float(r["ssd_bw_r_GiBps"])
            den += T * int(k)
            Tsum += T
        gib = sum(hbytes(r, h, charge) for r in rows)
        res[lab] = dict(rows=rows, GiB=gib, ttft=tt, util=(num / den if den > 0 else float("nan")), gph=(gib / (Tsum / 3600.0) if Tsum > 0 else float("nan")),
                        rec_tokens=int(sum(int(r["recompute_prefix_blocks"]) for r in rows)) * BLOCK_TOKENS, label=lab,
                        charged_GiB=sum(float(r[f"prim_rc_{sfx}_GiB"]) for r in rows))
    chosen, other = ORACLE_VARIANT_LABEL[ovar], ORACLE_VARIANT_LABEL["tier" if ovar == "greedy" else "greedy"]
    out = {k: v for k, v in res.items() if k not in ("O", "O-tier")}
    if chosen in res:
        out["O"] = res[chosen]
    if other in res:
        out["O_alt"] = res[other]
    return out


def best_of(res, candidates=None, guard=True, thr=GUARD, ref="C0"):
    """最好的對手：非 oracle（或給定的 candidates）裡合格（ttft ≤ thr × ref）、主指標最少的。
    回傳 dict(label, GiB, excluded, nm, passed)；沒有任何一個 → label None。"""
    cands = [c for c in (candidates or NON_ORACLE) if c in res]
    out = dict(label=None, GiB=float("nan"), excluded=[], nm=[], passed=[], n_candidates=len(cands))
    if "C0" not in res or not cands or (ref == "min_C0_O" and guard and "O" not in res):
        return out
    rv = guard_ref_value(res, ref) if guard else None
    for c in cands:
        ok, nmf = guard_status(res[c]["ttft"], rv, thr) if guard else (True, False)
        if not ok:
            out["excluded"].append(c)
            continue
        if nmf:
            out["nm"].append(c)
        out["passed"].append(c)
    if out["passed"]:
        lab = min(out["passed"], key=lambda c: (res[c]["GiB"], LABELS.index(c)))
        out.update(label=lab, GiB=res[lab]["GiB"])
    return out


def savings_point(best, o):
    """(best − O)/best。回傳 (值, best_zero, nan_flag)：best＝0 且 O＝0 → (0, True, "")；best＝0 而 O＞0 → (NaN, True, "best_zero_o_pos")（不是 0%）；
    任一個 NaN → (NaN, False, "nan")。"""
    if best is None or o is None or math.isnan(best) or math.isnan(o):
        return float("nan"), False, "nan"
    if best == 0.0:
        return (float("nan"), True, "best_zero_o_pos") if o > 0 else (0.0, True, "")
    return (best - o) / best, False, ""


def guard_nm_inputs(res):
    """這個格子的護欄輸入裡有 NOT_MEASURED 的做法（C0、判定用的 O、所有對手的等待時間）；沒有 → []（第三輪審查 #7）。"""
    return sorted(c for c in ["C0", "O"] + [x for x in NON_ORACLE if x != "C0"] if c in res and res[c]["ttft"][2])


def eval_cell(res, ref="C0", thr=GUARD, guard=True):
    """一個格子在護欄參考 ref 下的結果：最好的對手、省多少（點）、O 過不過護欄、GO 的資格。
    進判定（evaluable）要求：C0 的 SSD 使用率 ≥5%（否則「SSD 不忙」，GO、NO-GO 都不算；第三輪審查 #5）、護欄的輸入沒有任何 NOT_MEASURED（#7）。
    GO 的資格 go_ok＝evaluable、有最好的對手、點估計有定義、O 過護欄。最好的對手的使用率照舊報告（util_best），不再當門檻。"""
    best = best_of(res, ref=ref, thr=thr, guard=guard)
    o = res["O"]
    og, onm = guard_status(o["ttft"], res["C0"]["ttft"], thr) if guard else (True, False)
    c0u = res["C0"]["util"]
    nm_in = guard_nm_inputs(res)
    busy = bool(not math.isnan(c0u) and c0u >= UTIL_MIN)
    d = dict(res=res, best=best, o_guard_ok=og, o_guard_nm=onm, nan_flag="", best_zero=False, point=float("nan"), util_best=float("nan"),
             util_O=o["util"], best_is_C0=False, best_rec=None, O_rec=o["rec_tokens"], best_gph=float("nan"), O_gph=o["gph"],
             util_C0=c0u, ssd_busy=busy, guard_nm_inputs=nm_in, evaluable=bool(busy and not nm_in),
             why_not=("SSD 不忙（C0 使用率 <5%）" if not busy else "") + (("；" if not busy else "") + f"護欄輸入有 NOT_MEASURED：{','.join(nm_in)}" if nm_in else ""))
    if best["label"] is None:
        d["nan_flag"] = "no_eligible_rival"
    else:
        pt, bz, nf = savings_point(best["GiB"], o["GiB"])
        b = res[best["label"]]
        d.update(point=pt, best_zero=bz, nan_flag=nf, util_best=b["util"], best_is_C0=best["label"] == "C0", best_rec=b["rec_tokens"], best_gph=b["gph"])
    d["go_ok"] = bool(best["label"] is not None and d["nan_flag"] == "" and og and d["evaluable"])
    return d


def cell_flags(d):
    """表上顯示的旗標（字串列表）。"""
    f = []
    if d["nan_flag"]:
        f.append({"best_zero_o_pos": "best=0、O>0（NaN）", "no_eligible_rival": "沒有合格的對手", "nan": "NaN"}.get(d["nan_flag"], d["nan_flag"]))
    elif d["best_zero"]:
        f.append("best=0")
    if not d["o_guard_ok"]:
        f.append("O 沒過護欄")
    if d["best"]["label"] is not None and d["best_is_C0"]:
        f.append("最好=C0")
    if not d["ssd_busy"]:
        f.append(f"SSD 不忙（C0 使用率 {fmt_pct(d['util_C0'])}<5%；GO、NO-GO 都不算）")
    if d["guard_nm_inputs"]:
        f.append("護欄輸入有 NOT_MEASURED（GO、NO-GO 都不算）")
    return f


# ====================================================================== bootstrap（paired、stratified、max-T）
def effective_sessions(n_req):
    n = np.asarray(n_req, float)
    p = n / n.sum()
    return float(1.0 / np.sum(p * p))


def bootstrap_maxT(strata, cells, n_boot=N_BOOT, seed=None):
    """strata：[(名字, 這個層的 session 數 S, {cell: (x_best (S,), x_O (S,) 或 None)})]，每層用同一組重抽（所有格子、所有做法 paired）。
    θ_c＝(Σ best − Σ O)/Σ best（Σ best＝0 → 0）。回傳 dict(theta_hat, theta_star(B,C), se, q, lo, hi)。"""
    C = len(cells)
    names = [s[0] for s in strata]
    mats = []
    for (_, S, d) in strata:
        X = np.zeros((S, 2 * C))
        for j, c in enumerate(cells):
            xb, xo = d[c]
            X[:, 2 * j] = xb
            X[:, 2 * j + 1] = xo
        mats.append(X)
    full = sum(X.sum(axis=0) for X in mats)

    def theta(tot):
        b, o = tot[0::2], tot[1::2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(b > 0, (b - o) / b, 0.0)
        return t
    th_hat = theta(full)
    rng = np.random.default_rng(seed)
    order = sorted(range(len(strata)), key=lambda i: names[i])
    th_star = np.zeros((n_boot, C))
    for b in range(n_boot):
        tot = np.zeros(2 * C)
        for i in order:
            S = strata[i][1]
            w = np.bincount(rng.integers(0, S, S), minlength=S).astype(np.float64)
            tot += w @ mats[i]
        th_star[b] = theta(tot)
    se = th_star.std(axis=0, ddof=1)
    ok = se > 0
    if ok.any():
        T = np.max(np.abs(th_star[:, ok] - th_hat[ok]) / se[ok], axis=1)
        q = float(np.quantile(T, 0.95))
    else:
        q = 0.0
    return dict(theta_hat=th_hat, theta_star=th_star, se=se, q=q, lo=th_hat - q * se, hi=th_hat + q * se)


# ====================================================================== 判定（§6、§10）
def decide(traces, mode="full"):
    """traces：{名字: dict(eligible, complete, synthetic, source, cells={cell: dict(lo, hi, point)})}；cell＝(外層, 記憶體)。
    mode＝'full'（§6 照用全部資料）｜'early'（§10：只看不含 FreeInference 的資料；GO→定案 GO，否則 PENDING_B）。
    只有 eligible、complete、非合成的資料進判定。回傳 dict(verdict, reason, go_cells, used)。"""
    use = {t: v for t, v in traces.items() if v["eligible"] and v["complete"] and not v.get("synthetic")
           and not (mode == "early" and v["source"] == "freeinference")}
    why_not = {t: ("合成" if v.get("synthetic") else "有效 session <100" if not v["eligible"] else "不完整" if not v["complete"] else "early 模式不含 FreeInference")
               for t, v in traces.items() if t not in use}
    go_cells = []
    cells = sorted({c for v in use.values() for c in v["cells"]})
    for c in cells:
        for t1, v1 in use.items():
            c1 = v1["cells"].get(c)
            if c1 is None or not c1.get("go_ok", True) or not c1["lo"] >= GO_STRONG:
                continue
            for t2, v2 in use.items():
                if v2["source"] == v1["source"]:
                    continue
                c2 = v2["cells"].get(c)
                if c2 is not None and c2.get("go_ok", True) and c2["lo"] >= GO_WEAK:
                    go_cells.append(dict(cell=c, strong=t1, weak=t2, strong_lo=c1["lo"], weak_lo=c2["lo"],
                                         strong_best_is_C0=bool(c1.get("best_is_C0", False)), weak_best_is_C0=bool(c2.get("best_is_C0", False)),
                                         strong_twin=c1.get("twin", float("nan")), weak_twin=c2.get("twin", float("nan"))))
    res = dict(used=sorted(use), not_used=why_not, go_cells=go_cells, mode=mode)
    if go_cells:
        g = go_cells[0]
        res.update(verdict="GO" if mode == "full" else "GO（早期，定案）",
                   reason=f"{g['cell'][0]}@{g['cell'][1]:g} GiB：{g['strong']} 的 CI 下界 {g['strong_lo']:.1%} ≥15%，不同來源的 {g['weak']} 的 CI 下界 {g['weak_lo']:.1%} ≥5%"
                          f"（共 {len(go_cells)} 組；最好的對手是 C0？ {g['strong']}：{'是' if g['strong_best_is_C0'] else '否'}、{g['weak']}：{'是' if g['weak_best_is_C0'] else '否'}；"
                          f"O 都過護欄、C0 的 SSD 使用率都 ≥5%、護欄輸入都沒有 NOT_MEASURED；"
                          f"這一格寬鬆版雙胞胎的省多少（只報告）：{g['strong']} {fmt_pct(g['strong_twin'])}、{g['weak']} {fmt_pct(g['weak_twin'])}）")
        return res
    if mode == "early":
        res.update(verdict="PENDING_B", reason="不含 FreeInference 的資料沒有滿足 GO；判定等 B 的 A1 範圍跑完（04 §10）")
        return res
    if not use:
        res.update(verdict="ASK", reason="沒有任何進判定的真實資料（有效 session <100、不完整或只有合成資料）")
        return res
    incomplete = [t for t, v in traces.items() if v["eligible"] and not v["complete"] and not v.get("synthetic")]
    # NO-GO 只看「能判定的格子」（C0 使用率 ≥5% 而且護欄輸入沒有 NOT_MEASURED）；不能判定的格子 GO、NO-GO 都不算（第三輪審查 #5、#7）
    evc = {t: {c: x for c, x in v["cells"].items() if x.get("evaluable", True)} for t, v in use.items()}
    n_ev = {t: len(x) for t, x in evc.items()}
    res["evaluable_cells"] = n_ev
    res["not_evaluable"] = {t: {f"{c[0]}@{c[1]:g}": x.get("why_not", "") for c, x in v["cells"].items() if not x.get("evaluable", True)} for t, v in use.items()}
    all_low = all(n > 0 for n in n_ev.values()) and all(c["hi"] < NOGO for cs in evc.values() for c in cs.values())
    if all_low and not incomplete:
        res.update(verdict="NO-GO", reason=f"所有進判定的資料（{', '.join(sorted(use))}）、所有能判定的主格（{n_ev}；其餘格子 SSD 不忙或護欄輸入有 NOT_MEASURED，不算），同時 CI 上界都 <5%")
    else:
        top = max((c["lo"] for cs in evc.values() for c in cs.values()), default=float("nan"))
        none_ev = [t for t, n in n_ev.items() if n == 0]
        res.update(verdict="ASK", reason="沒有滿足 GO，也不是全部上界 <5%" + (f"；不完整的資料：{incomplete}" if incomplete else "")
                   + (f"；這些資料沒有任何能判定的格子（SSD 不忙或護欄輸入有 NOT_MEASURED）：{none_ev}" if none_ev else "")
                   + f"（能判定的格子數 {n_ev}；最高的 CI 下界 {top:.1%}）")
    return res




# ====================================================================== 切片齊全檢查（#4）
def resolve_expected(ix, mode, expected):
    """expected：None＝EXPECTED[mode]；'present'＝只供測試（用資料裡有的切片）；或自己給 {trace: [slices]}。"""
    if expected is None:
        return EXPECTED[mode]
    if expected == "present":
        return {t: sorted(s) for t, s in ix.slices.items()}
    return expected


def check_complete(ix, expected):
    """每種模式寫死應有的切片與每個切片的 15 個主格 × 所有做法（H／2H 版；主視角、strict）。不齊 → IncompleteMap（列出缺什麼），不給判定。"""
    errs = []
    have = {t: set(s) for t, s in ix.slices.items()}
    for t in sorted(set(expected) - set(have)):
        errs.append(f"整份資料缺：{t}（應有切片 {expected[t]}）")
    for t in sorted(set(have) - set(expected)):
        errs.append(f"多出不在這個模式的資料：{t}（切片 {sorted(have[t])}）")
    for t in sorted(set(expected) & set(have)):
        miss_s, extra_s = sorted(set(expected[t]) - have[t]), sorted(have[t] - set(expected[t]))
        if miss_s:
            errs.append(f"{t} 缺切片：{miss_s}")
        if extra_s:
            errs.append(f"{t} 多出不在這個模式的切片：{extra_s}")
    miss_c = []
    for t in sorted(set(expected) & set(have)):
        for s in sorted(set(expected[t]) & have[t]):
            for (ol, mem) in PRIMARY_CELLS:
                for lab in NEEDED:
                    for h in ("H", "2H"):
                        if ix.row(t, s, ENGINE_OF[ol], ol, mem, lab, h) is None:
                            miss_c.append((t, s, ol, mem, lab, h))
    if miss_c:
        errs.append(f"缺 {len(miss_c)} 個格子（切片 × 主格 × 做法 × H／2H），前 8 個：{miss_c[:8]}")
    # 寬鬆版雙胞胎也要齊（第三輪審查 #6）：每個切片 × 主格 × 做法（H；非 oracle 是 both）；用 --skip 拿掉就不給判定
    miss_l = []
    for t in sorted(set(expected) & set(have)):
        for s in sorted(set(expected[t]) & have[t]):
            for (ol, mem) in PRIMARY_CELLS:
                for lab in NEEDED:
                    if ix.row(t, s, ENGINE_OF[ol], ol, mem, lab, "H", rule="loose") is None:
                        miss_l.append((t, s, ol, mem, lab))
    if miss_l:
        errs.append(f"寬鬆版雙胞胎缺 {len(miss_l)} 個格子（rule=loose；切片 × 主格 × 做法），前 8 個：{miss_l[:8]}")
    if errs:
        raise IncompleteMap("地圖不齊，不給判定：\n  - " + "\n  - ".join(errs))


# ====================================================================== 主分析
def analyze(df, mode="full", n_boot=N_BOOT, h_main="H", verbose=True, guard_ref="C0", expected=None, guard_ref_source="",
            oracle_variant="greedy", recompute_charge="none"):
    assert guard_ref in GUARD_REFS, guard_ref
    assert oracle_variant in ORACLE_VARIANTS, oracle_variant
    assert recompute_charge in RECOMPUTE_CHARGES, recompute_charge
    OCH = ORACLE_VARIANT_LABEL[oracle_variant]          # 判定用的「O」在資料裡的做法名
    ix = Index(df)
    exp = resolve_expected(ix, mode, expected)
    check_complete(ix, exp)
    out_rows, boot_info, traces, tradeoff = [], {}, {}, []
    tr_present = [t for t in ix.slices if t in TRACES or t == "synth"]
    for trace in sorted(tr_present, key=lambda t: (TRACES.index(t) if t in TRACES else 99)):
        slices = sorted(ix.slices[trace])
        info = dict(slices=slices, source=SOURCE.get(trace, "synthetic"), synthetic=(trace == "synth"), cells={}, eligible=False, complete=True)
        # 有效 session 數：用任一個主格 C0 的 npz 裡每個 session 的請求數（所有切片合起來）
        nreq_all = []
        for s in slices:
            r0 = ix.row(trace, s, "session", "E2", MEMS[0], "C0", "both")
            nreq_all.append(np.load(r0["npz_path"])["n_req"])
        info["eff_sessions"] = effective_sessions(np.concatenate(nreq_all))
        info["eligible"] = bool(info["eff_sessions"] >= MIN_EFF_SESSIONS)
        for hh in ("H", "2H"):
            tab = {}
            for (ol, mem) in PRIMARY_CELLS:
                res = trace_cell(ix, trace, slices, ENGINE_OF[ol], ol, mem, hh, labels=NEEDED, ovar=oracle_variant, charge=recompute_charge)
                assert all(l in res for l in NEEDED_ROLE), (trace, ol, mem, hh, [l for l in NEEDED_ROLE if l not in res])      # check_complete 之後不該發生
                tab[(ol, mem)] = eval_cell(res, guard_ref)
                if hh == "H":
                    for lab in NEEDED_ROLE:
                        tradeoff.append(_tradeoff_row(trace, ol, mem, res, lab))
            # bootstrap：只做有最好的對手、點估計有定義的格子（需要每個切片的 per-session 位元組：best 與 O 兩個做法）
            cells = [c for c in PRIMARY_CELLS if tab[c]["best"]["label"] is not None and tab[c]["nan_flag"] == ""]
            bs, jidx = None, {}
            if cells:
                per_slice = {}
                for s in slices:
                    d, ref = {}, None
                    for c in cells:
                        eng = ENGINE_OF[c[0]]
                        rb = ix.row(trace, s, eng, c[0], c[1], tab[c]["best"]["label"], hh)
                        ro = ix.row(trace, s, eng, c[0], c[1], OCH, hh)
                        sb, xb = sess_bytes(rb, hh, recompute_charge)
                        so, xo = sess_bytes(ro, hh, recompute_charge)
                        assert np.array_equal(sb, so), f"{trace}/{s}/{c}: best 與 O 的 session 對不上"
                        if ref is None:
                            ref = sb
                        assert np.array_equal(ref, sb), f"{trace}/{s}/{c}: 不同格子的 session 編號對不上"
                        d[c] = (xb, xo)
                    per_slice[s] = (len(ref), d)
                strata = [(s, per_slice[s][0], per_slice[s][1]) for s in slices]
                rng_seed = [SEED, TRACES.index(trace) if trace in TRACES else 99]
                bs = bootstrap_maxT(strata, cells, n_boot, rng_seed)
                # 驗算：bootstrap 的全樣本點估計和 CSV 算出來的點估計一致
                for j, c in enumerate(cells):
                    jidx[c] = j
                    if not tab[c]["best_zero"]:
                        assert abs(bs["theta_hat"][j] - tab[c]["point"]) < 1e-9 * max(1, abs(tab[c]["point"])) + 1e-9, (trace, c, bs["theta_hat"][j], tab[c]["point"])
                boot_info[(trace, hh)] = dict(q=bs["q"], cells=len(cells))
            for c in PRIMARY_CELLS:
                t = tab[c]
                nan = float("nan")
                j = jidx.get(c)
                row = dict(kind="primary", trace=trace, source_group=info["source"], synthetic=info["synthetic"], outer=c[0], mem_GiB=c[1], h=hh,
                           guard_ref=guard_ref, best_label=t["best"]["label"], best_GiB=t["best"]["GiB"], O_GiB=t["res"]["O"]["GiB"], savings=t["point"],
                           se=float(bs["se"][j]) if j is not None else nan, q=bs["q"] if bs else nan,
                           ci_lo=float(bs["lo"][j]) if j is not None else nan, ci_hi=float(bs["hi"][j]) if j is not None else nan,
                           best_zero=t["best_zero"], nan_flag=t["nan_flag"], best_is_C0=t["best_is_C0"], o_guard_ok=t["o_guard_ok"], o_guard_nm=t["o_guard_nm"],
                           go_ok=t["go_ok"], best_util=t["util_best"], O_util=t["util_O"], best_GiB_per_h=t["best_gph"], O_GiB_per_h=t["O_gph"],
                           oracle_label=OCH, oracle_variant=oracle_variant, recompute_charge=recompute_charge, c0_util=t["util_C0"], ssd_busy=t["ssd_busy"],
                           evaluable=t["evaluable"], why_not=t["why_not"], O_alt_label=t["res"]["O_alt"]["label"], O_alt_GiB=t["res"]["O_alt"]["GiB"],
                           savings_O_alt=savings_point(t["best"]["GiB"], t["res"]["O_alt"]["GiB"])[0],
                           best_charged_GiB=(t["res"][t["best"]["label"]]["charged_GiB"] if t["best"]["label"] else float("nan")),
                           O_charged_GiB=t["res"]["O"]["charged_GiB"],
                           best_recompute_tokens=t["best_rec"] if t["best_rec"] is not None else nan, O_recompute_tokens=t["O_rec"],
                           guard_excluded=";".join(t["best"]["excluded"]), guard_nm=";".join(t["best"]["nm"]), n_boot=n_boot,
                           eligible=info["eligible"], eff_sessions=info["eff_sessions"])
                out_rows.append(row)
                if hh == h_main:
                    info["cells"][c] = dict(lo=row["ci_lo"], hi=row["ci_hi"], point=row["savings"], go_ok=t["go_ok"], best_is_C0=t["best_is_C0"],
                                            evaluable=t["evaluable"], why_not=t["why_not"])
        traces[trace] = info
        if verbose:
            print(f"[{trace}] 切片 {len(slices)}；有效 session {info['eff_sessions']:.1f}；進判定 {info['eligible']}；完整 {info['complete']}", flush=True)
    # 寬鬆版雙胞胎（只報告）先算好：GO 的理由要附上那一格雙胞胎的省多少（第三輪審查 #6）
    twin_rows = loose_twin(ix, traces, guard_ref, oracle_variant, recompute_charge)
    twin = {(r["trace"], r["outer"].split("@")[0], r["mem_GiB"]): r["savings"] for r in twin_rows}
    for t_, info_ in traces.items():
        for c_, x_ in info_["cells"].items():
            x_["twin"] = twin.get((t_, c_[0], c_[1]), float("nan"))
    verdict = decide(traces, mode)
    # 合成資料「如果它是真實的」會是什麼（只標示，不是判定）
    syn = {t: v for t, v in traces.items() if v.get("synthetic")}
    pseudo = None
    if syn:
        fake = {t: dict(v, synthetic=False, source="synthetic") for t, v in syn.items()}
        pseudo = decide(fake, "full")
    reports = report_only(ix, traces, guard_ref, oracle_variant, recompute_charge, twin_rows)
    return dict(rows=out_rows, traces=traces, verdict=verdict, synthetic_pseudo_verdict=pseudo, reports=reports, boot=boot_info, tradeoff=tradeoff,
                guard_ref=guard_ref, guard_ref_source=guard_ref_source, expected={t: list(s) for t, s in exp.items()},
                oracle_variant=oracle_variant, recompute_charge=recompute_charge, n_boot=n_boot)


def _tradeoff_row(trace, ol, mem, res, lab):
    """每個主格、每個做法（含 O）：(主指標位元組、等待時間平均、p99、重算 token、兩種護欄參考下合格與否)。"""
    r = res[lab]
    tt = r["ttft"]
    el = {}
    for ref in GUARD_REFS:
        rv = guard_ref_value(res, ref)
        ok, nmf = guard_status(tt, rv)
        el[ref] = ("通過（NOT_MEASURED）" if nmf else "通過") if ok else "不合格"
    return dict(trace=trace, outer=ol, mem_GiB=mem, policy=r.get("label", lab), is_O=(lab == "O"), is_O_alt=(lab == "O_alt"), primary_GiB=r["GiB"], GiB_per_h=r["gph"], ssd_util=r["util"],
                ttftq_H_mean_s=(NOT if tt[2] else tt[0]), ttftq_H_p99_s=(NOT if tt[2] else tt[1]), recompute_tokens=r["rec_tokens"],
                eligible_ref_C0=el["C0"], eligible_ref_min_C0_O=el["min_C0_O"])


# ====================================================================== 只報告的項目
def report_only(ix, traces, guard_ref="C0", ovar="greedy", charge="none", twin_rows=None):
    """只算點估計（不做 bootstrap、不進判定）：每份資料 × 每個外層做法 × 每個記憶體 × 每個視角（A2 的整張地圖，含 E1、E5 的另兩個 T、E6 與 E2 的 block 版、
    非主視角），加上 O-loose、拆解、敏感度（Qwen、70B、疊加、CPU 忙、3090）、兩種護欄參考、寬鬆版雙胞胎。"""
    rows = []
    seen = sorted({(k[0], k[2], k[3], k[4], k[5], k[6], k[7], k[8]) for k in ix.rows if k[9] == "strict"})   # (trace, engine, olabel, mem, model, overlay, view_group, view)
    for (trace, eng, ol, mem, model, ov, vg, view) in seen:
        if trace not in traces:
            continue
        slices = sorted(ix.slices[trace])
        primary_view = view == PRIMARY_VIEW and vg == "main"
        if ol == "E1":      # E1 沒有 SSD，不進判定、也沒有「省多少」：只報等待時間與重算（04 §6）
            if model == "llama" and ov == 1 and primary_view:
                for lab in LABELS:
                    rs = [ix.row(trace, s_, eng, ol, mem, lab, "H", model, ov, vg, view) for s_ in slices]
                    if any(r is None for r in rs):
                        continue
                    try:
                        t = trace_ttft(rs, "H")
                    except NoPooledTtft:
                        continue
                    rows.append(dict(kind="report_E1", trace=trace, source_group=SOURCE.get(trace, "synthetic"), synthetic=(trace == "synth"),
                                     outer=f"{ol}@{eng}", mem_GiB=mem, h="H", item=lab, model=model, overlay=ov, view_group=vg, view=view,
                                     ttftq_mean_s=t[0] if not t[2] else AM_NM, ttftq_p99_s=t[1] if not t[2] else AM_NM,
                                     ret_recompute_prefix_blocks=int(sum(r["ret_recompute_prefix_blocks"] for r in rs)),
                                     surprise_blocks=int(sum(r["surprise_blocks"] for r in rs))))
            continue
        for hh in ("H", "2H"):
            res = trace_cell(ix, trace, slices, eng, ol, mem, hh, model, ov, vg, view=view, ovar=ovar, charge=charge)
            if "O" not in res or "C0" not in res:
                continue
            d = eval_cell(res, guard_ref)
            best = d["best"]
            parts = ([vg[5:] if vg.startswith("sens_") else vg] if vg != "main" else []) + ([model] if model != "llama" else []) + \
                ([f"ov{ov}"] if ov > 1 else [])
            if parts:
                kind = "sens_" + "_".join(parts)
            elif not primary_view:
                kind = "view"
            else:
                kind = "primary_point" if ((ol, mem) in PRIMARY_CELLS and eng == ENGINE_OF.get(ol)) else "map"
            if eng == "block" and ol == "E2" and not parts and primary_view:
                kind = "sens_E2_block"
            base = dict(trace=trace, source_group=SOURCE.get(trace, "synthetic"), synthetic=(trace == "synth"), outer=f"{ol}@{eng}", mem_GiB=mem,
                        h=hh, model=model, overlay=ov, view_group=vg, view=view, guard_ref=guard_ref, best_label=best["label"], best_GiB=best["GiB"],
                        guard_excluded=";".join(best["excluded"]), guard_nm=";".join(best["nm"]), best_zero=d["best_zero"], nan_flag=d["nan_flag"],
                        best_is_C0=d["best_is_C0"], o_guard_ok=d["o_guard_ok"], best_util=d["util_best"], O_util=d["util_O"],
                        best_GiB_per_h=d["best_gph"], O_GiB_per_h=d["O_gph"], c0_util=d["util_C0"], ssd_busy=d["ssd_busy"], evaluable=d["evaluable"],
                        oracle_label=res["O"]["label"])
            rows.append(dict(base, kind=kind, item="O", O_GiB=res["O"]["GiB"], savings=d["point"]))
            if "O_alt" in res and not math.isnan(best["GiB"]) and hh == "H" and primary_view and vg == "main" and model == "llama" and ov == 1:
                s_alt, _, _ = savings_point(best["GiB"], res["O_alt"]["GiB"])         # 另一個 O（判定沒用它）：只報告
                rows.append(dict(base, kind="report_oracle_alt", item=res["O_alt"]["label"], O_GiB=res["O_alt"]["GiB"], savings=s_alt))
            if hh != "H" or vg != "main" or model != "llama" or ov != 1 or not primary_view:
                continue
            # 兩種護欄參考（使用者決定中；兩種都報告）：主格與整張地圖都有
            for rf in GUARD_REFS:
                dd = eval_cell(res, rf)
                rows.append(dict(base, kind="report_guard_ref", item=f"ref_{rf}", guard_ref=rf, best_label=dd["best"]["label"], best_GiB=dd["best"]["GiB"],
                                 guard_excluded=";".join(dd["best"]["excluded"]), guard_nm=";".join(dd["best"]["nm"]), best_zero=dd["best_zero"],
                                 nan_flag=dd["nan_flag"], best_is_C0=dd["best_is_C0"], o_guard_ok=dd["o_guard_ok"], best_util=dd["util_best"],
                                 best_GiB_per_h=dd["best_gph"], O_GiB=res["O"]["GiB"], savings=dd["point"], go_ok=dd["go_ok"],
                                 best_recompute_tokens=dd["best_rec"] if dd["best_rec"] is not None else float("nan"), O_recompute_tokens=dd["O_rec"]))
            for gl, gthr in GUARD_SENS:          # 護欄的敏感度（04 §4，只報告）：同一批做法、換護欄的鬆緊（參考＝判定用的 guard_ref）
                bg = best_of(res, guard=gthr is not None, thr=gthr if gthr is not None else GUARD, ref=guard_ref)
                sg, bzg, _ = savings_point(bg["GiB"], res["O"]["GiB"])
                rows.append(dict(base, kind="report_guard_sens", item=f"guard_{gl}", guard_thr=gl, best_label=bg["label"], best_GiB=bg["GiB"],
                                 guard_excluded=";".join(bg["excluded"]), guard_nm=";".join(bg["nm"]), best_zero=bzg, O_GiB=res["O"]["GiB"], savings=sg))
            # 拆解與 O-loose（只在主視角、Llama、H）
            for lab in ("O-loose", "O-len", "O-reuse"):
                if lab in res and not math.isnan(best["GiB"]):
                    s2, _, _ = savings_point(best["GiB"], res[lab]["GiB"])
                    rows.append(dict(base, kind="report_oracle_variant", item=lab, O_GiB=res[lab]["GiB"], savings=s2))
            if "O" in res and "O-len" in res and best["GiB"] > 0:
                rows.append(dict(base, kind="report_decomp", item="O_minus_Olen", O_GiB=res["O"]["GiB"], savings=(res["O-len"]["GiB"] - res["O"]["GiB"]) / best["GiB"]))
            if "O" in res and "O-reuse" in res and best["GiB"] > 0:
                rows.append(dict(base, kind="report_decomp", item="O_minus_Oreuse", O_GiB=res["O"]["GiB"], savings=(res["O-reuse"]["GiB"] - res["O"]["GiB"]) / best["GiB"]))
            for lab, fam in (("O-len", "C4r"), ("O-reuse", "D-idle")):
                b2 = best_of(res, [c for c in NON_ORACLE if c.startswith(fam)], ref=guard_ref)
                if lab in res and b2["label"]:
                    s2, _, _ = savings_point(b2["GiB"], res[lab]["GiB"])
                    rows.append(dict(base, kind="report_decomp", item=f"{lab}_vs_best_{fam}", best_label=b2["label"], best_GiB=b2["GiB"],
                                     guard_excluded=";".join(b2["excluded"]), O_GiB=res[lab]["GiB"], savings=s2))
    rows += twin_rows if twin_rows is not None else loose_twin(ix, traces, guard_ref, ovar, charge)
    return rows


def loose_twin(ix, traces, guard_ref, ovar="greedy", charge="none"):
    """寬鬆版雙胞胎（04 §4，第二輪審查 #3a；只報告）：O 與所有對手都用寬鬆品質規則（rule=loose 的工作；32K 過關的格式在任何長度都過關）。
    每份資料 × 主格、H；點估計（不做 bootstrap）。缺工作的格子不報（報告用，不擋判定）。"""
    rows = []
    for trace in traces:
        slices = sorted(ix.slices[trace])
        for (ol, mem) in PRIMARY_CELLS:
            eng = ENGINE_OF[ol]
            res = trace_cell(ix, trace, slices, eng, ol, mem, "H", labels=NEEDED, rule="loose", ovar=ovar, charge=charge)
            if any(l not in res for l in NEEDED_ROLE):
                continue
            d = eval_cell(res, guard_ref)
            rows.append(dict(kind="report_loose_twin", trace=trace, source_group=SOURCE.get(trace, "synthetic"), synthetic=(trace == "synth"),
                             outer=f"{ol}@{eng}", mem_GiB=mem, h="H", model="llama", overlay=1, view_group="main", view=PRIMARY_VIEW, item="O-loose-rule",
                             guard_ref=guard_ref, best_label=d["best"]["label"], best_GiB=d["best"]["GiB"], O_GiB=res["O"]["GiB"], savings=d["point"],
                             best_zero=d["best_zero"], nan_flag=d["nan_flag"], best_is_C0=d["best_is_C0"], o_guard_ok=d["o_guard_ok"],
                             best_util=d["util_best"], O_util=d["util_O"], best_GiB_per_h=d["best_gph"], O_GiB_per_h=d["O_gph"], go_ok=d["go_ok"],
                             c0_util=d["util_C0"], ssd_busy=d["ssd_busy"], evaluable=d["evaluable"], oracle_label=res["O"]["label"],
                             guard_excluded=";".join(d["best"]["excluded"]), guard_nm=";".join(d["best"]["nm"]),
                             best_recompute_tokens=d["best_rec"] if d["best_rec"] is not None else float("nan"), O_recompute_tokens=d["O_rec"]))
    return rows


# ====================================================================== 輸出
def fmt_pct(x):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.1f}%"


def _flags_of_row(r):
    f = []
    if r.get("nan_flag"):
        f.append({"best_zero_o_pos": "best=0、O>0（NaN）", "no_eligible_rival": "沒有合格的對手", "nan": "NaN"}.get(r["nan_flag"], r["nan_flag"]))
    elif r.get("best_zero"):
        f.append("best=0")
    if not r.get("o_guard_ok", True):
        f.append("O 沒過護欄")
    if r.get("best_label") is not None and not (isinstance(r.get("best_label"), float)):
        if r.get("best_is_C0"):
            f.append("最好=C0")
    if r.get("ssd_busy") is False:
        f.append(f"SSD 不忙（C0 使用率 {fmt_pct(r.get('c0_util'))}<5%；GO、NO-GO 都不算）")
    if "NOT_MEASURED" in str(r.get("why_not", "")):
        f.append("護欄輸入有 NOT_MEASURED（GO、NO-GO 都不算）")
    return f


def write_outputs(result, prefix, mode, cells_path):
    os.makedirs(os.path.dirname(os.path.abspath(prefix)) or ".", exist_ok=True)
    rows = result["rows"]
    # summary CSV：主格（有 CI）＋ 只報告
    allrows = rows + result["reports"]
    keys = []
    for r in allrows:
        for k in r:
            if k not in keys:
                keys.append(k)
    rid = os.environ.get("RUN_ID", "adhoc")
    ts = pd.Timestamp.now(tz="Asia/Taipei").isoformat(timespec="seconds")
    with open(prefix + "_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["run_id", "ts"] + keys, restval="")
        w.writeheader()
        for r in allrows:
            w.writerow(dict(run_id=rid, ts=ts, **r))
    # 取捨表：每個主格、每個做法（含 O）的 (主指標位元組、等待時間平均、p99、重算 token、兩種護欄參考下合格與否)
    if result.get("tradeoff"):
        with open(prefix + "_tradeoff.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["run_id", "ts"] + list(result["tradeoff"][0]), restval="")
            w.writeheader()
            for r in result["tradeoff"]:
                w.writerow(dict(run_id=rid, ts=ts, **r))
    v = result["verdict"]
    js = dict(run_id=rid, ts=ts, mode=mode, cells_csv=cells_path, verdict=v["verdict"], reason=v["reason"], go_cells=v["go_cells"], used=v["used"],
              not_used=v["not_used"], n_boot=result.get("n_boot", N_BOOT), seed=SEED, guard=GUARD, guard_ref=result["guard_ref"], guard_ref_source=result["guard_ref_source"],
              oracle_variant=result["oracle_variant"], recompute_charge=result["recompute_charge"], evaluable_cells=v.get("evaluable_cells"),
              not_evaluable=v.get("not_evaluable"), freeze=result.get("freeze"),
              expected_slices=result["expected"],
              traces={t: dict(slices=i["slices"], eligible=i["eligible"], complete=i["complete"], eff_sessions=i["eff_sessions"], source=i["source"],
                              synthetic=i["synthetic"]) for t, i in result["traces"].items()},
              guard_sensitivity=[{k: (None if isinstance(x, float) and math.isnan(x) else x) for k, x in r.items()} for r in result["reports"] if r["kind"] == "report_guard_sens"],
              guard_ref_report=[{k: (None if isinstance(x, float) and math.isnan(x) else x) for k, x in r.items()} for r in result["reports"] if r["kind"] == "report_guard_ref"],
              loose_twin=[{k: (None if isinstance(x, float) and math.isnan(x) else x) for k, x in r.items()} for r in result["reports"] if r["kind"] == "report_loose_twin"],
              synthetic_pseudo_verdict=result["synthetic_pseudo_verdict"], boot_q={f"{k[0]}|{k[1]}": b for k, b in result["boot"].items()})
    json.dump(js, open(prefix + "_verdict.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    # markdown：整張地圖（主格有 CI；其他只有點）
    L = [f"# 地圖判定（{mode}）", "", f"- 判定：**{v['verdict']}**", f"- 理由：{v['reason']}", f"- 進判定的資料：{', '.join(v['used']) or '（無）'}；沒進的：{json.dumps(v['not_used'], ensure_ascii=False)}",
         f"- 護欄的參考（判定用）：**{result['guard_ref']}**（{result['guard_ref_source'] or '04 §9'}）；兩種參考都在下面「護欄的參考」節",
         f"- 完美預測（判定用的 O）：**{result['oracle_variant']}**（{ORACLE_VARIANT_LABEL[result['oracle_variant']]}；另一個在 report_oracle_alt）；重算計價：**{result['recompute_charge']}**",
         f"- 能判定的格子（C0 的 SSD 使用率 ≥{UTIL_MIN:.0%}、護欄輸入沒有 NOT_MEASURED；其餘格子 GO、NO-GO 都不算）：{v.get('evaluable_cells')}",
         f"- bootstrap {N_BOOT} 次、seed {SEED}、max-T 同時 95% CI；等待時間護欄 {GUARD}×參考（截尾版 ttftq_H_*）；GO 另要求 O 過護欄；run `{rid}`", ""]
    if result["synthetic_pseudo_verdict"]:
        L += [f"- **合成資料**（不是判定）：如果當成真實資料，規則會給 {result['synthetic_pseudo_verdict']['verdict']}（{result['synthetic_pseudo_verdict']['reason']}）", ""]
    for hh, title in (("H", "截尾 H（判定用）"), ("2H", "截尾 2H（只報告）")):
        L += [f"## 15 個主格：省多少＝(最好的對手 − O) ÷ 最好的對手；{title}；點估計［同時 CI 下界, 上界］", ""]
        trs = [t for t in result["traces"]]
        L += ["| 外層做法 | 記憶體 GiB | " + " | ".join(f"{t}{'（合成）' if result['traces'][t]['synthetic'] else ''}" for t in trs) + " |",
              "|:--|--:|" + "|".join(":--" for _ in trs) + "|"]
        for (ol, mem) in PRIMARY_CELLS:
            cells = []
            for t in trs:
                r = next((x for x in rows if x["trace"] == t and x["outer"] == ol and x["mem_GiB"] == mem and x["h"] == hh), None)
                if r is None:
                    cells.append("—")
                    continue
                txt = f"{fmt_pct(r['savings'])} [{fmt_pct(r['ci_lo'])}, {fmt_pct(r['ci_hi'])}]"
                if hh == "H":
                    txt += f"<br>最好：{r['best_label']}（{r['best_GiB_per_h']:.3g} GiB/h、使用率 {fmt_pct(r['best_util'])}）；O：{r['O_GiB_per_h']:.3g} GiB/h、{fmt_pct(r['O_util'])}"
                    txt += f"<br>重算 token：最好 {r['best_recompute_tokens']:.3g}／O {r['O_recompute_tokens']:.3g}"
                    fl = _flags_of_row(r)
                    if fl:
                        txt += "<br>⚑ " + "；".join(fl)
                cells.append(txt)
            L.append(f"| {ol} | {mem:g} | " + " | ".join(cells) + " |")
        L.append("")
    gr = [r for r in result["reports"] if r["kind"] == "report_guard_ref" and r["outer"].split("@")[1] == ENGINE_OF.get(r["outer"].split("@")[0])
          and (r["outer"].split("@")[0], r["mem_GiB"]) in PRIMARY_CELLS]
    if gr:
        trs = [t for t in result["traces"]]
        L += ["## 護欄的參考（兩種都報告；點估計；H；主格）：C0＝對手要 ≤1.05×C0；min(C0, O)＝要 ≤1.05×C0 與 O 逐項取小", "",
              "| 外層做法 | 記憶體 GiB | 參考 | " + " | ".join(trs) + " |", "|:--|--:|:--|" + "|".join(":--" for _ in trs) + "|"]
        for (ol, mem) in PRIMARY_CELLS:
            for rf in GUARD_REFS:
                cells = []
                for t in trs:
                    r = next((x for x in gr if x["trace"] == t and x["outer"].split("@")[0] == ol and x["mem_GiB"] == mem and x["guard_ref"] == rf), None)
                    cells.append("—" if r is None else f"{fmt_pct(r['savings'])}（{r['best_label']}）" + ("" if not _flags_of_row(r) else " ⚑" + "；".join(_flags_of_row(r))))
                L.append(f"| {ol} | {mem:g} | {rf} | " + " | ".join(cells) + " |")
        L.append("")
    lt = [r for r in result["reports"] if r["kind"] == "report_loose_twin"]
    if lt:
        trs = [t for t in result["traces"]]
        L += ["## 寬鬆版雙胞胎（只報告；點估計；H；O 與所有對手都用寬鬆品質規則＝32K 過關的格式任何長度都過關）", "",
              "| 外層做法 | 記憶體 GiB | " + " | ".join(trs) + " |", "|:--|--:|" + "|".join(":--" for _ in trs) + "|"]
        for (ol, mem) in PRIMARY_CELLS:
            cells = []
            for t in trs:
                r = next((x for x in lt if x["trace"] == t and x["outer"].split("@")[0] == ol and x["mem_GiB"] == mem), None)
                cells.append("—" if r is None else f"{fmt_pct(r['savings'])}（{r['best_label']}）" + ("" if not _flags_of_row(r) else " ⚑" + "；".join(_flags_of_row(r))))
            L.append(f"| {ol} | {mem:g} | " + " | ".join(cells) + " |")
        L.append("")
    gs = [r for r in result["reports"] if r["kind"] == "report_guard_sens" and (r["outer"].split("@")[0], r["mem_GiB"]) in PRIMARY_CELLS
          and r["outer"].split("@")[1] == ENGINE_OF.get(r["outer"].split("@")[0])]
    if gs:
        trs = [t for t in result["traces"]]
        L += [f"## 護欄敏感度（只報告；點估計；H；主格；參考＝{result['guard_ref']}）：護欄 1.05 是登記的（判定用），其他只用來看判定有多依賴 5%", "",
              "| 外層做法 | 記憶體 GiB | 護欄 | " + " | ".join(trs) + " |", "|:--|--:|:--|" + "|".join(":--" for _ in trs) + "|"]
        for (ol, mem) in PRIMARY_CELLS:
            for gl, _ in GUARD_SENS:
                cells = []
                for t in trs:
                    r = next((x for x in gs if x["trace"] == t and x["outer"].split("@")[0] == ol and x["mem_GiB"] == mem and x["h"] == "H"
                              and x["guard_thr"] == gl), None)
                    cells.append("—" if r is None else f"{fmt_pct(r['savings'])}（{r['best_label']}）")
                L.append(f"| {ol} | {mem:g} | {gl if gl != 'none' else '不設'} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## 整張地圖（只有點估計，H；只報告）", ""]
    rep = [r for r in result["reports"] if r["kind"] in ("map", "primary_point", "sens_E2_block") and r["h"] == "H" and r["item"] == "O"]
    if rep:
        d = pd.DataFrame(rep)
        d["col"] = d.trace
        for (eng_ol), g in d.groupby("outer"):
            L += [f"### {eng_ol}", "", "| 記憶體 GiB | " + " | ".join(sorted(g.col.unique())) + " |", "|--:|" + "|".join("--:" for _ in g.col.unique()) + "|"]
            for mem, g2 in g.groupby("mem_GiB"):
                L.append(f"| {mem:g} | " + " | ".join(fmt_pct(float(g2[g2.col == c].savings.iloc[0])) if (g2.col == c).any() else "—" for c in sorted(g.col.unique())) + " |")
            L.append("")
    vw = [r for r in result["reports"] if r["kind"] == "view" and r["h"] == "H"]
    if vw:
        L += ["## 其他視角（只報告；點估計；H；主視角以外的時間設定與 GPU 忙）", "", "| 資料 | 外層 | 記憶體 | 視角 | 最好的對手 | 省多少 |", "|:--|:--|--:|:--|:--|--:|"]
        for r in vw:
            L.append(f"| {r['trace']} | {r['outer']} | {r['mem_GiB']:g} | {r['view']} | {r['best_label']} | {fmt_pct(r['savings'])} |")
        L.append("")
    sens = [r for r in result["reports"] if str(r["kind"]).startswith("sens_") and r["kind"] != "sens_E2_block" and r["h"] == "H"]
    if sens:
        L += ["## 敏感度（只報告；點估計；半合成標在 kind）", "", "| kind | 資料 | 外層 | 記憶體 | 最好的對手 | 省多少 |", "|:--|:--|:--|--:|:--|--:|"]
        for r in sens:
            L.append(f"| {r['kind']} | {r['trace']} | {r['outer']} | {r['mem_GiB']:g} | {r['best_label']} | {fmt_pct(r['savings'])} |")
        L.append("")
    open(prefix + "_map.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    return prefix


def resolve_guard_ref(arg, frozen=None, prereg_text=None):
    """護欄的參考（04 §9 的 `護欄的參考：…`）。arg＝'auto'（讀 04；--frozen 給 commit 就讀那個 commit 的版本；沒有那一行 → 預設 C0 並寫明「未凍結」）｜'C0'｜'min_C0_O'。
    回傳 (ref, 來源說明)。"""
    if arg in GUARD_REFS:
        return arg, f"命令列指定 {arg}（不是 04 §9 的值）"
    assert arg == "auto", arg
    if prereg_text is None:
        if frozen:
            prereg_text = lockfile._git(REPO, "show", f"{frozen}:{lockfile.PREREG}").stdout
        else:
            prereg_text = open(os.path.join(REPO, lockfile.PREREG), encoding="utf-8").read()
    ref = lockfile.parse_guard_ref(prereg_text)
    if ref is None:
        return "C0", "04 §9 還沒有『護欄的參考：…』那一行（未凍結）→ 預設 C0"
    return ref, f"04 §9：{ref}" + (f"（commit {frozen[:8]}）" if frozen else "（工作目錄的 04）") + \
        ("；04 還是未凍結的草稿，這個值是草稿寫的（凍結時以凍結的 commit 為準）" if "未凍結" in prereg_text else "")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--mode", choices=["early", "full"], default="full")
    ap.add_argument("--synthetic", action="store_true", help="資料是合成切片（synth/small-seed1）：應有的切片清單換成合成的；判定參數可以用命令列選（測試用）")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--boot", type=int, default=None, help="bootstrap 次數：真實資料固定 1000（給了別的值就拒絕）；合成資料預設 1000")
    ap.add_argument("--guard-ref", default=None, choices=["auto", "C0", "min_C0_O"], help="合成資料用；真實資料只從凍結的 04 §9 讀（給了就拒絕）")
    ap.add_argument("--oracle-variant", default=None, choices=["greedy", "tier"], help="合成資料用（預設 greedy）；真實資料只從凍結的 04 §9 讀")
    ap.add_argument("--recompute-charge", default=None, choices=["none", "pass_size"], help="合成資料用（預設 none）；真實資料只從凍結的 04 §9 讀")
    ap.add_argument("--frozen", default=None, help="真實資料：必須給（04 凍結的 commit；過 lockfile.verify_freeze，列的 commit 要和它對得上）")
    a = ap.parse_args(argv)
    freeze = None
    if a.synthetic:
        n_boot = a.boot or N_BOOT
        ref, src = resolve_guard_ref(a.guard_ref or "auto", a.frozen)
        ovar, chg = a.oracle_variant or "greedy", a.recompute_charge or "none"
        src += "（合成資料：判定參數可以用命令列選）"
    else:
        # ---- 真實資料的凍結閘門（第三輪審查 #2）：一律要 --frozen、過 verify_freeze；判定參數只從凍結的 04 讀；n_boot 固定 1000
        if not a.frozen:
            print("拒絕：真實資料的分析一律要 --frozen <04 凍結的 commit>", file=sys.stderr)
            return 2
        over = [n for n, v in (("--guard-ref", a.guard_ref), ("--oracle-variant", a.oracle_variant), ("--recompute-charge", a.recompute_charge)) if v is not None]
        if over or (a.boot is not None and a.boot != N_BOOT):
            print(f"拒絕：真實資料的護欄參考、完美預測、重算計價只從凍結的 04 §9 讀，n_boot 固定 {N_BOOT}；不准用命令列覆蓋：{over or ['--boot']}", file=sys.stderr)
            return 2
        try:
            freeze = lockfile.verify_freeze(a.frozen, REPO)
        except SystemExit as e:
            print(str(e), file=sys.stderr)
            return 2
        ref, ovar, chg, n_boot = freeze["guard_ref"], freeze["oracle_variant"], freeze["recompute_charge"], N_BOOT
        src = f"04 §9（commit {freeze['frozen'][:8]}）：護欄的參考 {ref}、完美預測 {ovar}、重算計價 {chg}"
    print(f"護欄的參考：{ref}（{src}）；完美預測：{ovar}；重算計價：{chg}；n_boot {n_boot}", flush=True)
    df = load_cells(a.cells)
    if freeze is not None:
        miss = [c for c in ("frozen_commit", "code_commit", "run_id") if c not in df.columns]
        if miss:
            print(f"拒絕：{a.cells} 缺欄 {miss}（用新版 run_map.py 重跑）", file=sys.stderr)
            return 2
        errs = lockfile.check_row_commits(df["frozen_commit"].fillna("").astype(str), df["code_commit"].fillna("").astype(str), freeze, REPO,
                                          df["run_id"].fillna("").astype(str))
        if errs:
            print("拒絕：列的 commit 檢查失敗：\n  - " + "\n  - ".join(errs), file=sys.stderr)
            return 2
    try:
        res = analyze(df, a.mode, n_boot, guard_ref=ref, guard_ref_source=src, expected=EXPECTED["synthetic"] if a.synthetic else None,
                      oracle_variant=ovar, recompute_charge=chg)
    except (IncompleteMap, DuplicateCell) as e:
        print(str(e), file=sys.stderr)
        return 2
    if freeze is not None:
        res["freeze"] = dict(frozen=freeze["frozen"], code_commit=freeze["code_commit"], head=freeze["head"], prereg_sha256=freeze["prereg_sha256"],
                             n_locked_files=len(freeze["files"]))
    write_outputs(res, a.out_prefix, a.mode, a.cells)
    print(json.dumps(dict(verdict=res["verdict"]["verdict"], reason=res["verdict"]["reason"], guard_ref=ref, oracle_variant=ovar, recompute_charge=chg), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
