"""p3_formats/build_format_table.py — 第三階段的壓縮格式表（docs/phase3_20261017/04_prereg_map.md §1、§3、§5 的輸入）。

只用 CPU、只讀第二階段（MI300X）與 X7（3090）既有的結果；不量任何東西。每一格都附 run_id 與來源檔。
沒有的寫 NOT_MEASURED；第二階段還在跑、會有的寫 PENDING（附 stage）。

每列＝（模型 × 格式）。格式＝09 §2.1 的 13 種＋BF16，另加每個模型「另一種 FP8 粒度」（有計時，不在 09 的清單）。

欄位群（細節見 docs/phase3_20261017/05_format_table.md）：
  位元組   bytes_per_block_code（照 m10_codec／m10_codec_x 的程式寫出的公式）、bytes_per_block_encode_cpu（這個 run 在 CPU 上
           真的呼叫 encode() 的 Enc.nbytes）、MI300X 存下來的實際大小（parts、還原格子、海選）、3090 的實際大小；
           bytes_per_token＝每 block ÷ 512；ratio_vs_bf16。
  反量化   MI300X：parts（真實 KV、單獨量、每 block、閒／忙；原寫法 K0＝t_dequant_torch、融合 K1＝t_dequant_fused）、
           42b A 段 kernel 微測（合成、寫進整個 KV 視圖；k0／k1／k2／K*）、stage 40 bench（閒；和 3090 同一支工具）、
           還原中（真實 CPU 層 32K、Cake 與只載；每 block＝median(t_dequant ÷ 讀到的低精度 block 數)；K0＝主線格子／2b／43，
           融合＝stage 42）。
           3090：X7 的 bench（閒；忙 NOT_MEASURED）、Qwen 還原中（從 run 目錄的原始 restores.csv 重算）。
  品質     32K。生成答對率差＋95% CI（主線 m10_analyze，1,140 列；四種情境）；teacher forcing（41 海選，96 題，ALL＝block 1 以後
           全部低精度）的 first_ok 差、KL；格式小測（20 題）的 KL。過 2 分界線＝CI 下界 > −2（生成）；teacher forcing 只有點估計。

用法：TIARA_NO_GPU=1 bash code/run3090.sh p3fmt-table python code/p3_formats/build_format_table.py
"""
from __future__ import annotations

import datetime
import gzip
import io
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "code"))
RUN_ID = os.environ.get("RUN_ID", "adhoc")
RUN_DIR = os.environ.get("RUN_DIR") or os.path.join("/ssd8/hungwei/paper-hkv/runs", RUN_ID)
OUT = os.path.join(REPO, "results", "p3_formats")
TS = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
NM = "NOT_MEASURED"
PEND = "PENDING"

R10 = "results/m10_precision_mi300x"
R10S = "results/m10_sweep_mi300x"
X7 = "results/p3_explore/x7"
X5 = "results/p3_explore/x5"
# 平台（2026-10-19，H200 移植）：P3_CONST 沒設＝MI300X，和原本逐位元相同。P3_CONST=h200：同一套規則讀 H200 的同一批實測
# （code/p3_platform.py 的目錄與 run 對應），欄名的 mi300x 換成 h200、沒有 3090 欄（H200 是唯一平台），輸出 results/p3_formats_h200/。
import p3_platform as PF  # noqa: E402
PL = PF.PLAT
if PL != "mi300x":
    OUT = os.path.join(REPO, "results", f"p3_formats_{PL}")
    R10 = os.path.relpath(PF.rdir("m10_precision_mi300x"), REPO)
    R10S = os.path.relpath(PF.rdir("m10_sweep_mi300x"), REPO)
    X5 = os.path.relpath(PF.explore_out("x5"), REPO)
    RUN_DIR = os.environ.get("RUN_DIR") or os.path.join(os.environ.get("TIARA_RUNS", "."), RUN_ID)
QRG = "speed_qwen_regrid" if PL == "mi300x" else "speed_qwen"     # H200 沒有 Qwen 的重量（regrid），只量一次
SCREEN_RUNS = {"mi300x": {"llama": ["20261016-135544-m10-sweep-screen-llama-s0", "20261016-135700-m10-sweep-screen-llama-s1"],
                          "qwen": ["20261016-135815-m10-sweep-screen-qwen-s0", "20261016-135930-m10-sweep-screen-qwen-s1"]},
               "h200": {"llama": ["20261018-064714-m10h-sweep-screen-llama-s0", "20261018-064829-m10h-sweep-screen-llama-s1"],
                        "qwen": ["20261018-074402-m10h-sweep-screen-qwen-s0", "20261018-074517-m10h-sweep-screen-qwen-s1"]}}[PL]
G43 = {"mi300x": {"llama": {"int2_kivi": "20261017-042229-m10-confirm-grid-llama-int2_kivi",
                            "kv-int4_kivi-int2_kivi": "20261017-054512-m10-confirm-grid-llama-kv-int4_kivi-int2_kivi",
                            "int4_kivi": "20261017-070052-m10-confirm-grid-llama-int4_kivi",
                            "mxfp4": "20261017-074133-m10-confirm-grid-llama-mxfp4"},
                  "qwen": {"kv-int4_kivi-int2_kivi": "20261017-082406-m10-confirm-grid-qwen-kv-int4_kivi-int2_kivi",
                           "int4_kivi": "20261017-090805-m10-confirm-grid-qwen-int4_kivi",
                           "kv-int8_token-int4_kivi": "20261017-094858-m10-confirm-grid-qwen-kv-int8_token-int4_kivi"}},
       # H200 的 43 是 H200 自己的決賽（13_h200_repro；和 MI300X 的選手不完全相同：Llama 有 nvfp4、沒有 mxfp4）
       "h200": {"llama": {"int2_kivi": "20261018-164945-m10h-confirm-grid-llama-int2_kivi",
                          "kv-int4_kivi-int2_kivi": "20261018-174423-m10h-confirm-grid-llama-kv-int4_kivi-int2_kivi",
                          "nvfp4": "20261018-182504-m10h-confirm-grid-llama-nvfp4"},
                "qwen": {"kv-int4_kivi-int2_kivi": "20261018-185126-m10h-confirm-grid-qwen-kv-int4_kivi-int2_kivi",
                         "kv-int8_token-int4_kivi": "20261018-191507-m10h-confirm-grid-qwen-kv-int8_token-int4_kivi",
                         "int8_token": "20261018-193816-m10h-confirm-grid-qwen-int8_token",
                         "int4_kivi": "20261018-195840-m10h-confirm-grid-qwen-int4_kivi"}}}[PL]
BENCH_RID = {"mi300x": "20261016-130602-m10-dqfast-bench",   # RUNLOG_MI300X 11.2f、11_issues 問題一 §4（檔內沒寫 run_id）
             "h200": "20261018-143442-m10h-dqfast-bench"}[PL]  # RUNLOG_H200（job 20912 的 DQ-BENCH；檔內沒寫 run_id）
ORIGIN_MAIN = "origin/main"          # stage 43 速度格子只在 origin/main（本地 ref，最後一次 fetch：commit 1ddba0b，10/17 13:44Z）

T, D = 512, 128
SHAPE = {"llama": (32, 8), "qwen": (28, 4)}            # (層數 L, KV head 數 H)；m10_dqsweep.SHAPES 同
MAIN_FP8 = {"llama": "fp8_token", "qwen": "fp8_head"}
FP8K0 = {"llama": "fp8k0_token", "qwen": "fp8k0_head"}
OTHER_FP8 = {"llama": "fp8_head", "qwen": "fp8_token"}

# (格式, 名稱, 家族, 在 09 清單, 04 §1／§3 裡的角色)
def row_specs(m: str):
    return [
        ("bf16", "BF16（不壓）", "16-bit", True, "C0；品質基準 E0"),
        (MAIN_FP8[m], f"FP8 e4m3 主線粒度（{'每 token' if m == 'llama' else '每 head'} scale）", "8-bit", True,
         "8-bit（04 §1：Llama 用 FP8）" if m == "llama" else "不用（04 §1：Qwen 的 FP8 掉 32.8 分）"),
        ("fp8e5_head", "FP8 e5m2 每 head", "8-bit", True, "09 大掃描格式（04 沒用）"),
        (FP8K0[m], "FP8-K0（K 先減每 channel 平均）", "8-bit", True, "09 大掃描格式（04 沒用）"),
        ("int8_token", "INT8 每 token", "8-bit", True,
         "8-bit 候選（04 §1 Llama 選 FP8）" if m == "llama" else "8-bit（04 §1：Qwen 用 INT8）"),
        ("kv-int8_token-int4_kivi", "K8V4（K INT8 每 token、V INT4 KIVI）", "mixed", True, "09 大掃描格式（04 沒用）"),
        ("int4_kivi", "INT4 KIVI（K 每 channel、V 每 token，g32）", "4-bit", True, "INT4（04 §1；C1／C2／C4）"),
        ("int4_token", "INT4 每 token", "4-bit", True, "09 大掃描格式（04 沒用）"),
        ("int4h_token", "INT4＋Hadamard（每 token）", "4-bit", True, "09 大掃描格式（04 沒用）"),
        ("mxfp4", "MXFP4（每 32 值 E8M0 scale）", "4-bit", True, "09 大掃描格式（04 沒用）"),
        ("nvfp4", "NVFP4（每 16 值 E4M3 scale＋全域 FP32）", "4-bit", True, "09 大掃描格式（04 沒用）"),
        ("kv-int4_kivi-int2_kivi", "K4V2（K INT4 KIVI、V INT2 KIVI）", "mixed", True, "09 大掃描格式（04 沒用）"),
        ("int2_kivi", "INT2 KIVI（g32）", "2-bit", True, "INT2（04 §1、§3：O 給「不會再被讀」的）"),
        ("int2h_g32", "INT2＋Hadamard（每 token、每 32 channel）", "2-bit", True, "09 大掃描格式（04 沒用）"),
        (OTHER_FP8[m], f"FP8 e4m3 另一種粒度（{'每 head' if m == 'llama' else '每 token'}；格式小測落選）", "8-bit", False,
         "不在 09 的清單（只有計時與格式小測）"),
    ]


def cell(value=None, unit="", status="measured", run_id=None, src=None, method="", note=""):
    if value is None:
        value = NM if status == "NOT_MEASURED" else (PEND if status == "pending" else value)
    return {"value": value, "unit": unit, "status": status, "run_id": run_id, "src": src, "method": method, "note": note}


def nm(note="", unit=""):
    return cell(NM, unit, "NOT_MEASURED", None, None, "", note)


def pend(stage, note="", unit=""):
    return cell(PEND, unit, "pending", None, None, "", f"{stage}：{note}" if note else stage)


def r4(x, k=4):
    return None if x is None else round(float(x), k)


# ============================================================================ 位元組：照程式寫的公式
def half_bytes(f: str, L: int, H: int) -> int:
    """m10_codec_x.enc_half（K 或 V 一半，[L,H,T,D]）存的位元組；scale／zero／全域 scale 全部算進去。"""
    M = L * H * T * D
    if f == "bf16":
        return 2 * M                                        # {"x": bf16}
    if f == "int8_token":
        return M + 4 * L * H * T                            # int8 q＋FP32 scale 每（層,head,token）
    if f in ("int4_token", "int4h_token"):
        return M // 2 + 4 * L * H * T                       # 半 byte q＋FP32 scale 每（層,head,token）
    if f in ("fp8_head", "fp8e5_head"):
        return M + 4 * L * H                                # uint8 q＋FP32 scale 每（層,head）
    if f in ("fp8_token", "fp8e5_token"):
        return M + 4 * L * H * T
    if f == "fp8_channel":
        return M + 4 * L * H * D
    if f == "mxfp4":
        return M // 2 + M // 32                             # E2M1 半 byte＋uint8 E8M0 每 32 值
    if f == "nvfp4":
        return M // 2 + M // 16 + 4 * L                     # E2M1＋uint8 E4M3 每 16 值＋FP32 全域 scale 每層
    if f == "kivi4":
        return M // 2 + 2 * 2 * (M // 32)                   # 半 byte q＋FP16 scale＋FP16 zero 每 32 值
    if f in ("kivi2", "int2h_g32"):
        return M // 4 + 2 * 2 * (M // 32)                   # 2-bit q＋FP16 scale／zero 每 32 值
    raise ValueError(f)


ALIAS = {"int4_kivi": "kivi4", "int2_kivi": "kivi2"}


def block_bytes_formula(m: str, fmt: str) -> int:
    L, H = SHAPE[m]
    if fmt.startswith("fp8k0_"):                            # m10_codec：fp8_<g> 的 q／scale＋K 平均（BF16，每 層,head,channel）
        return 2 * half_bytes("fp8_" + fmt[6:], L, H) + 2 * L * H * D
    if fmt.startswith("kv-"):
        rest = fmt[3:]
        for k in ("int8_token", "int4_kivi", "int4_token", "int2_kivi"):
            if rest.startswith(k + "-"):
                kf, vf = k, rest[len(k) + 1:]
                break
        else:
            raise ValueError(fmt)
    else:
        kf = vf = fmt
    return half_bytes(ALIAS.get(kf, kf), L, H) + half_bytes(ALIAS.get(vf, vf), L, H)


def bytes_encode_cpu(m: str, fmts: list[str]) -> dict:
    """真的在 CPU 上呼叫 m10_codec_x.encode（主線格式會轉手給 m10_codec，逐位元相同）取 Enc.nbytes。"""
    import torch
    import m10_codec_x as X
    assert not torch.cuda.is_available(), "這支只准用 CPU（TIARA_NO_GPU=1）"
    L, H = SHAPE[m]
    g = torch.Generator().manual_seed(0)
    x = (torch.randn((L, 2, H, T, D), generator=g) * 2).to(torch.bfloat16)
    x[:, 0, :, :, 5] += 40                                   # K 的離群通道（同 m10_dqsweep A 段）
    return {f: int(X.encode(x, f).nbytes) for f in fmts}


# ============================================================================ 讀檔小工具
def rp(p):
    return os.path.join(REPO, p)


def jload(p):
    return json.load(open(rp(p)))


def git_show_csv(ref: str, path: str) -> pd.DataFrame:
    raw = subprocess.run(["git", "-C", REPO, "show", f"{ref}:{path}"], capture_output=True, check=True).stdout
    if path.endswith(".gz"):
        raw = gzip.decompress(raw)
    return pd.read_csv(io.BytesIO(raw))


def git_show_json(ref: str, path: str) -> dict:
    raw = subprocess.run(["git", "-C", REPO, "show", f"{ref}:{path}"], capture_output=True, check=True).stdout
    return json.loads(raw)


# ============================================================================ 還原中的反量化（每個 block）
def inrestore(df: pd.DataFrame, run_id: str, config: str, mode: str, gpu: str, fmt: str | None = None,
              bw: str = "cpu", length: int = 32768, lowcol: str = "fp8_read"):
    q = df[(df.run_id == run_id) & (df.length == length) & (df.bw_name == bw) & (df.config == config)
           & (df["mode"] == mode) & (df.gpu_state == gpu)]
    if fmt is not None:
        q = q[q.fmt == fmt]
    if len(q) == 0:
        return None
    contam = q.contaminated_gpu.astype(str).str.lower().eq("true").sum()
    kvbad = int(pd.to_numeric(q.kv_bad, errors="coerce").fillna(0).sum())
    assert contam == 0, f"{run_id} {config} {mode} {gpu}：有 {contam} 列被標污染"
    assert kvbad == 0, f"{run_id}：kv_bad={kvbad}"
    nlow = q[lowcol].astype(float)
    nlow = nlow.where(nlow > 0)
    per = (q.t_dequant.astype(float) / nlow).median() * 1000
    return dict(ms=r4(per), n=int(len(q)), nlow_median=float(nlow.median()),
                t_dequant_median_ms=r4(q.t_dequant.astype(float).median() * 1000, 3),
                bytes_stored_median=float(q.bytes_stored.median()) if "bytes_stored" in q else None,
                n_blocks=int(q.n_blocks.iloc[0]) if "n_blocks" in q else None)


# ============================================================================ 主程式
def main():
    os.makedirs(OUT, exist_ok=True)
    crosschecks, problems = [], []
    tables = {}
    for m in ("llama", "qwen"):
        specs = row_specs(m)
        fmts = [s[0] for s in specs]
        L, H = SHAPE[m]
        bf16_block = block_bytes_formula(m, "bf16")
        assert bf16_block == L * 2 * H * T * D * 2

        # ---------------- 位元組
        enc = bytes_encode_cpu(m, fmts)
        parts = jload(f"{R10S}/parts/parts_{m}.json")
        parts_src = f"{R10S}/parts/parts_{m}.json"
        assert parts["contaminated"] is False
        pmed = parts["median_per_block"]
        # 海選彙整（ALL＝block 0 BF16＋block 1–63 低精度）
        agg = pd.read_csv(rp(f"{R10S}/analysis/agg_{m}.csv.gz"))
        screen_runs = SCREEN_RUNS[m]
        agg_src = f"{R10S}/analysis/agg_{m}.csv.gz"
        e0agg = agg[(agg.policy == "E0") & (agg.readout == "A")].iloc[0]

        # ---------------- 主線還原格子（K0）＋2b＋42（融合）＋43（origin/main，K0）
        # run_id：MI300X 的原值經 PF.run（P3_CONST=h200 時換成 H200 的同一格）
        if m == "llama":
            g_fp8 = (f"{R10}/speed_llama/restores.csv.gz", PF.run("20261015-161226-m10-grid-fp8-llama"))
            g_int4 = (f"{R10}/speed_llama/restores.csv.gz", PF.run("20261016-034902-m10-grid-int4-llama"))
            g_2b = {"fp8k0_token": (f"{R10}/speed_llama/restores.csv.gz", PF.run("20261015-191338-m10-grid-2b-fp8k0_token-llama")),
                    "int8_token": (f"{R10}/speed_llama/restores.csv.gz", PF.run("20261015-192112-m10-grid-2b-int8_token-llama"))}
            f42 = {"fp8_token": (f"{R10S}/speed_dqfast_llama/restores.csv", PF.run("20261016-232855-m10-dqfast-grid-llama")),
                   "int4_kivi": (f"{R10S}/speed_dqfast_int4_llama/restores.csv", PF.run("20261016-235027-m10-dqfast-grid-int4-llama"))}
        else:
            g_fp8 = (f"{R10}/{QRG}/restores.csv.gz", PF.run("20261015-194150-m10-regrid-fp8-qwen"))
            g_int4 = (f"{R10}/speed_qwen/restores.csv.gz", PF.run("20261016-052143-m10-grid-int4-qwen"))
            g_2b = {"fp8k0_head": (f"{R10}/speed_qwen/restores.csv.gz", PF.run("20261015-192845-m10-grid-2b-fp8k0_head-qwen")),
                    "int8_token": (f"{R10}/{QRG}/restores.csv.gz", PF.run("20261015-210456-m10-regrid-2b-int8_token-qwen"))}
            f42 = {"fp8_head": (f"{R10S}/speed_dqfast_qwen/restores.csv", PF.run("20261016-234030-m10-dqfast-grid-qwen")),
                   "int4_kivi": (f"{R10S}/speed_dqfast_int4_qwen/restores.csv", PF.run("20261017-000209-m10-dqfast-grid-int4-qwen"))}
        g43 = G43[m]
        cache = {}

        def load(path):
            if path not in cache:
                cache[path] = pd.read_csv(rp(path))
            return cache[path]
        p43 = f"{R10S}/confirm/speed_{m}/restores.csv.gz"
        if PL != "mi300x":         # H200 的 43 在這個分支裡（job 20917），直接讀檔
            d43 = pd.read_csv(rp(p43)) if os.path.exists(rp(p43)) else None
            g43_src = p43
            if d43 is None:
                problems.append(f"{m}: 讀不到 {p43} → stage 43 的還原中反量化寫 NOT_MEASURED")
        else:
            try:
                d43 = git_show_csv(ORIGIN_MAIN, p43)
                g43_src = f"git show {ORIGIN_MAIN}:{p43}"
            except subprocess.CalledProcessError as ex:
                d43 = None
                problems.append(f"{m}: 讀不到 {ORIGIN_MAIN}:{p43}（{ex}）→ stage 43 的還原中反量化寫 NOT_MEASURED")

        # 43 的 gpu_guard（origin/main；H200 讀本地檔）
        g43_guard = {}
        if d43 is not None:
            for f, rid in g43.items():
                vs = []
                for li in range(3):
                    gp = f"{R10S}/confirm/speed_{m}/gpu_guard_{rid}_L{li}.json"
                    vs.append((jload(gp) if PL != "mi300x" else git_show_json(ORIGIN_MAIN, gp)).get("verdict"))
                g43_guard[f] = vs
                if any(v != "CLEAN" for v in vs):
                    problems.append(f"{m} 43 {f}: gpu_guard {vs}")

        # ---------------- 42b A 段（kernel 微測）
        A_csv = f"{R10S}/dqsweep/A/kernels_{m}.csv"
        A = pd.read_csv(rp(A_csv))
        A_choice = jload(f"{R10S}/dqsweep/A/choice_{m}.json")
        assert A_choice["contaminated"] is False and set(A.contaminated_gpu.astype(str)) == {"False"}
        A_rid = A_choice["run_id"]
        amed = {(r.fmt, r.kernel, r.gpu_state): float(r.median_ms) for r in A.itertuples()}
        # ---------------- stage 40 bench（和 3090 同一支 m10_dequant_fast --self-test --bench；閒；檔內沒有 run_id）
        bench = jload(f"{R10S}/parts/dequant_fast_bench.json")["bench"]
        bench_rid = BENCH_RID
        # ---------------- 3090（H200 的表沒有 3090 欄：memory h200-phase1-reproduction「只有 H200 平台」）
        if PL == "mi300x":
            dq3 = pd.read_csv(rp(f"{X7}/dequant_bench_3090.csv"))
            dq3 = dq3[dq3["shape"].astype(str) == f"{m}_block"]
            dq3_rid = "20261017-180859-p3x7-dqbench"
            assert set(dq3.run_id) == {dq3_rid} and set(dq3.guard) == {"CLEAN"}
            c3090 = jload(f"{X7}/constants_3090.json")
            raw3090 = "/ssd8/hungwei/paper-hkv/runs/20261017-181306-p3x7-grid-qwen-L0/restores.csv"
            r3 = pd.read_csv(raw3090) if (m == "qwen" and os.path.exists(raw3090)) else None
            if r3 is not None:
                g3 = json.load(open(os.path.join(os.path.dirname(raw3090), "gpu_guard.json")))
                assert g3["verdict"] == "CLEAN"
        # ---------------- 品質
        qgen = {}
        if m == "llama":
            qgen[MAIN_FP8[m]] = (f"{R10}/analysis/fp8/quality_llama_step1.json", "E1", "主線 Step 1（08 §2.3）")
            qgen[FP8K0[m]] = ("results/p3_formats/derived/quality_llama_step2b.json", "E1s@fp8k0_token",
                              "本表補算（凍結的 m10_analyze，第二階段沒報過；run 20261018-002550-p3fmt-qderive）")
            qgen["int8_token"] = ("results/p3_formats/derived/quality_llama_step2b.json", "E1s@int8_token",
                                  "本表補算（凍結的 m10_analyze，第二階段沒報過；run 20261018-002550-p3fmt-qderive）")
            qgen["int4_kivi"] = (f"{R10}/analysis/int4/quality_llama_int4.json", "E1", "主線 INT4 端點（08 §2.5.3）")
        else:
            qgen[MAIN_FP8[m]] = (f"{R10}/analysis/fp8/quality_qwen_step1.json", "E1", "主線 Step 1（08 §2.3）")
            qgen[FP8K0[m]] = (f"{R10}/analysis/fp8/quality_qwen_step2b.json", "E1s@fp8k0_head", "主線 Step 2b（08 §2.4）")
            qgen["int8_token"] = (f"{R10}/analysis/fp8/quality_qwen_step2b.json", "E1s@int8_token", "主線 Step 2b（08 §2.4）")
            qgen["int4_kivi"] = (f"{R10}/analysis/int4/quality_qwen_int4.json", "E1", "主線 INT4 端點（08 §2.5.3）")
        if PL != "mi300x":
            # H200 的主線走到 FP8 階（RUNLOG_H200 10.12：K3 不成立），INT4 端點的品質在「照 MI300X 路徑」的對照分析裡
            # （同一支 m10_analyze、同一批 INT4 列；只是判定樹的標籤不同）。Llama 的 FP8-K0／INT8 由 derive_quality.py 在 H200 的列上補算
            qgen["int4_kivi"] = (f"{R10}/analysis/compare_mi300x_path/int4/quality_{m}_int4.json", "E1",
                                 "INT4 端點（H200：analysis/compare_mi300x_path，08 §2.5.3 的同一個分析）")
            if m == "llama":
                for f_, cfg_ in ((FP8K0[m], "E1s@fp8k0_token"), ("int8_token", "E1s@int8_token")):
                    qgen[f_] = (f"{os.path.relpath(OUT, REPO)}/derived/quality_llama_step2b.json", cfg_,
                                "H200 補算（code/p3_formats/derive_quality.py，P3_CONST=h200；凍結的 m10_analyze）")
        fmtsel = jload(f"{R10}/fmt_select/fmt_choice.json")
        # 43 複賽（生成品質）會量的 E1s@格式（confirm_plan.json）
        plan43 = jload(f"{R10S}/confirm_plan.json")["models"][m]["configs"]
        fmts43 = sorted({c.split("@", 1)[1] for c in plan43 if c.startswith("E1s@")})
        qgen_missing = {}
        if PL != "mi300x":   # H200：補算的檔不在（derive_quality 的重現檢查沒過就不寫）→ 這格寫 NOT_MEASURED 附原因，不猜
            for f_ in [f_ for f_, v_ in qgen.items() if not os.path.exists(rp(v_[0]))]:
                qgen_missing[f_] = f"{qgen.pop(f_)[0]} 不存在：H200 的 derive_quality 沒有產出（重現檢查沒過，見 RUNLOG_H200_P3）"
                problems.append(f"{m} {f_}: {qgen_missing[f_]}")

        rows = []
        for fmt, label, fam, in09, role in specs:
            c = {}
            # ---------------- 位元組
            fb = block_bytes_formula(m, fmt)
            c["bytes_per_block_code"] = cell(fb, "B", run_id=RUN_ID, src="code/m10_codec.py＋code/m10_codec_x.py（公式寫在 build_format_table.half_bytes）",
                                             method="照程式的 tensor 形狀＋dtype 算（含 scale／zero／全域 scale）")
            c["bytes_per_block_encode_cpu"] = cell(enc[fmt], "B", run_id=RUN_ID, src="m10_codec_x.encode(x).nbytes（本 run，CPU，合成 BF16 block）",
                                                   method="真的編碼一次取 Enc.nbytes")
            if enc[fmt] != fb:
                problems.append(f"{m} {fmt}: 公式 {fb} ≠ encode {enc[fmt]}")
            meas = {}
            pk = pmed.get(f"{fmt}|idle")
            if pk is not None:
                c[f"bytes_per_block_{PL}_parts"] = cell(int(pk["bytes"]), "B", run_id=parts["run_id"], src=parts_src,
                                                         method="MI300X 真實 KV，Enc.nbytes 的中位數（64 block × 3 次）")
                meas["parts"] = int(pk["bytes"])
            else:
                c[f"bytes_per_block_{PL}_parts"] = nm("parts 沒有這個格式")
            # 主線還原格子的 bytes_stored
            rs = None
            if fmt == "bf16":
                x = inrestore(load(g_fp8[0]), g_fp8[1], "E0", "load_only", "idle")
                rs = (x["bytes_stored_median"] / x["n_blocks"], g_fp8, "E0 load_only bytes_stored ÷ 64")
            elif fmt == MAIN_FP8[m]:
                x = inrestore(load(g_fp8[0]), g_fp8[1], "E1", "load_only", "idle")
                rs = (x["bytes_stored_median"] / x["n_blocks"], g_fp8, "E1 load_only bytes_stored ÷ 64")
            elif fmt == "int4_kivi":
                x = inrestore(load(g_int4[0]), g_int4[1], "E1", "load_only", "idle")
                rs = (x["bytes_stored_median"] / x["n_blocks"], g_int4, "E1 load_only bytes_stored ÷ 64")
            elif fmt in g_2b:
                x = inrestore(load(g_2b[fmt][0]), g_2b[fmt][1], "E1s", "cake", "idle", bw="cpu")
                rs = ((x["bytes_stored_median"] - bf16_block) / (x["n_blocks"] - 1), g_2b[fmt],
                      "E1s cake (bytes_stored − 1 個 BF16 block) ÷ 63")
            if rs is not None:
                c[f"bytes_per_block_{PL}_restores"] = cell(r4(rs[0], 2), "B", run_id=rs[1][1], src=rs[1][0], method=rs[2])
                meas["restores"] = rs[0]
            elif d43 is not None and fmt in g43:
                x = inrestore(d43, g43[fmt], "E1s", "cake", "idle", fmt=fmt)
                v = (x["bytes_stored_median"] - bf16_block) / (x["n_blocks"] - 1)
                c[f"bytes_per_block_{PL}_restores"] = cell(r4(v, 2), "B", run_id=g43[fmt], src=g43_src,
                                                            method="stage 43 E1s cake (bytes_stored − 1 個 BF16 block) ÷ 63")
                meas["restores"] = v
            else:
                c[f"bytes_per_block_{PL}_restores"] = nm("沒有這個格式的還原格子")
            # 海選
            if fmt == "bf16":
                v = float(e0agg.bytes_stored) / 64
                c[f"bytes_per_block_{PL}_screen"] = cell(r4(v, 2), "B", run_id=";".join(screen_runs), src=agg_src,
                                                          method="E0 bytes_stored ÷ 64（agg 檔沒有 run_id 欄，run_id 取自 tables_<m>.md）")
                meas["screen"] = v
            else:
                a = agg[(agg.fmt == fmt) & (agg.policy == "ALL") & (agg.readout == "A")]
                if len(a):
                    v = (float(a.iloc[0].bytes_stored) - bf16_block) / 63
                    c[f"bytes_per_block_{PL}_screen"] = cell(r4(v, 2), "B", run_id=";".join(screen_runs), src=agg_src,
                                                              method="ALL (bytes_stored − 1 個 BF16 block) ÷ 63（agg 檔沒有 run_id 欄）")
                    meas["screen"] = v
                else:
                    c[f"bytes_per_block_{PL}_screen"] = nm("海選沒有這個格式")
            # 3090（H200 的表沒有）
            if PL != "mi300x":
                pass
            elif fmt == "bf16":
                v = c3090["models"][m]["chunk_bytes"]
                c["bytes_per_block_3090"] = cell(v, "B", run_id=c3090["models"][m]["f_src_run"], src=f"{X7}/constants_3090.json",
                                                 method="models.<m>.chunk_bytes（calib-f）")
                meas["3090"] = v
            elif r3 is not None and fmt in ("fp8_head", "int4_kivi"):
                q = r3[(r3.config == "E1") & (r3.fmt == fmt) & (r3["mode"] == "load_only") & (r3.tier == "cpu")]
                v = float(q.bytes_read.median()) / 64
                c["bytes_per_block_3090"] = cell(r4(v, 2), "B", run_id="20261017-181306-p3x7-grid-qwen-L0", src=raw3090,
                                                 method="E1 load_only bytes_read ÷ 64（run 目錄的原始 restores.csv）")
                meas["3090"] = v
            else:
                c["bytes_per_block_3090"] = nm("3090 沒有這個格式的還原格子")
            agree = all(abs(v - fb) < 0.5 for v in meas.values())
            if not agree:
                problems.append(f"{m} {fmt}: 實際大小和公式不同 {meas} vs {fb}")
            c["bytes_measured_agree_with_code"] = cell(bool(agree), "", run_id=RUN_ID, src="本表",
                                                       method=f"比了 {sorted(meas)}；全部 == 公式（±0.5 B）")
            c["bytes_basis"] = cell("code+measured" if meas else "code_only", "", run_id=RUN_ID, src="本表",
                                    method="code＝公式與 encode；measured＝MI300X／3090 存下來的大小")
            c["bytes_per_token"] = cell(r4(fb / T, 3), "B/token", run_id=RUN_ID, src="bytes_per_block_code ÷ 512",
                                        method="每 512-token block 攤（per-head／per-layer 的 scale 依 block 長度攤；KIVI 每 32 token 一組）")
            c["kib_per_token"] = cell(r4(fb / T / 1024, 4), "KiB/token", run_id=RUN_ID, src="bytes_per_block_code ÷ 512 ÷ 1024")
            c["bits_per_value"] = cell(r4(fb * 8 / (L * 2 * H * T * D), 4), "bit", run_id=RUN_ID, src="bytes_per_block_code×8 ÷ 值數")
            c["ratio_vs_bf16"] = cell(r4(fb / bf16_block, 5), "", run_id=RUN_ID, src="bytes_per_block_code ÷ BF16")

            # ---------------- MI300X parts（單獨、真實 KV、每 block）
            for st in ("idle", "busy"):
                pk = pmed.get(f"{fmt}|{st}")
                for key, col, what in ((f"dq_{PL}_k0", "t_dequant_torch_ms", "原寫法 K0（PyTorch 一步一步）"),
                                       (f"dq_{PL}_fused", "t_dequant_fused_ms", "融合 K1（m10_dequant_fast，Triton）"),
                                       (f"enc_{PL}", "t_encode_ms", "編碼（寫入時）"),
                                       (f"h2d_{PL}", "t_h2d_ms", "pinned → GPU（單一 block，不是還原中）")):
                    k = f"{key}_{st}_ms"
                    if fmt == "bf16" and key in (f"dq_{PL}_k0", f"dq_{PL}_fused", f"enc_{PL}"):
                        c[k] = cell(0.0, "ms", run_id=parts["run_id"], src=parts_src, method="BF16 不用編碼／反量化",
                                    note="定義為 0")
                        continue
                    v = None if pk is None else pk.get(col)
                    if v is None:
                        note = ("融合 kernel 只支援 fp8_head／channel／token、int8_token、int4_token、int4_kivi（11 問題一 §3）；"
                                "新格式的融合 kernel 計時是 43d（13 號 Q3）→ PENDING") if key == f"dq_{PL}_fused" else "parts 沒有這格"
                        if key == f"dq_{PL}_fused" and fmt in g43 and fmt != "int4_kivi":
                            note += "；這個格式的融合 kernel 只在 stage 43d 的還原中量（見 dq_mi300x_cake_fused_*），沒有單獨量"
                        c[k] = nm(note, "ms")
                    else:
                        c[k] = cell(r4(v), "ms", run_id=parts["run_id"], src=parts_src,
                                    method=f"{what}；parts（stage 40）每 block 中位數（64 block × 3 次），GPU {st}"
                                           + ("（背景 matmul duty 0.7）" if st == "busy" else ""))
            # ---------------- 42b A 段
            for kname, kk in (("k0", "k0"), ("k1", "k1"), ("k2", "k2")):
                for st in ("idle", "busy"):
                    v = amed.get((fmt, kk, st))
                    key = f"dqA_{PL}_{kname}_{st}_ms"
                    c[key] = (cell(r4(v, 5), "ms", run_id=A_rid, src=A_csv,
                                   method=f"42b A 段微測：合成資料寫進整個 KV 視圖，暖機 5、量 50 取中位數，GPU {st}")
                              if v is not None else nm("A 段只量 fp8_token／fp8_head／int8_token／int4_token／int4_kivi", "ms"))
            ks = A_choice["choice"].get(fmt)
            c[f"dqA_{PL}_kstar"] = cell(ks, "", run_id=A_rid, src=f"{R10S}/dqsweep/A/choice_{m}.json") if ks else nm("A 段沒有這個格式")
            for st in ("idle", "busy"):
                v = amed.get((fmt, ks, st)) if ks else None
                c[f"dqA_{PL}_kstar_{st}_ms"] = (cell(r4(v, 5), "ms", run_id=A_rid, src=A_csv, method=f"K*＝{ks}（09 §6.4.3-1 的挑法）")
                                                 if v is not None else nm("A 段沒有這個格式", "ms"))
            # ---------------- stage 40 bench（閒）
            for impl, key in (("torch", f"dqbench_{PL}_k0_idle_ms"), ("fused", f"dqbench_{PL}_fused_idle_ms"),
                              ("torch_compile", f"dqbench_{PL}_compile_idle_ms")):
                b = bench.get(f"{m}_block|{fmt}|{impl}")
                c[key] = (cell(r4(b["median_ms"]), "ms", run_id=bench_rid, src=f"{R10S}/parts/dequant_fast_bench.json",
                               method="m10_dequant_fast.py --self-test --bench（和 3090 同一支），GPU 閒；檔內沒有 run_id、"
                                      "這支自測沒開 gpu_guard（11 問題一 §4）")
                          if b else nm("stage 40 bench 只量 fp8_head／fp8_token／int8_token／int4_token／int4_kivi", "ms"))
            # ---------------- 還原中（真實 CPU 層，32K）
            src_k0 = None
            if fmt == MAIN_FP8[m]:
                src_k0 = (g_fp8, "E1")
            elif fmt == "int4_kivi":
                src_k0 = (g_int4, "E1")
            elif fmt in g_2b:
                src_k0 = (g_2b[fmt], "E1s")
            for mode, mk in (("cake", "cake"), ("load_only", "loadonly")):
                for st in ("idle", "busy"):
                    key = f"dq_{PL}_{mk}_k0_{st}_ms"
                    if fmt == "bf16":
                        c[key] = cell(0.0, "ms", src="—", method="BF16 不反量化", note="定義為 0")
                        c[f"dq_{PL}_{mk}_fused_{st}_ms"] = cell(0.0, "ms", src="—", method="BF16 不反量化", note="定義為 0")
                        continue
                    x = None
                    if src_k0 is not None:
                        (path, rid), cfg = src_k0
                        x = inrestore(load(path), rid, cfg, mode, st)
                        if x is not None:
                            c[key] = cell(x["ms"], "ms", run_id=rid, src=path,
                                          method=f"{cfg} {mode}，CPU 層 32K，median(t_dequant ÷ 讀到的低精度 block 數)，"
                                                 f"n={x['n']}，讀到 {x['nlow_median']:.0f} 個；原寫法 K0")
                    if x is None and d43 is not None and fmt in g43 and mode == "cake":
                        x = inrestore(d43, g43[fmt], "E1s", "cake", st, fmt=fmt)
                        if x is not None:
                            c[key] = cell(x["ms"], "ms", run_id=g43[fmt], src=g43_src,
                                          method=f"stage 43 速度格子 E1s cake，CPU 層 32K，median(t_dequant ÷ 讀到的低精度 block 數)，"
                                                 f"n={x['n']}；原寫法 K0（43 不用融合 kernel）",
                                          note=("只在 origin/main（本地 ref 1ddba0b）；" if PL == "mi300x" else "")
                                               + "43 的速度判讀等品質一起寫（09 §4.3），這裡只取反量化時間")
                    if x is None:
                        if mode == "load_only" and fmt in g_2b:
                            c[key] = nm("2b 的速度格子沒跑只載參考線（--refs 空）", "ms")
                        elif mode == "load_only" and fmt in g43:
                            c[key] = nm("stage 43 的速度格子沒跑只載參考線（refs＝[]）", "ms")
                        else:
                            c[key] = nm("沒有這個格式的還原格子", "ms")
                    # 融合（stage 42）
                    key = f"dq_{PL}_{mk}_fused_{st}_ms"
                    if fmt in f42:
                        path, rid = f42[fmt]
                        x = inrestore(load(path), rid, "E1", mode, st)
                        c[key] = (cell(x["ms"], "ms", run_id=rid, src=path,
                                       method=f"stage 42（M10_DEQUANT_FAST=1，K1，S0），E1 {mode}，CPU 層 32K，"
                                              f"median(t_dequant ÷ 讀到的低精度 block 數)，n={x['n']}")
                                  if x is not None else nm("stage 42 沒有這格", "ms"))
                    elif fmt in g43 and fmt != "int4_kivi" and mode == "cake":
                        c[key] = pend("stage 43d（13 號 Q3）", "新格式融合 kernel 的 CPU 層還原，跑 43 的速度配置（含 E1s）", "ms")
                    else:
                        c[key] = nm("stage 42 只跑主線 FP8 與 INT4 KIVI", "ms")
            # ---------------- 3090（H200 的表沒有）
            if PL == "mi300x":
                for impl, key in (("torch", "dq_3090_k0_idle_ms"), ("fused", "dq_3090_fused_idle_ms"),
                                  ("torch_compile", "dq_3090_compile_idle_ms")):
                    q = dq3[(dq3.fmt == fmt) & (dq3.impl == impl)]
                    if fmt == "bf16" and impl != "torch_compile":
                        c[key] = cell(0.0, "ms", src="—", method="BF16 不反量化", note="定義為 0")
                    elif len(q) and str(q.iloc[0].median_ms) != NM:
                        c[key] = cell(r4(float(q.iloc[0].median_ms)), "ms", run_id=dq3_rid, src=f"{X7}/dequant_bench_3090.csv",
                                      method="m10_dequant_fast.py --self-test --bench（GPU 閒，單獨量）；guard CLEAN")
                    elif len(q):
                        c[key] = nm("torch.compile 在 sm_86 編不起 e4m3fn（X7 §4.3，完整錯誤在 CSV 的 error 欄）", "ms")
                    else:
                        c[key] = nm("X7 bench 只量 fp8_head／fp8_token／int8_token／int4_token／int4_kivi", "ms")
                for key in ("dq_3090_k0_busy_ms", "dq_3090_fused_busy_ms"):
                    c[key] = (cell(0.0, "ms", src="—", method="BF16 不反量化", note="定義為 0") if fmt == "bf16"
                              else nm("X7 的 bench 只在 GPU 閒時量（沒有忙的單獨量測）", "ms"))
                for mode, mk in (("cake", "cake"), ("load_only", "loadonly")):
                    for st in ("idle", "busy"):
                        key = f"dq_3090_{mk}_k0_{st}_ms"
                        if fmt == "bf16":
                            c[key] = cell(0.0, "ms", src="—", method="BF16 不反量化", note="定義為 0")
                            continue
                        if r3 is not None and fmt in ("fp8_head", "int4_kivi"):
                            lowc = "n_load_fp8" if fmt.startswith("fp8") else "n_load_int4"
                            q = r3[(r3.config == "E1") & (r3.fmt == fmt) & (r3["mode"] == mode) & (r3.tier == "cpu")
                                   & (r3.gpu_state == st)]
                            assert len(q) and not q.contaminated_gpu.astype(str).str.lower().eq("true").any()
                            nlow = q[lowc].astype(float).where(q[lowc] > 0)
                            v = (q.t_dequant / nlow).median() * 1000
                            x7v = c3090["models"]["qwen"]["dequant_ms_per_block_in_restore_cpu"].get(f"{fmt}|{mode}|{st}")
                            c[key] = cell(r4(v), "ms", run_id="20261017-181306-p3x7-grid-qwen-L0", src=raw3090,
                                          method=f"E1 {mode}，CPU 層（3090 真的 H2D）32K，median(t_dequant ÷ 讀到的低精度 block 數)，n={len(q)}（1 launch × 5）；原寫法",
                                          note=f"X7 constants 用中位數之比＝{x7v}；host_level＝HEAVY、同 switch 鄰卡有 PCIe 流量（X7 §4.4）")
                            crosschecks.append({"what": f"3090 {m} {fmt} {mode} {st} 還原中每 block 反量化", "ours_median_of_ratio_ms": r4(v),
                                                "x7_ratio_of_medians_ms": x7v, "src_x7": f"{X7}/constants_3090.json"})
                        elif m == "llama" and fmt in ("fp8_token", "int4_kivi"):
                            c[key] = nm("X7 的 Llama 還原格子兩次被別人的行程污染（runs.json rejected）", "ms")
                        else:
                            c[key] = nm("X7 的還原格子只跑 Qwen 的 fp8_head、int4_kivi", "ms")
                for mode, mk in (("cake", "cake"), ("load_only", "loadonly")):
                    for st in ("idle", "busy"):
                        c[f"dq_3090_{mk}_fused_{st}_ms"] = (cell(0.0, "ms", src="—", method="BF16 不反量化", note="定義為 0") if fmt == "bf16"
                                                            else nm("X7 沒有跑融合 kernel 的還原格子（runs.json grid_fused＝[]）", "ms"))

            # ---------------- 品質：生成（主線）
            if fmt == "bf16":
                c["q_gen_metric"] = cell("reference", "", src="—", method="E0 本身就是基準")
                for cond in ("A", "B_idle", "B_busy", "B_i2b"):
                    for s in ("diff_pts", "ci_lo_pts", "ci_hi_pts"):
                        c[f"q_gen_{cond}_{s}"] = cell(0.0, "pt", src="—", note="E0 對自己")
                    c[f"q_gen_{cond}_pass2"] = cell(True, "", src="—", note="E0 對自己")
                c["q_gen_eps_pass_all4"] = cell(True, "", src="—", note="E0 對自己")
                c["q_gen_mean_kl_A"] = cell(0.0, "", src="—")
                c["q_gen_same_as_e0_A"] = cell(1.0, "", src="—")
                c["q_gen_config"] = cell("E0", "", src="—")
                c["q_gen_n_rows"] = cell(1140, "rows", src="—")
            elif fmt in qgen_missing:
                for k in ("q_gen_metric", "q_gen_config", "q_gen_eps_pass_all4", "q_gen_mean_kl_A", "q_gen_same_as_e0_A", "q_gen_n_rows"):
                    c[k] = nm(qgen_missing[fmt])
                for cond in ("A", "B_idle", "B_busy", "B_i2b"):
                    for s_ in ("diff_pts", "ci_lo_pts", "ci_hi_pts"):
                        c[f"q_gen_{cond}_{s_}"] = nm(qgen_missing[fmt], "pt")
                    c[f"q_gen_{cond}_pass2"] = nm(qgen_missing[fmt])
            elif fmt in qgen:
                path, cfg, why = qgen[fmt]
                d = jload(path)
                rid = ";".join(d["run_ids"])
                c["q_gen_metric"] = cell("gen_acc_diff_paired_item_bootstrap", "", run_id=rid, src=path,
                                         method="對 E0 的配對答對率差，以題為單位 bootstrap 10,000 次的 95% CI；T2–T5、兩輪，1,140 列／600 題", note=why)
                c["q_gen_config"] = cell(cfg, "", run_id=rid, src=path)
                ok_all = True
                for cond, ck in (("A", "A"), ("B_idle", "B-idle"), ("B_busy", "B-busy"), ("B_i2b", "B-i2b")):
                    e = d["configs"][cfg][ck]
                    c[f"q_gen_{cond}_diff_pts"] = cell(r4(100 * e["diff"], 2), "pt", run_id=rid, src=path)
                    c[f"q_gen_{cond}_ci_lo_pts"] = cell(r4(100 * e["ci95"][0], 2), "pt", run_id=rid, src=path)
                    c[f"q_gen_{cond}_ci_hi_pts"] = cell(r4(100 * e["ci95"][1], 2), "pt", run_id=rid, src=path)
                    c[f"q_gen_{cond}_pass2"] = cell(bool(e["main_pass"]), "", run_id=rid, src=path, method="CI 下界 > −2 個百分點（main_pass）")
                    ok_all = ok_all and bool(e["eps_pass"])
                c["q_gen_eps_pass_all4"] = cell(ok_all, "", run_id=rid, src=path,
                                                method="四種情境都 eps_pass（CI 下界 > −2 且每任務族每輪點估計 ≥ −5）")
                c["q_gen_mean_kl_A"] = cell(r4(d["configs"][cfg]["A"]["mean_kl_vs_e0"], 6), "", run_id=rid, src=path,
                                            method="生成的 token 上的平均 KL（m10_analyze 的 mean_kl_vs_e0）")
                c["q_gen_same_as_e0_A"] = cell(r4(d["configs"][cfg]["A"]["same_as_e0"]), "", run_id=rid, src=path)
                c["q_gen_n_rows"] = cell(int(d["configs"][cfg]["A"]["n_rows"]), "rows", run_id=rid, src=path)
            elif not in09:
                for k in ("q_gen_metric", "q_gen_config", "q_gen_eps_pass_all4", "q_gen_mean_kl_A", "q_gen_same_as_e0_A", "q_gen_n_rows"):
                    c[k] = nm("不在 09 的 13 種格式裡：43、44c 都不會量；只有格式小測的 KL（q_fmtsel_kl_A）")
                for cond in ("A", "B_idle", "B_busy", "B_i2b"):
                    for s_ in ("diff_pts", "ci_lo_pts", "ci_hi_pts"):
                        c[f"q_gen_{cond}_{s_}"] = nm("不在 09 的清單", "pt")
                    c[f"q_gen_{cond}_pass2"] = nm("不在 09 的清單")
            else:
                stage = []
                if fmt in fmts43:
                    stage.append(f"stage 43（43ab 複賽品質，E1s@{fmt}，新題 seed 50000 起，T2–T4 各 90、T5 30、兩輪）")
                stage.append("stage 44c（10 號：32K／64K／128K，每個格式的 ALL 會真的生成；32K 那點同一支程式重跑，題數由 44b 試跑決定）")
                p = "；".join(stage)
                c["q_gen_metric"] = pend(p, "還沒有生成答對率")
                c["q_gen_config"] = pend(p)
                for cond in ("A", "B_idle", "B_busy", "B_i2b"):
                    for s in ("diff_pts", "ci_lo_pts", "ci_hi_pts"):
                        c[f"q_gen_{cond}_{s}"] = pend(p, unit="pt") if (cond == "A" or fmt in fmts43) else nm(
                            "44c 只用讀法 A 生成（10 §3.1）；43 沒有這個格式", "pt")
                    c[f"q_gen_{cond}_pass2"] = pend(p) if (cond == "A" or fmt in fmts43) else nm("同上")
                c["q_gen_eps_pass_all4"] = pend(p) if fmt in fmts43 else nm("44c 只有讀法 A；四種情境的判準只有 43 的格式會有")
                c["q_gen_mean_kl_A"] = pend(p)
                c["q_gen_same_as_e0_A"] = pend(p)
                c["q_gen_n_rows"] = pend(p)

            # ---------------- 品質：teacher forcing（41 海選）
            tf_note = "first_ok＝第一個答案 token 對不對（便宜的代理指標，不是生成答對率）；96 題、只有第 1 輪；沒有 CI（逐題 CSV 在 MI300X，不在 repo）"
            if fmt == "bf16":
                for rd in ("A", "B_idle", "B_busy"):
                    c[f"q_tf_{rd}_dfirst_ok_pts"] = cell(0.0, "pt", src="—", note="E0 對自己")
                c["q_tf_A_kl"] = cell(0.0, "", src="—")
                c["q_tf_A_dgold_lp"] = cell(0.0, "", src="—")
                c["q_tf_A_pass2_point"] = cell(True, "", src="—")
            else:
                a = agg[(agg.fmt == fmt) & (agg.policy == "ALL")]
                if len(a):
                    for rd in ("A", "B_idle", "B_busy"):
                        rr = a[a.readout == rd]
                        c[f"q_tf_{rd}_dfirst_ok_pts"] = (cell(r4(100 * float(rr.iloc[0].d_first_ok), 2), "pt", run_id=";".join(screen_runs),
                                                              src=agg_src, method=f"ALL（block 1–63）讀法 {rd}；B 用推算的 b32 會合點",
                                                              note=tf_note) if len(rr) else nm("", "pt"))
                    ra = a[a.readout == "A"].iloc[0]
                    c["q_tf_A_kl"] = cell(r4(float(ra.kl), 6), "", run_id=";".join(screen_runs), src=agg_src,
                                          method="標準答案每個位置的下一個 token 分布 KL（E0 對此配置）的平均")
                    c["q_tf_A_dgold_lp"] = cell(r4(float(ra.d_gold_lp), 4), "nat", run_id=";".join(screen_runs), src=agg_src,
                                                method="標準答案 log 機率（加總）對 E0 的差")
                    c["q_tf_A_pass2_point"] = cell(bool(float(ra.d_first_ok) > -0.02), "", run_id=";".join(screen_runs), src=agg_src,
                                                   method="first_ok 差的點估計 > −2 個百分點（09 §4.2-3 的前緣點規則；沒有 CI）")
                else:
                    for rd in ("A", "B_idle", "B_busy"):
                        c[f"q_tf_{rd}_dfirst_ok_pts"] = nm("海選沒有這個格式", "pt")
                    c["q_tf_A_kl"] = nm("海選沒有這個格式")
                    c["q_tf_A_dgold_lp"] = nm("海選沒有這個格式")
                    c["q_tf_A_pass2_point"] = nm("海選沒有這個格式")
            c["q_tf_first_ok_e0"] = cell(r4(float(e0agg.first_ok)), "", run_id=";".join(screen_runs), src=agg_src)
            c["q_tf_n_items"] = cell(int(e0agg.n_items), "items", run_id=";".join(screen_runs), src=agg_src)
            kl20 = fmtsel["mean_kl"][m].get(fmt)
            c["q_fmtsel_kl_A"] = (cell(r4(kl20, 6), "", run_id=";".join(r for r in fmtsel["run_ids"] if m in r),
                                       src=f"{R10}/fmt_select/fmt_choice.json", method="凍結前的格式小測：E1 讀法 A，20 題（T2–T5 各 5），平均 KL")
                                  if kl20 is not None else nm("格式小測只比 fp8_head／channel／token、int4_token、int4_kivi"))

            # ---------------- 總結：品質判定、狀態
            if fmt == "bf16":
                verdict, how = "REFERENCE", "E0"
            elif c["q_gen_eps_pass_all4"]["status"] == "measured":
                verdict = "PASS" if c["q_gen_eps_pass_all4"]["value"] else "FAIL"
                how = "生成答對率（全載入＋Cake b32 閒／忙／閒→忙）"
            elif c["q_tf_A_pass2_point"]["status"] == "measured":
                verdict = "UNVERIFIED_TF_ONLY"
                how = ("teacher forcing first_ok 點估計 " + ("過" if c["q_tf_A_pass2_point"]["value"] else "沒過")
                       + " −2（不是登記的判準）；生成答對率 PENDING")
            else:
                verdict, how = NM, "沒有任何品質數字"
            c["quality_verdict_32k"] = cell(verdict, "", run_id=RUN_ID, src="本表",
                                            method="PASS／FAIL＝第二階段判準（四種情境 eps_pass）；UNVERIFIED_TF_ONLY＝只有海選", note=how)
            dq_ok = c[f"dq_{PL}_k0_idle_ms"]["status"] == "measured" and c[f"dq_{PL}_k0_busy_ms"]["status"] == "measured"
            if fmt == "bf16" or (dq_ok and verdict in ("PASS", "FAIL")):
                status = "measured"
            elif verdict == "UNVERIFIED_TF_ONLY":
                status = "pending"
            else:
                status = NM
            import re
            pend_list = sorted({t for v in c.values() if v["status"] == "pending" for t in re.findall(r"stage ([0-9]+[a-z]*)", v["note"])})
            c["status"] = cell(status, "", run_id=RUN_ID, src="本表",
                               method="measured＝位元組＋MI300X 反量化（K0 閒／忙）＋生成品質都有；pending＝生成品質等 43／44c")
            c["pending_stages"] = cell(";".join(pend_list) if pend_list else "none", "", run_id=RUN_ID, src="本表",
                                       method="哪些 stage 會補這一列的 PENDING 格（43＝複賽品質、43d＝新格式融合 kernel、44c＝長 context 篩選含 32K 生成）")
            rows.append(dict(model=m, fmt=fmt, label=label, family=fam, in_09_list=in09, prereg_role=role, cells=c))
        tables[m] = rows

        # ---------------- 和 X5 對拍（INT4 KIVI，還原中每 block）
        if PL != "mi300x" and not os.path.exists(rp(f"{X5}/x5_cost_constants.json")):
            problems.append(f"{m}: 還沒有 {X5}/x5_cost_constants.json（X5 的 H200 重跑還沒做）→ 不對拍")
            tables[m] = rows
            continue
        x5 = jload(f"{X5}/x5_cost_constants.json")[m]
        r_int4 = next(r for r in rows if r["fmt"] == "int4_kivi")["cells"]
        for k5, kk in (("k0_idle", f"dq_{PL}_cake_k0_idle_ms"), ("k0_busy", f"dq_{PL}_cake_k0_busy_ms"),
                       ("fused_idle", f"dq_{PL}_cake_fused_idle_ms"), ("fused_busy", f"dq_{PL}_cake_fused_busy_ms")):
            ours, theirs = r_int4[kk]["value"], x5["dequant_ms"][k5]
            crosschecks.append({"what": f"{PL} {m} int4_kivi Cake CPU 層每 block 反量化 {k5}", "ours_ms": ours,
                                "x5_ms": r4(theirs), "same": abs(ours - theirs) < 5e-4, "src_x5": f"{X5}/x5_cost_constants.json",
                                "x5_run_id": jload(f"{X5}/x5_cost_constants.json")["run_id"]})
        for k5, fmtk in (("BF16", "bf16"), ("INT4", "int4_kivi")):
            ours = next(r for r in rows if r["fmt"] == fmtk)["cells"]["bytes_per_block_code"]["value"]
            crosschecks.append({"what": f"{m} {fmtk} 每 block 位元組", "ours_code_B": ours, "x5_B": x5["bytes_per_block"][k5],
                                "same": ours == x5["bytes_per_block"][k5]})
        # 43 的 int4_kivi E1s 對主線 INT4 格子（同格式、同層，不同 run）
        if d43 is not None and "int4_kivi" in g43:
            for st in ("idle", "busy"):
                a43 = inrestore(d43, g43["int4_kivi"], "E1s", "cake", st, fmt="int4_kivi")
                amain = inrestore(load(g_int4[0]), g_int4[1], "E1s", "cake", st)
                crosschecks.append({"what": f"MI300X {m} int4_kivi E1s Cake CPU 層每 block 反量化 {st}：stage 43 vs 主線 INT4 格子",
                                    "stage43_ms": a43["ms"], "main_grid_ms": amain["ms"], "run_43": g43["int4_kivi"], "run_main": g_int4[1],
                                    "rel_diff": r4(a43["ms"] / amain["ms"] - 1, 3)})
        # Qwen FP8：判定用的重量 vs 第一次量（X7 用第一次量）
        if m == "qwen" and PL == "mi300x":
            d1 = load(f"{R10}/speed_qwen/restores.csv.gz")
            for mode in ("cake", "load_only"):
                for st in ("idle", "busy"):
                    try:
                        a1 = inrestore(d1, "20261015-175050-m10-grid-fp8-qwen", "E1", mode, st)
                    except AssertionError as ex:
                        a1 = {"ms": f"有污染列：{ex}"}
                    a2 = inrestore(load(g_fp8[0]), g_fp8[1], "E1", mode, st)
                    crosschecks.append({"what": f"MI300X qwen fp8_head E1 {mode} CPU 層每 block 反量化 {st}：重量（判定用）vs 第一次量（X7 用）",
                                        "regrid_ms": a2["ms"], "first_ms": a1["ms"]})

    # ============================================================================ 輸出
    pending = quality_pending()
    notes = ["每格 {value, unit, status, run_id, src, method, note}；status＝measured／pending／NOT_MEASURED。",
             "stage 43 的還原格子讀自 git ref origin/main（本地，最後 fetch 在 commit 1ddba0b），沒有切分支。",
             "derived：Llama 的 FP8-K0／INT8 生成品質是本表用凍結的 m10_analyze 補算（20261018-002550-p3fmt-qderive），重現檢查兩項逐欄相同。"]
    if PL != "mi300x":
        # 規則和 MI300X 版逐條相同；cells 的 method／note 文字裡寫的「MI300X」指的是本表的平台 → 換掉，免得讀的人誤會資料來源（src 欄是實際路徑）
        for m_ in tables:
            for r in tables[m_]:
                for v in r["cells"].values():
                    for k_ in ("method", "note"):
                        if isinstance(v.get(k_), str):
                            v[k_] = v[k_].replace("MI300X", PL.upper())
        notes = [notes[0],
                 f"平台 {PL.upper()}（P3_CONST={PL}，code/p3_platform.py）：規則和 results/p3_formats/format_table.json（MI300X 版）逐條相同，"
                 "只換成 H200 的同一批實測；沒有 3090 欄（H200 是唯一平台）。限速層與每 block 位元組是格式定義、和 GPU 無關。",
                 f"stage 43 的還原格子讀自 {R10S}/confirm/（H200 自己的決賽，選手和 MI300X 不完全相同）；Qwen 的 FP8／INT8 用 speed_qwen（H200 沒有重量）。",
                 "INT4 端點品質取自 analysis/compare_mi300x_path（H200 的主線走 FP8 階）；Llama 的 FP8-K0／INT8 是 H200 補算（derived/）。"]
    js = {"run_id": RUN_ID, "ts": TS, "tool": "code/p3_formats/build_format_table.py",
          "shape": {m: {"L": SHAPE[m][0], "H_kv": SHAPE[m][1], "T_block": T, "D": D} for m in SHAPE},
          "rows": [r for m in tables for r in tables[m]], "crosschecks": crosschecks, "problems": problems,
          "quality_pending": pending,
          "notes": notes}
    if PL != "mi300x":
        js["platform"] = PL            # MI300X 版沒有這個鍵（輸出和原本逐位元相同）
    os.makedirs(OUT, exist_ok=True)
    json.dump(js, open(os.path.join(OUT, "format_table.json"), "w"), indent=1, ensure_ascii=False)
    # 寬表 CSV：每格三欄（值、run_id、來源）
    keys = []
    for m in tables:
        for r in tables[m]:
            keys += [k for k in r["cells"] if k not in keys]
    recs = []
    for m in tables:
        for r in tables[m]:
            rec = {"run_id": RUN_ID, "ts": TS, "model": m, "fmt": r["fmt"], "label": r["label"], "family": r["family"],
                   "in_09_list": r["in_09_list"], "prereg_role": r["prereg_role"]}
            for k in keys:
                v = r["cells"].get(k)
                rec[k] = "" if v is None else v["value"]
            for k in keys:
                v = r["cells"].get(k)
                rec[f"{k}__run_id"] = "" if v is None or v["run_id"] is None else v["run_id"]
                rec[f"{k}__src"] = "" if v is None or v["src"] is None else v["src"]
            recs.append(rec)
    pd.DataFrame(recs).to_csv(os.path.join(OUT, "format_table.csv"), index=False)
    pd.DataFrame([{"run_id": RUN_ID, "ts": TS, **p} for p in pending]).to_csv(os.path.join(OUT, "quality_pending.csv"), index=False)
    print(json.dumps({"rows": len(recs), "cols": len(recs[0]), "problems": problems}, ensure_ascii=False, indent=1))
    for cc in crosschecks:
        print("XCHK", json.dumps(cc, ensure_ascii=False))
    if problems:
        print("PROBLEMS（照實寫進 05_format_table.md）：", *problems, sep="\n  ")


def quality_pending():
    plan = jload(f"{R10S}/confirm_plan.json")
    out = []
    for m in ("llama", "qwen"):
        e1s = sorted({c.split("@", 1)[1] for c in plan["models"][m]["configs"] if c.startswith("E1s@")})
        if PL != "mi300x":     # H200 的 43 已經跑完（job 20918）；本表照 MI300X 版的規則不收 43 的品質（之後要收，兩個平台一起改）
            out.append({"stage": "43 (43ab)", "model": m, "what": "32K 生成答對率（複賽：新題 seed 50000 起，T2–T4 各 90、T5 30、兩輪；讀法 A＋B b32）",
                        "formats_uniform": ";".join(e1s), "status": "available_not_used",
                        "src": f"{R10S}/confirm_plan.json；{R10S}/confirm/confirm_{m}.json（H200 job 20918，RUNLOG_H200 10-H200）",
                        "note": "H200 有結果，但本表照 MI300X 版的規則（10/18 建表時 43 還在跑）不收；均勻（E1s）只有這些格式"})
            continue
        out.append({"stage": "43 (43ab)", "model": m, "what": "32K 生成答對率（複賽：新題 seed 50000 起，T2–T4 各 90、T5 30、兩輪；讀法 A＋B b32）",
                    "formats_uniform": ";".join(e1s), "status": "pending",
                    "src": "results/m10_sweep_mi300x/confirm_plan.json；origin/main RUNLOG 11.2o（品質 10/17 13:44Z 開跑）",
                    "note": "均勻（E1s＝block 0 BF16）只有這些格式；其他放法的配置（SW-…）不是均勻格式"})
    out.append({"stage": "43d", "model": "both", "what": "新格式融合反量化 kernel 的 CPU 層還原時間（int2_kivi、mxfp4、K4V2、K8V4）",
                "formats_uniform": "int2_kivi;mxfp4;kv-int4_kivi-int2_kivi;kv-int8_token-int4_kivi", "status": "pending",
                "src": "docs/precision_20261013/13_restore2_plan.md §5", "note": "只報告；43 的判定照原寫法"})
    out.append({"stage": "44a", "model": "both", "what": "64K／128K 速度格子（Cake、只載；FP8 主線粒度、INT4 KIVI；原寫法）",
                "formats_uniform": "fp8 主線粒度;int4_kivi", "status": "pending", "src": "docs/precision_20261013/10_longctx_plan.md §3.4",
                "note": "長 context 的每 block 反量化與會合點"})
    out.append({"stage": "44b→44c", "model": "both", "what": "32K／64K／128K 品質：13 種格式＋BF16 的 teacher forcing，每個格式的 ALL 真的生成（讀法 A）；主線三格式 F／M／B／R0 25% 也生成",
                "formats_uniform": "全部 13 種", "status": "pending", "src": "docs/precision_20261013/10_longctx_plan.md §3.1、§5",
                "note": "Qwen 的 INT2 KIVI 等 43 沒排的格式，32K 的生成答對率只會在這裡出現；題數由 44b 試跑決定（比主線少）"})
    out.append({"stage": "44d", "model": "both", "what": "長 context 的還原第二輪（S2k8+G4 等），13 號 Q1", "formats_uniform": "fp8 主線粒度;int4_kivi",
                "status": "pending", "src": "docs/precision_20261013/13_restore2_plan.md §7", "note": "只報告"})
    return out


if __name__ == "__main__":
    main()
