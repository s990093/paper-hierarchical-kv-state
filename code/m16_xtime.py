#!/usr/bin/env python3
"""m16_xtime.py — 第三階段 X3：壓縮組合的「還原成本 × 連線速度 × GPU 忙不忙」（使用者說的 B：KV 搬到 CPU／SSD 時順便壓縮，
壓多少依連線速度和 GPU 忙不忙決定）。

設計與停止條件：docs/stage3_20261019/x3_time_howto.md（**看到數字前寫死**）；
來源：lit/L5c_compress_on_offload.md §5（常數表、N1–N5、D／F／S／A、STOP-0～STOP-3）、
      lit/L5d_stacking.md §6.4（第三階段：各解碼段 × duty、模擬還原、4–8 格驗證、T-a～T-d）、
      lit/L5a_lowrank.md §8 E-L3（重建 vs 反量化）。

  python code/m16_xtime.py --self-test                   只用 CPU、**不 import torch**：map 的計算與判定用合成數字驗證
  python code/m16_xtime.py --smoke --out <dir>           GPU：每個解碼段最小設定跑一次＋map／validate 的管線（冒煙）
  python code/m16_xtime.py micro --model llama --out <dir>     GPU、量時間（兩卡獨佔，job 腳本負責）
  python code/m16_xtime.py map [--micro-dir D] [--out D]        只用 CPU
  python code/m16_xtime.py validate [--map-dir D] [--out D]     GPU、量時間；map 沒判「值得驗證」就寫原因、exit 0

## micro（GPU）：32K 的載入區形狀＝最後 32 個 512-token block（block 32–63）、全層
每個「解碼段」（segment）量 h2d（壓縮後位元組的 pinned H2D）與解碼（dq_oneshot＝32 個 block 一次拆；dq_perblock＝逐 block 拆），
GPU 背景負載用 m10_speed.Busy（m7_busy，duty 0／0.4／0.7／0.9，讀 results/m10_restore2_mi300x/busy_levels_q4.json，
和第二階段 13 §3.3 同一份；0.9 那級第二階段沒有校準過重算慢化，這裡的 recompute 探針會量）。
  bf16              不解碼，只量 BF16 逐 block H2D 進 KV 緩衝的視圖（＝Restorer._load 的搬法）
  fp8／int8_token／int4_kivi
                    主線格式（m10_codec）：h2d_stack＝逐 block 搬進「一次拆」的暫存區（＝Restorer._load_defer）；
                    dq_oneshot＝m10_dequant_fast.decode_multi（＝第二階段 P3／S4）；dq_perblock＝decode_variant（＝P0）；
                    kernel 用第二階段 A 段挑的 K*（m10_dqsweep.kstar）。fp8 的粒度照 fmt_choice.json（Llama fp8_token、Qwen fp8_head）
  int2_kivi／k4v2   新格式（m10_codec_x；k4v2＝kv-int4_kivi-int2_kivi）：逐 block 拆用 m10_dequant_fast_x.decode_x_fast；
                    一次拆＝把 32 個 block 的 part 沿 token 軸接成一段（KIVI 的分組是 32 個 token，不跨 block，所以和逐 block 編碼逐位元相同，
                    冒煙驗），一次 decode_x_fast。第二階段的還原器（defer）不支援新格式的一次拆 → validate 只能驗逐 block
  lr_r<r>           低秩重建（校準 PCA 式：每（層, KV head）一組基底，常駐 GPU，不搬）：K、V 的係數 [L,H,T,r]（BF16）×基底 [r,128]
                    ＝GEMM（torch.matmul）＋融合收尾 kernel（加平均、K 再套 RoPE、寫進 KV 視圖；m16_xtime_kern.py）。r∈{16,24,32,40,64}
  xkv_g4_f<f>       xKV 式跨層重建：每 4 層共用 token 係數 [T, r]，基底 [r, 4·H·128] 是每個 prompt 算的（算進要搬的位元組），r＝4·H·128÷f
  merge_g<g>        相鄰 g 個 token 合併成一個（存 T/g），還原時展開（複製 g 份＝在 attention 裡等於 log g 的偏置）
  droptok_k<k>      每（層, head）各自留 k% 的 token（＋int16 位置），還原時 scatter 進 KV 視圖＋建 key_keep 遮罩
  lrk24_v4          組合：K 低秩 r＝24（＋RoPE）、V 用 INT4 KIVI（＝L5a 的 K24V4，位元組比 0.25）
  droptok_k50_int4  組合：每 head 留 50% token，留下的再 INT4 KIVI；解碼後 scatter
  merge_g2_int4     組合：合併 2 個 token，再 INT4 KIVI；解碼後展開
  xkv_k<a>v<b>[_int4|_int8]   10/21 補缺口：X2 最好的 xKV 組合（m16_xstack 的 xl:k<a>:v<b>[+q:int4_kivi|+q:int8_token]），定義照 m16_xstack：
                    每 4 個相鄰層一個窗口、K、V 秩不同（聯合秩＝a×4×H、b×4×H）、token 係數 [L/4, 1, T, r]、基底 [r, 4·H·128] BF16 每個
                    prompt 一份（算進要搬的位元組）；係數存 BF16、或用 m10_codec_x.enc_half(…, "kivi4"／"int8_token", "k") 量化。
                    解碼＝係數反量化（m16_xtime_kern.coef_dequant，和 dec_half 逐位元相同）→ BF16 GEMM → 每層 store（K 套 RoPE）。
                    多量一個 h2d_req（只搬基底）：map 用它把基底的 H2D 當「每次還原一次」，不跟著 block 數線性外推。
另外：wcost（--wcost；10/21 補缺口）＝xKV 的寫入（壓縮）成本：每題、每窗口 torch.svd_lowrank（K 先逆 RoPE；niter、多取幾個、
      seed 照 m16_xstack）＋係數量化，載入區形狀、duty 0 與 0.9；只出表（write_cost.csv），不進還原決策圖。
另外：recompute 探針（同一個 process 載 KVModel，量 forward_span 在幾個位置、各 duty 的時間 → 重算慢化倍數；
      第二階段只有 duty 0.7 的倍數，0.4 是第一階段單一 chunk，0.9 沒有）。
計時：每個（segment, op）先暖機、定 inner（一次重複至少 --min-rep-s，跨過幾個 m7_busy 週期，取平均＝期望值），
      reps 次（預設 5，照 I4）且每次重複把順序打散；記中位數與四分位。
背景被拖慢多少（bg）：每個（duty, segment, op）開一個新的 Busy、前景連續跑 --bg-seconds 秒，用 m7_busy 的 DONE 計數，
      扣掉「沒前景」那段（基準＝前後各一次只有背景）算前景期間的背景 matmul/s，÷ 基準。
污染：gpu_guard.GpuWatcher 包住整個 process（m10_speed._watcher；Busy 的 pid 不算外人）；每列記當下的 contaminated 與整個 run 的
      run_contaminated，任一為真的列 map 不用（CLAUDE.md §3：量時間的 run 被污染就作廢重量）；有污染時 run 目錄留 CONTAMINATED。
正確性：每個 segment 建好後先驗一次（逐位元或 BF16 捨入內），不過就 exit 1（規則 2），不計時。

## map（只用 CPU）：每一層 × duty 的 32K（64 個 block）還原時間，最佳組合決策圖
  還原模型＝第二階段還原器的兩條線（m10_sweep_sim.simulate，和 Restorer 的 claim 規則同一套）：
    重算線 f_d(i)＝f_閒(i)（bw.json，calib-f）× 重算慢化（micro 的 recompute 探針；沒有就用第二階段的常數，duty 0.9 沒有就 NOT_MEASURED）
    載入線每 block＝max(位元組÷連線, h2d_d)（＋逐 block 拆的 dq）；一次拆在載入線結束後補 dq×載入的 block 數
    還原時間＝min(Cake, 全部載入)（≈第二階段的 G5b）
  層（名目，每個都有出處）：cpu_pinned（真實 pinned H2D）、vllm_cpu（vLLM 0.28 CPU 路徑：copy 次數模型，c 由 c6 的 32K 實測反推）、
      b16／b32／b48（bw.json）、ssd_meas（7.87，calib_c2）、ssd_param（6.98，tier_params）、nfs_meas（0.383）、nfs_param（0.331）。
      決策圖另有 L5c §5.4 的連線格點（0.25–20 GiB/s＋cpu_pinned）。
  三種模式（動作集）：
    L＝只准無損：raw＋無損＋CPU 解（S1 的壓縮比與解壓速度；沒有就只剩 raw → NOT_MEASURED）
    Q＝允許有損、只用第二階段過 ε 的：L ∪ {Llama：fp8、int8_token、int4_kivi；Qwen：int8_token、int4_kivi}×{一次拆, 逐 block}
    X＝疊加：Q ∪ X2 判定可行的（results/m16_stage3_mi300x/x_stack/x2_feasible.csv）；沒有 X2 就是「只看時間」，不下結論
  比較（每個模式都報）：D＝不壓縮（vLLM 只搬原始位元組）、F＝事後最好的固定動作（每個模型一個）、
      S＝只看連線的分層靜態（每（模型, 層）一個）、A＝連線＋GPU 忙（每格最好；忙碌已知的上界）。
  判定（寫死，見 x3_time_howto.md §4）：STOP-0、STOP-1、STOP-2a／2b／2c、GO_N5、T-a～T-d、E-L3。
  輸出：map_cells.csv、map_decision.csv、map_summary.json、flip_cells.json、constants_used.csv、map_report.md、decision_*.svg
  10/21：--micro-dir 可以給多個目錄（第一個為主；後面的只補前面沒有的 segment，重疊的 segment 當跨 run 漂移的錨點、不覆蓋）；
        有 wcost_<m>.csv 時另寫 write_cost.csv（寫入路徑成本，不進決策）。合併多個目錄時不准把輸出寫進第一版的目錄。

## validate（GPU）：map 挑的 4–8 格做端到端還原，和預測比（STOP-3）
  第二階段還原器支援的動作（raw、fp8／int8／int4 的一次拆或逐 block、int2／k4v2 的逐 block）→ 子行程跑 m10_speed.py grid
      （E0／E1、Cake＋全部載入、2 launch × 5 rep、限速器＝那一層的頻寬）；
  其他（低秩、跨層、合併、丟 token、組合）→ 同一個 process 的「載入線」實測（全部載入、真的 H2D＋限速＋解碼），和 map 的全部載入預測比；
  無損＋CPU 解、vllm_cpu：沒有還原路徑 → NOT_MEASURED。

規則：模組層不 import torch／vllm／transformers（GPU 路徑在函式裡才 import）；--self-test 開頭與結尾檢查 sys.modules 沒有 torch。
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.dirname(HERE)
RES = os.path.join(REPO, "results", "m16_stage3_mi300x", "x3_time")
GiB = 1 << 30
CHUNK = 512
N_CTX = 64                                     # 32K＝64 個 512-token block
NM = "NOT_MEASURED"
SHAPES = {"llama": (32, 8, 128), "qwen": (28, 4, 128)}        # (層, KV head, head_dim)；micro 用 config.json 交叉驗證（規則 6）
MODELS = ("llama", "qwen")
BUSY_LEVELS = os.path.join(REPO, "results", "m10_restore2_mi300x", "busy_levels_q4.json")
GPU_STATES = ("idle", "busy1", "busy2", "busy3")
LR_RANKS = (16, 24, 32, 40, 64)
XKV = ((4, 8), (4, 4))                          # (每組幾層 G, 位元組 1/f)
MERGE_G = (2, 4)
DROP_KEEP = (50, 25)
MAIN_Q = ("fp8", "int8_token", "int4_kivi")
NEW_Q = {"int2_kivi": "int2_kivi", "k4v2": "kv-int4_kivi-int2_kivi"}
COMBOS = ("lrk24_v4", "droptok_k50_int4", "merge_g2_int4")
# 10/21 補缺口：X2 最好的 xKV 組合（tables_x2.md、x2_unmapped.csv）。名稱 → (K 每 head 每層等效秩, V 的, 係數格式)。
# 定義照 code/m16_xstack.py 的 xl 階段（聯合秩＝秩×XKV_W×H；係數 [L/W, 1, T, 聯合秩]；基底 [聯合秩, W·H·D] BF16、每題一份；
# 係數量化＝m10_codec_x.enc_half(…, 格式, "k")；位元組＝m16_xstack.predict_bytes，自測逐項核對）。名稱是和寫 X2 的 agent 約好的，不要改。
XKV_W = 4
XKV2 = {"xkv_k11v17": (11, 17, "bf16"), "xkv_k22v34": (22, 34, "bf16"), "xkv_k28v42": (28, 42, "bf16"),
        "xkv_k11v17_int4": (11, 17, "kivi4"), "xkv_k22v34_int4": (22, 34, "kivi4"), "xkv_k28v42_int4": (28, 42, "kivi4"),
        "xkv_k22v34_int8": (22, 34, "int8_token")}
XKV2_Q = {"bf16": "", "kivi4": "+q:int4_kivi", "int8_token": "+q:int8_token"}
XKV2_SPEC = {s: f"xl:k{a}:v{b}{XKV2_Q[f]}" for s, (a, b, f) in XKV2.items()}      # ↔ m16_xstack 的組合寫法（canonical）
XKV2_PAIRS = sorted({(a, b) for a, b, _ in XKV2.values()})
COMBO_SEGS = set(COMBOS) | {s for s, (_, _, f) in XKV2.items() if f != "bf16"}      # T-a 的「組合」
# xKV 的 svd_lowrank 參數：照 m16_xstack 的 XL_NITER、XL_OVERSAMPLE、XL_SEED（自測和 GPU 路徑都和它核對；不一樣就停）
XL_NITER, XL_OVERSAMPLE, XL_SEED = 4, 16, 20261020
BG_X2 = ["xkv_k22v34:dq_oneshot", "xkv_k22v34_int4:dq_oneshot"]                     # 10/21 micro 的背景拖慢（另外兩個新段的代表）
WCOST_STATES = ("idle", "busy3")
BG_DEFAULT = ["bf16:h2d_region", "int4_kivi:h2d_stack", "int4_kivi:dq_oneshot", "int4_kivi:dq_perblock",
              "fp8:dq_oneshot", "int8_token:dq_oneshot", "k4v2:dq_oneshot", "lr_r24:dq_oneshot", "lr_r40:dq_oneshot",
              "xkv_g4_f8:dq_oneshot", "merge_g2:dq_oneshot", "droptok_k50:dq_oneshot", "lrk24_v4:dq_oneshot"]
REC_POS = (0, 8, 16, 24, 32, 40, 48, 56, 63)
# 第二階段過 ε 的有損格式（模式 Q）。出處：L5c §5.1 H-B0（08_report_mi300x.md §2.3–2.5；INT4 KIVI 24 格全過，
# FP8 只有 Llama 過——Qwen 的 fp8_head −32.8；INT8 兩個模型都過）。這是分類，不是數字。
QUALITY_Q = {"llama": ("fp8", "int8_token", "int4_kivi"), "qwen": ("int8_token", "int4_kivi")}
QUALITY_Q_SRC = "L5c §5.1 H-B0（docs/precision_20261013/08_report_mi300x.md §2.3–2.5）"
LINK_GRID = (0.25, 0.33, 0.383, 0.83, 1.0, 3.0, 6.98, 7.87, 12.0, 20.0)     # L5c §5.4 的連線格點（GiB/s），只給決策圖
NFS_CLASS_GIBPS = 1.0                         # T-b：連線 < 1 GiB/s 算「NFS 級」
GPU_LOSSLESS_DECODER = False                  # hipCOMP／SplitZip 沒裝（要使用者同意），STOP-1 的「沒有可用的 GPU 解碼器」
# 寫死的門檻（x3_time_howto.md §4；L5c §5.5、L5d §6.4、L5a E-L3）
TH = dict(stop0_flip=0.05, stop1_ratio=1.15, stop1_GiBps=0.5, s2a=0.05, s2b=0.05, go_cell=0.15, go_frac=0.25, go_gm=0.15,
          ta=0.05, tc=0.20, td=0.15, el3_hi=1.5, el3_lo=0.20, el3_h2d_hi=0.50, el3_h2d_lo=0.25, stop3_err=0.10,
          stop3_cells=2, stop3_gain=0.05, anchor_warn=0.15, near_tie=0.05)

DEFAULT_PATHS = {
    "bw": os.path.join(REPO, "results", "m10_precision_mi300x", "speed_{m}", "bw.json"),
    "summary_int4": os.path.join(REPO, "results", "m10_precision_mi300x", "speed_{m}", "summary_int4.csv"),
    "calib_c2": os.path.join(REPO, "results", "m7_write_policy_mi300x", "calib_c2.csv"),
    "tier_params": os.path.join(REPO, "results", "m7_write_policy_mi300x", "tier_params.json"),
    "busy_levels": os.path.join(REPO, "results", "m7_write_policy_mi300x", "busy_levels.json"),
    "busy_levels_q4": BUSY_LEVELS,
    "c6": os.path.join(REPO, "results", "m7_write_policy_mi300x", "c6_vllm.csv"),
    "fmt_choice": os.path.join(REPO, "results", "m10_precision_mi300x", "fmt_select", "fmt_choice.json"),
    "s1": os.path.join(REPO, "results", "m16_stage3_mi300x", "s1_lossless", "summary.json"),
    "x2": os.path.join(REPO, "results", "m16_stage3_mi300x", "x_stack", "x2_feasible.csv"),
    "i4": os.path.join(REPO, "results", "m16_stage3_mi300x", "i4_copy_batching", "copybench_sdma1.csv"),
    "q4": os.path.join(REPO, "results", "m10_restore2_mi300x", "q4"),
    "q4duty": os.path.join(REPO, "results", "m10_restore2_mi300x", "q4duty"),
}

MICRO_FIELDS = ["run_id", "ts", "model", "gpu_state", "duty", "segment", "family", "kind", "fmt", "op", "impl", "kernel",
                "n_blocks", "bytes_op", "bytes_bf16", "bytes_ratio", "n_copies", "reps", "inner", "t_med_s", "t_q1_s",
                "t_q3_s", "t_min_s", "t_max_s", "contaminated", "run_contaminated", "host_level", "hsa_enable_sdma",
                "store_impl"]
BG_FIELDS = ["run_id", "ts", "model", "gpu_state", "duty", "segment", "op", "fg_s", "fg_iters", "fg_mean_s", "bg_n",
             "bg_total_s", "bg_rate_base", "bg_rate_fg", "bg_ratio", "contaminated", "run_contaminated"]
REC_FIELDS = ["run_id", "ts", "model", "gpu_state", "duty", "chunk_idx", "rep", "s", "contaminated", "run_contaminated"]
WC_FIELDS = ["run_id", "ts", "model", "gpu_state", "duty", "pair", "rk", "rv", "op", "half", "rank_joint", "coef_fmt", "W",
             "n_windows", "T", "svd_q", "niter", "oversample", "reps", "t_med_s", "t_q1_s", "t_q3_s", "t_min_s", "t_max_s",
             "contaminated", "run_contaminated", "host_level", "linalg"]
WCT_FIELDS = ["run_id", "ts", "micro_run_id", "model", "segment", "spec", "duty", "t_svd_k_s", "t_svd_v_s", "t_quant_s",
              "t_total_s", "t_total_per_block_s", "n_blocks", "bytes_coef", "bytes_basis", "ratio_vs_bf16",
              "recompute_region_idle_s", "total_over_recompute", "note"]
CELL_FIELDS = ["run_id", "ts", "micro_run_id", "model", "grid", "tier", "tier_kind", "link_GiBps", "gpu_state", "duty",
               "action", "segment", "dec", "family", "kind", "bytes_ratio", "T_s", "restore_mode", "b", "T_cake_s",
               "T_loadall_s", "flags"]
DEC_FIELDS = ["run_id", "ts", "micro_run_id", "mode", "model", "tier", "link_GiBps", "duty", "A", "T_A", "second",
              "T_second", "margin2", "S", "T_S", "F", "T_F", "T_D", "gain_vs_D", "gain_vs_F", "gain_vs_S", "flags"]
VAL_FIELDS = ["run_id", "ts", "cell", "mode", "model", "tier", "link_GiBps", "duty", "gpu_state", "action", "role",
              "how", "pred_s", "meas_s", "rel_err", "meas_detail", "src", "contaminated"]


# ====================================================================== 共用小工具（不碰 torch）
def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_id():
    return os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"


def run_dir():
    rd = os.path.join(os.environ.get("TIARA_RUNS", "/mlsteam/data/tiara/runs"), run_id())
    return rd if os.path.isdir(rd) else None


def gm(xs):
    xs = list(xs)
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def med(xs):
    xs = list(xs)
    return statistics.median(xs) if xs else None


def quart(xs):
    xs = sorted(xs)
    if len(xs) < 2:
        return (xs[0], xs[0]) if xs else (None, None)
    q = statistics.quantiles(xs, n=4)
    return q[0], q[2]


def read_csv(p):
    op = gzip.open(p, "rt") if p.endswith(".gz") else open(p, newline="")
    with op as f:
        return list(csv.DictReader(f))


def write_csv(p, rows, fields):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def rel(p):
    try:
        return os.path.relpath(p, REPO) if os.path.isabs(p) and p.startswith(REPO) else p
    except ValueError:
        return p


def dkey(x) -> float:
    return round(float(x), 3)


def seg_catalog(ranks=LR_RANKS):
    """segment → family（raw／quant／lowrank／xlayer／merge／drop／combo）、kind（none／single／combo）、ops、feed（解碼 op 吃哪個 h2d op）。"""
    cat = {"bf16": dict(family="raw", kind="none", ops=["h2d_region"], feed={})}
    for q in MAIN_Q:
        cat[q] = dict(family="quant", kind="single", ops=["h2d_stack", "dq_oneshot", "dq_perblock"],
                      feed={"dq_oneshot": "h2d_stack", "dq_perblock": "h2d_stack"})
    for q in NEW_Q:
        cat[q] = dict(family="quant", kind="single", ops=["h2d_stack", "h2d_region", "dq_oneshot", "dq_perblock"],
                      feed={"dq_oneshot": "h2d_region", "dq_perblock": "h2d_stack"})
    for r in ranks:
        cat[f"lr_r{r}"] = dict(family="lowrank", kind="single", ops=["h2d_region", "dq_oneshot", "dq_perblock"],
                               feed={"dq_oneshot": "h2d_region", "dq_perblock": "h2d_region"})
    for G, f in XKV:
        cat[f"xkv_g{G}_f{f}"] = dict(family="xlayer", kind="single", ops=["h2d_region", "dq_oneshot"],
                                     feed={"dq_oneshot": "h2d_region"})
    for g in MERGE_G:
        cat[f"merge_g{g}"] = dict(family="merge", kind="single", ops=["h2d_region", "dq_oneshot", "dq_perblock"],
                                  feed={"dq_oneshot": "h2d_region", "dq_perblock": "h2d_region"})
    for k in DROP_KEEP:
        cat[f"droptok_k{k}"] = dict(family="drop", kind="single", ops=["h2d_region", "dq_oneshot"],
                                    feed={"dq_oneshot": "h2d_region"})
    for c in COMBOS:
        cat[c] = dict(family="combo", kind="combo", ops=["h2d_region", "dq_oneshot"], feed={"dq_oneshot": "h2d_region"})
    for s, (_, _, f) in XKV2.items():                # 10/21：BF16 係數＝單一方法（xlayer）；係數再量化＝組合
        q = f != "bf16"
        cat[s] = dict(family="combo" if q else "xlayer", kind="combo" if q else "single",
                      ops=["h2d_region", "h2d_req", "dq_oneshot", "dq_perblock"],
                      feed={"dq_oneshot": "h2d_region", "dq_perblock": "h2d_region"})
    return cat


def xkv_coef_bytes(Lg, T, r, fmt):
    """xKV 一半（K 或 V）的 token 係數 [Lg, 1, T, r] 用 fmt 存的位元組＝m16_xstack.half_bytes(fmt, (Lg,1,T,r), "k")
    （m10_codec_x.enc_half 的實際 tensor 大小；kivi4 的 K 角色：每主成分、每 32 個 token 一組 FP16 scale＋zero）。"""
    n = Lg * T * r
    if fmt == "bf16":
        return 2 * n
    if fmt == "kivi4":
        if T % 32 or r % 2:
            raise ValueError(f"kivi4 係數要 T 是 32 的倍數、r 是偶數：T={T} r={r}")
        return n // 2 + 2 * 2 * Lg * (T // 32) * r
    if fmt == "int8_token":
        return n + Lg * T * 4
    raise ValueError(fmt)


def xkv_bytes(model, rk, rv, fmt, T):
    """10/21 的 xKV 段：T 個 token 的 (係數位元組, 基底位元組)。和 m16_xstack.predict_bytes 的 k.coef＋v.coef、k.basis＋v.basis 相同（自測核對）。"""
    L, H, D = SHAPES[model]
    if L % XKV_W:
        raise ValueError(f"{model} 的層數 {L} 不能被 {XKV_W} 整除")
    Lg = L // XKV_W
    rs = [rh * XKV_W * H for rh in (rk, rv)]
    return (sum(xkv_coef_bytes(Lg, T, r, fmt) for r in rs), sum(Lg * r * XKV_W * H * D * 2 for r in rs))


# ====================================================================== 常數（讀檔、附出處；不抄數字）
class Consts:
    def __init__(self):
        self.v, self.rows = {}, []

    def add(self, model, key, value, unit, file, rid, how):
        self.v[(model, key)] = value
        shown = value
        if isinstance(value, list):
            shown = f"{len(value)} 個值；總和 {sum(value):.6g}"
        self.rows.append(dict(model=model, const=key, value=shown, unit=unit, file=rel(file), run_id=rid, how=how))

    def nm(self, model, key, why):
        self.v[(model, key)] = None
        self.rows.append(dict(model=model, const=key, value=NM, unit="", file="", run_id="", how=why))

    def get(self, model, key):
        return self.v.get((model, key))


def load_constants(models=MODELS, paths=None) -> Consts:
    """L5c §5.2 的常數：每一個都從檔案讀，記檔名與 run_id。必要的檔（bw.json、calib_c2、tier_params、busy_levels_q4）讀不到就丟例外；
    選用的（c6、summary_int4、busy_levels 的 busy1）讀不到記 NOT_MEASURED（規則 7：查不到不等於沒有，照實標出）。"""
    p = dict(DEFAULT_PATHS, **(paths or {}))
    C = Consts()
    for m in models:
        bwp = p["bw"].format(m=m)
        bw = json.load(open(bwp))
        f = [x / 1e3 for x in bw["f_ms_median_by_chunk"]]
        if len(f) != N_CTX or bw.get("n_blocks") != N_CTX:
            raise ValueError(f"{bwp}：f(i) 不是 64 個 chunk（{len(f)}）")
        C.add(m, "f_idle_s", f, "s/chunk", bwp, bw["run_id"], "calib-f：每 chunk 重算時間中位數（GPU 閒，32K）")
        C.add(m, "chunk_bytes", int(bw["chunk_bytes"]), "B", bwp, bw["run_id"], "BF16 每個 512-token block 的位元組")
        C.add(m, "real_pinned_h2d_GiBps", float(bw["real_pinned_h2d_GiBps"]), "GiB/s", bwp, bw["run_id"],
              "真實 pinned H2D（BF16 chunk 進 KV 視圖，20 次中位數）；只當 micro 的交叉參考")
        for k in ("b16", "b32", "b48"):
            C.add(m, f"link_{k}", float(bw["targets"][k]["GiBps"]), "GiB/s", bwp, bw["run_id"],
                  f"第二階段限速層 {k}（Cake 會合點 b*={bw['targets'][k]['b_star']} 反推的頻寬）")
        sp = p["summary_int4"].format(m=m)
        try:
            rows = [r for r in read_csv(sp) if r["length"] == "32768" and r["mode"] == "compute_only" and r["bw_name"] == "cpu"]
            idle = [float(r["median_t_restore"]) for r in rows if r["gpu_state"] == "idle"]
            busy = [float(r["median_t_restore"]) for r in rows if r["gpu_state"] == "busy"]
            if len(idle) != 1 or len(busy) != 1:
                raise ValueError(f"compute_only 列數 idle={len(idle)} busy={len(busy)}")
            C.add(m, "mult_f_stage2_d0.7", busy[0] / idle[0], "×", sp, rows[0]["run_ids"],
                  "第二階段 compute_only（全部重算 64 塊）忙(duty 0.7)/閒；micro 沒有 recompute 探針時的備援")
        except (OSError, KeyError, ValueError) as ex:
            C.nm(m, "mult_f_stage2_d0.7", f"讀不到 {rel(sp)}：{ex!r}")
        fc = json.load(open(p["fmt_choice"]))
        C.add(m, "fp8_fmt", fc["choice"][m]["fp8"], "-", p["fmt_choice"], ";".join(fc["run_ids"]),
              "第二階段每個模型選定的 FP8 粒度（KL 最小）")
    c2 = read_csv(p["calib_c2"])
    for dev, key in (("local", "link_ssd_meas"), ("nfs", "link_nfs_meas")):
        v = [float(r["MiBps"]) for r in c2 if r["device"] == dev and r["op"] == "read_sustained"]
        if not v:
            raise ValueError(f"{p['calib_c2']} 沒有 {dev} read_sustained")
        C.add("-", key, med(v) / 1024, "GiB/s", p["calib_c2"], ";".join(sorted({r["run_id"] for r in c2})),
              f"第一階段 C2 實測 {dev} read_sustained 中位數（n={len(v)}，範圍 {min(v)}–{max(v)} MiB/s）")
    tp = json.load(open(p["tier_params"]))
    for dev, key in (("local", "link_ssd_param"), ("nfs", "link_nfs_param")):
        C.add("-", key, float(tp[dev]["read_GiBps"]), "GiB/s", p["tier_params"], ";".join(tp["source_runs"]),
              f"第二階段限速器參數（{tp['rule']}）")
    try:
        bl = json.load(open(p["busy_levels"]))
        C.add("-", "mult_f_phase1_d0.4", float(bl["_measured_slowdown"]["busy1"]), "×", p["busy_levels"],
              ";".join(bl["_source_runs"]), "第一階段 C5：duty 0.4 單一 chunk 的重算慢化（只當備援）")
    except (OSError, KeyError, ValueError) as ex:
        C.nm("-", "mult_f_phase1_d0.4", f"讀不到：{ex!r}")
    q4 = json.load(open(p["busy_levels_q4"]))
    for k in GPU_STATES:
        C.add("-", f"duty_{k}", float(q4[k]), "duty", p["busy_levels_q4"], "-",
              "m7_busy 的 duty（13 §3.3；busy3＝0.9 沒有校準過）")
    try:
        ts_, bs_, rid = [], [], set()
        for r in read_csv(p["c6"]):
            if r["kind"] != "warm_cpu_load" or r["L"] != "32768":
                continue
            s = r["offload_metrics"]
            b = re.search(r'kv_offload_total_bytes_total\{[^}]*CPU_to_GPU[^}]*\}": ([0-9.e+]+)', s)
            t = re.search(r'kv_offload_total_time_total\{[^}]*CPU_to_GPU[^}]*\}": ([0-9.e+]+)', s)
            if b and t:
                ts_.append(float(t.group(1)))
                bs_.append(float(b.group(1)))
                rid.add(r["run_id"])
        if not ts_ or len(set(bs_)) != 1:
            raise ValueError(f"c6 32K 列 {len(ts_)}，位元組 {set(bs_)}")
        C.add("llama", "vllm_c6_T_s", med(ts_), "s", p["c6"], ";".join(sorted(rid)),
              f"vLLM 0.28 OffloadingConnector CPU→GPU 32K 傳輸時間中位數（n={len(ts_)}，{bs_[0] / GiB:.3g} GiB）")
        C.add("llama", "vllm_c6_bytes", bs_[0], "B", p["c6"], ";".join(sorted(rid)), "同上那次搬的位元組")
    except (OSError, KeyError, ValueError) as ex:
        C.nm("llama", "vllm_c6_T_s", f"讀不到 c6：{ex!r}")
    return C


def load_i4(path):
    """I4（m16_i4_copybench，SDMA 預設）的 foreach／scatter 每片 µs：busy2÷idle → vLLM CPU 路徑每次 copy 在 duty 0.7 的倍數。
    沒有檔 → None（vllm_cpu 在忙的格子就是 NOT_MEASURED）。"""
    if not os.path.exists(path):
        return None
    rows = [r for r in read_csv(path) if r.get("method") == "foreach" and r.get("layout") == "scatter"
            and r.get("status") == "ok" and str(r.get("contaminated")) == "False"]
    v = {r["gpu_state"]: float(r["us_per_frag"]) for r in rows if r.get("us_per_frag")}
    if "idle" not in v or "busy2" not in v:
        return None
    return dict(k_busy2=v["busy2"] / v["idle"], run_id=";".join(sorted({r["run_id"] for r in rows})), file=path)


def load_s1(path, models=MODELS):
    """S1（m16_s1_lossless analyze）的 summary.json → 每個模型（T4 自然文字、32K）的無損選項與 STOP-1 的輸入。沒有檔 → None。"""
    if not os.path.exists(path):
        return None
    S = json.load(open(path))
    out = {}
    for dn, s in S["dumps"].items():
        meta = s.get("meta", {})
        m, task = meta.get("model"), meta.get("task")
        if m not in models or meta.get("n_tokens") not in (None, N_CTX * CHUNK):
            continue
        ratios = {k: v["ratio"] for k, v in s.get("bf16", {}).items() if k.startswith("all|") and "ratio" in v}
        opts = []
        for key, sp in s.get("speed", {}).items():
            var, cs, th = key.split("|")
            k = int(th.lstrip("t"))
            if k <= 16 and sp.get("decomp_MBps") and sp.get("ratio_framed"):
                opts.append(dict(name=f"ll:{var}|{cs}|t{k}", var=var, codec=cs, threads=k, r=float(sp["ratio_framed"]),
                                 D_Bps=float(sp["decomp_MBps"]) * 1e6))
        out.setdefault(m, {})[task] = dict(dump=dn, best_ratio=max(ratios.values()) if ratios else None,
                                           best_ratio_key=max(ratios, key=ratios.get) if ratios else None,
                                           max_decomp_GiBps=max((o["D_Bps"] for o in opts), default=0) / GiB, opts=opts)
    return dict(run_id=S.get("run_id", "NOT_SET"), file=path, models=out)


def load_x2(path):
    """X2 的品質可行集。約定（給寫 X2 的 agent 核對）：CSV 欄位至少 model、x3_segment（本檔 seg_catalog 的名字）、feasible（1/0／True/False）、run_id。
    沒有檔 → None（模式 X 只看時間）；有檔但欄位不對 → 丟例外（規則 7：讀不懂不等於沒有）。"""
    if not os.path.exists(path):
        return None
    rows = read_csv(path)
    need = {"model", "x3_segment", "feasible", "run_id"}
    if not rows or not need <= set(rows[0]):
        raise ValueError(f"{path} 欄位不符約定 {sorted(need)}：{sorted(rows[0]) if rows else '空檔'}")
    cat = seg_catalog()
    out, unknown = {}, sorted({r["x3_segment"] for r in rows} - set(cat))
    if unknown:
        raise ValueError(f"{path} 有 seg_catalog 沒有的 x3_segment：{unknown}")
    for r in rows:
        if str(r["feasible"]).strip().lower() in ("1", "true", "yes"):
            out.setdefault(r["model"], set()).add(r["x3_segment"])
    return dict(feasible=out, run_id=";".join(sorted({r["run_id"] for r in rows})), file=path)


# ====================================================================== micro 的輸出 → map 的輸入
def load_micro(micro_dir, models=MODELS):
    """每個模型取最新那次 micro（meta 的 run_id）；污染列（contaminated 或 run_contaminated 不是 False）不用。"""
    M = {}
    for m in models:
        mp, cp = os.path.join(micro_dir, f"micro_meta_{m}.json"), os.path.join(micro_dir, f"micro_{m}.csv")
        if not (os.path.exists(mp) and os.path.exists(cp)):
            M[m] = None
            continue
        meta = json.load(open(mp))
        rid = meta["run_id"]
        rows = [r for r in read_csv(cp) if r["run_id"] == rid]
        t, dropped = {}, []

        def clean(r):
            return str(r.get("contaminated")) == "False" and str(r.get("run_contaminated")) == "False"
        for r in rows:
            k = (dkey(r["duty"]), r["segment"], r["op"])
            if not clean(r):
                dropped.append("|".join(map(str, k)))
                continue
            t[k] = dict(med=float(r["t_med_s"]), q1=float(r["t_q1_s"]), q3=float(r["t_q3_s"]), n=int(r["reps"]))
        rec, recp = {}, os.path.join(micro_dir, f"micro_recompute_{m}.csv")
        if os.path.exists(recp):
            per = {}
            for r in read_csv(recp):
                if r["run_id"] != rid or not clean(r):
                    continue
                per.setdefault((dkey(r["duty"]), int(r["chunk_idx"])), []).append(float(r["s"]))
            idle = {i: med(v) for (d, i), v in per.items() if d == 0.0}
            for d in sorted({d for d, _ in per}):
                ratios = [med(per[(d, i)]) / idle[i] for (dd, i) in per if dd == d and i in idle]
                if ratios and idle:
                    rec[d] = dict(mult=med(ratios), n_pos=len(ratios), run_id=rid,
                                  f_idle_probe={i: idle[i] for i in sorted(idle)})
        bg, bgp = {}, os.path.join(micro_dir, f"micro_bg_{m}.csv")
        if os.path.exists(bgp):
            for r in read_csv(bgp):
                if r["run_id"] == rid and clean(r) and r["bg_ratio"] not in ("", NM):
                    bg[(dkey(r["duty"]), r["segment"], r["op"])] = float(r["bg_ratio"])
        wc, wcp = {}, os.path.join(micro_dir, f"wcost_{m}.csv")            # 10/21：xKV 的寫入成本（同一個 micro run）
        if os.path.exists(wcp):
            for r in read_csv(wcp):
                if r["run_id"] != rid:
                    continue
                k = (dkey(r["duty"]), r["pair"], r["op"])
                if not clean(r):
                    dropped.append("wcost|" + "|".join(map(str, k)))
                    continue
                wc[k] = dict(med=float(r["t_med_s"]), q1=float(r["t_q1_s"]), q3=float(r["t_q3_s"]), n=int(r["reps"]), run_id=rid)
        M[m] = dict(run_id=rid, meta=meta, n_blocks=int(meta["n_blocks"]), seg=meta["segments"], t=t, rec=rec, bg=bg,
                    dropped=dropped, smoke=bool(meta.get("smoke")), file=cp, seg_rid={s: rid for s in meta["segments"]},
                    drift=[], wcost=wc, dirs=[micro_dir])
    return M


def load_micro_dirs(micro_dirs, models=MODELS):
    """10/21：多個 micro 目錄合併成一份（第一個為主）。後面的目錄只補「前面沒有的 segment」（時間、背景拖慢、位元組、出處 run_id 都跟著
    那個 segment）；重疊的 segment（錨點，例如 bf16、int4_kivi）不覆蓋，只拿來算跨 run 漂移（drift：後 ÷ 前）。
    recompute 探針、wcost 用第一個有的。不同目錄的載入區 block 數不同 → 丟例外（兩份時間不能拼）。"""
    if isinstance(micro_dirs, str):
        micro_dirs = [micro_dirs]
    per = [load_micro(d, models) for d in micro_dirs]
    M = {}
    for m in models:
        parts = [P[m] for P in per if P.get(m)]
        if not parts:
            M[m] = None
            continue
        base = dict(parts[0], seg=dict(parts[0]["seg"]), t=dict(parts[0]["t"]), bg=dict(parts[0]["bg"]),
                    dropped=list(parts[0]["dropped"]), seg_rid=dict(parts[0]["seg_rid"]), drift=[],
                    wcost=dict(parts[0]["wcost"]), dirs=list(parts[0]["dirs"]))
        rids = [parts[0]["run_id"]]
        for Mm in parts[1:]:
            if Mm["n_blocks"] != base["n_blocks"]:
                raise ValueError(f"{m}：micro 的載入區 block 數不同（{base['n_blocks']} vs {Mm['n_blocks']}，{Mm['dirs']}）→ 不能合併")
            new = {s for s in Mm["seg"] if s not in base["seg"]}
            for s in new:
                base["seg"][s] = Mm["seg"][s]
                base["seg_rid"][s] = Mm["run_id"]
            for k, v in Mm["t"].items():
                if k[1] in new:
                    base["t"][k] = v
                elif k in base["t"]:
                    a, b = base["t"][k]["med"], v["med"]
                    base["drift"].append(dict(duty=k[0], segment=k[1], op=k[2], t_first_s=a, t_later_s=b, ratio=b / a,
                                              first_run_id=base["seg_rid"].get(k[1], rids[0]), later_run_id=Mm["run_id"],
                                              warn=abs(b / a - 1) > TH["anchor_warn"]))
            base["bg"].update({k: v for k, v in Mm["bg"].items() if k[1] in new})
            base["dropped"] += [f"{Mm['run_id']}:{x}" for x in Mm["dropped"]]
            if not base["rec"] and Mm["rec"]:
                base["rec"] = Mm["rec"]
            for k, v in Mm["wcost"].items():
                base["wcost"].setdefault(k, v)
            base["smoke"] = base["smoke"] or Mm["smoke"]
            base["dirs"] += Mm["dirs"]
            rids.append(Mm["run_id"])
        base["run_id"] = ";".join(rids)
        M[m] = base
    return M


# ====================================================================== 還原時間模型（只用 CPU）
def actions_for(Mm, s1m=None, ranks=None):
    """micro 量到的 segment → 動作（含解碼方式）；再加 S1 的無損＋CPU 解。"""
    acts = []
    for s, info in Mm["seg"].items():
        if s == "bf16":
            acts.append(dict(name="raw", seg="bf16", dec="none", family="raw", kind="none"))
            continue
        for dec in ("oneshot", "perblock"):
            if f"dq_{dec}" in info["feed"]:
                acts.append(dict(name=f"{s}@{dec}", seg=s, dec=dec, family=info["family"], kind=info["kind"]))
    for o in (s1m or {}).get("T4", {}).get("opts", []):
        acts.append(dict(name=o["name"], seg="bf16", dec="cpu", family="lossless", kind="none", r=o["r"], D_Bps=o["D_Bps"]))
    return acts


def blk_costs(Mm, act, duty):
    """每個 block 的 h2d、逐 block 拆、一次拆（攤到每 block）與位元組；缺資料（沒量、被污染）回 None。"""
    n, t = Mm["n_blocks"], Mm["t"]
    seg = Mm["seg"][act["seg"]]
    bytes_blk = seg["bytes_blk"] + seg.get("bytes_req", 0) / N_CTX
    if act["dec"] in ("none", "cpu"):
        th = t.get((duty, "bf16", "h2d_region"))
        if th is None:
            return None
        return dict(h2d=th["med"] / n, dq_pb=0.0, dq_os=0.0, bytes=Mm["seg"]["bf16"]["bytes_blk"])
    dq = f"dq_{act['dec']}"
    th, td = t.get((duty, act["seg"], seg["feed"][dq])), t.get((duty, act["seg"], dq))
    if th is None or td is None:
        return None
    h2d, req = th["med"] / n, ""
    if seg.get("bytes_req"):
        # 每次還原搬一次的位元組（xKV 的基底）：micro 的 h2d 是「n 個 block 的係數＋1 份基底」。÷n 再 ×64 會把基底算 64/n 次。
        # 10/21 的段另外量了只搬基底的 h2d_req → 每 block＝(h2d − h2d_req)÷n ＋ h2d_req÷64（基底一次攤到 64 塊，和 bytes 的算法一致）。
        tr = t.get((duty, act["seg"], "h2d_req"))
        if tr is not None and th["med"] > tr["med"]:
            h2d, req = (th["med"] - tr["med"]) / n + tr["med"] / N_CTX, "split"
        else:
            req = "linear"                     # 第一版的 xkv_g4_f*：沒有 h2d_req，基底的 H2D 跟著 block 數線性外推（高估）
    return dict(h2d=h2d, dq_pb=td["med"] / n if act["dec"] == "perblock" else 0.0,
                dq_os=td["med"] / n if act["dec"] == "oneshot" else 0.0, bytes=bytes_blk, req=req)


def f_mult(Mm, C, model, duty):
    """重算慢化倍數：micro 的 recompute 探針優先；沒有就用第二階段（0.7）／第一階段（0.4）的常數；都沒有 → None（NOT_MEASURED）。"""
    if duty == 0.0:
        return 1.0, "定義"
    r = Mm.get("rec", {}).get(duty)
    if r:
        return r["mult"], f"micro recompute 探針（{r.get('run_id', Mm['run_id'])}，{r['n_pos']} 個位置）"
    if abs(duty - 0.7) < 1e-9 and C.get(model, "mult_f_stage2_d0.7"):
        return C.get(model, "mult_f_stage2_d0.7"), "第二階段 summary_int4 compute_only"
    if abs(duty - 0.4) < 1e-9 and C.get("-", "mult_f_phase1_d0.4"):
        return C.get("-", "mult_f_phase1_d0.4"), "第一階段 C5（單一 chunk）"
    return None, NM


def vllm_copy_cost(Mm_llama, C, i4, duty):
    """vLLM CPU 路徑每次 copy 的固定成本 c（秒）：c6 的 32K 實測時間扣掉「同樣位元組在真實 pinned H2D 的時間」，÷ copy 次數。
    copy 次數＝層×2（K/V）×（32768÷16）（vLLM block 16 token，F1 §4.6 的計法）。忙時乘 I4 的倍數（只有 duty 0.7）。"""
    T6, B6 = C.get("llama", "vllm_c6_T_s"), C.get("llama", "vllm_c6_bytes")
    if not T6 or not Mm_llama:
        return None, "c6 或 Llama 的 micro 沒有"
    raw = blk_costs(Mm_llama, dict(seg="bf16", dec="none"), 0.0)
    if raw is None:
        return None, "Llama micro 沒有閒時的 bf16 h2d"
    L = SHAPES["llama"][0]
    n_copy = L * 2 * (N_CTX * CHUNK // 16)
    t_bytes = B6 / Mm_llama["seg"]["bf16"]["bytes_blk"] * raw["h2d"]
    c0 = (T6 - t_bytes) / n_copy
    if c0 <= 0:
        return None, f"反推的 c ≤ 0（c6 {T6:.4f} s，位元組時間 {t_bytes:.4f} s）"
    if duty == 0.0:
        return c0, "c6 反推"
    if abs(duty - 0.7) < 1e-9 and i4:
        return c0 * i4["k_busy2"], f"c6 反推 × I4 foreach scatter 忙/閒 {i4['k_busy2']:.3f}（{i4['run_id']}）"
    return None, "忙時的每次 copy 成本沒有量（I4 只有 duty 0.7）"


def restore_time(f_idle, mult, act, tier, duty, Mm, vctx=None):
    """一個（動作, 層, duty）的 32K 還原時間（64 個 block）。回傳 dict(T, mode, b, T_cake, T_la) 或 None（缺資料）。"""
    import m10_sweep_sim as SIM                    # 純標準函式庫（m10_sweep_sim 的 simulate）
    if mult is None:
        return None
    f = [x * mult for x in f_idle]
    n = len(f)
    c = blk_costs(Mm, act, duty)
    if c is None:
        return None
    if tier["kind"] == "vllm":                     # 只有全部載入（vLLM 的 connector 載入命中的所有 block，不和重算賽跑）
        if not vctx or vctx.get("c") is None:
            return None
        L = vctx["L"]
        n_copy = L * 2 * (N_CTX * CHUNK // 16)
        if act["dec"] == "cpu":
            t = max(n_copy * vctx["c"] + n * c["h2d"], n * c["bytes"] / act["D_Bps"])
        else:
            t = n_copy * vctx["c"] + n * (c["h2d"] + c["dq_pb"] + c["dq_os"])
        return dict(T=t, mode="loadall", b=0, T_cake=None, T_la=t)
    if act["dec"] == "cpu":                        # 無損：讀壓縮後位元組、CPU 解、再 H2D 原始位元組；三段完全重疊（L5c §4.3 的假設，樂觀）
        dec_s = c["bytes"] / act["D_Bps"]
        if tier["kind"] == "pinned":
            load = max(dec_s, c["h2d"])
        else:
            load = max(c["bytes"] / (act["r"] * tier["GiBps"] * GiB), dec_s, c["h2d"])
        loads = [load] * n
        p, t, tr, tl = SIM.simulate(f, loads)
        t_cake, t_la = t, sum(loads)
    else:
        base = c["h2d"] if tier["kind"] == "pinned" else max(c["bytes"] / (tier["GiBps"] * GiB), c["h2d"])
        load = base + c["dq_pb"]
        p, t, tr, tl = SIM.simulate(f, [load] * n)
        t_cake = max(tr, tl + c["dq_os"] * (n - p))
        t_la = n * load + c["dq_os"] * n
    if t_cake <= t_la:
        return dict(T=t_cake, mode="cake", b=p, T_cake=t_cake, T_la=t_la)
    return dict(T=t_la, mode="loadall", b=0, T_cake=t_cake, T_la=t_la)


def tiers_named(model, C):
    T = [dict(name="cpu_pinned", kind="pinned", GiBps=None, src="真實 pinned H2D（micro 的 h2d 實測）"),
         dict(name="vllm_cpu", kind="vllm", GiBps=None, src="vLLM 0.28 CPU 路徑：copy 次數 × c ＋ 位元組 ÷ pinned H2D")]
    th = [dict(name=k, kind="throttle", GiBps=C.get(model, f"link_{k}"), src="bw.json") for k in ("b16", "b32", "b48")]
    for k in ("ssd_meas", "ssd_param", "nfs_meas", "nfs_param"):
        th.append(dict(name=k, kind="throttle", GiBps=C.get("-", f"link_{k}"), src="calib_c2" if k.endswith("meas") else "tier_params"))
    th = [x for x in th if x["GiBps"]]
    return T + sorted(th, key=lambda x: -x["GiBps"])


def tiers_grid():
    return [dict(name="cpu_pinned", kind="pinned", GiBps=None, src="真實 pinned H2D")] + \
        [dict(name=f"link_{g:g}", kind="throttle", GiBps=g, src="L5c §5.4 格點") for g in sorted(LINK_GRID, reverse=True)]


def build_cells(M, C, s1, i4, models=MODELS, grid=False):
    """所有（模型, 層, duty, 動作）的還原時間列。"""
    rows = []
    vc = {}
    for m in models:
        Mm = M.get(m)
        if not Mm:
            continue
        f_idle = C.get(m, "f_idle_s")
        acts = actions_for(Mm, (s1 or {}).get("models", {}).get(m) if s1 else None)
        duties = sorted({k[0] for k in Mm["t"]})
        tiers = tiers_grid() if grid else tiers_named(m, C)
        for d in duties:
            mult, msrc = f_mult(Mm, C, m, d)
            if d not in vc:
                vc[d] = vllm_copy_cost(M.get("llama"), C, i4, d)
            vctx = dict(c=vc[d][0], L=SHAPES[m][0], src=vc[d][1])
            for tier in tiers:
                for a in acts:
                    r = restore_time(f_idle, mult, a, tier, d, Mm, vctx)
                    seg = Mm["seg"][a["seg"]]
                    br = (seg["bytes_blk"] + seg.get("bytes_req", 0) / N_CTX) / Mm["seg"]["bf16"]["bytes_blk"]
                    if a["dec"] == "cpu":
                        br = 1.0 / a["r"]
                    flags = []
                    if mult is None:
                        flags.append("重算慢化 NOT_MEASURED")
                    if tier["kind"] == "vllm":
                        flags.append(f"vllm:{vctx['src']}")
                        if m != "llama":
                            flags.append("每次 copy 的成本 c 取自 Llama 的 c6（同一台機器、同一條 copy 路徑；〔推算〕）")
                    if tier["kind"] == "throttle" and d > 0:
                        flags.append("慢層＋忙：第二階段 09 §5 的模擬在這種格子偏快 23–24%")
                    if a["dec"] == "cpu":
                        flags.append("無損：讀／解／H2D 完全重疊（樂觀）")
                    if seg.get("bytes_req") and a["dec"] not in ("none", "cpu"):
                        cc = blk_costs(Mm, a, d)
                        if cc and cc["req"] == "linear":
                            flags.append(f"基底的 H2D 跟著 block 數線性外推（{N_CTX} 塊時算 {N_CTX // Mm['n_blocks']} 份，高估；"
                                         "第一版的段沒有量 h2d_req）")
                        elif cc and cc["req"] == "split":
                            flags.append("基底的 H2D 用 h2d_req 單獨算（每次還原一份）")
                    if Mm.get("seg_rid", {}).get(a["seg"], Mm["run_id"]) != Mm.get("seg_rid", {}).get("bf16", Mm["run_id"]):
                        flags.append(f"這個段的時間來自另一個 micro run（{Mm['seg_rid'][a['seg']]}；跨 run 漂移見報告）")
                    if msrc not in ("定義", NM) and not msrc.startswith("micro"):
                        flags.append(f"重算慢化取自 {msrc}")
                    rows.append(dict(model=m, grid="grid" if grid else "named", tier=tier["name"], tier_kind=tier["kind"],
                                     link_GiBps=tier["GiBps"] if tier["GiBps"] else "", duty=d, action=a["name"],
                                     segment=a["seg"], dec=a["dec"], family=a["family"], kind=a["kind"],
                                     bytes_ratio=round(br, 5), T_s=r["T"] if r else None,
                                     restore_mode=r["mode"] if r else NM, b=r["b"] if r else "",
                                     T_cake_s=r["T_cake"] if r else "", T_loadall_s=r["T_la"] if r else "",
                                     flags="；".join(flags), micro_run_id=Mm.get("seg_rid", {}).get(a["seg"], Mm["run_id"])))
    return rows


# ====================================================================== D／F／S／A 與判定
def feasible_sets(M, s1, x2, models=MODELS):
    """每個模式、每個模型可以用的動作名。X：有 X2 → Q ∪ X2 可行的 segment；沒有 → 全部（只看時間）。"""
    out = {"L": {}, "Q": {}, "X": {}}
    for m in models:
        Mm = M.get(m)
        if not Mm:
            continue
        acts = actions_for(Mm, (s1 or {}).get("models", {}).get(m) if s1 else None)
        L_ = {a["name"] for a in acts if a["family"] in ("raw", "lossless")}
        Q_ = L_ | {a["name"] for a in acts if a["seg"] in QUALITY_Q[m]}
        out["L"][m], out["Q"][m] = L_, Q_
        if x2 is None:
            out["X"][m] = {a["name"] for a in acts}
        else:
            ok = x2["feasible"].get(m, set())
            out["X"][m] = Q_ | {a["name"] for a in acts if a["seg"] in ok}
    return out


def decide(rows, feas):
    """rows：build_cells 的列；feas：{model: set(動作)}。回傳每格的決定、F、S 與聚合。"""
    cells = {}
    for r in rows:
        if r["T_s"] is None or r["model"] not in feas or r["action"] not in feas[r["model"]]:
            continue
        cells.setdefault((r["model"], r["tier"], r["duty"]), {})[r["action"]] = r["T_s"]
    cells = {k: v for k, v in cells.items() if "raw" in v}            # D 沒有的格子不算
    meta = {(r["model"], r["tier"]): r["link_GiBps"] for r in rows}
    F, S = {}, {}
    for m in sorted({k[0] for k in cells}):
        mc = [v for k, v in cells.items() if k[0] == m]
        cand = set.intersection(*[set(v) for v in mc]) if mc else set()
        if cand:
            F[m] = min(sorted(cand), key=lambda a: gm(v[a] for v in mc))
    for (m, tier) in sorted({(k[0], k[1]) for k in cells}):
        tc = [v for k, v in cells.items() if k[0] == m and k[1] == tier]
        cand = set.intersection(*[set(v) for v in tc])
        if cand:
            S[(m, tier)] = min(sorted(cand), key=lambda a: gm(v[a] for v in tc))
    out = []
    for k in sorted(cells, key=lambda k: (k[0], str(k[1]), k[2])):
        v = cells[k]
        order = sorted(v, key=lambda a: (v[a], a))
        A = order[0]
        sec = order[1] if len(order) > 1 else None
        f_, s_ = F.get(k[0]), S.get((k[0], k[1]))
        T_A, T_D = v[A], v["raw"]
        T_F = v.get(f_) if f_ else None
        T_S = v.get(s_) if s_ else None
        out.append(dict(model=k[0], tier=k[1], duty=k[2], link_GiBps=meta.get((k[0], k[1]), ""), A=A, T_A=T_A,
                        second=sec or "", T_second=v[sec] if sec else None,
                        margin2=(v[sec] / T_A - 1) if sec else None, S=s_ or "", T_S=T_S, F=f_ or "", T_F=T_F, T_D=T_D,
                        gain_vs_D=1 - T_A / T_D, gain_vs_F=(1 - T_A / T_F) if T_F else None,
                        gain_vs_S=(1 - T_A / T_S) if T_S else None, all=v))
    return dict(cells=out, F=F, S={f"{a}|{b}": s for (a, b), s in S.items()})


def aggregates(dec):
    cs = dec["cells"]
    if not cs:
        return dict(n_cells=0)

    def ag(key, tkey):
        xs = [c[tkey] for c in cs if c[tkey]]
        if len(xs) != len(cs):
            return None
        g = gm(c["T_A"] / c[tkey] for c in cs)
        return dict(gm_gain=1 - g, max_regret=max(c[tkey] / c["T_A"] - 1 for c in cs))
    gAS = [c["gain_vs_S"] for c in cs if c["gain_vs_S"] is not None]
    return dict(n_cells=len(cs), vs_D=ag("D", "T_D"), vs_F=ag("F", "T_F"), vs_S=ag("S", "T_S"),
                frac_AS_ge5=sum(g >= TH["s2a"] for g in gAS) / len(gAS) if gAS else None,
                frac_AS_gt15=sum(g > TH["go_cell"] for g in gAS) / len(gAS) if gAS else None,
                n_A_ne_S=sum(c["A"] != c["S"] for c in cs), F=dec["F"], S=dec["S"])


def fmt_of(action):
    return "lossless" if action.startswith("ll:") else ("bf16" if action == "raw" else action.split("@")[0])


def stop0(dec):
    """STOP-0（模式 Q）：每個模型裡，最常勝出的格式為主；某一格有別的格式比「主格式最好的動作」快 ≥5% 才算翻轉。沒有翻轉 → 成立。"""
    res, flips = {}, []
    if not dec["cells"]:
        return dict(holds=None, per_model={}, flips=[], verdict=NM, rule="沒有格子（micro 沒有或全被污染）")
    for m in sorted({c["model"] for c in dec["cells"]}):
        cs = [c for c in dec["cells"] if c["model"] == m]
        cnt = {}
        for c in cs:
            cnt[fmt_of(c["A"])] = cnt.get(fmt_of(c["A"]), 0) + 1
        dom = max(sorted(cnt), key=lambda k: cnt[k])
        strict = sorted(cnt)
        for c in cs:
            td = min((t for a, t in c["all"].items() if fmt_of(a) == dom), default=None)
            if fmt_of(c["A"]) != dom and (td is None or td / c["T_A"] - 1 >= TH["stop0_flip"]):
                flips.append(dict(model=m, tier=c["tier"], duty=c["duty"], A=c["A"], dominant=dom,
                                  gain_vs_dominant=None if td is None else 1 - c["T_A"] / td))
        res[m] = dict(dominant=dom, best_formats_strict=strict, counts=cnt)
    return dict(holds=not flips, per_model=res, flips=flips,
                rule="每個模型以最常勝出的格式為主；別的格式在某一格比主格式快 ≥5% 才算翻轉；沒有翻轉 → STOP-0 成立")


def stop1(s1):
    """STOP-1（模式 L）：兩個模型 BF16 無損比（T4、所有變體與 codec 的最大值）都 <1.15；或 ≤16 執行緒的 CPU 解壓連 0.5 GiB/s 都跟不上、
    又沒有 GPU 解碼器 → 成立。S1 沒有 → NOT_MEASURED。"""
    if not s1:
        return dict(holds=None, verdict=NM, why="S1 的 summary.json 沒有")
    per = {}
    for m in MODELS:
        e = s1["models"].get(m, {}).get("T4")
        per[m] = None if not e else dict(best_ratio=e["best_ratio"], best_ratio_key=e["best_ratio_key"],
                                         max_decomp_GiBps=e["max_decomp_GiBps"], dump=e["dump"])
    if any(v is None or v["best_ratio"] is None for v in per.values()):
        return dict(holds=None, verdict=NM, why="S1 缺某個模型的 T4 32K dump", per_model=per)
    low_ratio = all(v["best_ratio"] < TH["stop1_ratio"] for v in per.values())
    slow = all(v["max_decomp_GiBps"] < TH["stop1_GiBps"] for v in per.values()) and not GPU_LOSSLESS_DECODER
    return dict(holds=low_ratio or slow, low_ratio=low_ratio, slow_cpu_no_gpu=slow, per_model=per,
                verdict="STOP-1 成立（模式 L NO-GO）" if (low_ratio or slow) else "STOP-1 不成立", s1_run_id=s1["run_id"])


def stop2(agg):
    """STOP-2a／2b／GO_N5／STOP-2c（L5c §5.5）。"""
    if not agg.get("n_cells") or agg.get("vs_F") is None or agg.get("frac_AS_ge5") is None:
        return dict(verdict=NM, why="格子不完整（F 或 S 不是每格都有）")
    gAF = agg["vs_F"]["gm_gain"]
    if agg["frac_AS_ge5"] == 0:
        v = "STOP-2a：每一格 A 對 S 都好不到 5% → GPU 忙碌這個軸沒有價值（退回前作）"
        code = "STOP-2a"
    elif gAF < TH["s2b"]:
        v, code = f"STOP-2b：A 對 F 的幾何平均只好 {gAF:.1%}（<5%）→ NO-GO", "STOP-2b"
    elif agg["frac_AS_gt15"] >= TH["go_frac"] and gAF > TH["go_gm"]:
        v, code = f"GO_N5：A 對 S 在 {agg['frac_AS_gt15']:.0%} 的格子好 >15%，A 對 F 幾何平均 {gAF:.1%}", "GO_N5"
    else:
        v, code = (f"STOP-2c：介於中間（A 對 F 幾何平均 {gAF:.1%}；A 對 S >15% 的格子 {agg['frac_AS_gt15']:.0%}）→ 停下來問使用者"), "STOP-2c"
    return dict(code=code, verdict=v, gain_AF_gm=gAF, frac_AS_ge5=agg["frac_AS_ge5"], frac_AS_gt15=agg["frac_AS_gt15"])


def stacking_verdicts(dec, M):
    """L5d §6.4 第三階段的 T-a～T-d（只對模式 X 有意義）。"""
    cs = dec["cells"]
    if not cs:
        return dict(verdict=NM)
    ta_cells, wins = [], []
    for c in cs:
        comb = {a: t for a, t in c["all"].items() if a.split("@")[0] in COMBO_SEGS}
        sing = {a: t for a, t in c["all"].items() if a.split("@")[0] not in COMBO_SEGS}
        if not comb or not sing:
            continue
        bc, bs = min(comb.values()), min(sing.values())
        g = 1 - bc / bs
        ta_cells.append(dict(model=c["model"], tier=c["tier"], duty=c["duty"], gain_combo_vs_single=g,
                             link=c["link_GiBps"]))
        if g >= TH["ta"]:
            wins.append(ta_cells[-1])
    ta = bool(ta_cells) and not wins
    tb = None
    if wins:
        only_nfs = all(w["link"] not in ("", None) and float(w["link"]) < NFS_CLASS_GIBPS for w in wins)
        cpu_lose = all(x["gain_combo_vs_single"] < 0 for x in ta_cells if x["tier"] == "cpu_pinned")
        tb = only_nfs and cpu_lose
    # T-c：所有非量化解碼段（含組合）的忙/閒慢化，和 INT4 一次拆差 <20%
    tc_detail, tc = [], True
    for m, Mm in M.items():
        if not Mm:
            continue
        t = Mm["t"]
        base = {d: t[(d, "int4_kivi", "dq_oneshot")]["med"] for d in {k[0] for k in t}
                if (d, "int4_kivi", "dq_oneshot") in t}
        if 0.0 not in base:
            continue
        for s, info in Mm["seg"].items():
            if info["family"] in ("raw", "quant") or (0.0, s, "dq_oneshot") not in t:
                continue
            for d in sorted(base):
                if d == 0.0 or (d, s, "dq_oneshot") not in t:
                    continue
                sa = t[(d, s, "dq_oneshot")]["med"] / t[(0.0, s, "dq_oneshot")]["med"]
                si = base[d] / base[0.0]
                diff = sa / si - 1
                tc_detail.append(dict(model=m, segment=s, duty=d, slow_seg=sa, slow_int4=si, rel_diff=diff))
                if abs(diff) >= TH["tc"]:
                    tc = False
    # T-d：最佳組合隨 duty 換人而且比 S 快 ≥15%（新軸）；只隨連線換人另記（前作）
    duty_flip = [dict(model=c["model"], tier=c["tier"], duty=c["duty"], A=c["A"], S=c["S"], gain_vs_S=c["gain_vs_S"])
                 for c in cs if c["A"] != c["S"] and c["gain_vs_S"] is not None and c["gain_vs_S"] >= TH["td"]]
    link_flip = []
    for m in sorted({c["model"] for c in cs}):
        for d in sorted({c["duty"] for c in cs if c["model"] == m}):
            row = [c for c in cs if c["model"] == m and c["duty"] == d]
            if len({c["A"] for c in row}) > 1 and any((c["gain_vs_F"] or 0) >= TH["td"] for c in row):
                link_flip.append(dict(model=m, duty=d, best_by_tier={c["tier"]: c["A"] for c in row}))
    return dict(T_a=dict(holds=ta, n_cells=len(ta_cells), n_wins=len(wins), wins=wins[:20],
                         rule="每一格 ε 內最快的組合都沒有比最快的單一方法快 ≥5% → NO-GO"),
                T_b=dict(holds=tb, rule="組合只在 NFS 級（<1 GiB/s）贏、在 CPU 層輸 → 寫成條件規則，不算新意"),
                T_c=dict(holds=tc if tc_detail else None, detail=tc_detail,
                         rule="所有非量化解碼段的忙/閒慢化和 INT4 一次拆差 <20% → GPU 忙不改變排序，不另立成本項"),
                T_d=dict(holds=bool(duty_flip), duty_flip=duty_flip[:20], link_flip=link_flip[:20],
                         rule="最佳動作隨 duty 換人、而且比 S 快 ≥15% → 新意成立；只隨連線換人＝前作"))


def el3(M):
    """L5a E-L3：低秩重建（r＝24、40，一次拆）的忙(0.7)/閒倍數 vs INT4 一次拆；忙時重建 vs 同位元組的 H2D（它自己的 h2d）。"""
    out = {}
    for m, Mm in M.items():
        if not Mm:
            continue
        t = Mm["t"]
        for r in (24, 40):
            s = f"lr_r{r}"
            need = [(0.0, s, "dq_oneshot"), (0.7, s, "dq_oneshot"), (0.0, "int4_kivi", "dq_oneshot"),
                    (0.7, "int4_kivi", "dq_oneshot"), (0.7, s, "h2d_region")]
            if not all(k in t for k in need):
                out[f"{m}|{s}"] = dict(verdict=NM)
                continue
            rl = t[(0.7, s, "dq_oneshot")]["med"] / t[(0.0, s, "dq_oneshot")]["med"]
            ri = t[(0.7, "int4_kivi", "dq_oneshot")]["med"] / t[(0.0, "int4_kivi", "dq_oneshot")]["med"]
            frac = t[(0.7, s, "dq_oneshot")]["med"] / t[(0.7, s, "h2d_region")]["med"]
            if rl >= TH["el3_hi"] * ri or frac >= TH["el3_h2d_hi"]:
                v = "新成本項（重建在忙時比反量化敏感，或和搬運同量級）"
            elif abs(rl / ri - 1) < TH["el3_lo"] and frac < TH["el3_h2d_lo"]:
                v = "不另立項（重建只是另一種反量化）"
            else:
                v = "中間（照實報，不下結論）"
            out[f"{m}|{s}"] = dict(slow_lr=rl, slow_int4=ri, busy_recon_over_h2d=frac, verdict=v)
    return out


def anchors(M, C, paths=None):
    """map 的組成（micro 的 h2d＋dq）和第二階段實測（CPU 層、全部載入、E0／E1）比，量組成模型本身準不準。"""
    p = dict(DEFAULT_PATHS, **(paths or {}))
    out = []
    for m, Mm in M.items():
        if not Mm:
            continue
        fp8 = C.get(m, "fp8_fmt")
        specs = [("q4", f"{m}_int4_P3", "E0", "raw", {"idle": 0.0, "busy": 0.7}),
                 ("q4", f"{m}_int4_P3", "E1", "int4_kivi@oneshot", {"idle": 0.0, "busy": 0.7}),
                 ("q4", f"{m}_int4_P0", "E1", "int4_kivi@perblock", {"idle": 0.0, "busy": 0.7}),
                 ("q4", f"{m}_fp8_P3", "E1", "fp8@oneshot", {"idle": 0.0, "busy": 0.7}),
                 ("q4duty", f"{m}_int4_P3", "E1", "int4_kivi@oneshot", {"busy1": 0.4, "busy2": 0.7, "busy3": 0.9})]
        for base, sub, cfg, act, states in specs:
            d = os.path.join(p[base], sub)
            rp = os.path.join(d, "restores.csv")
            if not os.path.exists(rp):
                continue
            rows = read_csv(rp)
            last = max(r["run_id"] for r in rows)
            rows = [r for r in rows if r["run_id"] == last]
            gs = [json.load(open(g)) for g in glob.glob(os.path.join(d, f"gpu_guard_{last}_*.json"))]
            usable = bool(gs) and all(g.get("verdict") == "CLEAN" for g in gs) and \
                all(r["contaminated_gpu"] == "False" for r in rows)
            a = dict(name=act, seg=act.split("@")[0] if act != "raw" else "bf16",
                     dec=act.split("@")[1] if "@" in act else "none")
            for st, duty in states.items():
                v = [float(r["t_restore"]) for r in rows if (r["config"], r["mode"], r["gpu_state"], r["bw_name"])
                     == (cfg, "loadall", st, "cpu")]
                if not v or a["seg"] not in Mm["seg"]:
                    continue
                c = blk_costs(Mm, a, duty)
                if c is None:
                    continue
                pred = N_CTX * (c["h2d"] + c["dq_pb"] + c["dq_os"])
                meas = med(v)
                out.append(dict(model=m, stage2=f"{base}/{sub}", run_id=last, usable=usable, config=cfg, action=act,
                                duty=duty, meas_s=meas, pred_s=pred, rel_err=pred / meas - 1,
                                warn=abs(pred / meas - 1) > TH["anchor_warn"]))
    return out


def write_cost_table(M, C, models=MODELS):
    """10/21：xKV 的寫入（壓縮）成本表（不進還原決策）。每（模型, 10/21 的 xKV 段, duty）：K、V 兩半的 svd_lowrank（含逆 RoPE、組 Z、取 A／B）
    ＋係數量化（bf16＝轉型；kivi4／int8_token＝m10_codec_x.enc_half）。參考尺度：同一個載入區（最後 n 個 block）閒時重算的總時間
    （bw.json 的 f(i) 加總）——只是比例尺，不是判定。缺任何一項 → NOT_MEASURED（不是 0）。"""
    rows = []
    for m in models:
        Mm = M.get(m)
        if not Mm or not Mm.get("wcost"):
            continue
        n = Mm["n_blocks"]
        f_idle = C.get(m, "f_idle_s")
        rec = sum(f_idle[N_CTX - n:]) if f_idle else None
        cb = C.get(m, "chunk_bytes") or SHAPES[m][0] * 2 * SHAPES[m][1] * CHUNK * SHAPES[m][2] * 2
        duties = sorted({k[0] for k in Mm["wcost"]})
        for seg, (rk, rv, cf) in XKV2.items():
            pair = f"k{rk}v{rv}"
            coef, basis = xkv_bytes(m, rk, rv, cf, n * CHUNK)
            for d in duties:
                parts = [Mm["wcost"].get((d, pair, op)) for op in ("svd_k", "svd_v", f"q_{cf}")]
                ok = all(parts)
                tot = sum(p["med"] for p in parts) if ok else None
                rids = sorted({p["run_id"] for p in parts if p})
                rows.append(dict(model=m, segment=seg, spec=XKV2_SPEC[seg], duty=d,
                                 t_svd_k_s=parts[0]["med"] if parts[0] else NM, t_svd_v_s=parts[1]["med"] if parts[1] else NM,
                                 t_quant_s=parts[2]["med"] if parts[2] else NM, t_total_s=tot if ok else NM,
                                 t_total_per_block_s=tot / n if ok else NM, n_blocks=n, bytes_coef=coef, bytes_basis=basis,
                                 ratio_vs_bf16=round((coef + basis) / (n * cb), 6), recompute_region_idle_s=rec if rec else NM,
                                 total_over_recompute=(tot / rec) if ok and rec else NM, micro_run_id=";".join(rids),
                                 note="" if ok else "缺 " + "、".join(op for op, p in zip(("svd_k", "svd_v", f"q_{cf}"), parts)
                                                                       if not p) + "（沒量或被污染）"))
    return rows


def select_flips(decs, verdicts, max_cells=8):
    """validate 的格子：GO_N5 的模式（L、Q；X 要有 X2 且 T-d 成立）裡，(1) A≠S 且好 ≥5%、(2) A≠F 且好 ≥5%、(3) 前兩名差 <10%。
    vllm_cpu 不選（沒有還原路徑可以驗）。每格要量的動作＝A、S、F、D（去重）。"""
    modes = [md for md in ("L", "Q") if verdicts.get(md, {}).get("stop2", {}).get("code") == "GO_N5"]
    if verdicts.get("X", {}).get("x2") and (verdicts["X"].get("stop2", {}).get("code") == "GO_N5"
                                           or verdicts["X"].get("stacking", {}).get("T_d", {}).get("holds")):
        modes.append("X")
    cand = []
    for md in modes:
        for c in decs[md]["cells"]:
            if c["tier"] == "vllm_cpu":
                continue
            gs, gf = c["gain_vs_S"] or 0, c["gain_vs_F"] or 0
            if c["A"] != c["S"] and gs >= TH["s2a"]:
                pr = 0
            elif c["A"] != c["F"] and gf >= TH["s2b"]:
                pr = 1
            elif c["margin2"] is not None and c["margin2"] < 0.10:
                pr = 2
            else:
                continue
            cand.append((pr, -max(gs, gf), md, c))
    cand.sort(key=lambda x: (x[0], x[1], x[2], x[3]["model"], x[3]["tier"], x[3]["duty"]))
    seen, out = set(), []
    for pr, _, md, c in cand:
        key = (c["model"], c["tier"], c["duty"])
        if key in seen:
            continue
        seen.add(key)
        meas = []
        for role, a in (("A", c["A"]), ("S", c["S"]), ("F", c["F"]), ("D", "raw")):
            if not a:
                continue
            hit = [x for x in meas if x["action"] == a]
            if hit:                                            # 同一個動作身兼數職（例：A＝S）→ 只量一次，角色記成 A+S
                hit[0]["role"] += "+" + role
            else:
                meas.append(dict(action=a, role=role, pred_s=c["all"].get(a)))
        out.append(dict(cell=f"{c['model']}|{c['tier']}|{c['duty']}", mode=md, model=c["model"], tier=c["tier"],
                        link_GiBps=c["link_GiBps"], duty=c["duty"], priority=pr, A=c["A"], S=c["S"], F=c["F"],
                        gain_vs_S=c["gain_vs_S"], gain_vs_F=c["gain_vs_F"], margin2=c["margin2"], measure=meas))
        if len(out) >= max_cells:
            break
    return modes, out


# ====================================================================== 輸出（CSV／JSON／Markdown／SVG）
FAM_COLOR = {"raw": "#9aa0a6", "lossless": "#2a9d8f", "quant": "#3a6ea5", "lowrank": "#e07a1f", "xlayer": "#8d5a3b",
             "merge": "#7b4fa3", "drop": "#c0392b", "combo": "#2e8b57"}


def short(a):
    if a == "raw":
        return "BF16"
    if a.startswith("ll:"):
        v, c, t = a[3:].split("|")
        return f"LL {v} {c} {t}"
    s, d = a.split("@")
    return f"{s}·{'一次' if d == 'oneshot' else '逐塊'}"


def family_of(a, rows_by_action):
    return rows_by_action.get(a, "raw")


def svg_map(path, title, dec, fam, tiers, duties):
    """決策圖：列＝層（快 → 慢），欄＝duty；格子顏色＝最佳動作的家族，字＝動作與對 D 的好處；第一名和第二名差 <5% 畫斜線。"""
    cw, ch, lw, top = 150, 34, 150, 50
    W, Hh = lw + cw * len(duties) + 20, top + ch * len(tiers) + 70
    by = {(c["tier"], c["duty"]): c for c in dec["cells"]}
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Noto Sans CJK TC, sans-serif" font-size="11">',
         '<defs><pattern id="h" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
         '<line x1="0" y1="0" x2="0" y2="6" stroke="#fff" stroke-width="2"/></pattern></defs>',
         '<rect width="100%" height="100%" fill="#ffffff"/>',
         f'<text x="10" y="20" font-size="14" font-weight="bold">{title}</text>']
    for j, d in enumerate(duties):
        o.append(f'<text x="{lw + j * cw + cw / 2}" y="{top - 8}" text-anchor="middle">duty {d:g}</text>')
    for i, t in enumerate(tiers):
        y = top + i * ch
        o.append(f'<text x="{lw - 6}" y="{y + ch / 2 + 4}" text-anchor="end">{t}</text>')
        for j, d in enumerate(duties):
            x = lw + j * cw
            c = by.get((t, d))
            if not c:
                o.append(f'<rect x="{x}" y="{y}" width="{cw - 2}" height="{ch - 2}" fill="#eeeeee"/>'
                         f'<text x="{x + 4}" y="{y + ch / 2 + 4}" fill="#666">{NM}</text>')
                continue
            col = FAM_COLOR.get(fam.get(c["A"], "raw"), "#999")
            o.append(f'<rect x="{x}" y="{y}" width="{cw - 2}" height="{ch - 2}" fill="{col}"/>')
            if c["margin2"] is not None and c["margin2"] < TH["near_tie"]:
                o.append(f'<rect x="{x}" y="{y}" width="{cw - 2}" height="{ch - 2}" fill="url(#h)"/>')
            o.append(f'<text x="{x + 4}" y="{y + 13}" fill="#fff">{short(c["A"])}</text>'
                     f'<text x="{x + 4}" y="{y + 26}" fill="#fff">{c["T_A"] * 1e3:.1f} ms，比 D {c["gain_vs_D"]:+.0%}</text>')
    ly = top + ch * len(tiers) + 18
    for k, (f_, col) in enumerate(FAM_COLOR.items()):
        o.append(f'<rect x="{10 + k * 95}" y="{ly}" width="12" height="12" fill="{col}"/>'
                 f'<text x="{26 + k * 95}" y="{ly + 10}">{f_}</text>')
    o.append(f'<text x="10" y="{ly + 32}" fill="#444">斜線＝第一名和第二名差 &lt;5%；灰＝缺資料（NOT_MEASURED）</text>')
    o.append("</svg>")
    open(path, "w").write("\n".join(o))


def _f(x, k=3, pct=False):
    if x is None or x == "":
        return "—"
    return f"{x:+.1%}" if pct else f"{x:.{k}f}"


def write_report(out, summ, decs, C):
    md = [f"# X3：壓縮組合的還原時間 × 連線速度 × GPU 忙（map，run {summ['run_id']}）", "",
          f"> 產生時間 {summ['ts']}；micro：{', '.join(f'{m} {r}' for m, r in summ['micro_run_ids'].items())}。"
          " 全部是〔模擬〕：零件是 micro 的實測，還原規則照第二階段的還原器（m10_sweep_sim）。",
          f"> micro 目錄：{summ.get('micro_dirs')}；X2 品質可行集：{summ['x2']}。", ""]
    if summ.get("smoke"):
        md += ["**冒煙資料：數字不可用，只驗管線。**", ""]
    md += ["## 判定（門檻在看數字前寫死，x3_time_howto.md §4）", ""]
    v = summ["verdicts"]
    md += [f"- STOP-0（模式 Q）：{ {True: '成立', False: '不成立', None: NM}[v['Q']['stop0']['holds']] }；"
           f"翻轉 {len(v['Q']['stop0']['flips'])} 格。",
           f"- STOP-1（模式 L）：{v['L']['stop1'].get('verdict')}（{v['L']['stop1'].get('why', '')}）"]
    for mode in ("L", "Q", "X"):
        s2 = v[mode]["stop2"]
        md.append(f"- 模式 {mode}：{s2.get('verdict', s2)}" + ("（只看時間：X2 沒有，不下結論）" if mode == "X" and not v["X"]["x2"] else ""))
    st = v["X"].get("stacking", {})
    if "T_a" in st:
        md.append(f"- T-a：{st['T_a']['holds']}；T-b：{st['T_b']['holds']}；T-c：{st['T_c']['holds']}；T-d：{st['T_d']['holds']}"
                  + ("（只看時間：X2 沒有，T-a～T-d 不下結論）" if not v["X"]["x2"] else ""))
    else:
        md.append(f"- T-a～T-d：{st.get('verdict', NM)}")
    if v["X"].get("x2_feasible_unmeasured"):
        md.append(f"- ⚠ 模式 X 不完整：X2 判可行、但沒有 micro 時間的段 {v['X']['x2_feasible_unmeasured']}")
    for k, e in summ["el3"].items():
        md.append(f"- E-L3 {k}：{e['verdict']}" + (f"（重建忙/閒 {e['slow_lr']:.2f}×、INT4 {e['slow_int4']:.2f}×、"
                                                   f"忙時重建÷H2D {e['busy_recon_over_h2d']:.2f}）" if "slow_lr" in e else ""))
    md += ["", f"validate：{'要跑' if summ['worth_validate'] else '不跑'}——{summ['validate_reason']}", "",
           "格點版本（L5c §5.4 的連線格點，只當敏感度，不拿來判定）：" +
           "；".join(f"{md_}：{v[md_]['grid_stop2'].get('code', v[md_]['grid_stop2'].get('verdict'))}" for md_ in ("L", "Q", "X")), ""]
    md += ["## 聚合（每個模式；gain＝1 − T_A ÷ T_基準 的幾何平均）", "",
           "| 模式 | 格子 | A 對 D | A 對 F | A 對 S | A 對 S ≥5% 的格子 | >15% | F（每個模型） |", "|:--|--:|--:|--:|--:|--:|--:|:--|"]
    for mode in ("L", "Q", "X"):
        a = summ["aggregates"][mode]
        if not a.get("n_cells"):
            md.append(f"| {mode} | 0 | — | — | — | — | — | — |")
            continue
        md.append(f"| {mode} | {a['n_cells']} | {_f((a['vs_D'] or {}).get('gm_gain'), pct=True)} | "
                  f"{_f((a['vs_F'] or {}).get('gm_gain'), pct=True)} | {_f((a['vs_S'] or {}).get('gm_gain'), pct=True)} | "
                  f"{_f(a['frac_AS_ge5'], 2)} | {_f(a['frac_AS_gt15'], 2)} | {a['F']} |")
    md.append("")
    for mode in ("L", "Q", "X"):
        md += [f"## 模式 {mode}：每格的最佳動作", "",
               "| 模型 | 層 | GiB/s | duty | A | T_A (ms) | 第二名 | 差 | S | F | 對 D | 對 F | 對 S |",
               "|:--|:--|--:|--:|:--|--:|:--|--:|:--|:--|--:|--:|--:|"]
        for c in decs[mode]["cells"]:
            lk = f"{float(c['link_GiBps']):.3g}" if c["link_GiBps"] not in ("", None) else "—"
            md.append(f"| {c['model']} | {c['tier']} | {lk} | {c['duty']:g} | "
                      f"{c['A']} | {c['T_A'] * 1e3:.2f} | {c['second'] or '—'} | {_f(c['margin2'], pct=True)} | {c['S']} | "
                      f"{c['F']} | {_f(c['gain_vs_D'], pct=True)} | {_f(c['gain_vs_F'], pct=True)} | {_f(c['gain_vs_S'], pct=True)} |")
        md.append("")
    md += ["## 組成模型對第二階段實測（CPU 層、全部載入）", "",
           "| 模型 | 第二階段 | 配置 | 動作 | duty | 實測 (ms) | 組成 (ms) | 差 | 可用 |", "|:--|:--|:--|:--|--:|--:|--:|--:|:--|"]
    for x in summ["anchors"]:
        md.append(f"| {x['model']} | {x['stage2']}（{x['run_id']}） | {x['config']} | {x['action']} | {x['duty']:g} | "
                  f"{x['meas_s'] * 1e3:.2f} | {x['pred_s'] * 1e3:.2f} | {x['rel_err']:+.1%}{' ⚠' if x['warn'] else ''} | "
                  f"{'是' if x['usable'] else '**否**'} |")
    if summ.get("drift"):
        md += ["", "## micro 跨 run 漂移（10/21：多個 micro 目錄合併；錨點＝重疊的 segment，用第一個目錄的值，這裡只看差多少）", "",
               f"差 >{TH['anchor_warn']:.0%} 打 ⚠（不停；那個 duty 下兩次 run 拼起來的比較要打折看）。", "",
               "| 模型 | duty | segment | op | 第一個 (ms) | 後來 (ms) | 後÷前 | run |", "|:--|--:|:--|:--|--:|--:|--:|:--|"]
        for x in summ["drift"]:
            md.append(f"| {x['model']} | {x['duty']:g} | {x['segment']} | {x['op']} | {x['t_first_s'] * 1e3:.3f} | "
                      f"{x['t_later_s'] * 1e3:.3f} | {x['ratio']:.3f}{' ⚠' if x['warn'] else ''} | "
                      f"{x['first_run_id']} → {x['later_run_id']} |")
    if summ.get("write_cost"):
        md += ["", "## 寫入路徑成本：xKV 壓縮（10/21；不進還原決策圖）", "",
               "每題一次：K、V 兩半的 torch.svd_lowrank（每 4 層一個窗口、K 先逆 RoPE；niter、多取幾個、seed 照 m16_xstack）＋係數量化。"
               "載入區形狀（最後 n 個 block）。「÷ 重算」＝除以同一段閒時重算的總時間（bw.json，只是比例尺）。", "",
               "| 模型 | 段 | 組合 | duty | SVD K (ms) | SVD V (ms) | 量化 (ms) | 合計 (ms) | 每 block (ms) | 位元組比 | ÷ 重算 |",
               "|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for x in summ["write_cost"]:
            def ms(v):
                return f"{v * 1e3:.1f}" if isinstance(v, (int, float)) else str(v)
            md.append(f"| {x['model']} | {x['segment']} | {x['spec']} | {x['duty']:g} | {ms(x['t_svd_k_s'])} | {ms(x['t_svd_v_s'])} | "
                      f"{ms(x['t_quant_s'])} | {ms(x['t_total_s'])} | {ms(x['t_total_per_block_s'])} | {x['ratio_vs_bf16']:.4f} | "
                      + (f"{x['total_over_recompute']:.2f}" if isinstance(x["total_over_recompute"], float) else str(x["total_over_recompute"]))
                      + " |")
    md += ["", "## 常數（出處見 constants_used.csv）", "", "| 模型 | 常數 | 值 | 單位 | 檔 | run_id |", "|:--|:--|:--|:--|:--|:--|"]
    for r in C.rows:
        val = f"{r['value']:.6g}" if isinstance(r["value"], float) else r["value"]
        md.append(f"| {r['model']} | {r['const']} | {val} | {r['unit']} | {r['file']} | {r['run_id']} |")
    md += ["", "## 注意", ""] + [f"- {x}" for x in summ["notes"]]
    open(os.path.join(out, "map_report.md"), "w").write("\n".join(md) + "\n")


# ====================================================================== map 主程式（只用 CPU）
def run_map(micro_dir, out, paths=None, models=MODELS, C=None, M=None, s1=None, x2=None, i4=None, smoke=None):
    """可以從檔案讀（cmd_map），也可以直接給 C／M／s1／x2／i4（自測用）。回傳 summary dict。"""
    p = dict(DEFAULT_PATHS, **(paths or {}))
    rid, ts = run_id(), now_iso()
    os.makedirs(out, exist_ok=True)
    if C is None:
        C = load_constants(models, p)
    if M is None:
        M = load_micro_dirs(micro_dir, models)
    if s1 is None and p.get("s1"):                     # paths 給空字串＝不讀（自測用）
        s1 = load_s1(p["s1"], models)
    if x2 is None and p.get("x2"):
        x2 = load_x2(p["x2"])
    if i4 is None and p.get("i4"):
        i4 = load_i4(p["i4"])
    notes = []
    for m in models:
        Mm = M.get(m)
        if not Mm:
            notes.append(f"{m}：沒有 micro 的輸出（{micro_dir}）→ 這個模型整個 NOT_MEASURED")
            continue
        cb = C.get(m, "chunk_bytes")
        if cb is not None and Mm["seg"]["bf16"]["bytes_blk"] != cb:          # 規則 6：兩條路徑算同一個量要對得上
            raise ValueError(f"{m}：micro 的 BF16 每 block 位元組 {Mm['seg']['bf16']['bytes_blk']} ≠ bw.json 的 {cb}")
        if Mm["dropped"]:
            notes.append(f"{m}：{len(Mm['dropped'])} 列被污染，不用（例：{Mm['dropped'][:3]}）→ 那些格子是 NOT_MEASURED，要重量")
        for d in sorted({k[0] for k in Mm["t"]}):
            mult, src = f_mult(Mm, C, m, d)
            notes.append(f"{m} duty {d:g}：重算慢化 {('%.3f' % mult) if mult else NM}（{src}）")
    if smoke is None:
        smoke = any(Mm and Mm["smoke"] for Mm in M.values())
    drift = [dict(model=m, **x) for m in models if M.get(m) for x in M[m].get("drift", [])]
    if any(x["warn"] for x in drift):
        notes.append(f"跨 run 漂移：{sum(x['warn'] for x in drift)} 個錨點的兩次 micro 差 >{TH['anchor_warn']:.0%}（見報告的漂移表）")
    for m in models:
        if M.get(m) and len(M[m].get("dirs", [])) > 1:
            notes.append(f"{m}：micro 由 {len(M[m]['dirs'])} 個目錄合併（{'、'.join(rel(d) for d in M[m]['dirs'])}；第一個為主，"
                         f"後面只補沒有的 segment）")
    wct = write_cost_table(M, C, models)
    rows = build_cells(M, C, s1, i4, models)
    grows = build_cells(M, C, s1, i4, models, grid=True)
    for r in rows + grows:
        r.update(run_id=rid, ts=ts)
    write_csv(os.path.join(out, "map_cells.csv"), rows + grows, CELL_FIELDS)
    feas = feasible_sets(M, s1, x2, models)
    decs, gdecs, aggs, verd = {}, {}, {}, {}
    fam = {r["action"]: r["family"] for r in rows}
    for mode in ("L", "Q", "X"):
        decs[mode] = decide(rows, feas[mode])
        gdecs[mode] = decide(grows, feas[mode])
        aggs[mode] = aggregates(decs[mode])
        verd[mode] = dict(stop2=stop2(aggs[mode]), grid_stop2=stop2(aggregates(gdecs[mode])))
    verd["Q"]["stop0"] = stop0(decs["Q"])
    verd["L"]["stop1"] = stop1(s1)
    if verd["L"]["stop1"].get("holds") is None:
        verd["L"]["stop2"] = dict(code=NM, verdict="NOT_MEASURED（S1 沒有：模式 L 只剩 raw）")
    elif verd["L"]["stop1"]["holds"]:
        verd["L"]["stop2"] = dict(code="STOP-1", verdict="STOP-1 成立 → 模式 L NO-GO（D／F／S／A 的數字照報）",
                                  stop2_would_be=verd["L"]["stop2"])
    verd["X"]["x2"] = bool(x2)
    if x2:                                     # 10/21：X2 判可行、但 micro 沒有時間的段 → 模式 X 的判定不完整（照實寫出來，不默默略過）
        miss = {m: sorted(x2["feasible"].get(m, set()) - set(M[m]["seg"])) for m in models if M.get(m)}
        miss = {m: v for m, v in miss.items() if v}
        verd["X"]["x2_feasible_unmeasured"] = miss
        if miss:
            notes.append(f"⚠ X2 判可行、但 micro 沒有時間的段：{miss} → 模式 X 的判定少了它們（不完整）")
    verd["X"]["stacking"] = stacking_verdicts(decs["X"], M)
    if not x2:
        verd["X"]["stop2"] = dict(verd["X"]["stop2"], code="TIME_ONLY",
                                  verdict="只看時間（X2 品質可行集還沒有）：數字照報，不下結論")
    e3 = el3(M)
    anc = anchors(M, C, p)
    if any(x["warn"] for x in anc):
        notes.append("組成模型和第二階段實測差 >15% 的格子見「組成模型對第二階段實測」表；那種動作的預測要先打折看")
    vmodes, flips = select_flips(decs, verd)
    if vmodes and len(flips) >= 1:
        worth, reason = True, f"模式 {vmodes} 判 GO_N5（或 X 的 T-d 成立）；挑了 {len(flips)} 格"
    else:
        reason = "；".join(f"{md}：{verd[md]['stop2'].get('code', verd[md]['stop2'].get('verdict'))}" for md in ("L", "Q", "X"))
        if verd["Q"]["stop0"]["holds"]:
            reason += "；STOP-0 成立"
        worth, reason = False, "沒有模式判 GO_N5 → 照 L5c §5.5 不做 N5 驗證（" + reason + "）"
    if smoke:
        worth, reason = bool(flips), "冒煙：不判，只驗管線"
    # 決策圖
    for mode in ("L", "Q", "X"):
        for m in models:
            if not M.get(m):
                continue
            sub = dict(cells=[c for c in decs[mode]["cells"] if c["model"] == m])
            if sub["cells"]:
                tiers = [t["name"] for t in tiers_named(m, C)]
                duties = sorted({c["duty"] for c in sub["cells"]})
                svg_map(os.path.join(out, f"decision_{mode}_{m}.svg"),
                        f"X3 模式 {mode}・{m}（名目層）{'・只看時間' if mode == 'X' and not x2 else ''}", sub, fam, tiers, duties)
            gsub = dict(cells=[c for c in gdecs[mode]["cells"] if c["model"] == m])
            if gsub["cells"]:
                svg_map(os.path.join(out, f"decision_grid_{mode}_{m}.svg"), f"X3 模式 {mode}・{m}（L5c 連線格點）", gsub, fam,
                        [t["name"] for t in tiers_grid()], sorted({c["duty"] for c in gsub["cells"]}))
    drows = []
    for mode in ("L", "Q", "X"):
        for c in decs[mode]["cells"]:
            drows.append(dict({k: v for k, v in c.items() if k != "all"}, run_id=rid, ts=ts, mode=mode,
                              micro_run_id=M[c["model"]]["run_id"],
                              flags="近乎平手" if c["margin2"] is not None and c["margin2"] < TH["near_tie"] else ""))
    write_csv(os.path.join(out, "map_decision.csv"), drows, DEC_FIELDS)
    if wct:
        write_csv(os.path.join(out, "write_cost.csv"), [dict(r, run_id=rid, ts=ts) for r in wct], WCT_FIELDS)
    write_csv(os.path.join(out, "constants_used.csv"),
              [dict(r, run_id_map=rid, ts=ts) for r in C.rows],
              ["run_id_map", "ts", "model", "const", "value", "unit", "file", "run_id", "how"])
    summ = dict(run_id=rid, ts=ts, smoke=smoke, micro_run_ids={m: M[m]["run_id"] for m in models if M.get(m)},
                s1=dict(run_id=s1["run_id"], file=rel(s1["file"])) if s1 else NM,
                x2=dict(run_id=x2["run_id"], file=rel(x2["file"])) if x2 else NM,
                i4=dict(k_busy2=i4["k_busy2"], run_id=i4["run_id"]) if i4 else NM,
                thresholds=TH, quality_Q=dict(sets=QUALITY_Q, src=QUALITY_Q_SRC),
                aggregates=aggs, verdicts=verd, el3=e3, anchors=anc, worth_validate=worth, validate_reason=reason,
                notes=notes, micro_dirs={m: [rel(d) for d in M[m].get("dirs", [])] for m in models if M.get(m)},
                drift=drift, write_cost=wct)
    json.dump(summ, open(os.path.join(out, "map_summary.json"), "w"), indent=1, ensure_ascii=False, default=str)
    json.dump(dict(run_id=rid, ts=ts, worth_validate=worth, reason=reason, modes=vmodes, cells=flips,
                   micro_run_ids=summ["micro_run_ids"], smoke=smoke),
              open(os.path.join(out, "flip_cells.json"), "w"), indent=1, ensure_ascii=False, default=str)
    write_report(out, summ, decs, C)
    return summ


def guard_map_out(out, micro_dirs):
    """10/21：合併多個 micro 目錄（補缺口）的 map 不准寫進第一版的目錄（會蓋掉第一版的 map_*）。回傳錯誤訊息或 None。"""
    if len(micro_dirs) > 1 and os.path.realpath(out) == os.path.realpath(RES):
        return f"合併 {len(micro_dirs)} 個 micro 目錄的 map 不准寫進第一版的目錄 {rel(RES)}（用 --out {rel(os.path.join(RES, 'v2'))}）"
    return None


def guard_micro_out(out, segments):
    """10/21：只量部分 segment 的 micro 不准寫進第一版的目錄（會蓋掉全量的 micro_<m>.csv）。回傳錯誤訊息或 None。"""
    if segments and set(segments) | {"bf16"} != set(seg_catalog()) and os.path.realpath(out) == os.path.realpath(RES):
        return f"只量部分 segment（{len(set(segments))} 個）的 micro 不准寫進第一版的目錄 {rel(RES)}（會蓋掉全量的 micro_<m>.csv）"
    return None


def cmd_map(a):
    dirs = a.micro_dir if isinstance(a.micro_dir, list) else [a.micro_dir]
    err = guard_map_out(a.out, dirs)
    if err:
        raise SystemExit(err)
    paths = {}
    for k in ("s1", "x2", "i4"):
        v = getattr(a, k)
        if v:
            if not os.path.exists(v):                # 明講要讀的檔不在 → 停（規則 7：查不到不等於沒有；不默默當成「沒有 X2」）
                raise SystemExit(f"--{k} {v} 不存在")
            paths[k] = v
    s = run_map(dirs, a.out, paths)
    print(json.dumps(dict(worth_validate=s["worth_validate"], reason=s["validate_reason"],
                          Q_stop0=s["verdicts"]["Q"]["stop0"]["holds"],
                          stop2={md: s["verdicts"][md]["stop2"].get("code") for md in ("L", "Q", "X")}),
                     ensure_ascii=False, indent=1))
    print("寫好", a.out)


# ====================================================================== GPU：micro（函式裡才 import torch）
def check_config(model):
    """規則 6：SHAPES 和模型 config.json 交叉驗證；回傳 rope_theta。"""
    from m10_speed import model_glob
    ps = sorted(glob.glob(model_glob(model)))
    if not ps:
        raise FileNotFoundError(model_glob(model))
    cfg = json.load(open(os.path.join(ps[-1], "config.json")))
    L, H, D = SHAPES[model]
    hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    got = (cfg["num_hidden_layers"], cfg["num_key_value_heads"], hd)
    if got != (L, H, D):
        raise ValueError(f"{model} 的 config.json {got} ≠ SHAPES {(L, H, D)}")
    return float(cfg.get("rope_theta") or (cfg.get("rope_parameters") or {}).get("rope_theta") or 10000.0), ps[-1]


class GSeg:
    def __init__(self, name, cat):
        self.name, self.family, self.kind, self.feed = name, cat["family"], cat["kind"], cat["feed"]
        self.ops, self.h2d_one, self.dq_one, self.dq_all, self.extra_ops = {}, {}, None, None, {}
        self.bytes_blk = self.bytes_req = self.bytes_resident = 0
        self.n_copies, self.impl, self.fmt, self.kernel, self.check = {}, {}, "", "", "not_run"


class GCtx:
    """一個模型一份：KV 緩衝、載入區位置、RoPE 表、亂數。"""

    def __init__(self, torch, model, kv, n, theta, store_impl, seed=0):
        self.torch, self.model, self.kv, self.n = torch, model, kv, n
        self.L, self.H, self.D = SHAPES[model]
        self.dev = kv.device
        self.lo = (N_CTX - n) * CHUNK
        self.T = n * CHUNK
        self.hi = self.lo + self.T
        self.store_impl = store_impl
        self.g = torch.Generator(device=self.dev).manual_seed(seed)
        D = self.D
        inv = 1.0 / (theta ** (torch.arange(0, D, 2, device=self.dev, dtype=torch.float32) / D))
        pos = torch.arange(self.lo, self.hi, device=self.dev, dtype=torch.float32)
        fr = torch.outer(pos, inv)
        emb = torch.cat((fr, fr), dim=-1)
        self.cos, self.sin = emb.cos().contiguous(), emb.sin().contiguous()

    def randn(self, *shape, scale=1.0):
        t = self.torch
        return (t.randn(shape, generator=self.g, device=self.dev) * scale).to(t.bfloat16)

    def region(self):
        return self.kv[:, :, :, self.lo:self.hi]

    def bview(self, j):
        return self.kv[:, :, :, self.lo + j * CHUNK:self.lo + (j + 1) * CHUNK]

    def pin(self, x):
        return x.cpu().pin_memory()

    def sync(self):
        self.torch.cuda.synchronize()


def _relerr(torch, a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def build_segments(ctx, names, kern_of, fp8_fmt):
    """建每個 segment 的資料（GPU 暫存區＋pinned 主機副本）與 op；建好立刻驗正確性（不過就丟例外，規則 2）。"""
    torch = ctx.torch
    import m10_codec as C
    import m10_codec_x as X
    import m10_dequant_fast as DF
    import m10_dequant_fast_x as FX
    import m16_xtime_kern as KK
    L, H, D, n, T, lo = ctx.L, ctx.H, ctx.D, ctx.n, ctx.T, ctx.lo
    cat = seg_catalog()
    segs = {}
    xreg = ctx.randn(L, 2, H, T, D, scale=2.0)
    xreg[:, 0, :, :, 5] += 40                             # K 的離群 channel（同 m10_dqsweep 的 kernel 微測）

    def xb(j):
        return xreg[:, :, :, j * CHUNK:(j + 1) * CHUNK].contiguous()

    def finish(sg):
        for h, fn1 in sg.h2d_one.items():
            sg.ops[h] = (lambda fn1=fn1: [fn1(j) for j in range(n)])
        if sg.dq_one is not None:
            sg.ops["dq_perblock"] = lambda: [sg.dq_one(j) for j in range(n)]
        if sg.dq_all is not None:
            sg.ops["dq_oneshot"] = sg.dq_all
        sg.ops.update(sg.extra_ops)                       # 不是逐 block 的 op（10/21：xKV 的 h2d_req＝只搬基底）
        assert set(sg.ops) == set(cat[sg.name]["ops"]), (sg.name, sorted(sg.ops), cat[sg.name]["ops"])

    def region_parts(host_parts, dev):
        """每個 block 的 part 沿第 2 維（token 軸）接起來的 GPU 暫存區。"""
        out = {}
        for k, t in host_parts[0].items():
            sh = list(t.shape)
            sh[2] *= n
            out[k] = torch.empty(sh, dtype=t.dtype, device=dev)
        return out

    def h2d_region_parts(dst, host_parts):
        def one(j):
            for k, t in host_parts[j].items():
                tb = t.shape[2]
                dst[k].narrow(2, j * tb, tb).copy_(t, non_blocking=True)
        return one

    def lowrank_K(sg, r, tag):
        hK = [ctx.pin(ctx.randn(L, H, CHUNK, r)) for _ in range(n)]
        bK = ctx.randn(L, H, r, D, scale=1.0 / math.sqrt(r))
        mK = ctx.randn(L, H, D, scale=0.1)
        sK = torch.empty((L, H, T, r), dtype=torch.bfloat16, device=ctx.dev)
        sg.bytes_resident += bK.numel() * 2 + mK.numel() * 2
        return hK, bK, mK, sK

    for name in names:
        info = cat[name]
        sg = GSeg(name, info)
        if name == "bf16":
            host = [ctx.pin(xb(j)) for j in range(n)]
            sg.h2d_one["h2d_region"] = lambda j, host=host: ctx.bview(j).copy_(host[j], non_blocking=True)
            sg.bytes_blk, sg.n_copies["h2d_region"], sg.fmt = host[0].nbytes, n, "bf16"
            sg.impl["h2d_region"] = "逐 block copy_ 進 KV 視圖（Restorer._load）"
            finish(sg)
            ctx.region().zero_()
            sg.ops["h2d_region"]()
            ctx.sync()
            sg.check = "ok" if torch.equal(ctx.region(), xreg) else "FAIL：bf16 H2D 後不相等"
        elif name in MAIN_Q:
            fmt = fp8_fmt if name == "fp8" else name
            kern = kern_of(fmt)
            encs = [C.encode(xb(j), fmt) for j in range(n)]
            host = [e.map(lambda t: t.cpu().pin_memory()) for e in encs]
            e0 = C.Enc(fmt, encs[0].shape, {})
            stack = DF.group_parts(encs[0], n, ctx.dev)
            bids = torch.arange(N_CTX - n, N_CTX, dtype=torch.int64, device=ctx.dev)
            cfg = DF.kernel_cfg(kern)
            sg.h2d_one["h2d_stack"] = lambda j, stack=stack, host=host, e0=e0: \
                DF.sub_enc(e0, stack, j).copy_from(host[j], non_blocking=True)
            sg.dq_all = lambda fmt=fmt, stack=stack, bids=bids, cfg=cfg, sh=encs[0].shape: \
                DF.decode_multi(fmt, sh, stack, n, ctx.kv, bids, **cfg)
            sg.dq_one = lambda j, stack=stack, e0=e0, kern=kern: DF.decode_variant(DF.sub_enc(e0, stack, j), ctx.bview(j), kern)
            sg.bytes_blk, sg.fmt, sg.kernel = host[0].nbytes, fmt, kern
            sg.n_copies["h2d_stack"] = n * len(host[0].parts)
            sg.impl.update(h2d_stack="逐 block 搬進一次拆的暫存區（Restorer._load_defer）",
                           dq_oneshot="m10_dequant_fast.decode_multi（第二階段 P3／S4）",
                           dq_perblock="m10_dequant_fast.decode_variant（第二階段 P0）")
            finish(sg)
            sg.ops["h2d_stack"]()
            ctx.sync()
            msgs = []
            for op in ("dq_oneshot", "dq_perblock"):
                ctx.region().zero_()
                sg.ops[op]()
                ctx.sync()
                bad = [j for j in range(n) if not torch.equal(ctx.bview(j), C.decode(encs[j]))]
                if bad:
                    msgs.append(f"{op} 和 m10_codec.decode 不同（block {bad[:4]}）")
            sg.check = "ok" if not msgs else "FAIL：" + "；".join(msgs)
            del encs
        elif name in NEW_Q:
            fx = NEW_Q[name]
            encs = [X.encode(xb(j), fx) for j in range(n)]
            host = [e.map(lambda t: t.cpu().pin_memory()) for e in encs]
            e0 = C.Enc(fx, encs[0].shape, {})
            stack = DF.group_parts(encs[0], n, ctx.dev)
            for k, t in host[0].parts.items():
                if t.dim() < 3 or t.shape[2] not in (CHUNK, CHUNK // 32):
                    raise ValueError(f"{fx} 的 part {k} {tuple(t.shape)} 第 2 維不是 token 軸")
            reg = region_parts([h.parts for h in host], ctx.dev)
            ereg = C.Enc(fx, (L, 2, H, T, D), reg)
            sg.h2d_one["h2d_stack"] = lambda j, stack=stack, host=host, e0=e0: \
                DF.sub_enc(e0, stack, j).copy_from(host[j], non_blocking=True)
            sg.h2d_one["h2d_region"] = h2d_region_parts(reg, [h.parts for h in host])
            sg.dq_all = lambda ereg=ereg: FX.decode_x_fast(ereg, out=ctx.region())
            sg.dq_one = lambda j, stack=stack, fx=fx: FX.decode_x_fast(
                C.Enc(fx, (L, 2, H, CHUNK, D), {k: t[j] for k, t in stack.items()}), out=ctx.bview(j))
            sg.bytes_blk, sg.fmt, sg.kernel = host[0].nbytes, fx, "m10_dequant_fast_x"
            sg.n_copies.update(h2d_stack=n * len(host[0].parts), h2d_region=n * len(host[0].parts))
            sg.impl.update(h2d_stack="逐 block 搬進每 block 連續的暫存區", h2d_region="逐 block 搬進沿 token 接起來的暫存區",
                           dq_oneshot="decode_x_fast 一次解整段（part 沿 token 接起來）", dq_perblock="decode_x_fast 逐 block")
            finish(sg)
            sg.ops["h2d_stack"]()
            sg.ops["h2d_region"]()
            ctx.sync()
            msgs = []
            for op in ("dq_oneshot", "dq_perblock"):
                ctx.region().zero_()
                sg.ops[op]()
                ctx.sync()
                bad = [j for j in range(n) if not torch.equal(ctx.bview(j), X.decode(encs[j]))]
                if bad:
                    msgs.append(f"{op} 和 m10_codec_x.decode 不同（block {bad[:4]}）")
            sg.check = "ok" if not msgs else "FAIL：" + "；".join(msgs)
            del encs
        elif name.startswith("lr_r") or name == "lrk24_v4":
            r = 24 if name == "lrk24_v4" else int(name[4:])
            hK, bK, mK, sK = lowrank_K(sg, r, name)
            kvK = ctx.kv[:, 0, :, lo:ctx.hi]
            if name == "lrk24_v4":
                encs = [X.encode(xb(j), "kv-bf16-int4_kivi") for j in range(n)]
                hv = [{k[2:]: encs[j].parts[k].cpu().pin_memory() for k in encs[j].parts if k.startswith("v.")} for j in range(n)]
                rv = region_parts(hv, ctx.dev)
                h1v = h2d_region_parts(rv, hv)

                def h2d(j, hK=hK, sK=sK, h1v=h1v):
                    sK[:, :, j * CHUNK:(j + 1) * CHUNK].copy_(hK[j], non_blocking=True)
                    h1v(j)

                def dq_all(sK=sK, bK=bK, mK=mK, rv=rv, kvK=kvK):
                    KK.store(torch.matmul(sK, bK), mK, ctx.cos, ctx.sin, kvK, True, ctx.store_impl)
                    FX._half(rv, "kivi4", "v", ctx.kv[:, 1, :, lo:ctx.hi], 16)
                sg.h2d_one["h2d_region"], sg.dq_all = h2d, dq_all
                sg.bytes_blk = hK[0].nbytes + sum(t.nbytes for t in hv[0].values())
                sg.n_copies["h2d_region"] = n * (1 + len(hv[0]))
                sg.fmt = "K：PCA r24（BF16 係數）＋V：int4_kivi"
            else:
                hV, bV, mV, sV = lowrank_K(sg, r, name)
                kvV = ctx.kv[:, 1, :, lo:ctx.hi]

                def h2d(j, hK=hK, sK=sK, hV=hV, sV=sV):
                    sK[:, :, j * CHUNK:(j + 1) * CHUNK].copy_(hK[j], non_blocking=True)
                    sV[:, :, j * CHUNK:(j + 1) * CHUNK].copy_(hV[j], non_blocking=True)

                def dq_all(sK=sK, bK=bK, mK=mK, sV=sV, bV=bV, mV=mV, kvK=kvK, kvV=kvV):
                    KK.store(torch.matmul(sK, bK), mK, ctx.cos, ctx.sin, kvK, True, ctx.store_impl)
                    KK.store(torch.matmul(sV, bV), mV, None, None, kvV, False, ctx.store_impl)

                def dq_one(j, sK=sK, bK=bK, mK=mK, sV=sV, bV=bV, mV=mV):
                    sl = slice(j * CHUNK, (j + 1) * CHUNK)
                    a0, a1 = lo + j * CHUNK, lo + (j + 1) * CHUNK
                    KK.store(torch.matmul(sK[:, :, sl], bK), mK, ctx.cos[sl], ctx.sin[sl], ctx.kv[:, 0, :, a0:a1], True,
                             ctx.store_impl)
                    KK.store(torch.matmul(sV[:, :, sl], bV), mV, None, None, ctx.kv[:, 1, :, a0:a1], False, ctx.store_impl)
                sg.h2d_one["h2d_region"], sg.dq_all, sg.dq_one = h2d, dq_all, dq_one
                sg.bytes_blk = hK[0].nbytes + hV[0].nbytes
                sg.n_copies["h2d_region"] = 2 * n
                sg.fmt = f"PCA r{r}（K、V 的 BF16 係數；基底常駐）"
            sg.impl.update(h2d_region="逐 block 搬係數進沿 token 接起來的暫存區",
                           dq_oneshot=f"torch.matmul＋m16_xtime_kern.store（{ctx.store_impl}）一次整段",
                           dq_perblock=f"每 block 一次 torch.matmul＋store（{ctx.store_impl}）")
            finish(sg)
            sg.ops["h2d_region"]()
            ctx.sync()
            ctx.region().zero_()
            sg.ops["dq_oneshot"]()
            ctx.sync()
            refK = KK.store_torch(torch.matmul(sK.float(), bK.float()), mK, ctx.cos, ctx.sin,
                                  torch.empty((L, H, T, D), dtype=torch.bfloat16, device=ctx.dev), True)
            msgs = []
            e = _relerr(torch, kvK, refK)
            if e > 1e-2:
                msgs.append(f"K 和 FP32 參考差 {e:.3g}")
            if name == "lrk24_v4":
                want = torch.cat([X.decode(encs[j])[:, 1] for j in range(n)], dim=2)
                if not torch.equal(ctx.kv[:, 1, :, lo:ctx.hi], want):
                    msgs.append("V（int4_kivi 半邊）和 m10_codec_x.decode 不同")
                del encs
            else:
                refV = KK.store_torch(torch.matmul(sV.float(), bV.float()), mV, None, None,
                                      torch.empty((L, H, T, D), dtype=torch.bfloat16, device=ctx.dev), False)
                e = _relerr(torch, kvV, refV)
                if e > 1e-2:
                    msgs.append(f"V 和 FP32 參考差 {e:.3g}")
                one = ctx.region().clone()
                ctx.region().zero_()
                sg.ops["dq_perblock"]()
                ctx.sync()
                e = _relerr(torch, ctx.region(), one)
                if e > 1e-2:
                    msgs.append(f"逐 block 和一次整段差 {e:.3g}")
                del one, refV
            del refK
            sg.check = "ok" if not msgs else "FAIL：" + "；".join(msgs)
        elif name in XKV2:
            # 10/21 補缺口：照 m16_xstack 的 xl 階段。窗口 w＝第 w·W..(w+1)·W−1 層（相鄰層；y5[:, :, g] 寫到第 g＋w·W 層＝kv[g::W] 的第 w 個）。
            # 係數 [Lg, 1, T, r]（r＝秩×W×H），K、V 秩不同；BF16 存，或 m10_codec_x.enc_half(…, cf, "k") 量化（kivi4：每主成分、沿 token
            # 每 32 個一組，不跨 512-token block → 逐 block 編碼＝整段編碼，下面驗）。基底 [r, W·H·D] BF16、每個 prompt 一份。
            rk_, rv_, cf = XKV2[name]
            W = XKV_W
            if L % W:
                raise ValueError(f"{ctx.model} 的層數 {L} 不能被 {W} 整除")
            Lg, GHD = L // W, W * H * D
            rs = (rk_ * W * H, rv_ * W * H)
            coefs = {kv_: [torch.randn((Lg, 1, CHUNK, rs[kv_]), generator=ctx.g, device=ctx.dev) for _ in range(n)]
                     for kv_ in (0, 1)}

            def enc(c32, cf=cf):
                return {"x": c32.to(torch.bfloat16)} if cf == "bf16" else X.enc_half(c32.contiguous(), cf, "k")
            hp = {kv_: [{k: t.cpu().pin_memory() for k, t in enc(c).items()} for c in coefs[kv_]] for kv_ in (0, 1)}
            stg = {kv_: region_parts(hp[kv_], ctx.dev) for kv_ in (0, 1)}
            h1 = {kv_: h2d_region_parts(stg[kv_], hp[kv_]) for kv_ in (0, 1)}
            hb = {kv_: ctx.pin(ctx.randn(Lg, rs[kv_], GHD, scale=1.0 / math.sqrt(rs[kv_]))) for kv_ in (0, 1)}
            sb = {kv_: torch.empty((Lg, rs[kv_], GHD), dtype=torch.bfloat16, device=ctx.dev) for kv_ in (0, 1)}
            cbuf = {kv_: (stg[kv_]["x"][:, 0] if cf == "bf16" else
                          torch.empty((Lg, T, rs[kv_]), dtype=torch.bfloat16, device=ctx.dev)) for kv_ in (0, 1)}
            m0 = torch.zeros((L, H, D), dtype=torch.bfloat16, device=ctx.dev)

            def h2d(j, h1=h1, hb=hb, sb=sb):
                for kv_ in (0, 1):
                    h1[kv_](j)
                    if j == 0:                                   # 基底每個 prompt 一份：每次還原搬一次
                        sb[kv_].copy_(hb[kv_], non_blocking=True)

            def h2d_req(hb=hb, sb=sb):
                for kv_ in (0, 1):
                    sb[kv_].copy_(hb[kv_], non_blocking=True)

            def coef(kv_, j=None, stg=stg, cbuf=cbuf, cf=cf):
                """BF16 係數 [Lg, T 或 512, r]（量化的先反量化進 cbuf；j＝None 是整段）。"""
                if cf == "bf16":
                    return cbuf[kv_] if j is None else cbuf[kv_][:, j * CHUNK:(j + 1) * CHUNK]
                p = stg[kv_] if j is None else {k: v.narrow(2, j * (v.shape[2] // n), v.shape[2] // n) for k, v in stg[kv_].items()}
                out = cbuf[kv_] if j is None else cbuf[kv_][:, j * CHUNK:(j + 1) * CHUNK]
                return KK.coef_dequant(p, cf, out)

            def recon(kv_, cb, j=None, sb=sb, m0=m0, W=W, Lg=Lg):
                Ts = cb.shape[1]
                a0 = lo if j is None else lo + j * CHUNK
                cos = ctx.cos if j is None else ctx.cos[j * CHUNK:(j + 1) * CHUNK]
                sin = ctx.sin if j is None else ctx.sin[j * CHUNK:(j + 1) * CHUNK]
                y5 = torch.matmul(cb, sb[kv_]).view(Lg, Ts, W, H, D)
                for g in range(W):
                    KK.store(y5[:, :, g].permute(0, 2, 1, 3), m0[g::W], cos, sin, ctx.kv[g::W, kv_, :, a0:a0 + Ts], kv_ == 0,
                             ctx.store_impl)

            def dq_all(coef=coef, recon=recon):                 # 預設參數綁住這一輪的 coef／recon（迴圈下一個 xKV 段會重新綁定名字）
                for kv_ in (0, 1):
                    recon(kv_, coef(kv_))

            def dq_one(j, coef=coef, recon=recon):
                for kv_ in (0, 1):
                    recon(kv_, coef(kv_, j), j)
            sg.h2d_one["h2d_region"], sg.dq_all, sg.dq_one = h2d, dq_all, dq_one
            sg.extra_ops["h2d_req"] = h2d_req
            sg.bytes_blk = sum(t.nbytes for kv_ in (0, 1) for t in hp[kv_][0].values())
            sg.bytes_req = sum(hb[kv_].nbytes for kv_ in (0, 1))
            want = xkv_bytes(ctx.model, rk_, rv_, cf, CHUNK)              # 規則 6：實際 tensor 大小 vs m16_xstack 的位元組帳
            if (sg.bytes_blk, sg.bytes_req) != want:
                raise ValueError(f"{name}：實際位元組 {(sg.bytes_blk, sg.bytes_req)} ≠ 位元組帳 {want}")
            sg.n_copies.update(h2d_region=n * (len(hp[0][0]) + len(hp[1][0])) + 2, h2d_req=2)
            sg.kernel = "m16_xtime_kern.coef_dequant" if cf != "bf16" else ""
            sg.fmt = (f"xKV（{XKV2_SPEC[name]}）：每 {W} 層聯合秩 K {rs[0]}、V {rs[1]}；係數 {cf}；"
                      "基底 BF16 每個 prompt 一份（算進搬的位元組）")
            sg.impl.update(h2d_region="逐 block 搬係數（含 scale／zero）＋每次還原搬一次基底", h2d_req="只搬基底（K、V 各一份）",
                           dq_oneshot=("" if cf == "bf16" else "coef_dequant（Triton）→ ") +
                           f"BF16 torch.matmul＋每層 store（{ctx.store_impl}）一次整段",
                           dq_perblock=("" if cf == "bf16" else "coef_dequant → ") + f"每 block 一次 matmul＋store（{ctx.store_impl}）")
            finish(sg)
            sg.ops["h2d_region"]()
            ctx.sync()
            msgs = []
            if cf != "bf16":
                for kv_ in (0, 1):
                    full = X.enc_half(torch.cat(coefs[kv_], dim=2).contiguous(), cf, "k")
                    bad = [k for k in full if not torch.equal(full[k], stg[kv_][k])]
                    if bad:
                        msgs.append(f"{'KV'[kv_]}：逐 block 編碼接起來 ≠ 整段編碼（{bad}）")
                    ref = X.dec_half(stg[kv_], cf, "k", (Lg, 1, T, rs[kv_])).to(torch.bfloat16)[:, 0]
                    if not torch.equal(coef(kv_), ref):
                        msgs.append(f"{'KV'[kv_]}：coef_dequant 和 m10_codec_x.dec_half 不同")
                    del full, ref
            del coefs
            ctx.region().zero_()
            sg.ops["dq_oneshot"]()
            ctx.sync()
            for kv_, rope in ((0, True), (1, False)):
                c32 = stg[kv_]["x"][:, 0].float() if cf == "bf16" else X.dec_half(stg[kv_], cf, "k", (Lg, 1, T, rs[kv_]))[:, 0]
                y5 = torch.matmul(c32, sb[kv_].float()).view(Lg, T, W, H, D)
                for g in (0, W - 1):
                    ref = KK.store_torch(y5[:, :, g].permute(0, 2, 1, 3), m0[g::W], ctx.cos, ctx.sin,
                                         torch.empty((Lg, H, T, D), dtype=torch.bfloat16, device=ctx.dev), rope)
                    e = _relerr(torch, ctx.kv[g::W, kv_, :, lo:ctx.hi], ref)
                    if e > 1e-2:
                        msgs.append(f"{'KV'[kv_]} 第 {g} 組和 FP32 參考差 {e:.3g}")
                del y5, c32
            one = ctx.region().clone()
            ctx.region().zero_()
            sg.ops["dq_perblock"]()
            ctx.sync()
            e = _relerr(torch, ctx.region(), one)
            if e > 1e-2:
                msgs.append(f"逐 block 和一次整段差 {e:.3g}")
            del one
            sg.check = "ok" if not msgs else "FAIL：" + "；".join(msgs)
        elif name.startswith("xkv_"):
            G, f = [int(x) for x in re.match(r"xkv_g(\d+)_f(\d+)", name).groups()]
            if L % G:
                raise ValueError(f"{ctx.model} 的層數 {L} 不能被 {G} 整除")
            Lg, GHD = L // G, G * H * D
            r = GHD // f
            hc = {kv_: [ctx.pin(ctx.randn(Lg, CHUNK, r)) for _ in range(n)] for kv_ in (0, 1)}
            hb = {kv_: ctx.pin(ctx.randn(Lg, r, GHD, scale=1.0 / math.sqrt(r))) for kv_ in (0, 1)}
            sc = {kv_: torch.empty((Lg, T, r), dtype=torch.bfloat16, device=ctx.dev) for kv_ in (0, 1)}
            sb = {kv_: torch.empty((Lg, r, GHD), dtype=torch.bfloat16, device=ctx.dev) for kv_ in (0, 1)}
            m0 = torch.zeros((L, H, D), dtype=torch.bfloat16, device=ctx.dev)

            def h2d(j, hc=hc, hb=hb, sc=sc, sb=sb):
                for kv_ in (0, 1):
                    sc[kv_][:, j * CHUNK:(j + 1) * CHUNK].copy_(hc[kv_][j], non_blocking=True)
                    if j == 0:                                   # 基底每個 prompt 一份：每次還原搬一次
                        sb[kv_].copy_(hb[kv_], non_blocking=True)

            def dq_all(sc=sc, sb=sb, m0=m0, G=G, Lg=Lg):
                for kv_, rope in ((0, True), (1, False)):
                    y5 = torch.matmul(sc[kv_], sb[kv_]).view(Lg, T, G, H, D)
                    for g in range(G):
                        KK.store(y5[:, :, g].permute(0, 2, 1, 3), m0[g::G], ctx.cos, ctx.sin,
                                 ctx.kv[g::G, kv_, :, lo:ctx.hi], rope, ctx.store_impl)
            sg.h2d_one["h2d_region"], sg.dq_all = h2d, dq_all
            sg.bytes_blk = 2 * hc[0][0].nbytes
            sg.bytes_req = 2 * hb[0].nbytes
            sg.n_copies["h2d_region"] = 2 * n + 2
            sg.fmt = f"xKV：每 {G} 層共用 token 係數 r={r}，基底每個 prompt 一份（算進搬的位元組）"
            sg.impl.update(h2d_region="逐 block 搬係數＋每次還原搬一次基底",
                           dq_oneshot=f"torch.matmul＋每層 store（{ctx.store_impl}）")
            finish(sg)
            sg.ops["h2d_region"]()
            ctx.sync()
            ctx.region().zero_()
            sg.ops["dq_oneshot"]()
            ctx.sync()
            msgs = []
            for kv_, rope in ((0, True), (1, False)):
                y5 = torch.matmul(sc[kv_].float(), sb[kv_].float()).view(Lg, T, G, H, D)
                for g in (0, G - 1):
                    ref = KK.store_torch(y5[:, :, g].permute(0, 2, 1, 3), m0[g::G], ctx.cos, ctx.sin,
                                         torch.empty((Lg, H, T, D), dtype=torch.bfloat16, device=ctx.dev), rope)
                    e = _relerr(torch, ctx.kv[g::G, kv_, :, lo:ctx.hi], ref)
                    if e > 1e-2:
                        msgs.append(f"{'KV'[kv_]} 第 {g} 組和 FP32 參考差 {e:.3g}")
                del y5
            sg.check = "ok" if not msgs else "FAIL：" + "；".join(msgs)
        elif name.startswith("merge_") or name == "merge_g2_int4":
            g = 2 if name == "merge_g2_int4" else int(name[len("merge_g"):])
            Tb = CHUNK // g
            if name == "merge_g2_int4":
                kern = kern_of("int4_kivi")
                encs = [C.encode(ctx.randn(L, 2, H, Tb, D), "int4_kivi") for _ in range(n)]
                hp = [{k: t.cpu().pin_memory() for k, t in e.parts.items()} for e in encs]
                reg = region_parts(hp, ctx.dev)
                ereg = C.Enc("int4_kivi", (L, 2, H, n * Tb, D), reg)
                dec = torch.empty((L, 2, H, n * Tb, D), dtype=torch.bfloat16, device=ctx.dev)
                sg.h2d_one["h2d_region"] = h2d_region_parts(reg, hp)

                def dq_all(ereg=ereg, dec=dec, kern=kern, g=g):
                    DF.decode_variant(ereg, dec, kern)
                    ctx.region().view(L, 2, H, n * Tb, g, D).copy_(dec.unsqueeze(4).expand(L, 2, H, n * Tb, g, D))
                sg.dq_all = dq_all
                sg.bytes_blk, sg.kernel = sum(t.nbytes for t in hp[0].values()), kern
                sg.n_copies["h2d_region"] = n * len(hp[0])
                sg.fmt = "合併 2 個 token＋int4_kivi"
                src = dec
            else:
                host = [ctx.pin(ctx.randn(L, 2, H, Tb, D)) for _ in range(n)]
                st = torch.empty((L, 2, H, n * Tb, D), dtype=torch.bfloat16, device=ctx.dev)
                sg.h2d_one["h2d_region"] = lambda j, host=host, st=st, Tb=Tb: \
                    st[:, :, :, j * Tb:(j + 1) * Tb].copy_(host[j], non_blocking=True)
                sg.dq_all = lambda st=st, g=g, Tb=Tb: ctx.region().view(L, 2, H, n * Tb, g, D).copy_(
                    st.unsqueeze(4).expand(L, 2, H, n * Tb, g, D))
                sg.dq_one = lambda j, st=st, g=g, Tb=Tb: ctx.bview(j).view(L, 2, H, Tb, g, D).copy_(
                    st[:, :, :, j * Tb:(j + 1) * Tb].unsqueeze(4).expand(L, 2, H, Tb, g, D))
                sg.bytes_blk = host[0].nbytes
                sg.n_copies["h2d_region"] = n
                sg.fmt = f"相鄰 {g} 個 token 合併（BF16）"
                src = st
            sg.impl.update(h2d_region="逐 block 搬進沿 token 接起來的暫存區", dq_oneshot="展開 copy（expand）一次整段",
                           dq_perblock="每 block 展開 copy")
            finish(sg)
            sg.ops["h2d_region"]()
            ctx.sync()
            ctx.region().zero_()
            sg.ops["dq_oneshot"]()
            ctx.sync()
            msgs = []
            if name == "merge_g2_int4":
                want = torch.cat([C.decode(e) for e in encs], dim=3).repeat_interleave(g, dim=3)
                del encs
            else:
                want = src.repeat_interleave(g, dim=3)
            if not torch.equal(ctx.region(), want):
                msgs.append("展開後和 repeat_interleave 參考不同")
            if sg.dq_one is not None:
                ctx.region().zero_()
                sg.ops["dq_perblock"]()
                ctx.sync()
                if not torch.equal(ctx.region(), want):
                    msgs.append("逐 block 展開和參考不同")
            del want
            sg.check = "ok" if not msgs else "FAIL：" + "；".join(msgs)
        elif name.startswith("droptok_"):
            keep = int(re.match(r"droptok_k(\d+)", name).group(1))
            with_int4 = name.endswith("_int4")
            Tkb = CHUNK * keep // 100
            idx = [(torch.rand((L, H, CHUNK), generator=ctx.g, device=ctx.dev).argsort(dim=-1)[..., :Tkb]
                    .sort(dim=-1).values + j * CHUNK).to(torch.int16) for j in range(n)]
            hi_ = [ctx.pin(x) for x in idx]
            si = torch.empty((L, H, n * Tkb), dtype=torch.int16, device=ctx.dev)
            ones = torch.ones((L, H, n * Tkb), dtype=torch.bool, device=ctx.dev)
            kmask = torch.zeros((L, H, N_CTX * CHUNK + CHUNK), dtype=torch.bool, device=ctx.dev)
            if with_int4:
                kern = kern_of("int4_kivi")
                encs = [C.encode(ctx.randn(L, 2, H, Tkb, D), "int4_kivi") for _ in range(n)]
                hp = [{k: t.cpu().pin_memory() for k, t in e.parts.items()} for e in encs]
                reg = region_parts(hp, ctx.dev)
                ereg = C.Enc("int4_kivi", (L, 2, H, n * Tkb, D), reg)
                src = torch.empty((L, 2, H, n * Tkb, D), dtype=torch.bfloat16, device=ctx.dev)
                h1 = h2d_region_parts(reg, hp)
                sg.bytes_blk = sum(t.nbytes for t in hp[0].values()) + hi_[0].nbytes
                sg.n_copies["h2d_region"] = n * (len(hp[0]) + 1)
                sg.kernel = kern
                sg.fmt = f"每（層, head）留 {keep}% token＋int4_kivi＋int16 位置"
            else:
                hv = [ctx.pin(ctx.randn(L, 2, H, Tkb, D)) for _ in range(n)]
                src = torch.empty((L, 2, H, n * Tkb, D), dtype=torch.bfloat16, device=ctx.dev)
                h1 = (lambda j, hv=hv, src=src, Tkb=Tkb: src[:, :, :, j * Tkb:(j + 1) * Tkb].copy_(hv[j], non_blocking=True))
                sg.bytes_blk = hv[0].nbytes + hi_[0].nbytes
                sg.n_copies["h2d_region"] = 2 * n
                sg.fmt = f"每（層, head）留 {keep}% token（BF16）＋int16 位置"

            def h2d(j, h1=h1, hi_=hi_, si=si, Tkb=Tkb):
                h1(j)
                si[:, :, j * Tkb:(j + 1) * Tkb].copy_(hi_[j], non_blocking=True)

            def dq_all(si=si, src=src, ones=ones, kmask=kmask, Tkb=Tkb, ereg=(ereg if with_int4 else None),
                       kern=(kern if with_int4 else None)):
                if ereg is not None:
                    DF.decode_variant(ereg, src, kern)
                ix = si.long()
                ctx.region().scatter_(3, ix[:, None, :, :, None].expand(L, 2, H, n * Tkb, D), src)
                km = kmask[:, :, lo:ctx.hi]                      # 位置是載入區內的相對位置，遮罩也取載入區的視圖
                km.zero_()
                km.scatter_(2, ix, ones)
            sg.h2d_one["h2d_region"], sg.dq_all = h2d, dq_all
            sg.impl.update(h2d_region="逐 block 搬留下的值與位置", dq_oneshot="（int4 解碼）＋scatter_ 進 KV 視圖＋建 key_keep 遮罩")
            finish(sg)
            sg.ops["h2d_region"]()
            ctx.sync()
            ctx.region().zero_()
            sg.ops["dq_oneshot"]()
            ctx.sync()
            msgs = []
            ix = si.long()
            got = ctx.region().gather(3, ix[:, None, :, :, None].expand(L, 2, H, n * Tkb, D))
            want = torch.cat([C.decode(e) for e in encs], dim=3) if with_int4 else src
            if not torch.equal(got, want):
                msgs.append("scatter 後取回的值和來源不同")
            cnt = kmask[:, :, lo:ctx.hi].sum(dim=-1)
            if not bool((cnt == n * Tkb).all()):
                msgs.append("key_keep 遮罩的個數不對")
            if with_int4:
                del encs
            del got, want
            sg.check = "ok" if not msgs else "FAIL：" + "；".join(msgs)
        else:
            raise ValueError(name)
        segs[name] = sg
        print(f"[build] {ctx.model} {name}: {sg.check}；每 block {sg.bytes_blk} B（BF16 {L * 2 * H * CHUNK * D * 2} B）", flush=True)
        if not sg.check.startswith("ok"):
            raise SystemExit(f"{ctx.model} {name} 正確性沒過：{sg.check}（規則 2：停下來）")
    del xreg
    ctx.region().zero_()
    ctx.sync()
    return segs


def _time_ops(torch, ops, reps, min_rep_s, rng):
    """ops：[(seg, op, fn)]。暖機、定 inner，再 reps 次（每次順序打散）。回傳 {(seg, op): (每次重複的每 op 秒數 list, inner)}。"""
    sync = torch.cuda.synchronize
    inner = {}
    for s, o, fn in ops:
        fn()
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        one = max(time.perf_counter() - t0, 1e-6)
        inner[(s, o)] = max(1, min(2000, math.ceil(min_rep_s / one)))
    res = {k: [] for k in inner}
    for _ in range(reps):
        order = list(ops)
        rng.shuffle(order)
        for s, o, fn in order:
            k = inner[(s, o)]
            sync()
            t0 = time.perf_counter()
            for _ in range(k):
                fn()
            sync()
            res[(s, o)].append((time.perf_counter() - t0) / k)
    return {k: (v, inner[k]) for k, v in res.items()}


def _time_slow(torch, ops, reps, rng, warm=True):
    """慢的 op（一次幾百毫秒到幾十秒，例如 svd_lowrank）：warm＝True 時照清單順序暖機一次（svd 在量化前；量化吃 svd 的輸出；
    第一次呼叫有 rocSOLVER／MAGMA 的初始化），再 reps 次、每次重複把順序打散，每次跑一遍（inner＝1）。
    同一個 process 第二個 duty 起不再暖機（函式庫已初始化、狀態已有，省 1/(reps+1) 的時間）。回傳 {(key, op): 每次的秒數 list}。"""
    sync = torch.cuda.synchronize
    for _, _, fn in (ops if warm else []):
        fn()
        sync()
    res = {(k, o): [] for k, o, _ in ops}
    for _ in range(reps):
        order = list(ops)
        rng.shuffle(order)
        for k, o, fn in order:
            sync()
            t0 = time.perf_counter()
            fn()
            sync()
            res[(k, o)].append(time.perf_counter() - t0)
    return res


def xstack_consts_check():
    """GPU 路徑：svd_lowrank 的參數和 m16_xstack 核對（不一樣 → 丟例外）。m16_xstack 由別的 agent 維護，import 失敗照實記 NOT_CHECKED。"""
    mine = dict(XL_NITER=XL_NITER, XL_OVERSAMPLE=XL_OVERSAMPLE, XL_SEED=XL_SEED, W=XKV_W)
    try:
        import m16_xstack as XS
        got = dict(XL_NITER=XS.XL_NITER, XL_OVERSAMPLE=XS.XL_OVERSAMPLE, XL_SEED=XS.XL_SEED,
                   W=XS.parse_spec("xl:k8:v8")[0]["w"])
    except Exception as ex:                                   # noqa: BLE001（照實記，不當成「一致」）
        return dict(mine, status=f"NOT_CHECKED（import m16_xstack 失敗：{ex!r}）")
    if got != mine:
        raise ValueError(f"svd_lowrank 參數和 m16_xstack 不同：本檔 {mine}，m16_xstack {got}")
    return dict(mine, status="ok（和 m16_xstack 相同）")


def build_wcost(ctx, pairs, fmts):
    """10/21：xKV 寫入（壓縮）路徑的 op。照 m16_xstack.t_xl_half 的編碼那一半（解碼那一半不算寫入成本）：每個窗口 w（第 w·W..(w+1)·W−1 層），
    K 先逆 RoPE（和 m16_xstack.t_rope_inv 同一個式子，用 ctx 的 FP32 cos/sin）→ [T, H·D] → 沿最後一維接 W 層成 Z [T, W·H·D]（FP32）→
    torch.manual_seed(XL_SEED＋w) → torch.svd_lowrank(Z, q＝min(r＋XL_OVERSAMPLE, min(Z.shape)), niter＝XL_NITER) → A＝U[:, :r]·S[:r]、
    B＝V[:, :r]ᵀ 轉 BF16；A 疊成 [Lg, 1, T, r]。量化＝bf16 轉型，或 m10_codec_x.enc_half(A, 格式, "k")（＝m16_xstack.t_halves_enc）。
    pairs＝[(rk, rv)]；fmts＝{(rk, rv): {係數格式}}。來源 KV 是亂數（BF16，K 有離群 channel，同 micro 的 xreg）。
    回傳 (ops＝[(key, op, fn)], info＝{(key, op): 欄位}, state)。"""
    torch = ctx.torch
    import m10_codec_x as X
    L, H, D, T = ctx.L, ctx.H, ctx.D, ctx.T
    W = XKV_W
    if L % W:
        raise ValueError(f"{ctx.model} 的層數 {L} 不能被 {W} 整除")
    Lg = L // W
    src = ctx.randn(L, 2, H, T, D, scale=2.0)
    src[:, 0, :, :, 5] += 40
    h = D // 2
    c, s = ctx.cos[:, :h], ctx.sin[:, :h]                     # [T, D/2]（前後半相同，同 m16_xstack 取 cos[..., :h]）
    state, ops, info = {}, [], {}

    def rope_inv(y):                                          # = m16_xstack.t_rope_inv（y：[H, T, D] BF16 → FP32）
        det = c * c + s * s
        yf = y.float()
        y1, y2 = yf[..., :h], yf[..., h:]
        return torch.cat(((y1 * c + y2 * s) / det, (y2 * c - y1 * s) / det), dim=-1)

    def enc_half_of(hi, rh):
        r = rh * W * H
        As, Bs = [], []
        for w in range(Lg):
            Z = torch.cat([(rope_inv(src[l, 0]) if hi == 0 else src[l, 1].float()).permute(1, 0, 2).reshape(T, H * D)
                           for l in range(w * W, (w + 1) * W)], dim=1)
            torch.manual_seed(XL_SEED + w)
            U, S, V = torch.svd_lowrank(Z, q=min(r + XL_OVERSAMPLE, min(Z.shape)), niter=XL_NITER)
            As.append(U[:, :r] * S[:r])
            Bs.append(V[:, :r].T.contiguous().to(torch.bfloat16))
            del Z, U, S, V
        state[(hi, rh)] = (torch.stack(As).unsqueeze(1), Bs)

    def quant(rk, rv, f):
        for hi, rh in ((0, rk), (1, rv)):
            A4 = state[(hi, rh)][0]
            state[("q", hi, rh, f)] = A4.to(torch.bfloat16) if f == "bf16" else X.enc_half(A4.contiguous(), f, "k")

    for rk, rv in pairs:
        key = f"k{rk}v{rv}"
        for hi, rh, op in ((0, rk, "svd_k"), (1, rv, "svd_v")):
            r = rh * W * H
            ops.append((key, op, lambda hi=hi, rh=rh: enc_half_of(hi, rh)))
            info[(key, op)] = dict(pair=key, rk=rk, rv=rv, half="kv"[hi], rank_joint=r, coef_fmt="", W=W, n_windows=Lg, T=T,
                                   svd_q=min(r + XL_OVERSAMPLE, T, W * H * D), niter=XL_NITER, oversample=XL_OVERSAMPLE)
        for f in sorted(fmts[(rk, rv)]):
            ops.append((key, f"q_{f}", lambda rk=rk, rv=rv, f=f: quant(rk, rv, f)))
            info[(key, f"q_{f}")] = dict(pair=key, rk=rk, rv=rv, half="kv", rank_joint=f"{rk * W * H}/{rv * W * H}", coef_fmt=f,
                                         W=W, n_windows=Lg, T=T, svd_q="", niter="", oversample="")
    return ops, info, state


def wcost_check(ctx, state, pairs, fmts):
    """寫入路徑的輸出核對（規則 6）：A 的形狀、B 的形狀、量化後的位元組＝位元組帳（xkv_coef_bytes）、值有限。回傳訊息 list（空＝過）。"""
    torch = ctx.torch
    W, H, D, T = XKV_W, ctx.H, ctx.D, ctx.T
    Lg = ctx.L // W
    msgs = []
    for rk, rv in pairs:
        for hi, rh in ((0, rk), (1, rv)):
            r = rh * W * H
            A4, Bs = state[(hi, rh)]
            if tuple(A4.shape) != (Lg, 1, T, r) or len(Bs) != Lg or tuple(Bs[0].shape) != (r, W * H * D):
                msgs.append(f"k{rk}v{rv} {'KV'[hi]}：形狀 A {tuple(A4.shape)} B {len(Bs)}×{tuple(Bs[0].shape)}")
            if not bool(torch.isfinite(A4).all()):
                msgs.append(f"k{rk}v{rv} {'KV'[hi]}：A 有非有限值")
            for f in fmts[(rk, rv)]:
                q = state[("q", hi, rh, f)]
                nb = q.nbytes if f == "bf16" else sum(t.nbytes for t in q.values())
                if nb != xkv_coef_bytes(Lg, T, r, f):
                    msgs.append(f"k{rk}v{rv} {'KV'[hi]} {f}：位元組 {nb} ≠ 位元組帳 {xkv_coef_bytes(Lg, T, r, f)}")
    return msgs


def gpu_micro(model, out, n_blocks=32, gpu_states=GPU_STATES, busy_levels=BUSY_LEVELS, reps=5, min_rep_s=0.1,
              segments=None, bg_list=None, bg_seconds=2.0, recompute=True, rec_pos=REC_POS, rec_reps=3,
              store_impl="triton", smoke=False, seed=0, wcost=None, wcost_states=WCOST_STATES, wcost_reps=3):
    """wcost：None 或 [(rk, rv)]（10/21：xKV 寫入成本；係數格式取 XKV2 裡這個秩組合用到的格式），在 wcost_states 各量一次。"""
    import torch
    from m10_speed import Busy, _watcher, model_glob
    import m10_dqsweep as DQ
    rid = run_id()
    os.makedirs(out, exist_ok=True)
    theta, mpath = check_config(model)
    L, H, D = SHAPES[model]
    dev = torch.device("cuda:0")
    km = None
    if recompute:
        os.environ["M7_MODEL_GLOB"] = model_glob(model)
        from m7_model import KVModel
        km = KVModel(max_len=N_CTX * CHUNK + CHUNK)
        if (km.L, km.H, km.D) != (L, H, D):
            raise ValueError(f"KVModel 形狀 {(km.L, km.H, km.D)} ≠ SHAPES {(L, H, D)}")
        kv = km.kv
    else:
        kv = torch.empty((L, 2, H, N_CTX * CHUNK + CHUNK, D), dtype=torch.bfloat16, device=dev)
    kv.zero_()
    fp8_fmt = json.load(open(DEFAULT_PATHS["fmt_choice"]))["choice"][model]["fp8"]

    def kern_of(fmt):
        return DQ.kstar(DQ.SW_DEFAULT, model, fmt)
    names = list(segments or seg_catalog())
    if "bf16" not in names:
        names.insert(0, "bf16")
    ctx = GCtx(torch, model, kv, n_blocks, theta, store_impl, seed)
    t_build = time.perf_counter()
    segs = build_segments(ctx, names, kern_of, fp8_fmt)
    t_build = time.perf_counter() - t_build
    bl = json.load(open(busy_levels))
    states = [(s, 0.0 if s == "idle" else float(bl[s])) for s in gpu_states]
    wc_info = None
    if wcost:
        wc_info = dict(params=xstack_consts_check(), pairs=[list(p) for p in wcost], states=list(wcost_states), reps=wcost_reps,
                       linalg=str(torch.backends.cuda.preferred_linalg_library()))
    watch, level = _watcher(os.path.join(out, f"gpu_guard_{rid}_{model}.json"))
    busy_pids, busy_stats = set(), []

    def contaminated():
        if watch is None:
            return "NOT_CHECKED"
        bad = {p: v for p, v in watch.intruders.items() if p not in busy_pids}
        return bool(bad) or watch.started_clean is False
    sdma = os.environ.get("HSA_ENABLE_SDMA", "unset(預設 1)")
    bytes_bf16 = n_blocks * segs["bf16"].bytes_blk
    rng = random.Random(f"x3|{model}|{seed}")
    rows, bg_rows, rec_rows = [], [], []
    ops = [(s, o, fn) for s, sg in segs.items() for o, fn in sg.ops.items()]
    for st, duty in states:
        with Busy(duty, watch) as b:
            if b.p:
                busy_pids.add(b.p.pid)
            res = _time_ops(torch, ops, reps, min_rep_s, rng)
            cflag = contaminated()
        if b.p:
            busy_stats.append(dict(phase="timing", state=st, **(b.stats or {})))
        for (s, o), (v, inner) in res.items():
            sg = segs[s]
            q1, q3 = quart(v)
            bop = sg.bytes_req if o == "h2d_req" else n_blocks * sg.bytes_blk + sg.bytes_req
            rows.append(dict(run_id=rid, ts=now_iso(), model=model, gpu_state=st, duty=duty, segment=s, family=sg.family,
                             kind=sg.kind, fmt=sg.fmt, op=o, impl=sg.impl.get(o, ""), kernel=sg.kernel, n_blocks=n_blocks,
                             bytes_op=bop, bytes_bf16=bytes_bf16, bytes_ratio=round(bop / bytes_bf16, 5),
                             n_copies=sg.n_copies.get(o, ""), reps=len(v), inner=inner, t_med_s=med(v), t_q1_s=q1,
                             t_q3_s=q3, t_min_s=min(v), t_max_s=max(v), contaminated=cflag, host_level=level,
                             hsa_enable_sdma=sdma, store_impl=store_impl))
        print(f"[timing] {model} {st} 完成（{len(res)} 個 op）", flush=True)
    # 10/21：xKV 的寫入（壓縮）成本；只出表，不進還原決策
    wc_rows = []
    if wcost:
        wfmts = {tuple(p): {f for a_, b_, f in XKV2.values() if (a_, b_) == tuple(p)} or {"bf16"} for p in wcost}
        wops, winfo, wstate = build_wcost(ctx, [tuple(p) for p in wcost], wfmts)
        for i_st, st in enumerate(wcost_states):
            duty = 0.0 if st == "idle" else float(bl[st])
            with Busy(duty, watch) as b:
                if b.p:
                    busy_pids.add(b.p.pid)
                res = _time_slow(torch, wops, wcost_reps, rng, warm=(i_st == 0))
                cflag = contaminated()
            if b.p:
                busy_stats.append(dict(phase="wcost", state=st, **(b.stats or {})))
            for (k, o), v in res.items():
                q1, q3 = quart(v)
                wc_rows.append(dict(winfo[(k, o)], run_id=rid, ts=now_iso(), model=model, gpu_state=st, duty=duty, op=o, reps=len(v),
                                    t_med_s=med(v), t_q1_s=q1, t_q3_s=q3, t_min_s=min(v), t_max_s=max(v), contaminated=cflag,
                                    host_level=level, linalg=wc_info["linalg"]))
            print(f"[wcost] {model} {st} 完成（{len(res)} 個 op）", flush=True)
        msgs = wcost_check(ctx, wstate, [tuple(p) for p in wcost], wfmts)
        wc_info["check"] = "ok" if not msgs else "FAIL：" + "；".join(msgs)
        wstate.clear()
        del wops
        torch.cuda.empty_cache()
        if msgs:
            if watch is not None:
                watch.__exit__(None, None, None)
            raise SystemExit(f"{model} wcost 的輸出核對沒過：{msgs}（規則 2：停下來）")
    # 背景被拖慢多少：每個 (duty, seg:op) 一個新的 Busy；前後各一次只有背景當基準
    want = bg_list if bg_list is not None else BG_DEFAULT
    pairs = [tuple(x.split(":")) for x in want]
    pairs = [(s, o) for s, o in pairs if s in segs and o in segs[s].ops]
    for st, duty in states:
        if duty <= 0 or not pairs:
            continue
        seq = [("none", "none")] + pairs + [("none", "none")]
        got = []
        for s, o in seq:
            with Busy(duty, watch) as b:
                if b.p:
                    busy_pids.add(b.p.pid)
                t0, it = time.perf_counter(), 0
                if s == "none":
                    time.sleep(bg_seconds)
                else:
                    fn = segs[s].ops[o]
                    while time.perf_counter() - t0 < bg_seconds:
                        fn()
                        torch.cuda.synchronize()
                        it += 1
                fg_s = time.perf_counter() - t0
                cflag = contaminated()
            stt = b.stats or {}
            got.append(dict(seg=s, op=o, fg_s=fg_s, it=it, n=stt.get("n_matmul"), tot=stt.get("seconds"), c=cflag,
                            err=stt.get("done_error")))
        base = [g["n"] / g["tot"] for g in got if g["seg"] == "none" and g["n"] and g["tot"]]
        br = statistics.mean(base) if base else None
        for g in got:
            fg_rate = ratio = NM
            if br and g["n"] is not None and g["tot"] and g["seg"] != "none":
                fg_rate = (g["n"] - br * (g["tot"] - g["fg_s"])) / g["fg_s"]
                ratio = fg_rate / br
            bg_rows.append(dict(run_id=rid, ts=now_iso(), model=model, gpu_state=st, duty=duty, segment=g["seg"], op=g["op"],
                                fg_s=round(g["fg_s"], 4), fg_iters=g["it"], fg_mean_s=(g["fg_s"] / g["it"]) if g["it"] else "",
                                bg_n=g["n"] if g["n"] is not None else NM, bg_total_s=g["tot"] if g["tot"] else NM,
                                bg_rate_base=br if br else NM, bg_rate_fg=fg_rate, bg_ratio=ratio, contaminated=g["c"]))
        print(f"[bg] {model} {st} 完成（{len(pairs)} 個 op）", flush=True)
    # 重算慢化（recompute 探針）
    if km is not None:
        ids = torch.randint(1000, min(120000, km.cfg.vocab_size), (N_CTX * CHUNK,),
                            generator=torch.Generator().manual_seed(seed)).to(dev)
        km.kv.zero_()
        for st, duty in states:
            with Busy(duty, watch) as b:
                if b.p:
                    busy_pids.add(b.p.pid)
                for i in rec_pos[:2]:
                    km.forward_span(ids[i * CHUNK:(i + 1) * CHUNK], i * CHUNK)
                torch.cuda.synchronize()
                for rep in range(rec_reps):
                    for i in rec_pos:
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        km.forward_span(ids[i * CHUNK:(i + 1) * CHUNK], i * CHUNK)
                        torch.cuda.synchronize()
                        rec_rows.append(dict(run_id=rid, ts=now_iso(), model=model, gpu_state=st, duty=duty, chunk_idx=i,
                                             rep=rep, s=time.perf_counter() - t0, contaminated=contaminated()))
            if b.p:
                busy_stats.append(dict(phase="recompute", state=st, **(b.stats or {})))
        print(f"[recompute] {model} 完成", flush=True)
    run_c = bool(watch.contaminated) if watch is not None else "NOT_CHECKED"
    if watch is not None:
        # 外人＝不在 Busy 的 pid 裡的（Busy 是自己的子行程，GpuWatcher 的 own_root 已排除；這裡再保守一次）
        run_c = bool({p for p in watch.intruders if p not in busy_pids}) or watch.started_clean is False
        watch.__exit__(None, None, None)
    for r in rows + bg_rows + rec_rows + wc_rows:
        r["run_contaminated"] = run_c
    if run_c is not False:
        # CLAUDE.md §3：被污染的量時間 run，結果不得寫進 results/ → 改寫到 run 目錄（adhoc 時寫到 out 底下標明的子目錄），map 讀不到
        rd = run_dir() or out
        open(os.path.join(rd, "CONTAMINATED"), "w").write(f"gpu_guard：{model} micro 有外來行程或查不到（{run_c}）\n")
        out = os.path.join(rd, f"contaminated_micro_{model}")
        os.makedirs(out, exist_ok=True)
        print(f"⚠ {model}：這次 micro 被污染（{run_c}）→ 輸出改寫到 {out}，不進 results；要重量", flush=True)
    write_csv(os.path.join(out, f"micro_{model}.csv"), rows, MICRO_FIELDS)
    write_csv(os.path.join(out, f"micro_bg_{model}.csv"), bg_rows, BG_FIELDS)
    if rec_rows:
        write_csv(os.path.join(out, f"micro_recompute_{model}.csv"), rec_rows, REC_FIELDS)
    if wc_rows:
        write_csv(os.path.join(out, f"wcost_{model}.csv"), wc_rows, WC_FIELDS)
    meta = dict(run_id=rid, ts=now_iso(), model=model, model_path=mpath, rope_theta=theta, shape=dict(L=L, H=H, D=D),
                n_blocks=n_blocks, region_tokens=[ctx.lo, ctx.hi], reps=reps, min_rep_s=min_rep_s, bg_seconds=bg_seconds,
                rec_pos=list(rec_pos) if km else [], rec_reps=rec_reps if km else 0, states=states, busy_levels=rel(busy_levels),
                hsa_enable_sdma=sdma, store_impl=store_impl, smoke=smoke, t_build_s=round(t_build, 1),
                torch=torch.__version__, hip=torch.version.hip, device=torch.cuda.get_device_name(0),
                fp8_fmt=fp8_fmt, contaminated=run_c, host_level=level, busy_stats=busy_stats, wcost=wc_info,
                note_rope="RoPE 用 rope_theta 的預設 inv_freq（沒有套 llama3 的 rope_scaling）：只影響數值，不影響時間",
                segments={s: dict(family=sg.family, kind=sg.kind, feed=sg.feed, ops=list(sg.ops), bytes_blk=sg.bytes_blk,
                                  bytes_req=sg.bytes_req, bytes_resident=sg.bytes_resident, fmt=sg.fmt, impl=sg.impl,
                                  kernel=sg.kernel, n_copies=sg.n_copies, check=sg.check) for s, sg in segs.items()})
    json.dump(meta, open(os.path.join(out, f"micro_meta_{model}.json"), "w"), indent=1, ensure_ascii=False, default=str)
    meta["out_dir"] = out
    del segs
    if km is not None:
        del km
    torch.cuda.empty_cache()
    return meta


def parse_pairs(xs):
    """["k22v34", ...] → [(22, 34), ...]；"x2" → 10/21 的 xKV 段用到的所有秩組合。"""
    if not xs:
        return None
    out = []
    for x in xs:
        if x == "x2":
            out += [p for p in XKV2_PAIRS if p not in out]
            continue
        m_ = re.fullmatch(r"k(\d+)v(\d+)", x)
        if not m_:
            raise SystemExit(f"--wcost 的秩組合要寫成 k<a>v<b> 或 x2：{x}")
        out.append((int(m_.group(1)), int(m_.group(2))))
    return out


def cmd_micro(a):
    err = guard_micro_out(a.out, a.segments)
    if err:
        raise SystemExit(err)
    if a.segments:
        unknown = sorted(set(a.segments) - set(seg_catalog()))
        if unknown:
            raise SystemExit(f"--segments 有 seg_catalog 沒有的名字：{unknown}")
    for m in a.model:
        gpu_micro(m, a.out, n_blocks=a.n_blocks, gpu_states=a.gpu_states, busy_levels=a.busy_levels, reps=a.reps,
                  min_rep_s=a.min_rep_s, segments=a.segments, bg_list=a.bg, bg_seconds=a.bg_seconds,
                  recompute=not a.no_recompute, rec_reps=a.rec_reps, store_impl=a.store_impl,
                  wcost=parse_pairs(a.wcost), wcost_states=a.wcost_states, wcost_reps=a.wcost_reps)


# ====================================================================== GPU：validate
def harness_spec(action, model, fp8_fmt):
    """第二階段的還原器（m10_speed grid）能不能跑這個動作；能 → (設定, 格式, 環境變數)，不能 → None。"""
    import m10_dqsweep as DQ
    if action == "raw":
        return dict(cfg="E0", fmt="int4_kivi", meth="S4", env=DQ.method_env("S4", model, "int4_kivi"))
    if "@" not in action:
        return None
    s, dec = action.split("@")
    if s in MAIN_Q:
        fmt = fp8_fmt if s == "fp8" else s
        meth = "S4" if dec == "oneshot" else "S0"
        return dict(cfg="E1", fmt=fmt, meth=meth, env=DQ.method_env(meth, model, fmt))
    if s in NEW_Q and dec == "perblock":
        fmt = NEW_Q[s]
        env = dict(DQ.method_env("S0", model, fmt), M10_CODEC_X="1", M10_DEQUANT_FAST_X="1")
        return dict(cfg="E1", fmt=fmt, meth="S0", env=env)
    return None


def _state_of(duty, bl):
    for k in GPU_STATES:
        if abs(float(bl[k]) - duty) < 1e-9:
            return k
    raise ValueError(f"duty {duty} 不在 busy_levels 裡")


def gpu_loadpath(model, items, reps, out, smoke=False):
    """不在第二階段還原器裡的動作：全部載入的「載入線」實測（真的 H2D＋限速＋解碼），回傳每個 item 的中位數。"""
    import torch
    from m10_speed import Busy, _watcher
    from m7_restore_harness import Tier, sleep_until
    import m10_dqsweep as DQ
    theta, _ = check_config(model)
    L, H, D = SHAPES[model]
    n = 4 if smoke else N_CTX
    kv = torch.zeros((L, 2, H, N_CTX * CHUNK + CHUNK, D), dtype=torch.bfloat16, device="cuda:0")
    ctx = GCtx(torch, model, kv, n, theta, "triton")
    fp8_fmt = json.load(open(DEFAULT_PATHS["fmt_choice"]))["choice"][model]["fp8"]
    names = sorted({"bf16"} | {it["seg"] for it in items})
    segs = build_segments(ctx, names, lambda f: DQ.kstar(DQ.SW_DEFAULT, model, f), fp8_fmt)
    bl = json.load(open(BUSY_LEVELS))
    rid = run_id()
    watch, level = _watcher(os.path.join(out, f"gpu_guard_{rid}_loadpath_{model}.json"))
    busy_pids = set()
    res = []
    for duty in sorted({it["duty"] for it in items}):
        with Busy(duty, watch) as b:
            if b.p:
                busy_pids.add(b.p.pid)
            for it in [x for x in items if x["duty"] == duty]:
                sg = segs[it["seg"]]
                dec = it["dec"]
                hop = "h2d_region" if dec == "none" else sg.feed[f"dq_{dec}"]
                h1 = sg.h2d_one[hop]
                tier = Tier("x3", read_Bps=it["GiBps"] * GiB, write_Bps=it["GiBps"] * GiB) if it["GiBps"] else None
                ts_, t_item = [], time.perf_counter()
                for rep in range(reps + 1):
                    if rep > 3 and time.perf_counter() - t_item > 60:      # 慢層一次要十幾秒：至少 3 次、最多花 60 秒
                        break
                    if tier:
                        tier.reset_clock()
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for j in reversed(range(n)):                       # 載入線從最後一個 block 往前（同 Restorer）
                        done = tier.reserve_read(sg.bytes_blk + (sg.bytes_req if j == 0 else 0)) if tier else 0.0
                        h1(j)
                        ev = torch.cuda.Event()
                        ev.record()
                        if tier:
                            sleep_until(done)
                        ev.synchronize()
                        if dec == "perblock":
                            sg.dq_one(j)
                            torch.cuda.synchronize()
                    if dec == "oneshot":
                        sg.dq_all()
                    torch.cuda.synchronize()
                    if rep:
                        ts_.append(time.perf_counter() - t0)            # 第 0 次是暖機
                bad = {p for p in (watch.intruders if watch else {}) if p not in busy_pids}
                res.append(dict(it, meas_s=med(ts_) * (N_CTX / n), n_reps=len(ts_), q=quart(ts_),
                                contaminated=(bool(bad) or (watch.started_clean is False)) if watch else "NOT_CHECKED",
                                note=f"{n} 個 block 量到再 ×{N_CTX // n}" if n != N_CTX else ""))
    if watch:
        watch.__exit__(None, None, None)
    del segs, kv
    torch.cuda.empty_cache()
    return res


def cmd_validate(a):
    rid, ts = run_id(), now_iso()
    os.makedirs(a.out, exist_ok=True)
    fl = json.load(open(os.path.join(a.map_dir, "flip_cells.json")))
    if not fl.get("worth_validate") and not a.force:
        msg = f"validate 不跑：{fl.get('reason')}"
        json.dump(dict(run_id=rid, ts=ts, skipped=True, reason=fl.get("reason"), map_run_id=fl.get("run_id")),
                  open(os.path.join(a.out, "validate_summary.json"), "w"), indent=1, ensure_ascii=False)
        print(msg)
        return 0
    cells = fl["cells"][:a.max_cells]
    bl = json.load(open(BUSY_LEVELS))
    tiers = {}
    C = load_constants()
    for m in MODELS:
        for t in tiers_named(m, C):
            tiers[(m, t["name"])] = t
    fp8 = {m: C.get(m, "fp8_fmt") for m in MODELS}
    hjobs, lp, plan = {}, {}, []
    for c in cells:
        t = tiers[(c["model"], c["tier"])]
        st = _state_of(c["duty"], bl)
        for x in c["measure"]:
            a_ = x["action"]
            base = dict(cell=c["cell"], mode=c["mode"], model=c["model"], tier=c["tier"], link_GiBps=t["GiBps"] or "",
                        duty=c["duty"], gpu_state=st, action=a_, role=x["role"], pred_s=x["pred_s"])
            if a_.startswith("ll:") or t["kind"] == "vllm":
                plan.append(dict(base, how="NOT_SUPPORTED", meas_s=None, meas_detail="無損＋CPU 解／vLLM 路徑沒有實作的還原路徑"))
                continue
            hs = harness_spec(a_, c["model"], fp8[c["model"]])
            if hs:
                key = (c["model"], hs["fmt"], hs["meth"], c["tier"])
                j = hjobs.setdefault(key, dict(states=set(), env=hs["env"], GiBps=t["GiBps"]))
                j["states"].add(st)
                plan.append(dict(base, how="harness", hkey=key, cfg=hs["cfg"]))
            else:
                s_, dec = (a_.split("@") + ["none"])[:2] if a_ != "raw" else ("bf16", "none")
                k_ = len(lp.setdefault(c["model"], []))
                lp[c["model"]].append(dict(base, how="loadpath", seg=s_, dec=dec, GiBps=t["GiBps"], lkey=k_))
                plan.append(dict(base, how="loadpath", lkey=k_))
    # 1) 第二階段的還原器（子行程）
    hres = {}
    for key, j in sorted(hjobs.items()):
        m, fmt, meth, tier = key
        d = os.path.join(a.out, "harness", f"{m}_{fmt}_{meth}_{tier}")
        os.makedirs(d, exist_ok=True)
        bwj = os.path.join(d, "bw_x3.json")
        tname = "cpu" if j["GiBps"] is None else tier
        json.dump({"targets": {} if j["GiBps"] is None else {tier: {"GiBps": j["GiBps"]}},
                   "_note": "X3 validate：限速器頻寬＝map 的名目層（出處見 map 的 constants_used.csv）"}, open(bwj, "w"))
        blocks = os.path.join(run_dir() or d, f"blocks_{m}_{fmt}_{meth}_{tier}.csv")
        cmd = [sys.executable, os.path.join(HERE, "m10_speed.py"), "grid", "--model", m, "--fmt", fmt,
               "--base-fmt", fp8[m], "--out-dir", d, "--bw-json", bwj, "--bw-names", tname,
               "--gpu-states", *sorted(j["states"], key=GPU_STATES.index), "--busy-levels", BUSY_LEVELS,
               "--configs", "E0", "E1", "--refs", "--lengths", "--extra-modes", "loadall",
               "--launches", str(a.launches), "--reps", str(a.reps), "--launch0", "400", "--blocks-out", blocks]
        if a.main_length:
            cmd += ["--main-length", str(a.main_length)]
        env = dict(os.environ, **j["env"], RUN_ID=rid)               # 子行程的列要能用同一個 run_id 撈回來
        print("[validate] 跑", " ".join(cmd), j["env"], flush=True)
        r = subprocess.run(cmd, env=env)
        if r.returncode != 0:
            raise SystemExit(f"m10_speed grid 失敗（exit {r.returncode}）：{key}（規則 2：停下來）")
        rows = [x for x in read_csv(os.path.join(d, "restores.csv")) if x["run_id"] == rid]
        gs = [json.load(open(g)) for g in glob.glob(os.path.join(d, f"gpu_guard_{rid}_*.json"))]
        clean = bool(gs) and all(g.get("verdict") == "CLEAN" for g in gs) and all(x["contaminated_gpu"] == "False" for x in rows)
        hres[key] = dict(rows=rows, clean=clean, dir=d, tname=tname)
    # 2) 載入線實測（同一個 process）
    lres = {}
    for m, items in lp.items():
        lres[m] = {x["lkey"]: x for x in gpu_loadpath(m, items, a.lp_reps, a.out, smoke=a.smoke)}   # 依 duty 分組跑，回來要照 lkey 對回去
    vrows = []
    for p in plan:
        row = {k: p.get(k) for k in VAL_FIELDS}
        row.update(run_id=rid, ts=ts)
        if p["how"] == "harness":
            h = hres[p["hkey"]]
            sel = [x for x in h["rows"] if x["config"] == p["cfg"] and x["bw_name"] == h["tname"] and x["gpu_state"] == p["gpu_state"]]
            mc = med(float(x["t_restore"]) for x in sel if x["mode"] == "cake")
            ml = med(float(x["t_restore"]) for x in sel if x["mode"] == "loadall")
            meas = min(v for v in (mc, ml) if v is not None) if (mc or ml) else None
            bad = [x["kv_bad"] for x in sel if x["kv_bad"] not in ("", "0", "SKIP_nodq")]
            row.update(meas_s=meas, meas_detail=f"cake {mc} / loadall {ml}；kv_bad {bad or 0}", src=rel(h["dir"]),
                       contaminated=not h["clean"])
            if bad:
                raise SystemExit(f"還原後 KV 不對：{p['cell']} {p['action']} kv_bad={bad}（規則 2）")
        elif p["how"] == "loadpath":
            x = lres[p["model"]][p["lkey"]]
            row.update(meas_s=x["meas_s"], meas_detail=f"全部載入（載入線）；{x['note']}", src="同一個 process",
                       contaminated=x["contaminated"])
            row["pred_s"] = _pred_loadall(a.map_dir, p)
        else:
            row.update(meas_s=None, contaminated="")
        if row.get("meas_s") and row.get("pred_s"):
            row["rel_err"] = row["meas_s"] / row["pred_s"] - 1
        vrows.append(row)
    write_csv(os.path.join(a.out, "validate_cells.csv"), vrows, VAL_FIELDS)
    s3 = stop3(vrows, cells)
    n_c = sum(str(r.get("contaminated")) not in ("False", "") for r in vrows)
    s3.update(run_id=rid, ts=ts, map_run_id=fl.get("run_id"), n_cells=len(cells), smoke=a.smoke, n_contaminated=n_c,
              n_not_supported=sum(r.get("how") == "NOT_SUPPORTED" for r in plan))
    json.dump(s3, open(os.path.join(a.out, "validate_summary.json"), "w"), indent=1, ensure_ascii=False, default=str)
    print(json.dumps({k: v for k, v in s3.items() if k != "pairs"}, ensure_ascii=False, indent=1, default=str))
    return 0


def _pred_loadall(map_dir, p):
    """載入線實測要和「全部載入」的預測比（不是 min(Cake, 全部載入)）。"""
    for r in read_csv(os.path.join(map_dir, "map_cells.csv")):
        if (r["grid"], r["model"], r["tier"], dkey(r["duty"]), r["action"]) == ("named", p["model"], p["tier"], dkey(p["duty"]),
                                                                                  p["action"]) and r["T_loadall_s"]:
            return float(r["T_loadall_s"])
    return None


def stop3(vrows, cells):
    """STOP-3：≥2 格的實測和預測差 >10%；或實測的 A 對 F <5%（只比同一種量法的配對）。污染列不算。"""
    ok = [r for r in vrows if r.get("meas_s") and r.get("pred_s") and str(r.get("contaminated")) in ("False", "")]
    bad_cells = sorted({r["cell"] for r in ok if abs(r["rel_err"]) > TH["stop3_err"]})
    gains = []
    for c in cells:
        rs = [r for r in ok if r["cell"] == c["cell"]]
        a_ = next((r for r in rs if "A" in str(r["role"]).split("+")), None)
        f_ = next((r for r in rs if "F" in str(r["role"]).split("+")), None)
        if c["A"] == c["F"] or not a_ or not f_ or a_["how"] != f_["how"]:
            continue
        gains.append(a_["meas_s"] / f_["meas_s"])
    gAF = (1 - gm(gains)) if gains else None
    holds = len(bad_cells) >= TH["stop3_cells"] or (gAF is not None and gAF < TH["stop3_gain"])
    return dict(holds=holds, bad_cells=bad_cells, n_compared=len(ok), gain_AF_measured=gAF, n_AF_pairs=len(gains),
                verdict=("STOP-3 成立：預測圖不可信或沒有好處 → 停" if holds else "STOP-3 不成立"),
                rule="≥2 格實測和預測差 >10%，或實測的 A 對 F（同量法）幾何平均好 <5% → 停")


# ====================================================================== 冒煙（GPU）
def cmd_smoke(a):
    out = a.out
    os.makedirs(out, exist_ok=True)
    fails = []
    mdir = os.path.join(out, "micro")
    segs = list(a.segments) if a.segments else None             # 10/21：只冒煙這些段（＋bf16）；不載模型、不量重算探針
    if segs:
        unknown = sorted(set(segs) - set(seg_catalog()))
        if unknown:
            raise SystemExit(f"--segments 有 seg_catalog 沒有的名字：{unknown}")
    want = set(segs or seg_catalog()) | {"bf16"}
    wc = parse_pairs(a.wcost) if a.wcost is not None else None
    for m in a.models:
        try:
            meta = gpu_micro(m, mdir, n_blocks=4, gpu_states=("idle", "busy2"), reps=1, min_rep_s=0.005, segments=segs,
                             bg_list=(BG_X2[:1] if segs else ["int4_kivi:dq_oneshot", "lr_r24:dq_oneshot"]), bg_seconds=0.5,
                             recompute=not segs, rec_pos=(0, 3), rec_reps=1, store_impl=a.store_impl, smoke=True,
                             wcost=wc, wcost_states=("idle", "busy2"), wcost_reps=1)
            if meta["contaminated"] is not False:                  # 冒煙只驗程式；污染不算失敗，但照實印出
                print(f"⚠ 冒煙 micro {m} 被污染（{meta['contaminated']}）：輸出在 {meta['out_dir']}，map 冒煙會缺這個模型", flush=True)
            bad = [s for s, v in meta["segments"].items() if v["check"] != "ok"]
            if bad:
                fails.append(f"{m}：正確性沒過 {bad}")
            if set(meta["segments"]) != want:
                fails.append(f"{m}：segment 不齊 {sorted(want - set(meta['segments']))}")
            if wc:
                wi = meta.get("wcost") or {}
                if wi.get("check") != "ok":
                    fails.append(f"{m}：wcost 核對 {wi.get('check')}")
                print(f"wcost {m}：svd 參數 {wi.get('params')}；linalg {wi.get('linalg')}", flush=True)
                wp = os.path.join(meta["out_dir"], f"wcost_{m}.csv")
                if os.path.exists(wp):
                    for r in read_csv(wp):
                        print(f"  wcost {m} {r['gpu_state']} {r['pair']} {r['op']}：{float(r['t_med_s']) * 1e3:.1f} ms"
                              f"（T={r['T']}，q={r['svd_q']}）", flush=True)
                else:
                    fails.append(f"{m}：沒有 wcost_{m}.csv")
            print(f"PASS micro {m}", flush=True)
        except BaseException as ex:                                # 冒煙要把錯誤照實記下來再停
            fails.append(f"{m} micro：{ex!r}")
            print(f"FAIL micro {m}: {ex!r}", flush=True)
    if not fails:
        try:
            s = run_map(mdir, os.path.join(out, "map"), smoke=True)
            print("PASS map（冒煙資料）", s["validate_reason"], flush=True)
        except BaseException as ex:
            fails.append(f"map：{ex!r}")
    if not fails:
        try:
            fl = os.path.join(out, "map", "flip_cells.json")
            d = json.load(open(fl))
            C = load_constants()
            b32 = C.get("llama", "link_b32")
            d["worth_validate"], d["reason"] = True, "冒煙：固定兩格"
            xs = [x for x in (segs or []) if x in XKV2]
            if segs and xs:                                       # 10/21：新段的載入線（h2d 含基底、一次拆、逐 block 拆）
                xa = next((x for x in xs if XKV2[x][2] == "kivi4"), xs[-1])          # Llama：INT4 係數的一次拆＋逐 block、BF16 係數的一次拆
                xb = next((x for x in xs if XKV2[x][2] == "int8_token"), xs[-1])     # Qwen：int8 係數的逐 block
                d["cells"] = [dict(cell="llama|b32|0.0", mode="X", model="llama", tier="b32", link_GiBps=b32, duty=0.0,
                                   priority=0, A=f"{xa}@oneshot", S=f"{xa}@perblock", F=f"{xs[0]}@oneshot",
                                   measure=[dict(action=f"{xa}@oneshot", role="A", pred_s=None),
                                            dict(action=f"{xa}@perblock", role="S", pred_s=None),
                                            dict(action=f"{xs[0]}@oneshot", role="F", pred_s=None)]),
                              dict(cell="qwen|b32|0.0", mode="X", model="qwen", tier="b32", link_GiBps=C.get("qwen", "link_b32"),
                                   duty=0.0, priority=0, A=f"{xb}@perblock", S=f"{xb}@perblock", F=f"{xb}@perblock",
                                   measure=[dict(action=f"{xb}@perblock", role="A", pred_s=None)])]
                d["cells"] = [c for c in d["cells"] if c["model"] in a.models]
            else:
                d["cells"] = [dict(cell="llama|cpu_pinned|0.0", mode="Q", model="llama", tier="cpu_pinned", link_GiBps="", duty=0.0,
                                   priority=0, A="int4_kivi@oneshot", S="int4_kivi@oneshot", F="int4_kivi@oneshot",
                                   measure=[dict(action="int4_kivi@oneshot", role="A", pred_s=None),
                                            dict(action="raw", role="D", pred_s=None)]),
                              dict(cell="llama|b32|0.0", mode="X", model="llama", tier="b32", link_GiBps=b32, duty=0.0,
                                   priority=0, A="lr_r24@oneshot", S="lr_r24@oneshot", F="lr_r24@oneshot",
                                   measure=[dict(action="lr_r24@oneshot", role="A", pred_s=None)])]
            json.dump(d, open(fl, "w"), indent=1, ensure_ascii=False)
            va = argparse.Namespace(map_dir=os.path.join(out, "map"), out=os.path.join(out, "validate"), force=True, max_cells=8,
                                    launches=1, reps=1, lp_reps=1, main_length=4096, smoke=True)
            cmd_validate(va)
            vr = read_csv(os.path.join(out, "validate", "validate_cells.csv"))
            if not all(r["meas_s"] not in ("", "None") for r in vr):
                fails.append(f"validate：有格子沒有實測 {[(r['cell'], r['action']) for r in vr if r['meas_s'] in ('', 'None')]}")
            else:
                print("PASS validate（冒煙）", flush=True)
        except BaseException as ex:
            fails.append(f"validate：{ex!r}")
    print("ALL PASS" if not fails else f"{len(fails)} FAIL：" + "；".join(fails), flush=True)
    return 0 if not fails else 1


# ====================================================================== 自測（只用 CPU，不 import torch）
def _synthetic_micro(model, spec, duties=(0.0, 0.4, 0.7, 0.9), n=32, rec=None):
    """spec：{(seg, op): 秒（duty 0）}；忙時的倍數 spec_busy 另給。回傳 load_micro 同格式的 dict。"""
    L, H, D = SHAPES[model]
    bb = L * 2 * H * CHUNK * D * 2
    cat = seg_catalog()
    ratio = {"bf16": 1.0, "fp8": 0.5, "int8_token": 0.53, "int4_kivi": 0.3125, "int2_kivi": 0.1875, "k4v2": 0.25,
             "lr_r24": 0.1875, "lr_r40": 0.3125, "merge_g2": 0.5, "lrk24_v4": 0.25}
    segs = {}
    for s in {k[0] for k in spec["t"]}:
        if s in XKV2:                                    # 10/21 的 xKV 段：位元組照位元組帳（基底算 bytes_req）
            cb_, br_ = xkv_bytes(model, *XKV2[s], CHUNK)
            segs[s] = dict(cat[s], bytes_blk=cb_, bytes_req=br_)
        else:
            segs[s] = dict(cat[s], bytes_blk=int(bb * ratio.get(s, 0.25)), bytes_req=0)
    t = {}
    for (s, o), v in spec["t"].items():
        for d in duties:
            x = spec.get("busy", {}).get((s, o), {}).get(d, 1.0 + 0.5 * d)
            t[(d, s, o)] = dict(med=v * x, q1=v * x, q3=v * x, n=5)
    return dict(run_id=f"SELFTEST-{model}", meta={}, n_blocks=n, seg=segs, t=t, rec=rec or {}, bg={}, dropped=[],
                smoke=False, file="(合成)")


def _synthetic_consts(models=("llama",)):
    C = Consts()
    for m in models:
        L, H, D = SHAPES[m]
        C.add(m, "f_idle_s", [0.030 + 0.001 * i for i in range(N_CTX)], "s", "(合成)", "SELFTEST", "")
        C.add(m, "chunk_bytes", L * 2 * H * CHUNK * D * 2, "B", "(合成)", "SELFTEST", "")
        for k, v in (("b16", 5.0), ("b32", 1.5), ("b48", 0.4)):
            C.add(m, f"link_{k}", v, "GiB/s", "(合成)", "SELFTEST", "")
        C.add(m, "mult_f_stage2_d0.7", 2.0, "×", "(合成)", "SELFTEST", "")
        C.add(m, "fp8_fmt", "fp8_token", "-", "(合成)", "SELFTEST", "")
    for k, v in (("ssd_meas", 7.87), ("ssd_param", 6.98), ("nfs_meas", 0.383), ("nfs_param", 0.331)):
        C.add("-", f"link_{k}", v, "GiB/s", "(合成)", "SELFTEST", "")
    C.add("-", "mult_f_phase1_d0.4", 1.5, "×", "(合成)", "SELFTEST", "")
    C.add("llama", "vllm_c6_T_s", 0.97, "s", "(合成)", "SELFTEST", "")
    C.add("llama", "vllm_c6_bytes", 4.0 * GiB, "B", "(合成)", "SELFTEST", "")
    return C


def _write_synth_micro(d, model, rid, segs, times, n=32, rec=None, wcost=None, smoke=False):
    """自測用：合成的 micro 寫成 gpu_micro 的檔案格式。segs：{seg: (bytes_blk, bytes_req)}；times：{(duty, seg, op): (秒, 污染)}；
    rec：{duty: 重算慢化倍數}；wcost：[(duty, 秩組合, op, 秒, 污染)]。"""
    cat = seg_catalog()
    meta = dict(run_id=rid, n_blocks=n, smoke=smoke,
                segments={s: dict(cat[s], bytes_blk=b, bytes_req=r) for s, (b, r) in segs.items()})
    json.dump(meta, open(os.path.join(d, f"micro_meta_{model}.json"), "w"))
    write_csv(os.path.join(d, f"micro_{model}.csv"),
              [dict(run_id=rid, ts="t", model=model, duty=du, segment=sg, op=op, t_med_s=v, t_q1_s=v, t_q3_s=v, reps=5,
                    contaminated=c, run_contaminated=False) for (du, sg, op), (v, c) in times.items()], MICRO_FIELDS)
    if rec:
        write_csv(os.path.join(d, f"micro_recompute_{model}.csv"),
                  [dict(run_id=rid, ts="t", model=model, duty=du, chunk_idx=i, rep=0, s=0.03 * k, contaminated=False,
                        run_contaminated=False) for du, k in {0.0: 1.0, **rec}.items() for i in (0, 8)], REC_FIELDS)
    if wcost:
        write_csv(os.path.join(d, f"wcost_{model}.csv"),
                  [dict(run_id=rid, ts="t", model=model, duty=du, pair=pr, op=op, reps=3, t_med_s=v, t_q1_s=v, t_q3_s=v,
                        t_min_s=v, t_max_s=v, contaminated=c, run_contaminated=False) for du, pr, op, v, c in wcost], WC_FIELDS)


def _self_test_x2gap(check):
    """10/21 補缺口的自測（只用 CPU、不 import torch）：新段的目錄；位元組帳（和 m16_xstack.predict_bytes 逐項核對，再和 X2 實際寫出的
    ratio 核對）；基底 H2D 的攤法；多個 micro 目錄的合併與漂移；map 讀 --x2；寫入成本表；第一版目錄的保護。"""
    import tempfile
    cat = seg_catalog()
    want = {"xkv_k11v17", "xkv_k22v34", "xkv_k28v42", "xkv_k11v17_int4", "xkv_k22v34_int4", "xkv_k28v42_int4", "xkv_k22v34_int8"}
    check("10/21：seg_catalog 有 7 個新的 xKV 段（名字照和 X2 的約定）", want <= set(cat) and set(XKV2) == want,
          sorted(set(XKV2) ^ want))
    check("10/21：新段的 ops／feed／家族（BF16 係數＝xlayer 單一方法；係數量化＝combo）", all(
        set(cat[s]["ops"]) == {"h2d_region", "h2d_req", "dq_oneshot", "dq_perblock"}
        and cat[s]["feed"] == {"dq_oneshot": "h2d_region", "dq_perblock": "h2d_region"}
        and (cat[s]["family"], cat[s]["kind"]) == (("xlayer", "single") if XKV2[s][2] == "bf16" else ("combo", "combo"))
        for s in want))
    check("10/21：新段對應的 X2 組合寫法", XKV2_SPEC == {
        "xkv_k11v17": "xl:k11:v17", "xkv_k22v34": "xl:k22:v34", "xkv_k28v42": "xl:k28:v42",
        "xkv_k11v17_int4": "xl:k11:v17+q:int4_kivi", "xkv_k22v34_int4": "xl:k22:v34+q:int4_kivi",
        "xkv_k28v42_int4": "xl:k28:v42+q:int4_kivi", "xkv_k22v34_int8": "xl:k22:v34+q:int8_token"}, XKV2_SPEC)
    try:
        import m16_xstack as XS
    except Exception as ex:                                      # noqa: BLE001
        check("10/21：import m16_xstack（模組層只有標準函式庫）", False, repr(ex))
        return
    check("10/21：import m16_xstack 沒有帶進 torch", "torch" not in sys.modules)
    check("10/21：svd_lowrank 參數和 m16_xstack 相同（niter、多取幾個、seed、窗口）",
          (XS.XL_NITER, XS.XL_OVERSAMPLE, XS.XL_SEED, XS.parse_spec("xl:k8:v8")[0]["w"]) == (XL_NITER, XL_OVERSAMPLE, XL_SEED, XKV_W))
    check("10/21：m16_xstack.MODEL_INFO 的形狀＝SHAPES",
          all((XS.MODEL_INFO[m]["L"], XS.MODEL_INFO[m]["H"], XS.MODEL_INFO[m]["D"]) == SHAPES[m] for m in MODELS))
    bad = []
    for m in MODELS:
        for sname, spec in XKV2_SPEC.items():
            try:
                st = XS.validate(XS.parse_spec(spec), XS.MODEL_INFO[m], 32)
                by = XS.predict_bytes(st, XS.MODEL_INFO[m], 32)
                got = (by["k.coef"] + by["v.coef"], by["k.basis"] + by["v.basis"])
                mine, blk = xkv_bytes(m, *XKV2[sname], 32 * CHUNK), xkv_bytes(m, *XKV2[sname], CHUNK)
                if XS.canonical(st) != spec or got != mine or (blk[0] * 32, blk[1]) != mine or \
                        set(by) != {"k.coef", "v.coef", "k.basis", "v.basis"}:
                    bad.append((m, sname, XS.canonical(st), got, mine, blk))
            except Exception as ex:                              # noqa: BLE001
                bad.append((m, sname, repr(ex)))
    check("10/21：位元組帳＝m16_xstack.predict_bytes（32 個 block、兩個模型、7 段；逐 block ×32＝整段；組合寫法是 canonical）",
          not bad, bad[:3])
    xp0 = os.path.join(REPO, "results", "m16_stage3_mi300x", "x_stack", "x2_unmapped.csv")
    try:
        inv = {v: k for k, v in XKV2_SPEC.items()}
        n_cmp, bad = 0, []
        for r in read_csv(xp0):
            if r["combo"] not in inv:
                continue
            m, nreg = r["model"], XS.N_BLOCKS - 1
            L, H, D = SHAPES[m]
            c_, b_ = xkv_bytes(m, *XKV2[inv[r["combo"]]], nreg * CHUNK)
            ratio = round((c_ + b_) / (nreg * L * 2 * H * CHUNK * D * 2), 6)
            n_cmp += 1
            if abs(ratio - float(r["ratio"])) > 1e-6:
                bad.append((m, r["combo"], ratio, r["ratio"]))
        check("10/21：位元組帳 vs X2 實際寫出的 ratio（第一版 x2_unmapped.csv，63 個 block；規則 6 的資料自身核對）",
              n_cmp >= 8 and not bad, f"比了 {n_cmp} 列；{bad[:3]}")
    except (OSError, KeyError, ValueError) as ex:
        check("10/21：讀 X2 的 x2_unmapped.csv", False, repr(ex))
    # 基底 H2D 的攤法
    sp = {("bf16", "h2d_region"): 0.040, ("xkv_k22v34_int4", "h2d_region"): 0.010, ("xkv_k22v34_int4", "h2d_req"): 0.004,
          ("xkv_k22v34_int4", "dq_oneshot"): 0.006, ("xkv_k22v34_int4", "dq_perblock"): 0.009,
          ("xkv_g4_f8", "h2d_region"): 0.008, ("xkv_g4_f8", "dq_oneshot"): 0.004}
    Mx = _synthetic_micro("llama", dict(t=sp, busy={k: {d: 1.0 for d in (0.0, 0.4, 0.7, 0.9)} for k in sp}))
    Mx["seg"]["xkv_g4_f8"]["bytes_req"] = 67108864
    ax = dict(name="xkv_k22v34_int4@oneshot", seg="xkv_k22v34_int4", dec="oneshot", family="combo", kind="combo")
    cx = blk_costs(Mx, ax, 0.0)
    sx = Mx["seg"]["xkv_k22v34_int4"]
    check("10/21：基底的 H2D＝(h2d − h2d_req)÷n ＋ h2d_req÷64；位元組＝係數＋基底÷64",
          cx["req"] == "split" and abs(cx["h2d"] - ((0.010 - 0.004) / 32 + 0.004 / 64)) < 1e-15
          and abs(cx["bytes"] - (sx["bytes_blk"] + sx["bytes_req"] / 64)) < 1e-6, cx)
    c8 = blk_costs(Mx, dict(ax, name="xkv_g4_f8@oneshot", seg="xkv_g4_f8", family="xlayer", kind="single"), 0.0)
    check("10/21：沒有 h2d_req 的舊段照第一版（線性外推、標 linear）", c8["req"] == "linear" and abs(c8["h2d"] - 0.008 / 32) < 1e-15)
    pin = dict(name="cpu_pinned", kind="pinned", GiBps=None)
    rr = restore_time([0.03] * N_CTX, 1.0, ax, pin, 0.0, Mx)
    check("10/21：xKV 一次拆、CPU 層的全部載入＝64×h2d ＋ 64×dq÷32", abs(rr["T_la"] - (64 * cx["h2d"] + 64 * 0.006 / 32)) < 1e-12)
    rows = build_cells({"llama": Mx}, _synthetic_consts(), None, None, ("llama",))
    fx = [r["flags"] for r in rows if r["action"] == "xkv_k22v34_int4@oneshot" and r["tier"] == "cpu_pinned"]
    f8 = [r["flags"] for r in rows if r["action"] == "xkv_g4_f8@oneshot" and r["tier"] == "cpu_pinned"]
    check("10/21：map_cells 的旗標（新段 h2d_req、舊段線性外推）", fx and f8 and all("h2d_req" in x for x in fx)
          and all("線性外推" in x for x in f8), (fx[:1], f8[:1]))
    check("10/21：--wcost 的秩組合", parse_pairs(["x2"]) == [(11, 17), (22, 34), (28, 42)] and parse_pairs(["k22v34"]) == [(22, 34)]
          and parse_pairs(None) is None)
    check("10/21 保護：合併多個 micro 的 map 不准寫進第一版目錄",
          bool(guard_map_out(RES, ["a", "b"])) and bool(guard_map_out(RES + "/", ["a", "b"])) and guard_map_out(RES, ["a"]) is None
          and guard_map_out(os.path.join(RES, "v2"), ["a", "b"]) is None)
    check("10/21 保護：只量部分段的 micro 不准寫進第一版目錄",
          bool(guard_micro_out(RES, ["xkv_k22v34"])) and guard_micro_out(RES, None) is None
          and guard_micro_out(RES, list(seg_catalog())) is None and guard_micro_out(os.path.join(RES, "v2"), ["xkv_k22v34"]) is None)
    # 多個 micro 目錄的合併 → map（讀 --x2）→ 寫入成本表
    L, H, D = SHAPES["llama"]
    bb = L * 2 * H * CHUNK * D * 2
    with tempfile.TemporaryDirectory() as td:
        d1, d2, d3, d4 = (os.path.join(td, x) for x in ("v1", "v2", "out", "v4"))
        for x in (d1, d2, d4):
            os.makedirs(x)
        duties = (0.0, 0.9)
        t1, t2 = {}, {}
        for du in duties:
            k = 1.0 + du
            anc = {(du, "bf16", "h2d_region"): 0.040 * k, (du, "int4_kivi", "h2d_stack"): 0.0137 * k,
                   (du, "int4_kivi", "dq_oneshot"): 0.0017 * k, (du, "int4_kivi", "dq_perblock"): 0.0022 * k}
            t1.update({kk: (v, False) for kk, v in anc.items()})
            t1.update({(du, "xkv_g4_f8", "h2d_region"): (0.0073 * k, False), (du, "xkv_g4_f8", "dq_oneshot"): (0.004 * k, False)})
            t2.update({kk: (v * (1.1 if kk[1] == "bf16" else 1.0), False) for kk, v in anc.items()})
            t2.update({(du, "xkv_k22v34_int4", "h2d_region"): (0.0050 * k, False),
                       (du, "xkv_k22v34_int4", "h2d_req"): (0.0025 * k, False),
                       (du, "xkv_k22v34_int4", "dq_oneshot"): (0.0060 * k, False),
                       (du, "xkv_k22v34_int4", "dq_perblock"): (0.0080 * k, du > 0)})
        _write_synth_micro(d1, "llama", "R1", {"bf16": (bb, 0), "int4_kivi": (bb * 5 // 16, 0), "xkv_g4_f8": (8388608, 67108864)},
                           t1, rec={0.9: 2.5})
        wc = [(du, "k22v34", op, v * (1 + du), op == "q_bf16") for du in duties
              for op, v in (("svd_k", 2.0), ("svd_v", 3.0), ("q_kivi4", 0.01), ("q_int8_token", 0.008), ("q_bf16", 0.002))]
        _write_synth_micro(d2, "llama", "R2", {"bf16": (bb, 0), "int4_kivi": (bb * 5 // 16, 0),
                                               "xkv_k22v34_int4": xkv_bytes("llama", 22, 34, "kivi4", CHUNK)}, t2, wcost=wc)
        M2 = load_micro_dirs([d1, d2], ("llama",))["llama"]
        check("10/21 合併：新段來自第二個目錄；舊段、錨點、重算探針用第一個", M2["seg_rid"]["xkv_k22v34_int4"] == "R2"
              and M2["seg_rid"]["bf16"] == "R1" and abs(M2["t"][(0.0, "bf16", "h2d_region")]["med"] - 0.040) < 1e-12
              and (0.0, "xkv_k22v34_int4", "h2d_req") in M2["t"] and M2["run_id"] == "R1;R2"
              and M2["rec"].get(0.9, {}).get("run_id") == "R1")
        dr = {(x["duty"], x["segment"], x["op"]): x for x in M2["drift"]}
        check("10/21 合併：重疊的段只算漂移（bf16 後÷前＝1.1、不覆蓋；4 個 op × 2 個 duty）",
              len(dr) == 8 and abs(dr[(0.0, "bf16", "h2d_region")]["ratio"] - 1.1) < 1e-9
              and not dr[(0.0, "bf16", "h2d_region")]["warn"] and abs(dr[(0.9, "int4_kivi", "dq_oneshot")]["ratio"] - 1) < 1e-9)
        check("10/21 合併：污染列不用（新段忙時的 dq_perblock、wcost 的 q_bf16）",
              (0.9, "xkv_k22v34_int4", "dq_perblock") not in M2["t"] and (0.0, "k22v34", "q_bf16") not in M2["wcost"]
              and (0.0, "k22v34", "svd_k") in M2["wcost"])
        _write_synth_micro(d4, "llama", "R4", {"bf16": (bb, 0)}, {(0.0, "bf16", "h2d_region"): (0.04, False)}, n=4)
        try:
            load_micro_dirs([d1, d4], ("llama",))
            raised = False
        except ValueError:
            raised = True
        check("10/21 合併：載入區 block 數不同 → 丟例外", raised)
        xp = os.path.join(td, "x2_feasible.csv")
        write_csv(xp, [dict(model="llama", x3_segment="xkv_k22v34_int4", feasible="1", run_id="X2V2"),
                       dict(model="llama", x3_segment="xkv_g4_f8", feasible="0", run_id="X2V2"),
                       dict(model="llama", x3_segment="xkv_k28v42_int4", feasible="1", run_id="X2V2")],
                  ["model", "x3_segment", "feasible", "run_id"])
        fs = feasible_sets({"llama": M2}, None, load_x2(xp), ("llama",))
        check("10/21 --x2：可行的新段進模式 X、不可行的不進、模式 Q 不變",
              {"xkv_k22v34_int4@oneshot", "xkv_k22v34_int4@perblock"} <= fs["X"]["llama"]
              and "xkv_g4_f8@oneshot" not in fs["X"]["llama"] and "xkv_k22v34_int4@oneshot" not in fs["Q"]["llama"])
        try:
            cmd_map(argparse.Namespace(micro_dir=[d1, d2], out=d3, s1=None, x2=xp, i4=None))
            sm = json.load(open(os.path.join(d3, "map_summary.json")))
            cells = read_csv(os.path.join(d3, "map_cells.csv"))
            acts = {r["action"] for r in cells}
            check("10/21 map --x2：用指定的檔、模式 X 有 X2、新段進動作集", sm["x2"]["file"] == rel(xp) and sm["verdicts"]["X"]["x2"]
                  and {"xkv_k22v34_int4@oneshot", "xkv_k22v34_int4@perblock"} <= acts, sm["x2"])
            fl = [r["flags"] for r in cells if r["action"] == "xkv_k22v34_int4@oneshot" and r["tier"] == "cpu_pinned"
                  and r["grid"] == "named"]
            check("10/21 map：新段標出「另一個 micro run」與 h2d_req、micro_run_id＝R2",
                  fl and all("另一個 micro run" in x and "h2d_req" in x for x in fl)
                  and {r["micro_run_id"] for r in cells if r["segment"] == "xkv_k22v34_int4"} == {"R2"})
            check("10/21 map：T-a 把 xKV＋INT4 當組合", (sm["verdicts"]["X"]["stacking"].get("T_a") or {}).get("n_cells", 0) > 0)
            check("10/21 map：X2 判可行但 micro 沒量的段照實列出（模式 X 不完整）",
                  sm["verdicts"]["X"].get("x2_feasible_unmeasured") == {"llama": ["xkv_k28v42_int4"]}
                  and "模式 X 不完整" in open(os.path.join(d3, "map_report.md")).read(), sm["verdicts"]["X"].get("x2_feasible_unmeasured"))
            wct = read_csv(os.path.join(d3, "write_cost.csv"))
            row = [r for r in wct if r["segment"] == "xkv_k22v34_int4" and float(r["duty"]) == 0.0]
            check("10/21 寫入成本表：合計＝SVD K＋SVD V＋量化；有 run_id、ts、micro_run_id",
                  row and abs(float(row[0]["t_total_s"]) - 5.01) < 1e-9 and row[0]["micro_run_id"] == "R2"
                  and all(r["run_id"] and r["ts"] for r in wct), row[:1])
            nm_ = [r for r in wct if r["segment"] == "xkv_k22v34" and float(r["duty"]) == 0.0]
            oth = [r for r in wct if r["segment"] == "xkv_k11v17_int4"]
            check("10/21 寫入成本表：缺一項（q_bf16 被污染）、沒量的秩組合 → NOT_MEASURED（不是 0）",
                  nm_ and nm_[0]["t_total_s"] == NM and "q_bf16" in nm_[0]["note"] and oth and all(r["t_total_s"] == NM for r in oth))
            rp = open(os.path.join(d3, "map_report.md")).read()
            check("10/21 map 報告：有漂移表、寫入成本表、micro 目錄與 X2 的出處", "跨 run 漂移" in rp and "寫入路徑成本" in rp
                  and "X2 品質可行集" in rp and "X2V2" in rp)
        except Exception as ex:                                  # noqa: BLE001
            check("10/21 map（合成的兩個 micro 目錄＋--x2）", False, repr(ex))
        try:
            cmd_map(argparse.Namespace(micro_dir=[d1, d2], out=os.path.join(td, "o2"), s1=None, x2=os.path.join(td, "nope.csv"), i4=None))
            raised = False
        except SystemExit:
            raised = True
        check("10/21 map：--x2 給了但檔不在 → 停、什麼都不寫（不當成「沒有 X2」）",
              raised and not os.path.exists(os.path.join(td, "o2", "map_summary.json")))


def self_test(out=None) -> bool:
    import tempfile
    assert "torch" not in sys.modules, "自測開始時 torch 已經被 import 了"
    ok, n_chk = True, 0

    def check(name, cond, info=""):
        nonlocal ok, n_chk
        n_chk += 1
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name + (f"  {info}" if info else ""), flush=True)

    import m10_sweep_sim as SIM
    # 1. simulate 的手算例：f 每塊 1、載入每塊 1、4 塊 → 重算 2 塊、載入 2 塊、時間 2
    p, t, tr, tl = SIM.simulate([1.0] * 4, [1.0] * 4)
    check("simulate 手算（4 塊、各 1 s）", (p, t) == (2, 2.0), f"{(p, t)}")
    # 2. 還原模型：pinned 層、一次拆的組成
    base = {("bf16", "h2d_region"): 0.032, ("int4_kivi", "h2d_stack"): 0.010, ("int4_kivi", "dq_oneshot"): 0.0016,
            ("int4_kivi", "dq_perblock"): 0.0048}
    Mm = _synthetic_micro("llama", dict(t=base, busy={(s, o): {d: 1.0 for d in (0.0, 0.4, 0.7, 0.9)} for s, o in base}))
    C = _synthetic_consts()
    f_idle = C.get("llama", "f_idle_s")
    pin = dict(name="cpu_pinned", kind="pinned", GiBps=None)
    a_os = dict(name="int4_kivi@oneshot", seg="int4_kivi", dec="oneshot", family="quant", kind="single")
    r = restore_time(f_idle, 1.0, a_os, pin, 0.0, Mm)
    want_la = 64 * (0.010 / 32) + 64 * (0.0016 / 32)
    check("一次拆、CPU 層：全部載入＝64×(h2d＋dq)", abs(r["T_la"] - want_la) < 1e-12 and r["mode"] == "loadall",
          f"{r['T_la']:.6f} vs {want_la:.6f}，mode {r['mode']}")
    # 3. 限速層：每塊＝max(位元組÷頻寬, h2d)
    slow = dict(name="nfs", kind="throttle", GiBps=0.331)
    c = blk_costs(Mm, a_os, 0.0)
    load = max(c["bytes"] / (0.331 * GiB), c["h2d"])
    p, t, tr, tl = SIM.simulate(f_idle, [load] * 64)
    r = restore_time(f_idle, 1.0, a_os, slow, 0.0, Mm)
    check("限速層 Cake＝max(重算線, 載入線＋dq×載入塊數)", abs(r["T_cake"] - max(tr, tl + c["dq_os"] * (64 - p))) < 1e-12
          and r["mode"] == "cake", f"b={r['b']}")
    # 4. 逐 block 拆把 dq 加在每塊載入上
    a_pb = dict(a_os, name="int4_kivi@perblock", dec="perblock")
    r2 = restore_time(f_idle, 1.0, a_pb, pin, 0.0, Mm)
    check("逐 block：全部載入＝64×(h2d＋dq_pb)", abs(r2["T_la"] - 64 * (0.010 / 32 + 0.0048 / 32)) < 1e-12)
    # 5. duty 的重算慢化缺 → None
    check("重算慢化缺 → None（NOT_MEASURED）", restore_time(f_idle, None, a_os, slow, 0.9, Mm) is None)
    m9, src = f_mult(Mm, C, "llama", 0.9)
    check("duty 0.9 沒有探針也沒有常數 → NOT_MEASURED", m9 is None and src == NM)
    m7, src7 = f_mult(Mm, C, "llama", 0.7)
    check("duty 0.7 沒有探針 → 用第二階段常數", m7 == 2.0 and "第二階段" in src7)
    Mm2 = dict(Mm, rec={0.9: dict(mult=3.1, n_pos=9)})
    check("有探針 → 用探針", f_mult(Mm2, C, "llama", 0.9)[0] == 3.1)
    # 6. 污染列不用、CSV 往返（load_micro）
    with tempfile.TemporaryDirectory() as d:
        meta = dict(run_id="R1", n_blocks=32, segments={"bf16": dict(seg_catalog()["bf16"], bytes_blk=100, bytes_req=0)})
        json.dump(meta, open(os.path.join(d, "micro_meta_llama.json"), "w"))
        rows = [dict(run_id="R1", ts="t", model="llama", gpu_state="idle", duty=0.0, segment="bf16", op="h2d_region",
                     t_med_s=0.03, t_q1_s=0.03, t_q3_s=0.03, reps=5, contaminated=False, run_contaminated=False),
                dict(run_id="R1", ts="t", model="llama", gpu_state="busy2", duty=0.7, segment="bf16", op="h2d_region",
                     t_med_s=0.05, t_q1_s=0.05, t_q3_s=0.05, reps=5, contaminated=True, run_contaminated=True),
                dict(run_id="R0", ts="t", model="llama", gpu_state="busy1", duty=0.4, segment="bf16", op="h2d_region",
                     t_med_s=0.04, t_q1_s=0.04, t_q3_s=0.04, reps=5, contaminated=False, run_contaminated=False)]
        write_csv(os.path.join(d, "micro_llama.csv"), rows, MICRO_FIELDS)
        rec = [dict(run_id="R1", ts="t", model="llama", gpu_state=s, duty=dd, chunk_idx=i, rep=0, s=v,
                    contaminated=False, run_contaminated=False)
               for s, dd, k in (("idle", 0.0, 1.0), ("busy3", 0.9, 3.0)) for i, v in ((0, 0.03 * k), (8, 0.04 * k))]
        write_csv(os.path.join(d, "micro_recompute_llama.csv"), rec, REC_FIELDS)
        LM = load_micro(d, ("llama",))["llama"]
        check("load_micro：污染列丟掉、只取最新 run", set(LM["t"]) == {(0.0, "bf16", "h2d_region")} and len(LM["dropped"]) == 1,
              f"{sorted(LM['t'])}")
        check("load_micro：探針倍數＝各位置倍數的中位數", abs(LM["rec"][0.9]["mult"] - 3.0) < 1e-9)
    # 7. decide／D／F／S／A：手算表
    rows = []
    T = {("t1", 0.0): {"raw": 10, "x@oneshot": 8, "y@oneshot": 9}, ("t1", 0.7): {"raw": 10, "x@oneshot": 12, "y@oneshot": 9},
         ("t2", 0.0): {"raw": 10, "x@oneshot": 5, "y@oneshot": 6}, ("t2", 0.7): {"raw": 10, "x@oneshot": 5, "y@oneshot": 6}}
    for (tier, d), v in T.items():
        for a_, tt in v.items():
            rows.append(dict(model="llama", tier=tier, duty=d, action=a_, T_s=float(tt), link_GiBps=1.0))
    dec = decide(rows, {"llama": {"raw", "x@oneshot", "y@oneshot"}})
    by = {(c["tier"], c["duty"]): c for c in dec["cells"]}
    # F：x 的幾何平均＝(8·12·5·5)^(1/4)=6.85；y＝(9·9·6·6)^(1/4)=7.35 → F＝x
    check("F＝幾何平均最小的固定動作", dec["F"]["llama"] == "x@oneshot", dec["F"])
    # S：t1 上 x＝√(8·12)=9.80、y＝9 → y；t2 上 x
    check("S＝每層幾何平均最小", dec["S"] == {"llama|t1": "y@oneshot", "llama|t2": "x@oneshot"}, dec["S"])
    c = by[("t1", 0.0)]
    check("A、對 S 的好處（t1 閒：x=8 vs S=y 9）", c["A"] == "x@oneshot" and abs(c["gain_vs_S"] - (1 - 8 / 9)) < 1e-12)
    c = by[("t1", 0.7)]
    check("對 F 的好處（t1 忙：A=y 9 vs F=x 12）", c["A"] == "y@oneshot" and abs(c["gain_vs_F"] - 0.25) < 1e-12)
    ag = aggregates(dec)
    # 只有 t1 閒那一格 A≠S（x 8 vs y 9，好 11%）；其他三格 A＝S
    check("A 對 S ≥5% 的格子比例＝1/4", abs(ag["frac_AS_ge5"] - 0.25) < 1e-12, ag["frac_AS_ge5"])
    want = 1 - gm([8 / 8, 9 / 12, 5 / 5, 5 / 5])
    check("A 對 F 的幾何平均好處", abs(ag["vs_F"]["gm_gain"] - want) < 1e-12)
    # 8. STOP-2 的分類（直接給聚合）
    check("STOP-2a：沒有任何格子 A 對 S ≥5%", stop2(dict(n_cells=4, vs_F=dict(gm_gain=0.3), frac_AS_ge5=0.0,
                                                    frac_AS_gt15=0.0))["code"] == "STOP-2a")
    check("STOP-2b：A 對 F <5%", stop2(dict(n_cells=4, vs_F=dict(gm_gain=0.04), frac_AS_ge5=0.5,
                                         frac_AS_gt15=0.5))["code"] == "STOP-2b")
    check("GO_N5：≥25% 格子 >15% 且 A 對 F >15%", stop2(dict(n_cells=4, vs_F=dict(gm_gain=0.2), frac_AS_ge5=0.5,
                                                         frac_AS_gt15=0.25))["code"] == "GO_N5")
    check("STOP-2c：A 對 F 在 5–15%", stop2(dict(n_cells=4, vs_F=dict(gm_gain=0.10), frac_AS_ge5=0.5,
                                               frac_AS_gt15=0.5))["code"] == "STOP-2c")
    check("STOP-2c：A 對 F >15% 但 >15% 的格子不到 25%", stop2(dict(n_cells=4, vs_F=dict(gm_gain=0.2), frac_AS_ge5=0.5,
                                                               frac_AS_gt15=0.2))["code"] == "STOP-2c")
    # 9. STOP-0：主格式 int4；另一格式只快 3% 不算翻轉、快 10% 算
    def mk(Ttab):
        rr = []
        for (tier, d), v in Ttab.items():
            for a_, tt in v.items():
                rr.append(dict(model="llama", tier=tier, duty=d, action=a_, T_s=float(tt), link_GiBps=""))
        return decide(rr, {"llama": {a for v in Ttab.values() for a in v}})
    d0 = mk({("a", 0.0): {"raw": 10, "int4_kivi@oneshot": 5, "fp8@oneshot": 6},
             ("b", 0.0): {"raw": 10, "int4_kivi@oneshot": 5, "fp8@oneshot": 6},
             ("c", 0.0): {"raw": 10, "int4_kivi@oneshot": 5, "fp8@oneshot": 4.86}})
    check("STOP-0 成立（另一格式只快 <5%）", stop0(d0)["holds"])
    d1 = mk({("a", 0.0): {"raw": 10, "int4_kivi@oneshot": 5, "fp8@oneshot": 6},
             ("b", 0.0): {"raw": 10, "int4_kivi@oneshot": 5, "fp8@oneshot": 6},
             ("c", 0.0): {"raw": 10, "int4_kivi@oneshot": 5, "fp8@oneshot": 4.5}})
    s0 = stop0(d1)
    check("STOP-0 不成立（另一格式快 10%）", not s0["holds"] and len(s0["flips"]) == 1, s0["flips"])
    # 10. STOP-1
    s1 = dict(run_id="S", models={m: {"T4": dict(dump=f"{m}_T4_64", best_ratio=1.10, best_ratio_key="all|planes|zstd:1",
                                                 max_decomp_GiBps=3.0, opts=[])} for m in MODELS})
    check("STOP-1：兩模型無損比都 <1.15 → 成立", stop1(s1)["holds"] is True and stop1(s1)["low_ratio"])
    s1["models"]["qwen"]["T4"]["best_ratio"] = 1.2
    check("STOP-1：一個模型 ≥1.15、解壓夠快 → 不成立", stop1(s1)["holds"] is False)
    for m in MODELS:
        s1["models"][m]["T4"]["max_decomp_GiBps"] = 0.3
    check("STOP-1：解壓 <0.5 GiB/s 又沒有 GPU 解碼器 → 成立", stop1(s1)["holds"] is True and stop1(s1)["slow_cpu_no_gpu"])
    check("STOP-1：S1 沒有 → NOT_MEASURED", stop1(None)["verdict"] == NM)
    # 11. 無損＋CPU 解的還原時間（三段重疊）
    a_ll = dict(name="ll:planes|zstd:1|t8", seg="bf16", dec="cpu", family="lossless", kind="none", r=1.25, D_Bps=4e9)
    r = restore_time(f_idle, 1.0, a_ll, slow, 0.0, Mm)
    cb = blk_costs(Mm, dict(seg="bf16", dec="none"), 0.0)
    ld = max(cb["bytes"] / (1.25 * 0.331 * GiB), cb["bytes"] / 4e9, cb["h2d"])
    check("無損：每塊＝max(壓縮後讀, CPU 解, H2D)", abs(r["T_la"] - 64 * ld) < 1e-9)
    # 12. vLLM 路徑：copy 次數 × c ＋位元組時間；c 由 c6 反推
    c0, why = vllm_copy_cost(Mm, C, None, 0.0)
    raw = blk_costs(Mm, dict(seg="bf16", dec="none"), 0.0)
    want_c = (0.97 - 4.0 * GiB / Mm["seg"]["bf16"]["bytes_blk"] * raw["h2d"]) / (32 * 2 * 2048)
    check("vLLM：c＝(c6 − 位元組時間)÷copy 次數", abs(c0 - want_c) < 1e-15, f"{c0:.3e}")
    check("vLLM：忙時沒有 I4 → None", vllm_copy_cost(Mm, C, None, 0.7)[0] is None)
    check("vLLM：忙時有 I4 → 乘倍數", abs(vllm_copy_cost(Mm, C, dict(k_busy2=1.4, run_id="I"), 0.7)[0] - 1.4 * c0) < 1e-18)
    rv = restore_time(f_idle, 1.0, dict(name="raw", seg="bf16", dec="none", family="raw", kind="none"),
                      dict(name="vllm_cpu", kind="vllm", GiBps=None), 0.0, Mm, dict(c=c0, L=32))
    check("vLLM：raw 的預測＝c6（建構上相等）", abs(rv["T"] - 0.97) < 1e-9, f"{rv['T']:.6f}")
    # 13. T-a／T-b／T-c／T-d
    def mkX(Ttab):
        rr = []
        for (tier, d, link), v in Ttab.items():
            for a_, tt in v.items():
                rr.append(dict(model="llama", tier=tier, duty=d, action=a_, T_s=float(tt), link_GiBps=link))
        return decide(rr, {"llama": {a for v in Ttab.values() for a in v}})
    dX = mkX({("cpu_pinned", 0.0, ""): {"raw": 10, "int4_kivi@oneshot": 5, "lrk24_v4@oneshot": 5.5},
              ("nfs_meas", 0.0, 0.383): {"raw": 10, "int4_kivi@oneshot": 5, "lrk24_v4@oneshot": 4.0}})
    sv = stacking_verdicts(dX, {})
    check("T-a 不成立（NFS 級組合快 20%）", sv["T_a"]["holds"] is False and sv["T_a"]["n_wins"] == 1)
    check("T-b 成立（只在 NFS 級贏、CPU 層輸）", sv["T_b"]["holds"] is True)
    dX2 = mkX({("cpu_pinned", 0.0, ""): {"raw": 10, "int4_kivi@oneshot": 5, "lrk24_v4@oneshot": 5.0}})
    check("T-a 成立（組合沒有快 ≥5%）", stacking_verdicts(dX2, {})["T_a"]["holds"] is True)
    spec = dict(t={("int4_kivi", "dq_oneshot"): 0.002, ("lr_r24", "dq_oneshot"): 0.004, ("int4_kivi", "h2d_stack"): 0.01,
                   ("lr_r24", "h2d_region"): 0.006},
                busy={("int4_kivi", "dq_oneshot"): {0.0: 1.0, 0.7: 1.1}, ("lr_r24", "dq_oneshot"): {0.0: 1.0, 0.7: 2.0},
                      ("lr_r24", "h2d_region"): {0.0: 1.0, 0.7: 1.5}})
    MmX = _synthetic_micro("llama", spec, duties=(0.0, 0.7))
    tcv = stacking_verdicts(dX2, {"llama": MmX})["T_c"]
    check("T-c 不成立（低秩忙/閒 2.0× vs INT4 1.1×）", tcv["holds"] is False)
    e = el3({"llama": MmX})["llama|lr_r24"]
    check("E-L3：忙/閒倍數 ≥1.5×INT4 → 新成本項", e["verdict"].startswith("新成本項"), f"{e}")
    spec["busy"][("lr_r24", "dq_oneshot")] = {0.0: 1.0, 0.7: 1.15}
    spec["t"][("lr_r24", "h2d_region")] = 0.03                  # 忙時重建 0.0046 s ÷ H2D 0.045 s＝0.10 < 0.25
    e = el3({"llama": _synthetic_micro("llama", spec, duties=(0.0, 0.7))})["llama|lr_r24"]
    check("E-L3：倍數差 <20% 且忙時重建 <25% H2D → 不另立項", e["verdict"].startswith("不另立項"), f"{e}")
    dX3 = mkX({("ssd", 0.0, 7.87): {"raw": 10, "a@oneshot": 5, "b@oneshot": 6},
               ("ssd", 0.7, 7.87): {"raw": 10, "a@oneshot": 9, "b@oneshot": 6}})
    td = stacking_verdicts(dX3, {})["T_d"]
    # S(ssd)：a＝√45=6.71、b＝6 → b；忙時 A＝b＝S；閒時 A＝a、比 S 快 1−5/6＝16.7% ≥15% → 成立
    check("T-d：隨 duty 換人且比 S 快 ≥15% → 成立", td["holds"] is True and len(td["duty_flip"]) == 1, td["duty_flip"])
    # 14. select_flips：只挑 GO_N5 的模式、vllm 不挑、最多 8 格、每格量 A／S／F／D
    decs = {"L": dict(cells=[]), "Q": dX3, "X": dict(cells=[])}
    verd = {"L": dict(stop2=dict(code="STOP-2a")), "Q": dict(stop2=dict(code="GO_N5")), "X": dict(stop2=dict(code="TIME_ONLY"), x2=False)}
    modes, fl = select_flips(decs, verd)
    roles = {r for x in fl[0]["measure"] for r in x["role"].split("+")} if fl else set()
    check("select_flips：Q 的 GO_N5 → 挑格子、A／S／F／D 都有人量", modes == ["Q"] and fl and
          roles == {"A", "S", "F", "D"} and fl[0]["A"] == "a@oneshot", fl[:1])
    check("select_flips：同一個動作身兼數職只量一次", all(len({x["action"] for x in c["measure"]}) == len(c["measure"])
                                                       for c in fl))
    verd["Q"]["stop2"]["code"] = "STOP-2c"
    check("select_flips：沒有 GO_N5 → 不挑", select_flips(decs, verd)[1] == [])
    # 15. STOP-3
    cells = [dict(cell="c1", A="a", F="b"), dict(cell="c2", A="a", F="b")]
    vr = [dict(cell="c1", role="A", how="harness", pred_s=1.0, meas_s=1.05, rel_err=0.05, contaminated=False),
          dict(cell="c1", role="F+S", how="harness", pred_s=1.3, meas_s=1.3, rel_err=0.0, contaminated=False),
          dict(cell="c2", role="A", how="harness", pred_s=1.0, meas_s=1.2, rel_err=0.2, contaminated=False),
          dict(cell="c2", role="F", how="harness", pred_s=1.3, meas_s=1.6, rel_err=0.23, contaminated=False)]
    s3 = stop3(vr, cells)
    check("STOP-3：只有一格差 >10%、A 對 F 實測好 >5% → 不成立", not s3["holds"] and s3["bad_cells"] == ["c2"]
          and s3["n_AF_pairs"] == 2, s3)
    vr[0]["meas_s"], vr[0]["rel_err"] = 1.25, 0.25
    s3 = stop3(vr, cells)
    check("STOP-3：兩格差 >10% → 成立", s3["holds"] and len(s3["bad_cells"]) == 2)
    vr2 = [dict(r, contaminated=True) if r["cell"] == "c2" else r for r in vr]
    check("STOP-3：污染列不算", stop3(vr2, cells)["bad_cells"] == ["c1"])
    vr3 = [dict(r, meas_s=1.0, rel_err=0.0) for r in vr]
    check("STOP-3：實測 A 對 F 好不到 5% → 成立", stop3(vr3, cells)["holds"] and stop3(vr3, cells)["gain_AF_measured"] == 0.0)
    # 16. 規則 6：micro 的 BF16 位元組和 bw.json 不同要丟例外
    with tempfile.TemporaryDirectory() as d:
        Cb = _synthetic_consts()
        Cb.v[("llama", "chunk_bytes")] = 123
        try:
            run_map(d, d, models=("llama",), C=Cb, M={"llama": Mm}, paths=dict(s1="", x2="", i4="", q4=d, q4duty=d))
            raised = False
        except ValueError:
            raised = True
        check("規則 6：BF16 位元組對不上 → 丟例外", raised)
    # 17. 整條 map（合成：INT4 每格都最快 → STOP-0 成立、validate 不跑）
    sp = {("bf16", "h2d_region"): 0.040, ("int4_kivi", "h2d_stack"): 0.0125, ("int4_kivi", "dq_oneshot"): 0.0016,
          ("int4_kivi", "dq_perblock"): 0.005, ("int8_token", "h2d_stack"): 0.021, ("int8_token", "dq_oneshot"): 0.002,
          ("int8_token", "dq_perblock"): 0.006, ("fp8", "h2d_stack"): 0.020, ("fp8", "dq_oneshot"): 0.0018,
          ("fp8", "dq_perblock"): 0.005, ("lr_r24", "h2d_region"): 0.0075, ("lr_r24", "dq_oneshot"): 0.004,
          ("lr_r24", "dq_perblock"): 0.012}
    Ms = {m: _synthetic_micro(m, dict(t=sp), rec={0.9: dict(mult=2.8, n_pos=9), 0.4: dict(mult=1.5, n_pos=9),
                                                   0.7: dict(mult=2.1, n_pos=9)}) for m in MODELS}
    Cs = _synthetic_consts(MODELS)
    with tempfile.TemporaryDirectory() as d:
        s = run_map(d, d, models=MODELS, C=Cs, M=Ms, paths=dict(s1="", x2="", i4="", q4=d, q4duty=d))
        files = ["map_cells.csv", "map_decision.csv", "map_summary.json", "flip_cells.json", "constants_used.csv",
                 "map_report.md", "decision_Q_llama.svg", "decision_grid_X_qwen.svg"]
        check("map：輸出檔都在", all(os.path.exists(os.path.join(d, f)) for f in files),
              [f for f in files if not os.path.exists(os.path.join(d, f))])
        check("map：模式 Q 的 STOP-0 成立（合成：INT4 每格最快）", s["verdicts"]["Q"]["stop0"]["holds"],
              s["verdicts"]["Q"]["stop0"]["flips"][:2])
        check("map：沒有 S1 → 模式 L NOT_MEASURED", s["verdicts"]["L"]["stop2"]["code"] == NM)
        check("map：沒有 X2 → 模式 X 只看時間", s["verdicts"]["X"]["stop2"]["code"] == "TIME_ONLY")
        check("map：validate 不跑、有原因", s["worth_validate"] is False and "GO_N5" in s["validate_reason"],
              s["validate_reason"])
        rr = read_csv(os.path.join(d, "map_cells.csv"))
        check("map：CSV 有 run_id、ts 欄", rr and all(r["run_id"] and r["ts"] for r in rr))
        dd = read_csv(os.path.join(d, "map_decision.csv"))
        check("map：決策 CSV 有 run_id、ts、micro_run_id", dd and all(r["run_id"] and r["ts"] and r["micro_run_id"] for r in dd))
        check("map：vllm_cpu 在忙的格子沒有 I4 → 沒有值（不是 0）",
              all(r["T_s"] in ("", "None") for r in rr if r["tier"] == "vllm_cpu" and float(r["duty"]) > 0))
    # 17b. 有翻轉的合成：INT4 一次拆在忙時被拖慢 ×40（別的格式不受影響）→ 忙時換人；判定和挑格子要一致
    spf = dict(sp)
    busyf = {k: {d: 1.0 for d in (0.0, 0.4, 0.7, 0.9)} for k in spf}
    busyf[("int4_kivi", "dq_oneshot")] = {0.0: 1.0, 0.4: 40.0, 0.7: 40.0, 0.9: 40.0}
    busyf[("int4_kivi", "dq_perblock")] = {0.0: 1.0, 0.4: 40.0, 0.7: 40.0, 0.9: 40.0}
    Mf = {m: _synthetic_micro(m, dict(t=spf, busy=busyf), rec={0.9: dict(mult=2.8, n_pos=9), 0.4: dict(mult=1.5, n_pos=9),
                                                               0.7: dict(mult=2.1, n_pos=9)}) for m in MODELS}
    with tempfile.TemporaryDirectory() as d:
        s = run_map(d, d, models=MODELS, C=Cs, M=Mf, paths=dict(s1="", x2="", i4="", q4=d, q4duty=d))
        q = s["verdicts"]["Q"]
        fl = json.load(open(os.path.join(d, "flip_cells.json")))
        check("翻轉合成：STOP-0 不成立", not q["stop0"]["holds"], f"{len(q['stop0']['flips'])} 格翻轉")
        go = any(s["verdicts"][md]["stop2"].get("code") == "GO_N5" for md in ("L", "Q"))
        check("翻轉合成：worth_validate 和 GO_N5 一致、格子 ≤8、不挑 vllm、每格都量 D",
              fl["worth_validate"] == go and len(fl["cells"]) <= 8 and all(c["tier"] != "vllm_cpu" for c in fl["cells"])
              and all(any("D" in x["role"].split("+") for x in c["measure"]) for c in fl["cells"]),
              f"Q：{q['stop2'].get('code')}，{len(fl['cells'])} 格")
        import xml.etree.ElementTree as ET
        ET.parse(os.path.join(d, "decision_Q_llama.svg"))
        check("決策圖 SVG 是合法的 XML", True)
    # 17d. 沒有 micro（或全被污染）→ 全部 NOT_MEASURED，不是「成立」
    with tempfile.TemporaryDirectory() as d:
        s = run_map(d, d, models=MODELS, C=Cs, M={m: None for m in MODELS}, paths=dict(s1="", x2="", i4="", q4=d, q4duty=d))
        check("沒有 micro → STOP-0 是 NOT_MEASURED、validate 不跑", s["verdicts"]["Q"]["stop0"]["holds"] is None
              and s["verdicts"]["Q"]["stop2"]["verdict"] == NM and s["worth_validate"] is False)
    # 17c. S1、X2 的讀檔約定
    with tempfile.TemporaryDirectory() as d:
        S = dict(run_id="S1RID", dumps={
            "llama_T4_64": dict(meta=dict(model="llama", task="T4", n_tokens=32768),
                                bf16={"all|planes|zstd:1": dict(ratio=1.21), "all|raw|zstd:1": dict(ratio=1.02),
                                      "all|entropy": dict(h_exp=2.0)},
                                speed={"planes|zstd:1|t4": dict(decomp_MBps=3000.0, ratio_framed=1.2),
                                       "planes|zstd:1|t32": dict(decomp_MBps=9000.0, ratio_framed=1.2)}),
            "llama_T4_252": dict(meta=dict(model="llama", task="T4", n_tokens=129024), bf16={}, speed={})})
        json.dump(S, open(os.path.join(d, "summary.json"), "w"))
        s1r = load_s1(os.path.join(d, "summary.json"))
        o = s1r["models"]["llama"]["T4"]
        check("load_s1：只取 32K 的 T4、≤16 執行緒、最大無損比", o["dump"] == "llama_T4_64" and abs(o["best_ratio"] - 1.21) < 1e-12
              and [x["name"] for x in o["opts"]] == ["ll:planes|zstd:1|t4"] and abs(o["opts"][0]["D_Bps"] - 3e9) < 1)
        xp = os.path.join(d, "x2.csv")
        write_csv(xp, [dict(model="llama", x3_segment="lr_r24", feasible="1", run_id="X2R"),
                       dict(model="qwen", x3_segment="lr_r24", feasible="0", run_id="X2R")],
                  ["model", "x3_segment", "feasible", "run_id"])
        x2r = load_x2(xp)
        fs = feasible_sets(Ms, None, x2r)
        check("load_x2：可行的 segment 進模式 X（只有 Llama）", "lr_r24@oneshot" in fs["X"]["llama"]
              and "lr_r24@oneshot" not in fs["X"]["qwen"] and "lr_r24@oneshot" not in fs["Q"]["llama"])
        write_csv(xp, [dict(model="llama", seg="lr_r24", ok="1")], ["model", "seg", "ok"])
        try:
            load_x2(xp)
            raised = False
        except ValueError:
            raised = True
        check("load_x2：欄位不符約定 → 丟例外（不當成沒有）", raised)
        write_csv(xp, [dict(model="llama", x3_segment="pca_r24", feasible="1", run_id="X")],
                  ["model", "x3_segment", "feasible", "run_id"])
        try:
            load_x2(xp)
            raised = False
        except ValueError:
            raised = True
        check("load_x2：不認得的 segment 名 → 丟例外", raised)
    # 18. 真實常數檔讀得到、每個都有出處（只讀檔）
    try:
        Cr = load_constants()
        need = [("llama", "f_idle_s"), ("qwen", "f_idle_s"), ("llama", "link_b32"), ("-", "link_ssd_meas"),
                ("-", "link_nfs_param"), ("-", "duty_busy3"), ("llama", "vllm_c6_T_s"), ("qwen", "mult_f_stage2_d0.7")]
        miss = [k for k in need if Cr.get(*k) is None]
        noprov = [r for r in Cr.rows if r["value"] != NM and (not r["file"] or not r["run_id"])]
        check("真實常數：讀得到、都有檔名與 run_id", not miss and not noprov, f"缺 {miss}；沒出處 {noprov[:2]}")
        check("真實常數：SSD 實測中位數 7.86 GiB/s 左右（8053.8 MiB/s，calib_c2）",
              abs(Cr.get("-", "link_ssd_meas") - 8053.8 / 1024) < 1e-9)
        check("真實常數：duty 0.9 讀自 busy_levels_q4.json", Cr.get("-", "duty_busy3") == 0.9)
    except Exception as ex:
        check("真實常數檔", False, repr(ex))
    try:
        an = anchors({"llama": Ms["llama"]}, Cs, {})
        check("第二階段錨點讀得到（q4／q4duty 的 restores.csv）", any(x["stage2"].startswith("q4/llama_int4_P3") for x in an)
              and all(x["run_id"] for x in an), f"{len(an)} 列")
    except Exception as ex:
        check("第二階段錨點", False, repr(ex))
    check("seg_catalog：組合都標成 combo、ops 和 feed 一致", all(
        (v["kind"] == "combo") == (k in COMBO_SEGS) and set(v["feed"].values()) <= set(v["ops"]) for k, v in seg_catalog().items()))
    _self_test_x2gap(check)
    check("整個自測沒有 import torch", "torch" not in sys.modules)
    print(f"{n_chk} 項；" + ("ALL PASS" if ok else "SOME FAILED"), flush=True)
    if out:
        json.dump(dict(n_checks=n_chk, all_pass=ok, run_id=run_id(), ts=now_iso()), open(out, "w"), indent=1)
    return ok


# ====================================================================== main
def main():
    ap = argparse.ArgumentParser(description="X3：壓縮組合的還原時間 × 連線速度 × GPU 忙")
    ap.add_argument("--self-test", action="store_true", help="只用 CPU、不 import torch")
    ap.add_argument("--smoke", action="store_true", help="GPU 冒煙：每個解碼段最小設定跑一次＋map／validate 管線")
    ap.add_argument("--out", help="--smoke 的輸出目錄；--self-test 的結果 JSON")
    ap.add_argument("--models", nargs="+", default=list(MODELS), help="--smoke 用哪些模型")
    ap.add_argument("--store-impl", default="triton", choices=["triton", "torch"], help="--smoke 的低秩收尾寫法")
    ap.add_argument("--segments", nargs="+", default=None, help="--smoke 只冒煙這些 segment（＋bf16；10/21 補缺口用）")
    ap.add_argument("--wcost", nargs="*", default=None, help="--smoke 也冒煙 xKV 寫入成本（k22v34 …，或 x2）")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("micro", help="GPU：各解碼段的時間 × duty")
    p.add_argument("--model", nargs="+", choices=list(MODELS), required=True)
    p.add_argument("--out", default=RES)
    p.add_argument("--n-blocks", type=int, default=32, help="載入區 block 數（預設 32＝block 32–63）")
    p.add_argument("--gpu-states", nargs="+", default=list(GPU_STATES))
    p.add_argument("--busy-levels", default=BUSY_LEVELS)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--min-rep-s", type=float, default=0.1)
    p.add_argument("--segments", nargs="*", default=None, help="只量這些 segment（預設全部）")
    p.add_argument("--bg", nargs="*", default=None, help="量背景拖慢的 seg:op（預設 BG_DEFAULT）")
    p.add_argument("--bg-seconds", type=float, default=2.0)
    p.add_argument("--no-recompute", action="store_true", help="不載模型、不量重算慢化")
    p.add_argument("--rec-reps", type=int, default=3)
    p.add_argument("--store-impl", default="triton", choices=["triton", "torch"])
    p.add_argument("--wcost", nargs="*", default=None,
                   help="10/21：xKV 寫入成本的秩組合（k22v34 …，或 x2＝10/21 的 xKV 段全部）；不給就不量")
    p.add_argument("--wcost-states", nargs="+", default=list(WCOST_STATES))
    p.add_argument("--wcost-reps", type=int, default=3)
    p = sub.add_parser("map", help="CPU：決策圖與判定")
    p.add_argument("--micro-dir", nargs="+", default=[RES],
                   help="micro 的輸出目錄；可以給多個（第一個為主，後面的只補沒有的 segment，重疊的當漂移錨點）")
    p.add_argument("--out", default=RES)
    p.add_argument("--s1", default=None)
    p.add_argument("--x2", default=None, help="X2 的 x2_feasible.csv（預設 results/m16_stage3_mi300x/x_stack/x2_feasible.csv；給了但不存在 → 停）")
    p.add_argument("--i4", default=None)
    p = sub.add_parser("validate", help="GPU：map 挑的格子做端到端實測")
    p.add_argument("--map-dir", default=RES)
    p.add_argument("--out", default=os.path.join(RES, "validate"))
    p.add_argument("--force", action="store_true", help="map 判不值得驗證也跑（只給冒煙用）")
    p.add_argument("--max-cells", type=int, default=8)
    p.add_argument("--launches", type=int, default=2)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--lp-reps", type=int, default=10)
    p.add_argument("--main-length", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test(a.out) else 1)
    if a.smoke:
        if not a.out:
            ap.error("--smoke 要 --out")
        sys.exit(cmd_smoke(a))
    if a.cmd == "micro":
        cmd_micro(a)
    elif a.cmd == "map":
        cmd_map(a)
    elif a.cmd == "validate":
        sys.exit(cmd_validate(a))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
