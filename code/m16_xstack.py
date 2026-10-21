"""m16_xstack.py — 第三階段「KV 壓縮疊加」：X1（teacher forcing 大量篩選）與 X2（挑出的組合量生成答對率）。

設計來源（寫死，不改）：docs/stage3_20261019/lit/L5d_stacking.md（三階段設計、停止條件、位元組照實算、干擾機制）、
lit/L5a_lowrank.md（E-L1／E-L2：pre-RoPE、去平均、校準 PCA、P40／K24V4／P24）、lit/L5b_merge_crosslayer.md（TM-g＝AsymKV 等價式、
MiniCache 式、xKV 每 4 層聯合 SVD、KVSharer 式）、lit/L6_mainstream_practice.md §7（常見部署對照、MLPerf 門檻）。
怎麼跑、停止條件原文、時間估計：docs/stage3_20261019/x_stack_howto.md。組合清單：code/m16_xstack_combos.json。

## 壓縮階段（GPU，torch；每一階段照實算位元組，和 m10_codec_x 的 Enc.nbytes 同一個慣例）
  drop:<方法>:<比例>  block 粒度（512 token）。順序＝m10_drop.unit_order(cx, "B", 方法)（DS1／S2 同一個「先丟」順序，全段算、
                      只留區內）；丟 round(比例×區內 block 數) 個，用 KVModel.key_keep 遮掉。位元組＝留下的 token＋每 block 1 bit 的遮罩。
  merge:<g>           TM-g：block 內每 g 個相鄰 token，key 換成組平均（post-RoPE），value 存組平均（＝總和÷g；g 固定，計數不必存）。
                      AsymKV 式 (8)–(10)：「g 個槽都放 k*、value 原樣」和「存 1 個 k*＋value 總和＋計數」attention 輸出相同；
                      所以存的形式＝每組 1 個 key＋1 個 value（位元組 1/g），解碼＝每組展開 g 份（key＝k*，value＝組平均）。
  lr:k<r>[:v<r>][:joint]  校準 PCA（calib 子命令；每層每 KV head，或每層跨 head 聯合）。K 用模型自己的 cos/sin 逆旋轉成 pre-RoPE、
                      去平均（校準的平均），投到前 r 個主成分；重建後再用模型自己的 apply_rope 轉回。V 一樣（不用 RoPE）。
                      r＝0 的那一半不降秩、保持原樣。位元組＝係數（BF16 或 q 量化後）；基底＋平均跨請求共用，另列 bytes_shared。
  xl:k<r>[:v<r>][:w<W>]   xKV 式：每題、每 W（預設 4）個相鄰層聯合 SVD（K 先逆 RoPE，V 直接；不去平均＝照 xKV），
                      秩＝r×W×H（r 是「每 head 每層等效」秩，兩個模型通用）。torch.svd_lowrank（固定 seed、niter 4）。
                      位元組＝每個 token 的係數＋每個窗口的基底 B（r×W·H·D，BF16，每題一份、算在這個請求裡）。
  mc[:t<t>][:g<γ>]    MiniCache 式（只當對照）：深半段相鄰兩層同一 token、同一 head 的向量 SLERP（t＝0.6）成一個方向、各自存範數，
                      角距離最大的 γ＝5% token 原樣保留（照 MiniCache 正文語意）＋索引。
  share               KVSharer 式（只當對照）：深半段奇數層直接用前一層的 KV（位元組 0.75）。
  q:<格式>            量化。只有原始 KV（沒有 lr／xl）：每個存下來的 block（merge 後是 512/g 個 token）照 m10_codec_x.encode／decode
                      （第二階段同一段，逐位元相同；fp8k0_* 等整塊格式也可以）。有 lr／xl：係數用 m10_codec_x.enc_half／dec_half
                      （KIVI 式 K 角色＝每個主成分各自 scale、沿 token 每 32 個一組；只允許 bf16／int8_token／int4_token／int4_kivi／int2_kivi），
                      沒降秩的那一半照原始 KV 的半邊格式逐 block 編碼。
                      q:vllm_fp8＝模擬 vLLM 0.28 在 ROCm 上未校準的 fp8 KV（per-tensor、scale＝1.0、e4m3fnuz、最大 240；
                      超過的截在 ±240——ROCm kernel 是截還是溢位 NOT_VERIFIED，這裡假設截）。m10 沒有這個格式，在這裡自己實作。
  無損（zstd）        不在 GPU 迴圈裡跑：report 用 S1 的實測壓縮比（results/m16_stage3_mi300x/s1_lossless/summary.json）當乘數另列。
  KIVI 殘差窗口：m10 的 KIVI 沒有全精度殘差窗口（context 全部量化），這裡也沒有（位元組 0）。sink：block 0 一律 BF16（區外）。

組合語法：階段用 + 串起來、照套用順序；合法順序 drop? merge? (lr|xl)? q?，或 xl+drop(+q)（基底在全部 token 上估的順序對照）。
擋掉：drop 在 merge／lr 之後、merge 和 lr／xl 一起、lr 和 xl 一起、q 不在最後、mc／share 和別的一起、係數用不相容的格式、秩不合。

## 子命令
  calib         校準 PCA 基底與平均（題目和評測題不重疊：seed 61000 起），存檔（附 run_id）；先做 RoPE 逆旋轉的資料自身核對（規則 6）
  screen        X1：每題 prefill 一次；先跑 S2 的原本各組（直接呼叫 m16_s2_equalbytes.run_item，S2＝X1 的子集，逐列另存成 S2 的格式），
                再跑組合清單；每列記 TF 指標（kl、gold_lp、top1、first_ok；和 S2／DS1 同定義）、位元組、代理指標
                （K／V 相對重建誤差、用第 1 輪問題的 query 算的 attention 輸出誤差）。逐題續跑、--shard、--budget-min、GpuWatcher 品質規則。
  report        CPU：Pareto 前緣、各家族最好的單一方法、代理的 Spearman／崩潰 AUROC、L5d 停止條件、x2_selection.json
  gen           X2：讀 x2_selection.json，用 m10_sweep.Item.gen() 量生成答對率（第二階段同一個計分）；--ctx 32K／128K（44c 的題組）
  gen-report    CPU：X2 的 M1／M2／完整性判定（L5d §6.3），另報 MLPerf 式 99%／99.9% 門檻（L6；不改判定）；
                另寫 X3（m16_xtime.load_x2）要的 x2_feasible.csv（model、x3_segment、feasible＝ε 判定、run_id）與對不到的 x2_unmapped.csv
  gpu-selftest  冒煙時在 GPU 上跑：torch 實作 vs numpy 參考、量化和 m10_codec_x 逐位元相同、vllm_fp8 對 numpy 的 fnuz 格點、RoPE 核對
  --self-test   只用 CPU、numpy／純 Python（不 import torch）：語法、位元組、merge／lowrank／xlayer／mc 的數學、停止條件與挑選邏輯

## 10/21 延伸（x_stack_howto.md §14；舊子命令的行為不變，只新增）
  xl:k<r>:v<r>:cal   xKV、事先校準的基底：每 4 層、K（逆 RoPE）／V 的基底用校準題（seed 61000 起 8 題）事先算好、所有題共用
                     （calib-xl 子命令：不去平均的 ΣZᵀZ 的前 R 個特徵向量）；係數＝Z·U、重建用 BF16 基底。位元組帳不含每題基底（另列 shared）。
  X3 對應            X3_MAP_1021：xl:k11:v17／k22:v34／k28:v42（BF16、+int4_kivi）、xl:k22:v34+q:int8_token → xkv_k*（X3 同名段）
  calib-xl      GPU：xKV 校準基底（存到每 head 每層等效秩 64）
  plan          CPU：X2b／xKV 校準生成的計畫檔（gen 的 --selection；roles／budgets 照第一版 x2_selection.json）
  pending       CPU：工作池用，印某個 shard 還有幾題沒做
  gen           新旗子 --calib-xl（:cal 的配置）、--xl-cache（同題同秩的每題 SVD 只算一次；預設關＝照舊）
  screen        新旗子 --calib-xl
  gpu-selftest  新旗子 --new-checks x2b|xlcal|both（只跑新檢查）、--calib-xl
  x2b-report    CPU：X2＋X2b 合併（A 只用 X2／B 只用 X2 限制池子／C 只用 X2b／D 合併）、M1／M2／MLPerf 照第一版規則、v2 的 x2_feasible.csv
  xlcal-report  CPU：xKV 校準基底 vs 每題 SVD（TF 只報、生成判定事先寫死：XLCAL_TEXT）
  兩卡共用工作池在 code/m16_queue/jobs/x_pool.sh（bash）；--self-test 用假的 m7run 驗每個單位恰好被領一次。

模組頂層只有標準函式庫與 m16_c1a_predict（同樣沒有副作用）的 import；torch、numpy、m10_* 一律在函式裡才 import。
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m16_c1a_predict import (CHUNK, N_BLOCKS, PARAMS, REPO, done_items, finish_watch, host_level, intruders,  # noqa: E402
                             item_id_of, open_out, parse_shard, read_rows, side_path, start_watch, ts)

OUT_DIR = os.path.join(REPO, "results", "m16_stage3_mi300x", "x_stack")
DATA = os.environ.get("X_DATA", "/mlsteam/data/tiara/stage3/xstack")        # 逐列原始輸出、校準基底（不進 git）
COMBOS = os.path.join(HERE, "m16_xstack_combos.json")
S1_SUMMARY = os.path.join(REPO, "results", "m16_stage3_mi300x", "s1_lossless", "summary.json")
LC44C = "/mlsteam/data/tiara/longctx_m10/screen_{model}_252_s*.csv.gz"      # 第二階段 44c 的 128K 逐列（重現檢查）

# 規則 6：形狀在 GPU 行程裡一律和 km.cfg 核對（check_info）
MODEL_INFO = {"llama": dict(L=32, H=8, Hq=32, D=128), "qwen": dict(L=28, H=4, Hq=28, D=128)}
BUDGETS = (0.3125, 0.25, 0.125, 0.0625)        # 區內位元組比（對 BF16）；0.3125＝INT4 KIVI 錨點，其餘＝4×／8×／16×（L5d §6.0）
BUDGET_TOL = 1.05                                # 「在預算內」＝位元組比 ≤ 1.05×預算（L5d §6.2「±5%」）
EPS = 0.02                                       # 第二階段的 ε（2 個百分點）
MLPERF = (0.99, 0.999)                           # L6 §5.2：MLPerf 門檻（答對率 ≥ E0 的 99%／99.9%）；只報，不改判定
COLLAPSE_FIRST_OK = -0.05                        # L5d §6.2：「崩」＝對 E0 差 ≤ −5 個百分點（TF 用 first_ok）
LOSS_RATIO = 0.95                                # L5d S1-b：組合的損失 ≥ 0.95×單一方法 → 沒有加乘
R_NOGO, R_GO = 0.95, 0.85                        # L5d M1
SPEARMAN_OK = 0.5                                # L5d S1-a
HOLM_ALPHA = 0.025                               # 單邊（＝雙邊 0.05；和 DS2 相同）
K_FRONT, MAX_SAME_SIG = 4, 2                     # 每個預算前緣挑 4 個、同一種家族組合最多 2 個
MIN_ITEMS = 48                                   # X1 每個模型至少要有這麼多題做完才挑 X2（不然 go＝false）
X1_SEED0, CALIB_SEED0, X2_SEED0 = 60000, 61000, 62000
LC_SEED0, LC_N, LC_BLOCKS = 40000, 12, 252       # 44c：m10_sweep screen 預設 seed0 40000、--n 12、--n-blocks 252（run 的 cmd.sh）
XL_NITER, XL_OVERSAMPLE, XL_SEED = 4, 16, 20261020
N_BOOT = 10000
# 10/21 延伸（x_stack_howto.md §14）：X2b 加題數、xKV 事先校準的基底。題目的 seed 互不重疊（--self-test 核對）：
# X1 60000+、校準 61000+（PCA 與 xKV 校準同一套 8 題）、X2 62000+、X2b 63000+；xKV 校準的 TF＝X1 的前 12 題／任務、生成＝X2 的 96 題。
OUT_DIR_V2 = os.path.join(OUT_DIR, "v2")                 # 10/21 延伸的摘要（不覆蓋第一版的檔）
COMBOS_V2 = os.path.join(HERE, "m16_xstack_combos_v2.json")
X2B_SEED0, X2B_N = 63000, 48                             # X2b：T2–T5 各 48 題＝192 題
XLCAL_TF_N, XLCAL_GEN_N = 12, 24                         # xKV 校準：TF 48 題（X1 的題）、生成 96 題（X2 的題）
XLCAL_RANK_MAX = 64                                      # 校準基底存到「每 head 每層等效秩」64（Llama 2048／4096 維、Qwen 1024／2048 維）

DROP_METHODS = ("KEYDIFF", "KNORM", "SNAP", "TOVA", "H2O", "EXPATT", "QRY", "STREAM", "NEW", "RND0", "RND1", "RND2", "ORC")
NOT_DEPLOYABLE = ("QRY", "ORC")                  # 要知道問題／答案位置 → 不是寫入時方法
MERGE_G = (2, 4, 8, 16)
M10_FULL = ("fp8_head", "fp8_channel", "fp8_token", "fp8k0_head", "fp8k0_channel", "fp8k0_token", "int8_token", "int4_token",
            "int4_kivi")                         # m10_codec.FORMATS（除 bf16）
NEW_FULL = ("fp8e5_head", "fp8e5_token", "mxfp4", "nvfp4", "int2_kivi", "int4h_token", "int2h_g32")   # m10_codec_x.NEW_FULL
HALF = ("bf16", "int8_token", "int4_token", "fp8_head", "fp8_token", "fp8e5_head", "fp8e5_token", "mxfp4", "nvfp4", "kivi4",
        "kivi2", "int4h_token", "int2h_g32")     # m10_codec_x.HALF
ALIAS = {"int4_kivi": "kivi4", "int2_kivi": "kivi2"}
COEF_HALF = ("bf16", "int8_token", "int4_token", "kivi4", "kivi2")   # 對任意 [T, r] 都和 block 結構無關的格式
OWN_FMTS = ("vllm_fp8",)
FNUZ_MAX = 240.0                                 # float8_e4m3fnuz 的最大值
FAMILY = {"drop": "drop", "merge": "merge", "lr": "lowrank", "xl": "xlayer", "q": "quant", "mc": "xshare", "share": "xshare"}
ALLOWED_SEQ = [(), ("drop",), ("merge",), ("lr",), ("xl",), ("drop", "merge"), ("drop", "lr"), ("drop", "xl"), ("xl", "drop")]
REGIONS = {"full": None, "region": (32, 63)}     # full＝block 1..最後；region＝S2 的載入區（只在 32K）
PROXIES = ("relerr_k", "relerr_v", "cenerr_k", "attn_err", "attn_err_max", "attn_err_wo")

# X3（m16_xtime.py seg_catalog）的解碼段 ↔ 我們的組合（逐一核對過 X3 的實作：lr_r<r>＝K、V 同秩、每 head 校準基底、BF16 係數；
# xkv_g4_f<f>＝每 4 層、K、V 同秩 W·H·D/f；merge_g<g>＝存 T/g、展開；lrk24_v4＝K PCA r24 BF16＋V int4_kivi；fp8＝fmt_choice 的粒度）。
# 對不到的一律不硬對，寫進 x2_unmapped.csv（原因見 x3_segment）。
X3_MAP = {"none": "bf16", "q:int8_token": "int8_token", "q:int4_kivi": "int4_kivi", "q:int2_kivi": "int2_kivi",
          "q:kv-int4_kivi-int2_kivi": "k4v2", "lr:k16:v16": "lr_r16", "lr:k24:v24": "lr_r24", "lr:k32:v32": "lr_r32",
          "lr:k40:v40": "lr_r40", "lr:k64:v64": "lr_r64", "xl:k16:v16": "xkv_g4_f8", "xl:k32:v32": "xkv_g4_f4",
          "merge:2": "merge_g2", "merge:4": "merge_g4", "lr:k24+q:kv-bf16-int4_kivi": "lrk24_v4",
          "merge:2+q:int4_kivi": "merge_g2_int4"}
X3_FP8 = {"llama": "q:fp8_token", "qwen": "q:fp8_head"}           # results/m10_precision_mi300x/fmt_select/fmt_choice.json
X3_SEGMENTS = ("bf16", "fp8", "int8_token", "int4_kivi", "int2_kivi", "k4v2", "lr_r16", "lr_r24", "lr_r32", "lr_r40", "lr_r64",
               "xkv_g4_f8", "xkv_g4_f4", "merge_g2", "merge_g4", "droptok_k50", "droptok_k25", "lrk24_v4", "droptok_k50_int4",
               "merge_g2_int4")
# 10/21 延伸：X3 另一個 agent 加的同名 xKV 解碼段（K、V 不同秩＝照 xKV 的 1:1.5、每 4 層聯合、每題基底算進要搬的位元組；
# _int4＝係數用 KIVI int4（K 角色）、_int8＝係數用 int8_token）。名字由協調者給定；X3 的 seg_catalog 有沒有加好，
# 由 38bz 的 load_x2 核對（沒有就 NEEDS_HUMAN，規則 7），--self-test 只印 INFO。
X3_MAP_1021 = {"xl:k11:v17": "xkv_k11v17", "xl:k22:v34": "xkv_k22v34", "xl:k28:v42": "xkv_k28v42",
               "xl:k11:v17+q:int4_kivi": "xkv_k11v17_int4", "xl:k22:v34+q:int4_kivi": "xkv_k22v34_int4",
               "xl:k28:v42+q:int4_kivi": "xkv_k28v42_int4", "xl:k22:v34+q:int8_token": "xkv_k22v34_int8"}
X3_MAP.update(X3_MAP_1021)
X3_SEGMENTS = X3_SEGMENTS + tuple(X3_MAP_1021.values())


def x3_segment(spec: str, model: str):
    """我們的組合 → X3 的解碼段名稱；對不到 → (None, 原因)。"""
    if spec == X3_FP8.get(model):
        return "fp8", ""
    if spec in X3_MAP:
        return X3_MAP[spec], ""
    st = parse_spec(spec)
    ops = [s["op"] for s in st]
    if any(s.get("cal") for s in st):
        return None, ("xKV 校準基底版（:cal）：基底跨請求共用、常駐，不用每題搬；X3 的 xkv_* 段把每題基底算進要搬的位元組 → "
                      "位元組與搬運都不同，不硬對")
    if "drop" in ops:
        why = ("我們丟的是 512-token block（所有層、head 一起，key_keep 遮掉；沒有逐 token 位置、不需要 scatter），"
               "X3 的 droptok_k* 是每（層, head）各自留 token＋int16 位置；位元組與解碼方式都不同")
        if len(ops) > 1:
            why += "；其餘階段等於『只搬留下的 block』的對應解碼段，X3 若加 block 數縮放才對得上"
        return None, why
    if "xl" in ops:
        return None, ("這個 xKV 的秩組合或係數量化 X3 沒有解碼段（X3 有 K、V 同秩 BF16 的 xkv_g4_f8／f4，與 10/21 加的 "
                      "xkv_k11v17／k22v34／k28v42（BF16、_int4）、xkv_k22v34_int8）")
    if "lr" in ops:
        return None, "低秩的秩組合、聯合 PCA 或係數量化不在 X3 的 lr_r16–64／lrk24_v4 裡"
    if "merge" in ops:
        return None, "合併的 g 或後面的量化不在 X3 的 merge_g2／g4／merge_g2_int4 裡"
    if ops in (["mc"], ["share"]):
        return None, "跨層逐 token 共用（MiniCache／KVSharer 式）X3 沒有解碼段"
    return None, "這個量化格式 X3 沒有解碼段（X3 只有 fp8（fmt_choice）、int8_token、int4_kivi、int2_kivi、k4v2）"


FIELDS_X1 = ["run_id", "ts", "shard", "model", "item_id", "task", "band", "seed", "needle_blocks", "range", "region", "spec",
             "groups", "roles", "family", "families", "n_lossy", "deployable", "attn_path", "k_drop", "kept_frac", "needle_kept",
             "bytes_region", "bytes_bf16_region", "ratio_region", "bytes_total", "ratio_total", "bits_per_value_region",
             "bytes_shared", "bytes_detail", "kl", "gold_lp", "gold_lp_e0", "top1", "first_ok", "first_ok_e0",
             "relerr_k", "relerr_v", "cenerr_k", "attn_err", "attn_err_max", "attn_err_wo", "proxy_layers", "proxy_nq",
             "t_pipe_s", "t_tf_s", "calib_run_id", "n_rows_item", "own_gpu_intruders", "host_level"]
FIELDS_X2 = ["run_id", "ts", "shard", "model", "ctx", "n_blocks", "item_id", "task", "band", "seed", "spec", "roles", "tiers",
             "budgets", "family", "families", "n_lossy", "deployable", "attn_path", "k_drop", "kept_frac", "needle_kept",
             "bytes_region", "ratio_region", "bits_per_value_region", "gen_score", "gen_stop", "score_e0", "t_gen_s",
             "calib_run_id", "n_rows_item", "own_gpu_intruders", "host_level"]


# ============================================================================ 組合語法（純 Python）
class SpecError(ValueError):
    pass


def drop_k(frac: float, nreg: int) -> int:
    """丟幾個 block＝round(比例×區內 block 數)（和 S2 的 int(round(fr * nreg))、DS1 的 round(frac×63) 同一個寫法）。"""
    return int(round(frac * nreg))


def deep_layers(L: int) -> list:
    return list(range(L // 2, L))


def layer_pairs(deep: list) -> list:
    return [(deep[i], deep[i + 1]) for i in range(0, len(deep) - 1, 2)]


def parse_spec(spec: str) -> list:
    spec = spec.strip()
    if spec in ("", "none"):
        return []
    out = []
    for tok in spec.split("+"):
        p = tok.strip().split(":")
        op = p[0]
        try:
            if op == "drop":
                if len(p) != 3:
                    raise SpecError("drop 要寫成 drop:<方法>:<比例>")
                m, fr = p[1].upper(), float(p[2])
                if m not in DROP_METHODS:
                    raise SpecError(f"不認得的丟法 {p[1]}（可用 {DROP_METHODS}）")
                if not 0 < fr < 1:
                    raise SpecError(f"丟的比例要在 (0, 1)：{p[2]}")
                out.append(dict(op="drop", method=m, frac=fr))
            elif op == "merge":
                if len(p) != 2 or int(p[1]) not in MERGE_G:
                    raise SpecError(f"merge 的 g 只能是 {MERGE_G}")
                out.append(dict(op="merge", g=int(p[1])))
            elif op in ("lr", "xl"):
                d = dict(op=op, rk=0, rv=0)
                if op == "lr":
                    d["joint"] = False
                else:
                    d["w"] = 4
                for x in p[1:]:
                    if op == "lr" and x == "joint":
                        d["joint"] = True
                    elif op == "xl" and x == "cal":
                        d["cal"] = True                          # 10/21：事先校準的基底（只在 True 時加這個鍵，舊組合的 dict 不變）
                    elif x[:1] == "k" and x[1:].isdigit():
                        d["rk"] = int(x[1:])
                    elif x[:1] == "v" and x[1:].isdigit():
                        d["rv"] = int(x[1:])
                    elif op == "xl" and x[:1] == "w" and x[1:].isdigit():
                        d["w"] = int(x[1:])
                    else:
                        raise SpecError(f"{op} 不認得的參數 {x}")
                out.append(d)
            elif op == "q":
                if len(p) != 2 or not p[1]:
                    raise SpecError("q 要寫成 q:<格式>")
                out.append(dict(op="q", fmt=p[1]))
            elif op == "mc":
                d = dict(op="mc", t=0.6, gamma=0.05)
                for x in p[1:]:
                    if x[:1] == "t":
                        d["t"] = float(x[1:])
                    elif x[:1] == "g":
                        d["gamma"] = float(x[1:])
                    else:
                        raise SpecError(f"mc 不認得的參數 {x}")
                if not (0 < d["t"] < 1 and 0 <= d["gamma"] < 1):
                    raise SpecError("mc 的 t 要在 (0,1)、γ 在 [0,1)")
                out.append(d)
            elif op == "share":
                if len(p) != 1:
                    raise SpecError("share 沒有參數")
                out.append(dict(op="share"))
            else:
                raise SpecError(f"不認得的階段 {op}")
        except SpecError:
            raise
        except ValueError as ex:
            raise SpecError(f"{tok}：{ex}") from None
    return out


def _g(x: float) -> str:
    return format(x, "g")


def canonical(stages: list) -> str:
    if not stages:
        return "none"
    parts = []
    for s in stages:
        op = s["op"]
        if op == "drop":
            parts.append(f"drop:{s['method']}:{_g(s['frac'])}")
        elif op == "merge":
            parts.append(f"merge:{s['g']}")
        elif op == "lr":
            parts.append(f"lr:k{s['rk']}" + (f":v{s['rv']}" if s["rv"] else "") + (":joint" if s["joint"] else ""))
        elif op == "xl":
            parts.append(f"xl:k{s['rk']}" + (f":v{s['rv']}" if s["rv"] else "") + (f":w{s['w']}" if s["w"] != 4 else "")
                         + (":cal" if s.get("cal") else ""))
        elif op == "q":
            parts.append(f"q:{s['fmt']}")
        elif op == "mc":
            parts.append("mc" + (f":t{_g(s['t'])}" if s["t"] != 0.6 else "") + (f":g{_g(s['gamma'])}" if s["gamma"] != 0.05 else ""))
        elif op == "share":
            parts.append("share")
    return "+".join(parts)


def halves(fmt: str):
    """格式 → (K 半邊, V 半邊)（m10_codec_x.halves 的純 Python 版，別名換成 kivi4／kivi2）；只有整塊的格式（fp8k0_* 等）→ None。"""
    if fmt.startswith("kv-"):
        rest = fmt[3:]
        for k in sorted(list(HALF) + list(ALIAS), key=len, reverse=True):
            if rest.startswith(k + "-"):
                kf, vf = k, rest[len(k) + 1:]
                break
        else:
            raise SpecError(f"拆不開的 kv- 格式 {fmt}")
    elif fmt in HALF or fmt in ALIAS:
        kf = vf = fmt
    else:
        return None
    kf, vf = ALIAS.get(kf, kf), ALIAS.get(vf, vf)
    if kf not in HALF or vf not in HALF:
        raise SpecError(f"{fmt}：半邊格式 {kf}／{vf} 不認得")
    return kf, vf


def known_fmt(fmt: str) -> bool:
    if fmt in M10_FULL or fmt in NEW_FULL or fmt in OWN_FMTS:
        return True
    if fmt.startswith("kv-"):
        halves(fmt)
        return True
    return False


def validate(stages: list, info: dict | None = None, nreg: int | None = None) -> list:
    """擋掉不合法的組合（丟 SpecError，訊息說為什麼）。info＝MODEL_INFO[model]；nreg＝區內 block 數（給了才檢查丟的個數）。"""
    ops = [s["op"] for s in stages]
    if len(set(ops)) != len(ops):
        raise SpecError("同一種階段只能出現一次")
    if "mc" in ops or "share" in ops:
        if len(ops) != 1:
            raise SpecError("mc／share 只能單獨用（跨層逐 token 共用只當對照）")
        return stages
    if "q" in ops and ops[-1] != "q":
        raise SpecError("q 必須是最後一個階段")
    seq = tuple(o for o in ops if o != "q")
    if "lr" in seq and "xl" in seq:
        raise SpecError("lr 和 xl 不能一起用（兩種低秩表示疊在一起沒有定義）")
    if "merge" in seq and ("lr" in seq or "xl" in seq):
        raise SpecError("merge 後的 key 是不同位置的平均、不在任何單一位置上，pre-RoPE 低秩沒有定義 → merge 不能和 lr／xl 一起用")
    if seq not in ALLOWED_SEQ:
        raise SpecError(f"順序不合法：{'+'.join(ops)}（drop 要在最前面，只有 xl 可以排在它前面當順序對照；merge 在 drop 之後；q 最後）")
    if any(s.get("cal") for s in stages) and seq[:2] == ("xl", "drop"):
        raise SpecError("xl:…:cal 的基底和題目無關，xl 排在 drop 前的順序對照沒有意義（結果和 drop+xl:…:cal 相同）→ 只接受 drop 在前")
    D = (info or {}).get("D", 128)
    for s in stages:
        if s["op"] == "drop" and nreg is not None:
            k = drop_k(s["frac"], nreg)
            if not 1 <= k <= nreg - 1:
                raise SpecError(f"丟 {k} 個 block（區內 {nreg} 個）：至少丟 1 個、至少留 1 個")
        if s["op"] in ("lr", "xl"):
            for r in (s["rk"], s["rv"]):
                if r < 0 or r > D:
                    raise SpecError(f"秩 {r} 要在 0..{D}")
            if s["rk"] + s["rv"] == 0:
                raise SpecError("K、V 的秩不能都是 0（0＝那一半不降秩）")
            if s["op"] == "lr" and (s["rk"] % 4 or s["rv"] % 4):
                raise SpecError("lr 的秩要是 4 的倍數（2-bit 打包沿最後一維）")
            if s["op"] == "xl":
                if s["w"] < 1 or (info and info["L"] % s["w"]):
                    raise SpecError(f"xl 的窗口 {s['w']} 要整除層數")
                if info and any((r * s["w"] * info["H"]) % 4 for r in (s["rk"], s["rv"])):
                    raise SpecError("xl 的聯合秩要是 4 的倍數")
    q = next((s for s in stages if s["op"] == "q"), None)
    if q is not None:
        f = q["fmt"]
        if not known_fmt(f):
            raise SpecError(f"不認得的格式 {f}")
        low = next((s for s in stages if s["op"] in ("lr", "xl")), None)
        if low is not None:
            if f in OWN_FMTS:
                raise SpecError(f"{f} 只能套在原始 KV 上")
            hv = halves(f)
            if hv is None:
                raise SpecError(f"{f} 是整塊的格式、拆不成 K／V 兩半，不能和 lr／xl 一起用")
            for r, hf, name in ((low["rk"], hv[0], "K"), (low["rv"], hv[1], "V")):
                if r > 0 and hf not in COEF_HALF:
                    raise SpecError(f"{name} 是低秩係數，只能用 {COEF_HALF}（{hf} 的分組依賴 block 結構或 head 維度）")
    return stages


def spec_attrs(stages: list) -> dict:
    fams = sorted({FAMILY[s["op"]] for s in stages})
    n = len(stages)
    dep = not any(s["op"] == "drop" and s["method"] in NOT_DEPLOYABLE for s in stages)
    return dict(family=("none" if n == 0 else fams[0] if n == 1 else "combo"), families="+".join(fams) or "none",
                n_lossy=n, deployable=int(dep))


# ============================================================================ 位元組（純 Python；GPU 路徑一律 assert 實際＝預測）
def half_bytes(f: str, shape: tuple, role: str) -> int:
    """一半（K 或 V）或係數 [A, B, T, R] 用半邊格式 f 存的位元組（m10_codec_x.enc_half 的實際 tensor 大小）。"""
    A, B, T, R = shape
    n = A * B * T * R
    f = ALIAS.get(f, f)
    if f == "bf16":
        return n * 2
    if f == "int8_token":
        return n + A * B * T * 4
    if f in ("int4_token", "int4h_token"):
        return n // 2 + A * B * T * 4
    if f in ("fp8_head", "fp8e5_head"):
        return n + A * B * 4
    if f in ("fp8_token", "fp8e5_token"):
        return n + A * B * T * 4
    if f == "mxfp4":
        return n // 2 + n // 32
    if f == "nvfp4":
        return n // 2 + n // 16 + A * 4
    if f in ("kivi4", "kivi2"):
        grp = A * B * (T // 32) * R if role == "k" else A * B * T * (R // 32)
        return (n // 2 if f == "kivi4" else n // 4) + 2 * 2 * grp
    if f == "int2h_g32":
        return n // 4 + 2 * 2 * (n // 32)
    raise SpecError(f"half_bytes 不認得 {f}")


def full_fmt_bytes(fmt: str, shape: tuple) -> int:
    """整個 block [L, 2, H, T, D] 用格式 fmt 存的位元組（m10_codec／m10_codec_x 的 Enc.nbytes；vllm_fp8＝每值 1 byte）。"""
    L, _, H, T, D = shape
    n = L * 2 * H * T * D
    if fmt == "vllm_fp8":
        return n
    if fmt in M10_FULL:
        if fmt.startswith("fp8k0_"):
            return full_fmt_bytes("fp8_" + fmt[6:], shape) + L * H * D * 2
        if fmt.startswith("fp8_"):
            return n + {"head": L * 2 * H, "channel": L * 2 * H * D, "token": L * 2 * H * T}[fmt[4:]] * 4
        if fmt == "int8_token":
            return n + L * 2 * H * T * 4
        if fmt == "int4_token":
            return n // 2 + L * 2 * H * T * 4
        if fmt == "int4_kivi":
            return n // 2 + 2 * 2 * (L * H * (T // 32) * D + L * H * T * (D // 32))
    kf, vf = halves(fmt)
    return half_bytes(kf, (L, H, T, D), "k") + half_bytes(vf, (L, H, T, D), "v")


def mc_nret(gamma: float, T: int) -> int:
    return int(math.floor(gamma * T + 0.5))


def predict_bytes(stages: list, info: dict, nreg: int, chunk: int = CHUNK) -> dict:
    """每個元件的位元組（'shared.*'＝跨請求共用，不算進這個請求）。torch 和 numpy 的實作都要和這裡逐項相等。"""
    L, H, D = info["L"], info["H"], info["D"]
    st = {s["op"]: s for s in stages}
    Tr = nreg * chunk
    by = {}
    if "mc" in st:
        pr = layer_pairs(deep_layers(L))
        nr = mc_nret(st["mc"]["gamma"], Tr)
        bits = max(1, math.ceil(math.log2(Tr)))
        np_ = len(pr) * 2 * H                          # （層對, K/V, head）的個數
        by["mc.dir"] = np_ * Tr * D * 2
        by["mc.norm"] = np_ * 2 * Tr * 2
        by["mc.keep"] = np_ * nr * 2 * D * 2
        by["mc.idx"] = np_ * ((nr * bits + 7) // 8)
        by["raw"] = (L - 2 * len(pr)) * 2 * H * Tr * D * 2
        return by
    if "share" in st:
        by["raw"] = (L - len(layer_pairs(deep_layers(L)))) * 2 * H * Tr * D * 2
        return by
    kept = nreg
    if "drop" in st:
        kept = nreg - drop_k(st["drop"]["frac"], nreg)
        by["mask"] = (nreg + 7) // 8
    Tk = kept * chunk
    g = st["merge"]["g"] if "merge" in st else 1
    Ts, Tc = Tk // g, chunk // g
    fmt = st["q"]["fmt"] if "q" in st else None
    if fmt == "vllm_fp8":
        by["shared.scale"] = L * 2 * 4                 # vLLM 的 k_scale／v_scale：每層一個 FP32（＝1.0），跨請求共用
    low = st.get("lr") or st.get("xl")
    if low is None:
        if fmt is None:
            by["raw"] = L * 2 * H * Ts * D * 2
        else:
            by["q"] = (Ts // Tc) * full_fmt_bytes(fmt, (L, 2, H, Tc, D))
        return by
    hv = halves(fmt) if fmt else (None, None)
    for hi, h in enumerate("kv"):
        r = low["rk"] if hi == 0 else low["rv"]
        f = hv[hi]
        if r == 0:
            by[f"{h}.raw"] = L * H * Tk * D * 2 if f is None else (Tk // chunk) * half_bytes(f, (L, H, chunk, D), h)
        elif low["op"] == "lr":
            shape = (L, 1, Tk, r * H) if low["joint"] else (L, H, Tk, r)
            by[f"{h}.coef"] = shape[0] * shape[1] * shape[2] * shape[3] * 2 if f is None else half_bytes(f, shape, "k")
            by[f"shared.{h}.basis"] = (L * (H * D * r * H + H * D) if low["joint"] else L * H * (D * r + D)) * 2
        else:
            W = low["w"]
            rr = r * W * H
            shape = (L // W, 1, Tk, rr)
            by[f"{h}.coef"] = shape[0] * Tk * rr * 2 if f is None else half_bytes(f, shape, "k")
            # 每題 SVD：基底每題一份、算在這個請求裡；:cal（10/21）：事先校準、跨請求共用 → 另列 shared（同樣大小、BF16）
            by[f"shared.{h}.basis" if low.get("cal") else f"{h}.basis"] = (L // W) * rr * W * H * D * 2
    return by


def bytes_summary(by: dict, info: dict, nreg: int, n_blocks: int, chunk: int = CHUNK) -> dict:
    cb = info["L"] * 2 * info["H"] * chunk * info["D"] * 2
    reg = sum(v for k, v in by.items() if not k.startswith("shared."))
    tot = (n_blocks - nreg) * cb + reg
    return dict(bytes_region=reg, bytes_bf16_region=nreg * cb, ratio_region=round(reg / (nreg * cb), 6), bytes_total=tot,
                ratio_total=round(tot / (n_blocks * cb), 6), bits_per_value_region=round(16 * reg / (nreg * cb), 4),
                bytes_shared=sum(v for k, v in by.items() if k.startswith("shared.")),
                bytes_detail=json.dumps(by, sort_keys=True))


def predict_ratio(spec: str, model: str, nreg: int = N_BLOCKS - 1) -> float:
    st = validate(parse_spec(spec), MODEL_INFO[model], nreg)
    return bytes_summary(predict_bytes(st, MODEL_INFO[model], nreg), MODEL_INFO[model], nreg, nreg + 1)["ratio_region"]


def load_combos(path: str, model: str, smoke: bool = False, nreg: int = N_BLOCKS - 1) -> list:
    """組合清單 → [{spec（標準寫法）, stages, groups, roles}]；同一個組合在多組出現時合併。每一條都先過 validate。"""
    d = json.load(open(path))
    info = MODEL_INFO[model]
    if smoke:
        entries = [(s, "smoke", "smoke") for s in d["smoke"]]
    else:
        entries = [(s, g["id"], g["role"]) for g in d["groups"] if model in g.get("models", list(MODEL_INFO)) for s in g["specs"]]
    out = {}
    for spec, gid, role in entries:
        st = validate(parse_spec(spec), info, nreg)
        c = canonical(st)
        e = out.setdefault(c, dict(spec=c, stages=st, groups=[], roles=[]))
        if gid not in e["groups"]:
            e["groups"].append(gid)
        if role not in e["roles"]:
            e["roles"].append(role)
    return list(out.values())


def lossless_key(spec: str):
    """S1 量得到的儲存格式（只有原始 KV＋bf16／fp8_token／fp8_head／int8_token／int4_kivi，可以先丟 block）；其他回 None（NOT_MEASURED）。"""
    if spec == "E0":
        return "bf16"
    st = parse_spec(spec)
    ops = [s["op"] for s in st]
    if any(o not in ("drop", "q") for o in ops):
        return None
    q = next((s["fmt"] for s in st if s["op"] == "q"), None)
    if q is None:
        return "bf16"
    return q if q in ("fp8_token", "fp8_head", "int8_token", "int4_kivi") else None


def s1_multipliers(path: str, model: str, codec: str = "zstd:3") -> dict:
    """S1 實測的無損壓縮比（T4 自然文字的 dump、codec 固定 zstd:3；bf16 用 planes 變體）。沒有 → {}（呼叫端寫 NOT_MEASURED）。"""
    if not os.path.exists(path):
        return {}
    S = json.load(open(path))
    d = S.get("dumps", {}).get(f"{model}_T4_64")
    if not d:
        return {}
    out = {}
    b = d.get("bf16", {}).get(f"all|planes|{codec}")
    if b:
        out["bf16"] = b["ratio"]
    for f, v in d.get("fmts", {}).items():
        if codec in v:
            out[f] = v[codec]["ratio_vs_fmt"]
    out["_run_id"] = S.get("run_id")
    return out


# ============================================================================ 10/21 延伸：出題、計畫檔、待做題數（純 Python，不 import torch）
def rotations_from_source() -> list:
    """m10_quality.ROTATIONS（m10_quality 在模組頂層 import torch，CPU 這邊不能 import）：用 ast 從原始碼讀那一行的字面值，不抄數字。
    GPU 這邊 gpu-selftest 的 plan_cpu_eq_m10 再核對一次 plan_interleaved_cpu＝plan_interleaved。"""
    import ast
    src = open(os.path.join(HERE, "m10_quality.py"), encoding="utf-8").read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "ROTATIONS" for t in node.targets):
            return [tuple(x) for x in ast.literal_eval(node.value)]
    raise RuntimeError("m10_quality.py 裡找不到 ROTATIONS（規則 7：查不到不等於沒有）")


def plan_interleaved_cpu(tasks, n, seed0) -> list:
    """plan_interleaved 的純 Python 版（m10_quality.plan_items 的同一個規則：T5 單輪、其他照 ROTATIONS 輪流；seed＝seed0＋k）。
    給工作池的 pending 與 self-test 用（不 import torch）。"""
    rot = rotations_from_source()
    per = {t: [(t, None, None, seed0 + k) if t == "T5" else (t, *rot[k % len(rot)], seed0 + k) for k in range(n)] for t in tasks}
    return [per[t][k] for k in range(n) for t in tasks]


def shard_pending(tasks, n, seed0, shard: str, out: str) -> list:
    """這個 shard（k % 份數 == 編號，和 cmd_gen／cmd_screen 同一個切法）還沒做完的題（done_union：同一組 shard 檔任何一個寫完的都算）。"""
    si, sk = parse_shard(shard)
    with contextlib.redirect_stdout(sys.stderr):          # read_rows 遇到寫到一半的 .gz 會印到 stdout；stdout 只留題數
        done = done_union(out)
    return [item_id_of(*p) for k, p in enumerate(plan_interleaved_cpu(tasks, n, seed0)) if k % sk == si and item_id_of(*p) not in done]


def cmd_pending(a):
    """工作池用：印這個 shard 還有幾題沒做（只印一個整數）。只用 CPU、不 import torch。"""
    print(len(shard_pending(a.tasks, a.n, a.seed0, a.shard, a.out)))


def plan_specs(which: str, model: str, path: str = COMBOS_V2) -> list:
    """combos_v2.json 裡事先寫死的配置：x2b＝X2b 的配置（Llama 11、Qwen 12；照列的順序；同一個每題 SVD 的配置排在一起，給 --xl-cache）；
    xlcal＝xKV 校準的生成比較（每一對：校準基底版、每題 SVD 的同名配置）。"""
    d = json.load(open(path))
    if which == "x2b":
        return list(d["x2b"][model])
    if which == "xlcal":
        return [s for pr in d["xlcal_pairs"] for s in pr]
    raise ValueError(which)


def xlcal_pairs(path: str = COMBOS_V2) -> list:
    return [tuple(p) for p in json.load(open(path))["xlcal_pairs"]]


def build_plan(which: str, v1sel: dict) -> dict:
    """X2b／xKV 校準生成的計畫檔（gen 的 --selection 吃的格式）。roles／budgets 照第一版 x2_selection.json（M2 的主要組合、最好的單一方法
    都是第一版事先選的，不重選）；第一版沒有的配置 roles 只有 x2b。tiers 一律 core（gen --tier core）。"""
    models = {}
    for m in MODEL_INFO:
        v1 = {e["spec"]: e for e in v1sel.get("models", {}).get(m, {}).get("configs", [])}
        cf = []
        for s in plan_specs(which, m):
            c = canonical(validate(parse_spec(s), MODEL_INFO[m], N_BLOCKS - 1))
            if c != s:
                raise SpecError(f"combos_v2.json 的 {s} 不是標準寫法（{c}）")
            if which == "x2b":
                e = v1.get(c)
                cf.append(dict(spec=c, roles=(list(e["roles"]) if e else []) + ["x2b"], tiers=["core"],
                               budgets=list(e["budgets"]) if e else [], v1_listed=bool(e),
                               why=("第一版 X2 也量過（roles 照第一版）" if e else "第一版 X2 沒量（只在「只用 X2b」報）")))
            else:
                cal = any(x.get("cal") for x in parse_spec(c))
                cf.append(dict(spec=c, roles=["xlcal" if cal else "xlcal_ref"], tiers=["core"], budgets=[],
                               why="校準基底版" if cal else "每題 SVD 的同名配置（同一題、同一個行程配對）"))
        models[m] = dict(configs=cf)
    if which == "x2b":
        return dict(go=True, which=which, reason="X2b 事先寫死的配置（code/m16_xstack_combos_v2.json 的 x2b）", tasks=["T2", "T3", "T4", "T5"],
                    seed0=X2B_SEED0, n=X2B_N, v1_selection_run_id=v1sel.get("run_id"), models=models)
    return dict(go=True, which=which, reason="xKV 校準基底 vs 每題 SVD（code/m16_xstack_combos_v2.json 的 xlcal_pairs）",
                tasks=["T2", "T3", "T4", "T5"], seed0=X2_SEED0, n=XLCAL_GEN_N, tf_seed0=X1_SEED0, tf_n=XLCAL_TF_N,
                tf_combos=os.path.relpath(COMBOS_V2, REPO), models=models)


PLAN_KEYS = ("go", "which", "tasks", "seed0", "n", "tf_seed0", "tf_n", "models")


def cmd_plan(a):
    v1 = json.load(open(a.selection)) if a.which == "x2b" else {}
    if a.which == "x2b" and not v1.get("go"):
        raise SystemExit(f"第一版 {a.selection} 的 go 不是 true → X2b 沒有可以照的 roles（停，規則 2）")
    p = build_plan(a.which, v1)
    if os.path.exists(a.out):
        old = json.load(open(a.out))
        if {k: old.get(k) for k in PLAN_KEYS} == {k: p.get(k) for k in PLAN_KEYS}:
            print(f"{a.out} 已有、內容相同（run {old.get('run_id')}）→ 不重寫", flush=True)
            return
        raise SystemExit(f"{a.out} 已存在但內容不同 → 不默默改事先寫死的計畫（要改另開版本）")
    p.update(run_id=os.environ.get("RUN_ID", "adhoc"), ts=ts())
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(p, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"寫好 {a.out}：" + "；".join(f"{m} {len(v['configs'])} 個配置" for m, v in p["models"].items()), flush=True)


# ============================================================================ numpy 參考實作（self-test 與 gpu-selftest 用）
def _np():
    import numpy as np
    return np


def np_bf16(x):
    """float → 最接近的 BF16（round-to-nearest-even），回傳 float32（和 torch 的 .to(bfloat16) 相同；不處理 NaN）。"""
    np = _np()
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = ((u + (np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1)))) >> np.uint32(16)) << np.uint32(16)
    return r.astype(np.uint32).view(np.float32)


def np_pack(q, bits):
    """沿最後一維打包（m10_codec._pack4／m10_codec_x._pack2 的排法），只為了得到實際的位元組。"""
    np = _np()
    u = np.asarray(q).astype(np.uint8)
    if bits == 4:
        return u[..., 0::2] | (u[..., 1::2] << 4)
    return u[..., 0::4] | (u[..., 1::4] << 2) | (u[..., 2::4] << 4) | (u[..., 3::4] << 6)


def np_half_enc(t, f: str, role: str):
    """[A,B,T,R] float32 → (解碼後 float32（還沒捨入成 BF16）, 位元組)。逐步照 m10_codec_x.enc_half／dec_half 的 FP32 運算。"""
    np = _np()
    t = np.asarray(t, dtype=np.float32)
    A, B, T, R = t.shape
    f = ALIAS.get(f, f)
    if f == "bf16":
        return np_bf16(t), t.size * 2
    if f in ("int8_token", "int4_token"):
        qmax = np.float32(127.0 if f == "int8_token" else 7.0)
        amax = np.abs(t).max(axis=-1, keepdims=True)
        s = np.where(amax > 0, amax / qmax, np.float32(1.0)).astype(np.float32)
        q = np.clip(np.round(t / s), -qmax, qmax).astype(np.float32)
        stored = q.astype(np.int8).nbytes if f == "int8_token" else np_pack(q + 8, 4).nbytes
        return (q * s).astype(np.float32), stored + s.nbytes
    if f in ("kivi4", "kivi2"):
        bits = 4 if f == "kivi4" else 2
        lv = np.float32(2 ** bits - 1)
        if role == "k":
            g, red = t.reshape(A, B, T // 32, 32, R), 3
        else:
            g, red = t.reshape(A, B, T, R // 32, 32), 4
        mn, mx = g.min(axis=red, keepdims=True), g.max(axis=red, keepdims=True)
        mn16 = mn.astype(np.float16)
        sc16 = ((mx - mn) / lv).astype(np.float16)
        sc = np.where(sc16 > 0, sc16, np.float16(1.0)).astype(np.float32)
        q = np.clip(np.round((g - mn16.astype(np.float32)) / sc), np.float32(0), lv)
        dec = (q * sc.astype(np.float16).astype(np.float32) + mn16.astype(np.float32)).reshape(A, B, T, R)
        nb = np_pack(q.reshape(A, B, T, R), bits).nbytes + sc.astype(np.float16).nbytes + mn16.nbytes
        return dec.astype(np.float32), nb
    raise NotImplementedError(f"numpy 參考沒有 {f}")


def np_m10_int4_kivi(x):
    """m10_codec.py:139–157（encode）＋177–182（decode）逐行照抄成 numpy：x＝[L,2,H,T,D] float32（BF16 值）→ (解碼 float32, 位元組)。
    self-test 拿它和 np_half_enc(kivi4) 比，證明「係數用的 KIVI 量化器」和第二階段的 int4_kivi 是同一個定義。"""
    np = _np()
    L, _, H, T, D = x.shape
    G = 32
    k = x[:, 0].astype(np.float32).reshape(L, H, T // G, G, D)
    kmn, kmx = k.min(axis=3, keepdims=True), k.max(axis=3, keepdims=True)
    v = x[:, 1].astype(np.float32).reshape(L, H, T, D // G, G)
    vmn, vmx = v.min(axis=4, keepdims=True), v.max(axis=4, keepdims=True)
    parts, dec = {}, {}
    for name, t, mn, mx in (("k", k, kmn, kmx), ("v", v, vmn, vmx)):
        mn16, sc16 = mn.astype(np.float16), ((mx - mn) / np.float32(15.0)).astype(np.float16)
        sc = np.where(sc16 > 0, sc16, np.float16(1.0)).astype(np.float32)
        q = np.clip(np.round((t - mn16.astype(np.float32)) / sc), 0, 15)
        parts[f"{name}_q"] = np_pack(q.reshape(L, H, T, D), 4)
        parts[f"{name}_scale"] = sc.astype(np.float16)
        parts[f"{name}_zero"] = mn16
        dec[name] = (q * parts[f"{name}_scale"].astype(np.float32) + mn16.astype(np.float32)).reshape(L, H, T, D)
    y = np.stack((dec["k"], dec["v"]), axis=1)
    return y.astype(np.float32), sum(p.nbytes for p in parts.values())


def np_fnuz(x):
    """float32 → 最接近的 float8_e4m3fnuz 值（round-to-nearest-even），先截在 ±240（vLLM 未校準、scale＝1.0）。
    e4m3fnuz：指數偏移 8、沒有無限大、只有一個 NaN 碼；最小正規數 2^-7、次正規間距 2^-10、最大 1.875×2^7＝240。"""
    np = _np()
    x = np.asarray(x, dtype=np.float32)
    a = np.minimum(np.abs(x), np.float32(FNUZ_MAX)).astype(np.float64)
    _, ex = np.frexp(a)
    e = np.maximum(ex - 1, -7)
    step = np.ldexp(1.0, e - 3)
    q = np.minimum(np.rint(a / step) * step, FNUZ_MAX)
    return (np.sign(x) * q).astype(np.float32)


def np_rope_fwd(x, c, s):
    """x [..., T, D]；c、s [T, D/2]（模型的 BF16 cos/sin 前半）；rotate_half 配對（j 與 j+D/2）；float64。"""
    np = _np()
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)


def np_rope_inv(y, c, s):
    np = _np()
    h = y.shape[-1] // 2
    y1, y2 = y[..., :h], y[..., h:]
    det = c * c + s * s
    return np.concatenate([(y1 * c + y2 * s) / det, (y2 * c - y1 * s) / det], axis=-1)


class NCtx:
    """numpy 參考的上下文：c、s＝[位置, D/2]（float64，模型的 BF16 cos/sin）、calib（float64）、drop 的區內順序、deep 層、chunk。"""

    def __init__(self, c, s, calib=None, orders=None, deep=None, chunk=CHUNK, xlcal=None):
        self.c, self.s, self.calib, self.orders, self.deep, self.chunk = c, s, calib, orders or {}, deep, chunk
        self.xlcal = xlcal                       # 10/21：xKV 校準基底 {k_U, v_U: [窗口, W·H·D, R]}（float64）


def np_full_q(Sm, fmt: str, Tc: int):
    np = _np()
    L, _, H, Ts, D = Sm.shape
    outs, nb = [], 0
    for c0 in range(0, Ts, Tc):
        ch = Sm[:, :, :, c0:c0 + Tc]
        if fmt == "vllm_fp8":
            y, b = np_fnuz(ch), ch.size
        elif fmt == "int4_kivi":
            y, b = np_m10_int4_kivi(ch)
        else:
            hv = halves(fmt)
            if hv is None:
                raise NotImplementedError(fmt)
            ys = []
            b = 0
            for hi, h in enumerate("kv"):
                d, bb = np_half_enc(ch[:, hi], hv[hi], h)
                ys.append(d)
                b += bb
            y = np.stack(ys, axis=1)
        outs.append(np_bf16(y))
        nb += b
    return np.concatenate(outs, axis=3), nb


def np_raw_half(t, f, role, C):
    np = _np()
    if f is None:
        return np.asarray(t, dtype=np.float32), t.size * 2
    outs, nb = [], 0
    for c0 in range(0, t.shape[2], C):
        d, b = np_half_enc(t[:, :, c0:c0 + C], f, role)
        outs.append(np_bf16(d))
        nb += b
    return np.concatenate(outs, axis=2), nb


def np_lr_half(t, P, hi, r, joint, f, cx):
    np = _np()
    L, H, T, D = t.shape
    c, s = cx.c[P], cx.s[P]
    h = "kv"[hi]
    mu = cx.calib[f"{h}_mean"]
    coefs = []
    for l in range(L):
        x = t[l].astype(np.float64)
        xp = np_rope_inv(x, c, s) if hi == 0 else x
        if joint:
            Z = xp.transpose(1, 0, 2).reshape(T, H * D) - mu[l].reshape(1, H * D)
            coefs.append((Z @ cx.calib[f"{h}_Uj"][l][:, :r * H])[None])
        else:
            coefs.append((xp - mu[l][:, None, :]) @ cx.calib[f"{h}_U"][l][:, :, :r])
    C4 = np.stack(coefs).astype(np.float32)
    if f in (None, "bf16"):
        Cd, nb = np_bf16(C4), C4.size * 2
    else:
        Cd, nb = np_half_enc(C4, f, "k")
    outs = []
    for l in range(L):
        if joint:
            Zh = (Cd[l, 0].astype(np.float64) @ cx.calib[f"{h}_Uj"][l][:, :r * H].T + mu[l].reshape(1, H * D))
            Zh = Zh.reshape(T, H, D).transpose(1, 0, 2)
        else:
            Zh = Cd[l].astype(np.float64) @ cx.calib[f"{h}_U"][l][:, :, :r].transpose(0, 2, 1) + mu[l][:, None, :]
        outs.append(np_bf16(np_rope_fwd(Zh, c, s) if hi == 0 else Zh))
    sh = (L * (H * D * r * H + H * D) if joint else L * H * (D * r + D)) * 2
    return np.stack(outs), nb, sh


def np_xl_half(src, srcpos, P, hi, rhe, W, f, cx, keep_idx):
    """xKV（numpy 參考用精確 SVD）。src [L,H,Ts,D]＝要分解的 token；回傳 (留下的 token 的解碼 [L,H,Tk,D], 係數位元組, 基底位元組, 每窗相對誤差)。"""
    np = _np()
    L, H, T, D = src.shape
    r = rhe * W * H
    c0, s0 = cx.c[srcpos], cx.s[srcpos]
    As, Bs, errs = [], [], []
    for w in range(L // W):
        Z = np.concatenate([(np_rope_inv(src[l].astype(np.float64), c0, s0) if hi == 0 else src[l].astype(np.float64))
                            .transpose(1, 0, 2).reshape(T, H * D) for l in range(w * W, (w + 1) * W)], axis=1)
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        errs.append(float(np.sqrt((S[r:] ** 2).sum() / (S ** 2).sum())))
        A = U[:, :r] * S[:r]
        if keep_idx is not None:
            A = A[keep_idx]
        As.append(A.astype(np.float32))
        Bs.append(np_bf16(Vt[:r].astype(np.float32)))
    A4 = np.stack(As)[:, None]
    if f in (None, "bf16"):
        Ad, nb = np_bf16(A4), A4.size * 2
    else:
        Ad, nb = np_half_enc(A4, f, "k")
    Tk = A4.shape[2]
    c, s = cx.c[P], cx.s[P]
    outs = []
    for w in range(L // W):
        Zh = Ad[w, 0].astype(np.float64) @ Bs[w].astype(np.float64)
        for j in range(W):
            x = Zh[:, j * H * D:(j + 1) * H * D].reshape(Tk, H, D).transpose(1, 0, 2)
            outs.append(np_bf16(np_rope_fwd(x, c, s) if hi == 0 else x))
    return np.stack(outs), nb, (L // W) * r * W * H * D * 2, errs


def np_xl_half_cal(src, P, hi, rhe, W, f, cx):
    """xKV、事先校準的基底（10/21；numpy 參考）。src [L,H,T,D]＝留下的 token、P＝它們的位置。每個窗口：Z（K 先逆 RoPE、不去平均＝照 xKV）
    投到校準基底 U[w][:, :r]（係數＝Z·U，float64），係數照常量化；重建用 BF16 的基底（和每題 SVD 的基底同一個存法）。
    回傳 (解碼 [L,H,T,D], 係數位元組, 共用基底位元組, 每窗 pre-RoPE 相對誤差（量化前）)。"""
    np = _np()
    L, H, T, D = src.shape
    r = rhe * W * H
    U = cx.xlcal[f"{'kv'[hi]}_U"]
    if U.shape[-1] < r:
        raise ValueError(f"校準基底只存到 {U.shape[-1]} 維，要 {r}")
    c, s = cx.c[P], cx.s[P]
    As, Bs, errs = [], [], []
    for w in range(L // W):
        Z = np.concatenate([(np_rope_inv(src[l].astype(np.float64), c, s) if hi == 0 else src[l].astype(np.float64))
                            .transpose(1, 0, 2).reshape(T, H * D) for l in range(w * W, (w + 1) * W)], axis=1)
        Uw = U[w][:, :r].astype(np.float64)
        A = Z @ Uw
        errs.append(float(np.linalg.norm(Z - A @ Uw.T) / np.linalg.norm(Z)))
        As.append(A.astype(np.float32))
        Bs.append(np_bf16(Uw.T.astype(np.float32)))
    A4 = np.stack(As)[:, None]
    if f in (None, "bf16"):
        Ad, nb = np_bf16(A4), A4.size * 2
    else:
        Ad, nb = np_half_enc(A4, f, "k")
    outs = []
    for w in range(L // W):
        Zh = Ad[w, 0].astype(np.float64) @ Bs[w].astype(np.float64)
        for j in range(W):
            x = Zh[:, j * H * D:(j + 1) * H * D].reshape(T, H, D).transpose(1, 0, 2)
            outs.append(np_bf16(np_rope_fwd(x, c, s) if hi == 0 else x))
    return np.stack(outs), nb, (L // W) * r * W * H * D * 2, errs


def np_xl_calib(Zs, R):
    """xKV 校準（numpy 參考；GPU 版＝cmd_calib_xl）：一個窗口、一半的 token 矩陣清單 [T_i, W·H·D] → 不去平均的二次動差 G＝Σ Zᵀ Z
    的前 R 個特徵向量（由大到小）與全部特徵值。G 的特徵向量＝把所有校準 token 疊起來做 SVD 的右奇異向量（xKV 不去平均）。"""
    np = _np()
    G = sum(z.astype(np.float64).T @ z.astype(np.float64) for z in Zs)
    ev, U = np.linalg.eigh(G)
    return U[:, ::-1][:, :R], ev[::-1]


def np_mc(X, deep, t, gamma):
    np = _np()
    L, _, H, T, D = X.shape
    Y = np.array(X, dtype=np.float32, copy=True)
    nr = mc_nret(gamma, T)
    for l1, l2 in layer_pairs(deep):
        for h in (0, 1):
            x1, x2 = X[l1, h].astype(np.float64), X[l2, h].astype(np.float64)
            n1, n2 = np.linalg.norm(x1, axis=-1, keepdims=True), np.linalg.norm(x2, axis=-1, keepdims=True)
            u1, u2 = x1 / np.maximum(n1, 1e-30), x2 / np.maximum(n2, 1e-30)
            om = np.arccos(np.clip((u1 * u2).sum(-1), -1, 1))
            so = np.sin(om)
            small = so < 1e-6
            sod = np.where(small, 1.0, so)
            w1 = np.where(small, 1 - t, np.sin((1 - t) * om) / sod)
            w2 = np.where(small, t, np.sin(t * om) / sod)
            e = w1[..., None] * u1 + w2[..., None] * u2
            e = e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-30)
            e16 = np_bf16(e).astype(np.float64)
            y1, y2 = e16 * np_bf16(n1), e16 * np_bf16(n2)
            if nr:
                for hh in range(H):
                    ix = np.argsort(-om[hh], kind="stable")[:nr]
                    y1[hh, ix], y2[hh, ix] = x1[hh, ix], x2[hh, ix]
            Y[l1, h], Y[l2, h] = np_bf16(y1), np_bf16(y2)
    return Y


def np_pipeline(stages, X, pos, cx, info):
    """numpy 參考的整條 pipeline（語意和 t_pipeline 相同）。X＝[L,2,H,Tr,D] float32（BF16 值）。回傳 (dec, idx, bytes, extra)。"""
    np = _np()
    L, _, H, Tr, D = X.shape
    C = cx.chunk
    nreg = Tr // C
    st = {s["op"]: s for s in stages}
    ops = [s["op"] for s in stages]
    by, extra = {}, {}
    if "mc" in st:
        Y = np_mc(X, cx.deep, st["mc"]["t"], st["mc"]["gamma"])
        return Y, None, predict_bytes(stages, dict(info, L=L), nreg, C), extra
    if "share" in st:
        Y = np.array(X, dtype=np.float32, copy=True)
        for l1, l2 in layer_pairs(cx.deep):
            Y[l2] = X[l1]
        return Y, None, predict_bytes(stages, dict(info, L=L), nreg, C), extra
    idx = None
    if "drop" in st:
        k = drop_k(st["drop"]["frac"], nreg)
        dropped = set(cx.orders[st["drop"]["method"]][:k])
        kept = [b for b in range(nreg) if b not in dropped]
        idx = np.concatenate([np.arange(b * C, (b + 1) * C) for b in kept])
        by["mask"] = (nreg + 7) // 8
    S = X if idx is None else X[:, :, :, idx]
    P = pos if idx is None else pos[idx]
    Tk = S.shape[3]
    g = st["merge"]["g"] if "merge" in st else 1
    Ts, Tc = Tk // g, C // g
    Sm = np_bf16(S.reshape(L, 2, H, Ts, g, D).astype(np.float64).mean(4)) if g > 1 else S
    fmt = st["q"]["fmt"] if "q" in st else None
    low = st.get("lr") or st.get("xl")
    if low is None:
        if fmt is None:
            dec_s = np.asarray(Sm, dtype=np.float32)
            by["raw"] = Sm.size * 2
        else:
            dec_s, by["q"] = np_full_q(Sm, fmt, Tc)
        if fmt == "vllm_fp8":
            by["shared.scale"] = L * 2 * 4
        return (np.repeat(dec_s, g, axis=3) if g > 1 else dec_s), idx, by, extra
    hv = halves(fmt) if fmt else (None, None)
    xl_first = ops[:2] == ["xl", "drop"]
    outs = []
    for hi, h in enumerate("kv"):
        r = low["rk"] if hi == 0 else low["rv"]
        f = hv[hi]
        if r == 0:
            d, by[f"{h}.raw"] = np_raw_half(S[:, hi], f, h, C)
        elif low["op"] == "lr":
            d, by[f"{h}.coef"], by[f"shared.{h}.basis"] = np_lr_half(S[:, hi], P, hi, r, low["joint"], f, cx)
        elif low.get("cal"):
            d, by[f"{h}.coef"], by[f"shared.{h}.basis"], extra[f"{h}.cal_err"] = np_xl_half_cal(S[:, hi], P, hi, r, low["w"], f, cx)
        else:
            src, sp = (X[:, hi], pos) if xl_first else (S[:, hi], P)
            d, by[f"{h}.coef"], by[f"{h}.basis"], extra[f"{h}.svd_err"] = np_xl_half(src, sp, P, hi, r, low["w"], f, cx,
                                                                                      idx if xl_first else None)
        outs.append(d)
    if fmt == "vllm_fp8":
        by["shared.scale"] = L * 2 * 4
    return np.stack(outs, axis=1), idx, by, extra


def np_attn_out(q, K, V, keep=None):
    """q [m, D]、K／V [T, D]、keep [T] bool → attention 輸出 [m, D]（float64）。"""
    np = _np()
    s = q @ K.T / math.sqrt(q.shape[-1])
    if keep is not None:
        s = np.where(keep[None, :], s, -np.inf)
    s = s - s.max(-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(-1, keepdims=True)
    return p @ V


# ============================================================================ 統計（純 Python／numpy）
def ranks(x):
    """平均名次（同分取平均），1 起。"""
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = ranks(list(x)), ranks(list(y))
    mx, my = statistics.mean(rx), statistics.mean(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def spearman_ci(x, y, n_boot=1000, seed=0):
    """以配置為單位 bootstrap 的 95% CI（L5d §6.2）。"""
    np = _np()
    rho = spearman(x, y)
    if rho is None:
        return None
    x, y = list(x), list(y)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(x), size=len(x))
        r = spearman([x[i] for i in ix], [y[i] for i in ix])
        if r is not None:
            bs.append(r)
    return dict(rho=rho, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n=len(x))


def auroc(scores, labels):
    """Mann–Whitney：分數越高越像『崩』；同分算 0.5。只有一類時 None。"""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    w = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return w / (len(pos) * len(neg))


def boot2(values, n_boot: int = N_BOOT, seed: int = 0):
    """平均、95% 百分位 CI、單邊 p（bootstrap 平均 ≤ 0 的比例）。以題目為單位。"""
    np = _np()
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return None
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(1)
    return dict(mean=float(v.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                p_le0=float((bs <= 0).mean()), n=int(v.size))


def holm(pvals: dict, alpha: float = HOLM_ALPHA) -> dict:
    keys = sorted(pvals, key=lambda k: pvals[k])
    out, stop = {}, False
    for j, k in enumerate(keys):
        thr = alpha / (len(keys) - j)
        rej = (not stop) and pvals[k] <= thr
        if not rej:
            stop = True
        out[k] = dict(p=pvals[k], threshold=thr, reject=bool(rej))
    return out


def pareto_front(points):
    """points＝[(位元組比, 品質（越大越好）, key)] → 前緣上的 key（沒有任何點位元組更少或相同、而且品質更好）。"""
    pts = sorted(points, key=lambda p: (p[0], -p[1], str(p[2])))
    out, best = [], -math.inf
    for r, q, k in pts:
        if q > best:
            out.append(k)
            best = q
    return out


# ============================================================================ 判定（純 Python；X1＝L5d S1-a／S1-b，X2＝M1／M2／完整性）
STOP_TEXT = {
    "S1-a": "（L5d §6.2，代理不可靠）所有代理對 KL 與 Δgold_lp 的 Spearman |ρ| 都 < 0.5（兩個模型都是）→ 代理只當崩潰警報。"
            "我們的 X1 在 GPU 上直接量 teacher forcing，挑前緣用 TF 指標、不用代理，所以 S1-a 只當研究問題 (b)（代理跨家族能不能預測）的結果，不改流程。",
    "S1-b": "（L5d §6.2 改成 TF 版）每個預算（0.25、0.125、0.0625）、兩個模型：最好的組合的損失（−平均 Δgold_lp）≥ 0.95×最好的單一方法的損失"
            "（沒有組合在預算內也算成立；沒有單一方法在預算內而有組合 → 不成立）。三個預算、兩個模型全部成立 → 停，不跑 X2，寫成"
            "「在 TF 層級，疊加沒有加乘」。",
    "S1-c": "（L5d §6.2）S1 實測的無損增益疊在 INT4 KIVI 上 < 1.05× → 第二、三階段拿掉無損這一軸（我們本來就只在 report 另列）。",
    "X1-complete": f"X1 每個模型至少 {MIN_ITEMS} 題做完才挑 X2；不夠 → go＝false、寫 NEEDS_HUMAN。",
    "M1": "（L5d §6.3）每個家族在 ε 內（生成答對率對 E0 的配對差 95% CI 下界 ≥ −2 個百分點）能到的最小位元組 B*；"
          "r＝B*(最好的組合)／B*(最好的單一方法)。兩個模型都 r > 0.95（沒有組合過 ε 也算）→ NO-GO（疊加沒有加乘）；"
          "沒有模型 < 0.85、但有 0.85–0.95 → 停下來問人；任一模型 r < 0.85 → GO（進第三階段）。",
    "M2": "（L5d §6.3）每個預算：主要組合（X1 事先選的前緣第 1 名）− 最好的單一方法（X1 選的），生成答對率配對差。"
          "E0 − 最好的單一方法 < 2 個百分點的預算記成「沒有空間」；有空間的預算全部「差 < 2 個百分點或 CI 含 0」→ 支持 NO-GO。"
          "Holm 校正（3 預算 × 2 模型，單邊 α＝0.025）。",
    "completeness": "（L5d §6.3）最好的組合在 M1／M2 贏，但天真疊加（naive）本身已經一樣好（naive − 主要組合的 CI 含 0 或 ≥ −ε）"
                    "→ 結論改寫成「疊加有用，但不需要聯合分配」。",
    "MLPerf": "（L6 §5.2，只報、不改判定）MLPerf 式門檻：答對率 ≥ E0 的 99%（以及 99.9%），點估計；另報同樣門檻下的 B*。",
}


def eps_tf(c) -> bool:
    return c["d_first_ok"] is not None and c["d_first_ok"]["mean"] >= -EPS


def is_single(c) -> bool:
    return c["n_lossy"] == 1 and c["deployable"] and "single" in c["roles"]


def is_combo(c) -> bool:
    return c["n_lossy"] >= 2 and c["deployable"] and bool({"combo", "naive", "order"} & set(c["roles"]))


def in_budget(c, B) -> bool:
    return c["ratio"] <= BUDGET_TOL * B + 1e-12


def qual(c) -> float:
    return c["d_gold_lp"]["mean"] if c.get("d_gold_lp") else -math.inf


def best_of(cands):
    return max(cands, key=lambda c: (qual(c), -c["ratio"], c["spec"])) if cands else None


def bstar(cands, ok) -> float | None:
    v = [c["ratio"] for c in cands if ok(c)]
    return min(v) if v else None


def judge_x1(tabs: dict, n_items: dict) -> dict:
    """tabs[model]＝{spec: 配置統計}（config_table 的輸出）。回傳每個模型、每個預算的成分，以及 S1-b、完整性與 go。"""
    out = {"models": {}, "rule_text": STOP_TEXT}
    s1b_all = []
    for m, tab in tabs.items():
        cs = list(tab.values())
        S, Cb = [c for c in cs if is_single(c)], [c for c in cs if is_combo(c)]
        M = out["models"][m] = {"n_items": n_items.get(m, 0), "budgets": {}}
        for B in BUDGETS:
            bs, bc = best_of([c for c in S if in_budget(c, B)]), best_of([c for c in Cb if in_budget(c, B)])
            ls = -qual(bs) if bs else None
            lc = -qual(bc) if bc else None
            if B == BUDGETS[0]:
                cond = None                                      # 0.3125 是錨點，不進 S1-b
            elif bc is None:
                cond = True
            elif bs is None:
                cond = False
            else:
                cond = lc >= LOSS_RATIO * ls
            M["budgets"][str(B)] = dict(best_single=bs and bs["spec"], best_single_ratio=bs and bs["ratio"], loss_single=ls,
                                        best_combo=bc and bc["spec"], best_combo_ratio=bc and bc["ratio"], loss_combo=lc,
                                        s1b_holds=cond)
            if cond is not None:
                s1b_all.append(cond)
        bss, bsc = bstar(S, eps_tf), bstar(Cb, eps_tf)
        M["tf_M1"] = dict(bstar_single=bss, bstar_combo=bsc, r=(bsc / bss) if bss and bsc else None,
                          eps_rule="TF 版 ε 內＝first_ok 對 E0 的平均差 ≥ −0.02（點估計；X1 是篩選，判定只看 X2）")
        for th in MLPERF:
            ok = (lambda c, th=th: c["first_ok"] >= th * c["first_ok_e0"])
            b1, b2 = bstar(S, ok), bstar(Cb, ok)
            M[f"tf_M1_mlperf{th}"] = dict(bstar_single=b1, bstar_combo=b2, r=(b2 / b1) if b1 and b2 else None)
    out["S1b_stop"] = bool(s1b_all) and all(s1b_all)
    out["complete"] = all(out["models"][m]["n_items"] >= MIN_ITEMS for m in tabs) and len(tabs) == len(MODEL_INFO)
    out["go"] = out["complete"] and not out["S1b_stop"]
    out["reason"] = ("X1 不完整（" + "、".join(f"{m} {out['models'][m]['n_items']} 題" for m in tabs) + "）") if not out["complete"] \
        else ("S1-b 成立：TF 層級沒有加乘 → 不跑 X2" if out["S1b_stop"] else "S1-b 不成立 → 跑 X2")
    return out


def front_pick(cands, k=K_FRONT, max_same=MAX_SAME_SIG):
    picked, cnt = [], {}
    for c in sorted(cands, key=lambda c: (-qual(c), c["ratio"], c["spec"])):
        sig = c["families"]
        if cnt.get(sig, 0) >= max_same:
            continue
        picked.append(c)
        cnt[sig] = cnt.get(sig, 0) + 1
        if len(picked) >= k:
            break
    return picked


def build_naive_x1(tab: dict, model: str) -> dict:
    """事先登記的規則：drop、lowrank（沒有就用 merge）、quant 三個家族，各取「ε 內（TF）最省位元組的單一設定」，照合法順序直接疊。"""
    fam = {}
    for F in ("drop", "lowrank", "merge", "quant"):
        cands = [c for c in tab.values() if is_single(c) and c["families"] == F and eps_tf(c)]
        if cands:
            fam[F] = min(cands, key=lambda c: (c["ratio"], -qual(c), c["spec"]))
    stages, note = [], []
    if "drop" in fam:
        stages += parse_spec(fam["drop"]["spec"])
    if "lowrank" in fam:
        stages += parse_spec(fam["lowrank"]["spec"])
    elif "merge" in fam:
        stages += parse_spec(fam["merge"]["spec"])
    if "quant" in fam:
        q = parse_spec(fam["quant"]["spec"])
        try:
            validate(stages + q, MODEL_INFO[model], N_BLOCKS - 1)
            stages += q
        except SpecError as ex:
            note.append(f"量化 {fam['quant']['spec']} 疊不上去：{ex}")
    if len(stages) < 2:
        return dict(spec=None, parts={k: v["spec"] for k, v in fam.items()}, note=note + ["可疊的家族不到兩個"])
    validate(stages, MODEL_INFO[model], N_BLOCKS - 1)
    spec = canonical(stages)
    return dict(spec=spec, ratio=predict_ratio(spec, model), parts={k: v["spec"] for k, v in fam.items()}, note=note)


def select_x2(tab: dict, model: str) -> list:
    """事先登記的 X2 挑選規則（x_stack_howto.md §5）。回傳 [{spec, roles, tiers, budgets, why, x1}]。"""
    sel = {}

    def add(spec, role, tiers, budget=None, why=""):
        if spec is None:
            return
        e = sel.setdefault(spec, dict(spec=spec, roles=[], tiers=[], budgets=[], why=[]))
        for x, k in ((role, "roles"), (budget, "budgets")):
            if x is not None and x not in e[k]:
                e[k].append(x)
        for t in tiers:
            if t not in e["tiers"]:
                e["tiers"].append(t)
        if why:
            e["why"].append(why)

    cs = list(tab.values())
    S, Cb = [c for c in cs if is_single(c)], [c for c in cs if is_combo(c)]
    add("none", "anchor_mask0", ["core"], why="MASK0：換 attention 路徑的雜訊")
    add("q:int4_kivi", "anchor_int4", ["core", "128k"], why="第二階段錨點 INT4 KIVI（0.3125）")
    for c in cs:
        if "deploy" in c["roles"]:
            add(c["spec"], "deploy", ["core", "128k"], why="常見部署對照（L6 §7.1）")
        if "deploy_broken" in c["roles"]:
            add(c["spec"], "deploy_broken", ["core"], why="已知會壞的部署對照（L6；只報）")
    for B in BUDGETS:
        Sb, Cbb = [c for c in S if in_budget(c, B)], [c for c in Cb if in_budget(c, B)]
        bs = best_of(Sb)
        fr = front_pick(Cbb)
        if B == BUDGETS[0]:
            add(bs and bs["spec"], "best_single", ["ext"], B, "錨點預算的最好單一方法")
            for c in fr[:2]:
                add(c["spec"], "front", ["ext"], B, "錨點預算的前緣")
            continue
        add(bs and bs["spec"], "best_single", ["core", "128k"], B, "這個預算最好的單一方法（各家族掃完超參數後取最好）")
        for j, c in enumerate(fr):
            add(c["spec"], "primary" if j == 0 else "front", ["core", "128k"] if j == 0 else ["core"] if j == 1 else ["ext"], B,
                "前緣第 1 名（M2 的主要組合，事先選定）" if j == 0 else f"前緣第 {j + 1} 名")
        nv = best_of([c for c in Cbb if "naive" in c["roles"]])
        add(nv and nv["spec"], "naive", ["core"], B, "天真疊加（事先登記的那組裡這個預算最好的）")
        rnd = [c for c in cs if c["families"] == "drop" and c["spec"].startswith("drop:RND0:") and in_budget(c, B)]
        if rnd:
            add(max(rnd, key=lambda c: c["ratio"])["spec"], "random_drop", ["core"], B, "同保留率的隨機丟")
        for F in ("quant", "drop", "lowrank", "xlayer", "merge"):
            bf = best_of([c for c in Sb if c["families"] == F])
            add(bf and bf["spec"], f"family_best:{F}", ["ext"], B, f"{F} 家族在這個預算最好的單一方法")
    nx = build_naive_x1(tab, model)
    if nx.get("spec"):
        add(nx["spec"], "naive_x1", ["core"], why=f"各家族 ε 內最省的單一設定直接疊：{nx['parts']}")
    for c in sorted(cs, key=lambda c: c["spec"]):                  # X3 的解碼段：不在 core 的放 x3 這一層（37c，可不跑）
        seg = x3_segment(c["spec"], model)[0]
        if seg is not None:
            add(c["spec"], "x3_segment", ["x3"], why=f"X3 解碼段 {seg}（x2_feasible.csv）")
    for e in sel.values():                                          # 32K 的層互斥（core > x3 > ext），同一個組合只量一次；128k 另算
        t = set(e["tiers"])
        if "core" in t:
            t -= {"x3", "ext"}
        elif "x3" in t:
            t -= {"ext"}
        e["tiers"] = sorted(t)
    for e in sel.values():
        c = tab.get(e["spec"])
        e["x1"] = None if c is None else dict(ratio=c["ratio"], d_gold_lp=c["d_gold_lp"], d_first_ok=c["d_first_ok"],
                                                kl_median=c["kl_median"], n=c["n"])
        if c is None:
            e["x1_ratio_predicted"] = predict_ratio(e["spec"], model)
    return list(sel.values())


def judge_x2(acc: dict, sel: dict) -> dict:
    """acc[model][spec]＝{ratio, roles, budgets, diff（boot2：配對差 vs E0）, acc, acc_e0, n_lossy, deployable}（只用 32K）。"""
    out = {"models": {}, "rule_text": STOP_TEXT}
    rs, pv = {}, {}
    for m, A in acc.items():
        M = out["models"][m] = {}
        def ok(c):
            return c["diff"] is not None and c["diff"]["lo"] >= -EPS
        singles = [c for c in A.values() if c["n_lossy"] == 1 and c["deployable"] and not ({"deploy_broken", "random_drop"} & set(c["roles"]))]
        combos = [c for c in A.values() if c["n_lossy"] >= 2 and c["deployable"] and "random_drop" not in c["roles"]]
        b1, b2 = bstar(singles, ok), bstar(combos, ok)
        r = (b2 / b1) if (b1 and b2) else (math.inf if b1 and not b2 else None)
        M["M1"] = dict(bstar_single=b1, bstar_combo=b2, r=r)
        rs[m] = r
        for th in MLPERF:
            okm = (lambda c, th=th: c["acc"] >= th * c["acc_e0"])
            x1, x2 = bstar(singles, okm), bstar(combos, okm)
            M[f"M1_mlperf{th}"] = dict(bstar_single=x1, bstar_combo=x2, r=(x2 / x1) if x1 and x2 else None)
        M["M2"] = {}
        for B in BUDGETS[1:]:
            prim = next((c for c in A.values() if "primary" in c["roles"] and B in c["budgets"]), None)
            bs = next((c for c in A.values() if "best_single" in c["roles"] and B in c["budgets"]), None)
            nv = next((c for c in A.values() if "naive" in c["roles"] and B in c["budgets"]), None)
            e = M["M2"][str(B)] = dict(primary=prim and prim["spec"], best_single=bs and bs["spec"], naive=nv and nv["spec"])
            if not (prim and bs):
                e["status"] = "NOT_MEASURED（缺主要組合或最好的單一方法）"
                continue
            e["room"] = bool(bs["acc_e0"] - bs["acc"] >= EPS)
            d = paired_diff(prim, bs)
            e["diff_primary_minus_single"] = d
            if d is not None:
                pv[f"{m}|{B}"] = d["p_le0"]
            e["supports_nogo"] = (None if not e["room"] else bool(d is None or d["mean"] < EPS or d["lo"] <= 0 <= d["hi"]))
            if nv:
                dn = paired_diff(nv, prim)
                e["naive_minus_primary"] = dn
                e["naive_as_good"] = None if dn is None else bool(dn["lo"] <= 0 <= dn["hi"] or dn["mean"] >= -EPS)
    out["holm"] = holm(pv) if pv else {}
    for k, v in out["holm"].items():
        m, B = k.split("|")
        out["models"][m]["M2"][B]["holm"] = v
    vals = [rs[m] for m in rs]
    if not vals or any(v is None for v in vals):
        verdict = "UNDEFINED（有模型沒有過 ε 的單一方法或沒有資料）"
    elif all(v > R_NOGO for v in vals):
        verdict = "NO-GO"
    elif any(v < R_GO for v in vals):
        verdict = "GO"
    else:
        verdict = "ASK"
    out["M1_verdict"] = verdict
    m2 = [e.get("supports_nogo") for M in out["models"].values() for e in M["M2"].values() if e.get("room")]
    out["M2_supports_nogo"] = (all(m2) if m2 else None)
    out["M2_no_room_anywhere"] = not m2
    out["completeness_flags"] = {f"{m}|{B}": e.get("naive_as_good") for m, M in out["models"].items() for B, e in M["M2"].items()
                                 if e.get("naive_as_good") is not None}
    return out


def paired_diff(a, b):
    """兩個配置的逐題配對差（a − b），只用兩邊都有的題。a／b 帶 per_item={item_id: score}。"""
    common = sorted(set(a["per_item"]) & set(b["per_item"]))
    if not common:
        return None
    return boot2([a["per_item"][i] - b["per_item"][i] for i in common])


# ============================================================================ torch 實作（GPU）
def check_info(km, info: dict):
    got = dict(L=km.L, H=km.H, Hq=km.Hq, D=km.D)
    if got != {k: info[k] for k in got}:
        raise RuntimeError(f"MODEL_INFO {info} 和模型 {got} 對不上（規則 6）")


def t_rope_inv(y, cos, sin):
    """RoPE 後的 K [..., T, D]（BF16）→ RoPE 前（FP32）：用同一組 BF16 cos/sin 的 2×2 旋轉矩陣精確求反（除以 c²＋s²），
    和 m16_kvstruct.rope_invert 同一個式子。cos/sin＝[1, T, D]（cat(freqs, freqs)，前後半相同）。"""
    import torch
    h = y.shape[-1] // 2
    c, s = cos[0, :, :h].float(), sin[0, :, :h].float()
    det = c * c + s * s
    yf = y.float()
    y1, y2 = yf[..., :h], yf[..., h:]
    return torch.cat(((y1 * c + y2 * s) / det, (y2 * c - y1 * s) / det), dim=-1)


def t_rope_fwd(x, cos, sin, apply_rope):
    """FP32 [H, T, D] → BF16 後用模型自己的 apply_rope（和 forward_span 同一個函式、同一組 BF16 cos/sin）。"""
    import torch
    xb = x.to(torch.bfloat16).unsqueeze(0)
    _, k = apply_rope(xb, xb, cos, sin)
    return k[0]


class TCtx:
    """一題的 GPU 上下文：模型的 BF16 cos/sin（位置 0..n_pos−1，和 forward_span 同一組）、校準基底（GPU）、drop 的區內順序、deep 層。"""

    def __init__(self, km, n_pos: int, calib=None, orders=None, deep=None, chunk: int = CHUNK):
        import torch
        self.km, self.chunk, self.calib, self.orders, self.deep = km, chunk, calib, orders or {}, deep
        m = km.hf.model
        x = torch.zeros(1, 1, km.D, dtype=torch.bfloat16, device=km.device)
        with torch.inference_mode():
            self.cos, self.sin = m.rotary_emb(x, torch.arange(n_pos, device=km.device).view(1, -1))
        self.apply_rope = km.apply_rope
        self.xlcal = None          # 10/21：xKV 校準基底（load_calib_xl）；沒有 :cal 的組合用不到
        self.xl_cache = None       # 10/21：gen --xl-cache 時的同題 SVD 快取（xl_cache_plan）；None＝照舊每個組合各算一次

    def cs(self, P):
        return self.cos[:, P], self.sin[:, P]


def t_vllm_fp8(x):
    """vLLM 未校準 fp8（per-tensor、scale＝1.0、e4m3fnuz）：先截在 ±240 再轉型（torch 的 fp8 轉型超出範圍會變 NaN，m10_codec 檔頭）。"""
    import torch
    q = x.float().clamp(-FNUZ_MAX, FNUZ_MAX).to(torch.float8_e4m3fnuz)
    return q.float().to(torch.bfloat16), q.numel()


def t_halves_enc(t, f, role):
    """原始 KV 的一半或係數 [A,B,T,R] → (解碼後 FP32, 位元組)。f＝None／bf16：存 BF16；其餘直接用 m10_codec_x.enc_half／dec_half。"""
    import torch
    import m10_codec_x as X10
    if f in (None, "bf16"):
        b = t.to(torch.bfloat16)
        return b.float(), b.numel() * 2
    f = ALIAS.get(f, f)
    p = X10.enc_half(t.contiguous(), f, role)
    nb = sum(v.numel() * v.element_size() for v in p.values())
    return X10.dec_half(p, f, role, tuple(t.shape)), nb


def t_full_q(Sm, fmt, Tc):
    import torch
    import m10_codec_x as X10
    outs, nb = [], 0
    for c0 in range(0, Sm.shape[3], Tc):
        ch = Sm[:, :, :, c0:c0 + Tc].contiguous()
        if fmt == "vllm_fp8":
            y, b = t_vllm_fp8(ch)
        else:
            e = X10.encode(ch, fmt)
            b, y = e.nbytes, X10.decode(e)
        outs.append(y)
        nb += b
    return torch.cat(outs, dim=3), nb


def t_raw_half(t, f, role, C):
    import torch
    if f is None:
        return t, t.numel() * 2
    outs, nb = [], 0
    for c0 in range(0, t.shape[2], C):
        d, b = t_halves_enc(t[:, :, c0:c0 + C], f, role)
        outs.append(d.to(torch.bfloat16))
        nb += b
    return torch.cat(outs, dim=2), nb


def t_lr_half(t, P, hi, r, joint, f, cx):
    """校準 PCA 的一半：t [L,H,T,D] BF16（留下的 token）、P＝絕對位置。回傳 (解碼 BF16, 係數位元組, 共用基底位元組)。"""
    import torch
    L, H, T, D = t.shape
    h = "kv"[hi]
    cos, sin = cx.cs(P)
    mu = cx.calib[f"{h}_mean"]
    coefs = []
    for l in range(L):
        xp = t_rope_inv(t[l], cos, sin) if hi == 0 else t[l].float()
        if joint:
            U = cx.calib[f"{h}_Uj"][l][:, :r * H]
            if U.shape[1] != r * H:
                raise RuntimeError(f"聯合基底只存到 {cx.calib[f'{h}_Uj'].shape[-1]} 維，要 {r * H}")
            coefs.append(((xp.permute(1, 0, 2).reshape(T, H * D) - mu[l].reshape(1, H * D)) @ U).unsqueeze(0))
        else:
            coefs.append((xp - mu[l][:, None, :]) @ cx.calib[f"{h}_U"][l][:, :, :r])
    C4 = torch.stack(coefs)
    Cd, nb = t_halves_enc(C4, f, "k")
    del C4, coefs
    outs = []
    for l in range(L):
        if joint:
            Zh = (Cd[l, 0] @ cx.calib[f"{h}_Uj"][l][:, :r * H].T + mu[l].reshape(1, H * D)).reshape(T, H, D).permute(1, 0, 2)
        else:
            Zh = Cd[l] @ cx.calib[f"{h}_U"][l][:, :, :r].transpose(-1, -2) + mu[l][:, None, :]
        outs.append(t_rope_fwd(Zh, cos, sin, cx.apply_rope) if hi == 0 else Zh.to(torch.bfloat16))
    sh = (L * (H * D * r * H + H * D) if joint else L * H * (D * r + D)) * 2
    return torch.stack(outs), nb, sh


def t_xl_half(src, srcpos, P, hi, rhe, W, f, cx, keep_idx):
    """xKV：src [L,H,Ts,D]＝要分解的 token（xl 在 drop 前時是全部 token）；P＝留下的 token 的位置。回傳 (解碼 BF16, 係數位元組, 基底位元組)。"""
    import torch
    L, H, T, D = src.shape
    r = rhe * W * H
    c0, s0 = cx.cs(srcpos)
    As, Bs = [], []
    for w in range(L // W):
        Z = torch.cat([(t_rope_inv(src[l], c0, s0) if hi == 0 else src[l].float()).permute(1, 0, 2).reshape(T, H * D)
                       for l in range(w * W, (w + 1) * W)], dim=1)
        torch.manual_seed(XL_SEED + w)
        U, S, V = torch.svd_lowrank(Z, q=min(r + XL_OVERSAMPLE, min(Z.shape)), niter=XL_NITER)
        A = U[:, :r] * S[:r]
        if keep_idx is not None:
            A = A.index_select(0, keep_idx)
        As.append(A)
        Bs.append(V[:, :r].T.contiguous().to(torch.bfloat16))
        del Z, U, S, V
    A4 = torch.stack(As).unsqueeze(1)
    Ad, nb = t_halves_enc(A4, f, "k")
    Tk = A4.shape[2]
    cos, sin = cx.cs(P)
    outs = []
    for w in range(L // W):
        Zh = Ad[w, 0] @ Bs[w].float()
        for j in range(W):
            x = Zh[:, j * H * D:(j + 1) * H * D].reshape(Tk, H, D).permute(1, 0, 2)
            outs.append(t_rope_fwd(x, cos, sin, cx.apply_rope) if hi == 0 else x.to(torch.bfloat16))
    return torch.stack(outs), nb, (L // W) * r * W * H * D * 2


def t_xl_half_cal(src, P, hi, rhe, W, f, cx):
    """xKV、事先校準的基底（10/21）：和 t_xl_half 同樣的窗口與 Z（K 逆 RoPE、不去平均），但基底＝cx.xlcal 的 U[w][:, :r]（校準題算好、
    所有題共用），係數＝Z·U（FP32）；重建用 BF16 的基底（和每題 SVD 的基底同一個存法）。基底跨請求共用 → 位元組帳另列 shared。
    回傳 (解碼 BF16, 係數位元組, 共用基底位元組)。"""
    import torch
    L, H, T, D = src.shape
    r = rhe * W * H
    U = cx.xlcal[f"{'kv'[hi]}_U"]
    if U.shape[-1] < r or U.shape[0] != L // W or U.shape[1] != W * H * D:
        raise RuntimeError(f"校準基底形狀 {tuple(U.shape)} 不夠或對不上（要 [{L // W}, {W * H * D}, ≥{r}]）")
    cos, sin = cx.cs(P)
    As, Bs = [], []
    for w in range(L // W):
        Z = torch.cat([(t_rope_inv(src[l], cos, sin) if hi == 0 else src[l].float()).permute(1, 0, 2).reshape(T, H * D)
                       for l in range(w * W, (w + 1) * W)], dim=1)
        Uw = U[w, :, :r]
        As.append(Z @ Uw)
        Bs.append(Uw.T.contiguous().to(torch.bfloat16))
        del Z
    A4 = torch.stack(As).unsqueeze(1)
    Ad, nb = t_halves_enc(A4, f, "k")
    outs = []
    for w in range(L // W):
        Zh = Ad[w, 0] @ Bs[w].float()
        for j in range(W):
            x = Zh[:, j * H * D:(j + 1) * H * D].reshape(T, H, D).permute(1, 0, 2)
            outs.append(t_rope_fwd(x, cos, sin, cx.apply_rope) if hi == 0 else x.to(torch.bfloat16))
    return torch.stack(outs), nb, (L // W) * r * W * H * D * 2


def xl_cache_plan(configs) -> dict:
    """gen --xl-cache（10/21）：數每個 (K/V, 每 head 等效秩, W) 的每題 SVD 在這題的配置裡會用幾次；只快取用 ≥2 次的、最後一次用完就丟。
    只算「沒有 drop、不是 :cal」的 xl（src＝整個區內原始 KV，每個配置前都先還原，所以同一個鍵的 SVD 輸入逐位元相同）。"""
    uses = {}
    for c in configs:
        st = {s["op"]: s for s in c["stages"]}
        low = st.get("xl")
        if low is None or low.get("cal") or "drop" in st:
            continue
        for hi, r in ((0, low["rk"]), (1, low["rv"])):
            if r:
                k = (hi, r, low["w"])
                uses[k] = uses.get(k, 0) + 1
    return dict(uses=uses, store={}, hits=0, misses=0)


def t_xl_half_cached(src, srcpos, P, hi, rhe, W, f, cx):
    """t_xl_half（沒有 keep_idx 的情形）＋同一題內的 SVD 快取（10/21，gen --xl-cache）。SVD 與之後的量化、重建逐行照 t_xl_half；
    gpu-selftest 的 xl_cache_eq_uncached 核對和 t_xl_half 逐位元相同。"""
    import torch
    L, H, T, D = src.shape
    r = rhe * W * H
    C = cx.xl_cache
    key = (hi, rhe, W)
    ent = C["store"].get(key)
    if ent is None:
        c0, s0 = cx.cs(srcpos)
        As, Bs = [], []
        for w in range(L // W):
            Z = torch.cat([(t_rope_inv(src[l], c0, s0) if hi == 0 else src[l].float()).permute(1, 0, 2).reshape(T, H * D)
                           for l in range(w * W, (w + 1) * W)], dim=1)
            torch.manual_seed(XL_SEED + w)
            U, S, V = torch.svd_lowrank(Z, q=min(r + XL_OVERSAMPLE, min(Z.shape)), niter=XL_NITER)
            As.append(U[:, :r] * S[:r])
            Bs.append(V[:, :r].T.contiguous().to(torch.bfloat16))
            del Z, U, S, V
        C["misses"] += 1
        if C["uses"].get(key, 0) > 1:
            C["store"][key] = (As, Bs)
    else:
        As, Bs = ent
        C["hits"] += 1
    C["uses"][key] = C["uses"].get(key, 1) - 1
    if C["uses"][key] <= 0:
        C["store"].pop(key, None)
    A4 = torch.stack(As).unsqueeze(1)
    Ad, nb = t_halves_enc(A4, f, "k")
    Tk = A4.shape[2]
    cos, sin = cx.cs(P)
    outs = []
    for w in range(L // W):
        Zh = Ad[w, 0] @ Bs[w].float()
        for j in range(W):
            x = Zh[:, j * H * D:(j + 1) * H * D].reshape(Tk, H, D).permute(1, 0, 2)
            outs.append(t_rope_fwd(x, cos, sin, cx.apply_rope) if hi == 0 else x.to(torch.bfloat16))
    return torch.stack(outs), nb, (L // W) * r * W * H * D * 2


def t_mc(X, deep, t, gamma):
    import torch
    L, _, H, T, D = X.shape
    Y = X.clone()
    nr = mc_nret(gamma, T)
    for l1, l2 in layer_pairs(deep):
        for h in (0, 1):
            x1, x2 = X[l1, h].float(), X[l2, h].float()
            n1, n2 = x1.norm(dim=-1, keepdim=True), x2.norm(dim=-1, keepdim=True)
            u1, u2 = x1 / n1.clamp_min(1e-30), x2 / n2.clamp_min(1e-30)
            om = torch.arccos((u1 * u2).sum(-1).clamp(-1, 1))
            so = torch.sin(om)
            small = so < 1e-6
            sod = torch.where(small, torch.ones_like(so), so)
            w1 = torch.where(small, torch.full_like(so, 1 - t), torch.sin((1 - t) * om) / sod)
            w2 = torch.where(small, torch.full_like(so, t), torch.sin(t * om) / sod)
            e = w1[..., None] * u1 + w2[..., None] * u2
            e = (e / e.norm(dim=-1, keepdim=True).clamp_min(1e-30)).to(torch.bfloat16).float()
            y1, y2 = e * n1.to(torch.bfloat16).float(), e * n2.to(torch.bfloat16).float()
            if nr:
                ix = torch.topk(om, nr, dim=-1).indices.unsqueeze(-1).expand(-1, -1, D)
                y1.scatter_(1, ix, x1.gather(1, ix))
                y2.scatter_(1, ix, x2.gather(1, ix))
            Y[l1, h], Y[l2, h] = y1.to(torch.bfloat16), y2.to(torch.bfloat16)
    return Y


def t_pipeline(stages, X, pos, cx, info):
    """X：區內原始 KV [L,2,H,Tr,D] BF16（KV 緩衝的視圖，不會被改）；pos：區內每個 token 的絕對位置 [Tr]（GPU long）。
    回傳 (dec＝留下的 token 解碼後 [L,2,H,Tk,D] BF16（沒有任何改動時就是 X 本身）, idx＝留下的 token 在區內的索引（None＝全部）,
    bytes＝每個元件的實際位元組, dropped＝丟掉的區內 block)。"""
    import torch
    L, _, H, Tr, D = X.shape
    C = cx.chunk
    nreg = Tr // C
    st = {s["op"]: s for s in stages}
    ops = [s["op"] for s in stages]
    by = {}
    if "mc" in st:
        Y = t_mc(X, cx.deep, st["mc"]["t"], st["mc"]["gamma"])
        return Y, None, predict_bytes(stages, dict(info, L=L), nreg, C), []     # mc 的位元組由形狀決定（存的形式見 predict_bytes）
    if "share" in st:
        Y = X.clone()
        for l1, l2 in layer_pairs(cx.deep):
            Y[l2] = X[l1]
        return Y, None, predict_bytes(stages, dict(info, L=L), nreg, C), []
    idx, dropped = None, []
    if "drop" in st:
        k = drop_k(st["drop"]["frac"], nreg)
        dropped = list(cx.orders[st["drop"]["method"]][:k])
        kept = [b for b in range(nreg) if b not in set(dropped)]
        idx = torch.cat([torch.arange(b * C, (b + 1) * C, device=X.device) for b in kept])
        by["mask"] = (nreg + 7) // 8
    S = X if idx is None else X.index_select(3, idx)
    P = pos if idx is None else pos.index_select(0, idx)
    Tk = S.shape[3]
    g = st["merge"]["g"] if "merge" in st else 1
    Ts, Tc = Tk // g, C // g
    Sm = S.float().reshape(L, 2, H, Ts, g, D).mean(4).to(torch.bfloat16) if g > 1 else S
    fmt = st["q"]["fmt"] if "q" in st else None
    low = st.get("lr") or st.get("xl")
    if low is None:
        if fmt is None:
            dec_s = Sm
            by["raw"] = Sm.numel() * 2
        else:
            dec_s, by["q"] = t_full_q(Sm, fmt, Tc)
        if fmt == "vllm_fp8":
            by["shared.scale"] = L * 2 * 4
        return (dec_s.repeat_interleave(g, dim=3) if g > 1 else dec_s), idx, by, dropped
    hv = halves(fmt) if fmt else (None, None)
    xl_first = ops[:2] == ["xl", "drop"]
    outs = []
    for hi, h in enumerate("kv"):
        r = low["rk"] if hi == 0 else low["rv"]
        f = hv[hi]
        if r == 0:
            d, by[f"{h}.raw"] = t_raw_half(S[:, hi], f, h, C)
        elif low["op"] == "lr":
            d, by[f"{h}.coef"], by[f"shared.{h}.basis"] = t_lr_half(S[:, hi], P, hi, r, low["joint"], f, cx)
        elif low.get("cal"):                                   # 10/21：事先校準的基底
            d, by[f"{h}.coef"], by[f"shared.{h}.basis"] = t_xl_half_cal(S[:, hi], P, hi, r, low["w"], f, cx)
        elif getattr(cx, "xl_cache", None) is not None and "drop" not in st:   # 10/21：gen --xl-cache（預設關）
            d, by[f"{h}.coef"], by[f"{h}.basis"] = t_xl_half_cached(S[:, hi], P, P, hi, r, low["w"], f, cx)
        else:
            src, sp = (X[:, hi], pos) if xl_first else (S[:, hi], P)
            d, by[f"{h}.coef"], by[f"{h}.basis"] = t_xl_half(src, sp, P, hi, r, low["w"], f, cx, idx if xl_first else None)
        outs.append(d)
    return torch.stack(outs, dim=1), idx, by, dropped


def t_relerr(X, dec, idx):
    """區內：‖X̂−X‖／‖X‖（K、V 分開；丟掉的 token 算全部遺失＝X̂ 為 0）；K 另算中心化（分母減掉每（層, head, channel）的 token 平均）。"""
    import torch
    L, _, H, Tr, D = X.shape
    acc = torch.zeros(5, dtype=torch.float64, device=X.device)
    dm = None
    if idx is not None:
        dm = torch.ones(Tr, dtype=torch.bool, device=X.device)
        dm[idx] = False
    for l in range(L):
        for h in (0, 1):
            x = X[l, h].float()
            xs = x if idx is None else x.index_select(1, idx)
            e = (dec[l, h].float() - xs).pow(2).sum(dtype=torch.float64)
            if dm is not None:
                e = e + x[:, dm].pow(2).sum(dtype=torch.float64)
            acc[2 * h] += e
            acc[2 * h + 1] += x.pow(2).sum(dtype=torch.float64)
            if h == 0:
                acc[4] += (x - x.mean(1, keepdim=True)).pow(2).sum(dtype=torch.float64)
    a = acc.tolist()
    f = (lambda n, d: round(math.sqrt(n / d), 7) if d > 0 else "")
    return dict(relerr_k=f(a[0], a[1]), relerr_v=f(a[2], a[3]), cenerr_k=f(a[0], a[4]))


_WO = {}


class Proxy:
    """attention 輸出誤差（L5d P2／P2w）：第 1 輪問題的 query（E0 的 teacher forcing 時用 forward hook 抓 q_proj 輸出、套 RoPE）對 context 的 key。
    每層獨立算（q 用 E0 的，看不到誤差跨層累積）；只看 context [0, n)（問題自己的 key 兩邊一樣，略過）。"""

    def __init__(self, km, qcap: dict, layers: list, n: int, wo: bool = True):
        self.km, self.n, self.layers = km, n, layers
        self.rep = km.Hq // km.H
        self.qg = {}
        for l in layers:
            Hq, nq, D = qcap[l].shape
            self.qg[l] = qcap[l].reshape(km.H, self.rep * nq, D)
        self.nq = nq
        self.wo = None
        if wo:
            for l in layers:
                if (id(km), l) not in _WO:
                    _WO[(id(km), l)] = km.hf.model.layers[l].self_attn.o_proj.weight.float()
            self.wo = {l: _WO[(id(km), l)] for l in layers}
        self.ref = {l: self._out(l, None) for l in layers}

    def _out(self, l, keep):
        import torch
        km = self.km
        K, V = km.kv[l, 0, :, :self.n].float(), km.kv[l, 1, :, :self.n].float()
        s = torch.matmul(self.qg[l], K.transpose(1, 2)) / math.sqrt(km.D)
        if keep is not None:
            s.masked_fill_(~keep.view(1, 1, -1), float("-inf"))
        o = torch.matmul(s.softmax(-1), V)
        ow = None
        if self.wo is not None:
            flat = o.reshape(km.Hq, self.nq, km.D).permute(1, 0, 2).reshape(self.nq, km.Hq * km.D)
            ow = flat @ self.wo[l].T
        return o, ow

    def measure(self, keep) -> dict:
        e, ew = [], []
        for l in self.layers:
            o, ow = self._out(l, keep)
            o0, ow0 = self.ref[l]
            e.append(float((o - o0).norm() / o0.norm().clamp_min(1e-30)))
            if ow is not None:
                ew.append(float((ow - ow0).norm() / ow0.norm().clamp_min(1e-30)))
        return dict(attn_err=round(statistics.mean(e), 7), attn_err_max=round(max(e), 7),
                    attn_err_wo=round(statistics.mean(ew), 7) if ew else "")


def capture_q(ans, st, nq_cap: int):
    """E0 的 teacher forcing（m10_sweep.Item.score）時，用 forward hook 抓每層 q_proj 的輸出（不改 m7_model），取第 1 輪問題的最後
    nq_cap 個 token，套模型自己的 RoPE（位置＝ctx_len＋j）→ {層: [Hq, nq, D] FP32}。回傳 (lp0, q)。"""
    import torch
    km = ans.km
    layers = km.hf.model.layers
    store = {l: [] for l in range(km.L)}
    hs = [layers[l].self_attn.q_proj.register_forward_hook(lambda mod, inp, out, l=l: store[l].append(out.detach().clone()))
          for l in range(km.L)]
    try:
        lp0 = st.score()
    finally:
        for h in hs:
            h.remove()
    nqry = len(st.it["turns"][0]["query_ids"])
    j0 = max(0, nqry - nq_cap)
    pos = torch.arange(st.ctx_len + j0, st.ctx_len + nqry, device=km.device)
    x = torch.zeros(1, 1, km.D, dtype=torch.bfloat16, device=km.device)
    out = {}
    with torch.inference_mode():
        cos, sin = km.hf.model.rotary_emb(x, pos.view(1, -1))
        for l in range(km.L):
            Q = torch.cat(store[l], dim=1)[0]
            if Q.shape[0] != len(st.seq):
                raise RuntimeError(f"hook 抓到 {Q.shape[0]} 個 token，seq 有 {len(st.seq)} 個")
            q = Q[j0:nqry].view(nqry - j0, km.Hq, km.D).permute(1, 0, 2).unsqueeze(0)
            qr, _ = km.apply_rope(q, q, cos, sin)
            out[l] = qr[0].float()
    return lp0, out


def restore_blocks(km, st, blocks):
    for b in blocks:
        km.chunk_view(b).copy_(st.bf16[b])
        st.src[b] = "bf16"


def load_calib(path: str, km, model: str, eval_keys: set | None = None):
    import torch
    if not path or not os.path.exists(path):
        raise SystemExit(f"沒有校準檔 {path}（先跑 calib）")
    d = torch.load(path, map_location=km.device)
    meta = d["meta"]
    if meta["model"] != model or (meta["L"], meta["H"], meta["D"]) != (km.L, km.H, km.D):
        raise SystemExit(f"校準檔 {path} 是 {meta['model']} {meta['L']}/{meta['H']}/{meta['D']}，和這個模型對不上")
    if eval_keys:
        ov = {tuple(x) for x in meta["item_keys"]} & eval_keys
        if ov:
            raise SystemExit(f"校準題和評測題重疊：{sorted(ov)[:5]}")
    return d


def calib_for_layers(calib, lo: int, hi: int):
    return {k: (v[lo:hi] if hasattr(v, "shape") else v) for k, v in calib.items()}


def load_calib_xl(path: str, km, model: str, eval_keys: set | None = None, need_w: int = 4):
    """10/21：xKV 校準基底（cmd_calib_xl 寫的）。核對模型、形狀、窗口，以及校準題和評測題不重疊（重疊就停）。"""
    import torch
    if not path or not os.path.exists(path):
        raise SystemExit(f"沒有 xKV 校準檔 {path}（先跑 calib-xl）")
    d = torch.load(path, map_location=km.device)
    meta = d["meta"]
    if meta["model"] != model or (meta["L"], meta["H"], meta["D"]) != (km.L, km.H, km.D) or meta["W"] != need_w:
        raise SystemExit(f"xKV 校準檔 {path} 是 {meta['model']} L{meta['L']}/H{meta['H']}/D{meta['D']}/W{meta['W']}，和這個模型／窗口對不上")
    if eval_keys:
        ov = {tuple(x) for x in meta["item_keys"]} & eval_keys
        if ov:
            raise SystemExit(f"xKV 校準題和評測題重疊：{sorted(ov)[:5]}")
    return d


def calib_rids(calib, calib_xl) -> str:
    """逐列的 calib_run_id：沒有 xKV 校準時照舊（PCA 校準的 run_id）；有的話接上「xl:<run_id>」。"""
    r = (calib or {}).get("meta", {}).get("run_id", "")
    if calib_xl is None:
        return r
    return ";".join(x for x in (r, "xl:" + calib_xl["meta"]["run_id"]) if x)


def plan_interleaved(tasks, n, seed0):
    """題目照任務輪流排（T2 第 0 題、T3 第 0 題、…）：時間預算用完時做完的題在任務間是平衡的。題目本身＝m10_quality.plan_items。"""
    from m10_quality import plan_items
    per = {t: plan_items([t], n, seed0) for t in tasks}
    return [per[t][k] for k in range(n) for t in tasks]


def done_union(out: str) -> set:
    """同一組 shard（檔名 …_s<i>.csv[.gz]）裡任何一個寫完的題都算做完（續跑時 shard 數可以換）。"""
    pat = re.sub(r"_s\d+(\.csv(\.gz)?)$", r"_s*\1", out)
    paths = sorted(set(glob.glob(pat)) | {out})
    s = set()
    for p in paths:
        s |= done_items(p)
    return s


@contextlib.contextmanager
def reuse_item(SW, st):
    """S2.run_item 開頭的 SW.Item(ans, it) 改成回傳這一題已經 prefill 好的 Item（省一次 32K prefill）。只在這個行程裡暫時換掉
    m10_sweep.Item 這個名字，不改任何檔案；Item.metrics 等靜態方法照舊（子類別繼承）。"""
    orig = SW.Item
    calls = []

    class _Reuse(orig):
        def __new__(cls, ans, it):
            calls.append(it["item_id"])
            return st

    SW.Item = _Reuse
    try:
        yield calls
    finally:
        SW.Item = orig


# ============================================================================ calib
def calib_settings(a) -> dict:
    return dict(model=a.model, tasks=list(a.tasks), n=a.n, seed0=a.seed0, params_file=os.path.relpath(a.params_file, REPO),
                skip_block0=True, joint_rank_max_per_head=a.joint_rank_max)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def rope_probe(km, kpre, tctx, pos0: int) -> dict:
    """規則 6（資料自身核對）：第 0 層 k_proj 的輸出（hook 抓的 RoPE 前 K）vs 我們對 KV 緩衝裡 RoPE 後 K 的逆旋轉。對照錯的寫法要明顯更差。"""
    import torch
    T = kpre.shape[0]
    pre = kpre.view(T, km.H, km.D).permute(1, 0, 2).float()
    post = km.kv[0, 0, :, pos0:pos0 + T]
    P = torch.arange(pos0, pos0 + T, device=km.device)

    def rel(x):
        return float((x - pre).norm() / pre.norm())
    c, s = tctx.cs(P)
    res = dict(correct=rel(t_rope_inv(post, c, s)), no_inverse=rel(post.float()),
               shifted_pos=rel(t_rope_inv(post, *tctx.cs(P + 1))), wrong_sign=rel(t_rope_inv(post, c, -s)))
    res["pass"] = bool(res["correct"] <= 0.02 and res["correct"] < 0.5 * min(res["no_inverse"], res["shifted_pos"], res["wrong_sign"]))
    return res


def cmd_calib(a):
    out = a.out or os.path.join(DATA, f"calib_{a.model}.pt")
    side = out + ".json"
    settings = calib_settings(a)
    if os.path.exists(out) and not a.force:
        old = json.load(open(side)) if os.path.exists(side) else {}
        if old.get("settings") == settings and old.get("sha256") == sha256(out):
            print(f"校準檔已有、設定相同、sha256 對得上：{out}（run {old.get('run_id')}）→ 跳過", flush=True)
            return
        raise SystemExit(f"{out} 已存在但設定或 sha256 不同（{side}）→ 不默默蓋掉；要重算加 --force")
    import torch
    import m10_tasks as T
    from m10_quality import Answerer
    rid = os.environ.get("RUN_ID", "adhoc")
    ans = Answerer(a.model)
    km = ans.km
    check_info(km, MODEL_INFO[a.model])
    P = json.load(open(a.params_file))["params"][a.model]
    L, H, D = km.L, km.H, km.D
    HD = H * D
    items = plan_interleaved(a.tasks, a.n, a.seed0)
    acc = {h: dict(s=torch.zeros(L, HD, dtype=torch.float64, device=km.device),
                   G=torch.zeros(L, HD, HD, dtype=torch.float64, device=km.device)) for h in "kv"}
    N, probe, ids_done, t0 = 0, None, [], time.time()
    for idx, (task, b1, b2, seed) in enumerate(items):
        it = T.make_item(task, b1, b2, seed, ans.tok, P.get(task))
        ids = torch.tensor(it["context_ids"], device=km.device)
        cap, hk = [], None
        if idx == 0:
            hk = km.hf.model.layers[0].self_attn.k_proj.register_forward_hook(lambda m, i, o: cap.append(o.detach().clone()))
        try:
            km.key_keep = None
            km.prefill_chunked(ids)
        finally:
            if hk is not None:
                hk.remove()
        n = len(ids)
        tctx = TCtx(km, n)
        if idx == 0:
            probe = rope_probe(km, cap[1][0], tctx, CHUNK)
            print("RoPE 核對（第 0 層、位置 512–1023）：", json.dumps(probe), flush=True)
            if not probe["pass"]:
                raise SystemExit("RoPE 逆旋轉的資料自身核對沒過（規則 6／2：停）")
        cos, sin = tctx.cs(torch.arange(CHUNK, n, device=km.device))
        with torch.inference_mode():
            for l in range(L):
                for hi, h in enumerate("kv"):
                    x = km.kv[l, hi, :, CHUNK:n]
                    xp = t_rope_inv(x, cos, sin) if hi == 0 else x.float()
                    Z = xp.permute(1, 0, 2).reshape(n - CHUNK, HD).double()
                    acc[h]["s"][l] += Z.sum(0)
                    acc[h]["G"][l] += Z.T @ Z
        N += n - CHUNK
        ids_done.append(it["item_id"])
        print(f"[{idx + 1}/{len(items)}] {it['item_id']} {time.time() - t0:.1f}s", flush=True)
    Rj = min(HD, a.joint_rank_max * H)
    save = dict(meta=dict(model=a.model, run_id=rid, ts=ts(), L=L, H=H, D=D, n_tokens=N, items=ids_done,
                          item_keys=[[t, s] for t, _, _, s in items], settings=settings, joint_rank_max=Rj, rope_probe=probe))
    energy = {}
    with torch.inference_mode():
        for h in "kv":
            mean = acc[h]["s"] / N
            cov = acc[h]["G"] / N - mean.unsqueeze(-1) * mean.unsqueeze(-2)
            cov = (cov + cov.transpose(-1, -2)) / 2
            ch = torch.stack([cov[:, i * D:(i + 1) * D, i * D:(i + 1) * D] for i in range(H)], dim=1)   # [L,H,D,D]
            ev, U = torch.linalg.eigh(ch)
            ev, U = ev.flip(-1), U.flip(-1)
            evj, Uj = torch.linalg.eigh(cov)
            evj, Uj = evj.flip(-1), Uj.flip(-1)
            save[f"{h}_mean"] = mean.view(L, H, D).float()
            save[f"{h}_U"] = U.float().contiguous()
            save[f"{h}_ev"] = ev.float()
            save[f"{h}_Uj"] = Uj[:, :, :Rj].float().contiguous()
            save[f"{h}_evj"] = evj.float()
            tot = ev.clamp_min(0).sum(-1)
            totj = evj.clamp_min(0).sum(-1)
            energy[h] = {str(r): dict(per_head=float((ev.clamp_min(0)[..., :r].sum(-1) / tot).mean()),
                                      joint=float((evj.clamp_min(0)[..., :r * H].sum(-1) / totj).mean()) if r * H <= HD else None)
                         for r in (4, 8, 12, 16, 24, 32, 40, 48, 64)}
    save["meta"]["energy_captured_centered"] = energy
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.save(save, out)
    summ = dict(run_id=rid, ts=ts(), file=out, sha256=sha256(out), settings=settings, n_tokens=N, items=ids_done,
                rope_probe=probe, energy_captured_centered=energy,
                note="能量＝校準資料上去平均共變異的前 r 個特徵值佔比（per_head：每層每 head 平均；joint：每層跨 head、秩 r×H）。")
    json.dump(summ, open(side, "w"), indent=1, ensure_ascii=False)
    os.makedirs(a.summary_dir, exist_ok=True)
    json.dump(summ, open(os.path.join(a.summary_dir, f"calib_{a.model}.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(energy, ensure_ascii=False), flush=True)
    print(f"寫好 {out}（{N} token、{len(ids_done)} 題、{time.time() - t0:.0f}s）", flush=True)


# ============================================================================ calib-xl（10/21：xKV 事先校準的基底）
XL_ENERGY_RANKS = (4, 8, 11, 16, 17, 22, 28, 32, 34, 42, 64)


def calib_xl_settings(a) -> dict:
    return dict(model=a.model, tasks=list(a.tasks), n=a.n, seed0=a.seed0, params_file=os.path.relpath(a.params_file, REPO),
                skip_block0=True, W=a.w, rank_max_per_head=a.rank_max, centered=False)


def cmd_calib_xl(a):
    """xKV 的基底事先用校準題算好、所有題共用（10/21）。校準題＝和 PCA 的 calib 同一套（seed 61000 起、T2–T5 各 2 題，和所有評測題不重疊）。
    每個窗口（W 個相鄰層）、K／V 各一個：Z＝[token, W·H·D]（K 先逆 RoPE；不去平均＝照 xKV 與每題 SVD 版），FP64 累加 G＝Σ ZᵀZ，
    最後 eigh(G) 取前 R＝rank_max×W×H 個特徵向量（由大到小；＝所有校準 token 疊起來的右奇異向量）。第一題做 RoPE 的資料自身核對（規則 6）。"""
    out = a.out or os.path.join(DATA, f"calib_xl_{a.model}.pt")
    side = out + ".json"
    settings = calib_xl_settings(a)
    if os.path.exists(out) and not a.force:
        old = json.load(open(side)) if os.path.exists(side) else {}
        if old.get("settings") == settings and old.get("sha256") == sha256(out):
            print(f"xKV 校準檔已有、設定相同、sha256 對得上：{out}（run {old.get('run_id')}）→ 跳過", flush=True)
            return
        raise SystemExit(f"{out} 已存在但設定或 sha256 不同（{side}）→ 不默默蓋掉；要重算加 --force")
    import torch
    import m10_tasks as T
    from m10_quality import Answerer
    rid = os.environ.get("RUN_ID", "adhoc")
    ans = Answerer(a.model)
    km = ans.km
    check_info(km, MODEL_INFO[a.model])
    P = json.load(open(a.params_file))["params"][a.model]
    L, H, D, W = km.L, km.H, km.D, a.w
    if L % W:
        raise SystemExit(f"窗口 {W} 不整除層數 {L}")
    nW, WHD = L // W, W * H * D
    R = min(WHD, a.rank_max * W * H)
    items = plan_interleaved(a.tasks, a.n, a.seed0)
    G = {h: torch.zeros(nW, WHD, WHD, dtype=torch.float64, device=km.device) for h in "kv"}
    N, probe, ids_done, t0 = 0, None, [], time.time()
    for idx, (task, b1, b2, seed) in enumerate(items):
        it = T.make_item(task, b1, b2, seed, ans.tok, P.get(task))
        ids = torch.tensor(it["context_ids"], device=km.device)
        cap, hk = [], None
        if idx == 0:
            hk = km.hf.model.layers[0].self_attn.k_proj.register_forward_hook(lambda m, i, o: cap.append(o.detach().clone()))
        try:
            km.key_keep = None
            km.prefill_chunked(ids)
        finally:
            if hk is not None:
                hk.remove()
        n = len(ids)
        tctx = TCtx(km, n)
        if idx == 0:
            probe = rope_probe(km, cap[1][0], tctx, CHUNK)
            print("RoPE 核對（第 0 層、位置 512–1023）：", json.dumps(probe), flush=True)
            if not probe["pass"]:
                raise SystemExit("RoPE 逆旋轉的資料自身核對沒過（規則 6／2：停）")
        cos, sin = tctx.cs(torch.arange(CHUNK, n, device=km.device))
        with torch.inference_mode():
            for hi, h in enumerate("kv"):
                for w in range(nW):
                    Z = torch.cat([(t_rope_inv(km.kv[l, hi, :, CHUNK:n], cos, sin) if hi == 0 else km.kv[l, hi, :, CHUNK:n].float())
                                   .permute(1, 0, 2).reshape(n - CHUNK, H * D) for l in range(w * W, (w + 1) * W)], dim=1).double()
                    G[h][w] += Z.T @ Z
                    del Z
        N += n - CHUNK
        ids_done.append(it["item_id"])
        print(f"[{idx + 1}/{len(items)}] {it['item_id']} {time.time() - t0:.1f}s", flush=True)
    save = dict(meta=dict(model=a.model, run_id=rid, ts=ts(), L=L, H=H, D=D, W=W, R=R, n_tokens=N, items=ids_done,
                          item_keys=[[t, s] for t, _, _, s in items], settings=settings, rope_probe=probe, centered=False))
    energy, t1 = {}, time.time()
    with torch.inference_mode():
        for h in "kv":
            Gs = (G[h] + G[h].transpose(-1, -2)) / 2
            ev, U = torch.linalg.eigh(Gs)
            ev, U = ev.flip(-1), U.flip(-1)
            save[f"{h}_U"] = U[..., :R].float().contiguous()
            save[f"{h}_ev"] = ev.float()
            evp = ev.clamp_min(0)
            tot = evp.sum(-1)
            energy[h] = {str(r): float((evp[..., :r * W * H].sum(-1) / tot).mean()) for r in XL_ENERGY_RANKS if r * W * H <= WHD}
            del Gs, U
    save["meta"]["energy_captured_uncentered"] = energy
    save["meta"]["t_eigh_s"] = round(time.time() - t1, 1)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.save(save, out)
    summ = dict(run_id=rid, ts=ts(), file=out, sha256=sha256(out), settings=settings, n_tokens=N, items=ids_done, R=R,
                rope_probe=probe, energy_captured_uncentered=energy, t_eigh_s=save["meta"]["t_eigh_s"],
                note="能量＝校準資料上不去平均的二次動差 ΣZᵀZ 的前 r×W×H 個特徵值佔比（每個窗口平均；r＝每 head 每層等效秩）。"
                     "這是校準資料上的數字，不是評測題上的重建誤差。")
    json.dump(summ, open(side, "w"), indent=1, ensure_ascii=False)
    os.makedirs(a.summary_dir, exist_ok=True)
    json.dump(summ, open(os.path.join(a.summary_dir, f"calib_xl_{a.model}.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(energy, ensure_ascii=False), flush=True)
    print(f"寫好 {out}（{N} token、{len(ids_done)} 題、R＝{R}、eigh {save['meta']['t_eigh_s']}s、共 {time.time() - t0:.0f}s）", flush=True)


# ============================================================================ screen（X1）
def s2_config(a, model: str):
    """S2 的原本各組：照 m16_s2_equalbytes.cmd_run 的寫法組參數（不改 S2 的檔）。"""
    import m10_drop as DR
    import m16_s2_equalbytes as S2
    wm, src = list(a.write_methods), "cli"
    if a.ds2_plan and os.path.exists(a.ds2_plan):
        wm = json.load(open(a.ds2_plan))["models"][model]["rank"]["B"]["top2"]
        src = f"ds2_plan:{os.path.relpath(a.ds2_plan, REPO)}"
    extra = [m for m in a.extra_methods if m not in wm]
    roles = {m: "write" for m in wm}
    roles.update({m: ("qry" if m == "QRY" else "random" if m.startswith("RND") else "write" if m in DR.WRITE_TIME
                      else "position") for m in extra})
    fmts = S2.resolve_formats(model, S2.DEFAULT_FORMATS, S2.FMT_CHOICE)
    s2a = SimpleNamespace(model=model, variants=list(a.s2_variants), matched=True, quant_paths=["mask", "sdpa"])
    ds1_pat = a.check_ds1 if a.check_ds1 is not None else S2.DS1_RAW.format(model=model)
    ds1 = None
    if ds1_pat and glob.glob(ds1_pat):
        ds1, paths = S2.load_ds1(ds1_pat, model)
        print(f"DS1 逐列檔 {len(paths)} 個、{len(ds1)} 列（{ds1_pat}）", flush=True)
    return dict(a=s2a, drop_methods=wm + extra, roles=roles, fmts=fmts, src=src, mask_check=ds1 is not None, ds1=ds1)


def x1_item(ans, it, a, combos, calib, s2cfg, info, calib_xl=None):
    """一題：S2 子集（可選）→ E0（抓 q）→ 每個組合：還原區內 → pipeline → 位元組核對 → 重建誤差 → 放回 → 代理 → teacher forcing。"""
    import torch
    import m10_drop as DR
    import m10_sweep as SW
    import m16_s2_equalbytes as S2
    km = ans.km
    km.key_keep = None
    st = SW.Item(ans, it)
    tr = it["turns"][0]
    n = st.ctx_len
    checks, rows_s2 = {}, []
    if s2cfg is not None:
        with reuse_item(SW, st) as calls:
            rows_s2, ch2 = S2.run_item(ans, it, s2cfg["a"], s2cfg["drop_methods"], s2cfg["roles"], s2cfg["fmts"], s2cfg["src"],
                                       s2cfg["mask_check"])
        if calls != [it["item_id"]]:
            raise RuntimeError(f"S2.run_item 建 Item 的次數不對：{calls}（reuse_item 的假設不成立，規則 2：停）")
        checks["s2"] = ch2
        if s2cfg["ds1"] is not None:
            checks["s2_vs_ds1"] = S2.compare_ds1(s2cfg["ds1"], it["item_id"], rows_s2)
        st.dec = None
        torch.cuda.empty_cache()
    km.key_keep = None
    lp0, qcap = capture_q(ans, st, a.proxy_nq)
    m0 = SW.Item.metrics(lp0, lp0, st.gold_ids)
    if rows_s2:
        e0 = next(r for r in rows_s2 if r["group"] == "E0")
        checks["e0_vs_s2"] = dict(equal=e0["gold_lp"] == m0["gold_lp"] and e0["first_ok"] == m0["first_ok"],
                                  absdiff=abs(e0["gold_lp"] - m0["gold_lp"]))
    need = sorted({s["method"] for c in combos for s in c["stages"] if s["op"] == "drop"})
    t1 = time.time()
    orders = {}
    if need:
        cx = DR.Ctx(km, st.ids, n, st.bf16, CHUNK, it["seed"], st.needle, tr["query_ids"])
        for m in need:
            orders[m] = DR.unit_order(cx, "B", m)
        del cx
    t_scores = round(time.time() - t1, 3)
    L = km.L
    layers = list(range(L)) if a.proxy_layers == "all" else list(range(0, L, int(a.proxy_layers)))
    prox = Proxy(km, qcap, layers, n, wo=not a.no_proxy_wo)
    nb = n // CHUNK
    nd = torch.tensor([t for b in st.needle for t in range(b * CHUNK, (b + 1) * CHUNK)], dtype=torch.long, device=km.device)
    base = dict(model=a.model, item_id=it["item_id"], task=it["task"], band=tr["band"], seed=it["seed"],
                needle_blocks=json.dumps(list(st.needle)), gold_lp_e0=m0["gold_lp"], first_ok_e0=m0["first_ok"],
                proxy_layers=a.proxy_layers, proxy_nq=prox.nq, calib_run_id=calib_rids(calib, calib_xl))
    rows = [dict(base, range="", region="", spec="E0", groups="", roles="ref", family="none", families="none", n_lossy=0,
                 deployable=1, attn_path="sdpa", k_drop=0, kept_frac=1.0, needle_kept=1.0, t_pipe_s="", t_tf_s="",
                 relerr_k=0.0, relerr_v=0.0, cenerr_k=0.0, attn_err=0.0, attn_err_max=0.0, attn_err_wo=0.0,
                 bytes_region="", bytes_bf16_region="", ratio_region=1.0, bytes_total=nb * km.chunk_bytes, ratio_total=1.0,
                 bits_per_value_region=16.0, bytes_shared=0, bytes_detail="", **m0)]
    dirty = set()
    for rng in a.ranges:
        lo, hi = (1, nb - 1) if REGIONS[rng] is None else REGIONS[rng]
        nreg = hi - lo + 1
        s0, s1 = lo * CHUNK, (hi + 1) * CHUNK
        loc = {m: [b - lo for b in o if lo <= b <= hi] for m, o in orders.items()}
        tctx = TCtx(km, n, calib=calib, orders=loc, deep=deep_layers(L))
        tctx.xlcal = calib_xl                     # 10/21：xl:…:cal 用（None＝沒有 :cal 的組合）
        pos = torch.arange(s0, s1, device=km.device)
        for c in combos:
            if dirty:
                restore_blocks(km, st, sorted(dirty))
                dirty.clear()
            X = km.kv[:, :, :, s0:s1]
            torch.cuda.synchronize()
            ta = time.time()
            with torch.inference_mode():
                dec, idx, by, dropped = t_pipeline(c["stages"], X, pos, tctx, info)
                pred = predict_bytes(c["stages"], info, nreg)
                if by != pred:
                    raise RuntimeError(f"{c['spec']}：實際位元組 {by} ≠ 預測 {pred}（規則 2：停）")
                rel = t_relerr(X, dec, idx)
                if dec.data_ptr() != X.data_ptr():
                    if idx is None:
                        X.copy_(dec)
                    else:
                        X.index_copy_(3, idx, dec)
                del dec
            dirty.update(range(lo, hi + 1))
            torch.cuda.synchronize()
            tb = time.time()
            dabs = [lo + b for b in dropped]
            kk = S2.keep_tensor(km, dabs)
            with torch.inference_mode():
                pm = prox.measure(kk[:n] if dabs else None)
            mm = S2.tf(st, lp0, kk)
            tc = time.time()
            kept, ndk = S2.kept_stats(kk, n, nd)
            at = spec_attrs(c["stages"])
            rows.append(dict(base, range=rng, region=f"{lo}-{hi}", spec=c["spec"], groups="|".join(c["groups"]),
                             roles="|".join(c["roles"]), **at, attn_path="mask", k_drop=len(dabs), kept_frac=kept,
                             needle_kept=ndk, **bytes_summary(by, info, nreg, nb), **rel, **pm,
                             t_pipe_s=round(tb - ta, 3), t_tf_s=round(tc - tb, 3), **mm))
    if dirty:
        restore_blocks(km, st, sorted(dirty))
    checks["t_scores_s"] = t_scores
    checks.update(compare_x1_s2(rows, rows_s2, s2cfg))
    if s2cfg is not None and s2cfg["ds1"] is not None:
        checks["x1_vs_ds1"] = compare_x1_ds1(rows, s2cfg["ds1"], it["item_id"])
    del st, prox
    return rows_s2, rows, checks


def compare_x1_s2(rows, rows_s2, s2cfg) -> dict:
    """同一題：X1 的 MASK0（none）與量化單一格式（全段、遮罩路徑）必須和 S2 子集的同一格逐值相同（同樣的值放進同樣的位置）。"""
    if not rows_s2:
        return {}
    out = {"x1_vs_s2": []}
    s2 = {}
    for r in rows_s2:
        if r["group"] == "MASK0":
            s2["none"] = r
        elif r["group"] == "quant" and r["variant"] == "full" and r["attn_path"] == "mask":
            s2[f"q:{r['fmt']}"] = r
    for r in rows:
        if r.get("range") == "full" and r["spec"] in s2:
            o = s2[r["spec"]]
            d = {k: abs(float(r[k]) - float(o[k])) for k in ("kl", "gold_lp", "top1", "first_ok")}
            out["x1_vs_s2"].append(dict(spec=r["spec"], equal=all(v == 0 for v in d.values()), absdiff=d))
    return out


def compare_x1_ds1(rows, ds1, item_id) -> list:
    out = []
    for r in rows:
        if r.get("range") != "full" or not r["spec"].startswith("drop:") or "+" in r["spec"]:
            continue
        _, m, fr = r["spec"].split(":")
        o = ds1.get((item_id, "B", m, round(float(fr), 4)))
        if o is None:
            continue
        d = {k: abs(float(r[k]) - float(o[k])) for k in ("kl", "gold_lp", "top1", "first_ok", "kept_frac") if o.get(k, "") != ""}
        out.append(dict(spec=r["spec"], run_id_ds1=o["run_id"], equal=all(v == 0 for v in d.values()), absdiff=d))
    return out


def check_failures(checks: dict, tol: float = 1e-4) -> list:
    """數值重現檢查：差 > tol 的列出來（GPU 不決定性的小差只記錄；--smoke 時有任何一項就 exit 1）。"""
    bad = []
    for key in ("x1_vs_s2", "x1_vs_ds1", "s2_vs_ds1"):
        for c in checks.get(key, []) or []:
            if any(v > tol for v in c["absdiff"].values()):
                bad.append(f"{key}:{c.get('spec', c.get('key'))}:{c['absdiff']}")
    e = checks.get("e0_vs_s2")
    if e and e["absdiff"] > tol:
        bad.append(f"e0_vs_s2:{e}")
    s2 = checks.get("s2") or {}
    for c in s2.get("mask_vs_m10_drop_keep_mask", []):
        if not c["equal"]:
            bad.append(f"s2_mask:{c}")
    return bad


def cmd_screen(a):
    import torch
    import m10_tasks as T
    from m10_quality import Answerer
    import m16_s2_equalbytes as S2
    if a.smoke:
        a.tasks, a.n, a.shard = ["T2"], 1, "0/1"
    info = MODEL_INFO[a.model]
    combos = load_combos(a.combos, a.model, smoke=a.smoke)
    if a.only_specs:
        want = {canonical(parse_spec(s)) for s in a.only_specs}
        combos = [c for c in combos if c["spec"] in want]
    if a.only_groups:
        combos = [c for c in combos if set(c["groups"]) & set(a.only_groups)]
    print(f"組合 {len(combos)} 個（{a.model}）", flush=True)
    plan = plan_interleaved(a.tasks, a.n, a.seed0)
    ans = Answerer(a.model)
    check_info(ans.km, info)
    calib = None
    if any(s["op"] == "lr" for c in combos for s in c["stages"]):
        calib = load_calib(a.calib, ans.km, a.model, {(t, s) for t, _, _, s in plan})
    calib_xl = None
    if any(s.get("cal") for c in combos for s in c["stages"]):        # 10/21
        calib_xl = load_calib_xl(getattr(a, "calib_xl", None), ans.km, a.model, {(t, s) for t, _, _, s in plan})
    P = json.load(open(a.params_file))["params"][a.model]
    s2cfg = None if a.no_s2 else s2_config(a, a.model)
    si, sk = parse_shard(a.shard)
    done = done_union(a.out)
    s2out = a.s2_out or re.sub(r"x1_([^/]*)$", r"s2part_\1", a.out)
    if s2cfg is not None and os.path.abspath(s2out) == os.path.abspath(a.out):
        raise SystemExit("--out 的檔名要以 x1_ 開頭（S2 子集寫到同目錄的 s2part_…），或另給 --s2-out")
    fo, w = open_out(a.out, FIELDS_X1)
    fo2, w2 = open_out(s2out, S2.FIELDS) if s2cfg is not None else (None, None)
    chk = open(side_path(a.out, ".checks.jsonl"), "a")
    watch = start_watch(a.out)
    rid = os.environ.get("RUN_ID", "adhoc")
    t_all, n_new, bad = time.time(), 0, []
    try:
        for k, (task, b1, b2, seed) in enumerate(plan):
            if k % sk != si or item_id_of(task, b1, b2, seed) in done:
                continue
            if a.budget_min and (time.time() - t_all) / 60 > a.budget_min:
                print(f"時間預算 {a.budget_min} 分鐘到了：剩下的題留給下一個 job（逐題續跑）", flush=True)
                break
            t0 = time.time()
            it = T.make_item(task, b1, b2, seed, ans.tok, P.get(task))
            assert it["item_id"] == item_id_of(task, b1, b2, seed), it["item_id"]
            rows_s2, rows, checks = x1_item(ans, it, a, combos, calib, s2cfg, info, calib_xl=calib_xl)
            hl, intr = host_level(), intruders(watch)
            if rows_s2:
                for r in rows_s2:
                    r.update(run_id=rid, ts=ts(), shard=a.shard, n_rows_item=len(rows_s2), own_gpu_intruders=intr, host_level=hl)
                    w2.writerow(r)
                fo2.flush()
            for r in rows:
                r.update(run_id=rid, ts=ts(), shard=a.shard, n_rows_item=len(rows), own_gpu_intruders=intr, host_level=hl)
                w.writerow(r)
            fo.flush()
            chk.write(json.dumps(dict(run_id=rid, ts=ts(), model=a.model, item_id=it["item_id"], checks=checks,
                                      own_gpu_intruders=intr, host_level=hl), ensure_ascii=False, default=str) + "\n")
            chk.flush()
            fb = check_failures(checks)
            if fb:
                bad.append((it["item_id"], fb))
                print(f"  重現檢查有差：{fb[:5]}", flush=True)
            n_new += 1
            print(f"[{k + 1}] {it['item_id']} rows={len(rows)}+{len(rows_s2)} {time.time() - t0:.1f}s "
                  f"avg {(time.time() - t_all) / n_new:.1f}s/題 peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            if a.smoke:
                _print_x1(rows)
            torch.cuda.empty_cache()
    finally:
        fo.close()
        if fo2 is not None:
            fo2.close()
        chk.close()
        finish_watch(watch, a.out)
    if bad:
        print(f"重現檢查有差的題 {len(bad)}：{bad[:3]}（看 {side_path(a.out, '.checks.jsonl')}）", flush=True)
        if a.smoke:
            sys.exit(1)


def _print_x1(rows):
    print(f"  {'spec':<52} {'比':>7} {'Δgold_lp':>9} {'KL':>8} {'fo':>3} {'relK':>7} {'relV':>7} {'attn':>7} {'attnWO':>7} {'t_pipe':>6}")
    for r in rows:
        print(f"  {r['spec']:<52} {float(r['ratio_region']):>7.4f} {r['gold_lp'] - r['gold_lp_e0']:>+9.3f} {r['kl']:>8.4f} "
              f"{r['first_ok']:>3} {str(r['relerr_k'])[:7]:>7} {str(r['relerr_v'])[:7]:>7} {str(r['attn_err'])[:7]:>7} "
              f"{str(r['attn_err_wo'])[:7]:>7} {str(r['t_pipe_s']):>6}", flush=True)


# ============================================================================ report（CPU；不碰 torch）
def _f(r, k):
    v = r.get(k, "")
    return None if v in ("", None) else float(v)


def dedupe_complete(rows, key_cols):
    last = {}
    for r in rows:
        last[tuple(r[k] for k in key_cols)] = r
    rows = list(last.values())
    cnt, want = {}, {}
    for r in rows:
        k = (r["model"], r.get("ctx", ""), r["item_id"])
        cnt[k] = cnt.get(k, 0) + 1
        want[k] = max(want.get(k, 0), int(r["n_rows_item"]))
    ok = {k for k in cnt if cnt[k] >= want[k]}
    return [r for r in rows if (r["model"], r.get("ctx", ""), r["item_id"]) in ok], sorted(f"{m}/{c}/{i}" for m, c, i in set(cnt) - ok)


def config_table(rows: list) -> dict:
    """一個（模型, 範圍）的逐列 → {spec: 配置統計}。品質差全部以題目為單位 bootstrap（10,000 次）。"""
    by = {}
    for r in rows:
        by.setdefault(r["spec"], {})[r["item_id"]] = r
    tab = {}
    for spec, items in by.items():
        rs = list(items.values())
        ratios = sorted({round(float(r["ratio_region"]), 6) for r in rs})
        roles = sorted({x for r in rs for x in r["roles"].split("|") if x})
        groups = sorted({x for r in rs for x in r["groups"].split("|") if x})
        c = dict(spec=spec, n=len(rs), ratio=ratios[-1], ratio_values=ratios, roles=roles, groups=groups,
                 family=rs[0]["family"], families=rs[0]["families"], n_lossy=int(rs[0]["n_lossy"]),
                 deployable=int(rs[0]["deployable"]),
                 d_gold_lp=boot2([float(r["gold_lp"]) - float(r["gold_lp_e0"]) for r in rs]),
                 d_first_ok=boot2([int(r["first_ok"]) - int(r["first_ok_e0"]) for r in rs]),
                 first_ok=statistics.mean(int(r["first_ok"]) for r in rs),
                 first_ok_e0=statistics.mean(int(r["first_ok_e0"]) for r in rs),
                 kl_median=statistics.median(float(r["kl"]) for r in rs), kl_mean=statistics.mean(float(r["kl"]) for r in rs),
                 top1=statistics.mean(float(r["top1"]) for r in rs),
                 per_item={r["item_id"]: float(r["gold_lp"]) - float(r["gold_lp_e0"]) for r in rs})
        for p in PROXIES:
            v = [_f(r, p) for r in rs if _f(r, p) is not None]
            c[p] = statistics.mean(v) if v else None
        for th in MLPERF:
            c[f"mlperf{th}"] = bool(c["first_ok"] >= th * c["first_ok_e0"])
        tab[spec] = c
    return tab


def proxy_eval(tab: dict) -> dict:
    """代理 vs KL、Δgold_lp：Spearman（以配置為單位，bootstrap CI）、跨家族與家族內；崩潰 AUROC（崩＝Δfirst_ok ≤ −5 個百分點）。"""
    cs = [c for c in tab.values() if c["spec"] not in ("E0",)]
    out = {}
    for p in PROXIES:
        xs = [c for c in cs if c[p] is not None]
        if len(xs) < 3:
            out[p] = None
            continue
        e = out[p] = {}
        e["all_vs_kl"] = spearman_ci([c[p] for c in xs], [c["kl_mean"] for c in xs])
        e["all_vs_d_gold_lp"] = spearman_ci([c[p] for c in xs], [qual(c) for c in xs])
        e["by_family_vs_d_gold_lp"] = {}
        for F in sorted({c["families"] for c in xs if c["n_lossy"] == 1}) + ["combo"]:
            ys = [c for c in xs if (c["n_lossy"] >= 2 if F == "combo" else c["n_lossy"] == 1 and c["families"] == F)]
            e["by_family_vs_d_gold_lp"][F] = spearman([c[p] for c in ys], [qual(c) for c in ys]) if len(ys) >= 3 else None
        lab = [c["d_first_ok"]["mean"] <= COLLAPSE_FIRST_OK for c in xs]
        e["collapse_auroc"] = auroc([c[p] for c in xs], lab)
        e["n_collapse"] = sum(lab)
    return out


def plot_pareto(tab: dict, front: list, model: str, path: str) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as ex:
        return f"NOT_PLOTTED（{ex}）"
    fig, ax = plt.subplots(figsize=(8, 5))
    style = {"single": ("o", "#4C78A8"), "combo": ("^", "#E45756"), "control": ("x", "#9D9D9D")}
    for c in tab.values():
        if c["spec"] == "E0" or c.get("d_gold_lp") is None:
            continue
        k = "single" if is_single(c) else "combo" if is_combo(c) else "control"
        mk, col = style[k]
        ax.scatter(c["ratio"], qual(c), marker=mk, color=col, s=18, alpha=0.7)
    fp = sorted((tab[s]["ratio"], qual(tab[s])) for s in front)
    if fp:
        ax.plot([x for x, _ in fp], [y for _, y in fp], "-", color="#222222", lw=1)
    for B in BUDGETS:
        ax.axvline(B, color="#BBBBBB", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("bytes ratio vs BF16 (region, measured)")
    ax.set_ylabel("mean Δ gold log-prob vs E0 (teacher forcing)")
    ax.set_title(f"X1 Pareto — {model} (o single, ^ combo, x control)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def cmd_report(a):
    import csv
    paths = sorted({p for g in a.raw for p in glob.glob(g)})
    rows, partial = dedupe_complete(read_rows(paths), ("model", "item_id", "range", "spec"))
    os.makedirs(a.out_dir, exist_ok=True)
    rid = os.environ.get("RUN_ID", "adhoc")
    rep = dict(run_id=rid, ts=ts(), files=paths, src_run_ids=sorted({r["run_id"] for r in rows}), incomplete_items=partial,
               rule_text=STOP_TEXT, eps=EPS, budgets=BUDGETS, budget_tol=BUDGET_TOL,
               contention=dict(rows_with_intruders=sum(1 for r in rows if r["own_gpu_intruders"] not in ("0", "NOT_CHECKED")),
                               rows_not_checked=sum(1 for r in rows if r["own_gpu_intruders"] == "NOT_CHECKED"),
                               host_levels=sorted({r["host_level"] for r in rows})), models={})
    tabs, n_items = {}, {}
    L = ["# X1：疊加篩選（teacher forcing、第 1 輪；只報告，判定見 §停止條件）", "",
         f"run：{'、'.join(rep['src_run_ids'])}；不齊全的題（沒用）：{len(partial)}", ""]
    for m in sorted({r["model"] for r in rows}):
        R = [r for r in rows if r["model"] == m and r["range"] == a.range_name or r["model"] == m and r["spec"] == "E0"]
        n_items[m] = len({r["item_id"] for r in R})
        tab = config_table([r for r in R if r["spec"] != "E0"])
        tabs[m] = tab
        mult = s1_multipliers(a.s1_summary, m, a.lossless_codec)
        dep = [c for c in tab.values() if c["deployable"] and not ({"control", "ref", "deploy_broken"} & set(c["roles"]))]
        front = pareto_front([(c["ratio"], qual(c), c["spec"]) for c in dep])
        pe = proxy_eval(tab)
        M = rep["models"][m] = dict(n_items=n_items[m], n_configs=len(tab), front=front, proxies=pe,
                                    lossless_multipliers=mult or "NOT_MEASURED",
                                    S1c=(None if "int4_kivi" not in mult else bool(mult["int4_kivi"] < 1.05)))
        M["pareto_png"] = plot_pareto(tab, front, m, os.path.join(a.out_dir, f"pareto_{m}.png"))
        cols = ["run_id", "ts", "model", "range", "spec", "groups", "roles", "family", "families", "n_lossy", "deployable",
                "n", "ratio", "on_front", "d_gold_lp", "d_gold_lp_lo", "d_gold_lp_hi", "d_first_ok", "d_first_ok_lo",
                "d_first_ok_hi", "first_ok", "first_ok_e0", "mlperf0.99", "mlperf0.999", "kl_median", "kl_mean", "top1",
                *PROXIES, "lossless_key", "lossless_mult", "ratio_lossless"]
        with open(os.path.join(a.out_dir, f"configs_{m}.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=cols)
            wr.writeheader()
            for c in sorted(tab.values(), key=lambda c: (c["ratio"], c["spec"])):
                lk = lossless_key(c["spec"])
                mu = mult.get(lk) if lk else None
                wr.writerow({"run_id": rid, "ts": rep["ts"], "model": m, "range": a.range_name, "spec": c["spec"],
                             "groups": "|".join(c["groups"]), "roles": "|".join(c["roles"]), "family": c["family"],
                             "families": c["families"], "n_lossy": c["n_lossy"], "deployable": c["deployable"], "n": c["n"],
                             "ratio": c["ratio"], "on_front": int(c["spec"] in front),
                             "d_gold_lp": c["d_gold_lp"]["mean"], "d_gold_lp_lo": c["d_gold_lp"]["lo"],
                             "d_gold_lp_hi": c["d_gold_lp"]["hi"], "d_first_ok": c["d_first_ok"]["mean"],
                             "d_first_ok_lo": c["d_first_ok"]["lo"], "d_first_ok_hi": c["d_first_ok"]["hi"],
                             "first_ok": c["first_ok"], "first_ok_e0": c["first_ok_e0"], "mlperf0.99": int(c["mlperf0.99"]),
                             "mlperf0.999": int(c["mlperf0.999"]), "kl_median": c["kl_median"], "kl_mean": c["kl_mean"],
                             "top1": c["top1"], **{p: ("" if c[p] is None else c[p]) for p in PROXIES},
                             "lossless_key": lk or "", "lossless_mult": mu if mu else "NOT_MEASURED",
                             "ratio_lossless": round(c["ratio"] / mu, 6) if mu else "NOT_MEASURED"})
        L += [f"## {m}（{n_items[m]} 題、{len(tab)} 個配置）", "", "### 前緣（可部署、非對照；位元組比 vs 平均 Δgold_lp）", "",
              "| 組合 | 位元組比 | Δgold_lp [CI] | Δfirst_ok（百分點） | KL 中位數 | attn_err |", "|:--|--:|--:|--:|--:|--:|"]
        for s in front:
            c = tab[s]
            ae = "" if c["attn_err"] is None else f"{c['attn_err']:.4f}"
            L.append(f"| {s} | {c['ratio']:.4f} | {_ci(c['d_gold_lp'])} | {_ci(c['d_first_ok'], 100, 1)} | {c['kl_median']:.4f} | {ae} |")
        L += ["", "### 每個預算：最好的單一方法、各家族最好的、最好的組合", "",
              "| 預算 | 種類 | 組合 | 位元組比 | Δgold_lp [CI] | Δfirst_ok |", "|:--|:--|:--|--:|--:|--:|"]
        for B in BUDGETS:
            S = [c for c in tab.values() if is_single(c) and in_budget(c, B)]
            Cb = [c for c in tab.values() if is_combo(c) and in_budget(c, B)]
            items = [("最好的單一", best_of(S))] + [(f"家族 {F}", best_of([c for c in S if c["families"] == F]))
                                                     for F in ("quant", "drop", "lowrank", "xlayer", "merge", "xshare")]
            items += [("最好的組合", best_of(Cb))]
            for lab, c in items:
                if c:
                    L.append(f"| {B} | {lab} | {c['spec']} | {c['ratio']:.4f} | {_ci(c['d_gold_lp'])} | {_ci(c['d_first_ok'], 100, 1)} |")
        L += ["", "### 代理（Spearman ρ，以配置為單位；崩潰 AUROC）", "", "| 代理 | ρ vs KL [CI] | ρ vs Δgold_lp [CI] | 崩潰 AUROC |",
              "|:--|--:|--:|--:|"]
        for p, e in pe.items():
            if e:
                au = "—" if e["collapse_auroc"] is None else f"{e['collapse_auroc']:.3f}"
                L.append(f"| {p} | {_rho(e['all_vs_kl'])} | {_rho(e['all_vs_d_gold_lp'])} | {au} |")
        L.append("")
    J = judge_x1(tabs, n_items)
    rep["judge"] = J
    pe_all = [rep["models"][m]["proxies"] for m in tabs]
    s1a = []
    for pe in pe_all:
        best_abs = max([abs(e[k]["rho"]) for e in pe.values() if e for k in ("all_vs_kl", "all_vs_d_gold_lp") if e[k]] or [0])
        s1a.append(best_abs < SPEARMAN_OK)
    J["S1a_proxies_unreliable"] = bool(s1a) and all(s1a)
    S1c = [rep["models"][m]["S1c"] for m in tabs]
    J["S1c_drop_lossless"] = None if any(x is None for x in S1c) else all(S1c)
    sel = dict(run_id=rid, ts=rep["ts"], src_run_ids=rep["src_run_ids"], go=J["go"], reason=J["reason"],
               rule="x_stack_howto.md §5（事先登記）", eps=EPS, models={})
    if J["complete"]:
        for m, tab in tabs.items():
            sel["models"][m] = dict(configs=select_x2(tab, m), naive_x1=build_naive_x1(tab, m))
    json.dump(sel, open(os.path.join(a.out_dir, "x2_selection.json"), "w"), indent=1, ensure_ascii=False, default=str)
    for m, tab in tabs.items():
        for c in tab.values():
            c.pop("per_item", None)
    rep["configs"] = {m: tab for m, tab in tabs.items()}
    json.dump(rep, open(os.path.join(a.out_dir, "x1_report.json"), "w"), indent=1, ensure_ascii=False, default=str)
    L += ["## 停止條件（事先寫死，L5d；判讀由人）", "", "```", json.dumps({k: v for k, v in J.items() if k != "rule_text"},
                                                                          ensure_ascii=False, indent=1, default=str), "```", ""]
    L += [f"- **{k}**：{v}" for k, v in STOP_TEXT.items()]
    L += ["", f"X2 挑選：{'go' if J['go'] else '不跑'}（{J['reason']}）→ x2_selection.json"]
    open(os.path.join(a.out_dir, "tables_x1.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:60]))
    print(json.dumps({k: J[k] for k in ("go", "reason", "S1b_stop", "complete", "S1a_proxies_unreliable")}, ensure_ascii=False))


def _ci(b, scale=1.0, nd=3):
    if not b:
        return "—"
    return f"{scale * b['mean']:+.{nd}f} [{scale * b['lo']:+.{nd}f}, {scale * b['hi']:+.{nd}f}]"


def _rho(e):
    return "—" if not e else f"{e['rho']:+.2f} [{e['lo']:+.2f}, {e['hi']:+.2f}]"


# ============================================================================ gen（X2）
def x2_item(ans, it, configs, calib, info, nb, calib_xl=None, xl_cache=False):
    import torch
    import m10_drop as DR
    import m10_sweep as SW
    import m16_s2_equalbytes as S2
    km = ans.km
    km.key_keep = None
    st = SW.Item(ans, it)
    tr = it["turns"][0]
    n = st.ctx_len
    t0 = time.time()
    g0 = st.gen()
    rows = [dict(spec="E0", roles="ref", tiers="", budgets="", family="none", families="none", n_lossy=0, deployable=1,
                 attn_path="sdpa", k_drop=0, kept_frac=1.0, needle_kept=1.0, bytes_region="", ratio_region=1.0,
                 bits_per_value_region=16.0, gen_score=g0[0], gen_stop=g0[1], t_gen_s=round(time.time() - t0, 3))]
    need = sorted({s["method"] for c in configs for s in c["stages"] if s["op"] == "drop"})
    orders = {}
    if need:
        cx = DR.Ctx(km, st.ids, n, st.bf16, CHUNK, it["seed"], st.needle, tr["query_ids"])
        orders = {m: DR.unit_order(cx, "B", m) for m in need}
        del cx
    lo, hi = 1, nb - 1
    nreg = hi - lo + 1
    s0, s1 = lo * CHUNK, (hi + 1) * CHUNK
    tctx = TCtx(km, n, calib=calib, orders={m: [b - lo for b in o if lo <= b <= hi] for m, o in orders.items()},
                deep=deep_layers(km.L))
    tctx.xlcal = calib_xl                         # 10/21：xl:…:cal 用
    if xl_cache:                                  # 10/21：gen --xl-cache（同題同秩的每題 SVD 只算一次）
        tctx.xl_cache = xl_cache_plan(configs)
    pos = torch.arange(s0, s1, device=km.device)
    nd = torch.tensor([t for b in st.needle for t in range(b * CHUNK, (b + 1) * CHUNK)], dtype=torch.long, device=km.device)
    dirty = False
    for c in configs:
        if dirty:
            restore_blocks(km, st, range(lo, hi + 1))
        X = km.kv[:, :, :, s0:s1]
        with torch.inference_mode():
            dec, idx, by, dropped = t_pipeline(c["stages"], X, pos, tctx, info)
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
        dabs = [lo + b for b in dropped]
        kk = S2.keep_tensor(km, dabs)
        km.key_keep = kk
        t1 = time.time()
        try:
            g = st.gen()
        finally:
            km.key_keep = None
        kept, ndk = S2.kept_stats(kk, n, nd)
        bs = bytes_summary(by, info, nreg, nb)
        rows.append(dict(spec=c["spec"], roles="|".join(c["roles"]), tiers="|".join(c["tiers"]),
                         budgets="|".join(str(b) for b in c["budgets"]), **spec_attrs(c["stages"]), attn_path="mask",
                         k_drop=len(dabs), kept_frac=kept, needle_kept=ndk, bytes_region=bs["bytes_region"],
                         ratio_region=bs["ratio_region"], bits_per_value_region=bs["bits_per_value_region"],
                         gen_score=g[0], gen_stop=g[1], t_gen_s=round(time.time() - t1, 3)))
    if dirty:
        restore_blocks(km, st, range(lo, hi + 1))
    base = dict(item_id=it["item_id"], task=it["task"], band=tr["band"], seed=it["seed"], score_e0=g0[0])
    del st
    return [dict(base, **r) for r in rows]


def cmd_gen(a):
    import torch
    import m10_tasks as T
    import m10_sweep as SW
    from m10_quality import Answerer
    nb = 64 if a.ctx == "32K" else LC_BLOCKS
    seed0 = a.seed0 if a.seed0 is not None else (X2_SEED0 if nb == 64 else LC_SEED0)
    n = a.n if a.n is not None else (24 if nb == 64 else LC_N)
    sel = json.load(open(a.selection))
    if not sel.get("go"):
        print(f"x2_selection.json 的 go＝false（{sel.get('reason')}）→ 不跑 X2", flush=True)
        return
    info = MODEL_INFO[a.model]
    want_tier = {"core": {"core"}, "ext": {"ext"}, "x3": {"x3"}, "all": {"core", "ext", "x3"}, "128k": {"128k"}}[a.tier]
    configs = []
    for e in sel["models"][a.model]["configs"]:
        if not (set(e["tiers"]) & want_tier):
            continue
        st = validate(parse_spec(e["spec"]), info, nb - 1)
        configs.append(dict(e, stages=st))
    if a.only_specs:
        want = {canonical(parse_spec(s)) for s in a.only_specs}
        configs = [c for c in configs if c["spec"] in want]
    print(f"X2 {a.model} {a.ctx} tier={a.tier}：{len(configs)} 個配置、題目 seed0={seed0} n={n}", flush=True)
    plan = plan_interleaved(a.tasks, n, seed0)
    SW.N_BLOCKS = nb                                  # m10_sweep.Item 讀模組常數（m10_sweep screen 的 --n-blocks 同一個作法）
    ans = Answerer(a.model, extra=(nb - 64) * CHUNK + 4096)
    check_info(ans.km, info)
    calib = None
    if any(s["op"] == "lr" for c in configs for s in c["stages"]):
        calib = load_calib(a.calib, ans.km, a.model, {(t, s) for t, _, _, s in plan})
    calib_xl = None
    if any(s.get("cal") for c in configs for s in c["stages"]):       # 10/21
        calib_xl = load_calib_xl(getattr(a, "calib_xl", None), ans.km, a.model, {(t, s) for t, _, _, s in plan})
    P = json.load(open(a.params_file))["params"][a.model]
    lc = None
    if nb == LC_BLOCKS:
        pat = a.check_44c or LC44C.format(model=a.model)
        lcp = sorted(glob.glob(pat))
        if lcp:
            lc = {}
            for r in read_rows(lcp):
                if r["policy"] == "E0" or (r["policy"] == "ALL" and r["fmt"] == "int4_kivi" and r["readout"] == "A"):
                    lc[(r["item_id"], "E0" if r["policy"] == "E0" else "q:int4_kivi")] = r["gen_score"]
            print(f"44c 逐列檔 {len(lcp)} 個（E0／ALL int4_kivi 的生成分數，重現檢查用）", flush=True)
        else:
            print(f"44c 逐列檔找不到（{pat}）：128K 的重現檢查 NOT_FOUND，照樣跑", flush=True)
    si, sk = parse_shard(a.shard)
    done = done_union(a.out)
    fo, w = open_out(a.out, FIELDS_X2)
    chk = open(side_path(a.out, ".checks.jsonl"), "a")
    watch = start_watch(a.out)
    rid = os.environ.get("RUN_ID", "adhoc")
    cal_rid = calib_rids(calib, calib_xl)
    t_all, n_new = time.time(), 0
    try:
        for k, (task, b1, b2, seed) in enumerate(plan):
            if k % sk != si or item_id_of(task, b1, b2, seed) in done:
                continue
            if a.budget_min and (time.time() - t_all) / 60 > a.budget_min:
                print(f"時間預算 {a.budget_min} 分鐘到了：剩下的題留給下一個 job", flush=True)
                break
            t0 = time.time()
            it = T.make_item(task, b1, b2, seed, ans.tok, P.get(task), n_blocks=nb)
            assert it["item_id"] == item_id_of(task, b1, b2, seed), it["item_id"]
            rows = x2_item(ans, it, configs, calib, info, nb, calib_xl=calib_xl, xl_cache=getattr(a, "xl_cache", False))
            hl, intr = host_level(), intruders(watch)
            for r in rows:
                r.update(run_id=rid, ts=ts(), shard=a.shard, model=a.model, ctx=a.ctx, n_blocks=nb, calib_run_id=cal_rid,
                         n_rows_item=len(rows), own_gpu_intruders=intr, host_level=hl)
                w.writerow(r)
            fo.flush()
            ck = {}
            if lc is not None:
                for r in rows:
                    o = lc.get((it["item_id"], r["spec"]))
                    if o is not None:
                        ck[r["spec"]] = dict(ours=r["gen_score"], c44=float(o), equal=float(o) == float(r["gen_score"]),
                                             note="E0 兩邊都是 sdpa＝應該逐值相同；int4 我們走遮罩路徑、44c 走 sdpa＝只供參考")
            chk.write(json.dumps(dict(run_id=rid, ts=ts(), model=a.model, item_id=it["item_id"], vs_44c=ck or "NOT_FOUND",
                                      own_gpu_intruders=intr, host_level=hl), ensure_ascii=False, default=str) + "\n")
            chk.flush()
            n_new += 1
            print(f"[{k + 1}] {it['item_id']} rows={len(rows)} {time.time() - t0:.1f}s avg {(time.time() - t_all) / n_new:.1f}s/題 "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            torch.cuda.empty_cache()
    finally:
        fo.close()
        chk.close()
        finish_watch(watch, a.out)


# ============================================================================ gen-report（CPU）
def gen_table(rows):
    e0 = {r["item_id"]: float(r["gen_score"]) for r in rows if r["spec"] == "E0"}
    by = {}
    for r in rows:
        if r["spec"] != "E0":
            by.setdefault(r["spec"], {})[r["item_id"]] = r
    tab = {}
    for spec, items in by.items():
        rs = [r for i, r in items.items() if i in e0]
        if not rs:
            continue
        sc = {r["item_id"]: float(r["gen_score"]) for r in rs}
        acc = statistics.mean(sc.values())
        acc0 = statistics.mean(e0[i] for i in sc)
        st = parse_spec(spec)
        c = dict(spec=spec, n=len(rs), ratio=float(rs[0]["ratio_region"]), roles=rs[0]["roles"].split("|"),
                 budgets=[float(b) for b in rs[0]["budgets"].split("|") if b], n_lossy=len(st),
                 deployable=spec_attrs(st)["deployable"], families=spec_attrs(st)["families"], acc=acc, acc_e0=acc0,
                 diff=boot2([sc[i] - e0[i] for i in sc]), per_item=sc)
        for th in MLPERF:
            c[f"mlperf{th}"] = bool(acc >= th * acc0)
        c["eps_pass"] = bool(c["diff"]["lo"] >= -EPS)
        tab[spec] = c
    return tab, statistics.mean(e0.values()) if e0 else None


X3_FIELDS = ["model", "x3_segment", "feasible", "run_id", "ts", "combo", "ratio", "acc", "acc_e0", "acc_delta", "acc_delta_lo",
             "acc_delta_hi", "eps", "mlperf99", "mlperf999", "n_items", "src_run_ids"]


def write_x3_files(acc32: dict, rid: str, ts_: str, src: str, out_dir: str) -> dict:
    """X3（m16_xtime.load_x2）要的 x2_feasible.csv：model、x3_segment、feasible（1/0＝ε 判定：生成答對率對 E0 的配對差 95% CI 下界 ≥ −2 個百分點）、
    run_id；另附組合、位元組比、答對率差、MLPerf 99%／99.9% 欄（不影響 feasible）。對不到 X3 解碼段的寫 x2_unmapped.csv（附原因）。
    沒有任何 32K 的 X2 列 → 兩個檔都不寫（X3 的 load_x2 遇到空檔會丟例外；沒有檔＝模式 X 只看時間）。"""
    import csv
    feas, unm = [], []
    for m, tab in sorted(acc32.items()):
        for spec, c in sorted(tab.items()):
            seg, why = x3_segment(spec, m)
            row = dict(model=m, combo=spec, feasible=int(c["eps_pass"]), run_id=rid, ts=ts_, ratio=c["ratio"],
                       acc=round(c["acc"], 6), acc_e0=round(c["acc_e0"], 6), acc_delta=round(c["diff"]["mean"], 6),
                       acc_delta_lo=round(c["diff"]["lo"], 6), acc_delta_hi=round(c["diff"]["hi"], 6), eps=EPS,
                       mlperf99=int(c["mlperf0.99"]), mlperf999=int(c["mlperf0.999"]), n_items=c["n"], src_run_ids=src)
            if seg is not None:
                assert seg in X3_SEGMENTS, seg
                feas.append(dict(row, x3_segment=seg))
            else:
                unm.append(dict(row, reason=why))
    out = {}
    for name, rows, fields in (("x2_feasible.csv", feas, X3_FIELDS),
                               ("x2_unmapped.csv", unm, [f for f in X3_FIELDS if f != "x3_segment"] + ["reason"])):
        p = os.path.join(out_dir, name)
        if not rows:
            out[name] = "沒有列 → 不寫"
            continue
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        out[name] = dict(rows=len(rows), feasible=sum(r["feasible"] for r in rows))
    return out


def cmd_gen_report(a):
    paths = sorted({p for g in a.raw for p in glob.glob(g)})
    rows, partial = dedupe_complete(read_rows(paths), ("model", "ctx", "item_id", "spec"))
    rid = os.environ.get("RUN_ID", "adhoc")
    rep = dict(run_id=rid, ts=ts(), files=paths, src_run_ids=sorted({r["run_id"] for r in rows}), incomplete_items=partial,
               rule_text=STOP_TEXT, eps=EPS, mlperf=MLPERF, by_ctx={})
    L = ["# X2：挑出的組合的生成答對率（第 1 輪；判定照 L5d §6.3，MLPerf 門檻只報）", "",
         f"run：{'、'.join(rep['src_run_ids'])}；不齊全的題（沒用）：{len(partial)}", ""]
    acc32 = {}
    for ctx in sorted({r["ctx"] for r in rows}):
        for m in sorted({r["model"] for r in rows if r["ctx"] == ctx}):
            tab, a0 = gen_table([r for r in rows if r["ctx"] == ctx and r["model"] == m])
            if ctx == "32K":
                acc32[m] = tab
            rep["by_ctx"].setdefault(ctx, {})[m] = dict(acc_e0=a0, configs={s: {k: v for k, v in c.items() if k != "per_item"}
                                                                             for s, c in tab.items()})
            L += [f"## {m} {ctx}（E0 答對率 {a0 if a0 is None else round(100 * a0, 1)}%）", "",
                  "| 組合 | 角色 | 位元組比 | 答對率 | 對 E0（百分點，CI） | ε | ≥99% E0 | ≥99.9% E0 |", "|:--|:--|--:|--:|--:|:--|:--|:--|"]
            for c in sorted(tab.values(), key=lambda c: (c["ratio"], c["spec"])):
                L.append(f"| {c['spec']} | {'|'.join(c['roles'])} | {c['ratio']:.4f} | {100 * c['acc']:.1f} | {_ci(c['diff'], 100, 1)} | "
                         f"{'過' if c['eps_pass'] else '沒過'} | {'過' if c['mlperf0.99'] else '沒過'} | {'過' if c['mlperf0.999'] else '沒過'} |")
            L.append("")
    J = judge_x2(acc32, json.load(open(a.selection)) if a.selection and os.path.exists(a.selection) else {})
    n32 = {m: len({r["item_id"] for r in rows if r["ctx"] == "32K" and r["model"] == m and r["spec"] == "E0"}) for m in MODEL_INFO}
    J["n_items_32K"] = n32
    J["complete"] = all(v >= MIN_ITEMS for v in n32.values())
    if not J["complete"]:
        J["M1_verdict"] = f"INCOMPLETE（32K 每個模型至少要 {MIN_ITEMS} 題：{n32}；下面的成分只供參考）"
    rep["judge"] = J
    os.makedirs(a.out_dir, exist_ok=True)
    rep["x3_files"] = write_x3_files(acc32, rid, rep["ts"], ";".join(rep["src_run_ids"]), a.out_dir)
    L += ["## 給 X3 的可行集（32K；feasible＝ε 判定，MLPerf 99% 只另列）", "", json.dumps(rep["x3_files"], ensure_ascii=False), ""]
    json.dump(rep, open(os.path.join(a.out_dir, "x2_report.json"), "w"), indent=1, ensure_ascii=False, default=str)
    L += ["## 判定（事先寫死；32K）", "", "```", json.dumps({k: v for k, v in J.items() if k != "rule_text"}, ensure_ascii=False,
                                                             indent=1, default=str), "```", ""]
    L += [f"- **{k}**：{v}" for k, v in STOP_TEXT.items() if k in ("M1", "M2", "completeness", "MLPerf")]
    open(os.path.join(a.out_dir, "tables_x2.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:80]))


# ============================================================================ 10/21 延伸的報告（CPU；不碰 torch）
X2B_TEXT = {
    "design": f"X2b：新題 seed {X2B_SEED0} 起（T2–T5 各 {X2B_N} 題＝{4 * X2B_N} 題），和 X1（{X1_SEED0}）、校準（{CALIB_SEED0}）、X2（{X2_SEED0}）"
              "不重疊；32K；配置＝code/m16_xstack_combos_v2.json 的 x2b（協調者列的 10 個＋只用第一版 X2 決定的池子缺口單一方法："
              "兩個模型的 q:int2_kivi、Qwen 的 xl:k22:v34；Llama 11 個、Qwen 12 個；roles／budgets 照第一版 x2_selection.json）。",
    "pairing": "合併（X2＋X2b）：配置集合 C＝X2b 的配置裡、第一版 X2 有量的（≥90% 的 X2 題有這個配置；X2 每一層都是全部 96 題，"
               "少數題缺＝那題寫到一半）；題目＝E0 與 C 的每個配置都有的題（每題所有配置都有，逐題配對；被排除的題數另列）。"
               "X2b 有、X2 沒有的配置（例如 Qwen 的 xl:k22:v34+q:int8_token）只在「只用 X2b」報。",
    "rules": "M1／M2／完整性／MLPerf 照第一版寫死的規則（STOP_TEXT 的 M1、M2、completeness、MLPerf；judge_x2 原樣）重算，"
             "只是單一方法池與組合池限制在 C。:cal 的配置不進任何池（校準基底另判，見 xlcal）。",
    "pool_gap": "池子缺口（事先寫死；只標註、不改判定）：第一版池子裡、不在 C 的配置——單一方法若位元組比 < 合併後的 B*(單一)（或 B*(單一) 不存在）"
                "而且只用 X2 的 CI 上界 ≥ −ε（多量可能過 ε）→ 合併的 r 可能偏小（偏向 GO）；組合同理（位元組比 < B*(組合)（或不存在）"
                "且 CI 上界 ≥ −ε）→ r 可能偏大（偏向 NO-GO）。",
    "variants": "A＝只用 X2、第一版的整個池（重現第一版）；B＝只用 X2、池子限制在 C；C＝只用 X2b（池子 C；X2b 獨有的配置另列）；"
                "D＝X2＋X2b（池子 C）。主要讀 D；A→B＝限制池子的效果、B→D＝加題數的效果、C＝獨立重複。",
    "feasible_v2": "v2/x2_feasible.csv：每個配置用能拿到最多題的那一組：在 C → D；只在 X2b → C；其他 → A（第一版的題）；欄位 source 標明。",
    "complete": f"X2b 每個模型 {4 * X2B_N} 題齊全才算完成（x2b_complete）；判定本身照第一版的 ≥{MIN_ITEMS} 題規則。",
}
XLCAL_TEXT = {
    "design": f"xKV 的基底事先用校準題算好（seed {CALIB_SEED0} 起的 8 題，和 PCA 校準同一套、和所有評測題不重疊；每 4 層一個窗口、"
              "K（逆 RoPE）／V 各一個；不去平均的 ΣZᵀZ 的前 R 個特徵向量），所有題共用。比較＝同一題、同一個行程裡，"
              "校準基底版 vs 每題 SVD 的同名配置（同秩、同係數量化）。",
    "tf": f"TF 篩選（X1 的前 {XLCAL_TF_N} 題／任務＝{4 * XLCAL_TF_N} 題）：只報告、不判定。每題 SVD 的 TF 取自 X1 同一題的列（跨 run；"
          "E0 是否逐值重現另列），另在同一個行程量 xl:k11:v17 當錨點（同 run 配對＋和 X1 的重現）。",
    "rule": f"判定（生成答對率、X2 的 {4 * XLCAL_GEN_N} 題、逐題配對：校準 − 每題 SVD，bootstrap 10,000 次的 95% CI）："
            "CI 上界 < −2 個百分點 → MUST_PER_ITEM（必須每題算 SVD）；CI 下界 ≥ −2 個百分點 → CAN_PRECOMPUTE（可以事先算好基底）；"
            "其他 → UNDETERMINED（題數不夠分辨）。",
    "aggregate": f"每個模型：三對都 CAN_PRECOMPUTE → CAN_PRECOMPUTE；任一對 MUST_PER_ITEM → MUST_PER_ITEM（至少那個秩要每題算）；其他 → UNDETERMINED；"
                 f"題數 < {MIN_ITEMS} → INCOMPLETE。整體：兩個模型都 CAN → CAN；任一模型 MUST → MUST；其他照實寫。",
    "bytes": "同秩時，校準版的每請求位元組＝每題 SVD 版減掉每題的基底（基底跨請求共用，列在 bytes_shared，大小和每題的基底相同）。"
             "所以 CAN_PRECOMPUTE＝同秩不比每題 SVD 差、而且每請求更省。攤提〔算術〕：共用基底 S 由 N 個請求（或同一個請求的多次還原）"
             "分攤，每請求多 S/N；N＝1 時正好等於每題 SVD 版（打平點恆為 1），report 列 N＝1／8／64 的攤提後位元組比。"
             "實際部署只要存最高秩的一份基底（低秩取前幾欄），S 以最高秩那份為準會更大，這裡照每個配置自己的秩算。",
}


def _gen_rows(globs):
    """X2／X2b／xlcal 的生成逐列：讀檔、去重、只留做完的題、只留 32K。另回傳原始列（E0 重現檢查用）。"""
    paths = sorted({p for g in globs for p in glob.glob(g)})
    raw = read_rows(paths)
    rows, partial = dedupe_complete(raw, ("model", "ctx", "item_id", "spec"))
    return [r for r in rows if r.get("ctx", "32K") == "32K"], partial, paths, raw


def e0_repro(raw) -> dict:
    """同一題的 E0 被量了不只一次（X2 每一層的 run 各量一次 E0）：分數是否全部相同（決定性的證據；只記錄）。"""
    by = {}
    for r in raw:
        if r["spec"] == "E0" and r.get("ctx", "32K") == "32K":
            by.setdefault((r["model"], r["item_id"]), []).append(float(r["gen_score"]))
    multi = {k: v for k, v in by.items() if len(v) > 1}
    diff = sorted(f"{m}/{i}" for (m, i), v in multi.items() if len(set(v)) > 1)
    return dict(items_measured_more_than_once=len(multi), items_with_different_e0=len(diff), examples=diff[:10])


def restrict_rows(rows, specs):
    """只留 E0 與 specs 的列，而且只留 E0 與 specs 每一個都有的題（每題所有配置都有）。回傳 (列, 題的集合)。"""
    want = set(specs) | {"E0"}
    have = {}
    for r in rows:
        if r["spec"] in want:
            have.setdefault(r["item_id"], set()).add(r["spec"])
    items = {i for i, s in have.items() if s >= want}
    return [r for r in rows if r["spec"] in want and r["item_id"] in items], items


def with_roles(tab: dict, v1: dict, plan: dict) -> dict:
    """roles／budgets 一律照事先寫死的檔：第一版 x2_selection.json（M2 的主要組合、最好的單一方法），第一版沒有的照 X2b 計畫檔。"""
    for s, c in tab.items():
        e = v1.get(s) or plan.get(s)
        c["roles"] = [x for x in (e["roles"] if e else []) if x != "x2b"] or ["x2b_only"]
        c["budgets"] = [float(b) for b in (e["budgets"] if e else [])]
    return tab


def x2_single(c) -> bool:
    """judge_x2 的單一方法池（同一個條件；self-test 核對）。"""
    return c["n_lossy"] == 1 and bool(c["deployable"]) and not ({"deploy_broken", "random_drop"} & set(c["roles"]))


def x2_combo(c) -> bool:
    return c["n_lossy"] >= 2 and bool(c["deployable"]) and "random_drop" not in c["roles"]


def pool_gaps(full: dict, sub: dict, M1: dict) -> dict:
    """X2B_TEXT['pool_gap']：第一版池子裡、不在 sub 的配置，多量之後可能改變 B* 的。"""
    b1, b2 = M1.get("bstar_single"), M1.get("bstar_combo")
    out = {"single_bias_to_go": [], "combo_bias_to_nogo": []}
    for s, c in sorted(full.items()):
        if s in sub or c["diff"]["hi"] < -EPS:
            continue
        if x2_single(c) and (b1 is None or c["ratio"] < b1):
            out["single_bias_to_go"].append(dict(spec=s, ratio=c["ratio"], diff=c["diff"]))
        elif x2_combo(c) and (b2 is None or c["ratio"] < b2):
            out["combo_bias_to_nogo"].append(dict(spec=s, ratio=c["ratio"], diff=c["diff"]))
    return out


def _judge_variant(acc: dict) -> dict:
    J = judge_x2(acc, {})
    J.pop("rule_text", None)
    J["n_items"] = {m: (max(c["n"] for c in tab.values()) if tab else 0) for m, tab in acc.items()}
    J["complete"] = all(v >= MIN_ITEMS for v in J["n_items"].values()) and set(acc) == set(MODEL_INFO)
    if not J["complete"]:
        J["M1_verdict"] = f"INCOMPLETE（每個模型至少 {MIN_ITEMS} 題：{J['n_items']}；成分只供參考）／" + J["M1_verdict"]
    return J


def write_x3_files_v2(entries: list, rid: str, ts_: str, out_dir: str) -> dict:
    """v2 的 x2_feasible.csv／x2_unmapped.csv：同 write_x3_files 的欄位＋source（x2+x2b／x2b／x2）。entries＝[(model, spec, c, source, src_run_ids)]。"""
    import csv
    feas, unm = [], []
    for m, spec, c, source, src in sorted(entries, key=lambda e: (e[0], e[1])):
        if any(x.get("cal") for x in parse_spec(spec)):
            continue
        seg, why = x3_segment(spec, m)
        row = dict(model=m, combo=spec, feasible=int(c["eps_pass"]), run_id=rid, ts=ts_, ratio=c["ratio"],
                   acc=round(c["acc"], 6), acc_e0=round(c["acc_e0"], 6), acc_delta=round(c["diff"]["mean"], 6),
                   acc_delta_lo=round(c["diff"]["lo"], 6), acc_delta_hi=round(c["diff"]["hi"], 6), eps=EPS,
                   mlperf99=int(c["mlperf0.99"]), mlperf999=int(c["mlperf0.999"]), n_items=c["n"], src_run_ids=src, source=source)
        if seg is not None:
            assert seg in X3_SEGMENTS, seg
            feas.append(dict(row, x3_segment=seg))
        else:
            unm.append(dict(row, reason=why))
    out = {}
    for name, rows, fields in (("x2_feasible.csv", feas, X3_FIELDS + ["source"]),
                               ("x2_unmapped.csv", unm, [f for f in X3_FIELDS if f != "x3_segment"] + ["source", "reason"])):
        if not rows:
            out[name] = "沒有列 → 不寫"
            continue
        with open(os.path.join(out_dir, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        out[name] = dict(rows=len(rows), feasible=sum(r["feasible"] for r in rows))
    return out


def _guard_v2(out_dir: str):
    if os.path.abspath(out_dir) == os.path.abspath(OUT_DIR):
        raise SystemExit(f"--out-dir 是第一版的目錄 {OUT_DIR}：10/21 延伸的輸出一律寫到新目錄（預設 {OUT_DIR_V2}），不覆蓋第一版的檔")


def _hw(d):
    return None if not d else (d["hi"] - d["lo"]) / 2


def cmd_x2b_report(a):
    """X2＋X2b 的合併報告（X2B_TEXT）：A／B／C／D 四種並列、M1／M2／完整性／MLPerf 照第一版的規則重算，另寫 v2 的 x2_feasible.csv。"""
    _guard_v2(a.out_dir)
    rows2, part2, paths2, raw2 = _gen_rows(a.raw_x2)
    rowsb, partb, pathsb, rawb = _gen_rows(a.raw_x2b)
    v1sel = json.load(open(a.selection))
    plan = json.load(open(a.plan))
    rid = os.environ.get("RUN_ID", "adhoc")
    rep = dict(run_id=rid, ts=ts(), rule_text=X2B_TEXT, x2_rule_text={k: STOP_TEXT[k] for k in ("M1", "M2", "completeness", "MLPerf")},
               eps=EPS, files_x2=paths2, files_x2b=pathsb, src_run_ids_x2=sorted({r["run_id"] for r in rows2}),
               src_run_ids_x2b=sorted({r["run_id"] for r in rowsb}), incomplete_items_x2=part2, incomplete_items_x2b=partb,
               e0_repro_x2=e0_repro(raw2), e0_repro_x2b=e0_repro(rawb), plan_run_id=plan.get("run_id"),
               contention_x2b=dict(rows_with_intruders=sum(1 for r in rowsb if r.get("own_gpu_intruders") not in ("0", "NOT_CHECKED")),
                                   host_levels=sorted({r.get("host_level", "") for r in rowsb})), models={})
    acc = {k: {} for k in "ABCD"}
    full_tabs, entries = {}, []
    src2, srcb = ";".join(rep["src_run_ids_x2"]), ";".join(rep["src_run_ids_x2b"])
    for m in MODEL_INFO:
        v1 = {e["spec"]: e for e in v1sel["models"].get(m, {}).get("configs", [])}
        pl = {e["spec"]: e for e in plan["models"][m]["configs"]}
        specs_b = [s for s in pl if not any(x.get("cal") for x in parse_spec(s))]
        R2 = [r for r in rows2 if r["model"] == m]
        Rb = [r for r in rowsb if r["model"] == m]
        items2 = {r["item_id"] for r in R2 if r["spec"] == "E0"}
        have2 = {}
        for r in R2:
            have2.setdefault(r["spec"], set()).add(r["item_id"])
        C = [s for s in specs_b if items2 and len(have2.get(s, set()) & items2) >= 0.9 * len(items2)]
        only_b = [s for s in specs_b if s not in C]
        tabA, _ = gen_table(R2)
        rB, iB = restrict_rows(R2, C)
        rC, iC = restrict_rows(Rb, C)
        rCx, iCx = restrict_rows(Rb, specs_b)
        rD, iD = restrict_rows(R2 + Rb, C)
        assert not (iB & iC), "X2 與 X2b 的題重疊（seed 應該互不重疊）"
        tabs = {"A": tabA, "B": gen_table(rB)[0], "C": gen_table(rC)[0], "D": gen_table(rD)[0]}
        tabCx = with_roles(gen_table(rCx)[0], v1, pl)
        for k in tabs:
            with_roles(tabs[k], v1, pl)
            acc[k][m] = {s: c for s, c in tabs[k].items() if not any(x.get("cal") for x in parse_spec(s))}
        full_tabs[m] = tabs["A"]
        for s, c in tabs["A"].items():
            if s not in C and s not in only_b:
                entries.append((m, s, c, "x2", src2))
        for s in C:
            if s in tabs["D"]:
                entries.append((m, s, tabs["D"][s], "x2+x2b", src2 + ";" + srcb))
        for s in only_b:
            if s in tabCx:
                entries.append((m, s, tabCx[s], "x2b", srcb))
        rep["models"][m] = dict(C=C, only_x2b=only_b, n_items=dict(A=len(items2), B=len(iB), C=len(iC), C_all_x2b_configs=len(iCx), D=len(iD)),
                                x2b_complete=len(iCx) >= 4 * X2B_N,
                                items_dropped_x2=sorted(items2 - iB)[:20], n_items_dropped_x2=len(items2 - iB),
                                configs={k: {s: {kk: vv for kk, vv in c.items() if kk != "per_item"} for s, c in tabs[k].items()
                                             if k == "A" and s in C or k != "A"} for k in tabs},
                                configs_x2b_only={s: {kk: vv for kk, vv in tabCx[s].items() if kk != "per_item"} for s in only_b if s in tabCx})
    J = {k: _judge_variant(acc[k]) for k in "ABCD"}
    for k in "BCD":
        J[k]["pool_gaps"] = {m: pool_gaps(full_tabs[m], acc[k][m], J[k]["models"][m]["M1"]) for m in acc[k]}
    rep["judge"] = J
    os.makedirs(a.out_dir, exist_ok=True)
    rep["x3_files"] = write_x3_files_v2(entries, rid, rep["ts"], a.out_dir)
    json.dump(rep, open(os.path.join(a.out_dir, "x2b_report.json"), "w"), indent=1, ensure_ascii=False, default=str)
    L = ["# X2＋X2b：加題數後的生成答對率與判定（10/21 延伸；規則照第一版，事先寫死的合併方式見最後）", "",
         f"X2 run：{len(rep['src_run_ids_x2'])} 個；X2b run：{'、'.join(rep['src_run_ids_x2b']) or '（沒有）'}；"
         f"不齊全的題（沒用）：X2 {len(part2)}、X2b {len(partb)}", "",
         f"E0 重現（同一題量了不只一次）：X2 {rep['e0_repro_x2']['items_measured_more_than_once']} 題、分數不同 "
         f"{rep['e0_repro_x2']['items_with_different_e0']} 題", ""]
    for m in MODEL_INFO:
        M = rep["models"][m]
        L += [f"## {m}：題數 A／B／C／D＝{M['n_items']['A']}／{M['n_items']['B']}／{M['n_items']['C']}／{M['n_items']['D']}"
              f"（X2b 齊全：{'是' if M['x2b_complete'] else '否'}；X2 因缺配置被排除 {M['n_items_dropped_x2']} 題）", "",
              "| 組合 | 位元組比 | 只用 X2：答對率、對 E0（百分點，CI） | 只用 X2b | X2＋X2b | ε（D） | CI 半寬 X2→D | ≥99% E0（D） | ≥99.9% E0（D） |",
              "|:--|--:|--:|--:|--:|:--|--:|:--|:--|"]
        TB, TC, TD = (acc[k][m] for k in "BCD")
        for s in sorted(M["C"], key=lambda s: (TD[s]["ratio"] if s in TD else 9, s)):
            b, c, d = TB.get(s), TC.get(s), TD.get(s)

            def cell(x):
                return "—" if not x else f"{100 * x['acc']:.1f}、{_ci(x['diff'], 100, 1)}（n {x['n']}）"
            hw = f"{100 * _hw(b['diff']):.1f}→{100 * _hw(d['diff']):.1f}" if b and d else "—"
            L.append(f"| {s} | {(d or b or c)['ratio']:.4f} | {cell(b)} | {cell(c)} | {cell(d)} | "
                     f"{'過' if d and d['eps_pass'] else '沒過'} | {hw} | {'過' if d and d['mlperf0.99'] else '沒過'} | "
                     f"{'過' if d and d['mlperf0.999'] else '沒過'} |")
        for s in M["only_x2b"]:
            x = M["configs_x2b_only"].get(s)
            if x:
                L.append(f"| {s}（只有 X2b） | {x['ratio']:.4f} | — | {100 * x['acc']:.1f}、{_ci(x['diff'], 100, 1)}（n {x['n']}） | — | "
                         f"{'過' if x['eps_pass'] else '沒過'}（X2b） | — | {'過' if x['mlperf0.99'] else '沒過'} | {'過' if x['mlperf0.999'] else '沒過'} |")
        L.append("")
    L += ["## 判定（A＝只用 X2、第一版的池；B＝只用 X2、池 C；C＝只用 X2b；D＝X2＋X2b）", "",
          "| 變體 | 模型 | B*(單一) | B*(組合) | r | M1（整體） | MLPerf99 r | MLPerf99.9 r | 池子缺口（偏 GO／偏 NO-GO） |", "|:--|:--|--:|--:|--:|:--|--:|--:|:--|"]
    for k in "ABCD":
        for m in MODEL_INFO:
            Mm = J[k]["models"].get(m)
            if not Mm:
                continue
            g = J[k].get("pool_gaps", {}).get(m)
            gs = "—" if g is None else (f"{len(g['single_bias_to_go'])}：{', '.join(x['spec'] for x in g['single_bias_to_go']) or '無'}／"
                                         f"{len(g['combo_bias_to_nogo'])}：{', '.join(x['spec'] for x in g['combo_bias_to_nogo']) or '無'}")
            L.append(f"| {k} | {m} | {Mm['M1']['bstar_single']} | {Mm['M1']['bstar_combo']} | {Mm['M1']['r']} | {J[k]['M1_verdict']} | "
                     f"{Mm['M1_mlperf0.99']['r']} | {Mm['M1_mlperf0.999']['r']} | {gs} |")
    L += ["", "M2（每個預算：主要組合 − 最好的單一方法，配對差，百分點）：", "",
          "| 變體 | 模型 | 預算 | 主要組合 | 最好的單一 | 有空間 | 差 [CI] | 支持 NO-GO | Holm |", "|:--|:--|:--|:--|:--|:--|--:|:--|:--|"]
    for k in "ABCD":
        for m in MODEL_INFO:
            for B, e in J[k]["models"].get(m, {}).get("M2", {}).items():
                d = e.get("diff_primary_minus_single")
                L.append(f"| {k} | {m} | {B} | {e.get('primary')} | {e.get('best_single')} | {e.get('room', '—')} | "
                         f"{_ci(d, 100, 1) if d else e.get('status', '—')} | {e.get('supports_nogo', '—')} | "
                         f"{(e.get('holm') or {}).get('reject', '—')} |")
    L += ["", "## 給 X3 的可行集（v2；feasible＝ε 判定，source 標明用哪一組題）", "", json.dumps(rep["x3_files"], ensure_ascii=False), "",
          "## 事先寫死的合併方式與規則", ""]
    L += [f"- **{k}**：{v}" for k, v in X2B_TEXT.items()]
    L += [f"- **第一版 {k}**：{v}" for k, v in rep["x2_rule_text"].items()]
    open(os.path.join(a.out_dir, "tables_x2b.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:90]))


def xlcal_verdict(d) -> str:
    if d is None:
        return "NOT_MEASURED"
    if d["hi"] < -EPS:
        return "MUST_PER_ITEM"
    if d["lo"] >= -EPS:
        return "CAN_PRECOMPUTE"
    return "UNDETERMINED"


def xlcal_aggregate(vs: list) -> str:
    if not vs or "NOT_MEASURED" in vs:
        return "NOT_MEASURED"
    if "MUST_PER_ITEM" in vs:
        return "MUST_PER_ITEM"
    if all(v == "CAN_PRECOMPUTE" for v in vs):
        return "CAN_PRECOMPUTE"
    return "UNDETERMINED"


def _by_item_spec(rows):
    out = {}
    for r in rows:
        out.setdefault(r["model"], {}).setdefault(r["item_id"], {})[r["spec"]] = r
    return out


def cmd_xlcal_report(a):
    """xKV 校準基底 vs 每題 SVD（XLCAL_TEXT）：TF 篩選（只報）＋生成答對率的判定、位元組與攤提、和 X1／X2 的重現。"""
    _guard_v2(a.out_dir)
    pairs = xlcal_pairs()
    rid = os.environ.get("RUN_ID", "adhoc")
    tf_paths = sorted({p for g in a.raw_tf for p in glob.glob(g)})
    tf_rows, tf_part = dedupe_complete(read_rows(tf_paths), ("model", "item_id", "range", "spec"))
    x1_paths = sorted({p for g in a.raw_x1 for p in glob.glob(g)})
    want_x1 = {"E0"} | {ref for _, ref in pairs}
    x1_rows = [r for r in read_rows(x1_paths) if r["spec"] in want_x1 and r.get("range", "") in ("", "full")]
    gen_rows, gen_part, gen_paths, _ = _gen_rows(a.raw_gen)
    x2_rows = _gen_rows(a.raw_x2)[0] if a.raw_x2 else []
    rep = dict(run_id=rid, ts=ts(), rule_text=XLCAL_TEXT, eps=EPS, pairs=pairs, files_tf=tf_paths, files_gen=gen_paths,
               files_x1=x1_paths, src_run_ids_tf=sorted({r["run_id"] for r in tf_rows}),
               src_run_ids_gen=sorted({r["run_id"] for r in gen_rows}), incomplete_items_tf=tf_part, incomplete_items_gen=gen_part,
               calib_xl={}, models={})
    for p in sorted({p for g in (a.calib_xl_summary or []) for p in glob.glob(g)}):
        s = json.load(open(p))
        rep["calib_xl"][s["settings"]["model"]] = {k: s.get(k) for k in ("run_id", "ts", "sha256", "n_tokens", "R", "settings",
                                                                          "energy_captured_uncentered", "rope_probe")}
    TF, X1, G, X2 = _by_item_spec(tf_rows), _by_item_spec(x1_rows), _by_item_spec(gen_rows), _by_item_spec(x2_rows)
    verdicts = {}
    for m in MODEL_INFO:
        info = MODEL_INFO[m]
        M = rep["models"][m] = {}
        # ---- TF（只報）
        cal_specs = [c for c, _ in pairs]
        anchor = "xl:k11:v17"
        ours, x1 = TF.get(m, {}), X1.get(m, {})
        items = sorted(i for i, d in ours.items() if "E0" in d and all(c in d for c in cal_specs)
                       and i in x1 and "E0" in x1[i] and all(r in x1[i] for _, r in pairs))
        e0eq = [i for i in items if float(ours[i]["E0"]["gold_lp"]) == float(x1[i]["E0"]["gold_lp"])
                and ours[i]["E0"]["first_ok"] == x1[i]["E0"]["first_ok"]]
        anc = [i for i in items if anchor in ours[i]]
        anc_eq = [i for i in anc if float(ours[i][anchor]["gold_lp"]) == float(x1[i][anchor]["gold_lp"])]
        adiff = [abs(float(ours[i][anchor]["gold_lp"]) - float(x1[i][anchor]["gold_lp"])) for i in anc]
        tf = M["tf"] = dict(n_items=len(items), e0_bitwise_repro_vs_x1=f"{len(e0eq)}/{len(items)}",
                            anchor_bitwise_repro_vs_x1=f"{len(anc_eq)}/{len(anc)}", anchor_max_abs_diff=max(adiff) if adiff else None,
                            pairs={})

        def dlp(r):
            return float(r["gold_lp"]) - float(r["gold_lp_e0"])

        def dfo(r):
            return int(r["first_ok"]) - int(r["first_ok_e0"])
        for c, ref in pairs:
            e = tf["pairs"][f"{c} vs {ref}"] = dict(
                d_gold_lp_cal=boot2([dlp(ours[i][c]) for i in items]), d_gold_lp_ref_x1=boot2([dlp(x1[i][ref]) for i in items]),
                cal_minus_ref_gold_lp=boot2([dlp(ours[i][c]) - dlp(x1[i][ref]) for i in items]),
                cal_minus_ref_first_ok=boot2([dfo(ours[i][c]) - dfo(x1[i][ref]) for i in items]),
                kl_median_cal=statistics.median([float(ours[i][c]["kl"]) for i in items]) if items else None,
                kl_median_ref=statistics.median([float(x1[i][ref]["kl"]) for i in items]) if items else None)
            for p in ("relerr_k", "relerr_v", "attn_err"):
                vc = [_f(ours[i][c], p) for i in items if _f(ours[i][c], p) is not None]
                vr = [_f(x1[i][ref], p) for i in items if _f(x1[i][ref], p) is not None]
                e[f"{p}_cal"] = statistics.mean(vc) if vc else None
                e[f"{p}_ref"] = statistics.mean(vr) if vr else None
            if c.startswith(anchor + ":cal") and anc:
                e["within_run_cal_minus_anchor_gold_lp"] = boot2([dlp(ours[i][c]) - dlp(ours[i][anchor]) for i in anc])
        # ---- 生成（判定）
        g = G.get(m, {})
        specs = [s for pr in pairs for s in pr]
        gi = sorted(i for i, d in g.items() if "E0" in d and all(s in d for s in specs))
        gen = M["gen"] = dict(n_items=len(gi), configs={}, pairs={})
        for s in specs:
            sc = [float(g[i][s]["gen_score"]) for i in gi]
            e0 = [float(g[i]["E0"]["gen_score"]) for i in gi]
            d = boot2([x - y for x, y in zip(sc, e0)]) if gi else None
            br = int(g[gi[0]][s]["bytes_region"]) if gi else None
            gen["configs"][s] = dict(acc=statistics.mean(sc) if gi else None, acc_e0=statistics.mean(e0) if gi else None, diff=d,
                                     eps_pass=bool(d and d["lo"] >= -EPS), ratio=float(g[gi[0]][s]["ratio_region"]) if gi else None,
                                     bytes_region=br)
        vs = []
        for c, ref in pairs:
            d = boot2([float(g[i][c]["gen_score"]) - float(g[i][ref]["gen_score"]) for i in gi]) if gi else None
            v = xlcal_verdict(d)
            stg = validate(parse_spec(c), info, N_BLOCKS - 1)
            shared = sum(x for k, x in predict_bytes(stg, info, N_BLOCKS - 1).items() if k.startswith("shared."))
            bc, bref = gen["configs"][c]["bytes_region"], gen["configs"][ref]["bytes_region"]
            cb = info["L"] * 2 * info["H"] * CHUNK * info["D"] * 2 * (N_BLOCKS - 1)
            gen["pairs"][f"{c} vs {ref}"] = dict(cal_minus_ref=d, verdict=v, bytes_region_cal=bc, bytes_region_ref=bref,
                                                 bytes_shared_basis=shared,
                                                 amortized_ratio=({str(N): round((bc + shared / N) / cb, 6) for N in (1, 8, 64)}
                                                                  if bc is not None else None))
            vs.append(v)
        verdicts[m] = "INCOMPLETE" if len(gi) < MIN_ITEMS else xlcal_aggregate(vs)
        gen["verdict"] = verdicts[m]
        rp = [(i, s) for i in gi for s in ["E0"] + [ref for _, ref in pairs] if i in X2.get(m, {}) and s in X2[m][i]]
        gen["repro_vs_x2"] = dict(n=len(rp), equal=sum(1 for i, s in rp if float(X2[m][i][s]["gen_score"]) == float(g[i][s]["gen_score"])),
                                  by_spec={s: f"{sum(1 for i, t in rp if t == s and float(X2[m][i][s]['gen_score']) == float(g[i][s]['gen_score']))}"
                                              f"/{sum(1 for _, t in rp if t == s)}" for s in ["E0"] + [ref for _, ref in pairs]})
    vals = list(verdicts.values())
    rep["verdict"] = ("CAN_PRECOMPUTE" if vals and all(v == "CAN_PRECOMPUTE" for v in vals) else
                      "MUST_PER_ITEM" if "MUST_PER_ITEM" in vals else "；".join(f"{m} {v}" for m, v in verdicts.items()))
    rep["verdict_by_model"] = verdicts
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(rep, open(os.path.join(a.out_dir, "xlcal_report.json"), "w"), indent=1, ensure_ascii=False, default=str)
    L = ["# xKV：事先校準的基底 vs 每題 SVD（10/21 延伸；判定事先寫死，見最後）", "",
         f"TF run：{'、'.join(rep['src_run_ids_tf']) or '（沒有）'}；生成 run：{'、'.join(rep['src_run_ids_gen']) or '（沒有）'}", "",
         f"**整體判定：{rep['verdict']}**（每個模型：{json.dumps(verdicts, ensure_ascii=False)}）", ""]
    for m in MODEL_INFO:
        M = rep["models"][m]
        cx = rep["calib_xl"].get(m, {})
        L += [f"## {m}", "", f"校準基底：run {cx.get('run_id', 'NOT_FOUND')}、{cx.get('n_tokens')} token、R＝{cx.get('R')}", "",
              f"### 生成（{M['gen']['n_items']} 題；判定 {M['gen']['verdict']}；和 X2 的逐值重現 {M['gen']['repro_vs_x2']['equal']}/{M['gen']['repro_vs_x2']['n']}）", "",
              "| 校準版 vs 每題 SVD | 校準 − 每題 SVD（百分點，CI） | 判定 | 每請求位元組比 校準／每題 | 共用基底（MB） | 攤提後 N＝1／8／64 |", "|:--|--:|:--|--:|--:|--:|"]
        for k, e in M["gen"]["pairs"].items():
            c, ref = k.split(" vs ")
            rc, rr = M["gen"]["configs"][c]["ratio"], M["gen"]["configs"][ref]["ratio"]
            am = e["amortized_ratio"] or {}
            L.append(f"| {k} | {_ci(e['cal_minus_ref'], 100, 1)} | {e['verdict']} | {rc}／{rr} | {e['bytes_shared_basis'] / 1e6:.1f} | "
                     f"{'／'.join(str(am.get(x)) for x in ('1', '8', '64'))} |")
        L += ["", "| 配置 | 位元組比 | 答對率 | 對 E0（百分點，CI） | ε |", "|:--|--:|--:|--:|:--|"]
        for s, c in M["gen"]["configs"].items():
            if c["acc"] is not None:
                L.append(f"| {s} | {c['ratio']:.4f} | {100 * c['acc']:.1f} | {_ci(c['diff'], 100, 1)} | {'過' if c['eps_pass'] else '沒過'} |")
        L += ["", f"### TF 篩選（{M['tf']['n_items']} 題；只報）：E0 和 X1 逐值重現 {M['tf']['e0_bitwise_repro_vs_x1']}、"
                  f"錨點 xl:k11:v17 和 X1 逐值重現 {M['tf']['anchor_bitwise_repro_vs_x1']}", "",
              "| 校準版 vs 每題 SVD（X1） | Δgold_lp 校準 | Δgold_lp 每題 | 校準 − 每題 Δgold_lp | 校準 − 每題 Δfirst_ok（百分點） | relerr K 校準／每題 | relerr V 校準／每題 |",
              "|:--|--:|--:|--:|--:|--:|--:|"]
        for k, e in M["tf"]["pairs"].items():
            def f4(x):
                return "—" if x is None else f"{x:.4f}"
            L.append(f"| {k} | {_ci(e['d_gold_lp_cal'])} | {_ci(e['d_gold_lp_ref_x1'])} | {_ci(e['cal_minus_ref_gold_lp'])} | "
                     f"{_ci(e['cal_minus_ref_first_ok'], 100, 1)} | {f4(e['relerr_k_cal'])}／{f4(e['relerr_k_ref'])} | "
                     f"{f4(e['relerr_v_cal'])}／{f4(e['relerr_v_ref'])} |")
        L.append("")
    L += ["## 事先寫死的設計與判定", ""] + [f"- **{k}**：{v}" for k, v in XLCAL_TEXT.items()]
    open(os.path.join(a.out_dir, "tables_xlcal.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:80]))


# ============================================================================ gpu-selftest
def cmd_gpu_selftest(a):
    if getattr(a, "new_checks", "none") != "none":
        return cmd_gpu_selftest_new(a)          # 10/21：只跑新的檢查；原本的檢查照舊用不加 --new-checks 的指令
    import numpy as np
    import torch
    import m10_codec_x as X10
    import m10_sweep as SW
    import m10_tasks as T
    from m10_quality import Answerer
    res, ok = {"checks": {}, "model": a.model, "run_id": os.environ.get("RUN_ID", "adhoc"), "ts": ts()}, True

    def check(name, cond, **kw):
        nonlocal ok
        ok &= bool(cond)
        res["checks"][name] = {"pass": bool(cond), **kw}
        print(("PASS " if cond else "FAIL ") + name, json.dumps(kw, default=str)[:400], flush=True)

    info = MODEL_INFO[a.model]
    ans = Answerer(a.model)
    km = ans.km
    check_info(km, info)
    calib = load_calib(a.calib, km, a.model)
    P = json.load(open(a.params_file))["params"][a.model]
    it = T.make_item("T2", "front", "back", X1_SEED0, ans.tok, P.get("T2"))
    cap = []
    hk = km.hf.model.layers[0].self_attn.k_proj.register_forward_hook(lambda m, i, o: cap.append(o.detach().clone()))
    try:
        st = SW.Item(ans, it)
    finally:
        hk.remove()
    n = st.ctx_len
    tctx = TCtx(km, n, calib=calib, deep=deep_layers(km.L))
    # 1. RoPE（規則 6）
    pr = rope_probe(km, cap[1][0], tctx, CHUNK)
    check("rope_probe_k_proj_vs_inverse", pr["pass"], **pr)
    c_np = tctx.cos[0, :, :km.D // 2].double().cpu().numpy()
    s_np = tctx.sin[0, :, :km.D // 2].double().cpu().numpy()
    blk = km.kv[0, 0, :, CHUNK:2 * CHUNK]
    P1 = torch.arange(CHUNK, 2 * CHUNK, device=km.device)
    ti = t_rope_inv(blk, *tctx.cs(P1)).double().cpu().numpy()
    ni = np_rope_inv(blk.float().cpu().numpy().astype(np.float64), c_np[CHUNK:2 * CHUNK], s_np[CHUNK:2 * CHUNK])
    d = float(np.abs(ti - ni).max() / np.abs(ni).max())
    check("rope_inverse_torch_vs_numpy", d <= 1e-5, max_rel=d)
    fw = t_rope_fwd(torch.tensor(ni, device=km.device, dtype=torch.float32), *tctx.cs(P1), km.apply_rope).float()
    d = float((fw - blk.float()).norm() / blk.float().norm())
    check("rope_inverse_then_forward_roundtrip", d <= 1e-2, rel=d)
    # 2. 量化：pipeline 的原始 KV 路徑＝m10_codec_x 逐位元；位元組＝Enc.nbytes
    x1 = st.bf16[1]
    fmts = list(SW.FORMATS[a.model]) + ["kv-bf16-int4_kivi", "kv-int2_kivi-int4_kivi"]
    for f in fmts:
        dec, idx, by, _ = t_pipeline([dict(op="q", fmt=f)], x1, torch.arange(CHUNK, 2 * CHUNK, device=km.device), tctx, info)
        e = X10.encode(x1, f)
        ref = X10.decode(e)
        check(f"q_bitwise_vs_m10_codec_x[{f}]", torch.equal(dec, ref) and by == {"q": e.nbytes} == predict_bytes(
            [dict(op="q", fmt=f)], info, 1), nbytes=by, enc_nbytes=e.nbytes)
    dec, _, by, _ = t_pipeline([dict(op="q", fmt="vllm_fp8")], x1, torch.arange(CHUNK, 2 * CHUNK, device=km.device), tctx, info)
    ref = np_bf16(np_fnuz(x1.float().cpu().numpy()))
    same = np.array_equal(dec.float().cpu().numpy(), ref)
    check("vllm_fp8_torch_fnuz_vs_numpy_grid", same and by["q"] == x1.numel(),
          n_diff=int((dec.float().cpu().numpy() != ref).sum()), clamped_frac=float((x1.float().abs() > FNUZ_MAX).float().mean()),
          max_abs_k=float(x1[:, 0].float().abs().max()))
    # 係數用的 KIVI（enc_half，K 角色）在原始 K 上＝第二階段 int4_kivi 的 K 半邊（同一個定義）
    full = X10.decode(X10.encode(x1, "int4_kivi"))
    for hi, role in ((0, "k"), (1, "v")):
        dh, nbh = t_halves_enc(x1[:, hi], "kivi4", role)
        check(f"coef_kivi4_role_{role}_eq_int4_kivi_half", torch.equal(dh.to(torch.bfloat16), full[:, hi]), nbytes=nbh)
    # 3. 小切片：前 8 層、block 1–4（2048 token）上 torch vs numpy 參考
    Ls = 8
    Xs = km.kv[:Ls, :, :, CHUNK:5 * CHUNK].clone()
    pos = torch.arange(CHUNK, 5 * CHUNK, device=km.device)
    cal_s = calib_for_layers(calib, 0, Ls)
    tcs = TCtx(km, n, calib=cal_s, orders={"KEYDIFF": [2, 0, 3, 1], "RND0": [1, 3, 0, 2]}, deep=list(range(Ls // 2, Ls)))
    ncal = {k: v.double().cpu().numpy() for k, v in cal_s.items() if hasattr(v, "shape")}
    ncx = NCtx(c_np, s_np, calib=ncal, orders=tcs.orders, deep=tcs.deep)
    Xn = Xs.float().cpu().numpy()
    posn = np.arange(CHUNK, 5 * CHUNK)
    info_s = dict(info, L=Ls)
    tol = {"exact": 0.0, "merge": 1e-3, "lr": 1e-2, "mc": 1e-2}
    specs = [("none", "exact"), ("drop:KEYDIFF:0.5", "exact"), ("q:int4_kivi", "exact"), ("q:int2_kivi", "exact"),
             ("q:int8_token", "exact"), ("q:vllm_fp8", "exact"), ("drop:KEYDIFF:0.5+q:kv-int4_kivi-int2_kivi", "exact"),
             ("share", "exact"), ("merge:4", "merge"), ("merge:2+q:int4_kivi", "lr"), ("lr:k24:v24", "lr"),
             ("lr:k16:v16:joint", "lr"), ("lr:k32:v32+q:int4_kivi", "lr"), ("lr:k24+q:kv-bf16-int4_kivi", "lr"),
             ("lr:k32:v32+q:int8_token", "lr"), ("drop:KEYDIFF:0.5+lr:k24+q:int4_kivi", "lr"),
             ("drop:KEYDIFF:0.25+merge:2+q:kv-int4_kivi-int2_kivi", "lr"), ("mc", "mc")]
    for spec, kind in specs:
        stg = validate(parse_spec(spec), info_s, 4)
        with torch.inference_mode():
            td, tidx, tby, _ = t_pipeline(stg, Xs, pos, tcs, info_s)
        nd_, nidx, nby, _ = np_pipeline(stg, Xn, posn, ncx, info_s)
        tdn = td.float().cpu().numpy()
        same_idx = (tidx is None and nidx is None) or (tidx is not None and nidx is not None and np.array_equal(tidx.cpu().numpy(), nidx))
        pb = predict_bytes(stg, info_s, 4)
        if kind == "exact":
            nbad = int((tdn != nd_).sum())
            cond = nbad <= 1e-5 * tdn.size
            check(f"torch_vs_numpy[{spec}]", cond and same_idx and tby == nby == pb, n_diff=nbad, n=tdn.size, bytes=tby, np_bytes=nby)
        else:
            rel = float(np.linalg.norm(tdn - nd_) / max(np.linalg.norm(nd_), 1e-30))
            check(f"torch_vs_numpy[{spec}]", rel <= tol[kind] and same_idx and tby == nby == pb, rel=rel, tol=tol[kind], bytes=tby,
                  np_bytes=nby)
    # xKV：隨機 SVD（torch）vs 精確 SVD（numpy）的重建誤差比
    for spec in ("xl:k11:v17", "xl:k5:v9", "xl:k11:v17+drop:KEYDIFF:0.5"):
        stg = validate(parse_spec(spec), info_s, 4)
        with torch.inference_mode():
            td, tidx, tby, _ = t_pipeline(stg, Xs, pos, tcs, info_s)
        nd_, nidx, nby, ex = np_pipeline(stg, Xn, posn, ncx, info_s)
        ref = Xn if nidx is None else Xn[:, :, :, nidx]
        et = float(np.linalg.norm(td.float().cpu().numpy() - ref) / np.linalg.norm(ref))
        en = float(np.linalg.norm(nd_ - ref) / np.linalg.norm(ref))
        check(f"xl_randomized_vs_exact[{spec}]", 0.999 * en <= et <= 1.05 * en and tby == nby == predict_bytes(stg, info_s, 4),
              err_torch=et, err_exact=en, svd_err=ex)
    # lr 滿秩：只剩 RoPE 逆轉／轉回與 BF16 捨入
    stg = validate(parse_spec("lr:k128:v128"), info, N_BLOCKS - 1)
    Xf = km.kv[:, :, :, CHUNK:n]
    with torch.inference_mode():
        td, _, _, _ = t_pipeline(stg, Xf, torch.arange(CHUNK, n, device=km.device), tctx, info)
        rel = t_relerr(Xf, td, None)
    check("lr_full_rank_relerr", rel["relerr_k"] <= 1e-2 and rel["relerr_v"] <= 1e-2, **rel)
    del td
    # 4. merge：AsymKV 等價式在真實 KV 上（value 原樣 vs 存組平均再展開；同一組 key＝k*）
    q = torch.randn(4, km.D, device=km.device, generator=torch.Generator(device=km.device).manual_seed(0))
    K = km.kv[5, 0, 0, CHUNK:2 * CHUNK].float()
    V = km.kv[5, 1, 0, CHUNK:2 * CHUNK].float()
    for g in (2, 4, 8):
        Km = K.view(-1, g, km.D).mean(1).repeat_interleave(g, 0)
        Vm = V.view(-1, g, km.D).mean(1).repeat_interleave(g, 0)
        o1 = (q @ Km.T / math.sqrt(km.D)).softmax(-1) @ V
        o2 = (q @ Km.T / math.sqrt(km.D)).softmax(-1) @ Vm
        d = float((o1 - o2).norm() / o1.norm())
        check(f"asymkv_equivalence_g{g}", d <= 1e-4, rel=d)
    # 5. 代理：none＝0；drop＝只在留下的 key 上算
    lp0, qcap = capture_q(ans, st, 64)
    prox = Proxy(km, qcap, [0, km.L // 2, km.L - 1], n)
    z = prox.measure(None)
    check("proxy_zero_for_identity", z["attn_err"] == 0 and z["attn_err_wo"] == 0, **z)
    keep = torch.ones(n, dtype=torch.bool, device=km.device)
    keep[CHUNK * 10:CHUNK * 20] = False
    o, _ = prox._out(0, keep)
    Kk, Vk = km.kv[0, 0, :, :n][:, keep].float(), km.kv[0, 1, :, :n][:, keep].float()
    o2 = torch.matmul((torch.matmul(prox.qg[0], Kk.transpose(1, 2)) / math.sqrt(km.D)).softmax(-1), Vk)
    d = float((o - o2).norm() / o2.norm())
    # 10/20 冒煙：Llama 7.8e-8、Qwen 7.1e-5（run 20261020-140604-m16-x-gpuselftest-qwen）。兩邊都是 FP32，差在 GEMM 形狀不同 →
    # 加總順序不同；Qwen 第 0 層 K 有大值（max|K|≈422），分數大、softmax 放大這種捨入差。遮罩寫錯（等於少掉 10 個 block）會是
    # 量級大得多的差，所以門檻 1e-3；另記「完全不遮」的差當反向對照（只記錄，不擋）。
    o_none, _ = prox._out(0, None)
    d_neg = float((o_none - o2).norm() / o2.norm())
    check("proxy_masked_eq_subset", d <= 1e-3, rel=d, rel_unmasked_vs_subset=d_neg)
    # 6. TF：pipeline 放 int4_kivi（全段）＝第二階段 load_fmt／place 的同一格（逐值）
    import m16_s2_equalbytes as S2
    X = km.kv[:, :, :, CHUNK:n]
    with torch.inference_mode():
        dec, _, _, _ = t_pipeline([dict(op="q", fmt="int4_kivi")], X, torch.arange(CHUNK, n, device=km.device), tctx, info)
        X.copy_(dec)
    kk = S2.keep_tensor(km, [])
    m1 = S2.tf(st, lp0, kk)
    placed = X.clone()
    restore_blocks(km, st, range(1, n // CHUNK))
    st.load_fmt("int4_kivi")
    st.place(frozenset(range(1, n // CHUNK)))
    same_kv = torch.equal(km.kv[:, :, :, CHUNK:n], placed)
    m2 = S2.tf(st, lp0, kk)
    check("tf_pipeline_eq_load_fmt_place[int4_kivi]", same_kv and m1 == m2, m_pipeline=m1, m_place=m2)
    st.place(frozenset())
    res["all_pass"] = ok
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False, default=str)
    print("ALL PASS" if ok else "SOME FAILED", f"（{len(res['checks'])} 項）", flush=True)
    sys.exit(0 if ok else 1)


def cmd_gpu_selftest_new(a):
    """10/21 延伸的 GPU 檢查（38b／38c 開頭的冒煙）。--new-checks x2b：CPU 版出題＝m10_quality、X2b 每個配置的位元組＝預測、
    --xl-cache 和不用快取逐位元相同；xlcal：校準基底的格式與正交、torch vs numpy、Eckart–Young（校準基底的誤差不會比同一份資料的
    精確 SVD 小）、整個 32K 區的位元組與有限值。題目＝DS1／X1 的第一題（seed 60000）。"""
    import numpy as np
    import torch
    import m10_sweep as SW
    import m10_tasks as T
    from m10_quality import Answerer
    res, ok = {"checks": {}, "model": a.model, "new_checks": a.new_checks, "run_id": os.environ.get("RUN_ID", "adhoc"), "ts": ts()}, True

    def check(name, cond, **kw):
        nonlocal ok
        ok &= bool(cond)
        res["checks"][name] = {"pass": bool(cond), **kw}
        print(("PASS " if cond else "FAIL ") + name, json.dumps(kw, default=str)[:400], flush=True)

    info = MODEL_INFO[a.model]
    ans = Answerer(a.model)
    km = ans.km
    check_info(km, info)
    P = json.load(open(a.params_file))["params"][a.model]
    it = T.make_item("T2", "front", "back", X1_SEED0, ans.tok, P.get("T2"))
    st = SW.Item(ans, it)
    n = st.ctx_len
    tctx = TCtx(km, n, deep=deep_layers(km.L))
    c_np = tctx.cos[0, :, :km.D // 2].double().cpu().numpy()
    s_np = tctx.sin[0, :, :km.D // 2].double().cpu().numpy()
    Ls, W = 8, 4
    Xs = km.kv[:Ls, :, :, CHUNK:5 * CHUNK].clone()
    pos = torch.arange(CHUNK, 5 * CHUNK, device=km.device)
    Xn = Xs.float().cpu().numpy()
    posn = np.arange(CHUNK, 5 * CHUNK)
    info_s = dict(info, L=Ls)
    orders = {"KEYDIFF": [2, 0, 3, 1]}
    want = a.new_checks
    if want in ("x2b", "both"):
        for tasks, nn, s0 in ((["T2", "T3", "T4", "T5"], X2B_N, X2B_SEED0), (["T2", "T3", "T4", "T5"], XLCAL_GEN_N, X2_SEED0),
                              (["T2", "T3", "T4", "T5"], XLCAL_TF_N, X1_SEED0)):
            c1, c2 = plan_interleaved_cpu(tasks, nn, s0), plan_interleaved(tasks, nn, s0)
            check(f"plan_cpu_eq_m10[{s0}x{nn}]", c1 == c2, n=len(c1))
        tcs = TCtx(km, n, orders=orders, deep=list(range(Ls // 2, Ls)))
        for spec in plan_specs("x2b", a.model):
            stg = validate(parse_spec(spec), info_s, 4)
            with torch.inference_mode():
                td, _, tby, _ = t_pipeline(stg, Xs, pos, tcs, info_s)
            check(f"x2b_slice_bytes_finite[{spec}]", tby == predict_bytes(stg, info_s, 4) and bool(torch.isfinite(td.float()).all()),
                  bytes=tby)
        # --xl-cache：先跑 +int8（存快取）再跑 +int4（用快取）＝不用快取的 +int4；另記不用快取跑兩次是不是逐位元相同（決定性）
        for sa, sb in (("xl:k22:v34+q:int8_token", "xl:k22:v34+q:int4_kivi"), ("xl:k11:v17", "xl:k11:v17+q:int4_kivi")):
            cfgs = [dict(spec=s, stages=validate(parse_spec(s), info_s, 4)) for s in (sa, sb)]
            tcc = TCtx(km, n, orders=orders, deep=list(range(Ls // 2, Ls)))
            tcc.xl_cache = xl_cache_plan(cfgs)
            with torch.inference_mode():
                da, _, ba, _ = t_pipeline(cfgs[0]["stages"], Xs, pos, tcc, info_s)
                dc, _, bc, _ = t_pipeline(cfgs[1]["stages"], Xs, pos, tcc, info_s)
                du, _, bu, _ = t_pipeline(cfgs[1]["stages"], Xs, pos, tcs, info_s)
                du2, _, _, _ = t_pipeline(cfgs[1]["stages"], Xs, pos, tcs, info_s)
                dau, _, _, _ = t_pipeline(cfgs[0]["stages"], Xs, pos, tcs, info_s)
            det = bool(torch.equal(du, du2))
            same = bool(torch.equal(dc, du) and torch.equal(da, dau))
            rel = float((dc.float() - du.float()).norm() / du.float().norm().clamp_min(1e-30))
            rel2 = float((du2.float() - du.float()).norm() / du.float().norm().clamp_min(1e-30))
            # 決定性時要逐位元相同；SVD 本身不決定（du≠du2）時，快取的差不能比「重算一次」的差大（只會更一致）
            cond = (same if det else rel <= 2 * rel2 + 1e-7) and bc == bu and tcc.xl_cache["hits"] == 2 and not tcc.xl_cache["store"]
            check(f"xl_cache_eq_uncached[{sa}→{sb}]", cond, bitwise_equal=same, svd_deterministic=det, rel=rel, rel_recompute=rel2,
                  hits=tcc.xl_cache["hits"], misses=tcc.xl_cache["misses"])
    if want in ("xlcal", "both"):
        cal = load_calib_xl(a.calib_xl, km, a.model)
        meta = cal["meta"]
        R = meta["R"]
        need = max(r for s in plan_specs("xlcal", a.model) for x in parse_spec(s) if x["op"] == "xl" for r in (x["rk"], x["rv"]))
        orth = {}
        for h in "kv":
            U = cal[f"{h}_U"]
            I = torch.eye(U.shape[-1], device=U.device)
            orth[h] = float((U.transpose(-1, -2) @ U - I).abs().max())
        check("xlcal_meta_shape_orthonormal", meta["W"] == W and R >= need * W * km.H and tuple(cal["k_U"].shape) == (km.L // W, W * km.H * km.D, R)
              and max(orth.values()) <= 1e-3, R=R, need=need * W * km.H, orth_maxdev=orth, run_id=meta["run_id"],
              energy=meta.get("energy_captured_uncentered"))
        cal_s = {f"{h}_U": cal[f"{h}_U"][:Ls // W] for h in "kv"}
        tcs = TCtx(km, n, orders=orders, deep=list(range(Ls // 2, Ls)))
        tcs.xlcal = cal_s
        ncx = NCtx(c_np, s_np, orders=orders, deep=tcs.deep, xlcal={k: v.double().cpu().numpy() for k, v in cal_s.items()})
        for spec in ("xl:k11:v17:cal", "xl:k22:v34:cal+q:int4_kivi", "xl:k28:v42:cal+q:int4_kivi", "drop:KEYDIFF:0.5+xl:k11:v17:cal"):
            stg = validate(parse_spec(spec), info_s, 4)
            with torch.inference_mode():
                td, tidx, tby, _ = t_pipeline(stg, Xs, pos, tcs, info_s)
            nd_, nidx, nby, ex = np_pipeline(stg, Xn, posn, ncx, info_s)
            same_idx = (tidx is None and nidx is None) or (tidx is not None and nidx is not None and np.array_equal(tidx.cpu().numpy(), nidx))
            tdn = td.float().cpu().numpy()
            rel = float(np.linalg.norm(tdn - nd_) / max(np.linalg.norm(nd_), 1e-30))
            pb = predict_bytes(stg, info_s, 4)
            check(f"xlcal_torch_vs_numpy[{spec}]", rel <= 1e-2 and same_idx and tby == nby == pb
                  and not any(k in tby for k in ("k.basis", "v.basis")), rel=rel, bytes=tby, cal_err=ex)
        # Eckart–Young：同一份資料上，校準基底的誤差 ≥ 精確 SVD（numpy）的誤差（每個窗口、K／V）
        for rk, rv in ((11, 17), (22, 34)):
            _, _, _, ex_c = np_pipeline(validate(parse_spec(f"xl:k{rk}:v{rv}:cal"), info_s, 4), Xn, posn, ncx, info_s)
            _, _, _, ex_s = np_pipeline(validate(parse_spec(f"xl:k{rk}:v{rv}"), info_s, 4), Xn, posn, ncx, info_s)
            ratio = {h: [c / s for c, s in zip(ex_c[f"{h}.cal_err"], ex_s[f"{h}.svd_err"])] for h in "kv"}
            check(f"xlcal_eckart_young[k{rk}v{rv}]", all(x >= 0.999 for v in ratio.values() for x in v), err_ratio_cal_over_svd=ratio,
                  cal_err=ex_c, svd_err=ex_s)
        # 整個 32K 區（block 1..最後）：位元組＝預測、沒有每題基底、有限值；記重建誤差與時間（只記錄）
        tcf = TCtx(km, n, deep=deep_layers(km.L))
        tcf.xlcal = cal
        Xf = km.kv[:, :, :, CHUNK:n]
        nreg = n // CHUNK - 1
        for spec in plan_specs("xlcal", a.model):
            stg = validate(parse_spec(spec), info, nreg)
            if not any(x.get("cal") for x in stg):
                continue
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.inference_mode():
                td, _, tby, _ = t_pipeline(stg, Xf, torch.arange(CHUNK, n, device=km.device), tcf, info)
                torch.cuda.synchronize()
                tp = round(time.time() - t0, 3)
                rel = t_relerr(Xf, td, None)
            check(f"xlcal_full_region[{spec}]", tby == predict_bytes(stg, info, nreg) and bool(torch.isfinite(td.float()).all()),
                  bytes=tby, t_pipe_s=tp, **rel)
            del td
    res["all_pass"] = ok
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False, default=str)
    print("ALL PASS" if ok else "SOME FAILED", f"（{len(res['checks'])} 項）", flush=True)
    sys.exit(0 if ok else 1)


# ============================================================================ --self-test（CPU；不准 import torch）
def self_test() -> bool:
    np = _np()
    res, ok = {}, True

    def check(name, cond, **kw):
        nonlocal ok
        ok &= bool(cond)
        res[name] = {"pass": bool(cond), **kw}
        print(("PASS " if cond else "FAIL ") + name, json.dumps(kw, default=str, ensure_ascii=False)[:300], flush=True)

    rng = np.random.default_rng(0)
    # 1. 組合清單：每一條都合法、數量、每組有理由
    d = json.load(open(COMBOS))
    for m in MODEL_INFO:
        cs = load_combos(COMBOS, m)
        nall = len(cs)
        check(f"combos_count[{m}]", 120 <= nall <= 200, n=nall)
        check(f"combos_roundtrip[{m}]", all(canonical(parse_spec(c["spec"])) == c["spec"] for c in cs))
        sm = load_combos(COMBOS, m, smoke=True)
        check(f"smoke_valid[{m}]", len(sm) == len(d["smoke"]), n=len(sm))
        ratios = {c["spec"]: bytes_summary(predict_bytes(c["stages"], MODEL_INFO[m], 63), MODEL_INFO[m], 63, 64)["ratio_region"]
                  for c in cs}
        cover = {B: sum(1 for c in cs if spec_attrs(c["stages"])["n_lossy"] >= 2 and ratios[c["spec"]] <= BUDGET_TOL * B)
                 for B in BUDGETS}
        check(f"combos_cover_budgets[{m}]", all(v >= 3 for v in cover.values()), combos_per_budget=cover)
    check("combos_groups_have_why", all(g.get("why") and g.get("role") for g in d["groups"]))
    # 2. 語法：不合法的組合要擋掉
    bad = ["merge:2+drop:KEYDIFF:0.5", "lr:k24+drop:KEYDIFF:0.5", "q:int4_kivi+drop:KEYDIFF:0.5", "merge:2+lr:k16",
           "lr:k16+xl:k8:v8", "mc+q:int4_kivi", "share+drop:KEYDIFF:0.5", "lr:k24:v24+q:mxfp4", "lr:k24:v24+q:fp8k0_token",
           "lr:k24+q:kv-fp8_head-int4_kivi", "lr:k22:v22", "lr:k0:v0", "lr:k132", "drop:FOO:0.5", "drop:KEYDIFF:1.2",
           "drop:KEYDIFF:0.001", "merge:3", "q:int3_kivi", "xl:k8:v8:w5", "q:int4_kivi+q:int8_token", "xl:k8+q:vllm_fp8",
           "drop:KEYDIFF:0.5+drop:SNAP:0.25", "q:kv-int4_kivi", "lr:k8:x3"]
    for s in bad:
        try:
            validate(parse_spec(s), MODEL_INFO["llama"], 63)
            check(f"reject[{s}]", False)
        except SpecError as ex:
            check(f"reject[{s}]", True, why=str(ex)[:80])
    good = ["none", "xl:k11:v17+drop:KEYDIFF:0.5+q:int4_kivi", "drop:KEYDIFF:0.5+merge:4+q:fp8k0_token", "lr:k24+q:kv-bf16-int4_kivi",
            "lr:k0:v24", "q:vllm_fp8", "drop:QRY:0.5+q:vllm_fp8", "mc:t0.5:g0.1"]
    for s in good:
        try:
            validate(parse_spec(s), MODEL_INFO["qwen"], 63)
            check(f"accept[{s}]", True)
        except SpecError as ex:
            check(f"accept[{s}]", False, why=str(ex))
    check("canonical_forms", canonical(parse_spec("drop:keydiff:0.50+lr:k24:v0+q:int4_kivi")) == "drop:KEYDIFF:0.5+lr:k24+q:int4_kivi"
          and canonical(parse_spec("xl:k11:v17:w4")) == "xl:k11:v17" and canonical(parse_spec("mc:t0.6:g0.05")) == "mc")
    check("spec_attrs", spec_attrs(parse_spec("drop:QRY:0.5+q:int4_kivi")) ==
          dict(family="combo", families="drop+quant", n_lossy=2, deployable=0)
          and spec_attrs(parse_spec("lr:k24"))["family"] == "lowrank")
    # 3. 位元組公式 vs 第二階段實測的 parts 檔（真實 block 的 Enc.nbytes；規則 6：用另一條路徑的資料核對）
    for m, shape in (("llama", (32, 2, 8, 512, 128)), ("qwen", (28, 2, 4, 512, 128))):
        pj = os.path.join(REPO, "results", "m10_sweep_mi300x", "parts", f"parts_{m}.json")
        if not os.path.exists(pj):
            check(f"bytes_vs_parts_json[{m}]", False, why="NOT_FOUND " + pj)
            continue
        meas = {k.split("|")[0]: int(v["bytes"]) for k, v in json.load(open(pj))["median_per_block"].items() if k.endswith("|idle")}
        mism = {f: (b, full_fmt_bytes(f, shape)) for f, b in meas.items() if f != "bf16" and full_fmt_bytes(f, shape) != b}
        check(f"bytes_vs_parts_json[{m}]", not mism and len(meas) >= 13, n=len(meas), mismatch=mism)
    # 4. 同一個 KIVI 定義：係數用的量化器（K／V 角色）＝m10_codec int4_kivi 的逐行照抄
    x = np_bf16(rng.normal(size=(2, 2, 3, 64, 64)).astype(np.float32) * 3 + np.where(rng.random((1, 1, 1, 1, 64)) < 0.1, 50, 0))
    ym, nbm = np_m10_int4_kivi(x)
    yk, nbk = np_half_enc(x[:, 0], "kivi4", "k")
    yv, nbv = np_half_enc(x[:, 1], "kivi4", "v")
    check("kivi_coef_quantizer_eq_m10_int4_kivi", np.array_equal(ym[:, 0], yk) and np.array_equal(ym[:, 1], yv) and nbm == nbk + nbv
          == full_fmt_bytes("int4_kivi", x.shape), nbytes=nbm)
    for f in ("int8_token", "int4_token", "kivi4", "kivi2", "bf16"):
        for role in ("k", "v"):
            t = rng.normal(size=(2, 3, 64, 64)).astype(np.float32)
            _, nb = np_half_enc(t, f, role)
            check(f"half_bytes[{f},{role}]", nb == half_bytes(f, t.shape, role), np_bytes=nb, formula=half_bytes(f, t.shape, role))
    # 5. vllm_fp8 的 fnuz 格點
    v = np.array([0, 1e-4, 2 ** -10, 1.5 * 2 ** -10, 0.0078125, 1.0, 1.0625, 1.1, 15.9, 239.0, 240.0, 250.0, 1000.0, -1000.0,
                  -0.3, 232.0, 248.0], dtype=np.float32)
    got = np_fnuz(v).tolist()
    want = [0.0, 0.0, 2 ** -10, 2 * 2 ** -10, 0.0078125, 1.0, 1.0, 1.125, 16.0, 240.0, 240.0, 240.0, 240.0, -240.0, -0.3125, 224.0, 240.0]
    check("fnuz_grid", got == want, got=got, want=want)
    # 6. RoPE：逆轉＋轉回在 float64 精確；配對錯的不行
    D, T0 = 32, 96
    inv = 1.0 / (10000 ** (np.arange(0, D, 2) / D))
    ang = np.arange(T0)[:, None] * inv[None, :]
    c, s = np_bf16(np.cos(ang)).astype(np.float64), np_bf16(np.sin(ang)).astype(np.float64)
    xx = rng.normal(size=(2, T0, D))
    check("rope_inverse_exact", np.abs(np_rope_inv(np_rope_fwd(xx, c, s), c, s) - xx).max() < 1e-12)
    # 7. pre-RoPE 低秩：pre-RoPE PCA 秩 r 精確重建 RoPE 後的 K；直接在 RoPE 後做同秩 PCA 不行（機制）
    L_, H_, r_ = 2, 2, 4
    basis = np.linalg.qr(rng.normal(size=(D, D)))[0]
    def kpre(Tn, seed):
        g = np.random.default_rng(seed)
        return (g.normal(size=(L_, H_, Tn, r_)) * np.array([5, 3, 2, 1.0])) @ basis[:, :r_].T + 7 * basis[:, -1]
    cal_k = kpre(T0, 1)
    def fit(x):
        mu = x.mean(axis=2)
        U = np.zeros((L_, H_, D, D))
        for l in range(L_):
            for h in range(H_):
                ev, vec = np.linalg.eigh(np.cov((x[l, h] - mu[l, h]).T, bias=True))
                U[l, h] = vec[:, ::-1]
        return mu, U
    mu, U = fit(cal_k)
    test = kpre(T0, 2)
    post = np_rope_fwd(test, c, s)
    pre = np_rope_inv(post, c, s)
    rec = np_rope_fwd((pre - mu[:, :, None]) @ U[..., :r_] @ U[..., :r_].transpose(0, 1, 3, 2) + mu[:, :, None], c, s)
    mu2, U2 = fit(np_rope_fwd(cal_k, c, s))
    rec2 = (post - mu2[:, :, None]) @ U2[..., :r_] @ U2[..., :r_].transpose(0, 1, 3, 2) + mu2[:, :, None]
    e1 = np.linalg.norm(rec - post) / np.linalg.norm(post)
    e2 = np.linalg.norm(rec2 - post) / np.linalg.norm(post)
    check("lowrank_pre_rope_exact_post_rope_not", e1 < 1e-9 and e2 > 0.05, err_pre=e1, err_post=e2)
    # 8. merge：AsymKV 等價式（k* 共用、value 原樣 ＝ 存組平均展開 ＝ 存 1 個 key＋value 總和＋計數）
    Kx, Vx, qx = rng.normal(size=(64, D)), rng.normal(size=(64, D)), rng.normal(size=(3, D))
    for g in (2, 4, 8):
        Ks = Kx.reshape(-1, g, D).mean(1)
        o1 = np_attn_out(qx, np.repeat(Ks, g, 0), Vx)
        o2 = np_attn_out(qx, np.repeat(Ks, g, 0), np.repeat(Vx.reshape(-1, g, D).mean(1), g, 0))
        sc = qx @ Ks.T / math.sqrt(D)
        ex_ = np.exp(sc - sc.max(-1, keepdims=True))                          # 壓縮形式：每組 1 個 key、value 總和、計數 g
        o3 = (ex_ @ Vx.reshape(-1, g, D).sum(1)) / (ex_ * g).sum(-1, keepdims=True)
        check(f"asymkv_equivalence_g{g}", np.abs(o1 - o2).max() < 1e-12 and np.abs(o1 - o3).max() < 1e-12)
    # 9. 小模型上的整條 numpy pipeline：位元組＝預測；形狀；有限值；xl 的 SVD 誤差＝Eckart–Young
    info_s = dict(L=4, H=2, Hq=4, D=32)
    Ch, nreg = 64, 4
    Tr = Ch * nreg
    X = np_bf16(rng.normal(size=(4, 2, 2, Tr, 32)).astype(np.float32))
    X[:, 0] += 4.0
    X = np_bf16(X)
    posn = np.arange(Ch, Ch + Tr)
    angf = np.arange(Ch + Tr)[:, None] * inv[None, :]
    cf, sf = np_bf16(np.cos(angf)).astype(np.float64), np_bf16(np.sin(angf)).astype(np.float64)
    calx = np.concatenate([np_rope_inv(X[:, 0].astype(np.float64), cf[posn], sf[posn])[:, None], X[:, 1:2].astype(np.float64)], 1)
    cal = {}
    for hi, h in enumerate("kv"):
        z = calx[:, hi]
        mu_ = z.mean(axis=2)
        Uh = np.zeros((4, 2, 32, 32))
        Uj = np.zeros((4, 64, 64))
        for l in range(4):
            for hh in range(2):
                Uh[l, hh] = np.linalg.eigh(np.cov((z[l, hh] - mu_[l, hh]).T, bias=True))[1][:, ::-1]
            zj = z[l].transpose(1, 0, 2).reshape(Tr, 64)
            Uj[l] = np.linalg.eigh(np.cov((zj - zj.mean(0)).T, bias=True))[1][:, ::-1]
        cal[f"{h}_mean"], cal[f"{h}_U"], cal[f"{h}_Uj"] = mu_, Uh, Uj
    ncx = NCtx(cf, sf, calib=cal, orders={"KEYDIFF": [2, 0, 3, 1]}, deep=[2, 3], chunk=Ch)
    specs = ["none", "drop:KEYDIFF:0.5", "merge:4", "merge:2+q:int4_kivi", "drop:KEYDIFF:0.25+merge:2+q:kv-int4_kivi-int2_kivi",
             "q:int8_token", "q:vllm_fp8", "lr:k8:v8", "lr:k8:v8:joint", "lr:k8:v8+q:int4_kivi", "lr:k8+q:kv-bf16-int4_kivi",
             "lr:k4:v12+q:int8_token", "drop:KEYDIFF:0.5+lr:k8+q:int4_kivi", "xl:k2:v3", "xl:k2:v3+q:int4_kivi",
             "drop:KEYDIFF:0.5+xl:k2:v3", "xl:k2:v3+drop:KEYDIFF:0.5+q:kv-int4_kivi-int2_kivi", "mc", "share", "lr:k32:v32"]
    for sp in specs:
        stg = validate(parse_spec(sp), info_s, nreg)
        dd, idx, by, ex = np_pipeline(stg, X, posn, ncx, info_s)
        pb = predict_bytes(stg, info_s, nreg, Ch)
        Tk = Tr if idx is None else len(idx)
        check(f"np_pipeline_bytes_shape[{sp}]", by == pb and dd.shape == (4, 2, 2, Tk, 32) and np.isfinite(dd).all(), bytes=by)
    dd, _, _, _ = np_pipeline(validate(parse_spec("lr:k32:v32"), info_s, nreg), X, posn, ncx, info_s)
    check("lr_full_rank_np_exact_to_bf16", np.abs(dd - X).max() <= np.abs(X).max() * 2 ** -7, maxdiff=float(np.abs(dd - X).max()))
    _, _, _, ex = np_pipeline(validate(parse_spec("xl:k2:v3"), info_s, nreg), X, posn, ncx, info_s)
    check("xl_svd_err_recorded", all(0 < e < 1 for e in ex["k.svd_err"] + ex["v.svd_err"]), svd_err=ex)
    d1, i1, _, _ = np_pipeline(validate(parse_spec("xl:k2:v3+drop:KEYDIFF:0.5"), info_s, nreg), X, posn, ncx, info_s)
    d2, i2, _, _ = np_pipeline(validate(parse_spec("drop:KEYDIFF:0.5+xl:k2:v3"), info_s, nreg), X, posn, ncx, info_s)
    check("xl_order_control_differs", np.array_equal(i1, i2) and not np.array_equal(d1, d2))
    dm = np_mc(X, [2, 3], 0.6, 0.05)
    check("mc_untouched_shallow_and_kept_tokens", np.array_equal(dm[:2], X[:2]) and np.isfinite(dm).all())
    same = np.stack([X[2], X[2]])
    Xs2 = X.copy()
    Xs2[2:4] = same
    dms = np_mc(Xs2, [2, 3], 0.6, 0.0)
    check("mc_identical_layers_reconstruct", np.abs(dms - Xs2).max() <= np.abs(Xs2).max() * 2 ** -6)
    # 10. 統計
    check("spearman", abs(spearman([1, 2, 3, 4, 5], [5, 6, 7, 8, 7]) - 0.8207826816681233) < 1e-12 and
          spearman([1, 2, 3], [3, 2, 1]) == -1.0)
    check("auroc", auroc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0 and auroc([0.5, 0.5], [1, 0]) == 0.5)
    hb = holm({"a": 0.001, "b": 0.02, "c": 0.03})
    check("holm", hb["a"]["reject"] and not hb["b"]["reject"] and not hb["c"]["reject"])
    check("pareto_front", pareto_front([(0.5, 1.0, "a"), (0.25, 0.5, "b"), (0.25, 0.9, "c"), (0.1, -1.0, "d"), (0.3, 0.2, "e")])
          == ["d", "c", "a"])
    # 11. 判定與挑選（合成的配置表）
    def mk(spec, ratio, dlp, dfo=0.0, roles=("single",)):
        st_ = parse_spec(spec)
        at = spec_attrs(st_)
        return dict(spec=spec, ratio=ratio, d_gold_lp=dict(mean=dlp, lo=dlp - 0.1, hi=dlp + 0.1), d_first_ok=dict(mean=dfo, lo=dfo, hi=dfo),
                    roles=list(roles), n=96, kl_median=0.0, first_ok=0.7 + dfo, first_ok_e0=0.7, **at)
    tab_nogo = {c["spec"]: c for c in [
        mk("q:int4_kivi", 0.3125, -0.01), mk("q:kv-int4_kivi-int2_kivi", 0.25, -0.05), mk("drop:KEYDIFF:0.875", 0.127, -1.0),
        mk("drop:KEYDIFF:0.9375", 0.0635, -2.0, -0.1), mk("lr:k8:v8", 0.0625, -1.5, -0.1),
        mk("drop:KEYDIFF:0.5+q:kv-int4_kivi-int2_kivi", 0.123, -1.2, roles=("combo",)),
        mk("drop:KEYDIFF:0.75+q:kv-int4_kivi-int2_kivi", 0.063, -2.5, -0.2, roles=("combo",)),
        mk("drop:KEYDIFF:0.25+q:int4_kivi", 0.233, -0.2, roles=("combo", "naive")), mk("drop:RND0:0.875", 0.127, -1.1, roles=("control",))]}
    j = judge_x1({"llama": tab_nogo, "qwen": tab_nogo}, {"llama": 96, "qwen": 96})
    check("judge_x1_stop_when_no_synergy", j["S1b_stop"] and not j["go"], budgets=j["models"]["llama"]["budgets"])
    tab_go = dict(tab_nogo)
    tab_go["drop:KEYDIFF:0.5+q:kv-int4_kivi-int2_kivi"] = mk("drop:KEYDIFF:0.5+q:kv-int4_kivi-int2_kivi", 0.123, -0.3, roles=("combo",))
    j = judge_x1({"llama": tab_go, "qwen": tab_nogo}, {"llama": 96, "qwen": 96})
    check("judge_x1_go_when_one_budget_wins", not j["S1b_stop"] and j["go"])
    j = judge_x1({"llama": tab_go, "qwen": tab_go}, {"llama": 96, "qwen": 20})
    check("judge_x1_incomplete", not j["complete"] and not j["go"])
    sel = select_x2(tab_go, "llama")
    sp = {e["spec"]: e for e in sel}
    check("select_x2_anchors_and_primary", "none" in sp and "q:int4_kivi" in sp and
          "primary" in sp["drop:KEYDIFF:0.5+q:kv-int4_kivi-int2_kivi"]["roles"] and
          "best_single" in sp["drop:KEYDIFF:0.875"]["roles"] and "random_drop" in sp["drop:RND0:0.875"]["roles"],
          n=len(sel))
    cands = [mk(f"drop:KEYDIFF:0.5+q:{f}", 0.1, -0.1 * i, roles=("combo",)) for i, f in enumerate(["int4_kivi", "int8_token", "int2_kivi"])]
    cands.append(mk("lr:k16:v16+q:int4_kivi", 0.04, -0.5, roles=("combo",)))
    check("front_pick_diversity", [c["spec"] for c in front_pick(cands)] ==
          ["drop:KEYDIFF:0.5+q:int4_kivi", "drop:KEYDIFF:0.5+q:int8_token", "lr:k16:v16+q:int4_kivi"])
    nx = build_naive_x1({c["spec"]: c for c in [mk("drop:KEYDIFF:0.25", 0.746, -0.1), mk("drop:KEYDIFF:0.5", 0.49, -0.3, -0.03),
                                                  mk("lr:k40:v40", 0.3125, -0.1), mk("q:int2_kivi", 0.1875, -0.2), mk("q:mxfp4", 0.27, -0.1)]},
                        "llama")
    check("naive_x1_stack", nx["spec"] == "drop:KEYDIFF:0.25+lr:k40:v40+q:int2_kivi", naive=nx)

    def ga(spec, ratio, acc, roles, budgets=(), lo=None):
        st_ = parse_spec(spec)
        per = {f"i{k}": acc for k in range(10)}
        return dict(spec=spec, ratio=ratio, roles=list(roles), budgets=list(budgets), n_lossy=len(st_),
                    deployable=spec_attrs(st_)["deployable"], acc=acc, acc_e0=0.7,
                    diff=dict(mean=acc - 0.7, lo=(acc - 0.7 - 0.03) if lo is None else lo, hi=acc - 0.7 + 0.03, p_le0=0.5), per_item=per)
    A_nogo = {e["spec"]: e for e in [ga("q:int4_kivi", 0.3125, 0.70, ["anchor_int4"]), ga("q:kv-int4_kivi-int2_kivi", 0.25, 0.70, ["best_single"], [0.25], lo=-0.01),
                                     ga("drop:KEYDIFF:0.25+q:int4_kivi", 0.233, 0.66, ["primary"], [0.25])]}
    j2 = judge_x2({"llama": A_nogo, "qwen": A_nogo}, {})
    check("judge_x2_nogo", j2["M1_verdict"] == "NO-GO", r=j2["models"]["llama"]["M1"])
    A_go = dict(A_nogo)
    A_go["drop:KEYDIFF:0.5+q:kv-int4_kivi-int2_kivi"] = ga("drop:KEYDIFF:0.5+q:kv-int4_kivi-int2_kivi", 0.123, 0.70, ["primary"], [0.125], lo=-0.01)
    j2 = judge_x2({"llama": A_go, "qwen": A_nogo}, {})
    check("judge_x2_go", j2["M1_verdict"] == "GO", r=j2["models"]["llama"]["M1"])
    A_ask = dict(A_nogo)
    A_ask["drop:KEYDIFF:0.1+q:kv-int4_kivi-int2_kivi"] = ga("drop:KEYDIFF:0.1+q:kv-int4_kivi-int2_kivi", 0.225, 0.70, ["front"], [0.25], lo=-0.01)
    j2 = judge_x2({"llama": A_ask, "qwen": A_nogo}, {})
    check("judge_x2_ask", j2["M1_verdict"] == "ASK", r=j2["models"]["llama"]["M1"])
    # 12. 無損：查表與 NOT_MEASURED
    check("lossless_key", lossless_key("drop:KEYDIFF:0.5+q:int4_kivi") == "int4_kivi" and lossless_key("lr:k24") is None
          and lossless_key("none") == "bf16" and lossless_key("q:int2_kivi") is None)
    check("s1_multipliers_missing", s1_multipliers("/nonexistent/summary.json", "llama") == {})
    # 13. X3 介面：對應表的名字都在 X3 的目錄裡；組合清單裡每個 X3 段都對得到；report → gen-report → X3 的 load_x2 端到端（合成列）
    check("x3_map_names_in_catalog", set(X3_MAP.values()) | {"fp8"} <= set(X3_SEGMENTS))
    for m in MODEL_INFO:
        specs = {c["spec"] for c in load_combos(COMBOS, m)}
        mapped = {x3_segment(s, m)[0] for s in specs} - {None}
        check(f"x3_segments_covered[{m}]", mapped == set(X3_SEGMENTS) - {"droptok_k50", "droptok_k25", "droptok_k50_int4"},
              mapped=sorted(mapped))
    check("x3_unmapped_reason", x3_segment("drop:KEYDIFF:0.5+q:int4_kivi", "llama")[0] is None
          and x3_segment("xl:k5:v9", "qwen")[0] is None and x3_segment("q:fp8_head", "qwen")[0] == "fp8"
          and x3_segment("q:fp8_head", "llama")[0] is None
          and x3_segment("xl:k11:v17", "qwen")[0] == "xkv_k11v17"                       # 10/21 加的對應
          and x3_segment("xl:k11:v17:cal", "qwen")[0] is None and "校準" in x3_segment("xl:k11:v17:cal", "qwen")[1])
    # 先呼叫再合併：`ok &= f(check)` 會在 f 跑之前就讀 ok，f 裡面 check 失敗設的 False 會被蓋掉（10/21 抓到，原本的 e2e 也受影響）
    r_e2e = _selftest_report_e2e(check)
    r_1021 = _selftest_1021(check)
    ok = ok and r_e2e and r_1021
    # 14. 沒有 import torch（規則：第二階段在量時間）
    bad_mods = [m for m in ("torch", "vllm", "transformers") if m in sys.modules]
    check("no_torch_imported", not bad_mods, imported=bad_mods)
    n = len(res)
    print(f"{'ALL PASS' if ok else 'SOME FAILED'}（{n} 項）", flush=True)
    return ok


def x3_compatible_copy(src: str, dst: str) -> str:
    """self-test 用：X3（另一個 agent）的 seg_catalog 還沒加 10/21 的 xkv_k* 段時，load_x2 會丟例外（規則 7，照設計）。
    self-test 只驗我們的欄位與名字，所以把 catalog 還沒有的段濾掉、印 INFO（不算 PASS／FAIL）；正式的核對在 38bz（讀不懂 → NEEDS_HUMAN）。"""
    import csv
    import m16_xtime as XT
    cat = set(XT.seg_catalog())
    pending = sorted(set(X3_SEGMENTS) - cat)
    if pending:
        print(f"INFO（不算 PASS／FAIL）：X3 的 seg_catalog 還沒有 {pending}；self-test 的 load_x2 先濾掉這些列，38bz 會正式核對", flush=True)
    rows = list(csv.DictReader(open(src))) if os.path.exists(src) else []
    if not rows:
        return src
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows([r for r in rows if r["x3_segment"] in cat])
    return dst


def _selftest_1021(check) -> bool:
    """10/21 延伸（X2→X3 對應、:cal、X2b 計畫、合併報告的配對、xKV 校準的判定、工作池、舊子命令照舊）。合成數字只驗邏輯。"""
    import csv
    import gzip
    import tempfile
    np = _np()
    ok0 = True
    rng = np.random.default_rng(1021)
    # ---- a. X2→X3 的新對應
    for m in MODEL_INFO:
        good = all(canonical(parse_spec(k)) == k and x3_segment(k, m) == (v, "") and v in X3_SEGMENTS for k, v in X3_MAP_1021.items())
        check(f"x3_map_1021[{m}]", good and len(X3_MAP_1021) == 7, n=len(X3_MAP_1021))
    v1f = os.path.join(OUT_DIR, "x2_feasible.csv")
    v1u = os.path.join(OUT_DIR, "x2_unmapped.csv")
    if os.path.exists(v1f) and os.path.exists(v1u):
        rf, ru = list(csv.DictReader(open(v1f))), list(csv.DictReader(open(v1u)))
        same = all(x3_segment(r["combo"], r["model"])[0] == r["x3_segment"] for r in rf)
        moved = sorted({r["combo"] for r in ru if x3_segment(r["combo"], r["model"])[0] is not None})
        check("x3_map_v1_rows_unchanged_except_1021", same and set(moved) <= set(X3_MAP_1021), moved_to_feasible=moved)
    else:
        print("INFO：第一版的 x2_feasible.csv／x2_unmapped.csv 不在，跳過 x3_map_v1_rows_unchanged_except_1021", flush=True)
    # ---- b. :cal 的語法
    for s in ("xl:k22:v34:cal+q:int4_kivi", "drop:KEYDIFF:0.5+xl:k11:v17:cal", "xl:k11:v17:cal", "xl:k28:v42:cal+q:kv-int4_kivi-int2_kivi"):
        try:
            validate(parse_spec(s), MODEL_INFO["llama"], 63)
            check(f"accept_cal[{s}]", True)
        except SpecError as ex:
            check(f"accept_cal[{s}]", False, why=str(ex))
    for s in ("xl:k11:v17:cal+drop:KEYDIFF:0.5", "lr:k16:cal", "xl:k11:v17:cal+q:vllm_fp8", "xl:k11:v17:cal+q:mxfp4", "mc:cal"):
        try:
            validate(parse_spec(s), MODEL_INFO["llama"], 63)
            check(f"reject_cal[{s}]", False)
        except SpecError as ex:
            check(f"reject_cal[{s}]", True, why=str(ex)[:80])
    check("cal_canonical", canonical(parse_spec("xl:k22:v34:w4:cal+q:int4_kivi")) == "xl:k22:v34:cal+q:int4_kivi"
          and canonical(parse_spec("xl:k22:v34:cal:w2")) == "xl:k22:v34:w2:cal" and "cal" not in parse_spec("xl:k22:v34")[0]
          and spec_attrs(parse_spec("xl:k22:v34:cal+q:int4_kivi"))["families"] == "quant+xlayer")
    # ---- c. 位元組：校準版＝每題版減掉每題基底（基底改列 shared）；舊組合的位元組和 X1 實測（GPU 上 assert 過的實際位元組）逐一相同
    for m in MODEL_INFO:
        info = MODEL_INFO[m]
        okb, det = True, {}
        for rk, rv, q in ((11, 17, ""), (22, 34, "+q:int4_kivi"), (28, 42, "+q:int4_kivi"), (22, 34, "+q:int8_token")):
            sc, si = f"xl:k{rk}:v{rv}:cal{q}", f"xl:k{rk}:v{rv}{q}"
            bc, bi = predict_bytes(parse_spec(sc), info, 63), predict_bytes(parse_spec(si), info, 63)
            basis = bi["k.basis"] + bi["v.basis"]
            okb &= ("k.basis" not in bc and "v.basis" not in bc and bc["shared.k.basis"] + bc["shared.v.basis"] == basis
                    and sum(bc.values()) == sum(bi.values()))
            cb = info["L"] * 2 * info["H"] * CHUNK * info["D"] * 2 * 63
            rc, ri = predict_ratio(sc, m), predict_ratio(si, m)
            okb &= abs((ri - rc) - basis / cb) < 2e-6
            det[sc] = dict(ratio_cal=rc, ratio_per_item=ri, basis_MB=round(basis / 1e6, 1))
        check(f"cal_bytes[{m}]", okb, **det)
        cf = os.path.join(OUT_DIR, f"configs_{m}.csv")
        if os.path.exists(cf):
            rows = list(csv.DictReader(open(cf)))
            mism = [(r["spec"], float(r["ratio"]), predict_ratio(r["spec"], m)) for r in rows
                    if abs(float(r["ratio"]) - predict_ratio(r["spec"], m)) > 1e-6]
            check(f"bytes_unchanged_vs_x1_measured[{m}]", not mism and len(rows) >= 120, n=len(rows), mismatch=mism[:5])
        else:
            print(f"INFO：{cf} 不在，跳過 bytes_unchanged_vs_x1_measured[{m}]", flush=True)
    # ---- d. numpy：校準基底＝同一份資料的前 r 個特徵向量時，誤差＝Eckart–Young（精確 SVD）；別的資料的基底只會更差；位元組＝預測
    info_s = dict(L=4, H=2, Hq=4, D=32)
    Ch, nreg, D_ = 64, 4, 32
    Tr = Ch * nreg
    inv = 1.0 / (10000 ** (np.arange(0, D_, 2) / D_))
    angf = np.arange(Ch + Tr)[:, None] * inv[None, :]
    cf_, sf_ = np_bf16(np.cos(angf)).astype(np.float64), np_bf16(np.sin(angf)).astype(np.float64)
    posn = np.arange(Ch, Ch + Tr)

    def mkX(seed):
        g = np.random.default_rng(seed)
        low = g.normal(size=(4, 2, 2, Tr, 6)) @ g.normal(size=(6, D_)) + 0.3 * g.normal(size=(4, 2, 2, Tr, D_))
        return np_bf16(low.astype(np.float32))

    def windowZ(X, hi):
        src = X[:, hi]
        return np.concatenate([(np_rope_inv(src[l].astype(np.float64), cf_[posn], sf_[posn]) if hi == 0 else src[l].astype(np.float64))
                               .transpose(1, 0, 2).reshape(Tr, 2 * D_) for l in range(4)], axis=1)
    Xa, Xb = mkX(1), mkX(2)
    R = 64
    cal_same = {f"{h}_U": np_xl_calib([windowZ(Xa, hi)], R)[0][None] for hi, h in enumerate("kv")}
    cal_other = {f"{h}_U": np_xl_calib([windowZ(Xb, hi), windowZ(mkX(3), hi)], R)[0][None] for hi, h in enumerate("kv")}
    ncs = NCtx(cf_, sf_, orders={"KEYDIFF": [2, 0, 3, 1]}, deep=[2, 3], chunk=Ch, xlcal=cal_same)
    nco = NCtx(cf_, sf_, orders={"KEYDIFF": [2, 0, 3, 1]}, deep=[2, 3], chunk=Ch, xlcal=cal_other)
    _, _, _, ex_s = np_pipeline(validate(parse_spec("xl:k2:v3"), info_s, nreg), Xa, posn, ncs, info_s)
    _, _, _, ex_c = np_pipeline(validate(parse_spec("xl:k2:v3:cal"), info_s, nreg), Xa, posn, ncs, info_s)
    _, _, _, ex_o = np_pipeline(validate(parse_spec("xl:k2:v3:cal"), info_s, nreg), Xa, posn, nco, info_s)
    eq = all(abs(c - s) <= 1e-6 * max(s, 1e-12) for h in "kv" for c, s in zip(ex_c[f"{h}.cal_err"], ex_s[f"{h}.svd_err"]))
    ge = all(o >= s * (1 - 1e-9) for h in "kv" for o, s in zip(ex_o[f"{h}.cal_err"], ex_s[f"{h}.svd_err"]))
    check("xlcal_np_eckart_young", eq and ge, same_data=ex_c, other_data=ex_o, exact_svd=ex_s)
    for sp in ("xl:k2:v3:cal", "xl:k2:v3:cal+q:int4_kivi", "drop:KEYDIFF:0.5+xl:k2:v3:cal+q:kv-int4_kivi-int2_kivi", "xl:k2:v3:cal+q:int8_token"):
        stg = validate(parse_spec(sp), info_s, nreg)
        dd, idx, by, _ = np_pipeline(stg, Xa, posn, nco, info_s)
        Tk = Tr if idx is None else len(idx)
        check(f"xlcal_np_bytes_shape[{sp}]", by == predict_bytes(stg, info_s, nreg, Ch) and dd.shape == (4, 2, 2, Tk, D_)
              and np.isfinite(dd).all() and "k.basis" not in by, bytes=by)
    Zs = [rng.normal(size=(50, 12)), rng.normal(size=(30, 12))]
    U_, ev_ = np_xl_calib(Zs, 5)
    _, S_, Vt_ = np.linalg.svd(np.concatenate(Zs), full_matrices=False)
    check("xlcal_np_calib_eq_stacked_svd", np.allclose(np.abs(U_.T @ Vt_[:5].T), np.eye(5), atol=1e-8) and np.allclose(ev_[:5], S_[:5] ** 2))
    # ---- e. 計畫檔、題目不重疊、CPU 版出題
    check("rotations_from_source", rotations_from_source() == [("front", "back"), ("mid", "front"), ("back", "mid")])
    T4 = ["T2", "T3", "T4", "T5"]
    keys = {name: {(t, s) for t, _, _, s in plan_interleaved_cpu(T4, n_, s0)}
            for name, n_, s0 in (("x1", 24, X1_SEED0), ("calib", 2, CALIB_SEED0), ("x2", 24, X2_SEED0), ("x2b", X2B_N, X2B_SEED0))}
    tfk = {(t, s) for t, _, _, s in plan_interleaved_cpu(T4, XLCAL_TF_N, X1_SEED0)}
    gk = {(t, s) for t, _, _, s in plan_interleaved_cpu(T4, XLCAL_GEN_N, X2_SEED0)}
    disj = all(not (keys[a] & keys[b]) for a in keys for b in keys if a < b)
    check("seeds_disjoint", disj and tfk <= keys["x1"] and gk == keys["x2"] and len(keys["x2b"]) == 4 * X2B_N
          and not (keys["calib"] & (tfk | gk | keys["x2b"])), sizes={k: len(v) for k, v in keys.items()})
    x2raw = sorted(glob.glob(os.path.join(DATA, "x2", "x2_*_32K_*_s*.csv.gz")))
    if x2raw:
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            ids = {r["item_id"] for r in read_rows(x2raw[:4]) if r["spec"] == "E0"}
        want = {item_id_of(*p) for p in plan_interleaved_cpu(T4, 24, X2_SEED0)}
        check("plan_cpu_item_ids_eq_x2_raw", ids and ids <= want, n_raw=len(ids), n_plan=len(want))
    else:
        print("INFO：X2 的逐列檔不在，跳過 plan_cpu_item_ids_eq_x2_raw", flush=True)
    v1s = os.path.join(OUT_DIR, "x2_selection.json")
    v1sel = json.load(open(v1s)) if os.path.exists(v1s) else {"go": True, "models": {}}
    pb = build_plan("x2b", v1sel)
    px = build_plan("xlcal", v1sel)
    cfg = {m: {e["spec"]: e for e in pb["models"][m]["configs"]} for m in MODEL_INFO}
    okp = len(cfg["llama"]) == 11 and len(cfg["qwen"]) == 12 and all(list(cfg[m]) == plan_specs("x2b", m) for m in MODEL_INFO)
    okp &= all(s in cfg[m] for m in MODEL_INFO for s in ("q:int2_kivi", "xl:k11:v17", "xl:k16:v16", "xl:k22:v34+q:int4_kivi",
                                                         "xl:k28:v42+q:int4_kivi", "xl:k22:v34+q:int8_token", "q:kv-int4_kivi-int2_kivi"))
    okp &= "q:vllm_fp8" in cfg["llama"] and "q:int8_token" in cfg["qwen"] and "q:vllm_fp8" not in cfg["qwen"]
    if v1sel.get("models"):
        okp &= ("primary" in cfg["llama"]["xl:k22:v34+q:int8_token"]["roles"] and cfg["qwen"]["xl:k22:v34+q:int8_token"]["roles"] == ["x2b"]
                and "deploy" in cfg["llama"]["q:vllm_fp8"]["roles"] and "deploy" in cfg["qwen"]["q:int8_token"]["roles"])
    okp &= all(len(px["models"][m]["configs"]) == 6 for m in MODEL_INFO) and px["seed0"] == X2_SEED0 and px["n"] == XLCAL_GEN_N
    okp &= pb["seed0"] == X2B_SEED0 and pb["n"] == X2B_N and all(e["tiers"] == ["core"] for m in MODEL_INFO for e in pb["models"][m]["configs"])
    check("plan_x2b_xlcal", okp, x2b={m: list(cfg[m]) for m in cfg})
    lst = plan_specs("x2b", "llama")
    adj = all(abs(lst.index(a) - lst.index(b)) == 1 for a, b in (("xl:k11:v17", "xl:k11:v17+q:int4_kivi"),
                                                                ("xl:k22:v34+q:int4_kivi", "xl:k22:v34+q:int8_token")))
    xp = xl_cache_plan([dict(stages=parse_spec(s)) for s in lst])
    xq = xl_cache_plan([dict(stages=parse_spec(s)) for s in plan_specs("x2b", "qwen")])
    check("xl_cache_plan_x2b", adj and xp["uses"] == {(0, 11, 4): 2, (1, 17, 4): 2, (0, 16, 4): 1, (1, 16, 4): 1, (0, 22, 4): 2,
                                                     (1, 34, 4): 2, (0, 28, 4): 1, (1, 42, 4): 1}
          and xq["uses"][(0, 22, 4)] == 3 and xq["uses"][(1, 34, 4)] == 3, uses={str(k): v for k, v in xp["uses"].items()})
    with tempfile.TemporaryDirectory() as d:
        po = os.path.join(d, "x2b_selection.json")
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            cmd_plan(SimpleNamespace(which="x2b", selection=v1s, out=po))
            cmd_plan(SimpleNamespace(which="x2b", selection=v1s, out=po))       # 內容相同 → 不重寫
            j = json.load(open(po))
            j["models"]["llama"]["configs"] = j["models"]["llama"]["configs"][:-1]
            json.dump(j, open(po, "w"))
            try:
                cmd_plan(SimpleNamespace(which="x2b", selection=v1s, out=po))
                refused = False
            except SystemExit:
                refused = True
        check("plan_no_silent_overwrite", refused)
        # ---- f. pending：shard 的切法和 cmd_gen 相同（k % 份數）、做完的題（列數 ≥ n_rows_item）不算
        out = os.path.join(d, "x2b_llama_32K_s1.csv.gz")
        plan = plan_interleaved_cpu(T4, 8, X2B_SEED0)
        mine = [item_id_of(*p) for k, p in enumerate(plan) if k % 3 == 1]
        with gzip.open(out, "wt", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["run_id", "item_id", "spec", "n_rows_item"])
            w.writeheader()
            for s in ("E0", "a", "b"):
                w.writerow(dict(run_id="SYN", item_id=mine[0], spec=s, n_rows_item=3))
            w.writerow(dict(run_id="SYN", item_id=mine[1], spec="E0", n_rows_item=3))     # 寫到一半 → 沒做完
        pend = shard_pending(T4, 8, X2B_SEED0, "1/3", out)
        parts = [set(shard_pending(T4, 8, X2B_SEED0, f"{i}/3", os.path.join(d, "none_s0.csv"))) for i in range(3)]
        check("pending_shards", pend == mine[1:] and set().union(*parts) == {item_id_of(*p) for p in plan}
              and sum(len(p) for p in parts) == len(plan), n_pending=len(pend))
        # ---- g. 合併報告（X2＋X2b）的配對邏輯
        ok0 &= _selftest_x2b_report(check, d, v1s if os.path.exists(v1s) else None)
        ok0 &= _selftest_xlcal_report(check, d)
        ok0 &= _selftest_pool(check, d)
    # ---- h. 37d／37e／37f／37zz、02 的指令照樣解析得了（新旗子都有預設＝舊行為）
    ap = build_parser()
    sel, cal = os.path.join(OUT_DIR, "x2_selection.json"), "/mlsteam/data/tiara/stage3/xstack/calib_llama.pt"
    olds = [["gen", "--model", "llama", "--ctx", "32K", "--tier", "ext", "--n", "24", "--selection", sel, "--calib", cal, "--shard", "0/3",
             "--budget-min", "100", "--out", "/x/x2_llama_32K_ext_s0.csv.gz"],
            ["gen", "--model", "qwen", "--ctx", "128K", "--tier", "128k", "--selection", sel, "--calib", cal, "--shard", "1/2",
             "--budget-min", "100", "--out", "/x/x2_qwen_128K_128k_s1.csv.gz"],
            ["gen-report", "--raw", "/x/x2_*_s*.csv.gz", "--selection", sel, "--out-dir", OUT_DIR],
            ["gpu-selftest", "--model", "llama", "--calib", cal, "--out", "/x/g.json"],
            ["screen", "--model", "llama", "--smoke", "--calib", cal, "--out", "/x/x1_smoke_llama_s0.csv"]]
    pa = [ap.parse_args(x) for x in olds]
    check("old_commands_parse_with_old_defaults", pa[0].calib_xl is None and not pa[0].xl_cache and pa[1].ctx == "128K"
          and pa[3].new_checks == "none" and pa[4].calib_xl is None and all(COMMANDS[p.cmd] in globals() for p in pa))
    return ok0


def _selftest_x2b_report(check, d, v1s) -> bool:
    """合成：X2 96 題（第一版挑的配置，其中 1 題缺一個 C 的配置、Qwen 沒量 xl:k22:v34+q:int8_token）、X2b 192 題（計畫的全部配置）→ x2b-report。
    核對 C、被排除的題、D 的配對差＝直接算的、A＝第一版 gen-report 的判定、v2 的 x2_feasible 的 source、不准寫到第一版的目錄。"""
    import csv
    import gzip
    import random
    rnd = random.Random(7)
    v1sel = json.load(open(v1s)) if v1s else None
    if v1sel is None:
        print("INFO：第一版 x2_selection.json 不在，跳過合併報告的 self-test", flush=True)
        return True
    plan = build_plan("x2b", v1sel)
    pp = os.path.join(d, "x2b_plan.json")
    json.dump(plan, open(pp, "w"))
    vs = os.path.join(d, "v1sel.json")
    json.dump(v1sel, open(vs, "w"))
    acc_true = {}

    def p_ok(spec):
        return acc_true.setdefault(spec, 0.3 + 0.6 * rnd.random())
    expect = {}
    for tag, n_items, s0, src in (("x2", 96, X2_SEED0, "v1"), ("x2b", 192, X2B_SEED0, "plan")):
        path = os.path.join(d, f"{tag}_syn_s0.csv.gz")
        with gzip.open(path, "wt", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS_X2)
            w.writeheader()
            for m in MODEL_INFO:
                specs = [e["spec"] for e in (v1sel if src == "v1" else plan)["models"][m]["configs"]]
                if src == "v1" and m == "qwen":
                    specs = [s for s in specs if s != "xl:k22:v34+q:int8_token"]
                for k in range(n_items):
                    iid = f"T2_syn_s{s0 + k}"
                    e0 = int(rnd.random() < 0.75)
                    sp = [s for s in specs if not (src == "v1" and m == "llama" and k == 5 and s == "xl:k11:v17")]
                    rows = [dict(spec="E0", gen_score=e0, ratio_region=1.0)]
                    for s in sp:
                        if s == "xl:k28:v42+q:int4_kivi":
                            sc = e0 - (1 if (k % 10 == 0 and e0) else 0)        # 已知的配對差：每 10 題、E0 對的那題掉 1
                            expect.setdefault((m, tag), []).append((iid, sc - e0))
                        else:
                            sc = int(rnd.random() < p_ok(s))
                        rows.append(dict(spec=s, gen_score=sc, ratio_region=1.0 if s == "none" else predict_ratio(s, m)))
                    for r in rows:
                        w.writerow(dict(r, run_id=f"SYN-{tag}", ts="t", model=m, ctx="32K", item_id=iid, n_rows_item=len(rows),
                                        roles="", budgets="", own_gpu_intruders=0, host_level="QUIET"))
    out = os.path.join(d, "v2rep")
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        cmd_x2b_report(SimpleNamespace(raw_x2=[os.path.join(d, "x2_syn_s*.csv.gz")], raw_x2b=[os.path.join(d, "x2b_syn_s*.csv.gz")],
                                       selection=vs, plan=pp, out_dir=out))
    rep = json.load(open(os.path.join(out, "x2b_report.json")))
    L, Q = rep["models"]["llama"], rep["models"]["qwen"]
    okc = ("xl:k22:v34+q:int8_token" not in Q["C"] and Q["only_x2b"] == ["xl:k22:v34+q:int8_token"]
           and L["C"] == plan_specs("x2b", "llama") and len(Q["C"]) == len(plan_specs("x2b", "qwen")) - 1
           and L["n_items"] == dict(A=96, B=95, C=192, C_all_x2b_configs=192, D=287) and Q["n_items"]["D"] == 288 and L["x2b_complete"])
    dD = rep["judge"]["D"]
    vals = [v for tag in ("x2", "x2b") for iid, v in expect[("llama", tag)] if iid != f"T2_syn_s{X2_SEED0 + 5}"]
    got = L["configs"]["D"]["xl:k28:v42+q:int4_kivi"]["diff"]
    okd = abs(got["mean"] - statistics.mean(vals)) < 1e-12 and got["n"] == len(vals) == 287
    rows2 = [r for r in read_rows([os.path.join(d, "x2_syn_s0.csv.gz")])]
    J1 = judge_x2({m: with_roles(gen_table([r for r in rows2 if r["model"] == m])[0],
                                 {e["spec"]: e for e in v1sel["models"][m]["configs"]}, {}) for m in MODEL_INFO}, {})
    oka = all(rep["judge"]["A"]["models"][m]["M1"] == json.loads(json.dumps(J1["models"][m]["M1"], default=str)) for m in MODEL_INFO)
    fx = list(csv.DictReader(open(os.path.join(out, "x2_feasible.csv"))))
    src = {(r["model"], r["combo"]): r["source"] for r in fx}
    oks = (src.get(("llama", "q:int4_kivi")) == "x2+x2b" and src.get(("qwen", "xl:k22:v34+q:int8_token")) == "x2b"
           and any(v == "x2" for v in src.values()) and all(r["x3_segment"] in X3_SEGMENTS for r in fx))
    try:
        cmd_x2b_report(SimpleNamespace(raw_x2=[], raw_x2b=[], selection=vs, plan=pp, out_dir=OUT_DIR))
        okg = False
    except SystemExit:
        okg = True
    check("x2b_report_pairing", okc and okd and oka and oks and okg and "pool_gaps" in dD and dD["complete"],
          C_qwen=Q["C"], n_llama=L["n_items"], d_mean=(got["mean"], statistics.mean(vals)), A_eq_v1_judge=oka, sources_ok=oks,
          guard_v1_dir=okg, verdicts={k: rep["judge"][k]["M1_verdict"] for k in "ABCD"})
    try:
        fx3 = x3_compatible_copy(os.path.join(out, "x2_feasible.csv"), os.path.join(d, "v2_x3.csv"))
        import m16_xtime as XT
        x3 = XT.load_x2(fx3)
        check("x2b_report_x3_load_x2", x3 is not None, feasible={k: sorted(v) for k, v in x3["feasible"].items()})
    except ImportError as ex:
        check("x2b_report_x3_load_x2", False, why=f"m16_xtime 匯入不了：{ex}")
    return True


def _selftest_xlcal_report(check, d) -> bool:
    """合成：TF（本次＋X1 的列）與生成（6 個配置＋E0、60 題）→ xlcal-report。三對各造成 CAN／MUST／UNDETERMINED → 模型＝MUST；
    另一個模型三對都相同 → CAN；判定規則的邊界直接測。"""
    import csv
    import gzip
    import random
    rnd = random.Random(11)
    pairs = xlcal_pairs()
    check("xlcal_verdict_rules", xlcal_verdict(dict(lo=-0.019, hi=0.01)) == "CAN_PRECOMPUTE" and xlcal_verdict(dict(lo=-0.08, hi=-0.021)) ==
          "MUST_PER_ITEM" and xlcal_verdict(dict(lo=-0.05, hi=0.01)) == "UNDETERMINED" and xlcal_verdict(None) == "NOT_MEASURED"
          and xlcal_aggregate(["CAN_PRECOMPUTE", "MUST_PER_ITEM", "UNDETERMINED"]) == "MUST_PER_ITEM"
          and xlcal_aggregate(["CAN_PRECOMPUTE"] * 3) == "CAN_PRECOMPUTE" and xlcal_aggregate(["CAN_PRECOMPUTE", "UNDETERMINED"]) == "UNDETERMINED")
    gp, tp, xp = (os.path.join(d, f"xl_{t}_s0.csv.gz") for t in ("gen", "tf", "x1"))
    n = 60
    with gzip.open(gp, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS_X2)
        w.writeheader()
        for m in MODEL_INFO:
            for k in range(n):
                iid = f"T2_g_s{X2_SEED0 + k}"
                e0 = int(rnd.random() < 0.8)
                rows = [dict(spec="E0", gen_score=e0, ratio_region=1.0, bytes_region="")]
                for j, (c, ref) in enumerate(pairs):
                    r = int(rnd.random() < 0.8)
                    if m == "qwen" or j == 0:
                        cs = r                                                          # 相同 → CAN
                    elif j == 1:
                        cs = 0 if r and k % 3 else r                                   # 大幅變差 → MUST
                    else:
                        r = 1 if k % 20 == 0 else r                                    # 小差（60 題裡 3 題掉 1）→ UNDETERMINED
                        cs = 0 if k % 20 == 0 else r
                    for s, sc in ((ref, r), (c, cs)):
                        by = bytes_summary(predict_bytes(parse_spec(s), MODEL_INFO[m], 63), MODEL_INFO[m], 63, 64)
                        rows.append(dict(spec=s, gen_score=sc, ratio_region=by["ratio_region"], bytes_region=by["bytes_region"]))
                for r_ in rows:
                    w.writerow(dict(r_, run_id="SYNG", ts="t", model=m, ctx="32K", item_id=iid, n_rows_item=len(rows)))
    for path, specs, tag in ((tp, [c for c, _ in pairs] + ["xl:k11:v17"], "SYNT"), (xp, [r for _, r in pairs], "SYNX1")):
        with gzip.open(path, "wt", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS_X1)
            w.writeheader()
            for m in MODEL_INFO:
                for k in range(12):
                    iid = f"T2_t_s{X1_SEED0 + k}"
                    rows = [dict(spec="E0", range="", gold_lp=-1.0, first_ok=1)]
                    rows += [dict(spec=s, range="full", gold_lp=-1.0 - 0.1 * rnd.random(), first_ok=1, relerr_k=0.1, relerr_v=0.2, attn_err=0.05)
                             for s in specs]
                    for r_ in rows:
                        w.writerow(dict(r_, run_id=tag, ts="t", model=m, item_id=iid, gold_lp_e0=-1.0, first_ok_e0=1, kl=0.01,
                                        n_rows_item=len(rows)))
    out = os.path.join(d, "xlrep")
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        cmd_xlcal_report(SimpleNamespace(raw_tf=[tp], raw_gen=[gp], raw_x1=[xp], raw_x2=None, calib_xl_summary=[], out_dir=out))
    rep = json.load(open(os.path.join(out, "xlcal_report.json")))
    pv = {m: [e["verdict"] for e in rep["models"][m]["gen"]["pairs"].values()] for m in MODEL_INFO}
    bk = [(e["amortized_ratio"]["1"], rep["models"]["llama"]["gen"]["configs"][k.split(" vs ")[1]]["ratio"], e["amortized_ratio"]["64"])
          for k, e in rep["models"]["llama"]["gen"]["pairs"].items()]
    okv = (pv["llama"] == ["CAN_PRECOMPUTE", "MUST_PER_ITEM", "UNDETERMINED"] and pv["qwen"] == ["CAN_PRECOMPUTE"] * 3
           and rep["verdict_by_model"] == {"llama": "MUST_PER_ITEM", "qwen": "CAN_PRECOMPUTE"} and rep["verdict"] == "MUST_PER_ITEM"
           and all(abs(a1 - rr) <= 2e-6 and a64 < a1 for a1, rr, a64 in bk) and rep["models"]["llama"]["tf"]["n_items"] == 12
           and rep["models"]["llama"]["tf"]["e0_bitwise_repro_vs_x1"] == "12/12")
    check("xlcal_report_verdicts", okv, verdicts=pv, amortized_n1_vs_per_item_n64=bk, by_model=rep["verdict_by_model"])
    return True


def _selftest_pool(check, d) -> bool:
    """x_pool.sh（bash）：假的 m7run／oncard。14 個單位（Llama 6 份 44 GiB、Qwen 8 份 28 GiB 交錯）、兩張卡、每卡 ≤3 個行程／≤100 GiB；
    其中 1 個單位的待做檢查回 0（已做完 → 不開行程）。核對：每個 (模型, shard) 恰好被領一次、做完的沒開、其他各開一次、每卡同時的
    行程數與 GiB 沒超過、兩張卡都有領、回 0；另一輪讓 1 個單位失敗 → 回非 0。"""
    import subprocess
    pool_sh = os.path.join(HERE, "m16_queue", "jobs", "x_pool.sh")
    script = r'''
set -u
T=$1; FAIL=$2
XP_CAP_GB=100; XP_MAXP=3; XP_POLL_S=0.05
source "$3"
oncard() { local c=$1; shift; TIARA_GPU=$c "$@"; }
m7run() { local n=$1; shift; echo "S $n $TIARA_GPU $(date +%s%N)" >> $T/ev; "$@"; local r=$?; echo "E $n $TIARA_GPU $(date +%s%N)" >> $T/ev; return $r; }
nh() { echo "NH $*" >> $T/nh; }
P=$T/pool; xp_init $P
for u in $(xp_order llama 6 qwen 8); do m=${u%:*}; i=${u#*:}
  gb=28; [ $m = llama ] && gb=44
  pend=-; [ $u = qwen:3 ] && pend="echo 0"
  cmd="sleep 0.$((RANDOM % 3 + 1))"; [ "$FAIL" = 1 ] && [ $u = llama:2 ] && cmd=false
  xp_add $P $m $gb u-$m-$i "$pend" $cmd
done
xp_run $P 10 0 1 > $T/out.log 2>&1; echo $? > $T/rc
'''
    res = {}
    for fail in (0, 1):
        T = os.path.join(d, f"pool{fail}")
        os.makedirs(T)
        subprocess.run(["bash", "-c", script, "pool", T, str(fail), pool_sh], check=False, timeout=120)
        rc = int(open(os.path.join(T, "rc")).read().strip())
        ev = [ln.split() for ln in open(os.path.join(T, "ev"))] if os.path.exists(os.path.join(T, "ev")) else []
        res[fail] = (T, rc, ev)
    T, rc, ev = res[0]
    P = os.path.join(T, "pool")
    n = int(open(os.path.join(P, "n")).read())
    claims = sorted(os.listdir(os.path.join(P, "claim")))
    started = [e[1] for e in ev if e[0] == "S"]
    names = {open(os.path.join(P, "u", str(k), "name")).read().strip() for k in range(n)}
    gb = {nm: (44 if "llama" in nm else 28) for nm in names}
    peak, ok_cap = {}, True
    cur = {}
    for kind, nm, card, t in sorted(ev, key=lambda e: (int(e[3]), 0 if e[0] == "E" else 1)):
        c = cur.setdefault(card, [0, 0])
        if kind == "S":
            c[0] += 1
            c[1] += gb[nm]
        else:
            c[0] -= 1
            c[1] -= gb[nm]
        ok_cap &= c[0] <= 3 and c[1] <= 100
        peak[card] = max(peak.get(card, (0, 0)), tuple(c))
    cards = {open(os.path.join(P, "claim", k, "card")).read().strip() for k in claims}
    okp = (rc == 0 and n == 14 and len(claims) == 14 and sorted(started) == sorted(names - {"u-qwen-3"}) and len(started) == len(set(started))
           and "u-qwen-3" not in started and ok_cap and cards == {"0", "1"})
    T1, rc1, ev1 = res[1]
    check("pool_each_unit_claimed_once", okp and rc1 != 0, n_units=n, n_claimed=len(claims), n_started=len(started), cards=sorted(cards),
          peak_per_card=peak, rc=rc, rc_with_failure=rc1)
    return True


def _selftest_report_e2e(check) -> bool:
    """合成 X1 逐列（兩個模型、50 題、全部組合）→ report → x2_selection.json → 合成 X2 逐列 → gen-report → X3 的 load_x2 讀得懂。
    合成數字只驗管線（欄位、去重、挑選、輸出檔），和真實 KV 無關。"""
    import csv
    import gzip
    import random
    import tempfile
    np = _np()
    ok0 = True
    with tempfile.TemporaryDirectory() as d:
        rnd = random.Random(0)
        for m in MODEL_INFO:
            info = MODEL_INFO[m]
            p = os.path.join(d, f"x1_{m}_s0.csv.gz")
            with gzip.open(p, "wt", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS_X1)
                w.writeheader()
                cs = load_combos(COMBOS, m)
                for i in range(50):
                    e0, fo0 = -3.0 + rnd.random(), int(rnd.random() < 0.7)
                    base = dict(run_id="SYN", ts="t", shard="0/1", model=m, item_id=f"T2_x_s{i}", task="T2", band="front",
                                seed=i, gold_lp_e0=e0, first_ok_e0=fo0, n_rows_item=len(cs) + 1, own_gpu_intruders=0,
                                host_level="QUIET")
                    w.writerow(dict(base, spec="E0", range="", roles="ref", family="none", families="none", n_lossy=0,
                                    deployable=1, ratio_region=1.0, kl=0.0, gold_lp=e0, top1=1.0, first_ok=fo0))
                    for c in cs:
                        r = bytes_summary(predict_bytes(c["stages"], info, 63), info, 63, 64)["ratio_region"]
                        eff = -0.05 / r ** 0.7 * (0.6 if len(c["stages"]) >= 2 else 1.0) + rnd.gauss(0, 0.05)
                        at = spec_attrs(c["stages"])
                        w.writerow(dict(base, spec=c["spec"], range="full", region="1-63", groups="|".join(c["groups"]),
                                        roles="|".join(c["roles"]), **at, ratio_region=r, kl=max(0.0, -eff), gold_lp=e0 + eff,
                                        top1=0.9, first_ok=int(fo0 and rnd.random() > min(0.9, -eff)), relerr_k=-eff,
                                        relerr_v=-eff * 0.5, cenerr_k=-eff, attn_err=-eff + rnd.gauss(0, 0.01),
                                        attn_err_max=-eff, attn_err_wo=-eff))
        out = os.path.join(d, "rep")
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            cmd_report(SimpleNamespace(raw=[os.path.join(d, "x1_*_s*.csv.gz")], range_name="full", out_dir=out,
                                       s1_summary="/nonexistent", lossless_codec="zstd:3"))
        sel = json.load(open(os.path.join(out, "x2_selection.json")))
        cfg = {m: sel["models"].get(m, {}).get("configs", []) for m in MODEL_INFO}
        tiers = {t for m in cfg for e in cfg[m] for t in e["tiers"]}
        excl = all(len(set(e["tiers"]) & {"core", "x3", "ext"}) <= 1 for m in cfg for e in cfg[m])
        check("e2e_report_selection", "go" in sel and all(cfg[m] for m in MODEL_INFO) and {"core", "ext", "x3", "128k"} <= tiers
              and excl
              and all(os.path.exists(os.path.join(out, f)) for f in ("x1_report.json", "tables_x1.md", "configs_llama.csv")),
              go=sel["go"], reason=sel["reason"], n={m: len(cfg[m]) for m in cfg}, tiers=sorted(tiers))
        ok0 &= bool(cfg["llama"])
        p2 = os.path.join(d, "x2_s0.csv.gz")
        with gzip.open(p2, "wt", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS_X2)
            w.writeheader()
            for m in MODEL_INFO:
                for i in range(40):
                    s0 = int(rnd.random() < 0.7)
                    rows = [dict(spec="E0", gen_score=s0, ratio_region=1.0, roles="ref", budgets="")]
                    for e in cfg[m]:
                        rows.append(dict(spec=e["spec"], gen_score=int(s0 and rnd.random() > 0.05), roles="|".join(e["roles"]),
                                         budgets="|".join(str(b) for b in e["budgets"]),
                                         ratio_region=(e["x1"] or {}).get("ratio", e.get("x1_ratio_predicted"))))
                    for r in rows:
                        w.writerow(dict(r, run_id="SYN2", ts="t", model=m, ctx="32K", item_id=f"T2_y_s{i}",
                                        n_rows_item=len(rows)))
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            cmd_gen_report(SimpleNamespace(raw=[p2], selection=os.path.join(out, "x2_selection.json"), out_dir=out))
        rep2 = json.load(open(os.path.join(out, "x2_report.json")))
        check("e2e_gen_report", "M1_verdict" in rep2["judge"] and os.path.exists(os.path.join(out, "tables_x2.md")),
              verdict=rep2["judge"]["M1_verdict"], x3=rep2.get("x3_files"))
        fp = os.path.join(out, "x2_feasible.csv")
        try:
            fp_x3 = x3_compatible_copy(fp, os.path.join(d, "x2_feasible_x3.csv"))
            import m16_xtime as XT
            x3 = XT.load_x2(fp_x3)
            check("e2e_x3_load_x2_accepts_x2_feasible", x3 is not None and set(x3["feasible"]) <= set(MODEL_INFO),
                  feasible={k: sorted(v) for k, v in x3["feasible"].items()})
        except ImportError as ex:
            check("e2e_x3_load_x2_accepts_x2_feasible", False, why=f"m16_xtime 匯入不了：{ex}")
        rows = list(csv.DictReader(open(fp))) if os.path.exists(fp) else []
        check("e2e_x2_feasible_columns", rows and {"model", "x3_segment", "feasible", "run_id", "mlperf99"} <= set(rows[0])
              and all(r["x3_segment"] in X3_SEGMENTS for r in rows), n=len(rows))
        del np
    return ok0


# ============================================================================ main
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("calib")
    c.add_argument("--model", choices=list(MODEL_INFO), required=True)
    c.add_argument("--tasks", nargs="+", default=["T2", "T3", "T4", "T5"])
    c.add_argument("--n", type=int, default=2, help="每個任務幾題（預設 2 → 8 題 × 約 32K）")
    c.add_argument("--seed0", type=int, default=CALIB_SEED0)
    c.add_argument("--params-file", default=PARAMS)
    c.add_argument("--joint-rank-max", type=int, default=64, help="聯合基底存到 每 head 等效秩×H 維")
    c.add_argument("--out", help=f"預設 {DATA}/calib_<model>.pt")
    c.add_argument("--summary-dir", default=OUT_DIR, help="摘要 JSON（run_id、能量、RoPE 核對）放哪；冒煙時放 smoke 目錄、不進 results")
    c.add_argument("--force", action="store_true")
    s = sp.add_parser("screen")
    s.add_argument("--model", choices=list(MODEL_INFO), required=True)
    s.add_argument("--params-file", default=PARAMS)
    s.add_argument("--tasks", nargs="+", default=["T2", "T3", "T4", "T5"])
    s.add_argument("--n", type=int, default=24, help="每個任務幾題（DS1／S2：24）")
    s.add_argument("--seed0", type=int, default=X1_SEED0)
    s.add_argument("--combos", default=COMBOS)
    s.add_argument("--only-specs", nargs="+")
    s.add_argument("--only-groups", nargs="+")
    s.add_argument("--calib", default=None)
    s.add_argument("--ranges", nargs="+", default=["full"], choices=list(REGIONS))
    s.add_argument("--no-s2", action="store_true", help="不跑 S2 的原本各組")
    s.add_argument("--s2-variants", nargs="+", default=["region", "full"], choices=["region", "full"])
    s.add_argument("--write-methods", nargs="+", default=["SNAP", "KEYDIFF"])
    s.add_argument("--ds2-plan", default=os.path.join(REPO, "results", "m10_drop_mi300x", "ds2", "ds2_plan.json"))
    s.add_argument("--extra-methods", nargs="+", default=["QRY", "RND0"])
    s.add_argument("--check-ds1", default=None, help="DS1 逐列檔（預設 S2 的 DS1_RAW；檔不在就不比）")
    s.add_argument("--s2-out", help="S2 格式的逐列（預設＝--out 的 x1_ 換成 s2part_）")
    s.add_argument("--proxy-nq", type=int, default=64)
    s.add_argument("--proxy-layers", default="all", help="all 或 k（每 k 層一層）")
    s.add_argument("--no-proxy-wo", action="store_true")
    s.add_argument("--budget-min", type=float, default=0, help="開跑超過這麼多分鐘就不再開新的題（0＝不限）")
    s.add_argument("--shard", default="0/1")
    s.add_argument("--out", required=True)
    s.add_argument("--smoke", action="store_true", help="1 題（DS1 第一題）、組合用清單的 smoke、印每一列")
    s.add_argument("--calib-xl", default=None, help="10/21：xKV 校準基底（有 xl:…:cal 的組合才要）")
    r = sp.add_parser("report")
    r.add_argument("--raw", nargs="+", required=True)
    r.add_argument("--range-name", default="full")
    r.add_argument("--out-dir", default=OUT_DIR)
    r.add_argument("--s1-summary", default=S1_SUMMARY)
    r.add_argument("--lossless-codec", default="zstd:3")
    g = sp.add_parser("gen")
    g.add_argument("--model", choices=list(MODEL_INFO), required=True)
    g.add_argument("--selection", default=os.path.join(OUT_DIR, "x2_selection.json"))
    g.add_argument("--ctx", choices=["32K", "128K"], default="32K")
    g.add_argument("--tier", choices=["core", "ext", "x3", "all", "128k"], default="core")
    g.add_argument("--tasks", nargs="+", default=["T2", "T3", "T4", "T5"])
    g.add_argument("--n", type=int, default=None, help="每個任務幾題（預設 32K 24；128K 12＝44c）")
    g.add_argument("--seed0", type=int, default=None, help=f"預設 32K {X2_SEED0}（新題，避開 X1 的挑選偏差）；128K {LC_SEED0}（44c）")
    g.add_argument("--params-file", default=PARAMS)
    g.add_argument("--calib", default=None)
    g.add_argument("--only-specs", nargs="+")
    g.add_argument("--check-44c", default=None)
    g.add_argument("--budget-min", type=float, default=0)
    g.add_argument("--shard", default="0/1")
    g.add_argument("--out", required=True)
    g.add_argument("--calib-xl", default=None, help="10/21：xKV 校準基底（有 xl:…:cal 的配置才要）")
    g.add_argument("--xl-cache", action="store_true",
                   help="10/21：同一題裡同一個 (K/V, 秩) 的每題 SVD 只算一次（xl:k22:v34+q:int4_kivi 與 +q:int8_token 共用；預設關＝照舊）")
    gr = sp.add_parser("gen-report")
    gr.add_argument("--raw", nargs="+", required=True)
    gr.add_argument("--selection", default=os.path.join(OUT_DIR, "x2_selection.json"))
    gr.add_argument("--out-dir", default=OUT_DIR)
    t = sp.add_parser("gpu-selftest")
    t.add_argument("--model", choices=list(MODEL_INFO), required=True)
    t.add_argument("--calib", required=True)
    t.add_argument("--params-file", default=PARAMS)
    t.add_argument("--out", required=True)
    t.add_argument("--new-checks", choices=["none", "x2b", "xlcal", "both"], default="none",
                   help="10/21：只跑新的檢查（x2b：出題、X2b 配置、--xl-cache；xlcal：校準基底）；none＝原本的全部檢查")
    t.add_argument("--calib-xl", default=None, help="--new-checks xlcal／both 要")
    # ---- 10/21 延伸（x_stack_howto.md §14）
    cx = sp.add_parser("calib-xl", help="xKV 事先校準的基底（GPU）")
    cx.add_argument("--model", choices=list(MODEL_INFO), required=True)
    cx.add_argument("--tasks", nargs="+", default=["T2", "T3", "T4", "T5"])
    cx.add_argument("--n", type=int, default=2, help="每個任務幾題（預設 2 → 8 題；和 PCA 的 calib 同一套）")
    cx.add_argument("--seed0", type=int, default=CALIB_SEED0)
    cx.add_argument("--params-file", default=PARAMS)
    cx.add_argument("--w", type=int, default=4, help="每個窗口幾層（xKV 預設 4）")
    cx.add_argument("--rank-max", type=int, default=XLCAL_RANK_MAX, help="存到「每 head 每層等效秩」多少（×W×H 維）")
    cx.add_argument("--out", help=f"預設 {DATA}/calib_xl_<model>.pt")
    cx.add_argument("--summary-dir", default=OUT_DIR_V2)
    cx.add_argument("--force", action="store_true")
    pl = sp.add_parser("plan", help="X2b／xKV 校準生成的計畫檔（CPU）")
    pl.add_argument("--which", choices=["x2b", "xlcal"], required=True)
    pl.add_argument("--selection", default=os.path.join(OUT_DIR, "x2_selection.json"), help="第一版的挑選（roles／budgets 照它）")
    pl.add_argument("--out", required=True)
    pe = sp.add_parser("pending", help="工作池用：這個 shard 還有幾題沒做（CPU）")
    pe.add_argument("--tasks", nargs="+", default=["T2", "T3", "T4", "T5"])
    pe.add_argument("--n", type=int, required=True)
    pe.add_argument("--seed0", type=int, required=True)
    pe.add_argument("--shard", required=True)
    pe.add_argument("--out", required=True)
    xr = sp.add_parser("x2b-report", help="X2＋X2b 的合併報告（CPU）")
    xr.add_argument("--raw-x2", nargs="+", required=True)
    xr.add_argument("--raw-x2b", nargs="+", required=True)
    xr.add_argument("--selection", default=os.path.join(OUT_DIR, "x2_selection.json"))
    xr.add_argument("--plan", default=os.path.join(OUT_DIR_V2, "x2b_selection.json"))
    xr.add_argument("--out-dir", default=OUT_DIR_V2)
    cr = sp.add_parser("xlcal-report", help="xKV 校準基底 vs 每題 SVD 的報告（CPU）")
    cr.add_argument("--raw-tf", nargs="+", required=True)
    cr.add_argument("--raw-gen", nargs="+", required=True)
    cr.add_argument("--raw-x1", nargs="+", required=True)
    cr.add_argument("--raw-x2", nargs="+", default=None)
    cr.add_argument("--calib-xl-summary", nargs="+", default=[os.path.join(OUT_DIR_V2, "calib_xl_*.json")])
    cr.add_argument("--out-dir", default=OUT_DIR_V2)
    return ap


COMMANDS = {"calib": "cmd_calib", "screen": "cmd_screen", "report": "cmd_report", "gen": "cmd_gen", "gen-report": "cmd_gen_report",
            "gpu-selftest": "cmd_gpu_selftest", "calib-xl": "cmd_calib_xl", "plan": "cmd_plan", "pending": "cmd_pending",
            "x2b-report": "cmd_x2b_report", "xlcal-report": "cmd_xlcal_report"}


def main():
    if "--self-test" in sys.argv[1:]:
        sys.exit(0 if self_test() else 1)
    a = build_parser().parse_args()
    globals()[COMMANDS[a.cmd]](a)


if __name__ == "__main__":
    main()
