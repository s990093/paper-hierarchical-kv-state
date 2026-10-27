"""p3_platform.py — 第三階段的「成本常數平台」：探索（p3x）與 build_inputs 讀哪一台的第一、二階段實測檔（2026-10-19，H200 移植）。

  P3_CONST=mi300x（預設；3090 分支原本的行為，逐位元不變）
  P3_CONST=h200   （平台 C：讀 results/*_h200 的同一批實測；輸出改寫到 results/p3_explore_h200/）

規則（memory h200-phase1-reproduction、h200-p3-port）：
  * 只換「GPU 量出來的常數」：f(i)（C1、calib-f）、GPU 忙倍數（busy_levels.json）、還原格子（反量化時間、位元組）、b* 限速頻寬（bw.json）。
  * 限速層（cpu／local／nfs 的 GiB/s 與寫入干擾）兩台都用 MI300X 的 tier_params.json（03_plan_h200 §R2：只換 GPU、不換「裝置」；H200 沒有自己的 tier_params）。
  * 每 block 位元組（iowall.json）是格式的定義、和 GPU 無關 → 兩台都讀 MI300X 的檔（H200 沒有 iowall.json）；用的地方要斷言 BF16 位元組相同。
  * 對應不到的（H200 沒有那個 run）→ KeyError，不准退回 MI300X 的值（規則 7）。
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PLAT = os.environ.get("P3_CONST", "mi300x")
if PLAT not in ("mi300x", "h200"):
    raise SystemExit(f"P3_CONST={PLAT!r}：只接受 mi300x、h200")

# 第一、二階段的結果目錄：MI300X 名稱 → 這個平台的名稱（tier_params、iowall 例外，見上）
_DIRS = {
    "m7_write_policy_mi300x": {"mi300x": "m7_write_policy_mi300x", "h200": "m7_write_policy_h200"},
    "m10_precision_mi300x": {"mi300x": "m10_precision_mi300x", "h200": "m10_precision_h200"},
    "m10_sweep_mi300x": {"mi300x": "m10_sweep_mi300x", "h200": "m10_sweep_h200"},
    "m7_explore_mi300x": {"mi300x": "m7_explore_mi300x", "h200": "m7_explore_h200"},
    "m8_directions": {"mi300x": "m8_directions", "h200": "m8_directions_h200"},
}
# 還原格子的 run：MI300X run_id → H200 run_id（同一個 stage、同模型、同格式；13_h200_repro.md、RUNLOG_H200 10-H200）
_RUNS = {
    "20261015-154558-m10-calibf-llama": "20261017-074151-m10h-calibf-llama",
    "20261015-154647-m10-calibf-qwen": "20261017-074209-m10h-calibf-qwen",
    "20261015-161226-m10-grid-fp8-llama": "20261017-075056-m10h-grid-fp8-llama",
    "20261016-034902-m10-grid-int4-llama": "20261017-094511-m10h-grid-int4-llama",
    "20261015-191338-m10-grid-2b-fp8k0_token-llama": "20261017-093130-m10h-grid-2b-fp8k0_token-llama",
    "20261015-192112-m10-grid-2b-int8_token-llama": "20261017-093513-m10h-grid-2b-int8_token-llama",
    "20261015-175050-m10-grid-fp8-qwen": "20261017-084454-m10h-grid-fp8-qwen",
    "20261016-052143-m10-grid-int4-qwen": "20261017-103540-m10h-grid-int4-qwen",
    "20261015-192845-m10-grid-2b-fp8k0_head-qwen": "20261017-093854-m10h-grid-2b-fp8k0_head-qwen",
    "20261015-193516-m10-grid-2b-int8_token-qwen": "20261017-094204-m10h-grid-2b-int8_token-qwen",
    "20261016-232855-m10-dqfast-grid-llama": "20261018-143455-m10h-dqfast-grid-fp8-llama",
    "20261016-235027-m10-dqfast-grid-int4-llama": "20261018-144048-m10h-dqfast-grid-int4-llama",
    "20261016-234030-m10-dqfast-grid-qwen": "20261018-144636-m10h-dqfast-grid-fp8-qwen",
    "20261017-000209-m10-dqfast-grid-int4-qwen": "20261018-145130-m10h-dqfast-grid-int4-qwen",
    # MI300X 的 Qwen FP8／INT8 用重量（regrid；第一次的格子有污染列，format_table 的對拍）；H200 只量一次，那一次就是判定用的格子
    "20261015-194150-m10-regrid-fp8-qwen": "20261017-084454-m10h-grid-fp8-qwen",
    "20261015-210456-m10-regrid-2b-int8_token-qwen": "20261017-094204-m10h-grid-2b-int8_token-qwen",
}
TIER_PARAMS = os.path.join(REPO, "results", "m7_write_policy_mi300x", "tier_params.json")
IOWALL = os.path.join(REPO, "results", "m10_tier_mi300x", "iowall.json")
# T1（docs/h200/icpe/T1_h200_cpu_tier.md，2026-10-28；ICPE 只用 H200）：P3_TIER=h200cpu → CPU 層換成 H200 自己量的（SSD、NFS 照舊），
# 探索的輸出改到 results/icpe_h200/t1/explore/。沒設＝原本的行為，逐位元不變。
TIER = os.environ.get("P3_TIER", "")
if TIER not in ("", "h200cpu"):
    raise SystemExit(f"P3_TIER={TIER!r}：只接受 h200cpu（或不設）")
if TIER and PLAT != "h200":
    raise SystemExit(f"P3_TIER={TIER} 只配 P3_CONST=h200（現在是 {PLAT}）")
T1_ROOT = os.path.join(REPO, "results", "icpe_h200", "t1")
if TIER:
    TIER_PARAMS = os.path.join(T1_ROOT, "tier_params_h200cpu.json")


def rdir(mi300x_name, *parts):
    """results/<這個平台的目錄>/<parts>。mi300x_name 是原本程式裡寫的 MI300X 目錄名。"""
    return os.path.join(REPO, "results", _DIRS[mi300x_name][PLAT], *parts)


def run(mi300x_run_id):
    """原本寫死的 MI300X run_id → 這個平台對應的 run_id（mi300x 時原樣傳回）。"""
    return mi300x_run_id if PLAT == "mi300x" else _RUNS[mi300x_run_id]


def explore_out(x):
    """探索的輸出目錄：mi300x＝results/p3_explore/<x>（原本的）；h200＝results/p3_explore_h200/<x>（不覆寫 3090 的結果）；
    P3_TIER=h200cpu＝results/icpe_h200/t1/explore/<x>（不覆寫 H200 原本的結果）。"""
    if TIER:
        return os.path.join(T1_ROOT, "explore", x)
    return os.path.join(REPO, "results", "p3_explore" if PLAT == "mi300x" else "p3_explore_h200", x)


def check_m7_res():
    """m8_trace_oracle／m7_write_policy 在 import 時讀 M7_RES（calib_c1.csv 的位置）。它必須和 P3_CONST 是同一台，
    不然 f(i) 是一台、其他常數是另一台（env_h200.sh 預設把 M7_RES 指到 H200）。
    P3_CONST=mi300x 時不檢查（3090 分支原本的行為不變；在 H200 上跑 mi300x 常數時由 code/r_h200_p3.sh 明設 M7_RES）。"""
    if PLAT == "mi300x":
        return
    want = rdir("m7_write_policy_mi300x")
    got = os.environ.get("M7_RES")
    if got is None or os.path.realpath(got) != os.path.realpath(want):
        raise SystemExit(f"M7_RES={got} 和 P3_CONST={PLAT}（{want}）不是同一台：f(i) 和其他常數會混用 → 停")
