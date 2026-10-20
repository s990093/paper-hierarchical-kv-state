#!/bin/bash
#SBATCH --job-name=p3h
#SBATCH --partition=h200q
#SBATCH --account=ChihChungHsu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=48:00:00
#SBATCH -o slurm-%x-%j.out
# =============================================================================
# r_h200_p3.sh — 第三階段（p3-predictor，3090）在平台 C（H200）上照跑（2026-10-19，使用者：「把完整目前實驗改適配到 h200」）
#
#   sbatch --job-name=p3h-data    --dependency=afterany:<前一個> code/r_h200_p3.sh --part data
#   sbatch --job-name=p3h-simtest --dependency=afterok:<data>    code/r_h200_p3.sh --part simtest
#   bash code/r_h200_p3.sh --part data --list | --dry-run ；--from <步驟>（接著跑）
#
#   --part data     資料下載（compute node 的本地 /tmp）→ 統一表重建（照 3090 的最終 run，03b §0.1）→ 對拍 → 測試
#                   → 和 3090 的 results/p3_data 逐值比（不依硬體：應該逐位元相同）
#   --part simtest  模擬器的全部測試（3090 最後一次：20261019-194510-p3sim-verify-test5，121 passed）
#
# 使用者 10/19 的決定（memory h200-p3-port）：
#   - 大檔案（原始 trace、統一表、模擬器快取、地圖的大輸出）放 hgx-cn01 的本地 /tmp（$P3_LOCAL），不佔 lab 的 /work 配額；
#     Slurm 沒有每個 job 私有的 /tmp（JobContainerType=none），所以 job 之間留得住；重開機或被別人塞滿會不見 → 每個 part 開頭驗，缺了就停（data 會重下載）。
#   - 純 CPU 的工作包在 1-GPU job 裡（h200q 不收 --gres=gpu:0）。GPU 這時是閒的。
#   - 第二階段 session 的計時 job（m10h-*）跑的時候不准同時跑（memory h200-no-self-cotenancy-for-timing；CPU 重的也算）→ 一律 --dependency 接在它們後面。
# 結果：results/p3_data_h200/（和 3090 的 results/p3_data 並排，不覆寫 3090 的檔）。每一步一個 $TIARA_RUNS/<RUN_ID>/（m7run）。
# 停止：任何一步失敗（規則 2）。
# =============================================================================
set -uo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
if [ -n "${REPO:-}" ] && [ -f "$REPO/code/env_h200.sh" ]; then :
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/code/env_h200.sh" ]; then REPO=$SLURM_SUBMIT_DIR
elif [ -f "$_here/env_h200.sh" ]; then REPO=$(cd "$_here/.." && pwd)
else echo "[r_h200_p3] 找不到 repo（code/env_h200.sh）" >&2; exit 2
fi
export REPO
if [ -z "${R_H200_P3_REEXEC:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  export R_H200_P3_REEXEC=1
  echo "[r_h200_p3] 改執行 repo 裡的版本：$REPO/code/r_h200_p3.sh（git $(git -C "$REPO" rev-parse --short HEAD)）"
  exec bash "$REPO/code/r_h200_p3.sh" "$@"
fi

PART=""; DRY=0; LIST=0; FROM=""
while [ $# -gt 0 ]; do
  case "$1" in
    --part) PART="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --list) LIST=1; DRY=1; shift ;;
    --from) FROM="$2"; shift 2 ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "不認得的參數：$1" >&2; exit 2 ;;
  esac
done
# --part 可以給好幾個、逗號分開，依序跑（同一個 job、同一張卡；排隊只排一次）。cal 是量時間的，只能單獨跑
for _p in ${PART//,/ }; do
  case "$_p" in data|simtest|explore|map) ;; *) echo "要 --part data|simtest|explore|map（可逗號分開），不是 '$_p'" >&2; exit 2 ;; esac
done
[ -n "$PART" ] || { echo "要 --part" >&2; exit 2; }

[ "$DRY" = 1 ] && export TIARA_SKIP_CONDA=1
# shellcheck disable=SC1091
source "$REPO/code/env_h200.sh"
cd "$REPO"
ulimit -c 0

# ---- 第三階段的環境（env_h200.sh 的值不改：TIARA_HOME、TIARA_RUNS 是第二階段 session 的 job 也在用的）----
export TIARA_P3_VENV="${TIARA_P3_VENV:-$TIARA_HOME/venvs/tiara-p3}"        # code/h200_setup_p3_env.sh 建的（tiara ＋ 第三階段套件）
export P3_LOCAL="${P3_LOCAL:-/tmp/hungweilai-p3}"                          # hgx-cn01 本地 md0，不在 lab 配額裡
export TIARA_DATASETS="$P3_LOCAL/datasets"                                 # 第三階段的原始 trace 與 p3_unified/（schema.DATASETS）
export P3SIM_CACHE="$P3_LOCAL/cache/p3_sim"                                # trace.prepare_cached 的 P 快取
export P3_DATA_CACHE="$P3_LOCAL/cache/p3_data"                             # load_wildchat 的 token 數快取
export P3_MAP_OUT="$P3_LOCAL/p3_map"                                       # run_map 的大輸出
export P3_DATA_RESULTS="$REPO/results/p3_data_h200"                        # 資料層的摘要／對拍（和 3090 的 results/p3_data 並排）
export TIARA_HF_HUB="$HF_HOME/hub"                                         # p3x 的 tokenizer／config 路徑
# venv 的 python 一律用絕對路徑：m7run 經 tiara.sbatch 會再 source env_h200.sh（conda activate tiara），PATH 上的 python 會變回 conda 的
export P3PY="$TIARA_P3_VENV/bin/python"
export TIARA_PY="$P3PY"                                                    # setup_3090_data.sh 用
if [ "$DRY" = 0 ]; then
  [ -n "${SLURM_JOB_ID:-}" ] || { echo "[r_h200_p3] 不在 Slurm job 裡" >&2; exit 2; }
  [ -x "$P3PY" ] || { echo "[r_h200_p3] 找不到 $P3PY（先在 head node 跑 bash code/h200_setup_p3_env.sh）" >&2; exit 2; }
  export TIARA_HF_CLI="$CONDA_PREFIX/bin/hf"                               # huggingface_hub 2.2.0 的 CLI（tiara）
  "$P3PY" -c "import sklearn, lightgbm, xgboost, statsmodels, lifelines, sksurv, pycox, pytest, numpy, pandas, pyarrow" \
    || { echo "[r_h200_p3] 第三階段的套件 import 不了 → 停" >&2; exit 2; }
  # 本地 /tmp：只有自己能讀寫；記下剩多少（被塞滿時要知道）
  mkdir -p "$P3_LOCAL" && chmod 700 "$P3_LOCAL" || { echo "[r_h200_p3] 建不了 $P3_LOCAL" >&2; exit 2; }
  mkdir -p "$TIARA_DATASETS" "$P3SIM_CACHE" "$P3_DATA_CACHE" "$P3_MAP_OUT" "$P3_DATA_RESULTS"
  echo "[r_h200_p3] $(date -Is) part=$PART job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-?} git=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "[r_h200_p3] P3PY=$P3PY（$("$P3PY" -c 'import sys; print(sys.version.split()[0], sys.prefix, sys.base_prefix)')）"
  df -h "$P3_LOCAL" /work | sed 's/^/[r_h200_p3] /'
  quota -gs 2>/dev/null | tail -1 | sed 's/^/[r_h200_p3] lab quota: /'
  echo "[r_h200_p3] 本地 /tmp 的清理規則："; grep -h -E '^[^#].*\s/tmp(\s|$)' /usr/lib/tmpfiles.d/*.conf /etc/tmpfiles.d/*.conf 2>/dev/null | sed 's/^/[r_h200_p3]   /'
  echo "[r_h200_p3] 同節點我們的 job（計時 job 在跑就不該有這個 job，見表頭）："; squeue -u "$USER" -w "${SLURMD_NODENAME:-hgx-cn01}" -h -o '  %i %j %T %M' 2>&1
fi

STARTED=1; [ -n "$FROM" ] && STARTED=0
SEEN=""
_gate() {
  local name=$1
  SEEN="$SEEN $name"
  if [ "$STARTED" = 0 ]; then
    if [ "$name" = "$FROM" ]; then STARTED=1; else [ "$LIST" = 0 ] && echo "[skip] $name"; return 1; fi
  fi
  return 0
}
step() {
  local name=$1; shift
  _gate "$name" || return 0
  if [ "$LIST" = 1 ]; then echo "$name"; return 0; fi
  if [ "$DRY" = 1 ]; then printf '[%s] ' "$name"; printf '%q ' "$@"; echo; return 0; fi
  echo "=== $(date -Is) $name"
  "$@"; local rc=$?
  if [ $rc -ne 0 ]; then
    echo "=== $(date -Is) $name 失敗（exit $rc）→ 停下、回報（規則 2）。接著跑：sbatch ... code/r_h200_p3.sh --part $PART --from $name" >&2
    exit $rc
  fi
}

# ===== data =====
P3_SOURCES="mooncake cachewise bailian burstgpt wildchat freeinference"   # 統一表的 6 個真實來源（03b §0；Azure 只有 X9 用，不在這次範圍）
data_dl() {
  # 半成品：目錄在、README.txt（最後才寫）不在 → 上次下載中斷。setup 的 fresh() 會把它當成已完成而跳過，所以先改名留著（不刪），再重下
  local it d
  for it in $P3_SOURCES; do
    case $it in
      mooncake) d=mooncake ;; cachewise) d=cachewise-coding-traces ;; bailian) d=qwen-bailian-usagetraces-anon ;;
      burstgpt) d=burstgpt ;; wildchat) d=wildchat-1m ;; freeinference) d=freeinference_agentic_trace ;;
    esac
    if [ -d "$TIARA_DATASETS/$d" ] && [ ! -f "$TIARA_DATASETS/$d/README.txt" ]; then
      echo "[data] $d 沒有 README.txt（上次沒下載完）→ 改名成 $d.incomplete-$(date +%Y%m%d-%H%M%S)"
      mv "$TIARA_DATASETS/$d" "$TIARA_DATASETS/$d.incomplete-$(date +%Y%m%d-%H%M%S)" || return 1
    fi
  done
  # hf download 的 Xet chunk 快取預設放 $HF_HOME/xet（＝~/hf-cache，在 /work 的 lab 配額裡；FreeInference 7.8 GB）→
  # 這一步的 HF_HOME 指到本地 /tmp、chunk 快取關掉。檔案本身由 --local-dir 直接寫進 $TIARA_DATASETS（也在本地 /tmp）
  lab_quota_ok 5 || return 1
  # shellcheck disable=SC2086
  HF_HOME="$P3_LOCAL/hf-home-dl" HF_XET_CHUNK_CACHE_SIZE_BYTES=0 m7run p3h-data-dl bash code/setup_3090_data.sh $P3_SOURCES
}
lab_quota_ok() {  # lab 的 /work 配額剩不到 $1 GiB 就停：寫滿時第二階段 session 的 job 也會失敗（它 10/19 提醒的）
  local free
  # quota -gw：第 3 行起「檔案系統 已用KiB 配額KiB …」（head node 是 hgx-cn01:/work，compute node 是 /dev/md1；超額時已用的數字後面有 *）
  free=$(quota -gw 2>/dev/null | awk 'NR>2 && NF>=4 {u=$2; q=$3; gsub(/\*/,"",u); print int((q-u)/1048576); exit}')
  [ -n "$free" ] || { echo "[r_h200_p3] 查不到 lab 配額（quota -gw）→ 停（規則 7）" >&2; return 1; }
  echo "[r_h200_p3] lab /work 配額剩 ${free} GiB（門檻 $1 GiB）"
  [ "$free" -ge "$1" ] || { echo "[r_h200_p3] lab 配額剩不到 $1 GiB → 停" >&2; return 1; }
}
data_verify() {   # 每個檔的 sha256 和下載時寫的 README.txt 相同（/tmp 不保證保留：每次用之前都驗）
  # shellcheck disable=SC2086
  m7run p3h-data-verify "$P3PY" -I code/h200_p3_verify_data.py "$TIARA_DATASETS" $P3_SOURCES
}
bailian_build() { # 3090 的 20261017-220343-p3data-bailian 也是 exit 5：traceB 沒有 parent 連結，斷言照預期不過（03b §0）。只准這一種失敗
  m7run p3h-data-bailian "$P3PY" -I code/p3_data/build.py bailian; local rc=$?
  [ "$rc" = 0 ] && return 0
  [ "$rc" = 5 ] || return $rc
  local f; f=$(ls "$TIARA_DATASETS"/p3_unified/bailian/*/FAILED.json 2>/dev/null)
  if [ "$f" = "$TIARA_DATASETS/p3_unified/bailian/traceB/FAILED.json" ]; then
    echo "[data] bailian exit 5，只有 traceB 的 FAILED.json（和 3090 相同，預期中）"; return 0
  fi
  echo "[data] bailian exit 5，但 FAILED.json 是：${f:-無}（不是只有 traceB）→ 停"; return 5
}
data_cmp() {
  m7run p3h-data-cmp "$P3PY" code/h200_p3_compare.py --ref results/p3_data --new "$P3_DATA_RESULTS" \
    --files summary.csv crosscheck.csv raw_id_chain.csv service_constants.json synth_fit_fi_week01.json \
    --out "$P3_DATA_RESULTS/compare_3090.json"
}
data_diag_const3090() {   # 10/20（job 21116 的 DATA-CMP）：FI 服務時間迴歸（statsmodels QuantReg）的 A／B／C 在 H200 和 3090 差 ≤ 7.4e-5（相對；
  # 輸入的分位數逐位元相同 → 差在數值庫），Mooncake 的 service_est_ms 四捨五入成整數 ms → think_ms 差 ±1 ms。
  # 診斷：用 3090 的 service_constants.json 把 Mooncake 重建到另一個目錄，summary 的 mooncake 列應和 3090 逐位元相同。
  # 只診斷：不寫 $TIARA_DATASETS/p3_unified、不動 $P3_DATA_RESULTS 的主檔；H200 照用自己的迴歸（P3_PORTING_NOTES 第 9 項）
  local D="$P3_LOCAL/diag-const3090"
  rm -rf "$D" && mkdir -p "$D/results" && cp results/p3_data/service_constants.json "$D/results/" || return 1
  m7run p3h-data-diag-const3090 env P3_DATA_RESULTS="$D/results" P3_UNIFIED="$D/unified" bash -c \
    '"$1" -I code/p3_data/build.py mooncake && "$1" -I code/p3_data/build.py summarize && "$1" code/h200_p3_compare.py --ref results/p3_data --new "$P3_DATA_RESULTS" --files summary.csv --keep source=mooncake --out "$2"' \
    _ "$P3PY" "$P3_DATA_RESULTS/diag_const3090_mooncake.json"
}

# ===== simtest =====
need_data() {     # simtest 之後的 part 都要資料層：缺了就停（不在這裡偷偷重建；重跑 --part data）
  local v
  for v in bailian/coder bailian/traceA wildchat/1m mooncake/conv-rule-strict freeinference/week01 synth/small-seed1; do
    [ -f "$TIARA_DATASETS/p3_unified/$v/turns.parquet" ] || { echo "[r_h200_p3] 缺 $TIARA_DATASETS/p3_unified/$v/turns.parquet（本地 /tmp 被清了？）→ 先跑 --part data" >&2; return 1; }
  done
}

# ===== explore：X1／X3／X4／X5／X8 換 H200 常數（P3_CONST=h200），照 3090 各方向「登記的主結果」那幾步（docs/phase3_20261017/explore/X*_*.md §3）
# 事後補的變體（FreeInference 版、截尾、嚴格規則…）這次不跑。輸出 results/p3_explore_h200/x*，大的原始檔放本地 /tmp（$RAW）
RAW="$P3_LOCAL/raw"
const_h200() { export P3_CONST=h200 M7_RES="$REPO/results/m7_write_policy_h200" P3_MAP_PLATFORM=h200; }
const_mi300x() { export P3_CONST=mi300x M7_RES="$REPO/results/m7_write_policy_mi300x" P3_MAP_PLATFORM=mi300x; }
lastrun() { ls -d "$TIARA_RUNS"/*-"$1" 2>/dev/null | sort | tail -1; }
fresh_dir() {     # 輸出目錄裡已經有上一次的（--from 接著跑或重跑）→ 改名留著，不刪、不附加（x1_sim sweep 是附加寫入）
  local d=$1; [ -e "$d" ] && { echo "[r_h200_p3] $d 已存在 → 改名 $d.stale-$(date +%Y%m%d-%H%M%S)"; mv "$d" "$d.stale-$(date +%Y%m%d-%H%M%S)" || return 1; }
  mkdir -p "$d"
}
X=$REPO/results/p3_explore_h200
x1_sweep() { fresh_dir "$RAW/x1-sweep" && { [ ! -e "$X/x1/x1_sweep.csv" ] || mv "$X/x1/x1_sweep.csv" "$X/x1/x1_sweep.csv.stale-$(date +%Y%m%d-%H%M%S)"; } && mkdir -p "$X/x1" \
  && m7run p3h-x1-sweep env RUN_DIR="$RAW/x1-sweep" "$P3PY" code/p3x/x1_sim.py sweep --workers 14; }
x4_analyze() { local d; d=$(lastrun p3h-x4-sweep); [ -f "$d/x4_sweep.csv" ] || { echo "沒有 $d/x4_sweep.csv"; return 1; }
  m7run p3h-x4-analyze "$P3PY" code/p3x/x4_sim.py analyze --inp "$d/x4_sweep.csv"; }
x8_traces() { fresh_dir "$RAW/x8-traces" && mkdir -p "$X/x8" && m7run p3h-x8-traces "$P3PY" -I code/p3x/x8_traces.py --out "$RAW/x8-traces" --csv "$X/x8/x8_traces.csv" \
  && m7run p3h-x8-fi "$P3PY" -I code/p3x/x8_fi.py --out "$RAW/x8-traces" --csv "$X/x8/x8_traces.csv"; }
x3_leads() {
  fresh_dir "$RAW/x3-events" || return 1
  local c; for c in consts bailian "mooncake --rule loose" "mooncake --rule strict" "wildchat --files 2" cachewise freeinference; do
    # shellcheck disable=SC2086
    m7run "p3h-x3-leads-$(echo $c | tr -cs 'a-z0-9' '-' | sed 's/-$//')" bash -c "\"$P3PY\" code/p3x/x3_leads.py $c --run-id \"\$RUN_ID\" --run-dir \"$RAW/x3-events\"" || return 1
  done
}
x3_hide() { local ev; ev=$(ls "$RAW"/x3-events/events_*.csv.gz | wc -l); echo "[x3] 事件檔 $ev 個（3090 是 12 個）"
  m7run p3h-x3-hide bash -c "\"$P3PY\" code/p3x/x3_hide.py --run-id \"\$RUN_ID\" --tables \"$RAW/x3-tables/tables\" --events $RAW/x3-events/events_*.csv.gz"; }

# ===== map：地圖的工具（不是地圖本身；完整地圖等 04 v4 凍結，使用者 10/19）
# 先用 MI300X 常數重現 3090 的合成端到端與 C0 使用率（逐值比），再換 H200 常數（P3_MAP_PLATFORM=h200）
MAPRES=$REPO/results/p3_map_h200
SIMRES=$REPO/results/p3_sim_h200
map_syn() {       # $1＝mi300x|h200：合成端到端 A1（3090：20261019-004253-p3sim-mapsyn4）＋五組分析（anasyn5）
  local plat=$1 out="$P3_MAP_OUT/synthetic_e2e_A1_v6_$1" g o r
  fresh_dir "$out" || return 1
  m7run "p3h-mapsyn-$plat" "$P3PY" code/p3_sim/run_map.py --stage A1 --slices synth/small-seed1 --procs 14 --out "$out" || return 1
  mkdir -p "$MAPRES"
  for g in "C0 greedy none" "C0 greedy pass_size" "C0 tier none" "C0 tier pass_size" "min_C0_O greedy none"; do
    set -- $g
    m7run "p3h-anasyn-$plat-$1-$2-$3" "$P3PY" code/p3_sim/analyze_map.py --cells "$out/map_cells.csv" --synthetic --mode full \
      --guard-ref "$1" --oracle-variant "$2" --recompute-charge "$3" --out-prefix "$MAPRES/synthetic_e2e_A1_v6_${plat}_$1_$2_$3" || return 1
  done
}
map_c0util() {    # $1＝mi300x|h200（3090：20261019-003643-p3sim-c0util，--procs 16；這裡 12：job 只有 170 GB，一個 worker 約 10 GB）
  local plat=$1; fresh_dir "$P3_MAP_OUT/c0util_$plat" || return 1; mkdir -p "$SIMRES"
  m7run "p3h-c0util-$plat" "$P3PY" code/p3_sim/run_map.py --c0-util --out "$P3_MAP_OUT/c0util_$plat" --c0-util-csv "$SIMRES/c0_ssd_util_$plat.csv" --procs 12 --warm-procs 2
}
map_cmp_mi300x() {  # MI300X 常數的重跑 vs 3090 的檔（不依硬體：應該逐值相同）
  local c
  m7run p3h-map-cmp "$P3PY" code/h200_p3_compare.py --ref results/p3_sim --new "$SIMRES" --files c0_ssd_util.csv --out "$SIMRES/compare_3090_c0util.json" \
    --rename "c0_ssd_util.csv=c0_ssd_util_mi300x.csv" || return 1
  for c in C0_greedy_none C0_greedy_pass_size C0_tier_none C0_tier_pass_size min_C0_O_greedy_none; do
    m7run "p3h-map-cmp-$c" "$P3PY" code/h200_p3_compare.py --ref results/p3_map --new "$MAPRES" \
      --files "synthetic_e2e_A1_v6_${c}_summary.csv" "synthetic_e2e_A1_v6_${c}_tradeoff.csv" "synthetic_e2e_A1_v6_${c}_verdict.json" \
      --rename "synthetic_e2e_A1_v6_${c}_summary.csv=synthetic_e2e_A1_v6_mi300x_${c}_summary.csv" \
               "synthetic_e2e_A1_v6_${c}_tradeoff.csv=synthetic_e2e_A1_v6_mi300x_${c}_tradeoff.csv" \
               "synthetic_e2e_A1_v6_${c}_verdict.json=synthetic_e2e_A1_v6_mi300x_${c}_verdict.json" \
      --ignore cells_csv --out "$MAPRES/compare_3090_$c.json" || return 1
  done
}

# ===== cal：不做（2026-10-19 決定，RUNLOG_H200_P3 P3H-0b）。3090 的 CPU 忙校準是「整台主機忙」（N 到 48、兩個 NUMA 節點）；
# Slurm 的 1-GPU job 只有 16 個 CPU 的 cpuset，做不出同一個條件，整台加壓又會影響同節點別的 lab 的 job。
# 這個數字只進地圖的敏感度（只報告），所以 H200 的 CPU 忙寫 NOT_MEASURED，不量、也不借 3090 的值。

run_part() {
case "$1" in
  explore)
    const_h200
    step EX-NEED need_data
    step EX-VERIFY data_verify
    step X1-SELFTEST m7run p3h-x1-selftest "$P3PY" code/p3x/x1_sim.py selftest
    step X1-SWEEP x1_sweep
    step X1-ANALYZE m7run p3h-x1-analyze "$P3PY" code/p3x/x1_sim.py analyze --sweep x1_sweep.csv --traces conversation toolagent
    step X1-REPORT m7run p3h-x1-report "$P3PY" code/p3x/x1_report.py --sweep x1_sweep.csv
    step X1-BOOT m7run p3h-x1-boot "$P3PY" code/p3x/x1_boot.py --run-dirs "$RAW/x1-sweep" --B 2000
    step X3-VALIDATE m7run p3h-x3-validate bash -c "\"$P3PY\" code/p3x/x3_restore.py validate --run-id \"\$RUN_ID\""
    step X3-TABLES fresh_dir "$RAW/x3-tables"
    step X3-TABLES-RUN m7run p3h-x3-tables bash -c "\"$P3PY\" code/p3x/x3_restore.py tables --procs 14 --run-id \"\$RUN_ID\" --run-dir \"$RAW/x3-tables\""
    step X3-LEADS x3_leads
    step X3-HIDE x3_hide
    step X4-TRACE m7run p3h-x4-trace "$P3PY" code/p3x/x4_sim.py trace --rules loose strict
    step X4-VALIDATE m7run p3h-x4-validate "$P3PY" code/p3x/x4_sim.py validate
    step X4-SWEEP m7run p3h-x4-sweep "$P3PY" code/p3x/x4_sim.py sweep --procs 14 --rules loose strict --out x4_sweep.csv
    step X4-ANALYZE x4_analyze
    step X5-GRID fresh_dir "$RAW/x5-grid"
    step X5-GRID-RUN m7run p3h-x5-grid "$P3PY" code/p3x/x5_run.py --out-dir "$X/x5" --raw-dir "$RAW/x5-grid" --procs 14
    step X5-ANALYZE m7run p3h-x5-analyze "$P3PY" code/p3x/x5_analyze.py --out-dir "$X/x5" --raw-dir "$RAW/x5-grid"
    # X5 的成本常數出來之後重建 H200 格式表：補上和 X5 的對拍（head node 的版本 20261019-211449-p3h-fmt-table2 沒有這兩項）
    step FMT-TABLE m7run p3h-fmt-table3 "$P3PY" code/p3_formats/build_format_table.py
    step X8-TRACES x8_traces
    step X8-CAPACITY m7run p3h-x8-capacity "$P3PY" -I code/p3x/x8_capacity.py "$RAW/x8-traces"
    step X8-GRID m7run p3h-x8-grid3 "$P3PY" -I code/p3x/x8_sim.py --traces "$RAW/x8-traces" --outdir "$X/x8" --rawdir "$RAW/x8-grid3" --procs 14 \
      --wl mc_conv mc_tool mc_conv_s mc_tool_s bl_coder cw_orig cw_stack
    step X8-GRID-FI m7run p3h-x8-grid3-fi env X8_NMAX=2560 "$P3PY" -I code/p3x/x8_sim.py --traces "$RAW/x8-traces" --outdir "$X/x8" --rawdir "$RAW/x8-grid3-fi" \
      --procs 14 --wl fi fi_clip --lens orig --tag _fi
    step X8-VERDICT m7run p3h-x8-verdict "$P3PY" -I code/p3x/x8_verdict.py
    ;;
  map)
    step MAP-NEED need_data
    step MAP-VERIFY data_verify
    const_mi300x
    step MAP-SYN-MI300X map_syn mi300x
    step MAP-C0-MI300X map_c0util mi300x
    step MAP-CMP-3090 map_cmp_mi300x
    const_h200
    step MAP-SYN-H200 map_syn h200
    step MAP-C0-H200 map_c0util h200
    ;;
  data)
    step DATA-DL data_dl
    step DATA-VERIFY data_verify
    # 統一表：照 3090 最終 run 的順序與分批（03b §0.1）
    step DATA-FI-W01 m7run p3h-data-fi2-w01 "$P3PY" -I code/p3_data/build.py freeinference --variants week01 week02 week03 week04 week05 week06 week07 week08 week09 week10 week11 week12
    step DATA-FI-W13 m7run p3h-data-fi2-w13 "$P3PY" -I code/p3_data/build.py freeinference --variants week13
    step DATA-FI-W14 m7run p3h-data-fi2-w14 "$P3PY" -I code/p3_data/build.py freeinference --variants week14 week15
    step DATA-FI-W16 m7run p3h-data-fi2-w16 "$P3PY" -I code/p3_data/build.py freeinference --variants week16
    step DATA-FICONST m7run p3h-data-ficonst "$P3PY" -I code/p3_data/build.py fi-constants
    step DATA-BAILIAN bailian_build
    step DATA-MOONCAKE m7run p3h-data-mooncake "$P3PY" -I code/p3_data/build.py mooncake
    step DATA-BURSTGPT m7run p3h-data-burstgpt "$P3PY" -I code/p3_data/build.py burstgpt
    step DATA-CACHEWISE m7run p3h-data-cachewise "$P3PY" -I code/p3_data/build.py cachewise
    step DATA-WILDCHAT m7run p3h-data-wildchat "$P3PY" -I code/p3_data/build.py wildchat
    step DATA-SYNTHFIT m7run p3h-data-synthfit "$P3PY" -I code/p3_data/synth.py fit
    step DATA-SYNTH m7run p3h-data-synth "$P3PY" -I code/p3_data/build.py synth
    step DATA-SUMMARY m7run p3h-data-summary "$P3PY" -I code/p3_data/build.py summarize
    step DATA-XCHECK m7run p3h-data-crosscheck "$P3PY" -I code/p3_data/crosscheck.py
    # raw_id_chain.csv：3090 是另外用 --chain-only 寫的（d0c4658）；21116 漏了這步 → DATA-CMP 報 MISSING
    step DATA-CHAIN m7run p3h-data-chain "$P3PY" -I code/p3_data/crosscheck.py --chain-only
    step DATA-TEST m7run p3h-data-test "$P3PY" -I -m pytest -q code/p3_data/test_p3_data.py
    step DATA-CMP data_cmp
    step DATA-DIAG-CONST3090 data_diag_const3090
    ;;
  simtest)
    const_mi300x                # 測試照 3090：MI300X 常數（x1_sim.constants）；M7_RES 要明設（env_h200.sh 預設是 H200 的）
    step SIM-NEED need_data
    step SIM-VERIFY data_verify
    step SIM-TEST m7run p3h-sim-test "$P3PY" -m pytest code/p3_sim/ -q
    ;;
esac
}
for _p in ${PART//,/ }; do run_part "$_p"; done
if [ "$LIST" = 0 ] && [ "$DRY" = 0 ]; then
  echo "[r_h200_p3] $(date -Is) part=$PART 全部完成"
  du -sh "$P3_LOCAL"/* 2>/dev/null | sed 's/^/[r_h200_p3] /'
fi
if [ -n "$FROM" ] && [ "$STARTED" = 0 ]; then echo "沒有叫做 '$FROM' 的步驟。可用的：$SEEN" >&2; exit 2; fi
