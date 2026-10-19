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
case "$PART" in data|simtest) ;; *) echo "要 --part data|simtest" >&2; exit 2 ;; esac

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

# ===== simtest =====
need_data() {     # simtest 之後的 part 都要資料層：缺了就停（不在這裡偷偷重建；重跑 --part data）
  local v
  for v in bailian/coder bailian/traceA wildchat/1m mooncake/conv-rule-strict freeinference/week01 synth/small-seed1; do
    [ -f "$TIARA_DATASETS/p3_unified/$v/turns.parquet" ] || { echo "[r_h200_p3] 缺 $TIARA_DATASETS/p3_unified/$v/turns.parquet（本地 /tmp 被清了？）→ 先跑 --part data" >&2; return 1; }
  done
}

case "$PART" in
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
    step DATA-TEST m7run p3h-data-test "$P3PY" -I -m pytest -q code/p3_data/test_p3_data.py
    step DATA-CMP data_cmp
    ;;
  simtest)
    step SIM-NEED need_data
    step SIM-VERIFY data_verify
    step SIM-TEST m7run p3h-sim-test "$P3PY" -m pytest code/p3_sim/ -q
    ;;
esac
if [ "$LIST" = 0 ] && [ "$DRY" = 0 ]; then
  echo "[r_h200_p3] $(date -Is) part=$PART 全部完成"
  du -sh "$P3_LOCAL"/* 2>/dev/null | sed 's/^/[r_h200_p3] /'
fi
if [ -n "$FROM" ] && [ "$STARTED" = 0 ]; then echo "沒有叫做 '$FROM' 的步驟。可用的：$SEEN" >&2; exit 2; fi
