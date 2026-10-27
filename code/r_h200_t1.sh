#!/bin/bash
#SBATCH --job-name=icpe-t1
#SBATCH --partition=h200q
#SBATCH --account=ChihChungHsu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=4:00:00
#SBATCH -o slurm-%x-%j.out
# =============================================================================
# r_h200_t1.sh — ICPE T1（docs/h200/icpe/T1_h200_cpu_tier.md，登記 2026-10-27T22:50:34Z）：CPU 層換成 H200 的常數（P3_TIER=h200cpu），
# 重跑 ICPE_EVIDENCE B、C 節讀 CPU 層常數的模擬。只用 CPU、不量時間（同 ICPE 的 cpu／p12 part；h200q 不收 --gres=gpu:0，GPU 閒著）。
#
#   env -u TMPDIR -u CLAUDE_CODE_TMPDIR sbatch code/r_h200_t1.sh --part t1
#   bash code/r_h200_t1.sh --part t1 --list | --dry-run ；--from <步驟> 接著跑
#
# 每一步和原本同一支程式、同樣的參數（照抄 r_h200_icpe.sh 的 cpu／p12 part 與 r_h200_p3.sh 的 explore part），只多 P3_TIER、
# 輸出改到 results/icpe_h200/t1/（大的原始檔在 hgx-cn01 本地 /tmp/hungweilai-p3 的 t1 子目錄）。不改 r_h200_icpe.sh／r_h200_p3.sh。
#   B1／B2／B7  P0-1 (c) 的 X1，V0／V1／V2：sweep、analyze、--dead-h 1800／3600
#   C2         P0-2 fullsim（全尺寸 HINT-PF）
#   B3（X1）    P1-2 A2：X1 FreeInference 前 6 週（Cake）
#   C1         X3：validate、tables、leads（7 個來源）、hide
#   B4         X4 主掃描：trace、validate、sweep、analyze（排隊占比 p12_queue.py 在 head node 跑）
# X5 不跑（不讀 CPU 層常數；t1.py selftest 第 3 項）。停止：任何一步失敗（規則 2）。
# =============================================================================
set -uo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
if [ -n "${REPO:-}" ] && [ -f "$REPO/code/env_h200.sh" ]; then :
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/code/env_h200.sh" ]; then REPO=$SLURM_SUBMIT_DIR
elif [ -f "$_here/env_h200.sh" ]; then REPO=$(cd "$_here/.." && pwd)
else echo "[r_h200_t1] 找不到 repo（code/env_h200.sh）" >&2; exit 2
fi
export REPO
if [ -z "${R_H200_T1_REEXEC:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  export R_H200_T1_REEXEC=1
  echo "[r_h200_t1] 改執行 repo 裡的版本：$REPO/code/r_h200_t1.sh（git $(git -C "$REPO" rev-parse --short HEAD)）"
  exec bash "$REPO/code/r_h200_t1.sh" "$@"
fi

PART=""; DRY=0; LIST=0; FROM=""
while [ $# -gt 0 ]; do
  case "$1" in
    --part) PART="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --list) LIST=1; DRY=1; shift ;;
    --from) FROM="$2"; shift 2 ;;
    *) echo "不認得的參數：$1" >&2; exit 2 ;;
  esac
done
[ "$PART" = t1 ] || { echo "要 --part t1" >&2; exit 2; }

[ "$DRY" = 1 ] && export TIARA_SKIP_CONDA=1
# shellcheck disable=SC1091
source "$REPO/code/env_h200.sh"
cd "$REPO"
ulimit -c 0

# ---- T1 的開關（P3_TIER 讓 p3_platform 換 CPU 層、輸出改到 t1；M7_TIER_PARAMS 給 m7_*：x4_sim 本來就會設，這裡再設一次當保險）----
T1=$REPO/results/icpe_h200/t1
export P3_CONST=h200 P3_TIER=h200cpu M7_TIER_PARAMS=$T1/tier_params_h200cpu.json
export M7_RES=$REPO/results/m7_write_policy_h200 P3_MAP_PLATFORM=h200      # r_h200_p3.sh 的 const_h200
# ---- r_h200_icpe.sh 的環境（cpu／p12 part）----
P01=$REPO/results/icpe_h200/p01                                           # V1 的校準（calib/）在這裡；只讀
export X1_TRACE_DIR=$TIARA_HOME/datasets/traces X5_TRACE_DIR=$TIARA_HOME/datasets/traces
export M10_SWEEP_DIR=$REPO/results/m10_sweep_h200 M10_PRECISION_DIR=$REPO/results/m10_precision_h200
export M10_BUSY_LEVELS=$REPO/results/m7_write_policy_h200/busy_levels.json
P3LOCAL=/tmp/hungweilai-p3/icpe-t1
# ---- r_h200_p3.sh 的環境（explore part）----
export TIARA_P3_VENV="${TIARA_P3_VENV:-$TIARA_HOME/venvs/tiara-p3}"
export P3_LOCAL="${P3_LOCAL:-/tmp/hungweilai-p3}"
P3DATA="$P3_LOCAL/datasets"                                               # explore 與 p12 用的 TIARA_DATASETS（各步驟自己設，同原本）
export P3SIM_CACHE="$P3_LOCAL/cache/p3_sim" P3_DATA_CACHE="$P3_LOCAL/cache/p3_data" TIARA_HF_HUB="$HF_HOME/hub"
export P3PY="$TIARA_P3_VENV/bin/python"
RAW="$P3_LOCAL/raw-t1"
P3_SOURCES="mooncake cachewise bailian burstgpt wildchat freeinference"

if [ "$DRY" = 0 ]; then
  [ -n "${SLURM_JOB_ID:-}" ] || { echo "[r_h200_t1] 不在 Slurm job 裡" >&2; exit 2; }
  [ -x "$P3PY" ] || { echo "[r_h200_t1] 找不到 $P3PY" >&2; exit 2; }
  [ -f "$M7_TIER_PARAMS" ] || { echo "[r_h200_t1] 找不到 $M7_TIER_PARAMS（先在 head node 跑 code/icpe/t1.py build）" >&2; exit 2; }
  for f in calib_c1.csv calib_c5.csv; do [ -f "$P01/calib/$f" ] || { echo "[r_h200_t1] 缺 $P01/calib/$f（V1 要用）" >&2; exit 2; }; done
  mkdir -p "$T1" "$P3LOCAL" "$RAW"
  echo "[r_h200_t1] $(date -Is) part=$PART job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-?} git=$(git rev-parse --short HEAD)"
  echo "[r_h200_t1] 同節點我們的 job："; squeue -u "$USER" -w "${SLURMD_NODENAME:-hgx-cn01}" -h -o '  %i %j %T %M' 2>&1
  df -h "$P3_LOCAL" /work | sed 's/^/[r_h200_t1] /'
  python3 - "$TIARA_HOME/datasets/traces" <<'PY' || { echo "[r_h200_t1] Mooncake trace 的 sha256 不對 → 停" >&2; exit 2; }
import hashlib, sys
want = {"conversation_trace.jsonl": "b8cbb061a85206d7", "toolagent_trace.jsonl": "48a2db1a13d3bc05"}
for f, h in want.items():
    got = hashlib.sha256(open(f"{sys.argv[1]}/{f}", "rb").read()).hexdigest()
    print("[r_h200_t1]", f, got[:16]); assert got.startswith(h), (f, got)
PY
fi

STARTED=1; [ -n "$FROM" ] && STARTED=0
step() {
  local name=$1; shift
  if [ "$STARTED" = 0 ]; then if [ "$name" = "$FROM" ]; then STARTED=1; else return 0; fi; fi
  if [ "$LIST" = 1 ]; then echo "$name"; return 0; fi
  if [ "$DRY" = 1 ]; then printf '[%s] ' "$name"; printf '%q ' "$@"; echo; return 0; fi
  echo "=== $(date -Is) $name"
  "$@"; local rc=$?
  if [ $rc -ne 0 ]; then echo "=== $(date -Is) $name 失敗（exit $rc）→ 停下、回報（規則 2）。接著跑：sbatch ... code/r_h200_t1.sh --part t1 --from $name" >&2; exit $rc; fi
}
lastrun() { command ls -d "$TIARA_RUNS"/*-"$1" 2>/dev/null | sort | tail -1; }
fresh() {         # 已經有上一次的 → 改名留著（x1_sim sweep 是附加寫入；--from 重跑時不能疊上去）
  local d
  for d in "$@"; do [ -e "$d" ] && { echo "[r_h200_t1] $d 已存在 → 改名 $d.stale-$(date +%Y%m%d-%H%M%S)"; mv "$d" "$d.stale-$(date +%Y%m%d-%H%M%S)" || return 1; }; done
  return 0
}
need_data() {     # 同 r_h200_p3.sh
  local v
  for v in bailian/coder bailian/traceA wildchat/1m mooncake/conv-rule-strict freeinference/week01 synth/small-seed1; do
    [ -f "$P3DATA/p3_unified/$v/turns.parquet" ] || { echo "[r_h200_t1] 缺 $P3DATA/p3_unified/$v/turns.parquet（本地 /tmp 被清了？）" >&2; return 1; }
  done
}
x1_sweep() {      # x1_sweep <短名> <版本> <out-root> <raw> [環境變數指派...] -- <x1_sim sweep 的參數...>
  local nm=$1 V=$2 root=$3 raw=$4; shift 4
  local ev=(); while [ $# -gt 0 ] && [ "$1" != -- ]; do ev+=("$1"); shift; done; shift
  fresh "$raw" "$root/x1/x1_sweep.csv" "$root/x1/x1_sweep_fi.csv" && mkdir -p "$raw" || return 1
  m7run "$nm" env RUN_DIR="$raw" ${ev[@]+"${ev[@]}"} python code/icpe/p3_fvar.py --variant "$V" --calib "$P01/calib" --out-root "$root" x1 sweep "$@"
}
x3_leads() {      # 同 r_h200_p3.sh（7 個來源）
  local c; for c in consts bailian "mooncake --rule loose" "mooncake --rule strict" "wildchat --files 2" cachewise freeinference; do
    # shellcheck disable=SC2086
    m7run "icpe-t1-x3-leads-$(echo $c | tr -cs 'a-z0-9' '-' | sed 's/-$//')" env TIARA_DATASETS="$P3DATA" bash -c "\"$P3PY\" code/p3x/x3_leads.py $c --run-id \"\$RUN_ID\" --run-dir \"$RAW/x3-events\"" || return 1
  done
}

step T1-SELFTEST m7run icpe-t1-selftest python code/icpe/t1.py selftest
# ---- B1／B2／B7：P0-1 (c) 的 X1（r_h200_icpe.sh cpu part；X5 不跑）----
for V in V0 V1 V2; do
  FV="python code/icpe/p3_fvar.py --variant $V --calib $P01/calib --out-root $T1/p01/sim_$V"
  step "T1-X1-$V-SWEEP" x1_sweep "icpe-t1-x1-sweep-$V" "$V" "$T1/p01/sim_$V" "$P3LOCAL/x1sweep_$V" -- --workers 14
  step "T1-X1-$V-AN"    m7run "icpe-t1-x1-an-$V" $FV x1 analyze --sweep x1_sweep.csv --traces conversation toolagent
  step "T1-X1-$V-H1800" m7run "icpe-t1-x1-h1800-$V" $FV x1 analyze --sweep x1_sweep.csv --traces conversation toolagent --dead-h 1800 --out-prefix x1h1800
  step "T1-X1-$V-H3600" m7run "icpe-t1-x1-h3600-$V" $FV x1 analyze --sweep x1_sweep.csv --traces conversation toolagent --dead-h 3600 --out-prefix x1h3600
done
# ---- C2：P0-2 全尺寸 HINT-PF ----
step T1-FULLSIM m7run icpe-t1-p02-fullsim python code/icpe/x1_gpu.py fullsim --out "$T1/p02"
# ---- B3（X1）：P1-2 A2（r_h200_icpe.sh p12 part）----
WEEKS="fi:2026-05-17_2026-05-23 fi:2026-05-24_2026-05-30 fi:2026-05-31_2026-06-06 fi:2026-06-07_2026-06-13 fi:2026-06-14_2026-06-20 fi:2026-06-21_2026-06-27"
FV12="python code/icpe/p3_fvar.py --variant V0 --out-root $T1/p12"
step T1-P12-VERIFY m7run icpe-t1-p12-data-verify "$P3PY" -I code/h200_p3_verify_data.py "$P3DATA" mooncake freeinference
step T1-P12-X1FI-SWEEP  x1_sweep icpe-t1-p12-x1fi-sweep V0 "$T1/p12" "$P3LOCAL/p12_x1fi" TIARA_DATASETS="$P3DATA" X4_TRACE_DIR="$TIARA_HOME/datasets/traces" -- --workers 14 --traces $WEEKS --rules session_id --modes cake --out x1_sweep_fi.csv
step T1-P12-X1FI-AN     m7run icpe-t1-p12-x1fi-an env TIARA_DATASETS="$P3DATA" $FV12 x1 analyze --sweep x1_sweep_fi.csv --traces freeinference --out-prefix x1fi
step T1-P12-X1FI-AN-USER m7run icpe-t1-p12-x1fiuser-an env TIARA_DATASETS="$P3DATA" $FV12 x1 analyze --sweep x1_sweep_fi.csv --traces freeinference --metric mean_ret_user --out-prefix x1fiuser
# ---- C1、B4：explore 的 X3、X4（r_h200_p3.sh explore part；P3_TIER 讓 explore_out 指到 t1/explore）----
step T1-EX-NEED need_data
# shellcheck disable=SC2086
step T1-EX-VERIFY m7run icpe-t1-ex-data-verify "$P3PY" -I code/h200_p3_verify_data.py "$P3DATA" $P3_SOURCES
[ "$DRY" = 1 ] || mkdir -p "$T1/explore/x3" "$T1/explore/x4"
step T1-X3-VALIDATE m7run icpe-t1-x3-validate env TIARA_DATASETS="$P3DATA" bash -c "\"$P3PY\" code/p3x/x3_restore.py validate --run-id \"\$RUN_ID\""
step T1-X3-TABLES fresh "$RAW/x3-tables"
step T1-X3-TABLES-RUN m7run icpe-t1-x3-tables env TIARA_DATASETS="$P3DATA" bash -c "mkdir -p \"$RAW/x3-tables\" && \"$P3PY\" code/p3x/x3_restore.py tables --procs 14 --run-id \"\$RUN_ID\" --run-dir \"$RAW/x3-tables\""
step T1-X3-EVENTS bash -c "{ [ ! -e \"$RAW/x3-events\" ] || mv \"$RAW/x3-events\" \"$RAW/x3-events.stale-\$(date +%Y%m%d-%H%M%S)\"; } && mkdir -p \"$RAW/x3-events\""
step T1-X3-LEADS x3_leads
step T1-X3-HIDE m7run icpe-t1-x3-hide env TIARA_DATASETS="$P3DATA" bash -c "\"$P3PY\" code/p3x/x3_hide.py --run-id \"\$RUN_ID\" --tables \"$RAW/x3-tables/tables\" --events $RAW/x3-events/events_*.csv.gz"
step T1-X4-TRACE    m7run icpe-t1-x4-trace    env TIARA_DATASETS="$P3DATA" "$P3PY" code/p3x/x4_sim.py trace --rules loose strict
step T1-X4-VALIDATE m7run icpe-t1-x4-validate env TIARA_DATASETS="$P3DATA" "$P3PY" code/p3x/x4_sim.py validate
step T1-X4-SWEEP    m7run icpe-t1-x4-sweep    env TIARA_DATASETS="$P3DATA" "$P3PY" code/p3x/x4_sim.py sweep --procs 14 --rules loose strict --out x4_sweep.csv
if [ "$DRY" = 1 ]; then X4S="<icpe-t1-x4-sweep 的 run 目錄>"; else X4S=$(lastrun icpe-t1-x4-sweep); fi
step T1-X4-ANALYZE  m7run icpe-t1-x4-analyze  env TIARA_DATASETS="$P3DATA" "$P3PY" code/p3x/x4_sim.py analyze --inp "$X4S/x4_sweep.csv"
echo "=== $(date -Is) T1 compute-node 部分完成。head node：p12_queue.py、t1.py report、E2E-C 的 T1 版"
