#!/bin/bash
# =============================================================================
# r_h200_cpu.sh — 第一階段的 CPU 模擬，用 H200 的常數重跑（2026-10-16，使用者：「gogo」）
#
#   bash code/r_h200_cpu.sh --part a            # 只需要 R2 常數的那一批（現在就能跑）
#   bash code/r_h200_cpu.sh --part d4d5         # D4（Mooncake trace、SCBench）、D5（模型 config）：資料由 code/h200_dl_cpu_data.sh 下載
#   bash code/r_h200_cpu.sh --part d6           # D6：反量化成本用 H200 的 F3 工作 A（results/m9_followup_h200/f3_dequant.csv）
#   bash code/r_h200_cpu.sh --part b            # 08 消融、B 的模擬驗證、第 1 輪的分析：要 r-h200-rest（P1–P3）與 r-h200-qwen3 跑完
#   bash code/r_h200_cpu.sh --part brr          # B 重跑（job 20631）的合併組：模擬驗證、08 消融、H1、B 全部設定，和舊的並排（code/h200_brr.py）
#   bash code/r_h200_cpu.sh --part a2c7         # A2、C7 錨點（job 20643，只有限速層）的分析，和 MI300X 並排（code/h200_a2c7.py）
#   bash code/r_h200_cpu.sh --part d3           # D3 κ 地圖（--no-3090）：舊 B 一次、B 重跑的合併組一次（要先跑 --part brr、a2c7 之後的 c7_anchor）
#   bash code/r_h200_cpu.sh --part r1h          # 第 1 輪 H200 自己的 3 個存活設定（job 20660）的 GPU 確認：A、B、C 各一次 m7_round1_analyze gpu
#   bash code/r_h200_cpu.sh --part a --dry-run | --list ；--from <步驟> [--until <步驟>]（接著跑）
#
# 在 head node 跑（h200q 不收純 CPU 的 job：--gres=gpu:0 → 「Invalid qos specification」，10/16 試過）。
# 用 nice -n 19、PROCS 個 process（預設 8；MI300X 是 24。結果只看各設定自己的 seed，process 數不影響數字）。
# H200_SLURM.md §5 寫「head node 不跑重的 CPU 工作」〔推論〕：這批是使用者同意在 head node 跑的。
#
# 常數（03 §R2：只換 GPU、不換「裝置」）：
#   f(i)              H200 的 C1（$M7_RES/calib_c1.csv，R2；env_h200.sh 的 M7_RES）；LongAlpaca 用 $M7_EXPLORE 的 C1
#   限速層（cpu/local/nfs） MI300X 的 tier_params.json（M7_TIER_PARAMS，和 GPU 端一樣）
#   capture 的 D2H     H200 C0 的 D2H_pinned 64 MiB 中位數，取一位小數（MI300X 用同一個算法得到 44.8）→ M7_D2H_GIBPS
#   vLLM 的 CPU 頻寬 3.69 GiB/s（m7_round1_sim 的 GRID 等）：照 r-h200-rest 的 P1-cpuvllm，沿用 MI300X 的值
# 輸出：$M7_RES（sim_*）、results/m7_explore_h200/（r1_sim_*）、results/m8_directions_h200/（d1_*）、results/m9_followup_h200/（f6_*、f7_*）
#
# 指令照 MI300X 留下結果的那些 run（docs/migration_20261013/inventory/run_commands.tsv；註解裡是 MI300X 的 run_id），
# 只換常數與目錄。冒煙測試（d1-smoke、f6-smoke）和中途改過的版本（sim-sweep 第一次、r1-sim 第一次、d6-arith 前幾次）不跑。
# 每一步一個 $TIARA_RUNS/<RUN_ID>/（§4.1：cmd.sh、context.txt、stdout.log、stderr.log、exit_code）。失敗就停（規則 2）。
# =============================================================================
set -uo pipefail

DRY=0; LIST=0; FROM=""; UNTIL=""; DONE=0; PART=""; PROCS=${PROCS:-8}
while [ $# -gt 0 ]; do
  case "$1" in
    --part) PART="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --list) LIST=1; DRY=1; shift ;;
    --from) FROM="$2"; shift 2 ;;
    --until) UNTIL="$2"; shift 2 ;;
    -h|--help) sed -n '2,23p' "$0"; exit 0 ;;
    *) echo "不認得的參數：$1（--help 看用法）" >&2; exit 2 ;;
  esac
done
case "$PART" in a|d6|d4d5|b|brr|a2c7|d3|r1h) ;; *) echo "[r_h200_cpu] 要指定 --part a、d4d5、d6、b、brr、a2c7、d3 或 r1h" >&2; exit 2 ;; esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO
[ "$DRY" = 1 ] && export TIARA_SKIP_CONDA=1
# shellcheck disable=SC1091
source "$REPO/code/env_h200.sh"
cd "$REPO"
export M7_EXPLORE=$REPO/results/m7_explore_h200
export M8_OUT=$REPO/results/m8_directions_h200
export M9_OUT=$REPO/results/m9_followup_h200
export HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES=      # 和 MI300X 的指令一樣：模擬不碰 GPU
IDS=$M8_OUT/r_h200_cpu_run_ids.tsv                     # 標籤<TAB>run_id：後面的步驟要讀前面 sweep 的輸出

# capture 的 D2H：H200 C0 的 D2H_pinned 64 MiB 中位數（MI300X：44.756 → 程式寫 44.8）
if [ "$DRY" = 0 ]; then
  M7_D2H_GIBPS=$(python - "$M7_RES/calib_c0.csv" <<'PY'
import csv, statistics, sys
v = [float(r["value"]) for r in csv.DictReader(open(sys.argv[1])) if r["item"] == "D2H_pinned" and r["size_mib"] == "64"]
assert len(v) >= 3, f"C0 的 D2H_pinned 64 MiB 只有 {len(v)} 筆"
print(round(statistics.median(v), 1))
PY
) || { echo "[r_h200_cpu] 算不出 M7_D2H_GIBPS（$M7_RES/calib_c0.csv）→ 停" >&2; exit 2; }
  export M7_D2H_GIBPS
  [ -f "$M7_RES/calib_c1.csv" ] || { echo "[r_h200_cpu] 找不到 H200 的 $M7_RES/calib_c1.csv" >&2; exit 2; }
  [ -f "$M7_EXPLORE/calib_c1_longalpaca7b.csv" ] || { echo "[r_h200_cpu] 找不到 $M7_EXPLORE/calib_c1_longalpaca7b.csv" >&2; exit 2; }
  mkdir -p "$M8_OUT" "$M9_OUT" "$TIARA_RUNS"
  echo "[r_h200_cpu] $(date -Is) host=$(hostname) part=$PART PROCS=$PROCS M7_RES=$M7_RES M7_TIER_PARAMS=$M7_TIER_PARAMS M7_D2H_GIBPS=$M7_D2H_GIBPS git=$(git rev-parse --short HEAD)"
else
  M7_D2H_GIBPS="<H200 C0>"
fi

# ---- 步驟（和 r_h200_r1x.sh 同一套：失敗就停、--from／--until）----------------------------
STARTED=1; [ -n "$FROM" ] && STARTED=0
SEEN=""
_gate() {
  local name=$1
  SEEN="$SEEN $name"
  if [ "$DONE" = 1 ]; then [ "$LIST" = 0 ] && echo "[skip] $name（--until $UNTIL 之後）"; return 1; fi
  if [ "$STARTED" = 0 ]; then
    if [ "$name" = "$FROM" ]; then STARTED=1; else [ "$LIST" = 0 ] && echo "[skip] $name"; return 1; fi
  fi
  return 0
}
# hrun <標籤> <短名> <指令...>：head node 版的 §4.1 記錄殼（tiara.sbatch 只在 Slurm job 裡跑）。成功才記「標籤<TAB>run_id」
hrun() {
  local label=$1 short=$2; shift 2
  [ "$LIST" = 1 ] && { echo "$label"; return 0; }
  if [ "$DRY" = 1 ]; then printf '[%s] ' "$label"; printf '%q ' "$@"; echo; return 0; fi
  local id; id="$(date +%Y%m%d-%H%M%S)-$short"
  local run=$TIARA_RUNS/$id
  mkdir -p "$run"
  { printf 'cd %q\n' "$REPO"; printf 'env RUN_ID=%q ' "$id"; printf '%q ' "$@"; echo; } > "$run/cmd.sh"
  { date -Is; hostname; echo "git $(git rev-parse HEAD)；未 commit 的檔：$(git status --short | wc -l)"; echo "label=$label PROCS=$PROCS"
    env | grep -E '^(M7_|M8_|M9_|TIARA_|HF_HOME|CUDA_VISIBLE|HIP_VISIBLE)' | sort; } > "$run/context.txt"
  echo "=== $(date -Is) $label → $id"
  nice -n 19 env RUN_ID="$id" "$@" > "$run/stdout.log" 2> "$run/stderr.log"; local rc=$?
  echo $rc > "$run/exit_code"
  if [ $rc -ne 0 ]; then
    echo "[r_h200_cpu] $label 失敗（exit $rc）→ 停下、回報（CLAUDE.md 規則 2）。stderr 末 20 行：" >&2; tail -20 "$run/stderr.log" >&2
    echo "接著跑：bash code/r_h200_cpu.sh --part $PART --from $label" >&2
    exit $rc
  fi
  printf '%s\t%s\n' "$label" "$id" >> "$IDS"
}
step() {    # step <標籤> <短名> <指令...>
  local label=$1
  _gate "$label" || return 0
  [ -n "$UNTIL" ] && [ "$label" = "$UNTIL" ] && DONE=1
  hrun "$@"
}
rid() {     # rid <標籤>：那一步最後一次成功的 run 目錄
  if [ "$DRY" = 1 ]; then echo "\$TIARA_RUNS/<$1>"; return 0; fi
  local id; id=$(awk -F'\t' -v l="$1" '$1==l{x=$2} END{print x}' "$IDS" 2>/dev/null)
  [ -n "$id" ] || { echo "[r_h200_cpu] $IDS 裡沒有 $1（那一步還沒成功過）" >&2; echo "MISSING-$1"; return 1; }
  echo "$TIARA_RUNS/$id"
}

P=python
SIM=code/m7_sim.py; R1=code/m7_round1_sim.py; C=code/m8_conc_sim.py; F6=code/m9_f6_analyze.py; F7=code/m9_f7_fscale_sim.py
TW13="S1 S2b S4 S4+ S4B S4W S4C S4PD S4BT S4WT S4BD S4WD S1D"
TW15_S5T="$TW13 S4WD50 S4WT50"
TW14_S1D="S1 S2b S4 S4+ S4B S4W S4C S4PD S4BT S4WT S4BD S4WD S4WD50 S4WT50"
TW17_S5T="$TW15_S5T S4CT S4CT50"
TW16_S1D="$TW14_S1D S4CT S4CT50"
ST_AF="S1 S2b S4 S4+ S4B S4W S4C S4PD S4BT S4WT S5T S5L S4BD S4WD S1D"
ST_F6="S1 S2b S4 S4+ S4B S4W S4C S4PD S4BT S4WT S5T S4BD S4WD S1D S4WD50 S4WT50 S4CT S4CT50"

if [ "$PART" = a ]; then
# ===== m7_sim：08 的掃描（MI300X 20261009-065112-m7-sim-sweep2）與 probe（065301–065343）=====
# validate 逐格對第一階段 B 的 GPU 列：要等 r-h200-rest 的 P1 跑完（--part b）
step SIM-sweep            m7h-sim-sweep            $P $SIM sweep
step SIM-probe-g1         m7h-sim-probe-g1         $P $SIM probe --cpu-gibps 3.69 --release hold --workload doc --seeds 0 1 2 3 4 --ssd-devs nfs local
step SIM-probe-c35-f025   m7h-sim-probe-doc-c35.4159-free-f0.25 $P $SIM probe --cpu-gibps 35.4159 --release free --workload doc --f-scale 0.25 --seeds 0 1 2 3 4 --ssd-devs nfs local
step SIM-probe-c11-f025   m7h-sim-probe-doc-c11.6-free-f0.25    $P $SIM probe --cpu-gibps 11.6 --release free --workload doc --f-scale 0.25 --seeds 0 1 2 3 4 --ssd-devs nfs local
step SIM-probe-c11h-f05   m7h-sim-probe-doc-c11.6-hold-f0.5     $P $SIM probe --cpu-gibps 11.6 --release hold --workload doc --f-scale 0.5 --seeds 0 1 2 3 4 --ssd-devs nfs local
step SIM-probe-c35-f05    m7h-sim-probe-doc-c35.4159-free-f0.5  $P $SIM probe --cpu-gibps 35.4159 --release free --workload doc --f-scale 0.5 --seeds 0 1 2 3 4 --ssd-devs nfs local

# ===== 第 1 輪的模擬（MI300X 20261009-180002-r1-sim2-*）=====
step R1SIM-llama  m7h-r1-sim-llama31-8b  $P $R1 --model llama31_8b
step R1SIM-la     m7h-r1-sim-longalpaca7b env M7_CHUNK_BYTES=$((256 << 20)) $P $R1 --model longalpaca7b --f-csv "$M7_EXPLORE/calib_c1_longalpaca7b.csv"

# ===== D1：多 session 併發模擬（MI300X 20261010-0546xx–0603xx）=====
step D1-validate     m7h-d1-validate      $P $C validate                      # 對 R1SIM-llama 的 r1_sim_llama31_8b.csv
step D1-sweep        m7h-d1-sweep         $P $C sweep --procs "$PROCS" --out d1_sweep.csv
step D1-analyze      m7h-d1-analyze       $P $C analyze --inp "$(rid D1-sweep)/d1_sweep.csv"
step D1-posthoc      m7h-d1-posthoc       $P $C sweep --procs "$PROCS" --strats S4BD S4WD S1D S5T S4BT --out d1_sweep_posthoc.csv
step D1-regress      m7h-d1-regress       $P $C regress --inp "$(rid D1-sweep)/d1_sweep.csv" "$(rid D1-posthoc)/d1_sweep_posthoc.csv"
step D1-an-posthoc   m7h-d1-analyze-posthoc $P $C analyze --inp "$(rid D1-sweep)/d1_sweep.csv" "$(rid D1-posthoc)/d1_sweep_posthoc_new.csv" --twins $TW13 --out-seed d1_posthoc_seed.csv --out-cfg d1_posthoc_cfg.csv
step D1-admitfirst   m7h-d1-admitfirst    $P $C sweep --procs "$PROCS" --admit-first --strats $ST_AF --out d1_sweep_admitfirst.csv
step D1-an-af        m7h-d1-analyze-admitfirst $P $C analyze --inp "$(rid D1-admitfirst)/d1_sweep_admitfirst.csv" --twins $TW13 --out-seed d1_admitfirst_seed.csv --out-cfg d1_admitfirst_cfg.csv
step D1-wm50         m7h-d1-wm50          $P $C sweep --procs "$PROCS" --admit-first --strats S4WD50 S4WT50 --out d1_sweep_wm50.csv
step D1-local8       m7h-d1-local8        $P $C sweep --procs "$PROCS" --admit-first --ssd-dev local8 --strats $ST_AF S4WD50 S4WT50 --out d1_sweep_local8.csv
step D1-qmax16       m7h-d1-qmax16        $P $C sweep --procs "$PROCS" --qmax 16 --ssd-dev nfs --workload doc --lam 0.05 0.1 0.2 --out d1_sweep_qmax16.csv
step D1-an-mean      m7h-d1-an-mean       $P $C analyze --inp "$(rid D1-sweep)/d1_sweep.csv" --metric mean --out-seed d1_verdict_mean_seed.csv --out-cfg d1_verdict_mean_cfg.csv
step D1-an-q16       m7h-d1-an-q16        $P $C analyze --inp "$(rid D1-qmax16)/d1_sweep_qmax16.csv" --out-seed d1_qmax16_seed.csv --out-cfg d1_qmax16_cfg.csv
step D1-an-x3s5t     m7h-d1-an-x3s5t      $P $C analyze --inp "$(rid D1-admitfirst)/d1_sweep_admitfirst.csv" "$(rid D1-wm50)/d1_sweep_wm50.csv" --twins $TW15_S5T --out-seed d1_x3_S5T_seed.csv --out-cfg d1_x3_S5T_cfg.csv
step D1-an-x3s1d     m7h-d1-an-x3s1d      $P $C analyze --inp "$(rid D1-admitfirst)/d1_sweep_admitfirst.csv" "$(rid D1-wm50)/d1_sweep_wm50.csv" --cand S1D --twins $TW14_S1D --out-seed d1_x3_S1D_seed.csv --out-cfg d1_x3_S1D_cfg.csv
step D1-an-l8s5t     m7h-d1-an-l8s5t      $P $C analyze --inp "$(rid D1-local8)/d1_sweep_local8.csv" --twins $TW15_S5T --out-seed d1_local8_S5T_seed.csv --out-cfg d1_local8_S5T_cfg.csv
step D1-an-l8s1d     m7h-d1-an-l8s1d      $P $C analyze --inp "$(rid D1-local8)/d1_sweep_local8.csv" --cand S1D --twins $TW14_S1D --out-seed d1_local8_S1D_seed.csv --out-cfg d1_local8_S1D_cfg.csv
step D1-ct           m7h-d1-ct            $P $C sweep --procs "$PROCS" --admit-first --strats S4CT S4CT50 --out d1_sweep_ct.csv
step D1-ct-l8        m7h-d1-ct-l8         $P $C sweep --procs "$PROCS" --admit-first --ssd-dev local8 --strats S4CT S4CT50 --out d1_sweep_ct_local8.csv
step D1-an-x4        m7h-d1-an-x4         $P $C analyze --inp "$(rid D1-admitfirst)/d1_sweep_admitfirst.csv" "$(rid D1-wm50)/d1_sweep_wm50.csv" "$(rid D1-ct)/d1_sweep_ct.csv" --cand S1D --twins $TW16_S1D --out-seed d1_x4_S1D_seed.csv --out-cfg d1_x4_S1D_cfg.csv
step D1-an-x4l8      m7h-d1-an-x4l8       $P $C analyze --inp "$(rid D1-local8)/d1_sweep_local8.csv" "$(rid D1-ct-l8)/d1_sweep_ct_local8.csv" --cand S1D --twins $TW16_S1D --out-seed d1_x4_S1D_local8_seed.csv --out-cfg d1_x4_S1D_local8_cfg.csv
step D1-an-x4m       m7h-d1-an-x4m        $P $C analyze --inp "$(rid D1-admitfirst)/d1_sweep_admitfirst.csv" "$(rid D1-wm50)/d1_sweep_wm50.csv" "$(rid D1-ct)/d1_sweep_ct.csv" --cand S1D --twins $TW16_S1D --metric mean --out-seed d1_x4_S1D_mean_seed.csv --out-cfg d1_x4_S1D_mean_cfg.csv
step D1-an-x4l8m     m7h-d1-an-x4l8m      $P $C analyze --inp "$(rid D1-local8)/d1_sweep_local8.csv" "$(rid D1-ct-l8)/d1_sweep_ct_local8.csv" --cand S1D --twins $TW16_S1D --metric mean --out-seed d1_x4_S1D_local8_mean_seed.csv --out-cfg d1_x4_S1D_local8_mean_cfg.csv
step D1-an-s5l       m7h-d1-an-s5l        $P $C analyze --inp "$(rid D1-sweep)/d1_sweep.csv" --cand S5L --out-seed d1_S5L_seed.csv --out-cfg d1_S5L_cfg.csv
step D1-ovsum        m7h-d1-ovsum         $P $C ovsum --inp "$(rid D1-sweep)/d1_sweep.csv" --out-cfg d1_overflow_main.csv
step D1-ovsum-af     m7h-d1-ovsum-af      $P $C ovsum --inp "$(rid D1-admitfirst)/d1_sweep_admitfirst.csv" --out-cfg d1_overflow_admitfirst.csv
step D1-af-q16       m7h-d1-af-q16        $P $C sweep --procs "$PROCS" --admit-first --qmax 16 --ssd-dev nfs local local8 --strats S1 S2b S4 S4+ S4B S4W S4C S4PD S4BT S4WT S5T S5L S4BD S4WD S1D S4WD50 S4WT50 S4CT S4CT50 --out d1_sweep_af_qmax16.csv
step D1-an-afq16-s5t m7h-d1-an-afq16-s5t  $P $C analyze --inp "$(rid D1-af-q16)/d1_sweep_af_qmax16.csv" --twins $TW17_S5T --out-seed d1_afq16_S5T_seed.csv --out-cfg d1_afq16_S5T_cfg.csv
step D1-an-afq16-s1d m7h-d1-an-afq16-s1d  $P $C analyze --inp "$(rid D1-af-q16)/d1_sweep_af_qmax16.csv" --cand S1D --twins $TW16_S1D --out-seed d1_afq16_S1D_seed.csv --out-cfg d1_afq16_S1D_cfg.csv
step D1-an-x4s5t     m7h-d1-an-x4s5t      $P $C analyze --inp "$(rid D1-admitfirst)/d1_sweep_admitfirst.csv" "$(rid D1-wm50)/d1_sweep_wm50.csv" "$(rid D1-ct)/d1_sweep_ct.csv" --twins $TW17_S5T --out-seed d1_x4_S5T_seed.csv --out-cfg d1_x4_S5T_cfg.csv
step D1-an-x4l8s5t   m7h-d1-an-x4l8s5t    $P $C analyze --inp "$(rid D1-local8)/d1_sweep_local8.csv" "$(rid D1-ct-l8)/d1_sweep_ct_local8.csv" --twins $TW17_S5T --out-seed d1_x4_S5T_local8_seed.csv --out-cfg d1_x4_S5T_local8_cfg.csv
# 重跑一次主掃描，逐列要相同（MI300X 20261010-060319／060344）
step D1-final-rerun  m7h-d1-final-rerun   $P $C sweep --procs "$PROCS" --out d1_sweep_rerun.csv
step D1-final-regress m7h-d1-final-regress $P $C regress --inp "$(rid D1-sweep)/d1_sweep.csv" "$(rid D1-final-rerun)/d1_sweep_rerun.csv" --regress-strats S1 S2b S4 S4+ S4B S4W S4C S4PD S4BT S4WT S5T S5L --regress-extract 0

# ===== F6：D1 的穩健性（MI300X 20261010-0801xx–0810xx）=====
for spec in "main S5T x4_S5T $TW17_S5T" "main S1D x4_S1D $TW16_S1D" "l8 S5T x4_S5T_local8 $TW17_S5T" "l8 S1D x4_S1D_local8 $TW16_S1D" \
            "q16 S5T afq16_S5T $TW17_S5T" "q16 S1D afq16_S1D $TW16_S1D"; do
  set -- $spec; g=$1 cand=$2 ref=$3; shift 3; tw="$*"
  case $g in
    main) inp=("$(rid D1-admitfirst)/d1_sweep_admitfirst.csv" "$(rid D1-wm50)/d1_sweep_wm50.csv" "$(rid D1-ct)/d1_sweep_ct.csv") ;;
    l8)   inp=("$(rid D1-local8)/d1_sweep_local8.csv" "$(rid D1-ct-l8)/d1_sweep_ct_local8.csv") ;;
    q16)  inp=("$(rid D1-af-q16)/d1_sweep_af_qmax16.csv") ;;
  esac
  step "F6-chk-$g-${cand,,}" "m7h-f6-chk-$g-${cand,,}" $P $F6 check --k 3 --tag "chk_${g/l8/l8}_${cand}" --cand "$cand" --twins $tw --inp "${inp[@]}" --ref "$M8_OUT/d1_${ref}_cfg.csv"
done
step F6-chk-main-s1d-mean m7h-f6-chk-main-s1d-mean $P $F6 check --k 3 --tag chk_main_S1D_mean --cand S1D --twins $TW16_S1D --metrics mean --inp "$(rid D1-admitfirst)/d1_sweep_admitfirst.csv" "$(rid D1-wm50)/d1_sweep_wm50.csv" "$(rid D1-ct)/d1_sweep_ct.csv" --ref "$M8_OUT/d1_x4_S1D_mean_cfg.csv"
step F6-chk-l8-s1d-mean   m7h-f6-chk-l8-s1d-mean   $P $F6 check --k 3 --tag chk_l8_S1D_mean --cand S1D --twins $TW16_S1D --metrics mean --inp "$(rid D1-local8)/d1_sweep_local8.csv" "$(rid D1-ct-l8)/d1_sweep_ct_local8.csv" --ref "$M8_OUT/d1_x4_S1D_local8_mean_cfg.csv"
step F6-reg-main   m7h-f6-reg-main   $P $C sweep --procs "$PROCS" --admit-first --seeds 0 1 2 3 4 --strats $ST_F6 --out f6_reg_main.csv
step F6-reg-l8     m7h-f6-reg-l8     $P $C sweep --procs "$PROCS" --admit-first --seeds 0 1 2 3 4 --ssd-dev local8 --strats $ST_F6 --out f6_reg_local8.csv
step F6-reg-l8q16  m7h-f6-reg-l8q16  $P $C sweep --procs "$PROCS" --admit-first --qmax 16 --seeds 0 1 2 3 4 --ssd-dev local8 --strats $ST_F6 --out f6_reg_l8q16.csv
step F6-regcmp-main  m7h-f6-regcmp-main  $P $F6 regress --tag regcmp_main --new "$(rid F6-reg-main)/f6_reg_main.csv" --old "$(rid D1-admitfirst)/d1_sweep_admitfirst.csv" "$(rid D1-wm50)/d1_sweep_wm50.csv" "$(rid D1-ct)/d1_sweep_ct.csv"
step F6-regcmp-l8    m7h-f6-regcmp-l8    $P $F6 regress --tag regcmp_l8 --new "$(rid F6-reg-l8)/f6_reg_local8.csv" --old "$(rid D1-local8)/d1_sweep_local8.csv" "$(rid D1-ct-l8)/d1_sweep_ct_local8.csv"
step F6-regcmp-l8q16 m7h-f6-regcmp-l8q16 $P $F6 regress --tag regcmp_l8q16 --new "$(rid F6-reg-l8q16)/f6_reg_l8q16.csv" --old "$(rid D1-af-q16)/d1_sweep_af_qmax16.csv" --ssd-dev local8
step F6-sweep-main   m7h-f6-sweep-main   $P $C sweep --procs "$PROCS" --admit-first --seeds 5 6 7 8 9 10 11 12 13 14 --strats $ST_F6 --out f6_sweep_main.csv
step F6-sweep-l8     m7h-f6-sweep-l8     $P $C sweep --procs "$PROCS" --admit-first --seeds 5 6 7 8 9 10 11 12 13 14 --ssd-dev local8 --strats $ST_F6 --out f6_sweep_local8.csv
step F6-sweep-l8q16  m7h-f6-sweep-l8q16  $P $C sweep --procs "$PROCS" --admit-first --qmax 16 --seeds 5 6 7 8 9 10 11 12 13 14 --ssd-dev local8 --strats $ST_F6 --out f6_sweep_l8q16.csv
step F6-an-main-s5t  m7h-f6-an-main-s5t  $P $F6 verdict --k 6 --tag main   --cand S5T --twins $TW17_S5T --metrics median mean p90 --inp "$(rid F6-sweep-main)/f6_sweep_main.csv"
step F6-an-main-s1d  m7h-f6-an-main-s1d  $P $F6 verdict --k 6 --tag main   --cand S1D --twins $TW16_S1D --metrics median mean p90 --inp "$(rid F6-sweep-main)/f6_sweep_main.csv"
step F6-an-l8-s5t    m7h-f6-an-l8-s5t    $P $F6 verdict --k 6 --tag local8 --cand S5T --twins $TW17_S5T --metrics median mean p90 --inp "$(rid F6-sweep-l8)/f6_sweep_local8.csv"
step F6-an-l8-s1d    m7h-f6-an-l8-s1d    $P $F6 verdict --k 6 --tag local8 --cand S1D --twins $TW16_S1D --metrics median mean p90 --inp "$(rid F6-sweep-l8)/f6_sweep_local8.csv"
step F6-an-l8q16-s5t m7h-f6-an-l8q16-s5t $P $F6 verdict --k 6 --tag l8q16  --cand S5T --twins $TW17_S5T --metrics median mean p90 --inp "$(rid F6-sweep-l8q16)/f6_sweep_l8q16.csv"
step F6-an-l8q16-s1d m7h-f6-an-l8q16-s1d $P $F6 verdict --k 6 --tag l8q16  --cand S1D --twins $TW16_S1D --metrics median mean p90 --inp "$(rid F6-sweep-l8q16)/f6_sweep_l8q16.csv"
# (c)：主判定用 H200 自己的 D1 過載格（d1_afq16_S1D_cfg.csv 的 pass10：3 格，run D1-an-afq16-s1d）；
# MI300X 的 5 格另外當對照（使用者 10/16：「兩組都報」）。--d1-overload-expect 3 是 H200 D1 表的格數，只用來核對讀到的檔
step F6-summary      m7h-f6-summary      $P $F6 summary --d1-overload-expect 3 \
  --extra-overload-ref "$REPO/results/m8_directions/d1_afq16_S1D_cfg.csv" --extra-overload-out f6_c_overload_mi300x_cells.csv

# ===== F7：f 縮放（MI300X 20261012-084917／084936）=====
step F7-run    m7h-f7-fscale-sim  $P $F7 run --scales 1.0 0.5 0.6
step F7-judge  m7h-f7-judge       $P $F7 judge
fi

if [ "$PART" = d4d5 ]; then
# ===== D4：真實 trace 的重用（MI300X 20261010-0542xx–0606xx）、D5：架構與模態（055012）=====
# 資料：code/h200_dl_cpu_data.sh 的 run（traces、scbench 放 $TIARA_DATASETS；D5 的 config 在下載 run 目錄的 cfg/，和 MI300X 一樣）。
# tokenizer（SCBench 的 token 數）：M7_LLAMA_GLOB（和 MI300X 同一個 revision）
dlrun() {   # dlrun <短名>：最後一次成功的下載 run 目錄
  if [ "$DRY" = 1 ]; then echo "\$TIARA_RUNS/<$1>"; return 0; fi
  local d; for d in $(ls -td "$TIARA_RUNS"/*-"$1" 2>/dev/null); do [ "$(cat "$d/exit_code" 2>/dev/null)" = 0 ] && { echo "$d"; return 0; }; done
  echo "[r_h200_cpu] 找不到成功的下載 run *-$1（先跑 code/h200_dl_cpu_data.sh）" >&2; echo "MISSING-$1"; return 1
}
TO="$P -I code/m8_trace_oracle.py"
step D4-char        m7h-d4-char       $TO char
step D4-char2       m7h-d4-char2      $TO char2
step D4-sim         m7h-d4-sim        $TO sim --cake
step D4-scbench     m7h-d4-scbench    $TO scbench --max-tok 1000
# m8_arch_arith 檢查 calib_c1.csv 的 run_id：H200 的是 R2 的 C1（run 20261016-061311-m7h-c1，job 20417）
step D5-arith       m7h-d5-arith      env M7_CALIB_RUN=20261016-061311-m7h-c1 $P code/m8_arch_arith.py --cfg-vl "$(dlrun h200-dl-d5-qwen3vl-cfg)/cfg/config.json" --cfg "$(dlrun h200-dl-d5-cfgs)/cfg"
step D4-sim2        m7h-d4-sim2       $TO sim --cake --posthoc --gpu-blocks 0 2400 --out d4_sim2.csv
step D4-gap         m7h-d4-gap        $TO gap --sim d4_sim.csv --src-run "$(basename "$(rid D4-sim)")"
step D4-sim3        m7h-d4-sim3       $TO sim --cake --posthoc --cpu-frac 0.02 0.05 --gpu-blocks 0 --out d4_sim3.csv
step D4-gap2b       m7h-d4-gap2b      $TO gap --sim d4_sim2.csv --src-run "$(basename "$(rid D4-sim2)")" --out d4_gap_posthoc.csv --verdict-out d4_verdict_posthoc.csv --rule-label posthoc_sim2_KEV_local_G0_total_median6_not_decisive
step D4-gap3b       m7h-d4-gap3b      $TO gap --sim d4_sim3.csv --src-run "$(basename "$(rid D4-sim3)")" --out d4_gap_smallcap.csv --verdict-out d4_verdict_smallcap.csv --rule-label posthoc_sim3_cpu2and5pct_local_G0_total_median4_not_decisive
step D4-nocache     m7h-d4-nocache    $TO nocache
step D4-gap-prereg  m7h-d4-gap-prereg $TO gap --sim d4_sim.csv --src-run "$(basename "$(rid D4-sim)")" --deferred-set prereg --out d4_gap_prereg7.csv --verdict-out d4_verdict_prereg7.csv --rule-label D4_sec2.3_but_deferred_set_restricted_to_7_policies_listed_in_sec2.2
fi

if [ "$PART" = d6 ]; then
# ===== D6：精度的算術（MI300X 20261010-055349-d6-arith、055854-d6-thresh）=====
# 使用者 10/16：「D6 不要 3090，都 H200，但要跑實驗」→ MI300X 的 --dmeas fp8=0.087,int4=0.814 是 3090 的 warm TTFT 差（F3 §4 查過），
# 這裡換成 H200 的 F3 工作 A（job 20614，反量化一個 64 MiB chunk 的 GPU 時間）。照 F3 的判準取值：cold、輸出正確的實作裡最快的那個的中位數。
# 精度對應 D6 的位元組模型（m8_precision_arith BYTES）：fp8＝fp8_tensor（32 MiB，整個 tensor 一個 scale）、
# int8＝int8_ptk（33 MiB）、int4＝int4_sym（17 MiB，每 (token, head) 一個 scale）。int8 3090 沒量過、MI300X 的 D6 沒給；H200 量到了就給。
F3CSV=$M9_OUT/f3_dequant.csv
if [ "$DRY" = 0 ]; then
  read -r DMEAS DSRC < <(python - "$F3CSV" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["cache_state"] == "cold" and r["correct"] == "True"]
pick = {"fp8": "fp8_tensor", "int8": "int8_ptk", "int4": "int4_sym"}
out = []
for k, prec in pick.items():
    c = [r for r in rows if r["precision"] == prec]
    assert c, f"{prec} 沒有輸出正確的 cold 列"
    b = min(c, key=lambda r: float(r["median_ms"]))
    out.append(f"{k}={float(b['median_ms']):g}")
    print(f"[d6] {k} ← {prec}/{b['impl']} cold median {b['median_ms']} ms", file=sys.stderr)
print(",".join(out), rows[0]["run_id"] + "_f3A_cold_fastest_correct_median_ms_per_chunk")
PY
) || { echo "[r_h200_cpu] 讀不到 $F3CSV 的反量化成本 → 停" >&2; exit 2; }
else
  DMEAS="<f3_dequant.csv>"; DSRC="<run>"
fi
step D6-arith   m7h-d6-arith  $P code/m8_precision_arith.py --dmeas "$DMEAS" --dmeas-src "$DSRC"
step D6-thresh  m7h-d6-thresh $P code/m8_precision_arith.py --threshold
fi

if [ "$PART" = b ]; then
# ===== 08 消融、第一階段 B 的模擬驗證、第 1 輪的分析（MI300X RUNLOG M7b、M7c）=====
# 輸入：r-h200-rest（job 20554）的 P1、P2、P3，r-h200-qwen3（job 20556）的 C1，--part a 的 sim_sweep、r1_sim_*。
# 這一批用 20554 的原始資料：P1 八個和 P2-G1-seed0 和我們自己的 F2 job 同時跑過（記憶 h200-no-self-cotenancy-for-timing）。
# 重跑（job 20631）的版本之後另外做，兩組都報。
# MI300X 只有 m7_sim validate 走過 §4.1 的殼（20261009-064908-m7-sim-validate），其他分析照各程式檔頭的用法。
step ABL-gain     m7h-abl-gain      $P code/m7_ablate_analyze.py          # sim_sweep.csv → sim_sweep_gain.csv
step SIM-validate m7h-sim-validate  $P $SIM validate                      # b_share／b_fifo／b_share_cpu3.69 → sim_validate.csv、.json
step B2-analyze   m7h-b2-analyze    $P code/m7_b2_analyze.py              # b2_share.csv → b2_summary.csv、b2_verdict.csv
step R1-h1        m7h-r1-h1         $P code/m7_round1_analyze.py h1       # → $M7_EXPLORE/h1_write_volume.csv、h1_verdict.csv
step R1-kappa     m7h-r1-kappa      $P code/m7_round1_analyze.py kappa    # 四個 f(i)（含 Qwen3 grouped／eager）→ kappa_screen.csv
step R1-sim       m7h-r1-sim-an     $P code/m7_round1_analyze.py sim      # r1_sim_*.csv → r1_sim_verdict.csv、r1_sim_configs.csv
step R1-gpu       m7h-r1-gpu-an     $P code/m7_round1_analyze.py gpu      # P3 的 r1_gpu_llama_doc_free_local.csv → r1_gpu_summary.csv、r1_gpu_verdict.csv
fi

if [ "$PART" = brr ]; then
# ===== B 重跑的合併組（使用者 10/16：「照重疊重跑」；新舊兩組都報）=====
# results/m7_write_policy_h200_merged/＝沒有和我們自己的 F2 job 重疊的 B：舊的 R3-B-main＋job 20631 的 9 個 -rr run＋舊的 b2 G1-s1、G2（code/h200_brr.py 檔頭）。
# 分析和 --part b 同一套指令，只把 M7_RES 指到合併目錄；f(i) 用同一份 H200 C1（M7_F_CSV）；H1 的輸出也寫合併目錄（M7_EXPLORE），不覆寫 m7_explore_h200 的。
MG=$REPO/results/m7_write_policy_h200_merged
FC=$M7_RES/calib_c1.csv
step BRR-merge        m7h-brr-merge         $P code/h200_brr.py merge
step BRR-sim-validate m7h-brr-sim-validate  env M7_RES="$MG" M7_F_CSV="$FC" $P $SIM validate
step BRR-b2           m7h-brr-b2-analyze    env M7_RES="$MG" M7_F_CSV="$FC" $P code/m7_b2_analyze.py
step BRR-h1           m7h-brr-r1-h1         env M7_RES="$MG" M7_F_CSV="$FC" M7_EXPLORE="$MG" $P code/m7_round1_analyze.py h1
step BRR-ball         m7h-brr-b-all         $P code/h200_brr.py ball --old "$M7_RES" --new "$MG"
step BRR-side         m7h-brr-side          $P code/h200_brr.py side --old "$M7_RES" --new "$MG"
fi

if [ "$PART" = a2c7 ]; then
# ===== A2、C7 錨點：m7_analyze 的 calib()、a2()（MI300X 10/08 同一套），H200 只有限速層 → 真實裝置 NOT_MEASURED =====
# 先回歸：m7_analyze 改成「沒有 real 列也能算」之後，MI300X 的 summary_a2、summary_c7_anchor、verdict 要和 git 相同
step A2C7-regress  m7h-a2c7-mi300x-regress  $P code/h200_a2c7.py regress
step A2C7-run      m7h-a2c7-an              $P code/h200_a2c7.py run
fi

if [ "$PART" = d3 ]; then
# ===== D3 κ 地圖（MI300X 20261010-054706-d3-kappa-map、054955-d3-r1-b100、054753-d3-gputest-prereg）=====
# 使用者 10/16：只有 H200 平台 → --no-3090；c7_anchor 只有限速層（job 20643）→ real 兩列 NOT_MEASURED（m8_kappa_map 檔頭）。
# --prereg-test 的 D*＝同一組的主程式擬合的（stdout 最後一行「D*=…」）。
# 兩組都報（使用者：「照重疊重跑」）：(1) 舊 B＝job 20554 的原始資料（含和 F2 重疊的 run）→ $M8_OUT；
# (2) 合併組（results/m7_write_policy_h200_merged，--part brr）→ $M8_OUT/d3_merged；輸入目錄＝D3M-stage 的 run 目錄 res/（舊目錄＋合併組的 B 檔）
KM=code/m8_kappa_map.py
dstar() {   # dstar <標籤>：那一步主程式印的 D*
  if [ "$DRY" = 1 ]; then echo "<$1 的 D*>"; return 0; fi
  local v; v=$(grep -oP 'D\*=\K[0-9.]+' "$(rid "$1")/stdout.log" | tail -1)
  [ -n "$v" ] || { echo "[r_h200_cpu] $1 的 stdout 沒有 D*" >&2; echo "MISSING-DSTAR"; return 1; }
  echo "$v"
}
step D3-main     m7h-d3-kappa-map       $P $KM --out "$M8_OUT" --no-3090
step D3-b100     m7h-d3-r1-b100         $P $KM --out "$M8_OUT" --no-3090 --b100
step D3-prereg   m7h-d3-gputest-prereg  $P $KM --out "$M8_OUT" --no-3090 --prereg-test --dstar "$(dstar D3-main)"
step D3M-stage   m7h-d3m-stage          $P code/h200_brr.py stage --old "$M7_RES" --new "$REPO/results/m7_write_policy_h200_merged"
D3M_RES="$(rid D3M-stage)/res"
step D3M-main    m7h-d3m-kappa-map      env M7_RES="$D3M_RES" $P $KM --out "$M8_OUT/d3_merged" --no-3090
step D3M-b100    m7h-d3m-r1-b100        env M7_RES="$D3M_RES" $P $KM --out "$M8_OUT/d3_merged" --no-3090 --b100
step D3M-prereg  m7h-d3m-gputest-prereg env M7_RES="$D3M_RES" $P $KM --out "$M8_OUT/d3_merged" --no-3090 --prereg-test --dstar "$(dstar D3M-main)"
fi

if [ "$PART" = r1h ]; then
# ===== 第 1 輪：H200 自己的 3 個模擬存活設定上 GPU（job 20660，code/r_h200_r1h.sh）=====
# 格與事先選定的策略、判準：RUNLOG_H200「第 1 輪：H200 自己的模擬存活設定上 GPU」（job 送出之前寫死）。
# --cpu 是模擬那一格的 cpu_gibps（取一位小數比對）：B、C 的 35.42＝tier_params 的 CPU 層（模擬 35.4159）。
G1=code/m7_round1_analyze.py
step R1H-A  m7h-r1h-gpu-an-A  $P $G1 gpu --csv r1h_gpu_llama_doc_hold_nfs_c11.6_f0.5.csv    --cpu 11.6  --release hold --ssd nfs   --frac 0.5  --wt S5L --tag A
step R1H-B  m7h-r1h-gpu-an-B  $P $G1 gpu --csv r1h_gpu_llama_doc_free_local_c35.42_f0.5.csv --cpu 35.42 --release free --ssd local --frac 0.5  --wt S5L --tag B
step R1H-C  m7h-r1h-gpu-an-C  $P $G1 gpu --csv r1h_gpu_llama_doc_hold_nfs_c35.42_f0.25.csv --cpu 35.42 --release hold --ssd nfs   --frac 0.25 --wt S5c --tag C
fi

if [ -n "$UNTIL" ] && [ "$DONE" = 0 ] && [ "$STARTED" = 1 ]; then
  echo "[r_h200_cpu] 沒有叫做 '$UNTIL' 的步驟（--until）。可用的：$SEEN" >&2; exit 2
fi
if [ "$STARTED" = 0 ]; then
  echo "[r_h200_cpu] 沒有叫做 '$FROM' 的步驟（--part $PART）。可用的：$SEEN" >&2; exit 2
fi
[ "$DRY" = 0 ] && echo "[r_h200_cpu] $(date -Is) part=$PART 全部完成"
exit 0
