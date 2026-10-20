#!/bin/bash
#SBATCH --job-name=m10h-rs2
#SBATCH --partition=h200q
#SBATCH --account=ChihChungHsu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=48:00:00
#SBATCH -o slurm-%x-%j.out
# =============================================================================
# r_h200_rs2.sh — MI300X 的 13 號（還原第二輪，13_restore2_plan.md）與 14 號（載入／計算配對、丟 block，14_match_drop_plan.md）
#                 在 H200 上照跑（事先登記：13_h200_repro.md §12）
#
#   sbatch --job-name=m10h-rs2 --time=30:00:00 code/r_h200_rs2.sh --part rs2   # 量時間：43a 43b 43c 43d 43e0 43e 44cz 44d
#   sbatch --job-name=m10h-md  --time=16:00:00 code/r_h200_rs2.sh --part md    # 冒煙＋只量分數：44e0 44e1 44e1z 44e2
#   sbatch --job-name=m10h-mdt --time=16:00:00 code/r_h200_rs2.sh --part mdt   # 量時間：44e3 44e4 44e5 44ez
#   sbatch --job-name=m10h-attnb --time=4:00:00 code/r_h200_rs2.sh --part attnb # 量時間：43e0nv 43eb（H200 才有，13_h200_repro §13）
#   bash code/r_h200_rs2.sh --part <p> --list | --dry-run ；--from <stage> 接著跑
#
# 每個 stage＝MI300X 的佇列腳本（code/m10_queue_mi300x/queue/<stage>.sh）經 code/h200_port_stage.py 轉成 H200 版：
#   只換路徑、run 名稱（m10h-）、vLLM 的 python（Qwen 用拿掉 DCA 的變體）、43d 讀 H200 自己的 43 腳本、43b 背景強度用 H200 的 duty。
#   指令、題數、設定不動。MI300X 的腳本自己處理失敗（寫 NEEDS_HUMAN、跳過那段、exit 0；RUNLOG_MI300X 14.5），這裡照用；
#   腳本本身 exit ≠ 0 才停（規則 2）。43a 冒煙沒過時，它會把用到那段程式的 stage 從 queue/ 移到 hold/，這裡就跳過。
# H200 才有的 stage（code/h200_queue/<stage>.sh，MI300X 沒有對應）不經轉換，只把 __JOBS__ 換成工作目錄。
# 共用的工作目錄 $TIARA_RUNS/_m10h_work/rs2（lib.sh、queue/、hold/、NEEDS_HUMAN、smoke_md/ 的旗子）：三個 part 之間靠它傳。
# 量時間的 part（rs2、mdt）不和我們自己的其他 GPU job 同時跑（送出時用 dependency 排開）。
# sbatch 送出時會複製一份；開跑時改執行 repo 裡的版本（R_H200_RS2_REEXEC）。
# =============================================================================
set -uo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
if [ -n "${REPO:-}" ] && [ -f "$REPO/code/env_h200.sh" ]; then :
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/code/env_h200.sh" ]; then REPO=$SLURM_SUBMIT_DIR
elif [ -f "$_here/env_h200.sh" ]; then REPO=$(cd "$_here/.." && pwd)
else echo "[r_h200_rs2] 找不到 repo（code/env_h200.sh）" >&2; exit 2
fi
export REPO
if [ -z "${R_H200_RS2_REEXEC:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  export R_H200_RS2_REEXEC=1
  echo "[r_h200_rs2] 改執行 repo 裡的版本：$REPO/code/r_h200_rs2.sh（git $(git -C "$REPO" rev-parse --short HEAD)）"
  exec bash "$REPO/code/r_h200_rs2.sh" "$@"
fi

PART=""; DRY=0; LIST=0; FROM=""
while [ $# -gt 0 ]; do
  case "$1" in
    --part) PART="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --list) LIST=1; DRY=1; shift ;;
    --from) FROM="$2"; shift 2 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "不認得的參數：$1" >&2; exit 2 ;;
  esac
done
case "$PART" in
  rs2) STAGES="43a_rs2_smoke 43b_rs2_busy 43c_rs2_g5 43d_rs2_newfmt 43e0_rs2_attn_resmoke 43e_rs2_attn 44cz_rs2_plan44d 44d_rs_longctx" ;;
  md)  STAGES="44e0_md_smoke 44e1_ds1_screen 44e1z_ds_pick 44e2_ds2_confirm" ;;
  mdt) STAGES="44e3_d3_speed 44e4_m2_fmt 44e5_m1_unit 44ez_md_report" ;;
  attnb) STAGES="43e0nv_attn_selftest_bm64 43eb_attn_b" ;;
  *) echo "要 --part rs2|md|mdt|attnb" >&2; exit 2 ;;
esac

[ "$DRY" = 1 ] && export TIARA_SKIP_CONDA=1
# shellcheck disable=SC1091
source "$REPO/code/env_h200.sh"
cd "$REPO"
ulimit -c 0

OUT=$REPO/results/m10_precision_h200
SW=$REPO/results/m10_sweep_h200
P=$REPO/results/m10_pilot_mi300x/e0_tune/CANDIDATE_PARAMS.json
BUSY=$REPO/results/m7_write_policy_h200/busy_levels.json
NODCA=$HOME/models/Qwen2.5-7B-Instruct-1M-noDCA                         # 13_h200_repro §11.1
C43=$TIARA_RUNS/_m10h_work/x-t2/queue/43_sweep_confirm.sh                # t2 的 CONFIRM-PLAN 寫的（H200 自己的選手）
JOBS=$TIARA_RUNS/_m10h_work/rs2                                          # 三個 part 共用（旗子、hold、NEEDS_HUMAN）
SRCQ=$REPO/code/m10_queue_mi300x/queue

lib() {           # MI300X 的 stage 都 source <JOBS>/lib.sh：給它 H200 的版本
  mkdir -p "$JOBS/queue" "$JOBS/hold" "$JOBS/log"
  cat > "$JOBS/lib.sh" <<EOF
source $REPO/code/env_h200.sh
REPO=$REPO; JOBS=$JOBS; OUT=$OUT; P=$P
export M10_BUSY_LEVELS=$BUSY M10_SWEEP_DIR=$SW M10_PRECISION_DIR=$OUT M10_RESTORE2_DIR=$REPO/results/m10_restore2_h200
fmt_of() { python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['choice'][sys.argv[2]][sys.argv[3]])" "$OUT/fmt_select/fmt_choice.json" "\$1" "\$2"; }
lastrun() { ls -d "\$TIARA_RUNS"/*-"\$1" | sort | tail -1 | xargs basename; }
pool() { (cd $JOBS && python $REPO/code/h200_pool.py "\$@"); }
EOF
}
H2Q=$REPO/code/h200_queue
port() {          # port <來源> <目的>；H200 才有的 stage 只換 __JOBS__
  if [ "$(dirname "$1")" = "$H2Q" ]; then sed "s#__JOBS__#$JOBS#g" "$1" > "$2"; return; fi
  python3 "$REPO/code/h200_port_stage.py" "$1" "$2" --repo "$REPO" --jobs "$JOBS" --raw-root "$TIARA_HOME" --nodca "$NODCA" --confirm43 "$C43"
}

if [ "$DRY" = 0 ]; then
  [ -n "${SLURM_JOB_ID:-}" ] || { echo "[r_h200_rs2] 不在 Slurm job 裡" >&2; exit 2; }
  python - "$OUT/fmt_select/fmt_vs_mi300x.json" <<'PY' || { echo "[r_h200_rs2] 格式還沒選或和 MI300X 不同（13 §3-4）→ 停" >&2; exit 2; }
import json, sys
sys.exit(0 if json.load(open(sys.argv[1]))["same"] else 1)
PY
  [ -f "$NODCA/config.json" ] && ! grep -q dual_chunk_attention_config "$NODCA/config.json" || { echo "[r_h200_rs2] 找不到 $NODCA 或 config 還有 DCA" >&2; exit 2; }
  [ -f "$C43" ] || { echo "[r_h200_rs2] 找不到 H200 的 43 腳本 $C43" >&2; exit 2; }
  lib
  echo "[r_h200_rs2] $(date -Is) part=$PART job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-?} git=$(git rev-parse --short HEAD) JOBS=$JOBS"
  nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.used --format=csv
  # 先把這個 part 的 stage 全部轉好放進 queue/（43a 的 hold() 要在 queue/ 裡找得到後面的 stage）；已經在 hold/ 的不放回來
  for st in $STAGES; do
    [ "$st" = 44d_rs_longctx ] && continue                         # 44cz 才寫 44d
    [ -f "$JOBS/hold/$st.sh" ] && { echo "[r_h200_rs2] $st 在 hold/（之前的冒煙沒過）→ 不放回 queue/"; continue; }
    src=$SRCQ/$st.sh; [ -f "$H2Q/$st.sh" ] && src=$H2Q/$st.sh
    port "$src" "$JOBS/queue/$st.sh" || { echo "[r_h200_rs2] $st 轉換失敗 → 停" >&2; exit 2; }
  done
fi

STARTED=1; [ -n "$FROM" ] && STARTED=0
SEEN=""
for st in $STAGES; do
  SEEN="$SEEN $st"
  if [ "$STARTED" = 0 ]; then
    if [ "$st" = "$FROM" ]; then STARTED=1; else [ "$LIST" = 0 ] && echo "[skip] $st"; continue; fi
  fi
  if [ "$LIST" = 1 ]; then echo "$st"; continue; fi
  if [ "$DRY" = 1 ]; then src=$SRCQ/$st.sh; [ -f "$H2Q/$st.sh" ] && src=$H2Q/$st.sh; echo "[$st] port $src → \$JOBS/queue/$st.sh；bash \$JOBS/queue/$st.sh"; continue; fi
  f=$JOBS/queue/$st.sh
  if [ "$st" = 44d_rs_longctx ]; then                                # 44cz（plan-44d）寫的是 MI300X 的範本 → 轉成 H200 版
    [ -f "$f" ] || { echo "=== $st：44cz 沒有寫出 44d（見 NEEDS_HUMAN）→ 跳過"; continue; }
    port "$f" "$JOBS/queue/$st.h200.sh" || { echo "[r_h200_rs2] 44d 轉換失敗 → 停" >&2; exit 2; }
    f=$JOBS/queue/$st.h200.sh
  fi
  [ -f "$f" ] || { echo "=== $st 不在 queue/（在 hold/：冒煙沒過）→ 跳過"; continue; }
  nh0=$(wc -l < "$JOBS/NEEDS_HUMAN" 2>/dev/null || echo 0)
  echo "=== $(date -Is) $st"
  bash "$f" > "$JOBS/log/$st.log" 2>&1; rc=$?
  tail -5 "$JOBS/log/$st.log"
  nh1=$(wc -l < "$JOBS/NEEDS_HUMAN" 2>/dev/null || echo 0)
  [ "$nh1" -gt "$nh0" ] && { echo "--- $st 新的 NEEDS_HUMAN："; tail -n $((nh1 - nh0)) "$JOBS/NEEDS_HUMAN"; }
  if [ $rc -ne 0 ]; then
    echo "=== $(date -Is) $st 失敗（exit $rc；$JOBS/log/$st.log）→ 停下、回報（規則 2）。接著跑：sbatch ... code/r_h200_rs2.sh --part $PART --from $st" >&2
    exit $rc
  fi
  mv "$f" "$JOBS/queue/done_$(basename "$f")"
done
if [ "$STARTED" = 0 ]; then echo "[r_h200_rs2] 沒有叫做 '$FROM' 的 stage。可用的：$SEEN" >&2; exit 2; fi
if [ "$DRY" = 0 ]; then
  # 量時間的結果：gpu_guard 不是 CLEAN 的列出來（pick／report 會自己排除；照 CLAUDE.md §3 要重量的另外決定）
  python3 - "$REPO/results/m10_restore2_h200" "$REPO/results/m10_match_h200" <<'PY'
import glob, json, sys
bad, n = [], 0
for root in sys.argv[1:]:
    for f in glob.glob(f"{root}/**/gpu_guard_*.json", recursive=True):
        n += 1
        d = json.load(open(f))
        if d.get("verdict") != "CLEAN" or d.get("contaminated"):
            bad.append(f)
print(f"[guard] 目前 {n} 份 gpu_guard JSON，不是 CLEAN 的 {len(bad)} 份：{bad[:10]}")
PY
  echo "[r_h200_rs2] $(date -Is) part=$PART 跑完；NEEDS_HUMAN：$JOBS/NEEDS_HUMAN；記進 RUNLOG_H200"
fi
exit 0
