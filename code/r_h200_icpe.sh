#!/bin/bash
#SBATCH --job-name=icpe
#SBATCH --partition=h200q
#SBATCH --account=ChihChungHsu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=12:00:00
#SBATCH -o slurm-%x-%j.out
# =============================================================================
# r_h200_icpe.sh — ICPE 2027 的 H200 補量（docs/h200/icpe/）。每一步一個 run 目錄（m7run）。
#
#   sbatch --job-name=icpe-p01s1 --time=3:00:00 code/r_h200_icpe.sh --part p01s1   # 量時間：S1 attention 比較＋勝者的校準（P0-1 §2）
#   sbatch --job-name=icpe-p01ab --time=6:00:00 code/r_h200_icpe.sh --part p01ab   # 量時間：(a) B 主設定、(b) 還原格子（P0-1 §3.1–3.2）
#   bash code/r_h200_icpe.sh --part <p> --list | --dry-run ；--from <步驟> 接著跑
#
# 量時間的 part 開跑前檢查：節點上有我們自己的別的 job（同一個 Slurm 使用者，含 DL_Lab4）→ 退出（memory h200-no-self-cotenancy-for-timing；
# 使用者 10/22：DL_Lab4 也算自己的，排隊中的先 hold）。sbatch 送出時會複製一份；開跑時改執行 repo 裡的版本（R_H200_ICPE_REEXEC）。
# 這支在 job 跑的時候不要原地改（bash 邊跑邊讀）：寫新檔再 mv 過去。
# =============================================================================
set -uo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
if [ -n "${REPO:-}" ] && [ -f "$REPO/code/env_h200.sh" ]; then :
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/code/env_h200.sh" ]; then REPO=$SLURM_SUBMIT_DIR
elif [ -f "$_here/env_h200.sh" ]; then REPO=$(cd "$_here/.." && pwd)
else echo "[r_h200_icpe] 找不到 repo（code/env_h200.sh）" >&2; exit 2
fi
export REPO
if [ -z "${R_H200_ICPE_REEXEC:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  export R_H200_ICPE_REEXEC=1
  echo "[r_h200_icpe] 改執行 repo 裡的版本：$REPO/code/r_h200_icpe.sh（git $(git -C "$REPO" rev-parse --short HEAD)）"
  exec bash "$REPO/code/r_h200_icpe.sh" "$@"
fi
PART=""; DRY=0; LIST=0; FROM=""
while [ $# -gt 0 ]; do
  case "$1" in
    --part) PART="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --list) LIST=1; DRY=1; shift ;;
    --from) FROM="$2"; shift 2 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "不認得的參數：$1" >&2; exit 2 ;;
  esac
done
case "$PART" in
  p01s1|p01ab) TIMING=1 ;;
  *) echo "要 --part p01s1|p01ab" >&2; exit 2 ;;
esac
[ "$DRY" = 1 ] && export TIARA_SKIP_CONDA=1
# shellcheck disable=SC1091
source "$REPO/code/env_h200.sh"
cd "$REPO"
ulimit -c 0
P01=$REPO/results/icpe_h200/p01
export M10_SWEEP_DIR=$REPO/results/m10_sweep_h200 M10_PRECISION_DIR=$REPO/results/m10_precision_h200
export M10_BUSY_LEVELS=$REPO/results/m7_write_policy_h200/busy_levels.json

if [ "$DRY" = 0 ]; then
  [ -n "${SLURM_JOB_ID:-}" ] || { echo "[r_h200_icpe] 不在 Slurm job 裡" >&2; exit 2; }
  python -c 'import sys; sys.path.insert(0, "code"); from gpu_guard import smi_index; print("[r_h200_icpe] smi_index(0) =", smi_index(0))' \
    || { echo "[r_h200_icpe] cuda:0 對不到 nvidia-smi 的卡 → 停" >&2; exit 2; }
  if [ "$TIMING" = 1 ]; then
    others=$(squeue -h -w "${SLURMD_NODENAME:-$(hostname -s)}" -u "$USER" -t R -o '%i %j' | awk -v me="$SLURM_JOB_ID" '$1!=me')
    if [ -n "$others" ]; then
      echo "[r_h200_icpe] 節點上還有我們自己的 job 在跑 → 停（量時間不和自己人同時跑）：" >&2; echo "$others" >&2; exit 2
    fi
  fi
  echo "[r_h200_icpe] $(date -Is) part=$PART job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-?} git=$(git rev-parse --short HEAD)"
  echo "[r_h200_icpe] 同節點的 job（開跑時）："; squeue -h -w "${SLURMD_NODENAME:-$(hostname -s)}" -o '%i %u %a %b %T %M'
  nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.used --format=csv
  mkdir -p "$P01/attn" "$P01/calib" "$P01/b" "$P01/grid"
fi

STARTED=1; [ -n "$FROM" ] && STARTED=0
SEEN=""
step() {
  local name=$1; shift
  SEEN="$SEEN $name"
  if [ "$STARTED" = 0 ]; then if [ "$name" = "$FROM" ]; then STARTED=1; else return 0; fi; fi
  if [ "$LIST" = 1 ]; then echo "$name"; return 0; fi
  if [ "$DRY" = 1 ]; then printf '[%s] ' "$name"; printf '%q ' "$@"; echo; return 0; fi
  echo "=== $(date -Is) $name"
  "$@"; local rc=$?
  if [ $rc -ne 0 ]; then echo "=== $(date -Is) $name 失敗（exit $rc）→ 停下、回報（規則 2）。接著跑：sbatch ... code/r_h200_icpe.sh --part $PART --from $name" >&2; exit $rc; fi
}
# S1 的勝者（P0-1 §2.3）：STOP＝停損（沒有比 sdpa 快 ≥5%）；dry-run 時還沒有 choice.json
win() {
  [ "$DRY" = 1 ] && { echo "<勝者:$1>"; return; }
  python3 -c "import json,sys; d=json.load(open(sys.argv[1]))['models'][sys.argv[2]]; print('STOP' if d['stop_loss'] else d['winner'])" "$P01/attn/choice.json" "$1"
}
fmt_of() { python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['choice'][sys.argv[2]][sys.argv[3]])" "$M10_PRECISION_DIR/fmt_select/fmt_choice.json" "$1" "$2"; }
skip() { echo "=== $(date -Is) $1：$2 → 跳過"; }

# ============================================================== P0-1 S1（§2）
if [ "$PART" = p01s1 ]; then
  step P01-ATTN-LLAMA  m7run icpe-p01-attn-llama python code/icpe/attn_bench.py run --model llama --out-dir "$P01/attn"
  step P01-ATTN-QWEN   m7run icpe-p01-attn-qwen  python code/icpe/attn_bench.py run --model qwen  --out-dir "$P01/attn"
  step P01-ATTN-CHOOSE m7run icpe-p01-attn-choose python code/icpe/attn_bench.py choose --out-dir "$P01/attn"
  WL=$(win llama); WQ=$(win qwen)
  [ "$DRY" = 0 ] && echo "[r_h200_icpe] 勝者：llama=$WL qwen=$WQ"
  # §2.4：勝者的校準（停損的模型不做）。C5、C4 的輸出跟著 M7_RES（＝$P01/calib）
  if [ "$WL" != STOP ]; then
    step P01-C1        m7run icpe-p01-c1 env M7_ATTN="$WL" M7_RES="$P01/calib" python code/m7_calib.py c1 --out "$P01/calib/calib_c1.csv" --reps 3 --max-chunks 80
    step P01-CALIBF-LLAMA m7run icpe-p01-calibf-llama env M7_ATTN="$WL" bash -c "mkdir -p $P01/calib/speed_llama && python code/m10_speed.py calib-f --model llama --out-dir $P01/calib/speed_llama"
    step P01-C5        m7run icpe-p01-c5 env M7_ATTN="$WL" M7_RES="$P01/calib" python code/m7_write_policy.py busy-calib --duties 0.6 1.0
    step P01-C4-SAME   m7run icpe-p01-c4-same-t1 env M7_ATTN="$WL" M7_RES="$P01/calib" python code/m7_write_policy.py noise --reps 10 --slot t1 --proc-mode same
    for i in 0 1 2 3 4 5 6 7 8 9; do
      step P01-C4-RESTART-$i m7run icpe-p01-c4-restart-t1 env M7_ATTN="$WL" M7_RES="$P01/calib" python code/m7_write_policy.py noise --reps 1 --slot t1 --proc-mode restart
    done
  else
    skip "P01-C1／CALIBF-LLAMA／C5／C4" "Llama 停損（S1 沒有比 sdpa 快 ≥5% 的候選）"
  fi
  if [ "$WQ" != STOP ]; then
    step P01-CALIBF-QWEN m7run icpe-p01-calibf-qwen env M7_ATTN="$WQ" bash -c "mkdir -p $P01/calib/speed_qwen && python code/m10_speed.py calib-f --model qwen --out-dir $P01/calib/speed_qwen"
  else
    skip P01-CALIBF-QWEN "Qwen 停損"
  fi
fi

# ============================================================== P0-1 S2 (a)(b)（§3.1、§3.2＋§8 偏離 1）
if [ "$PART" = p01ab ]; then
  WL=$(win llama); WQ=$(win qwen)
  # (a) B 主設定：和 R3-B-main 同一條指令，只改 --reps 6 與環境（M7_ATTN、M7_F_CSV、M7_RES）
  if [ "$WL" != STOP ]; then
    step P01A-B m7run icpe-p01a-b-nfs-gap0 env M7_ATTN="$WL" M7_F_CSV="$P01/calib/calib_c1.csv" M7_RES="$P01/b" M7_IO_MODEL=share \
      python code/m7_write_policy.py b --reps 6 --ssd-dev nfs --gap 0 --wl-seed 0 --verify
  else
    skip P01A-B "Llama 停損"
  fi
  # (b) 還原格子：bw.json 複本加 nfs；sdpa 與勝者交錯（L500：sdpa、勝者；L501：sdpa、勝者）
  step P01B-BW m7run icpe-p01b-bw python code/icpe/p01_tools.py bw-nfs --out-dir "$P01/grid"
  for L in 500 501; do
    for m in llama qwen; do
      W=$WL; [ $m = qwen ] && W=$WQ
      [ "$W" = STOP ] && { skip "P01B-$m-L$L" "$m 停損"; continue; }
      F=$(fmt_of $m int4); G=$(fmt_of $m fp8)
      for arm in sdpa "$W"; do
        # 判定用：P3（S4，一次拆完）下的 Cake（claim）＋全部載入
        K4=$(python code/m10_dqsweep.py env S4 --model $m --fmt "$F")
        step "P01B-$m-S4-$arm-L$L" m7run "icpe-p01b-$m-s4-$arm" env $K4 M7_ATTN="$arm" bash -c \
          "python code/m10_speed.py grid --model $m --fmt $F --base-fmt $G --bw-json $P01/grid/bw_$m.json --configs E0 E1 --bw-names cpu b32 nfs --gpu-states idle busy --extra-modes loadall --refs --lengths --launches 1 --reps 5 --launch0 $L --out-dir $P01/grid/${m}_s4_$arm --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv"
        # 只報告：S0（逐塊拆）下的 G5b，同一個 run 另量 Cake 與全部載入
        K0=$(python code/m10_dqsweep.py env S0 --model $m --fmt "$F")
        step "P01B-$m-G5B-$arm-L$L" m7run "icpe-p01b-$m-g5b-$arm" env $K0 M7_ATTN="$arm" M10_MEET=g5b bash -c \
          "python code/m10_speed.py grid --model $m --fmt $F --base-fmt $G --bw-json $P01/grid/bw_$m.json --configs E0 E1 --bw-names cpu b32 nfs --gpu-states idle busy --extra-modes loadall claim --refs --lengths --launches 1 --reps 5 --launch0 $L --out-dir $P01/grid/${m}_g5b_$arm --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv"
      done
    done
  done
fi

if [ "$STARTED" = 0 ]; then echo "[r_h200_icpe] 沒有叫做 '$FROM' 的步驟。可用的：$SEEN" >&2; exit 2; fi
[ "$DRY" = 0 ] && echo "[r_h200_icpe] $(date -Is) part=$PART 全部完成"
exit 0
