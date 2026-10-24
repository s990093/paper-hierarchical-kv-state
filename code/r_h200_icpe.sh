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
#   sbatch --job-name=icpe-p01brr --time=0:30:00 code/r_h200_icpe.sh --part p01brr # 量時間：P0-1 (b) 和 G_swa 重疊的 2 個 run 照重疊重跑（qwen fa3 L501）
#   sbatch --job-name=icpe-p02   --time=8:00:00 code/r_h200_icpe.sh --part p02     # 量時間：P0-2 的 GPU 重播（冒煙＋12 個 run）
#   sbatch --job-name=icpe-cpu   --time=6:00:00 code/r_h200_icpe.sh --part cpu     # 只用 CPU：P0-1 (c) X1／X5 三個版本、P0-2 的全尺寸 HINT-PF 模擬
#   sbatch --job-name=icpe-p12   --time=6:00:00 code/r_h200_icpe.sh --part p12     # 只用 CPU：P1-2（X1 FI、X5 事後版本、X4 f×0.5）
#   sbatch --job-name=icpe-p2    --time=2:00:00 code/r_h200_icpe.sh --part p2      # 量時間：P2（G5b CPU 層忙，2 個新 seed）
#   sbatch --job-name=icpe-p11q  --time=8:00:00 code/r_h200_icpe.sh --part p11q    # 品質（不量時間，但不和量時間的 job 重疊）：P1-1 (Q)
#   sbatch --job-name=icpe-p11t  --time=3:00:00 code/r_h200_icpe.sh --part p11t    # 量時間：P1-1 (T)（全部載入、壓縮時間）
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
  p01s1|p01ab|p01brr|p02|p2|p11t) TIMING=1 ;;
  cpu|p12|p11q) TIMING=0 ;;
  *) echo "要 --part p01s1|p01ab|p01brr|p02|cpu|p12|p2|p11q|p11t" >&2; exit 2 ;;
esac
[ "$DRY" = 1 ] && export TIARA_SKIP_CONDA=1
# shellcheck disable=SC1091
source "$REPO/code/env_h200.sh"
cd "$REPO"
ulimit -c 0
P01=$REPO/results/icpe_h200/p01
P02=$REPO/results/icpe_h200/p02
# 3b 的模擬（X1、X5）與 P0-2：Mooncake trace 用 /work 上那份（sha256 和 X1 文件相同：conversation b8cbb061…、toolagent 48a2db1a…）
export P3_CONST=h200 X1_TRACE_DIR=$TIARA_HOME/datasets/traces X5_TRACE_DIR=$TIARA_HOME/datasets/traces
P3LOCAL=/tmp/hungweilai-p3/icpe                                          # 大的原始輸出（npz）放 hgx-cn01 本地，不進 /work 配額
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
  mkdir -p "$P01/attn" "$P01/calib" "$P01/b" "$P01/grid" "$P02" "$P3LOCAL"
  python3 - "$TIARA_HOME/datasets/traces" <<'PY' || { echo "[r_h200_icpe] Mooncake trace 的 sha256 不對 → 停" >&2; exit 2; }
import hashlib, sys
want = {"conversation_trace.jsonl": "b8cbb061a85206d7", "toolagent_trace.jsonl": "48a2db1a13d3bc05"}
for f, h in want.items():
    got = hashlib.sha256(open(f"{sys.argv[1]}/{f}", "rb").read()).hexdigest()
    print("[r_h200_icpe]", f, got[:16]); assert got.startswith(h), (f, got)
PY
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

# ============================================================== P0-1 (b) 照重疊重跑（10/22 23:41 起 G_swa 和 qwen fa3 L501 的兩步重疊；grid/EXCLUDED_RUNS.csv）
if [ "$PART" = p01brr ]; then
  m=qwen; L=501; arm=$(win qwen); F=$(fmt_of $m int4); G=$(fmt_of $m fp8)
  K4=$(python code/m10_dqsweep.py env S4 --model $m --fmt "$F"); K0=$(python code/m10_dqsweep.py env S0 --model $m --fmt "$F")
  step "P01BRR-$m-S4-$arm-L$L" m7run "icpe-p01b-$m-s4-$arm-rr" env $K4 M7_ATTN="$arm" bash -c \
    "python code/m10_speed.py grid --model $m --fmt $F --base-fmt $G --bw-json $P01/grid/bw_$m.json --configs E0 E1 --bw-names cpu b32 nfs --gpu-states idle busy --extra-modes loadall --refs --lengths --launches 1 --reps 5 --launch0 $L --out-dir $P01/grid/${m}_s4_$arm --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv"
  step "P01BRR-$m-G5B-$arm-L$L" m7run "icpe-p01b-$m-g5b-$arm-rr" env $K0 M7_ATTN="$arm" M10_MEET=g5b bash -c \
    "python code/m10_speed.py grid --model $m --fmt $F --base-fmt $G --bw-json $P01/grid/bw_$m.json --configs E0 E1 --bw-names cpu b32 nfs --gpu-states idle busy --extra-modes loadall claim --refs --lengths --launches 1 --reps 5 --launch0 $L --out-dir $P01/grid/${m}_g5b_$arm --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv"
fi

# ============================================================== P0-2 GPU 重播（§5–§7）
if [ "$PART" = p02 ]; then
  if [ "$DRY" = 1 ]; then KK="<k>"; else
    KK=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['chosen_k'])" "$P02/select.json") || { echo "沒有 $P02/select.json" >&2; exit 2; }
    python3 - "$P02/selftest.csv" <<'PY' || { echo "[r_h200_icpe] P0-2 的 CPU 自測沒有全過（§7）→ 停" >&2; exit 2; }
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
last = max(r["run_id"] for r in rows)
bad = [r["check"] for r in rows if r["run_id"] == last and r["passed"] != "True"]
print("[r_h200_icpe] selftest", last, "fail:", bad); sys.exit(1 if bad or not rows else 0)
PY
  fi
  step P02-SMOKE m7run icpe-p02-smoke python code/icpe/x1_gpu.py gpu --out "$P02" --k "$KK" --cpu 128 --policy LRU --max-req 100 --csv gpu_smoke.csv
  # 12 個 run 的順序（固定 seed 20261023 打亂；§6）
  ORDER=$(python3 - <<'PY'
import random
runs = [(c, p, 0) for c in (128, 256) for p in ("LRU", "FREQ2Q", "O-BEL", "HINT-PF", "O-PF")] + [(128, "LRU", 1), (256, "LRU", 1)]
random.Random(20261023).shuffle(runs)
for c, p, r in runs:
    print(c, p, r)
PY
)
  while read -r c pol rep; do
    step "P02-$c-$pol-r$rep" m7run "icpe-p02-$c-$pol-r$rep" python code/icpe/x1_gpu.py gpu --out "$P02" --k "$KK" --cpu "$c" --policy "$pol" --rep "$rep"
  done <<< "$ORDER"
fi

# ============================================================== 只用 CPU：P0-1 (c)（§3.3）＋P0-2 全尺寸 HINT-PF（只報告）
if [ "$PART" = cpu ]; then
  VS="V0 V2"
  if [ "$DRY" = 1 ] || { [ -f "$P01/calib/calib_c1.csv" ] && [ -f "$P01/calib/calib_c5.csv" ] && [ "$(win llama)" != STOP ]; }; then VS="V0 V1 V2"
  else skip "P0-1 (c) V1" "Llama 沒有 V1 的校準（S1 停損或還沒跑）"; fi
  for V in $VS; do
    FV="python code/icpe/p3_fvar.py --variant $V --calib $P01/calib --out-root $P01/sim_$V"
    step "P01C-$V-X1-SWEEP" m7run "icpe-p01c-x1-sweep-$V" env RUN_DIR="$P3LOCAL/x1sweep_$V" bash -c "mkdir -p $P3LOCAL/x1sweep_$V && $FV x1 sweep --workers 14"
    step "P01C-$V-X1-AN"    m7run "icpe-p01c-x1-an-$V" $FV x1 analyze --sweep x1_sweep.csv --traces conversation toolagent
    step "P01C-$V-X1-H1800" m7run "icpe-p01c-x1-h1800-$V" $FV x1 analyze --sweep x1_sweep.csv --traces conversation toolagent --dead-h 1800 --out-prefix x1h1800
    step "P01C-$V-X1-H3600" m7run "icpe-p01c-x1-h3600-$V" $FV x1 analyze --sweep x1_sweep.csv --traces conversation toolagent --dead-h 3600 --out-prefix x1h3600
    step "P01C-$V-X5-GRID"  m7run "icpe-p01c-x5-grid-$V" bash -c "rm -rf $P3LOCAL/x5_$V && mkdir -p $P3LOCAL/x5_$V $P01/sim_$V/x5 && $FV x5 --out-dir $P01/sim_$V/x5 --raw-dir $P3LOCAL/x5_$V --procs 14"
    step "P01C-$V-X5-AN"    m7run "icpe-p01c-x5-an-$V" $FV x5a --out-dir "$P01/sim_$V/x5" --raw-dir "$P3LOCAL/x5_$V"
  done
  step P02-FULLSIM m7run icpe-p02-fullsim python code/icpe/x1_gpu.py fullsim --out "$P02"
fi

# ============================================================== P1-2（只用 CPU；docs/h200/icpe/P1-2_cpu_symmetric_reruns.md §2）
if [ "$PART" = p12 ]; then
  P12=$REPO/results/icpe_h200/p12
  export TIARA_DATASETS=/tmp/hungweilai-p3/datasets X4_TRACE_DIR=$TIARA_HOME/datasets/traces    # FreeInference 在 hgx-cn01 本地（P3 移植下載的）
  P3PY=$TIARA_HOME/venvs/tiara-p3/bin/python
  step P12-VERIFY m7run icpe-p12-data-verify "$P3PY" -I code/h200_p3_verify_data.py "$TIARA_DATASETS" mooncake freeinference
  FV="python code/icpe/p3_fvar.py --variant V0 --out-root $P12"
  WEEKS="fi:2026-05-17_2026-05-23 fi:2026-05-24_2026-05-30 fi:2026-05-31_2026-06-06 fi:2026-06-07_2026-06-13 fi:2026-06-14_2026-06-20 fi:2026-06-21_2026-06-27"
  # A2：X1 的 FreeInference（前 6 週，只跑 Cake）
  step P12-A2-SWEEP   m7run icpe-p12-x1fi-sweep env RUN_DIR="$P3LOCAL/p12_x1fi" bash -c "mkdir -p $P3LOCAL/p12_x1fi && $FV x1 sweep --workers 14 --traces $WEEKS --rules session_id --modes cake --out x1_sweep_fi.csv"
  step P12-A2-AN      m7run icpe-p12-x1fi-an $FV x1 analyze --sweep x1_sweep_fi.csv --traces freeinference --out-prefix x1fi
  step P12-A2-AN-USER m7run icpe-p12-x1fiuser-an $FV x1 analyze --sweep x1_sweep_fi.csv --traces freeinference --metric mean_ret_user --out-prefix x1fiuser
  # B1–B4：X5 事後版本（同一個 out-dir，檔名用 sfx 分開）
  x5v() {   # x5v <短名> <raw 子目錄> <x5_run 參數...> -- <x5_analyze 參數...>
    local nm=$1 raw=$P3LOCAL/$2; shift 2
    local run=() an=(); while [ $# -gt 0 ] && [ "$1" != -- ]; do run+=("$1"); shift; done; shift; an=("$@")
    m7run "icpe-p12-x5-$nm" bash -c "rm -rf $raw && mkdir -p $raw $P12/x5 && $FV x5 --out-dir $P12/x5 --raw-dir $raw --procs 14 ${run[*]}" || return 1
    m7run "icpe-p12-x5-$nm-an" $FV x5a --out-dir "$P12/x5" --raw-dir "$raw" "${an[@]}"
  }
  step P12-B1-STRICT       x5v strict       p12_x5_strict       --rule strict --variants reg -- --sfx _strict --rule strict
  step P12-B2-FI           x5v fi           p12_x5_fi           --fi --variants reg -- --sfx _fi
  step P12-B3-CENS1800     x5v cens1800     p12_x5_cens1800     --censor-H 1800 --main-only -- --sfx _cens1800
  step P12-B3-STRICT-1800  x5v strict-cens1800 p12_x5_strict_cens1800 --rule strict --censor-H 1800 --main-only --variants reg -- --sfx _strict_cens1800 --rule strict
  step P12-B3-FI-1800      x5v fi-cens1800  p12_x5_fi_cens1800  --fi --censor-H 1800 --main-only --variants reg -- --sfx _fi_cens1800
  step P12-B3-FI-3600      x5v fi-cens3600  p12_x5_fi_cens3600  --fi --censor-H 3600 --main-only --variants reg -- --sfx _fi_cens3600
  step P12-B4-LOADONLY     x5v loadonly     p12_x5_loadonly     --variants loadonly --main-only -- --sfx _loadonly --variant loadonly
  # B5：只算 ≤32K（主格子的 raw＝P0-1 (c) V0 的 X5 格子，和 git 的 H200 主格子逐值相同才算數；B1 的 strict raw）
  step P12-B5-LE32K m7run icpe-p12-x5-le32k $FV x5s --out-dir "$P12/x5" --sets "subtree::$P3LOCAL/x5_V0" "strict:_strict:$P3LOCAL/p12_x5_strict"
  # C1：X4 f×0.5
  step P12-C1-SWEEP m7run icpe-p12-x4f05-sweep $FV x4 sweep --procs 14 --rules loose strict --fscale 0.5 --p 0.3 0.4 0.5 --out x4_sweep_f05.csv
  if [ "$DRY" = 1 ]; then F05="<x4f05-sweep 的 run 目錄>"; else F05=$(ls -d "$TIARA_RUNS"/*-icpe-p12-x4f05-sweep 2>/dev/null | sort | tail -1); fi
  step P12-C1-AN m7run icpe-p12-x4f05-an $FV x4 analyze --inp "$F05/x4_sweep_f05.csv" --out-seed x4_f05_verdict_seed.csv --out-cfg x4_f05_verdict_cfg.csv
fi

# ============================================================== P2（量時間；docs/h200/icpe/P2_g5b_cpu_busy_reseed.md §2）
if [ "$PART" = p2 ]; then
  P2D=$REPO/results/icpe_h200/p2
  for L in 310 311; do
    for m in llama qwen; do
      for tag in fp8 int4; do
        F=$(fmt_of $m $tag); G=$(fmt_of $m fp8)
        K0=$(python code/m10_dqsweep.py env S0 --model $m --fmt "$F")
        step "P2-$m-$tag-L$L" m7run "icpe-p2-$m-$tag-L$L" env $K0 M10_MEET=g5b bash -c \
          "python code/m10_speed.py grid --model $m --fmt $F --base-fmt $G --bw-json $M10_PRECISION_DIR/speed_$m/bw.json --configs E0 E1 R25 --refs --lengths --bw-names cpu --gpu-states busy --extra-modes loadall claim --launches 1 --reps 10 --launch0 $L --out-dir $P2D/${m}_${tag}_cpu_g5b_L$L --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv"
      done
    done
  done
fi

# ============================================================== P1-1 (Q)（docs/h200/icpe/P1-1_xkv_int4_h200.md §2）
if [ "$PART" = p11q ]; then
  P11=$REPO/results/icpe_h200/p11
  export X_DATA=$P3LOCAL/xstack                                             # 校準基底（每個約 150 MB）放本地；逐列輸出進 results
  XD=$X_DATA
  [ "$DRY" = 0 ] && mkdir -p "$XD" "$P11/raw" "$P11/calib"
  step P11Q-PLAN m7run icpe-p11-plan python code/icpe/p11_tools.py plan
  for m in qwen llama; do
    step "P11Q-CALIB-$m"   m7run "icpe-p11-calib-$m"   python code/m16_xstack.py calib --model $m --out "$XD/calib_$m.pt" --summary-dir "$P11/calib"
    step "P11Q-CALIBXL-$m" m7run "icpe-p11-calibxl-$m" python code/m16_xstack.py calib-xl --model $m --out "$XD/calib_xl_$m.pt" --summary-dir "$P11/calib"
    step "P11Q-SELFTEST-$m" m7run "icpe-p11-gpuselftest-$m" python code/m16_xstack.py gpu-selftest --model $m --calib "$XD/calib_$m.pt" \
      --calib-xl "$XD/calib_xl_$m.pt" --new-checks both --out "$P11/calib/gpu_selftest_$m.json"
    step "P11Q-GEN-$m" m7run "icpe-p11-gen-$m" python code/m16_xstack.py gen --model $m --selection "$P11/p11_selection.json" --ctx 32K \
      --tier core --n 48 --seed0 67000 --calib-xl "$XD/calib_xl_$m.pt" --xl-cache --out "$P11/raw/p11_gen_$m.csv.gz"
  done
  step P11Q-REPORT m7run icpe-p11-report-q python code/icpe/p11_tools.py report-q --raw "$P11/raw/p11_gen_qwen.csv.gz" "$P11/raw/p11_gen_llama.csv.gz"
fi

# ============================================================== P1-1 (T)（量時間；§3）
if [ "$PART" = p11t ]; then
  P11=$REPO/results/icpe_h200/p11
  [ "$DRY" = 0 ] && mkdir -p "$P11/time"
  for m in llama qwen; do
    step "P11T-SMOKE-$m" m7run "icpe-p11t-smoke-$m" python code/icpe/p11_time.py loadpath --model $m --out "$P11/time" --reps 1 --smoke
    step "P11T-LP-$m"    m7run "icpe-p11t-lp-$m"    python code/icpe/p11_time.py loadpath --model $m --out "$P11/time" --reps 5
    step "P11T-WC-$m"    m7run "icpe-p11t-wc-$m"    python code/icpe/p11_time.py wcost --model $m --out "$P11/time" --reps 3
  done
  step P11T-REPORT m7run icpe-p11-report-t python code/icpe/p11_tools.py report-t
fi

if [ "$STARTED" = 0 ]; then echo "[r_h200_icpe] 沒有叫做 '$FROM' 的步驟。可用的：$SEEN" >&2; exit 2; fi
[ "$DRY" = 0 ] && echo "[r_h200_icpe] $(date -Is) part=$PART 全部完成"
exit 0
