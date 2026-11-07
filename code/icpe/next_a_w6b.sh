#!/bin/bash
#SBATCH --job-name=icpe-next-w6b
#SBATCH --partition=h200q
#SBATCH --account=ChihChungHsu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=2-00:00:00
#SBATCH -o slurm-%x-%j.out
# =============================================================================
# next_a_w6b.sh — W6b：Mooncake 對話「淘汰順序」（O-ORD）的 GPU 縮小重播（PREREG §5、偏離 1、偏離 3 第 3 點；
# run set 要在送出之前登記成 PREREG 的偏離）。和 next_a_w6.sh 分開寫：W6c／W6a 的 job 排隊中，開跑時會執行 repo 裡的
# next_a_w6.sh，那支在它們跑完前不能改。
#
#   sbatch --dependency=afterany:24444 code/icpe/next_a_w6b.sh --reps 6     # PREREG 偏離 16：6 次；96 個 run 約 26 小時〔推算〕
#   bash code/icpe/next_a_w6b.sh --reps <n> --dry-run | --list ；--from <步驟> 接著跑
#
# 設定同 W6a／T1G P02：Mooncake conversation R1、k＝8、CPU 128／256 GiB ÷ 8、ssd／idle／cake、T1 常數（P3_TIER=h200cpu、
# tier_params_h200cpu.json）、不設 M7_ATTN（sdpa）、Llama-3.1-8B。做法＝O-ORD 的 ②（LRU、LRU2、ARC、S3FIFO、GDSF、PENS）
# 和 ④（O-BEL、O-BELS），由 code/icpe/next_a_w6b.py 呼叫 x1_gpu.py 的重播（不改 x1_gpu.py）。
# 先每個做法冒煙 100 個請求，再跑 rep 0..n-1：一輪 16 個設定、n 輪，每輪內用 seed 20261108＋輪次打亂。
# 量時間（memory h200-no-self-cotenancy-for-timing；2026-11-07 改）：開跑前與每個 run 前都等到節點上沒有我們自己的別的 job（t1g.py wait-clear，
# 最多 NEXT_A_WAIT_MIN 分鐘，預設 1440）；每個 run 跑完用 sacct 查重疊（t1g.py overlap → results/icpe_next/a/w6b/overlap.csv）。
# 被污染（GpuWatcher：外來 process 或查詢失敗，exit 3）的 run 照「暴露」重跑，最多 3 次；污染的 run 留著（CONTAMINATED），分析排除。
# sbatch 送出時會複製一份；開跑時改執行 repo 裡的版本。job 跑的時候不要原地改這支（bash 邊跑邊讀）。
# =============================================================================
set -uo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
if [ -n "${REPO:-}" ] && [ -f "$REPO/code/env_h200.sh" ]; then :
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/code/env_h200.sh" ]; then REPO=$SLURM_SUBMIT_DIR
elif [ -f "$_here/../env_h200.sh" ]; then REPO=$(cd "$_here/../.." && pwd)
else echo "[next_a_w6b] 找不到 repo（code/env_h200.sh）" >&2; exit 2
fi
export REPO
if [ -z "${NEXT_A_W6B_REEXEC:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  export NEXT_A_W6B_REEXEC=1
  echo "[next_a_w6b] 改執行 repo 裡的版本：$REPO/code/icpe/next_a_w6b.sh（git $(git -C "$REPO" rev-parse --short HEAD)）"
  exec bash "$REPO/code/icpe/next_a_w6b.sh" "$@"
fi
REPS=""; DRY=0; LIST=0; FROM=""; REP_FROM=""; REP_TO=""; SMOKE=1; TAG=""
# 2026-11-07 PREREG 偏離 21（W6 用 4 張卡）：--rep-from／--rep-to 只跑其中幾輪（每輪內的打亂不變：seed 20261108＋輪次）、
# --no-smoke 不跑冒煙、--tag T 讓這一份寫自己的 gpu_requests_T.csv／overlap_T.csv（同時跑的幾份不寫同一個檔；跑完再合併）。
# 不帶這些參數時和原本一樣。
while [ $# -gt 0 ]; do
  case "$1" in
    --reps) REPS="$2"; shift 2 ;;
    --rep-from) REP_FROM="$2"; shift 2 ;;
    --rep-to) REP_TO="$2"; shift 2 ;;
    --no-smoke) SMOKE=0; shift ;;
    --tag) TAG="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --list) LIST=1; DRY=1; shift ;;
    --from) FROM="$2"; shift 2 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "不認得的參數：$1" >&2; exit 2 ;;
  esac
done
[[ "$REPS" =~ ^[1-9][0-9]*$ ]] || { echo "要 --reps <正整數>（PREREG 登記的次數）" >&2; exit 2; }
REP_FROM=${REP_FROM:-0}; REP_TO=${REP_TO:-$((REPS - 1))}
[[ "$REP_FROM" =~ ^[0-9]+$ && "$REP_TO" =~ ^[0-9]+$ ]] && [ "$REP_FROM" -le "$REP_TO" ] && [ "$REP_TO" -lt "$REPS" ] \
  || { echo "--rep-from／--rep-to 要在 0..$((REPS - 1)) 之內、而且 from ≤ to" >&2; exit 2; }
[ -z "$TAG" ] || [[ "$TAG" =~ ^[A-Za-z0-9]+$ ]] || { echo "--tag 只能是英數字" >&2; exit 2; }
SFX=${TAG:+_$TAG}
[ "$DRY" = 1 ] && export TIARA_SKIP_CONDA=1
# 模型權重只在 hgx-cn01 本地 /tmp（memory h200-models-on-tmp）；這支只檢查、不下載
export ICPE_MODEL_ROOT="${ICPE_MODEL_ROOT:-/tmp/hungweilai-models/hf-cache}"
export HF_HOME="$ICPE_MODEL_ROOT"
unset M7_LLAMA_GLOB M7_QWEN_GLOB M7_MODEL_GLOB M7_ATTN
# shellcheck disable=SC1091
source "$REPO/code/env_h200.sh"
cd "$REPO"
ulimit -c 0
TP=$REPO/results/icpe_h200/t1/tier_params_h200cpu.json
W6B=$REPO/results/icpe_next/a/w6b
export P3_CONST=h200 P3_TIER=h200cpu M7_TIER_PARAMS="$TP" X1_TRACE_DIR=$TIARA_HOME/datasets/traces

if [ "$DRY" = 0 ]; then
  [ -n "${SLURM_JOB_ID:-}" ] || { echo "[next_a_w6b] 不在 Slurm job 裡（head node 不量時間）" >&2; exit 2; }
  python -c 'import sys; sys.path.insert(0, "code"); from gpu_guard import smi_index; print("[next_a_w6b] smi_index(0) =", smi_index(0))' \
    || { echo "[next_a_w6b] cuda:0 對不到 nvidia-smi 的卡 → 停" >&2; exit 2; }
  # 2026-11-07 使用者：每個 run 前等、跑完查重疊（DL_Lab5 會在中途送新 job）。開跑前也先等。
  python3 code/icpe/t1g.py wait-clear --job "$SLURM_JOB_ID" --max-min "${NEXT_A_WAIT_MIN:-1440}" \
    || { echo "[next_a_w6b] 等不到節點上沒有我們自己的別的 job → 停" >&2; exit 2; }
  python3 "$REPO/code/s3/fetch_models.py" check --root "$ICPE_MODEL_ROOT" --models llama \
    || { echo "[next_a_w6b] Llama 權重不在 $ICPE_MODEL_ROOT（或戳記對不上）→ 停（exit 3）。先跑 s3-fetch，再重送" >&2; exit 3; }
  echo "[next_a_w6b] $(date -Is) reps=$REPS job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-?} git=$(git rev-parse --short HEAD) M7_LLAMA_GLOB=$M7_LLAMA_GLOB"
  echo "[next_a_w6b] 同節點的 job（開跑時）："; squeue -h -w "${SLURMD_NODENAME:-$(hostname -s)}" -o '%i %u %a %b %T %M'
  nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.used --format=csv
  mkdir -p "$W6B"
  python3 - "$TIARA_HOME/datasets/traces" <<'PY' || { echo "[next_a_w6b] Mooncake trace 的 sha256 不對 → 停" >&2; exit 2; }
import hashlib, sys
want = {"conversation_trace.jsonl": "b8cbb061a85206d7", "toolagent_trace.jsonl": "48a2db1a13d3bc05"}
for f, h in want.items():
    got = hashlib.sha256(open(f"{sys.argv[1]}/{f}", "rb").read()).hexdigest()
    print("[next_a_w6b]", f, got[:16]); assert got.startswith(h), (f, got)
PY
fi

STARTED=1; [ -n "$FROM" ] && STARTED=0
OVL="$W6B/overlap$SFX.csv"
# step <名字> <短名> <指令...>：等節點上沒有我們自己的別的 job → 跑 → sacct 查重疊（t1g.py overlap，寫 $OVL）。
# 被污染（exit 3）或重疊 → 照暴露重跑，最多 3 次；3 次都不乾淨 → 停下回報。其他非 0 → 停下回報（規則 2）。分析只用 overlap＝0 的 run。
step() {
  local name=$1 short=$2; shift 2
  if [ "$STARTED" = 0 ]; then if [ "$name" = "$FROM" ]; then STARTED=1; else return 0; fi; fi
  if [ "$LIST" = 1 ]; then echo "$name"; return 0; fi
  if [ "$DRY" = 1 ]; then printf '[%s] m7run %s ' "$name" "$short"; printf '%q ' "$@"; echo; return 0; fi
  local att rc orc t0 t1
  for att in 1 2 3; do
    python3 code/icpe/t1g.py wait-clear --job "$SLURM_JOB_ID" --max-min "${NEXT_A_WAIT_MIN:-1440}" \
      || { echo "=== $(date -Is) $name：等不到節點空出來 → 停、回報。接著跑：--from $name" >&2; exit 2; }
    t0=$(date -Is); echo "=== $t0 $name（第 $att 次）"
    m7run "$short" "$@"; rc=$?
    t1=$(date -Is)
    if [ $rc -ne 0 ] && [ $rc -ne 3 ]; then
      echo "=== $t1 $name 失敗（exit $rc）→ 停下、回報（規則 2）。接著跑：sbatch ... code/icpe/next_a_w6b.sh --reps $REPS --from $name" >&2; exit $rc
    fi
    python3 code/icpe/t1g.py overlap --job "$SLURM_JOB_ID" --start "$t0" --end "$t1" --step "$name" --short "$short" --log "$OVL"; orc=$?
    [ $orc -eq 2 ] && { echo "=== $name：sacct 查不到重疊 → 停（規則 7）" >&2; exit 2; }
    [ $orc -eq 0 ] && [ $rc -eq 0 ] && return 0
    echo "=== $(date -Is) $name：$([ $rc -eq 3 ] && echo '被污染（exit 3）' || echo '和我們自己的 job 重疊') → 照暴露重跑" >&2
  done
  echo "=== $(date -Is) $name 連續 3 次被污染或重疊 → 停下、回報。接著跑：--from $name" >&2; exit 3
}

POLS="LRU LRU2 ARC S3FIFO GDSF PENS O-BEL O-BELS"
if [ "$SMOKE" = 1 ]; then
  for pol in $POLS; do
    step "W6B-SMOKE-$pol" "icpe-next-w6b-smoke-$pol" python code/icpe/next_a_w6b.py gpu --out "$W6B" --cpu 128 --policy "$pol" --max-req 100 --csv "gpu_smoke$SFX.csv"
  done
fi
ORDER=$(python3 - "$REP_FROM" "$REP_TO" <<'PY'
import random, sys
pols = "LRU LRU2 ARC S3FIFO GDSF PENS O-BEL O-BELS".split()
for rep in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
    runs = [(c, p) for c in (128, 256) for p in pols]
    random.Random(20261108 + rep).shuffle(runs)
    for c, p in runs:
        print(c, p, rep)
PY
)
while read -r c pol rep; do
  step "W6B-$c-$pol-r$rep" "icpe-next-w6b-$c-$pol-r$rep" python code/icpe/next_a_w6b.py gpu --out "$W6B" --cpu "$c" --policy "$pol" --rep "$rep" --csv "gpu_requests$SFX.csv"
done <<< "$ORDER"
[ "$DRY" = 0 ] && echo "[next_a_w6b] $(date -Is) reps=$REPS rep $REP_FROM..$REP_TO tag=${TAG:-無} 完成"
exit 0
