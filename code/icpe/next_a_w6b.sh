#!/bin/bash
#SBATCH --job-name=icpe-next-w6b
#SBATCH --partition=h200q
#SBATCH --account=ChihChungHsu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=36:00:00
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
# 量時間：開跑前節點上有我們自己的別的 job → 每 60 s 再查、最多 15 次，還有就退出（memory h200-no-self-cotenancy-for-timing）。
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
REPS=""; DRY=0; LIST=0; FROM=""
while [ $# -gt 0 ]; do
  case "$1" in
    --reps) REPS="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --list) LIST=1; DRY=1; shift ;;
    --from) FROM="$2"; shift 2 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "不認得的參數：$1" >&2; exit 2 ;;
  esac
done
[[ "$REPS" =~ ^[1-9][0-9]*$ ]] || { echo "要 --reps <正整數>（PREREG 登記的次數）" >&2; exit 2; }
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
  for _try in $(seq 1 15); do
    others=$(squeue -h -w "${SLURMD_NODENAME:-$(hostname -s)}" -u "$USER" -t R -o '%i %j' | awk -v me="$SLURM_JOB_ID" '$1!=me')
    [ -z "$others" ] && break
    echo "[next_a_w6b] $(date -Is) 節點上還有我們自己的 job（第 $_try／15 次，60 s 後再查）：$(echo $others)" >&2
    [ "$_try" -lt 15 ] && sleep 60
  done
  if [ -n "$others" ]; then
    echo "[next_a_w6b] 節點上還有我們自己的 job 在跑 → 停（量時間不和自己人同時跑）：" >&2; echo "$others" >&2; exit 2
  fi
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
# step <名字> <短名> <指令...>：exit 3（被污染）→ 照暴露重跑，最多 3 次；其他非 0 → 停下回報（規則 2）
step() {
  local name=$1 short=$2; shift 2
  if [ "$STARTED" = 0 ]; then if [ "$name" = "$FROM" ]; then STARTED=1; else return 0; fi; fi
  if [ "$LIST" = 1 ]; then echo "$name"; return 0; fi
  if [ "$DRY" = 1 ]; then printf '[%s] m7run %s ' "$name" "$short"; printf '%q ' "$@"; echo; return 0; fi
  local att rc
  for att in 1 2 3; do
    echo "=== $(date -Is) $name（第 $att 次）"
    m7run "$short" "$@"; rc=$?
    [ $rc -eq 0 ] && return 0
    if [ $rc -eq 3 ]; then echo "=== $(date -Is) $name 被污染（exit 3）→ 重跑" >&2; continue; fi
    echo "=== $(date -Is) $name 失敗（exit $rc）→ 停下、回報（規則 2）。接著跑：sbatch ... code/icpe/next_a_w6b.sh --reps $REPS --from $name" >&2; exit $rc
  done
  echo "=== $(date -Is) $name 連續 3 次被污染 → 停下、回報" >&2; exit 3
}

POLS="LRU LRU2 ARC S3FIFO GDSF PENS O-BEL O-BELS"
for pol in $POLS; do
  step "W6B-SMOKE-$pol" "icpe-next-w6b-smoke-$pol" python code/icpe/next_a_w6b.py gpu --out "$W6B" --cpu 128 --policy "$pol" --max-req 100 --csv gpu_smoke.csv
done
ORDER=$(python3 - "$REPS" <<'PY'
import random, sys
pols = "LRU LRU2 ARC S3FIFO GDSF PENS O-BEL O-BELS".split()
for rep in range(int(sys.argv[1])):
    runs = [(c, p) for c in (128, 256) for p in pols]
    random.Random(20261108 + rep).shuffle(runs)
    for c, p in runs:
        print(c, p, rep)
PY
)
while read -r c pol rep; do
  step "W6B-$c-$pol-r$rep" "icpe-next-w6b-$c-$pol-r$rep" python code/icpe/next_a_w6b.py gpu --out "$W6B" --cpu "$c" --policy "$pol" --rep "$rep"
done <<< "$ORDER"
[ "$DRY" = 0 ] && echo "[next_a_w6b] $(date -Is) reps=$REPS 完成"
exit 0
