#!/bin/bash
#SBATCH --job-name=icpe-next-w6
#SBATCH --partition=h200q
#SBATCH --account=ChihChungHsu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=14:00:00
#SBATCH -o slurm-%x-%j.out
# =============================================================================
# next_a_w6.sh — NEXT_PLAN_20261104 W6 的 GPU 實測（docs/icpe_brief_20261030/next/PREREG.md §5、偏離 1、偏離 3）。
# 使用者 11/4：「H200 的 GPU 以後只做 W6 實測，現在不排」→ 這支先寫好、只 dry-run；送出要使用者說。
#
#   sbatch --job-name=icpe-next-w6c --time=1:00:00  code/icpe/next_a_w6.sh --part w6c   # 量時間：重算校準 C1 延伸到 256 塊（128K）
#   sbatch --job-name=icpe-next-w6a --time=14:00:00 code/icpe/next_a_w6.sh --part w6a   # 量時間：C2 的 GPU 縮小重播 × 6 次
#   bash code/icpe/next_a_w6.sh --part w6c|w6a --dry-run | --list ；--from <步驟> 接著跑
#   （PREREG 的順序：W6c → W6a → W6b；W6b 要先改 code/icpe/x1_gpu.py，還沒寫）
#
# 量時間：開跑前節點上有我們自己的別的 job → 每 60 s 再查、最多 15 次，還有就退出（memory h200-no-self-cotenancy-for-timing）。
# 被污染（GpuWatcher：外來 process 或查詢失敗）的 run 照「暴露」重跑，最多 3 次；污染的 run 留在 run 目錄（CONTAMINATED），分析排除。
# sbatch 送出時會複製一份；開跑時改執行 repo 裡的版本。job 跑的時候不要原地改這支（bash 邊跑邊讀）。
# =============================================================================
set -uo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
if [ -n "${REPO:-}" ] && [ -f "$REPO/code/env_h200.sh" ]; then :
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/code/env_h200.sh" ]; then REPO=$SLURM_SUBMIT_DIR
elif [ -f "$_here/../env_h200.sh" ]; then REPO=$(cd "$_here/../.." && pwd)
else echo "[next_a_w6] 找不到 repo（code/env_h200.sh）" >&2; exit 2
fi
export REPO
if [ -z "${NEXT_A_W6_REEXEC:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  export NEXT_A_W6_REEXEC=1
  echo "[next_a_w6] 改執行 repo 裡的版本：$REPO/code/icpe/next_a_w6.sh（git $(git -C "$REPO" rev-parse --short HEAD)）"
  exec bash "$REPO/code/icpe/next_a_w6.sh" "$@"
fi
PART=""; DRY=0; LIST=0; FROM=""
while [ $# -gt 0 ]; do
  case "$1" in
    --part) PART="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --list) LIST=1; DRY=1; shift ;;
    --from) FROM="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "不認得的參數：$1" >&2; exit 2 ;;
  esac
done
case "$PART" in
  w6c|w6a) : ;;
  w6b) echo "[next_a_w6] W6b 還沒寫（要先讓 x1_gpu.py 能重播 O-ORD 的其他固定對手；PREREG 偏離 3）" >&2; exit 2 ;;
  *) echo "要 --part w6c|w6a" >&2; exit 2 ;;
esac
[ "$DRY" = 1 ] && export TIARA_SKIP_CONDA=1
# 模型權重只在 hgx-cn01 本地 /tmp（memory h200-models-on-tmp）；這支只檢查、不下載
export ICPE_MODEL_ROOT="${ICPE_MODEL_ROOT:-/tmp/hungweilai-models/hf-cache}"
export HF_HOME="$ICPE_MODEL_ROOT"
unset M7_LLAMA_GLOB M7_QWEN_GLOB M7_MODEL_GLOB
# shellcheck disable=SC1091
source "$REPO/code/env_h200.sh"
cd "$REPO"
ulimit -c 0
P01=$REPO/results/icpe_h200/p01
TP=$REPO/results/icpe_h200/t1/tier_params_h200cpu.json
OUT=$REPO/results/icpe_next/a
W6C=$OUT/w6c
W6A=$OUT/w6a
export P3_CONST=h200 X1_TRACE_DIR=$TIARA_HOME/datasets/traces

if [ "$DRY" = 0 ]; then
  [ -n "${SLURM_JOB_ID:-}" ] || { echo "[next_a_w6] 不在 Slurm job 裡（head node 不量時間）" >&2; exit 2; }
  python -c 'import sys; sys.path.insert(0, "code"); from gpu_guard import smi_index; print("[next_a_w6] smi_index(0) =", smi_index(0))' \
    || { echo "[next_a_w6] cuda:0 對不到 nvidia-smi 的卡 → 停" >&2; exit 2; }
  for _try in $(seq 1 15); do
    others=$(squeue -h -w "${SLURMD_NODENAME:-$(hostname -s)}" -u "$USER" -t R -o '%i %j' | awk -v me="$SLURM_JOB_ID" '$1!=me')
    [ -z "$others" ] && break
    echo "[next_a_w6] $(date -Is) 節點上還有我們自己的 job（第 $_try／15 次，60 s 後再查）：$(echo $others)" >&2
    [ "$_try" -lt 15 ] && sleep 60
  done
  if [ -n "$others" ]; then
    echo "[next_a_w6] 節點上還有我們自己的 job 在跑 → 停（量時間不和自己人同時跑）：" >&2; echo "$others" >&2; exit 2
  fi
  python3 "$REPO/code/s3/fetch_models.py" check --root "$ICPE_MODEL_ROOT" --models llama \
    || { echo "[next_a_w6] Llama 權重不在 $ICPE_MODEL_ROOT（或戳記對不上）→ 停（exit 3）。先跑 s3-fetch，再重送" >&2; exit 3; }
  echo "[next_a_w6] $(date -Is) part=$PART job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-?} git=$(git rev-parse --short HEAD) M7_LLAMA_GLOB=$M7_LLAMA_GLOB"
  echo "[next_a_w6] 同節點的 job（開跑時）："; squeue -h -w "${SLURMD_NODENAME:-$(hostname -s)}" -o '%i %u %a %b %T %M'
  nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.used --format=csv
  mkdir -p "$W6C" "$W6A"
  python3 - "$TIARA_HOME/datasets/traces" <<'PY' || { echo "[next_a_w6] Mooncake trace 的 sha256 不對 → 停" >&2; exit 2; }
import hashlib, sys
want = {"conversation_trace.jsonl": "b8cbb061a85206d7", "toolagent_trace.jsonl": "48a2db1a13d3bc05"}
for f, h in want.items():
    got = hashlib.sha256(open(f"{sys.argv[1]}/{f}", "rb").read()).hexdigest()
    print("[next_a_w6]", f, got[:16]); assert got.startswith(h), (f, got)
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
    echo "=== $(date -Is) $name 失敗（exit $rc）→ 停下、回報（規則 2）。接著跑：sbatch ... code/icpe/next_a_w6.sh --part $PART --from $name" >&2; exit $rc
  done
  echo "=== $(date -Is) $name 連續 3 次被污染 → 停下、回報" >&2; exit 3
}

# ============================================================== W6c：C1 延伸到 256 塊（PREREG §5、偏離 3）
# 和 V1 的校準（run 20261022-205934-icpe-p01-c1：M7_ATTN=fa3 … m7_calib.py c1 --reps 3 --max-chunks 80）同一條指令，只改 --max-chunks 256；
# 另外用 m7_guard_run.py 包 GpuWatcher（原本那次沒有包）。每次嘗試寫自己的檔，乾淨的那次才複製成 calib_c1.csv。
if [ "$PART" = w6c ]; then
  if [ "$DRY" = 1 ]; then WL="<P01 attn/choice.json 的 llama 勝者，必須是 fa3>"
  else
    WL=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1]))['models']['llama']; print('STOP' if d['stop_loss'] else d['winner'])" "$P01/attn/choice.json")
    [ "$WL" = fa3 ] || { echo "[next_a_w6] P01 的 llama 勝者是 $WL，不是 V1 用的 fa3 → 停" >&2; exit 2; }
    [ -e "$W6C/calib_c1.csv" ] && { echo "[next_a_w6] $W6C/calib_c1.csv 已存在 → 停（不覆寫）" >&2; exit 2; }
  fi
  for att in 1 2 3; do
    T="$W6C/calib_c1_try$att.csv"
    if [ "$DRY" = 1 ]; then
      step W6C-C1 icpe-next-w6c-c1 env M7_ATTN="$WL" M7_RES="$W6C" python code/m7_guard_run.py python code/m7_calib.py c1 --out "$T" --reps 3 --max-chunks 256
      break
    fi
    [ "$STARTED" = 0 ] && [ "$FROM" != W6C-C1 ] && break
    STARTED=1
    echo "=== $(date -Is) W6C-C1（第 $att 次）"
    m7run icpe-next-w6c-c1 env M7_ATTN="$WL" M7_RES="$W6C" python code/m7_guard_run.py python code/m7_calib.py c1 --out "$T" --reps 3 --max-chunks 256; rc=$?
    if [ $rc -eq 0 ]; then cp "$T" "$W6C/calib_c1.csv"; echo "[next_a_w6] W6C-C1 乾淨 → $W6C/calib_c1.csv"; break; fi
    if [ $rc -eq 3 ]; then echo "=== $(date -Is) W6C-C1 被污染 → 重跑（$T 留著，不用）" >&2; [ $att -lt 3 ] && continue; echo "連續 3 次被污染 → 停" >&2; exit 3; fi
    echo "=== $(date -Is) W6C-C1 失敗（exit $rc）→ 停下、回報（規則 2）" >&2; exit $rc
  done
fi

# ============================================================== W6a：C2 的 GPU 縮小重播 × 6 次（PREREG §5）
# 設定同 C2 的 GPU 實測（T1G P02 的 x1_gpu.py gpu --k 8；CPU 128／256 ÷ 8；T1 常數；不設 M7_ATTN＝sdpa，和那次相同）。
# 做法：LRU、HINT-PF、O-PF；每個設定 6 次（rep 0–5，新的輸出目錄，不和舊的混）。順序：一輪 6 個設定、6 輪，每輪內固定 seed 打亂。
if [ "$PART" = w6a ]; then
  export P3_TIER=h200cpu M7_TIER_PARAMS="$TP"
  step W6A-SMOKE icpe-next-w6a-smoke python code/icpe/x1_gpu.py gpu --out "$W6A" --k 8 --cpu 128 --policy LRU --max-req 100 --csv gpu_smoke.csv
  ORDER=$(python3 - <<'PY'
import random
for rep in range(6):
    runs = [(c, p) for c in (128, 256) for p in ("LRU", "HINT-PF", "O-PF")]
    random.Random(20261105 + rep).shuffle(runs)
    for c, p in runs:
        print(c, p, rep)
PY
)
  while read -r c pol rep; do
    step "W6A-$c-$pol-r$rep" "icpe-next-w6a-$c-$pol-r$rep" python code/icpe/x1_gpu.py gpu --out "$W6A" --k 8 --cpu "$c" --policy "$pol" --rep "$rep"
  done <<< "$ORDER"
  unset P3_TIER
fi
[ "$DRY" = 0 ] && echo "[next_a_w6] $(date -Is) part=$PART 完成"
exit 0
