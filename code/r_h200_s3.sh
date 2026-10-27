#!/bin/bash
#SBATCH --job-name=s3
#SBATCH --partition=h200q
#SBATCH --account=ChihChungHsu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=4:00:00
#SBATCH -o slurm-%x-%j.out
# =============================================================================
# r_h200_s3.sh — 第三階段（docs/stage3_design_20261024/）在 H200 的 runner。每一步一個 run 目錄（m7run）。
#
#   sbatch --job-name=s3-fetch --time=1:00:00 code/r_h200_s3.sh --part fetch     # 模型權重 → hgx-cn01 /tmp（不量時間）
#   sbatch --job-name=s3-a0smoke-L128 --time=1:00:00 code/r_h200_s3.sh --part a0-smoke --cell L128   # A0：校準＋gpu-selftest＋2 題冒煙（不進 results）
#   sbatch --job-name=s3-a0tf-L128-0 --time=4:00:00 code/r_h200_s3.sh --part a0-tf --cell L128 --shard 0/1   # A0 TF 篩選的一個分片
#   sbatch --job-name=s3-a0gen-L128-0 --time=4:00:00 code/r_h200_s3.sh --part a0-gen --cell L128 --shard 0/2 # A0 生成的一個分片
#   sbatch --job-name=s3-b0prep --time=1:00:00 code/r_h200_s3.sh --part b0-prep                  # B0 冒煙（不量時間）：P0e、背景自檢、少量格子、MPS 冒煙
#   sbatch --job-name=s3-b0-llama-b0-stream --time=4:00:00 code/r_h200_s3.sh --part b0-run --model llama --setup b0 --modes stream,stream_hi  # 量時間
#   sbatch --job-name=s3-b0-llama-b0-mps --time=6:00:00 code/r_h200_s3.sh --part b0-run --model llama --setup b0 --modes mps,mps87,mps75,mps50  # 量時間（自動開 MPS）
#   sbatch --job-name=s3-w0gfio --time=1:00:00 code/r_h200_s3.sh --part w0gfio                    # 量時間：W0（兩個模型）＋G-fio
#   sbatch --job-name=s3-gm06 --time=1:00:00 --gres=gpu:4 --cpus-per-task=64 --mem=680G code/r_h200_s3.sh --part gm06   # 4 卡、量時間
#   sbatch --job-name=s3-kvs-s0-L128-0 --time=4:00:00 code/r_h200_s3.sh --part kvs-s0 --cell L128 --shard 0/1   # KVS S0-2（品質型；第一次先 1 題冒煙）
#   bash code/r_h200_s3.sh --part <p> --list | --dry-run ；--from <步驟> 接著跑
#
# 排程（使用者 2026-10-25「排滿：stage3 品質先填空檔」，memory stage3-gpu-scheduling）：
#   品質／不量時間的 part（TIMING=0）：節點上有 ICPE 的量時間 job 在跑 → 不開跑（exit 3）。反方向（我們在跑時 ICPE 量時間的 job 開跑）
#     由送出端處理：送出時把下一個要跑的 ICPE 量時間 job 加上 afterany:<這個 job>（不動 hold 中的 ICPE job）。
#   量時間的 part（TIMING=1）：節點上有我們自己的任何別的 job（含 DL_Lab4、ICPE）→ 不開跑（memory h200-no-self-cotenancy-for-timing）。
# 模型權重放 hgx-cn01 本地 /tmp（使用者 2026-10-25；/work 的 hf-cache 自 10/23 起是空的）：要 GPU 的 part 開跑前 ensure_models。
# sbatch 送出時會複製一份；開跑時改執行 repo 裡的版本（R_H200_S3_REEXEC）。跑的時候不要原地改這支：寫新檔再 mv 過去。
# =============================================================================
set -uo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
if [ -n "${REPO:-}" ] && [ -f "$REPO/code/env_h200.sh" ]; then :
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/code/env_h200.sh" ]; then REPO=$SLURM_SUBMIT_DIR
elif [ -f "$_here/env_h200.sh" ]; then REPO=$(cd "$_here/.." && pwd)
else echo "[r_h200_s3] 找不到 repo（code/env_h200.sh）" >&2; exit 2
fi
export REPO
if [ -z "${R_H200_S3_REEXEC:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  export R_H200_S3_REEXEC=1
  echo "[r_h200_s3] 改執行 repo 裡的版本：$REPO/code/r_h200_s3.sh（git $(git -C "$REPO" rev-parse --short HEAD)）"
  exec bash "$REPO/code/r_h200_s3.sh" "$@"
fi
ORIG_ARGS=("$@")                                              # 讓位時原樣重送用
PART=""; DRY=0; LIST=0; FROM=""; CELL=""; SHARD="0/1"; BUDGET=210; MODEL=""; SETUP=""; MODES=""
while [ $# -gt 0 ]; do
  case "$1" in
    --part) PART="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --list) LIST=1; DRY=1; shift ;;
    --from) FROM="$2"; shift 2 ;;
    --cell) CELL="$2"; shift 2 ;;
    --shard) SHARD="$2"; shift 2 ;;
    --budget-min) BUDGET="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --setup) SETUP="$2"; shift 2 ;;
    --modes) MODES="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "不認得的參數：$1" >&2; exit 2 ;;
  esac
done
case "$PART" in
  fetch|a0-smoke|a0-gensmoke|a0-tf|a0-gen|b0-prep|gm12-q|d0-prep|d0-run|kvs-s0|gm12-diag) TIMING=0 ;;
  b0-run|b0-diag|gm11-redo|w0gfio|gm12-s|gm06) TIMING=1 ;;
  *) echo "要 --part fetch|a0-smoke|a0-gensmoke|a0-tf|a0-gen|b0-prep|b0-run|b0-diag|gm11-redo|w0gfio|gm12-q|gm12-s|gm12-diag|d0-prep|d0-run|gm06|kvs-s0" >&2; exit 2 ;;
esac
if [ "$PART" = b0-run ]; then
  [ -n "$MODEL" ] && [ -n "$SETUP" ] && [ -n "$MODES" ] || { echo "b0-run 要 --model --setup --modes" >&2; exit 2; }
fi
case "$PART" in a0-*|gm12-q|kvs-s0) [ -n "$CELL" ] || { echo "--part $PART 要 --cell" >&2; exit 2; } ;; esac
ICPE_TIMING_RE='^icpe-(p01s1|p01ab|p01brr|p02|p2|p11t)$'     # r_h200_icpe.sh 裡 TIMING=1 的 part

# 模型權重：本地 /tmp（code/s3/fetch_models.py）。env_h200.sh 的 M7_*_GLOB 用 ${HF_HOME}，所以先設 HF_HOME
export S3_MODEL_ROOT=/tmp/hungweilai-models/hf-cache
export HF_HOME=$S3_MODEL_ROOT
export S3_LOCAL=/tmp/hungweilai-s3                                      # 校準基底等大檔（本地）
[ "$DRY" = 1 ] && export TIARA_SKIP_CONDA=1
# shellcheck disable=SC1091
source "$REPO/code/env_h200.sh"
eval "$(python3 "$REPO/code/s3/fetch_models.py" env --root "$S3_MODEL_ROOT")"       # M7_LLAMA_GLOB、M7_QWEN_GLOB、M7_QWEN14_GLOB
cd "$REPO"
ulimit -c 0
S3=$REPO/results/s3_h200

if [ "$DRY" = 0 ]; then
  [ -n "${SLURM_JOB_ID:-}" ] || { echo "[r_h200_s3] 不在 Slurm job 裡" >&2; exit 2; }
  NODE="${SLURMD_NODENAME:-$(hostname -s)}"
  mine=$(squeue -h -w "$NODE" -u "$USER" -t R -o '%i %j' | awk -v me="$SLURM_JOB_ID" '$1!=me')
  icpe_t=$(printf '%s\n' "$mine" | awk '{print $2}' | grep -E "$ICPE_TIMING_RE" || true)
  if [ -n "$icpe_t" ]; then
    echo "[r_h200_s3] 節點上有 ICPE 量時間的 job 在跑（$icpe_t）→ 不開跑（exit 3；stage3 不和 ICPE 量時間重疊）" >&2; exit 3
  fi
  # 讓位（code/s3/sched.py）：ICPE 量時間的 job 如果只剩在等 stage3（或已經在等資源），這個 job 不佔這個空檔：
  # 原樣重送一份、排在那個 ICPE job 之後，然後結束（exit 0；它在 ICPE 的依賴裡是 afterany，所以 ICPE 可以開跑）
  ytgt=$(python3 "$REPO/code/s3/sched.py" yield-target 2>/dev/null || true)
  # 品質的 job 也讓位給「只剩在等 stage-3」的 stage-3 量時間 job（s3-T-*；同一個重送做法）
  if [ -z "$ytgt" ] && [ "$TIMING" = 0 ]; then
    ytgt=$(python3 "$REPO/code/s3/sched.py" s3-timing-target --me "$SLURM_JOB_ID" 2>/dev/null || true)
  fi
  if [ -n "$ytgt" ]; then
    tl=$(scontrol show job "$SLURM_JOB_ID" | grep -o 'TimeLimit=[^ ]*' | cut -d= -f2)
    dep=$(echo "$ytgt" | tr ' ' '\n' | sed 's/^/afterany:/' | paste -sd,)
    nj=$(env -u TMPDIR -u CLAUDE_CODE_TMPDIR sbatch --parsable --job-name="$SLURM_JOB_NAME" --time="$tl" --dependency="$dep" \
         -D "$REPO" "$REPO/code/r_h200_s3.sh" "${ORIG_ARGS[@]}")
    echo "[r_h200_s3] 讓位給 ICPE 量時間的 job $ytgt：重送成 $nj（$dep），這個 job 結束" >&2
    exit 0
  fi
  if [ "$TIMING" = 1 ] && [ -n "$mine" ]; then
    # 量時間不和自己人同時跑：重送一份排在那些 job 之後（不直接失敗）；品質的 job 會讓位給排隊中的它
    run_ids=$(printf '%s\n' "$mine" | awk '{print $1}')
    dep=$(echo "$run_ids" | sed 's/^/afterany:/' | paste -sd,)
    tl=$(scontrol show job "$SLURM_JOB_ID" | grep -o 'TimeLimit=[^ ]*' | cut -d= -f2)
    gres=$(scontrol show job "$SLURM_JOB_ID" | grep -o 'TresPerNode=[^ ]*' | cut -d= -f2 | sed 's/gres\/gpu:/gpu:/; s/gres:gpu:/gpu:/')
    cpus=$(scontrol show job "$SLURM_JOB_ID" | grep -o 'NumCPUs=[0-9]*' | cut -d= -f2)
    nj=$(env -u TMPDIR -u CLAUDE_CODE_TMPDIR sbatch --parsable --job-name="$SLURM_JOB_NAME" --time="$tl" --dependency="$dep" \
         ${gres:+--gres=$gres} ${cpus:+--cpus-per-task=$cpus} -D "$REPO" "$REPO/code/r_h200_s3.sh" "${ORIG_ARGS[@]}")
    echo "[r_h200_s3] 量時間：節點上還有我們自己的 job（$(echo $run_ids)）→ 重送成 $nj（$dep），這個 job 結束" >&2
    exit 0
  fi
  echo "[r_h200_s3] $(date -Is) part=$PART job=$SLURM_JOB_ID node=$NODE git=$(git rev-parse --short HEAD) timing=$TIMING"
  echo "[r_h200_s3] 同節點的 job（開跑時）："; squeue -h -w "$NODE" -o '%i %u %a %j %b %T %M'
  nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.used --format=csv
  mkdir -p "$S3/env"
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
  if [ $rc -ne 0 ]; then echo "=== $(date -Is) $name 失敗（exit $rc）→ 停下、回報（規則 2）。接著跑：sbatch ... code/r_h200_s3.sh --part $PART --from $name" >&2; exit $rc; fi
}
# 要 GPU 的 part 開跑前呼叫：戳記對不上 → 重抓一次 → 再核對（重抓後還不對就停）
ensure_models() {
  [ "$DRY" = 1 ] && { echo "[ensure_models $*]"; return 0; }
  python3 code/s3/fetch_models.py check --root "$S3_MODEL_ROOT" --models "$@" \
    && { python3 code/s3/fetch_models.py refresh --root "$S3_MODEL_ROOT" --models "$@"; return $?; }   # /tmp 是 noatime：touch 一次
  echo "[r_h200_s3] 權重不在或對不上 → 重抓：$*"
  m7run s3-fetch python code/s3/fetch_models.py fetch --root "$S3_MODEL_ROOT" --models "$@" || return 2
  python3 code/s3/fetch_models.py check --root "$S3_MODEL_ROOT" --models "$@"
}

if [ "$PART" = fetch ]; then
  step FETCH m7run s3-fetch python code/s3/fetch_models.py fetch --root "$S3_MODEL_ROOT" --models llama qwen qwen14
  step CHECK-FULL python3 code/s3/fetch_models.py check --root "$S3_MODEL_ROOT" --models llama qwen qwen14 --full
fi

# ---- A0（docs/stage3_design_20261024/runs/A0_progressive_restore.md）----------------------------------------
# 校準基底：沒有就算（同一個模型的兩個分片同時開跑時用 flock 排隊；m16_xstack.cmd_calib_xl 遇到設定、sha256 相同的檔會跳過）
ensure_calib() {
  local M=$1
  [ "$DRY" = 1 ] && { echo "[ensure_calib $M]"; return 0; }
  mkdir -p "$S3_LOCAL/xstack"
  flock "$S3_LOCAL/.calib-$M.lock" bash -c "[ -f '$S3_LOCAL/xstack/calib_xl_$M.pt' ] || TIARA_SHORT=s3-a0-calib-$M bash '$REPO/code/tiara.sbatch' python code/s3/a0.py calib --model $M" \
    && touch "$S3_LOCAL/xstack/calib_xl_$M.pt" "$S3_LOCAL/xstack/calib_xl_$M.pt.json"     # noatime：touch 才不會被 30 天清掉
}
if [[ "$PART" == a0-* ]]; then
  M=$(python3 -c "import sys; sys.path.insert(0, 'code/s3'); import a0; print(a0.CELLS[sys.argv[1]]['model'])" "$CELL") \
    || { echo "[r_h200_s3] 不認得的 --cell $CELL" >&2; exit 2; }
  SI=${SHARD%%/*}
  step MODELS ensure_models "$M"
  step CALIB-$M ensure_calib "$M"
  if [ "$PART" = a0-smoke ]; then
    step SELFTEST-$CELL m7run s3-a0-selftest-$CELL bash -c "python code/s3/a0.py gpu-selftest --cell $CELL --out \"\$RUN_DIR/gpu_selftest.json\""
    step SMOKE-$CELL m7run s3-a0-smoke-$CELL bash -c "python code/s3/a0.py tf --cell $CELL --smoke --max-items 2 --gm04 --out \"\$RUN_DIR/smoke_tf.csv.gz\""
  fi
  # 生成的冒煙：單輪 1 題＋第 2 輪 1 題（--max-items 照題組的順序，第 2 輪的題在後面 → 用 --shard 讓兩組各出一題）
  [ "$PART" = a0-gensmoke ] && step GENSMOKE-$CELL m7run s3-a0-gensmoke-$CELL bash -c "python code/s3/a0.py gen --cell $CELL --smoke --max-items 2 --shard 0/192 --out \"\$RUN_DIR/smoke_gen.csv.gz\""
  [ "$PART" = a0-tf ] && step TF-$CELL-$SI m7run s3-a0-tf-$CELL-s$SI python code/s3/a0.py tf --cell "$CELL" --shard "$SHARD" --budget-min "$BUDGET" --gm04
  # 生成的閘（10/26）：這份程式（a0／a0_gpu 的雜湊）在這一格還沒過生成冒煙 → 先在同一個 job 裡跑一次（2 題、輸出在 run 目錄、不看分數）。
  # 為什麼不用 job 依賴：冒煙 job 會讓位、重送成新編號，afterok 就斷了（S3-B0-GATE 同一個理由）。失敗 → step 停下（規則 2）。
  if [ "$PART" = a0-gen ]; then
    GATES=$TIARA_RUNS/.s3_gates; [ "$DRY" = 1 ] || mkdir -p "$GATES"
    GSIG=$(cat code/s3/a0.py code/s3/a0_gpu.py | sha256sum | cut -c1-16)
    if [ ! -f "$GATES/a0gen_${CELL}_$GSIG" ]; then
      step GENSMOKE-$CELL m7run s3-a0-gensmoke-$CELL bash -c "python code/s3/a0.py gen --cell $CELL --smoke --max-items 2 --shard 0/192 --out \"\$RUN_DIR/smoke_gen.csv.gz\""
      [ "$DRY" = 1 ] || echo "$(date -Is) job=$SLURM_JOB_ID" > "$GATES/a0gen_${CELL}_$GSIG"
    fi
  fi
  [ "$PART" = a0-gen ] && step GEN-$CELL-$SI m7run s3-a0-gen-$CELL-s$SI python code/s3/a0.py gen --cell "$CELL" --shard "$SHARD" --budget-min "$BUDGET"
fi

# ---- KVS S0-2（docs/kv_sharing_hiding_20261026/runs/S0_union.md）：兩個使用者挑的塊的聯集（品質型，不量時間）----------
# 重用 A0 的題目與打分（不改 A0）。這份程式（kvs_s0／a0_gpu／a0 的雜湊）在這一格還沒冒煙過 → 先跑 1 題冒煙（輸出在 run 目錄、不看分數）。
if [ "$PART" = kvs-s0 ]; then
  M=$(python3 -c "import sys; sys.path.insert(0, 'code/s3'); import a0; print(a0.CELLS[sys.argv[1]]['model'])" "$CELL") \
    || { echo "[r_h200_s3] 不認得的 --cell $CELL" >&2; exit 2; }
  SI=${SHARD%%/*}
  step MODELS ensure_models "$M"
  GATES=$TIARA_RUNS/.s3_gates; [ "$DRY" = 1 ] || mkdir -p "$GATES"
  S0SIG=$(cat code/s3/kvs_s0.py code/s3/a0_gpu.py code/s3/a0.py | sha256sum | cut -c1-16)
  if [ ! -f "$GATES/kvs_s0_${CELL}_$S0SIG" ]; then
    step KVS-S0-SMOKE-$CELL m7run s3-kvs-s0-smoke-$CELL bash -c "python code/s3/kvs_s0.py run --cell $CELL --smoke --max-items 1 --out \"\$RUN_DIR/smoke_s0.csv.gz\""
    [ "$DRY" = 1 ] || echo "$(date -Is) job=$SLURM_JOB_ID" > "$GATES/kvs_s0_${CELL}_$S0SIG"
  fi
  step KVS-S0-$CELL-$SI m7run s3-kvs-s0-$CELL-s$SI python code/s3/kvs_s0.py run --cell "$CELL" --shard "$SHARD" --budget-min "$BUDGET"
fi

# ---- B0＋G-M08（docs/stage3_design_20261024/runs/B0_interference.md）---------------------------------------
# MPS：job 自己的 pipe／log 目錄（和 h200_queue/h2_mps_p4.sh 同一個做法）；結束時關 daemon、把 log 留在 run 的結果旁邊
mps_on() {
  [ "$DRY" = 1 ] && { echo "[mps_on]"; return 0; }
  command -v nvidia-cuda-mps-control >/dev/null || { echo "[r_h200_s3] 找不到 nvidia-cuda-mps-control" >&2; return 2; }
  export CUDA_MPS_PIPE_DIRECTORY=$S3_LOCAL/mps_pipe_${SLURM_JOB_ID} CUDA_MPS_LOG_DIRECTORY=$S3_LOCAL/mps_log_${SLURM_JOB_ID}
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  nvidia-cuda-mps-control -d || return 2
  export GPU_GUARD_JOB_CGROUP_OWN=1
  trap 'echo quit | nvidia-cuda-mps-control; sleep 2; mkdir -p $S3/b0/mps_log_$SLURM_JOB_ID; cp $CUDA_MPS_LOG_DIRECTORY/* $S3/b0/mps_log_$SLURM_JOB_ID/ 2>/dev/null; rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY' EXIT
  echo "[r_h200_s3] MPS daemon 開了（pipe $CUDA_MPS_PIPE_DIRECTORY）"
}
if [ "$PART" = b0-prep ]; then
  step MODELS ensure_models llama qwen
  step CALIB-llama ensure_calib llama
  step CALIB-qwen ensure_calib qwen
  step B0-ENV m7run s3-b0-env bash -c 'python code/s3/b0.py env --out "$RUN_DIR/env.json"'
  for m in llama qwen; do
    step B0-BGCHECK-$m m7run s3-b0-bgcheck-$m bash -c "python code/s3/b0.py bgcheck --model $m --out \"\$RUN_DIR/bgcheck_$m.csv\""
  done
  step B0-SMOKE m7run s3-b0-smoke bash -c 'python code/s3/b0.py run --model llama --setup b0 --modes stream,stream_hi --bgs idle,decode:8:8192,cprefill:2048:65536 --prims h2d_batch,sm_gather_12,dq_int4,dq_fp8,dq_nvfp4,dq_vq,lr_recon,recompute1,e2e_cake_bf16_cpu,e2e_load_int4_b32 --reps 6 --out "$RUN_DIR/smoke.csv"'
  step B0-SMOKE-M08 m7run s3-b0-smoke-m08 bash -c 'python code/s3/b0.py run --model qwen --setup m08 --modes stream --bgs idle,decode:4:258048 --prims recompute1,qprefill,dq_vq,lr_recon --reps 6 --out "$RUN_DIR/smoke_m08.csv"'
  step MPS-ON mps_on
  step B0-SMOKE-MPS m7run s3-b0-smoke-mps bash -c 'python code/s3/b0.py run --model llama --setup b0 --modes mps,mps75 --bgs decode:32:8192 --prims recompute1,h2d_batch,dq_vq --reps 6 --out "$RUN_DIR/smoke_mps.csv"'
fi
if [ "$PART" = b0-run ]; then
  step MODELS ensure_models "$MODEL"
  step CALIB-$MODEL ensure_calib "$MODEL"
  # 量測前的閘（10/26）：這組（模型, 設定）在這份程式（b0／bgengine／guard／numa_probe／gpu_guard 的雜湊）上還沒冒煙過 →
  # 先在同一個獨佔的 allocation 裡跑一次：背景 idle、stream、這個設定的全部原語各 6 次，輸出在 run 目錄（不進 results）。
  # 為什麼不用 job 依賴：品質 job（b0prep2）會讓位給量時間的 job、重送成新編號，依賴就斷了。失敗 → step 停下（規則 2）。
  GATES=$TIARA_RUNS/.s3_gates; mkdir -p "$GATES"
  B0SIG=$(cat code/s3/b0.py code/s3/bgengine.py code/s3/guard.py code/s3/numa_probe.py code/gpu_guard.py | sha256sum | cut -c1-16)
  if [ ! -f "$GATES/b0_${MODEL}_${SETUP}_$B0SIG" ]; then
    step B0-GATE-$MODEL-$SETUP m7run "s3-b0-gate-$MODEL-$SETUP" bash -c "python code/s3/b0.py run --model $MODEL --setup $SETUP --modes stream --bgs idle --reps 6 --seed 9 --out \"\$RUN_DIR/gate_smoke.csv\""
    [ "$DRY" = 1 ] || echo "$(date -Is) job=$SLURM_JOB_ID" > "$GATES/b0_${MODEL}_${SETUP}_$B0SIG"
  fi
  case "$MODES" in *mps*) step MPS-ON mps_on ;; esac
  mkdir -p "$S3/b0/raw"
  TAG=$(echo "$MODES" | tr ',' '-')
  step B0-RUN m7run "s3-b0-$MODEL-$SETUP-$TAG" bash -c "python code/s3/b0.py run --model $MODEL --setup $SETUP --modes $MODES --reps 6 --seed 1 --out $S3/b0/raw/b0_${MODEL}_${SETUP}_${TAG}_\$RUN_ID.csv"
fi

# ---- B0 MPS 的診斷（S3-B0-FAIL-2：22275 在 MPS 下 lr_recon 時 illegal memory access）：輸出只在 run 目錄、不進 results
# 每個背景各跑一次（CUDA_LAUNCH_BLOCKING=1 定位 kernel；背景小／大對照），各自記 exit code，第一個失敗也照樣跑第二個
if [ "$PART" = b0-diag ]; then
  step MODELS ensure_models llama
  step CALIB-llama ensure_calib llama
  step MPS-ON mps_on
  for bg in decode:8:8192 decode:64:8192; do
    step "B0-DIAG-${bg//:/_}" m7run s3-b0-diag bash -c "CUDA_LAUNCH_BLOCKING=1 python code/s3/b0.py run --model llama --setup b0 --modes mps --bgs $bg --prims lr_recon,recompute1 --reps 6 --seed 7 --out \"\$RUN_DIR/diag.csv\"; echo DIAG_EXIT=\$?"
  done
fi

# ---- G-M11 重量（runs/G_M11_vq.md 偏離 1；使用者 10/26「都做掉」）：只量判準的四格（閒／matmul × dq_vq／dq_int4，b0、stream），
# 每格 12 次、兩個模型；輸出放 results/s3_h200/b0/gm11/（不進 B0 的 raw，判定只用這個 run 的四格）
if [ "$PART" = gm11-redo ]; then
  step MODELS ensure_models llama qwen
  step CALIB-llama ensure_calib llama
  step CALIB-qwen ensure_calib qwen
  mkdir -p "$S3/b0/gm11"
  for m in llama qwen; do
    step GM11-$m m7run s3-gm11-$m bash -c "python code/s3/b0.py run --model $m --setup b0 --modes stream --bgs idle,matmul --prims dq_vq,dq_int4 --reps 12 --seed 11 --out $S3/b0/gm11/b0_gm11_${m}_\$RUN_ID.csv"
  done
  step GM11-REPORT python3 code/s3/b0.py report --raw-dir "$S3/b0/gm11" --name gm11_report.json
fi

# ---- G-M12（runs/G_M12_sparse_recompute.md）-----------------------------------------------------------------
if [ "$PART" = gm12-q ]; then
  M=$(python3 -c "import sys; sys.path.insert(0, 'code/s3'); import a0; print(a0.CELLS[sys.argv[1]]['model'])" "$CELL")
  SI=${SHARD%%/*}
  step MODELS ensure_models "$M"
  [ "$SI" = 0 ] && step GM12-SELFTEST-$CELL m7run s3-gm12-selftest-$CELL bash -c "python code/s3/gm12.py gpu-selftest --cell $CELL --out \"\$RUN_DIR/gm12_selftest.json\""
  step GM12-Q-$CELL-$SI m7run s3-gm12-q-$CELL-s$SI python code/s3/gm12.py quality --cell "$CELL" --shard "$SHARD" --budget-min "$BUDGET"
fi
# G-M12 自檢失敗的診斷（S3-GM12-FAIL-1；只報告，判準不改）：Q128 與 L128（對照）各跑 default／num_splits=1 兩輪
if [ "$PART" = gm12-diag ]; then
  step MODELS ensure_models llama qwen
  for c in Q128 L128; do
    step GM12-DIAG-$c m7run s3-gm12-diag-$c bash -c "python code/s3/gm12.py diag --cell $c --out \"\$RUN_DIR/gm12_diag_$c.json\""
  done
fi
if [ "$PART" = gm12-s ]; then
  step MODELS ensure_models llama qwen
  for m in llama qwen; do step GM12-S-$m m7run s3-gm12-s-$m python code/s3/gm12.py speed --model $m; done
fi

# ---- D0 的可行性（P0f）：MiniCPM4.1-8B＋infllmv2_cuda_impl（裝在本地的獨立目錄，不碰 conda 環境）------------------------
D0_SRC_COMMIT=93cf2ec28e5a                                   # OpenBMB/infllmv2_cuda_impl main（10/25 讀；feature_infer 分支已不在）
d0_build() {
  [ "$DRY" = 1 ] && { echo "[d0_build]"; return 0; }
  local D=$S3_LOCAL/d0
  mkdir -p "$D"
  if [ -f "$D/pylibs/.built_$D0_SRC_COMMIT" ]; then echo "[d0] kernel 已編好（$D0_SRC_COMMIT）"; return 0; fi
  rm -rf "$D/src" "$D/pylibs" "$D/wheels"
  git clone https://github.com/OpenBMB/infllmv2_cuda_impl.git "$D/src" || return 2
  git -C "$D/src" checkout "$D0_SRC_COMMIT" || return 2
  git -C "$D/src" submodule update --init --recursive || return 2
  export CUDA_HOME=/cm/shared/apps/cuda13.3 PATH=/cm/shared/apps/cuda13.3/bin:$PATH MAX_JOBS=16 TORCH_CUDA_ARCH_LIST=9.0
  nvcc --version | tail -2
  (cd "$D/src" && pip wheel --no-deps --no-build-isolation -w "$D/wheels" .) || return 2
  pip install --no-deps --target "$D/pylibs" "$D"/wheels/*.whl || return 2
  touch "$D/pylibs/.built_$D0_SRC_COMMIT"
}
# MiniCPM 的 modeling 程式是照 transformers 4.56.1 寫的（config.json；5.x 沒有 is_torch_fx_available、cache layer API 不同）
# → 側裝到 D0 自己的 pylibs（不動 conda 環境）；hub／tokenizers 照 4.56.1 的相依範圍。裝了什麼印進 log＋.pip_list
d0_pylibs() {
  [ "$DRY" = 1 ] && { echo "[d0_pylibs]"; return 0; }
  local D=$S3_LOCAL/d0
  if [ -f "$D/pylibs/.tf_4.56.1" ]; then echo "[d0] transformers 4.56.1 已裝"; cat "$D/pylibs/.pip_list"; return 0; fi
  pip install --no-deps --target "$D/pylibs" "transformers==4.56.1" "huggingface_hub>=0.34,<1.0" "tokenizers>=0.22,<=0.23.0" || return 2
  pip list --path "$D/pylibs" 2>/dev/null | tee "$D/pylibs/.pip_list"
  touch "$D/pylibs/.tf_4.56.1"
}
if [ "$PART" = d0-prep ]; then
  step MODELS ensure_models minicpm
  step D0-BUILD m7run s3-d0-build bash -c "$(declare -f d0_build); S3_LOCAL=$S3_LOCAL D0_SRC_COMMIT=$D0_SRC_COMMIT DRY=0 d0_build"
  step D0-PYLIBS m7run s3-d0-pylibs bash -c "$(declare -f d0_pylibs); S3_LOCAL=$S3_LOCAL DRY=0 d0_pylibs"
  # 轉接層（code/s3/d0_shim：flash_attn → infllm_v2 的 dense varlen）＋pylibs 排在最前面
  step D0-SMOKE m7run s3-d0-smoke bash -c "PYTHONPATH=$REPO/code/s3/d0_shim:$S3_LOCAL/d0/pylibs:\$PYTHONPATH python code/s3/d0.py smoke --out \"\$RUN_DIR/d0_smoke.json\""
fi
# D0 量測（runs/D0_native_index.md；品質）：要先有 PASS 的冒煙（kernel 與 pylibs 已在 /tmp；不在就重編）
if [ "$PART" = d0-run ]; then
  step MODELS ensure_models minicpm
  step D0-BUILD m7run s3-d0-build bash -c "$(declare -f d0_build); S3_LOCAL=$S3_LOCAL D0_SRC_COMMIT=$D0_SRC_COMMIT DRY=0 d0_build"
  step D0-PYLIBS m7run s3-d0-pylibs bash -c "$(declare -f d0_pylibs); S3_LOCAL=$S3_LOCAL DRY=0 d0_pylibs"
  # D0 冒煙的閘（10/26；理由同生成的閘）：這份 d0.py＋轉接層還沒過冒煙 → 先跑 P0f 冒煙（輸出在 run 目錄）；失敗 → 停
  GATES=$TIARA_RUNS/.s3_gates; [ "$DRY" = 1 ] || mkdir -p "$GATES"
  DSIG=$(cat code/s3/d0.py code/s3/d0_shim/flash_attn/*.py | sha256sum | cut -c1-16)
  if [ ! -f "$GATES/d0_$DSIG" ]; then
    step D0-SMOKE m7run s3-d0-smoke bash -c "PYTHONPATH=$REPO/code/s3/d0_shim:$S3_LOCAL/d0/pylibs:\$PYTHONPATH python code/s3/d0.py smoke --out \"\$RUN_DIR/d0_smoke.json\""
    [ "$DRY" = 1 ] || echo "$(date -Is) job=$SLURM_JOB_ID" > "$GATES/d0_$DSIG"
  fi
  step D0-RUN-${SHARD%%/*} m7run "s3-d0-run-s${SHARD%%/*}" bash -c "PYTHONPATH=$REPO/code/s3/d0_shim:$S3_LOCAL/d0/pylibs:\$PYTHONPATH python code/s3/d0.py run --shard $SHARD --budget-min $BUDGET"
fi

# ---- G-M06＋M07（runs/G_M06_multipath.md）：4 卡 job ------------------------------------------------------------
if [ "$PART" = gm06 ]; then
  [ "$DRY" = 1 ] || [ "$(python3 -c 'import torch;print(torch.cuda.device_count())')" -ge 4 ] || { echo "[r_h200_s3] gm06 要 4 張卡" >&2; exit 2; }
  mkdir -p "$S3/gm06"
  step GM06 m7run s3-gm06 python code/s3/gm06.py run --out-dir "$S3/gm06"
  step GM06-REPORT python3 code/s3/gm06.py report
fi

# ---- W0、G-fio（runs/W0_write_cost.md、runs/G_fio.md）------------------------------------------------------
if [ "$PART" = w0gfio ]; then
  step MODELS ensure_models llama qwen
  step CALIB-llama ensure_calib llama
  step CALIB-qwen ensure_calib qwen
  mkdir -p "$S3/w0/raw" "$S3/gfio"
  for m in llama qwen; do
    step W0-$m m7run s3-w0-$m bash -c "python code/s3/w0.py run --model $m --threads 16 --reps 6 --out $S3/w0/raw/w0_${m}_\$RUN_ID.csv"
  done
  step W0-REPORT python3 code/s3/w0.py report
  step GFIO m7run s3-gfio bash -c "python code/s3/gfio.py run --dir $S3_LOCAL/gfio --gib 16 --total-gib 4 --dir $TIARA_HOME/tmp/gfio --gib 4 --total-gib 2 --reps 6 --out $S3/gfio/gfio_\$RUN_ID.csv"
fi

if [ "$STARTED" = 0 ]; then echo "[r_h200_s3] 沒有叫做 '$FROM' 的步驟。可用的：$SEEN" >&2; exit 2; fi
exit 0
