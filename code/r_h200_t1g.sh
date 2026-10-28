#!/bin/bash
#SBATCH --job-name=icpe-t1g
#SBATCH --partition=h200q
#SBATCH --account=ChihChungHsu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=1-12:00:00
#SBATCH -o slurm-%x-%j.out
# =============================================================================
# r_h200_t1g.sh — ICPE T1G（docs/h200/icpe/T1G_gpu_rerun.md）：CPU 層換成 H200 自己量的常數（T1），重量 GPU 實測的 run。
#
#   sbatch --job-name=icpe-t1g code/r_h200_t1g.sh --part all      # a1pp → a1p → a2 → p02（量時間）
#   sbatch --job-name=icpe-t1g2 code/r_h200_t1g.sh --part a3      # T1G-2（docs/h200/icpe/T1G2_round1_rerun.md）：a3_plan.tsv 的存活設定 × seed 0–4（量時間；不在 all 裡）
#   bash code/r_h200_t1g.sh --part <p> --list | --dry-run ；--from <步驟> 接著跑
#
# 指令和原本的 run 相同，只改：M7_TIER_PARAMS／P3_TIER＝T1 的常數、輸出目錄＝results/icpe_h200/t1/gpu/<part>（登記 §2）。
# 量時間不和我們自己的 job 同時跑（使用者 10/28「照規則，等 kc-* 空檔」）：每一步開跑前等節點上沒有我們自己的別的 job；
# 跑完用 sacct 查這一步的時間窗有沒有重疊 → 有就照重疊重跑，最多 3 次；3 次都重疊 → 記成重疊、不進判定，接著下一步（登記 §4）。
# 這支在 job 跑的時候不要原地改（bash 邊跑邊讀）：寫新檔再 mv 過去。
# =============================================================================
set -uo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
if [ -n "${REPO:-}" ] && [ -f "$REPO/code/env_h200.sh" ]; then :
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/code/env_h200.sh" ]; then REPO=$SLURM_SUBMIT_DIR
elif [ -f "$_here/env_h200.sh" ]; then REPO=$(cd "$_here/.." && pwd)
else echo "[r_h200_t1g] 找不到 repo（code/env_h200.sh）" >&2; exit 2
fi
export REPO
if [ -z "${R_H200_T1G_REEXEC:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  export R_H200_T1G_REEXEC=1
  echo "[r_h200_t1g] 改執行 repo 裡的版本：$REPO/code/r_h200_t1g.sh（git $(git -C "$REPO" rev-parse --short HEAD)）"
  exec bash "$REPO/code/r_h200_t1g.sh" "$@"
fi
PART=""; DRY=0; LIST=0; FROM=""
while [ $# -gt 0 ]; do
  case "$1" in
    --part) PART="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --list) LIST=1; DRY=1; shift ;;
    --from) FROM="$2"; shift 2 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "不認得的參數：$1" >&2; exit 2 ;;
  esac
done
case "$PART" in
  a1pp|a1p|a2|p02|all|a3) ;;
  *) echo "要 --part a1pp|a1p|a2|p02|all|a3" >&2; exit 2 ;;
esac
[ "$DRY" = 1 ] && export TIARA_SKIP_CONDA=1
# 模型權重在 hgx-cn01 本地 /tmp（同 r_h200_icpe.sh）
export ICPE_MODEL_ROOT="${ICPE_MODEL_ROOT:-/tmp/hungweilai-models/hf-cache}"
export HF_HOME="$ICPE_MODEL_ROOT"
unset M7_LLAMA_GLOB M7_QWEN_GLOB M7_MODEL_GLOB
# shellcheck disable=SC1091
source "$REPO/code/env_h200.sh"
cd "$REPO"
ulimit -c 0
T1=$REPO/results/icpe_h200/t1
T1G=$T1/gpu
TP=$T1/tier_params_h200cpu.json
M7H=$REPO/results/m7_write_policy_h200
P01=$REPO/results/icpe_h200/p01
export P3_CONST=h200 X1_TRACE_DIR=$TIARA_HOME/datasets/traces
OVL=$T1G/overlap.csv

if [ "$DRY" = 0 ]; then
  [ -n "${SLURM_JOB_ID:-}" ] || { echo "[r_h200_t1g] 不在 Slurm job 裡" >&2; exit 2; }
  [ -f "$TP" ] || { echo "[r_h200_t1g] 找不到 $TP" >&2; exit 2; }
  python -c 'import sys; sys.path.insert(0, "code"); from gpu_guard import smi_index; print("[r_h200_t1g] smi_index(0) =", smi_index(0))' \
    || { echo "[r_h200_t1g] cuda:0 對不到 nvidia-smi 的卡 → 停" >&2; exit 2; }
  python3 "$REPO/code/s3/fetch_models.py" check --root "$ICPE_MODEL_ROOT" --models llama qwen \
    || { echo "[r_h200_t1g] 模型權重不在 $ICPE_MODEL_ROOT → 停（exit 3）" >&2; exit 3; }
  want=$(python3 "$REPO/code/s3/fetch_models.py" env --root "$ICPE_MODEL_ROOT" --models llama qwen | sort)
  have=$(printf 'export HF_HOME=%s\nexport M7_LLAMA_GLOB=%s\nexport M7_QWEN_GLOB=%s\n' "$HF_HOME" "$M7_LLAMA_GLOB" "$M7_QWEN_GLOB" | sort)
  [ "$want" = "$have" ] || { echo "[r_h200_t1g] env_h200.sh 的模型路徑和 fetch_models 的不同 → 停" >&2; exit 2; }
  awk -v d="$M7_DISK" '$3==d{f=1} END{exit !f}' /proc/diskstats || { echo "[r_h200_t1g] M7_DISK=$M7_DISK 不在 /proc/diskstats → 停" >&2; exit 2; }
  python3 - "$TIARA_HOME/datasets/traces" <<'PY' || { echo "[r_h200_t1g] Mooncake trace 的 sha256 不對 → 停" >&2; exit 2; }
import hashlib, sys
got = hashlib.sha256(open(f"{sys.argv[1]}/conversation_trace.jsonl", "rb").read()).hexdigest()
print("[r_h200_t1g] conversation_trace.jsonl", got[:16]); assert got.startswith("b8cbb061a85206d7"), got
PY
  echo "[r_h200_t1g] $(date -Is) part=$PART job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-?} git=$(git rev-parse --short HEAD) TP=$TP"
  echo "[r_h200_t1g] 同節點的 job（開跑時）："; squeue -h -w "${SLURMD_NODENAME:-$(hostname -s)}" -o '%i %u %a %b %T %M'
  nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.used --format=csv
  mkdir -p "$T1G/a1pp" "$T1G/a1p" "$T1G/a2" "$T1G/p02" "$T1G/a3"
fi

STARTED=1; [ -n "$FROM" ] && STARTED=0
SEEN=""; OVL_STEPS=""
step() {   # step <名稱> <m7run 短名> <指令...>：等節點沒有自己人 → 跑 → 查重疊；重疊就重跑（最多 3 次）
  local name=$1 short=$2; shift 2
  SEEN="$SEEN $name"
  if [ "$STARTED" = 0 ]; then if [ "$name" = "$FROM" ]; then STARTED=1; else return 0; fi; fi
  if [ "$LIST" = 1 ]; then echo "$name"; return 0; fi
  if [ "$DRY" = 1 ]; then printf '[%s] ' "$name"; printf '%q ' "$@"; echo; return 0; fi
  local k rc orc t0 t1
  for k in 1 2 3; do
    python3 code/icpe/t1g.py wait-clear --job "$SLURM_JOB_ID" --max-min 360 || { echo "=== $name：等不到節點空出來 → 停、回報" >&2; exit 2; }
    t0=$(date -Is); echo "=== $t0 $name（第 $k 次）"
    "$@"; rc=$?
    t1=$(date -Is)
    if [ $rc -ne 0 ] && [ $rc -ne 3 ]; then
      echo "=== $t1 $name 失敗（exit $rc）→ 停下、回報（規則 2）。接著跑：sbatch ... code/r_h200_t1g.sh --part $PART --from $name" >&2; exit $rc
    fi
    python3 code/icpe/t1g.py overlap --job "$SLURM_JOB_ID" --start "$t0" --end "$t1" --step "$name" --short "$short" --log "$OVL"; orc=$?
    [ $orc -eq 2 ] && { echo "=== $name：sacct 查不到重疊 → 停（規則 7）" >&2; exit 2; }
    if [ $orc -eq 0 ] && [ $rc -eq 0 ]; then return 0; fi
    echo "=== $(date -Is) $name：$([ $rc -eq 3 ] && echo 'gpu_guard 污染' || echo '和我們自己的 job 重疊') → 照重疊重跑（登記 §4）"
  done
  echo "=== $(date -Is) $name：3 次都重疊或污染 → 記成重疊、不進判定，接著下一步（登記 §4）" >&2
  OVL_STEPS="$OVL_STEPS $name"
  return 0
}
want_part() { { [ "$PART" = all ] && [ "$1" != a3 ]; } || [ "$PART" = "$1" ]; }

# ============================================================== A1''：B 主設定、fa3（原 run 20261022-210344-icpe-p01a-b-nfs-gap0）
if want_part a1pp; then
  step T1G-A1PP icpe-t1g-a1pp-b-nfs-gap0 m7run icpe-t1g-a1pp-b-nfs-gap0 env M7_TIER_PARAMS="$TP" M7_ATTN=fa3 M7_F_CSV="$P01/calib/calib_c1.csv" \
    M7_RES="$T1G/a1pp" M7_IO_MODEL=share python code/m7_write_policy.py b --reps 6 --ssd-dev nfs --gap 0 --wl-seed 0 --verify
fi
# ============================================================== A1'：同上、sdpa（原 run 20261016-063929-m7h-b-nfs-gap0＋…-172445-…-r36-rr；登記 §2 改成一次 6 rep）
if want_part a1p; then
  step T1G-A1P icpe-t1g-a1p-b-nfs-gap0 m7run icpe-t1g-a1p-b-nfs-gap0 env M7_TIER_PARAMS="$TP" M7_F_CSV="$M7H/calib_c1.csv" \
    M7_RES="$T1G/a1p" M7_IO_MODEL=share python code/m7_write_policy.py b --reps 6 --ssd-dev nfs --gap 0 --wl-seed 0 --verify
fi
# ============================================================== A2：08 消融（原 run 20261016-202603-…-s0-rr、…-132601-…-s1、…-142019-…-g2-chat-hold-s0）
if want_part a2; then
  for s in 0 1; do
    step T1G-A2-G1-s$s icpe-t1g-a2-b2-g1-doc-hold-cpu3.69-s$s m7run icpe-t1g-a2-b2-g1-doc-hold-cpu3.69-s$s env M7_TIER_PARAMS="$TP" \
      M7_F_CSV="$M7H/calib_c1.csv" M7_RES="$T1G/a2" M7_IO_MODEL=share M7_CPU_GIBPS=3.69 python code/m7_write_policy.py b --b2 --workload doc \
      --release hold --ssd-dev nfs --wl-seed $s --strategies S1 S2b S4+ S4L S4B S5 S5L S5c --cpu-fracs 0.25 0.5 --reps 3 --verify
  done
  step T1G-A2-G2-s0 icpe-t1g-a2-b2-g2-chat-hold-s0 m7run icpe-t1g-a2-b2-g2-chat-hold-s0 env M7_TIER_PARAMS="$TP" M7_F_CSV="$M7H/calib_c1.csv" \
    M7_RES="$T1G/a2" M7_IO_MODEL=share python code/m7_write_policy.py b --b2 --workload chat --release hold --ssd-dev nfs --wl-seed 0 \
    --strategies S1 S2b S4+ S4B S5 S5c --cpu-fracs 0.25 0.5 --reps 3 --verify
fi
# ============================================================== B6／C2：P0-2 的 GPU 重播（k＝8 固定；原 job 21655）
if want_part p02; then
  export P3_TIER=h200cpu M7_TIER_PARAMS="$TP"
  step T1G-P02-SMOKE icpe-t1g-p02-smoke m7run icpe-t1g-p02-smoke python code/icpe/x1_gpu.py gpu --out "$T1G/p02" --k 8 --cpu 128 --policy LRU --max-req 100 --csv gpu_smoke.csv
  # 12 個 run 的順序：和原本相同（固定 seed 20261023 打亂）
  ORDER=$(python3 - <<'PY'
import random
runs = [(c, p, 0) for c in (128, 256) for p in ("LRU", "FREQ2Q", "O-BEL", "HINT-PF", "O-PF")] + [(128, "LRU", 1), (256, "LRU", 1)]
random.Random(20261023).shuffle(runs)
for c, p, r in runs:
    print(c, p, r)
PY
)
  while read -r c pol rep; do
    step "T1G-P02-$c-$pol-r$rep" "icpe-t1g-p02-$c-$pol-r$rep" m7run "icpe-t1g-p02-$c-$pol-r$rep" python code/icpe/x1_gpu.py gpu --out "$T1G/p02" --k 8 --cpu "$c" --policy "$pol" --rep "$rep"
  done <<< "$ORDER"
  unset P3_TIER
fi

# ============================================================== A3（T1G-2）：第 1 輪存活設定的 GPU 確認（同 code/r_h200_r1h.sh 的做法）
if want_part a3; then
  PLAN=$T1G/a3/a3_plan.tsv
  [ -f "$PLAN" ] || { echo "[r_h200_t1g] 沒有 $PLAN（先在 head node 跑 t1g.py a3-plan）" >&2; exit 2; }
  CTRL="S1 S2b S4 S4+ S4+P S4L S4B S4W S4C"
  while IFS=$'\t' read -r tag over rel ssd frac wt _rest; do
    case "$tag" in ''|'#'*) continue ;; esac
    OUTA=$T1G/a3/r1h_gpu_llama_doc_$tag.csv
    if [ "$DRY" = 0 ] && [ -z "$FROM" ] && [ -e "$OUTA" ]; then echo "[r_h200_t1g] $OUTA 已經存在（附加寫入會混在一起）→ 停" >&2; exit 2; fi
    CPUENV=""; [ "$over" != "-" ] && CPUENV="M7_CPU_GIBPS=$over"
    for seed in 0 1 2 3 4; do
      step "T1G-A3-$tag-s$seed" "icpe-t1g2-r1h-$tag-s$seed" m7run "icpe-t1g2-r1h-$tag-s$seed" env M7_TIER_PARAMS="$TP" M7_F_CSV="$M7H/calib_c1.csv" \
        M7_RES="$T1G/a3" M7_IO_MODEL=share $CPUENV M7_B2_OUT="$OUTA" python code/m7_write_policy.py b --b2 --workload doc --release "$rel" \
        --ssd-dev "$ssd" --wl-seed $seed --strategies $CTRL "$wt" --cpu-fracs "$frac" --reps 3 --verify
    done
  done < <(grep -v '^#' "$PLAN")
fi

if [ "$STARTED" = 0 ]; then echo "[r_h200_t1g] 沒有叫做 '$FROM' 的步驟。可用的：$SEEN" >&2; exit 2; fi
if [ "$DRY" = 0 ]; then
  echo "[r_h200_t1g] $(date -Is) part=$PART 完成；3 次都重疊的步驟：${OVL_STEPS:-無}"
fi
exit 0
