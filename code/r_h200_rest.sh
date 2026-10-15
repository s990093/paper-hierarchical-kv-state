#!/bin/bash
#SBATCH --job-name=r-h200-rest
#SBATCH --partition=h200q
#SBATCH --account=ChihChungHsu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=30:00:00
#SBATCH -o slurm-%x-%j.out
# =============================================================================
# r_h200_rest.sh — 第一階段「其餘」的 GPU 實驗在 H200 上復現（2026-10-15，使用者：「先把所有東西都復現，第二階段不動」）
#
#   sbatch --dependency=afterok:<R0–R3 的 job> code/r_h200_rest.sh   # make submit-rest 會自動接在 r-h200-all 後面
#   sbatch code/r_h200_rest.sh --from P2-G1-seed0                    # 從某一步接著跑（--list 看步驟名）
#   bash   code/r_h200_rest.sh --dry-run | --list                    # 只印指令（任何機器都能跑）
#
# 範圍＝r_h200.sh（03 §R1–R3：正確性、校準、A0、A1、B 主設定、C6）之外、MI300X 上付費跑過、只用 Llama-3.1-8B 的 GPU 實驗：
#   P1  B 的 8 個敏感度設定           MI300X：m7_run_phase1f.sh（10/8）       → $M7_RES/b_*.csv
#   P2  08 消融的 GPU 確認（3 個）     MI300X：m7_run_ablate.sh（10/9）        → $M7_RES/b2_share.csv
#   P3  破解第 1 輪：Llama 的 C1、vLLM f(i)、存活設定 seed 0–4（10/9） → results/m7_explore_h200/
#   P4  D2：真實 vLLM 0.28 卸載會不會丟資料（9 個 run＋collect；10/10）  → results/m8_directions_h200/
# 指令照 docs/migration_20261013/inventory/run_commands.tsv 的實際指令（cmd.sh 沒記到的環境變數照當時的腳本補：
#   M7_IO_MODEL、M7_CPU_GIBPS＝3.69／11.6、M7_B2_OUT）；只換路徑。和 R3 一樣：限速層、CPU 速度沿用 MI300X 的值，只換 GPU。
#   D2 的 9 條指令在 run_commands.tsv 裡被 PATH 截斷，參數由 results/m8_directions/d2_cells.csv（cfg、workload、conc、n_sess）
#   與 D2_vllm_drop.md §3.2 還原，其餘用預設值（m8_vllm_drop.py 的 argparse）。
#   vLLM f(i) 只跑現在的 m7_vllm_f.py（MI300X 的第二版 vllmf2；第一版 vllmf 的程式沒有留下）。
#
# 不在這支裡的（寫進 RUNLOG_H200）：
#   - 第 1 輪的 LongAlpaca-7B（13G）、Qwen3-30B-A3B（約 61G）：要先下載，lab 的 /work 配額只剩約 118G（共用），使用者決定。
#   - F2 能耗 κ：m9_f2_energy.py 用 amdsmi 的能量計數器，要先改成 NVML，再重做計數器驗證。
#   - F3（反量化、vLLM 原生 FP8 撈針）：會看到低精度的答對率，第二階段的事先登記凍結前不跑（03 開頭）。
#   - F1：ROCm 專屬（ROCm 上開卸載就換成慢的 attention）。
#   - C2、C7-anchor、A2：量的是 MLSteam 的真實本地 SSD／NFS，H200 上裝置不同，不是復現（03 §R2）。
#   - CPU 就能跑的模擬與分析（D1、D3–D8、F4、F6、F7、m7_sim）：之後在 head node 用 H200 的常數另跑。
#
# 前提：R0–R3（r_h200_all.sh）過了——$M7_RES/repro_verdict.json 是 REPRODUCED／CHECK、busy_levels.json 在。
#   M7_DISK 照 r_h200_all.sh：env_h200.sh 還沒回填時，用 results/hw_h200.json 的建議（必須在 /proc/diskstats）。
# 任何一步失敗 → 停（CLAUDE.md 規則 2）。這支只量測；摘要與「和 MI300X 比方向」之後在 CPU 上做，不覆寫 R3 的判定檔。
# ⚠️ 接著跑（--from）之前：m7_write_policy 的 CSV 是附加寫入，被中斷那一步已寫的列要先處理（同 r_h200.sh 檔頭）。
# =============================================================================
set -uo pipefail

DRY=0; LIST=0; FROM=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --list) LIST=1; DRY=1; shift ;;
    --from) FROM="$2"; shift 2 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "不認得的參數：$1（--help 看用法）" >&2; exit 2 ;;
  esac
done

# ---- 找 repo（和 r_h200.sh 同一個順序）----------------------------------------------------
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
if [ -n "${REPO:-}" ] && [ -f "$REPO/code/env_h200.sh" ]; then :
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/code/env_h200.sh" ]; then REPO=$SLURM_SUBMIT_DIR
elif [ -f "$_here/env_h200.sh" ]; then REPO=$(cd "$_here/.." && pwd)
elif [ -f "$HOME/paper-hierarchical-kv-state/code/env_h200.sh" ]; then REPO=$HOME/paper-hierarchical-kv-state
else echo "[r_h200_rest] 找不到 repo（code/env_h200.sh）。請在 repo 根目錄 sbatch，或先 export REPO" >&2; exit 2
fi
export REPO
[ "$DRY" = 1 ] && export TIARA_SKIP_CONDA=1
# shellcheck disable=SC1091
source "$REPO/code/env_h200.sh"
cd "$REPO"
P=code/m7_write_policy.py
R=$M7_RES
X=$REPO/results/m7_explore_h200
D2=$REPO/results/m8_directions_h200
D2_IDS=$D2/d2_run_ids.txt      # 成功的 D2 run（collect 只收這些；--from 重跑時失敗的不會混進來）

# ---- 前置檢查（真的要跑時才擋）----------------------------------------------------------
if [ "$DRY" = 0 ]; then
  [ -n "${SLURM_JOB_ID:-}" ] || { echo "[r_h200_rest] 不在 Slurm job 裡。用 sbatch 送" >&2; exit 2; }
  python - "$R/repro_verdict.json" <<'PY' || { echo "[r_h200_rest] R0–R3 沒過或還沒跑完（$R/repro_verdict.json）→ 停（03 §R3：不重現就停在那裡回報）" >&2; exit 2; }
import json, sys
v = json.load(open(sys.argv[1]))
print("[r_h200_rest] R3 overall =", v["overall"])
sys.exit(0 if str(v["overall"]).startswith(("REPRODUCED", "CHECK")) else 1)
PY
  [ -f "$R/busy_levels.json" ] || { echo "[r_h200_rest] 找不到 $R/busy_levels.json（R2-C5）" >&2; exit 2; }
  if [ "$M7_DISK" = R0_UNSET ]; then
    M7_DISK=$(python - "$REPO/results/hw_h200.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["storage"]
print(s["suggested_M7_DISK"] if s.get("M7_DISK_in_diskstats") == "yes" and s.get("suggested_M7_DISK") not in (None, "", "NONE", "NOT_MEASURED") else "")
PY
)
    [ -n "$M7_DISK" ] || { echo "[r_h200_rest] R0 沒有給出在 /proc/diskstats 裡的 M7_DISK → 停（規則 7）" >&2; exit 2; }
    export M7_DISK
  fi
  awk -v d="$M7_DISK" '$3==d{f=1} END{exit !f}' /proc/diskstats \
    || { echo "[r_h200_rest] M7_DISK=$M7_DISK 不在 /proc/diskstats → 停（規則 7）" >&2; exit 2; }
  [ -f "$M7_TIER_PARAMS" ] || { echo "[r_h200_rest] 找不到 M7_TIER_PARAMS=$M7_TIER_PARAMS" >&2; exit 2; }
  compgen -G "$M7_LLAMA_GLOB" >/dev/null || { echo "[r_h200_rest] 找不到模型 $M7_LLAMA_GLOB" >&2; exit 2; }
  python -c 'import sys; sys.path.insert(0, "code"); from gpu_guard import smi_index; print("[r_h200_rest] smi_index(0) =", smi_index(0))' \
    || { echo "[r_h200_rest] cuda:0 對不到 nvidia-smi 的卡（gpu_guard.smi_index）→ GpuWatcher 會看錯卡。照錄、回報" >&2; exit 2; }
  mkdir -p "$R" "$X" "$D2"
  echo "[r_h200_rest] $(date -Is) job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-?} REPO=$REPO M7_RES=$R M7_DISK=$M7_DISK git=$(git rev-parse --short HEAD)"
fi

# ---- 步驟（和 r_h200.sh 同一套：失敗就停、--from 接著跑）---------------------------------
STARTED=1; [ -n "$FROM" ] && STARTED=0
SEEN=""
_gate() {
  local name=$1
  SEEN="$SEEN $name"
  if [ "$STARTED" = 0 ]; then
    if [ "$name" = "$FROM" ]; then STARTED=1; else [ "$LIST" = 0 ] && echo "[skip] $name"; return 1; fi
  fi
  return 0
}
step() {    # step <名稱> <指令...>：失敗就停（規則 2）
  local name=$1; shift
  _gate "$name" || return 0
  if [ "$LIST" = 1 ]; then echo "$name"; return 0; fi
  if [ "$DRY" = 1 ]; then printf '[%s] ' "$name"; printf '%q ' "$@"; echo; return 0; fi
  echo "=== $(date -Is) $name"
  "$@"; local rc=$?
  if [ $rc -ne 0 ]; then
    echo "=== $(date -Is) $name 失敗（exit $rc）→ 停下、回報（CLAUDE.md 規則 2）。接著跑：sbatch code/r_h200_rest.sh --from $name" >&2
    exit $rc
  fi
}
d2run() {   # 同 m7run，成功才把 run_id 記進 $D2_IDS（給 collect）
  local short=$1; shift
  local id; id="$(date +%Y%m%d-%H%M%S)-$short"
  RUN_ID=$id TIARA_SHORT=$short bash "$REPO/code/tiara.sbatch" env RUN_ID="$id" "$@"; local rc=$?
  [ $rc -eq 0 ] && echo "$id" >> "$D2_IDS"
  return $rc
}

C="--cpu-fracs 0.25 0.5"

# ===== P1　B 的 8 個敏感度設定（MI300X：m7_run_phase1f.sh 的順序；run_commands.tsv 的實際參數）=====
K="--strategies S1 S2b S4 S4+ S4+P S5 S5s"
S6="--strategies S1 S4 S4+ S4+P S5 S5s"
# shellcheck disable=SC2086
{
step P1-B-rep36   m7run m7h-b-nfs-gap0-r36 env M7_IO_MODEL=share python $P b --reps 3 --rep-start 3 --ssd-dev nfs --gap 0 --wl-seed 0 $C $K
step P1-B-seed1   m7run m7h-b-nfs-seed1    env M7_IO_MODEL=share python $P b --reps 3 --ssd-dev nfs --gap 0 --wl-seed 1 $C $S6
step P1-B-cpuvllm m7run m7h-b-nfs-cpuvllm  env M7_IO_MODEL=share M7_CPU_GIBPS=3.69 python $P b --reps 3 --ssd-dev nfs --gap 0 --wl-seed 0 $C $S6
step P1-B-local   m7run m7h-b-local-gap0   env M7_IO_MODEL=share python $P b --reps 3 --ssd-dev local --gap 0 --wl-seed 0 $C $S6
step P1-B-fifo    m7run m7h-b-nfs-fifo     env M7_IO_MODEL=fifo  python $P b --reps 3 --ssd-dev nfs --gap 0 --wl-seed 0 $C $S6
step P1-B-seed2   m7run m7h-b-nfs-seed2    env M7_IO_MODEL=share python $P b --reps 3 --ssd-dev nfs --gap 0 --wl-seed 2 $C $S6
step P1-B-gap2    m7run m7h-b-nfs-gap2     env M7_IO_MODEL=share python $P b --reps 3 --ssd-dev nfs --gap 2 --wl-seed 0 $C $S6
step P1-B-ssd50   m7run m7h-b-nfs-ssd50    env M7_IO_MODEL=share python $P b --reps 3 --ssd-dev nfs --gap 0 --wl-seed 0 $C --ssd-frac 0.5 $S6

# ===== P2　08 消融的 GPU 確認（MI300X：m7_run_ablate.sh）=====
S8="--strategies S1 S2b S4+ S4L S4B S5 S5L S5c"
step P2-G1-seed0    m7run m7h-b2-g1-doc-hold-cpu3.69-s0 env M7_IO_MODEL=share M7_CPU_GIBPS=3.69 python $P b --b2 --workload doc --release hold --ssd-dev nfs --wl-seed 0 $S8 $C --reps 3 --verify
step P2-G1-seed1    m7run m7h-b2-g1-doc-hold-cpu3.69-s1 env M7_IO_MODEL=share M7_CPU_GIBPS=3.69 python $P b --b2 --workload doc --release hold --ssd-dev nfs --wl-seed 1 $S8 $C --reps 3 --verify
step P2-G2-chat-hold m7run m7h-b2-g2-chat-hold-s0       env M7_IO_MODEL=share python $P b --b2 --workload chat --release hold --ssd-dev nfs --wl-seed 0 --strategies S1 S2b S4+ S4B S5 S5c $C --reps 3 --verify

# ===== P3　破解第 1 輪，只有 Llama（MI300X：m7_run_round1.sh 的 c1-llama、round1d 的 vllm2-llama、round1e）=====
step P3-c1-llama     m7run m7h-r1-c1-llama31-8b     python code/m7_guard_run.py python code/m7_calib.py c1 --out "$X/calib_c1_llama31_8b.csv" --max-chunks 72 --reps 3
step P3-vllmf2-llama m7run m7h-r1-vllmf2-llama31-8b python code/m7_guard_run.py python code/m7_vllm_f.py "$M7_LLAMA_GLOB" "$X/vllmf2_llama31_8b.csv"
for seed in 0 1 2 3 4; do
  step P3-gpu-s$seed m7run m7h-r1-gpu-llama-doc-free-local-s$seed \
    env M7_IO_MODEL=share M7_CPU_GIBPS=11.6 M7_B2_OUT="$X/r1_gpu_llama_doc_free_local.csv" \
    python $P b --b2 --workload doc --release free --ssd-dev local --wl-seed $seed --strategies S1 S2b S4 S4+ S4+P S4L S4B S4W S4C S5L --cpu-fracs 0.5 --reps 3 --verify
done

# ===== P4　D2：真實 vLLM 0.28 的 CPU 卸載（MI300X：10/10 的 9 個 run；D2_vllm_drop.md §3.2）=====
for wl in doc chat; do          # MI300X 的順序：doc 的 off→cpu50→cpu25，再 chat
  for cfg in off cpu50 cpu25; do
    step P4-d2-$cfg-$wl d2run m7h-d2-$cfg-$wl python code/m7_guard_run.py python code/m8_vllm_drop.py --cfg $cfg --workload $wl
  done
done
step P4-d2-cpu4g-doc      d2run m7h-d2-cpu4g-doc      python code/m7_guard_run.py python code/m8_vllm_drop.py --cfg cpu4g  --workload doc --conc 4 16
step P4-d2-tier50-doc     d2run m7h-d2-tier50-doc     python code/m7_guard_run.py python code/m8_vllm_drop.py --cfg tier50 --workload doc --conc 4 --n-sess 8
step P4-d2-cpu50bs512-doc d2run m7h-d2-cpu50bs512-doc python code/m7_guard_run.py python code/m8_vllm_drop.py --cfg cpu50  --workload doc --conc 1 --n-sess 8 --block-size 512
step P4-d2-collect m7run m7h-d2-collect bash -c \
  'python code/m8_vllm_drop.py collect "$0" $(sort -u "$1" | sed "s#^#$TIARA_RUNS/#")' "$D2" "$D2_IDS"
}

if [ "$STARTED" = 0 ]; then
  echo "[r_h200_rest] 沒有叫做 '$FROM' 的步驟。可用的：$SEEN" >&2; exit 2
fi
[ "$DRY" = 0 ] && echo "[r_h200_rest] $(date -Is) 全部完成。結果：$R/b_*.csv、$R/b2_share.csv、$X/、$D2/；摘要與比方向之後在 CPU 上做，記進 RUNLOG_H200"
exit 0
