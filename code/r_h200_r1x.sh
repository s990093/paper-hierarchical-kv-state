#!/bin/bash
#SBATCH --job-name=r-h200-r1x
#SBATCH --partition=h200q
#SBATCH --account=ChihChungHsu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=05:00:00
#SBATCH -o slurm-%x-%j.out
# =============================================================================
# r_h200_r1x.sh — 第一階段裡「要先下載模型或改程式」的兩塊，在 H200 上復現（2026-10-16，使用者對兩項都說「好」）
#
#   sbatch --job-name=r-h200-la-f2 --dependency=afterok:<r-h200-all>                     code/r_h200_r1x.sh --part la
#   sbatch --job-name=r-h200-qwen3 --dependency=afterok:<r-h200-all>,afterany:<r-h200-rest> code/r_h200_r1x.sh --part qwen3
#   make submit-r1x                      # 上面兩個一起送，自動接好 dependency
#   bash code/r_h200_r1x.sh --part la --dry-run | --list ；sbatch ... --part la --from F2-sweep（接著跑）
#
#   --part la     第 1 輪的 LongAlpaca-7B（正確性、C1 f(i)、vLLM f(i)）＋F2 能耗 κ（NVML 版：probe→gate→cakee→sweep→final，
#                 再用 LongAlpaca 補一次 sweep）。和 r-h200-rest 平行跑（另一張卡）。
#   --part qwen3  第 1 輪的 Qwen3-30B-A3B（正確性、C1 grouped_mm／eager、vLLM f(i)）。模型約 58 GiB，**等 r-h200-rest 跑完才開始**：
#                 它的 D2 tier50 會把約 35.5 GB 的 fs 層暫存寫在 /work（MI300X 實測 FS_TIER_BYTES＝35,458,647,792，
#                 D2_vllm_drop.md），lab 的 /work 配額是共用的（10/16 早上 1419G／1536G）；兩個疊在一起只剩約 13G。
#                 模型還沒下載時，這個 job 自己下載（R0：compute node 連得到 huggingface.co），下載前先檢查配額。
#
# 指令照 MI300X 的 code/m7_run_round1.sh、round1b、round1d（M7_MODEL_GLOB、M7_EXPERTS_IMPL 照當時）與
# run_commands.tsv 的 F2 指令；只換路徑。差別（寫進 RUNLOG_H200）：
#   - vLLM：MI300X 的 m7_vllm_f.py 用的是 tiara venv 的 vLLM 0.19.1；這裡只有 0.28.0。
#   - F2：amdsmi → NVML（code/m9_f2_energy.py 的 NvSmi）。F2 §2.3 事先寫的計數器檢查（更新週期 ≤ 最短還原的 5%、≥10 s 區段
#     ΔE/Δt 和功率讀值差 ≤5%）在 MI300X 上是看完 probe 手動判的；這裡寫成 m9_f2_analyze.py gate，沒過就停（規則 2）。
#     Cake-E 的切點由 cakee 依 H200 的 probe 與 H200 的 C1 重算（MI300X 是 nfs=33；LongAlpaca nfs=60），用 --cake-e-from 帶進 sweep。
#   - F2 的 f(i)：Llama 用 R2 的 $M7_RES/calib_c1.csv（MI300X 用第一階段的 C1）；LongAlpaca 用這支剛量的 C1，chunk 256 MiB。
#   - 沒有 MI300X 第一版 vllmf（相減法）：程式沒有留下。CPU 的模擬（r1-sim、m7_round1_analyze）之後在 head node 做。
# 前提（和 r_h200_rest.sh 一樣）：R0–R3 過了（$M7_RES/repro_verdict.json 是 REPRODUCED／CHECK），另外要有 R2 的 calib_c1.csv。
# 任何一步失敗 → 停（CLAUDE.md 規則 2）。結果：results/m7_explore_h200/、results/m9_followup_h200/。
# =============================================================================
set -uo pipefail

DRY=0; LIST=0; FROM=""; PART=""
while [ $# -gt 0 ]; do
  case "$1" in
    --part) PART="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --list) LIST=1; DRY=1; shift ;;
    --from) FROM="$2"; shift 2 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "不認得的參數：$1（--help 看用法）" >&2; exit 2 ;;
  esac
done
case "$PART" in la|qwen3) ;; *) echo "[r_h200_r1x] 要指定 --part la 或 --part qwen3" >&2; exit 2 ;; esac

# ---- 找 repo（和 r_h200_rest.sh 同一個順序）--------------------------------------------------
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
if [ -n "${REPO:-}" ] && [ -f "$REPO/code/env_h200.sh" ]; then :
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/code/env_h200.sh" ]; then REPO=$SLURM_SUBMIT_DIR
elif [ -f "$_here/env_h200.sh" ]; then REPO=$(cd "$_here/.." && pwd)
elif [ -f "$HOME/paper-hierarchical-kv-state/code/env_h200.sh" ]; then REPO=$HOME/paper-hierarchical-kv-state
else echo "[r_h200_r1x] 找不到 repo（code/env_h200.sh）。請在 repo 根目錄 sbatch，或先 export REPO" >&2; exit 2
fi
export REPO
[ "$DRY" = 1 ] && export TIARA_SKIP_CONDA=1
# shellcheck disable=SC1091
source "$REPO/code/env_h200.sh"
cd "$REPO"
R=$M7_RES
X=$REPO/results/m7_explore_h200
F2R=$REPO/results/m9_followup_h200
IDS=$F2R/f2_run_ids.tsv        # F2 每一步成功的 run_id（標籤<TAB>run_id）：後面的步驟要用前面的 run_id，--from 接著跑也找得到

# 第 1 輪的模型：固定 revision（migration_20261013/inventory/models.tsv，和 MI300X 同一版）。不放進 env_h200.sh：
# 那支每一步都會被 source，r-h200-all／r-h200-rest 排隊或執行中時不改它
LA_REPO=Yukang/LongAlpaca-7B;               LA_REV=14c462dc860ba1818194ef0b5a31387ba57f0e4d
QM_REPO=Qwen/Qwen3-30B-A3B-Instruct-2507;   QM_REV=0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe
LA=$HF_HOME/hub/models--Yukang--LongAlpaca-7B/snapshots/$LA_REV
QM=$HF_HOME/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/$QM_REV
QM_NEED_GIB=58          # MI300X 上 du 的大小（models.tsv）；code/h200_dl_model.py 下載前檢查配額
QUOTA_MARGIN_GIB=15     # 下載完 lab 至少還要剩這麼多

# ---- 前置檢查（真的要跑時才擋）----------------------------------------------------------
if [ "$DRY" = 0 ]; then
  [ -n "${SLURM_JOB_ID:-}" ] || { echo "[r_h200_r1x] 不在 Slurm job 裡。用 sbatch 送" >&2; exit 2; }
  python - "$R/repro_verdict.json" <<'PY' || { echo "[r_h200_r1x] R0–R3 沒過或還沒跑完（$R/repro_verdict.json）→ 停（03 §R3）" >&2; exit 2; }
import json, sys
v = json.load(open(sys.argv[1]))
print("[r_h200_r1x] R3 overall =", v["overall"])
sys.exit(0 if str(v["overall"]).startswith(("REPRODUCED", "CHECK")) else 1)
PY
  if [ "$PART" = la ]; then
    [ -f "$R/calib_c1.csv" ] || { echo "[r_h200_r1x] 找不到 R2 的 $R/calib_c1.csv（F2 的 f(i)）" >&2; exit 2; }
    [ -d "$LA" ] || { echo "[r_h200_r1x] 找不到 LongAlpaca-7B $LA（head node 先下載：RUNLOG_H200 R-r1x）" >&2; exit 2; }
    compgen -G "$M7_LLAMA_GLOB" >/dev/null || { echo "[r_h200_r1x] 找不到模型 $M7_LLAMA_GLOB" >&2; exit 2; }
    python -c 'import pynvml' || { echo "[r_h200_r1x] 沒有 pynvml（F2 的 NVML 後端）" >&2; exit 2; }
  fi
  python -c 'import sys; sys.path.insert(0, "code"); from gpu_guard import smi_index; print("[r_h200_r1x] smi_index(0) =", smi_index(0))' \
    || { echo "[r_h200_r1x] cuda:0 對不到 nvidia-smi 的卡（gpu_guard.smi_index）→ 停、回報" >&2; exit 2; }
  # Qwen3 的下載要在 compute node 上讀 lab 的配額：先記下這裡的 quota 能不能用（la 先跑，讀不到就改在 head node 下載）
  echo "[r_h200_r1x] quota -gw（$(hostname)）:"; quota -gw 2>&1 | tail -2 || echo "[r_h200_r1x] quota 不能用"
  mkdir -p "$X" "$F2R"
  echo "[r_h200_r1x] $(date -Is) part=$PART job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-?} REPO=$REPO git=$(git rev-parse --short HEAD)"
fi

# ---- 步驟（和 r_h200_rest.sh 同一套：失敗就停、--from 接著跑）-------------------------------
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
    echo "=== $(date -Is) $name 失敗（exit $rc）→ 停下、回報（CLAUDE.md 規則 2）。接著跑：sbatch code/r_h200_r1x.sh --part $PART --from $name" >&2
    exit $rc
  fi
}
idrun() {   # idrun <標籤> <短名> <指令...>：同 m7run，成功才把「標籤<TAB>run_id」記進 $IDS
  local label=$1 short=$2; shift 2
  local id; id="$(date +%Y%m%d-%H%M%S)-$short"
  RUN_ID=$id TIARA_SHORT=$short bash "$REPO/code/tiara.sbatch" env RUN_ID="$id" "$@"; local rc=$?
  [ $rc -eq 0 ] && printf '%s\t%s\n' "$label" "$id" >> "$IDS"
  return $rc
}
rid() {     # rid <標籤>：最後一次成功的 run_id。dry-run 時印佔位字
  if [ "$DRY" = 1 ]; then echo "<$1 的 run_id>"; return 0; fi
  local id; id=$(awk -F'\t' -v l="$1" '$1==l{x=$2} END{print x}' "$IDS" 2>/dev/null)
  [ -n "$id" ] || { echo "[r_h200_r1x] $IDS 裡沒有 $1 的 run_id（那一步還沒成功過）" >&2; echo "MISSING-$1"; return 1; }
  echo "$id"
}

if [ "$PART" = la ]; then
# ===== 第 1 輪：LongAlpaca-7B（MI300X：m7_run_round1.sh 的 ok／c1 被 kill，round1b 的 -fix 重跑；round1d 的 vllm2）=====
# M7_MODEL_GLOB 放在記錄的指令裡（cmd.sh 會記到；MI300X 的 cmd.sh 沒記到，是從腳本補的）
step LA-ok      m7run m7h-r1-ok-longalpaca7b env M7_MODEL_GLOB="$LA" python code/m7_guard_run.py python code/test_m7_correctness.py "$X/correct_longalpaca7b.json"
step LA-c1      m7run m7h-r1-c1-longalpaca7b env M7_MODEL_GLOB="$LA" python code/m7_guard_run.py python code/m7_calib.py c1 --out "$X/calib_c1_longalpaca7b.csv" --max-chunks 72 --reps 3
step LA-vllmf2  m7run m7h-r1-vllmf2-longalpaca7b python code/m7_guard_run.py python code/m7_vllm_f.py "$LA" "$X/vllmf2_longalpaca7b.csv" --max-chunks 62

# ===== F2 能耗 κ（MI300X：run_commands.tsv 10/10 的 f2-probe、f2-cakee、f2-sweep、f2-final、f2-sweep-la7b、f2-final-la7b）=====
step F2-probe   idrun f2-probe m7h-f2-probe python code/m7_guard_run.py python code/m9_f2_energy.py probe --reps 3
step F2-gate    idrun f2-gate  m7h-f2-gate  python code/m9_f2_analyze.py gate  --probe-run "$(rid f2-probe)" --c1 "$R/calib_c1.csv" --res "$F2R"
step F2-cakee   idrun f2-cakee m7h-f2-cakee python code/m9_f2_analyze.py cakee --probe-run "$(rid f2-probe)" --c1 "$R/calib_c1.csv" --res "$F2R"
step F2-sweep   idrun f2-sweep m7h-f2-sweep python code/m7_guard_run.py python code/m9_f2_energy.py sweep --reps 5 --seed 21 --cake-e-from "$F2R/f2_cakee.csv"
step F2-final   idrun f2-final m7h-f2-final python code/m9_f2_analyze.py final --probe-run "$(rid f2-probe)" --sweep-runs "$(rid f2-sweep)" --c1 "$R/calib_c1.csv" --res "$F2R"
# LongAlpaca 補充（不影響判定，F2 §2.2）：Cake-E 切點用 Llama 的 P_b／P_wait＋LongAlpaca 的 f(i)（MI300X 同樣做法，§4.8）
step F2-cakee-la7b idrun f2-cakee-la7b m7h-f2-cakee-la7b python code/m9_f2_analyze.py cakee --tag la7b --probe-run "$(rid f2-probe)" --c1 "$X/calib_c1_longalpaca7b.csv" --chunk-mib 256 --res "$F2R"
step F2-sweep-la7b idrun f2-sweep-la7b m7h-f2-sweep-la7b env M7_MODEL_GLOB="$LA" python code/m7_guard_run.py python code/m9_f2_energy.py sweep --reps 3 --seed 21 --verify 0 --m-grid 8 16 32 48 56 --cake-e-from "$F2R/f2_la7b_cakee.csv"
step F2-final-la7b idrun f2-final-la7b m7h-f2-final-la7b python code/m9_f2_analyze.py final --tag la7b --probe-run "$(rid f2-probe)" --sweep-runs "$(rid f2-sweep-la7b)" --c1 "$X/calib_c1_longalpaca7b.csv" --chunk-mib 256 --res "$F2R"
# MI300X 上手動刪掉的三個檔（F2 §7 第 8 點）：--tag la7b 也會寫出 probe 摘要、計數器驗證（內容是 Llama 的 probe，會誤導）和空的 κ_E
step F2-la7b-clean rm -f "$F2R/f2_la7b_probe_summary.csv" "$F2R/f2_la7b_counter_validation.csv" "$F2R/f2_la7b_kappa_e.csv"
fi

if [ "$PART" = qwen3 ]; then
# ===== 第 1 輪：Qwen3-30B-A3B（MI300X：m7_run_round1.sh 的 ok／c1 grouped／c1 eager；round1d 的 vllm2）=====
step QM-download   m7run m7h-dl-qwen3-30b-a3b python code/h200_dl_model.py "$QM_REPO" "$QM_REV" --need-gib "$QM_NEED_GIB" --margin-gib "$QUOTA_MARGIN_GIB"
step QM-ok         m7run m7h-r1-ok-qwen3-30b-a3b env M7_MODEL_GLOB="$QM" python code/m7_guard_run.py python code/test_m7_correctness.py "$X/correct_qwen3_30b_a3b.json"
step QM-c1-grouped m7run m7h-r1-c1-qwen3-30b-a3b-grouped env M7_MODEL_GLOB="$QM" M7_EXPERTS_IMPL=grouped_mm python code/m7_guard_run.py python code/m7_calib.py c1 --out "$X/calib_c1_qwen3_30b_a3b_grouped.csv" --max-chunks 72 --reps 3
step QM-c1-eager   m7run m7h-r1-c1-qwen3-30b-a3b-eager   env M7_MODEL_GLOB="$QM" M7_EXPERTS_IMPL=eager      python code/m7_guard_run.py python code/m7_calib.py c1 --out "$X/calib_c1_qwen3_30b_a3b_eager.csv" --max-chunks 72 --reps 3
step QM-vllmf2     m7run m7h-r1-vllmf2-qwen3-30b-a3b python code/m7_guard_run.py python code/m7_vllm_f.py "$QM" "$X/vllmf2_qwen3_30b_a3b.csv"
fi

if [ "$STARTED" = 0 ]; then
  echo "[r_h200_r1x] 沒有叫做 '$FROM' 的步驟（--part $PART）。可用的：$SEEN" >&2; exit 2
fi
[ "$DRY" = 0 ] && echo "[r_h200_r1x] $(date -Is) part=$PART 全部完成。結果：$X/、$F2R/；摘要與比方向之後在 CPU 上做，記進 RUNLOG_H200"
exit 0
