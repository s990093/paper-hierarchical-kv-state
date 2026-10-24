#!/bin/bash
#SBATCH --job-name=r-h200
#SBATCH --partition=h200q
#SBATCH --account=ChihChungHsu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=12:00:00
#SBATCH -o slurm-%x-%j.out
# =============================================================================
# r_h200.sh — H200 上開跑前「先復現第一階段」：03_plan_h200.md §R1–§R3，一個 job 依序跑（不並行，和 MI300X 一樣）
#
#   sbatch code/r_h200.sh                   # 全部（在 repo 根目錄送出）
#   sbatch code/r_h200.sh --from R3-A1      # 從某一步接著跑（48 小時上限、被砍時用；--list 看步驟名）
#   bash   code/r_h200.sh --dry-run         # 只印每一步的指令（任何機器都能跑，不用 GPU、不用 Slurm）
#   bash   code/r_h200.sh --list            # 只印步驟名
#
# 前提（R0 做完）：
#   - code/env_h200.sh 的 M7_DISK 已填（h200_probe.sh 的「建議的 M7_DISK」）；R0_UNSET 時 Guard 會丟錯，這支一開始就擋下。
#   - 模型（Llama、Qwen 固定 revision）已下載到 $HF_HOME；m10_verify 需要的 HotpotQA parquet、wonderwords 詞表已放好。
#
# 每一步走 m7run（env_h200.sh；tiara.sbatch 當記錄殼）→ 一步一個 $TIARA_RUNS/<RUN_ID>/。
# 任何一步失敗 → 照 CLAUDE.md 規則 2：記錄、停下（整個 job 結束，exit 非 0）、回報。
# 例外：R1-verify（m10 一鍵驗證）失敗只記下、不擋第一階段（它驗的是精度這條線的程式）。
#
# ⚠️ 接著跑（--from）之前：m7_write_policy／m7_calib 的 CSV 都是附加寫入。被中斷那一步已寫的列要先處理
#    （每列有 run_id：保留或刪掉都要寫進 RUNLOG_H200），不然會混進 m7_repro_compare 的統計；
#    compare 的 repro_verdict.json 會列出每個 CSV 裡有哪些 run_id，方便檢查。
#
# 和 MI300X 指令的差別（照 03 §R2、§R3；都寫進 RUNLOG_H200）：
#   - 不做 C2、C7-anchor、A2（真實本地 SSD／NFS；這週只用限速層，03 §R2）。
#   - 限速層參數沿用 MI300X 的 tier_params.json（M7_TIER_PARAMS，env_h200.sh），只換 GPU、不換「裝置」。
#   - C5 的 duty 掃 0.1–1.0（MI300X 是 0.2/0.4/0.6/0.8/1.0 再補 0.7），busy_levels.json 由
#     m7_repro_compare.py busy-levels 依「最接近 1.5×、2×」選（MI300X 的那份是手寫的，同一個規則重算得到同樣的 0.4/0.7）。
#   - C7-self 直接用修正後的限速器（MI300X 上第一次是修正前，修正後另存 calib_c7_self_fixed.csv）。
#   - M7_IO_MODEL=share 用 `env M7_IO_MODEL=share` 寫在指令裡（MI300X 的 cmd.sh 沒記到這個變數）。
# =============================================================================
set -uo pipefail

DRY=0; LIST=0; FROM=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --list) LIST=1; DRY=1; shift ;;
    --from) FROM="$2"; shift 2 ;;
    -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
    *) echo "不認得的參數：$1（--help 看用法）" >&2; exit 2 ;;
  esac
done

# ---- 找 repo（sbatch 會把這支複製到 spool；和 tiara.sbatch 同一個順序）----------------------
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
if [ -n "${REPO:-}" ] && [ -f "$REPO/code/env_h200.sh" ]; then :
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/code/env_h200.sh" ]; then REPO=$SLURM_SUBMIT_DIR
elif [ -f "$_here/env_h200.sh" ]; then REPO=$(cd "$_here/.." && pwd)
elif [ -f "$HOME/paper-hierarchical-kv-state/code/env_h200.sh" ]; then REPO=$HOME/paper-hierarchical-kv-state
else echo "[r_h200] 找不到 repo（code/env_h200.sh）。請在 repo 根目錄 sbatch，或先 export REPO" >&2; exit 2
fi
export REPO
[ "$DRY" = 1 ] && export TIARA_SKIP_CONDA=1      # 只印指令時不碰 conda（在沒有 conda 的機器上也能看）
# shellcheck disable=SC1091
source "$REPO/code/env_h200.sh"
if [ "$DRY" = 1 ]; then   # 舊版 env_h200.sh 會 mkdir（10/15 已拿掉）；保留這段，只印指令時不留下空目錄
  for _d in "$M7_RES" "$TIARA_RUNS"; do [ -d "$_d" ] && [ -z "$(ls -A "$_d")" ] && rmdir "$_d"; done
fi
cd "$REPO"
P=code/m7_write_policy.py
R=$M7_RES

# ---- 前置檢查（真的要跑時才擋）----------------------------------------------------------
if [ "$DRY" = 0 ]; then
  [ -n "${SLURM_JOB_ID:-}" ] || { echo "[r_h200] 不在 Slurm job 裡。用 sbatch 送（H200_SLURM §5.1）" >&2; exit 2; }
  if [ "$M7_DISK" = R0_UNSET ] || ! awk -v d="$M7_DISK" '$3==d{f=1} END{exit !f}' /proc/diskstats; then
    echo "[r_h200] M7_DISK=$M7_DISK 不在 /proc/diskstats。先做 R0（code/h200_probe.sh），把建議的 M7_DISK 填進 env_h200.sh 或 export（規則 7：查不到不等於沒有）" >&2
    exit 2
  fi
  [ -f "$M7_TIER_PARAMS" ] || { echo "[r_h200] 找不到 M7_TIER_PARAMS=$M7_TIER_PARAMS" >&2; exit 2; }
  for g in "$M7_LLAMA_GLOB" "$M7_QWEN_GLOB"; do
    compgen -G "$g" >/dev/null || { echo "[r_h200] 找不到模型 $g（H200_SLURM §4.2）" >&2; exit 2; }
  done
  # GpuWatcher 要看的是自己那張卡（gpu_guard.smi_index；H200_SLURM §5.3）：對不上就丟錯，在這裡先擋
  python -c 'import sys; sys.path.insert(0, "code"); from gpu_guard import smi_index; print("[r_h200] smi_index(0) =", smi_index(0))' \
    || { echo "[r_h200] cuda:0 對不到 nvidia-smi 的卡（gpu_guard.smi_index）→ GpuWatcher 會看錯卡。照錄、回報" >&2; exit 2; }
  mkdir -p "$R"
  echo "[r_h200] $(date -Is) job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-?} REPO=$REPO M7_RES=$R M7_DISK=$M7_DISK git=$(git rev-parse --short HEAD)"
fi

# ---- 步驟 ----------------------------------------------------------------------------------
STARTED=1; [ -n "$FROM" ] && STARTED=0
SEEN=""
_gate() {   # --from：還沒到起點的步驟跳過
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
    echo "=== $(date -Is) $name 失敗（exit $rc）→ 停下、回報（CLAUDE.md 規則 2）。接著跑：sbatch code/r_h200.sh --from $name" >&2
    exit $rc
  fi
}
soft_step() {   # 失敗只記下，不停（只給 R1-verify）
  local name=$1; shift
  _gate "$name" || return 0
  if [ "$LIST" = 1 ]; then echo "$name"; return 0; fi
  if [ "$DRY" = 1 ]; then printf '[%s] ' "$name"; printf '%q ' "$@"; echo "   （失敗不擋）"; return 0; fi
  echo "=== $(date -Is) $name"
  "$@" || echo "=== $(date -Is) $name 失敗（不擋第一階段；照錄、回報）" >&2
}
gpu_mem_gib() {   # 這張卡的記憶體（GiB）；查不到回 0（→ 不跑 32K，照錄）
  local m
  m=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | awk '{print int($1/1024)}')
  echo "${m:-0}"
}

# ===== R1　環境與正確性（03 §R1）=====
soft_step R1-verify m7run m7h-verify bash code/m10_verify.sh --quick
step R1-corr-llama-8k m7run m7h-corr-llama-8k env M7_MODEL_GLOB="$M7_LLAMA_GLOB" \
  python code/test_m7_correctness.py "$R/correctness_llama_8k.json"
step R1-corr-qwen-8k  m7run m7h-corr-qwen-8k  env M7_MODEL_GLOB="$M7_QWEN_GLOB" \
  python code/test_m7_correctness.py "$R/correctness_qwen_8k.json"
if [ "$DRY" = 1 ] || [ "$(gpu_mem_gib)" -ge 60 ]; then
  step R1-corr-qwen-32k m7run m7h-corr-qwen-32k env M7_MODEL_GLOB="$M7_QWEN_GLOB" M7_TEST_L=32768 \
    python code/test_m7_correctness.py "$R/correctness_qwen_32k.json"
  R1_FILES=("$R/correctness_llama_8k.json" "$R/correctness_qwen_8k.json" "$R/correctness_qwen_32k.json")
else
  echo "[r_h200] GPU 記憶體 < 60 GiB → 不跑 Qwen 32K（照錄）"
  R1_FILES=("$R/correctness_llama_8k.json" "$R/correctness_qwen_8k.json")
fi
step R1-check m7run m7h-r1-check python code/m7_repro_compare.py r1-check "${R1_FILES[@]}"   # 不過 → 整個停

# ===== R2　第一階段的校準（03 §R2；同樣的指令，見 run_commands.tsv）=====
step R2-C0        m7run m7h-c0 python code/m7_calib.py c0 --out "$R/calib_c0.csv" --reps 5
step R2-C0-duplex m7run m7h-duplex python code/m7_duplex.py "$R/calib_c0_duplex.csv"
step R2-C1        m7run m7h-c1 python code/m7_calib.py c1 --out "$R/calib_c1.csv" --reps 3 --max-chunks 80
step R2-C5        m7run m7h-c5 python $P busy-calib --duties 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
step R2-busy-levels m7run m7h-busy-levels python code/m7_repro_compare.py busy-levels --res "$R"
step R2-C4-t1-same m7run m7h-c4-same-t1 python $P noise --reps 10 --slot t1 --proc-mode same
for i in $(seq 1 10); do
  step R2-C4-t1-restart m7run m7h-c4-restart-t1 python $P noise --reps 1 --slot t1 --proc-mode restart
done
step R2-C7-self   m7run m7h-c7self python code/m7_calib.py c7 --out "$R/calib_c7_self.csv" --reps 3

# ===== R3　第一階段主要結論的復現（03 §R3；判準事先寫死在 03 和 m7_repro_compare.py 檔頭）=====
step R3-A0     m7run m7h-a0 python $P a0 --reps 6
step R3-A1     m7run m7h-a1 env M7_IO_MODEL=share python $P a1 --reps 6
step R3-B-main m7run m7h-b-nfs-gap0 env M7_IO_MODEL=share python $P b --reps 3 --ssd-dev nfs --gap 0 --wl-seed 0 --verify

# ===== R2（續）　C6：vLLM 0.28＋OffloadingConnector；記下實際用的 attention backend（03 §R2 C6 列）=====
step R2-C6 m7run m7h-c6-vllm python code/m7_c6_vllm.py --out "$R/c6_vllm.csv" --reps 6
step R2-C6-backend m7run m7h-c6-backend bash -c \
  'd=$(ls -d "$TIARA_RUNS"/*-m7h-c6-vllm 2>/dev/null | tail -1); test -n "$d" -a -f "$d/vllm_server.log" || { echo "找不到 C6 的 vllm_server.log" >&2; exit 1; }; { echo "source: $d/vllm_server.log"; grep -iE "backend|attention|sampl" "$d/vllm_server.log" | head -40; } | tee "$M7_RES/c6_vllm_backend.txt"'
# 第二個時段的雜訊（MI300X 上 t2 也是在最後量的）
step R2-C4-t2-same m7run m7h-c4-same-t2 python $P noise --reps 10 --slot t2 --proc-mode same
for i in $(seq 1 5); do
  step R2-C4-t2-restart m7run m7h-c4-restart-t2 python $P noise --reps 1 --slot t2 --proc-mode restart
done

# ===== 判定：兩台並排、只比方向（03 §R3）=====
step R-compare m7run m7h-repro-compare bash -c \
  'python code/m7_repro_compare.py compare --mi300x results/m7_write_policy_mi300x --h200 "$M7_RES" --write-summaries --fig-dir "$RUN_DIR/figs"'

if [ "$STARTED" = 0 ]; then
  echo "[r_h200] 沒有叫做 '$FROM' 的步驟。可用的：$SEEN" >&2; exit 2
fi
[ "$DRY" = 0 ] && echo "[r_h200] $(date -Is) 全部完成。結果：$R/repro_verdict.json、$R/repro_vs_mi300x.csv；記進 results/RUNLOG_H200.md 的 §R 一節"
exit 0
