#!/usr/bin/env bash
# =============================================================================
# m10_verify.sh — 一鍵驗證：精度這條線的程式「能不能跑、算得對不對」（2026-10-15）
#
# 把 MI300X 上做過的檢查（RUNLOG_MI300X 的 Milestone 10-pilot）包成一支，
# 換到 NVIDIA（H200 第一天，或更早在 3090）時一次跑完。這是驗證、不是實驗：
# 輸出放 --out（預設 $TIARA_RUNS/<時間>-m10-verify），**不寫進 results/**。
#
# 用法：  bash code/m10_verify.sh [--quick] [--out <目錄>]
#   --quick   只做 1–4，和 8K 的 5、6（約 10 分鐘〔推論〕）
#   預設      全部（加 Qwen 32K（記憶體 ≥ 60 GiB）、較大的 mixed、BF16 回歸）
#
# 環境（先 source 好再跑；腳本不替你 activate）：
#   H200：  source code/env_h200.sh   （已設 M7_LLAMA_GLOB、M7_QWEN_GLOB、HF_HOME、PAPER_HKV_PYLIBS）
#           另外要設 M10_HOTPOT（HotpotQA parquet；env_h200.sh 沒有）
#   3090：  source /ssd7/hungwei/paper-hkv/venv/vllm/bin/activate
#           export HF_HOME=/ssd7/hungwei/hf-cache PAPER_HKV_PYLIBS=/ssd7/hungwei/paper-hkv/pylibs
#           export M7_LLAMA_GLOB="$HF_HOME/hub/models--unsloth--Llama-3.1-8B-Instruct/snapshots/*"
#           export M7_QWEN_GLOB="$HF_HOME/hub/models--Qwen--Qwen2.5-7B-Instruct-1M/snapshots/*"
#           export M10_HOTPOT=<hotpot_qa_distractor_validation.parquet 的路徑>
#           export CUDA_VISIBLE_DEVICES=$(python code/gpu_guard.py --idle-gpus | awk '{print $1}')   # 共用機：挑空閒的卡
#   沒設 M7_*_GLOB 時用 m10_tasks.MODELS（MI300X 的路徑）。
#   M10_GPU_LOCK=<鎖檔>：每個用 GPU 的步驟都先 flock（和別的 GPU 工作排隊；沒設就不鎖）。
#
# 每一項 PASS／FAIL／SKIP；失敗不中斷，照錄錯誤（<out>/<編號>.log），最後印總表、寫 verify_summary.json。
# 有任何 FAIL → 以 1 結束（CLAUDE.md 規則 2：記錄、停下、回報）。
# =============================================================================
set -u

QUICK=0
OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) QUICK=1; shift ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "不認得的參數：$1（--help 看用法）" >&2; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 2
PY="${PYTHON:-python}"
TS="$(date +%Y%m%d-%H%M%S)"
if [ -z "$OUT" ]; then
  if [ -n "${TIARA_RUNS:-}" ]; then OUT="$TIARA_RUNS/$TS-m10-verify"; else OUT="$HOME/m10_verify/$TS"; fi
fi
mkdir -p "$OUT" || exit 2
STEPS="$OUT/steps.tsv"
: > "$STEPS"

_default_glob() { "$PY" -c "import sys; sys.path.insert(0, 'code'); import m10_tasks as T; print(T.MODELS['$1'])" 2>/dev/null; }
LLAMA="${M7_LLAMA_GLOB:-$(_default_glob llama)}"
QWEN="${M7_QWEN_GLOB:-$(_default_glob qwen)}"
HOTPOT="${M10_HOTPOT:-$("$PY" -c "import sys; sys.path.insert(0, 'code'); import m10_tasks as T; print(T.HOTPOT)" 2>/dev/null)}"
PYLIBS="${PAPER_HKV_PYLIBS:-$("$PY" -c "import sys; sys.path.insert(0, 'code'); import m10_tasks as T; print(T.PYLIBS)" 2>/dev/null)}"
OLD_COMMIT=f6595f0      # 加低精度路徑之前的 m7_restore_harness.py（BF16 回歸的「舊版」）

echo "m10_verify  $(date -Is)  quick=$QUICK"
echo "  repo   $REPO  ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo 'no git'))"
echo "  python $(command -v "$PY")"
echo "  out    $OUT"
echo "  llama  $LLAMA"
echo "  qwen   $QWEN"
echo "  hotpot $HOTPOT"
echo "  pylibs $PYLIBS"
echo "  lock   ${M10_GPU_LOCK:-（不鎖）}"

# ── 小工具 ────────────────────────────────────────────────────────────────
gpu() {  # 等鎖的時間另外印進 log（總表的秒數含等鎖）
  if [ -z "${M10_GPU_LOCK:-}" ]; then "$@"; return; fi
  flock "$M10_GPU_LOCK" bash -c 'echo "[lock] 等了 $(( $(date +%s) - $1 )) 秒" >&2; shift; exec "$@"' _ "$(date +%s)" "$@"
}
_resolve() { ls -d $1 2>/dev/null | sort | tail -1; }      # glob → 最後一個 snapshot（和 m7_model.model_path 同規則）

record() {  # id name status rc secs log note
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6" "$7" >> "$STEPS"
}

run_step() {  # id name note -- cmd...
  local id=$1 name=$2 note=$3; shift 3
  local log="$OUT/$id.log" t0 rc secs
  echo
  echo "── [$id] $name"
  t0=$(date +%s)
  "$@" > "$log" 2>&1
  rc=$?
  secs=$(( $(date +%s) - t0 ))
  if [ $rc -eq 0 ]; then
    echo "   PASS（${secs}s）"
    record "$id" "$name" PASS "$rc" "$secs" "$log" "$note"
  else
    echo "   FAIL（rc=$rc，${secs}s）— $log 末 15 行："
    tail -15 "$log" | sed 's/^/   | /'
    record "$id" "$name" FAIL "$rc" "$secs" "$log" "$note"
  fi
}

skip_step() {  # id name reason
  echo
  echo "── [$1] $2"
  echo "   SKIP：$3"
  record "$1" "$2" SKIP "-" 0 "-" "$3"
}

# ── 0. 前置檢查（缺東西時後面對應的步驟會失敗；這裡先講清楚缺什麼）────────────
prereq() {
  local bad=0 f
  for m in "$LLAMA" "$QWEN"; do
    if [ -n "$(_resolve "$m")" ]; then echo "OK   模型 $(_resolve "$m")"; else echo "缺   模型 $m"; bad=1; fi
  done
  for f in nounlist.txt adjectivelist.txt verblist.txt; do
    if [ -f "$PYLIBS/wonderwords/assets/$f" ]; then echo "OK   $PYLIBS/wonderwords/assets/$f"
    else echo "缺   $PYLIBS/wonderwords/assets/$f（T1–T3、T5 的詞表；pip install --target \$PAPER_HKV_PYLIBS wonderwords）"; bad=1; fi
  done
  if [ -f "$HOTPOT" ]; then echo "OK   $HOTPOT ($(sha256sum "$HOTPOT" | cut -c1-16)…)"
  else
    echo "缺   $HOTPOT（T4；設 M10_HOTPOT。來源與 sha256 見 MI300X 的 /mlsteam/data/tiara/datasets/hotpotqa/README.txt："
    echo "     https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet"
    echo "     sha256 c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6）"
    bad=1
  fi
  if git -C "$REPO" cat-file -e "$OLD_COMMIT:code/m7_restore_harness.py" 2>/dev/null; then echo "OK   git $OLD_COMMIT 在"
  else echo "缺   git commit $OLD_COMMIT（BF16 回歸的舊版；shallow clone 要 git fetch --unshallow）"; bad=1; fi
  if command -v nvidia-smi >/dev/null; then echo "OK   nvidia-smi：$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
  elif command -v rocm-smi >/dev/null || command -v amd-smi >/dev/null; then echo "OK   rocm-smi／amd-smi（AMD）"
  else echo "缺   nvidia-smi 和 rocm-smi 都沒有（gpu_guard 會查不到）"; bad=1; fi
  return $bad
}
run_step 0 "前置檢查（模型、詞表、HotpotQA、舊版 commit、smi）" "" prereq

# ── 1. 環境 ─────────────────────────────────────────────────────────────────
run_step 1 "環境（torch／CUDA／HIP、GPU、記憶體）" "" gpu "$PY" code/m10_verify_summary.py env "$OUT/env.json"
MEM=$(grep -o 'MEM_GIB=[0-9.]*' "$OUT/1.log" | tail -1 | cut -d= -f2)
MEM=${MEM:-0}
MEMI=${MEM%.*}
echo "   GPU 記憶體 ${MEM} GiB"

# ── 2. codec 自測（最重要：GPU 與 CPU 逐位元相同）──────────────────────────────
run_step 2 "codec 自測（GPU＝CPU 逐位元）" "" gpu "$PY" code/m10_codec.py --self-test --out "$OUT/codec_selftest.json"

# ── 3. 出題自測（CPU）──────────────────────────────────────────────────────────
tasks_selftest() { PAPER_HKV_PYLIBS="$PYLIBS" M10_HOTPOT="$HOTPOT" "$PY" -I code/m10_tasks.py --self-test --seeds 1 --no-show --models "$LLAMA,$QWEN"; }
run_step 3 "出題自測（兩個 tokenizer，CPU）" "" tasks_selftest

# ── 4. gpu_guard 的 Slurm 判斷（mock）────────────────────────────────────────
run_step 4 "gpu_guard Slurm 單元測試（mock）" "" "$PY" code/test_gpu_guard_slurm.py

# ── 5. 正確性（03 §R1）───────────────────────────────────────────────────────
corr_step() {  # glob L tag
  M7_MODEL_GLOB="$1" M7_TEST_L="$2" gpu "$PY" code/test_m7_correctness.py "$OUT/correctness_$3.json" \
    && "$PY" code/m10_verify_summary.py corr "$OUT/correctness_$3.json"
}
run_step 5a "正確性 Llama 8K" "" corr_step "$LLAMA" 8192 llama_8k
run_step 5b "正確性 Qwen 8K" "" corr_step "$QWEN" 8192 qwen_8k
if [ $QUICK -eq 1 ]; then
  skip_step 5c "正確性 Qwen 32K" "--quick"
elif [ "${MEMI:-0}" -ge 60 ]; then
  run_step 5c "正確性 Qwen 32K" "" corr_step "$QWEN" 32768 qwen_32k
else
  skip_step 5c "正確性 Qwen 32K" "GPU 記憶體 ${MEM} GiB < 60"
fi

# ── 6. 低精度載入（每個 block 逐位元等於預期）─────────────────────────────────
mixed_step() {  # glob fmt n tag
  M7_MODEL_GLOB="$1" gpu "$PY" code/test_m10_restore.py mixed --fmt "$2" --n "$3" --out "$OUT/mixed_$4.json" \
    && "$PY" code/m10_verify_summary.py mixed "$OUT/mixed_$4.json"
}
if [ $QUICK -eq 1 ]; then NQ=16; NL=16
else NQ=$("$PY" code/m10_verify_summary.py pick-n qwen "$MEM"); NL=$("$PY" code/m10_verify_summary.py pick-n llama "$MEM"); fi
run_step 6a "低精度載入 Qwen fp8_head（n=$NQ）" "n=$NQ" mixed_step "$QWEN" fp8_head "$NQ" qwen_fp8_head
run_step 6b "低精度載入 Qwen int4_kivi（n=$NQ）" "n=$NQ" mixed_step "$QWEN" int4_kivi "$NQ" qwen_int4_kivi
run_step 6c "低精度載入 Llama fp8_head（n=$NL）" "n=$NL" mixed_step "$LLAMA" fp8_head "$NL" llama_fp8_head

# ── 7. BF16 回歸：每個 process 只建一個 Restorer，新舊交替兩輪 ─────────────────
regress_step() {  # glob
  local d="$OUT/regress" old="$OUT/regress/m7_restore_harness_$OLD_COMMIT.py" k impl
  mkdir -p "$d"
  git -C "$REPO" show "$OLD_COMMIT:code/m7_restore_harness.py" > "$old" || return 1
  for k in 1 2; do
    for impl in new old; do
      echo "== $impl$k"
      M7_MODEL_GLOB="$1" gpu "$PY" code/test_m10_restore.py regress --old "$old" --impls "$impl" --reps 5 \
        --out "$d/$impl$k.json" || return 1
    done
  done
  "$PY" code/m10_verify_summary.py regress "$d"
}
if [ $QUICK -eq 1 ]; then
  skip_step 7 "BF16 回歸（新舊還原器）" "--quick"
else
  # 16K 的 Llama 要約 22 GiB〔算術〕；小卡改用 Qwen（新舊比的是同一個模型，判準不變）
  if [ "${MEMI:-0}" -ge 40 ]; then RG="$LLAMA"; RGN=llama; else RG="$QWEN"; RGN=qwen; fi
  run_step 7 "BF16 回歸（新舊還原器，$RGN，每 process 一個）" "model=$RGN" regress_step "$RG"
fi

# ── 8. Slurm 下的 gpu_guard（只有在 job 裡才做）──────────────────────────────
slurm_guard() {
  local lvl
  "$PY" code/gpu_guard.py --host | tee "$OUT/host_contention.json" || return 1
  lvl=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['level'])" "$OUT/host_contention.json") || return 1
  echo "host level = $lvl"
  "$PY" code/gpu_guard.py --check 0 || return 1
  case "$lvl" in UNOBSERVABLE_HOST|QUIET) return 0 ;; *) echo "預期 UNOBSERVABLE_HOST 或 QUIET，得到 $lvl"; return 1 ;; esac
}
if [ -n "${SLURM_JOB_ID:-}" ]; then
  run_step 8 "Slurm：gpu_guard --host／--check 0" "" slurm_guard
else
  skip_step 8 "Slurm：gpu_guard --host／--check 0" "不在 Slurm job 裡"
fi

"$PY" code/m10_verify_summary.py report "$OUT"
