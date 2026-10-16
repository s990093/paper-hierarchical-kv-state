#!/bin/bash
#SBATCH --job-name=m10h-x
#SBATCH --partition=h200q
#SBATCH --account=ChihChungHsu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=48:00:00
#SBATCH -o slurm-%x-%j.out
# =============================================================================
# r_h200_x.sh — 第二階段「主線以外」在 H200 上復現（13_h200_repro.md §8、§9；清單在 RUNLOG_H200 Milestone 10-H200）
#
#   --part q3 --model llama|qwen   只量分數（1 張卡，工作池）：41 大掃描海選（n＝24，同 MI300X）、45 FP8 混合、55 T1＋LongBench、
#                                  60 敏感度地圖、88 INT4 的 T1＋LongBench；llama 另加上游 RULER CWE（m10_ruler_cwe_check、hfgen）與 75（vLLM）
#   --part t2                      量時間（1 張卡，不和我們自己的 job 同時跑）：39 的 GPU 自測、40 反量化微測、42 融合反量化的 CPU 層格子、
#                                  42b／42c 反量化方法掃描、44 兩層混合還原、70 vLLM TTFT、43 複賽的冒煙與速度
#   --part q5                      只量分數：43 複賽的品質（m10_confirm plan 寫的 12 個 process）
#   --dry-run | --list | --from <步驟>
#
# sbatch 送出時會把這支複製一份；開跑時改執行 repo 裡的這支（R_H200_X_REEXEC），所以送出之後補的程式在開跑時才生效。
# 停止：任何一步失敗（規則 2）、量時間的步驟 gpu_guard 不是 CLEAN（CLAUDE.md §3）。
# =============================================================================
set -uo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
if [ -n "${REPO:-}" ] && [ -f "$REPO/code/env_h200.sh" ]; then :
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/code/env_h200.sh" ]; then REPO=$SLURM_SUBMIT_DIR
elif [ -f "$_here/env_h200.sh" ]; then REPO=$(cd "$_here/.." && pwd)
else echo "[r_h200_x] 找不到 repo（code/env_h200.sh）" >&2; exit 2
fi
export REPO
if [ -z "${R_H200_X_REEXEC:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  export R_H200_X_REEXEC=1
  echo "[r_h200_x] 改執行 repo 裡的版本：$REPO/code/r_h200_x.sh（git $(git -C "$REPO" rev-parse --short HEAD)）"
  exec bash "$REPO/code/r_h200_x.sh" "$@"
fi

PART=""; MODEL=""; DRY=0; LIST=0; FROM=""
while [ $# -gt 0 ]; do
  case "$1" in
    --part) PART="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --list) LIST=1; DRY=1; shift ;;
    --from) FROM="$2"; shift 2 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "不認得的參數：$1" >&2; exit 2 ;;
  esac
done
case "$PART" in
  q3) case "$MODEL" in llama|qwen) ;; *) echo "--part q3 要 --model llama|qwen" >&2; exit 2 ;; esac ;;
  t2|q5) ;;
  *) echo "要 --part q3|t2|q5" >&2; exit 2 ;;
esac

[ "$DRY" = 1 ] && export TIARA_SKIP_CONDA=1
# shellcheck disable=SC1091
source "$REPO/code/env_h200.sh"
cd "$REPO"
ulimit -c 0

OUT=$REPO/results/m10_precision_h200
SW=$REPO/results/m10_sweep_h200
TO=$REPO/results/m10_tier_h200
P=$REPO/results/m10_pilot_mi300x/e0_tune/CANDIDATE_PARAMS.json
BUSY=$REPO/results/m7_write_policy_h200/busy_levels.json
export M10_BUSY_LEVELS=$BUSY                # m10_dqsweep／m10_tier2／m10_confirm 的 GPU 忙（沒設時是 MI300X 的；13 §8）
export M10_SWEEP_DIR=$SW M10_PRECISION_DIR=$OUT   # m10_confirm／m10_dqsweep／m10_confirm_analyze 的目錄（同上）
SWRAW=${SWRAW:-$TIARA_HOME/sweep_m10h}      # 海選逐列 CSV（大，不進 git）
RULER=$TIARA_DATASETS/ruler_ref             # README.txt：c3f5e3b＋LFS 詞表（sha256 affcd6d4…）＋shim
WORK=$TIARA_RUNS/_m10h_work/x-$PART${MODEL:+-$MODEL}      # 固定路徑：--from 接著跑時讀得到前一個 job 寫的檔
LB="LB-narrativeqa LB-qasper LB-multifieldqa_en LB-hotpotqa LB-2wikimqa LB-musique"

if [ "$DRY" = 0 ]; then
  [ -n "${SLURM_JOB_ID:-}" ] || { echo "[r_h200_x] 不在 Slurm job 裡" >&2; exit 2; }
  python - "$OUT/fmt_select/fmt_vs_mi300x.json" <<'PY' || { echo "[r_h200_x] 格式還沒選或和 MI300X 不同（13 §3-4）→ 停" >&2; exit 2; }
import json, sys
sys.exit(0 if json.load(open(sys.argv[1]))["same"] else 1)
PY
  echo "[r_h200_x] 和 MI300X（bcfe5b1）不同的程式（13 §8：只記錄）："; git diff --stat bcfe5b1 -- code | tail -20
  mkdir -p "$WORK/queue" "$SWRAW"
  echo "[r_h200_x] $(date -Is) part=$PART model=${MODEL:-} job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-?} git=$(git rev-parse --short HEAD)"
  nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.used --format=csv
fi

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
step() {
  local name=$1; shift
  _gate "$name" || return 0
  if [ "$LIST" = 1 ]; then echo "$name"; return 0; fi
  if [ "$DRY" = 1 ]; then printf '[%s] ' "$name"; printf '%q ' "$@"; echo; return 0; fi
  echo "=== $(date -Is) $name"
  "$@"; local rc=$?
  if [ $rc -ne 0 ]; then
    echo "=== $(date -Is) $name 失敗（exit $rc）→ 停下、回報（規則 2）。接著跑：sbatch ... code/r_h200_x.sh --part $PART ${MODEL:+--model $MODEL }--from $name" >&2
    exit $rc
  fi
}
lastrun() { ls -d "$TIARA_RUNS"/*-"$1" | sort | tail -1; }
fmt_of() { python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['choice'][sys.argv[2]][sys.argv[3]])" "$OUT/fmt_select/fmt_choice.json" "$1" "$2"; }
pool() { (cd "$WORK" && python "$REPO/code/h200_pool.py" "$1" "$2"); }
guard_files_clean() {
  python - "$@" <<'PY'
import json, sys
fs = sys.argv[1:]
bad = [f for f in fs if json.load(open(f)).get("verdict") != "CLEAN" or json.load(open(f)).get("contaminated")]
print(f"[guard] {len(fs)} 份 gpu_guard JSON，不是 CLEAN 的：{bad or '無'}")
sys.exit(0 if fs and not bad else 1)
PY
}

# ===== q3：只量分數，一個模型一張卡 =====
q3_jobs() {       # 41（海選，n＝24、MI300X 的「全部」組合）、45、55、60、88 的指令（照 MI300X 的 queue 腳本），寫進一個工作池
  local m=$MODEL grid
  grid="$(lastrun m10h-grid-fp8-$m)/blocks.csv"
  [ -f "$grid" ] && [ -f "$SW/parts/parts_$m.json" ] || { echo "缺 $grid 或 $SW/parts/parts_$m.json（20701／20704）"; return 1; }
  mkdir -p "$OUT/quality" "$OUT/quality_sealed"
  python3 - "$m" "$P" "$OUT" "$SW" "$SWRAW" "$grid" "$LB" "$WORK/jobs_q3.json" <<'PY' || return 1
import json, sys
m, P, OUT, SW, RAW, grid, LB, J = sys.argv[1:9]
ch = json.load(open(f"{OUT}/fmt_select/fmt_choice.json"))["choice"][m]
G, I = ch["fp8"], ch["int4"]; g = G[4:]
need = {"sweep": 40, "s1": 42, "x": 32} if m == "llama" else {"sweep": 30, "s1": 28, "x": 24}
POL = "F M B R0 R1 R2 R3 R4 ERR ATT AXE ORC"
FR = " ".join(str(round(0.10 + 0.05 * i, 2)) for i in range(17))
jobs = []
# 41 海選：09 §4.1；MI300X 的 sweep_plan_n 選了 n＝24、全部組合（13 §9：H200 用同樣的 n）
for s in range(2):
    jobs.append(dict(name=f"sws-{m}-s{s}", need_gb=need["sweep"],
                     cmd=f"m7run m10h-sweep-screen-{m}-s{s} python code/m10_sweep.py screen --model {m} --params-file {P} --n 24 "
                         f"--shard {s}/2 --policies {POL} --fracs {FR} --parts {SW}/parts/parts_{m}.json --grid-blocks {grid} "
                         f"--out {RAW}/screen_{m}_s{s}.csv.gz"))
q8 = f"python code/m10_quality.py run --model {m} --params-file {P} --fmt {G} --bws b32 --meets {OUT}/speed_{m}/meets_fp8tier.json"
q4 = f"python code/m10_quality.py run --model {m} --params-file {P} --fmt {G} --bws b32 --meets {OUT}/speed_{m}/meets.json"
# 45：FP8 混合配置（只報告、封存）
for s in range(6):
    jobs.append(dict(name=f"f8m-{m}-T24-s{s}", need_gb=need["x"], cmd=f"m7run m10h-f8m-{m}-T24-s{s} {q8} --configs F25 M25 B25 R25 --modes A B --tasks T2 T3 T4 --n 180 --shard {s}/6 --out {OUT}/quality_sealed/fp8mix_{m}_T24_s{s}.csv"))
jobs.append(dict(name=f"f8m-{m}-T5", need_gb=need["x"], cmd=f"m7run m10h-f8m-{m}-T5 {q8} --configs F25 M25 B25 R25 --modes A B --tasks T5 --n 60 --out {OUT}/quality_sealed/fp8mix_{m}_T5.csv"))
# 55：Step 1 配置的 T1、LongBench；FP8 混合的 T1
s1 = f"--configs E0 E1 E1s E1sr E1s@fp8k0_{g} E1s@int8_token"
for s in range(2):
    jobs.append(dict(name=f"s1-{m}-T1-s{s}", need_gb=need["s1"], cmd=f"m7run m10h-s1-{m}-T1-s{s} {q8} {s1} --modes A B --tasks T1 --n 180 --shard {s}/2 --out {OUT}/quality/step1_{m}_T1_s{s}.csv"))
    jobs.append(dict(name=f"s1-{m}-LB-s{s}", need_gb=need["s1"], cmd=f"m7run m10h-s1-{m}-LB-s{s} {q8} {s1} --modes A --tasks {LB} --lb-n 50 --shard {s}/2 --out {OUT}/quality/step1_{m}_LB_s{s}.csv"))
jobs.append(dict(name=f"f8m-{m}-T1", need_gb=need["s1"], cmd=f"m7run m10h-f8m-{m}-T1 {q8} --configs F25 M25 B25 R25 --modes A B --tasks T1 --n 180 --out {OUT}/quality_sealed/fp8mix_{m}_T1.csv"))
# 88：INT4 端點與混合配置的 T1、LongBench（H200 兩個模型的 INT4 混合都跑了，13 §4）
i4 = f"--configs E1@{I} E1s@{I} E1sr@{I}"
jobs.append(dict(name=f"i4e-{m}-T1", need_gb=need["s1"], cmd=f"m7run m10h-i4e-{m}-T1 {q4} {i4} --modes A B --tasks T1 --n 180 --out {OUT}/quality_sealed/int4end_{m}_T1.csv"))
jobs.append(dict(name=f"i4e-{m}-LB", need_gb=need["s1"], cmd=f"m7run m10h-i4e-{m}-LB {q4} {i4} --modes A --tasks {LB} --lb-n 50 --out {OUT}/quality_sealed/int4end_{m}_LB.csv"))
jobs.append(dict(name=f"i4m-{m}-T1", need_gb=need["s1"], cmd=f"m7run m10h-i4m-{m}-T1 {q4} --configs F25@{I} M25@{I} B25@{I} R25@{I} --modes A B --tasks T1 --n 180 --out {OUT}/quality_sealed/int4mix_{m}_T1.csv"))
# 60：逐 block 敏感度地圖（兩階的格式）
for tier, F in (("fp8", G), ("int4", I)):
    for s in range(2):
        jobs.append(dict(name=f"sens-{m}-{F}-s{s}", need_gb=need["x"], cmd=f"m7run m10h-sens-{m}-{tier}-s{s} python code/m10_quality.py sensmap --model {m} --fmt {F} --params-file {P} --n 10 --shard {s}/2 --out {OUT}/quality_sealed/sensmap_{m}_{F}_s{s}.csv"))
json.dump(jobs, open(J, "w"), indent=1)
print(f"[q3] {m}: {len(jobs)} 個 process（41 海選 2、45 七、55 五、88 三、60 四）")
PY
}
q3_pool() { q3_jobs && pool "$WORK/jobs_q3.json" 4; }
sweep_analyze() { mkdir -p "$SW/analysis"; m7run m10h-sweep-analyze-$MODEL python code/m10_sweep_analyze.py --model $MODEL --csv "$SWRAW"/screen_${MODEL}_s*.csv.gz --out-dir "$SW/analysis"; }
cwe_gen() {       # 10-pilot 的上游 RULER CWE（MI300X run 20261015-061335）：產生上游 jsonl＋三組 E0
  mkdir -p "$OUT/ruler_cwe"
  m7run m10h-ruler-cwe-llama bash -c "python code/m10_ruler_cwe_check.py --ruler $RULER/RULER_lfs --shim $RULER/shim --raw \$TIARA_RUNS/\$RUN_ID/ruler_cwe --out $OUT/ruler_cwe/ruler_cwe_upstream_llama.csv"
}
cwe_jsonl() { local d; d=$(lastrun m10h-ruler-cwe-llama); echo "$d/ruler_cwe/cwe/validation.jsonl"; }
cwe_hfgen() { m7run m10h-ruler-cwe-hfgen python code/m10_ruler_cwe_check.py --hf-check 10 --reuse-jsonl "$(cwe_jsonl)" --out "$OUT/ruler_cwe/ruler_cwe_upstream_llama.csv"; }
cwe_vllm() { mkdir -p "$OUT/ruler_cwe_vllm"; m7run m10h-cwe-vllm python code/m10_ruler_cwe_vllm.py --jsonl "$(cwe_jsonl)" --out "$OUT/ruler_cwe_vllm/cwe_vllm_llama.csv"; }

# ===== t2：量時間 =====
gpu_selftests() { # 39 的 GPU 部分：融合反量化逐位元自測、Llama 大小緩衝的 offset、dqsweep 的還原路徑（NVIDIA 第一次跑 Triton）
  mkdir -p "$SW/dqsweep"
  m7run m10h-dqfast-selftest python code/m10_dequant_fast.py --self-test --out "$SW/dqfast_selftest_gpu.json" || return 1
  m7run m10h-dqoffset python code/m10_queue_mi300x/tests/dq_fast_offset_gpu.py || return 1
  local m; for m in qwen llama; do
    m7run m10h-dqsweep-smoke-$m python code/m10_dqsweep.py smoke --model $m --fmts int4_kivi "$(fmt_of $m fp8)" --sw "$SW/dqsweep" || return 1
  done
}
dq_bench() { mkdir -p "$SW/parts"; m7run m10h-dqfast-bench python code/m10_dequant_fast.py --self-test --bench --out "$SW/parts/dequant_fast_bench.json"; }
dqfast_grid() {   # 42：融合反量化之後，主線 8 配置在真實 CPU 層（FP8 主線粒度、INT4 KIVI）
  local m=$1 tier=$2 g f d rid
  g=$(fmt_of $m fp8); f=$(fmt_of $m $tier); d=$SW/speed_dqfast$([ $tier = int4 ] && echo _int4)_$m; mkdir -p "$d"
  M10_DEQUANT_FAST=1 m7run m10h-dqfast-grid-$tier-$m bash -c "python code/m10_speed.py grid --model $m --fmt $f --base-fmt $g --out-dir $d --bw-json $OUT/speed_$m/bw.json --busy-levels $BUSY --bw-names cpu --lengths --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" || return 1
  rid=$(basename "$(lastrun m10h-dqfast-grid-$tier-$m)")
  guard_files_clean "$d"/gpu_guard_${rid}_L*.json
}
dqsweep_A() { local m=$1; mkdir -p "$SW/dqsweep/A"; m7run m10h-dqs-A-$m python code/m10_dqsweep.py kernels --model $m --out-dir "$SW/dqsweep/A"; }
dqsweep_B() {     # 42b B 段：每個方法一次 grid（int4_kivi、CPU 層、E0 E1 R25、1 launch × 10）；失敗只記，不擋其他方法（同 MI300X）
  local m=$1 g meth
  g=$(fmt_of $m fp8); mkdir -p "$SW/dqsweep/B"
  for meth in $(python3 code/m10_dqsweep.py list B); do
    ( export $(python3 code/m10_dqsweep.py env $meth --model $m --fmt int4_kivi --sw "$SW/dqsweep")
      m7run m10h-dqs-B-$m-$meth bash -c "python code/m10_speed.py grid --model $m --fmt int4_kivi --base-fmt $g --out-dir $SW/dqsweep/B/${m}_$meth --bw-json $OUT/speed_$m/bw.json --busy-levels $BUSY --bw-names cpu --configs E0 E1 R25 --refs --lengths --launches 1 --reps 10 --launch0 100 --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" ) \
      || echo "dqsweep B $m $meth 失敗（runs/*-m10h-dqs-B-$m-$meth）" | tee -a "$WORK/NEEDS_HUMAN"
  done
  return 0
}
h200_lib() {      # m10_dqsweep plan／m10_confirm plan 寫的腳本會 source <jobs>/lib.sh：給它 H200 的版本
  cat > "$WORK/lib.sh" <<EOF
source $REPO/code/env_h200.sh
REPO=$REPO; JOBS=$WORK; OUT=$OUT; P=$P
export M10_BUSY_LEVELS=$BUSY M10_SWEEP_DIR=$SW M10_PRECISION_DIR=$OUT
fmt_of() { python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['choice'][sys.argv[2]][sys.argv[3]])" "$OUT/fmt_select/fmt_choice.json" "\$1" "\$2"; }
lastrun() { ls -d "\$TIARA_RUNS"/*-"\$1" | sort | tail -1 | xargs basename; }
pool() { (cd $WORK && python $REPO/code/h200_pool.py "\$@"); }
EOF
}
dqsweep_C() {     # 42b plan → 42c：MI300X 版寫的腳本裡 m10_speed grid 沒指定 GPU 忙 → 加 H200 的；run 名稱 m10- → m10h-
  h200_lib
  python3 code/m10_dqsweep.py plan --sw "$SW/dqsweep" --jobs "$WORK" || return 1
  sed -i -e "s|code/m10_speed.py grid |code/m10_speed.py grid --busy-levels $BUSY |g" -e "s|m7run m10-|m7run m10h-|g" "$WORK/queue/42c_dqsweep_confirm.sh"
  bash "$WORK/queue/42c_dqsweep_confirm.sh"
}
tier2() {         # 44：兩層混合還原（12_tier_plan §3）；慢層參數沿用 MI300X 的 tier_params（H200 第一階段的「只用限速層」）
  local m=$1 d=$TO/$1
  mkdir -p "$d" "$WORK/smoke_tier2"
  m7run m10h-tier2-smoke-$m python code/m10_tier2.py smoke --model $m --out-dir "$WORK/smoke_tier2/$m" --busy-levels $BUSY || return 1
  m7run m10h-tier2-$m bash -c "python code/m10_tier2.py grid --model $m --out-dir $d --busy-levels $BUSY --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" || return 1
  guard_files_clean "$d"/gpu_guard_*.json || return 1
  python code/m10_tier2.py summary --restores "$d/restores.csv" --noise "$OUT/speed_$m/noise.json" --out-dir "$TO" --fig && gzip -kf "$d/restores.csv"
}
vllm_ttft() { mkdir -p "$OUT/vllm_ttft"; m7run m10h-vllm-$1-$2 python code/m10_vllm_ttft.py --model $1 --chunking $2 --out-dir "$OUT/vllm_ttft"; }
confirm_plan() {  # 42a：照兩個模型的 finalists 寫 43（09 §4.3），再切成兩段：43a 冒煙＋速度（t2，量時間）、43b 品質＋判定（q5）
  local m; for m in llama qwen; do [ -f "$SW/analysis/finalists_$m.json" ] || { echo "沒有 $SW/analysis/finalists_$m.json（q3 的海選彙整）"; return 1; }; done
  h200_lib
  python3 code/m10_confirm.py plan --jobs "$WORK" --sw-dir "$SW" > "$WORK/confirm_plan.out" 2>&1 || { tail -20 "$WORK/confirm_plan.out"; return 1; }
  tail -3 "$WORK/confirm_plan.out"
  local f=$WORK/queue/43_sweep_confirm.sh
  sed -i "s|m7run m10-|m7run m10h-|g" "$f"
  awk 'NR<=4{print; next} /^# 2\. /{exit} {print}' "$f" > "$WORK/queue/43a_speed.sh"
  awk 'NR<=4{print; next} /^# 2\. /{p=1} p{print}' "$f" > "$WORK/queue/43b_quality.sh"
  grep -q 'm10_confirm.py speed' "$WORK/queue/43a_speed.sh" && grep -q 'jobs_43q.json' "$WORK/queue/43b_quality.sh" \
    || { echo "43 的腳本切開後內容不對（$WORK/queue/43a_speed.sh、43b_quality.sh）"; return 1; }
}
confirm_speed() { # 43a：冒煙＋速度格子。MI300X 版失敗時只寫 NEEDS_HUMAN、exit 0；這裡照規則 2 停下
  : > "$WORK/NEEDS_HUMAN.43"; cp "$WORK/NEEDS_HUMAN" "$WORK/NEEDS_HUMAN.before" 2>/dev/null || : > "$WORK/NEEDS_HUMAN.before"
  bash "$WORK/queue/43a_speed.sh" || return 1
  if [ -f "$WORK/NEEDS_HUMAN" ] && ! diff -q "$WORK/NEEDS_HUMAN" "$WORK/NEEDS_HUMAN.before" >/dev/null; then
    echo "43a 寫了 NEEDS_HUMAN："; diff "$WORK/NEEDS_HUMAN.before" "$WORK/NEEDS_HUMAN" | sed -n 's/^> //p'; return 1
  fi
  guard_files_clean "$SW"/confirm/speed_*/gpu_guard_*.json
}

if [ "$PART" = q3 ]; then
  step Q3-POOL-$MODEL      q3_pool
  step SWEEP-ANA-$MODEL    sweep_analyze
  if [ "$MODEL" = llama ]; then
    step CWE-RULER  cwe_gen
    step CWE-HFGEN  cwe_hfgen
    step CWE-VLLM   cwe_vllm
  fi
elif [ "$PART" = t2 ]; then
  step GPU-SELFTEST gpu_selftests
  step DQ-BENCH     dq_bench
  for m in llama qwen; do for tier in fp8 int4; do step DQGRID-$m-$tier dqfast_grid $m $tier; done; done
  for m in llama qwen; do step DQS-A-$m dqsweep_A $m; done
  for m in llama qwen; do step DQS-B-$m dqsweep_B $m; done
  step DQS-C        dqsweep_C
  for m in llama qwen; do step TIER2-$m tier2 $m; done
  for m in llama qwen; do for c in default 512; do step TTFT-$m-$c vllm_ttft $m $c; done; done
  step CONFIRM-PLAN confirm_plan
  step CONFIRM-SPEED confirm_speed
else
  T2W=$TIARA_RUNS/_m10h_work/x-t2       # 43b 是 t2 的 confirm_plan 寫的；它 source 那裡的 lib.sh、讀那裡的 jobs_43q.json
  step CONFIRM-QUALITY bash -c "[ -f $T2W/queue/43b_quality.sh ] || { echo '沒有 43b（t2 的 CONFIRM-PLAN 沒跑完）'; exit 1; }; cp $T2W/NEEDS_HUMAN $T2W/NEEDS_HUMAN.q5before 2>/dev/null || : > $T2W/NEEDS_HUMAN.q5before; bash $T2W/queue/43b_quality.sh || exit 1; [ ! -f $T2W/NEEDS_HUMAN ] || diff -q $T2W/NEEDS_HUMAN $T2W/NEEDS_HUMAN.q5before >/dev/null || { echo '43b 寫了 NEEDS_HUMAN'; diff $T2W/NEEDS_HUMAN.q5before $T2W/NEEDS_HUMAN; exit 1; }"
fi

if [ "$STARTED" = 0 ]; then
  echo "[r_h200_x] 沒有叫做 '$FROM' 的步驟。可用的：$SEEN" >&2; exit 2
fi
[ "$DRY" = 0 ] && echo "[r_h200_x] $(date -Is) part=$PART ${MODEL:+model=$MODEL }全部完成；記進 RUNLOG_H200"
exit 0
