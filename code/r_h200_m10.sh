#!/bin/bash
#SBATCH --job-name=m10h
#SBATCH --partition=h200q
#SBATCH --account=ChihChungHsu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=24:00:00
#SBATCH -o slurm-%x-%j.out
# =============================================================================
# r_h200_m10.sh — 第二階段（精度）主線在 H200 上的復現（事先登記：docs/precision_20261013/13_h200_repro.md）
#
#   sbatch --job-name=m10h-speed code/r_h200_m10.sh --part speed
#   sbatch --job-name=m10h-q-llama --time=40:00:00 --dependency=afterok:<speed> --kill-on-invalid-dep=yes \
#          code/r_h200_m10.sh --part quality --model llama          # make submit-m10 一次送三個
#   sbatch ... code/r_h200_m10.sh --part speed --from GRID-FP8-qwen  # 從某一步接著跑（--list 看步驟名）
#   bash   code/r_h200_m10.sh --part speed --dry-run | --list         # 只印指令（任何機器都能跑）
#
# --part speed（1 張卡，量時間；不能和我們自己的其他 GPU job 同時跑）：
#   V0 m10_verify.sh 完整模式 → E0 難度確認（seed 2000 起）→ K／V 分布 → 格式小測（07 §3）＋和 MI300X 比
#   → calib-f（頻寬）→ noise（δ*）→ FP8 速度格子＋K0 → Step 2b 速度 → INT4 速度格子＋K0
#   MI300X 的對應：m10_queue_mi300x/ 的 p1_speed_calib.sh、queue/10、20、21、33a（只換路徑；GPU 忙用 H200 的 busy_levels）
# --part quality --model <m>（1 張卡，只量分數；同一張卡上同時開多個 process，h200_pool.py）：
#   Step 1（FP8 端點＋2b，讀法 A＋B）→ INT4 端點 → INT4 混合配置（兩個模型都跑、封存，13 §4）
#   MI300X 的對應：queue/30、33b、33c（同樣的指令、題數、shard）
#
# 停止條件（13 §3）：任何一步失敗、程式和 MI300X 不同、E0 難度不在 60–95%、格式和 MI300X 不同、速度格子的 gpu_guard 不是 CLEAN。
# 判定（CPU）不在這支：code/h200_m10_analyze.sh，在 head node 上跑。
# =============================================================================
set -uo pipefail

PART=""; MODEL=""; DRY=0; LIST=0; FROM=""
while [ $# -gt 0 ]; do
  case "$1" in
    --part) PART="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --list) LIST=1; DRY=1; shift ;;
    --from) FROM="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "不認得的參數：$1（--help 看用法）" >&2; exit 2 ;;
  esac
done
case "$PART" in
  speed) [ -z "$MODEL" ] || { echo "--part speed 不收 --model（兩個模型都跑）" >&2; exit 2; } ;;
  quality) case "$MODEL" in llama|qwen) ;; *) echo "--part quality 要 --model llama|qwen" >&2; exit 2 ;; esac ;;
  *) echo "要 --part speed|quality" >&2; exit 2 ;;
esac

# ---- 找 repo（和 r_h200_rest.sh 同一個順序）----------------------------------------------------
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
if [ -n "${REPO:-}" ] && [ -f "$REPO/code/env_h200.sh" ]; then :
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/code/env_h200.sh" ]; then REPO=$SLURM_SUBMIT_DIR
elif [ -f "$_here/env_h200.sh" ]; then REPO=$(cd "$_here/.." && pwd)
elif [ -f "$HOME/paper-hierarchical-kv-state/code/env_h200.sh" ]; then REPO=$HOME/paper-hierarchical-kv-state
else echo "[r_h200_m10] 找不到 repo（code/env_h200.sh）。請在 repo 根目錄 sbatch，或先 export REPO" >&2; exit 2
fi
export REPO
[ "$DRY" = 1 ] && export TIARA_SKIP_CONDA=1
# shellcheck disable=SC1091
source "$REPO/code/env_h200.sh"
cd "$REPO"
ulimit -c 0                          # MI300X 上 abort 留過 21 GB 的 core；lab 配額只剩幾十 GB

OUT=$REPO/results/m10_precision_h200
MI=$REPO/results/m10_precision_mi300x
P=$REPO/results/m10_pilot_mi300x/e0_tune/CANDIDATE_PARAMS.json
BUSY=$REPO/results/m7_write_policy_h200/busy_levels.json     # H200 R2-C5：busy2＝duty 1.0（重算慢 1.945 倍）
G=code/h200_m10_gates.py
WORK=$TIARA_RUNS/_m10h_work/${PART}${MODEL:+-$MODEL}-${SLURM_JOB_ID:-dry}   # 工作池的 jobs.json、pool_*.log
CF="E0 E1@fp8_head E1@fp8_channel E1@fp8_token E1@int4_token E1@int4_kivi"   # 07 §3 的 5 種候選

# ---- 前置檢查（真的要跑時才擋）----------------------------------------------------------
if [ "$DRY" = 0 ]; then
  [ -n "${SLURM_JOB_ID:-}" ] || { echo "[r_h200_m10] 不在 Slurm job 裡。用 sbatch 送" >&2; exit 2; }
  python $G code   || { echo "[r_h200_m10] 程式和 MI300X 不同 → 停（13 §3-2）" >&2; exit 2; }
  python $G params || exit 2
  [ -f "$BUSY" ] || { echo "[r_h200_m10] 找不到 $BUSY（第一階段 R2-C5）" >&2; exit 2; }
  compgen -G "$M7_LLAMA_GLOB" >/dev/null && compgen -G "$M7_QWEN_GLOB" >/dev/null \
    || { echo "[r_h200_m10] 找不到模型（$M7_LLAMA_GLOB／$M7_QWEN_GLOB）" >&2; exit 2; }
  python -c 'import sys; sys.path.insert(0, "code"); from gpu_guard import smi_index; print("[r_h200_m10] smi_index(0) =", smi_index(0))' \
    || { echo "[r_h200_m10] cuda:0 對不到 nvidia-smi 的卡（gpu_guard.smi_index）→ 停" >&2; exit 2; }
  if [ "$PART" = quality ]; then
    python - "$OUT/fmt_select/fmt_vs_mi300x.json" <<'PY' || { echo "[r_h200_m10] 格式還沒選或和 MI300X 不同（13 §3-4）→ 停" >&2; exit 2; }
import json, sys
sys.exit(0 if json.load(open(sys.argv[1]))["same"] else 1)
PY
    for f in "$OUT/speed_$MODEL/meets.json" "$OUT/speed_$MODEL/bw.json"; do
      [ -f "$f" ] || { echo "[r_h200_m10] 找不到 $f（--part speed 還沒跑完）→ 讀法 B 沒有 H200 的會合點" >&2; exit 2; }
    done
  fi
  mkdir -p "$OUT" "$WORK"
  echo "[r_h200_m10] $(date -Is) part=$PART model=${MODEL:-both} job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-?} git=$(git rev-parse --short HEAD) OUT=$OUT"
  nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.used --format=csv
fi

# ---- 步驟（和 r_h200_rest.sh 同一套：失敗就停、--from 接著跑）---------------------------------
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
    echo "=== $(date -Is) $name 失敗（exit $rc）→ 停下、回報（CLAUDE.md 規則 2）。接著跑：sbatch ... code/r_h200_m10.sh --part $PART ${MODEL:+--model $MODEL }--from $name" >&2
    exit $rc
  fi
}
lastrun() { ls -d "$TIARA_RUNS"/*-"$1" | sort | tail -1 | xargs basename; }
fmt_of() { python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['choice'][sys.argv[2]][sys.argv[3]])" "$OUT/fmt_select/fmt_choice.json" "$1" "$2"; }
# 工作池：jobs.json 由 python 產生（指令照 MI300X 的 queue 腳本），在 $WORK 裡跑
pool() { (cd "$WORK" && python "$REPO/code/h200_pool.py" "$1" "$2"); }
guard_clean() {   # 13 §3-5：這次 grid run 的每一份 gpu_guard JSON 都要 CLEAN（同 MI300X route_after_int4.guard_clean）
  local dir=$1 rid=$2
  python - "$dir" "$rid" <<'PY'
import glob, json, os, sys
d, rid = sys.argv[1], sys.argv[2]
fs = sorted(glob.glob(os.path.join(d, f"gpu_guard_{rid}_L*.json")))
bad = [os.path.basename(f) + "=" + str(json.load(open(f)).get("verdict")) for f in fs
       if json.load(open(f)).get("verdict") != "CLEAN" or json.load(open(f)).get("contaminated")]
print(f"[guard] {rid}: {len(fs)} 份 gpu_guard JSON，不是 CLEAN 的：{bad or '無'}")
sys.exit(0 if fs and not bad else 1)
PY
}

# ===== 速度那一半的每一步（函式：一步裡有好幾條指令）=====
e0_check() {      # 13 §2：seed 2000 起、兩個模型 × T1–T5 各 60 題、只跑 E0（品質 run，時間欄不用）
  mkdir -p "$OUT/e0_check"
  python3 - "$P" "$OUT" "$WORK/jobs_e0.json" <<'PY' || return 1
import json, shlex, sys
P, OUT, J = sys.argv[1:4]
prm = json.load(open(P))["params"]
jobs = []
for t in ("T1", "T2", "T3", "T4", "T5"):
    for m, need in (("llama", 42), ("qwen", 28)):
        jobs.append(dict(name=f"e0-{m}-{t}", need_gb=need,
                         cmd=f"m7run m10h-e0chk-{m}-{t} python code/m10_quality.py tune --model {m} --tasks {t} --n 60 "
                             f"--seed0 2000 --params {shlex.quote(json.dumps(prm[m]))} "
                             f"--out {OUT}/e0_check/e0_{m}_{t}.csv"))
json.dump(jobs, open(J, "w"), indent=1)
PY
  pool "$WORK/jobs_e0.json" 6 || return 1
  m7run m10h-gate-e0 python $G e0 "$OUT/e0_check/e0_check.json" $(ls "$OUT"/e0_check/e0_*_T[1-5].csv)
}
fmt_select() {    # MI300X queue/10_fmt_select.sh
  mkdir -p "$OUT/fmt_select"
  python3 - "$P" "$OUT" "$WORK/jobs_fmt.json" "$CF" <<'PY' || return 1
import json, sys
P, OUT, J, CF = sys.argv[1:5]
jobs = []
for m, need in (("llama", 48), ("qwen", 30)):
    for s in (0, 1):
        jobs.append(dict(name=f"fmtsel-{m}-s{s}", need_gb=need,
                         cmd=f"m7run m10h-fmtsel-{m}-s{s} python code/m10_quality.py run --model {m} --params-file {P} "
                             f"--seed0 4000 --n 5 --tasks T2 T3 T4 T5 --modes A --configs {CF} --shard {s}/2 "
                             f"--out {OUT}/fmt_select/mini_{m}_s{s}.csv"))
json.dump(jobs, open(J, "w"), indent=1)
PY
  pool "$WORK/jobs_fmt.json" 4 || return 1
  m7run m10h-fmtselect python code/m10_fmt_select.py --csv $(ls "$OUT"/fmt_select/mini_*_s*.csv) --out "$OUT/fmt_select/fmt_choice.json"
}
grid_fp8() {      # MI300X queue/20_grid_fp8.sh（＋GPU 忙用 H200 的 busy_levels）
  local m=$1 g rid
  g=$(fmt_of "$m" fp8) || return 1
  m7run m10h-grid-fp8-$m bash -c "python code/m10_speed.py grid --model $m --fmt $g --base-fmt $g --out-dir $OUT/speed_$m --bw-json $OUT/speed_$m/bw.json --busy-levels $BUSY --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" || return 1
  rid=$(lastrun m10h-grid-fp8-$m)
  guard_clean "$OUT/speed_$m" "$rid" || return 1
  cp "$OUT/speed_$m/summary.csv" "$OUT/speed_$m/summary_fp8.csv"; cp "$OUT/speed_$m/spec_grid.json" "$OUT/speed_$m/spec_grid_fp8.json"
  m7run m10h-k0v-fp8-$m python code/m10_speed.py k0-verdict --model $m --out-dir "$OUT/speed_$m" --from-run "$rid" --delta-json "$OUT/speed_$m/noise.json" || return 1
  cp "$OUT/speed_$m/k0_verdict.json" "$OUT/speed_$m/k0_verdict_fp8.json"
  m7run m10h-anaspeed-fp8-$m python code/m10_analyze.py speed --model $m --summary "$OUT/speed_$m/summary_fp8.csv" --noise "$OUT/speed_$m/noise.json" --out-dir "$OUT/analysis/fp8"
}
grid_2b() {       # MI300X queue/21_grid_2b.sh
  local m=$1 g f rid
  g=$(fmt_of "$m" fp8) || return 1
  for f in "fp8k0_${g#fp8_}" int8_token; do
    m7run m10h-grid-2b-$f-$m bash -c "python code/m10_speed.py grid --model $m --fmt $f --base-fmt $g --configs E1s --refs --lengths --out-dir $OUT/speed_$m --bw-json $OUT/speed_$m/bw.json --busy-levels $BUSY --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" || return 1
    rid=$(lastrun m10h-grid-2b-$f-$m)
    guard_clean "$OUT/speed_$m" "$rid" || return 1
    cp "$OUT/speed_$m/summary.csv" "$OUT/speed_$m/summary_2b_$f.csv"; cp "$OUT/speed_$m/spec_grid.json" "$OUT/speed_$m/spec_grid_2b_$f.json"
  done
}
grid_int4() {     # MI300X queue/33a_grid_int4.sh
  # INT4 格子會加上 @int4 的鍵、並覆寫 E0 的會合點（m10_speed.write_meets）。MI300X 的 Step 1 品質（30）在 33a 之前跑，
  # 讀的是 FP8 格子＋2b 格子之後的 meets.json；之後 45、55 改讀 33a 前的快照 meets_fp8tier.json。H200 的品質排在全部格子之後，
  # 所以 Step 1 讀這份快照（只在第一次建立：續跑時不被 INT4 的結果蓋掉）。
  local m=$1 g i rid
  g=$(fmt_of "$m" fp8) || return 1; i=$(fmt_of "$m" int4) || return 1
  [ -f "$OUT/speed_$m/meets_fp8tier.json" ] || cp "$OUT/speed_$m/meets.json" "$OUT/speed_$m/meets_fp8tier.json" || return 1
  m7run m10h-grid-int4-$m bash -c "python code/m10_speed.py grid --model $m --fmt $i --base-fmt $g --out-dir $OUT/speed_$m --bw-json $OUT/speed_$m/bw.json --busy-levels $BUSY --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" || return 1
  rid=$(lastrun m10h-grid-int4-$m)
  guard_clean "$OUT/speed_$m" "$rid" || return 1
  cp "$OUT/speed_$m/summary.csv" "$OUT/speed_$m/summary_int4.csv"; cp "$OUT/speed_$m/spec_grid.json" "$OUT/speed_$m/spec_grid_int4.json"
  m7run m10h-k0v-int4-$m python code/m10_speed.py k0-verdict --model $m --out-dir "$OUT/speed_$m" --from-run "$rid" --delta-json "$OUT/speed_$m/noise.json" || return 1
  cp "$OUT/speed_$m/k0_verdict.json" "$OUT/speed_$m/k0_verdict_int4.json"
}

# ===== 品質那一半：MI300X queue/30、33b、33c 的指令，一個模型 =====
quality_pool() {  # quality_pool <tag> <shards> <need_gb> <out 子目錄> <檔名前綴> <meets 檔名> <configs...>
  local tag=$1 shards=$2 need=$3 sub=$4 pre=$5 meets=$6; shift 6
  mkdir -p "$OUT/$sub"
  [ -f "$OUT/speed_$MODEL/$meets" ] || { echo "找不到 $OUT/speed_$MODEL/$meets"; return 1; }
  python3 - "$MODEL" "$P" "$OUT" "$tag" "$shards" "$need" "$sub" "$pre" "$WORK/jobs_$tag.json" "$meets" "$@" <<'PY' || return 1
import json, sys
m, P, OUT, tag, shards, need, sub, pre, J, meets = sys.argv[1:11]
C = " ".join(sys.argv[11:])
shards, need = int(shards), int(need)
fmt = json.load(open(f"{OUT}/fmt_select/fmt_choice.json"))["choice"][m]["fp8"]
base = (f"python code/m10_quality.py run --model {m} --params-file {P} --fmt {fmt} --configs {C} --modes A B --bws b32 "
        f"--meets {OUT}/speed_{m}/{meets}")
jobs = [dict(name=f"{tag}-{m}-T24-s{s}", need_gb=need,
             cmd=f"m7run m10h-{tag}-{m}-T24-s{s} {base} --tasks T2 T3 T4 --n 180 --shard {s}/{shards} "
                 f"--out {OUT}/{sub}/{pre}_{m}_T24_s{s}.csv") for s in range(shards)]
jobs.append(dict(name=f"{tag}-{m}-T5", need_gb=need,
                 cmd=f"m7run m10h-{tag}-{m}-T5 {base} --tasks T5 --n 60 --out {OUT}/{sub}/{pre}_{m}_T5.csv"))
json.dump(jobs, open(J, "w"), indent=1)
print(f"[quality_pool] {tag} {m}: {len(jobs)} 個 process，configs={C}")
PY
  pool "$WORK/jobs_$tag.json" 6
}

if [ "$PART" = speed ]; then
  step V0-verify  m7run m10h-verify bash -c 'bash code/m10_verify.sh --out "$TIARA_RUNS/$RUN_ID/verify"'
  step E0-check   e0_check
  for m in llama qwen; do
    step KV-$m    m7run m10h-kvstats-$m bash -c "python code/m10_kv_stats.py --model $m --out $OUT/kv_stats --raw \$TIARA_RUNS/\$RUN_ID/raw"
  done
  step FMT-select fmt_select
  step FMT-gate   m7run m10h-gate-fmt python $G fmt "$OUT/fmt_select/fmt_choice.json" "$MI/fmt_select/fmt_choice.json" "$OUT/fmt_select/fmt_vs_mi300x.json"
  for m in llama qwen; do      # MI300X p1_speed_calib.sh：先兩個模型的 calib-f，再兩個的 noise
    step CALIB-$m m7run m10h-calibf-$m bash -c "mkdir -p $OUT/speed_$m && python code/m10_speed.py calib-f --model $m --out-dir $OUT/speed_$m"
  done
  for m in llama qwen; do
    step NOISE-$m m7run m10h-noise-$m bash -c "python code/m10_speed.py noise --model $m --out-dir $OUT/speed_$m --bw-json $OUT/speed_$m/bw.json --busy-levels $BUSY --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv"
  done
  for m in llama qwen; do step GRID-FP8-$m  grid_fp8  $m; done
  for m in llama qwen; do step GRID-2B-$m   grid_2b   $m; done
  for m in llama qwen; do step GRID-INT4-$m grid_int4 $m; done
else
  m=$MODEL
  if [ "$DRY" = 1 ] && [ ! -f "$OUT/fmt_select/fmt_choice.json" ]; then g="fp8_<選定>"; i="int4_<選定>"
  else g=$(fmt_of $m fp8); i=$(fmt_of $m int4); fi
  # need_gb：MI300X 的值（30：42／28；33b、33c：32／24）
  if [ $m = llama ]; then N1=42; N2=32; else N1=28; N2=24; fi
  # Step 1 讀 INT4 格子之前的會合點（meets_fp8tier.json，見 grid_int4）；INT4 的品質讀 INT4 格子之後的 meets.json（同 MI300X 33b、33c）
  step S1-$m   quality_pool s1  6 $N1 quality        step1   meets_fp8tier.json E0 E1 E1s E1sr "E1s@fp8k0_${g#fp8_}" E1s@int8_token
  step I4E-$m  quality_pool i4e 5 $N2 quality_sealed int4end meets.json "E1@$i" "E1s@$i" "E1sr@$i"
  step I4M-$m  quality_pool i4m 6 $N2 quality_sealed int4mix meets.json "F25@$i" "M25@$i" "B25@$i" "R25@$i"
fi

if [ "$STARTED" = 0 ]; then
  echo "[r_h200_m10] 沒有叫做 '$FROM' 的步驟。可用的：$SEEN" >&2; exit 2
fi
[ "$DRY" = 0 ] && echo "[r_h200_m10] $(date -Is) part=$PART ${MODEL:+model=$MODEL }全部完成。判定：bash code/h200_m10_analyze.sh（head node）；記進 RUNLOG_H200"
exit 0
