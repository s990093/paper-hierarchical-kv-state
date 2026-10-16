#!/usr/bin/env bash
# =============================================================================
# h200_m10_analyze.sh — H200 第二階段復現的判定（CPU，head node；13_h200_repro.md）
#
#   bash code/h200_m10_analyze.sh all            # 照判定流程走到底（下面每一段，走不到的不分析）
#   bash code/h200_m10_analyze.sh step1|int4     # 單獨一段
#   bash code/h200_m10_analyze.sh 2b|step2 <m>
#
# 判定一律由同一支凍結的 code/m10_analyze.py 算；這支只是 MI300X 的 analyze_step1.sh、analyze_2b.sh、analyze_int4.sh、
# analyze_step2_int4.sh 換成 H200 的路徑。和 MI300X 的差別只有一處：MI300X 的 Qwen 速度用 stage 22／23 重量的那次
# （原 run 被 gpu_guard 標污染）；H200 的速度格子在 r_h200_m10.sh 裡就要求 CLEAN（13 §3-5），沒有重量的那份。
# 「走到哪」照凍結的 cmd_decide 的分支、MI300X 的 route_after_*.py 的規則；走到需要 H200 沒跑的 GPU 工作時停下來（13 §4）。
# 結果寫進 results/m10_precision_h200/analysis/；每一段印出判定，記進 RUNLOG_H200。
# =============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
OUT=${M10H_OUT:-$REPO/results/m10_precision_h200}     # M10H_OUT：只給回歸測試（拿 MI300X 的資料重算）
PY=${PYTHON:-python3}
fmt_of() { "$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['choice'][sys.argv[2]][sys.argv[3]])" "$OUT/fmt_select/fmt_choice.json" "$1" "$2"; }
need() { for f in "$@"; do [ -e "$f" ] || { echo "[h200_m10_analyze] 找不到 $f（GPU 工作還沒跑完？）" >&2; exit 1; }; done; }

step1() {       # MI300X analyze_step1.sh
  local A=$OUT/analysis/fp8 m
  mkdir -p "$A"
  for m in llama qwen; do
    need "$OUT/quality/step1_${m}_T5.csv" "$A/speed_$m.json"
    "$PY" code/m10_analyze.py quality --model $m --csv "$OUT"/quality/step1_${m}_*.csv --configs E1 E1s E1sr --tag step1 --out-dir "$A"
  done
  "$PY" code/m10_analyze.py decide --tier fp8 --llama "$A/quality_llama_step1.json" --qwen "$A/quality_qwen_step1.json" \
    --speed-llama "$A/speed_llama.json" --speed-qwen "$A/speed_qwen.json" --out-dir "$A"
}
b2() {          # MI300X analyze_2b.sh（Qwen 的 regrid 特例拿掉）
  local m=$1 A=$OUT/analysis/fp8 G g F
  G=$(fmt_of $m fp8); g=${G#fp8_}
  "$PY" code/m10_analyze.py quality --model $m --csv "$OUT"/quality/step1_${m}_*.csv --configs E1s@fp8k0_$g E1s@int8_token --tag step2b --out-dir "$A"
  for F in fp8k0_$g int8_token; do
    need "$OUT/speed_$m/summary_2b_$F.csv"
    "$PY" code/m10_analyze.py speed --model $m --summary "$OUT/speed_$m/summary_fp8.csv" --noise "$OUT/speed_$m/noise.json" \
      --configs E1s@$F --x-summary "$OUT/speed_$m/summary_2b_$F.csv" --x-from E1s --tag _2b_$F --out-dir "$A"
  done
  "$PY" code/m10_analyze.py k3 --quality "$A/quality_${m}_step2b.json" --speed "$A/speed_${m}_2b_fp8k0_$g.json" "$A/speed_${m}_2b_int8_token.json" --out-dir "$A"
}
int4() {        # MI300X analyze_int4.sh
  local A=$OUT/analysis/int4 m I
  mkdir -p "$A"
  for m in llama qwen; do
    I=$(fmt_of $m int4)
    need "$OUT/quality_sealed/int4end_${m}_T5.csv" "$OUT/speed_$m/summary_int4.csv"
    "$PY" code/m10_analyze.py quality --model $m --csv "$OUT"/quality/step1_${m}_*.csv "$OUT"/quality_sealed/int4end_${m}_*.csv \
      --strip-fmt $I --configs E1 E1s E1sr --tag int4 --out-dir "$A"
    "$PY" code/m10_analyze.py speed --model $m --summary "$OUT/speed_$m/summary_int4.csv" --noise "$OUT/speed_$m/noise.json" --out-dir "$A"
  done
  "$PY" code/m10_analyze.py decide --tier int4 --llama "$A/quality_llama_int4.json" --qwen "$A/quality_qwen_int4.json" \
    --speed-llama "$A/speed_llama.json" --speed-qwen "$A/speed_qwen.json" --out-dir "$A"
}
step2() {       # MI300X analyze_step2_int4.sh（暫存檔放 $TIARA_RUNS 底下，不放 results/）
  local m=$1 A=$OUT/analysis/int4 I MIX TMP
  I=$(fmt_of $m int4)
  TMP=${TIARA_RUNS:-$HOME/tiara-runs}/_m10h_work/step2_int4_tmp; mkdir -p "$TMP"
  need "$OUT/quality_sealed/int4mix_${m}_T5.csv"
  MIX=$(ls "$OUT"/quality_sealed/int4mix_${m}_T24_s*.csv "$OUT"/quality_sealed/int4mix_${m}_T5.csv)
  # shellcheck disable=SC2086
  "$PY" code/m10_analyze.py quality --model $m --csv "$OUT"/quality/step1_${m}_T24_s*.csv "$OUT"/quality/step1_${m}_T5.csv $MIX \
    --strip-fmt $I --configs F25 M25 B25 R25 --tag step2 --out-dir "$A"
  # shellcheck disable=SC2086
  "$PY" - "$I" "$TMP/int4mix_${m}_stripped.csv" $MIX <<'PY'
import csv, sys
fmt, out, srcs = sys.argv[1], sys.argv[2], sys.argv[3:]
n = 0
with open(out, "w", newline="") as fo:
    w = None
    for p in srcs:
        for r in csv.DictReader(open(p, newline="")):
            if r.get("run_id") == "run_id":
                continue
            name, _, f = r["config"].partition("@")
            if f != fmt:
                continue
            r["config"] = name
            if w is None:
                w = csv.DictWriter(fo, fieldnames=list(r.keys())); w.writeheader()
            w.writerow(r); n += 1
print(f"stripped {n} rows ({fmt}) → {out}")
if n == 0:
    sys.exit("沒有任何 @%s 的列" % fmt)
PY
  "$PY" code/m10_analyze.py step2 --model $m --csv "$TMP/int4mix_${m}_stripped.csv" \
    --speed "$A/speed_$m.json" --summary "$OUT/speed_$m/summary_int4.csv" --tag int4 --out-dir "$A"
}

route() {       # route fp8|int4 → 印一行：「GO <下一段> [模型]」或「STOP <原因>」（照凍結的 cmd_decide、MI300X route_after_*.py）
  "$PY" - "$OUT" "$1" <<'PY'
import json, os, sys
OUT, tier = sys.argv[1], sys.argv[2]
d = json.load(open(f"{OUT}/analysis/{tier}/decide_{tier}.json"))
k0 = {m: json.load(open(f"{OUT}/speed_{m}/k0_verdict_{tier}.json"))["verdict"].startswith("K0 成立") for m in ("llama", "qwen")}
print(f"# decide_{tier}.next = {d['next']}", file=sys.stderr)
print(f"# K0({tier}) = {k0}", file=sys.stderr)
blk = [m for m in ("llama", "qwen") if d[m].get("power_blocks")]
L, Q = d["llama"]["model_pass"], d["qwen"]["model_pass"]
if all(k0.values()):
    print(f"STOP K0 在 {tier} 兩個模型都成立（沒有賺頭）→ 位置策略停")
elif blk:
    print(f"STOP 檢定力擋判定（{blk}）→ 先加題（07 §9），問使用者")
elif L is None or Q is None:
    print("STOP 資料不全：" + d["next"])
elif L and Q:
    print("GO int4" if tier == "fp8" else "STOP INT4 階兩個模型都過 → 整個停（03 §2.5 第 3 步）")
else:
    m = "qwen" if (L and not Q) else "llama"
    if tier == "fp8":
        print(f"GO 2b {m}")
    elif k0[m]:
        print(f"STOP Step 2 會在 {m} 上做，但 {m} 的 K0（INT4）成立 → 位置策略停")
    else:
        print(f"GO step2 {m}")
PY
}

all() {
  local r m
  step1
  r=$(route fp8); echo "[route fp8] $r"
  case "$r" in
    "GO int4") ;;
    "GO 2b "*)
      m=${r#GO 2b }
      b2 "$m"
      if "$PY" -c "import json,sys;sys.exit(0 if json.load(open(sys.argv[1]))['K3'] else 1)" "$OUT/analysis/fp8/k3_$m.json"; then
        echo "[route 2b] $m 的 K3 成立 → FP8 階不需要位置策略 → INT4"
      else
        echo "[route 2b] STOP $m 的 K3 不成立 → Step 2 要在 FP8 階的 $m 上做，H200 沒跑 FP8 混合配置（13 §4）→ 問使用者"; return 0
      fi ;;
    *) return 0 ;;
  esac
  int4
  r=$(route int4); echo "[route int4] $r"
  case "$r" in
    "GO step2 "*) m=${r#GO step2 }; step2 "$m"
      "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));print('[step2]', d['model'], 'stable_gap_1_2 =', d['stable_gap_1_2'], 'K2 =', d['K2'])" "$OUT/analysis/int4/step2_${m}_int4.json" ;;
    *) return 0 ;;
  esac
}

case "${1:-}" in
  all) all ;;
  step1) step1 ;;
  int4) int4 ;;
  2b) b2 "${2:?要模型}" ;;
  step2) step2 "${2:?要模型}" ;;
  route) route "${2:?fp8|int4}" ;;
  *) sed -n '2,14p' "$0"; exit 2 ;;
esac
