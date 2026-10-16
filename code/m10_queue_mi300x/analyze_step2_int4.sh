#!/usr/bin/env bash
# Step 2（INT4 階的混合配置；穩定差距第 1、2 條、K2）：只在 33bz 判出要做 Step 2 的模型上跑（03 §1.1）。用法：analyze_step2_int4.sh <model>
# 和 analyze_step2.sh（FP8 階）同一支凍結的 m10_analyze.py，只差在：
#   - 品質用 --strip-fmt（"F25@int4_kivi" → "F25"），和 analyze_int4.sh 一樣；
#   - step2 子命令沒有 --strip-fmt，所以先把 int4mix 的列照 cmd_quality 同樣的規則改名（只留這個格式的列）寫到暫存檔，再交給它；判準一行都不動。
#   - 速度：summary_int4.csv（33a）、analysis/int4/speed_<m>.json（只拿 δ*）。
set -euo pipefail
m=$1
REPO=/mlsteam/workspace/paper-hierarchical-kv-state; OUT=$REPO/results/m10_precision_mi300x; A=$OUT/analysis/int4
JOBS=/tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs; TMP=$JOBS/step2_int4_tmp; mkdir -p $TMP
source $REPO/code/m7_env.sh >/dev/null
I=$(python3 -c "import json;print(json.load(open('$OUT/fmt_select/fmt_choice.json'))['choice']['$m']['int4'])")
MIX=$(ls $OUT/quality_sealed/int4mix_${m}_T24_s*.csv $OUT/quality_sealed/int4mix_${m}_T5.csv)
python $REPO/code/m10_analyze.py quality --model $m --csv $OUT/quality/step1_${m}_T24_s*.csv $OUT/quality/step1_${m}_T5.csv $MIX \
  --strip-fmt $I --configs F25 M25 B25 R25 --tag step2 --out-dir $A
python3 - "$I" "$TMP/int4mix_${m}_stripped.csv" $MIX <<'PY'
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
python $REPO/code/m10_analyze.py step2 --model $m --csv $TMP/int4mix_${m}_stripped.csv \
  --speed $A/speed_$m.json --summary $OUT/speed_$m/summary_int4.csv --tag int4 --out-dir $A
