# GPU-NEED: none
# AFTER: 38b1_x2b_gen 38b2_x2b_gen 38b3_x2b_gen 38b4_x2b_gen
# X2＋X2b 的合併報告（只用 CPU、不 import torch；cpu 佇列：nice 19、0–3 號核心）→ results/m16_stage3_mi300x/x_stack/v2/
# {x2b_report.json, tables_x2b.md, x2_feasible.csv, x2_unmapped.csv}（不覆蓋第一版的檔）。A／B／C／D 四種並列、M1／M2／MLPerf 照第一版的規則重算
# （x_stack_howto.md §14.3）。沒排的 38b* 要從上面的 AFTER 拿掉（runner 要所有 AFTER 都有 .done 才開）。
# 之後核對（只寫 NEEDS_HUMAN、exit 0，不擋 cpu 佇列；報告已寫出供人看）：X2b 每個模型 192 題齊全；X3 的 load_x2 讀得懂 v2/x2_feasible.csv
# （X3 的 seg_catalog 要有 10/21 的 xkv_k* 段）。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_common.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_pool.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_ext.sh
JOB=38bz_x2b_report
export OMP_NUM_THREADS=4
m7run m16-x2b-report $XPY x2b-report --raw-x2 "$XD/x2/x2_*_s*.csv.gz" --raw-x2b "$XD/x2b/x2b_*_s*.csv.gz" --plan $X2B_PLAN --out-dir $XR2 \
  || { nh "$JOB：x2b-report 失敗（$(lastrun m16-x2b-report)）"; exit 1; }
python3 -I - $XR2/x2b_report.json <<'PY' || nh "$JOB：X2b 題數不齊（$XR2/x2b_report.json 的 n_items）→ 報告是部分結果"
import json, sys
r = json.load(open(sys.argv[1]))
n = {m: (v["n_items"], v["x2b_complete"]) for m, v in r["models"].items()}
print("題數：", json.dumps(n, ensure_ascii=False))
sys.exit(0 if all(c for _, c in n.values()) else 1)
PY
if [ -e $XR2/x2_feasible.csv ]; then
  python3 -I -c "import sys; sys.path.insert(0, 'code'); import m16_xtime as XT; d = XT.load_x2(sys.argv[1]); print('X3 load_x2：', {k: sorted(v) for k, v in d['feasible'].items()})" \
    $XR2/x2_feasible.csv || nh "$JOB：X3 的 load_x2 讀不懂 $XR2/x2_feasible.csv（多半是 X3 的 seg_catalog 還沒有 10/21 的 xkv_k* 段）→ 要人協調"
fi
echo "$JOB 結束 $(date -u +%FT%TZ)"
