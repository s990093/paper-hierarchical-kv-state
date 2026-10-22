# GPU-NEED: none
# AFTER: 38c1_xkv_cal 38c2_xkv_cal
# xKV 校準基底 vs 每題 SVD 的報告（只用 CPU、不 import torch；cpu 佇列）→ results/m16_stage3_mi300x/x_stack/v2/{xlcal_report.json, tables_xlcal.md}
# （校準摘要 calib_xl_<m>.json 由 calib-xl 寫在同一個目錄）。判定事先寫死（m16_xstack.py 的 XLCAL_TEXT；howto §14.4）：生成答對率、逐題配對
# 校準 − 每題 SVD 的 95% CI 上界 < −2 個百分點 → 必須每題算 SVD；下界 ≥ −2 → 可以事先算好基底；其他 → 題數不夠分辨。TF 只報。
# 之後核對（只寫 NEEDS_HUMAN、exit 0）：每個模型生成 ≥48 題（不夠 → 判定是 INCOMPLETE）。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_common.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_pool.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_ext.sh
JOB=38cz_xkv_cal_report
export OMP_NUM_THREADS=4
m7run m16-xlcal-report $XPY xlcal-report --raw-tf "$XD/xlcal/tf_*_s*.csv.gz" --raw-gen "$XD/xlcal/gen_*_s*.csv.gz" \
  --raw-x1 "$XD/x1/x1_*_s*.csv.gz" --raw-x2 "$XD/x2/x2_*_s*.csv.gz" --out-dir $XR2 \
  || { nh "$JOB：xlcal-report 失敗（$(lastrun m16-xlcal-report)）"; exit 1; }
python3 -I - $XR2/xlcal_report.json <<'PY' || nh "$JOB：xKV 校準的生成題數不夠（$XR2/xlcal_report.json）→ 判定是 INCOMPLETE"
import json, sys
r = json.load(open(sys.argv[1]))
print("判定：", r["verdict"], json.dumps(r["verdict_by_model"], ensure_ascii=False))
sys.exit(0 if all(v["gen"]["n_items"] >= 48 for v in r["models"].values()) else 1)
PY
echo "$JOB 結束 $(date -u +%FT%TZ)"
