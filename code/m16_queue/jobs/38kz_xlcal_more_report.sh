# GPU-NEED: none
# AFTER: 38k1_xlcal_more 38k2_xlcal_more
# xKV 校準基底 vs 每題 SVD 加題數的報告（只用 CPU、不 import torch；cpu 佇列）→ results/m16_stage3_mi300x/x_stack/v3/{xlcal3_report.json, tables_xlcal3.md}
# （不覆蓋 v2）。並列「只用 38c 的 96 題」「只用新的 192 題」「合併 288 題」，主要讀合併；判定照 XLCAL_TEXT 原文，aggregate 改成這兩對（寫在報告裡）。
# 之後核對（只寫 NEEDS_HUMAN、exit 0，報告已寫出供人看）：新題每個模型 192 題齊全。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_common.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_pool.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/xlcal3_common.sh
JOB=38kz_xlcal_more_report
export OMP_NUM_THREADS=4
m7run m16-xlcal3-report $XL3PY report --raw-old "$XD/xlcal/gen_*_32K_s*.csv.gz" --raw-new "$XD/xlcal3/gen_*_32K_s*.csv.gz" \
  --calib-sha $XR3/calib_xl_sha.json --out-dir $XR3 || { nh "$JOB：report 失敗（$(lastrun m16-xlcal3-report)）"; exit 1; }
python3 -I - $XR3/xlcal3_report.json <<'PY' || nh "$JOB：新題不齊（$XR3/xlcal3_report.json 的 new192 n_items）→ 報告是部分結果"
import json, sys
r = json.load(open(sys.argv[1]))
n = {m: v["n_items"] for m, v in r["variants"]["new192"]["models"].items()}
print("判定（合併）：", r["verdict"], json.dumps(r["verdict_by_model"], ensure_ascii=False), "；新題題數：", n)
sys.exit(0 if all(x >= 192 for x in n.values()) else 1)
PY
echo "$JOB 結束 $(date -u +%FT%TZ)"
