# GPU-NEED: none
# 第三階段的完整 PDF 報告（code/m16_report_pdf.py build；只用 CPU、不 import torch／vllm／transformers；cpu 佇列：nice 19、0–3 號核心）。
# 輸出：/mlsteam/data/tiara/stage3/report/{stage3_report.pdf, stage3_report.html, figs/, fonts/, preview/, build_log.json}；
#       另存一份 history/stage3_report_<UTC 時間>.pdf（重跑會覆蓋主檔）。
# 圖表的數字全部在 build 當下從 results/m16_stage3_mi300x/ 讀；沒有結果的畫 NOT_MEASURED 佔位；
# 解讀從 docs/stage3_20261019/report/narrative/<節名>.html 讀（不存在 → 「（待撰寫）」框）。說明：docs/stage3_20261019/report/README.md。
# 驗收（PDF 只能有 Noto Sans CJK TC、每頁有頁碼）沒過 → NEEDS_HUMAN、exit 1（規則 2）。
# AFTER：先不寫，由協調者決定（例如等最後一批報告 job 都 done）。沒排的 job 不要寫進 AFTER（runner 要所有 AFTER 都有 .done）。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
JOB=90_report_pdf
OUTD=/mlsteam/data/tiara/stage3/report
export OMP_NUM_THREADS=4
m7run m16-report-pdf python code/m16_report_pdf.py build --out $OUTD \
  || { nh "$JOB：報告產生或驗收失敗（$(lastrun m16-report-pdf)；看 $OUTD/build_log.json）"; exit 1; }
mkdir -p $OUTD/history && cp $OUTD/stage3_report.pdf $OUTD/history/stage3_report_$(date -u +%Y%m%dT%H%MZ).pdf
echo "$JOB 結束 $(date -u +%FT%TZ)"
