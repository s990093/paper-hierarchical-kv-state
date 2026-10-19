# GPU-NEED: one　（14 §8.5 DS2 複賽：生成、兩輪、讀法 A；只量品質，只用 main 那張卡）
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
{ [ -e $JOBS/smoke_md/NO_DS ] || [ -e $JOBS/smoke_md/NO_DS2 ] || [ ! -f $JOBS/jobs_ds2.json ]; } && { echo "沒有 DS2 的 jobs → 跳過" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
D2=$REPO/results/m10_drop_mi300x/ds2
pool $JOBS/jobs_ds2.json 6 || { echo "44e2：DS2 pool 有 job 失敗（$JOBS/pool_ds2-*.log）→ 只用跑完的題做表" | tee -a $JOBS/NEEDS_HUMAN; }
python3 $REPO/code/m10_drop.py confirm-report --out-dir $D2 > $D2/confirm_report.out 2>&1 \
  || echo "44e2：DS2 判讀失敗（$D2/confirm_report.out）" | tee -a $JOBS/NEEDS_HUMAN
for f in $D2/quality_*.csv; do [ -f "$f" ] && gzip -kf $f; done
exit 0
