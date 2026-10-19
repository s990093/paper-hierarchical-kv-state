# GPU-NEED: one　（14 號的速度表；只用 CPU）
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
MA=$REPO/results/m10_match_mi300x
python3 $REPO/code/m10_drop.py speed-report --root $MA > $MA/speed_report.out 2>&1 || echo "44ez：速度表失敗（$MA/speed_report.out）" | tee -a $JOBS/NEEDS_HUMAN
find $MA -name restores.csv -size +900k -exec gzip -kf {} \;
exit 0
