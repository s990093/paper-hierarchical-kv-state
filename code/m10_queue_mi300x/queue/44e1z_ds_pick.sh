# GPU-NEED: one　（14 §8.5：照 DS1 的表寫出 DS2 的配置與 jobs；只用 CPU）
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
{ [ -e $JOBS/smoke_md/NO_DS ] || [ -e $JOBS/smoke_md/NO_DS2 ]; } && { echo "DS1 沒有表 → 跳過 DS2" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
D2=$REPO/results/m10_drop_mi300x/ds2; mkdir -p $D2; cp $REPO/results/m10_drop_mi300x/ds1/ds1_table.json $D2/ds1_table.json
python3 $REPO/code/m10_drop.py pick --out-dir $D2 --params-file $P --shards 6 --need-gb 40 --jobs-out $JOBS/jobs_ds2.json > $D2/pick.out 2>&1 \
  || { echo "44e1z：DS2 挑選失敗（$D2/pick.out）→ 跳過 DS2" | tee -a $JOBS/NEEDS_HUMAN; touch $JOBS/smoke_md/NO_DS2; }
exit 0
