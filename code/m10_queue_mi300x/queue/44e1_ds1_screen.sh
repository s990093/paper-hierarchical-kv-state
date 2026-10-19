# GPU-NEED: one　（14 §8.4 DS1 海選：只量品質的 teacher forcing，只用 main 那張卡）
# 每個模型 T2–T5 各 24 題（seed 60000 起），6 個 shard（10/19：兩張卡分著跑，切細一點比較平均）；逐列 CSV 放 /mlsteam/data/tiara/drop_m10（很大，不進 git），表進 results。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
[ -e $JOBS/smoke_md/NO_DS ] && { echo "44e0 冒煙沒過 → 跳過 DS1" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
RAW=/mlsteam/data/tiara/drop_m10; mkdir -p $RAW; D=$REPO/results/m10_drop_mi300x/ds1; mkdir -p $D
python3 - <<PY
import json
jobs = []
for m, gb in (("llama", 40), ("qwen", 30)):
    for s in range(6):
        jobs.append(dict(name=f"ds1-{m}-s{s}", need_gb=gb,
                         cmd=f"m7run m10-ds1-{m}-s{s} python $REPO/code/m10_drop.py screen --model {m} --n 24 --shard {s}/6 --out $RAW/ds1_{m}_s{s}.csv.gz"))
json.dump(jobs, open("$JOBS/jobs_ds1.json", "w"), indent=1)
PY
pool $JOBS/jobs_ds1.json 6 || { echo "44e1：DS1 pool 有 job 失敗（$JOBS/pool_ds1-*.log）→ 只用跑完的題做表" | tee -a $JOBS/NEEDS_HUMAN; }
python3 $REPO/code/m10_drop.py screen-report --raw "$RAW/ds1_*.csv.gz" --out-dir $D > $D/screen_report.out 2>&1 \
  || { echo "44e1：DS1 表失敗（$D/screen_report.out）" | tee -a $JOBS/NEEDS_HUMAN; touch $JOBS/smoke_md/NO_DS2; }
exit 0
