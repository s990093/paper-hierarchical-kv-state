# 10/19 02:2xZ：容器重啟（加第二張卡）前備份；重啟後用這份取代 44a_longctx_speed.running。
# 已經完成、gpu_guard 全 CLEAN、2 個 launch 都有的格子跳過（不重量）；跑到一半被重啟砍掉的那格（20261018-234358-…-llama-129024-int4_kivi）重跑。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
LC=$REPO/results/m10_longctx_mi300x; mkdir -p $LC
done_ok() { python3 - "$1" <<'PY'
import sys, glob, json, os
sys.path.insert(0, "/mlsteam/workspace/paper-hierarchical-kv-state/code")
import m10_restore2 as R
d = sys.argv[1]
rows = R.latest_rows(d)
ok = bool(rows) and R.usable(d, rows) and len(glob.glob(os.path.join(d, f"gpu_guard_{rows[0]['run_id']}_*.json"))) == 2 \
     and os.path.exists(os.path.join(d, "summary.csv"))
sys.exit(0 if ok else 1)
PY
}
for tier in fp8 int4; do
  for L in 129024 65536; do
    for m in llama qwen; do
      G=$(fmt_of $m fp8); F=$(fmt_of $m $tier); D=$LC/speed_${m}_${L}_$F; mkdir -p $D
      if done_ok $D; then echo "44a：$m $L $F 已完成（CLEAN），跳過"; continue; fi
      m7run m10-lc-grid-$m-$L-$F bash -c "python $REPO/code/m10_speed.py grid --model $m --fmt $F --base-fmt $G --out-dir $D --bw-json $OUT/speed_$m/bw.json --main-length $L --lengths --launches 2 --reps 5 --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" || echo "44a：長 context 速度 $m $L $F 失敗（見 runs/*-m10-lc-grid-$m-$L-$F）" >> $JOBS/NEEDS_HUMAN
    done
  done
done
