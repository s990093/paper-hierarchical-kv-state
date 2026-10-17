# 13 §6 (b)：attention 對照組在 GPU 上重新自測（43a 那次：fp8 讀法要 139 KB 共享記憶體 > 64 KB；改成三種讀法都 num_stages=1、
# 離線編譯量過 16–32 KB）。過 → OK_attn，43e 才跑 (b)；沒過 → NEEDS_HUMAN。exit 0。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
S=$REPO/results/m10_restore2_mi300x/smoke; mkdir -p $S
m7run m10-rs2-attn-selftest2 python $REPO/code/m10_attn_q.py selftest --device cuda --out $S/attn_selftest_gpu.json \
  && touch $S/OK_attn || echo "43e0：attention 對照組重新自測還是沒過（$S/attn_selftest_gpu.json、runs/*-m10-rs2-attn-selftest2）→ 43e 只跑 (a)、(d)" | tee -a $JOBS/NEEDS_HUMAN
exit 0
