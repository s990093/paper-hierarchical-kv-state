# H200 才有的 stage（13_h200_repro §13）：attention 對照組的 GPU 自測再跑一次，加 M10_ATTN_BM64_CHECK=1——
# BM＝16（decode）時 Triton 在 H200 走 mma.sync，融合讀法和讀 BF16 的運算元排法不同（kWidth 4 vs 2），同一條指令裡沿 k 的加總順序不同，
# 所以不逐位元相同；這時同一個 split 改 BM＝64（wgmma）再比一次，逐位元相同才算過（證明拆出來的值相同）。對 SDPA 的誤差門檻不變（0.05）。
# 43a／43e0 那兩次沒過的紀錄（smoke/attn_selftest_gpu.json）留著不蓋。過 → OK_attn（43eb 才跑 (b)）；沒過 → NEEDS_HUMAN。exit 0。
source __JOBS__/lib.sh
set -uo pipefail
S=$REPO/results/m10_restore2_h200/smoke; mkdir -p $S
m7run m10h-rs2-attn-selftest3-bm64 env M10_ATTN_BM64_CHECK=1 python $REPO/code/m10_attn_q.py selftest --device cuda --out $S/attn_selftest_gpu_bm64.json \
  && touch $S/OK_attn || echo "43e0nv：attention 對照組自測（加 BM＝64 檢查）還是沒過（$S/attn_selftest_gpu_bm64.json、runs/*-m10h-rs2-attn-selftest3-bm64）→ 不跑 (b)" | tee -a $JOBS/NEEDS_HUMAN
exit 0
