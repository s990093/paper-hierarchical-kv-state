# H200 才有的 stage（13_h200_repro §13）：43e 的 (b)（attention 微測）補跑。指令和 MI300X 43e 的 (b) 一字不改
# （32K／64K／128K × 256／1 個 query × 閒／忙，decode 另加外部參考 vLLM-KIVI；kernel 設定照 MI300X：BM、split 份數、num_warps、num_stages）。
# 只報告。計時的 stage。exit 0。
source __JOBS__/lib.sh
set -uo pipefail
R=$REPO/results/m10_restore2_h200; Q=$R/q5; mkdir -p $Q
if [ -f $R/smoke/OK_attn ]; then
  for m in llama qwen; do
    m7run m10h-rs2-q5b-$m python $REPO/code/m10_attn_q.py bench --model $m --lengths 32768 65536 129024 --queries 256 1 --out-dir $Q/attn \
      || echo "43eb：Q5 (b) attention 微測 $m 失敗" >> $JOBS/NEEDS_HUMAN
  done
else
  echo "43eb：attention 自測（BM＝64 檢查）沒過 → 不跑 (b)" | tee -a $JOBS/NEEDS_HUMAN
fi
exit 0
