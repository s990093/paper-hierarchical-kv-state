# 10/20 13:5xZ：90 的 guard 從外面包，把 vLLM 關 engine 那幾秒也看進去 → 收尾假污染（09-15 同型）。
# 改在 m10_vllm_ttft.py 裡開 guard（--guard-gpus 0 1；從開 engine 前到最後一次量完）。量時間 → 兩張卡獨佔；a6:00.0。輸出 vllm_ttft_guarded2/。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
mkdir -p $OUT/vllm_ttft_guarded2
for m in llama qwen; do for c in default 512; do
  m7run m10-vllmg2-$m-$c env PATH=/opt/rocm/bin:$PATH /mlsteam/workspace/venv/tiara-v028/bin/python $REPO/code/m10_vllm_ttft.py --model $m --chunking $c --out-dir $OUT/vllm_ttft_guarded2 --guard-gpus 0 1 \
    || echo "91：vLLM TTFT（程式內 guard）$m $c 失敗或被污染（見 runs/*-m10-vllmg2-$m-$c）" >> $JOBS/NEEDS_HUMAN
done; done
exit 0
