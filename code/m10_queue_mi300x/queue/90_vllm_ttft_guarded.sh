# 10/20 09:5xZ：70（vLLM TTFT）重量一次、這次有 gpu_guard。70 當時兩張卡獨佔，但 m10_vllm_ttft.py 沒開 guard，乾淨沒有紀錄（CLAUDE.md §3）。
# 量時間 → 兩張卡獨佔（沒寫 GPU-NEED＝both）；a6:00.0。指令和 70 相同，只多包一層 code/m10_guarded_run.py（兩張卡都看）；輸出到 vllm_ttft_guarded/。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
mkdir -p $OUT/vllm_ttft_guarded
for m in llama qwen; do for c in default 512; do
  m7run m10-vllmg-$m-$c env PATH=/opt/rocm/bin:$PATH python $REPO/code/m10_guarded_run.py --gpus 0 1 --out $OUT/vllm_ttft_guarded -- /mlsteam/workspace/venv/tiara-v028/bin/python $REPO/code/m10_vllm_ttft.py --model $m --chunking $c --out-dir $OUT/vllm_ttft_guarded \
    || echo "90：guarded vLLM TTFT $m $c 失敗或被污染（見 runs/*-m10-vllmg-$m-$c）" >> $JOBS/NEEDS_HUMAN
done; done
exit 0
