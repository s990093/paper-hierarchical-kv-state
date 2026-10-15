# 03 §5.2 的表（只報告）：vLLM 0.28 只算的 TTFT，預設切法 vs 512 切法；獨佔 GPU
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
mkdir -p $OUT/vllm_ttft
for m in llama qwen; do for c in default 512; do
  m7run m10-vllm-$m-$c env PATH=/opt/rocm/bin:$PATH /mlsteam/workspace/venv/tiara-v028/bin/python $REPO/code/m10_vllm_ttft.py --model $m --chunking $c --out-dir $OUT/vllm_ttft || exit 1
done; done
