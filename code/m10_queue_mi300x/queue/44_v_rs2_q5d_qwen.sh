# 10/18 15:5xZ：43e 的 (d)（vLLM 0.28 的 KV decode，auto vs fp8）Qwen 兩個都啟動失敗：
#   TypeError: RocmAttentionImpl.__init__() got an unexpected keyword argument 'layer_idx'（Qwen2.5-1M 的 config 有 dual_chunk_attention_config，ROCm attention 不支援 DCA）。
# 修：m10_vllm_kvdecode.py／m10_vllm_ttft.py 的 Qwen 改用 noDCA 目錄（同一份權重 symlink，只拿掉 DCA；CLAUDE.md §3 第 3 點）。這裡只重跑 Qwen 的 (d)；Llama 的不動。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
Q=$REPO/results/m10_restore2_mi300x/q5; mkdir -p $Q/vllm_kvdecode
m=qwen
for kd in auto fp8; do
  m7run m10-rs2-q5d-$m-$kd-nodca env PATH=/opt/rocm/bin:$PATH /mlsteam/workspace/venv/tiara-v028/bin/python $REPO/code/m10_vllm_kvdecode.py --model $m --kv-dtype $kd --out-dir $Q/vllm_kvdecode \
    || echo "44_v：(d) vLLM kv=$kd $m（noDCA）又失敗（見 runs/*-m10-rs2-q5d-$m-$kd-nodca）" >> $JOBS/NEEDS_HUMAN
done
exit 0
