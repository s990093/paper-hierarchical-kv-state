# H200 版的 MI300X 91／92（13_h200_repro §16）：70（vLLM TTFT）當初沒有 gpu_guard 紀錄 → 用程式內 guard 重量一次
# （m10_vllm_ttft.py --guard-gpus：從開 engine 前到最後一次量完，不含關 engine）。Qwen 用拿掉 DCA 的變體（§11.1）。
# 四組（2 模型 × default／512）；不是全 CLEAN 的那組再量一次（同 92）。輸出 vllm_ttft_guarded/。量時間的 stage。exit 0。
source __JOBS__/lib.sh
set -uo pipefail
G=$(python -c "import sys;sys.path.insert(0,'$REPO/code');import gpu_guard;print(gpu_guard.smi_index(0))")
D=$OUT/vllm_ttft_guarded; mkdir -p $D
clean() { python3 - "$D" "$1" <<'PY'
import glob, json, sys
gs = [json.load(open(g)) for g in glob.glob(f"{sys.argv[1]}/gpu_guard_*{sys.argv[2]}_gpu*.json")]
sys.exit(0 if gs and all(g.get("verdict") == "CLEAN" for g in gs) else 1)
PY
}
for m in llama qwen; do for c in default 512; do
  for try in 1 2; do
    m7run m10h-vllmg$try-$m-$c env M7_QWEN_GLOB=$HOME/models/Qwen2.5-7B-Instruct-1M-noDCA python $REPO/code/m10_vllm_ttft.py --model $m --chunking $c --out-dir $D --guard-gpus $G \
      || echo "h3：vLLM TTFT（程式內 guard）$m $c 第 $try 次失敗（runs/*-m10h-vllmg$try-$m-$c）" >> $JOBS/NEEDS_HUMAN
    clean "m10h-vllmg$try-$m-$c" && break
    [ $try = 2 ] && echo "h3：vLLM TTFT $m $c 兩次都不是全 CLEAN → 不採用" >> $JOBS/NEEDS_HUMAN
  done
done; done
exit 0
