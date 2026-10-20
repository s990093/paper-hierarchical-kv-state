# 10/20 13:5xZ：91 裡 guard 不是全 CLEAN 的組合重量（91 的 llama-512 是 gpu_guard.foreign_on 的時序 bug：先算自己人、後列行程，
# 中間剛 spawn 的 EngineCore 被當外人；已修成列完行程再算自己人）。量時間 → 兩張卡獨佔；a6:00.0。輸出 vllm_ttft_guarded3/。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
mkdir -p $OUT/vllm_ttft_guarded3
todo=$(python3 -c "
import json
m = json.load(open('$OUT/vllm_ttft_guarded2/vllm_meta.json'))
want = {(a, b) for a in ('llama', 'qwen') for b in ('default', '512')}
ok = {tuple(k.split('|')) for k, v in m.items() if isinstance(v.get('guard'), dict) and set(v['guard'].values()) == {'CLEAN'}}
print(' '.join(f'{a}|{b}' for a, b in sorted(want - ok)))
")
echo "92：要重量的組合：${todo:-（沒有）}"
for mc in $todo; do m=${mc%|*}; c=${mc#*|}
  m7run m10-vllmg3-$m-$c env PATH=/opt/rocm/bin:$PATH /mlsteam/workspace/venv/tiara-v028/bin/python $REPO/code/m10_vllm_ttft.py --model $m --chunking $c --out-dir $OUT/vllm_ttft_guarded3 --guard-gpus 0 1 \
    || echo "92：vLLM TTFT（程式內 guard、修過時序）$m $c 失敗或被污染（見 runs/*-m10-vllmg3-$m-$c）" >> $JOBS/NEEDS_HUMAN
done
exit 0
