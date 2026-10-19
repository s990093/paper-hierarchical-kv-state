# GPU-NEED: one　（10/19 雙卡：只量品質（vLLM 生成計分））
# cards/V2 的 ❌（T5 cwe）：同一份上游 RULER jsonl 改用 vLLM 0.28 跑（只有 E0，只報告）
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
mkdir -p $OUT/ruler_cwe_vllm
m7run m10-cwe-vllm env PATH=/opt/rocm/bin:$PATH /mlsteam/workspace/venv/tiara-v028/bin/python $REPO/code/m10_ruler_cwe_vllm.py --jsonl /mlsteam/data/tiara/runs/20261015-061335-m10-ruler-cwe-llama/ruler_cwe/cwe/validation.jsonl --out $OUT/ruler_cwe_vllm/cwe_vllm_llama.csv || { echo "75：RULER CWE（vLLM）失敗（見 runs/*-m10-cwe-vllm）" >> $JOBS/NEEDS_HUMAN; exit 0; }
