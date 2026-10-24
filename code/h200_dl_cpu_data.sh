#!/bin/bash
# =============================================================================
# h200_dl_cpu_data.sh — D4、D5 的 CPU 模擬要的資料，在 head node 下載（2026-10-16，使用者：「下載並核對後再跑」）
#
#   bash code/h200_dl_cpu_data.sh            # 三個 run：Mooncake trace、SCBench、D5 的 config
#   bash code/h200_dl_cpu_data.sh --dry-run
#
# MI300X 上的來源（沒有記 revision／sha256，所以「同一份」靠重跑核對：MI300X 的常數＋這裡的資料 → D4／D5 的輸出要和 git 相同）：
#   - Mooncake FAST25 trace：kvcache-ai/Mooncake 的 FAST25-release/traces/{conversation,synthetic,toolagent}_trace.jsonl
#     （E02 卡片記的 main 245e7106，2026-10-06；E10 卡片 1d0e4c75。D4 跑在 10/10）
#   - SCBench：HF dataset microsoft/SCBench 的 data/*.jsonl（MI300X 903 MB，inventory/datasets.tsv）
#   - D5：run 20261010-053802-d5-qwen3vl-cfg（Qwen3-VL-8B-Instruct 的 4 個 config）、20261010-054103-d5-cfgs（m8_arch_arith 讀的 6 個 config.json，
#     檔名＝repo id 的 / 換成 _）。MI300X 的下載腳本在 scratchpad，沒留下；這裡照 D5 §5 的清單重寫。
# 每個 run 一個 $TIARA_RUNS/<RUN_ID>/（cmd.sh、context.txt、stdout.log、stderr.log、exit_code、sha256.txt）。失敗就停（規則 2）。
# =============================================================================
set -uo pipefail
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ "$DRY" = 1 ] && export TIARA_SKIP_CONDA=1
# shellcheck disable=SC1091
source "$REPO/code/env_h200.sh"
cd "$REPO"
MOONCAKE_SHA=245e710604d46a14c46f8882e4381bd87bcd94ce

hrun() {   # hrun <短名> <bash 指令字串>：head node 版 §4.1 記錄殼
  local short=$1 cmd=$2
  local id; id="$(date +%Y%m%d-%H%M%S)-$short"; local run=$TIARA_RUNS/$id
  if [ "$DRY" = 1 ]; then echo "[$short] $cmd"; return 0; fi
  mkdir -p "$run"
  printf '%s\n' "$cmd" > "$run/cmd.sh"
  { date -Is; hostname; echo "git $(git rev-parse HEAD)"; env | grep -E '^(TIARA_|HF_HOME)' | sort; } > "$run/context.txt"
  echo "=== $(date -Is) $short → $id"
  RUN_ID=$id RUN_DIR=$run bash "$run/cmd.sh" > "$run/stdout.log" 2> "$run/stderr.log"; local rc=$?
  echo $rc > "$run/exit_code"
  if [ $rc -ne 0 ]; then echo "[h200_dl_cpu_data] $short 失敗（exit $rc）→ 停、回報。stderr：" >&2; tail -20 "$run/stderr.log" >&2; exit $rc; fi
  tail -5 "$run/stdout.log"
}

hrun h200-dl-mooncake-traces "set -euo pipefail
mkdir -p \"$TIARA_DATASETS/traces\"
for t in conversation synthetic toolagent; do
  curl -fsSL -o \"$TIARA_DATASETS/traces/\${t}_trace.jsonl\" https://raw.githubusercontent.com/kvcache-ai/Mooncake/$MOONCAKE_SHA/FAST25-release/traces/\${t}_trace.jsonl
done
cd \"$TIARA_DATASETS/traces\"; sha256sum *_trace.jsonl | tee \"\$RUN_DIR/sha256.txt\"; wc -l *_trace.jsonl; du -sh ."

hrun h200-dl-scbench "set -euo pipefail
python - <<'PY'
import os
from huggingface_hub import snapshot_download, HfApi
dst = os.path.join(os.environ['TIARA_DATASETS'], 'scbench')
info = HfApi().dataset_info('microsoft/SCBench')
print('revision', info.sha)
p = snapshot_download('microsoft/SCBench', repo_type='dataset', revision=info.sha, allow_patterns=['data/*.jsonl'], local_dir=dst)
print('->', p)
PY
cd \"$TIARA_DATASETS/scbench/data\"; sha256sum *.jsonl | tee \"\$RUN_DIR/sha256.txt\"; ls | wc -l; du -sh ."

hrun h200-dl-d5-qwen3vl-cfg "set -euo pipefail
python - <<'PY'
import json, os, shutil
from huggingface_hub import hf_hub_download, HfApi
rd = os.environ['RUN_DIR']; cd = os.path.join(rd, 'cfg'); os.makedirs(cd, exist_ok=True)
# Qwen3-VL-8B-Instruct（MI300X run 20261010-053802-d5-qwen3vl-cfg）：m8_arch_arith --cfg-vl 讀 <run>/cfg/config.json
rid = 'Qwen/Qwen3-VL-8B-Instruct'; rev = HfApi().model_info(rid).sha
for fn in ('config.json', 'preprocessor_config.json', 'video_preprocessor_config.json', 'generation_config.json'):
    shutil.copy(hf_hub_download(rid, fn, revision=rev), os.path.join(cd, fn))
json.dump({rid: rev}, open(os.path.join(rd, 'revisions.json'), 'w'), indent=1); print(rid, rev)
PY
cd \"\$RUN_DIR\"; sha256sum cfg/*.json | tee sha256.txt"

hrun h200-dl-d5-cfgs "set -euo pipefail
python - <<'PY'
import json, os, shutil
from huggingface_hub import hf_hub_download, HfApi
rd = os.environ['RUN_DIR']; cd = os.path.join(rd, 'cfg'); os.makedirs(cd, exist_ok=True)
# m8_arch_arith --cfg 讀的 6 個 config.json（MI300X run 20261010-054103-d5-cfgs；D5 §5 列了 12 個，其他 6 個用文獻表，程式不讀）
revs = {}
for rid in ('unsloth/gemma-3-12b-it', 'unsloth/gemma-3-27b-it', 'openai/gpt-oss-20b', 'openai/gpt-oss-120b',
            'nvidia/Nemotron-H-8B-Base-8K', 'Qwen/Qwen3-Next-80B-A3B-Instruct'):
    revs[rid] = HfApi().model_info(rid).sha
    shutil.copy(hf_hub_download(rid, 'config.json', revision=revs[rid]), os.path.join(cd, rid.replace('/', '_') + '.json'))
json.dump(revs, open(os.path.join(rd, 'revisions.json'), 'w'), indent=1); print(json.dumps(revs, indent=1))
PY
cd \"\$RUN_DIR\"; sha256sum cfg/*.json | tee sha256.txt"
