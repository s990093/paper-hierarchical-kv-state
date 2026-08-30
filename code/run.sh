#!/usr/bin/env bash
# 可追溯 run 的統一入口 — 實作 CLAUDE.md §4.1 的記錄協定。
#
# 每一次會產生數字的執行都要走這個殼，讓 results/*.csv 的每一列
# 都能用 run_id 反查到「哪一條指令、什麼時候、在哪張卡、哪個 commit」。
#
# 用法:
#   code/run.sh <短名> <指令...>
#   CUDA_VISIBLE_DEVICES=3 code/run.sh m1-qwen-131072 vllm serve Qwen/... --max-model-len 131072
#
# 產出 $BIG/runs/<run_id>/ ：
#   cmd.sh  context.json  stdout.log  stderr.log  exit_code  gpu_before.csv  gpu_after.csv
# run_id 會印到 stdout 最後一行，方便呼叫端接走寫進 CSV。

set -uo pipefail

BIG="${PAPER_HKV_BIG:-/ssd7/hungwei/paper-hkv}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ $# -lt 2 ]; then
  echo "usage: $0 <short-name> <command...>" >&2
  exit 2
fi

SHORT="$1"; shift
RUN_ID="$(date +%Y%m%d-%H%M%S)-${SHORT}"
RUN="$BIG/runs/$RUN_ID"
mkdir -p "$RUN"

# 1. 先把指令本身存成可重跑的檔案（不是只記在 log 裡）
{
  echo '#!/usr/bin/env bash'
  echo '# 由 code/run.sh 自動產生，可直接重跑'
  echo "set -x"
  printf '%q ' "$@"
  echo
} > "$RUN/cmd.sh"
chmod +x "$RUN/cmd.sh"

# 2. 環境快照
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu \
  --format=csv > "$RUN/gpu_before.csv" 2>/dev/null || true

python3 - "$RUN/context.json" "$RUN_ID" "$REPO" <<'PY' 2>/dev/null || true
import json, os, subprocess, sys, socket, datetime
out, run_id, repo = sys.argv[1], sys.argv[2], sys.argv[3]
def sh(*c):
    try:
        return subprocess.run(c, capture_output=True, text=True,
                              cwd=repo, timeout=20).stdout.strip() or None
    except Exception:
        return None
json.dump({
    "run_id": run_id,
    "ts": datetime.datetime.now().astimezone().isoformat(),
    "host": socket.gethostname(),
    "cwd": os.getcwd(),
    "git_commit": sh("git", "rev-parse", "HEAD"),
    "git_dirty": bool(sh("git", "status", "--porcelain")),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset:all>"),
    "hf_home": os.environ.get("HF_HOME"),
    "python": sys.executable,
}, open(out, "w"), indent=2, ensure_ascii=False)
PY

# 3. 跑，stdout/stderr 分開存
START=$(date +%s)
"$RUN/cmd.sh" > "$RUN/stdout.log" 2> "$RUN/stderr.log"
RC=$?
END=$(date +%s)

echo "$RC" > "$RUN/exit_code"
echo "$((END - START))" > "$RUN/duration_s"
nvidia-smi --query-gpu=index,memory.used,memory.total \
  --format=csv > "$RUN/gpu_after.csv" 2>/dev/null || true

# 4. 回報。失敗時把錯誤帶到眼前 —— EXPERIMENT_PLAN.md §0 禁令 2：不准跳過失敗。
echo "run_id=$RUN_ID  exit=$RC  duration=$((END - START))s  dir=$RUN"
if [ "$RC" -ne 0 ]; then
  echo "--- stderr (tail 40) ---" >&2
  tail -40 "$RUN/stderr.log" >&2
fi
exit "$RC"
