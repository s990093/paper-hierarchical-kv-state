#!/usr/bin/env bash
# =============================================================================
# oc_task.sh — 把一件工作交給 opencode 做（2026-10-18；使用者：「基本上都叫 opencode 做事，因為有免費額度」）
#
#   用法：TIARA_NO_GPU=1 bash code/run3090.sh oc-<短名> bash code/oc_task.sh <任務目錄> [選項]
#     <任務目錄>/prompt.md   要交代的事（必須先寫好）
#     --model M      預設 anthropic/claude-opus-5-5
#     --variant V    預設 high（審查用 xhigh；xhigh 想太久會用完輸出上限，見下）
#     --agent A      預設 build（會改檔）；審查用 plan（唯讀）
#     --dir D        opencode 的工作目錄。預設 repo
#     --timeout S    最多跑 S 秒。預設 28800（8 小時）
#
#   權限：OPENCODE_CONFIG＝/ssd8/hungwei/paper-hkv/opencode/work.jsonc（擋 git commit/push/reset/checkout、rm -r、sudo、
#   CUDA_VISIBLE_DEVICES=、curl/wget、pip install、改 main.tex／CLAUDE.md／04_prereg_map.md／.claude/）。
#   輸出上限：OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX＝64000。預設 32000 時，2026-10-18 的 xhigh 審查在思考中就用完，
#   最後的答案沒寫出來（finish＝length）。
#   跑完：把 session 匯出成 <任務目錄>/session_export.json，最後的文字放 final.txt，花費放 stats.txt。
# =============================================================================
set -uo pipefail

main() {
  local task="${1:?用法: oc_task.sh <任務目錄> [選項]}"; shift
  local model=anthropic/claude-opus-5-5 variant=high agent=build dir="" timeout_s=28800
  while [ $# -gt 0 ]; do
    case "$1" in
      --model) model=$2; shift 2 ;;
      --variant) variant=$2; shift 2 ;;
      --agent) agent=$2; shift 2 ;;
      --dir) dir=$2; shift 2 ;;
      --timeout) timeout_s=$2; shift 2 ;;
      *) echo "[oc_task] 未知選項 $1" >&2; return 2 ;;
    esac
  done
  local here repo; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; repo="$(cd "$here/.." && pwd)"
  [ -n "$dir" ] || dir="$repo"
  [ -s "$task/prompt.md" ] || { echo "[oc_task] 找不到 $task/prompt.md" >&2; return 2; }
  local oc="$HOME/.opencode/bin/opencode"
  export OPENCODE_CONFIG=/ssd8/hungwei/paper-hkv/opencode/work.jsonc
  export OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=64000
  echo "[oc_task] $(date -Is) model=$model variant=$variant agent=$agent dir=$dir timeout=${timeout_s}s" >&2
  timeout "$timeout_s" "$oc" run -m "$model" --variant "$variant" --agent "$agent" --dir "$dir" --format json \
    "$(cat "$task/prompt.md")" > "$task/events.jsonl"
  local rc=$?
  echo "[oc_task] $(date -Is) opencode exit=$rc" >&2
  local sid
  sid=$(grep -o '"sessionID":"ses_[A-Za-z0-9]*"' "$task/events.jsonl" | head -1 | cut -d'"' -f4)
  if [ -n "$sid" ]; then
    echo "$sid" > "$task/session_id"
    "$oc" export "$sid" > "$task/session_export.json" 2> "$task/export_stderr.log"
    python3 -I - "$task/session_export.json" "$task/final.txt" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
msgs = d.get("messages", [])
last = [m for m in msgs if m.get("info", {}).get("role") == "assistant"]
out = []
if last:
    info = last[-1]["info"]
    out.append(f"# finish={info.get('finish')} error={info.get('error')}\n")
    out += [p.get("text", "") for p in last[-1].get("parts", []) if p.get("type") == "text"]
open(sys.argv[2], "w").write("\n".join(out))
EOF
  else
    echo "[oc_task] 事件裡找不到 sessionID，沒有匯出" >&2
  fi
  "$oc" stats --days 1 --models 2>&1 | sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' > "$task/stats.txt"
  return $rc
}
main "$@"
exit $?
