#!/usr/bin/env bash
# =============================================================================
# gpu_wait_run.sh — 等到有乾淨的 GPU 再跑（2026-10-18；使用者：「別人在跑就等，寫腳本看合適沒有」）
#
#   用法：bash code/gpu_wait_run.sh [選項] <短名> <指令...>
#     --need N        要幾張卡（1 或 2；2 張時另設 TIARA_TWO_GPUS=1，只給不計時的檢查用，見 run3090.sh）。預設 1
#     --gpus "0 2 4"  只從這些卡裡挑。預設全部
#     --stable K      同一張卡要連續 K 次檢查都乾淨才算可用（避免剛好碰到別人的空檔）。預設 2
#     --interval S    每 S 秒檢查一次。預設 120
#     --timeout S     最多等 S 秒，等不到就放棄（exit 5）。預設 28800（8 小時）
#     --dry-run       只印會選哪張卡，不跑
#
#   流程：每 interval 秒跑一次 `gpu_guard.py --idle-gpus`，記每張卡「連續乾淨」的次數；
#   湊滿 need 張連續 K 次乾淨的卡，就設 CUDA_VISIBLE_DEVICES 交給 code/run3090.sh（它開跑前會再 --check 一次，
#   跑的時候用 GpuWatcher 監看；被污染一樣留 CONTAMINATED）。
#   gpu_guard 查不到（nvidia-smi 失敗）時，那一輪當成「沒有乾淨的卡」，不當成乾淨（CLAUDE.md 規則 7）。
# =============================================================================
set -uo pipefail

main() {
  local need=1 gpus="" stable=2 interval=120 timeout=28800 dry=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --need) need=$2; shift 2 ;;
      --gpus) gpus=$2; shift 2 ;;
      --stable) stable=$2; shift 2 ;;
      --interval) interval=$2; shift 2 ;;
      --timeout) timeout=$2; shift 2 ;;
      --dry-run) dry=1; shift ;;
      --) shift; break ;;
      -*) echo "[gpu_wait_run] 未知選項 $1" >&2; return 2 ;;
      *) break ;;
    esac
  done
  [ $# -ge 2 ] || [ "$dry" = 1 ] || { echo "用法: $0 [選項] <短名> <指令...>" >&2; return 2; }
  [[ "$need" =~ ^[12]$ ]] || { echo "[gpu_wait_run] --need 只能是 1 或 2" >&2; return 2; }

  local here repo; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; repo="$(cd "$here/.." && pwd)"
  declare -A streak=()
  local t0; t0=$(date +%s)
  while :; do
    local idle rc
    idle=$(python3 "$repo/code/gpu_guard.py" --idle-gpus 2>&1); rc=$?
    if [ $rc -gt 1 ] || ! [[ "$idle" =~ ^[0-9\ ]*$ ]]; then   # rc 1＝沒有乾淨的卡；其他＝查不到
      echo "[gpu_wait_run] $(date -Is) gpu_guard 查不到（rc=$rc），這一輪當成沒有乾淨的卡：${idle//$'\n'/ | }" >&2
      idle=""
    fi
    local g ok=()
    for g in $(seq 0 15); do
      if [[ " $idle " == *" $g "* ]] && { [ -z "$gpus" ] || [[ " $gpus " == *" $g "* ]]; }; then
        streak[$g]=$(( ${streak[$g]:-0} + 1 ))
      else
        streak[$g]=0
      fi
      [ "${streak[$g]}" -ge "$stable" ] && ok+=("$g")
    done
    local waited=$(( $(date +%s) - t0 ))
    if [ ${#ok[@]} -ge "$need" ]; then
      local pick="${ok[0]}"; [ "$need" = 2 ] && pick="${ok[0]},${ok[1]}"
      echo "[gpu_wait_run] $(date -Is) 等了 ${waited}s，選 GPU $pick（連續 ${stable} 次乾淨；這一輪乾淨的：${idle:-無}）" >&2
      if [ "$dry" = 1 ]; then echo "$pick"; return 0; fi
      if [ "$need" = 2 ]; then
        CUDA_VISIBLE_DEVICES=$pick TIARA_TWO_GPUS=1 bash "$repo/code/run3090.sh" "$@"; return $?
      fi
      CUDA_VISIBLE_DEVICES=$pick bash "$repo/code/run3090.sh" "$@"; return $?
    fi
    if [ "$waited" -ge "$timeout" ]; then
      echo "[gpu_wait_run] $(date -Is) 等了 ${waited}s 還是沒有 ${need} 張連續乾淨的卡，放棄（exit 5）。最後一輪乾淨的：${idle:-無}" >&2
      return 5
    fi
    echo "[gpu_wait_run] $(date -Is) 還沒有可用的卡（乾淨的：${idle:-無}；要 ${need} 張、各連續 ${stable} 次），${interval}s 後再看" >&2
    sleep "$interval"
  done
}
main "$@"
exit $?
