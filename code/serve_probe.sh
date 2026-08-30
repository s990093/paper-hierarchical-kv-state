#!/usr/bin/env bash
# 啟動一個 vLLM server、確認它能回應、然後關掉。
# EXPERIMENT_PLAN.md 驗收 A2（能跑）與 A3（卸載連接器可用）都用這支。
#
# 用法:
#   code/serve_probe.sh <model> <max_model_len> [kv_transfer_config_json]
#
# 例：
#   A2: code/serve_probe.sh Qwen/Qwen3-0.6B 8192
#   A3: code/serve_probe.sh Qwen/Qwen3-0.6B 8192 '{"kv_connector":"OffloadingConnector",...}'
#
# 通過條件（兩者都要）：
#   1. /v1/completions 回傳合法 JSON 且 choices 非空
#   2. 給了 kv_transfer_config 時，server log 出現 offloading 相關訊息
#
# 退出碼 0 = 通過。非 0 一律視為失敗，不要重試到有輸出為止（禁令 2）。

set -uo pipefail

MODEL="${1:?model required}"
MAXLEN="${2:?max_model_len required}"
KVCFG="${3:-}"

BIG="${PAPER_HKV_BIG:-/ssd7/hungwei/paper-hkv}"
PORT="${PORT:-$((18000 + RANDOM % 900))}"
GPU_UTIL="${GPU_UTIL:-0.90}"
TIMEOUT="${TIMEOUT:-600}"
VENV="$BIG/venv/vllm"

# venv/bin 必須進 PATH：flashinfer 會 JIT 編譯 sampling kernel，需要 ninja 與 nvcc，
# 而它們是以 pip 套件裝在 venv 裡的。直接呼叫 $VENV/bin/vllm 而不 activate 時，
# EngineCore 子行程看不到它們 -> FileNotFoundError: 'ninja'。（2026-08-30 實測）
export PATH="$VENV/bin:${PATH}"
# JIT / 編譯快取一律寫 /ssd7，不要污染 $HOME
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$BIG/xdg-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$BIG/triton-cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$BIG/vllm-cache}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$BIG/flashinfer-cache}"
mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$FLASHINFER_WORKSPACE_BASE"

TAG="$(echo "$MODEL" | tr '/' '-')-len$MAXLEN$([ -n "$KVCFG" ] && echo '-kv')"
OUT="$BIG/runs/$(date +%Y%m%d-%H%M%S)-probe-$TAG"
mkdir -p "$OUT"

cmd=("$VENV/bin/vllm" serve "$MODEL"
     --port "$PORT"
     --max-model-len "$MAXLEN"
     --gpu-memory-utilization "$GPU_UTIL")
[ -n "$KVCFG" ] && cmd+=(--kv-transfer-config "$KVCFG")

{
  echo "model=$MODEL max_model_len=$MAXLEN port=$PORT gpu_util=$GPU_UTIL"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "kv_transfer_config=${KVCFG:-<none>}"
  printf '%q ' "${cmd[@]}"; echo
} > "$OUT/cmd.txt"
echo "[probe] out=$OUT port=$PORT"

"${cmd[@]}" > "$OUT/server.log" 2>&1 &
SRV=$!
cleanup() { kill -TERM "$SRV" 2>/dev/null; sleep 3; kill -KILL "$SRV" 2>/dev/null; }
trap cleanup EXIT

# 等 server 起來（或死掉）
READY=0
for _ in $(seq 1 "$TIMEOUT"); do
  if ! kill -0 "$SRV" 2>/dev/null; then
    echo "[probe] FAIL server exited early" | tee -a "$OUT/result.txt"
    echo "--- server.log tail 40 ---"; tail -40 "$OUT/server.log"
    echo "FAIL_SERVER_DIED" > "$OUT/verdict"; exit 1
  fi
  if curl -sf "http://127.0.0.1:$PORT/health" -o /dev/null 2>/dev/null; then READY=1; break; fi
  sleep 1
done
if [ "$READY" -ne 1 ]; then
  echo "[probe] FAIL timeout after ${TIMEOUT}s" | tee -a "$OUT/result.txt"
  tail -40 "$OUT/server.log"
  echo "FAIL_TIMEOUT" > "$OUT/verdict"; exit 1
fi

nvidia-smi --query-gpu=index,memory.used --format=csv > "$OUT/gpu_loaded.csv" 2>/dev/null

# 驗收 1：真的產生 token
curl -s "http://127.0.0.1:$PORT/v1/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"hello\",\"max_tokens\":8,\"temperature\":0}" \
  > "$OUT/completion.json"
curl -s "http://127.0.0.1:$PORT/metrics" > "$OUT/metrics.txt" 2>/dev/null

if ! python3 -c "
import json,sys
d=json.load(open('$OUT/completion.json'))
assert d.get('choices'), 'no choices'
print('[probe] completion text:', repr(d['choices'][0]['text']))
" ; then
  echo "[probe] FAIL bad completion" | tee -a "$OUT/result.txt"
  cat "$OUT/completion.json"; echo "FAIL_COMPLETION" > "$OUT/verdict"; exit 1
fi

# 驗收 2：有給 kv config 時，log 必須看得到 offloading 真的接上
VERDICT=PASS
if [ -n "$KVCFG" ]; then
  HITS=$(grep -icE 'offload|kv_connector|OffloadingConnector|cache policy|tiering' "$OUT/server.log")
  echo "[probe] offloading log hits: $HITS"
  grep -iE 'offload|kv_connector|OffloadingConnector|cache policy|tiering' "$OUT/server.log" \
    | head -20 > "$OUT/offload_evidence.txt"
  if [ "$HITS" -lt 1 ]; then
    echo "[probe] FAIL kv_transfer_config given but no offloading in log" | tee -a "$OUT/result.txt"
    VERDICT=FAIL_NO_OFFLOAD_EVIDENCE
  fi
fi

echo "$VERDICT" > "$OUT/verdict"
echo "[probe] $VERDICT  out=$OUT"
[ "$VERDICT" = PASS ] || exit 1
