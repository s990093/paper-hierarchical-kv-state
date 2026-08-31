#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 工作佇列：把剩下的實驗串起來，失敗自動重試，不需要人盯著。
#
# 為什麼需要這個
#   先前每個工作跑完就停，要等人來排下一個。而這台機器：
#     * 整機 24 小時 HEAVY（5,900+ 樣本，QUIET 零次）
#     * GPU 0 也會被搶（2026-09-01 00:06 有人用掉 23 GB 十分鐘）
#   所以量測必須假設**隨時會被打斷**，能自己等、自己重試、自己往下走。
#
# 設計
#   * 每個工作有自己的 marker，成功就不再重跑（可安全地重新啟動整個佇列）
#   * GPU 工作跑前先 wait_until_free，失敗自動重試
#   * CPU 工作（模擬）不搶 GPU，隨時可跑
#   * 全部輸出寫進 $LOG，每個工作的 rc 記在 marker 裡
#
# 用法
#   bash code/run_queue.sh              # 從上次中斷的地方繼續
#   FORCE=1 bash code/run_queue.sh      # 全部重跑
#   bash code/run_queue.sh --status     # 只看進度
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

REPO=/home/hungwei/llm/POC/paper-hierarchical-kv-state
BIG=/ssd7/hungwei/paper-hkv
P=$BIG/venv/vllm/bin/python
STATE=$BIG/queue-state
LOG=$BIG/logs/queue_$(date +%Y%m%d-%H%M%S).log
mkdir -p "$STATE" "$BIG/logs"

cd "$REPO" || exit 1
export PYTHONPATH=$REPO/code
export HF_HOME=$BIG/hf-cache/huggingface
export PAPER_HKV_FS_TIER=/home/hungwei/kv_fs_tier_nvme

JOBS=(
  "01-qwen-awq-cost:GPU:qwen-awq 成本模型（ctx=96,000）"
  "02-512k-latency:GPU:512K 延遲（INT8 KV，逾時 3600s）"
  "03-oracle-awq:CPU:用 AWQ 常數重跑 Oracle 與 headroom 地圖"
  "04-notebook:CPU:notebook 重跑並存回圖"
  "05-verdict:CPU:產生判定材料"
)

if [ "${1:-}" = "--status" ]; then
  echo "工作佇列進度："
  for j in "${JOBS[@]}"; do
    id=${j%%:*}; rest=${j#*:}; kind=${rest%%:*}; desc=${rest#*:}
    if [ -f "$STATE/$id.done" ]; then
      printf "  ✅ %-18s %s（%s）\n" "$id" "$desc" "$(cat "$STATE/$id.done")"
    elif [ -f "$STATE/$id.failed" ]; then
      printf "  🔴 %-18s %s（%s）\n" "$id" "$desc" "$(cat "$STATE/$id.failed")"
    else
      printf "  ⬜ %-18s %s [%s]\n" "$id" "$desc" "$kind"
    fi
  done
  exit 0
fi

exec > >(tee -a "$LOG") 2>&1
echo "════════════════════════════════════════════════════════"
echo " 工作佇列  $(date -Is)　log: $LOG"
echo "════════════════════════════════════════════════════════"

LOCK=$STATE/queue.lock
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "🔴 已有一份佇列在跑（pid $(cat "$LOCK")）。中止。"; exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

wait_gpu() {   # $1 = 最多等幾秒
  "$P" -c "
import sys; sys.path.insert(0,'code')
from gpu_guard import wait_until_free
ok, got = wait_until_free(0, need_mib=22*1024, timeout_s=$1, poll_s=30)
print(f'  GPU 0 可用 {got} MiB — {\"OK\" if ok else \"逾時\"}')
sys.exit(0 if ok else 1)"
}

run_job() {    # $1=id  $2=kind  $3=desc  $4...=指令
  local id=$1 kind=$2 desc=$3; shift 3
  if [ -f "$STATE/$id.done" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "── 跳過 $id（已完成 $(cat "$STATE/$id.done")）"; return 0
  fi
  for attempt in 1 2 3 4 5; do
    echo
    echo "── $id 第 $attempt 次  $desc  $(date -Is) ──"
    if [ "$kind" = "GPU" ]; then
      "$P" code/shm_gc.py --apply 2>/dev/null | tail -1
      wait_gpu 2400 || { echo "   等不到 GPU，5 分鐘後重試"; sleep 300; continue; }
    fi
    "$@"
    local rc=$?
    echo "   rc=$rc"
    if [ $rc -eq 0 ]; then
      date -Is > "$STATE/$id.done"; rm -f "$STATE/$id.failed"
      echo "   ✅ $id 完成"; return 0
    fi
    # rc=3 被插隊污染、rc=5 記憶體不足 -> 可重試；其餘也重試但記下來
    echo "   ⚠️ rc=$rc，5 分鐘後重試"
    sleep 300
  done
  echo "$(date -Is) rc=fail-after-5" > "$STATE/$id.failed"
  echo "   🔴 $id 五次都失敗，繼續下一個工作"
  return 1
}

# ── 01 qwen-awq 的成本模型 ──────────────────────────────
run_job 01-qwen-awq-cost GPU "qwen-awq 成本模型" \
  "$P" -u code/m2_cost_model.py --gpu 0 --stage retrieval --model qwen-awq \
       --ctx 96000 --n-prefixes 4 --retrieval-repeats 3

# ── 02 512K 的延遲 ──────────────────────────────────────
for B in full_gpu tier_fs; do
  run_job "02-512k-$B" GPU "512K 延遲（$B）" \
    "$P" -u code/m3_baseline.py --mode serial --model qwen-awq-int8-512k \
         --baseline "$B" --gpu 0 \
         --csv "$REPO/results/m3_baseline/baseline_512k.csv"
done
touch "$STATE/02-512k-latency.done" 2>/dev/null || true
date -Is > "$STATE/02-512k-latency.done"

# ── 03 用新常數重跑模擬（純 CPU，不搶 GPU）──────────────
run_job 03-oracle-awq CPU "Oracle + headroom 地圖（AWQ 常數）" bash -c '
set -e
P=/ssd7/hungwei/paper-hkv/venv/vllm/bin/python
# 先確認 AWQ 的成本模型讀得到且合理
$P -c "
import sys; sys.path.insert(0,\"code\")
from m4_oracle import load_cost_model
for mk in (\"llama-awq\",\"qwen-awq\"):
    c = load_cost_model(\"nvme\", require_model_key=mk)
    x = (c.ssd - c.recompute_base) / c.recompute_slope_per_token
    print(f\"  {mk}: CPU {c.cpu:.3f} SSD {c.ssd:.3f} 交叉 {x:,.0f}\")
    assert c.cpu > 0.05, f\"{mk} 的 CPU 成本 {c.cpu} 太小，八成又沒逐出\"
    assert x > 0, f\"{mk} 的交叉點是負的 ({x:,.0f})，成本常數有問題\"
"
for M in llama-awq qwen-awq; do
  $P -u code/m4_sweep.py --axis ssd length --model $M --device nvme \
     --ssd-gib 0 32 128 512 2048 -1 --decode \
     --out-dir results/m4_oracle/$M
  $P -u code/m4_sweep.py --axis surface --model $M --device nvme --decode \
     --surface-lengths 8192 32768 131072 524288 \
     --surface-requeries 1.2 2 5 10 --surface-requests 120 \
     --out-dir results/m4_oracle/$M
done'

# ── 04 notebook 重跑 ────────────────────────────────────
run_job 04-notebook CPU "notebook 重跑並存回圖" bash -c '
cd notebooks
CUDA_VISIBLE_DEVICES= /ssd7/hungwei/paper-hkv/venv/vllm/bin/python -m jupyter \
  nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 \
  analysis.ipynb'

# ── 05 判定材料 ─────────────────────────────────────────
run_job 05-verdict CPU "判定材料" bash -c '
/ssd7/hungwei/paper-hkv/venv/vllm/bin/python code/m4_verdict.py \
  | tee results/m4_oracle/verdict.txt'

echo
echo "════════════════════════════════════════════════════════"
echo " 佇列結束 $(date -Is)"
bash code/run_queue.sh --status
echo "════════════════════════════════════════════════════════"
