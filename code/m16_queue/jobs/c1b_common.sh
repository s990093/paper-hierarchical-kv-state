# C1b 正式執行的共用本體（由 38da_c1b.sh、38db_c1b.sh source；job 先設 JOB、BUDGET_MIN、DO_QTIME）。
# **這個檔不是 job，不要放進佇列**（runner 會把佇列裡的 *.sh 都當成 job）；job 一律用 repo 裡的路徑 source 它。
# 設計、判定、時間估計：docs/stage3_20261019/c1b_howto.md。
#
# 流程：
#   0. pool-status：全部題目都寫完了 → 不佔 GPU、exit 0。
#   1. 兩張卡各一個行程：selfcheck（卡 0 Llama、卡 1 Qwen）；DO_QTIME＝1（或 auto 且 results 裡還沒有 qtime_<模型>.csv）時接著跑 qtime
#      （這時每張卡只有這一個行程＝閒）。selfcheck 不過 → NEEDS_HUMAN、exit 1（規則 2）。qtime 被污染（exit 3）→ 記 NEEDS_HUMAN、品質照跑。
#   2. 兩卡共用工作池：固定的工作清單＝(模型, shard i/N)，兩個模型交錯（llama s0、qwen s0、llama s1、…，共 2N 個）。
#      每張卡 S 個 worker，各自依序領下一個還沒被領的工作（m16_c1b_restore.py pool-claim：os.mkdir 當領取鎖，原子）→ 在自己那張卡上跑
#      `run --shard i/N`，跑完再領，直到清單領完或時間預算用完。所以兩張卡都忙到工作池空，不會一張先閒下來。
#      S（每張卡同時幾個行程）照 selfcheck 量到的峰值定：S＝min(C1B_S, ⌊176 GiB ÷ (1.15 × 兩個模型裡較大的 peak_reserved)⌋)，至少 1。
#      〔估〕peak ≤ 45 GiB（C1a 冒煙實測 Llama 25.2／Qwen 39.9 GiB，＋本程式多留 64 個 BF16 block：Llama 4 GiB、Qwen 0.9 GiB）→ S＝3。
#   3. 逐題續跑：run 會跳過同目錄任何 c1b_<模型>_s*.csv.gz 裡已寫完的題；被砍的題整題重做，report 只取同一次嘗試的齊全列。
#      已全部寫完的 shard，pool-claim 直接標 SKIPPED，不載模型。
# 任何一個 run 失敗 → NEEDS_HUMAN、在工作池放 STOP（其他 worker 不再領新工作）、job exit 1（規則 2）。
# 同一張卡上的其他行程會被 GpuWatcher 記成入侵者（run 目錄留 CONTAMINATED）：品質實驗的規則照用——分數保留、t_gen_s 作廢（CLAUDE.md §3）。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
C=$RES/c1_selective_restore                    # 進 git 的摘要（report、qtime）
CR=${C1B_RAW:-/mlsteam/data/tiara/stage3/c1b}  # 逐列原始輸出（> 1 MB，不進 git）
mkdir -p $C $CR
export C1B_RAW=$CR
PY="python code/m16_c1b_restore.py"
N=${C1B_N:-12}             # 每個模型切幾份（工作數＝2N）。38da、38db 用同一個值（續跑是逐題的，改了也不會重做，只是 shard 對到的題不同）
S_MAX=${C1B_S:-3}          # 每張卡最多同時幾個行程（再照 selfcheck 的峰值往下調）
MIN_LEFT=${C1B_MIN_LEFT:-10}   # 剩不到這麼多分鐘就不再領新工作（載模型 1–2 分鐘＋至少幾題）
T0=$(date +%s)
left() { echo $(( BUDGET_MIN - ($(date +%s) - T0) / 60 )); }

# ---- 0. 全部做完了 → 不佔 GPU
st=$($PY pool-status --n $N --raw-dir $CR) || { nh "$JOB: pool-status 失敗（CPU；看 job log）"; exit 1; }
echo "$st"
if [ "$(echo "$st" | head -1)" = "ALL_COMPLETE" ]; then echo "$JOB：C1b 的題已全部寫完，不佔 GPU $(date -u +%FT%TZ)"; exit 0; fi

# ---- 1. selfcheck（＋qtime）
SM=$SMOKE/c1b_${JOB}_$(date -u +%Y%m%dT%H%M%SZ)_$$; mkdir -p $SM
check() {   # check <模型>
  local m=$1 rc
  m7run m16-c1b-selfcheck-$m $PY selfcheck --model $m --out $SM/c1b_selfcheck_$m.json \
    || { nh "$JOB: selfcheck $m 沒過（$(lastrun m16-c1b-selfcheck-$m)；看 $SM/c1b_selfcheck_$m.json 的 pass）"; return 1; }
  if [ "$DO_QTIME" = 1 ] || { [ "$DO_QTIME" = auto ] && [ ! -s $C/qtime_$m.csv ]; }; then
    m7run m16-c1b-qtime-$m $PY qtime --model $m --out $C/qtime_$m.csv; rc=$?
    case $rc in
      0) ;;
      3) nh "$JOB: qtime $m 的 GPU 不乾淨（$(lastrun m16-c1b-qtime-$m)）→ Quest 打分時間 NOT_MEASURED，沒寫進 results；品質照跑" ;;
      *) nh "$JOB: qtime $m 失敗（$(lastrun m16-c1b-qtime-$m)）"; return 1 ;;
    esac
  fi
  return 0
}
oncard 0 check llama & p0=$!
oncard 1 check qwen & p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
[ $r0 = 0 ] && [ $r1 = 0 ] || exit 1
S=$(python3 -I - $SM/c1b_selfcheck_llama.json $SM/c1b_selfcheck_qwen.json $S_MAX <<'PY'
import json, math, sys
pk = max(json.load(open(p))["peak_reserved_gib"] for p in sys.argv[1:3])
s = max(1, min(int(sys.argv[3]), math.floor(176 / (1.15 * pk))))
print(s)
print(f"selfcheck 峰值（reserved，兩個模型較大者）{pk} GiB → 每張卡 {s} 個行程", file=sys.stderr)
PY
) || { nh "$JOB: 讀不到 selfcheck 的峰值（$SM）"; exit 1; }
echo "每張卡 S=$S 個行程；工作池 2×$N 個工作；預算 $BUDGET_MIN 分鐘（已用 $(( ($(date +%s) - T0) / 60 ))）"

# ---- 2. 工作池
POOL=$CR/pool_${JOB}_$(date -u +%Y%m%dT%H%M%SZ)_$$
mkdir -p $POOL
slot() {    # slot <卡> <第幾個>：領工作 → 在這張卡上跑 → 再領，直到領完、預算到、或有人放 STOP
  local card=$1 sl=$2 u m i lf
  while :; do
    [ -e $POOL/STOP ] && return 1
    lf=$(left)
    [ $lf -le $MIN_LEFT ] && { echo "卡 $card worker $sl：剩 $lf 分鐘，不再領新工作"; return 0; }
    u=$($PY pool-claim --pool $POOL --n $N --raw-dir $CR --worker c${card}w$sl) \
      || { nh "$JOB: pool-claim 失敗（卡 $card worker $sl）"; touch $POOL/STOP; return 1; }
    [ -z "$u" ] && { echo "卡 $card worker $sl：工作池領完了"; return 0; }
    m=${u% *}; i=${u#* }
    echo "$(date -u +%FT%TZ) 卡 $card worker $sl 領到 $m shard $i/$N（預算剩 $lf 分鐘）"
    if oncard $card m7run m16-c1b-$m-s$i $PY run --model $m --shard $i/$N --budget-min $lf --out $CR/c1b_${m}_s$i.csv.gz; then
      $PY pool-done --pool $POOL --n $N --unit "$u" --rc 0
    else
      nh "$JOB: $m shard $i/$N 失敗（卡 $card；$(lastrun m16-c1b-$m-s$i)）"
      $PY pool-done --pool $POOL --n $N --unit "$u" --rc 1
      touch $POOL/STOP
      return 1
    fi
  done
}
pids=()
for sl in $(seq 1 $S); do
  for card in 0 1; do slot $card $sl & pids+=($!); sleep 3; done   # 錯開幾秒：同時載模型的峰值不要疊在一起
done
rc=0
for p in "${pids[@]}"; do wait $p || rc=1; done
$PY pool-status --n $N --raw-dir $CR --pool $POOL
/bin/ls -la $CR | tail -n 30
echo "$JOB 結束（rc=$rc）$(date -u +%FT%TZ)"
