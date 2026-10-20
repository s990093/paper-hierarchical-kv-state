# 疊加實驗（X1／X2，code/m16_xstack.py）的 job 共用設定與函式。由 02_smoke_x、36*、37* 在 lib.sh 之後 source。
# **不要把這個檔放進佇列目錄**（runner 會把佇列裡的 *.sh 都當成 job）；job 一律用 repo 裡的路徑 source 它。
# 說明：docs/stage3_20261019/x_stack_howto.md。
XR=$RES/x_stack                                   # 進 git 的摘要（report、x2_selection.json、x2_feasible.csv、校準摘要）
XD=/mlsteam/data/tiara/stage3/xstack              # 逐列原始輸出、校準基底（不進 git）
mkdir -p $XR $XD/x1 $XD/x2 $XD/smoke
XPY="python code/m16_xstack.py"
# 每張卡同時開幾個 process：〔估〕Llama 每個 ≤50 GiB、Qwen ≤35 GiB（DS1 實測 peak 39.9／25.3 GiB＋本程式多的 KV 暫存）。
# 冒煙（02_smoke_x）的 log 會印 peak，量到之後改這裡（或在 job 前 export）。NEED-GB 要跟著改。
X_SHARDS_LLAMA=${X_SHARDS_LLAMA:-3}
X_SHARDS_QWEN=${X_SHARDS_QWEN:-4}
X2_SHARDS_LLAMA=${X2_SHARDS_LLAMA:-3}
X2_SHARDS_QWEN=${X2_SHARDS_QWEN:-4}
X2L_SHARDS_LLAMA=${X2L_SHARDS_LLAMA:-1}           # 128K：Llama 一個 process 約 80–100 GiB〔估〕
X2L_SHARDS_QWEN=${X2L_SHARDS_QWEN:-2}
X1_N=${X1_N:-24}                                  # X1 每個任務幾題（預設＝DS1／S2 的 24；改小的話題目是前面那幾題，續跑安全）
X2_N=${X2_N:-24}                                  # X2 32K 每個任務幾題（新題 seed 62000 起）

x_calib() {   # x_calib <模型>：已有同設定、sha256 對得上的校準檔就跳過（m16_xstack 自己核對，不 import torch）
  m7run m16-x-calib-$1 $XPY calib --model $1 --out $XD/calib_$1.pt
}

x_par() {     # x_par <run 短名> <N> <指令…>：指令裡的 {s}、{N} 換成 shard 編號與份數；N 個並行，等全部結束；任一失敗回 1
  local name=$1 N=$2; shift 2
  local s p rc=0 args pids=()
  for s in $(seq 0 $((N - 1))); do
    args=("${@//\{s\}/$s}")
    args=("${args[@]//\{N\}/$N}")
    m7run $name-s$s "${args[@]}" & pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || rc=1; done
  return $rc
}

x1_model() {  # x1_model <模型> <shard 數> <分鐘預算>：先校準（有就跳過）再 screen；逐題續跑
  local m=$1 N=$2 B=$3
  x_calib $m || { nh "X1 $m：calib 失敗（$(lastrun m16-x-calib-$m)）"; return 1; }
  x_par m16-x1-$m $N $XPY screen --model $m --n $X1_N --calib $XD/calib_$m.pt --shard {s}/{N} --budget-min $B \
    --out $XD/x1/x1_${m}_s{s}.csv.gz || { nh "X1 $m：有 shard 失敗（$(lastrun m16-x1-$m)）"; return 1; }
}

x2_model() {  # x2_model <模型> <32K|128K> <core|x3|ext|128k> <shard 數> <分鐘預算>
  local m=$1 ctx=$2 tier=$3 N=$4 B=$5 nn=""
  [ $ctx = 32K ] && nn="--n $X2_N"
  x_par m16-x2-$m-$ctx-$tier $N $XPY gen --model $m --ctx $ctx --tier $tier $nn --selection $XR/x2_selection.json \
    --calib $XD/calib_$m.pt --shard {s}/{N} --budget-min $B --out $XD/x2/x2_${m}_${ctx}_${tier}_s{s}.csv.gz \
    || { nh "X2 $m $ctx $tier：有 shard 失敗（$(lastrun m16-x2-$m-$ctx-$tier)）"; return 1; }
}

x2_seq() {    # x2_seq <模型> <shard 數> <總分鐘預算> <tier…>：32K 各層照順序跑、共用一個時間預算（沒做完的留給下一段 job）
  local m=$1 N=$2 B=$3; shift 3
  local t0 tier left
  t0=$(date +%s)
  for tier in "$@"; do
    left=$(( B - ($(date +%s) - t0) / 60 ))
    [ $left -le 5 ] && { echo "X2 $m：時間預算用完，$tier 留給下一段"; break; }
    x2_model $m 32K $tier $N $left || return 1
  done
}

x2_go() {     # x2_selection.json 的 go：0＝跑、3＝X1 判定不跑（照實記 log）、其他＝讀不到（錯誤）
  python3 -c "import json,sys; d=json.load(open('$XR/x2_selection.json')); print('X2 go =', d['go'], '／', d.get('reason')); sys.exit(0 if d['go'] else 3)"
}

x2_gate() {   # 37* 開頭用：go＝false → exit 0（不佔 GPU）；讀不到 → NEEDS_HUMAN、exit 1
  x2_go; local g=$?
  [ $g = 3 ] && { echo "X1 的判定是不跑 X2（見 $XR/x2_selection.json 的 reason）→ $1 不做事"; exit 0; }
  [ $g = 0 ] || { nh "$1：讀不到 $XR/x2_selection.json（36z_x1_report 跑了嗎？）"; exit 1; }
}
