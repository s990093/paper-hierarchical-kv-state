# X4（code/m16_x4.py；38e* X4a TF、38f* X4b 生成＋dump、38g* X4c 128K、38h* X4d 兩輪、38i1 Qwen 256K）的 job 共用設定與函式。
# 由那些 job 在 lib.sh、x_common.sh、x_pool.sh 之後 source。**不要把這個檔放進佇列目錄**。說明：docs/stage3_20261019/x_stack_howto.md §15。
# 輸出：摘要 results/m16_stage3_mi300x/x_stack/x4/（不覆蓋第一版與 v2 的檔）；逐列 $XD/x4/；dump 在本地盤 $KVD/x4（可重生、容器重啟會不見）。
X4R=$RES/x_stack/x4
X4D=$XD/x4
mkdir -p $X4R $X4D/tf $X4D/gen $X4D/c128 $X4D/turns $X4D/q256 $X4D/smoke $XD/pool
X4PY="python code/m16_x4.py"
X4_DUMP=$KVD/x4
X4B_SEL=$X4R/x4b_selection.json
X4C_SEL=$X4R/x4c_selection.json
X4D_SEL=$X4R/x4d_selection.json
# 題目（和 m16_x4.py 的常數同一組；self-test 核對不重疊）
X4A_SEED0=64000; X4A_N=24          # X4a：T2–T5 各 24 題
X4B_SEED0=65000; X4B_N=48          # X4b：T2–T5 各 48 題
X4C_SEED0=40000; X4C_N=12          # X4c：44c 的 128K 題組（252 block）
X4D_SEED0=70000; X4D_N=32          # X4d：C1a 的兩輪題組（T2–T4）
Q256_N=6                           # Qwen 256K（504 block）：T2–T5 各 6 題
# 每個模型切幾份（n 在兩張卡之間共用）
X4A_SHARDS_LLAMA=${X4A_SHARDS_LLAMA:-8};  X4A_SHARDS_QWEN=${X4A_SHARDS_QWEN:-4}
X4B_SHARDS_LLAMA=${X4B_SHARDS_LLAMA:-12}; X4B_SHARDS_QWEN=${X4B_SHARDS_QWEN:-8}
X4C_SHARDS_LLAMA=${X4C_SHARDS_LLAMA:-4};  X4C_SHARDS_QWEN=${X4C_SHARDS_QWEN:-4}
X4D_SHARDS_LLAMA=${X4D_SHARDS_LLAMA:-6};  X4D_SHARDS_QWEN=${X4D_SHARDS_QWEN:-4}
Q256_SHARDS=${Q256_SHARDS:-2}
# 每個行程登記的 GiB（工作池每卡 ≤ XP_CAP_GB＝165）。依據：32K 實測 peak Llama 39.8（X2 gen）／41.8（X1 screen）、Qwen 25.1／26.6；
# 128K 實測 Llama 78.6、Qwen 42.1（02_smoke_x 的 gen128，run 20261020-141153／141848）；＋共用 SVD 的 U·S（Llama 32K 約 6 GB、128K ×4〔算術〕）。
# 256K 沒量過〔估〕＝Qwen 128K 的 2 倍＋SVD。
X4_GB_LLAMA=${X4_GB_LLAMA:-50};   X4_GB_QWEN=${X4_GB_QWEN:-32}
X4C_GB_LLAMA=${X4C_GB_LLAMA:-100}; X4C_GB_QWEN=${X4C_GB_QWEN:-50}
Q256_GB=${Q256_GB:-95}
X4_BUDGET=${X4_BUDGET:-100}

x4_sha() { cat code/m16_x4.py code/m16_xstack.py code/m16_xstack_combos_x4.json code/m16_queue/jobs/x_pool.sh code/m16_queue/jobs/x4_common.sh | sha256sum | cut -c1-16; }
x4_mark_smoke() { echo "$(x4_sha) $(date -u +%FT%TZ) $JOB" > $X4D/SMOKE_OK_$1; }
x4_check_smoke() { # x4_check_smoke <階段>：那個階段第 1 段的冒煙沒過、或之後程式改過 → NEEDS_HUMAN、回 1
  local f=$X4D/SMOKE_OK_$1
  [ -e $f ] || { nh "$JOB：$f 不在（$1 第 1 段的冒煙沒跑或沒過）→ 不跑"; return 1; }
  [ "$(cut -d' ' -f1 $f)" = "$(x4_sha)" ] || { nh "$JOB：程式在 $1 的冒煙之後改過（$f）→ 重排那一段的第 1 段再冒煙"; return 1; }
}
x4_gb() { case "$1:$2" in llama:32K) echo $X4_GB_LLAMA ;; qwen:32K) echo $X4_GB_QWEN ;; llama:128K) echo $X4C_GB_LLAMA ;;
          qwen:128K) echo $X4C_GB_QWEN ;; qwen:256K) echo $Q256_GB ;; *) echo 999 ;; esac; }
x4_nsel() { python3 -I -c "import json,sys; print(len(json.load(open(sys.argv[1]))['models'][sys.argv[2]]['configs']))" "$1" "$2"; }

x4_rows_ok() {     # x4_rows_ok <逐列 CSV> <要幾列>：1 題、列數＝要的、分數都不是空的
  python3 -I - "$1" "$2" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
want = int(sys.argv[2])
bad = [r["spec"] for r in rows if (r.get("gen_score") or r.get("gold_lp") or "") == ""]
ok = len(rows) == want and len({r["item_id"] for r in rows}) == 1 and not bad
print(f"冒煙列：{len(rows)} 列、要 {want}；空分數 {bad}")
sys.exit(0 if ok else 1)
PY
}

x4_selftest() {    # x4_selftest <模型>：m16_x4 的 gpu-selftest（共用 SVD vs 精確、xl+drop、兩輪＝m10_quality、整個 32K 區的 SVD 時間、dump 位元組）
  m7run m16-x4-gpuselftest-$1 $X4PY gpu-selftest --model $1 --out $X4D/smoke/gpu_selftest_$1.json \
    || { nh "$JOB：$1 gpu-selftest 沒過（$X4D/smoke/gpu_selftest_$1.json；$(lastrun m16-x4-gpuselftest-$1)）"; return 1; }
}

x4_smoke_tf() {    # x4_smoke_tf <模型>：X4a 1 題（冒煙的 5 個配置）
  local out=$X4D/smoke/tf_smoke_$1_s0.csv
  rm -f $out
  m7run m16-x4a-smoke-$1 $X4PY tf --model $1 --smoke --out $out || { nh "$JOB：$1 X4a 冒煙失敗（$(lastrun m16-x4a-smoke-$1)）"; return 1; }
  x4_rows_ok $out 6 || { nh "$JOB：$1 X4a 冒煙的列不對（$out）"; return 1; }
  grep -h "peak" $TIARA_RUNS/$(lastrun m16-x4a-smoke-$1)/stdout.log | tail -1 || true
}

x4_smoke_gen() {   # x4_smoke_gen <模型> <計畫檔> <n_blocks> <輪數> <任務> <seed0> <名字>：生成 1 題（計畫裡全部配置）
  local m=$1 sel=$2 nb=$3 turns=$4 task=$5 s0=$6 nm=$7 out k
  out=$X4D/smoke/${nm}_smoke_${m}_s0.csv
  rm -f $out
  k=$(x4_nsel $sel $m) || { nh "$JOB：讀不到 $sel（$m）"; return 1; }
  m7run m16-$nm-smoke-$m $X4PY gen --model $m --selection $sel --n-blocks $nb --turns $turns --tasks $task --n 1 --seed0 $s0 --smoke \
    --out $out || { nh "$JOB：$m $nm 冒煙失敗（$(lastrun m16-$nm-smoke-$m)）"; return 1; }
  x4_rows_ok $out $(( (k + 1) * turns )) || { nh "$JOB：$m $nm 冒煙的列不對（$out）"; return 1; }
  grep -h "peak" $TIARA_RUNS/$(lastrun m16-$nm-smoke-$m)/stdout.log | tail -1 || true
}

x4_units() {       # x4_units <池> <名字> <子目錄> <ctx> <shards_llama> <shards_qwen> <tasks（空白分隔）> <n> <seed0> <子命令的其他參數…>
  local P=$1 nm=$2 sub=$3 ctx=$4 nl=$5 nq=$6 tasks=$7 n=$8 s0=$9 u m i k out; shift 9
  for u in $(xp_order llama $nl qwen $nq); do
    m=${u%:*}; i=${u#*:}
    if [ $m = llama ]; then k=$nl; else k=$nq; fi
    [ $k = 0 ] && continue
    out=$X4D/$sub/${nm}_${m}_${ctx}_s$i.csv.gz
    xp_add $P $m $(x4_gb $m $ctx) m16-$nm-$m-s$i \
      "$(printf '%q ' $XPY pending --tasks $tasks --n $n --seed0 $s0 --shard $i/$k --out $out)" \
      $X4PY "$@" --model $m --tasks $tasks --n $n --seed0 $s0 --shard $i/$k --budget-min @B@ --out $out
  done
}

x4_run_pool() {    # x4_run_pool <分鐘預算> <填單位的指令…>
  local B=$1 P=$XD/pool/$JOB; shift
  xp_init $P
  "$@" $P || { nh "$JOB：建工作池失敗"; return 1; }
  xp_run $P $B 0 1 || { nh "$JOB：工作池有單位失敗（$P/log）"; return 1; }
}
x4a_fill() { x4_units $1 x4a tf 32K $X4A_SHARDS_LLAMA $X4A_SHARDS_QWEN "T2 T3 T4 T5" $X4A_N $X4A_SEED0 tf; }
x4b_fill() { x4_units $1 x4b gen 32K $X4B_SHARDS_LLAMA $X4B_SHARDS_QWEN "T2 T3 T4 T5" $X4B_N $X4B_SEED0 gen --selection $X4B_SEL --n-blocks 64; }
x4c_fill() { x4_units $1 x4c c128 128K $X4C_SHARDS_LLAMA $X4C_SHARDS_QWEN "T2 T3 T4 T5" $X4C_N $X4C_SEED0 gen --selection $X4C_SEL --n-blocks 252; }
x4d_fill() { x4_units $1 x4d turns 32K $X4D_SHARDS_LLAMA $X4D_SHARDS_QWEN "T2 T3 T4" $X4D_N $X4D_SEED0 gen --selection $X4D_SEL --n-blocks 64 --turns 2; }
q256_fill() { x4_units $1 x4q256 q256 256K 0 $Q256_SHARDS "T2 T3 T4 T5" $Q256_N $X4C_SEED0 gen --selection $X4C_SEL --n-blocks 504; }

x4_select_cd() {   # X4b → X4c／X4d 的計畫檔（CPU；已有而內容相同 → 不重寫；不同 → 停）
  m7run m16-x4-selectcd $X4PY select-cd --raw "$X4D/gen/x4b_*_32K_s*.csv.gz" --out-c $X4C_SEL --out-d $X4D_SEL \
    || { nh "$JOB：select-cd 失敗（$(lastrun m16-x4-selectcd)；X4b 題數不夠？）"; return 1; }
}

x4_report() {      # 全部 X4 的報告（CPU；有什麼算什麼，沒有的寫 NOT_MEASURED）
  m7run m16-x4-report $X4PY report --raw-tf "$X4D/tf/x4a_*_s*.csv.gz" \
    --raw-gen "$X4D/gen/x4b_*_s*.csv.gz" "$X4D/c128/x4c_*_s*.csv.gz" "$X4D/q256/x4q256_*_s*.csv.gz" \
    --raw-turns "$X4D/turns/x4d_*_s*.csv.gz" --out-dir $X4R || { nh "$JOB：x4 report 失敗（$(lastrun m16-x4-report)）"; return 1; }
}
