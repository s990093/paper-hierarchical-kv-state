# 10/21 延伸（38b* X2b 加題數、38c* xKV 事先校準的基底）的 job 共用設定與函式。
# 由 38b*／38bz／38c*／38cz 在 lib.sh、x_common.sh、x_pool.sh 之後 source。**不要把這個檔放進佇列目錄**。
# 說明：docs/stage3_20261019/x_stack_howto.md §14。輸出：摘要 $XR/v2（不覆蓋第一版 $XR 的檔）；逐列 $XD/x2b、$XD/xlcal；工作池 $XD/pool。
XR2=$XR/v2
mkdir -p $XR2 $XD/x2b $XD/xlcal $XD/pool $XD/smoke/v2
X2B_PLAN=$XR2/x2b_selection.json                  # plan 子命令寫（roles／budgets 照第一版 x2_selection.json）；題目 seed0／n 也從這裡讀
XLCAL_PLAN=$XR2/xlcal_selection.json
X2B_SHARDS_LLAMA=${X2B_SHARDS_LLAMA:-12}          # 192 題 → 每個 shard 16 題（Llama 每題較貴，切細一點讓工作池補洞）
X2B_SHARDS_QWEN=${X2B_SHARDS_QWEN:-8}             # 每個 shard 24 題
XLCAL_TF_SHARDS=${XLCAL_TF_SHARDS:-2}             # TF 48 題 → 每個 shard 24 題
XLCAL_GEN_SHARDS_LLAMA=${XLCAL_GEN_SHARDS_LLAMA:-6}
XLCAL_GEN_SHARDS_QWEN=${XLCAL_GEN_SHARDS_QWEN:-4}
# 每個行程登記的 GiB（工作池每卡上限 XP_CAP_GB＝165）：實測 peak Llama 39.8（X2 gen，run 20261020-214139-m16-x2-llama-32K-core-s0）、
# 41.8（X1 screen，20261020-155519-m16-x1-llama-s0）；Qwen 25.1、26.6（同批的 qwen）＋ --xl-cache 的快取（Llama ≤1.9 GB〔算術〕）與校準基底（≤0.6 GB）。
XP_GB_LLAMA=${XP_GB_LLAMA:-46}
XP_GB_QWEN=${XP_GB_QWEN:-30}
X_EXT_BUDGET=${X_EXT_BUDGET:-100}                 # 每段 job 的工作池預算（分鐘；超過就不再開新的題）

xgb() { if [ "$1" = llama ]; then echo $XP_GB_LLAMA; else echo $XP_GB_QWEN; fi; }
plan_get() { python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(d[sys.argv[2]])" "$1" "$2"; }
code_sha() { cat code/m16_xstack.py code/m16_xstack_combos_v2.json code/m16_queue/jobs/x_pool.sh code/m16_queue/jobs/x_ext.sh | sha256sum | cut -c1-16; }

x_mark_smoke() {   # x_mark_smoke <x2b|xlcal>：冒煙過了 → 記下這版程式的 sha（後面幾段 job 只在同一版程式上跑）
  echo "$(code_sha) $(date -u +%FT%TZ) $JOB" > $XD/$1/SMOKE_OK
}
x_check_smoke() {  # x_check_smoke <x2b|xlcal>：第 1 段的冒煙沒過、或之後程式改過 → NEEDS_HUMAN、回 1
  local f=$XD/$1/SMOKE_OK
  [ -e $f ] || { nh "$JOB：$f 不在（第 1 段的冒煙沒跑或沒過）→ 不跑"; return 1; }
  [ "$(cut -d' ' -f1 $f)" = "$(code_sha)" ] || { nh "$JOB：程式在冒煙之後改過（$f 的 sha ≠ 現在的 $(code_sha)）→ 重排第 1 段再冒煙"; return 1; }
}

smoke_rows_ok() {  # smoke_rows_ok <逐列 CSV> <計畫檔> <模型>：1 題、E0＋計畫的每個配置各一列、分數不是空的
  python3 -I - "$1" "$2" "$3" <<'PY'
import csv, json, sys
rows = list(csv.DictReader(open(sys.argv[1])))
want = {"E0"} | {e["spec"] for e in json.load(open(sys.argv[2]))["models"][sys.argv[3]]["configs"]}
got = [r["spec"] for r in rows]
bad = [r["spec"] for r in rows if r.get("gen_score", r.get("gold_lp", "")) in ("", None)]
ok = set(got) == want and len(got) == len(want) and len({r["item_id"] for r in rows}) == 1 and not bad
print(f"冒煙列：{len(got)} 列、要 {len(want)}；少 {sorted(want - set(got))}、多 {sorted(set(got) - want)}、空分數 {bad}")
sys.exit(0 if ok else 1)
PY
}

# ------------------------------------------------------------------ X2b（38b*）
x2b_plan() { m7run m16-x2b-plan $XPY plan --which x2b --selection $XR/x2_selection.json --out $X2B_PLAN; }

x2b_smoke() {      # x2b_smoke <模型>：gpu-selftest 的新檢查（x2b：出題、X2b 每個配置、--xl-cache 逐位元）＋X2b 的第一題（全部配置）→ $XD/smoke/v2
  local m=$1 s0 out=$XD/smoke/v2/x2b_smoke_${m}_s0.csv
  s0=$(plan_get $X2B_PLAN seed0) || { nh "$JOB：讀不到 $X2B_PLAN"; return 1; }
  m7run m16-x2b-gpuselftest-$m $XPY gpu-selftest --model $m --calib $XD/calib_$m.pt --new-checks x2b \
    --out $XD/smoke/v2/gpu_selftest_x2b_$m.json \
    || { nh "$JOB：$m gpu-selftest（x2b）沒過（$XD/smoke/v2/gpu_selftest_x2b_$m.json；$(lastrun m16-x2b-gpuselftest-$m)）"; return 1; }
  rm -f $out
  m7run m16-x2b-smoke-$m $XPY gen --model $m --ctx 32K --tier core --tasks T2 --n 1 --seed0 $s0 --selection $X2B_PLAN --xl-cache \
    --out $out || { nh "$JOB：$m X2b 冒煙 1 題失敗（$(lastrun m16-x2b-smoke-$m)）"; return 1; }
  smoke_rows_ok $out $X2B_PLAN $m || { nh "$JOB：$m X2b 冒煙的列不對（$out）"; return 1; }
  grep -h "peak" $TIARA_RUNS/$(lastrun m16-x2b-smoke-$m)/stdout.log | tail -1 || true      # 只印 peak GiB（決定 XP_GB_*），找不到不算失敗
}

x2b_pool() {       # x2b_pool <分鐘預算>：X2b 的工作池（兩卡共用）
  local P=$XD/pool/$JOB u m i n s0 nn out
  s0=$(plan_get $X2B_PLAN seed0) && nn=$(plan_get $X2B_PLAN n) || { nh "$JOB：讀不到 $X2B_PLAN"; return 1; }
  xp_init $P
  for u in $(xp_order llama $X2B_SHARDS_LLAMA qwen $X2B_SHARDS_QWEN); do
    m=${u%:*}; i=${u#*:}
    if [ $m = llama ]; then n=$X2B_SHARDS_LLAMA; else n=$X2B_SHARDS_QWEN; fi
    out=$XD/x2b/x2b_${m}_32K_s$i.csv.gz
    xp_add $P $m $(xgb $m) m16-x2b-$m-s$i "$(printf '%q ' $XPY pending --n $nn --seed0 $s0 --shard $i/$n --out $out)" \
      $XPY gen --model $m --ctx 32K --tier core --n $nn --seed0 $s0 --selection $X2B_PLAN --xl-cache \
      --shard $i/$n --budget-min @B@ --out $out
  done
  xp_run $P $1 0 1 || { nh "$JOB：X2b 工作池有單位失敗（$P/log；各單位的 run 見 log）"; return 1; }
}

# ------------------------------------------------------------------ xKV 事先校準的基底（38c*）
xlcal_plan() { m7run m16-xlcal-plan $XPY plan --which xlcal --out $XLCAL_PLAN; }

xlcal_calib() {    # xlcal_calib <模型>：校準基底（已有、設定相同、sha256 對得上就跳過；m16_xstack 自己核對）
  m7run m16-xlcal-calib-$1 $XPY calib-xl --model $1 --out $XD/calib_xl_$1.pt --summary-dir $XR2 \
    || { nh "$JOB：$1 calib-xl 失敗（$(lastrun m16-xlcal-calib-$1)）"; return 1; }
}

xlcal_smoke() {    # xlcal_smoke <模型>：gpu-selftest 的新檢查（xlcal）＋TF 1 題（screen --smoke）＋生成 1 題（全部 6 個配置）→ $XD/smoke/v2
  local m=$1 s0 tf=$XD/smoke/v2/xlcal_tf_smoke_${m}_s0.csv gen=$XD/smoke/v2/xlcal_gen_smoke_${m}_s0.csv
  s0=$(plan_get $XLCAL_PLAN seed0) || { nh "$JOB：讀不到 $XLCAL_PLAN"; return 1; }
  m7run m16-xlcal-gpuselftest-$m $XPY gpu-selftest --model $m --calib $XD/calib_$m.pt --new-checks xlcal --calib-xl $XD/calib_xl_$m.pt \
    --out $XD/smoke/v2/gpu_selftest_xlcal_$m.json \
    || { nh "$JOB：$m gpu-selftest（xlcal）沒過（$XD/smoke/v2/gpu_selftest_xlcal_$m.json；$(lastrun m16-xlcal-gpuselftest-$m)）"; return 1; }
  rm -f $tf $gen
  m7run m16-xlcal-smoke-tf-$m $XPY screen --model $m --smoke --no-s2 --combos code/m16_xstack_combos_v2.json \
    --calib-xl $XD/calib_xl_$m.pt --out $tf || { nh "$JOB：$m xKV 校準 TF 冒煙失敗（$(lastrun m16-xlcal-smoke-tf-$m)）"; return 1; }
  m7run m16-xlcal-smoke-gen-$m $XPY gen --model $m --ctx 32K --tier core --tasks T2 --n 1 --seed0 $s0 --selection $XLCAL_PLAN \
    --calib-xl $XD/calib_xl_$m.pt --out $gen || { nh "$JOB：$m xKV 校準生成冒煙失敗（$(lastrun m16-xlcal-smoke-gen-$m)）"; return 1; }
  smoke_rows_ok $gen $XLCAL_PLAN $m || { nh "$JOB：$m xKV 校準生成冒煙的列不對（$gen）"; return 1; }
  grep -h "peak" $TIARA_RUNS/$(lastrun m16-xlcal-smoke-gen-$m)/stdout.log | tail -1 || true      # 只印 peak GiB（決定 XP_GB_*），找不到不算失敗
}

xlcal_pool() {     # xlcal_pool <分鐘預算>：TF 單位（兩個模型）排前面，再生成單位；兩卡共用
  local P=$XD/pool/$JOB u m i n s0 nn t0 tn out
  s0=$(plan_get $XLCAL_PLAN seed0) && nn=$(plan_get $XLCAL_PLAN n) && t0=$(plan_get $XLCAL_PLAN tf_seed0) && tn=$(plan_get $XLCAL_PLAN tf_n) \
    || { nh "$JOB：讀不到 $XLCAL_PLAN"; return 1; }
  xp_init $P
  for u in $(xp_order llama $XLCAL_TF_SHARDS qwen $XLCAL_TF_SHARDS); do
    m=${u%:*}; i=${u#*:}
    out=$XD/xlcal/tf_${m}_s$i.csv.gz
    xp_add $P $m $(xgb $m) m16-xlcal-tf-$m-s$i "$(printf '%q ' $XPY pending --n $tn --seed0 $t0 --shard $i/$XLCAL_TF_SHARDS --out $out)" \
      $XPY screen --model $m --no-s2 --combos code/m16_xstack_combos_v2.json --n $tn --seed0 $t0 --calib-xl $XD/calib_xl_$m.pt \
      --shard $i/$XLCAL_TF_SHARDS --budget-min @B@ --out $out
  done
  for u in $(xp_order llama $XLCAL_GEN_SHARDS_LLAMA qwen $XLCAL_GEN_SHARDS_QWEN); do
    m=${u%:*}; i=${u#*:}
    if [ $m = llama ]; then n=$XLCAL_GEN_SHARDS_LLAMA; else n=$XLCAL_GEN_SHARDS_QWEN; fi
    out=$XD/xlcal/gen_${m}_32K_s$i.csv.gz
    xp_add $P $m $(xgb $m) m16-xlcal-gen-$m-s$i "$(printf '%q ' $XPY pending --n $nn --seed0 $s0 --shard $i/$n --out $out)" \
      $XPY gen --model $m --ctx 32K --tier core --n $nn --seed0 $s0 --selection $XLCAL_PLAN --calib-xl $XD/calib_xl_$m.pt \
      --shard $i/$n --budget-min @B@ --out $out
  done
  xp_run $P $1 0 1 || { nh "$JOB：xKV 校準工作池有單位失敗（$P/log）"; return 1; }
}
