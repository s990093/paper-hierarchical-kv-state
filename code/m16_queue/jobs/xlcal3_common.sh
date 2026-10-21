# xKV 校準基底 vs 每題 SVD 加題數（38k*；code/m16_xlcal3.py＋m16_xstack.py gen）的 job 共用設定與函式。
# 由 38k1／38k2／38kz 在 lib.sh、x_common.sh、x_pool.sh 之後 source。**不要把這個檔放進佇列目錄**。說明：docs/stage3_20261019/x_stack_howto.md §16。
# 輸出：摘要 results/m16_stage3_mi300x/x_stack/v3/（不覆蓋 v2）；逐列 $XD/xlcal3/。
# 注意（10/21 38b1 的教訓）：`local a=$1 b=…${a}` 在 set -u 下會 unbound（local 的參數先全部展開）→ 分開寫。
XR3=$RES/x_stack/v3
mkdir -p $XR3 $XD/xlcal3 $XD/smoke/v3 $XD/pool
XL3PY="python code/m16_xlcal3.py"
XL3_PLAN=$XR3/xlcal3_selection.json
XL3_SEED0=66000; XL3_N=48                          # 新題：T2–T5 各 48 題＝192 題（和 m16_xlcal3.py 的常數同一組；self-test 核對不重疊）
XL3_SHARDS_LLAMA=${XL3_SHARDS_LLAMA:-12}           # Llama 每個 shard 16 題（每題較貴，切細讓工作池補洞）
XL3_SHARDS_QWEN=${XL3_SHARDS_QWEN:-8}              # 每個 shard 24 題
XL3_GB_LLAMA=${XL3_GB_LLAMA:-46}                   # 和 38c 的生成單位同樣的登記（38c 照這個跑完、沒有 OOM）
XL3_GB_QWEN=${XL3_GB_QWEN:-30}
XL3_BUDGET=${XL3_BUDGET:-100}

xl3_sha() { cat code/m16_xlcal3.py code/m16_xstack.py code/m16_queue/jobs/x_pool.sh code/m16_queue/jobs/xlcal3_common.sh | sha256sum | cut -c1-16; }
xl3_mark_smoke() { echo "$(xl3_sha) $(date -u +%FT%TZ) $JOB" > $XD/xlcal3/SMOKE_OK; }
xl3_check_smoke() {
  local f
  f=$XD/xlcal3/SMOKE_OK
  [ -e $f ] || { nh "$JOB：$f 不在（38k1 的冒煙沒跑或沒過）→ 不跑"; return 1; }
  [ "$(cut -d' ' -f1 $f)" = "$(xl3_sha)" ] || { nh "$JOB：程式在冒煙之後改過（$f）→ 重排 38k1 再冒煙"; return 1; }
}
xl3_gb() { if [ "$1" = llama ]; then echo $XL3_GB_LLAMA; else echo $XL3_GB_QWEN; fi; }

xl3_plan() { m7run m16-xlcal3-plan $XL3PY plan --out $XL3_PLAN || { nh "$JOB：plan 失敗（$(lastrun m16-xlcal3-plan)）"; return 1; }; }
xl3_check_calib() {  # 沿用 38c 的校準檔：sha256 要＝v2/calib_xl_<m>.json 記的（不重算；不符 → NEEDS_HUMAN、停）
  m7run m16-xlcal3-calibsha $XL3PY check-calib --out $XR3/calib_xl_sha.json \
    || { nh "$JOB：校準檔的 sha256 和 38c 記的不同或檔案不在（$XR3/calib_xl_sha.json；$(lastrun m16-xlcal3-calibsha)）"; return 1; }
}

xl3_smoke() {        # xl3_smoke <模型>：新題的第 1 題（E0＋4 個配置＝5 列）→ $XD/smoke/v3
  local m out
  m=$1
  out=$XD/smoke/v3/xlcal3_smoke_${m}_s0.csv
  rm -f $out
  m7run m16-xlcal3-smoke-$m $XPY gen --model $m --ctx 32K --tier core --tasks T2 --n 1 --seed0 $XL3_SEED0 --selection $XL3_PLAN \
    --calib-xl $XD/calib_xl_$m.pt --xl-cache --out $out || { nh "$JOB：$m 冒煙失敗（$(lastrun m16-xlcal3-smoke-$m)）"; return 1; }
  python3 -I - "$out" <<'PY' || { nh "$JOB：$m 冒煙的列不對（$out）"; return 1; }
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
want = {"E0", "xl:k22:v34:cal+q:int4_kivi", "xl:k22:v34+q:int4_kivi", "xl:k28:v42:cal+q:int4_kivi", "xl:k28:v42+q:int4_kivi"}
got = [r["spec"] for r in rows]
bad = [r["spec"] for r in rows if r.get("gen_score", "") == ""]
cal = {r["calib_run_id"] for r in rows if r["spec"] != "E0"}
print(f"冒煙列：{len(got)} 列；少 {sorted(want - set(got))}、多 {sorted(set(got) - want)}、空分數 {bad}；calib_run_id {cal}")
sys.exit(0 if set(got) == want and len(got) == 5 and not bad and all("xl:" in c for c in cal) else 1)
PY
  grep -h "peak" $TIARA_RUNS/$(lastrun m16-xlcal3-smoke-$m)/stdout.log | tail -1 || true
}

xl3_pool() {         # xl3_pool <分鐘預算>：兩卡共用工作池（Llama 12 份、Qwen 8 份交錯）
  local P u m i n out
  P=$XD/pool/$JOB
  xp_init $P
  for u in $(xp_order llama $XL3_SHARDS_LLAMA qwen $XL3_SHARDS_QWEN); do
    m=${u%:*}; i=${u#*:}
    if [ $m = llama ]; then n=$XL3_SHARDS_LLAMA; else n=$XL3_SHARDS_QWEN; fi
    out=$XD/xlcal3/gen_${m}_32K_s$i.csv.gz
    xp_add $P $m $(xl3_gb $m) m16-xlcal3-$m-s$i "$(printf '%q ' $XPY pending --n $XL3_N --seed0 $XL3_SEED0 --shard $i/$n --out $out)" \
      $XPY gen --model $m --ctx 32K --tier core --n $XL3_N --seed0 $XL3_SEED0 --selection $XL3_PLAN --calib-xl $XD/calib_xl_$m.pt --xl-cache \
      --shard $i/$n --budget-min @B@ --out $out
  done
  xp_run $P $1 0 1 || { nh "$JOB：工作池有單位失敗（$P/log）"; return 1; }
}
