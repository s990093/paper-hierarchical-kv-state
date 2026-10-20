# GPU-NEED: one
# CARDS: 0,1
# NEED-GB: 120
# MAX-H: 1
# 疊加實驗（code/m16_xstack.py）的冒煙：Llama 在卡 0、Qwen 在卡 1，兩個背景行程、等兩個都結束；任一失敗 → NEEDS_HUMAN、exit 1。
# 每個模型：(1) 1 題的校準（只驗管線＋RoPE 的資料自身核對；正式的校準在 36a）→ (2) gpu-selftest（torch vs numpy、量化和 m10_codec_x 逐位元、
# vllm_fp8 對 fnuz 格點、RoPE 核對、TF 路徑和 load_fmt／place 逐值相同）→ (3) screen --smoke（1 題＝DS1 第一題、組合清單的 smoke 17 個＋S2 子集、
# 和 S2／DS1 的逐值重現檢查，有差 exit 1）→ (4) gen 32K 1 題 4 個配置 → (5) gen 128K 1 題 2 個配置（44c 第一題；E0 對 44c 的重現）。
# 最後（CPU）report／gen-report 吃冒煙的輸出（只驗程式跑得完，題數不夠，判定是 INCOMPLETE）。輸出全部在 $XD/smoke，不進 results。
# 〔估〕15–30 分鐘（每張卡載 5 次模型；128K 的 prefill＋生成約數分鐘）。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_common.sh
SM=$XD/smoke; mkdir -p $SM/rep
python3 - "$SM/x2_sel_smoke.json" <<'EOF'
import json, sys
cf = [dict(spec="none", roles=["anchor_mask0"], tiers=["core", "128k"], budgets=[]),
      dict(spec="q:int4_kivi", roles=["anchor_int4"], tiers=["core", "128k"], budgets=[]),
      dict(spec="drop:KEYDIFF:0.5+lr:k24+q:int4_kivi", roles=["primary"], tiers=["core"], budgets=[0.125]),
      dict(spec="xl:k11:v17", roles=["best_single"], tiers=["core"], budgets=[0.125])]
json.dump(dict(go=True, reason="smoke", models={m: dict(configs=cf) for m in ("llama", "qwen")}), open(sys.argv[1], "w"), indent=1)
EOF
smoke_model() {
  local m=$1
  m7run m16-x-smoke-calib-$m $XPY calib --model $m --tasks T4 --n 1 --force --out $SM/calib_$m.pt --summary-dir $SM \
    || { nh "02_smoke_x：$m calib 失敗（$(lastrun m16-x-smoke-calib-$m)）"; return 1; }
  m7run m16-x-gpuselftest-$m $XPY gpu-selftest --model $m --calib $SM/calib_$m.pt --out $SM/gpu_selftest_$m.json \
    || { nh "02_smoke_x：$m gpu-selftest 沒過（$SM/gpu_selftest_$m.json；$(lastrun m16-x-gpuselftest-$m)）"; return 1; }
  m7run m16-x-smoke-screen-$m $XPY screen --model $m --smoke --calib $SM/calib_$m.pt --out $SM/x1_smoke_${m}_s0.csv \
    || { nh "02_smoke_x：$m screen --smoke 失敗或重現檢查有差（$(lastrun m16-x-smoke-screen-$m)）"; return 1; }
  m7run m16-x-smoke-gen32-$m $XPY gen --model $m --ctx 32K --tier core --tasks T2 --n 1 --selection $SM/x2_sel_smoke.json \
    --calib $SM/calib_$m.pt --out $SM/x2_smoke_${m}_32K_s0.csv \
    || { nh "02_smoke_x：$m gen 32K 失敗（$(lastrun m16-x-smoke-gen32-$m)）"; return 1; }
  m7run m16-x-smoke-gen128-$m $XPY gen --model $m --ctx 128K --tier 128k --tasks T2 --n 1 --selection $SM/x2_sel_smoke.json \
    --calib $SM/calib_$m.pt --out $SM/x2_smoke_${m}_128K_s0.csv \
    || { nh "02_smoke_x：$m gen 128K 失敗（$(lastrun m16-x-smoke-gen128-$m)）"; return 1; }
}
oncard 0 smoke_model llama & p0=$!
oncard 1 smoke_model qwen & p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
[ $r0 = 0 ] && [ $r1 = 0 ] || exit 1
export OMP_NUM_THREADS=4
m7run m16-x-smoke-report taskset -c 0-3 $XPY report --raw "$SM/x1_smoke_*_s0.csv" --out-dir $SM/rep \
  || { nh "02_smoke_x：report 跑不完（$(lastrun m16-x-smoke-report)）"; exit 1; }
m7run m16-x-smoke-genreport taskset -c 0-3 $XPY gen-report --raw "$SM/x2_smoke_*_s0.csv" --selection $SM/x2_sel_smoke.json \
  --out-dir $SM/rep || { nh "02_smoke_x：gen-report 跑不完（$(lastrun m16-x-smoke-genreport)）"; exit 1; }
grep -h "peak" $TIARA_RUNS/*m16-x-smoke-screen-*/stdout.log | tail -2
echo "02_smoke_x 全過 $(date -u +%FT%TZ)"
