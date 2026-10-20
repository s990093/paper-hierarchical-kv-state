# GPU-NEED: both
# GPU: 1
# NEED-GB: 80
# MAX-H: 1
# AFTER: 03_smoke_xtime
# X3 micro（docs/stage3_20261019/x3_time_howto.md §2）：32K 的載入區（block 32–63、全層）上，各解碼段的 H2D 與解碼時間 × duty 0／0.4／0.7／0.9，
#   背景 matmul 被拖慢多少，重算慢化（recompute 探針）。Llama、Qwen 各一個 run（各自的 run_id）。量時間 → 兩卡獨佔（卡 1）。
# GPU 時間〔估〕每個模型 10–15 分鐘、合計 0.3–0.5 小時（依據見 howto §7：每個 op 5 次 × ≥0.1 s × 約 53 個 op × 4 個 duty、
#   背景拖慢 3 個 duty × 15 次 m7_busy 啟動、模型載入約 1 分鐘）；MAX-H 1 留緩衝。
# HSA_ENABLE_SDMA：第二階段所有量時間的 job 都沒有設這個變數（ROCr 預設 1＝H2D 走 SDMA copy engine）→ 這裡照做（unset）；
#   SDMA 0／1 的差別由 I4（20／21）量，不在這裡。
# 污染（gpu_guard）：被污染的那個模型自動重量一次；再污染 → NEEDS_HUMAN，該模型在 map 是 NOT_MEASURED（輸出不進 results，CLAUDE.md §3）。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
unset HSA_ENABLE_SDMA
export M10_BUSY_TEARDOWN_PAUSE=1
OUT=$RES/x3_time; mkdir -p $OUT
for m in llama qwen; do
  for try in 1 2; do
    m7run m16-x3-micro-$m python code/m16_xtime.py micro --model $m --out $OUT \
      || { nh "39_x3_micro：$m 失敗（$(lastrun m16-x3-micro-$m)）"; exit 1; }
    r=$(lastrun m16-x3-micro-$m)
    [ -e $TIARA_RUNS/$r/CONTAMINATED ] || break
    echo "39_x3_micro：$m 第 $try 次被污染（$r）"
    [ $try = 2 ] && nh "39_x3_micro：$m 重量後仍被污染（$r）→ map 會把它標 NOT_MEASURED；要人決定何時重量"
  done
done
exit 0
