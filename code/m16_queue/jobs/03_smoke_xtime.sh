# GPU-NEED: both
# GPU: 1
# NEED-GB: 80
# MAX-H: 0.5
# X3（壓縮組合的還原時間 × 連線速度 × GPU 忙；docs/stage3_20261019/x3_time_howto.md）的冒煙：m16_xtime.py --smoke。
#   Llama、Qwen 各一次最小設定：每個解碼段（量化／低秩／跨層／合併／丟 token／組合）載入區 4 個 block、量一次；duty 0 與 0.7；
#   背景拖慢只量 2 個 op、每個 0.5 s；recompute 探針 2 個位置；接著 map 吃冒煙資料、validate 跑一格第二階段還原器（4K）＋一格載入線。
#   輸出到 $SMOKE/x3（不進 results）。每個解碼段建好就逐位元（或 BF16 捨入內）比對，不過就失敗。
# 量時間的程式 → 兩卡獨佔（timing 角色、卡 1）。HSA_ENABLE_SDMA 照第二階段：不設（ROCr 預設 1＝SDMA copy engine）。
# 失敗 → NEEDS_HUMAN、exit 1（39_x3_* 都靠這支程式）。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
unset HSA_ENABLE_SDMA
export M10_BUSY_TEARDOWN_PAUSE=1        # 同第二階段的 lib.sh：m7_busy 收尾時 amd-smi 還列著它的 VRAM，收尾期間暫停 gpu_guard 取樣
mkdir -p $SMOKE/x3
m7run m16-smoke-xtime python code/m16_xtime.py --smoke --out $SMOKE/x3 \
  || { nh "03_smoke_xtime 失敗（$(lastrun m16-smoke-xtime)；stdout.log 最後一行列出哪幾項 FAIL）"; exit 1; }
echo "03_smoke_xtime 全過 $(date -u +%FT%TZ)"
