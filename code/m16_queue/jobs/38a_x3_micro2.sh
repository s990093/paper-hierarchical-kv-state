# GPU-NEED: both
# GPU: 1
# NEED-GB: 80
# MAX-H: 1.5
# X3 補缺口（10/21；docs/stage3_20261019/x3_time_howto.md §10）：X2 最好的 xKV 組合（K、V 秩不同、係數 BF16／INT4 KIVI／int8_token）
#   在第一版的解碼段目錄裡沒有 → 模式 X 的 NO-GO 不完整。這支只量新加的 7 個段＋xKV 的寫入（壓縮）成本，不重量舊段。
#   新段：xkv_k11v17、xkv_k22v34、xkv_k28v42（BF16 係數）、xkv_k11v17_int4、xkv_k22v34_int4、xkv_k28v42_int4（INT4 KIVI）、
#         xkv_k22v34_int8（int8_token）。每段量 h2d_region（壓縮後位元組，含一份基底）、h2d_req（只搬基底）、一次拆、逐 block 拆，
#         duty 0／0.4／0.7／0.9；建好先驗正確性（係數反量化和 m10_codec_x.dec_half 逐位元相同、重建和 FP32 參考差 <1e-2、逐 block＝一次整段）。
#   漂移錨點：bf16（一定會量）、int4_kivi、xkv_g4_f8 也量一次——map 不拿它們取代第一版，只算「這次 ÷ 第一版」看兩次 run 能不能拼。
#   寫入成本（--wcost x2）另外一個 run、另外一個目錄（v2/wcost/；只帶 bf16、只在 duty 0 量它當錨點）：每題、每 4 層一個窗口的
#   torch.svd_lowrank（niter 4、多取 16、seed 照 m16_xstack）＋係數量化，duty 0 與 0.9。分開跑是為了被污染時只重量那一段。
#   都不載模型（--no-recompute）：重算慢化用第一版 micro 的探針。
# 開頭先跑新段的冒煙（--smoke --segments …，4 個 block、duty 0／0.7、寫入成本也冒煙）；失敗 → NEEDS_HUMAN、exit 1。
# 量時間 → 兩卡獨佔（timing 角色、卡 1）。HSA_ENABLE_SDMA 照第二階段：不設（ROCr 預設 1）。
# GPU 時間〔估〕約 0.5–0.8 小時（依據 howto §10.5：冒煙約 8–12 分鐘；解碼段計時＋背景每個模型約 4–5 分鐘（第一版 Llama 9.8 分鐘＝
#   建段 85 s＋計時 4×約 35 s＋背景 39 次 m7_busy 約 6 分鐘＋探針；這次約 34 個 op、背景 2 個 op、不載模型）；寫入成本 Llama 約 10–15 分鐘、
#   Qwen 約 5–8 分鐘（X1 逐列的 t_pipe_s：63 個 block 的 xKV 每題 16–79 秒，不是乾淨的計時；32 個 block 照 token 數線性推一半）；
#   被污染重量一次最多再加一段 → 最壞約 1.2 小時）；MAX-H 1.5 留緩衝。
# 污染（gpu_guard）：被污染的那一段自動重量一次；再污染 → NEEDS_HUMAN，那一段在 map v2 是 NOT_MEASURED（輸出不進 results）。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
unset HSA_ENABLE_SDMA
export M10_BUSY_TEARDOWN_PAUSE=1        # 同 03／39：m7_busy 收尾時 amd-smi 還列著它的 VRAM，收尾期間暫停 gpu_guard 取樣
NEW="xkv_k11v17 xkv_k22v34 xkv_k28v42 xkv_k11v17_int4 xkv_k22v34_int4 xkv_k28v42_int4 xkv_k22v34_int8"
ANCHOR="int4_kivi xkv_g4_f8"
BG="xkv_k22v34:dq_oneshot xkv_k22v34_int4:dq_oneshot"
# 1) 冒煙（輸出不進 results）
mkdir -p $SMOKE/x3v2
m7run m16-smoke-xtime2 python code/m16_xtime.py --smoke --out $SMOKE/x3v2 --segments $NEW --wcost x2 \
  || { nh "38a_x3_micro2：冒煙失敗（$(lastrun m16-smoke-xtime2)；stdout.log 最後一行列出哪幾項 FAIL）"; exit 1; }
echo "38a_x3_micro2：冒煙全過 $(date -u +%FT%TZ)"
# 2) 正式（輸出到 v2/，不碰第一版的檔；程式也會擋「部分 segment 寫進第一版目錄」）
OUT=$RES/x3_time/v2; WOUT=$OUT/wcost; mkdir -p $OUT $WOUT
run2() {   # run2 <短名> <模型> <參數…>：被污染自動重量一次
  local short=$1 m=$2; shift 2
  for try in 1 2; do
    m7run $short-$m python code/m16_xtime.py micro --model $m "$@" \
      || { nh "38a_x3_micro2：$short $m 失敗（$(lastrun $short-$m)）"; exit 1; }
    r=$(lastrun $short-$m)
    [ -e $TIARA_RUNS/$r/CONTAMINATED ] || return 0
    echo "38a_x3_micro2：$short $m 第 $try 次被污染（$r）"
    [ $try = 2 ] && nh "38a_x3_micro2：$short $m 重量後仍被污染（$r）→ map v2 裡這一段是 NOT_MEASURED；要人決定何時重量"
  done
  return 0
}
for m in llama qwen; do
  run2 m16-x3-micro2 $m --out $OUT --segments $NEW $ANCHOR --no-recompute --bg $BG
done
for m in llama qwen; do
  run2 m16-x3-wcost $m --out $WOUT --segments bf16 --gpu-states idle --bg --no-recompute \
       --wcost x2 --wcost-states idle busy3 --wcost-reps 3
done
exit 0
