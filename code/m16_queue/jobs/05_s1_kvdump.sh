# GPU-NEED: one
# CARDS: 0,1
# NEED-GB: 45
# MAX-H: 1
# S1：存真的 KV（BF16＋fp8_token／fp8_head／int8_token／int4_kivi 的實際位元組）。32K 全層；128K 只存 5 層（省空間）。
# T4＝HotpotQA 段落（自然文字），T2＝RULER noise haystack（重複句子，分開看）。
# 10/20：128K 的層改成含相鄰層對（L5b／K1 說明檔 §注意）。兩張卡並行（Llama 在卡 0、Qwen 在卡 1）；這份 dump 也給 K1（KV 結構）、疊加組合掃描、B（無損壓縮比）用。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
mkdir -p $KVD
dump() {   # dump <模型> <128K 用的層>
  local m=$1 L=$2 t
  for t in T4 T2; do
    m7run m16-s1dump-$m-$t-64 python code/m16_s1_kvdump.py --model $m --task $t --n-blocks 64 --out $KVD/${m}_${t}_64 \
      || { nh "05_s1_kvdump: $m $t 64 失敗"; return 1; }
  done
  m7run m16-s1dump-$m-T4-252 python code/m16_s1_kvdump.py --model $m --task T4 --n-blocks 252 --layers $L \
    --out $KVD/${m}_T4_252 || { nh "05_s1_kvdump: $m T4 252 失敗"; return 1; }
}
oncard 0 dump llama 0,8,9,16,17,24,25,31 & p0=$!     # 128K 存成對的相鄰層（8/9、16/17、24/25），K1 的跨層相似在 128K 才量得到
oncard 1 dump qwen 0,7,8,14,15,21,22,27 & p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
du -sh $KVD/*
[ $r0 = 0 ] && [ $r1 = 0 ]
