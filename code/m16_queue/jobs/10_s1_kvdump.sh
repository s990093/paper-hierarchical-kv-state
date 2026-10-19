# GPU-NEED: one
# GPU: 1
# NEED-GB: 45
# MAX-H: 1
# S1：存真的 KV（BF16＋fp8_token／fp8_head／int8_token／int4_kivi 的實際位元組）。32K 全層；128K 只存 5 層（省空間）。
# T4＝HotpotQA 段落（自然文字），T2＝RULER noise haystack（重複句子，分開看）。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
mkdir -p $KVD
for m in llama qwen; do
  for t in T4 T2; do
    m7run m16-s1dump-$m-$t-64 python code/m16_s1_kvdump.py --model $m --task $t --n-blocks 64 --out $KVD/${m}_${t}_64 \
      || { nh "10_s1_kvdump: $m $t 64 失敗"; exit 1; }
  done
done
m7run m16-s1dump-llama-T4-252 python code/m16_s1_kvdump.py --model llama --task T4 --n-blocks 252 --layers 0,8,16,24,31 \
  --out $KVD/llama_T4_252 || { nh "10_s1_kvdump: llama T4 252 失敗"; exit 1; }
m7run m16-s1dump-qwen-T4-252 python code/m16_s1_kvdump.py --model qwen --task T4 --n-blocks 252 --layers 0,7,14,21,27 \
  --out $KVD/qwen_T4_252 || { nh "10_s1_kvdump: qwen T4 252 失敗"; exit 1; }
du -sh $KVD/*
