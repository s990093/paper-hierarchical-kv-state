# GPU-NEED: none
# AFTER: 05_s1_kvdump
# K1：KV 本身的結構（低秩程度、相鄰層像不像、相鄰 token 像不像），只用 numpy（docs/stage3_20261019/k1_kvstruct_howto.md）。
# cpu 佇列（nice 19、0–3 號核心）。先用冒煙的 dump 做 RoPE 核對（要 PASS_PROBE），不過就停。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
OUT=$RES/k1_kvstruct; mkdir -p $OUT /mlsteam/data/tiara/stage3/k1
export OMP_NUM_THREADS=4
m7run m16-k1-ropecheck python -I code/m16_kvstruct.py ropecheck --dump $SMOKE/kv_llama_t4_4 \
  || { nh "06_k1_kvstruct：RoPE 核對沒過（$(lastrun m16-k1-ropecheck)）"; exit 1; }
m7run m16-k1-kvstruct python -I code/m16_kvstruct.py analyze \
  --dump $KVD/llama_T4_64 $KVD/llama_T2_64 $KVD/qwen_T4_64 $KVD/qwen_T2_64 $KVD/llama_T4_252 $KVD/qwen_T4_252 \
  --out $OUT --spectra /mlsteam/data/tiara/stage3/k1/spectra_$(date -u +%Y%m%dT%H%MZ).npz \
  || { nh "06_k1_kvstruct 失敗（$(lastrun m16-k1-kvstruct)）"; exit 1; }
