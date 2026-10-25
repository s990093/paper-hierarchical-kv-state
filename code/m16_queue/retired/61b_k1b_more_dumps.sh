# GPU-NEED: one
# CARDS: 0,1
# NEED-GB: 45
# MAX-H: 0.7
# K1b 加題（10/22 使用者：「再重新檢視所有實驗、需要補充的都一起上」）：K1／K1b 每個（模型, 任務）只有 1 題 KV。
# 每個模型多匯出 3 題 32K 全層 KV：T3（seed 70000）、T5（seed 70000）、T4（seed 70001，換一題），和 05_s1_kvdump 同一支程式。
# 之後 61bz（cpu 佇列）對這 6 份跑 m16_k1_tucker.py，報告把 10 份一起算；K1b 的事先登記判定仍以原本 4 份（T4／T2）為主，新的 6 份是穩健性。
# GPU 時間〔估〕：每份約 2–4 分鐘（prefill 32K＋寫檔）→ 兩卡並行約 15 分鐘。寫到本地盤 $KVD（可重生）。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
mkdir -p $KVD
dump3() {
  local m=$1
  m7run m16-k1bdump-$m-T3 python code/m16_s1_kvdump.py --model $m --task T3 --n-blocks 64 --seed 70000 --out $KVD/${m}_T3_64 || { nh "61b: $m T3 失敗"; return 1; }
  m7run m16-k1bdump-$m-T5 python code/m16_s1_kvdump.py --model $m --task T5 --n-blocks 64 --seed 70000 --out $KVD/${m}_T5_64 || { nh "61b: $m T5 失敗"; return 1; }
  m7run m16-k1bdump-$m-T4b python code/m16_s1_kvdump.py --model $m --task T4 --n-blocks 64 --seed 70001 --out $KVD/${m}_T4b_64 || { nh "61b: $m T4b 失敗"; return 1; }
}
oncard 0 dump3 llama & p0=$!
oncard 1 dump3 qwen & p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
du -sh $KVD/*_T3_64 $KVD/*_T5_64 $KVD/*_T4b_64
[ $r0 = 0 ] && [ $r1 = 0 ]
