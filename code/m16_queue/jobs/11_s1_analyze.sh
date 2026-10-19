# GPU-NEED: one
# GPU: 1
# NEED-GB: 0
# MAX-H: 2
# S1 分析（只用 CPU，但拿 GPU 鎖：確保這段時間沒有任何量時間的 job，CPU 速度才乾淨）。綁 0–15 號核心、thread 1／4／8／16。
# breakeven 的頻寬（出處）：cpu＝49.858 GiB/s（真實 pinned H2D，results/m10_precision_mi300x/speed_llama/bw.json）；
#   local＝7.757 GiB/s（第一階段實測本地讀 7,943 MiB/s 的下緣，stage2_factcheck_and_toolbox.md §1 第 10 條）；
#   nfs_real＝0.368 GiB/s（實測 NFS 讀 377 MiB/s 的下緣，同上）；nfs_sim＝0.331（第二階段限速器參數）；b32＝1.4668（bw.json）。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
OUT=$RES/s1_lossless; mkdir -p $OUT
m7run m16-s1-analyze taskset -c 0-15 python code/m16_s1_lossless.py analyze \
  --dump $KVD/llama_T4_64 $KVD/llama_T2_64 $KVD/qwen_T4_64 $KVD/qwen_T2_64 $KVD/llama_T4_252 $KVD/qwen_T4_252 \
  --out $OUT --max-blocks 16 --speed-blocks 4 --reps 3 --threads 1 4 8 16 \
  || { nh "11_s1_analyze 失敗"; exit 1; }
m7run m16-s1-breakeven python code/m16_s1_lossless.py breakeven --summary $OUT/summary.json \
  --tier cpu=49.858 --tier local=7.757 --tier nfs_real=0.368 --tier nfs_sim=0.331 --tier b32=1.4668 \
  || { nh "11_s1_analyze: breakeven 失敗"; exit 1; }
