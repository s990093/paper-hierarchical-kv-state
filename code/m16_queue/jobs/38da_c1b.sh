# GPU-NEED: one
# CARDS: 0,1
# NEED-GB: 150
# MAX-H: 2
# AFTER: 33z_c1a_report
# C1b 第 1 段（code/m16_c1b_restore.py；docs/stage3_20261019/c1b_howto.md）：選擇性還原——回來的問題只載入 k% 的 block，(a) 不補載 vs (b) 補載。
# 題目＝C1a 的同一組（DS2 兩輪題 seed 70000 起：T2–T4 各 45、T5 15；每個模型 150 題 × 41 格）。本體見 c1b_common.sh：
#   先在兩張卡上各跑一個 selfcheck（Llama 卡 0、Qwen 卡 1；失敗 → NEEDS_HUMAN、exit 1）＋qtime（閒的 GPU 量 Quest 打分的時間），
#   再開兩卡共用的工作池（2×12 個 (模型, shard)，兩個模型交錯；每張卡 S≤3 個行程，照 selfcheck 的峰值定），兩張卡都忙到工作池空。
# 時間〔估，howto §5〕：兩個模型合計約 9.4 行程·小時 ÷ 6 個行程 ≈ 94 分鐘＋載模型約 6 分鐘＋selfcheck／qtime 約 10 分鐘 ≈ 1.8 小時（±50%）。
#   預算 105 分鐘（剩 < 10 分鐘不再領新工作；每個行程自己也照 --budget-min 停），沒做完的題留給 38db_c1b（逐題續跑）。
# NEED-GB 150：每張卡 3 個行程 × ≤ 50 GiB〔估：peak ≤ 45 GiB，見 c1b_common.sh〕。
JOB=38da_c1b; BUDGET_MIN=105; DO_QTIME=1
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/c1b_common.sh
exit $rc
