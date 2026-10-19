# GPU-NEED: one　（10/19 04:4xZ：44c 的接續——只量品質）
# 44c 的第 3 個 job（lcs-llama-252-s2）03:39Z OOM：開的時候卡上空 93 GB，但前兩個 Llama 128K（各 need 78 GB）還沒長到峰值。
# pool 只看「現在空多少」→ 改成全部 pool 共用一本顯存帳（RUNLOG 15.6）。照規則 pool 不再開新的，所以把 44c 的 pool 停掉
# （正在跑的 s0、s1 不動，已登記在帳上），剩下的 10 個 job 在這裡用新的 pool 跑；指令、題數、設定都和 44c 相同（screen 照題號續跑）。
# 最後等 s0、s1 也跑完才彙整、結束（之後的 44ca 是量時間的，要兩張卡都空）。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
pool $JOBS/jobs_92_rest.json 4 || echo "44c_x：pool jobs_92_rest.json 有 job 失敗（$JOBS/pool_lcs-*.log）" >> $JOBS/NEEDS_HUMAN
while pgrep -f "runsh m10-lc-screen-" > /dev/null; do sleep 60; done
grep -q "exit=0" $JOBS/pool_lcs-llama-252-s0.log && grep -q "exit=0" $JOBS/pool_lcs-llama-252-s1.log \
  || echo "44c_x：44c 的 s0／s1 沒有正常結束（看 pool_lcs-llama-252-s[01].log）" >> $JOBS/NEEDS_HUMAN
python $REPO/code/m10_longctx_analyze.py --raw /mlsteam/data/tiara/longctx_m10 --out-dir $REPO/results/m10_longctx_mi300x/analysis > $JOBS/queue/44c_analyze.out 2>&1 \
  || echo "longctx analyze 失敗（queue/44c_analyze.out）" >> $JOBS/NEEDS_HUMAN
exit 0
