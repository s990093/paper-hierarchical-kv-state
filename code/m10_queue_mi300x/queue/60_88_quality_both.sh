# GPU-NEED: one　（10/20 09:5xZ：從 queue_g0 移過來。main 的佇列做完了，helper 還卡在 55 最後一個 job（f8m-qwen-T1，約 4 小時），
#   照原本的順序 60、88 要等它，兩張卡大多空著。改成 main 這邊兩個 pool 同時跑（共用顯存帳 vram_reserved.json；兩張卡都能用）。
#   兩段的內容沒改，原檔在 parts/。只量品質（分數）；時間欄不用。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
bash $JOBS/parts/60_sensmap.part & p60=$!
sleep 90   # 讓 60 的 job 先登記顯存，再開 88
bash $JOBS/parts/88_report_only_int4_T1_LB.part & p88=$!
wait $p60; r60=$?; wait $p88; r88=$?
echo "[60_88] 60 rc=$r60、88 rc=$r88 $(date -u +%FT%TZ)"
exit 0
