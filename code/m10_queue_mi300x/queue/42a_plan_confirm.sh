# 09 §4.3：照海選的 finalists_<m>.json（41 的彙整寫的）寫出 43_sweep_confirm.sh（只用 CPU）。規則 10/15 寫死；程式 10/16 14:xxZ 寫（41 跑完之前）。
# 失敗 → NEEDS_HUMAN；exit 0，不擋 42b 之後的工作（43 就不會出現，44 照跑）。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
for m in llama qwen; do
  [ -f $REPO/results/m10_sweep_mi300x/analysis/finalists_$m.json ] || { echo "42a：沒有 finalists_$m.json（41 的彙整沒寫出來）→ 43 沒排" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
done
python3 $REPO/code/m10_confirm.py plan --jobs $JOBS > $JOBS/queue/42a_plan.out 2>&1 || { echo "42a：m10_confirm.py plan 失敗（queue/42a_plan.out）" | tee -a $JOBS/NEEDS_HUMAN; tail -20 $JOBS/queue/42a_plan.out; exit 0; }
tail -3 $JOBS/queue/42a_plan.out
exit 0
