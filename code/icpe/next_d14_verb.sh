#!/usr/bin/env bash
# next_d14_verb.sh — icpe-data：PREREG 偏離 14 第一段的版本 B 檢查表（*_p1，換上 d14 列）＋「其他元件不變」核對。
# STATUS_DATA 2026-11-07 16:34（使用者在 icpe-main 的決定：第二段的重現檢查要用這張表）。正式的論文表由 icpe-main 另外算、互相核對。
# 用法（在 §4.1 的 run 目錄裡，RUN 已設；DATAREPO＝icpe-next-data 的 worktree）：
#   RUN=… DATAREPO=… bash next_d14_verb.sh <家族：tl sa bl mc fi sg wc uc lm ms oh> <b 的根目錄> <原本的版本 B 表 *_p1_headroom_components.csv[.xz]>
# 步驟（任何一步失敗就停）：
#   1. next_d14_patch.py：家族的 main 與 P1 _ttl 每個 block，原本的列換上 d14 的列 → $RUN/patched/
#   2. main_rowcheck（icpe-next-main b54d3cff 的封存版）：main 54 格 × 17 CPU × 8 GPU、_ttl 54 格 × 15 CPU × 8 GPU
#   3. next_sim_analyze --sweep M T --pf-target cpu → $RUN/out/<家族>_p1_headroom_components.csv（同版本 B 的算法）
#   4. next_d14_cmp.py table：O-PF、O-ALL(report) 以外的元件和原本的版本 B 表 7 欄逐值相同 → D14_TABLE_RESULT IDENTICAL
set -euo pipefail
FAM=$1; B=$2; VERB=$3
: "${RUN:?要在 run 目錄裡跑}" "${DATAREPO:?}"
cd $DATAREPO
PY="nice -n 10 ${NEXT_B_PY:-/ssd6/hungwei/icpe-next/venv/bin/python} -I"
RC=${MAIN_ROWCHECK:-/ssd6/hungwei/icpe-next/code-archive/main-b54d3cff/main_rowcheck.py}
[ "$(sha256sum $RC | cut -c1-8)" = cde7fb3f ] || { echo "main_rowcheck 不是 b54d3cff 的封存版（cde7fb3f）→ 停"; exit 1; }
LIST=results/icpe_next/sim/dev14_rerun_list.csv
case $FAM in
  tl) MP=tl_main; MG='tl_c*';  TP=tl_p1;  TG='tl_c*_ttl';;
  sa) MP=sa_main; MG='sa_N*';  TP=sa_p1;  TG='sa_N*_ttl';;
  bl) MP=bl_main; MG='bailian3'; TP=bl_p1; TG='bailian3_ttl';;
  mc) MP=old_main; MG='mooncake'; TP=old_p1; TG='mooncake_ttl';;
  fi) MP=old_main; MG='fi6w';  TP=old_p1; TG='fi6w_ttl';;
  sg|lm|uc) MP=chat_main; MG="${FAM}_T*"; TP=chat_p1; TG="${FAM}_T*_ttl";;
  wc) MP=chat_main; MG='wc_s*'; TP=chat_p1; TG='wc_s*_ttl';;
  ms|oh) MP=agent_main; MG="${FAM}_N*"; TP=agent_p1; TG="${FAM}_N*_ttl";;
  *) echo "不認得的家族 $FAM"; exit 1;;
esac
blk() { ( cd $B/$1 && for d in $2; do [ -d "$d" ] && echo "$1/$d"; done ); }
MB=$(blk $MP "$MG" | grep -v '_ttl$' || true); TB=$(blk $TP "$TG")
[ -n "$MB" ] && [ -n "$TB" ] || { echo "找不到 block：$MP/$MG 或 $TP/$TG → 停"; exit 1; }
echo "== patch $(date -Is) main=$(echo $MB | wc -w) ttl=$(echo $TB | wc -w)"
$PY code/icpe/next_d14_patch.py --b-root $B --list $LIST --blocks $MB $TB --out-dir $RUN/patched --summary $RUN/patch_summary.csv | tail -3
M=$(for x in $MB; do echo $RUN/patched/$x/x1_sweep.csv.xz; done); T=$(for x in $TB; do echo $RUN/patched/$x/x1_sweep.csv.xz; done)
echo "== rowcheck $(date -Is)"
$PY $RC --label ${FAM}_main_d14 --sweep $M --expect-cells 54 --expect-cpu 17 --expect-gpu 8 --out $RUN/rowcheck.csv
$PY $RC --label ${FAM}_p1_ttl_d14 --sweep $T --expect-cells 54 --expect-cpu 15 --expect-gpu 8 --out $RUN/rowcheck.csv
echo "== analyze $(date -Is)"
mkdir -p $RUN/out
$PY code/icpe/next_sim_analyze.py --sweep $M $T --out-dir $RUN/out --out-prefix ${FAM}_p1 --pf-target cpu > $RUN/analyze.log 2>&1 || { tail -20 $RUN/analyze.log; exit 1; }
echo "== 其他元件和原本的版本 B 表比 $(date -Is)"
$PY code/icpe/next_d14_cmp.py table $RUN/out/${FAM}_p1_headroom_components.csv $VERB --skip-components O-PF "O-ALL(report)" \
  --out $RUN/table_vs_verB.csv | tee $RUN/table_vs_verB.log
grep -q "D14_TABLE_RESULT IDENTICAL" $RUN/table_vs_verB.log || { echo "O-PF／O-ALL 以外的元件變了 → 停"; exit 1; }
echo "D14_VERB_DONE $FAM $RUN/out/${FAM}_p1_headroom_components.csv"
