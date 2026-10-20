# C1a 正式執行的共用本體（由 33a_c1a.sh、33b_c1a.sh source；job 先設 JOB、SHARDS）。這個檔不是 job，不要放進佇列。
# 設計、停止條件：docs/stage3_20261019/c1a_s2_howto.md（§1、§7；10/20 排程見檔尾）。
# 每個模型 150 題（T2–T4 各 45、T5 15；seed 70000 起）切成 N＝6 個 shard：--shard s/6＝第 k 題給 k mod 6＝s 的行程。
# 單一 shard 的位置帶不平均（T2–T4 的帶是 k mod 3 輪流），但 s0–2（33a）、s3–5（33b）各自涵蓋每個任務、每種位置帶組合
# （33a：T2 24、T3 21、T4 24、T5 6 題，帶組合各三分之一）。兩張卡並行：卡 0 跑 Llama、卡 1 跑 Qwen；每張卡同時開 SHARDS 裡的幾個行程。
# 同一張卡上的 shard 彼此會被 GpuWatcher 記成入侵者（own_gpu_intruders ≥ 2、run 目錄留 CONTAMINATED）：
#   品質實驗的規則照用（CLAUDE.md §3）：分數保留，t_score_s 作廢；每題秒數看 04_smoke_c1a（一張卡一個行程）。
# 續跑：同一個 job 重下即可（寫完的題跳過，寫一半的題整題重算，report 去重）。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
C=$RES/c1a_predictive; mkdir -p $C
N=6          # 總 shard 數。33a、33b 必須相同：改了 N，shard 對到的題就變了，續跑判斷（逐檔）會失準
shards() {   # shards <模型> <shard...>：同一張卡上同時跑這幾個 shard，全部成功才回 0
  local m=$1 s x rc=0
  local -a ps=()
  shift
  for s in "$@"; do
    m7run m16-c1a-$m-s$s python code/m16_c1a_predict.py run --model $m --shard $s/$N --out $C/c1a_${m}_s$s.csv &
    ps+=("$!:$s")
  done
  for x in "${ps[@]}"; do
    wait ${x%%:*} || { rc=1; nh "$JOB: $m shard ${x##*:}/$N 失敗（$(lastrun m16-c1a-$m-s${x##*:})）"; }
  done
  return $rc
}
oncard 0 shards llama $SHARDS & p0=$!
oncard 1 shards qwen $SHARDS & p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
/bin/ls -la $C
[ $r0 = 0 ] && [ $r1 = 0 ] || exit 1
echo "$JOB 完成（shard $SHARDS／$N）$(date -u +%FT%TZ)"
