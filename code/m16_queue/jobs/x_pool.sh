# 兩卡共用工作池（10/21；38b*／38c* 用）。由 job 在 lib.sh、x_common.sh 之後 source。
# **不要把這個檔放進佇列目錄**（runner 會把佇列裡的 *.sh 都當成 job）；job 一律用 repo 裡的路徑 source 它。說明：x_stack_howto.md §14.5。
#
# 為什麼：X1／X2 的 job 是「卡 0 跑 Llama、卡 1 跑 Qwen」，先做完的那張卡會閒到整個 job 結束（10/21 37d：卡 1 閒著）。
# 作法：
#   - 工作切成固定的單位清單（(模型, shard i/n, 階段)；n 在兩張卡之間共用、不重疊——同一個 shard 只會有一個行程）。
#   - 每張卡一個 worker，照清單順序領下一個還沒被領的單位；領取鎖＝mkdir $池/claim/<k>（原子操作：同一個單位只會被一張卡領到）。
#   - 一張卡上同時最多 XP_MAXP 個行程、登記的 GiB 合計 ≤ XP_CAP_GB（每個單位登記自己的 GiB：照實測 peak 加餘量）。放不下就等手上的一個結束。
#   - 領到後先跑單位的「待做檢查」（m16_xstack.py pending，只用 CPU、印題數）：0 → 這個 shard 已經做完，跳過、不開行程、不載模型。
#   - 指令裡的 @B@ 換成「開跑時離工作池預算結束還有幾分鐘」（--budget-min）：逐題續跑照舊，沒做完的題留給下一段 job（下一段重建工作池）。
#   - 任一行程失敗 → 那張卡不再領新的單位、等手上的跑完 → xp_run 回 1（job 寫 NEEDS_HUMAN、exit 1；規則 2）。
#   - 單位照清單順序領：job 用 xp_order 把兩個模型的單位交錯排（照 (i+0.5)/n），兩張卡都忙到工作池空。
# 不動 x_common.sh 的 x_par／x2_model／x2_seq（37e／37f 還在用）。
# self-test（python code/m16_xstack.py --self-test）用假的 m7run／oncard 跑這支：每個單位恰好被領一次、跳過做完的、GiB／行程數上限、兩張卡都有領、失敗回非 0。

XP_CAP_GB=${XP_CAP_GB:-165}     # 每張卡登記的 GiB 上限（MI300X 192 GB；job 檔頭 NEED-GB 跟著它）
XP_MAXP=${XP_MAXP:-4}           # 每張卡同時最多幾個行程
XP_MIN_LEFT=${XP_MIN_LEFT:-5}   # 預算剩不到這麼多分鐘就不再領新的單位
XP_POLL_S=${XP_POLL_S:-5}       # 等行程結束時多久看一次（每個單位結束時由它自己的子 shell 寫 $池/claim/<k>/rc；不用 wait -n：
                                #   bash 已經回收的背景工作 wait -n 會說 no such job，10/21 self-test 抓到）

xp_log() { local P=$1; shift; echo "$(date -u +%FT%TZ) $*" | tee -a "$P/log"; }

xp_init() {   # xp_init <池目錄>：清空、重建（每一段 job 一個新的池；做完的 shard 由待做檢查跳過）
  rm -rf "$1"; mkdir -p "$1/u" "$1/claim"; : > "$1/log"; echo 0 > "$1/n"
}

xp_add() {    # xp_add <池> <模型> <GiB> <run 短名> <待做檢查（一個字串，eval 後印題數；- ＝不查）> <指令…（@B@＝剩下的分鐘）>
  local P=$1 m=$2 gb=$3 name=$4 pend=$5 k; shift 5
  k=$(cat "$P/n")
  mkdir -p "$P/u/$k"
  printf '%s\n' "$m" > "$P/u/$k/model"; printf '%s\n' "$gb" > "$P/u/$k/gb"; printf '%s\n' "$name" > "$P/u/$k/name"
  printf '%s\n' "$pend" > "$P/u/$k/pend"
  printf '%q ' "$@" > "$P/u/$k/cmd"
  echo $((k + 1)) > "$P/n"
}

xp_order() {  # xp_order <模型> <n> [<模型> <n> …]：印「模型:i」，照 (i+0.5)/n 排（兩個模型交錯；同分時照參數順序）
  local m n i
  while [ $# -ge 2 ]; do
    m=$1 n=$2; shift 2
    for ((i = 0; i < n; i++)); do awk -v i=$i -v n=$n -v m=$m 'BEGIN{printf "%.6f %s:%d\n", (i + 0.5) / n, m, i}'; done
  done | LC_ALL=C sort -s -k1,1n | awk '{print $2}'
}

xp_worker() { # xp_worker <池> <卡> <預算結束的 epoch 秒>
  local P=$1 c=$2 tend=$3
  local n k=0 k_ j fit used=0 rc=0 left gb np pend name cmd
  local -A run_gb=()                                                         # 這張卡上還在跑的單位 → GiB
  n=$(cat "$P/n")
  while :; do
    while [ $k -lt $n ] && [ -d "$P/claim/$k" ]; do k=$((k + 1)); done      # k＝第一個還沒被領的
    [ $k -ge $n ] && break
    [ $rc = 0 ] || break
    left=$(( (tend - $(date +%s)) / 60 ))
    if [ $left -le $XP_MIN_LEFT ]; then xp_log "$P" "卡 $c：預算剩 $left 分鐘 → 不再領新的單位（從 $k 起留給下一段）"; break; fi
    fit=-1                                                                   # 照順序找第一個還沒被領、這張卡放得下的
    if [ ${#run_gb[@]} -lt $XP_MAXP ]; then
      for ((j = k; j < n; j++)); do
        [ -d "$P/claim/$j" ] && continue
        [ $((used + $(cat "$P/u/$j/gb"))) -le $XP_CAP_GB ] && { fit=$j; break; }
      done
    fi
    if [ $fit -lt 0 ]; then                                                  # 滿了（行程數或 GiB）→ 等手上的一個結束
      if [ ${#run_gb[@]} = 0 ]; then xp_log "$P" "卡 $c：單位 $k 要 $(cat "$P/u/$k/gb") GiB > 上限 $XP_CAP_GB"; rc=1; break; fi
      xp_reap
      continue
    fi
    mkdir "$P/claim/$fit" 2>/dev/null || continue                            # 另一張卡先領到了 → 重找
    k_=$k; k=$fit
    gb=$(cat "$P/u/$k/gb")
    echo "$c" > "$P/claim/$k/card"
    pend=$(cat "$P/u/$k/pend"); np="?"
    if [ "$pend" != - ]; then
      np=$(eval "$pend") || { xp_log "$P" "卡 $c：單位 $k 的待做檢查失敗（$pend）"; echo 1 > "$P/claim/$k/rc"; rc=1; break; }
      echo "$np" > "$P/claim/$k/pending"
      if [ "$np" = 0 ]; then xp_log "$P" "卡 $c：單位 $k（$(cat "$P/u/$k/name")）已經做完 → 跳過"; k=$k_; continue; fi
    fi
    name=$(cat "$P/u/$k/name"); cmd=$(cat "$P/u/$k/cmd"); cmd=${cmd//@B@/$left}
    xp_log "$P" "卡 $c：領到單位 $k（$name，$gb GiB，待做 $np 題，剩 $left 分鐘；卡上已登記 $used GiB／${#run_gb[@]} 個行程）"
    ( eval "oncard $c m7run $name $cmd"; echo $? > "$P/claim/$k/rc.tmp"; mv "$P/claim/$k/rc.tmp" "$P/claim/$k/rc" ) &
    echo $! > "$P/claim/$k/pid"; run_gb[$k]=$gb; used=$((used + gb))
    k=$k_
  done
  while [ ${#run_gb[@]} -gt 0 ]; do xp_reap; done
  wait
  return $rc
}

xp_reap() {   # （xp_worker 內部用；動態範圍讀寫它的 P、c、run_gb、used、rc）等到手上至少一個單位寫了 rc，全部回收
  local k st any=0
  while :; do
    for k in "${!run_gb[@]}"; do
      [ -e "$P/claim/$k/rc" ] || continue
      st=$(cat "$P/claim/$k/rc"); used=$((used - run_gb[$k])); unset "run_gb[$k]"; any=1
      xp_log "$P" "卡 $c：單位 $k 結束 rc=$st"
      [ "$st" = 0 ] || rc=1
    done
    [ $any = 1 ] && return 0
    sleep $XP_POLL_S
  done
}

xp_run() {    # xp_run <池> <分鐘預算> <卡…>：每張卡一個 worker（背景），等全部結束；回 0＝沒有任何單位失敗
  local P=$1 B=$2 tend c p rc=0 pids=(); shift 2
  tend=$(( $(date +%s) + B * 60 ))
  xp_log "$P" "工作池開始：$(cat "$P/n") 個單位、卡 $*、預算 $B 分鐘、每卡 ≤$XP_MAXP 個行程／≤$XP_CAP_GB GiB"
  for c in "$@"; do xp_worker "$P" "$c" "$tend" & pids+=($!); done
  for p in "${pids[@]}"; do wait $p || rc=1; done
  xp_summary "$P"
  return $rc
}

xp_summary() { # 每個單位：哪張卡領、跳過／rc；沒被領的（留給下一段）
  local P=$1 k n skipped=0 ran=0 failed=0 left=0 line=""
  n=$(cat "$P/n")
  for ((k = 0; k < n; k++)); do
    if [ ! -d "$P/claim/$k" ]; then left=$((left + 1)); continue; fi
    if [ "$(cat "$P/claim/$k/pending" 2>/dev/null)" = 0 ]; then skipped=$((skipped + 1)); continue; fi
    ran=$((ran + 1))
    [ "$(cat "$P/claim/$k/rc" 2>/dev/null)" = 0 ] || failed=$((failed + 1))
    line+=" $(cat "$P/u/$k/name")@卡$(cat "$P/claim/$k/card")"
  done
  xp_log "$P" "工作池結束：跑了 $ran 個（失敗 $failed）、已做完跳過 $skipped 個、沒領到（留給下一段）$left 個；$line"
}
