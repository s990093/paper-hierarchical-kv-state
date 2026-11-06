#!/usr/bin/env bash
# icpe-rux 的 run 殼（CLAUDE.md §4.1）。用法：rux_run.sh <短名> '<指令>'
# 產出 $ROOT/runs/<RUN_ID>/{cmd.sh,context.txt,stdout.log,stderr.log,exit_code,mem.log}
# 記憶體監看：每 5 s 記 MemAvailable 與 load；MemAvailable < 3 GiB 就砍掉整個行程群組。
set -u
source /root/icpe-next/tools/env-rux.sh
name=$1; shift
cmd=$1
RUN_ID=$(date +%Y%m%d-%H%M%S)-icpe-rux-$name
RUN=$ROOT/runs/$RUN_ID
mkdir -p "$RUN"
{ echo 'source /root/icpe-next/tools/env-rux.sh'; echo "export RUN_ID=$RUN_ID RUN=$RUN"; printf '%s\n' "$cmd"; } > "$RUN/cmd.sh"
{
  date -Is; hostname; uptime; free -g; nproc
  echo "repo_head $(git -C "$REPO" rev-parse HEAD) branch $(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
  echo "repo_dirty $(git -C "$REPO" status --porcelain | wc -l)"
} > "$RUN/context.txt" 2>&1
setsid bash "$RUN/cmd.sh" > "$RUN/stdout.log" 2> "$RUN/stderr.log" &
pid=$!
(
  min=999999999
  while kill -0 $pid 2>/dev/null; do
    ma=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    ld=$(cut -d' ' -f1-3 /proc/loadavg)
    echo "$(date +%H:%M:%S) MemAvailable_kB=$ma load=$ld" >> "$RUN/mem.log"
    if [ "$ma" -lt 3145728 ]; then
      echo "$(date -Is) MEM_STOP MemAvailable_kB=$ma < 3 GiB, killing pgid $pid" >> "$RUN/mem.log"
      kill -TERM -- -$pid 2>/dev/null
      break
    fi
    sleep 5
  done
) &
mon=$!
wait $pid; ec=$?
echo $ec > "$RUN/exit_code"
kill $mon 2>/dev/null; wait $mon 2>/dev/null
{ echo "end $(date -Is)"; uptime; } >> "$RUN/context.txt"
echo "RUN=$RUN exit=$ec"
