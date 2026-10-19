#!/usr/bin/env bash
# MI300X GPU 佇列的狀態（10/19；像 H200 那邊用 make 看 queue）。用法：qstat.sh [status|q|gpu|jobs|nh|tail|done|ledger]
# 在 repo 根目錄：make status ／ make q ／ make gpu ／ make jobs ／ make nh ／ make tail ／ make done ／ make watch
export LC_ALL=C
J=/tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs
cmd=${1:-status}
[ -d "$J/queue" ] || { echo "找不到佇列 $J（容器重啟後還沒還原？見 /mlsteam/workspace/RESUME_20261019_2GPU.md）"; exit 1; }

hr() { printf '%s\n' "────────────────────────────────────────────────────────────────────────"; }
now() { echo "現在：$(date -u '+%m/%d %H:%M UTC')　＝　台灣 $(TZ=Asia/Taipei date '+%m/%d %H:%M')"; }
el() { python3 -c "import datetime as D,sys;t=D.datetime.fromisoformat(sys.argv[1].replace('Z','+00:00'));s=int((D.datetime.now(D.timezone.utc)-t).total_seconds());print(f'{s//3600}h{s%3600//60:02d}m')" "$1"; }

runners() {
  for role in main helper; do
    log=$J/runner.log; [ $role = helper ] && log=$J/runner_g0.log
    pid=""
    for p in $(pgrep -f "scratchpad/jobs/runner.sh"); do
      grep -qz "^ROLE=$role$" /proc/$p/environ 2>/dev/null && pid=$p
    done
    last=$(grep "runner:$role" $log 2>/dev/null | tail -1)
    if [ -n "$pid" ]; then st="活著（pid $pid）"; else st="⚠ 沒在跑"; fi
    gpu=1; [ $role = helper ] && gpu=0
    echo "[$role・卡 $gpu] $st"
    if echo "$last" | grep -q " START "; then
      name=$(echo "$last" | awk '{print $4}'); t=$(echo "$last" | grep -oE '[0-9-]+T[0-9:]+Z' | tail -1)
      echo "    正在跑：$name（$(echo "$last" | grep -oE 'need=[a-z]+')；已跑 $(el $t)）"
    else
      echo "    最後一筆：$last"
    fi
  done
  [ -e $J/WANT_BOTH ] && echo "    ⏸ WANT_BOTH：main 在量時間，兩張卡獨佔（helper 暫停）"
}

gpu() {
  amd-smi list 2>/dev/null | awk '/GPU:/{g=$2} /BDF:/{print g, $2}' > /tmp/.qstat_bdf.$$
  amd-smi metric --mem-usage --usage --json 2>/dev/null | python3 -c "
import json,sys
bdf={l.split()[0]:l.split()[1] for l in open('/tmp/.qstat_bdf.$$')}
d=json.load(sys.stdin); d=d if isinstance(d,list) else d['gpu_data']
for g in d:
    m=g['mem_usage']; u=g.get('usage',{}); a=u.get('gfx_activity'); a=a.get('value') if isinstance(a,dict) else a
    tot=m['total_vram']['value']/1024; free=m['free_vram']['value']/1024
    role='main（量時間的卡）' if g['gpu']==1 else 'helper'
    print(f\"卡 {g['gpu']}（{bdf.get(str(g['gpu']),'?')}，{role}）：顯存 {tot-free:5.0f}/{tot:.0f} GB　使用率 {a}%\")
"
  rm -f /tmp/.qstat_bdf.$$
}

queue() {
  for q in queue queue_g0; do
    [ -d $J/$q ] || continue
    role=main; [ $q = queue_g0 ] && role=helper
    echo "[$role] 排隊中："
    n=0
    for f in $(ls $J/$q/*.sh 2>/dev/null | sort); do
      need=$(grep -m1 -oE "^# GPU-NEED: *(one|both)" $f | awk '{print $3}'); [ -z "$need" ] && need=both
      [ $role = helper ] && need=one
      tag="一張卡（品質／CPU）"; [ $need = both ] && tag="量時間（兩張卡獨佔）"
      echo "    $(basename $f .sh)　— $tag"; n=$((n+1))
    done
    [ $n = 0 ] && echo "    （空）"
  done
}

jobs() {
  for q in queue queue_g0; do
    for f in $(ls $J/$q/*.running 2>/dev/null); do
      name=$(basename $f .running)
      echo "[$name]"
      dg=1; [ $q = queue_g0 ] && dg=0
      if grep -q "^\[pool\]" $J/$q/$name.log 2>/dev/null; then
        python3 - "$J/$q/$name.log" "$J" "$dg" <<'PY'
import sys, re, os, glob, json
log, J, dg = sys.argv[1], sys.argv[2], sys.argv[3]
st, dn, fl = {}, set(), set()
for l in open(log):
    m = re.match(r"\[pool\] start (\S+)(?: gpu(\d))?", l)
    if m: st[m.group(1)] = m.group(2) or dg
    m = re.match(r"\[pool\] done (\S+)", l)
    if m: dn.add(m.group(1))
    m = re.match(r"\[pool\] FAILED (\S+)", l)
    if m: fl.add(m.group(1))
total = None
for jf in glob.glob(os.path.join(J, "jobs_*.json")):
    try:
        names = [x["name"] for x in json.load(open(jf))]
    except Exception:
        continue
    if names and all(n in names for n in st):
        total = len(names) if total is None or len(names) < total else total
run = [n for n in st if n not in dn and n not in fl]
print(f"    job：{len(dn)} 完成、{len(fl)} 失敗、{len(run)} 在跑" + (f"、共 {total}" if total else ""))
for n in run:
    p = os.path.join(J, f"pool_{n}.log")
    last = ""
    if os.path.exists(p):                   # job 的進度在 run 目錄的 stdout（runsh 把 dir 寫在 pool log）
        dirs = [x.split("dir=", 1)[1].strip() for x in open(p, errors="replace") if "[runsh] dir=" in x]
        so = os.path.join(dirs[-1], "stdout.log") if dirs else ""
        if so and os.path.exists(so):
            ls = [x.strip() for x in open(so, errors="replace") if re.match(r"\[\d+\]", x)]
            last = ls[-1][:100] if ls else "（載入中）"
    print(f"      卡 {st[n]}  {n:24s} {last}")
for n in sorted(fl):
    print(f"      ✗ 失敗：{n}（看 {J}/pool_{n}.log）")
PY
      else
        if [ -s $J/$q/$name.log ]; then tail -2 $J/$q/$name.log | cut -c1-140 | sed 's/^/    /'; else echo "    （還沒開任何 job；可能在等顯存，看「顯存帳」）"; fi
      fi
    done
  done
}

done_() {
  python3 - $J/runner.log $J/runner_g0.log <<'PY'
import re, sys, datetime as D
ev = []
for f in sys.argv[1:]:
    try:
        for l in open(f):
            m = re.search(r"\[(runner[^\]]*)\] (START|DONE|FAILED|PREEMPTED) (\S+).*?(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)", l)
            if m: ev.append(m.groups())
    except FileNotFoundError:
        pass
st = {}
out = []
for r, k, n, t in ev:
    t = D.datetime.fromisoformat(t.replace("Z", "+00:00"))
    if k == "START": st[n] = t
    elif n in st:
        tw = (t + D.timedelta(hours=8)).strftime("%m/%d %H:%M")
        out.append(f"    {k:9s} {n:32s} {(t - st[n]).total_seconds() / 3600:5.2f} h　（台灣 {tw}）")
print("最近完成：")
print("\n".join(out[-8:]) or "    （沒有）")
PY
}

nh() {
  n=$(grep -c . $J/NEEDS_HUMAN 2>/dev/null); n=${n:-0}
  if [ "$n" = 0 ]; then echo "NEEDS_HUMAN：沒有待處理的（處理過的在 NEEDS_HUMAN.handled）"; else
  echo "NEEDS_HUMAN：$n 行待處理"; tail -6 $J/NEEDS_HUMAN | cut -c1-160 | sed 's/^/    /'; fi
}

ledger() {
  python3 - $J/vram_reserved.json <<'PY'
import json, os, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    d = {}
d = {k: v for k, v in d.items() if os.path.exists(f"/proc/{v['pid']}")}
for g in (0, 1):
    xs = [f"{k}({v['need']:.0f})" for k, v in d.items() if v["gpu"] == g]
    print(f"    卡 {g} 已登記 {sum(v['need'] for v in d.values() if v['gpu'] == g):.0f} GB：{' '.join(xs) or '—'}")
PY
}

case $cmd in
  status)
    now; hr; runners; hr; gpu; echo "顯存帳（pool 開 job 前登記的峰值）："; ledger; hr; jobs; hr; queue; hr; done_; hr; nh
    b=/mlsteam/data/tiara/jobs_backup/auto/BACKUP_TIME; [ -f $b ] && echo "佇列備份：$(cat $b)（每 30 分鐘）" ;;
  q) now; queue ;;
  gpu) now; gpu; ledger ;;
  jobs) now; jobs ;;
  nh) nh ;;
  done) done_ ;;
  ledger) ledger ;;
  tail)
    f=$(ls $J/queue/*.running 2>/dev/null | head -1); [ -n "$f" ] && { echo "== $(basename $f .running)"; tail -${N:-20} ${f%.running}.log; }
    f=$(ls $J/queue_g0/*.running 2>/dev/null | head -1); [ -n "$f" ] && { echo "== $(basename $f .running)（helper）"; tail -${N:-20} ${f%.running}.log; } ;;
  *) echo "用法：qstat.sh [status|q|gpu|jobs|nh|tail|done|ledger]"; exit 1 ;;
esac
