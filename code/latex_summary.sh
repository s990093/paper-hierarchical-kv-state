#!/usr/bin/env bash
# `make` 編譯完之後的摘要：三種警告的計數＋前幾筆的位置，與 PDF 頁數。
# 用法：bash code/latex_summary.sh main
set -u
M=${1:-main}; LOG=$M.log; N=${N:-5}
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then R=$'\e[31m'; G=$'\e[32m'; Y=$'\e[33m'; D=$'\e[2m'; B=$'\e[1m'; X=$'\e[0m'
else R= G= Y= D= B= X=; fi
[ -f "$LOG" ] || { echo "${R}找不到 $LOG${X}"; exit 1; }

# 一行一筆：<種類>\t<內容>
items() {
  case $1 in
    font)  grep -E 'Font Warning' "$LOG" | sed -E 's/^.*Font Warning: *//' ;;
    box)   grep -E '^Overfull' "$LOG" | sed -E 's/^Overfull \\hbox \(([0-9.]+pt) too wide\) (in paragraph )?(at lines? [0-9-]+|detected at line [0-9]+|has occurred while \\output is active).*/\1 \3/' ;;
    undef) grep -oE "(Citation|Reference) [\`'][^']*' (on page [0-9]+ )?undefined" "$LOG" | sort -u ;;
  esac
}
row() {  # 標籤 種類
  local n; n=$(items "$2" | grep -c . || true)
  if [ "$n" -eq 0 ]; then printf '  %s%-14s 0%s\n' "$G" "$1" "$X"
  else printf '  %s%-14s %d%s\n' "$Y" "$1" "$n" "$X"; items "$2" | head -n "$N" | sed "s/^/      ${D}/; s/\$/${X}/"
       [ "$n" -gt "$N" ] && printf '      %s… 還有 %d 筆（grep %s）%s\n' "$D" $((n - N)) "$LOG" "$X"; fi
}
echo "${B}── build summary ──${X}"
row "font warnings" font
row "overfull box"  box
row "undefined"     undef
out=$(grep -oE "Output written on $M.pdf \([^)]*\)" "$LOG" | tail -1)
[ -n "$out" ] && echo "  ${G}✓${X} ${out#Output written on }" || echo "  ${R}✗ 沒有產生 $M.pdf${X}"
