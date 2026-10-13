#!/usr/bin/env bash
# 編譯給老師的進度報告：xelatex 兩次（交叉引用），中間檔放 /tmp 以外的暫存目錄，PDF 複製回本目錄
set -euo pipefail
cd "$(dirname "$0")"
OUT="${BUILD_DIR:-/mlsteam/data/tiara/runs/advisor_update_build}"
mkdir -p "$OUT"
export PATH=/root/.TinyTeX/bin/x86_64-linux:$PATH
for i in 1 2; do xelatex -interaction=nonstopmode -halt-on-error -output-directory="$OUT" advisor_update_20261012.tex > "$OUT/xelatex_pass$i.log" 2>&1; done
cp "$OUT/advisor_update_20261012.pdf" .
echo "PDF: $(pwd)/advisor_update_20261012.pdf"
