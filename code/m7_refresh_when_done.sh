#!/usr/bin/env bash
# 等第一階段執行鏈（chainF）跑完，重新產生分析、圖與報告（不手打數字）
cd /mlsteam/workspace/paper-hierarchical-kv-state
source code/m7_env.sh
until grep -q "=== .* DONE" /mlsteam/data/tiara/runs/m7_chainF_*.log; do sleep 60; done
python code/m7_analyze.py > /mlsteam/data/tiara/runs/m7_final_analyze.log 2>&1
python code/m7_report_fill.py >> /mlsteam/data/tiara/runs/m7_final_analyze.log 2>&1
echo "refreshed $(date -Is)" >> /mlsteam/data/tiara/runs/m7_final_analyze.log
