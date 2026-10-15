# m10_speed 彩排（MI300X）——規模縮小，不進任何圖或判定

用途只有一個：確認 `code/m10_speed.py` 的 calib-f、noise、k0、grid 在真 GPU 上跑得通、欄位對、會合點方向合理。
- 規模：launches 1–2、reps 2–3、只 2 個頻寬（或 1 個）、GPU 閒＋忙各一。正式規模（03 §4.2：3 launches × 10 reps、4 頻寬、64 格）在 H200 上跑。
- 合成 token、Qwen2.5-7B-Instruct-1M、fp8_head；GPU 忙用 MI300X 的 busy_levels.json（busy2，duty 0.7）。
- MI300X 的時間數字不和 H200 混用（03 §1）。逐 block 檔放在 `$TIARA_RUNS/<run_id>/blocks.csv`（大檔，不進 git）。
- run_id 與指令見 `results/RUNLOG_MI300X.md`（Milestone 10-pilot）或各 CSV 的 run_id 欄。
