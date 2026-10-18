δ* 用 5% 的對照（13 §11.3；只報告，不是 H200 的判定——H200 的判定照 δ*＝0.284，在 analysis/fp8/）。
做法：results/m10_precision_h200 的副本裡，只把 speed_llama/noise.json 的 delta_star 換成 0.05（inputs/noise_llama_delta005.json），
重算 Llama 的 k0-verdict（FP8、INT4；inputs/）與 m10_analyze speed，再用同一支 code/h200_m10_analyze.sh 走 all 與 compare。其他輸入全部不動。
fp8/：H200 的路（δ* 0.05）。compare_mi300x_path/：假設 K3 成立、照 MI300X 的路（δ* 0.05）。
