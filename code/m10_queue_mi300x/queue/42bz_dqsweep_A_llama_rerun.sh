# 10/17 00:2xZ 加（42b 跑的時候；照 CLAUDE.md §3「污染的時間要重量」）：
# 42b 的 A 段 Llama（run 20261017-001207-m10-dqs-A-llama）被監控標 CONTAMINATED_DURING_RUN：24 個 VRAM 0 MiB 的行程，
# 只在 00:12:23 那一次取樣出現（開跑後 16 秒、逐位元檢查那一段，k2＝torch.compile 第一次編譯的時候）。
# 〔推論〕是 Inductor 開的編譯 worker（預設 compile_threads＝32，subprocess），不是外人；Qwen 的 A 同一支程式是 CLEAN。
# 做法：關掉編譯 worker（TORCHINDUCTOR_COMPILE_THREADS=1：只影響編譯在哪裡做，不影響編出來的 kernel）重量 A-llama。
#   - 乾淨，且 int4_kivi 的 K* 不是 k1 → B 段 Llama 的 12 個方法用新的 K* 重量（同目錄；plan 只讀每個目錄最新的 run_id），再重寫 42c。
#   - 還是污染 → 推論不成立：寫 NEEDS_HUMAN，維持 42b 的退路（K*＝k1），不再試。
# 被標污染的 kernels_llama.csv／choice_llama.json 先搬回那次的 run 目錄（不留在 results/，CLAUDE.md §3）；gpu_guard JSON 留著當紀錄。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
SW=$REPO/results/m10_sweep_mi300x/dqsweep; m=llama
OLD=$TIARA_RUNS/20261017-001207-m10-dqs-A-llama
[ -f $SW/A/kernels_llama.csv ] && mv $SW/A/kernels_llama.csv $OLD/kernels_llama.CONTAMINATED.csv
[ -f $SW/A/choice_llama.json ] && mv $SW/A/choice_llama.json $OLD/choice_llama.CONTAMINATED.json
m7run m10-dqs-A-$m-rerun env TORCHINDUCTOR_COMPILE_THREADS=1 python $REPO/code/m10_dqsweep.py kernels --model $m --out-dir $SW/A \
  || { echo "42bz：A $m 重量還是污染或失敗（見 runs/*-m10-dqs-A-$m-rerun）→ 推論不成立，維持 K*＝k1" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
K=$(python3 $REPO/code/m10_dqsweep.py env S0 --model $m --fmt int4_kivi | tr ' ' '\n' | sed -n 's/^M10_DQ_KERNEL=//p')
echo "42bz：A $m 乾淨；int4_kivi 的 K*=$K"
[ "$K" = "k1" ] && { echo "42bz：K* 就是 k1，B 不用重量"; exit 0; }
G=$(fmt_of $m fp8)
for meth in $(python3 $REPO/code/m10_dqsweep.py list B); do
  ( export $(python3 $REPO/code/m10_dqsweep.py env $meth --model $m --fmt int4_kivi)
    m7run m10-dqs-B-$m-$meth-rerun bash -c "python $REPO/code/m10_speed.py grid --model $m --fmt int4_kivi --base-fmt $G --out-dir $SW/B/${m}_$meth --bw-json $OUT/speed_$m/bw.json --bw-names cpu --configs E0 E1 R25 --refs --lengths --launches 1 --reps 10 --launch0 100 --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" ) \
    || echo "42bz：B $m $meth 重量失敗（見 runs/*-m10-dqs-B-$m-$meth-rerun）" >> $JOBS/NEEDS_HUMAN
done
python3 $REPO/code/m10_dqsweep.py plan --sw $SW --jobs $JOBS || echo "42bz：dqsweep plan 失敗（42c 沒重寫）" >> $JOBS/NEEDS_HUMAN
exit 0
