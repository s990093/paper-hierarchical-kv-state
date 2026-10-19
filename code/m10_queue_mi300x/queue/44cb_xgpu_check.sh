# 10/19（使用者：「真的不能兩張卡嗎」）：量時間的 stage 能不能讓另一張卡同時跑品質工作？先量再說。量時間 → 這一段本身兩張卡獨佔。
# 判準（寫的時候沒有任何這個實驗的數字；RUNLOG 15.10）：
#   1. 監控分卡：在卡 0 放一個假的 GPU 行程（HIP_VISIBLE_DEVICES=0），卡 1 的 gpu_guard 不能把它算成入侵者、卡 0 的要看得到。
#   2. 干擾：卡 1 上量還原（S0、int4_kivi、E0／E1s、cpu＋b32、閒／忙、Cake＋cpu 全部載入；1 launch × 10 rep），
#      卡 0「空著」(A) 和「跑 4 個真的品質 process」(B) 交替 A B A B，兩個模型。
#      每格 B 的中位數 ÷ A 的中位數：所有格子的幾何平均 ≤ 1.02、而且沒有一格 > 1.08；A、B 的 gpu_guard 全 CLEAN。
#   全過 → touch XGPU_OK：之後量時間的 stage 只拿卡 1 的鎖，helper 在卡 0 照跑（runner v2）。沒過 → 照舊兩張卡獨佔。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
X=$REPO/results/m10_xgpu_mi300x; mkdir -p $X; rm -f $JOBS/XGPU_OK
# 1. 監控分卡
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 TIARA_GPU=0 python -c "import torch,time; x=torch.empty(2**31,dtype=torch.uint8,device='cuda'); time.sleep(120)" &
DUMMY=$!
sleep 40
python - <<'PY' > $X/attribution.json 2> $X/attribution.err
import json, os, sys
sys.path.insert(0, "/mlsteam/workspace/paper-hierarchical-kv-state/code")
import gpu_guard as G
f1, f0 = G.foreign_on(1), G.foreign_on(0)
ok = (f1 == []) and any(x.get("kind") == "same_container" for x in f0)
print(json.dumps(dict(card1_view=f1, card0_view=f0, ok=ok), ensure_ascii=False, indent=1))
PY
kill $DUMMY 2>/dev/null; wait $DUMMY 2>/dev/null
python3 -c "import json,sys; sys.exit(0 if json.load(open('$X/attribution.json'))['ok'] else 1)" \
  || { echo "44cb：gpu_guard 分卡沒過（$X/attribution.json）→ 量時間照舊兩張卡獨佔" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
sleep 20
# 2. 干擾 A／B
bload_start() {
  BL=""
  for i in 0 1 2 3; do
    HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 TIARA_GPU=0 python $REPO/code/m10_quality.py run --model qwen --params-file $P \
      --configs E0 E1@int4_kivi --modes A --seed0 $((99000 + 100 * i)) --tasks T2 --n 60 --out $X/bload_$1_$i.csv > $X/bload_$1_$i.log 2>&1 &
    BL="$BL $!"
  done
  sleep 150
}
bload_stop() { for p in $BL; do kill $p 2>/dev/null; done; for p in $BL; do wait $p 2>/dev/null; done; sleep 30; }
l0=900
for m in qwen llama; do
  G=$(fmt_of $m fp8); E=$(python3 $REPO/code/m10_restore2.py env P0 --model $m --fmt int4_kivi)
  for cond in A1 B1 A2 B2; do
    case $cond in B*) bload_start ${m}_$cond ;; esac
    m7run m10-xgpu-$m-$cond env $E bash -c "python $REPO/code/m10_speed.py grid --model $m --fmt int4_kivi --base-fmt $G --out-dir $X/${m}_$cond --bw-json $OUT/speed_$m/bw.json --bw-names cpu b32 --configs E0 E1s --refs --lengths --extra-modes loadall --extra-tiers cpu --launches 1 --reps 10 --launch0 $l0 --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" \
      || echo "44cb：干擾實驗 $m $cond 失敗" >> $JOBS/NEEDS_HUMAN
    case $cond in B*) bload_stop ;; esac
    l0=$((l0 + 1))
  done
done
# 3. 判定
python3 - $X <<'PY' || echo "44cb：判定程式失敗" >> $JOBS/NEEDS_HUMAN
import csv, glob, json, math, os, statistics, sys
sys.path.insert(0, "/mlsteam/workspace/paper-hierarchical-kv-state/code")
import m10_restore2 as R
X = sys.argv[1]
out, ratios, noise, guards_ok = {}, [], [], True
for m in ("qwen", "llama"):
    rows = {}
    for c in ("A1", "B1", "A2", "B2"):
        d = os.path.join(X, f"{m}_{c}")
        rr = R.latest_rows(d)
        u = R.usable(d, rr)
        guards_ok &= bool(u)
        out[f"{m}_{c}_usable"] = bool(u)
        rows[c] = rr
    keys = sorted({(r["bw_name"], r["gpu_state"], r["config"], r["mode"]) for r in rows["A1"]})
    for k in keys:
        def med(cs):
            v = [float(r["t_restore"]) for c in cs for r in rows[c] if (r["bw_name"], r["gpu_state"], r["config"], r["mode"]) == k]
            return statistics.median(v) if v else None
        a, b, a1, a2 = med(["A1", "A2"]), med(["B1", "B2"]), med(["A1"]), med(["A2"])
        if a and b:
            ratios.append(b / a)
            noise.append(a2 / a1 if a1 and a2 else 1.0)
            out["|".join((m,) + k)] = dict(A=a, B=b, ratio=round(b / a, 4), A2_over_A1=round(a2 / a1, 4) if a1 and a2 else None)
gm = math.exp(sum(math.log(x) for x in ratios) / len(ratios)) if ratios else None
gmn = math.exp(sum(math.log(x) for x in noise) / len(noise)) if noise else None
ok = bool(ratios) and guards_ok and gm <= 1.02 and max(ratios) <= 1.08
res = dict(rule="44cb：B/A 幾何平均 ≤ 1.02、沒有一格 > 1.08、guard 全 CLEAN", n_cells=len(ratios), geomean_B_over_A=gm,
           max_B_over_A=max(ratios) if ratios else None, min_B_over_A=min(ratios) if ratios else None,
           noise_geomean_A2_over_A1=gmn, guards_ok=guards_ok, pass_=ok, cells=out)
json.dump(res, open(os.path.join(X, "verdict.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in res.items() if k != "cells"}, ensure_ascii=False, indent=1))
sys.exit(0 if ok else 3)
PY
rc=$?
if [ $rc -eq 0 ]; then touch $JOBS/XGPU_OK; echo "44cb：過 → XGPU_OK（之後量時間的 stage 和 helper 平行）";
else echo "44cb：干擾實驗沒過（$X/verdict.json）→ 量時間照舊兩張卡獨佔" | tee -a $JOBS/NEEDS_HUMAN; fi
exit 0
