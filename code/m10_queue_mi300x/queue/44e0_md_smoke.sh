# 14 號（docs/precision_20261013/14_match_drop_plan.md §5、§8.6）：新程式在 GPU 上的冒煙。量時間的部分在最後 → 兩張卡獨佔（沒寫 GPU-NEED＝both）。
# 沒過 → 寫 NEEDS_HUMAN、立旗子（NO_DS／NO_D3／NO_M1／NO_M2），後面用到那段程式的 stage 自己跳過；其他照跑。輸出放 scratch，不進 results。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
S=$JOBS/smoke_md; rm -rf $S; mkdir -p $S
MA=$REPO/results/m10_match_mi300x; mkdir -p $MA
# 0. M2／D3 用的頻寬表：每個模型自己的 b16／b32／b48 ＋ nfs（12 號的 NFS 級，m10_tier2.slow_tiers()）
for m in llama qwen; do
  python3 - $OUT/speed_$m/bw.json $MA/bw_$m.json <<'PY'
import json, sys
sys.path.insert(0, "/mlsteam/workspace/paper-hierarchical-kv-state/code")
import m10_tier2 as T
d = json.load(open(sys.argv[1]))
d["targets"]["nfs"] = {"GiBps": T.slow_tiers()["nfs"], "note": "14 §3：12 號的 NFS 級（tier_params.json）"}
json.dump(d, open(sys.argv[2], "w"), indent=1)
PY
done
# 1. DS1 冒煙：每個模型 1 題（T2）、全部粒度 × 挑法 × 佔比
for m in llama qwen; do
  m7run m10-md-smoke-ds1-$m python $REPO/code/m10_drop.py screen --model $m --tasks T2 --n 1 --out $S/ds1_$m.csv > $S/ds1_$m.log 2>&1 \
    || { echo "44e0：DS1 冒煙 $m 失敗（$S/ds1_$m.log）→ 跳過 DS1／DS2" | tee -a $JOBS/NEEDS_HUMAN; tail -20 $S/ds1_$m.log; touch $S/NO_DS; }
done
[ -e $S/NO_DS ] || python3 - $S <<'PY' || { echo "44e0：DS1 冒煙的檢查沒過（$S/ds1_check.txt）→ 跳過 DS1／DS2" | tee -a $JOBS/NEEDS_HUMAN; touch $S/NO_DS; }
import csv, sys, statistics
sys.path.insert(0, "/mlsteam/workspace/paper-hierarchical-kv-state/code")
import m10_drop as DR
S = sys.argv[1]
out, ok = [], True
for m in ("llama", "qwen"):
    rows = list(csv.DictReader(open(f"{S}/ds1_{m}.csv")))
    want = 1 + 2 + sum(len(DR.METHODS[g]) for g in ("T", "P", "B")) * len(DR.FRACS)
    by = {(r["gran"], r["method"], r["frac"]): r for r in rows}
    m0 = [float(r["kl"]) for r in rows if r["method"] in ("MASK0", "MASK0H")]
    big = [float(r["kl"]) for r in rows if r["frac"] in ("0.9",) and r["method"] not in ("E0", "MASK0", "MASK0H")]
    kept = [abs(float(r["kept_frac"]) - (1 - float(r["frac"]))) for r in rows
            if r["gran"] == "T" and r["method"] not in ("PYR", "E0", "MASK0H") and r["method"] != ""]
    t = [float(r["t_score_s"]) for r in rows if r["t_score_s"] not in ("",)]
    c = (len(rows) == want and max(m0) < 0.02 and statistics.median(big) > 0.01 and max(kept) < 0.01)
    ok &= c
    out.append(f"{m}: rows {len(rows)}/{want} MASK KL max {max(m0):.4g} 90% KL 中位數 {statistics.median(big):.4g} "
               f"T 留下比例誤差 max {max(kept):.4g} 分數秒數 {sorted(set(t))[:8]} → {'過' if c else '沒過'}")
open(f"{S}/ds1_check.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
sys.exit(0 if ok else 1)
PY
# 2. DS2 冒煙：m10_quality run，1 題、幾個 DS 配置（兩輪、生成）
[ -e $S/NO_DS ] || for m in llama qwen; do
  m7run m10-md-smoke-ds2-$m python $REPO/code/m10_quality.py run --model $m --params-file $P \
    --configs E0 DS-B-NONE-0 DS-T-NONE-0 DS-T-SNAP-25 DS-B-QRY-50 DS-T-ADA-50 DS-B-ORC-25 DS-T-PYR-50 --modes A --seed0 70000 \
    --tasks T2 --n 1 --out $S/ds2_$m.csv > $S/ds2_$m.log 2>&1 \
    || { echo "44e0：DS2 冒煙 $m 失敗（$S/ds2_$m.log）→ 跳過 DS2" | tee -a $JOBS/NEEDS_HUMAN; tail -20 $S/ds2_$m.log; touch $S/NO_DS; }
done
# 3. 速度冒煙（Qwen，1 launch × 1 rep）：D3 的 drop 配置＋KV 檢查＋nfs 層；M1 單位 128；M2 新格式＋融合 X kernel＋nfs
m=qwen; G=$(fmt_of $m fp8)
E=$(python3 $REPO/code/m10_restore2.py env P0 --model $m --fmt int4_kivi)
m7run m10-md-smoke-d3 env $E bash -c "python $REPO/code/m10_speed.py grid --model $m --fmt int4_kivi --base-fmt $G --out-dir $S/d3 --bw-json $MA/bw_$m.json --bw-names cpu b32 nfs --configs E1s E1s-DB25 E0-DF50 E0-DR25 --refs --lengths --extra-modes loadall --extra-tiers cpu --launches 1 --reps 1 --launch0 800 --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" > $S/d3.log 2>&1 \
  || { echo "44e0：D3 速度冒煙失敗（$S/d3.log）→ 跳過 D3" | tee -a $JOBS/NEEDS_HUMAN; touch $S/NO_D3; }
[ -e $S/NO_D3 ] || python3 - $S/d3/restores.csv <<'PY' || { echo "44e0：D3 冒煙檢查沒過（KV 或載入的位元組）→ 跳過 D3" | tee -a $JOBS/NEEDS_HUMAN; touch $S/NO_D3; }
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
bad = [r for r in rows if r["kv_bad"] not in ("", "0")]
by = {(r["config"], r["bw_name"], r["mode"], r["gpu_state"]): float(r["bytes_read"]) for r in rows}
less = by.get(("E1s-DB25", "cpu", "loadall", "idle"), 1e18) < by.get(("E1s", "cpu", "loadall", "idle"), 0)
print(f"rows {len(rows)} kv_bad {len(bad)} 丟 25% 讀的位元組比較少：{less}")
sys.exit(0 if rows and not bad and less else 1)
PY
E1=$(python3 $REPO/code/m10_restore2.py env P0 --model $m --fmt int4_kivi)
m7run m10-md-smoke-m1 env $E1 M10_UNIT=128 bash -c "python $REPO/code/m10_speed.py grid --model $m --fmt int4_kivi --base-fmt $G --out-dir $S/m1 --bw-json $MA/bw_$m.json --bw-names cpu b32 --configs E0 E1s --refs --lengths --launches 1 --reps 1 --launch0 600 --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" > $S/m1.log 2>&1 \
  || { echo "44e0：M1 速度冒煙失敗（$S/m1.log）→ 跳過 M1" | tee -a $JOBS/NEEDS_HUMAN; touch $S/NO_M1; }
E2=$(python3 $REPO/code/m10_restore2.py env P0 --model $m --fmt kv-int4_kivi-int2_kivi)
m7run m10-md-smoke-m2 env $E2 M10_CODEC_X=1 M10_DEQUANT_FAST_X=1 bash -c "python $REPO/code/m10_speed.py grid --model $m --fmt kv-int4_kivi-int2_kivi --base-fmt $G --out-dir $S/m2 --bw-json $MA/bw_$m.json --bw-names nfs --configs E1s --refs E1s --lengths --launches 1 --reps 1 --launch0 700 --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" > $S/m2.log 2>&1 \
  || { echo "44e0：M2 速度冒煙失敗（$S/m2.log）→ 跳過 M2" | tee -a $JOBS/NEEDS_HUMAN; touch $S/NO_M2; }
for k in m1 m2; do
  [ -e $S/NO_${k^^} ] || python3 -c "
import csv,sys;r=list(csv.DictReader(open('$S/$k/restores.csv')));b=[x for x in r if x['kv_bad'] not in ('','0')];print('$k rows',len(r),'kv_bad',len(b));sys.exit(0 if r and not b else 1)" \
    || { echo "44e0：$k 冒煙 KV 檢查沒過 → 跳過 $k" | tee -a $JOBS/NEEDS_HUMAN; touch $S/NO_${k^^}; }
done
ls $S
exit 0
