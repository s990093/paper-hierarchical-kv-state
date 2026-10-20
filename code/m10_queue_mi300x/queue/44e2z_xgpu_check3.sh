# 10/20 00:1xZ：第三次（RUNLOG 15.17）。44cb（HIP）、44d_xgpu_check2（ROCR）都沒過：行程一律開兩張卡的 render 節點，amd-smi 把 VRAM 兩張都列。
# gpu_guard 改從 KFD sysfs（/sys/class/kfd/kfd/proc/<pid>/vram_<gpu_id>）讀每張卡的 VRAM（分卡準確），用預設的 HIP 設定再驗一次。量時間 → 兩張卡獨佔。
#   1. 卡 0 的假行程（HIP_VISIBLE_DEVICES=0）在 KFD 只有卡 0 的 VRAM；卡 1 的 guard 看不到、卡 0 的看得到。
#   2. 卡 1 量還原（HIP=1），卡 0「空著」(A)／「跑 4 個品質 process（HIP=0）」(B) 交替 A B A B，兩個模型；
#      B／A 中位數比的幾何平均 ≤ 1.02、沒有一格 > 1.08、guard 全 CLEAN。
#   全過 → touch XGPU_OK（量時間時 helper 照跑；USE_ROCR 不用）。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
X=$REPO/results/m10_xgpu_mi300x/kfd; mkdir -p $X; rm -f $JOBS/XGPU_OK
R0="env HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 TIARA_GPU=0"
R1="env HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 TIARA_GPU=1"
# 1. 分卡
$R0 python -c "import torch,time; x=torch.empty(2**31,dtype=torch.uint8,device='cuda'); print(hex(torch.cuda.get_device_properties(0).pci_bus_id), torch.cuda.device_count(), flush=True); time.sleep(150)" > $X/dummy.out 2>&1 &
DUMMY=$!
sleep 45
DPID=$(pgrep -f "torch.empty\(2\*\*31" | head -1)
ls -la /proc/$DPID/fd 2>/dev/null | grep -oE '/dev/(kfd|dri/[a-zA-Z]+[0-9]+)' | sort | uniq -c > $X/dummy_fds.txt
amd-smi process --json > $X/amd_smi_process.json 2>&1
python - $X <<'PY' > $X/attribution.json 2> $X/attribution.err
import json, sys
sys.path.insert(0, "/mlsteam/workspace/paper-hierarchical-kv-state/code")
import gpu_guard as G
X = sys.argv[1]
f1, f0 = G.foreign_on(1), G.foreign_on(0)
fds = open(f"{X}/dummy_fds.txt").read()
kfd = [r for r in G._kfd_compute_apps() if r["used_mib"] > 0]; smi = [r for r in G._amd_smi_compute_apps() if r["used_mib"] > 0]
ok = (f1 == []) and any(x.get("kind") == "same_container" for x in f0)
print(json.dumps(dict(card1_view=f1, card0_view=f0, dummy_fds=fds, kfd_vram_rows=kfd, amd_smi_vram_rows=smi, ok=ok), ensure_ascii=False, indent=1))
PY
kill $DUMMY 2>/dev/null; wait $DUMMY 2>/dev/null
python3 -c "import json,sys; sys.exit(0 if json.load(open('$X/attribution.json'))['ok'] else 1)" \
  || { echo "44e2z_xgpu_check3：KFD 分卡也沒過（$X/attribution.json）→ 量時間照舊兩張卡獨佔" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
sleep 20
# 2. 干擾 A／B（預設的 HIP 設定）
bload_start() {
  BL=""
  for i in 0 1 2 3; do
    $R0 python $REPO/code/m10_quality.py run --model qwen --params-file $P --configs E0 E1@int4_kivi --modes A \
      --seed0 $((99700 + 100 * i)) --tasks T2 --n 60 --out $X/bload_$1_$i.csv > $X/bload_$1_$i.log 2>&1 &
    BL="$BL $!"
  done
  sleep 150
}
bload_stop() { for p in $BL; do kill $p 2>/dev/null; done; for p in $BL; do wait $p 2>/dev/null; done; sleep 30; }
l0=920
for m in qwen llama; do
  G=$(fmt_of $m fp8); E=$(python3 $REPO/code/m10_restore2.py env P0 --model $m --fmt int4_kivi)
  for cond in A1 B1 A2 B2; do
    case $cond in B*) bload_start ${m}_$cond ;; esac
    m7run m10-xgpu3-$m-$cond $R1 $E bash -c "python $REPO/code/m10_speed.py grid --model $m --fmt int4_kivi --base-fmt $G --out-dir $X/${m}_$cond --bw-json $OUT/speed_$m/bw.json --bw-names cpu b32 --configs E0 E1s --refs --lengths --extra-modes loadall --extra-tiers cpu --launches 1 --reps 10 --launch0 $l0 --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" \
      || echo "44e2z_xgpu_check3：干擾實驗 $m $cond 失敗" >> $JOBS/NEEDS_HUMAN
    case $cond in B*) bload_stop ;; esac
    l0=$((l0 + 1))
  done
done
# 3. 判定（同 44cb）
python3 - $X <<'PY'
import glob, json, math, os, statistics, sys
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
res = dict(rule="15.17：B/A 幾何平均 ≤ 1.02、沒有一格 > 1.08、guard 全 CLEAN（KFD 分卡、HIP）", n_cells=len(ratios), geomean_B_over_A=gm,
           max_B_over_A=max(ratios) if ratios else None, min_B_over_A=min(ratios) if ratios else None,
           noise_geomean_A2_over_A1=gmn, guards_ok=guards_ok, pass_=ok, cells=out)
json.dump(res, open(os.path.join(X, "verdict.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in res.items() if k != "cells"}, ensure_ascii=False, indent=1))
sys.exit(0 if ok else 3)
PY
rc=$?
if [ $rc -eq 0 ]; then touch $JOBS/XGPU_OK; echo "44e2z_xgpu_check3：過 → XGPU_OK";
else echo "44e2z_xgpu_check3：干擾實驗沒過（$X/verdict.json）→ 量時間照舊兩張卡獨佔" | tee -a $JOBS/NEEDS_HUMAN; fi
exit 0
