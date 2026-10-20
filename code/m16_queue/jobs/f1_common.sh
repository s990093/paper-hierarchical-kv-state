# F1 前綴復用地圖的共用本體（由 51_*/52_* 的 job source；job 先設 L、CFG、CAP）。設計：docs/stage3_20261019/v0_f1_i3_howto.md §3。
# 事先寫死的規則（10/19 14:2xZ，看到任何 F1 數字之前）：
#   - 每個長度的第一個 job（CFG＝off、CAP＝ample）量：t1＝單請求冷 prefill 的 TTFT（n＝1，burst，重疊 0）；
#     λ_sat＝n_req ÷ makespan（burst、重疊 0、不復用、容量充足）。寫進 calib_<L>.json，同長度其他 job 都讀這份。
#   - Poisson 到達率：0.5×λ_sat 與 0.9×λ_sat；TTFT SLO：1.5×t1 與 5×t1。
#   - 容量：ample＝vLLM 預設；c075／c040＝ceil(0.75／0.40 × 工作集)，工作集＝同長度、重疊 0% 那格的不同 16-token block 數（含輸出）。
#   - 三個設定都用 TRITON_ATTN（卸載的 connector 會讓 attention 退回 TRITON_ATTN；統一 kernel 才能只比「復用」）。
#   - 重疊先只跑 {0, 0.5, 1.0}；有訊號再補 0.25／0.75。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
JOBNAME=f1_${L}_${CFG}_${CAP}
source $REPO/code/m16_queue/jobs/vllm_lib.sh
OUT=$RES/f1_prefix_reuse; mkdir -p $OUT
WD=/mlsteam/data/tiara/stage3/workloads/f1; mkdir -p $WD
CAL=$OUT/calib_${L}.json
NREQ=$([ $L = 8192 ] && echo 64 || echo 32)
W0=$WD/w_${L}_0_burst.jsonl
[ -f $W0 ] || $W gen-prefix --n-req $NREQ --prompt-len $L --overlap 0 --n-groups 4 --out-len 128 --arrival burst --seed 0 --out $W0 \
  || { nh "$JOBNAME：gen-prefix 失敗"; exit 1; }
WS=$($PY -c "import json;print(json.loads(open('$W0').readline())['_meta']['workset']['ws_blocks_incl_output'])") \
  || { nh "$JOBNAME：讀不到工作集"; exit 1; }
case $CAP in
  ample) CAPARG="--block-size 16" ;;
  c075)  CAPARG="--block-size 16 --gpu-blocks $($PY -c "import math;print(math.ceil(0.75*$WS))")" ;;
  c040)  CAPARG="--block-size 16 --gpu-blocks $($PY -c "import math;print(math.ceil(0.40*$WS))")" ;;
esac
case $CFG in
  off)    CFGARG="--apc off" ;;
  apc)    CFGARG="--apc on" ;;
  apccpu) CFGARG="--apc on --offload cpu:64" ;;
esac
srv_start f1-$L-$CFG-$CAP $CFGARG $CAPARG --attention-backend TRITON_ATTN --max-model-len 34816 || exit 1
summ() { cat $TIARA_RUNS/$(lastrun "m16-$1")/summary.json; }
if [ $CFG = off ] && [ $CAP = ample ]; then
  W1=$WD/w_${L}_single.jsonl
  [ -f $W1 ] || $W gen-prefix --n-req 1 --prompt-len $L --overlap 0 --n-groups 1 --out-len 128 --arrival burst --seed 1 --out $W1
  for k in 1 2 3; do bench_run f1-$L-t1-$k run --workload $W1 --out-dir $OUT --label f1_${L}_t1_$k || exit 1; done
  bench_run f1-$L-sat run --workload $W0 --out-dir $OUT --label f1_${L}_sat || exit 1
  $PY - <<PY || { nh "$JOBNAME：寫 calib 失敗"; exit 1; }
import json, glob, os, statistics
runs = os.environ["TIARA_RUNS"]
def last(short):
    ds = sorted(glob.glob(f"{runs}/*-m16-{short}"))
    return ds[-1]
t1s = [json.load(open(last(f"f1-$L-t1-{k}") + "/summary.json"))["ttft_s"]["p50"] for k in (1, 2, 3)]
sat = json.load(open(last("f1-$L-sat") + "/summary.json"))
lam = sat["throughput"]["req_per_s"]
t1 = statistics.median(t1s)
cal = dict(L=$L, t1_runs=t1s, t1=t1, lambda_sat=lam, rates=[round(0.5 * lam, 4), round(0.9 * lam, 4)],
           slo_ttft=[round(1.5 * t1, 4), round(5 * t1, 4)], sat_run=os.path.basename(last("f1-$L-sat")),
           rule="Poisson 0.5/0.9×λ_sat；SLO 1.5/5×t1（f1_common.sh 檔頭，事先寫死）")
json.dump(cal, open("$CAL", "w"), indent=1, ensure_ascii=False)
print(cal)
PY
fi
[ -f $CAL ] || { nh "$JOBNAME：沒有 $CAL（同長度的 off_ample 要先跑）"; exit 1; }
read R1 R2 S1 S2 < <($PY -c "import json;c=json.load(open('$CAL'));print(*c['rates'],*c['slo_ttft'])")
for ov in 0 0.5 1.0; do
  for arr in burst p1 p2; do
    case $arr in burst) A="--arrival burst"; RT=0 ;; p1) A="--arrival poisson --rate $R1"; RT=$R1 ;; p2) A="--arrival poisson --rate $R2"; RT=$R2 ;; esac
    WF=$WD/w_${L}_${ov}_${arr}_${RT}.jsonl
    [ -f $WF ] || $W gen-prefix --n-req $NREQ --prompt-len $L --overlap $ov --n-groups 4 --out-len 128 $A --seed 0 --out $WF \
      || { nh "$JOBNAME：gen-prefix 失敗（$ov $arr）"; exit 1; }
    bench_run f1-$L-$CFG-$CAP-$ov-$arr run --workload $WF --out-dir $OUT --slo-ttft $S1 $S2 \
      --label f1_${L}_${CFG}_${CAP}_${ov}_${arr} || exit 1
  done
done
srv_stop
echo "$JOBNAME 結束 $(date -u +%FT%TZ)"
