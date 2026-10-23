# GPU-NEED: both
# GPU: 1
# NEED-GB: 180
# MAX-H: 0.95

# I3g 換 Qwen（10/23 00:5xZ 使用者選；截止延到 03:30Z 也是使用者同意的）：同一份工作負載（隨機 token ID 1000–120000，在 Qwen 詞表裡都是一般 token）、
# 同一組 serve 參數，**每一層的 token 容量和 Llama 那次相同**：HBM 6,020 個 16-token block；CPU 與檔案層上限照 KV 位元組比例縮
# （Qwen 57,344 B／token、Llama 131,072 B／token → ×0.4375：CPU 33→14.4 GiB、檔案層 150→66 GiB）。拆兩段（a/b、c/d），每段約 35 分鐘。
# 判定寫在 experiment_log.md「I3g 換 Qwen（66）」（排進佇列前寫）。輸出 i3g_agentic/qwen/。硬性截止：開一個設定前，若 now＋22 分 > 03:30Z 就不開（記在 skipped.txt）。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
JOBNAME=66a_i3g_qwen_ab
MODEL=qwen
source $REPO/code/m16_queue/jobs/vllm_lib.sh
OUT=$RES/i3g_agentic/qwen; mkdir -p $OUT
WF=/mlsteam/data/tiara/stage3/workloads/i3g/i3g_w.jsonl
SHA=$($PY -c "import json;print(json.loads(open('$WF').readline())['_meta']['sha256'])") || { nh "$JOBNAME：讀不到工作負載 $WF"; exit 1; }
[ "$SHA" = 59f397fe741adb44c018f3fee9cca1e928a9ba53cf1d5245ed3ec07d45e29100 ] || { nh "$JOBNAME：工作負載的 sha256 不是事先登記的那份（$SHA）"; exit 1; }
DEADLINE=$(date -u -d 2026-10-23T03:30:00Z +%s)
FS_LOCAL=/var/tmp/m16_i3g_fs_qwen
FS_NFS=/mlsteam/data/tiara/stage3/i3g_fs_qwen
for k in a b; do
  if [ $(($(date +%s) + 1320)) -gt $DEADLINE ]; then echo "$k skipped $(date -u +%FT%TZ)（now＋22 分 > 03:30Z）" | tee -a $OUT/skipped.txt; continue; fi
  case $k in
    a) OFF=none; FS="" ;;
    b) OFF=cpu:14.4; FS="" ;;
    c) OFF=tier:14.4:$FS_LOCAL:66; FS=$FS_LOCAL ;;
    d) OFF=tier:14.4:$FS_NFS:66; FS=$FS_NFS ;;
  esac
  if [ -n "$FS" ]; then rm -rf "$FS"; mkdir -p "$FS"; fi
  srv_start i3gq-$k --apc on --offload $OFF --gpu-blocks 6020 --block-size 16 --attention-backend TRITON_ATTN --max-model-len 26624 || exit 1
  bench_run i3gq-$k run --workload $WF --out-dir $OUT --label i3g_$k --timeout-s 2400 || { srv_stop; [ -n "$FS" ] && rm -rf "$FS"; exit 1; }
  srv_stop
  if [ -n "$FS" ]; then
    echo "$k $(du -sb "$FS" | cut -f1) bytes $(find "$FS" -type f | wc -l) files $(date -u +%FT%TZ) $FS" > $OUT/fs_bytes_$k.txt
    rm -rf "$FS"
  fi
done
echo "$JOBNAME 結束 $(date -u +%FT%TZ)"
