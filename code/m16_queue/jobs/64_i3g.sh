# GPU-NEED: both
# GPU: 1
# NEED-GB: 180
# MAX-H: 2.5
# I3g（10/22 使用者：「全跑（含 NFS）」）：agent 工具暫停時 KV 的去留，vLLM 0.28 端到端，驗 i3_pause_sim.md §7 的「翻轉」前提。
# 設計與事先寫死的判定：experiment_log.md「I3g（64_i3g）」。Llama、卡 1、四個 server 依序：a APC、b APC＋CPU、c ＋檔案層（本地 /var/tmp）、
# d ＋檔案層（NFS /mlsteam/data）；全部 TRITON_ATTN、HBM 6,020 個 16-token block（參考工作集的 0.25 倍）、CPU 卸載 33 GiB。
# 工作負載 /mlsteam/data/tiara/stage3/workloads/i3g/i3g_w.jsonl（sha256 59f397fe…；64 session、約 14 分鐘）。
# GPU 時間〔估〕：每個設定約 15–20 分鐘（server 啟動 2–3 分＋工作負載約 14 分）→ 約 1.2–1.5 小時。
# 檔案層是自己的快取：每個設定跑完記下大小（fs_bytes_<k>.txt）再刪掉，NFS 那組最多約 100 GiB。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
JOBNAME=64_i3g
MODEL=llama
source $REPO/code/m16_queue/jobs/vllm_lib.sh
OUT=$RES/i3g_agentic; mkdir -p $OUT
WF=/mlsteam/data/tiara/stage3/workloads/i3g/i3g_w.jsonl
SHA=$($PY -c "import json;print(json.loads(open('$WF').readline())['_meta']['sha256'])") || { nh "$JOBNAME：讀不到工作負載 $WF"; exit 1; }
[ "$SHA" = 59f397fe741adb44c018f3fee9cca1e928a9ba53cf1d5245ed3ec07d45e29100 ] || { nh "$JOBNAME：工作負載的 sha256 不是事先登記的那份（$SHA）"; exit 1; }
FS_LOCAL=/var/tmp/m16_i3g_fs
FS_NFS=/mlsteam/data/tiara/stage3/i3g_fs
for k in a b c d; do
  case $k in
    a) OFF=none; FS="" ;;
    b) OFF=cpu:33; FS="" ;;
    c) OFF=tier:33:$FS_LOCAL:150; FS=$FS_LOCAL ;;
    d) OFF=tier:33:$FS_NFS:150; FS=$FS_NFS ;;
  esac
  if [ -n "$FS" ]; then rm -rf "$FS"; mkdir -p "$FS"; fi
  srv_start i3g-$k --apc on --offload $OFF --gpu-blocks 6020 --block-size 16 --attention-backend TRITON_ATTN --max-model-len 26624 || exit 1
  bench_run i3g-$k run --workload $WF --out-dir $OUT --label i3g_$k --timeout-s 2400 || { srv_stop; exit 1; }
  srv_stop
  if [ -n "$FS" ]; then
    echo "$k $(du -sb "$FS" | cut -f1) bytes $(find "$FS" -type f | wc -l) files $(date -u +%FT%TZ) $FS" > $OUT/fs_bytes_$k.txt
    rm -rf "$FS"
  fi
done
m7run m16-i3g-report python -I code/m16_i3g_report.py --in $OUT || { nh "$JOBNAME：報告失敗（$(lastrun m16-i3g-report)）"; exit 1; }
echo "$JOBNAME 結束 $(date -u +%FT%TZ)"
