# GPU-NEED: one
# CARDS: 0,1
# NEED-GB: 45
# MAX-H: 0.5
# C1a 的冒煙（docs/stage3_20261019/c1a_s2_howto.md §3.1、§3.2）：每個模型先 selfcheck（同一題走 m10_drop screen 的路徑和本程式的路徑，
# 逐值比對），過了再 run --smoke（1 題 T2_front-back_s70000，印每個分數前 5 名 block 與答案 block）。
# 兩張卡並行：Llama 在卡 0、Qwen 在卡 1（和 33a／33b 同一個擺法，順便看另一張卡有行程時 guard 記的 own_gpu_intruders）。
# 每張卡同時只有 1 個行程 → NEED-GB 45〔估：09 海選 Llama 每個行程峰值 28.3 GiB（run 20261016-130624-m10-sweep-pilot-llama-s0）；
#   selfcheck 多留一份 64 個 BF16 block ≈ 4 GiB〕。
# 輸出到 $SMOKE/c1a_<UTC 時間>_<PID>/，每次開新目錄：run 會跳過 CSV 裡已寫完的題，沿用舊檔的話冒煙會什麼都沒算就「過」。
# 任何一步不過 → NEEDS_HUMAN、exit 1（規則 2）。selfcheck 的 prefill_bitwise 不過可能只是 GPU 不決定性，看 JSON 的 maxabs 由人判。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
SM=$SMOKE/c1a_$(date -u +%Y%m%dT%H%M%SZ)_$$; mkdir -p $SM          # 加 PID：同一秒重下也不會接到舊檔（假替身測試抓到）
one() {   # one <模型>：selfcheck → smoke → 結構核對（只用標準函式庫讀 CSV）
  local m=$1
  m7run m16-c1a-selfcheck-$m python code/m16_c1a_predict.py selfcheck --model $m --out $SM/c1a_selfcheck_$m.json \
    || { nh "04_smoke_c1a: selfcheck $m 沒過（$(lastrun m16-c1a-selfcheck-$m)；看 $SM/c1a_selfcheck_$m.json 的 pass、maxabs）"; return 1; }
  m7run m16-c1a-smoke-$m python code/m16_c1a_predict.py run --model $m --smoke --out $SM/c1a_smoke_$m.csv \
    || { nh "04_smoke_c1a: smoke $m 失敗（$(lastrun m16-c1a-smoke-$m)）"; return 1; }
  # 結構核對：寫死的、和數值無關的事（STREAM／NEW 的順序是定義；答案 block 要在題目指定的位置帶：front＝1–16、back＝48–63）。
  # QRY1 的 recall_t1@16、QRY2 的 recall@16「應該很高」只印出來給人看，不設門檻。
  python3 -I - $SM/c1a_smoke_$m.csv <<'PY' || { nh "04_smoke_c1a: smoke $m 的結構核對沒過（$SM/c1a_smoke_$m.csv）"; return 1; }
import csv, json, sys
rows = list(csv.DictReader(open(sys.argv[1])))
by = {r["method"]: r for r in rows}
err = []
want_m = ["SNAP", "EXPATT", "KNORM", "KEYDIFF", "H2O", "STREAM", "NEW", "RND0", "RND1", "RND2",
          "QRY1", "QRY2", "QRY2_ALT", "QUEST_MEAN", "QUEST_MAX", "FULLK_MEAN", "FULLK_MAX"]
if sorted(by) != sorted(want_m) or len(rows) != len(want_m):
    err.append(f"列：{len(rows)}、方法 {sorted(by)}（要 {len(want_m)} 列、{sorted(want_m)}）")
if {r["item_id"] for r in rows} != {"T2_front-back_s70000"}:
    err.append(f"item_id {sorted({r['item_id'] for r in rows})}")
if any(r["n_rows_item"] != str(len(want_m)) for r in rows):
    err.append("n_rows_item 不對")
for m, w in (("STREAM", "63 62 61 60 59"), ("NEW", "1 2 3 4 5")):
    if by.get(m, {}).get("top5") != w:
        err.append(f"{m} 前 5＝{by.get(m, {}).get('top5')!r}（要 {w}）")
for r in rows:
    if sorted(int(x) for x in r["order"].split()) != list(range(1, 64)):
        err.append(f"{r['method']} 的 order 不是 1..63 的排列")
r0 = rows[0] if rows else {}
n1, n2 = json.loads(r0.get("needle_t1", "[]")), json.loads(r0.get("needle_tgt", "[]"))
if not n1 or not all(1 <= b <= 16 for b in n1) or not n2 or not all(48 <= b <= 63 for b in n2):
    err.append(f"答案 block：第 1 輪 {n1}（要在 1–16）、第 2 輪 {n2}（要在 48–63）")
print("給人看：QRY1 recall_t1@16＝%s、QRY2 recall@16＝%s、QRY2_ALT ρ＝%s、own_gpu_intruders＝%s、host_level＝%s" % (
    by.get("QRY1", {}).get("recall_t1_16"), by.get("QRY2", {}).get("recall16"), by.get("QRY2_ALT", {}).get("spearman_vs_target"),
    sorted({r["own_gpu_intruders"] for r in rows}), sorted({r["host_level"] for r in rows})))
print("結構核對：" + ("；".join(err) if err else "OK"))
sys.exit(1 if err else 0)
PY
  grep -h "peak" $TIARA_RUNS/$(lastrun m16-c1a-smoke-$m)/stdout.log | tail -1 || true   # 給人看：每題秒數與峰值 GiB（33a／33b 一張卡開幾個 shard 靠這個）
  return 0
}
oncard 0 one llama & p0=$!
oncard 1 one qwen & p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
[ $r0 = 0 ] && [ $r1 = 0 ] || exit 1
echo "04_smoke_c1a 全過 → $SM $(date -u +%FT%TZ)"
