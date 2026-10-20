# GPU-NEED: none
# AFTER: 33a_c1a 33b_c1a
# C1a 的 report（cpu 佇列：nice 19、0–3 號核心）。只用 numpy（以題目為單位 bootstrap 10,000 次），不 import torch、不開 /dev/kfd。
# 輸出 results/m16_stage3_mi300x/c1a_predictive/{c1a_summary.json, tables_c1a.md}；L1 G2／G4 的成分照算，判讀由人（howto §7）。
# report 之後再核對（只用標準函式庫）：每個模型 T2–T4 共 135 題、T5 15 題都齊全；*.checks.jsonl 裡沒有任何一題的檢查是 false。
# 不齊或有 false → NEEDS_HUMAN、exit 1（report 檔已寫出，供人看；規則 2）。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
C=$RES/c1a_predictive
export OMP_NUM_THREADS=4
m7run m16-c1a-report python code/m16_c1a_predict.py report --raw "$C/c1a_*_s*.csv" --out-dir $C \
  || { nh "33z_c1a_report: report 失敗（$(lastrun m16-c1a-report)）"; exit 1; }
python3 -I - $C <<'PY' || { nh "33z_c1a_report: 題數不齊或逐題檢查有 false（看 $C/c1a_summary.json、$C/*.checks.jsonl）"; exit 1; }
import glob, json, os, sys
C = sys.argv[1]
s = json.load(open(os.path.join(C, "c1a_summary.json")))
bad = []
for m in ("llama", "qwen"):
    for g, n in (("T2-T4", 135), ("T5", 15)):
        got = s["models"].get(m, {}).get(g, {}).get("n_items")
        if got != n:
            bad.append(f"{m}/{g} {got} 題（要 {n}）")
if s["incomplete_items"]:
    bad.append(f"不齊全的題 {len(s['incomplete_items'])}：{s['incomplete_items'][:5]}")
fails = {}
for p in sorted(glob.glob(os.path.join(C, "c1a_*_s*.checks.jsonl"))):
    for line in open(p):
        try:
            d = json.loads(line)
        except ValueError:                      # 被砍的殘行
            continue
        f = [k for k, v in d.get("checks", {}).items() if v is False]
        if f:
            fails[(d["model"], d["item_id"])] = f
if fails:
    bad.append(f"逐題檢查有 false 的題 {len(fails)}：{list(fails.items())[:5]}")
print("run_id：", s["run_ids"])
print("污染（品質規則：分數照用、t_score_s 作廢）：", json.dumps(s["contention"], ensure_ascii=False))
print("核對：" + ("；".join(bad) if bad else "每個模型 150 題齊全、逐題檢查全過"))
sys.exit(1 if bad else 0)
PY
