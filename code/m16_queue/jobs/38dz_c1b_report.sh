# GPU-NEED: none
# AFTER: 38da_c1b 38db_c1b
# C1b 的 report（cpu 佇列：nice 19、0–3 號核心）。只用 numpy（以題目為單位 bootstrap 10,000 次），不 import torch、不開 /dev/kfd。
# 輸出 results/m16_stage3_mi300x/c1_selective_restore/{c1b_summary.json, tables_c1b.md, c1b_cells.csv}；判定照 c1b_howto.md §3（事先寫死）。
# report 之後再核對（只用標準函式庫）：每個模型 T2–T4 齊全的題 ≥ 48；*.checks.jsonl 沒有 false（含選法和 C1a 逐字相同）。
# 不夠或有 false → NEEDS_HUMAN、exit 1（report 檔已寫出，供人看；規則 2）。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
C=$RES/c1_selective_restore; CR=${C1B_RAW:-/mlsteam/data/tiara/stage3/c1b}
export OMP_NUM_THREADS=4
m7run m16-c1b-report python code/m16_c1b_restore.py report --raw "$CR/c1b_*_s*.csv.gz" --qtime "$C/qtime_*.csv" --out-dir $C \
  || { nh "38dz_c1b_report: report 失敗（$(lastrun m16-c1b-report)）"; exit 1; }
python3 -I - $C <<'PY' || { nh "38dz_c1b_report: 題數不夠或逐題檢查有 false（看 $C/c1b_summary.json 的 checks_false、incomplete_items）"; exit 1; }
import json, os, sys
s = json.load(open(os.path.join(sys.argv[1], "c1b_summary.json")))
bad = []
for m in ("llama", "qwen"):
    g = s["models"].get(m, {}).get("T2-T4", {})
    n = g.get("n_items", 0)
    if n < 48:
        bad.append(f"{m} T2–T4 只有 {n} 題（要 ≥ 48）")
    j = g.get("judge", {})
    print(f"{m}：T2–T4 {n} 題、T5 {s['models'].get(m, {}).get('T5', {}).get('n_items', 0)} 題；"
          f"判定 設計卡（QUEST_MEAN）＝{j.get('verdict_design_card')}、可部署版（QUEST_BLIND）＝{j.get('verdict_deployable')}")
if s["checks_false"]:
    bad.append(f"逐題檢查有 false 的題 {len(s['checks_false'])}：{list(s['checks_false'].items())[:5]}")
print("run_id：", len(s["run_ids"]), "個；qtime：", s["qtime_run_ids"] or "NOT_MEASURED")
print("污染（品質規則：分數照用、t_gen_s 作廢）：", json.dumps(s["contention"], ensure_ascii=False))
print("不齊全的題（沒用）：", len(s["incomplete_items"]))
print("核對：" + ("；".join(bad) if bad else "OK"))
sys.exit(1 if bad else 0)
PY
