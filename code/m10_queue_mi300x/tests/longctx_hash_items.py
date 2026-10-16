# 主線 32K 題目的指紋（改 m10_tasks 前後各跑一次，必須逐字相同）
import hashlib, json, os, sys
sys.path.insert(0, "/mlsteam/workspace/paper-hierarchical-kv-state/code")
import m10_tasks as T
from m10_quality import plan_items
P = json.load(open("/mlsteam/workspace/paper-hierarchical-kv-state/results/m10_pilot_mi300x/e0_tune/CANDIDATE_PARAMS.json"))["params"]
out = {}
for m in ("llama", "qwen"):
    tok = T.load_tokenizer(T.MODELS[m])
    for task, b1, b2, seed in plan_items(["T2", "T3", "T4", "T5"], 3, 10000) + plan_items(["T2", "T4"], 2, 40000):
        it = T.make_item(task, b1, b2, seed, tok, P[m].get(task))
        out[f"{m}|{it['item_id']}"] = hashlib.sha256(json.dumps(it, sort_keys=True).encode()).hexdigest()[:16]
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(len(out), "items")
