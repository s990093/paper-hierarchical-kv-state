"""把 acvlab runs/ 底下的實驗輸出彙總成 results/ 的小 CSV（每列帶 run_id、ts、host）。只收有授權的紀錄（不收 CacheWise）。"""
import json, glob, os, csv, sys
RUNS = "/ssd6/hungwei/icpe-next/runs"
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
BAN = ("cachewise", "cw_")


def ok(name):
    return not any(b in name.lower() for b in BAN)


def write(name, rows):
    rows = [r for r in rows if ok(json.dumps(r))]
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(name, len(rows))


# 1. 直譯器（SemiAnalysis）
rows = []
for f in sorted(glob.glob(f"{RUNS}/*-icpe-ana-bench-sa-*/bench.jsonl")):
    rows += [json.loads(l) for l in open(f)]
write("interp_bench_sa.csv", rows)
# 2. Rust 對拍（逐 job）
rows = []
for d in sorted(glob.glob(f"{RUNS}/*-icpe-ana-diff*")):
    if not ok(d) or not os.path.exists(f"{d}/diff.jsonl"):
        continue
    ver = ("v4_noise_on_0b0ef1e" if "-diff-v4s-" in d else "v4_noise_on_v3" if "-diff-v4-" in d else
           "v3_prefix_fork" if "-diff-fork-" in d else "v2_nextreq_cache" if "-diff-cache2-" in d else
           "v1_port" if "-diff-cache-" not in d else None)
    if ver is None:
        continue                       # 第一次 cache 對拍：比對腳本解析錯誤、沒有比到（見 RUST_NOTES 踩過的坑）
    for l in open(f"{d}/diff.jsonl"):
        r = json.loads(l); r["engine"] = ver; rows.append(r)
write("rust_difftest_jobs.csv", rows)
# 3. 編譯設定／執行緒
rows = []
for f in sorted(glob.glob(f"{RUNS}/*-icpe-ana-variants-sa/variants.jsonl")):
    rows += [json.loads(l) for l in open(f)]
write("rust_variants_threads_sa.csv", rows)
# 4. 整個 block（照正式切法）
rows = []
for f in sorted(glob.glob(f"{RUNS}/*-icpe-ana-grid-sa_N16_s1/grid.jsonl")):
    rows += [json.loads(l) for l in open(f)]
write("sa_block_grid.csv", rows)
# 5. 低保真度版本的花費
rows = []
for d in sorted(glob.glob(f"{RUNS}/*-icpe-ana-lowfi-sa_N16_s1")):
    for l in open(f"{d}/stdout.log"):
        if l.startswith("{"):
            r = json.loads(l); r.update(run_id=os.path.basename(d), host="acvlab", trace="nb:semianalysis_N16_s1"); rows.append(r)
write("lowfi_cost_sa.csv", rows)
# 6. 近似方法的誤差
rows = []
for d in sorted(glob.glob(f"{RUNS}/*-icpe-ana-approx-*")):
    p = f"{d}/out/approx_eval.csv"
    if os.path.exists(p):
        rows += list(csv.DictReader(open(p)))
write("approx_eval_sa.csv", rows)
# 7. 瓶頸分析（修改前／修改後的分段比例）
rows = []
for d in sorted(glob.glob(f"{RUNS}/*-icpe-ana-bottleneck-sa")):
    p = f"{d}/bottleneck.jsonl"
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l); sh = r.pop("share")
            r.update({f"share_{k}": v for k, v in sh.items()}); rows.append(r)
write("bottleneck_sa.csv", rows)
