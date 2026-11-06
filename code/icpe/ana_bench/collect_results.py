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
    if not ok(d) or not os.path.exists(f"{d}/diff.jsonl") or "-diff-v5reg-" in d:   # v5reg：只跑 Rust（--no-ref），另收在下面
        continue
    ver = ("v5_noise_b2" if "-diff-v5-" in d else "v4_noise_on_0b0ef1e" if "-diff-v4s-" in d else "v4_noise_on_v3" if "-diff-v4-" in d else
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
# CPython 3.12 set 走訪順序 vs Rust PySet（v5 的 B2 用）
rows = []
for f in sorted(glob.glob(f"{RUNS}/*-icpe-ana-pyset-fuzz/summary.jsonl")):
    rows += [json.loads(l) for l in open(f)]
write("pyset_order_fuzz.csv", rows)
# v5 回歸：只跑 Rust，逐 job 和先前已對過 Python 的 run 的 sha_ref 比
REG = {"noise-nb_bailian_coder": "v4s-noise-nb_bailian_coder", "noise-conversation": "v4s-noise-conversation",
       "broad-nb_bailian_coder": "v4s-broad-nb_bailian_coder", "narrow-conversation": "fork-narrow-conversation"}
rows = []
for d in sorted(glob.glob(f"{RUNS}/*-icpe-ana-diff-v5reg-*")):
    key = d.split("-diff-v5reg-")[1]
    refs = sorted(glob.glob(f"{RUNS}/*-icpe-ana-diff-{REG[key]}"))
    if key not in REG or not refs:
        continue
    ref = {}
    for l in open(f"{refs[-1]}/diff.jsonl"):
        r = json.loads(l); ref[r["job"]] = r
    for l in open(f"{d}/diff.jsonl"):
        r = json.loads(l); o = ref.get(r["job"])
        rows.append(dict(trace=r["trace"], matrix=r["matrix"], job=r["job"], policy=r["policy"], cpu_GiB=r["cpu_GiB"], gpu=r["gpu"],
                         noise=r.get("noise", "none"), sha_rust_v5=r["sha_rust"], sha_ref_python=None if o is None else o["sha_ref"],
                         ref_run_id=os.path.basename(refs[-1]), identical=None if o is None else o["sha_ref"] == r["sha_rust"],
                         ts=r["ts"], host=r["host"], run_id=r["run_id"], engine="v5_noise_b2"))
write("rust_v5_regression.csv", rows)
