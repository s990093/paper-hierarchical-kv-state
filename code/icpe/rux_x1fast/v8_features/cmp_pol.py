"""同一個 x1_sweep.csv 裡，GPU 規則 A 的列和 B 的列逐格比（鍵＝除了 gpu_policy 以外的所有設定欄；不比 run_id、ts、elapsed_s、code_sha、host）。
用法：python -I cmp_pol.py <x1_sweep.csv> APC LRU [BELP BEL ...]（兩兩一組）。印 POL_RESULT IDENTICAL／DIFFERENT。"""
import csv, sys
f, pairs = sys.argv[1], sys.argv[2:]
R = list(csv.DictReader(open(f)))
KEY = ["trace", "rule", "cpu_GiB", "policy", "slow", "gpu", "restore", "gpu_GiB", "gpu_write", "pf_target", "slow_cap_frac", "slow_policy"]
KEY = [k for k in KEY if k in R[0]] + [k for k in R[0] if k.startswith("noise_")]
SKIP = set(KEY) | {"run_id", "ts", "elapsed_s", "code_sha", "host", "gpu_policy"}
bad = 0
for a, b in zip(pairs[::2], pairs[1::2]):
    A = {tuple(r[k] for k in KEY): r for r in R if r["gpu_policy"] == a}
    B = {tuple(r[k] for k in KEY): r for r in R if r["gpu_policy"] == b}
    assert A and set(A) == set(B), (a, b, len(A), len(B))
    n = d = 0
    for k in A:
        for c in A[k]:
            if c in SKIP:
                continue
            n += 1
            if A[k][c] != B[k][c]:
                d += 1
                if d <= 10:
                    print("  DIFF", a, b, k, c, A[k][c], B[k][c])
    print(f"{a} vs {b}: rows {len(A)} cells {n} differing {d}")
    bad += d
print("POL_RESULT", "IDENTICAL" if bad == 0 else "DIFFERENT")
sys.exit(1 if bad else 0)
