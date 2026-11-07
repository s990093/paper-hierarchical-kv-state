"""icpe-rux v6 的正式對拍：Python 參考（x1 sweep，預設引擎）對 Rust v6（sweep_v6.py --engine rust）。
  python -I cmp_v6.py csv <ref.csv[.gz|.xz]> <new.csv>         逐格比 x1_sweep.csv
  python -I cmp_v6.py npz <ref_req_dir> <new_req_dir>          逐檔、逐陣列比逐請求 npz（allow_pickle=False，NaN 視為相同）
CSV：去掉 run_id、ts、elapsed_s、code_sha、host；鍵＝下面 KEY 裡兩邊都有的欄；要求兩邊欄位集合相同、鍵唯一、鍵集合相同、
     每一個非鍵欄逐格字串相同（同 next_sim_g1.py／cmp_noise.py 的比法）。不同 → 印前 50 個差異、exit 1。
NPZ：檔名集合相同；每個檔的陣列名集合、dtype、shape、內容（浮點 NaN 視為相同、其他逐值）都相同。不同 → 印出、exit 1。"""
import csv, gzip, lzma, os, sys
import numpy as np

DROP = ("run_id", "ts", "elapsed_s", "code_sha", "host")
KEY = ("trace", "rule", "cpu_GiB", "policy", "slow", "gpu", "restore", "gpu_GiB", "gpu_write", "gpu_policy", "pf_target",
       "noise_model", "noise_lambda", "noise_draw", "noise_seed", "slow_cap_frac", "slow_policy")


def opener(p):
    return gzip.open if p.endswith(".gz") else lzma.open if p.endswith(".xz") else open


def load(p):
    with opener(p)(p, "rt", newline="") as fh:
        rd = csv.DictReader(fh)
        return [c for c in rd.fieldnames if c not in DROP], list(rd)


def cmp_csv(ref, new):
    rc, rr = load(ref)
    nc, nr = load(new)
    assert set(rc) == set(nc), f"欄位不同：只在 ref {sorted(set(rc) - set(nc))}；只在 new {sorted(set(nc) - set(rc))}"
    key = [k for k in KEY if k in rc]
    vals = [c for c in rc if c not in key]

    def idx(rows, nm):
        d = {}
        for r in rows:
            k = tuple(r[c] for c in key)
            assert k not in d, f"{nm} 鍵重複 {k}"
            d[k] = r
        return d
    R, N = idx(rr, "ref"), idx(nr, "new")
    only_r, only_n = set(R) - set(N), set(N) - set(R)
    diff = [(k, c, R[k][c], N[k][c]) for k in R if k in N for c in vals if R[k][c] != N[k][c]]
    print(f"key_cols={len(key)} {key}")
    print(f"rows ref={len(rr)} new={len(nr)} only_ref={len(only_r)} only_new={len(only_n)}")
    print(f"cols_compared={len(vals)} cells compared={len(R.keys() & N.keys()) * len(vals)} differing={len(diff)}")
    for k in list(only_r)[:10]:
        print("  ONLY_REF", k)
    for k in list(only_n)[:10]:
        print("  ONLY_NEW", k)
    for d in diff[:50]:
        print("  DIFF", d)
    ok = not diff and not only_r and not only_n
    print("CSV_RESULT " + ("IDENTICAL" if ok else "DIFFERENT"))
    return ok


def cmp_npz(rd, nd):
    rf = sorted(f for f in os.listdir(rd) if f.endswith(".npz"))
    nf = sorted(f for f in os.listdir(nd) if f.endswith(".npz"))
    bad = []
    if rf != nf:
        bad.append(("file_set", sorted(set(rf) - set(nf))[:10], sorted(set(nf) - set(rf))[:10]))
    narr = 0
    for f in sorted(set(rf) & set(nf)):
        with np.load(os.path.join(rd, f), allow_pickle=False) as A, np.load(os.path.join(nd, f), allow_pickle=False) as B:
            if sorted(A.files) != sorted(B.files):
                bad.append((f, "keys", sorted(A.files), sorted(B.files)))
                continue
            for k in A.files:
                x, y = A[k], B[k]
                narr += 1
                if x.dtype != y.dtype or x.shape != y.shape:
                    bad.append((f, k, "dtype/shape", str(x.dtype), x.shape, str(y.dtype), y.shape))
                elif x.dtype.kind == "f":
                    if not np.array_equal(x, y, equal_nan=True):
                        i = int(np.flatnonzero(~((x == y) | (np.isnan(x) & np.isnan(y))))[0])
                        bad.append((f, k, "values", i, repr(x[i]), repr(y[i])))
                elif not np.array_equal(x, y):
                    i = int(np.flatnonzero(x != y)[0])
                    bad.append((f, k, "values", i, repr(x[i]), repr(y[i])))
    print(f"files ref={len(rf)} new={len(nf)} arrays_compared={narr} differing={len(bad)}")
    for b in bad[:50]:
        print("  DIFF", b)
    ok = not bad
    print("NPZ_RESULT " + ("IDENTICAL" if ok else "DIFFERENT"))
    return ok


if __name__ == "__main__":
    mode, a, b = sys.argv[1:4]
    ok = cmp_csv(a, b) if mode == "csv" else cmp_npz(a, b)
    sys.exit(0 if ok else 1)
