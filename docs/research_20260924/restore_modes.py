# Arithmetic only (NOT a measurement). Constants from main.tex tab:costmodels (qwen-awq, NVMe, ms/block, 16-token blocks).
BLOCK = 16
R0, ALPHA = 3.546, 0.000178            # recompute block at token position p: R0 + ALPHA*p
PROFILES = {
    "vLLM fs tier (measured 10.245 ms/blk)": 10.245,
    # efficient path: 896 KiB/blk at the raw NVMe sequential read measured on the same disk (2,085 MiB/s)
    "raw-NVMe path (896KiB / 2085MiB/s)": 16 * 57344 / (2085 * 2**20) * 1000,
}
def R(k):  # k = block index
    return R0 + ALPHA * k * BLOCK
def cake(N, L):
    # choose M (blocks recomputed from the front) minimising max(sum_{k<M} R(k), (N-M)*L)
    best, bestM, acc = None, 0, 0.0
    for M in range(N + 1):
        t = max(acc, (N - M) * L)
        if best is None or t < best:
            best, bestM = t, M
        if M < N:
            acc += R(M)
    return best, bestM
print(f"{'path':40s} {'ctx':>7s} {'recomp-all s':>12s} {'load-all s':>11s} {'prefix-opt':>11s} {'cake s':>8s} {'front%':>7s} {'cake gain':>9s}")
for name, L in PROFILES.items():
    for ctx in (8192, 32768, 65536, 131072, 262144):
        N = ctx // BLOCK
        rec = sum(R(k) for k in range(N)) / 1000
        load = N * L / 1000
        pref = min(rec, load)
        c, M = cake(N, L)
        print(f"{name:40s} {ctx:7d} {rec:12.1f} {load:11.1f} {pref:11.1f} {c/1000:8.1f} {100*M/N:6.1f}% {pref/(c/1000):8.2f}x")
    print()
print("P* (vLLM path) =", round((10.245 - R0) / ALPHA), "tokens;  2*P* =", round(2 * (10.245 - R0) / ALPHA))
print("raw path per-block load ms =", round(PROFILES['raw-NVMe path (896KiB / 2085MiB/s)'], 3), "-> P* =", max(0, round((PROFILES['raw-NVMe path (896KiB / 2085MiB/s)'] - R0) / ALPHA)))
