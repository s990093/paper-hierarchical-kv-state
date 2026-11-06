import random, sys
# 隨機 add／discard 序列，記下真正的 CPython set 走訪順序，和 Rust 版比
seed, n, keyspace = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
rng = random.Random(seed); S = set(); ops = []; exp = []
for i in range(n):
    if rng.random() < 0.55 or not S:
        k = rng.randrange(keyspace); S.add(k); ops.append(f"a {k}")
    else:
        k = rng.choice(list(S)) if rng.random() < 0.9 else rng.randrange(keyspace); S.discard(k); ops.append(f"d {k}")
    if rng.random() < 0.05:
        ops.append("p"); exp.append(" ".join(map(str, S)))
open(sys.argv[4], "w").write("\n".join(ops) + "\n"); open(sys.argv[5], "w").write("\n".join(exp) + "\n")
