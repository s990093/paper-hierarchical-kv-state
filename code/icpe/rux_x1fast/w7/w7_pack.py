"""w7_pack.py — 把 w7gen 的原始陣列包成 X1 的 nb 檔（x1_nb.write_nb，icpe-sim 定的介面，不改），同時做獨立檢查（不用 w7gen 的程式碼）。
用法：python -I w7_pack.py --raw <w7gen gen 的輸出目錄> --name <nb 名稱> --out-dir <nb 目錄> [--keep-raw]
檢查（任一條不過 → AssertionError，不寫檔）：
  1. x1_nb.check（write_nb 寫之前跑）：長度、時間非遞減、kind／parent、鏈式前綴。
  2. 時刻遞推：回來的請求 t[i] ＝ (t[parent] ＋ svc[parent]) ＋ gap[i]（逐位元）；gap ≥ 0；session 的第一個請求 gap＝NaN。
  3. 固定並行：任何時刻占用的槽（[start, end)）≤ N；最大值＝N；每個 session 的 start＝它第一個請求的時刻、end＝最後一個請求＋svc。
  4. 鍵：回來的請求＝上一輪的完整塊＋新鍵；新對話的第一個請求＝系統 prompt 的塊 0..min(sys_full, nf)−1＋新鍵；
     fork 的第一個請求＝另一個「當下活著的」session 在那個時刻（含）之前最新請求的完整塊＋新鍵；「新鍵」＝之前沒出現過的鍵。
  5. user_trigger：新對話的第一個請求＝true、fork 的第一個請求＝false。nall ≤ 2048（1,048,576 token）。
  6. 活工作集：w7_ws.ws_profile（獨立的 Python 實作）算的 WS_ref、WS_max 和 w7gen 寫在 meta 的逐位元相同。
輸出：<out-dir>/<name>.npz、<name>.meta.json、<name>.diag.npz（svc、gap、start、end，只給檢查與分析用）。"""
import argparse, json, math, os, shutil, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ["REPO"]
sys.path.insert(0, os.path.join(REPO, "code", "p3x"))
sys.path.insert(0, HERE)
import x1_nb  # noqa: E402
import w7_ws  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--keep-raw", action="store_true")
a = ap.parse_args()
R = a.raw
rd = lambda nm, dt: np.fromfile(os.path.join(R, nm), dtype=dt)
t, sess, parent, kind, nall = rd("t.f64", "<f8"), rd("sess.i64", "<i8"), rd("parent.i64", "<i8"), rd("kind.i8", "i1"), rd("nall.i64", "<i8")
off, key, user = rd("full_off.i64", "<i8"), rd("full_key.i64", "<i8"), rd("user.u8", "u1").astype(bool)
svc, gap, start, end = rd("svc.f64", "<f8"), rd("gap.f64", "<f8"), rd("start.f64", "<f8"), rd("end.f64", "<f8")
meta = json.load(open(os.path.join(R, "meta.json")))
P = meta["params"]
n = len(t)
N = int(P["N"])
sys_full = int(P["sys_tokens"]) // 512
# 2. 時刻遞推
c1 = kind == 1
p = parent[c1]
assert np.isnan(gap[~c1]).all() and (gap[c1] >= 0).all(), "gap 不對"
assert np.array_equal((t[p] + svc[p]) + gap[c1], t[c1]), "時刻遞推不成立"
# 3. 固定並行
first = np.full(sess.max() + 1, -1, np.int64)
lastr = np.full(sess.max() + 1, -1, np.int64)
for i in range(n):
    s = sess[i]
    if first[s] < 0:
        first[s] = i
    lastr[s] = i
assert np.array_equal(start, t[first]), "start ≠ 第一個請求的時刻"
assert np.array_equal(end, t[lastr] + svc[lastr]), "end ≠ 最後一個請求＋svc"
ev = sorted([(e, 0) for e in end] + [(s_, 1) for s_ in start])     # 同一時刻先釋放再占用
occ = mx = 0
for _, k in ev:
    occ += 1 if k else -1
    assert occ <= N, "占用的槽超過 N"
    mx = max(mx, occ)
assert mx == N, f"最大占用 {mx} ≠ N {N}"
# 4. 鍵、5. 旗標
seen = set()
cur = {}                                      # session → 最新請求的 index（依排序後的順序）
n_fork_ok = 0
for i in range(n):
    ks = key[off[i]:off[i + 1]].tolist()
    s = int(sess[i])
    if kind[i] == 1:
        pk = key[off[parent[i]]:off[parent[i] + 1]].tolist()
        assert ks[:len(pk)] == pk, f"請求 {i}：不是上一輪的延伸"
        pre = len(pk)
    elif kind[i] == 0:
        m = min(sys_full, len(ks))
        assert ks[:m] == list(range(m)), f"請求 {i}：系統 prompt 的塊不對"
        assert user[i], f"請求 {i}：新對話的第一個請求應該是 user"
        pre = m
    else:
        assert not user[i], f"請求 {i}：fork 的第一個請求不該是 user"
        ok = False
        cur = {s2: j for s2, j in cur.items() if end[s2] > t[i]}      # 只留還占著槽的 session（也讓 cur 保持很小）
        for s2, j in cur.items():
            if s2 != s and start[s2] <= t[i] < end[s2]:
                pk = key[off[j]:off[j + 1]].tolist()
                if ks[:len(pk)] == pk and (len(ks) == len(pk) or ks[len(pk)] not in seen):
                    ok = True
                    pre = len(pk)
                    break
        assert ok, f"請求 {i}：fork 的前綴找不到活著的來源"
        n_fork_ok += 1
    fresh = ks[pre:]
    assert not (set(fresh) & seen) and len(set(fresh)) == len(fresh), f"請求 {i}：新鍵之前出現過"
    if kind[i] != 1 and kind[i] != 2:
        seen.update(ks[:pre])
    seen.update(fresh)
    cur[s] = i
assert (nall <= 2048).all() and (nall >= np.diff(off)).all()
# 6. 活工作集（獨立實作）
W = w7_ws.ws_profile(t, sess, off, key)
assert W["WS_ref_GiB"] == meta["WS_ref_GiB"] and W["WS_max_GiB"] == meta["WS_max_GiB"], (W, meta["WS_ref_GiB"], meta["WS_max_GiB"])
# 1. 寫 nb（write_nb 先跑 x1_nb.check）
os.makedirs(a.out_dir, exist_ok=True)
meta.update(trace="w7_synthetic", name=a.name, run_id=os.environ.get("RUN_ID", "adhoc"),
            pack_checks=dict(n_req=n, n_fork_checked=n_fork_ok, max_occupancy=mx, WS_p50_GiB=W["WS_p50_GiB"], WS_p90_GiB=W["WS_p90_GiB"]))
full = [key[off[i]:off[i + 1]].tolist() for i in range(n)]
st = x1_nb.write_nb(os.path.join(a.out_dir, f"{a.name}.npz"), t, sess, parent, kind, nall, full, user_trigger=user, meta=meta)
meta["write_check"] = st
z = np.load(os.path.join(a.out_dir, f"{a.name}.npz"), allow_pickle=False)
assert np.array_equal(z["full_off"], off) and np.array_equal(z["full_key"], key) and np.array_equal(z["t"], t)
np.savez_compressed(os.path.join(a.out_dir, f"{a.name}.diag.npz"), svc=svc, gap=gap, start=start, end=end)
with open(os.path.join(a.out_dir, f"{a.name}.meta.json"), "w") as fh:
    json.dump(meta, fh, ensure_ascii=False, indent=1)
if not a.keep_raw:
    shutil.rmtree(R)
print(f"W7_PACK_OK {a.name} n_req={n} n_sess={len(first)} forks={n_fork_ok} WS_ref={meta['WS_ref_GiB']:.3f}GiB span_h={t[-1] / 3600:.2f}")
