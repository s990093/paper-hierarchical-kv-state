"""W3 線索 (a) 的每請求旗標 cue.u8（SEMANTICS 4-2：cue[r]＝這一輪的輸出有工具呼叫），順序＝nb 檔的請求順序。
不改 icpe-data 的轉換程式：照 code/icpe/next_b_convert.py 的 conv_tracelab＋finalize 的排序步驟重做一次（不產生 chunk 鍵，記憶體很小），
另外寫 t、sess 陣列的 sha256，讓有 nb 檔的機器核對對齊（`check`）。
用法：
  python -I make_cue.py tracelab --compress 1 2 4 --out <目錄>        → <目錄>/tracelab_c<c>.cue.u8、.cue.json
  python -I make_cue.py proxy <nb 名稱> --out <目錄>                   → 代用：cue[r]＝同一個 session 的下一個請求不是使用者觸發（nb 的 user_trigger）。
                                                                        「none」會被當成工具；只給沒有工具欄位的紀錄測管線用
  python -I make_cue.py check <nb 名稱> <目錄>                         → nb 檔的 t、sess 和 .cue.json 記的 sha256 相同才 PASS
TraceLab 的 `tools`（這一輪輸出的工具呼叫）不在 agent_traces.load_tracelab 的欄位裡，這裡另外讀一次原始檔，用 (session_id, round_index) 對上。"""
import argparse, gzip, hashlib, json, os, sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
for p in (os.path.join(REPO, "code"), os.path.join(REPO, "code", "icpe"), os.path.join(REPO, "code", "p3x")):
    sys.path.insert(0, p)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def order_like_finalize(sess_raw, t, extra):
    """next_b_convert.finalize 的排序（trunc 不做）：session 內時間非遞減 → 依 (t, 原順序) 穩定排序 → session 編號依出現順序。"""
    t = np.array(t, np.float64).copy()
    last = {}
    for i, s in enumerate(sess_raw):
        if s in last and t[i] < last[s]:
            t[i] = last[s]
        last[s] = t[i]
    o = np.lexsort((np.arange(len(t)), t))           # = sort_values(["t", "_o"], kind="stable")
    sid = {}
    sess = np.array([sid.setdefault(sess_raw[i], len(sid)) for i in o], np.int64)
    ts = t[o]
    tt = ts - ts[0] if len(ts) else ts
    return tt, sess, extra[o]


def cmd_tracelab(a):
    import agent_traces as AT
    d, _ = AT.load_tracelab()
    d = d.rename(columns={"sid": "sess"}).sort_values(["sess", "round"], kind="stable").reset_index(drop=True)
    nt = {}
    with gzip.open(f"{AT.DS}/tracelab/syfi_coding_trace.jsonl.gz", "rt") as fh:
        for ln in fh:
            r = json.loads(ln)
            nt[(r["session_id"], r["round_index"])] = len(r.get("tools") or [])
    cue_raw = np.array([1 if nt[(s, k)] > 0 else 0 for s, k in zip(d.sess.values, d["round"].values)], np.uint8)
    T0 = d.start_ms.min()
    s0 = d.groupby("sess").start_ms.transform("min")
    os.makedirs(a.out, exist_ok=True)
    for c in a.compress:
        t = (((T0 + (s0 - T0) / c) + (d.start_ms - s0) - T0) / 1000).values
        tt, sess, cue = order_like_finalize(list(d.sess.values), t, cue_raw)
        name = f"tracelab_c{c}"
        cue.tofile(os.path.join(a.out, f"{name}.cue.u8"))
        meta = dict(name=name, rule="cue[r]＝這一輪輸出的 tools 非空（TraceLab 原始紀錄）", n_req=int(len(cue)), n_cue=int(cue.sum()),
                    sha_t=sha(tt), sha_sess=sha(sess), source="agent_traces.load_tracelab＋原始檔的 tools；排序照 next_b_convert.finalize")
        json.dump(meta, open(os.path.join(a.out, f"{name}.cue.json"), "w"), ensure_ascii=False, indent=1)
        print(json.dumps(meta, ensure_ascii=False), flush=True)


def cmd_proxy(a):
    import x1_nb
    z = np.load(os.path.join(x1_nb.NB_DIR, f"{a.name}.npz"), allow_pickle=False)
    sess, ut = z["sess"], z["user_trigger"].astype(bool)
    n = len(sess)
    cue = np.zeros(n, np.uint8)
    nxt = {}
    for r in range(n - 1, -1, -1):
        s = int(sess[r])
        if s in nxt:
            cue[r] = 0 if ut[nxt[s]] else 1
        nxt[s] = r
    os.makedirs(a.out, exist_ok=True)
    cue.tofile(os.path.join(a.out, f"{a.name}.cue.u8"))
    meta = dict(name=a.name, rule="代用：cue[r]＝同一 session 的下一個請求 user_trigger＝False（none 也算工具）", n_req=int(n), n_cue=int(cue.sum()),
                sha_t=sha(z["t"]), sha_sess=sha(z["sess"]))
    json.dump(meta, open(os.path.join(a.out, f"{a.name}.cue.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(meta, ensure_ascii=False), flush=True)


def cmd_check(a):
    import x1_nb
    z = np.load(os.path.join(x1_nb.NB_DIR, f"{a.name}.npz"), allow_pickle=False)
    m = json.load(open(os.path.join(a.dir, f"{a.name}.cue.json")))
    ok = sha(z["t"]) == m["sha_t"] and sha(z["sess"]) == m["sha_sess"] and len(z["t"]) == m["n_req"]
    print(f"CUE_CHECK {'PASS' if ok else 'FAIL'} {a.name} n_req nb={len(z['t'])} cue={m['n_req']}")
    sys.exit(0 if ok else 1)


ap = argparse.ArgumentParser()
sp = ap.add_subparsers(dest="cmd", required=True)
x = sp.add_parser("tracelab"); x.add_argument("--compress", nargs="+", type=int, default=[1]); x.add_argument("--out", required=True)
x = sp.add_parser("proxy"); x.add_argument("name"); x.add_argument("--out", required=True)
x = sp.add_parser("check"); x.add_argument("name"); x.add_argument("dir")
a = ap.parse_args()
{"tracelab": cmd_tracelab, "proxy": cmd_proxy, "check": cmd_check}[a.cmd](a)
