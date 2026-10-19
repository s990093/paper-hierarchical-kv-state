"""x3_leads.py — X3「拿得到的提前量」：從真實 trace 建出每個回來事件（session 的第 k≥2 輪），記下
  n_old_tokens（要還原的舊 context）、新一輪使用者輸入 token 數、到達間隔、上一輪輸出長度；之後由 x3_hide.py 算可藏起來的比例。

子命令（每個都寫 events_<trace>.csv.gz 到 run 目錄、checks 到 results/p3_explore/x3/）：
  consts     用本機 tokenizer 量：chat template 每輪的固定開銷（Qwen2.5／Llama-3.1）、WildChat 英文使用者訊息的「每 token 幾個字元」
  bailian    Qwen-Bailian（traceA／traceB／thinking／coder）：parent_chat_id 明確連結
  mooncake   Mooncake FAST'25 conversation／toolagent：用前綴鏈重建（D4 的規則：子請求的前綴含父請求最後一個完整 block，block≥1）
  wildchat   WildChat-1M 英文多輪對話：真實訊息文字（token 用 Llama-3.1 tokenizer 算）、assistant 每輪的時間戳
  cachewise  CacheWise（Claude Code）：工具呼叫的 execution_time／result_timestamp、使用者 prompt 的時間戳（只報告）

規則 6（CLAUDE.md）：每個載入函式都有「用 A 欄驗算 B 欄」的斷言，見各函式的 assert 與 checks CSV。
下載的資料是不受信任的：只讀 JSON／parquet，不執行裡面的程式。
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("RAYON_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
import p3_platform as PF  # noqa: E402  （P3_CONST：沒設＝MI300X；h200＝results/*_h200）
OUT = PF.explore_out("x3")
DS = os.environ.get("TIARA_DATASETS", "/ssd8/hungwei/paper-hkv/datasets")
HUB = os.environ.get("TIARA_HF_HUB", "/ssd8/hungwei/paper-hkv/hf-cache/hub")
TOK = {"qwen": f"{HUB}/models--Qwen--Qwen2.5-7B-Instruct-1M/snapshots/e28526f7bb80e2a9c8af03b831a9af3812f18fba",
       "llama": f"{HUB}/models--unsloth--Llama-3.1-8B-Instruct/snapshots/4699cc75b550f9c6f3173fb80f4703b62d946aa5"}
EV_FIELDS = ["trace", "session", "event", "turn", "type", "n_old_tokens", "new_tokens_raw", "typed_tokens", "typed_chars",
             "gap_s", "gen_tokens", "gen_prefill_tokens", "resp_end_known", "think_meas_s", "lead_tool_s",
             "n_tools", "tool_names"]


def ts():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Checks:
    def __init__(self, run_id, name):
        self.run_id, self.rows = run_id, []
        self.path = os.path.join(OUT, f"x3_checks_{name}.csv")

    def put(self, trace, check, value, expect="", ok="", note=""):
        self.rows.append(dict(run_id=self.run_id, ts=ts(), trace=trace, check=check, value=value, expect=expect,
                              ok=ok, note=note))
        print(f"[check] {trace} {check} = {value} {expect} {ok} {note}", flush=True)

    def save(self):
        os.makedirs(OUT, exist_ok=True)
        with open(self.path, "w", newline="") as fh:
            w = csv.DictWriter(fh, list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)
        print("wrote", self.path)


def write_events(run_dir, trace, evs, run_id):
    p = os.path.join(run_dir, f"events_{trace}.csv.gz")
    with gzip.open(p, "wt", newline="") as fh:
        w = csv.DictWriter(fh, ["run_id", "ts"] + EV_FIELDS)
        w.writeheader()
        t = ts()
        for e in evs:
            row = {k: e.get(k, "") for k in EV_FIELDS}
            row.update(run_id=run_id, ts=t, trace=trace)
            w.writerow(row)
    print("wrote", p, len(evs))
    return p


def pct(x, qs=(1, 5, 10, 25, 50, 75, 90, 95, 99)):
    x = np.asarray(x, float)
    return " ".join(f"p{q}={np.percentile(x, q):.4g}" for q in qs) if len(x) else "empty"


def common_prefix(a, b):
    k, m = 0, min(len(a), len(b))
    while k < m and a[k] == b[k]:
        k += 1
    return k


# ------------------------------------------------------------------ consts
def cmd_consts(a):
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    C = Checks(a.run_id, "consts")
    res = {"run_id": a.run_id}
    # 1. chat template 每輪的固定開銷：加一輪 (assistant 回覆 a、使用者訊息 u) 之後 prompt 多出來的 token，扣掉 a 與 u 自己的 token
    probes = [("Hi there.", "Sure, here is the answer you asked for.", "Thanks! Can you explain step two in more detail?"),
              ("写一首关于秋天的诗。", "好的，这是一首关于秋天的诗。", "再写一首关于冬天的。"),
              ("def f(x): return x", "This function returns its input.", "ok")]
    for m, path in TOK.items():
        tk = AutoTokenizer.from_pretrained(path)
        ov = []
        for u1, a1, u2 in probes:
            m1 = [{"role": "user", "content": u1}]
            m2 = m1 + [{"role": "assistant", "content": a1}, {"role": "user", "content": u2}]
            # 先產生字串再 encode（transformers 5 的 tokenize=True 回傳 BatchEncoding）；template 已含特殊 token，不再加 BOS
            t1 = tk.encode(tk.apply_chat_template(m1, tokenize=False, add_generation_prompt=True), add_special_tokens=False)
            t2 = tk.encode(tk.apply_chat_template(m2, tokenize=False, add_generation_prompt=True), add_special_tokens=False)
            assert t2[:len(t1)] == t1, f"{m}: 第二輪的 prompt 不以第一輪的 prompt 開頭（template 會改前文）"
            na = len(tk.encode(a1, add_special_tokens=False))
            nu = len(tk.encode(u2, add_special_tokens=False))
            ov.append(len(t2) - len(t1) - na - nu)
        C.put(m, "template_overhead_tokens_per_turn_incl_eos", " ".join(map(str, ov)), "", "",
              "len(prompt2)-len(prompt1)-len(assistant)-len(user2)；含 assistant 結尾的 EOS（若 trace 的 output_length 已含 EOS，實際開銷少 1）")
        res[f"template_overhead_{m}"] = int(min(ov))
    # 2. WildChat 英文使用者訊息：每 token 幾個字元
    f0 = os.path.join(DS, "wildchat-1m", "data", "train-00000-of-00014.parquet")
    t = pq.read_table(f0, columns=["conversation", "language"])
    texts = {"English": [], "Chinese": []}
    for conv, lang in zip(t.column("conversation").to_pylist(), t.column("language").to_pylist()):
        if lang not in texts or len(texts[lang]) >= 20000:
            continue
        for msg in conv:
            if msg["role"] == "user" and msg["language"] == lang and msg["content"]:
                texts[lang].append(msg["content"])
                break
    for m, path in TOK.items():
        tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
        for lang, xs in texts.items():
            enc = tok.encode_batch(xs, add_special_tokens=False)
            nt = np.array([len(e.ids) for e in enc], float)
            nc = np.array([len(x) for x in xs], float)
            keep = nt > 0
            cpt_pooled = nc[keep].sum() / nt[keep].sum()
            cpt_median = float(np.median(nc[keep] / nt[keep]))
            C.put(f"wildchat_{lang}", f"chars_per_token_{m}", round(cpt_pooled, 4), "", "",
                  f"n_msgs={int(keep.sum())} pooled=Σchars/Σtokens；per-message median={cpt_median:.4f}；"
                  f"msg tokens {pct(nt[keep], (10, 50, 90))}")
            res[f"cpt_{m}_{lang}"] = cpt_pooled
            res[f"cpt_{m}_{lang}_median_msg"] = cpt_median
    res["wildchat_file"] = f0
    json.dump(res, open(os.path.join(OUT, "x3_typing_consts.json"), "w"), indent=1)
    C.save()


# ------------------------------------------------------------------ bailian
def cmd_bailian(a):
    C = Checks(a.run_id, "bailian")
    for name in ("traceA", "traceB", "thinking", "coder"):
        path = os.path.join(DS, "qwen-bailian-usagetraces-anon", f"qwen_{name}_blksz_16.jsonl")
        rows = [json.loads(l) for l in open(path)]
        n = len(rows)
        # 規則 6：hash_ids 的單位（16 token）用 input_length 驗算
        match = sum(len(r["hash_ids"]) == math.ceil(r["input_length"] / 16) for r in rows) / n
        assert match >= 0.99, f"{name}: len(hash_ids)==ceil(input_length/16) 只有 {match}"
        C.put(name, "unit_len_hash_eq_ceil_input_over_16", round(match, 6), ">=0.99", "OK")
        tss = [r["timestamp"] for r in rows]
        assert tss == sorted(tss), name
        by = {r["chat_id"]: r for r in rows}
        assert len(by) == n, f"{name}: chat_id 不唯一"
        kids = [r for r in rows if r["parent_chat_id"] != -1]
        C.put(name, "n_requests", n)
        C.put(name, "n_children", len(kids))
        if not kids:
            C.put(name, "sessions_with_return", 0, "", "", "沒有 parent 連結 → 不能當 session trace")
            write_events(a.run_dir, f"bailian_{name}", [], a.run_id)
            continue
        miss = sum(r["parent_chat_id"] not in by for r in kids)
        turn_ok = sum(by[r["parent_chat_id"]]["turn"] + 1 == r["turn"] for r in kids if r["parent_chat_id"] in by)
        C.put(name, "children_parent_missing", miss)
        C.put(name, "turn_eq_parent_turn_plus1_frac", round(turn_ok / max(1, len(kids) - miss), 6), "==1")
        # timestamp 單位（秒）：同 session 間隔 ÷ 上一輪輸出 token（應在 0.01–10 s/token 的量級）
        evs = []
        parent = {}
        inc, cpr, gapr, rat = [], [], [], []
        for r in kids:
            p = by.get(r["parent_chat_id"])
            if p is None:
                continue
            parent[r["chat_id"]] = p["chat_id"]
            cp = common_prefix(p["hash_ids"], r["hash_ids"])
            new_raw = r["input_length"] - p["input_length"] - p["output_length"]
            gap = r["timestamp"] - p["timestamp"]
            inc.append(new_raw)
            cpr.append(cp * 16 / max(1, p["input_length"] + p["output_length"]))
            gapr.append(gap)
            if p["output_length"] >= 50:
                rat.append(gap / p["output_length"])
            # 父請求自己的 prefill（估 TTFT 用）：父請求和它的父請求的共同前綴之外的 token
            pp = by.get(p["parent_chat_id"])
            p_cached = common_prefix(pp["hash_ids"], p["hash_ids"]) * 16 if pp is not None else 0
            evs.append(dict(session=r["chat_id"], event=len(evs), turn=r["turn"], type=r["type"],
                            n_old_tokens=cp * 16, new_tokens_raw=new_raw, gap_s=round(gap, 3),
                            gen_tokens=p["output_length"],
                            gen_prefill_tokens=max(0, p["input_length"] - p_cached), resp_end_known=0))
        # session 數：沿 parent 往上找 root
        root = {}
        for r in rows:
            c, path = r["chat_id"], []
            while c in parent and c not in root:
                path.append(c)
                c = parent[c]
            base = root.get(c, c)
            for x in path:
                root[x] = base
            root.setdefault(r["chat_id"], base)
        sz = Counter(root[r["chat_id"]] for r in rows)
        nret = sum(v >= 2 for v in sz.values())
        rootof = {e["session"]: root[e["session"]] for e in evs}
        for e in evs:
            e["session"] = rootof[e["session"]]
        C.put(name, "sessions_total", len(sz))
        C.put(name, "sessions_with_return", nret, ">=1000 才算真實 trace 門檻", "OK" if nret >= 1000 else "BELOW")
        C.put(name, "return_events", len(evs))
        C.put(name, "new_tokens_raw_pctl", pct(inc), "", "", "child.input−parent.input−parent.output")
        C.put(name, "new_tokens_raw_neg_frac", round(float(np.mean(np.array(inc) < 0)), 6))
        C.put(name, "common_prefix_over_parent_context_pctl", pct(cpr), "", "",
              "共同前綴 block×16 ÷ (父輸入＋父輸出)；<1 表示上一輪的 KV 只有一部分能重用")
        C.put(name, "gap_s_pctl", pct(gapr))
        C.put(name, "gap_s_per_parent_output_token_median", round(float(np.median(rat)), 4) if rat else "",
              "0.01–10", "OK" if rat and 0.01 < np.median(rat) < 10 else "CHECK", "timestamp 單位＝秒 的驗算")
        types = Counter(e["type"] for e in evs)
        C.put(name, "return_types", json.dumps(types))
        write_events(a.run_dir, f"bailian_{name}", evs, a.run_id)
    C.save()


# ------------------------------------------------------------------ mooncake
def parents_strict(rows):
    """嚴格重建（README §4-2；同 research/calc/mooncake_chain2.py 的規則）：子請求延伸的「父請求完整前綴」
    總共只被 ≤3 個請求用到、候選父請求唯一、且時間差 >0。其他子請求不連（當成新 session 的第一輪）。"""
    pop = Counter()
    for r in rows:
        h = r["hash_ids"]
        for L in range(1, len(h) + 1):
            pop[tuple(h[:L])] += 1
    full = [r["input_length"] // 512 for r in rows]
    cand = defaultdict(list)
    for i, r in enumerate(rows):
        if full[i] >= 1:
            cand[tuple(r["hash_ids"][:full[i]])].append(i)
    par = [None] * len(rows)
    for j, r in enumerate(rows):
        h = r["hash_ids"]
        for L in range(len(h) - 1, 0, -1):
            ps = [i for i in cand.get(tuple(h[:L]), []) if i < j and rows[i]["timestamp"] < r["timestamp"]]
            if ps:
                if pop[tuple(h[:L])] <= 3 and len(ps) == 1:
                    par[j] = ps[0]
                break
    return par


def cmd_mooncake(a):
    C = Checks(a.run_id, f"mooncake_{a.rule}")
    for name in ("conversation", "toolagent"):
        rows = [json.loads(l) for l in open(os.path.join(DS, "mooncake", "traces", f"{name}_trace.jsonl"))]
        n = len(rows)
        match = sum(len(r["hash_ids"]) == math.ceil(r["input_length"] / 512) for r in rows) / n
        assert match >= 0.99, f"{name}: hash_ids 不是 512 token（規則 6）"
        C.put(name, "unit_len_hash_eq_ceil_input_over_512", round(match, 6), ">=0.99", "OK")
        tss = [r["timestamp"] for r in rows]
        assert tss == sorted(tss)
        lastfull, par = {}, [None] * n
        for ri, r in enumerate(rows):
            h = r["hash_ids"]
            for k in range(len(h) - 1, 0, -1):   # 不看 block 0（conversation 全部請求共用）
                if h[k] in lastfull:
                    par[ri] = lastfull[h[k]]
                    break
            nf = r["input_length"] // 512
            if nf >= 2:
                lastfull[h[nf - 1]] = ri
        if a.rule == "strict":
            par = parents_strict(rows)
        evs, inc, rat = [], [], []
        root = {}
        for ri, p in enumerate(par):
            root[ri] = root[p] if p is not None else ri
        for ri, p in enumerate(par):
            if p is None:
                continue
            r, q = rows[ri], rows[p]
            cp = common_prefix(q["hash_ids"], r["hash_ids"])
            new_raw = r["input_length"] - q["input_length"] - q["output_length"]
            gap = (r["timestamp"] - q["timestamp"]) / 1000.0
            inc.append(new_raw)
            if q["output_length"] >= 50:
                rat.append(gap / q["output_length"])
            pp = par[p]
            p_cached = common_prefix(rows[pp]["hash_ids"], q["hash_ids"]) * 512 if pp is not None else 0
            evs.append(dict(session=root[ri], event=len(evs), turn="", type="", n_old_tokens=min(cp * 512, q["input_length"] + q["output_length"]),
                            new_tokens_raw=new_raw, gap_s=round(gap, 3), gen_tokens=q["output_length"],
                            gen_prefill_tokens=max(0, q["input_length"] - p_cached), resp_end_known=0))
        sz = Counter(root.values())
        nret = sum(v >= 2 for v in sz.values())
        pc = Counter(p for p in par if p is not None)
        C.put(name, "return_events", len(evs))
        C.put(name, "sessions_with_return", nret, ">=1000", "OK" if nret >= 1000 else "BELOW",
              f"前綴鏈重建，規則={a.rule}（loose＝D4 規則；strict＝前綴 ≤3 個請求共用、父唯一、間隔>0）")
        C.put(name, "parents_with_multiple_children", sum(v > 1 for v in pc.values()))
        C.put(name, "new_tokens_raw_pctl", pct(inc))
        C.put(name, "new_tokens_raw_neg_frac", round(float(np.mean(np.array(inc) < 0)), 6), "", "",
              "負的＝子請求比「父輸入＋父輸出」短：截斷、或分叉被誤當同一 session")
        C.put(name, "gap_s_per_parent_output_token_median", round(float(np.median(rat)), 4), "0.01–10",
              "OK" if 0.01 < np.median(rat) < 10 else "CHECK", "timestamp 單位＝毫秒 的驗算（D4 §4.1）")
        C.put(name, "gap_s_pctl", pct([e["gap_s"] for e in evs]))
        C.put(name, "ts_resolution_frac_multiple_of_1000ms", round(float(np.mean([t % 1000 == 0 for t in tss])), 4),
              "", "", "D4：解析度約 1 秒")
        write_events(a.run_dir, f"mooncake_{name}" + ("_strict" if a.rule == "strict" else ""), evs, a.run_id)
    C.save()


# ------------------------------------------------------------------ wildchat
def cmd_wildchat(a):
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    C = Checks(a.run_id, "wildchat")
    tok = Tokenizer.from_file(os.path.join(TOK["llama"], "tokenizer.json"))
    tov = json.load(open(os.path.join(OUT, "x3_typing_consts.json")))["template_overhead_llama"]
    files = sorted(glob.glob(os.path.join(DS, "wildchat-1m", "data", "train-*.parquet")))[:a.files]
    evs = []
    nconv = nmulti = 0
    skipped_ts = 0
    gen_rat = []
    for f in files:
        t = pq.read_table(f, columns=["conversation_hash", "conversation", "language", "turn", "model"])
        convs = t.column("conversation").to_pylist()
        langs = t.column("language").to_pylist()
        hashes = t.column("conversation_hash").to_pylist()
        nconv += len(convs)
        sel = [(h, c) for h, c, l in zip(hashes, convs, langs) if l == "English" and len(c) >= 4]
        nmulti += len(sel)
        # 一次編碼所有訊息
        flat = [m["content"] or "" for _, c in sel for m in c]
        enc = tok.encode_batch(flat, add_special_tokens=False)
        ntok = [len(e.ids) for e in enc]
        i = 0
        for h, c in sel:
            k = len(c)
            nt = ntok[i:i + k]
            i += k
            roles = [m["role"] for m in c]
            if roles[0::2] != ["user"] * len(roles[0::2]) or roles[1::2] != ["assistant"] * len(roles[1::2]):
                continue
            ats = [m["timestamp"] for m in c]
            ctx = 0
            for j in range(0, k - 1, 2):   # j＝user，j+1＝assistant
                if j >= 2:
                    # 回來事件：user j（第 j/2+1 輪）。上一輪回應結束＝ats[j−1]；這一輪回應結束＝ats[j+1]
                    if ats[j - 1] is None or ats[j + 1] is None:
                        skipped_ts += 1
                    else:
                        gap = (ats[j + 1] - ats[j - 1]).total_seconds()   # 上一輪結束 → 這一輪結束
                        out_now = nt[j + 1]
                        if out_now >= 50 and gap > 0:
                            gen_rat.append(gap / out_now)
                        # gap_s＝上一輪結束→這一輪結束，要扣掉「這一輪」的 prefill 與生成（gen_*）才是思考＋打字時間
                        evs.append(dict(session=h, event=len(evs), turn=j // 2 + 1, type="text",
                                        n_old_tokens=ctx, new_tokens_raw=nt[j] + tov, typed_tokens=nt[j],
                                        typed_chars=len(c[j]["content"] or ""), gap_s=round(gap, 3),
                                        gen_tokens=out_now, gen_prefill_tokens=nt[j],
                                        resp_end_known=1, think_meas_s=""))
                ctx += nt[j] + nt[j + 1] + tov
        print(f, len(sel), len(evs), flush=True)
    # 規則 6：assistant 的時間戳是不是「回應結束」？user 沒有時間戳（全部 None），所以用
    # 「上一輪結束→這一輪結束」÷ 這一輪輸出 token 的下緣：若時間戳是回應結束，下緣應 ≥ 解碼時間（每 token 數 ms 以上）
    C.put("wildchat", "files_used", len(files), "", "", " ".join(os.path.basename(x) for x in files))
    C.put("wildchat", "conversations_read", nconv)
    C.put("wildchat", "english_multiturn_conversations", nmulti, ">=1000", "OK" if nmulti >= 1000 else "BELOW")
    C.put("wildchat", "return_events", len(evs))
    C.put("wildchat", "events_skipped_missing_assistant_ts", skipped_ts)
    C.put("wildchat", "gap_end_to_end_per_output_token_s_pctl", pct(gen_rat, (0.1, 1, 5, 50)), "", "",
          "下緣（p1）＝能觀察到的最快『思考＋打字＋生成』每 token 時間；p1 若 < 典型解碼時間就表示時間戳不是回應結束")
    C.put("wildchat", "user_timestamp_present", "False", "", "", "WildChat-1M 的 user 訊息 timestamp 全是 None（已檢查第 0 個檔前 3,000 筆）")
    write_events(a.run_dir, "wildchat_en", evs, a.run_id)
    C.save()


# ------------------------------------------------------------------ cachewise
def _pts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def cmd_cachewise(a):
    C = Checks(a.run_id, "cachewise")
    root = os.path.join(DS, "cachewise-coding-traces", "repo")
    man = json.load(open(os.path.join(root, "parsed_traces", "manifest.json")))
    tool_evs, human_evs = [], []
    bad_exec = n_tools = 0
    for s in man["sessions"]:
        p = os.path.join(root, s["output_path"].replace("release/", ""))
        ev = json.load(open(p))
        llm = {}
        for e in ev:
            if e["event_type"] == "llm_call":
                llm[e["message_id"]] = e   # 同一 message_id 的最後一筆（usage 相同）
        bymsg = defaultdict(list)
        for e in ev:
            if e["event_type"] == "tool_call" and e.get("has_result") and e.get("result_timestamp"):
                bymsg[e["associated_llm_call_id"]].append(e)
        for mid, tools in bymsg.items():
            L = llm.get(mid)
            if L is None:
                continue
            t0 = min(_pts(x["timestamp"]) for x in tools)
            t1 = max(_pts(x["result_timestamp"]) for x in tools)
            lead = (t1 - t0).total_seconds()
            for x in tools:
                n_tools += 1
                ex = x.get("execution_time_ms")
                d = (_pts(x["result_timestamp"]) - _pts(x["timestamp"])).total_seconds() * 1000
                if ex is not None and abs(ex - d) > 1.0:
                    bad_exec += 1
            ctx = (L.get("input_tokens") or 0) + (L.get("cache_creation_input_tokens") or 0) + \
                  (L.get("cache_read_input_tokens") or 0) + (L.get("output_tokens") or 0)
            tool_evs.append(dict(session=s["session_id"], event=len(tool_evs), type="tool", n_old_tokens=ctx,
                                 lead_tool_s=round(lead, 3), n_tools=len(tools),
                                 tool_names="|".join(sorted(set(x["tool_name"] for x in tools)))))
        # 使用者 prompt：raw trace 裡 type=user、content 是字串或 text 清單、非 sidechain；思考時間＝它 − 前一個 assistant 訊息
        raw = os.path.join(root, s["input_path"].replace("release/", ""))
        last_asst, last_ctx = None, 0
        for l in open(raw):
            d = json.loads(l)
            if d.get("type") == "assistant":
                last_asst = _pts(d["timestamp"])
                u = d.get("message", {}).get("usage") or {}
                try:
                    last_ctx = sum(int(u.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens",
                                                                 "cache_read_input_tokens", "output_tokens"))
                except (TypeError, ValueError):
                    pass
            elif d.get("type") == "user" and str(d.get("isSidechain")) == "False":
                cont = d.get("message", {}).get("content")
                is_prompt = isinstance(cont, str) or (isinstance(cont, list) and cont and
                                                      all(isinstance(x, dict) and x.get("type") == "text" for x in cont))
                if is_prompt and last_asst is not None:
                    think = (_pts(d["timestamp"]) - last_asst).total_seconds()
                    human_evs.append(dict(session=s["session_id"], event=len(human_evs), type="human",
                                          n_old_tokens=last_ctx, think_meas_s=round(think, 3), resp_end_known=1))
    C.put("cachewise", "sessions", len(man["sessions"]), ">=1000", "BELOW", "只報告（README §1）")
    C.put("cachewise", "tool_calls_with_result", n_tools)
    C.put("cachewise", "tool_exec_ms_vs_result_minus_issue_mismatch_gt1ms", bad_exec, "==0",
          "OK" if bad_exec == 0 else "CHECK", "規則 6：execution_time_ms 用 result_timestamp − timestamp 驗算")
    C.put("cachewise", "tool_return_events", len(tool_evs))
    C.put("cachewise", "tool_lead_s_pctl", pct([e["lead_tool_s"] for e in tool_evs]))
    C.put("cachewise", "tool_return_ctx_tokens_pctl", pct([e["n_old_tokens"] for e in tool_evs]))
    C.put("cachewise", "human_prompt_events", len(human_evs))
    C.put("cachewise", "human_think_s_pctl", pct([e["think_meas_s"] for e in human_evs]), "", "",
          "使用者 prompt 時間 − 前一個 assistant 訊息時間（含讀、想、打字；assistant 訊息時間戳是否＝回應結束未驗證）")
    write_events(a.run_dir, "cachewise_tool", tool_evs, a.run_id)
    write_events(a.run_dir, "cachewise_human", human_evs, a.run_id)
    C.save()


# ------------------------------------------------------------------ freeinference（README §4-1：登記之後才加）
def cmd_freeinference(a):
    """FreeInference agentic trace（processed/*.parquet，16 週）。每個 session 內依 step 排序；回來事件＝step≥1 的請求。
    間隔＝這個請求 start_ms − 上一個請求 end_ms（回應結束是實測，不用估）。step_trigger＝tool → 訊號 (c) 的提前量；
    ＝user → 訊號 (b) 的思考時間，(a) 用 role_tokens.user 的增量（o200k token）當新使用者 token 數（上界：harness 可能把系統提示放在 user）。
    要還原的長度＝上一個請求的輸入（role_tokens 總和，tiktoken）＋輸出（completion_tokens）；用前兩週的 block_ids 共同前綴驗算。"""
    import pyarrow.parquet as pq
    C = Checks(a.run_id, "freeinference")
    root = os.path.join(DS, "freeinference_agentic_trace", "release")
    weeks = sorted(os.listdir(root))
    tool_evs, user_evs = [], []
    unit_ok = unit_n = 0
    neg_gap = 0
    cp_ratio = []
    nsess_tool, nsess_user = set(), set()
    cols = ["session_id", "step", "start_ms", "end_ms", "ttft_ms", "step_trigger", "prompt_tokens", "completion_tokens",
            "role_tokens", "transition"]
    for wi, w in enumerate(weeks):
        pth = os.path.join(root, w, "processed", "requests.parquet")
        withblk = wi < 2
        t = pq.read_table(pth, columns=cols + (["block_ids"] if withblk else []))
        d = t.to_pandas()
        rt = d.role_tokens.apply(lambda x: sum(v for k, v in (x.items() if isinstance(x, dict) else x) if v is not None)
                                 if x is not None else np.nan)
        ru = d.role_tokens.apply(lambda x: (dict(x).get("user") if x is not None else np.nan))
        d["in_tok"] = rt
        d["user_tok"] = ru
        if withblk:
            nb = d.block_ids.apply(len)
            unit_ok += int((nb == (d.in_tok // 16)).sum())
            unit_n += len(d)
        d = d.sort_values(["session_id", "step"]).reset_index(drop=True)
        prev = d.groupby("session_id").shift(1)
        for i in range(len(d)):
            if pd_isna(prev.start_ms.iat[i]):
                continue
            gap = (d.start_ms.iat[i] - prev.end_ms.iat[i]) / 1000.0
            sid = f"{w}/{d.session_id.iat[i]}"
            if gap < 0:
                neg_gap += 1
                continue
            n_old = float(prev.in_tok.iat[i]) + float(prev.completion_tokens.iat[i] or 0)
            if withblk and d.transition.iat[i] == "append" and len(cp_ratio) < 200000:
                cp = common_prefix(list(prev.block_ids.iat[i]), list(d.block_ids.iat[i])) * 16
                cp_ratio.append(cp / max(1.0, n_old))
            e = dict(session=sid, event=0, turn=int(d.step.iat[i]), type=d.step_trigger.iat[i], n_old_tokens=int(n_old),
                     gap_s=round(gap, 3), resp_end_known=1, gen_tokens=0, gen_prefill_tokens=0)
            if d.step_trigger.iat[i] == "tool":
                e.update(lead_tool_s=round(gap, 3))
                tool_evs.append(e)
                nsess_tool.add(sid)
            elif d.step_trigger.iat[i] == "user":
                du = d.user_tok.iat[i] - prev.user_tok.iat[i]
                e.update(think_meas_s=round(gap, 3), typed_tokens=(int(du) if du == du else ""),
                         new_tokens_raw=int(d.in_tok.iat[i] - n_old) if d.in_tok.iat[i] == d.in_tok.iat[i] else "")
                user_evs.append(e)
                nsess_user.add(sid)
        print(w, len(d), len(tool_evs), len(user_evs), flush=True)
    for k, e in enumerate(tool_evs):
        e["event"] = k
    for k, e in enumerate(user_evs):
        e["event"] = k
    C.put("freeinference", "weeks", len(weeks))
    C.put("freeinference", "unit_len_block_ids_eq_floor_tiktoken_over_16_weeks1-2", round(unit_ok / max(1, unit_n), 6),
          ">=0.95", "OK" if unit_ok / max(1, unit_n) >= 0.95 else "CHECK", "README：o200k、16 token、丟掉最後不滿的 block")
    C.put("freeinference", "negative_gap_excluded", neg_gap, "", "", "start_ms 早於上一個請求的 end_ms（重疊）")
    C.put("freeinference", "cp16_over_prev_in_plus_out_append_pctl", pct(cp_ratio), "", "",
          "前兩週、transition=append：block 共同前綴 ×16 ÷（上一個輸入＋輸出）；≈1 表示 n_old 的定義對")
    C.put("freeinference", "tool_return_events", len(tool_evs))
    C.put("freeinference", "tool_sessions", len(nsess_tool), ">=1000", "OK" if len(nsess_tool) >= 1000 else "BELOW")
    C.put("freeinference", "tool_lead_s_pctl", pct([e["lead_tool_s"] for e in tool_evs]))
    C.put("freeinference", "user_return_events", len(user_evs))
    C.put("freeinference", "user_sessions", len(nsess_user), ">=1000", "OK" if len(nsess_user) >= 1000 else "BELOW")
    C.put("freeinference", "user_think_s_pctl", pct([e["think_meas_s"] for e in user_evs]))
    ut = [e["typed_tokens"] for e in user_evs if e["typed_tokens"] != ""]
    C.put("freeinference", "user_role_token_increment_pctl", pct(ut), "", "", "role_tokens.user 的增量（o200k）；負的＝壓縮／改寫")
    C.put("freeinference", "ctx_tokens_pctl_tool", pct([e["n_old_tokens"] for e in tool_evs]))
    write_events(a.run_dir, "freeinference_tool", tool_evs, a.run_id)
    write_events(a.run_dir, "freeinference_user", user_evs, a.run_id)
    C.save()


def pd_isna(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["consts", "bailian", "mooncake", "wildchat", "cachewise", "freeinference"])
    ap.add_argument("--rule", default="loose", choices=["loose", "strict"], help="mooncake 的 session 重建規則")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--files", type=int, default=2, help="wildchat：讀前幾個 parquet")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
