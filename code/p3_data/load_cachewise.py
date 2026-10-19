"""load_cachewise.py — CacheWise CATraces（cachewise-project/cachewise-coding-traces @ 181c435a）→ 統一格式。

⚠ repo 沒有 LICENSE（GitHub API license＝null）→ NEEDS_USER：用在論文前要向作者確認。只有 94 個 session（低於 1,000 的門檻）→ 只報告。
只讀 JSON（raw_traces/*.jsonl 與 parsed_traces/*/events.json）；repo 附的 .py 一律不執行。

一輪＝一個 LLM 請求＝raw trace 裡同一個 requestId 的 assistant 紀錄（Claude Code 每個 content block 一行）；
沒有 requestId 的用 message.id，再沒有就用 uuid（各自成一個請求）。只取主線（isSidechain=False）；子 agent（sidechain）的紀錄計數後不轉。
  trigger ← 這個請求第一則 assistant 紀錄的 parentUuid 指到的紀錄：user 且內容是 tool_result → tool；user 且是文字 → user；其他 → unknown
  t_start_ms ← 那則觸發紀錄的 timestamp（trace：Claude Code 寫入 user／tool_result 的時刻，請求隨即送出）；
     找不到觸發紀錄 → 第一則 assistant 紀錄的時間（t_start_source=estimated，計數）
  t_end_ms ← 這個請求最後一則 assistant 紀錄的 timestamp（trace：最後一個 content block 寫入的時刻）
  think_ms ← 下一個請求的 t_start − t_end（下一輪是 tool 觸發時＝工具時長，user 觸發時＝使用者停頓）
  input_tokens ← usage 的 input＋cache_creation_input＋cache_read_input（同一請求取最大）；output ← output_tokens（取最大）；
     cached ← cache_read_input_tokens；tokenizer＝anthropic(provider)；context_tokens＝input_tokens；new_user_tokens＝null（內容已遮蔽）
  tool_name ← 這個請求回應裡的 tool_use 名稱（| 串）；tool_latency_ms ← parsed events 的 execution_time_ms（取最大）
  session_id ← "<project_id>/<session_id>"；user_id ← null（project_id 不等於使用者）；req_type ← model
  obs_end_ms ← 這個 session 最後一則紀錄的時間（**各 session 的時間戳被各自平移過**〔判讀〕，沒有共同的觀察窗）
  split ← repo 的 tool_duration_prediction/datasets/splits/train_sessions.json（34 個）→ train，其餘 → test（只報告用）

規則 6 的斷言（hard）
  parsed events.json 的 request_id 集合 == raw 主線＋sidechain 的 requestId 集合（session 比例）≥ 0.95；
  parsed events 的 execution_time_ms == result_timestamp − timestamp（±1 ms）≥ 0.95；
  每個請求至少一則 assistant 有 usage ≥ 0.95；parsed manifest 的 session 數 == raw 檔數（94）
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from . import schema as S
except ImportError:
    import schema as S

SOURCE = "cachewise"
ROOT = os.path.join(S.DATASETS, "cachewise-coding-traces", "repo")
VARIANTS = ["catraces"]


def _ms(s):
    return int(round(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000))


def _is_tool_result(rec):
    c = rec.get("message", {}).get("content") if isinstance(rec.get("message"), dict) else None
    return isinstance(c, list) and any(isinstance(x, dict) and x.get("type") == "tool_result" for x in c)


def load(variant):
    ck = S.Checks(SOURCE, variant)
    man = json.load(open(os.path.join(ROOT, "parsed_traces", "manifest.json")))
    train = set(json.load(open(os.path.join(ROOT, "tool_duration_prediction", "datasets", "splits", "train_sessions.json")))["train_sessions"])
    inputs = [S.input_record(os.path.join(ROOT, "parsed_traces", "manifest.json")),
              S.input_record(os.path.join(ROOT, "tool_duration_prediction", "datasets", "splits", "train_sessions.json"))]
    ck.eq("manifest num_sessions == len(sessions)", man["num_sessions"], len(man["sessions"]))
    recs = []
    n_req_match = n_sess = n_tool = n_tool_ok = n_side = n_nouse = n_req_total = n_no_trig = 0
    for s in man["sessions"]:
        raw = os.path.join(ROOT, s["input_path"].replace("release/", ""))
        evp = os.path.join(ROOT, s["output_path"].replace("release/", ""))
        L = [json.loads(l) for l in open(raw, encoding="utf-8")]
        ev = json.load(open(evp, encoding="utf-8"))
        n_sess += 1
        # parsed 的工具時長
        exec_by_msg = {}
        for e in ev:
            if e.get("event_type") == "tool_call" and e.get("execution_time_ms") is not None and e.get("result_timestamp"):
                n_tool += 1
                n_tool_ok += abs((_ms(e["result_timestamp"]) - _ms(e["timestamp"])) - e["execution_time_ms"]) <= 1
                k = e.get("associated_llm_call_id")
                exec_by_msg[k] = max(exec_by_msg.get(k, 0), e["execution_time_ms"])
        ev_req = {e.get("request_id") for e in ev if e.get("event_type") == "llm_call" and e.get("request_id")}
        raw_req = {d.get("requestId") for d in L if d.get("type") == "assistant" and d.get("requestId")}
        n_req_match += ev_req == raw_req
        by_uuid = {d.get("uuid"): d for d in L}
        sid = f"{s['project_id']}/{s['session_id']}"
        reqs = {}
        last_ts = max((_ms(d["timestamp"]) for d in L if d.get("timestamp")), default=None)
        for i, d in enumerate(L):
            if str(d.get("isSidechain")) == "True":
                n_side += 1
                continue
            if d.get("type") != "assistant":
                continue
            m = d.get("message") if isinstance(d.get("message"), dict) else {}
            key = d.get("requestId") or m.get("id") or d.get("uuid") or f"line{i}"
            t = _ms(d["timestamp"])
            u = m.get("usage") or {}
            r = reqs.get(key)
            if r is None:
                p = by_uuid.get(d.get("parentUuid"))
                if p is not None and p.get("type") == "user" and p.get("timestamp"):
                    trig = "tool" if _is_tool_result(p) else "user"
                    t0, t0src = _ms(p["timestamp"]), "trace"
                else:
                    trig, t0, t0src = "unknown", t, "estimated"
                r = reqs[key] = dict(session_id=sid, trigger=trig, t_start_ms=t0, t_start_source=t0src, t_end_ms=t, inp=0, out=0,
                                     cached=0, has_usage=False, tools=[], msg_ids=set(), model=m.get("model"), _order=i,
                                     raw=s["input_path"].replace("release/", ""))
            r["t_end_ms"] = max(r["t_end_ms"], t)
            if u:
                r["has_usage"] = True
                tot = sum(int(u.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
                r["inp"] = max(r["inp"], tot)
                r["out"] = max(r["out"], int(u.get("output_tokens") or 0))
                r["cached"] = max(r["cached"], int(u.get("cache_read_input_tokens") or 0))
            if m.get("id"):
                r["msg_ids"].add(m["id"])
            for x in (m.get("content") or []):
                if isinstance(x, dict) and x.get("type") == "tool_use":
                    r["tools"].append(str(x.get("name")))
        for r in reqs.values():
            n_req_total += 1
            n_nouse += not r["has_usage"]
            n_no_trig += r["t_start_source"] == "estimated"
            lat = [exec_by_msg[mid] for mid in r["msg_ids"] if mid in exec_by_msg]
            r["tool_latency_ms"] = int(round(max(lat))) if lat else None
            r["obs_end_ms"] = last_ts
            r["split"] = "train" if s["session_id"] in train else "test"
            recs.append(r)
    ck.ge("parsed request_id set == raw requestId set (sessions)", n_req_match / n_sess, 0.95)
    ck.ge("tool execution_time_ms == result_ts - ts (±1 ms)", n_tool_ok / max(n_tool, 1), 0.95, f"n_tool={n_tool}")
    ck.ge("requests with usage", 1 - n_nouse / max(n_req_total, 1), 0.95)
    ck.report("sidechain records (not converted)", n_side)
    ck.report("requests without a user/tool_result parent (t_start=first output, estimated)", n_no_trig)
    ck.report("sessions with zero main-thread requests", int(n_sess - len({r["session_id"] for r in recs})))
    df = pd.DataFrame(recs)
    n = len(df)
    # 觸發紀錄的時間晚於第一個輸出（不可能）→ 計數
    ck.report("t_start > t_end (should be 0)", int((df["t_start_ms"] > df["t_end_ms"]).sum()))
    out = pd.DataFrame(dict(
        source=SOURCE, variant=variant, session_id=df["session_id"].values, user_id=None,
        session_rule="official session (Claude Code session file), main thread only", trigger=df["trigger"].values,
        t_start_ms=df["t_start_ms"].values.astype(np.int64), t_end_ms=pd.array(df["t_end_ms"].values, dtype="Int64"),
        t_end_source="trace", input_tokens=pd.array(df["inp"].values, dtype="Int64"), output_tokens=pd.array(df["out"].values, dtype="Int64"),
        cached_tokens=pd.array(df["cached"].values, dtype="Int64"), tokenizer="anthropic(provider)",
        new_user_tokens=pd.array([pd.NA] * n, dtype="Int64"), context_tokens=pd.array(df["inp"].values, dtype="Int64"),
        block_ids=None, block_tokens=pd.array([pd.NA] * n, dtype="Int32"),
        tool_name=["|".join(t) if t else None for t in df["tools"]], tool_category=None,
        tool_latency_ms=df["tool_latency_ms"].astype("Int64").values, req_type=df["model"].values,
        text_ref=[f"{a}#line={o}" for a, o in zip(df["raw"], df["_order"])],
        obs_end_ms=df["obs_end_ms"].values.astype(np.int64), t_start_source=df["t_start_source"].values, think_source="trace",
        service_est_ms=pd.array([pd.NA] * n, dtype="Int64"), split=df["split"].values, split_user=None, harness="claude-code",
        is_subagent=False, parent_session_id=None, next_waited_for=None, provider_prompt_tokens=pd.array([pd.NA] * n, dtype="Int64"),
        _order=df["_order"].values))
    out, fst = S.finalize(out, chain=True)
    ck.report("think_ms<0", fst["think_negative"])
    S.apply_split_window(out, {"train": None, "valid": None, "test": None})
    out = out.drop(columns=["_order"])
    extra = dict(split_rule="repo train_sessions.json（34 個 session）→ train；其餘 → test（論文說 80%，repo 是 34／94）",
                 obs_end_rule="各 session 最後一則紀錄的時間（時間戳各自平移，沒有共同觀察窗）", finalize=fst, tier="R",
                 license="NO LICENSE（GitHub API null）→ NEEDS_USER", note="94 session，只報告")
    ck.enforce()
    return out, ck, inputs, extra
