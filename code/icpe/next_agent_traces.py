"""next_agent_traces.py — icpe-data：公開 agent 軌跡的載入（PREREG 偏離 17；STATUS_DATA 2026-11-07 12:55 的登記）。
next_b_convert.py 的 `agent` 子命令呼叫這裡。只讀 JSON（軌跡）與 Python 原始碼的文字（OpenHands，用 ast 取字面值，不執行）。

回傳每條軌跡：dict(sid, msgs=[(role, text)], calls=[dict(m=這次呼叫產生的 assistant 訊息在 msgs 的位置, t_req, t_end, ptok=供應商的 prompt_tokens 或 None)])
  * role ∈ system／user／assistant／ipython（工具結果）。msgs 只往後加；第 k 次呼叫的輸入＝msgs[:m] 加 assistant 開頭。
"""
from __future__ import annotations

import ast
import json
import os

MAX_MSG_CHARS_OH = None   # 由軌跡 metadata.llm_config.max_message_chars 決定（這個 run 是 30000）


def _tool_calls_text(tcs):
    out = []
    for tc in tcs or []:
        fn = tc.get("function") or {}
        out.append(json.dumps({"name": fn.get("name"), "arguments": fn.get("arguments")}, ensure_ascii=False))
    return "\n".join(out)


# ------------------------------------------------------------------ mini-SWE-agent
def load_miniswe(d):
    """d：資料集目錄（有 trajectories/*.traj.json）。"""
    out, stats = [], dict(n_traj=0, n_ok=0, calls_ok=0, calls_formaterror=0, api_eq=0, reasons={})
    tdir = os.path.join(d, "trajectories")
    for fn in sorted(os.listdir(tdir)):
        if not fn.endswith(".traj.json"):
            continue
        stats["n_traj"] += 1
        t = json.load(open(os.path.join(tdir, fn)))
        ms = t["messages"]
        msgs, calls = [], []
        last_ts = None
        n_fe = n_as = 0
        why = None
        for m in ms:
            role, e = m["role"], (m.get("extra") or {})
            if role == "assistant":
                r = e.get("response") or {}
                if "timestamp" not in e or r.get("created") is None or not (r.get("usage") or {}).get("prompt_tokens"):
                    why = why or "assistant_missing_ts_or_usage"
                tend = float(e.get("timestamp", "nan"))
                treq = max(last_ts, float(r.get("created"))) if last_ts is not None else float(r.get("created", "nan"))
                calls.append(dict(m=len(msgs), t_req=treq, t_end=tend, ptok=(r.get("usage") or {}).get("prompt_tokens")))
                text = (m.get("content") or "")
                tct = _tool_calls_text(m.get("tool_calls"))
                msgs.append(("assistant", text + ("\n" + tct if tct else "")))
                n_as += 1
            elif role == "tool":
                msgs.append(("ipython", m.get("content") or ""))
            elif role in ("system", "user"):
                if role == "user" and e.get("interrupt_type") == "FormatError":
                    n_fe += 1
                msgs.append((role, m.get("content") or ""))
            elif role == "exit":
                continue                       # 結束標記，不是送給模型的訊息
            else:
                why = why or f"role_{role}"
            if "timestamp" in e:
                last_ts = float(e["timestamp"])
        stats["calls_formaterror"] += n_fe
        stats["api_eq"] += int(t["info"]["model_stats"]["api_calls"] == n_as + n_fe)
        if why is None and len(calls) < 2:
            why = "lt2_calls"
        if why:
            stats["reasons"][why] = stats["reasons"].get(why, 0) + 1
            continue
        stats["n_ok"] += 1
        stats["calls_ok"] += len(calls)
        out.append(dict(sid=t.get("instance_id") or fn, msgs=msgs, calls=calls))
    return out, stats


# ------------------------------------------------------------------ OpenHands（CodeActAgent、function calling）
def _oh_source(src_dir, commit):
    """從 function_calling.py 的 AST 取 SYSTEM_PROMPT 與工具定義（字面值；描述的變數名解析成模組層的字串常數）。不執行原始碼。"""
    tree = ast.parse(open(os.path.join(src_dir, commit, "function_calling.py")).read())
    consts, tools = {}, {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name, v = node.targets[0].id, node.value
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                consts[name] = v.value
            elif isinstance(v, ast.Call) and getattr(v.func, "id", "") == "ChatCompletionToolParam":
                fkw = {k.arg: k.value for k in v.keywords}
                fn = fkw["function"]
                kw = {k.arg: k.value for k in fn.keywords}
                desc = kw.get("description")
                desc = consts[desc.id] if isinstance(desc, ast.Name) else ast.literal_eval(desc)
                spec = {"name": ast.literal_eval(kw["name"]), "description": desc}
                if "parameters" in kw:
                    spec["parameters"] = ast.literal_eval(kw["parameters"])
                tools[spec["name"]] = {"type": "function", "function": spec}
    return consts["SYSTEM_PROMPT"], tools


def _trunc(s, n):
    if n is None or n == -1 or len(s) <= n:
        return s
    h = n // 2
    return s[:h] + "\n[... Observation truncated due to length ...]\n" + s[-h:]


def _oh_obs_text(e, maxc):
    """照 codeact_agent.get_observation_message（原始碼 61874438）。"""
    o, c, ex = e.get("observation"), e.get("content") or "", e.get("extras") or {}
    if o == "run":
        return "OBSERVATION:\n" + _trunc(c + (ex.get("interpreter_details") or ""), maxc) + f"\n[Command finished with exit code {ex.get('exit_code')}]"
    if o == "run_ipython":
        return _trunc("OBSERVATION:\n" + c, maxc)
    if o == "error":
        return "OBSERVATION:\n" + _trunc(c, maxc) + "\n[Error occurred in processing last action]"
    return "OBSERVATION:\n" + _trunc(c, maxc)


def load_openhands(path, src_dir, tool_names=("execute_bash", "finish", "str_replace_editor")):
    out = []
    stats = dict(n_traj=0, n_ok=0, calls_ok=0, calls_no_usage=0, reasons={}, commits={}, tools=list(tool_names))
    from datetime import datetime

    def ts(e):
        return datetime.fromisoformat(e["timestamp"]).timestamp()
    srcs = {}
    with open(path) as fh:
        for ln in fh:
            r = json.loads(ln)
            stats["n_traj"] += 1
            commit = r["metadata"]["git_commit"]
            stats["commits"][commit] = stats["commits"].get(commit, 0) + 1
            if commit not in srcs:
                srcs[commit] = _oh_source(src_dir, commit)
            sysp, tools = srcs[commit]
            maxc = r["metadata"]["llm_config"].get("max_message_chars")
            msgs = [("system", sysp + "\n\n" + json.dumps([tools[n] for n in tool_names], ensure_ascii=False))]
            calls, seen = [], set()
            last_t = None
            why = None
            for e in r["history"]:
                tm = e.get("tool_call_metadata")
                if "action" in e:
                    if e.get("source") == "user":
                        msgs.append(("user", (e.get("args") or {}).get("content") or e.get("message") or ""))
                    elif tm:
                        mr = tm.get("model_response") or {}
                        rid = mr.get("id")
                        if rid not in seen:
                            seen.add(rid)
                            msg = ((mr.get("choices") or [{}])[0].get("message")) or {}
                            pt = (mr.get("usage") or {}).get("prompt_tokens")
                            calls.append(dict(m=len(msgs), t_req=last_t, t_end=ts(e), ptok=pt))
                            if pt is None:
                                stats["calls_no_usage"] += 1
                            tct = _tool_calls_text(msg.get("tool_calls"))
                            msgs.append(("assistant", (msg.get("content") or "") + ("\n" + tct if tct else "")))
                    elif e.get("source") == "agent":   # 沒有 tool call 的 agent 訊息（也是一次 LLM 呼叫；沒有 usage）
                        calls.append(dict(m=len(msgs), t_req=last_t, t_end=ts(e), ptok=None))
                        stats["calls_no_usage"] += 1
                        msgs.append(("assistant", e.get("message") or (e.get("args") or {}).get("content") or ""))
                elif "observation" in e:
                    if e.get("observation") in ("agent_state_changed", "null"):
                        pass
                    else:
                        msgs.append(("ipython" if tm else "user", _oh_obs_text(e, maxc)))
                last_t = ts(e)
            if any(c["t_req"] is None for c in calls):
                why = "call_without_prior_event"
            elif len(calls) < 2:
                why = "lt2_calls"
            if why:
                stats["reasons"][why] = stats["reasons"].get(why, 0) + 1
                continue
            stats["n_ok"] += 1
            stats["calls_ok"] += len(calls)
            out.append(dict(sid=r["instance_id"], msgs=msgs, calls=calls))
    return out, stats
