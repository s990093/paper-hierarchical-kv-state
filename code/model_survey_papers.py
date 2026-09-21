#!/usr/bin/env python3
"""「主流 paper 常見模型」的證據：統計近期 KV cache / 長上下文論文實際評測用了哪些模型。

做法：HF papers 搜尋 → 過濾 2025-01-01 以後且標題/摘要與 KV/長上下文相關 →
抓 arXiv HTML 全文 → 以正規式判定「該篇是否提到某模型」（每篇只算一次）。
全文存在 <out>/text/<arxiv_id>.txt，命中表存 <out>/mentions.csv，可逐篇複查。

限制（寫進輸出）：提到 ≠ 評測用；related work 裡順帶提到的也會被算進去。
所以只拿來排「常見程度」，不拿來宣稱「某篇用了某模型」。
"""
import argparse, csv, html, json, os, re, sys, time, urllib.parse, urllib.request

QUERIES = [
    "KV cache offloading", "KV cache compression", "KV cache eviction", "KV cache quantization",
    "KV cache management long context", "prefix caching LLM serving", "long-context LLM inference",
    "sparse attention long context inference", "KV cache retrieval decoding", "hierarchical KV cache storage",
    "KV cache recomputation", "context caching LLM", "million token context inference",
    "hybrid linear attention long context", "KV cache CPU offload", "KV cache SSD",
]
RELEVANT = re.compile(r"\bKV\b|key[- ]value cache|KV-?cache|long[- ]context|prefix[- ]cach|context cach|million[- ]token|1M[- ]token", re.I)

# (家族, 正規式)。刻意寫得保守：要有尺寸才算，避免 "Llama" 這種泛稱灌票。
MODELS = [
    ("Llama-2-7B",            r"llama[- ]?2[- ](?:chat[- ])?7b|llama-2-7b"),
    ("Llama-2-13B",           r"llama[- ]?2[- ](?:chat[- ])?13b"),
    ("Llama-2-70B",           r"llama[- ]?2[- ](?:chat[- ])?70b"),
    ("LongChat-7B-32K",       r"longchat[- ]?(?:v1\.5[- ])?7b"),
    ("Llama-3-8B",            r"(?:meta[- ])?llama[- ]?3[- ]8b(?![- ]?(?:instruct[- ])?(?:gradient[- ])?1048k)"),
    ("Llama-3-8B-1048K",      r"llama[- ]?3[- ]8b[- ]?(?:instruct[- ])?(?:gradient[- ])?1048k|gradient[- ]?(?:ai[- ])?(?:llama[- ]?3[- ]8b)?.{0,20}1048k"),
    ("Llama-3-70B",           r"llama[- ]?3[- ]70b"),
    ("Llama-3.1-8B",          r"llama[- ]?3\.1[- ]8b"),
    ("Llama-3.1-70B",         r"llama[- ]?3\.1[- ]70b"),
    ("Llama-3.1-405B",        r"llama[- ]?3\.1[- ]405b"),
    ("Llama-3.2-1B/3B",       r"llama[- ]?3\.2[- ][13]b"),
    ("Llama-3.3-70B",         r"llama[- ]?3\.3[- ]70b"),
    ("Llama-4-Scout",         r"llama[- ]?4[- ]scout"),
    ("Llama-4-Maverick",      r"llama[- ]?4[- ]maverick"),
    ("UltraLong-8B-1M/4M",    r"ultralong"),
    ("Mistral-7B",            r"mistral[- ]7b"),
    ("Mistral-Small/Nemo",    r"mistral[- ](?:small|nemo)"),
    ("Mixtral-8x7B",          r"mixtral[- ]?8x7b"),
    ("Mixtral-8x22B",         r"mixtral[- ]?8x22b"),
    ("Qwen2-7B",              r"qwen[- ]?2[- ]7b"),
    ("Qwen2.5-3B",            r"qwen[- ]?2\.5[- ]3b"),
    ("Qwen2.5-7B",            r"qwen[- ]?2\.5[- ]7b(?![- ]?instruct[- ]1m)"),
    ("Qwen2.5-7B-1M",         r"qwen[- ]?2\.5[- ]7b[- ]instruct[- ]1m|qwen2\.5[- ]7b[- ]1m"),
    ("Qwen2.5-14B",           r"qwen[- ]?2\.5[- ]14b"),
    ("Qwen2.5-32B",           r"qwen[- ]?2\.5[- ]32b"),
    ("Qwen2.5-72B",           r"qwen[- ]?2\.5[- ]72b"),
    ("QwQ-32B",               r"qwq[- ]32b"),
    ("Qwen3-4B",              r"qwen[- ]?3[- ]4b"),
    ("Qwen3-8B",              r"qwen[- ]?3[- ]8b"),
    ("Qwen3-14B",             r"qwen[- ]?3[- ]14b"),
    ("Qwen3-32B",             r"qwen[- ]?3[- ]32b"),
    ("Qwen3-30B-A3B",         r"qwen[- ]?3[- ](?:coder[- ])?30b[- ]a3b"),
    ("Qwen3-235B-A22B",       r"qwen[- ]?3[- ]235b"),
    ("Qwen3-Next-80B-A3B",    r"qwen[- ]?3[- ]next"),
    ("Qwen3-Coder-480B",      r"qwen[- ]?3[- ]coder[- ]480b"),
    ("Qwen3.5/3.6",           r"qwen[- ]?3\.[5-8][- ]"),
    ("DeepSeek-V2-Lite",      r"deepseek[- ]v2[- ]lite"),
    ("DeepSeek-V2/V2.5",      r"deepseek[- ]v2(?![- ]lite)"),
    ("DeepSeek-V3/V3.x",      r"deepseek[- ]v3"),
    ("DeepSeek-R1(671B)",     r"deepseek[- ]r1(?![- ]distill)"),
    ("R1-Distill-Llama-8B",   r"r1[- ]distill[- ]llama[- ]8b"),
    ("R1-Distill-Qwen-7B",    r"r1[- ]distill[- ]qwen[- ]7b"),
    ("R1-Distill-Qwen-14B/32B", r"r1[- ]distill[- ]qwen[- ](?:14|32)b"),
    ("gpt-oss-20b",           r"gpt[- ]oss[- ]20b"),
    ("gpt-oss-120b",          r"gpt[- ]oss[- ]120b"),
    ("GLM-4-9B-1M",           r"glm[- ]?4[- ]9b"),
    ("GLM-4.5/4.6/4.7",       r"glm[- ]?4\.[5-7]"),
    ("Yi-9B/34B-200K",        r"yi[- ](?:9|34|6)b"),
    ("Phi-3/3.5-mini",        r"phi[- ]3(?:\.5)?[- ]mini"),
    ("Phi-3.5-MoE",           r"phi[- ]3\.5[- ]moe"),
    ("Phi-4",                 r"phi[- ]4\b"),
    ("Gemma-2",               r"gemma[- ]?2[- ](?:2|9|27)b"),
    ("Gemma-3",               r"gemma[- ]?3[- ](?:1|4|12|27)b"),
    ("Gemma-4",               r"gemma[- ]?4"),
    ("InternLM2.5-7B-1M",     r"internlm[- ]?2\.5[- ]7b"),
    ("LWM-Text-1M",           r"\blwm[- ]text"),
    ("OPT-6.7B~175B",         r"\bopt[- ](?:6\.7|13|30|66|175)b"),
    ("Jamba",                 r"\bjamba"),
    ("Mamba/Mamba2",          r"\bmamba[- ]?2?[- ](?:130m|370m|790m|1\.4b|2\.8b|7b)|\bmamba-2\b"),
    ("Nemotron-H",            r"nemotron[- ]h\b|nemotron[- ]h[- ]"),
    ("Nemotron-3",            r"nemotron[- ]3[- ](?:nano|super|ultra)"),
    ("Kimi-Linear",           r"kimi[- ]linear"),
    ("Kimi-K2",               r"kimi[- ]k2"),
    ("MiniMax-M1/Text-01",    r"minimax[- ](?:m1|text[- ]01|01)"),
    ("MiniMax-M2",            r"minimax[- ]m2"),
    ("Falcon-H1",             r"falcon[- ]h1"),
    ("Zamba2",                r"zamba"),
    ("Granite-4.0-H",         r"granite[- ]4"),
    ("Seed-OSS-36B",          r"seed[- ]oss"),
    ("Hunyuan-A13B",          r"hunyuan[- ]a13b"),
]
MODEL_RES = [(n, re.compile(p, re.I)) for n, p in MODELS]

def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "tiara-model-survey (research; contact via repo)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")

def strip_html(s):
    s = re.sub(r"(?is)<(script|style|math)[^>]*>.*?</\1>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = html.unescape(s)
    return re.sub(r"\s+", " ", s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--since", default="2025-01-01")
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "text"), exist_ok=True)
    papers = {}
    for q in QUERIES:
        try:
            res = json.loads(get("https://huggingface.co/api/papers/search?q=" + urllib.parse.quote(q)))
        except Exception as e:
            print("search fail", q, e); continue
        for x in res:
            p = x.get("paper", x)
            pid, pub = p.get("id"), (p.get("publishedAt") or "")[:10]
            if not pid or pub < args.since: continue
            blob = (p.get("title") or "") + " " + (p.get("summary") or "")
            if not RELEVANT.search(blob): continue
            papers.setdefault(pid, {"id": pid, "published": pub, "title": re.sub(r"\s+", " ", p.get("title") or ""), "queries": []})
            papers[pid]["queries"].append(q)
        time.sleep(0.5)
    print(f"candidate papers: {len(papers)}", flush=True)
    json.dump(papers, open(os.path.join(args.out, "papers.json"), "w"), indent=1, ensure_ascii=False)

    rows = []
    for i, (pid, meta) in enumerate(sorted(papers.items())):
        tp = os.path.join(args.out, "text", f"{pid}.txt")
        status = "cached"
        if not os.path.exists(tp):
            try:
                txt = strip_html(get(f"https://arxiv.org/html/{pid}"))
                if len(txt) < 5000:
                    status = "html_too_short"
                open(tp, "w").write(txt)
                status = status if status != "cached" else "ok"
            except Exception as e:
                status = f"html_error:{e!r}"[:80]
                open(tp, "w").write("")
            time.sleep(args.sleep)
        txt = open(tp).read()
        hits = [n for n, rx in MODEL_RES if rx.search(txt)]
        rows.append({"arxiv_id": pid, "published": meta["published"], "title": meta["title"][:150],
                     "fetch_status": status, "text_chars": len(txt), "n_models": len(hits), "models": ";".join(hits)})
        if i % 25 == 0:
            print(f"[{i}/{len(papers)}] {pid} {status} chars={len(txt)} models={len(hits)}", flush=True)
    with open(os.path.join(args.out, "mentions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    usable = [r for r in rows if r["text_chars"] >= 5000]
    from collections import Counter
    cnt = Counter(m for r in usable for m in r["models"].split(";") if m)
    cnt25 = Counter(m for r in usable if r["published"] < "2026-01-01" for m in r["models"].split(";") if m)
    cnt26 = Counter(m for r in usable if r["published"] >= "2026-01-01" for m in r["models"].split(";") if m)
    n25 = sum(1 for r in usable if r["published"] < "2026-01-01"); n26 = len(usable) - n25
    with open(os.path.join(args.out, "model_frequency.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "papers_all", "share_all", "papers_2025", "share_2025", "papers_2026", "share_2026", "n_usable_all", "n_2025", "n_2026"])
        for m, c in cnt.most_common():
            w.writerow([m, c, f"{c/len(usable):.3f}", cnt25[m], f"{cnt25[m]/max(n25,1):.3f}", cnt26[m], f"{cnt26[m]/max(n26,1):.3f}", len(usable), n25, n26])
    print(f"usable full texts: {len(usable)} (2025: {n25}, 2026: {n26})")
    for m, c in cnt.most_common(45):
        print(f"  {m:<26} {c:>4}  {c/len(usable):6.1%}   2025 {cnt25[m]/max(n25,1):6.1%}  2026 {cnt26[m]/max(n26,1):6.1%}")

if __name__ == "__main__":
    main()
