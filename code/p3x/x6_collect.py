#!/usr/bin/env python3
"""X6: collect GitHub search results and the hand-curated issue classification.

Read-only on its inputs (the search run directories). Writes three CSVs:

  results/p3_explore/x6/search_queries.csv  one row per (batch, repo, query): exit code, #hits
  results/p3_explore/x6/search_hits.csv     deduplicated hits over all queries
  results/p3_explore/x6/issues.csv          hand-classified reports (see doc X6 section 2.2)

The classification in CURATED is a manual judgement made while reading each thread
(saved under <search run>/threads/).  It is data, not a computed number: the doc
X6_real_systems.md section 2.2 defines the categories.  Numbers quoted in the
`symptom` column are what the reporter wrote, not our measurements.

Usage (stdlib only, run with `python -I`):
  python -I code/p3x/x6_collect.py --run <batch-1 run dir> <batch-2 run dir> --out results/p3_explore/x6
"""
import argparse
import csv
import datetime as dt
import glob
import json
import os
import sys
from collections import defaultdict


def iso_mtime(path):
    return dt.datetime.fromtimestamp(os.path.getmtime(path), dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def parse_batches(runs):
    """Return list of (run_dir, batch, json_subdir, stdout_rows); batch 1 uses json/, batch 2 json2/."""
    out = []
    for run in runs:
        logp = os.path.join(run, "stdout.log")
        if not os.path.exists(logp):
            raise SystemExit("missing %s" % logp)
        sub, batch = ("json2", "2") if os.path.isdir(os.path.join(run, "json2")) else ("json", "1")
        rows = []
        for line in open(logp):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            rows.append(parts)
        out.append((run, batch, sub, rows))
    return out


def safe_name(repo, q):
    return (repo + "__" + q).replace("/", "_").replace(" ", "_")


# ---------------------------------------------------------------------------
# Hand-curated classification.  group = independence group (section 2.2 item 2).
# category: P-hint | P-pred | K | K-wish | B | C | S | U | feature (no symptom)
# counted: yes only for P-* rows that are the first row of their group and are not K.
# user_predicts: the knob asked for needs the user to supply a return-time value.
# ---------------------------------------------------------------------------
CURATED = [
    # id, group, system, repo, number, kind, created, author, evidence, category,
    # user_predicts, knob_now, knob_solves, counted, found_via, symptom, reason
    ("R01", "G-llmd", "vLLM", "vllm-project/vllm", "37003", "rfc", "2026-03-13", "vMaroon",
     "design+citations", "P-hint", "yes", "none in vLLM core (PR #38514 open)", "no", "yes", "search",
     "agent tool-call pauses leave blocks unreferenced; under concurrent load other agents evict them via LRU; resume = full recompute of 70K-200K tokens (RFC's numbers)",
     "orchestrator knows which sessions are live; asks priority+duration directives; duration is set by orchestrator"),
    ("R02", "G-llmd", "vLLM", "vllm-project/vllm", "38514", "pr", "2026-03-30", "hyeongyun0916",
     "benchmark (reporter's)", "P-hint", "yes", "not merged", "no", "no", "link from R01",
     "retention API e2e vs LRU: server TTFT -43%/-12%/-12% at in-flight 1/4/8 (gpt-oss-120b); MI250 agentic synthetic with 5 s tool delay (reporter's numbers)",
     "implementation of R01, co-authored by vMaroon -> same group; also rejects priorities that increase with token position (prefix dependency)"),
    ("R03", "G-llmd", "llm-d", "llm-d/llm-d-kv-cache", "395", "issue", "2026-03-05", "vMaroon",
     "design", "P-hint", "yes", "n/a", "no", "no", "search",
     "PoC plan for priority+TTL retention directives, benchmark with concurrent multi-turn ReAct sessions",
     "same author/initiative as R01; closed not_planned (moved to vLLM RFC)"),
    ("R04", "G-kcross", "vLLM/llm-d", "vllm-project/vllm", "37003#issuecomment-5087831234", "comment", "2026-07-27", "k-cross",
     "simulation (llm-d simulator)", "P-pred", "yes", "none in vLLM", "no", "yes", "thread of R01",
     "LRU residency 6.8 s (512 blocks/pod, qps 5); for gap/residency>=4 LRU 0/480 zero-recompute pairs vs ~25% with router pins; SLRU and promotion hints 0%; pin TTL set as k x predicted gap (tool runtime mean/variance)",
     "value exists only when gap > LRU residency; a local policy (SLRU) cannot capture it; TTL must be predicted per tool -> P-pred (simulation, flagged)"),
    ("R05", "G-mgabor", "vLLM", "vllm-project/vllm", "37003#issuecomment-6090931030", "comment", "2026-10-09", "mgabor3141",
     "production traffic", "P-hint", "yes", "none in vLLM", "no", "yes", "thread of R01",
     "4x RTX 3090, vLLM 0.30 hybrid MoE, 10-20 coding agents, 50-250K contexts: 91% follow-ups <1 s; 1-5 s gaps still lose context under LRU when the pool is under pressure; turns queued >1 min almost all missed (1 of 56); proxy cannot tell engine which blocks to keep",
     "proxy knows the turn is queued; asks a hold whose duration covers queue wait -> P-hint with user-supplied duration"),
    ("R06", "G-natalie", "vLLM", "vllm-project/vllm", "37003#issuecomment-6049188103", "comment", "2026-10-07", "nataliekung",
     "statement", "U", "n/a", "none", "no", "no", "thread of R01",
     "'the tool-call-pause eviction problem matches what I have seen'; prototyped retention_deadline on free blocks",
     "no concrete symptom (section 2.2 item 1) -> not counted"),
    ("R07", "G-rich", "vLLM", "vllm-project/vllm", "37003#issuecomment-5078100306", "comment", "2026-07-25", "rich7420",
     "replay (reporter's)", "S", "n/a", "n/a", "n/a", "no", "thread of R01",
     "prefix_cache_queries counts per scheduling attempt (1.3-2.0x prompt tokens under queue pressure); hypothesis that misses are 15-40 s gaps / multi-replica routing misses",
     "measurement + routing, not return-time knowledge"),
    ("R08", "G-dynamo", "vLLM", "vllm-project/vllm", "51428", "rfc", "2026-08-07", "karen-sy",
     "design", "P-hint", "yes", "none in vLLM (kv_hints envelope only)", "no", "yes", "link from vllm#53421",
     "'local policy cannot tell a short tool gap from a long one'; 'the workload knows a request will resume soon'; proposes Retain/Pin(TTL)/Prefetch/Demote hints",
     "info lives in router/orchestrator; Dynamo team initiative (with sglang#27574, vllm#53421, dynamo#13279) counted once"),
    ("R09", "G-dynamo", "SGLang", "sgl-project/sglang", "27574", "rfc", "2026-06-08", "ishandhanani",
     "design", "P-hint", "yes", "SGLang TTL pin existed then; author proposed reverting it (#21045)", "no", "no", "search",
     "'the engine does not know why a trajectory will reuse a prefix or how long an external tool/subagent gap will last'",
     "same initiative as R08 (R08 cites it as aligned)"),
    ("R10", "G-dynamo", "SGLang", "sgl-project/sglang", "21045", "pr", "2026-03-20", "ishandhanani",
     "maintainer statement", "P-hint", "yes", "HiRadixCache TTL pin (#18941)", "no", "no", "search",
     "'Hard pinning couples routing to a specific worker and wastes host memory when the router can't honor sticky affinity at scale' -> proposes soft retention_seconds",
     "evidence that a fixed TTL pin knob is not enough; same group as R08"),
    ("R11", "G-dynamo", "Dynamo", "ai-dynamo/dynamo", "docs:thunderagent-program-scheduler.md#L95-L105", "doc", "2026-10-17", "Dynamo docs",
     "official doc", "P-hint", "no", "ThunderAgent program scheduler (pause at tool boundary)", "partial", "no", "code read",
     "'The engine evicts useful blocks under pressure or refuses admission, and every next turn pays a re-prefill tax'",
     "uses known program state (REASONING/ACTING), not predicted time; same org as R08"),
    ("R12", "G-fermat", "vLLM", "vllm-project/vllm", "52113", "rfc", "2026-08-13", "FermatGo",
     "design + use cases", "P-hint", "yes", "none in vLLM", "no", "yes", "link from vllm#53421",
     "'vLLM cannot distinguish a temporarily idle but valuable agent context from a prefix that will never be used again'; agent waits for tool -> protect for short TTL; resume -> prefetch",
     "agent runtime knows tool outstanding / branch finished"),
    ("R13", "G-stmat", "SGLang", "sgl-project/sglang", "24656", "rfc", "2026-05-08", "stmatengss",
     "design", "P-hint", "yes", "SGLang --enable-session-radix-cache (merged 2026-08-02) soft-protects active sessions", "partial", "no", "search",
     "'cache entries that are likely to be reused by upcoming steps may be evicted too early'; hints include expected_tool_duration_ms, cache_ttl_ms",
     "a session-level knob now exists in SGLang and nobody in the thread says it is insufficient -> strict: not counted"),
    ("R14", "G-stmat", "Mooncake", "kvcache-ai/Mooncake", "2098", "rfc", "2026-05-14", "stmatengss",
     "design", "P-hint", "yes", "soft pin with request-level TTL", "partial", "no", "search",
     "same text as sglang#24656 for Mooncake",
     "same author/initiative as R13"),
    ("R15", "G-mks", "SGLang", "sgl-project/sglang", "42434", "pr", "2026-10-03", "mks-hash",
     "benchmark (reporter's)", "P-hint", "no", "none (prefetch policy only controls when to stop an on-arrival prefetch)", "no", "yes", "search",
     "L3 restore starts only when /generate arrives although the orchestrator knows the exact prefix during a tool call; proactive restore: median continuation TTFT 577->166 ms at gaps >=500 ms (L4, Qwen2.5-1.5B, 4K tokens, n=3)",
     "prefetch-too-late; needs the 'next prefix + it will come' hint before arrival"),
    ("R16", "G-lin", "Mooncake", "kvcache-ai/Mooncake", "2819", "rfc", "2026-07-09", "Lin-z-w",
     "design", "K-wish", "no", "soft/hard pin exist", "partial", "no", "search",
     "session_id + retention_priority hints; 'KV objects of the same session related in lifecycle'",
     "about value/lifecycle grouping, not when a session returns; closed for inactivity"),
    ("R17", "G-lcc", "vLLM", "vllm-project/vllm", "47802", "issue", "2026-07-07", "lcc-star",
     "experiment (reporter's)", "K-wish", "no", "none in vLLM", "no", "no", "search",
     "request priority affects scheduling but not prefix-cache retention: cached tokens follow prompt group, not priority, under pressure",
     "a static priority knob would solve it (SGLang has one); no return-time info needed"),
    ("R18", "G-kang", "vLLM", "vllm-project/vllm", "23083", "issue", "2025-08-18", "kangyishuai",
     "use case", "K-wish", "no", "none (closed not_planned)", "no", "no", "search",
     "pin system-prompt / NL2SQL prefixes so they are never evicted",
     "fixed pin is enough -> K-type need, not P"),
    ("R19", "G-luolun", "vLLM", "vllm-project/vllm", "23641", "rfc", "2025-08-26", "luolun",
     "design", "U", "no", "n/a", "n/a", "no", "search",
     "frequency x compute-cost eviction instead of LRU",
     "past-access policy; no return information"),
    ("R20", "G-limark", "SGLang", "sgl-project/sglang", "36208", "rfc", "2026-08-24", "limarkdcunha",
     "design", "K-wish", "no", "priority (per request) exists", "partial", "no", "search",
     "per-token-range retention priority (system prompt vs user turn)",
     "static priority; modeled on TRT-LLM"),
    ("R21", "G-nafis", "LMCache", "LMCache/LMCache", "4456", "pr", "2026-08-07", "nafis271",
     "use case", "K", "no", "L2 pins (no TTL); PR adds TTL", "partial", "no", "search",
     "per-request retention TTL for L2 eviction: 'this prompt's KV survives the next hour' (paid prompt caching)",
     "fixed TTL product -> K"),
    ("R22", "G-cfb", "SGLang", "sgl-project/sglang", "32724", "issue", "2026-07-29", "cfbdsirlijun-maker",
     "observation", "S", "no", "fixed upstream (closed completed)", "yes", "no", "search",
     "HiCache L3 prefetch finished but finalized only when the FCFS waiting-queue scan reaches it -> seconds of extra TTFT under load",
     "scheduling coupling, not return-time knowledge"),
    ("R23", "G-yong", "SGLang", "sgl-project/sglang", "33984", "issue", "2026-08-07", "Yong-ee",
     "benchmark (reporter's)", "C", "no", "capacity (hicache ratio)", "yes", "no", "search",
     "HiCache L1+L2+Mooncake SSD: cache hit rate / TTFT vary with hicache ratio",
     "capacity/config"),
    ("R24", "G-dhx", "Mooncake", "kvcache-ai/Mooncake", "4272", "pr", "2026-09-22", "DHX98",
     "design", "S", "no", "promotion_on_hit (default off)", "partial", "no", "search",
     "SSD-only keys are read synchronously on get(); use exist() probe as an early 'will read soon' signal to prefetch to DRAM",
     "uses the probe->get window (request already arrived); no return-time prediction"),
    ("R25", "G-nike", "Mooncake", "kvcache-ai/Mooncake", "2769", "pr", "2026-07-06", "Nike232",
     "design", "S", "no", "n/a", "n/a", "no", "search",
     "promotion-on-hit tasks that start after the useful window waste SSD/network; add deadline budget",
     "I/O scheduling"),
    ("R26", "G-kfir", "vLLM", "vllm-project/vllm", "39021", "pr", "2026-04-05", "kfirtoledo",
     "code", "B", "no", "n/a", "n/a", "no", "search",
     "touch offloaded blocks once at request_finished in prefix order so earlier prefix blocks are MRU and later blocks evict first ('preserving KV cache prefix dependency')",
     "mechanism; shows tail-first eviction is intentional"),
    ("R27", "G-yl", "SGLang", "sgl-project/sglang", "34012", "pr", "2026-08-07", "yl3469",
     "benchmark (reporter's)", "K", "no", "--radix-eviction-policy tlru (merged 2026-09-17)", "yes", "no", "search",
     "on agentic traces plain LRU protects the tail at the expense of other conversations' tail latency by up to 43% (reporter's number)",
     "solved by a fixed-parameter rule (threshold, next_prompt_estimate)"),
    ("R28", "G-zmj", "SGLang", "sgl-project/sglang", "29173", "pr", "2026-06-24", "Zhangmj0621",
     "design", "K", "no", "--enable-session-radix-cache (merged 2026-08-02)", "yes", "no", "search",
     "radix cache 'may evict an active session's prefix while unrelated, unreferenced KV remains cached'",
     "the knob introduced by this PR addresses it using only 'session still open' (app calls /close_session)"),
    ("R29", "G-zmj", "SGLang", "sgl-project/sglang", "27574#issuecomment-4704224856", "comment", "2026-06-15", "Zhangmj0621",
     "statement", "C", "no", "admission control", "partial", "no", "thread of R09",
     "workflow-aware eviction alone is not sufficient: as trajectories grow, capacity causes mutual eviction; fixed by throttling batch size",
     "capacity/admission, not return-time knowledge"),
    ("R30", "G-dynamo", "Dynamo", "ai-dynamo/dynamo", "6571", "issue", "2026-02-25", "ishandhanani",
     "use case", "K-wish", "yes", "nvext.cache_control TTL (#6213) -> SGLang pin", "partial", "no", "search",
     "support Anthropic-style cache_control {ttl: 5m|1h} so Claude Code can pin its conversation prefix",
     "client picks the TTL (user predicts); same org as R08"),
    ("R31", "G-f0r", "SGLang", "sgl-project/sglang", "43738", "issue", "2026-10-12", "F0rtiter",
     "observation", "B", "no", "n/a", "n/a", "no", "search",
     "DeepSeek-V4.1-Flash Claude Code traffic: multi-turn prefix hit stuck at 11,520 tokens; 60-80K prefix re-prefilled every turn",
     "model-specific cache bug"),
    ("R32", "G-cmt", "vLLM", "vllm-project/vllm", "48435", "issue", "2026-07-12", "claudematttest-dev",
     "observation", "B", "no", "n/a", "n/a", "no", "search",
     "hybrid-SWA prefix caching collapses to zero for all requests in multi-session round-robin at ~25% pool occupancy (Gemma-4-31B)",
     "hybrid-model bug"),
    ("R33", "G-nato", "vLLM", "vllm-project/vllm", "44294", "issue", "2026-06-02", "natoscott",
     "benchmark (reporter's)", "S", "no", "n/a", "n/a", "no", "search",
     "OffloadingConnector serialises concurrent requests behind one load of a shared prefix: up to 12x TTFT",
     "load-path serialisation"),
    ("R34", "G-fja", "vLLM", "vllm-project/vllm", "52113#issuecomment-5371751492", "comment", "2026-08-21", "franciscojavierarceo",
     "benchmark (reporter's)", "S", "no", "KV-aware routing", "yes", "no", "thread of R12",
     "KV-aware routing raised mean KV hit rate 49.54% -> 99.08%, continuation latency 811 -> 315 ms (12-pair responses benchmark)",
     "routing, not eviction/return time"),
    ("R35", "G-ruocco", "llm-d", "llm-d/llm-d-router", "2088", "issue", "2026-07-20", "ruocco",
     "design (cites Continuum)", "K", "no", "turn-priority strategy (Beta) in llm-d router", "partial", "no", "search",
     "evicting a session's KV during tool pauses destroys multi-turn reuse; order queued flows by turn number. Guide benchmark (llm-d/llm-d guides/turn-priority-fairness README L604-L635, reporter's numbers): concurrency 128, 200 deep agentic sessions: fcfs finished 0 sessions, p50 586.7 s; turn-priority finished 17, p50 37.8 s ('each one holds a growing prefix that keeps getting evicted')",
     "addressed by a fixed-parameter scheduling heuristic (turn number), not by return-time info"),
    ("R36", "G-nafis", "TRT-LLM", "NVIDIA/TensorRT-LLM", "16251", "issue", "2026-07-10", "nafis271",
     "production (DeepInfra, reporter's)", "K", "no", "KvCacheRetentionConfig exists for GPU/host; asks disk tier + per-request retention period", "partial", "no", "search",
     "v1 KV manager has GPU+host only; want disk tier and 'a request can ask that its KV cache stay reusable for a given amount of time'",
     "same author as R21; fixed retention period product -> K"),
    ("R37", "G-wenbo", "production-stack", "vllm-project/production-stack", "244", "issue", "2025-03-07", "wenboown",
     "design", "S", "no", "LMCache sharing + routing", "partial", "no", "search",
     "agentic workflows: redundant KV recomputation across agents; wants cross-agent KV sharing and session-aware routing",
     "sharing/routing, not return time"),
    ("R38", "G-zhh", "vLLM", "vllm-project/vllm", "57103", "rfc", "2026-09-16", "zhhangBian",
     "design + cites vLLM AgentX traffic stats", "P-hint", "yes", "none in vLLM", "no", "yes", "search batch 2",
     "'During a pause, the KV is often the most valuable prefix for the next request, yet it may have no active request reference and enter an ordinary LRU eviction order'; cites AgentX post: median 43 turns/session, 142K-token input (RFC's numbers)",
     "router supplies session lifecycle; engine has no knob for it"),
    ("R39", "G-inbar", "vLLM", "vllm-project/vllm", "50422", "pr", "2026-07-30", "InbarShapira",
     "benchmark plan (ShareGPT multi-turn)", "K-wish", "no", "CPU offload lru/arc only (PR open)", "no", "no", "search batch 2",
     "LRU/ARC treat blocks independently; evicting a mid-chain block forces recompute of everything after it; session-aware eviction (session-worst-first, tail-first)",
     "structural (prefix-chain) policy, no return-time information"),
    ("R40", "G-dynamo", "vLLM", "vllm-project/vllm", "59708", "rfc", "2026-10-01", "karen-sy",
     "design", "feature", "n/a", "PR #59714 (G1 Retain via priority) open", "no", "no", "search batch 2",
     "KV hint primitives COPY_TO / MOVE_TO / SET_PRIORITY; Retain/Prefetch composed from them",
     "same initiative as R08"),
    ("R41", "G-llmd", "vLLM", "vllm-project/vllm", "48501", "rfc", "2026-07-13", "vMaroon",
     "design", "feature", "n/a", "session_id plumbing merged (#48048)", "partial", "no", "search batch 2",
     "session-centric orchestration above the engine; engine should expose session-coordinate KV events and accept intent-free directives",
     "same author as R01"),
]

CUR_FIELDS = ["report_id", "group", "system", "repo", "number", "kind", "created", "author",
              "evidence", "category", "user_predicts", "knob_now", "knob_solves", "counted",
              "found_via", "symptom", "reason"]


def url_for(repo, number):
    if number.startswith("docs:"):
        return "https://github.com/%s/blob/f24c59b936ba23c933cc35d2f8c7f9ff71f54c3d/docs/fern/pages/use-cases/agents/%s" % (
            repo, number[len("docs:"):])
    base, _, frag = number.partition("#")
    return "https://github.com/%s/issues/%s%s" % (repo, base, ("#" + frag) if frag else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, nargs="+", help="search run dirs (first one holds threads/)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run_ids = {r: os.path.basename(r.rstrip("/")) for r in a.run}
    run_id = run_ids[a.run[0]]
    os.makedirs(a.out, exist_ok=True)

    # 1) per-query table
    qrows = []
    hits = {}
    hit_queries = defaultdict(set)
    first_ts = {}
    hit_run = {}
    for run, batch, sub, rows in parse_batches(a.run):
        for repo, q, rc, n in rows:
            jp = os.path.join(run, sub, safe_name(repo, q) + ".json")
            ts = iso_mtime(jp) if os.path.exists(jp) else ""
            qrows.append([run_ids[run], ts, batch, repo, q, rc, n])
            if not os.path.exists(jp):
                continue
            try:
                data = json.load(open(jp))
            except Exception as e:  # keep going, but make it visible
                print("BADJSON", jp, e, file=sys.stderr)
                continue
            for it in data:
                u = it["url"]
                hits[u] = it
                hit_queries[u].add("b%s:%s" % (batch, q))
                first_ts.setdefault(u, ts)
                hit_run.setdefault(u, run_ids[run])
    with open(os.path.join(a.out, "search_queries.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "ts", "batch", "repo", "query", "exit_code", "n_results"])
        w.writerows(qrows)

    # 2) deduplicated hits
    with open(os.path.join(a.out, "search_hits.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "ts", "repo", "number", "is_pr", "state", "created", "author",
                    "comments", "n_queries", "title", "url", "queries"])
        for u, it in sorted(hits.items(), key=lambda kv: (kv[1]["repository"]["nameWithOwner"], kv[1]["number"])):
            w.writerow([hit_run[u], first_ts[u], it["repository"]["nameWithOwner"], it["number"],
                        it.get("isPullRequest"), it["state"], it["createdAt"][:10],
                        (it.get("author") or {}).get("login"), it.get("commentsCount"),
                        len(hit_queries[u]), it["title"], u, "|".join(sorted(hit_queries[u]))])

    # 3) curated table; ts = mtime of the saved thread JSON when we fetched it
    tdir = os.path.join(a.run[0], "threads")
    n_counted = 0
    groups = set()
    with open(os.path.join(a.out, "issues.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "ts"] + CUR_FIELDS + ["url", "thread_file"])
        for row in CURATED:
            d = dict(zip(CUR_FIELDS, row))
            base = d["number"].split("#")[0]
            cand = glob.glob(os.path.join(tdir, "%s_%s.json" % (d["repo"].replace("/", "_"), base)))
            if not cand:
                cand = glob.glob(os.path.join(tdir, "%s_%s_comments.json" % (d["repo"].split("/")[1], base)))
            if not cand:
                cand = glob.glob(os.path.join(tdir, "*_%s_comments.json" % base))
            tf = cand[0] if cand else ""
            ts = iso_mtime(tf) if tf else dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if d["counted"] == "yes":
                assert d["category"].startswith("P-"), d["report_id"]
                assert d["group"] not in groups, ("two counted rows in one group", d["group"])
                groups.add(d["group"])
                n_counted += 1
            w.writerow([run_id, ts] + [d[k] for k in CUR_FIELDS] + [url_for(d["repo"], d["number"]),
                                                                   os.path.basename(tf)])
    cats = defaultdict(int)
    for row in CURATED:
        cats[row[9]] += 1
    print("queries", len(qrows), "unique_hits", len(hits), "curated", len(CURATED),
          "counted_groups", n_counted, "categories", dict(sorted(cats.items())))
    print("counted:", sorted(groups))


if __name__ == "__main__":
    main()
