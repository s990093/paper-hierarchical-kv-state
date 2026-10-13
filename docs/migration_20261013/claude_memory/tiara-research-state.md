---
name: tiara-research-state
description: Where the write-time KV placement research stands as of 2026-10-10 and what comes next
metadata:
  type: project
---
As of 2026-10-10: phase 1 (07), 08 ablation (09) and breakthrough round 1 (10/11/12) all say write-time position placement (S5) is matched by simpler lazy/background/write-through strategies on MI300X + Llama-3.1-8B (GQA). The one sim survivor (Llama, doc, CPU 11.6 GiB/s, free, local SSD, 50%) failed GPU confirmation 0/5 seeds (sim was ~5 pts optimistic for S5L vs ~+1% for twins), so sim screens need a ≥10% margin or GPU for every survivor. Entry doc: `docs/phase1_20261008/FULL_REPORT.md`.
**Why:** a zero-context red-team review (docs/research_20261009_explore/REDTEAM_round1.md) showed the negative result partly rests on a modeling assumption: harness/simulator serve one request at a time, so lazy strategies never lack idle time.
**How to apply:** next step (pending advisor decision) is to build a concurrency + bounded write queue + "in-flight chunks can't be evicted / drop when full" model (H3), measure truncation frequency in real vLLM 0.28 first, then test position-aware truncation vs Pensieve-style drop, AsymCache, cancellable write-back. Cite Krul, TierKV, Lachesis, Pensieve, AsymCache as prior work. See [[deferral-test-rule]].
2026-10-10: launched 8 parallel direction probes D1–D8 (concurrency sim, real vLLM drop rate [only GPU user], κ map, trace oracle, arch/modality, precision tier, OS page cache/IO, wildcard). Outputs go to `docs/research_20261010_directions/`, with the pre-registration in its README.md.
2026-10-10 later: D1–D8 done (SUMMARY.md: write-time placement dead from 8 angles). The user said "都做掉", so follow-ups F1–F5 were launched (docs/research_20261010_followup/README.md):
- F1: why vLLM CPU offload slows TTFT
- F2: energy κ
- F3: dequant cost and whether FP8 took effect
- F4: data-integrity audit of main.tex numbers
- F5: negative-result paper draft in docs/paper_negative_20261010/
2026-10-10 F1–F6 done (docs/research_20261010_followup/SUMMARY.md, commit 1ce077b):
- F1: vLLM 0.28 ROCm falls back ROCM_ATTN→TRITON_ATTN whenever a KV connector is set; that caused the offload slowdown, and a K/V-split fix exists.
- F2: energy-optimal ≠ time-optimal, but this is narrow.
- F3: D6 is dead; the September FP8 run was real.
- F4: main.tex has 59/241 inconsistent numbers.
- F5: draft in docs/paper_negative_20261010.
- F6: D1 confirmed, 0/216.

KEY OPEN ITEM: the harness recomputes 1.7–2× slower than default vLLM (C6 compared against the slow path), so real f ≈ f×0.5–0.6. That is exactly 09's only unexcluded exception (cheap recompute). Re-measure f(i) with a fast attention backend, then rerun with the full twin set plus a GPU check before claiming the negative result.

