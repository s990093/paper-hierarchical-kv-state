# English abstract (for venue scouting) and candidate venues

> F5, 2026-10-10 (updated the same day with F1–F4 and F6). Draft: [draft.md](draft.md). Outline: [outline.md](outline.md). Self-review only; no external or cross-model review yet.
> Every number below maps to a run in draft.md Appendix B (see "Number sources" after the abstract).

## Title (working)

**Deferral Catches Up: Write-Time KV Cache Placement Under Read-Adaptive Restore**

## Abstract (249 words)

Tiered KV-cache managers for returning LLM sessions decide where each chunk lives: fast CPU memory, slow SSD, or nowhere, recomputed on return. Several recent systems decide at write time, often by token position. Read-adaptive restore (Cake) exploits position at read time, recomputing the prefix while loading the suffix. We ask whether write-time placement matters when reads adapt. On one AMD MI300X with Llama-3.1-8B, we use a KV harness with bit-exact restores, rate-limited CPU and SSD tiers, and a GPU-validated simulator, and compare write-time placement against its deferred twins: the same rule applied at eviction, in the background, or by pre-cleaning, plus write-through. First, Cake helps only within a bandwidth band: 1.38–1.93x at 0.33–3.73 GiB/s for 32K tokens, but 1.05x at 11.6 GiB/s. Our harness recomputes 1.7–2x slower than default vLLM, so the band's absolute position must be re-measured. Second, write-time placement never beat all deferred twins by 5% in 38 GPU comparisons, because position is equally visible at eviction. Deferral loses only under a sustained SSD write backlog with a bounded queue: a pre-registered rerun on ten fresh seeds found 0 of 216 winning cells without sustained backlog, and only one of five earlier overloaded winners, at an assumed shared-SSD bandwidth, reproduced, on the median only. Third, the winning policy depends on session return order more than on hardware. Separately, vLLM 0.28 on ROCm swaps its attention kernel whenever a KV connector is configured, which explains most of its offload slowdown; a lossless K/V-split fix removes 45–74% of it.

**Number sources** (keys in draft.md Appendix B; all recomputed from the CSVs by F5 unless noted):
- 1.38–1.93x, 0.33–3.73 GiB/s, 1.05x, 11.6 GiB/s, 32K → A0 (`20261008-133544-m7-a0`, `summary_a0_speedup.csv`).
- 1.7–2x → harness compute-only 645 / 3,759 ms (A0, `summary_a0.csv`) vs default vLLM 0.318 / 2.244 s (F1, `results/m9_followup/f1_metrics.csv`, `cfg`=off); arithmetic, setups not identical (draft §3.1).
- 38 comparisons = 17 (B, `summary_b_verdict.csv`) + 16 (B2, `b2_verdict.csv`, `src`=gpu) + 5 (R1GPU, `r1_gpu_verdict.csv`, `stat`=median); 5% is the pre-registered threshold (10 §2, 11).
- 0 of 216, ten fresh seeds, one of five → F6 (`20261010-081031-f6-summary`; `f6_summary.csv`, `f6_c_overload.csv`; simulation, pre-registered 2026-10-10T07:57:08Z, code frozen).
- Kernel swap, 45–74% → F1 (`f1_repro.csv`, `f1_decomp.csv`, `f1_profile_summary.csv`; 3 seeds; upstream issue vLLM #60316). F5 recomputed the reproduction table, seed 1 of the decomposition and the profiler times; the 3-seed medians (45–74%) are as reported by F1.

**Scope sentence to keep in any version**: one MI300X; GPU write-side tests on one GQA model; the harness recomputes about 1.7–2x slower than default vLLM, and in simulation faster recompute is where write-time placement comes closest to winning (3 of 30 configs survive under a ≥3/5-seed rule, none GPU-tested); rate-limited tiers; single-request GPU harness, concurrency only in simulation (no GPU validation); the vLLM drop count is from one seed; energy (F2) is one seed, GPU-only.

**Caveats a reviewer will find**:
- The concurrency simulation's original pre-registered verdict was 5 of 192 cells passing; it became 0 only after two twin-modeling errors were fixed post hoc. F6 then re-pre-registered the corrected setup and confirmed it on fresh seeds (0 of 216). Report all three.
- Phase 1 called the harness "not a strawman" by comparing it to a vLLM run that had silently fallen back to the slow attention kernel. That claim is withdrawn (draft §3.1).
- The original paper's numbers (`main.tex`) are under audit: of 241, 95 consistent, 59 inconsistent, 17 without source, 70 questionable (F4, `docs/audit_20261010/DATA_AUDIT.md`). This paper uses none of them.

## Candidate venues (checked against `VENUES.md`, 2026-10-08 section)

`VENUES.md` §10 says plain measurement results fit ICPE workshops or EuroMLSys better than HotOS, and recommends route W (a 6-page workshop paper). ✅ = verified on the official page in VENUES.md; 🔶 = estimated or third-party, must re-check.

| # | Venue | Where / when | Deadline | Length | Fit for this paper | Risk |
|:--|:--|:--|:--|:--|:--|:--|
| 1 | **ICPE 2027 workshops or Emerging Research track** (e.g. HotCloudPerf) | Gothenburg, 5/24–28, 2027 ✅ | 🔶 TBA; ICPE 2025 workshops closed early Jan to early Feb | 🔶 HotCloudPerf 2026 was 5 pages + 1 page of references | Best fit: a performance-measurement venue that accepts negative and characterization results; matches "measurement paper" framing and the user's Nordic preference | Track details still "TBA" in VENUES.md; same week as HotOS, so pick one |
| 2 | **EuroMLSys 2027** (workshop with EuroSys) | Rabat, ~4/19, 2027 🔶 | 🔶 ~Feb 2027; CFP not out | 6 pages | ML-systems audience knows vLLM, LMCache, Cake; a deferral-test negative result is in scope | Archival (ACM DL): a later full paper needs ≥25% new content; acceptance rate unknown |
| 3 | **HotOS 2027** | Burghausen, 5/24–26, 2027 ✅ | ✅ 2/1, notification 4/9 | 5 pages, references excluded ✅ | Only if reframed as a position: "stop deciding at write time; evaluate early-decision designs with a deferral test" | ~20–26% acceptance (VENUES.md count); wants new directions, not measurements; in-person attendance required |

**Backup**: ISC 2027 Workshops with Proceedings (Hamburg, 6/11; 🔶 deadline ~March, set per workshop; published in FGCS). Workshop list is fixed on 11/30.

**Not recommended for this paper**: CCGrid 2027 (12/1, 10 pages): the negative result alone does not fill a full paper, and a CCGrid submission blocks the ICPE-workshop and HotOS options until 2/1 (VENUES.md §10).

**Before submitting anywhere** (F1, F4 and the D1 rerun are now done):
- Re-measure recompute f(i) at default-vLLM speed (or give the harness an equally fast attention kernel), then redo the Cake band (A0) and rerun the faster-recompute simulations with the full twin set and a fixed ≥3/5 seed rule; GPU-confirm any survivor.
- Redo Appendix A with a zero-context reviewer (needs user approval, CLAUDE.md §5).
- Run `citation-audit` (AsymCache has no card; several venues unverified).
- Decide whether the vLLM kernel finding (F1) goes in the main text or an appendix; it is an engineering result, not evidence for write-time placement.
