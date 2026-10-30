# Fig 1 / Fig 2 diagrams: design notes (2026-10-30)

Source: `diagrams.py` in this folder (matplotlib patches only). These replace the box-and-arrow drafts `fig1_system.svg` and `fig2_method.svg`, which are left in place. PAPER.md still embeds the `.svg` files; switching it to the PNGs is the PAPER.md owner's call.

## Reference figures I looked at (the images, not just the captions)
The arXiv HTML serves most of these figures as SVG, which this machine cannot rasterize. So I rendered the arXiv PDF pages locally with poppler, in the scratchpad only. No reference images are in the repo.
| Paper (arXiv) | Figures | What I borrowed |
|:--|:--|:--|
| Cake, *Compute Or Load KV Cache? Why Not Both?* (2410.03065) | Fig 1 (GPU/CPU/disk decision flow, Cake region shaded), Fig 2 (chunk grid: compute from the left, I/O from the right, merge in the middle) | the block bar in Fig 1(b): recompute from the first block, load from the last block, a "meet" marker |
| CacheFlow (2604.25080) | Fig 1 (system on the left, zoomed restoration panel on the right, dashed divider; orange = recompute, blue = load; legend at the bottom right) | the two-panel layout with a dashed divider, orange/blue for recompute/load, legend at the bottom right of Fig 1 |
| LMCache (2510.09665) | Fig 5 (three-layer stack), Fig 6 (one teal accent for the system's own parts, everything else white, zoom callout) | **one accent color**, used only for foresight; everything else neutral grey |
| Mooncake (2407.00079) | Fig 1 (pools as dash-dot regions), Fig 4 (steps s1–s4 written on the arrows; dotted vs. solid arrows) | numbered markers placed where the action happens; a dashed style for a different kind of action (here, an action that needs foresight) |
| CachedAttention (2403.19708) | Fig 5 (HBM / host memory / disk stack with labeled load/save/fetch/evict arrows), Fig 6–8 (two-lane execution/read timelines), Fig 9 (job queue with look-ahead prefetch and eviction windows over DRAM/disk rows) | tiers drawn as **swimlanes against a time axis**, with the "between requests" window between two event lines |
| HCache (2410.05004) | Fig 1 (restore-time bars, computation vs. I/O, for recompute / offload / HCache), Fig 7 (legend row; restore and save paths in different arrow colors) | the restore-time bars in Fig 1(b): load all / recompute all / Cake, with a dashed line where Cake finishes |
| Strata (2508.18572) | Fig 4 (architecture with short verbs on the arrows), Fig 6 | one- or two-word verbs on arrows (write, prefetch, restore) |
| *Where Should the KV Cache Live?* (2609.16215) | Fig 1 (three numbered panels), Fig 4 (HBM/DRAM/SSD as wide bands with a property sub-band, promote/demote arrows, a policy box fed by signal pills, prefetch engine) | tiers as wide bands with a one-line property label; policy hooks as marked inputs rather than boxes in the data path. Its figures are raster; ours stay vector |
| *When Fancy Eviction Fails* (2609.28870), a limit study | Fig 1 (Findings → Implications → Techniques: header pills, boxes with an accent edge, thin grey arrows) | Fig 2 as one top-to-bottom logic chain with bold stage headers and thin grey arrows; the verdict words appear in the figure itself |

Common patterns across these figures: one data path plus one zoom panel; at most two fill colors plus one accent; labels on the arrows rather than in a separate key; figure text about the size of the caption (about 7–9 pt at print size).

## What the figures show
**Fig 1, system model and option space** (full width, figure*, 7.0 × 3.45 in; revised for the decision-map story). (a) One session's KV in swimlanes: tiers as rows (GPU HBM; CPU tier, DRAM; slow tier, SSD class (throttled) / NFS class (throttled); dropped), time across (turn k, between requests, turn k+1). At turn end all blocks are written to the CPU tier; when it is full the session is demoted or dropped; at the return it is restored, and the TTFT bracket spans restore plus prefill. Prediction is now an add-on: a dashed magenta band over the gap ("prefetch, eviction order, drop") and the dashed prefetch arrow. (b) The options at the restore step, as chips grouped by direction (Reuse, Compress, Focus, Expand; GPU share is cross-cutting). Solid chips are options in the base map: load all / Cake / recompute all; BF16 / INT4 / low-rank+INT4; CPU tier / slow tier; stream / MPS. Dashed grey chips are add-ons outside the base: approximate recompute, progressive/selective restore, an extra path via a neighbour GPU, borrowed compute. The dashed magenta chip is prediction (the oracle study). The chip outcomes are in Fig 11, not here.

**Fig 11, decision flow** (full width, 7.0 × 3.75 in; revised after the v4 reviews). Brackets name three steps: Step 1, pick restore method and format (box 1). Step 2, share the GPU (box 2). Step 3, add prediction or a mechanism? (boxes 3a and 3b). Each step lists condition → action rows, built only from EVIDENCE.md §M plus B3, C2-GPU, G-M06 and S2:
1. CPU tier: if INT4 is unpacked once, load all; if per-block or unknown, Cake. Always using Cake costs +6.7% on average and 60.5% at worst vs the best (M3-f, H200 kernel split). BF16 only: Cake when busy. b32/b48/NFS class: INT4 + Cake, 11–41% less time than the next best (Qwen + busy GPU misses ε). b16: Cake and load all are close, and load all may win. Low-rank+INT4 only if quality allows and the GPU is idle.
2. MPS vs stream: uncapped MPS gives 31–63% lower TTFT and 0–6.5 pp lower background throughput (S2, simulated; the GPU check agrees in sign).
3. Simulate the oracle bound on your own trace. A high bound (e.g. Mooncake) means prefetch with a free signal (typing hint ≈0.8 of the oracle on GPU replay); eviction order has no free signal tested. A low bound (e.g. FreeInference, < 15% at every tested size) means don't add it. Write-time placement: no.
4. Neighbour idle: a second path, 1.99× (2 GPUs).

I did not find the "~6.8×" unpack-once vs per-block gap in EVIDENCE.md, so the figure omits it. Bottom band: what failed its gate, and what is not covered (SSD class, > 32K, no-verdict items, synthetic hint timing). Each wording was checked against the "cannot write" list #24–30 of EVIDENCE.md.

**Fig 2, method** (single column, 3.33 × 4.12 in). The chain runs top to bottom:
1. Three real traces.
2. Two side-by-side boxes. The trace-driven simulator (② return time) has two constants chips: "CPU tier + GPU recompute: measured on H200" and, dashed, "SSD / NFS class: throttled emulation (constants from an earlier platform)". The GPU box covers ① write and ③ restore time and INT4 (on vs. off), and supplies the cost constants. MPS is split across the boxes: "MPS: microbench + check" in the GPU box and "MPS: trade-off curve" in the simulator box, because the paper's MPS trade-off curve is simulated. A double arrow labels the scaled-down replay check: measured / simulated TTFT 0.911–0.987, same ranking.
3. The oracle ladder (schematic bars): base (restore with Cake) → best rival → free signal → oracle. A dimension line marks headroom = 1 − T_oracle / T_rival.
4. One line of text: "Preregistered threshold: worth it if the bound is ≥15%". The threshold ruler was dropped at the user's request; as in the reference papers, thresholds belong as lines on the result plots plus one sentence in the method.

## Rules the script enforces or follows
- **Drawn at final size.** Coordinates are in inches and fonts in points; the smallest text is 7 pt, enforced by an assert. Fonts are Liberation Sans (en) and Noto Sans TC (zh). PDFs embed TrueType/CID fonts (`pdf.fonttype 42`), no Type 3.
- **Built-in checks.** The script prints any overlapping text boxes, any text outside the canvas, and any font below 7 pt. The final run had 0 overlaps in all four figures.
- **Meaning never by color alone.** Plug points are circles with digits; levers are dark squares; recompute is hatched and load is solid; the foresight action is a dashed arrow. Previews in grayscale and simulated deuteranopia stayed readable.
- **No measured numbers in Fig 1 and Fig 2** except the thresholds (5%, 15%) and 0.911–0.987 [B6]. Their bar lengths are schematic, and both captions should say so. Fig 11 uses a few numbers from EVIDENCE.md §M, B3, C2-GPU and G-M06.
- **Double-blind safe.** No author names, logos, cluster names or repository links. Trace and GPU names are public.
- **Caption caveats** (from PAPER.md §2–§3):
  - Fig 1(b)'s base box states both versions of the base: the simulator uses Cake over the whole prefix, and the GPU's best is a fixed per-tier rule. The caption should spell out that rule.
  - MPS: the trade-off curve is simulated; the GPU ran only a microbenchmark and a check. Fig 2 marks this by splitting MPS across the two boxes.
  - The replay check covers only conversation, SSD class, Llama, and TTFT without queueing.

## Regenerate
```
bash -c 'source code/env_h200.sh >/dev/null 2>&1; python docs/icpe_paper_20261030/figs/diagrams.py --lang both'
```
This writes `fig1_system_*`, `fig2_method_*` and `fig11_decision_*` in zh and en, as PNG (200 dpi, each under 400 KB) and vector PDF. It copies the zh PNGs to `fig1_system.png`, `fig2_method.png` and `fig11_decision.png`. The script also warns when text is wider than its chip or cell. All labels are in the `LABELS` dict.
