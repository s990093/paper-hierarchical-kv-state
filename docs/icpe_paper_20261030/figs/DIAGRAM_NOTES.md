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
**Fig 1, system model** (full width, figure*, 7.0 × 3.05 in). (a) One session's lifetime in swimlanes. Rows are tiers: GPU HBM, CPU tier (DRAM, BF16), slow tier (INT4; SSD class (throttled), NFS class (throttled)), dropped. The x axis is time: turn k, between requests, turn k+1. The walk-through: at turn end, all blocks are written to the CPU tier. When the CPU tier is full, the session is demoted to the slow tier (thinner bar = fewer bytes) or dropped. At the return, the session is restored into HBM, and the TTFT bracket spans the restore plus the prefill. The three foresight plug points are magenta numbered circles on the two event lines and in the gap: ① write time, ② return time (prefetch, eviction order, drop never-returners), ③ restore time. Prefetch is the only drawn foresight action (a dashed magenta arrow). The two levers that need no foresight are dark square tags: INT4 on the slow tier, and MPS on the GPU lane ("reserve SMs for restore"). (b) The restore step: a Cake block bar and schematic restore-time bars (load all / recompute all / Cake). Below them is the base rule (CPU tier: load all; slow tier: Cake).

**Fig 2, method** (single column, 3.33 × 4.62 in). The chain runs top to bottom:
1. Three real traces.
2. Two side-by-side boxes. The trace-driven simulator (② return time) has two constants chips: "CPU tier + GPU recompute: measured on H200" and, dashed, "SSD / NFS class: throttled emulation (constants from an earlier platform)". The GPU box covers ① write and ③ restore time, INT4 and MPS (on vs. off), and supplies the cost constants. A double arrow labels the scaled-down replay check: measured / simulated TTFT 0.911–0.987, same ranking.
3. The oracle ladder (schematic bars): base (restore with Cake) → best rival → free signal → oracle. A dimension line marks headroom = 1 − T_oracle / T_rival.
4. The preregistered ruler: dead end < 5% ≤ possible < 15% ≤ worth it.

## Rules the script enforces or follows
- **Drawn at final size.** Coordinates are in inches and fonts in points; the smallest text is 7 pt, enforced by an assert. Fonts are Liberation Sans (en) and Noto Sans TC (zh). PDFs embed TrueType/CID fonts (`pdf.fonttype 42`), no Type 3.
- **Built-in checks.** The script prints any overlapping text boxes, any text outside the canvas, and any font below 7 pt. The final run had 0 overlaps in all four figures.
- **Meaning never by color alone.** Plug points are circles with digits; levers are dark squares; recompute is hatched and load is solid; the foresight action is a dashed arrow. Previews in grayscale and simulated deuteranopia stayed readable.
- **No measured numbers in the figures** except the thresholds (5%, 15%) and 0.911–0.987 [B6]. All bar lengths are schematic, and both captions should say so.
- **Double-blind safe.** No author names, logos, cluster names or repository links. Trace and GPU names are public.
- **Caption caveats** (from PAPER.md §2–§3):
  - Fig 1(b)'s base rule is the GPU-measured best rule. The simulator instead applies Cake over the whole prefix.
  - MPS end-to-end results are simulated; only the MPS microbenchmarks run on the GPU.
  - The replay check covers only conversation, SSD class, Llama, and TTFT without queueing.

## Regenerate
```
bash -c 'source code/env_h200.sh >/dev/null 2>&1; python docs/icpe_paper_20261030/figs/diagrams.py --lang both'
```
This writes `fig1_system_{zh,en}.{png,pdf}`, `fig2_method_{zh,en}.{png,pdf}` (PNG at 200 dpi, each under 400 KB) and copies the zh PNGs to `fig1_system.png` and `fig2_method.png`. All labels are in the `LABELS` dict.
