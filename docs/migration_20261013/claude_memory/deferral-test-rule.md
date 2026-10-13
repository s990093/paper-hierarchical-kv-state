---
name: deferral-test-rule
description: Every write-time idea must be compared against its lazy, background, pre-clean and write-through twins, fixed before seeing results
metadata:
  type: feedback
---
Any "decide at write time" strategy only counts if it beats ALL deferred twins (lazy S4B-style, background watermark S4W, pre-clean S4C, write-through S1/S2b, read-time Cake) by >=5% on GPU across >=3 seeds. The user accepted this rule on 2026-10-09.
**Why:** 08 found S5L +18.9% vs simple strategies but only +0.7% vs its lazy twin; the red-team also flagged that adding twins only after seeing a winner is a forking-paths bias.
**How to apply:** define twins and criteria in a pre-registration doc before running; fix the single candidate strategy in advance (don't pick best-of-5 per seed); list every post-hoc deviation explicitly; timestamps via `date -u`.
