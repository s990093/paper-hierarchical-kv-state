---
name: explore-many-directions
description: "When the research hits a dead end, explore many directions in parallel in plain language instead of pushing one jargon-heavy recommendation"
metadata:
  node_type: memory
  type: feedback
  originSessionId: a23f9c53-26e5-4f69-a5de-3695271a708c
  modified: 2026-10-10T05:33:30.928Z
---

On 2026-10-10 the user said they couldn't follow my single "(推薦) 先建併發／寫入積壓模型" next step and asked me to find directions from many angles instead.
**Why:** a single technical recommendation phrased in jargon (寫入積壓、N3、twins) lost them. They want to see a broad set of options, each with a quick verdict.
**How to apply:**
- Explain the situation first with an everyday analogy. Moving house worked well: decide at the door vs. put everything in the living room and move it to storage later.
- Then probe several directions in parallel. Use subagents, one per direction, each with pre-registered kill criteria. Only one agent gets the single GPU.
- Return a table of verdicts (有看頭／可能／死路) rather than one pick.

See [[user-profile]] and [[tiara-research-state]].
