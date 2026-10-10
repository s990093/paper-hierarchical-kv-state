# D8 卡片：vLLM Optimization and Tuning——搶佔（系統文件）

- **連結**：https://docs.vllm.ai/en/stable/configuration/optimization.html
- **類型／日期**：系統文件（stable 版；抓取日 2026-10-10）；不是論文
- **讀了哪裡**：搶佔一節（WebFetch）
- **三行摘要**：
  1. V1 的預設搶佔方式是 `RECOMPUTE`，不是 `SWAP`；被搶佔的請求等 KV 空間空出來後重算〔文件〕。
  2. 減少搶佔的建議：調高 `gpu_memory_utilization`；調低 `max_num_seqs` 或 `max_num_batched_tokens`；加大 tensor／pipeline 平行〔文件〕。
  3. 和本機原始碼一致：`_preempt_request` 把 `num_computed_tokens` 歸零〔`/mlsteam/workspace/src/vllm/vllm/v1/core/sched/scheduler.py:956–976`〕。
- **和 D8 的關係**：方向 6 的「預設」與「調小 `max_num_seqs`」兩個對照組的出處。
