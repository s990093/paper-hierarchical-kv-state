# D8 卡片：vLLM：Async Reinforcement Learning（系統文件）

- **連結**：https://docs.vllm.ai/en/stable/training/async_rl/
- **類型／日期**：系統文件（頁面日期 2026-07-28，stable 版）；不是論文
- **讀了哪裡**：全文（WebFetch）
- **三行摘要**：
  1. 訓練端有新權重時，先用 `mode="keep"` 暫停生成，再把權重同步進推論引擎〔文件〕。
  2. `clear_cache` 決定暫停後要不要清掉 KV 與 prefix cache〔文件〕。
  3. `clear_cache=False` 時「context 裡有些 token 可能還是舊權重算的（過期 KV）」〔文件〕。
- **和 D8 的關係**：方向 7：「全清」和「全留」兩個選項都已內建，中間的「部分刷新」沒有。
