# D8 卡片：vLLM Ascend：Recompute CPU Offload Connector（系統文件）

- **連結**：https://docs.vllm.ai/projects/ascend/en/v0.24.0rc/user_guide/feature_guide/recompute_cpu_offload.html
- **類型／日期**：系統文件，v0.24.0rc；不是論文
- **讀了哪裡**：全文（WebFetch）
- **三行摘要**：
  1. 請求被 decode 端搶佔時，先把已算好的 KV block 從 HBM 複製到 CPU DRAM，排回來時再複製回 HBM〔文件〕。
  2. CPU 空間不夠時跳過卸載，回到原本的重算〔文件〕。
  3. 動機：P/D 分離的 decode 節點不擅長重算長 prefill；文件說傳輸目前「以正確性為主，還沒為吞吐最佳化」，沒有效能數字〔文件〕。
- **和 D8 的關係**：方向 6：「搶佔時卸載、回來時載入」已經是現成功能（只載入，不並行重算）。
