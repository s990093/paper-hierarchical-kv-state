# D5_vLLM028_HybridOffload vLLM 0.28 的 OffloadingConnector：滑動視窗與 Mamba 也能卸載、分層

- **出處**：vLLM v0.28.0 原始碼，本機 `/mlsteam/workspace/src/vllm-v0.28.0`（tag v0.28.0，commit `2cf0a6915c`，2026-08-24）；對照本機 0.19.1 build `/mlsteam/workspace/src/vllm`（commit `b1388b1`，2026-04-17）。D5 agent 自己讀的程式碼，**沒有啟動**（NOT_MEASURED）。
- **改了什麼（0.19.1 → 0.28.0）**
  - 0.19.1：只要設了 `--kv-transfer-config`，預設關掉混合 KV 管理器（HMA）〔程式碼 0.19.1 `vllm/config/vllm.py` L1227–1244〕。關掉後滑動視窗層改成全注意力格式、不丟視窗外的 KV〔程式碼 0.19.1 `vllm/v1/core/kv_cache_utils.py` L1160–1216〕；注意力＋Mamba 無法統一，丟 `ValueError`（同檔 L1214–1219）。OffloadingConnector 沒有繼承 `SupportsHMA`〔0.19.1 `offloading_connector.py` L44〕。
  - 0.28.0：只有在 connector 不支援 HMA 時才關〔程式碼 0.28 `vllm/config/vllm.py` L1750–1772〕；`OffloadingConnector(KVConnectorBase_V1, SupportsHMA)`〔0.28 `offloading_connector.py` L49〕。上游的系列 PR「[kv_offload+HMA][0/N]–[13/N]」2026-03-13 到 05-01，13/N（#41445，commit `2fa1f8ec00`，2026-05-01）正式打開；之後「Enable HMA models for Tiering Offloading」（#44287，06-03）、「Mamba CPU Offloading」（#44599，06-12）、「Fix Mamba all-mode CPU offload boundary alignment」（#51100，08-06）〔`git log`〕。
- **寫入時做了什麼決定**
  - 滑動視窗層：store 時**每個算出來、GPU block 還在的 chunk 都存**；只跳過已被 HMA 釋放的 block（id 0），以及「全注意力 chunk 比 SWA chunk 大」時永遠用不到的 chunk〔0.28 `offloading/scheduler.py` L1275–1312、L129–146〕。Gemma-3 這種兩組 block 一樣大的模型，`alignment_chunk_count` 是 None，等於**局部層的 KV 全部存**〔判讀，同檔 L207–218〕。
  - 讀取（lookup）時，滑動視窗組只要求「從命中點往回 W 個 token 的 chunk 連續都在」〔同檔 L633–663〕，所以**載入只載視窗內的**。
  - Mamba 組：一個命中點只需要 1 個 chunk（那一點的狀態）〔同檔 L109–126〕；`all` 模式每個 block 邊界都存狀態，`align` 模式只在排程步結尾對齊時存〔同檔 L149–169；`vllm/model_executor/models/config.py`〕。
  - 分層：`TieringOffloadingManager`「Always offload to all tiers」——存進 CPU 主層的 block 會**寫穿到所有次層**（檔案系統、物件儲存、p2p）〔0.28 `vllm/v1/kv_offload/tiering/manager.py` L11–13〕。
- **用什麼資訊做決定？寫完還在不在？（N1）**：規則固定，不看負載。局部層視窗外的 KV、Mamba 的中間狀態，在 GPU 上一離開視窗／被覆蓋就沒了；所以「要不要留」必須在寫入時決定〔判讀〕。vLLM 的答案是：**全部留、全部寫穿**。
- **有沒有和延後版、寫穿版比較？**：沒有實驗（程式碼）。它本身就是寫穿版。
- **和本研究的關係**
  - Lit-C §5 說「接 OffloadingConnector 時滑動視窗存全部、SSM 會報錯」——**只對 0.19.1 成立**；0.28 已經修好機制（HMA＋卸載＋分層，含 Mamba 狀態）。
  - 0.28 仍存全部局部層 KV：對「只會接著問」的負載是浪費（Gemma-3-12B 32K 時 81%，見 D5 §4c）；但對「從中間任意位置分叉」的負載，那些局部 KV 正是讓命中成立的東西（和 SSM 檢查點同一種結構）。
  - D5(b)(c) 的「分層放置」機制已有現成系統；剩下的是**策略**（哪些命中點值得留、留在哪一層）。
- **證據等級**：〔程式碼 file:line〕；「Gemma-3 局部層全存」是讀程式碼的推論，未實測。
