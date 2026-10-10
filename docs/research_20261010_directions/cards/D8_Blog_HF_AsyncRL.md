# D8 卡片：Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries（Hugging Face 部落格）

- **連結**：https://huggingface.co/blog/async-rl-training-landscape
- **類型／日期**：部落格，2026-03-10（Dirhoussi、Gallouédec、Rasul、Tunstall 等）；不是論文
- **讀了哪裡**：全文（WebFetch，只問了 KV 相關段落）
- **三行摘要**：
  1. 比較 16 個開源 RL 函式庫的非同步設計〔部落格〕。
  2. ART 的睡眠模式：還有進行中的請求時（level-1）把 KV 卸到 CPU；level-2 整個丟掉〔部落格 Axis 5 表〕。
  3. TRL 的設計提案：權重中途更新時存下 KV 前綴、用新策略接續；作者說這需要推論引擎支援「序列中途換權重」；並說重算 KV 會浪費訓練用的算力〔部落格 §5.2、§6〕。
- **和 D8 的關係**：方向 6 的「共置 RL 睡／醒」子情境；方向 7 的背景。注意：另一篇搜尋摘要把「Magistral、Nemotron 試過重算 KV」歸給這篇，**實際上不在這篇**，在 `D8_Blog_AsyncRLSolved.md`。
