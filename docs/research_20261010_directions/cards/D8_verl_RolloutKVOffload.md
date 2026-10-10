# D8 卡片：verl：Rollout KV Cache Offload via Mooncake-Store（系統文件）

- **連結**：https://verl.readthedocs.io/en/latest/perf/rollout_kv_offload.html
- **類型／日期**：系統文件，頁面標「Last updated: 05/27/2026」；不是論文
- **讀了哪裡**：全文（WebFetch）
- **三行摘要**：
  1. 把 vLLM rollout 引擎的 prefix KV block 卸到共用的 Mooncake store，讓 system prompt、agent 工具歷史、同一 prompt 的 n 個樣本跨請求與跨副本去重〔文件〕。
  2. 長尾負載搬到閒置副本時，共用 prefix KV 可以減少重新 prefill〔文件〕。
  3. 每次權重更新都清掉本地與 Mooncake 的 KV，避免重用舊策略的 KV；沒有效能數字〔文件〕。
- **和 D8 的關係**：方向 6、7：RL 裡的 KV 分層已經進入主流框架；「權重更新就全清」是它的預設。
