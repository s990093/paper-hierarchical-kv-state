# D8 卡片：Pull, Don't Recompute: Peer-to-Peer KV Cache Sharing in llm-d（技術部落格）

- **連結**：https://llm-d.ai/blog/p2p-kv-cache-sharing-llm-d
- **venue／年份**：llm-d 官方部落格，2026-08-15（IBM Research、Red Hat 作者）；不是論文
- **讀了哪裡**：全文（WebFetch）
- **三行摘要**：
  1. KV 在 pod 之間用 NIXL（UCX、RDMA）CPU 對 CPU 傳；沒命中就重算〔原文〕。
  2. GLM-5.2 上「抓」和「重算」約在 8K token 打平；抓取有約 1.2–1.3 s 的固定下限，短 prefix 時重算比較快；文中警告沒校準時短 prefix 的抓取可能比重算慢〔原文〕。
  3. 文中只看到「抓或算二選一」，沒看到兩者同時做〔原文沒提〕。
- **和 D8 的關係**：方向 3 的主要對照組：校準過的「二選一」，以及「固定下限」這個真實參數。
