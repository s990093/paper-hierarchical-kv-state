# D5_SGLangHiCacheSWA SGLang HiCache 的滑動視窗（SWA）層卸載

- **出處**：SGLang 開源程式碼（系統）。`sgl-project/sglang@3831e7e0918052be50ef38b3ff62154573e213d7`（main，2026-10-10）的 `python/sglang/srt/mem_cache/unified_cache/components/swa.py` 與同目錄 `README.md`。PR 描述：#23391（2026-05-06 合併）、#27557（未合併）。
- **寫入時做了什麼決定**：**每種層型分開決定要不要寫到 host／L3**（分層感知卸載），不是一個寫入時的「選位置」。
  - GPU 上：SWA 元件在 insert 時檢查視窗邊界，必要時分裂節點，父節點變成 tombstone（SWA 資料丟掉、節點留著），子節點保有 SWA 資料〔程式碼 components/README.md:274〕；淘汰時內部節點也是 tombstone〔程式碼 components/README.md:160〕。
  - 寫到 host（cache mode）：只備份「從被備份的節點往上、一個滑動視窗內、還在 device 上、還沒有 host 複本」的 SWA 資料；buffer mode 一次只備份一個節點〔程式碼 swa.py:101–137〕。Full 層的 KV 則照整個節點備份〔判讀，依 components/README.md:61–65 與 Full 元件的備份路徑〕。
  - 從 host 讀回：只往上走到湊滿 `sliding_window_size` 為止〔程式碼 swa.py:1129–1164〕。
  - 寫到 L3：只寫 host 複本中整 page 的尾段，命中規則 `TRAILING_PAGES`〔程式碼 swa.py:1166–1180〕。
  - `unified_kv` 模式把 SWA 當成只在 device 上的 ring，完全不卸載〔程式碼 swa.py:1099–1101〕。
- **用什麼資訊做決定？寫完後還在不在？（N1）**：用的是模型結構（視窗大小）和 radix tree 的節點邊界。寫完後還在，**不是 N1**。視窗外的 SWA KV 在 GPU 上也還在（直到被 tombstone），所以「先寫、之後再丟」是做得到的〔判讀〕。
- **有沒有和延後版、寫穿版、背景版比較？**
  - #23391 只比「開／不開 HiCache」（gpt-oss-120b、TP2、多輪 bench），沒有比「SWA 全寫 vs 只寫視窗內」〔PR #23391 描述〕。
  - **#27557（未合併）是最接近的比較**：作者說目前 L3 對**每個 page** 都存一份 SWA 快取，DeepSeek-V4-Flash、1M token 在 L3 佔約 18 GB，而壓縮注意力的快取理論上只要約 3.6 GB。改成每 2K token 才存一次 SWA 檢查點：L3 用量 18.25→5.74 GB（−68%），命中時 TTFT 3,039→3,536 ms（+16%）；每 16K：4.03 GB（−78%），TTFT 4,660 ms（+53%）。Hopper、CP4、Mooncake store〔PR #27557 描述，作者自報〕。也就是說：即使有「只備份視窗內」的規則，每個 chunk 節點在被備份時自己的視窗都還在 device 上，所以實際上很多中間位置的 SWA 視窗都被寫出去了〔判讀，與 #27557 的描述一致〕。
  - 沒有和「背景版」（例如搬到 L3 時才丟）比過。
- **硬體、各層頻寬、模型**：#23391 用 gpt-oss-120b；#27557 用 DeepSeek-V4-Flash、Hopper。頻寬：未查證。
- **和 D5(c) 的關係**
  - **「分層感知卸載本身」在 SGLang 已經有了**：host 備份是層型感知的（SWA 只備份視窗內、讀回只讀視窗內）〔程式碼 swa.py:101–137、1129–1164〕。依 D5 判準 (c)，「已經有人做 → 死路」這一條對 SGLang 成立〔判讀〕。
  - **還沒做的是「SWA 視窗檢查點要存幾個、放哪一層」**：#27557 的固定間隔（2K／16K）是一個寫入時規則，和 vLLM 的 retention interval（見 D5_vLLMRetentionInterval）同一個形狀；但它沒合併，也沒有依重疊分布選間隔，也沒有「GPU 密、L3 疏」的分層差異〔判讀〕。
  - 對 Gemma-3／gpt-oss：SWA 視窗外的 KV 在這套規則下不會被 host 備份；但多 chunk prefill 時每個 chunk 邊界的視窗都可能被寫出去，浪費比例要看 chunk 大小與視窗大小的比〔判讀；未量〕。
- **證據等級**：程式碼〔程式碼 repo@commit file:line〕；PR 數字是作者自報（未重現）；分層效果的推論是〔判讀〕。
