# A_CachedAttention_補充 Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention（補充卡）

> CachedAttention 已在 `03_paper_map.md` A3（E04）與 `02_sota_write_techniques.md`。這張卡**不重做**，只補幾點：§1.3 說它是「回來前預取」的出處（要查證）、它其實也有「背景水位線」，以及一個 N4 的實例。

- **出處**：Bin Gao, Zhuomin He, Puru Sharma, Qingxuan Kang, Djordje Jevdjic, Junbo Deng, Xingkun Yang, Zhou Yu, Pengfei Zuo（NUS、SJTU、Huawei Cloud）。**USENIX ATC 2024**（arXiv 2403.19708 comment：「Accepted to USENIX ATC 2024」；Cake 的參考文獻也寫 ATC 24, pp.111–126）。**名稱**：arXiv v1（2024-03-23）的標題是「AttentionStore: Cost-effective Attention Reuse across Multi-turn Conversations in Large Language Model Serving」，v3 改成 CachedAttention（用 arXiv abs 頁核對）。本機 PDF `cachedattention2024_arXiv2403.19708.pdf`，這次讀了 p.5–8、p.11（PDF 實體頁）。

## 補充 1：「回來前預取」——查證屬實
- scheduler-aware fetching：在 job queue 上開一個 look-ahead 預取視窗，看等待中的 job，把它們的 KV 從磁碟先搬到 host 記憶體；視窗長度由可用的預取空間 C_mem 除以平均 session KV 大小 S_kv 決定〔原文 p.7〕。
- 效果（含淘汰）：128 GB DRAM／10 TB SSD 時整體命中率 86%，LRU 58%、FIFO 48%；GPU 時間最多快 2.7 倍；LRU、FIFO 的 DRAM 命中率只有約 0.5–0.6%，因為它們不能事先從磁碟預取〔原文 p.11〕。
- 〔判讀〕這是 10 §2「讀取時版（預取）」的現成實作；它依賴「知道誰快回來」（job queue），這資訊在請求排隊時才有。

## 補充 2：它也有「背景水位線」——不是等滿了才搬
- 原文：「當可用記憶體降到設定的門檻時，CachedAttention 觸發從 host 記憶體到磁碟的淘汰，以確保 host 記憶體緩衝一直可用」〔原文 p.7〕。淘汰對象用 look-ahead eviction window 挑（視窗長度 (C_mem＋C_disk)/S_kv，在視窗內的 job 豁免，視窗尾端的優先被搬）〔原文 p.7〕。
- 〔判讀〕這就是 10 §1.3 要新加的「背景提早搬（水位線）」對照組，已有發表過的 KV 系統在用。11_round1 的 S4W 可以引用它作為出處（加上 Pensieve 的 GPU→CPU 版本：GPU 空位 <25% 時提前換出，`related_papers_update.md` a2，Pensieve p.6）。

## 補充 3：寫入格式決定讀取時能做什麼（N4 的實例）
- CachedAttention 把「存 KV 的時間點」移到**加上位置編碼（RoPE）之前**，存不含位置編碼的 KV；之後超出 context window 要截斷時，取回截斷後的 KV 再套新的位置編碼〔原文 p.8〕。
- 〔判讀〕寫入時選「pre-RoPE」格式，讀取時才能截斷重用；如果存的是加過 RoPE 的 KV，事後要轉回來得反旋轉（可做，但要多一次計算）。所以這是 N4，但不是嚴格「結構上不能」——反旋轉在數學上可逆。

## 補充 4：寫入時機（03／02 已有，這裡只補頁碼）
- 非同步保存：prefill、decode 逐層非同步寫回 host，HBM 留寫入緩衝，沒寫完的 KV 先搬進寫入緩衝以免擋住〔原文 p.6〕。

## 硬體
- 4×A100 80GB、128 GB DRAM、10 TB SSD、PCIe Gen4〔原文 p.8〕。LLaMA-1 65B、LLaMA-2 13B／70B、Falcon 40B〔原文 p.8〕。

## 和本研究的關係
- **讀取時版、背景版都已有出處**：§2 的四個對照組裡，「背景版」和「讀取時版（預取）」都能引用 CachedAttention〔原文 p.7〕。
- **H4／H8**：它的預取與淘汰都用「job queue 裡誰快回來」——這是比「預測 session 會不會回來」更確定、但更晚才有的資訊〔判讀〕。
- **證據等級**：〔原文 p.X〕；判讀如標示。
