# D6_KVServe KVServe: Service-Aware KV Cache Compression for Communication-Efficient Disaggregated LLM Serving

- 出處：Zedong Liu, Xinyang Ma, … Dingwen Tao, Guangming Tan（中科院計算所等），arXiv 2605.13734v1（2026-05-13），頁首標 SIGCOMM'26（venue 未另外查證）。https://arxiv.org/abs/2605.13734 。讀了 p.1–6、p.8、p.10–12（用 grep 定位後細讀）；p.7、p.9 只掃過。

- 精度決定：
  - **做什麼**：從「transform → quantizer → codec」組合出來的策略空間挑一個 profile〔原文 p.6，Fig. 7〕。quantizer 支援 layer-wise 與 head-wise 混合精度；它自己提出的 MixHQ 讓 retrieval head 保持高精度，streaming head 用超低 bit〔原文 p.6〕。
  - **何時**：KV 要開始搬（P→D 遷移，或從遠端 KV pool 取回）的時候選 profile，整個 request 內固定不變〔原文 p.4 §3.1、p.8 §6.1〕。對「存在 KV pool」的情況，原文沒寫清楚壓縮是在寫回時還是讀取時做〔判讀：未查證〕。
  - **粒度**：每個 request。
  - **用什麼資訊**：工作負載類別 w、當下有效頻寬 B、SLO、品質下限〔原文 p.4〕。離線用貝氏最佳化挑出 Pareto 候選；線上用解析式延遲模型加一個輕量 bandit 修正〔原文 p.1–2〕。

- 有沒有和重算／載入並行結合？精度跟位置有關嗎？
  - **沒有重算路徑**。全文 grep「recomput」只出現在描述 CacheGen 的地方：CacheGen「無法滿足 SLO 時會退回昂貴的重算」，而 KVServe 在 5–6 Gbps 仍能找到可行 profile，最高比重算快 32.8×〔原文 p.11〕。
  - 精度跟 token 位置無關。原文說 MixHQ 框架「可延伸到 token 維度（例如保留 SnapKV 的 heavy hitter）」〔原文 p.6〕，但沒有實作位置相依的版本。

- 反量化／解碼成本：
  - **關鍵數字**：Fig. 4 量到每種壓縮法只在某個頻寬以下才划算；超過門檻後，壓縮加解壓的時間就比省下的傳輸時間多，比不壓還慢。CacheGen、MixHQ、KIVI 的門檻分別是 **50／55／110 Gbps**〔原文 p.3〕。Fig. 4 用的硬體在我讀的段落裡沒寫。
  - 短 context 的工作（GSM8K、HumanEval）上，static baseline 的（解）壓縮開銷大過通訊節省，JCT 比不壓還差〔原文 p.10〕。
  - 線上決策 < 1 ms〔原文 p.11〕。測試平台：離線 profile 用 4×A100 40GB，decode 用 H100；prefill 節點分 10／50／100 Gbps 三級〔原文 p.10〕。

- 品質與位置的關係：沒有量位置。Table 1（Qwen2.5-7B、97% 相對準確率約束）：CacheGen 平均相對準確率 65.76%（壓縮比 6.17），KIVI 97.43%（4.40），KVServe-Aware 100.35%（8.28）〔原文 p.11〕。

- 有沒有和延後版比？沒有。

- 和 D6 的關係〔判讀〕：
  - **支持 D6 的動機**：精度的價值取決於頻寬；高頻寬時低精度反而變慢〔原文 p.3〕。Tiara 的 CPU→GPU（PCIe）頻寬遠高於 KVServe 測的網路頻寬，所以在 CPU 層量化可能根本不划算，要量 Tiara 自己的 dequant 成本才知道（**NOT_MEASURED**）。
  - **威脅「依頻寬自適應選精度」的新穎性**：KVServe、CacheGen 都已經在做，只是粒度是 request／chunk，不是位置。
  - 它沒有重算，所以跟 Cake 的 b 無關。

- 證據等級：B（相關段落 grep 定位後細讀，非全文逐頁）。
