# E05　逐出與快取策略：評測卡

> 抽取者：子 agent E05，2026-10-06。複核：V05，2026-10-07（逐格回原文；紀錄見檔尾「複核紀錄」）。
> 範圍：Fancy-eviction、AsymCache、Marconi、AdaptCache、EvicPress、KVCache cache in the wild（Alibaba）、LRB、HALP。
> 頁碼規則：arXiv 論文用 **PDF 頁**（p.N）；USENIX 論文同時給 **會議論文集頁碼**（proc. p.NNN）與 PDF 頁（PDF 第 1 頁是 USENIX 封面）。

## 來源清單（查證日 2026-10-06）

| 簡稱 | 讀的版本 | URL | 程式碼／資料 |
|:--|:--|:--|:--|
| Fancy-eviction | arXiv **v2**（2026-10-01，20 頁）全文；另下載 v1（19 頁）抽查關鍵詞（19.9、partial-node、BeladyCompute、AgentX 等出現次數相同） | https://arxiv.org/abs/2609.28870 | 原文說「將開源 trace 與模擬器」（p.1）；GitHub／網路搜尋未找到 repo |
| AsymCache | arXiv **v1**（2026-06-01，15 頁）全文 | https://arxiv.org/abs/2606.02964 | 原文提到 "our artifact"（p.6）但沒有給 URL；未找到 |
| Marconi | arXiv **v3**（2025-04-10，頁腳為 MLSys'25 proceedings）全文＋Artifact Appendix | https://arxiv.org/abs/2411.19379 | https://github.com/ruipeterpan/marconi ，commit `0801661`（2025-03-05）；Zenodo 10.5281/zenodo.14970139 |
| AdaptCache | arXiv **v2**（2026-01-15，3 頁正文）全文；BigMem'25 PDF（Microsoft Research，3 頁）逐行 diff：除作者欄排版外文字相同 | https://arxiv.org/abs/2509.00105 ；https://www.microsoft.com/en-us/research/wp-content/uploads/2025/10/adaptcache-bigmem25.pdf | 未公開（與 `workloads_eval.md` §3.5 一致） |
| EvicPress | arXiv **v1**（2025-12-16，16 頁）全文 | https://arxiv.org/abs/2512.14946 | 原文無 repo 連結 |
| KVCache in the wild | arXiv **v5**（2026-02-14，18 頁〔複核修正：原寫 19 頁；`pdfinfo` 顯示 PDF 為 18 頁〕）全文；**未讀 ATC'25 會議版** | https://arxiv.org/abs/2506.02634 | trace：https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon （README commit `5f7439c`，2026-04-23）；策略：vLLM PR #22236（**已關閉、未合併**，最後更新 2026-02-09） |
| LRB | NSDI'20 會議版全文（proc. pp.529–544） | https://www.usenix.org/system/files/nsdi20-paper-song.pdf | https://github.com/sunnyszy/lrb ，最後 commit `9e8b442`（2023-01-20），BSD-2-Clause；含 Wikipedia trace |
| HALP | NSDI'23 會議版全文（proc. pp.1149–1163） | https://www.usenix.org/system/files/nsdi23-song-zhenyu.pdf | 程式碼未公開；兩條 YouTube trace 需簽資料共享協議（proc. p.1150 腳註） |
| （輔助）LongBench v1 task.md | commit `2e00731`（2025-01-15） | https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/task.md | 用來核對 AdaptCache／EvicPress 的資料集筆數與長度單位 |

---

## 本組的共同模式（跨論文比較）

1. **模擬器 vs 真機：CDN 傳統是「模擬器跑廣度、原型或生產跑開銷與端到端」，LLM 這組大多只做其中一半。**
   LRB 的模擬器與 ATS 原型共用同一個 C++ 函式庫（proc. p.535–536），HALP 用模擬比演算法、用生產 A/B 測試量效果與 CPU（proc. p.1155）〔原文〕。
   LLM 這邊分成兩派：
   * trace 驅動模擬派：Fancy-eviction（C++，以 libCacheSim 為基礎）、Marconi（Python）。
   * 真引擎派：AsymCache、EvicPress、AdaptCache、in-the-wild 的策略評估，只比 3–6 個對手。
   模擬器的驗證都很弱：Fancy 只說與 vLLM 的命中率很接近，沒有給誤差數字（p.4）。Marconi 的 TTFT 不是量的，是拿剖析表查出來的（`plotting/ttft.py`）〔原文／程式碼〕。

2. **快取大小這條軸：CDN 用絕對容量、對數尺度；LLM 這組各用各的，沒有一篇以「佔工作集的百分比」當主軸。**
   * LRB：64 GB–4 TB，共 33 個 trace×容量組合（proc. p.539）。
   * HALP：Wikipedia 64–1024 GiB（proc. p.1157）。
   * Fancy：照硬體取 24／48／96 GiB 與 1 TiB，附錄再密掃 65 點（p.4、p.17）。
   * in-the-wild：用「CPU 快取 ÷ HBM」的倍數（p.12）。
   * Marconi：絕對 GB（p.9；程式碼 1–100 GB）。
   * AsymCache、EvicPress、AdaptCache：容量固定不掃，改掃 QPS、α 或壓縮率。

   Fancy 的密掃顯示，只取少數幾個容量會藏住策略交叉與非單調（p.17）〔原文〕。

3. **命中率要分「計數」與「成本加權」：CDN 的 byte／object 之分，在 LLM 裡變成 token 位置之分。**
   * CDN 明確區分兩種指標。byte miss ratio 對應 WAN 成本（LRB proc. p.529；HALP 用的是 P95 byte miss）。object miss ratio 對應延遲。HALP 還指出兩者會衝突：HALP 降了 byte miss，object miss 可能上升，使磁碟延遲有 +5% 的尾巴（proc. p.1156）〔原文〕。
   * LLM 這組的對應做法：Fancy 提出 compute-savings ratio，按每個 block 的重算 FLOPs 給命中加權（p.10）。Marconi 用 FLOP efficiency 做逐出分數（p.6–7）。AsymCache 的逐出分數是 f·ΔT，ΔT 隨位置線性增加（p.4–5）。
   * 〔判讀〕在 CDN 裡物件「大小」決定成本；在 LLM 前綴快取裡，block 的「深度」決定成本。

4. **對手與上界：只有 Fancy 完整承襲 CDN 的「大量對手＋Belady」傳統。**
   * LRB：14 種演算法，另加 B-LRU、Belady MIN 與 relaxed Belady。其中 4 種的參數與原作者核對過（proc. p.536–537）。
   * HALP：與 LRU、FIFO、ARC 比，再與 LRB（特徵改成與 HALP 相同、記憶窗有調）及 Adaptive-TinyLFU 比（proc. p.1157）。沒有 Belady。
   * Fancy：14 種線上演算法（**含 LRU 本身**），加上 Belady、BeladyCompute 與 ILP 最佳解（p.4、p.10）。
   * 其餘 LLM 論文都沒有上界：
     * in-the-wild 只算了無限容量的理想命中率；〔複核補充〕另有「理想策略（永不逐出之後還會用到的 KV）需要多少容量」的分析（p.9 Fig. 22），但兩者都不是有限容量下的 Belady 式上界；
     * Marconi 的 artifact 有「靜態 α 的離線最佳」，但沒有放進論文（p.16）；
     * AsymCache、EvicPress、AdaptCache 都沒有上界。

5. **學習式方法的協定：CDN 兩篇把「調參期、暖機期、量測期」分得很清楚，LLM 這組多半沒有。**
   * LRB：
     * 前 20% trace 當驗證段，用來調超參數；
     * 暖機期一律比驗證段長，暖機期間不記指標；
     * 每累積 128K 筆標記樣本重訓一次；
     * metadata 從快取容量裡扣掉（proc. p.537）。
   * HALP：
     * 線上訓練，每 1024 筆標記資料更新一次；
     * 3 天的 trace 以第 1 天暖機；
     * 生產環境用「實驗／no-op／對照」三組，再用去卷積扣掉噪聲（proc. p.1153–1155）。
   * LLM 這組：
     * Marconi 只在 bootstrap 期結束時做一次 α 的格點搜尋（程式碼）；
     * in-the-wild 用最近一小時擬合指數分布；
     * EvicPress 有 50 題訓練／50 題測試的 query 切分（p.8）；
     * 都沒有說明暖機或量測窗口怎麼排除。

6. **「位置」的方向，兩派相反。**
   * 以命中率為目標的 in-the-wild 策略：重用機率相同時，先逐出 offset 大的尾端 block，理由是前段的空間局部性較好（p.11）。
   * 以計算為目標的 AsymCache 與 Fancy RandomCompute：先逐出淺的、便宜的 block（AsymCache p.4–5；Fancy p.10–11）。
   * Fancy 給了一個可以直接沿用的診斷：同一份 trace 上 hit-optimal Belady 與 BeladyCompute 的差距，就是「成本感知值不值得做」的上限。FreeInference 在 24 GiB 時差 12.6 點，其他五條 trace 都 ≤1.1 點（p.10）〔原文〕。

---

### Fancy-eviction：When Fancy Eviction Fails: Rethinking Cache Replacement For LLM Prefix Reuse（arXiv 2026；arXiv 2609.28870）

- **讀了什麼**：〔全文〕v2（2026-10-01），https://arxiv.org/pdf/2609.28870v2 ，查證 2026-10-06；v1 只抽查關鍵詞。
- **一句話**：用兩條生產 agent trace 檢驗傳統逐出法是否適用 LLM 前綴快取，並提出成本感知與粒度原則。
- **評測要證明的主張**：
  * LRU 很穩，多數精巧演算法沒有更好，頻率派更差；但離 Belady 仍有空間（p.1、p.4）。
  * 原因是重用跟著 session 的節奏走；另外未命中成本隨深度增加，session 的佔用量呈重尾（p.6–9）。
  * 成本感知加上 partial-node 逐出，能在真引擎上降低平均 TTFT（p.11）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | Qwen3-Coder-30B。用在三處：tokenizer、成本模型、真引擎驗證。注意力類型與 dtype 原文沒寫〔未查證〕。〔計算〕Table 2：24 GiB＝16,384 blocks，所以每個 16-token block 是 1.5 MiB，即每 token 96 KiB | p.4 §3.1、Table 2；p.8 Fig. 7–8；p.10 §5.2 〔原文〕〔計算〕 |
| 硬體 | 真引擎驗證：單張 NVIDIA H200，KV 快取 48 GiB。模擬的容量則對到硬體：24 GiB 是 native-context 下限；48 GiB 是 1×H200（記憶體比例 0.8）；96 GiB 是 1×B200（bf16 權重）；1 TiB 是 Mooncake 的一整個節點 | p.4 Table 2；p.11 Fig. 13 〔原文〕 |
| 軟體與版本 | 自寫 C++ 模擬器，以 libCacheSim 為基礎，加上「請求級駐留限制」：一個請求的整段前綴必須同時在記憶體裡，所以會預先逐出騰出空間。比 vLLM 的 Python 實作快最多 160×。真引擎是 vLLM，**版本沒寫**；也沒說怎麼改 vLLM 讓它支援「洞」後面的 block 命中 | p.4 §3.1；p.11 〔原文〕；〔未查證〕版本 |
| 資料／負載 | 兩條自採 trace：<br>• FreeInference：327.5K 請求，34.3% 屬於多輪 session，7.0 天，處理 10.5B token，其中 unique 0.63B<br>• Chutes：515.8K 請求，26.1% 屬於多輪，130.9 天，處理 9.7B token，其中 unique 2.23B<br>延伸驗證另用四條 Qwen Bailian trace 與 AgentX。AgentX 是 393 個 Claude Code session；因為沒有絕對時間戳，人工排成固定併發 16；hash 只在 session 內有效，所以跨 session 共享視為 0。傳統負載的對照是 cacheMon 的 Wikipedia 2019 與 CloudPhysics w01 | p.3 Table 1；p.5 Table 3；p.6 Table 4；p.9 §4.6；p.17 Table 6 〔原文〕 |
| 長度 | 單位都是 token（Qwen3-Coder tokenizer），上限 256K；超過的略過：FreeInference 4.3%，Chutes 0.03%。<br>• 平均請求長度：FreeInference 32.0K，Chutes 18.7K；p99 分別 219.3K、98.3K<br>• 中位數：全部請求 3,386，多輪請求 66,605；p90 是 111K | p.3 Table 1；p.4；p.8 Fig. 9 〔原文〕 |
| 到達與併發 | • 模擬：照 trace 的順序與時間戳重播〔判讀：AgentX 因為沒有時間戳才需要人工排程，p.9，由此反推其他 trace 用真實時間〕。<br>• 真引擎的 10,000 個請求怎麼到達，沒寫〔未查證〕。<br>• FreeInference 平均約 12 個 session 同時活躍。<br>• 附錄 B 固定容量（48 GiB／1 TiB），把 73,637 個多輪請求重新縮放到目標併發 8–1024 | p.9；p.17–18 App. B 〔原文〕〔判讀〕 |
| 重用結構 | 多輪 session（agent 與人類對話）、跨 session 共用的系統提示、單輪請求。〔FreeInference〕系統提示佔 distinct block 的 4.7%，貢獻 18.4% 的存取；多輪歷史佔 37.6%，貢獻 70.2%；單輪請求佔 57.8%，貢獻 11.4% | p.5 §4.1 〔原文〕 |
| 掃描的自變數 | • 快取容量：主文 4 點（24、48、96 GiB、1 TiB）；附錄 65 點，橫跨 24 GiB–12 TiB；AgentX 26 點<br>• 成本模型三種：uniform、Qwen3-Coder 實測、linear<br>• 逐出粒度：block 級與 session 級<br>• session 併發度 | p.4；p.12；p.17–19 〔原文〕 |
| 對手 | 14 種線上演算法，**含 LRU 本身**：<br>• 近期：LRU（基準）<br>• 快速降級：ARC、Sieve、S3-FIFO、S4-FIFO、LIRS<br>• 解析模型：LHD<br>• 頻率：LFU、W-TinyLFU<br>• 學習式：LeCaR、LRB、3LCache<br>• 前綴快取專用：Workload-aware（即 in-the-wild 的策略）、AsymCache<br>另有 Belady（hit 最佳）、BeladyCompute、ILP 最佳解；自提 RandomCompute 與 PartialNode RandomCompute。<br>**各演算法的參數沒有說明**。LRB 因為太慢，沒放進附錄的密掃 | p.4 §3.2；p.10；p.17 App. A 〔原文〕 |
| 系統指標 | 真引擎驗證：平均、中位數、P99 TTFT 與 prefill 吞吐；TTFT 和吞吐「分開兩次跑」 | p.11 Fig. 13 〔原文〕 |
| 模擬器／真機與驗證 | 主要結果全部來自模擬器。模擬器的驗證只有一句「hit ratios are very close to those of the native vLLM」，**沒有誤差數字或比對條件**。真引擎只驗證一個點：48 GiB、10,000 個 FreeInference 請求、LRU 對 PartialNode RandomCompute〔複核修正：Fig. 13 也包含不做 partial-node 的 block 級 RandomCompute，共三個策略；容量仍只有一點〕 | p.4；p.11 Fig. 13 〔原文〕 |
| 命中率定義與換算 | • **block hit ratio**：每個 16-token KV block 是一個物件。<br>• **compute-savings ratio**：每次命中按「該 block 省下的計算」加權；uniform 成本時就退化成 hit ratio。分母沒有明寫〔未查證〕。<br>• 實體逐出與邏輯前綴樹脫鉤：block 可以從任何位置被丟，所以會有「洞」（要重算的連續 block），而洞之後的 block 仍算命中。〔判讀〕這不是 `workloads_eval.md` §2.2 的「前綴算到第一個缺口為止」語意。<br>• 從命中率換到延遲：只有上述單點的真引擎量測 | p.3 §2.2；p.10 §5.2；p.11 腳註 2 〔原文〕〔判讀〕 |
| 上界／Oracle | • hit ratio 的上界是 Belady（block 等大）。<br>• compute savings 的上界有兩個：<br>　◦ ILP（以重用間隔為變數、容量為限制）<br>　◦ BeladyCompute：分數＝ComputeIntensity×TimeUntilNextAccess，逐出分數最高者；在 24／48／96 GiB 分別比 ILP 低 1.03／0.41／0.08 點<br>• 駐留限制下 Belady 是否仍然最佳，原文沒有討論 | p.10 §5.2 〔原文〕 |
| 品質指標 | 不適用：前綴重用是精確的，未命中只付重算 | p.13 §7 〔原文〕 |
| 主要結果 | (1) 頻率派大幅落後。以下這群與 LRU 相近或略差：ARC、LHD、LeCaR、3LCache、LRB、Workload-aware、AsymCache。快速降級類則依 trace 而定（p.4）。<br>(2) 在 24 GiB、FreeInference、Qwen3-Coder 成本模型下：RandomCompute 的 compute-savings ratio 是 0.638，比 LRU 高 10.4 點，也超過 hit-optimal 的 Belady（p.11）。<br>(3) 真引擎上，PartialNode RandomCompute 對比 LRU：<br>　• 平均 TTFT 1.29→1.04 s（−19.9%）<br>　• prefill 吞吐 55.2K→65.6K tok/s（+18.8%）<br>　• P99 TTFT 26.0→16.8 s<br>　• **中位數 TTFT 反而變差**（112→171 ms）<br>　• 不做 partial-node 的 block 級 RandomCompute：TTFT 變 4.0×、吞吐只剩 1/3.9（p.11）<br>(4) 在 1 TiB，每條 trace 上 LRU 都在最佳線上演算法 0.4 點以內（p.9）〔複核修正：這句在 §4.6，範圍是 Fig. 11 的三條 trace（Qwen To-B、Qwen Coder、AgentX），不是全部六條；比的是「最佳線上演算法」，不是 Belady〕。附錄 C：1 TiB 時所有策略**連同兩個 oracle**都在 0.2 點以內（p.19）〔複核補充：這句講的是 FreeInference 的 compute-savings ratio（Fig. 12／19 的策略集合）〕 | 見各格 〔原文〕 |
| 消融／敏感度／開銷 | • 成本模型三種（p.12 §6.1；p.18–19 App. C）<br>• block 級 vs session 級：24 GiB 時，LRU 的 hit ratio 少 10.5 點，RandomCompute 的 compute savings 少 18.7 點；48 GiB 時分別少 3.3、5.5 點（p.11–12 §5.3）<br>• 洞的分布：RandomCompute 平均每請求 9.21 個洞，P99 143 個；PartialNode 把洞數降 8.7×，在 24 GiB 只損失 0.17 點（p.11；0.17 點出自 p.19 App. C〔複核修正：原寫 App. D〕；P99 143 個洞出自 p.19 App. D）<br>• 併發度：48 GiB 時，Belady 從 0.926 降到 0.713，LRU 從 0.898 降到 0.517（併發 8→32）（p.18）<br>• 密掃（App. A）：Qwen To-C 上 LIRS 從比 LRU 高 4.2 點變成在 724 GB 時低 10.1 點；W-TinyLFU 在 FreeInference 上 97 GB 0.866 → 906 GB 0.759 → 12 TB 0.933，非單調（p.17）<br>• 模擬器速度：比 vLLM Python 實作快最多 160×（p.4） | 〔原文〕 |
| 重複與統計 | 未說明。模擬是確定性的。真引擎單點沒有說跑幾次，沒有誤差棒 | 〔原文〕（全文未見） |
| 程式碼／資料 | 原文說將開源 trace 與模擬器（p.1–2）；查證日未找到公開 repo。AgentX 與 Bailian 是公開資料 | 〔原文〕；〔未查證〕repo |
| 設計理由（原文） | • 公開 trace 都缺以下至少一項：跨 session 的細粒度 block 身分、多輪 agent 流量、多日時間戳。所以自採 trace（p.4 §3.1、Table 3）。<br>• libCacheSim 這類 block 級模擬器忽略請求級駐留限制，所以自寫模擬器（p.4）。<br>• 容量分成兩個 regime，對應每張卡的 HBM 與全域 DRAM／SSD 池（p.4）。<br>• 先用兩個 oracle 的差距來判斷某份負載值不值得做成本感知（p.10） | 〔原文〕 |
| 設計理由〔判讀〕 | 用硬體對應的 GiB 而不是「工作集 %」，是為了讓結論直接對到部署。但 trace 的 unique 量差很多（0.63B vs 2.23B token），同一個 GiB 在兩條 trace 上的壓力不同，跨 trace 比較要小心 | 〔判讀〕 |
| 原文沒講清楚的地方 | 1. 13 個對手的參數（LRB 的記憶窗、LeCaR 的學習率等）。<br>2. AsymCache 與 Workload-aware 在模擬器裡怎麼實作：AsymCache 原本要配合 MSA 與成本模型。<br>3. compute-savings ratio 的分母，以及成本模型到底是「FLOPs」（p.10）還是「measured」（p.18 App. C 的用詞）。<br>4. 真引擎實驗的 vLLM 版本、到達方式，以及怎麼支援洞之後的命中。<br>5. 模擬器與 vLLM 的命中率差多少 | 〔原文〕 |
| 與既有整理不一致 | • **sota.txt §2.11「Belady 上界在各種容量下仍有明顯空間」：過度概括**〔複核：✅ 指控成立，但原證據引錯，已改〕。原文的空間主要在 HBM 容量：Fig. 2（p.5）在 24 GiB 時 Belady 約比 LRU 高 8 點，到 1 TiB 時 FreeInference 上兩者幾乎重合、Chutes 上只差約 1–2 點（讀圖）；§5 說 LRU 在每條量過的 trace 上都會「reliably converging to the optimal offline ceiling」（p.9）；App. C 說 1 TiB 時 FreeInference 的 compute-savings 連兩個 oracle 都在 0.2 點內（p.19）。〔複核修正：原本引的「1 TiB 時 LRU 離最佳**線上**演算法不到 0.4 點」比的是線上演算法、不是 Belady，且只涵蓋 §4.6 的三條 trace，不能當證據。〕另須註明：原文**摘要**（p.1）與 §3.2（p.4「Belady's offline oracle still sits far above the best online algorithm」）本身就沒加容量限定，sota 的說法是跟著摘要走，問題在原文正文與摘要不一致。<br>• intro.txt §5.2 的「生產 trace 上連 Belady 上界都還有明顯空間」：〔複核：⚠️ 建議成立，但不算誤引〕intro 沒寫「各種容量」，與原文摘要措辭一致；建議補「在 HBM 容量下」，理由同上。<br>• sota／intro 的「14 種逐出法……大多沒有比 LRU 好」：14 種含 LRU 本身（p.4 §3.2 的分組：LRU 1＋快速降級 5＋LHD 1＋頻率 2＋學習式 3＋前綴專用 2），被拿來和 LRU 比的是 13 種〔複核：✅ 事實成立；但原文 p.1 自己也寫「evaluate 14 advanced algorithms … most offer little improvement over LRU」，使用者文件是照原文說法，屬於小瑕疵〕。另外 AsymCache 被判為「≈LRU」用的是 uniform 成本的 hit ratio（p.4「under a uniform cost model」），這不是 AsymCache 的目標指標；而且 Fig. 12 的 compute-savings 比較沒有放 AsymCache（p.10 圖例）〔判讀〕。〔複核補充：AsymCache 原文 p.5 自己寫「若所有 block 的重算成本相同，演算法退化為 LRU」；Fancy 模擬中 AsymCache 內部用哪種成本，原文未說明〕<br>• intro 表「成本感知逐出 TTFT ↓19.9%」、sota「從節點開頭整段丟，比 LRU 少 19.9% TTFT」：〔複核：✅ 指控成立〕數字正確，但只是單一真引擎點（48 GiB、10K 請求、單張 H200）的**平均** TTFT（p.11「cuts average TTFT by 19.9%」）；中位數變差（112→171 ms）。「開頭」指的是 radix **節點**的開頭（p.11「from the beginning of a node」），不是整段 context 的開頭；選哪個節點靠隨機取樣與分數。〔複核修正：sota §2.11 已寫「從**節點**開頭」，這部分沒錯；漏掉「節點」的是 sota §2.11「跟本研究的關係」的「『從開頭整段逐出』是位置感知的近親」與 sota §5（新穎性威脅）表中的「從開頭整段逐出的成本感知法」〕<br>• intro §8.6 與 sota 的「SSD 卸載多半沒有必要」：原文確實這樣寫，但出自 §6.3 的討論，根據是 block 壽命，**沒有做 SSD 實驗**（p.12）〔複核：✅〕。<br>• intro §8.6「平均 18.7K–32K、session 中位數不到 1 分鐘」：已核對一致（p.3 Table 1；p.6 Fig. 3b 中位 55 s）。但 session 長度統計只來自 FreeInference〔複核：✅，§4 的刻畫只拿 FreeInference 對照 Wikipedia／CloudPhysics，p.4–5〕。<br>• sota「session 壽命……p99 約 4 小時」：已核對一致（p99 15,006 s；p.5 正文寫「99th percentile ∼4 hours」）〔複核：✅〕。<br>• `workloads_eval.md` §2.4「文獻慣例是 partial、前綴算到第一個缺口為止」：本篇是反例，洞之後的 block 也算命中〔判讀〕 | 見格內 |
| 對本研究的意義〔判讀〕 | • 可沿用：<br>　◦ 兩個容量 regime 的設定<br>　◦ compute-savings ratio（換成我們的位置成本模型 κ(pos)）<br>　◦ **Belady vs BeladyCompute 的差距診斷**，可以直接回答「位置感知值不值得做」<br>　◦ ILP 上界<br>　◦ 密掃容量<br>• 要小心：<br>　◦ 結論「SSD 沒必要」與「LRU 在大池收斂」都對我們的多層前提不利；必須用長 context、長間隔的 trace 檢驗<br>　◦ 它的真引擎驗證只有一點<br>　◦ 洞之後的命中需要引擎支援，對應我們的「區段命中」問題 | 〔判讀〕 |

---

### AsymCache：Multi-Segment Attention: Enabling Efficient KV-Cache Management for Faster Large Language Model Serving（arXiv 2026；arXiv 2606.02964）

- **讀了什麼**：〔全文〕v1（2026-06-01），https://arxiv.org/pdf/2606.02964v1 ，查證 2026-10-06。**已確認 AsymCache 就是這篇的系統名**（摘要與內文多次出現，p.1）。
- **一句話**：在 GPU 內的逐出決策裡，同時考慮重用機率與「隨位置增加的重算成本」，並用 MSA kernel 支援不連續的命中。
- **評測要證明的主張**：
  * 綜合重用機率與位置重算成本的逐出（f_B(t)·ΔT_B），在 TTFT／TPOT 上勝過 LRU、Max-score、Pensieve+MSA（p.9）。
  * O(log n) 的逐出演算法開銷可以忽略（p.6–7）。
  * 可以疊在 Continuum 這類 agent 系統上（p.11–12）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | Llama 3.1-8B-Instruct（32 層、hidden 4096、1 GPU）、Llama 3.1-70B-Instruct（80 層、hidden 8192、4 GPU TP），都是 GQA。KV dtype 沒寫〔未查證〕 | p.8 §6.1、Table 1 〔原文〕 |
| 硬體 | AMD EPYC 9K84（96 核）＋4× NVIDIA H20 96 GB，NVLink；CUDA 12.8；容器內，不限 CPU 與記憶體。**KV 只放在 GPU**，沒有 CPU／SSD 層（p.12 §7 自承） | p.8 §6.1；p.12 〔原文〕 |
| 軟體與版本 | 改 vLLM（版本沒寫）：Python 約 6K 行、C++／CUDA 約 2K 行；MSA 以 CUDA＋CUTLASS 擴充 FlashAttention／PagedAttention。所有系統都整合 POD-Attention | p.8 §5.3；p.9 〔原文〕 |
| 資料／負載 | LongBench 與 LooGLE。照 LoopServe 的做法，把 QA 對組成多輪對話；每個資料集只取 300 個請求。為了消除取樣隨機性，先產生輸出，再逐步改寫輸出 token，讓每次跑的輸出長度一致。另用 BFCL v4 的 Web Search 部分當 agent 負載，輸出用 GPT-5.1 預先產生 | p.8 §6.1；p.11 §6.5 〔原文〕 |
| 長度 | 平均 input／output：LongBench 約 34.8K／2.6K，LooGLE 約 24.4K／0.7K。單位是 token〔判讀：原文寫 K，沒標單位〕。快取空間：8B 是 487,744 tokens，70B 是 505,152 tokens | p.8 Table 1、§6.1 〔原文〕 |
| 到達與併發 | 每個 session 第一輪的到達間隔取自 Gamma 分布（CV 0.25），同一 session 內的輪間間隔取自另一個 Gamma 過程。「跨 session 到達率 ÷ session 內到達率」設兩種：5:1（低分散）、10:1（高分散）。QPS 在文中出現 0.02 與 0.04；圖 11–12 的軸是 0.01／0.02／0.04（圖字型需解碼） | p.8 §6.1；p.9–10 〔原文〕 |
| 重用結構 | 多輪對話：共享前綴（系統提示）加上 session 內歷史。agent：工具呼叫的迴圈。原文的觀察：命中集中在序列兩端（短的共享前綴、幾乎完整的歷史），中段較少（Fig. 3） | p.3 §3 〔原文〕 |
| 掃描的自變數 | QPS（0.01–0.04）、分散度（5:1／10:1）、模型（8B／70B）、資料集；超參數敏感度：lifespan、reuse probability、slope change ratio | p.9–11 〔原文〕 |
| 對手 | • vLLM-LRU<br>• Max-score：作者自己實作 in-the-wild 的策略 [50]，O(n) 選受害者，用 Eq. 9 估重用機率<br>• Pensieve+MSA：Pensieve 沒開源，自己重做並接上 MSA<br>• agent 實驗另比 Continuum，以及 Continuum+AsymCache<br>對手的參數沒有說明 | p.8–9 §6.1；p.11 §6.5 〔原文〕 |
| 系統指標 | 平均 TTFT、平均 TPOT；agent 實驗用平均與 P90 job latency。另報 request 級與 block 級的命中率 | p.8 §6.1；p.11 〔原文〕 |
| 模擬器／真機與驗證 | 全部是真引擎。沒有模擬器 | 〔原文〕 |
| 命中率定義與換算 | 報 request 級與 block 級的 hit rate，**兩者的公式都沒有給**。Table 2 只標 "Hit(%)"，沒說粒度；Fig. 14 標明是 block 級。命中與延遲直接量 TTFT，不做換算。允許多段命中（前綴＋後綴），所以也不是「前綴算到第一個缺口」的語意 | p.8；p.10–11 〔原文〕〔判讀〕 |
| 上界／Oracle | 無 | 〔原文〕（全文未見） |
| 品質指標 | 不適用：無損、bitwise 相同的輸出 | p.1 〔原文〕 |
| 主要結果 | • 70B：TTFT 分別比 vLLM-LRU／Max-score／Pensieve+MSA 快最多 1.86×／1.91×／1.86×；TPOT 最多 1.62×／1.63×／1.71×（p.9 §6.2）。<br>• LongBench qps=0.04（有排隊）加上 chunk 排程器：TTFT 比 vLLM-LRU 少最多 60%（p.9–10）。<br>• Table 2（8B、LongBench、qps 0.02）：<br>　◦ 低分散：TTFT 5.458 s vs LRU 6.582 s；命中 29.1% vs 28.7%<br>　◦ 高分散：7.068 s vs 8.695 s；命中 12.50% vs 2.09%（p.10）<br>• Continuum+AsymCache 對 Continuum：平均 job latency −4.4% 到 −18.1%（p.12）。<br>〔複核修正：原寫「數字前後不一……§6.2 最大只到 1.91×」，**不成立**〕§6.2 的 1.86／1.91／1.86× 明寫是「for example, with the 70B model」（p.9），對應 Fig. 11 的 70B＋LooGLE、低分散、QPS 0.04。摘要的 **1.90–2.03×** 出自 Fig. 11 的 **8B＋LooGLE、低分散、QPS 0.04** 這一組：對 vLLM-LRU／Max-Score／Pensieve+MSA 依序 1.90×／1.98×／2.03×（p.9 圖上數字標籤）。摘要的 TPOT 1.62–1.71× 出自 Fig. 12 的 70B＋LongBench、高分散、QPS 0.04（1.62×／1.63×／1.71×，p.10）。真正的不一致只剩措辭：結論（p.12）寫「**average** 1.90–2.03×」，但這是單一設定下對三個對手的**最大值**，不是平均；摘要寫「up to」才對 | 見各格 〔原文〕 |
| 消融／敏感度／開銷 | • O(log n) vs O(n)：2.4K 請求、8K blocks 時，控制面時間約 500 s（每請求約 200 ms，約 prefill 的 10%）vs 約 5 s（約 2 ms，0.1%）（p.6–7）。<br>• MSA 單 kernel vs 兩個 kernel：1K 與 10K 長度、128 個新 token（p.10–11）。<br>• 超參數：slope change ratio 取 40；lifespan 的轉折點取 P99；reuse probability 在 0.3–0.7 都接近最佳（p.6、p.11）。<br>• 成本模型：用約 1.1K 筆剖析擬合線性模型，R²>0.999（p.5） | 〔原文〕 |
| 重複與統計 | 未說明 | 〔原文〕（全文未見） |
| 程式碼／資料 | 沒有連結；p.6 提到「見 artifact 的證明」，但沒有 URL | 〔原文〕；〔未查證〕 |
| 設計理由（原文） | • 前綴與後綴都值得快取（Obs. 1）；命中集中在兩端（Obs. 2）（p.3）。<br>• 頻率項用分段指數函數，因為只有指數函數符合 order-preserving 規則，才能用平衡樹做到 O(log n)（p.5–6）。<br>• lifespan 取 P99 是保守做法（p.7–8）。<br>• 只取 300 個請求，原文說代表性子集上的系統行為與全集很接近（p.8） | 〔原文〕 |
| 設計理由〔判讀〕 | 為了消除輸出長度的隨機性而重寫輸出，是好做法。但 QPS 只有 0.01–0.04，負載很輕；搭配單 GPU 約 49 萬 token 的快取，容量壓力主要來自長 context 與分散度，而不是併發 | 〔判讀〕 |
| 原文沒講清楚的地方 | 1. request／block 命中率的定義。<br>2. 「300 requests」指的是 session 數還是請求數。<br>3. Gamma 過程的平均間隔（絕對秒數）。<br>4. vLLM 版本與 dtype。<br>5. ~~摘要「2.03×」的來源~~〔複核修正：已查到，出自 Fig. 11 的 8B＋LooGLE、低分散、QPS 0.04 對 Pensieve+MSA，p.9〕。<br>6. Max-score 用的 lifespan 參數 | 〔原文〕 |
| 與既有整理不一致 | • ~~intro 表 AsymCache「TTFT 1.90–2.03×」：與摘要、結論一致，但與 §6.2 正文（最大 1.86／1.91／1.86×）對不上。建議引用 §6.2 的數字，並註明對手。~~〔複核修正：❌ 指控不成立。§6.2 的 1.86／1.91／1.86× 只是「70B 的例子」；1.90／1.98／2.03× 就在 Fig. 11（8B＋LooGLE、低分散、QPS 0.04，p.9）。intro 表 6 的欄位定義是「各篇報告的最佳值」，引用 1.90–2.03× 沒有錯。可以補充的只是：這是同一設定下對三個不同對手的倍數，不是跨設定的範圍；sota §2.10「TPOT 快 1.62–1.71 倍」同理，出自 Fig. 12 的 70B＋LongBench 高分散 QPS 0.04（p.10）〕<br>• sota §2.10「AsymCache〔摘要〕……要讀全文」：本卡已讀全文。「逐出時同時考慮命中率和依位置而變的重算成本；MSA 支援不連續 KV」：已核對一致（p.4–6）。<br>• intro §6「位置感知逐出」、使用者提示中的「位置感知的重算成本逐出」：已核對一致。<br>• intro 表 6 的「主要限制」欄寫「細節待讀全文」，可以補上：**只在 GPU 內、沒有 CPU／SSD 層**（p.12 §7 自承）；負載是 QPS 0.01–0.04，每個資料集 300 個請求。intro 表 14 把「多層」列為我們與它的差異，與原文一致。<br>• PAPERS_BY_LEVEL「用『重算成本隨位置線性增加』做逐出，並拿 Pensieve 當 baseline」：已核對一致（Eq. 7 的 ΔT_B＝2k₅(l₁+q₁)+常數；Pensieve 是自己重做的） | 見格內 |
| 對本研究的意義〔判讀〕 | • MSA 是「前段不存、後段命中」在引擎上可行的證據；它的 1K／10K 微基準可以拿來對照我們的區段命中開銷。<br>• 它的線性成本模型（Eq. 4–7）是單 GPU 的特例；我們需要跨層、跨精度的成本。<br>• 它在 Fancy 的 uniform 成本模擬裡約等於 LRU：**評測指標的選擇決定了位置感知策略看起來有沒有用**。我們的 PoC 必須同時報 hit ratio 與成本加權指標 | 〔判讀〕 |

---

### Marconi：Prefix Caching for the Era of Hybrid LLMs（MLSys 2025；arXiv 2411.19379）

- **讀了什麼**：〔全文〕arXiv v3（2025-04-10，MLSys'25 proceedings 版頁腳），https://arxiv.org/pdf/2411.19379v3 ；〔程式碼〕https://github.com/ruipeterpan/marconi commit `0801661`（2025-03-05）。查證 2026-10-06。
- **一句話**：為 Attention＋SSM 混合模型設計前綴快取的准入與 FLOP-aware 逐出。
- **評測要證明的主張**：
  * 選擇性准入讓 token 命中率比細粒度 checkpoint（vLLM+）高 4.5–34.4×。
  * FLOP-aware 逐出讓 token 命中率比 LRU（SGLang+）高 19.0–219.7%，並降低 P95 TTFT。
  * 序列越長、SSM 比例越高、state 維度越大，效果越好（p.7）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | • 主結果：7B 混合模型，Attention／SSM／MLP 各 4／24／28 層。artifact 說預設設定是 "NVIDIA's Attention-Mamba2 7B Hybrid"。<br>• TTFT：Jamba-1.5-Mini（12B active／52B total），state 維度 128，用 vLLM 實作跑在 4×A100-40GB。<br>• 全部 FP16 | p.8 §5.1；p.16 B.7 〔原文〕 |
| 硬體 | AWS p4d.24xlarge：8× A100-40GB、96 vCPU（Xeon 8275CL）、1152 GB DDR4。artifact 是在 CloudLab 節點測的（Python 3.11.9） | p.8 §5.1；p.16 B.3.2 〔原文〕 |
| 軟體與版本 | 擴充 vLLM 與 SGLang 支援混合模型。**命中率實驗是 Python 的 trace 驅動模擬**：`policy_exploration.py`＋`radix_cache_hybrid.py`。tokenizer 一律用 Llama-2-7b-hf | p.8；p.15–16 Artifact 〔原文〕〔程式碼〕 |
| 資料／負載 | LMSys-Chat-1M、ShareGPT，以及 SWE-Agent 在 SWE-Bench 上的軌跡。程式碼裡每條 trace 取前 100 個 session（`num_sessions=[100]`） | p.8 §5.1；`policy_exploration.py` L161–175 〔原文〕〔程式碼〕 |
| 長度 | 分布見 Fig. 6：LMSys 多數 <10K，ShareGPT 多數 <2K，SWEBench 從數百到數萬。單位是 token。程式碼裡 session 累積超過 32,768 token 就截斷；SWEBench 另外最多 50 輪 | p.7 Fig. 6；p.8–9；`utils/generate_trace.py` L88、L271 〔原文〕〔程式碼〕 |
| 到達與併發 | • 原文只說會改變 session 間與請求間的到達時間（p.8）。<br>• 程式碼：session 的開始時間是 `session_id / sessions_per_second`，**固定間隔，不是 Poisson**。<br>• LMSys／ShareGPT 的輪間間隔＝使用者輸入字數 ÷ 打字速度（90 字／分）。<br>• SWEBench 的輪間間隔是 Poisson(λ＝平均回應時間 5／7.5／10 s)。<br>• sessions_per_second 掃 0.25–10 | p.8；`utils/generate_trace.py` L46–99、L268 〔原文〕〔程式碼〕 |
| 重用結構 | 只有「純輸入共享前綴」與「輸入＋輸出的對話歷史」兩類（原文的分類法） | p.5 §4.1 〔原文〕 |
| 掃描的自變數 | • 快取大小：原文 Fig. 11 是 60–140 GB 五點；程式碼主掃描是 LMSys 1–5 GB、ShareGPT 1–10 GB、SWEBench 40–100 GB<br>• session 到達率 0.5–2／s<br>• 輪間回應時間 5–10 s<br>• 層組成（SSM:Attn）<br>• SSM state 維度 16–128 | p.9–10 §5.4；`policy_exploration.py` L161–175 〔原文〕〔程式碼〕 |
| 對手 | • Vanilla（不做前綴快取）<br>• vLLM+：每個 32-token block 都存 state；32 是 vLLM 支援的最大 block，原文說這樣對 vLLM+ 有利<br>• SGLang+：加上與 Marconi 相同的准入，但逐出仍用 LRU<br>都是作者自己擴充的 | p.8 §5.1 〔原文〕 |
| 系統指標 | P5／P50／P95 TTFT（ms）。〔程式碼〕TTFT 不是端到端量的：先用 Jamba-1.5-Mini 剖析出「TTFT vs 序列長度」表，再用（input 長度 − 命中長度）查表，最後取 P95；沒有排隊。程式碼註解還提到剖析時單位換算錯誤，所以乘了 10 | p.8；`plotting/ttft.py` L14–23、L195–201 〔原文〕〔程式碼〕 |
| 模擬器／真機與驗證 | 命中率全部來自 Python 模擬，TTFT 由查表推得。**沒有拿模擬器與真引擎比對** | 〔程式碼〕；原文未見驗證 |
| 命中率定義與換算 | token hit rate＝跳過 prefill 的 token 數 ÷ 總 input token（p.8）。程式碼同時算 request hit rate 與 FLOPs saved。原文說 token 命中率可以近似省下的 FLOP，因為 prefill 是算力瓶頸（p.8） | p.8 〔原文〕；`radix_cache_hybrid.py` L204–214 〔程式碼〕 |
| 上界／Oracle | 論文沒有。artifact 有「V3：靜態 α 的離線最佳」，但結果沒有放進論文（p.16 B.6） | 〔原文〕 |
| 品質指標 | 不評：重用是精確的（p.8） | 〔原文〕 |
| 主要結果 | • token 命中率平均比 vLLM+ 高 4.5×／7.3×／34.4×（LMSys／ShareGPT／SWEBench）（p.8）。<br>• 對 SGLang+ 的 P95 勝幅：LMSys 45.6%、ShareGPT 19.0%、SWEBench 219.7%（p.8）。<br>• P95 TTFT 比 vLLM+ 多降 36.1%／71.1%／46.8%（275.4／103.3／617.0 ms）（p.9）。<br>• SWEBench 細看：SGLang+ 命中 16.4%，Marconi 32.7%；<7K 的序列最多少 3.0%，>7K 的最多多 25.5%；P5 TTFT 差 6.3%（p.9 §5.3） | 〔原文〕 |
| 消融／敏感度／開銷 | 快取競爭程度（5 種大小）、層組成、state 維度、到達率（p.9–10）。α 的格點搜尋原文說通常只要幾秒（p.7）；〔程式碼〕格點是 0–2.0、步長 0.1 | p.7；`config_tuner.py` L61 〔原文〕〔程式碼〕 |
| 學習式協定 | • α 起始為 0（即 LRU），直到第一次逐出。<br>• 接著進入 bootstrap 期，長度是「第一次逐出前看到的請求數」的 5–15 倍，期間繼續用 LRU。<br>• 然後重播 bootstrap 期的請求做格點搜尋，取命中率最高的 α（p.7）。<br>• 〔程式碼〕只調一次；「continuous tuning」被標成 legacy。<br>• 沒有區分調參期與量測期，命中率包含 bootstrap 期 | p.7 〔原文〕；`radix_cache_hybrid.py` L113、L147–153 〔程式碼〕 |
| 重複與統計 | Fig. 7 用箱形圖，顯示多個設定的分布（P5–P95）；沒有重複次數 | p.8 〔原文〕 |
| 程式碼／資料 | 公開（CC BY-NC），附 artifact appendix 與處理後的 trace（Google Drive） | p.15–16 〔原文〕 |
| 設計理由（原文） | • vLLM+ 用 block 32 是為了對它有利（p.8）。<br>• 不評下游品質，因為重用是精確的（p.8）。<br>• 使用 FLOP efficiency，因為 SSM 狀態的大小與它省下的計算無關，size 不再能當成本代理（p.7） | 〔原文〕 |
| 設計理由〔判讀〕 | 用查表推 TTFT，可以快速掃大量設定，但沒有排隊與批次干擾，所以 TTFT 的改善是上限性質的 | 〔判讀〕 |
| 原文沒講清楚的地方 | 1. Fig. 11 的 60–140 GB 是在哪個資料集與設定下跑的；程式碼主掃描的 SWEBench 是 40–100 GB。<br>2. 快取是 GPU 還是 CPU：原文只是單層容量。<br>3. TTFT 是查表推得的，這點正文沒有說明。<br>4. 〔程式碼〕`radix_cache_hybrid.py` L526 的 MLP FLOP 差值把父節點那項寫成 `get_attn_flops`，疑似筆誤，影響未查證 | 〔原文〕〔程式碼〕 |
| 與既有整理不一致 | • intro 表「token 命中 ↑34.4×；只在 GPU」：34.4× 是 SWEBench 對 vLLM+ 的平均。「只在 GPU」原文沒說，實際是單層容量的模擬〔判讀〕〔複核：✅ p.8 §5.2「average of 4.5×, 7.3×, and 34.4×」；全文未說快取在 GPU 或 CPU，背景段只泛稱 GPU/CPU memory（p.3）〕。<br>• ~~PAPERS_BY_LEVEL「准入：預測一個項目未來被重用的機率」：**不準確**。原文是規則式分類：投機插入時找出分支點的 state，加上最後一個 decode token 的 state，**沒有算機率**（p.5）。sota S1「依命中情境估重用機率」也一樣。~~〔複核修正：❌ 指控不成立。原文摘要自己寫准入依據是「forecasts of their reuse likelihood across a taxonomy of different hit scenarios」（p.1），§4.1 也寫「to estimate the reuse likelihood, Marconi uses a radix tree」（p.5）。PAPERS_BY_LEVEL 與 sota S1「依命中情境估重用機率」是原文措辭的忠實改寫。可以**補充**（不是更正）：這個「估計」是規則式分類——投機插入時發現分支點就存該處 state，另外一律存最後一個 decode token 的 state——沒有算出數值機率（p.5）。〕<br>• PAPERS_BY_LEVEL「TTFT 少 71.1%」：71.1% 是 ShareGPT 上「比 vLLM+ 多降的 P95 TTFT 百分比」。摘要把它與 SWEBench 的 617 ms 並列，而且 TTFT 是查表推得的。〔複核：✅ 摘要「71.1% or 617 ms lower TTFT」（p.1）；p.9 §5.2：對 vLLM+ 的 P95 TTFT 多降 36.1%／71.1%／46.8%（275.4／103.3／617.0 ms），71.1% 是 ShareGPT、617 ms 是 SWEBench。PAPERS_BY_LEVEL 照抄摘要，數字本身出自原文〕<br>• PAPERS_BY_LEVEL「自建 7B 混合模型」：原文只寫層數；artifact 說是 NVIDIA Attention-Mamba2 7B 的設定；命中率模擬並沒有真的跑這個模型。〔複核：✅ p.8 §5.1 只寫 {4,24,28} 層；p.16 B.7 寫預設設定是 NVIDIA 的 Attention-Mamba2 7B Hybrid；`radix_cache_hybrid.py` 只用層數與維度算 state 大小與 FLOPs，不跑模型〕<br>• intro §5.5「剛啟動時先用 LRU，再線上調整權重」與 sota S1「線上調整權重」：前半一致；後半應改成「bootstrap 期結束時格點搜尋**一次**」（程式碼）〔複核：⚠️ 部分成立。論文正文只說 bootstrap 後非同步啟動格點搜尋、取命中率最高的 α（p.7），沒說只做一次或會重做；「線上」與正文不衝突。「只做一次」只有程式碼支持：`radix_cache_hybrid.py` L147 的觸發條件是 `len(self.request_history) == self.bootstrap_window_size`，只會成立一次；L113 把 continuous tuning 標為 legacy（commit `0801661`）。建議改寫為「bootstrap 結束時調一次（依程式碼）」，而不是說原說法錯〕 | 見格內 |
| 對本研究的意義〔判讀〕 | • 可沿用：token hit rate 的定義；「FLOP 效率＝省下的計算 ÷ 佔用空間」可以推廣成我們每個動作的 κ。<br>• 不可比：TTFT 是查表推的。<br>• 要小心：它的 session 開始時間是固定間隔，100 個 session 很少；我們的 PoC 若要引用它，要重新跑 | 〔判讀〕 |

---

### AdaptCache：AdaptCache: KV Cache Native Storage Hierarchy for Low-Delay and High-Quality Language Model Serving（SOSP'25 BigMem workshop；arXiv 2509.00105）

- **讀了什麼**：〔全文〕arXiv v2（2026-01-15）＋BigMem'25 PDF（3 頁），兩者內文相同。https://arxiv.org/pdf/2509.00105v2 。查證 2026-10-06。
- **一句話**：對每段 KV 決定壓縮法、壓縮率與放 DRAM 或 SSD，以提高 DRAM 命中。
- **評測要證明的主張**：在同品質下延遲省 1.43–2.4×；在同延遲下品質高 6–55%；對手是固定壓縮率的方法（p.1–2）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | 只有 Llama-3.1-8B-Instruct | p.2 〔原文〕 |
| 硬體 | 1× NVIDIA A100（100 GB DRAM、400 GB SSD）；磁碟讀取 1 GB/s | p.2 §4 〔原文〕 |
| 軟體與版本 | 沒寫引擎與版本 | 〔原文〕（未見）；〔未查證〕 |
| 資料／負載 | 「1,100 contexts from six LongBench datasets」。對照參考文獻是 SAMSum、LCC（LongCoder）、TriviaQA、RepoBench-P、HotpotQA、QMSum，涵蓋摘要、QA、程式三類任務。〔計算〕LongBench v1 官方筆數依序是 200／500／200／500／200／200，合計 1,800，**對不上 1,100**；原文也沒說怎麼抽樣 | p.2 §4；LongBench task.md 〔原文〕〔計算〕 |
| 長度 | 原文沒寫。〔文件〕官方平均長度（詞數）：SAMSum 6,258、LCC 1,235、TriviaQA 8,209、RepoBench-P 4,206、HotpotQA 9,151、QMSum 10,614 | LongBench task.md L5–25 〔文件〕 |
| 到達與併發 | Poisson 到達，請求率有多種，數值沒寫。**哪個 context 被重用幾次（流行度分布）沒有說明** | p.2 §4 〔原文〕 |
| 重用結構 | 同一段 context 被多個 query 重用。query 從哪來沒寫，只有估計器用 GPT-4o 產生問題 | p.2 §3 〔原文〕 |
| 掃描的自變數 | α（延遲與品質的權重）、請求率；圖 2 是品質–TTFT 曲線 | p.2 〔原文〕 |
| 對手 | Without Compression（DRAM＋SSD 卸載）、KIVI LRU、StreamingLLM LRU（固定壓縮率、LRU 卸載）、Prefill（重算） | p.2 §4 〔原文〕 |
| 系統指標 | TTFT（延遲） | p.2 〔原文〕 |
| 模擬器／真機與驗證 | 原文說是在 A100 上跑實驗，沒有提到模擬 | p.2 〔原文〕 |
| 命中率定義與換算 | 只有「DRAM cache hit rate」：KIVI LRU 2-bit 是 38%，AdaptCache 依 α 在 81／56／44／11%。定義沒給 | p.2–3 〔原文〕 |
| 上界／Oracle | 無。原文說最佳解是 NP-hard（多選擇背包問題），所以用 greedy（p.2） | 〔原文〕 |
| 品質指標 | 與「原始 prefill 答案」的相似度，用任務指標 F1／ROUGE-L／CodeBLEU（p.1 腳註 1） | 〔原文〕 |
| 主要結果 | • 對 prefill 與卸載：TTFT −56%，品質下降在 15% 以內。<br>• 對 KIVI：同品質下 TTFT −69%。<br>• 對 StreamingLLM：同 TTFT 下品質 +15–89%（p.2）。<br>• 摘要寫的是 1.43–2.4× 與 6–55% | 〔原文〕 |
| 消融／敏感度／開銷 | 無 | 〔原文〕 |
| 學習式協定 | 估計器離線剖析：<br>• 用 dummy question 量傳輸延遲與解壓開銷；<br>• 每個資料集抽 10 筆，用 GPT-4o 產生問題，畫出「品質–壓縮率」曲線；<br>• 未來命中頻率用歷史命中頻率估（p.2 §3）。<br>剖析資料與評測資料是否分開，沒說明 | 〔原文〕 |
| 重複與統計 | 未說明 | 〔原文〕 |
| 程式碼／資料 | 未公開 | 〔原文〕；`workloads_eval.md` §3.5 〔二手〕 |
| 設計理由（原文） | 先前把 KV 放在 DRAM＋SSD 的做法，多數命中落在慢的 SSD；有損壓縮可以把更多 entry 留在 DRAM（p.1） | 〔原文〕 |
| 設計理由〔判讀〕 | 這是 workshop 的初步結果，評測規模本來就小 | 〔判讀〕 |
| 原文沒講清楚的地方 | 1,100 筆怎麼抽；請求率數值；context 流行度分布；query 來源；AdaptCache 可選的壓縮法有哪些；DRAM 命中率的分母 | 〔原文〕 |
| 與既有整理不一致 | • sota §2.7「品質評估很薄（AdaptCache 每個資料集抽 10 筆）」：**需要更正**。10 筆是估計器畫品質曲線用的剖析樣本；評測用的是 1,100 段 context。不過評測品質是用什麼 query 量的，原文沒說，所以「薄」的判斷仍成立。PAPERS_BY_LEVEL 寫的是「品質曲線每個資料集只抽 10 筆」，正確。〔複核：⚠️ 事實成立、但「需要更正」說重了。原文 p.2 §3：估計器「samples ten entries from each dataset」，用 GPT-4o 產生的問題畫品質–壓縮率曲線；評測是「1,100 contexts from six LongBench datasets」（p.2 §4）。sota 這句列在「做不到」（方法限制），前一句是「未來命中頻率用過去估」，「品質評估」可讀成系統內的品質**估計**（＝估計器只用 10 筆），這樣讀就沒錯；若讀成**評測**就錯。建議改寫成「品質曲線每個資料集只用 10 筆剖析」以消除歧義，而不是當成事實錯誤〕<br>• sota「效用 ＝ 預估的未來命中頻率 ×（品質的權重 − 大小 ÷ 頻寬）」：已核對一致（p.2）。<br>• intro 表把 AdaptCache／EvicPress 合併成一列，標「BigMem'25、TTFT 2.19×」：2.19× 屬於 EvicPress；AdaptCache 是 1.43–2.4×。EvicPress 只是 arXiv，不是 BigMem。〔複核：✅ AdaptCache 摘要 1.43–2.4×（p.1）；EvicPress 摘要 2.19×（p.1），PDF 只標 arXiv v1，未見會議名；sota §2.7 標題已分開寫 BigMem'25／arXiv，問題只在 intro 表 6 的合併列〕<br>• `workloads_eval.md` §3.5：「three」與「six」的矛盾、1,800 筆對不上 1,100、未公開——已核對一致 | 見格內 |
| 對本研究的意義〔判讀〕 | 只能當「壓縮＋放置聯合決定」的概念先例。評測設定無法重現，不能當數字對手 | 〔判讀〕 |

---

### EvicPress：EvicPress: Joint KV-Cache Compression and Eviction for Efficient LLM Serving（arXiv 2025；arXiv 2512.14946）

- **讀了什麼**：〔全文〕v1（2025-12-16），https://arxiv.org/pdf/2512.14946v1 ，查證 2026-10-06。
- **一句話**：以效用函數為每段 context 同時決定壓縮法、壓縮率與存放層（GPU／CPU／SSD）。
- **評測要證明的主張**：
  * 在相同品質下，TTFT 比「固定壓縮＋LRU」快 1.43–3.77×；比「只逐出」或全 prefill 快 1.22–2.19×，品質下降小於 3%（p.3、p.7）。
  * 在品質目標 80% 下，吞吐高 2.0–3.6×（p.9）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | Llama-3.1-8B-Instruct、Qwen2.5-14B-Instruct、LongChat-7b-v1.5-32k、Mistral-7B-Instruct-v0.3、Qwen3-30B-A3B-Instruct-2507（MoE）。KV 大小：Qwen3-30B-A3B 每 1K token 0.0915 GB，Llama-3.1-8B 每 1K token 0.12 GB | p.8；p.10 〔原文〕 |
| 硬體 | 1× H100 80GB；配置 80 GB CPU DRAM、800 GB SSD；遠端儲存假設無限 | p.8 §6.1 〔原文〕 |
| 軟體與版本 | vLLM v0.11.2＋LMCache v0.3.9post2＋PyTorch v2.9，自己加了約 3K 行 Python | p.7 §5 〔原文〕 |
| 資料／負載 | LongBench 的 12 個資料集，隨機抽 555 段 context。每段用 GPT-5 產生 100 題：50 題用來訓練剖析器，50 題用來測試。週期重剖析的實驗另用 Azure trace 的時間戳，內容換成自己產生的題目 | p.8；p.10 §6.3 〔原文〕 |
| 長度 | Table 2 的平均長度從 12K 到 108K，**沒有標單位**。〔計算〕除以 LongBench 官方平均詞數，12 個資料集的比值都在 5.4–6.6，判讀為**字元數**（與 `workloads_eval.md` §3.5 的計算一致）。§3 的 Table 1 是另一組長度，也沒有單位 | p.3 Table 1；p.7 Table 2；LongBench task.md 〔原文〕〔計算〕 |
| 到達與併發 | 掃 QPS（Fig. 8：0–15、0–30、0–7.5），到達分布沒寫。**各 context 被存取的頻率分布沒有說明**，但效用函數裡有 frequency 這一項 | p.9 Fig. 8 〔原文〕 |
| 重用結構 | 同一段 context 被多個 query 重用（文件 QA 型） | p.8 〔原文〕 |
| 掃描的自變數 | α（品質與延遲的權衡）、各對手的壓縮率或保留比例、QPS、模型、資料集 | p.6；p.8–10 〔原文〕 |
| 對手 | • Prefill：vLLM v0.11.2，不做前綴快取<br>• Eviction only：LRU，不壓縮<br>• keydiff、knorm、snapkv 三種壓縮各加 LRU：改壓縮率畫出 trade-off<br>• IMPRESS：用 8 個 attention head 中的 3 個找重要 token，保留 top X%，改 X 畫曲線<br>是否重做，原文沒有明說〔判讀：自己實作〕 | p.8 §6.1 〔原文〕 |
| 系統指標 | • TTFT：從 query 到達到第一個 token，含從儲存層取 KV 與 prefill。<br>• ITL；端到端延遲。<br>• 吞吐比較：在相同 TTFT 或 ITL 下可承受的 QPS | p.9 〔原文〕 |
| 模擬器／真機與驗證 | 真引擎（vLLM＋LMCache）；沒有模擬器 | 〔原文〕 |
| 命中率定義與換算 | Fig. 9 用「CPU-hit request %」與「SSD-hit request %」（請求級，分層）；正文又寫「cache hit tokens」，兩種混用。定義沒有給 | p.9–10 〔原文〕；與 `workloads_eval.md` §2.1 一致 |
| 上界／Oracle | 無。配置選擇是多選擇背包問題（NP-hard），用 greedy 解（p.7、p.16 Alg. 1） | 〔原文〕 |
| 品質指標 | quality score：壓縮後答案與未壓縮答案，用 MiniLM-L6-v2 嵌入算 cosine 相似度。**比的對象是未壓縮的模型答案，不是標準答案** | p.8–9 〔原文〕 |
| 主要結果 | • 對壓縮＋LRU：同品質下 TTFT 快 1.43–3.77×，同 TTFT 下品質高 13.58–55.40%。<br>• 對全 prefill：品質下降 3% 以內時快 1.29–2.19×。<br>• 對 LRU 只逐出：快 1.22–1.56×。<br>• 對 IMPRESS：快 1.5–5.2×（p.9）。<br>• 在 12 個資料集中有 11 個改善；例外是 musique（p.11） | 〔原文〕 |
| 消融／敏感度／開銷 | • 週期重剖析：品質增益約 11%，代價是剖析期間的延遲尖峰（p.10 Fig. 10）。<br>• 配置分布（Fig. 11）。<br>• 各資料集的拆解（Fig. 12） | 〔原文〕 |
| 學習式協定 | 每段 context 用 50 題訓練、50 題測試。線上時，當測試品質比剖析品質低 X% 而且 GPU 有空閒，就重新剖析；Azure 實驗的門檻是 0.3 | p.6–8；p.10 〔原文〕 |
| 重複與統計 | 未說明 | 〔原文〕 |
| 程式碼／資料 | 無連結 | 〔原文〕 |
| 設計理由（原文） | • 不同 context 對壓縮的敏感度差很多（CV 0.078–0.394）；最佳壓縮法與壓縮率無法由長度或任務類型決定（p.4 §3）。<br>• 以 context 為單位做決策，因為這樣才看得到整段 context 的重要 token 分布（p.6） | 〔原文〕 |
| 設計理由〔判讀〕 | 品質分數是和未壓縮答案的相似度，所以量到的是「偏離程度」，不是任務正確率 | 〔判讀〕 |
| 原文沒講清楚的地方 | 1. 存取頻率分布與到達分布。<br>2. §3 寫「從 12 個資料集各抽 50 段」，但 Table 1 只列 6 個資料集（p.3–4），前後不一。<br>3. 結論段把系統叫 "CacheServe"，Fig. 1 也有這個名字（p.1、p.12），疑似舊名殘留。<br>4. Fig. 10 說用 "Azure inference trace [46]"，但 [46] 是 BurstGPT（p.10、p.15），引用對不上。<br>5. 長度的單位 | 〔原文〕 |
| 與既有整理不一致 | • sota §2.7、PAPERS_BY_LEVEL、intro 表的「同品質下 TTFT 最多快 2.19 倍」：照抄摘要。正文裡，2.19× 是「對全 prefill、品質下降 <3%」的上限；真正「同品質」的比較是對壓縮＋LRU 的 1.43–3.77×，以及對 IMPRESS 的 1.5–5.2×（p.9）。〔複核：✅ 指控成立。摘要寫「up to 2.19× faster TTFT at equivalent generation quality」（p.1），但 §6 要點與 §6.2 都寫 2.19× 是對全 prefill（要點寫 full prefill or full eviction）且「within 3% of quality drop」（p.7、p.9）。使用者文件是照摘要抄，錯在原文摘要與正文的條件不同；建議改成「品質下降 3% 內，比全 prefill 最多快 2.19 倍」〕<br>• sota「EvicPress 承認 DRAM 夠大或各層速度相近時優勢消失」：原文用的是 "diminishes"／"will decrease"（p.12），是變小，不是消失；建議改成「變小」。〔複核：✅〕<br>• sota「兩篇都沒算上界」：已核對一致。<br>• `workloads_eval.md` §3.5：12 個資料集、555 段、GPT-5 100 題、單位判讀為字元、未公開——已核對一致 | 見格內 |
| 對本研究的意義〔判讀〕 | 它的「每段 context 的品質敏感度」概念可以借來做「每個位置段的精度敏感度」。但它的 TTFT 倍數有一部分來自有損壓縮，不能直接與無損的放置策略比 | 〔判讀〕 |

---

### KVCache in the wild：KVCache Cache in the Wild: Characterizing and Optimizing KVCache Cache at a Large Cloud Provider（USENIX ATC 2025；arXiv 2506.02634）

- **讀了什麼**：〔全文〕arXiv **v5**（2026-02-14），https://arxiv.org/pdf/2506.02634v5 ；〔文件〕trace README（`5f7439c`）；〔程式碼〕vLLM PR #22236 的狀態。查證 2026-10-06。**未讀 ATC'25 會議版**；v5 晚於會議版，可能有修正（致謝提到「修正了 token hash 的說明」，p.13）。
- **一句話**：刻畫阿里雲百煉（Tongyi）兩種生產負載的 KV 重用特性，並提出 workload-aware 逐出。
- **評測要證明的主張**：
  * 重用偏斜；to-B 負載中 97% 的命中來自單輪請求。
  * 每類請求的重用時間可以預測（指數分布）；KV 壽命短，所需容量不大（p.2）。
  * workload-aware 策略的命中率比 LRU／LFU／FIFO／S3-FIFO 高，QTTFT 更低（p.12）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | 評估用 Qwen2-7B 與 Llama2-13B（各 1 GPU），以及 Llama3-70B（4 GPU TP）。刻畫部分另用 MHA 模型做對照。Qwen2-7B 每 16 token 的 KV 是 0.875 MB | p.9；p.12 〔原文〕 |
| 硬體 | 8× A800-80GB；NVLink 400 GBps 雙向；GPU 與主機之間 PCIe Gen4，最多 32 GBps 雙向 | p.11–12 §4.3 〔原文〕 |
| 軟體與版本 | 在 vLLM 上自己實作 CPU–GPU 兩層 KV 快取（版本沒寫），原文說已校準到與 CachedAttention 在合成負載上效能相近。另實作 Mooncake 式的全域排程器；WA 策略只作用在單一實例 | p.12 〔原文〕 |
| 資料／負載 | • 採集：Aliyun Tongyi 一個叢集兩週的資料（2025-02、2024-12），論文只呈現每條 trace 有代表性的一天。<br>• Trace A（to-C）：Text 78%、File 4%、Multimodal 3%、Search 15%；多輪比例分別 47／51／10／49%。<br>• Trace B（to-B）：全是 API，多輪比例 0.08%。<br>• hash：**論文 v5 寫的是每 4 個連續 token 一個 SipHash**；**公開 README 寫的是 16-token block**（SipHash-2-4）。<br>• 評估時用 TraceSplitter 式的方法把 trace 縮到測試平台處理得了的量，保留時間型態。<br>• 每個匿名 block 隨機生成 token，確保 hash 彼此不同；輸出長度限制成 trace 的值，並把模型輸出換成構造好的 token，以保留多輪的命中型態 | p.3–4 §3.1、Fig. 5；p.12；README L69–81 〔原文〕〔文件〕 |
| 長度 | Trace A、Qwen2-7B 的平均總長（input＋output）：單輪 973、多輪 5,953 token。有超過 12,000 token 的長請求 | p.9 〔原文〕 |
| 到達與併發 | 真實時間戳，經過縮放。原文說 API 請求常常超過 10 QPS（p.4） | p.4；p.12 〔原文〕 |
| 重用結構 | 單輪請求共享系統提示（to-B），以及多輪對話（to-C）。跨使用者的命中極少（Fig. 6） | p.4–5 〔原文〕 |
| 掃描的自變數 | CPU 快取容量，表示為「CPU 快取 ÷ HBM」的倍數（Fig. 25–27）；模型；trace | p.12 〔原文〕 |
| 對手 | LRU、FIFO、LFU、S3-FIFO；消融時加 GDFS。參數沒有說明 | p.12–13 〔原文〕 |
| 系統指標 | hit rate，以及 **QTTFT**＝TTFT＋等待 GPU 處理的時間 | p.13 〔原文〕 |
| 模擬器／真機與驗證 | 刻畫部分（理想命中率、容量需求）是 trace 分析；策略比較在真引擎上做 | p.4–9；p.12 〔原文〕 |
| 命中率定義與換算 | **block 級**：一條 trace 中需要計算的 N 個 KV block 裡，能重用先前請求已快取 KV 的比例（p.4）。換到延遲：直接量 QTTFT | p.4 〔原文〕 |
| 上界／Oracle | 沒有 Belady。只有：<br>• 無限容量下的理想命中率：A 62%、B 54%（p.4；Fig. 4 標 62.4%／54.2%）；<br>• 「永不逐出未來會用到的 KV」的理想策略所需的容量：Trace A、Llama3-70B 約 4× HBM；Trace B 比預留給 KV 的 HBM 還小（p.9 Fig. 22）。〔複核修正：原寫「to-B 約 2× 每 GPU 的 HBM」並歸在理想策略下，不對。2× 出自 p.2 的要點，條件是「用 LRU 這類標準策略、2× 每 GPU HBM 的容量就能接近無限容量的理想命中率」，不是理想策略所需容量〕 | 〔原文〕 |
| 品質指標 | 不適用 | — |
| 主要結果 | • WA 的命中率比其他對手高 8.1–23.9%，比最好的對手高 1.5–3.9%；QTTFT 降 28.3–41.9%（p.12–13）。<br>• 前言卻寫「up to 41.4% mean response time」（p.2），與 41.9% 不一致。<br>• 刻畫：<br>　◦ 10% 的 block 貢獻 77% 的重用（p.2）<br>　◦ Trace A 有 80% 的重用時間在 10 分鐘內，B 在 10 秒內（p.6）<br>　◦ 90% 的 block 在 A 經過 612 s、在 B 經過 0.3 s 之後就不再被重用（p.8） | 〔原文〕 |
| 消融／敏感度／開銷 | • Fig. 28（Qwen2-7B、CPU÷HBM＝1）：分布式估計 +1%，lifespan 調節 +2.4%（p.13）。<br>• 每次逐出 79 µs，總計是 vLLM 排程開銷的 1.2%；複雜度從 O(N) 降到 O(W)，W 是負載類別數（p.11） | 〔原文〕 |
| 學習式協定 | 在背景取最近一段資料（例：9–10 點）擬合每個類別（請求類型×輪次）的指數分布，並週期更新。附錄 A.1 顯示同一時段跨 5 個工作日的分布相近，可以用前一天預測後一天。lifespan 是超參數 | p.7；p.11；p.18 〔原文〕 |
| 重複與統計 | 未說明 | 〔原文〕 |
| 程式碼／資料 | • trace 樣本公開，Apache-2.0，四條各 2 小時，16-token block（README）。<br>• 策略在 vLLM PR #22236：**已關閉、未合併** | p.2；README；GitHub PR 〔原文〕〔文件〕 |
| 設計理由（原文） | • 不用頻率：KV 壽命短，高頻的 block 也可能很快死掉（p.11）。<br>• 加 lifespan：否則長尾的重用時間分布會讓舊 block 長期維持高機率（p.11）。<br>• offset 越大優先序越低：空間局部性，前段命中多（p.7、p.11） | 〔原文〕 |
| 設計理由〔判讀〕 | 把容量正規化成 HBM 倍數，可以讓不同模型與 GPU 的結果互比；代價是讀者無法還原成「佔工作集多少」 | 〔判讀〕 |
| 原文沒講清楚的地方 | 1. hash 粒度：論文 4 token、公開檔 16 token；論文自己的分析用哪一種，〔未查證〕。<br>2. Fig. 25–27 的容量倍數點。<br>3. 縮放的倍率。<br>4. 隨機生成內容後，prompt 實際長度與 trace 長度的對應。<br>5. 〔複核補充〕採集期間前後不一：p.2 Discussion 寫「one week production traces」，§3.1（p.3）寫兩週（2025-02 與 2024-12） | 〔原文〕 |
| 與既有整理不一致 | • sota §4 表「Alibaba Bailian（ATC'25）16-token 的 SipHash block」：與公開 README 一致；但**論文 v5 p.4 寫的是每 4 個 token**，建議註明「依公開檔」。〔複核：✅ 差異成立。v5 p.4 §3.1 第 7 項：SipHash「for every four consecutive tokens」；README（commit `5f7439c`）L69、L76 寫 16 tokens per block。sota 這格講的是公開 trace，本身沒錯，只是缺註明〕<br>• `workloads_eval.md` §3.1「每條 2 小時」：只適用公開樣本；論文分析的是「一天」。<br>• `workloads_eval.md` §2.1 的命中率稽核表沒有收錄本篇；本篇是 block 級、無限容量時 62%／54%。另外 `workloads_eval.md` §2.2 用公開檔重算出 Bailian traceA 的 block 級命中為 57.9%、traceB 為 54.2%（判讀：公開的 2 小時樣本與論文的一天不同，數字不必一致）。<br>• AsymCache 的 Max-score 對手就是本篇的策略（AsymCache p.8–9 [50]）；Fancy 的 Workload-aware 也是 | 見格內 |
| 對本研究的意義〔判讀〕 | • 「平手時先丟尾端」的空間局部性論點，和我們「前段不存」的方向相反，必須在論文裡正面回應。~~它的根據是 Trace B 這類系統提示共享的負載~~〔複核修正：原文的空間局部性證據涵蓋兩條 trace（Fig. 16），也包括 Trace A 的多輪請求（Fig. 17：重用中段對多輪對話幾乎沒有好處）與單輪請求（Fig. 18），p.7；不只是系統提示共享。原文也承認 text／multimodal 有明顯空間局部性、file／search 幾乎沒有〕；長 context agent 負載可能不同。<br>• Bailian 的公開 trace 長度太短（`workloads_eval.md` §3.1：≥32K 只佔 0–0.18%），只能當短 context 的負對照 | 〔判讀〕 |

---

### LRB：Learning Relaxed Belady for Content Distribution Network Caching（NSDI 2020）

- **讀了什麼**：〔全文〕NSDI'20 會議版，https://www.usenix.org/system/files/nsdi20-paper-song.pdf ；程式碼只查了 repo 的中繼資料（`9e8b442`）。查證 2026-10-06。
- **一句話**：用 GBDT 模仿「relaxed Belady」（下一次存取超過 Belady 邊界就可逐出），降低 CDN 的 byte miss ratio。
- **評測要證明的主張**：
  * 在 6 條生產 CDN trace 上，所有 trace×容量組合都勝過 14 種 SOTA 演算法，WAN 流量比 B-LRU 少 4–25%。
  * 原型的開銷可以接受（proc. p.529、p.539）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型（ML） | GBM（LightGBM）回歸，預測 log(time-to-next-request)，L2 損失。特徵：32 個 delta、10 個 EDC，加上物件大小與類型等靜態特徵 | proc. p.533–535 §4.3（PDF p.6–8）〔原文〕 |
| 硬體 | 原型用三台 Google Cloud n1-standard-64 VM（64 vCPU、240 GB DRAM），分別當 client、快取伺服器、origin；8 顆 375 GB NVMe 做軟體 RAID；快取 1 TB flash。人工加入網路延遲：client–proxy 約 10 ms，origin–proxy 約 100 ms | proc. p.537 §6.1（PDF p.10）〔原文〕 |
| 軟體與版本 | Apache Traffic Server 8.0.3；LRB 是約 1,400 行的 C++ 函式庫，模擬器與原型**共用**。模擬器以 AdaptSize 的模擬器為基礎，連同 14 種演算法約 11K 行 C++ | proc. p.535–536 §5（PDF p.8–9）〔原文〕 |
| 資料／負載 | 6 條生產 trace，來自 3 家 CDN（Table 4）：<br>• Wikipedia：14 天、28 億請求、3,700 萬物件、總量 90 TB、unique 6 TB、暖機 24 億請求、平均物件 33 KB<br>• CDN-A1、A2：分別 8 天／4.53 億請求、5 天／4.10 億請求<br>• CDN-B1–B3：各 9 天，請求數 18–23 億<br>每條 trace 已按單顆 SSD（約 1–2 TB）分片 | proc. p.530、p.537 Table 4（PDF p.3、p.10）〔原文〕 |
| 長度 | 不適用；物件大小平均 33–644 KB，最大約 1–1.6 GB | Table 4 〔原文〕 |
| 到達與併發 | • 模擬：照時間戳順序。<br>• 原型的吞吐實驗：1024 個 client 執行緒，closed loop。<br>• 延遲實驗：照原始時間戳，open loop | proc. p.537（PDF p.10）〔原文〕 |
| 重用結構 | CDN 物件；約 75% 的物件在兩天內沒有第二次請求（one-hit-wonders，引用前人工作） | proc. p.530 〔原文〕 |
| 掃描的自變數 | 快取大小 64 GB–4 TB，對數尺度，共 33 個 trace×容量組合；另外掃特徵數、模型種類、訓練集大小、候選數（以 good decision ratio 為準） | proc. p.534–535、p.539 〔原文〕 |
| 對手 | 14 種：<br>• 經典：LRUK、LFUDA、S4LRU、LRU、FIFO、Hyperbolic、GDSF、GDWheel<br>• 學習或自適應：Adaptive-TinyLFU（直接用原作者 caffeine 的實作）、LeCaR、UCB、LFO、LHD、AdaptSize<br>另有基準 B-LRU（Bloom filter 准入＋LRU）。**TinyLFU、LFO、LHD、AdaptSize 的參數已核對與原作者一致**。原型的對手是未修改的 ATS（近似 FIFO） | proc. p.536–537 §5.2、§6.1（PDF p.9–10）〔原文〕 |
| 系統指標 | byte miss ratio、相對 B-LRU 的 WAN 流量減少；原型另量吞吐、peak CPU、peak 記憶體、P90／P99 延遲、object misses | proc. p.537–539 〔原文〕 |
| 模擬器／真機與驗證 | 兩者都有，**共用程式碼**；Wikipedia 1 TB 在模擬與原型都跑。原文沒有報模擬與原型的 byte miss 數字差 | proc. p.535–537 〔原文〕 |
| 命中率定義與換算 | • **byte miss ratio**：使用者請求的位元組中沒被快取服務的比例，對應 WAN 成本（proc. p.529）。<br>• **object misses**：每個請求等權，原型用它來解釋 P90 延遲的改善（proc. p.538）。<br>• 原型上 byte miss 換算成 WAN 頻寬：平均 −44%，P95 頻寬 −43%（Wikipedia 第 12–14 天） | proc. p.529、p.537–538 〔原文〕 |
| 上界／Oracle | Belady MIN 與 relaxed Belady：<br>• Belady 與所有線上策略的差距是 25–40%（1 TB，proc. p.530 Fig. 2）。<br>• relaxed Belady 比 MIN 多 9–13% 的 miss（proc. p.532 Table 2）。<br>• LRB 把 SOTA 到 Belady 的差距縮小約 1/4，離 relaxed Belady 還有 1/3 到 1/2 的距離（proc. p.539 §6.5）。<br>**物件大小不一時 Belady 怎麼算，原文沒有說明** | 〔原文〕 |
| 品質指標 | 不適用 | — |
| 主要結果 | • 33 個組合全部最低 byte miss；WAN 流量平均比 B-LRU 少超過 13%（proc. p.539）。<br>• 原型：P90 延遲 110→72 ms；peak CPU 9%→16%；記憶體 39→36 GB；metadata 不到快取容量的 3%（proc. p.537–538 Table 5–6）。<br>• good decision ratio 74–86%（proc. p.539） | 〔原文〕 |
| 消融／敏感度／開銷 | • 特徵累加、模型種類（GBM、LogReg、LinReg、SVM、NN）、預測目標、8 種損失函數、訓練集大小（最多 512K）、候選數（proc. p.534–535）。<br>• LRB-OPT：用全 trace 選記憶窗，大快取時可再多省 1–4%（proc. p.539）。<br>• 開銷：訓練 300 ms；64 個候選預測 30 µs（proc. p.534） | 〔原文〕 |
| 學習式協定 | • **前 20% trace 是驗證段**，用來調滑動記憶窗；小快取直接找最佳值，大快取用最小平方迴歸外插。<br>• **暖機期一律比驗證段長、不記指標**（Wikipedia 的暖機是 28 億請求中的 24 億）。<br>• 開始時用 LRU 代替，直到訓練資料足夠。<br>• 每累積 **128K** 筆標記樣本就訓練新模型取代舊的。<br>• 標記：等到再次被請求，或超過 Belady 邊界時標記。<br>• **所有演算法的 metadata 都從快取容量扣除**（Adaptive-TinyLFU 例外） | proc. p.533、p.535、p.537 〔原文〕 |
| 重複與統計 | 未說明（模擬是確定性的；原型只有單次的時間序列圖） | 〔原文〕（未見） |
| 程式碼／資料 | 公開（BSD-2-Clause），含模擬器、原型與 Wikipedia trace | proc. p.530 腳註 1；GitHub 〔原文〕〔文件〕 |
| 設計理由（原文） | • 用 good decision ratio 做設計探索，因為 byte miss 要完整模擬好幾小時（proc. p.529–532）。<br>• 隨機取 64 個候選，因為效益已遞減（proc. p.535）。<br>• 隨機取樣「物件」而不是「請求」，是為了避免訓練資料偏向熱門物件（proc. p.533）。<br>• CDN 伺服器沒有 GPU，平均有約 90% 的 CPU 閒置（proc. p.531 Table 1） | 〔原文〕 |
| 設計理由〔判讀〕 | 「metadata 從容量扣除」是公平比較學習式與啟發式時不可省的規則；LLM 這組都沒做（KV 的 metadata 相對小，但學習式模型的特徵存放仍應計入） | 〔判讀〕 |
| 原文沒講清楚的地方 | 變長物件的 Belady 算法；14 種演算法中其餘 10 種的參數；模擬與原型的數字差距 | 〔原文〕 |
| 與既有整理不一致 | • intro 表 9「LRB 每累積 128K 筆標記樣本就訓練新模型取代舊的」：已核對一致（proc. p.535）。<br>• sota S1「LRB……保底：無」：**不準確**。LRB 在訓練資料不足時以 LRU 代替（proc. p.533 §4.1），和同表 Marconi 的「有：剛啟動時先用 LRU」是同一種機制，標準不一。〔複核：✅ 指控成立。proc. p.533 原文：實作在訓練資料足夠前用 LRU 當 fallback；Marconi 也是啟動時 α＝0 即 LRU（Marconi p.7）。兩者應同標「有（僅啟動期）」或同標「無（無預測失準時的保底）」〕<br>• intro 表 9「預測目標：下次被存取的時間；執行位置 CPU；重訓原因：負載漂移」：已核對一致（proc. p.530–535）。<br>• sota S1「原文說負載會變，所以必須定期重訓」：已核對一致（〔複核修正：頁碼由 proc. p.533 改為 proc. p.532 §4 開頭「As workloads vary over time the model must be periodically retrained」；負載快速變動的論述另見 proc. p.530 §2.1〕） | 見格內 |
| 對本研究的意義〔判讀〕 | • 可沿用的評測規矩：驗證段與暖機段分開、metadata 從容量扣除、Belady 與 relaxed Belady 雙上界、good decision ratio（可以變成「放置決策正確率」）。<br>• 不可比：CDN 物件大小不一，KV block 等大，成本卻隨位置變 | 〔判讀〕 |

---

### HALP：Heuristic Aided Learned Preference Eviction Policy for YouTube Content Delivery Network（NSDI 2023）

- **讀了什麼**：〔全文〕NSDI'23 會議版，含附錄 A–B。https://www.usenix.org/system/files/nsdi23-song-zhenyu.pdf ，查證 2026-10-06。
- **一句話**：用 LRU 取 4 個候選，再用單隱藏層 NN 做兩兩比較，逐出 YouTube CDN 的 DRAM 物件。
- **評測要證明的主張**：
  * 在生產環境中，P95 byte miss 平均降 9.1%，幾乎沒有退步，CPU 只多 1.8%（proc. p.1149）。
  * 在模擬中勝過 LRU／FIFO／ARC，並以更低開銷達到與 LRB 相當的效果（proc. p.1157）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型（ML） | 一個隱藏層的 MLP（部署時 20 個神經元），做兩兩二元分類：哪一個會先被再次存取。交叉熵損失 log(1+e^Δ)。特徵：32 個存取間隔、10 個 EDC、存取次數、平均間隔、距上次存取的時間、end-of-chunk | proc. p.1153 Table 1；p.1158；p.1162 App. A 〔原文〕 |
| 硬體 | YouTube CDN 邊緣叢集的生產機器（DRAM、SSD、HDD 多層）；只針對 DRAM 層 | proc. p.1150–1151 〔原文〕 |
| 軟體與版本 | XLA 加上手寫 C++；以 Google SmartChoices 為基礎；使用 per-CPU spinlock 與 RCU | proc. p.1153 〔原文〕 |
| 資料／負載 | • 模擬：隨機選一小部分地點，每個地點取 4 條 trace，各 3 天，分別來自 2021 年的不同季度；retrain 實驗改用 6 天。<br>• 對 LRB：兩條 2020 年的 trace（已開發與新興市場），各 4 天。<br>• 公開 trace：LRB 的 Wikipedia，物件大小統一改成 32 KiB | proc. p.1155；p.1157–1158 〔原文〕 |
| 長度 | 不適用 | — |
| 到達與併發 | 生產流量；模擬照 trace 順序 | 〔原文〕 |
| 重用結構 | 影片 chunk 的循序 range request，空間局部性強（proc. p.1151） | 〔原文〕 |
| 掃描的自變數 | 隱藏層神經元數 1–256、候選數 2–16、retrain 間隔 1–10⁸；Wikipedia 上的快取大小 64–1024 GiB | proc. p.1157–1159 Fig. 10–11 〔原文〕 |
| 對手 | • 生產：先前的生產啟發式（請求率分數＋end-of-chunk 分數）。<br>• 模擬：LRU、FIFO、ARC；LRB（**特徵改成與 HALP 相同**，記憶窗用 LRB 公開實作調過，候選數比 64 與 4 兩種）；Adaptive-TinyLFU。<br>• Wikipedia：LRB、Adaptive-TinyLFU、LeCaR、B-LRU、LRU，用 LRB 的公開模擬器與論文參數 | proc. p.1151；p.1157–1158 〔原文〕 |
| 系統指標 | • **P95 byte miss ratio**：尖峰時段，原文說這段最影響 QoE（proc. p.1151）。<br>• 磁碟 first-byte latency、join latency、每請求的 P95 CPU。<br>• 討論 object miss ratio 與 byte miss 的衝突（proc. p.1156） | 〔原文〕 |
| 模擬器／真機與驗證 | 兩者都用。原文說模擬只是生產行為的不完美代理，所以用生產實驗量效果與開銷，用模擬比演算法與超參數（proc. p.1150、p.1155） | 〔原文〕 |
| 命中率定義與換算 | byte miss ratio：使用者請求的位元組中沒有命中的比例（proc. p.1149）；object miss ratio：沒有命中的請求比例（proc. p.1156）。原文說，變長物件可以看成「逐位元組逐出」（proc. p.1151） | 〔原文〕 |
| 上界／Oracle | 無 | 〔原文〕（未見） |
| 品質指標 | 不適用 | — |
| 主要結果 | • 生產：P95 byte miss 平均 −9.1%，最多 −24%，退步可以忽略；若不扣噪聲，會看到 1.5% 的 rack 變差（proc. p.1155–1156 Fig. 6）。<br>• 磁碟 first-byte 延遲平均 −3.8%（範圍 −13% 到 +5%）；join latency −1.22%（proc. p.1156）。<br>• 模擬：92.6% 的 trace 嚴格最佳，7% 打平，0.4% 較差（proc. p.1157）。<br>• P95 BMR：HALP 0.475／0.564，LRB 0.479／0.565，LRB 只用 4 個候選時 0.489／0.575，Adaptive-TinyLFU 0.515／0.598。每次逐出的預測：HALP 2.1 µs，LRB 60 µs（proc. p.1157） | 〔原文〕 |
| 消融／敏感度／開銷 | • 神經元數超過 8 就沒有幫助；候選數從 2 增到 4，BMR 從 60.4% 降到 59.3%；retrain 間隔從 1 到 10⁸ 只差不到 0.2%（proc. p.1158–1159）。<br>• CPU +1.8%，變異小（proc. p.1156）。<br>• 每次兩兩預測 720 ns（proc. p.1153）。<br>• 原文估計 LRB 在他們的系統上約多 19.2% CPU（proc. p.1150） | 〔原文〕 |
| 學習式協定 | • 線上訓練，權重從隨機初始化開始。<br>• 每次兩兩比較都存一份特徵快照，等其中一個物件先被存取時才補上標籤。<br>• replay buffer 累積 1024 筆就做一個 mini-batch 更新。<br>• ghost cache 保存已逐出物件的特徵，容量是快取物件數的倍數。<br>• 模擬：3 天 trace 的第 1 天暖機，量後 2 天；retrain 實驗先用 3 天訓練到損失穩定，再量後 3 天。<br>• 生產：確認訓練穩定後取一天的資料；保留 1% 的機器跑舊演算法，作為告警基準 | proc. p.1152–1155 〔原文〕 |
| 重複與統計 | • **本組唯一有正式噪聲模型的**：每個 rack 隨機分成實驗機（HALP）、no-op 機（舊演算法，用來量噪聲）、對照機。<br>• 噪聲用 MLE 擬合零均值的 t 分布；impact 從 beta、非中心 t、skewed normal 中挑最佳，用離散格點去卷積。<br>• 原文說，這只有在有 200 多個國家地區的大量樣本時才可行。<br>• 附錄 B 用 bootstrap 95% CI 分析 reranking 的理論效益 | proc. p.1153–1154 §4；p.1163 App. B 〔原文〕 |
| 程式碼／資料 | 程式碼未公開；兩條 trace 要簽資料共享協議才能取得 | proc. p.1150 腳註 1 〔原文〕 |
| 設計理由（原文） | • 學習式部署的三個阻礙：計算開銷、退步風險、生產噪聲。<br>• 啟發式先篩候選，可以降開銷，也提供決策品質的下限（proc. p.1150、p.1152）。<br>• 優化 P95，因為尖峰最影響 QoE；不直接優化 QoE，因為它當回饋太吵（proc. p.1151） | 〔原文〕 |
| 設計理由〔判讀〕 | 用 A/A 測試量噪聲再去卷積，是量測「策略在異質環境中的效果分布」的嚴謹做法；LLM 這組的真引擎實驗大多只跑單次 | 〔判讀〕 |
| 原文沒講清楚的地方 | 學習率與最佳化器；ghost cache 是快取物件數的幾倍；「一小部分地點」的實際數量 | 〔原文〕 |
| 與既有整理不一致 | • intro §5.5「HALP 先用 LRU 篩出候選，再由學習模型排序」：已核對一致（proc. p.1152）。<br>• sota S1「HALP……訓練／更新：—」：**漏填**，應為「線上訓練，每 1024 筆標記資料更新一次，從隨機權重開始」（proc. p.1152–1153）。〔複核：✅ 原文 proc. p.1152：從隨機初始化權重開始；replay buffer 累積 1024 筆就做一次 mini-batch 更新。註：sota 表 S1 沒有說明「—」的意思；若比照 intro 表 7 的圖例（「原文未涉及或未查證」），這是「可補」而非「寫錯」〕<br>• sota S1「保底：有：先用 LRU 篩候選」：與原文一致〔複核：✅〕；原文說啟發式提供決策品質的下限（proc. p.1152），另有 1% 機器的保留組告警（proc. p.1155）。<br>• sota S1「預測什麼：逐出偏好」：已核對一致（兩兩比較誰先被再次存取） | 見格內 |
| 對本研究的意義〔判讀〕 | • 「啟發式取候選、學習模型重排」與我們「規則保底、預測給建議」的設計直接對應；附錄 B 的 reranking 分析可以當理論引用。<br>• A/B/A 噪聲模型可以借來處理共用機器上的量測污染（CLAUDE.md 的 GpuWatcher 問題）：以 no-op 組估計噪聲分布 | 〔判讀〕 |

---

## 本組對 PoC 設計的建議〔判讀〕

1. **兩段式評測**：
   * 主結果用 trace 驅動模擬器，跑大量容量與策略組合；
   * 在真引擎上至少驗證 2–3 個容量點，**報出模擬與真機的命中率與 TTFT 誤差**，這是 Fancy 缺的；
   * 模擬器與引擎的策略最好共用同一份程式碼，比照 LRB。
2. **容量軸**：對數尺度至少 15–20 點。同時報三種單位：絕對 GiB、HBM 倍數（比照 in-the-wild）、佔 trace unique KV 的百分比，讓讀者能和三派文獻對照。主結果再挑 3–4 個代表點。
3. **至少三個指標並列**：
   * block 級命中率（等權）；
   * 成本加權的節省率：仿 Fancy 的 compute-savings ratio，權重換成我們實測的「位置×層級」成本 κ；
   * 各層命中分布：GPU／CPU／SSD／重算。

   命中語意要寫明：洞之後算不算命中。真引擎再量 TTFT 的平均、中位數與 P99（Fancy 的例子顯示平均變好、中位數可能變差）。
4. **上界**：
   * hit-optimal 的 Belady；
   * BeladyCompute；
   * 小規模用 ILP 求成本最佳。

   把「Belady 與 BeladyCompute 的差距」當作第一個診斷，**先用它回答位置感知值不值得做**。多層、異質動作成本的離線最佳需要新的 ILP 或 min-cost flow 形式；這是可以寫成貢獻的地方。
5. **對手清單**：
   * LRU、S3-FIFO、ARC、W-TinyLFU（libCacheSim 都有）；
   * Workload-aware（in-the-wild，注意它平手時先丟尾端，方向與我們相反）；
   * AsymCache 式的 f·ΔT；
   * Fancy 的 PartialNode RandomCompute；
   * 學習式可選 LeCaR 或 LRB（libCacheSim 或 LRB 公開模擬器）。

   參數全部寫明，並比照 LRB 與原作者設定核對；metadata 從容量扣除。
6. **學習或擬合元件的切分**：
   * 依時間切：前段 trace 用來調參或擬合，比照 LRB 的 20%；
   * 暖機期不記指標；
   * 跨日或跨時段驗證，比照 in-the-wild 的 5 個工作日、HALP 的 4 個季度。
7. **負載**：
   * Fancy 的兩條 trace 若釋出就優先用；
   * 目前可用的是 AgentX（session 內 hash，跨 session 共享為 0）與 Bailian（16-token，偏短，當負對照）；
   * 依 CLAUDE.md 規則 6，載入時要驗算 block 單位，例如 `ceil(input/16)==len(hash_ids)`。
8. **統計**：真引擎上每點至少重複 3 次並報誤差棒。共用機器上，借 HALP 的 A/A 思路：同時跑一組 no-op（同策略），估計環境噪聲。
9. **必須正面處理的反證**：
   * Fancy 指出 SSD 多半不必要，且 LRU 在大容量池會收斂到上界；
   * in-the-wild 指出前段的空間局部性較好。

   PoC 要顯示：在 16K–512K、長間隔的負載下，這三點分別在什麼條件下不成立。

## 未查證清單

1. Fancy-eviction：
   * trace 與模擬器的公開 repo（查證日未找到）；
   * 13 個對手的參數；
   * 真引擎實驗的 vLLM 版本與支援「洞」的修改；
   * compute-savings ratio 的分母，以及成本模型是 FLOPs 還是實測；
   * 模擬器與 vLLM 的命中率誤差。
2. AsymCache：artifact 或程式碼的 URL；request 與 block 命中率的定義；〔複核：「摘要 2.03× 的來源」已查到（Fig. 11，p.9），從清單移除〕「300 requests」的單位；KV dtype 與 vLLM 版本。
3. Marconi：論文 Fig. 11（60–140 GB）的實際設定，程式碼主掃描是 40–100 GB；`radix_cache_hybrid.py` L526 疑似筆誤對結果的影響。
4. AdaptCache：1,100 段 context 怎麼抽；評測 query 的來源；請求率數值；可選的壓縮法；引擎與版本。
5. EvicPress：存取頻率與到達分布；Table 1 與 Table 2 的長度單位（判讀為字元）；IMPRESS 是否重做；Azure trace 引用 [46] 對不上。
6. KVCache in the wild：論文 hash 粒度（4 token）與公開檔（16 token）的差異；ATC'25 會議版與 arXiv v5 的差異（未讀會議版）；Fig. 25–27 的容量點與縮放倍率。
7. LRB：變長物件時 Belady MIN 的算法；其餘 10 種對手的參數；只看了 repo 中繼資料，沒讀程式碼。
8. HALP：學習率與最佳化器；ghost cache 的倍數；模擬用了幾個地點。

---

## 複核紀錄

- **複核者**：V05（未參與抽取；只讀卡片與原文檔，沒有讀抽取者的推理或筆記）
- **日期**：2026-10-07
- **用到的原文**：抽取者下載到 `scratchpad/E05/` 的原始 PDF 與其文字檔（先核對過標題與版本：Fancy v2 2609.28870v2、v1；AsymCache 2606.02964v1；Marconi 2411.19379v3（MLSys'25 頁腳）；AdaptCache 2509.00105v2 與 BigMem PDF；EvicPress 2512.14946v1；in-the-wild 2506.02634v5；LRB、HALP 的 USENIX 會議版；Bailian README；LongBench task.md；Marconi 程式碼片段）。圖上數字用 `pdftoppm` 轉圖後直接看（Fancy p.5 Fig. 2；AsymCache p.9–10 Fig. 11–12；in-the-wild p.4 Fig. 4–5），轉檔放在 `scratchpad/V05/`。repo 狀態用 `gh`／`git ls-remote` 在 2026-10-07 查：Marconi HEAD `0801661`（2025-03-05）、LRB HEAD `9e8b442`（2023-01-20，BSD-2-Clause）、Bailian HEAD `5f7439c`（2026-04-23）、LongBench `2e00731`（2025-01-15）、vLLM PR #22236 為 CLOSED、未合併（2026-02-09 關閉）；用 `gh search repos` 找 Fancy 的 repo，沒有結果。

### 各卡檢查格數與判定

格數＝卡頭三項（讀了什麼／一句話／主張）＋表格每一列。

| 區塊 | 檢查格數 | ✅ | ❌ | ⚠️ |
|:--|--:|--:|--:|--:|
| 來源清單 | 9 | 8 | 1 | 0 |
| 本組的共同模式 | 6 | 6（第 4 點加〔複核補充〕） | 0 | 0 |
| Fancy-eviction | 26 | 22 | 1 | 3 |
| AsymCache | 26 | 23 | 2 | 1 |
| Marconi | 27 | 26 | 1 | 0 |
| AdaptCache | 27 | 26 | 0 | 1 |
| EvicPress | 27 | 27 | 0 | 0 |
| KVCache in the wild | 27 | 25 | 2 | 0 |
| LRB | 27 | 26 | 0 | 1 |
| HALP | 27 | 27 | 0 | 0 |
| **合計** | **229** | **216** | **7** | **6** |

（❌＝原文明確不支持、已改；⚠️＝出處頁碼、範圍不精確，或指控說得太重，已補註。一格裡有多條時，以最嚴重的一條計。）

### 逐條修改（原內容 → 新內容＋出處）

1. **來源清單／in-the-wild**（❌）：「v5，19 頁」→「18 頁」。`pdfinfo` 顯示 18 頁；文字檔多一個空的第 19 頁標記。
2. **Fancy／模擬器與真機**（⚠️）：「真引擎只驗證 LRU 對 PartialNode RandomCompute」→ 補上 Fig. 13 也有 block 級 RandomCompute（p.11）。
3. **Fancy／主要結果 (4)**（⚠️）：「1 TiB 時每條 trace 上 LRU 都在最佳線上演算法 0.4 點內」→ 註明這句只涵蓋 §4.6 Fig. 11 的三條 trace（Qwen To-B、Qwen Coder、AgentX），比的是線上演算法，不是 Belady（p.9）；App. C 的「0.2 點內」講的是 FreeInference 的 compute-savings（p.19）。
4. **Fancy／消融**（⚠️）：「0.17 點（p.19 App. D）」→ 0.17 點在 App. C，P99 143 洞才在 App. D（p.19）。
5. **Fancy／與既有整理不一致**（❌）：Belady 指控的**結論保留**，但原證據「1 TiB 時 LRU 離最佳線上演算法 <0.4 點」比的不是 Belady → 換成 Fig. 2（p.5，1 TiB 時 LRU 與 Belady 幾乎重合或只差約 1–2 點，讀圖）、§5「reliably converging to the optimal offline ceiling」（p.9）、App. C（p.19）。另註明原文摘要（p.1）與 §3.2（p.4）本身沒加容量限定。intro §5.2 那條改標「⚠️ 建議成立，但不算誤引」。「14 種含 LRU」補註原文 p.1 自己也這樣寫。「開頭＝節點開頭」補註 sota §2.11 已寫「節點」，漏寫的是 sota §2.11「跟本研究的關係」與 §5 表。〔複核補充〕AsymCache p.5：成本一律時退化為 LRU。
6. **AsymCache／主要結果**（❌）：「數字前後不一……§6.2 最大只到 1.91×」→ 不成立。§6.2 的 1.86／1.91／1.86× 明寫是「70B 的例子」（p.9）；1.90／1.98／2.03× 是 Fig. 11 的 8B＋LooGLE、低分散、QPS 0.04（p.9 圖上標籤）；TPOT 1.62／1.63／1.71× 是 Fig. 12 的 70B＋LongBench、高分散、QPS 0.04（p.10）。真正的不一致只剩結論把「最大值」寫成「average」（p.12）。
7. **AsymCache／與既有整理不一致**（❌）：「intro 表 1.90–2.03× 與正文不符，建議改引 §6.2」→ 劃掉，改為指控不成立（同第 6 條）。intro 表 6 的欄位定義就是「各篇報告的最佳值」。
8. **AsymCache／原文沒講清楚第 5 點**（⚠️）與**未查證清單第 2 條**：「摘要 2.03× 的來源」→ 已查到（Fig. 11，p.9），從清單移除。
9. **Marconi／與既有整理不一致**（❌）：「PAPERS_BY_LEVEL『預測重用機率』不準確；sota S1『依命中情境估重用機率』也一樣」→ 劃掉，改為指控不成立。原文摘要寫「forecasts of their reuse likelihood across a taxonomy of different hit scenarios」（p.1），§4.1 寫「to estimate the reuse likelihood」（p.5）。保留補充：實作是規則式分類，沒有算出數值機率。同格「線上調整權重應改成格點搜尋一次」改標 ⚠️ 部分成立：正文沒說幾次，「一次」只有程式碼（`radix_cache_hybrid.py` L147、L113）支持。另替 PAPERS「71.1%」「自建 7B」兩條加上出處（p.1、p.8–9、p.16 B.7）。
10. **AdaptCache／與既有整理不一致**（⚠️）：「sota『每個資料集抽 10 筆』需要更正」→ 事實保留（10 筆是估計器剖析樣本，p.2 §3；評測是 1,100 段，p.2 §4）。但 sota 這句列在「做不到」，「品質評估」可以讀成系統內的品質估計，所以改為「措辭有歧義、建議改寫」，不算事實錯誤。
11. **EvicPress／與既有整理不一致**（✅，加註）：2.19× 指控確認成立。摘要寫「at equivalent generation quality」（p.1），正文 §6 要點與 §6.2 是對全 prefill、「within 3% of quality drop」（p.7、p.9）。使用者文件是照摘要抄的。
12. **in-the-wild／上界**（❌）：「理想策略所需容量：to-B 約 2× HBM」→ 2× 出自 p.2 要點，條件是「用 LRU 等標準策略、2× 每 GPU HBM 就接近無限容量命中率」；理想策略的容量分析（p.9 Fig. 22）是 Trace A Llama3-70B 約 4×、Trace B 比預留 HBM 還小。
13. **in-the-wild／對本研究的意義〔判讀〕**（❌，判讀格裡寫成事實的部分）：「它的根據是 Trace B 這類系統提示共享的負載」→ 劃掉。原文空間局部性證據涵蓋兩條 trace（Fig. 16），包括 Trace A 多輪（Fig. 17）與單輪（Fig. 18），p.7。
14. **in-the-wild／原文沒講清楚**（〔複核補充〕）：p.2 寫「one week」，§3.1 寫兩週，前後不一。
15. **in-the-wild／與既有整理不一致**（✅，加註）：Bailian hash 粒度差異確認。v5 p.4 是「every four consecutive tokens」；README L69、L76 是 16 tokens。
16. **LRB／與既有整理不一致**（⚠️）：「負載會變所以必須定期重訓（proc. p.533 §4）」→ 頁碼改為 proc. p.532（§4 開頭），另見 p.530 §2.1。「保底：無」指控加註確認（proc. p.533 LRU fallback；Marconi p.7 α＝0 即 LRU）。
17. **HALP／與既有整理不一致**（✅，加註）：「訓練／更新：—」可補（proc. p.1152）；註明 sota 表 S1 沒有定義「—」。
18. **共同模式第 4 點**（〔複核補充〕）：in-the-wild 除了無限容量命中率，還有理想策略的所需容量分析（p.9），但兩者都不是 Belady 式上界。

### 對使用者文件（intro.txt／sota.txt）指控的逐條判定

| 指控 | 判定 | 依據 |
|:--|:--|:--|
| sota §2.11「Belady 上界在各種容量下仍有明顯空間」說過頭 | ✅ 成立（原證據已換） | Fancy Fig. 2 p.5（讀圖）、§5 p.9、App. C p.19；但原文摘要 p.1 與 p.4 本身沒加容量限定 |
| intro §5.2「連 Belady 上界都還有明顯空間」應加「在 HBM 容量下」 | ⚠️ 建議成立，不算誤引 | intro 措辭與 Fancy 摘要一致 |
| intro 表 AsymCache「1.90–2.03×」與正文不符 | ❌ 不成立 | AsymCache Fig. 11（8B＋LooGLE、低分散、QPS 0.04：1.90／1.98／2.03×），p.9 |
| EvicPress「同品質下最多快 2.19 倍」其實不是同品質 | ✅ 成立 | EvicPress p.7、p.9：對全 prefill、品質下降 <3%；摘要 p.1 寫「equivalent quality」 |
| sota「AdaptCache 每資料集 10 筆」其實是剖析樣本 | ⚠️ 事實成立，但 sota 句子可讀成估計器，屬措辭歧義 | AdaptCache p.2 §3、§4 |
| sota S1 LRB「保底：無」標準不一 | ✅ 成立 | LRB proc. p.533；Marconi p.7 |
| sota S1 HALP「訓練／更新：—」漏填 | ✅ 成立（可補） | HALP proc. p.1152 |
| sota S1／PAPERS Marconi「依命中情境估重用機率」不準確 | ❌ 不成立 | Marconi 摘要 p.1、§4.1 p.5 原文就是這樣寫 |
| intro §5.5／sota S1 Marconi「線上調整權重」應改成「調一次」 | ⚠️ 部分成立（只有程式碼支持） | Marconi p.7；`radix_cache_hybrid.py` L113、L147 |
| intro／sota「19.9%」只是平均 TTFT | ✅ 成立 | Fancy p.11 |
| 「從開頭整段丟」的開頭是節點開頭 | ✅ 成立，但 sota §2.11 已寫「節點」，只有 §2.11 末段與 §5 表漏寫 | Fancy p.11 |
| Bailian hash 粒度：論文 4 token、公開檔 16 token | ✅ 成立（sota 寫 16 是依公開檔，本身沒錯，只缺註明） | in-the-wild v5 p.4；README L69、L76 |
| 「14 種」含 LRU 本身 | ✅ 事實成立，屬小瑕疵（原文 p.1 自己也這樣說） | Fancy p.4 §3.2 |
| intro 表 AdaptCache／EvicPress 合併列標 BigMem'25＋2.19× | ✅ 成立 | AdaptCache p.1；EvicPress p.1 |
| sota「EvicPress 優勢消失」應為「變小」 | ✅ 成立 | EvicPress p.12 |
| SSD 卸載「多半沒必要」只是討論、沒做 SSD 實驗 | ✅ 成立 | Fancy p.12 §6.3 |

### 沒有檢查或只抽查的部分（如實記錄）

- 「本組對 PoC 設計的建議〔判讀〕」整節：只抽查其中引用的事實（AgentX 跨 session 共享為 0、Bailian 16-token、in-the-wild 5 個工作日、HALP 4 個季度、Fancy 的 SSD 與 LRU 收斂說法），沒有評論建議本身。
- 「未查證清單」：只更新 AsymCache 那條，其他沒有逐條再查。
- Fancy v1：沿用抽取者的做法，只比對關鍵詞次數（19.9、partial-node、BeladyCompute、AgentX、0.4 points、0.2 points，v1 與 v2 次數相同），沒有逐段 diff。
- in-the-wild：同樣沒讀 ATC'25 會議版；Fig. 25–27 的容量點沒有讀圖。
- Marconi：`utils/generate_trace.py` 的行號只確認了截斷條件（>32,768、>50 輪）、打字速度與 Poisson 間隔的程式碼，沒有逐行核對 L46–99 的範圍。
- HALP Fig. 10、Fancy Fig. 15–17 的曲線數值沒有讀圖，只核對正文。
- （已補查）EvicPress Table 2 長度 ÷ LongBench 官方平均詞數：12 個資料集的比值是 5.37–6.63（含 2WikiMQA 30K／4,887＝6.14），與卡片的「5.4–6.6」一致。
