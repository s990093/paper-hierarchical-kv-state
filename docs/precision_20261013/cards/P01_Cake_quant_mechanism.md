# P01｜Cake 的量化實驗與還原機制（精讀）

> 一句話：Cake 的量化實驗只量 TTFT。它把「8-bit」和「KVQuant 3-bit」接在雙向還原上，結果是精度越低、對只算的加速越大。全文沒有任何品質數字。也沒說重算的前段是什麼精度、反量化在哪做。

## 出處

- 作者：Shuowei Jin\*, Xueshen Liu\*, Qingzhao Zhang, Z. Morley Mao（University of Michigan）。
- 題目：Compute Or Load KV Cache? Why Not Both?
- Venue：ICML 2025，PMLR 267:28031–28043。arXiv 2410.03065。
- 讀的版本（2026-10-13 查證）：
  - **ICML 正式版**：PMLR PDF（`raw.githubusercontent.com/mlresearch/v267/main/assets/jin25d/jin25d.pdf`，13 頁，PDF 建立日 2025-06-10）。**全文 p.1–13 讀過**。
  - **arXiv v2**（2025-02-20，12 頁）：就是本機 `papers/cake2025.pdf`（第 1 頁側欄印 `arXiv:2410.03065v2`）。**全文 p.1–12 讀過**。
  - **arXiv v1**（2024-10-07，16 頁）：只讀量化相關段落 p.7–10、App. B–C p.13–14。
- 頁碼一律是 PDF 頁。兩版的印刷頁碼與 PDF 頁相同。ICML 的 PMLR 頁碼＝28030＋PDF 頁。
- OpenReview（WOyOtaO6lQ）被人機驗證擋下，沒讀到審稿意見。沒有繞過。

### 版本對照（量化相關）

| 內容 | arXiv v2 | ICML 正式版 |
|:--|:--|:--|
| 方法句（接 KVQuant） | §5.1 / p.6 | §5.1 / p.6 |
| 結果表 | **Table 6**，§5.5 / p.7 | **Table 7**，§5.6 / p.8 |
| Discussion（相容性主張） | 無 | §6 / p.9 |
| 章節總覽提到壓縮 | p.5（Section 5.5） | p.5（Section 5.6） |

ICML 在 §5.3 新增 chunk 大小實驗（Table 4），所以之後的節號、表號都加一〔原文 ICML §5.3 / p.6–7〕。量化表的數字兩版完全相同〔原文 v2 Table 6 / p.7；ICML Table 7 / p.8，逐格比對〕。

## 機制

### A. 量化實驗到底做了什麼

1. **用了什麼方法**。原文說：把「最常見的 8-bit 量化」和「KVQuant 提出的 3-bit 量化」接到 Cake 上〔原文 ICML §5.1 / p.6；v2 §5.1 / p.6〕。3-bit 明確歸給 KVQuant。8-bit 是不是 KVQuant 的實作，句子有歧義〔判讀〕。
2. **量化哪一部分**。原文只說壓縮讓 KV 變小、「更容易經 I/O 傳」，所以載入變快〔原文 ICML §5.1 / p.6；§5.6 / p.8〕。沒說只量化載入的那段。
3. **只算基線、只載基線有沒有量化**：原文沒寫。但可以從表推〔計算〕：
   - T_io／T_comp ＝（對只載的加速）÷（對只算的加速）。
   - 16→8 bit，T_io 縮 1.88–1.94×；16→3 bit，縮 4.21–5.02×（7–56 Gbps 四點）。
   - 如果只載基線仍載 BF16，這個比值應該不隨位元數變。所以**只載基線也是載量化後的 KV**〔判讀〕。
   - 只算基線不載 KV，不受量化影響。
4. **重算的前段是什麼精度**：原文沒寫。〔判讀〕前段是 vLLM 現場從 token 算出來的，預設 KV dtype 是 BF16〔原文 ICML §5.1 / p.5〕，所以前段在量化實驗裡應是全精度。
5. **何時量化、在哪反量化**：原文沒寫。I/O 是延遲模擬：依 chunk 大小與頻寬算延遲，模擬的儲存後端暫停送往 GPU 記憶體，到時才放行〔原文 ICML §5.1 / p.5〕。〔判讀〕量化實驗最可能只是把延遲公式裡的「chunk 大小」換成量化後的位元組數。
6. **反量化時間有沒有算進去**：原文沒寫。從表推〔計算〕：
   - 把 T_io／T_comp 對 1/頻寬 做直線擬合（四點，最大殘差 ≤0.011）：
     - 16-bit：斜率 51.1，常數 0.069。
     - 8-bit：斜率 26.2，常數 0.057。
     - 3-bit：斜率 9.85，常數 0.060。
   - 斜率比是 1.95 與 5.18（理想值 2 與 5.33）。換算約 8.2 與 3.1 bits／元素〔計算〕。多出來的可能是 scale 等 metadata，原文沒寫。
   - 和頻寬無關的常數，在 8-bit、3-bit 並**沒有**比 16-bit 大。
   - 〔判讀〕表上看不到反量化成本。數字和「只縮位元組」一致。但不能排除反量化很小、被四捨五入吃掉。
7. **實驗設定**：LongAlpaca-13B（MHA；Table 1 列 800 kB／token）、16K、chunk（token budget）512、2×A100（tensor parallel）、GPU 利用率 100%、頻寬 7／25／32／56 Gbps〔原文 ICML Table 7 圖說 / p.8；v2 Table 6 / p.7；TP 見 ICML §5.2 / p.6〕。KV 預設 BF16〔原文 ICML §5.1 / p.5〕。
8. **表上的數字**（格式：對只載的加速 \ 對只算的加速）〔原文 ICML Table 7 / p.8；v2 Table 6 / p.7〕：

   | 頻寬 | 16-bit | 8-bit | 3-bit |
   |:--|:--|:--|:--|
   | 7 Gbps | 8.10\1.10 | 4.63\1.22 | 2.39\1.63 |
   | 25 Gbps | 3.03\1.43 | 2.07\1.87 | 1.41\3.09 |
   | 32 Gbps | 2.63\1.59 | 1.85\2.12 | 1.37\3.73 |
   | 56 Gbps | 1.97\2.00 | 1.49\2.85 | 1.26\5.38 |

   - 16-bit 欄和 Table 3 的「2×A100、100%」列逐格相同〔原文 ICML Table 3 / p.6〕。所以 16-bit 欄不是另外跑的，是沿用主表。
   - 換算 Cake 本身的 TTFT（相對只算）〔計算〕：32 Gbps 時 16／8／3-bit 分別是 0.63／0.47／0.27。
9. **原文的結論**：低精度像 GQA 一樣縮小 KV，I/O 變快，所以對只算的加速隨精度降低而上升〔原文 ICML §5.6 / p.8〕。對只載的加速下降，原文沒有文字討論，只在表上看得到。
10. **v1 是不同的實驗**〔原文 v1 §5.1 / p.7；§5.4 / p.8–9；Fig. 6 / p.8；App. C / p.14〕：
    - 比的是「8-bit」和「CacheGen」，不是 3-bit KVQuant。
    - 14K、圖說寫 Alpaca-7B、單張 A100、GPU 90% 與 10%、頻寬 2000–10000 Mbps。
    - 原文說理論上載入時間分別縮 2× 與 8.6×；實測 I/O 加速最多 1.95× 與 7.97×。
    - Cake＋CacheGen 比 Cake＋原始 KV 的 TTFT 少 32.8–73.5%。
    - v1 也沒有品質數字。

### B. 品質

11. 三個版本全文搜尋 accuracy／quality／perplexity／F1／lossless：**量化實驗沒有任何品質或準確率數字**〔原文 v1、v2、ICML 全文〕。唯一相關的是相關研究段落轉述「量化方法降低精度、同時維持準確率」（引別人）〔原文 ICML §2 / p.2〕。
12. 「重算的前段是全精度」：**原文沒寫**。見第 4 點的判讀。

### C. Discussion 的相容性主張（只在 ICML）

13. §6 把 KV 量化與 eviction 歸為「KV 大小縮減」類。主張 Cake 把 KV 當資料，"agnostic to its specific representation or compression scheme"。並說 §5.6 的評估證明能無縫整合、再降延遲〔原文 ICML §6 / p.9〕。
14. 〔判讀〕這個主張的證據只有 TTFT。「把 KV 當資料」也代表 Cake 不會依 chunk 位置或「會被重算還是被載入」去選精度。

### D. 還原機制（和我們最相關的系統面）

15. **雙向**：收到請求→切 chunk→找儲存裡最後一個存在的 prefix hash→兩個指標。計算從 0 往後，I/O 從尾往前〔原文 ICML App. A / p.12〕。
16. **會合點是自然湧現，不是事先規劃**：
    - 計算執行緒每算完一個 chunk，檢查下一個 chunk 是否已在 CPU 記憶體；在的話，通知 I/O 停，然後結束〔原文 ICML App. A, Algorithm 1 / p.12〕。
    - I/O 執行緒在兩個指標交錯時停〔同上〕。
    - 接到 vLLM 時：每個 engine step 後用 `is_resident` 檢查下一個 chunk；已駐留就中斷 chunked prefill、開始生成〔原文 ICML App. B.2 / p.13〕。
    - **沒有估計器**。原文把「估計機制」「一方明顯較快時退回單一資源」列為未來工作〔原文 ICML §5.4 / p.7；§5.5 / p.7〕。
17. **chunk 大小**：
    - 計算端＝vLLM token budget，預設 512〔原文 ICML §5.1 / p.5〕。ICML 另掃 64–2048（1×A100、100%、32 Gbps、16K）〔原文 ICML §5.3 / p.6–7 / Table 4〕。
    - I/O 端每個 chunk 128 token，理由是經驗上平衡頻寬利用與粒度〔原文 ICML §5.1 / p.5〕。
    - 兩者怎麼對齊，原文沒寫。App. B.2 又寫「依排定的 token budget 切 chunk、把 hash 倒序推進 queue」〔原文 ICML App. B.2 / p.13〕，和 128 的說法沒交代關係。
18. **GPU 忙碌怎麼處理**：
    - 用 token budget 佔比模擬。例：budget 512，Cake 拿 256＝50%〔原文 ICML §5.1 / p.6〕。
    - Adaptive scheduling 的優先序：decode ＞ 沒有 prefix cache 的 chunked prefill ＞ 有 prefix cache 的 chunked prefill〔原文 ICML §4 Part 3 / p.5〕。
    - 只有一個例子：16K 請求＋22 個 32–448 token 的請求，完成時間 1.5→1.19 s（吞吐 +26%）〔原文 ICML §5.8 / p.8〕。
19. **頻寬變動**：計算 budget 在 0–512、頻寬在 0–25 Gbps 隨機抽，只給一張軌跡圖（A100、LongAlpaca-7B、16K）〔原文 ICML §5.7 / p.8 / Fig. 5〕。原文說會合點會跟著移動〔原文 ICML §4 / p.5〕。
20. **實作**：擴充 LMCache 並接上 vLLM，約 1,000 行。新增非同步 get（worker 從 task queue 讀 chunk key、取回後放進 resident dictionary）、chunk buffer 預先配置〔原文 ICML App. B.1 / p.12–13〕。基線用 vLLM v0.6.2、LMCache v0.1.4〔原文 ICML §5.1 / p.5〕。
21. **儲存層與限速**：主實驗**沒有真實儲存裝置**，是延遲注入〔原文 ICML §5.1 / p.5〕。v1 的寫法是讓 LMCache sleep 到期後才放進 CPU 記憶體〔原文 v1 §5.1 / p.7〕。Table 2 只是把頻寬點對到物理設定：7 Gbps＝GCP 標準 egress、25＝GCP tier-1 egress、32＝Lambda SSD 讀、56＝Samsung 980 Pro 讀、100＝RoCE〔原文 ICML Table 2 / p.6〕。
22. **重算和載入真的同時跑嗎**：設計上是兩個並行執行緒〔原文 ICML App. A / p.12〕。Fig. 5 的「compute thread」從 token 0 往上、「I/O load thread」從約 16K 往下，兩條線同時推進，約 1.25 s 在 token index 約 9K 處相遇〔原文圖讀值 ICML Fig. 5 / p.8〕。v1 Fig. 8a 是同類的圖〔原文 v1 / p.9〕。開銷：每步只檢查下一個 chunk，step time 和原 vLLM 幾乎重疊〔原文 ICML §5.9 / p.8；Fig. 7 / p.9〕。
23. **會合點落在哪**：原文沒有把 b 當指標報。只能讀到兩個點：(i) Fig. 5 的波動軌跡，約 9K／16K 由計算完成（約 58%）〔原文圖讀值 ICML Fig. 5 / p.8〕；(ii) v1 說 10000 Mbps、全 A100 時只有 75% 的 token 要算〔原文 v1 §5.2 / p.8〕。量化 run 的 b 完全沒報。
24. **下一輪**：原文只在動機說多輪對話的後續查詢會重用 KV〔原文 ICML §1 / p.1〕。評估時「事先把所有請求的 KV 算好存好」〔原文 ICML §5.1 / p.5〕。還原後新算的前段要不要寫回、下一輪怎麼辦：**原文沒寫**。只有單次還原。

### E. 程式碼

25. **沒有公開程式碼**（2026-10-13 再查一次）：
    - arXiv、PMLR、ICML poster 頁都沒有 code 連結〔文件〕。作者頁 `shuoweijin.com/publication/cake` 只連 arXiv〔文件〕。
    - 一作 GitHub 帳號 `jsw-zorro`（作者頁上的連結）：`jsw-zorro/vllm` 相對上游 ahead 0 commits〔程式碼 GitHub compare API〕。沒有 LMCache fork。
    - 二作 `xenshinu/LMCache`：本人 commit 只有 2024-09 的 `fast_serde.py` 與 2025-01 的 `local_backend.py`。`fast_serde.py` 只是把 tensor 轉 raw bytes、再用 frombuffer 讀回 BF16〔程式碼 xenshinu/LMCache@46bdb8f `lmcache/storage_backend/serde/fast_serde.py:12–29`〕。**沒有量化，也沒有雙向邏輯**。
    - `gh search` 搜「Cake KV cache compute load」「bidirectional kv cache loading」都沒結果。
    - 結論：量化 KV 怎麼載入、會合點邏輯在哪，**都無法對到程式碼**〔未查證〕。

## 原文沒寫的地方

- 8-bit 用哪個實作（KVQuant？一般 INT8？FP8？）。
- 量化的時機、粒度（per-channel／per-token）、outlier 處理。
- 反量化在 CPU 還是 GPU、花多少時間、有沒有算進 TTFT。
- 載入的 KV 進 vLLM 後是什麼 dtype。〔判讀〕vLLM v0.6.2 的 KV cache 是 BF16，3-bit 進 GPU 前必須反量化。
- 量化 run 的品質。
- 重算前段的精度。
- 量化 run 時只載基線有沒有量化（只能由比值推）。
- 每格重複幾次、有沒有誤差。
- I/O chunk 128 與計算 chunk 512 的對齊方式、會合處被部分載入的 chunk 怎麼處理。
- 每次量化 run 的會合點 b。
- 還原後寫回與下一輪。

## 和我們實驗的關係〔判讀〕

1. **「前段 FP8 沒有品質損失」在 Cake 式還原下可能是平凡結果**。
   - 前段由 GPU 從 token 重算（BF16），存著的 FP8 前段根本沒被讀。
   - 只有 I/O 端越過 25% 邊界（b ＜ 0.25N，N＝block 數）時，前段 FP8 才會被讀到。
   - 所以每個 run 都要記：會合的 block 索引 b、**實際被讀到的 FP8 block 數**、被讀到的 FP8 block 的位置。
   - 只有「被讀到的 FP8 block 數 ＞ 0」的 run，才能拿來談「前段 FP8 的品質」。
2. **要讓 b 掉進前段，必須掃 GPU 忙碌度與頻寬**。照 Cake 的定義用 token budget 佔比（12.5%／50%／87.5%／100%）。Cake 的 12.5% 就是讓 I/O 吃掉大部分的設定。
3. **FP8 會讓 b 往前移**。FP8 block 位元組減半，I/O 變快，載入的範圍變大。比較 BF16 與 FP8 配置時，要在**同一組**頻寬與 GPU 忙碌度下比，並報各自的 b。否則延遲差異和品質差異混在一起。
4. **公平比較格**（單張 3090、Llama-3.1-8B 或 Qwen2.5-7B-1M、≤32K）：
   - 還原機制固定（Cake 式：前算後載、會合點湧現），只改存的精度配置：(a) 全 BF16、(b) 全校準 FP8、(c) 25% FP8 放前／後／隨機。
   - 只載基線要用**同一份**精度配置（第 3 點的推算顯示 Cake 是這樣做）。報兩個比值（對只載＼對只算），格式照 Cake，外加 b、品質。
   - I/O 若用延遲模擬，FP8 block 的位元組數照實際算（含 scale）。也要量一次真實路徑（pinned CPU→GPU 或 SSD），因為 Cake 沒有和真實裝置校準。
   - **反量化成本 Cake 沒給**。我們要自己量，並寫明有沒有算進 TTFT（建議算）。NOT_MEASURED。
   - 粒度：Cake 是 I/O 128、計算 512。我們若兩邊都用 512-token block，是偏離 Cake 的地方，要寫明。
5. **位元組量**〔計算，依 HF config.json〕：
   - Llama-3.1-8B：32 層 × 8 KV heads × 128 dim × 2（K、V）× 2 B＝128 KiB／token（和 Cake Table 1 一致）。512-token block＝64 MiB；FP8 約 32 MiB＋scale。32K＝4 GiB（BF16）。config 取自 `unsloth/Meta-Llama-3.1-8B-Instruct` 鏡像（官方 repo 需登入）。
   - Qwen2.5-7B-Instruct-1M：28 層 × 4 KV heads × 128 × 2 × 2 B＝56 KiB／token。512-token block＝28 MiB；FP8 約 14 MiB。32K＝1.75 GiB。config 取自 `Qwen/Qwen2.5-7B-Instruct-1M`。
6. **KV 越小，FP8 能省的時間越少**。Cake 的 Llama-3.1-8B 在 32 Gbps、100% 時對只載只有 1.37×〔原文 ICML Table 6 / p.8〕。Qwen 的 KV 又比 Llama 小一半多。位置政策的延遲收益天花板可能很低。
7. **老師說的「Cake 已測過搭配量化」只成立在延遲面**。「Cake＋量化的品質」本身是空白。依老師 10/13 的停損規則，應先量 (b) 全校準 FP8 的品質；夠了就停。
8. **attention sink**：Cake 從第一個 chunk 開始算，前幾個 token 通常是 BF16。但這只在計算端至少完成第一個 chunk 時成立。GPU 很忙（budget 接近 0）時，I/O 可能一路載到最前面〔判讀〕。原文沒討論。

## 附：有沒有別人做「Cake 式雙向＋逐 chunk 選精度」

查法（2026-10-13）：Semantic Scholar 取 Cake 的 39 篇引用（API 回傳數，可能不完整），篩標題／摘要含量化、精度、壓縮的；另做 3 次 WebSearch。**沒找到同時做「前算後載會合」＋「依 chunk 位置或依會被載入／重算而選精度」的論文**。最接近的：

| 論文 | arXiv | 機制一句話 | 證據 |
|:--|:--|:--|:--|
| SparKV | 2604.21231 | 逐（chunk, 層, head）決定串流或本地算；把「Cake＋串流段量化」當 Strong Hybrid 基線；精度不隨位置變；明說位置是成本的差代理 | 既有卡 D6_SparKV（B 級）；本次未重讀 |
| KVPR | 2411.17089 | 前段從存下的激活重算、後段載入，均勻 4-bit；在 decode 階段 | 既有卡 D6_KVPR；本次未重讀 |
| HA-RAG | 2510.20878 | 依存取熱度給 KV chunk 選四種 8-bit 格式；沒有重算並行 | 本次讀摘要；既有卡 D6_HARAG |
| CacheTune | 2605.24022 | 非 prefix 重用；只重算關鍵 token，並依硬體調重算比例來平衡計算與搬移 | 只讀摘要〔未查證〕 |
| KVBoost | 2608.21362 | chunk 級重用＋偏差導向重算＋int8／int4 量化；精度是否逐 chunk 選，摘要沒說 | 只讀摘要〔未查證〕 |

〔判讀〕「量化讓 b 往前移」已被 Cake Table 7、SparKV 基線、KVPR 佔掉。「精度依位置或依離 b 多遠而變」目前沒找到前例；但只做了有限搜尋，不能當成新穎性結論。

## 與舊卡不一致之處（修正清單）

1. **D6_Cake**：「評估時搭配 KVQuant 的 8-bit 與 3-bit 量化」→ 原文句子只把 3-bit 明確歸給 KVQuant；8-bit 寫成「最常見的 8-bit 量化」，歸屬有歧義〔原文 ICML §5.1 / p.6〕。
2. **D6_Cake**：表號只寫「Table 6」並標「ICML 版 p.8 §5.6」→ ICML 版是 **Table 7**；v2 是 Table 6、§5.5、p.7。（E03 卡的對照是對的。）
3. **D6_Cake／E03**：「原文沒有說只量化載入的那段」「8-bit、3-bit 只縮位元組還是也算反量化，沒寫」→ 仍是「原文沒寫」，但補上〔計算〕證據：只載基線也是載量化 KV（T_io 隨位元數縮 1.88–1.94×、4.21–5.02×）；表上看不到反量化成本。
4. **D6_Cake**：「Cake 的 runtime 開銷……〔ICML 版 p.9，Fig. 7〕」→ 文字在 ICML §5.9 / p.8，Fig. 7 在 p.9。
5. **D6_Cake**：「Cake 一定從第一個 chunk 開始重算，所以 attention sink 永遠是 BF16」→ 要加條件：計算端至少完成第一個 chunk。§5.7 的計算 budget 抽樣範圍包含 0〔原文 ICML §5.7 / p.8〕；原文沒保證。
6. **D6_Cake**：本機 PDF 路徑 `/mlsteam/...` 在這台 Mac 上不存在。這台的 `papers/cake2025.pdf` 是 arXiv v2（2025-02-20，12 頁）；ICML 版這台沒有本機檔。
7. **E03（卡 1「程式碼／資料」）**：「xenshinu/LMCache……無法確認是否為 Cake 實作」→ 已讀 `fast_serde.py` 內容：只是 raw-byte serde，**確認不是** Cake 的雙向或量化邏輯〔程式碼〕。另補查一作 `jsw-zorro/vllm`：ahead 0 commits。
8. **E03（版本差異第 6 點）**：「v1 比較對象多了 CacheGen 壓縮」正確；補：v1 用的是 8-bit＋CacheGen（沒有 3-bit），14K、單張 A100，實測 I/O 加速最多 1.95×／7.97×〔原文 v1 §5.4 / p.8–9〕。
9. **D8_Cake_補充**：沒發現錯誤（節號用的是 ICML 版，正確）。

## 證據等級

- **A**：ICML 正式版與 arXiv v2 全文逐頁讀過；量化表逐格比對兩版；「沒有品質數字」以三個版本全文搜尋確認。
- **〔計算〕**：T_io／T_comp 的推算與擬合，只用 Table 7 的兩位小數，誤差約 ±1%。
- **程式碼**：沒有公開實作；查證過程見 E 段。會合點與量化載入的實作細節屬〔未查證〕。
