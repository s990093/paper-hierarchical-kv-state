# L3：KV Sharing × Latency Hiding × Scheduling 的古典對應機制

- **日期**：2026-10-26
- **執行者**：文獻 agent。**本檔是 self-review，不是 cross-model review**（沒有另一個模型或零脈絡審查者核對過）。
- **查了什麼**
  - 古典機制：VoD 的 batching、patching、piggybacking、stream merging；DBMS 的 circular／cooperative／shared scans、QPipe、CJOIN、Crescando、SharedDB、shared-scan 排程、「分享不一定好」；anticipatory disk scheduling；request coalescing（nginx、Varnish、Go `singleflight`、memcache leases、cache stampede）；delayed hits（只補 LLM 應用）；inference serving 的等待湊批（Clipper、Nexus、BatchMaker）；協調者補充的兩機 flow shop（Johnson's rule、flexible operation、lot streaming、加權完成時間）。
  - LLM／KV 端：用來回答「有沒有 LLM 論文用過它」與「最接近的 LLM 前作」，共下載 30 篇 LLM 論文的 PDF（其中 22 篇用完整古典詞表全文 grep，8 篇用部分詞表 grep；命中段落再細讀），另讀了 SGLang 原始碼與文件、vLLM 一則 RFC。
- **怎麼查**
  - 書目：Crossref API 逐篇核對作者、年份、venue、卷期頁、DOI。DBLP 的 API 這次回傳 bot 檢查頁，**沒有繞過**；改用 Semantic Scholar（S2）回傳的 DBLP key 當第二來源（有拿到的列在附錄 A）。
  - 引用：S2 citations API 抓每篇古典論文的**全部**引用者標題（上限 9,000 筆），用關鍵字（LLM、language model、KV、transformer、prefill、token、serving 等）過濾；另用 OpenAlex 的 `cites:` 過濾加標題＋摘要全文搜尋；再對 30 篇 LLM 論文全文 grep 古典術語；最後用 web 搜尋補洞。完整搜尋紀錄見附錄 A。
  - 下載的 PDF 放在 session 的 scratchpad（`.../scratchpad/L3/pdfs/<名稱>/`；暫存區，session 結束後可能消失，不進 git），用 `pdftotext -layout` 轉文字；頁碼一律是 **PDF 頁**。
- **證據等級**
  - 〔全文〕＝我讀了 PDF 原文的相關段落，標 §／PDF 頁／方程式或定理編號。
  - 〔部分〕＝只讀了全文中的某幾段（例如只讀摘要與一節），或只做全文 grep 再讀命中段落。
  - 〔摘要〕＝只核到摘要或出版社／索引頁的摘要。
  - 〔文件〕＝官方文件、原始碼、GitHub issue（附連結與日期／commit）。
  - 〔標題〕＝只核到書目，內容沒讀到。
  - 〔判讀〕＝本檔的推論，不是任何論文說的。
  - NOT_FOUND＝照附錄 A 的方法找過、沒找到（≠ 不存在，CLAUDE.md 規則 7）。NOT_VERIFIED＝有線索但沒核到原文。NOT_MEASURED＝需要量測、本檔沒有量。
- **本檔沒有任何新量測。** 文中的數字全部是論文原文數字（附出處）；我們情境的「128K token、約 16 GiB」是協調者給的背景，不是本檔量的。

---

## §0 結論

1. **最可借用的古典機制是 VoD 的 patching（含 stream merging），不是 DB 的 circular scan。** 原因就是 (a)。VoD 的觀眾也必須從片頭看起：Eager 等人把「整個檔案由客戶端依序消耗」列為分析前提〔全文：Eager et al. TKDE 2001 §1，PDF p.2〕。patching 的做法是：晚到者馬上加入進行中的 multicast，**先把現在播的後段存起來**，另外開一條短的 patch 串流補錯過的前段〔全文：同上 §2，PDF p.3〕。這正好對應「晚到的請求加入進行中的逐層還原，緩存目前送來的後面幾層，錯過的前幾層另外補」。DB 的 circular scan（attach、PostgreSQL `synchronize_seqscans`）靠的是「從中間開始、繞回來補前段」，**前提是消費者不在乎順序**〔全文：Zukowski et al. VLDB'07 §3，PDF p.3；文件：PostgreSQL 手冊〕；QPipe 把「必須依序」的掃描歸為 *spike overlap*：除非同時開始，否則不能分享〔全文：Harizopoulos et al. SIGMOD'05 §3.2，PDF p.4–5〕。(a) 讓 circular scan 直接失效，但 patching 本來就是為「依序消耗」設計的。
2. **第二個可借用的是「可分享工作的排程規則」**：Agrawal、Kifer、Olston 的優先權 P_i = B_i² / (λ_i · t_i^s)：排隊人數越多越該做；未來同類請求越多、可分享的掃描越長，越該延後等人〔全文：VLDB'08 §4.1，PDF p.5〕。Dan 等人在 VoD batching 中發現：選「等最久的片」（FCFS）可以勝過選「排隊最長的片」（MQL）〔摘要〕。Iyer & Druschel 的 anticipatory scheduling 則給出「該不該空等」的成本效益式與 95 百分位逾時〔全文：SOSP'01 §3.3，PDF p.5〕。三者合起來，正好回答「一份共享文件要不要延後還原、先還原哪一份、等多久」。
3. **有沒有 LLM 論文用過**：
   - VoD batching、patching、piggybacking、stream merging：**NOT_FOUND**。我掃過 Hua'98（673 篇引用）、Dan'94（707）、Dan'96（154）、Eager'01（279）、Golubchik'96（68）、Aggarwal'96（125）、Bar-Noy–Ladner（38）的全部 S2 引用者，加上 OpenAlex 查核，沒有任何 LLM／KV 論文。LLM 論文裡出現的 "piggybacking" 都是 Sarathi 標題的另一個意思（decode 搭 chunked prefill 的便車）。
   - DB 的 shared／cooperative scans、QPipe、CJOIN、SharedDB、PBM：在 KV 還原的語境下 **NOT_FOUND**。最接近的是 DB 社群寫的 agentic workflow 系統：Halo 引用 SharedDB 與 DataLoader，用在「工具呼叫的合併」而不是 KV〔部分：arXiv 2509.02121v2，PDF p.3〕；Helium 引用 Sellis 的 MQO 做 workflow 子圖消除〔部分：arXiv 2603.16104v1，PDF p.4〕。
   - Delayed hits：**只有 Strata** 引用 Atre 等人並處理 delay hit 與 bundle hit〔全文：arXiv 2508.18572v1 §3.2、§4.3.1–4.3.2，PDF p.4、p.6–7〕。Atre'20 的 67 篇 S2 引用者中，LLM／KV 相關的只有 Strata。
   - Request coalescing：有工程實作，但**沒有人用這個名字、也沒有逐層版本**：SGLang 的 in-batch prefix caching 只先排一個「領頭」請求〔文件：原始碼〕；vLLM 原生的 CPU offloading 禁止同一批 block 同時 onload，因此出現整個請求粒度的隊頭阻塞，RFC #33526 提議改成分批漸進載入〔文件〕；PEEK 讓前綴群組的「先鋒」先進場〔摘要〕。
   - Anticipatory scheduling：**NOT_FOUND**（S2 的 289 篇引用者裡沒有 LLM／KV）。Continuum 的 KV TTL（依重載成本與排隊延遲決定在 GPU 留多久）精神相同，但沒有引用它〔部分：arXiv 2511.02230v7，PDF p.1、p.5–6〕。
   - Flow shop／Johnson's rule：**在 KV 載入與計算的排程上 NOT_FOUND**。最接近的是 CacheFlow：它把一批請求的還原寫成 min max(T_comp, T_load)，在 token × layer 平面上切「樓梯形」分界，用 DP 解，但沒有提 flow shop〔部分：arXiv 2604.25080v2 §3.2，PDF p.5；Lemma 1，PDF p.15〕。唯一引用 Johnson 1954 的 LLM 論文是邊緣端 speculative decoding，用的是 draft→verify 兩段管線的完成時間遞迴〔部分：arXiv 2510.11331v1，PDF p.7〕。
4. **和 (a)–(e) 最關鍵的衝突**〔判讀〕：
   - **(e) 決定 patching 是否有意義。** 在 VoD，緩衝在每個客戶端；在我們的情境，「客戶端緩衝」是**共用的 HBM**。如果 R1 還原好的層因為 R1 自己 decode 也要用而留在 HBM，晚到的 R2 錯過的前幾層其實就在那裡，patch 是免費的，問題退化成 delayed hit（已有 Strata）。patching 只在「層用完就丟」時才有意義：文件 KV 放不進 HBM 而必須每輪重送（FlexGen 式），或多份文件互搶 HBM，或 decode 階段只依問題挑部分頁 (d)。
   - **I/O 是共用的一條管子。** VoD 的 patch 串流和主串流都從「伺服器頻寬」出，客戶端可以同時收兩倍速率。我們的 patch 和主還原走同一條 PCIe／SSD，patch 會拖慢 R1。目標函數必須算進「先來者被拖慢」，這是 VoD 沒有的耦合。
   - **(b) 給了 VoD 沒有的第三種補法：GPU 重算。** 但 (c) 讓重算錯過的前幾層很貴：要算整份文件在那幾層的 attention，成本隨位置上升。Cake 式切法（前段 token 重算、後段載入）比「重算前 k 層」便宜。
   - **(d) 讓 VoD 的整檔模型失效，但讓 QPipe 的 materialization 變便宜**：晚到者只需要少數頁時，可以只從進行中的串流挑出自己要的頁來存，不必緩存整層。
5. **最可能變成新 KV 機制的組合**〔判讀，是推論〕：「**逐層 patching＋依下一次使用時間保留**」。晚到者加入進行中的逐層還原；HBM 依「下一次被哪個請求在多久後用到」保留層，這是 PBM 的 in-order 版 cooperative scan〔全文：Świtakowski et al. PVLDB'12 §3，PDF p.6–7〕；錯過的層依門檻決定是 patch（載入或部分重算）還是等下一輪；外層用 Agrawal 式優先權加 anticipatory 逾時決定要不要延後等人。最接近的 LLM 前作是 Strata（delay hit 延後、bundle hit 同批）、vLLM RFC #33526（block 粒度漸進載入）、CachedAttention（HBM read buffer、依佇列 look-ahead 預取與淘汰）、FlexGen（column-by-column：一層權重載一次給多個 batch）。**可能失敗的條件**：每個還原時間窗內同一文件的到達數 N = λT 很小（VoD 的節省都建立在 N ≫ 1 上；我們的 N 是 NOT_MEASURED）；HBM 夠大到文件 KV 一直留著（H200 類卡上可能如此）；問題本身的 prefill 很短，晚到者早點開始算也省不了多少；PCIe 被 patch 分走後先來者變慢的量超過省下的量。

---

## §1 情境速記與兩個「regime」

**情境（協調者給定）**：一份長共享文件（例如 128K token，Llama-3.1-8B 的 KV 約 16 GiB）平時放在 CPU 記憶體或 SSD；一群使用者在不同時間來問同一份文件。還原時 KV 逐層（layer 0 → L−1）送進 GPU。

**(a)–(e)**：(a) 層的順序依賴；(b) 可以重算；(c) 成本依位置；(d) 每個請求只需要一部分；(e) HBM 能暫存多少層是限制。

**本檔用到的一個區分**〔判讀〕：
- (a) 是「**消費順序**」的限制，不是共享資料本身的依賴。文件第 10 層的 KV 和誰來問無關，晚到者**可以先收下來存著**，只是要等自己前 9 層算完才能用。這就是 VoD 觀眾「先錄下後段、晚點看」的處境，所以 VoD 比 DB scan 更像我們。
- 文件 KV 被消費兩種次數：問題的 prefill 每層讀一次；**decode 每一步都要把所有層再讀一次**。所以：
  - **Regime R（可常駐）**：HBM 放得下一份文件 KV 直到 session 結束。大家讀同一份 HBM 拷貝（prefix caching）。晚到者遇到的是 delayed hit：已到的層立刻可用，只等還沒到的層。
  - **Regime S（只能串流）**：HBM 放不下或被別的文件搶走，層用完就丟、下一輪再送（FlexGen 式 offload，或 decode 時依問題挑頁）。這時 VoD 的 batching／patching／merging 才是「原樣」的問題。
  - **Regime D（依問題挑頁）**：(d) 成立時，每個請求每層只要一部分頁，分享程度＝頁集合的重疊。這時 cooperative scans 的「依需求的人數挑下一塊」比 VoD 的整檔模型合適。

---

## §2 VoD（隨選視訊）

### 2.1 Batching：Dan, Sitaram, Shahabuddin（ACM MM 1994；Multimedia Systems 1996）

1. **定義與決策規則**
   - 輸入：每部片的等待佇列（人數、等待時間）、伺服器剩餘串流數。輸出：有空串流時要 multicast 哪部片。目標：在串流數的硬上限下，降低等待與放棄（reneging）。
   - 規則比較：FCFS（選有最老請求的那部片）對 MQL（選排隊最多的那部片）。他們發現 FCFS 可以勝過 MQL，哪個好取決於觀眾願意等多久；若能保證最大等待時間並因此改變觀眾行為，對熱門片固定間隔播放（FCFS-n）更有效；用實際片單分佈估計，batching 可以把需要的伺服器容量降低約 60%〔摘要：IBM Research 出版頁，MM'94 與 Multimedia Systems'96 兩版〕。
   - 後續：Aggarwal, Wolf, Yu 提出 MFQ（maximum factored queue length）：把佇列長度乘上一個對熱門片不利的權重，權重由最佳化求出，用來同時改善延遲、放棄率與公平性〔摘要：ICMCS 1996〕。權重的確切公式 NOT_VERIFIED（沒讀到全文）。
2. **假設**：請求可以延後開始；同一部片的所有觀眾要的是完全相同、從頭到尾的資料；伺服器串流數是硬上限；觀眾有放棄機率〔摘要〕。
3. **LLM 有沒有用**：**NOT_FOUND**。Dan'94 的 707 篇、Dan'96 的 154 篇 S2 引用者，標題關鍵字命中 0 篇；OpenAlex 對 Dan'94 的過濾只命中一篇 2014 年的社群網路論文（無關）。**精神相近、但沒有引用**的 LLM 作法：
   - Strata 的 *bundle hit*：共用同一段 context 的請求排進同一個 prefill batch，既平衡載入與計算，也省 GPU 記憶體〔全文：arXiv 2508.18572v1 §4.3.2，PDF p.6–7〕。
   - SGLang 的 cache-aware scheduling（longest-shared-prefix-first），離線情況下 DFS 順序可達最佳命中率（Theorem 3.1）〔部分：arXiv 2312.07104v2，PDF p.6〕。
   - DLPM 在「前綴局部性」與「公平」之間取捨〔摘要：arXiv 2501.14312v1〕。這和 Dan 的 MQL 對 FCFS 是同一個兩難〔判讀〕。
   - RAGCache 的 cache-aware reordering：優先權＝已快取長度 ÷ 需計算長度，並用等待窗避免餓死〔部分：arXiv 2404.12457v2 §5.2，PDF p.7〕。
4. **對應到我們〔判讀〕**
   - (a)：**相容**。batching 本來就讓一群人「同時從頭開始」，正好繞過 (a)。
   - (e)：在 Regime S 下，batch 越大，每層要給越多人算，HBM 裡的 activation 與各人自己的 KV 越多。FlexGen 的 column-by-column 排程就停在「activation 與 KV 塞滿記憶體」的地方〔全文：arXiv 2303.06865v2 §4.2，PDF p.4–5〕。
   - (d)：大家要的層一樣，但頁可能不同；batching 只省「共同的頁」。
   - **延後的代價和 VoD 不同**：VoD 的延後是純等待；我們的延後若讓 I/O 閒著，就是把可以隱藏的還原時間白白浪費。所以 batching 的等待窗應該只開在「I/O 正忙於別的工作」的時候（見 §4 anticipatory 的判讀）。
   - **可借用**：FCFS 對 MQL 的結論（選人多的文件不一定比選等最久的好）；「保證最大等待時間」當作設計約束。

### 2.2 Patching：Hua, Cai, Sheu（ACM MM 1998）及其最佳化版本

1. **定義與決策規則**
   - 原始想法：已經在播的 multicast 可以「長大」來服務新來的客戶，讓請求不必等待〔摘要：UCF STARS 收錄頁〕。
   - 機制（依 Eager 等人的轉述）：晚到的客戶收聽進行中的 multicast 並**緩存**收到的資料；伺服器另外給它一條 unicast 的 *patch* 串流，補上它錯過的前段；patch 串流播到它加入 multicast 的位置就結束。客戶端接收頻寬需要是播放速率的兩倍〔全文：Eager et al. TKDE 2001 §2，PDF p.3〕。
   - 門檻規則（grace patching／stream tapping／controlled multicast）：最近一條完整 multicast 已播出的比例超過門檻 y 時，下一個請求改開新的完整 multicast〔全文：同上 §2，PDF p.3–4〕。Gao & Towsley 對 controlled multicast 推導了最佳門檻〔摘要：ICMCS 1999〕。
   - **定量結果**〔全文：Eager et al. §2 式 (1)，PDF p.4〕：Poisson 到達、N = λT（一部片播放時間 T 內的平均到達數）時，最佳門檻 y* = (√(2N+1) − 1)/N，所需伺服器頻寬（以播放速率為單位）為 **√(2N+1) − 1**，也就是隨請求率的**平方根**成長；不用 multicast 則線性成長。
2. **假設**〔全文：Eager et al. §1–2，PDF p.2–3〕：客戶端有足夠空間緩存；整個檔案依序消耗、沒有暫停或快轉；伺服器以播放速率送（r = 1）；客戶端能同時收兩條串流；Poisson 到達。
3. **LLM 有沒有用**：**NOT_FOUND**。Hua'98 的 673 篇 S2 引用者中，關鍵字只命中一篇 2002 年的串流快取論文（標題含 "token"，與 LLM 無關）；Eager'01 的 279 篇命中 0；OpenAlex 過濾命中 0；LLM 論文全文 grep 中，"patching" 只出現在 "dispatching" 一詞裡。
   - **最接近、但沒有用這個概念的 LLM 作法**：vLLM RFC #33526 提議把 CPU→GPU onload 從「整個請求一次」改成「分批漸進」，讓共用前綴的短請求不必等長請求的所有 block〔文件：<https://github.com/vllm-project/vllm/issues/33526>，2026-02-01 開，狀態 CLOSED，是否實作 NOT_VERIFIED〕。這是 token／block 方向的「早點開始用」，不是逐層的 patch。
4. **對應到我們〔判讀〕**
   - (a)：**完全對得上**。這是本檔認為 VoD 比 DB scan 更合適的核心原因。
   - (e)：patch 期間，R2 要把 R1 正在送的後面幾層存在 HBM。若 R2 晚了 k 層，而 patch 速率等於主串流速率，R2 大約要存 k 層才追得上，所以 k 接近 L 時等於存整份文件 KV，也就回到 Regime R。可行條件約為 k ≤ min(y*·L, W)，W 是 HBM 能給的層數（推論，未驗證）。
   - **Regime R 下 patch 是免費的**：R1 還原好的前 k 層若留在 HBM（R1 decode 也要用），R2 不需要 patch。這時 patching 退化成「逐層的 delayed hit」：R2 先在已到的層上算自己的問題，再跟在 R1 後面。增益最多是 R2 在前 k 層的計算時間，前提是 I/O 才是瓶頸。
   - **共用 I/O 的耦合**：VoD 的頻寬式只算伺服器總頻寬，沒有「patch 拖慢原 multicast」這件事。我們的 patch 和主還原走同一條 PCIe／SSD，R1 會被拖慢。y* 的推導前提（r = 1、客戶端 2 倍接收）**不成立**，不能直接套用 y*，只能當作起點。
   - (b)：patch 可以改成 GPU 重算，這等於 VoD 沒有的第二個通道。但重算「前 k 層」要對整份文件在那些層算 attention，(c) 讓成本隨文件長度非線性上升；Cake 式切法（前段 token 重算、其餘載入）較便宜。Cake 的做法是從頭往後重算、從尾往前載入、在中間相遇〔部分：arXiv 2410.03065v2 摘要〕。
   - (c)：VoD 的 patch 長度與晚到時間成正比；我們的「載入型 patch」也與晚到層數 k 成正比（每層大小相同），所以線性假設對載入型 patch 成立，對重算型 patch 不成立。
   - (d)：見 §3.3 的 materialization：只存自己要的頁，(e) 的代價下降。

### 2.3 Piggybacking：Golubchik, Lui, Muntz（SIGMETRICS'95；Multimedia Systems 1996）；Aggarwal, Wolf, Yu（SIGMETRICS'96）

1. **定義與決策規則**：對同一部片、開始時間相近的串流，**微調播放速率**（後者加速、前者減速）讓兩者到達同一位置後合併成一條〔全文：Eager et al. §1 的轉述，PDF p.1〕。Golubchik 等人的期刊版摘要把目標說成：在 I/O 頻寬是關鍵資源的前提下，透過分享熱門物件的請求來降低總 I/O 與新請求延遲〔摘要（二手）：由 web 搜尋取得的摘要片段，原文 PDF 沒取得〕。Aggarwal 等人在一大類合理方法中推導最佳的 piggyback 合併策略〔摘要：SIGMETRICS'96〕。
2. **假設**：只需要客戶端接收頻寬等於播放速率、緩衝最小；觀眾可接受小幅速率變化〔全文：Eager et al. §1，PDF p.1〕。
3. **LLM 有沒有用**：**NOT_FOUND**（Golubchik'96 的 68 篇、Aggarwal'96 的 125 篇 S2 引用者，LLM 命中 0；後者命中一篇 2001 年 near-VoD 論文，無關）。
4. **對應到我們〔判讀〕**
   - 在我們的情境裡，「調速」反過來比 VoD 容易：GPU 算一層通常比 I/O 送一層快，所以晚到者在 Regime R 下天生就能「追上」先來者（已到的層全在 HBM）。
   - 在 Regime S 下，追上需要錯過的層比主串流更快到達。走同一條 PCIe 做不到，只能靠 (b) 重算，又受 (c) 限制。
   - 「讓先來者減速等人」＝延後 R1 的還原或計算，在 I/O-bound 時沒有好處。
   - **可借用的不多**：主要是「合併後只送一份」的概念，和 Regime R 下「晚到者自然追上」的觀察。

### 2.4 Stream merging、HMSM、bandwidth skimming，以及下界：Eager, Vernon, Zahorjan（TKDE 2001；MMCN 2000）；Bar-Noy & Ladner（SODA 2001／J. Algorithms 2003）

1. **定義與決策規則**〔全文：Eager et al.〕
   - HMSM（hierarchical multicast stream merging）：每條串流都 multicast；客戶端以超過播放速率的速度累積資料（同時收多條或收加速串流），追上較早的客戶；客戶被**反覆合併成越來越大的群組**；合併後一起收同一條串流的剩餘部分（§5.1，PDF p.11）。
   - **下界**（§4.1 式 (5)，PDF p.8）：要讓每個請求「立即、即時」開始，任何技術所需的伺服器頻寬至少是 **ln(N + 1)**（N = λT）。推導方式：檔案位置 x 的資料，對 t 時到達的客戶，最晚要在 t + x 送出；在 t 到 t + x 之間到達的客戶都能收同一份。在客戶端接收頻寬 2 倍播放速率、緩衝足夠時，可以接近這個下界（摘要與 §4.2）。
   - §1 的貢獻列表說 patching 的頻寬隨 √N 成長，dynamic skyscraper 隨 log N 成長（PDF p.2）；摘要說 HMSM 比 partitioned dynamic skyscraper 更省，並在很寬的請求率範圍內接近下界（PDF p.1）。
   - Bar-Noy & Ladner 對線上 stream merging 做 competitive 分析〔標題：J. Algorithms 48(1):59–90, 2003〕。二手資料（一篇後續論文與 Dagstuhl 摘要）說它是 O(log n)-competitive，但 NOT_VERIFIED（沒讀到原文）。
2. **假設**：客戶端緩衝足夠、整檔依序消耗（§1，PDF p.2）；下界在「到達成批」時會更低（§4.1，PDF p.8）。
3. **LLM 有沒有用**：**NOT_FOUND**（Eager'01 的 279 篇、Bar-Noy–Ladner 的 38 篇 S2 引用者，LLM 命中 0）。
4. **對應到我們〔判讀〕**
   - **ln(N+1) 這條下界的結構可以搬過來**：「文件第 ℓ 層 KV 對 t 時到達的請求，最晚要在 t + (前 ℓ 層的計算時間) 送到」，這和 VoD「位置 x 最晚在 t + x 送出」同構，層 ↔ 位置、每層計算時間 ↔ 播放速率。
   - 但 VoD 假設送的速率等於消耗速率（r = 1）。我們通常是 I/O 比計算慢，請求本來就做不到「立即開始」。這時下界要改寫成「給定啟動延遲」的版本。Bar-Noy 等人另有 guaranteed start-up delay 的論文（SPAA'03／J. Discrete Algorithms 2006），本次只核到書目〔標題〕。
   - 「越併越大的群組」只有在 N ≫ 1 時才值得：VoD 的 √N 對 log N 差距要 N 夠大才明顯。我們每份文件在一個還原時間 T 內的到達數 N 是 **NOT_MEASURED**。
   - **可借用**：下界的推導方式，可用來回答「共享文件的還原 I/O 最少要多少」，作為新機制的理論上限比較。

---

## §3 DBMS 的 shared／cooperative scans 與 work sharing

### 3.1 Circular（synchronized）scans：商用系統；PostgreSQL；DB2 的 grouping and throttling

1. **定義與決策規則**
   - *attach*：新查詢看所有正在跑的掃描，挑剩餘重疊最多的一個，從它**目前的位置**開始讀；讀到範圍末端後回到開頭，讀到原來的位置為止。Zukowski 等人說 SQL Server、RedBrick、Teradata 都用這種作法〔全文：Zukowski et al. VLDB'07 §3，PDF p.3〕。
   - *elevator*：全系統只有一個嚴格依序的游標；查詢速度降到最慢者的速度，範圍查詢可能要等很久游標才到〔全文：同上 §3，PDF p.3〕。
   - PostgreSQL `synchronize_seqscans`（預設開）：掃描可能從表的中間開始、繞過結尾再回來，以便和進行中的掃描同步；這會改變沒有 ORDER BY 時的回傳順序〔文件：<https://www.postgresql.org/docs/current/runtime-config-compatible.html>〕。
   - DB2：依執行時行為動態分組、再分組相似的掃描，並依群組特性**節流**較快的掃描，讓它們保持在一起〔摘要：Lang et al. ICDE 2007〕。
2. **假設**：消費者不在乎資料到達的順序；掃描整個（或一段）表；查詢速度相近時效果最好（速度差太大會「脫鉤」）〔全文：Zukowski et al. §3，PDF p.3〕。
3. **LLM 有沒有用**：**NOT_FOUND**（見 3.2）。
4. **對應到我們〔判讀〕**：
   - (a) **直接違反**「從中間開始、繞回來」：晚到者不能先吃第 10 層再回頭吃第 0 層。這正是協調者在 (a) 舉的例子。
   - DB2 的「節流快的掃描」在我們這裡＝讓先來者等晚到者。I/O-bound 時沒有好處，見 §2.3。
   - **唯一可借用的是 elevator 的單游標**：大家共用一次由第 0 層到第 L−1 層的「掃描」，晚到者只能等下一輪。這就是 Regime S 下「以一輪為單位的 batching」，也是 Orca 式 iteration-level 加入的逐層版。

### 3.2 Cooperative Scans：Zukowski, Héman, Nes, Boncz（VLDB 2007），與其 in-order 後繼 PBM：Świtakowski, Boncz, Zukowski（PVLDB 2012）

1. **定義與決策規則**〔全文：Zukowski et al.〕
   - 架構：CScan 運算子事先向 Active Buffer Manager（ABM）登記要讀的範圍，並**接受資料亂序到達**；ABM 看所有 CScan 的需求，動態決定下一個載入哪一塊（chunk）、交給誰（§1，PDF p.1–2；§4，PDF p.4）。
   - *relevance* 策略（Figure 3，PDF p.4）：
     - `queryRelevance`：只有「餓了」（手上可用 chunk 少於 2 個）的查詢有優先權；還需要越少 chunk 的查詢越優先，並加上等待時間項以免長查詢永遠餓著。
     - `loadRelevance`：優先載入被最多「餓了」的查詢需要的 chunk，其次是被最多查詢需要的。
     - `useRelevance`：查詢先吃「感興趣的人最少」的 chunk，讓它可以早點被淘汰。
     - `keepRelevance`：淘汰時把「快餓了」的查詢也當成餓了來保護。
   - 動機例（§1，PDF p.1–2）：Q1 要 30 頁、Q2 要 10 頁、同時到達。round-robin 平均延遲 30；elevator 若先服務 Q2 平均 25，順序反過來則 35（兩者各為 30 與 40）。
   - 結果（§5.1，PDF p.6）：平均查詢延遲比 normal 好 3 倍、比 attach 好 2 倍。
   - 限制（§7.2，PDF p.11）：merge join、ordered aggregation 等**依賴順序的運算子**需要特別處理；兩邊都要掃且都用 relevance 時，作者建議退回普通 Scan。
   - **PBM（in-order 版）**〔全文：Świtakowski et al.〕：整合進 Vectorwise 時，有些查詢不能接受亂序（例如依實體順序的 ordered aggregation），於是讓 CScan 可以要求依序交付，代價是失去彈性（§2，PDF p.6）。另提出 Predictive Buffer Management：追蹤每個掃描的位置與速度，估計每一頁的**下一次被消耗時間**，越快會被用到的越保留，近似 Belady 的 OPT；效果接近 CScans、架構影響小很多（摘要，PDF p.1；§3，PDF p.6–7）。
2. **假設**：多數關聯運算子對順序不敏感（§3、§7.2）；chunk 夠大、可以攤提尋道；ABM 事先知道每個查詢的需求範圍。PBM 另假設長時間掃描的速度可以估計。
3. **LLM 有沒有用**：**NOT_FOUND**。Cooperative Scans 的 149 篇 S2 引用者 0 命中；PBM 的 12 篇 0 命中；web 搜尋 "cooperative scans"／"shared scans" + LLM 也沒有。
   - **精神最接近的 LLM 作法**：CachedAttention 依 job queue 開 look-ahead 預取窗（長度＝主機記憶體容量 ÷ 平均 session KV 大小），以及 look-ahead 淘汰窗：要淘汰的項目若在窗內，就改淘汰窗外的〔部分：arXiv 2403.19708v3 §3.3，PDF p.7〕。這就是「依佇列裡的未來需求保留」，和 PBM 同族，但只到 session 粒度、沒有逐層。
   - PEEK 用「保護排隊中請求會用到的祖先 block」的淘汰鉤子〔摘要：arXiv 2607.02525v2〕，也同族。
4. **對應到我們〔判讀〕**
   - (a)：cooperative scans 的核心是亂序交付，**直接衝突**；**PBM 才是我們要的版本**：依序交付，但用「下一次消耗時間」管理緩衝。換成層：請求 i 目前在第 ℓ_i 層、每層計算速度 v_i，則文件第 m 層對它的下一次消耗時間 ≈ (m − ℓ_i)/v_i（m > ℓ_i 時）；取所有請求的最小值決定保留順序。這是 PBM 原式（距離 ÷ 速度）的直接移植（推論）。
   - (d)：relevance 的「被最多餓了的請求需要的塊先載」在依問題挑頁時很自然：哪一頁被最多正在等的請求選中就先送。但必須限制在「同一層」內排序，因為 (a)。
   - (e)：relevance 在緩衝越小時優勢越大（§5.2、§9，PDF p.7、p.12）。我們的 HBM 緩衝通常也小，這是有利訊號。
   - (b)(c)：DB 沒有重算選項；載入一個 chunk 的成本也與位置無關。
   - **可借用**：PBM 的「下一次消耗時間」保留規則；relevance 的「優先服務需要最少的請求」（類 SRPT）與「載入被最多人需要的塊」。

### 3.3 QPipe：Harizopoulos, Shkapenyuk, Ailamaki（SIGMOD 2005）

1. **定義與決策規則**〔全文〕
   - On-demand Simultaneous Pipelining（OSP）：執行期偵測不同查詢的相同運算子，讓後到者「搭」先到者的運算子，把輸出同時送給多個上層（§1、§4）。
   - 例子（§3.1，PDF p.4）：記憶體只有兩頁，Q1 掃到一半時 Q2 到達；若 Q2 不在乎讀頁順序，就可以 piggyback 在 Q1 的掃描上，到檔尾再由新掃描補讀跳過的頁。若 Q2 要求依檔案順序，就做不到。
   - **Window of Opportunity（WoP）四類**（§3.2，PDF p.4–5）：
     - *linear*：任何時候加入都能用剩下的部分，節省從 100% 線性降到 0%，例如不在乎順序的掃描。
     - *step*：第一個輸出產生前加入可以完全分享。
     - *full*：整個生命期都能完全分享，例如單一聚合值。
     - *spike*：除非同時開始，否則不能分享，例如必須依表順序輸出的掃描。
   - 兩個「強化函數」（§3.2，PDF p.5）：*buffering*：運算子緩存 N 個輸出，可以把 spike 變成 step；*materialization*：若新的依序查詢選擇性很高（只要很少 tuple），就從進行中的掃描把它要的 tuple 存起來、之後依序重用，把 spike 變成斜率較小的 linear。
   - 依序運算子（§3.3，PDF p.5）：要求表順序的掃描是 spike WoP，不能利用進行中的掃描。
2. **假設**：運算子層級可以解耦查詢範圍；有記憶體做 buffering／materialization。
3. **LLM 有沒有用**：**NOT_FOUND**（S2 的 224 篇引用者 0 命中；OpenAlex 命中兩篇 DB 論文，無關）。
4. **對應到我們〔判讀〕**
   - **QPipe 的分類就是 (a) 的精確語言**：逐層還原對「必須從第 0 層開始」的晚到者是 spike WoP；buffering（在 HBM 存 W 層）把它變成 step（在落後 W 層以內都能分享）；這正是 (e) 的角色。
   - **materialization 對 (d) 很有用**：晚到者若每層只要少數頁，就只從進行中的串流存它要的頁，HBM 代價小很多，等於把 spike 變成 linear。
   - **可借用**：WoP 分類可以直接用來描述我們的機制（「逐層還原是 spike；HBM 緩衝讓它變 step；依問題挑頁讓它近似 linear」）。這是寫論文時很好的古典對照語言。

### 3.4 CJOIN（PVLDB 2009）、Crescando（PVLDB 2009）、SharedDB（PVLDB 2012）：批次化的全域計畫

1. **定義與決策規則**
   - CJOIN：用一個「常開」的非阻塞運算子管線服務所有進行中的 join 查詢，分享 I/O、計算與 tuple 儲存；控制器持續看查詢組合做執行期最佳化；在數十到數百個並行查詢下比商用系統快一個數量級〔摘要〕。
   - Crescando：主記憶體中的平行協作掃描，加上 data-stream 的 query-data join；不一定對每個工作負載最佳，但對所有工作負載給延遲與新鮮度保證〔摘要〕。
   - SharedDB〔全文〕：處理一批查詢與更新時，新到的先排隊；這批做完就清空佇列形成下一批，像「心跳」（§3，PDF p.4）；最壞情況下批次讓延遲變 2 倍：一個週期排隊、一個週期處理（PDF p.5）；換來的是計算量有上界、不隨並行查詢數線性成長，所以能給回應時間保證。吞吐量約為一個頂級商用系統的 2 倍、MySQL 的近 8 倍（TPC-W，PDF p.1）。
2. **假設**：高並行、查詢之間重疊大；低負載時會比逐查詢處理差（SharedDB 自己承認，PDF p.1）。
3. **LLM 有沒有用**：KV 還原語境 **NOT_FOUND**（CJOIN 105、Crescando 127、SharedDB 171 篇 S2 引用者，0 命中）。DB 社群的 agentic workflow 系統 Halo 引用 SharedDB，把相同的工具或 DB 呼叫合併成一次實體執行〔部分：arXiv 2509.02121v2，PDF p.3〕；對 LLM 運算子，它依賴的是 intra-batch prefix sharing。
4. **對應到我們〔判讀〕**：SharedDB 的「一批在跑時新到的排隊、跑完下一批」＝Regime S 的「以一輪逐層還原為一個心跳」。它的 2 倍延遲上界可以直接搬：晚到者最多等一輪再跑一輪。但若一輪很長，這比 patching 差。**可借用**：延遲上界的論證方式（2 倍）與「低負載時要退回逐請求」。

### 3.5 Shared scans 的排程：Agrawal, Kifer, Olston（PVLDB 2008）

1. **定義與決策規則**〔全文〕
   - 問題：Map-Reduce 類環境中，大量工作要掃少數幾個大檔；可以延後工作，讓同一檔的工作湊在一起共用一次掃描（§1，PDF p.1）。他們說自己是第一個提出這個排程問題的；cooperative scans 解決的是「同時跑的工作怎麼分享」，他們解決的是「怎麼排佇列，讓可分享的工作一起跑」（§2.1，PDF p.2）。
   - 模型：工作族 F_i 的可分享部分（掃描）時間 t_i^s、不可分享部分 t_i^n、到達率 λ_i、目前佇列長度 B_i。排程器每次空閒就挑一族，把整族當一批執行（§3.3–4.1，PDF p.4–5）。
   - SJF 推廣：P_i = −(t_i^s / B_i + t_i^n)。這在有分享時表現差：會太少排熱門族（PDF p.4）。
   - **AA Policy 1：P_i = B_i² / (λ_i · t_i^s)**；AA Policy 2 再減 t_i^s · Σ_j λ_j。直觀解釋：未來同類工作越多、可分享部分越長（λ_i · t_i^s，作者稱為 *sharability*），越該延後等人；佇列越長，優先權以平方成長，最終逼它執行；不可分享的單一工作（t^s = 0）優先權無限大，先做（PDF p.5）。
   - 假設之一是 **No idle**：佇列不空就不讓執行器閒著（PDF p.4）。
2. **假設**：Poisson 類到達、λ 可線上估計；同族工作可以完全共用掃描；執行器一次只跑一批；延遲以 perceived wait time 衡量。
3. **LLM 有沒有用**：**NOT_FOUND**（S2 的 93 篇引用者 0 命中；OpenAlex 命中 3 篇 MapReduce 論文）。
4. **對應到我們〔判讀〕**
   - **這是本檔找到最直接可用的「還原排程」規則**。t_i^s＝文件 i 的 KV 還原時間（可分享）；t_i^n＝每個請求自己問題的 prefill 與 decode（不可分享）；B_i＝正在等文件 i 的請求數；λ_i＝文件 i 的到達率。
   - (a)：規則以「一整批一起從頭還原」為單位，與 (a) 相容。
   - (b)：t_i^s 不是常數，可以用 Cake 式切分改變（部分重算），等於「可分享部分的長度」也是決策變數。
   - (c)(d)：可分享部分的大小依位置與問題而變；兩個請求只分享頁的交集。
   - (e)：一批越大，HBM 裡的 per-request 狀態越多。
   - **差異**：它的「No idle」與 anticipatory 的「可以空等」相反；在我們的情境，兩者可以合起來：佇列裡有別的工作時用 AA 排序，佇列只剩一族且 I/O 會閒時用 anticipatory 判斷要不要等（§4）。

### 3.6 分享不一定好：Johnson et al.（VLDB 2007）；Psaroudakis, Athanassoulis, Ailamaki（PVLDB 2013）；MQO（Sellis, TODS 1988）

1. **定義與結論**
   - Johnson 等人〔全文〕：在多核上，激進的 work sharing 會在分享點（pivot operator）**部分序列化**分享者的執行；若序列化落在關鍵路徑上，平行度的損失會超過省下的工作。他們用一個簡單分析模型預測何時該分享；依模型選擇性分享，比「永不分享」平均快 20%，比「總是分享」快 2.5 倍（摘要與 §1，PDF p.1；§3，PDF p.4）。
   - Psaroudakis 等人〔部分：讀了摘要與引言〕：比較 Simultaneous Pipelining（SP）與 Global Query Plans（GQP）。兩者正交；push 式 SP 在低並行時有負面影響，pull 式可消除；低並行時以「逐查詢＋SP」較好，高並行時以「GQP＋SP」較好（摘要，PDF p.1）。
   - Sellis 的 multiple-query optimization：辨識並行查詢的共同子運算並共用〔標題〕。
2. **假設**：瓶頸在 CPU 平行度而非 I/O；對 I/O-bound 的工作，消除重複 I/O 的分享特別有效（Johnson et al. 自述，PDF p.1）。
3. **LLM 有沒有用**：KV 語境 **NOT_FOUND**。Helium（DB 視角的 agentic workflow serving）引用 Sellis 做 common subgraph elimination，對象是 workflow 的子圖〔部分：arXiv 2603.16104v1，PDF p.4〕。Johnson et al. 2007 的引用掃描沒完成：S2 與 OpenAlex 都沒拿到這篇的 ID，web 搜尋也沒看到 LLM 引用，所以只能算部分查核下的 NOT_FOUND。
4. **對應到我們〔判讀〕**：這是新機制的**主要失敗模式**。逐層共享讓晚到者與先來者在每一層同步：誰慢誰拖住大家，類似 elevator 降到最慢者速度。若瓶頸是 GPU（計算型還原、Cake 前段重算），強迫同步就是序列化；若瓶頸是 I/O，分享才划算。**可借用**：先用簡單模型判斷「這一刻是 I/O-bound 還是 compute-bound」，再決定要不要分享；不要預設分享一定好。

---

## §4 Anticipatory disk scheduling：Iyer & Druschel（SOSP 2001）

1. **定義與決策規則**〔全文〕
   - 現象：*deceptive idleness*。同步 I/O 的 process 剛完成一個請求、還沒發下一個時，work-conserving 的排程器以為它沒事了，立刻轉去服務別的 process；結果每個 process 只能連續被服務一個請求，失去空間局部性（§1–2，PDF p.1–2）。
   - 三個必要條件：多個 process 並行發同步請求；磁碟請求不可搶佔；排程器 work-conserving。解法是拿掉第三個：完成一個請求後**短暫等待**，看同一個 process 會不會發一個鄰近的請求（§3，PDF p.3）。
   - 成本效益式（seek-reducing 排程器，§3.3，PDF p.5）：benefit＝候選請求的定位時間 − 最近那個 process 的預期定位時間；cost＝max(0, 預期中位 think time − 已過時間)；等待長度＝max(0, 預期 **95 百分位** think time − 已過時間)；benefit > cost 才等。等待期間新請求不會重設逾時，避免無限等待（§3.2，PDF p.4）。
   - 結果（摘要，PDF p.1）：FreeBSD 實作中，Apache 在磁碟密集負載下吞吐量多 29%–71%；TPC-B 變體改善 2%–60%；Andrew 檔案系統基準快 8%。
2. **假設**（§3.1，PDF p.4）：#1 有依賴的同步請求來自**同一個 process**，所以只需要等「上一個發請求的 process」；#2 每個 process 的局部性大致穩定，可以線上學習。另一個隱含前提是「切換的代價（尋道）大」。
3. **LLM 有沒有用**：**NOT_FOUND**。S2 的 289 篇引用者 0 命中；web 搜尋 "anticipatory scheduling" + LLM／KV 也沒有。
   - 精神相同、但沒有引用的 LLM 作法：
     - Continuum：工具呼叫期間把 KV 釘在 GPU，TTL 依預測的工具呼叫時間、重載成本與驅逐造成的排隊延遲決定；超時就驅逐（PDF p.1、p.5–6）〔部分〕。
     - SGLang HiCache 從 L3 預取的終止策略：`best_effort`（GPU 可以算就停）、`wait_complete`（等全部到齊）、`timeout`（等到時限）；停止後已取回的部分與本地資料一起做 prefill〔文件：`docs/docs/advanced_features/hicache_design.mdx`，§Prefetch from L3，sglang main 分支 2026-10-26 讀取〕。
4. **對應到我們〔判讀〕**
   - 假設 #1 在「同一使用者的下一輪」成立（多輪對話、agent 工具呼叫），這就是 Continuum 的情境。在「**不同使用者問同一份文件**」上不成立：我們要等的是另一個 process（另一個使用者），它的到達率不能用 think time 學，要用文件熱度 λ_i 估。
   - 改寫後的規則：只有 I/O 閒著、佇列裡沒有別的工作時才考慮等。等待的 benefit＝在時間窗內有人來的機率 × 省下的一次還原；cost＝先來者多等的時間。等待長度取到達間隔的高百分位，類比 95 百分位。
   - 另一個更自然的對應是「**保留**」而不是「空等 I/O」：文件 A 的請求剛做完，不要立刻為文件 B 驅逐 A 的 KV，先保留一下等 A 的下一個使用者。這是 non-work-conserving 的驅逐，Continuum 已經在 session 粒度做了。
   - (a)(b)(c)(d)：無直接衝突；(e) 決定能保留多久。
   - **可借用**：成本效益式的形式、95 百分位逾時、「逾時不可被新事件重設」。

---

## §5 Request coalescing 與 thundering herd

1. **定義與決策規則**
   - nginx `proxy_cache_lock`（1.1.12 起）：開啟時，同一個 cache key 一次只放一個請求去後端填快取，其他請求等回應出現或等鎖釋放；`proxy_cache_lock_timeout`（預設 5s）到時就自己去後端，但回應不寫入快取；`proxy_cache_lock_age`（1.7.8 起，預設 5s）：上一個填快取的請求太久沒完成，就再放一個去〔文件：<https://nginx.org/en/docs/http/ngx_http_proxy_module.html>〕。
   - Varnish：多個 client 要同一頁時，只送一個請求到後端，其他的掛起等待，自動進行（他們稱為 request coalescing）；文件也指出等待佇列很大時，一次放出所有等待者本身就會造成 thundering herd；*grace* 讓 TTL 過期後的一段時間內先回舊物件、背景抓新的〔文件：<https://www.varnish.org/docs/users-guide/vcl-grace/>〕。
   - Go `singleflight`：「duplicate function call suppression」；`Do` 保證同一個 key 同時只有一個執行在飛，回傳的 `shared` 表示結果是否給了多個呼叫者〔文件：<https://pkg.go.dev/golang.org/x/sync/singleflight>〕。
   - memcache leases〔全文：Nishtala et al. NSDI'13 §3.2.1，PDF p.4〕：cache miss 時發 lease token；每個 key 每 10 秒只發一次 token，其他請求收到「稍等再試」；一週量測中，易發生 thundering herd 的 key 讓資料庫峰值查詢率從 17K/s 降到 1.3K/s。
   - Probabilistic early expiration（XFetch）〔全文：Vattani et al. PVLDB 2015〕：§1（PDF p.1）列出 locking 的問題：沒拿到鎖的請求要等、要回舊值、要調鎖的 TTL，而且不容錯。他們的作法是在過期前以機率提早重算，機率由重算時間 Δ 與參數 β 決定（XFetch，PDF p.6）；指數分佈在「提早過期間隔」與「stampede 大小」的取捨上是最佳的（Theorem 7–8，PDF p.4）。
2. **假設**：key 完全相同；結果一次性完整產生；所有等待者都要**完整**結果；等待者無法自己用更便宜的方式拿到結果，除非逾時後重做。
3. **LLM 有沒有用**：**有工程實作，沒有論文用這個名字，也沒有逐層版本。**
   - SGLang in-batch prefix caching：等待佇列中，若一個請求和已有快取的匹配很短（≤ `IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD`，預設 32 token），但和**另一個等待中的請求**共用的前綴 ≥ `IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD`（預設 32），就暫時降低它的優先權，只先排一個，讓後來的命中快取。這就是 singleflight 的領頭／跟隨〔文件：`python/sglang/srt/managers/schedule_policy.py` `_compute_prefix_matches`，該檔最新 commit 8252e3a，2026-10-25〕。
   - vLLM 原生 CPU offloading：同一批 KV block 不允許同時 onload，所以共用前綴的短請求必須等長請求的所有 block 都載完，形成隊頭阻塞；RFC #33526 提議分批漸進載入〔文件〕。
   - Strata：會被 delay hit 影響的請求延後到 delay hit 解除後再執行〔全文：§4.3.1，PDF p.6〕。
   - PEEK：讓前綴群組的「先鋒」先進場，其他成員繼承剛快取的前綴〔摘要：arXiv 2607.02525v2〕。
   - Halo：合併相同的工具或 DB 呼叫，引用 SharedDB 與 DataLoader〔部分：PDF p.3〕。
   - S2 掃描 Vattani'15（1 篇引用者）0 命中。
4. **對應到我們〔判讀〕**
   - **「全有或全無」假設被 (a) 的逐層結構打破**：KV 一層一層到，部分結果就可以用。所以 KV 版 coalescing 應該是「**跟隨者逐層跟上**」，不是等領頭者整個做完。vLLM RFC 的隊頭阻塞正是全有或全無的代價。
   - nginx 的 `lock_timeout`／`lock_age` 是「不等了、自己去拿」；在我們這裡，「自己去拿」可以是 (b) 自己重算一部分。HiCache 的 `timeout` 策略也是這個形狀，但只用在自己的預取上。
   - (d)：跟隨者要的頁可能只是領頭者的子集，也可能多出一些；多出的部分要另外取。
   - (e)：跟隨者落後時，要有地方存領頭者已經送來的層。
   - Varnish 指出的「一次放出所有等待者」造成 herd，在 GPU 上對應「一批跟隨者同時開始 prefill」，可能變成計算尖峰，這和 §3.6 的序列化問題同源。
   - **可借用**：領頭／跟隨結構、逾時改走別條路、「不要一次全部放出」。

---

## §6 Delayed hits：只補 LLM 應用面

（機制本身已在 `docs/research_20260924/os_toolbox.md` B 組核對，這裡不重做。）

- **Strata**〔全文：arXiv 2508.18572v1〕：§3.2 說明 delay hit（引用 Atre 等人 [5]）在長 context 服務中更常發生（PDF p.4）。§4.3.1 列出兩種造成重複計算的情況：共用同一個 cache miss 的多個請求被排進同一批；或在 miss 解決前被排進後續批次。處理方式是把可能受影響的請求延後到 delay hit 解除後（PDF p.6）。§4.3.2 的 balanced batch formation 用「載入量 ÷ 計算量」判斷一批是否 loading-bound；共用同一 context 的請求排在一起叫 *bundle hit*，和 delay hit 相對（PDF p.6–7）。
- **其他 LLM 論文**：Atre'20 的 67 篇 S2 引用者中，LLM／KV 相關只有 Strata（另有 Vulcan 用 LLM 搜尋系統啟發式，以及兩篇「生成式 CDN」，都不是 KV 快取）。web 搜尋 "delayed hits" + KV／prefix caching 沒有找到其他論文。結論：**除 Strata 外 NOT_FOUND**。
- **(a)–(e) 對照〔判讀〕**：Strata 的粒度是整段 context。在 Regime R 下，逐層版的 delay hit（晚到者在已到的層上先算）是 Strata 沒做、vLLM RFC 只在 block 方向做的空間。Atre 的離線最優 Belatedly 以 min-cost multi-commodity flow 求延遲最優（os_toolbox B1 已核對），可以當新機制的離線上界工具。

---

## §7 Inference serving 的等待湊批（只寫和「等共享者」相關的部分）

1. **Clipper**（NSDI 2017）〔全文〕：用 AIMD 調整最大 batch size（§4.3，PDF p.8）；*delayed batching*（§4.3.2，PDF p.8）：負載中等或突發時，短暫延後送出讓更多查詢到達。動機類似 Nagle 演算法：效益來自「每批固定成本 ÷ 增加一個的邊際成本」。Scikit-Learn SVM 延後 2ms 得到 3.3 倍吞吐；Spark SVM 沒有增益。
2. **Nexus**（SOSP 2019）〔全文〕：*lazy dropping*（錯過期限才丟，依最早請求剩下的預算決定 batch 大小，像 Clipper）在固定成本高時會讓 batch 太小、越落越後（§4.3，PDF p.6）；*early drop* 用滑動窗跳過會讓批次變差的請求，模擬中吞吐最多高 25%（PDF p.10）。*prefix batching*：只差最後幾層的模型，把共同的前幾層一起 batch，後綴再分開跑（PDF p.10）。
3. **BatchMaker**（EuroSys 2018）〔摘要〕：*cellular batching*：以 RNN cell（共享權重的子圖）為粒度做批次決策，請求隨時加入或離開。
4. **FlexGen**（arXiv 2303.06865v2）〔全文〕：計算圖的列是 batch、行是層；逐列走要反覆載入權重；**逐行走（column-by-column）讓一層權重留在 GPU 給整行的 batch 用**，但不能一路走到底，因為 activation 與 KV 要存，所以停在記憶體滿的地方，形成 zig-zag block 排程（§4.2，PDF p.4–5）；Theorem 4.1：其 I/O 複雜度在最優的 2 倍以內（PDF p.5）。
5. **Hydragen**〔摘要：arXiv 2402.05099v2〕：把共享前綴與各自後綴的 attention 分開算，前綴部分把跨序列的 query 批在一起，減少重複記憶體讀取。Feather〔摘要：arXiv 2605.06046v3〕：在前綴共享負載下，小而前綴同質的 batch 可能比大而異質的 batch decode 更快，用 RL 學 batch 大小與同質性的取捨。

**LLM 有沒有用**：Clipper、Nexus、BatchMaker 被大量 LLM serving 論文引用：BatchMaker 被 vLLM／PagedAttention、Pensieve 等引用（OpenAlex）；Clipper 被 Parrot、Pensieve 引用（全文 grep）。但都是當一般批次化背景。「**為了等共享者而延後**」的規則：NOT_FOUND。最接近的是 Strata 的 bundle hit（同批，不等）與 SGLang 的領頭／跟隨（排序，不等）。

**對應到我們〔判讀〕**
- **BatchMaker 能在中途加入，是因為共享的東西（cell 權重）每一步都被重用**。我們共享的東西（文件第 ℓ 層 KV）對每個請求在一輪裡只用一次而且依序，晚到者無法用「現在這一層」。這正好說明為什麼要用 patching 而不是 cellular batching 來對應 (a)。但 decode 例外：decode 每一步都重讀所有層，所以在 Regime S 的 decode 中，「每一步一輪、在一輪的邊界加入」就是 cellular／iteration-level 的逐層版。
- **FlexGen 是 LLM 自己的「VoD batching under (a)+(e)」**，只是共享物件是權重而不是文件 KV。它的 2 倍定理是我們想做的「共享文件 KV 的逐層 batching」最直接的前作與理論參照。
- Clipper 的 Nagle 式論證可以搬來判斷「延後多久值得」：每次還原的固定成本（例如 SSD 讀取的啟動、逐層 DMA 的小傳輸開銷）÷ 多一個共享者的邊際成本。
- Nexus 的 prefix batching 是「共同前段一起做、後段分開」，在模型維度；Hydragen 是同樣的事在 attention／token 維度。我們的版本在「還原 I/O」維度。

---

## §8 兩機 flow shop（協調者補充）

### 8.1 Johnson's rule（Johnson 1954）

1. **定義**：每個工件先在機器 1、再在機器 2 加工，每台機器一次只做一件，已知各工件在兩台機器的時間；論文給出使總完工時間（makespan）最小的簡單決策規則，並討論一個受限的三機情況〔摘要：RePEc／IDEAS 收錄的 NRLQ 摘要〕。
   - 規則的標準教科書陳述：機器 1 時間 ≤ 機器 2 時間的工件排前面、依機器 1 時間遞增；其餘排後面、依機器 2 時間遞減。這是常識性陳述，**原文措辭 NOT_VERIFIED**（沒取得 PDF；RAND P-402 與 Bellman 的 RAND P-456 都沒下載成功）。
   - 複雜度邊界〔摘要：Garey, Johnson, Sethi, Math. of OR 1976〕：m ≥ 3 的 flow shop 最小 makespan 是 NP-complete（m = 2 有有效演算法）；**最小平均流程時間（mean flow time，等價於 ΣC_j）在 m ≥ 2 時就是 NP-complete**，而且是強 NP。Gonzalez & Sahni（Operations Research 1978）證明可搶佔與不可搶佔的 flow shop／job shop 最小完工時間是 NP-complete〔摘要；兩機可搶佔的特例本次 NOT_VERIFIED〕。
2. **假設**：兩台機器各自獨立、一次一件、加工時間固定；工件之間沒有共享；中間緩衝無限；目標是 makespan。
3. **LLM 有沒有用**：**KV 載入與計算的排程 NOT_FOUND。**
   - OpenAlex：Johnson 1954 的 3,259 篇引用者中，「LLM／language model／KV／transformer」過濾命中 14 篇。只有兩篇是 LLM 推論：一篇 ICC 2026 的協作邊緣推論論文，以及 Zhu 等人的 arXiv 2510.11331。後者把 draft 與 verify 視為兩段管線，引用 Johnson 寫出「第 m 批的第二段要等自己的第一段與第 m−1 批的第二段都完成」的遞迴，**沒有用 Johnson 的排序規則**〔部分：PDF p.7〕。前者 NOT_VERIFIED（沒讀）。
   - web 搜尋 "Johnson's rule" + LLM／KV／transformer：沒有找到。
   - Garey'76 的 LLM 過濾命中中，只有 Hetis（SC'25，異質 GPU serving）是 LLM serving；引用用途 NOT_VERIFIED（沒讀）。

### 8.2 Flexible operation：Gupta, Koulamas, Kyparisis, Potts, Strusevich（Annals of OR 2004）；Crama & Gültekin（J. Scheduling 2010）

- **協調者記得的論文存在，書目已核對**：J. N. D. Gupta, C. P. Koulamas, G. J. Kyparisis, C. N. Potts, V. A. Strusevich, "Scheduling three-operation jobs in a two-machine flow shop to minimize makespan," *Annals of Operations Research* 129:171–185, 2004, doi:10.1023/B:ANOR.0000030687.72169.c7（Crossref；S2；OpenAlex 一致）。
  - 內容〔摘要：RePEc／IDEAS〕：每個工件三道工序，第一道必在機器 1、第三道必在機器 2，**中間那道可在任一台機器上做但不可搶佔**；問題 NP-hard；任意順序、不插入閒置的排程最壞比 2；建構四個排程取最好者最壞比 3/2；另有 PTAS。
- Crama & Gültekin, "Throughput optimization in two-machine flowshops with flexible operations," *J. Scheduling* 13(3):227–243（Crossref 的 published 欄為 2009，卷 13 為 2010）〔摘要：ORBi 收錄頁〕：相同工件的兩機 flow shop，部分工序預先指定機器，其餘「彈性工序」可以在任一台做，而且同一彈性工序對不同工件可以放在不同機器；目標是最大化產出率；分別處理不同工件數與**兩機之間緩衝容量**的情況。
- **LLM 有沒有用**：**NOT_FOUND**（S2：Gupta'04 的 20 篇、Crama'10 的 14 篇引用者 0 命中；OpenAlex 0）。

### 8.3 可分割的版本：lot streaming（Potts & Baker 1989；Trietsch & Baker 1993）

- Lot streaming＝把一個工件切成子批（sublot），讓工序可以重疊。Potts & Baker 研究在 lot streaming 下最小化 flow shop 的完工時間：何時單工件模型的各機器可用相同子批大小、以等大小子批作為啟發式，並指出 n 工件情況的困難〔摘要：IAOR 轉述，原文 NOT_VERIFIED〕。Trietsch & Baker 綜述連續與離散子批、有無間歇閒置、一致與可變子批等模型，先解兩機再推廣〔摘要〕。
- 「malleable／divisible load」：Bharadwaj, Ghose, Robertazzi 的 divisible load theory 綜述（Cluster Computing 2003）〔標題〕，本次未細查。
- **LLM 有沒有用**：**NOT_FOUND**（Potts–Baker 的 152 篇、Trietsch–Baker 的 158 篇 OpenAlex 引用者，LLM 過濾命中的都是製造業論文）。

### 8.4 加權完成時間

- 單機：Smith（NRLQ 1956）〔標題：RePEc 無摘要〕。WSPT（依 w_j/p_j 遞減排序可最小化 Σw_jC_j）是教科書常識，原文頁碼與措辭 NOT_VERIFIED。
- 兩機 flow shop：ΣC_j 已是強 NP-complete（Garey et al. 1976，見 8.1），所以 F2‖Σw_jC_j 至少一樣難（由特例 w_j = 1 推得）。

### 8.5 對應到我們〔判讀〕：哪裡成立、哪裡不成立

**成立的部分**
- 機器 1＝I/O（CPU／SSD → HBM），機器 2＝GPU 計算。
- 一個還原工作＝「載入 a_j」接「計算 b_j（問題的 prefill、或重算部分）」。
- **逐層還原就是 lot streaming**：子批＝層，大小一致；(a) 讓機器 2 必須依序處理子批，與 lot streaming 的「一致子批」模型相合。LMCache、HiCache、CachedAttention 的「載入第 N+1 層時計算第 N 層」就是 L 個子批的兩機 lot streaming。
- **Cake 的切分＝flexible operation**：中段 token 的 KV 可以「在機器 1 載入」或「在機器 2 重算」。和 Gupta 等人的差別是：我們的中段**可以任意分割**（任何 token 邊界都行），不是不可搶佔的整道工序。單一工作在可分割時很好解：Cake 在中間相遇；CacheFlow 求 min max(T_comp, T_load)。難的是**多工作的排序**。
- **共享群組＝權重**：等同一份文件的請求數 w_j 當作 Σw_jC_j 的權重。若 I/O 是瓶頸機器、計算很小，就退化成單機 WSPT：依「等待人數 ÷ 還原 I/O 量」遞減排序。這和 Agrawal 的 B_i/t_i^s 同一家族；AA Policy 1 的 B_i²/(λ_i t_i^s) 再把未來到達算進來（§3.5）。
- **已在 HBM 的請求只需要機器 2**：加工時間為 0 的「缺工序」，flow shop 本來允許。
- Johnson 的直覺可以借：先排 I/O 輕的工作讓 GPU 早點有事做，最後排計算輕的工作避免 GPU 尾巴太長。

**不成立或要改的部分**
1. **GPU 不是一次一件的機器。** 它可以同時跑多個工作（多條 stream、MPS），而且批次處理時間對批大小是次線性的。共享同一文件的工作可以批在一起（Hydragen）。所以機器 2 更像 batch-processing machine，不是 unary machine。
2. **兩台機器不獨立。** I/O 的 DMA 與計算搶 HBM 頻寬（解壓縮或重建時還搶 SM），所以加工時間依「同時在跑什麼」而變，違反固定 p_ij。repo 的 stage-3 設計把「爭用感知還原」當主線，就是在處理這件事（`docs/stage3_design_20261024/`，本檔沒有重讀細節）。
3. **機器 2 上有背景負載**：其他已在 HBM 的請求的 decode 迭代持續佔用 GPU，等於機器 2 有不可用時段或被分走容量。
4. **線上到達**：共享者陸續到達，是 F2|r_j|Σw_jC_j；權重 w_j 還會隨時間增加（更多人來等同一份文件）。古典模型沒有「權重會長大」。
5. **(e)＝有限中間緩衝**：機器 1 送完、機器 2 還沒用的層要佔 HBM。這是有限緩衝（blocking）flow shop；Crama & Gültekin 正好分析了不同的兩機間緩衝容量〔摘要〕。
6. **(c)＝彈性工序在兩台機器上的時間不對稱而且依位置**：重算 token 區間的成本隨位置上升（attention），載入成本與位置無關。Gupta 等人的模型裡，彈性工序在兩台機器上可以有不同時間，但不依「切在哪」而變。
7. **目標不同**：Johnson 最小化 makespan；我們在意每個請求的 TTFT（或其 P95），比較接近 ΣC_j、加權版或 max lateness。這些在兩機時就已是強 NP-hard（8.1），只能用啟發式或近似。

**LLM 前作**
- CacheFlow：一批請求的 token × layer 樓梯形切分 + DP，目標 min max(T_comp, T_load)，Lemma 1 給出含後綴 prefill 的 makespan 上下界〔部分：arXiv 2604.25080v2 §3.2，PDF p.5；PDF p.15〕。它是「可分割彈性工序、兩資源、批次 makespan」，**沒有工作排序與加權完成時間，也沒有跨請求去重共享前綴**（batch DP 把每個請求的 chunk 分開計算，PDF p.5–6）。
- KVPR：用線性規劃選「先傳部分 activation 讓 GPU 重算、其餘 KV 同時傳輸」的切點〔部分：arXiv 2411.17089v2，全文 grep "linear program"／"split point"〕。
- Cake：雙向在中間相遇〔部分〕。
- 這三篇都沒有提 flow shop 或 Johnson（全文 grep）。

**可借用的方向**〔判讀〕：把多請求、多文件的還原寫成「兩機（I/O、GPU）＋可分割彈性工序＋有限緩衝＋加權（等待人數）＋線上到達」的 flow shop 變體；用 WSPT／AA 優先權排序、用 CacheFlow 式 DP 決定每個工作的切點、用 lot streaming（逐層）做管線。這個組合在排程文獻是否已有對應模型（例如兩段式 assembly flow shop）：**本次沒有查證**。

---

## §9 對照總表

| 機制（出處、證據） | 決策規則 | 主要假設 | 已被 LLM／KV 用過？ | (a)–(e) 哪些衝突〔判讀〕 | 可借用的部分〔判讀〕 |
|:--|:--|:--|:--|:--|:--|
| VoD batching（Dan et al. MM'94、MMSJ'96〔摘要〕；MFQ, Aggarwal et al. ICMCS'96〔摘要〕） | 有空串流時選一部片 multicast：FCFS（最老請求）可勝過 MQL（最多人）；MFQ 用對熱門片不利的權重 | 可延後開始；整檔相同；串流數硬上限；觀眾會放棄 | NOT_FOUND；精神相近：Strata bundle hit、SGLang LSPF、DLPM、RAGCache 重排 | (a) 相容；(e) 批越大 per-request 狀態越多；I/O 閒置時延後是浪費 | FCFS 對 MQL 的教訓；最大等待保證 |
| VoD patching（Hua et al. MM'98〔摘要〕；Eager et al. TKDE'01 §2〔全文〕；Gao & Towsley〔摘要〕） | 晚到者加入進行中 multicast 並緩存，另開 patch 補前段；超過門檻 y 就開新的完整串流；y* = (√(2N+1)−1)/N，頻寬 √(2N+1)−1 | 客戶端緩衝夠；依序消耗；r = 1；客戶端 2 倍接收；Poisson | **NOT_FOUND**；最近：vLLM RFC #33526（block 方向漸進載入） | **(a) 完全對得上**；(e) 緩衝在共用 HBM，Regime R 下 patch 免費、Regime S 才有意義；共用 PCIe 讓 patch 拖慢先來者；(b) 可重算補、(c) 讓重算型 patch 變貴 | **逐層 patching**；門檻式的形式（不能直接套 y*） |
| Piggybacking（Golubchik et al.〔摘要／二手〕；Aggarwal et al. SIGMETRICS'96〔摘要〕） | 調播放速率讓相近串流合併 | 接收頻寬＝播放速率；觀眾容忍小幅變速 | NOT_FOUND（"piggyback" 只出現在 Sarathi 標題的另一個意思） | Regime R 下晚到者天生追得上；Regime S 下追趕需要 (b) 重算 | 合併後只送一份 |
| Stream merging／HMSM／下界（Eager et al. §4–5〔全文〕；Bar-Noy & Ladner〔標題〕） | 反覆合併成更大群組；任何即時服務技術頻寬 ≥ ln(N+1) | 緩衝夠；依序；r = 1 | NOT_FOUND | 我們 I/O 比計算慢，做不到「立即」，下界要改成給定啟動延遲版；N ≫ 1 才值得 | 下界的推導結構（層 ↔ 位置） |
| Circular／synchronized scan（Zukowski §3〔全文〕；PostgreSQL〔文件〕；DB2 Lang et al.〔摘要〕） | 從進行中掃描的位置開始、繞回補前段；elevator 單游標；DB2 分組＋節流 | 消費者不在乎順序 | NOT_FOUND | **(a) 直接違反**（不能先吃後層再回頭） | 只剩「單游標、下一輪再加入」 |
| Cooperative Scans（VLDB'07〔全文〕）＋PBM（PVLDB'12〔全文〕） | ABM 依 relevance：先服務餓了且需要最少的查詢；載入被最多餓了的查詢需要的 chunk；PBM：依「下一次消耗時間」保留，近似 OPT，依序交付 | 多數運算子不在乎順序（PBM 放寬）；需求事先登記；速度可估 | NOT_FOUND；精神近：CachedAttention look-ahead 預取／淘汰窗、PEEK 保護祖先 block | 亂序交付違反 (a)；**PBM 相容**；(d) 時 relevance 很合適但要限制在層內 | **PBM 的下一次消耗時間＝(m−ℓ_i)/v_i**；「載入被最多等待者需要的頁」 |
| QPipe（SIGMOD'05〔全文〕） | OSP；WoP 四類：linear／step／full／spike；buffering 把 spike 變 step；materialization 把 spike 變 linear | 運算子可與查詢解耦；有記憶體 | NOT_FOUND | 逐層還原對晚到者是 spike；(e)＝buffering；(d) 讓 materialization 便宜 | **WoP 分類當論文的描述語言** |
| CJOIN／Crescando（PVLDB'09〔摘要〕）；SharedDB（PVLDB'12〔全文〕） | 常開共享管線；批次「心跳」：在跑時新到的排隊；最壞延遲 2 倍 | 高並行、高重疊；低負載較差 | KV 語境 NOT_FOUND；Halo 引用 SharedDB 合併工具呼叫 | 一輪很長時 2 倍上界太鬆 | 2 倍延遲上界的論證；低負載退回逐請求 |
| Shared-scan 排程（Agrawal et al. PVLDB'08〔全文〕） | **AA Policy 1：P_i = B_i²/(λ_i t_i^s)**；單一不可分享工作先做；No idle | λ 可估；同族完全共享；一次跑一批 | NOT_FOUND | t^s 可被 Cake 切分改變；(d) 只分享交集 | **還原排程的優先權式** |
| 分享不一定好（Johnson et al. VLDB'07〔全文〕；Psaroudakis et al.〔部分〕；Sellis MQO〔標題〕） | 用模型判斷何時分享：選擇性分享比永不分享快 20%、比總是分享快 2.5 倍 | 瓶頸在 CPU 平行度 | KV 語境 NOT_FOUND；Helium 用 Sellis 做 workflow CSE | 逐層同步讓最慢者拖住大家；compute-bound 時分享是序列化 | **新機制的失敗模式與開關條件** |
| Anticipatory scheduling（SOSP'01〔全文〕） | 完成後短暫空等同一 process 的鄰近請求；benefit > cost 才等；等到 95 百分位 think time；逾時不重設 | 依賴請求來自同一 process；局部性穩定；切換代價大 | NOT_FOUND；精神近：Continuum TTL、HiCache 預取 `timeout` | 共享者是「別的 process」，要用文件熱度而不是 think time；更自然的對應是「保留」 | 成本效益式、高百分位逾時、不重設逾時 |
| Request coalescing（nginx／Varnish／singleflight〔文件〕；memcache leases〔全文〕；XFetch〔全文〕） | 同 key 一次只放一個去後端，其餘等；逾時自己去；lease 每 key 每 10s 一個 token | key 相同；結果一次完整；等待者要完整結果 | **有工程實作、無論文命名**：SGLang in-batch prefix caching、vLLM offloading 不准同 block 同時 onload、Strata 延後、PEEK 先鋒先行、Halo 合併工具呼叫 | (a) 讓部分結果可用，全有或全無變成隊頭阻塞；(d) 跟隨者需求不同；(e) 存領頭者送來的層 | **逐層跟隨**；逾時改走重算 (b)；不要一次全部放出 |
| Delayed hits（Atre et al.；見 os_toolbox B1） | 延遲最優 ≠ 命中率最優；Belatedly（MCMCF） | 物件整體到達 | **Strata**（延後＋bundle hit），整段 context 粒度 | 逐層版沒人做 | Belatedly 當離線上界 |
| Serving 湊批（Clipper〔全文〕、Nexus〔全文〕、BatchMaker〔摘要〕；LLM 端 FlexGen〔全文〕、Hydragen〔摘要〕） | 延後送出等更多查詢（Nagle 式）；early drop；prefix batching；cellular batching；FlexGen 逐行走 | 共享物件（權重）每步重用；固定成本大 | 被 LLM 大量引用但只當背景；「為等共享者延後」NOT_FOUND | 文件 KV 每輪每請求只用一次且依序，中途加入不能用現在這層；decode 例外 | FlexGen＝LLM 自己的「batching under (a)+(e)」 |
| 兩機 flow shop（Johnson 1954〔摘要〕；Gupta et al. 2004〔摘要〕；Crama & Gültekin〔摘要〕；lot streaming〔摘要〕；Garey et al. 1976〔摘要〕；Smith 1956〔標題〕） | Johnson：I/O 輕的在前、計算輕的在後；彈性工序在任一機；lot streaming 子批重疊；WSPT | 兩機獨立、一次一件、時間固定、無共享、離線 | KV NOT_FOUND；最近：CacheFlow（批次 min max(T_comp,T_load)、樓梯形 DP，未提 flow shop）；邊緣 SD 論文用 Johnson 的兩段遞迴 | GPU 可分割且批次次線性；I/O 與計算搶 HBM 頻寬；背景 decode；線上到達且權重會長大；(e) 有限緩衝；(c) 彈性工序成本依切點 | 逐層＝lot streaming；Cake 切分＝可分割彈性工序；權重＝等待人數 → WSPT／AA |

---

## §10 對三向交集（Sharing × Latency Hiding × Scheduling）的意義

以下全是〔判讀〕，是推論，不是任何論文的結論。

### 10.1 最可能變成新 KV 機制的兩個古典機制

**第一名：VoD patching（加上 QPipe 的 WoP 語言與 PBM 的依序保留）→「逐層 patching／逐層搭便車」**

- **機制草案**：文件 D 的逐層還原進行到第 k 層時，同文件的新請求 R2 到達。
  1. **Regime R**（前 k 層仍在 HBM）：R2 立刻在第 0…k−1 層算自己的問題，之後跟在還原游標後面逐層前進。這是 coalescing 的逐層版。
  2. **Regime S**（前 k 層已被丟掉）：R2 加入進行中的串流，在 HBM 存第 k 層以後送來的層（緩衝上限 W 層）。錯過的前 k 層用 patch 補：(i) 另排載入；(ii) Cake 式部分重算；(iii) 若 k 太大或 W 不夠，等下一輪。門檻依「patch 成本＋拖慢先來者的成本」對「新開一輪的成本」決定。
  3. HBM 裡的層依 PBM 式「下一次消耗時間」保留：所有在途請求的位置與速度決定哪一層最快被再用。
  4. 依問題挑頁 (d) 時，用 QPipe 的 materialization：R2 只存自己要的頁。
- **最接近的 LLM 前作**與差別：
  - Strata：delay hit 延後、bundle hit 同批，整段 context 粒度、沒有 patch。
  - vLLM RFC #33526：block 粒度漸進 onload，是 token 方向。
  - CachedAttention：HBM read buffer、依佇列 look-ahead 預取與淘汰，單一請求、session 粒度。
  - FlexGen：權重的 column-by-column 共享。
  - LMCache、HiCache：單請求的逐層管線。
  - **沒有人做「晚到者加入另一個請求進行中的逐層還原、錯過的層另補」。**
- **可能失敗的條件**：
  - **N = λT 太小**：每份文件在一個還原時間 T 內很少有第二個請求，機制幾乎不會被觸發。VoD 的所有節省都隨 N 成長（√N、ln N）。我們的 N 是 **NOT_MEASURED**。若要量，可以從 trace 統計「同一前綴 block 在 T 內的到達數」；匯入 trace 時要依 CLAUDE.md 規則 6 先驗證 block 單位。
  - **HBM 夠大**：文件 KV 能一直留著（例如 H200 級容量對單一 16 GiB 文件），Regime S 不出現，只剩 Regime R，而 Regime R 的增益上限是晚到者在前 k 層的計算時間。問題短時這很小。
  - **I/O 共用**：Regime S 的 patch 與主串流走同一條 PCIe／SSD，patch 拖慢先來者。若沒有閒置頻寬，總延遲可能變差。
  - **compute-bound 時強迫同步就是序列化**（Johnson et al. VLDB'07 的教訓）：Cake 式還原的前段重算佔 GPU 時，再加晚到者的 prefill 會互相拖。
  - **(d) 的頁集合重疊低**時，共享的只有少數頁。

**第二名：shared-scan 排程（Agrawal AA policy）＋ anticipatory 等待 ＋ WSPT／Johnson → 「共享感知的還原排程器」**

- **機制草案**：多份文件與私有請求同時排隊時：
  1. 依 B_i²/(λ_i t_i^s) 或 w/a（WSPT）決定先還原哪份文件。t_i^s 本身由 Cake／CacheFlow 式切點決定，所以是「排序＋切分」的聯合決策。
  2. 佇列只剩一份文件、I/O 會閒時，用 anticipatory 的成本效益式決定要不要等第二個共享者，逾時取到達間隔的高百分位，逾時不重設。
  3. 依 Johnson 的直覺排 I/O 輕的工作在前，讓 GPU 早點有事做。
- **最接近的 LLM 前作**：
  - Strata balanced batch formation：以載入÷計算的比值判斷 loading-bound。
  - RAGCache 重排：已快取÷需計算。
  - SGLang LSPF／DFS（Theorem 3.1）、DLPM（局部性對公平）、PEEK（先鋒先行＋防餓死）。
  - CacheFlow：批次 min max(T_comp, T_load)。
  - **沒有人把「未來到達率 λ」放進共享文件的還原優先權，也沒有人用「等不等共享者」的明確規則。**
- **可能失敗的條件**：
  - λ_i 估不準（熱度變化快）。
  - GPU 不是 unary machine、I/O 與計算互搶頻寬，使「先 I/O 輕」的直覺失準。
  - 目標若是 P95 TTFT，WSPT 類規則可能餓死冷門文件，需要 DLPM／PEEK 式的公平項。

### 10.2 兩者的關係與一句話版本

- 第一名回答「**同一份文件**，晚到的人怎麼搭上進行中的還原」（Sharing × Latency Hiding）。第二名回答「**多份文件與私有請求之間**，誰先還原、要不要等人、切多少給 GPU」（Sharing × Scheduling）。合起來才是三向交集。
- 一句話〔判讀〕：**把逐層還原看成一條「依序播放的影片」，晚到的人用 patching 搭上；把多份文件的還原看成「可分享掃描的排程」，用 Agrawal 式優先權與 anticipatory 逾時決定順序與等待；用 QPipe 的 WoP（spike → step → linear）描述 (a)、(e)、(d) 的角色。**

### 10.3 在投入實作前最便宜的檢查（提案，未執行）

1. 用現有 trace 估 N = λT 的分佈：同一前綴、在一個還原時間窗內的到達數。若中位數 ≪ 1，第一名機制的上檔很小。
2. 列出我們手上硬體 × 模型 × 文件長度的組合，標出哪些落在 Regime R、哪些落在 Regime S（HBM 是否放得下一份文件 KV 加上模型與 decode 狀態）。這是純容量算術，但本檔沒有做。
3. 在模擬器加入 Strata 式整段 delay hit 與逐層版，比較 TTFT 差多少，先確定 Regime R 的增益上限。

---

## 附錄 A　查證紀錄與 NOT_FOUND 清單

### A.1 書目交叉核對

- Crossref（`api.crossref.org/works?query.bibliographic=...`）逐篇核對作者、標題、venue、卷期頁、DOI；結果存於 scratchpad `L3/meta/xref1.txt`。
- DBLP API 回傳 bot 檢查頁（HTTP 202 + "Making sure you're not a bot"），**沒有繞過**。改用 S2 回傳的 DBLP key 作第二來源：
  - `conf/mm/HuaCS98`（Patching）
  - `conf/mm/DanSS94`、`journals/mms/DanSS96`（Batching）
  - `journals/mms/GolubchikLM96`（Adaptive piggybacking）
  - `journals/jal/Bar-NoyL03`（Bar-Noy & Ladner）
  - `conf/vldb/ZukowskiHNB07`（Cooperative Scans）
- 無 DOI 的 USENIX／VLDB 2007 論文以 PDF 首頁的會議資訊核對：Cooperative Scans、To Share or Not to Share 首頁皆印 "VLDB '07, September 23-28, 2007, Vienna"；Clipper 首頁有 NSDI'17 連結；memcache PDF 頁尾為 NSDI '13。

### A.2 「LLM 有沒有用」的搜尋方法

1. **S2 citations API**：抓每篇古典論文全部引用者的標題，以 `LLM|LLMs|language model|KV|key-value cache|transformer|prefill|GPT|generative|attention|token|inference serving|serving` 過濾（腳本 `L3/scripts/s2cites.py`）。掃描數與命中如下：

   | 論文 | S2 引用者數 | LLM／KV 命中 |
   |:--|--:|:--|
   | Patching | 673 | 0（1 筆誤中，2002 串流快取） |
   | Dan'94 | 707 | 0 |
   | Dan'96 | 154 | 0 |
   | Eager'01 | 279 | 0 |
   | Golubchik'96 | 68 | 0 |
   | Aggarwal'96 piggyback | 125 | 0（1 筆誤中，2001 near-VoD） |
   | Bar-Noy–Ladner | 38 | 0 |
   | QPipe | 224 | 0 |
   | CJOIN | 105 | 0 |
   | Crescando | 127 | 0 |
   | SharedDB | 171 | 0 |
   | Psaroudakis | 56 | 0 |
   | Cooperative Scans | 149 | 0 |
   | Anticipatory | 289 | 0 |
   | Atre（delayed hits） | 67 | Strata（另 3 筆非 KV） |
   | Agrawal'08 | 93 | 0 |
   | PBM'12 | 12 | 0 |
   | Vattani'15 | 1 | 0 |
   | Gupta'04 | 20 | 0 |
   | Crama–Gültekin | 14 | 0 |

   S2 中途多次 HTTP 429；Johnson'54、Smith'56、Garey'76、Potts–Baker'89 的 S2 掃描沒有完成，改用 OpenAlex。
2. **OpenAlex**：`filter=cites:W…,title_and_abstract.search:(language model) OR LLM OR (KV cache) OR transformer OR (key-value cache)`（腳本 `L3/scripts/oa.py`，結果 `L3/meta/oa_cites.txt`）。這個搜尋很吵（"language" 會命中正規語言等），命中後逐筆人工看標題：
   - Johnson'54：14 筆，其中 LLM 推論 2 筆（邊緣 SD）。
   - Garey'76：18 筆，其中 LLM serving 1 筆（Hetis，用途 NOT_VERIFIED）。
   - Sellis'88：28 筆，其中 LLM 2 筆（Helium；ICDE'25 的圖文 MQO，非 KV）。
   - BatchMaker：12 筆，皆 serving 背景。
   - 其餘古典論文 0–3 筆，皆非 LLM。
3. **LLM 論文全文 grep**（30 篇，PDF 存於 `L3/pdfs/`）：
   - 用完整詞表 grep 的 22 篇：AlignedServe、BatchLLM、Hydragen、RAGCache、CachedAttention、FlexGen、Preble、DLPM、Parrot、SGLang、KVFlow、PTStore、"When Fancy Eviction Fails"、Cake、LMCache、BlendServe、CacheBlend、InferCept、Pensieve、ChunkAttention、Continuum、CacheFlow。
   - 只用部分詞表 grep 的 8 篇（針對該篇可能相關的詞）：LLM-SQL（2403.05821；MQO、shared scan、work sharing 等）、Strata（delayed hit、Atre）、KVPR（Johnson、flow shop、makespan）、Helium 與 Halo（MQO、Sellis、SharedDB、QPipe、Johnson）、邊緣 SD（Johnson、flow shop）、PEEK 與 Feather（參考文獻中的古典 DB／OS 文獻）。
   - grep 詞：anticipatory、Druschel、ski rental、Johnson's rule、flow shop、patching、piggyback、video-on-demand、cooperative／shared／circular scan、QPipe、SharedDB、CJOIN、Crescando、multi-query optimization、singleflight、request coalescing、thundering herd、stampede、delayed hit、Atre、Nagle、work sharing、Clipper、Nexus、BatchMaker、cellular batching。
   - 命中只有：Strata（Atre）；Halo（SharedDB、coalescing）；Helium（Sellis）；邊緣 SD（Johnson）；Parrot、Pensieve（Clipper、Nexus、BatchMaker）；BatchLLM、ChunkAttention（cellular batching）；以及 "piggybacking"（全為 Sarathi 標題）、"patching"（全為 "dispatching"）的誤中。
4. **web 搜尋**（extended）：
   - "video-on-demand" + KV cache + batching／patching
   - "cooperative scans" OR "shared scans" + LLM
   - Johnson's rule／flow shop + LLM／KV
   - anticipatory scheduling + LLM KV
   - delayed hits + LLM KV
   - singleflight／request coalescing + LLM prefix cache
   - batching restore same document + KV + wait window
   - LLM 端只找到本檔已列的 vLLM RFC、Continuum、CacheFlow、KVPR、PEEK、Feather；其餘 NOT_FOUND。

### A.3 NOT_VERIFIED／未取得全文清單

- **Hua et al. 1998（Patching）全文**：ACM DL 付費；UCF STARS 頁面寫 "not available"。機制細節改依 Eager et al. 的全文轉述。
- **Dan et al. 1994／1996 全文**：UFRJ 的 OCR 版回 404。只用 IBM 出版頁的摘要。
- **Aggarwal et al. MFQ 的權重公式**、**Bar-Noy–Ladner 的 competitive ratio**：只有摘要或二手資料。
- **Golubchik et al. 1995／1996 全文**：CUHK 的 web archive 連結回 HTML。只用二手摘要與 Eager et al. 的轉述。
- **Johnson 1954 原文的規則措辭**、**Smith 1956 內容**：只核到摘要（Johnson）或書目（Smith）。WSPT 與 Johnson's rule 的陳述是教科書常識，未對原文頁碼。
- **Gonzalez & Sahni 1978 在兩機可搶佔下的結論**：摘要只說一般情況 NP-complete，兩機特例未核。
- **Potts & Baker 1989 全文**：只有 IAOR 轉述。
- **BatchMaker 全文**：作者頁連結失效，只用摘要。
- **CachedAttention、Strata、CacheFlow、SGLang、FlexGen 的正式發表處**：本檔只引用 arXiv 版本號，正式 venue 本次未核對（repo 其他文件有收錄）。
- **Linux anticipatory I/O scheduler 的版本史**：本檔沒有寫，因為沒核對。

---

## 附錄 B　參考書目（IEEE 格式；證據等級見正文）

**VoD**
1. A. Dan, D. Sitaram, P. Shahabuddin, "Scheduling policies for an on-demand video server with batching," in *Proc. ACM Multimedia '94*, pp. 15–23, 1994. doi:10.1145/192593.192614
2. A. Dan, D. Sitaram, P. Shahabuddin, "Dynamic batching policies for an on-demand video server," *Multimedia Systems*, vol. 4, no. 3, pp. 112–121, 1996. doi:10.1007/s005300050016
3. C. C. Aggarwal, J. L. Wolf, P. S. Yu, "On optimal batching policies for video-on-demand storage servers," in *Proc. IEEE ICMCS*, pp. 253–258, 1996. doi:10.1109/MMCS.1996.534983
4. K. A. Hua, Y. Cai, S. Sheu, "Patching: A multicast technique for true video-on-demand services," in *Proc. ACM Multimedia '98*, pp. 191–200, 1998. doi:10.1145/290747.290771
5. L. Gao, D. Towsley, "Supplying instantaneous video-on-demand services using controlled multicast," in *Proc. IEEE ICMCS*, vol. 2, pp. 117–121, 1999. doi:10.1109/MMCS.1999.778179
6. L. Golubchik, J. C. S. Lui, R. R. Muntz, "Adaptive piggybacking: A novel technique for data sharing in video-on-demand storage servers," *Multimedia Systems*, vol. 4, no. 3, pp. 140–155, 1996. doi:10.1007/s005300050019
7. L. Golubchik, J. C. S. Lui, R. Muntz, "Reducing I/O demand in video-on-demand storage servers," in *Proc. ACM SIGMETRICS*, pp. 25–36, 1995. doi:10.1145/223587.223590
8. C. C. Aggarwal, J. L. Wolf, P. S. Yu, "On optimal piggyback merging policies for video-on-demand systems," in *Proc. ACM SIGMETRICS*, pp. 200–209, 1996. doi:10.1145/233013.233044
9. D. Eager, M. Vernon, J. Zahorjan, "Minimizing bandwidth requirements for on-demand data delivery," *IEEE Trans. Knowl. Data Eng.*, vol. 13, no. 5, pp. 742–757, 2001. doi:10.1109/69.956098（全文：作者預印本 <https://www.cs.usask.ca/ftp/pub/discus/recent_2002/TKDEpaper.pdf>）
10. D. L. Eager, M. K. Vernon, J. Zahorjan, "Bandwidth skimming: A technique for cost-effective video-on-demand," in *Proc. SPIE MMCN*, vol. 3969, pp. 206–215, 2000. doi:10.1117/12.373523
11. A. Bar-Noy, R. E. Ladner, "Competitive on-line stream merging algorithms for media-on-demand," *J. Algorithms*, vol. 48, no. 1, pp. 59–90, 2003. doi:10.1016/S0196-6774(03)00045-2
12. K. A. Hua, S. Sheu, "Skyscraper broadcasting," in *Proc. ACM SIGCOMM '97*, pp. 89–100, 1997. doi:10.1145/263105.263144

**DBMS**

13. M. Zukowski, S. Héman, N. Nes, P. Boncz, "Cooperative scans: Dynamic bandwidth sharing in a DBMS," in *Proc. VLDB*, 2007.（DBLP key conf/vldb/ZukowskiHNB07；全文 <https://ir.cwi.nl/pub/12225/12225B.pdf>）
14. M. Świtakowski, P. Boncz, M. Zukowski, "From cooperative scans to predictive buffer management," *PVLDB*, vol. 5, no. 12, pp. 1759–1770, 2012. doi:10.14778/2367502.2367515
15. S. Harizopoulos, V. Shkapenyuk, A. Ailamaki, "QPipe: A simultaneously pipelined relational query engine," in *Proc. ACM SIGMOD*, pp. 383–394, 2005. doi:10.1145/1066157.1066201
16. G. Candea, N. Polyzotis, R. Vingralek, "A scalable, predictable join operator for highly concurrent data warehouses," *PVLDB*, vol. 2, no. 1, pp. 277–288, 2009. doi:10.14778/1687627.1687659
17. P. Unterbrunner, G. Giannikis, G. Alonso, D. Fauser, D. Kossmann, "Predictable performance for unpredictable workloads," *PVLDB*, vol. 2, no. 1, pp. 706–717, 2009. doi:10.14778/1687627.1687707
18. G. Giannikis, G. Alonso, D. Kossmann, "SharedDB: Killing one thousand queries with one stone," *PVLDB*, vol. 5, no. 6, pp. 526–537, 2012. doi:10.14778/2168651.2168654
19. C. A. Lang, B. Bhattacharjee, T. Malkemus, S. Padmanabhan, K. Wong, "Increasing buffer-locality for multiple relational table scans through grouping and throttling," in *Proc. IEEE ICDE*, pp. 1136–1145, 2007. doi:10.1109/ICDE.2007.368972
20. P. Agrawal, D. Kifer, C. Olston, "Scheduling shared scans of large data files," *PVLDB*, vol. 1, no. 1, pp. 958–969, 2008. doi:10.14778/1453856.1453960
21. R. Johnson, N. Hardavellas, I. Pandis, N. Mancheril, S. Harizopoulos, K. Sabirli, A. Ailamaki, B. Falsafi, "To share or not to share?," in *Proc. VLDB*, 2007.（全文 <https://www.vldb.org/conf/2007/papers/research/p351-johnson.pdf>）
22. I. Psaroudakis, M. Athanassoulis, A. Ailamaki, "Sharing data and work across concurrent analytical queries," *PVLDB*, vol. 6, no. 9, pp. 637–648, 2013. doi:10.14778/2536360.2536364
23. T. K. Sellis, "Multiple-query optimization," *ACM Trans. Database Syst.*, vol. 13, no. 1, pp. 23–52, 1988. doi:10.1145/42201.42203
24. PostgreSQL Global Development Group, "Server configuration: Version and platform compatibility — `synchronize_seqscans`," PostgreSQL documentation (current). <https://www.postgresql.org/docs/current/runtime-config-compatible.html>

**OS／快取**

25. S. Iyer, P. Druschel, "Anticipatory scheduling: A disk scheduling framework to overcome deceptive idleness in synchronous I/O," in *Proc. ACM SOSP*, pp. 117–130, 2001. doi:10.1145/502034.502046（全文：作者 "to appear" 版 <https://people.mpi-sws.org/~druschel/publications/nwcs.pdf>）
26. R. Nishtala et al., "Scaling Memcache at Facebook," in *Proc. USENIX NSDI*, 2013.
27. A. Vattani, F. Chierichetti, K. Lowenstein, "Optimal probabilistic cache stampede prevention," *PVLDB*, vol. 8, no. 8, pp. 886–897, 2015. doi:10.14778/2757807.2757813
28. nginx, "Module ngx_http_proxy_module — `proxy_cache_lock`," <https://nginx.org/en/docs/http/ngx_http_proxy_module.html>
29. Varnish Cache, "Grace mode and keep," <https://www.varnish.org/docs/users-guide/vcl-grace/>
30. Go Authors, "Package singleflight," <https://pkg.go.dev/golang.org/x/sync/singleflight>
31. N. Atre, J. Sherry, W. Wang, D. S. Berger, "Caching with delayed hits," in *Proc. ACM SIGCOMM*, 2020. doi:10.1145/3387514.3405883（已於 os_toolbox B1 全文核對）

**Serving**

32. D. Crankshaw, X. Wang, G. Zhou, M. J. Franklin, J. E. Gonzalez, I. Stoica, "Clipper: A low-latency online prediction serving system," in *Proc. USENIX NSDI*, 2017.
33. H. Shen et al., "Nexus: A GPU cluster engine for accelerating DNN-based video analysis," in *Proc. ACM SOSP*, pp. 322–337, 2019. doi:10.1145/3341301.3359658
34. P. Gao, L. Yu, Y. Wu, J. Li, "Low latency RNN inference with cellular batching," in *Proc. EuroSys*, 2018. doi:10.1145/3190508.3190541

**排程理論**

35. S. M. Johnson, "Optimal two- and three-stage production schedules with setup times included," *Naval Res. Logist. Q.*, vol. 1, no. 1, pp. 61–68, 1954. doi:10.1002/nav.3800010110
36. W. E. Smith, "Various optimizers for single-stage production," *Naval Res. Logist. Q.*, vol. 3, no. 1–2, pp. 59–66, 1956. doi:10.1002/nav.3800030106
37. M. R. Garey, D. S. Johnson, R. Sethi, "The complexity of flowshop and jobshop scheduling," *Math. Oper. Res.*, vol. 1, no. 2, pp. 117–129, 1976. doi:10.1287/moor.1.2.117
38. T. Gonzalez, S. Sahni, "Flowshop and jobshop schedules: Complexity and approximation," *Oper. Res.*, vol. 26, no. 1, pp. 36–52, 1978. doi:10.1287/opre.26.1.36
39. J. N. D. Gupta, C. P. Koulamas, G. J. Kyparisis, C. N. Potts, V. A. Strusevich, "Scheduling three-operation jobs in a two-machine flow shop to minimize makespan," *Ann. Oper. Res.*, vol. 129, pp. 171–185, 2004. doi:10.1023/B:ANOR.0000030687.72169.c7
40. Y. Crama, H. Gültekin, "Throughput optimization in two-machine flowshops with flexible operations," *J. Scheduling*, vol. 13, no. 3, pp. 227–243, 2010. doi:10.1007/s10951-008-0096-2
41. C. N. Potts, K. R. Baker, "Flow shop scheduling with lot streaming," *Oper. Res. Lett.*, vol. 8, no. 6, pp. 297–303, 1989. doi:10.1016/0167-6377(89)90013-8
42. D. Trietsch, K. R. Baker, "Basic techniques for lot streaming," *Oper. Res.*, vol. 41, no. 6, pp. 1065–1076, 1993. doi:10.1287/opre.41.6.1065
43. V. Bharadwaj, D. Ghose, T. G. Robertazzi, "Divisible load theory: A new paradigm for load scheduling in distributed systems," *Cluster Computing*, vol. 6, no. 1, pp. 7–17, 2003. doi:10.1023/A:1020958815308

**LLM／KV（本檔引用的版本）**

44. Z. Xie et al., "Strata: Hierarchical context caching for long context language model serving," arXiv:2508.18572v1, 2025.
45. S. Nian et al., "CacheFlow: Efficient LLM serving via automated 3D-parallel KV cache restoration," arXiv:2604.25080v2, 2026.
46. Y. Sheng et al., "FlexGen: High-throughput generative inference of large language models with a single GPU," arXiv:2303.06865v2, 2023.
47. B. Gao et al., "Cost-efficient large language model serving for multi-turn conversations with CachedAttention," arXiv:2403.19708v3, 2024.
48. S. Jin, X. Liu, Q. Zhang, Z. M. Mao, "Compute or load KV cache? Why not both?," arXiv:2410.03065v2, 2025.
49. C. Jiang et al., "KVPR: Efficient LLM inference with I/O-aware KV cache partial recomputation," arXiv:2411.17089v2, 2025.
50. L. Zheng et al., "SGLang: Efficient execution of structured language model programs," arXiv:2312.07104v2, 2024；SGLang 原始碼 `schedule_policy.py`（commit 8252e3a, 2026-10-25）與 `hicache_design.mdx`。
51. vLLM, "[RFC]: Progressive KV cache CPU onloading," GitHub issue #33526, 2026. <https://github.com/vllm-project/vllm/issues/33526>
52. H. Li et al., "Continuum: Efficient and robust multi-turn LLM agent scheduling with KV cache time-to-live," arXiv:2511.02230v7, 2026.
53. C. Jin et al., "RAGCache: Efficient knowledge caching for retrieval-augmented generation," arXiv:2404.12457v2, 2024.
54. S. Cao et al., "Locality-aware fair scheduling in LLM serving," arXiv:2501.14312v1, 2025.
55. J. Juravsky et al., "Hydragen: High-throughput LLM inference with shared prefixes," arXiv:2402.05099v2, 2024.
56. B. Xie et al., "PEEK: Predictive queue-informed KV cache management for LLM serving," arXiv:2607.02525v2, 2026.
57. S. Rathi, Preeti, M. Vutukuru, "Requests of a feather must flock together: Batch size vs. prefix homogeneity in LLM inference," arXiv:2605.06046v3, 2026.
58. J. Shen, N. Wadlom, Y. Lu, "Batch query processing and optimization for agentic workflows" (Halo), arXiv:2509.02121v2, 2026.
59. N. Wadlom, J. Shen, Y. Lu, "Efficient LLM serving for agentic workflows: A data systems perspective" (Helium), arXiv:2603.16104v1, 2026.
60. B. Zhu et al., "Efficient LLM inference over heterogeneous edge networks with speculative decoding," arXiv:2510.11331v1, 2025.
