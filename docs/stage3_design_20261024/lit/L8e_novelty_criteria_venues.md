# L8e — 系統研究的「貢獻／新穎性」判準，與 negative measurement 論文的投稿場地

- **日期**：2026-10-24
- **執行者**：Claude（subagent；只讀網頁，只寫本檔，未改任何 repo 檔案）
- **怎麼查的**：約 50 次 WebSearch／WebFetch／curl。**優先讀官方 CFP 頁**：用 curl 抓 HTML，轉成純文字後逐段讀；PDF 用 `pdftotext -layout`（Roscoe 2007 審稿筆記、ICPE'24 p178）；論文摘要走 arXiv API。第三方彙整站（myhuiban、PaperPilot、beri.net、clocate、conferenceindex）**只當線索、不當證據**；凡是只在這類網站出現的日期，一律標 `NOT_POSTED`。
- **引用方式**：受版權規則限制，本檔對原文以「**中文意譯＋精確出處（URL＋段落名）**」為主，逐字引文只留一處（ISPASS 的主題條目名）。track 名稱、論文類型名稱（例如 EERCS、Operational Systems track）屬於專有名稱，直接保留英文。要看原文措辭，請點對應 URL。
- **審查性質**：**self-review，非 cross-model**。沒有找第二個模型或第二個人覆核。
- **時區**：AoE = Anywhere on Earth（UTC−12）。

---

## 0. 結論

### 0.1 貢獻類型 rubric（7 類；細節見 §1）

| # | 類型 | 一句話判準 | 對應導師的三條準則 |
|---|---|---|---|
| R1 | 新觀察／新問題 | 量到一個沒人量過、而且**推翻某個具名假設**的現象 | 準則 (1) |
| R2 | 新機制 | 讓原本的 trade-off **消失**，而不是在曲線上換一個點 | 準則 (2) |
| R3 | 有預測力的模型／理論 | 事前就能說出「何時 A 會贏 B」，並且被量測驗證 | （與 (1) 互補） |
| R4 | 新量測／特徵化 | 在新平台或新資料上做系統性量測，並給出可以重複使用的 insight | （與 (1) 互補） |
| R5 | Negative result（含解釋與替代方案） | 公平地測試一個合理的想法，證明它不成立，**解釋為什麼**，再**給出替代方案** | 準則 (1) 的一種 |
| R6 | 新抽象 | 換一種問題表述，讓舊問題變簡單 | 準則 (3) |
| R7 | 工具／benchmark／trace／artifact | 判準是能不能讓後續研究變得可行或放大 | — |

### 0.2 ICPE 2027 適不適合 path 1 的 negative measurement 論文？

**判定：適合，但有條件。**

適合的理由：
- **ICPE 2027 的主題清單明確列出 LLM 量測與 LLM 服務的 caching。**「Measurement and Empirical Evaluation」底下有「AI/ML 系統、LLM inference pipeline 與 agentic workflow 的量測與評估」；「Managing Systems at Runtime」底下有「ML/AI inference 與 LLM serving 的 batching、routing、caching 的執行期管理」。來源：https://icpe2027.spec.org/call-for-contributions/
- **Research Track 有專門給實證研究的論文類型。**它把投稿分成 Regular 與 **EERCS（Empirical / Experience / Reproduction / Case Study）** 兩種。EERCS 指的是以實證評估、實際經驗、重現或案例研究為主的論文，可以只描述「套用既有方法所得的新經驗或 insight」。ICPE 2026 也有同樣的分類，可見這個類型是穩定的。來源：https://icpe2027.spec.org/tracks-and-submissions/research-paper-track/ 、https://icpe2026.spec.org/tracks-and-submissions/research-paper-track/
- **ICPE Research Track 近年接受過帶有 negative 成分的 characterization 論文。**
  - ICPE'24：Tensor Core 對 GCN 幾乎沒有幫助，有些實作甚至變慢。
  - ICPE'25：公開 LLM 服務的中斷事件特徵化。
  - ICPE'26：推翻「能耗與輸入／輸出長度呈線性關係」的 SweetSpot 模型。

  細節見 §4。

需要小心的條件：
- **審查準則仍然列出 novelty。** Research Track 的審查準則是 novelty、relevance、technical quality、scientific soundness、clarity。CFP 也說 Research Track 收的是 novel research contributions。所以「沒有新方法」可以，但「沒有新知識」不行。
- **「歡迎 negative」的明文只限於重現研究。** Research Track 頁面只寫「正面與負面的**重現**研究都歡迎」。一般性的 negative results，只有 **Emerging Research Track** 明文歡迎（≤6 頁，2027-02-01 截止）。來源同上，以及 https://icpe2027.spec.org/tracks-and-submissions/emerging-research-track/
- **投稿時若要選 EERCS，該怎麼宣告並沒有說清楚。** CFP 沒寫是勾選欄位還是標題前綴，記為 `NOT_STATED`。建議在 HotCRP 表單上確認，或寫信問 PC chairs（Valeria Cardellini、Pooyan Jamshidi）。

### 0.3 要撐過「no new method」這句審稿意見，論文必須具備的內容

這份清單由 §1 的 canonical 來源加上 §4 的接受範例歸納而來：

1. **具名的被推翻假設（對應 R1）。** 要寫出「誰、在哪幾個系統裡，假設寫入時依 token 位置決定 KV 層級／精度是有利的」，並附上引用。沒有具名的假設，就構不成「推翻」，只是「試了一個點子，結果沒用」。
2. **證明是公平的測試。** 要交代 pre-registered 門檻、強 baseline、調參預算，以及 MI300X 與 H200 兩個平台。這是用來擋「你沒有調好」的質疑，參照 Heiser 的 benchmarking crimes，見 §5。
3. **機制層級的解釋：為什麼不行。** WDDD 明確要求 negative result 要附上「為什麼不 work」的解釋；Levin & Redell 也要求說清楚結論所依賴的假設。範例論文的做法是 roofline、root-cause 分解這類分析。
4. **替代方案與它的證據（對應 R5）。** 你的替代方案是「延後放置、讀取時再選還原方式」，要量出它的收益或上界。§4 裡被接受的範例幾乎都是「負面觀察→原因→可用的替代或工具」三段式。
5. **一條有預測力的判準（R5 升級成 R3）。** 也就是「在什麼條件下（例如成本比、重用分佈），寫入時決策才會贏」。這是對「no new method」最強的回應：新貢獻是一個**可以被證偽的判準**，而不是一個機制。SweetSpot（ICPE'26）就是這個形狀。
6. **適用範圍與敏感度。** Levin & Redell 在 Lessons／Context 兩節要求：明說結論依賴哪些假設、對擾動有多敏感，並且不要從單一例子推廣。
7. **Artifact。** ICPE 強烈鼓勵投 Artifact Evaluation Track（目前細節 TBA）；SOSP 2026 甚至要求 Data Availability 段落。
8. **一句話貢獻不能是「我們調了 X 的 policy」。** 參考骨架（不含數字）：「在兩種加速器上，依位置在寫入時決定 KV 的層級與精度，並不能帶來可量測的收益；收益在讀取時才出現。我們給出判斷寫入時決策何時划算的準則。」

### 0.4 備案場地（遵守「不可同時投稿」的規則）

ICPE Research Track 在 **2027-01-25** 通知。同一篇論文在 ICPE 審稿期間**不能**同時投以下場地：
- ISPASS（截止 12/18）
- CCGrid（12/1）
- OSDI（12/8）
- SIGMETRICS Winter（2027-01-11）
- MLSys（2026-10-30）

依據是各 CFP 的同時投稿禁令，見 §3。所以真正可行的「ICPE 被拒之後」備案如下：

| 優先 | 場地 | 日期（官方） | 說明 |
|---|---|---|---|
| 1 | **ICPE 2027 Emerging Research Track** | paper **2027-02-01** AoE | ≤6 頁，明文歡迎 negative results，由 Research Track 的 PC 審稿。被 RT 拒的稿子能不能轉投 ERT，CFP 沒有寫（`NOT_STATED`），需要先問 chairs |
| 2 | **Euro-Par 2027**（荷蘭 Groningen，2027-08-23–27） | CFP **NOT_POSTED** | 2026 年的模式是 abstract 2026-03-06、paper 2026-03-13（延長後），≤14 頁 LNCS（含參考文獻），single-blind。符合使用者偏好的荷蘭 |
| 3 | **IISWC 2027** | **NOT_POSTED** | 2026 年的模式是 abstract 2026-05-14、paper 2026-05-21，正文 ≤10 頁不含參考文獻，主題就是 workload characterization |
| （改主投） | **ISPASS 2027** | abstract **2026-12-11**、paper **2026-12-18** AoE | 如果決定**不投 ICPE**，這是更對口的選擇：ISPASS 的主題清單明列 **"Confirmations or refutations of important prior results"**，而且截止日比 ICPE 晚一個月。只能二選一 |

HotOS XXI（2027-02-01，≤5 頁 position paper）比較適合 stage 3（path 2）的新方向，不適合 path 1。

---

## 1. Q1 — canonical 判準與 rubric

### 1.1 Rubric 表

| 類型 | 定義 | Canonical 出處（意譯） | 論文裡怎麼證明 |
|---|---|---|---|
| **R1 新觀察／新問題** | 量到一個新現象，而它推翻或動搖一個廣泛採用的假設 | **SOSP 2025／2026 CFP**（Topics 段）：偏好開拓新領域、延續重要研究對話、或反思最先進實作的經驗或量測的工作。**SOSP CFP**（Submission Criteria 段）：處理新問題的論文，評估標準可以不同。**OSDI '27 CFP**：好論文要先 motivate 一個重要問題。**Heilmeier Q2**：現在怎麼做、現行做法的極限在哪 | 具名的假設＋引用；說明量測方法為什麼可信；說明現象的普遍性（跨平台、跨模型）；寫出「如果這個假設錯了，誰的設計要改」 |
| **R2 新機制** | 提出並實作一個解法，讓 trade-off 消失，或嚴格優於既有做法 | **OSDI '27**（Submitting a Paper）：提出有趣且有說服力的解法、證明實用性與收益、清楚說明超越近期與歷史工作之處。**SOSP**：要「提出**並實作**」解法。**EuroSys 2027**：要用嚴謹評估呈現解法的**好處與限制**。**Levin & Redell**（Original Ideas）：明顯的延伸通常不值得發表 | 證明原本的兩個極端都被支配；baseline 是當前最強；同時報告限制 |
| **R3 有預測力的模型／理論** | 模型能事前預測結果，而且被驗證 | **Levin & Redell**（Context）：提出形式模型時，要問它有沒有帶來新資訊與 insight；一個深刻的定理勝過一千個定義。**SIGMETRICS 2027**（Measurement & Applied Modeling track）：提出並應用新模型或模擬工具來捕捉真實現象。**ICPE 2027** 的 Modeling 主題：explanatory 與 predictive models | 事前寫下預測，再跑量測核對；報告誤差（例如 MAPE）；說明模型在哪裡失效 |
| **R4 新量測／特徵化** | 在新平台、新資料或新 workload 上做系統性量測，並萃取出可以重複使用的 insight | **SIGMETRICS 2027**（M&AM track）：透過新資料來源、新實驗或對既有資料的新分析，得出重要 insight。**ISPASS**（首頁）：量測、特徵化、建模、模擬的主要論壇。**IISWC 2026**：特徵化與分析各類系統 workload。**SOSP**：對最先進實作的量測 | 方法學要夠嚴謹，見 §5 Heiser；要有可以操作的 insight，而不只是一堆圖；要公開資料或 artifact |
| **R5 Negative result（含解釋與替代）** | 公平地測試一個合理的想法，證實它不成立；解釋原因；指出替代方案 | **Levin & Redell**（Original Ideas）：對先前未經證實的想法做延伸、**驗證或否定（repudiation）**，都是合格的貢獻；支持**或反駁**既有設計的實作經驗極有價值。**ISPASS 2027 主題**："Confirmations or refutations of important prior results"。**OSDI '26／'27 Operational Systems track**：推翻或強化既有假設、加深對既有問題的理解（不要求新點子，但限於 operational systems）。**WDDD**（ISCA workshop）：負面結果要附上為什麼不 work 的解釋 | 預先登記的門檻；公平性證據；root-cause；替代方案的收益或上界；這個結果能讓哪些後續研究**不用再走冤枉路** |
| **R6 新抽象** | 換一種問題表述，讓舊問題變簡單 | 這是導師的準則 (3)。**我在讀過的頂會 CFP 中沒有找到把「新抽象」單獨列為貢獻類型的句子**。最接近的是 HotOS XXI CFP：為長期問題倡議創新做法、開啟一條研究線；以及 MLSys 2027 主題「ML programming models and abstractions」 | 證明在新抽象下，舊問題的某個難點（例如 NP-hard、需要預知未來）消失或降階；至少給兩個以上受益的實例 |
| **R7 工具／benchmark／trace** | 新工具、benchmark、資料集或 trace | **ISPASS 2027**（Tool and benchmark papers）：主要評的是能否促成、放大未來研究，接受後要開源。**IISWC 2026**：有 tool and benchmark 類別。**ATC 2026**：附帶前所未有的 artifact（trace、原始資料、原始碼、工具）的投稿評價更高。**ICPE 2027**：有 Artifact track | 開源；寫出至少一個「沒有它就做不到」的後續分析 |

### 1.2 各 canonical 來源的重點（意譯＋出處）

**(a) Levin & Redell，"An Evaluation of the Ninth SOSP Submissions -or- How (and How Not) to Write a Good Systems Paper"，ACM SIGOPS OSR 17(3)，1983 年 7 月，pp. 35–40。** 兩位作者是 SOSP-9 的 PC co-chairs；當屆 83 篇投稿接受 16 篇。全文：https://www.usenix.org/legacy/events/samples/submit/advice_old.html （USENIX 經作者授權轉載）
- **三類論文**：真實系統、未實作但點子值得知道的系統、理論（含 performance modelling）。
- **Original Ideas 節**：
  - 論文至少要有一個新點子，而且能用一段話講清楚，這段話就拿去當 abstract。
  - 要說明問題為什麼不能用已發表的技術解決。
  - 明顯的延伸不值得發表。
  - 對先前未經證實想法的**延伸、驗證或否定**，是合格的貢獻；支持或反駁既有設計的實作經驗**極有價值**。
- **Lessons 節**：你學到什麼？可推廣到多廣？結論依賴哪些假設要寫清楚；避免從單一例子推廣。
- **Choices 節**：記錄考慮過的替代方案，讓後人不必再走同樣的死路。這正是 negative result 的價值所在。
- **Context 節**：假設是否切合現實？結論對假設的擾動有多敏感？形式模型要帶來新的 insight。

**(b) OSDI '26／'27 CFP。** https://www.usenix.org/conference/osdi26/call-for-papers 、https://www.usenix.org/conference/osdi27/call-for-papers
- 概述：OSDI 強調創新研究，以及經過量化或具洞見的系統設計與實作經驗。
- 審查準則：novelty、significance、interest、clarity、relevance、correctness。
- 「excellent paper」清單：motivate 重要問題、提出有說服力的解法、證明實用性與收益、結論恰當、清楚說明貢獻、說明超越近期與歷史工作之處。
- OSDI '27 新增：把系統技術套用到 ML 等新領域的論文，必須對上系統領域的既有工作、拿已知技術當 baseline，並說清楚新的系統挑戰。
- **Operational Systems track**：
  - 對象是營運中系統的設計、實作、分析與經驗。**不要求新的研究點子**，但要擴充公眾知識，例如推翻或強化既有假設、加深對既有問題的理解、在新規模或新環境驗證已知技術。
  - OSDI '26 版本還寫明：以前會投 USENIX ATC 的稿子，也歡迎來這個 track。
- 我數過 CFP 全文的關鍵字：**OSDI '26／'27 都沒有出現 "negative" 或 "measurement"**。
- OSDI '27 日期：abstract 2026-12-01、paper 2026-12-08（5:59 pm EST）；投稿 ≤12 頁，不含參考文獻。

**(c) SOSP 2025／2026 CFP。** https://sigops.org/s/conferences/sosp/2025/cfp.html 、https://sigops.org/s/conferences/sosp/2026/cfp.html
- 審查準則同樣是那六項。
- Topics 段：依 SOSP 傳統，偏好開拓新領域、延續重要研究對話，或**反思最先進實作的經驗或量測**的工作。
- 好論文清單要求「提出**並實作**」解法。
- 鼓勵開創性的新方向，並承認處理新問題的論文評估標準可以不同。
- SOSP 2026 新增 Data Availability 段落：寫明 artifact 能重現哪些 claim、不能重現哪些。
- 全文沒有出現 "negative"。
- 兩屆截止都已過：SOSP 2025 是 2025-04-17；SOSP 2026 是 2026-04-01 AoE。

**(d) EuroSys 2027 CFP。** https://2027.eurosys.org/cfp.html
- 審查準則同樣六項。好的投稿要用嚴謹評估呈現解法的**好處與限制**，以及相對既有工作的優勢。
- **歡迎 experience papers**：描述實際部署中有趣的觀察或 lessons learned，前提是這些 lessons 具一般性、有嚴謹的量化分析，而且對系統社群有用。
- **查證結果：EuroSys 2027 沒有獨立的 negative-results track**；CFP 全文出現 "negative" 0 次。所謂 experience 類型，限定在「實際部署」。
- 兩個 cycle 都已截止：spring 是 2026-05-14，fall 是 2026-09-24。2028.eurosys.org 連不上，記為 `NOT_POSTED`。

**(e) Heilmeier Catechism（DARPA 官方頁，8 題；美國政府著作）。** https://www.darpa.mil/about/heilmeier-catechism
1. 你想做什麼？完全不用術語講出目標。
2. 現在怎麼做？現行做法的極限在哪？
3. 你的做法新在哪？為什麼認為會成功？
4. 誰在乎？成功了會有什麼不同？
5. 風險是什麼？
6. 要花多少錢？
7. 要多久？
8. 期中與期末用什麼「考試」檢驗成功？

### 1.3 導師三個寫作前問題 ↔ canonical 來源

| 導師的問題 | 對應來源 | 對 path 1 的含意 |
|---|---|---|
| 是否建立在量到的觀察上？ | Levin & Redell「Reality」節：論文描述的東西是否真的實作過；「Context」節：假設是否切合現實。Heilmeier Q2 | path 1 天生滿足；要確保觀察是**跨平台**的，而且有方法學上的可信度 |
| 是消除 trade-off，還是只在 trade-off 上挑一個點？ | Levin & Redell「Original Ideas」節：明顯延伸不值得發表、要清楚區分與既有工作的差異。OSDI '27：說明超越近期與歷史工作之處 | path 1 是 R5；要升級成 R1／R3，必須具名被推翻的假設，並給出一條「何時寫入時決策會贏」的判準 |
| 能不能用一句話講清楚，而且不是「我們調了 X 的 policy」？ | Levin & Redell：新點子要能用一段話寫下，拿去當 abstract。Heilmeier Q1：不用術語講目標 | 見 §0.3 第 8 點的骨架句 |

### 1.4 補充：其他頂會的「不要求新點子」通道都限定部署系統

以下 track 都寫「不要求新點子」，但適用對象有限制：
- **OSDI Operational Systems track**：營運中的系統。
- **FAST '27 deployed-systems papers**：真實營運或線上 production 的系統；不需要新點子，但要對實務者有用；鼓勵推翻或強化既有假設，以 production 資料優先。https://www.usenix.org/conference/fast27/call-for-papers
- **SIGMETRICS 2027 Operational Systems Track**：production 規模、實際大量使用的系統；可以證明或推翻既有假設；必須寫 lessons learned。https://www.sigmetrics.org/sigmetrics2027/pages/cfp.html
- **MLSys 2027 industrial track**：第一作者與多數作者必須來自業界。https://mlsys.org/Conferences/2027/CallForResearchPapers
- **ATC 2026 operational systems track**：同樣是營運系統。https://atc.sigops.org/2026/cfp.html

**含意**：實驗室 testbed 上的 negative measurement **進不了這些通道**，只能走 research track。走 research track 就必須拿「insight／判準／模型」當 novelty。

---

## 2. Q2 — ICPE 2027 CFP 事實

所有資訊都直接讀自官方頁；讀取日期 2026-10-24。

| 項目 | 內容 | 來源 |
|---|---|---|
| 會議 | ACM/SPEC ICPE 2027，2027-05-24–28，瑞典 Gothenburg | https://www.spec.org/notes/2026/news260904/ ；地點也見 https://icpe2027.spec.org/location/ |
| Research Track 日期（AoE） | abstract **2026-11-09（一）**；paper **2026-11-16（一）**；通知 2027-01-25；camera-ready 2027-03-12 | https://icpe2027.spec.org/important-dates/ |
| 範圍 | 收 novel research contributions，主題依 Call for Contributions | https://icpe2027.spec.org/tracks-and-submissions/research-paper-track/ |
| 論文類型 | ① Regular：原創研究，要有有說服力的評估。② **EERCS（Empirical / Experience / Reproduction / Case Study）**：以實證評估、實際經驗、重現或案例研究為主，可以描述套用既有方法所得的新經驗或 insight；**正面與負面的重現研究都歡迎** | 同上 |
| 「negative results」措辭 | Research Track 只在**重現研究**的脈絡出現。明文歡迎一般 negative results 的是 **Emerging Research Track** | 同上；ERT 頁見下 |
| 審查準則 | novelty、relevance、technical quality、scientific soundness、clarity；每篇至少 3 位 RT PC 審 | Research Track 頁 |
| 頁數 | ACM 雙欄；**≤10 頁**（含圖表，不含參考文獻與附錄）；附錄**不會出版**，所以正文要能獨立成立；camera-ready 可以多 2 頁，但不能加入大量未審內容 | Research Track 頁 |
| 匿名 | **double-anonymous**：不能有作者名；自己的舊作用第三人稱引用；程式碼或 repo 要匿名；與 PC 的所有溝通都經過 PC chairs | Research Track 頁 |
| 事前公開 | 可以先放 arXiv、做 invited talk、投無正式論文集的 workshop，但仍須遵守匿名規則 | Research Track 頁 |
| Artifact | 強烈鼓勵 RT 被接受的論文投 **Artifact Evaluation Track**，可取得 ACM badges。AE Track 頁目前是 **Details TBA**（chairs：Lishan Yang、Gabriele Russo Russo） | Research Track 頁；https://icpe2027.spec.org/tracks-and-submissions/artifact-evaluation-track/ |
| 主題（與本研究直接相關的條目） | Measurement and Empirical Evaluation：受控實驗設計、性能除錯；**AI/ML 系統、LLM inference pipeline 的量測與評估**。Modeling：explanatory／predictive models；跨 workload 與環境的模型驗證與泛化；cost／energy／performance trade-off 的模型。Managing Systems at Runtime：**LLM serving 的 batching、routing、caching 的執行期管理**。Platform：加速器、GPU 與**新型記憶體系統**的效能；latency 與 throughput 的 trade-off。Benchmarking：可重現性、可重複性、可重用的 artifact | https://icpe2027.spec.org/call-for-contributions/ |
| 投稿系統 | HotCRP：https://icpe2027.hotcrp.com | Research Track 頁 |
| Program chairs | Valeria Cardellini（Univ. of Rome Tor Vergata）、Pooyan Jamshidi（Univ. of South Carolina） | Research Track 頁 |
| 費用 | ACM 100% open access。作者機構若不在 ACM Open 名單，2027 年 APC 為 **USD 500（ACM／SIG 會員）／USD 750（非會員）**；需要 ORCID | https://icpe2027.spec.org/call-for-contributions/ |
| **Emerging Research Track** | 兩種類型：Position／Vision／Idea，以及 Work-in-Progress，都是 **≤6 頁**，標題要加後綴；double-anonymized；由 Research Track 的 PC 依各自的 rubric 審；收進 companion proceedings；**明文歡迎能讓社群學到東西的 negative results**；截止 **2027-02-01**，通知 03-05 | https://icpe2027.spec.org/tracks-and-submissions/emerging-research-track/ ；日期見 important-dates |
| Industry Track | 與 RT 同日截止；**不採 double-blind**；以業界實務者為對象，評審標準是「工業應用的原創性」。CFP 明說：research、theory、experience 類論文**不論作者來自哪裡**，都應該投 Research Track | https://icpe2027.spec.org/tracks-and-submissions/industry-track/ |
| Data Challenge Track | **Details TBA**；日期 TBA（chairs：David Bermbach、Wes Lloyd） | https://icpe2027.spec.org/tracks-and-submissions/data-challenge-track/ |
| Posters／Demos | 截止 2027-02-08 | important-dates |
| 觀察（非 CFP 明文） | ICPE 2026 的接受名單裡，Research Track 有標「(Short paper)」的論文，但 2027 CFP **沒有提到 short paper**，記為 `NOT_STATED` | https://icpe2026.spec.org/accepted-papers/ |

---

## 3. Q3 — 其他歡迎 characterization／measurement／negative 的場地

讀取日期都是 2026-10-24。「衝突」指與 ICPE RT 的審稿期（2026-11-16 → 2027-01-25）重疊、不能同時投同一篇。各場地的同時投稿禁令都寫在 CFP 裡：
- SIGMETRICS：明寫這是 ACM 政策。
- EuroSys、MLSys：明文禁止。
- ISPASS 投稿說明：不得在他處審稿中。
- CCGrid：不得在他處審稿中。

| 場地 | 截止（官方，AoE 除非另註） | 頁數 | 與 measurement／negative 相關的範圍（意譯） | 衝突 | 來源 |
|---|---|---|---|---|---|
| **ISPASS 2027**（2027-05-09–11，NYU） | abstract **2026-12-11**；paper **2026-12-18**；rebuttal 2027-02-15–19；通知 03-08 | 正文 ≤**9 頁** IEEE 雙欄，參考文獻不限；double-blind；必須親自出席發表 | 首頁自稱是量測、特徵化、建模、模擬的主要論壇；主題含 ML 訓練與推論 workload、加速器與 GPU、記憶體系統，以及 **"Confirmations or refutations of important prior results"**；另有 tool／benchmark 論文類別 | 是 | https://ispass.org/2027/cfp/ 、https://ispass.org/2027/dates/ 、https://ispass.org/2027/submission/ |
| **IISWC 2026** | 已過：paper 2026-05-21 | regular ≤10 頁不含參考文獻；tool／benchmark 可以 6 頁 | 各類系統 workload 的特徵化與分析，含面向商用 production 系統的轉譯研究 | — | https://iiswc.org/iiswc2026/cfp.html |
| **IISWC 2027** | **NOT_POSTED**（iiswc.org/iiswc2027/ 回 404） | — | 依 2026 模式，約在 5 月 | 否（推估在 ICPE 通知之後） | 同上 |
| **SIGMETRICS 2027**（2027-06-07–11，Atlanta，FCRC） | Fall：abstract 2026-10-02、paper 10-09（已過）。**Winter：abstract 2027-01-04、paper 2027-01-11**；通知 03-10 | ≤20 頁 acmsmall 單欄，不含參考文獻；double-anonymous | **Measurement & Applied Modeling track**：從新資料或實驗中得出重要 insight，或提出新模型捕捉真實現象。Operational Systems track 限 production 系統。被拒的論文 12 個月內不能再投 SIGMETRICS | 是（01-11 早於 01-25） | https://www.sigmetrics.org/sigmetrics2027/pages/cfp.html |
| **MLSys 2027**（2027-06-21–25，Bellevue） | paper **2026-10-30 20:00 UTC**；通知 2027-02-28 | ≤10 頁雙欄，不含參考文獻；double-blind | 主題含高效推論與 serving、LLM inference、大規模 ML 的儲存系統、ML benchmark。industrial track 不要求新點子，但需業界作者過半 | 是，而且只剩 6 天 | https://mlsys.org/Conferences/2027/CallForResearchPapers 、https://mlsys.org/ |
| **EuroMLSys 2027**（EuroSys workshop） | **NOT_POSTED**（euromlsys.eu 仍是 2026 版） | 2026 版：≤6 頁，single-blind | 2026 版的截止是 2026-02-15，延到 02-24。2026 接受過「與一般假設不符」的 trace 特徵化論文，見 §4 | 否（推估） | https://euromlsys.eu/ |
| **HotOS XXI**（2027-05-24–26，德國 Burghausen） | paper **2027-02-01**（hard deadline） | ≤**5 頁**，參考文獻另計；double-blind；ACM acmart | 收 position papers：提出新研究方向、為長期問題倡議創新做法，或從真實系統經驗得到深刻 insight。偏好能開啟一條研究線的早期工作 | 否 | https://sigops.org/s/conferences/hotos/2027/cfp.html |
| **CCGrid 2027**（2027-05-24–27，Dallas–Fort Worth） | abstract **2026-11-24**；paper **2026-12-01**；通知 2027-02-01 | ≤**10 頁含參考文獻** IEEE；double-blind；camera-ready 可以付費加 2 頁 | Track 5（Performance Modeling, Analysis, Optimization）：benchmark、workload 特徵化、profiling、瓶頸分析，**要提供可推廣的效能 insight**。Track 3／4：AI／LLM 系統，含 caching 與 state | 是 | https://easychair.org/cfp/ccgrid27 （官網 hpcclab.org/ccgrid27 回 406，讀不到） |
| **ICDCS 2027** | **NOT_POSTED** | — | 官方網站找不到或連不上；第三方說是 Melbourne、2027-07，未證實 | — | （無官方 URL） |
| **Euro-Par 2027**（2027-08-23–27，Groningen） | **NOT_POSTED**（官網已上線，但 calls 頁回 404） | Format 頁：regular 建議 12 頁、上限 14 頁 | 官網提到效能工程與可重現性。2026 版是 single-blind、≤14 頁 LNCS（含參考文獻），截止 2026-03-13（延長後） | 否（推估在 3 月） | https://2027.euro-par.org/ 、https://2027.euro-par.org/about-euro-par/format 、https://2026.euro-par.org/calls/papers |
| **USENIX ATC → ACM SIGOPS ATC** | ATC 2026（香港，2026-11-16–18）截止是 2026-06-10。**ATC 2027：NOT_POSTED**（atc.sigops.org/2027/ 回 404） | ATC 2026：long ≤12 頁、short ≤6 頁；第一輪先審 2 頁 extended abstract | 範圍含效能評估與 workload 特徵化；operational systems track 不要求新點子 | — | https://atc.sigops.org/2026/cfp.html |
| **FAST '27** | spring 2026-03-17、fall 2026-09-15（都已過） | long ≤12、short ≤6 | 主題含 workload characterization、deployed storage 的實證評估。deployed-systems 論文不需要新點子 | — | https://www.usenix.org/conference/fast27/call-for-papers |
| **FAST '28** | **NOT_POSTED**（404） | — | — | — | — |
| **OSDI '27** | abstract 2026-12-01；paper 2026-12-08（EST） | ≤12 頁 | Operational Systems track，見 §1.2(b) | 是 | https://www.usenix.org/conference/osdi27/call-for-papers |
| **EuroSys 2027／2028** | 2027 兩個 cycle 都已過；**2028：NOT_POSTED** | ≤12 頁 | experience papers，見 §1.2(d) | — | https://2027.eurosys.org/cfp.html |
| **PerFail 2027**（PerCom workshop，2027-03-08–12，印度 Goa） | paper **2026-11-17** | 2027 頁面上**沒讀到頁數**，記為 `NOT_VERIFIED` | 專門收 pervasive computing、networking 與 systems 的 negative results | 是（同一份內容不能同投） | https://perfail-workshop.github.io/2027/ |

---

## 4. Q4 — 2023–2026 被接受的量測型／負面型 LLM inference 論文範例

「為何被接受」只依各論文摘要的自述來意譯，**不是**我對審稿結果的推測。

| # | 場地／年 | 論文 | URL | 類型 | 摘要自述的主要貢獻（意譯） | 借鏡 |
|---|---|---|---|---|---|---|
| 1 | **ICPE 2024** Research Track | Hanindhito & John, *Accelerating ML Workloads using GPU Tensor Cores: The Good, the Bad, and the Ugly* | https://doi.org/10.1145/3629526.3653835 ；PDF：https://research.spec.org/icpe_proceedings/2024/proceedings/p178.pdf ；收錄：https://icpe2024.spec.org/accepted-papers/ | R4＋R5 | 在 H100 與 A100 上量測：CNN 用 Tensor Core 可快 1.91×（TF32）與 2.42×（FP16）；GCN 頂多 1.03×（FP16），**部分實作甚至變慢**。用 roofline 與微基準解釋原因；是最早一批第三方 H100 量測 | **負面結果配上 roofline 與微基準的解釋**，就能在 ICPE RT 被接受；第三方的新硬體量測本身也算貢獻 |
| 2 | **ICPE 2025** Research Track | Chu, Talluri, Lu, Iosup, *An Empirical Characterization of Outages and Incidents in Public Services for Large Language Models* | https://arxiv.org/abs/2501.12469 （journal-ref：ICPE 2025）；收錄：https://icpe2025.spec.org/accepted-papers/ | R4＋R7 | 指出這個新興領域研究很少；收集 3 家供應商、8 個 LLM 服務的資料；提出 10 多條 observation；資料與程式碼依 FAIR 原則公開 | 新資料、多條可以檢驗的 observation、開放 artifact |
| 3 | **ICPE 2026** Research Track | Pizzini Cavagna et al., *SweetSpot: An Analytical Model for Predicting Energy Efficiency of LLM Inference* | https://arxiv.org/abs/2602.05695 ；收錄：https://icpe2026.spec.org/accepted-papers/ | R1＋R3 | 指出既有做法假設能耗是輸入／輸出長度的線性函數，並證明實際上非線性、存在最佳點；從 Transformer 複雜度推出解析模型；在 H100 上跨 1B–9B 模型，平均 MAPE 1.79% | **推翻具名假設＋有預測力的模型**，是 path 1 升級的模板 |
| 4 | **MLSys 2025** | Gao, Zhou, Sun, Zhang, Wen, *Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving* | https://arxiv.org/abs/2503.24000 （comment：published to MLSys2025）；搜尋命中的 https://mlsys.org/virtual/2025/3236 我沒有打開 | R4＋R5＋R7 | 從 production 角度重新檢視 KV cache 壓縮：既有 benchmark 漏了某些量測面向；現行 kernel 沒有針對 production serving 優化，所以 throughput 不理想；壓縮可能讓輸出變長，反而拉高端到端延遲；逐樣本分析找出壓縮會失效的任務。提供工具並開源 | **與你最接近**：「理論上省記憶體，實際上不一定更快」的 KV 論文被 MLSys 接受，三根支柱是補上缺漏的量測、給出原因、提供工具 |
| 5 | **ISPASS 2025** | *Characterizing and Optimizing LLM Inference Workloads on CPU-GPU Coupled Architectures* | https://arxiv.org/abs/2504.11750 （comment：Accepted for ISPASS 2025）；收錄：https://ispass.org/ispass2025/accepted-papers.php （#26） | R4（反直覺觀察）＋R7（新指標） | GH200 在大 batch 時 prefill 快 1.9–2.7×，但在比鬆耦合系統大 4 倍的 batch 範圍內仍然受 CPU 限制；新指標 TKLQT 能定位 CPU-bound 與 GPU-bound 的轉折點 | **反直覺現象＋一個能定位轉折點的指標**，形狀類似「何時寫入時決策會贏」的判準 |
| 6 | **ISPASS 2026** | *TaxBreak: Unmasking the Hidden Costs of LLM Inference Through Overhead Decomposition* | https://arxiv.org/abs/2603.12465 （comment：Accepted at IEEE ISPASS 2026）；議程：https://ispass.org/ispass2026/program.php | R4＋R7 | 指出既有做法只把 host 端開銷當作一個總殘差，無法判斷該優化哪一層；提出 trace-driven 的三段分解，在 H100 與 H200 上驗證，並推出 HDBI 指標 | H200 上的量測方法學論文，可以被 ISPASS 接受 |
| 7 | **SIGMETRICS 2026** | *Characterizing Performance-Energy Trade-offs of Large Language Models in Multi-Request Workflows* | 收錄：https://www.sigmetrics.org/sigmetrics2026/accepted.html ；摘要：https://arxiv.org/abs/2604.09611 | R4（含局部負面） | 第一份多請求 LLM workflow 的性能與能耗系統性特徵化。batch size 影響最大，但效益依 workload 而定：對循序摘要**無效**，對多 agent coding 只部分有效。最後給出可操作的 guideline | 「某個旋鈕在某類 workload 無效」可以寫成 guideline；這也說明 SIGMETRICS 會收這類論文 |
| 8 | **EuroMLSys 2026**（EuroSys workshop） | Karatay Demiray et al., *Where the Time Goes: Analysis of a Public LLM Serving System* | https://euromlsys.eu/ （2026 接受名單與摘要） | R1＋R4＋R7 | 分析約 5 個月、33.7 萬個請求的 trace。發現有幾點**與一般假設不符**：TTFT 常由排隊而不是 prefill 決定；8B 與 70B 的使用者感知延遲幾乎相同；輸入特徵很難預測輸出長度。trace 將會公開 | 用「與一般假設不符」直接定位成 R1 |
| 9 | **EuroMLSys 2026** | Xin et al., *Why Smaller Is Slower? Dimensional Misalignment in Compressed LLMs* | https://arxiv.org/abs/2604.09595 ；收錄見 https://euromlsys.eu/ | R5→R2 | 壓縮後參數變少卻沒有變快：ASVD 壓縮的 Llama-3-8B 少 15% 參數，但不比原版快，因為 95% 的維度沒有對齊。從 framework、library、hardware 三層做 root cause 分析，並提出 GAC | **negative → root cause → 小而通用的修正**，正好是 path 1 銜接 path 2 的橋 |
| 10 | **IISWC 2024**（信心較低） | Na et al., *Understanding Performance Implications of LLM Inference on CPUs* | 議程：https://iiswc.org/iiswc2025/program.html （頁內含 2024 Session 4）；https://ieeexplore.ieee.org/document/10763564 | R4 | 依搜尋摘要：把權重與 KV cache 卸載到 CPU 記憶體時，PCIe 會造成顯著退化；改為分析在新一代 CPU 上直接推論的表現。**IEEE 全文頁我沒有打開**，只讀到搜尋片段 | 與 KV offload 主題相鄰 |
| 附 | **USENIX ATC 2025**（不在題目指定的清單內，但與 KV 直接相關） | Wang et al., *KVCache Cache in the Wild: Characterizing and Optimizing KVCache Cache at a Large Cloud Provider* | https://www.usenix.org/conference/atc25/presentation/wang-jiahao | R4→R2 | 第一份大型供應商 KV$ workload 的系統性特徵化，提出合成 workload 研究沒有涵蓋的觀察：重用分佈偏斜、單輪請求的重用同樣重要、在單一類別內可以預測；再提出 workload-aware 的 eviction policy | 特徵化再推出一個小機制，是 KV 領域的範本 |

**跨範例的共同形狀**（歸納，非來源明文）：被接受的量測型論文幾乎都同時具備：
- 一個具名的缺口或假設；
- 多平台或真實資料；
- 機制層級的解釋；
- 一個可以重複使用的東西（工具、指標、模型或 trace）；
- 可以操作的 guidance。

我**沒有看到**只寫「X 沒用」而缺少後三項的例子。但要注意：我只看得到被接受的論文，這是存活者偏誤。

---

## 5. Q5 — 關於發表 negative results 的建議

1. **Levin & Redell（1983，SOSP-9 PC co-chairs）** https://www.usenix.org/legacy/events/samples/submit/advice_old.html
   - 對先前想法做**驗證或否定**，是合格的貢獻；支持或反駁既有設計的實作經驗極有價值。
   - 記錄考慮過的替代方案，讓後人避開死路。
   - 寫清楚結論依賴的假設，以及結論對假設擾動的敏感度。
   - **對你的含意**：negative 論文要寫成「替社群排除一條路，並說明在哪些假設下成立」。

2. **Timothy Roscoe，"Writing reviews for systems conferences"（2007-03，寫給 SOSP 2007 Shadow PC）** https://people.inf.ethz.ch/troscoe/pubs/review-writing.pdf （第 3 頁的 Specific comments 清單）
   - 審稿人會問：新在哪裡？相關工作會不會讓這個貢獻失效？會不會引發有趣的討論？聽眾能不能帶走一些可以拿去用的點子，而不只是「扎實但平淡」？與 CFP 是否相符？
   - 同一份筆記也提醒：作者沒有寫清楚貢獻是常見錯誤。
   - **對你的含意**：negative 論文最大的風險是被歸成「扎實但平淡」，所以第 5 點的判準（R3）是讓審稿人有東西可帶走的關鍵。

3. **WDDD — Workshop on Duplicating, Deconstructing, and Debunking（ISCA workshop；第 6 屆在 2007 年 ISCA-34）** https://pharm.ece.wisc.edu/wddd/2007/index.html
   - 這是給系統研究社群發表「驗證或重複先前結果、拆解、揭穿」工作的論壇。
   - 主題包括：對直覺上應該可行的點子提出負面結果，**並解釋為何不行**；以清楚理由反駁先前發現；附有意義分析的獨立驗證。
   - 理由：讓後人學到別人付出代價得來的教訓，而不必重做。
   - **對你的含意**：沒有解釋的負面結果，連專門收負面結果的 workshop 都不收。

4. **PerFail 2027 — 6th Workshop on Negative Results in Pervasive Computing（與 IEEE PerCom 2027 合辦）** https://perfail-workshop.github.io/2027/
   - 對 negative result 的定義：研究做得正確、符合良好實務，但沒能證明假設，或結果不顯著。
   - 明確**排除**程式錯誤或 bug 這類「非根本因素」，因為它們無法說明方法或假設本身好不好。
   - 論文必須突顯 lessons learned；範圍含 networking 與 systems。截止 2026-11-17。
   - **對你的含意**：要先證明負面結果「不是 bug」，再談學到什麼。

5. **Gernot Heiser，"Systems Benchmarking Crimes"** https://gernot-heiser.org/benchmarking-crimes.html
   - 列出系統量測的常見罪狀：選擇性 benchmark（他稱為「所有罪狀之母」）、只跑子集、選擇性資料集、不當 baseline、只報相對數字、不做顯著性檢定、用算術平均等。
   - **對你的含意**：negative 結論最容易被「你挑了不利的設定」或「baseline 太弱」推翻；逐條自查可以預先擋掉。

6. **CFP 層級的共識（非部落格，但代表 PC 立場）**
   - 支持 negative 的 track：
     - ICPE 2027 ERT 明文歡迎有教訓的 negative results。
     - ISPASS 2027 主題列入「對重要先前結果的證實或反駁」。
     - OSDI、FAST、SIGMETRICS、ATC 的 operational 或 deployed track 都寫「推翻或強化既有假設」可以取代新點子，但**限定在部署系統**。
   - 研究型 track（OSDI、SOSP、EuroSys）仍以「問題＋解法＋評估」為主軸。

7. **沒找到的東西（如實記錄）**
   - 我搜過 SIGARCH Computer Architecture Today、SIGOPS、PL Perspectives，**沒有找到**系統領域 PC chair 專門談「如何發表 negative results」的部落格文章。
   - 這只代表「我沒查到」，不代表「不存在」（見 CLAUDE.md §1 第 7 條）。
   - 另外，Feitelson 2015 年發表在 SIGOPS OSR 的 *From repeatability to reproducibility and corroboration*，作者網站上的 PDF 抓下來是 HTML，**沒讀到全文，不列為證據**。

---

## 6. References

讀取日期都是 2026-10-24。狀態：OK = 已讀內容；404／NOT_POSTED = 官方頁不存在或尚未公布；SNIPPET = 只讀到搜尋片段；THIRD-PARTY = 只當線索。

| # | URL | 用途 | 狀態 |
|---|---|---|---|
| 1 | https://icpe2027.spec.org/tracks-and-submissions/research-paper-track/ | ICPE RT 類型、頁數、匿名、artifact、審查準則 | OK |
| 2 | https://icpe2027.spec.org/call-for-contributions/ | ICPE 範圍、主題、APC | OK |
| 3 | https://icpe2027.spec.org/important-dates/ | ICPE 日期 | OK |
| 4 | https://icpe2027.spec.org/tracks-and-submissions/emerging-research-track/ | ERT、negative results | OK |
| 5 | https://icpe2027.spec.org/tracks-and-submissions/industry-track/ | Industry track | OK |
| 6 | https://icpe2027.spec.org/tracks-and-submissions/data-challenge-track/ | Data Challenge | OK（Details TBA） |
| 7 | https://icpe2027.spec.org/tracks-and-submissions/artifact-evaluation-track/ | AE Track | OK（Details TBA） |
| 8 | https://www.spec.org/notes/2026/news260904/ | ICPE 2027 會期與地點 | OK |
| 9 | https://icpe2026.spec.org/tracks-and-submissions/research-paper-track/ | 2026 版 EERCS 對照 | OK |
| 10 | https://icpe2026.spec.org/accepted-papers/ 、https://icpe2025.spec.org/accepted-papers/ 、https://icpe2024.spec.org/accepted-papers/ | ICPE 接受名單 | OK |
| 11 | https://www.usenix.org/legacy/events/samples/submit/advice_old.html | Levin & Redell 1983 全文 | OK |
| 12 | https://www.darpa.mil/about/heilmeier-catechism | Heilmeier 8 問 | OK |
| 13 | https://www.usenix.org/conference/osdi26/call-for-papers | OSDI '26 | OK |
| 14 | https://www.usenix.org/conference/osdi27/call-for-papers | OSDI '27 | OK |
| 15 | https://sigops.org/s/conferences/sosp/2025/cfp.html | SOSP 2025 | OK |
| 16 | https://sigops.org/s/conferences/sosp/2026/cfp.html | SOSP 2026 | OK |
| 17 | https://2027.eurosys.org/cfp.html 、https://2027.eurosys.org/ | EuroSys 2027 | OK |
| 18 | https://ispass.org/ 、https://ispass.org/2027/cfp/ 、https://ispass.org/2027/dates/ 、https://ispass.org/2027/submission/ | ISPASS 2027 | OK（投稿系統尚未開放） |
| 19 | https://iiswc.org/iiswc2026/cfp.html 、https://iiswc.org/iiswc2026/ | IISWC 2026 | OK |
| 20 | https://iiswc.org/iiswc2027/ | IISWC 2027 | 404 → NOT_POSTED |
| 21 | https://www.sigmetrics.org/sigmetrics2027/pages/cfp.html 、https://www.sigmetrics.org/sigmetrics2027/ | SIGMETRICS 2027 | OK |
| 22 | https://www.sigmetrics.org/sigmetrics2026/accepted.html | SIGMETRICS 2026 接受名單 | OK |
| 23 | https://mlsys.org/Conferences/2027/CallForResearchPapers 、https://mlsys.org/ | MLSys 2027 | OK（另一個 CallForPapers 頁寫「尚未公布」，但 CallForResearchPapers 頁已公布） |
| 24 | https://euromlsys.eu/ | EuroMLSys 2026 版、範例 8 與 9 | OK（2027 版 NOT_POSTED） |
| 25 | https://sigops.org/s/conferences/hotos/2027/ 、https://sigops.org/s/conferences/hotos/2027/cfp.html | HotOS XXI | OK |
| 26 | https://easychair.org/cfp/ccgrid27 | CCGrid 2027 CFP | OK（官網 hpcclab.org/ccgrid27 回 406） |
| 27 | https://2027.euro-par.org/ 、https://2027.euro-par.org/about-euro-par/format | Euro-Par 2027 | OK（CFP NOT_POSTED） |
| 28 | https://2026.euro-par.org/calls/papers | Euro-Par 2026 模式 | OK |
| 29 | https://atc.sigops.org/2026/cfp.html | ATC 2026 | OK |
| 30 | https://atc.sigops.org/2027/ | ATC 2027 | 404 → NOT_POSTED |
| 31 | https://www.usenix.org/conference/fast27/call-for-papers | FAST '27 | OK |
| 32 | https://www.usenix.org/conference/fast28/call-for-papers | FAST '28 | 404 → NOT_POSTED |
| 33 | https://www.clocate.com/ieee-international-conference-on-distributed-computing-systems-icdcs/27485/ | ICDCS 2027（Melbourne 等資訊） | THIRD-PARTY，未採信 → NOT_POSTED |
| 34 | https://research.spec.org/icpe_proceedings/2024/proceedings/p178.pdf | 範例 1 全文首頁 | OK（pdftotext） |
| 35 | https://arxiv.org/abs/2501.12469 | 範例 2 | OK（arXiv API 摘要） |
| 36 | https://arxiv.org/abs/2602.05695 | 範例 3 | OK |
| 37 | https://arxiv.org/abs/2503.24000 | 範例 4 | OK |
| 38 | https://arxiv.org/abs/2504.11750 、https://ispass.org/ispass2025/accepted-papers.php | 範例 5 | OK |
| 39 | https://arxiv.org/abs/2603.12465 、https://ispass.org/ispass2026/program.php | 範例 6 | OK |
| 40 | https://arxiv.org/abs/2604.09611 | 範例 7 | OK |
| 41 | https://arxiv.org/abs/2604.09595 | 範例 9 | OK |
| 42 | https://iiswc.org/iiswc2025/program.html 、https://ieeexplore.ieee.org/document/10763564 | 範例 10 | 議程 OK；摘要 SNIPPET |
| 43 | https://www.usenix.org/conference/atc25/presentation/wang-jiahao | 附例（KVCache in the Wild） | OK |
| 44 | https://people.inf.ethz.ch/troscoe/pubs/review-writing.pdf | Roscoe 2007 | OK（pdftotext） |
| 45 | https://pharm.ece.wisc.edu/wddd/2007/index.html | WDDD | OK |
| 46 | https://perfail-workshop.github.io/2027/ | PerFail 2027 | OK |
| 47 | https://gernot-heiser.org/benchmarking-crimes.html | Heiser benchmarking crimes | OK |
| 48 | https://www.cse.huji.ac.il/~feit/papers/Repeat15SIGOPS.pdf | Feitelson 2015 | 抓到的是 HTML，未讀，不採用 |
