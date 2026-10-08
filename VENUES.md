---
type: report
created: 2026-08-30
updated: 2026-10-08
tags: [paper, venues, deadlines, kv-cache, europe, usa, australia]
related_ideas: ["[[idea-20260828-hierarchical-kv-state]]"]
---

# 投稿場所調查（查證日：2026-08-30；2026-10-08 重新查證）

對應 idea：[[idea-20260828-hierarchical-kv-state]]／論文 `main.tex`

> **前提**：本論文目前**沒有任何實驗結果**。所有建議都以「Oracle + 消融 (B) 何時能跑完」為前提。

---

## 🔄 2026-10-08 重新查證（本節優先於下方 8/30 的內容）

**一句話**：第一篇不再以 ISC 為首選，改成兩條路擇一：
- **重地點（北歐）與時間 → 路線 W**：11 月做出 6 頁稿，投 1–2 月截稿的 workshop，例如 ICPE workshop（🇸🇪 Gothenburg）或 HotOS（德奧邊境）。**依使用者 10/8 的偏好，預設走這條**，見第十節。
- **重等級與退路 → 路線 F**：補到 10 頁，投 **CCGrid 2027（12/1 截止，Dallas）**，見第三節。

範圍擴大到歐、美、澳的主要好處是每個學期都有被拒後的退路；真正的瓶頸仍是**論文數量**，不是場所。

**前提（10/8）**：碩一；國科會研究生補助**每年度一次**；目標是**每學期出國一次**；同時在準備雅思。
研究進度：老師 10/7 指示的兩週驗證（約 10/21 結束），新方向是寫入時依位置放置 vs 寫回 + 成本感知逐出。

✅ = 本次從官方頁面查證　🔶 = 推估或只有第三方來源，**投稿前必須自行確認**

### 一、8/30 版本的錯誤與過時之處

| 項目 | 8/30 寫的 | 10/8 查證 |
|---|---|---|
| **ISC 2027 會期** | 6/23–25 | ✅ **6/7–11**（論文場次 6/8–10）。摘要 12/14、全文 12/21（no extension）、rebuttal 3/1、通知 3/15、camera-ready 4/8 |
| ISC 是否收 AI 主題 | 待確認 | ✅ 有 **Artificial Intelligence** 類別，明列 "Efficient AI – Low-Precision Formats, Sparsity, and Compression" 與 "Foundation Models – Training and Inference at Scale"；系統類明列 "Memory Technologies and Hierarchies" |
| ISC poster 頁的 6/8–10 | 疑為未更新 | 其實 **6/8–10 才是對的**，錯的是 8/30 寫的 6/23–25。✅ Research Poster 截止 **1/20**、通知 3/2；workshop 日為 **6/11**、tutorial 日為 6/7 |
| ICPE 2027 | 🔶 ~11/17、通知 ~1/19 | ✅ 摘要 **11/9**、全文 **11/16**、通知 **1/25**、camera-ready 3/12；會期 5/24–28 |
| CCGrid 2027 | 12/8、通知 2/10 | ✅ 摘要 **11/24**、全文 **12/1**、通知 **2/1**（EasyChair 官方 CFP） |
| PDP 2027 | 🔶 ~10–11 月 | ✅ 摘要 10/19、全文 10/26、8 頁。**實驗跑不完，不考慮** |
| EuroSys 2027 秋輪 | 9/24 | ❌ 已過 |
| **ICPE 與 CVPR 同一天** | — | ✅ CVPR 2027 全文也是 **11/16**（Seattle，約 6/20–25）。同一份工作只能投其中一個 |

### 二、碩一下能投的場所（歐、美、澳合併）

「10/21 後剩」是從兩週驗證結束起算，可以拿來寫論文的天數。
等級欄來自第三方網站 myhuiban（官方 ICORE 頁面連線回 403，無法直接查）；CVPR 為一般共識；ISC 的 CORE C 來自 ICORE 查詢結果摘要。

| 場所 | 地點 | 會期 | 截止 | 通知 | CCF / CORE | 10/21 後剩 |
|---|---|---|---|---|---|---|
| CVPR 2027 | 🇺🇸 Seattle | 約 6/20–25 | ✅ 11/16 | 🔶 約 2/25 | A / — | 26 天 |
| ICPE 2027 | 🇸🇪 Gothenburg | 5/24–28 | ✅ 11/16 | ✅ 1/25 | — / B | 26 天 |
| **CCGrid 2027** | 🇺🇸 **Dallas** | 5/24–27 | ✅ **12/1** | ✅ **2/1** | C / B | **41 天** |
| ISC 2027 | 🇩🇪 Hamburg | 6/7–11 | ✅ 12/21 | ✅ 3/15 | — / C | 61 天 |
| SIGMETRICS 2027（冬季輪） | 🇺🇸 Atlanta（FCRC） | 6/7–11 | ✅ 1/11（摘要 1/4） | ✅ 3/10 | B / **A*** | 82 天 |
| HotOS 2027（workshop，5 頁） | 🇩🇪 Burghausen | 5/24–26 | ✅ **2/1** | ✅ 4/9 | B / A | 103 天 |
| ICDCS 2027 | 🇦🇺 **Melbourne** | 7/5–8 | 🔶 未公布 | — | B / A | — |
| Euro-Par 2027 | 🇳🇱 Groningen | 8/23–27 | 🔶 約 3 月中（前兩屆 3/17、3/13） | — | B / B | — |
| EuroMLSys 2027（workshop） | 🇲🇦 Rabat | 約 4/19 | 🔶 CFP 未出 | — | — | — |

已經排除的：IPDPS 2027（Bellevue，6/1–5）摘要 10/1、全文約 10/8，已過；MLSys 2027 10/30，太趕。

### 三、路線 F：為什麼正式論文首選 CCGrid

- **主題最對口。** CFP 有 Track 4 "Systems for LLM Applications and Agentic AI"，明列 LLM serving、scheduling、placement；Track 5 是 "Performance Modeling, Analysis, and Optimization"。
- **時間和退路平衡最好。** 寫作時間比 ICPE 多 15 天；2/1 就通知，比 ISC 早一個半月。被拒後還能接 ICDCS（墨爾本 7 月）或 Euro-Par（8 月）。
- **等級和 ICPE 同為 CORE B，高於 ISC 的 C。**
- 規格：10 頁（**含**參考文獻）、IEEE 格式、雙盲；CFP 沒寫 rebuttal 階段。

ISC 只贏在「在歐洲」和「多 20 天」。只投 ISC 的話，3/15 被拒碩一下就出不去了。

SIGMETRICS 留給第二篇：它是 A*，要 20 頁單欄，被拒之後 12 個月內不能再投 SIGMETRICS；但它是量測與建模的場所，正好適合「預測準則」那篇。
它有「Resubmit」結果（大修後可投後續三個截止日之一），而且論文刊在 POMACS 期刊。

### 四、國科會規則補充（10/8 查證）

| 規則 | 對你的影響 |
|---|---|
| 研究生**每年度一次**；合著論文每篇只補助一位研究生 | 碩士期間可用的只剩 **2027、2028 兩次**。2026 那次因為會議前置時間（截稿到開會約 5–6 個月）本來就用不到 |
| 論文須為**首次發表** | 同一份工作不能先發 A 會議，再拿去 B 會議當海報申請第二次 |
| **申請及出席時都要具在學身分** | 碩二下的會議要在畢業前開 |
| 學生端最遲在**開會前兩個月的月底**送出；接受證明可補送，最晚在會議首日四週前 | CCGrid（5/24）約 3/31 前送；ISC（6/7）約 4/30 前送 |
| 機票原則上限搭**本國籍班機**，不能搭時要另填申請書；補助屬部分補助 | 去美、澳之前先問系辦補助上限 |

**校內管道**：陽明交大有自己的「獎助研究生出席國際會議」。115 年度的收件已在 9/18 截止，116 年度的時程要問研發處。

### 五、每學期出國的最大化排法

每學期出國 = 3 趟（碩一下、碩二上、碩二下）= **3 篇不同貢獻的被接受論文**，加上**至少一趟不靠國科會**。
**底線目標**：國科會兩次都用到（2 篇）。第三趟算加分，要看老師計畫裡有沒有國外差旅費。

| 學期 | 作品 | 主投 → 被拒時的備案 | 錢從哪來 |
|---|---|---|---|
| **碩一下／暑假** | A：寫入時依位置放置 vs 成本感知逐出，加上「時間花在哪」 | **W**：ICPE workshop 🇸🇪 或 HotOS（德奧邊境）→ ISC workshop 🇩🇪 或 Euro-Par workshop 🇳🇱<br>**F**：CCGrid 12/1 → 2/1 被拒就改投 ICDCS 🇦🇺 或 Euro-Par 🇳🇱 | 國科會 2027 |
| **碩二上** | B：預測準則（什麼時候值得聰明放置） | 2027 年春天截止的場所。例如 SC27（🔶 Denver 11/14–19，只有第三方來源；照往年約 4 月截稿；CCF A，很難），或 Cluster 2027（未公布） | 老師計畫或學校 |
| **碩二下** | B 或 C（影片／VLM 場景、DL 預測器） | 2027 年秋天截止的場所：SIGMETRICS／ICPE／ISC／CCGrid 2028 | 國科會 2028 |

碩二上和碩二下的場所多半還沒公布 CFP，上表是照往年時程推估。
A 若在 CCGrid 被拒，就往暑假的 ICDCS 或 Euro-Par 推，碩一下的機會只會延後，不會整個沒了。

### 六、10/21 的決策規則

| 10/21 的狀況 | 投哪裡 |
|---|---|
| 老師要走 CVPR，且有真實的影像場景 | CVPR 11/16（Seattle） |
| 結果很乾淨、寫得快，想去歐洲 | ICPE 主會 11/16 |
| **想去北歐、11 月做得出 6 頁（預設）** | **路線 W**：ICPE workshop 或 HotOS 2/1（見第十節）；雅思排 12 月 |
| 想要正式論文的等級與退路 | **路線 F**：CCGrid 12/1；雅思排在 12 月中到 1 月考 |
| 結果還需要時間 | ISC 12/21，或路線 W |

這幾個選項只能擇一（一稿不能多投）。例如 ICPE 要 1/25 才通知，所以投了 ICPE 就不能再投 12/1 的 CCGrid。

### 七、分級怎麼看

| 標準 | 說明 |
|---|---|
| **CCF**（中國計算機學會推薦目錄） | A / B / C |
| **CORE**（現改名 **ICORE**） | A* / A / B / C。2026 版約 7.5% A*、13% A、30% B、46% C |
| 社群口碑 | 看程式委員會裡有誰、領域裡的人去不去 |

三套標準常對不起來：HotOS 是 workshop，卻是 CCF B / CORE A；ISC 只有 CORE C，但它是歐洲 HPC 的主場，TOP500 每年就在 ISC 和 SC 公布。
**先問老師或系上看哪一套**，台灣各校認定的名單不同。

本論文 `refs.bib` 的會議來源以 ICML、EuroSys、SOSP、ATC、ISCA、MICRO 為主，沒有一篇在 ICPE、ISC 或 Euro-Par。
所以投中階 HPC 會議時，審稿人可能不熟 KV cache：競爭較少，但要多花篇幅鋪背景。

**Workshop 激烈嗎？要看是哪個。** Hot 系列很競爭：HotOS 2023 為 31/117（約 26.5%）、2025 為 31/150（約 20.7%）（從投稿系統頁首數得，非官方統計）。
EuroMLSys 找不到官方接受率，不估。主會擋的是「沒做完」，好的 workshop 擋的是「不夠新、引不起討論」。

### 八、仍待確認

| 項目 | 為什麼 |
|---|---|
| **ICDCS 2027 截止日** | 若在 2/1 之後，可當 CCGrid 的備案；若在之前，只能二選一 |
| EuroMLSys 2027、Euro-Par 2027 的 CFP | 尚未發布 |
| **ICPE 2027 的 workshops／Emerging Research／Poster 三個 track** | 官方頁面都寫 "Details TBA"；是瑞典路線的關鍵 |
| ISC 2027 各 workshop 的論文截止 | workshop 名單 11/30 才核定 |
| SC27 地點與截止 | 只有第三方列表，supercomputing.org 尚未公布 |
| 國科會「首次發表」對 workshop 擴寫版是否成立 | 先發 workshop、再發擴寫的正式論文時，第二次能否申請補助要問系辦 |
| ICS 2027、ISPASS 2027、HPDC 2027、ARCS 2027 | 本次查無 CFP |
| CVPR 2027 的強制註冊日（第三方寫 11/10） | 官方頁面未確認 |
| 各場所的 CCF / CORE 等級 | 本次以 myhuiban 為準，ICORE 官方頁面連線回 403 |

### 九、偏好地區：北歐、瑞典、奧地利、荷蘭

使用者偏好這些高消費國家。**國科會只部分補助經濟艙機票與註冊費**，住宿和生活費不補。
所以在這些國家，住宿費要靠老師計畫或自費，這點要先問老師。

| 地點 | 場所 | 日期 | 截止 | 備註 |
|---|---|---|---|---|
| 🇸🇪 Gothenburg | ICPE 2027 主會 | 5/24–28 | ✅ 11/16 | CORE B |
| 🇸🇪 Gothenburg | ICPE 2027 的 workshops／Emerging Research／Poster | 5/24–28 | 🔶 TBA（ICPE 2025 的 workshop 截止在 1 月初到 2 月初） | 參考：HotCloudPerf 2026 是 5 頁 + 1 頁參考文獻，摘要 1/19 |
| 🇩🇪 Burghausen（德奧邊境） | HotOS 2027 | 5/24–26 | ✅ 2/1 | 5 頁（不含參考文獻），4/9 通知，**須親自出席** |
| 🇳🇱 Groningen | Euro-Par 2027 主會與 workshops | 8/23–27（暑假） | 🔶 主會約 3 月中 | CCF B |
| 🇩🇪 Hamburg | ISC 2027 workshops | 6/11 | 🔶 由各 workshop 自訂，通知最晚 4/9 | 有 proceedings 的發在 FGCS 期刊 |
| 🇩🇪 Hamburg | ISC 2027 Research Poster | 6/8–10 | ✅ 1/20（通知 3/2） | non-archival |

- **ICPE 和 HotOS 同一週（5/24 起）**，只能去一個。
- **維也納、哥本哈根、奧斯陸、赫爾辛基、斯德哥爾摩**：2027 年查不到系統／HPC 的主要會議。斯德哥爾摩只有 ICMLT（見下方 8/30 的分析，不值得投）和 DIS（HCI）。
- **2028 年**：Euro-Par 2028 在 🇫🇷 Lyon（Euro-Par 歷史頁面所列）；ICPE 2028、EuroSys 2028 未公布；ICML 2027 在南美、2028 在美東（ICML 官方 Future Meetings）。
- 8/30 版本把「奧」當成澳洲處理；澳洲的 ICDCS（墨爾本 7 月）保留為額外備案。

### 十、「11 月做出 6 頁 workshop 論文」可行嗎

**可行，而且份量剛好對得上兩週驗證之後能做出的東西。但要知道三件事。**

**1. Workshop 截稿多半在 1–2 月，不在 11 月。** 11 月完成等於有餘裕，這份稿子有兩條路，只能選一條：

| 路線 | 做法 | 碩一下去哪 | 代價 |
|---|---|---|---|
| **W（workshop，偏歐洲）** | 留著投 1–2 月的 workshop | 🇸🇪 Gothenburg 或德奧邊境 | 第一篇等級較低；之後擴寫成正式論文要 ≥25% 新內容 |
| **F（正式論文）** | 補到 10 頁，投 CCGrid 12/1 | 🇺🇸 Dallas | 12/1 前要做完完整評估；被拒（2/1）後才能改投 workshop |

走 F 的話，CCGrid 通知（2/1）和 HotOS 截止（2/1）同一天，ICPE 的 workshop 往年在 1 月截止，所以這兩個基本上都接不到；剩下的 workshop 備案是 EuroMLSys 和 ISC workshop。

**2. Workshop 不一定比較好上。** HotOS 要的是 "position papers that propose new directions of systems research"，接受率約 20–26%。單純的量測結果比較適合 ICPE 的 workshop 或 EuroMLSys。

**3. 大多是 archival。** HotOS、EuroMLSys、ISC 有 proceedings 的 workshop 都會正式出版。之後擴寫成正式論文時，要 ≥25% 新內容並主動揭露。國科會要求「首次發表」，擴寫版第二次申請是否成立要問系辦。

**建議（照使用者的偏好）**：走 **W**。
- 預設投 **ICPE 2027 的 workshop 或 Emerging Research track（瑞典）**，等 CFP 出來確認截止日與頁數。
- 如果 11 月的稿子能講成一個「系統研究新方向」的論點（例如何時值得聰明放置的準則），就改投 **HotOS（2/1）**。等級最高（CCF B / CORE A），而且就在德奧邊境。
- 被拒的備案：ISC workshop（漢堡 6/11），或 Euro-Par workshops（格羅寧根，暑假）。
- 雅思可以排在 12 月考，不會撞到截稿。

### 本節來源（2026-10-08 查閱）

- [HotOS 2027](https://www.sigops.org/s/conferences/hotos/2027) · [HotOS 2027 CFP](https://www.sigops.org/s/conferences/hotos/2027/cfp.html)
- [ISC 2027 Submissions（所有 track 的日期）](https://isc-hpc.com/submissions/)
- [ICPE 2027 Tracks](https://icpe2027.spec.org/tracks-and-submissions/) · [ICPE 2027 Emerging Research Track](https://icpe2027.spec.org/tracks-and-submissions/emerging-research-track/)
- [HotCloudPerf](https://hotcloudperf.spec.org/) · [ICPE 2025 collective CFP for workshops](https://lists.cs.umbc.edu/pipermail/agents/Week-of-Mon-20241209/014867.html)
- [Euro-Par history（含 2027 Groningen、2028 Lyon）](https://2020.euro-par.org/about-euro-par/history)
- [ICML Future Meetings](https://icml.cc/Conferences/FutureMeetings)
- [EuroSys 2027](https://2027.eurosys.org/)

- [ICPE 2027 Important Dates](https://icpe2027.spec.org/important-dates/) · [SPEC news: ICPE 2027 CfC](https://www.spec.org/notes/2026/news260904/)
- [ISC 2027 Research Paper](https://isc-hpc.com/submissions/research-paper/)
- [PDP 2027](https://research.ac.upc.edu/PDP2027/index.html)
- [CCGrid 2027 CFP (EasyChair)](https://easychair.org/cfp/ccgrid27)
- [SIGMETRICS 2027 CFP](https://sigmetrics.org/sigmetrics2027/pages/cfp.html)
- [CVPR 2027 Call for Papers](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)
- [ICDCS 2027](https://icdcs2027.icdcs.org/) · [ICORE: ICDCS](https://portal.core.edu.au/conf-ranks/1002)
- [HotOS 2027](https://www.sigops.org/s/conferences/hotos/2027) · [HotOS 2023 HotCRP](https://hotos23.hotcrp.com/) · [HotOS 2025 HotCRP](https://hotos25.hotcrp.com/)
- [IPDPS 2027 (beri.net)](https://www.beri.net/events/ipdps-2027)
- [SC27 listing (showsbee)](https://www.showsbee.com/fairs/100096-SuperComputing-Conference-2027.html)
- [Euro-Par 2026 CFP](https://2026.euro-par.org/calls/papers) · [Euro-Par 2025 CFP](https://2025.euro-par.org/calls/papers)
- [MLSys Deadlines](https://mldeadlines.com/conference/mlsys/)
- [myhuiban systems ranking](https://www.myhuiban.com/conferences/rank/field/systems.md) · [myhuiban ICPE](https://www.myhuiban.com/conference/1537.md) · [myhuiban SIGMETRICS](https://www.myhuiban.com/conference/327.md)
- [ICORE: ISC](https://portal.core.edu.au/conf-ranks/1340)
- [國科會補助國內研究生出席國際學術會議作業要點（政大統計系存檔）](https://stat.nccu.edu.tw/uploads/asset/data/6881a642311fc83967f11c77/%E5%9C%8B%E7%A7%91%E6%9C%83%E8%A3%9C%E5%8A%A9%E5%9C%8B%E5%85%A7%E7%A0%94%E7%A9%B6%E7%94%9F%E5%87%BA%E5%B8%AD%E5%9C%8B%E9%9A%9B%E5%AD%B8%E8%A1%93%E6%9C%83%E8%AD%B0%E4%BD%9C%E6%A5%AD%E8%A6%81%E9%BB%9E.pdf)
- [陽明交大 研發處學生補助系統](https://spsys.ord.nycu.edu.tw/student)

---

> 以下為 **2026-08-30 原始版本**。事實錯誤已就地修正並標註「10/8」；推薦與時程分析部分已被上方更新節取代，保留作為紀錄。

---

## 一覽表

✅ = 官網明載　🔶 = 由前一屆推估，**投稿前必須自行確認**

### 🇪🇺 歐洲場所（含北歐）

| 場所 | 地點 | 會議日 | **截止日** | 出版 | 難度 |
|---|---|---|---|---|---|
| **ISC High Performance 2027**<br>（Research Paper） | 🇩🇪 **Hamburg** | ~~6/23–25~~ ✅ **6/7–11, 2027**（10/8 修正） | ✅ **12/21/2026**<br>（明載 no extension!） | **IEEE Xplore**<br>開放取用，10 頁 | 🟡 中階 |
| **ISC 2027 Workshop**<br>（with Proceedings） | 🇩🇪 **Hamburg** | ~~6/26~~ ✅ **6/11, 2027**（10/8 修正） | 🔶 ~3 月初 2027<br>（各 workshop 自訂） | 🌟 **FGCS**<br>Elsevier **Q1**, IF 8.22 | 🟢 較易 |
| **ICPE 2027** | 🇸🇪 **Gothenburg** | 5/24–28, 2027 | ✅ **11/16/2026**<br>（摘要 11/9；10/8 查證） | ACM | 🟡 中階 |
| **Euro-Par 2027** | 🇳🇱 **Groningen** | 8/23–27, 2027 | 🔶 ~2–3 月 2027 | Springer LNCS | 🟡 CCF B |
| **PDP 2027**（Euromicro） | 🇪🇸 **Barcelona** | 3/17–19, 2027 | ✅ 10/26/2026<br>（摘要 10/19；10/8 查證） | IEEE CPS | 🟢 容易 |
| **ARCS 2027**（第 40 屆） | 🇩🇪 德國（城市未定） | 🔶 ~3 月 2027 | 🔶 ~2 月 2027 | Springer LNCS | 🟢 容易 |
| **EuroMLSys 2027**（workshop） | 🇲🇦 Rabat（隨 EuroSys） | ~4/19, 2027 | 🔶 ~2 月 2027 | ACM DL | 🟢 最容易（**6 頁**）|
| EuroSys 2027 秋輪 | 🇲🇦 Rabat | 4/19–23, 2027 | ❌ 9/24/2026（已過） | ACM | 🔴 頂會 |
| HiPEAC 2027 | 🏴 Glasgow | 1/18–20, 2027 | — | — | 特殊形式 |

### 非歐洲（對照）

| 場所 | 地點 | **截止日** | 難度 |
|---|---|---|---|
| **MLSys 2027** | 未公布 | ✅ **10/30/2026** | 🔴 頂會 |
| **CCGrid 2027** | 🇺🇸 Dallas | ✅ **12/1/2026**（~~12/8~~，10/8 修正；摘要 11/24） | 🟡 CCF C |
| ~~Middleware 2026~~ | 🇪🇸 Tarragona | ❌ 已過 | — |

---

## 🥇 首選：ISC High Performance 2027（Hamburg 🇩🇪）

> ⚠️ **10/8**：已不再是首選，改成路線 W／F 擇一，見最上方更新節。會期與主題的錯誤已在下表修正。

**兼顧「德國 + 時程剛好 + 發表價值高 + 適合申請補助」。**

| | |
|---|---|
| 地點 | **Hamburg, 德國** |
| 會議 | ~~6/23–25, 2027（tutorial 6/22、workshop 6/26）~~ ✅ **6/7–11, 2027**（tutorial 6/7、論文場次 6/8–10、workshop 6/11；10/8 修正） |
| **Research Paper 截止** | ✅ **2026-12-21，官網明載「no extension!」** |
| 通知 | 2027-03-15 |
| **出版** | **IEEE Xplore，完全開放取用**（會議支付出版費用） |
| **頁數** | **10 頁**（不含參考文獻），camera-ready 可 +1 頁 |
| **審稿** | 雙盲、3–4 份審稿、**含 rebuttal**、現場討論共識決 |

### 三個優點

1. **10 頁、開放取用、審稿嚴謹。** IEEE Xplore 完全開放取用且由會議支付出版費；審稿為雙盲、每篇 3–4 份、**含 rebuttal 階段**、審稿人現場討論共識決。10 頁也比 12–18 頁的場所好寫。
2. **時程剛好。** 距今約 113 天——Oracle + 消融 + baseline 對照做得完，且比 MLSys（61 天）寬鬆得多。
3. **註冊費貴反而是優點。** ISC 帶展覽性質，註冊費高於一般學術會議——**這正是申請補助的正當理由**，補助額度通常以實際支出核銷。

### 一個要注意的

ISC 的主場是 HPC／超級電腦。本論文的 **MI300X + ROCm + 記憶體階層 + 能耗量測**這條線契合，但**純 LLM 服務的角度可能偏離**。
👉 **投稿前務必看 9/1 開放的 CFP topics，確認有 AI/ML systems 類別。**
✅ **10/8 已確認**：有 Artificial Intelligence 類別，也有 "Memory Technologies and Hierarchies"。

---

## 🥈 北歐選項：ICPE 2027（Gothenburg 🇸🇪）

**契合度其實比 ISC 更高。**

ICPE 是 performance engineering 場所，重視**量測方法學**——本論文的 §2.4（為何該用 FLOP/byte 而非 HBM:link）、κ 的跨平台量測、ROCm SDMA counter 對 `rocprof` 不可見所以改用 `rocprofv3 --memory-copy-trace`——**這些在 ML 場所常被當枝節，在 ICPE 是加分項。**

🔶 截止推估 ~11 月 2026（ICPE 2026：摘要 11/10、全文 11/17）。哥德堡 5 月天氣宜人。

---

## 🥉 時程最寬鬆：Euro-Par 2027（Groningen 🇳🇱）

CCF B，8/23–27/2027，截止 ~2–3 月 2027。**給你半年做實驗**，Springer LNCS 出版。

---

## 🟢 最容易的兩個

| | 地點 | 出版 | 說明 |
|---|---|---|---|
| **PDP 2027** | 🇪🇸 Barcelona | IEEE CPS | Euromicro 系列，3 月會議，門檻低，Scopus 索引 |
| **ARCS 2027** | 🇩🇪 德國 | Springer LNCS | GI/ITG 主辦，第 40 屆，德國本土，門檻低 |

**ARCS 對「想去德國」這個需求最直接**——它就是德國的會議，每年在不同德國城市（2026 在 Mainz）。

---

## 🇸🇪 斯德哥爾摩專查（2026-08-30）

**系統／HPC 類：查無。** 2027 年斯德哥爾摩沒有本領域的會議。

查到的只有兩個，都有問題：

| 場所 | 日期 | 截止 | 問題 |
|---|---|---|---|
| **ICMLT 2027**<br>Machine Learning Technologies | 5/21–23, 2027 | ✅ **12/10/2026** | **層級低**（見下） |
| ACM **DIS 2027**<br>Designing Interactive Systems | 6/28–7/2, 2027 | — | **主題完全不對**（HCI） |

### ICMLT 2027 值不值得投

**是合法會議，不是掠奪性期刊**：第 12 屆，ACM ICPS 出版，proceedings 進 **ACM Digital Library**，並由 **EI Compendex 與 Scopus** 索引。

**但要清楚知道代價：**

| | |
|---|---|
| ❌ **在 systems 領域沒有份量** | 履歷上的重量遠低於 ISC／ICPE／Euro-Par |
| ❌ **可能擋住後續投稿** | 已發表內容再投頂會需大幅擴充並揭露 |
| ⚠️ **官網當日連線被拒**（ECONNREFUSED） | 非決定性，但投稿前請自行確認網站與 CFP 有效 |
| ✅ Scopus／EI 索引 | 若只是要一篇有索引的論文，這點是成立的 |

**建議**：本論文的主張夠強，投 ICMLT 是浪費。**除非你有「必須在特定期限前有一篇 Scopus 論文」的硬性需求。**

### 務實解法：發 ICPE，順路玩斯德哥爾摩

**ICPE 2027 在 Gothenburg（哥德堡），到斯德哥爾摩搭 SJ 高鐵約 3 小時。**
5/24 開會，會後直接北上玩，機票與住宿都在補助的核銷範圍內（依規定）。

漢堡（ISC）到斯德哥爾摩也只是一段短程飛行。

---

## ⚠️ 查證時抓到的一個陷阱

搜尋「ICML 2027」會跑出 **Eurac Research（義大利 Bolzano，6/22–26/2027）**，看起來像是機器學習的 ICML。

**那是 International Conference on Minority Languages（少數語言國際會議），不是 International Conference on Machine Learning。** 同縮寫不同會議。

**機器學習的 ICML 2027 地點與截止日，官網（icml.cc）目前未公布。**

另：**ACM SIGMETRICS 2027 在 Atlanta（美國）**，非歐洲，但採三輪滾動投稿（秋輪 10/9/2026、冬輪 1/11/2027），時程彈性大，可列為備案。

---

## 🚫 一稿多投？各場所的實際條文

### 同時投多個場所 = **嚴格禁止**

**EuroSys 2027 CFP 原文：**
> "Submissions should contain original, unpublished material. **Simultaneous submission of the same work to multiple venues** and submission of previously published work **are not allowed**."

**MLSys CFP 原文：**
> "We will not accept any paper which, **at the time of submission, is under review for another conference** or has already been published."
> 且審稿期間不得投往其他會議。

**ACM 全域政策：**
> "Under no circumstances shall a paper (or substantially the same paper) be **simultaneously submitted to two or more publications**, or to a second publication **while still under review elsewhere**."

**後果**：ACM 明載違反者將被調查，**可能導致論文全面撤稿及其他處分**（通常還包括通知作者所屬機構、列入黑名單）。

### 可以做的三件事

| 做法 | 可行嗎 | 條件 |
|---|---|---|
| **被拒之後投下一個** | ✅ **完全正常** | 這是學術界的常態，沒有任何限制 |
| **先放 arXiv** | ✅ 可以 | EuroSys：「arXiv 等非同儕審查的發表**不算**同時投稿」；MLSys 同樣允許 |
| **Workshop 先發，再擴寫** | ✅ 可以 | 須**顯著擴充**並主動揭露。ACM 標準為**至少 25% 新內容**，投稿時附信說明差異並提供原 workshop 論文 |

⚠️ EuroSys 另有一條容易忽略的規定：arXiv 版本須使用**顯著不同的標題與系統名稱**。

---

## 📄 Workshop 論文 vs 會議論文：差在哪

### 最根本的差別是「它在問什麼問題」

| | 審稿人在問 |
|---|---|
| **會議論文** | 「你**做完了**嗎？站得住腳嗎？比既有方法好嗎？」 |
| **Workshop 論文** | 「這個**方向值得追**嗎？有沒有討論價值？」 |

**會議論文會因為這些被拒**：baseline 不夠、沒有消融、沒跟最新工作比、實驗規模太小。
**Workshop 論文不會**——只要你誠實寫明這是初步結果。

### 規格差異

| | Workshop | 會議論文 |
|---|---|---|
| 頁數 | **4–8 頁** | 10–18 頁 |
| 審稿人數 | 2–3 | 3–5 |
| 接受率 | 約 40–70% | 15–35% |
| Rebuttal | 通常無 | 常有 |
| 報告時間 | 10–15 分鐘 | 20–30 分鐘 |
| 履歷份量 | 低 | **高** |

---

### 🔴 真正要搞懂的分野：archival vs non-archival

**這比「workshop 還是會議」重要得多。**

| | Archival（有正式 proceedings） | Non-archival（無 proceedings） |
|---|---|---|
| 進資料庫？ | ✅ ACM DL／IEEE Xplore／Springer | ❌ 只有摘要或什麼都沒有 |
| 算不算「已發表」 | ✅ **算** | ❌ **不算** |
| 之後投完整版 | ⚠️ **觸發 25% 新內容規則** | ✅ **完全不受限制** |
| 履歷上能寫 | ✅ | 只能寫「受邀報告」 |

**Non-archival 的意義**：你去報告、拿回饋、認識人，但**在紀錄上等於沒發表過**——完整版之後投哪裡都不受影響。

> 查證到的通則：「Non-archival submissions allow authors to publish only the abstract… accommodates publication of the work **or a superset at a later date** in a conference or journal which does not allow previously archived work.」
> 且「Non-archival submissions are **expected to describe the same quality of work** as archival submissions and are reviewed following the same procedure」——**審稿一樣嚴，差別只在出版與後續投稿資格。**

### 本文相關場所的歸類

| 場所 | 類型 | 會綁住你嗎 |
|---|---|---|
| **EuroMLSys** | **Archival**（ACM DL） | ⚠️ **會**，之後完整版須 ≥25% 新內容 |
| **ISC「Workshops with Proceedings」** | **Archival** | ⚠️ 會（但見下方，出版管道很好） |
| ISC「Regular Workshops」 | 多為 non-archival（僅由主辦方選擇性提交摘要論文） | ✅ 通常不會 |
| ICML／NeurIPS 系列 workshop | 多為 **non-archival** | ✅ 不會 |

---

### 💡 新發現：ISC 的 workshop 論文發在 Q1 期刊

**ISC 2027「Workshops with Proceedings」的論文出版於 Future Generation Computer Systems（Elsevier）**，四週全球開放取用。

**FGCS 是 Q1 期刊：IF 8.22、CiteScore 18.7、h-index 193。**

也就是說——**這條路是「workshop 的門檻，Q1 期刊的出版」**。

| | |
|---|---|
| Workshop 提案截止 | 2026-10-07（**這是主辦方提 workshop 的期限，不是你投論文的期限**）|
| 提案通知 | 2026-11-30 |
| **論文截止** | 🔶 建議為 **2027 年 3 月初**，由各 workshop 自訂 |
| 論文通知 | 最遲 2027-04-09 |
| Camera-ready | 2027-04-23 |

⚠️ **個別 workshop 的 CFP 要等 11/30 核准後才會出現。** 到時要看有沒有 AI/ML systems 相關的 workshop。

---

## ✈️ 「一直投 workshop 來一直出國」可行嗎

**可行，但瓶頸不在論文，在四件事。**

### ① 經費：一年只有一次（已查證）

國科會不管你發幾篇，**一年就補助一次**。刷十篇 workshop 也一樣。
👉 真正能「一直去」的是**指導教授計畫項下的國外差旅費**（無次數限制，只受預算限制）。

### ② 內容：每篇必須是不同的東西

同一個 idea 拆成三篇 workshop = **salami slicing**，審稿人與同領域的人看得出來，且損害聲譽。
👉 「一直刷」的真正成本是**你要一直有新東西**。

### ③ Archival 會累積綁住你 ⚠️

每發一篇 **archival** workshop，那部分內容就進了「已發表」清單。
發多了，最後完整版能宣稱的新內容\emph{湊不到 25%}。**這是最實際的風險。**

### ④ 履歷效果遞減，甚至反向

三篇 workshop 的份量**小於**一篇好的會議論文。而「只有 workshop 沒有會議論文」在申請博班或求職時是明顯的負面訊號。

---

### 🔑 正解：poster 與 non-archival 軌

**要「出國但不燒掉論文」，走這條。**

**ISC 2027 Research Poster（已查證）**

| | |
|---|---|
| 要交什麼 | 250 字摘要 + **1000 字延伸摘要** + 一張 A0 海報 PDF |
| 審稿 | 3 位審稿人，單盲 |
| **是否 archival** | ❌ **不是**。僅於會場平台提供給與會者，**不進 proceedings** |
| **截止** | ✅ **2027-01-20**（通知 3/02） |

~~⚠️ 該頁寫「展示期間 June 8–10, 2027」，與投稿總覽頁的 6/23–25 不符，疑為未更新，投稿前確認。~~
✅ **10/8 修正**：6/8–10 才是對的，錯的是 6/23–25。截止 1/20、通知 3/2 已重新查證。

**為什麼這是正解**：
- 工作量極小（1000 字 vs 10 頁）
- **不算發表 → 完全不綁住完整版**
- 一樣去漢堡、一樣能申請補助

### 正確的組合

```
一份工作 ──┬─→ 完整會議論文（ICPE／ISC 主軌）  ← 履歷主力，出國一次
           │
           └─→ poster／non-archival workshop    ← 額外出國，零代價
```

**而不是**把一個 idea 切成好幾篇 archival workshop——那會同時傷害履歷與後續投稿空間。

---

## 📚 一個 idea 能發幾篇？拆成 PoC 級與完整級

**可以，而且是 systems 領域的標準做法**，但每一階必須有實質新增。

| 階段 | 內容 | 場所 | 頁數 |
|---|---|---|---|
| **① PoC 級** | Oracle + 消融 (B)：核心主張本身 | EuroMLSys 等 workshop | **6** |
| **② 完整級** | 完整系統、多平台、完整 baseline | ICPE／ISC／Euro-Par／MLSys | 10–18 |
| **③ 期刊版** | 再擴充：更多模型、更深分析 | TPDS／FGCS／JPDC | 不限 |

**規則**（ACM 全域政策）：每一階須**至少 25% 新內容**，且投稿時**主動揭露**前一階並附上原文與差異說明。

⚠️ **界線在哪**：真正的階段性發展（先驗證概念、再做完整系統）可以；把\emph{同一個}貢獻硬切成最小單位（salami slicing）會被審稿人抓出來，且損害聲譽。
本論文的分界很清楚——**PoC 級證明「損失函數要編碼成本」，完整級證明「這個系統實用」**——是兩個不同的主張。

### ⚠️ 但順序不該是「先 workshop」

因為時程剛好允許反向操作：

```
ICPE 截止 11/17 ──→ 1/19 通知
                        │
                   沒中 ↓
EuroMLSys 截止 ~2/24  ← 還來得及 ✅
```

**先投高的，被拒再投 workshop。** 反過來不行——workshop 一旦發表就觸發 25% 規則，後面反而綁手綁腳。

**除非**你的實驗到 11 月只做得出 Oracle + 消融，那就直接鎖定 6 頁的 workshop。

---

## 💥 那 impact 是不是很虧？

**三個層次分開看。**

### ① 優先權與引用：**arXiv 解決大半**

所有查過的場所都允許 arXiv 預印本。**投稿當天同步放 arXiv**，時間戳記與引用就開始累積，不必等審稿。

⚠️ **EuroSys 是例外**，其 CFP 明載：

> "the submitted version should have **a substantially different title and use a different system/tool name**, if applicable."

意思是若已放 arXiv，投 EuroSys 的版本要換標題換系統名。**這條相當罕見，若鎖定 EuroSys 就先別放 arXiv。**

### ② 場所層級：**有差，但差在履歷不在影響力**

在 systems 領域，MLSys／EuroSys 的名氣確實高於 ICPE／Euro-Par。但實際被引用與被採用，取決於**東西好不好、找不找得到**——後者由 arXiv 決定。

### ③ 真正的損失：**時間，不是層級**

這個領域跑得很快——KVP、ForesightKV、LookaheadKV 都是 ICML'26／ICLR'26，**幾個月就一輪**。

```
投 ICPE 被拒，損失 2 個月      ← 可承受
投 ISC 被拒，損失 5 個月       ← 期間可能被別人做掉
死等 MLSys 2028，損失 12 個月  ← 最虧
```

**被搶先發表，比發在中階場所虧得多。**

### 結論

| 擔心 | 實際狀況 |
|---|---|
| 「發中階會不會沒人看到」 | arXiv 解決 |
| 「一個 idea 只能發一篇好虧」 | 可拆 PoC → 完整 → 期刊三階 |
| 「投低了浪費」 | 真正浪費的是**等待與被搶先** |

---

## ⏱️ 時程衝突分析：只能選一條路

因為不能同時投，**每次投稿都會鎖住你到通知日為止**。已知的通知日：

| 場所 | 截止 | **通知** | 鎖住期間 |
|---|---|---|---|
| MLSys 2027 | ✅ 10/30/26 | ✅ **2/28/27** | 4 個月 |
| ICPE 2027 | ✅ 11/16/26（10/8 修正） | ✅ **1/25/27** | **2 個月** |
| CCGrid 2027 | ✅ 12/1/26（10/8 修正） | ✅ 2/1/27 | 2 個月 |
| ISC HPC 2027 | ✅ 12/21/26 | ✅ **3/15/27** | **3 個月** |
| EuroMLSys / ARCS / Euro-Par | 🔶 ~2–3 月 2027 | — | — |

### 三條路的後果

**🅰️ 先投 ICPE（11/17）**
```
11/17 投 ──→ 1/19 通知
                │
        中 ✅ ──→ 5/24 哥德堡
        沒中 ──→ 2–3 月還能投 EuroMLSys／ARCS／Euro-Par ✅
```
**只損失 2 個月，二月三月的場所全部保留。**

**🅱️ 先投 ISC（12/21）**
```
12/21 投 ──→ 3/15 通知
                │
        中 ✅ ──→ IEEE Xplore + 漢堡
        沒中 ──→ ❌ 2–3 月的場所全部錯過
                  下一個要等 EuroSys 2028 春輪（~5 月）
```
**沒中就損失 5 個月。**

**🅲 先投 MLSys（10/30）**
```
10/30 投 ──→ 2/28 通知
        沒中 ──→ ICPE／CCGrid／ISC 全錯過
                  Euro-Par（~3 月）可能還來得及，很緊
```

### 結論

| 你的偏好 | 選這條 |
|---|---|
| **想要保險、被拒後還有路** | 🅰️ **ICPE**（79 天準備，被拒只損失 2 個月） |
| **想要最好的發表、準備時間最多** | 🅱️ **ISC**（113 天準備，IEEE Xplore 開放取用，但被拒損失 5 個月） |
| 想拚名氣 | 🅲 MLSys（61 天，最緊，接受率最低） |

⚠️ **ICPE 2027 的日期是由 2026 屆推估**，官網未公布。**若實際截止日晚於 12/21，🅰️ 與 🅱️ 的比較會改變**——投稿前務必確認。

---

## 📏 難度校準：拿 IEEE GCCE 當基準

**先看最關鍵的一個數字：**

| | 頁數 | 要交什麼 |
|---|---|---|
| **IEEE GCCE** | ✅ **2 頁**（官網明載） | 一個想法 + 初步結果 |
| ICPE / Euro-Par / ISC | **12–18 頁** | **完整實驗 + 與前人比較 + 可重現** |

**差別不在接受率，在「投稿前要完成多少工作」。**
2 頁投的是「我有個點子而且看起來可行」；12 頁投的是「我做完了，而且證明比既有方法好」。

### 已查證的接受率

| 場所 | 接受率 | 來源年份 |
|---|---|---|
| **ICPE** | **34.7%** | 2025（2024: 34.6%、2023: 32.6%、2022: 24%）|
| **Euro-Par** | 28.4%（2017）／41.5% 長期平均（1995–2011） | — |
| ISC HPC | 未公開 | — |
| EuroSys / MLSys | ~15–20%（業界共識，非官方數字） | — |
| IEEE GCCE | 未公開 | — |

**ICPE 的 34.7% 其實不可怕——三篇中一篇。** 對一篇實驗紮實的論文而言完全在射程內。

### 難度階梯

```
IEEE GCCE（2 頁）              ← 你投過的位置
   ↓  ← 這一段落差最大：從「點子」變成「完整實驗」
ICMLT（全文，但層級低）
   ↓
PDP / ARCS                     ← 歐洲全文的入門
   ↓
Euro-Par（~28–41%）/ ICPE（~35%）
   ↓
ISC HPC（IEEE Xplore，10 頁）
   ↓  ← 這一段落差也很大
MLSys / EuroSys（~15–20%）
```

### 本論文目前的位置

**寫作與論證已經在 ICPE／Euro-Par 之上**：46 筆全查證文獻、形式化問題定義、可證偽預測、誠實的限制章節。

**缺的純粹是實驗。** 不是「寫得不夠好」，是「還沒跑」。

👉 **所以難度問題的答案是：對你來說，難的不是寫，是跑完 Oracle 與消融。** 一旦有數字，投 ICPE／Euro-Par 的成功機率不低。

---

## 💰 補助

### 主要管道：國科會（NSTC）

| 方案 | 對象 |
|---|---|
| **補助專家學者出席國際學術會議** | 教師／研究人員 |
| **補助國內研究生出席國際學術會議** | 研究生 |

**⚠️ 已查證的關鍵時程規則：**

> 校方最遲彙送日期為**會議首日所屬月份之前一個月之首日**，逾期不受理。
> 論文接受證明未能檢具者應註明補送，至遲應於**會議舉行首日四週前**傳送國科會。

**換算成實際期限：**

| 會議 | 會議首日 | **校內彙送最遲** |
|---|---|---|
| ISC HPC 2027 | 2027-06-07（10/8 修正） | **2027-05-01** |
| ICPE 2027 | 2027-05-24 | **2027-04-01** |
| Euro-Par 2027 | 2027-08-23 | **2027-07-01** |
| PDP 2027 | 2027-03-17 | **2027-02-01** |

👉 **校內截止通常比國科會更早**，實際日期問系上或研發處。

### ⚠️ 次數限制：一年只有一次

已查證的條文：

| 對象 | 條文 |
|---|---|
| **研究生** | 作業要點第五點第(一)款：「研究生**每年度內以補助一次為限**；論文為合著者，每一篇論文以**補助一位**研究生發表為限。」 |
| **專家學者** | 作業要點第五點：「出席人員在**同一會計年度內以補助一次為限**。」 |
| **例外** | 第十四點：擔任國際學會理監事、國際期刊編委等特殊職務且**不發表論文**者，不受一次之限制。 |
| **懲罰** | 未依期限完成報告與結報者，**次一年度不得提出申請**。 |

**所以不是三個月一次，是一年一次。** 且合著論文只補助一位研究生。

### 那要怎麼「一直出國」

**出國次數本身沒有上限，受限的是經費來源。** 國科會那一次是額外的，主力是別的：

| 管道 | 次數限制 | 說明 |
|---|---|---|
| 🥇 **指導教授計畫項下國外差旅費** | **無次數限制** | 只受計畫預算限制。**這才是主力**，多數人靠這個 |
| 國科會補助 | **一年一次** | 額外的一次，額度以機票旅費為主 |
| 校內／院內／系上補助 | 依各校辦法 | **陽明交大有自己的辦法，去問系辦或研發處** |
| 產學合作計畫 | 依計畫 | 有業界合作時可用 |

👉 **實務做法**：把國科會那一次留給最貴的那場（例如 ISC 在漢堡），其餘用計畫項下差旅費。

### 其他
- 部分會議有 student travel grant，但歐洲場名額多以歐盟機構為主，非歐盟申請者機會較低

---

## 建議路線

> ⚠️ **10/8**：本節已被最上方更新節的第五、六、十節取代。MLSys 與 ISC-first 的路線不再適用。

```
現在 ──→ 9 月底：Oracle 結果
              │
      ┌───────┴──────┐
   有 headroom      沒有 → 停損改題
      │
      ├─→ 想拚頂會：MLSys 2027（10/30，剩 61 天，很緊）
      │
      └─→ 🥇 ISC HPC 2027（12/21，剩 113 天）🇩🇪 Hamburg
                 │  IEEE Xplore 開放取用，10 頁，含 rebuttal
                 │  註冊費貴 → 補助理由充分
                 │
                 └─ 沒中 ─→ ICPE 2027（~11 月）🇸🇪 Gothenburg
                            或 Euro-Par 2027（~2–3 月）🇳🇱 Groningen
                            或 ARCS / PDP（門檻最低）🇩🇪 🇪🇸
```

**為什麼把 ISC 排在 MLSys 前面**：多 52 天做實驗、發表是期刊、地點在德國。
MLSys 名氣大但只剩 61 天，且投稿量大、接受率低。

**如果只想穩穩發一篇順便去玩**：**ARCS 2027（德國）或 PDP 2027（Barcelona）**，門檻最低，Springer／IEEE 出版，一樣可申請補助。

---

## 待自行確認

| 項目 | 為什麼要自己查 |
|---|---|
| EuroMLSys 2027 CFP | 尚未發布；~2026 年底至 2027 年初上線，追 `euromlsys.eu` |
| ~~ICPE 2027 important dates~~ | ✅ 10/8 已查到：11/9 摘要、11/16 全文、1/25 通知 |
| Euro-Par 2027 CFP | 官網只有地點與日期（10/8 仍未發布） |
| MLSys 2027 頁數與格式 | Dates 頁已上線，CFP 細節未上線 |
| ~~ISC 2027 的 topics 是否含 AI/ML systems~~ | ✅ 10/8 已確認有 AI 類別 |
| ARCS 2027 地點與截止日 | 官網仍停在 2026（Mainz）；10/8 未重查 |
| ~~PDP 2027 截止日~~ | ✅ 10/8 已查到：10/26（摘要 10/19） |
| 校內補助送件期限 | 通常早於國科會，問系上或研發處 |

> 10/8 新增的待確認事項見最上方更新節的第八節。

---

## 相關筆記
- [[idea-20260828-hierarchical-kv-state]]
- 論文與待辦：`main.tex`、`OPEN_ISSUES.md`、`EXPERIMENT_PLAN.md`
