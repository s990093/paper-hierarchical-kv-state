---
type: report
created: 2026-08-30
tags: [paper, venues, deadlines, kv-cache, europe]
related_ideas: ["[[idea-20260828-hierarchical-kv-state]]"]
---

# 投稿場所調查（查證日：2026-08-30）

對應 idea：[[idea-20260828-hierarchical-kv-state]]／論文 `main.tex`

> **前提**：本論文目前**沒有任何實驗結果**。所有建議都以「Oracle + 消融 (B) 何時能跑完」為前提。

---

## 一覽表

✅ = 官網明載　🔶 = 由前一屆推估，**投稿前必須自行確認**

### 🇪🇺 歐洲場所（含北歐）

| 場所 | 地點 | 會議日 | **截止日** | 出版 | 難度 |
|---|---|---|---|---|---|
| **ISC High Performance 2027**<br>（Research Paper） | 🇩🇪 **Hamburg** | 6/23–25, 2027 | ✅ **12/21/2026**<br>（明載 no extension!） | **IEEE Xplore**<br>開放取用，10 頁 | 🟡 中階 |
| **ISC 2027 Workshop**<br>（with Proceedings） | 🇩🇪 **Hamburg** | 6/26, 2027 | 🔶 ~3 月初 2027<br>（各 workshop 自訂） | 🌟 **FGCS**<br>Elsevier **Q1**, IF 8.22 | 🟢 較易 |
| **ICPE 2027** | 🇸🇪 **Gothenburg** | 5/24, 2027 | 🔶 ~11 月 2026<br>（2026 屆 11/17） | ACM | 🟡 中階 |
| **Euro-Par 2027** | 🇳🇱 **Groningen** | 8/23–27, 2027 | 🔶 ~2–3 月 2027 | Springer LNCS | 🟡 CCF B |
| **PDP 2027**（Euromicro） | 🇪🇸 **Barcelona** | 3/17–19, 2027 | 🔶 ~10–11 月 2026 | IEEE CPS | 🟢 容易 |
| **ARCS 2027**（第 40 屆） | 🇩🇪 德國（城市未定） | 🔶 ~3 月 2027 | 🔶 ~2 月 2027 | Springer LNCS | 🟢 容易 |
| **EuroMLSys 2027**（workshop） | 🇲🇦 Rabat（隨 EuroSys） | ~4/19, 2027 | 🔶 ~2 月 2027 | ACM DL | 🟢 最容易（**6 頁**）|
| EuroSys 2027 秋輪 | 🇲🇦 Rabat | 4/19–23, 2027 | ✅ 9/24/2026 | ACM | 🔴 頂會 |
| HiPEAC 2027 | 🏴 Glasgow | 1/18–20, 2027 | — | — | 特殊形式 |

### 非歐洲（對照）

| 場所 | 地點 | **截止日** | 難度 |
|---|---|---|---|
| **MLSys 2027** | 未公布 | ✅ **10/30/2026** | 🔴 頂會 |
| CCGrid 2027 | 🇺🇸 Dallas | ✅ 12/8/2026 | 🟡 CCF C |
| ~~Middleware 2026~~ | 🇪🇸 Tarragona | ❌ 已過 | — |

---

## 🥇 首選：ISC High Performance 2027（Hamburg 🇩🇪）

**兼顧「德國 + 時程剛好 + 發表價值高 + 適合申請補助」。**

| | |
|---|---|
| 地點 | **Hamburg, 德國** |
| 會議 | 6/23–25, 2027（tutorial 6/22、workshop 6/26） |
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

⚠️ 該頁寫「展示期間 June 8–10, 2027」，與投稿總覽頁的 6/23–25 不符，疑為未更新，投稿前確認。

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
| ICPE 2027 | 🔶 ~11/17/26 | 🔶 **~1/19/27** | **2 個月** |
| CCGrid 2027 | ✅ 12/8/26 | ✅ 2/10/27 | 2 個月 |
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
| ISC HPC 2027 | 2027-06-22 | **2027-05-01** |
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
| ICPE 2027 important dates | 官網目前只有 PC 自薦日（9/13/2026），日期頁未上線 |
| Euro-Par 2027 CFP | 官網只有地點與日期 |
| MLSys 2027 頁數與格式 | Dates 頁已上線，CFP 細節未上線 |
| **ISC 2027 的 topics 是否含 AI/ML systems** | **CFP 於 2026-09-01 開放，務必先確認契合度再投** |
| ARCS 2027 地點與截止日 | 官網仍停在 2026（Mainz） |
| PDP 2027 截止日 | 已知會議日 3/17–19，截止日未上線 |
| 校內補助送件期限 | 通常早於國科會，問系上或研發處 |

---

## 相關筆記
- [[idea-20260828-hierarchical-kv-state]]
- 論文與待辦：`main.tex`、`OPEN_ISSUES.md`、`EXPERIMENT_PLAN.md`
