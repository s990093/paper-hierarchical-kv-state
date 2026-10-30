# ICPE 2027 論文版：章節、圖、表與負面結果的寫法參考

> **日期**：2026-10-30。**讀者**：起草中文「論文版」markdown 的人。這份只談怎麼寫、怎麼畫，**不含任何新數字**；範例句裡的數字一律用 ⟨⟩ 佔位。
> **不重抄**：ICPE 的口味、標題候選、摘要 8 步、措辭表見 [STYLE.md](../icpe_story_20261028/STYLE.md)；頁數與格式見 [CHECKLIST.md](../icpe_story_20261028/CHECKLIST.md) §A；故事見 [README.md](../icpe_story_20261028/README.md)；前作的差異見 [NOVELTY_CHECK_20261030.md](../NOVELTY_CHECK_20261030.md)。
> **讀法的限制**：
> - arXiv HTML 是用 WebFetch 讀的（小模型摘要全文，不是逐字閱讀）。
> - **圖片本身看不到**。凡是「這張圖怎麼畫」的描述，都是從圖說與內文推斷的。
> - 三份 PDF（Wall 1993、ICPE'26 的 ORION 與 VGO）沒有現成工具可讀，是自己解壓文字流後讀的，只讀了標題、圖表說明和幾段內文，標 [全文\*]。
> - 這是 self-review，不是 cross-model review。

---

## 0. 結論：五條

1. **全文的骨架是一條「oracle 階梯」**，從 Wall 1993 借來：Wall 定義七個越來越強的模型（Stupid → Perfect），每張圖都報同一組梯級。我們的梯級：Cake＋固定規則（底）→ 對手（rival）→ 免費訊號 → 單一決策 oracle（→ 聯合 oracle，若有算）。讀者每節只要看兩件事：梯子有多高、免費訊號爬到哪。
2. **計分表是主角，出現兩次**：第 1–2 頁一張小圖；§9 一張完整表，沒過的列也列。判定字詞固定、事先定義在圖說裡。2609.16215 的 Table IV 用自由文字寫判定；我們有事先登記的門檻，不要這樣做。
3. **系統模型圖要同時回答三件事**：KV 停在哪一層、Cake 怎麼重疊、三種預知各插在哪個時間點。審稿人多半是效能建模的背景，這是他們唯一能一眼看懂整個問題的圖。
4. **每個結果節用同一套句式**：節首一句可證偽的主張；節尾一段「Finding & Implication」（2609.28870 的格式）；圖說第一句就是結論。這三處就是 writing-systems-papers skill 的「三次陳述」。
5. **負面結果寫成「上界」，不寫成「失敗」**：主詞是 oracle，不是「我們」；同時給機制（為什麼小）與例外（什麼時候大）；永遠和免費訊號並列。Wall 的教訓：只看 Perfect 模型會走進「危險的花園小徑」。

---

## 1. 章節骨架與頁數

**篇幅**：ICPE Research Track ≤10 頁，含圖表，不含參考文獻與附錄；ACM 雙欄。附錄不出版，正文必須自給自足（CHECKLIST §A）。一張雙欄寬的圖約佔 0.3–0.4 頁、單欄圖約 0.2 頁；下表的圖表合計約 2.8 頁，已經算進各節。

| § | 暫定標題 | 頁 | 這節的一句話 | 圖表 |
|:--|:--|:--|:--|:--|
| 摘要 | — | 0.2 | 照 STYLE.md §2 的 8 步 | — |
| 1 | Introduction | 1.2 | 具名的假設（誰在預測什麼）→ 什麼變了（Cake 不需要預測；有了真實的 agent trace）→ 3 個 RQ → 答案 → 貢獻（每項附 §） | 圖 1 |
| 2 | Background & System Model | 0.9 | 分層、算進 TTFT 的還原、Cake 的會合點、固定分層規則；三種預知插在哪 | 圖 2 |
| 3 | Methodology: A Limit Study | 1.5 | oracle 階梯、三種預知的定義、trace、模擬器與 GPU 驗證、事先登記與噪音規則、污染防護、AI 使用揭露（CHECKLIST §H） | 表 1、表 2、圖 3、圖 4 |
| 4 | Write-time Foresight | 0.7 | 在 GPU 上量不到收益 | 圖 5（可省） |
| 5 | Return-time Foresight | 1.4 | agent 上界低；chat 上界高，但免費訊號拿走大部分 | 圖 6 |
| 6 | Restore-time Foresight | 0.6 | 固定規則就夠；線上選擇器常常選錯 | 併入表 3，或一張小表 |
| 7 | When Does Foresight Pay? | 0.5 | 例外區域寫成判準，最好寫成命題 | 圖 7 |
| 8 | Gains That Need No Foresight | 0.8 | INT4 讓位元組變小；MPS 是取捨的旋鈕，不是免費的收益 | 圖 8 |
| 9 | Scorecard and Threats to Validity | 0.8 | 完整計分表（含沒過的列）＋效度威脅表 | 表 3、表 4 |
| 10 | Related Work | 0.6 | 定位表＋4 段 run-in | 表 5 |
| 11 | Conclusion | 0.2 | 三句：問題、上界、力氣該放哪 | — |

合計 9.4 頁，留 0.6 頁給浮動物件的排版損耗。

- **順序**：§4–6 依「決定的時刻」排（寫入 → 回來 → 還原），和圖 2 的插入點 ①②③ 一一對應，讀者看圖就知道在讀哪一節。
- **每節自己帶「為什麼」**，不另開一節。2609.28870 把「沒用」「為什麼」「什麼有用」分成 §3／§4／§5，那是 20 頁才放得下。
- **超過頁數時依序砍**：刪圖 5（改成表 3 的一列）→ 圖 6(d) 移到 artifact → 表 5 縮成 5 列。

---

## 2. 圖：每張的內容、形式、帶走什麼、借誰的

**全文通用的規則**：
- **圖說**：第一句是結論，第二句是設定，最後標〔模擬〕或〔實測〕。範例：2609.16215 Fig.7「DRAM absorbs most of what overflows HBM; SSD is a shallow backstop.」、Table III「The gain equals the tier's capacity multiple, not a policy effect.」；ORION 的圖說也都是結論句。paper-writing skill 的 G7：粗體結論，加最多一個子句。
- **門檻線**：δ\*、5%、15% 全文用同一種線型與顏色（2609.16215 Fig.10 每個面板都畫 SLO 線，Fig.6 用虛線標預設值）。
- **模擬與實測**：空心標記＝模擬、實心＝實測，圖說也寫（STYLE.md §5）。
- **分佈不只報平均**：VGO Fig.1 直接在面板上寫 n、mean、median、sd；CacheFlow Fig.4 用 CDF 呈現 TTFT。
- **不用內部代號**（A1'、X5、O-PF、S4+）：圖上用描述性名稱，例如「回來時：完美預取」；代號只留在 artifact 的對照表。

**圖 1｜計分小圖**（§1，第 1–2 頁，單欄）
- **內容**：每列是「一種預知 × 一條 trace」（寫入時；回來時的預取、淘汰順序、丟棄、知道誰不回來；還原時的選擇器）。橫軸是「對 Cake＋固定規則的 TTFT 改善（%）」；長條＝oracle 上界，長條上的點＝免費訊號或對手拿到的；垂直參考線＝δ\*、5%、15%；列尾標判定字。最下面兩列「不用預知」（INT4、MPS）換一個顏色。
- **形式**：水平長條＋點（dot-on-bar）。chat 與 agent 尺度差很多，分兩個面板或用對數軸；截斷的軸一定要標示。
- **帶走**：大部分長條在 5% 線左邊；越過 15% 的只有一個有名字的例外；最下面兩列不需要預知。
- **借鏡**：Wall Fig.11–12（七個模型的定義矩陣，同一張圖畫全部梯級）；FOO Fig.1（線上策略、舊上下界、新界放在同一條軸上，看「還剩多少空間」）；2609.28870 Fig.1（findings → implications → techniques 的總覽）；2609.16215 Fig.10（參考線）。
- **注意**：MPS 是取捨，不是一個數字。用區間或註腳「見圖 8」，不要畫成一根正向長條。

**圖 2｜系統模型**（§2，雙欄寬；必備）
- **(a) 層與路徑**：由上而下 GPU HBM／CPU DRAM／SSD 級／NFS 級；SSD、NFS 用虛線框並標「限速模擬」；每層旁標頻寬符號（B_cpu、B_ssd…，數值在表 2）。兩種箭頭：寫出（請求結束 → 降層）、還原（請求回來 → GPU）。GPU 上的還原器標「固定分層規則：CPU 層全部載入，慢層用 Cake」。
- **三個插入點**（圓圈編號標在路徑上，旁邊各寫一行「oracle 事先知道：＿＿」）：① 寫入時：降層那一刻，放哪層、存什麼精度；② 回來時：兩次請求之間，預取、淘汰順序、丟棄；③ 還原時：請求到達，選還原方式。
- **(b) 還原時間軸**：橫軸是 token 位置；GPU 從前段往後重算、I/O 從後段往前載入，在會合點相遇；下方括號標「算進 TTFT」；下面兩列對照：全部載入、全部重算。
- **帶走**：還原時間就在 TTFT 裡；三種預知各自只能改變其中一段。
- **借鏡**：2609.16215 Fig.4（層堆疊＋policy 方塊＋prefetch engine＋成本與頻寬小圖）；Cake Fig.1（把它作用的階段在流程中上色）與 Fig.2（前段算、後段載、中間會合）；CachedAttention Fig.6（讀取流與執行流兩條軌、每層一個方塊、三個面板對照）與 Fig.9（job queue 上的 look-ahead 預取窗與淘汰窗，畫 ② 時借）；LMCache Fig.5（快取層夾在引擎與儲存之間）。CacheFlow Fig.1 在 token × layer 格子上畫還原切分，只在要解釋 Cake 會合點時才需要。

**圖 3｜方法流程**（§3；必備；方塊流程圖，用 TikZ 或 figure-spec skill）
- **主流程**（左到右）：兩條 trace → 前處理（含「用 A 欄位驗算 B 欄位」的單位斷言）→ 模擬器（成本常數來自 H200 微基準，標出是哪幾個）→ 同一條 trace 跑四種臂：底／對手／免費訊號／oracle → 指標（TTFT，含與不含排隊兩種）→ 事先登記的門檻（δ\*、5%、15%）→ 判定字。
- **回饋迴路**（畫在下方）：抽一組設定在 H200 harness 上實跑 → 實測÷模擬、排序一致、正負號一致 → 「模擬可以用來排序與判定」。登記檔的 commit 畫在「看到數字之前」。
- **帶走**：判定是機械的（門檻先定好）；模擬有被 GPU 檢查過。
- **借鏡**：ORION Fig.1（§3 一開頭就是 High Level Workflow）；VGO Fig.3（Generalized methodology）；2609.16215 Fig.1（三步驟的 study overview）；Wall §2 先說明「oracle-driven trace-based simulation」（trace → 可設定的 oracle → 排程）。

**圖 4｜模擬驗證**（§3）
- **內容**：主面板是散點，x＝模擬、y＝實測，畫 y＝x 線與 ±band；小面板是模擬與實測的排序對照（連線）。
- **帶走**：絕對值有偏差，但排序與正負號一致，而判定需要的正是這兩樣。
- **借鏡**：CacheFlow Fig.10（predicted vs. measured，內文報「多少比例落在 10% 內」）；ORION Table 3（誤差旁附平均實測值，讓讀者知道誤差相對有多大）。
- **注意**：「TTFT 不含排隊」寫進圖說。

**圖 5｜寫入時的預知**（§4；頁數不夠時第一個刪）
- **內容與形式**：forest plot。每個設定一個點＝「依位置決定」減「延後決定」的差，附 CI，依差值排序；灰色帶＝±δ\*。
- **帶走**：全部落在噪音帶裡。
- **借鏡**：CacheFlow Fig.2 的 paired gap 圖，圖說直接寫「positive means … faster」，正負號的意義不必猜。

**圖 6｜回來時的上界與免費訊號**（§5，雙欄寬）
- **面板**：(a) agent、(b) chat：每種回來時 oracle 一組長條，分層（CPU、SSD 級、NFS 級），門檻線同圖 1。(c) 打字提示拿到 oracle 改善的比例：跨設定的 CDF 或 box，加 1.0 參考線。(d)（可選）截尾長度 H 的敏感度：丟棄 oracle 的上界隨 H 的曲線。
- **帶走**：agent 天花板低；chat 天花板高，但免費訊號拿走大部分；丟棄的上界大半來自 trace 被截尾。
- **借鏡**：2609.28870 Fig.2（14 種策略和 Belady 放同一張）；FOO Fig.17（每條 trace 一條 miss-ratio 曲線加 OPT 的界）；CacheFlow Fig.4（CDF）。2609.28870 附錄提醒：只在幾個容量點比較，會藏住交叉與非單調，所以敏感度畫掃描線，不給單點。

**圖 7｜預知什麼時候值得**（§7）
- **內容**：熱圖，x＝慢層頻寬、y＝CPU 容量，格值＝「知道誰不回來」的 oracle 改善；畫 5%、15% 等高線；每條 trace 一個面板。
- **帶走**：值得做的區域小、有名字、可以檢驗。
- **借鏡**：2609.16215 Fig.5（policy × workload 的格子，框出最佳格）；Cake Table 3（硬體 × 頻寬的格子）；To Offload（ICPE'26，見 STYLE.md）把 crossover 寫成 Lemma。我們若也能寫成命題，這張熱圖就是它的驗證圖。

**圖 8｜不用預知的收益**（§8）
- **(a) INT4**：每層一組長條（BF16 對 INT4 的還原時間），旁附品質差的小表或標註（ε 判定）。
- **(b) MPS**：前緣圖。x＝背景吞吐比、y＝前景 TTFT 加速；stream 基線畫成一個點；各 MPS 等級連成曲線；被支配的點淡化。
- **帶走**：INT4 的收益來自位元組變小；MPS 是取捨的旋鈕，不是免費。
- **借鏡**：2609.16215 Fig.9（cost 對 P99 TTFT 的前緣，被支配點淡化）；Cake Table 6（16／8／3-bit 各一欄）。

---

## 3. 表的慣例

**表 1｜三種預知的分類**（§3）
- **欄位**：種類｜決定的時刻｜預測什麼｜影響哪個決策｜我們的 oracle｜免費訊號或對手｜具名前作（誰在預測這個）｜節。
- **借鏡**：Wall Fig.11 的「模型 × 技術」矩陣（一格是一個設定，例如 branch predict：none／…／perfect）。這張表就是 README §1.2「被檢驗的假設要具名」的落地版。
- **規則**：「前作」欄只填讀過內文的；只讀過摘要的標〔摘要〕。

**表 2｜設定**（§3；三個區塊合成一張）
- **(a) 平台**：GPU、driver／CUDA、PyTorch、harness 與 vLLM 版本、attention backend、模型 checkpoint、KV dtype、INT4 方法、batch、prefix caching、MPS 設定（CHECKLIST E3）。
- **(b) 層**：層｜怎麼實現（真實／限速模擬）｜頻寬（量測值）｜怎麼量的｜用在哪些實驗。
- **(c) trace**：trace｜類型｜session 數｜請求數｜回來間隔中位數與 p90｜時間跨度｜過濾與截尾｜授權；註腳寫明單位怎麼交叉驗算。
- **借鏡**：Cake Table 1（層數、每 token 的 KV 大小、KV heads）與 Table 2（Physical Config｜Bandwidth）；HCache Table 2（GPU｜HBM｜FLOPS｜傳輸速度）；2609.28870 Table 1（trace 特性）與 Table 3（和公開 trace 比較）；2609.16215 Table II（principal settings：tiers、policies、baselines、workloads 一張表）。

**表 3｜完整計分表**（§9）
- **欄位**：#｜加在 Cake 上的東西｜要預知嗎｜臂（oracle／免費訊號／實測）｜值與單位｜門檻｜判定｜證據〔模擬〕或〔實測〕｜節。
- **判定字詞**：固定 4–5 個，在表說裡事先定義，例如「<δ\*：量不到」「<5%：天花板低」「5–15%：停下來問」「≥15%：值得」「取捨」。對照：2609.16215 Table IV 的判定是自由文字（Holds／Does not hold as written／Holds, scenario-dependent），好讀但無法機械比對。
- **沒過的列也放**（CHECKLIST E11 的選擇性報告）。Wall 在作者註裡說：完整結果放附錄，挑選呈現的結果時盡量不加評論。我們的附錄不出版，所以完整表放 artifact，正文表註明「完整列見 artifact」。
- **每列指回一節**：2609.28870 Table 5 是「特性摘要＋節號」的寫法。

**表 4｜效度威脅**（§9）
- **欄位**：威脅｜類別（construct／internal／external／conclusion；empirical SE 論文常見的分法〔慣例，這次沒有查證出處〕）｜對上界的偏向（高估／低估／不定）｜我們做了什麼｜剩下的風險。
- **「偏向」欄是 limit study 特有的**：審稿人想知道「上界會不會其實更高」，例如單一決策的 oracle 可能低估聯合上界。每列的方向要靠證據判斷，不要在這裡猜。
- **借鏡**：Wall 的結論逐條列「樂觀的假設」，並說任何一條都可能讓收益少掉三分之一；2609.16215 §VI 以「Several limitations bound these claims」起頭接條列；ICPE'25 outages 有專節「Limitations and Validity」（見 STYLE.md）。

**表 5｜相關研究定位**（§10）
- **欄位**：工作｜miss 時的底（重算／載入／Cake）｜層｜預知什麼｜有沒有 oracle 上界｜workload（真實或合成）｜指標｜有沒有 GPU 驗證。最後一列是本文；≤8 列，格子用 ✓、✗、– 或短詞。
- **組織方式**：2609.28870 的 related work 沒有表，用 5 段 run-in（例如「Prefix caching and storage.」）。我們頁數緊，建議表加 4 段 run-in：預測型 KV 管理、還原機制、分層與放置、trace 特性與 limit study。
- **規則**：只填讀過內文的格；照抄 NOVELTY_CHECK 表「讀到哪」那一欄的標記。

---

## 4. 負面結果怎麼寫

| # | 句型（中文） | 英文來源的句型 | 出處 |
|:--|:--|:--|:--|
| 1 | **主詞是 oracle**：「即使是完美的預測器，在 ⟨trace⟩ 上對 Cake＋固定規則的 TTFT 改善也不超過 ⟨X⟩%；任何做得出來的預測器只會更少。」 | 「a future-reading oracle that is an upper bound no real system can build」 | 2609.16215 §V-E |
| 2 | **範圍貼在句子上**：「在 H200、單卡、7–8B、限速模擬層的設定下，⟨結論⟩。」寫一次就好，不要每句都 hedge | 「at this scale」、「as shipped」、「at the default cache size」 | 2609.16215 |
| 3 | **先給機制，再給含意**：「上界小，是因為 ⟨機制⟩；所以 ⟨設計上的含意⟩。」 | 「Finding & Implication:」段落；「making recency very crucial and rendering frequency-based signals ineffective」 | 2609.28870 §4 |
| 4 | **分清「追不上」與「沒空間」**：「這不是線上策略追不上 oracle，而是 oracle 本身就離底很近。」同時處理我們和 2609.28870 的方向差異 | 2609.28870 寫「No current 'fancy' mechanism fundamentally solves…」，並配上和 Belady 之間的大差距；我們是反面 | 2609.28870 §3.2 |
| 5 | **報整條梯子**：「只報 oracle 會高估做得到的收益，所以我們同時報免費訊號拿到的比例。」 | 「A study using the Perfect model alone would lead us down a dangerous garden path」 | Wall 1993 §3.1 |
| 6 | **具名做法對照判定**：「『⟨具名做法⟩』對 TTFT 不成立；它對 ⟨SSD 寫入量／壽命⟩ 的好處不在本文的量測範圍。」 | 「Does not hold as written」；「The two elements of the original recommendation that do not survive」 | 2609.16215 Table IV、結論 |
| 7 | **例外寫成條件**：「只有在 ⟨A⟩、⟨B⟩、⟨C⟩ 同時成立時，預知才值 ≥15%。」 | crossover 條件寫成 Lemma | To Offload（ICPE'26，見 STYLE.md） |
| 8 | **取捨不寫成收益**：「前景延遲的改善是用背景吞吐換來的；它是一個旋鈕，不是免費的加速。」 | 「bought at a latency premium the SLO does not accept」；「Throughput is therefore the wrong lens.」 | 2609.16215 |

**引言或結論可借的一句**：Wall 認為這份研究的負面結果至少和正面結果一樣重要，而且數字應該當路標（guideposts），不是命令（mandates）。中文：「這些上界告訴設計者哪裡不必蓋預測器。」

**要避免的**：
- 「預測沒用／沒意義」：全稱句，而且我們自己就有反例（知道誰不回來那一格）。
- 「我們試了 X，結果不好」，卻沒有機制也沒有上界：讀起來就是失敗。主詞改成 oracle 或上界。
- 只報落在門檻下的格子，或看到數字之後才定門檻（Heiser 列的選擇性報告罪狀）。
- 「negligible」「marginal」卻不附數字、不附 δ\*；或把沒量的目標（SSD 壽命、寫入量）一起否定。
- 「Surprisingly」「Interestingly」堆疊，或結論前一串 may／might／could（paper-writing 的機械檢查會抓 flourish）。
- 模擬與實測放在同一句或同一條軸，卻沒有標示。

---

## 5. 來源

**標記**：[全文]＝讀了 HTML 正文（WebFetch，限制見開頭）；[全文\*]＝PDF 自己解壓文字，只讀了標題、圖表說明與幾段內文。「圖怎麼畫」一律是從圖說推斷。

**limit study 與 oracle 上界**
- [全文] *When Fancy Eviction Fails: Rethinking Cache Replacement For LLM Prefix Reuse*（arXiv 2609.28870 v2，2026-10-01，沒有 venue）— https://arxiv.org/html/2609.28870v2 。用到：章節安排、「Finding & Implication」與「Takeaway.」段落、Fig.1 總覽、Table 1／3／5、附錄的範圍聲明、related work 的 run-in。§6 Discussion 的內文沒有完整讀到。
- [全文] *Where Should the KV Cache Live? Placement Policies Across GPU, CPU, and SSD for Long-Lived Sessions*（arXiv 2609.16215 v1）— https://arxiv.org/html/2609.16215 。用到：Fig.1／4／5／9／10、Table II／III／IV、§VI 的限制段、範圍限定的措辭。
- [全文\*] D. W. Wall, *Limits of Instruction-Level Parallelism*, DEC WRL Research Report 93/6, 1993-11 — 課程鏡像 https://www.csd.uoc.gr/~hy425/2012f/lectures/wall-ilp.pdf 。讀了作者註、§1–2、Fig.11（七個模型）、§3.1、結論。
- [全文] D. S. Berger, N. Beckmann, M. Harchol-Balter, *Practical Bounds on Optimal Caching with Variable Object Sizes*（POMACS 2(2)，2018；arXiv 1711.03709）— https://ar5iv.labs.arxiv.org/html/1711.03709 。用到：Fig.1、Fig.14–17、上下界的不等式鏈、評估各小節的標題。

**KV 還原與卸載的圖**
- [全文] Cake：*Compute Or Load KV Cache? Why not Both?*（arXiv 2410.03065，ICML'25）— https://arxiv.org/html/2410.03065
- [全文] *CacheFlow: Efficient LLM Serving via Automated 3D-Parallel KV Cache Restoration*（arXiv 2604.25080）— https://arxiv.org/html/2604.25080
- [全文] *Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention*（ATC'24；arXiv 2403.19708）— https://arxiv.org/html/2403.19708
- [全文] *Fast State Restoration in LLM Serving with HCache*（EuroSys'25；arXiv 2410.05004）— https://arxiv.org/html/2410.05004v1
- [全文] *LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference*（arXiv 2510.09665）— https://arxiv.org/html/2510.09665

**ICPE'26（STYLE.md 以外補的兩篇）**
- [全文\*] *ORION: Integrated Runtime Modelling for Predicting Deep Learning Training Time*（ICPE'26，DOI 10.1145/3777884.3797001）— https://icpe2026.spec.org/preprint/ORION_Integrated_Runtime_Modelling_for_Predicting_Deep_Learning_training_time.pdf 。用到：Fig.1 workflow、附平均實測值的誤差表、leave-one-GPU-out 的圖。
- [全文\*] *Variability-Guided Performance Optimization*（ICPE'26，DOI 10.1145/3777884.3796994）— https://icpe2026.spec.org/preprint/Variability-Guided_Performance_Optimization.pdf 。用到：面板上標 n／mean／median／sd 的分佈圖（Fig.1）、Fig.3 methodology、「因素｜緩解｜變異降低」的表（Table 3）。
- ICPE 2026 接受名單（拿來挑上面兩篇）：https://icpe2026.spec.org/accepted-papers/

**沿用 STYLE.md、這次沒有重讀**：SweetSpot（2602.05695）、To Offload or Not To Offload（2504.15162）、ICPE'25 outages（2501.12469），連結見 STYLE.md 的 Sources。

**查過但沒有採用**：LRB（NSDI'20，*Learning Relaxed Belady for Content Distribution Network Caching*）與 ICPE'26 *Benchmarking the Overhead of Distributed Tracing Agents*，兩篇都只看到搜尋摘錄，沒讀全文。
