# ICPE 2027 投稿前檢查清單（PREFLIGHT）

> 2026-10-30 編。官方頁面當天用 WebFetch／curl 重讀，來源代號 [U1]–[U14] 見文末。acm.org、libraries.acm.org、authors.acm.org 當天全部回 403，所以 ACM 的規則只有二手來源。self-review，不是 cross-model review。
> 檢查對象：`PAPER.md` v2（11:52 讀取；mtime 11:37:03，md5 6da6279f）。v3 正在改寫，「狀態」欄記的是 v2；「待 v3」＝要等 v3 才能判斷。結尾又查了一次，見〈結果〉。
> 標記：**必須**／建議；狀態 ✓ 符合、✗ 不符合、待 v3；規則後的 ✓[Ux]＝10-30 重新核對過、NOT_FOUND＝頁面沒寫；〔判讀〕是我的解讀，〔算術〕是由量到的值算出。

## 0. 和 10-28 的 CHECKLIST.md 比，有哪些變動

1. HotCRP 現在列出**全文**截止：兩個投稿入口都是 Tue Nov 17 2026 06:59:59 AM EST（＝11/16 AoE＝台灣 11/17 19:59:59〔算術〕）。註冊截止不變 [U3]。
2. 新讀到 **Double-Anonymous FAQ** [U5]：PDF 不能有所屬機構連結、個人網站、GitHub／GitLab 個人頁、**致謝**；可以投 arXiv，但**題目要和投稿版「sufficiently different」**；投稿時要申報與 PC 的利益衝突。
3. Emerging Research Track 這次直接讀到原頁 [U6]（10-28 是轉引）：≤6 頁，參考文獻與匿名附錄不算；雙盲；明文歡迎「negative results that provide learning for the community」；收進 companion proceedings。AE Track 仍是「Details TBA」[U7]。
4. FreeInference：在 HF 資料卡上確認是 CC-BY-4.0，**卡片要求引用論文**；HF 現在的 head 是 16d3c04d…，和我們固定用的版本相同 [U10]。Mooncake：repo 是 Apache-2.0；README 寫這兩條 trace 是「sampled from one hour of online request data」[U11]。
5. acmart：CTAN 版本 2.20（2026-08-16），GitHub 上 2.21（2026-10-29）。文件寫明：超過 2 頁的論文必須有 CCS concepts 與 keywords；每張圖要有 `\Description` 或 alt 文字；圖要能用灰階讀 [U9]。
6. 沒有變的：各項日期、10 頁上限、兩種論文類型、審查標準、AI 不能列作者、ORCID、APC。ACM 新版 AI 政策的二手引用 [U8] 內容和 10-28 讀到的一樣，ACM 原頁仍是 403。

## (a) 投稿規則

| # | 規則（驗證） | 必須／建議 | v2 狀態 | 在哪改 |
|:--|:--|:--|:--|:--|
| a1 | 截止：摘要 11/9、全文 11/16、通知 2027-01-25、camera-ready 03-12，**全部 AoE** ✓[U1]；HotCRP：註冊 Nov 10、投稿 Nov 17，都是 06:59:59 AM EST ✓[U3] | 必須 | — | 見 (i) |
| a2 | 兩個入口 `Regular-Research-Papers`／`Empirical-Experience-Reproduction-CaseStudy-Papers`，註冊時就要選 ✓[U3]。**已定（使用者 10/30）：EERCS。**Regular＝「original research content with a compelling evaluation」；EERCS＝「empirical evaluation, real-life experiences, reproductions, or case studies」✓[U2] | 必須 | ✗ 還沒選 | 使用者決定。〔判讀〕limit study 兩邊都說得通，EERCS 的「empirical evaluation」比較貼近 |
| a3 | Research Track 明文歡迎的只有「positive and negative reproduction studies」✓[U2]；一般的負面結果明文歡迎的是 ERT（2/1 截止，≤6 頁）✓[U6][U1]；被 Research Track 拒後能不能轉投 ERT：NOT_FOUND | 建議 | 待 v3 | 〔判讀〕摘要寫成「地圖＋上界」，不要寫成單純的負面結果 |
| a4 | 至少 3 位 PC 審；看 novelty、relevance、technical quality、scientific soundness、clarity ✓[U2]。主題清單有「LLM inference pipelines, and agentic workflows」以及「LLM serving, batching, routing, caching」✓[U4] | 建議 | ✓ | 摘要用這些關鍵詞（會影響 bidding） |
| a5 | 接受後強烈鼓勵投 artifact，可以拿 ACM badge ✓[U2]；AE Track 細節 TBA ✓[U7]；投稿時要不要附 data availability statement：NOT_FOUND | 建議 | — | 接受後再做 |
| a6 | ORCID：接受論文的作者都要提供 ✓[U4]。APC：全面 OA；機構在 ACM Open 內就免費，否則 2027 年 USD 500（會員）／750；豁免很少 ✓[U4]。你們學校在不在名單：NOT_FOUND（名單頁 403） | 必須（接受後） | — | 現在先申請 ORCID；問學校圖書館 |
| a7 | 語言、出席規定、一稿多投：ICPE 頁面都沒寫，NOT_FOUND [U2][U4]。〔判讀〕寫英文；審查期間（到 1/25）不投別處（10-28 §F 的 ACM 規則這次因 403 沒能重驗）。地點 Gothenburg、會期 5 天 ✓[U13]；確切日期首頁沒寫 | 必須〔判讀〕 | ✗ 目前是中文 | v3 定稿後翻譯 |

## (b) 雙盲與匿名化

| # | 規則（驗證） | 必須／建議 | v2 狀態 | 在哪改 |
|:--|:--|:--|:--|:--|
| b1 | 不寫作者名 ✓[U2]；不能有所屬機構連結、個人網站、GitHub／GitLab 個人頁 ✓[U5]；acmart 的 `anonymous` 選項會隱藏作者資訊，連作者人數也藏 ✓[U9] | 必須 | 待 v3 | LaTeX 版 |
| b2 | 審查版**不能有致謝**（人或計畫都不行）✓[U5] | 必須 | ✓ 沒有致謝 | — |
| b3 | 引用自己的舊作用第三人稱，不能透露身分 ✓[U2][U5] | 必須 | ✓ 沒有自引 | 如果引用 MI300X 階段的成果，也用第三人稱 |
| b4 | 程式與 repo 要匿名 ✓[U2]；PDF 不能連到會透露身分的 artifact ✓[U5]；要用哪個匿名服務：NOT_FOUND | 必須 | ✓ 沒有連結 | 真的要放連結，就照 10-28 §I 做乾淨的快照 |
| b5 | 可以投 arXiv，但題目要和投稿版明顯不同 ✓[U5]；投稿時申報與 PC 的利益衝突 ✓[U5]（PC 名單在 `/program-committee/`） | 必須 | ✗ 還沒做 | 11/9 前完成 |
| b6 | 匿名化的目標「not to make it impossible to guess」，而是避免審稿人「automatically aware」✓[U5]。〔判讀〕repo `s990093/paper-hierarchical-kv-state` 是公開的；v2 的內部代號（STAIR、G5b、BRIDGE-NO-GAP、E2E-NOGO、MPS-GPU-INVALID、X1／X5、T1／T1G）在 repo 裡一字不差，用 GitHub 搜得到 | 建議 | ✗ | 論文改用描述性的名稱（同時解決 c5）；或審查期間把 repo 改成 private（由使用者決定） |
| b7 | 〔判讀〕登記檔的 commit hash 和 repo 連結都會透露身分 | 必須 | ✓ 稿中沒寫 | 審查版只寫「先登記，看到數字後才判定」；camera-ready 再補連結 |
| b8 | PDF metadata 的 Author、Title 欄不能有名字（10-28 §C） | 必須 | 待 v3 | 最後的 PDF 用 Python 讀 metadata（這台機器沒有 pdfinfo） |

## (c) 篇幅與格式

| # | 規則（驗證）／現況 | 必須／建議 | v2 狀態 | 在哪改 |
|:--|:--|:--|:--|:--|
| c1 | **≤10 頁，含圖表，不含參考文獻與附錄**；附錄不出版，正文要能獨立讀懂；camera-ready 可以多 2 頁 ✓[U2] | 必須 | 待 v3 | 要在 LaTeX 實際排版後才知道 |
| c2 | 規定是 ACM 雙欄的會議論文格式 ✓[U2]；頁面沒寫範本名（NOT_FOUND）。〔判讀〕用 `\documentclass[sigconf,anonymous,review]{acmart}`（`review` 會加行號）；超過 2 頁的論文必須有 CCS concepts 與 keywords ✓[U9] | 必須 | ✗ | v2 是中文 markdown，用 matplotlib 排成 A4 PDF。這台機器 `command -v xelatex pdflatex lualatex latexmk pandoc` 都找不到，`/ssd7` 也沒掛載 → 改用 Overleaf，或 3090 那台的 TinyTeX |
| c3 | v2 有 9 張圖、9 張表、7,813 個漢字〔算術〕；STYLE_REF §1 給圖表的篇幅只有約 2.8 頁 | 必須 | 待 v3 | 照 STYLE_REF §1「超過頁數時依序砍」 |
| c4 | 內部用的東西：閱讀指引、13 個 ⏳、「待決事項」與「數字出處」兩節、EVIDENCE 列號（如 [B1]）、連到 AUDIT.md／EVIDENCE.md 的連結、L400 那句「Claude subagent 稽核」 | 必須 | ✗ | 翻成英文版前全部拿掉 |
| c5 | X1、X5、T1、T1G、T3、O-PF、R1／R2、b16 等代號在正文沒有定義 | 必須 | ✗ | 改成描述性名稱，或在第一次出現時定義 |

## (d) 審稿人會抓的量測問題（CHECKLIST E1–E12，加上 AUDIT.md 還沒處理的項目）

| # | 會被問到的 | v2 狀態 | 必須／建議 | 在哪改 |
|:--|:--|:--|:--|:--|
| E1 | 指標與分佈 | headroom 用的是回來請求的**平均** TTFT，不含排隊 ✓；含排隊的 8.1／4.7% 有寫 ✓；**沒有 P50／P90／P99** ✗ | 建議 | `x1_sweep.csv.gz` 已經有 `median_ret`、`p90_ret` 欄，GPU 的 `gpu_requests.csv.xz` 有每個請求的 `ttft`。先寫進 EVIDENCE 再引用 |
| E2 | baseline 夠不夠強 | 沒解釋為什麼不用 vLLM 內建的 offloading、LMCache、SGLang HiCache ✗；harness 的重算比 vLLM 慢有寫（表 8）✓ | 建議 | §3 補一段 |
| E3 | 設定有沒有寫完整 | 表 2 有 GPU、vLLM、torch、模型 ✓；缺 CUDA／driver、各實驗的 attention backend、INT4 的量化細節、prefix caching、batch、MPS 參數、主機的 CPU／DRAM／NUMA ✗ | 建議 | 表 2（數值從 EVIDENCE、`hw_h200.json` 抄） |
| E4 | 重複次數與統計 | δ\*、雜訊、6 次重複、CI 都有 ✓；沒說模擬是確定性的、只跑一次，也沒說比值怎麼彙總 ✗ | 建議 | §3 |
| E6 | 工作負載 | FreeInference 前 6 週、截尾 H 都有寫 ✓；沒寫 Mooncake 只是「一小時」的抽樣 [U11] ✗；FreeInference 是 o200k_base tokenizer 的 16-token block [U10]，怎麼換成 512-token block 沒寫 ✗ | **必須**（換算方式）／建議（時長） | §2、表 2。先從原始資料核對時長與換算方式，寫進 EVIDENCE（CLAUDE.md 規則 6） |
| E7 | 有損方法的品質 | INT4 有 192 題和 CI ✓；沒寫任務名稱與 seed ✗ | 建議 | §8.1 |
| E10 | 事先登記與偏離 | 有寫「先 commit 再看數字」✓；**沒有偏離清單** ✗ | 必須 | 見 e3 |
| E13 | 門檻怎麼定的 | 沒說 5%／15% 的理由 ✗ | 建議 | §3 加一句，理由照登記檔 |
| E5、E8、E9、E11、E12 | 模擬的可信度（B6、常數出處）、規模、共用叢集的污染、選擇性報告（表 7 列出沒過的）、泛化（題目寫 on H200） | ✓ | — | — |
| AUDIT #33 | 文獻上的說法 | 還是 NOT_VERIFIED（待決 14）✗ | 必須 | 見 g4、g5 |
| AUDIT #17 | 圖 9 用哪一版數字 | sidecar 和 EVIDENCE 對 T1 的說法不一致（待決 5）✗ | 必須 | 先統一，再定圖說 |
| AUDIT #27 | S1 的「0 翻轉」 | 不判的 379 格、NOT_MEASURED 84 列有寫 ✓；沒寫「有 2 個組合因為雜訊沒判」✗ | 建議 | §6 |

AUDIT 的 3 個高嚴重度問題（#1「agent trace」、#2「很窄」、#3 常數出處），v2 正文都改好了 ✓；還留在圖裡的部分見 f4。

## (e) 誠實性

| # | 項目 | v2 狀態 | 必須／建議 | 在哪改 |
|:--|:--|:--|:--|:--|
| e1 | 每個數字都出自 EVIDENCE.md | ✓（AUDIT 從原始檔重算 41 組，沒有一個錯）；但有 4 個數字只出自 sidecar（表 4 的 LRU 命中率、36.8%、每格 30 對、≤1.3 pp）✗ | 必須 | 補進 EVIDENCE |
| e2 | EVIDENCE「不能寫的」23 條 | 逐條都避開了 ✓；第 2、6 條的措辭還在待決 ✗ | 必須 | v3 定稿後跑 `paper-claim-audit`（註明是同一模型家族，不是 cross-model） |
| e3 | 偏離登記的清單 | ✗。候選項目（要對照各登記檔，確認哪些真的算偏離）：SSD／NFS 常數沿用 MI300X 的（使用者的選擇）；O-PF 的對手只有 {LRU, LRU+RPF}，對 LRU2 的結果只報告；STAIR 只當敏感度；A4' 走的是對照路徑，不是 H200 的登記路徑；E2E-C 有 8 格 NOT_MEASURED | 必須 | §3 或表 8 加一張小表 |
| e4 | NOT_MEASURED 要照寫：X5 在 FreeInference 16–64 GiB、淘汰順序的免費訊號、聯合 oracle、真實的打字時間 | ✓ | 必須 | 翻譯時不能變成「<15%」 |
| e5 | 合成的、模擬的要標清楚：打字的提前量是合成的；SSD／NFS 是限速模擬層 | 正文 ✓，圖 ✗（f4） | 必須 | 見 f4 |

## (f) 圖表

| # | 規則／現況 | 必須／建議 | v2 狀態 | 在哪改 |
|:--|:--|:--|:--|:--|
| f1 | sigconf 的版面寬 506 pt（7.0 in）、欄寬 241 pt（3.34 in）〔算術，用 acmart 的 sigconf 版面參數 [U9] 算〕。make_figs.py 畫的圖寬 5.4–7.4 in、字 7–9.5 pt：放成欄寬，7 pt 會縮成 3.2–4.4 pt；放成版面寬，7.2–7.4 in 的圖縮成約 6.8 pt〔算術〕。≥7 pt 是本清單自訂的標準，ACM 的字級規定 NOT_FOUND | 必須 | ✗ | `make_figs.py` 的 figsize：單欄圖以 3.34 in 重畫；雙欄圖 ≤7.0 in |
| f2 | 要能用灰階讀，不能只靠色相區分 ✓[U9]。換成灰階〔算術，8-bit luma〕：BLUE 107、AQUA 125、MUTED 135、ORANGE 137、BLUE_LIGHT 174。**fig5_int4 的 BF16（MUTED）和 INT4（AQUA）只差色相** ✗；fig4、fig7 的橘色與灰色靠實心／空心區分 ✓ | 必須 | ✗ fig5 | fig5 加斜線，或改用深淺不同的顏色 |
| f3 | PNG 是 200 dpi（1440 px ÷ 7.2 in）；fig1／fig2 正在改成 PNG | 建議 | ✗ | 改輸出 `savefig(...pdf)`；示意圖維持 SVG，再轉成 PDF |
| f4 | 圖裡的文字要和正文一致：fig2_method.svg 寫「H200-measured cost constants」，trace 只列 chat、agent 兩條（AUDIT #3、#12）；fig3 的軸寫「H200 constants (T1)」（make_figs.py L357）；fig9 寫「FreeInference (agent trace) … H200 constants」（L898）；圖裡有 T1／T3／STAIR 這類代號 | 必須 | ✗（fig1／fig2 待 v3） | fig1／fig2 正在重畫；fig3、fig9 改 make_figs.py |
| f5 | 圖說第一句寫結論 ✓（9 張的圖說都以粗體結論開頭）；每張圖要有 `\Description` 或 alt 文字，沒有的話 acmart 會警告 ✓[U9] | 建議 | 待 v3 | LaTeX 版 |

## (g) 授權與引用

| # | 項目 | 必須／建議 | v2 狀態 | 在哪改 |
|:--|:--|:--|:--|:--|
| g1 | FreeInference：CC-BY-4.0 ✓[U10]；卡片寫「If you use this dataset, please cite the paper below」→ *From Requests to Sessions: A Large-Scale Characterization of Human-Driven Agentic Workloads*（Nixon, Tian, Zheng, Gunawi, Yang, 2026；BibTeX 是 @article，沒有 venue）✓[U10] | 必須 | ✗ refs.bib 沒有這篇 | 加引用，並寫明用了哪 6 週、HF revision 16d3c04d、授權 CC BY 4.0 |
| g2 | Mooncake：repo 是 Apache-2.0，README 請使用者引用 FAST'25 ✓[U11]；refs.bib 有 `mooncake2025` ✓ | 必須 | ✓ | trace 的 release URL 放在註腳（不會透露身分） |
| g3 | Llama-3.1-8B-Instruct 用 Llama 3.1 Community License：散布模型或衍生物時，要附授權、標示「Built with Llama」、附 Notice 檔 ✓[U12]。Qwen2.5-7B-Instruct-1M 是 Apache-2.0 ✓[U12]；我們的 noDCA 版是改過的衍生物 | 必須（論文要引用模型）／建議（artifact） | 待 v3 | artifact 不放權重，只放 `make_nodca_model.py` |
| g4 | 只讀過摘要的文獻（NOVELTY_CHECK 標〔摘要〕或更弱）：PBKV 2605.06472、SAGA 2605.00528、CacheScout 2608.14624、CacheWise 2606.16824、Continuum 2511.02230、KVFlow（NeurIPS'25）、Ask the Tool 2609.18849、SARIA 2610.31169、Gandalf 2610.30791、CacheFlow 2604.25080（摘要，加上 10/01 讀過內文）；ICPE'25 *Better Memory Tiering*（只核對書目）；*From Requests to Sessions*（UNVERIFIED）。v2 靠它們撐的說法：「這些系統的對照組是沒命中就重算」（摘要、§1、§10、表 9）、「Ask the Tool 接近 oracle」、SARIA／Gandalf 的目標 | 必須 | ✗ | 投稿前讀內文；讀不到就刪掉，或改寫成「作者稱」 |
| g5 | Cake 的出處（refs.bib 的 `cake2025` 寫 ICML／PMLR；AUDIT #33 標 NOT_VERIFIED）；Dhakal et al. CHI'18 的 51.56 WPM（劍橋的新聞稿寫約 52 WPM [U14]，論文的表格沒讀） | 必須 | ✗ | 跑 `citation-audit` |
| g6 | refs.bib 裡完全沒有（用標題 grep）：Fancy Eviction、Ask the Tool、PBKV、SAGA、CacheScout、CacheWise、Continuum、KVFlow、SARIA、Gandalf、Better Memory Tiering、Dhakal、From Requests to Sessions | 必須 | ✗ | 建新的 bib，每一筆都對 arXiv／DOI 核對 |
| g7 | 別組的近題論文 HotCarbon'26 *Energy Characterization of KV Cache Offloading Under Agentic Workloads*（`hotcarbon2026`），v2 的相關研究沒提；9/14–10/22 有 6 篇相關預印本（NOVELTY_CHECK L93） | 建議 | ✗ | §10 加一句；11/9 前再查一次 arXiv |

## (h) AI 使用揭露

| # | 規則／現況 | 必須／建議 | v2 狀態 | 在哪改 |
|:--|:--|:--|:--|:--|
| h1 | ICPE：AI 不能列為作者；照 ACM Policy on Authorship 揭露 ✓[U4]。ACM 新版（二手 [U8]，10-30 重讀內容相同）：研究過程用到 AI，要「described in detail in the methods section」；只用來幫忙寫作，「ACM no longer requires the disclosure」；所有掛名作者都要負責。ACM 原頁 403，原文與改版日期 NOT_FOUND | 必須 | ✓ 揭露段放在 §3（方法） | — |
| h2 | 揭露段沒寫模型名稱與版本 ✗；「作者人工檢查了 oracle、模擬器、量化與計時程式」這句還在等使用者確認（待決 10）✗。寫「Claude Code」不會透露身分 ✓ | 必須 | ✗ | 作者真的讀過這幾段程式才能留這句；寫出用了哪些 Claude 模型 |

## (i) 時程（今天 10/30）

**11/9 AoE 前（台灣 11/10 19:59:59）在 HotCRP 註冊**：① 必須：選 EERCS 入口（a2，已定）；② 框架已定為 (C) 組合地圖（使用者 10/30），寫好英文題目與摘要——bidding 看的是這版摘要；③ 必須：填作者、所屬機構、與 PC 的利益衝突；如果要上 arXiv，題目要另取（b5）；④ 建議：再查一次 arXiv（g7）。

**11/16 AoE 前（台灣 11/17 19:59:59）交全文**：① v3 翻成英文，用 acmart sigconf,anonymous,review，≤10 頁，含 CCS 與 keywords（c1–c3）；② 拿掉內部用的東西，代號改名或定義（c4、c5、b6）；③ 修圖（f1、f2、f4）；④ 偏離清單、sidecar 數字補進 EVIDENCE、E6 的 block 換算、待決 1–14 全部定案（e1、e3）；⑤ 讀文獻內文、補 bib，跑 `citation-audit` 與 `paper-claim-audit`，並註明不是 cross-model（g1、g4–g6）；⑥ 作者讀過核心程式後，定稿 AI 揭露（h2）；⑦ 對最後的 PDF 再掃一次匿名（用〈結果〉的 pattern，加上 metadata）。以上全部必須。**接受後**：ORCID、APC、AE（細節 TBA）。

## 結果：匿名化掃描（只讀，沒有改任何檔）

- **時間**：PAPER.md 第一次在 11:52:57 讀（v2；mtime 11:37:03，md5 6da6279f）。寫完本檔後在 12:02:23 再查：PAPER.md 已在 12:01:54 改成 v3 的中途版（465 行，md5 39a60390），build_pdf.py 也在 12:01:42 改過；另外新出現 `figs/diagrams.py`。同一組 pattern 重掃這些檔，仍是 0 筆。
- **pattern**：`/work/`、`ChihChung`、`hungwei`、`s990093`、`github.com`、`Claude-Session`、`claude.ai`、email（`@….com|edu|tw|org`）、作者 email 的前綴、`hgx-cn`、`/ssd7`、`/home/`、`/tmp/`、`Tiara`、`paper-hkv`、`paper-hierarchical`、`H.W.`、`Co-Authored`。
- **掃過的檔**：PAPER.md、build_pdf.py、figs/make_figs.py、figs/*.sources.json（7 個）、figs/fig1_system.svg、figs/fig2_method.svg、figs/specs/*.json。**命中 0 筆。** 7 張 PNG 都沒有 tEXt／iTXt／zTXt（make_figs.py L158 設了 `metadata={"Software": None}`）；兩個 SVG 都沒有 `<metadata>`。
- **PAPER.pdf**：第一次查時不存在；12:02 出現（884,207 B，15 頁 A4）。用 Python 讀原始位元組：只有 /Title（v3 的中文標題）、/Creator「build_pdf.py (matplotlib)」、/Producer「Matplotlib pdf backend v3.11.2」，**沒有 /Author** ✓；原始位元組裡上面那組 pattern 0 筆（正文是 Type 3 字形，無法這樣搜，所以正文是靠掃 PAPER.md 檢查的）。這份是中文 A4 預覽，不是投稿版。
- **v3 中途版（12:02）的變化**：新增「名詞表」（c5 部分解決）；⏳ 改成 9 個〔待決〕；正文仍有 13 處 STAIR／G5b／BRIDGE／X1／X5／T1G 之類的代號；「對照表」「數字出處」兩節標了「內部用，投稿版刪除」（c4 仍待處理）。L138 寫「FreeInference 以 CC-BY-4.0 釋出（授權待核對原始頁面）」→ 這次已核對 ✓，但 *From Requests to Sessions* 仍沒有引用（g1 ✗）。§3.4 L197 寫偏離「列在匿名 artifact（審查期間不附連結）」→ 審稿人會看不到，建議至少把主要偏離寫進正文（e3）。§3.5 的 AI 揭露仍沒寫模型版本，人工檢查那句仍待決（h2 ✗）。v3 引用的 `figs/fig1_system.png`、`figs/fig2_method.png` 在 12:02 還不存在（只有 .svg）。
- **不算洩漏、但可以被搜尋到**：① 7 個 sidecar 共有 166 個 run_id，以及 repo 相對路徑（`results/…`、`docs/h200/icpe/…`），這些在公開 repo 裡一字不差 → 審查期間的 artifact 不要原封不動放 sidecar；② v2 的內部代號（b6）；③ make_figs.py L15 提到 `.claude/skills/figure-spec/…`，只透露用了什麼工具、不透露身分，和 AI 揭露一致。
- **不是匿名問題、但投稿版必須拿掉**：PAPER.md 連到 AUDIT.md／EVIDENCE.md 的連結，以及 L400 的「Claude subagent」稽核說明（c4）。

## 來源（2026-10-30 讀取）

[U1] https://icpe2027.spec.org/important-dates/ ・[U2] https://icpe2027.spec.org/tracks-and-submissions/research-paper-track/ ・[U3] https://icpe2027.hotcrp.com/deadlines 、https://icpe2027.hotcrp.com/ ・[U4] https://icpe2027.spec.org/call-for-contributions/ ・[U5] https://icpe2027.spec.org/double-anonymous/ ・[U6] https://icpe2027.spec.org/tracks-and-submissions/emerging-research-track/ ・[U7] https://icpe2027.spec.org/tracks-and-submissions/artifact-evaluation-track/
[U8] https://2027.sigcse-ts.acm.org/info/policies-ai （ACM 政策的逐字引用；原頁 https://www.acm.org/publications/policies/new-acm-policy-on-authorship 回 403）・[U9] https://ctan.org/pkg/acmart 、https://raw.githubusercontent.com/borisveytsman/acmart/master/acmart.dtx （anonymous、Description、灰階、CCS／keywords、sigconf 的版面參數）
[U10] https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace （README 原文與 `/api/datasets/…`：license、citation、sha）・[U11] https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release （README；GitHub API 回報 license Apache-2.0）
[U12] https://huggingface.co/api/models/meta-llama/Llama-3.1-8B-Instruct 、https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct-1M 、https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_1/LICENSE ・[U13] https://icpe2027.spec.org/ ・[U14] https://www.cam.ac.uk/research/news/what-makes-a-faster-typist
