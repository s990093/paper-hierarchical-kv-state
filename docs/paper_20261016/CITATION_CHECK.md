# 引用核對紀錄（main.tex 英文重寫版，2026-10-17）

> 回到 [README](README.md)。`refs.bib` 51 筆，全部由三個獨立的核對 agent 逐篇讀原文（PDF／出版社頁／arXiv）確認書目資料，並檢查「論文裡怎麼引用它」有沒有被原文支持。
> **審查性質**：核對 agent 沒參與寫作，但和寫作者是同一個模型（Claude），**不是 cross-model review**（CLAUDE.md §5）。DBLP 被反爬蟲擋住，書目改用 Crossref、出版社頁、OpenReview、ACL Anthology、USENIX、arXiv。
> 頁碼是核對 agent 讀的那個版本的 PDF 頁（各列有註明）。原始下載與抽出的文字在 scratchpad（不進 git）。

## 0. 結論

- 51 筆的 venue：**VERIFIED 33 筆**；**UNVERIFIED 18 筆**（只有 arXiv，或頁首寫了 venue 但查不到 DOI／錄取名單），在 `refs.bib` 每筆上方都有註解。
- 引用語境：核對時找到 **18 處要改的說法**，全部已改進 `main.tex`（§2）。改完之後，正文每一處引用都對得上原文。
- 寫進 bib 但正文沒引用的：`strata2026`、`tutti2026`、`lostmiddle2024`（BibTeX 不會印出來；留著備用）。

## 1. 逐篇（只列正文用到的說法）

標記：✅ 支持　⚠️ 部分支持（已改寫）　❌ 不支持（已刪或改）

| key | venue | 正文的說法 | 判定與出處 |
|:--|:--|:--|:--|
| cake2025 | ICML'25（PMLR 267）✅ | 前段 GPU 重算、後段 I/O 載入、在中間會合；會合點執行時才出現；事先存好全部 KV | ✅ §4 p.4；§5.1 p.5 "precompute and store all"；量化實驗只報速度（ICML 表 7／arXiv v2 表 6） |
| cachegen2024 | SIGCOMM'24 ✅ | 每 chunk 事先編多個等級、送的時候依量到的頻寬挑；可改送文字重算 | ✅ §5.3 p.6–7；§1 p.2 |
| pensieve2025 | EuroSys'25 ✅ | GPU→CPU 是 swap；CPU 滿才丟；保留值＝重算成本÷閒置時間；丟掉的回來時重算 | ⚠️ 原本寫成「直接逐出前段」，已改。§4.3.1–4.3.4 p.6–7，V＝Cost(s,l)/T |
| cachedattention2024 | ATC'24 ✅ | HBM／DRAM／disk 分層存多輪對話的 KV | ✅ 摘要 p.111；§3.3 |
| mooncake2025 | FAST'25 ✅ | 把 KV 存到 CPU／SSD 池重用；我們用它公開的 conversation、toolagent trace | ✅ FAST 標題和 arXiv 不同（bib 用 FAST 版）。512-token block 出自 arXiv v4 §4 p.7 與 repo README，不是 FAST 論文 |
| lmcache2025 | arXiv（UNVERIFIED） | KV 層，把 KV 存到 GPU 外並跨引擎共用 | ✅ 摘要；§1 p.2 |
| vllm2023、sglang2024 | SOSP'23、NeurIPS'24 ✅ | 只引用系統本身 | — |
| hcache2025 | EuroSys'25 ✅ | 從 hidden state 還原，依層分工 | ✅ §4.1 p.6（分法是每個硬體平台離線決定一次） |
| kvpr2025 | Findings of ACL'25 ✅ | 部分重算和傳輸重疊 | ✅ 摘要；§1 p.2 |
| cacheflow2026 | arXiv（UNVERIFIED） | 還原時規劃重算與載入 | ✅ 摘要；§3 p.3–6 |
| krul2025 | arXiv（UNVERIFIED；PVLDB 頁首是沒填的樣板） | 對話閒置時壓縮，每層存「依硬體校準的一部分」，其餘還原時重算 | ⚠️ 原本寫成「每層決定前段存多少」，已改。§6.1 p.8 |
| tierkv2026 | arXiv（UNVERIFIED；PDF 寫 EuroSys'27 但 DOI 404） | 解碼前依位置把 token 分成未壓縮／低秩／flash 三段 | ✅ §3.1 p.3–4（邊界由預測的長度再解出來） |
| lachesis2026 | arXiv（UNVERIFIED） | 配置時把 segment 放 HBM 或 flash | ✅ 摘要；p.2 |
| efficientagent2026 | arXiv（UNVERIFIED） | host 層的寫入准入 | ✅ §4 p.5（正文沒用 36% 這個數字） |
| hymcache2026 | arXiv（UNVERIFIED） | 寫入緩衝滿了就跳過寫入 | ✅ §3 p.6 |
| wherekvlive2026 | arXiv（UNVERIFIED） | 收益主要來自層的容量，不是放置策略 | ✅ 摘要 |
| kvdrive2026 | arXiv（UNVERIFIED） | prefill 時把全部 KV 存 SSD，再依 attention 分數挑進 HBM／DRAM | ⚠️ 原本歸成「有損的放置決定」，已改。§7.1 p.13 |
| adaptcache2025 | BigMem'25＠SOSP（workshop；作者端來源） | 新條目進來時決定壓縮等級與層，層滿時再調整 | ✅ §2 p.2 |
| evicpress2025 | arXiv（UNVERIFIED） | 同上，層滿時再壓或降級 | ✅ §4.4–4.5 p.7 |
| bottlenecks2026 | arXiv 2025（UNVERIFIED；「Submitted to MLSys 2026」，錄取名單上沒有） | 推導 κcrit；H100 上傳輸最多佔 99% | ❌→⚠️ 原計畫寫「H100／B200 上 κcrit 1–76」：76.4 是 A100，且是理論值，已不用這個數字。§6.3 |
| hotcarbon2026 | HotCarbon'26 ✅（EIR 卷期未定） | H200 高 KV 壓力下，卸載比「被搶佔後重算」省 14.4–31.9% 扣掉閒置的 GPU 能量（每輸出 token） | ⚠️ 指標要寫清楚是「扣閒置的邊際 GPU 能量」，已照寫。p.3 §4.1 |
| kivi2024 | ICML'24 ✅ | K 逐 channel、V 逐 token；最近 R 個 token 保持全精度（R＝128，group 32） | ⚠️ 「最多 R 個」，已改成 "full-precision residual"。§1 p.2；§3.3、§4.1 p.6 |
| kvquant2024 | NeurIPS'24 ✅ | 第一個 token 保留 fp16 | ✅ §3.5 p.6 |
| skvq2024 | COLM'24 ✅ | 最近 128 個＋5 個 sink 全精度 | ✅ §4.2–4.3 p.7–8 |
| streamingllm2024 | ICLR'24 ✅ | 前幾個 token 是 attention sink（驅逐情境，沒有量化實驗） | ✅ §3.1 p.4–5 |
| kvsink2025 | COLM'25 ✅ | sink 也可能出現在開頭之外 | ✅ 摘要；p.2 Fig.1 |
| zipcache2024 | NeurIPS'24 ✅ | 累積 attention 偏向前面的 token，用每欄非零數正規化 | ✅ §4.2 p.6 Eq.8 |
| kvtuner2025 | ICML'25 ✅ | Qwen2.5-3B／7B-Instruct 在 4／2-bit key 時明顯掉；逐層混精度 | ⚠️ 只有 3B／7B（14B／32B 反而穩），也沒測 -1M 版，已照寫。App. F p.28 |
| diffkv2025、cocktail2025、semkv2026 | SOSP'25 ✅、DATE'25 ✅、arXiv | 依內容／attention／內部指標選精度 | ✅（正文只用概括說法） |
| qevict2026 | arXiv（UNVERIFIED） | sink＋最近保持全精度 | ✅ p.5、p.7（最近區的大小原文前後不一：32／64／128） |
| pmkvq2025 | **ICLR 2026** ✅（key 名稱是 2025） | 第一個 token＋最近 128 個用 INT16 | ✅ §4.1.2 p.7；p.3 |
| patchkv2026 | arXiv（UNVERIFIED） | 只量化會被載入的 block，重算預測的那一組；卸載時依「attention 加權的重建風險」凍結每 block 精度 | ⚠️ 原本寫「依 attention 風險」「依和編輯點的距離」，已改。§IV p.5–6 Eq.13 |
| sparkv2026 | IEEE IoT-J 13(14) ✅ | 拿「Cake＋載入段量化」當 baseline | ✅ §III-C p.4、§VI-A p.8（「精度不隨位置變」原文沒寫，已不用） |
| ruler2024 | COLM'24 ✅ | 合成長上下文任務：多值撈針、變數追蹤、常見詞、QA | ✅ §3 p.4–5（作者 7 人或 8 人兩個來源不同，bib 用 COLM 頁首的 8 人） |
| hotpotqa2018 | EMNLP'18 ✅ | 兩段 gold 段落 | ✅ §3 p.4 |
| llama3_2024、qwen25_1m_2025 | arXiv 技術報告 | 模型出處 | ✅（Qwen 的 1M 要靠 DCA；我們在 vLLM 外自己的 harness 跑 32K，不受影響） |
| fp8formats2022 | arXiv | E4M3＝4 位指數、3 位尾數 | ✅ 表 1 p.3 |
| marconi2025 | MLSys'25 ✅ | 混合 SSM 模型在 prefill 時就地覆寫狀態，所以快取要在 prefill 時決定存哪些 | ✅ §3 p.4、§4.1 p.5（「必須」是推論，原文支持） |
| gill2008 | FAST'08 ✅（pp.49–65） | PROMOTE vs DEMOTE：demote 免費時差很小 | ✅ §V-B p.61 |
| wong2002 | ATC'02 ✅ | DEMOTE 加上下層「先丟剛送上去的」才接近 exclusive | ⚠️ 原本寫成「DEMOTE 就是 exclusive」，已改。§1.2 p.162 |
| tachyon2014 | SoCC'14 ✅ | lineage 讓 checkpoint 在背景做；記憶體塞滿未 checkpoint 的檔案時改成同步 | ❌→⚠️ 原計畫引用「寫入比磁碟頻寬快就失效」：那句是在批評**固定間隔** checkpoint，不是 Tachyon 自己，已改成 §4.1 p.6 的同步 fallback |
| tpp2023 | ASPLOS'23 ✅ | 回收與配置水位線分開 | ✅ §5.2 p.749 |
| vllm_pr43660 | GitHub PR（2026-05-28 合併）✅ | 開 KV connector 就排除 ROCM_ATTN，因為它的 KV 排法不是 blocks-first | ⚠️ 原本寫 "per-block contiguous"，已改成 PR 原文的 "blocks-first"。`rocm.py` diff |
| vllm_issue60316 | GitHub issue（open）✅ | 同一個排除機制在 PD 分離的 decode 上也出現 | ✅ 正文**沒有**引用「maintainer 說不在 roadmap」：回覆者在 GitHub 上是 CONTRIBUTOR，不是 maintainer |

## 2. 改進 main.tex 的地方（共 18 處）

1. Pensieve：swap 到 CPU、CPU 滿才丟、保留值的公式。
2. Krul：每層存「依硬體校準的一部分」，不是逐層決定。
3. TierKV：加上「解碼開始前」。
4. KVDrive／AdaptCache／EvicPress：分開寫，KVDrive 是先全部寫 SSD。
5. Bottlenecks：不用 1–76，改寫推導的內容與 H100 上 99%。
6. 生產系統那句：刪掉「沒有依位置的寫入規則」（我們沒有逐一核對原始碼）。
7. KIVI：改成 "full-precision residual"。
8. sink 的範圍：從「1–32 個」改成有引用支持的「1–5 個」；加 KVSink 的但書。
9. KVTuner：限定 Qwen2.5-3B／7B-Instruct。
10. PatchKV：風險分數與重算集合的定義。
11. 量化前作那句：刪掉沒逐篇核對的「多在 2–4 bit」，改成「低位元、在 GPU 內」。
12. Wong & Wilkes：DEMOTE 要搭配下層的插入策略。
13. Tachyon：改用它自己的同步 fallback，不用那句被誤植的引文。
14. vLLM PR：用 "blocks-first"。
15. vLLM issue：只引用機制，不引用回覆者的話。
16. HotCarbon：指標寫成「扣閒置的每 token GPU 能量」。
17. Gill：只寫「demote 免費時差很小」，沒寫 3.42／5.05 ms（那組數字要附很多條件）。
18. 「前作幾乎都沒和延後版比過」：從 "none" 改成 "few"（文獻摘要只說「幾乎沒有」）。

## 3. 還沒做的

- 沒有 cross-model 的引用稽核（`citation-audit` skill 原設計要 Codex；這台沒有）。
- UNVERIFIED 的 18 筆，投稿前要再查一次有沒有正式 venue（尤其 TierKV 的 EuroSys'27）。
