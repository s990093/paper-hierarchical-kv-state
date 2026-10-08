# 用 OS 的概念整理 KV cache 管理，以及論文地圖

> **一句話**：KV cache 管理可以對應到 OS 的五個概念：放置、替換、寫入策略、資料搬移、粒度。KV 另有一個 OS 沒有的動作：重算。KV 算完就不會再被修改，所以沒有 dirty 的問題。讀過的 61 篇論文與系統依這個框架分類之後：同一類（跨請求、無損、分層）的研究多在替換和讀取還原；「寫入時、依位置、無損地決定放哪一層」在評測卡涵蓋的範圍內是空的，這就是本研究的位置（動手前要再查新）。〔複核修正：補上範圍限定〕

**上層**：核心問題見 [../CORE_QUESTION_20261008.md](../CORE_QUESTION_20261008.md)，這個資料夾是回答它的工具之一。
**日期**：2026-10-08
**狀態**：文獻整理與概念對照，沒有新的實驗結果。
**出處**：
- 論文的事實只取自 `docs/research_20261006_eval/cards/`（E01–E08，已讀原文並獨立複核）、`docs/research_20261007_vlm/V01`、`docs/write_path_20261008/`（已複核），以及 RESEARCH_INTRO §3。〔複核補充：另外用到 `docs/EVAL_FOUNDATIONS_20261007.md` §2（KV 層系統）、RESEARCH_INTRO 表 8／12／13、`results/RUNLOG_MI300X.md` 發現 13（commit `9deda4f`），以及 `docs/phase1_20261008/` 的策略與實驗設計〕
- OS 名詞依教科書的一般用法〔教科書〕；對應與分類是本文的整理〔判讀〕。

---

## 你的問題，簡短回答

| 問題 | 簡短回答 | 詳細 |
|:--|:--|:--|
| 學習式（模型預測）的方法怎麼處理寫入？ | 幾乎都不處理。LARU、LPC、SAECache 都是單層 GPU 快取的**逐出**；KVP 等是單一請求內的有損 token 逐出。在寫入時（prefill 或 token 產生時）就決定的有 KVP、LookaheadKV、TRIM-KV（TRIM-KV 最接近：產生時給分），但它們決定的都是丟不丟，不是放哪一層〔複核修正：原只列 TRIM-KV；KVP、LookaheadKV 也是 prefill 時一次決定〕 | `02` §4 |
| OS 的寫入配置／不配置（write-allocate）也要考慮嗎？ | 要，而且它剛好可以描述你的方法：新的 KV 要不要佔用 CPU 這一層。S5 就是**依位置逐段決定配置與否**：後段配置在 CPU，前段直接寫 SSD | `01` §3 |
| 之前只考慮讀取，我改變是因為在讀取之前就決定，想打破什麼？ | 對。你打破的是 Cake 等論文的前提「KV 早就全部存好」。決定提前到寫入時，所以一定要談寫入策略 | `04` §3 |
| 你貼的 OS ↔ KV 對照表對嗎？ | 大部分對。要改四處：dirty bit 不是「KV 有沒有更新」（KV 不會更新）；LRU 那一列把「重用」和「重要度」混在一起；eviction 不只是 offload，也可以直接丟（之後重算）；write-back 不是「寫到下一層」，而是「被擠出時才寫」。另外要補上「重算」〔複核修正：原寫「三處」，漏了 `01` §1 表中標 ⚠️ 的 eviction 那一列〕 | `01` §1 |
| LLM 那邊沒有更新、dirty 的問題嗎？ | 沒有。KV 算完就不會再改，丟了也能重算，所以「dirty」只代表「下一層還沒有備份」。要注意的是，在 MI300X 上，KV 從 CPU／SSD 載回和全在 GPU 的輸出並不逐位元相同（最終答案 8/120 不同），原因還沒判定〔複核修正：原寫「重算在 MI300X 上不一定逐位元相同」；RUNLOG_MI300X 發現 13 比的是卸載路徑，不是重算，見 `01` §4〕 | `01` §4 |
| 我們自己設定的策略和 OS 的，是兩套嗎？ | 是兩套：KV 管理器一套（我們設定），作業系統一套（page cache、swap）。兩套會打架，所以 KV 系統通常會繞過作業系統：pinned 記憶體、O_DIRECT、讓 GPU 直接發 I/O | `01` §5 |
| 替換用 LRU 不夠、改用模型，或看前綴，原理不同嗎？ | 不同，是三件事：模型是把「會不會再用」猜得更準；前綴是標準前綴快取的結構限制（後面的塊要前面的在才有用；Pensieve、AsymCache、Fancy-eviction 有放寬）；注意力是「對答案多重要」（有損） | `02` §3 |
| 你貼的「KV 逐出」說明對嗎？ | 混了兩個問題：跨請求的快取替換（無損），和單一請求內的 token 逐出（有損）。「KV100 很久沒用、之後又被注意力用到」只在稀疏注意力或檢索式取用時成立。本研究看的是「會不會再用」和「拿回來的代價」，不是注意力分數 | `02` §1–2 |
| 這麼多概念，哪個最重要？ | 對第一階段：寫入策略 × 放置（主角）、重算 ＞ 替換（對手）、搬移（預取留到後面）＞ 粒度（固定）。注意力重要度先不碰〔複核修正：原寫成嚴格排序「主角＞重算＞替換＞搬移」，與 `04` §4 表（前兩者同為 ★★★、中間兩者同為 ★★）不一致〕 | `04` §4 |
| 大家的論文都在做什麼？之前的 L0–L5 沒考慮寫入嗎？ | 61 篇的分類見論文地圖。L0–L5 是依「決定的對象」分，沒有把「什麼時候決定」分開，所以寫入策略散在各層裡、看不出來 | `03`、`04` §1–2 |

---

## 怎麼讀

| 檔案 | 內容 |
|:--|:--|
| [01_os_mapping.md](01_os_mapping.md) | OS ↔ KV 對照總表（含你貼的表哪些要改）、五個分支加上重算、寫入策略（寫穿、延後寫、寫入配置、准入）、為什麼沒有 dirty、兩套策略 |
| [02_eviction_and_learned.md](02_eviction_and_learned.md) | 「逐出」是兩個不同的問題、三種訊號（機率、代價、品質）、替換的五種原理、學習式方法怎麼處理寫入 |
| [03_paper_map.md](03_paper_map.md) | **論文地圖**：61 篇論文與系統，每篇在做 OS 的哪一件事、何時決定、無損或有損 |
| [04_where_research_sits.md](04_where_research_sits.md) | L0–L5 × 何時決定、為什麼以前只看讀取、你為什麼要轉到寫入、哪個概念最重要 |

## 圖

| 圖 | 內容 |
|:--|:--|
| ![圖 1](figures/fig1_os_kv_tree.svg) | 圖 1：用 OS 的概念看 KV cache 管理 |
| ![圖 2](figures/fig2_write_allocate.svg) | 圖 2：寫入配置 vs 不配置，以及 S5 |
| ![圖 3](figures/fig3_two_layers.svg) | 圖 3：兩套策略 |
| ![圖 4](figures/fig4_levels_by_time.svg) | 圖 4：L0–L3 × 何時決定 |

---

## 複核紀錄

- **複核者**：zero-context subagent（沒看過本文件的撰寫過程）。**與撰寫者是同一個模型家族，屬 self-review，不是 cross-model review。**
- **日期**：2026-10-08
- **方法**：逐條回到出處核對：評測卡 E01–E08（含卡內複核紀錄）、V01、`docs/EVAL_FOUNDATIONS_20261007.md` §2、`docs/write_path_20261008/`、RESEARCH_INTRO §3 表 5、§5.2 表 8、§8.4 表 12–13 與參考文獻 [41]、`git show 9deda4f:results/RUNLOG_MI300X.md` 發現 13、`docs/phase1_20261008/`。`03` 的 61 個「一句話」用程式逐字比對卡片。這次沒有另外下載原文：卡片與已複核的 write_path 已足以判定每一條，沒有遇到卡片與原文衝突需要裁決的情況。OS 名詞（寫穿／延後寫、寫入配置的通常搭配、dirty bit、page cache、O_DIRECT）依教科書一般用法檢查，沒有發現錯誤。
- **檢查了 245 條**：✅ 203、❌ 41（已就地改正，標〔複核修正〕）、⚠️ 1（引文找不到原句，已改寫）。

| 檔案 | 條數 | ✅ | ❌ | ⚠️ |
|:--|--:|--:|--:|--:|
| `03_paper_map.md`（61 列＋篇數計算＋A 組標題＋§F 表與讀法） | 67 | 52 | 15 | 0 |
| `01_os_mapping.md` | 56 | 53 | 3 | 0 |
| `02_eviction_and_learned.md` | 30 | 23 | 6 | 1 |
| `04_where_research_sits.md` | 32 | 25 | 7 | 0 |
| `README.md` | 12 | 6 | 6 | 0 |
| 4 張圖的文字 | 48 | 44 | 4 | 0 |
| **合計** | **245** | **203** | **41** | **1** |

### 確認無誤、值得一提的
- 「61 篇」的計算正確：E01 7＋E03 7＋E04 8＋E05 8＋E06 7＋E07 9＋E08 9＝55；E02 的 KV 層系統 5 個（EVAL §2.4）扣掉已在 E04 的 LMCache＝4；V01 的 ReKV、MuKV＝2。`03` 的列數也是 61（A 組 33）。
- `04` §1 的 L0–L5 表與 RESEARCH_INTRO 表 5 一致；GreedyDual-Size 是 RESEARCH_INTRO 的 [41]（Cao & Irani, USITS 1997，表 8、表 12 都引）。
- Fancy-eviction 測 14 種、LRU 很穩、頻率派落後；LARU 的逐出候選限於 RadixTree 的 leaf node；SGLang 先逐出葉節點——都與卡片一致（E05、E06 LARU「軟體與版本」、E01 SGLang）。
- vLLM block 16（E01 vLLM 卡、EVAL §5）、LMCache chunk 256、寫穿（LMCache、SGLang 上游預設）、延後寫（Mooncake Store）、`store_threshold`、Dynamo 頻率 ≥2、CachedAttention 的容量需求比（RCC／CCpUT）都有出處（EVAL §2.2–2.4、E04）。
- KV「算完不會被修改」的論證〔判讀〕本身沒有問題；KIVI（量化）、CacheBlend（拼接時部分重算）、CachedAttention（存的時候去掉位置編碼以便截斷）的描述與 E07、E04 卡一致。

### 逐條修改（改前 → 改後；出處）

**`03_paper_map.md`**
1. py-kvcache：放在 A1「寫入策略與准入」、「寫入時」 → 移到 A4，「讀取時的准入：載不載」。卡片：門檻在 lookup 時拒絕載入，程式碼 `prepare_store` 照樣存所有新 block（E03 卡 7「與既有整理不一致」①④、共同模式 4）。
2. KVP：「每一步」 → 「寫入時（prefill 後壓一次）」（E06 KVP 學習設定「推論開銷」，[KVP] p17）。
3. ForesightKV：「每一步」 → 「解碼中每 L＝256 步一次」（E06 ForesightKV，[FKV] p6）。
4–11. 8 列「一句話」是節錄、不是照抄 → 改回卡片原句：HCache（漏「並用其他方法補 pipeline 空泡」，E03）、DistServe（漏「以最大化每張 GPU 的 goodput」，E01）、Splitwise（漏「並用模擬器設計……叢集」，E01）、Etalon（漏「各自」「以 token 期限為基礎的」，E01）、KVCache in the wild（漏「（Tongyi）」，E05）、LRB（漏 relaxed Belady 的定義，E05）、kvpress（漏「在 HF transformers 上」「（press）」，E08）、MuKV（漏「的框架」「問答時兩階段檢索」，V01）。
12. A 組標題「跨請求、無損」 → 加註例外：KVDrive 沒有跨請求重用、有損稀疏注意力（E04）；KVPR 是單一 batch 解碼期卸載（E03 卡 5）；AdaptCache、EvicPress 有損壓縮；LRB、HALP 是 CDN（E05）。
13. §F 表：「KVDrive（依重要度）……（後兩者有損）」會讀成 KVDrive 無損 → 「KVDrive（依重要度，有損稀疏注意力）」；py-kvcache 移到「讀取還原 × 讀取時」（E04 KVDrive 卡③：RULER 比 Full 低約 5–6 點）。
14. §F 讀法：「讀取還原、搬移（8 項）」 → 9 項。
15. §F 讀法：「准入開關（存或不存，依次數或門檻）」 → 依次數（HiCache selective、`store_threshold`、Dynamo）或依前綴樹分支點（Marconi，單層），並補上 MTDS（E05 Marconi 卡，原文 p.5；EVAL §2.3–2.4）。

**`01_os_mapping.md`**
16. §3 准入：「例如命中 2 次才寫 SSD（Strata）」 → 「命中 2 次才從 GPU 備份到 CPU（Strata、HiCache selective），或頻率 ≥2 才寫磁碟（Dynamo KVBM）」（E04 Strata 卡「分層實作」；EVAL §2.4）。
17. §4：「重算不一定逐位元相同……不同設定下的輸出不完全一樣……可能有極小的差別」 → 發現 13 比的是 KV 從 CPU／SSD 載回（`cpu_lru`、`tier_fs`）對全在 GPU（`full_gpu`）：sha1 36/120 不同、**最終答案 8/120 不同**，原因待判定（缺同設定重跑的對照）；不能說是「極小」，也不是專指重算（RUNLOG_MI300X 發現 13，commit `9deda4f`）。
18. §5：「讓 GPU 直接發 I/O：……HCache 用 SPDK＋GDRCopy」 → 「Tutti 由 GPU 發 I/O；HCache 讀回時用 SPDK＋GDRCopy 直接 P2P 進 GPU，寫入則經主機」（E03 HCache 卡「I/O 怎麼實現」；write_path `02` §2B）。

**`02_eviction_and_learned.md`**
19. §1「單一請求內的 token 逐出……生成很長時」 → 「prompt 或生成很長時」（SnapKV、PyramidKV、KVP、LookaheadKV 都在 prefill 後壓；E07、E06）。
20. §2「學習式預測（LARU、LPC、Bidaw 用回答長度）」 → Bidaw 改列為啟發式預測（回答長度估重用距離下界＋Belady 影子快取，沒有訓練模型；E04 Bidaw 卡）。
21. §3 結構限制「只能從樹葉（尾端）開始丟」 → 限定「標準前綴快取」，並註明 Pensieve（從開頭逐出、非連續 KV kernel，E03）、AsymCache（MSA 不連續命中，E05）、Fancy-eviction（洞之後仍算命中，E05）放寬了這個限制。`01` §2、README 同步。
22. §4 表：KVP 與 ForesightKV 同列「每一步」 → 拆開：ForesightKV「每 L＝256 步」、KVP「prefill 後一次，算是寫入時決定（有損）」（同第 2、3 條）。
23. §4 結論：「寫入時就決定的（LookaheadKV、TRIM-KV）」 → 補上 KVP。
24. §4 末段：「L2 的 label……不用跑 LLM；L1 的 label 來自目標 LLM 的注意力」 → 補上例外：SAECache 的多輪分類器讀 LLM hidden state；TRIM-KV 是對原模型輸出的蒸餾（E06 共同模式 3）。
25. ⚠️ §4 末段引號「預測器只預測負載、不碰 KV 內容」標為 RESEARCH_INTRO 表 13 → 表 13 沒有這句話。改寫成表 13 的實際內容（輸入都是負載訊號），並註明「只預測負載」出自 phase1 `02` §9。

**`04_where_research_sits.md`**
26. §2 L1 列：「逐出時：KVP、ForesightKV、LookaheadKV」 → KVP、LookaheadKV 移到「寫入時」（prefill 後一次），逐出時只留 ForesightKV（與 `03`、`02` §4 一致；E06）。
27. §2 L2 列「寫入時」漏了 AdaptCache、EvicPress → 補上；「讀取時」補 py-kvcache（`03` §F；write_path `02` §1：AdaptCache arXiv v2 p.2 §2、EvicPress p.6–7）。
28. §2 讀法：「寫入時就決定的只有 KVDrive 和 MTDS」 → 「有 KVDrive（有損）、MTDS（優先順序）、AdaptCache、EvicPress（有損壓縮），沒有無損、依位置的」，並補 HCache 是 L0 的無損、依層前例。
29. §3 Cake 引文「precompute and store all in advance」 → 卡片記錄的原句「we precompute and store all requests' KV cache in advance」（E03 Cake 卡，ICML p5）。
30. §3「評測：論文量的是 TTFT 和吞吐，很少量寫入……」 → 加上「評測卡涵蓋的論文裡」與出處（write_path `03`）。
31. §3「老師點名要比的，也正是這些」（含選擇性寫入） → 老師點名的是延後寫入、成本感知逐出、先寫再丟、一開始不寫；選擇性寫入是 write_path `03` §3.2 的建議（phase1 `01` §1 老師原文）。
32. §4「也是唯一的空白」 → 「也是評測卡範圍內看到的空白（動手前要再查新）」。

**`README.md`**
33. 一句話「……是空的」 → 加上「在評測卡涵蓋的範圍內」「動手前要再查新」。
34. 學習式那一列「最接近寫入時決定的是 TRIM-KV」 → 補上 KVP、LookaheadKV 也是 prefill 時決定（都是丟不丟）。
35. 對照表那一列「要改三處」 → 「四處」（漏了 `01` §1 標 ⚠️ 的 eviction）。
36. dirty 那一列「重算在 MI300X 上不一定逐位元相同」 → 同第 17 條。
37. 替換原理那一列「前綴是結構限制」 → 「標準前綴快取的結構限制（有系統放寬）」（同第 21 條）。
38. 「哪個最重要」寫成嚴格排序 → 改成與 `04` §4 表一致（前兩者同為 ★★★、替換與搬移同為 ★★）。

**圖**
39. 圖 3「GPU 直接發 I/O：Tutti（io_uring）、HCache（SPDK＋GDRCopy）」 → 「I/O 直達 GPU：Tutti 由 GPU 發（io_uring）；HCache 讀回用 SPDK＋GDRCopy 直通」（同第 18 條）。
40. 圖 4 L1「逐出時：KVP、ForesightKV、LookaheadKV」 → 寫入時欄加「KVP、LookaheadKV：prefill 後丟 token」，逐出時欄只留 ForesightKV（同第 26 條）。
41. 圖 4 L2「寫入時」漏 AdaptCache、EvicPress → 補上；讀取時欄補 py-kvcache（同第 27 條）。
42. 圖 4 讀法第二行「寫入時就決定的 L2 工作有 KVDrive 和 MTDS」 → 補 AdaptCache、EvicPress（同第 28 條）。

（第 25 條是 ⚠️，其餘 41 條是 ❌。）

**只改排版、不算在上面的**：圖 3 原本「O_DIRECT：讀寫 SSD 不經 page cache」與底部說明文字超出框線（以 `rsvg-convert -w 960` 轉 PNG 確認），已縮短文字、底部說明拆成兩行、畫布高 420→436。四張圖都通過 `xml.dom.minidom` 解析，並重新轉 PNG 確認沒有重疊。

**`〔複核補充〕`（沒有錯，補條件）**：Fancy-eviction 的 14 種含 LRU 本身；KVPR 列補「沒有跨請求重用」；`01` §2 ② 補「標準前綴快取」；README「出處」補上實際用到的 EVAL_FOUNDATIONS、RUNLOG_MI300X、phase1。

### 沒有檢查或要注意的
- 所有判定都以評測卡與已複核的 write_path 為準，沒有重新讀原文；若卡片本身有錯，這裡也會跟著錯。
- 「寫入時、依 token 位置、無損地決定放哪一層是空的」只代表評測卡涵蓋的 61 篇論文與系統，不是查新。最接近的前例有三類，查新時都要引用並劃清界線：HCache（無損、寫入時依成本，但單位是層）、KVDrive（寫入時決定放哪層，但有損、依重要度、無跨請求）、AdaptCache／EvicPress（存入時決定壓縮與層，但有損、以整段 context 為單位）。Marconi 也在寫入時依前綴樹的位置（分支點）決定存哪些狀態，但只有單層。
