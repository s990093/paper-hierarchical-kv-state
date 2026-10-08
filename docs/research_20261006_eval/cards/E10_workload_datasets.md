# E10　負載資料集說明書：ShareGPT、Alpaca、LMSYS、Arena、WildChat、OpenOrca、BurstGPT、Azure、Mooncake、Bailian／ServeGen、TraceLab，以及長文件負載

> **範圍**：serving 與 KV cache 論文拿來當「負載」的資料集與 trace。每張卡回答：它是什麼、多大、多長、有沒有時間與多輪、license、**哪篇 serving 論文怎麼用它**（取樣、過濾、截斷、到達怎麼補）、為什麼大家用它、陷阱。另有長文件負載表、到達過程一節，以及對 PoC 的建議。
> **抽取者**：E10（子 agent），查證 2026-10-06～07。
> **硬規則**：依 `../README.md`。每格附出處與證據等級。頁碼一律是**我讀的那個 PDF 的頁序**（USENIX 會議版含封面頁）。百分位數：我自己算的用 numpy `linear`；為了比對 `workloads_eval.md`，另用 `higher` 法（見抽查表）。
> **暫存與腳本**：`<SCRATCH>/E10/`（`data/`、`pdf/`、`txt/`、`mc_stats.py`、`bailian_stats.py`、`bidaw_stats.py`、`tracelab_stats.py`、`sharegpt_stats.py`、`pg.py`）。`<SCRATCH>` 的定義見 README。

## 來源清單（讀了什麼）

**論文全文（PDF → `pdftotext -layout`）**
| 簡稱 | 版本 | 用到的頁 |
|:--|:--|:--|
| vLLM | arXiv 2309.06180v1（SOSP'23） | p9、p10、p11、p12 |
| DistServe | OSDI'24 USENIX 版 | p10 |
| Sarathi-Serve | OSDI'24 USENIX 版 | p9、p11 |
| Splitwise | arXiv 2311.18677v2（ISCA'24） | p3、p4、p9 |
| Llumnix | arXiv 2406.03243v1（OSDI'24） | p3、p4、p9 |
| DynamoLLM | arXiv 2408.00741v1（HPCA'25） | p2、p4、p9 |
| Mooncake | FAST'25 USENIX 版；另讀 repo 附的 `Mooncake-FAST25.pdf`（無封面，頁序少 1）；arXiv 2407.00079（早期技術報告） | FAST p4、p10、p12、p13；arXiv p3、p15。〔複核補充〕arXiv 的頁碼對應 **v4**（目前最新版，23 頁）；v1（21 頁）的同一張 Table 2 在 p13 |
| CachedAttention | ATC'24 USENIX 版 | p2、p10 |
| Pensieve | arXiv 2312.05516v3（EuroSys'25） | p9、p12 |
| Strata | OSDI'26 USENIX 版 | p5、p9、p10 |
| HCache | arXiv 2410.05004v1（EuroSys'25） | p1、p3、p4、p9、p10、p13 |
| Tutti | arXiv 2605.03375v1 | p9 |
| Bidaw | FAST'26 USENIX 版 | p3、p4、p8、p11、p12 |
| Marconi | arXiv 2411.19379v3（MLSys'25） | p8、p9 |
| SAECache | arXiv 2605.18825v1 | p4、p6、p8、p16 |
| MTDS | Complex & Intelligent Systems 2026 | p10 |
| Vidur | arXiv 2405.05465v2（MLSys'24） | p7 |
| LMCache | arXiv 2510.09665v2 | p11、p12 |
| EvicPress | arXiv 2512.14946 | p10 |
| Fancy-eviction | arXiv 2609.28870v2 | p3、p5 |
| LMSYS-Chat-1M | arXiv 2309.11998v4（ICLR'24） | p1–p4 |
| WildChat | arXiv 2405.01470v1（ICLR'24） | p2、p3 |
| BurstGPT | arXiv 2401.17644v5（KDD'25） | p1–p6 |
| MT-Bench／Arena | arXiv 2306.05685 | p1、p3 |
| ServeGen | arXiv 2505.09999v3（NSDI'26）只讀 p1–p2 與 p10 腳註 | — |
| KVCache in the Wild（Bailian） | arXiv 2506.02634v5（ATC'25）只讀 p1–p4 | — |

**資料檔與文件（皆下載到 `<SCRATCH>/E10/data/`，附 commit／sha）**
* Mooncake `FAST25-release/`，commit `1d0e4c75`（2026-07-03）：README、三個 `traces/*.jsonl`、`arxiv-trace/mooncake_trace.jsonl`、`Mooncake-FAST25.pdf`。
* Bailian `alibaba-edu/qwen-bailian-usagetraces-anon`，commit `5f7439c5`（2026-04-23）：README、`docs/qa-context-growth-pattern.md`、四個 jsonl（LFS，共 312 MB）。
* TraceLab `uw-syfi/TraceLab`，commit `11b8b14c`（2026-08-22）：README、LICENSE-DATASET；release `v0.0.1`（sha256 `9d265eae…` 與 README 相符）與 `v0.0.2`（sha256 `11ce51ec…`）。
* Bidaw trace `ShipengHu-777/Interactive-conversation-workload`，commit `6f3281b9`（2026-03-09）：README＋7 個 part。
* Azure `AzurePublicDataset`，commit `215becda`（2026-10-01）：2023／2024 兩份 md、2023 兩個 CSV。2024 的兩個 CSV 為 692 MB 與 1.14 GB，**超過 500 MB，未下載**。
* BurstGPT release v2.0：`BurstGPT_1.csv`、`BurstGPT_3.csv`；README commit `d895a53b`。
* ServeGen `alibaba/ServeGen`，commit `d70c8b2d`（2026-08-27）：README、`data/conversations/conversations_hashed.json`。
* ShareGPT：原 repo `anon8231489123/ShareGPT_Vicuna_unfiltered`，commit `192ab218`（2023-04-12）的 README 與 3 支腳本。主檔 673 MB **未下載**；改用 **sha256 相同**（`35f0e213…`）的鏡像 `learnanything/sharegpt_v3_unfiltered_cleaned_split`，取 HF 自動轉的 parquet（235 MB）。
* HF 資料卡與 datasets-server：LMSYS-Chat-1M、Chatbot Arena、WildChat-1M、WildChat-4.8M、OpenOrca、Alpaca（另取 parquet 24 MB）、Inferact Codex、ccdv/arxiv-summarization。
* 程式碼：vLLM clone（E02 的 `<SCRATCH>/E02/vllm`，commit `31e2443c`，2026-10-06）；MLPerf `mlcommons/inference` commit `3fbc3299`（`language/llama2-70b/README.md`、`processorca.py`）；MLPerf policies（E02 clone，commit `d3eba2f2`）。
* Tokenizer：`Qwen/Qwen2.5-7B-Instruct` 的 `tokenizer.json`；`facebook/opt-13b` 的 `vocab.json`＋`merges.txt`（GPT-2 BPE）。

---

## 重點摘要

1. **經典 serving 負載都是短 context，而且被設計成短的。** 最常用的 `ShareGPT_V3_unfiltered_cleaned_split.json` 有 94,145 筆，**每筆是長對話切成約 2K token 的一段**（README 與切段腳本）。用 Qwen2.5 tokenizer 算，每筆總長 p99 只有 2,873，≥16K 只占 0.018%〔計算〕。vLLM benchmark 工具再把它限制在 prompt ≤1024、總長 ≤2048〔程式碼〕。Alpaca 中位 prompt 15 token，OpenOrca（MLPerf）限 <1024。這些資料對 16K–512K 研究**只能當短 context 的負對照**。
2. **vLLM 的 `ShareGPTDataset` 只取每筆的前兩則，而且不檢查說話者。** V3 檔有 36.7% 的條目以 `gpt` 開頭（切段造成）。所以通過過濾的樣本中，有 35.5% 的第一則不是 human〔計算＋程式碼〕；〔複核修正〕其中嚴格屬於「拿 GPT 的回答當 prompt、拿使用者的話當輸出」（gpt→human）的是 35.3%（28,678／81,267），其餘 0.2% 是 system→human 等〔計算，V10 `v_sharegpt2.py`〕。它是單輪、無重用、無時間的負載。
3. **沒有時間戳，serving 論文幾乎一律補 Poisson。** 例子：vLLM、DistServe、Sarathi-Serve、Llumnix、CachedAttention、Pensieve、Strata、Tutti、HCache、MTDS、LMCache。**Splitwise 連自家的 Azure trace 都只取長度分布、到達用 Poisson**（p9）。Bidaw 原文證實：換成 Poisson 時間後，依前一輪回答長度的驅逐策略不再降低 miss rate（p12）。
4. **有真實時間＋前綴 hash 的公開 trace 只有 Mooncake（512-token block，1 小時）與 Bailian（16-token block，2 小時）。** 兩者的單位都已用資料本身驗證：`ceil(input/B)==len(hash_ids)` 在 7 個檔都 100% 成立〔計算〕（〔複核修正〕連 Mooncake 的 synthetic 檔一起算是 8 個檔，全部 100%）。但長度都短：中位 574–6,909。〔複核補充〕ServeGen 對話檔也有 hash 列表與真實時間，但 block 大小沒有文件，無法驗證（見 C10）。
5. **長 context＋真實時間的只有 agent 型 trace，而且版本在變。** TraceLab README 仍釘 v0.0.1（357,161 列、43 人、中位 124,018），**但 2026-07-24 已發布 v0.0.2**：665,453 列、52 人、中位 132,092、>262,144 占 14.1%〔計算〕。用 `latest` 連結下載會拿到 v0.0.2，引用時必須寫版本。
6. **「長文件負載」其實都是 benchmark 加合成到達。** LongBench、arXiv summarization、L-Eval、LooGLE、NarrativeQA：DistServe 把 LongBench 截到 2,048；Sarathi-Serve 濾掉 >16,384；Strata 濾掉 >128K 的 NarrativeQA 文件並抽 50 份；Tutti 輪流從子集抽取；到達全部是 Poisson（HCache 另做 Zipf）。〔複核修正〕HCache 的 L-Eval 主實驗沒有到達過程：原文說 GPU 只放得下 1–3 個長請求，所以 batch size 設為 1（p10）；Zipf 只用在 p13 的 GPU 重用子實驗，而且 Zipf 描述的是「哪一份 context 被請求」的熱度。**DistServe 用的是 LongBench 的 summarization，不是 arXiv summarization**（p10）。用 arXiv summarization 的是 Sarathi-Serve、Vidur、Mooncake 的 arXiv 版。
7. **使用者文件的錯誤**（詳見抽查表 U 系列）：
   * 表 17「Mooncake 模擬 16K–128K、快取比例 50%」出自 **arXiv 版**（2407.00079 Table 2，p15），不是所引的 FAST'25 版 [11]。FAST'25 版的合成負載是 ShareGPT＋L-Eval＋LooGLE 1:1:1（p10）。
   * 「Mooncake 的 128K 是模擬資料」不準確：FAST'25 的**真實** conversation trace 最長 126,195〔計算〕，原文也說最長達 128k（p10）。
   * HCache 主實驗不是 Zipf 到達：ShareGPT4 是「Poisson 的 session 到達＋輪間固定 30 s」（p10）；Zipf 只出現在 on-GPU 重用的子實驗（p13）。
   * 表 S4 的註「重用率為無限容量下的估計」對 TraceLab（95.7%）與 Codex（94.2%）不成立：那兩個數字是 provider **實際**的快取命中。
8. **抽查結果**：從 `workloads_eval.md` 抽 31 項（§3 的 29 項＋§6.2 的 2 項）：26 項 ✅（其中 3 項要用 `higher` 百分位法才能逐位重現）、4 項 ⚠️（版本、措辭、頁碼）、0 項 ❌。〔複核修正〕複核後為 27 ✅、3 ⚠️：A22 已用檔頭檔尾確認為一週，改 ✅；A31 的 ⚠️ 理由改寫（p17 確有該數字，問題在語意，見 A31）。另外 1 項把它標「未查證」的事改成定論：Codex trace 檔內**沒有**逐呼叫時間欄位。使用者文件 16 項：10 ✅、3 ⚠️、3 ❌。

---

## 抽查表

### A. `docs/research_20260924/workloads_eval.md` §3（與 §6）的數字

方法欄的「〔計算〕」都是我用上面列的檔重算的，腳本在 `<SCRATCH>/E10/`。

| # | 項目 | workloads_eval 寫法 | 我回到原始來源的結果 | 判定 |
|:--|:--|:--|:--|:--|
| A1 | Mooncake hash block 大小 | 512 token，`ceil(input/512)=len(hash_ids)` 100% | README（commit `1d0e4c75`）寫明 512 tokens〔文件〕。四個檔 conversation 12,031／12,031、toolagent 23,608／23,608、synthetic 3,993／3,993、arxiv-trace 23,608／23,608 全部成立；改用 16 檢查則為 0、0、49、0〔計算，`mc_stats.py`〕 | ✅ |
| A2 | Mooncake conversation 筆數／中位／p90／最大 | 12,031／6,909／27,367／126,195 | 12,031／6,909／27,367／126,195；平均 12,035 與 README、FAST'25 Table 2（p10）相同〔計算〕 | ✅ |
| A3 | Mooncake toolagent p90 | 16,810 | `linear` 16,806；`higher` 16,810〔計算〕 | ✅（差異來自百分位法） |
| A4 | Mooncake 無限容量 token 級重用 | 37.4%／57.1%（§6.1） | 37.4%／57.1%；synthetic 65.1%。原文 Table 2 的 cache ratio 是 40%／59%／66%（p10）〔計算＋原文〕 | ✅ |
| A5 | Bailian block 大小 | 16-token salted SipHash | README 寫明 16 tokens per block、SipHash-2-4〔文件〕。四檔 `ceil(input/16)=len(hash_ids)` 皆 100%（43,058、172,800、10,812、43,011）〔計算，`bailian_stats.py`〕 | ✅ |
| A6 | Bailian traceA 筆數／中位／最大 | 43,058／1,046／89,286 | 43,058／1,046／89,286；跨度 2.00 h〔計算〕 | ✅ |
| A7 | Bailian traceB | 172,800 筆，中位 574，全為單輪 | 172,800、中位 574；`parent_chat_id=-1` 的根請求 172,800（100%）、turn=1 占 100%〔計算〕 | ✅ |
| A8 | Bailian thinking／coder | 10,812 中位 3,680 最大 51,622；43,011 中位 4,540 最大 25,777 | 完全相同〔計算〕 | ✅ |
| A9 | Bailian ≥32K 比例 | 0%–0.18% | traceA 0.04%、traceB 0.01%、thinking 0.18%、coder 0.00%〔計算〕 | ✅ |
| A10 | Bailian token 級無限容量重用 | traceA 58.1%、traceB 54.6%；四條 46.2–66.5% | 58.1%、54.6%、46.2%、66.5%〔計算〕。〔複核〕V10 獨立重算一致；這組數字是「前綴算到第一個 miss 為止」的定義。若不要求連續（任何位置見過的 block 都算），traceA 為 59.4%、traceB 為 61.2%，thinking、coder 不變；Mooncake 三檔兩種定義相同〔計算〕 | ✅ |
| A11 | TraceLab 規模 | 357,161 次呼叫、43 位開發者、4,265 個 session | v0.0.1：357,161 列、43 user、4,265 session；claude 140,338／codex 216,823，與 README 表相同〔計算＋文件〕。**但 v0.0.2（2026-07-24）為 665,453 列、52 user、8,058 session**〔計算〕 | ⚠️ 數字對，但需標版本 |
| A12 | TraceLab input 中位／p90／p99 | 124,018／256,767／822,895 | `higher` 法 124,018／256,767／822,895；`linear` 的 p99 為 822,887〔計算〕 | ✅ |
| A13 | TraceLab ≥100K／>131,072／>262,144 | 60.9%／47.0%／9.7% | 60.9%／47.0%／9.7%（v0.0.1）；v0.0.2 為 63.4%／50.4%／14.1%〔計算〕 | ✅ |
| A14 | TraceLab `prefix+new=input`；Claude 列 `prefix_tokens=cache_read`；95.7% | 100%；完全相同；95.7% | 100.00%；140,338／140,338；Σprefix÷Σinput＝95.7%（v0.0.2 為 95.6%）〔計算〕。README 的帳目公式也相同〔文件〕 | ✅ |
| A15 | TraceLab 時間範圍 | 2025-09-23 → 2026-06-04 | v0.0.1 的 `timing_events` 最早 2025-09-23T21:52:39Z、最晚 2026-06-04T06:27:34Z〔計算〕 | ✅ |
| A16 | TraceLab license | 資料 CC BY 4.0；程式 Apache-2.0 | `LICENSE-DATASET.md` 為 CC BY 4.0；GitHub API 的 repo license 為 Apache-2.0〔文件〕 | ✅ |
| A17 | Bidaw 輪數 | 1,268,346 輪、56,573 人、平均 22.42 輪 | 1,268,346、56,573、22.42；中位 18、p90 45；`round_index` 連續、時間單調皆 56,573／56,573〔計算，`bidaw_stats.py`〕。原文 p4：平均／中位／P90＝22／18／45 | ✅ |
| A18 | Bidaw 長度與間隔 | query 35.3、response 44.6；輪間 p50 43 s、p99 142 s；21.35 h | 35.27、44.61；p50 43 s、p99 142 s；21.35 h〔計算〕。原文 p4 寫 query 36、response 45 | ✅ |
| A19 | Bidaw「含歷史 prompt」 | 中位 1,066、最大 47,624、歷史占 97.7% | 中位 1,066、p90 3,480、p99 8,168、最大 47,624、97.7%（假設 prompt＝該使用者所有先前 query＋response＋本輪 query）〔計算〕 | ✅ |
| A20 | Bidaw repo 沒有 license | `license=null` | GitHub API `license: null`〔文件〕 | ✅ |
| A21 | Azure 2023 | code 8,819 中位 1,469 最大 7,437；conv 19,366 中位 1,020 最大 14,050；檔內時間 2023-11-16，md 寫 November 11th | 全部相同；conv 18:15:46–19:14:08、code 18:17:03–19:14:19〔計算〕。md 與 Splitwise 原文 p3 都寫 November 11th〔文件＋原文〕 | ✅ |
| A22 | Azure 2024 | 有，一週（2024-05）；未計算 | md 寫「May 10th–19th 2024」（字面 10 天）；檔名 `_1week`；DynamoLLM p2 寫「one week」〔文件＋原文〕。檔案 >500 MB，未下載，未計算。〔複核補充〕V10 只用 HTTP Range 讀兩個檔的前、後 300 bytes（未下載全檔）：code 檔 2024-05-10 00:00:00 → 05-16 23:59:59，conv 檔 2024-05-12 00:00:00 → 05-18 23:59:59，各剛好 7 天；兩檔合起來是 5/10–5/18，md 的「10th–19th」大概是兩者的聯集（到 19 日 0 時為止）〔計算，假設檔案依時間排序；檔案大小 691,989,454／1,135,195,393 bytes〕 | ~~⚠️~~ ✅〔複核修正〕「一週」正確（每個檔各 7 天，起始日不同） |
| A23 | BurstGPT_3 | 5,344,021 筆、conversation 233,617、中位 327、≥32K 0.029%、55,920 session、中位 2 輪、輪間 p50 131 s | 5,344,021；233,617；327；0.029%；55,920；中位 2（平均 4.18）；p50 131 s（p90 2,340 s）〔計算〕 | ✅ |
| A24 | ServeGen 對話檔 | 1,616 段、5,720 輪、2025-02-16～17；中位 7,750、p90 55,321、最大 262,644、18.1% ≥32K | 1,616／5,720；UTC 2025-02-16 16:01 → 02-17 15:57；`higher` 法 7,750／55,321（`linear` 7,745／55,276）；262,644；18.1%〔計算〕 | ✅ |
| A25 | Codex SWE-bench Pro 數字 | 610 trial、20,230 次呼叫、中位 63,917、94.2% | 資料卡全部相同（commit `0d52ae8c`）〔文件〕 | ✅ |
| A26 | Codex 檔內有無逐呼叫時間戳 | 未查證 | datasets-server `/info` 推出的 schema 只有 `conversations[{from,value}]`，610 列，**沒有任何時間欄位**〔文件〕。資料卡第一句說它是用 SWE-bench Pro 任務加 Codex agent **生成**的 | 新結論：沒有 |
| A27 | WildChat-1M | 837,989 段；ungated；user 訊息的時間為 null | datasets-server 837,989 列；`gated=False`；資料卡：只有 assistant 回合有 `timestamp`，意義是後端收到完整回應的時間〔文件〕 | ✅ |
| A28 | Chatbot Arena | 33K、平均 1.2 輪、有 `tstamp`、prompt CC-BY-4.0／輸出 CC-BY-NC-4.0 | 資料卡全部相同；`judge` 欄為匿名使用者 id〔文件〕 | ✅ |
| A29 | LMSYS-Chat-1M 無時間戳、禁止再散布 | 同左 | features 只有 `conversation_id, model, conversation, turn, language, openai_moderation, redacted`；授權條款的 Prohibited Transfers 一條禁止轉給第三方〔文件〕 | ✅ |
| A30 | §6.2「Strata 給 ShareGPT 插的 60 s **固定** thinking time（p10）」 | 固定 60 s | Strata p10 只寫插入 60 秒 thinking time，並說依照 Pensieve 的方法，**沒有說固定或隨機**。Pensieve p9 是從**指數分布**取樣，p12 說預設平均 60 s〔原文〕。〔複核補充〕Strata 的字面（"insert a 60-second thinking time"）讀起來像固定值，但它又說與 Pensieve 一致；兩種讀法原文都沒排除 | ⚠️「固定」無依據（也無法排除） |
| A31 | §6.2 SAECache log-normal 參數「μ≈4.1、σ≈1.0（p17）」；同段寫成「SAECache 對 Bailian 聊天擬合的」 | p17 | 我讀的 v1 PDF 在 p6 寫 µ=4.15、σ=0.97（以 chat 為主的 trace 擬合）〔原文〕。頁碼不同，可能是版本或頁序差異。〔複核修正〕v1 的 p17 **確實**寫 µ≈4.1、σ≈1.0，頁碼沒錯；但那是「線上更新規則收斂到的值」（chat），原文說它會低估 σ。對 Bailian 的**離線 MLE 擬合**是 µ=4.82、σ=1.25（p17 Table 3、Table 4）。p6 的 µ=4.15、σ=0.97 是另一組「chat 為主的 trace」參數，原文沒說是 Bailian〔原文〕。V10 用 traceA 沿 `parent_chat_id` 重算：19,957 個間隔、9,012 個 session、P50 110.6 s、P99 2,207.3 s、log-normal MLE µ=4.82、σ=1.25，與 Table 3 完全相同〔計算，`v_bailian_gap.py`〕 | ⚠️ 頁碼對，但「對 Bailian 擬合」措辭不準：應寫 4.82／1.25（離線）或註明 4.1／1.0 是線上估計 |

**小結**：§3 的數字幾乎都能逐位重現。它的百分位用的是 `higher`（不內插）法，建議在附錄 A 寫明。真正要更新的是**版本**：TraceLab 有 v0.0.2。另外有兩個措辭要改：A22（Azure 2024 的期間）與 A30（Strata 的「固定」）。〔複核修正〕A22 已確認無誤（見該列）；要改的措辭是 A30 與 A31（SAECache 參數的語意）。

### B. 使用者文件（`intro.txt`＝RESEARCH_INTRO、`sota.txt`＝SOTA_ANALYSIS 附檔）

| # | 位置 | 使用者寫法 | 原始來源 | 判定 |
|:--|:--|:--|:--|:--|
| U1 | intro §1.3 | TraceLab 357,161 次呼叫、43 位開發者、中位約 12.4 萬、60.9% 超過 10 萬 | v0.0.1 完全相同（A11–A13）；最新 v0.0.2 為 665,453／52 人／中位 132,092／63.4% | ⚠️ 要標「v0.0.1」 |
| U2 | intro §1.3 | 兩份生產 trace 的平均請求長度 18.7K 與 32.0K；歷史區塊只占不同區塊的 37.6%，卻貢獻 70.2% 的存取 [13] | Fancy-eviction p3 的表：FreeInference 32.0 K、Chutes 18.7 K；p5：multi-turn session history 占 37.6% distinct blocks、70.2% cache accesses〔原文〕。〔複核補充〕§4 的分類只用 FreeInference 一條 trace（p4 末行到 p5：拿 FreeInference Trace 與 Wikipedia、CloudPhysics 比較）〔原文〕 | ✅（37.6%／70.2% 只來自 FreeInference；使用者句子不算錯，但若寫成兩份 trace 的共同結論就不準） |
| U3 | intro §7.1 | Strata 各資料集的平均輸入最長約 55K | Strata p9 Table 1：NarrativeQA avg in 54,797〔原文〕 | ✅ |
| U4 | intro §7.1 | HCache 最長 16K | HCache p9：把模型最大 context 擴到 16K；p3：歷史截在 16K〔原文〕 | ✅ |
| U5 | intro §7.1 | Mooncake 的 128K 是模擬資料 [11] | [11] 是 FAST'25 版。該版 p10 說真實 conversation 負載最長達 128k、平均約 12k；trace 實際最長 126,195〔原文＋計算〕。8k–128k 的固定長度只出現在 Fig. 14 的延遲分解（p13，prefix ratio 0%／95%）。〔複核補充〕V10 重算：conversation trace 有 63 筆 ≥100,000、17 筆 ≥120,000；toolagent 也有 51 筆 ≥100,000〔計算〕 | ❌ 真實 trace 本身就到約 126K；「模擬」只適用 arXiv 版或 Fig. 14（V10 複核：指控成立） |
| U6 | intro §11 | Mooncake 每個 hash block 是 512 token | A1 | ✅ |
| U7 | intro §11 | Bidaw（只有長度和時間） | README 欄位：`User_id, Timestamp, Query_length, Response_length, Round_index`〔文件〕 | ✅（也有 user id 與輪次） |
| U8 | intro §11 | Bailian 輸入中位數 574–4,540 | A6–A8 | ✅ |
| U9 | intro §11 | TraceLab（中位數 124K）；這類 agent trace 重用率 94–99.5% | 124,018 ✅（v0.0.1）。94.2%（Codex）、95.7%（TraceLab）已核；99.0–99.5%（MLPerf Agentic）本卡**未重算** | ⚠️ 部分未核 |
| U10 | intro 表 17 | Mooncake [11]：模擬 16K–128K、快取比例 50% 的請求 | 這列出自 **arXiv 2407.00079 的 Table 2**（p15：Simulated Data 16k／32k／64k／128k、輸出 512、cache 50%、Poisson）。FAST'25 版 [11] 沒有這張表，它的合成負載是 ShareGPT＋L-Eval＋LooGLE 1:1:1（p10）〔原文〕。〔複核補充〕p15 是 arXiv **v4**；v1 在 p13。V10 在 FAST'25 全文搜尋 "simulat"、"50%"、"16k"：沒有任何 16K–128K、cache 50% 的負載；唯一的 8k–128k 是 Fig. 14 的 prefix ratio 0%／95%（p13）〔原文〕 | ❌ 版本錯置：引用 FAST'25，內容卻是 arXiv 版（V10 複核：指控成立） |
| U11 | intro 表 17 | HCache [50]：真實的長文件，以 Zipf 分布合成到達 | HCache p10：ShareGPT4 以 Poisson 產生 session 到達，同一 session 輪間 30 s；L-Eval 主實驗抽 200 個請求（p10），到達過程原文段落未讀到；Zipf（α=1.2–2 與 uniform）只用在「GPU 上重用 KV」的子實驗（p13，Fig. 15）〔原文〕。〔複核補充〕L-Eval 主實驗的設定在 p10 6.1.2：GPU 只放得下 1–3 個長請求，所以 **batch size＝1**，沒有到達過程。p13 的 Zipf 是「合成 L-Eval 各 context 的到達模式」（哪份 context 被請求），對照組 uniform 的命中率只有 15%〔原文〕 | ⚠️ 過度簡化（V10 複核：指控成立；主實驗不是 Zipf，Zipf 只在子實驗） |
| U12 | sota §2.5 Bidaw | 超過一百萬輪的真實聊天 trace；trace 公開 | p4「more than one million conversation rounds」；p11 腳註指向 repo〔原文〕；1,268,346 輪〔計算〕 | ✅ |
| U13 | sota §2.5 Bidaw | 原文承認換成 Poisson 到達就失效；問題平均只有 36 token | p12：ShareGPT 用 Poisson 模擬時間後，previous-answer 驅逐不再降低 miss rate；p4：average query length is 36〔原文〕 | ✅ |
| U14 | sota §2.6 MTDS | 用 ShareGPT 加 Poisson 到達 | MTDS p10：ShareGPT，Poisson（λ=1）；另有固定長度的 Random 資料集〔原文〕 | ✅ |
| U15 | sota 表 S4 | Bidaw「含歷史的 prompt 中位 1,066、p99 8,168」 | A19：1,066、8,168〔計算〕 | ✅ |
| U16 | sota 表 S4 註 | 重用率為無限容量下的估計 | Mooncake、Bailian 的區間是無限容量估計（A4、A10）✅。但 TraceLab 95.7% 是 provider 實際命中（A14），Codex 94.2% 是資料卡報的實際 cache hit（A25），**都不是無限容量估計**。〔複核補充〕V10 重算 TraceLab：Claude 列的 `prefix_tokens` 與 `claude_cache_read_input_tokens` 在 140,338／140,338 列相同，Σprefix÷Σinput＝95.75%；Codex 資料卡原文是 "Overall cache hit rate: 94.2% (1,301.5M of 1,382.3M input tokens served from cache)"（commit `0d52ae8c`）〔計算＋文件〕 | ❌ 註腳對兩列不成立（V10 複核：指控成立） |

---

## 資料集卡

> 每張卡的「誰怎麼用」只列我讀過原文的論文。列不到的不代表沒人用。

### C1｜ShareGPT：使用者分享的 ChatGPT 對話（`ShareGPT_V3_unfiltered_cleaned_split.json` 為主）

- **讀了什麼**：〔文件〕原 repo README、`split_long_conversation.py`（commit `192ab218`）；〔計算〕sha256 相同的鏡像 parquet 全量 94,145 筆；〔程式碼〕vLLM `datasets.py`、`serve.py`（commit `31e2443c`）；〔原文〕上列 11 篇 serving 論文。
- **一句話**：sharegpt.com 爬下的 ChatGPT 對話。Vicuna 訓練時清洗並切段，後來被 serving 社群當成「真實聊天長度分布」的預設負載。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 是什麼 | 使用者用 ShareGPT 外掛分享的 ChatGPT 對話。vLLM 描述為 user-shared conversations with ChatGPT。HF repo 2023-04-02 建立，最後 commit 2023-04-12 | vLLM p10〔原文〕；HF API〔文件〕 |
| 版本與清洗 | README：約 100k 段對話，篩到 53k。做法：去非英文、去過多 unicode、去重複字元、刪除含 100 多個「AI 說教」關鍵詞的對話，再依 FastChat 流程切成 2048-token 的段。另有 `_no_imsorry` 版。切段腳本的 `--max-length` 預設 2304（與 README 的 2048 不一致；實際用的值未知） | README、`split_long_conversation.py`〔文件＋程式碼〕 |
| 規模 | V3 split 檔 672,837,942 bytes、**94,145 筆**。其中 550 筆空對話、92,886 筆 ≥2 則；去掉 `_k` 後綴後有 50,142 個原始對話，其中 13,488 個被切成多段 | 〔計算〕鏡像 `learnanything/sharegpt_v3_unfiltered_cleaned_split`（LFS sha256 與原檔同為 `35f0e213…`），datasets-server `num_rows=94145`；WildChat Table 1（p2）也列 ShareGPT 94,145〔二手〕 |
| 結構 | `id`、`conversations[{from, value}]`。第一則的說話者：human 58,827、**gpt 34,538（36.7%）**、空 550、system 212、user 12、chatgpt 4。43,459 筆的 id 是 `_k`（k>0）續段。〔複核補充〕另有 bing 2 筆（合計才是 94,145）。切段腳本 `split_long_conversation.py` 只依累積 token 數切，L61 的 TODO 註解自問「是否只從特定說話者開始」，也就是沒有強制從 human 開始，這直接解釋了 36.7% 從 gpt 開始 | 〔計算〕`sharegpt_stats.py`；V10 `v_sharegpt.py` 重算一致；〔程式碼〕腳本 commit `192ab218` |
| 長度 | 每筆把所有則串起來（不套 chat template），Qwen2.5 tokenizer：平均 1,437、中位 1,655、p90 2,257、p99 2,873、最大 221,060；>4,096 占 0.27%；≥16K 占 0.018%；≥32K 占 0.006%。每筆平均 7.46 則（中位 6） | 〔計算〕全量，非抽樣。〔複核〕V10 用 `Qwen/Qwen2.5-7B-Instruct` 的 `tokenizer.json`（git blob 與 HF 相同），每則分別 tokenize 後相加、不加特殊 token，逐位重現這組數字；改用 `\n` 串接得 p99 2,873.6，結論不變 |
| vLLM 論文報的長度 | Fig. 11：ShareGPT input 平均 161.31、output 337.99；Alpaca 19.31／58.45。原文說 ShareGPT 的 input 長 8.4 倍、output 長 5.8 倍 | vLLM p9 Fig. 11、p10〔原文〕 |
| 重現 vLLM 的數字 | 用 OPT tokenizer 試三種組法都**沒重現** 161.31／337.99：①vLLM 現行取樣（前兩則＋過濾）得 228.27／226.21；②所有 human→gpt 配對 87.13／327.44；③配對＋vLLM 過濾 56.02／326.79。Alpaca 則重現得到（C2）。所以論文當時怎麼組 ShareGPT 請求，**無法從原文確定** | 〔計算〕 |
| 時間戳、多輪、身分 | **無時間戳、無使用者身分**；有多輪，但被切段 | 檔案結構〔計算〕；vLLM p10、DistServe p10、Bidaw p12 都說沒有時間戳〔原文〕 |
| License | HF 標 `apache-2.0`。內容爬自第三方網站，實際權利〔未查證〕 | HF API〔文件〕 |
| 為什麼大家用（原文） | vLLM：含真實 LLM 服務的輸入輸出文字，且比 Alpaca 長、變異大（p10）。Strata：用來測短 context，且先前的分層 KV 研究用過（p10）。CachedAttention：73% 的對話是多輪（p2） | 〔原文〕 |
| 為什麼大家用〔判讀〕 | vLLM 論文與 benchmark 工具把它變成預設，下載只要一行 `wget`（vLLM `docs/benchmarking/cli.md` L22）。用它方便與前作比較，審稿人也熟悉 | 〔判讀〕＋〔文件〕 |

**vLLM benchmark 工具怎麼處理 ShareGPT**（〔程式碼〕vLLM commit `31e2443c`，`vllm/benchmarks/datasets/datasets.py`）
* `ShareGPTDataset.load_data`（L1329 起）：只留 `conversations` 至少 2 則的條目；用 seed（`DEFAULT_SEED=0`，L102）打亂。
* `sample`：每筆**只取 `conversations[0]` 當 prompt、`conversations[1]` 當 completion**，不檢查 `from`。
* `is_valid_sequence`（L342–365）丟掉以下四種：prompt <4、output <4（指定 `output_len` 時不檢查）、prompt >1024、prompt＋output >2048。註解說這些門檻沿用舊的 `benchmark_serving.py`。
* 樣本不足時，`maybe_oversample_requests`（L291 起）會**有放回地重抽**補足。
* 結果〔計算，OPT tokenizer、加 1 個 BOS〕：92,886 筆中 81,267 筆通過（87.5%）；通過者 prompt 平均 228、中位 97、最大 1,024；output 平均 226。**通過者有 35.5% 的第一則不是 human。**〔複核〕V10 用 OPT 的 `vocab.json`＋`merges.txt` 自建 GPT-2 BPE（prompt 與 completion 各加 1 個 BOS，對應 `tokenizer(...).input_ids`）逐位重現；配對分布為 human→gpt 64.4%、gpt→human 35.3%、system→human 0.2%、其他 <0.1%。結果依 tokenizer 而變：改用 Qwen2.5 時通過 82,744 筆、非 human 開頭 37.0%。
* 到達：`serve.py` L405–491。`--burstiness 1.0` 為 Poisson；其他值為 Gamma（shape＝burstiness、scale＝1/(rate×burstiness)）；`inf` 為固定間隔。〔複核補充〕`get_request` 在 L400；沒有 ramp-up 時，L483–497 會把累積間隔整體縮放，使最後一個請求剛好落在 N/rate 秒（註解說為了穩定不同 seed 的吞吐）。所以它不是嚴格的 Poisson 過程，而是「總時長固定」的版本〔程式碼，commit `31e2443c`〕。

**誰怎麼用（serving 論文）**

| 論文 | 取樣與前處理 | 截斷／過濾 | 到達 | 出處 |
|:--|:--|:--|:--|:--|
| vLLM（SOSP'23） | tokenize 後只取輸入輸出長度來合成請求；多數實驗跑 1 小時的 trace（OPT-175B 為 15 分鐘） | 未說明 | Poisson，掃 request rate | p10〔原文〕 |
| vLLM chatbot 實驗 | 用 ShareGPT 合成「歷史＋最後一句」當 prompt；**不在輪間保留 KV** | prompt 截到最後 1024 token，最多生成 1024 | 同上 | p12〔原文〕 |
| DistServe（OSDI'24） | chatbot 應用；從資料集抽樣請求 | 未說明 | Poisson | p10〔原文〕 |
| Llumnix（OSDI'24） | ShareGPT（GPT4）的輸入輸出長度；每條 trace 10,000 個請求；長度表 In 平均 306／P50 74、Out 平均 500 | 未說明 | Poisson 與 Gamma（調 CV） | p9 Table 1〔原文〕 |
| Sarathi-Serve（OSDI'24） | `openchat_sharegpt4`；每一輪當成一個請求（含歷史）；prompt 中位 1,730、P90 5,696 〔複核修正〕原文只說每一輪互動是一個獨立請求、多輪造成 prompt 長度變異大（p11）；「含歷史」是推論，原文沒明說〔判讀〕 | 去掉總長 >8,192 | Poisson | p11 Table 2〔原文〕 |
| CachedAttention（ATC'24） | 9K 個 session，平均 5.75 輪，約 52K 輪；前 10K 輪暖機，評 42K 輪 | 超過 context window 時，對照組丟掉最早的一半 token | **session** 到達為 Poisson（λ=1.0/s） | p10〔原文〕 |
| Pensieve（EuroSys'25） | 48,159 段、平均 5.56 輪；請求 input 平均 37.77（只算新增部分）〔複核修正〕Table 2 只寫 "Mean request input length"，「只算新增部分」是由數值推的〔判讀〕 | 最大 context 16,384，丟 0.57% 的對話 | Poisson；同一對話的下一輪要等上一輪回應完成，再加**指數分布**的 think time（預設平均 60 s，敏感度掃到 600 s） | p9 Table 2、p12〔原文〕 |
| Mooncake（FAST'25） | 合成負載的「短對話」成分；每一輪映射成一個請求，含先前所有輸入輸出；與 L-Eval、LooGLE 以 1:1:1 混合；保留多輪先後順序，其餘打亂 | 未說明 | Poisson | p10〔原文〕 |
| Strata（OSDI'26） | 短 context 對照組；200,869 個 query、avg in 680.9；保留輪間依賴 | 把 GPU 記憶體限制在約 500K token 以製造壓力 | Poisson＋60 秒 thinking time（說依照 Pensieve）。〔複核補充〕所有資料集的 in-flight 請求上限 128（原文強調是併發上限，不是固定併發） | p9 Table 1、p10〔原文〕 |
| Bidaw（FAST'26） | 公開對照負載；原文說 ShareGPT 平均 5.7 輪 | 未說明 | Poisson 模擬時間；結果增益縮小 | p4、p11、p12〔原文〕 |
| HCache（EuroSys'25） | ShareGPT4（GPT-4 對話）；歷史長度 CDF 截在 16K，一半對話超過 2.5K | context 上限 16K | Poisson 的 session 到達；輪間固定 30 s | p3、p9、p10〔原文〕 |
| Marconi（MLSys'25） | 多輪對話；原文說 ShareGPT 的輸出短（幾十到幾百 token） | 未說明 | 調整 session 間與請求間的到達時間 | p8〔原文〕 |
| SAECache（arXiv'26） | 74% 的請求屬於多輪 session | 未說明 | **固定注入間隔** 0.02／0.03／0.05／0.08 s | p8〔原文〕 |
| MTDS（CIS'26） | ShareGPT＋固定長度的 Random 資料集。〔複核補充〕它用的 ShareGPT 是 HF `shibing624/sharegpt_gpt4`（p15 資料可得性聲明），又是另一個版本 | 未說明 | Poisson（λ=1） | p10、p15〔原文〕 |

**陷阱**
1. **版本混亂。** 「ShareGPT」至少有 V3 split（94,145）、`_no_imsorry`、ShareGPT4／openchat、原始 90k 等版本。vLLM [51]、DistServe [8]、Bidaw [39] 都只引用 `sharegpt.com`，無法得知用哪個檔。各篇報的統計因此對不上：Bidaw 平均 5.7 輪（p4）、Pensieve 48,159 段 5.56 輪、CachedAttention 9K 段 5.75 輪、WildChat 表 94,145 段 3.51 輪〔原文〕。
2. **切段破壞多輪鏈。** 36.7% 的條目從 gpt 開始；13,488 個原始對話被拆成多筆〔計算〕。把每筆當成獨立 session，會把同一段對話的續段誤當成新 session。
3. **vLLM 工具讀出的是單輪負載。** 只取前兩則、說話者不檢查、總長 ≤2048，而且沒有重用。用它量 prefix cache 或分層 KV 沒有意義。
4. **選擇偏差。** README 刪掉非英文與含倫理字詞的對話，長度分布不是原始使用者流量〔文件〕〔判讀〕。
5. **長度天花板。** 切段之後 p99 只有約 2.9K token〔計算〕，不可能代表 16K 以上的情境。

**與既有整理不一致**：`workloads_eval` §3.2 把「多輪鏈可能不完整」標為判讀。現在可以改成計算結果：43,459／94,145 筆是續段，36.7% 以 gpt 開頭。

**對本研究的意義〔判讀〕**：只能當「短 context、無重用」的負對照。若審稿人要求跑，要：①從 human 開始重組 session；②用完整歷史的多輪模式；③時間從真實 trace 移植；④寫明檔名、sha256、tokenizer 與過濾掉的比例。

---

### C2｜Alpaca：Stanford Alpaca 52K 指令資料

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 是什麼 | 用 `text-davinci-003`、改造過的 Self-Instruct 流程生成的 52K 條指令與回答；每條指令只生成一個實例 | HF 資料卡（commit `dce01c9b`）〔文件〕；vLLM p10 描述為 GPT-3.5 以 self-instruct 生成〔原文〕 |
| 規模 | 52,002 列（`instruction, input, output, text`） | datasets-server〔文件〕；WildChat Table 1 同為 52,002〔二手〕 |
| 長度 | OPT tokenizer（加 BOS）：prompt（instruction＋input）平均 19.15、中位 15、最大 508；output 平均 **58.45**、中位 43、最大 1,470。Qwen2.5：17.66／55.85 | 〔計算〕全量。〔複核〕V10 重現；這組數字對應「instruction＋換行＋input（input 空時只用 instruction）」；直接相連得 18.74，用空格得 18.71（parquet sha256 `06391b65…` 與 HF 相同） |
| 與 vLLM 論文比對 | 論文 Fig. 11：input 19.31、output 58.45。**output 平均與我的 OPT 結果完全相同**；input 差 0.16，可能是 instruction 與 input 的串接方式不同〔判讀〕 | vLLM p9〔原文〕＋〔計算〕 |
| 時間戳、多輪、身分 | 無、單輪、無 | 欄位〔文件〕 |
| License | CC BY-NC 4.0（非商用） | 資料卡〔文件〕 |
| 誰怎麼用 | vLLM：與 ShareGPT 並列為主負載，長度取自 tokenize 後的資料、Poisson 到達（p10）；parallel sampling 與 beam search 實驗只用 Alpaca（p11 Fig. 14）〔複核修正〕Fig. 14、15 畫的是 Alpaca，但 p11 正文另報同一組實驗在 ShareGPT 上的記憶體節省（parallel sampling 16.2–30.5%、beam search 44.3–66.3%），所以不是「只用 Alpaca」；block size 實驗用 ShareGPT 與 Alpaca，在固定 request rate 下比較（p12） | 〔原文〕 |
| 為什麼用（原文） | 與 ShareGPT 形成「短輸入短輸出」對比（p10）；長度很短，block size 太大時會浪費（p12） | 〔原文〕 |
| 為什麼用〔判讀〕 | 短而整齊，適合把「每步 batch 數」推到很高，凸顯記憶體管理的效果 | 〔判讀〕 |
| 陷阱 | 不是真實使用者流量；長度極短；非商用授權；沒有重用與時間 | 〔文件〕〔判讀〕 |
| 對本研究的意義〔判讀〕 | 與 16K–512K 研究無關，不建議使用 | 〔判讀〕 |

---

### C3｜LMSYS-Chat-1M（arXiv 2309.11998，ICLR'24）

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 是什麼 | Vicuna demo 與 Chatbot Arena 網站上的真實對話，2023 年 4–8 月收集，涵蓋 25 個模型 | p1–p2〔原文〕；資料卡〔文件〕 |
| 規模 | 1,000,000 段、210,479 個 user（以 IP 計）、154 種語言 | p3 Table 1〔原文〕 |
| 長度 | 平均 2.0 輪；prompt 平均 69.5、回應 214.5 token（Llama 2 tokenizer） | p3 Table 1〔原文〕 |
| 時間戳、多輪、身分 | features 只有 `conversation_id, model, conversation, turn, language, openai_moderation, redacted`：**沒有時間戳，也沒有 user id** | 資料卡 YAML（commit `200748d9`）〔文件〕 |
| License | gated；需同意 LMSYS-Chat-1M Dataset License Agreement，其中禁止把資料轉給第三方 | 資料卡〔文件〕 |
| 誰怎麼用 | **Vidur（MLSys'24）**：Chat-1M 共 2M 個 query，prefill 平均 786、中位 417、p90 1,678，decode 平均 215；另有「總長 ≤4K」的版本（p7 Table 1）。**Marconi**：多輪對話負載，原文說其輸出常到幾千 token（p8）。**SAECache**：33% 的請求屬於多輪（p8） | 〔原文〕 |
| 為什麼用（原文） | 原文強調規模大、模型多、多樣（p3）；資料卡列出「模型選擇與請求分派」等用途 | 〔原文〕〔文件〕 |
| 為什麼用〔判讀〕 | 比 ShareGPT 新、規模大、有官方來源與統計 | 〔判讀〕 |
| 陷阱 | ①沒有時間與使用者，多輪只存在於同一段對話內。②原文發現部分叢集有大量同模板的樣本，疑似腳本批次送出（p4），會製造人為的前綴重用〔原文〕〔判讀〕。③禁止再散布，衍生出的 trace 不能公開。④短 | 〔原文〕〔文件〕〔判讀〕 |
| 對本研究的意義〔判讀〕 | 不適合當主負載；授權也妨礙公開可重現的負載 | 〔判讀〕 |

---

### C4｜Chatbot Arena conversations（`lmsys/chatbot_arena_conversations`；arXiv 2306.05685）

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 是什麼 | Chatbot Arena 的匿名對戰對話與人類偏好投票，2023 年 4–6 月，13K 個 IP | 資料卡（commit `1b6335d4`）〔文件〕；論文說公開 30K 段帶偏好的對話（p1、p3）〔原文〕 |
| 規模與長度 | 33,000 段、20 個模型、13,383 個 user、96 種語言；平均 1.2 輪；prompt 52.3、回應 189.5 token | 資料卡；LMSYS-Chat-1M p3 Table 1〔文件＋原文〕 |
| 時間戳、多輪、身分 | 有 `tstamp`；`judge` 為匿名 user id；每筆含兩個模型的對話（`conversation_a/b`）與勝負 | 資料卡〔文件〕 |
| License | gated（auto）；使用者 prompt CC-BY-4.0，模型輸出 CC-BY-NC-4.0 | 資料卡〔文件〕 |
| 誰怎麼用 | SAECache：只有 12% 的請求屬於多輪，用來代表「以單輪為主」的流量；到達同樣是固定注入間隔（p8） | 〔原文〕 |
| 為什麼用〔判讀〕 | 公開對話資料中少數有時間戳的；規模小、方便 | 〔判讀〕 |
| 陷阱 | ①幾乎是單輪。②同一個 prompt 同時送給兩個模型，直接展開成請求會製造人為的前綴重複〔判讀，未逐筆驗證〕。③時間只涵蓋約 2 個月、33K 段，流量稀疏 | 〔文件〕〔判讀〕 |
| 對本研究的意義〔判讀〕 | 不適合 | 〔判讀〕 |

---

### C5｜WildChat（arXiv 2405.01470；HF `allenai/WildChat-1M`、`WildChat-4.8M`）

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 是什麼 | 免費提供 GPT-3.5／GPT-4 的聊天服務，收集使用者同意公開的對話；2023-04-09 到 2024-05-01 | p2〔原文〕 |
| 規模（論文） | 公開版 1,039,785 段（2,639,415 輪）、204,736 個 IP；平均 2.54 輪；user 每輪 295.58±1609.18、chatbot 441.34±410.91 token（Llama-2 tokenizer） | p2 Table 1、p3〔原文〕 |
| 規模（HF 現況） | WildChat-1M：**837,989** 段（2024-07-22 移除所有 toxic 對話，2024-10-17 再移除 PII）。WildChat-4.8M（2025-08-08 建立）：**3,199,860** 段非 toxic 對話，來自 4,804,190 段原始資料，涵蓋到 2025-08-01，含 111,836 段 o1 對話 | datasets-server；資料卡（commit `7d6490e4`、`c827c6df`）〔文件〕 |
| 時間戳、多輪、身分 | 對話層的 `timestamp` 是最後一輪的時間；assistant 回合有 `timestamp`，意義是後端收到完整回應的時間；user 回合有 `hashed_ip`、`header`，資料卡說兩者可一起用來串同一使用者的多段對話 | 資料卡〔文件〕 |
| License | ODC-BY；兩版 ungated（含 toxic 的 Full 版 gated） | HF API〔文件〕 |
| 長度分布 | 〔未計算〕。論文的 user token 標準差 1,609，表示有長貼文的長尾〔判讀〕 | — |
| 誰怎麼用 | 我讀過的 serving 論文中沒有人用；`workloads_eval` 的 35 篇也是 0 篇 | 〔原文〕（反向查證） |
| 為什麼值得考慮〔判讀〕 | 公開資料中同時有內容、逐輪時間、使用者串連的聊天集 | 〔判讀〕 |
| 陷阱 | ①時間是**回應完成**的時間，不是請求到達；到達時間要用「上一輪回應完成＋使用者時間」反推〔判讀〕。②移除 toxic 是整段移除，使用者的 session 會有缺口。③用 IP＋header 串使用者是啟發式。④聊天流量，長度仍以短為主〔判讀〕 | 〔文件〕〔判讀〕 |
| 對本研究的意義〔判讀〕 | 適合當「人類節奏的輪間時間」來源，搬到別的內容上；不適合當長 context 主負載 | 〔判讀〕 |

---

### C6｜OpenOrca（`Open-Orca/OpenOrca`；MLPerf Llama-2-70B 與 Mixtral 用）

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 是什麼 | 把 FLAN Collection 的問題送給 GPT-4 或 GPT-3.5，取得 reasoning 式回答，用來重現 Orca 論文（2306.02707）的資料 | 資料卡（commit `e9c87b4a`）〔文件〕 |
| 規模 | 約 1M 筆 GPT-4、約 3.2M 筆 GPT-3.5 completions（資料卡）。datasets-server 顯示 2,942,029 列，但標 `partial=True`，**不是全量** | 〔文件〕 |
| 欄位 | `id`（含 niv／t0／cot／flan 子集）、`system_prompt`、`question`、`response`；單輪 | 資料卡〔文件〕 |
| License | MIT | HF API〔文件〕 |
| 時間戳 | 無 | 〔文件〕 |
| **MLPerf Llama-2-70B 怎麼用** | 取 `1M-GPT4-Augmented.parquet`，用 `processorca.py`：①濾掉非 ASCII；②input、output token 各需 <1024（`io_token_limit`，預設 `--seqlen_limit=1024`）；③濾掉預期答案少於 2 個字的；④濾掉 Llama2 會產生壞輸出的；⑤從 COT、NIV、FLAN、T0 四個子集平均抽樣，共 **24,576** 筆 | `language/llama2-70b/README.md` L132–151、`processorca.py` L129–135、L302–345（commit `3fbc3299`）〔程式碼＋文件〕 |
| MLPerf 規則 | Llama2-70b 用 OpenOrca（max_seq_len=1024）、24,576 筆；準確度門檻為 rouge1=44.4312 等的 99.9%，且 tokens per sample 需達參考值 294.45 的 90% 以上；conversational 類 TTFT／TPOT 2000／200 ms，interactive 類 450／40 ms。Mixtral-8x7B 用 OpenOrca 5k（max_seq_len 2048）＋GSM8K＋MBXP。Server 情境由 LoadGen 以 **Poisson** 送 query | `inference_rules.adoc` L141、L267、L269（commit `d3eba2f2`）〔文件〕 |
| 為什麼用（原文） | 規則文件只給設定，沒寫選它的理由 | 〔文件〕 |
| 為什麼用〔判讀〕 | 有參考答案可算 ROUGE，當準確度 gate；授權寬鬆 | 〔判讀〕 |
| 陷阱 | ①長度上限 1024，屬短 context。②`system_prompt` 只有少數幾種，會製造跨請求的共同前綴〔判讀，未計算〕。③合成資料，不是使用者流量 | 〔判讀〕 |
| 對本研究的意義〔判讀〕 | 只當「業界標準 benchmark 長什麼樣」的參考 | 〔判讀〕 |

---

### C7｜BurstGPT（arXiv 2401.17644v5，KDD'25；`HPMLL/BurstGPT`）

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 是什麼 | 某區域 Azure OpenAI GPT 服務供應商（服務企業、校園，超過 3,000 位使用者）的請求 log；只記長度與時間，不含內容 | p1–p2〔原文〕；README〔文件〕 |
| 規模 | 論文：10.31M 筆、213 天。README：BurstGPT_1＋_2 共 121 天、約 5.29M 行；BurstGPT_3 另 110 天、約 5.34M 行。**論文與 README 的總數、天數對不上**（10.63M 行、231 天）〔未查證原因〕 | 〔原文〕〔文件〕 |
| 欄位 | `Timestamp`（秒，從第一天 0:00 起算）、`Session ID` 與 `Elapsed time`（v2.0 起，只有 BurstGPT_3 有）、`Model`（ChatGPT／GPT-4）、`Request tokens`、`Response tokens`、`Total tokens`、`Log Type`（Conversation log／API log） | README〔文件〕；`BurstGPT_3.csv` 表頭〔計算〕 |
| 長度（BurstGPT_3） | 5,344,021 列；API 5,110,404、Conversation 233,617；GPT-4 790,239。request 中位 327、平均 478、最大 125,591；≥32K 只占 0.029% | 〔計算〕 |
| 多輪（BurstGPT_3） | 55,920 個 session，每個中位 2 輪、平均 4.18 輪；session 內輪間隔 p50 131 s、p90 2,340 s | 〔計算〕 |
| License | CC-BY-4.0 | GitHub API〔文件〕 |
| 原文的到達模型 | 以 Gamma 分布描述突發性，參數隨服務類型變化（p3–p4）；請求長度符合 Zipf（p4–p5）。BurstGPT-Perf 提供兩種用法：RPS 縮放，或用 Gamma＋Zipf 參數生成。demo 設 α=0.5、β=2（p5–p6） | 〔原文〕 |
| 誰怎麼用 | **Llumnix**：取 BurstGPT（GPT4-Conversation）的長度（In 平均 830、Out 平均 271），到達仍用 Poisson／Gamma（p9 Table 1）。**vLLM `BurstGPTDataset`**（L3178–3250）：只留 GPT-4 列且 `Response tokens>0`，隨機抽列取長度，prompt 用 `(i+j) mod vocab` 的合成 token，**不用時間戳**。〔複核補充〕它用**欄位位置**取長度（`int(data[i][2])`、`int(data[i][3])`，L3229–3230）。舊格式（BurstGPT_1／_2）第 2、3 欄是 Request／Response tokens；但 BurstGPT_3 多了 `Session ID`、`Elapsed time` 兩欄，第 2 欄變成 `Elapsed time`、第 3 欄是 `Model` 字串，直接餵 _3 會讀錯欄或在 `int("GPT-4")` 報錯〔程式碼，commit `31e2443c`；判讀：未實際執行〕。vLLM 文件給的下載連結是 v1.1 的 `BurstGPT_without_fails_2.csv`（`docs/benchmarking/cli.md` L22–26） | 〔原文〕〔程式碼〕 |
| 為什麼用（原文） | 公開 serving 負載不足；BurstGPT 能揭露突發變化下的效率與穩定性問題（p1） | 〔原文〕 |
| 陷阱 | ①沒有內容，算不出重用；vLLM 工具的合成 prompt 還會讓 prefix cache 的命中成為假象〔判讀〕。②請求很短。③session 欄位只有 BurstGPT_3 有。④論文與 release 的規模不一致 | 〔計算〕〔判讀〕 |
| 對本研究的意義〔判讀〕 | 可當「真實突發性與輪間時間」的來源；長度與重用不可用 | 〔判讀〕 |

---

### C8｜Azure LLM inference traces（2023：Splitwise，ISCA'24；2024：DynamoLLM，HPCA'25）

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 是什麼 | Azure 兩個 LLM 推論服務（coding 與 conversation）的請求樣本；只有 `TIMESTAMP, ContextTokens, GeneratedTokens` | 兩份 md〔文件〕；Splitwise p3〔原文〕 |
| 2023 版 | md 寫 2023-11-11 收集；檔內時間是 **2023-11-16** 18:15–19:14（約 1 小時）。code 8,819 筆，context 中位 1,469、最大 7,437、生成中位 13；conv 19,366 筆，中位 1,020、最大 14,050、生成中位 129。Splitwise 原文的中位數（prompt 1,020；生成 13 與 129）與檔案相符。〔複核補充〕但 Splitwise p3 寫 coding 的 prompt 中位是 **1500**，檔案是 1,469；原文的特徵分析用 20 分鐘 trace，公開的是「a subset」（p3），兩者本來就不是同一份 | 〔計算〕〔文件〕；Splitwise p3〔原文〕 |
| 2024 版 | md：2024-05-10～19；release 的 `AzureLLMInferenceTrace_code_1week.csv` 692 MB、`_conv_1week.csv` 1.14 GB，**未下載**。〔複核補充〕HTTP Range 讀檔頭檔尾：code 5/10 00:00 → 5/16 23:59:59、conv 5/12 00:00 → 5/18 23:59:59，各 7 天，時間戳到微秒並帶 `+00:00`（見 A22）。DynamoLLM 原文：一週的 coding 與 conversation，是公開 trace 的超集；有明顯的日週期（conv 尖峰是平均的 1.7 倍、谷底的 3.3 倍；coding 為 2.8 倍與 34.6 倍） | md、release API〔文件〕；DynamoLLM p2、p4〔原文〕 |
| License | CC-BY-4.0 | 〔文件〕 |
| 誰怎麼用 | **Splitwise**：特徵分析用 20 分鐘的 trace（p3）；單機實驗用縮小到 2 req/s 的版本（p4）；叢集評估**只用長度分布，到達用可調的 Poisson rate**（p9）。**DynamoLLM**：特徵分析用一週的 trace（p2、p4）；評估圖用「1 小時的公開 trace」（p9 Fig. 6–9）。**EvicPress**：用 Azure 的**真實時間戳**，內容換成自己生成的請求（p10）。**Bidaw**：引用為「缺對話或使用者資訊」的例子（p4 引用 [5]） | 〔原文〕 |
| 為什麼用（原文） | 公有雲的生產 trace；md 說 prompt 內容不影響他們量的指標，只有長度會 | md〔文件〕 |
| 陷阱 | ①「內容不影響指標」在有 prefix cache 或分層 KV 時**不成立**〔判讀〕。②md 寫的日期與檔內時間不一致。③沒有 session 與 hash。④2023 版只有 1 小時 | 〔計算〕〔判讀〕 |
| 對本研究的意義〔判讀〕 | 可當日週期與突發的時間來源（2024 版）；長度短，重用無法算 | 〔判讀〕 |

---

### C9｜Mooncake traces（FAST'25；`kvcache-ai/Mooncake/FAST25-release`）

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 是什麼 | Kimi 線上叢集取樣的請求 trace：只有 `timestamp`（ms）、`input_length`、`output_length`、`hash_ids`；不含文字 | README〔文件〕；FAST'25 p10〔原文〕 |
| 檔案 | `traces/` 三個檔是 FAST'25 版：conversation、toolagent（真實，各取樣 1 小時）、synthetic（公開資料合成、Poisson 到達）。`arxiv-trace/mooncake_trace.jsonl` 是 arXiv 技術報告的舊版 | README（commit `1d0e4c75`）〔文件〕 |
| 規模與長度 | conversation 12,031 筆，平均 12,035、中位 6,909、p90 27,367、最大 126,195，≥32K 6.89%。toolagent 23,608 筆，平均 8,596、中位 6,346、最大 126,195，≥32K 3.08%。synthetic 3,993 筆，平均 15,325、中位 11,587、p90 38,611、最大 191,378，≥32K 16.15%。舊版 23,608 筆、最大 125,546；真實 trace 跨度約 0.98–1.00 h | 〔計算〕；平均值與 README、FAST'25 Table 2（p10）相同 |
| 單位 | 512 token 的前綴 block hash；相同 id 代表可共用的前綴。斷言 `ceil(input/512)==len(hash_ids)` 四檔皆 100% | README〔文件〕；〔計算〕 |
| 重用 | 原文 Table 2 cache ratio：40%（conv）、59%（toolagent）、66%（synthetic）。無限容量 token 級：37.4%、57.1%、65.1% | p10〔原文〕；〔計算〕 |
| 時間、多輪、身分 | ms 時間戳，要依時間重播；**沒有 session 或 user id**，多輪只能由 hash 推斷 | README〔文件〕。〔複核補充〕repo 附的 `Mooncake-FAST25.pdf` 附錄 A.1（p14）說：conversation 與 tool&agent 是從**不同叢集**各取樣 1 小時，而且取樣時**優先收同一使用者 session 的請求**，以保留快取關係〔原文〕。所以 trace 的重用率可能高於隨機取樣的線上流量〔判讀〕 |
| License | repo 為 Apache-2.0；trace 沒有另外標示 | `workloads_eval` §3.1；本卡未另查。〔複核〕V10 查 GitHub API：repo license＝Apache-2.0；`FAST25-release/README.md`（blob `c4460e62`）沒有任何 license 段落〔文件〕 |
| 原文怎麼用（FAST'25） | 依時間戳送出請求，輸出達到預定長度就強制結束（p10）。synthetic：ShareGPT、L-Eval、LooGLE 各自處理後以 1:1:1 混合；每一輪變成一個含歷史的請求；同一長 prompt 的多個問題各自成為一個請求；保留多輪順序，其餘打亂；Poisson（p10）。Fig. 14 另用 8k–128k 的固定長度，prefix ratio 0% 與 95%（p13） | 〔原文〕 |
| arXiv 版怎麼用（不同！） | Table 2：ArXiv Summarization（avg in 8,088、cache ≈0%）、L-Eval（19,019、>80%）、Simulated Data（16k／32k／64k／128k、輸出 512、cache 50%）、Real Data（7,955、約 50%、23,000 個請求，依時間戳）；除 Real Data 外都是 Poisson | arXiv 2407.00079 p15〔原文〕 |
| 誰怎麼用 | **Strata**：toolagent 有 38% 的請求共用至少 6k token 的前綴（p5）〔複核修正〕原文有兩個條件不能省：前綴**不含系統提示**，而且是與**一秒內到達**的另一個請求共用（用來說明 delay hit）。**Bidaw**：引用 conversation 的平均 query 12,035、response 343（p4）。**Fancy-eviction**：在公開 trace 比較表中把它列為粗粒度（512 token）（p5 Table 3） | 〔原文〕 |
| 為什麼用（原文） | 原文說 conversation 負載含大量長 context 請求，與當前長 context 資料集的長度相當（p10）；toolagent 有大量重複的長系統提示（p10） | 〔原文〕 |
| 陷阱 | ①**block 粒度 512**：直接拿去配 vLLM 的 16-token block，工作集與位置會差 32 倍（專案 CLAUDE.md 規則 6 的教訓）。②只有 1 小時，看不到長間隔的重用。③conversation 與 toolagent 的 hash id 都從 0 開始，跨檔是否同一命名空間未說明，不應合併計算〔判讀〕。④沒有 session id。⑤FAST'25 與 arXiv 版的負載不同，引用時要對版本。〔複核補充〕③可以下定論：三個 FAST'25 檔的 id 各自是連續的 0…N−1（conversation 0–182,789、toolagent 0–183,299、synthetic 0–43,923）〔計算〕，加上附錄 A.1 說兩條真實 trace 來自不同叢集（repo PDF p14）〔原文〕，所以是逐檔重新編號，**跨檔相同 id 不代表相同內容**。⑥vLLM 內建的 `timed_trace` 資料集（commit `31e2443c`）欄位名預設就是 Mooncake 的 `timestamp／input_length／output_length／hash_ids`，但 `--timed-trace-chunk-hash-size` 預設 16、`--timed-trace-sec-multiplier` 預設 1（`datasets.py` L1713–1734）。直接餵 Mooncake 不會報錯：`_expand_prompt` 每個 hash 只展開 16 token，prompt 會短約 32 倍；ms 時間戳被當成秒，重播慢 1,000 倍。必須明確傳 512 與 0.001〔程式碼；判讀：未實際執行〕 | 〔文件〕〔計算〕〔判讀〕 |
| 與既有整理不一致 | 使用者文件 U5、U10（見抽查表） | — |
| 對本研究的意義〔判讀〕 | 有真實時間＋hash 的中度重用錨點；長度只到 126K，16K–128K 區間可用，再往上要合成 | 〔判讀〕 |

---

### C10｜Alibaba Qwen-Bailian traces（ATC'25 "KVCache Cache in the Wild"）與 ServeGen（NSDI'26）

**Bailian**

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 是什麼 | 阿里雲百煉上單一 Qwen 服務叢集的匿名請求 trace，四條、各 2 小時：traceA（To-C 聊天）、traceB（To-B API，2024-12 收集）、thinking（推理）、coder（寫程式） | README（commit `5f7439c5`）〔文件〕；論文 p1–p4 討論 to-C 與 to-B〔原文〕 |
| 欄位 | `chat_id`、`parent_chat_id`（-1 為根）、`timestamp`（秒）、`input_length`、`output_length`、`type`、`turn`、`hash_ids`（salted SipHash-2-4，16 token 一塊，再重新映射成連續整數） | README〔文件〕 |
| 規模與長度 | traceA 43,058（中位 1,046、平均 2,331、最大 89,286；類型 text 31,744、search 8,187、image 1,617、file 1,510）。traceB 172,800（中位 574、最大 66,446；api 150,936、text 21,864；全部單輪）。thinking 10,812（中位 3,680、最大 51,622）。coder 43,011（中位 4,540、最大 25,777） | 〔計算〕 |
| 重用 | 無限容量 token 級：traceA 58.1%、traceB 54.6%、thinking 46.2%、coder 66.5% | 〔計算〕 |
| FAQ 的兩個坑 | ①下一輪的輸入少了 `<think>`、`</think>`，因為應用程式組下一輪 context 時剝掉了。②最後一個 input block 可能含 padding，第一個輸出 token 會蓋掉部分 padding，所以 hash 會變 | `docs/qa-context-growth-pattern.md`〔文件〕 |
| 其他 | hash 是在**套用 chat template 之後**的 token 上算的，重播時不要再套 template；官方有 Rust 重播器 `blitz-serving/trace-replayer` | README〔文件〕 |
| License | Apache-2.0 | GitHub API〔文件〕 |
| 誰怎麼用 | **SAECache**：沿 `parent_chat_id` 重建 session，用子請求與父請求的時間差算輪間間隔（p16）；chat 的間隔 P50≈110 s、P80≈372 s、P99≈2,207 s，以 log-normal 擬合（p4），參數 µ=4.15、σ=0.97（p6）。〔複核修正〕µ=4.15、σ=0.97 在 p6 只說是「chat 為主的 trace」擬合值，沒說是 Bailian。對 Bailian 的離線 MLE 擬合是 **µ=4.82、σ=1.25**（19,957 個間隔、9,012 個 session，p17 Table 3／4）；線上更新規則收斂到 µ≈4.1、σ≈1.0（p17）。V10 用 traceA 重算得到 19,957／9,012／P50 110.6 s／P80 372.4 s／P99 2,207.3 s／µ=4.82／σ=1.25，與 Table 3 完全一致，可見 SAECache 用的是 traceA〔計算〕。README 另列 LMetric（OSDI'26）。Fancy-eviction 把它列為細粒度（16 token）但 agentic 很少（p5 Table 3） | 〔原文〕〔文件〕 |
| 陷阱 | ①長度短：≥32K 只有 0–0.18%。②每條只有 2 小時，各有自己的時鐘，不能串接。③最後一塊 hash 會變，使「完整命中」偏低。④thinking 的思考 token 被剝除，前後輪的 hash 不會連續〔判讀〕〔複核修正〕FAQ 說被剝掉的只有 chat template 注入的 `<think>`、`</think>` 兩個特殊 token（例：輸出 `3 4 5 6 7`，下一輪只剩 `4 5 6`），思考內容本身仍在；hash 會在這兩個 token 的位置錯開〔文件〕 | 〔計算〕〔文件〕〔判讀〕 |
| 對本研究的意義〔判讀〕 | 16 token 與 vLLM 預設 block 相同，最適合當「短 context、真實節奏」的負對照 | 〔判讀〕 |

**ServeGen**

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 是什麼 | 依阿里雲百煉 12 個生產模型、數十億請求的分析，建立的**負載生成器**。主資料是**每個 client 的分布**（rate、CV、長度），分 language（m-large／m-mid／m-small）、reason（deepseek-r1）、multimodal（mm-image）。另附 `conversations_hashed.json` 與 offline（ACDC，SOSP'26）trace | README（commit `d70c8b2d`）〔文件〕；〔複核〕ACDC／SOSP'26 寫在 `data/offline/README.md` L3，不在主 README |
| 原文怎麼說 | 基於保密，釋出的是參數化、清洗過的資料，不是原始資料 | arXiv 2505.09999v3 p10 腳註〔原文〕 |
| 對話檔 | 1,616 段對話、5,720 輪，UTC 2025-02-16 16:01 → 02-17 15:57。每輪有 `turn, timestamp, input_token_count, output_token_count, input_tokens（hash 列表）, output_tokens`。input_token_count 中位 7,745（`higher` 法 7,750）、p90 55,276、最大 262,644，18.1% ≥32K | 〔計算〕 |
| License | Apache-2.0 | GitHub API〔文件〕 |
| 陷阱 | ①主資料是分布，不是逐請求 trace。②對話檔 `input_tokens` 的 hash 語意沒有文件說明（`workloads_eval` §3.1 已指出），重用無法驗證。〔複核補充〕README 只寫 input／output 內容「換成 block hash」，沒給 block 大小。V10 檢查：`len(output_tokens)==output_token_count` 在 5,720 輪 100% 成立（等於逐 token）；input 的 count÷len 中位約 511，但 `ceil(count/512)==len` 只有 12.2%，試過 8–512 都不成立〔計算〕 | 〔文件〕；`workloads_eval` |
| 對本研究的意義〔判讀〕 | 對話檔是少數「長 context＋真實時間＋多輪」的公開資料，但只有 1,616 段、一天；可當長度與輪次分布的參考 | 〔判讀〕 |

---

### C11｜TraceLab（UW SyFI，2026；`uw-syfi/TraceLab`）

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 是什麼 | 開發者本機 Claude Code 與 Codex session 檔，經正規化與去識別後釋出。每列是一次 LLM 呼叫（round），附 token 帳目、`timing_events`（user_message、tool_result 等）與工具 metadata；`tools[].input` 已刪除，只留字元數 | README（commit `11b8b14c`）〔文件〕 |
| 版本 | release：`v2026-06-08-syfi-trace`（06-13）、`v0.0.1`（06-22，jsonl.gz 53.6 MB，sha256 `9d265eae…` 與 README 相符）、**`v0.0.2`（07-24，100.9 MB）**。README 的資料表仍寫 v0.0.1；README 也提供 `latest` 下載連結，那會拿到 v0.0.2 | GitHub releases API、README〔文件〕。〔複核補充〕GitHub API 顯示 `v2026-06-08-syfi-trace` 的 jsonl.gz digest 也是 `9d265eae…`，與 v0.0.1 是同一個檔（duckdb 不同）；v0.0.2 的 digest `11ce51ec…` 與本地檔相符 |
| v0.0.1 | 357,161 列、43 user、4,265 session；claude 140,338、codex 216,823；2025-09-23 → 2026-06-04。input 中位 124,018、p90 256,767、p99 822,895（`higher`）、平均 153,707、最大 999,888。≥32K 92.5%、≥100K 60.9%、>131,072 47.0%、>262,144 9.7% | 〔計算〕＋README 表 |
| v0.0.2 | 665,453 列、52 user、8,058 session；到 2026-07-24。input 中位 132,092、p90 338,661、≥100K 63.4%、>131,072 50.4%、>262,144 14.1%；provider 命中 95.6% | 〔計算〕 |
| token 帳目 | `input_tokens_total = prefix_tokens + newly_append_tokens`，兩版皆 100% 成立；Claude 列的 `prefix_tokens` 等於 `claude_cache_read_input_tokens`，兩版皆 100% | README〔文件〕；〔計算〕 |
| 時間、多輪、身分 | ms 級逐事件時間；session、round、匿名 user | 〔文件〕〔計算〕 |
| License | 資料 CC BY 4.0（`LICENSE-DATASET.md`）；程式 Apache-2.0 | 〔文件〕 |
| 誰怎麼用 | Fancy-eviction 把它列進公開 trace 比較表：357K 列、54.9B token、253 天且分散、無 block id；並加註它是 43 位開發者各自的 session，不是連續的生產服務流 | p5 Table 3〔原文〕 |
| 陷阱 | ①不是服務端的流量，是個別開發者的 session，並發程度要自己設定。②token 是 provider tokenizer 算的，換模型要換算，並在 loader 裡用 `prefix+new==input` 當斷言。③`prefix_tokens` 是 provider 的實際命中（下界），不是理想上限。④沒有內容與 hash，跨 session 的共用看不到。⑤版本在變，引用要寫 release tag 與 sha256 | 〔文件〕〔計算〕〔判讀〕 |
| 對本研究的意義〔判讀〕 | 目前唯一公開、長度落在 16K–512K（甚至 >512K）且有真實時間的 trace，適合當「時間與 session 骨架」 | 〔判讀〕 |

---

### C12｜Bidaw Interactive-conversation-workload（FAST'26；`ShipengHu-777/Interactive-conversation-workload`）

> 任務清單沒有列它，但它是抽查重點，也是唯一有「每使用者 >20 輪、人類節奏」的公開 trace，所以補一張卡。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 是什麼 | 業界夥伴的互動對話負載，超過一百萬輪 | Bidaw p4〔原文〕 |
| 檔案與欄位 | `total_workload/total_traces_part1–7.txt`，空白分隔，表頭 `user_id time_stamp(seconds) query_length response_length round_index` | repo（commit `6f3281b9`）〔文件〕 |
| 規模 | 1,268,346 輪、56,573 位使用者、21.35 h；每人 22.42 輪（中位 18、p90 45）；輪間 p50 43 s、p99 142 s | 〔計算〕；原文 p4 為 22／18／45 |
| 長度 | query 平均 35.3、response 平均 44.6（原文 36／45）。長度單位 README 與原文都沒寫明〔未查證〕。若假設 prompt＝完整歷史，中位 1,066、p99 8,168、最大 47,624 | 〔計算〕〔原文〕 |
| License | 沒有 license 檔（GitHub API `null`） | 〔文件〕 |
| 原文怎麼用 | 取 8:00–20:00 之間不同小時開始的 10 分鐘 trace，每組再改變使用者到達率（p8）；README 說要改到達率就對使用者做抽樣 | 〔原文〕〔文件〕 |
| 陷阱 | ①沒有內容；重用量取決於「prompt＝完整歷史」的假設。②每輪都極短。③時間是整數秒。④沒有 license，再散布的權利不明 | 〔判讀〕 |
| 對本研究的意義〔判讀〕 | 適合用來比較 Bidaw 的 previous-answer 驅逐，以及取得人類節奏的輪間時間 | 〔判讀〕 |

---

## 長文件類當負載用（C13）

| 資料集 | 誰用 | 怎麼取樣 | 取多少 | 長度 | 到達 | 出處 |
|:--|:--|:--|:--|:--|:--|:--|
| **LongBench（summarization）** | DistServe（OSDI'24） | 用 LongBench 的 summarization 任務，從資料集抽樣請求 | 未說明 | **截到 2,048**（OPT 的位置編碼上限） | Poisson | p10、腳註 4〔原文〕 |
| **LongBench（TriviaQA）** | LMCache（arXiv'25） | LongBench 的 TriviaQA | 未說明 | 未說明 | Poisson，指定 QPS | p11–p12〔原文〕 |
| **HumanEval** | DistServe | 164 題 code completion | 164 題 | 短 | Poisson | p10〔原文〕 |
| **arXiv summarization**（Cohan et al. 2018；HF `ccdv/arxiv-summarization` train 203,037／val 6,436／test 6,440） | Sarathi-Serve（OSDI'24） | 只用長度特徵生成 trace | 未說明 | prompt 中位 7,059、P90 12,985；輸出中位 208；**濾掉總長 >16,384** | Poisson | p9、p11 Table 2〔原文〕；datasets-server〔文件〕 |
| 同上 | Vidur（MLSys'24） | 全量 | 203k 筆；另有總長 ≤4K 的 Arxiv-4K（28k 筆） | prefill 平均 9,882、中位 7,827、p90 18,549 | 未讀 | p7 Table 1〔原文〕 |
| 同上 | Mooncake **arXiv 版** | — | — | avg in 8,088、out 229、cache 約 0% | Poisson | 2407.00079 p15〔原文〕 |
| **L-Eval** | Mooncake FAST'25 | 合成負載的一份（模擬帶長系統提示的 tool／agent 請求）；同一長 prompt 的每個問題各成一個請求；1:1:1 混合 | 合成負載共 3,993 筆 | 合成負載平均 15,325 | Poisson | p10〔原文〕；README〔文件〕 |
| 同上 | Mooncake arXiv 版 | — | — | avg in 19,019、out 72、cache >80% | Poisson | p15〔原文〕 |
| 同上 | HCache（EuroSys'25） | 三個代表性子任務，另抽 200 個請求做混合 | 200（mixed） | context 平均最多 16K；指令與輸出通常 <100；模型 context 擴到 16K | 主實驗未讀到；子實驗用 Zipf（uniform、α=1.2–2）〔複核補充〕主實驗 batch size＝1，因為 GPU 只放得下 1–3 個長請求（p10 §6.1.2），所以沒有到達過程；Zipf 合成的是各 context 的到達熱度（p13） | p4 Table 1、p9、p10、p13〔原文〕 |
| 同上 | Tutti（arXiv'26） | 從各子集**輪流**抽取請求 | 未說明 | 3K–200K | Poisson（資料集沒有到達時間） | p9〔原文〕 |
| **LooGLE** | Strata（OSDI'26） | 只用 Wikipedia 部分，含長短 query；query 隨機抽 | 105 份 context、2,410 個 query | avg in 21,613、out 15.6 | Poisson | p9 Table 1、p10〔原文〕 |
| 同上 | Mooncake FAST'25 | 合成負載的一份（長文 QA 與摘要） | 同上 | 原文說輸入最長達 100k | Poisson | p10〔原文〕 |
| 同上 | Tutti | 輪流從 4 個子任務抽 | 未說明 | 許多樣本 >100k | Poisson | p9〔原文〕 |
| **NarrativeQA** | Strata | **濾掉 >128K 的文件**，再抽 50 份 | 50 份 context、1,461 個 query | avg in 54,797、out 13.0 | Poisson | p9 Table 1〔原文〕 |
| **ReviewMT**（多 agent 審稿對話） | Strata | 保留輪間依賴 | 100 份 context、1,092 個 query | avg in 17,708、out 208.3 | Poisson | p9 Table 1、p10〔原文〕 |
| **Bilingual-Web-Book** | Vidur | 全量；另有 ≤4K 版 | 195k；BWB-4K 33k | prefill 平均 2,418 | 未讀 | p7〔原文〕 |

**觀察〔判讀〕**
* 這些負載的「重用」完全由研究者決定：多少個 query 共用一份文件、抽樣是否輪流、Zipf α 多少。所以它們**不能當作命中率或 headroom 的證據**，只能用來量「給定重用下的系統行為」與品質。
* 最長的真實平均輸入是 Strata 的 NarrativeQA（54,797），單筆最長是 Tutti 的 L-Eval（約 200K）。**沒有一篇用 >256K 的文件當 serving 負載。**

---

## 到達過程從哪來（C14）

| 方式 | 用例（原文頁碼） | 已知問題 |
|:--|:--|:--|
| **Poisson**（指數分布的到達間隔，掃 rate） | vLLM p10；DistServe p10；Sarathi-Serve p11；**Splitwise p9（連自家的 Azure trace 也只取長度，到達用 Poisson）**；Llumnix p9；Mooncake FAST'25 synthetic p10；CachedAttention p10（以 session 為單位，λ=1.0）；Pensieve p9；Strata p10；Tutti p9；HCache p10（session 到達）；MTDS p10；LMCache p12；MLPerf Server（`inference_rules.adoc` L141） | ①**Bidaw p12**：ShareGPT 換成 Poisson 時間後，previous-answer 驅逐不再降低 miss rate，增益明顯縮水。原因是 Poisson 抹掉了「回答長度→使用者停多久」的關聯（原文 p8 的觀察）。〔複核修正〕p12 原文只說「因為時間是 Poisson 模擬的，所以」驅逐不再降低 miss rate；p8 的觀察是「每次 KV 存取的加權重用距離下界與上一輪回答長度正相關」。把兩者連成「抹掉回答長度與停留時間的關聯」是合理推論，但屬〔判讀〕。②沒有突發：BurstGPT 實測 Gamma 參數隨服務變化（p3–p4）；ServeGen README 開宗明義說真實到達比 Poisson 更 bursty。③沒有日週期：DynamoLLM 實測尖峰／谷底達 34.6 倍（p4）。④若以「請求」為單位套 Poisson，會打散同一 session 的輪次 |
| **Gamma**（可調 CV） | Llumnix p9（改變 CV 調突發）；vLLM `--burstiness`（`serve.py` L405–491）；BurstGPT p4–p6（擬合 Gamma；demo α=0.5、β=2） | 仍是平穩的 renewal 過程：沒有日週期，沒有 session；CV 的選擇通常沒有依據〔判讀〕 |
| **固定間隔**／一次全送 | SAECache p8（固定注入間隔 0.02–0.08 s，刻意讓 GPU 飽和）；vLLM `burstiness=inf` 為固定間隔（`serve.py` L470） | 沒有排隊的隨機性；只能看飽和吞吐，不能看尾端延遲〔判讀〕 |
| **從別的 trace 移植時間** | EvicPress p10（Azure 的真實時間戳＋自己生成的請求內容）；Splitwise、DynamoLLM 以 RPS 縮放自家 trace（p4、p9）〔複核修正〕只有 Splitwise p4 的單機特徵分析把 trace 縮到 2 req/s；Splitwise p9 的叢集評估改用可調的 Poisson rate（不是縮放 trace）；DynamoLLM p9 只說用 1 小時公開 trace 與自家 1 天、1 週 trace 設定負載，沒提縮放；BurstGPT README 建議依系統規模縮放 RPS；Bidaw README 建議抽樣使用者來改到達率 | 時間與內容的關聯被切斷，例如「長回答之後停比較久」〔判讀〕；縮放倍數會改變重用距離，必須記錄 |
| **思考時間／閉迴路** | Pensieve p9、p12（下一輪等上一輪回應完成，再加指數分布 think time，平均 60 s，敏感度掃到 600 s）；Strata p10（60 s thinking time，說依照 Pensieve）；HCache p10（輪間 30 s）；Marconi p8（調整 session 間與請求間的時間）；SAECache p4、p6（從 Bailian 與 CC-Bench 擬合 log-normal 輪間間隔：chat P50≈110 s、agent P50≈8.5 s） | think time 的平均直接決定 KV 在快層待多久（Pensieve p12 顯示 think time 越長吞吐越低），等於研究者在決定結果；固定值（30 s、60 s）抹掉變異〔判讀〕 |
| **真實時間戳重播** | Mooncake FAST'25 p10（依時間戳送出，輸出到預定長度就結束）；Bidaw p8（不同時段的 10 分鐘 trace＋使用者抽樣）；Bailian 官方重播器（README）。〔複核補充〕vLLM `vllm bench serve --dataset-name timed_trace`（commit `31e2443c`）也能依時間戳重播 hash trace，預設 `self_timed=True`、`ignore_eos=True`（`serve.py` L2150–2156），但 block 大小與時間單位要自己設（見 C9 陷阱⑥） | 只有少數 trace 有；長度短；1–2 小時的窗看不到長間隔重用〔判讀〕 |

**真實輪間時間的參考值**（同一使用者或 session 內）

| 來源 | p50 | 尾端 | 出處 |
|:--|:--|:--|:--|
| Bidaw trace | 43 s | p99 142 s | 〔計算〕 |
| BurstGPT_3 conversation | 131 s | p90 2,340 s | 〔計算〕 |
| Bailian（SAECache 擬合） | ≈110 s | P99 ≈2,207 s | SAECache p4〔原文〕；〔複核〕V10 用 traceA 重算 110.6 s／2,207.3 s〔計算〕 |
| CC-Bench agent（SAECache） | ≈8.5 s | P80 ≈25 s | SAECache p4〔原文〕 |
| TraceLab | 8.8 s | p99 1,782 s | `workloads_eval` §3.1〔二手，本卡未重算〕 |

〔判讀〕人類聊天的中位數是 40–130 s，agent 是 8–9 s，相差一個數量級。Pensieve、Strata 用的 60 s 落在人類聊天區間；**對 agent 型的長 context 負載，60 s 太長**。

---

## 對 PoC 設計的建議〔判讀〕

以下都是判讀，依據是上面各卡的事實。

1. **主負載用「真實骨架＋長內容」拼接，並寫明是拼接的。**
   * **時間與 session 結構**：用 TraceLab（寫明 v0.0.1 或 v0.0.2 與 sha256）。它的長度本身就落在 16K–512K。
   * **內容**：量時間（TTFT、搬移量）時，可以依 TraceLab 每輪的 `prefix_tokens`／`newly_append_tokens` 合成 token，重用結構就照原始 trace。量品質時，換成真實長文件（LooGLE、NarrativeQA、LongBench v2、L-Eval），把每個 session 對應到一份文件、每一輪對應到一個 query。
   * **中度重用的對照**：Mooncake conversation／toolagent（真實 hash），只涵蓋到 126K。
2. **短 context 負對照**：Bailian traceA（16-token hash、真實節奏）。若要正面比較 Bidaw，用 Bidaw trace。
3. **不要當主負載**：ShareGPT、Alpaca、LMSYS、Arena、OpenOrca（短，或沒有時間，或授權受限）；BurstGPT 與 Azure（只有長度）。它們最多提供「時間來源」：WildChat、BurstGPT_3、Azure 2024 的突發與日週期。
4. **到達**：主實驗用真實時間（必要時記錄縮放倍數）。Poisson 與 Gamma 只當敏感度分析。agent 負載的 think time 不要沿用 60 s。
5. **>256K 的請求要事先定規則。** 真實資料只有 TraceLab 會超過 262,144（v0.0.1 為 9.7%，v0.0.2 為 14.1%）。〔複核修正〕ServeGen 對話檔也有 1 輪超過（最大 262,644），另有 125 輪（2.19%）>131,072〔計算〕；比例上仍以 TraceLab 為主。Qwen noDCA 的上限是 262,144、Llama-3.1 是 131,072。截斷、丟棄或縮放三選一，並記錄比例。512K 一端在公開資料中**只能靠合成**，論文要明說。
6. **每個 loader 一條單位斷言**（CLAUDE.md 規則 6）：
   * Mooncake：`ceil(input/512)==len(hash_ids)`
   * Bailian：`ceil(input/16)==len(hash_ids)`
   * TraceLab：`prefix+new==input`
   * Bidaw：`round_index` 在每位使用者內連續
   * ShareGPT：第一則必須是 human
7. **manifest 必填**：資料檔的 sha256 或 release tag、tokenizer、百分位方法、過濾規則與丟棄比例、時間縮放倍數、暖機窗。

---

## 未查證清單

* ShareGPT 內容的實際權利；vLLM、DistServe、Bidaw 各自用了哪個 ShareGPT 檔（原文只引用 sharegpt.com）。
* vLLM 論文 Fig. 11 的 ShareGPT 平均 161.31／337.99 怎麼算出來的：三種組法都沒重現（見 C1）。
* 切段腳本實際用的 `max_length` 是 2048（README）還是 2304（腳本預設）。
* SGLang `bench_serving` 的 ShareGPT 過濾規則（本卡沒讀程式碼；E02 的範圍）。
* WildChat-1M／4.8M 的長度分布（未計算）。
* Azure 2024 一週 trace 的統計（檔案 >500 MB，未下載）；md 寫的「5 月 10–19 日」與「一週」的關係。〔複核：後半已解，每個檔各 7 天，見 A22；統計仍未算〕
* BurstGPT 論文（10.31M、213 天）與 release（約 10.63M 行、231 天）不一致的原因。
* OpenOrca 的總列數（datasets-server 為 partial）；OpenOrca 的 system prompt 種類數。
* ~~HCache L-Eval 主實驗（p10 Fig. 10）的到達過程。~~〔複核：已解，batch size＝1，見 C13〕
* Tutti、DistServe、LMCache 各自取了多少筆請求。
* Fancy-eviction 的 FreeInference／Chutes trace 是否公開；~~37.6%／70.2% 是單一 trace 還是兩者合併。~~〔複核：後半已解，只來自 FreeInference，見 U2〕
* MLPerf Agentic 的 99.0–99.5% 重用與長度（本卡未重算，沿用 `workloads_eval` §3.4）。
* Bidaw trace 的長度單位（token 或字元）。
* TraceLab v0.0.2 相對 v0.0.1 的變更紀錄（只比對了統計）；TraceLab 輪間時間（沿用 `workloads_eval`）。
* ServeGen 論文正文（只讀 p1–p2 與 p10 腳註）。
* Chatbot Arena 的 `conversation_a`／`conversation_b` 第一句是否逐字相同（推論，未逐筆驗證）。
* ~~Mooncake 不同 trace 檔的 hash id 是否同一命名空間。~~〔複核：已解，逐檔重新編號，見 C9 陷阱③〕
* Mooncake trace 本身的 license（repo 為 Apache-2.0，trace 沒有另外標示；本卡未另查）。〔複核：已查 GitHub API 與 FAST25-release README，結論同上；trace 本身仍無獨立 license 聲明〕

---

## 狀態

* 抽取完成：2026-10-07，E10。
* 複核：2026-10-07，V10 完成，見下方「複核紀錄」。

---

## 複核紀錄

* **複核者**：V10（獨立子 agent，未參與抽取，沒有讀抽取者的推理、筆記或 `*_stats.py`）。
* **日期**：2026-10-07。
* **做法**：
  * **資料**：先驗證抽取者下載的檔是原檔，再用自己的程式重算。Mooncake 四檔與 Bidaw 七檔的 git blob 與 GitHub 樹（`1d0e4c75`、`6f3281b9`）相同。Bailian 四檔的 sha256 與 LFS 指標（`5f7439c5`）相同。TraceLab 兩版與 BurstGPT_1／_3 的 sha256 與 GitHub release digest 相同。Azure 2023 兩個 CSV 與 ServeGen 對話檔的 git blob 與 repo 相同。ShareGPT parquet 與 HF `refs/convert/parquet` 的 sha256 相同，原 JSON 的 LFS oid（`35f0e213…`）在原 repo 與鏡像一致。Alpaca parquet 與 HF 相同。Qwen 的 tokenizer 檔、OPT 的 vocab／merges 與 HF 的 git blob 相同；OPT 的 `tokenizer.json` 是抽取者自建的，我沒有用它，改由 vocab＋merges 自建。
  * **重算腳本**：`<SCRATCH>/V10/` 下的 `v_mc.py`、`v_tl.py`、`v_bidaw.py`、`v_az_bg_sg.py`、`v_sharegpt.py`、`v_sharegpt2.py`、`v_alpaca.py`、`v_bailian_gap.py`，輸出在同名 `.out`。
  * **論文**：arXiv 的 20 篇 PDF 自己重新下載（指定版本），USENIX 的 DistServe、Sarathi-Serve、Mooncake FAST'25、CachedAttention 自己從 usenix.org 下載。Strata、Bidaw、MTDS 使用 E04 留下的 PDF，第一頁核對過是 OSDI'26、FAST'26、C&IS 2026 正式版。全部自己跑 `pdftotext -layout`，頁碼以 PDF 頁序為準。
  * **程式碼與文件**：vLLM `datasets.py`、`serve.py`、`docs/benchmarking/cli.md` 用 raw.githubusercontent 抓 commit `31e2443c`（2026-10-06）。MLPerf README、`processorca.py` 抓 `3fbc3299`，`inference_rules.adoc` 抓 `d3eba2f2`。HF 資料卡與 datasets-server 為 2026-10-07 即時查詢。
* **〔計算〕數字**：約 190 個重算數字全部對上，0 個不符。分布：Mooncake 約 32 個、Bailian 約 34、TraceLab 兩版 30、Bidaw 17、Azure 10、BurstGPT_3 13、ServeGen 9、ShareGPT 約 35、Alpaca 9。另外重算 SAECache 引用的 Bailian 輪間統計 8 個，與其 Table 3 完全相同。
* **抽取者重點發現的確認**：
  * (a) vLLM `ShareGPTDataset.sample` 只取 `conversations[0]`、`[1]`，不看 `from` ✅〔程式碼，L1373–1376〕。V3 檔 36.7% 以 gpt 開頭 ✅（34,538／94,145）。35.5% ✅，但它是「第一則不是 human」；嚴格的 gpt→human 是 35.3%，措辭已修。p99 2,873 ✅，用的是 `Qwen/Qwen2.5-7B-Instruct` tokenizer，每則分別 tokenize 後相加，不套 template。
  * (b) TraceLab v0.0.2 ✅：2026-07-24 發布，665,453 列、52 人、8,058 session、中位 132,092。
  * (c) Splitwise 叢集評估只取 Azure trace 的長度分布、到達用可調的 Poisson rate ✅（arXiv v2 p9）。Bidaw p12 ✅。
* **對使用者文件的四項指控**（逐條回原文判定）：
  * U10 表 17「Mooncake 模擬 16K–128K、快取 50%」：✅ 指控成立。FAST'25 全文沒有這組負載，它來自 arXiv Table 2（v4 p15／v1 p13）。
  * U5「Mooncake 的 128K 是模擬資料」：✅ 指控成立。FAST'25 p10 說真實 conversation 負載最長達 128k。trace 最長 126,195，有 63 筆 ≥100K。
  * U16 表 S4「重用率為無限容量估計」：✅ 指控成立（對 TraceLab 與 Codex 兩列）。兩者都是 provider 實際的快取命中。
  * U11 HCache「以 Zipf 合成到達」：✅ 指控成立，程度為 ⚠️ 過度簡化。主實驗是 ShareGPT4 Poisson session 加輪間 30 s，以及 L-Eval batch size 1；Zipf 只出現在 p13 的子實驗。
  * 另檢查使用者文件 [13] Fancy-eviction 的書目：寫的是 arXiv:2609.28870、Sep. 2026，與 PDF 一致，**沒有錯**。我一度把相鄰書目的 SODA 2022 誤讀成它的，已確認是 [48] 那一筆，未列為指控。
* **每張卡檢查的格數與判定**（✅ 與原文一致／❌ 錯誤已改／⚠️ 不精確、已補或已改）：

| 區塊 | 檢查格數 | ✅ | ❌ | ⚠️ |
|:--|--:|--:|--:|--:|
| 來源清單 | 33 | 32 | 0 | 1 |
| 重點摘要 | 8 | 5 | 0 | 3 |
| 抽查表 A（workloads_eval） | 31 | 29 | 1 | 1 |
| 抽查表 B（使用者文件） | 16 | 16 | 0 | 0 |
| C1 ShareGPT | 39 | 36 | 0 | 3 |
| C2 Alpaca | 11 | 10 | 1 | 0 |
| C3 LMSYS-Chat-1M | 9 | 9 | 0 | 0 |
| C4 Chatbot Arena | 8 | 8 | 0 | 0 |
| C5 WildChat | 10 | 10 | 0 | 0 |
| C6 OpenOrca | 11 | 11 | 0 | 0 |
| C7 BurstGPT | 11 | 11 | 0 | 0 |
| C8 Azure | 8 | 7 | 0 | 1 |
| C9 Mooncake | 13 | 12 | 0 | 1 |
| C10 Bailian＋ServeGen | 16 | 14 | 2 | 0 |
| C11 TraceLab | 11 | 11 | 0 | 0 |
| C12 Bidaw | 9 | 9 | 0 | 0 |
| C13 長文件負載 | 19 | 19 | 0 | 0 |
| C14 到達過程 | 11 | 9 | 1 | 1 |
| PoC 建議（只查其中的事實） | 7 | 6 | 1 | 0 |
| 未查證清單 | 18 | 18 | 0 | 0 |
| **合計** | **299** | **282** | **6** | **11** |

（〔判讀〕格只檢查有沒有被寫成事實。未查證清單的「✅」表示確實未查證，其中 4 項已由複核解決並標註。）

* **改了什麼**（原內容 → 新內容＋出處）：
  1. ❌ **A31**：原寫「頁碼不同，可能是版本或頁序差異」→ 改成：v1 p17 確有 µ≈4.1、σ≈1.0（線上更新收斂值）；Bailian 離線 MLE 是 µ=4.82、σ=1.25（p17 Table 3／4），我重算一致；p6 的 4.15／0.97 是另一組。判定仍為 ⚠️，但理由改成語意問題。
  2. ❌ **C10 Bailian「誰怎麼用」**：原寫「參數 µ=4.15、σ=0.97（p6）」，等於說是 Bailian 的擬合 → 補上 Bailian 的實際擬合 4.82／1.25（SAECache p17）與 V10 重算結果。
  3. ❌ **C10 Bailian 陷阱④**：原寫「thinking 的思考 token 被剝除」→ 改成：只剝掉 `<think>`、`</think>` 兩個特殊 token（`docs/qa-context-growth-pattern.md` Q1）。
  4. ❌ **C2 Alpaca「誰怎麼用」**：原寫「parallel sampling 與 beam search 實驗只用 Alpaca」→ 補上 vLLM p11 正文另報 ShareGPT 的記憶體節省（16.2–30.5%、44.3–66.3%）。
  5. ❌ **C14「從別的 trace 移植時間」**：原寫「Splitwise、DynamoLLM 以 RPS 縮放自家 trace（p4、p9）」→ 只有 Splitwise p4 縮到 2 req/s；Splitwise p9 用 Poisson；DynamoLLM p9 沒提縮放。
  6. ❌ **PoC 第 5 點**：原寫「真實資料只有 TraceLab 會超過 262,144」→ 補上 ServeGen 對話檔有 1 輪超過（262,644），125 輪 >131,072〔計算〕。
  7. ⚠️ **重點摘要 2**：35.5% 的措辭 → 補上「第一則非 human 35.5%，其中 gpt→human 35.3%」〔計算〕。
  8. ⚠️ **重點摘要 4**：「7 個檔」→ 補上：連 synthetic 共 8 個檔，全部 100%。
  9. ⚠️ **重點摘要 6**：「HCache 另做 Zipf」→ 補上 L-Eval 主實驗是 batch size 1（HCache p10）。
  10. ⚠️ **A22**：⚠️ → ✅。用 HTTP Range 讀檔頭檔尾，確認兩個 2024 檔各 7 天（未下載全檔）。
  11. ⚠️ **C1 結構**：第一說話者清單補上 bing 2 筆；補上切段腳本 L61 的 TODO 作為 36.7% 的成因。
  12. ⚠️ **C1 Sarathi-Serve**：「含歷史」改標〔判讀〕（p11 沒明說）。
  13. ⚠️ **C1 Pensieve**：「只算新增部分」改標〔判讀〕（Table 2 只寫 mean request input length）。
  14. ⚠️ **C8 2023 版**：「中位數與檔案相符」→ 補上 Splitwise p3 的 coding prompt 中位是 1500，檔案是 1,469。
  15. ⚠️ **C9 Strata 38%**：補上原文的兩個條件：不含系統提示、一秒內到達（Strata p5）。
  16. ⚠️ **C14 Bidaw 成因**：「Poisson 抹掉回答長度與停留時間的關聯」改標〔判讀〕（p12 原文只說因為時間是模擬的）。
  17. ⚠️ **來源清單**：Mooncake arXiv 頁碼註明是 v4（v1 的 Table 2 在 p13）。
* **〔複核補充〕**（抽取者漏掉、對評測設定重要）：
  * vLLM `get_request` 會把 Poisson 間隔整體縮放成總時長 N/rate（`serve.py` L483–497）。
  * vLLM `BurstGPTDataset` 用欄位位置取長度，與 BurstGPT_3 的新欄位不相容（L3229–3230）。
  * vLLM `timed_trace` 預設 16-token、秒為單位，直接餵 Mooncake 會靜默錯 32 倍與 1,000 倍（L1713–1734、`serve.py` L2150–2156）。
  * Mooncake 附錄 A.1：兩條真實 trace 來自不同叢集，取樣時優先收同一 session 的請求；三個 FAST'25 檔的 hash id 各自從 0 連續編號。
  * Strata 的 in-flight 上限是 128（p10）。
  * MTDS 用的 ShareGPT 是 `shibing624/sharegpt_gpt4`（p15）。
  * ServeGen：`output_tokens` 逐 token；`input_tokens` 的 block 大小無法判定。
  * TraceLab：`v2026-06-08` 快照的 jsonl 與 v0.0.1 同檔。
  * Bailian：前綴定義與任意位置定義的重用率差距（A10）。
  * Fancy-eviction 的 37.6%／70.2% 只來自 FreeInference（U2）。
  * Codex 94.2% 的原文出處（U16）。
* **因時間沒有檢查的格**（如實列出）：
  * 沒有逐行核對 `PAPERS_BY_LEVEL.md`。
  * 沒有重算 TraceLab 輪間時間（8.8 s／1,782 s，卡片本來就標〔二手〕）、MLPerf Agentic 的 99.0–99.5%、WildChat 長度分布、OpenOrca 全量列數與 system prompt 種類、Arena `conversation_a/b` 首句是否相同。
  * 沒有逐一核對 Mooncake FAST'25 p4／p12、Bidaw p3、Llumnix p3／p4 這些「用到的頁」。
  * vLLM `serve.py` 的「L405–491」只核到大致範圍：`get_request` 實際從 L400 開始。
  * 標〔判讀：未實際執行〕的兩項 vLLM 行為（BurstGPT_3 報錯、timed_trace 縮短 prompt）是讀程式碼推的，沒有實際跑 vLLM。
  * 沒有解開 vLLM 論文 Fig. 11 的 161.31／337.99 怎麼算出來；V10 重算抽取者的三種組法，得到相同的三組數字，仍然對不上。
