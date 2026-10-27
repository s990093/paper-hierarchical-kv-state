# V：36 條事實宣稱的原始來源核對

- **日期**：2026-10-26
- **執行者**：零前文 verifier（Claude subagent，Opus 5.5）。**這是 self-review 的一種，不是 cross-model review**：同一個模型家族，只是不帶前文脈絡。
- **怎麼查的**：
  - 論文：arXiv abs 頁確認存在與版本，下載 PDF，用 `pdftotext` 轉文字後逐段讀。Strata 另外下載了 OSDI '26 正式版（USENIX `osdi26-xie-zhiqiang.pdf`）。兩個版本內容不同，見 §0。
  - GitHub：`gh issue view`／`gh pr view`／`gh api .../timeline`，全部唯讀。程式碼直接讀 raw.githubusercontent.com 上 commit `0f6b2619…` 的檔案。
  - 古典文獻：能拿到的都讀了全文（作者版或 VLDB 開放版）。Gupta et al. 2004 拿不到全文，只用 Crossref／OpenAlex 核書目、用 RePEc 與 Greenwich 典藏的摘要核內容，標〔摘要〕。
  - 本機 repo：直接讀檔。#34、#36 另外用 repo 內的結果檔（CSV／JSON）和常數自己重算過。
- **讀的版本**：

| 來源 | 版本 | 頁數 |
|---|---|---|
| Strata | arXiv v1（2025-08-26） | 13 |
| Strata | OSDI '26 USENIX PDF | 17（第 1 頁是封面，所以印刷頁＝PDF 頁 −1） |
| PEEK | arXiv v2 | 26 |
| TempoKV | arXiv v1 | 8 |
| DLPM | arXiv v1 | 17 |
| HiSparse | arXiv v1 | 20 |
| RetroInfer | arXiv v3（標示 VLDB 2026） | 16 |
| SwiftCache | arXiv v1 | 15 |
| LMCache | arXiv v2 | 19 |
| KVFlow | arXiv v1 | 12 |
| CacheFlow | arXiv v2 | 17 |
| ConServe | PMLR v306 PDF | 15 |
| Eager et al. | usask 預印本（標示「to appear in TKDE」），**不是** TKDE 排版版 | 22 |
| Iyer & Druschel | MPI-SWS 預印本（標示「To appear in SOSP 2001」） | 10 |
| QPipe | CMU PDL 作者版 | 12 |

- 下載的檔案在 `/private/tmp/claude-501/-Users-hungwei-Desktop-Proj-paper-hierarchical-kv-state--claude-worktrees-hidden-latency-kv-cache-330751/6c0adbfb-479b-42f7-986a-069d8cf5ea05/scratchpad/V/pdfs/`。
- **刻意沒做的事**：沒有改其他 repo 檔、沒有 commit、沒有安裝套件、沒有跑 GPU。

---

## §0 統計

| 判定 | 條數 | 編號 |
|---|---|---|
| OK | 30 | 2–4、6、7、9–16、18、20–33、35、36 |
| FIX | 6 | 1、5、8、17、19、34 |
| UNVERIFIABLE | 0 | — |

**最重要的 FIX（依影響排序）**

1. **#17 PEEK 不只拿 cluster 大小當 tie-break。**
   - 排序鍵是 (section, −h, −score, −|cluster|, arrival)。
   - 在 cluster 大小之前的 `score` 本身就是「等待中共用者的深度加權需求」：路徑上 pending_count×邊長的加總。
   - 另有選用的 group-major，會讓同一 cluster 連續送出；淘汰（PE）也用 pending_count×depth 來保護 block。
   - 所以「PEEK 不依等待人數排序」不能成立。它確實依等待人數排序，只是沒有 host／SSD 這一層。
2. **#34 I1 的「每一格都負 −14%～−60%」和自己的結果檔不完全一致。**
   - 結果檔是 `prediction_llama.json`。N＝4、8 共 32 格。
   - homog_cpu 的 8 格裡，joint 等於各自最佳（0%）。
   - 其餘 24 格：平均 TTFT −12.6%～−59.9%，P95 −13.8%～−48.3%。「−14」看起來是取了 P95 的端點，「−60」取了平均的端點。
3. **#1、#5、#8：Strata 的細節要照原文改。**
   - #1：transient node 有 in-queue 和 in-flight 兩種，不只是「正在計算」。
   - #5：SSD 層有 best-effort 預取、wait-complete、timeout 三種策略。best-effort 中止預取後，剩下的部分只能 prefill，等於隱含的退回重算〔推論〕。所以「完全沒有 recompute vs load」要改寫成「沒有以成本比較的選擇」。
   - #8：要補「不計 system prompt」這個限定語。
4. **Strata 的版本問題**（#6 的 §6、#8、#9）。這三處只出現在 **OSDI '26 版**，arXiv v1 沒有。v1 的 §6 是 Related Work，結論只寫要降低 I/O kernel 開銷。引用時要寫 OSDI 版。
5. **#19 DLPM 的節號**：「誰付」那段在 **§9** Limitation and Future Work（p.12），不是 §8。§8 是 Related Work。

**附帶發現（不影響判定，但寫作時要知道）**

- **#11**：PR #44295 的表沒有標出指標。從 issue #44294 的 2026-06-03 留言可以看出，指標是「相對不開卸載的吞吐變化」。
  - 併發 20 時，「本 PR」−32.0% 比「未修」−20.5% 更差。作者留言卻寫它比 unpatched 好，自相矛盾。
  - PR 表的 GPU 狀態寫「2×／5× overflow」，issue 本文寫 gmu 0.9 時 GPU KV 沒有受限，兩者不一致。
- **#35**：F2 量的是 burst makespan（vLLM 0.28，MI300X）。#44294 量的是 TTFT 與吞吐（0.19.1，L40S）。「沒重現」是在不同指標、不同版本下的結論。
- **#13、#11**：issue #44882 在 2026-10-06、PR #44295 在 2026-10-15 都被標了 stale。照 bot 規則，30 天內沒有動靜就會被自動關閉。

---

## §1 逐條核對

格式：原文欄一律是轉述，不直接引用；技術名詞、符號名、數字照原樣。

### Strata（arXiv 2508.18572 v1；OSDI '26）

| # | 判定 | 看到的原文（轉述） | 位置 | 正確寫法（FIX） |
|---|---|---|---|---|
| 1 | **FIX** | HiRadixTree 加 transient node，帶兩種標記：in-queue 是「有請求引用了新 context」，in-flight 是「該段 cache 正在計算」。排程掃佇列時會插入 in-queue 節點。請求命中 transient node，就延到下一輪並放在等待佇列最前面。只有命中 token 數超過門檻（預設 100）才延後。要防的兩種情況：(i) 同一批裡共用同一個 miss；(ii) async 排程不知道 cache 還在算。 | OSDI §4.3.1，PDF p.7–8（印刷 6–7）；arXiv v1 §4.3.1，p.6 | 命中的是 **transient node（in-queue：本輪掃描中已有請求要算、尚未開算的新 context；或 in-flight：正在計算）**，不只是「正在被計算」。命中 token 數超過門檻（預設 100）時，延到下一輪、放在等待佇列最前面。 |
| 2 | OK | 延後只看「命中 transient node 的 token 數 > 門檻」，是二元判斷。沒有依共用者人數排序，也沒有為了湊更多共用者而等。transient node 是在「掃當下佇列」時插入的。Fig. 4 的 `hit_count` 和 §4.4 的計數器只用於 selective write-through 的備份門檻。全文搜不到「依人數排序」「等待累積」「lookahead」這類機制。 | OSDI §4.3.1–4.3.2、§4.4（PDF p.8–9）；Fig. 4（PDF p.5） | — |
| 3 | OK | Alg. 1 的 `AddBundleHit`：每加入一個請求，就掃剩下的佇列，把 bundle-hit（共用同一 context）的請求優先加進同一批。bundle hit 被定義為 delay hit 的對應面。 | OSDI Alg. 1、§4.3.2（PDF p.8）；arXiv v1 p.6–7 | — |
| 4 | OK | `loading_bound` 定義為「批的總載入量÷總計算量」，超過門檻就算 loading-bound，預設比值 100，對應 Fig. 1 開始出現 stall 的點。超過門檻的請求移到降級清單，最後補位。bubble filling：批仍然 loading-bound 時，先不跑已準備好的 prefill，改送 decode batch 和載入並行。原文也提到可以塞 prefill batch，比較適合 P-D 分離。 | OSDI §4.3.2–4.3.3（PDF p.8）；arXiv v1 p.7 | — |
| 5 | **FIX** | §3.2：既有工作要嘛把載入當成可以忽略，要嘛在成本划算時改重算（引用 [23]＝Jin et al.〈Compute or Load KV Cache? Why Not Both?〉ICML 2025，即 Cake；arXiv v1 是 [20]）。長 context 重算成本隨長度上升，不吸引人。§4.2.1：SSD 層預設 best-effort 預取，請求被排上時中止預取，只用已經在 host／GPU 的 cache。OSDI 版另外可設 wait-complete 或 timeout 策略。 | OSDI §3.2（PDF p.5，印刷 4）、§4.2.1（PDF p.7）、ref [23]（PDF p.16）；arXiv v1 §3.2 p.4、§4.2.1 p.5、ref [20] | Strata **沒有以成本比較 recompute 與 load 的逐請求決策**。它認為長 context 重算不吸引人，並把「成本划算時改重算」歸給 Cake。**但 SSD 層有策略選擇**：預設 best-effort，被排上就中止預取、只用已到 host/GPU 的部分（其餘只能 prefill，屬隱含的退回重算〔推論，原文未明寫〕）；OSDI 版可改成 wait-complete 或 timeout。 |
| 6 | OK | 摘要提到 GPU HBM／CPU memory／SSD 三層，Fig. 4 有 External Storage。§4.2.1：storage 層命中時，伺機把資料預取到 host，預取延遲和請求的排隊延遲重疊（另有 wait-complete／timeout，見 #5）。§6 Discussion 的 Fairness 段：現在的策略只藏 host→HBM。為「從更慢 storage 預取」創造排程餘裕，同樣是有前景的方向，超越現在只靠排隊延遲重疊的做法。 | OSDI 摘要（PDF p.2）、§4.2.1（PDF p.7）、§6（PDF p.14，印刷 13）；arXiv v1 §4.2.1 p.5 | （註：**§6 這段只在 OSDI 版**。arXiv v1 的 §6 是 Related Work，結論只談降低 I/O kernel 開銷。v1 也寫明 benchmark 都沒用 disk。） |
| 7 | OK | §5.3：沒特別說明時，分析都用 LooGLE、Qwen2.5-14B、H200。§5.3.3 用 LooGLE 造出 min cache distance（同 context 的請求排在一起）和 max cache distance（平均分散）兩種負載。min 時 delay-hit 緩解讓峰值吞吐 +42%；max 時沒有效益。 | OSDI §5.3、§5.3.3、Fig. 11（PDF p.11–12，印刷 10–11）；arXiv v1 §5.3.3、Fig. 11（p.10） | — |
| 8 | **FIX** | Mooncake agent tool trace 中，38% 的請求和「1 秒視窗內到達的至少另一個請求」共用至少 6k token 的前綴，而且**不計 system prompt**。 | OSDI §3.2（PDF p.5，印刷 4）。**arXiv v1 沒有這句** | 補上限定語：「……共用 ≥6K token 前綴（**不計 system prompt**）」。並註明出處是 **OSDI '26 版** §3.2。 |
| 9 | OK | §6 Discussion 的 Fairness in Scheduling：雖然 §4.3.2 有防餓死機制，排程以整體 I/O 與計算效率優先，仍可能對請求不均，違反個別 SLO。自然的延伸是讓組批 SLO-aware、fairness-aware。 | OSDI §6（PDF p.14，印刷 13）；arXiv v1 沒有 | （註：只在 OSDI 版。） |

### vLLM

| # | 判定 | 看到的原文（轉述） | 位置 | 正確寫法（FIX） |
|---|---|---|---|---|
| 10 | OK | 環境：2×L40S（TP＝2）、Qwen3-8B BF16、vLLM 0.19.1、50 個併發請求共用 10,000-token 前綴、`gpu_memory_utilization`＝0.9。在併發 50 下，0.17.1 開卸載 TTFT 0.90 s，0.19.1 開卸載 11.46 s，不開卸載 5.95 s。關閉時間 2026-10-07T01:58:04Z，stateReason＝NOT_PLANNED，標籤 stale，由 github-actions bot 以閒置為由關閉。 | `gh issue view 44294`：body 的表與 Environment 段；timeline 的 closed 事件 | — |
| 11 | OK | state OPEN，未合併。標籤：bug、needs-rebase、stale（2026-10-15 標上）、kv-connector。表格（L40S、Qwen3-8B、10K 前綴）：併發 20 時 Unpatched／Convoy-only fix／This fix＝−20.5%／−38.0%／−32.0%；併發 50 時＝−40.7%／−33.8%／−32.4%。**指標沒寫清楚**：表頭沒有指標名稱。Problem 段寫的是「相對 0.17.1 吞吐少 34%」，基準又不同。要看 issue #44294 在 2026-06-03 的留言（例如 158 vs 267 tok/s＝−40.7%）才知道是「相對不開卸載的吞吐」。 | `gh pr view 44295`：state、labels、body；issue #44294 的 comments | （註：併發 20 時本 PR −32.0% 比未修的 −20.5% 差，作者在 issue 留言卻寫它比 unpatched 好。PR 表寫 2×／5× overflow，issue 本文寫 GPU KV 沒有受限。） |
| 12 | OK | 完整標題是〈OffloadingConnector: Prevent redundant loads〉。合併時間 2026-01-21T01:15:43Z，merge commit `7013e9ac8f65…`。 | `gh pr view 29087` | — |
| 13 | OK | issue #44882 是 OPEN。2026-06-08 曾被關閉後隨即重開，2026-10-06 被標 stale。內容要求：第一個請求只 materialize 一次外部前綴；之後的請求把 block table 掛到同一批 block 並增加 refcount；如果還在載入就等同一份，不另發載入。PR #57418 是 OPEN、非 draft：在多個併發請求之間共用 GPU 目的地與一次 in-flight 外部前綴載入；block pool 索引 pending prefix，並 pin 住 block 直到傳輸完成；後到者透過一般的配置／引用 API 掛上。目前只支援內建 CPU offloading，條件很窄。效能沒有驗證，PR 是 AI 協助撰寫。 | `gh issue view 44882`（body、timeline）；`gh pr view 57418`（body） | — |
| 14 | OK | `_chunks_being_loaded`：定義在 650–654 行（650–651 是註解，652–654 是 `set() if enable_prefix_caching else None`）。延後區塊在 1026–1050 行：1026 註解、1027 `if self._chunks_being_loaded:`、1043 檢查任何 key 正在載入、1050 `return None`。延後區塊位於 `OffloadingConnectorScheduler._lookup_complete_chunks`（854 行開始）。其他用處：1128、1470–1471、2219–2220、2370–2371。commit 是 2026-10-26 的〈Add Muse Spark Open support (#61879)〉。 | `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` @ `0f6b26191064384aa35060c87f4ed570c7875e66`（共 2374 行） | — |
| 15 | OK | 標題〈[Attention] Remove cascade attention〉，MERGED，合併時間 2026-10-12T17:41:08Z。外掛用的 API 留下來當 deprecated no-op；`--disable-cascade-attn` 仍可解析，但只發警告、沒有作用。 | `gh pr view 61126` | — |

### 其他系統

| # | 判定 | 看到的原文（轉述） | 位置 | 正確寫法（FIX） |
|---|---|---|---|---|
| 16 | OK | PR 加入 HRRN 排程，2026-09-08 合併。conc＝100 的表：FCFS p99 TTFT 60.5 s，LPM 213.3 s（+252.56%）。測試環境：真實生產流量（混合 agent 負載，輸入最長 1M）、GLM-5.2 nvfp4、1P1D on 2×8×B200；prefill 開 `hicache`、attn-cp＝8。 | `gh pr view 32911 --repo sgl-project/sglang`：Motivation 的表與 Test environment | — |
| 17 | **FIX** | cLPM 的 key(r)＝(section, −h, −score, −\|cluster\|, arrival)，即 Eq. (1)。score 是路徑上 pending_count×\|edge\| 的加總（depth-weighted demand）。原文說新增的是兩個 cluster-aware tiebreak：−score 和 −\|cluster\|。選用的 group-major（GM）會讓一個 cluster 連續送出。PE 淘汰時用 max pending_count×depth 保護 block。評估：H100、SGLang `mem_fraction_static` 0.88／vLLM gmu 0.9；KV pressure 定義為 working set÷arena。HiCache、offload、host、SSD 這些字只出現在相關工作。 | §3.1–3.2、Eq. (1)（p.4）；Table 3（附錄）；Table 5（p.16）；Limitations（p.10） | 排序鍵含 cluster 大小，它在 −score 之後當 tie-break。**但排在它前面的 −score 本身就是「等待中共用者的深度加權需求」**。另有 group-major 讓整個 cluster 連續送出，淘汰也依 pending count 保護。評估只有 GPU KV，沒有 host／SSD tier。 |
| 18 | OK | provider 對同一個 KV 物件合併 staging，只為還沒被其他已承諾 claim 覆蓋的位元組預留容量，避免重複的 SSD 讀取。摘要說實作在 vLLM＋LMCache 上，不改請求排程。§2.3 拿 Bidaw 做對照：TempoKV 只決定承諾時機，不改 runtime 的請求排程。§4 說整合保留 vLLM 的請求排程策略。 | 摘要（p.1）；§2.3（p.2）；§3.2「Shared staging and resource lifecycle」段（p.4）；§4（p.4） | — |
| 19 | **FIX** | 「Prefix Sharing Among Clients」段：本文只探討單一 client 內的 prefix 共用。不同 client 之間如何公平共用（例如多個 client 共用同一 prefix 時由誰扣 quantum）留作未來工作。 | p.12，**§9 Limitation and Future Work**（§8 是 Related Work，§10 是 Conclusion；用 `-layout` 文字確認） | 頁碼 p.12 正確，**節號改成 §9**。 |
| 20 | OK | 每個請求、每一層保留固定大小、B 個 slot 的 GPU cache（B ≥ k，和 context 長度無關）。每個 request-layer cache 各自用 LRU 管理。完整 KV 放在 host pool。 | §3.1「Bounded device footprint」（p.5）；§3.2（p.6） | — |
| 21 | OK | 稀疏度的變異有兩個來源。同一 context 裡的不同 query、或不同任務，在同一個 attention head 上的稀疏比例差很多，見 Fig. 4(b)。圖說寫 (b) 是 decoding steps and tasks，模型 Llama3-8B-1048K。 | §2.3（p.4 左欄）；Fig. 4 本體在 p.3 | （註：Fig. 4 本體在 p.3，引用它的文字在 p.4。） |
| 22 | OK | PMLR v306 頁面：ICML（第 43 屆，Seoul），pp. 101429–101443，citation_publication_date 2026/09/29。§4.4 Incremental KV Cache Management：每輪只非同步存下新 token 的狀態（incremental checkpointing）；GPU 記憶體空出來時，在背景逐步預取被逐出 token 的 KV，和計算完全重疊（Fig. 5(d)）。 | PMLR 頁面的 meta；PDF p.1 頁尾；§4.4（p.6）；Fig. 5（p.5） | （註：對象是 offline 請求的搶占與恢復，不是前綴共用。） |
| 23 | OK | KV 需求低的模型（worker）把閒置的 KV block 借給需求高的模型（master），存放 master 的 prefix cache，經 NVLink 搬運。master 逐層把 KV 從 worker 的 GPU 取回並計算。相關工作段強調的是異質架構之間共用「KV 記憶體緩衝」。 | 摘要（p.1）；§1（p.2）；§3.3（p.6）；§3.5（p.8）；相關工作 | （註：「另一個模型不讀」是由設計推得：KV 是 master 的格式，worker 架構不同。原文沒有逐字寫「worker 不讀」。） |
| 24 | OK | 摘要：LMCache 在引擎與 query 之間共用 KV；支援 cache offloading（跨 query 的 prefix 重用）和 PD 分離，PD 分離被括號標成 cross-engine/GPU cache transfer。Fig. 2(b) 把 PD 分離標為 cross-engine sharing。另有 P2P KV cache sharing：某個實例 miss 時，可以去別的 peer 實例取。 | 摘要（p.1）；Fig. 2（p.3）；§2 列點（p.4）；controller API 的 P2P 段（p.10） | （註：原文沒有逐字寫「同一模型」，也沒有跨模型共用的內容。字面上的 cross-engine 指的是 PD 分離。） |
| 25 | OK | Status-Aware Scheduling：每個 cache node 有四種狀態（在 GPU、備份在 CPU、loading、offloading）。請求的 prefix 還在載入，就暫時跳過，先做其他已就緒的請求。遇到 loading 的節點會跳過，避免重複載入。載入完成後由背景執行緒更新狀態。 | §3.2（p.6） | — |
| 26 | OK | 單一請求：在 K≥j₁≥…≥j_L≥0 的條件下求 min max(T_comp, T_load)，是 Eq. (2)。token–layer 平面做階梯狀切分，用 DP 求解。批次版把每個 request–block 對當成規劃項目，是 Eq. (5)，再用同樣的平衡步驟選全批的操作點。全文「shop」出現 0 次。附錄有用到 makespan 這個詞。 | §3.2（p.5–6）、Eq. (2)、Eq. (5)、Fig. 3 | — |

### 古典文獻

| # | 判定 | 看到的原文（轉述） | 位置 | 正確寫法（FIX） |
|---|---|---|---|---|
| 27 | OK | 推下界與比較時有兩個假設：(1) 客戶端緩衝足夠；(2) 整個檔案由客戶端依序消耗，不用暫停、倒轉、快轉。§2 描述的 stream tapping／grace patching：晚到的客戶加入進行中的 multicast 並緩存，另開一條 unicast patch stream 補上它到達前已送出的前段；patch 到會合點就結束。超過門檻時改開新的完整 multicast。 | 預印本 §1（p.2）、§2（p.3）。書目由 Crossref 確認：TKDE 13(5):742–757，DOI 10.1109/69.956098 | （註：patching 是作者回顧的既有技術，出自 Hua et al. 1998、Carter & Long 1997，不是本文提出。本文提出的是 HMSM。） |
| 28 | OK | 平均絕對 PWT 的 **AA Policy 1**：P_i＝B_i²÷(λ_i·t_i^s)。分母 λ_i·t_i^s 被稱為 sharability。另有 AA Policy 2 再減 t_i^s·Σλ_j。 | §4.1，PVLDB 1(1) p.962（PDF p.5）；DOI 10.14778/1453856.1453960 | （註：這是兩個公式中的 Policy 1。） |
| 29 | OK | anticipation heuristic 是一條成本效益式。benefit＝候選請求定位時間減去該 process 的預期定位時間；cost＝max(0, 預期中位 thinktime − elapsed)；benefit > cost 才等待。等待長度是預期 95 百分位 thinktime 減去 elapsed。 | 預印本 §3.3（p.5）；SOSP '01 pp.117–130，DOI 10.1145/502034.502046 | — |
| 30 | OK〔摘要〕 | 兩機 flow shop、每個 job 有三道工序：第一道在 M1，第三道在 M2，中間那道可在任一台做但不可中斷。摘要說問題是 NP-hard。任意排序、不插入閒置時間的最差比為 2；建四個排程取最好的，最差比 3/2；另有 PTAS。 | 摘要（RePEc 與 Greenwich 典藏頁）；Crossref：Annals of OR 129(1–4):171–185，2004，DOI 10.1023/B:ANOR.0000030687.72169.c7 | （註：沒拿到全文，NP-hard 是證明的還是引用的無法確認。） |
| 31 | OK | Zukowski、Héman、Nes、Boncz，〈Cooperative Scans: Dynamic Bandwidth Sharing in a DBMS〉，VLDB '07（Vienna），提出 Active Buffer Manager 與 relevance 策略。PBM 追蹤各 scan 的進度，估計每個 disk page 下一次被消耗的時間，用來近似 OPT：留下快要用到的頁，逐出最晚才用的頁。 | Zukowski PDF p.1；PBM：§1（p.1）、§3（p.6），PVLDB 5(12):1759–1770，DOI 10.14778/2367502.2367515 | — |
| 32 | OK | Fig. 4a 把 overlap 分成四類：linear、step、full、spike。spike 是除非同時開始、否則不能重疊的操作，例子是必須依表序輸出的 table scan，只有在第一個輸出頁還在記憶體時才能搭便車。依序要求嚴格的 file scan 屬於 spike。 | §3.2（PDF p.4–5）；§4.3.2（PDF p.8）；SIGMOD '05 pp.383–394，DOI 10.1145/1066157.1066201 | （註：原文也寫了例外。buffering 能把 spike 變成 step。§4.3.2 有兩個例外：父運算子不在意順序時拆成兩個 packet；選擇率高時用 materialization。） |

### 本機 repo 的數字

| # | 判定 | 看到的原文 | 位置 | 正確寫法（FIX） |
|---|---|---|---|---|
| 33 | OK | llama・T2-T4（135 題）的 QRY1 列：重疊@16 − k/63＝+61.0 [+59.4, +62.7]，recall@16 − k/63＝+14.6 [+8.1, +21.4]。qwen・T2-T4 的 QRY1 列：+51.3 [+48.4, +54.0]、+21.3 [+15.8, +27.0]。說明列寫 k/63：16→0.254（16÷63＝0.2540）。 | `results/m16_stage3_mi300x/c1a_predictive/tables_c1a.md`：5、7、21、189、203 行 | — |
| 34 | **FIX** | 文件 40 行：「各請求各自最佳 → 同時考慮所有 session（joint）」每一格都是負的（−14% 到 −60%），標〔模擬〕。116 行：grid 實測平均 TTFT÷模擬，每格取 3 次的中位數，全部落在 0.973–1.050。117 行重述 14–60%。**用結果檔重算**：`prediction_llama.json` 的 `*_gainB_joint_vs_perreq`，N＝4、8 共 32 格。homog_cpu 8 格是 **0.0**。其餘 24 格：平均 −12.6%～−59.9%，P95 −13.8%～−48.3%。`grid_llama.csv`（171 列，全部 idle、未污染）的每格中位數 ratio_mean＝0.973–1.050，和文件相符；但單次最高到 3.58。 | `docs/stage3_20261019/experiment_log.md`：40、116–117 行；`results/m16_stage3_mi300x/i1_concurrent_restore/prediction/prediction_llama.json`；`grid_llama.csv` | 「N＝4、8 的 32 格中，homog_cpu 8 格 joint 和各自最佳相同（0%）；其餘 24 格 joint 較差：平均 TTFT −12.6%～−59.9%、P95 −13.8%～−48.3%〔模擬，互不干擾模型，joint 只試全算／全載〕。GPU 實測÷模擬（每格 3 次中位數，GPU 閒，N＝1、4、8）＝0.973–1.050。」 |
| 35 | OK | F2：#44294 的「開卸載慢一倍」在 MI300X 上沒有重現。和同一個注意力後端比，開卸載打平或更快。真正的代價是開了 CPU 卸載後，vLLM 在 ROCm 上強制改用較慢的 TRITON 注意力，高併發時慢約一倍。experiment_log 的 F2 段數字支持這個說法：P＝10K、N＝64 時，A′ 4.41 s 對 B 4.42 s。 | `docs/stage3_20261019/results_analysis.md` §4.13（243–254 行，F2 在 247 行）；`experiment_log.md` 252–256 行 | （註：F2 量的是 burst makespan，vLLM 0.28。#44294 量的是 TTFT 與吞吐，vLLM 0.19.1。指標和版本都不同。） |
| 36 | OK | §2 表的「全部 BF16」：Llama-3.1-8B・128K 從 CPU／本地 NVMe／NFS 載入是 0.30／2.29／48.34 s。§4：a＝2.148e-05、c＝4.843e-10，由 H200 實測 16K 0.482 s、32K 1.224 s 擬合；128K 外推 11.14 s。**重算結果**見下方。 | `docs/stage3_design_20261024/mechanisms/arith.md`：24–26、39、49、55 行 | — |

**#36 的重算細節**

| 項目 | 自己算的結果 | 文件或宣稱的數字 |
|---|---|---|
| 擬合常數（由 16K、32K 兩點反解） | a＝2.1484e-05、c＝4.8429e-10 | a＝2.148e-05、c＝4.843e-10，相符 |
| 128K 外推（n＝131072） | 11.136 s，n² 項佔 75% | 11.14 s，相符 |
| 10,000 token 的 KV 大小（每 token 128 KiB） | 1.2207 GiB | — |
| CPU 載入（53.55 GiB/s） | **0.0228 s** | ≈0.023 |
| NVMe 載入（16 GiB÷2.29 s） | **0.1747 s** | ≈0.175 |
| NFS 載入（16 GiB÷48.34 s） | **3.688 s** | ≈3.69 |
| 重算 a·n＋c·n² | **0.2632 s** | ≈0.26 |
| 重算÷載入 | CPU 11.5×、NVMe 1.51×、NFS 0.071× | — |

四個數字都成立。使用時要注意三點：

- 10K 低於擬合點（16K、32K），所以是往下外推。
- CPU 的頻寬是 H200 實測，NVMe 和 NFS 的常數來自 MI300X 的 tier_params，跨了平台。
- 重算指的是整段 10K 冷 prefill。
