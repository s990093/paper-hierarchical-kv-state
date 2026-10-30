# 查新報告：ICPE 2027 的 limit study（2026-10-30）

> 上一次查新是 2026-10-01（[`NOVELTY_CHECK_20261001.md`](NOVELTY_CHECK_20261001.md)），查的是當時的方法「寫入時依位置決定」。之後主張改成 ICPE 的 limit study（只用 H200，記憶 `icpe-h200-only-thesis`），這次查的是新主張。重點放在 2026-05 之後的論文。
>
> **審查方式**：self-review，不是 cross-model。這台機器沒有 Codex MCP；依 CLAUDE.md §5，開 subagent 當審查者要使用者明確要求，這次沒有開。
>
> **論文核實**：repo 裡沒有 `verify_papers.py`，所以每一篇都直接抓 arXiv 摘要頁，確認標題、作者與日期。標記：〔內文〕＝讀了 arXiv HTML 全文（或大部分）；〔摘要〕＝只讀摘要頁；〔UNVERIFIED〕＝只看到搜尋摘錄，細節不能引用。
>
> **我們的數字**：只抄 [`docs/h200/ICPE_EVIDENCE.md`](h200/ICPE_EVIDENCE.md) 的 H200 列與 T1／T3 的結果節（[`T1`](h200/icpe/T1_h200_cpu_tier.md)、[`T3`](h200/icpe/T3_planner_sensitivity.md)），全部是〔模擬〕，除非另外標〔實測〕。

## 提案（這次查的東西）

在 Cake（ICML'25，從前段重算、從後段載入、中間會合）加上固定的分層規則之後，再加「預知」最多值多少？用 oracle（完美預測器）量上界，兩條真實 trace（Mooncake conversation、FreeInference agent），模擬在 GPU 上驗證過，門檻事先登記。發現：

1. 寫入時的預知（依位置決定層級或精度）在 GPU 上量不到收益（A 節〔實測〕）。
2. 回來時間的 oracle：agent trace（FreeInference）上 X1 最大 0.30%（T1）、X5 最大 2.44%；聊天 trace（conversation）上界高（O-PF 50.5%／46.5%，T1），但免費的打字提示拿到 O-PF 改善的大部分（中位：SSD、b32 兩種規劃器都是 0.92–0.94；NFS 級 Cake 1.12、STAIR 0.93）。
3. 還原方式用固定的分層規則就夠（CPU 層全部載入、慢層 Cake），不需要線上選擇器。
4. 值得做的兩件事不用預知：INT4（慢層還原快 37–55%〔實測〕）、MPS 的固定 SM 預留（是延遲與背景吞吐的取捨，不是免費的收益；E2E-NOGO）。
5. 可證偽的判準：預知只在 CPU ≤256 GiB、慢層慢、聊天型的回來時值 ≥15%（X5）。

## 核心主張與最接近的前作

1. **「預知的天花板」本身（limit study 的框架）**
    - 最接近：**When Fancy Eviction Fails**（Liu、Yu、Yang，arXiv 2609.28870，v1 2026-09-24，v2 10-01）〔內文〕。用 FreeInference＋Chutes 兩條 production trace（Bailian 四條、AgentX 驗證外推），比 14 種淘汰演算法，HBM 24–120 GiB 與 DRAM／SSD 池 0.25–12 TiB 兩種設定。結論：花俏的策略幾乎贏不了 LRU，**但和 Belady 的差距很大**；原因是 session 的節奏讓「最近用過」特別準。有兩個離線 oracle（Belady、BeladyCompute）。
    - 第二接近：**Where Should the KV Cache Live?**（arXiv 2609.16215，2026-09-14）〔內文〕。GPU／CPU／SSD 三層的離散事件模擬，有一個「看得到未來的預取 oracle」，結論是預取不值得它吃的頻寬，連 oracle 也贏不了不預取。
    - 差別：
        - Liu 等人的 miss 成本只算**重算**（隨位置變大），**沒有模擬從 DRAM／SSD 載回的延遲**，也沒提 Cake〔內文 §4.4、§5.2、§6.3〕；指標是命中率與 compute-savings ratio。我們的 miss 是「從慢層用 Cake 還原」，指標是 TTFT，oracle 加在 Cake 之上。
        - 方向不同：他們說「oracle 的差距大、但線上策略拿不到」；我們說「Cake 之上，連 oracle 的差距都小」（同一條 FreeInference）。**兩者不矛盾的前提**是：在我們的模型裡，CPU 層滿了就降級到慢層（`x1_sim.py` 開頭的模型摘要），所以命中率上的差距，大多變成「慢層還原比 CPU 層慢一點」，而不是「整段重算」〔判讀，沒有量過〕。
        - 2609.16215 用的是合成 workload、miss 時重算、沒有 Cake、沒有 GPU 驗證；摘要寫 oracle「全部格子都沒贏」，內文寫「超過一半的格子比不預取慢」，兩者不一致〔內文 §V-E〕。我們在聊天 trace 上 O-PF 的上界是高的，跟它相反；可能的原因是慢層慢、真實 trace 有長的回來間隔〔判讀〕。
2. **免費訊號拿到 oracle 的大部分（C2：打字提示預取）**
    - 最接近：**Ask the Tool, Don't Guess**（arXiv 2609.18849，2026-09-16）〔摘要〕：讓工具回報進度，用來決定 KV 留、走、何時載回；工具呼叫後 p90 TTFT 比 LRU 少 20.7%（只有 HBM）／20.8%（HBM＋DRAM），作者說接近 oracle。**這是 agent 端的同一個論點**。
    - 其他免費訊號：CachedAttention（ATC'24，用排程佇列預取，refs.bib `cachedattention2024`）；PEEK（arXiv 2607.02525，排隊中的請求）〔摘要〕；PCR（arXiv 2603.23049，排隊時從 SSD 拉到 DRAM）〔UNVERIFIED〕；SGLang、vLLM、Mooncake 都有「session 要回來了，先預取」的 RFC（issue #45128、#52113、#3417）〔UNVERIFIED，只看到搜尋摘錄〕。
    - 差別：我們的訊號是**使用者端**的打字提示，trace 是聊天，慢層是 SSD／NFS 級，底是 Cake，並且直接量「拿到 oracle 的幾成」（全尺寸模擬中位 ≥0.92；GPU 縮小版 0.81〔實測〕）。**沒找到用打字訊號預取 KV 的論文**。
3. **寫入時的預知沒用（A 節）**
    - 最接近（反方向，都說寫入時的決定**有用**）：SARIA（arXiv 2610.31169，2026-10-22）〔摘要〕：依重用歷史決定哪些塊寫 SSD，SSD 寫入少 57.1–58.9%，TTFT 比只用 DRAM 少 42.3–79.7%；Gandalf（arXiv 2610.30791，2026-10-22）〔摘要〕：依 attention 類型決定准入 SSD，SSD 壽命 7.3 倍；EfficientAgent（arXiv 2609.33762，2026-09-27）〔內文〕：依容量決定准不准寫。ICPE'25 *Better Memory Tiering, Right from the First Placement*（DOI 10.1145/3676151.3719378）〔UNVERIFIED，只確認書目〕：一般記憶體分層要「第一次就放對」。
    - 差別：這些的收益是 **SSD 壽命、寫入量、或「多一層 SSD 對只用 DRAM」**；我們量的是「寫入時依位置決定層級或精度」對 TTFT 的收益（對延後決定的版本）。兩者不衝突，但我們的主張要縮成「對 TTFT」，不能寫成「寫入時的決定沒用」。
4. **固定的分層還原規則，不需要線上選擇器**
    - 最接近：Cake；CacheFlow（arXiv 2604.25080，v2 2026-09-26）〔摘要；10/01 讀過內文〕：token × 層 × GPU 的批次 DP 規劃，還原快 2.24–3.00 倍；HCache（EuroSys'25）；When the KV Cache Is Slower Than the GPU（arXiv 2610.21500，2026-10-15）〔摘要〕：遠端層的載入與重算的成本模型，短於約 17K token 的命中不論頻寬多高都贏不了重算。
    - 差別：這些都是「怎麼還原更快」；我們量的是「依當下狀況線上挑還原方式」的價值，結論是固定規則就夠、線上選擇器常選錯（E3〔實測〕）。T3 用照 CacheFlow 想法寫的 STAIR 檢查過，判定字不變。
5. **不用預知的收益：INT4、MPS**
    - INT4 放冷層：UltraQuant（arXiv 2606.20474，EMNLP'26 Industry）〔摘要〕是 GPU 內的 4-bit KV、沒有分層；AMD ROCm 部落格「4-bit KV Caching in LMCache」〔UNVERIFIED〕把 4-bit KV 卸載到 CPU；TensorRT-LLM NVFP4（repo `07_advisor_qa.md` Q2）。
    - SM 預留：PD-Multiplexing（arXiv 2504.14489，green context）〔UNVERIFIED〕、DuetServe（arXiv 2511.04791）〔UNVERIFIED〕、Tutti、vLLM #42212（repo `08_kill_argument.md` §1.1）。
    - 差別：**這兩件都是已知做法**，我們只貢獻「在 H200 上值多少、不需要預知」的量測（和 10/28 大綱 §4 的結論相同）。
6. **預測型系統（我們的 limit study 要「具名檢驗」的對象）**
    - PBKV（arXiv 2605.06472）〔摘要〕：預測接下來幾步呼叫哪個 agent，對 LRU 最多 1.85 倍；SAGA（arXiv 2605.00528）〔摘要〕：KV 重用預測在 Bélády 的 1.31 倍內；CacheScout（arXiv 2608.14624）〔摘要〕：線上學 agent 轉移，TTFT 少 18–45%；CacheWise（arXiv 2606.16824）〔摘要〕：用工具呼叫的 metadata 預測，session 完成時間最多 3.5 倍；Continuum（arXiv 2511.02230，v7 2026-09-08）〔摘要〕：依重載成本設 TTL，把 KV 留在 GPU；KVFlow（NeurIPS'25）；Predictive Multi-Tier（arXiv 2604.26968）〔摘要〕。
    - 這些都報告預測有大收益。**從摘要看，沒有一篇的對照組是「分層儲存＋Cake 還原」**，大多是 GPU 內淘汰、miss 時重算〔摘要，要讀內文確認〕。如果確認成立，這就是我們最好的切入點：**它們的收益是對「miss 就重算」量的；換成分層儲存＋Cake，預知剩下的空間就小了**。

## 最接近的前作

| 論文 | 時間 | 出處 | 重疊 | 主要差別 | 讀到哪 |
|---|---|---|---|---|---|
| When Fancy Eviction Fails (2609.28870) | 2026-09-24／10-01 | arXiv | 同一條 FreeInference；「簡單策略就夠」；離線 oracle | 淘汰、命中率；miss＝重算；沒有載回延遲、沒有 Cake；它的 oracle 差距大 | 〔內文〕 |
| Where Should the KV Cache Live? (2609.16215) | 2026-09-14 | arXiv | 三層放置；預取 oracle 沒用 | 合成 workload；miss＝重算；沒有 GPU 驗證；摘要與內文不一致 | 〔內文〕 |
| Ask the Tool, Don't Guess (2609.18849) | 2026-09-16 | arXiv | 免費訊號（工具進度）接近 oracle | agent 工具呼叫、HBM／HBM＋DRAM；我們是使用者打字、慢層、Cake 之上 | 〔摘要〕 |
| CachedAttention (2403.19708) | 2024 | ATC'24 | 用排程佇列預取（免費訊號） | 沒有 oracle 上界；沒有 Cake | refs.bib 已有 |
| SARIA (2610.31169) | 2026-10-22 | arXiv | 寫入時決定寫不寫 SSD | 目標是 SSD 寫入量；收益對「只用 DRAM」 | 〔摘要〕 |
| Gandalf (2610.30791) | 2026-10-22 | arXiv | SSD 准入 | 目標是 SSD 壽命 | 〔摘要〕 |
| EfficientAgent (2609.33762) | 2026-09-27 | arXiv | 卸載什麼時候值得；寫入准入 | 只有 host 層；沒有 oracle；沒有 Cake | 〔內文〕 |
| CacheFlow (2604.25080) | 2026-04／09-26 | arXiv | 還原時重算＋載入的規劃 | 只在讀取時；沒有預取、沒有預測 | 〔摘要〕＋10/01 內文 |
| PBKV／SAGA／CacheScout／CacheWise／Continuum／KVFlow | 2025–2026 | arXiv／NeurIPS'25 | 預測帶來大收益（被檢驗的假設） | 對照組看起來是 GPU 內、miss 重算〔要讀內文〕 | 〔摘要〕 |
| KVCache Cache in the Wild (2506.02634) | 2025 | ATC'25 | trace 特性、「理想命中率」 | 沒有分層還原 | 〔摘要〕 |
| From Requests to Sessions（FreeInference trace 的論文） | 2026 | 沒有 venue、沒有 arXiv 編號（只有資料集卡的 BibTeX） | 同一條 trace 的特性分析 | 內容 NOT_VERIFIED | 〔UNVERIFIED〕 |

## 整體判斷

- **分數**：6／10。有明確的鄰居，但查到的論文都沒有這個結果。
- **建議**：**PROCEED WITH CAUTION**。
- **主要差異點**：查到的論文裡，沒有一篇在 **Cake 式分層還原之上**量預知的 **oracle 上界**（以 TTFT 計），也沒有一篇用真實的聊天＋agent 兩條 trace、模擬經 GPU 驗證、門檻事先登記。
- **差別為什麼薄，具體在哪**：
    1. **「簡單策略就夠、預測沒必要」這句話已經有人在同一條 trace 上說了**（Liu 等人，9/24）。我們不能把它當成新發現；新的地方只能是「**分層儲存＋Cake 之下，連 oracle 的上界都小**」，以及這跟他們「Belady 差距大」怎麼兜在一起。
    2. **C2「免費訊號拿到 oracle 的大部分」在 agent 端已經有人說了**（Ask the Tool，9/16）。我們的是使用者端（打字）、聊天 trace、慢層。
- **審稿人最可能引的前作**：When Fancy Eviction Fails、Ask the Tool、Where Should the KV Cache Live?、CachedAttention，以及預測型的 PBKV／CacheScout／Continuum／KVFlow。

## 什麼能讓差別站得住

1. **補一座橋接 Liu 等人**：在同一段 FreeInference 上，先算 LRU 與 Belady 的**命中率**差距（重現他們的方向），再算同一組設定在**分層＋Cake 之下的 TTFT** oracle 上界。如果命中率差距大、TTFT 上界小，就直接證明「差距被分層＋Cake 吸收了」——這是一張圖、只用 head node 的模擬〔沒有跑，要先登記〕。
2. **讀預測型論文的內文**（PBKV、CacheScout、CacheWise、Continuum、KVFlow），確認它們的對照組是不是「miss 就重算」或「只有 GPU」。成立的話，§1.2「被檢驗的假設」就有具名的對象，論文可以寫「它們的收益是對重算量的」。
3. **C2 定位成 Ask the Tool 的使用者端對應**，並寫清楚：我們量的是「拿到 oracle 的幾成」（中位 ≥0.92〔模擬〕、0.81〔實測縮小〕），它量的是對 LRU 的改善。
4. **A 節的主張縮成「對 TTFT」**，正面引 SARIA、Gandalf、EfficientAgent：寫入時的准入對 SSD 壽命與寫入量有用，那是不同的目標。
5. **正面處理 2609.16215 的反方向結果**（它說預取 oracle 沒用、我們說聊天 trace 上預取上界高）：指出合成 workload、miss 重算、競爭的 PCIe 這三個差別。

## 建議定位（一句話，審稿人可驗證）

> 近期的系統用預測改善 KV 快取（PBKV、CacheScout、Continuum），而 Liu 等人發現在命中率上離 Belady 還很遠；本工作在 H200 上，於 Cake 式分層還原之上量預知的 oracle 上界（以 TTFT 計），發現在真實 agent trace 上 ≤2.44%、在聊天 trace 上免費的打字提示拿到 oracle 預取改善的 92% 以上（全尺寸模擬的中位；GPU 縮小版 81%），並給出預知值得做的條件（CPU ≤256 GiB、慢層慢、聊天型回來）。

## 對 ICPE 故事的影響

- **主軸不用換**：查到的論文沒有一篇已經有「Cake 之上的 oracle 上界」這個結果。
- **要改說法的地方**：
    - 「簡單就夠」不能當賣點，要改成「**為什麼**簡單就夠：分層＋Cake 讓 miss 變便宜，連 oracle 都沒空間」。
    - C2 要引 Ask the Tool；A 節要縮成「對 TTFT」。
    - INT4、MPS 照 10/28 大綱，寫成量測，不寫成提出。
- **時間上的風險**：9/14–10/22 之間至少有 6 篇相關的預印本（2609.16215、2609.18849、2609.28870、2609.33762、2610.30791、2610.31169）。這是競賽，不是否決；但 ICPE 摘要 11/9 截止前，最好再快速查一次 arXiv。
- **對「換成方法當主角」（待決事項 4）的參考**：近 6 個月裡，agent 的 KV 管理方法至少有 PBKV、SAGA、CacheScout、CacheWise、SARIA、Gandalf、EfficientAgent 這些，方法論文的新穎性門檻比 limit study 高〔判讀〕。

## 來源

- https://arxiv.org/abs/2609.28870 （When Fancy Eviction Fails；全文 https://arxiv.org/html/2609.28870v2）
- https://arxiv.org/abs/2609.16215 （Where Should the KV Cache Live?；全文 https://arxiv.org/html/2609.16215）
- https://arxiv.org/abs/2609.18849 （Ask the Tool, Don't Guess）
- https://arxiv.org/abs/2609.33762 （EfficientAgent；全文 https://arxiv.org/html/2609.33762）
- https://arxiv.org/abs/2610.31169 （SARIA）
- https://arxiv.org/abs/2610.30791 （Gandalf）
- https://arxiv.org/abs/2604.25080 （CacheFlow）
- https://arxiv.org/abs/2410.03065 （Cake；PMLR v267 `jin25d`）
- https://arxiv.org/abs/2605.06472 （PBKV）
- https://arxiv.org/abs/2605.00528 （SAGA）
- https://arxiv.org/abs/2608.14624 （CacheScout）
- https://arxiv.org/abs/2606.16824 （CacheWise）
- https://arxiv.org/abs/2511.02230 （Continuum）
- https://arxiv.org/abs/2604.26968 （Predictive Multi-Tier）
- https://arxiv.org/abs/2607.02525 （PEEK）
- https://arxiv.org/abs/2606.20474 （UltraQuant）
- https://arxiv.org/abs/2610.21500 （When the KV Cache Is Slower Than the GPU）
- https://arxiv.org/abs/2506.02634 （KVCache Cache in the Wild，ATC'25）
- https://arxiv.org/abs/2608.13573 （A Year in LLM Serving；同一個實驗室的 Chutes trace）
- https://github.com/HarvardMadSys/freeinference_agentic_trace 、https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace （FreeInference trace 與論文的 BibTeX）
- https://api.openalex.org/works/doi:10.1145%2F3676151.3719378 （ICPE'25 Better Memory Tiering 的書目）
- 只看到搜尋摘錄、沒有核實（〔UNVERIFIED〕）：arXiv 2603.23049（PCR）、2504.14489（PD-Multiplexing）、2511.04791（DuetServe）；AMD ROCm 部落格 4-bit KV in LMCache；SGLang #45128、vLLM #52113、Mooncake #3417
- 查過、和本題無關：arXiv 2606.01065（Leyline）、2610.28577（Cache Engineering for Agents）、2606.20577（Human-Less LLM Serving）、2509.16857（ShadowServe：摘要沒提 MPS／green context）
