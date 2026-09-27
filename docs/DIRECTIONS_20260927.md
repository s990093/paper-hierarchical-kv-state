# Tiara 方向補充：9/24 之後的 15 個新方向（2026-09-27）

> **這份是什麼**：接在 `DIRECTIONS_20260924.md`（故事 1–7）後面。9/24 那份覺得方向不夠多，所以這次專找「9/24 沒涵蓋的」：跨 agent／adapter 的 KV、hybrid 模型的 recurrent state、P/D 與遠端傳輸、reasoning 模型、契約／經濟／隔離。編號接著 9/24：**故事 1–7 → 方向 8–22**。
>
> **怎麼查的**：`deep-research` workflow（run `wf_00cb5dd9-097`，110 個子 agent）搜 5 個角度、讀 28 個來源、擷取 140 條 claim，挑 25 條做三票對抗式驗證（10 條通過、15 條否決）。之後我自己抽查 13 個關鍵來源的原始頁面，並讀了 vLLM `v0.28.0` 的兩個原始檔。
>
> **標記**：沿用 9/24 的【A】【B】【原文】【網路】【算術】【判讀】，另加四個證據等級：
> - 【3票】通過三票對抗式驗證（投票者是 Claude subagent，**不是 cross-model**）。
> - 【抽查】只有一個 agent 擷取，但我在 9/27 重讀了原始頁面，核對過標題、作者、日期與關鍵句（不是全文驗證）。
> - 【擷取】只有一個 agent 擷取，沒人核對。**引用前必須回原文。**
> - 【原始碼】我讀了 vLLM `v0.28.0` 的原始碼，但沒有執行。
>
> **出處編號**：`S01`–`S28` 指 `research_20260927/evidence_claims.md` 的來源（每條 claim 都附原文引句與票數）；`F0`–`F8` 指 `research_20260927/verified_findings.md` 的綜合結論原文。
>
> **這份文件裡沒有任何一個數字是我們的量測。** 引用的數字都是別人在別的硬體上量的；【算術】是用別人的數字算的；可行性的週數是規劃估計。依 `CLAUDE.md` 規則 1，這些數字只能當假設，不能寫進 `results/`。

---

## 0. 先講結論

1. **15 個新方向分成四組**（總表見 §2）：
   - **I. 要放的東西不只 KV（方向 8–15）**：hybrid 模型的 recurrent-state checkpoint、跨 adapter／fine-tune 借用的 KV、跨位置共享的 canonical chunk、同一 block 的多精度副本與 bit-plane。每一種都打掉 `main.tex` 形式化的一條前提（每個 block 一個 ℓ_i、一個 π_i；只有 KV；前綴命中；KV 只屬於一個模型）【F0】。
   - **II. 傳輸路徑（方向 16–17）**：故事 2 的「每筆固定成本」在另一條路徑（NIXL P/D、GB200）上被獨立量到。而且那個案例裡藏著一個耦合：**命中越多、實體佈局越碎、描述符越多、搬得越慢**。
   - **III. reasoning／agent 改變了負載形狀（方向 12、18、19）**：thinking token 產生的「死 block」、decode 生出來的 KV、品質損失拉長 agent 軌跡。
   - **IV. 契約、經濟、隔離（方向 20–22）**：TTL 最短壽命契約與 keepalive、快取隔離範圍的代價與「哪一階」側通道、碳與美元。
2. **證據最硬的是方向 8**（hybrid 模型的 state checkpoint 跟 KV 一起分層）【3票】。但它也最擠：8/31–9/23 這四週內就冒出 DASC、Tail-Replay、SGLang PR #39436、SGLang RFC #40865。
3. **最便宜、最新、而且直接接得上既有故事的是方向 18**：vLLM `v0.28.0` 的 `OffloadingConnector` 預設會把 decode 產生的 block 一起卸載（只有開 `offload_prompt_only` 才不會）【原始碼】。而 client 在下一輪剝掉 thinking 之後，這些 block 永遠不可能再命中【抽查，S15】。3090 上幾天就能量到「寫進 CPU／SSD 卻永遠不會被讀回」的比例。它也給故事 7 一個答案：**token 的角色（prompt／thinking／answer／tool）在寫入當下就知道**，是注意力分數（AUC ≈ 0.5）以外的准入訊號。
4. **如果老師一定要機制型，方向 15（bit-plane 分層）最貼近我們手上的結果**。它把 INT8 拆成 INT4 anchor 加 INT4 residual：anchor 常駐 HBM、residual 放到 CPU／SSD，decode 先投機、再驗證。這是唯一能把故事 1 的「量化取代放置」翻成「互補」的方向，而且翻轉的地方正好在 INT4 會崩的模型（Qwen2.5 族）上。
5. **三個會打到 9/24 第一篇（故事 1＋2＋6）的新威脅，寫之前必須處理**（§1）：
   - **vLLM 官方部落格（2026-01-08）**已經寫過：v0.12.0 把 KV 改成跨層連續，實體 block 放大 2·num_layers 倍，offload 吞吐因此提升約一個數量級（H100、Llama-3.1-8B：TTFT 最多降 4 倍、吞吐最多 5 倍）【抽查，S20】。這是故事 2 最接近的前作。
   - **UChicago 的 KV 重用經濟學**（arXiv 2503.14647）已經有「節省上限由被重用 context 的 prefill 佔比決定」這種 Amdahl 式的式子【擷取，S28】。故事 6 的「prefill 佔比」那一項不是新的。
   - **Rethinking KV Compression（MLSys'25）**：在 paged 框架裡，4-bit KV 的 decode 吞吐只有 FP16 的 0.88–1.02 倍，而且壓縮會讓輸出分布偏長【擷取，S17】。故事 1 必須嚴格限定在 prefill 口徑，端到端要另外量。
6. **一個幾小時就能做、可能改寫故事 2 的檢查**：照上面那篇部落格，CUDA 上從 v0.12.0 起每個 block 已經是跨層連續的大塊。可是我們在 MI300X 上用 vLLM **v0.28.0**（跟平台 A 同版，`RUNLOG_MI300X.md` 發現 5）量到，Qwen-7B-1M 與 Llama-8B 仍是每層一筆 32／64 KiB 的描述符，只有 Qwen3-30B-A3B 拿到 1.5 MiB 的整塊【B，`EXPERIMENTS_20260919.md` §2.3】。**要先查清楚是哪個條件讓跨層佈局沒生效**（ROCm？attention backend 是 TRITON_ATTN？模型設定？）：
   - 如果是 ROCm 特有，故事 2 就變成「上游已經在 CUDA 修掉的問題，在 ROCm 上仍在，量級是 17 倍」。定位更清楚，也比較不怕被說「vLLM 早就知道」。
   - 如果只是某個設定沒開，修法可能只是一個旗標，故事 2 的貢獻要縮小。
   - 兩平台同版，正好可以對照【判讀】：如果 3090（CUDA）上的佈局已經跨層，那 3090 的 CPU 階只拿到 PCIe 約 1/7（9/24 κ 地圖 §4.3）的原因就不是描述符粒度，要另外查。
7. **覆蓋缺口**：角度 B（P/D、RDMA、CXL、GDS）與 D（經濟、碳、隔離）**沒有任何一條 claim 進入三票驗證**，原因是驗證預算只挑了 25/140 條，不是被否決。所以方向 16、17、19–22 的證據等級是【抽查】或【擷取】。要不要再跑一輪專門補這兩個角度，見 §5。

---

## 1. 對 9/24 七個故事的影響：新佐證與新威脅

| 故事 | 新佐證 | 新威脅 | 要做的事 |
|---|---|---|---|
| **1** INT4 取代放置 | Anthropic 文件保證「命中與未命中的輸出 identical」→ 在這種產品契約下，有損精度只能是全域開關，跟故事 1 的「bit 是全域開關」一致【擷取，S25】。DASC 把 state 量化成 INT8 並疊上 channel 選擇，得到 8.11 倍的 checkpoint 容量，品質接近 dense → 「量化 vs 放置」可以移植到 state【擷取，S05.5】 | ① paged 框架下 4-bit KV 的 decode 吞吐 0.88–1.02 倍，壓縮讓輸出偏長（但作者用來當標題的「>20% 樣本變長 1.5 倍」跟溫度擾動分不開）【擷取，S17】。② agent 閉迴路：品質差 → 步數多 → context 長（方向 12）【3票，S09.3】。③ Lynx：Qwen3-32B 上直接 INT4 讓 MMLU 掉 8.7%【擷取，S23.5】 | 數字只寫 prefill 口徑；端到端（含 decode 與輸出長度變化）另外量；加一個閉迴路 agent 檢驗（方向 12） |
| **2** 傳輸由程式決定 | ① vLLM #55434：GB200、NIXL P/D，每次 TP-rank 傳輸約 4.9 GB 要發 91k–120k 個描述符；cuda_ipc 每筆 4.067 µs、IB 0.204 µs、RoCE 0.352 µs（約差 20 倍）；NVLink peer copy 在 45 KiB 粒度只剩 2.4 GB/s，連續搬是 778.6 GB/s【抽查，S19】。擷取者算出 4 個點符合 t ≈ n·19 µs + bytes/779 GB/s，跟我們的 t0 ≈ 13 µs 同形【擷取者算術，S19.4】。② llm-d：後端的 post 開銷在 400G IB 上看得到（Mooncake 約 42 vs UCX/UCCL 約 49.5 GB/s），在 100G RoCE 上看不到（三者都 12.0–12.2 GB/s）【擷取，S21】→ 正是 X = S/(t0 + S/B) 的形狀。③ llm-d：不支援 HMA 的 offload connector 會**靜默關掉** HMA；改成支援後讀回快 1.8–1.9 倍【抽查，S04】。④ SpeCache：非連續 gather 比連續傳輸慢約 5 倍【擷取，S18.2】 | **vLLM 官方部落格（2026-01-08）**：跨層連續佈局讓傳輸單位放大 2·num_layers 倍，吞吐提升約一個數量級；微基準顯示 DMA 只在大 block 才快。沒有每筆固定成本的模型，也沒有 AMD 的數字【抽查，S20】 | ① 做 §0.6 的佈局檢查。② 故事 2 的新穎性改寫成：跨路徑（ROCm 的 PCIe DMA、NVLink IPC、RDMA）的同一條「每筆固定成本」定律，加上方向 16 的重用與碎片化耦合。③ 正面引用這篇部落格 |
| **3** ε 是模型 × 任務 | ε 隨「動作」而變：跨 adapter 借 KV、近似重建 state、跨 fine-tune 借 KV 都一樣【3票，F8】。平均分數會藏住逐樣本的失敗，最脆弱的是摘要與 QA【擷取，S17.4】 | — | ε 協定擴成 ε(模型, 任務, 動作)；報尾端而不只報平均 |
| **4** 命中不該只是前綴 | ① vLLM #39321：A1 的 KV 明明在快取裡，卻因為 hash 鏈裡卡著 T1 只能重算【抽查，S15】；SGLang PR #23315 連 answer token 也一起丟，理由是剝掉 thinking 後有 RoPE 位置偏移【抽查】→ 前綴語意加 RoPE 逼出重算的生產案例。② SGLang RFC #40865：hybrid 模型在 state checkpoint 被逐出時會「退回」較早的 checkpoint，並對整段 [C,H) 重跑 prefill，即使那段 KV 還在；state 池 16 格時 93% 的請求退回【抽查，S02】→ 在 hybrid 模型上，「前面載、後面算」是結構性的 | MEPIC、MiniPIC 已經做到可共享的非前綴命中（K 存成未旋轉的形式、chunk 對齊 block）【3票，S11.1；MiniPIC 未讀】。CacheTune（arXiv 2605.24022）宣稱用 r0 = 1/(1+κ) 決定「載多少、算多少」，**但兩條相關 claim 都被否決**（0-3、1-2），要讀原文才知道是否屬實 | 故事 4 的設計要存 canonical（未旋轉）chunk（方向 13）；讀 CacheTune 原文 |
| **5** SSD 卡的是寫 | vLLM `v0.28.0` 預設會卸載 decode 產生的 block【原始碼】→ 死掉的 thinking block 也在消耗 SSD 寫入（方向 18） | GreenCache 把 SSD 壽命當固定攤提（3–7 年），沒有建模寫入耐久【擷取，S26.3】→ 這一格仍空著 | 併入方向 18 |
| **6** 事前判準 | — | ① 2503.14647：雲端價格下的 KV 重用成本模型，節省上限是 Amdahl 式的 prefill 佔比【擷取，S28；摘要沒有提「每小時一次」門檻】。② keepalive 論文：客戶端的「租 vs 重 prefill」閉式解 I_max = τ(w/r − 1)（Anthropic 約 46 分鐘）【抽查，S27】。③ GreenCache：快取的淨效益會隨電網碳強度變號【抽查＋擷取，S26】 | 判準的差異要放在：每條路徑的 κ、精度與 ε、兩平台實測。這三篇都要引 |
| **7** 注意力預測不了跨請求重用 | 替代訊號：token 的角色在寫入時就知道（方向 18）。SpeCache：在單一請求內，H2O 式累積注意力也預測不準下一步的 top-k【擷取，S18.5】 | — | 在第一篇裡把「角色」當成反例旁邊的正例 |

---

## 2. 新方向一覽

> 「可行性」指 1×MI300X＋7×3090、約 1 個月。「建議」是我的判讀。

| # | 方向 | 一句話 | 證據 | 新穎性風險 | 可行性 | 建議 |
|---|---|---|---|---|---|---|
| 8 | hybrid 模型的 state 與 KV 一起分層 | GDN／KDA 的 checkpoint 就地覆寫、只在邊界有效、單份數十 MB；沒人把它和 KV 一起放進 HBM／DRAM／SSD＋精度＋重算 | 【3票】 | 中偏高（正在變擠） | 中 | ⭐ 機制型的載體之一 |
| 9 | 兩個池互相牽制 | state 池滿時 host KV 命中歸零，即使 KV 池還有 35% 空間 | 【3票】（單一 PR、8 次重放） | 單獨成篇低 | 高 | 併入 8 當動機量測 |
| 10 | state 的有損重建是一個帶 ε 的動作 | 近似重建保留 92.8–98.9%；直接歸零則崩 | 【3票】（單一來源） | 機制高，放置中 | 高 | 併入 8 |
| 11 | 借用別的 adapter／fine-tune 的 KV | 借 KV 是一個帶 ε 的新動作，要跟 BF16/FP8/INT4、DROP 放在同一個動作空間比 | 【3票】 | 共享機制高；長上下文與分層中 | 高 | 第二條線 |
| 12 | 品質損失拉長 agent 軌跡 | INT4 省下的容量可能被變長的軌跡吃回去 | 【3票】（原文用 "tend to"） | 中 | 中 | 當故事 1 的必要檢驗 |
| 13 | 非前綴命中要 canonical 才省得到容量 | 逐請求修補的 KV 在不同請求間分歧，不能 page 共享 | 【3票】（2-1） | 中偏高 | 中 | 當故事 4 的設計約束 |
| 14 | 同一 block 多精度副本 | INT4 在 HBM 做 draft、完整 KV 在 DRAM 做 verify，greedy 下輸出相同 | 【3票】 | 機制高 | 中 | 主要是威脅；延伸故事 6 的判準 |
| 15 | bit-plane 分層 | INT8 ＝ INT4 anchor ＋ INT4 residual：anchor 常駐、residual 下放，不重複存位元組 | 【抽查】＋【擷取】 | 中偏高 | 中 | ⭐ 機制型首選 |
| 16 | 命中越多、搬得越慢 | prefix 重用讓實體佈局變碎，描述符變多，每 byte 的傳輸變慢 | 【抽查】（單一 issue） | 中（上游已提修法） | 高 | ⭐ 第一篇可加一節 |
| 17 | P/D 與遠端階的「拉還是算」 | 每條傳輸路徑都有自己的 t0；拉 KV 還是本地重算應逐請求決定，引擎現在只把重算當失敗 fallback | 【抽查】＋【擷取】 | 中 | 低（要兩節點） | 暫緩 |
| 18 | reasoning 的死 block | thinking token 被快取、被卸載，但剝掉之後永遠不會命中；角色是准入訊號 | 【抽查】＋【原始碼】 | 中低 | 高（3090、數天） | ⭐ 最便宜 |
| 19 | decode 生的 KV 才是 reasoning 時代分層的主體 | 長輸出下 KV 幾乎全是 decode 生的，壓力來自請求內增長而不是跨請求重用 | 【抽查】（摘要）＋【擷取】 | 中 | 中 | 當第一篇的適用範圍 |
| 20 | TTL 契約與 keepalive | 生產快取賣的是最短壽命契約；keepalive「製造 recency」，LRU 沒東西可排 | 【抽查】＋【擷取】 | 中 | 高（純模擬） | 便宜，可做 |
| 21 | 隔離範圍的代價與「哪一階」側通道 | per-user／per-org／global 的命中率代價沒人量過；多層命中的延遲洩漏的是哪一階 | 【抽查】＋【擷取】 | 中（要查新） | 高／中 | 可做 |
| 22 | 碳與美元 | 快取的淨效益在碳單位下會變號；INT4 讓每個快取 token 少佔 3.77 倍位元組 | 【抽查】（摘要）＋【擷取】 | 中偏高 | 中（只能算術） | 附錄等級 |

---

## 3. 逐一展開

### 總綱：從「同質的 KV block」到「異質的推論狀態物件」【F0，3票綜合】

2026 年的服務系統已經同時快取四類性質不同的物件：
- hybrid 模型的 recurrent-state checkpoint：單一大物件，只在邊界有效；
- 跨 adapter 或 fine-tune 借用的 KV：帶 ε；
- position-free 的 canonical chunk KV：可以跨位置共享；
- 同一 block 的多精度副本：有損版放 HBM、無損版放 host。

在本輪讀到的論文裡，沒有一篇跨物件類型做 HBM/DRAM/SSD＋精度＋重算的聯合放置。**風險**：審稿人很容易把它看成「把幾個機制拼在一起」。要站得住，得有一個統一的成本模型，也就是把故事 6 的「五個常數＋MRC」判準推廣成按（硬體, 路徑, 物件類型）各算一次。

---

### 方向 8：hybrid 模型的 recurrent-state checkpoint 與 KV 一起分層【3票】

**一句話**：Gated DeltaNet／KDA 類 hybrid 模型（Qwen3-Next、Kimi-Linear、Qwen3.5／3.6、OLMo-Hybrid）的 state checkpoint 是一種新快取物件。把它跟 full-attention KV 一起放進 HBM/DRAM/SSD＋精度＋重算的動作空間，是本輪最清楚的空白。

**推翻的假設**：
- 快取單位是 per-token 的 KV block，各自獨立逐出。
- κ 和五常數判準只需要 KV 的常數。
- DROP 的重算成本隨位置增加。對 recurrent 層來說，重算成本只取決於從最近的 checkpoint 要重播幾個 token，跟絕對位置無關【判讀】。

**證據**：
- **物件性質**（DASC，arXiv 2608.30386）【3票，S05.1–2】：
  - recurrent state 會就地覆寫，所以 SGLang 每隔固定 token 數就要存一份涵蓋所有 linear-attention 層的完整 checkpoint，跟 KV block 並存。
  - 沒快取到的 checkpoint 只能靠 prefill 重算。
  - DASC 的實驗全部在 HBM 內，全文搜 offload、CPU、SSD、DRAM 都是 0 次命中。
- **命中成本**（Sparse Prefix Caching，arXiv 2605.05219）【3票，S01.1–2】：
  - checkpoint 放置有精確的 O(NM) DP。
  - 命中成本分三項：checkpoint 從 CPU 載入、recurrent 層重播 suffix、attention 層載入 KV。
  - 只放 CPU DRAM；DP 目標只算重算 token 數；硬體是一張 RTX 2080 Super。
- **大小**（綜合者的【算術】，非實測）：一份 checkpoint 約 77.3 MB（Qwen3-Next-80B）與 43.4 MB（Kimi-Linear-48B），約等於 3,144 與 5,384 個 token 的 KV。SGLang RFC #40865 在另一個 hybrid 模型上報每個 slot 78 MB【抽查，S02】，同一量級。
- **狀態重用不保證位元一致**：RFC #40865 在 32 token 的 greedy 續寫測試中，重用 state 只有 14/20 exact match；只在 seed 位置跟參考的即時位置不同時才分歧【抽查，S02】。→ 我們在 `CLAUDE.md` §0 寫的「CPU/SSD 無損」在 hybrid 模型上要加註：**位元組無損 ≠ 輸出相同**。
- **引擎現況**：
  - vLLM `v0.28.0` 的 offloading scheduler 已經處理 `MambaSpec`：把它當成只有 1 個 chunk 的 sliding window，而且 `mamba_cache_mode == "align"` 時有 copy-on-write 處理【原始碼】。
  - vLLM 部落格（2026-01）說 hybrid 模型「currently not optimized for the offloading connector」【抽查，S20】；llm-d（2026-06）說 checkpoint 有被維護與卸載，但沒給任何數字【抽查，S04】。

**最接近前作**：Marconi（已讀）、Sparse Prefix Caching、DASC、Tail-Replay、LinearKV（2608.11231）、HYPIC（2607.01299）、ProphetKV（2602.02579）、HeadWiseKV（2609.02029，只讀摘要）、ReplaySSM、Kimi K3 的 KDA state 持久化。

**新穎性風險**：中偏高。空白只在這幾篇裡確認過；驗證者明講「the gap exists in this paper, not in the field」。

**跟既有故事的連結**：
- 故事 1 可以直接移植：把 FP32 checkpoint 降成 BF16／FP8，是不是也會吃掉放置的空間？
- 故事 2 預測：state 是少量大物件、KV 是大量小描述符，兩者的有效頻寬和 κ 會不同【判讀，待量】。
- **審稿人一定會問**：MLPerf v6.1 Agentic 已經採用 Kimi K3（KDA）與 Qwen3.6-35B-A3B（GDN hybrid）【9/24 `workloads_eval.md` §4】。所以就算第一篇不做機制，也要有一段說明結論在 hybrid 模型上還成不成立。

**可行性**：中，4 週。
- 第 1 週：確認 vLLM 0.28 對 Qwen3.5-4B、OLMo-Hybrid-7B 的 prefix caching 與 offload 是否可用；卸載的有沒有包含 state；state 不在時 host 命中會不會歸零（重現方向 9）。
- 第 2–3 週：量 state 的 κ，以及「checkpoint 間隔 × 階層」的成本曲面。
- 第 4 週：在 Sparse Prefix Caching 的 DP 裡加上各階的載入成本。
- ROCm 對這些模型的支援未查證，是主要風險。

### 方向 9：兩個池互相牽制【3票，S03.2】

**一句話**：hybrid 模型的 host tier 其實是兩個池，一個放 KV、一個放 state checkpoint。容量切分與逐出必須聯合決定。

**證據**：SGLang PR #39436（open、未審查、CI 失敗；單一設定、8 次重放）。
- GLM-5.3-Flash TP4、4× RTX PRO 6000 Blackwell；預設切分只給 565 個 host state slot，對應 2,680,896 token 的 KV host tier。
- 灌入 200 個 8,448-token 的 prompt（每個 3 個 checkpoint，共 600 個 > 565）後重放 8 個已逐出的 prompt：host 命中 0/8、TTFT 0.82 s（等於 cold），當時 KV 池還有 35% 是空的。
- 把 state 池加到 734 個 slot：命中 8/8、TTFT 0.15 s。
- 還原停在 8,192 token（跟 chunk 對齊的最深 checkpoint），剩下 256 token 要重算。
- ⚠️ 把這點推廣成「dead memory／單池規劃會系統性失效」的 claim 以 1-2 被否決。目前只能說**有一個重現**。

**用法**：當方向 8 的動機量測，或「兩種物件的聯合 MRC／聯合容量切分」理論小節的實證。3090 上 1–2 週可重現。這支 PR 的 AMD CI 目前失敗，不要先在 MI300X 上走。

### 方向 10：state 的有損重建是一個帶 ε 的新動作【3票，S07.2】

**一句話**：hybrid 模型的 recurrent state 可以用一個 replay 預算 r 做近似重建，帶有自己的 ε(模型, 任務, r)。但直接 DROP state（歸零）的代價很大。

**證據**：Tail-Replay（arXiv 2608.30310，標註 NeurIPS'26 ML for Systems workshop）。
- r = 5% 時：LongBench 保留 full prefill 的 92.8–98.9%，RULER 保留 93.1–99.9%。
- 只重用 FA KV、state 歸零時：RULER 平均掉到 0.215–0.415（full prefill 是 0.812–0.971）。
- OLMo-Hybrid-7B 最差，跟故事 3 一致。
- 限制：每個設定只跑一次；RULER 只到 16K；沒整合進 vLLM／SGLang。
- ⚠️ 描述 Tail-Replay 機制細節的那條 claim 以 0-3 被否決，引用機制前要回原文重讀。
- ⚠️ **反向證據**：SGLang RFC #40865 在一個 48 GDN＋16 FA 的交錯架構上論證 tail-replay「saving zero」。理由是 FA 跳過 [C,H) 之後，後面的 GDN 層沒有 hidden state 可用；ring buffer 每個請求要約 25 GB，只換到約 13 s 的重 prefill。作者因此撤回 replay 路徑，改主張 checkpoint 的保留政策才是真槓桿【抽查，S02】。兩者的機制與架構不同，**哪一種架構下 replay 划算**本身是個問題。

**用法**：動作空間多一個「近似重建（預算 r）」，直接跟「把精確 checkpoint 存在 DRAM／SSD」比。

### 方向 11：借用別人的 KV（跨 adapter／跨 fine-tune）【3票，S09.3、S10.2】

**一句話**：在 multi-LoRA agent 之間、或同一底座不同 fine-tune 的模型之間，「借用別人的 KV」是一個帶 ε 的新動作，應該跟 BF16/FP8/INT4、DROP 放在同一個動作空間裡比。

**證據**：
- LRAgent（ICML'26 poster）：完整共享、完全不重算時，平均準確率最多掉 5.28 分；只共享 base cache 最多掉 0.67；DroidSpeak 式部分層重算最多掉 2.63。context 只有 1.0–1.5K token。
- DroidSpeak（NSDI'26）：8 組同架構不同 fine-tune 的模型對；prefill 延遲降 1.7–3.1 倍（離線量的，不是負載下的 TTFT）；「可忽略」的定義是 profiling 時品質下降 ≤ 5%。
- ⚠️ 被否決、不可引用：DroidSpeak「平均只有 11% 關鍵層」（0-3）、LRAgent 的 K/V 可共享性不對稱（0-3）、LRAgent 的 base＋LR 分解細節（1-2）。
- **vLLM 的現況**【原始碼】：`v0.28.0` 的 `_gen_lora_extra_hash_keys` 把 LoRA 名稱放進 block hash。所以**跨 adapter 的 prefix 命中預設就是 0**；要做這條，得在 hash 層開一個「允許借用」的路徑。

**還空著的**：
- 32K–128K 下借用 KV 的 ε。
- 借用 KV 的 ε 跟 INT4 的 ε 疊在一起，是相加還是互相放大？
- 把 base KV（大、共享、重用多）跟各 adapter 的差量（小、私有）當成不同物件來分層。

**可行性**：高，2–3 週。用 HF transformers 以 adapter A 做 prefill、adapter B 做 decode 來模擬，不用改引擎。

### 方向 12：品質損失拉長 agent 軌跡【3票，S09.3；原文用 hedge】

**一句話**：在 agent 迴圈裡，準確率較低的快取方法會讓 agent 多走幾步、累積更長的 context。這對故事 1「全域 INT4＋LRU ≈ oracle」是還沒檢驗過的威脅，因為 INT4 省下的容量可能被變長的軌跡吃回去。

**證據**：
- LRAgent 原文：「lower-accuracy methods tend to take more steps and accumulate longer contexts」。
- LRAgent 附錄 Table 17（LLaMA）：平均序列長度 Non-Shared 1,093 vs FullShared 1,514 token。
- **反例**（驗證者整理）：
  - Ministral 上 FullShared 的 E2E 延遲反而比較低。
  - BaseLRShared 與 FullShared 的準確率差很多，序列長度卻幾乎一樣。
  - 作者沒量 KV 佔用；context 只有 1.0–1.5K。
- 相關：Rethinking KV Compression 在單一請求層級也看到壓縮讓輸出變長【擷取，S17.1】。

**推翻的假設**：我們的 oracle 與 headroom 是在固定 trace 上算的，隱含「軌跡長度不隨放置決策改變」。

**可行性**：中，約 2 週。
- 做法：3090、Llama-3.1-8B，分別用 BF16、FP8、INT4 KV 跑 AutoAct 式的 HotpotQA 三角色 agent。量步數、總 token、KV 峰值與成功率，多跑幾個 seed。
- **不要用 Qwen2.5**：它在 FP8／INT4 下會直接崩（故事 3），量到的是整體失敗，不是「微小 ε 拉長軌跡」。

### 方向 13：非前綴命中要 canonical 才省得到容量【3票（2-1），S11.1】

**一句話**：CacheBlend、EPIC 這類 position-independent caching 對每個請求各自做重算與位置調整，結果同一個 chunk 的 KV 在不同請求間分歧，不能在 paged KV cache 裡共享。故事 4 的區段命中必須遵守一條設計約束：**修補結果不能寫進共享頁**。

**證據**：MEPIC（Huawei，preprint）§2.1.3 原文：「the resulting KV representations diverge across requests and cannot be page-aligned or shared in the paged KV cache」。
- 重複只發生在 HBM；CPU／disk 階每個 chunk 只有一份。
- MEPIC 與 IBM 的 MiniPIC（2606.13126，未讀）已經用「K 存未旋轉形式、到 kernel 內才套 RoPE、chunk 對齊 block」做到共享。
- ⚠️ MEPIC 的 NoPE 機制細節與評測範圍兩條 claim 都以 0-3 被否決。KVShareArena（2609.10266）自稱非前綴重用的 benchmark，但三條 claim 全部 0-3 否決，不能引用。

**還空著的**：32K 以上、NVIDIA／AMD 硬體上，canonical chunk 在 HBM/DRAM/SSD 之間的放置與重建成本；把故事 2 的每筆固定成本套進來。

**可行性**：中。先用我們的 trace 以算術／模擬求出「canonical」與「逐請求修補」兩種做法的 HBM 佔用差，放進故事 4。

### 方向 14：同一 block 的多精度副本（主要是威脅）【3票，S13.1】

**一句話**：VeriCache 把壓縮過的 KV 放在 HBM 做 draft、完整 KV 放在 CPU DRAM（或 storage）做 verify，greedy 下輸出跟完整 KV 相同。這直接推翻 `main.tex`「每個 block 只有一個 ℓ_i、一個 π_i」的前提。

**證據**：VeriCache（arXiv 2605.17613，UChicago／Tensormesh／Samsung／MSR）：建在 vLLM＋LMCache 上；每次 verify 都從 CPU 重載完整 KV；sampling 時 KL < 0.01 nats；全文沒有 SSD、NVMe、AMD。更早的：QuantSpec（ICML'25）、TriForce、MagicDec、SparseSpec。

**審稿人可以直接問**：「為什麼不同時保留 INT4@HBM 和 BF16@DRAM？」

**我們還能做的【判讀，待量】**：VeriCache 的收益取決於兩件事。
- draft 的接受率：這就是故事 3 的 ε。Qwen2.5 在 INT4 崩潰，表示它的 draft 很差。
- 每次 verify 重載完整 KV 的成本：這就是故事 2，每筆 13 µs、跨模型 17 倍。

合起來的預測是：**最需要無損保證的模型，正好是 draft 最差、verify 最貴的模型**。這可以寫成「何時值得用多副本做無損分層」的事前判準，是故事 6 的延伸。方向 15 是它的「不重複存」版本。

---

### 方向 15：bit-plane 分層——MSB 常駐 HBM、LSB 下放【抽查＋擷取】

**一句話**：精度階不必是互斥的整份副本。INT8 可以拆成 INT4 anchor（高 4 bit）加 INT4 residual（低 4 bit）：anchor 常駐 HBM、residual 放 CPU／SSD，decode 先用 anchor 投機生成，residual 到了再驗證。這樣位元組不重複（總共 8 bit，對照 VeriCache 的 INT4＋BF16 兩份），而品質不允許 INT4 的模型也能拿回無損。

**推翻的假設**：
- `main.tex` 的 s_t：每個 block 一個 π_i，位元組不可分。
- 故事 1 隱含 INT4 的品質損失是永久的。在 draft／verify 下，ε 變成延遲成本（接受率），於是「量化」與「放置」在 residual 這一段又變成互補。

**證據**：
- **Lynx**（arXiv 2607.01831，原投 SIGCOMM'26）【抽查，S23】：
  - 摘要：「partitioning the KV cache into a high-priority Anchor stream carrying the most significant bits and a low-priority Residual stream」；anchor 一到就投機 decode，之後驗證保證等同高精度。
  - TTFT 接近 INT4，比 INT8 最多快 1.43 倍，準確度比 SOTA 最多高 5.1%。
  - 【擷取，S23.2–5】只評估 P/D 跨伺服器傳輸，頻寬用 rate-limiter 人為限制在 10–50 Gbps；頻寬越高收益越小（10 Gbps 領先 0.86 s，50 Gbps 只剩 0.18 s）；Qwen3-32B 上直接 INT4 讓 MMLU 掉 8.7%，Lynx-INT4 掉 1.7%。
- **QuantSpec**（arXiv 2502.10424，ICML'25）【擷取，S14】：hierarchical quantized KV，INT8 ＝ 高 4 bit＋低 4 bit；draft 只讀高 4 bit，target 讀兩個 plane。全部在 HBM 內；128K 下 draft 接受率 94.31%、端到端 2.49 倍；只測了 Llama-2 系。
- **SpeCache**（arXiv 2503.16163）【抽查，S18】：VRAM 留 1–2 bit 副本，完整 16-bit 放 CPU，每步依低位元副本取回 top-k。
- **VeriCache**（方向 14）【3票】。

**我們的切入【判讀】**：
1. residual 放到 CPU／SSD，讀取只發生在 verify、可以批次。這正好可以攤提故事 2 的每筆固定成本。反過來，Lynx 的雙 stream 會讓描述符加倍；擷取者推論，在每筆 13 µs 的路徑上收益可能被吃掉【擷取，S23.2】。所以**故事 2 可以預測它在哪個平台成立**。
2. 故事 3 預測接受率：Qwen2.5 在 INT4 崩潰 → anchor 接受率可能極低 → 這就是「何時值得」的邊界。
3. 前綴重用：anchor 與 residual 可以各自有不同的逐出順序（例如 residual 先逐出，命中時退化成 INT4＋驗證失敗重算）。這在四篇裡都沒有。

**最接近前作**：QuantSpec（巢狀精度、HBM 內）、Lynx（bit-plane、網路傳輸）、VeriCache（跨層多副本）、SpeCache（低位元索引＋CPU 取回）。

**新穎性風險**：中偏高。「bit-plane 跨 HBM/CPU/SSD＋前綴重用＋每筆固定成本」這個組合在本輪沒看到，但要先 novelty-check「bit-plane KV offload」「progressive KV」。

**可行性**：中，2–3 週。
- 第 1 週：用 HF transformers 在 3090 上量 anchor 接受率 α(模型, 任務)，模型取 Llama-3.1-8B、Qwen2.5-7B、Qwen3-8B，不改引擎。
- 第 2 週：解析模型，由 α、residual 位元組數、每筆固定成本推出有效 decode 延遲。
- 第 3 週：用兩平台 M2 的實測常數預測成立區間。
- 引擎整合一個月內做不到。

### 方向 16：命中越多、搬得越慢——重用與碎片化的耦合【抽查，S19】

**一句話**：prefix 重用會把命中的 block 散在 pool 各處，實體佈局越碎，每次搬運要發的描述符就越多。在每筆有固定成本的路徑上，**命中率越高、每 byte 的傳輸越慢**。所以放置的成本模型必須有「每段不連續 run」這一項。

**推翻的假設**：傳輸成本 ＝ 位元組 ÷ 頻寬（或 κ 只看 token 數）；命中率與傳輸效率彼此獨立。

**證據**：vLLM issue #55434（2026-09-05，open，沒有 maintainer 回覆）【抽查】。
- 環境：GB200、vLLM 0.28.0、NIXL P/D、GLM-5.3（DeepSeek-V3.2 架構：MLA＋DSA indexer）。
- agentic 負載下，每次 TP-rank 傳輸約 4.9 GB 要發 91k–120k 個描述符【抽查】。走 cuda_ipc 時 56.9% 的傳輸時間花在 post 描述符【擷取，原文引句見 S19.1】。負載的 prefix hit 是 88–92%【擷取】。
- 每筆成本：cuda_ipc 4.067 µs、rc_x IB 0.204 µs、RoCE 0.352 µs，約差 20 倍。名義上較快的 MNNVL，端到端反而比 4-rail IB 慢。
- **關鍵句**：prompt 從 10.3k 放大到 41.1k token，payload 變 4 倍，描述符**恆為 101 個**，也就是每個註冊 region 一筆。長度本身不會增加描述符，碎片化才會。
- 回報者提的修法（未量測）：用 vLLM 既有的 `_register_packed_kv_cache` 做 packed 單一 region 註冊（約 112k → 1.1k）、讓同一 sequence 的 block 保持連續、批次化 cuda_ipc 的提交。

**我們的切入**：
- 先確認我們的 `OffloadingConnector` 路徑**會不會合併相鄰 block**（`EXPERIMENTS_20260919.md` §2.3 提到 `_canonical_copy_plans` 能跨層合併，但只有 Qwen3-30B-A3B 拿到）。
  - 如果會合併，耦合就存在：用 MI300X 已有的 connector 計數器加一個維度（描述符數／請求 vs 命中率 vs 連續 run 數），1–2 天。
  - 如果不會，它是「還沒合併所以沒有耦合」。修掉故事 2 之後耦合才會出現，這要寫進修法的評估：合併後的收益會隨命中率下降。
- 若成立，故事 2 從「跨模型 17 倍」擴成「跨路徑同一條定律＋重用耦合」。放置策略也多一個目標：命中的 block 要連續（promote 時順便壓實、按 session 連續配置）。

**新穎性風險**：中。上游可能幾週內修掉 region 放大，所以貢獻必須是「耦合的刻畫＋成本模型＋壓實策略」，不是發現描述符很多。

**可行性**：高，1–2 週，兩平台都能做，不需要多節點。

### 方向 17：P/D 與遠端階——每條路徑一個 t0，「拉 KV 還是本地重算」逐請求決定【抽查＋擷取】

**一句話**：P/D 分離與遠端 KV 階（RDMA、NVLink IPC、GPU-direct storage）每條路徑都有自己的每筆固定成本。所以 decode 端「拉 KV 還是本地重算」應該逐請求、逐路徑決定；現在的引擎只把重算當成失敗時的 fallback。

**證據**：
- llm-d networking（2026-06-23）【擷取，S21】：
  - vLLM `NixlConnector` 的 `kv_load_failure_policy` 預設 `fail`，可設 `recompute`。重算只是失敗 fallback，沒有依成本逐請求選擇。
  - 400G IB 上 UCCL／UCX 約 49.5 GB/s、Mooncake 約 42 GB/s；100G RoCE 上三者都 12.0–12.2 GB/s。
- vLLM #55434：同一台機器上各 transport 的每筆成本差約 20 倍【抽查】。
- AMD ROCm Infinity Context（2026-07-22）【擷取，S22】：
  - 在 HBM、DRAM 之下多一個 RDMA 網路儲存層，用 hipFile 直接進 HBM、不經 host DRAM。
  - MI300X 列為 Supported，Fall 2026 ROCm release 是 tech preview。
  - **全文沒有任何量測**：沒有 TTFT、頻寬、每筆開銷、命中率。它的「HBM3e 約 $50/GB」註明依據是「AMD internal calculations」。

**新穎性風險**：中。P/D 文獻很多（DistServe、Splitwise、Mooncake），「逐請求 pull vs recompute」要查新。

**可行性**：**低**。我們只有單節點；3090 是 GeForce，PCIe P2P 通常不可用；MLSteam 容器看不到 RDMA 儲存。**一個月內不建議做**。可以在故事 2 引用「同一條定律在 NIXL／RDMA 上也成立」，並把 AIC 列為平台 B 的 future work。

### 方向 18：reasoning 的死 block——thinking token 被快取、被卸載、永遠不會命中【抽查＋原始碼】

**一句話**：reasoning 模型的 thinking token 會被 prefix cache 快取、被 `OffloadingConnector` 卸載到 CPU/SSD。只要 client 在下一輪剝掉 thinking，它們（連同接在後面的 answer）就永遠不可能命中。token 的「角色」在寫入當下就知道，是比注意力好得多的准入訊號。

**推翻的假設**：
- LRU 的前提「最近寫入的就是熱的」。
- 故事 7 之後「沒有好訊號能預測跨請求重用」：協定層其實有。
- 寫進 SSD 的 block 都有機會被讀回。

**證據**：
- **vLLM #39321**（2026-04-08，Closed）【抽查，S15】：
  - APC 會雜湊並快取所有輸出 token，包括 thinking。
  - turn 2 以 [Q1, A1, Q2] 送來時 hash 鏈分岔，「A1 must be recomputed despite its KV being in the cache」。
  - 估計 QwQ-32B 每條死鏈約 1.3 GB、R1-Distill-70B 約 1.6 GB；50 個並行對話 → 65–80 GB 的死 KV。**這是紙上算術**，假設每輪約 5,000 thinking token。
- **SGLang PR #23315**（2026-04-21 merged）【抽查】：`--strip-thinking-cache`，預設關閉。answer token 也一起丟，理由是「RoPE position shift after stripping thinking」。
- **vLLM `v0.28.0` 的預設行為**【原始碼】：
  - `offloading/scheduler.py` 的 `_build_store_jobs` 以 `num_computed_tokens + num_scheduled_tokens`（請求結束時用 `num_tokens`）決定可卸載量。
  - `_calc_num_offloadable_tokens` 只有在 `offload_prompt_only` 時才截到 prompt 長度。
  - → **預設設定下，decode 產生的 block（含 thinking）會被卸載到 CPU／SSD**。
- 【擷取，未抽查】vLLM 另有兩個修正 PR（#39806、#41939）因閒置關閉、未合併；#39806 引用 SGLang 修正後的數字，QwQ-32B 命中率 11.1% → 17.1%。
- **反例**【擷取】：支援 interleaved thinking 的 agent 工作流會把 reasoning 帶進下一輪。所以可達性取決於 client 協定與 chat template，不能一律丟。

**我們的切入**：
- 量「寫進 CPU／SSD、卻永遠不會被讀回」的位元組比例。這是故事 5（寫入預算）與故事 2（傳輸路徑）一個具體、可量化的浪費來源。
- 提出「依協定／角色的准入」：prompt、tool output、answer、thinking 各一條規則。跟 SGLang 的全丟開關、vLLM 的 `offload_prompt_only` 比較。

**新穎性風險**：中低。旗標已經存在（SGLang 全丟、vLLM prompt-only），但「角色感知的分層准入＋對 SSD 寫入預算的量化」本輪沒看到。要查新：請求內的 thinking KV 壓縮（例如 R-KV 這類）屬於相鄰但不同的題目，要正面引用。

**可行性**：高，3090 上數天。
- vLLM 0.28.0，Qwen3-8B（AWQ）開 thinking，或 DeepSeek-R1-Distill-Llama-8B。
- 構造多輪 reasoning 對話，client 端照官方文件剝掉 thinking；量 CPU／SSD 寫入位元組、之後被讀回的比例、TTFT。
- **限制**：我們手上的 trace 都沒有 thinking 長度，多輪 reasoning 的真實 trace 要另外找。構造的 trace 要照 `CLAUDE.md` 規則 6 在載入時加單位斷言。

### 方向 19：decode 生的 KV 才是 reasoning 時代分層的主體【抽查（摘要）＋擷取】

**一句話**：長輸出 reasoning 模型的 KV 幾乎全是 decode 自己生出來的。分層的壓力來自「執行中請求的 KV 持續長大」，而不是跨請求重用。在這個區間，故事 1、6 的判準（prefill 佔比）會說放置沒用，但用 CPU 階撐大 batch 反而是最大的槓桿。

**證據**：SparseSpec（arXiv 2512.01278，Zhao … Kasikci, Han, Stoica）。
- 【抽查，摘要】長 CoT 讓推論瓶頸從 compute-bound 變成 memory-bound；吞吐最多 2.13 倍。
- 【擷取，S16】H100、Qwen3-8B、batch 128、輸出 8192：每步載入 KV 約 21 ms，佔端到端 70% 以上。
- 【擷取，S16】動態 KV 管理：超額接納、快 OOM 時以請求為單位 FIFO 卸載到 host、避免重算。消融中它貢獻最大（1.61 倍，依序累加的歸因）。
- 【擷取者算術，S16.4】它的 offload 頻寬論證是紙上算術，而且照它自己的數字差 10 倍（18.9 MB ÷ 10 ms ≈ 1.9 GB/s，文中寫 18 GB/s）。它沒報 chunk 大小、pinned memory、每筆固定成本。
- 相關【擷取】：
  - Rethinking KV Compression：壓縮讓輸出分布偏長（S17）。
  - Anthropic 文件：TTL 從請求**開始**算，生成時間也算在壽命裡。長輸出直接縮短 prefix 的可重用窗口；改 thinking 設定也會讓 cache 失效（S25.4）。

**我們的切入**：
- 在兩平台量「每步增量卸載」的真實成本。我們的 M2 只量過 prefill 的 block 搬運。
- 比較 INT4 KV 與 CPU 卸載在「撐大 batch」這個目標下是不是也互相替代，也就是故事 1 的 decode 版。

**新穎性風險**：中。SparseSpec、SpeCache、ShadowKV、InfiniGen 都在做 decode 端的卸載；「跨請求放置 vs 請求內增長」的區分可能只是論述貢獻。

**可行性**：中，2–3 週。3090 的 24 GB 對 8B 模型加長輸出 batch 很緊，MI300X 比較合適。

**建議**：至少當第一篇的「適用範圍」一節，說明故事 1、6 的結論只適用 prefill-heavy 的重用負載。

### 方向 20：TTL 契約與 keepalive——「製造出來的 recency」【抽查＋擷取】

**一句話**：生產 prompt cache 賣的是「最短壽命」契約（5 分鐘／1 小時 TTL、按次計價），不是容量驅動的 LRU。而 agent 客戶端的 keepalive 會製造 recency，讓 LRU 沒東西可排。放置問題因此變成兩件事：以最低的 HBM/DRAM/SSD 成本兌現 TTL 保證，以及在人為刷新下仍排得出優先序。

**推翻的假設**：快取是容量驅動、逐出由 LRU 或成本決定；客戶端的存取模式是外生的。

**證據**：
- **Anthropic 官方文件**（2026-09-27 擷取）【擷取，S25；原文引句在 evidence】：
  - KV 表示與 hash「held in memory only and are not stored at rest」。
  - TTL 是最短保證壽命，到期後 promptly（非立即）刪除；組織間隔離。
  - 寫入 5 分鐘 1.25 倍、1 小時 2 倍，命中 0.1 倍（部分新模型更低）；命中與未命中輸出 identical。
  - → 至少一家主要供應商的生產 prompt cache 不做 SSD 持久化【判讀】。
- **Keepalive**（arXiv 2607.19214，單一作者、5 頁）【抽查，S27】：
  - 四家 API 的保留只有幾分鐘。
  - keepalive 最省的間隔是略小於 TTL（Anthropic 約 4 分鐘，不是常見的 30 秒）。
  - 損益兩平 I_max = τ(w/r − 1)，Anthropic 約 46 分鐘。
  - 普遍採用後「gives LRU eviction nothing to rank」，是未定價的外部性；作者預測供應商會改按 token-hour 計費。
  - 伺服器端的對應工作是 CacheWise（arXiv 2606.16824，coding agent 的重用感知逐出），**未讀**。

**我們的切入**：
- 用我們的模擬器與 Mooncake／TraceLab trace 注入 keepalive 客戶端（比例 p、間隔 τ），量 LRU、`tier_fs`、oracle 的命中率與 headroom 怎麼變。
- 把「TTL 保證」寫成約束，求最小成本的階層放置：保證期內放最便宜但仍滿足延遲的階。

**新穎性風險**：中；要查 CacheWise 與 prompt cache TTL 的相關工作。

**可行性**：高，1–2 週，純模擬、不用 GPU。

### 方向 21：隔離範圍的命中率代價，與「哪一階」的時間側通道【抽查＋擷取】

**一句話**：prompt cache 要不要跨使用者共享是安全決策，但沒有人量過 per-user、per-org、global 三種隔離範圍的命中率與 TTFT 代價。而多層快取命中時的延遲依「在哪一階」而不同，洩漏的不只是有沒有命中。

**證據**：
- **Gu et al.**（ICML'25，arXiv 2502.07776）：摘要寫明在 7 家 API（含 OpenAI）偵測到跨使用者的全域共享【抽查，S24】。「2024 年 9–10 月稽核 17 家、8 家有 prompt caching」出自內文【擷取，S24.1】。
- 【擷取，S24】：
  - 主要緩解是只允許 per-user 快取，並「宣稱」保留多數效益，但沒有任何命中率或 trace 分析。
  - 時間分類在 prompt ≳ 1,000 token 時穩定。長上下文正是分層系統的目標區間。
  - 另一個緩解是刻意延遲命中的回應：使用者就看不到延遲收益，快取的價值只剩 GPU 時間。
- **vLLM 的現有機制**【原始碼】：`cache_salt` 只加在第一個 block 的 hash。這就是 vLLM 做租戶隔離的方式，可以直接拿來實作三種隔離範圍。

**我們的切入**：
- (a) 在有使用者 ID 的 trace 上模擬三種隔離範圍，量命中率、headroom、各階容量需求，回答「per-user 保留多少效益」。可用的 trace：Bidaw 的 56,573 位使用者、TraceLab 的 43 位、WildChat 的 `hashed_ip`。
- (b) 在 3090 上量 HBM/DRAM/SSD 三種命中的 TTFT 分布，看分不分得出是哪一階。這是**防禦性量測**，不對任何真實服務做攻擊。如果分得出來，「補齊延遲」的緩解會把放置目標從延遲換成成本。

**新穎性風險**：中。prompt cache 的時間側通道已有多篇，要 novelty-check；階層層級的洩漏與隔離代價的量化本輪沒看到【判讀】。

**可行性**：(a) 高，數天到 1 週；(b) 中。

### 方向 22：碳與美元——量化取代放置在成本單位下是否仍成立【抽查（摘要）＋擷取】

**一句話**：KV 快取的淨效益在碳單位下會變號（低碳電網上快取反而增碳）。INT4 讓每個被快取的 token 少佔 3.77 倍位元組，所以「量化取代放置」在成本單位下可能更強。但這條只能做算術，不能在平台 A 做能耗結論。

**證據**：
- **GreenCache**（arXiv 2505.23970 v3，現名 "Cache Your Prompt When It's Green"）【抽查，摘要】：FR 電網平均減碳約 15%、最高 25%，>90% 請求守住延遲。
- 【擷取，S26】：
  - 同一個 16 TB 快取在 MISO（485 gCO2e/kWh）減碳 7.5%，在 FR（33）反而增碳 16.5%。
  - SSD 佔伺服器 embodied carbon 的 76.6%。
  - 只調 SSD 容量一個維度；8K context；沒有精度階；壽命當固定攤提。
- **2503.14647**（UChicago，Li/Liu/Cheng/Du/Jiang）【抽查摘要＋擷取】：
  - 雲端價格下，KV 重用可以同時省延遲與成本；節省上限由被重用 context 的 prefill 佔比決定。
  - ⚠️ 摘要沒有「每小時重用一次以上」的門檻。擷取者另外算出，它的例子依腳註數字是 6.6 倍，不是文中的「7 倍以上」。

**新穎性風險**：中偏高。GreenCache 已經佔了 SSD 容量這一維，2503.14647 已經佔了美元模型。

**限制**：`CLAUDE.md` §3 與 §8 規定平台 A 不做能耗結論。MI300X 的 `amd-smi` 功耗是整卡的，也不分記憶體與計算。所以只能用文獻的 embodied carbon 參數做算術，全部標【算術】。

**建議**：附錄等級，或併進方向 20 的「成本目標」。

---

## 4. 怎麼接到 9/24 的計畫

### 4.1 第一篇（分析型：故事 1＋2＋6）要補的

- **必須正面引用並劃界**：
  - vLLM 官方部落格（故事 2）。
  - 2503.14647 與 keepalive 論文的「租 vs 重 prefill」（故事 6）。
  - Rethinking KV Compression（故事 1 的端到端口徑）。
  - Lynx／QuantSpec／VeriCache（故事 1 的「替代」不涵蓋 draft／verify 的情況）。
- **可以加的便宜量測**：
  - §0.6 的 ROCm 佈局檢查，幾小時。
  - 方向 16 的重用與碎片化耦合，MI300X 計數器，1–2 天。
  - 方向 18 的死 block 寫入比例，3090，數天。
- **適用範圍一節**：方向 8（hybrid 模型的 state 是另一種物件）、方向 19（reasoning 的 decode）、方向 20（TTL 契約）說明結論在哪些區間不適用。因為 MLPerf v6.1 Agentic 已經用 hybrid 模型，審稿人很可能會問。

### 4.2 如果轉機制型（第二篇）

| 選項 | 內容 | 優點 | 風險 |
|---|---|---|---|
| A | 方向 15：bit-plane 分層 | 跟故事 1–3 的關係最緊，能把負面結果轉成機制；α 可在 3090 先量 | 相鄰工作多（QuantSpec、Lynx、VeriCache、SpeCache） |
| B | 方向 8＋9＋10：hybrid 模型的狀態分層 | 證據最硬（3 票），3090 上的小模型可做 | 撞車風險最高，四週內就冒出四個相關工作 |
| C | 故事 4＋方向 13：canonical 區段命中 | 9/24 已列，推導完整 | 要改 attention kernel 的 RoPE 處理，一個月偏緊 |

### 4.3 先做什麼（插進 9/24 的週計畫，不取代）

| 什麼時候 | 做什麼 | 為什麼 |
|---|---|---|
| 本週（9/28–9/30） | ① §0.6：查 vLLM v0.28.0 在 ROCm 上，Qwen-7B／Llama-8B 為什麼仍是每層一筆描述符；同時查 3090（CUDA、同版）的實際佈局<br>② 對四組關鍵字做 `novelty-check`：「hybrid／GDN state offload tiering」「bit-plane／progressive KV offload」「reasoning thinking token KV admission」「prompt cache isolation hit-rate cost」<br>③ 讀原文：CacheTune（2605.24022）、MiniPIC（2606.13126）、CacheWise（2606.16824） | ① 可能改寫故事 2 的定位<br>② 決定方向 8、15、18、21 還剩多少空間<br>③ 三篇都是被否決或沒讀、但可能是最接近前作的 |
| 第 2 週 | 方向 18 在 3090 的量測（2–3 天）；方向 16 在 MI300X 的計數器分析（1–2 天） | 都便宜，而且能直接進第一篇 |
| 之後 | 依老師對 §5 Q1–Q2 的回答 | — |

---

## 5. 需要你回答的問題

1. **第一篇要不要涵蓋 hybrid 模型？** MLPerf v6.1 Agentic 已經用 Kimi K3（KDA）與 Qwen3.6-35B-A3B（GDN），審稿人很可能問「你的結論在 hybrid 模型上還成立嗎」。至少要有一段適用範圍的說明。
2. **機制型要選哪一個？** A（bit-plane）、B（hybrid state）、C（區段命中），見 §4.2。
3. **要不要再跑一輪 deep-research，專門補角度 B 與 D 的三票驗證？** 這次只驗了 25/140 條，B、D 一條都沒驗。這輪的成本是 110 個子 agent、約 4.9 小時。
4. **方向 21 涉及安全（時間側通道），你和老師能接受這類題目嗎？** 做法只限防禦性量測與 trace 模擬。
5. **要不要我把 §0.6 的檢查和方向 16、18 寫成實驗計畫**（照 `EXPERIMENT_PLAN.md` 的格式、附記錄協定）？

---

## 6. 限制

- **證據等級分布**：【3票】的 10 條全部來自角度 A、C 與 VeriCache。【抽查】只核對標題、作者、日期與關鍵句，不是全文驗證。其餘是【擷取】。
- **不是 cross-model review**：三票驗證的投票者都是 Claude subagent（`CLAUDE.md` §5 的已知落差）。本文的綜合與判讀是 self-review 等級。
- **「沒人做過」** 只相對於本輪的 28 個來源與 9/24 的 35 篇；動手前要先做 `novelty-check`。
- **時間敏感**：vLLM、SGLang 對 hybrid 模型的快取與卸載行為每一版都在變。PR #39436 仍 open；RFC #40865 已關閉。
- **數字都不是我們的**：它們來自 H100、GB200、RTX PRO 6000 Blackwell、Ascend 910B、A100、A6000、RTX 2080 Super，只能當假設。
- **被否決、不可引用的 claim**（完整列表在 `research_20260927/verified_findings.md` 最後一節）：
  - TokenDance（2604.03143）的跨 agent 冗餘與 Diff-Aware Storage（1-2）
  - CacheTune（2605.24022）的 κ 比例與「依儲存層調重算比」（0-3、1-2）
  - KVShareArena（2609.10266）的三條（0-3）
  - LRAgent 的分解細節（1-2）與 K/V 不對稱（0-3）
  - DroidSpeak「11% 關鍵層」（0-3）
  - MEPIC 的 NoPE 機制與評測範圍（0-3）
  - DASC 的 state／KV 敏感度不對稱（1-2）
  - Tail-Replay 的機制細節（0-3）
  - SGLang PR 的 dead-memory 解讀（1-2）

  否決代表那條 claim 的細節在原文找不到支持，不代表論文不存在。

## 附錄：檔案索引

| 內容 | 位置 |
|---|---|
| 逐條證據（S01–S28、140 條 claim、票數、抽查紀錄） | `docs/research_20260927/evidence_claims.md` |
| workflow 綜合結論原文（F0–F8）、caveats、被否決清單 | `docs/research_20260927/verified_findings.md` |
| 產生方式、抽查清單、讀過的原始碼位置 | `docs/research_20260927/README.md` |
| 9/24 的故事 1–7 | `docs/DIRECTIONS_20260924.md`、`docs/research_20260924/` |
