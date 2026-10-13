# P02｜CacheGen 的逐 chunk 精度選擇與重算退路（精讀＋程式碼）

> 一句話：CacheGen 在寫入時把每個 chunk 編成好幾個等級，讀取時逐 chunk 依量到的吞吐量挑一個；來得及就送文字讓 GPU 重算。重算是依序的、要靠前面 chunk 的 KV，沒有和 Cake 一樣兩頭並行。論文寫的機制和公開程式碼有好幾處不一樣；現行 LMCache 只剩一個固定 codec。

## 出處

- 作者：Yuhan Liu, Hanchen Li, Yihua Cheng, Siddhant Ray, Yuyang Huang, Qizheng Zhang, Kuntai Du, Jiayi Yao, Shan Lu, Ganesh Ananthanarayanan, Michael Maire, Henry Hoffmann, Ari Holtzman, Junchen Jiang（UChicago、Microsoft、Stanford）。
- 題目：CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving。
- Venue：ACM SIGCOMM 2024（PDF 頁首與 ACM 版權框印 SIGCOMM '24；DOI 10.1145/3651890.3672274）〔原文 p.1〕。arXiv 2310.07240，abs 頁 comment 欄寫 SIGCOMM'24。
- 讀的版本：**arXiv v6**（2024-07-19，19 頁；arXiv 上最新版）。2026-10-13 下載。
- 讀了哪裡：**p.1–12 正文全讀**；**p.17–19 附錄全讀**（附錄頁首註明「未經同儕審查」〔原文 p.17〕）；p.12–16 參考文獻只查了 [48]、[101]、[120]。
- 頁碼是 PDF 頁（等於 ACM 印刷頁）。
- 程式碼（〔程式碼〕標記的出處）：
  - 論文 artifact：`github.com/UChi-JCL/CacheGen`（摘要列出〔原文 p.1〕），commit `6bed34c`（2024-10-09，main 最新）。
  - 現行 LMCache：`github.com/LMCache/LMCache` 預設分支 dev，commit `240506e`（2026-10-13）。

## 機制

### A. 編碼（寫入時、離線、多個等級）

1. **三步**：算 delta → 分層量化 → 算術編碼〔原文 §5.2 / p.5–6〕。
2. **token group 與 anchor**：每 10 個連續 token 一組。組內第一個 token 是 anchor，獨立壓縮。其他 token 存「相對 anchor」的 delta，不是相鄰 token 的 delta，好讓壓縮／解壓可以平行〔原文 §5.2 / p.6 / Fig. 6〕。anchor 用 8-bit，因為它的精度會影響整組 delta〔原文 §5.2 / p.6〕。
3. **分層量化**：層切成前、中、後三等份，bin size 由淺到深變大（淺層較精）〔原文 §5.2 / p.6〕。預設 bin size 0.5、1、1.5〔原文 App. C.2 / p.19〕。用 vectorwise 量化（引 LLM.int8() [48]）〔原文 §5.2 / p.6〕。
4. **算術編碼（AC）**：依 channel×layer 分組取機率分布。
   - p.6 說：離線 profile，每個 channel-layer 一個分布，同一個 LLM 產生的所有 KV 共用〔原文 §5.2 / p.6〕。
   - p.8 卻說：機率分布是數「這個 context」的量化符號頻率得到〔原文 §6 / p.8〕。
   - **原文兩處不一致**。
   - 比「全域單一分布」最多再小 53%〔原文 §5.2 / p.6〕。GPU 版 AC，每個 CUDA thread 負責一個 token〔原文 §6 / p.8〕。
5. **多個等級怎麼產生**：先 prefill 整段 context，沿 token 維度切 chunk，每個 chunk 用多個量化等級各編一份。chunk 之間互相獨立可解碼（只要 chunk 比一個 token group 長）。可以離線做〔原文 §5.3 / p.7〕。
   - 有幾個等級、每個等級的參數，正文只給「預設等級」〔原文 App. C.2 / p.19〕。**等級數原文沒寫**。
6. **chunk 長度**：預設 1.5K token。理由：太大反應不了頻寬變化；太小則送文字時吃不滿 GPU 批次〔原文 §5.3 / p.7〕。
7. **第一個 chunk**：沒有頻寬資訊時用中等等級（Llama-7B 約 140 MB／chunk）〔原文 §5.3 / p.7〕。

### B. 讀取時逐 chunk 選擇（Algorithm 1）

8. **選項（configuration）**：某個編碼等級，或送文字讓 LLM 重算這個 chunk 的 K、V〔原文 §5.3 / p.6〕。
9. **正文的規則**〔原文 §5.3 / p.7〕：
   - 用「上一個 chunk 量到的吞吐量」估頻寬，假設之後都不變。
   - 每種設定的預期延遲＝大小 ÷ 吞吐量。
   - 挑「壓縮損失最小、而且預期延遲仍在 SLO 內」的設定。原文把**文字（重算）列為損失最小**的選項之一。
   - 頻寬變了，反應最多晚一個 chunk。
10. **Algorithm 1 的寫法**〔原文 App. C.1 / p.19〕（我用自己的話改寫）：
    - 逐 chunk 迴圈。剩餘時間＝SLO－已用時間。
    - 若「重算時間 ≤ 剩餘時間」→ 這個 chunk 送文字。
    - 否則 → 挑最大的 level，使「**剩下所有 chunk** 在這個 level 的大小 ÷ 吞吐量 ≤ 剩餘時間」，編碼後送出。
    - 送完再處理下一個。
    - **原文沒定義**：time_recompute 是這個 chunk 還是剩下全部；level 數字大代表品質高還是低（正文說「最低等級」損失最小，演算法卻取 max）。
11. **文字 chunk 怎麼重算**：用「前一個已收到並解碼的 chunk 的 KV」來算〔原文 §5.3 / p.7〕。所以重算要等前面的 chunk 到齊；而且前面若是有損 KV，重算結果會帶著前面的誤差〔判讀〕。
12. **多請求**：T 秒內到的最多 B 個請求一起批次串流。第 c 個 chunk 的預期延遲＝含 chunk c 的請求數 × 單請求延遲〔原文 §5.3 / p.7〕。
13. **短 context**：短於約 1K token 時自動退回全送文字，因為 TTFT 較低〔原文 §7.3 / p.10〕。
14. **品質與大小的取捨**：一個 chunk 用小等級，只增加那個 chunk 的壓縮損失，不影響其他 chunk。但若頻寬低到大多數 chunk 都只能用高壓縮，品質仍會掉〔原文 §5.3 / p.7〕。

### C. 重疊與位置

15. **唯一明寫的重疊**：傳 chunk i 的同時解碼 chunk i−1〔原文 §6 / p.8〕。
16. **重算有沒有和傳輸並行**：原文沒寫。Algorithm 1 是「決定 → 送 → 下一個」的依序迴圈〔原文 App. C.1 / p.19〕。〔判讀〕加上第 11 點的依賴，重算的 chunk 必須等前面 chunk 收完、解完。這和 Cake「前段從 token 算、後段同時載入」的結構不同。
17. **哪些 chunk 被重算**：沒有位置規則。看迴圈走到那個 chunk 時的頻寬與剩餘時間〔原文 App. C.1 / p.19〕。Fig. 7 的例子：t＝2 s 頻寬掉到 0.2 Gbps，改送文字重算；t＝4 s 回到 1 Gbps，再改送小等級〔原文 §5.3 / p.6–7 / Fig. 7〕。所以被重算的可以是中間的 chunk。
18. **重算成本與位置**：原文沒有「越後面的 chunk 重算越貴」的模型。time_recompute 沒定義〔原文 App. C.1 / p.19〕。（程式碼是固定常數，見 E 段。）
19. **和位置有關的發現**：
    - token-wise locality：同層同 channel 的相鄰 token 值相近；delta 的變異數比原值低 2.4–2.9×〔原文 §5.1.1 / p.4–5 / Fig. 3〕。
    - 依 token 位置分組，熵下降遠小於依 channel 或依 layer 分組〔原文 §5.1.3 / p.5 / Fig. 5〕。這是**壓縮**觀點（哪個維度好分組），不是品質觀點。
    - anchor token 保持 8-bit〔原文 §5.2 / p.6〕。
    - **沒有量「哪個位置的 token 對品質敏感」**。

### D. 品質評估

20. **層的敏感度**：對某個層群組加 rounding 損失，淺層掉最多，深層掉很少（Llama-7B、Llama-13B；LongChat 100 個 9.2–9.6K 的 context）〔原文 §5.1.2 / p.5 / Fig. 4〕。原文的解釋是淺層的損失會往後傳。
21. **資料與指標**〔原文 §7.1 / p.8 / Table 2〕：
    - 662 個 context：LongChat 200（中位 9.4K）、TriviaQA 200（9.3K）、NarrativeQA 200（14K）、WikiText 62（5.9K）。
    - LongChat 用 accuracy（生成答案含正確主題）；TriviaQA、NarrativeQA 用 F1；WikiText 用 perplexity。
    - 設計編碼器用的資料是評估資料的子集〔原文 p.8〕。
22. **模型與硬體**：長 context 微調版的 Mistral-7B、Llama-34B、Llama-70B（最多 32K）〔原文 §7.1 / p.8〕。Fig. 8b 標成 Llama-33B，原文內不一致〔原文 Fig. 8 / p.9〕。一台 4×A40 伺服器〔原文 §7.1 / p.9〕。實作約 2K 行 Python＋1K 行 CUDA，PyTorch 2.0、CUDA 12.0，透過 HuggingFace transformers 介面；text 基線用 vLLM〔原文 §6 / p.7；§7.1 / p.9〕。
23. **掉多少**：
    - 和預設量化相同品質下，大小小 3.5–4.3×〔原文 §7.2 / p.9〕。
    - 有損壓縮的掉幅：accuracy ≤2%、F1 ＜0.1%、perplexity ＜0.1〔原文 §7.2 / p.9〕。
    - Table 1（Mistral-7B、LongChat）：8-bit 622 MB／acc 1.00；CacheGen 176 MB／0.98〔原文 Table 1 / p.2〕。
    - 原文兩次稱 8-bit 量化「幾乎無損」〔原文 §1 / p.2；§7.2 / p.9〕。
24. **延遲**（3 Gbps）：比 text context 快 3.1–4.7×，比預設量化快 3.2–3.7×，比 8-bit 快 1.67–1.81×〔原文 §7.2 / p.9〕。頻寬掃 0.4–15 與 15–400 Gbps（16K）；超過約 20 Gbps 時對量化基線的絕對改善變小〔原文 §7.3 / p.10 / Fig. 11〕。
25. **適應實驗**：每個 chunk 的頻寬從 0.1–10 Gbps 隨機抽，20 條軌跡，LongChat、Mistral-7B。SLO 0.5 s 時，同品質下違約率比量化基線低 60%；SLO 1 s 時從 81% 降到 8%〔原文 §7.4 / p.10–11 / Fig. 13〕。
26. **開銷**：解碼在 GPU、和傳輸 pipeline，只有 Fig. 14a 一張圖、沒有數字〔原文 §7.5 / p.11〕。編碼約 200 ms〔原文 §7.5 / p.11 / Fig. 14c〕。多版本的儲存成本和量化基線相當〔原文 Fig. 14d / p.11〕。8.5K token、Llama-13B 的全部版本約 5 GB〔原文 App. E / p.19〕。
27. **限制與未來工作**：高階 GPU、低頻寬時可能不比 text 基線好〔原文 §9 / p.12〕。未來工作：像 SVC 一樣先送低品質 KV、再送差值補品質〔原文 §9 / p.12〕。

### E. 程式碼（〔程式碼〕）

**論文 artifact：UChi-JCL/CacheGen@6bed34c**

28. **逐 chunk 選擇是模擬的**：
    - `config_selection` 在 `src/utils.py:294–314`。
    - 每個 chunk 的頻寬直接取頻寬軌跡上**該 chunk 的值**（`:300`），不是用上一個 chunk 的實測吞吐量。
    - 由 level 3 試到 level 1（`:303`）。條件是「這個 chunk 的傳送時間 ＜ SLO ÷ chunk 數」（`:305`），不是「剩餘時間」。
    - 三個等級都塞不進，才選文字（config 0），並加一個固定的 `chunk_delay`（`:310–312`）。`chunk_delay＝0.2` 秒（`run_adaptation.py:77`）。**重算成本不隨位置變**。
    - TTFT 是把這些數加起來的估算值（`:306`、`:311`），不是端到端實測。
    - **和正文 Algorithm 1 的順序相反**：正文先試文字，程式碼最後才用文字。
29. **「重算」的品質是用原始 KV 代替**：`merge` 在 `src/utils.py:357–376`。config 0 的 chunk 直接切原始全精度 KV（`:365–366`），沒有真的接在有損前段之後重算。〔判讀〕所以 artifact 的適應實驗品質偏樂觀。另外，不滿一個 chunk 的尾巴不放進 KV（`:363–364`）。
30. **等級定義**：只對 Mistral-7B 有 `QUANT_LEVEL` 1／2／3〔`LMCache/lmcache/storage_backend/serde/cachegen_basics.py:32–68`〕。
    - level 3 的 bins 最多（K 三組都 32、V 兩組都 32），level 1 最少（K 16／16／12、V 16／12）。
    - K 分 0–9、10–19、20–31 三組；**V 只分 0–1、2–31 兩組**。和正文「三等份、bin size 0.5／1／1.5」寫法不同。
    - 這裡的「bins」是量化格數（對稱，最大值＝bins÷2－1），不是 bin size〔`cachegen_encoder.py:42–60`〕。
31. **編碼器沒有 anchor／delta**：逐（layer, token）在 channel 方向取 absmax 做對稱量化，再做 AC〔`LMCache/lmcache/storage_backend/serde/cachegen_encoder.py:42–60、261–263`〕。CDF 每次編碼現算，放進輸出〔同檔 `:265–267、294–296`〕。和正文第 2、4 點不同。
32. **腳本參數可能對不上**〔判讀，沒實際跑〕：`scripts/adapt.sh` 的編碼步驟沒給 `--chunk_size`（預設 1000，`run_adaptation.py:19`）；評估步驟給 1500（`adapt.sh:22`）。兩步切 chunk 的方式不同。

**現行 LMCache：LMCache/LMCache dev@240506e（2026-10-13）**

33. **只剩一組固定設定，沒有多等級**：K 0–9 層 32 bins、其餘 16；V 0–1 層 32、其餘 16。沒有 `QUANT_LEVEL`〔`lmcache/v1/storage_backend/naive_serde/cachegen_basics.py:35–58、110–129`〕。
34. **編碼器同樣沒有 anchor／delta**：逐（layer, token）absmax 量化〔`lmcache/storage_backend/serde/cachegen_encoder.py:47–70`〕。v1 的 serializer 呼叫同一個 `encode_function`〔`lmcache/v1/storage_backend/naive_serde/cachegen_encoder.py:96–112`〕。
35. **啟用方式是全域設定**：`remote_serde: "cachegen"`〔`docs/source/kv_cache_optimizations/compression/cachegen.rst:25、32`〕。這頁標為 in-process mode（已 deprecated）。MP mode 的 CacheGen 明寫「尚未實作」〔`docs/source/mp/cachegen.rst:4–8`〕。
36. **沒有逐 chunk 適應、沒有「送文字重算」**：在 tree 上找 adaptive、bandwidth、slo、streaming、recompute 相關檔名，只找到 `benchmarks/rag/precompute.py`；`gh search code` 找 `QUANT_LEVEL` 沒有結果；提到 cachegen 的程式檔只有上面的 serde、CUDA／SYCL 的 AC kernel 與 torch_ops 綁定，沒有等級選擇邏輯〔程式碼 tree listing＋code search，2026-10-13〕。〔判讀〕LMCache 沒命中的 token 由 vLLM 照常 prefill，那是 prefix 沒命中，不是依頻寬做的選擇。
37. **有一個「指定 token 範圍壓縮」的管理介面**：controller 提供 `compress／decompress(instance_id, method, location, tokens)`，可對指定 token 的 KV chunk、在指定儲存位置用 `cachegen` 壓縮或解壓〔`docs/source/kv_cache_management/compress.rst:15–16、87–94、108–115`〕。這頁同樣標為 in-process mode（deprecated）。〔判讀〕這是由外部下指令的「靜態、依範圍」壓縮，不是讀取時依頻寬選；但它是現成可用來做「依位置只壓某些 block」的掛點之一。沒有實際跑過。

## 原文沒寫的地方

- 等級數、各等級的參數（只有預設等級）。
- Algorithm 1 的 time_recompute 定義、level 的方向。
- 文字 chunk 的重算能否和後面 chunk 的傳輸並行。
- 重算成本怎麼估（有沒有考慮位置）。
- 文字 chunk 接在有損前段後面時的品質。
- token 位置對品質的敏感度。
- 解碼延遲的數字（只有圖）。
- 逐 chunk 適應實驗裡，實際有多少 chunk 被重算、在什麼位置。

## 和我們實驗的關係〔判讀〕

1. **CacheGen 是「讀取時才選精度」的代表**。寫入時每個 chunk 存多份，讀取時依頻寬挑。它天生避開「寫入時決定、不能回頭」的問題，代價是多存。我們的位置政策是「寫入時只存一份」。
2. **公平比較要加一個 CacheGen 式基線**（單張 3090、Llama-3.1-8B 或 Qwen2.5-7B-1M、≤32K）：
   - 每個 512-token block 同時存 BF16 與 FP8（共 1.5× BF16 的位元組）。
   - 讀取時逐 block 依「上一個 block 量到的吞吐量」挑 BF16 或 FP8，規則照正文 §5.3：在 SLO 內挑損失最小的。
   - 這是「品質／延遲」的參考上界。要問的是：位置政策能不能用 ≤1× 的儲存逼近它。
3. **重算在我們的設定下只能從最前面連續做**。vLLM 的 prefix 語意下，重算中間的 block 要先有前面所有 block 的 KV。〔判讀〕所以「CacheGen 式逐 chunk 選＋重算」在 3090 上會收斂成：前段重算（像 Cake）＋後段逐 block 選 BF16／FP8。它和 Cake＋位置政策的差別只剩「精度在讀取時選（要多存）」還是「寫入時定（只存一份）」。這正是要比的那條軸。
4. **品質量測時，重算的 block 要真的重算**。不要像 artifact 一樣直接拿原始 KV 代替（第 29 點）。如果重算接在有損 block 後面，重算結果也會帶誤差，要量進去。
5. **重算成本要隨位置變**。CacheGen artifact 用固定 0.2 s／chunk（第 28 點）；Cake 量到成本隨 chunk index 線性上升（見 P01）。我們的成本模型要用後者，並在 3090 上實測。NOT_MEASURED。
6. **不要把現行 LMCache 的 CacheGen 當「逐 chunk 選擇」基線**。現行碼只有固定 codec（第 33–37 點）。要比 CacheGen 的適應邏輯，要自己照 Algorithm 1 實作，並寫明採正文版（先試重算）還是 artifact 版（先試編碼）。
7. **品質證據的方向**：CacheGen 說 8-bit 幾乎無損（Table 1：Mistral-7B、LongChat，acc 1.00）。這和老師「如果全 FP8 就夠就停」的停損一致：先量全校準 FP8。注意 CacheGen 的 8-bit 不是 FP8，模型與資料也不同，不能直接搬。
8. **位置**：CacheGen 唯一和位置有關的是 anchor 8-bit 與熵結果。它沒量 token 位置的品質敏感度。我們的前／後／隨機 25% 實驗不和它衝突，但也得不到它的支持。
9. **粒度與選項**：CacheGen 1.5K-token chunk、多等級＋AC；我們 512-token block、只有 BF16／FP8 兩級。32K 時 CacheGen 約 21 個 chunk，我們 64 個 block。這是刻意簡化，要寫明。

## 與舊卡不一致之處（修正清單）

1. **D6_CacheGen**：「算術編碼的機率分布按 channel×layer 離線 profile〔原文 p.6〕」→ 原文 p.6 與 p.8 自相矛盾（p.8 說是對該 context 計數）；artifact 與現行 LMCache 都是每次編碼現算 CDF〔程式碼〕。
2. **D6_CacheGen**：「每 10 個 token 一組，anchor 8-bit，其餘存 delta」→ 原文確實這樣寫，但**公開程式碼（artifact 與現行 LMCache）都沒有 anchor／delta**，只有逐（layer, token）absmax 量化＋AC〔程式碼〕。
3. **D6_CacheGen**：「層分三組，bin 0.5／1／1.5」→ 原文 App. C.2 這樣寫；artifact 的 K 分三組（0–9、10–19、20–31）、V 只分兩組（0–1、2–31），用 bins 數表示；現行 LMCache 只剩一組固定設定〔程式碼〕。
4. **D6_CacheGen／D5_CacheGen**：Algorithm 1 的描述要補三點：(a) 選等級時看的是「剩下所有 chunk」在該等級的大小，不只當前 chunk；(b) 文字（重算）被列為「損失最小」的選項，只要來得及就優先，不只是「頻寬太低時」才用〔原文 §5.3 / p.7〕；(c) time_recompute 原文沒定義。
5. **D6_CacheGen／D5_CacheGen**：沒提程式碼。補：artifact 的實際邏輯和 Algorithm 1 相反（先試編碼、最後才用文字），用頻寬軌跡值而不是上一 chunk 的實測，重算成本固定 0.2 s，重算品質用原始 KV 代替〔程式碼〕。
6. **D8_CacheGen**：「方向 3 的對照組之一（逐 chunk 靜態選擇）」→ 不是靜態。是依上一個 chunk 量到的吞吐量**動態**逐 chunk 選〔原文 §5.3 / p.7〕。另外第 2 點「頻寬太低時改傳文字」只說了一半，見第 4(b) 點。
7. **D5_CacheGen**：模型寫「Llama-34B」正確〔原文 p.8〕；補註 Fig. 8b 標 Llama-33B，原文內不一致〔原文 p.9〕。
8. **D6_CacheGen**：PDF 路徑 `/mlsteam/...` 在這台 Mac 上不存在；本次從 arXiv 重新下載 v6。
9. 其他數字（1.5K chunk、140 MB、＜1K 退回文字、≤2% 掉幅、4×A40、200 ms 編碼、5 GB 儲存）**已逐項核對一致**。

## 證據等級

- **A**：v6 正文 p.1–12 與附錄 p.17–19 逐頁讀過；所有數字附頁碼。
- **程式碼**：artifact 與現行 LMCache 的相關檔案逐行讀過（只讀、沒有執行）。行號以上面列的 commit 為準。
- **〔判讀〕**：重算依賴前段的推論、artifact 品質偏樂觀、腳本參數可能對不上，都沒有實際跑過驗證。
