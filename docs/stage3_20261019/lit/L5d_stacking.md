# L5d　把多種 KV 壓縮疊在一起用（量化＋低秩＋丟／合併 token＋跨層＋無損）：查新、機制、代理指標與三階段探索設計

> **日期**：2026-10-20（UTC）
> **執行者**：文獻 agent（只讀檔、只上網）。**沒有開任何 GPU 行程，沒有 import torch／vllm／transformers**。CPU 只用來把 HTML／PDF 轉成文字、跑一個 4 行的熵模擬（§2.4），一律 `nice -n 19 taskset -c 0-3`。除了本檔，沒有改任何既有檔案。
> **範圍**：只看「組合」。單一方法的卡片在 `L5a_lowrank.md`（低秩）、`L5b_merge_crosslayer.md`（合併、跨層）、`L5c_compress_on_offload.md`（卸載時依連線速度壓縮）、`L1`（eviction）、`L2`（量化／無損／卸載），本檔只引用。
> **先讀過的本專案檔案**：`research_map.md`、`lit/L1`、`lit/L2`、`lit/L5a`／`L5b`／`L5c` 的一頁結論、`docs/precision_20261013/08_report_mi300x.md`（§0、§2.3–2.5、§2.9）、`cards/P02_CacheGen_mechanism.md`、`results/m10_longctx_mi300x/analysis/tables_longctx.md`、`results/m10_drop_mi300x/ds1/tables_ds1.md`、`experiment_log.md`、`experiment_matrix.md`、`code/m16_s1_kvdump.py`、`code/m16_kvstruct.py`、`code/m16_s2_equalbytes.py`（只讀檔頭說明）。
> **下載的檔**：`/mlsteam/data/tiara/papers/stage3_L5/L5d/`（約 30 篇 arXiv HTML 與轉出的 .txt、`abs/` 下的 abs 頁、`pdf/qhitter_mlsys24.pdf` 與抽出的文字、`pdf/cachegen_v6.txt`）。都 <20 MB。
> **本檔是 self-review，不是 cross-model review**（CLAUDE.md §5）。

**證據等級**
- 〔全文〕＝把 arXiv HTML（或 PDF）轉成文字後，正文＋相關附錄逐段讀過。本檔讀全文的有：**GEAR、KVTC、MiKV、More Tokens Lower Precision、JoLT**，以及被作者自己撤回結論的 2604.11501（v1、v2 都讀）。
- 〔部分〕＝讀了方法段、指定的表與附錄、限制段（RDKV、AATC、Palu、ZipCache、MiniKV、ThinKV、ReST-KV 附錄 I、DefensiveKV 附錄 E、KVzip §4.2、Rethinking（MLSys'25）、Q-Hitter（PDF 抽字）、CacheGen（PDF 抽字＋repo 的 P02 卡）、MiniCache、XQuant、AQUA-KV、SVDq、ThinK、TailorKV、DiffKV、2607.05399）。
- 〔摘要〕＝只讀 abs 頁（用 curl 實際開過，§8 的「核」欄）或會議頁。
- 〔README〕〔程式碼〕＝讀 GitHub README、本專案程式碼。〔實測〕＝本專案已量的數字，附檔案或 run_id。
- 〔判讀〕＝我對原文的解讀；〔推論〕＝原文沒寫、我推的；〔算術〕＝我用原文或本專案的數字算的，式子寫在旁邊。
- **NOT_FOUND**＝這輪查了沒找到（≠不存在，CLAUDE.md 規則 7）；**NOT_MEASURED**＝還沒有人（或我們）量過。
- 不同硬體、模型、context、預算的數字**只並列、不排名**。

---

## 0. 一頁結論

1. **「全部疊加會不會更好？」——文獻的答案是：有條件地會，而且條件很窄。**
   - **會更好的情況**：預算緊到「單一方法自己撐不住」的時候（大約每個值 <3–4 bit，或只留 <25% 的 token）。這時兩兩組合在多篇論文裡都贏單一方法，而且是同位元組比的：
     - 丟 token＋量化：More Tokens Lower Precision〔全文〕在同記憶體下「4 倍 token、4 bit」穩定贏「1 倍 token、16 bit」；RDKV〔部分〕用同一個分配器做消融，「只量化」「只丟」「兩者聯合」在 128L／512L 預算下是 39.06／40.59、42.53／47.18、47.43／49.22（LongBench，Llama-3.1-8B-Instruct）。
     - 低秩＋量化：JoLT〔全文〕在 2–3× 這一段贏所有低秩基線。
     - 量化＋無損：KVTC〔全文〕的 DEFLATE 平均再省約 1.23×；CacheGen〔部分〕把 H2O 的 KV 再壓到 71 MB（單用 CacheGen 176 MB、8-bit 622 MB）。
   - **不會更好的情況**：一個好的量化格式已經幾乎無損時。JoLT 在約 4× 同位元組下，「低秩＋量化殘差」輸給單純的 KIVI-2、TurboQuant、INT4（GSM8K 35.94 vs 40.18／38.89／37.07）〔全文〕。GEAR 三件全上，在 LongBench 2-bit 反而低於只用兩件的 GEAR-L（25.48 vs 27.90）〔全文〕。**我們第二階段已量到 INT4 KIVI（每值 5 bit、位元組 0.3125）在 32K–128K 兩個模型都不掉**（`tables_longctx.md`）——在這個位元組附近，疊加沒有空間可以贏。
   - **三種以上「各自有損」的方法疊在同位元組、≥32K 下比單一方法好：NOT_FOUND。** 最接近的 KVTC 是兩件有損（PCA 降維＋量化）加一件無損，而且它明寫「準確率下降讓各種方法無法疊加出複合的好處」（KVTC §1）——它的做法是在同一個預算下**聯合分配**位元，不是把各自調好的方法疊上去。
2. **疊加會互相干擾，而且干擾的方向已經有人量過**（§2）：
   - 量化會讓「該留哪些 token」挑錯，而且被留下的重要 token 偏偏不好量化（Q-Hitter，MLSys'24〔部分〕）。
   - KIVI 的 K 是「逐 channel、沿 token 分組」量化；在組裡丟 token 會破壞分組，要另外設計（MiKV 附錄 C〔全文〕、MiniKV〔部分〕）。
   - 低秩後的係數分布前幾維特別大：若「逐 token 跨維」量化就會被撐開，要先做 Hadamard（Palu〔部分〕）；若「每一維各自給位元」反而更好（SVDq、KVTC、AATC）。
   - RoPE：低秩要在 RoPE 前做，K 才是低秩的（KVTC、JoLT、L5a）；代價是還原時要重算 RoPE，這是 JoLT 自己說的主要延遲來源〔全文〕。
   - 無損：量化越細（每 32 個值一組）、越接近均勻分布，熵越接近位元數。**我的模擬**：INT4、32 一組時 order-0 熵編碼最多再省 1.03–1.06×；INT2 或 128 一組才有 1.15–1.48×〔算術，合成資料，§2.4〕。
3. **疊加有時間成本，這件事比品質更少人量**（§4）：
   - GEAR 的 prefill 吞吐比 FP16 低，而且差距隨長度變大；ShareGPT 上平均端到端延遲 GEAR 13.4 s、FP16 11.4 s、KIVI 9.1 s（Rethinking KV Cache Compression，MLSys'25〔部分〕）。
   - JoLT 的融合 kernel 在 batch 1 比不壓的 attention 慢 7.5–65.9 倍〔全文〕。
   - KVTC 在同一張 GPU 上壓縮／解壓，4–10 個 client 時 TTFT 比不壓慢 25–42%，要等 host 記憶體裝不下才贏〔全文表 22；算術〕。
   - ThinKV：重疊執行的 gather 壓實會和 attention 搶 HBM，attention 最多慢約 35%〔部分〕。
   - **「組合 × 連線速度 × GPU 忙」三者一起看：NOT_FOUND。** KVServe 依頻寬選組合，但把 GPU 爭用當成殘差修正（L2 B2、L5c）。
4. **便宜的代理指標：「重建誤差」跨方法不可靠；「attention 輸出誤差（加權）」比較好，但沒人系統驗證過**（§3）。
   - JoLT 附錄 E〔全文〕在 LLaMA-2-13B、約 4× 時：KIVI-2 的 Frobenius 誤差（K 0.201、V 0.383）遠大於 INT4（0.077／0.124）與 JoLT（0.120／0.087），困惑度卻是 KIVI 最好之一（+1.59% vs +8.08%、+14.16%）。**排序整個反過來**。
   - KVTC 附錄 B.5〔全文〕：**同一個方法**內，誤差和平均分數的 Pearson −0.95〔原文表 8〕——所以「同家族內排序」可以用，「跨家族排序」不行。
   - AATC〔部分〕：只用變異數（＝重建誤差）分位元，在 2 bit 時 RULER 隨長度掉；改用含 query 與 W_O 的 attention 感知失真就撐住。
   - **本專案第二階段自己的證據**：海選勝出的格子 KL 比隨機低 31–37%，答對率卻只差 0–0.5 個百分點（08 §2.9）；但 KL 抓得到整個壞掉的格式（Qwen FP8 KL 3.84 對應 −32.8 個百分點）。→ **KL 是好的「崩潰警報」，不是好的「排名器」**。
5. **查新判定**（§5）：
   - **兩兩組合已經很擠**（至少 25 篇）：丟＋量化、低秩＋量化、跨層＋量化、量化＋熵編碼、丟＋低秩，都有人做，也有人做「同一個預算下聯合分配」（RDKV、JoLT、KVTC、AATC）。
   - **還空著的**：(a) 三軸以上、照實算位元組（含 scale、索引、基底）、≥32K、GQA 模型上的 Pareto 前緣圖；(b) 代理指標**跨方法家族**能不能預測答對率；(c) 疊加的還原成本在 **GPU 忙／不同層的連線速度**下會不會翻轉排序——這一條和研究地圖缺口 #1（干擾）是同一條線，也是 Tiara 能講的話；(d) Cake 還原下只壓載入區；(e) MI300X／ROCm。
   - 風險先講：(a)(b) 是評測型貢獻；(c) 最可能有新意，但 L5c 已預測「允許有損時整張圖都是 INT4」——要靠 §6 的停止條件判。
6. **三階段設計**（§6）：
   - **(1) CPU 掃描**：在真實 KV dump 上掃約 300–500 種組合，位元組照實算，先用第二階段已量到答對率的約 30–60 個配置**校準代理指標**（Spearman ≥0.6 才拿來挑前緣）。約 8–25 CPU（4 核）小時〔估計〕，GPU 只要補一次 KV＋Q 的匯出（約 0.3 小時）。
   - **(2) GPU 品質**：每個預算（4×、8×、16×）挑前緣 2–6 個組合，和 INT4 KIVI、各家族最好的單一方法比生成答對率。約 4–5.5 小時牆鐘（2 張卡）〔估計〕；最小版約 2 小時。
   - **(3) 時間**：解碼成本微測（GPU 忙 duty 0／0.4／0.7／0.9）＋模擬還原時間＋4–8 格實測驗證。約 1–1.5 GPU 小時＋<1 CPU 小時〔估計〕。
   - **主要停止條件**（事先寫死，§6.5）：在 ε 內能省下的位元組，最好的組合比最好的單一方法少不到 5%（每個預算、兩個模型都是）→ NO-GO，寫成「疊加沒有加乘」；5–15% → 停下來問人；>15% → 進第三階段。
7. **事前預測**（§6.7）：Llama 在 4× 沒有空間（K4V2 已在 ε 內）；Qwen 在 4× 有空間（K4V2 32K −5.1 個百分點）；8×、16× 我們的格式裡**沒有任何單一量化格式到得了**，組合幾乎一定贏「只丟」或「只降秩」——所以真正的問題不是「疊加好不好」，而是「最好的組合在 CPU 層、GPU 忙時，扣掉解碼成本後還快不快」。

---

## 1. 已經做過的「組合」

### 1.1 總表

「同位元組比過？」：✓＝在相同或接近的位元組下和單一方法比；◐＝比了但位元組不同（通常組合更小）；✗＝沒比。「結論」用三個詞：**加乘**（組合比任一單一方法好）、**抵消**（疊上去反而變差，或要重新設計才不變差）、**看預算**（某些預算加乘、某些不）。

| # | 作品（年，venue） | 疊了哪幾種 | 疊的順序 | 同位元組比過？ | 結論 | ≥32K？ | 還原成本 | 深度 |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| S1 | **GEAR**（2024，NeurIPS ENLSP workshop，PMLR 262） | 量化＋低秩殘差＋稀疏離群值 | 先挑每向量 s% 離群值存 FP16 → 其餘量化 → 殘差做每 head 低秩（power iteration） | ✓（KV 大小 % 有列；GEAR-L 23.6% vs KIVI 21.7%） | **看任務／預算**：CoT 推理 2-bit 加乘；LongBench 2-bit 三件全上比兩件差 | ✗（GSM8K 前綴 ~0.9–1.3K、LongBench 平均 3.6K） | V100 上時間拆解稱「可忽略」；MLSys'25 實測 prefill 吞吐變低、端到端變慢 | 〔全文〕 |
| S2 | **KVTC**（2025，ICLR'26） | PCA 降維＋DP 分位元量化＋DEFLATE（無損） | 留 sink 4 個＋最近 128 → 拿掉 RoPE → PCA → DP 分位元（很多尾端 0 bit）→ DEFLATE | ◐（KIVI／GEAR 5×、H2O／TOVA 8×、kvtc 8–64×） | **加乘**（PCA＋量化遠勝只刪主成分；DEFLATE 平均再 ~1.23×）；但作者說「天真疊加」不行 | 部分（NIAH 100K 只在 70B；其餘 ≤8K；LMCache 測 ~64K） | H100、8K×8：壓 379 ms、解 267 ms；同 GPU 多 client 時變慢 | 〔全文〕 |
| S3 | **CacheGen**（2023，SIGCOMM'24） | delta 編碼＋分層量化＋算術編碼；另疊在 H2O、LLMLingua 上 | delta（對 anchor）→ 分層量化 → 依 channel×layer 機率表做 AC | ◐（同品質下位元組少 3.5–4.3×；疊在 H2O 上 71 vs 176 MB） | **加乘** | ✗（1.4K–16K） | GPU 解碼會拖慢同機計算（ShadowServe、KVCodec 量到，L2 B4／B5） | 〔部分〕＋P02〔全文〕 |
| S4 | **Palu**（2024，ICLR'25） | 低秩（權重分解，快取 latent）＋量化 | 低秩投影 → Hadamard（吸收進權重）→ 逐 token 量化 latent | ◐（Palu-30%＋3 bit 7.59× vs KIVI-2 5.05×，準確率相近） | **加乘**（對只降秩）；對只量化是「同準確率更省」 | ✗（LongBench，模型上限 32K） | RoPE 模型的 K 要線上重建；其 4-bit 版本 K 不量化（kernel 只支援 FP16） | 〔部分〕 |
| S5 | **MiKV**（2024，arXiv；venue NOT_FOUND） | 丟 token（H2O）的那一份改存低精度＋重要的存高精度 | 重要度 → 重要 token FP16／INT8／INT4，其餘 INT2／INT3（含 query-key 通道平衡） | ✓（壓縮率－準確率曲線對 H2O、RTN） | **加乘**（對只丟）；被丟的存 INT2 也大幅救回 | ✗（line retrieval 20 行；記憶體表用 4K） | 用權重量化 kernel；要算 attention 分數 | 〔全文〕 |
| S6 | **ZipCache**（2024，NeurIPS'24） | 依顯著度混合精度（4／2 bit）＝「軟性丟 token」＋量化 | 正規化 attention 分數（只算 10% token）→ 顯著 4 bit、其餘 2 bit | ◐（4.98× vs MiKV 同比；GEAR 3.0×、KIVI 3.46×） | **加乘**（短 context） | ✗（GSM8K ~840、line retrieval 3K） | 和 FlashAttention 相容的近似分數 | 〔部分〕 |
| S7 | **Q-Hitter**（2024，MLSys'24） | 丟 token（H2O）＋量化 | 累積 attention＋「量化友善度」線性組合 → 選 token → 量化 | ✓（同位元下對 H2O＋量化） | **天真疊加會抵消**，重新選 token 才加乘 | ✗（Llama-2 等，≤4K） | 要算 attention 與量化誤差 | 〔部分〕（PDF 抽字） |
| S8 | **More Tokens, Lower Precision**（2024，EMNLP Findings'25） | 丟 token（6 種）＋量化（KIVI、FlexGen、KVQuant） | 先丟 → 留下的量化 | ✓（同記憶體：1×16b vs 2×8b vs 4×4b） | **看預算**：低預算 4 bit 最好、高預算 8 bit 最好；2 bit 全面崩 | 部分（Llama-3 8K、Mistral NIAH 到 32K、RULER 4–16K） | 作者自承反量化沒最佳化，沒有加速 | 〔全文〕 |
| S9 | **RDKV**（2026，arXiv） | 丟 token＋量化，當成同一條位元分配曲線的兩端 | attention 失真權重 → 反向注水分配 {0,2,4,8,16} bit（0＝丟）；V 逐 token、K 逐 channel | ✓（同預算消融：只量化／只丟／三態／聯合） | **加乘**（在 1.5–6% 保留率這種極緊預算） | ✓（RULER 到 128K、InfiniteBench） | 打包解碼 TriZone：18 ms／token vs 不打包 36 ms | 〔部分〕 |
| S10 | **JoLT**（2026，投 ICLR'27） | 低秩（partial Tucker）＋旋轉後的低位元殘差 | pre-RoPE K → Tucker（token、feature 兩個 mode）→ 殘差隨機旋轉 → 0／2／4／8 bit；Lagrangian 聯合分配 | ✓（約 4× 對 KIVI-2、TurboQuant、INT4） | **看預算**：2–3× 贏低秩基線；約 4× 輸給單純量化 | ✓（RULER 64K） | 融合 kernel 比不壓慢 7.5–65.9×（batch 1） | 〔全文〕 |
| S11 | **AATC**（2026，arXiv） | 去相關（白化 SVD）＋attention 感知分位元（0 bit＝降秩） | 白化 → 反向注水 | ◐（各法 5.0–5.9×） | 單法；但它比了 Palu（低秩＋量化）與 KIVI、KVQuant，**在我們兩個模型上** | ✓（RULER 4K–32K） | 沒報 | 〔部分〕 |
| S12 | 2604.11501 v1「Quantization Dominates Rank Reduction」→ v2 改名 | 低秩＋量化的「hybrid」對純量化 | — | v1 稱 ✓；**v2 撤回「同儲存比較」**（實驗是改權重，不是改快取） | **不可用** | ✗ | — | 〔全文〕（v1、v2） |
| S13 | **MiniKV**（2024，ACL Findings'25） | 丟 token（pyramid）＋2-bit | 預填時選定「持續」的 token → 子 channel K／逐 token V 2-bit | ✓（同 KV 大小對 H2O、SnapKV、Q-Hitter） | **加乘**（要重新設計才不抵消） | LongBench | 兩段式 Triton selective flash-attention | 〔部分〕 |
| S14 | **ThinKV**（2025，ICLR'26 Oral） | 依「思考類型」混合精度＋丟 token | 依 attention 稀疏度分類 → TBQ 量化、TBE 丟 | ✓（Pareto 對只量化 KIVI、只丟 R-KV） | **加乘**（推理模型：量化會讓輸出變長，丟可壓住） | 長輸出（推理） | 壓實會和 attention 搶 HBM（最多 ~35%） | 〔部分〕 |
| S15 | **ReST-KV**（2026，ICLR'26），附錄 I | 丟 token＋KIVI／KVQuant 4-bit | 先丟 → 4 bit | ◐（總 6.25% vs 2-bit 全量化） | **加乘**（文字＋圖，數字只在圖上） | LongBench | — | 〔部分〕 |
| S16 | **DefensiveKV**（2025，arXiv），附錄 E | 丟 token（留 40%）＋int4（Quanto） | 先丟 → int4 | ✗（只和不量化的 40% 比） | 記憶體 10%，LongBench 49.21 → 48.55 | LongBench | — | 〔部分〕 |
| S17 | **KVzip**（2025，NeurIPS'25 Oral），§4.2 | 丟 token（70%）＋QServe KV4 | 量化模型上直接打分、丟 | ✗ | 124K 時 16.3 GB → 1.2 GB，「損失可忽略」 | ✓（124K；也測 Qwen2.5-7B-1M，但量化組合只在 LLaMA3-8B） | — | 〔部分〕 |
| S18 | **MiniCache**（2024，NeurIPS'24〔依二手引用〕） | 跨層合併（深半段）＋4-bit | 合併 → 量化 | ◐（5.02× vs KIVI-2 3.95×） | **加乘**（Llama-2-7B-chat，LongBench：35.44 vs 31.51，基線 36.41）；L5b：xKV 在 Llama-3.1-8B 重測時 1.3× 就崩 | ✗ | 逐元素縮放 | 〔部分〕＋L5b |
| S19 | **XQuant**（2025，EMNLP'25） | 跨層共用量化碼＋資料無關校準 | 相鄰層共用量化後的值、各自 scale／zero | ◐（~1.4 bit vs KIVI-2、AsymKV-1.5） | 加乘（作者稱） | LongBench | — | 〔部分〕 |
| S20 | **AQUA-KV**（2025，ICML'25） | 跨層／K→V 線性預測＋量化殘差（＋H2O） | 預測 → 殘差向量量化；可再接丟 token | ✓（同位元寬） | 加乘（2–2.5 bit 近無損）；**要校準訓練小預測器 1–6 GPU 小時** | LongBench | — | 〔部分〕 |
| S21 | **xKV＋4-bit**（2025，ICML'26） | 跨層共用 SVD 基底＋RTN 4-bit | SVD → 量化係數 | ◐ | 25.6× 時 88.85 → 87.64（RULER 64K） | ✓ | 每層一次 GEMM＋補 RoPE | 依 L5b〔全文〕 |
| S22 | **SVDq**（2025，arXiv） | SVD 潛在通道＋依重要度混合精度＋key 稀疏 | SVD → 分位元；再接 top-k | ◐ | 加乘（K 達 410×，用於 attention 計算） | ✓（RULER；L5a：Qwen2.5-7B 64K） | 每 prompt SVD | 〔部分〕＋L5a |
| S23 | **ThinK**（2024，ICLR'25 spotlight） | K 的 channel 剪枝＋丟 token 或 KIVI | 剪 channel → 再接 H2O／SnapKV／KIVI | ◐ | 加乘（和 KIVI 疊到 2.8× 峰值記憶體減少） | LongBench、NIAH | — | 〔部分〕 |
| S24 | **ShadowKV**（2024，ICML'25）／**Loki**（NeurIPS'24） | 低秩 K（當索引）＋稀疏選取＋V 卸載 | pre-RoPE 低秩 K 在 GPU → 挑 → 只搬選到的 V | — | decode 時的組合；不是儲存格式 | ✓ | 重建 K 與搬 V 重疊 | 依 L1／L2／L5a |
| S25 | **TailorKV**（2025，arXiv） | 逐層二選一：1-bit 量化 或 卸載＋top-k 取回 | 依「稠密度」把層分兩類 | ◐ | 加乘（作者稱） | ✓（128K 在 3090） | — | 〔部分〕 |
| S26 | **DiffKV**（2024，SOSP'25） | K／V 不同精度＋token 重要度三級（高／低／剪）＋每 head 稀疏 | 分級 → 不規則記憶體 → GPU 上平行壓實 | ◐ | 2.7–5.7×、吞吐 1.9–5.4× | 推理模型 | 要專用記憶體管理器 | 〔摘要〕＋目次 |
| S27 | **Joint encoding of KV blocks**（2026，arXiv） | 跨請求合併相似 block＋量化 | 合併 → 量化 | ✗ | 合併 4.38×、加量化到 5× | 未查 | — | 〔摘要〕 |
| S28 | **QEvict**（2026，arXiv） | 全精度／可回復 INT2／丟 三階 | — | ✓ | 加乘（作者稱） | ✓ | — | 依 L1 A22 |
| S29 | **SemKV**（2026，arXiv） | 依重要度兩級精度 | — | ✓ | 「懸崖以上，8 種重要度指標統計上沒差；混精度的好處只是補出均勻量化做不到的平均位元」 | 未查 | — | 〔摘要〕 |
| S30 | **系統依頻寬選組合**：KVServe（SIGCOMM'26）、CacheGen、AdaptCache（BigMem@SOSP'25）、EvicPress（2025） | Transform×Quantizer×Codec；或壓縮＋丟＋放哪一層 | 依頻寬／SLO／命中頻率選 | — | 見 §4 | — | GPU 爭用沒建模 | 依 L2、L5c |

**系統性比較／benchmark**（只比單一方法、沒有疊加的也列，因為它們回答「哪個單一方法最好」）：

| 作品 | 比了什麼 | 有沒有比組合 | 和本題相關的發現 | 深度 |
|:--|:--|:--|:--|:--|
| KV Cache Compression, But What Must We Give in Return?（2407.01527，venue 未查證） | 量化、丟 token、prompt 壓縮、線性時間模型、混合架構 | ✗ | — | 〔部分〕（grep「combin」只有架構混合） |
| Rethinking KV Cache Compression…（2503.24000，MLSys'25） | KIVI、GEAR、H2O、StreamingLLM 的吞吐、長度、負樣本 | ✗（GEAR 本身是組合） | GEAR prefill 吞吐低、端到端最慢；壓縮讓輸出變長 ×1.55–1.76；平均分數藏住逐題失敗 | 〔部分〕 |
| Benchmarking KV-Cache Optimizations…（2607.05399，arXiv） | KIVI、TurboQuant、SnapKV、CaM | ✗ | 「壓縮比本身是很差的端到端效能預測指標」；KIVI4 品質最穩 | 〔部分〕 |
| Not All Bits Are Equal（2510.10964，arXiv） | 推理模型上權重位元、KV 丟與量化 | 未見 | 有效模型小時丟 token 比量化好；大模型兩者相當 | 〔摘要〕＋目次 |
| More Tokens, Lower Precision（S8） | 6 種丟法 × 3 種量化 × 預算 × 模型大小 × 長度 | ✓ | 見卡片 | 〔全文〕 |
| RDKV 消融（S9） | 同分配器下只量化／只丟／三態／聯合 | ✓ | 見卡片 | 〔部分〕 |
| JoLT 附錄 C／E（S10） | 低秩 6 種、量化 3 種、8-bit 2 種，同位元組 | ✓ | 見卡片 | 〔全文〕 |

### 1.2 指定的六篇

#### S1　GEAR〔全文〕
- **出處**：Hao Kang, Qingru Zhang, Souvik Kundu, Geonhwa Jeong, Zaoxing Liu, Tushar Krishna, Tuo Zhao。arXiv [2403.05527](https://arxiv.org/abs/2403.05527)（v4，2024-09-30；arXiv 頁沒有 venue）。正式版題目改為 *GEAR: An Efficient Error Reduction Framework for KV Cache Compression in LLM Inference*，PMLR 262:305–321（NeurIPS 2024 ENLSP workshop；10/20 開 PMLR v262 目錄核過）。
- **疊了什麼、順序**：(i) 每個 channel（K）或 token（V）抽出最大、最小各 s/2% 的值存 FP16 稀疏矩陣 → (ii) 其餘用 KIVI 或 KCVT（KIVI 但不細分組）量化 → (iii) 殘差 R 每個 head 做 rank-r 近似（power iteration），預填 r＝4、解碼時每一批 n_b 個新 token（緩衝區；KCVT 骨幹可設 20，KIVI 骨幹用 64）r＝2。只用 (ii)(iii) 叫 GEAR-L〔原文 §3、§4〕。
- **同位元組比較**（「KV size」＝佔 FP16 的百分比，已含 scale／zero、殘差、稀疏、低秩）：
  - CoT 推理（GSM8K／AQuA／BBH，8-shot，預填 900–1,304 token，生成 256）LLaMA3-8B 2-bit：KIVI 30.17／25.36／30.92（21.7%）；GEAR-L 52.62／38.19／51.44（23.6%）；GEAR 54.59／38.19／50.30（27.6%）；FP16 54.21／38.19／53.66〔原文表 1〕。→ **在 2 bit 這個「量化自己撐不住」的點，加乘很明顯**。
  - LongBench（LLaMA2-7B，21 個任務，平均輸入 3,642）：FP16 26.82；KIVI（g32）2-bit 27.83（19.7%）；GEAR-L 27.90（19.1%）；**GEAR 25.48（23.1%）**〔原文表 2〕。→ 三件全上**比兩件差、也比單純 KIVI 差**；作者自己寫「這類簡單任務量化本來就近乎無損」。
  - 只加稀疏（Outlier-A）不加低秩：LLaMA3-8B GSM8K 36.01（24.5%），遠低於 GEAR-L 52.99（23.4%）〔原文表 8〕。作者：「拿掉稀疏只會小幅變差，因為量化分組本身已部分處理離群值」〔原文 §4.3〕。
  - 和丟 token 比：H2O 丟 50%（16 bit）GSM8K 6.82，GEAR（KCVT 4-bit，32.4%）16.14，FP16 16.33〔原文表 10，LLaMA2-7B〕。
- **還原成本**：V100、LLaMA2-7B、輸入 1,000、生成 500、權重 8-bit：作者的時間拆解圖說低秩與稀疏「可忽略」；吞吐對 FP16 最多 5.07×（因為 batch 變大）〔原文 §4.2〕。**外部量測相反**：MLSys'25 的 Rethinking 量到 GEAR 讓 prefill 吞吐變低、而且長度越長越明顯，ShareGPT 平均端到端延遲 13.4 s（FP16 11.4、KIVI 9.1）〔部分，§4.2／表 8〕。
- **限制**：所有層、head 同一個 rank 與稀疏比〔原文 §6.1〕；沒有 ≥32K；沒有和「同位元組的更溫和量化」（例如 GEAR-L 23.6% 對 KIVI 4-bit）在 CoT 上直接比——表 1 的 KIVI 4-bit（34.2%）是 46.25／36.22／48.03，GEAR-L 4-bit（29.0%）53.44／38.98／52.23。
- **對我們**〔判讀〕：GEAR 的低秩是「修量化誤差」，不是「另一種壓縮」；它證明的是「2-bit 加一點低秩殘差 ≫ 2-bit」，不是「疊越多越好」。

#### S2　KVTC〔全文〕
- **出處**：Konrad Staniszewski, Adrian Łańcucki（NVIDIA）。arXiv [2511.01815](https://arxiv.org/abs/2511.01815)（v2，2026-03-11），arXiv comment「Accepted to ICLR 2026」。L2 B3 已有卡片，這裡只補組合相關的部分。
- **順序**：最前 4 個 sink 與最近 128 個 token 不壓（壓了在 64× 會整個崩，圖 4、附錄 B.3）→ 拿掉 RoPE（「位置嵌入會扭曲 key 的低秩結構」）→ 多層串接後 PCA（離線校準一次，約 200K token、H100 數分鐘）→ DP 對每個主成分分位元（很多尾端 0 bit，等於降秩）→ DEFLATE（nvCOMP，GPU）〔原文 §2–3〕。
- **組合各段的貢獻**：
  - 只刪主成分、不做 DP 量化（「-DPQ」）：長 context 任務大幅變差；作者：「量化是 kvtc 不可缺的一段」「DEFLATE 在量化後的資料上有效得多」〔原文 B.9〕。
  - 無損段：DEFLATE 平均在量化之上再 **~1.23×**〔原文 §4〕。Mistral NeMo 12B、kvtc32× 時：Identity 29.6–32.4×、DEFLATE 34.7–42.9×、zstd 34.6–46.3×、LZ4 29.5–35.5×〔原文表 14〕；RULER-VT 特別好壓，作者推測是重複的填充文字〔原文 B.8〕。→ DEFLATE 相對 Identity 是 1.17–1.33×〔算術：34.7/29.6、42.9/32.3〕。
  - 跨層串接越多層 PCA 越好（1 層 → 4 層：Llama GSM8K 27.5 → 53.3，用 -DPQ、窗口 16）〔原文表 16〕。
- **和單一方法比**（Llama 3.1 8B，表 2）：GEAR 2-bit、KIVI 2-bit 都是 5×：GSM8K 52.8／52.8、LITM 96.9／88.8；kvtc16×（DEFLATE 後 18–22×）：56.9、99.3（vanilla 56.8、99.4）。H2O、TOVA（8×）在 LITM 是 20.2、1.2。→ **在更高壓縮比下勝過兩個 2-bit 量化**。
- **作者對「疊加」的立場**：「這些技術很脆弱，**準確率下降讓各種方法無法疊加出複合的好處**」〔原文 §1〕；又說 kvtc 不改 KV 結構、「可直接和丟 token 法相容」，但**沒有做**〔原文 §5〕。
- **長度**：RULER-VT、NIAH 用 8K（70B 的 NIAH 用 100K）；校準資料 1–32K；附錄 B.16 用約 64K 的多 client vLLM＋LMCache 測〔原文〕。
- **代理指標**：見 §3（表 8 的 Pearson）。
- **還原成本**：H100、Mistral NeMo 12B、BS 8、8K：壓 379 ms、解 267 ms（投影 156、反量化 37、DEFLATE 64）；BS 2、16K：194／143 ms〔原文表 5〕。多 client（附錄 B.16，同一張 GPU 做壓縮與解壓）見 §4。

#### S3　CacheGen〔部分：PDF 抽字＋repo P02 卡〔全文〕〕
- **出處**：Yuhan Liu 等 14 人。arXiv [2310.07240](https://arxiv.org/abs/2310.07240)，SIGCOMM'24（abs comment 核過）。機制、程式碼落差見 `docs/precision_20261013/cards/P02_CacheGen_mechanism.md`。
- **順序**：每 10 個 token 一組，存對 anchor 的 delta → 依層分三段、淺層較精的量化 → 依 channel×layer 機率表做算術編碼（比全域單一分布最多再小 53%）〔P02，原文 §5.2〕。
- **和單一方法比**（Mistral-7B，LongChat，原文表 1）：8-bit 量化 622 MB、準確率 1.00；CacheGen 176 MB、0.98；**H2O 282 MB、0.97；CacheGen 疊在 H2O 上 71 MB、0.97**；LLMLingua 492 MB、0.94；疊上 CacheGen 183 MB、0.94。同品質下比量化基線少 3.5–4.3× 頻寬〔原文 §1〕。
- **注意**：H2O 用的是「理想版」——離線壓縮時用了 prompt 的 query〔原文 §7.2〕。context 1.4K–16K。
- **結論**：**加乘**（丟 token × 量化 × 熵編碼三件），但 ≤16K、而且位元組不同的比較。

#### S4　Palu〔部分〕
- **出處**：Chi-Chih Chang 等 10 人。arXiv [2407.21118](https://arxiv.org/abs/2407.21118)（v2）；ICLR 2025（iclr.cc poster 29993 與 proceedings 頁，10/20 搜尋核過）。
- **順序**：對 W_K、W_V 做（分組）低秩分解，快取 latent → Walsh-Hadamard（吸收進分解後的權重，執行時不多花）→ 逐 token 非對稱整數量化〔原文 §3〕。
- **為什麼要 Hadamard**：「低秩分解會在 latent 裡**造出**嚴重的離群值……SVD 把大的特徵值排在前面，latent 值快速遞減，撐開分布、傷量化」〔原文 §3.4，圖 4〕。→ 機制見 §2.1。
- **和單一方法比**（LongBench，Mistral-7B-v0.2，原文表 3）：基線 42.54；Palu-30%（1.43×）41.55；Palu-50%（2×）36.23；KIVI-2（5.05×）41.45；**Palu-30%＋3 bit（7.59×）約 40.77**。→ 只降秩很差；降秩＋量化在「同準確率」下比 KIVI-2 多省 ~1.5 倍〔算術：7.59/5.05〕。
- **第三方在我們兩個模型上的數字**（AATC 表 II〔部分〕，Palu 30% 降秩＋4 bit＋Hadamard，5.39×）：Llama-3.1-8B-Instruct 推理掉很多（GSM8K 0.617 vs KIVI 5× 0.730、基線 0.789）；Qwen-2.5-7B-Instruct 在 KIVI 崩掉時 Palu 好很多（LongBench 0.483 vs 0.351；RULER-32k 0.625 vs 0.233），但仍低於 KVQuant（0.558、0.643）。→ **低秩＋量化 vs 單純量化，誰好依模型而變**。
- **還原**：RoPE 模型的 K 要線上重建再套 RoPE；Palu 的 4-bit 版本在 RoPE 模型上**不量化 K**（kernel 只支援 FP16）〔原文 §4.5〕。L5a：發布版不支援 Llama-3.1、GQA 形狀會報錯。

#### S5　MiKV〔全文〕
- **出處**：June Yong Yang 等 8 人（KAIST、NAVER Cloud、POSTECH）。arXiv [2402.18096](https://arxiv.org/abs/2402.18096)（只有 v1，2024-02-28）。**venue：NOT_FOUND**（arXiv 頁沒寫；搜尋也沒找到正式發表）。
- **順序**：H2O 類重要度 → 重要的 token 存高精度（FP16／INT8／INT4），「本來要丟的」存低精度（INT3／INT2）；低精度用「query-key 通道平衡」處理離群通道：把 √(max|q|/max|k|) 乘進 K、除進 Q，分組大小用 head 維度的一半（減輕 RoPE 的配對效應）〔原文 §3〕。
- **關鍵數字**（line retrieval，20 行，很短）：
  - 只留 20% 重要的、其餘丟掉：準確率 4.0%；其餘改存 INT2：64.0%（快取 32%）；INT2＋通道平衡：92.6%（33%）；重要的也降到 INT4：92.0%（18%）；降到 INT2：65.0%（16%）〔原文表 1–3〕。
  - 附錄 C：K 逐 channel 量化（模擬）在 INT2 時 99.2%，比通道平衡（92.6%）更好；但「從逐 channel 分組量化的張量裡丟掉一個 KV 不簡單，因為 tile 大小會變得不一致，丟 token 的規則必須改」〔原文附錄 C〕。→ **KIVI 式 K 量化 × 丟 token 的干擾**，§2.2。
- **同位元組比較**：圖 6 的壓縮率－準確率曲線（GSM8K、HumanEval、line retrieval、MMLU）對 H2O、RTN；MiKV 在 20% 快取時仍保住 GSM8K／HumanEval，RTN「掙扎」〔原文 §4.2，數字只在圖上〕。
- **反面證據**：ZipCache 在 4.69–4.98× 時量到 MiKV 掉很多（LLaMA3-8B GSM8K 35.48 vs ZipCache 53.75、FP16 55.88），原因是「累積 attention 分數挑錯顯著 token」〔ZipCache 原文表 1〕。
- **結論**：「被丟的改存低精度」比丟乾淨好（加乘），但重要度怎麼挑決定成敗。

#### S6　ZipCache〔部分〕
- **出處**：Yefei He 等 6 人。arXiv [2405.14256](https://arxiv.org/abs/2405.14256)；NeurIPS 2024（nips.cc poster 96563，搜尋核過）。
- **順序**：用「正規化 attention 分數」（只算 10% token 的 attention，其餘可走 FlashAttention）找顯著 token → 顯著 4 bit、其餘 2 bit，channel-separable 量化〔原文 §4〕。
- **數字**（GSM8K，平均輸入 840；原文表 1）：LLaMA3-8B：FP16 55.88；H2O（2.50×）27.82；GEAR 4/4（3.00×）49.43；KIVI 16/2（3.46×）47.99；MiKV 4/2（4.69×）35.48；ZipCache 4/2（4.69×）53.75。
- **結論**：混合精度（軟性丟 token）＋好的顯著度 ＞ 單一量化，但只在短 context。SemKV（2026〔摘要〕）提出相反的讀法：在「品質懸崖」（Llama-3.1-8B-Instruct 約 2.0–2.32 bit）以上，8 種重要度指標統計上沒差，混精度的好處只是**補出均勻量化做不到的平均位元數**——和我們第二階段「放哪裡都和隨機一樣」（08 §0）是同一件事。

### 1.3 最接近「系統性比較組合」的兩篇（〔全文〕）＋兩篇補充

#### S8　More Tokens, Lower Precision〔全文〕
- **出處**：Jiebin Zhang 等 9 人（北京大學、華為）。arXiv [2412.12706](https://arxiv.org/abs/2412.12706)（v2）；EMNLP 2025 Findings（ACL Anthology 2025.findings-emnlp.429，搜尋核過）。程式碼 `github.com/zhzihao/QPruningKV`。
- **設計**：先用丟 token 法（StreamingLLM、H2O、SnapKV、PyramidKV、Ada-KV、HeadKV）留下 s 個 token，再把留下的量化（KIVI 為主；FlexGen、KVQuant、加 1% 離群值）〔原文 §3〕。同記憶體比三種：1× token×16 bit、2×8 bit、4×4 bit〔原文圖 1〕。scale 的額外位元組沒有算進預算（group 64，預設 HQQ 實作）〔原文附錄 B〕——**所以「同記憶體」是近似**〔判讀：4 bit＋group 64 的 FP16 scale／zero 約 +0.5 bit／值〕。
- **主要結果**：
  - 量化到 8 bit 幾乎無損、4 bit 輕微、**2 bit 全面大跌**（例如 Llama-3-8B、2048 token：SnapKV LongBench 41.7／41.7／41.0／22.9 對應 16／8／4／2 bit）〔原文表 1〕。
  - 「同樣省 4 倍，砍 token 傷得多、砍精度傷得少」〔原文 §5〕。
  - 1/16 預算、PyramidKV、Llama-3-8B-Instruct：RULER-8k 512×16 bit 67.5、1024×8 bit 74.9、2048×4 bit 82.2〔原文表 2〕。
  - 高預算時 8 bit×2 倍 token 略好；低預算（1/128）時 4 bit×4 倍 token 最好〔原文 §5〕。
  - 偏推理的任務（Code、Synthetic、Multi-doc QA）「換更多 token」的好處小，有時更少 token、更高精度反而好〔原文表 2、§6.1〕。
  - 逐 head 給不同 token 數的丟法（Ada-KV、HeadKV）和 KIVI 不相容，只能用 FlexGen，2 bit 時崩得更厲害〔原文表 1〕。→ §2.2。
  - 中間層「多 token、低精度」最重要；首尾層影響小〔原文 §6.4〕。
- **長度**：Llama-3-8B-Instruct 只有 8K（超過的截頭尾）；Mistral NIAH 到 32K；RULER 4K／8K／16K〔原文附錄 A〕。**沒有 64K–128K**。
- **缺的對照**〔判讀〕：沒有和「全部 token、低精度（只量化）」在**每個**預算下比——1/4 預算時 4×4 bit 本身就是只量化；1/8 以下只量化得用 2 bit，而 2 bit 已知崩，所以「低預算時組合贏」很大程度是因為單一量化到不了。
- **結論**：**看預算**。

#### S10　JoLT〔全文〕
- **出處**：Rahul Krishnan, Volker Schulz（Universität Trier）。arXiv [2607.12550](https://arxiv.org/abs/2607.12550)（v4，2026-09-24），comment「Under review at ICLR 2027」。程式碼 `github.com/rahulk98/JoLT-Master-Thesis`。
- **設計**：每組相鄰層的 K 或 V 當 4 階張量（層×head×token×feature），只在 token 與 feature 兩個 mode 做 partial Tucker；截斷的殘差做隨機正交旋轉後存 0／2／4／8 bit；用一個 Lagrangian 乘數在**同一個位元組預算**下聯合分配 rank 與殘差位元〔原文 §3〕。K 在 RoPE 前分解，因為「RoPE 後的 K 在同儲存下重建誤差高 48–62%」〔原文 §3，引同作者 2609.28029〕。
- **同位元組比較**：
  - 2× 時：JoLT 困惑度 +0.01%，最好的低秩基線（KQ-SVD）+5.27%；但 **int8／fp8 在 2× 也近無損**，只是沒有 >2× 的工作點〔原文表 2〕。
  - **約 4× 時，單純量化贏組合**：Mistral 困惑度 TurboQuant +0.28%、KIVI-2 +1.70%、JoLT +5.65%〔原文表 2〕；GSM8K（Mistral，同 int4 位元組）KIVI-2（3.6×）40.18、TurboQuant（3.8×）38.89、int4 per-channel（4.0×）37.07、**JoLT（4.1×）35.94**、不壓 40.56〔原文表 8〕。
  - 作者自己的結論：「JoLT 的優勢只在 2× 以上、約 4× 以下——也就是固定位元量化沒有工作點的那一段」〔原文 §4.2〕。
  - 拿掉殘差（只降秩）2× 困惑度從 6.54 變 7.27，「加 rank 補不回來」；殘差位元到 4 bit 就飽和〔原文 §4.4〕。
- **長 context**：Llama-3.1-8B、RULER 64K（4 個任務、3 seed、每組 600 列）：2× +0.23、3× −0.10、4× −0.90、5× −2.40 個百分點；同一個 5× 的困惑度卻 +51.27%——作者：「困惑度與 RULER 量的是模型品質的不同切面」〔原文 §4.3〕。Mistral 4× 的懲罰隨長度變小：4K −6.10、32K −0.20〔原文表 9〕。
- **還原成本**：融合 Triton kernel 在 A100、batch 1 比不壓的 SDPA 慢 7.5–65.9×；對「先重建再 attention」是 0.7–2.2×；RULER NIAH 8K 端到端 20.7 s（2×）vs 1.6 s（不壓）。「主要瓶頸是 RoPE：位置相關的旋轉和 token mode 分解不可交換，K 必須每個 tile 完整重建」〔原文 §4.5、§5〕。
- **代理指標**：附錄 E 有跨方法的重建誤差與困惑度（§3.1）。
- **結論**：**看預算**——而且它給了「組合只在單一量化沒有工作點的區間贏」最乾淨的證據。

#### S9　RDKV〔部分〕
- **出處**：Junkai Zhang, Hang Guo, Luca Benini, Yawei Li（ETH Zurich、清華）。arXiv [2605.08317](https://arxiv.org/abs/2605.08317)（v1，2026-05-08），沒有 venue。
- **設計**：V 以 token 為單位、K 以 channel 為單位；丟一個 token 對 attention 分布的 TV 距離＝它收到的 attention（＝累積 attention 分數），丟一個 K channel 的 logit 偏差範數＝‖Q[:,c]‖‖K[:,c]‖/√d；用這些權重做反向注水，每個單位分到 {0,2,4,8,16} bit，0 就是丟〔原文 §3.1〕。預填後做一次。
- **同預算消融**（Llama-3.1-8B-Instruct，LongBench 平均；同一組權重與分配器）〔原文表 4〕：

  | 動作集 | 128L | 512L |
  |:--|--:|--:|
  | 只量化 | 39.06 | 40.59 |
  | 只丟 | 42.53 | 47.18 |
  | 三態 {0,4,16} | 46.91 | 49.05 |
  | 聯合 {0,2,4,8,16} | 47.43 | 49.22 |
  | 不壓 | 49.82 | 49.82 |

  作者：「拿掉丟（只量化）的代價最大——分配器得在每個 token 上花位元，把關鍵少數的精度稀釋了」。
- **預算有多緊**：B＝128L 是每層「相當於 128 個 FP16 token」；摘要說 LongBench 用 2.48% 保留率就拿回 97.81%〔原文〕。→ **這是「單一量化不可能到」的區間**（每值 <1 bit）〔判讀〕。
- **長 context**：RULER 到 128K：RDKV 67.0、AdaKV 58.4〔原文 §4.2〕；A100 64 GB 上 128K 解碼 4.5×、峰值記憶體 1.9× 少〔原文〕。
- **還原**：TriZone 打包解碼 18 ms／token vs 先解成 FP16 再 FA2 的 36 ms〔原文表 5〕。
- **結論**：**加乘**，但只在極緊的預算驗證；分配後凍結、不隨解碼更新〔原文附錄 K〕。

#### S11　AATC〔部分〕
- **出處**：Hannah Laus, Claudio Mayrink Verdun, Hao Wang, Flavio du Pin Calmon, Felix Krahmer。arXiv [2608.14191](https://arxiv.org/abs/2608.14191)（v1，2026-08-14），沒有 venue。
- **為什麼列在這**：它在**我們兩個模型**（Llama-3.1-8B-Instruct、Qwen-2.5-7B-Instruct）、RULER 4K–32K 上，比了單一量化（KIVI 5.00×、KVQuant 5.91×）、低秩＋量化（Palu 5.39×）、與自己的「轉換編碼」（5.82×）。
- **理論**：在白雜訊量化模型下，attention 輸出的期望失真分成 K、V 兩項，各自「token 權重 × channel 權重」可分解：D_K＝(Σᵢ aᵢ²‖vᵢW_O − o‖²)·(Σ_c q_c² E[δk_c²])，D_V 類似〔原文定理 1、式 (5)〕。表 I 把丟 token（aᵢ²）、輸出相關（‖vᵢW_O − o‖²）、query 感知 K 精度（q_c²）、W_O 感知 V 精度、逐 channel 敏感度歸成不同的因子——**等於說：這些方法在理論上是「乘在不同因子上」，可以疊**〔判讀〕。
- **數字**（表 II）：Llama：KIVI 5× RULER-32k 0.763（基線 0.775），幾乎無損；**Qwen：KIVI 5× 崩**（LongBench 0.351 vs 0.574、RULER-32k 0.233 vs 0.720）。和我們第二階段一致：Qwen 的 INT2 KIVI 在 32K 掉 12.9 個百分點（`tables_longctx.md`）。
- **代理指標**：見 §3（只用變異數分位元 vs attention 感知）。

#### S12　2604.11501：同儲存的「量化 vs 降秩（含混合）」比較——**已被作者撤回**〔全文，v1 與 v2〕
- **出處**：Samuel Salfati（fraQtl AI Research）。v1（2026-04-13）題目 *Quantization Dominates Rank Reduction for KV-Cache Compression*；v2（2026-09-08）改名 *Attention-Weighted Value Projection for KV-Cache Compression*。
- v1 宣稱：同儲存下量化在 5 個模型都贏降秩，「降秩後再量化留下的維度（hybrid）也追不上」（Mistral 512 bit：全維 INT4 9.42、rank-64 INT8 13.06、rank-32 FP16 44.19）。
- **v2 附錄 A 明寫撤回**：v1 的實驗是「把權重做 rounding／投影」，「不是 packed 的快取量化」，「不能確立同快取儲存下哪個 codec 比較好」；LAMBADA 的評測器會對錯目標 token；「r＝0.99 的 attention KL 相關」等詮釋也一併移除。
- **處理**：本檔**不把它當證據**。列出來是因為搜尋會找到 v1 的結論，而它是最常被引用的「量化 vs 低秩混合」同預算比較——**被撤回這件事本身就說明這種比較很容易做錯**（位元組怎麼算、改權重還是改快取、評測器是否對齊）。這是 §6 要把「位元組照實算」與「用快取層級的編解碼」寫死的理由之一。

### 1.4 其他組合的重點（一兩句）

- **丟＋量化**：
  - Q-Hitter（MLSys'24〔部分，PDF 抽字〕）：低位元（≤4 bit）量化會「明顯降低生成期間挑 heavy hitter 的準確度，特別在深層」；H2O 挑出的 token「不一定適合量化」——只有 top 1% 的 heavy hitter 有量化友善的性質，遠少於 H2O 需要保留的 20%（原文 §1）。用 λ＝0.5 線性組合「累積 attention」與「量化誤差」來挑。
  - MiniKV（ACL Findings'25〔部分〕）：丟 token 會讓子 channel 分組裡的元素在每一步改變 → 改成「預填時選定、整段生成不再更新」的持續選擇（60–80% heavy hitter 會持續）；另寫兩段式 selective flash-attention 解決「要 attention 分數」和 FlashAttention 不相容。
  - ThinKV（ICLR'26 Oral〔部分〕）：推理模型上「激進量化會讓生成變長，吃掉省下的記憶體、又傷準確率」；丟 token 不會讓輸出變長；混合才在 Pareto 前緣。
  - ReST-KV（ICLR'26〔部分〕）附錄 I：丟 token＋4-bit 達總 6.25%，仍比「整份 2-bit KIVI／KVQuant」準（Llama3.1-8B-Instruct，LongBench；數字在圖上，未讀）。
  - DefensiveKV（arXiv〔部分〕）附錄 E：留 40%＋int4 → 約 10% 記憶體，LongBench 49.21 → 48.55。
  - KVzip（NeurIPS'25 Oral〔部分〕）：LLaMA3-8B-W8A8KV4（QServe）124K 輸入、丟 70%：16.3 GB → 1.2 GB，「損失可忽略」〔算術：約 13.6×〕。
  - DiffKV（SOSP'25〔摘要〕）、TailorKV（arXiv〔部分〕）、QEvict（L1 A22）、SemKV（〔摘要〕）：見總表。
  - Not All Bits Are Equal（〔摘要〕）：推理模型、有效模型小時丟比量化好。
- **低秩＋量化**：SVDq（L5a：Qwen2.5-7B-Instruct 64K RULER 只壓 K、3 bit：直接逐 channel 39.9、SVDq 69.7、不壓 70.6）；STAR-KV（ICML'26〔依 JoLT 參考文獻，未直接查〕）；ReCalKV（L5a）；xKV＋4-bit（L5b）。
- **跨層／合併＋量化**：MiniCache、XQuant、AQUA-KV、Joint encoding（總表）。L5b 的警告：在 Llama-3.1-8B、Qwen2.5-7B-1M、長 context 上，逐 token 的相鄰層相似度第三方實測幾乎都不成立；合併「丟掉的 token」到留下的，加分很少或反而變差（GRKV、SelKV）。
- **丟＋低秩**：ShadowKV、Loki（低秩 K 當索引）、SVDq＋key 稀疏、ThinK（K channel 剪枝＋丟 token 或 KIVI）。LESS（ICML'24，arXiv [2402.09398](https://arxiv.org/abs/2402.09398)〔摘要〕）用低秩狀態補回被丟的 token，但要訓練小模組 → 本次不可行。
- **現成工具**：kvpress 有 `ComposedPress`（「把多個 press 的 forward hook 串起來」）與 transformers 的 `QuantizedCache`〔README〕，但 README 沒說兩者能否同時用；L1 已記錄它的逐 head press 不省峰值記憶體、也沒在 ROCm 驗證。

### 1.5 讀出來的規律〔判讀〕

| 組合 | 加乘的證據 | 抵消／沒好處的證據 | 一句話 |
|:--|:--|:--|:--|
| 丟＋量化 | S8、S9、S13、S14、S15、S16、S17、S6 | S7（天真疊加）、SemKV（懸崖以上只是內插）、S8（2 bit 崩、逐 head 丟法和 KIVI 不相容） | **預算緊時加乘**；量化 ≥4 bit 是地板；挑 token 要考慮量化 |
| 低秩＋量化 | S4、S10（2–3×）、S11、SVDq、S21、S2 | S10（約 4× 輸單純量化）、S11（Palu 在 Llama 推理掉很多）、S12（撤回，不算） | **只在單一量化沒有工作點的位元組區間贏**；要「每個成分分位元」而不是「逐 token 跨成分量化」 |
| 量化＋無損 | S2（~1.23×）、S3（AC 比全域分布再小 53%）、SplitZip（FP8 E5M2 再 1.14×，L2 B1） | SplitZip FP8 E4M3 用 top-8 反而變大（L2 B1）；§2.4 模擬：INT4 g32 ≤1.06× | **量化越粗、分布越尖，無損越有用** |
| 跨層／合併＋量化 | S18、S19、S20、S21、S27 | L5b：Llama-3.1-8B／Qwen-1M 長 context 上逐 token 合併很差；合併被丟 token 加分小 | 子空間層級（xKV）可以疊；逐 token 合併在我們的模型上前提不成立 |
| 三件以上 | S1（CoT 加乘）、S2（2 有損＋1 無損）、S3（丟＋量化＋AC） | S1（LongBench 三件 < 兩件）；KVTC：「天真疊加不能複合」 | **同位元組、≥32K、三件有損：NOT_FOUND** |

---

## 2. 機制：哪些組合理論上會互相干擾

### 2.1 低秩 × 量化：低秩後的係數分布

| 現象 | 出處 | 對組合的意思 |
|:--|:--|:--|
| SVD／PCA 把大特徵值排在前面，latent 值快速遞減，前幾維像「離群值」，撐開「逐 token 跨維」量化的範圍 | Palu §3.4、圖 4〔部分〕 | 低秩後若用逐 token 量化，**要先旋轉**（Hadamard，Palu 吸收進權重） |
| 換成「每個潛在通道各自給位元」，誤差反而比原空間逐 channel 量化小（作者稱 ×0.1 或更低，理論＋實驗） | SVDq 摘要與 §4.3〔部分〕；L5a 的 Qwen 64K 數字 | 同一個「前幾維很大」的性質，在**分位元**時是優點、在**均勻量化**時是缺點 |
| DP 分位元時很多尾端主成分拿 0 bit；DEFLATE 在量化後的資料上「有效得多」 | KVTC §3.2、B.9〔全文〕 | 「降秩」和「量化」在轉換編碼裡是同一個分配問題（0 bit＝降秩） |
| 殘差用隨機正交旋轉攤平後再低位元量化；殘差位元 4 bit 飽和 | JoLT §3、§4.4〔全文〕 | 旋轉放在「降秩之後、殘差之上」，避免打亂主成分的變異數排序 |
| **KIVI 的 K 逐 channel 量化 vs 低秩投影**：KIVI／KVQuant 對 K 逐 channel 給 scale，是因為 K 有固定的大值 channel（GEAR §2 轉述 KIVI、KVQuant〔全文〕；本專案 Qwen 的 K 固定偏移，08 §2.4）。PCA 會把這些 channel 的能量轉進前幾個成分（不中心化時集中在平均方向），原本的離群 channel 不再是座標軸 | 〔推論〕；SVDq、KVTC 的實驗支持「投影後改成逐成分給位元」 | 投影後若仍沿 token 分組、逐「成分」給 scale，就是把 KIVI 的做法搬到新座標（好）；若改成逐 token 跨成分，就回到 Palu 的離群問題（壞）。中心化 PCA 另外把 Qwen 的固定偏移拿掉（等於第二階段的 fp8k0） |
| **旋轉與分位元的衝突**：Hadamard／隨機旋轉會把各成分的變異數攪勻，正好抹掉「依變異數分位元」需要的排序 | 〔推論〕，由 Palu（旋轉）與 KVTC／SVDq（分位元）對照得出 | 疊的順序要二選一：要嘛「降秩＋旋轉＋均勻位元」，要嘛「去相關＋依成分分位元」；兩者都疊會互相抵銷一部分好處 |

### 2.2 丟 token × 量化

| 現象 | 出處 | 對組合的意思 |
|:--|:--|:--|
| 被重要度挑出的 token 和「量化誤差大」的 token 高度重疊；量化又讓之後挑 heavy hitter 挑錯（深層更嚴重） | Q-Hitter 摘要、§4.3〔部分〕 | 「丟完剩下的都是重要 token，量化誤差的影響變大」**有直接證據**；挑法要把量化誤差算進去，或先挑再量化（不要用量化後的 K 打分） |
| 丟掉一個 token 後 softmax 會重新正規化，其餘 token 的權重一起變大 | RDKV 式 (1)〔部分〕；KV-Surgeon 摘要（「softmax 正規化讓各個刪除決定非線性地耦合」） | 〔推論〕依 AATC 的 D_V ∝ Σ aᵢ²·E[δvᵢ²]：丟 token 讓留下的 aᵢ 變大，**同樣的量化雜訊在留下的 token 上被放大**；組合的誤差不是兩者相加 |
| KIVI 式 K「逐 channel、沿 32 個 token 分組」：組裡丟 token 會讓 tile 大小不一，丟的規則必須改 | MiKV 附錄 C〔全文〕；MiniKV §3.1〔部分〕 | 〔推論〕**以 512-token block 為單位丟**（＝16 個 32-token 組）不會破壞 KIVI 分組；逐 head 逐 token 丟則要重分組或改用逐 token 量化 |
| 逐 head 給不同 token 數的丟法（Ada-KV、HeadKV）和 KIVI 不相容，只能配 FlexGen，2 bit 時崩更兇 | More Tokens 表 1〔全文〕 | 同上 |
| 2 bit 以下品質懸崖：Llama-3.1-8B-Instruct 在 2.0–2.32 bit 之間崩，懸崖以上重要度指標可互換 | SemKV 摘要；More Tokens 表 1；本專案 INT2 KIVI（3 bit 含 scale）Llama 128K −7.2、Qwen 32K −12.9 | 組合裡的量化段**不要低於 4 bit（含 scale 約 5 bit）**，省位元組要靠丟／降秩 |
| 推理模型：激進量化讓輸出變長；丟不會 | ThinKV §2〔部分〕；Rethinking 表 4（KIVI-4、GEAR-4 輸出長度 ×1.69–1.70） | 時間成本要用「整段生成」算，不能只算每 token |

### 2.3 丟 token × 低秩

| 現象 | 出處 | 對組合的意思 |
|:--|:--|:--|
| PCA 降維時最前面的 token 重建誤差最大；sink 壓了在高壓縮比會整個崩 | KVTC 圖 6、附錄 B.3〔全文〕 | 〔推論〕丟 token 留下的正是 sink、高 attention 的 token，偏偏是低秩表示得最差的 → 兩者疊在一起，誤差集中在最重要的 token 上。KVTC 的解法是 sink 與最近窗口不壓 |
| 基底要在「全部 token」還是「留下的 token」上估？ | NOT_FOUND（沒看到有人比較） | §6 的順序對照要做 |
| 低秩 K 當成「挑 token 的索引」很常見（ShadowKV、Loki、SVDq＋稀疏） | L1、L5a | 這是另一種組合：低秩不存、只用來挑；和「低秩當儲存格式」要分開評 |

### 2.4 量化 × 無損：量化後的熵

- **文獻**：KVTC 的 DEFLATE 在 PCA＋DP 量化後再 ~1.23×〔全文〕；CacheGen 在 delta＋量化後用 AC 比全域分布再小 53%〔P02〕；SplitZip 只壓 BF16 指數 1.31–1.33×、FP8 E5M2 再 1.14×、E4M3 用 top-8 反而變大（L2 B1）。
- **機制**〔推論〕：有損量化把尾數裡的雜訊（本來就壓不動的部分）丟掉了，剩下的碼的熵取決於「分布和量化格子的形狀差多少」。每組範圍用 min-max 對齊時，組越小，碼越接近均勻、熵越接近位元數。
- **我的模擬**〔算術：合成資料，i.i.d.，只算碼、不含 scale；`nice -n 19 taskset -c 0-3`，20 萬個值〕：

  | 分布 | 每組 32、4 bit | 每組 128、4 bit | 每組 32、2 bit | 每組 128、2 bit |
  |:--|:--|:--|:--|:--|
  | 常態 | 熵 3.87 bit → 最多 1.03× | 3.63 → 1.10× | 1.73 → 1.15× | 1.47 → 1.36× |
  | Laplace | 3.77 → 1.06× | 3.40 → 1.18× | 1.66 → 1.21× | 1.35 → 1.48× |
  | t(3) | 3.85 → 1.04× | 3.55 → 1.13× | 1.70 → 1.18× | 1.41 → 1.42× |

  → **我們的 INT4 KIVI（32 一組）疊無損，order-0 上限只有約 1.03–1.06×**；INT2、128 一組、或 PCA 尾端的低位元成分才有 1.15–1.5×。真實 KV 有「相鄰 token 相似」（CacheGen 用 delta 抓這個），order-0 熵看不到，所以真實比可能更高——**NOT_MEASURED**，S1（`m16_s1_lossless.py`）已經可以量。這和 L2 X1 的預測「INT4 KIVI 打包後 ≈1.0×」一致。
- **丟 token × 無損**〔推論〕：逐 head 逐 token 丟要存索引（每個留下的 token、每個 head 約 log₂T≈15–17 bit），會吃掉一部分省下的位元組；block 層級的丟只要一個遮罩。

### 2.5 RoPE

| 現象 | 出處 | 對組合的意思 |
|:--|:--|:--|
| K 在 RoPE 前才低秩；RoPE 後同儲存重建誤差高 48–62% | KVTC §3.1〔全文〕；JoLT §3（引 2609.28029）〔全文〕；L5a（Loki、ShadowKV） | 低秩段要在 RoPE 前做 |
| RoPE 和 token mode 分解不可交換，K 必須逐 tile 重建再旋轉，是解碼延遲的主因 | JoLT §4.5、§5〔全文〕；Palu §4.5（RoPE 模型的 K 不量化） | 「低秩＋量化」的還原成本**比單純反量化多一個 GEMM＋RoPE**，在 GPU 忙時會更明顯（§4、§6 第三階段） |
| RoPE 會複製離群值；分組取 head 維的一半可減輕 | MiKV §3.2〔全文〕；「因為 rotate_half 把第 j 與 j+D/2 維配對」是我的解釋〔推論〕 | 逐 token 量化 K 時，分組邊界要對齊 RoPE 配對 |
| 量化在 RoPE 前做（KVQuant）比較好 | KVTC §6 引 KVQuant〔全文〕 | 〔推論〕Qwen 的 K 有固定大偏移（第二階段 K3：FP8 −32.8 → 先減平均 +0.4，08 §2.4）。RoPE 會把這個偏移轉成隨位置變的向量，所以**去平均在 RoPE 前做最自然**；中心化 PCA（KVTC 存 μ）順便做掉這件事 |
| 同一位置的 K 在不同層受同一個旋轉，跨層 cosine 在 RoPE 前後相同 | `code/m16_kvstruct.py` 檔頭說明〔程式碼〕 | 跨層合併／共用不受 RoPE 影響；層內低秩才受影響 |
| 本專案的 dump 存的是 RoPE **後**的 K | `m16_kvstruct.py` 檔頭（m7_model.py:105–106）〔程式碼〕 | CPU 掃描要先逆旋轉（`m16_kvstruct.py` 已寫、有規則 6 的自我核對） |

### 2.6 合併／跨層 × 其他〔判讀＋推論〕

- 跨層共用量化碼（XQuant）：兩層共用同一組碼時，量化誤差在層間**相關**而非獨立，可能沿殘差流同向累積〔推論〕；它用各層各自的 scale／zero 減輕。
- 預測後量化殘差（AQUA-KV）：殘差比原值小，所以同位元下更準——這是少數「先壓一種、讓另一種更好做」的正向耦合〔部分〕。
- 合併 token 後的平均向量變異數變小，量化時 scale 較小〔推論〕；但 L5b 記錄的 GRKV／SelKV 顯示「合併被丟的 token」本身加分很少，盲目合併比只丟還差。

### 2.7 小結〔判讀〕

疊加不是「誤差相加」：(1) 丟 token 會放大留下 token 的量化誤差；(2) 重要 token 正好是最難量化、最難低秩表示的；(3) 分組量化的分組結構限制了丟的粒度；(4) 旋轉和分位元互相抵銷；(5) 低秩把還原成本從「反量化」變成「GEMM＋RoPE」。**所以「各自調到最好再疊」幾乎一定不是最好的組合**——這也是 RDKV、JoLT、KVTC、AATC 都改成「同一個預算下聯合分配」的原因。

---

## 3. 便宜的代理指標（只用 KV 和少量 query、在 CPU 上算）

### 3.1 文獻證據

| 代理指標 | 能不能預測答對率 | 出處 |
|:--|:--|:--|
| **K／V 相對 Frobenius 重建誤差** | **同一方法家族內**：好（KVTC 表 8，Mistral NeMo 12B，3 個壓縮比 × 4 種校準量＝12 點：K 誤差對平均分數 Pearson −0.948、V −0.939；困惑度對平均分數 −0.992）。**跨家族**：會整個排反 | KVTC 附錄 B.5〔全文〕；作者自己也寫「這個指標不保證任務層級的好處，預測力可能依任務而異」〔原文 §5〕 |
| 同上，跨家族 | LLaMA-2-13B、約 4×：KIVI-2（4.20×）K／V 誤差 0.201／0.383，困惑度 +1.59%；int4 per-channel（3.97×）0.077／0.124，+8.08%；FlashJoLT 0.120／0.087，+14.16%；TurboQuant（3.82×）0.227／0.097，+0.25%。fp8 e4m3 的 K 誤差是 int8 的 6 倍，困惑度仍在 0.04% 內 | JoLT 附錄 E 表 15、16〔全文〕 |
| **困惑度** | 不一定代表檢索任務：JoLT 5× 時困惑度 +51%，RULER 64K 只掉 2.40 個百分點 | JoLT §4.3〔全文〕 |
| **只看變異數分位元（＝最小化重建誤差）** vs **attention 感知失真（含 q_c²、W_O、aᵢ²）** | 中等壓縮（2.5 bit）兩者打平；2 bit 時只看變異數的版本 RULER 隨長度掉，attention 感知的撐住（兩個模型） | AATC §VI-B、圖 1〔部分〕 |
| 正規化有效秩（低秩可壓性） | 和 SVD 截斷後的效能下降「強相關」 | KV-CoRE 摘要（2602.05929）〔摘要〕 |
| attention KL | v1 稱和困惑度 r＝0.99（三種方法），但 **v2 已撤回相關詮釋** | 2604.11501〔全文〕；不採用 |
| 平均答對率本身 | 平均分數小差不代表每題都沒事（負樣本）；保住答案不代表保住推理證據 | Rethinking §4.4〔部分〕；Does Accuracy Equal Evidence?（2608.01631）〔摘要〕 |
| **本專案：teacher forcing 的 KL** | 海選勝出格子的 KL 比隨機低 31–37%（中位數），第一個答案 token 的答對率只差 0–0.5 個百分點；複賽 0／12 成立（08 §2.9）。**但** Qwen FP8 KL 3.84（Llama 0.0035）對應 −32.8 個百分點；壞掉的格式 KL 7–13 | 〔實測〕`results/m10_sweep_mi300x/analysis/`、`results/m10_precision_mi300x/`（08 §2.3、§2.9） |

### 3.2 判讀

1. **沒有一篇系統性驗證「CPU 上算得出來的代理指標，跨方法家族能不能預測答對率」**：NOT_FOUND。KVTC 只在單一家族內算相關；JoLT 的表格顯示跨家族會排反；AATC 證明 attention 感知的分配比較好，但沒有拿代理指標去預測答對率。
2. **最可能可靠的是「attention 輸出誤差」**〔推論〕：它同時看到 K 的誤差（經過 softmax，含 Q-Hitter／2604 v1 說的「路由翻轉」）、V 的誤差、以及丟 token 後的重新正規化（§2.2）；而 Frobenius 誤差看不到哪些 channel 被 query 讀、哪些 token 被注意。KVTC 附錄 B.4 也說 K 的精確度對檢索比較重要（V 可以壓得比 K 狠）。
3. **任何每層獨立算的代理都看不到三件事**：誤差跨層累積、生成長度改變（ThinKV、Rethinking）、答案以外的推理證據。→ 第二階段必須量**生成答對率**，代理只能拿來縮小範圍。
4. **本專案第二階段的教訓直接適用**：KL 抓得到「崩」，排不出「近乎無損之間的差」。所以第一階段的代理要先在**已知答案**的配置上校準（§6.2 第 1 步），並且分兩種用途：崩潰警報（門檻）與排名（相關係數）。

### 3.3 建議的代理階梯（全都只要 KV＋少量 query，CPU 可算）〔推論〕

| 代號 | 定義 | 要什麼 | 抓得到什麼 |
|:--|:--|:--|:--|
| P0 | K、V 各自的相對 Frobenius 誤差（K 在 RoPE 前、後各算） | KV | 只適合同家族內排序 |
| P1 | attention 分布的 KL／TV：softmax(qKᵀ) vs softmax(qK̂ᵀ)（丟掉的 token 權重設 0、重新正規化） | KV＋q | K 誤差、丟 token、路由翻轉 |
| P2 | attention 輸出相對誤差：‖softmax(qK̂ᵀ)V̂ − softmax(qKᵀ)V‖／‖softmax(qKᵀ)V‖，每 head | KV＋q | K、V、丟 token 的總效果 |
| P2w | P2 乘上 W_O（AATC 的 W_O 權重） | KV＋q＋W_O（safetensors 可用純 numpy 讀，不碰 torch） | 輸出投影後的失真 |
| P3 | top-k 重疊率／top-1 翻轉率 | KV＋q | 檢索型任務的「看錯地方」 |
| P4 | 答案區段保留率：問題 query 落在答案所在 token 的 attention 質量，壓縮前後的比 | KV＋q＋答案位置（T2／T3 合成題已知） | 檢索題最直接的代理 |

q 用兩種：context 最後 64 個 token 的 query（和 SnapKV 的觀察窗同義）、問題 token 的 query。Llama 每個 KV head 對 4 個 query head、Qwen 對 7 個（GQA）。

---

## 4. 疊加 × 分層／搬運：位元組越少，還原越貴

### 4.1 已有的東西

| 作品 | 組合 × 連線速度 | × GPU 忙 | 出處 |
|:--|:--|:--|:--|
| KVServe（SIGCOMM'26） | ✓：Transform×Quantizer×Codec 的組合依頻寬／SLO 選；損益兩平 B*＝(1−1/cr)·s | ✗：承認 GPU 負載會改變解壓吞吐，但只當殘差、由 EWMA 修正 | L2 B2、L5c〔全文〕 |
| CacheGen | ✓：逐 chunk 依量到的吞吐換等級，來不及就送文字重算 | ✗ | P02 |
| AdaptCache、EvicPress | ✓：壓縮法＋壓縮率＋丟＋放哪一層（DRAM／SSD／遠端） | ✗；EvicPress 把頻寬變動列為未來工作 | L5c〔全文〕 |
| ShadowServe、KVCodec | 只有量化＋DEFLATE／H.265 一種組合 | 量了干擾，解法是把解壓搬離 GPU（靜態） | L2 B4、B5 |
| **KVTC 附錄 B.16** | 一種組合（PCA＋量化＋DEFLATE），host 記憶體 128 GiB／GPU，約 64K 輸入 | 同一張 GPU 做壓縮與解壓：TTFT 4／6／8／10 client 時 1.2→1.7、2.0→2.7、2.8→3.5、3.4→4.5 s（**慢 25–42%**）；12 client 起 host 裝不下，不壓的要重算（136.6 s）、KVTC 5.6 s | 〔全文表 22；算術：1.7/1.2 等〕 |
| **ThinKV §5.1** | — | 重疊執行的 gather 壓實和下一層 attention 搶 HBM：attention 最多慢約 35%；不重疊時 TPOT 最多慢 37× | 〔部分〕 |
| **Rethinking（MLSys'25）** | — | GEAR、H2O 的 prefill 吞吐低於 FP16 且隨長度惡化；端到端延遲 GEAR 13.4 s vs FP16 11.4 s | 〔部分〕 |
| **JoLT §4.5** | — | 融合 kernel 比不壓慢 7.5–65.9×（batch 1）；作者：目前的價值是容量、不是速度 | 〔全文〕 |
| **RDKV 表 5** | — | 不打包、先解成 FP16：36 ms／token；打包解碼：18 ms | 〔部分〕 |
| **本專案第二階段** | INT4 對 BF16 的損益兩平 17.6–29.9 GiB/s；限速層 INT4 快 36–57% | FP8 反量化忙時慢 2.4–3.8 倍；融合後一次拆幾乎不受背景影響（INT4，duty 0.4／0.7／0.9：35.2／36.1／36.7 ms） | 08 §2.8、11 §3d、L5c〔實測〕 |

### 4.2 判讀

- **「組合 × 連線速度 × GPU 忙」三者一起看：NOT_FOUND。** KVServe 最接近，但沒有把 GPU 爭用放進模型；KVTC 附錄 B.16 是唯一一個「組合＋同 GPU＋多 client」的端到端數字，但沒有掃連線速度。
- 疊得越多，「每省 1 GiB 要多付的解碼時間」越大：反量化（逐元素）→ 低秩重建（GEMM＋RoPE）→ 熵解碼（序列性強，GPU 上要特別寫；ROCm 上只有 early-access 的 hipCOMP，L5c）→ 丟 token 的壓實（gather，搶 HBM）。〔推論〕所以「組合的最佳點」應該隨連線速度往慢層移、隨 GPU 忙往慢層移——**這是可以畫成一張決策圖、而且前人沒畫的東西**。
- 但 L5c 的預測要先講：在允許有損時，INT4 KIVI＋一次拆在 CPU 層閒／忙都最快，「自適應」對「事後最好的固定格式」可能約等於 0。疊加只在比 INT4 更省位元組時才有意義，而那正是解碼更貴的區間——**兩個效應方向相反，誰大要量**（§6 第三階段）。

---

## 5. 查新判定

### 5.1 已經被做過的

| 主張 | 判定 | 主要前作 |
|:--|:--|:--|
| 丟 token 和量化可以疊、而且在緊預算下加乘 | **已做（多篇）** | More Tokens（EMNLP F'25）、RDKV、MiniKV（ACL F'25）、ThinKV（ICLR'26）、ZipCache（NeurIPS'24）、MiKV、Q-Hitter（MLSys'24）、ReST-KV（ICLR'26）、KVzip（NeurIPS'25）、DefensiveKV、DiffKV（SOSP'25） |
| 低秩和量化可以疊 | **已做（多篇）** | GEAR、Palu（ICLR'25）、SVDq、STAR-KV、ReCalKV、JoLT、xKV＋4-bit、KVTC、AATC |
| 量化後再熵編碼 | **已做** | CacheGen（SIGCOMM'24）、KVTC（ICLR'26）、KVComp／PackKV、KVCodec、SplitZip（L2） |
| 跨層／合併和量化疊 | **已做** | MiniCache、XQuant（EMNLP'25）、AQUA-KV（ICML'25）、Joint encoding |
| 「同一個預算下聯合分配」取代「各自調好再疊」 | **已做** | RDKV（丟＋量化）、JoLT（降秩＋量化）、KVTC／AATC（降秩＋量化，0 bit＝降秩） |
| 同預算下比較兩種軸的取捨 | **已做（兩軸）** | More Tokens（token×精度）、RDKV 消融、JoLT 附錄 C／E |
| 依頻寬選壓縮組合 | **已做** | KVServe、CacheGen、AdaptCache、EvicPress（L5c） |

### 5.2 還空著的（＝可能的新意）

| # | 空白 | 搜尋結果 | 新意風險 |
|:--|:--|:--|:--|
| a | **三軸以上**（量化 × 低秩 × 丟 × 合併 × 無損）在**照實算的位元組**（含 scale／zero、索引、基底、窗口）、**≥32K**、**GQA 模型**上的 Pareto 前緣 | NOT_FOUND（最多兩軸；三件的只有 GEAR、KVTC、CacheGen，都 ≤16K 或主要 ≤8K） | 中：評測型貢獻；很可能得到「4 bit 地板＋丟／降秩補位元組」這條簡單規則 |
| b | **代理指標跨方法家族**能不能預測答對率 | NOT_FOUND（KVTC 只做家族內；JoLT 表格顯示會排反） | 中：小而乾淨；可以順便給「什麼時候不必上 GPU」的規則 |
| c | 組合的**還原成本**在 **GPU 忙**、**不同層的連線速度**下會不會翻轉最佳組合 | NOT_FOUND（KVServe 沒建 GPU 爭用；KVTC B.16 只有一種組合、沒掃連線） | **中低**：和研究地圖缺口 #1 同一條線，我們已有干擾的實測；但 L5c 預測有損時 INT4 支配，空間可能 <5% |
| d | Cake 式還原下**只壓載入區**的組合 | NOT_FOUND（L1 G1 的延伸） | 低：多半是乾淨的負面或簡單規則 |
| e | MI300X／ROCm 上的任何組合量測 | NOT_FOUND | 低（只是平台） |
| f | Qwen2.5-7B-Instruct-1M 在 64K–128K 的組合 | 只有 KVzip 用了這個模型（但量化組合在 LLaMA3-8B）、S4R／xKV 的低秩（L5a） | 低 |

---

## 6. 三階段探索設計（只寫設計，不寫程式）

### 6.0 前提

**預算與位元組帳**（相對同一段 token 的 BF16 位元組；「4×」＝0.25）：

| 預算 | 位元組比 | 我們手上能到的單一方法（第二階段已實作或已量） | 已知品質 |
|:--|:--|:--|:--|
| 參考 | 0.3125 | INT4 KIVI（每值 5.00 bit，含 scale） | 32K–128K 兩模型都不掉（`tables_longctx.md`） |
| **4×** | 0.25 | K4V2（`kv-int4_kivi-int2_kivi`，4.00 bit）；丟 75%（BF16）；降秩到 r＝32（BF16）；合併 g＝4 | K4V2：Llama 128K −0.1、**Qwen 32K −5.1 [−11.0, −0.1]** |
| **8×** | 0.125 | **沒有單一量化格式**（INT2 KIVI 是 3.00 bit＝0.1875）；只有丟 87.5%、降秩 r＝16、合併 g＝8 | INT2 KIVI（0.1875）：Llama 128K −7.2、Qwen 32K −12.9 |
| **16×** | 0.0625 | 只有丟 93.75%、降秩 r＝8、合併 g＝16 | — |

〔判讀〕所以「每個預算下最好的單一方法」要分家族各自取最好（量化、丟、降秩、合併各自掃自己的超參數：分組大小、rank、保留率、粒度），**不能拿一個固定設定當代表**——否則「組合贏」可能只是單一方法沒調好（2604.11501 被撤回的教訓）。

**位元組照實算**（寫死）：量化碼＋scale＋zero（FP16）＋KIVI 的全精度殘差窗口＋不壓的 sink／最近窗口＋丟 token 的索引（逐 head 逐 token：每個留下的 token、每個 head 記 ⌈log₂T⌉ bit；page／block 層級：每 block 一個遮罩位元）＋低秩係數＋基底（每個 prompt 線上算的基底算進該請求；離線校準、跨請求共用的基底只計一次、另列）＋合併的計數＋無損容器的額外位元組。block 0（sink）一律 BF16，照第二階段的規定。

**兩種範圍**：「全段」（block 1 以後全壓；和文獻可比）為主；「只壓載入區」（照 S2，block 32–63）為輔、只在第二階段用。

**資料**：
- 已有的程式：`code/m16_s1_kvdump.py`（KV dump，GPU）、`code/m16_kvstruct.py`（逆 RoPE、低秩譜、跨層、相鄰 token；numpy only）、`code/m16_s1_lossless.py`（無損，numpy＋zstandard）、`code/m16_s2_equalbytes.py`（同位元組、teacher forcing）、`code/m10_codec.py`／`m10_codec_x.py`（13 種格式）、`code/m10_drop.py`（丟 token 的分數）。
- 目前 `/mlsteam/data/tiara/` 下**還沒有任何 KV dump**（10/20 04:3xZ 查過）；S1 的匯出在 GPU 佇列裡排著（L5b）。

### 6.1 要先補的匯出（GPU，第一階段的前置）

| 項目 | 內容 | 為什麼 | GPU〔估計〕 |
|:--|:--|:--|:--|
| KV dump | 每個模型 32K × 3 種題（T2 合成撈針、T4 HotpotQA 自然文字、T5 常見詞抽取）× 全部層；128K × 1 題 × 5 層（照 kvdump 已有的 `--layers`） | 自然文字和重複填充的可壓性差很多（KVTC 附錄 B.8 的 RULER-VT 現象；kvdump 檔頭也這樣寫） | 已排的 S1 dump 加題，約 0.2 h |
| **Q dump**（新） | 每層、每個 query head：context 最後 64 個 token 與問題 token 的 query（RoPE 後）；另存答案所在的 token 範圍 | P1–P4 要用；kvdump 目前**不存 Q** | 同一次 prefill 順便存，+<0.1 h；Llama 32 層 × 32 head × ~96 × 128 × 2 B ≈ 25 MiB／題〔算術〕 |
| W_O | 只為 P2w；從模型檔用純 numpy 讀 safetensors | AATC 的 W_O 因子 | 0（CPU） |

**大小**〔算術〕：Llama 32K 全層 BF16 KV＝32 層 × 2 × 8 head × 32,768 × 128 × 2 B＝4.0 GiB／題；Qwen 28 × 2 × 4 × 32,768 × 128 × 2 B＝1.75 GiB／題。共約 17 GiB＋128K 子集，放 `/mlsteam/data/tiara/`（不進 git）。

### 6.2 第一階段：CPU 大量掃描（位元組 vs 代理誤差的 Pareto 前緣）

**步驟 1（校準代理，先做、約 1–2 CPU 小時）**
- 對第二階段**已經量到答對率**的配置算 P0–P4：
  - 13 種 ALL 格式（08 §2.9 表；`tables_longctx.md` 32K–128K）；
  - DS1 的丟 token：12 種挑法 × 5 種比例 × T／P／B 三種粒度（`results/m10_drop_mi300x/ds1/tables_ds1.md`）；
  - 43 複賽的 12 個選手（08 §2.9）。
  - 每個模型約 30–60 個（配置，答對率差）點。
- 每個代理對「答對率差（對 E0）」算 Spearman ρ（bootstrap CI，以配置為單位），以及「崩潰分類」的 AUROC（崩＝對 E0 差 ≤ −5 個百分點）。
- **規則 6 式的自我核對**：CPU 版 codec（numpy）編出來的位元組，要和 dump 裡 GPU 編好的 `<fmt>.<part>.bin` 逐位元一致才用；逆 RoPE 用 `m16_kvstruct.py ropecheck` 先過。

**步驟 2（掃描）**
- **軸**（每個軸都含「不用」）：

  | 軸 | 取值 |
  |:--|:--|
  | 量化 | INT8（逐 token）、INT4 KIVI g32、INT4 KIVI g128、K4V2、INT2 KIVI g32、INT2 g128；Llama 加 FP8 e4m3，Qwen 用 FP8 先減平均（fp8k0） |
  | 低秩 | pre-RoPE、中心化 PCA，基底用**其他題**校準（跨 prompt）；K 的 r ∈ {96, 64, 48, 32, 16}、V 的 r ∈ {96, 64}；係數「均勻量化」或「依成分分位元（KVTC 式 DP）」兩種 |
  | 丟 token | 只用**不需要 query** 的分數（KEYDIFF、KNORM；DS1 裡最好的兩個）＋有 Q dump 後的 SNAP；粒度 T／P（16 token）／B（512 token）；保留 75／50／25／12.5% |
  | 合併 | 相鄰 token 合併 g ∈ {2, 4}（AsymKV 的等價式，L5b）；跨層用 xKV 式共用基底（併進低秩軸） |
  | 無損 | 不用／zstd-3（byte shuffle 後）——用**實際壓出來的位元組** |
  | 順序對照 | (i) 先丟再降秩再量化再無損（標準）；(ii) 基底在「全部 token」vs「留下的 token」上估；(iii) 量化分組對齊 block vs 逐 head 丟後重分組；(iv) 丟的分數用原始 K vs 量化後的 K（Q-Hitter 的干擾） |
  | 聯合分配對照 | 「各軸各自調到同品質再疊」vs「RDKV／KVTC 式同一個預算下聯合分配」（測 KVTC 說的「天真疊加不複合」） |

- **怎麼不爆炸**：先掃單軸（各家族自己的前緣）→ 兩兩組合（量化×丟、量化×降秩、降秩×丟、量化×合併，各再 ± 無損）→ 只從最好的兩兩組合往上加第三、四軸。只保留位元組落在 0.25／0.125／0.0625 ±5% 或在凸包上的組合。**總數約 300–500 個**。
- **輸出**：每個（模型，題型）一張「位元組 vs 代理誤差」圖，三條預算線；表：每個預算下最好的組合、最好的單一方法、各家族最好的單一方法。CSV 帶 `run_id`、`ts`（CLAUDE.md §4.3）。

**CPU 時間**〔估計〕：
- 單價：每個組合、每題、8 層 × 全部 KV head：量化／降秩約 8M 個值的逐元素運算＋最多 0.5 G 次乘加的投影；attention 代理約 384 個 query 向量 × 32,768 × 128 × 2 ≈ 3.2 G 次乘加／（層，head）。用 4 核 numpy 約 10–30 秒。
- 300–500 個組合 × 2 模型 × 3 題 ≈ **8–25 CPU（4 核）小時**；128K 子集另加約 2 小時。
- **先跑 1 小時試點量實際單價**；外推超過 12 小時就把層數從 8 降到 4、query 從 96 降到 32（寫進 run 的 cmd.sh）。
- 一律 `OMP_NUM_THREADS=4 nice -n 19 taskset -c 0-3`；第二階段在計時時，gpu_guard 看不到 CPU 爭用，所以開跑前先和第二階段的 session 對時段（L2 §5 前提）。

**事先寫死的停止條件（第一階段）**
- **S1-a（代理不可靠）**：步驟 1 中，所有代理對答對率的 Spearman ρ 都 <0.5（兩個模型都是）→ 代理只當「崩潰警報」（用已知崩掉配置的代理值當門檻，例如 Qwen 的 FP8 e4m3、INT2h_g32），第二階段改成在「沒觸發警報」的組合裡**隨機抽**，不挑「前緣」。
- **S1-b（代理層級就沒有加乘）**：用最好的代理，每個預算下「最好的組合」的代理誤差 ≥ 0.95 ×「該預算下最好的單一方法」（三個預算、兩個模型全部成立）→ **停，不上 GPU**，寫成負面結果「在代理層級，疊加沒有加乘」。
- **S1-c（無損沒用）**：照實壓出的無損增益，在每個前緣組合上都 <1.05× → 第二、三階段拿掉無損這一軸（和 L2 X1／L5c 的 <1.15× 門檻一致，這裡用更寬的 1.05，因為是疊在有損之上）。

### 6.3 第二階段：GPU 品質（前緣上的 10–20 個組合）

- **選誰**：每個預算挑前緣上 2–6 個組合（偏好「不同家族組合」以分散風險）。三個預算合計約 10–18 個。
- **對照組**（每個模型都要）：E0（BF16）、MASK0（遮罩路徑雜訊）、INT4 KIVI 單獨（0.3125，第二階段錨點）、**每個預算下各家族最好的單一方法**（量化：K4V2／INT2；丟：DS1 最好的分數在 B 與 T 粒度；降秩：同 r 的 pre-RoPE PCA；合併：同 g）、同保留率的隨機丟（RND）、「天真疊加」版（各自調好再疊）。
- **題目與度量**：
  - 32K：S2／DS1 同一套題（seed 60000 起，T2–T5），先跑 teacher forcing（gold_lp、first_ok、KL）當篩選，再對每個預算的前 2 名組合＋最好的單一方法跑**生成答對率**（第二階段的經驗：KL 的差不會變成答對率的差，判定只用生成）。
  - 長度：64K／128K 用 44c 的 48 題切片，只跑每個預算的第 1 名＋最好的單一方法＋INT4。
  - 統計：以題目為單位的配對 bootstrap（10,000 次），Holm 校正（3 預算 × 2 模型＝6 個檢定）；ε＝2 個百分點（第二階段同一個）。
- **要改的程式**（只列）：在 `m10_codec_x` 加「降秩」「合併」「組合 pipeline」格式（encode→decode，走既有的 `load_fmt`／`place`，不改 attention kernel）；丟 token 用既有的 `key_keep`；位元組帳照 §6.0。
- **GPU 時間**〔估計〕：
  - 篩選：每個模型約 33 個配置 × 192 題 × 0.5 秒（L1 §5 的單價）＋prefill ≈ 1 卡小時；兩個模型約 1 小時牆鐘（2 張卡）。
  - 生成：每個模型約 11 個配置 × 192 題 ≈ 2,100 列；整機吞吐取 43ab 的實際值（37,620 列／19.9 小時≈1,890 列／小時，2 張卡、最多 8 個 process）〔算術〕→ 兩個模型約 2.2 小時牆鐘。
  - 128K 切片：約 700 列，每列較慢，約 1–2 小時牆鐘。
  - **合計約 4–5.5 小時牆鐘**；**最小版**（只 32K、每個模型 7 個配置）約 2 小時。
- **事先寫死的停止條件（第二階段）**——兩個主要指標，都照 CLAUDE.md 的 >15％／5–15％／<5% 三段：
  - **M1（同品質省多少位元組）**：每個家族在 ε 內（對 E0 的配對差 CI 下界 ≥ −2）能到的最小位元組 B*。比 r＝B*(最好的組合)／B*(最好的單一方法)。
    - 每個模型都 r > 0.95 → **NO-GO**：「在本設定下疊加沒有加乘」（照 CLAUDE.md 規則 4，這是有價值的負面結果）。
    - 0.85 ≤ r ≤ 0.95 → 停下來問人。
    - 任一模型 r < 0.85 → GO，進第三階段。
  - **M2（同位元組準確率）**：每個預算下，最好的組合 − 最好的單一方法（生成答對率，配對 CI）。
    - **先看有沒有空間**：E0 − 最好的單一方法 <2 個百分點的預算，記成「沒有空間」（組合最多打平，不算失敗也不算成功）。
    - 有空間的預算全部「差 <2 個百分點或 CI 含 0」→ 支持 NO-GO。
  - **完整性條件**：如果最好的組合在 M1／M2 贏，但只贏「天真疊加版」以外的對照，而「天真疊加」本身已經一樣好 → 結論改寫成「疊加有用，但不需要聯合分配」（和 KVTC 的說法相反，要特別報）。

### 6.4 第三階段：時間（編碼／解碼成本 × 連線速度 × GPU 閒／忙）

- **只對第二階段過關（ε 內）的組合做**，外加 INT4 KIVI＋一次拆（現在的最佳）與 BF16（部署預設＝vLLM OffloadingConnector 搬原始位元組）。
- **微測**（一張卡、計時要整張卡、gpu_guard CLEAN）：每種解碼段 × 背景 duty 0／0.4／0.7／0.9（`m7_busy`，第二階段同一套）× 32K／128K：
  - 反量化（已有融合 kernel，`m10_dequant_fast.py`）；
  - 低秩重建（GEMM＋補 RoPE；L5a 的 E-L3 可以合併做）；
  - 丟 token 後的壓實／gather（ThinKV 說會搶 HBM）；
  - 合併的展開；
  - 無損：CPU 多執行緒 zstd 解壓（第一階段已量單執行緒吞吐）；GPU 端 ROCm 只有 early-access 的 hipCOMP（L5c），**期限內不做 GPU 熵解碼**。
- **模擬**：還原時間用已驗證的 `m10_sweep_sim`／`m10_iowall`（09 §5：閒時誤差 1–2%，慢層＋GPU 忙會偏快 23–24%，要標出來）；層：真實 CPU（pinned H2D 約 50 GiB/s）、限速 b16／b32／b48、NFS 級 0.331 GiB/s（參數，不是量測）；G5b 規則自動選 Cake 或全部載入。
- **驗證**：挑模擬裡「組合贏最多」與「組合輸最多」的 4–8 個格子實測。
- **GPU 時間**〔估計〕：微測約 0.5 小時＋驗證 0.5–1 小時＝**1–1.5 GPU 小時**；模擬 <1 CPU 小時。
- **事先寫死的停止條件（第三階段）**：
  - **T-a**：每個（層，duty）格子，「ε 內最快的組合」都沒有比「ε 內最快的單一方法（含 INT4＋一次拆）」快 ≥5% → NO-GO：「疊加省下的位元組被解碼成本吃掉」。
  - **T-b**：組合只在 NFS 級贏、在 CPU 層輸 → 不算新意，寫成一條條件規則（L2 G4 早就預測會退化成簡單規則）。
  - **T-c**：所有組合的「忙÷閒」解碼慢化，都和 INT4 反量化差 <20% → GPU 忙不改變排序，不另立成本項（和 L5a E-L3 同一個門檻）。
  - **T-d**：只有在「最佳組合隨 duty 或連線速度換人」而且換人後至少一格快 ≥15% 時，才算 §5.2 (c) 的新意成立。

### 6.5 停止條件與時間總表

| 階段 | GPU〔估計〕 | CPU〔估計〕 | 停止條件（任一成立就停或改道） | 對照組 |
|:--|:--|:--|:--|:--|
| 前置匯出 | 0.2–0.3 h | — | dump 自我核對（逆 RoPE、codec 位元組一致）不過 → 停（規則 2） | — |
| 第一階段 | 0 | 8–25 h（4 核）＋校準 1–2 h | S1-a 改用「警報＋隨機抽」；S1-b 停；S1-c 拿掉無損 | 各家族單一方法前緣、隨機丟、天真疊加 |
| 第二階段 | 4–5.5 h 牆鐘（最小版 2 h） | 分析 <1 h | M1 r>0.95（兩模型）→ NO-GO；0.85–0.95 問人；M2 有空間的預算全差 <2 pp | E0、MASK0、INT4 KIVI、各家族最好的單一方法、RND、天真疊加 |
| 第三階段 | 1–1.5 h | <1 h | T-a NO-GO；T-b 只寫規則；T-c 不加成本項；T-d 才算新意 | INT4＋一次拆、BF16（部署預設）、ε 內最快的單一方法 |

〔判讀〕GPU 總需求約 6–7 小時牆鐘（最小版約 3.5 小時），在「只剩約 2 天、第二階段佇列優先」的條件下，**建議先只做前置匯出＋第一階段**（GPU 不到半小時），讓第一階段的 S1-a／S1-b 決定要不要花第二階段的 GPU。

### 6.6 會被問的兩個設計選擇〔判讀〕

- **為什麼不直接用文獻的方法（KVTC、RDKV、JoLT）**：它們都沒有 ROCm 版、多數沒釋出程式（L5a：KVTC、SVDq 的官方碼 NOT_FOUND；JoLT 有 repo 但是 Triton＋CUDA 環境、未驗證 ROCm）。我們要的是「同一套位元組帳、同一套題目、同一個還原路徑」下的比較，自己用 numpy／既有 codec 組合比較可控；文獻方法當成「組合空間裡的特定點」（例如 KVTC＝PCA＋DP＋DEFLATE、RDKV＝丟＋量化的聯合分配）。
- **為什麼主指標用「同品質省多少位元組」而不是「同位元組準確率」**：Tiara 的決策本來就是「品質約束下的線上決策」（CLAUDE.md §0）；而且第二階段已知 32K 時好的格式答對率都貼著 E0，同位元組準確率常常「沒有空間」，看不出差別。

### 6.7 事前預測（寫死，跑完對答案）

1. **4×**：Llama 沒有空間（K4V2 在 ε 內），組合最多打平；Qwen 有空間（K4V2 −5.1），預測最好的組合是「INT4 KIVI＋丟約 20%（B 粒度，KEYDIFF／KNORM）」或「INT4 KIVI（K）＋降秩 V」，會過 ε。
2. **8×、16×**：組合幾乎一定贏「只丟」「只降秩」「只合併」（因為單一量化到不了，而 DS1 顯示 B 粒度丟 50% 以上 gold_lp 就掉很多）——M2 在這兩個預算會過；**但這不是新意**（文獻已有）。
3. **無損**：疊在 INT4 g32 上 <1.06×（觸發 S1-c 的機率高）；疊在 PCA 尾端低位元成分上 ≥1.15×。
4. **代理**：Frobenius（P0）跨家族 ρ <0.4；attention 輸出誤差（P2／P2w）ρ ≥0.6；P4（答案區段保留率）在 T2／T3 最好。
5. **時間**：CPU 層，有低秩重建的組合比 INT4 單獨慢；NFS 級，位元組主導、組合快；GPU 忙把「組合開始划算的連線速度」往慢層推。T-a 在 CPU 層成立、在 NFS 級不成立的機率高 → 多半落在 T-b。

---

## 7. 限制

- self-review，不是 cross-model（CLAUDE.md §5）。〔部分〕的論文只讀了指定段落；圖上的數字沒讀（ReST-KV 附錄 I、MiKV 圖 6、ThinKV 的 Pareto 圖）。
- 「同位元組」在不同論文裡算法不同（有的不算 scale、有的不算殘差窗口、有的算 FP16 等效 token），**所有跨論文的數字只並列、不排名**。
- §2.4 的熵是合成 i.i.d. 資料的 order-0 熵，不是我們的 KV；真實比 NOT_MEASURED。
- §6 的時間全部是〔估計〕；單價來自 L1 §5（teacher forcing 0.5 秒／配置／題）與 43ab 的實際吞吐，要以試點為準。
- venue 只在 arXiv comment、會議頁、ACL Anthology、PMLR 查得到時才寫；STAR-KV、MiniCache 的 venue 是二手引用。

---

## 8. 參考清單

「核」：A＝這輪用 curl 開過 arXiv abs 頁（標題、作者、日期、comment）；H＝讀了 arXiv HTML 全文或段落；P＝會議頁或 proceedings 頁核過 venue；—＝非 arXiv。

### 指定六篇
| # | 文獻 | 連結 | 核 |
|:--|:--|:--|:--|
| [S1] | H. Kang, Q. Zhang, S. Kundu, G. Jeong, Z. Liu, T. Krishna, T. Zhao, "GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM"（正式版題目：GEAR: An Efficient Error Reduction Framework for KV Cache Compression in LLM Inference），PMLR 262:305–321（NeurIPS 2024 ENLSP workshop） | https://arxiv.org/abs/2403.05527 ；https://proceedings.mlr.press/v262/ ；https://github.com/HaoKang-Timmy/GEAR | A＋H＋P |
| [S2] | K. Staniszewski, A. Łańcucki, "KV Cache Transform Coding for Compact Storage in LLM Inference," ICLR 2026 | https://arxiv.org/abs/2511.01815 | A＋H |
| [S3] | Y. Liu *et al.*, "CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving," SIGCOMM'24 | https://arxiv.org/abs/2310.07240 | A＋PDF |
| [S4] | C.-C. Chang *et al.*, "Palu: KV-Cache Compression with Low-Rank Projection," ICLR 2025 | https://arxiv.org/abs/2407.21118 ；https://iclr.cc/virtual/2025/poster/29993 | A＋H＋P |
| [S5] | J. Y. Yang *et al.*, "No Token Left Behind: Reliable KV Cache Compression via Importance-Aware Mixed Precision Quantization"（MiKV），arXiv 2024（venue NOT_FOUND） | https://arxiv.org/abs/2402.18096 | A＋H |
| [S6] | Y. He *et al.*, "ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification," NeurIPS 2024 | https://arxiv.org/abs/2405.14256 ；https://nips.cc/virtual/2024/poster/96563 | A＋H＋P |

### 系統性比較與聯合分配
| # | 文獻 | 連結 | 核 |
|:--|:--|:--|:--|
| [S8] | J. Zhang *et al.*, "More Tokens, Lower Precision: Towards the Optimal Token-Precision Trade-off in KV Cache Compression," Findings of EMNLP 2025 | https://arxiv.org/abs/2412.12706 ；https://preview.aclanthology.org/dashboard/2025.findings-emnlp.429 | A＋H＋P（搜尋） |
| [S9] | J. Zhang, H. Guo, L. Benini, Y. Li, "RDKV: Rate-Distortion Bit Allocation for Joint Eviction and Quantization of the KV Cache," arXiv 2026 | https://arxiv.org/abs/2605.08317 | A＋H |
| [S10] | R. Krishnan, V. Schulz, "A JoLT for the KV cache: Near-Lossless KV Cache Compression via Joint Rank-bit Allocation," arXiv 2026（投 ICLR 2027） | https://arxiv.org/abs/2607.12550 | A＋H |
| [S11] | H. Laus, C. M. Verdun, H. Wang, F. du Pin Calmon, F. Krahmer, "KV Cache Compression Through the Lens of Transform Coding," arXiv 2026 | https://arxiv.org/abs/2608.14191 | A＋H |
| [S12] | S. Salfati, "Quantization Dominates Rank Reduction for KV-Cache Compression"（v1）→ "Attention-Weighted Value Projection for KV-Cache Compression"（v2，撤回同儲存比較），arXiv 2026 | https://arxiv.org/abs/2604.11501 | A＋H（v1、v2） |
| [B1] | J. Yuan *et al.*, "KV Cache Compression, But What Must We Give in Return? A Comprehensive Benchmark of Long Context Capable Approaches," arXiv 2024（venue 未查證） | https://arxiv.org/abs/2407.01527 | A＋H |
| [B2] | W. Gao, X. Zhou, P. Sun *et al.*, "Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving," MLSys 2025（arXiv comment） | https://arxiv.org/abs/2503.24000 | A＋H |
| [B3] | N. Agrawal, R. Mayer, "Benchmarking KV-Cache Optimizations across Task Quality and System Performance for Long-Context Serving," arXiv 2026 | https://arxiv.org/abs/2607.05399 | A＋H |
| [B4] | J. Kim *et al.*, "Not All Bits Are Equal: Scale-Dependent Memory Optimization Strategies for Reasoning Models," arXiv 2025 | https://arxiv.org/abs/2510.10964 | A |

### 其他組合
| # | 文獻 | 連結 | 核 |
|:--|:--|:--|:--|
| [S7] | Z. Zhang, S. Liu, R. Chen, B. Kailkhura, B. Chen, Z. Wang, "Q-Hitter: A Better Token Oracle for Efficient LLM Inference via Sparse-Quantized KV Cache," MLSys 2024 | https://proceedings.mlsys.org/paper_files/paper/2024/file/bbb7506579431a85861a05fff048d3e1-Paper-Conference.pdf | P（PDF 抽字） |
| [S13] | A. Sharma, H. Ding, J. Li, N. Dani, M. Zhang, "MiniKV: Pushing the Limits of 2-Bit KV Cache via Compression and System Co-Design for Efficient Long Context Inference," Findings of ACL 2025（arXiv 題目：MiniKV: Pushing the Limits of LLM Inference via 2-Bit Layer-Discriminative KV Cache） | https://arxiv.org/abs/2411.18077 ；https://preview.aclanthology.org/setup/2025.findings-acl.952 | A＋H＋P（搜尋） |
| [S14] | A. Ramachandran *et al.*, "ThinKV: Thought-Adaptive KV Cache Compression for Efficient Reasoning Models," ICLR 2026 Oral | https://arxiv.org/abs/2510.01290 | A＋H |
| [S15] | Y. An *et al.*, "ReST-KV: Robust KV Cache Eviction with Layer-wise Output Reconstruction and Spatial-Temporal Smoothing," ICLR 2026 | https://arxiv.org/abs/2605.08840 | A＋H |
| [S16] | Y. Feng *et al.*, "Taming the Fragility of KV Cache Eviction in LLM Inference"（DefensiveKV），arXiv 2025 | https://arxiv.org/abs/2510.13334 | A＋H |
| [S17] | J.-H. Kim *et al.*, "KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction," NeurIPS 2025 Oral | https://arxiv.org/abs/2505.23416 | A＋H |
| [S18] | A. Liu *et al.*, "MiniCache: KV Cache Compression in Depth Dimension for Large Language Models," NeurIPS 2024（依 ThinKV 參考文獻） | https://arxiv.org/abs/2405.14366 | A＋H |
| [S19] | H. Yang *et al.*, "XQuant: Achieving Ultra-Low Bit KV Cache Quantization with Cross-Layer Compression," EMNLP 2025 | https://arxiv.org/abs/2510.11236 | A＋H |
| [S20] | A. Shutova *et al.*, "Cache Me If You Must: Adaptive Key-Value Quantization for Large Language Models"（AQUA-KV），ICML 2025（PMLR 267） | https://arxiv.org/abs/2501.19392 ；https://proceedings.mlr.press/v267/shutova25a.html | A＋H＋P（搜尋） |
| [S21] | C.-C. Chang *et al.*, "xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction," ICML 2026 | https://arxiv.org/abs/2503.18893 | A（全文見 L5b） |
| [S22] | H. Yankun *et al.*, "SVDq: 1.25-bit and 410x Key Cache Compression for LLM Attention," arXiv 2025 | https://arxiv.org/abs/2502.15304 | A＋H |
| [S23] | Y. Xu *et al.*, "ThinK: Thinner Key Cache by Query-Driven Pruning," ICLR 2025 Spotlight | https://arxiv.org/abs/2407.21018 | A＋H |
| [S24a] | H. Sun *et al.*, "ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference," ICML 2025（依 L2） | https://arxiv.org/abs/2410.21465 | A |
| [S24b] | P. Singhania *et al.*, "Loki: Low-rank Keys for Efficient Sparse Attention," NeurIPS 2024 | https://arxiv.org/abs/2406.02542 | A |
| [S25] | D. Yao *et al.*, "TailorKV: A Hybrid Framework for Long-Context Inference via Tailored KV Cache Optimization," arXiv 2025（venue 未查證） | https://arxiv.org/abs/2505.19586 | A＋H |
| [S26] | Y. Zhang *et al.*, "DiffKV: Differentiated Memory Management for Large Language Models with Parallel KV Compaction," SOSP 2025 | https://arxiv.org/abs/2412.03131 | A＋H（摘要與目次） |
| [S27] | J. Kampeas, E. Haleva, "Joint Encoding of KV-Cache Blocks for Scalable LLM Serving," arXiv 2026 | https://arxiv.org/abs/2601.03067 | A |
| [S28] | A. Garg *et al.*, "QEvict: Recoverable Quantized KV Eviction for Attention-Drift-Robust Long-Context Decoding," arXiv 2026 | https://arxiv.org/abs/2608.05326 | A |
| [S29] | D. Lee, D.-H. Kim, J.-H. Kim, "SemKV: Semantic Mixed-Precision KV Cache Quantization Guided by the Quality Cliff for Long-Context LLM Inference," arXiv 2026 | https://arxiv.org/abs/2608.28911 | A |
| [O1] | KIVI（ICML'24） | https://arxiv.org/abs/2402.02750 | A |
| [O2] | KVQuant（NeurIPS 2024） | https://arxiv.org/abs/2401.18079 | A |
| [O3] | LESS（Dong *et al.*，2024） | https://arxiv.org/abs/2402.09398 | A |
| [O4] | CommVQ（ICML 2025 poster） | https://arxiv.org/abs/2506.18879 | A |
| [O5] | STAR-KV（Bhatnagar *et al.*，2026；ICML 2026 依 JoLT 參考文獻） | https://arxiv.org/abs/2606.08382 | A |
| [O6] | R. Krishnan, V. Schulz, "Tensor Decomposition of Transformer Key-Value Caches: Spectral Structure and Format Comparison," arXiv 2026 | https://arxiv.org/abs/2609.28029 | A |
| [O7] | KV-CoRE（2026） | https://arxiv.org/abs/2602.05929 | A |
| [O8] | "Does Accuracy Equal Evidence? Reasoning Faithfulness under KV Cache Compression"（2026） | https://arxiv.org/abs/2608.01631 | A |
| [O9] | KV-Surgeon（2026） | https://arxiv.org/abs/2610.25933 | A |
| [O10] | VECTOR："A Simple Plug-in for Improving Eviction-Based KV Cache Compression"（2026） | https://arxiv.org/abs/2605.23258 | A |
| [O11] | VarRate（2026） | https://arxiv.org/abs/2607.15498 | A（內容見 L5a） |
| [O12] | kvpress README（ComposedPress、QuantizedCache） | https://github.com/NVIDIA/kvpress | —（README） |
| [O13] | KVServe（SIGCOMM'26）、SplitZip（NeurIPS'26）、ShadowServe、KVCodec、HACK、AdaptCache、EvicPress | 見 L2 §6、L5c §7 | A（本輪重開 KVServe、SplitZip、HACK、AdaptCache、EvicPress 的 abs） |

### 本專案的檔案（引用的實測）
- `docs/precision_20261013/08_report_mi300x.md` §0、§2.3、§2.4、§2.9（run_id 見該檔附錄 A；海選 `20261016-135544-…` 等，複賽 43ab）。
- `results/m10_longctx_mi300x/analysis/tables_longctx.md`（Llama run `20261019-033004-m10-lc-screen-llama-252-s0` 等 6 個；Qwen `20261019-055235-m10-lc-screen-qwen-252-s0` 等 6 個）。
- `results/m10_drop_mi300x/ds1/tables_ds1.md`（run `20261020-001157-m10-ds1-llama-s0` 等 12 個）。
- `results/RUNLOG_MI300X.md` 11.2o（43ab：12 個 process、最多 8 個同時，13:44:26Z 開跑）。
- `code/m16_s1_kvdump.py`、`code/m16_kvstruct.py`、`code/m16_s2_equalbytes.py`、`code/m10_codec_x.py`（只讀檔頭與函式名）。

### 查了沒找到（NOT_FOUND）
- 三種以上「各自有損」的方法疊在同位元組、≥32K、GQA 模型上，並和單一方法比較的研究。
- 代理指標（CPU 可算）**跨方法家族**預測答對率的系統性驗證。
- 把「組合 × 連線速度 × GPU 背景負載」放在同一個模型或實驗裡的研究。
- 疊加組合在 Cake 式還原（只壓載入區）下的比較。
- 任何 MI300X／ROCm 上的組合量測。
- MiKV 的正式發表 venue。
- 低秩基底該在「全部 token」或「丟完留下的 token」上估的比較。
