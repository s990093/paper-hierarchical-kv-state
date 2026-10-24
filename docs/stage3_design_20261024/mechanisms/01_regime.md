# 01　為什麼把主設定換成 128K–256K、8B–14B：六個物理事實

> **日期**：2026-10-24。**性質**：整理已量到的數字＋算術推估，**沒有新的量測**。
> 標記：〔實測〕附出處；〔算術〕＝[arith.md](arith.md) 的某一節（`code/s3_mech_arith.py` 產生，可重跑）；〔假設〕〔判讀〕照舊。
> 回到這一組的核心頁：[README](README.md)。

---

## 0. 一句話

在 32K，回來那一輪的還原只要 0.05–0.1 秒，怎麼做都差不多；到 128K–256K、14B，**HBM 只放得下 1–6 個 session，回來要搬 7–48 GiB，重算要十幾秒，decode 卻讓 tensor core 閒著**。所有新機制都是在這個區域才有大的槓桿。

![圖 1](figures/fig1_regime.svg)

---

## 1. 六個事實

| # | 事實 | 數字 | 出處 | 對哪些機制有利 |
|:--|:--|:--|:--|:--|
| F1 | **HBM 放不下幾個長 session**，暫停／回來是常態 | 完整 KV 放得下的 session 數：Qwen2.5-14B-1M 256K **1 個**、128K 3 個；Mistral-NeMo-12B 128K 4 個；Llama-3.1-8B 128K 6 個；Qwen2.5-7B-1M 256K 6 個。換成 16 倍粗底：28／56／71／97／111 個 | 〔算術〕arith §1（HBM 141 GB×0.9、activation 預留 6 GiB〔假設〕） | M01（粗底常駐） |
| F2 | **回來要搬的位元組很大**，慢層上是秒到分鐘 | 一個 session 7–48 GiB。全部 BF16：CPU 層 0.26–0.90 s；本地 NVMe 2.0–6.9 s；NFS 42–145 s。實測錨點（harness Cake E0，GPU 閒）：Llama 128K CPU 層 0.345 s、b48（約 0.72 GiB/s）8.43 s；Qwen 256K CPU 層 0.373 s、b48（約 0.36 GiB/s）18.6 s | 〔算術〕arith §2；〔實測〕`origin/h200:results/m10_longctx_h200/analysis/tables_longctx.md` §5、`tables_longctx256.md` §5 | M01、M02、M06、M13 |
| F3 | **重算的貴在注意力那一項**，而且隨長度變重 | vLLM 冷 prefill（Llama）外推到 128K 約 11.1 s，其中 n² 項佔 75%；重算÷CPU 層載入由 32K 的 16.4 倍變成 128K 的 37.3 倍。harness 逐塊重算：128K 最後一塊有 74% 是注意力 | 〔實測〕H200 vLLM 冷 prefill 16K 0.482 s、32K 1.224 s（`origin/h200:results/RUNLOG_H200.md` C6）；fa3 斜率 0.210 ms／塊、Σ64＝1,583 ms（commit 98598c5）；〔算術〕arith §4 | M12（近似重算） |
| F4 | **長 context 的 decode 讓 tensor core 閒著** | 每一步 2.6–4.0 FLOP／byte；H200 實測 matmul÷HBM＝189 FLOP／byte。TPOT 下界 7.3–19.4 ms（batch 1） | 〔實測〕BF16 matmul 788.1 TFLOPS、HBM 複製 4,173.9 GB/s（RUNLOG_H200 GPU smoke）；〔算術〕arith §8 | M08（搭 decode 的便車）、M12 的「後補精」 |
| F5 | **低位元的傷害隨長度變大**，均勻壓縮在長 context 更危險 | Qwen int2_kivi：32K −9.7、256K −14.4 個百分點；int4_kivi 256K −3.3 [−10.0, +1.1]；Llama int2_kivi 128K −9.2 [−16.2, −2.7]，128K−32K 為 −13.0（CI 不含 0）。Qwen fp8_head 在 256K 掉到 0% | 〔實測〕`tables_longctx.md` §2（Llama，n＝48）、`tables_longctx256.md` §2（Qwen T2–T4，n＝36） | M01–M03（只讓少數塊清楚） |
| F6 | **長 context 服務的背景本來就在「頻寬型」和「算力型」之間交替** | decode（F4，頻寬型）一次持續很多步；新文件進來的 128K prefill 約 11 s（F3，算力型）。不用人造 matmul 背景，B 的翻轉就有機會自然出現 | 〔判讀〕由 F3、F4 推；**真實背景下的翻轉沒量**（02 §2 第 2 點） | M08、M09、M10 |

### 1.1 F5 的讀法要小心

- 128K／256K 每格只有 36–48 題，CI 很寬；int4_kivi 256K 的 CI 含 0，只能說「方向是變差」。主實驗照 05 §2 的規矩，每個判定格 ≥288 題。
- E0 本身沒有飽和：Llama 128K 42.2%、Qwen 256K 55.1%（同上兩個表 §1）。這是好事：題目夠難，好壞才分得開。

---

## 2. 這個設定為什麼「有利但合理」

- **模型本身就支援**：Qwen2.5-7B／14B-Instruct-1M 的 `dual_chunk_attention_config.original_max_position_embeddings`＝262,144（HF config.json，10/24 讀取），no-DCA 變體上限 262,144，和我們 7B 的做法相同（`code/make_nodca_model.py`）；Llama-3.1-8B、Mistral-NeMo-12B 原生 128K（config.json 的 `max_position_embeddings`＝131,072）。**不用開 YaRN**（CLAUDE.md §8）。
- **14B 剛好一張 H200 放得下**：權重 27.5 GiB＋一個 256K session 48 GiB（arith §1）。單卡、不用 TP，符合 CLAUDE.md §3「容量懸崖在單卡量」。
- **還沒查證**：真實服務裡 128K 以上的 session 有多常見。我們手上的 Mooncake trace（D4）只算了重用，沒算長度分布〔NOT_VERIFIED〕。論文裡要引一個長度分布的來源，不然審稿人會說是挑的。

---

## 3. 必須同時報的不利設定（不報就是挑數據）

| 不利的地方 | 數字 | 怎麼報 |
|:--|:--|:--|
| CPU 層很快 | Llama 128K 全部 BF16 只要 0.30 s〔算術〕、0.345 s〔實測 Cake E0〕；問題長的時候，prefill 本身就 0.11–0.37 s（arith §6） | 每個機制都要有 CPU 層的列，即使好處只剩 3–4 倍或更少 |
| 每個 GPU job 只有 170 GB 主機記憶體 | 14B 256K 一個 session 48 GiB → CPU 層只放得下 3 個 | 多 session 實驗用 Qwen-7B 256K（14 GiB／個）或 4 卡 job（680 GB）；寫進限制 |
| H200 的本地 NVMe 沒量過 | arith 用的是 MI300X 的 6.981 GiB/s | 第 0 段先用 `fio` 量 `/tmp`（md0），換掉算術常數 |
| `/work` 是不是 NFS，有兩份紀錄互相矛盾 | `H200_SLURM.md` 說 `$HOME`＝`/work/...`（NFS）；`RUNLOG_H200.md:468` 說我們能寫的是本地 md0、md1（`/work`），NFS 只有 `/home`、`/cm/shared` | 開跑前 `mount`、`df -T /work` 確認；確認前，NFS 的列一律用限速層代替並標明 |
| 32K 已經沒什麼可贏 | Qwen 32K CPU 層 Cake 0.053 s〔實測〕 | 32K 只當對照列，主結論不放 |
| 任務性質 | 摘要、彙整類的問題需要幾乎每一塊（ρ→1） | 任務分兩組：「找答案」類（有利）與「彙整」類（不利），分開報 |
