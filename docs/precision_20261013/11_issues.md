# 11　之後要注意的問題清單

> 回到 [README](README.md)。這一頁記「已經看到、還沒解決、之後做任何相關實驗或寫結論時都要記得」的問題。每一條寫：看到什麼（附 run_id）、為什麼重要、做了什麼、還不知道什麼、之後要注意什麼。
> 只記實測到的；推論標〔推論〕；沒量到寫 NOT_MEASURED。

---

## 問題一：反量化的開銷（低精度在「真實 CPU 層」反而變慢）

**狀態**：開放（2026-10-16 記，使用者要求「記錄這個問題、之後要注意」）。

### 1. 看到什麼

- 低精度（FP8、INT4）讓載入的位元組變少，在限速層快很多（FP8 22–39%、INT4 36–57%）。
- **但在真實 CPU 層（不限速、真的 H2D，約 50 GiB/s）打平或變慢**，INT4 比 FP8 更慢（32K、Cake、中位數）：

  | 模型 | GPU | 還原 E0 → FP8 E1 → INT4 E1（ms） | 反量化 FP8 → INT4（ms） | H2D 省下的（E0 → INT4，ms） |
  |:--|:--|:--|:--|:--|
  | Llama | 閒 | 95.5 → 71.7 → 78.5 | 24.0 → 44.2 | 79.8 → 27.2 |
  | Llama | 忙 | 164.1 → 170.8 → 189.4 | 64.0 → 112.6 | 116.4 → 29.4 |
  | Qwen | 閒 | 51.8 → 54.9 → 61.0 | 10.7 → 25.5 | 36.5 → 14.4 |
  | Qwen | 忙 | 65.9 → 74.1 → **145.9** | 26.1 → 64.3 | 50.1 → 15.7 |

  run：FP8 Llama `20261015-161226-m10-grid-fp8-llama`、Qwen `20261015-194150-m10-regrid-fp8-qwen`；INT4 Llama `20261016-034902-m10-grid-int4-llama`、Qwen `20261016-052143-m10-grid-int4-qwen`（08 §2.1、§2.5）。

- **10/16 10:29Z INT4 品質出來：兩個模型 24 格全過**（08 §2.5.3）。所以 INT4 階判「均勻不夠」的**唯一原因就是這個問題**（真實 CPU 層的速度）；老師選 08 §6 第 1 點的哪個選項，直接決定判定流程是繼續 Step 2 還是整個停。
- **比喻**：搬家用壓縮袋。搬得遠（限速層）時省下很多趟；搬到隔壁（真實 CPU 層）時，拆封多花的時間比搬運省下的還多。

### 2. 為什麼慢（已經知道的）

- 主線用凍結時的**原寫法**反量化：PyTorch 一步一步做（拆 4-bit／轉 FP32／乘 scale／加 zero／轉 BF16／複製進 KV），每步一個 GPU kernel，同一份資料來回讀寫 4–5 趟。實測只用到 HBM 頻寬的不到一成（09 §6、P07 §4）。
- **真正的成本是「和重算搶 GPU」**，不只是反量化自己的時間：
  - 真實 CPU 層的會合點 b＝1–3，最後結束的是重算線（`waiting_line`＝load：載入線先做完在等）。
  - Qwen GPU 忙：重算線只算 1–2 個 block，E0 時 64.5 ms，INT4 時變 144.7 ms——被反量化拖慢約 80 ms。
  - 還原中每個 block 的反量化（Llama INT4 忙約 1.8 ms、Qwen 約 1.0 ms）比單獨量（冒煙，零點幾 ms）大很多。
- GPU 忙時更嚴重：反量化和背景負載也在搶（FP8 反量化閒 → 忙變 2–3 倍）。前作 ShadowServe 講的是同一件事（P07 §4）。
- **「整數跳動」**（10/16 07:1xZ 拆 33a 的逐次資料）：Qwen INT4、GPU 忙、真實 CPU 層 30 次還原裡，重算 1 塊的 1 次是 85 ms，重算 2 塊的 29 次是 146 ms；兩種情況載入線都在約 84 ms 做完。也就是反量化讓載入線慢一點，重算線就多開一個 512 token 的 block（GPU 忙時要約 60 ms），大家等它。Qwen・忙慢掉的約 80 ms 裡，約 60 ms 是這個。Llama・忙則兩邊都重算 2 塊，是重算線本身被拖慢（163 → 188 ms）。

### 3. 已經做了什麼

- **融合反量化 kernel**（`code/m10_dequant_fast.py`，Triton）：一個 kernel 讀一次、算完直接寫進 KV。支援 fp8_head／channel／token、int8_token、int4_token、int4_kivi。
  - **GPU 上和原寫法逐位元相同**（10/16 06:44Z，run `20261016-064403-m10-dqfast-selftest-gpu-int4`、`20261016-064414-m10-dqfast-shapes-gpu`）→ 換成它**只改時間、不改任何答案**。
  - 冒煙（每格 1 次，scratch，不是結果）：FP8 每個 block，Llama 閒 0.24 → 0.09 ms、忙 0.58 → 0.10 ms；Qwen 閒 0.11 → 0.08 ms、忙 0.16 → 0.14 ms（run `20261016-034530-m10-sweep-smoke-parts-qwen`、`-034701-…-llama`）。
- 寫進 08 §6 第 1 點，要老師決定「速度不輸」要不要管真實 CPU 層（到目前為止，所有「速度不輸」沒過的格子都在這一層）。
- 前作整理：P07 §4（QServe、BitDecoding、FLUTE、vLLM FP8 Marlin、Petit、ShadowServe、PackKV）。
- **修掉一個 bug**（10/16 07:1xZ）：融合 kernel 寫進「整個 KV 緩衝的視圖」時，位址用 32 位元整數算；Llama 32K 的緩衝最遠的偏移約 2.18e9，超過 2^31−1 → 繞回，數值寫錯而且寫到緩衝前面（CPU 直譯模式重現：`code/m10_queue_mi300x/tests/dq_fast_offset_cpu_interp.py`）。只影響 Llama 的 fp8／int8／int4_token（KIVI 剛好在上限內）；還沒有任何計時用到它（42 還沒跑）。已改成 64 位元：CPU 上 5 種格式在 Llama 大小的緩衝裡全對、護欄乾淨；GPU 重驗逐位元相同（run `20261016-071948-m10-dqfast-selftest-gpu-i64`、`-072000-m10-dqfast-shapes-gpu-i64`）。
- **登記了方法掃描**（09 §6.4，使用者 10/16「好加入」「找很多方法掃描」「hidden latency 可以測試」）：K 組（怎麼轉）、S 組（何時轉，藏延遲：邊搬邊拆、多個 block 一組、重算優先、最後一次拆）、G 組（切多細：256／128 token、預估會合）。海選 → 複賽，只報告。

### 4. 還不知道的

- **INT4 融合 kernel 的速度**（10/16 13:06Z，stage 40 開頭的微測，run `20261016-130602-m10-dqfast-bench`，`results/m10_sweep_mi300x/parts/dequant_fast_bench.json`；單獨量、每個 block、中位數；這支自測沒有開 gpu_guard，同一時段我們沒有其他 GPU 工作）：

  | 模型 | 格式 | 原寫法 | 融合 kernel | torch.compile |
  |:--|:--|--:|--:|--:|
  | Llama | INT4 KIVI | 0.449 ms | 0.101 ms（4.4 倍） | 0.087 ms |
  | Llama | INT4 每 token | 0.318 | 0.080（4.0 倍） | 0.084 |
  | Llama | FP8（token） | 0.227 | 0.081（2.8 倍） | 0.059 |
  | Qwen | INT4 KIVI | 0.254 | 0.076（3.3 倍） | 0.054 |
  | Qwen | FP8（head） | 0.099 | 0.053（1.9 倍） | 0.039 |

  - 6 種格式的融合版、torch.compile 版都和原寫法逐位元相同（同一個 JSON 的 checks）。
  - torch.compile（PyTorch 自動融合）在多數格式比我們手寫的 Triton kernel 還快一點；方法掃描的 K 組有放它（09 §6.4，k2）。
  - **還原中的實際成本仍未知**：還原中原寫法每個 block 要 0.69 ms（Llama・閒）到 1.76 ms（Llama・忙），比單獨量的 0.449 ms 大，多出來的是和重算搶 GPU。這部分由 stage 42 量。
- **融合之後，真實 CPU 層的還原時間：NOT_MEASURED**。單獨量的 kernel 時間推不出來，因為主要成本是干擾。由 stage 42 量（FP8、INT4，`M10_DEQUANT_FAST=1`，09 §6.3-2／2b；只報告，不改主線判定）。
- 要省多少才夠（〔算術〕，用上面的實測）：
  - Llama・忙：INT4 189 ms → 「速度不輸」門檻 174 ms、贏 E0 要 < 164 ms → 要省 16–25 ms（反量化現在 113 ms）。〔推論〕很可能夠。
  - Qwen・閒：61.0 → 門檻 58.5、E0 51.8 → 要省 3–9 ms（反量化 26 ms）。〔推論〕很可能夠。
  - Qwen・忙：146 → 門檻 76、E0 66 → 要省 70–80 ms，幾乎是全部的干擾。〔推論〕光靠融合 kernel 很可能不夠。
  - 混合配置裡也有 16 個 INT4 block，換成融合版門檻也會跟著動。

### 5. 之後要注意（做相關實驗、寫任何結論時）

1. **說「低精度讓還原變快」一定要講是哪一層**。限速層成立；真實 CPU 層不成立，而且主要是反量化的寫法造成的。
2. **大掃描（09）的新格式反量化更貴**：冒煙裡原寫法每個 block，mxfp4 0.43–0.73 ms、int2h_g32 1.0–4.4 ms、K8V4 0.21–0.67 ms（FP8 原寫法 0.11–0.58 ms）。這些格式在真實 CPU 層的時間會被反量化主導；融合 kernel 目前只支援 FP8／INT8／INT4，其他格式退回原寫法。報告時要分開列反量化。
3. **長 context（10）**：block 數變 2–4 倍，反量化的總量也跟著變多；64K／128K 的速度格子要看 `t_dequant`。
4. **H200 復現時要重量**：MI300X 的硬體 FP8 是 fnuz 版，我們存的是 OCP e4m3fn，所以在 MI300X 上只能用軟體查表轉換（P07 §4）；H200（Hopper）的硬體 FP8 就是 OCP 的 e4m3。反量化的成本兩台可能差很多，不能把 MI300X 的數字搬過去。
5. **主線的判定不改**：07 凍結時規定速度用原寫法；融合版的結果只報告。要改判準必須老師同意、另外登記（08 §6 第 1 點的選項 c）。
6. 如果融合 kernel 還不夠（特別是 Qwen・忙），下一步的選項（都還沒做）：
   - 把所有 block 的反量化合成一次啟動、或放到低優先權的 stream，減少和重算搶 GPU（便宜，先試）。
   - 把反量化塞進 attention（QServe／BitDecoding／PackKV 的做法）：最徹底，但換了 attention 路徑，違反 07「低精度和 BF16 走同一條 attention」，只能當系統面的延伸。

7. **使用者 10/16 的方向**：「CPU 層省不到時間」主要是反量化寫法造成，跟位置無關；真正要看的是**慢層**（SSD／NFS 級的頻寬）和**長 context**（KV 變大、CPU 放不下就要去 SSD），低精度的價值是把 IO 牆等效往後推（同樣的頻寬搬 2–3.2 倍的 token、同樣的記憶體放 2–3.2 倍的對話）。報告要**分層寫**，CPU 層單獨列。
   - 〔算術，用上面的實測〕兩條線都和 block 數成正比：Llama・閒每個 block 的 H2D，E0 約 1.25 ms、INT4 約 0.43 ms（省約 0.82 ms），原寫法的反量化約 0.69 ms → 只差一點；所以光是 context 變長，CPU 層的結論〔推論〕不會自己翻過來，要靠反量化變快（stage 42、42b／42c）。64K／128K 的 CPU 層實測：NOT_MEASURED（stage 90 的速度格子有含）。
   - **和這個推論相反的跡象**（39a 冒煙，Qwen、FP8、128K、GPU 閒、只跑 1 次，scratch，不能當結論；run `20261016-130008-m10-lc-smoke-grid-qwen`）：真實 CPU 層 E0 0.166 s、E1 0.150 s（E1 快約 10%）；32K 主線是 E0 0.052、E1 0.055 s（E1 慢約 5%）。128K 的會合點 E0 b＝6、E1 b＝5（32K 是 1–2），而且這兩列被標 copy_bound。所以 context 變長，CPU 層的結論**可能會變**，以 stage 90 的實測為準。
   - 「慢層」在這裡是限速模擬（老師：保留這個範圍，01 第 12 列）。對照第一階段 MI300X 量的真實裝置（03 §R2 的 `tier_params.json`：本地 SSD 讀 6.98 GiB/s、NFS 0.331 GiB/s）：Llama b16＝5.3 GiB/s 和本地 SSD 同一級，b48＝0.42 GiB/s 和 NFS 同一級。

### 6. 相關文件

08 §2.1（FP8 的 CPU 層拆解）、§2.5（INT4）、§6 第 1 點；09 §6（反量化開銷與加速）；cards/P07 §4（前作）；RUNLOG_MI300X 10.9、11.2、11.2b、11.2c；`code/m10_dequant_fast.py`。

---

## 問題二：INT4 品質全過——是題目太簡單、context 太短，還是程式有錯？

**狀態**：32K 的部分查完（10/16 12:xZ，使用者問）；長 context 由 10 號計畫（stage 90–92）回答。

### 1. 看到什麼

- INT4 KIVI 兩個模型 24 格全過（08 §2.5.3）。同一批題 Qwen 的 FP8 掉 32.8 個百分點。

### 2. 查了什麼（CPU；`code/m10_int4_sensitivity.py` → `results/m10_precision_mi300x/analysis/int4/sensitivity.{json,md}`；run 見該檔，Step 1 與 33b 那兩批）

- **題目不是太簡單**：E0 答對率 64–87%（Llama 67.2–80.6、Qwen 64.0–87.2），沒有碰到天花板或地板。
- **測得出傷害**（正對照）：同一批題、同一支程式，Qwen FP8 掉 32.8 個百分點、HotpotQA 掉約 72。
- **劑量反應**（讀法 A，和 E0 輸出逐字相同的比例；KL 中位數）：
  - Llama：INT8 91.4%（0.0003）> FP8-K0 84.6%（0.0011）> FP8 81.7%（0.0016）> **INT4 74.6%（0.0033）**。
  - Qwen：INT8 83.2%（0.0014）、FP8-K0 83.5%（0.0015）、**INT4 73.9%（0.0053）**；FP8 10.3%（格式問題，§2.4）。
  - → INT4 確實改到了模型的計算，而且改得比 8 bit 多。
- **比答對率敏感的指標**（標準答案每個 token 的 log 機率，對 E0 配對，95% CI）：INT4 有小、但看得到的下降：
  - Llama T3 −0.0077 ± 0.0037、T5 −0.0067 ± 0.0031；Qwen T5 −0.0275 ± 0.0051（每個 token 的機率約低 0.7%／2.7%）。
  - 8 bit 大多在 0 附近（例外：Qwen INT8 T2 −0.0176 ± 0.0027）。
  - → **INT4 有傷，但在 32K 還不夠大，答案翻不過去**。最敏感的是 T5（整段統計常見詞，要用到整段 context）。
- **程式**：
  - CPU 假資料（K 的某個 channel 加 400，模擬 Qwen）：KIVI 每一組（K：32 token × 1 channel；V：1 token × 32 channel）最多 16 個不同值；連 scale／zero 每個值 5 bit（0.625 byte）；`m10_codec.py --self-test` ALL PASS（它的第 6 項其實用到了 GPU，見 RUNLOG 10.11 的更正）。
  - 同一份假資料，int4_token（每 token 一個 scale）其他 channel 的 K 相對誤差＝1.0（全被壓成 0），KIVI 是 0.078 → 和「Qwen 的 FP8 壞、KIVI 不壞」同一個機制。
  - 讀法 A 每題讀到 64／64 個 INT4 block（CSV 的 `lp_read`）。
  - 和原版 KIVI 的差別：原版把最近 128 個 token 留全精度；我們 context 全部量化（只有問題本身的 KV 是 BF16）→ 比原版嚴格。

### 3. 文獻怎麼報（摘要級，未逐頁核對；self-review）

- KIVI（ICML 2024，arXiv 2402.02750）：2 bit 在 Llama／Mistral 最多掉約 2%；只有 MQA 的 Falcon-7B 要用 4 bit。
- 長 context 壓縮 benchmark（arXiv 2407.01527）：KIVI-4bit 的 LongBench 平均和不壓縮相同（43.8，單一模型）；2 bit 42.6。
- 2026 的比較（arXiv 2607.05399，Llama-3.1-8B、Mistral-7B）：KIVI4 品質最穩定。
- 推理模型（arXiv 2504.04823）：4 bit KV 幾乎不掉（≤1%），但難題（AIME）掉得比簡單題（GSM8K）多到 4 倍。
- 長度（NeurIPS 2026「Length, Not Bit Width」，70B 級、單針、4K–128K、int8／int4）：量化讓「找不到」的長度提前（Qwen 約 96K → 約 32K）。
- SCBench（ICLR 2025，多輪共用 context，和我們最像）：KIVI 2 bit（1/8 記憶體）在 Qwen2.5-72B 明顯掉；我們是 4 bit。
- → **32K、4 bit 全過和文獻一致**；風險在更長的 context、更難或要用到整段的題、更低的 bit。

### 4. 還不知道的

- 64K／128K：NOT_MEASURED（stage 44a–44c，原 90–92；格式含主線 FP8、INT4 KIVI；「傷害隨長度變大」用 128K−32K 的配對 CI，10 §5）。
- 更低 bit（INT2、MXFP4 等）：海選（41，teacher forcing 的 first_ok，96 題）有了，見 08 §2.9。Llama 連 INT2 KIVI 全壓也只掉 1 個百分點，Qwen 掉 6.2；Qwen 的逐 token／逐 head 4-bit 格式全壞（KL 10–14）。生成的答對率要等複賽 43。
  - → 在 32K，連 3 bit 的好格式都還翻不過答案。這更支持「問題在長度和題目，不在 bit 數」，要看 44c。
- 長生成的推理題：沒測（我們的答案都短）。

### 5. 之後要注意

1. 寫「INT4 不掉」一定要加「32K、這四種題」。
2. 報告並列 Δlog p，不只放答對率（答對率看不到的小傷害，它看得到）。
3. 長 context 的 E0 可能很低（地板效應），要並列 E0。
