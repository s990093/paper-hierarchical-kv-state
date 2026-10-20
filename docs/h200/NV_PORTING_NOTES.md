# H200（NVIDIA）移植與兩台差異：寫論文時數字要注意的地方

> 使用者 10/19：「code 需要改是配在 NV 上；到時候 paper 數字要注意這些。」
> 這一頁是**總表**：每一項寫「改了什麼／哪裡不同」「會不會影響數字」「寫論文時怎麼寫」。之後每多一個 NV 專用的改動或差異，都要加在這裡。
> 細節與證據在 `results/RUNLOG_H200.md`（逐項有 run_id）、`docs/precision_20261013/13_h200_repro.md`（事先登記）。
> 第三階段（p3-predictor 搬到 H200）和 3090／MI300X 不同的地方另列在 `docs/h200/P3_PORTING_NOTES.md`（10 項，格式同這頁）；執行記錄 `results/RUNLOG_H200_P3.md`。
> 2026-10-19 寫，H200 第二階段主線、大掃描、長 context、42／43／44 都跑完；13、14 號 10/19 剛送出（job 21109–21111）。

## 0. 三條總原則（寫論文時）

1. **兩台的數字不放進同一個判定、同一張圖的同一軸**（13_h200_repro §5）。BF16 的運算 kernel 兩台不同，E0 本身就有少數題答案不同；只比「判定節點」（相同／不同），數字並排列出、各自標平台。
2. **每個 H200 數字標「H200（NVIDIA、學校 Slurm 共用節點）」**，並寫 run_id。MI300X 的數字標「MI300X（ROCm）」。
3. **下面標 ⚠ 的項目會直接改變數字或判定**，引用時要在文中或註腳寫出來；標 · 的只影響路徑、排程、牆鐘，不影響數字。

## 1. 硬體與平台

| | 項目 | MI300X | H200 | 對數字的影響／寫法 |
|:--|:--|:--|:--|:--|
| ⚠ | GPU 記憶體 | 192 GB | 141 GB（HBM3e，sm_90） | 長 context 的品質 process 能同時開幾個不同（只影響牆鐘）；H200 的工作池改成預約記憶體（`h200_pool.py`），不影響數字。256K 只在 H200 做了（Qwen、T2–T4）。 |
| ⚠ | 硬體 FP8 方言 | e4m3**fnuz**（最大 240） | OCP e4m3（**fn**，最大 448） | **我們的 codec（m10_codec）兩台都用 e4m3fn、uint8 存、軟體拆**，所以我們自己的 FP8 數字兩台是同一套數值。只有「用到硬體 FP8」的地方不同：vLLM 的 `--kv-cache-dtype fp8`（43e (d)）兩台是不同實作（MI300X fnuz＋vLLM 自己的 scale；H200 OCP＋硬體轉換），只能並列，不能說「同一個 FP8」。 |
| ⚠ | 第一階段的 FP8 Triton 反量化（`m9_f3_dequant`） | 可以跑 | **編不過**（`cannot cast int32 to fp8e4nv`，job 20614） | 照使用者 10/16 決定凍結，AMD 做完前不改。第一階段用到它的 H200 數字是 NOT_MEASURED。第二階段的融合反量化（`m10_dequant_fast`）讀 uint8 再查表，H200 上編得過、真實 KV 逐位元相同（RUNLOG_H200 job 20704）。 |
| ⚠ | 真實 CPU 層（pinned H2D） | Llama 49.9／Qwen 48.6 GiB/s | Llama 48.7／Qwen 48.4 GiB/s | 幾乎一樣（`speed_<m>/bw.json`）。 |
| ⚠ | **限速層 b16／b32／b48 的 GiB/s** | Llama 5.27／1.47／0.42；Qwen 2.77／0.76／0.22 | Llama 8.83／2.49／0.72；Qwen 4.33／1.23／0.36 | **限速層是照「會合點在第 b\* 塊」定義的**（03 §4.1：B＝(n−b\*)×chunk÷Σf），f 是各平台自己量的重算時間。H200 重算比較快，所以同一個「b32」在 H200 是更高的頻寬。**寫論文時說「會合點 b32 的那一層」，不要說「2.5 GiB/s 的那一層」**；兩台的同名層 GiB/s 不同。 |
| ⚠ | GPU 忙（背景負載） | busy2＝duty 0.7（慢 2.07 倍） | busy2＝duty **1.0**（慢 1.945 倍） | 同一個目標「約慢 2 倍」，duty 不同（`m7_write_policy_<平台>/busy_levels.json`）。busy1：MI300X duty 0.4（1.48 倍）、H200 duty 0.6（1.51 倍）。13 號 Q4 的背景強度掃描：H200 只有 busy1、busy2，沒有 MI300X 的 busy3（duty 0.9），因為 H200 的 busy2 已經是 duty 1.0。 |
| ⚠ | 共用節點 | 容器獨佔（10/19 起兩張卡） | 學校 Slurm，同一個節點上有別的 lab 的 job | gpu_guard 在 Slurm 上主機層是 UNOBSERVABLE_HOST，只能抓「自己這張卡上的外來 process」。別人的 job 搶主機記憶體／PCIe 看不到。H200 的 Llama δ\*＝0.284 來自 `cpu|busy` 一格的第一次 launch 慢 28%，原因沒查證（見 §3）。我們自己的量時間 job 一律不和自己的其他 job 同時跑。 |
| · | 卡的數量 | 1 → 2（10/19） | lab 上限 4 張、每個 job 1 張 | 只影響牆鐘。 |

## 2. 軟體

| | 項目 | MI300X | H200 | 對數字的影響／寫法 |
|:--|:--|:--|:--|:--|
| · | PyTorch／Triton／vLLM／transformers | ROCm 版 | 2.13.0+cu130／3.7.1／0.28.0／5.19.0 | vLLM 同版本 0.28.0。 |
| ⚠ | vLLM 的 attention 後端 | ROCm attention | FLASH_ATTN | 70（TTFT）、43e (d)（KV decode）的時間是各自的 production kernel，只並列。 |
| ⚠ | **Qwen2.5-1M 在 vLLM 上** | DCA 起不來 → 拿掉 DCA 的變體 | DCA 起不來（`FlashAttentionImpl ... 'layer_idx'`）→ 拿掉 DCA 的變體 | 兩台都用「config 拿掉 `dual_chunk_attention_config`、權重相同」的變體（H200：`~/models/Qwen2.5-7B-Instruct-1M-noDCA`）。≤ 262,144 token 時 DCA 在一個 chunk 內就是一般 attention；我們的 HF harness（transformers）本來就沒有 DCA。**報告 Qwen 的 vLLM 數字要註明「拿掉 DCA 的變體」**。 |
| ⚠ | tokenizer／transformers 版本 | — | 5.19 | 上游 RULER 的 CWE 產生器（邊加詞邊數 token）在兩台停在不同詞數 → **30 題一題都不同**（我們自己的產生器 30 題全同）。原因沒驗證。寫成「同樣的產生程序、不同的題目，結論相同（≈0 分）」，不能說同一批題。 |
| ⚠ | ROCm `HSA_CU_MASK`（留運算單元給還原） | 有 | **沒有** | 13 號 Q4 的 P4a／P4b 在 H200 是 NOT_MEASURED（平台）。NVIDIA 上類似的機制是 CUDA MPS 的 active thread percentage（限制背景 process 能用的 SM 比例）；機制不同（MPS 會讓 client 共用一個 context），要做就另外登記成 **H200 專屬的變體**，不和 MI300X 的 P4 並成同一欄。 |
| ⚠ | Triton 在 sm_90 的 MMA 選擇（13 號 43e (b) 的 attention 對照組） | MFMA；融合讀法 ≡ 讀 BF16，10／10 格逐位元相同 | BM＝64 走 wgmma，逐位元相同；**BM＝16（decode）走 mma.sync，融合讀法的運算元 kWidth 4、讀 BF16 是 2 → 沿 k 的加總順序不同，最後幾位不同**（誤差和 BF16 kernel 同級） | H200 的 (b)：「融合 ≡ BF16」只說到「拆出來的值逐位元相同」（用 BM＝64 再比一次證明，`M10_ATTN_BM64_CHECK`，13_h200_repro §13）。kernel 的 split 份數照 MI300X 的 304 CU 定（H200 132 SM，沒調）→ 只比同一台各讀法的相對時間。 |

## 3. 判定與挑選：兩台各自量、各自挑的（數字一定不同）

| | 項目 | MI300X | H200 | 寫法 |
|:--|:--|:--|:--|:--|
| ⚠ | δ\*（速度「不輸」的雜訊門檻） | Llama 0.05、Qwen 0.0943 | **Llama 0.284**、Qwen 0.05 | 照 03 §5.3 各平台自己量。H200 Llama 的 0.284 撐住了 Step 1 Llama 的 K1 與對照路徑的 INT4 Llama E1sr；**δ\* 用 5% 的對照**在 `results/m10_precision_h200/analysis/sensitivity_delta005/`（只報告）。引用 H200 的 K1／K3 時要提這個敏感度。 |
| ⚠ | 主線判定路徑 | K3 成立 → INT4 → Llama Step 2 → K2 | K3 不成立 → FP8 階 Qwen Step 2 → K2 | **終點相同（K2）、路徑不同**；分岔全在「CPU 層、GPU 忙」那一格（H200 上低精度比 BF16 混合慢 12–20%，用原寫法 K0／S0 的拆）。換成融合 kernel＋一組拆之後，同一格 H200 的 E1 反而比 E0 快 50–65%（42c）。 |
| ⚠ | 格式選擇 | Llama fp8_token、Qwen fp8_head、INT4 都 int4_kivi | 同 | 相同（13 §3-4 的停止條件沒觸發）。 |
| ⚠ | 反量化方法 K\*、42c 冠軍 | 冠軍 S2k8 | 冠軍 **S2k16**；K\* 各自的 `dqsweep/A/choice_<m>.json` | 13 號 44d 的「冠軍」用各平台自己的；Q4 的 P2 照定義是 S2k8（兩台同一個方法）。 |
| ⚠ | 大掃描決賽選手（41 → 43） | 各 6 個 | 各 6 個；和 MI300X 完全相同的：Llama 2、Qwen 4 | 43 複賽用各自的選手（13 §9）；43 的結論兩台都是「0／12 成立」。 |
| ⚠ | 43 的速度格子、13 號 43d | MI300X 的選手 | H200 的選手 | 同上。 |

## 4. 為 NVIDIA／H200 改的程式（全部「沒設環境變數或旗子時和 MI300X 逐字相同」）

| | 檔案 | 改了什麼 | 會不會改數字 |
|:--|:--|:--|:--|
| · | `code/env_h200.sh`、`code/m10_tasks.py` | 路徑、模型目錄 | 不會（13 §1 的程式比對：主線 16 個檔＝`bcfe5b1`，只差模型路徑） |
| ⚠ | `code/m10_sweep.py parts --busy-duty` | 原本寫死 duty 0.7 → 可指定（H200 用 1.0） | 會，照設計（GPU 忙用 H200 的 duty） |
| ⚠ | `code/m10_speed.py`、`code/m10_attn_q.py` 的 `--busy-levels` 預設讀 `M10_BUSY_LEVELS` | 原本預設 MI300X 的 busy_levels | 會，照設計 |
| · | `m10_confirm`、`m10_confirm_analyze`、`m10_dqsweep`、`m10_tier2`、`m10_longctx_analyze`、`m10_iowall`、`m10_restore2` 的目錄 | `M10_SWEEP_DIR`、`M10_PRECISION_DIR`、`M10_RUNS_DIR`、`M10_RESTORE2_DIR` | 不會（讀寫 H200 自己的目錄） |
| ⚠ | `m10_dqsweep.py report` | δ\* 與「主線／42」參考欄原本寫死讀 MI300X 的檔 | 修之前第一次跑出的 H200 表那一欄是 MI300X 的數字（已發現、修好、重跑；RUNLOG_H200 10/19）。**引用 42c 表時用 10/19 重跑那份** |
| · | `m10_ruler_cwe_vllm.py`、`m10_vllm_ttft.py`、`m10_vllm_kvdecode.py` | 模型路徑讀 `M7_<MODEL>_GLOB`；Qwen 指到拿掉 DCA 的變體 | 見 §2 |
| · | `m10_restore2.py smoke-check` | 只有遮罩生效（OK_cumask）時才要求「有傳遮罩」 | 不會（H200 沒有遮罩，否則冒煙會誤判不過） |
| · | `m10_attn_q.py selftest` | `M10_ATTN_BM64_CHECK=1`：BM＝16 逐位元不同時，同一個 split 改 BM＝64 再比，逐位元相同才算過（13_h200_repro §13）；微測（bench）不動 | 不會（只決定 (b) 跑不跑；沒設時不變，CPU 直譯自測 ALL PASS） |
| · | `m10_restore2.py report` | 背景強度欄位照 `busy_levels_q4.json`（H200 沒有 busy3）、Q3 的原寫法路徑照 `M10_SWEEP_DIR`、(d) 標 CUDA／ROCm | 不會（只改標籤；MI300X 的表與 JSON 重跑逐字相同） |
| · | `code/h200_queue/`、`r_h200_rs2.sh --part attnb` | H200 才有的 stage（43e0nv 自測、43eb 補跑 (b)），不經轉換 | 不會 |
| · | `code/h200_pool.py` | 工作池加記憶體預約（20705 OOM 的修正） | 不會（只影響同時開幾個 process） |
| · | `code/h200_longctx_plan.py` | 試跑時間改成每題取實際跑的那次（screen 會續跑）；`LONGCTX_VARIANT=256` | 題數規則不變；32K–128K 選出 n＝12（同 MI300X） |
| · | `code/h200_longctx256_analyze.py` | 256K 的彙整另寫（借用 `m10_longctx_analyze` 的函式） | 不會；256K 只有 T2–T4，和 32K–128K 的四任務平均不能直接比（表裡都只取 T2–T4） |
| · | `code/h200_port_stage.py`、`code/r_h200_*.sh` | MI300X 的佇列腳本換路徑、run 名稱（m10h-） | 不會 |
| · | `m10_cake_ratio.py`、`m10_int4_sensitivity.py` | `M10_PRECISION_DIR`（沒有 Qwen regrid 就用 speed_qwen）、`M10_SENS_OUT` | 不會（MI300X 資料回歸逐字相同） |
| ⚠ | `m10_iowall.py` | `M10_IOWALL_GRIDS`：run 編號表換成 H200 的主線格子（`results/m10_tier_h200/iowall_grids_h200.json`） | 會，照設計（用 H200 自己量的格子）；Qwen FP8 用原格子（H200 沒有重量那次） |
| ⚠ | `code/tests_h200/` | Triton 在 sm_90 的 AOT 編譯測試、反向對照 | 證據，不是數字 |

## 5. 只在一台有的結果

| 只在 MI300X | 只在 H200 |
|:--|:--|
| 13 號 P4a／P4b（CU 遮罩） | 256K（Qwen、T2–T4） |
| 44cb（同容器兩張卡的干擾檢查） | δ\* 5% 的對照 |
| 22、23（Qwen 重量；MI300X 原 run 被標污染） | Qwen 的 TTFT（MI300X 的 70 排在後面，10/19 還沒跑） |

## 6. 寫論文前的檢查清單

- [ ] 每個 H200 數字有 run_id、平台標籤；沒有和 MI300X 的數字放進同一個判定或同一軸。
- [ ] 限速層用「會合點 b\*」稱呼，不用 GiB/s 當名字（§1）。
- [ ] 引用 GPU 忙的格子時寫各平台的 duty（§1）。
- [ ] 引用 H200 的速度「不輸」判定時，提 δ\* 0.284 與 5% 的對照（§3）。
- [ ] vLLM 的數字：Qwen 註明拿掉 DCA；FP8 KV 註明兩台方言不同（§1、§2）。
- [ ] RULER 上游 CWE：寫「不同題目」（§2）。
- [ ] 42c 表用 10/19 重跑那份（§4）。
- [ ] CU 遮罩的結果只寫 MI300X；H200 若做 MPS 變體，另外一欄（§2）。
- [ ] 第三階段（地圖、探索 X1–X8、格式表）的 H200 數字：照 `P3_PORTING_NOTES.md` 標（成本常數只換 GPU 量的、CPU 忙 H2D 倍數 NOT_MEASURED、格式表沒有 3090 欄）。
