# P1-1　壓縮疊加（跨層低秩＋INT4）在 H200 重現：品質＋慢層的實測還原

> ICPE 補量 P1-1（[README](README.md)）。平台 H200（hgx-cn01）。**§1–§6 是事先登記**，結果寫在 §7 以後，登記那幾節不改。
> **登記寫於 2026-10-22T04:21:13Z（`date -u`），repo HEAD `9f67df9`**；寫的時候 H200 上沒有任何這些配置的資料，P1-1 的程式也還沒寫。
> 標記同 repo 慣例。MI300X 的對照全部來自 `origin/main` 的 `results/m16_stage3_mi300x/x_stack/`（X4、v2、v3）與 `x3_time/v2/validate/`。

---

## 1. 問題

論文說「不用預知就有用的是把位元組變小：INT4 3.2 倍；低秩＋INT4 約 8 倍」，後者**只在 MI300X 量過**：

- 品質（X4b，192 題，seed 65000）：Qwen `xl:k34:v51+q:int4_kivi`（8.01 倍）對 E0 −0.7 [−3.5, +2.0]，過 MLPerf 99%、沒過 ε；Llama `xl:k19:v29+q:int4_kivi`（12.14 倍）−3.6 [−6.9, −0.5]；Llama 8 倍帶 TF 最好的 `xl:k29:v44+q:int4_kivi`（7.98 倍）。
- 事先校準的共用基底（`:cal`，38k，192 題）：兩個模型都 UNDETERMINED，點估計 −1.5 到 +2.1；好處是不用每題做數十秒的 SVD。
- 時間（X3 validate，`results/m16_stage3_mi300x/x3_time/v2/validate/`）：**STOP-3**——模型低估 5–26%（25 格裡 6 格誤差 >10%）。實測：SSD 級 xKV＋INT4 比 INT4 快 2.5–3.4 倍；CPU 層、GPU 忙（duty 0.9）時 xKV＋INT4 比 BF16 還慢。→ H200 要**直接實測**，不用模擬。

**要做的**：(Q) 在 H200 上用新題重量這幾個配置的生成品質；(T) 在慢層（b32、NFS 級）實測端到端還原時間，INT4 對 低秩＋INT4，GPU 閒／忙；並量壓縮本身的時間。

## 2. (Q) 品質

- **配置**（事先列死；`:cal` 是主角，每題 SVD 版是和 X4b 同一個配置的對照）：

| 模型 | 配置 | 角色 |
|:--|:--|:--|
| Qwen | `none`（E0）、`q:int4_kivi` | 對照 |
| Qwen | `xl:k34:v51:cal+q:int4_kivi` | 主（8 倍，校準基底） |
| Qwen | `xl:k34:v51+q:int4_kivi` | X4b 同配置（每題 SVD） |
| Llama | `none`（E0）、`q:int4_kivi` | 對照 |
| Llama | `xl:k29:v44:cal+q:int4_kivi`、`xl:k19:v29:cal+q:int4_kivi` | 主（8 倍、12 倍，校準基底） |
| Llama | `xl:k29:v44+q:int4_kivi`、`xl:k19:v29+q:int4_kivi` | X4b 同配置（每題 SVD） |

- **題目**：32K，T2–T5 各 48 題＝192 題，**seed0＝67000**（新；程式核對不和 X1、PCA／xKV 校準、X2、X2b、X4a、X4b、38k（66000）、C1a、44c 重疊）。同一題的所有配置在同一個行程裡配對。
- **校準基底**：H200 自己算（`m16_xstack.py calib-xl --model <m>`：seed 61000、T2–T5 各 2 題、聯合秩存到每 head 等效 64；和 MI300X 38c 同一套校準題），sha256 記下。不借 MI300X 的檔（不在這台）。
- **路徑**：`m16_xstack.py gen --model <m> --selection <P1-1 的選擇檔> --ctx 32K --tier core --n 48 --seed0 67000 --calib-xl <校準檔> --xl-cache --out <逐列>`（和 38k 同一條生成路徑；位元組照 `predict_bytes`，`:cal` 的共用基底另列）。H200 上要改的只有路徑（環境變數，MI300X 預設不變），程式先過 `gpu-selftest`（`--new-checks both`）再跑。
- **指標**：答對率；對 E0、對 `q:int4_kivi` 的逐題配對差＋95% bootstrap CI（`m16_xstack.boot2`，同 X4b）；ε�＝CI 下界 ≥ −2 個百分點；MLPerf 99%＝答對率 ≥ E0 的 99%（點估計）。
- **判定（描述，不是 go/no-go）**：
  - Q-a「Qwen 約 8 倍接近無損」：主配置 `xl:k34:v51:cal+q:int4_kivi` 過 MLPerf 99% → 在 H200 成立；另報每題 SVD 版。
  - Q-b「Llama 8–12 倍約掉 1–3 個百分點」：照實報兩個主配置的點估計與 CI。
  - Q-c 和 MI300X 相容：三個 X4b 同配置，H200 對 E0 的 CI 和 MI300X X4b 的 CI 有交集 → 相容；ε／MLPerf 99% 的判定字相同 → 判定相同。
  - Q-d 校準 vs 每題 SVD：每對照 `m16_xstack.XLCAL_TEXT['rule']` 原文判 CAN_PRECOMPUTE／MUST_PER_ITEM／UNDETERMINED。
  - 題數 < 48 的格寫 INCOMPLETE。

## 3. (T) 時間

- **格子**：模型 {Llama, Qwen} × 層 {b32（H200 各模型自己 `bw.json` 的 b32：Llama 2.49、Qwen 1.23 GiB/s）、NFS 級 0.331 GiB/s} × GPU {閒, 忙（duty 1.0）}＝8 格。CPU 層（真實 pinned H2D）只報告。32K（64 塊）。
- **INT4**（`q:int4_kivi`）：
  - 端到端（harness 的 `m10_speed grid`，一次拆 S4，Cake 與全部載入取較快者）：**直接用 P0-1 (b) 的 sdpa 那一組**（同樣的格子、同一個 job 量的；P0-1 §3.2）。
  - 全部載入的載入線（和 xKV 同一種量法，給同量法的對照）。
- **低秩＋INT4**：上面 (Q) 的三個秩組合（k34v51、k29v44、k19v29），各兩種：每題 SVD（每次還原多搬一份基底）、`:cal`（共用基底常駐 GPU，不搬）。量法＝`m16_xtime.gpu_loadpath`（全部載入：從最後一塊往前、限速器＋真實 H2D、最後一次拆＋重建；MI300X validate 的同一個函式）。xKV 沒有 Cake 路徑（第一階段的還原器不支援這種格式），所以只量全部載入。
- **壓縮本身**：每題 SVD（`svd_lowrank`，m16_xtime 的 wcost，同 MI300X 38a）與 `:cal` 的投影（Z·U＋係數 INT4 量化），32K 區域，GPU 閒／忙。
- **H200 移植**（環境變數，MI300X 預設不變）：busy 等級、K\*（`M10_SWEEP_DIR`）、格式選擇改讀 H200 的檔；新的秩組合與 `:cal` 段用 `M16_XKV_EXTRA` 加（沒設時 m16_xtime 的段清單逐位元不變）。
- **判定**：
  - 每格「低秩＋INT4 在這一層比 INT4 快」＝`:cal` 版的全部載入時間比 INT4 端到端最好的還快 >5%；8 格都成立 → 「慢層上低秩＋INT4 比 INT4 快」在 H200 成立。另報每題 SVD 版（多搬基底）的同樣判定，以及快幾倍。
  - 只報告：CPU 層的同樣比較（MI300X 在 CPU 層、GPU 忙時 xKV 反而慢）；位元組下界（位元組 ÷ 頻寬）÷ 實測；壓縮時間 ÷ 同長度 prefill。
  - 不做模擬對實測的 STOP-3（使用者：H200 要實測不要模擬）；只把「位元組下界」列出來當參考。

## 4. 停損、失敗

- 品質：`gpu-selftest` 沒過 → 停；任何配置的 KV 檢查或位元組帳對不上 → 停。題數不夠的照 INCOMPLETE。
- 時間：gpu_guard 污染的格重量一次，第二次還污染就停；載入線的位元組和位元組帳對不上 → 停（m16_xtime 本來就斷言）。
- 品質 job 不量時間，但一樣不和我們自己的量時間 job 重疊（同一台主機）。
- 任何指令失敗：完整錯誤記下、停下來回報。

## 5. 時間〔估計〕

品質：Llama 約 1.5–2 卡小時、Qwen 約 1 卡小時（每題 prefill＋6／4 個配置的生成＋每題 SVD；MI300X 38k 的同類 job 約 2.6 卡小時／192 題×2 對）。校準基底每個模型約 10 分。時間：約 30–40 分。

## 6. 和登記之前就知道的事

- MI300X 的數字（§1）在寫這份之前就知道；P0-1 (b) 的 INT4 數字在寫這份時還沒有。H200 上這些配置的任何數字都還沒有。

---

## 7. 結果

（還沒有。）

## 8. 偏離登記

（還沒有。）
