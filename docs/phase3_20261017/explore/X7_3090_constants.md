# X7　換到 3090：成本常數變了，會不會出現 MI300X 上沒有的情況？

（§1、§2 在任何 X7 的 GPU 量測之前寫好。寫的時候看過的 3090 數字只有：`nvidia-smi` 的卡況與拓撲（7 張 3090、PCIe link gen max＝3、GPU 0–3 與 4–6 各在一個 PCIe switch 後面）、環境 agent 的冒煙輸出 `20261017-174142-p3-0-prefill-qwen-2k`（Qwen 2K chunk 化 prefill 0.98 s、peak 14.4 GiB，冒煙等級），以及模型權重的 sha256 驗證（`20261017-173936-p3x7-sha`，8／8 OK）。§0、§3 以後是結果出來才寫的。）

## 0. 一句話結論＋判定

**判定：可能**（照 §2.2 字面；`results/p3_explore/x7/verdict.json`，self-review）。

- 3090 上真實 CPU 層的會合點還是在最前面：Qwen 32K 實測 **閒 3、忙 2**（MI300X 2／1）；Llama 的還原格子兩次被別人的行程污染，實測 NOT_MEASURED，兩條線模型（在 Qwen 上 16／16 格差 ≤1 block）預測 4／3（MI300X 實測 3／2）。原因：這台的 PCIe 是 **Gen3**、pinned H2D 只有 **≈7 GiB/s**（MI300X 50），CPU 層一個 block 的載入慢 7 倍；但重算也慢 3.2–5.3 倍，f(0)÷ℓ 只從 22–42 掉到 15–32。→ (a) 不成立。
- Cake 有用的帶（TTFT ≥1.3×）整條往低頻寬移 3–4 倍：Qwen 閒 0.04–0.60、Llama 閒 0.07–1.27 GiB/s（MI300X 用同一個模型：0.13–2.56、0.25–4.89）。3090 的 CPU 層（≈7 GiB/s）在帶外 → (b) 不成立。
- 但 3090 的帶下緣比 MI300X 低，多出約 0.02–0.25 GiB/s 這段「MI300X 上只算比較快、3090 上 Cake 有用」的頻寬 → (ii) 不成立，所以照字面**不是死路**。〔判讀〕這段新區域比論文裡最慢的層（NFS 0.331 GiB/s）還慢，**CPU 層本身沒有出現新的「讀取時要決定」的事**；實質上接近死路。
- 只報告、但最值得追：**3090 上低精度讓真實 CPU 層變快**（Qwen FP8 只載入 −45%、Cake −30%；INT4 只載入 −51%），和 MI300X（main.tex §5.1：CPU 層低精度變慢）相反——H2D 慢 7 倍、反量化只慢 2–4 倍。這會直接改變 X5 的取捨。
- 常數：`results/p3_explore/x7/constants_3090.json`（f(i)、ℓ、忙 1.91×、b\* 頻寬、反量化、Llama 最長 64K），給 X1／X4／X5 加跑，標〔3090 常數〕。

## 1. 問題與要證明什麼

- 第一階段在 MI300X 上：CPU 層載入一個 block 比重算最便宜的 block 快 16 倍，所以 Cake 在 CPU 層的會合點只在 block 0–3（main.tex §4.1），「讀取時決定」在 CPU 層幾乎沒事做；Cake 有用的頻寬帶在 0.33–3.73 GiB/s（32K，1.38–1.93×）。
- 3090 的重算比較慢、PCIe 比較慢（這台是 Gen3，不是 Gen4，見 §3），兩者都會動，方向相反：重算慢 → 會合點往前、帶往低頻寬移；PCIe 慢 → CPU 層的 ℓ 變大 → 會合點往後。**哪一個贏要量。**
- 要回答：3090 上有沒有出現「CPU 層的讀取時決定很重要」的區域（MI300X 上沒有的）？
- 附帶產出：3090 的成本常數 `results/p3_explore/x7/constants_3090.json`，給 X1、X4、X5 加跑用（分開列，標〔3090 常數〕）。

## 2. 判準（開跑前寫）

**寫於**：2026-10-17T09:46:06Z（`date -u`）。可以比 README §2 X7 細，不比它鬆。

### 2.1 量什麼（定義寫死）

| 量 | 定義 | 工具 |
|:--|:--|:--|
| f(i) | 一個 512-token block 的重算時間（GPU 閒，CUDA 同步牆鐘），block 0–63（32K），每個 block 3 次取中位數。Qwen 到 32K；Llama 32K，另外往上試（40K、48K、…）到 OOM 為止，OOM 照實記 | `m10_speed.py calib-f`（MI300X 同一支）；長度階梯用 `m7_calib.py c1` |
| 真實 CPU 層頻寬 | pinned 主機記憶體 → KV 緩衝的 strided 視圖，一個 BF16 block（Llama 64 MiB、Qwen 28 MiB），中位數；另記 C0 的連續 64 MiB／1 GiB pinned／pageable H2D、D2H，以及 H2D 與 D2H 同時跑 | `m10_speed.py calib-f`、`m7_calib.py c0`、`m7_duplex.py` |
| GPU 忙的變慢倍數 | 背景 `m7_busy.py` duty 0.7 時，block 32 的重算時間 ÷ GPU 閒時（C5 的定義） | `m7_write_policy.py busy-calib` 或 `code/p3x/x7_busy.py`；另記背景實際做到的 duty（matmul 次數 × 單次時間 ÷ 牆鐘），因為 3090 的 matmul 比 MI300X 慢很多，0.7 的設定不一定真的是 0.7 |
| 每個 block 的反量化時間 | Llama `fp8_token`、Qwen `fp8_head`、兩者 `int4_kivi`（和 MI300X 主線同格式）；三種寫法：主線 PyTorch（`m10_codec.decode`）、融合 Triton（`m10_dequant_fast`）、`torch.compile`；另記還原時（真實 CPU 層）實際的 `t_dequant` | `m10_dequant_fast.py --self-test --bench`、`code/p3x/x7_restore.py` |
| 會合點 b | Cake 還原時重算線做了幾個 block（`n_recompute`），32K（64 block）；每格中位數 | `code/p3x/x7_restore.py`（import `m7_restore_harness`，不改它） |
| Cake 有用的頻寬帶 | 兩條線模型（`m7_restore_harness.write_boundary` 的同一個算式：T(m)＝max(Σ_{i<m} f(i), (n−m)·ℓ)，Cake＝min_m T(m)；只算＝Σf；只載＝n·ℓ；ℓ＝block 位元組 ÷ 頻寬）算出 Cake 對「只算、只載中較快者」的加速比，**≥1.3× 的頻寬區間**＝帶。1.3 的理由：MI300X 實測帶內 1.38–1.93×、帶外 1.17×（6.98 GiB/s）與 1.05×（11.6 GiB/s），1.3 剛好把兩邊分開。另報 1.2×、1.38× 的帶當敏感度（不改判定）。同一個模型、同一個門檻用 MI300X 的 f 算 MI300X 的帶，兩台並列 | `code/p3x/x7_band.py`（只用 CPU） |
| 模型驗證 | 在 3090 上真的跑 Cake（限速層）至少 3 個頻寬，模型預測的 b 和實測中位數差 ≤1 個 block 的格子要 ≥2/3；不過就不用模型帶下判定，改用實測點，照實寫 | `code/p3x/x7_restore.py` |

**補充（2026-10-17T10:03Z，算任何帶之前）**：帶的加速比用 **TTFT**（還原＋新一輪 256 token），和 main.tex Fig. cake_band 與第一階段 A0（`summary_a0_speedup.csv` 用 ttft）同一個定義，1.3 的門檻也是從那些 TTFT 數字來的；新一輪的時間用同一台、同一模型、32K 實測的 `t_new_turn` 中位數。只看還原時間的帶另報（不改判定）。模型的「Cake」用兩條線的貪婪模擬（先空下來的線領下一個 block，和 harness 的行為相同），最佳切點（write_boundary）另報。

**補充 2（2026-10-17T13:36Z，看過除錯輸出之後，照實寫）**：用 Qwen 的還原格子除錯 `x7_band.py` 時（Llama 的新一輪時間當時還沒量，借用 Qwen 的值，那次輸出不用），看到加速比在 1.3 附近因為 block 是整數而鋸齒，帶被切成 2–5 段。所以把「帶」寫死成**外包絡**（加速比曾經 ≥ 門檻的最低到最高頻寬），另列含峰值的那一段（main）。兩種定義在 (b)、(ii) 的結論相同時才算數；不同就照實寫兩個，判定取較不利於「有看頭」的那個。

### 2.2 判定（照字面）

- **有看頭**：下面任一成立。
  - (a) 3090 真實 CPU 層（BF16、不限速、真的 H2D）、32K 的會合點中位數 ≥8 個 block，在任一（模型 × GPU 閒／忙）格；Llama 放不下 32K 時用放得下的最長長度，照 block 數比例換算成 64 block 的等價值另報，但判定用實際 block 數。
  - (b) 3090 的 Cake 帶（≥1.3×，GPU 閒或忙，任一模型）涵蓋 3090 實測的真實 CPU 層頻寬（該模型 block 的 strided H2D 中位數）。
- **死路**：下面兩條同時成立。
  - (i) 3090 真實 CPU 層的會合點中位數在每一格都 ≤3（MI300X 的 CPU 層範圍是 block 0–3，main.tex §4.1）；
  - (ii) 3090 的 Cake 帶（≥1.3×）在每個模型、每個 GPU 狀態都包含在 MI300X 用同一模型、同一門檻算出的帶裡面（下緣 ≥ MI300X 下緣、上緣 ≤ MI300X 上緣）。
- **可能**：其他。
- 只報告、不改判定：3090 在 MI300X 的 b\* 頻寬（Llama 5.27／1.47／0.42、Qwen 2.77／0.76／0.22 GiB/s）與 MI300X 的兩個真實裝置參數（本地 SSD 6.98、NFS 0.331 GiB/s）上的會合點；3090 自己的 b\*∈{16,32,48} 頻寬；CPU 層上低精度（FP8、INT4）是比 BF16 快還是慢（MI300X 上 CPU 層 INT4 讓 Qwen 忙時慢 2.2 倍）；Cake 在 CPU 層是否比只載入慢（MI300X 上 0.165 vs 0.123 s）。
- **污染**：`GpuWatcher` 標污染的 run，時間欄作廢重量，不進 `results/`。同 PCIe switch 的其他卡（外人或環境 agent）有負載時，H2D 量測另記 `host_contention` 等級與同 switch 卡的 PCIe 流量，不當成乾淨；結論只用 QUIET 或「同 switch 卡 PCIe 流量可忽略」的量測，其他的照實列。

## 3. 做了什麼

全部在 `/ssd8/hungwei/paper-hkv/runs/<run_id>/`（`cmd.sh`、`context.txt`、`stdout.log`、`stderr.log`、`exit_code`、`gpu_guard.json`）。GPU 1（PCIe switch 在 GPU 0–3 那一組，NUMA 0），`taskset -c 0-23,48-71`（GPU 的 NUMA 節點）。記錄殼前段用 `code/run3090.sh` 的快照（`/ssd8/hungwei/paper-hkv/logs/p3x7/run3090.snap.sh`，sha256 `01f1039e…`，和之後提交的 `code/run3090.sh` 相同），後段用 `source code/env_3090.sh; m7run`。

| run_id | 做什麼 | 工具 | 狀態 |
|:--|:--|:--|:--|
| `20261017-173936-p3x7-sha` | 模型權重 sha256（LFS blob 名＝sha256） | `sha256sum -c` | 8／8 OK |
| `20261017-174813-p3x7-probe` | SDPA 在 sm_86 上走哪個 kernel；PCIe 負載下的世代 | `code/p3x/x7_probe.py` | CLEAN |
| `20261017-174843-p3x7-c0a` | C0 H2D／D2H（GPU 1＋taskset）＋duplex | `m7_calib.py c0 --reps 5`、`m7_duplex.py` | CLEAN |
| `20261017-174933-p3x7-c0-nopin` | 同上，不 taskset | `m7_calib.py c0` | CLEAN |
| `20261017-175001-p3x7-c0-g6` | 同上，GPU 6（另一個 switch，NUMA 1） | `m7_calib.py c0` | CLEAN |
| `20261017-175055-p3x7-calibf-qwen` | Qwen f(i) 32K×3 次、Qwen block 的 strided H2D、3090 的 b\* 頻寬 | `m10_speed.py calib-f` | CLEAN |
| `20261017-175208-p3x7-calibf-llama` | 同上，Llama | `m10_speed.py calib-f` | CLEAN |
| `20261017-175327-p3x7-llama-ladder` | Llama 長度階梯 40K→68K，f(i) 到 OOM | `m7_calib.py c1 --reps 1` | CLEAN；68K OOM |
| `20261017-180058-p3x7-busy-qwen`、`20261017-180431-p3x7-busy-llama` | GPU 忙的變慢倍數，4 種背景設定，block 0／32／63 | `code/p3x/x7_busy.py` | CLEAN |
| `20261017-180859-p3x7-dqbench` | codec 自測；反量化 torch／Triton 融合／torch.compile | `m10_codec.py --self-test`、`m10_dequant_fast.py --self-test --bench` | CLEAN |
| `20261017-181306-p3x7-grid-qwen-L0` | Qwen 32K 還原格子：9 個頻寬 × {Cake、只載、只算} × {BF16、FP8、INT4} × GPU 閒／忙 × 5 次 | `code/p3x/x7_restore.py`（`logs/p3x7/grid_qwen.sh`） | CLEAN，210 列，kv_bad＝0 |
| `20261017-213420-p3x7-grid-qwen-L1` | Qwen 第二個 launch | 同上 | **CONTAMINATED**（§6），作廢 |
| `20261017-215041-p3x7-grid-llama-L0` | Llama 32K 還原格子 | 同上（`grid_llama.sh`） | **CONTAMINATED＋OOM**（§6），沒有量到 |
| `20261017-221130-p3x7-grid-llama-L0` | Llama 32K 還原格子（第 2 次，GPU 6） | 同上 | **CONTAMINATED**（§6），作廢 |
| `20261017-181500-p3x7-mi300xref` | 從 repo 裡已提交的 MI300X 檔讀對照值 | `code/p3x/x7_mi300x_ref.py`（CPU） | — |
| `20261017-223447-p3x7-summary` | 整理成 `results/p3_explore/x7/`、同 switch 鄰居的 PCIe 流量標記、兩條線模型、判定（最後一次；`215350`、`215826`、`223229` 是之前的版本） | `x7_summarize.py`、`x7_pcie_flag.py`、`x7_band_spec.py`、`x7_band.py`、`x7_compare.py`（CPU） | — |

產出（`results/p3_explore/x7/`，每個 CSV 有 `run_id`、`ts`、`summarize_run_id`）：`h2d_3090.csv`、`f_3090.csv`、`busy_3090.csv`、`dequant_bench_3090.csv`、`restore_3090_summary.csv`、`meet_compare.csv`、`model_check.csv`、`band_points.csv`、`band_edges.csv`、`mi300x_ref.csv`、`constants_3090.json`、`verdict.json`、`runs.json`（收了哪些 run、拒了哪些、為什麼）、`band_spec.json`（模型輸入）。

GPU 時間（各 run 的 context→context_end 加總）：約 63 分鐘，其中作廢的約 30 分鐘（Qwen L1 12 分鐘、Llama L0 第 1 次 0.5 分鐘、第 2 次 17 分鐘）。上限 4 小時。

## 4. 結果

### 4.1 這台 3090 的 I/O 路徑

- PCIe：`nvidia-smi` 的 `pcie.link.gen.max`＝3、`width`＝16；閒置時降到 Gen1，連續 H2D 時是 **Gen3 x16**（`20261017-174813-p3x7-probe`）。GPU 0–3、4–6 各在一顆 **PLX PEX 8796**（Gen3 switch）後面（`lspci -tv`；`nvidia-smi topo -m` 的 PIX），switch 的上行接 Xeon Gold 6248R 的 root port。**不是** migration README 寫的 PCIe 4.0。
- pinned H2D 只有 **6.0–7.1 GiB/s**（Gen3 x16 的理論值 15.75 GB/s＝14.7 GiB/s，量到的約 48%）。量過的三種：GPU 1＋taskset 7.05（1 GiB 連續）／6.99（一個 Llama block 進 KV 的 strided 視圖）；GPU 1 不 taskset 6.17／6.22；GPU 6（另一顆 switch、NUMA 1）6.57／5.88（`h2d_3090.csv`）。為什麼只有一半：NOT_MEASURED（沒有 root 看不到 switch 的 LnkSta／MaxReadReq；量測時主機 loadavg 約 45，別人有大量 CPU 行程；C0、calib-f 期間同 switch 其他卡的 PCIe 流量 ≤0.05 GB/s，`dmon_pcie.txt`）。
- 換算：3090 的 CPU 層≈7 GiB/s，**剛好和 MI300X 的「本地 SSD」限速參數（6.98 GiB/s）一樣**。

### 4.2 常數對照表（32K＝64 block；3090 全部是 GPU 1 的 CLEAN run）

| 量 | 3090 | MI300X | 3090÷MI300X | 3090 run_id | MI300X 來源（run_id） |
|:--|--:|--:|--:|:--|:--|
| PCIe | Gen3 x16（PLX switch） | Gen5 x16 | — | `174813-probe` | 07 §2.1 |
| pinned H2D，1 GiB 連續 | 7.06 GiB/s | 53.6 | 0.13 | `174843-c0a` | `130257-m7-c0` |
| pinned H2D，Llama block→KV 視圖 | 6.99（C0）／6.99（calib-f） | 50.0（C0）／49.9 | 0.14 | `c0a`、`175208-calibf-llama` | `130257-m7-c0`、`154558-m10-calibf-llama` |
| pinned H2D，Qwen block→KV 視圖 | 6.91 | 48.6 | 0.14 | `175055-calibf-qwen` | `154647-m10-calibf-qwen` |
| pinned D2H，1 GiB | 7.13 | 45.2 | 0.16 | `c0a` | `130257-m7-c0` |
| H2D 同時有 D2H | 6.04（單獨 7.24，−17%） | 46.1（單獨 53.5，−14%） | — | `c0a` | `20261008-132007-m7-duplex` |
| pageable H2D，1 GiB | 3.56 | 52.3 | 0.07 | `c0a` | `130257-m7-c0` |
| ℓ（CPU 層每個 block） | Llama 8.94 ms、Qwen 3.96 ms | Llama 1.28 ms、Qwen 0.56 ms（真實 H2D；論文的限速器 35.4 GiB/s 是 1.76 ms） | 7.0、7.0 | calib-f | m10 calib-f |
| f(0) | Llama 130.4 ms、Qwen 126.1 ms | 29.0、23.6 | 4.5、5.3 | calib-f | m10 calib-f |
| f(63) | Llama 288.2、Qwen 233.8 | 84.8、72.5 | 3.4、3.2 | calib-f | m10 calib-f |
| f 斜率 | Llama 2.51、Qwen 1.71 ms/block | 0.886、0.775 | 2.8、2.2 | calib-f | m10 calib-f |
| Σf（32K 全部重算） | Llama 13.42 s、Qwen 11.48 s | 3.63、3.08 | 3.7、3.7 | calib-f | m10 calib-f |
| f(0)÷ℓ_CPU | Llama 14.6、Qwen 31.9 | 22.6、42.1（真實 H2D）；16.4（論文，限速器） | 0.65、0.76 | — | — |
| Llama 放得下的最長長度（harness，單卡） | 64K（128 block）；68K 在配置 KV 時 OOM（「Tried to allocate 8.56 GiB … 8.34 GiB is free」） | 192 GiB，不受限 | — | `175327-llama-ladder` | — |
| Llama f(127)（64K，1 次） | 455.0 ms；Σf（64K）37.39 s | NOT_MEASURED | — | ladder | — |
| GPU 忙（`m7_busy --duty 0.7`，週期 20 ms，8192²）變慢 | Llama 1.906×、Qwen 1.918×（block 0／32／63 都在 1.88–1.92） | 2.067×（Llama，block 32） | 0.92 | `180431`、`180058` | `130956-m7-c5b`（calib_c5.csv） |
| 同上，背景單獨跑時 GPU 使用率 | 69.7%、70.1%（一次 matmul 15.8 ms） | 使用率 NOT_MEASURED；一次 matmul 33／35 筆 1.646–1.671 ms（m10 dqsweep 的 busy JSON） | — | 同上 | — |
| b\*＝16／32／48 的限速頻寬 | Llama 1.257／0.369／0.110；Qwen 0.591／0.179／0.055 GiB/s | Llama 5.275／1.467／0.420；Qwen 2.771／0.764／0.218 | 0.21–0.26 | calib-f | m10 calib-f |
| 新一輪 256 token（32K） | Qwen 閒 0.132 s、忙 0.251 s；Llama 閒 0.143 s（c1 在 block 64，1 次）、忙 NOT_MEASURED | Qwen 0.066／0.209；Llama 0.075／0.236 | 2.0 | `181306-grid-qwen-L0` | m10 grid（`mi300x_ref.csv`） |

GPU 忙的另外三種設定（只報告）：`--duty 0.4` 實際使用率 54%（matmul 15.8 ms 比 8 ms 的「開」還長，duty 設不準）、變慢 1.71–1.72×；`--period-ms 200` 1.67–1.69×；`--n 4096` 1.64–1.72×（`busy_3090.csv`）。**3090 上 MI300X 的 0.7 設定剛好做得到約 70%**，所以主格子用同一個設定。

### 4.3 反量化（每個 block）

| 寫法 | Llama `fp8_token` | Llama `int4_kivi` | Qwen `fp8_head` | Qwen `int4_kivi` |
|:--|--:|--:|--:|--:|
| 主線 PyTorch（`m10_codec.decode`），單獨量 | 0.976 ms | 2.053 ms | 0.456 ms | 0.925 ms |
| 融合 Triton（`m10_dequant_fast`），單獨量 | 0.172 | 0.247 | 0.123 | 0.149 |
| `torch.compile` | **跑不起來**（下面） | 0.324 | **跑不起來** | 0.175 |
| 還原時實測（CPU 層、只載入、GPU 閒，主線寫法） | NOT_MEASURED | NOT_MEASURED | 0.432 | 0.874 |
| 同上，GPU 忙 | NOT_MEASURED | NOT_MEASURED | 0.674 | 1.373 |
| MI300X 還原時實測（CPU 層、只載入，閒／忙） | 0.233／0.888 | 0.447／1.701 | 0.105／0.406 | 0.234／0.949 |

（3090：`20261017-180859-p3x7-dqbench`、`181306-grid-qwen-L0`；MI300X：`20261015-161226-m10-grid-fp8-llama`、`20261016-034902-m10-grid-int4-llama`、`20261015-175050-m10-grid-fp8-qwen`、`20261016-052143-m10-grid-int4-qwen`。MI300X 沒有存單獨的 bench 數字：`results/m10_sweep_mi300x/dqfast_selftest_gpu.json` 只有 checks → NOT_MEASURED。）

- **Triton 融合 kernel 在 CUDA sm_86 上能跑**，而且和主線逐位元相同（`m10_dequant_fast --self-test` ALL PASS；`m10_codec --self-test` 62 項 PASS）。它不碰 float8 dtype（FP8 用 256 項查表），所以 sm_86 沒有 FP8 也沒關係。
- **`torch.compile` 的 FP8 反量化在 sm_86 上編不起來**（Inductor 產生的 Triton kernel 用到 e4m3fn），完整錯誤（stderr.log 第 39–40 行，三種 FP8 粒度都一樣）：
  `ValueError: type fp8e4nv not supported in this architecture. The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')` → `triton.compiler.errors.CompilationError` → `InductorError`。INT8／INT4 的 compile 版逐位元相同。
- 主線 PyTorch 寫法在 3090 上每個 block 慢 2–4 倍（相對 MI300X），但 3090 的 H2D 慢 7 倍，所以**「少搬的位元組」比「多做的反量化」划算**（§4.6）。

### 4.4 會合點（BF16、32K；`meet_compare.csv`）

| 頻寬（GiB/s） | 3090 Qwen 實測 閒／忙 | 3090 模型 閒／忙 | MI300X Qwen 實測 閒／忙 | MI300X 模型 閒／忙 |
|:--|:--|:--|:--|:--|
| **真實 CPU 層**（3090 6.91；MI300X 48.6） | **3／2** | 2／2 | **2／1** | 2／1 |
| 6.98（MI300X 的本地 SSD 參數） | 3／2 | 2／2 | NOT_MEASURED | 9／5 |
| 2.77（MI300X Qwen b16） | 5／3 | 5／3 | 16／8 | 16／10 |
| 0.764（MI300X Qwen b32） | 13／8 | 14／8 | 32／21 | 32／23 |
| 0.591（3090 Qwen b16） | 16／10 | 16／10 | NOT_MEASURED | 36／26 |
| 0.331（MI300X 的 NFS 參數） | 23／15 | 23／15 | NOT_MEASURED | 43／34 |
| 0.218（MI300X Qwen b48） | 29／20 | 29／20 | 48／38 | 48／39 |
| 0.179（3090 Qwen b32） | 32／23 | 32／23 | NOT_MEASURED | 50／42 |
| 0.055（3090 Qwen b48） | 47／39 | 48／40 | NOT_MEASURED | 58／54 |

- 每格 5 次；CPU 層 5 次的 b 全部相同（閒 3、忙 2）。KV 逐位元檢查：42 格各第一次，kv_bad 總和 0。
- **同 switch 鄰居卡的 PCIe 流量**（`x7_pcie_flag.py`，`nvidia-smi dmon`；GPU 1 的鄰居是 0、2、3）：這個 run 期間 GPU 0、2 有別人的流量（>500 MB/s 的取樣各 86、40 次／509），CPU 層 30 列裡閒 20 列、忙 28 列在還原區間內鄰居 ≥100 MB/s（最高 1.7 GB/s）。鄰居乾淨的列裡 b 一樣（忙 2；閒的 E0 Cake 5 列全部有鄰居流量 → 乾淨的 NOT_MEASURED）。〔判讀〕鄰居流量只會讓我們的 H2D 變慢、ℓ 變大、b 變大，所以「乾淨時 b ≤ 有鄰居時的 b＝3」，(a) 不成立、(i) 成立這兩個結論不受影響；時間欄（§4.6）則可能偏高，只當參考。H2D 常數本身（calib-f、C0）量的時候鄰居 ≤0.05 GB/s（GPU 0 49 MB/s、GPU 2／3 ≤2 MB/s）。
- **模型驗證**：3090 上 16 個限速格，模型（兩條線貪婪模擬，3090 的 f 與忙的倍數）和實測中位數差 ≤1 block 的有 16／16 → 過 §2.1 的 2/3 門檻，帶可以用模型。MI300X 第一階段 A0 的 32K 7 個頻寬，同一個模型的 b 有 6／7 差 ≤1（本地 SSD 參數那格實測 12.5、模型 14），加速比差 ≤0.08（`band_points.csv` 的 `MI300X_phase1` 對 `summary_a0_speedup.csv`）。
- Llama 的實測：**NOT_MEASURED（Llama 還原格子兩次被污染，§6）**。模型預測 3090 Llama 真實 CPU 層 閒 4、忙 3（`band_points.csv`；模型在 Qwen 上 16／16 格差 ≤1），MI300X 實測 3／2。
- 白話：3090 的 PCIe 慢 7 倍，CPU 層一個 block 的載入時間（ℓ）變長；但重算也慢 3.2–5.3 倍，f(0)÷ℓ 只從 22–42 掉到 15–32，**會合點還是在 block 2–4**。同一個 7 GiB/s 的層，在 MI300X 上會合點是 9（Qwen）／14（Llama，模型），在 3090 上是 2–4：慢的重算把會合點拉回前面。

### 4.5 Cake 有用的頻寬帶（TTFT 加速比 ≥1.3×；兩條線模型；`band_edges.csv`）

| | 3090 外包絡（main 段） | MI300X 外包絡（main 段） | 3090 的真實 CPU 層 |
|:--|:--|:--|:--|
| Qwen 閒 | 0.038–0.60（0.044–0.56） | 0.128–2.56（0.151–2.39） | 6.91 → 帶外 |
| Qwen 忙 | 0.021–0.31（0.023–0.27） | 0.062–1.22（0.073–1.07） | 帶外 |
| Llama 閒 | 0.073–1.27（0.085–1.11） | 0.248–4.89（0.293–4.57） | 6.99 → 帶外 |
| Llama 忙 | 0.038–0.73（0.041–0.63）＊ | 0.120–2.33（0.142–2.05） | 帶外 |

＊Llama 忙的新一輪時間 NOT_MEASURED，這一列用只看還原時間的加速比（`band_points.csv` 的 `ttft_fallback`）；Llama 閒的新一輪時間用 ladder run 的 c1 t_new256（block 64，1 次，0.143 s）。MI300X 的模型帶（Llama 閒 0.25–4.9）和論文實測的帶（0.33–3.73 GiB/s 內 1.38–1.93×）一致。峰值：3090 約 2.1–2.2×（Qwen 在 0.15 GiB/s、Llama 在 0.30 GiB/s），MI300X 約 2.2×（0.57、1.09 GiB/s）。**整條帶往低頻寬移了約 3–4 倍（下緣 2.9–3.4 倍、上緣 3.8–4.3 倍），峰值高度幾乎不變。**

### 4.6 只報告：CPU 層上的低精度、Cake 對只載入

Qwen 32K、真實 CPU 層、還原時間中位數（s；5 次；`restore_3090_summary.csv`）：

| | BF16 | FP8（`fp8_head`） | INT4（`int4_kivi`） | MI300X 同格（BF16→FP8→INT4） |
|:--|--:|--:|--:|:--|
| Cake，閒 | 0.406 | 0.283（−30%） | 0.297（−27%） | 0.052→0.055→0.061 |
| Cake，忙 | 0.509 | 0.286（−44%） | 0.568（+12%） | 0.066→0.074→0.146 |
| 只載入，閒 | 0.369 | 0.204（−45%） | 0.180（−51%） | 0.038→0.027→0.033 |
| 只載入，忙 | 0.414 | 0.294（−29%） | 0.289（−30%） | 0.054→0.049→0.081 |

- **3090 上低精度讓真實 CPU 層變快**（除了 INT4＋Cake＋忙），和 MI300X 相反（main.tex §5.1：Qwen CPU 層 FP8 −5.1%、INT4 −17.9%（閒）、−121%（忙），負＝變慢）。原因就是 §4.3：H2D 慢 7 倍，反量化只慢 2–4 倍。這還是主線 PyTorch 反量化；換融合 kernel 的 CPU 層格子：NOT_MEASURED（沒排到乾淨的卡）。Llama：NOT_MEASURED。
- **Cake 在 3090 的 CPU 層比只載入慢**：BF16 閒 0.406 vs 0.369 s（+10%）、忙 0.509 vs 0.414（+23%）；FP8 閒 +39%。MI300X 也一樣（BF16 閒 0.052 vs 0.038，+37%）。重算線一定先拿 block 0（3090 上 126 ms），而只載入全部只要 64×3.96 ms≈0.25 s。

### 4.7 判定（照 §2.2 字面；`verdict.json`）

| 條件 | 結果 |
|:--|:--|
| (a) 3090 真實 CPU 層會合點中位數 ≥8 | **否**：Qwen 閒 3、忙 2（Llama 實測 NOT_MEASURED；模型 4／3） |
| (b) 3090 帶涵蓋 3090 的真實 CPU 層頻寬 | **否**：CPU 層 6.9–7.0 GiB/s，帶的上緣最高 1.27 GiB/s（外包絡、main 段都一樣） |
| (i) CPU 層會合點每格 ≤3 | **是**（Qwen 3／2） |
| (ii) 3090 帶包含在 MI300X 帶裡 | **否**：3090 的帶下緣（0.021–0.073）比 MI300X（0.062–0.248）低，四格都一樣；外包絡、main 段結論相同 |

→ **可能**（不是有看頭：(a)(b) 都不成立；不是死路：(ii) 不成立）。讓它「不是死路」的唯一原因是帶整條往低頻寬移，多出 0.02–0.25 GiB/s 這一段「MI300X 上只算比較快、3090 上 Cake 有用」的區域；**CPU 層本身完全沒有新的事情可以決定。**

## 5. 前作

- 只用了本專案自己的量測與文件：main.tex §2.2、§3.1、§3.2、§4.1、§5.1；`docs/phase1_20261008/05_experiments.md`、`07_report.md`（C0–C7 的量法）；`docs/migration_20261013/README.md` §2.3。這個方向**沒有讀新的論文**。
- Cake（main.tex 引用的 cake2025）原文說會合點隨 GPU 負載與頻寬移動；這裡只是在另一台硬體上量這件事，沒有新的前作主張要核對。

## 6. 偏離與問題（照實列）

1. **PCIe 是 Gen3，不是 Gen4**（migration README §2.3 寫 4.0）；而且 pinned H2D 只有 6–7 GiB/s，約 Gen3 理論值的一半，原因 NOT_MEASURED（§4.1）。
2. **共用機的污染**：
   - `20261017-213420-p3x7-grid-qwen-L1`：21:35:56 起，別人的 `reserve_gpus.py`（pid 870864）在 GPU 1 放了 254 MiB 的 context（同時佔了 GPU 0／4／5 各 16.7 GiB），`gpu_guard` 標 CONTAMINATED_DURING_RUN → 數字作廢。
   - `20261017-215041-p3x7-grid-llama-L0`：開跑前 GPU 3 乾淨，21:51:05 別人（pid 1050579）在 GPU 3 開了 15.7 GiB，我們的 process 載權重時 OOM（「Tried to allocate 112.00 MiB … 98.00 MiB is free … Process 1050579 has 15.70 GiB memory in use」）→ 沒有量到。
   - `20261017-221130-p3x7-grid-llama-L0`：等卡腳本（`logs/p3x7/chain3.sh`）在 GPU 6 乾淨時開跑（22:11:30 `gpu_guard --check` CLEAN），6 秒後別人的 lm-eval（pid 1225514，1648 MiB）上了 GPU 6、一直到 22:28:06 都在（311 筆取樣，幾乎整段）→ 作廢。
   - 21:46 以後 7 張卡一直有別人：ron 的 `reserve_gpus.py` 佔 GPU 0／1／4／5，pojen 的 lm-eval 在每一張卡開 1.6 GiB 的行程，GPU 2、3、6 另有人。照規則每個步驟最多重跑 1 次，Llama 兩次都被污染，等卡腳本就停了。所以 **Llama 32K 還原格子、Qwen 第二個 launch、Llama 64K 的 CPU 層、融合 kernel 的 CPU 層格子、第三次 C0 都是 NOT_MEASURED**。
3. **`code/run3090.sh` 在我的 run 跑到一半被環境 agent 改**（bash 邊跑邊讀）→ `20261017-175844-p3x7-busy-qwen` 跑完後在第 111 行語法錯誤、沒有 exit_code（那次是 `--out /dev/null` 的冒煙，留 INTERRUPTED 檔、不用）。之後改用快照，最後換回已提交、內容相同的 `code/run3090.sh`。
4. **`m7_busy.py` 的 duty 在 3090 上不一定準**：一次 8192² BF16 matmul 15.8 ms（MI300X 上 35 筆裡 33 筆在 1.646–1.671 ms（另兩筆 4.3、5.0 ms）：`results/m10_sweep_mi300x/dqsweep/C/*/busy_*.json` 的 READY 行，例如 run `20261017-033651-m10-dqs-C-qwen-int4-S4`），`--duty 0.4` 實際只做到 54% 使用率；0.7 剛好約 70%。沒改 `m7_busy.py`；`x7_busy.py` 另外量了實際使用率。
5. **MI300X 忙的 f(i)** 在模型裡是「閒的 f × 2.067」（第一階段 C5 只量 block 32）；3090 忙的 f 用 block 0／32／63 三點內插。兩台的忙用同一個背景設定，但 MI300X 的變慢只有一個點。
6. **帶是模型算的**，不是逐頻寬實測；模型在 3090 上 16／16 格、在 MI300X A0 上 6／7 格會合點差 ≤1 block（另一格差 1.5）。外包絡規則是看過除錯輸出才寫死的（§2 補充 2），兩種規則結論相同。
7. **重複次數少**：每格 5 次、只有 1 個 launch（第二個 launch 被污染）；MI300X 是 3 launch×10 次。CPU 層的 b 在 5 次裡完全不變，但 launch 之間的變異 NOT_MEASURED。
8. `host_contention` 每個 run 都是 HEAVY（別人在其他卡上）。H2D 常數的 run（C0、calib-f）期間同 switch 鄰居 ≤0.05 GB/s；Qwen 還原格子期間鄰居有到 1.7 GB/s 的流量（§4.4），時間欄可能偏高。**主機 CPU／記憶體頻寬被別人用掉多少看不到**（loadavg 約 45）。§2.2 的「結論只用鄰居可忽略的量測」我用的門檻是 100 MB/s（dmon 每秒平均），這是看過資料才定的，照實寫。
9. 需要主 session 決定、我沒改的既有程式：`m7_calib.py` 的 c2 把磁碟名寫死成 `sda`（這台 `/ssd8` 是 `sdg`，env_3090.sh 有 `M7_DISK` 但 c2 沒讀）；`test_m7_correctness.py` 預設 8K 在 24 GiB 上 OOM（環境 agent 已另外處理）。這次沒有量真實磁碟（C2），NOT_MEASURED。
10. `torch.compile` 的 FP8 在 sm_86 不能用（§4.3）；主線與融合 kernel 都不受影響。

## 7. 如果要繼續，下一步

1. 機器安靜時補完（`bash /ssd8/hungwei/paper-hkv/logs/p3x7/chain3.sh`，會自己等乾淨的卡）：Llama 32K 還原格子（模型說 CPU 層 4／3，<8；就算實測是 4，判定也還是「可能」，因為 (ii) 已經不成立）、Qwen 第二個 launch、Llama 64K CPU 層、融合 kernel 的 CPU 層格子、第三次 C0（不同時段）。
2. 〔判讀〕最值得追的不是會合點，而是 **§4.6：3090 上 CPU 層的低精度變快**（Qwen FP8 只載入 −45%、INT4 −51%），和 MI300X（main.tex §5.1「low precision slows the CPU tier」）相反。這直接影響 X5：在慢 PCIe 的平台，「CPU 裡放 INT4」可能同時省容量又變快，取捨就消失了；要先補 Llama 與融合 kernel，再用 `constants_3090.json` 加跑 X5。
3. 給 X1／X4／X5 加跑：用 `constants_3090.json` 的 f(i)、ℓ_CPU、忙 1.91×、b\* 頻寬、反量化時間；標〔3090 常數〕，和 MI300X 常數分開列。注意 3090 的 CPU 層（7 GiB/s）和 MI300X 的「本地 SSD」參數一樣快，「CPU 層 vs 慢層」的差距縮小很多（MI300X 是 35.4 vs 6.98）。
4. 查 H2D 只有一半的原因（MaxReadReq、PLX 上行、主機記憶體爭用）需要 root；沒有就只能當這台機器的常數。
