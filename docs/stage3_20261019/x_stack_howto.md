# X1／X2：KV 壓縮疊加的品質實驗怎麼跑（第三階段；10/20 寫）

> **狀態（2026-10-20 05:3xZ）**：程式**還沒有在 GPU 上跑過任何一行**。寫的時候第二階段正在兩張 MI300X 上計時，不能開 GPU 行程、不能 import torch。
> 做過的：`python -m py_compile`；`python code/m16_xstack.py --self-test`（只用 CPU、numpy、不 import torch）**111 項 ALL PASS**；
> 靜態核對：用到的 `m10_*`／`m16_s2_equalbytes` 屬性全部存在（AST）；job 腳本 `bash -n` 過、helper 函式用假的 `m7run` 跑過一次（指令展開正確）。
> **停止條件、挑選規則、ε 都在看到任何 X1／X2 數字之前寫死在這裡（§5）與程式常數（`m16_xstack.py` 的 `STOP_TEXT`）**，之後不改；要改只能另開版本並註明。
> 本頁的時間全部是〔估〕，依據寫在 §8。沒量到的一律 NOT_MEASURED。self-review，不是 cross-model review（CLAUDE.md §5）。

設計來源（不改）：`lit/L5d_stacking.md`（三階段設計、停止條件、位元組照實算、干擾機制）、`lit/L5a_lowrank.md`（E-L1／E-L2）、
`lit/L5b_merge_crosslayer.md`（TM-g、MiniCache、xKV、KVSharer）、`lit/L6_mainstream_practice.md` §5.2、§7（常見部署對照、MLPerf 門檻；10/20 協調者補充）。

---

## 0. 檔案

| 檔案 | 內容 |
|:--|:--|
| `code/m16_xstack.py` | 子命令 `calib`、`screen`（X1）、`report`（X1，CPU）、`gen`（X2）、`gen-report`（X2，CPU）、`gpu-selftest`；`--self-test`（CPU） |
| `code/m16_xstack_combos.json` | 組合清單（每個模型 161 個；每組寫了 role 和理由）＋冒煙用的 17 個 |
| `code/m16_queue/jobs/x_common.sh` | job 共用的函式與旋鈕（shard 數、題數）。**不要放進佇列目錄**（runner 會把佇列裡的 `*.sh` 都當 job） |
| `code/m16_queue/jobs/02_smoke_x.sh` | 冒煙（GPU，兩張卡） |
| `code/m16_queue/jobs/36a/36b/36c_x1_screen.sh` | X1（GPU，兩張卡；三段，逐題續跑） |
| `code/m16_queue/jobs/36z_x1_report.sh` | X1 報告＋x2_selection.json（cpu 佇列） |
| `code/m16_queue/jobs/37a/37b/37c/37d_x2_gen.sh` | X2 32K（GPU；四段，層 core → x3 → ext，逐題續跑） |
| `code/m16_queue/jobs/37e/37f_x2_gen.sh` | X2 128K（GPU；44c 的題組） |
| `code/m16_queue/jobs/37z_x2_report.sh` | X2 報告＋**給 X3 的 `x2_feasible.csv`**（cpu 佇列；X3 的 `39z_x3_map.sh` 等的就是這個名字） |
| `code/m16_queue/jobs/37zz_x2_report_final.sh` | 加上 ext／128K 後重出一次（cpu；可不排） |

輸出位置：
- 逐列原始輸出、校準基底：`/mlsteam/data/tiara/stage3/xstack/`（`x1/x1_<m>_s<i>.csv.gz`、`x1/s2part_<m>_s<i>.csv.gz`、`x2/x2_<m>_<ctx>_<tier>_s<i>.csv.gz`、`calib_<m>.pt`；不進 git）。每列有 `run_id`、`ts`。
- 進 git 的摘要：`results/m16_stage3_mi300x/x_stack/`（`calib_<m>.json`、`x1_report.json`、`tables_x1.md`、`configs_<m>.csv`、`pareto_<m>.png`、`x2_selection.json`、`x2_report.json`、`tables_x2.md`、`x2_feasible.csv`、`x2_unmapped.csv`、`s2_subset/`）。
- 冒煙：`/mlsteam/data/tiara/stage3/xstack/smoke/`（不進 results）。

既有檔案**一個都沒改**（`m7_model.py`、`m10_*.py`、`m16_s2_equalbytes.py`、`runner.sh`、`lib.sh` 都只用 import／source）。

---

## 1. 和 L5d 三階段的對應（為什麼這樣改）

| L5d | 這裡 | 改了什麼、為什麼 |
|:--|:--|:--|
| 第一階段：CPU 在 KV dump 上掃 300–500 組、用代理挑前緣 | **X1**：GPU 上直接量 teacher forcing（TF），每題 prefill 一次，161 個組合＋S2 的原本各組 | 協調者指定 X1 用 TF 大量篩選；直接量 TF 比「先校準代理再挑」少一層假設。代理（K／V 重建誤差、attention 輸出誤差）照樣每列記，拿來回答 L5d §5.2 (b)「代理跨家族能不能預測」，**不用來挑前緣** |
| 第二階段：前緣 10–20 組量生成答對率 | **X2**：照 X1 挑出的組合量生成答對率（core／x3／ext 三層＋128K 切片） | 同 L5d §6.3；題目用**新題**（seed 62000 起）避開 X1 的挑選偏差（09 的「海選／複賽分開題目」做法） |
| 第三階段：時間 | X3（另一個 agent 的 `m16_xtime.py`） | X2 寫 `x2_feasible.csv` 給它（§7） |

---

## 2. 題目、路徑、度量

- **X1 題目**＝DS1／S2 同一套：seed 60000 起、T2–T5 各 24 題（`--n`，job 用 `X1_N`）、32K、`CANDIDATE_PARAMS.json`。處理順序照任務輪流排（T2 第 0 題、T3 第 0 題、…），時間預算用完時做完的題在任務間是平衡的；題目本身和 `m10_quality.plan_items` 相同。
- **每題**：`m10_sweep.Item` prefill 一次 → **S2 子集**（直接呼叫 `m16_s2_equalbytes.run_item`，參數照 S2 的 `cmd_run`：寫入時方法 SNAP、KEYDIFF（有 `ds2_plan.json` 就用它的 B 粒度 top2）＋QRY、RND0；格式 FP8（fmt_choice）、INT8、INT4 KIVI、INT2 KIVI、K4V2；`region`＋`full` 兩種範圍；mask／sdpa 兩種路徑；DS1 逐列檔在就做 DS1 重現檢查）→ E0（同時用 forward hook 抓 q_proj）→ 組合清單。
  - S2 子集逐列寫成 **S2 自己的格式**（`s2part_*.csv.gz`），S2 的 `report` 可以直接吃（36z 會跑一次，輸出到 `x_stack/s2_subset/`）。所以 **S2＝X1 的子集**；另外單獨排 S2 是重複的（§13）。
  - 為了省一次 32K prefill，`run_item` 開頭的 `SW.Item(ans, it)` 在這個行程裡暫時換成「回傳已經 prefill 好的這一題」（`reuse_item`，不改檔；程式核對 run_item 剛好建一次 Item，不然停）。
- **組合的範圍**：預設 `full`＝block 1–63（block 0 一律 BF16；和文獻可比，L5d §6.0 的「全段」）；`--ranges full region` 可另做 S2 的載入區 32–63（時間加倍，預設不做）。
- **attention 路徑**：所有組合走遮罩路徑（`key_keep`；沒丟 block 時是全 True），drop 才能和其他配對；E0 走 sdpa。`none`（MASK0）和 E0 的差＝換路徑的雜訊。
- **TF 度量**（和 S2／DS1 同定義，`m16_s2_equalbytes.tf`＝`m10_sweep.Item.metrics`）：`kl`（對 E0）、`gold_lp`（標準答案 log 機率）、`top1`、`first_ok`。
- **代理**（每列）：
  - `relerr_k`／`relerr_v`＝區內 ‖X̂−X‖_F／‖X‖_F（丟掉的 token 算全部遺失＝X̂ 為 0）；`cenerr_k`＝分母減掉每（層, head, channel）的 token 平均（Qwen 的 K 有大的固定偏移）。
  - `attn_err`＝第 1 輪問題的最後 ≤64 個 token 的 query（E0 的 TF 時 hook 抓 q_proj 輸出、套模型自己的 RoPE）對 context 全部 key 的 attention 輸出相對誤差，每層算、對層平均（`attn_err_max`＝最大層）；`attn_err_wo`＝再乘 W_O（AATC 的 P2w）。每層獨立算（看不到跨層累積）；只看 context 的 key。
- **X2 題目**：32K＝新題 seed 62000 起、T2–T5 各 24 題（`X2_N`）；128K＝**第二階段 44c 的題組**：`m10_sweep screen --n-blocks 252 --tasks T2 T3 T4 T5 --n 12`，seed 預設 40000（run `20261019-033004-m10-lc-screen-llama-252-s0` 的 `cmd.sh`），出題用 `m10_tasks.make_item(…, n_blocks=252)`、`m10_sweep.N_BLOCKS＝252`、`Answerer(extra=(252−64)×512＋4096)`，和 44c 同一個作法。E0 的生成分數逐題對 44c 的逐列檔比（`.checks.jsonl` 的 `vs_44c`；兩邊都是 sdpa，應該逐值相同）。
- **X2 度量**：`m10_sweep.Item.gen()`（第 1 輪、greedy、T5 生成 120、其他 64；`m10_tasks.score`）＝第二階段同一個計分。
- **校準**（`calib`）：seed 61000 起、T2–T5 各 2 題（8 題、約 25 萬 token），block 1–63 的 token；K 逆 RoPE 後，每層 FP64 累加平均與共變異（每層跨 head 聯合 [H·D, H·D]；每 head 的就是對角區塊），GPU 上 eigh。程式核對校準題和評測題不重疊（重疊就停）。第一題時做 **RoPE 的資料自身核對**（規則 6）：hook 抓第 0 層 k_proj 的輸出（RoPE 前的 K），和我們對 KV 緩衝裡 RoPE 後 K 的逆旋轉比，相對誤差要 ≤0.02，而且比「不逆轉」「位置錯 1」「sin 正負號反」都小一半以上，不然停。

---

## 3. 壓縮階段與位元組帳（照實算）

**組合語法**：階段用 `+` 串起來、照套用順序。合法順序：`drop? merge? (lr|xl)? q?`，或 `xl+drop(+q)`（xKV 基底在全部 token 上估的順序對照）。程式擋掉的（`validate`，self-test 有 24 個反例）：drop 在 merge／lr 之後、merge 和 lr／xl 一起（合併後的 key 不在任何單一位置上，pre-RoPE 低秩沒有定義）、lr 和 xl 一起、q 不在最後、mc／share 和別的一起、係數用依賴 block 結構或 head 維度的格式（mxfp4、nvfp4、fp8_*、整塊的 fp8k0_*）、lr 秩不是 4 的倍數、秩超過 128、窗口不整除層數、丟 0 個或丟光。

| 階段 | 做什麼 | 存下來的位元組（每個請求） | 跨請求共用（另列 `bytes_shared`） |
|:--|:--|:--|:--|
| `drop:<方法>:<比例>` | 512-token block 粒度；順序＝`m10_drop.unit_order(cx,"B",方法)`（全段算、只留區內）；丟 round(比例×區內 block 數) 個，`key_keep` 遮掉 | 留下的 token；遮罩每 block 1 bit（⌈區內 block 數/8⌉ byte）；沒有逐 token 索引 | — |
| `merge:<g>` | TM-g：block 內每 g 個相鄰 token，key＝組平均（post-RoPE）、value＝組平均 | 每組 1 個 key＋1 個 value（＝1/g）；g 固定，計數不存 | — |
| `lr:k<r>[:v<r>][:joint]` | 校準 PCA；K 逆 RoPE（模型自己的 BF16 cos/sin，2×2 旋轉精確求反）→ 減校準平均 → 投到前 r 個主成分 → 重建 → 加平均 → 模型自己的 `apply_rope`；V 一樣不用 RoPE；`joint`＝每層跨 head 聯合、秩 r×H；r＝0 那一半原樣 | 係數 [層, head, token, r]（BF16，或 q 量化後） | 基底（前 r 欄）＋平均：每 head 版 L·H·(D·r＋D)·2 B；聯合版 L·(H·D·rH＋H·D)·2 B |
| `xl:k<r>[:v<r>][:w<W>]` | xKV：每題、每 W（4）個相鄰層聯合 SVD（K 逆 RoPE、不去平均＝照 xKV）；聯合秩＝r×W×H；`torch.svd_lowrank`（seed 固定、niter 4、多取 16 個） | token 係數 [token, 聯合秩]＋**每個窗口的基底 B（聯合秩×W·H·D，BF16，每題一份）** | — |
| `mc` | MiniCache 式（對照）：深半段相鄰兩層同一 token、同一 head 的向量 SLERP（t＝0.6），各自存範數；角距離最大的 5% token 原樣保留 | 方向（BF16）＋2 個範數＋保留 token 的兩份原值＋索引（⌈log₂T⌉ bit）；淺半段原樣 | — |
| `share` | KVSharer 式（對照）：深半段奇數層直接用前一層 | 0.75 | — |
| `q:<格式>` | 只有原始 KV：每個存下來的 block（merge 後是 512/g 個 token）照 `m10_codec_x.encode／decode`（第二階段同一段，含 fp8k0_* 等整塊格式）。有 lr／xl：係數用 `m10_codec_x.enc_half／dec_half`，K 角色（**每個主成分各自 scale、沿 token 每 32 個一組**＝L5d §2.1 的「逐成分」；`int8_token` 是逐 token 跨成分），沒降秩的那一半照原始 KV 的半邊格式逐 block 編碼 | Enc.nbytes（含 scale／zero／全域 scale） | `vllm_fp8` 的每層 K／V scale（FP32） |
| `q:vllm_fp8` | **新增**（m10 沒有）：模擬 vLLM 0.28 在 ROCm 上未校準的 fp8 KV＝per-tensor、scale＝1.0、`float8_e4m3fnuz`（最大 240）；超出先截在 ±240 再轉型 | 每值 1 byte | 每層 2 個 FP32 scale |
| 無損（zstd） | **不在 GPU 迴圈裡跑**：report 用 S1 的實測壓縮比（`results/m16_stage3_mi300x/s1_lossless/summary.json`，T4 dump、`zstd:3`；BF16 用 planes 變體）當乘數另列 `ratio_lossless`；只有 S1 量得到的儲存格式（bf16、fp8_token、fp8_head、int8_token、int4_kivi，可以先丟 block）才乘，其他 NOT_MEASURED；summary 還沒有 → 整欄 NOT_MEASURED | — | — |

- **照實算**：GPU 路徑每個組合都 `assert 實際位元組（各元件 tensor 的 numel×element_size）＝predict_bytes(…)`，不等就停（規則 2）。`predict_bytes` 的公式在 CPU self-test 用兩條獨立路徑核對：(1) numpy 參考實作實際打包後的陣列大小；(2) **第二階段實測的 `results/m10_sweep_mi300x/parts/parts_{llama,qwen}.json`（真實 block 的 Enc.nbytes，13 種格式，兩個模型）逐一相等**。
- **KIVI 殘差窗口**：m10 的 KIVI 沒有全精度殘差窗口（context 全部量化），這裡也沒有（0）。**sink**：block 0 一律 BF16，在區外，算進 `bytes_total`。
- **位元組比**：`ratio_region`＝區內位元組 ÷ 區內 BF16；預算比較一律用它（L5d 的「全段」）。`ratio_total` 另含 block 0。
- **「同一個 KIVI 定義」的證明**：CPU self-test 把 `m10_codec.py` 的 int4_kivi（encode 139–157 行、decode 177–182 行）逐行照抄成 numpy，和「係數用的 KIVI 量化器」（K／V 角色）比，解碼逐值相同、位元組相同；GPU 冒煙再在真實 KV 上驗 `enc_half(kivi4,'k'／'v')`＝`m10_codec_x.encode(int4_kivi)` 的兩半（逐位元）。

---

## 4. 組合清單（`code/m16_xstack_combos.json`；每個模型 161 個）

位元組比由 `predict_bytes` 算（〔算術〕，不是量測；Llama／Qwen 只有 xKV 的基底項不同）。每個預算（≤1.05×）內的組合數：0.3125：83、0.25：81、0.125：50／53、0.0625：15／17。

| 組 | role | 內容 | 為什麼在 |
|:--|:--|:--|:--|
| C0 | control | `none` | MASK0：換路徑的雜訊 |
| Q1／Q1L／Q1Q | single | 第二階段的 13 種格式（Llama FP8＝fp8_token、fp8k0_token；Qwen＝fp8_head、fp8k0_head） | 量化家族的超參數掃描；INT4 KIVI（0.3125）是錨點 |
| Q2 | single | K2V4、K8V2、只壓 K、只壓 V | K／V 哪一半能多犧牲（KVTuner） |
| DEP-L／DEP-Q／DEP-QB | deploy／deploy_broken | Llama `q:vllm_fp8`；Qwen `q:int8_token`（＝vLLM int8_per_token_head）；Qwen `q:vllm_fp8`（已知會壞） | §6 |
| D1 | single | KEYDIFF、KNORM、SNAP × 丟 25／50／75／87.5／93.75% | 丟的家族（不用 query 的兩個＝L5d 說 DS1 最好的；SNAP＝S2 預設） |
| D2／D3 | control／ref | RND0、QRY × 同上 | 隨機丟（X2 對照）；知道問題的參考上限（不可部署） |
| L1–L4 | single | 每 head PCA r＝8–64（K、V 同秩；含 L5a 的 P24、P40）、只降 K、K／V 不同秩、每層跨 head 聯合 | 低秩家族；機制（pre-RoPE K 是不是低秩、V 譜平） |
| X1／X3M | single | xKV K:V≈1:1.5（每 head 等效 3/4、5/9、11/17、22/34、28/42、只 K 11）；K、V 同秩 16/16、32/32 | 跨層家族；X3M＝和 X3 的 `xkv_g4_f8／f4` 同設定 |
| M1 | single | TM-2／4／8／16 | 合併家族 |
| XS | single | `mc`、`share` | L5b 說在我們的模型上很可能壞，照樣量一組 |
| PDQ1–6 | combo（PDQ4 control、PDQ5 ref） | 丟（KEYDIFF、KNORM、SNAP、RND0、QRY）× INT4／K4V2／INT8／INT2 | 文獻最多加乘證據的一對；PDQ4＝疊量化後「挑哪些 block」還有沒有用（Q-Hitter 的干擾） |
| PLQ1–4 | combo | PCA 係數 × INT4（逐成分）／INT8（逐 token）／INT2；K24V4（L5a）與變體 | 低秩＋量化；逐成分 vs 逐 token（Palu 的離群問題） |
| PXQ | combo | xKV＋INT4／INT8 | xKV 附錄 D.5 的 RTN-4 疊加 |
| PMQ | combo | TM-2／4／8＋INT4／K4V2／INT8 | L5b §5.2 |
| PDL／PDM／PDX | combo | 丟＋低秩／合併／xKV | 干擾：留下的正是最難表示的 token（L5d §2.3） |
| T1–T3（TC control） | combo | 三件：丟＋低秩＋量化（含任務說明的 `drop:KEYDIFF:0.5+lr:k24+q:int4_kivi`）、丟＋合併＋量化、丟＋xKV＋量化 | 只從最有希望的兩兩往上加（L5d §6.2） |
| N | naive | 各家族用寫清單時已知的安全設定（INT4、丟 25%、P40、TM-2）直接疊 | 測 KVTC 說的「天真疊加不複合」 |
| O1 | order | `xl+drop` vs `drop+xl` | xKV 基底在全部 token vs 留下的 token 上估（L5d：NOT_FOUND） |

**沒做的順序對照**（寫在 JSON 的 `not_implemented`）：丟的分數用量化後的 K 打分（Q-Hitter，L5d (iv)）——全部用原始 BF16 打分；量化分組對齊 vs 逐 head 丟後重分組（(iii)）——我們只做 block 粒度，KIVI 分組永遠完整，對照不存在；聯合分配（RDKV／KVTC）——沒做。

---

## 5. 停止條件與挑選規則（**事先寫死**；程式常數 `STOP_TEXT`）

ε＝2 個百分點（第二階段同一個）。預算＝區內位元組比 0.3125（錨點）、0.25、0.125、0.0625；「在預算內」＝位元組比 ≤1.05×預算（L5d §6.2 的 ±5%）。

**家族與池**：
- 單一方法（single）＝只有一個有損階段、可部署（沒有 QRY／ORC）、role 含 single。RND0 丟、`deploy_broken` 不算；`deploy` 只有和 single 重疊時（Qwen 的 int8_token）才算。
- 組合（combo 池）＝兩個以上有損階段、可部署、role 是 combo／naive／order（RND0 的組合是對照，不進池）。

### 5.1 X1（TF；L5d 第一階段的停止條件，改成 TF 版）

- **TF 品質**：平均 Δgold_lp（對 E0，逐題配對，bootstrap 10,000 次）；「ε 內（TF）」＝平均 Δfirst_ok ≥ −0.02（點估計；X1 是篩選，判定只看 X2）。
- **S1-a（代理不可靠）**：所有代理對 KL 與 Δgold_lp 的 Spearman |ρ|（以配置為單位、bootstrap CI）都 < 0.5（兩個模型）。**我們的 X1 直接量 TF，挑前緣不用代理，所以 S1-a 只當研究問題 (b) 的結果、不改流程**（和 L5d 不同的地方，理由：L5d 的代理是用來省 GPU，我們已經在 GPU 上量了 TF）。另報家族內 ρ、崩潰 AUROC（崩＝Δfirst_ok ≤ −5 個百分點）。
- **S1-b（TF 層級沒有加乘 → 停，不跑 X2）**：每個預算（0.25、0.125、0.0625）、兩個模型，「最好的組合的損失（−平均 Δgold_lp）≥ 0.95 × 最好的單一方法的損失」。預算內沒有組合 → 算成立；預算內沒有單一方法而有組合 → 不成立。**三個預算、兩個模型全部成立 → `go＝false`**，寫成「在 TF 層級，疊加沒有加乘」。
- **S1-c（無損沒用）**：S1 實測疊在 INT4 KIVI 上的無損增益 < 1.05× → 第二、三階段拿掉無損這一軸（我們本來就只在 report 另列；沒有 S1 → NOT_MEASURED）。
- **完整性**：每個模型至少 48 題做完才挑 X2；不夠 → `go＝false`、NEEDS_HUMAN。
- **另報（不改判定）**：TF 版 M1 的 r（B*＝ε 內最省的位元組）；MLPerf 式（first_ok 率 ≥ E0 的 99%／99.9%）的 B*。

### 5.2 X2 挑選規則（`select_x2`；X1 跑完自動算）

- **錨點**：`none`（MASK0）、`q:int4_kivi`；**部署對照**：Llama `q:vllm_fp8`、Qwen `q:int8_token`（core＋128K）、Qwen `q:vllm_fp8`（core，只報）。
- **每個預算（0.25、0.125、0.0625）**：
  - 最好的單一方法（各家族掃完自己的超參數後取最好；core＋128K）；
  - 前緣：預算內的組合依平均 Δgold_lp 排，同一種家族組合最多 2 個，取 4 個——第 1 名＝**主要組合（M2 用、事先選定，不事後挑）**（core＋128K）、第 2 名（core）、第 3–4 名（ext）；
  - 天真疊加：N 組裡這個預算最好的（core）；
  - 隨機丟：同預算內保留比最接近的 `drop:RND0`（core）；
  - 各家族最好的單一方法（quant、drop、lowrank、xlayer、merge；ext）。
- **0.3125**：最好的單一方法＋前緣前 2 名（ext）。
- **naive_x1**（core）：drop、lowrank（沒有就 merge）、quant 三個家族各取「ε 內（TF）最省位元組的單一設定」，照合法順序直接疊（疊不上就去掉量化並記原因）。
- **x3 層**：組合清單裡對得到 X3 解碼段（§7）而不在 core 的組合。
- 32K 的層互斥（core > x3 > ext），同一個組合只量一次；128K 另算。

### 5.3 X2（生成答對率；L5d 第二階段的停止條件，原樣）

- **ε 內**＝生成答對率對 E0 的逐題配對差 95% CI 下界 ≥ −2 個百分點。
- **M1（同品質省多少位元組；主判定）**：B*＝ε 內最省的位元組比（單一方法池、組合池各一個）；r＝B*(組合)／B*(單一)。
  - 兩個模型都 r > 0.95（沒有組合過 ε 也算）→ **NO-GO**（「在本設定下疊加沒有加乘」，CLAUDE.md 規則 4：有價值的負面結果）；
  - 沒有模型 < 0.85、但有 0.85–0.95 → **停下來問人**；
  - 任一模型 r < 0.85 → **GO**（進第三階段）。
- **M2（同位元組答對率）**：每個預算，主要組合 − 最好的單一方法（配對差）。E0 − 最好的單一方法 < 2 個百分點的預算記「沒有空間」；有空間的預算全部「差 < 2 個百分點或 CI 含 0」→ 支持 NO-GO。Holm（3 預算 × 2 模型，單邊 α＝0.025）。
- **完整性**：組合在 M1／M2 贏、但天真疊加本身一樣好（naive − 主要組合的 CI 含 0 或 ≥ −ε）→ 結論改寫成「疊加有用，但不需要聯合分配」（和 KVTC 的說法相反，要特別報）。
- **題數**：32K 每個模型至少 48 題，不夠 → `INCOMPLETE`（成分照列、只供參考）。
- **128K**：探索，不進判定。

### 5.4 MLPerf 式門檻（L6 §5.2；**只報，不改判定**）

每個配置另列「答對率 ≥ E0 的 99%」「≥ 99.9%」（點估計；X1 用 first_ok 率、X2 用生成答對率），和 ε 並列；另報這兩個門檻下的 B* 與 r。停止條件仍只用上面寫死的 ε。〔算術〕L6：在 Llama 32K 的 68.2% 基線上，99% 只容許掉 0.68 個百分點、99.9% 是 0.068，我們的 ε 約等於 97.1%——所以 MLPerf 門檻在我們的題數下幾乎一定比 ε 嚴，判讀時要註明題數。

---

## 6. 常見部署對照（L6 §7.1；10/20 協調者補充）

- **Llama**：`q:vllm_fp8`＝模擬 vLLM 0.28 在 ROCm 上**未校準**的 fp8 KV：per-tensor、scale＝1.0、`float8_e4m3fnuz`（最大 240；L6：fnuz 平台、checkpoint 沒有 scale → 一律 1.0）。m10_codec_x 沒有這個格式，在 `m16_xstack.py` 自己實作（`t_vllm_fp8`）。**超出 ±240 的值：假設截在 ±240**（ROCm kernel 是截還是溢位：NOT_VERIFIED，L6 §2.1）——torch 的 fp8 轉型超出範圍會變 NaN（m10_codec 檔頭），所以先截再轉。冒煙驗：torch 的轉型＝numpy 參考的 fnuz 格點（最近值、平手取偶數、截在 240），逐值；並記這個 block 有多少比例被截、max|K|。
- **Qwen**：`q:int8_token`＝vLLM `int8_per_token_head` 最接近的格式：對稱 INT8、scale 每（層, K/V, head, token）＝amax÷127（`m10_codec.py` 檔頭寫的就是這個分組）；我們的 scale 存 FP32（vLLM 的 scale dtype NOT_VERIFIED）。
- **Qwen 也照樣跑一列 `q:vllm_fp8`**，role＝`deploy_broken`（L6：max|K|≈422 > 240，九月撈針 2/20），只報、不當主對照、不進任何「最好的單一方法」或 M1 的單一方法池。
- 這三列在 X1、X2 core 都有；Llama 的 vllm_fp8 與 Qwen 的 int8_token 另進 128K 切片。

---

## 7. 給 X3 的介面（`x2_feasible.csv`、`x2_unmapped.csv`）

`37z_x2_report.sh`（**實際檔名；X3 的 `39z_x3_map.sh` 寫的 `AFTER: 37z_x2_report` 對得上**）跑 `gen-report`，寫：
- `results/m16_stage3_mi300x/x_stack/x2_feasible.csv`：`model, x3_segment, feasible, run_id, ts, combo, ratio, acc, acc_e0, acc_delta, acc_delta_lo, acc_delta_hi, eps, mlperf99, mlperf999, n_items, src_run_ids`。**feasible＝ε 判定**（32K 生成答對率對 E0 的配對差 CI 下界 ≥ −2 個百分點）；MLPerf 99%／99.9% 只另列。
- `x2_unmapped.csv`：對不到 X3 解碼段的組合（同樣的欄位，沒有 x3_segment、多一欄 `reason`）。
- 沒有任何 32K 的 X2 列（例如 X1 判定不跑 X2）→ **兩個檔都不寫**（X3 的 `load_x2` 遇到空檔會丟例外；沒有檔＝模式 X 只看時間）。
- CPU self-test 端到端驗過：合成的 X1 列 → report → x2_selection → 合成的 X2 列 → gen-report → **`m16_xtime.load_x2` 讀得懂**（欄位、名稱都在 seg_catalog 裡）。

**對應表**（逐一讀過 X3 的實作：`lr_r<r>`＝K、V 同秩、每 head 校準基底、BF16 係數；`xkv_g4_f<f>`＝每 4 層、K、V 同秩 4·H·128/f；`merge_g<g>`＝存 T/g、展開；`lrk24_v4`＝K PCA r24 BF16＋V int4_kivi；`fp8`＝fmt_choice 的粒度）：

| 我們的組合 | x3_segment |
|:--|:--|
| `none`（MASK0） | bf16 |
| Llama `q:fp8_token`／Qwen `q:fp8_head` | fp8 |
| `q:int8_token` | int8_token |
| `q:int4_kivi` | int4_kivi |
| `q:int2_kivi` | int2_kivi |
| `q:kv-int4_kivi-int2_kivi` | k4v2 |
| `lr:k16:v16`、`lr:k24:v24`、`lr:k32:v32`、`lr:k40:v40`、`lr:k64:v64` | lr_r16、lr_r24、lr_r32、lr_r40、lr_r64 |
| `xl:k16:v16`、`xl:k32:v32`（X3M 組，為了對上才加的） | xkv_g4_f8、xkv_g4_f4 |
| `merge:2`、`merge:4` | merge_g2、merge_g4 |
| `lr:k24+q:kv-bf16-int4_kivi`（L5a 的 K24V4） | lrk24_v4 |
| `merge:2+q:int4_kivi` | merge_g2_int4 |
| （沒有） | droptok_k50、droptok_k25、droptok_k50_int4 |

**對不到、不硬對的**（寫進 `x2_unmapped.csv` 的原因）：
- **所有 drop**：我們丟的是 512-token block（所有層、head 一起，key_keep 遮掉；沒有逐 token 位置、不用 scatter），X3 的 `droptok_k*` 是每（層, head）各自留 token＋int16 位置——位元組與解碼都不同。block 丟＋量化其實等於「對應的量化段只搬留下的 block」，X3 若加一個 block 數縮放就能用（§13 要不要請 X3 補）。
- xKV 的 1:1.5 秩（X1 組）、有量化的 xKV；聯合 PCA、其他秩、係數量化的低秩；g＝8／16 或其他量化的合併；mc／share；X3 沒有的量化格式（fp8k0、fp8e5、mxfp4、nvfp4、int4_token、int4h、int2h、K8V4、K2V4…、vllm_fp8）。
- 只有被 X2 量過的組合才會出現在兩個檔裡（X2 的 x3 層就是為了把對得上的段都量到）。

---

## 8. 時間估計（全部〔估〕）

**依據**：
- **A（X1 的單價）**：DS1（44e1）實測——兩個模型各 96 題 × 203 次 TF＋32K prefill＋6 種分數，12 個 process 分在兩張卡，00:11:57 → 00:57:50Z（run `20261020-001157-m10-ds1-llama-s0` … `20261020-003931-m10-ds1-qwen-s5` 的 stdout 時間）＝45.9 分鐘 → 每張卡每題約 28.7 秒（含 prefill、分數）、每列 TF 約 0.141 卡秒。DS1 的 Llama peak 39.9 GiB、Qwen 25.3 GiB（同上 stdout）。
- **B（X2 的單價）**：43ab 實測——37,620 列生成（＋TF）／19.88 小時（10/17 13:44:26 → 10/18 09:37:21Z，RUNLOG_MI300X 11.2o、11.2p），兩張卡、最多 8 個 process → 1,892 列／小時（L5d 也用這個）→ 每列約 3.8 卡秒。
- **C（新的工作，NOT_MEASURED）**：每個組合的壓縮／解碼、重建誤差、代理（每層兩個 [H, 256, 32768] 的 FP32 matmul）——〔估〕每個組合 0.1–0.3 秒；**xKV 每個組合要 8 個窗口 × 2 半的 `svd_lowrank`（[32256, 4096] FP32，含 QR）——ROCm 上的 QR 速度不知道，〔估〕每個組合 1–5 秒**，X1 每題約 18 個 xKV 組合。冒煙會印每列的 `t_pipe_s`。

| job | 內容 | GPU 時間〔估〕 | 依據 |
|:--|:--|:--|:--|
| 02_smoke_x | 每張卡：1 題校準、gpu-selftest、1 題 screen（17 組合＋S2 子集）、gen 32K 1 題、gen 128K 1 題＋CPU 報告 | 0.25–0.5 h | 每張卡載 5 次模型；128K prefill 約 32K 的 4–6 倍（32K＝8 秒，e0_tune 實測） |
| 36a／36b／36c | X1：每題約 230 列 TF（S2 子集約 68＋組合 161＋E0）＋C | 每個模型 96 題共 **1.8–4.6 h**（每題 67–171 卡秒＝A×230/203＋C）；三段的容量約 5.5 h；做到 48 題（report 的下限）約 0.9–2.3 h | A＋C |
| 36z | CPU 報告 | <0.3 h（CPU） | bootstrap 約 640 次 × 10,000 |
| 37a–37d | X2 32K：core（17–19 個配置＋E0）→ x3（13–15）→ ext（約 14），每個模型 96 題 | core **1.8–2.0 h**、x3 1.4–1.6 h、ext 約 1.5 h（每張卡；兩張並行）→ 合計 **4.7–5.1 h**，四段的容量 6.7 h | B（配置數是合成資料跑 select_x2 的數，真的數要等 X1） |
| 37e／37f | X2 128K：8 個配置＋E0 × 48 題 | 約 **1.3–2.0 h** | B × 3（44c：252 block 的 forward 是 64 block 的 1.9 倍，解碼看 4 倍的 key）＋每題 prefill 約 40–60 秒 |
| 37z／37zz | CPU 報告 | 數分鐘（CPU） | — |

**合計**：全部約 **8.8–12.2 小時的兩卡牆鐘**；**最小版**（02＋X1 做到約 48 題＋X2 core）約 **3.5–5 小時**。旋鈕（在 `x_common.sh` 或 job 前 export）：`X1_N`（24 → 12 約省一半 X1）、`X2_N`、各 shard 數；或只排 37a／37b（core）＋37z。**MI300X 租約 10/23 09:00Z、第二階段優先，排不排、排到哪一段請協調者決定。**

每個 GPU job 都用 `--budget-min`（36a 95 分鐘、其他 100 分鐘）：超過就不再開新的題，所以單一 job ≤2 小時（最後一題的尾巴〔估〕≤10 分鐘＋校準／載模型）；沒做完的題由下一段 job 接著做（逐題續跑、`done_union` 看同一組 shard 檔，換 shard 數也安全）。

---

## 9. 怎麼跑

```bash
# 佇列順序（runner 在兩個第二階段佇列都空時才開 GPU job；cpu 佇列可以和 GPU job 同時跑）
cp code/m16_queue/jobs/02_smoke_x.sh code/m16_queue/jobs/36[abc]_x1_screen.sh code/m16_queue/jobs/37[a-f]_x2_gen.sh \
   /mlsteam/data/tiara/stage3/jobs/queue/
cp code/m16_queue/jobs/36z_x1_report.sh code/m16_queue/jobs/37z_x2_report.sh code/m16_queue/jobs/37zz_x2_report_final.sh \
   /mlsteam/data/tiara/stage3/jobs/queue_c/
# x_common.sh 不要複製（job 從 repo source 它）
```

- 依賴：36a／36b／36c `AFTER: 02_smoke_x`；36z `AFTER: 36a 36b 36c`；37a–37f `AFTER: 36z_x1_report`；37z `AFTER: 37a 37b 37c`；37zz `AFTER: 37z 37d 37e 37f`。**runner 要所有 AFTER 都有 .done 才開**：沒排的 job 要從後面 job 的 AFTER 拿掉（例如不排 36c → 改 36z 的 AFTER）。
- 37* 開頭都先看 `x2_selection.json` 的 `go`：false → 不佔 GPU、exit 0（照實寫 log）。
- 手動跑單一步（GPU 窗口；`source code/m7_env.sh` 之後）：
  - `m7run m16-x-calib-llama python code/m16_xstack.py calib --model llama`
  - `m7run m16-x1-llama-s0 python code/m16_xstack.py screen --model llama --calib /mlsteam/data/tiara/stage3/xstack/calib_llama.pt --shard 0/3 --out /mlsteam/data/tiara/stage3/xstack/x1/x1_llama_s0.csv.gz`
  - `python code/m16_xstack.py report --raw "/mlsteam/data/tiara/stage3/xstack/x1/x1_*_s*.csv.gz"`（CPU）
  - `m7run m16-x2-llama-32K-core-s0 python code/m16_xstack.py gen --model llama --ctx 32K --tier core --calib … --shard 0/3 --out …/x2/x2_llama_32K_core_s0.csv.gz`
  - `python code/m16_xstack.py gen-report --raw "/mlsteam/data/tiara/stage3/xstack/x2/x2_*_s*.csv.gz"`（CPU）
- CPU 自測：`OMP_NUM_THREADS=4 nice -n 19 taskset -c 0-3 python code/m16_xstack.py --self-test`（不 import torch；111 項）。

---

## 10. 輸出怎麼讀

- **`tables_x1.md`**（每個模型）：前緣（可部署、非對照；位元組比 vs 平均 Δgold_lp、Δfirst_ok、KL、attn_err）；每個預算的最好單一方法、各家族最好的、最好的組合；代理的 ρ 與 AUROC；最後是停止條件的成分 JSON（判讀由人）與 X2 go／不跑。
- **`configs_<m>.csv`**：每個組合一列（run_id、ts、位元組比、Δgold_lp／Δfirst_ok 的 CI、KL、MLPerf 欄、代理、`on_front`、無損乘數與 `ratio_lossless`）。`pareto_<m>.png`：圓點＝單一、三角＝組合、叉＝對照，虛線＝四個預算。
- **`x2_selection.json`**：`go`、`reason`；每個模型的配置清單，每個配置有 `roles`（anchor／deploy／best_single／primary／front／naive／random_drop／family_best:*／naive_x1／x3_segment）、`tiers`、`budgets`、`why`、X1 的數字。
- **`tables_x2.md`**：每個（模型, 長度）一張表（答對率、對 E0 的差與 CI、ε、≥99%、≥99.9%）；最後是 M1／M2／Holm／完整性的成分 JSON。
- **`s2_subset/tables_s2.md`**：S2 原本的報告（L1 G1 的成分），用 X1 跑出來的 S2 子集。
- **污染**：品質實驗。GpuWatcher 污染時分數保留、逐列記 `own_gpu_intruders`、`host_level`（AMD 上是 `UNOBSERVABLE_HOST`，規則 7），run 目錄留 `CONTAMINATED`；`t_pipe_s`／`t_tf_s`／`t_gen_s` 在污染的 run 作廢、**任何時候都不拿來做時間結論**（時間是 X3 的事）。
- **重現檢查**（`*.checks.jsonl`）：`x1_vs_s2`（X1 的 MASK0 與量化單一格式＝S2 子集同一格，逐值）、`x1_vs_ds1`（X1 的丟 25／50／75%＝DS1 同一格）、`s2_vs_ds1`（S2 自己的）、`e0_vs_s2`、X2 128K 的 `vs_44c`。正式跑時差 > 1e-4 只記錄（GPU 不決定性）；冒煙時有差就 exit 1。

---

## 11. 冒煙（02_smoke_x）要特別看的

1. **gpu-selftest 全過**（`smoke/gpu_selftest_<m>.json`）：
   - `rope_probe_k_proj_vs_inverse`：correct ≤0.02、比三種錯的寫法小一半以上（規則 6）；
   - `q_bitwise_vs_m10_codec_x[*]`：13 種格式＋2 種 kv- 逐位元、位元組＝Enc.nbytes；
   - `vllm_fp8_torch_fnuz_vs_numpy_grid`：**ROCm 上 `torch.float8_e4m3fnuz` 的轉型能不能在 GPU 上用、是不是最近值平手取偶**；同一欄也印被截的比例與 max|K|（Qwen 應該看得到 >240）；
   - `coef_kivi4_role_*_eq_int4_kivi_half`；`torch_vs_numpy[*]`（低秩、合併、mc 的容許誤差 1e-3–1e-2）；
   - `xl_randomized_vs_exact[*]`：隨機 SVD 的誤差 ≤ 精確 SVD 的 1.05 倍；不過的話要調 niter（**這是寫死的設定，改了要記**）；
   - `lr_full_rank_relerr`：只剩 RoPE 逆轉／轉回與 BF16 捨入，應該 ≪1e-2；`asymkv_equivalence_g*`；`proxy_*`；`tf_pipeline_eq_load_fmt_place`。
2. **screen --smoke**：`x1_vs_s2`、`x1_vs_ds1`、`s2_vs_ds1` 全部逐值相同；位元組的 assert 沒有觸發；**每個 process 的 peak GiB**（決定 `X_SHARDS_*` 和 NEED-GB）；**每列的 `t_pipe_s`**（特別是 `xl:*`、`lr:*`、`mc`）與每題秒數 → 回頭修 §8 的估計與 shard 數。
3. **calib**：RoPE 核對的數字、能量表（`smoke/calib_<m>.json`；1 題，只看合不合理）、花多久。
4. **gen 128K**：Llama 的 peak（〔估〕80–100 GiB；KEYDIFF 的分數在 128K 會多約 17 GiB 暫存）、`vs_44c` 的 E0 是否逐值相同、每題秒數。
5. **report／gen-report** 跑得完（題數不夠 → `go＝false`、`INCOMPLETE`，這是預期的）。

---

## 12. 還沒驗證的假設、已知限制

1. **GPU 路徑從沒跑過**：上面的冒煙是第一次。
2. **vllm_fp8 的截斷**：假設 vLLM 的 ROCm kernel 對超出 ±240 是截（不是溢位成 NaN）。NOT_VERIFIED（L6 也沒核到）。
3. **代理只看單層、用 E0 的 q**：看不到誤差跨層累積、生成長度變化（L5d §3.2 第 3 點）。
4. **xKV 用隨機 SVD**（照 xKV 用 `svd_lowrank`；seed 固定），冒煙比精確 SVD；niter 4、多取 16 個是寫死的。
5. **xKV 不去平均**（照 xKV）；PCA 去平均（照 L5a，Qwen 的 K bias）。
6. **TM-g 的 key 是 post-RoPE 平均**（L5b 的 TM-g）；pre-RoPE 平均再轉到組中心的版本沒做。
7. **MiniCache 的保留**照正文語意保留「最不像」的 5%（原文式 (4) 字面相反，L5b X1 的 ⚠️）。
8. **lr／xl 的 K 走逆 RoPE → RoPE**：即使滿秩也有 BF16 捨入量級的誤差（gpu-selftest 量給你看）。
9. **校準**：8 題、約 25 萬 token（L5a 建議 13–26 萬、KVTC 16 萬）；T2 的重複句子也在裡面。基底跨任務退化（L5a §4.4）沒有另外量。
10. **ε（TF）**：X1 用 first_ok 平均 ≥ −0.02 的點估計當「ε 內」，只拿來挑、不判定。
11. **X2 新題**（seed 62000）沒有第二階段的 E0 可以對照；E0 在 X2 裡自己量。
12. **128K 的切片**只放 8 個配置左右，探索、不判定；校準基底是 32K 的題算的（pre-RoPE 和位置無關）。
13. **時間欄**只是紀錄（品質實驗，多個 process 同卡），不拿來做任何時間結論。

---

## 13. 需要人決定的事

1. **排多少**（§8）：全部約 8.8–12.2 小時兩卡牆鐘，最小版約 3.5–5 小時。租約與第二階段優先，請協調者決定排到哪一段（可只排 02、36a–c、36z、37a、37b、37z）。
2. **S2 要不要另外跑**：X1 已經用 S2 自己的 `run_item` 跑了 S2 的全部格子（region＋full、DS1 重現檢查），寫成 S2 的格式、36z 也跑 S2 的 report。若佇列裡另有 S2 的 job，是重複的；反過來若只要 S2、不要 X1，S2 單獨跑比較快。
3. **X3 的 droptok**：我們的丟是 block 粒度，對不到 X3 的 `droptok_k*`；要不要請 X3 補「block 丟（只搬留下的 BF16 block、不用解碼）」與「block 丟＋量化」兩種段，或在 X3 的模式 X 用「量化段 × 留下的比例」近似。
4. **vllm_fp8 的截／溢位**：如果要確定 vLLM 的行為，需要另外讀 ROCm kernel 或在 vLLM 上實測（不在本任務範圍）。
5. **共用檔**：這次**不需要改任何共用檔**。唯一的「侵入」是 `reuse_item` 在 X1 的行程裡暫時換掉 `m10_sweep.Item` 這個名字（省一次 prefill；有次數核對）；不想要的話，改成每題 prefill 兩次（每題多約 8 秒、每個模型多約 13 分鐘〔估〕）。
