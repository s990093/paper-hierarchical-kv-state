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

---

## 14. 10/21 延伸：X2→X3 對應、X2b（加題數）、xKV 事先校準的基底、兩卡共用工作池

> **狀態（2026-10-21 03:5xZ）**：程式與 job 檔寫好，**還沒有在 GPU 上跑**（寫的時候第三階段自己的 37d 正在 GPU 上量，不能開 GPU 行程、不能 import torch）。
> 做過的：`python -m py_compile`；`--self-test`（CPU、不 import torch）**147 項 ALL PASS**（10/21 加的見 §14.9）；job 檔 `bash -n`；用假的 `m7run`（GPU 指令只記錄、`plan`／`pending` 真的跑）把 38b1／38c1 的工作池整個建一次（34 個單位、兩張卡都有領、每個產生的指令都用真的 argparse 解析過）；
> `x2b-report` 在第一版 X2 的真實逐列上跑一次（輸出到 scratchpad、不進 results）：變體 A 的 M1 和第一版 `x2_report.json` 逐項相同。
> **判定規則在看到任何 X2b／xKV 校準的數字之前寫死在這裡與程式常數（`X2B_TEXT`、`XLCAL_TEXT`）**，之後不改。時間全部〔估〕，依據在 §14.6。self-review，不是 cross-model review。

### 14.1 為什麼

1. **統計力**：X2 每個模型 96 題時，答對率差的 95% CI 半寬約 3–4 個百分點（例：Llama `xl:k28:v42+q:int4_kivi` −2.2 [−5.8, +0.7]），ε 規則（CI 下界 ≥ −2）連 INT4 KIVI（Llama −0.8 [−3.7, +1.4]）都過不了，Llama 的 M1 判不了。→ **X2b**：同樣的生成計分、新題 192 題／模型，和 X2 的 96 題合併成 288 題。
2. **實用性**：現在的 xKV 每題用整段 KV 算 SVD（基底每題一份、算在請求的位元組裡）。如果基底可以事先用校準題算好、所有題共用，每請求就不用帶基底（Llama 的 `xl:k28:v42+q:int4_kivi`：0.1202 → 0.0854〔算術〕），也省掉每題的 SVD。→ **`xl:…:cal`**。
3. **X3**：X3 的 map 說「X NO-GO 但 xKV+INT4 段缺」；X3 另一個 agent 加了同名的 `xkv_k*` 解碼段 → 我們補 **X2→X3 對應**。
4. **只有一張卡在動**（協調者 10/21 補充）：X1／X2 的 job 是「卡 0 Llama、卡 1 Qwen」，先做完的那張卡閒到 job 結束。→ 38b／38c 改用 **兩卡共用工作池**（§14.5）。

### 14.2 X2→X3 對應（`X3_MAP_1021`）

| 我們的組合 | x3_segment |
|:--|:--|
| `xl:k11:v17`、`xl:k22:v34`、`xl:k28:v42` | `xkv_k11v17`、`xkv_k22v34`、`xkv_k28v42` |
| `xl:k11:v17+q:int4_kivi`、`xl:k22:v34+q:int4_kivi`、`xl:k28:v42+q:int4_kivi` | `xkv_k11v17_int4`、`xkv_k22v34_int4`、`xkv_k28v42_int4` |
| `xl:k22:v34+q:int8_token` | `xkv_k22v34_int8` |
| `xl:…:cal`（任何） | 對不到（基底常駐、不用每題搬；X3 的 xkv_* 把每題基底算進要搬的位元組） |

- 名字和 `m16_xtime.py` 的 `XKV2` 逐字相同（10/21 核對過；`--self-test` 的 `e2e_x3_load_x2_accepts_x2_feasible`／`x2b_report_x3_load_x2` 用 X3 的 `load_x2` 讀我們寫的檔）。
- **注意（要人決定，§14.11 第 1 點）**：對應是全域的，所以佇列裡的 **37zz（重出第一版報告）會把這 7 個組合從第一版的 `x2_unmapped.csv` 移到 `x2_feasible.csv`**（其他列不變；self-test `x3_map_v1_rows_unchanged_except_1021` 核對）。
- 舊的 self-test 有個 bug 一起修了：`ok &= _selftest_report_e2e(check)` 會在函式跑之前就讀 `ok`，函式裡的失敗被蓋掉；改成先呼叫再合併。修完之後原本的 e2e 檢查照樣全過。

### 14.3 X2b：加題數（38b*）

**設計（事先寫死；`X2B_TEXT`、`code/m16_xstack_combos_v2.json` 的 `x2b`）**
- 題目：新題 **seed 63000 起、T2–T5 各 48 題＝192 題**／模型，32K，和 X1（60000）、校準（61000）、X2（62000）不重疊（self-test `seeds_disjoint`）。
- 配置（照列的順序量；同一個每題 SVD 的排在一起）：協調者列的 10 個——`none`、`q:int4_kivi`、部署對照（Llama `q:vllm_fp8`、Qwen `q:int8_token`）、`q:kv-int4_kivi-int2_kivi`、`xl:k11:v17`、`xl:k16:v16`、`xl:k11:v17+q:int4_kivi`、`xl:k22:v34+q:int4_kivi`、`xl:k28:v42+q:int4_kivi`、`xl:k22:v34+q:int8_token`；
  **另加「池子缺口」的單一方法**（只用第一版 X2 的數字決定、看到 X2b 之前）：第一版 M1 單一方法池裡、位元組比 < 第一版 B*(單一) 而且 CI 上界 ≥ −ε 的——兩個模型的 `q:int2_kivi`、Qwen 的 `xl:k22:v34`。不加的話合併後 B*(單一) 可能偏大 → r 偏小（**偏向 GO**）。Qwen `xl:k32:v32`（0.2659）沒加（每題多一次 SVD，約 0.5 卡小時〔估〕），由 report 的 `pool_gaps` 標出。→ Llama 11 個、Qwen 12 個（加的 3 個多半共用快取或沒有 SVD，成本約 +0.3 卡小時〔估〕）。
- roles／budgets 照第一版 `x2_selection.json`（M2 的主要組合、最好的單一方法不重選）；計畫檔 `v2/x2b_selection.json` 由 `plan --which x2b` 寫（已存在而內容不同 → 停，不默默改）。
- `gen --xl-cache`：同一題裡同一個 (K/V, 秩) 的每題 SVD 只算一次（`xl:k22:v34+q:int4_kivi` 與 `+q:int8_token`、`xl:k11:v17` 與 `+q:int4_kivi`、Qwen 的 `xl:k22:v34` 三個）。SVD 的輸入逐位元相同（每個配置前都還原 KV），所以結果應該和不用快取逐位元相同——**冒煙的 gpu-selftest `xl_cache_eq_uncached` 核對**（SVD 本身若不決定，退一步要求快取的差 ≤ 重算一次的差）。

**合併與判定（事先寫死；照第一版的規則重算，不換規則）**
- 配置集合 **C**＝X2b 的配置裡、第一版 X2 有量的（≥90% 的 X2 題有）；題目＝E0 與 C 的每個配置都有的題（**每題所有配置都有、逐題配對**；排除的題數另列）。X2b 有、X2 沒有的（Qwen `xl:k22:v34+q:int8_token`）只在「只用 X2b」報。
- 四種變體並列：**A**＝只用 X2、第一版的整個池（重現第一版）；**B**＝只用 X2、池子限制在 C；**C**＝只用 X2b；**D**＝X2＋X2b。主要讀 **D**；A→B＝限制池子的效果、B→D＝加題數的效果、C＝獨立重複。
- 寫的時候的試算（不是結果、不進 results）：`x2b-report --raw-x2 "/mlsteam/data/tiara/stage3/xstack/x2/x2_*_s*.csv.gz" --raw-x2b <沒有檔>`（輸出在 scratchpad）——變體 A 的 M1 和第一版 `x2_report.json` 逐項相同；兩個模型共 192 題的 E0 被量了不只一次，分數不同的 0 題。正式數字以 38bz 的 `x2b_report.json` 為準。
- 每個變體照第一版的 `judge_x2` 原樣算 M1（B*、r、GO／ASK／NO-GO）、M2（每個預算的主要組合 − 最好的單一方法、Holm）、完整性、MLPerf 99%／99.9% 的 B* 與 r；每個配置另列答對率、對 E0 的差與 CI、ε、≥99%／≥99.9%、CI 半寬（X2 → D）。`:cal` 不進任何池。
- **池子缺口**（只標註、不改判定）：第一版池子裡、不在 C 的配置——單一方法若位元組比 < D 的 B*(單一)（或不存在）且 X2 的 CI 上界 ≥ −ε → r 可能偏小（偏向 GO）；組合同理 → r 可能偏大（偏向 NO-GO）。用第一版的真實逐列試算 B 時，Llama 的單一方法缺口是 `q:int2_kivi`（已加）、組合缺口 `drop:KEYDIFF:0.25+q:int4_kivi` 等 3 個；Qwen 單一方法缺口 `q:int2_kivi`、`xl:k22:v34`（已加）、`xl:k32:v32`。
- 完整性：X2b 每個模型 192 題齊全才算完成（`x2b_complete`）；判定本身照第一版的 ≥48 題。
- 給 X3：`v2/x2_feasible.csv`、`v2/x2_unmapped.csv`（同第一版欄位＋`source`）：在 C → D 的題；只在 X2b → C；其他 → A（第一版的題）。**不覆蓋第一版的檔**（`--out-dir` 是第一版的目錄就停）。

### 14.4 xKV 事先校準的基底（38c*）

**作法**
- `calib-xl`：校準題＝和 PCA 的 `calib` 同一套（seed 61000 起、T2–T5 各 2 題、約 25.8 萬 token；和所有評測題不重疊，`load_calib_xl` 再核對一次）。每個窗口（4 層）、K（模型自己的 cos/sin 逆 RoPE）／V 各一個：Z＝[token, 4·H·128]，**不去平均**（照 xKV 與每題 SVD 版），FP64 累加 ΣZᵀZ，最後 eigh 取前 R 個特徵向量（R＝每 head 每層等效秩 64 × 4 × H：Llama 2048／4096 維、Qwen 1024／2048 維；存檔 Llama 每半邊 268 MB〔算術〕）。第一題做 RoPE 的資料自身核對（規則 6）。摘要寫 `v2/calib_xl_<m>.json`（run_id、sha256、能量）。
- `xl:k<r>:v<r>:cal`：係數＝Z·U[:, :r]（FP32）、照常量化；重建用 BF16 的基底（和每題 SVD 的基底同一個存法）。和每題版同樣的窗口、Z、RoPE、量化——只差基底從哪裡來。不接受 `xl:…:cal+drop`（基底和題目無關，順序對照沒有意義）。
- **位元組帳**：每請求＝係數（＋量化的 scale／zero）；基底列 `shared.k.basis`／`shared.v.basis`（大小和每題的基底相同），不算進請求。**攤提**〔算術〕：S 由 N 個請求分攤，每請求多 S/N；N＝1 時正好等於每題 SVD 版（打平點恆為 1）。例（區內位元組比）：

| 配置 | 每題 SVD | 校準（N→∞） | N＝8 | N＝64 | 共用基底 |
|:--|--:|--:|--:|--:|--:|
| Llama `xl:k11:v17` | 0.1233 | 0.1094 | 0.1111 | 0.1096 | 58.7 MB |
| Llama `xl:k22:v34+q:int4_kivi` | 0.0961 | 0.0684 | 0.0718 | 0.0688 | 117.4 MB |
| Llama `xl:k28:v42+q:int4_kivi` | 0.1202 | 0.0854 | 0.0898 | 0.0860 | 146.8 MB |
| Qwen `xl:k11:v17` | 0.1163 | 0.1094 | 0.1102 | 0.1095 | 12.8 MB |
| Qwen `xl:k22:v34+q:int4_kivi` | 0.0822 | 0.0684 | 0.0701 | 0.0686 | 25.7 MB |
| Qwen `xl:k28:v42+q:int4_kivi` | 0.1028 | 0.0854 | 0.0876 | 0.0857 | 32.1 MB |

  實際部署只要存最高秩的一份基底（低秩取前幾欄），所以 S 以最高秩那份為準；表裡照每個配置自己的秩算。

**流程**：`calib-xl` → **TF 篩選**（X1 的前 12 題／任務＝48 題；`screen --no-s2 --combos code/m16_xstack_combos_v2.json`：E0＋3 個校準版＋每題 SVD 的錨點 `xl:k11:v17`）→ **生成**（X2 的 96 題、seed 62000：3 個校準版＋3 個每題 SVD 的同名配置＋E0，**同一題、同一個行程配對**）。

**判定（事先寫死；`XLCAL_TEXT`）**
- 每一對（`xl:k11:v17:cal` vs `xl:k11:v17`、`xl:k22:v34:cal+q:int4_kivi` vs `xl:k22:v34+q:int4_kivi`、`xl:k28:v42:cal+q:int4_kivi` vs `xl:k28:v42+q:int4_kivi`），生成答對率逐題配對差（校準 − 每題 SVD），bootstrap 10,000 次的 95% CI：
  - **CI 上界 < −2 個百分點 → MUST_PER_ITEM（必須每題算 SVD）**；
  - **CI 下界 ≥ −2 個百分點 → CAN_PRECOMPUTE（可以事先算好基底）**；
  - 其他 → UNDETERMINED（題數不夠分辨）。
- TF 篩選用的 48 題，X1 的逐列裡兩個模型都有 E0 與 3 個每題 SVD 的同名配置（10/21 核對：48／48）。
- 每個模型：三對都 CAN → CAN；任一對 MUST → MUST（至少那個秩要每題算）；其他 → UNDETERMINED；題數 < 48 → INCOMPLETE。整體：兩個模型都 CAN → CAN；任一 MUST → MUST；其他照實寫。
- 同秩比較：校準版的每請求位元組一定比每題版少（少掉每題基底），所以 CAN＝「同秩不比較差、而且每請求更省」。
- **TF 只報、不判定**：每題 SVD 的 TF 取自 X1 同一題的列（跨 run）；E0 的 gold_lp 和 X1 是否逐值相同、錨點 `xl:k11:v17` 和 X1 是否逐值相同都列出來（決定跨 run 的 TF 比較能不能信）。生成那段另列 E0 與 3 個每題 SVD 配置和 X2 同一題的逐值重現。

**gpu-selftest 的新檢查**（`--new-checks xlcal --calib-xl …`）：校準檔的形狀、窗口、R、`UᵀU≈I`（≤1e-3）；8 層切片上 torch vs numpy 參考（4 個 :cal 組合，含先 drop；相對誤差 ≤1e-2、位元組＝預測、沒有每題基底的鍵）；**Eckart–Young**（同一份資料上，校準基底的 pre-RoPE 誤差 ≥ 精確 SVD 的 0.999 倍，每個窗口、K／V）；整個 32K 區的 3 個 :cal 配置：位元組＝預測、有限值，記重建誤差與 `t_pipe_s`。
`--new-checks x2b`：CPU 版出題＝`m10_quality.plan_interleaved`（3 組 seed）；X2b 每個配置在切片上的位元組與有限值；`xl_cache_eq_uncached`（兩組）。

### 14.5 兩卡共用工作池（`code/m16_queue/jobs/x_pool.sh`；38b*／38c* 用）

- 工作切成固定的單位清單（(模型, shard i/n, 階段)；**n 在兩張卡之間共用、不重疊**）；X2b：Llama 12 份（每份 16 題）、Qwen 8 份（24 題）；xKV 校準：TF 每個模型 2 份、生成 Llama 6 份／Qwen 4 份。兩個模型照 (i+0.5)/n 交錯排（`xp_order`）。
- 每張卡一個 worker，照順序領「第一個還沒被領、這張卡放得下的」單位；**領取鎖＝`mkdir 池/claim/<k>`**（原子）。每卡同時 ≤ `XP_MAXP`＝4 個行程、登記 ≤ `XP_CAP_GB`＝165 GiB（Llama 每行程 46、Qwen 30：實測 peak 39.8／41.8、25.1／26.6 GiB＋快取／基底）→ 檔頭 NEED-GB 170。
- 領到先問 `m16_xstack.py pending`（CPU、不載模型）：0 題 → 跳過；否則開行程，`--budget-min`＝開跑時離工作池預算結束還有幾分鐘。逐題續跑照舊：每段 job 重建工作池，做完的 shard 被跳過。
- 每個單位結束時由它自己的子 shell 寫 `池/claim/<k>/rc`，worker 每 `XP_POLL_S`（5 秒）看一次——**不用 `wait -n`**（bash 已經回收的背景工作 `wait -n` 會說 no such job，self-test 抓到）。任一單位失敗 → 那張卡不再領新的、等手上的跑完 → job 寫 NEEDS_HUMAN、exit 1。
- 不動 `x_common.sh` 的 `x_par`／`x2_model`／`x2_seq`（37e／37f 還在用）。冒煙在第 1 段（38b1／38c1）跑一次、過了寫 `SMOKE_OK`（記程式的 sha）；後面幾段沒有它、或程式在冒煙之後改過 → NEEDS_HUMAN、exit 1。

### 14.6 時間估計（全部〔估〕）

**依據（實測）**：
- X2 的逐列檔（`/mlsteam/data/tiara/stage3/xstack/x2/*.csv.gz`；每題寫入的 `ts`、run 目錄名的開始時間；同一張卡上並行的 shard 合計）：Llama core（17 個配置、8 個 xKV、3 個 shard）**8.98 卡秒／列、158.7 卡秒／題**；Llama x3（14 個配置、1 個 xKV）3.44 卡秒／列；Qwen core（17、6 個 xKV、4 個 shard）3.78；Qwen x3 3.14、ext 3.41。對照 43ab 的 1,892 列／小時／兩卡＝3.8 卡秒／列（§8 的 B）。**xKV 的每題 SVD 是 Llama 的主要成本。**
- X1 逐列的 `t_pipe_s`（同卡 3–4 個行程時的中位數）：Llama `xl:k11:v17` 38.4 s、`xl:k16:v16` 33.4、`xl:k22:v34+q:int4_kivi` 58.5、`+q:int8_token` 51.9、`xl:k28:v42+q:int4_kivi` 79.0；Qwen 15.8、19.8、34.4、34.9、44.4；量化 ≤0.44 s。
- 由這兩個推每個行程每題的時間 ≈ 列數 × 每列基本時間（Llama 5.6 s、Qwen 6.6–8.1 s：X2 每題時間扣掉 xKV 的 `t_pipe_s` 後平均）＋Σ xKV 的 `t_pipe_s`（快取命中的算 1–2 s），卡秒／題＝÷ 每卡行程數（X1／X2 量的時候 Llama 3、Qwen 4）。

| job | 內容 | GPU 時間〔估〕 |
|:--|:--|:--|
| 38b1 | plan＋冒煙（每卡：gpu-selftest x2b、1 題全部配置）＋工作池 | 冒煙 10–20 分鐘；本段 ≤ 約 1.8 小時（預算 100 分鐘＋最後一題的尾巴） |
| 38b2／38b3／38b4 | 工作池續跑 | 每段 ≤ 約 1.8 小時；做完的段數幾分鐘就結束 |
| X2b 全部 | Llama 192 題 × 12 列（Σ xKV 約 213 s，快取後）≈ 93 卡秒／題 → **4.0–6.0 卡小時**；Qwen 192 × 13 列（Σ 約 117 s）≈ 51–56 卡秒／題 → **2.7–3.0 卡小時** | 合計 6.7–9.0 卡小時 → 兩卡共用約 **3.3–4.5 小時牆鐘**（不用快取約 +1.3 卡小時）；38b1–38b3 的容量約 4.6 小時，38b4 是保險 |
| 38c1 | plan＋calib-xl（每個模型 8 題 prefill＋4096 維 FP64 的 ΣZᵀZ 與 eigh；**ROCm 上 4096 維 FP64 eigh 的速度 NOT_MEASURED**；PCA 的 calib 8 題 42 秒）＋冒煙＋工作池 | calib-xl 3–15 分鐘、冒煙 10–15 分鐘；本段 ≤ 約 1.8 小時 |
| xKV 校準全部 | TF：Llama 48 題 ≈ 22 卡秒／題（錨點 38 s＋校準版每個約 1–2 s）、Qwen ≈ 10 → 約 0.4 卡小時；生成：Llama 96 題 × 7 列（3 個每題 SVD Σ 176 s）≈ 74 卡秒／題 → 2.0 卡小時、Qwen ≈ 38 → 1.0 卡小時 | 合計約 3.4 卡小時 → 兩卡約 **1.7 小時**＋calib／冒煙 0.3–0.5 小時 → 38c1＋38c2（38c2 約 0.3–1 小時） |
| 38bz／38cz | CPU 報告 | 數分鐘（CPU；bootstrap 約 4 個變體 × 24 個配置 × 10,000 次） |

不確定的地方：同卡混跑兩個模型、每卡 4 個行程時的爭用（X1／X2 量的是同卡同模型 3–4 個）；Llama 每題的生成長度（壞掉的配置會生成到上限）。冒煙的 log 會印 peak GiB 與每題秒數，量到之後可以改 `XP_GB_*`、shard 數（`x_ext.sh`，或 job 前 export）。

### 14.7 怎麼排

```bash
cp code/m16_queue/jobs/38b[1-4]_x2b_gen.sh code/m16_queue/jobs/38c[12]_xkv_cal.sh /mlsteam/data/tiara/stage3/jobs/queue/
cp code/m16_queue/jobs/38bz_x2b_report.sh code/m16_queue/jobs/38cz_xkv_cal_report.sh /mlsteam/data/tiara/stage3/jobs/queue_c/
# x_common.sh、x_pool.sh、x_ext.sh 不要複製（job 從 repo source 它們）
```
- 檔名（位元組排序）：`38a_x3_micro2` < `38b1…38b4` < `38bz` < `38c1`、`38c2` < `38cz` < `38d*`（C1b）< `38z_x3_map2` < `39_x3_micro` < `39v_x3_validate` < `40_v0`。GPU job 都排在 39v、40 之前；佇列裡 38b 會整段先跑完才輪到 38c（runner 照排序、挑第一個依賴滿足的）——要先跑 38c 的話把 38c 改名或延後排 38b。
- 依賴：38b*／38c* `AFTER: 37z_x2_report`（已 done）；38bz `AFTER: 38b1–38b4`；38cz `AFTER: 38c1 38c2`。**沒排的 job 要從 AFTER 拿掉**（runner 要所有 AFTER 都有 .done）。
- CPU 自測：`source code/m7_env.sh; OMP_NUM_THREADS=4 nice -n 19 taskset -c 0-3 python code/m16_xstack.py --self-test`。

### 14.8 冒煙要看什麼

- **38b1**（`$XD/smoke/v2/`）：`gpu_selftest_x2b_<m>.json` 全過，特別是 `xl_cache_eq_uncached[*]` 的 `bitwise_equal`（不是 true 而 `svd_deterministic` 是 false 的話，要人看 rel／rel_recompute）與 `plan_cpu_eq_m10[*]`；`x2b_smoke_<m>_s0.csv`：1 題、E0＋全部配置各一列（`smoke_rows_ok`），gen 裡位元組＝預測的 assert 沒觸發；log 的 **peak GiB**（> 46／30 → 調 `XP_GB_*`）與這題花幾秒。
- **38c1**：`v2/calib_xl_<m>.json` 的 RoPE 核對（correct ≤0.02）、能量（只看合不合理）、`t_eigh_s`；`gpu_selftest_xlcal_<m>.json`：`xlcal_meta_shape_orthonormal`、`xlcal_torch_vs_numpy[*]`、`xlcal_eckart_young[*]`（`err_ratio_cal_over_svd` 也是第一眼的「校準基底在這題上差多少」）、`xlcal_full_region[*]` 的 `t_pipe_s`；TF 與生成各 1 題的列。
- 工作池：`$XD/pool/<job>/log`——兩張卡都有領、每個單位的 rc、做完被跳過的、沒領到留給下一段的。

### 14.9 輸出怎麼讀、自測

- `v2/tables_x2b.md`：每個模型一張表（只用 X2／只用 X2b／X2＋X2b 的答對率與 CI、ε、CI 半寬 X2→D、MLPerf）；判定表（A／B／C／D × 模型：B*、r、M1、MLPerf 的 r、池子缺口）；M2 表；給 X3 的檔。`v2/x2b_report.json` 有全部成分，含 `e0_repro_x2`（第一版 X2 每一層的 run 都各量一次 E0：同一題的 E0 分數是不是全部相同＝跨 run 決定性的證據）。
- `v2/tables_xlcal.md`：整體判定；每個模型的三對（差與 CI、判定、位元組比、共用基底、攤提）、每個配置對 E0、TF 篩選表與重現。
- `--self-test`：10/21 加了 a–h 八段（X2→X3 對應與第一版列的變化、:cal 語法、位元組（含「舊組合的位元組比＝X1 在 GPU 上 assert 過的實際值」161／161 個）、numpy 的 Eckart–Young、計畫檔與題目不重疊（含 CPU 版出題＝X2 逐列的題號）、pending 的切法、合併報告的配對（已知差的合成資料逐值核對、A＝第一版判定、不准寫第一版目錄）、xKV 校準判定的三種結果、**工作池每個單位恰好被領一次**、37d／37e／37f／37zz／02 的指令照樣解析）。

### 14.10 已知限制

1. GPU 路徑（calib-xl、:cal 的 torch 實作、--xl-cache、新的 gpu-selftest）都還沒在 GPU 上跑過；冒煙是第一次。
2. 校準基底不去平均（照 xKV）；校準題 8 題、約 25.8 萬 token，和 PCA 校準同一套。基底跨任務的退化沒有另外量（TF 的四個任務各有 12 題，表裡可以看分任務的差——report 沒有分任務列，要的話另算）。
3. X2b 的 C 只含第一版有量的配置；第一版池子裡其他的配置用 `pool_gaps` 標出，不補量。
4. xKV 校準的生成比較在 X2 的同一批 96 題上（為了和 X2 的每題 SVD 結果對照重現）；它和 X2b 的題不同。
5. 工作池的 GiB 是登記值，不是實際用量的上限（MI300X 192 GB；165 GiB 留約 27 GB 給其他）。

### 14.11 需要人決定的事

1. **37zz 會用新對應重寫第一版的 `x2_feasible.csv`／`x2_unmapped.csv`**（7 個 xKV 組合從 unmapped 移到 feasible；其他列不變）。若第一版的檔要保持 37z 當時的樣子，撤掉 37zz 或請我把新對應改成只在 v2 生效。
2. **X2b 多加了 3 個配置**（兩個模型的 `q:int2_kivi`、Qwen 的 `xl:k22:v34`；理由 §14.3，約 +0.3 卡小時〔估〕）。要嚴格照 10 個的話把 `combos_v2.json` 的這 3 個拿掉（要在 38b1 之前；之後改會被 plan 擋下）。
3. **排序**：照檔名 38b（約 3.3–4.5 小時）整段先跑完才輪到 38c（約 2 小時）。38c 的答案（能不能事先算基底）比較快出來；要先跑就調整排序。
4. **省 GPU 的選項（沒做）**：38c 的生成裡 3 個每題 SVD 的配置在 X2 已經量過同一批題（約佔 38c 的 2.6 卡小時）；X2 的 E0 跨 run 重量 192 次都逐值相同，跨 run 配對大概可信，但我照指示做同 run 配對。要省的話可以從 `xlcal_pairs` 的生成計畫拿掉每題 SVD、改用 X2 的列（要改 report 與事先寫死的規則，另開版本）。

---

## 15. X4：把「xKV 式跨層低秩＋量化」推到極限（10/21）

> **狀態（2026-10-21 04:3xZ）**：`code/m16_x4.py` 與 job 檔寫好，**還沒有在 GPU 上跑**（寫的時候第三階段的 37d 正在 GPU 上，不能開 GPU 行程、不能 import torch）。
> 做過的：`py_compile`；`python code/m16_x4.py --self-test`（CPU、不 import torch）**25 項 ALL PASS**；`m16_xstack.py` 沒有改（`git diff` 空），它的 self-test 照樣 147 項 ALL PASS（已排隊的 37e／37f、38b*、38c* 讀它，行為不變）；
> 用假的 m7run 把 X4 五種工作池（X4a、X4b、X4c、X4d、Qwen 256K）各建一次（52 個單位、`pending` 真的跑），產生的每條指令都用真的 argparse 解析過。
> 設計與判定照 `docs/stage3_20261019/x4_limit_plan.md`（**判定門檻沒改**；和計畫不同的地方在 §15.10，理由寫明）。時間全部〔估〕，依據在 §15.6。self-review，不是 cross-model review。

### 15.1 檔案

| 檔案 | 內容 |
|:--|:--|
| `code/m16_x4.py` | 子命令 `grid`、`tf`（X4a）、`select-b`、`gen`（X4b／X4c／256K／X4d 兩輪）、`select-cd`、`dump`、`lossless`、`report`、`gpu-selftest`；`--self-test`。只 import `m16_xstack` 的函式，**不改它** |
| `code/m16_xstack_combos_x4.json` | X4a 的配置（`m16_x4.py grid` 產生；每個配置有 roles、why、targets、倍數）；self-test 核對＝程式重算 |
| `code/m16_queue/jobs/x4_common.sh` | X4 job 的共用函式（工作池單位、冒煙、挑選、報告）。**不要放進佇列目錄** |
| `38e1–38e3_x4a_tf.sh` | X4a（GPU；38e1 開頭冒煙） |
| `38f1–38f4_x4b_gen.sh` | X4b（GPU；38f1 開頭挑選＋冒煙＋dump；38f4 是保險） |
| `38g1–38g2_x4c_128k.sh` | X4c 128K（GPU；38g1 開頭挑選前緣＋冒煙） |
| `38h1–38h2_x4d_turns.sh` | X4d 兩輪（GPU；38h1 開頭冒煙） |
| `38i1_x4c_256k_qwen.sh` | Qwen 256K 切片（GPU；排最後、可不排） |
| `38ez`、`38fy`、`38fz`、`38hz`、`38iz` | cpu 佇列：X4a 報告、無損、X4b 報告、完整報告、加 256K 的報告 |

輸出：摘要 `results/m16_stage3_mi300x/x_stack/x4/`（`x4_report.json`、`tables_x4.md`、`x4_best_allocation.csv`、`x4a_configs_<m>.csv`、`pareto_x4a_<m>.png`、`pareto_x4b_<m>.png`、`lossless_x4.json`、三個計畫檔 `x4b／x4c／x4d_selection.json`）；逐列 `/mlsteam/data/tiara/stage3/xstack/x4/`；dump 在本地盤 `/var/tmp/tiara_s3/kv_dumps/x4`（可重生）。**不寫第一版與 v2 的目錄**（`report` 擋）。

### 15.2 設計

**配比格點（Q-B；`grid`，位元組照 `predict_bytes` 實算：含每題基底、scale、遮罩）**
- 目標倍數 T∈{8, 12, 16, 20, 24, 32}；「落在 T」＝實際倍數在 T 的 ±10%。
- 每個 T 照優先順序放這些分法（每一種掃 K 秩 2–48、V 秩＝比例×K 秩，挑倍數最接近 T 且在 ±10% 內的）：
  1. 底：xKV＋INT4 KIVI（W＝4、K:V＝1:1.5、不丟）；2. 只降秩（BF16 係數）；3. INT8 係數；4. K4V2；5. K2V4；6. INT2；7. INT4、W＝2；8. INT4＋輕度丟 KEYDIFF 12.5%。
  每個 T 實際 7–8 種（Llama 的「只降秩」在 T＝20、32 找不到 ±10% 內的秩 → 寫在 JSON 的 `skipped`）。
- 輕度丟的完整掃描（Q-A）：T＝16、24 各 6 個（KEYDIFF／KNORM × 6.25／12.5／25%，INT4、W＝4）。
- 秩細格點（計畫 §2）：INT4、W＝4、不丟，12 組（1:1.5 的 6 組、K 少 V 多 5 組、(11,14)）——K／V 秩分配的軸在這裡。
- 對照：`none`、`q:int4_kivi`、`xl:k11:v17+q:int4_kivi`、`xl:k22:v34+q:int4_kivi`。
- 合計每個模型 71 個配置。共用 SVD 要的最大秩（每 head 每層等效）：Llama W＝4 K41／V62、W＝2 K34／V51；Qwen W＝4 K48／V72、W＝2 K37／V56（都是 T＝8 的低位元分法）。

**共用 SVD（和 X1／X2 不同）**：每題、每（K／V、W）只做一次 `torch.svd_lowrank`，q＝這題的配置裡最大的聯合秩＋16（seed、niter 照 m16_xstack），每個配置取前 r 欄（U·S＝係數、V＝基底）。X1／X2 是每個配置各做一次 q＝r＋16；q 越大，前 r 個奇異向量越接近精確 SVD，所以重建誤差 ≤ X1／X2 的同一個配置。不這樣做的話 X4a 光 SVD 就要約 16 卡小時〔估〕。gpu-selftest 核對：共用 SVD 對 numpy 精確 SVD 的重建誤差比在 [0.999, 1.05]（量化的配置 [0.98, 1.05]），另記對 m16_xstack 每配置 SVD 的誤差比。

**輕度丟一律 xl+drop**（基底在全部 token 上估、再丟 block 的係數；m16_xstack 的合法順序 `("xl","drop")`）：才能共用 SVD。X2 的 Qwen 兩種順序（丟 50%）−14.4 與 −15.4 差不多。

**題目**（self-test 核對不重疊）：X4a seed 64000 起 T2–T5 各 24 題（96 題）；X4b seed 65000 起各 48 題（192 題）；X4c＝44c 的 128K 題組（252 block、seed 40000、各 12 題＝48 題）；Qwen 256K＝504 block（258,048 token）、seed 40000、各 6 題；X4d＝C1a 的兩輪題組（seed 70000、T2–T4 各 32 題＝96 題）。lossless 的 dump＝X4a 的前 4 題（每個任務 1 題）。

### 15.3 X4a → X4b 的挑法（事先寫死；`X4_TEXT['select_b']`）

每個 T：X4a 裡倍數在 T 的 ±10% 內的 xKV 配置，平均 Δgold_lp 最高的（同分取位元組少的）＝帶內最好（最多 6 個）；另外為「疊加優於單降秩」放最多 2 個「只降秩」對照：依序看 T＝8、16、12、20、24、32，帶內最好的不是只降秩、而且有只降秩配置和它的位元組比相差 ±5% 內 → 放那裡面 Δgold_lp 最高的；加 `q:int4_kivi`（錨點）。X4a 每個模型 < 48 題 → 不挑（38f1 寫 NEEDS_HUMAN、exit 1）。計畫檔 `x4b_selection.json` 已存在而內容不同 → 停（不默默改）。

### 15.4 X4b → X4c／X4d 的挑法、無損

- 前緣（`X4_TEXT['select_cd']`）：X4b 每個 T 的生成最佳配比（帶內答對率最高的）；T*＝最佳配比在 ε 內的最大 T（沒有 → 最小的 T）。前緣 3 個＝T* 的、T* 的下一個較大 T 的（T*＝最大 → 下一個較小的）、T＝8 的（重複 → 依序 12、16、… 第一個沒重複的）。X4c＝前緣 3 個＋`q:int4_kivi`；**X4d＝前緣前 2 個**＋`q:int4_kivi`＋`drop:KEYDIFF:0.5+q:int4_kivi`（時間，§15.6）。
- 無損：38f1 用 `dump` 把 X4b 計畫裡每個 xKV 配置在 X4a 前 4 題上的**實際存下來的每一個 part**（係數的量化 q／scale／zero 或 BF16 係數、BF16 基底、遮罩）寫到本地盤，位元組＝`predict_bytes`（不等就停）。38fy（cpu 佇列）用 `m16_s1_lossless` 的函式量：每個 part zstd（−3／1／3；BF16 的 part 用 S1 的 planes 變體），每次都解回來逐位元比對；解壓速度＝全部 stream 切 4 MiB frame、1／4 個 thread。**判定：zstd:3 ≥ 1.05 倍 → 值得疊**（計畫 §3）。

### 15.5 判定（計畫 §3 原文的落實；`X4_TEXT`）

- **Q-A**：每一軸（輕度丟、量化 vs 只降秩、INT4 vs INT8、低於 4 bit vs INT4、W＝2 vs 4、K 少 V 多 vs 1:1.5）、每個 T：帶內「有這一軸的最好」−「沒有這一軸的最好」的逐題配對差（TF 的 Δgold_lp；X4b 有兩邊時也算生成）。**計畫沒寫統計門檻，這裡寫死成：95% CI 上界 < 0 → 差（不值得疊）；其他 → 沒有比較差（值得疊）**。另報「底」沿倍數的邊際代價（每多壓一倍掉多少 Δgold_lp）。合併 token 不能和低秩疊（validate 擋）、跨層逐 token 共用沒有依據（K1）——照計畫寫明。
- **Q-B**：每個 T 的 TF 最好（X4a）與生成最佳配比（X4b）並列成「目標倍數 → 最佳配比」表（`x4_best_allocation.csv`、`tables_x4.md`），附「×無損」後的倍數。
- **極限**：B*_ε＝ε 內（生成答對率對 E0 配對差 95% CI 下界 ≥ −2 個百分點）最小的位元組比；另報 B*_99、B*_99.9；每模型、每長度（32K、128K、256K）；< 48 題 → INCOMPLETE。另報「最佳配比第一個掉出 ε 的 T」。
- **疊加優於單降秩**：位元組比 ±5% 內最好的「秩＋量化」對「只降秩」的生成配對差 CI 下界 > 0 → 成立。
- **長上下文**：128K（256K）前緣配置對 E0 在 ε 內 → 撐住；超出 → 寫出從哪個倍數開始掉。
- **多輪**：第 2 輪對 E0 在 ε 內 → 多輪不壞；超出 → 「壓縮傷多輪」，並和 drop 對照比（配對差）。
- 所有比較同時報對 E0 與對 `q:int4_kivi`。

### 15.6 時間估計（全部〔估〕）與取捨

**成本模型（用實測擬合）**：每題卡秒 ≈ a·Σq ＋ 每列成本，Σq＝這題所有 SVD 的寬度（每次 SVD 的 q＝聯合秩＋16，K、V 相加）。
- 實測輸入（`/mlsteam/data/tiara/stage3/xstack/x1、x2` 逐列檔的 `ts` 與 run 目錄名的開始時間，同卡並行的 shard 合計）：
  X2 Llama core 161.6 卡秒／題（17 個配置、Σq 11,136）、x3 51.8（14、Σq 2,080）；Qwen core 68.1（17、Σq 3,760）、x3 43.9（13、Σq 1,600）、ext 47.8（13、Σq 2,096）；X1 Llama 319.2（162 個配置、Σq 22,976）、Qwen 141.7（162、Σq 11,744）。
- 擬合（core＋x3 兩點解兩個未知數）：Llama a＝0.0115 卡秒／q、生成每列 1.86 卡秒；Qwen a＝0.0069、生成每列 2.35。驗證：Qwen ext 預測 47.3、實測 47.8；代回 X1 得 TF 每列 0.24（Llama）／0.27（Qwen）卡秒。W＝2 的 SVD 每 q 當 W＝4 的 0.75 倍（矩陣寬一半，QR 不變）〔估〕；128K 的 SVD 當 32K 的 4 倍（token 4 倍）〔算術〕；128K 的 prefill／生成照 02 冒煙的實測（Llama 1 題 3 列 51.9 秒、單行程）。
- 共用 SVD 後：

| 階段 | 每題卡秒〔估〕 | 題數 | 卡小時〔估〕 | 兩卡牆鐘〔估〕 |
|:--|:--|--:|--:|--:|
| X4a | Llama SVD 50（W4 Σq 3,328、W2 1,392）＋TF 72×0.24＝67；Qwen 17.5＋19.4＝37 | 96 | 1.8＋1.0＝2.8 | 1.4 h |
| X4b | Llama SVD 27–50＋生成 10×1.86＝46–69；Qwen 10–18＋23.5＝34–41 | 192 | 2.5–3.7＋1.8–2.2＝4.3–5.9 | 2.2–3.0 h |
| X4c 128K | Llama SVD ×4 約 75＋prefill／生成約 50＝125；Qwen 約 68 | 48 | 1.7＋0.9＝2.6 | 1.3 h |
| X4d 兩輪 | Llama SVD 約 19＋生成 5×2×1.86＝40；Qwen 6＋23.5＝30 | 96 | 1.1＋0.8＝1.9 | 0.95 h |
| 冒煙、挑選、dump | 38e1 10–20 分鐘、38f1 10–20、38g1 5–15、38h1 5–10 | | | 約 0.65 h |
| **合計** | | | | **約 6.5–7.3 h** |
| Qwen 256K（可不排） | 每題約 3–4 分鐘、兩個行程 | 24 | | 約 0.7–0.9 h（含冒煙） |

- **取捨（照協調者「超過就先縮 X4a 的格點或 X4d 的配置數」）**：沒縮之前約 8.2 h。縮了兩件：
  1. **X4a 拿掉 W 大（Llama 8、Qwen 7）**：它的 SVD 是 8192／3584 維，每題約 24／15 卡秒〔估〕＝X4a 約 1 卡小時（約 0.5 h 牆鐘）；而 W 只是 Q-A 的一個軸（Q-B 的配比不含 W）。W 軸只剩「W＝2 vs 4」。
  2. **X4d 的前緣從 3 個減到 2 個**（T* 與下一個較大的 T；約 0.2 h）。
  另外原本每個 T 都放的「K 少 V 多」分法拿掉，由秩細格點的 5 組涵蓋（K／V 分配只在 15–39 倍有資料）。
- 還要更省的話（沒做）：X4b 的「只降秩」對照 2 → 1（約 0.1 h）；X4a 的輕度丟掃描只留 T＝24（約 0.05 h）；X4d 題數 96 → 72。
- 最不確定的：X4b 的 SVD（看挑到的秩與 W）、同卡混跑兩個模型 4 個行程的爭用、128K 的 SVD（沒在 128K 跑過 xKV）。38e1 的 gpu-selftest 會印整個 32K 區最大秩的 SVD 時間（`full_region_maxrank[*]` 的 `t_svd`），冒煙的 log 印每題秒數與 peak GiB。

### 15.7 怎麼排

```bash
cp code/m16_queue/jobs/38e[123]_x4a_tf.sh code/m16_queue/jobs/38f[1234]_x4b_gen.sh code/m16_queue/jobs/38g[12]_x4c_128k.sh \
   code/m16_queue/jobs/38h[12]_x4d_turns.sh /mlsteam/data/tiara/stage3/jobs/queue/
cp code/m16_queue/jobs/38ez_x4a_report.sh code/m16_queue/jobs/38fy_x4_lossless.sh code/m16_queue/jobs/38fz_x4b_report.sh \
   code/m16_queue/jobs/38hz_x4_report.sh /mlsteam/data/tiara/stage3/jobs/queue_c/
# 時間夠才排：38i1_x4c_256k_qwen.sh（queue/）＋38iz_x4_report.sh（queue_c/）
# x_common.sh、x_pool.sh、x4_common.sh 不要複製
```
- 檔名排序：`38db_c1b` < `38dz` < **`38e*` < `38ez` < `38f*` < `38fy`／`38fz` < `38g*` < `38h*` < `38hz` < `38i1`／`38iz`** < `38z_x3_map2` < `39v_x3_validate` < `40_v0`。
- GPU job 的 AFTER 只寫 `37z_x2_report`（已 done）；各階段的先後靠檔名排序（runner 一次一個 GPU job、照排序）＋**job 裡的檢查**：38f1 挑 X4b 前要 X4a 每個模型 ≥48 題、38g1／38h1 挑前緣前要 X4b ≥48 題，不夠 → NEEDS_HUMAN、exit 1；後面幾段要第 1 段的冒煙記號（`$XD/x4/SMOKE_OK_<階段>`，記程式的 sha）。不用 GPU 之間的 AFTER 是因為：前一段被 02:00Z 的截止規則跳過（`.cutoff`）時，AFTER 會讓後面永遠不開，runner 就會先去跑 39v／40_v0。
- cpu job 的 AFTER：38ez `AFTER: 38e1 38e2 38e3`；38fy `AFTER: 38f1`；38fz `AFTER: 38f1–38f4 38fy`；38hz `AFTER: 38fz 38g1 38g2 38h1 38h2`；38iz `AFTER: 38i1 38hz`。**沒排的 job 要從 AFTER 拿掉**。
- CPU 自測：`source code/m7_env.sh; OMP_NUM_THREADS=4 nice -n 19 taskset -c 0-3 python code/m16_x4.py --self-test`。

### 15.8 冒煙要看什麼（`$XD/x4/smoke/`）

- **38e1** `gpu_selftest_<m>.json`：`shared_svd_vs_exact[W*|*]`（誤差比、`shared_over_per_config`）、`two_turn_eq_m10_quality_run_turns`（兩輪的 token 逐個相同、第 1 輪＝Item.gen）、`full_region_maxrank[W*]`（**SVD 時間與 peak GiB**：Llama 若 > 50 GiB → 調 `X4_GB_LLAMA`）、`dump_bytes_eq_predict`。`tf_smoke_<m>_s0.csv` 6 列。
- **38f1**：`x4b_selection.json`（每個模型 6–9 個＋錨點；roles 有 `band_best:T`／`rank_only_ref:T`）；`x4b_smoke` 的列數；dump 的 log（每個配置幾 MB）。
- **38g1**：`x4c_selection.json`／`x4d_selection.json` 的 `detail.<m>.why_frontier`（T* 與為什麼挑這 3 個）；128K 冒煙的 peak（Llama 登記 100 GiB）。
- **38h1**：兩輪冒煙 1 題的列數＝（配置數＋1）×2。
- **38i1**：256K 的題第一次出（`make_item(n_blocks=504)`）；冒煙過了才跑。

### 15.9 輸出怎麼讀

- `tables_x4.md`（每個模型）：「目標倍數 → 最佳配比」表（TF 最好、生成最佳配比、對 E0／對 INT4、ε、MLPerf、×無損）；極限表（32K／128K／256K 的 B*_ε、B*_99、B*_99.9、第一個掉出 ε 的 T）；Q-A 表（軸 × T：值得／不值得＋配對差 CI）；無損；底的邊際代價；疊加優於單降秩；128K／256K 表；多輪表（兩輪對 E0、第 2 輪判定、對 drop 對照、對 INT4）。
- `pareto_x4a_<m>.png`：TF（倍數 vs Δgold_lp；顏色＝係數位元、形狀＝W、空心＝輕度丟；黑線＝前緣）。`pareto_x4b_<m>.png`：生成（倍數 vs 對 E0 的差與 CI；32K／128K／256K；虛線＝−ε）。
- `x4_report.json`：全部成分（每個配置的統計、Q-A 每個比較的配對差、挑選的理由、污染統計）。

### 15.10 和計畫不同的地方、已知限制

1. 共用 SVD、輕度丟一律 xl+drop（§15.2；理由：時間、共用 SVD）。
2. W 大縮掉、X4d 前緣 2 個（§15.6；理由：時間）。
3. Q-A 的「差」寫死成配對差 CI 上界 < 0（計畫沒寫統計門檻；門檻 ε、1.05 倍、±5% 都照計畫）。
4. X4d 用 C1a 題組的 T2–T4 各 32 題（C1a 有各 45 題；T5 只有一輪）。第 2 輪的歷史照 m10_quality.run_turns 固定成第 1 輪的標準答案（第二階段同一個做法），不是模型自己的第 1 輪答案。
5. 256K：第二階段沒有 256K 的題組，這裡第一次用 `make_item(n_blocks=504)`；記憶體〔估〕。
6. 128K 的位元組比用 128K 的區（block 1–251）實算：基底攤在 4 倍的 token 上，同一個配置在 128K 的倍數比 32K 高。
7. GPU 路徑全部沒跑過；冒煙是第一次。

### 15.11 需要人決定的事

1. 時間：合計約 6.5–7.3 h〔估〕，在 5–7 h 的上緣；已縮 W 大與 X4d 前緣。要再縮見 §15.6 的清單。Qwen 256K 另約 0.7–0.9 h，時間不夠就不排（報告會寫「沒有資料」）。
2. W 大要不要另開一個小批次補量（X4a 只量 W＝8／7 的幾個配置、48 題：約 0.5 卡小時〔估〕）。

---

## 16. xKV 事先校準的基底 vs 每題 SVD：加題數（10/21 第三輪；38k*）

> **狀態（2026-10-21）**：`code/m16_xlcal3.py` 與 38k 的 job 寫好，**還沒在 GPU 上跑**。`--self-test` **10 項 ALL PASS**；`m16_xstack.py` 沒有改（GPU 這段直接用它的 `gen`）；
> 用 38c 的真實逐列試跑 report（輸出在 scratchpad）：「只用 38c 的 96 題」那一欄的兩對和 v2 的 `xlcal_report.json` 逐項相同（差與 CI 都一樣）；`check-calib` 在真的校準檔上算出的 sha256＝38c 記的（兩個模型）。

### 16.1 為什麼、做什麼
- 38c（`v2/tables_xlcal.md`，96 題）兩個模型都 UNDETERMINED。使用者 10/21：加題數判出來，**只比有 INT4 的兩對**（只降秩那對不做）：
  `xl:k22:v34:cal+q:int4_kivi` vs `xl:k22:v34+q:int4_kivi`、`xl:k28:v42:cal+q:int4_kivi` vs `xl:k28:v42+q:int4_kivi`，加 E0。
- 新題：**seed 66000 起、T2–T5 各 48 題＝192 題**，32K。和 X1（60000）、校準（61000）、X2＝38c 的生成題（62000）、X2b（63000）、X4a（64000）、X4b（65000）、C1a＝X4d（70000，T2–T4 各 45、T5 15）、44c 128K／Qwen 256K（40000）都不重疊（self-test `seeds_disjoint_xlcal3`）。
- GPU：`m16_xstack.py gen`（和 38c 同一條路徑；每題 SVD 是每個配置各做一次）＋`--calib-xl`＋`--xl-cache`（這兩對的秩不同，沒有可共用的 SVD，快取等於沒作用；照指示加上）。同一題、同一個行程配對。
- 校準基底**沿用 38c 的 `calib_xl_<m>.pt`，不重算**：`check-calib` 算檔案的 sha256，要等於 38c 的摘要 `v2/calib_xl_<m>.json` 記的（Llama `9433330226f5…`、Qwen `d291ae6dc16e…`，run `20261021-123131-m16-xlcal-calib-*`），不等或不在 → NEEDS_HUMAN、停；結果寫 `v3/calib_xl_sha.json`，報告引用。

### 16.2 判定（事先寫死；`m16_xlcal3.XLCAL3_TEXT`）
- 每一對：**照 `XLCAL_TEXT['rule']` 原文**——生成答對率逐題配對差（校準 − 每題 SVD）bootstrap 10,000 次的 95% CI：上界 < −2 個百分點 → MUST_PER_ITEM；下界 ≥ −2 → CAN_PRECOMPUTE；其他 → UNDETERMINED。
- **改動（寫在報告裡）**：aggregate 從「三對」改成「這兩對」——兩對都 CAN → CAN；任一 MUST → MUST；其他 UNDETERMINED；< 48 題 → INCOMPLETE。整體：兩個模型都 CAN → CAN；任一 MUST → MUST。理由：使用者 10/21 指定只比 INT4 的兩對。門檻不變。
- 並列三種：**只用 38c 的 96 題**、**只用新的 192 題**、**合併 288 題**；主要讀合併。
- **要注意（也寫進報告）**：「只比兩對」是看到 38c 之後決定的。38c 的 96 題在兩對的 aggregate 下，Llama 就從 UNDETERMINED 變成 CAN_PRECOMPUTE（k22 +2.6 [−1.6, +7.5]、k28 +3.5 [−0.6, +8.2]；差別全在拿掉的只降秩那對 −3.7 [−7.6, +0.2]）。所以合併 288 題裡有已經看過的 96 題，**「只用新的 192 題」才是沒看過資料的獨立檢驗**；兩者不一致時照實並列，判讀由人。

### 16.3 job 與輸出
- `38k1_xlcal_more.sh`（GPU）：plan → check-calib → 冒煙（每卡一個模型、新題的第 1 題、5 列；`calib_run_id` 要帶 `xl:`）→ 兩卡共用工作池（Llama 12 份、Qwen 8 份；每卡 ≤4 行程／≤165 GiB；登記 46／30 GiB＝38c 的生成單位）。`38k2_xlcal_more.sh`：續跑（要 38k1 的冒煙記號、再核一次 sha）。`38kz_xlcal_more_report.sh`（cpu 佇列，AFTER 38k1 38k2）。共用函式在 `xlcal3_common.sh`（不要放進佇列目錄）。
- 檔名：`38j_x3_validate2` < `38k1` < `38k2` < `38kz` < `38z_x3_map2` < `39v_x3_validate` < `40_v0`。
- 輸出：`results/m16_stage3_mi300x/x_stack/v3/`（`xlcal3_selection.json`、`calib_xl_sha.json`、`xlcal3_report.json`、`tables_xlcal3.md`）；逐列 `/mlsteam/data/tiara/stage3/xstack/xlcal3/`。不寫 v2（report 擋）。

### 16.4 時間估計（〔估〕；依據＝38c 的實測）
- 38c 的生成（96 題 × 7 列：E0＋3 個校準＋3 個每題 SVD）：把兩張卡的時間依同時在跑的單位平分（`$XD/pool/38c1_xkv_cal/log`、`38c2_xkv_cal/log` 的領到／結束時間），**Llama 2.32 卡小時（87 卡秒／題）、Qwen 0.92 卡小時（34.5 卡秒／題）**；38c1 的工作池 93.9 分鐘、38c2 22.9 分鐘（只剩 Llama）。
- 換算到這次（每題 5 列；每題 SVD 只剩 k22／k34 與 k28／k42 兩個，SVD 寬度 Σq：Llama 5,024 → 4,096、Qwen 2,560 → 2,080）：用 §15.6 的生成每列成本（Llama 1.86、Qwen 2.35 卡秒）把 38c 的每題時間拆成「SVD × Σq＋生成 × 列數」，**Llama 約 70 卡秒／題 → 192 題 3.7 卡小時；Qwen 約 26 卡秒／題 → 1.4 卡小時；合計約 5.1 卡小時 → 兩卡約 2.6 小時**。
- 冒煙＋plan＋check-calib 約 5–10 分鐘。38k1＋38k2 的容量約 3.2 小時；不夠的話照抄 38k2 成 38k3 再排一段（38kz 會寫題數不齊，報告照樣出、新題 < 192 題時註明）。
