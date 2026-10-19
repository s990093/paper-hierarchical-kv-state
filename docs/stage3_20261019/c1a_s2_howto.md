# C1a、S2 怎麼跑（第三階段；10/19 寫）

> **狀態**：兩支程式**還沒有執行過任何一行**。寫的時候第二階段正在兩張 MI300X 上計時，不能 import torch。所以只做了 `python -m py_compile`，以及 ast／symtable 的靜態檢查：頂層沒有副作用、沒有未定義的名稱、用到的 `m10_*` 屬性都存在。
> 本頁的時間全部是〔估計〕，依據寫在 §5。沒量到的一律 NOT_MEASURED。
> 計畫：`experiment_matrix.md` §3 的 C1a、S2。設計與停止條件照 `lit/L1_eviction_selective_loading.md` §4 G1／G2／G4、§5 X1、X2（＋X4），**沒有改**，原文抄在 §7。

---

## 0. 檔案

| 檔案 | 子命令 | 輸出 |
|:--|:--|:--|
| `code/m16_c1a_predict.py` | `run`（`--smoke`）、`selfcheck`、`report` | `results/m16_stage3_mi300x/c1a_predictive/`：逐列 CSV（每題 × 每個分數一列）、`*.checks.jsonl`、`c1a_summary.json`、`tables_c1a.md` |
| `code/m16_s2_equalbytes.py` | `run`（`--smoke`）、`report` | `results/m16_stage3_mi300x/s2_equal_bytes/`：逐列 CSV（每題 × 每個配置一列）、`*.checks.jsonl`、`s2_report.json`、`tables_s2.md` |

- 兩支都有 `run_id`、`ts`、`shard` 欄。也都有 `n_rows_item` 欄，續跑時用它判斷「一題寫完沒」，寫到一半的題會整題重算，`report` 會去重。
- 每列另記 `own_gpu_intruders`、`host_level`。
- 既有檔案一個都沒改，只用 import 重用。唯一照抄的是 `m10_drop._window_probs`，抄成 `m16_c1a_predict.probe`，只多兩件事：只加總「尾段」query 的機率；留下 post-RoPE q。`selfcheck` 會逐位元比對它和 `m10_drop` 的 QRY。

### 重用的函式（檔案:行號，10/19 HEAD）

| 來源 | 用在哪 |
|:--|:--|
| `m10_drop.py:55` `Ctx`、`:254` `scores`（→ `:127` SNAP、`:144` H2O、`:162` QRY、`:169` KNORM、`:175` KEYDIFF、`:183` EXPATT）、`:325` `unit_order`、`:119` `_pool_group`、`:350` `keep_mask`、`:42–47` `WRITE_TIME`／`METHODS`／`SCORED` | 兩支的分數與 block 順序；S2 的遮罩逐位元核對 |
| `m10_drop.py:70` `_window_probs` | 抄成 `probe`（見上） |
| `m10_quality.py:50` `Answerer`、`:137` `plan_items`、`:313` `ItemState`、`:513` `_watch` | 模型、出題順序、C1a 的 prefill、GPU 監控 |
| `m10_sweep.py:108` `Item`（`score`、`metrics`、`load_fmt`、`place`） | S2 的 prefill、teacher forcing、度量、量化 block 的放置（和 DS1／09 海選同一段） |
| `m10_tasks.py:722` `make_item`、`:817` `_Locator` | 出題；答案 block 的獨立驗算 |
| `m10_codec_x.py:233` `encode` | S2 每個格式的實際位元組 |

---

## 1. C1a 的定義（寫死）

- **題目**：DS2 的兩輪題（14 號 §8.5）。seed 70000 起；T2–T4 各 45 題，三種（第 1 輪帶, 第 2 輪帶）輪流；T5 15 題。參數用 `CANDIDATE_PARAMS.json`；32K。
  - 每題 prefill 一次，用 `ItemState`，和 DS1／DS2 同一條 chunk 化 prefill。只打分、不生成。
- **保留順序**：block 1–63 的排名。block 0 一律保留，不參加排名，所以隨機期望＝k/63。
  - **寫入時**：SNAP（無問題版）、EXPATT、KNORM、KEYDIFF、H2O 近似，用 `m10_drop.scores`；STREAM、NEW、RND0–2 照位置。
  - token 分數換成 block 順序一律用 `unit_order(…, "B", …)`，就是 14 §8.3 的百分位排名平均，反過來就是保留順序。
  - **QRY1／QRY2**：第 1 輪／第 2 輪問題，都用 `m10_drop` 的 QRY 定義。
  - **第 2 輪問題放哪**（`--q2-placement`）：
    - `direct`（預設）：`m10_drop` 的定義原樣，問題直接接在 context 後面。
    - `insitu`：實際的第 2 輪序列（context＋第 1 輪問題＋歷史＋第 2 輪問題），只取第 2 輪問題那幾個 token 的 attention。
    - 另一種放法一律另記一列 `QRY2_ALT`，它對 QRY2 的 ρ 就是「放法本身影響多大」。
  - **Quest 式摘要**：
    - 摘要＝每個 block、每個（層, KV head）的 post-RoPE K 逐維 min／max。
    - 目標問題每個 token、每層、每個 q head 的 post-RoPE q（GQA：q head h → KV head h // rep）。
    - 上界 U[b] = Σ_d max(q_d·min_d, q_d·max_d) / √D。
    - 彙總：
      1. 每個（層, q head, token）在 64 個 block 上做 softmax；
      2. 對 token 平均；
      3. 略過前 2 層（`--quest-skip-layers`，照 Quest 原文前兩層不套用）；
      4. 對（層, q head）取平均＝`QUEST_MEAN`、取最大＝`QUEST_MAX`。
  - **FULLK_MEAN／MAX**：同一個 forward、同樣的彙總，只是用真的 attention 機率的 block 總和。和 QUEST_* 只差「K 換成 min／max」這一件事，用來把 X4 的效應單獨拿出來。
- **目標**：兩輪題＝第 2 輪的 QRY2；T5（單輪）＝它唯一一輪的 QRY1。
- **答案所在的 block**：`m10_tasks` 組裝時記的 needle token 位置（`spans`）→ pos // 512。用兩條獨立路徑驗算（規則 6），任何一條不過就停（規則 2）：
  1. 換算結果必須等於 `needle_blocks`，且都在 1..63；
  2. `_Locator` 把 context 解碼成文字找答案字串：每個答案至少出現一次，而且每次都落在答案 block 裡；
  3. 答案 block 的每一段連續 block 直接 `tok.decode`，答案字串要在其中一段裡。
  - T5（全域）不驗算、不報 recall。
- **指標**（每題 × 每個分數一列）：
  - `recall{16,32}`（答案 block 落在前 k 的比例）；
  - `hit_all`／`hit_any`，以及它們在隨機排列下的精確期望（超幾何）；
  - `spearman_vs_target`；
  - `overlap{16,32}_vs_target`；
  - `recall_t1_*`：對第 1 輪答案的 recall，只當健全性檢查，QRY1 在這裡應該很高。
- **report**：
  - 以題目為單位 bootstrap 10,000 次。
  - 「≈隨機」＝95% CI 包含隨機期望；另附對 RND0–2 平均的配對差。
  - 列出 L1 G2／G4 的每個成分（`stop_rule_components`），**判讀由人**。

## 2. S2 的定義（寫死）

- **題目與度量**：和 DS1 同一套。seed 60000 起，T2–T5 各 24 題。用 `m10_sweep.Item` teacher forcing 第 1 輪，度量是 KL、gold_lp、top1、first_ok。QRY＝第 1 輪問題。
- **只動載入區 block 32–63**：block 0 一律 BF16；block 1–31 照 E0（BF16、不遮）。`--full-range` 另做 block 1–63 的全段版（補充）。
- **刪除組**：
  - 遮罩（`key_keep`，＝從 cache 拿掉，RoPE 位置不變）。
  - 順序：DS1 的 `unit_order`（全段 1–63 算的），只留區內 block、保持原順序，取前 k 個。**不是**只在區內重新排名，見 §8 假設 4。
  - 方法：`--write-methods`（預設 SNAP KEYDIFF；DS1 pick 出來之後改用 `--ds2-plan` 讀 B 粒度 top2）、QRY、RND0；ORC 只做 50%。
  - k 有兩種：
    - **名目**（L1 X1 的表）：區內丟 50／75／87.5% → 16／24／28 個 block（全段 32／47／55）。
    - **照實等位元組**（`--no-matched` 關掉）：k＝round(區內 block 數 ×（1 − 量化格式的實際位元組比）)。
- **低精度組**：
  - 格式：FP8（照 `fmt_choice.json`：Llama `fp8_token`、Qwen `fp8_head`）、`int8_token`、`int4_kivi`、`int2_kivi`、`kv-int4_kivi-int2_kivi`（K4V2）。
  - attention 路徑兩種都量：`mask`（全留的遮罩，和刪除組同一條路徑，**配對比較用這個**）、`sdpa`（和第二階段的量化數字同一條路徑）。
- **位元組照實算**：
  - 量化＝每個 block 的 `Enc.nbytes`（含 scale／zero）÷ BF16 的 chunk_bytes；
  - 刪除＝留下的 block × BF16（遮罩本身不算位元組）；
  - 每列記 `ratio_region`、`ratio_total`、`bits_per_value_region`。
- **report**：
  - 每個配置對 E0 的值；
  - 配對差（刪除 − 量化，同一題）：名目配對＝L1 的表，照實等位元組配對＝補充；
  - 「最好的寫入時刪除法」標 `post_hoc_selection`；
  - L1 G1 的成分（只用載入區、名目配對）；
  - MASK0 − E0（換路徑的雜訊）。

### 實際位元組比〔算術，照 m10_codec／m10_codec_x 的定義與自測的公式；smoke 會印實際值核對〕

| 格式 | 每個值的位元 | 對 BF16 | 名目 | 區內照實等位元組的 k（32 個 block） |
|:--|--:|--:|:--|--:|
| fp8_token（Llama 的 FP8） | 8.25 | 0.5156 | 1/2 | 16 |
| fp8_head（Qwen 的 FP8；scale 896 B／block） | ≈8.0005 | ≈0.50003 | 1/2 | 16 |
| int8_token | 8.25 | 0.5156 | 1/2 | 16 |
| int4_kivi | 5 | 0.3125 | 1/4 | 22 |
| kv-int4_kivi-int2_kivi（K4V2） | 4 | 0.25 | 1/8 | 24 |
| int2_kivi | 3 | 0.1875 | 1/8 | 26 |

- **名目的 1/4、1/8 不是等位元組**：
  - INT4 KIVI 實際是 0.3125，名目配對是丟 75%（0.25）；
  - INT2 KIVI 是 0.1875、K4V2 是 0.25，名目配對都是丟 87.5%（0.125）。
  - 所以名目配對裡量化多拿了 25–100% 的位元組；照實等位元組的配對在 report 的 `matched`。
- **Qwen 的 FP8 是 `fp8_head`**：`fmt_choice.json` 規則選出的，但它在選格式小測的平均 KL 是 3.40（Llama `fp8_token` 0.0024），可能是很差的 FP8。1/2 預算另有 INT8 可比。

---

## 3. GPU 窗口：先跑的檢查（照順序；任何一個不過 → 停、回報，規則 2）

```bash
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m7_env.sh     # venv、TIARA_RUNS、m7run；卡＝$TIARA_GPU（預設 0）
REPO=/mlsteam/workspace/paper-hierarchical-kv-state
C=$REPO/results/m16_stage3_mi300x/c1a_predictive
S=$REPO/results/m16_stage3_mi300x/s2_equal_bytes
SM=/mlsteam/data/tiara/stage3/smoke_c1a_s2; mkdir -p $SM                # 冒煙的輸出不進 results
```

**3.1 C1a selfcheck**：`m10_drop` 的分數在新程式裡算出來，和 `m10_drop screen` 對同一題逐值相同。每個模型約 1–2 分鐘〔估計〕。

```bash
for m in llama qwen; do
  m7run m16-c1a-selfcheck-$m python $REPO/code/m16_c1a_predict.py selfcheck --model $m --out $SM/c1a_selfcheck_$m.json
done
```

- 題目：DS1 的第一題 `T2_front-back_s60000`。
- 路徑 A：照 `m10_drop.cmd_screen` 第 418–432 行，`SW.Item` → `DR.Ctx(…, st.needle, 第 1 輪 query_ids)` → `DR.scores`。
- 路徑 B：本程式的 `compute_item`，重新 prefill，目標輪設成第 1 輪。
- 要全部 PASS：
  - `prefill_bitwise`：64 個 BF16 block 逐位元相同；
  - `score[SNAP|EXPATT|KNORM|KEYDIFF|H2O|QRY]` 逐值相同（`torch.equal`）；
  - `order[*]` 的 block 順序相同；
  - `probe(tail_from=0)` 和 `m10_drop` 的 QRY 逐位元相同；
  - Quest／FULLK 沒有 NaN；
  - `two_turn_path`：第 2 輪目標的路徑跑得完。
- `prefill_bitwise` 不過：代表 GPU 上重新 prefill 不是決定性的，後面的逐值比對也會跟著不過。這時看 JSON 裡的 `maxabs` 決定能不能接受。程式沒錯也可能這樣。

**3.2 C1a smoke**：1 題、印每個分數前 5 名 block 與答案 block，給人看。

```bash
for m in llama qwen; do
  m7run m16-c1a-smoke-$m python $REPO/code/m16_c1a_predict.py run --model $m --smoke --out $SM/c1a_smoke_$m.csv
done
```

題目是 `T2_front-back_s70000`：第 1 輪答案在前段（block 1–16），第 2 輪在後段（48–63）。要看：
- 第 1、2 輪答案 block 的兩條路徑（spans、解碼搜尋）都印出來、`驗算：OK`；
- STREAM 前 5＝`63 62 61 60 59`，NEW 前 5＝`1 2 3 4 5`；
- QRY1 的 `recall_t1@16` 應該很高，QRY2 的 `recall@16` 應該很高。這兩個是 QRY 實作的健全性檢查，不是結果；
- `檢查：` 裡 `probe_vs_drop_qry_bitwise`、`quest_finite`、`fullk_finite` 都是 true；
- log 的 `peak … GiB` 和每題秒數：決定 §4 一次開幾個 shard。

**3.3 S2 smoke＋DS1 重現檢查**：1 題（DS1 第一題）、全段版、印每個配置。

```bash
for m in llama qwen; do
  m7run m16-s2-smoke-$m python $REPO/code/m16_s2_equalbytes.py run --model $m --smoke --out $SM/s2_smoke_$m.csv
done
```

- `遮罩 vs m10_drop.keep_mask`：全段丟 50%／75% 的每個方法（SNAP、KEYDIFF、QRY、RND0、ORC），我們的遮罩和 `m10_drop.keep_mask` 必須 `equal: true`。不是的話是程式錯誤，smoke 會以 1 結束。
- **DS1 重現**：E0、MASK0、全段丟 50%／75% 的 kl、gold_lp、top1、first_ok、kept_frac、needle_kept，對 DS1 的逐列檔（`/mlsteam/data/tiara/drop_m10/ds1_<m>_s*.csv.gz`）逐值比對。
  - 預期全部逐值相同（同樣的程式、同樣的題）。
  - 有差：看最大絕對差。很小（例如 gold_lp ≤ 1e-3）＝GPU 不決定性，記下來；大＝停。
  - DS1 還沒跑到這一題（或還沒跑）時印「比了 0 格」，等 DS1 跑完再補做。
- `位元組／block`：要等於 §2 表的算術值。Llama BF16 一個 block＝67,108,864 B，int4_kivi＝20,971,520 B。

**3.4（可選）** 兩個 smoke 的 `.checks.jsonl` 留著當紀錄。

---

## 4. 正式執行

- 第三階段 runner（`code/m16_queue/runner.sh`）一次只跑一個 job，所以一個 job 裡用 `&`／`wait` 開幾個 shard。
- job 檔頭照 runner 的格式。每個 job ≤ 2 小時（`experiment_matrix.md` §6）。
- 下面的 shard 數是〔估計〕：照 09 海選 Llama 每個 process 峰值 28.3 GiB（run `20261016-130624-m10-sweep-pilot-llama-s0`）推的。smoke 量到峰值之後再定。

**C1a**（每個模型 150 題；兩個模型可以分兩個 job、兩張卡）：

```bash
# GPU-NEED: one
# GPU: 0
# NEED-GB: 120
# MAX-H: 2
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m7_env.sh
REPO=/mlsteam/workspace/paper-hierarchical-kv-state; C=$REPO/results/m16_stage3_mi300x/c1a_predictive; m=llama; N=3
for s in $(seq 0 $((N-1))); do
  m7run m16-c1a-$m-s$s python $REPO/code/m16_c1a_predict.py run --model $m --shard $s/$N --out $C/c1a_${m}_s$s.csv &
done
rc=0; for p in $(jobs -p); do wait $p || rc=1; done; exit $rc
```

做完兩個模型之後（CPU，只用 numpy，不碰 torch）：

```bash
python $REPO/code/m16_c1a_predict.py report --raw "$C/c1a_*_s*.csv" --out-dir $C
```

**S2**（每個模型 96 題）：

```bash
# GPU-NEED: one
# GPU: 1
# NEED-GB: 120
# MAX-H: 2
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m7_env.sh
REPO=/mlsteam/workspace/paper-hierarchical-kv-state; S=$REPO/results/m16_stage3_mi300x/s2_equal_bytes; m=llama; N=4
PLAN=$REPO/results/m10_drop_mi300x/ds2/ds2_plan.json; PL=""; [ -f $PLAN ] && PL="--ds2-plan $PLAN"   # DS1 pick 好了就用它的 top2
for s in $(seq 0 $((N-1))); do
  m7run m16-s2-$m-s$s python $REPO/code/m16_s2_equalbytes.py run --model $m --shard $s/$N --full-range $PL \
    --check-ds1 "/mlsteam/data/tiara/drop_m10/ds1_${m}_s*.csv.gz" --out $S/s2_${m}_s$s.csv &
done
rc=0; for p in $(jobs -p); do wait $p || rc=1; done; exit $rc
```

```bash
python $REPO/code/m16_s2_equalbytes.py report --raw "$S/s2_*_s*.csv" --out-dir $S
```

- **寫入時方法的來源**：DS1 還沒跑完時用預設 SNAP、KEYDIFF。每列的 `write_methods_source` 會記下來；report 也會列出。
  - **不要**混用兩種來源的 shard。DS1 pick 出來之後，若 top2 不是 SNAP、KEYDIFF，S2 要用 `--ds2-plan` 重跑。
- **檔案大小**：
  - S2 開全段版約每題 68 列。96 題 × 約 450 B ≈ 每個模型 3 MB〔估計〕，分 4 個 shard 每個 < 1 MB。不夠小就把 `--out` 改成 `.csv.gz`，程式支援。
  - C1a 每題 17 列（T5 15 列），每個模型約 1.5 MB〔估計〕，分 3 個 shard。
- **續跑**：同一個指令重下即可，寫完的題會跳過。

---

## 5. 時間〔估計〕與依據

| 項目 | 〔估計〕 | 依據 |
|:--|:--|:--|
| C1a 每題 | 20–40 s | 32K chunk 化 prefill 中位數：Llama 8.0 s、Qwen 7.0 s（實測，`results/m10_pilot_mi300x/e0_tune/p1_default_{llama,qwen}.csv` 的 `t_prefill`，run `20261015-030858`／`030854`）；EXPATT 會把 context 再跑一次（≈ 一次 prefill，加上 float64 的共變異數）；其他分數加探針：14 號 §8.6 估全部分數「約 15 秒」（那也是估計）、L1 X2 估每題 10–20 s |
| C1a 全部 | 每個模型 150 題 × 20–40 s ≈ 50–100 分鐘（單一 process）；3 個 shard 一起跑約 0.5–1 小時 | 同上；多個 process 搶同一張卡時每題會變慢，所以不是除以 3 |
| S2 每題（只有載入區） | 25–35 s | prefill＋分數同上；teacher forcing 約 32 次。09 海選實測 110–128 ms／forward（Llama）、106–113（Qwen），run `20261016-130624`／`130854-m10-sweep-pilot-*-s0` 的 stdout，含放 block 的時間；那時同一張卡上可能還有別的 shard，沒確認。另加 5 個格式的 `load_fmt` |
| S2 每題（加全段版） | 40–50 s | 再約 35 次 teacher forcing |
| S2 全部 | 每個模型 96 題 × 40–50 s ≈ 65–80 分鐘（單一 process）；4 個 shard 約 0.5–1 小時 | 同上 |
| 檢查（§3） | selfcheck 每個模型 1–2 分鐘；smoke 每個 1–2 分鐘（含載模型） | 同上 |

---

## 6. 輸出怎麼讀

- **C1a**：`tables_c1a.md` 每個模型、每組任務（T2–T4 合併、各自、T5）一張表，欄位：
  - recall@k − k/63、ρ、重疊@k − k/63（百分點，95% CI）；
  - recall_t1@16。
  - X4 一行：Quest 和完整 K 的前 k 重疊；「完整 K − Quest」對目標的重疊差。
  - `stop_rule_components`：G2（用 QUEST_MEAN／QUEST_MAX 各算一次）、G4 的每個成分。
  - `c1a_summary.json` 另有依第 2 輪位置帶分開的 recall（`by_band2`）。
- **S2**：`tables_s2.md` 三個部分：
  1. 每個配置對 E0 的值，附實際位元組比、預算標籤；
  2. 配對差（刪除 − 量化）：名目、照實等位元組；
  3. G1 的成分：每個預算、每個量化格式、每個方法的 `not_win`；`dominated_by_quant`；`all_budgets_dominated`。
- 污染：
  - `report` 會數 `own_gpu_intruders ≠ 0` 的列、`NOT_CHECKED` 的列。
  - 品質規則：分數照用，`t_score_s` 作廢。
  - run 目錄有 `CONTAMINATED` 檔的 run 照樣保留分數。

---

## 7. L1／矩陣的停止條件原文（沒有改）

**G1（S2；L1 §4）什麼時候停止投入**：
- 三個位元組預算（約 1/2、1/4、1/8）下，最好的寫入時刪除法（DS1 選出的前 2 名）都沒有贏過同位元組的均勻低精度：配對差的 CI 上界 ≤ 0，或贏不到 ε。
- 這時寫「在本設定下，有損刪除被量化支配」，第三階段不再做刪除的放置策略。
- 只有在 ORC／QRY 以外的方法贏過量化時才繼續。

**G2（C1a；L1 §4）什麼時候停止投入**：
- 寫入時的分數（SNAP、EXPATT、KNORM、KEYDIFF、H2O 近似）對第 2 輪答案所在 block 的召回率，和隨機的 CI 重疊；**而且**
- Quest min/max 摘要在 512-token block 上對 QRY 前 k 名的召回也≈隨機。
- 兩者都成立 → 還原時能用的只剩位置（就是現在的 Cake 規則），G2 停止。

**G4（C1a；L1 §4）什麼時候停止投入**：
- 寫入時的分數預測第 2 輪的能力≈隨機（同 G2）；**而且**
- 第 1 輪問題的 QRY 對第 2 輪也≈隨機。
- 這表示在我們的任務上，跨輪的重要度不可預測，唯一的策略是無損保留＋問題來了再選（RocketKV-MT 式）。

**X1（L1 §5）**：停止條件：見 G1。
**X2（＋X4）（L1 §5）**：停止條件：見 G2、G4。H-X2、H-X4 都成立 → 選擇性還原只能靠位置與下一輪的問題；第三階段不做「寫入時分數」的放置。
**C1a（experiment_matrix §3）停止**：寫入時分數≈隨機、且摘要也≈隨機 → 選擇性還原只能靠位置與下一輪的問題，C1b 不做（寫成負面結果）。
**S2（experiment_matrix §3）成功／停止**：若 INT4 均勻在每個預算都支配丟棄 → 丟棄只在 < 25% 的預算才值得討論；反之則 drop 進入第三階段的動作空間。

`report` 只把這些條件拆成成分算出來，不替人下結論。程式對原文的讀法寫在 §8 的 7–9，人可以照 JSON 裡的原始 CI 重新判。

---

## 8. 還沒驗證的假設

1. **GPU 上重新 prefill 逐位元相同**。selfcheck 的逐值比對、S2 的 DS1 重現都靠這個；第二階段 `selfcheck` 的 cake_vs_replay 在 MI300X 過過，但這兩支沒跑過。
2. **QRY2 預設用 `direct`**（`m10_drop` 的定義：第 2 輪問題直接接在 context 後面），不是實際的第 2 輪序列。兩種都算，`QRY2_ALT` 的 ρ 告訴我們差多少；差很多的話要決定換 `--q2-placement insitu` 重跑。
3. **Quest 的彙總是我們定的**：對 block 做 softmax、對問題 token 平均、略過前 2 層、對（層, q head）取平均或最大。Quest 原文是每個 decode token、每個 head 各自挑 page，而且 page 是 16 token，我們是 512-token block（L1 X4 的設計）。
4. **S2 的區內順序**＝全段算的 `unit_order` 只留區內 block，不是在區內重新做百分位排名。兩者在每個（層, head）上保序，但對（層, head）平均之後可能不同。
5. **答案 block＝needle 的每個 token 所在的 block**（T4 的 gold 段落可能只有結尾的換行落在某個 block）。recall 用這個定義，和 ORC／DS 相同。
6. **T5 的目標**＝它唯一一輪的問題（QRY1）；T5 只報 ρ 和重疊。
7. **G1 的 ε**：第二階段的 ε 是答對率 −2 個百分點；這裡套在 first_ok 上（`EPS_FIRST_OK = 0.02`）。「贏不到 ε」讀成「first_ok 配對差的平均 < 0.02」。gold_lp 沒有 ε，只看「CI 上界 ≤ 0」。
8. **「被量化支配」的讀法**：某個預算裡至少有一個量化格式，兩個寫入時方法都沒贏它；每個預算都這樣 → `all_budgets_dominated`。L1 的「FP8 或 INT8」「INT2 KIVI 或 K4V2」沒說是哪一個。
9. **G1 用名目配對**（L1 X1：「用 BF16 的比例對齊位元組，忽略 scale 的額外位元組，報告時照實列出」）。照實等位元組的配對只當補充，不進判讀。
10. **配對比較用 mask 路徑的量化**（和刪除同一條 attention 路徑）；sdpa 路徑的量化另外列，和第二階段的數字對照。
11. **每個 block 的位元組只由形狀決定**（`Enc.nbytes`）。程式用 block 1 算一次，`load_fmt` 再核對一次。
12. **記憶體**：每個 process 峰值〔估計〕≤ 40 GiB，NOT_MEASURED；smoke 的 log 會印 `peak`。
13. **GPU 監控**：`m10_quality._watch` 用 `smi_index(0)` 對 `HIP_VISIBLE_DEVICES`；在 AMD 上 `host_contention` 回 `UNOBSERVABLE_HOST`（看不到主機其他負載，規則 7），這會照實寫進 `host_level`。
14. **DS1 的逐列檔位置**＝`/mlsteam/data/tiara/drop_m10/ds1_<m>_s*.csv.gz`（44e1 的 job 寫的），而且欄位沒變。
15. **時間**全部是 §5 的估計。

## 9. 已知限制（設計上的，不是 bug）

- C1a 只打分，不生成。「分數有沒有預測力」用答案 block 的 recall 和對目標 QRY 的排名來看，不是答對率。
- S2 只量讀法 A（全部載入、不重算）的品質，和 14 §4.1 相同。是下限。
- S2 的刪除只在第 1 輪（DS1 的 teacher forcing）；多輪的「第 2 輪才發現被丟」是 C1b 的事。
