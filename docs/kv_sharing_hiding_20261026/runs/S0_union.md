# S0：多個人問同一份文件時，挑到的塊的聯集有多大？（候選 A）

> **性質**：判準登記。登記時間見文末「登記」一節（`date -u`）。登記之後這一節以上不改；和登記不同的做法寫成文末的「偏離 N」。
> **來源**：[../04_gates.md](../04_gates.md) S0；[../03_candidates.md](../03_candidates.md) 候選 A。
> **平台**：H200（hgx-cn01）。品質型：不量時間，照讓位規則和別人同時跑；不和 ICPE 量時間的 job 重疊（skill `stage3-h200` §4）。
> **這一關回答**：兩個使用者對同一份長 context 各問一題，各自依問題挑前 m 塊。兩人挑到的塊的聯集 u(2)＝|S₁∪S₂|÷(2m) 是多少？u(2) 越小，多人共用一份縮印版、只拿聯集，就越省。

---

## 1. 設定

### 1.1 題目與模型（重用 A0，不改 A0）

| 項目 | 內容 |
|:--|:--|
| 題組 | A0 的 **gen2**（第 2 輪題）：`code/s3/a0.py plan_set("gen2", cell)`，seed0＝73000，T2–T4 各 32 題，每格 96 題（A0 登記 §1.2） |
| 題型 | m10_tasks 的合成 RULER 式題：T2–T4 是「找答案」類，needle 放在指定的位置帶；每題有**兩個問題**：第 1 輪問 b1 帶的 needle、第 2 輪問 b2 帶的 needle（同一份 context） |
| 判定格 | L128（Llama-3.1-8B-Instruct `4699cc75…`，252 塊）、Q128（Qwen2.5-7B-Instruct-1M `e28526f7…`，252 塊）、Q256（同 Qwen，504 塊）、Q14-256（Qwen2.5-14B-Instruct-1M `620fad32…`，504 塊） |
| 模型權重 | hgx-cn01 本地 `/tmp/hungweilai-models/hf-cache`（`code/s3/fetch_models.py`） |
| 不開 YaRN | CLAUDE.md §8 |

- **為什麼重用 A0 的題**：同一批題在 A0 裡有品質數字；S0 和 A0 合起來才回答「縮圖還原在多人時還划不划算」。
- **為什麼不改 A0**：A0 已登記、正在跑。S0 只 import A0 的出題、prefill 與打分函式（`code/s3/a0_gpu.py`：`Session`、`forward_hooked`、`ctx_mass`、`topm`），不寫 A0 的任何輸出。

### 1.2 兩個使用者的問題怎麼放

- 每題 prefill context 一次（fa3，和 A0 相同），ctx_end＝context 長度。
- **使用者 1**＝第 1 輪問題的 `query_ids`；**使用者 2**＝第 2 輪問題的 `query_ids`。兩個都**直接接在 context 後面**（「direct」放法，和 C1a 的 QRY1／QRY2 相同），因為兩個是獨立的使用者，不是同一個人的第 1、2 輪。
- 每個問題跑一次 forward（`forward_hooked`，sdpa），hook 在每層算 `ctx_mass`：每層、每個 q head、每個問題 token 在 [0, ctx_end) 的 key 上 softmax(q·k/√D)，加總落在區內第 j 塊（block 1…nb−1）的機率 → 每個問題一個 [L, nreg] 的分數。

### 1.3 挑法與指標

- ρ ∈ {1/16, 1/8, 1/4}；m＝round(ρ·nreg)（＝A0 的 `m_of`）。
- **全域**：分數對層加總，取前 m 塊（`topm`，平手取編號小的）→ S₁、S₂。
- **逐層**：每一層各自取前 m 塊 → S₁(ℓ)、S₂(ℓ)。
- 每題、每個 ρ 記錄：

| 欄位 | 定義 |
|:--|:--|
| `u2_global` | \|S₁∪S₂\|÷(2m) |
| `u2_layer` | 各層 \|S₁(ℓ)∪S₂(ℓ)\|÷(2m) 的平均 |
| `u2_layer_p90` | 各層 u2 的第 90 百分位（看最差的層） |
| `ov_global` | \|S₁∩S₂\|÷m |
| `rec_own_2`、`rec_cross_2`、`rec_union_2` | 第 2 輪答案所在塊（`needle_blocks`−1，區內）落在 S₂、S₁、S₁∪S₂ 的比例（全域） |
| `rec_own_1`、`rec_cross_1` | 第 1 輪答案塊落在 S₁、S₂ 的比例（全域） |

- 另外把每題兩個問題的全域分數 [nreg] 與逐層分數 [L, nreg]（float32）存成 `$RUN_DIR/scores_<cell>_<item>.npz`，供 S0-3 之後重算（不進 results）。

## 2. 判準

**主指標**：每格的 ρ＝1/4 的 `u2_global` 中位數與 `u2_layer` 中位數（96 題）。兩個粒度都要看：逐層挑（M02）的聯集在每一層各自發生。

**決定性實驗（先跑 L128）**：

| 結果 | 條件 | 下一步 |
|:--|:--|:--|
| **STOP-S（當場）** | L128 的兩個中位數**都 > 0.85** | 其他 3 格不跑；回報 |
| 繼續 | 其他情況 | 跑 Q128、Q256、Q14-256 |

**11/17 預篩**（4 格都跑完之後）：

| 結果 | 條件 |
|:--|:--|
| **PASS-S0-2**（做 S0-3） | A0 的判定是 GO，**而且** ≥2 個判定格的兩個中位數都 ≤ 0.75 |
| **STOP-S** | A0 STOP；或 0 個判定格兩個中位數都 ≤ 0.75 |
| 問人 | 恰好 1 個判定格符合 |

- 0.75 的意思：兩個人挑的塊至少有一半相同（u(2)＝0.75 ⇔ 重疊 50%）。
- 只報告、不判定：ρ＝1/16、1/8；`u2_layer_p90`；三個 recall。特別要回報「重疊是不是只來自不分問題的塊、答案塊各不相同」（`rec_cross_2` 遠小於 `rec_own_2` 時）。

**S0-3（PASS 之後才登記）**：K 最多 8；合成（一份 context 放 K 根 needle、出 K 個問題）＋真實資料（NarrativeQA、LooGLE 這類一份文件多個問題的資料集；下載前先確認授權與大小）；判準照 04 的 u(4)：≤0.60 GO、≥0.85 STOP。

## 3. 成本與排程

- 每題：一次 prefill（A0 冒煙：Q256 每題約 31 s〔實測，S3-QUEUE-1〕）＋兩次問題 forward（含 ctx_mass）。四格合計約 2–5 GPU 小時〔估計〕。
- 切成可續跑的分片（`--shard i/n`、`--budget-min`），走 runner part `kvs-s0`（TIMING＝0），照讓位規則。
- 先送 L128（決定性實驗）；看到結果、照上表決定後，再送另外 3 格。

## 4. 已經看過的數字（登記前）

- C1a（MI300X、32K、合成題、135 題）：QRY1 對 QRY2 的前 16 塊重疊 0.864（Llama）／0.767（Qwen）〔實測，`results/m16_stage3_mi300x/c1a_predictive/tables_c1a.md`〕→ u(2)＝0.57／0.62〔算術〕。這是 32K、全域、所有層平均排名的版本，**不是本關的判定數字**。
- A0 的 TF 篩選數字（L128、Q128 的 `gen_selection_*.json`）我**沒有讀**；本關的判準不依賴它們。
- 沒有看過任何 128K／256K 的 u(2)。

## 5. 指令

```bash
python code/s3/kvs_s0.py --self-test                                   # CPU：只驗集合運算與欄位
sbatch --job-name=s3-kvs-s0-L128-0 --time=4:00:00 code/r_h200_s3.sh --part kvs-s0 --cell L128 --shard 0/1
python code/s3/kvs_s0.py report                                        # CPU：讀 raw → 判定表
```

---

## 登記

- 登記時間（`date -u`）：2026-10-26T05:35:53Z
- 登記時 repo HEAD：`81e7b71`（本檔隨下一個 commit 進 git）
