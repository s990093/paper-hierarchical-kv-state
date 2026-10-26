# T0：共享者多久來一個？（Mooncake trace 統計，CPU，只報告）

> **性質**：判準登記。登記時間見文末「登記」一節（`date -u`）。登記之後這一節以上不改；和登記不同的做法寫成文末的「偏離 N」。
> **來源**：[../04_gates.md](../04_gates.md) T0；[../03_candidates.md](../03_candidates.md) 候選 B、C。
> **這一關回答**：一段共享前綴正在還原的那幾秒內，平均有幾個人在等？同一段前綴的請求，相鄰多久到一次？這是 F0 的輸入，**本身不判 GO／STOP**；只有單位檢查會叫停。

---

## 1. 資料

| 項目 | 內容 |
|:--|:--|
| 來源 | Mooncake FAST'25 trace，`kvcache-ai/Mooncake` commit `245e7106`（RUNLOG_H200 D4：run `20261016-110413-h200-dl-mooncake-traces`） |
| 檔案 | `$TIARA_DATASETS/traces/{conversation,toolagent,synthetic}_trace.jsonl`；列數 12,031／23,608／3,993（同上） |
| 欄位 | `timestamp`、`input_length`、`output_length`、`hash_ids` |
| 單位 | `hash_ids` 一個＝512 token（`load_trace` 斷言 `len(hash_ids)==ceil(input_length/512)` 的吻合率 ≥0.95，並驗前綴鏈）。`timestamp` 當毫秒，**要用 §3 的單位檢查確認** |
| 模型常數（算還原時間 W） | Llama-3.1-8B 每 512 token 64 MiB；頻寬見 §2.3 |

## 2. 量什麼

### 2.1 定義

- **系統 prompt 塊數** sys：最大的 d，使得 ≥50% 的請求前 d 個 hash 完全相同（每份 trace 各算一次）。
- **共享深度**：兩個請求的共同前綴塊數（前綴鏈已驗證，所以「第 k 個 hash 相同」等於「前 k+1 塊都相同」）。
- **超過系統 prompt 的共享**：共享深度 − sys。
- 門檻 S ∈ {2K, 6K, 16K} token → d_S ＝ 4、12、32 塊。
- **群**：sys＋d_S 位置的 hash 相同的請求（＝超過系統 prompt 至少共享 d_S 塊）。

### 2.2 指標

| 編號 | 指標 | 切法 |
|:--|:--|:--|
| T0-a | 有另一個請求在 ±w 秒內到、而且超過系統 prompt 共享 ≥d_S 塊的請求比例 | 3 個 trace × 3 個 S × w ∈ {0.1, 0.5, 1, 2, 5, 10, 60} |
| T0-a′ | 同上，但**不扣**系統 prompt（原始共享深度 ≥ d_S） | 同上 |
| T0-b | 群內相鄰到達間隔的分布（p10、p25、p50、p75、p90、p99）與群大小分布 | 3 × 3 |
| T0-c | 一次還原進行中的等待人數 N_W：群內「前 W 秒沒有同群請求」的請求當作領頭（冷啟動），數它到達後 W 秒內同群的請求數；報 N_W 的平均、p50、p90、最大，以及「會變成跟隨者」的請求比例 | 3 × 3 × 平台 × tier |
| T0-d | 每個群的到達率 λ 與 N＝λ·W 的分布 | 同 T0-c |

### 2.3 還原時間 W（T0-c、T0-d 用）

- W＝領頭請求的全部塊數 × 64 MiB ÷ 頻寬（只算載入、不算重算；Llama-3.1-8B）。
- 頻寬：

| 平台 | CPU | 本地 NVMe | NFS 級（限速） |
|:--|:--|:--|:--|
| H200 | `results/m10_precision_h200/speed_llama/bw.json` 的 `real_pinned_h2d_GiBps` | `results/s3_h200/gfio/gfio_20261026-104426-s3-gfio.csv`：`/tmp`、seq、64 MiB、8 threads 那一列的 `gibps` | `results/m7_write_policy_mi300x/tier_params.json` 的 `nfs.read_GiBps` |
| MI300X | `results/m10_precision_mi300x/speed_llama/bw.json` 的 `real_pinned_h2d_GiBps` | `tier_params.json` 的 `local.read_GiBps` | 同上 |

### 2.4 輸出

- `results/s3_h200/kvs_t0/t0_summary.csv`（每列：run_id、ts、trace、metric、S、w 或平台／tier、值、單位）。
- F0 用的群到達清單（每群的到達時間與領頭的塊數，只收 ≥2 個請求的群）：`$TIARA_RUNS/<run_id>/t0_groups_<trace>_S<S>.json.gz`，路徑記在 summary 裡。

## 3. 單位檢查（會叫停的唯一一條）

Strata（OSDI'26 版 §3.2）報告：toolagent trace 中 38% 的請求在 1 秒窗口內，和至少一個其他請求共享 ≥6K token 前綴（不計 system prompt）。

- **檢查**：toolagent、S＝6K、w＝1 s 時，T0-a 或 T0-a′ 至少一個落在 [19%, 76%]（38% 的 0.5–2 倍）。
- **不過** → 停下：timestamp 可能不是毫秒，或 hash 單位有誤（CLAUDE.md §1 規則 6）。先查單位再跑任何下游（F0 的 trace 版）。
- 「w 是 ±1 秒還是之後 1 秒」原文沒寫清楚（L1 §1.4），所以用 ±w 並把「只看之後 w 秒」另記一欄，兩欄都報。

## 4. 已經看過的數字（登記前）

- Strata 的 38%（上面）。
- D4 的 trace 特徵（`results/m8_directions_h200/d4_reuse_dist.csv`）：重用距離、2-hit 間隔等。**和這裡的「群內相鄰間隔」不是同一個量**（D4 是同一 block 的兩次存取，不分群、不扣系統 prompt）。
- 沒有看過任何 T0-a～T0-d 的數字。

## 5. 指令

```bash
source code/env_h200.sh          # head node；numpy 在 conda
python code/s3/kvs_t0.py run --traces conversation,toolagent,synthetic
python code/s3/kvs_t0.py --self-test   # 合成 trace，只驗邏輯
```

---

## 登記

- 登記時間（`date -u`）：2026-10-26T05:35:53Z
- 登記時 repo HEAD：`81e7b71`（本檔隨下一個 commit 進 git）
