# Milestone 4 的結果檔

> 每個檔案：**是什麼、哪一條指令產生的、有效範圍是什麼。**
> 作廢的檔案在 `../superseded/`，不在這裡。

所有數字的判定材料由 `code/m4_verdict.py` 自動彙整（不手打），
它會剔除粒度過期、headroom 為負、或舊版模擬器產生的列。

## 目前有效的檔案

| 檔案 | 產生指令 | 內容 |
|---|---|---|
| `cost_model.json` | 任何 `m4_*` 腳本啟動時 | 從 M2 實測推出的成本常數，含來源檔與位置擬合上限 |
| `simulator_validation.json` | `m4_oracle.py --validate` | 模擬 vs M3 實測的比對（16K 差 8%、32K 差 3%） |
| `ssd_sweep.csv` | `m4_ssd_sweep.py` | **主結果**：SSD 容量 × headroom × 寫入可行性 |
| `by_length.csv` | `m4_by_length.py` | 節省按請求長度分箱（回答「長上下文才需要嗎」） |
| `prefix_gap_probe.csv` | `m4_prefix_probe.py` | 前綴語意為什麼幾乎不影響結果 |

## 每個檔案的有效範圍

**`ssd_sweep.csv`**
* 模型剖面 `llama-bf16`：GPU 48,128 token、KV 128 KiB/token、成本常數量自 `model_key=llama`
* 裝置 `nvme`：成本常數與 2,512 MiB/s 的持續寫入上限**同時**來自這顆碟
* 語意 `lookup=prefix`、`prefetch=on`、`oracle-dest=best`
* ⚠️ 重算成本的位置係數擬合上限是 24,576 token，而工作負載最大位置是
  126,208 token（5.1 倍外插），12.6–20.5% 的存取落在外插區

**`by_length.csv`**
* 分箱是請求長度，不是 block 位置
* `saving_pct_within_bin` 的分母是該箱的 baseline 時間，不是全部時間
* 節省率在 ~16K 之後飽和；佔比 1–2% 的 ≥64K 請求貢獻 26–31% 的總節省

**`prefix_gap_probe.csv`**
* **與裝置無關**：它統計的是 block 的常駐狀態，由容量與存取順序決定，
  與成本常數無關。所以 SATA 那次量的結果在 NVMe 下同樣成立。

## ⚠️ 已知的設定不一致（2026-08-31，待重跑）

`budget_sweep.csv` **與** `semantics_ablation.csv` 是用舊的
`m4_budget_sweep.py` / `m4_semantics_ablation.py` 產生的，
兩支都把 SSD 階**寫死成無限大**（`ssd_blocks=10**9`）——那是今天早上判定為「物理上不可能」
的設定（工作集要 10.4 TiB，NVMe 只有 3.6 TiB）。它在加入 SSD 容量軸時
沒有跟著更新。

證據：`budget_sweep.csv` 在預算 48,128 給出 7.41%，這個值對得上
`ssd_sweep.csv` 的**無限**那一列（7.41%），而不是 512 GiB 那一列（11.51%）。

合併後的 `m4_sweep.py --axis budget length semantics` 一律用
`--ssd-gib-fixed 512`（實體上放得下）。會用新版重跑，屆時本節刪除。

**在重跑之前只能引用趨勢，不能引用絕對值：**
* `budget_sweep.csv` → headroom 對 GPU 預算不敏感（16 倍範圍內 7.41%→6.89%）
* `semantics_ablation.csv` → 前綴語意的代價**落在 Oracle 身上而非 baseline**
  （toolagent：baseline 兩種語意下完全相同 58,562,989 ms，
   Oracle 由 51,968,919 變成 53,314,230，差 2.6%）。
  這與 `prefix_gap_probe.csv` 一致：缺口之後只有 0.000–0.464% 的 block
  還留在某一階，baseline 沒有東西可以損失；被罰的是 Oracle，
  因為它的逐出不是前綴感知的，會製造自己避不開的缺口。
  **淨效果是讓 Oracle 成為更保守的下界。**

## 不在這裡的東西

| 想找 | 去哪 |
|---|---|
| 原始 server log、每次 run 的完整輸出 | `/ssd7/hungwei/paper-hkv/runs/<run_id>/` |
| 逐步流水帳與每個錯誤的完整記錄 | `../RUNLOG.md` |
| 成本常數的原始量測 | `../m2_harness/` |
| 磁碟頻寬 | `../m2_harness/disk_bw.csv`、`disk_bw_sustained.csv` |
