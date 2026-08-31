# 作廢的結果檔

> **這個目錄裡的東西不得引用。** 留著只是為了讓 `results/RUNLOG.md` 裡
> 提到的數字仍然找得到出處。每個檔案下面都寫了「為什麼作廢」。

2026-08-31 一天之內在模擬器 `code/m4_oracle.py` 裡找到六個錯誤，
其中三個會改變結論。修正前產生的檔案**看起來完全正常**，
只是每個數字都建立在錯的前提上。

## 檔案

### `oracle.csv`（2026-08-31 15:52）
Zipf 合成工作負載的壓力掃描。**兩個原因作廢：**
1. Mooncake 的 trace 列使用了錯誤的 block 粒度（見下）
2. 壓力標籤名不副實——`pressure:8x` 只配了文件數，
   但請求數固定 400，Zipf 抽樣碰不到那麼多文件，實際壓力只有 2.8×

### `budget_sweep.csv`（2026-08-31 15:48）
GPU 預算掃描。**作廢原因：Mooncake 的 `hash_ids` 是 512-token 的 block，
被當成 16-token。** 工作集少算 32 倍，而且每個 block 的**絕對位置**也少算
32 倍——重算成本是位置的線性函數，所以「丟掉重算」這個動作被算得太便宜。

### `semantics_ablation.csv`（2026-08-31 16:31）
模擬器語意消融。粒度是對的，但 **Oracle 的成本感知目的地規則有 bug**：
它比較的是單一 block 的成本，沒有把前綴語意的「缺口之後整條尾巴都要重算」
算進去，因而做出局部便宜、全域昂貴的決策。
`prefix/prefetch` 那兩列的 headroom 是**負的**（Oracle 輸給 baseline），
在定義上不可能。

### `oracle.csv.mislabeled-pressure`
更早一版，同樣的壓力標籤問題。留在
`/ssd7/hungwei/paper-hkv/runs/superseded/`。

## 怎麼避免再發生

`code/m4_verdict.py` 有三層自動偵測，會在產生判定材料時剔除這類資料：

1. `unique_blocks` 與現行 trace 解碼不符 → 粒度過期
2. `headroom < 0` → Oracle 輸給 baseline，在定義上不可能
3. `sim_version`（`m4_oracle.py` 內容的 SHA-1 前 8 碼）與現行不符 → 舊版模擬器

第 3 項是 2026-08-31 之後才加的，所以更早的檔案沒有這個欄位。
