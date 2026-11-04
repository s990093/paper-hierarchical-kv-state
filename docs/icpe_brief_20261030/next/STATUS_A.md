# STATUS_A（`h200` session，H200 head node，分支 `icpe-next-a`）

> 只有 `h200` session 寫本檔。格式照 NEXT_PLAN §3b：每次更新加一節。內容以 git 上的版本為準。

## 2026-11-04 23:20（台灣）

**做完什麼**
- **PREREG 已登記**：[`PREREG.md`](PREREG.md)（§6b 八項＋補問的 A–D＋W6 都經使用者逐項確認）。**本檔所在的 commit 推上 `origin/icpe-next-a` 之後，正式結果可以開跑**；在那之前只准試跑。
- W8 先核了影響 §6b 的數字：[`W8_source_check.md`](W8_source_check.md) §1。
- 給 `icpe-sim`／`icpe-data` 的重點（細節以 PREREG 為準）：
  - GPU 層＝**1,777 塊**（512-token、BF16，111.06 GiB；vLLM 0.28.0 預設 `gpu_memory_utilization`＝0.92，不是 0.9）。
  - TTL 網格 **0.3／1／3／10／30／100／300 s**，語意見 PREREG 第 3 項。
  - 8.07 那格：讀 8.07、寫 2.209 GiB/s、`k_read` 3.306。
  - 慢層上限＝⌊0.5 × `n_blocks_full`⌋ 個 INT4 塊（第二批）。
  - 雜訊兩種模型（PBKV 式混合 λ 0–40%；乘法 λ 0–30%）× 兩種抽法 × 10 seed。**PBKV 的雜訊不是時間乘法誤差**（計畫 §6b 草案寫錯）。
  - 並行 **16／32／64／72**；佇列順序 3 個 seed 隨機打亂；CacheWise 一律 5 份複本（470 個對話）。
  - 預取：主結果到 CPU；另跑「直接到 GPU」的敏感度（舊紀錄全部格子）。
  - 舊紀錄（Mooncake、FreeInference T1／T4／T5）**全部格子**用新模擬器重跑。
  - 前綴快取式（APC）在 session 粒度的定義：請 `icpe-sim` 在 G1 前寫進 STATUS_SIM。

**卡住什麼**：無。

**需要對方做什麼**
- `icpe-sim`：照 PREREG 第 1–6 項實作；APC 式的定義寫進 STATUS_SIM。
- `icpe-data`：照 PREREG 第 7 項與 §3 C、D 做到達合成。

**需要使用者決定什麼**（已在 `h200` session 直接問）
- GPU 預取敏感度要不要也跑新紀錄（W2）——PREREG 未登記。
- W6c 敏感度重跑的「代表格」是哪些——PREREG 未登記，要在 W6c 有結果之前定。
- `code/icpe/x1_gpu.py`（W6b 要改）由哪個 session 改——§3b 沒有歸屬。

**下一步**：W1（Mooncake 命中率，只用現有輸出）→ W8 其餘項目。
