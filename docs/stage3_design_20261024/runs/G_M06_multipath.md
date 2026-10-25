# G-M06＋M07：借鄰卡的 PCIe 多路還原；NUMA 放置

> **性質**：判準登記（事先寫死）。來源：[mechanisms/03 M06、M07](../mechanisms/03_bandwidth_compute.md)、[mechanisms/06 §2](../mechanisms/06_eval_gates.md)（G-M06：「k＝2 端到端 ≥1.6 倍 → 過」）。使用者 10/25 選「G-M06＋M07」。
> **4 卡 job、量時間、獨佔**（lab 帳號的 4 張卡全部要空出來；節點上不能有我們自己的別的 job），排在 ICPE 的量時間鏈與 B0 之後。速度判準照 [05 §2.1](../05_poc_plan_h200.md)。程式：`code/s3/gm06.py`。

## 1. 量什麼（M06 卡的「最小證偽」四步＋M07 的 NUMA）
1. 拓撲：`nvidia-smi topo -m`、每張卡的 PCI 位址與 NUMA 節點、peer access、job 的 cpuset／mems。
2. 1／2／4 張卡同時 H2D（每張卡從自己的 pinned 緩衝搬 4 GiB）：合計與各自的 GiB/s。
3. NVLink 卡對卡：GPU j → GPU 0 的 4 GiB D2D（j＝1..3），GiB/s。
4. **16 GiB 經 1／2／4 條路徑送進 GPU 0**：k＝1：GPU 0 直接 H2D；k＝2、4：16 GiB 切成 256 MiB 的片輪流分給 k 條路徑，鄰卡先 H2D 到自己的兩塊暫存區（輪流），再用 D2D（NVLink）寫進 GPU 0，H2D 和 D2D 管線化。端到端時間＝所有路徑做完（同步）。抽查頭尾 1 MiB 和來源相同。
5. M07：GPU 0 從「本地 NUMA 節點」與「遠端節點」的 pinned 記憶體（set_mempolicy）H2D 4 GiB；cgroup 不允許遠端節點 → NOT_MEASURED。
- 每點 1 次暖機＋6 次；CSV 每列 `n_rep`、`sigma_rel`、`delta_star`。
## 2. 判準（事先寫死）
- 加速＝t(k＝1)／t(k＝2)（中位數）；95% CI＝兩臂各自對 rep 重抽 10,000 次。
- **G-M06-PASS**：加速 ≥ 1.6 **且** CI 下界 ≥ 1.6。點估計 ≥ 1.6 但 CI 下界 < 1.6 → **INCONCLUSIVE**（加重複次數）。點估計 < 1.6 → **G-M06-STOP**（主機端是瓶頸，寫成量測結論）。k＝1 那一臂 2σ > 5% → 先回報、不判定。
- 另報：k＝4 的加速、1／2／4 卡同時 H2D 的合計頻寬（看主機端或 root complex 是不是瓶頸，Beluga 的警告）、NVLink 頻寬、NUMA 本地／遠端的差。
- 不在這一關：鄰卡同時在做 decode 時的 TPOT 影響（M06 卡的評估設定；主實驗才做）。

## 登記
- 登記時間（`date -u`）：**2026-10-25T13:43:22Z**；commit 訊息開頭 `docs(s3-gm06): prereg`。登記時沒有任何 G-M06 的數字（程式只過了 CPU 自測）。

---

## 已知問題（不是偏離；量法不改）

- **M07 的「遠端」那一列不可用**（另一個 session 10/25 找到，使用者選「只推記錄、不修」）：\`gm06.py\` 先配本地的 4 GiB pinned、\`del\`，再配「遠端」的 4 GiB；torch 的 pinned 快取會把同一塊（同一個位址）還回來，所以 \`numa_remote\` 實際量到的是本地記憶體。每列的 \`src_numa\` 欄會照實記下緩衝真正落在哪個節點；報告時 \`numa_remote\` 一律標成「不可用（快取重用）」，不拿來比本地／遠端。
- 不影響 G-M06 的判準（k＝2 的端到端加速 ≥1.6）。要修的話是量法變更，必須在任何 gm06 數字之前寫成偏離。

## 偏離（登記之後）

### 偏離 1（2026-10-25T14:23:30Z；實作，判準不變；G-M06 還沒有任何時間數字）：量測期間加 GpuWatcher，被污染的 run 不寫進 results

- **改什麼**：`code/s3/guard.py`（RUNLOG S3-GUARD-2）。量測開始前開監看（開不起來 → 這個 run 直接停，不在看不到的情況下量）；結束時被污染（有外來 process，或監看期間 nvidia-smi 查詢失敗過，S3-GUARD-1）→ 結果寫到 run 目錄的 `<檔名>.CONTAMINATED`、留 `CONTAMINATED` 檔，**不寫進 results**，要重量；每列多 `own_gpu_intruders`（0／外來者數／`UNOBSERVED:<n>`）；report 再按 run_id 排除一次（`excluded_contaminated_runs`）。4 張卡**都**監看（任何一張被污染 → 整個 run 不進 results）。
- **為什麼**：CLAUDE.md §3（gpu_guard 必用；contaminated 的 run 不得寫進 results）。原本的程式只靠 runner 開跑時的獨佔檢查，量測期間沒有監看（ICPE session 10/25 提醒）。
- **影響的 run**：全部（都還沒跑）。
