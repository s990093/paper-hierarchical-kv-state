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
