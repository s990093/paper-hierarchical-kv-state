# G-M11：不吃 tensor core 的 codec——查表式 VQ 在算力型背景下是不是比較不會被拖慢

> **性質**：判準登記（事先寫死）。來源：[mechanisms/03 M11](../mechanisms/03_bandwidth_compute.md)、[mechanisms/06 §2](../mechanisms/06_eval_gates.md)（G-M11）。
> **量時間、獨佔**：**併在 B0 裡量**（B0 的還原動作多一個 `dq_vq`，B0 偏離 2）；不另外開 job。速度判準照 [05 §2.1](../05_poc_plan_h200.md)。

## 1. VQ 的定義
- product quantization：每 4 維一個子空間（D＝128 → 32 個子空間）、每個（層, K/V, 子空間）一本 256 項的碼本（head 共用），每個值 2 bit（BF16 的 8×）；碼本在那個 session 的 KV 上抽 65,536 個向量跑 6 次 Lloyd（k-means）。
- 解碼＝查表（PyTorch 的 gather），不用 tensor core、ALU 很少。這一關只看**解碼速度**；VQ 的品質不在這一關量（M11 卡：品質要另外和 INT4／低秩在同位元組下比）。
- 對照：`dq_int4`（INT4 KIVI，m10_dequant_fast 的融合 kernel）、`lr_recon`（xKV 粗底 B16 的解碼，吃 tensor core）。
## 2. 判準（事先寫死）
- 慢幾倍＝t(背景, stream)／t(閒, stream)，b0 setup（32K、64 塊）。
- 每個模型：**VQ 在 matmul 背景下比 INT4 少慢**＝INT4 的慢幾倍 ÷ VQ 的慢幾倍 −1 > δ\*（δ\*＝INT4 在 matmul 背景那一格的，05 §2.1），且這個比的 95% CI（四格各自對 rep 重抽 10,000 次）下界 > 1。
- **GM11-PASS**：兩個模型都成立；**PARTIAL**：只有一個；**GM11-STOP**：都不成立（M11 卡：「VQ 在 matmul 背景下沒有比 INT4 少慢 → 停」）。四格裡有 2σ > 5% 或不到 6 次 → 先回報、不判定。
- 另報：decode:32:8192、mixed、cprefill 背景下的慢幾倍；VQ 對 lr_recon。

## 登記
- 登記時間（`date -u`）：**2026-10-25T13:33:09Z**；commit 訊息開頭 `docs(s3-gm11): prereg`。登記時 B0 沒有任何時間數字（dq_vq 只在 CPU 上驗過編碼／查表邏輯）。

---

## 結果（2026-10-26；併在 B0，報告 run_id `20261026-162611-adhoc`）

- **判定：NOT_DETERMINED（兩個模型）**。判準的四格（閒／matmul × dq_vq／dq_int4，b0、stream、各 6 次）裡，**閒時的 dq_int4** 2σ > 5%：llama σ_rel **0.124**、qwen **0.086** → 照 §2「先回報、不判定」。其他三格都穩：dq_vq 閒 0.0017／0.0018、matmul 0.0026／0.0014；dq_int4 matmul 0.0001／0.0056。
- 要判定得重量這四格（加重複次數）；量法與判準不改。待使用者決定。
