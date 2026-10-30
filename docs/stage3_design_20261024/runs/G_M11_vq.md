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

## 偏離（登記之後）

### 偏離 1（2026-10-26T08:37:39Z；使用者 10/26「都做掉」：重量那四格；判準不改）
- **看過什麼之後決定**：看過 B0 報告的 G-M11 結果（NOT_DETERMINED）、四格各自的 σ_rel（上面結果節），以及**閒時 dq_int4 那一格的 6 次**：llama 3.431／2.790／2.770／2.764／3.428／2.723 ms，qwen 2.814／3.399／3.084／2.804／2.821／2.810 ms（分成約 2.76 與 3.43 兩群，**雙峰**，不是單一離群值）。**沒有看** VQ 對 INT4 的比或它的 CI。
- **改什麼**：另開一個量時間的 run（runner part `gm11-redo`），只量判準的四格（閒／matmul × dq_vq／dq_int4，b0、stream），**每格 12 次**（05 §2.1：量不準就加重複次數），兩個模型；輸出放 `results/s3_h200/b0/gm11/`，**判定只用這個 run 的四格**（σ 要在同一個 allocation 裡算，不和 B0 原本的 6 次合併）；報告 `b0.py report --raw-dir results/s3_h200/b0/gm11 --name gm11_report.json`。
- **如果還是 2σ > 5%**：照 §2 維持 NOT_DETERMINED、照實回報（雙峰的話加次數不一定會讓 σ 變小）。

### 偏離 1 的結果（2026-10-26 18:05–18:08；job 22478）

- **run_id**：llama `20261026-180520-s3-gm11-llama`、qwen `20261026-180657-s3-gm11-qwen`（各 48 列＝4 格 × 12 次；guard 都 `CLEAN`）→ `results/s3_h200/b0/gm11/`；報告 `b0.py report --raw-dir results/s3_h200/b0/gm11 --name gm11_report.json`（run_id `20261026-180825-adhoc`）。
- **判定：仍然 NOT_DETERMINED（兩個模型）**。閒時 dq_int4 的 σ_rel：llama **0.038**（2σ＝7.7%）、qwen **0.102**，仍 2σ > 5%；其他三格都穩：dq_vq 閒 0.0022／0.0030、matmul 0.0026／0.0020；dq_int4 matmul 0.0021／0.0013。照偏離 1 的事先約定，維持 NOT_DETERMINED，**沒有看** VQ 對 INT4 的比。
- **現象（只報告）**：只有「閒時的 INT4 融合 kernel」在兩次獨立的 run、兩個模型都不穩（B0 原本 6 次是雙峰）；有 matmul 背景時同一個 kernel 很穩。一個可能的解釋是閒時 GPU 時脈在兩個狀態之間切換——**NOT_VERIFIED**（沒有量時脈）。
- **還能怎麼做（待使用者決定，都要另寫偏離）**：(a) 量測時同時記 GPU 時脈（nvidia-smi `clocks.sm`），看雙峰是不是對到時脈；(b) 閒時那一格改用固定時脈（需要權限，可能做不到）；(c) 接受 NOT_DETERMINED，G-M11 這一輪不判。

## 結案（2026-10-28，使用者決定）

- 使用者 10/28「照建議」選 (c)：**接受 NOT_DETERMINED**，G-M11 這一輪不再量。理由：G-M11 只影響用到 VQ 解碼的論文形狀，06 §4 的 P1–P3 都已排除。閒時 dq_int4 的雙峰現象照上面「現象」一節保留，原因 NOT_VERIFIED。
