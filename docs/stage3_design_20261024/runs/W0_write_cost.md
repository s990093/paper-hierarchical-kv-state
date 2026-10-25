# W0：降級時才壓縮，來得及嗎（寫入端成本）

> **性質**：判準登記（事先寫死）。登記時間見文末。來源：[05 §8](../05_poc_plan_h200.md)（GW0）、06_tracker R04（CPU）、R17（GPU 側）。
> **量時間、獨佔**（節點上不能有我們自己的別的 job）；排在 ICPE 量時間的鏈之後。程式：`code/s3/w0.py`。

## 1. 量什麼
- 一個 32K 請求（Llama-3.1-8B、Qwen2.5-7B-1M；亂數 token、fa3 prefill）的區內 63 塊（和 A0 同一個口徑），壓成：
  - 粗底 B16（`xl:k20:v30:cal+q:int4_kivi`）或 B32（`xl:k13:v20:cal+q:kv-int4_kivi-int2_kivi`），A0 登記 §1.3 的同一個校準基底；
  - 精修：每塊 INT4 KIVI（simulcast，和 A0 的位元組帳一致）。
- **只量編碼**（要存的東西：係數＝Z·U 再量化、INT4 的碼與 scale），不量解碼。
- (i) CPU：16 個執行緒（job 拿到的 16 核），KV 已在主機（D2H 不算）；(ii) GPU：另一條 stream，閒時、和背景 decode:32:8192 一起跑（另記背景吞吐比）。
- 每格 1 次暖機＋6 次，取中位數。

## 2. 判準（事先寫死）
- 需求速率（HBM 滿時每秒要降多少）＝prefill 產生 KV 的速度：
  - Llama：vLLM 0.28 冷 prefill 32K＝1.224 s（RUNLOG_H200.md R2-C6）→ 26,771 token/s × 128 KiB＝**3.27 GiB/s**；
  - Qwen：H200 上 vLLM 的 Qwen prefill 沒量過（NOT_MEASURED）→ 用 harness fa3 的 calib-f Σ64＝1.3934 s（ICPE P0-1 S1）→ 23,516 token/s × 56 KiB＝**1.26 GiB/s**（vLLM 會更快，所以這是寬鬆的那一邊，報告要寫）。
- **GW0-PASS**：CPU（16 核）的「粗底＋精修」壓縮速度（輸入的 BF16 位元組 ÷ (t_粗底＋t_INT4)）在 B16、B32 兩種粗底都 ≥ 需求速率 → 「降級到慢層時才在 CPU 壓縮」成立，寫入不碰 GPU。
- **GW0-FAIL**：否則 → 要在 GPU→CPU 卸載時就在 GPU 上壓；GPU 側的時間與它對背景的拖慢（R17）就是 B 方向要算進去的成本。
- 另報：GPU 側的時間、背景吞吐比；每種粗底的輸出位元組（要和 A0 的 predict_bytes 相同）。

## 3. 成本〔估計〕約 0.3 GPU 小時（兩個模型）。

## 登記
- 登記時間（`date -u`）：**2026-10-25T12:48:11Z**；commit 訊息開頭 `docs(s3-w0,gfio): prereg`。登記時沒有任何數字。

---

## 偏離（登記之後）

### 偏離 1（2026-10-25T13:06:45Z；使用者要求；W0 還沒跑）：「雜訊」的操作定義 → 05 §2.1
- **改了什麼**：GW0 的比較對象（需求速率）是常數，σ 用被量的那一臂（CPU 壓縮的 6 次）。**GW0-PASS**：CPU「粗底＋精修」的壓縮速度中位數 > 需求速率 ×(1＋δ\*)，且速度的 95% CI（對 rep 重抽 10,000 次）下界 > 需求速率（B16、B32 都要）。**GW0-FAIL**：中位數 < 需求速率 ×(1−δ\*) 且 CI 上界 < 需求速率。其他 → **GW0-UNDETERMINED**（照實報告，不判 PASS／FAIL）。2σ > 5% 的點先回報。
- **CSV 加欄位**：`n_rep`、`sigma_rel`、`delta_star`（每個 (where, bg, part)）。
- **為什麼**：同 runs/README.md「共同規則」。**受影響**：W0 的全部 run（都還沒跑）。

### 偏離 2（2026-10-25T14:23:30Z；實作，判準不變；W0 還沒有任何時間數字）：量測期間加 GpuWatcher，被污染的 run 不寫進 results

- **改什麼**：`code/s3/guard.py`（RUNLOG S3-GUARD-2）。量測開始前開監看（開不起來 → 這個 run 直接停，不在看不到的情況下量）；結束時被污染（有外來 process，或監看期間 nvidia-smi 查詢失敗過，S3-GUARD-1）→ 結果寫到 run 目錄的 `<檔名>.CONTAMINATED`、留 `CONTAMINATED` 檔，**不寫進 results**，要重量；每列多 `own_gpu_intruders`（0／外來者數／`UNOBSERVED:<n>`）；report 再按 run_id 排除一次（`excluded_contaminated_runs`）。
- **為什麼**：CLAUDE.md §3（gpu_guard 必用；contaminated 的 run 不得寫進 results）。原本的程式只靠 runner 開跑時的獨佔檢查，量測期間沒有監看（ICPE session 10/25 提醒）。
- **影響的 run**：全部（都還沒跑）。
