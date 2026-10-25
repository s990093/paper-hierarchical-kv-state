# RUNLOG — 第三階段（docs/stage3_design_20261024/）在 H200（hgx-cn01）

> 計畫：[05_poc_plan_h200.md](../docs/stage3_design_20261024/05_poc_plan_h200.md)、[mechanisms/06_eval_gates.md](../docs/stage3_design_20261024/mechanisms/06_eval_gates.md)；追蹤表：[06_tracker.md](../docs/stage3_design_20261024/06_tracker.md)。
> runner：`code/r_h200_s3.sh`；新程式在 `code/s3/`；結果在 `results/s3_h200/<代號>/`；每個閘的判準登記在 `docs/stage3_design_20261024/runs/`（登記先、結果後附，登記不改）。
> 沒量到的寫 NOT_MEASURED；失敗照實記、停下回報（CLAUDE.md §1）。

---

## S3-0 — 開工前置（2026-10-25，head node）

**狀態**: 進行中
**git**: 從 `ae60a7f`（main）開始；本 worktree `paper-hkv-stage3` 在 main。

**看到的（照實）**
- `/work/ChihChungHsu/hungweilai/hf-cache/hub/` 是空的（目錄 mtime 2026-10-23 03:12 +08）；`/work/ChihChungHsu/hungweilai/models/` 也是空的（noDCA 變體不在）。
  10/22 的 `RUNLOG_H200_ICPE.md` ICPE-0b 還記著 Llama-3.1-8B（14.97 GiB）、Qwen2.5-7B-1M（14.2 GiB）、noDCA 變體都在。誰、為什麼刪：查不到（NOT_VERIFIED）。
  資料集（`datasets/{hotpotqa,longbench,ruler_ref,scbench,traces}`）還在。
- 群組配額 `quota -s -g ChihChungHsu`：1,487G／1,536G（剩 49G）。我們帳號：DL_Lab4 62G、其他約 4G。
- GPU：lab 帳號上限 4 張（和 hao、chihchung 共用），開工時 4／4（我們 3：icpe-cpu、DL_Lab4 final_ens、J_kd；hao 1）。
- head node 的 `/tmp` 清理：`/usr/lib/tmpfiles.d/tmp.conf` 是 `D /tmp 1777 root root 30d`，`systemd-tmpfiles-clean.timer` 每天一次（30 天沒碰的檔案刪掉、開機清空）。compute node 的設定由 fetch job 記錄（`node_tmp_policy`）。
- 從 head node `ssh hgx-cn01`：連線被拒（只能從 job 裡看 compute node）。

**使用者的決定（2026-10-25）**
1. GPU 排程「排滿：stage3 品質先填空檔」：品質 job 切成 ≤4 小時、可續跑的分片，一有空卡就跑；下一個要開跑的 ICPE 量時間 job 加 `afterany:<正在跑的 stage3 分片>`；hold 中的 ICPE job 不動；stage3 量時間的 job 排在 ICPE 量時間鏈之後（memory `stage3-gpu-scheduling`）。
2. 模型權重「全部只放 `/tmp`，由我通知 ICPE 那個 session 自己改路徑」：hgx-cn01 本地 `/tmp/hungweilai-models/hf-cache`（HF_HOME 的版面）；每個 GPU job 開跑前核對戳記，不在就重抓。
3. 要跟老師要配額（使用者自己談）：篩選閘的權重 57 GiB（Llama 14.96＋Qwen-7B 14.19＋Qwen-14B 27.51，HF API 逐檔加總）；主實驗再加 Mistral-NeMo-12B 22.8（只抓 HF 分片）＋DeepSeek-V2-Lite-Chat 29.26；D0 的 MiniCPM4.1-8B 15.25；X5 的 gpt-oss-20b 12.82／Nemotron-Nano-9B-v2 16.56／Qwen3-Next-80B-A3B-FP8 76.42 或 Qwen3.5-35B-A3B 66.97（GiB）。

**新程式**
- `code/s3/fetch_models.py`：固定 revision 下載（Llama `4699cc75…`、Qwen-7B `e28526f7…`、Qwen-14B `620fad32…`），每個檔和 HF 的雜湊比對（LFS：sha256；一般檔：git blob sha1），寫戳記與 `results/s3_h200/env/models_manifest_<run_id>.json`；`check` 用大小＋inode＋mtime 核對戳記。`--self-test` PASS（head node）。
- `code/r_h200_s3.sh`：stage3 的 runner；品質 part 遇到 ICPE 量時間的 job 在跑就 exit 3；量時間 part 遇到我們任何別的 job 就停。
- 觀察：Qwen2.5-7B-1M 和 14B-1M 的 `tokenizer.json` 是同一個 git blob（`443909a6…`）→ 題目可以兩個模型共用（開跑時再用 tokenizer 本身核對一次）。
