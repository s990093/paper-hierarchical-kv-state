# 2026-09-27 查詢報告（證據附件）

這個資料夾是 `../DIRECTIONS_20260927.md` 的證據附件。產生方式與可信度：

| 檔案 | 內容 | 可信度 |
|---|---|---|
| `verified_findings.md` | `deep-research` workflow 的綜合結論原文（F0–F8）、caveats、未回答問題、15 條被否決的 claim | 只用通過三票驗證的 10 條 claim；綜合者加的【算術】【判讀】沒有驗證 |
| `evidence_claims.md` | 28 個來源（S01–S28）的 140 條擷取 claim，每條附原文引句、是否進入三票驗證與票數、我的抽查紀錄 | 見每條的標記 |

## 怎麼產生的

- **workflow**：`deep-research`，run id `wf_00cb5dd9-097`（2026-09-27）。5 個搜尋角度 → 28 個來源 → 140 條 claim → 只挑 25 條進三票對抗式驗證（10 條通過、15 條否決）→ 綜合。共 110 個子 agent。
  - 三票驗證的投票者是 Claude subagent，**不是 cross-model review**（`CLAUDE.md` §5 的已知落差）。
  - 25 條被驗證的 claim 全部來自角度 A（跨 agent／adapter 重用）、C（hybrid 模型）、E 的 VeriCache。**角度 B（P/D、RDMA、GDS）與 D（經濟、碳、隔離）沒有任何一條進入驗證**——不是被否決，而是沒被選進驗證預算。
- **抽查**（2026-09-27，我用 WebFetch 重讀原始頁面，只核對標題、作者、日期與關鍵句，不是全文驗證）：
  vLLM blog 2026-01-08（S20）、vLLM #55434（S19）、arXiv 2503.14647（S28，⚠️ 摘要沒有「每小時一次」門檻）、Lynx 2607.01831（S23）、vLLM #39321（S15）、keepalive 2607.19214（S27）、GreenCache 2505.23970（S26）、Gu et al. 2502.07776（S24）、SparseSpec 2512.01278（S16，只到摘要）、SpeCache 2503.16163（S18，只到摘要）、SGLang RFC #40865（S02）、llm-d hybrid blog（S04）、SGLang PR #23315（不在 S 清單，見主文方向 18）。
- **原始碼閱讀**（沒有執行）：vLLM `v0.28.0` 的
  - `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`：`_calc_num_offloadable_tokens`、`_build_store_jobs`（decode 產生的 token 預設會被卸載，只有 `offload_prompt_only` 才截到 prompt）；`get_sliding_window_size_in_chunks` 把 `MambaSpec` 當成 1 個 chunk 的 sliding window。
  - `vllm/v1/core/kv_cache_utils.py`：`_gen_lora_extra_hash_keys` 把 LoRA 名稱放進 block hash；`cache_salt` 只加在第一個 block。

## 重要限制

- 文中所有數字都是**別人在別的硬體上**量的（H100、GB200、RTX PRO 6000 Blackwell、Ascend 910B、A100、A6000、RTX 2080 Super）。依 `CLAUDE.md` 規則 1，它們只能當我們實驗的假設，不能寫進 `results/`。
- 「沒人做過」只相對於本輪讀過的 28 個來源與 9/24 的 35 篇；動手前要先做 `novelty-check`。
- 被否決的 claim（`verified_findings.md` 最後一節）不得引用；否決代表那條 claim 的細節在原文找不到支持，不代表論文不存在。
- `evidence_claims.md` 的 S25（Anthropic 定價文件）裡，特定的 Claude 模型版本名稱已改成「〔某 Claude 模型〕」；定價倍數與原文其餘內容未動。
