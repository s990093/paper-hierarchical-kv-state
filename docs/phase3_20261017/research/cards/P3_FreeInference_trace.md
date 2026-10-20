# P3 卡：FreeInference agentic trace（Harvard MadSys，2026）

- **資料**：HF `harvardMadsys/freeinference_agentic_trace`，sha `16d3c04d`（lastModified 2026-10-14），CC-BY-4.0，**不 gated**〔文件 HF API〕。<https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace>；程式 <https://github.com/HarvardMadSys/freeinference_agentic_trace>（沒讀）；hub <https://data.freeinference.org/>。
- **論文**：「From Requests to Sessions: A Large-Scale Characterization of Human-Driven Agentic Workloads」（Nixon, Tian, Zheng, Gunawi, Juncheng Yang，2026）——**沒讀**〔未查證〕。
- **讀了哪裡**：HF README 全文；下載第 1 週（2026-05-17）的 `processed/requests.parquet`（22 MB）、`sessions.parquet`、`tool_calls.parquet` 自己算〔計算，`../calc/freeinference_week1.txt`〕。

## 規模與欄位〔文件 README〕
- 16 週（2026-05-17 → 09-05）、12,002 個 agent session、267 個帳號、14 個 harness、1,186,582 次 LLM 請求、1,213,347 次工具呼叫；沒有訊息文字。
- session：`session_id`、`user_id`、`harness`、`start_ms`、`end_ms`、`requests`；子 agent 的 session 巢狀在工具呼叫下。
- 請求：`step`、`turn_index`、`step_trigger`（user／tool）、`link_reason`、`start_ms`、`end_ms`、`ttft_ms`、`model`、`provider`、`provider_tokens`{prompt, completion, cached}、`role_tokens`、`mutation`（和上一個請求的差異，如 append）、`tool_definition_names`、`finish_reason`、`block_ids`、`tool_calls`。
- 工具呼叫：`name`、`category`、`programs`、去識別的 `arguments`、`outcome`、`result_tokens`、`latency_ms`（「到下一次 LLM 請求的間隔」）、`subagent`。
- `block_ids`：`o200k_base` 切成 16-token 的**鏈式**前綴 block，順序是工具定義 → 系統提示 → 訊息；最後不滿一塊的丟掉；id **每週各自編號**。
- session 是用重複的 tool-call id 與 assistant 訊息**重建**的；**session 不跨週**，跨週的會變成新 session；少於 3 個帳號用過的工具名或指令會被換成 `other`。

## 第 1 週〔計算〕
- 125 個 session（19 個子 agent）、8 個使用者；harness：kilo 46、opencode 31、hermes 28、cline 15、pi 5。
- 10,738 個請求：使用者發起 785、工具觸發 9,953。
- 單位斷言 `len(block_ids)==floor(Σrole_tokens/16)`：100%（ceil 只有 6.5%）。
- provider prompt token：中位 89,242、p90 139,809、p99 192,914；≥8K 99.2%、≥32K 88.2%、≥128K 13.9%。
- 使用者發起的下一輪（距上一個請求結束）：p10／50／90／99＝25／175／3,587／34,687 s。工具觸發的：0.5／3.4／30.5／294.7 s。
- 工具 `latency_ms`：中位 3,526、p90 27,423、p99 258,377。
- 每個 session 的使用者回合：中位 3、p90 11。

## 和第三階段的關係〔判讀〕
- 目前最適合當主 trace：長 context、session、使用者、前綴 block、真實間隔、≥1,000 session 全都有。
- 長度用 `o200k_base` 算，和我們的 Qwen／Llama tokenizer 不同；換算時要寫明，且單位斷言要照 README 的「丟掉最後不滿一塊」。
- 「回來」有兩種尺度：工具觸發（秒）與使用者發起（分鐘到小時）。降級／丟掉的決策主要落在後者。
- 每週的 session 截在週邊界，等於每週一次右設限；可以照 Copilot 論文「這週訓練、下週測」切。
