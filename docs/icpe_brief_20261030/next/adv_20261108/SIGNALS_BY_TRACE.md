# 每份紀錄提供了什麼訊號、我們測了什麼、做了哪些假設（icpe-adv，2026-11-08）

> 使用者 11/8：「跟這麼多測試集，真的要確認是否是這份 trace 沒提供這些訊號、做不了、假設等等？」
> 來源只有文件與載入程式；**沒有打開 `/ssd6/hungwei/icpe-next/data` 或任何原始資料**。查不到的寫 NOT_VERIFIED。

## 0　故事：為什麼要確認，結果是什麼

- **要確認的原因**：摘要寫「紀錄裡的工具呼叫旗標拿不到」「這些紀錄沒有線索」，讀起來像是對全部 6 份紀錄都測過，其實不是。
- **查到的事實**：
  - **線索只在 TraceLab 測過**：TraceLab 是唯一記了「這一輪有沒有呼叫工具」的工具類紀錄；FreeInference 也跑了，但它的空間 <1%，判不了。
  - Mooncake toolagent、SemiAnalysis 沒有工具欄位，**沒測**，推翻條件寫「未定」。SWE-bench 軌跡也**沒測**。
  - **「這些紀錄沒有工具進度」不能一概而論**：
    - SWE-bench 軌跡（OpenHands、mini-SWE-agent）存了工具輸出的全文，但只在工具**結束後**才記一筆。
    - Ask the Tool 是**重新執行**那些指令，才讀到「執行中」的進度（它的 Fig. 1、Fig. 3、Fig. 11）。
    - 所以能說的是「紀錄沒有存執行中的進度」，不是「工具沒有進度」。
  - **打字**：沒有任何一份紀錄有按鍵時間；打字提前量都是用字數合成的（同 SYMPHONY 的評估做法）。
- **所以摘要要改兩句**（第 3 節）：
  - 線索那句限定在 TraceLab。
  - 結尾拿掉「these traces offer no cue」。

## 1　逐份紀錄的訊號表

欄位說明：
- 「觸發」＝紀錄有沒有標這個請求是使用者還是工具結果觸發的。
- 「工具時間」＝有沒有工具開始／結束的時間。
- 「內容」＝有沒有文字（沒有就只有 token 數或 hash）。

| 紀錄 | 到達 | 對話分組 | 觸發 | 工具時間 | 內容 | 按鍵 | 我們測過的線索 | 出處 |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| FreeInference（6 週） | 真實 | 有 `session_id` | 有（`step_trigger`：user／tool） | 沒有 | 沒有（只有 block id） | 沒有 | W3：下一個請求是工具觸發；空間 <1%，所以判不了 | `agent_traces.py:49-69`；PREREG 偏離 9（`PREREG.md:290`） |
| Mooncake conversation | 真實 | **沒有**，用 R1／R2 兩種規則重建 | 沒有 | 沒有 | 沒有（512-token hash） | 沒有 | 打字提示（舊模擬器＋1/8 GPU 重播；提前量用 token 數合成） | `agent_traces.py:344-353`（`no_sessions=True`）；`STORY.md` 第 6 章 |
| Mooncake toolagent | 真實 | 沒有，同上 | 沒有 | 沒有 | 沒有 | 沒有 | **沒測**（沒有欄位）→ 推翻條件「未定」 | `PREREG.md:293`；`REPORT.md:100` |
| TraceLab c1（c2、c4＝壓縮開始時間） | 真實（多使用者） | 有 | 有（第一個 timing event 是 `tool_result`／`user_message`） | 有（`emitted_at`／`result_at`） | 工具輸出文字：NOT_VERIFIED（載入程式沒讀） | 沒有 | **W3**：「這一輪有工具呼叫」旗標；20／35／59 格都拿不到一半（86% 的回合都有工具呼叫） | `agent_traces.py:75-95`；`NEXT_PLAN_20261104.md:141`；`PREREG.md:285`；`ADVISOR.md:90` |
| SemiAnalysis（並行 16–72） | **合成**（固定並行） | 有（紀錄 id；子 agent 各自一個） | 沒有（轉換時一律當非使用者） | 只有 `think_time`、`api_time`；有沒有工具欄位：NOT_VERIFIED | 沒有（64-token hash） | 沒有 | **沒測** → 未定 | `agent_traces.py:289-319`；`next_b_convert.py` 的 `conv_semianalysis`；`NEXT_PLAN_20261104.md:142` |
| Bailian（coder／thinking／traceA） | 真實 | 用 `parent_chat_id` 串 | 沒有（有 `type`、`turn`） | 沒有 | 沒有（16-token hash） | 沒有 | 沒測（非工具類，只報告） | `agent_traces.py:221-285` |
| ShareGPT、LMSYS、UltraChat | **全合成**（Poisson 開始＋指數思考時間 T60／T600） | 有 | 不適用（全是使用者） | 不適用 | **有文字** | 沒有 | 沒測（可以照 SYMPHONY 用文字長度合成打字時間，但沒做） | `PREREG.md:720-725` |
| WildChat | 開始合成；對話內**真實**間隔（assistant 回覆的時間戳） | 有 | 不適用 | 不適用 | 有文字 | 沒有 | 沒測 | `PREREG.md:726` |
| mini-SWE-agent、OpenHands | **合成**（固定並行） | 有 | 有（assistant／tool 訊息） | 有：工具**結束**的時間戳；呼叫開始的時間 | **有**：observation 原文（工具結束後的輸出） | 不適用 | **沒測**（agent 沒跑 W3）→ 未定 | `PREREG.md:999-1004、1024`；`ADVISOR.md:117` |
| CacheWise（內部；不進論文） | 合成（固定並行） | 有 | 有（tool／user） | 有（`execution_time_ms`） | 有 | 沒有 | 沒測（使用者 11/7「CacheWise 都先不跑」） | `agent_traces.py:98-218`；`PREREG.md` 偏離 9、15 |

## 2　做不了的、做了假設的（論文的限制要照這張寫）

| 項目 | 哪些紀錄 | 假設或做法 | 影響方向 |
|:--|:--|:--|:--|
| 對話開始時間 | SemiAnalysis、SWE-bench 軌跡、CacheWise（固定並行）；聊天資料集（Poisson） | 對話內的間隔用真值，只合成「何時開始」 | 並行數決定記憶體壓力；結果和原始到達分開報 |
| 對話分組 | Mooncake | 用兩種規則重建，取較小的上限 | 偏保守 |
| 前綴共用 | TraceLab、CacheWise（只有 token 數） | 只在同一對話內接續；不同對話之間不共用 | 前綴快取的對手可能偏弱〔判讀〕 |
| 思考時間 | ShareGPT、LMSYS、UltraChat | 指數分佈，平均 60／600 s（照 Pensieve） | 不代表真實使用 |
| 打字提前量 | Mooncake 對話 | 新 token 數 × 4.2693 字元/token ÷（51.56 WPM × 5 ÷ 60） | 合成；中文比例、貼上的文字都沒考慮 |
| 工具線索 | 只有 TraceLab | 只測「有沒有工具呼叫」；沒測工具種類、預估時間、執行中的進度 | 只能說「這個旗標拿不到」 |
| 執行中的進度 | 全部 | 沒有任何紀錄存執行中的進度；SWE-bench 軌跡只有結束後的輸出 | 「工具其實知道、只是沒交給 serving」這種情況沒量（Ask the Tool 是重新執行才量到） |
| tokenizer | OpenHands（Claude 回報 vs Llama 重算差約 0.795） | 照實記，影響方向沒量 | `ADVISOR.md:117` |

## 3　摘要要改的兩句（`ABSTRACT_v3.md` 第 1 節）

- **第 8 句**：
  - 原句：*The tool-call flag in these traces captures none of this gap.*
  - 新句：*On the one real-arrival tool trace that records tool calls, a cue that keeps or moves a session's KV when a tool starts captures none of this gap.*
  - 太長的話用短句：*On the trace that records tool calls, a tool-call cue captures none of this gap.*
- **第 9 句（結尾）**：
  - 原句：*…, and should reserve prediction for eviction under memory pressure, where these traces offer no cue.*
  - 新句：*…, and should reserve prediction for eviction under memory pressure.*
  - 「紀錄沒有存執行中的進度」放正文的限制，不放摘要。
- **正文的寫法**（照老師第 8 點）：
  - 能說：「這些紀錄沒有提供工具執行中的進度；TraceLab 唯一提供的工具呼叫旗標拿不到收益。」
  - 不能說：「工具類的未來猜不到」。
  - 要分清楚兩種情況：「未來真的難猜」和「工具知道、只是沒交給 serving」（Ask the Tool）；我們只量了前者的一部分。
