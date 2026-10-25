# 06　實驗追蹤表（第三階段，H200）

> 每列一個 run 單位。狀態：TODO／REG（判準已登記）／RUN／DONE／STOP。跑完把 run_id 填進「備註」，結果寫進 `results/s3_h200/<代號>/`，每列 CSV 要有 `run_id`、`ts`（CLAUDE.md §4.3）。
> 「獨佔」＝量時間，同時間不跑我們自己的其他 job。GPU 小時都是估計（NOT_MEASURED）。
> 計畫本文：[05_poc_plan_h200.md](05_poc_plan_h200.md)。

| Run | 階段 | 目的 | 系統／變體 | 題組 | 指標 | 優先 | GPU 小時 | 獨佔 | 狀態 | 備註 |
|:--|:--|:--|:--|:--|:--|:--|--:|:--|:--|:--|
| R00 | 0 | 判準登記 | A0、B0、C0、D0、W0 的判準 commit＋`date -u` | — | — | MUST | 0 | — | TODO | 11/2 前 |
| R01 | 0 | 混合清晰度路徑 | harness 每塊各自的格式；逐塊記讀了哪一級 | 4 題冒煙 | 位元組帳斷言、和 E0 逐位元比對 | MUST | 0.5 | 否 | TODO | P0a |
| R02 | 0 | 真實背景產生器 | 同 process 雙 stream：decode B＝8／32／128、chunked prefill、混合批次 | — | 背景單獨跑的吞吐 | MUST | 0.5 | 是 | TODO | P0b |
| R03 | 0 | C0 成本模型 | 演算法 C＋殘差；四種切法 | 既有 MI300X／H200 格子 | 選錯代價 平均／最差 | MUST | 0（CPU） | — | TODO | P0c |
| R04 | 0 | W0 寫入成本（CPU） | 校準基底投影＋INT4＋殘差，16 核 | 32K × 2 模型 | 秒／請求、GB/s | MUST | 0（CPU） | — | TODO | P0d |
| R05 | 0 | 環境 | MPS、stream priority、green context 可用性 | — | 能／不能 | MUST | 0.3 | 是 | TODO | P0e |
| R06 | 0 | D0 模型冒煙 | 依 L8d 選的原生稀疏模型 | 2 題 | 能不能跑 32K | NICE | 0.5 | 否 | TODO | P0f |
| R10 | 1 | A0-TF Llama | 均勻 U(r) 6 個；漸進 base 3 × ρ 3 × 目標 2 × 挑法 4；丟塊對照 12 | X4a 式 96 題 | Δgold_lp（配對 CI） | MUST | 5 | 否 | TODO | |
| R11 | 1 | A0-TF Qwen | 同上 | 同上 | 同上 | MUST | 5 | 否 | TODO | |
| R12 | 1 | A0-gen Llama | TF 最好的約 12 個 | 新問題 192 題＋第 2 輪 150 題 | 答對率對 E0、對同位元組 U(r) | MUST | 7 | 否 | TODO | GA0 判定 |
| R13 | 1 | A0-gen Qwen | 同上 | 同上 | 同上 | MUST | 7 | 否 | TODO | GA0 判定 |
| R14 | 1 | B0 動作 × 背景 × 共用方式 | 7 種動作 × 8 種背景（含 prefill 節點）× 5 種共用方式 × 2 模型；另報只用生產格式的子集 | — | 還原時間（中位、P95）、背景吞吐比 | MUST | 8 | 是 | TODO | GB0 判定 |
| R15 | 1 | C1 新格子檢驗 | 12 個沒看過的（長度 × 負載 × kernel）格子 | — | 實測最好 vs 預測 | MUST | 4 | 是 | TODO | GC 判定 |
| R16 | 1 | D0 原生索引 recall | 原生索引 vs Quest-512 vs 完整 K | C1a 式 DS2 | 答案塊 recall@k | NICE | 6 | 否 | TODO | GD0 判定 |
| R17 | 1 | W0 GPU 側 | 同 R04，改在 GPU 側 stream | 32K × 2 模型 | 秒／請求、對背景的拖慢 | NICE | 1 | 是 | TODO | |
| R20 | 2 | A1 可部署版 | BASE-ATTN（或 DL1／原生索引）；3 模型 × 32K／64K／128K | 每格 ≥288 題 | 答對率、位元組 | MUST | 80 | 否 | TODO | GA0-GO 才做 |
| R21 | 2 | A2 端到端 | CPU／NVMe／NFS／b16／b32 × 閒／忙（雙 stream、MPS）× Cake 結合 × 打字預取 | — | 還原時間、TTFT | MUST | 20 | 是 | TODO | |
| R22 | 2 | B1 線上選擇器 | 演算法 B vs 最好固定 vs G5b vs KVServe 式 vs 事後最好 | 背景照真實負載序列 | 回來那一輪 TTFT P95、別人 TPOT P95 | MUST | 25 | 是 | TODO | GB0 任一成立才做 |
| R23 | 2 | A-DL1 學的挑塊器 | MLP／梯度提升樹 | A0 oracle 標籤 | 拿到 oracle 好處的比例 | NICE | 3 | 否 | TODO | GA0-SEL 50–70% 才做 |
| R30 | 3 | X1 新意分離 | 變糊 vs 丟；漸進 vs 均勻（同位元組） | A1 題組 | 答對率差 | MUST | 15 | 否 | TODO | |
| R31 | 3 | X2 簡單性 | 兩級 vs 三級；每塊 vs 每（層、塊）；simulcast vs 分層編碼 | A1 題組子集 | 答對率、儲存、寫入成本、讀取位元組 | MUST | 15 | 否 | TODO | |
| R32 | 3 | X3 學習元件 | A-DL1 vs BASE-ATTN；（選做）殘差向量量化 codec | 同上 | 同上 | NICE | 10 | 否 | TODO | |
| R33 | 3 | X4 規則 vs 現場 | 同 R22 的對照組，更多 trace | — | TTFT P95、TPOT P95 | MUST | 15 | 是 | TODO | |
| R34 | 3 | X5 規模 | 4 模型 × 32K–128K × 合成／SCBench／LongBench × 4 下層 × 4 背景 | 子集 | 全部 | MUST | 80 | 部分 | TODO | |
| R35 | 3 | V1 vLLM 整合 | OffloadingConnector 降級時壓縮、漸進式載入、真實 trace 重播 | I3g 式工作負載 | 回來那一輪 TTFT P95 | NICE | 30 | 是 | TODO | |
