#!/usr/bin/env python3
"""F4 稽核（fork B）：把 main.tex §平台 B（1452–1605）、附錄平台 B 設計（1915–1937）與表 tab:models 的
每個數字逐列寫成 claims_B.csv。

證據都來自：
  - run 目錄 /mlsteam/data/tiara/runs/<run_id>/stdout.log（由 code/m9_f4_platB_check.py 抽出，
    見 results/audit_20261010/fragments/platB_evidence.csv，抽取 run 20261010-074654-f4-platB）
  - git 52803bc^ 的舊檔（RUNLOG_MI300X.md、alignment_*.json、m1_capacity/capacity.csv 等）
這支只把人工核對的結果寫成 CSV，不產生新的量測。self-review，不是 cross-model。
"""
import csv
import os
from datetime import datetime, timezone

REPO = "/mlsteam/workspace/paper-hierarchical-kv-state"
OUT = os.path.join(REPO, "results/audit_20261010/fragments/claims_B.csv")
EV = "results/audit_20261010/fragments/platB_evidence.csv"
R = "/mlsteam/data/tiara/runs"
G = "git 52803bc^:results/RUNLOG_MI300X.md"

C = []  # (section, line, claim_text, paper_value, kind, source_path, source_run_id, evidence_value, status, tag, note)


def c(*a):
    assert len(a) == 11, a
    C.append(a)


S0 = "§平台 B 導言"
c(S0, 1455, "平台 B：192 GB HBM3", "192 GB", "外部規格", "git 52803bc^:results/hw_mi300x.json", "", "vram_gib=192.0（206,141,652,992 bytes）",
  "一致", "〔git 52803bc^:results/hw_mi300x.json〕", "注意：206.1e9 bytes＝192 GiB，不是 192 GB；全文 GB／GiB 混用")
c(S0, 1455, "ROCm 7.2.2", "7.2.2", "設定", "git 52803bc^:results/env_mi300x.json", "", "rocm_version=7.2.2", "一致",
  "〔git 52803bc^:results/env_mi300x.json〕", "")
c(S0, 1455, "vLLM v0.28.0", "v0.28.0", "設定", f"{R}/20260917-103206-m5-needle/bf16/server.log", "20260917-103206-m5-needle",
  "server.log：Initializing a V1 LLM engine (v0.28.0)", "一致", "〔20260917-103206-m5-needle〕", "品質 run 用 venv tiara-v028；harness 本身在 venv tiara（0.19.1）")
c(S0, 1456, "七個模型涵蓋 dense 7B–36B、MoE 與 13–15B 級", "13–15B", "設定", f"{R}/*-m1-b-*/measure/result.json", "20260916-211306-m1-b-mistral-nemo12b-bf16",
  "實際兩個中型模型是 Mistral-Nemo-12B 與 Qwen2.5-14B-1M（12–14B）", "不一致", "〔run 20260916-211306、20260916-205607〕", "輕微：應為 12–14B 級")
c(S0, 1458, "全部 77 格（7 模型 × 11 項量測）皆已完成", "77/77", "實測", f"{R}/20260919-150350-verify/stdout.log", "20260919-150350-verify",
  "stdout 第 9 行：完成 77/77 格", "一致", "〔20260919-150350-verify〕", "")
c(S0, 1459, "通過自動自檢……0 個 FAIL", "0 FAIL", "實測", f"{R}/20260919-150350-verify/stdout.log", "20260919-150350-verify",
  "唯一有 run 目錄的 verify_results_b.py 執行：結論 5 個 FAIL、2 個 WARN（5 個 gpu_guard_*.json 標 CONTAMINATED_DURING_RUN）",
  "不一致", "〔20260919-150350-verify stdout:44〕",
  "之後新增 results/contamination_explained.json 解釋這 5 筆（teardown 的 pid −1），但找不到任何 0 FAIL 的 verify run 目錄")

S1 = "§b-capacity"
c(S1, 1464, "七個模型的 KV 容量倍數收斂到 FP8 2.00×", "2.00×", "實測", EV, "20260915-105515-m1-b-llama8b-bf16-kvfp8",
  "7 模型 1.9945–2.0131（5 個恰 2.0000；Llama 1.99、Qwen7B 2.01）", "一致", "〔run 20260915-1052xx～20260916-2128xx m1-b-*〕",
  "用原始 token 數算；平台 A 是每 GiB 正規化後的值。全部 guard=CLEAN")
c(S1, 1464, "INT8 1.94×", "1.94×", "實測", EV, "20260915-122248-m1-b-llama8b-bf16-kvint8", "7 模型全為 1.9394", "一致", "〔m1-b-* result.json〕", "")
c(S1, 1464, "INT4 3.77×（且與平台 A 相同）", "3.77×", "實測", EV, "20260915-110217-m1-b-llama8b-bf16-kvint4",
  "7 模型 3.7543–3.7647，四捨五入是 3.75–3.76", "不一致", "〔m1-b-* result.json〕", "輕微：B 的實測沒有一個到 3.77；3.77 是平台 A 每 GiB 正規化的值")
c(S1, 1466, "Seed-OSS-36B 的 BF16 上限來自記憶體，實測 413,632 token", "413,632", "實測", f"{R}/20260915-115701-m1-b-seedoss36b-bf16/stdout.log",
  "20260915-115706-m1-b-seedoss36b-bf16", "懸崖 413,632；limited_by=memory；475,676 啟動失敗（RUNLOG）", "一致", "〔20260915-115706-m1-b-seedoss36b-bf16〕", "guard=CLEAN")
c(S1, 1466, "低於模型可定址的 524,288", "524,288", "設定", f"{R}/20260915-115701-m1-b-seedoss36b-bf16/stdout.log", "20260915-115706-m1-b-seedoss36b-bf16",
  "模型可定址上限: 524288", "一致", "〔20260915-115701 stdout〕", "")
c(S1, 1467, "其餘六者皆先撞到模型自身的長度上限", "6 個 limited_by=model", "實測", f"{R}/2026091*-m1-b-*-bf16/stdout.log", "",
  "Llama 131,072、UltraLong 1,073,152、Qwen7B 262,144、Qwen3 262,144、Qwen14B 262,144、Nemo 131,072 皆 limited_by=model", "一致",
  "〔m1-b-*-bf16 外層 run stdout〕", "")

S2 = "§b-recompute"
c(S2, 1473, "線性擬合 C0+aP 的 R² 皆 ≥ 0.947", "≥0.947", "實測", EV, "20260915-164640-m2-b-qwen7b-recompute",
  "7 模型（每點 3 次取最小值）R² 最低 0.9469（Qwen2.5-7B-1M）", "一致", "〔7 個 recompute run stdout，算術重擬合〕",
  "若改用中位數擬合，Qwen3 0.912、Seed 0.925、Qwen7B 0.947、Qwen14B 0.952（RUNLOG 發現 10：P≥163,840 出現雙模態）")
c(S2, 1473, "（五個 ≥ 0.999）", "5 個", "實測", EV, "", "≥0.999 的有 6 個：Llama .9998、Qwen3 .9999、Seed .9993、UltraLong .9993、Qwen14B .9999、Nemo .9995",
  "不一致", "〔recompute run 20260915-164331、165438、221352、231329、20260917-144949、183724〕", "輕微：應為六個（RUNLOG 只有 5 個模型時是 4 個）")
c(S2, 1474, "UltraLong-8B-1M 一路量到 P=1,048,576", "1,048,576", "實測", f"{R}/20260915-231329-m2-b-ultralong-recompute/stdout.log",
  "20260915-231329-m2-b-ultralong-recompute", "P 最大 1,048,576（3 次）", "一致", "〔20260915-231329-m2-b-ultralong-recompute〕", "")
c(S2, 1474, "從 86 ms 成長至 10,350 ms（120 倍）", "86→10,350 ms，120×", "實測", f"{R}/20260915-231329-m2-b-ultralong-recompute/stdout.log",
  "20260915-231329-m2-b-ultralong-recompute", "stdout 中位數表 86.0 → 10,350.1 ms，120.33×", "一致", "〔20260915-231329 stdout 表〕",
  "這兩個是中位數；圖說說每點取最小值（最小值 85.5 → 10,328.0）")
c(S2, 1474, "斜率 9.75 µs/千 token", "9.75 µs/千 token", "實測", EV, "20260915-231329-m2-b-ultralong-recompute",
  "最小值擬合 a＝9.747 ms／每千 token（＝9.75 µs/token）", "不一致", "〔算術：重擬合 stdout 各點〕",
  "數值對，單位錯 1000 倍：應寫 µs/token 或 ms/千 token")
c(S2, 1475, "兩個 8B 模型斜率 9.25／9.75", "9.25／9.75", "實測", EV, "20260915-164331-m2-b-llama8b-recompute",
  "Llama 9.252、UltraLong 9.747（最小值擬合）", "一致", "〔recompute stdout 重擬合〕", "同上，單位應為 µs/token")
c(S2, 1475, "Seed-OSS-36B 為 30.76（3.3 倍）", "30.76；3.3×", "實測", EV, "20260915-221352-m2-b-seedoss-recompute-v2",
  "30.756；30.76/9.25＝3.33", "一致", "〔20260915-221352 stdout；算術〕", "")
c("圖 b-recompute 說明", 1482, "來源：results/m2_harness_mi300x/recompute_position_*.csv", "引用檔", "設定", "results/m2_harness_mi300x/recompute_position_*.csv", "",
  "git 任何分支都沒有；本機也沒有（stdout 寫「wrote … recompute_position_b-*.csv」，檔案後來被刪）", "找不到來源", "〔git log --all〕",
  "數字可由 7 個 recompute run 的 stdout 逐點重建")

S3 = "圖 b-tiers 說明"
c(S3, 1489, "GPU 內精度階反量化成本 FP8 0.17 µs/token", "0.17", "實測", f"{R}/20260915-124058-m2-b-llama8b-retrieval/stdout.log",
  "20260915-124058-m2-b-llama8b-retrieval", "stdout 第 125 行 gpu_fp8 +2.7 ms → 0.17 µs/token", "一致", "〔20260915-124058 stdout:125〕",
  "只有 Llama 16K；圖畫七個模型，其他模型同一量測值不同（Qwen14B FP8 0.72、UltraLong 0.41、Nemo 0.09 µs/token）")
c(S3, 1489, "INT4 1.59 µs/token", "1.59", "實測", f"{R}/20260915-124058-m2-b-llama8b-retrieval/stdout.log",
  "20260915-124058-m2-b-llama8b-retrieval", "stdout 第 126 行 gpu_int4 +26.1 ms → 1.59 µs/token", "一致", "〔20260915-124058 stdout:126〕",
  "Qwen14B INT4 0.62、UltraLong 0.53；Nemo INT4 NOT_MEASURED（20260917-134355）")
c(S3, 1490, "比跨裝置搬運低一到兩個數量級", "10–100×", "算術", f"{R}/20260915-124058-m2-b-llama8b-retrieval/stdout.log",
  "20260915-124058-m2-b-llama8b-retrieval", "CPU 30.53 µs/token ÷ 1.59＝19×；÷ 0.17＝180×", "一致", "〔算術〕", "")
c(S3, 1490, "Llama 量於 ctx=16,384，其餘為 96,000", "16,384／96,000", "設定", f"{R}/2026091*-m2-*-retrieval*/cmd.sh", "",
  "Llama --ctx 16384；Qwen14B／Nemo／UltraLong --ctx 96000（cmd.sh）", "一致", "〔cmd.sh〕",
  "Qwen3 的 CPU 階 delta 為負（發現 11，不採用），Qwen7B／Qwen3 第一批 SSD 無效（發現 8），圖的實際內容無法核對，因為來源 CSV 不在")
c(S3, 1491, "來源：results/m2_harness_mi300x/cost_constants_mi300x.csv", "引用檔", "設定", "results/m2_harness_mi300x/cost_constants_mi300x.csv", "",
  "git 任何分支都沒有（12aac34「不含原始 CSV」）；本機也沒有", "找不到來源", "〔git log --all〕",
  "可由各 retrieval run stdout 的「warm ms／減基準／µs/token」表重建")

S4 = "§b-headroom"
c(S4, 1499, "壓力 1× 時一律為 0.0%", "0.0%", "實測", EV, "20260919-125631-m4-b-llama8b-pressure",
  "7 模型 pressure1x 全為 0.0%（best=full_gpu）", "一致", "〔20260919-1256～1313 m4-b-*-pressure〕", "")
c(S4, 1500, "壓力 2–8× 時為 21–43%", "21–43%", "實測", EV, "20260919-130722-m4-b-qwen3-30b-a3b-pressure",
  "7 模型 2–8× 實為 5.9–43.0%；Qwen3-30B-A3B 2× 只有 5.9%（MARGINAL）；去掉這格才是 21.9–43.0%", "不一致", "〔20260919-130722 stdout〕",
  "最早的 09-16 壓力掃描沒有 run 目錄；09-18、09-19 重跑數字相同")
c(S4, 1500, "四個模型皆越過 15% 的 GO 門檻", "4 個模型", "實測", EV, "20260919-130722-m4-b-qwen3-30b-a3b-pressure",
  "實際跑了 7 個模型；Qwen3-30B 在 2× 是 5.9%，未過 15%", "不一致", "〔20260919-130722 stdout〕", "")
c(S4, 1502, "平台 B 上 decode 佔總時間 78–82%", "78–82%", "實測", G, "",
  "只有 RUNLOG 發現 18 的文字（prefill 佔 19.1／21.7／17.8%）；任何 run 的 stdout 都沒印 prefill／decode 佔比", "找不到來源",
  "〔git 52803bc^:results/RUNLOG_MI300X.md:755〕", "來源可能是已遺失的 policy_sim.csv 的 prefill_ms／decode_ms 欄〔未查證〕")
c(S4, 1503, "含 decode 的端到端 oracle headroom 為 3.76–4.41%", "3.76–4.41%", "實測", f"{R}/20260919-015643-m5p-summary/stdout.log",
  "20260919-015643-m5p-summary", "summary E 段：toolagent/Llama 3.76、conversation/Llama 4.00、toolagent/Seed 4.41", "來源有疑慮",
  "〔20260918-150003、20260919-015528、20260919-015643〕",
  "4.00 與 4.41 有產生它的 policy-sim run；3.76（Llama toolagent）找不到產生它的 run（20260918-064132-m5p-policy-sim rc=1 KeyError），只出現在讀已遺失 policy_sim.csv 的 summary")
c(S4, 1504, "除以 prefill 佔比推算回 prefill-only 為 18.5–24.8%", "18.5–24.8%", "算術", G, "",
  "3.76/0.191＝19.7、4.00/0.217＝18.4、4.41/0.178＝24.8（算術對）", "來源有疑慮", "〔算術〕",
  "分母（prefill 佔比）找不到 run 來源；且 18.4 四捨五入不是 18.5")
c(S4, 1505, "prefill-only 與端到端相差約 5 倍", "約 5×", "算術", G, "", "1/0.217～1/0.178＝4.6–5.6×", "來源有疑慮", "〔算術〕", "同上，依賴找不到來源的 prefill 佔比")
c("圖 b-headroom 說明", 1511, "來源：results/m4_oracle_mi300x/oracle_*.csv", "引用檔", "設定", "results/m4_oracle_mi300x/oracle_*.csv", "",
  "git 只有 cost_model_*.json 與 simulator_validation.json；oracle_*.csv 不在 git 也不在本機", "找不到來源", "〔git ls-tree 52803bc^〕",
  "數字可由 m4 pressure run 的 stdout 重建（stdout 印「wrote …oracle_b-llama8b_pressure.csv」）")
c("圖 b-prefill 說明", 1518, "prefill 僅佔總時間的 18–22%", "18–22%", "實測", G, "", "同 1502：只有 RUNLOG 文字 17.8–21.7%", "找不到來源",
  "〔git 52803bc^:results/RUNLOG_MI300X.md:755〕", "")
c("圖 b-prefill 說明", 1519, "來源：results/m5_predictor_mi300x/policy_sim.csv", "引用檔", "設定", "results/m5_predictor_mi300x/policy_sim.csv", "",
  "git 只有 m5_predictor_mi300x/features_index.json；policy_sim.csv 不在 git 也不在本機", "找不到來源", "〔git ls-tree 52803bc^〕",
  "部分數字留在 policy-sim 與 summary run 的 stdout")

S5 = "§b-quality"
c(S5, 1525, "四套評測 × 五種 KV 精度", "4×5", "設定", EV, "", "撈針、GSM8K、LongBench、RULER；bf16/fp8/fp8_ptk/int8/int4", "一致", "〔m5／m5c run stdout〕", "")
c(S5, 1527, "Llama、UltraLong、Qwen3、Seed 在 64K–129K 撈針 100%（含 INT4）", "100%", "實測", EV,
  "20260916-062139-m5-b-llama8b-needle;20260917-103133-m5-llama8b-needle-131k;20260917-163616-m5-ultralong-needle;20260916-084900-m5-b-qwen3moe-needle;20260916-161649-m5-b-seedoss-needle",
  "四個模型五種精度全 20/20：Llama 65,536 與 129,024、UltraLong 129,024、Qwen3 131,072、Seed 65,536（要求的 ctx）", "一致",
  "〔5 個 needle run stdout〕", "原始逐題 CSV 已遺失，只剩 stdout 摘要；Llama 129K 的 FP8 輸出與 BF16 逐字元相同 20/20（F3 在查）")
c(S5, 1528, "LongBench 與 RULER 的五精度差距皆在 ±1 分內（64K–129K）", "±1 分", "實測", EV,
  "20260917-015936-m5c-b-llama8b-longbench;20260917-023056-m5c-b-llama8b-ruler;20260918-063345-m5c-b-ultralong8b-1m-longbench;20260918-070443-m5c-b-ultralong8b-1m-ruler;20260918-074128-m5c-b-qwen3-30b-a3b-longbench;20260918-082150-m5c-b-qwen3-30b-a3b-ruler;20260917-045616-m5c-b-seedoss36b-longbench;20260917-070425-m5c-b-seedoss36b-ruler",
  "(1) 長度不對：LongBench 32K 視窗、RULER 16K，不是 64K–129K。(2) 對 BF16 超過 ±1 的巨觀平均有 6 格：Llama RULER INT4 +1.26；UltraLong LB INT4 −1.24、RULER INT4 −1.16、RULER FP8-ptk +1.07；Qwen3 LB INT4 −1.03；Seed RULER FP8-ptk −1.18。最大－最小：Llama LB 0.80／RULER 1.97、UltraLong 1.39／2.23、Qwen3 1.21／0.63、Seed 0.67／2.07",
  "不一致", "〔8 個 m5c run stdout 巨觀平均列〕",
  "另外 Seed-OSS 的 BF16 絕對分數異常低（LB 20.09、RULER 46.18，其他模型 53–61／91–95），「無損」對它沒有鑑別力；LB n＝7×25、RULER n＝7×20；這些 run 有 CONTAMINATED 檔（pid −1 teardown），分數照規則保留")
c(S5, 1530, "Qwen2.5-7B-1M 撈針自 BF16 100% 跌至 FP8 10%、INT4 0%", "100→10／0%", "實測", f"{R}/20260916-065920-m5-b-qwen7b-needle/stdout.log",
  "20260916-065920-m5-b-qwen7b-needle", "ctx 131,072：bf16 20/20、fp8 2/20、fp8_ptk 0/20、int8 18/20、int4 0/20", "一致", "〔20260916-065920 stdout〕", "")
c(S5, 1531, "退化生成：重複同一字元、空字串、700 token 的無意義輸出", "700 token", "實測", G, "",
  "RUNLOG 發現 15 的逐題表（FP8 446 token、FP8-ptk 0、INT4 699 token），取自已遺失 needle_b-qwen7b-1m.csv 的 pred 欄", "來源有疑慮",
  "〔git 52803bc^:results/RUNLOG_MI300X.md:649〕", "stdout 沒有逐題輸出，無法重新核對")
c(S5, 1532, "且在 ctx=3K 就已發生（非長度效應）", "3K", "實測", f"{R}/20260917-095033-m5-qwen7b-needle-4096/stdout.log",
  "20260917-095033-m5-qwen7b-needle-4096", "最短的量測是要求 ctx=4,096：fp8 15%、fp8_ptk 0%、int8 90%、int4 0%；任何 stdout 都沒有「3K」", "找不到來源",
  "〔20260917-095033 stdout〕", "實際 prompt_tokens 只記在已遺失的 CSV；結論（短 ctx 也崩）成立，但數字應寫 4K（要求值）")
c(S5, 1532, "平台 A 同家族：BF16 100%／FP8 5%／INT8 95%／INT4 0%", "100/5/95/0", "實測", "results/m8_directions/d6_quality_inventory.csv",
  "20260831-181930-m5-needle", "D6 重算：qwen-awq 32K，BF16 20/20、FP8 1/20、INT8 19/20、INT4 0/20", "一致", "〔20260831-181930-m5-needle；D6 §4.5〕",
  "平台 A，AWQ 權重；B 的同家族是 BF16 權重，INT8 在 129K 為 90%")
c(S5, 1537, "GSM8K 在七個模型上的五精度差距皆在 ±5pp 內", "±5pp", "實測", EV,
  "20260917-012817-m5-b-qwen7b-1m-precision;20260917-010336-m5-b-mistral-nemo12b-precision;20260917-181725-m5-ultralong-precision",
  "超過 ±5pp（對 BF16）：Qwen2.5-7B-1M FP8／FP8-ptk／INT4 皆 −90.83pp（0.0%）；Nemo INT4 −7.5；UltraLong FP8-ptk +5.84；Llama FP8 與 INT4 恰 −5.00（邊界）",
  "不一致", "〔7 個 precision run stdout 摘要表〕", "只有 4 個模型（Qwen14B、Qwen3、Seed、Llama 邊界）符合")
c(S5, 1537, "n=120", "120", "設定", EV, "", "7 個 precision run 都是 --n-test 120", "一致", "〔cmd.sh〕", "")
c(S5, 1537, "95% CI 約 ±7pp", "±7pp", "算術", "", "", "單一比例 p≈0.8、n=120：1.96·√(0.16/120)＝±7.2pp", "一致", "〔算術〕",
  "這是單一正確率的 CI；兩設定差值的 CI（未配對）約 ±10pp，配對則看 discordant 數，論文沒算")
c(S5, 1538, "只看推理型評測會得出量化免費的錯誤結論，而同一批設定在檢索任務上可以掉 100pp", "100pp", "實測", EV,
  "20260916-065920-m5-b-qwen7b-needle;20260917-012817-m5-b-qwen7b-1m-precision",
  "平台 B 唯一撈針崩的模型（Qwen2.5-7B-1M）GSM8K 也崩：FP8／FP8-ptk／INT4 0.0%（−90.8pp）；其他 6 個模型撈針 95–100%。B 上沒有任何模型呈現「GSM8K 免費、撈針掉 100pp」",
  "不一致", "〔20260917-012817、20260916-065920 stdout〕", "這個對比只在平台 A（qwen-awq）成立")
c("圖 b-quality 說明", 1544, "同一批精度設定在兩類任務上的 ε 相差兩個數量級", "100×", "實測", EV, "",
  "同上：B 的同一模型兩類任務一起崩或一起不崩", "不一致", "〔m5 needle／precision stdout〕", "")
c("圖 b-quality 說明", 1545, "來源：results/m5_quality_mi300x/{needle,gsm8k_precision}_*.csv", "引用檔", "設定", "results/m5_quality_mi300x/*.csv", "",
  "git 只有 gpu_guard_*.json；CSV 不在 git 也不在本機（stdout 寫「wrote 100 rows -> …」）", "找不到來源", "〔git log --all〕",
  "摘要層級可由 stdout 重建（主 agent F4 負責）")

S6 = "§b-attention"
c(S6, 1552, "七個模型的注意力重要度 AUC 為 0.480–0.500", "0.480–0.500", "實測", "git 52803bc^:results/m5_attention_mi300x/alignment_*.json",
  "20260917-193756-m5c-attn-b-llama8b", "7 個 JSON：0.4799（Qwen14B）–0.4996（Qwen7B）", "一致", "〔git 52803bc^:alignment_*.json〕", "")
c(S6, 1553, "存取歷史為 0.721", "0.721", "實測", "git 52803bc^:results/m5_attention_mi300x/alignment_*.json", "20260917-193756-m5c-attn-b-llama8b",
  "7 個 JSON 都是 0.7214", "一致", "〔alignment_*.json〕", "7 個模型共用同一工作負載，存取歷史 AUC 本來就相同，不是 7 個獨立驗證")
c(S6, 1553, "Spearman ∈ [−0.047, +0.051]", "[−0.047, +0.051]", "實測", "git 52803bc^:results/m5_attention_mi300x/alignment_*.json", "",
  "−0.0469（Qwen14B）～+0.051（Qwen7B）", "一致", "〔alignment_*.json〕", "")
c(S6, 1554, "開頭 attention sink 6.2× 於中位", "6.2×", "實測", G, "20260917-193756-m5c-attn-b-llama8b",
  "RUNLOG 表：位置 0 為 23.59、中間約 3.8 → 6.2×（只有 Llama）", "來源有疑慮", "〔git 52803bc^:results/RUNLOG_MI300X.md:699〕",
  "attn_importance_*.csv 從未進 git、本機已刪；stdout 沒有逐位置表，無法重算")
c(S6, 1555, "前 50 名 block 重疊率 100%", "100%", "實測", G, "", "只有 RUNLOG 發現 17 的文字", "來源有疑慮", "〔git 52803bc^:results/RUNLOG_MI300X.md:720〕", "同上")

c("圖 b-attention 說明", 1562, "來源：results/m5_attention_mi300x/", "引用檔", "設定", "git 52803bc^:results/m5_attention_mi300x/", "",
  "alignment_*.json（7 個）在 git 52803bc^ 有，工作樹已刪；attn_importance_*.csv 從未進 git、本機也沒有", "來源有疑慮", "〔git ls-tree 52803bc^〕",
  "AUC／Spearman 可追；逐位置注意力形狀（右圖）無法重建")

S7 = "§b-negative 表"
c(S7, 1576, "toolagent／Llama：oracle 3.76%、對稱 −33.2%、成本加權 −35.0%", "3.76／−33.2／−35.0", "實測", f"{R}/20260919-015643-m5p-summary/stdout.log",
  "20260919-015643-m5p-summary", "summary E 段（W=79,439）：3.76%、sym −33.2%、cost −35.0%", "來源有疑慮", "〔20260919-015643 stdout〕",
  "找不到產生它的 policy-sim run：20260918-064132-m5p-policy-sim 失敗（KeyError 'nvme'）；summary 讀的是已遺失的 policy_sim.csv")
c(S7, 1577, "合成長上下文／Llama：8.33%、−47.3%、−29.7%", "8.33／−47.3／−29.7", "實測", f"{R}/20260919-015314-m5p-policy-lczipf-b/stdout.log",
  "20260919-015314-m5p-policy-lczipf-b", "oracle 8.33%；sym −47.32%；cost −29.65%", "一致", "〔20260919-015314 stdout〕",
  "前一版設定（20260918-145958，120 請求）oracle headroom 是 0.00%，之後改成 60 文件／600 請求才有 8.33%；此設定下預測器 AUC 只有 0.47–0.49")
c(S7, 1578, "conversation／Llama：4.00%、−45.3%、−43.3%", "4.00／−45.3／−43.3", "實測", f"{R}/20260918-150003-m5p-policy-conversation/stdout.log",
  "20260918-150003-m5p-policy-conversation", "oracle 4.00%；sym −45.26%；cost −43.29%", "一致", "〔20260918-150003 stdout〕", "")
c(S7, 1579, "toolagent／Seed-OSS：4.41%、−9.7%、−9.6%", "4.41／−9.7／−9.6", "實測", f"{R}/20260919-015528-m5p-policy-seedoss/stdout.log",
  "20260919-015528-m5p-policy-seedoss", "oracle 4.41%；sym −9.68%；cost −9.58%", "一致", "〔20260919-015528 stdout〕", "")
c(S7, 1584, "最佳 baseline 為 tier_fs", "tier_fs", "實測", f"{R}/20260919-015643-m5p-summary/stdout.log", "20260919-015643-m5p-summary",
  "四組都是 tier_fs", "一致", "〔summary E 段〕", "")
c(S7, 1584, "平台 A 於同類工作負載為 +81.85%", "+81.85%", "實測", "git 52803bc^:results/m5_predictor/policy_sim.csv",
  "20260905-214425-m5p-policy-toolagent-qwen-awq", "A 那列：lc128kz、qwen-awq、decode=0（只算 prefill）、tiara_sym_l2 拿到 81.85%", "來源有疑慮",
  "〔git 52803bc^:results/m5_predictor/policy_sim.csv:6〕", "數字對得上，但口徑不同：A 不含 decode、模型 qwen-awq；B 含 decode、Llama-8B。不是「同類」")

S8 = "§b-negative 內文"
c(S8, 1586, "預測器 AUC 0.917–0.922", "0.917–0.922", "實測", f"{R}/20260919-015643-m5p-summary/stdout.log", "20260919-015643-m5p-summary",
  "只有 conversation 是 0.9172–0.9216；toolagent 0.992–0.9995；合成長上下文（表中第 2 列）0.26–0.49（比隨機差）", "不一致",
  "〔summary B、D2 段〕", "「預測器本身並無問題」只對 conversation 成立")
c(S8, 1586, "ECE 0.003–0.004", "0.003–0.004", "實測", f"{R}/20260919-015643-m5p-summary/stdout.log", "20260919-015643-m5p-summary",
  "conversation 0.0033–0.0041；toolagent 0.0004–0.0025；lc128kz 0.235–0.388", "不一致", "〔summary B 段〕", "同上")
c(S8, 1586, "正樣本 Spearman 0.99", "0.99", "實測", f"{R}/20260919-015643-m5p-summary/stdout.log", "20260919-015643-m5p-summary",
  "conversation 0.991；toolagent 0.245–0.274；lc128kz −0.238–0.057", "不一致", "〔summary B 段〕", "同上")
c(S8, 1587, "推論延遲數百 µs／64 個候選", "數百 µs", "實測", f"{R}/20260919-015643-m5p-summary/stdout.log", "20260919-015643-m5p-summary",
  "248.4 µs／64 候選（conversation sym_l2，loadavg 7.7）", "一致", "〔summary F 段〕", "")
c(S8, 1587, "本地 NVMe 循序讀達 5,392 MiB/s", "5,392 MiB/s", "實測", f"{R}/20260916-030747-m3-diskbw/stdout.log", "20260916-030747-m3-diskbw",
  "/var/tmp 讀中位 5,392 MiB/s（3 次 4,771／5,527／5,392）", "一致", "〔20260916-030747-m3-diskbw stdout〕", "/var/tmp 是容器 overlay，不確定是專用 NVMe〔未查證〕")
c(S8, 1588, "GPU 有 192 GB HBM", "192 GB", "外部規格", "git 52803bc^:results/hw_mi300x.json", "", "192.0 GiB", "一致", "〔hw_mi300x.json〕", "單位應為 GiB")
c(S8, 1588, "四個 baseline 的端到端總時間僅相差 1.6%", "1.6%", "實測", f"{R}/20260919-015643-m5p-summary/stdout.log", "20260919-015643-m5p-summary",
  "只有 toolagent／Seed 是 1.62%；toolagent／Llama 2.54%、conversation 2.87%、合成長上下文 76.1%（full_gpu −76.08%）", "不一致",
  "〔summary E 段〕", "")
c(S8, 1590, "Seed 損失最小（−9.7% 對 Llama 的 −33.2%）", "−9.7／−33.2", "實測", f"{R}/20260919-015643-m5p-summary/stdout.log", "20260919-015528-m5p-policy-seedoss",
  "−9.68%；−33.2% 見 1576", "來源有疑慮", "〔summary E 段〕", "−33.2% 找不到產生它的 run（同 1576）")

S9 = "§b-lossless"
c(S9, 1599, "60 題 GSM8K", "60", "設定", EV, "", "lossless run 皆 --n-test 60", "一致", "〔cmd.sh〕", "")
c(S9, 1599, "CPU 階與磁碟階之間 0/60 不同", "0/60", "實測", "git 52803bc^:results/CLAIM_EVIDENCE_MI300X.md", "",
  "stdout 只印「各設定對 full_gpu」的相同數（例 Llama cpu_lru 42/60、tier_fs 42/60），沒有 CPU 對磁碟的逐題比對", "來源有疑慮",
  "〔git 52803bc^:results/CLAIM_EVIDENCE_MI300X.md C2〕", "逐題比對要用已遺失的 lossless_*.csv；兩者對基準的相同數一樣，只是相容，不是證明")
c(S9, 1600, "同一設定重跑兩次亦為 0/60", "0/60", "實測", f"{R}/20260916-223132-m5-llama8b-lossless-rep2/stdout.log", "20260916-223132-m5-llama8b-lossless-rep2",
  "rep2 run 存在，兩次的 full_gpu 正確率都是 47/60；但跨 run 逐題比對要用已遺失的 CSV", "來源有疑慮", "〔20260916-223132 stdout〕", "")
c(S9, 1600, "卸載相對不卸載有 13–25/60 的輸出位元不同", "13–25/60", "實測", EV, "",
  "不同數＝60－相同：Llama 18、Qwen7B 20、Qwen3 25、Seed 16、Qwen14B 14、Nemo 13、UltraLong 18", "一致", "〔7 個 lossless run stdout〕", "")
c(S9, 1601, "最終答案差異 ≤1 題", "≤1 題", "實測", EV, "",
  "正確題數差 ≤1（例 Llama 47→46）成立；但逐題「最終答案不同」Llama 是 8/120（每設定約 4/60，RUNLOG 發現 13）", "不一致",
  "〔lossless stdout；git 52803bc^:results/RUNLOG_MI300X.md:571〕", "措辭要改成「正確題數差 ≤1」")
c(S9, 1602, "平台 A 當時量到 60/60 完全相同", "60/60", "實測", "git 52803bc^:results/m5_quality/gsm8k_lossless.csv", "20260830-231251-m5-lossless",
  "cpu_lru 60/60、tier_fs 60/60 的 out_sha1 與 full_gpu 相同", "一致", "〔git 52803bc^:results/m5_quality/gsm8k_lossless.csv〕",
  "A 的 tier_fs CPU 階 24 GiB ≥ 工作集，磁碟階可能沒被用到（CLAIM_EVIDENCE D1）")

S10 = "附錄平台 B 設計"
c(S10, 1917, "MI300X（gfx942，192 GB HBM3）", "gfx942", "設定", "git 52803bc^:results/env_mi300x.json", "", "gfx_arch=gfx942", "一致", "〔env_mi300x.json〕", "")
c(S10, 1917, "ROCm 7.x", "7.x", "設定", "git 52803bc^:results/env_mi300x.json", "", "7.2.2", "一致", "〔env_mi300x.json〕", "")
c(S10, 1917, "LMCache ROCm wheel（gfx942）", "LMCache", "設定", G, "", "LMCache 在平台 B 未能執行（NOT_MEASURED，依賴 CUDA 專屬套件）", "不一致",
  "〔git 52803bc^:results/RUNLOG_MI300X.md:501〕", "附錄仍寫成可用")
c(S10, 1925, "8B–14B 模型於 128K 僅佔 HBM 的 8–10%", "8–10%", "算術", "", "",
  "Llama 128 KiB×128,000＝15.6 GiB；Qwen3-14B 160 KiB×128,000＝19.5 GiB；÷192 GiB（實際 vram）＝8.1–10.2%；若 192 GB 換成 178.8 GiB 則 8.7–10.9%",
  "一致", "〔算術〕", "GiB／GB 混用；平台 B 實際的 14B 模型是 Qwen2.5-14B（192 KiB/token → 23.4 GiB → 12%）")
c(S10, 1928, "主軸 128K、256K、512K、1M 的單請求上下文", "128K–1M", "設定", "", "",
  "實際：品質評測最長 129K（撈針），LongBench／RULER 只在 32K／16K；1M 只有 UltraLong 的重算掃描", "來源有疑慮", "〔m5／m2 run cmd.sh〕",
  "這是「尚未執行」附錄的設計文字，但 B 已執行且軸不同")
c(S10, 1930, "模型規模 8B 至 70B", "8B–70B", "設定", "", "", "實際最大 36B（Seed-OSS）；70B 從未在 B 上跑", "不一致", "〔m1-b-* run 清單〕", "")

S11 = "表 tab:models"
c(S11, 1948, "平台 A：Qwen2.5-7B-Inst-1M κ＝108", "108", "推算", "main.tex 表 tab:costmodels；git 52803bc^:results/RUNLOG.md", "",
  "實測（表 tab:costmodels，qwen-awq NVMe）：Drop 基底 3.546 ÷ CPU 0.298＝11.9", "不一致", "〔算術：main.tex:1680–1682〕",
  "108／54 是舊的「簡化 prefill 模型」算術估計（54–190 倍範圍）；與實測 κ 差約 9 倍")
c(S11, 1948, "Qwen 懸崖在 315K", "315K", "實測", "git 52803bc^:results/m1_capacity/capacity.csv", "20260830-232943-m1-qwen-awq",
  "實測容量 273,872（measure）；314,952 是 1.15× 的 verify_over 探測（啟動失敗）", "不一致", "〔git 52803bc^:results/m1_capacity/capacity.csv〕",
  "315K 是「放不下」的探測值，不是懸崖；1958 行同")
c(S11, 1949, "平台 A：Llama-3.1-8B-Inst κ＝54（減半）", "54", "推算", "main.tex 表 tab:kappa", "", "同一篇論文表 tab:kappa 的實測 κ_cpu＝8.9（SATA）／9.5（NVMe）",
  "不一致", "〔main.tex:304–305〕", "舊算術估計，與表 tab:kappa 自相矛盾")
c(S11, 1941, "平台 A 的兩個模型 κ 相差 2 倍", "2×", "算術", "main.tex 表 tab:costmodels", "",
  "108/54＝2（估計值）；用實測常數：qwen-awq 11.9 對 llama-bf16 NVMe 7.4（4.008/0.543）→ 1.6×", "不一致", "〔算術：main.tex:1680–1682〕", "")
c(S11, 1952, "平台 B：Llama-4-Scout κ＝8", "8", "推算", "git 52803bc^:results/model_survey/MODEL_SELECTION_MI300X.md", "",
  "Llama-4-Scout 從未在 B 上跑；選模調查寫它 BF16 權重 202.4 GiB，單卡放不下", "找不到來源",
  "〔git 52803bc^:results/model_survey/MODEL_SELECTION_MI300X.md:19,152〕", "B 實際的 7 個模型不含它")
c(S11, 1952, "Llama-4-Scout 原生 10M", "10M", "外部規格", "git 52803bc^:results/model_survey/MODEL_SELECTION_MI300X.md", "",
  "選模表：config 10M（iRoPE）", "一致", "〔MODEL_SELECTION_MI300X.md:152〕", "模型卡規格，非實驗數字")
c(S11, 1953, "平台 B：Qwen3-30B-A3B κ＝3", "3", "推算", G, "",
  "B 的 Qwen3 κ_cpu 被標為異常不採用（CPU delta −49 ms，發現 11）；κ_ssd＝5.77；沒有任何來源給出 3", "找不到來源",
  "〔git 52803bc^:results/RUNLOG_MI300X.md:381〕", "")
c("附錄平台 B 設計", 1958, "Llama-3.1-8B 在同平台的容量懸崖約 132K", "132K", "實測", "git 52803bc^:results/m1_capacity/capacity.csv", "20260830-232507-m1-llama-awq",
  "平台 A 實測：Llama AWQ 120,320（verify_over 138,368 失敗）；BF16 41,648", "不一致", "〔capacity.csv〕", "132K 找不到；這是平台 A 的數字，列在此表的討論中")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
hdr = ["claim_id", "section", "line", "claim_text", "paper_value", "kind", "source_path", "source_run_id", "evidence_value", "status", "tag", "note"]
allowed_status = {"一致", "不一致", "找不到來源", "來源有疑慮"}
allowed_kind = {"實測", "算術", "外部規格", "設定", "推算"}
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(hdr)
    for i, row in enumerate(C, 1):
        assert row[8] in allowed_status, row
        assert row[4] in allowed_kind, row
        w.writerow([f"B-{i:03d}"] + list(row))
from collections import Counter
cnt = Counter(r[8] for r in C)
print(f"wrote {len(C)} claims -> {OUT}")
print(dict(cnt))
print("audit_run_id", os.environ.get("RUN_ID", "NO_RUN_ID"), datetime.now(timezone.utc).isoformat(timespec="seconds"))
