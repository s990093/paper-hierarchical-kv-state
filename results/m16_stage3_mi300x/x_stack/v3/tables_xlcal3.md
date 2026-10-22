# xKV 事先校準的基底 vs 每題 SVD：加題數（10/21 第三輪；只比 INT4 的兩對；判定事先寫死，見最後）

**整體判定（新的 192 題＝獨立檢驗）：llama UNDETERMINED；qwen UNDETERMINED**（每個模型：{"llama": "UNDETERMINED", "qwen": "UNDETERMINED"}）；合併 288 題（參考）：{"llama": "UNDETERMINED", "qwen": "UNDETERMINED"}

38c 的 run：12 個；新的 run：20261022-013407-m16-xlcal3-llama-s0、20261022-013407-m16-xlcal3-llama-s1、20261022-013407-m16-xlcal3-llama-s2、20261022-013407-m16-xlcal3-llama-s3、20261022-013407-m16-xlcal3-qwen-s0、20261022-013407-m16-xlcal3-qwen-s1、20261022-013407-m16-xlcal3-qwen-s2、20261022-020808-m16-xlcal3-llama-s4、20261022-021229-m16-xlcal3-qwen-s3、20261022-021424-m16-xlcal3-llama-s5、20261022-021759-m16-xlcal3-qwen-s4、20261022-022029-m16-xlcal3-llama-s6、20261022-023055-m16-xlcal3-llama-s7、20261022-023845-m16-xlcal3-qwen-s5、20261022-025711-m16-xlcal3-llama-s8、20261022-025716-m16-xlcal3-llama-s9、20261022-030411-m16-xlcal3-qwen-s6、20261022-031319-m16-xlcal3-llama-s7、20261022-031319-m16-xlcal3-qwen-s5、20261022-031320-m16-xlcal3-llama-s10、20261022-031320-m16-xlcal3-llama-s8、20261022-031320-m16-xlcal3-llama-s9、20261022-031320-m16-xlcal3-qwen-s6、20261022-031320-m16-xlcal3-qwen-s7、20261022-032025-m16-xlcal3-llama-s11；不齊全的題（沒用）：38c 0、新 0

校準基底（沿用 38c，不重算）：llama sha256 9433330226f5575f6e0e5ba998bf0bcbb4bf35821057cb329c9f7afba642fa2e（＝ 38c 記的；calib run 20261021-123131-m16-xlcal-calib-llama）；qwen sha256 d291ae6dc16ea8c0a17a94ae551d325d267a992bde6c0a7a593f09729f532392（＝ 38c 記的；calib run 20261021-123131-m16-xlcal-calib-qwen）

## llama

| 校準版 vs 每題 SVD | 只用 38c 的 96 題（X2 的題） | 只用新的 192 題（seed 66000 起；主要判定） | 合併 288 題（參考：含已看過的 96 題） |
|:--|--:|--:|--:|
| xl:k22:v34:cal+q:int4_kivi vs xl:k22:v34+q:int4_kivi | +2.6 [-1.6, +7.5] CAN_PRECOMPUTE（n 96） | +2.1 [+0.0, +4.4] CAN_PRECOMPUTE（n 192） | +2.3 [+0.2, +4.5] CAN_PRECOMPUTE（n 288） |
| xl:k28:v42:cal+q:int4_kivi vs xl:k28:v42+q:int4_kivi | +3.5 [-0.6, +8.2] CAN_PRECOMPUTE（n 96） | -1.5 [-4.3, +1.2] UNDETERMINED（n 192） | +0.2 [-2.2, +2.5] UNDETERMINED（n 288） |
| **模型判定** | **CAN_PRECOMPUTE** | **UNDETERMINED** | **UNDETERMINED** |

合併 288 題，每個配置對 E0：

| 配置 | 位元組比 | 答對率 | 對 E0（百分點，CI） | ε |
|:--|--:|--:|--:|:--|
| xl:k22:v34:cal+q:int4_kivi | 0.0684 | 77.5 | -0.3 [-2.0, +1.4] | 過 |
| xl:k22:v34+q:int4_kivi | 0.0961 | 75.2 | -2.5 [-4.8, -0.4] | 沒過 |
| xl:k28:v42:cal+q:int4_kivi | 0.0854 | 76.8 | -1.0 [-2.9, +0.7] | 沒過 |
| xl:k28:v42+q:int4_kivi | 0.1202 | 76.6 | -1.2 [-2.9, +0.4] | 沒過 |

位元組（合併）：xl:k22:v34:cal+q:int4_kivi vs xl:k22:v34+q:int4_kivi：每請求 0.068359／0.096137、共用基底 117.4 MB、攤提 N＝1／8／64：0.096137／0.071832／0.068793；xl:k28:v42:cal+q:int4_kivi vs xl:k28:v42+q:int4_kivi：每請求 0.085449／0.120171、共用基底 146.8 MB、攤提 N＝1／8／64：0.120171／0.089789／0.085992

## qwen

| 校準版 vs 每題 SVD | 只用 38c 的 96 題（X2 的題） | 只用新的 192 題（seed 66000 起；主要判定） | 合併 288 題（參考：含已看過的 96 題） |
|:--|--:|--:|--:|
| xl:k22:v34:cal+q:int4_kivi vs xl:k22:v34+q:int4_kivi | -2.1 [-6.0, +2.0] UNDETERMINED（n 96） | -1.3 [-4.1, +1.5] UNDETERMINED（n 192） | -1.6 [-3.8, +0.7] UNDETERMINED（n 288） |
| xl:k28:v42:cal+q:int4_kivi vs xl:k28:v42+q:int4_kivi | -2.2 [-5.5, +1.1] UNDETERMINED（n 96） | -0.5 [-3.2, +2.3] UNDETERMINED（n 192） | -1.1 [-3.2, +1.1] UNDETERMINED（n 288） |
| **模型判定** | **UNDETERMINED** | **UNDETERMINED** | **UNDETERMINED** |

合併 288 題，每個配置對 E0：

| 配置 | 位元組比 | 答對率 | 對 E0（百分點，CI） | ε |
|:--|--:|--:|--:|:--|
| xl:k22:v34:cal+q:int4_kivi | 0.0684 | 73.3 | -2.3 [-4.5, -0.2] | 沒過 |
| xl:k22:v34+q:int4_kivi | 0.0822 | 74.9 | -0.8 [-2.8, +1.2] | 沒過 |
| xl:k28:v42:cal+q:int4_kivi | 0.0854 | 74.4 | -1.3 [-3.3, +0.9] | 沒過 |
| xl:k28:v42+q:int4_kivi | 0.1028 | 75.5 | -0.2 [-2.2, +1.9] | 沒過 |

位元組（合併）：xl:k22:v34:cal+q:int4_kivi vs xl:k22:v34+q:int4_kivi：每請求 0.068359／0.082248、共用基底 25.7 MB、攤提 N＝1／8／64：0.082248／0.070095／0.068576；xl:k28:v42:cal+q:int4_kivi vs xl:k28:v42+q:int4_kivi：每請求 0.085449／0.10281、共用基底 32.1 MB、攤提 N＝1／8／64：0.10281／0.087619／0.08572

## 事先寫死的判定與改動

- **rule**：判定（生成答對率、X2 的 96 題、逐題配對：校準 − 每題 SVD，bootstrap 10,000 次的 95% CI）：CI 上界 < −2 個百分點 → MUST_PER_ITEM（必須每題算 SVD）；CI 下界 ≥ −2 個百分點 → CAN_PRECOMPUTE（可以事先算好基底）；其他 → UNDETERMINED（題數不夠分辨）。
- **aggregate**：（10/21 改動：使用者指定只比有 INT4 的兩對，只降秩那對不做）每個模型：**這兩對**都 CAN_PRECOMPUTE → CAN_PRECOMPUTE；任一對 MUST_PER_ITEM → MUST_PER_ITEM；其他 → UNDETERMINED；題數 < 48 → INCOMPLETE。整體：兩個模型都 CAN → CAN；任一模型 MUST → MUST；其他照實寫。原文（XLCAL_TEXT['aggregate']）是三對；判定門檻（ε＝2 個百分點、CI 上下界的規則）不變。
- **variants**：並列三種題組：只用 38c 的 96 題、只用新的 192 題、合併 288 題（每題四個配置＋E0 都有、同一題同一行程配對；兩組題的 seed 不重疊）。主要判定讀新的 192 題（獨立檢驗）；合併 288 題只當參考（含看過結果之後才限定兩對的 96 題）。10/21 在看到新資料之前改定。
- **caveat**：「只比 INT4 的兩對」是看到 38c 的結果之後決定的（使用者 10/21）。38c 的 96 題在兩對的 aggregate 下，Llama 會從 UNDETERMINED 變成 CAN_PRECOMPUTE（差別全在拿掉的只降秩那對：−3.7 [−7.6, +0.2]）；所以合併 288 題含了已經看過的 96 題，**「只用新的 192 題」才是事先寫死、沒看過資料的獨立檢驗**——兩者不一致時照實並列，判讀由人。
- **design**：新題 seed 66000 起、T2–T5 各 48 題（和 X1／X2／X2b／X4a／X4b／校準／C1a（X4d）／44c 都不重疊，self-test 核對）；GPU 用 m16_xstack.py gen（和 38c 同一條路徑：每題 SVD 是每個配置各做一次，--xl-cache 在這兩對沒有可共用的 SVD、等於沒作用）；校準基底沿用 38c 的 calib_xl_<m>.pt（sha256 見 check-calib）。
- **bytes**：同秩時，校準版的每請求位元組＝每題 SVD 版減掉每題的基底（基底跨請求共用，列在 bytes_shared，大小和每題的基底相同）。所以 CAN_PRECOMPUTE＝同秩不比每題 SVD 差、而且每請求更省。攤提〔算術〕：共用基底 S 由 N 個請求（或同一個請求的多次還原）分攤，每請求多 S/N；N＝1 時正好等於每題 SVD 版（打平點恆為 1），report 列 N＝1／8／64 的攤提後位元組比。實際部署只要存最高秩的一份基底（低秩取前幾欄），S 以最高秩那份為準會更大，這裡照每個配置自己的秩算。
