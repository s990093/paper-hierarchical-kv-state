# P08　直接丟掉 token／block 的文獻（14 號 DS 用；10/19 整理；摘要級，self-review，未逐頁核對）

> 使用者 10/19：「從直接刪掉 token 做大規模掃描，跟看各種 paper」。這張卡是 14 §8 的文獻依據。
> 分數的定義以 NVIDIA kvpress 0.5.5（Apache-2.0；`pip download kvpress==0.5.5`，只讀原始碼）的實作為準，因為它把多數方法放在同一個介面下，比各自的原始程式好對齊。

## 1. 分類

| 類 | 方法 | 分數（kvpress 的寫法） | 什麼時候決定 | 粒度 | 我們怎麼做（14 §8） |
|:--|:--|:--|:--|:--|:--|
| 位置 | StreamingLLM（Xiao et al., ICLR 2024, arXiv 2309.17453） | 留前 n_sink＝4 個 token，從第 5 個開始丟 | 寫入時 | token | STREAM |
| 觀察窗 attention | SnapKV（Li et al., NeurIPS 2024, arXiv 2404.14469） | 最後 window＝64 個 query 對前面的 attention 平均，avg-pool kernel 5，GQA 組內平均；觀察窗保留 | 寫入時（kvpress 預設把問題分開，所以是 query-agnostic）；放了問題就是 query-aware | per (層, head) token | SNAP；問題當觀察窗＝QRY |
| 單一 query | TOVA（Oren et al., arXiv 2401.06104） | 最後一個 token 的 attention，所有 head 平均 | 寫入時 | token | TOVA |
| 累積 attention | H2O（Zhang et al., NeurIPS 2023, arXiv 2306.14048）；kvpress ObservedAttention | 所有 query 的 attention 加總 ÷ 看得到的 query 數 | 寫入時（要 eager attention） | token | H2O（只用 8 個均勻觀察窗，近似） |
| 每層預算 | PyramidKV（Cai et al., arXiv 2406.02069） | SnapKV 分數，低層多留、高層少留（beta 20） | 寫入時 | per (層, head) | PYR |
| 每個 head 預算 | AdaKV（Feng et al., arXiv 2407.11550） | 同一層的 head 一起挑，每個 head 至少留 20% | 寫入時 | per (層, head) | ADA |
| 預期 attention | Expected Attention（kvpress，2025） | 全部 query 的平均＋共變異數，套上接下來 512 個位置的平均 RoPE，exp(K·μ/√d＋½KᵀΣK/d) 的 softmax × ‖V‖ | 寫入時 | token | EXPATT |
| K 的幾何 | KNorm（Devoto et al., arXiv 2406.11430） | −‖K‖ | 寫入時 | token | KNORM |
| K 的幾何 | KeyDiff（arXiv 2504.15364） | −cos(K, 平均方向) | 寫入時 | token | KEYDIFF |
| 重建 | KVzip（Kim et al., arXiv 2505.23416） | 讓模型重建整段 context 時每個 KV 拿到的最大 attention | 寫入時、和問題無關（為多次重用設計） | token | **不做**（一題要再跑一次 32K 並記下 attention） |
| 逐步選 | Quest（Tang et al., ICML 2024）、InfiniGen（OSDI 2024）、ShadowKV | 每步用 query 估每個 page 的上界，只讀 top-k | 讀的時候（每步） | page／token | 不丟，所以不在這裡；QRY 是「知道問題」的上限 |
| 學出來的 | DuoAttention、Q-Filters、KVP（ICML'26）、ForesightKV（ICML'26）、LookaheadKV（ICLR'26）、TRIM-KV | 要訓練或每個模型的參數 | — | — | **不做**（CLAUDE.md §8） |

## 2. 和我們的問題最相關的一件事：多輪

- **SCBench**（Li et al., ICLR 2025, arXiv 2412.10319）：同一段 context 接好幾個問題（多輪、多請求）。記憶體少於 O(n) 的方法（會丟資料的）在多輪表現差，O(n) 記憶體的稀疏方法比較穩。
- 我們的情境就是多輪：KV 在這一輪結束時存，下一輪換問題。所以「寫入時」的挑法（SNAP、TOVA、H2O、EXPATT、KNORM…）不知道下一個問題；QRY 知道第 1 輪的問題，但第 2 輪換了。
- 14 §8.5 的判準 3（第 2 輪減第 1 輪）就是直接量這件事。
- 我們自己已經看到的：09 海選和 43 複賽裡，「寫入時的 attention」挑哪些 block 壓成低精度，沒有贏隨機（08 §2.9）。

## 3. 和實作有關的差異（照實寫）

- 我們把被丟的位置用遮罩擋掉，不真的縮短 cache；RoPE 位置不重排（kvpress 的 StreamingLLM 也預設不重排）。
- kvpress 是在 prefill 時就壓縮；我們在 prefill 完、還原的時候才挑。觀察窗的 query 一樣是 context 最後的 token，結果相同。
- H2O 只用 8 個觀察窗，不是全部 query。
- P、B 粒度（所有層、head 丟同一組 page／block）是我們加的，因為分層儲存一次搬的是 page／block。文獻多是 per-head token。
