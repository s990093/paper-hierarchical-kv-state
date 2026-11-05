# 文獻調查（2026-11-05；4 個 subagent 讀網頁，self-review，非 cross-model）

> 每一條都附來源；〔判讀〕是我們的解讀。沒查到的寫「沒查到」。

## A　快取模擬怎麼加速（精確 vs 近似）

| 技術 | 精確？ | 文獻報告的數字 | 對我們（TTFT、多層、session＋共用前綴、非堆疊規則）會不會失效 |
|:--|:--|:--|:--|
| Mattson 堆疊演算法 | 精確，限包含性規則 | 只適用 LRU／MRU／OPT 類 | 失效：只給命中率；ARC／S3-FIFO／TTL 不是堆疊式 |
| SHARDS（FAST'15） | 近似 | 抽樣 0.1% 平均絕對誤差 <0.02；記憶體最多省 10,800×、時間 204× | 只給 LRU 的 MRC；共用前綴與頻寬爭用失真〔推論〕 |
| Miniature simulation（ATC'17） | 近似 | ARC／LIRS／OPT，R=0.001 時 median MAE <0.005 | 可跑任意規則，但頻寬要按比例縮放，沒驗證過延遲 |
| Kosmo（FAST'24） | 近似 | 記憶體比縮小版模擬平均少 3.6× | 只算命中率 |
| TTLs matter（EuroSys'24） | 近似 | TTL-aware MRC | 只處理 TTL |
| 時間切分平行模擬 | 修正後精確 | 修正成本與快取大小成正比 | 計時回饋（Cake 會合點）讓修正難收斂〔推論〕 |
| libCacheSim | 精確 | 自稱 >20M req/s | 不支援 Cake、頻寬、多層 session；資料結構可參考 |
| Set／time 抽樣（Kessler, Hill, Wood 1994） | 近似 | set 抽樣達到 10% 目標、time 抽樣沒有 | **警告：set 之間有互動時不適用**——頻寬爭用就是 session 之間的互動 |
| delayed hits | — | 只看 hit-rate 低估延遲 14–63%（switch）、22–36%（CDN） | 支持我們的 H2：命中率不能代表延遲 |

來源：https://waldspurger.org/carl/papers/shards-fast15.pdf · https://www.usenix.org/conference/atc17/technical-sessions/presentation/waldspurger · https://usenix.org/conference/fast24/presentation/shakiba · https://www.eecg.utoronto.ca/~stumm/Sultan-EuroSys24.html · https://arxiv.org/pdf/1109.6643 · https://informs-sim.org/wsc04papers/041.pdf · https://github.com/1a1a11a/libCacheSim · https://ftp1.cs.wisc.edu/markhill/Papers/toc94_sampling.pdf · https://cs.utexas.edu/~venkatar/sys_perf_analysis/belatedly.pdf

## B　兩階段／多保真篩選

- 我們的模擬是**確定性的**：低保真的誤差是系統性偏差，不是雜訊 → OCBA／ranking-and-selection 的「重複次數→機率保證」不能直接套。
- 對得上的理論：非隨機 successive halving（Jamieson & Talwalkar 2015：低保真誤差包絡 γ < 差距的一半才保證選對；作者明說「永遠無法驗證某臂是最佳」）；MF-UCB（Kandasamy 2016：偏差上界 ζ 已知時，差距 > 2γ 才排除）。→ **都要先用完整版校準偏差上界**。
- 問題形狀最像的是「可行性判定／thresholding bandit」（只要判定高於或低於門檻）。
- 不對稱：錯丟最快的對手 → 假 GO（危險）；錯丟 oracle 那邊 → 偏向 NO-GO（保守）。
- 反例：Kareto 的剪枝沒有保證，mean TTFT 最大偏差 19.2%。

來源：https://arxiv.org/pdf/1502.07943 · https://proceedings.neurips.cc/paper/2016/hash/2ba596643cbbbc20318224181fa46b28-Abstract.html · https://www.informs-sim.org/wsc99papers/050.PDF · https://informs-sim.org/wsc14papers/includes/files/350.pdf · https://proceedings.mlr.press/v48/locatelli16.html · https://infoscience.epfl.ch/entities/publication/0da736c4-43f4-49c1-adec-1dde3368a035 · https://arxiv.org/html/2603.08739v1

## C　LLM serving／KV 模擬器（最接近的工作）

| 工作 | 做什麼 | 驗證 | 和我們的差別〔判讀〕 |
|:--|:--|:--|:--|
| Vidur（MLSys'24） | 運算子 profiling＋排程；Vidur-Search 一小時找部署設定 | 延遲誤差 <9% | 沒有 CPU／SSD 卸載、跨層逐出 |
| LLMServingSim 2.0（ISPASS'26 Best Paper） | radix tree 前綴快取、HBM→host→SSD／CXL | 對真機平均誤差 0.95% | 幫部署找設定；沒有 session 級完美預測、沒有 Cake |
| AgentServeSim（NeurIPS'26） | 工具間隔中的 KV 保留；用模擬器搜尋策略 | 對 vLLM 約 5% | 最接近；我們檢驗「既有主張是否穩健」 |
| NVIDIA Dynamo mocker | Rust 核心、block 級 LRU、重播 Mooncake 格式 | 沒讀到 | 「Rust＋重播」已有人做 |
| Kareto | 高保真模擬＋分層儲存 Pareto 搜尋＋剪枝 | mean TTFT 最大偏差 19.2% | 剪枝沒保證 |
| 〈Where Should the KV Cache Live?〉（arXiv 2609.16215） | GPU／CPU／SSD 三層、oracle prefetch | 沒有真機誤差、合成負載 | 結論「容量比策略重要」和我們部分重疊 |
| KVCache in the Wild（ATC'25）、KVSET（arXiv 2609.27746） | KV 特徵化；KVSET 用 Mattson 估工作集 | — | 只有命中率、多半只有 LRU |

來源：https://arxiv.org/abs/2405.05465 · https://arxiv.org/abs/2602.23036 · https://neurips.cc/virtual/2026/170686 · https://docs.nvidia.com/dynamo/user-guides/dynosim/mocker · https://arxiv.org/abs/2603.08739 · https://arxiv.org/abs/2609.16215 · https://www.usenix.org/conference/atc25/presentation/wang-jiahao · https://arxiv.org/abs/2609.27746

## D　創新性與投稿（對抗式評估）

- 只有「快又逐位元一致的模擬器」會被看成工程。能成立的是：**大規模掃描找到非掃描看不到的結論**（排名翻轉、命中率 ≠ TTFT、少數維度主導），或**有定理的篩選保證**。
- 審稿人最強的拒稿論點：只是重寫；對真機的驗證弱（我們的縮小版 GPU 重播 0.911–0.987、不含排隊，高壓力正是排隊主導）；大掃描沒有產出新結論；artifact 依賴無授權紀錄。
- 修正順序（成本由低到高）：和 Dynamo mocker／libCacheSim 正面比較 → 找出只有全空間才看得到的規則 → 速度–精度權衡研究 → LMCache／SGLang HiCache 全尺度含排隊驗證 → 不靠全量校準的偏差界定理。
- 投稿：SIGMETRICS 2027 冬季輪（摘要 2027-01-04、全文 01-11）；ICPE 2027 Emerging Research（2027-02-01）、Poster/Demo（02-08）；ISPASS 2027 截止日沒查到；MLSys 2027、EuroSys 2027 已過。
- 授權：CacheWise 沒有授權 → 不能進公開 artifact。

來源：https://icpe2027.spec.org/important-dates/ · https://sigmetrics.org/sigmetrics2027/pages/cfp.html · https://reviewers.acm.org/training-course/artifact-review-and-badging

## E　Rust 效能寫法

- `lto="fat"`、`codegen-units=1`、`panic="abort"`；Rust 預設不融合 FMA（RFC 3514），`target-cpu=native` 不改 IEEE 語意。
- 平行：一個行程多執行緒共用唯讀紀錄；兩顆 CPU 的機器可每顆一個行程（`numactl --cpunodebind --membind`）；最長的 job 先跑。
- 還能做：PGO（約 10%+）、buffer 重用、bitset、預先算好下次存取。
- 下次存取／Belady 的標準實作：預先算 next-access＋以它排序的 heap＋延遲刪除（libCacheSim、cachekit、gem5）。

來源：https://doc.rust-lang.org/cargo/reference/profiles.html · https://nnethercote.github.io/perf-book/build-configuration.html · https://github.com/rust-lang/rfcs/blob/master/text/3514-float-semantics.md · https://github.com/rust-lang/unsafe-code-guidelines/issues/237 · https://github.com/Kobzol/cargo-pgo · https://man7.org/linux/man-pages/man8/numactl.8.html · https://deepwiki.com/1a1a11a/libCacheSim/3.8-oracle-and-optimal-algorithms · https://oxidizelabs.github.io/cachekit/policies/roadmap/opt.html · https://gem5.googlesource.com/arm/gem5/+/m5_1.1/cpu/trace/opt_cpu.cc
