# NFS 等待拆解（run 20261023-012638-m16-nfslookup-report）

判定：**NOT_EXPLAINED_BY_LOOKUP_LATENCY**；**INTRINSIC_NFS_LATENCY**（NFS 的 WR／Q 平均延遲比 1.06）

| 目錄 | 階段 | 背景寫 MB/s | 背景讀 MB/s | 次數 | 平均（ms） | P50 | P95 | P99 | 命中平均 | 未命中平均 | ρ（推回 64） |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| local（/var/tmp） | Q | 0 | 0 | 5000 | 0.006 | 0.005 | 0.008 | 0.010 | 0.005 | 0.007 | 0.00 |
| local（/var/tmp） | W | 100 | 0 | 5000 | 0.006 | 0.005 | 0.009 | 0.015 | 0.005 | 0.008 | 0.00 |
| local（/var/tmp） | WR | 16 | 180 | 5000 | 0.007 | 0.006 | 0.012 | 0.017 | 0.005 | 0.009 | 0.00 |
| nfs（/mlsteam/data/tiara/stage3） | Q | 0 | 0 | 5000 | 0.100 | 0.013 | 0.207 | 0.398 | 0.017 | 0.205 | 0.03 |
| nfs（/mlsteam/data/tiara/stage3） | W | 100 | 0 | 5000 | 0.109 | 0.014 | 0.231 | 0.429 | 0.015 | 0.231 | 0.03 |
| nfs（/mlsteam/data/tiara/stage3） | WR | 12 | 176 | 5000 | 0.107 | 0.008 | 0.222 | 1.085 | 0.008 | 0.227 | 0.03 |

64 的對照：d chunk_queries 301413、makespan 1005 s、每請求查找等待 41.25 s；c 264398、838 s、0.15 s。
ρ＝查找執行緒的忙碌度（一條執行緒依序查）；ρ 接近或超過 1 時排隊會爆。量測和 64 不同時間，NFS 是共用的，延遲會隨別人的負載變。

規則：ρ_d（WR 階段 NFS 的平均延遲 × d 的 chunk_queries ÷ d 的 makespan）≥ 0.8 → LOOKUP_THREAD_SATURATED；< 0.3 → NOT_EXPLAINED_BY_LOOKUP_LATENCY；其他 PARTIAL。NFS 的 WR／Q 平均延遲比 ≥ 2 → IO_INFLATES_LOOKUP，否則 INTRINSIC_NFS_LATENCY。
