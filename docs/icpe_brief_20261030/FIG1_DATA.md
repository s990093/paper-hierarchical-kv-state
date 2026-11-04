# 圖 2 的資料表（預測上限；圖是 figs/brief_d_prediction_ceiling.png，只畫 SSD 級那一列；圖 1 是寫入時套用的對照）

**每個格子**：完美預知（oracle）比「同一個動作下最好的不預測做法」讓回來請求的 TTFT 縮短的 %，Cake 還原，模擬；每格取 GPU 閒／忙的較大者（和論文 Fig. 9 同一規則）。
**S**＝這格寫入飽和（慢層寫不及），論文主結果不計入。**—**＝沒跑（Mooncake 只跑 CPU 128–1,024 GiB；FreeInference 16–64 GiB 來自 T5、128–1,024 來自 T1）。
Mooncake 的對話是用兩種規則重建的，這裡取較小的（`rule=min`，同論文）。
來源：Mooncake `results/icpe_h200/t1/p01/sim_V0/x1/x1_headroom_components.csv`；FreeInference `docs/icpe_paper_20261030/figs/fig9_fi_capacity.sources.json`。

## 預取（回來前先搬好）（O-PF）

| 紀錄 | 慢層 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| FreeInference | b32 | -9.00 | 4.89 | 0.47 | 0.05 | 0.04 | 0.01 | 0.00 |
| FreeInference | SSD 級 | -4.38 | 2.65 | 0.26 | 0.02 | 0.01 | 0.00 | 0.00 |
| FreeInference | NFS 級 | -19.2 | -24.7 | -4.13 | 0.22 | 0.19 | 0.04 | 0.00 |
| Mooncake conversation | b32 | — | — | — | 49.1 | 44.6 | 28.6 | 13.1 |
| Mooncake conversation | SSD 級 | — | — | — | 50.5 | 46.5 | 30.3 | 14.6 |
| Mooncake conversation | NFS 級 | — | — | — | 26.8S | 30.1S | 45.2S | 48.9S |
| Mooncake toolagent | b32 | — | — | — | 28.9 | 23.1 | 9.91 | 2.10 |
| Mooncake toolagent | SSD 級 | — | — | — | 30.0 | 24.1 | 10.9 | 2.54 |
| Mooncake toolagent | NFS 級 | — | — | — | 26.6S | 29.8S | 41.6S | 23.2S |

## 淘汰順序（CPU 滿了先擠掉誰）（O-ORD）

| 紀錄 | 慢層 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| FreeInference | b32 | 2.31 | 3.60 | 0.58 | 0.06 | 0.05 | 0.02 | 0.01 |
| FreeInference | SSD 級 | 1.20 | 1.66 | 0.29 | 0.02 | 0.02 | 0.01 | 0.00 |
| FreeInference | NFS 級 | 5.92 | 11.8 | 3.64 | 0.30 | 0.25 | 0.08 | 0.03 |
| Mooncake conversation | b32 | — | — | — | 26.9 | 29.9 | 22.6 | 8.53 |
| Mooncake conversation | SSD 級 | — | — | — | 27.6 | 31.0 | 24.2 | 10.2 |
| Mooncake conversation | NFS 級 | — | — | — | 50.2S | 63.4S | 65.5S | 46.6S |
| Mooncake toolagent | b32 | — | — | — | 15.6 | 13.9 | 6.67 | 0.78 |
| Mooncake toolagent | SSD 級 | — | — | — | 17.0 | 15.9 | 7.87 | 1.59 |
| Mooncake toolagent | NFS 級 | — | — | — | 47.9S | 55.5S | 47.1S | 21.9S |

（欄＝CPU 層容量 GiB）

## GPU 閒與忙（Mooncake、SSD 級；圖 2 取兩者較大的）

| 紀錄 | 上限 | GPU | 128 | 256 | 512 | 1024 |
|:--|:--|:--|--:|--:|--:|--:|
| Mooncake conversation | 預取 | 閒 | 50.5 | 46.5 | 30.3 | 14.6 |
| Mooncake conversation | 預取 | 忙 | 38.4 | 34.0 | 20.7 | 8.6 |
| Mooncake conversation | 淘汰順序 | 閒 | 27.6 | 31.0 | 24.2 | 9.9 |
| Mooncake conversation | 淘汰順序 | 忙 | 25.3 | 28.8 | 23.0 | 10.2 |
| Mooncake toolagent | 預取 | 閒 | 30.0 | 24.1 | 10.9 | 2.5 |
| Mooncake toolagent | 預取 | 忙 | 20.8 | 15.5 | 5.3 | 0.1 |
| Mooncake toolagent | 淘汰順序 | 閒 | 16.5 | 14.7 | 6.2 | 0.4 |
| Mooncake toolagent | 淘汰順序 | 忙 | 17.0 | 15.9 | 7.9 | 1.6 |

## 拿掉 Cake 會怎樣（Mooncake、SSD 級、取 GPU 閒／忙較大者）

**Cake**＝邊重算前段、邊載入後段；**整段載入**＝沒有 Cake，整段從慢層讀回。其他條件相同（`restore=cake` 對 `restore=load`），模擬。
整段載入的上限比 Cake 高 1.9–13.8 個百分點，排序與「隨 CPU 容量下降」的趨勢不變。NFS 級的 Mooncake 格全是寫入飽和，不列。

| 紀錄 | 上限 | 還原 | 128 | 256 | 512 | 1024 |
|:--|:--|:--|--:|--:|--:|--:|
| Mooncake conversation | 預取 | Cake | 50.5 | 46.5 | 30.3 | 14.6 |
| Mooncake conversation | 預取 | 整段載入 | 59.2 | 56.3 | 39.2 | 20.9 |
| Mooncake conversation | 淘汰順序 | Cake | 27.6 | 31.0 | 24.2 | 10.2 |
| Mooncake conversation | 淘汰順序 | 整段載入 | 37.5 | 43.5 | 37.9 | 19.0 |
| Mooncake toolagent | 預取 | Cake | 30.0 | 24.1 | 10.9 | 2.5 |
| Mooncake toolagent | 預取 | 整段載入 | 36.8 | 29.7 | 14.7 | 4.5 |
| Mooncake toolagent | 淘汰順序 | Cake | 17.0 | 15.9 | 7.9 | 1.6 |
| Mooncake toolagent | 淘汰順序 | 整段載入 | 23.6 | 23.2 | 14.1 | 4.3 |
