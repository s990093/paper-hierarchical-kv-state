# 決策地圖（T1：CPU 層＝H200 的常數；〔模擬〕；只報告）

run `20261028-170127-icpe-decision-map`；來源：`results/icpe_h200/t1/p01/sim_V0/x1/x1{,h3600}_headroom_components.csv`、`results/icpe_h200/t1/p12/x1/x1fi_headroom_components.csv`（FreeInference 前 6 週，不截尾）、`results/icpe_h200/t1/p02/fullsim_hint.csv`

格內：oracle 最多贏最好的對稱對手的比例（最大的那個成分）；conversation 另附 HINT-PF 拿到 O-PF 改善的比例（>1＝O-PF 騰空間照 LRU，不是上界）。≥15% 有看頭、5–15% 灰區、<5% 沒有空間。

## 不截尾

| trace | 慢層 | GPU | CPU 128 GiB | CPU 256 GiB | CPU 512 GiB | CPU 1024 GiB |
|:--|:--|:--|--:|--:|--:|--:|
| conversation | b32 | busy | 35.3%（O-PF）；提示 0.96 | 30.9%（O-PF）；提示 0.97 | 20.7%（O-ORD）；提示 0.99 | 10.1%（O-DROP）；提示 0.99 |
| conversation | b32 | idle | 49.1%（O-PF）；提示 0.91 | 44.6%（O-PF）；提示 0.91 | 28.6%（O-PF）；提示 0.92 | 13.1%（O-PF）；提示 0.93 |
| conversation | nfs | busy | 50.2%（O-ORD）；提示 1.45 | 63.4%（O-ORD）；提示 1.30 | 65.5%（O-ORD）；提示 0.98 | 46.4%（O-ORD）；提示 0.83 |
| conversation | nfs | idle | 48.9%（O-ORD）；提示 1.32 | 62.3%（O-ORD）；提示 1.25 | 65.5%（O-ORD）；提示 0.96 | 48.9%（O-PF）；提示 0.81 |
| conversation | ssd | busy | 38.4%（O-PF）；提示 0.94 | 34.0%（O-PF）；提示 0.96 | 23.0%（O-ORD）；提示 0.98 | 11.8%（O-DROP）；提示 0.97 |
| conversation | ssd | idle | 50.5%（O-PF）；提示 0.90 | 46.5%（O-PF）；提示 0.91 | 30.3%（O-PF）；提示 0.92 | 14.6%（O-PF）；提示 0.92 |
| freeinference | b32 | busy | 0.0%（O-ORD） | 0.0%（O-ORD） | 0.0%（O-ORD） | 0.0%（O-DROP） |
| freeinference | b32 | idle | 0.1%（O-ORD） | 0.1%（O-ORD） | 0.0%（O-ORD） | 0.0%（O-DROP） |
| freeinference | nfs | busy | 0.2%（O-ORD） | 0.2%（O-ORD） | 0.1%（O-ORD） | 0.0%（O-DROP） |
| freeinference | nfs | idle | 0.3%（O-ORD） | 0.3%（O-ORD） | 0.1%（O-ORD） | 0.0%（O-DROP） |
| freeinference | ssd | busy | 0.0%（O-ORD） | 0.0%（O-ORD） | 0.0%（O-ORD） | 0.0%（O-DROP） |
| freeinference | ssd | idle | 0.0%（O-ORD） | 0.0%（O-ORD） | 0.0%（O-ORD） | 0.0%（O-DROP） |
| toolagent | b32 | busy | 17.7%（O-PF） | 16.1%（O-DROP） | 7.8%（O-DROP） | 1.4%（O-DROP） |
| toolagent | b32 | idle | 28.9%（O-PF） | 23.1%（O-PF） | 9.9%（O-PF） | 2.1%（O-PF） |
| toolagent | nfs | busy | 47.9%（O-ORD） | 55.5%（O-ORD） | 47.1%（O-ORD） | 24.1%（O-DROP） |
| toolagent | nfs | idle | 45.6%（O-ORD） | 53.1%（O-ORD） | 45.7%（O-ORD） | 23.2%（O-PF） |
| toolagent | ssd | busy | 20.8%（O-PF） | 20.8%（O-DROP） | 9.7%（O-DROP） | 2.1%（O-DROP） |
| toolagent | ssd | idle | 30.0%（O-PF） | 24.1%（O-PF） | 10.9%（O-PF） | 2.5%（O-NOW） |

格數：conversation 有看頭 20、沒有空間 0、灰區 4；freeinference 有看頭 0、沒有空間 24、灰區 0；toolagent 有看頭 16、沒有空間 4、灰區 4

## 截尾 H＝3,600 s（B2：O-DROP 大半是 trace 太短）

| trace | 慢層 | GPU | CPU 128 GiB | CPU 256 GiB | CPU 512 GiB | CPU 1024 GiB |
|:--|:--|:--|--:|--:|--:|--:|
| conversation | b32 | busy | 35.3%（O-PF）；提示 0.96 | 30.9%（O-PF）；提示 0.97 | 20.7%（O-ORD）；提示 0.99 | 8.5%（O-ORD）；提示 0.99 |
| conversation | b32 | idle | 49.1%（O-PF）；提示 0.91 | 44.6%（O-PF）；提示 0.91 | 28.6%（O-PF）；提示 0.92 | 13.1%（O-PF）；提示 0.93 |
| conversation | nfs | busy | 50.2%（O-ORD）；提示 1.45 | 63.4%（O-ORD）；提示 1.30 | 65.5%（O-ORD）；提示 0.98 | 46.4%（O-ORD）；提示 0.83 |
| conversation | nfs | idle | 48.9%（O-ORD）；提示 1.32 | 62.3%（O-ORD）；提示 1.25 | 65.5%（O-ORD）；提示 0.96 | 48.9%（O-PF）；提示 0.81 |
| conversation | ssd | busy | 38.4%（O-PF）；提示 0.94 | 34.0%（O-PF）；提示 0.96 | 23.0%（O-ORD）；提示 0.98 | 10.2%（O-ORD）；提示 0.97 |
| conversation | ssd | idle | 50.5%（O-PF）；提示 0.90 | 46.5%（O-PF）；提示 0.91 | 30.3%（O-PF）；提示 0.92 | 14.6%（O-PF）；提示 0.92 |
| freeinference | b32 | busy | 0.0%（O-ORD） | 0.0%（O-ORD） | 0.0%（O-ORD） | 0.0%（O-DROP） |
| freeinference | b32 | idle | 0.1%（O-ORD） | 0.1%（O-ORD） | 0.0%（O-ORD） | 0.0%（O-DROP） |
| freeinference | nfs | busy | 0.2%（O-ORD） | 0.2%（O-ORD） | 0.1%（O-ORD） | 0.0%（O-DROP） |
| freeinference | nfs | idle | 0.3%（O-ORD） | 0.3%（O-ORD） | 0.1%（O-ORD） | 0.0%（O-DROP） |
| freeinference | ssd | busy | 0.0%（O-ORD） | 0.0%（O-ORD） | 0.0%（O-ORD） | 0.0%（O-DROP） |
| freeinference | ssd | idle | 0.0%（O-ORD） | 0.0%（O-ORD） | 0.0%（O-ORD） | 0.0%（O-DROP） |
| toolagent | b32 | busy | 17.7%（O-PF） | 13.9%（O-ORD） | 6.7%（O-ORD） | 0.8%（O-ORD） |
| toolagent | b32 | idle | 28.9%（O-PF） | 23.1%（O-PF） | 9.9%（O-PF） | 2.1%（O-PF） |
| toolagent | nfs | busy | 47.9%（O-ORD） | 55.5%（O-ORD） | 47.1%（O-ORD） | 22.4%（O-PF） |
| toolagent | nfs | idle | 45.6%（O-ORD） | 53.1%（O-ORD） | 45.7%（O-ORD） | 23.2%（O-PF） |
| toolagent | ssd | busy | 20.8%（O-PF） | 15.9%（O-ORD） | 7.9%（O-ORD） | 1.6%（O-ORD） |
| toolagent | ssd | idle | 30.0%（O-PF） | 24.1%（O-PF） | 10.9%（O-PF） | 2.5%（O-PF） |

格數：conversation 有看頭 20、沒有空間 0、灰區 4；freeinference 有看頭 0、沒有空間 24、灰區 0；toolagent 有看頭 15、沒有空間 4、灰區 5

