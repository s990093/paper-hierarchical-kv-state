# 14 號的速度（D3、M1、M2；只報告）

中位數（s）；只用 gpu_guard 全 CLEAN 的 run（CLAUDE.md §3）。

## d3/llama_int4（可用；20261020-200011-m10h-md-d3-llama）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | E0 | cake | 0.4093 | 10 | 10 |
| 32768 | b16 | busy | E0-DB25 | cake | 0.3190 | 10 | 8 |
| 32768 | b16 | busy | E0-DB50 | cake | 0.1943 | 10 | 5 |
| 32768 | b16 | busy | E0-DF25 | cake | 0.3646 | 10 | 9 |
| 32768 | b16 | busy | E0-DF50 | cake | 0.2352 | 10 | 6 |
| 32768 | b16 | busy | E0-DR25 | cake | 0.3186 | 10 | 8 |
| 32768 | b16 | busy | E0-DR50 | cake | 0.2344 | 10 | 5 6 |
| 32768 | b16 | busy | E1s | cake | 0.2345 | 10 | 6 |
| 32768 | b16 | busy | E1s-DB25 | cake | 0.1922 | 10 | 5 |
| 32768 | b16 | busy | E1s-DB50 | cake | 0.1122 | 10 | 3 |
| 32768 | b16 | busy | E1s-DF25 | cake | 0.1934 | 10 | 5 |
| 32768 | b16 | busy | E1s-DF50 | cake | 0.1510 | 10 | 4 |
| 32768 | b16 | busy | E1s-DR25 | cake | 0.1932 | 10 | 5 |
| 32768 | b16 | busy | E1s-DR50 | cake | 0.1159 | 10 | 3 4 |
| 32768 | b16 | idle | E0 | cake | 0.3539 | 10 | 14 15 16 |
| 32768 | b16 | idle | E0-DB25 | cake | 0.2553 | 10 | 11 12 |
| 32768 | b16 | idle | E0-DB50 | cake | 0.1840 | 10 | 8 9 |
| 32768 | b16 | idle | E0-DF25 | cake | 0.3526 | 10 | 14 15 16 |
| 32768 | b16 | idle | E0-DF50 | cake | 0.2301 | 10 | 10 11 |
| 32768 | b16 | idle | E0-DR25 | cake | 0.2786 | 10 | 12 13 14 |
| 32768 | b16 | idle | E0-DR50 | cake | 0.2066 | 10 | 10 9 |
| 32768 | b16 | idle | E1s | cake | 0.1411 | 10 | 6 7 |
| 32768 | b16 | idle | E1s-DB25 | cake | 0.0990 | 10 | 5 |
| 32768 | b16 | idle | E1s-DB50 | cake | 0.0779 | 10 | 4 |
| 32768 | b16 | idle | E1s-DF25 | cake | 0.1195 | 10 | 5 6 |
| 32768 | b16 | idle | E1s-DF50 | cake | 0.0780 | 10 | 3 4 |
| 32768 | b16 | idle | E1s-DR25 | cake | 0.0991 | 10 | 4 5 6 |
| 32768 | b16 | idle | E1s-DR50 | cake | 0.0781 | 10 | 3 4 |
| 32768 | b32 | busy | E0 | cake | 1.0653 | 10 | 23 |
| 32768 | b32 | busy | E0-DB25 | cake | 0.7919 | 10 | 18 |
| 32768 | b32 | busy | E0-DB50 | cake | 0.5025 | 10 | 12 |
| 32768 | b32 | busy | E0-DF25 | cake | 1.0572 | 10 | 23 |
| 32768 | b32 | busy | E0-DF50 | cake | 0.7935 | 10 | 18 |
| 32768 | b32 | busy | E0-DR25 | cake | 0.8514 | 10 | 19 20 |
| 32768 | b32 | busy | E0-DR50 | cake | 0.6444 | 10 | 14 15 16 |
| 32768 | b32 | busy | E1s | cake | 0.4504 | 10 | 11 |
| 32768 | b32 | busy | E1s-DB25 | cake | 0.3552 | 10 | 8 9 |
| 32768 | b32 | busy | E1s-DB50 | cake | 0.2316 | 10 | 6 |
| 32768 | b32 | busy | E1s-DF25 | cake | 0.4056 | 10 | 10 |
| 32768 | b32 | busy | E1s-DF50 | cake | 0.2721 | 10 | 7 |
| 32768 | b32 | busy | E1s-DR25 | cake | 0.3582 | 10 | 9 |
| 32768 | b32 | busy | E1s-DR50 | cake | 0.2392 | 10 | 6 7 |
| 32768 | b32 | idle | E0 | cake | 0.8254 | 10 | 31 32 |
| 32768 | b32 | idle | E0-DB25 | cake | 0.6017 | 10 | 24 25 |
| 32768 | b32 | idle | E0-DB50 | cake | 0.3771 | 10 | 16 17 |
| 32768 | b32 | idle | E0-DF25 | cake | 0.8240 | 10 | 31 32 |
| 32768 | b32 | idle | E0-DF50 | cake | 0.7906 | 10 | 30 31 |
| 32768 | b32 | idle | E0-DR25 | cake | 0.6963 | 10 | 27 28 30 |
| 32768 | b32 | idle | E0-DR50 | cake | 0.5335 | 10 | 21 22 23 24 |
| 32768 | b32 | idle | E1s | cake | 0.3773 | 10 | 16 17 |
| 32768 | b32 | idle | E1s-DB25 | cake | 0.2825 | 10 | 12 13 |
| 32768 | b32 | idle | E1s-DB50 | cake | 0.1851 | 10 | 8 9 |
| 32768 | b32 | idle | E1s-DF25 | cake | 0.3786 | 10 | 16 17 |
| 32768 | b32 | idle | E1s-DF50 | cake | 0.2516 | 10 | 11 12 |
| 32768 | b32 | idle | E1s-DR25 | cake | 0.3006 | 10 | 13 14 15 |
| 32768 | b32 | idle | E1s-DR50 | cake | 0.2257 | 10 | 10 11 9 |
| 32768 | b48 | busy | E0 | cake | 2.1442 | 10 | 39 40 |
| 32768 | b48 | busy | E0-DB25 | cake | 1.5584 | 10 | 30 31 |
| 32768 | b48 | busy | E0-DB50 | cake | 0.9526 | 10 | 21 |
| 32768 | b48 | busy | E0-DF25 | cake | 2.1432 | 10 | 39 40 |
| 32768 | b48 | busy | E0-DF50 | cake | 2.1460 | 10 | 39 40 |
| 32768 | b48 | busy | E0-DR25 | cake | 1.8715 | 10 | 34 35 36 37 38 |
| 32768 | b48 | busy | E0-DR50 | cake | 1.4717 | 10 | 29 30 31 32 |
| 32768 | b48 | busy | E1s | cake | 1.1219 | 10 | 24 |
| 32768 | b48 | busy | E1s-DB25 | cake | 0.8405 | 10 | 19 |
| 32768 | b48 | busy | E1s-DB50 | cake | 0.5405 | 10 | 13 |
| 32768 | b48 | busy | E1s-DF25 | cake | 1.1179 | 10 | 24 |
| 32768 | b48 | busy | E1s-DF50 | cake | 0.8951 | 10 | 20 |
| 32768 | b48 | busy | E1s-DR25 | cake | 0.9092 | 10 | 20 21 |
| 32768 | b48 | busy | E1s-DR50 | cake | 0.6926 | 10 | 14 15 16 17 |
| 32768 | b48 | idle | E0 | cake | 1.4111 | 10 | 47 48 |
| 32768 | b48 | idle | E0-DB25 | cake | 1.0392 | 10 | 36 |
| 32768 | b48 | idle | E0-DB50 | cake | 0.6065 | 10 | 25 |
| 32768 | b48 | idle | E0-DF25 | cake | 1.4122 | 10 | 47 48 |
| 32768 | b48 | idle | E0-DF50 | cake | 1.4130 | 10 | 47 48 |
| 32768 | b48 | idle | E0-DR25 | cake | 1.2989 | 10 | 43 44 45 46 |
| 32768 | b48 | idle | E0-DR50 | cake | 1.0966 | 10 | 37 38 39 40 41 |
| 32768 | b48 | idle | E1s | cake | 0.8559 | 10 | 32 33 |
| 32768 | b48 | idle | E1s-DB25 | cake | 0.6235 | 10 | 25 |
| 32768 | b48 | idle | E1s-DB50 | cake | 0.4015 | 10 | 17 18 |
| 32768 | b48 | idle | E1s-DF25 | cake | 0.8555 | 10 | 33 |
| 32768 | b48 | idle | E1s-DF50 | cake | 0.8580 | 10 | 31 33 |
| 32768 | b48 | idle | E1s-DR25 | cake | 0.7270 | 10 | 28 29 31 |
| 32768 | b48 | idle | E1s-DR50 | cake | 0.5583 | 10 | 22 23 24 |
| 32768 | cpu | busy | E0 | cake | 0.1209 | 10 | 3 4 |
| 32768 | cpu | busy | E0 | loadall | 0.2398 | 10 | 0 |
| 32768 | cpu | busy | E0-DB25 | cake | 0.1170 | 10 | 3 |
| 32768 | cpu | busy | E0-DB25 | loadall | 0.1791 | 10 | 0 |
| 32768 | cpu | busy | E0-DB50 | cake | 0.0760 | 10 | 2 |
| 32768 | cpu | busy | E0-DB50 | loadall | 0.1186 | 10 | 0 |
| 32768 | cpu | busy | E0-DF25 | cake | 0.1171 | 10 | 3 |
| 32768 | cpu | busy | E0-DF25 | loadall | 0.1802 | 10 | 0 |
| 32768 | cpu | busy | E0-DF50 | cake | 0.0763 | 10 | 2 |
| 32768 | cpu | busy | E0-DF50 | loadall | 0.1196 | 10 | 0 |
| 32768 | cpu | busy | E0-DR25 | cake | 0.1169 | 10 | 3 |
| 32768 | cpu | busy | E0-DR25 | loadall | 0.1791 | 10 | 0 |
| 32768 | cpu | busy | E0-DR50 | cake | 0.0761 | 10 | 2 |
| 32768 | cpu | busy | E0-DR50 | loadall | 0.1199 | 10 | 0 |
| 32768 | cpu | busy | E1s | cake | 0.0764 | 10 | 2 |
| 32768 | cpu | busy | E1s | loadall | 0.2374 | 10 | 0 |
| 32768 | cpu | busy | E1s-DB25 | cake | 0.0752 | 10 | 2 |
| 32768 | cpu | busy | E1s-DB25 | loadall | 0.1777 | 10 | 0 |
| 32768 | cpu | busy | E1s-DB50 | cake | 0.0385 | 10 | 1 |
| 32768 | cpu | busy | E1s-DB50 | loadall | 0.1184 | 10 | 0 |
| 32768 | cpu | busy | E1s-DF25 | cake | 0.0753 | 10 | 2 |
| 32768 | cpu | busy | E1s-DF25 | loadall | 0.1773 | 10 | 0 |
| 32768 | cpu | busy | E1s-DF50 | cake | 0.0378 | 10 | 1 |
| 32768 | cpu | busy | E1s-DF50 | loadall | 0.1175 | 10 | 0 |
| 32768 | cpu | busy | E1s-DR25 | cake | 0.0763 | 10 | 2 |
| 32768 | cpu | busy | E1s-DR25 | loadall | 0.1782 | 10 | 0 |
| 32768 | cpu | busy | E1s-DR50 | cake | 0.0384 | 10 | 1 |
| 32768 | cpu | busy | E1s-DR50 | loadall | 0.1176 | 10 | 0 |
| 32768 | cpu | idle | E0 | cake | 0.0841 | 10 | 4 5 |
| 32768 | cpu | idle | E0 | loadall | 0.0852 | 10 | 0 |
| 32768 | cpu | idle | E0-DB25 | cake | 0.0812 | 10 | 3 4 |
| 32768 | cpu | idle | E0-DB25 | loadall | 0.0640 | 10 | 0 |
| 32768 | cpu | idle | E0-DB50 | cake | 0.0597 | 10 | 2 3 |
| 32768 | cpu | idle | E0-DB50 | loadall | 0.0429 | 10 | 0 |
| 32768 | cpu | idle | E0-DF25 | cake | 0.0815 | 10 | 3 4 |
| 32768 | cpu | idle | E0-DF25 | loadall | 0.0640 | 10 | 0 |
| 32768 | cpu | idle | E0-DF50 | cake | 0.0600 | 10 | 2 3 |
| 32768 | cpu | idle | E0-DF50 | loadall | 0.0429 | 10 | 0 |
| 32768 | cpu | idle | E0-DR25 | cake | 0.0813 | 10 | 3 4 |
| 32768 | cpu | idle | E0-DR25 | loadall | 0.0641 | 10 | 0 |
| 32768 | cpu | idle | E0-DR50 | cake | 0.0599 | 10 | 2 3 |
| 32768 | cpu | idle | E0-DR50 | loadall | 0.0431 | 10 | 0 |
| 32768 | cpu | idle | E1s | cake | 0.0457 | 10 | 1 2 |
| 32768 | cpu | idle | E1s | loadall | 0.0324 | 10 | 0 |
| 32768 | cpu | idle | E1s-DB25 | cake | 0.0311 | 10 | 1 2 |
| 32768 | cpu | idle | E1s-DB25 | loadall | 0.0247 | 10 | 0 |
| 32768 | cpu | idle | E1s-DB50 | cake | 0.0232 | 10 | 1 |
| 32768 | cpu | idle | E1s-DB50 | loadall | 0.0170 | 10 | 0 |
| 32768 | cpu | idle | E1s-DF25 | cake | 0.0437 | 10 | 1 2 |
| 32768 | cpu | idle | E1s-DF25 | loadall | 0.0248 | 10 | 0 |
| 32768 | cpu | idle | E1s-DF50 | cake | 0.0230 | 10 | 1 |
| 32768 | cpu | idle | E1s-DF50 | loadall | 0.0170 | 10 | 0 |
| 32768 | cpu | idle | E1s-DR25 | cake | 0.0435 | 10 | 1 2 |
| 32768 | cpu | idle | E1s-DR25 | loadall | 0.0247 | 10 | 0 |
| 32768 | cpu | idle | E1s-DR50 | cake | 0.0232 | 10 | 1 |
| 32768 | cpu | idle | E1s-DR50 | loadall | 0.0171 | 10 | 0 |

## d3/llama_nfs（可用；20261020-201147-m10h-md-d3nfs-llama）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | nfs | busy | E0 | cake | 2.8324 | 10 | 49 |
| 32768 | nfs | busy | E0-DB25 | cake | 2.0773 | 10 | 37 |
| 32768 | nfs | busy | E0-DB50 | cake | 1.3221 | 10 | 25 |
| 32768 | nfs | busy | E0-DF25 | cake | 2.8322 | 10 | 49 |
| 32768 | nfs | busy | E0-DF50 | cake | 2.8324 | 10 | 49 |
| 32768 | nfs | busy | E0-DR25 | cake | 2.6438 | 10 | 44 45 46 47 |
| 32768 | nfs | busy | E0-DR50 | cake | 2.2659 | 10 | 39 40 41 42 43 |
| 32768 | nfs | idle | E0 | cake | 1.7064 | 10 | 54 55 |
| 32768 | nfs | idle | E0-DB25 | cake | 1.3221 | 10 | 41 |
| 32768 | nfs | idle | E0-DB50 | cake | 0.7558 | 10 | 28 |
| 32768 | nfs | idle | E0-DF25 | cake | 1.7061 | 10 | 54 55 |
| 32768 | nfs | idle | E0-DF50 | cake | 1.7105 | 10 | 54 55 |
| 32768 | nfs | idle | E0-DR25 | cake | 1.6998 | 10 | 51 52 53 55 |
| 32768 | nfs | idle | E0-DR50 | cake | 1.5113 | 10 | 46 47 48 50 51 52 |

## d3/qwen_int4（可用；20261020-201722-m10h-md-d3-qwen）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | E0 | cake | 0.3662 | 10 | 10 |
| 32768 | b16 | busy | E0-DB25 | cake | 0.2856 | 10 | 8 |
| 32768 | b16 | busy | E0-DB50 | cake | 0.1712 | 10 | 5 |
| 32768 | b16 | busy | E0-DF25 | cake | 0.3236 | 10 | 9 |
| 32768 | b16 | busy | E0-DF50 | cake | 0.2094 | 10 | 6 |
| 32768 | b16 | busy | E0-DR25 | cake | 0.2864 | 10 | 8 |
| 32768 | b16 | busy | E0-DR50 | cake | 0.2087 | 10 | 6 |
| 32768 | b16 | busy | E1s | cake | 0.2447 | 10 | 6 7 |
| 32768 | b16 | busy | E1s-DB25 | cake | 0.1715 | 10 | 5 |
| 32768 | b16 | busy | E1s-DB50 | cake | 0.1329 | 10 | 3 4 |
| 32768 | b16 | busy | E1s-DF25 | cake | 0.1905 | 10 | 5 6 |
| 32768 | b16 | busy | E1s-DF50 | cake | 0.1339 | 10 | 3 4 |
| 32768 | b16 | busy | E1s-DR25 | cake | 0.1712 | 10 | 5 |
| 32768 | b16 | busy | E1s-DR50 | cake | 0.1340 | 10 | 3 4 |
| 32768 | b16 | idle | E0 | cake | 0.3141 | 10 | 15 16 |
| 32768 | b16 | idle | E0-DB25 | cake | 0.2279 | 10 | 10 12 |
| 32768 | b16 | idle | E0-DB50 | cake | 0.1642 | 10 | 8 9 |
| 32768 | b16 | idle | E0-DF25 | cake | 0.3134 | 10 | 16 |
| 32768 | b16 | idle | E0-DF50 | cake | 0.2049 | 10 | 11 |
| 32768 | b16 | idle | E0-DR25 | cake | 0.2476 | 10 | 12 13 14 |
| 32768 | b16 | idle | E0-DR50 | cake | 0.1842 | 10 | 10 9 |
| 32768 | b16 | idle | E1s | cake | 0.1258 | 10 | 6 7 |
| 32768 | b16 | idle | E1s-DB25 | cake | 0.0879 | 10 | 5 |
| 32768 | b16 | idle | E1s-DB50 | cake | 0.0692 | 10 | 3 4 |
| 32768 | b16 | idle | E1s-DF25 | cake | 0.1065 | 10 | 6 |
| 32768 | b16 | idle | E1s-DF50 | cake | 0.0694 | 10 | 3 4 |
| 32768 | b16 | idle | E1s-DR25 | cake | 0.0884 | 10 | 5 6 |
| 32768 | b16 | idle | E1s-DR50 | cake | 0.0694 | 10 | 3 4 |
| 32768 | b32 | busy | E0 | cake | 0.9452 | 10 | 23 |
| 32768 | b32 | busy | E0-DB25 | cake | 0.6962 | 10 | 17 18 |
| 32768 | b32 | busy | E0-DB50 | cake | 0.4456 | 10 | 12 |
| 32768 | b32 | busy | E0-DF25 | cake | 0.9467 | 10 | 22 23 |
| 32768 | b32 | busy | E0-DF50 | cake | 0.7082 | 10 | 18 |
| 32768 | b32 | busy | E0-DR25 | cake | 0.7570 | 10 | 19 20 |
| 32768 | b32 | busy | E0-DR50 | cake | 0.5652 | 10 | 13 14 15 16 |
| 32768 | b32 | busy | E1s | cake | 0.4029 | 10 | 11 |
| 32768 | b32 | busy | E1s-DB25 | cake | 0.3210 | 10 | 9 |
| 32768 | b32 | busy | E1s-DB50 | cake | 0.2059 | 10 | 6 |
| 32768 | b32 | busy | E1s-DF25 | cake | 0.3610 | 10 | 10 |
| 32768 | b32 | busy | E1s-DF50 | cake | 0.2449 | 10 | 7 |
| 32768 | b32 | busy | E1s-DR25 | cake | 0.3194 | 10 | 9 |
| 32768 | b32 | busy | E1s-DR50 | cake | 0.2424 | 10 | 6 7 |
| 32768 | b32 | idle | E0 | cake | 0.7294 | 10 | 31 32 |
| 32768 | b32 | idle | E0-DB25 | cake | 0.5343 | 10 | 24 25 |
| 32768 | b32 | idle | E0-DB50 | cake | 0.3361 | 10 | 16 17 |
| 32768 | b32 | idle | E0-DF25 | cake | 0.7303 | 10 | 31 32 |
| 32768 | b32 | idle | E0-DF50 | cake | 0.6975 | 10 | 31 |
| 32768 | b32 | idle | E0-DR25 | cake | 0.6125 | 10 | 27 28 30 |
| 32768 | b32 | idle | E0-DR50 | cake | 0.4725 | 10 | 21 22 23 24 |
| 32768 | b32 | idle | E1s | cake | 0.3347 | 10 | 14 17 |
| 32768 | b32 | idle | E1s-DB25 | cake | 0.2468 | 10 | 13 |
| 32768 | b32 | idle | E1s-DB50 | cake | 0.1638 | 10 | 9 |
| 32768 | b32 | idle | E1s-DF25 | cake | 0.3354 | 10 | 14 16 17 |
| 32768 | b32 | idle | E1s-DF50 | cake | 0.2250 | 10 | 12 |
| 32768 | b32 | idle | E1s-DR25 | cake | 0.2683 | 10 | 13 14 15 |
| 32768 | b32 | idle | E1s-DR50 | cake | 0.1925 | 10 | 10 11 |
| 32768 | b48 | busy | E0 | cake | 1.9017 | 10 | 39 40 |
| 32768 | b48 | busy | E0-DB25 | cake | 1.3746 | 10 | 30 |
| 32768 | b48 | busy | E0-DB50 | cake | 0.8470 | 10 | 21 |
| 32768 | b48 | busy | E0-DF25 | cake | 1.9090 | 10 | 39 40 |
| 32768 | b48 | busy | E0-DF50 | cake | 1.9090 | 10 | 39 40 |
| 32768 | b48 | busy | E0-DR25 | cake | 1.6570 | 10 | 34 35 36 37 38 |
| 32768 | b48 | busy | E0-DR50 | cake | 1.3060 | 10 | 29 30 31 32 |
| 32768 | b48 | busy | E1s | cake | 0.9938 | 10 | 24 |
| 32768 | b48 | busy | E1s-DB25 | cake | 0.7489 | 10 | 18 19 |
| 32768 | b48 | busy | E1s-DB50 | cake | 0.4861 | 10 | 13 |
| 32768 | b48 | busy | E1s-DF25 | cake | 0.9962 | 10 | 24 |
| 32768 | b48 | busy | E1s-DF50 | cake | 0.7985 | 10 | 20 |
| 32768 | b48 | busy | E1s-DR25 | cake | 0.8082 | 10 | 20 21 |
| 32768 | b48 | busy | E1s-DR50 | cake | 0.6229 | 10 | 14 15 16 17 |
| 32768 | b48 | idle | E0 | cake | 1.2446 | 10 | 47 48 |
| 32768 | b48 | idle | E0-DB25 | cake | 0.9167 | 10 | 36 |
| 32768 | b48 | idle | E0-DB50 | cake | 0.5350 | 10 | 25 |
| 32768 | b48 | idle | E0-DF25 | cake | 1.2430 | 10 | 48 |
| 32768 | b48 | idle | E0-DF50 | cake | 1.2451 | 10 | 47 48 |
| 32768 | b48 | idle | E0-DR25 | cake | 1.1457 | 10 | 43 44 45 46 |
| 32768 | b48 | idle | E0-DR50 | cake | 0.9659 | 10 | 37 38 39 40 41 |
| 32768 | b48 | idle | E1s | cake | 0.7594 | 10 | 33 |
| 32768 | b48 | idle | E1s-DB25 | cake | 0.5495 | 10 | 25 |
| 32768 | b48 | idle | E1s-DB50 | cake | 0.3572 | 10 | 17 18 |
| 32768 | b48 | idle | E1s-DF25 | cake | 0.7638 | 10 | 31 32 33 |
| 32768 | b48 | idle | E1s-DF50 | cake | 0.7578 | 10 | 33 |
| 32768 | b48 | idle | E1s-DR25 | cake | 0.6433 | 10 | 28 29 31 |
| 32768 | b48 | idle | E1s-DR50 | cake | 0.4947 | 10 | 22 23 24 |
| 32768 | cpu | busy | E0 | cake | 0.0689 | 10 | 2 |
| 32768 | cpu | busy | E0 | loadall | 0.2375 | 10 | 0 |
| 32768 | cpu | busy | E0-DB25 | cake | 0.0683 | 10 | 2 |
| 32768 | cpu | busy | E0-DB25 | loadall | 0.1776 | 10 | 0 |
| 32768 | cpu | busy | E0-DB50 | cake | 0.0342 | 10 | 1 |
| 32768 | cpu | busy | E0-DB50 | loadall | 0.1171 | 10 | 0 |
| 32768 | cpu | busy | E0-DF25 | cake | 0.0681 | 10 | 2 |
| 32768 | cpu | busy | E0-DF25 | loadall | 0.1788 | 10 | 0 |
| 32768 | cpu | busy | E0-DF50 | cake | 0.0341 | 10 | 1 |
| 32768 | cpu | busy | E0-DF50 | loadall | 0.1183 | 10 | 0 |
| 32768 | cpu | busy | E0-DR25 | cake | 0.0681 | 10 | 2 |
| 32768 | cpu | busy | E0-DR25 | loadall | 0.1784 | 10 | 0 |
| 32768 | cpu | busy | E0-DR50 | cake | 0.0340 | 10 | 1 |
| 32768 | cpu | busy | E0-DR50 | loadall | 0.1171 | 10 | 0 |
| 32768 | cpu | busy | E1s | cake | 0.0671 | 10 | 1 2 |
| 32768 | cpu | busy | E1s | loadall | 0.2366 | 10 | 0 |
| 32768 | cpu | busy | E1s-DB25 | cake | 0.0336 | 10 | 1 |
| 32768 | cpu | busy | E1s-DB25 | loadall | 0.1767 | 10 | 0 |
| 32768 | cpu | busy | E1s-DB50 | cake | 0.0334 | 10 | 1 |
| 32768 | cpu | busy | E1s-DB50 | loadall | 0.1185 | 10 | 0 |
| 32768 | cpu | busy | E1s-DF25 | cake | 0.0337 | 10 | 1 |
| 32768 | cpu | busy | E1s-DF25 | loadall | 0.1772 | 10 | 0 |
| 32768 | cpu | busy | E1s-DF50 | cake | 0.0334 | 10 | 1 |
| 32768 | cpu | busy | E1s-DF50 | loadall | 0.1184 | 10 | 0 |
| 32768 | cpu | busy | E1s-DR25 | cake | 0.0337 | 10 | 1 |
| 32768 | cpu | busy | E1s-DR25 | loadall | 0.1784 | 10 | 0 |
| 32768 | cpu | busy | E1s-DR50 | cake | 0.0332 | 10 | 1 |
| 32768 | cpu | busy | E1s-DR50 | loadall | 0.1184 | 10 | 0 |
| 32768 | cpu | idle | E0 | cake | 0.0538 | 10 | 2 3 |
| 32768 | cpu | idle | E0 | loadall | 0.0391 | 10 | 0 |
| 32768 | cpu | idle | E0-DB25 | cake | 0.0359 | 10 | 1 2 |
| 32768 | cpu | idle | E0-DB25 | loadall | 0.0295 | 10 | 0 |
| 32768 | cpu | idle | E0-DB50 | cake | 0.0351 | 10 | 1 2 |
| 32768 | cpu | idle | E0-DB50 | loadall | 0.0199 | 10 | 0 |
| 32768 | cpu | idle | E0-DF25 | cake | 0.0360 | 10 | 2 |
| 32768 | cpu | idle | E0-DF25 | loadall | 0.0294 | 10 | 0 |
| 32768 | cpu | idle | E0-DF50 | cake | 0.0349 | 10 | 1 2 |
| 32768 | cpu | idle | E0-DF50 | loadall | 0.0199 | 10 | 0 |
| 32768 | cpu | idle | E0-DR25 | cake | 0.0359 | 10 | 1 2 |
| 32768 | cpu | idle | E0-DR25 | loadall | 0.0295 | 10 | 0 |
| 32768 | cpu | idle | E0-DR50 | cake | 0.0351 | 10 | 1 2 |
| 32768 | cpu | idle | E0-DR50 | loadall | 0.0199 | 10 | 0 |
| 32768 | cpu | idle | E1s | cake | 0.0271 | 10 | 1 |
| 32768 | cpu | idle | E1s | loadall | 0.0175 | 10 | 0 |
| 32768 | cpu | idle | E1s-DB25 | cake | 0.0245 | 10 | 1 |
| 32768 | cpu | idle | E1s-DB25 | loadall | 0.0133 | 10 | 0 |
| 32768 | cpu | idle | E1s-DB50 | cake | 0.0218 | 10 | 1 |
| 32768 | cpu | idle | E1s-DB50 | loadall | 0.0092 | 10 | 0 |
| 32768 | cpu | idle | E1s-DF25 | cake | 0.0245 | 10 | 1 |
| 32768 | cpu | idle | E1s-DF25 | loadall | 0.0135 | 10 | 0 |
| 32768 | cpu | idle | E1s-DF50 | cake | 0.0219 | 10 | 1 |
| 32768 | cpu | idle | E1s-DF50 | loadall | 0.0092 | 10 | 0 |
| 32768 | cpu | idle | E1s-DR25 | cake | 0.0245 | 10 | 1 |
| 32768 | cpu | idle | E1s-DR25 | loadall | 0.0133 | 10 | 0 |
| 32768 | cpu | idle | E1s-DR50 | cake | 0.0219 | 10 | 1 |
| 32768 | cpu | idle | E1s-DR50 | loadall | 0.0093 | 10 | 0 |

## d3/qwen_nfs（可用；20261020-202728-m10h-md-d3nfs-qwen）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | nfs | busy | E0 | cake | 1.9558 | 10 | 40 41 |
| 32768 | nfs | busy | E0-DB25 | cake | 1.4046 | 10 | 31 |
| 32768 | nfs | busy | E0-DB50 | cake | 0.9091 | 10 | 21 |
| 32768 | nfs | busy | E0-DF25 | cake | 1.9724 | 10 | 40 41 |
| 32768 | nfs | busy | E0-DF50 | cake | 1.9689 | 10 | 40 41 |
| 32768 | nfs | busy | E0-DR25 | cake | 1.7349 | 10 | 35 36 37 38 39 |
| 32768 | nfs | busy | E0-DR50 | cake | 1.3595 | 10 | 29 30 31 32 33 34 |
| 32768 | nfs | idle | E0 | cake | 1.3221 | 10 | 48 |
| 32768 | nfs | idle | E0-DB25 | cake | 0.9091 | 10 | 37 |
| 32768 | nfs | idle | E0-DB50 | cake | 0.5788 | 10 | 25 |
| 32768 | nfs | idle | E0-DF25 | cake | 1.3221 | 10 | 48 |
| 32768 | nfs | idle | E0-DF50 | cake | 1.3221 | 10 | 48 |
| 32768 | nfs | idle | E0-DR25 | cake | 1.1673 | 10 | 43 44 45 46 47 |
| 32768 | nfs | idle | E0-DR50 | cake | 0.9971 | 10 | 39 40 41 43 |

## m1/llama_u128（可用；20261020-220737-m10h-md-m1-llama-u128）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | E0 | cake | 0.4850 | 10 | 10 11 12 14 17 21 |
| 32768 | b16 | busy | E1s | cake | 0.4708 | 10 | 10 12 13 14 9 |
| 32768 | b16 | idle | E0 | cake | 0.4458 | 10 | 11 18 26 27 7 8 |
| 32768 | b16 | idle | E1s | cake | 0.3166 | 10 | 3 4 6 7 |
| 32768 | b32 | busy | E0 | cake | 1.4271 | 10 | 26 27 28 29 36 38 |
| 32768 | b32 | busy | E1s | cake | 0.7979 | 10 | 14 15 17 21 24 26 |
| 32768 | b32 | idle | E0 | cake | 1.3462 | 10 | 25 33 37 44 46 47 |
| 32768 | b32 | idle | E1s | cake | 0.6988 | 10 | 10 18 26 28 9 |
| 32768 | b48 | busy | E0 | cake | 3.3052 | 10 | 101 103 104 107 110 111 |
| 32768 | b48 | busy | E1s | cake | 1.6251 | 10 | 39 45 47 51 52 54 |
| 32768 | b48 | idle | E0 | cake | 3.3327 | 10 | 110 111 134 143 148 76 |
| 32768 | b48 | idle | E1s | cake | 1.5335 | 10 | 27 28 29 36 37 52 |
| 32768 | cpu | busy | E0 | cake | 0.4503 | 10 | 10 11 9 |
| 32768 | cpu | busy | E1s | cake | 0.3927 | 10 | 10 6 8 9 |
| 32768 | cpu | idle | E0 | cake | 0.1239 | 10 | 2 3 4 6 |
| 32768 | cpu | idle | E1s | cake | 0.1679 | 10 | 2 |

## m1/llama_u256（可用；20261020-220356-m10h-md-m1-llama-u256）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | E0 | cake | 0.4354 | 10 | 11 12 13 16 7 8 |
| 32768 | b16 | busy | E1s | cake | 0.3096 | 10 | 10 5 7 8 9 |
| 32768 | b16 | idle | E0 | cake | 0.4422 | 10 | 11 13 23 7 8 |
| 32768 | b16 | idle | E1s | cake | 0.2268 | 10 | 3 4 8 |
| 32768 | b32 | busy | E0 | cake | 1.2766 | 10 | 24 25 27 28 29 32 |
| 32768 | b32 | busy | E1s | cake | 0.6038 | 10 | 10 11 12 14 18 9 |
| 32768 | b32 | idle | E0 | cake | 1.3131 | 10 | 23 24 25 26 29 41 |
| 32768 | b32 | idle | E1s | cake | 0.5561 | 10 | 10 13 14 8 9 |
| 32768 | b48 | busy | E0 | cake | 2.7114 | 10 | 57 62 63 65 67 68 |
| 32768 | b48 | busy | E1s | cake | 1.4558 | 10 | 26 27 28 31 32 33 |
| 32768 | b48 | idle | E0 | cake | 2.4722 | 10 | 60 62 63 65 71 75 |
| 32768 | b48 | idle | E1s | cake | 1.1118 | 10 | 26 33 39 48 49 50 |
| 32768 | cpu | busy | E0 | cake | 0.2738 | 10 | 6 7 |
| 32768 | cpu | busy | E1s | cake | 0.2761 | 10 | 3 4 5 |
| 32768 | cpu | idle | E0 | cake | 0.1198 | 10 | 2 3 4 6 |
| 32768 | cpu | idle | E1s | cake | 0.0843 | 10 | 1 |

## m1/llama_u512（可用；20261020-220108-m10h-md-m1-llama-u512）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | E0 | cake | 0.4109 | 10 | 10 8 9 |
| 32768 | b16 | busy | E1s | cake | 0.2310 | 10 | 3 4 6 |
| 32768 | b16 | idle | E0 | cake | 0.3971 | 10 | 10 11 13 15 16 7 |
| 32768 | b16 | idle | E1s | cake | 0.1612 | 10 | 3 4 5 6 7 |
| 32768 | b32 | busy | E0 | cake | 1.0812 | 10 | 20 21 22 23 |
| 32768 | b32 | busy | E1s | cake | 0.4500 | 10 | 10 11 |
| 32768 | b32 | idle | E0 | cake | 0.9719 | 10 | 21 22 23 25 26 27 |
| 32768 | b32 | idle | E1s | cake | 0.4781 | 10 | 11 14 16 17 7 8 |
| 32768 | b48 | busy | E0 | cake | 2.1645 | 10 | 38 39 40 |
| 32768 | b48 | busy | E1s | cake | 1.1511 | 10 | 22 23 24 |
| 32768 | b48 | idle | E0 | cake | 1.6188 | 10 | 40 43 44 47 48 |
| 32768 | b48 | idle | E1s | cake | 1.0471 | 10 | 23 26 28 33 |
| 32768 | cpu | busy | E0 | cake | 0.1283 | 10 | 3 4 |
| 32768 | cpu | busy | E1s | cake | 0.1165 | 10 | 2 |
| 32768 | cpu | idle | E0 | cake | 0.0842 | 10 | 2 3 4 |
| 32768 | cpu | idle | E1s | cake | 0.0558 | 10 | 1 2 |

## m1/qwen_u128（可用；20261020-221716-m10h-md-m1-qwen-u128）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | E0 | cake | 0.4111 | 10 | 10 11 12 13 14 15 |
| 32768 | b16 | busy | E1s | cake | 0.5650 | 10 | 10 11 12 13 14 15 |
| 32768 | b16 | idle | E0 | cake | 0.3987 | 10 | 10 11 13 26 7 8 |
| 32768 | b16 | idle | E1s | cake | 0.2494 | 10 | 4 6 |
| 32768 | b32 | busy | E0 | cake | 1.1910 | 10 | 28 29 34 38 41 47 |
| 32768 | b32 | busy | E1s | cake | 0.6502 | 10 | 13 14 15 17 20 21 |
| 32768 | b32 | idle | E0 | cake | 1.1773 | 10 | 25 35 39 41 44 51 |
| 32768 | b32 | idle | E1s | cake | 0.4383 | 10 | 10 14 22 26 27 9 |
| 32768 | b48 | busy | E0 | cake | 2.9975 | 10 | 100 103 104 111 124 86 |
| 32768 | b48 | busy | E1s | cake | 1.4002 | 10 | 32 33 36 38 49 56 |
| 32768 | b48 | idle | E0 | cake | 2.8416 | 10 | 101 107 108 115 117 124 |
| 32768 | b48 | idle | E1s | cake | 1.4061 | 10 | 31 35 36 37 56 57 |
| 32768 | cpu | busy | E0 | cake | 0.1402 | 10 | 6 7 8 |
| 32768 | cpu | busy | E1s | cake | 0.1758 | 10 | 10 11 6 7 8 |
| 32768 | cpu | idle | E0 | cake | 0.0762 | 10 | 2 3 |
| 32768 | cpu | idle | E1s | cake | 0.1487 | 10 | 2 |

## m1/qwen_u256（可用；20261020-221422-m10h-md-m1-qwen-u256）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | E0 | cake | 0.3740 | 10 | 10 15 16 8 9 |
| 32768 | b16 | busy | E1s | cake | 0.3027 | 10 | 10 11 5 6 8 |
| 32768 | b16 | idle | E0 | cake | 0.3969 | 10 | 10 11 14 25 8 9 |
| 32768 | b16 | idle | E1s | cake | 0.1933 | 10 | 3 4 5 8 |
| 32768 | b32 | busy | E0 | cake | 1.0760 | 10 | 25 29 30 32 33 39 |
| 32768 | b32 | busy | E1s | cake | 0.4528 | 10 | 12 13 15 16 17 18 |
| 32768 | b32 | idle | E0 | cake | 1.0121 | 10 | 25 29 36 37 39 41 |
| 32768 | b32 | idle | E1s | cake | 0.5006 | 10 | 10 11 26 8 9 |
| 32768 | b48 | busy | E0 | cake | 2.2261 | 10 | 64 65 68 69 71 74 |
| 32768 | b48 | busy | E1s | cake | 1.1943 | 10 | 29 33 34 35 37 38 |
| 32768 | b48 | idle | E0 | cake | 1.8907 | 10 | 66 68 74 77 78 79 |
| 32768 | b48 | idle | E1s | cake | 0.9167 | 10 | 30 38 44 46 52 56 |
| 32768 | cpu | busy | E0 | cake | 0.1311 | 10 | 4 |
| 32768 | cpu | busy | E1s | cake | 0.1511 | 10 | 3 4 5 |
| 32768 | cpu | idle | E0 | cake | 0.0522 | 10 | 1 2 3 |
| 32768 | cpu | idle | E1s | cake | 0.0427 | 10 | 1 |

## m1/qwen_u512（可用；20261020-221201-m10h-md-m1-qwen-u512）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | E0 | cake | 0.3662 | 10 | 10 7 8 9 |
| 32768 | b16 | busy | E1s | cake | 0.2211 | 10 | 4 5 6 7 |
| 32768 | b16 | idle | E0 | cake | 0.3133 | 10 | 11 15 16 9 |
| 32768 | b16 | idle | E1s | cake | 0.1313 | 10 | 3 4 6 7 |
| 32768 | b32 | busy | E0 | cake | 0.9435 | 10 | 21 22 23 |
| 32768 | b32 | busy | E1s | cake | 0.4043 | 10 | 10 11 |
| 32768 | b32 | idle | E0 | cake | 0.7526 | 10 | 26 28 29 30 31 32 |
| 32768 | b32 | idle | E1s | cake | 0.3440 | 10 | 11 12 17 8 9 |
| 32768 | b48 | busy | E0 | cake | 1.8869 | 10 | 38 39 40 |
| 32768 | b48 | busy | E1s | cake | 0.9972 | 10 | 23 24 |
| 32768 | b48 | idle | E0 | cake | 1.3750 | 10 | 44 45 46 48 |
| 32768 | b48 | idle | E1s | cake | 0.7926 | 10 | 25 26 29 31 32 33 |
| 32768 | cpu | busy | E0 | cake | 0.0744 | 10 | 2 3 |
| 32768 | cpu | busy | E1s | cake | 0.0686 | 10 | 1 2 |
| 32768 | cpu | idle | E0 | cake | 0.0541 | 10 | 2 3 |
| 32768 | cpu | idle | E1s | cake | 0.0278 | 10 | 1 |

## m2/llama_fp8_token（可用；20261020-203130-m10h-md-m2-llama-fp8_token）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | - | compute_only | 4.1156 | 10 | 64 |
| 32768 | b16 | busy | E0 | cake | 0.4099 | 10 | 10 |
| 32768 | b16 | busy | E0 | load_only | 0.4530 | 10 | 64 |
| 32768 | b16 | busy | E1s | cake | 0.2366 | 10 | 6 |
| 32768 | b16 | busy | E1s | load_only | 0.3445 | 10 | 64 |
| 32768 | b16 | idle | - | compute_only | 2.1217 | 10 | 64 |
| 32768 | b16 | idle | E0 | cake | 0.3535 | 10 | 15 16 |
| 32768 | b16 | idle | E0 | load_only | 0.4530 | 10 | 64 |
| 32768 | b16 | idle | E1s | cake | 0.2076 | 10 | 10 7 9 |
| 32768 | b16 | idle | E1s | load_only | 0.2372 | 10 | 64 |
| 32768 | b32 | busy | - | compute_only | 4.1136 | 10 | 64 |
| 32768 | b32 | busy | E0 | cake | 1.0611 | 10 | 23 |
| 32768 | b32 | busy | E0 | load_only | 1.6066 | 10 | 64 |
| 32768 | b32 | busy | E1s | cake | 0.6894 | 10 | 16 |
| 32768 | b32 | busy | E1s | load_only | 0.9016 | 10 | 64 |
| 32768 | b32 | idle | - | compute_only | 2.1237 | 10 | 64 |
| 32768 | b32 | idle | E0 | cake | 0.8237 | 10 | 31 32 |
| 32768 | b32 | idle | E0 | load_only | 1.6066 | 10 | 64 |
| 32768 | b32 | idle | E1s | cake | 0.5436 | 10 | 23 |
| 32768 | b32 | idle | E1s | load_only | 0.8408 | 10 | 64 |
| 32768 | b48 | busy | - | compute_only | 4.1219 | 10 | 64 |
| 32768 | b48 | busy | E0 | cake | 2.1450 | 10 | 39 40 |
| 32768 | b48 | busy | E0 | load_only | 5.5389 | 10 | 64 |
| 32768 | b48 | busy | E1s | cake | 1.5353 | 10 | 31 |
| 32768 | b48 | busy | E1s | load_only | 2.9824 | 10 | 64 |
| 32768 | b48 | idle | - | compute_only | 2.1392 | 10 | 64 |
| 32768 | b48 | idle | E0 | cake | 1.4111 | 10 | 47 48 |
| 32768 | b48 | idle | E0 | load_only | 5.5389 | 10 | 64 |
| 32768 | b48 | idle | E1s | cake | 1.1020 | 10 | 39 40 |
| 32768 | b48 | idle | E1s | load_only | 2.8987 | 10 | 64 |
| 32768 | nfs | busy | - | compute_only | 4.1179 | 10 | 64 |
| 32768 | nfs | busy | E0 | cake | 2.8324 | 10 | 49 |
| 32768 | nfs | busy | E0 | load_only | 12.0818 | 10 | 64 |
| 32768 | nfs | busy | E1s | cake | 2.2566 | 10 | 41 |
| 32768 | nfs | busy | E1s | load_only | 6.4168 | 10 | 64 |
| 32768 | nfs | idle | - | compute_only | 2.1215 | 10 | 64 |
| 32768 | nfs | idle | E0 | cake | 1.7069 | 10 | 54 55 |
| 32768 | nfs | idle | E0 | load_only | 12.0818 | 10 | 64 |
| 32768 | nfs | idle | E1s | cake | 1.4692 | 10 | 49 |
| 32768 | nfs | idle | E1s | load_only | 6.3250 | 10 | 64 |

## m2/llama_int2_kivi（可用；20261020-210619-m10h-md-m2-llama-int2_kivi）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | - | compute_only | 4.1010 | 10 | 64 |
| 32768 | b16 | busy | E1s | cake | 0.1505 | 10 | 3 4 |
| 32768 | b16 | busy | E1s | load_only | 0.2394 | 10 | 64 |
| 32768 | b16 | idle | - | compute_only | 2.1224 | 10 | 64 |
| 32768 | b16 | idle | E1s | cake | 0.0810 | 10 | 3 4 5 |
| 32768 | b16 | idle | E1s | load_only | 0.0910 | 10 | 64 |
| 32768 | b32 | busy | - | compute_only | 4.0991 | 10 | 64 |
| 32768 | b32 | busy | E1s | cake | 0.3137 | 10 | 8 |
| 32768 | b32 | busy | E1s | load_only | 0.3712 | 10 | 64 |
| 32768 | b32 | idle | - | compute_only | 2.1217 | 10 | 64 |
| 32768 | b32 | idle | E1s | cake | 0.2529 | 10 | 11 12 |
| 32768 | b32 | idle | E1s | load_only | 0.3219 | 10 | 64 |
| 32768 | b48 | busy | - | compute_only | 4.0965 | 10 | 64 |
| 32768 | b48 | busy | E1s | cake | 0.7874 | 10 | 18 |
| 32768 | b48 | busy | E1s | load_only | 1.1696 | 10 | 64 |
| 32768 | b48 | idle | - | compute_only | 2.1227 | 10 | 64 |
| 32768 | b48 | idle | E1s | cake | 0.6317 | 10 | 25 26 |
| 32768 | b48 | idle | E1s | load_only | 1.1091 | 10 | 64 |
| 32768 | nfs | busy | - | compute_only | 4.0926 | 10 | 64 |
| 32768 | nfs | busy | E1s | cake | 1.3426 | 10 | 28 |
| 32768 | nfs | busy | E1s | load_only | 2.4989 | 10 | 64 |
| 32768 | nfs | idle | - | compute_only | 2.1228 | 10 | 64 |
| 32768 | nfs | idle | E1s | cake | 0.9926 | 10 | 36 37 |
| 32768 | nfs | idle | E1s | load_only | 2.4197 | 10 | 64 |

## m2/llama_int4_kivi（可用；20261020-205004-m10h-md-m2-llama-int4_kivi）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | - | compute_only | 4.1088 | 10 | 64 |
| 32768 | b16 | busy | E1s | cake | 0.2330 | 10 | 6 |
| 32768 | b16 | busy | E1s | load_only | 0.2404 | 10 | 64 |
| 32768 | b16 | idle | - | compute_only | 2.1250 | 10 | 64 |
| 32768 | b16 | idle | E1s | cake | 0.1415 | 10 | 6 7 |
| 32768 | b16 | idle | E1s | load_only | 0.1467 | 10 | 64 |
| 32768 | b32 | busy | - | compute_only | 4.1051 | 10 | 64 |
| 32768 | b32 | busy | E1s | cake | 0.4474 | 10 | 11 |
| 32768 | b32 | busy | E1s | load_only | 0.5829 | 10 | 64 |
| 32768 | b32 | idle | - | compute_only | 2.1365 | 10 | 64 |
| 32768 | b32 | idle | E1s | cake | 0.3789 | 10 | 16 17 |
| 32768 | b32 | idle | E1s | load_only | 0.5196 | 10 | 64 |
| 32768 | b48 | busy | - | compute_only | 4.1056 | 10 | 64 |
| 32768 | b48 | busy | E1s | cake | 1.1116 | 10 | 24 |
| 32768 | b48 | busy | E1s | load_only | 1.8661 | 10 | 64 |
| 32768 | b48 | idle | - | compute_only | 2.1244 | 10 | 64 |
| 32768 | b48 | idle | E1s | cake | 0.8558 | 10 | 32 33 |
| 32768 | b48 | idle | E1s | load_only | 1.7907 | 10 | 64 |
| 32768 | nfs | busy | - | compute_only | 4.1092 | 10 | 64 |
| 32768 | nfs | busy | E1s | cake | 1.7904 | 10 | 35 |
| 32768 | nfs | busy | E1s | load_only | 4.0000 | 10 | 64 |
| 32768 | nfs | idle | - | compute_only | 2.1223 | 10 | 64 |
| 32768 | nfs | idle | E1s | cake | 1.2417 | 10 | 43 |
| 32768 | nfs | idle | E1s | load_only | 3.9079 | 10 | 64 |

## m2/llama_kv-int4_kivi-int2_kivi（可用；20261020-205827-m10h-md-m2-llama-kv-int4_kivi-int2_kivi）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | - | compute_only | 4.0986 | 10 | 64 |
| 32768 | b16 | busy | E1s | cake | 0.2319 | 10 | 5 6 |
| 32768 | b16 | busy | E1s | load_only | 0.2397 | 10 | 64 |
| 32768 | b16 | idle | - | compute_only | 2.1224 | 10 | 64 |
| 32768 | b16 | idle | E1s | cake | 0.1201 | 10 | 5 6 |
| 32768 | b16 | idle | E1s | load_only | 0.1189 | 10 | 64 |
| 32768 | b32 | busy | - | compute_only | 4.0959 | 10 | 64 |
| 32768 | b32 | busy | E1s | cake | 0.4053 | 10 | 10 |
| 32768 | b32 | busy | E1s | load_only | 0.4770 | 10 | 64 |
| 32768 | b32 | idle | - | compute_only | 2.1231 | 10 | 64 |
| 32768 | b32 | idle | E1s | cake | 0.3261 | 10 | 14 15 |
| 32768 | b32 | idle | E1s | load_only | 0.4207 | 10 | 64 |
| 32768 | b48 | busy | - | compute_only | 4.0976 | 10 | 64 |
| 32768 | b48 | busy | E1s | cake | 0.9775 | 10 | 21 22 |
| 32768 | b48 | busy | E1s | load_only | 1.5314 | 10 | 64 |
| 32768 | b48 | idle | - | compute_only | 2.1199 | 10 | 64 |
| 32768 | b48 | idle | E1s | cake | 0.7573 | 10 | 29 30 |
| 32768 | b48 | idle | E1s | load_only | 1.4499 | 10 | 64 |
| 32768 | nfs | busy | - | compute_only | 4.0969 | 10 | 64 |
| 32768 | nfs | busy | E1s | cake | 1.5932 | 10 | 32 |
| 32768 | nfs | busy | E1s | load_only | 3.2560 | 10 | 64 |
| 32768 | nfs | idle | - | compute_only | 2.1210 | 10 | 64 |
| 32768 | nfs | idle | E1s | cake | 1.1357 | 10 | 40 |
| 32768 | nfs | idle | E1s | load_only | 3.1632 | 10 | 64 |

## m2/llama_mxfp4（可用；20261020-211336-m10h-md-m2-llama-mxfp4）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | - | compute_only | 4.0980 | 10 | 64 |
| 32768 | b16 | busy | E1s | cake | 0.2322 | 10 | 6 |
| 32768 | b16 | busy | E1s | load_only | 0.2410 | 10 | 64 |
| 32768 | b16 | idle | - | compute_only | 2.1207 | 10 | 64 |
| 32768 | b16 | idle | E1s | cake | 0.1197 | 10 | 5 6 |
| 32768 | b16 | idle | E1s | load_only | 0.1258 | 10 | 64 |
| 32768 | b32 | busy | - | compute_only | 4.0930 | 10 | 64 |
| 32768 | b32 | busy | E1s | cake | 0.4043 | 10 | 10 |
| 32768 | b32 | busy | E1s | load_only | 0.4819 | 10 | 64 |
| 32768 | b32 | idle | - | compute_only | 2.1215 | 10 | 64 |
| 32768 | b32 | idle | E1s | cake | 0.3275 | 10 | 15 |
| 32768 | b32 | idle | E1s | load_only | 0.4454 | 10 | 64 |
| 32768 | b48 | busy | - | compute_only | 4.0958 | 10 | 64 |
| 32768 | b48 | busy | E1s | cake | 1.0024 | 10 | 22 |
| 32768 | b48 | busy | E1s | load_only | 1.6053 | 10 | 64 |
| 32768 | b48 | idle | - | compute_only | 2.1213 | 10 | 64 |
| 32768 | b48 | idle | E1s | cake | 0.7884 | 10 | 30 31 |
| 32768 | b48 | idle | E1s | load_only | 1.5351 | 10 | 64 |
| 32768 | nfs | busy | - | compute_only | 4.0980 | 10 | 64 |
| 32768 | nfs | busy | E1s | cake | 1.6333 | 10 | 32 33 |
| 32768 | nfs | busy | E1s | load_only | 3.4274 | 10 | 64 |
| 32768 | nfs | idle | - | compute_only | 2.1242 | 10 | 64 |
| 32768 | nfs | idle | E1s | cake | 1.1569 | 10 | 41 |
| 32768 | nfs | idle | E1s | load_only | 3.3490 | 10 | 64 |

## m2/qwen_fp8_head（可用；20261020-212139-m10h-md-m2-qwen-fp8_head）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | - | compute_only | 3.5986 | 10 | 64 |
| 32768 | b16 | busy | E0 | cake | 0.3620 | 10 | 10 |
| 32768 | b16 | busy | E0 | load_only | 0.4043 | 10 | 64 |
| 32768 | b16 | busy | E1s | cake | 0.2454 | 10 | 7 |
| 32768 | b16 | busy | E1s | load_only | 0.2399 | 10 | 64 |
| 32768 | b16 | idle | - | compute_only | 1.8650 | 10 | 64 |
| 32768 | b16 | idle | E0 | cake | 0.3122 | 10 | 15 16 17 |
| 32768 | b16 | idle | E0 | load_only | 0.4043 | 10 | 64 |
| 32768 | b16 | idle | E1s | cake | 0.1832 | 10 | 10 9 |
| 32768 | b16 | idle | E1s | load_only | 0.2055 | 10 | 64 |
| 32768 | b32 | busy | - | compute_only | 3.5976 | 10 | 64 |
| 32768 | b32 | busy | E0 | cake | 0.9393 | 10 | 23 |
| 32768 | b32 | busy | E0 | load_only | 1.4233 | 10 | 64 |
| 32768 | b32 | busy | E1s | cake | 0.5667 | 10 | 15 |
| 32768 | b32 | busy | E1s | load_only | 0.8021 | 10 | 64 |
| 32768 | b32 | idle | - | compute_only | 1.8666 | 10 | 64 |
| 32768 | b32 | idle | E0 | cake | 0.7295 | 10 | 31 32 |
| 32768 | b32 | idle | E0 | load_only | 1.4233 | 10 | 64 |
| 32768 | b32 | idle | E1s | cake | 0.4791 | 10 | 22 23 |
| 32768 | b32 | idle | E1s | load_only | 0.7230 | 10 | 64 |
| 32768 | b48 | busy | - | compute_only | 3.6009 | 10 | 64 |
| 32768 | b48 | busy | E0 | cake | 1.8824 | 10 | 39 40 |
| 32768 | b48 | busy | E0 | load_only | 4.8858 | 10 | 64 |
| 32768 | b48 | busy | E1s | cake | 1.3160 | 10 | 30 |
| 32768 | b48 | busy | E1s | load_only | 2.5545 | 10 | 64 |
| 32768 | b48 | idle | - | compute_only | 1.8634 | 10 | 64 |
| 32768 | b48 | idle | E0 | cake | 1.2434 | 10 | 47 48 |
| 32768 | b48 | idle | E0 | load_only | 4.8858 | 10 | 64 |
| 32768 | b48 | idle | E1s | cake | 0.9556 | 10 | 39 |
| 32768 | b48 | idle | E1s | load_only | 2.4815 | 10 | 64 |
| 32768 | nfs | busy | - | compute_only | 3.6024 | 10 | 64 |
| 32768 | nfs | busy | E0 | cake | 1.9489 | 10 | 40 41 |
| 32768 | nfs | busy | E0 | load_only | 5.2859 | 10 | 64 |
| 32768 | nfs | busy | E1s | cake | 1.3750 | 10 | 31 |
| 32768 | nfs | busy | E1s | load_only | 2.7602 | 10 | 64 |
| 32768 | nfs | idle | - | compute_only | 1.8632 | 10 | 64 |
| 32768 | nfs | idle | E0 | cake | 1.3221 | 10 | 48 49 |
| 32768 | nfs | idle | E0 | load_only | 5.2859 | 10 | 64 |
| 32768 | nfs | idle | E1s | cake | 0.9925 | 10 | 40 |
| 32768 | nfs | idle | E1s | load_only | 2.6847 | 10 | 64 |

## m2/qwen_int2_kivi（可用；20261020-214805-m10h-md-m2-qwen-int2_kivi）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | - | compute_only | 3.6358 | 10 | 64 |
| 32768 | b16 | busy | E1s | cake | 0.1368 | 10 | 4 |
| 32768 | b16 | busy | E1s | load_only | 0.2375 | 10 | 64 |
| 32768 | b16 | idle | - | compute_only | 2.1864 | 10 | 64 |
| 32768 | b16 | idle | E1s | cake | 0.0883 | 10 | 2 3 5 |
| 32768 | b16 | idle | E1s | load_only | 0.0813 | 10 | 64 |
| 32768 | b32 | busy | - | compute_only | 3.6118 | 10 | 64 |
| 32768 | b32 | busy | E1s | cake | 0.2795 | 10 | 8 |
| 32768 | b32 | busy | E1s | load_only | 0.3656 | 10 | 64 |
| 32768 | b32 | idle | - | compute_only | 2.0546 | 10 | 64 |
| 32768 | b32 | idle | E1s | cake | 0.2441 | 10 | 11 12 6 7 9 |
| 32768 | b32 | idle | E1s | load_only | 0.2852 | 10 | 64 |
| 32768 | b48 | busy | - | compute_only | 3.6249 | 10 | 64 |
| 32768 | b48 | busy | E1s | cake | 0.7068 | 10 | 18 |
| 32768 | b48 | busy | E1s | load_only | 1.0549 | 10 | 64 |
| 32768 | b48 | idle | - | compute_only | 2.0154 | 10 | 64 |
| 32768 | b48 | idle | E1s | cake | 0.5952 | 10 | 17 19 21 22 25 26 |
| 32768 | b48 | idle | E1s | load_only | 0.9948 | 10 | 64 |
| 32768 | nfs | busy | - | compute_only | 3.6392 | 10 | 64 |
| 32768 | nfs | busy | E1s | cake | 0.7595 | 10 | 17 18 19 |
| 32768 | nfs | busy | E1s | load_only | 1.1513 | 10 | 64 |
| 32768 | nfs | idle | - | compute_only | 1.9264 | 10 | 64 |
| 32768 | nfs | idle | E1s | cake | 0.5945 | 10 | 17 18 25 26 27 |
| 32768 | nfs | idle | E1s | load_only | 1.0629 | 10 | 64 |

## m2/qwen_int4_kivi（可用；20261020-213454-m10h-md-m2-qwen-int4_kivi）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | - | compute_only | 3.6274 | 10 | 64 |
| 32768 | b16 | busy | E1s | cake | 0.2417 | 10 | 3 4 6 7 |
| 32768 | b16 | busy | E1s | load_only | 0.2366 | 10 | 64 |
| 32768 | b16 | idle | - | compute_only | 1.8735 | 10 | 64 |
| 32768 | b16 | idle | E1s | cake | 0.1255 | 10 | 3 4 5 6 7 |
| 32768 | b16 | idle | E1s | load_only | 0.1310 | 10 | 64 |
| 32768 | b32 | busy | - | compute_only | 3.6283 | 10 | 64 |
| 32768 | b32 | busy | E1s | cake | 0.4015 | 10 | 10 11 |
| 32768 | b32 | busy | E1s | load_only | 0.5659 | 10 | 64 |
| 32768 | b32 | idle | - | compute_only | 1.8809 | 10 | 64 |
| 32768 | b32 | idle | E1s | cake | 0.3347 | 10 | 16 17 |
| 32768 | b32 | idle | E1s | load_only | 0.4603 | 10 | 64 |
| 32768 | b48 | busy | - | compute_only | 3.6314 | 10 | 64 |
| 32768 | b48 | busy | E1s | cake | 1.0022 | 10 | 23 24 |
| 32768 | b48 | busy | E1s | load_only | 1.6732 | 10 | 64 |
| 32768 | b48 | idle | - | compute_only | 1.8668 | 10 | 64 |
| 32768 | b48 | idle | E1s | cake | 0.7624 | 10 | 26 29 32 33 |
| 32768 | b48 | idle | E1s | load_only | 1.5796 | 10 | 64 |
| 32768 | nfs | busy | - | compute_only | 3.6604 | 10 | 64 |
| 32768 | nfs | busy | E1s | cake | 1.0425 | 10 | 24 25 |
| 32768 | nfs | busy | E1s | load_only | 1.7992 | 10 | 64 |
| 32768 | nfs | idle | - | compute_only | 1.8676 | 10 | 64 |
| 32768 | nfs | idle | E1s | cake | 0.8015 | 10 | 30 33 34 |
| 32768 | nfs | idle | E1s | load_only | 1.7089 | 10 | 64 |

## m2/qwen_kv-int4_kivi-int2_kivi（可用；20261020-214144-m10h-md-m2-qwen-kv-int4_kivi-int2_kivi）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | - | compute_only | 3.6122 | 10 | 64 |
| 32768 | b16 | busy | E1s | cake | 0.1329 | 10 | 4 5 |
| 32768 | b16 | busy | E1s | load_only | 0.2374 | 10 | 64 |
| 32768 | b16 | idle | - | compute_only | 1.8652 | 10 | 64 |
| 32768 | b16 | idle | E1s | cake | 0.1066 | 10 | 4 5 6 |
| 32768 | b16 | idle | E1s | load_only | 0.1061 | 10 | 64 |
| 32768 | b32 | busy | - | compute_only | 3.5993 | 10 | 64 |
| 32768 | b32 | busy | E1s | cake | 0.3562 | 10 | 10 |
| 32768 | b32 | busy | E1s | load_only | 0.4673 | 10 | 64 |
| 32768 | b32 | idle | - | compute_only | 1.8675 | 10 | 64 |
| 32768 | b32 | idle | E1s | cake | 0.2900 | 10 | 10 14 15 |
| 32768 | b32 | idle | E1s | load_only | 0.3727 | 10 | 64 |
| 32768 | b48 | busy | - | compute_only | 3.6034 | 10 | 64 |
| 32768 | b48 | busy | E1s | cake | 0.8470 | 10 | 21 |
| 32768 | b48 | busy | E1s | load_only | 1.3645 | 10 | 64 |
| 32768 | b48 | idle | - | compute_only | 1.8660 | 10 | 64 |
| 32768 | b48 | idle | E1s | cake | 0.6698 | 10 | 21 29 30 |
| 32768 | b48 | idle | E1s | load_only | 1.2790 | 10 | 64 |
| 32768 | nfs | busy | - | compute_only | 3.6066 | 10 | 64 |
| 32768 | nfs | busy | E1s | cake | 0.8981 | 10 | 21 22 |
| 32768 | nfs | busy | E1s | load_only | 1.4717 | 10 | 64 |
| 32768 | nfs | idle | - | compute_only | 1.8634 | 10 | 64 |
| 32768 | nfs | idle | E1s | cake | 0.6988 | 10 | 31 |
| 32768 | nfs | idle | E1s | load_only | 1.3837 | 10 | 64 |

## m2/qwen_mxfp4（可用；20261020-215417-m10h-md-m2-qwen-mxfp4）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | - | compute_only | 3.6408 | 10 | 64 |
| 32768 | b16 | busy | E1s | cake | 0.1970 | 10 | 3 4 5 6 |
| 32768 | b16 | busy | E1s | load_only | 0.2375 | 10 | 64 |
| 32768 | b16 | idle | - | compute_only | 2.2526 | 10 | 64 |
| 32768 | b16 | idle | E1s | cake | 0.1373 | 10 | 3 4 5 6 |
| 32768 | b16 | idle | E1s | load_only | 0.1154 | 10 | 64 |
| 32768 | b32 | busy | - | compute_only | 3.7198 | 10 | 64 |
| 32768 | b32 | busy | E1s | cake | 0.4043 | 10 | 10 11 8 9 |
| 32768 | b32 | busy | E1s | load_only | 0.4732 | 10 | 64 |
| 32768 | b32 | idle | - | compute_only | 2.0962 | 10 | 64 |
| 32768 | b32 | idle | E1s | cake | 0.3651 | 10 | 10 12 15 7 8 9 |
| 32768 | b32 | idle | E1s | load_only | 0.4146 | 10 | 64 |
| 32768 | b48 | busy | - | compute_only | 3.6511 | 10 | 64 |
| 32768 | b48 | busy | E1s | cake | 0.8970 | 10 | 21 22 |
| 32768 | b48 | busy | E1s | load_only | 1.4424 | 10 | 64 |
| 32768 | b48 | idle | - | compute_only | 2.0153 | 10 | 64 |
| 32768 | b48 | idle | E1s | cake | 0.7241 | 10 | 24 26 27 29 30 31 |
| 32768 | b48 | idle | E1s | load_only | 1.3770 | 10 | 64 |
| 32768 | nfs | busy | - | compute_only | 3.6500 | 10 | 64 |
| 32768 | nfs | busy | E1s | cake | 0.9625 | 10 | 21 22 23 |
| 32768 | nfs | busy | E1s | load_only | 1.5512 | 10 | 64 |
| 32768 | nfs | idle | - | compute_only | 2.0839 | 10 | 64 |
| 32768 | nfs | idle | E1s | cake | 0.7648 | 10 | 23 24 25 26 28 31 |
| 32768 | nfs | idle | E1s | load_only | 1.4828 | 10 | 64 |

