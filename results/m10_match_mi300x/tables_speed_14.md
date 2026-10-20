# 14 號的速度（D3、M1、M2；只報告）

中位數（s）；只用 gpu_guard 全 CLEAN 的 run（CLAUDE.md §3）。

## d3/llama_int4（可用；20261020-053032-m10-md-d3-llama）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | E0 | cake | 0.6653 | 10 | 8 9 |
| 32768 | b16 | busy | E0-DB25 | cake | 0.4988 | 10 | 6 7 |
| 32768 | b16 | busy | E0-DB50 | cake | 0.3994 | 10 | 4 5 |
| 32768 | b16 | busy | E0-DF25 | cake | 0.5712 | 10 | 7 |
| 32768 | b16 | busy | E0-DF50 | cake | 0.4006 | 10 | 5 |
| 32768 | b16 | busy | E0-DR25 | cake | 0.5707 | 10 | 7 |
| 32768 | b16 | busy | E0-DR50 | cake | 0.4003 | 10 | 5 |
| 32768 | b16 | busy | E1s | cake | 0.3214 | 10 | 4 |
| 32768 | b16 | busy | E1s-DB25 | cake | 0.2392 | 10 | 3 |
| 32768 | b16 | busy | E1s-DB50 | cake | 0.1622 | 10 | 2 |
| 32768 | b16 | busy | E1s-DF25 | cake | 0.2349 | 10 | 3 |
| 32768 | b16 | busy | E1s-DF50 | cake | 0.1628 | 10 | 2 |
| 32768 | b16 | busy | E1s-DR25 | cake | 0.2429 | 10 | 3 |
| 32768 | b16 | busy | E1s-DR50 | cake | 0.1631 | 10 | 2 |
| 32768 | b16 | idle | E0 | cake | 0.5760 | 10 | 16 |
| 32768 | b16 | idle | E0-DB25 | cake | 0.4494 | 10 | 13 |
| 32768 | b16 | idle | E0-DB50 | cake | 0.2946 | 10 | 9 |
| 32768 | b16 | idle | E0-DF25 | cake | 0.5750 | 10 | 16 |
| 32768 | b16 | idle | E0-DF50 | cake | 0.3706 | 10 | 11 |
| 32768 | b16 | idle | E0-DR25 | cake | 0.4553 | 10 | 13 14 |
| 32768 | b16 | idle | E0-DR50 | cake | 0.3325 | 10 | 10 11 |
| 32768 | b16 | idle | E1s | cake | 0.2254 | 10 | 7 |
| 32768 | b16 | idle | E1s-DB25 | cake | 0.1901 | 10 | 5 6 |
| 32768 | b16 | idle | E1s-DB50 | cake | 0.1233 | 10 | 4 |
| 32768 | b16 | idle | E1s-DF25 | cake | 0.1901 | 10 | 6 7 |
| 32768 | b16 | idle | E1s-DF50 | cake | 0.1241 | 10 | 4 5 |
| 32768 | b16 | idle | E1s-DR25 | cake | 0.1901 | 10 | 6 |
| 32768 | b16 | idle | E1s-DR50 | cake | 0.1238 | 10 | 4 5 |
| 32768 | b32 | busy | E0 | cake | 1.9354 | 10 | 20 21 |
| 32768 | b32 | busy | E0-DB25 | cake | 1.4018 | 10 | 16 |
| 32768 | b32 | busy | E0-DB50 | cake | 0.9243 | 10 | 11 |
| 32768 | b32 | busy | E0-DF25 | cake | 1.9358 | 10 | 21 |
| 32768 | b32 | busy | E0-DF50 | cake | 1.4000 | 10 | 16 |
| 32768 | b32 | busy | E0-DR25 | cake | 1.5059 | 10 | 17 18 |
| 32768 | b32 | busy | E0-DR50 | cake | 1.1098 | 10 | 12 13 14 |
| 32768 | b32 | busy | E1s | cake | 0.8319 | 10 | 10 |
| 32768 | b32 | busy | E1s-DB25 | cake | 0.6471 | 10 | 7 8 |
| 32768 | b32 | busy | E1s-DB50 | cake | 0.3985 | 10 | 5 |
| 32768 | b32 | busy | E1s-DF25 | cake | 0.7412 | 10 | 9 |
| 32768 | b32 | busy | E1s-DF50 | cake | 0.4815 | 10 | 6 |
| 32768 | b32 | busy | E1s-DR25 | cake | 0.6575 | 10 | 8 |
| 32768 | b32 | busy | E1s-DR50 | cake | 0.4799 | 10 | 5 6 |
| 32768 | b32 | idle | E0 | cake | 1.3734 | 10 | 32 |
| 32768 | b32 | idle | E0-DB25 | cake | 0.9941 | 10 | 25 |
| 32768 | b32 | idle | E0-DB50 | cake | 0.6402 | 10 | 17 |
| 32768 | b32 | idle | E0-DF25 | cake | 1.3748 | 10 | 32 |
| 32768 | b32 | idle | E0-DF50 | cake | 1.3720 | 10 | 31 32 |
| 32768 | b32 | idle | E0-DR25 | cake | 1.1516 | 10 | 28 30 |
| 32768 | b32 | idle | E0-DR50 | cake | 0.8946 | 10 | 21 22 23 24 |
| 32768 | b32 | idle | E1s | cake | 0.6626 | 10 | 18 |
| 32768 | b32 | idle | E1s-DB25 | cake | 0.4898 | 10 | 14 |
| 32768 | b32 | idle | E1s-DB50 | cake | 0.3316 | 10 | 10 |
| 32768 | b32 | idle | E1s-DF25 | cake | 0.6630 | 10 | 18 |
| 32768 | b32 | idle | E1s-DF50 | cake | 0.4488 | 10 | 13 |
| 32768 | b32 | idle | E1s-DR25 | cake | 0.5323 | 10 | 15 16 |
| 32768 | b32 | idle | E1s-DR50 | cake | 0.3787 | 10 | 11 12 13 |
| 32768 | b48 | busy | E0 | cake | 3.9468 | 10 | 38 |
| 32768 | b48 | busy | E0-DB25 | cake | 2.8613 | 10 | 29 |
| 32768 | b48 | busy | E0-DB50 | cake | 1.8203 | 10 | 20 |
| 32768 | b48 | busy | E0-DF25 | cake | 3.9551 | 10 | 38 |
| 32768 | b48 | busy | E0-DF50 | cake | 3.9509 | 10 | 38 |
| 32768 | b48 | busy | E0-DR25 | cake | 3.4288 | 10 | 33 34 35 36 |
| 32768 | b48 | busy | E0-DR50 | cake | 2.7475 | 10 | 26 27 28 29 |
| 32768 | b48 | busy | E1s | cake | 2.0488 | 10 | 22 |
| 32768 | b48 | busy | E1s-DB25 | cake | 1.4934 | 10 | 17 |
| 32768 | b48 | busy | E1s-DB50 | cake | 1.0150 | 10 | 12 |
| 32768 | b48 | busy | E1s-DF25 | cake | 2.0503 | 10 | 22 |
| 32768 | b48 | busy | E1s-DF50 | cake | 1.4940 | 10 | 17 18 |
| 32768 | b48 | busy | E1s-DR25 | cake | 1.6584 | 10 | 18 19 20 |
| 32768 | b48 | busy | E1s-DR50 | cake | 1.2004 | 10 | 13 14 15 |
| 32768 | b48 | idle | E0 | cake | 2.4204 | 10 | 48 |
| 32768 | b48 | idle | E0-DB25 | cake | 1.6669 | 10 | 37 |
| 32768 | b48 | idle | E0-DB50 | cake | 1.0438 | 10 | 25 |
| 32768 | b48 | idle | E0-DF25 | cake | 2.3957 | 10 | 48 |
| 32768 | b48 | idle | E0-DF50 | cake | 2.4109 | 10 | 48 |
| 32768 | b48 | idle | E0-DR25 | cake | 2.2355 | 10 | 43 44 45 46 |
| 32768 | b48 | idle | E0-DR50 | cake | 1.8945 | 10 | 37 38 39 40 42 |
| 32768 | b48 | idle | E1s | cake | 1.4726 | 10 | 33 |
| 32768 | b48 | idle | E1s-DB25 | cake | 1.0470 | 10 | 26 |
| 32768 | b48 | idle | E1s-DB50 | cake | 0.6655 | 10 | 18 |
| 32768 | b48 | idle | E1s-DF25 | cake | 1.4734 | 10 | 33 34 |
| 32768 | b48 | idle | E1s-DF50 | cake | 1.4733 | 10 | 33 |
| 32768 | b48 | idle | E1s-DR25 | cake | 1.2366 | 10 | 29 30 32 |
| 32768 | b48 | idle | E1s-DR50 | cake | 0.9466 | 10 | 23 24 25 26 |
| 32768 | cpu | busy | E0 | cake | 0.1643 | 10 | 2 |
| 32768 | cpu | busy | E0 | loadall | 0.1244 | 10 | 0 |
| 32768 | cpu | busy | E0-DB25 | cake | 0.1634 | 10 | 2 |
| 32768 | cpu | busy | E0-DB25 | loadall | 0.0928 | 10 | 0 |
| 32768 | cpu | busy | E0-DB50 | cake | 0.0792 | 10 | 1 |
| 32768 | cpu | busy | E0-DB50 | loadall | 0.0620 | 10 | 0 |
| 32768 | cpu | busy | E0-DF25 | cake | 0.1607 | 10 | 2 |
| 32768 | cpu | busy | E0-DF25 | loadall | 0.0933 | 10 | 0 |
| 32768 | cpu | busy | E0-DF50 | cake | 0.0822 | 10 | 1 |
| 32768 | cpu | busy | E0-DF50 | loadall | 0.0633 | 10 | 0 |
| 32768 | cpu | busy | E0-DR25 | cake | 0.1648 | 10 | 2 |
| 32768 | cpu | busy | E0-DR25 | loadall | 0.0929 | 10 | 0 |
| 32768 | cpu | busy | E0-DR50 | cake | 0.0824 | 10 | 1 |
| 32768 | cpu | busy | E0-DR50 | loadall | 0.0616 | 10 | 0 |
| 32768 | cpu | busy | E1s | cake | 0.0835 | 10 | 1 |
| 32768 | cpu | busy | E1s | loadall | 0.0570 | 10 | 0 |
| 32768 | cpu | busy | E1s-DB25 | cake | 0.0810 | 10 | 1 |
| 32768 | cpu | busy | E1s-DB25 | loadall | 0.0420 | 10 | 0 |
| 32768 | cpu | busy | E1s-DB50 | cake | 0.0778 | 10 | 1 |
| 32768 | cpu | busy | E1s-DB50 | loadall | 0.0306 | 10 | 0 |
| 32768 | cpu | busy | E1s-DF25 | cake | 0.0836 | 10 | 1 |
| 32768 | cpu | busy | E1s-DF25 | loadall | 0.0421 | 10 | 0 |
| 32768 | cpu | busy | E1s-DF50 | cake | 0.0813 | 10 | 1 |
| 32768 | cpu | busy | E1s-DF50 | loadall | 0.0294 | 10 | 0 |
| 32768 | cpu | busy | E1s-DR25 | cake | 0.0824 | 10 | 1 |
| 32768 | cpu | busy | E1s-DR25 | loadall | 0.0421 | 10 | 0 |
| 32768 | cpu | busy | E1s-DR50 | cake | 0.0817 | 10 | 1 |
| 32768 | cpu | busy | E1s-DR50 | loadall | 0.0301 | 10 | 0 |
| 32768 | cpu | idle | E0 | cake | 0.0954 | 10 | 3 |
| 32768 | cpu | idle | E0 | loadall | 0.0836 | 10 | 0 |
| 32768 | cpu | idle | E0-DB25 | cake | 0.0638 | 10 | 2 |
| 32768 | cpu | idle | E0-DB25 | loadall | 0.0630 | 10 | 0 |
| 32768 | cpu | idle | E0-DB50 | cake | 0.0620 | 10 | 1 2 |
| 32768 | cpu | idle | E0-DB50 | loadall | 0.0424 | 10 | 0 |
| 32768 | cpu | idle | E0-DF25 | cake | 0.0941 | 10 | 3 |
| 32768 | cpu | idle | E0-DF25 | loadall | 0.0629 | 10 | 0 |
| 32768 | cpu | idle | E0-DF50 | cake | 0.0623 | 10 | 2 |
| 32768 | cpu | idle | E0-DF50 | loadall | 0.0424 | 10 | 0 |
| 32768 | cpu | idle | E0-DR25 | cake | 0.0795 | 10 | 2 3 |
| 32768 | cpu | idle | E0-DR25 | loadall | 0.0631 | 10 | 0 |
| 32768 | cpu | idle | E0-DR50 | cake | 0.0620 | 10 | 2 |
| 32768 | cpu | idle | E0-DR50 | loadall | 0.0424 | 10 | 0 |
| 32768 | cpu | idle | E1s | cake | 0.0640 | 10 | 2 |
| 32768 | cpu | idle | E1s | loadall | 0.0365 | 10 | 0 |
| 32768 | cpu | idle | E1s-DB25 | cake | 0.0334 | 10 | 1 |
| 32768 | cpu | idle | E1s-DB25 | loadall | 0.0279 | 10 | 0 |
| 32768 | cpu | idle | E1s-DB50 | cake | 0.0321 | 10 | 1 |
| 32768 | cpu | idle | E1s-DB50 | loadall | 0.0192 | 10 | 0 |
| 32768 | cpu | idle | E1s-DF25 | cake | 0.0332 | 10 | 1 |
| 32768 | cpu | idle | E1s-DF25 | loadall | 0.0278 | 10 | 0 |
| 32768 | cpu | idle | E1s-DF50 | cake | 0.0322 | 10 | 1 |
| 32768 | cpu | idle | E1s-DF50 | loadall | 0.0193 | 10 | 0 |
| 32768 | cpu | idle | E1s-DR25 | cake | 0.0333 | 10 | 1 |
| 32768 | cpu | idle | E1s-DR25 | loadall | 0.0279 | 10 | 0 |
| 32768 | cpu | idle | E1s-DR50 | cake | 0.0323 | 10 | 1 |
| 32768 | cpu | idle | E1s-DR50 | loadall | 0.0192 | 10 | 0 |

## d3/llama_nfs（可用；20261020-055200-m10-md-d3nfs-llama）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | nfs | busy | E0 | cake | 4.3514 | 10 | 41 |
| 32768 | nfs | busy | E0-DB25 | cake | 3.2108 | 10 | 31 |
| 32768 | nfs | busy | E0-DB50 | cake | 2.0783 | 10 | 21 |
| 32768 | nfs | busy | E0-DF25 | cake | 4.3517 | 10 | 41 |
| 32768 | nfs | busy | E0-DF50 | cake | 4.3446 | 10 | 41 |
| 32768 | nfs | busy | E0-DR25 | cake | 3.7957 | 10 | 36 37 38 39 |
| 32768 | nfs | busy | E0-DR50 | cake | 3.0814 | 10 | 29 30 31 33 34 |
| 32768 | nfs | idle | E0 | cake | 2.6439 | 10 | 50 |
| 32768 | nfs | idle | E0-DB25 | cake | 1.8889 | 10 | 38 |
| 32768 | nfs | idle | E0-DB50 | cake | 1.1338 | 10 | 26 |
| 32768 | nfs | idle | E0-DF25 | cake | 2.6440 | 10 | 50 |
| 32768 | nfs | idle | E0-DF50 | cake | 2.6439 | 10 | 50 |
| 32768 | nfs | idle | E0-DR25 | cake | 2.4551 | 10 | 45 46 47 48 49 |
| 32768 | nfs | idle | E0-DR50 | cake | 2.0777 | 10 | 40 41 42 44 |

## d3/qwen_int4（可用；20261020-060107-m10-md-d3-qwen）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | E0 | cake | 0.5550 | 10 | 8 9 |
| 32768 | b16 | busy | E0-DB25 | cake | 0.4698 | 10 | 7 |
| 32768 | b16 | busy | E0-DB50 | cake | 0.3314 | 10 | 5 |
| 32768 | b16 | busy | E0-DF25 | cake | 0.4719 | 10 | 7 8 |
| 32768 | b16 | busy | E0-DF50 | cake | 0.3312 | 10 | 5 |
| 32768 | b16 | busy | E0-DR25 | cake | 0.4672 | 10 | 7 |
| 32768 | b16 | busy | E0-DR50 | cake | 0.3308 | 10 | 5 |
| 32768 | b16 | busy | E1s | cake | 0.2613 | 10 | 4 |
| 32768 | b16 | busy | E1s-DB25 | cake | 0.1932 | 10 | 3 |
| 32768 | b16 | busy | E1s-DB50 | cake | 0.1279 | 10 | 2 |
| 32768 | b16 | busy | E1s-DF25 | cake | 0.1929 | 10 | 3 |
| 32768 | b16 | busy | E1s-DF50 | cake | 0.1282 | 10 | 2 |
| 32768 | b16 | busy | E1s-DR25 | cake | 0.1927 | 10 | 3 |
| 32768 | b16 | busy | E1s-DR50 | cake | 0.1281 | 10 | 2 |
| 32768 | b16 | idle | E0 | cake | 0.4820 | 10 | 16 |
| 32768 | b16 | idle | E0-DB25 | cake | 0.3751 | 10 | 12 13 |
| 32768 | b16 | idle | E0-DB50 | cake | 0.2455 | 10 | 9 |
| 32768 | b16 | idle | E0-DF25 | cake | 0.4818 | 10 | 16 |
| 32768 | b16 | idle | E0-DF50 | cake | 0.3088 | 10 | 11 |
| 32768 | b16 | idle | E0-DR25 | cake | 0.3759 | 10 | 13 14 |
| 32768 | b16 | idle | E0-DR50 | cake | 0.2767 | 10 | 10 11 |
| 32768 | b16 | idle | E1s | cake | 0.1866 | 10 | 7 |
| 32768 | b16 | idle | E1s-DB25 | cake | 0.1584 | 10 | 5 6 |
| 32768 | b16 | idle | E1s-DB50 | cake | 0.1026 | 10 | 4 |
| 32768 | b16 | idle | E1s-DF25 | cake | 0.1579 | 10 | 6 |
| 32768 | b16 | idle | E1s-DF50 | cake | 0.1027 | 10 | 4 5 |
| 32768 | b16 | idle | E1s-DR25 | cake | 0.1579 | 10 | 6 |
| 32768 | b16 | idle | E1s-DR50 | cake | 0.1024 | 10 | 4 |
| 32768 | b32 | busy | E0 | cake | 1.6210 | 10 | 21 |
| 32768 | b32 | busy | E0-DB25 | cake | 1.1689 | 10 | 16 |
| 32768 | b32 | busy | E0-DB50 | cake | 0.7621 | 10 | 11 |
| 32768 | b32 | busy | E0-DF25 | cake | 1.6221 | 10 | 21 |
| 32768 | b32 | busy | E0-DF50 | cake | 1.1655 | 10 | 16 |
| 32768 | b32 | busy | E0-DR25 | cake | 1.2643 | 10 | 17 18 |
| 32768 | b32 | busy | E0-DR50 | cake | 0.9228 | 10 | 12 13 14 15 |
| 32768 | b32 | busy | E1s | cake | 0.6885 | 10 | 10 |
| 32768 | b32 | busy | E1s-DB25 | cake | 0.5399 | 10 | 8 |
| 32768 | b32 | busy | E1s-DB50 | cake | 0.3326 | 10 | 5 |
| 32768 | b32 | busy | E1s-DF25 | cake | 0.6154 | 10 | 9 |
| 32768 | b32 | busy | E1s-DF50 | cake | 0.3963 | 10 | 6 |
| 32768 | b32 | busy | E1s-DR25 | cake | 0.5387 | 10 | 8 |
| 32768 | b32 | busy | E1s-DR50 | cake | 0.3996 | 10 | 5 6 |
| 32768 | b32 | idle | E0 | cake | 1.1594 | 10 | 32 |
| 32768 | b32 | idle | E0-DB25 | cake | 0.8387 | 10 | 25 |
| 32768 | b32 | idle | E0-DB50 | cake | 0.5383 | 10 | 17 |
| 32768 | b32 | idle | E0-DF25 | cake | 1.1593 | 10 | 32 |
| 32768 | b32 | idle | E0-DF50 | cake | 1.1239 | 10 | 31 32 |
| 32768 | b32 | idle | E0-DR25 | cake | 0.9735 | 10 | 28 30 |
| 32768 | b32 | idle | E0-DR50 | cake | 0.7533 | 10 | 21 22 23 24 |
| 32768 | b32 | idle | E1s | cake | 0.5576 | 10 | 18 |
| 32768 | b32 | idle | E1s-DB25 | cake | 0.4099 | 10 | 14 |
| 32768 | b32 | idle | E1s-DB50 | cake | 0.2763 | 10 | 10 |
| 32768 | b32 | idle | E1s-DF25 | cake | 0.5553 | 10 | 18 |
| 32768 | b32 | idle | E1s-DF50 | cake | 0.3750 | 10 | 13 |
| 32768 | b32 | idle | E1s-DR25 | cake | 0.4491 | 10 | 15 16 |
| 32768 | b32 | idle | E1s-DR50 | cake | 0.3256 | 10 | 11 12 13 |
| 32768 | b48 | busy | E0 | cake | 3.3403 | 10 | 38 |
| 32768 | b48 | busy | E0-DB25 | cake | 2.4210 | 10 | 29 |
| 32768 | b48 | busy | E0-DB50 | cake | 1.5317 | 10 | 20 |
| 32768 | b48 | busy | E0-DF25 | cake | 3.3485 | 10 | 38 |
| 32768 | b48 | busy | E0-DF50 | cake | 3.3464 | 10 | 38 |
| 32768 | b48 | busy | E0-DR25 | cake | 2.8976 | 10 | 33 34 36 |
| 32768 | b48 | busy | E0-DR50 | cake | 2.3293 | 10 | 26 27 28 29 |
| 32768 | b48 | busy | E1s | cake | 1.7214 | 10 | 22 |
| 32768 | b48 | busy | E1s-DB25 | cake | 1.2549 | 10 | 17 |
| 32768 | b48 | busy | E1s-DB50 | cake | 0.8396 | 10 | 12 |
| 32768 | b48 | busy | E1s-DF25 | cake | 1.7204 | 10 | 22 |
| 32768 | b48 | busy | E1s-DF50 | cake | 1.3024 | 10 | 17 18 |
| 32768 | b48 | busy | E1s-DR25 | cake | 1.4006 | 10 | 18 19 20 |
| 32768 | b48 | busy | E1s-DR50 | cake | 1.0776 | 10 | 13 14 15 16 |
| 32768 | b48 | idle | E0 | cake | 2.0332 | 10 | 47 48 |
| 32768 | b48 | idle | E0-DB25 | cake | 1.4098 | 10 | 37 |
| 32768 | b48 | idle | E0-DB50 | cake | 0.8812 | 10 | 25 |
| 32768 | b48 | idle | E0-DF25 | cake | 2.0401 | 10 | 48 |
| 32768 | b48 | idle | E0-DF50 | cake | 2.0311 | 10 | 48 |
| 32768 | b48 | idle | E0-DR25 | cake | 1.8869 | 10 | 43 44 45 46 |
| 32768 | b48 | idle | E0-DR50 | cake | 1.5865 | 10 | 37 38 39 40 42 |
| 32768 | b48 | idle | E1s | cake | 1.2472 | 10 | 33 34 |
| 32768 | b48 | idle | E1s-DB25 | cake | 0.8893 | 10 | 26 |
| 32768 | b48 | idle | E1s-DB50 | cake | 0.5632 | 10 | 18 |
| 32768 | b48 | idle | E1s-DF25 | cake | 1.2468 | 10 | 33 34 |
| 32768 | b48 | idle | E1s-DF50 | cake | 1.2474 | 10 | 33 34 |
| 32768 | b48 | idle | E1s-DR25 | cake | 1.0459 | 10 | 29 30 32 |
| 32768 | b48 | idle | E1s-DR50 | cake | 0.8021 | 10 | 23 24 25 26 |
| 32768 | cpu | busy | E0 | cake | 0.0651 | 10 | 1 |
| 32768 | cpu | busy | E0 | loadall | 0.0561 | 10 | 0 |
| 32768 | cpu | busy | E0-DB25 | cake | 0.0654 | 10 | 1 |
| 32768 | cpu | busy | E0-DB25 | loadall | 0.0420 | 10 | 0 |
| 32768 | cpu | busy | E0-DB50 | cake | 0.0642 | 10 | 1 |
| 32768 | cpu | busy | E0-DB50 | loadall | 0.0285 | 10 | 0 |
| 32768 | cpu | busy | E0-DF25 | cake | 0.0651 | 10 | 1 |
| 32768 | cpu | busy | E0-DF25 | loadall | 0.0420 | 10 | 0 |
| 32768 | cpu | busy | E0-DF50 | cake | 0.0646 | 10 | 1 |
| 32768 | cpu | busy | E0-DF50 | loadall | 0.0285 | 10 | 0 |
| 32768 | cpu | busy | E0-DR25 | cake | 0.0645 | 10 | 1 |
| 32768 | cpu | busy | E0-DR25 | loadall | 0.0421 | 10 | 0 |
| 32768 | cpu | busy | E0-DR50 | cake | 0.0644 | 10 | 1 |
| 32768 | cpu | busy | E0-DR50 | loadall | 0.0286 | 10 | 0 |
| 32768 | cpu | busy | E1s | cake | 0.0656 | 10 | 1 |
| 32768 | cpu | busy | E1s | loadall | 0.0431 | 10 | 0 |
| 32768 | cpu | busy | E1s-DB25 | cake | 0.0647 | 10 | 1 |
| 32768 | cpu | busy | E1s-DB25 | loadall | 0.0348 | 10 | 0 |
| 32768 | cpu | busy | E1s-DB50 | cake | 0.0642 | 10 | 1 |
| 32768 | cpu | busy | E1s-DB50 | loadall | 0.0225 | 10 | 0 |
| 32768 | cpu | busy | E1s-DF25 | cake | 0.0650 | 10 | 1 |
| 32768 | cpu | busy | E1s-DF25 | loadall | 0.0356 | 10 | 0 |
| 32768 | cpu | busy | E1s-DF50 | cake | 0.0642 | 10 | 1 |
| 32768 | cpu | busy | E1s-DF50 | loadall | 0.0225 | 10 | 0 |
| 32768 | cpu | busy | E1s-DR25 | cake | 0.0646 | 10 | 1 |
| 32768 | cpu | busy | E1s-DR25 | loadall | 0.0346 | 10 | 0 |
| 32768 | cpu | busy | E1s-DR50 | cake | 0.0658 | 10 | 1 |
| 32768 | cpu | busy | E1s-DR50 | loadall | 0.0226 | 10 | 0 |
| 32768 | cpu | idle | E0 | cake | 0.0521 | 10 | 2 |
| 32768 | cpu | idle | E0 | loadall | 0.0388 | 10 | 0 |
| 32768 | cpu | idle | E0-DB25 | cake | 0.0512 | 10 | 2 |
| 32768 | cpu | idle | E0-DB25 | loadall | 0.0295 | 10 | 0 |
| 32768 | cpu | idle | E0-DB50 | cake | 0.0261 | 10 | 1 |
| 32768 | cpu | idle | E0-DB50 | loadall | 0.0201 | 10 | 0 |
| 32768 | cpu | idle | E0-DF25 | cake | 0.0515 | 10 | 2 |
| 32768 | cpu | idle | E0-DF25 | loadall | 0.0295 | 10 | 0 |
| 32768 | cpu | idle | E0-DF50 | cake | 0.0262 | 10 | 1 |
| 32768 | cpu | idle | E0-DF50 | loadall | 0.0201 | 10 | 0 |
| 32768 | cpu | idle | E0-DR25 | cake | 0.0513 | 10 | 1 2 |
| 32768 | cpu | idle | E0-DR25 | loadall | 0.0295 | 10 | 0 |
| 32768 | cpu | idle | E0-DR50 | cake | 0.0262 | 10 | 1 |
| 32768 | cpu | idle | E0-DR50 | loadall | 0.0202 | 10 | 0 |
| 32768 | cpu | idle | E1s | cake | 0.0346 | 10 | 1 |
| 32768 | cpu | idle | E1s | loadall | 0.0214 | 10 | 0 |
| 32768 | cpu | idle | E1s-DB25 | cake | 0.0322 | 10 | 1 |
| 32768 | cpu | idle | E1s-DB25 | loadall | 0.0162 | 10 | 0 |
| 32768 | cpu | idle | E1s-DB50 | cake | 0.0299 | 10 | 1 |
| 32768 | cpu | idle | E1s-DB50 | loadall | 0.0116 | 10 | 0 |
| 32768 | cpu | idle | E1s-DF25 | cake | 0.0322 | 10 | 1 |
| 32768 | cpu | idle | E1s-DF25 | loadall | 0.0165 | 10 | 0 |
| 32768 | cpu | idle | E1s-DF50 | cake | 0.0299 | 10 | 1 |
| 32768 | cpu | idle | E1s-DF50 | loadall | 0.0115 | 10 | 0 |
| 32768 | cpu | idle | E1s-DR25 | cake | 0.0321 | 10 | 1 |
| 32768 | cpu | idle | E1s-DR25 | loadall | 0.0164 | 10 | 0 |
| 32768 | cpu | idle | E1s-DR50 | cake | 0.0300 | 10 | 1 |
| 32768 | cpu | idle | E1s-DR50 | loadall | 0.0114 | 10 | 0 |

## d3/qwen_nfs（可用；20261020-061920-m10-md-d3nfs-qwen）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | nfs | busy | E0 | cake | 2.7191 | 10 | 32 |
| 32768 | nfs | busy | E0-DB25 | cake | 1.9837 | 10 | 24 |
| 32768 | nfs | busy | E0-DB50 | cake | 1.2517 | 10 | 17 |
| 32768 | nfs | busy | E0-DF25 | cake | 2.7191 | 10 | 32 |
| 32768 | nfs | busy | E0-DF50 | cake | 2.6215 | 10 | 31 |
| 32768 | nfs | busy | E0-DR25 | cake | 2.3150 | 10 | 27 28 29 |
| 32768 | nfs | busy | E0-DR50 | cake | 1.7716 | 10 | 21 22 23 |
| 32768 | nfs | idle | E0 | cake | 1.7354 | 10 | 43 |
| 32768 | nfs | idle | E0-DB25 | cake | 1.2399 | 10 | 33 |
| 32768 | nfs | idle | E0-DB50 | cake | 0.7494 | 10 | 23 |
| 32768 | nfs | idle | E0-DF25 | cake | 1.7369 | 10 | 43 |
| 32768 | nfs | idle | E0-DF50 | cake | 1.7363 | 10 | 43 |
| 32768 | nfs | idle | E0-DR25 | cake | 1.5421 | 10 | 38 39 40 42 |
| 32768 | nfs | idle | E0-DR50 | cake | 1.2870 | 10 | 31 32 33 34 36 |

## m1/llama_u128（可用；20261020-090752-m10-md-m1-llama-u128）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | E0 | cake | 0.7607 | 10 | 13 14 |
| 32768 | b16 | busy | E1s | cake | 0.2640 | 10 | 5 6 |
| 32768 | b16 | idle | E0 | cake | 0.6802 | 10 | 30 31 |
| 32768 | b16 | idle | E1s | cake | 0.2549 | 10 | 8 9 |
| 32768 | b32 | busy | E0 | cake | 2.3211 | 10 | 40 |
| 32768 | b32 | busy | E1s | cake | 0.9125 | 10 | 15 16 17 |
| 32768 | b32 | idle | E0 | cake | 1.9390 | 10 | 75 76 |
| 32768 | b32 | idle | E1s | cake | 0.7655 | 10 | 32 33 |
| 32768 | b48 | busy | E0 | cake | 6.1854 | 10 | 90 91 |
| 32768 | b48 | busy | E1s | cake | 2.6588 | 10 | 44 45 |
| 32768 | b48 | idle | E0 | cake | 4.4211 | 10 | 138 |
| 32768 | b48 | idle | E1s | cake | 2.1610 | 10 | 81 82 |
| 32768 | cpu | busy | E0 | cake | 0.2148 | 10 | 4 |
| 32768 | cpu | busy | E1s | cake | 0.1183 | 10 | 2 |
| 32768 | cpu | idle | E0 | cake | 0.1190 | 10 | 4 5 |
| 32768 | cpu | idle | E1s | cake | 0.0885 | 10 | 1 2 |

## m1/llama_u256（可用；20261020-090043-m10-md-m1-llama-u256）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | E0 | cake | 0.7411 | 10 | 11 12 |
| 32768 | b16 | busy | E1s | cake | 0.2970 | 10 | 5 |
| 32768 | b16 | idle | E0 | cake | 0.6298 | 10 | 24 25 |
| 32768 | b16 | idle | E1s | cake | 0.2543 | 10 | 10 9 |
| 32768 | b32 | busy | E0 | cake | 2.1151 | 10 | 30 |
| 32768 | b32 | busy | E1s | cake | 0.8832 | 10 | 14 |
| 32768 | b32 | idle | E0 | cake | 1.6202 | 10 | 52 53 |
| 32768 | b32 | idle | E1s | cake | 0.7498 | 10 | 27 28 29 |
| 32768 | b48 | busy | E0 | cake | 5.0142 | 10 | 61 |
| 32768 | b48 | busy | E1s | cake | 2.3613 | 10 | 33 |
| 32768 | b48 | idle | E0 | cake | 3.2190 | 10 | 85 |
| 32768 | b48 | idle | E1s | cake | 1.7627 | 10 | 55 56 |
| 32768 | cpu | busy | E0 | cake | 0.1842 | 10 | 3 |
| 32768 | cpu | busy | E1s | cake | 0.1210 | 10 | 1 2 |
| 32768 | cpu | idle | E0 | cake | 0.0956 | 10 | 4 |
| 32768 | cpu | idle | E1s | cake | 0.0573 | 10 | 1 2 |

## m1/llama_u512（可用；20261020-085540-m10-md-m1-llama-u512）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | E0 | cake | 0.6655 | 10 | 8 9 |
| 32768 | b16 | busy | E1s | cake | 0.3174 | 10 | 4 |
| 32768 | b16 | idle | E0 | cake | 0.5746 | 10 | 16 |
| 32768 | b16 | idle | E1s | cake | 0.2604 | 10 | 7 8 |
| 32768 | b32 | busy | E0 | cake | 1.9304 | 10 | 20 21 |
| 32768 | b32 | busy | E1s | cake | 0.8306 | 10 | 10 |
| 32768 | b32 | idle | E0 | cake | 1.3779 | 10 | 32 |
| 32768 | b32 | idle | E1s | cake | 0.6615 | 10 | 18 |
| 32768 | b48 | busy | E0 | cake | 3.9433 | 10 | 38 |
| 32768 | b48 | busy | E1s | cake | 2.0507 | 10 | 22 |
| 32768 | b48 | idle | E0 | cake | 2.3914 | 10 | 48 |
| 32768 | b48 | idle | E1s | cake | 1.4722 | 10 | 33 34 |
| 32768 | cpu | busy | E0 | cake | 0.1616 | 10 | 2 |
| 32768 | cpu | busy | E1s | cake | 0.0834 | 10 | 1 |
| 32768 | cpu | idle | E0 | cake | 0.0953 | 10 | 3 |
| 32768 | cpu | idle | E1s | cake | 0.0637 | 10 | 2 |

## m1/qwen_u128（可用；20261020-092705-m10-md-m1-qwen-u128）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | E0 | cake | 0.6352 | 10 | 14 15 |
| 32768 | b16 | busy | E1s | cake | 0.2917 | 10 | 7 |
| 32768 | b16 | idle | E0 | cake | 0.5691 | 10 | 26 27 28 29 30 |
| 32768 | b16 | idle | E1s | cake | 0.2064 | 10 | 6 7 8 |
| 32768 | b32 | busy | E0 | cake | 1.9243 | 10 | 42 43 |
| 32768 | b32 | busy | E1s | cake | 0.8084 | 10 | 19 20 |
| 32768 | b32 | idle | E0 | cake | 1.6201 | 10 | 75 76 77 78 |
| 32768 | b32 | idle | E1s | cake | 0.6513 | 10 | 28 29 30 |
| 32768 | b48 | busy | E0 | cake | 5.1255 | 10 | 94 |
| 32768 | b48 | busy | E1s | cake | 2.2414 | 10 | 48 49 |
| 32768 | b48 | idle | E0 | cake | 3.6649 | 10 | 139 140 141 |
| 32768 | b48 | idle | E1s | cake | 1.8321 | 10 | 82 83 84 |
| 32768 | cpu | busy | E0 | cake | 0.1245 | 10 | 2 3 |
| 32768 | cpu | busy | E1s | cake | 0.1580 | 10 | 3 4 |
| 32768 | cpu | idle | E0 | cake | 0.0651 | 10 | 2 3 |
| 32768 | cpu | idle | E1s | cake | 0.0882 | 10 | 2 |

## m1/qwen_u256（可用；20261020-092120-m10-md-m1-qwen-u256）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | E0 | cake | 0.5852 | 10 | 11 |
| 32768 | b16 | busy | E1s | cake | 0.2626 | 10 | 5 6 |
| 32768 | b16 | idle | E0 | cake | 0.5351 | 10 | 23 24 |
| 32768 | b16 | idle | E1s | cake | 0.2115 | 10 | 8 9 |
| 32768 | b32 | busy | E0 | cake | 1.7772 | 10 | 29 30 |
| 32768 | b32 | busy | E1s | cake | 0.7692 | 10 | 14 |
| 32768 | b32 | idle | E0 | cake | 1.3798 | 10 | 51 |
| 32768 | b32 | idle | E1s | cake | 0.6450 | 10 | 26 27 28 |
| 32768 | b48 | busy | E0 | cake | 4.2761 | 10 | 60 61 |
| 32768 | b48 | busy | E1s | cake | 1.9954 | 10 | 32 |
| 32768 | b48 | idle | E0 | cake | 2.7699 | 10 | 84 |
| 32768 | b48 | idle | E1s | cake | 1.5169 | 10 | 54 55 |
| 32768 | cpu | busy | E0 | cake | 0.1030 | 10 | 2 |
| 32768 | cpu | busy | E1s | cake | 0.1086 | 10 | 2 |
| 32768 | cpu | idle | E0 | cake | 0.0504 | 10 | 2 3 |
| 32768 | cpu | idle | E1s | cake | 0.0454 | 10 | 1 |

## m1/qwen_u512（可用；20261020-091654-m10-md-m1-qwen-u512）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | E0 | cake | 0.6084 | 10 | 8 9 |
| 32768 | b16 | busy | E1s | cake | 0.2598 | 10 | 4 |
| 32768 | b16 | idle | E0 | cake | 0.4796 | 10 | 16 |
| 32768 | b16 | idle | E1s | cake | 0.1893 | 10 | 7 8 |
| 32768 | b32 | busy | E0 | cake | 1.6229 | 10 | 21 |
| 32768 | b32 | busy | E1s | cake | 0.6879 | 10 | 10 |
| 32768 | b32 | idle | E0 | cake | 1.1557 | 10 | 32 |
| 32768 | b32 | idle | E1s | cake | 0.5548 | 10 | 18 |
| 32768 | b48 | busy | E0 | cake | 3.3442 | 10 | 38 |
| 32768 | b48 | busy | E1s | cake | 1.7135 | 10 | 22 |
| 32768 | b48 | idle | E0 | cake | 2.0282 | 10 | 48 |
| 32768 | b48 | idle | E1s | cake | 1.2513 | 10 | 33 34 |
| 32768 | cpu | busy | E0 | cake | 0.0657 | 10 | 1 |
| 32768 | cpu | busy | E1s | cake | 0.0655 | 10 | 1 |
| 32768 | cpu | idle | E0 | cake | 0.0520 | 10 | 2 |
| 32768 | cpu | idle | E1s | cake | 0.0337 | 10 | 1 |

## m2/llama_fp8_token（可用；20261020-062539-m10-md-m2-llama-fp8_token）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | - | compute_only | 7.7330 | 10 | 64 |
| 32768 | b16 | busy | E0 | cake | 0.6654 | 10 | 8 9 |
| 32768 | b16 | busy | E0 | load_only | 0.7593 | 10 | 64 |
| 32768 | b16 | busy | E1s | cake | 0.4465 | 10 | 5 6 |
| 32768 | b16 | busy | E1s | load_only | 0.4435 | 10 | 64 |
| 32768 | b16 | idle | - | compute_only | 3.6355 | 10 | 64 |
| 32768 | b16 | idle | E0 | cake | 0.5741 | 10 | 16 |
| 32768 | b16 | idle | E0 | load_only | 0.7588 | 10 | 64 |
| 32768 | b16 | idle | E1s | cake | 0.3704 | 10 | 11 |
| 32768 | b16 | idle | E1s | load_only | 0.4297 | 10 | 64 |
| 32768 | b32 | busy | - | compute_only | 7.7389 | 10 | 64 |
| 32768 | b32 | busy | E0 | cake | 1.9394 | 10 | 21 |
| 32768 | b32 | busy | E0 | load_only | 2.7281 | 10 | 64 |
| 32768 | b32 | busy | E1s | cake | 1.2026 | 10 | 14 |
| 32768 | b32 | busy | E1s | load_only | 1.4806 | 10 | 64 |
| 32768 | b32 | idle | - | compute_only | 3.6568 | 10 | 64 |
| 32768 | b32 | idle | E0 | cake | 1.3754 | 10 | 32 |
| 32768 | b32 | idle | E0 | load_only | 2.7275 | 10 | 64 |
| 32768 | b32 | idle | E1s | cake | 0.9444 | 10 | 24 |
| 32768 | b32 | idle | E1s | load_only | 1.4668 | 10 | 64 |
| 32768 | b48 | busy | - | compute_only | 7.7285 | 10 | 64 |
| 32768 | b48 | busy | E0 | cake | 3.9544 | 10 | 38 |
| 32768 | b48 | busy | E0 | load_only | 9.5335 | 10 | 64 |
| 32768 | b48 | busy | E1s | cake | 2.8099 | 10 | 28 29 |
| 32768 | b48 | busy | E1s | load_only | 5.0612 | 10 | 64 |
| 32768 | b48 | idle | - | compute_only | 3.6370 | 10 | 64 |
| 32768 | b48 | idle | E0 | cake | 2.3914 | 10 | 48 |
| 32768 | b48 | idle | E0 | load_only | 9.5347 | 10 | 64 |
| 32768 | b48 | idle | E1s | cake | 1.8693 | 10 | 40 |
| 32768 | b48 | idle | E1s | load_only | 5.0461 | 10 | 64 |
| 32768 | nfs | busy | - | compute_only | 7.7632 | 10 | 64 |
| 32768 | nfs | busy | E0 | cake | 4.3789 | 10 | 41 |
| 32768 | nfs | busy | E0 | load_only | 12.0821 | 10 | 64 |
| 32768 | nfs | busy | E1s | cake | 3.2127 | 10 | 32 |
| 32768 | nfs | busy | E1s | load_only | 6.3964 | 10 | 64 |
| 32768 | nfs | idle | - | compute_only | 3.6340 | 10 | 64 |
| 32768 | nfs | idle | E0 | cake | 2.6440 | 10 | 50 |
| 32768 | nfs | idle | E0 | load_only | 12.0831 | 10 | 64 |
| 32768 | nfs | idle | E1s | cake | 2.0670 | 10 | 43 |
| 32768 | nfs | idle | E1s | load_only | 6.3829 | 10 | 64 |

## m2/llama_int2_kivi（可用；20261020-072219-m10-md-m2-llama-int2_kivi）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | - | compute_only | 7.7024 | 10 | 64 |
| 32768 | b16 | busy | E1s | cake | 0.2352 | 10 | 3 |
| 32768 | b16 | busy | E1s | load_only | 0.2027 | 10 | 64 |
| 32768 | b16 | idle | - | compute_only | 3.6402 | 10 | 64 |
| 32768 | b16 | idle | E1s | cake | 0.1581 | 10 | 5 |
| 32768 | b16 | idle | E1s | load_only | 0.1522 | 10 | 64 |
| 32768 | b32 | busy | - | compute_only | 7.7284 | 10 | 64 |
| 32768 | b32 | busy | E1s | cake | 0.5709 | 10 | 7 |
| 32768 | b32 | busy | E1s | load_only | 0.6197 | 10 | 64 |
| 32768 | b32 | idle | - | compute_only | 3.6457 | 10 | 64 |
| 32768 | b32 | idle | E1s | cake | 0.4502 | 10 | 13 |
| 32768 | b32 | idle | E1s | load_only | 0.5886 | 10 | 64 |
| 32768 | b48 | busy | - | compute_only | 7.7237 | 10 | 64 |
| 32768 | b48 | busy | E1s | cake | 1.4022 | 10 | 16 |
| 32768 | b48 | busy | E1s | load_only | 1.9895 | 10 | 64 |
| 32768 | b48 | idle | - | compute_only | 3.6490 | 10 | 64 |
| 32768 | b48 | idle | E1s | cake | 1.1026 | 10 | 27 |
| 32768 | b48 | idle | E1s | load_only | 1.9587 | 10 | 64 |
| 32768 | nfs | busy | - | compute_only | 7.7335 | 10 | 64 |
| 32768 | nfs | busy | E1s | cake | 1.7018 | 10 | 19 |
| 32768 | nfs | busy | E1s | load_only | 2.5031 | 10 | 64 |
| 32768 | nfs | idle | - | compute_only | 3.6525 | 10 | 64 |
| 32768 | nfs | idle | E1s | cake | 1.2594 | 10 | 30 |
| 32768 | nfs | idle | E1s | load_only | 2.4721 | 10 | 64 |

## m2/llama_int4_kivi（可用；20261020-065408-m10-md-m2-llama-int4_kivi）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | - | compute_only | 7.7331 | 10 | 64 |
| 32768 | b16 | busy | E1s | cake | 0.3185 | 10 | 4 |
| 32768 | b16 | busy | E1s | load_only | 0.2917 | 10 | 64 |
| 32768 | b16 | idle | - | compute_only | 3.6494 | 10 | 64 |
| 32768 | b16 | idle | E1s | cake | 0.2260 | 10 | 7 8 |
| 32768 | b16 | idle | E1s | load_only | 0.2767 | 10 | 64 |
| 32768 | b32 | busy | - | compute_only | 7.7306 | 10 | 64 |
| 32768 | b32 | busy | E1s | cake | 0.8305 | 10 | 10 |
| 32768 | b32 | busy | E1s | load_only | 0.9363 | 10 | 64 |
| 32768 | b32 | idle | - | compute_only | 3.6543 | 10 | 64 |
| 32768 | b32 | idle | E1s | cake | 0.6657 | 10 | 18 |
| 32768 | b32 | idle | E1s | load_only | 0.9233 | 10 | 64 |
| 32768 | b48 | busy | - | compute_only | 7.7422 | 10 | 64 |
| 32768 | b48 | busy | E1s | cake | 2.0512 | 10 | 22 |
| 32768 | b48 | busy | E1s | load_only | 3.1494 | 10 | 64 |
| 32768 | b48 | idle | - | compute_only | 3.6562 | 10 | 64 |
| 32768 | b48 | idle | E1s | cake | 1.4742 | 10 | 33 34 |
| 32768 | b48 | idle | E1s | load_only | 3.1361 | 10 | 64 |
| 32768 | nfs | busy | - | compute_only | 7.7324 | 10 | 64 |
| 32768 | nfs | busy | E1s | cake | 2.4061 | 10 | 25 |
| 32768 | nfs | busy | E1s | load_only | 3.9765 | 10 | 64 |
| 32768 | nfs | idle | - | compute_only | 3.6493 | 10 | 64 |
| 32768 | nfs | idle | E1s | cake | 1.6711 | 10 | 36 37 |
| 32768 | nfs | idle | E1s | load_only | 3.9650 | 10 | 64 |

## m2/llama_kv-int4_kivi-int2_kivi（可用；20261020-070835-m10-md-m2-llama-kv-int4_kivi-int2_kivi）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | - | compute_only | 7.7102 | 10 | 64 |
| 32768 | b16 | busy | E1s | cake | 0.2371 | 10 | 3 4 |
| 32768 | b16 | busy | E1s | load_only | 0.2653 | 10 | 64 |
| 32768 | b16 | idle | - | compute_only | 3.6406 | 10 | 64 |
| 32768 | b16 | idle | E1s | cake | 0.1922 | 10 | 6 7 |
| 32768 | b16 | idle | E1s | load_only | 0.2301 | 10 | 64 |
| 32768 | b32 | busy | - | compute_only | 7.7126 | 10 | 64 |
| 32768 | b32 | busy | E1s | cake | 0.6648 | 10 | 8 9 |
| 32768 | b32 | busy | E1s | load_only | 0.7883 | 10 | 64 |
| 32768 | b32 | idle | - | compute_only | 3.6397 | 10 | 64 |
| 32768 | b32 | idle | E1s | cake | 0.5741 | 10 | 16 |
| 32768 | b32 | idle | E1s | load_only | 0.7581 | 10 | 64 |
| 32768 | b48 | busy | - | compute_only | 7.7074 | 10 | 64 |
| 32768 | b48 | busy | E1s | cake | 1.7634 | 10 | 19 20 |
| 32768 | b48 | busy | E1s | load_only | 2.5805 | 10 | 64 |
| 32768 | b48 | idle | - | compute_only | 3.6461 | 10 | 64 |
| 32768 | b48 | idle | E1s | cake | 1.3062 | 10 | 30 31 |
| 32768 | b48 | idle | E1s | load_only | 2.5498 | 10 | 64 |
| 32768 | nfs | busy | - | compute_only | 7.7023 | 10 | 64 |
| 32768 | nfs | busy | E1s | cake | 2.0450 | 10 | 22 |
| 32768 | nfs | busy | E1s | load_only | 3.2505 | 10 | 64 |
| 32768 | nfs | idle | - | compute_only | 3.6451 | 10 | 64 |
| 32768 | nfs | idle | E1s | cake | 1.4892 | 10 | 34 |
| 32768 | nfs | idle | E1s | load_only | 3.2196 | 10 | 64 |

## m2/llama_mxfp4（可用；20261020-073517-m10-md-m2-llama-mxfp4）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | - | compute_only | 7.7170 | 10 | 64 |
| 32768 | b16 | busy | E1s | cake | 0.3173 | 10 | 4 |
| 32768 | b16 | busy | E1s | load_only | 0.2747 | 10 | 64 |
| 32768 | b16 | idle | - | compute_only | 3.6396 | 10 | 64 |
| 32768 | b16 | idle | E1s | cake | 0.2256 | 10 | 7 |
| 32768 | b16 | idle | E1s | load_only | 0.2439 | 10 | 64 |
| 32768 | b32 | busy | - | compute_only | 7.7326 | 10 | 64 |
| 32768 | b32 | busy | E1s | cake | 0.7439 | 10 | 9 |
| 32768 | b32 | busy | E1s | load_only | 0.8351 | 10 | 64 |
| 32768 | b32 | idle | - | compute_only | 3.6377 | 10 | 64 |
| 32768 | b32 | idle | E1s | cake | 0.5819 | 10 | 16 |
| 32768 | b32 | idle | E1s | load_only | 0.7995 | 10 | 64 |
| 32768 | b48 | busy | - | compute_only | 7.7300 | 10 | 64 |
| 32768 | b48 | busy | E1s | cake | 1.8152 | 10 | 20 |
| 32768 | b48 | busy | E1s | load_only | 2.7324 | 10 | 64 |
| 32768 | b48 | idle | - | compute_only | 3.6397 | 10 | 64 |
| 32768 | b48 | idle | E1s | cake | 1.3380 | 10 | 31 |
| 32768 | b48 | idle | E1s | load_only | 2.6968 | 10 | 64 |
| 32768 | nfs | busy | - | compute_only | 7.7176 | 10 | 64 |
| 32768 | nfs | busy | E1s | cake | 2.1655 | 10 | 23 |
| 32768 | nfs | busy | E1s | load_only | 3.4417 | 10 | 64 |
| 32768 | nfs | idle | - | compute_only | 3.6399 | 10 | 64 |
| 32768 | nfs | idle | E1s | cake | 1.5358 | 10 | 34 35 |
| 32768 | nfs | idle | E1s | load_only | 3.4050 | 10 | 64 |

## m2/qwen_fp8_head（可用；20261020-074914-m10-md-m2-qwen-fp8_head）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | - | compute_only | 6.6021 | 10 | 64 |
| 32768 | b16 | busy | E0 | cake | 0.5545 | 10 | 8 9 |
| 32768 | b16 | busy | E0 | load_only | 0.6319 | 10 | 64 |
| 32768 | b16 | busy | E1s | cake | 0.3331 | 10 | 5 |
| 32768 | b16 | busy | E1s | load_only | 0.3625 | 10 | 64 |
| 32768 | b16 | idle | - | compute_only | 3.1010 | 10 | 64 |
| 32768 | b16 | idle | E0 | cake | 0.4796 | 10 | 16 |
| 32768 | b16 | idle | E0 | load_only | 0.6319 | 10 | 64 |
| 32768 | b16 | idle | E1s | cake | 0.3082 | 10 | 11 |
| 32768 | b16 | idle | E1s | load_only | 0.3463 | 10 | 64 |
| 32768 | b32 | busy | - | compute_only | 6.6084 | 10 | 64 |
| 32768 | b32 | busy | E0 | cake | 1.6187 | 10 | 21 |
| 32768 | b32 | busy | E0 | load_only | 2.2927 | 10 | 64 |
| 32768 | b32 | busy | E1s | cake | 1.0007 | 10 | 14 |
| 32768 | b32 | busy | E1s | load_only | 1.2128 | 10 | 64 |
| 32768 | b32 | idle | - | compute_only | 3.1003 | 10 | 64 |
| 32768 | b32 | idle | E0 | cake | 1.1555 | 10 | 32 |
| 32768 | b32 | idle | E0 | load_only | 2.2925 | 10 | 64 |
| 32768 | b32 | idle | E1s | cake | 0.7618 | 10 | 23 |
| 32768 | b32 | idle | E1s | load_only | 1.1993 | 10 | 64 |
| 32768 | b48 | busy | - | compute_only | 6.6157 | 10 | 64 |
| 32768 | b48 | busy | E0 | cake | 3.3451 | 10 | 38 |
| 32768 | b48 | busy | E0 | load_only | 8.0468 | 10 | 64 |
| 32768 | b48 | busy | E1s | cake | 2.3275 | 10 | 28 |
| 32768 | b48 | busy | E1s | load_only | 4.1521 | 10 | 64 |
| 32768 | b48 | idle | - | compute_only | 3.0915 | 10 | 64 |
| 32768 | b48 | idle | E0 | cake | 2.0230 | 10 | 48 |
| 32768 | b48 | idle | E0 | load_only | 8.0471 | 10 | 64 |
| 32768 | b48 | idle | E1s | cake | 1.5636 | 10 | 39 40 |
| 32768 | b48 | idle | E1s | load_only | 4.1368 | 10 | 64 |
| 32768 | nfs | busy | - | compute_only | 6.6041 | 10 | 64 |
| 32768 | nfs | busy | E0 | cake | 2.7271 | 10 | 31 32 |
| 32768 | nfs | busy | E0 | load_only | 5.2863 | 10 | 64 |
| 32768 | nfs | busy | E1s | cake | 1.8171 | 10 | 23 |
| 32768 | nfs | busy | E1s | load_only | 2.7327 | 10 | 64 |
| 32768 | nfs | idle | - | compute_only | 3.0977 | 10 | 64 |
| 32768 | nfs | idle | E0 | cake | 1.7354 | 10 | 43 |
| 32768 | nfs | idle | E0 | load_only | 5.2862 | 10 | 64 |
| 32768 | nfs | idle | E1s | cake | 1.2644 | 10 | 34 |
| 32768 | nfs | idle | E1s | load_only | 2.7286 | 10 | 64 |

## m2/qwen_int2_kivi（可用；20261020-083330-m10-md-m2-qwen-int2_kivi）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | - | compute_only | 6.6051 | 10 | 64 |
| 32768 | b16 | busy | E1s | cake | 0.1922 | 10 | 3 |
| 32768 | b16 | busy | E1s | load_only | 0.1606 | 10 | 64 |
| 32768 | b16 | idle | - | compute_only | 3.0965 | 10 | 64 |
| 32768 | b16 | idle | E1s | cake | 0.1300 | 10 | 5 |
| 32768 | b16 | idle | E1s | load_only | 0.1268 | 10 | 64 |
| 32768 | b32 | busy | - | compute_only | 6.6139 | 10 | 64 |
| 32768 | b32 | busy | E1s | cake | 0.4687 | 10 | 7 |
| 32768 | b32 | busy | E1s | load_only | 0.5208 | 10 | 64 |
| 32768 | b32 | idle | - | compute_only | 3.0986 | 10 | 64 |
| 32768 | b32 | idle | E1s | cake | 0.3782 | 10 | 13 |
| 32768 | b32 | idle | E1s | load_only | 0.4977 | 10 | 64 |
| 32768 | b48 | busy | - | compute_only | 6.6111 | 10 | 64 |
| 32768 | b48 | busy | E1s | cake | 1.1835 | 10 | 16 |
| 32768 | b48 | busy | E1s | load_only | 1.6869 | 10 | 64 |
| 32768 | b48 | idle | - | compute_only | 3.1023 | 10 | 64 |
| 32768 | b48 | idle | E1s | cake | 0.9257 | 10 | 27 |
| 32768 | b48 | idle | E1s | load_only | 1.6591 | 10 | 64 |
| 32768 | nfs | busy | - | compute_only | 6.6056 | 10 | 64 |
| 32768 | nfs | busy | E1s | cake | 0.9164 | 10 | 12 13 |
| 32768 | nfs | busy | E1s | load_only | 1.1295 | 10 | 64 |
| 32768 | nfs | idle | - | compute_only | 3.0954 | 10 | 64 |
| 32768 | nfs | idle | E1s | cake | 0.7106 | 10 | 22 |
| 32768 | nfs | idle | E1s | load_only | 1.1037 | 10 | 64 |

## m2/qwen_int4_kivi（可用；20261020-081028-m10-md-m2-qwen-int4_kivi）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | - | compute_only | 6.5935 | 10 | 64 |
| 32768 | b16 | busy | E1s | cake | 0.2610 | 10 | 4 |
| 32768 | b16 | busy | E1s | load_only | 0.2619 | 10 | 64 |
| 32768 | b16 | idle | - | compute_only | 3.1008 | 10 | 64 |
| 32768 | b16 | idle | E1s | cake | 0.1872 | 10 | 7 8 |
| 32768 | b16 | idle | E1s | load_only | 0.2320 | 10 | 64 |
| 32768 | b32 | busy | - | compute_only | 6.6061 | 10 | 64 |
| 32768 | b32 | busy | E1s | cake | 0.6868 | 10 | 10 |
| 32768 | b32 | busy | E1s | load_only | 0.8091 | 10 | 64 |
| 32768 | b32 | idle | - | compute_only | 3.1107 | 10 | 64 |
| 32768 | b32 | idle | E1s | cake | 0.5553 | 10 | 18 |
| 32768 | b32 | idle | E1s | load_only | 0.7807 | 10 | 64 |
| 32768 | b48 | busy | - | compute_only | 6.6169 | 10 | 64 |
| 32768 | b48 | busy | E1s | cake | 1.7209 | 10 | 22 |
| 32768 | b48 | busy | E1s | load_only | 2.6794 | 10 | 64 |
| 32768 | b48 | idle | - | compute_only | 3.1067 | 10 | 64 |
| 32768 | b48 | idle | E1s | cake | 1.2537 | 10 | 33 34 |
| 32768 | b48 | idle | E1s | load_only | 2.6504 | 10 | 64 |
| 32768 | nfs | busy | - | compute_only | 6.6083 | 10 | 64 |
| 32768 | nfs | busy | E1s | cake | 1.2651 | 10 | 17 |
| 32768 | nfs | busy | E1s | load_only | 1.7813 | 10 | 64 |
| 32768 | nfs | idle | - | compute_only | 3.0974 | 10 | 64 |
| 32768 | nfs | idle | E1s | cake | 0.9716 | 10 | 28 |
| 32768 | nfs | idle | E1s | load_only | 1.7552 | 10 | 64 |

## m2/qwen_kv-int4_kivi-int2_kivi（可用；20261020-082216-m10-md-m2-qwen-kv-int4_kivi-int2_kivi）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | - | compute_only | 6.5845 | 10 | 64 |
| 32768 | b16 | busy | E1s | cake | 0.1927 | 10 | 3 |
| 32768 | b16 | busy | E1s | load_only | 0.2081 | 10 | 64 |
| 32768 | b16 | idle | - | compute_only | 3.1002 | 10 | 64 |
| 32768 | b16 | idle | E1s | cake | 0.1653 | 10 | 6 7 |
| 32768 | b16 | idle | E1s | load_only | 0.1727 | 10 | 64 |
| 32768 | b32 | busy | - | compute_only | 6.6047 | 10 | 64 |
| 32768 | b32 | busy | E1s | cake | 0.6135 | 10 | 9 |
| 32768 | b32 | busy | E1s | load_only | 0.6658 | 10 | 64 |
| 32768 | b32 | idle | - | compute_only | 3.0973 | 10 | 64 |
| 32768 | b32 | idle | E1s | cake | 0.4800 | 10 | 16 |
| 32768 | b32 | idle | E1s | load_only | 0.6432 | 10 | 64 |
| 32768 | b48 | busy | - | compute_only | 6.6001 | 10 | 64 |
| 32768 | b48 | busy | E1s | cake | 1.5173 | 10 | 19 20 |
| 32768 | b48 | busy | E1s | load_only | 2.1787 | 10 | 64 |
| 32768 | b48 | idle | - | compute_only | 3.0957 | 10 | 64 |
| 32768 | b48 | idle | E1s | cake | 1.1054 | 10 | 30 31 |
| 32768 | b48 | idle | E1s | load_only | 2.1577 | 10 | 64 |
| 32768 | nfs | busy | - | compute_only | 6.5898 | 10 | 64 |
| 32768 | nfs | busy | E1s | cake | 1.0798 | 10 | 15 |
| 32768 | nfs | busy | E1s | load_only | 1.4474 | 10 | 64 |
| 32768 | nfs | idle | - | compute_only | 3.0967 | 10 | 64 |
| 32768 | nfs | idle | E1s | cake | 0.8387 | 10 | 25 |
| 32768 | nfs | idle | E1s | load_only | 1.4318 | 10 | 64 |

## m2/qwen_mxfp4（可用；20261020-084415-m10-md-m2-qwen-mxfp4）

| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |
|:--|:--|:--|:--|:--|--:|--:|:--|
| 32768 | b16 | busy | - | compute_only | 6.6046 | 10 | 64 |
| 32768 | b16 | busy | E1s | cake | 0.1971 | 10 | 3 4 |
| 32768 | b16 | busy | E1s | load_only | 0.2205 | 10 | 64 |
| 32768 | b16 | idle | - | compute_only | 3.0958 | 10 | 64 |
| 32768 | b16 | idle | E1s | cake | 0.1858 | 10 | 6 7 |
| 32768 | b16 | idle | E1s | load_only | 0.1755 | 10 | 64 |
| 32768 | b32 | busy | - | compute_only | 6.6016 | 10 | 64 |
| 32768 | b32 | busy | E1s | cake | 0.6122 | 10 | 9 |
| 32768 | b32 | busy | E1s | load_only | 0.7012 | 10 | 64 |
| 32768 | b32 | idle | - | compute_only | 3.1013 | 10 | 64 |
| 32768 | b32 | idle | E1s | cake | 0.5174 | 10 | 16 17 |
| 32768 | b32 | idle | E1s | load_only | 0.6779 | 10 | 64 |
| 32768 | b48 | busy | - | compute_only | 6.5919 | 10 | 64 |
| 32768 | b48 | busy | E1s | cake | 1.5271 | 10 | 20 |
| 32768 | b48 | busy | E1s | load_only | 2.3013 | 10 | 64 |
| 32768 | b48 | idle | - | compute_only | 3.0983 | 10 | 64 |
| 32768 | b48 | idle | E1s | cake | 1.1317 | 10 | 31 |
| 32768 | b48 | idle | E1s | load_only | 2.2807 | 10 | 64 |
| 32768 | nfs | busy | - | compute_only | 6.6096 | 10 | 64 |
| 32768 | nfs | busy | E1s | cake | 1.1670 | 10 | 16 |
| 32768 | nfs | busy | E1s | load_only | 1.5285 | 10 | 64 |
| 32768 | nfs | idle | - | compute_only | 3.0951 | 10 | 64 |
| 32768 | nfs | idle | E1s | cake | 0.8802 | 10 | 26 |
| 32768 | nfs | idle | E1s | load_only | 1.5116 | 10 | 64 |

