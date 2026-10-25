# 第三階段的 run 文件（H200）

> 每個閘一個 md：**最上面是事先登記的判準與量法（commit＋`date -u` 之後不改）**，結果與偏離附在後面。
> 規則照 `docs/h200/icpe/README.md`：登記在看到任何正式數字之前 commit；冒煙（smoke）只驗程式、不進 `results/`、不看分數；
> 和登記不同的地方一律寫成「偏離 N」附在文件後面，說明為什麼、在看到哪些數字之前或之後。

| 閘 | 文件 | 程式 | 結果 | 狀態 |
|:--|:--|:--|:--|:--|
| A0（＝G-M01/02） | [A0_progressive_restore.md](A0_progressive_restore.md) | `code/s3/a0.py`、`code/s3/a0_gpu.py` | `results/s3_h200/a0/` | 已登記，待冒煙 |
| B0＋G-M08（量時間） | [B0_interference.md](B0_interference.md) | `code/s3/b0.py`、`code/s3/bgengine.py` | `results/s3_h200/b0/` | 已登記，待冒煙 |
| W0（量時間） | [W0_write_cost.md](W0_write_cost.md) | `code/s3/w0.py` | `results/s3_h200/w0/` | 已登記 |
| G-fio（量時間） | [G_fio.md](G_fio.md) | `code/s3/gfio.py` | `results/s3_h200/gfio/` | 已登記 |

runner：`code/r_h200_s3.sh`；RUNLOG：`results/RUNLOG_H200_S3.md`；GPU 排程規則見 memory `stage3-gpu-scheduling`。
