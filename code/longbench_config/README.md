LongBench v1 的官方設定檔與 `metrics.py`、`pred.py`（THUDM/LongBench，MIT 授權；後兩支只給 m10_longbench 的自測對拍用），逐字複製自 MI300X 的 `/mlsteam/data/tiara/datasets/longbench/config/`（2026-09-15 下載）。
`code/h200_setup_data.sh` 會把 json 放到 `$M10_LONGBENCH/config/`、兩支 .py 放到 `$M10_LONGBENCH/`，`code/m10_longbench.py` 從那裡讀。
