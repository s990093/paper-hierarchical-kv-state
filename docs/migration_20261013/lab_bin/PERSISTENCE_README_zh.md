# ⚠ 這台 Lab 哪裡的資料會永久保存（2026-10-08）

平台是 Kubernetes：Lab 一關、重開、改掛載或改規格，container 整個重建。

| 路徑 | 類型 | 重開後 |
|---|---|---|
| /mlsteam/workspace        | NFS 950G（掛載） | ✅ 保留：repo、venv、node、npm-global、.claude 設定 |
| /mlsteam/data/tiara       | NFS（掛載，要在 Lab 設定裡開著） | ✅ 保留：hf-cache、models、datasets、runs、profiles |
| / （含 /root、/tmp、/usr） | overlay 本地碟 | ❌ 全部消失：apt 裝的、/root/.bashrc、/tmp |

規則：
* 實驗輸出一律寫 $TIARA_RUNS = /mlsteam/data/tiara/runs，**不要寫 /tmp 或 /root**。
* 不要手動 apt/pip 裝東西；要裝就寫進 /mlsteam/workspace/bin/*.sh，重開後跑 lab_up.sh 還原。
* 重開後：bash /mlsteam/workspace/bin/lab_up.sh；Mac 上 bash code/mi300x/set_ssh_port.sh <新port>。
