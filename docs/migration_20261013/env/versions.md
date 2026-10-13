# MI300X 環境版本（2026-10-13 匯出，來源：本機指令輸出）

| 項目 | 值 | 來源 |
|:--|:--|:--|
| GPU | AMD Instinct MI300X，gfx942，192 GiB，SPX，SR-IOV 1VF | REPORT_MI300X.md／results/hw_mi300x.json |
| 配額 | CPU 32、RAM 434.8 GiB、/dev/shm 178.8 GiB | 同上（cgroup） |
| ROCm | 7.2.2 | /opt/rocm/.info/version |
| Python | 3.12.3（四個 venv 都是） | venv/*/bin/python -V |
| venv tiara（主力，phase 1／m7–m9） | torch 2.10.0+rocm7.0、vLLM v0.19.1 editable @ b1388b1fbf5aaef47937fabe98931211684666a6、transformers 5.17.0 | pip_freeze_tiara.txt |
| venv tiara-v028（C6、F1、D2） | torch 2.12.1+rocm7.2、vLLM v0.28.0 editable @ 2cf0a6915ce544dc493a0990f2ea38d81601128a | pip_freeze_tiara-v028.txt |
| venv tiara-v022 | vLLM v0.22.0（只用來比對，主結果沒用到） | pip_freeze_tiara-v022.txt |
| venv tools | rocmtop 等監看工具 | pip_freeze_tools.txt |
| torch index | https://download.pytorch.org/whl/rocm7.0 | /mlsteam/workspace/.torch_index |
| vLLM 原始碼 | 三個 tree 都沒有本地修改（git status 乾淨） | src/vllm*、git status |

> 這些是 **ROCm** 版本，3090（CUDA）不能直接照裝；當作「當時跑出那些數字的環境」記錄。
