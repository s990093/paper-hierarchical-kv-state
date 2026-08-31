# notebooks

## 檔案

| 檔案 | 是什麼 |
|---|---|
| `analysis.py` | **所有讀檔、衍生計算與繪圖的唯一實作。** notebook 與論文的圖都來自這裡 |
| `analysis.ipynb` | 呈現層。只負責排版與敘述，不含計算邏輯 |
| `figures/` | `python analysis.py --figures` 產生的 PDF/PNG，供 `main.tex` 引用 |
| `measured_summary.json` | 所有實測數字的單一匯出檔，供論文表格引用 |

## 為什麼計算不寫在 cell 裡

論文的圖與 notebook 的圖**必須來自同一段程式**，否則兩者會漂移，
而漂移不會有任何錯誤訊息——只會在某天發現論文的數字跟 notebook 對不上。

## 執行

```bash
P=/ssd7/hungwei/paper-hkv/venv/vllm/bin/python

$P notebooks/analysis.py                  # 印出關鍵數字
$P notebooks/analysis.py --figures        # 另外產生 figures/
$P -m jupyter lab notebooks/              # 互動式，選 kernel「Tiara (vLLM venv)」
```

### Kernel

已註冊具名 kernel `tiara-vllm`（顯示為 **Tiara (vLLM venv)**），
在 Jupyter 的下拉選單直接選即可，不必自己設環境變數。

它在啟動時就帶好 `PAPER_HKV_BIG`、`HF_HOME` 與各項快取路徑，
並且**刻意把 `CUDA_VISIBLE_DEVICES` 設成空字串**——分析用的 kernel 不該碰 GPU，
否則它會默默佔住一張卡，而這台機器是共用的。

重裝：
```bash
$P -m ipykernel install --user --name tiara-vllm --display-name "Tiara (vLLM venv)"
# 然後把 env 區塊補回 ~/.local/share/jupyter/kernels/tiara-vllm/kernel.json
```

## 不使用預設值

`analysis.py` 的每個 loader 讀不到檔就 `FileNotFoundError`，**不回退到任何硬編數字**。
與 `EXPERIMENT_PLAN.md` §0 禁令 1 一致：沒量到就是沒量到。

被 GPU 插隊污染的列（`contaminated == True`）在讀取時直接跳過，不進任何分析。
