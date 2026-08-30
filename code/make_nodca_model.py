#!/usr/bin/env python3
"""建立 Qwen2.5-7B-Instruct-1M 的 no-DCA 變體（symlink + 改過的 config.json）。

## 為什麼要這個

vLLM 0.28.0 的 V1 engine **無法載入啟用 Dual Chunk Attention 的 Qwen2.5-1M**：

```
TypeError: FlashAttentionImpl.__init__() got an unexpected keyword argument 'layer_idx'
```

`model_executor/models/qwen2.py:189` 在 `dual_chunk_attention_config` 為真時，
把 `layer_idx` 與 `dual_chunk_attention_config` 傳給 `Attention`，
而 V1 的 attention backend 不吃這兩個參數。`v1/attention/` 底下也沒有任何
dual-chunk backend。**結論：vLLM 0.28.0 的 V1 沒有可用的 DCA 路徑。**

用 `--hf-overrides` 繞不過去：
* `{"dual_chunk_attention_config": null}` → `verify_dual_chunk_attention_config`
  對 None 做 item assignment → `TypeError: 'NoneType' object does not support item assignment`
* `{"dual_chunk_attention_config": {}}` → attention 路徑過了，但 rotary 的
  `get_rope()` 用的是 `is not None` 而非真值判斷，仍進 DCA 分支 →
  `DualChunkRotaryEmbedding.__init__() missing 2 required positional arguments`

**必須把這個 key 整個拿掉**，而 `--hf-overrides` 只能覆寫、不能刪除。
所以做一份本地的 model 目錄：權重用 symlink（不複製 15 GB），config.json 改寫。

## 這樣改代表什麼（必須誠實理解）

拿掉 DCA 之後，這個模型就是它**真正被訓練的樣子**：
`original_max_position_embeddings = 262144`。1M 的宣稱長度本來就是靠 DCA 外推來的。

對本研究**反而更乾淨**：`EXPERIMENT_PLAN.md` §3 要求「不要為了掃更長 context 而開 YaRN，
那會讓品質下降的來源無法歸因」。DCA 是不同機制但同樣的問題。拿掉它之後，
所有評測長度都在模型的真實訓練範圍內，ε（品質退化）可以乾淨地歸因到 KV 放置策略。

**代價**：評測長度上限從宣稱的 1M 降為 262,144。
在 24 GB 單卡上這不構成限制——BF16 權重下的懸崖遠低於此。

用法:
    python code/make_nodca_model.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BIG = Path(os.environ.get("PAPER_HKV_BIG", "/ssd7/hungwei/paper-hkv"))
# 預設是 BF16 原版；用 --src / --dst 可指向 AWQ 版本。
# AWQ 版同樣帶 dual_chunk_attention_config，同樣要拿掉才跑得起來。
SRC_REPO = "Qwen/Qwen2.5-7B-Instruct-1M"
DST = BIG / "models" / "Qwen2.5-7B-Instruct-1M-noDCA"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_REPO, help="HF repo id")
    ap.add_argument("--dst", default=str(DST), help="輸出目錄")
    a = ap.parse_args()
    dst_dir = Path(a.dst)

    os.environ.setdefault("HF_HOME", str(BIG / "hf-cache/huggingface"))
    from huggingface_hub import snapshot_download

    src = Path(snapshot_download(a.src, local_files_only=True))
    print(f"source snapshot: {src}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    linked, patched = 0, []
    for f in sorted(src.iterdir()):
        dst = dst_dir / f.name
        if f.name == "config.json":
            cfg = json.loads(f.read_text())
            removed = cfg.pop("dual_chunk_attention_config", None)
            # 把宣稱長度改成模型真正訓練的長度，不要留一個做不到的 1,010,000。
            old_max = cfg.get("max_position_embeddings")
            if removed and "original_max_position_embeddings" in removed:
                cfg["max_position_embeddings"] = removed["original_max_position_embeddings"]
            cfg["_tiara_note"] = (
                "dual_chunk_attention_config removed: vLLM 0.28.0 V1 engine has no "
                "working DCA path (FlashAttentionImpl rejects layer_idx). "
                f"max_position_embeddings lowered {old_max} -> "
                f"{cfg['max_position_embeddings']} (the model's real trained length). "
                "See results/RUNLOG.md and code/make_nodca_model.py."
            )
            cfg["_tiara_removed_dual_chunk_attention_config"] = removed
            dst.unlink(missing_ok=True)
            dst.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
            patched.append((old_max, cfg["max_position_embeddings"], removed))
        else:
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(f.resolve())
            linked += 1

    print(f"symlinked {linked} files -> {dst_dir}")
    for old, new, removed in patched:
        print(f"config.json patched:")
        print(f"  removed dual_chunk_attention_config = {removed}")
        print(f"  max_position_embeddings: {old:,} -> {new:,}")
    print(f"\nuse this path as the model id:\n  {dst_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
