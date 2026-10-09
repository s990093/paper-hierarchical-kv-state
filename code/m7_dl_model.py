"""下載 MHA 模型（08 消融：Cake 原文的 LongAlpaca-7B）。只抓 safetensors 與設定檔。"""
import sys

from huggingface_hub import snapshot_download

print(snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "tokenizer.model", "*.txt"]))
