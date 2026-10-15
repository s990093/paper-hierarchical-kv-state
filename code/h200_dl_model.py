"""h200_dl_model.py — 在 H200 上下載固定 revision 的模型；下載前先檢查 lab 的 /work 配額（2026-10-16）

  python code/h200_dl_model.py <repo> <revision> --need-gib 58 [--margin-gib 15] [--allow-pattern '*.json' ...]

為什麼要檢查：/work 的配額是 lab 共用的（quota -g）。寫滿時，lab 裡其他人的 job 也會一起寫不進去。
所以「下載後剩不到 margin」就停下來回報，不硬下載（CLAUDE.md 規則 2）；讀不到 quota 也當成不能下載（規則 7）。
已經有這個 revision 的 safetensors 時不檢查配額，只讓 snapshot_download 核對、補缺的檔。
revision 用 docs/migration_20261013/inventory/models.tsv 的值（和 MI300X 同一版）。
"""
import argparse
import os
import subprocess
import sys

from huggingface_hub import snapshot_download


def quota_gib():
    try:
        out = subprocess.run(["quota", "-gw"], capture_output=True, text=True).stdout.splitlines()
    except FileNotFoundError:
        sys.exit("[h200_dl_model] 這台沒有 quota 指令，讀不到 lab 的配額（查不到不等於有空間）→ 停。改在 head node 下載")
    rows = [l.split() for l in out if len(l.split()) >= 4 and l.split()[1].isdigit()]
    if not rows:
        sys.exit(f"[h200_dl_model] 讀不到 quota -gw 的數字（查不到不等於有空間）：{out}")
    return int(rows[0][1]) / 2**20, int(rows[0][3]) / 2**20      # KiB → GiB：(used, limit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("revision")
    ap.add_argument("--need-gib", type=float, required=True, help="要下載的大小（MI300X 上 du 的值）")
    ap.add_argument("--margin-gib", type=float, default=15.0, help="下載後 lab 至少要剩多少")
    ap.add_argument("--allow-pattern", nargs="*", default=None)
    a = ap.parse_args()
    snap = os.path.join(os.environ["HF_HOME"], "hub", "models--" + a.repo.replace("/", "--"), "snapshots", a.revision)
    if os.path.isdir(snap) and any(f.endswith(".safetensors") for f in os.listdir(snap)):
        print(f"[h200_dl_model] 已有 {snap} → 不檢查配額，只核對", flush=True)
    else:
        used, limit = quota_gib()
        left = limit - used - a.need_gib
        print(f"[h200_dl_model] 配額 {used:.1f}／{limit:.0f} GiB；下載約 {a.need_gib:.0f} GiB 後剩 {left:.1f}（要 ≥{a.margin_gib:.0f}）", flush=True)
        if left < a.margin_gib:
            sys.exit(f"[h200_dl_model] 配額不夠 → 停、回報（剩 {limit - used:.1f} GiB）")
    p = snapshot_download(a.repo, revision=a.revision, allow_patterns=a.allow_pattern)
    print("[h200_dl_model]", a.repo, a.revision, "->", p, flush=True)
    used, limit = quota_gib()
    print(f"[h200_dl_model] 下載後配額 {used:.1f}／{limit:.0f} GiB", flush=True)


if __name__ == "__main__":
    main()
