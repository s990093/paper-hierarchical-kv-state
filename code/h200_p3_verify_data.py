"""h200_p3_verify_data.py — 第三階段的原始 trace：每個檔的 sha256 和下載時寫的 README.txt 相同（2026-10-19）。

  python -I code/h200_p3_verify_data.py <TIARA_DATASETS> mooncake cachewise bailian burstgpt wildchat freeinference

平台 C（H200）的原始 trace 放在 compute node 的本地 /tmp（不保證保留、別人也寫得到同一顆碟），所以每個 part 用之前都驗一次。
README.txt 由 code/setup_3090_data.sh 的 readme() 寫：「  <sha256>  <bytes>  <相對路徑>」。缺檔、大小或 sha256 不對 → exit 1（規則 7：查不到不是沒事）。
"""
import hashlib
import os
import re
import sys

DIRS = dict(mooncake="mooncake", cachewise="cachewise-coding-traces", bailian="qwen-bailian-usagetraces-anon", burstgpt="burstgpt",
            wildchat="wildchat-1m", freeinference="freeinference_agentic_trace")
LINE = re.compile(r"^\s+([0-9a-f]{64})\s+(\d+)\s+(\S.*)$")


def main():
    root, items = sys.argv[1], sys.argv[2:]
    bad = 0
    for it in items:
        d = os.path.join(root, DIRS[it])
        rd = os.path.join(d, "README.txt")
        if not os.path.exists(rd):
            print(f"[verify] {it}: 沒有 {rd}")
            bad += 1
            continue
        rows = [m.groups() for m in (LINE.match(l.rstrip("\n")) for l in open(rd, encoding="utf-8")) if m]
        nb, tot = 0, 0
        for sha, nbytes, rel in rows:
            p = os.path.join(d, rel)
            if not os.path.exists(p) or os.path.getsize(p) != int(nbytes):
                print(f"[verify] {it}: {rel} 不見或大小不對")
                nb += 1
                continue
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for b in iter(lambda: f.read(1 << 24), b""):
                    h.update(b)
            if h.hexdigest() != sha:
                print(f"[verify] {it}: {rel} sha256 不對")
                nb += 1
            tot += int(nbytes)
        if not rows:
            print(f"[verify] {it}: README.txt 裡沒有檔案清單")
            nb += 1
        print(f"[verify] {it}: {len(rows)} 個檔、{tot / 2**30:.3f} GiB，壞 {nb}")
        bad += nb
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
