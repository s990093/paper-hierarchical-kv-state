"""main_report_pdf.py — icpe-main：用論文稿的 PDF 產生器（docs/icpe_paper_20261030/build_pdf.py，原樣 import、不改）把報告 Markdown 轉成 A4 PDF。

  python -I code/icpe/main_report_pdf.py --in <REPORT.md> --out <REPORT.pdf> [--max-bytes 2000000] [--png-pages all --png-dir <dir>]

為什麼要包一層：build_pdf.py 用 fc-list 找 "Noto Sans TC"；acvlab 只有 Noto Sans CJK 的 .ttc（合集）。
這裡用 fontTools 從 /usr/share/fonts/opentype/noto/NotoSansCJK-{Regular,Bold}.ttc 取出繁中（TC）那一套，
改名成 "Noto Sans TC" 存到暫存目錄（結束就刪），再讓 build_pdf._fc_files 回傳這兩個檔；其餘流程完全照 build_pdf.py。
"""
from __future__ import annotations

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "docs", "icpe_paper_20261030"))

TTC = {"Regular": "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
       "Bold": "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"}


def extract_tc(tmpdir):
    from fontTools.ttLib import TTCollection
    out = []
    for sub, path in TTC.items():
        coll = TTCollection(path)
        pick = None
        for f in coll.fonts:
            fam = f["name"].getDebugName(1) or ""
            if fam.startswith("Noto Sans CJK TC"):
                pick = f
                break
        if pick is None:
            sys.exit(f"ERROR: {path} 裡找不到 Noto Sans CJK TC")
        nm = pick["name"]
        for rec in list(nm.names):
            if rec.nameID in (1, 16):
                nm.setName("Noto Sans TC", rec.nameID, rec.platformID, rec.platEncID, rec.langID)
            elif rec.nameID in (2, 17):
                nm.setName(sub, rec.nameID, rec.platformID, rec.platEncID, rec.langID)
            elif rec.nameID == 4:
                nm.setName(f"Noto Sans TC {sub}", 4, rec.platformID, rec.platEncID, rec.langID)
            elif rec.nameID == 6:
                nm.setName(f"NotoSansTC-{sub}", 6, rec.platformID, rec.platEncID, rec.langID)
        dst = os.path.join(tmpdir, f"NotoSansTC-{sub}.otf")
        pick.save(dst)
        out.append(dst)
    return out


def main():
    import build_pdf as BP
    with tempfile.TemporaryDirectory(prefix="tc-font-") as td:
        files = extract_tc(td)
        orig = BP._fc_files
        BP._fc_files = lambda family: files if family == "Noto Sans TC" else orig(family)
        BP.main()


if __name__ == "__main__":
    main()
