"""REPORT.md → PDF（WeasyPrint）。只讀 docs/precision_20261013/report_stage2/，不讀 results/。

環境（10/20 建；容器重啟後系統套件會消失，venv 和字型留在 /mlsteam/workspace）：
  apt-get install -y libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0 libfontconfig1 fontconfig fonts-noto-cjk
  （沒網路時字型可從 /mlsteam/workspace/fonts/noto-cjk 複製到 /usr/share/fonts 再 fc-cache）
  python3 -m venv /mlsteam/workspace/venv/pdf && /mlsteam/workspace/venv/pdf/bin/pip install weasyprint markdown
用法：
  nice -n 19 /mlsteam/workspace/venv/pdf/bin/python code/report_stage2/make_pdf.py [--out /mlsteam/workspace/reports/stage2_report.pdf]
PDF 含 30 張 PNG 圖，超過 1 MB（CLAUDE.md §2），所以不進 git，預設寫到 repo 外。
"""
import argparse
import os
import re
from datetime import datetime, timezone

import markdown
from weasyprint import CSS, HTML

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RDIR = os.path.join(REPO, "docs", "precision_20261013", "report_stage2")
GH = "https://github.com/s990093/paper-hierarchical-kv-state/blob/main/"

CSS_TEXT = """
@page {
  size: A4; margin: 20mm 17mm 18mm 17mm;
  @top-center { content: string(doctitle); font: 7.5pt "Noto Sans CJK TC"; color: #777; }
  @bottom-center { content: counter(page) " / " counter(pages); font: 8pt "Noto Sans CJK TC"; color: #777; }
}
@page :first { @top-center { content: none; } }
html { font-family: "Noto Serif CJK TC", serif; font-size: 9.6pt; line-height: 1.6; color: #1a1a1a; }
h1 { string-set: doctitle content(); font-family: "Noto Sans CJK TC", sans-serif; font-size: 19pt; line-height: 1.3;
     margin: 0 0 6mm 0; color: #0b3a5b; }
h2 { font-family: "Noto Sans CJK TC", sans-serif; font-size: 14pt; color: #0b3a5b; border-bottom: 1.2pt solid #0b3a5b;
     padding-bottom: 1.5mm; margin: 9mm 0 3mm 0; page-break-after: avoid; }
h3 { font-family: "Noto Sans CJK TC", sans-serif; font-size: 11.5pt; color: #134e74; margin: 6mm 0 2mm 0; page-break-after: avoid; }
h2.newpage { page-break-before: always; }
p { margin: 1.5mm 0; }
ul, ol { margin: 1mm 0 1.5mm 0; padding-left: 5.5mm; }
li { margin: 0.6mm 0; }
li > ul, li > ol { margin: 0.5mm 0; }
strong { color: #000; }
code { font-family: "Noto Sans Mono CJK TC", monospace; font-size: 8pt; background: #f2f4f6; padding: 0 0.8mm; border-radius: 1mm; }
pre { background: #f2f4f6; padding: 2.5mm; font-size: 7.6pt; line-height: 1.45; white-space: pre-wrap; border-radius: 1.5mm; }
pre code { background: none; padding: 0; }
blockquote { margin: 2mm 0; padding: 1.5mm 3.5mm; border-left: 2.2pt solid #4a90c2; background: #f4f8fb; color: #333; font-size: 8.8pt; }
blockquote p { margin: 0.8mm 0; }
table { border-collapse: collapse; margin: 2.5mm 0 3mm 0; font-size: 8.1pt; line-height: 1.4; width: auto; max-width: 100%;
        page-break-inside: auto; }
thead { display: table-header-group; }
tr { page-break-inside: avoid; }
th { background: #e6eef5; font-family: "Noto Sans CJK TC", sans-serif; font-weight: bold; }
th, td { border: 0.5pt solid #b9c6d2; padding: 1mm 1.8mm; vertical-align: top; }
td code, th code { font-size: 7.3pt; }
img { display: block; max-width: 100%; max-height: 170mm; margin: 3mm auto 1mm auto; }
p.fig { text-align: center; margin: 0; page-break-inside: avoid; page-break-after: avoid; }
p.cap { font-size: 8.6pt; margin-top: 1mm; page-break-before: avoid; }
a { color: #1f5f8b; text-decoration: none; }
hr { border: none; border-top: 0.5pt solid #ccc; margin: 5mm 0; }
.cover-meta { font-family: "Noto Sans CJK TC", sans-serif; font-size: 9pt; color: #555; margin-bottom: 4mm; }
.toc { font-family: "Noto Sans CJK TC", sans-serif; font-size: 9.2pt; }
.toc ul { list-style: none; padding-left: 0; }
.toc ul ul { padding-left: 6mm; font-size: 8.6pt; color: #333; }
.toc li { margin: 0.8mm 0; }
.toc a::after { content: leader('.') target-counter(attr(href), page); color: #777; }
.toc-title { font-family: "Noto Sans CJK TC", sans-serif; font-size: 13pt; color: #0b3a5b; margin: 6mm 0 2mm 0; }
"""


def md_to_html(text):
    # 相對連結：.md／資料夾 → GitHub（PDF 裡點得到）；圖片照舊用相對路徑（base_url＝報告目錄）
    def fix_link(m):
        label, href = m.group(1), m.group(2)
        if re.match(r"^(https?:|#|figs/)", href):
            return m.group(0)
        p = os.path.normpath(os.path.join("docs/precision_20261013/report_stage2", href.split("#")[0]))
        return f"[{label}]({GH}{p})"
    text = re.sub(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)\)", fix_link, text)
    # GitHub 允許段落後面直接接列表，Python-Markdown 要空一行，不然會併成同一段（圖說後的條列）
    item = re.compile(r"^\s*(?:[-*+]|\d+\.)\s")
    out = []
    for ln in text.split("\n"):
        if item.match(ln) and out and out[-1].strip() and not item.match(out[-1]) and not out[-1].startswith(" "):
            out.append("")
        out.append(ln)
    text = "\n".join(out)
    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists", "toc", "attr_list"],
                           extension_configs={"toc": {"toc_depth": "2-3"}}, tab_length=2)
    body = md.convert(text)
    return body, md.toc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(RDIR, "REPORT.md"))
    ap.add_argument("--out", default="/mlsteam/workspace/reports/stage2_report.pdf")
    a = ap.parse_args()
    text = open(a.src, encoding="utf-8").read()
    body, toc = md_to_html(text)
    # 圖：<p><img></p> 加 class；緊接的「圖 Fxx」說明段落縮小字
    body = re.sub(r"<p>(<img [^>]+>)</p>", r'<p class="fig">\1</p>', body)
    body = re.sub(r"<p><strong>圖 (F\d+)</strong>", r'<p class="cap"><strong>圖 \1</strong>', body)
    # 封面：標題＋開頭說明框之後放目錄，正文從「摘要」換頁開始
    built = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    meta = f'<div class="cover-meta">Tiara 研究・第二階段（精度線）・AMD MI300X　｜　PDF 產生時間 {built}　｜　來源 docs/precision_20261013/report_stage2/REPORT.md</div>'
    i = body.find("<h2")
    head, rest = body[:i], body[i:]
    head = head.replace("</h1>", "</h1>" + meta, 1)
    toc_html = f'<div class="toc-title">目錄</div><div class="toc">{toc}</div>'
    rest = re.sub(r"<h2", '<h2 class="newpage"', rest, count=1)
    for h in ("1　問題與目標", "3　主線結果", "4　探索實驗", "5　討論與想法", "附錄 A"):   # 大章節換頁
        rest = re.sub(r"<h2 (id=\"[^\"]*\")>" + re.escape(h), r'<h2 class="newpage" \1>' + h, rest, count=1)
    html = f'<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><title>第二階段最終報告</title></head><body>{head}{toc_html}{rest}</body></html>'
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    HTML(string=html, base_url=RDIR + "/").write_pdf(a.out, stylesheets=[CSS(string=CSS_TEXT)])
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
