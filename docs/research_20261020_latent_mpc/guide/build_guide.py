"""把 content.py 排成 PDF（PyMuPDF Story，不需要 TeX 或瀏覽器）。

用法：python build_guide.py OUT.pdf
圖片：論文截圖在 FIG_DIR、示意圖在 DIA_DIR（由 make_diagrams.py 產生）。
注意：MuPDF 的 CSS `page-break-before: always` 會讓 Story 無限迴圈，所以每一章用獨立的 Story 從新頁開始。
"""
import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).parent))
from content import CHAPTERS  # noqa: E402

DATA = Path("/mlsteam/data/tiara/papers/latent_mpc_20261020")
FONT_DIR = Path("/mlsteam/data/tiara/fonts")
FIG_DIR = DATA / "fig"
DIA_DIR = DATA / "diagrams"

PAGE = pymupdf.paper_rect("a4")
BODY = PAGE + (54, 50, -54, -46)

CSS = """
@font-face {font-family: sans; src: url(NotoSansCJKtc-Regular.otf);}
@font-face {font-family: sans; src: url(NotoSansCJKtc-Bold.otf); font-weight: bold;}
@font-face {font-family: serif; src: url(NotoSerifCJKtc-Bold.otf);}
body {font-family: sans; font-size: 10pt; line-height: 1.62; color: #171a24;}
h1 {font-family: serif; font-size: 19pt; color: #3d4fc0; margin-top: 0; margin-bottom: 10pt; padding-bottom: 4pt; border-bottom: 1pt solid #c9cede;}
h2 {font-family: serif; font-size: 13.5pt; color: #171a24; margin-top: 14pt; margin-bottom: 6pt;}
h3 {font-size: 10.8pt; font-weight: bold; color: #3d4fc0; margin-top: 10pt; margin-bottom: 4pt;}
p {margin-top: 0; margin-bottom: 6pt;}
ul, ol {margin-top: 2pt; margin-bottom: 6pt;}
li {margin-bottom: 2pt;}
b {font-weight: bold;}
.figp {text-align: center; margin-top: 6pt; margin-bottom: 2pt;}
.cap {font-size: 8.3pt; color: #5b6070; text-align: center; margin-bottom: 10pt; line-height: 1.45;}
.ev {font-size: 7.4pt; color: #868b9c;}
.small {font-size: 7.4pt;}
.quote {font-family: serif; color: #4a2f1f; margin-left: 10pt;}
div.key {background-color: #e9ecfa; border: 0.8pt solid #3d4fc0; padding: 6pt 9pt 2pt 9pt; margin-top: 6pt; margin-bottom: 9pt;}
div.story {background-color: #fbf1ea; border: 0.8pt solid #b4521e; padding: 6pt 9pt 2pt 9pt; margin-top: 6pt; margin-bottom: 9pt;}
div.warn {background-color: #f8efda; border: 0.8pt solid #8f5b00; padding: 6pt 9pt 2pt 9pt; margin-top: 6pt; margin-bottom: 9pt;}
div.data {background-color: #edf6f0; border: 0.8pt solid #2e7d4f; padding: 6pt 9pt 2pt 9pt; margin-top: 6pt; margin-bottom: 9pt;}
table {border-collapse: collapse; width: 100%; margin-top: 4pt; margin-bottom: 8pt; font-size: 8.7pt; line-height: 1.45;}
th {border: 0.5pt solid #b9bfcf; border-bottom: 1.2pt solid #3d4fc0; padding: 3pt 4pt; text-align: left; font-weight: bold; color: #3d4fc0;}
td {border: 0.5pt solid #c9cede; padding: 3pt 4pt; vertical-align: top;}
table.toc td {border: none; padding: 1.5pt 2pt; font-size: 10pt;}
table.toc td.l1 {font-weight: bold; padding-top: 6pt;}
table.toc td.l2 {padding-left: 16pt; color: #333846;}
table.toc td.pg {text-align: right; width: 40pt;}
.cover-title {font-family: serif; font-size: 30pt; color: #3d4fc0; line-height: 1.3; margin-top: 40pt; margin-bottom: 8pt;}
.cover-sub {font-size: 13pt; color: #333846; margin-bottom: 4pt;}
.cover-meta {font-size: 9.5pt; color: #5b6070; margin-bottom: 18pt;}
"""


SUB_DIR = DATA / "fonts_subset"
FONTS = ["NotoSansCJKtc-Regular.otf", "NotoSansCJKtc-Bold.otf", "NotoSerifCJKtc-Bold.otf"]


def subset_fonts():
    """MuPDF 的 subset_fonts() 對這套 CFF 字型會出錯，所以排版前先用 fontTools 依實際用到的字裁子集。
    不裁的話三套完整字型會讓 PDF 超過 50 MB。"""
    import re
    from fontTools import subset
    text = COVER + "".join(h for h, _ in CHAPTERS) + "目錄封面0123456789/ "
    text = re.sub(r"<[^>]+>", "", text)
    chars = set(text) | set(chr(c) for c in range(0x20, 0x7F))
    SUB_DIR.mkdir(exist_ok=True)
    opts = subset.Options()
    opts.layout_features = ["*"]
    opts.notdef_outline = True
    opts.retain_gids = True  # 不保留原字形編號的話，MuPDF 會把這套 CID 字型的字對錯（中文全變空白）
    for f in FONTS:
        font = subset.load_font(str(FONT_DIR / f), opts)
        sub = subset.Subsetter(opts)
        sub.populate(text="".join(sorted(chars)))
        sub.subset(font)
        subset.save_font(font, str(SUB_DIR / f), opts)


def archive():
    a = pymupdf.Archive(str(SUB_DIR))
    a.add(pymupdf.Archive(str(FIG_DIR)))
    a.add(pymupdf.Archive(str(DIA_DIR)))
    return a


ARCH = None


def layout(items, writer=None, page0=0):
    """排版 items＝[(html, newpage)]；回傳 (總頁數, 標題清單[(level, text, id, page)])。"""
    heads = []
    page = [page0]
    for html, _ in items:
        story = pymupdf.Story(html=html, user_css=CSS, archive=ARCH)
        more = 1

        def cb(p):
            if p.heading in (1, 2) and (p.open_close & 1):
                heads.append((p.heading, p.text.strip(), p.id, p.page))

        while more:
            if writer is not None:
                dev = writer.begin_page(PAGE)
            more, _ = story.place(BODY)
            story.element_positions(cb, {"page": page[0]})
            # place() 之後一定要 draw() 內容才會往前推進；只量頁碼時畫到 None
            story.draw(dev if writer is not None else None)
            if writer is not None:
                writer.end_page()
            page[0] += 1
    return page[0] - page0, heads


COVER = """
<p class="cover-title">在腦中先想幾步</p>
<p class="cover-sub">Latent MPC 推理：入門導讀、論文精讀與研究計劃</p>
<p class="cover-meta">2026-10-20 ・ 給剛進入這個領域的人 ・ 對應計劃：把 V-JEPA 2-AC 的世界模型＋規劃搬到 LLM 推理</p>
<p class="figp"><img src="fig_story.png" width="487"/></p>
<div class="key">
<p><b>這份文件的三句話</b></p>
<ol>
<li>有人在 LLM 上做過潛空間搜尋，也有人做過推理用的世界模型；但「凍結 LLM＋小世界模型想像＋CEM＋MPC 規劃每一個推理步驟」這整套，這次沒有找到。</li>
<li>類比在一個地方斷掉：LLM「真的寫一步」很便宜，所以只往下想一步的世界模型沒有用。計劃要改成「第一步用真文字、後面才在低維潛動作上想像」。</li>
<li>PoC 有三道停損關（K1、K2b、K3），估計 1–2 GPU 天；這次沒有跑任何實驗。</li>
</ol>
</div>
<p class="cap">本文件的論文圖片均為原論文截圖，僅供學習導讀，請勿上傳到公開的 repo。</p>
"""


def toc_html(heads, offset):
    rows = ""
    for lvl, text, hid, pg in heads:
        cls = "l1" if lvl == 1 else "l2"
        rows += '<tr><td class="' + cls + '">' + text + '</td><td class="pg">' + str(pg + offset + 1) + '</td></tr>'
    return '<h1 id="toc">目錄</h1><table class="toc">' + rows + '</table>'


def split_at(items, names):
    """在指定圖片前把章節拆開，讓那張圖從新的一頁開始（MuPDF 不會把放不下的圖移到下一頁，而是縮小）。"""
    out = []
    for html, newpage in items:
        parts = [html]
        for nm in names:
            marker = '<p class="figp"><img src="' + nm + '"'
            nxt = []
            for part in parts:
                i = part.find(marker)
                if i > 0:
                    nxt += [part[:i], part[i:]]
                else:
                    nxt.append(part)
            parts = nxt
        out += [(parts[0], newpage)] + [(pp, True) for pp in parts[1:]]
    return out


def shrunk_images(pdf_path, items):
    """回傳被縮小（實際寬度比指定寬度小 8% 以上）的圖檔名。"""
    import re
    from PIL import Image
    want = {}
    for html, _ in items:
        for nm, w in re.findall(r'<img src="([^"]+)" width="(\d+)"', html):
            want[nm] = int(w)
    dims = {}
    for nm in want:
        for d in (FIG_DIR, DIA_DIR):
            if (d / nm).exists():
                dims[Image.open(d / nm).size] = nm
    doc = pymupdf.open(pdf_path)
    bad = set()
    for pg in doc:
        for im in pg.get_image_info():
            nm = dims.get((im["width"], im["height"]))
            if nm and nm in want:
                # 版面寬 487pt 的圖實際會畫成 467pt（左右留白），以此換算
                expect = want[nm] * 467 / 487 if want[nm] >= 487 else want[nm]
                if im["bbox"][2] - im["bbox"][0] < expect * 0.92:
                    bad.add(nm)
    return bad


def main(out):
    global ARCH
    subset_fonts()
    ARCH = archive()
    items = list(CHAPTERS)
    forced = set()
    for _ in range(6):
        build(out, items)
        bad = shrunk_images(out, items) - forced
        if not bad:
            break
        print("圖被縮小，改成從新頁開始：", sorted(bad))
        forced |= bad
        items = split_at(list(CHAPTERS), sorted(forced))
    else:
        raise SystemExit("6 輪後仍有圖被縮小")


def build(out, CHAPTERS):
    # 第 0 輪：只排章節，取得每個標題在章節內的相對頁碼
    _, heads = layout(CHAPTERS)
    # 量目錄要幾頁
    toc_pages, _ = layout([(toc_html(heads, 0), True)])
    offset = 1 + toc_pages  # 封面 1 頁＋目錄
    tmp_path = str(Path(out).with_suffix(".tmp.pdf"))
    w = pymupdf.DocumentWriter(tmp_path)
    n_cover, _ = layout([(COVER, True)], writer=w, page0=0)
    assert n_cover == 1, f"封面超過一頁：{n_cover}"
    n_toc, _ = layout([(toc_html(heads, offset), True)], writer=w, page0=1)
    assert n_toc == toc_pages, (n_toc, toc_pages)
    n_body, heads2 = layout(CHAPTERS, writer=w, page0=offset)
    w.close()
    assert [h[3] for h in heads2] == [h[3] + offset for h in heads], "第二輪頁碼和第一輪不一致"

    doc = pymupdf.open(tmp_path)
    for i, pg in enumerate(doc):
        if i == 0:
            continue
        pg.insert_text((54, PAGE.height - 22), "Latent MPC Reasoning - Primer & Plan (2026-10-20)", fontsize=7, color=(0.45, 0.47, 0.53))
        label = f"{i + 1} / {doc.page_count}"
        pg.insert_text((PAGE.width - 54 - pymupdf.get_text_length(label, fontsize=7.5), PAGE.height - 22), label, fontsize=7.5, color=(0.45, 0.47, 0.53))
    toc = [[1, "封面", 1], [1, "目錄", 2]] + [[lvl, text, pg + 1] for lvl, text, _, pg in heads2]
    doc.set_toc(toc)
    doc.set_metadata({"title": "在腦中先想幾步：Latent MPC 推理入門導讀與研究計劃", "author": "Claude Code（為使用者整理）", "subject": "Latent MPC reasoning primer", "creator": "PyMuPDF Story"})
    doc.save(out, garbage=4, deflate=True)
    Path(tmp_path).unlink()
    print("pages", doc.page_count, "toc_pages", toc_pages, "headings", len(heads2))
    for lvl, text, _, pg in heads2:
        if lvl == 1:
            print(f"  p{pg + 1:>3}  {text}")


if __name__ == "__main__":
    main(sys.argv[1])
