#!/usr/bin/env python3
"""m16_story_pdf.py — 「Tiara 研究全貌」：從 8 月的原始想法講到 10/23 的獨立短 PDF（繁體中文）。

和 m16_report_pdf.py（第三階段 90 頁的完整報告）的分工：
  - 這份只講故事：起點 → 第一階段 → 第二階段 → 原設計剩下什麼 → 第三階段 3a／3b → 平台 → 下一步 → 限制。
  - 文字只從 docs/stage3_20261019/story/tiara_story.html 讀（主 session 撰寫，數字已逐條對過出處）；
    <!--FIG:名稱--> 換成圖。新畫的圖（時間軸、I3g 兩個模型）在 build 當下從結果檔讀；
    其他圖直接沿用已經產生好的檔（第一階段 figures/*.pdf、第二階段 report_stage2/figs、第三階段報告的 figs），出處寫在圖說。
  - 字型裁切、HTML 清理、matplotlib 設定、PDF 驗收沿用 m16_report_pdf（不碰 GPU）。

  source code/m7_env.sh
  m7run m16-story-pdf nice -n 19 python code/m16_story_pdf.py build --out /mlsteam/data/tiara/stage3/story
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m16_report_pdf as R  # noqa: E402

REPO = R.REPO
RES = os.path.join(REPO, "results/m16_stage3_mi300x")
NARR = os.path.join(REPO, "docs/stage3_20261019/story/tiara_story.html")
S2FIG = os.path.join(REPO, "docs/precision_20261013/report_stage2/figs")
S3REP = "/mlsteam/data/tiara/stage3/report"
TITLE = "Tiara 研究全貌"
SUB = "從寫入時預測到還原時決定（2026-08-30 → 10-23）"


def s3fig(title: str) -> str:
    """第三階段報告的圖：照標題在 build_log.json 找編號（編號會隨章節變，標題不變）。"""
    b = json.load(open(os.path.join(S3REP, "build_log.json")))
    for f in b["figures"]:
        if f["title"] == title and f["status"] == "ok":
            return os.path.join(S3REP, "figs", f"fig{f['n']:02d}.png")
    raise SystemExit(f"第三階段報告裡找不到圖「{title}」（規則 2：停）")


# ---------------------------------------------------------------------------- 新畫的圖
def fig_timeline(ctx, path):
    plt = R.plt_setup(ctx)
    D = lambda m, d: dt.date(2026, m, d)  # noqa: E731
    COL = {"3090": "#8a94a6", "MI300X": R.C[1], "H200": R.C[2], "模擬": R.C[6]}
    rows = [  # (列名, 起, 迄, 平台, 計畫中？)
        ("原始 Tiara：M0–M5", D(8, 30), D(9, 5), "3090", False),
        ("移植、重量、文獻", D(9, 15), D(9, 30), "MI300X", False),
        ("第一階段：放置", D(10, 8), D(10, 12), "MI300X", False),
        ("第二階段：精度", D(10, 13), D(10, 20), "MI300X", False),
        ("3a 少搬一點（含 I3g）", D(10, 19), D(10, 23), "MI300X", False),
        ("3b 猜何時回來", D(10, 17), D(10, 22), "模擬", False),
        ("H200：第一、二階段復現", D(10, 15), D(10, 19), "H200", False),
        ("H200：ICPE 補量", D(10, 22), D(11, 2), "H200", True),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    t0 = D(8, 28)
    for i, (lab, a, b, plat, plan) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, (b - a).days + 1, left=(a - t0).days, height=0.56, color=COL[plat], alpha=0.35 if plan else 0.9,
                hatch="///" if plan else None, edgecolor=COL[plat], linewidth=0.8)
        ax.text((a - t0).days - 1, y, lab, ha="right", va="center", fontsize=7.6)
    for j, (d, lab) in enumerate(((D(10, 23), "MI300X 到期"), (D(11, 9), "ICPE 摘要"), (D(11, 16), "ICPE 全文"))):
        x = (d - t0).days
        ax.axvline(x, color=R.INK2, lw=0.8, ls="--")
        ax.text(x - 0.6, len(rows) - 0.45 - 0.9 * j, lab, fontsize=7, color=R.INK2, va="top", ha="right",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ticks = [D(9, 1), D(9, 15), D(10, 1), D(10, 15), D(11, 1), D(11, 15)]
    ax.set_xticks([(t - t0).days for t in ticks], [f"{t.month}/{t.day}" for t in ticks])
    ax.set_yticks([])
    ax.set_xlim(-38, (D(11, 18) - t0).days)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    hs = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in COL] + [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=R.INK2, hatch="///")]
    ax.legend(hs, ["RTX 3090", "MI300X", "H200", "模擬（CPU）", "計畫中"], ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=ctx.dpi)
    plt.close(fig)
    return ["docs/RESEARCH_STORY.md、docs/ADVISOR_REPORT_20260930.md、docs/phase1_20261008/13_advisor_update_20261012.md、"
            "docs/precision_20261013/report_stage2/REPORT.md、docs/stage3_20261019/experiment_log.md、origin/h200 的 results/RUNLOG_H200*.md、VENUES.md"]


def fig_i3g(ctx, path):
    plt = R.plt_setup(ctx)
    L = json.load(open(os.path.join(RES, "i3g_agentic/i3g_report.json")))
    Q = json.load(open(os.path.join(RES, "i3g_agentic/qwen/i3g_report.json")))
    P = json.load(open(os.path.join(RES, "i3g_agentic/rep/i3g_rep_report.json")))
    keys = ("a", "b", "c", "d")
    lab = {"a": "a 只有 APC", "b": "b ＋CPU 卸載", "c": "c ＋本地檔案層", "d": "d ＋NFS 檔案層"}
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    w = 0.38
    for j, (name, rep, col) in enumerate((("Llama-3.1-8B", L, R.C[0]), ("Qwen2.5-7B-1M", Q, R.C[1]))):
        xs = [i + (j - 0.5) * w for i in range(4)]
        v = [rep["configs"][k]["p95"] for k in keys]
        ax.bar(xs, v, width=w, color=col, label=f"{name}（判定 {rep['verdict_rank']}、{rep['verdict_flip']}）")
        for k, x, y in zip(keys, xs, v):
            top = max([y] + ([P["runs"][f"{k}_{r}"]["p95"] for r in ("r1", "r2", "r3") if f"{k}_{r}" in P["runs"]] if j == 0 else []))
            ax.text(x, top * 1.14, f"{y:.2f}" if y < 10 else f"{y:.0f}", ha="center", fontsize=7)
    for i, k in enumerate(("b", "c"), start=1):          # Llama 的 b、c 三次（r1＝64，r2／r3＝65）
        pts = [P["runs"][f"{k}_{r}"]["p95"] for r in ("r1", "r2", "r3") if f"{k}_{r}" in P["runs"]]
        ax.scatter([i - 0.5 * w] * len(pts), pts, s=12, color="black", zorder=3, label="Llama b、c 的三次重跑" if k == "b" else None)
    ax.set_yscale("log")
    ax.set_xticks(range(4), [lab[k] for k in keys])
    ax.set_ylabel("回來那一輪 TTFT P95（秒，對數）")
    ax.set_ylim(0.5, 300)
    ax.legend(fontsize=6.8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=ctx.dpi)
    plt.close(fig)
    rids = [c["run_id"] for c in L["configs"].values()] + [c["run_id"] for c in Q["configs"].values()] + [r["run_id"] for r in P["runs"].values()]
    return [f"i3g_agentic/i3g_report.json（{L['run_id']}）、qwen/i3g_report.json（{Q['run_id']}）、rep/i3g_rep_report.json（{P['run_id']}）", rids]


def pdf_to_png(src, dst, dpi=200):
    import pymupdf
    d = pymupdf.open(src)
    d[0].get_pixmap(dpi=dpi).save(dst)
    d.close()


# 名稱 → (產生方式, 寬 pt, 圖說)
FIGS = {
    "timeline": ("new:timeline", 500, "兩個半月的時間軸。顏色是平台；斜線是計畫中（ICPE 補量 11/2 前收齊）。"),
    "deferral": ("pdf:figures/deferral.pdf", 330,
                 "第一階段：每一點是一個寫入時規則。橫軸＝比最好的「不看位置」策略快多少，縱軸＝比自己的延後版快多少；右上角（兩者都 ≥5%）是空的。"
                 "出處：main.tex 圖 deferral（results/paper_20261016/fig_data.csv，code/paper_figs.py）。"),
    "reads": ("png:" + os.path.join(S2FIG, "F04_sources_read.png"), 480,
              "第二階段發現 1：每一條是一次還原的 64 塊。灰＝重算；斜線＝存了低精度但被重算蓋掉、沒讀到；藍＝真的讀到低精度；右邊是讀到÷存的。"
              "慢層上前段（F25）存 16 塊讀到 0 塊；真實 CPU 層讀到 14–16 塊。出處：report_stage2/figs/F04（restores.csv）。"),
    "quality_latency": ("png:" + os.path.join(S2FIG, "F07_quality_latency.png"), 480,
                        "第二階段發現 2：橫軸還原時間（越左越快），縱軸對全 BF16 的答對率差，虛線是 −2 分門檻。全部 INT4（最左）最快；前、中、後、隨機擠在同一條帶裡。出處：report_stage2/figs/F07。"),
    "k2": ("png:" + os.path.join(S2FIG, "F06_step2_K2.png"), 480,
           "第二階段判定 K2：左＝前、中、後對隨機的答對率差（虛線 ±2 分），右＝還原時間；全部壓（E1）最快。出處：report_stage2/figs/F06（analysis/int4/step2_llama_int4.json）。"),
    "ratios": ("s3:各實驗在門檻內能壓到的倍數", 470,
               "3a：各題組在 ε（或 MLPerf 99%）門檻內最省的壓縮倍數。出處：第三階段報告第 5 章（run 20261023-023854-m16-report-pdf）。"),
    "flip": ("s3:X3 翻轉格子：還原時間 vs GPU 忙碌（算力型背景（matmul））", 480,
             "3a：背景搶運算單元時，CPU 層的最快格式隨 GPU 忙碌翻轉（標題有「翻轉」的子圖）。出處：第三階段報告 4.14（x3_time/v3grid/x3grid_report.json）。"),
    "i3g": ("new:i3g", 430, "3b：vLLM 0.28 端到端，agent 工具暫停後回來那一輪的 TTFT P95。黑點是 Llama 的 b、c 三次重跑。"),
    "tucker": ("s3:K1b：同位元組下，xKV、Tucker、兩側分解的重建誤差", 470,
               "7.1：同位元組下的重建誤差。4 維 Tucker（紅）和「全部層一組」（深藍）幾乎重疊，都比每 4 層一組的 xKV（藍）低。出處：第三階段報告 4.1（k1_kvstruct/tucker）。"),
}


CSS = R.CSS_TMPL + """
h1 {font-size: 15pt;}
body {font-size: 9.8pt; line-height: 1.58;}
td, th {font-size: 8.4pt;}
.cover-title {margin: 150pt 0 10pt 0;}
"""


def build(out, dpi=170):
    t0 = time.time()
    ctx = types.SimpleNamespace(fonts_dir=R.DEFAULTS["fonts"], dpi=dpi)
    fdir = os.path.join(out, "figs")
    os.makedirs(fdir, exist_ok=True)
    narr = open(NARR, encoding="utf-8").read()
    narr = re.sub(r"<!--(?!FIG:).*?-->", "", narr, flags=re.S)
    used = re.findall(r"<!--FIG:(\w+)-->", narr)
    log = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               git=R.git_head(), narrative=os.path.relpath(NARR, REPO), figures=[])
    for n, name in enumerate(used, start=1):
        how, width, cap = FIGS[name]
        dst = os.path.join(fdir, f"{name}.png")
        src, rids = how, []
        if how == "new:timeline":
            src = fig_timeline(ctx, dst)[0]
        elif how == "new:i3g":
            src, rids = fig_i3g(ctx, dst)
            cap += "出處：" + src + "。"
        elif how.startswith("pdf:"):
            src = os.path.join(REPO, how[4:])
            pdf_to_png(src, dst)
        elif how.startswith("png:"):
            src = how[4:]
            shutil.copyfile(src, dst)
        elif how.startswith("s3:"):
            src = s3fig(how[3:])
            shutil.copyfile(src, dst)
        log["figures"].append(dict(n=n, name=name, src=src, run_ids=rids))
        html = (f"<div class='fig'><img src='figs/{name}.png' width='{width}'/></div>"
                f"<p class='cap'><b>圖 {n}</b>　{R.esc(cap)}</p>")
        narr = narr.replace(f"<!--FIG:{name}-->", html, 1)
    cover = (f"<p class='cover-title'>{TITLE}</p><p class='cover-sub'>{SUB}</p>"
             f"<p class='cover-meta'>KV cache 階層式放置（Tiara）研究的獨立摘要：第一、二階段、第三階段兩個子問題、三個平台、下一步。<br/>"
             f"細節與每個數字的 run_id：第二階段報告（docs/precision_20261013/report_stage2/REPORT.md）、第三階段完整報告（90 頁）、"
             f"experiment_log.md、origin/h200 的 RUNLOG_H200*.md。<br/>〔實測〕量到的；〔模擬〕用實測常數算的；〔推論〕我們的判讀。self-review，不是 cross-model review。<br/>"
             f"產生：code/m16_story_pdf.py；git {R.git_head()}；{log['ts'][:16]}Z</p>")
    body = cover + narr.replace("<h1 id=\"s0\">", "<h1 id=\"s0\" style=\"page-break-before:always\">")
    reg = os.path.join(ctx.fonts_dir, R.FONT_FILES[0])
    bold = os.path.join(ctx.fonts_dir, R.FONT_FILES[1])
    cmap = R.font_cmap(reg) & R.font_cmap(bold)
    body, repl = R.sanitize_html(body, cmap)
    log["replaced_chars"] = {f"U+{ord(k):04X}": v for k, v in repl.items()}
    os.makedirs(os.path.join(out, "fonts"), exist_ok=True)
    footer = R.FOOTER_TMPL.format(t=TITLE, n=0, tot=0)
    text = R.collect_chars(body, footer)
    log["fonts"] = [R.subset_font(reg, os.path.join(out, "fonts", "tc_regular.ttf"), text),
                    R.subset_font(bold, os.path.join(out, "fonts", "tc_bold.ttf"), text)]
    open(os.path.join(out, "tiara_story.html"), "w", encoding="utf-8").write(
        "<!doctype html><html><head><meta charset='utf-8'><title>" + TITLE + "</title><style>" + CSS + "</style></head><body>" + body + "</body></html>")
    pdf = os.path.join(out, "tiara_story.pdf")
    log["render"] = render(out, body, pdf, cmap)
    log["verify"] = R.verify_pdf(pdf)
    log["seconds"] = round(time.time() - t0, 1)
    json.dump(log, open(os.path.join(out, "build_log.json"), "w"), indent=1, ensure_ascii=False)
    return log


def render(out, body, pdf, cmap):
    import pymupdf
    arch = pymupdf.Archive(out)
    MB = pymupdf.paper_rect("a4")
    BODY = MB + (54, 50, -54, -58)
    heads = []

    def positionfn(p):
        if getattr(p, "id", None) and p.heading and getattr(p, "open_close", 1) & 1:
            heads.append((p.heading, p.text, p.page_num))

    want = [int(w) for w in re.findall(r"<img src='figs/\w+\.png' width='(\d+)'/>", body)]
    names = re.findall(r"<div class='fig'><img src='figs/(\w+)\.png'", body)
    breaks = []
    for _ in range(len(names) + 1):              # 圖被擠到頁底縮小 → 在第一張被縮的圖前面換頁，重排（同 m16_report_pdf）
        heads.clear()
        b = body
        for nm in breaks:
            b = b.replace(f"<div class='fig'><img src='figs/{nm}.png'", f"<div class='fig' style='page-break-before:always'><img src='figs/{nm}.png'", 1)
        doc = pymupdf.Story.write_stabilized_with_links(lambda pos: b, lambda n, f: (MB, BODY, None), user_css=CSS, em=10,
                                                        positionfn=positionfn, archive=arch, add_header_ids=False)
        shown = [pymupdf.Rect(im["bbox"]).width for pg in doc for im in pg.get_image_info()]
        shr = [nm for nm, w, d in zip(names, want, shown) if d < 0.95 * min(w, BODY.width) and nm not in breaks] if len(shown) == len(names) else []
        if not shr:
            break
        breaks.append(shr[0])
        doc.close()
    total = doc.page_count
    for i, page in enumerate(doc):
        ft, _ = R.sanitize_html(R.FOOTER_TMPL.format(t=TITLE, n=i + 1, tot=total), cmap)
        page.insert_htmlbox(pymupdf.Rect(54, MB.y1 - 46, MB.x1 - 54, MB.y1 - 18), ft, css=CSS, archive=arch)
    toc, seen = [], set()
    for lvl, txt, pg in heads:
        if lvl <= 2 and (lvl, txt) not in seen:
            seen.add((lvl, txt))
            toc.append([lvl, txt.strip(), pg])
    fixed, prev = [], 0
    for t in toc:
        t = [min(t[0], prev + 1), t[1], t[2]]
        fixed.append(t)
        prev = t[0]
    if fixed:
        doc.set_toc(fixed)
    doc.set_metadata(dict(title=TITLE, subject=SUB, author="Tiara（自動產生：code/m16_story_pdf.py）", creator="m16_story_pdf.py（PyMuPDF Story）"))
    tmp = pdf + ".tmp"
    doc.save(tmp, garbage=4, deflate=True)
    doc.close()
    os.replace(tmp, pdf)
    return dict(pages=total, toc=len(fixed), page_breaks_before=breaks, still_shrunk=shr)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--out", default="/mlsteam/data/tiara/stage3/story")
    b.add_argument("--dpi", type=int, default=170)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = build(a.out, a.dpi)
    v = log["verify"]
    print(f"PDF：{os.path.join(a.out, 'tiara_story.pdf')}（{v['pages']} 頁、{v['bytes'] / 1e6:.2f} MB、書籤 {v['toc_entries']}）；"
          f"不該出現的字型：{v['bad_fonts'] or '無'}；頁碼缺：{v['footer_missing'] or '無'}；換掉的字元：{log['replaced_chars'] or '無'}")
    if not v["ok"]:
        print("驗收沒過 → exit 1", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
