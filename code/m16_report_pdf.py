#!/usr/bin/env python3
"""m16_report_pdf.py — 第三階段前置探索的「完整 PDF 報告」產生器（繁體中文、以圖表為主）。

設計原則（CLAUDE.md 不可協商規則）：
  - 圖表與表格的每一個數字都在 build 當下從結果檔即時讀（results/m16_stage3_mi300x/ 與 BIG_FILES.md 指到的 run 目錄），
    程式裡不寫死任何結果數字；結果檔不存在 → 畫「NOT_MEASURED」佔位並在 build log 印警告。
  - 「解讀、結論、建議」的文字只從 docs/stage3_20261019/report/narrative/<節名>.html 讀（主 session 撰寫）；
    檔案不存在 → 放明顯的「（待撰寫）」框。<節名>.draft.html 是只列事實的草稿，報告不採用。
  - 不碰 GPU：不准 import torch／vllm／transformers／triton（模組載入時裝 import 阻擋器）。只用 matplotlib、PyMuPDF、fontTools。

子命令：
  build       產生全部圖表 → 組 HTML → 字型子集（CFF 轉 TrueType）→ PyMuPDF Story 排版 → PDF
              輸出：<out>/stage3_report.pdf（預設 /mlsteam/data/tiara/stage3/report/）、<out>/stage3_report.html、
                    <out>/figs/*.png、<out>/fonts/*.ttf、<out>/build_log.json、<out>/preview/*.png
  --self-test CPU 自測（合成資料）：字元收集、字型裁切與轉檔、Fallback 檢查（含反向對照）、NOT_MEASURED 佔位、
              narrative 待撰寫框、runner log 解析、失敗清單解析。

  source code/m7_env.sh
  OMP_NUM_THREADS=4 nice -n 19 taskset -c 0-3 python code/m16_report_pdf.py build
  OMP_NUM_THREADS=4 nice -n 19 taskset -c 0-3 python code/m16_report_pdf.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import hashlib
import html as _html
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter, OrderedDict, defaultdict


# ============================================================================ 不碰 GPU：import 阻擋器
class _NoGPUImports:
    BLOCK = ("torch", "vllm", "transformers", "triton")

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.BLOCK:
            raise ImportError(f"m16_report_pdf 不准 import {name}（報告產生器不碰 GPU）")
        return None


sys.meta_path.insert(0, _NoGPUImports())
for _m in _NoGPUImports.BLOCK:
    if _m in sys.modules:
        raise SystemExit(f"{_m} 已經被 import（報告產生器不准開 GPU 行程）")

os.environ.setdefault("MPLBACKEND", "Agg")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULTS = dict(
    res=os.path.join(REPO, "results/m16_stage3_mi300x"),
    docs=os.path.join(REPO, "docs/stage3_20261019"),
    out="/mlsteam/data/tiara/stage3/report",
    fonts="/mlsteam/data/tiara/fonts",
    jobs="/mlsteam/data/tiara/stage3/jobs",
)
FONT_FILES = ("NotoSansCJKtc-Regular.otf", "NotoSansCJKtc-Bold.otf")
FONT_FAMILY_PDF = "Noto Sans CJK TC"          # 驗收：PDF 裡只能有這個 family（Regular／Bold）
MODELS = ("llama", "qwen")
MODEL_LABEL = {"llama": "Llama-3.1-8B", "qwen": "Qwen2.5-7B-1M"}
TITLE = "Tiara 第三階段前置探索報告"
# 圖表顏色（dataviz 參考調色盤，淺色底；固定順序、不循環）
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#1d2433", "#52514e", "#d9dde3"
MODEL_COLOR = {"llama": C[0], "qwen": C[1]}
# 字型裡沒有的字元 → 換成意思最接近、字型裡有的（其餘換成 ? 並記警告）
CHAR_SUBST = {"\u2717": "×", "\u2718": "×", "\u200b": "", "\u2009": " ", "\u202f": " ", "\u25b8": "▶", "\u25ba": "▶",
              "\u207a": "+", "\u207b": "-", "\ufeff": "", "\ufe0f": "", "\ufe0e": ""}


class NotMeasured(Exception):
    """結果檔不存在、或還沒有該實驗的資料 → 佔位（NOT_MEASURED），不是錯誤。"""


# ============================================================================ 小工具
def esc(s) -> str:
    return _html.escape("" if s is None else str(s), quote=True)


def rel(p: str) -> str:
    p = os.path.abspath(p)
    return os.path.relpath(p, REPO) if p.startswith(REPO + os.sep) else p


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def minus(s: str) -> str:
    return s.replace("-", "−")


def pm(x, nd=1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return minus(f"{x:+.{nd}f}")


def fx(x, nd=3) -> str:
    if x is None or x == "":
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return esc(x)
    if math.isinf(v):
        return "∞"
    return minus(f"{v:.{nd}f}")


def fpct(x, nd=1) -> str:
    return "—" if x is None else minus(f"{100 * float(x):.{nd}f}%")


def ci_pp(d, nd=1, scale=100.0) -> str:
    """{mean, lo, hi} → '+0.3 [−3.2, +4.0]'（乘 scale；預設百分點）。"""
    if not d or d.get("mean") is None:
        return "—"
    return f"{pm(scale * d['mean'], nd)} [{pm(scale * d['lo'], nd)}, {pm(scale * d['hi'], nd)}]"


def need(path: str) -> str:
    if not path or not os.path.exists(path):
        raise NotMeasured(f"{rel(path)} 不存在")
    return path


def read_csv(path: str) -> list:
    need(path)
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: str):
    need(path)
    with open(path) as f:
        return json.load(f)


def ffloat(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def split_rids(s) -> list:
    return [x for x in re.split(r"[;,\s]+", str(s or "")) if re.match(r"^\d{8}-\d{6}-", x)]


def rids_of_rows(rows, col="run_id") -> list:
    out = set()
    for r in rows:
        out.update(split_rids(r.get(col)))
    return sorted(out)


def big_dir(bigfiles_md: str, tail: str) -> str:
    """BIG_FILES.md 寫的 run 目錄（>1 MB 的檔在那裡）。tail 例：'k1_full'。"""
    t = open(need(bigfiles_md)).read()
    m = re.search(r"`(/mlsteam/data/tiara/runs/[^`]*?/" + re.escape(tail) + r")/?`", t)
    if not m:
        raise NotMeasured(f"{rel(bigfiles_md)} 裡找不到 {tail} 的位置")
    return m.group(1)


def sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def family_of(spec: str) -> str:
    """組合語法 → 家族（只看語法，不看結果）。"""
    s = spec or ""
    if s in ("none", "E0", ""):
        return "對照"
    if "drop:RND0" in s:
        return "隨機丟（對照）"
    has = lambda k: (k + ":") in s or s.startswith(k)   # noqa: E731
    if has("xl") and "q:" in s and "drop" not in s:
        return "xKV＋量化"
    if has("xl") and "drop" in s:
        return "xKV＋丟"
    if has("xl"):
        return "xKV（跨層低秩）"
    if has("lr"):
        return "每層低秩（＋量化）" if "q:" in s else "每層低秩"
    if "merge" in s:
        return "合併 token"
    if "drop" in s:
        return "丟 block（＋量化）" if "q:" in s else "丟 block"
    if s in ("mc", "share"):
        return "跨層逐 token 共用"
    if s.startswith("q:"):
        return "量化"
    return "其他"


FAMILY_STYLE = OrderedDict([
    ("量化", (C[0], "o")), ("xKV（跨層低秩）", (C[2], "s")), ("xKV＋量化", (C[7], "D")), ("xKV＋丟", (C[6], "P")),
    ("每層低秩", (C[3], "v")), ("每層低秩（＋量化）", (C[3], "^")), ("丟 block", (C[1], "<")), ("丟 block（＋量化）", (C[1], ">")),
    ("合併 token", (C[4], "h")), ("跨層逐 token 共用", (C[5], "*")), ("隨機丟（對照）", ("#9aa0a6", "x")), ("對照", ("#555555", "X")),
    ("其他", ("#777777", ".")),
])


# ============================================================================ 報告的狀態（警告、出處、NOT_MEASURED）
class Ctx:
    def __init__(self, res, docs, out, fonts, jobs, narr=None, dpi=170):
        self.res, self.docs, self.out, self.fonts_dir, self.jobs = res, docs, out, fonts, jobs
        self.narr_dir = narr or os.path.join(docs, "report", "narrative")
        self.figdir = os.path.join(out, "figs")
        os.makedirs(self.figdir, exist_ok=True)
        self.dpi = dpi
        self.warnings, self.nm, self.figs, self.tables = [], [], [], []
        self.sources = defaultdict(list)          # 節 → [(檔, [run_id])]
        self.narr_status = OrderedDict()          # 節名 → ok／missing／draft_only
        self.headings = []                        # (level, id, text)
        self.fig_n = 0
        self.tab_n = 0
        self.fig_w = {}                           # 圖的 div id → 要求的寬（pt）；排版時被縮小就在它前面換頁
        self.now = now_utc()

    def R(self, *p) -> str:
        return os.path.join(self.res, *p)

    def D(self, *p) -> str:
        return os.path.join(self.docs, *p)

    def warn(self, sec, msg):
        self.warnings.append(dict(section=sec, msg=msg))
        print(f"[警告] {sec}：{msg}", file=sys.stderr, flush=True)

    def not_measured(self, sec, item, reason):
        self.nm.append(dict(section=sec, item=item, reason=reason))
        print(f"[NOT_MEASURED] {sec}：{item}（{reason}）", file=sys.stderr, flush=True)

    def src(self, sec, path, rids=None):
        if path and os.path.exists(path):
            rr = sorted(set(rids or []))
            for e in self.sources[sec]:
                if e[0] == path:
                    e[1][:] = sorted(set(e[1]) | set(rr))
                    return
            self.sources[sec].append((path, rr))


# ============================================================================ matplotlib
_PLT = None


def plt_setup(ctx: Ctx):
    global _PLT
    if _PLT is not None:
        return _PLT
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in FONT_FILES:
        p = os.path.join(ctx.fonts_dir, f)
        if os.path.exists(p):
            font_manager.fontManager.addfont(p)
        else:
            raise SystemExit(f"找不到字型 {p}（圖裡的中文會變方塊；規則 2：停）")
    plt.rcParams.update({
        "font.family": "Noto Sans CJK TC", "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.2, "axes.unicode_minus": True,
        "axes.edgecolor": "#9aa0a6", "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6, "savefig.facecolor": "white",
        "figure.facecolor": "white", "legend.frameon": False,
    })
    _PLT = plt
    return plt


def placeholder_png(ctx, path, title, reason, status="NOT_MEASURED"):
    plt = plt_setup(ctx)
    fig, ax = plt.subplots(figsize=(7.0, 1.9))
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0.01, 0.04), 0.98, 0.92, transform=ax.transAxes, fill=True, facecolor="#f2f4f7",
                               edgecolor="#8a94a6", linewidth=1.2, linestyle="--"))
    ax.text(0.5, 0.66, status, ha="center", va="center", fontsize=17, color="#a5281b", fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.5, 0.40, title, ha="center", va="center", fontsize=9.5, color=INK, transform=ax.transAxes)
    r = reason if len(reason) < 110 else reason[:107] + "…"
    ax.text(0.5, 0.18, r, ha="center", va="center", fontsize=7.5, color=INK2, transform=ax.transAxes)
    fig.savefig(path, dpi=ctx.dpi)
    plt.close(fig)


def add_fig(ctx: Ctx, sec: str, title: str, fn, width=490, note="") -> str:
    """fn() → (matplotlib Figure, 說明文字)。NotMeasured → 佔位；其他例外 → 佔位＋警告（讀不懂不等於沒有）。"""
    plt = plt_setup(ctx)
    ctx.fig_n += 1
    n = ctx.fig_n
    key = f"fig{n:02d}.png"
    path = os.path.join(ctx.figdir, key)
    try:
        fig, cap = fn()
        fig.savefig(path, dpi=ctx.dpi)
        plt.close(fig)
        st = "ok"
    except NotMeasured as e:
        plt.close("all")
        placeholder_png(ctx, path, title, str(e))
        st, cap = "NOT_MEASURED", f"<span class='nm'>NOT_MEASURED</span>：{esc(e)}"
        ctx.not_measured(sec, f"圖 {n}：{title}", str(e))
    except Exception as e:                       # noqa: BLE001
        plt.close("all")
        placeholder_png(ctx, path, title, f"{type(e).__name__}: {e}", status="NOT_MEASURED（讀取失敗）")
        st, cap = "ERROR", f"<span class='nm'>NOT_MEASURED（讀取失敗）</span>：{esc(type(e).__name__)}: {esc(e)}"
        ctx.warn(sec, f"圖 {n}「{title}」讀取失敗：{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
    ctx.figs.append(dict(n=n, section=sec, title=title, status=st, png=path))
    w = width
    try:
        from PIL import Image
        with Image.open(path) as im:
            iw, ih = im.size
        if ih / iw * w > 690:                    # 不超過一頁的可用高度
            w = int(690 * iw / ih)
    except Exception:                           # noqa: BLE001
        pass
    ctx.fig_w[f"fig{n:02d}"] = w
    return (f"<div class='fig' id='fig{n:02d}'><img src='figs/{key}' width='{w}'/>"
            f"<p class='cap'><b>圖 {n}</b>　{esc(title)}。{cap}{(' ' + note) if note else ''}</p></div>")


def html_table(headers, rows, widths=None, cls="") -> str:
    def _w(i):
        return f" style='width:{widths[i]}'" if widths and widths[i] else ""
    th = "".join(f"<th{_w(i)}>{h}</th>" for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table class='{cls}'><tr>{th}</tr>{body}</table>"


def add_table(ctx: Ctx, sec: str, title: str, fn) -> str:
    """fn() → (headers, rows, 說明)；cell 是已經 escape 的 HTML。"""
    ctx.tab_n += 1
    n = ctx.tab_n
    try:
        headers, rows, note = fn()
        if not rows:
            raise NotMeasured("結果檔裡沒有這張表要的列")
        st = "ok"
        body = html_table(headers, rows)
        cap = f"<p class='tcap'><b>表 {n}</b>　{esc(title)}。{note or ''}</p>"
        out = cap + body
    except NotMeasured as e:
        st = "NOT_MEASURED"
        ctx.not_measured(sec, f"表 {n}：{title}", str(e))
        out = (f"<p class='tcap'><b>表 {n}</b>　{esc(title)}。</p>"
               f"<div class='box-nm'><b>NOT_MEASURED</b>：{esc(e)}</div>")
    except Exception as e:                       # noqa: BLE001
        st = "ERROR"
        ctx.warn(sec, f"表 {n}「{title}」讀取失敗：{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
        out = (f"<p class='tcap'><b>表 {n}</b>　{esc(title)}。</p>"
               f"<div class='box-nm'><b>NOT_MEASURED（讀取失敗）</b>：{esc(type(e).__name__)}: {esc(e)}</div>")
    ctx.tables.append(dict(n=n, section=sec, title=title, status=st))
    return out


# ============================================================================ narrative、markdown
def narrative(ctx: Ctx, name: str, what: str) -> str:
    p = os.path.join(ctx.narr_dir, f"{name}.html")
    dp = os.path.join(ctx.narr_dir, f"{name}.draft.html")
    if os.path.exists(p):
        t = open(p, encoding="utf-8").read()
        t = re.sub(r"(?is)<(script|style)\b.*?</\1>", "", t)
        t = re.sub(r"(?is)</?(html|head|body|meta|link|title)\b[^>]*>", "", t)
        t = re.sub(r"(?i)font-family\s*:[^;\"']*;?", "", t)       # 字型一律用報告的（否則會出現 fallback 字型）
        ctx.narr_status[name] = "ok"
        return f"<div class='narr'>{t}</div>"
    ctx.narr_status[name] = "draft_only" if os.path.exists(dp) else "missing"
    hint = f"；事實草稿在 <b>{esc(name)}.draft.html</b>（只列事實，報告不採用）" if os.path.exists(dp) else ""
    return (f"<div class='box-todo'><b>（待撰寫）</b>{esc(what)}——由主 session 撰寫 "
            f"<b>{esc(rel(p))}</b>{hint}。</div>")


_MD = None


def md(text: str) -> str:
    global _MD
    if _MD is None:
        from markdown_it import MarkdownIt
        _MD = MarkdownIt("commonmark", {"html": False}).enable("table")
    return _MD.render(text)


def md_section(path: str, start_pat: str, end_pat: str = r"^## ", max_chars=None) -> str:
    """從 markdown 檔抽一段（start_pat 那一行之後到下一個 end_pat 之前）。"""
    lines = open(need(path), encoding="utf-8").read().splitlines()
    out, on = [], False
    for ln in lines:
        if not on and re.match(start_pat, ln):
            on = True
            continue
        if on and re.match(end_pat, ln):
            break
        if on:
            out.append(ln)
    if not out:
        raise NotMeasured(f"{rel(path)} 裡找不到「{start_pat}」那一段")
    t = "\n".join(out).strip()
    if max_chars and len(t) > max_chars:
        acc, n = [], 0
        for ln in t.splitlines():
            top = bool(re.match(r"^(\d+\.|[-*+])\s", ln)) or not ln.strip() or ln.startswith(("#", ">"))
            if acc and n > max_chars and top:      # 只在頂層段落／清單項的邊界截斷（不切斷表格或子清單）
                break
            acc.append(ln)
            n += len(ln)
        t = "\n".join(acc).rstrip() + "\n\n（以下略；全文見原檔）"
    return t


# ============================================================================ 實驗登記（狀態一覽用；只列檔名與 job 名，沒有結果數字）
EXPS = [
    dict(key="k1", sec="4.1", title="K1　KV 結構（低秩／跨層／相鄰 token）", files=["k1_kvstruct/summary.json", "k1_kvstruct/xlayer.csv", "k1_kvstruct/tucker/summary.json"],
         jobs=["06_k1"], kind="實測（CPU，KV 幾何量）"),
    dict(key="i4", sec="4.2", title="I4　KV 從 CPU 搬回 GPU 的搬法", files=["i4_copy_batching/copybench_sdma0.csv", "i4_copy_batching/copybench_sdma1.csv"],
         jobs=["20_i4", "21_i4"], kind="實測（量時間）"),
    dict(key="i1", sec="4.3", title="I1／I2　多 session 同時還原與互相干擾",
         files=["i1_concurrent_restore/prediction/prediction_llama.json", "i1_concurrent_restore/interf_llama.json", "i1_concurrent_restore/grid_llama.csv"],
         jobs=["30_i1", "31_i1"], kind="模擬＋實測驗證"),
    dict(key="c1a", sec="4.4", title="C1a　分數能不能猜中第 2 輪要的 block", files=["c1a_predictive/c1a_summary.json"],
         jobs=["33a", "33b", "33z"], kind="實測（品質，只 forward）"),
    dict(key="s1", sec="4.5", title="S1　無損壓縮比與 CPU 解壓", files=["s1_lossless/summary.json", "s1_lossless/breakeven.csv"],
         jobs=["05_s1", "35_s1"], kind="實測＋盈虧計算"),
    dict(key="x1", sec="4.6", title="X1　壓縮疊加的 teacher-forcing 篩選", files=["x_stack/x1_report.json", "x_stack/configs_llama.csv", "x_stack/configs_qwen.csv"],
         jobs=["36"], kind="實測（TF 篩選）"),
    dict(key="x2", sec="4.7", title="X2＋X2b　生成答對率（32K、128K）", files=["x_stack/x2_report.json", "x_stack/v2/x2b_report.json"],
         jobs=["37", "38b"], kind="實測（生成答對率）"),
    dict(key="xlcal", sec="4.8", title="xKV 事先校準的基底 vs 每題 SVD", files=["x_stack/v2/xlcal_report.json", "x_stack/v3/xlcal3_report.json"],
         jobs=["38c", "38k"], kind="實測（生成答對率）"),
    dict(key="x3", sec="4.9", title="X3　還原時間 × 連線 × GPU 忙（含寫入成本、validate）",
         files=["x3_time/map_summary.json", "x3_time/v2/map_summary.json", "x3_time/v2/write_cost.csv", "x3_time/v2/validate/validate_summary.json"],
         jobs=["39_x3", "39z", "38a", "38z", "38j", "39v"], kind="零件實測＋模擬；validate 實測"),
    dict(key="c1b", sec="4.10", title="C1b　選擇性還原", files=["c1_selective_restore/c1b_summary.json", "c1_selective_restore/c1b_cells.csv"],
         jobs=["38da", "38db", "38dz"], kind="實測（品質）＋還原時間計算"),
    dict(key="x4", sec="4.11", title="X4　多層低秩＋量化推到極限", files=["x_stack/x4/x4_report.json", "x_stack/x4/x4_best_allocation.csv", "x_stack/x4/lossless_x4.json"],
         jobs=["38e", "38f", "38g", "38h", "38i"], kind="實測（TF＋生成）"),
    dict(key="i3", sec="4.12", title="I3　agent 工具暫停時 KV 的去留（模擬）", files=["i3_pause_sim/summary.json", "i3_pause_sim/reread_baselines.json"],
         jobs=[], kind="模擬（常數實測）"),
    dict(key="vllm", sec="4.13", title="V0／F2／F1　vLLM 0.28 的復用與卸載",
         files=["v0_vllm_smoke/v0check_*.json", "f2_shared_prefix_burst/summary_*.json", "f1_prefix_reuse/summary_*.json"],
         jobs=["40_v0", "50_f2", "51_f1", "52_f1"], kind="實測（vLLM 端到端）"),
]


def exp_status(ctx: Ctx, e: dict):
    have = sum(1 for f in e["files"] if glob.glob(ctx.R(f)))
    st = "完成" if have == len(e["files"]) else ("部分" if have else "NOT_MEASURED")
    return st, have, len(e["files"])


def queue_state(ctx: Ctx, prefixes) -> str:
    if not prefixes:
        return "—（不用 GPU 佇列）"
    cnt = Counter()
    for q in ("queue", "queue_c"):
        for p in glob.glob(os.path.join(ctx.jobs, q, "*")):
            b = os.path.basename(p)
            m = re.match(r"^(.+?)\.(sh|running|done|failed|cutoff)$", b)
            if m and any(m.group(1).startswith(x) for x in prefixes):
                cnt[m.group(2)] += 1
    if not cnt:
        return "（佇列裡沒有）"
    name = {"done": "完成", "running": "跑中", "sh": "排隊", "failed": "失敗", "cutoff": "截止跳過"}
    return "、".join(f"{name[k]} {cnt[k]}" for k in ("done", "running", "sh", "failed", "cutoff") if cnt[k])


# ============================================================================ runner log（甘特圖）與 experiment_log（失敗清單）
LOG_RX = re.compile(r"^\[runner:s3:(\w+)\] (START|DONE|FAILED|SKIP|DEFER) (\S+?)(?:：|\s|$)(.*?)\s*(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)\s*$")


def parse_runner_logs(paths, now=None) -> list:
    now = now or now_utc()
    ev = []
    for p in paths:
        if not os.path.exists(p):
            continue
        for ln in open(p, encoding="utf-8", errors="replace"):
            m = LOG_RX.match(ln.strip())
            if m:
                role, act, name, rest, ts = m.groups()
                t = dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
                ev.append((t, role, act, name, rest))
    ev.sort(key=lambda x: x[0])
    open_, out = {}, []
    for t, role, act, name, rest in ev:
        if act == "START":
            nm = re.search(r"need=(\w+)", rest)
            need_ = nm.group(1) if nm else "?"
            two = re.search(r"cards=0,1", rest) is not None
            kind = "CPU 分析" if role == "cpu" else ("量品質（兩卡並行）" if (need_ == "one" and two) else "量時間（獨佔兩卡）")
            open_[name] = dict(name=name, role=role, kind=kind, start=t, end=None, status="跑中")
        elif act in ("DONE", "FAILED") and name in open_:
            j = open_.pop(name)
            j.update(end=t, status="完成" if act == "DONE" else "失敗")
            out.append(j)
    for j in open_.values():
        j["end_shown"] = now
        out.append(j)
    for j in out:
        j.setdefault("end_shown", j["end"])
    out.sort(key=lambda j: j["start"])
    return out


def parse_failures(text: str) -> list:
    """experiment_log.md 裡的「**失敗 N**」條目（含後面接著的修法行，到下一個條目為止）。"""
    out, cur = [], None
    for ln in text.splitlines():
        m = re.match(r"^\s*-\s+\*\*失敗\s*(\d+)\*\*(.*)$", ln)
        if m:
            cur = dict(n=int(m.group(1)), text=m.group(2).strip(" ：:"), more=[])
            out.append(cur)
            continue
        if cur is not None:
            if re.match(r"^\s{2,}\S", ln):
                cur["more"].append(ln.strip())
            else:
                cur = None
    return out


# ============================================================================ CSS 與頁面骨架
CSS_TMPL = """
@font-face {font-family: TC; src: url(fonts/tc_regular.ttf);}
@font-face {font-family: TC; font-weight: bold; src: url(fonts/tc_bold.ttf);}
html, body, p, div, span, h1, h2, h3, h4, h5, h6, table, tr, td, th, ul, ol, li, a, b, strong, i, em,
code, pre, tt, kbd, samp, sup, sub, small, blockquote, dl, dt, dd {font-family: TC;}
body {font-size: 9.4pt; line-height: 1.5; color: #1d2433;}
h1 {font-size: 16.5pt; color: #12355b; page-break-before: always; margin: 0 0 8pt 0; padding-bottom: 3pt;
    border-bottom: 1.2pt solid #12355b;}
h2 {font-size: 12.5pt; color: #12355b; margin: 14pt 0 5pt 0;}
h3 {font-size: 10.3pt; color: #2b4a6f; margin: 9pt 0 3pt 0;}
h4 {font-size: 9.6pt; color: #2b4a6f; margin: 6pt 0 2pt 0;}
p {margin: 2pt 0 5pt 0;}
ul, ol {margin: 2pt 0 5pt 14pt;}
li {margin: 0 0 1.5pt 0;}
table {border-collapse: collapse; margin: 3pt 0 9pt 0; width: 100%;}
td, th {border: 0.6pt solid #9aa7b8; padding: 1.8pt 3pt; font-size: 7.8pt; vertical-align: top; line-height: 1.32;}
th {font-weight: bold; color: #12355b; border-bottom: 1.2pt solid #12355b;}
a {color: #1a5fb4; text-decoration: none;}
code {color: #7a3e00;}
pre {white-space: pre-wrap; font-size: 7.2pt; border: 0.5pt solid #c9d1db; padding: 4pt; line-height: 1.3;}
blockquote {margin: 3pt 0 5pt 8pt; color: #4a5568;}
.fig {text-align: center; margin: 5pt 0 2pt 0;}
.cap {font-size: 8.1pt; color: #3a4558; text-align: left; margin: 2pt 0 9pt 0; line-height: 1.38;}
.tcap {font-size: 8.1pt; color: #3a4558; margin: 6pt 0 1pt 0;}
.box-todo {border: 1.5pt dashed #c0392b; color: #8f2318; padding: 6pt; margin: 6pt 0;}
.box-nm {border: 1pt solid #8a94a6; color: #4a5568; padding: 5pt; margin: 4pt 0 8pt 0;}
.box-rule {border: 0.8pt solid #9fb3cc; border-left: 3pt solid #12355b; padding: 5pt 7pt; margin: 4pt 0 7pt 0;}
.narr {border-left: 2.5pt solid #2e8b57; padding: 1pt 0 1pt 7pt; margin: 4pt 0 6pt 0;}
.meta {font-size: 8pt; color: #4a5568;}
.small {font-size: 7.8pt;}
.ok {color: #1e7b34; font-weight: bold;}
.nm {color: #a5281b; font-weight: bold;}
.part {color: #9a6700; font-weight: bold;}
.cover-title {font-size: 26pt; font-weight: bold; color: #12355b; margin: 120pt 0 10pt 0;}
.cover-sub {font-size: 13pt; color: #2b4a6f; margin: 0 0 40pt 0;}
.cover-meta {font-size: 10pt; color: #1d2433; line-height: 1.8;}
.toc td {border: none; border-bottom: 0.4pt dotted #b0b8c4; font-size: 8.4pt; padding: 0.4pt 2pt; line-height: 1.22;}
.toc .l2 {padding-left: 14pt; font-size: 7.9pt; color: #374151;}
"""

FOOTER_TMPL = "<p style='text-align:center;font-size:7.6pt;color:#6b7280'>{t}　·　第 {n}／{tot} 頁</p>"


class Doc:
    def __init__(self, ctx: Ctx):
        self.ctx, self.parts = ctx, []

    def raw(self, s):
        self.parts.append(s)

    def h(self, level, hid, text):
        self.ctx.headings.append((level, hid, text))
        self.parts.append(f"<h{level} id='{hid}'>{esc(text)}</h{level}>")

    def h3(self, text):
        self.parts.append(f"<h3>{esc(text)}</h3>")

    def p(self, s):
        self.parts.append(f"<p>{s}</p>")

    def html(self) -> str:
        return "\n".join(self.parts)


def provenance(ctx: Ctx, sec: str) -> str:
    rows = []
    for path, rids in ctx.sources.get(sec, []):
        if len(rids) > 6:
            rs = ", ".join(rids[:5]) + f", …（共 {len(rids)} 個）"
        else:
            rs = ", ".join(rids) if rids else "（檔內沒有 run_id 欄；見說明）"
        rows.append([esc(rel(path)), esc(rs)])
    if not rows:
        return "<div class='box-nm'>這一節還沒有任何結果檔（NOT_MEASURED）。</div>"
    return html_table(["檔案（相對 repo；/mlsteam 開頭的是 run 目錄的大檔）", "run_id"], rows)


def exp_head(d: Doc, ctx: Ctx, e: dict, spec: dict):
    """每個實驗節的固定骨架（前半）：狀態 → 要回答的問題 → 作法 → 「主要圖表」標題。"""
    sec = e["sec"]
    st, have, tot = exp_status(ctx, e)
    cls = {"完成": "ok", "部分": "part", "NOT_MEASURED": "nm"}[st]
    d.h(2, f"exp_{e['key']}", f"{sec}　{e['title']}")
    d.raw(f"<p class='meta'>狀態：<span class='{cls}'>{st}</span>（主要結果檔 {have}／{tot}）　·　佇列：{esc(queue_state(ctx, e['jobs']))}"
          f"　·　性質：{esc(e['kind'])}</p>")
    d.h3("要回答的問題")
    d.p(spec["q"])
    d.h3("作法")
    d.p(spec["method"])
    d.h3("主要圖表")


def exp_tail(d: Doc, ctx: Ctx, e: dict, spec: dict, rule_fn=None):
    """後半：事先寫死的判定與結果 → 解讀（narrative）→ 限制 → run_id 與檔案出處。"""
    sec = e["sec"]
    d.h3("事先寫死的判定與結果")
    d.raw(f"<div class='box-rule'>{spec['rule']}</div>")
    if rule_fn:
        rule_fn()
    d.h3("解讀（主 session 撰寫）")
    d.raw(narrative(ctx, e["key"], f"{e['title']} 的解讀與結論"))
    d.h3("限制")
    d.raw("<ul>" + "".join(f"<li>{x}</li>" for x in spec["limits"]) + "</ul>")
    d.h3("run_id 與檔案出處")
    d.raw(provenance(ctx, sec))


# ============================================================================ 第 0 章：封面、一頁摘要、目錄
def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=20).stdout.strip() or "NOT_AVAILABLE"
    except Exception:                           # noqa: BLE001
        return "NOT_AVAILABLE"


def ch_cover(d: Doc, ctx: Ctx):
    tw = ctx.now + dt.timedelta(hours=8)
    d.raw(f"<div class='cover-title'>{TITLE}</div>"
          "<div class='cover-sub'>Tiara（Tiered, quality-Aware, Recompute-Aware KV manager）· 第三階段選題前的探索</div>")
    rows = [
        ["產生時間", f"{ctx.now:%Y-%m-%d %H:%M} UTC（台灣 {tw:%Y-%m-%d %H:%M}）"],
        ["平台", "2 × AMD Instinct MI300X（MLSteam 租用）；硬體與軟體細節見第 2 章"],
        ["結果來源", f"{esc(rel(ctx.res))}／（每個數字都附 run_id；原始 log 在 /mlsteam/data/tiara/runs/&lt;run_id&gt;/）"],
        ["實驗紀錄", esc(rel(ctx.D("experiment_log.md")))],
        ["程式版本", f"git HEAD {esc(git_head())}（產生器 code/m16_report_pdf.py）"],
        ["產生方式", "程式從結果檔即時讀數字畫圖；沒量到的一律寫 NOT_MEASURED；「解讀、結論、建議」由主 session 撰寫"
                  "（綠色左線框），還沒寫的放紅色虛線框「（待撰寫）」"],
    ]
    d.raw("<div class='cover-meta'>" + html_table(["項目", "內容"], rows, widths=["64pt", None]) + "</div>")


def ch_summary(d: Doc, ctx: Ctx):
    d.h(1, "summary", "一頁摘要")
    d.raw(narrative(ctx, "summary", "一頁摘要（這一輪的結論、候選題目、建議下一步）"))
    d.h3("實驗狀態一覽（由結果檔與佇列自動判斷）")
    rows = []
    for e in EXPS:
        st, have, tot = exp_status(ctx, e)
        cls = {"完成": "ok", "部分": "part", "NOT_MEASURED": "nm"}[st]
        rows.append([e["sec"], f"<a href='#exp_{e['key']}'>{esc(e['title'])}</a>", esc(e["kind"]),
                     f"<span class='{cls}'>{st}</span>（{have}／{tot}）", esc(queue_state(ctx, e["jobs"]))])
    d.raw(html_table(["節", "實驗", "性質", "結果檔", "佇列"], rows))
    d.raw("<p class='small'>「結果檔」＝該節主要結果檔存在幾個；「佇列」＝/mlsteam/data/tiara/stage3/jobs 的 job 狀態"
          "（完成／跑中／排隊／失敗）。狀態只看檔案在不在，不代表判定。</p>")


def ch_toc(d: Doc, ctx: Ctx):
    d.raw("<h1 id='toc'>目錄</h1><!--TOC-->")


def toc_html(ctx: Ctx, pages: dict) -> str:
    rows = []
    for lvl, hid, text in ctx.headings:
        if hid == "toc" or lvl > 2:
            continue
        pg = pages.get(hid, "")
        cls = "l2" if lvl == 2 else ""
        tt = f"<b>{esc(text)}</b>" if lvl == 1 else esc(text)
        rows.append(f"<tr><td class='{cls}'><a href='#{hid}'>{tt}</a></td><td style='text-align:right;width:36pt'>{pg}</td></tr>")
    return "<table class='toc' style='width:100%'>" + "".join(rows) + "</table>"


# ============================================================================ 第 1 章：背景與問題（事實；解讀在 narrative/background.html）
def ch_background(d: Doc, ctx: Ctx):
    d.h(1, "ch1", "1　背景與問題")
    d.h(2, "ch1_q", "1.1　大問題")
    d.p("長上下文推論時，KV cache 放不下 GPU。每一塊 KV 要依什麼決定它的待遇：降精度、無損壓縮、搬到 CPU／SSD，"
        "還是丟掉（之後重算，或依重要性直接不要）？前兩個階段試過「依位置」決定（第一階段：放哪一層；第二階段：用多少位元），"
        "這一輪看剩下的資訊——<b>內容</b>、<b>場景</b>、<b>現場狀況</b>——哪一種在 MI300X 上真的有問題、前人也沒做，"
        "並回答使用者的兩個問題（見 1.4）。這一輪是篩選：替第三階段挑題目，不是做出論文主張。")
    d.h(2, "ch1_treat", "1.2　KV 的待遇與這一輪對應的實驗")
    rows = [
        ["降精度（FP8、INT4、INT2）＝有損壓縮", "可能", "有（第二階段）", "X1、X2（拿來和其他方法比）"],
        ["無損壓縮（zstd；Intel KV Shrink 只壓 BF16 的高位元組）", "不會", "沒有", "S1、X3"],
        ["低秩、合併 token、跨層共用", "可能", "沒有", "K1、X1、X2、xKV 校準、X4"],
        ["搬到 CPU／SSD", "不會", "有", "I4、I1、F1、X3"],
        ["丟掉，之後重算", "不會", "有（Cake、G5b）", "I1"],
        ["依重要性直接丟掉，不再算回來", "可能", "沒有", "C1a、C1b、X1"],
    ]
    d.raw(html_table(["待遇", "影響答對率？", "專案原本有沒有", "這一輪的實驗"], rows))
    d.h(2, "ch1_acts", "1.3　四幕故事：用什麼資訊決定 KV 的待遇")
    rows = [
        ["第一幕", "依位置", "前兩階段已做：依位置決定放哪一層、用多少位元", "（第一、二階段；本報告不重述）"],
        ["第二幕", "依內容（token／block 的重要性；使用者問的①）",
         "重要性只有兩條路帶來好處：省空間（同大小下丟不重要的比量化好嗎？）與省搬運（猜得中下一輪要哪些 block 嗎？）；"
         "再加上「丟、低秩、合併、跨層能不能和量化疊在一起」", "C1a、C1b、K1、X1、X2、X4"],
        ["第三幕", "依場景（agent 工具暫停；使用者問的②）", "工具暫停時 KV 該留、搬、還是丟", "I3（模擬）"],
        ["第四幕", "依現場狀況（併發、GPU 忙、搬法）", "前三幕都默認「拿回來要多久」是固定的；GPU 忙時重算與搬運會變慢",
         "I4、I1／I2、X3、V0／F2／F1"],
    ]
    d.raw(html_table(["幕", "用來決定待遇的資訊", "要問的事", "對應實驗"], rows))
    d.h(2, "ch1_user", "1.4　使用者的兩個問題")
    d.raw("<ol><li><b>① token 重要性能不能用？</b>同樣省一半空間，丟不重要的 token 和量化誰保住答對率？"
          "重要性分數猜不猜得中下一輪要的 block？又量化、又壓縮、又丟，全部疊起來會不會更好？（第 4.4–4.11 節）</li>"
          "<li><b>② 場景改成 agent 會怎樣？</b>agent 呼叫工具暫停時，KV 該留在 GPU、搬到 CPU／SSD 還是丟掉重算？"
          "（第 4.12 節；這一輪只做模擬）</li></ol>")
    d.h(2, "ch1_intel", "1.5　Intel「KV Cache 四件套」與這一輪的對應")
    rows = [
        ["KV Shrink（壓縮）", "存儲不足", "量化、依重要性丟 token、無損壓縮（Intel 版：QAT 做 DEFLATE、只壓 BF16 指數）",
         "唯一查得到程式（GitHub intel/intel-accel-for-llm），沒有 ROCm 後端、沒有公布效能", "S1、X1–X4、X3"],
        ["KV Cascade（聚焦）", "信息稀疏", "只搬和新問題相關的 block，其餘留在下層、需要時再拿", "只有媒體報導一句話", "C1a、C1b"],
        ["KV Fuse（復用）", "復用困難", "共用前綴的 KV；放不下時搬到 CPU 的共享快取", "只有媒體報導一句話", "V0、F2、F1"],
        ["KV Infinity（按需載入＋預測預取）", "擴展受限", "KV 放在 CPU／SSD，需要時才搬回；猜得到就提前搬；Intel 說用 DSA 加速搬運",
         "只有媒體報導一句話（AMD 沒有 DSA，對應的是 SDMA 引擎）", "I4、I1／I2、I3"],
    ]
    d.raw(html_table(["名稱", "影片的難題", "作法", "查證（lit/L0）", "這一輪的實驗"], rows))
    d.h3("背景的補充說明（主 session 撰寫）")
    d.raw(narrative(ctx, "background", "背景補充（為什麼這樣排、和前兩階段的關係）"))


# ============================================================================ 第 2 章：環境與方法
def ch_env(d: Doc, ctx: Ctx):
    sec = "2"
    d.h(1, "ch2", "2　環境與方法")
    snap = ctx.D("environment_snapshot.md")
    d.h(2, "ch2_hw", "2.1　硬體與軟體")
    try:
        d.raw(md(md_section(snap, r"^## 2\. ")) + md(md_section(snap, r"^## 3\. ")))
        ctx.src(sec, snap)
        d.raw(f"<p class='small'>出處：{esc(rel(snap))}（10/19 取樣，只用唯讀指令）。</p>")
    except NotMeasured as e:
        ctx.not_measured(sec, "硬體與軟體表", str(e))
        d.raw(f"<div class='box-nm'>NOT_MEASURED：{esc(e)}</div>")
    d.h(2, "ch2_tasks", "2.2　模型、任務與上下文")
    d.p("模型：Llama-3.1-8B-Instruct 與 Qwen2.5-7B-Instruct-1M（vLLM 用 noDCA 版本）。上下文：主要 32K（64 個 512-token block），"
        "長上下文切片 128K（第二階段 44c 的題組）與 Qwen 256K（只在 X4c）。block 0 放 chat template 與任務說明，一律 BF16。"
        "答案（needle）放在前／中／後三個位置帶之一；兩輪題的第 1、2 輪答案放在不同的帶。")
    try:
        t = open(need(os.path.join(REPO, "code/m10_tasks.py")), encoding="utf-8").read()
        m = re.search(r"## 任務\n\n(\|.*?)\n\n", t, re.S)
        if not m:
            raise NotMeasured("code/m10_tasks.py 裡找不到任務表")
        d.raw(md(m.group(1)))
        d.raw("<p class='small'>任務表直接取自 code/m10_tasks.py 的檔頭（RULER 上游的模板與計分）。這一輪用 T2–T5；"
              "計分＝第二階段同一個（m10_tasks.score）。</p>")
        ctx.src(sec, os.path.join(REPO, "code/m10_tasks.py"))
    except NotMeasured as e:
        d.raw(f"<div class='box-nm'>NOT_MEASURED：{esc(e)}</div>")
    d.h(2, "ch2_rules", "2.3　和第二階段共用機器的規矩")
    try:
        d.raw(md(md_section(snap, r"^## 5\. ", r"^## ")))
    except NotMeasured as e:
        d.raw(f"<div class='box-nm'>NOT_MEASURED：{esc(e)}</div>")
    d.raw("<ul><li>量時間的 job（I4、I1、X3 micro／validate、C1b 的 qtime、vLLM 那組）獨佔兩張卡、在卡 1 量，"
          "用 gpu_guard 包住；被污染（contaminated）的 run 時間欄作廢、不寫進 results，要重量。</li>"
          "<li>量品質的 job（X1、X2、X2b、xKV 校準、C1a、C1b、X4）兩卡並行；同卡有自己的其他行程時，分數照用，"
          "但時間欄（t_gen_s 等）作廢、不拿來做任何時間結論（CLAUDE.md §3 的品質規則）。</li>"
          "<li>CPU 分析一律 nice 19、綁 0–3 號核心、不准 import torch（不開 /dev/kfd）。</li>"
          "<li>10/23 02:00Z（台灣 10:00）之後不開預計跑不完的 GPU job。</li></ul>")
    d.h(2, "ch2_log", "2.4　記錄協定")
    d.raw("<ul><li>每一次 run：/mlsteam/data/tiara/runs/&lt;run_id&gt;/ 有 cmd.sh、context.txt、stdout、stderr、exit_code"
          "（m7run 殼，CLAUDE.md §4.1）。</li>"
          "<li>results/m16_stage3_mi300x/ 的每個 CSV 都有 run_id、ts 欄；&gt;1 MB 的檔留在 run 目錄，results 裡留 BIG_FILES.md 指過去。</li>"
          "<li>逐實驗的紀錄（run_id、產出檔、關鍵數字、判定、失敗）在 docs/stage3_20261019/experiment_log.md；"
          "數字索引 docs/stage3_20261019/results.csv（code/m16_index.py 產生）。</li>"
          "<li>判定門檻都在看到數字之前寫進說明檔或程式常數；〔模擬〕＝用實測常數算的、不是量測；〔計算〕＝算術。</li>"
          "<li>審查都是 self-review，不是 cross-model review（這台機器沒有 Codex MCP；CLAUDE.md §5）。</li></ul>")
    d.h(2, "ch2_gantt", "2.5　實際執行時間軸")
    logs = [os.path.join(ctx.jobs, f"runner_{r}.log") for r in ("timing", "cpu")]

    def gantt():
        jobs = parse_runner_logs(logs, ctx.now)
        if not jobs:
            raise NotMeasured("runner log（runner_timing.log、runner_cpu.log）沒有 START／DONE 紀錄")
        for p in logs:
            ctx.src(sec, p)
        plt = plt_setup(ctx)
        import matplotlib.dates as mdates
        kinds = {"量時間（獨佔兩卡）": C[7], "量品質（兩卡並行）": C[0], "CPU 分析": C[2]}
        n = len(jobs)
        fig, ax = plt.subplots(figsize=(7.2, min(9.6, 1.2 + 0.125 * n)))
        t0 = min(j["start"] for j in jobs)
        t1 = max(j["end_shown"] for j in jobs)
        minw = (t1 - t0).total_seconds() / 86400 * 0.004
        for i, j in enumerate(jobs):
            s = mdates.date2num(j["start"])
            w = max(mdates.date2num(j["end_shown"]) - s, minw)
            col = kinds.get(j["kind"], "#888888")
            hatch = "////" if j["status"] == "跑中" else ("xxxx" if j["status"] == "失敗" else None)
            ax.barh(i, w, left=s, height=0.72, color=col, alpha=0.9 if j["status"] == "完成" else 0.45,
                    edgecolor=("#a5281b" if j["status"] == "失敗" else col), hatch=hatch, linewidth=0.6)
        ax.set_yticks(range(n))
        ax.set_yticklabels([f"{j['name']}{'（跑中）' if j['status'] == '跑中' else ('（失敗）' if j['status'] == '失敗' else '')}"
                            for j in jobs], fontsize=5.6)
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3 if (t1 - t0).total_seconds() < 2.5 * 86400 else 6)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))
        ax.axvline(mdates.date2num(ctx.now), color="#888", ls=":", lw=1)
        ax.grid(axis="y", visible=False)
        from matplotlib.patches import Patch
        hs = [Patch(facecolor=c, label=k) for k, c in kinds.items()]
        hs += [Patch(facecolor="white", edgecolor="#555", hatch="////", label="跑中（畫到產生報告的時間）"),
               Patch(facecolor="white", edgecolor="#a5281b", hatch="xxxx", label="失敗（之後修好重開）")]
        ax.legend(handles=hs, loc="upper right", fontsize=6.5)
        ax.set_xlabel("時間（UTC；台灣＝UTC+8）")
        fig.tight_layout()
        return fig, (f"第三階段 runner 的 START／DONE／FAILED 紀錄（{n} 段）。紅＝量時間、藍＝量品質、綠＝CPU 分析。"
                     "出處：/mlsteam/data/tiara/stage3/jobs/runner_timing.log、runner_cpu.log。")
    d.raw(add_fig(ctx, sec, "第三階段 job 的實際執行時間軸", gantt))

    def gsum():
        jobs = parse_runner_logs(logs, ctx.now)
        if not jobs:
            raise NotMeasured("runner log 沒有紀錄")
        agg = defaultdict(lambda: [0, 0.0, 0, 0])
        for j in jobs:
            a = agg[j["kind"]]
            a[0] += 1
            a[1] += ((j["end"] or ctx.now) - j["start"]).total_seconds() / 3600
            a[2] += j["status"] == "失敗"
            a[3] += j["status"] == "跑中"
        rows = [[esc(k), str(v[0]), f"{v[1]:.1f}", str(v[2]), str(v[3])] for k, v in agg.items()]
        first = min(j["start"] for j in jobs)
        return (["種類", "段數", "合計牆鐘（小時）", "失敗", "跑中"], rows,
                f"第一個 GPU job 開始於 {first:%m/%d %H:%M}Z；跑中的段算到 {ctx.now:%m/%d %H:%M}Z。")
    d.raw(add_table(ctx, sec, "各種 job 的牆鐘時間", gsum))
    d.h3("出處")
    d.raw(provenance(ctx, sec))


# ============================================================================ 第 3 章：文獻與查新
LIT_FILES = [
    ("L0_intel_suite_and_amd.md", r"^## 0\. ", "Intel 四件套查證、AMD／ROCm 對應"),
    ("L1_eviction_selective_loading.md", r"^## 0\. ", "eviction、token 重要度、選擇性載入（Cascade）"),
    ("L2_compression_offload_prefetch.md", r"^## 0\. ", "壓縮 × 卸載 × 預取"),
    ("L3_reuse_serving_agentic.md", r"^## §0 ", "復用、多請求服務、agent 工作負載"),
    ("L4_novelty_fulltext.md", r"^## 0\. ", "六篇全文查新（Cascade、CacheSmith、CacheFlow 等）"),
    ("L5a_lowrank.md", r"^## 0\. ", "低秩／降維"),
    ("L5b_merge_crosslayer.md", r"^## 0\. ", "token 合併與跨層共用"),
    ("L5c_compress_on_offload.md", r"^## 0\. ", "卸載時壓縮、依連線與 GPU 忙碌選格式"),
    ("L5d_stacking.md", r"^## 0\. ", "多種壓縮疊加"),
    ("L6_mainstream_practice.md", r"^## 0\. ", "業界主流作法與精度"),
]


def first_point(text: str) -> str:
    m = re.search(r"\*\*(.+?)\*\*", text)
    s = m.group(1) if m else re.sub(r"\s+", " ", text)[:90]
    return s if len(s) <= 120 else s[:117] + "…"


def ch_lit(d: Doc, ctx: Ctx):
    sec = "3"
    d.h(1, "ch3", "3　文獻與查新")
    d.p("文獻地圖由多個 agent 在 10/19–10/20 讀過整理（lit/L0–L6；全部 self-review）。下面只摘錄各檔的「一頁結論」，"
        "細節、數字的條件與閱讀深度（〔全文〕〔部分〕〔摘要〕）都在原檔；不同硬體、batch、context 的數字只並列、不排名。"
        "「沒找到」＝這輪搜尋沒找到，不等於不存在。")
    rm = ctx.D("research_map.md")
    d.h(2, "ch3_map", "3.1　研究地圖的一頁結論")
    try:
        d.raw(md(md_section(rm, r"^## 0\. ")))
        ctx.src(sec, rm)
    except NotMeasured as e:
        d.raw(f"<div class='box-nm'>NOT_MEASURED：{esc(e)}</div>")
    d.h(2, "ch3_table", "3.2　文獻檔總表")

    def lit_table():
        rows = []
        for f, pat, topic in LIT_FILES:
            p = ctx.D("lit", f)
            if not os.path.exists(p):
                rows.append([esc(f), esc(topic), "—", "—", "<span class='nm'>NOT_FOUND</span>"])
                continue
            t = open(p, encoding="utf-8").read()
            ncard = len(re.findall(r"^#### ", t, re.M))
            nfull = t.count("〔全文〕")
            try:
                fp = first_point(md_section(p, pat))
            except NotMeasured:
                fp = "—"
            rows.append([esc(f.split("_")[0]), esc(topic), str(ncard), str(nfull), esc(fp)])
            ctx.src(sec, p)
        return (["檔", "範圍", "卡片（####）", "〔全文〕標記數", "一頁結論的第一個重點"], rows,
                "卡片數與〔全文〕標記數是直接數原檔的標題與標記（標記數含引用處，只當閱讀深度的粗略指標）。")
    d.raw(add_table(ctx, sec, "L0–L6 文獻檔", lit_table))
    d.h(2, "ch3_each", "3.3　各檔重點（摘錄一頁結論）")
    for f, pat, topic in LIT_FILES:
        p = ctx.D("lit", f)
        d.h3(f"{f.split('_')[0]}　{topic}")
        try:
            d.raw(md(md_section(p, pat, max_chars=900)))
        except NotMeasured as e:
            d.raw(f"<div class='box-nm'>NOT_FOUND：{esc(e)}</div>")
    d.h(2, "ch3_gaps", "3.4　缺口排序（研究地圖 §2）")
    try:
        d.raw(md(md_section(rm, r"^## 2\. ")))
    except NotMeasured as e:
        d.raw(f"<div class='box-nm'>NOT_MEASURED：{esc(e)}</div>")
    d.h3("文獻的解讀（主 session 撰寫）")
    d.raw(narrative(ctx, "literature", "文獻的整體判讀（哪些前提被推翻、哪些空白還在）"))


# ============================================================================ 第 4 章：實驗（每節：問題 → 作法 → 圖表 → 判定 → 解讀 → 限制 → 出處）
def nice_log(ax, axis="y"):
    """對數軸的刻度用 1、2、5×10^k，寫成一般數字（不寫 10^k）。"""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    a.set_major_formatter(FuncFormatter(lambda v, _: minus(f"{v:g}")))
    a.set_minor_formatter(NullFormatter())


def EX(key):
    return next(e for e in EXPS if e["key"] == key)


def two_panels(plt, w=7.2, h=3.0, sharey=False, n=2):
    fig, axs = plt.subplots(1, n, figsize=(w, h), sharey=sharey)
    return fig, axs


# ---------------------------------------------------------------------------- 4.1 K1
def sec_k1(d: Doc, ctx: Ctx):
    e = EX("k1")
    sec = e["sec"]
    K = ctx.R("k1_kvstruct")
    spec = dict(
        q="KV 本身有沒有能利用的冗餘？三件事：K（RoPE 前／後）與 V 是不是低秩；相鄰層的同一個 token 像不像"
          "（MiniCache／KVSharer 式跨層共用的前提）；相鄰 token 像不像（token 合併的前提）。這是 KV 的幾何量，不是答對率。",
        method="用 S1 匯出的真實 KV（兩個模型 × T4 自然文字、T2 重複句子各一題 × 32K 全層；128K 的 T4 一題選部分層），只在 CPU 上算："
               "每（層, head）的 SVD 重建相對誤差 ‖X−X_r‖_F／‖X‖_F；相鄰層同一 token、同一 head 的 cosine（含減掉平均的版本）；"
               "相鄰 token 的 cosine。RoPE 逆轉先做資料自身核對（每份 dump 都要 PASS 才用 RoPE 前的 K）。",
        rule="L5b 寫死的條件：深半段（後一半的層）<b>中心化</b> cosine 的中位數 &lt; 0.5 → MiniCache／KVSharer 式「逐 token 直接共用」"
             "在這兩個模型上沒有依據，停；xKV 式（共用子空間）不受這條影響，交給 X1 量品質。",
        limits=["每個（模型, 任務）只有一題的 KV；128K 只選部分層。",
                "幾何量不等於答對率：低秩誤差小不保證生成答對，能不能換成答對率要看 X1／X2。",
                "低秩是每（層, head）各自的 SVD（不是 xKV 的跨層聯合 SVD）；聯合版的誤差只在 lowrank.csv 的 head＝all 列。",
                "K1b（Tucker）只比重建誤差、不去平均（照 xKV），K 的平均方向佔大部分能量，所以誤差的絕對值看起來很小，要看比值；"
                "Tucker 只有上下界、沒有真的組核心張量；位元組只算結構（同精度），沒有含量化。"])
    exp_head(d, ctx, e, spec)

    def kdir():
        return big_dir(os.path.join(K, "BIG_FILES.md"), "k1_full")

    def f_lowrank():
        p = os.path.join(kdir(), "lowrank.csv")
        rows = read_csv(p)
        ctx.src(sec, p, rids_of_rows(rows))
        g = defaultdict(list)
        for r in rows:
            if r["head"] != "all":
                g[(r["dump"], r["kind"], int(r["rank"]))].append(float(r["rel_err"]))
        plt = plt_setup(ctx)
        fig, axs = two_panels(plt, h=2.9, sharey=True)
        kc = {"K_pre": C[0], "K_post": C[1], "V": C[2]}
        lab = {"K_pre": "K（RoPE 前）", "K_post": "K（RoPE 後）", "V": "V"}
        for ax, m in zip(axs, MODELS):
            for kind, col in kc.items():
                for dump, ls, tag in ((f"{m}_T4_64", "-", "32K"), (f"{m}_T4_252", "--", "128K")):
                    pts = sorted((rk, statistics.median(v)) for (dd, kk, rk), v in g.items() if dd == dump and kk == kind and rk <= 128)
                    if pts:
                        ax.plot([x for x, _ in pts], [y for _, y in pts], ls, marker="o", ms=3.5, color=col,
                                label=f"{lab[kind]}・{tag}")
            ax.set_xscale("log", base=2)
            rks = sorted({rk for (_, _, rk) in g if rk <= 128})
            ax.set_xticks(rks)
            ax.set_xticklabels([str(x) for x in rks])
            ax.minorticks_off()
            ax.set_title(f"{MODEL_LABEL[m]}（T4）")
            ax.set_xlabel("rank r（每個 head 的維度 128）")
            ax.set_ylim(0, 1)
        axs[0].set_ylabel("重建相對誤差（層×head 中位數）")
        axs[1].legend(fontsize=6.4, ncol=1)
        fig.tight_layout()
        return fig, "實線＝32K、虛線＝128K（選層）。越低越好；K 在 RoPE 前比 RoPE 後低秩得多、V 最不低秩。出處：k1_full/lowrank.csv（head≠all 的列）。"
    d.raw(add_fig(ctx, sec, "K1：每（層, head）的低秩重建誤差 vs rank", f_lowrank))

    def f_xlayer():
        p = os.path.join(K, "xlayer.csv")
        rows = read_csv(p)
        ctx.src(sec, p, rids_of_rows(rows))
        g = defaultdict(list)
        for r in rows:
            if r["head"] != "all":
                g[(r["dump"], r["kind"], int(r["layer"]))].append(float(r["p50"]))
        plt = plt_setup(ctx)
        fig, axs = two_panels(plt, h=2.8, sharey=True)
        kc = {"K": C[0], "K_centered": C[1], "V": C[2], "V_centered": C[6]}
        lab = {"K": "K", "K_centered": "K（減平均）", "V": "V", "V_centered": "V（減平均）"}
        for ax, m in zip(axs, MODELS):
            for kind, col in kc.items():
                pts = sorted((ly, statistics.median(v)) for (dd, kk, ly), v in g.items() if dd == f"{m}_T4_64" and kk == kind)
                if pts:
                    ax.plot([x for x, _ in pts], [y for _, y in pts], "-", marker=".", ms=3, color=col, label=lab[kind])
            ax.axhline(0.5, color=C[7], ls="--", lw=1)
            ax.text(0.5, 0.52, "L5b 門檻 0.5", color=INK2, fontsize=6.8)
            ax.axhline(0, color="#999", lw=0.6)
            ax.set_title(f"{MODEL_LABEL[m]}（T4，32K）")
            ax.set_xlabel("層 l（和 l+1 比）")
            ax.set_ylim(-0.2, 1.0)
        axs[0].set_ylabel("同一 token 相鄰層 cosine（head 中位數的 p50）")
        axs[1].legend()
        fig.tight_layout()
        return fig, "每一層對下一層、同一 token 同一 head 的 cosine 中位數（再取 head 的中位數）。出處：k1_kvstruct/xlayer.csv。"
    d.raw(add_fig(ctx, sec, "K1：相鄰層同一 token 的 cosine", f_xlayer))

    def f_adj():
        p = os.path.join(kdir(), "adjtok.csv")
        rows = read_csv(p)
        ctx.src(sec, p, rids_of_rows(rows))
        kinds = ["K_post", "K_pre", "K_pre_centered", "V", "V_centered"]
        lab = {"K_post": "K\nRoPE 後", "K_pre": "K\nRoPE 前", "K_pre_centered": "K RoPE 前\n減平均", "V": "V", "V_centered": "V\n減平均"}
        plt = plt_setup(ctx)
        fig, ax = plt.subplots(figsize=(7.2, 2.8))
        for j, m in enumerate(MODELS):
            data = [[float(r["p50"]) for r in rows if r["dump"] == f"{m}_T4_64" and r["kind"] == k and r["head"] != "all"] for k in kinds]
            pos = [i * 3 + j for i in range(len(kinds))]
            if any(data):
                bp = ax.boxplot([x if x else [float("nan")] for x in data], positions=pos, widths=0.8, patch_artist=True,
                                showfliers=False)
                for b in bp["boxes"]:
                    b.set(facecolor=MODEL_COLOR[m], alpha=0.55, edgecolor=MODEL_COLOR[m])
                for k2 in ("medians",):
                    for ln in bp[k2]:
                        ln.set(color=INK)
            for i, k in enumerate(kinds):
                fr = [float(r["frac_ge_0.9"]) for r in rows if r["dump"] == f"{m}_T4_64" and r["kind"] == k and r["head"] != "all"]
                if fr:
                    ax.text(pos[i], -0.25, f"≥0.9：\n{100 * statistics.median(fr):.1f}%", ha="center", fontsize=5.6, color=INK2)
        ax.set_xticks([i * 3 + 0.5 for i in range(len(kinds))])
        ax.set_xticklabels([lab[k] for k in kinds], fontsize=7)
        ax.set_ylim(-0.38, 1.02)
        ax.set_ylabel("相鄰 token cosine 的中位數（每層×head）")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=MODEL_COLOR[m], alpha=0.55, label=MODEL_LABEL[m]) for m in MODELS], loc="upper right")
        fig.tight_layout()
        return fig, ("每個盒子＝所有（層, head）的相鄰 token cosine 中位數的分布（T4，32K）；下方數字＝cosine ≥0.9 的相鄰對比例"
                     "（層×head 的中位數）。出處：k1_full/adjtok.csv。")
    d.raw(add_fig(ctx, sec, "K1：相鄰 token 的 cosine", f_adj))

    # ---- K1b（10/22 補）：Tucker 與跨層共用；程式 code/m16_k1_tucker.py，事先寫死的條件在 experiment_log.md「K1b」
    TK = os.path.join(K, "tucker")
    d.h3("K1b　Tucker（多維分解）與「跨層共用的到底是什麼」（10/22 補）")
    d.raw("<div class='box-rule'>使用者 10/22 問：「相鄰層同一 token 的 KV 完全不像，這樣不能跨層？」「Tucker 多一個維度壓縮？」"
          "同一批 32K dump（兩個模型 × T4／T2，全部層；K＝RoPE 前；不去平均＝照 X 系列的 xKV），只用 CPU。事先寫死：<br>"
          "<b>Q1</b>：相鄰層的 token 子空間重疊（每層前 64 個左奇異向量；‖U<sub>l</sub>ᵀU<sub>l+1</sub>‖²／64，減平均）≥0.5 且同 token cosine ≤0.1"
          " → 跨層共用的是「token 子空間」不是「向量」。<br>"
          "<b>Q2</b>：在 xKV 每 4 層（X 系列）的操作點、同樣位元組下，4 維／3 維 Tucker 的上界（HOSVD）或兩側分解的誤差 ≤0.9×xKV，"
          "而且兩個模型的 K 都有 → GO（值得做 GPU 答對率）；否則 STOP。指標＝相對 Frobenius 誤差，只能篩選（X1：和答對率只中度相關）。</div>")

    def f_k1b_sub():
        p = os.path.join(TK, "xlayer_subspace.csv")
        rows = read_csv(p)
        ctx.src(sec, p, rids_of_rows(rows))
        plt = plt_setup(ctx)
        fig, axs = two_panels(plt, h=2.8, sharey=True)
        for ax, m in zip(axs, MODELS):
            for kind, col in (("K", C[0]), ("V", C[2])):
                g = defaultdict(list)
                for r in rows:
                    if r["model"] == m and r["kind"] == kind:
                        g[int(r["l"])].append(r)
                ls_ = sorted(g)
                if not ls_:
                    continue
                ax.plot(ls_, [statistics.mean(float(r["overlap_centered"]) for r in g[x]) for x in ls_], "-", marker=".", ms=3,
                        color=col, label=f"{kind}：token 子空間重疊（減平均）")
                ax.plot(ls_, [statistics.mean(float(r["cos"]) for r in g[x]) for x in ls_], ":", color=col, lw=1.3,
                        label=f"{kind}：同一 token 的 cosine")
            ax.axhline(statistics.mean(float(r["overlap_random"]) for r in rows), color="#999", lw=0.6)
            ax.axhline(0.5, color=C[7], ls="--", lw=1)
            ax.set_title(f"{MODEL_LABEL[m]}（T4／T2 平均，32K）")
            ax.set_xlabel("層 l（和 l+1 比）")
            ax.set_ylim(-0.1, 1.0)
        axs[0].set_ylabel("重疊／cosine")
        axs[1].legend(fontsize=6.4)
        fig.tight_layout()
        return fig, ("實線＝相鄰層 token 子空間重疊（1＝同一個子空間；灰線＝隨機子空間的期望值 64／32768）；點線＝同一 token 的 cosine"
                     "（同一座標比，K1 的那個量）。紅虛線＝Q1 門檻 0.5。出處：k1_kvstruct/tucker/xlayer_subspace.csv。")
    d.raw(add_fig(ctx, sec, "K1b：相鄰層「向量不像」但「token 子空間很像」", f_k1b_sub))

    def f_k1b_curves():
        fs = sorted(glob.glob(os.path.join(TK, "tucker_*.json")))
        if not fs:
            raise NotMeasured("k1_kvstruct/tucker/tucker_*.json 不存在")
        js = [read_json(f) for f in fs]
        for f, j in zip(fs, js):
            ctx.src(sec, f, [j["run_id"]])
        plt = plt_setup(ctx)
        fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6))
        for i, m in enumerate(MODELS):
            jm = [j for j in js if j["model"] == m]
            if not jm:
                continue
            rg, Lm, G2 = jm[0]["rgrid"], jm[0]["L"], max(jm[0]["two"])
            for c, kind in enumerate(("K", "V")):
                ax = axs[i][c]

                def cv(k):
                    out = []
                    for n in range(len(rg)):
                        vals = [j["kinds"][kind]["curves"][k][n] for j in jm if j["kinds"][kind]["curves"][k][n] is not None]
                        out.append(statistics.mean(vals) if len(vals) == len(jm) else float("nan"))
                    return out
                lines = [("xkv_G1", "每層各自 SVD", "#9aa0a6", "-", 1.1), ("xkv_G4", "xKV 每 4 層（X 系列）", C[0], "-", 2.2),
                         (f"xkv_G{Lm}", f"全部 {Lm} 層一組（＝Tucker 的 token 模態）", "#0b2f5c", "-", 1.1),
                         ("tucker4_hi", "4 維 Tucker 上界（HOSVD）", C[7], "-", 1.3),
                         ("tucker4_lo", "4 維 Tucker 下界（最樂觀）", C[7], ":", 1.4),
                         (f"two_G{G2}", f"兩側：token 跨 {G2} 層＋每層通道基底", C[2], "--", 1.3)]
                ymax = 0.0
                for k, lab, col, ls, lw in lines:
                    if k not in jm[0]["kinds"][kind]["curves"]:
                        continue
                    ys = cv(k)
                    xs = [1 / r for r in rg]
                    ax.plot(xs, ys, ls, color=col, lw=lw, label=lab)
                    ymax = max([ymax] + [y for x, y in zip(xs, ys) if 1.5 <= x <= 48 and not math.isnan(y)])
                for t in jm[0]["kinds"][kind]["op_points"]:
                    if t["in_x_series"]:
                        ax.axvline(t["factor"], color="#bbbbbb", lw=0.6, ls="--")
                ax.set_xscale("log")
                ax.set_xlim(1.5, 48)
                ax.set_ylim(0, ymax * 1.05 if ymax > 0 else 1)
                ax.set_title(f"{MODEL_LABEL[m]}：{'K（RoPE 前）' if kind == 'K' else 'V'}")
                if i == 1:
                    ax.set_xlabel("結構壓縮倍數（同精度，不含量化）")
                if c == 0:
                    ax.set_ylabel("相對誤差（不去平均）")
        axs[0][1].legend(fontsize=6.0, loc="upper left")
        fig.tight_layout()
        return fig, ("同樣位元組（橫軸）下各種分解的重建誤差；灰虛線＝X 系列實際用的 xKV 操作點（K 11／22／28、V 17／34／42）。"
                     "Tucker 只畫得出上下界（不組核心張量）；下界是「任何 Tucker 都不可能更好」。兩題平均。出處：k1_kvstruct/tucker/tucker_*.json。")
    d.raw(add_fig(ctx, sec, "K1b：同位元組下，xKV、Tucker、兩側分解的重建誤差", f_k1b_curves))

    def t_k1b():
        s = read_json(os.path.join(TK, "summary.json"))
        ctx.src(sec, os.path.join(TK, "summary.json"), [s["run_id"]] + list(s["sources"]))
        out = []
        for key, q in s["q2"].items():
            m, kind = key.rsplit("_", 1)
            for t in q["table"]:
                if not t["in_x_series"]:
                    continue
                gs = [k for k in t if k.startswith("xkv_G") and not k.endswith("_ratio")]
                gall = max(gs, key=lambda k: int(k[5:]))
                two = [t[k + "_ratio"] for k in t if k.startswith("two_G") and not k.endswith("_ratio") and t[k + "_ratio"] is not None]
                out.append([esc(MODEL_LABEL.get(m, m)), kind, f"{kind.lower()}{t['rhe']}", fx(t["factor"], 1) + "×", fx(t["xkv_G4"], 4),
                            fx(t["xkv_G1_ratio"], 2), fx(t[gall + "_ratio"], 2), fx(t["tucker4_hi_ratio"], 2), fx(t["tucker4_lo_ratio"], 2),
                            fx(t["tucker3_hi_ratio"], 2), fx(min(two) if two else None, 2)])
        q1 = "；".join(f"{esc(k)} cosine {fx(v['cos'], 3)}、重疊（減平均）{fx(v['overlap_centered'], 2)} → <b>{esc(v['verdict'])}</b>"
                       for k, v in s["q1"].items())
        return (["模型", "種類", "X 系列設定", "結構倍數", "xKV-G4 誤差", "每層各自", "全部層一組", "4 維 Tucker 上界", "4 維 Tucker 下界",
                 "3 維 Tucker 上界", "兩側（最好）"], out,
                f"從「每層各自」起的欄位＝同位元組下的相對誤差 ÷ xKV-G4（&lt;1 比 xKV-G4 好）；兩題平均。Q1：{q1}。"
                f"Q2 判定：<b>{esc(s['verdict_q2'])}</b>。出處：k1_kvstruct/tucker/summary.json（op_points.csv 有全部操作點）。")
    d.raw(add_table(ctx, sec, "K1b：X 系列操作點上，其他分解相對 xKV-G4 的誤差比", t_k1b))

    def rule():
        def t():
            p = os.path.join(K, "xlayer.csv")
            rows = read_csv(p)
            nl = {}
            for r in rows:
                if r["dump"].endswith("_64"):
                    nl[r["model"]] = max(nl.get(r["model"], 0), int(r["layer_next"]) + 1)
            out = []
            for dump in sorted({r["dump"] for r in rows}):
                m = dump.split("_")[0]
                for kind in ("K_centered", "V_centered"):
                    v = [float(r["p50"]) for r in rows if r["dump"] == dump and r["kind"] == kind and r["head"] != "all"
                         and int(r["layer"]) >= nl.get(m, 0) // 2]
                    if v:
                        med = statistics.median(v)
                        out.append([esc(dump), esc(kind), fx(med, 3), str(len(v)),
                                    "<span class='ok'>成立（&lt;0.5 → 逐 token 共用停）</span>" if med < 0.5 else "<span class='nm'>不成立</span>"])
            return (["dump", "種類", "深半段 cosine 中位數", "（層×head）數", "L5b 條件"], out,
                    "深半段＝層 ≥ 總層數／2（總層數取 32K dump 的層數）；128K dump 只有選到的層。")
        d.raw(add_table(ctx, sec, "K1 的事先寫死條件（由 xlayer.csv 即時計算）", t))
        try:
            s = read_json(os.path.join(K, "summary.json"))
            ctx.src(sec, os.path.join(K, "summary.json"), [s.get("run_id")])
            rs = s.get("rope_status", {})
            d.p("RoPE 資料自身核對：" + "、".join(f"{esc(k)} <b>{esc(v)}</b>" for k, v in rs.items()) + "（summary.json 的 rope_status）。")
        except NotMeasured as ex:
            d.raw(f"<div class='box-nm'>NOT_MEASURED：{esc(ex)}</div>")
    exp_tail(d, ctx, e, spec, rule)


# ---------------------------------------------------------------------------- 4.2 I4
def sec_i4(d: Doc, ctx: Ctx):
    e = EX("i4")
    sec = e["sec"]
    D_ = ctx.R("i4_copy_batching")
    spec = dict(
        q="vLLM 0.28 在 ROCm 上的 CPU 卸載層很慢（第一階段量過，見 lit/L0、L2）。瓶頸是每次 copy 的固定成本還是頻寬？"
          "換搬法（逐片 copy、foreach、合併、CPU 端先收集再一次搬、GPU kernel 直接讀 pinned host）能快多少？"
          "SDMA copy engine 與 blit kernel、GPU 忙的時候有沒有差？",
        method="microbenchmark（code/m16_i4_copybench.py）：同樣的位元組切成 vLLM 大小的片（Llama：16 token × 層 × K／V），"
               "兩種版面（連續 contig／散在 scatter）× 五種搬法 × GPU 閒／busy2（背景 matmul）× HSA_ENABLE_SDMA＝0／1，"
               "每格 5 次取中位數；量時間的 job 獨佔兩張卡、gpu_guard 監控（污染的列不用）。",
        rule="experiment_matrix 寫死：最好的搬法 ≥ 3 倍於逐片 → 成功（vLLM 的數字要分「引擎現況」與「修好之後」兩組報告，"
             "而且這本身是一個可寫的系統問題）；差 &lt; 1.5 倍 → 停。缺口 1 的證偽之一：搬法在閒／忙的差 &lt; 5%。",
        limits=["單純搬運的 microbench，不在 vLLM 裡；沒有端到端還原。",
                "Qwen 的片大小（16 KiB）沒有另量；C1b 的 vLLM 版面時間沿用 Llama 的每片成本＝推算。",
                "GPU 忙只有 busy2 一個等級；同時跑反而比單獨快的格子原因未查。"])
    exp_head(d, ctx, e, spec)

    def rows_all():
        out = []
        for s in (0, 1):
            p = os.path.join(D_, f"copybench_sdma{s}.csv")
            if os.path.exists(p):
                rr = read_csv(p)
                ctx.src(sec, p, rids_of_rows(rr))
                out += rr
        if not out:
            raise NotMeasured("copybench_sdma{0,1}.csv 都不存在")
        return out

    def f_bw():
        rows = [r for r in rows_all() if r.get("status") == "ok" and r.get("contaminated") != "True"]
        methods = [m for m in ("py_loop", "foreach", "coalesce", "stage", "zerocopy") if any(r["method"] == m for r in rows)]
        combos = [(s, g) for s in ("0", "1") for g in ("idle", "busy2")]
        plt = plt_setup(ctx)
        fig, axs = two_panels(plt, h=3.0, sharey=True)
        for ax, lay in zip(axs, ("scatter", "contig")):
            w = 0.8 / len(combos)
            for k, (s, g) in enumerate(combos):
                ys = []
                for m in methods:
                    v = [float(r["GiBps"]) for r in rows if r["layout"] == lay and r["method"] == m and r["sdma"] == s and r["gpu_state"] == g]
                    ys.append(v[0] if v else float("nan"))
                ax.bar([i + (k - 1.5) * w for i in range(len(methods))], ys, width=w * 0.92, color=C[k],
                       label=f"SDMA {'開' if s == '1' else '關'}・GPU {'閒' if g == 'idle' else '忙（busy2）'}")
            ax.set_yscale("log")
            nice_log(ax)
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, fontsize=7)
            ax.set_title("散在小塊（scatter）" if lay == "scatter" else "連續（contig）")
        axs[0].set_ylabel("有效 H2D（GiB/s，對數）")
        h_, l_ = axs[0].get_legend_handles_labels()
        fig.legend(h_, l_, loc="lower center", ncol=4, fontsize=6.6)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        return fig, "每格 5 次的中位數。py_loop／foreach＝逐片 copy；coalesce＝合併相鄰片；stage＝CPU 端先收集再一次搬；zerocopy＝GPU kernel 直接讀 pinned host。"
    d.raw(add_fig(ctx, sec, "I4：各搬法的有效 GiB/s", f_bw))

    def rule():
        def t():
            rows = [r for r in rows_all() if r.get("status") == "ok"]
            out = []
            for s in ("0", "1"):
                for g in ("idle", "busy2"):
                    sc = {r["method"]: float(r["GiBps"]) for r in rows if r["sdma"] == s and r["gpu_state"] == g and r["layout"] == "scatter"}
                    if "py_loop" not in sc:
                        continue
                    best = max(sc, key=sc.get)
                    ratio = sc[best] / sc["py_loop"]
                    v = "≥3 倍（成功條件）" if ratio >= 3 else ("&lt;1.5 倍（停）" if ratio < 1.5 else "1.5–3 倍")
                    out.append([f"SDMA {'開' if s == '1' else '關'}", "閒" if g == "idle" else "busy2", esc(best),
                                fx(sc[best], 2), fx(sc["py_loop"], 2), fx(ratio, 1) + "×", v])
            return (["SDMA", "GPU", "最好的搬法", "最好 GiB/s", "逐片 py_loop GiB/s", "倍數", "判定"], out, "scatter 版面（vLLM 的實際情況）。")
        d.raw(add_table(ctx, sec, "I4：最好的搬法對逐片 copy（即時計算）", t))

        def t2():
            rows = [r for r in rows_all() if r.get("status") == "ok"]
            out = []
            for s in ("0", "1"):
                for lay in ("scatter", "contig"):
                    for m in ("py_loop", "foreach", "coalesce", "stage", "zerocopy"):
                        a = [float(r["GiBps"]) for r in rows if r["sdma"] == s and r["layout"] == lay and r["method"] == m and r["gpu_state"] == "idle"]
                        b = [float(r["GiBps"]) for r in rows if r["sdma"] == s and r["layout"] == lay and r["method"] == m and r["gpu_state"] == "busy2"]
                        u = [r.get("us_per_frag") for r in rows if r["sdma"] == s and r["layout"] == lay and r["method"] == m and r["gpu_state"] == "idle"]
                        if a and b:
                            out.append([f"SDMA {'開' if s == '1' else '關'}", lay, m, fx(a[0], 2), fx(b[0], 2), fx(b[0] / a[0], 3), fx(u[0], 3) if u else "—"])
            return (["SDMA", "版面", "搬法", "閒 GiB/s", "busy2 GiB/s", "忙／閒", "每片 µs（閒）"], out,
                    "忙／閒＝1 表示 GPU 忙不影響這個搬法；缺口 1 的證偽條件看這一欄是否都在 ±5% 內。")
        d.raw(add_table(ctx, sec, "I4：GPU 忙對各搬法的影響（即時計算）", t2))
    exp_tail(d, ctx, e, spec, rule)


# ---------------------------------------------------------------------------- 4.3 I1／I2
def sec_i1(d: Doc, ctx: Ctx):
    e = EX("i1")
    sec = e["sec"]
    D_ = ctx.R("i1_concurrent_restore")
    spec = dict(
        q="多個 session 同時回來時，「同時考慮所有 session 決定誰重算、誰載入」（joint）有沒有比各自用第二階段的規則好？"
          "重算、搬運、decode 同時跑會互相拖慢多少（前作 CacheFlow、InferCept、FSL-26-02 都假設互不干擾）？用實測常數做的模擬可不可信？",
        method="先用第二階段實測常數寫多 session 的 Cake 排程模擬（code/m16_mcake.py），在看任何 GPU 結果前做事先預測〔模擬〕；"
               "再在 GPU 上實測：(a) interf：重算、H2D、decode 單獨跑 vs 同時跑的時間比（I2 併入）；"
               "(b) grid：Llama 32K、N＝1／4／8、CPU 層／限速 b32／混合 × cake／g5b／joint × 輪流／先到先服務／最短優先，"
               "每格 3 次，和同一行程的模擬比。",
        rule="事先寫死：實測／模擬（平均 TTFT，每格 3 次的中位數）≤ 1.10 → 模擬可信、I1 收掉（「聯合決定」的負面結果成立）。"
             "設計卡：N＝8 時最好的聯合策略比各自 Cake 的 P95 少 ≥ 15% → 成功；&lt; 5% → 停。",
        limits=["只有 Llama、32K；grid 只在 GPU 閒；干擾只量了 busy2 一個等級。",
                "joint 只試了「每個 session 全算或全載」一種；更細的聯合（每個 session 的會合點一起調）沒試。",
                "事先預測是模擬（排程是模型，常數實測）；「載入在計算忙時 +50%」只是敏感度參數。"])
    exp_head(d, ctx, e, spec)

    def f_grid():
        p = os.path.join(D_, "grid_llama.csv")
        rows = read_csv(p)
        ctx.src(sec, p, rids_of_rows(rows))
        cells = defaultdict(list)
        for r in rows:
            if r.get("contaminated") == "True":
                continue
            cells[(r["workload"], int(r["N"]), r["mode"], r["order"])].append(float(r["ratio_mean"]))
        plt = plt_setup(ctx)
        fig, ax = plt.subplots(figsize=(7.2, 2.9))
        wl = sorted({k[0] for k in cells})
        mk = {"cake": "o", "g5b": "s", "joint": "^"}
        import random
        rnd = random.Random(0)
        for i, w in enumerate(wl):
            for (ww, n, mode, order), v in cells.items():
                if ww != w:
                    continue
                x = {1: 0, 4: 1, 8: 2}.get(n, 3) + (i - 1) * 0.22 + rnd.uniform(-0.06, 0.06)
                ax.scatter(x, statistics.median(v), color=C[i], marker=mk.get(mode, "x"), s=16, alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.axhline(1.10, color=C[7], ls="--", lw=1)
        ax.text(0.99, 1.10, "門檻 1.10", color=INK2, fontsize=7, ha="right", va="bottom", transform=ax.get_yaxis_transform())
        ax.axhline(1.0, color="#888", lw=0.7)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["N＝1", "N＝4", "N＝8"])
        ax.set_ylabel("實測 ÷ 模擬（平均 TTFT）")
        from matplotlib.lines import Line2D
        hs = [Line2D([], [], color=C[i], marker="o", ls="", label=w) for i, w in enumerate(wl)]
        hs += [Line2D([], [], color="#555", marker=m, ls="", label=k) for k, m in mk.items()]
        ax.legend(handles=hs, ncol=2, fontsize=6.5, loc="upper left")
        fig.tight_layout()
        return fig, f"每點＝一格（工作負載 × N × 規則 × 排隊順序），值是 3 次的中位數；顏色＝工作負載、形狀＝規則。共 {len(cells)} 格（污染的列不用）。"
    d.raw(add_fig(ctx, sec, "I1：多 session 還原的實測／模擬比", f_grid))

    def f_interf():
        p = os.path.join(D_, "interf_llama.json")
        j = read_json(p)
        ctx.src(sec, p, [j.get("run_id")])
        lab = {"rec|both": "重算（和 H2D 同時）", "h2d|both": "H2D（和重算同時）", "dec|both_d": "decode（和 H2D 同時）",
               "h2d|both_d": "H2D（和 decode 同時）"}
        plt = plt_setup(ctx)
        fig, ax = plt.subplots(figsize=(7.2, 2.4))
        keys = list(lab)
        for k2, g in enumerate(("idle", "busy2")):
            ys, er = [], [[], []]
            for k in keys:
                v = j["summary"].get(f"{g}|{k}")
                if v:
                    ys.append(v["k_median"])
                    q = v.get("k_q1q3") or [v["k_median"], v["k_median"]]
                    er[0].append(v["k_median"] - q[0])
                    er[1].append(q[1] - v["k_median"])
                else:
                    ys.append(float("nan"))
                    er[0].append(0)
                    er[1].append(0)
            ax.bar([i + (k2 - 0.5) * 0.38 for i in range(len(keys))], ys, width=0.36, color=C[k2], yerr=er, capsize=2,
                   label="GPU 閒" if g == "idle" else "GPU 忙（busy2）", error_kw=dict(lw=0.7))
        ax.axhline(1.0, color="#666", lw=0.8)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([lab[k] for k in keys], fontsize=7)
        ax.set_ylabel("同時跑 ÷ 單獨跑（中位數）")
        lo = min([x for x in [j["summary"][k]["k_median"] for k in j["summary"]] if x] + [0.9])
        ax.set_ylim(min(0.8, lo - 0.05), max(1.15, max(j["summary"][k]["k_median"] for k in j["summary"]) + 0.05))
        ax.legend()
        fig.tight_layout()
        cont = "（run 判定：污染）" if j.get("contaminated") else "（run 判定：乾淨）"
        return fig, f"誤差棒＝四分位距（decode 只有一個位置，沒有）。&gt;1＝同時跑變慢。{cont}"
    d.raw(add_fig(ctx, sec, "I1／I2：重算、搬運、decode 互相拖慢多少", f_interf))

    def f_pred():
        p = os.path.join(D_, "prediction", "prediction_llama.json")
        j = read_json(p)
        ctx.src(sec, p, [j.get("run_id")])
        cells = [c for c in j["cells"] if c["interference"] == "none" and c["N"] in (4, 8)]
        if not cells:
            raise NotMeasured("prediction_llama.json 沒有 N＝4／8、互不干擾的格子")
        wl = sorted({c["workload"] for c in cells})
        states = sorted({c["gpu_state"] for c in cells}, key=lambda x: (x != "idle", x))
        combos = [(n, g) for n in (4, 8) for g in states]
        plt = plt_setup(ctx)
        fig, axs = two_panels(plt, h=2.7, sharey=False)
        for ax, key, ttl in ((axs[0], "ttft_mean_gainA_perreq_vs_rr", "A：各請求各自最佳 vs 各自 Cake＋輪流"),
                             (axs[1], "ttft_mean_gainB_joint_vs_perreq", "B：聯合決定 vs 各請求各自最佳")):
            for k, (n, g) in enumerate(combos):
                ys = []
                for w in wl:
                    v = [c[key] for c in cells if c["workload"] == w and c["N"] == n and c["gpu_state"] == g]
                    ys.append(100 * v[0] if v else float("nan"))
                ax.bar([i + (k - (len(combos) - 1) / 2) * 0.8 / len(combos) for i in range(len(wl))], ys, width=0.76 / len(combos), color=C[k],
                       label=f"N＝{n}・GPU {'閒' if g == 'idle' else '忙（' + g + '）'}")
            ax.axhline(0, color="#666", lw=0.8)
            ax.set_xticks(range(len(wl)))
            ax.set_xticklabels(wl, fontsize=6.4, rotation=12)
            ax.set_title(ttl, fontsize=8.2)
            ax.set_ylabel("平均 TTFT 少多少（%；&gt;0＝較好）".replace("&gt;", ">"))
        axs[0].legend(fontsize=6.2)
        fig.tight_layout()
        return fig, "〔模擬〕事先預測（看 GPU 結果之前；互不干擾的模型）。出處：prediction/prediction_llama.json。"
    d.raw(add_fig(ctx, sec, "I1 事先預測〔模擬〕：好處從哪裡來", f_pred))

    def rule():
        def t():
            p = os.path.join(D_, "grid_llama.csv")
            rows = read_csv(p)
            cells = defaultdict(list)
            for r in rows:
                if r.get("contaminated") != "True":
                    cells[(r["workload"], r["N"], r["mode"], r["order"])].append(float(r["ratio_mean"]))
            med = [statistics.median(v) for v in cells.values()]
            if not med:
                raise NotMeasured("grid_llama.csv 沒有乾淨的列")
            ok = max(med) <= 1.10
            return (["格數", "最小", "中位數", "最大", "門檻", "判定"],
                    [[str(len(med)), fx(min(med), 3), fx(statistics.median(med), 3), fx(max(med), 3), "≤ 1.10",
                      "<span class='ok'>成立：模擬可信 → I1 收掉</span>" if ok else "<span class='nm'>不成立：模擬不可信</span>"]],
                    "實測／模擬＝平均 TTFT 的比，每格先取 3 次的中位數。")
        d.raw(add_table(ctx, sec, "I1 的事先寫死條件（由 grid_llama.csv 即時計算）", t))

        def t2():
            j = read_json(os.path.join(D_, "prediction", "prediction_llama.json"))
            rows = [[esc(k), fx(v["sim_restore"], 4), fx(v["meas_restore"], 4), fx(v["ratio"], 3)] for k, v in j["calibration_N1"].items()]
            return (["層", "模擬還原（s）", "第二階段實測（s）", "模擬／實測"], rows, "N＝1、GPU 閒的尺準：模擬對第二階段的單 session 實測。")
        d.raw(add_table(ctx, sec, "I1 模擬的尺準（N＝1）", t2))
    exp_tail(d, ctx, e, spec, rule)


# ---------------------------------------------------------------------------- 4.4 C1a
C1A_METHODS = [("SNAP", "寫入時"), ("EXPATT", "寫入時"), ("KNORM", "寫入時"), ("KEYDIFF", "寫入時"), ("H2O", "寫入時"),
               ("STREAM", "位置"), ("NEW", "位置"), ("RND0", "隨機"), ("QRY1", "第 1 輪問題"), ("QUEST_MEAN", "還原時摘要"),
               ("QUEST_MAX", "還原時摘要"), ("FULLK_MEAN", "上界（完整 K）")]


def sec_c1a(d: Doc, ctx: Ctx):
    e = EX("c1a")
    sec = e["sec"]
    D_ = ctx.R("c1a_predictive")
    spec = dict(
        q="卸載時就能算的 block 重要度（SNAP、KNORM、KEYDIFF 等）能不能猜中第 2 輪問題需要的 block？第 1 輪問題的注意力呢？"
          "等第 2 輪問題來了，用便宜的 block 摘要（Quest 式 K 的 min／max）挑呢？這是 C1b（選擇性還原）的閘門。",
        method="DS2 的兩輪題（seed 70000 起：T2–T4 各 45 題、T5 15 題）× 兩個模型；每題一次 32K prefill＋兩個問題的短 forward＋打分，不生成。"
               "指標：第 2 輪答案所在 block 落在前 16 名的比例（recall@16），對隨機期望 16/63 的差；以題目為單位 bootstrap 10,000 次的 95% CI。",
        rule="(讀檔)",
        limits=["只量「猜不猜得中」，不量答對率與時間（那是 C1b）。",
                "T5 是全域任務，沒有目標 block，表上不列。",
                "寫入時分數裡最好的那一種（C1b 用的 WRITE）是在同一批題目上挑的＝事後挑選。"])
    exp_head(d, ctx, e, spec)
    try:
        sj = read_json(os.path.join(D_, "c1a_summary.json"))
        rt = sj.get("rule_text_L1", {})
        spec["rule"] = "<br/>".join(f"<b>{esc(k)}</b>：{esc(v)}" for k, v in rt.items()) or "（summary 沒有 rule_text_L1）"
        ctx.src(sec, os.path.join(D_, "c1a_summary.json"), sj.get("run_ids", []))
    except NotMeasured as ex:
        sj = None
        spec["rule"] = ("L1 G2／G4 寫死：寫入時分數對第 2 輪答案 block 的召回≈隨機（CI 含隨機），而且 Quest 摘要也≈隨機 → "
                        f"C1b 不做（寫成負面結果）。<br/><span class='nm'>NOT_MEASURED</span>：{esc(ex)}")

    def f_recall():
        if sj is None:
            raise NotMeasured("c1a_summary.json 不存在")
        plt = plt_setup(ctx)
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        meths = [m for m, _ in C1A_METHODS if any(m in sj["models"].get(x, {}).get("T2-T4", {}).get("methods", {}) for x in MODELS)]
        for j, m in enumerate(MODELS):
            M = sj["models"].get(m, {}).get("T2-T4", {}).get("methods", {})
            for i, meth in enumerate(meths):
                v = (M.get(meth) or {}).get("k16", {}).get("recall_minus_random")
                if not v:
                    continue
                y = i + (j - 0.5) * 0.32
                ax.errorbar(100 * v["mean"], y, xerr=[[100 * (v["mean"] - v["lo"])], [100 * (v["hi"] - v["mean"])]], fmt="o",
                            color=MODEL_COLOR[m], ms=4, capsize=2, lw=1, label=MODEL_LABEL[m] if i == 0 else None)
        ax.axvline(0, color="#666", lw=0.8)
        ax.set_yticks(range(len(meths)))
        ax.set_yticklabels([f"{m}（{dict(C1A_METHODS)[m]}）" for m in meths], fontsize=6.8)
        ax.invert_yaxis()
        ax.set_xlabel("recall@16 − 隨機期望（百分點，95% CI）")
        ax.legend(loc="upper right")
        fig.tight_layout()
        rnd = sj.get("random_expectation", {}).get("k16")
        return fig, (f"T2–T4（每個模型 {', '.join(str(sj['models'].get(m, {}).get('T2-T4', {}).get('n_items', '—')) for m in MODELS)} 題）。"
                     f"隨機期望＝{fpct(rnd)}。0 的右邊＝比隨機好。")
    d.raw(add_fig(ctx, sec, "C1a：各分數對第 2 輪答案 block 的 recall@16（對隨機）", f_recall))

    def rule():
        def t():
            if sj is None:
                raise NotMeasured("c1a_summary.json 不存在")
            out = []
            for m in MODELS:
                comp = sj["models"].get(m, {}).get("T2-T4", {}).get("stop_rule_components", {})
                for k, v in comp.items():
                    out.append([MODEL_LABEL[m], esc(k), esc("；".join(f"{a}={b}" for a, b in v.get("write_time_recall_approx_random", {}).items())),
                                esc(str(v.get("quest_overlap_with_target_approx_random"))), esc(str(v.get("qry1_recall_approx_random"))),
                                f"G2（QUEST_MEAN）{v.get('G2_stop_with_QUEST_MEAN')}；G4 {v.get('G4_stop')}"])
            return (["模型", "k", "寫入時分數≈隨機？", "Quest 摘要≈隨機？", "QRY1≈隨機？", "停止條件成立？"], out,
                    "直接列 c1a_summary.json 的 stop_rule_components（True＝CI 含隨機）。停止條件全是 False → C1b 照計畫可以做。")
        d.raw(add_table(ctx, sec, "C1a 的停止條件成分（讀檔）", t))
    exp_tail(d, ctx, e, spec, rule)


# ---------------------------------------------------------------------------- 4.5 S1
S1_VARIANTS = [("raw", "原樣"), ("bytes2", "高低位元組分開"), ("bytes2hi", "只壓高位元組（Intel）"), ("planes", "指數／尾數分平面"),
               ("planes_e", "只壓指數")]
S1_TIERS = [("cpu", "CPU 層"), ("local", "本地 SSD"), ("b32", "限速 b32"), ("nfs_real", "NFS 實測"), ("nfs_sim", "NFS 級（限速）")]


def sec_s1(d: Doc, ctx: Ctx):
    e = EX("s1")
    sec = e["sec"]
    D_ = ctx.R("s1_lossless")
    spec = dict(
        q="BF16 KV 用 CPU 端的通用無損壓縮（zstd／zlib；含 Intel KV Shrink 的「只壓高位元組」）能壓多少？CPU 解壓多快？在哪一種連線上划算？",
        method="真實 KV dump（兩個模型 × T4／T2 × 32K；128K 的 T4 選層）；五種排列變體 × 多種 codec 設定；每一次壓縮都解回來逐位元比對。"
               "CPU 解壓速度量 1–16 個 thread；盈虧〔計算〕＝讀與解完全重疊（pipelined）時「讀＋解」的時間對不壓縮。",
        rule="L5c STOP-1：BF16 的無損比 &lt; 1.15（experiment_matrix 寫 &lt; 1.2）→ 無損這一軸拿掉；或 ≤16 thread 的 CPU 解壓都 &lt; 0.5 GiB/s"
             "而且沒有 GPU 解碼器 → 停。成功：≥ 1.3 而且解壓速度 &gt; NFS 頻寬 → 列為慢層的候選。",
        limits=["只有 CPU 解壓（GPU 端解碼器 hipCOMP／SplitZip 沒裝）。",
                "盈虧是計算，假設讀、解完全重疊（樂觀；L5c 同一個假設）；SSD／NFS 讀進來之後的 H2D 兩邊一樣、沒算。",
                "每份 dump 一題；「NFS 級」是限速器參數，另列實測 NFS 讀取速度。",
                "breakeven.csv 沒有 run_id 欄；它和 summary.json 同一次分析（run 見出處；experiment_log 的 S1 節）。"])
    exp_head(d, ctx, e, spec)

    def summ():
        p = os.path.join(D_, "summary.json")
        s = read_json(p)
        ctx.src(sec, p, [s.get("run_id")])
        return s

    def f_ratio():
        s = summ()
        dumps = list(s["dumps"])
        plt = plt_setup(ctx)
        fig, ax = plt.subplots(figsize=(7.2, 2.9))
        w = 0.8 / len(S1_VARIANTS)
        for k, (v, lab) in enumerate(S1_VARIANTS):
            ys = []
            for dn in dumps:
                b = s["dumps"][dn]["bf16"]
                r = [x["ratio"] for kk, x in b.items() if kk.startswith(f"all|{v}|") and "ratio" in x]
                ys.append(max(r) if r else float("nan"))
            ax.bar([i + (k - 2) * w for i in range(len(dumps))], ys, width=w * 0.9, color=C[k], label=lab)
        ent = [s["dumps"][dn]["bf16"].get("all|entropy", {}).get("ideal_ratio_planes") for dn in dumps]
        ax.scatter(range(len(dumps)), [x if x else float("nan") for x in ent], marker="_", s=260, color=INK, zorder=5,
                   label="order-0 熵的理想比（planes）")
        ax.axhline(1.15, color=C[7], ls="--", lw=1)
        ax.text(0.01, 1.153, "STOP-1 門檻 1.15", color=INK2, fontsize=6.8, ha="left", va="bottom", transform=ax.get_yaxis_transform())
        ax.set_xticks(range(len(dumps)))
        ax.set_xticklabels(dumps, fontsize=6.8)
        ax.set_ylim(1.0, None)
        ax.set_ylabel("無損壓縮比（對 BF16；各變體取最好的 codec）")
        ax.legend(ncol=3, fontsize=6.2, loc="upper left")
        fig.tight_layout()
        return fig, "dump 名＝模型_任務_block 數（64＝32K、252＝128K 選層）。"
    d.raw(add_fig(ctx, sec, "S1：BF16 KV 的無損壓縮比", f_ratio))

    def f_be():
        p = os.path.join(D_, "breakeven.csv")
        rows = read_csv(p)
        s = summ()
        ctx.src(sec, p, [s.get("run_id")])
        dumps = sorted({r["dump"] for r in rows}, key=lambda x: list(s["dumps"]).index(x) if x in s["dumps"] else 99)
        lossless = {v for v, _ in S1_VARIANTS}
        plt = plt_setup(ctx)
        fig, ax = plt.subplots(figsize=(7.2, 2.9))
        tiers = [t for t in S1_TIERS if any(r["tier"] == t[0] for r in rows)]
        w = 0.8 / max(1, len(dumps))
        for k, dn in enumerate(dumps):
            ys, ys4 = [], []
            for t, _ in tiers:
                g = [float(r["gain_pipelined"]) for r in rows if r["dump"] == dn and r["tier"] == t and r["method"].split("|")[0] in lossless
                     and r["gain_pipelined"] != ""]
                ys.append(1 - max(g) if g else float("nan"))
            ax.bar([i + (k - (len(dumps) - 1) / 2) * w for i in range(len(tiers))], ys, width=w * 0.9,
                   color=C[k % len(C)], label=dn)
        ax.axhline(1.0, color="#444", lw=0.9)
        ax.set_yscale("log")
        nice_log(ax)
        ax.set_xticks(range(len(tiers)))
        bw = {t: next((r["GiBps"] for r in rows if r["tier"] == t), "") for t, _ in tiers}
        ax.set_xticklabels([f"{lab}\n{fx(bw[t], 3)} GiB/s" for t, lab in tiers], fontsize=6.8)
        ax.set_ylabel("讀＋解的時間 ÷ 不壓縮（對數；&lt;1 才划算）".replace("&lt;", "<"))
        ax.legend(ncol=3, fontsize=6.0)
        fig.tight_layout()
        return fig, "〔計算〕每個（dump, 層）取最划算的無損變體（pipelined：max(讀壓縮後, CPU 解)）。&lt;1＝比不壓縮快。"
    d.raw(add_fig(ctx, sec, "S1：無損壓縮在各連線上的盈虧〔計算〕", f_be))

    def rule():
        def t():
            s = summ()
            out = []
            for m in MODELS:
                best, bk, bd = 0, None, None
                for dn, x in s["dumps"].items():
                    if not dn.startswith(m):
                        continue
                    for k, v in x["bf16"].items():
                        if k.startswith("all|") and "ratio" in v and v["ratio"] > best:
                            best, bk, bd = v["ratio"], k, dn
                sp = [v["decomp_MBps"] for dn, x in s["dumps"].items() if dn.startswith(m) for kk, v in x["speed"].items()]
                out.append([MODEL_LABEL[m], fx(best, 3), esc(f"{bd}｜{bk}"), fx(max(sp) / 1024 if sp else None, 2),
                            "<span class='ok'>不觸發（&gt;1.15）</span>" if best >= 1.15 else "<span class='nm'>觸發（停）</span>"])
            return (["模型", "最好的 BF16 無損比", "dump｜變體｜codec", "最快 CPU 解壓（GiB/s，以未壓縮位元組計）", "STOP-1"], out,
                    "解壓速度取所有變體、codec、thread 數的最大值（MB/s ÷ 1024）。")
        d.raw(add_table(ctx, sec, "S1 的事先寫死條件（由 summary.json 即時計算）", t))

        def t2():
            s = summ()
            out = []
            for dn, x in s["dumps"].items():
                for k in ("bytes2hi|zstd:1|t1", "bytes2hi|zstd:1|t4", "bytes2hi|zstd:1|t16", "planes|zstd:1|t1", "planes|zstd:1|t16"):
                    v = x["speed"].get(k)
                    if v:
                        out.append([esc(dn), esc(k), fx(v["ratio_framed"], 3), fx(v["decomp_MBps"], 0), fx(v["comp_MBps"], 0)])
            return (["dump", "變體｜codec｜thread", "壓縮比（4 MiB frame）", "解壓 MB/s", "壓縮 MB/s"], out,
                    "MB/s 以未壓縮位元組計；只列兩種代表變體。")
        d.raw(add_table(ctx, sec, "S1：CPU 解壓速度（代表變體）", t2))
    exp_tail(d, ctx, e, spec, rule)


# ---------------------------------------------------------------------------- 4.6 X1
X1_FAM = {"quant": "量化", "xlayer": "xKV（跨層低秩）", "lowrank": "每層低秩", "drop": "丟 block", "merge": "合併 token",
          "xshare": "跨層逐 token 共用", "combo": "組合（兩種以上）", "none": "對照"}
X1_STYLE = {"量化": (C[0], "o"), "xKV（跨層低秩）": (C[2], "s"), "每層低秩": (C[3], "v"), "丟 block": (C[1], "<"),
            "合併 token": (C[4], "h"), "跨層逐 token 共用": (C[5], "*"), "組合（兩種以上）": (C[7], "^"), "對照": ("#555555", "X")}


def symlog_axis(ax, linthresh=1.0):
    from matplotlib.ticker import FuncFormatter
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: minus(f"{v:g}")))


def sec_x1(d: Doc, ctx: Ctx):
    e = EX("x1")
    sec = e["sec"]
    D_ = ctx.R("x_stack")
    rep = None
    try:
        rep = read_json(os.path.join(D_, "x1_report.json"))
        ctx.src(sec, os.path.join(D_, "x1_report.json"), [rep.get("run_id")] + list(rep.get("src_run_ids", [])))
        rule_txt = "<br/>".join(f"<b>{esc(k)}</b>：{esc(rep['rule_text'][k])}" for k in ("S1-b", "S1-a", "X1-complete") if k in rep.get("rule_text", {}))
    except NotMeasured as ex:
        rule_txt = f"L5d §6.2 的 TF 版停止條件（S1-b）。<span class='nm'>NOT_MEASURED</span>：{esc(ex)}"
    spec = dict(
        q="同樣的位元組預算下，把丟 block、合併 token、每層低秩、跨層低秩（xKV）、量化疊在一起，會不會比最好的單一方法好？"
          "用 teacher forcing 大量篩選，挑出值得量生成答對率（X2）的組合。",
        method="每個模型 161 個配置（各家族自己的超參數、兩兩、三種以上、順序對照、天真疊加）＋S2 子集；32K、block 1–63（block 0 一律 BF16）；"
               "每題 prefill 一次，各配置量 teacher forcing 的 KL、Δgold_lp（標準答案 log 機率對 E0 的差）、first_ok；"
               "位元組照實算（每個配置 assert 實際 tensor 位元組＝預測），預算比較用「區內位元組 ÷ 區內 BF16」。",
        rule=rule_txt,
        limits=["teacher forcing、第 1 輪；答對率以 X2 的生成為準。",
                "xKV 的基底是每題用整段 KV 算的（壓縮在 prefill 之後做），壓縮本身的 SVD 成本沒有算進位元組或時間（成本見 X3 的寫入成本）。",
                "代理指標（重建誤差、attention 誤差）只當粗篩、不用來挑前緣。",
                "兩個模型的題數不同（見圖說的 n）。"])
    exp_head(d, ctx, e, spec)

    def f_pareto():
        plt = plt_setup(ctx)
        fig, axs = two_panels(plt, h=4.0, sharey=True)
        budgets = (rep or {}).get("budgets") or []
        any_ = False
        for ax, m in zip(axs, MODELS):
            p = os.path.join(D_, f"configs_{m}.csv")
            if not os.path.exists(p):
                ax.text(0.5, 0.5, "NOT_MEASURED", transform=ax.transAxes, ha="center", color=C[7])
                continue
            rows = read_csv(p)
            ctx.src(sec, p, rids_of_rows(rows))
            any_ = True
            n = max((int(r["n"]) for r in rows if r["n"]), default=0)
            seen = set()
            for r in rows:
                x, y = ffloat(r["ratio"]), ffloat(r["d_gold_lp"])
                if x is None or y is None:
                    continue
                fam = X1_FAM.get(r["family"], "組合（兩種以上）")
                col, mk = X1_STYLE.get(fam, ("#777", "."))
                front = r.get("on_front") == "1"
                ax.scatter(x, y, color=col, marker=mk, s=22 if front else 11, alpha=0.95 if front else 0.55,
                           edgecolor=INK if front else "none", linewidth=0.6, label=fam if fam not in seen else None, zorder=3 if front else 2)
                seen.add(fam)
            for b in budgets:
                ax.axvline(b, color="#aaaaaa", ls=":", lw=0.8)
            ax.set_xscale("log")
            nice_log(ax, "x")
            symlog_axis(ax, 1.0)
            ax.axhline(0, color="#666", lw=0.7)
            ax.set_title(f"{MODEL_LABEL[m]}（{n} 題）")
            ax.set_xlabel("位元組比（區內 ÷ BF16，對數）")
        if not any_:
            raise NotMeasured("configs_llama.csv／configs_qwen.csv 都不存在")
        axs[0].set_ylabel("平均 Δgold_lp 對 E0（symlog；越高越好）")
        h_, l_ = [], []
        for a_ in axs:
            for x_, y_ in zip(*a_.get_legend_handles_labels()):
                if y_ not in l_:
                    h_.append(x_)
                    l_.append(y_)
        fig.legend(h_, l_, loc="lower center", ncol=4, fontsize=6.6)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        return fig, "每點＝一個配置；粗邊大點＝前緣（on_front）；點虛線＝四個位元組預算。Δgold_lp 在 ±1 以內是線性刻度、以外是對數刻度。"
    d.raw(add_fig(ctx, sec, "X1：位元組比 vs teacher-forcing 品質（各家族）", f_pareto))

    def rule():
        def t():
            if rep is None:
                raise NotMeasured("x1_report.json 不存在")
            out = []
            for m, j in rep["judge"]["models"].items():
                for b, v in j["budgets"].items():
                    h = v.get("s1b_holds")
                    out.append([MODEL_LABEL.get(m, m), esc(b), f"{esc(v['best_single'])}（{fx(v['best_single_ratio'])}）", fx(-v["loss_single"], 3),
                                f"{esc(v['best_combo'])}（{fx(v['best_combo_ratio'])}）", fx(-v["loss_combo"], 3),
                                "—" if h is None else ("成立（組合沒有更好）" if h else "<b>不成立（組合明顯較好）</b>")])
                tm = j.get("tf_M1", {})
                out.append([MODEL_LABEL.get(m, m), "TF 版 M1", f"B*單一 {fx(tm.get('bstar_single'))}", "", f"B*組合 {fx(tm.get('bstar_combo'))}", "",
                            f"r＝{fx(tm.get('r'), 2)}（只報）"])
            g = rep["judge"]
            note = f"go＝{esc(g.get('go'))}；{esc(g.get('reason', ''))}"
            return (["模型", "預算", "最好的單一方法（位元組比）", "平均 Δgold_lp", "最好的組合（位元組比）", "平均 Δgold_lp", "S1-b"], out, note)
        d.raw(add_table(ctx, sec, "X1 的判定（讀 x1_report.json）", t))
    exp_tail(d, ctx, e, spec, rule)


# ---------------------------------------------------------------------------- 4.7 X2＋X2b
def x2_rows_32k(ctx: Ctx, sec: str):
    """32K 的每個配置：優先讀 v2（X2＋X2b 合併；每個配置用題數最多的那一組），沒有 v2 才讀第一版。"""
    D_ = ctx.R("x_stack")
    for sub in ("v2", ""):
        fs = [os.path.join(D_, sub, f) for f in ("x2_feasible.csv", "x2_unmapped.csv")]
        if all(os.path.exists(f) for f in fs):
            rows = []
            for f in fs:
                rr = read_csv(f)
                ctx.src(sec, f, rids_of_rows(rr) + rids_of_rows(rr, "src_run_ids"))
                rows += rr
            return rows, ("v2（X2＋X2b）" if sub else "第一版（X2）")
    raise NotMeasured("x_stack/v2/x2_feasible.csv＋x2_unmapped.csv 與第一版都不存在")


def plot_delta_panel(ax, pts, eps_pp, mlp_pp, title, clip=-25.0):
    """pts：[(multiple, mean_pp, lo_pp, hi_pp, family, pass_eps)]。"""
    seen, ncrash = set(), 0
    for x, y, lo, hi, fam, ok in pts:
        col, mk = FAMILY_STYLE.get(fam, ("#777", "."))
        if y < clip:
            ncrash += 1
            ax.scatter(x, clip + 0.6, marker="v", color=col, s=14, alpha=0.7)
            continue
        ax.errorbar(x, y, yerr=[[y - max(lo, clip)], [hi - y]], fmt=mk, color=col, ms=4.2, capsize=1.6, lw=0.8,
                    mfc=col if ok else "white", label=fam if fam not in seen else None)
        seen.add(fam)
    ax.axhline(-eps_pp, color=C[7], ls="--", lw=1)
    ax.axhline(0, color="#666", lw=0.7)
    if mlp_pp is not None:
        ax.axhline(-mlp_pp, color=C[6], ls=":", lw=1)
    ax.set_xscale("log")
    nice_log(ax, "x")
    ax.set_ylim(clip, 8)
    ax.set_title(title, fontsize=8.6)
    if ncrash:
        ax.text(0.02, 0.03, f"▼ {ncrash} 個配置 < {clip:.0f} 個百分點（崩）", transform=ax.transAxes, fontsize=6.3, color=INK2)


def sec_x2(d: Doc, ctx: Ctx):
    e = EX("x2")
    sec = e["sec"]
    D_ = ctx.R("x_stack")
    rep = rep2b = None
    try:
        rep = read_json(os.path.join(D_, "x2_report.json"))
        ctx.src(sec, os.path.join(D_, "x2_report.json"), [rep.get("run_id")] + list(rep.get("src_run_ids", [])))
    except NotMeasured:
        pass
    try:
        rep2b = read_json(os.path.join(D_, "v2", "x2b_report.json"))
        ctx.src(sec, os.path.join(D_, "v2", "x2b_report.json"), [rep2b.get("run_id")] + list(rep2b.get("src_run_ids_x2b", [])))
    except NotMeasured:
        pass
    rt = (rep or {}).get("rule_text", {})
    spec = dict(
        q="X1 挑出的組合，生成答對率（第 1 輪、greedy、第二階段同一個計分）對 E0 掉多少？在 ε 內最省的位元組比，組合對單一方法是多少（M1 的 r）？"
          "X2b 把 32K 的題數加到 288 題；128K 撐不撐得住？",
        method="32K：新題（X2：seed 62000 起 96 題；X2b：seed 63000 起 192 題；合併 288 題＝變體 D，主要讀）；128K：第二階段 44c 的題組"
               "（探索、不進判定）。逐題配對、以題目為單位 bootstrap 的 95% CI；MLPerf 式 99%／99.9% 門檻另報（點估計）。",
        rule=("<br/>".join(f"<b>{esc(k)}</b>：{esc(rt[k])}" for k in ("M1", "M2", "MLPerf") if k in rt)
              or "M1／M2（L5d §6.3；x_stack_howto §5.3）。<span class='nm'>NOT_MEASURED</span>：x2_report.json 不存在"),
        limits=["ε 規則要 CI 下界 ≥ −2 個百分點；題數少時連 INT4 KIVI 都可能過不了，「沒過 ε」可能只是統計力不夠，不一定是變差。",
                "128K 每個任務的題數少、CI 寬，只當探索。",
                "xKV 每題 SVD 的壓縮成本不在位元組裡（見 X3 寫入成本）；X2b 只量第一版有的配置（池子缺口在 x2b_report.json 標出）。"])
    exp_head(d, ctx, e, spec)

    def f_delta():
        rows32, src = x2_rows_32k(ctx, sec)
        plt = plt_setup(ctx)
        fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.6))
        for c, m in enumerate(MODELS):
            R = [r for r in rows32 if r["model"] == m]
            if not R:
                axs[0][c].text(0.5, 0.5, "NOT_MEASURED", transform=axs[0][c].transAxes, ha="center", color=C[7])
            else:
                eps = 100 * (ffloat(R[0].get("eps")) or 0.02)
                acc0 = statistics.mean(ffloat(r["acc_e0"]) for r in R if ffloat(r["acc_e0"]) is not None)
                pts = [(1 / float(r["ratio"]), 100 * float(r["acc_delta"]), 100 * float(r["acc_delta_lo"]), 100 * float(r["acc_delta_hi"]),
                        family_of(r["combo"]), r["feasible"] in ("1", "True")) for r in R if ffloat(r["ratio"])]
                ns = sorted({r.get("n_items") for r in R})
                plot_delta_panel(axs[0][c], pts, eps, acc0, f"{MODEL_LABEL[m]}・32K（題數 {'／'.join(ns)}；{src}）")
            if rep and "128K" in rep.get("by_ctx", {}) and m in rep["by_ctx"]["128K"]:
                B = rep["by_ctx"]["128K"][m]
                acc0 = B.get("acc_e0")
                pts = [(1 / v["ratio"], 100 * v["diff"]["mean"], 100 * v["diff"]["lo"], 100 * v["diff"]["hi"], family_of(s), v.get("eps_pass"))
                       for s, v in B["configs"].items() if v.get("ratio") and v.get("diff")]
                ns = sorted({str(v.get("n")) for v in B["configs"].values()})
                plot_delta_panel(axs[1][c], pts, 100 * rep.get("eps", 0.02), acc0 if acc0 else None,
                                 f"{MODEL_LABEL[m]}・128K（每配置 {'／'.join(ns)} 題；探索）")
            else:
                axs[1][c].text(0.5, 0.5, "NOT_MEASURED（128K）", transform=axs[1][c].transAxes, ha="center", color=C[7])
        for a in axs[1]:
            a.set_xlabel("壓縮倍數（BF16 ÷ 位元組，對數）")
        for a in (axs[0][0], axs[1][0]):
            a.set_ylabel("生成答對率 − E0（百分點，95% CI）")
        h, lab = [], []
        for a in axs.flat:
            hh, ll = a.get_legend_handles_labels()
            for x, y in zip(hh, ll):
                if y not in lab:
                    h.append(x)
                    lab.append(y)
        fig.legend(h, lab, loc="lower center", ncol=5, fontsize=6.3)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        return fig, ("紅虛線＝−ε（2 個百分點）；紫點線＝MLPerf 99% 門檻（E0 答對率的 1%）。實心＝過 ε（CI 下界 ≥ −ε）、空心＝沒過。"
                     "32K 讀 x_stack/v2/x2_feasible.csv＋x2_unmapped.csv（source 欄：D＝X2＋X2b 288 題、C＝只 X2b、A＝只 X2）；"
                     "128K 讀 x2_report.json 的 by_ctx。")
    d.raw(add_fig(ctx, sec, "X2＋X2b：答對率對 E0 的差 vs 壓縮倍數（32K、128K）", f_delta, width=490))

    def rule():
        def t():
            out = []
            if rep:
                for m, j in rep["judge"]["models"].items():
                    M1 = j.get("M1", {})
                    out.append(["X2（第一版，96 題）", MODEL_LABEL.get(m, m), fx(M1.get("bstar_single"), 4), fx(M1.get("bstar_combo"), 4),
                                fx(M1.get("r"), 3), fx(j.get("M1_mlperf0.99", {}).get("r"), 3), esc(rep["judge"].get("M1_verdict", ""))])
            if rep2b:
                for var, J in rep2b["judge"].items():
                    for m, j in J["models"].items():
                        M1 = j.get("M1", {})
                        n = J.get("n_items", {}).get(m, "")
                        out.append([f"X2b 變體 {esc(var)}（{n} 題）", MODEL_LABEL.get(m, m), fx(M1.get("bstar_single"), 4), fx(M1.get("bstar_combo"), 4),
                                    fx(M1.get("r"), 3), fx(j.get("M1_mlperf0.99", {}).get("r"), 3), esc(J.get("M1_verdict", ""))])
            if not out:
                raise NotMeasured("x2_report.json 與 v2/x2b_report.json 都不存在")
            return (["資料", "模型", "B*(單一)", "B*(組合)", "r", "r（MLPerf 99%）", "M1 判定（兩個模型合起來）"], out,
                    "B*＝ε 內最小的位元組比；r＝B*(組合)÷B*(單一)；任一模型 r &lt; 0.85 → GO（事先寫死）。∞＝沒有組合過 ε。"
                    "變體 A＝只用 X2（重現第一版）、B＝只用 X2、池子限制在 X2b 的配置、C＝只用 X2b、D＝X2＋X2b（主要讀）。")
        d.raw(add_table(ctx, sec, "X2 的 M1（讀 x2_report.json、v2/x2b_report.json）", t))

        def t2():
            if not rep2b:
                raise NotMeasured("v2/x2b_report.json 不存在")
            out = []
            for m, j in rep2b["judge"].get("D", {}).get("models", {}).items():
                for b, v in j.get("M2", {}).items():
                    if "diff_primary_minus_single" not in v:
                        out.append([MODEL_LABEL.get(m, m), esc(b), esc(v.get("primary")), esc(v.get("best_single")), "—", esc(v.get("status", "")), "—"])
                        continue
                    hm = v.get("holm", {})
                    out.append([MODEL_LABEL.get(m, m), esc(b), esc(v.get("primary")), esc(v.get("best_single")),
                                ci_pp(v["diff_primary_minus_single"]), "有" if v.get("room") else "沒有",
                                f"p {fx(hm.get('p'), 4)}／門檻 {fx(hm.get('threshold'), 4)}：{'拒絕' if hm.get('reject') else '不拒絕'}"])
            return (["模型", "預算", "主要組合", "最好的單一方法", "主要 − 單一（百分點）", "空間", "Holm"], out,
                    "變體 D（X2＋X2b）。空間＝E0 − 最好的單一方法 ≥ 2 個百分點。")
        d.raw(add_table(ctx, sec, "X2 的 M2（同位元組答對率；變體 D）", t2))
    exp_tail(d, ctx, e, spec, rule)


# ---------------------------------------------------------------------------- 4.8 xKV 校準基底
def sec_xlcal(d: Doc, ctx: Ctx):
    e = EX("xlcal")
    sec = e["sec"]
    D2, D3 = ctx.R("x_stack", "v2"), ctx.R("x_stack", "v3")
    r2 = r3 = None
    try:
        r2 = read_json(os.path.join(D2, "xlcal_report.json"))
        ctx.src(sec, os.path.join(D2, "xlcal_report.json"), [r2.get("run_id")] + list(r2.get("src_run_ids_gen", [])))
    except NotMeasured:
        pass
    try:
        r3 = read_json(os.path.join(D3, "xlcal3_report.json"))
        ctx.src(sec, os.path.join(D3, "xlcal3_report.json"), [r3.get("run_id")] + list(r3.get("src_run_ids_new", [])))
    except NotMeasured:
        pass
    rt = (r2 or {}).get("rule_text", {})
    rt3 = (r3 or {}).get("rule_text", {})
    rule = "<br/>".join(f"<b>{esc(k)}</b>：{esc(rt[k])}" for k in ("rule", "aggregate") if k in rt)
    if rt3:
        rule += "<br/>" + "<br/>".join(f"<b>38k {esc(k)}</b>：{esc(rt3[k])}" for k in ("aggregate", "variants", "caveat") if k in rt3)
    spec = dict(
        q="xKV 的基底能不能事先用校準題算好、所有請求共用（不用每題做數十秒的 SVD，每請求也不用帶基底），而答對率不輸每題 SVD？",
        method="校準題 8 題（seed 61000 起；和所有評測題不重疊），每 4 層一個窗口、K（逆 RoPE）／V 各一個基底；同一題、同一個行程配對比較"
               "「校準版 − 每題 SVD 版」的生成答對率。38c 用 X2 的 96 題（三對）；38k 加新題 192 題（只比有 INT4 係數的兩對）。",
        rule=rule or "（讀檔）xlcal_report.json 不存在：NOT_MEASURED",
        limits=["38k 的「只比兩對」是看過 38c 之後才定的：新的 192 題才是事先寫死、沒看過資料的獨立檢驗，合併 288 題只當參考。",
                "基底跨任務的退化沒有另外量；校準題含 T2 的重複句子。",
                "位元組攤提是〔算術〕：共用基底由 N 個請求分攤。"])
    exp_head(d, ctx, e, spec)

    def f_forest():
        items = []   # (label, mean, lo, hi, color)
        if r2:
            for m in MODELS:
                for pk, v in r2["models"].get(m, {}).get("gen", {}).get("pairs", {}).items():
                    d_ = v.get("cal_minus_ref")
                    if d_:
                        items.append((f"{MODEL_LABEL[m]}｜{pk.split(' vs ')[0].replace(':cal', '')}｜38c 96 題", d_, C[0]))
        if r3:
            for key, lab, col in (("new192", "38k 新 192 題（主要）", C[7]), ("merged288", "合併 288 題（參考）", C[6])):
                for m in MODELS:
                    for pk, v in r3["variants"].get(key, {}).get("models", {}).get(m, {}).get("pairs", {}).items():
                        d_ = v.get("cal_minus_ref")
                        if d_:
                            items.append((f"{MODEL_LABEL[m]}｜{pk.split(' vs ')[0].replace(':cal', '')}｜{lab}", d_, col))
        if not items:
            raise NotMeasured("v2/xlcal_report.json、v3/xlcal3_report.json 都沒有配對差")
        items.sort(key=lambda x: x[0])
        plt = plt_setup(ctx)
        fig, ax = plt.subplots(figsize=(7.2, 0.9 + 0.24 * len(items)))
        for i, (lab, d_, col) in enumerate(items):
            ax.errorbar(100 * d_["mean"], i, xerr=[[100 * (d_["mean"] - d_["lo"])], [100 * (d_["hi"] - d_["mean"])]], fmt="o", color=col,
                        ms=4, capsize=2, lw=1)
        ax.axvline(0, color="#666", lw=0.8)
        ax.axvline(-2, color=C[7], ls="--", lw=1)
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([x[0] for x in items], fontsize=6.4)
        ax.invert_yaxis()
        ax.set_xlabel("校準版 − 每題 SVD（生成答對率，百分點，95% CI）")
        fig.tight_layout()
        return fig, ("紅虛線＝−2 個百分點。判定（事先寫死）：CI 下界 ≥ −2 → 可以事先算（CAN_PRECOMPUTE）；CI 上界 &lt; −2 → 必須每題算；其他 → 分辨不了。"
                     "藍＝38c（X2 的 96 題）；紅＝38k 新 192 題；紫＝合併 288 題。")
    d.raw(add_fig(ctx, sec, "xKV 校準基底 − 每題 SVD 的配對差", f_forest))

    def rule_fn():
        def t():
            out = []
            if r2:
                for m in MODELS:
                    G = r2["models"].get(m, {}).get("gen", {})
                    for pk, v in G.get("pairs", {}).items():
                        out.append(["38c（96 題）", MODEL_LABEL[m], esc(pk), ci_pp(v.get("cal_minus_ref")), esc(v.get("verdict")),
                                    esc("／".join(str(v.get("amortized_ratio", {}).get(k, "—")) for k in ("1", "8", "64")))])
                    out.append(["38c（96 題）", MODEL_LABEL[m], "<b>模型判定</b>", "", f"<b>{esc(r2.get('verdict_by_model', {}).get(m))}</b>", ""])
            if r3:
                for key in ("new192", "merged288", "old96"):
                    V = r3["variants"].get(key, {})
                    for m in MODELS:
                        M = V.get("models", {}).get(m, {})
                        for pk, v in M.get("pairs", {}).items():
                            out.append([f"38k {esc(key)}（{M.get('n_items')} 題）", MODEL_LABEL[m], esc(pk), ci_pp(v.get("cal_minus_ref")),
                                        esc(v.get("verdict")), esc("／".join(str((v.get("amortized_ratio") or {}).get(k, "—")) for k in ("1", "8", "64")))])
                        if M:
                            out.append([f"38k {esc(key)}", MODEL_LABEL[m], "<b>模型判定</b>", "", f"<b>{esc(M.get('verdict'))}</b>", ""])
            if not out:
                raise NotMeasured("xlcal 報告都不存在")
            note = ""
            if r3:
                note = f"38k 主要判定（新 192 題）：<b>{esc(r3.get('verdict'))}</b>；合併 288 題（參考）：{esc(json.dumps(r3.get('verdict_merged288_reference'), ensure_ascii=False))}。"
            elif r2:
                note = f"38c 整體判定：<b>{esc(r2.get('verdict'))}</b>；38k（加題數）：<span class='nm'>NOT_MEASURED</span>（v3/xlcal3_report.json 不存在）。"
            return (["資料", "模型", "配對", "校準 − 每題 SVD（百分點）", "判定", "每請求位元組比（攤提 N＝1／8／64）"], out, note)
        d.raw(add_table(ctx, sec, "xKV 校準基底的判定", t))
    exp_tail(d, ctx, e, spec, rule_fn)


# ---------------------------------------------------------------------------- 4.9 X3
X3_TIERS = ["cpu_pinned", "vllm_cpu", "ssd_meas", "ssd_param", "b16", "b32", "b48", "nfs_meas", "nfs_param"]
X3_TIER_LAB = {"cpu_pinned": "CPU 層\n(pinned)", "vllm_cpu": "vLLM\nCPU 路徑", "ssd_meas": "本地 SSD\n(實測)", "ssd_param": "SSD\n(限速參數)",
               "b16": "限速\nb16", "b32": "限速\nb32", "b48": "限速\nb48", "nfs_meas": "NFS\n(實測)", "nfs_param": "NFS 級\n(限速)"}


def sec_x3(d: Doc, ctx: Ctx):
    e = EX("x3")
    sec = e["sec"]
    D1, D2 = ctx.R("x3_time"), ctx.R("x3_time", "v2")
    spec = dict(
        q="壓縮組合還原時要多花解碼時間；在不同連線速度（CPU 層、SSD、限速層、NFS 級）和 GPU 忙碌程度下，哪一種格式還原最快？"
          "「GPU 忙不忙」這個輸入會不會改變最佳選擇？xKV 的寫入（壓縮）成本多大？模擬的預測準不準（validate）？",
        method="micro（GPU，量時間）：每種解碼段在 duty 0／0.4／0.7／0.9 的 H2D 與反量化／重建時間；map（CPU，〔模擬〕）：把零件放進第二階段還原器的"
               "兩條線模型，算 32K 還原時間 min(Cake, 全部載入)；三種動作集（L 只准無損、Q 允許第二階段過 ε 的量化、X 再加 X2 判可行的組合），"
               "報 A（每格最好）對 D（不壓縮）、F（每模型一個固定動作）、S（每層一個動作）。10/21 補 xKV＋量化的解碼段與寫入成本，出 map v2；"
               "v2 判要驗證的格子做端到端實測（validate），和預測比。",
        rule="x3_time_howto §4（看到數字前寫死）：<b>STOP-0</b>（模式 Q）沒有任何一格別的格式比主格式快 ≥5% → 有損自適應 NO-GO；"
             "<b>STOP-2a</b> 每格 A 對 S 都 &lt;5% → GPU 忙這個軸沒有價值；<b>STOP-2b</b> A 對 F 幾何平均 &lt;5% → NO-GO；"
             "<b>GO_N5</b> A 對 S 在 ≥25% 的格子 &gt;15% 而且 A 對 F &gt;15% → 做 validate；<b>STOP-2c</b> 介於中間 → 停下來問使用者；"
             "<b>T-d</b> 有格子最佳動作隨 duty 換人而且比 S 快 ≥15%；<b>E-L3</b> 低秩重建在忙時的敏感度；"
             "<b>STOP-3</b>（validate）≥2 格實測和預測差 &gt;10%，或實測的 A 對 F &lt;5% → 預測圖不可信或沒有好處 → 停。",
        limits=["map 是模擬（零件實測、排程是模型）；慢層是限速器，不是真的讀 SSD／NFS；第二階段驗過這套模擬在「慢層＋GPU 忙」會偏快（docs/precision_20261013/09_sweep_plan.md §5；map 在這些格子的 flags 標出）。",
                "一次拆的成本由 32 塊線性外推到 64 塊；無損的讀／解／H2D 完全重疊是樂觀假設。",
                "寫入成本只含 GPU 上的 SVD＋係數量化，不含 D2H 與寫入儲存層；而且是每題一次的成本，不能直接和每次還原的時間相加。",
                "validate 只驗 map 挑出的格子；低秩／xKV 等只驗「全部載入」那一條線。"])
    exp_head(d, ctx, e, spec)

    def f_restore():
        bd = big_dir(os.path.join(D2, "BIG_FILES.md"), "x3_full")
        p = os.path.join(bd, "map_cells.csv")
        rows = read_csv(p)
        ctx.src(sec, p, rids_of_rows(rows) + rids_of_rows(rows, "micro_run_id"))
        dec = read_csv(os.path.join(D2, "map_decision.csv"))
        ctx.src(sec, os.path.join(D2, "map_decision.csv"), rids_of_rows(dec) + rids_of_rows(dec, "micro_run_id"))
        T = {}
        for r in rows:
            if r.get("grid") == "named" and ffloat(r.get("T_s")) is not None:
                T[(r["model"], r["tier"], float(r["duty"]), r["action"])] = float(r["T_s"])
        A = {(r["model"], r["tier"], float(r["duty"])): (r["A"], ffloat(r["T_A"])) for r in dec if r["mode"] == "X"}
        plt = plt_setup(ctx)
        fig, axs = plt.subplots(2, 1, figsize=(7.2, 6.0))
        duties = (0.0, 0.9)
        for ax, m in zip(axs, MODELS):
            tiers = [t for t in X3_TIERS if any(k[0] == m and k[1] == t for k in T)]
            series = [("BF16（不壓縮）", C[0], lambda t, du: T.get((m, t, du, "raw"))),
                      ("INT4 KIVI（一次拆）", C[2], lambda t, du: T.get((m, t, du, "int4_kivi@oneshot"))),
                      ("模式 X 的最好動作（含疊加組合）", C[7], lambda t, du: (A.get((m, t, du)) or (None, None))[1])]
            k = 0
            w = 0.8 / (len(series) * len(duties))
            for lab, col, fn in series:
                for du in duties:
                    ys = [fn(t, du) for t in tiers]
                    ax.bar([i + (k - (len(series) * len(duties) - 1) / 2) * w for i in range(len(tiers))],
                           [1000 * y if y else float("nan") for y in ys], width=w * 0.92, color=col, alpha=1.0 if du == 0 else 0.55,
                           hatch=None if du == 0 else "////", edgecolor=col, linewidth=0.4,
                           label=f"{lab}・{'GPU 閒' if du == 0 else 'GPU 忙 duty 0.9'}")
                    k += 1
            ax.set_yscale("log")
            nice_log(ax)
            ax.set_xticks(range(len(tiers)))
            ax.set_xticklabels([X3_TIER_LAB.get(t, t) for t in tiers], fontsize=6.6)
            ax.set_ylabel("32K 還原時間（ms，對數）")
            ax.set_title(MODEL_LABEL[m])
        h_, l_ = axs[0].get_legend_handles_labels()
        fig.legend(h_, l_, loc="upper center", ncol=3, fontsize=6.2)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        return fig, ("〔模擬：零件實測〕map v2（run 見出處）。模式 X 的最好動作名稱見下表。限速層與「NFS 級」是限速器；"
                     "vLLM CPU 路徑只有全部載入。")
    d.raw(add_fig(ctx, sec, "X3：各層連線下的 32K 還原時間（BF16、INT4、最好的組合；閒／忙）", f_restore))

    def t_best():
        dec = read_csv(os.path.join(D2, "map_decision.csv"))
        out = []
        for m in MODELS:
            for t in X3_TIERS:
                rr = {float(r["duty"]): r for r in dec if r["mode"] == "X" and r["model"] == m and r["tier"] == t}
                if not rr:
                    continue
                cells = []
                for du in (0.0, 0.9):
                    r = rr.get(du)
                    cells.append(f"{esc(r['A'])}（{fx(1000 * float(r['T_A']), 1)} ms）" if r else "NOT_MEASURED")
                r0 = rr.get(0.0) or next(iter(rr.values()))
                out.append([MODEL_LABEL[m], esc(t), cells[0], cells[1], esc(r0["F"]), fpct(ffloat(r0["gain_vs_F"])), fpct(ffloat(r0["gain_vs_D"]))])
        return (["模型", "層", "最好動作（GPU 閒）", "最好動作（duty 0.9）", "F（固定動作）", "閒時 A 對 F", "閒時 A 對 D"], out,
                "讀 x3_time/v2/map_decision.csv 的模式 X。")
    d.raw(add_table(ctx, sec, "X3 模式 X：每層的最好動作（map v2）", t_best))

    def t_wcost():
        p = os.path.join(D2, "write_cost.csv")
        rows = read_csv(p)
        ctx.src(sec, p, rids_of_rows(rows) + rids_of_rows(rows, "micro_run_id"))
        out = [[MODEL_LABEL.get(r["model"], r["model"]), esc(r["segment"]), fx(r["duty"], 1), fx(r["t_svd_k_s"], 2), fx(r["t_svd_v_s"], 2),
                fx(ffloat(r["t_quant_s"]) * 1000 if ffloat(r["t_quant_s"]) is not None else None, 2), fx(r["t_total_s"], 2),
                fx(r["recompute_region_idle_s"], 2), fx(r["total_over_recompute"], 2)] for r in rows]
        return (["模型", "段", "duty", "SVD K（s）", "SVD V（s）", "係數量化（ms）", "合計（s）", "同區閒時重算（s）", "合計 ÷ 重算"], out,
                "xKV 每題一次的壓縮（寫入）成本：載入區 T＝16,384 token、全部窗口、svd_lowrank（niter 4、多取 16）。"
                "「÷ 重算」只是比例尺，不是判定。")
    d.raw(add_table(ctx, sec, "X3：xKV 的寫入（壓縮）成本", t_wcost))

    def rule_fn():
        def t():
            out = []
            for tag, D_ in (("map v1", D1), ("map v2", D2)):
                p = os.path.join(D_, "map_summary.json")
                if not os.path.exists(p):
                    continue
                s = read_json(p)
                ctx.src(sec, p, [s.get("run_id")])
                V = s.get("verdicts", {})
                for mode in ("L", "Q", "X"):
                    v = V.get(mode, {})
                    s2 = v.get("stop2", {})
                    extra = ""
                    if mode == "Q" and "stop0" in v:
                        extra = f"STOP-0 {'成立' if v['stop0'].get('holds') else '不成立'}；"
                    if mode == "L" and "stop1" in v:
                        extra = f"{esc(v['stop1'].get('verdict'))}；"
                    agg = s.get("aggregates", {}).get(mode, {})
                    out.append([tag, mode, esc(s2.get("code")), fpct(s2.get("gain_AF_gm")), fpct(s2.get("frac_AS_gt15")),
                                fpct(agg.get("vs_D", {}).get("gm_gain")), extra + esc(s2.get("verdict", ""))])
                out.append([tag, "validate", "", "", "", "", esc(f"worth_validate＝{s.get('worth_validate')}；{s.get('validate_reason', '')}")])
            if not out:
                raise NotMeasured("x3_time/map_summary.json 與 v2/map_summary.json 都不存在")
            return (["map", "模式", "代號", "A 對 F（幾何平均）", "A 對 S &gt;15% 的格子", "A 對 D", "判定"], out, "讀 map_summary.json 的 verdicts 與 aggregates。")
        d.raw(add_table(ctx, sec, "X3 的判定（map v1、v2）", t))

        def t_el3():
            s = read_json(os.path.join(D2, "map_summary.json"))
            out = [[esc(k), fx(v.get("slow_lr"), 2) + "×", fx(v.get("slow_int4"), 2) + "×", fx(v.get("busy_recon_over_h2d"), 2), esc(v.get("verdict"))]
                   for k, v in s.get("el3", {}).items()]
            return (["模型｜段", "低秩重建 忙／閒", "INT4 反量化 忙／閒", "忙時重建 ÷ 同位元組 H2D", "E-L3 判定"], out, "duty 0.7（map v2）。")
        d.raw(add_table(ctx, sec, "X3 E-L3：GPU 忙時低秩重建的成本", t_el3))

        def f_val():
            found = []
            for tag, D_ in (("v2", os.path.join(D2, "validate")), ("v1", os.path.join(D1, "validate"))):
                ps = os.path.join(D_, "validate_summary.json")
                if os.path.exists(ps):
                    found.append((tag, D_, read_json(ps)))
            if not found:
                raise NotMeasured("x3_time/v2/validate/validate_summary.json 不存在（38j_x3_validate2 還沒跑完）")
            rows_all = []
            for tag, D_, s in found:
                ctx.src(sec, os.path.join(D_, "validate_summary.json"), [s.get("run_id")])
                pc = os.path.join(D_, "validate_cells.csv")
                if os.path.exists(pc):
                    rr = read_csv(pc)
                    ctx.src(sec, pc, rids_of_rows(rr))
                    rows_all += [(tag, r) for r in rr if ffloat(r.get("pred_s")) and ffloat(r.get("meas_s"))]
            if not rows_all:
                msg = "；".join(f"{tag}：{s.get('reason') or s.get('verdict')}" for tag, _, s in found)
                raise NotMeasured(f"validate 沒有可比的列（{msg}）")
            plt = plt_setup(ctx)
            fig, ax = plt.subplots(figsize=(5.6, 4.2))
            for i, m in enumerate(MODELS):
                xs = [float(r["pred_s"]) * 1000 for _, r in rows_all if r["model"] == m]
                ys = [float(r["meas_s"]) * 1000 for _, r in rows_all if r["model"] == m]
                cont = [str(r.get("contaminated")) not in ("False", "") for _, r in rows_all if r["model"] == m]
                ax.scatter(xs, ys, color=MODEL_COLOR[m], s=[30 if not c else 12 for c in cont], label=MODEL_LABEL[m],
                           facecolors=[MODEL_COLOR[m] if not c else "white" for c in cont], edgecolors=MODEL_COLOR[m])
            lo = min(min(float(r["pred_s"]), float(r["meas_s"])) for _, r in rows_all) * 1000 * 0.7
            hi = max(max(float(r["pred_s"]), float(r["meas_s"])) for _, r in rows_all) * 1000 * 1.4
            ax.plot([lo, hi], [lo, hi], color="#555", lw=0.8)
            ax.fill_between([lo, hi], [lo * 0.9, hi * 0.9], [lo * 1.1, hi * 1.1], color="#bbbbbb", alpha=0.35, label="±10%")
            ax.set_xscale("log")
            nice_log(ax, "x")
            ax.set_yscale("log")
            nice_log(ax, "y")
            ax.set_xlabel("預測（ms）")
            ax.set_ylabel("實測（ms）")
            ax.legend()
            fig.tight_layout()
            sm = "；".join(f"{tag}：{s.get('verdict')}（不準的格子 {len(s.get('bad_cells') or [])}、實測 A 對 F {fpct(s.get('gain_AF_measured'))}、污染 {s.get('n_contaminated')} 列）"
                          for tag, _, s in found)
            return fig, f"空心＝被污染的列（STOP-3 不算）。{esc(sm)}"
        d.raw(add_fig(ctx, sec, "X3 validate：還原時間的實測 vs 預測", f_val, width=400))
    exp_tail(d, ctx, e, spec, rule_fn)


# ---------------------------------------------------------------------------- 4.10 C1b
C1B_METHODS = ["QUEST_MEAN", "QUEST_BLIND", "FULLK_MEAN", "WRITE", "QRY1", "POS", "RND"]


def sec_c1b(d: Doc, ctx: Ctx):
    e = EX("c1b")
    sec = e["sec"]
    D_ = ctx.R("c1_selective_restore")
    sj = None
    try:
        sj = read_json(os.path.join(D_, "c1b_summary.json"))
        ctx.src(sec, os.path.join(D_, "c1b_summary.json"), sj.get("run_ids", []))
    except NotMeasured:
        pass
    vt = (sj or {}).get("rule", {}).get("text", {})
    spec = dict(
        q="使用者帶著第 2 輪問題回來時，只搬挑中的 k%（10／25／50%）block、其餘留在下層，答對率掉多少？"
          "「留在下層、需要時補載」(b) 有沒有比「直接丟掉」(a) 好？",
        method="同 C1a 的兩輪題（T2–T4 各 45 題、T5 15 題，32K）。選法：QUEST_MEAN（設計卡指定；q 看過完整 context，偏樂觀）、QUEST_BLIND"
               "（可部署版：只用 block 0＋tail 的 q）、FULLK（上界）、寫入時最好的分數、第 1 輪 QRY、位置、隨機；(a) 不補載＝丟、(b) 每 4 個生成 "
               "token 用當下的 q 重打分、前 m 名沒載入的補載。對完整還原（FULL，同一條遮罩路徑）逐題配對。還原時間是〔計算〕；"
               "Quest 的額外時間由 qtime 在閒的 GPU 上實測。",
        rule=("c1b_howto §3（看到數字前寫死）：範圍 T2–T4、每個模型 ≥48 題。A＝(a) k＝25% 在 ε 內而且搬運 ≤30%；B＝(b) k＝25% 同上（含補載）；"
              "R＝(b)−(a) 的 CI 下界 &gt; 0；STOP＝k＝50% 時兩種 query-aware 選法 × (a)(b) 四格平均都掉超過 ε。類別依序："
              "INSUFFICIENT_N → STOP → SUCCESS（B 且 R）→ SUCCESS_NO_REFILL_NEEDED（A）→ PARTIAL（B）→ INCONCLUSIVE。"
              "設計卡的判定＝QUEST_MEAN 的類別；可部署版另列。"
              + ("<br/>" + "<br/>".join(f"<b>{esc(k)}</b>：{esc(v)}" for k, v in vt.items()) if vt else "")),
        limits=["首 token 時間不是端到端實測（還原時間計算＋Quest 額外時間實測）。",
                "128K 與 SCBench 子集沒做；T5 是單輪、只另報。",
                "寫入時最好的分數（WRITE）是在同一批題目上挑的；混合還原（前段 Cake 重算＋後段只載選中的）的品質沒量。",
                "品質 run 每張卡有多個行程，t_gen_s 全部作廢（分數照用）。"])
    exp_head(d, ctx, e, spec)

    def f_cells():
        p = os.path.join(D_, "c1b_cells.csv")
        rows = read_csv(p)
        ctx.src(sec, p, rids_of_rows(rows))
        plt = plt_setup(ctx)
        fig, axs = two_panels(plt, h=3.4, sharey=True)
        for ax, m in zip(axs, MODELS):
            R = [r for r in rows if r["model"] == m and r["group"] == "T2-T4" and r["method"] not in ("FULL", "E0_SDPA")]
            if not R:
                ax.text(0.5, 0.5, "NOT_MEASURED", transform=ax.transAxes, ha="center", color=C[7])
                continue
            n = max(int(r["n"]) for r in R)
            for i, meth in enumerate(C1B_METHODS):
                for r in R:
                    if r["method"] != meth or ffloat(r["diff_mean"]) is None:
                        continue
                    x = ffloat(r["ratio_moved"])
                    if x is None:
                        x = (1 + round(int(r["k_pct"]) / 100 * 63)) / 64     # oracle：只畫初始載入比（位置用）
                    y = 100 * float(r["diff_mean"])
                    lo, hi = 100 * float(r["diff_lo"]), 100 * float(r["diff_hi"])
                    filled = r.get("variant") == "b"
                    ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="o" if filled else "s", color=C[i], ms=4, capsize=1.5, lw=0.7,
                                mfc=C[i] if filled else "white")
                    ax.annotate(f"{r['k_pct']}", (x, y), fontsize=5.2, color=INK2, xytext=(2, 2), textcoords="offset points")
            ax.axhline(-2, color=C[7], ls="--", lw=1)
            ax.axhline(0, color="#666", lw=0.7)
            ax.axvline(0.30, color="#999", ls=":", lw=1)
            ax.set_title(f"{MODEL_LABEL[m]}（T2–T4，{n} 題）")
            ax.set_xlabel("搬運比（初始＋tail＋摘要＋補載）÷ 完整還原")
        axs[0].set_ylabel("答對率 − 完整還原（百分點，95% CI）")
        from matplotlib.lines import Line2D
        hs = [Line2D([], [], color=C[i], marker="o", ls="-", lw=0.8, label=m_) for i, m_ in enumerate(C1B_METHODS)]
        hs += [Line2D([], [], color="#555", marker="s", mfc="white", ls="", label="(a) 不補載（丟）"),
               Line2D([], [], color="#555", marker="o", ls="", label="(b) 補載")]
        fig.legend(handles=hs, loc="lower center", ncol=5, fontsize=6.4)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        return fig, "實心圓＝(b) 補載、空心方塊＝(a) 不補載（丟）；點旁數字＝k（%）。紅虛線＝−ε；灰點線＝搬運 30%。"
    d.raw(add_fig(ctx, sec, "C1b：答對率 vs 搬運比例（各選法、k、補載與否）", f_cells))

    def t_qtime():
        out = []
        for m in MODELS:
            p = os.path.join(D_, f"qtime_{m}.csv")
            if not os.path.exists(p):
                continue
            rows = read_csv(p)
            ctx.src(sec, p, rids_of_rows(rows))
            for r in rows:
                out.append([MODEL_LABEL[m], esc(r["op"]), esc(r["n_tokens"]), fx(r["t_med_ms"], 3), f"[{fx(r['t_q1_ms'], 3)}, {fx(r['t_q3_ms'], 3)}]",
                            fx(r["GiBps"], 2) if r.get("GiBps") else "—", esc(r["gpu_verdict"]), esc(r["run_id"])])
        if not out:
            raise NotMeasured("qtime_llama.csv／qtime_qwen.csv 不存在")
        return (["模型", "操作", "token", "中位數（ms）", "四分位", "GiB/s", "GPU", "run_id"], out,
                "qtime：在閒的 GPU 上量 Quest 打分、可部署版 q 的 forward、補載檢查等的額外時間（GPU 不乾淨的不收）。")
    d.raw(add_table(ctx, sec, "C1b：Quest 額外時間（qtime 實測）", t_qtime))

    def t_rt():
        if not sj:
            raise NotMeasured("c1b_summary.json 不存在（38dz_c1b_report 還沒跑）")
        out = []
        for m in MODELS:
            RT = sj.get("models", {}).get(m, {}).get("restore_time", {})
            for r in RT.get("rows", []):
                if r.get("k_pct") != 25:
                    continue
                out.append([MODEL_LABEL[m], esc(f"{r['layout']}／{r['tier']}"), fx(r["GiBps"], 2), fx(r["t_sel_s"] * 1000, 1), fx(r["t_full_load_s"] * 1000, 1),
                            fx(r["t_cake_s"] * 1000, 1) if r.get("t_cake_s") is not None else "—", fx(r["ratio_vs_full"], 3),
                            fx(r["ratio_vs_cake"], 3) if r.get("ratio_vs_cake") is not None else "—"])
        return (["模型", "版面／層", "GiB/s", "只搬 25%（ms）", "全部搬（ms）", "Cake（ms）", "對全部搬", "對 Cake"], out,
                "〔計算〕第二階段常數＋I4 每片成本；只列 k＝25%。")
    d.raw(add_table(ctx, sec, "C1b：k＝25% 的還原時間〔計算〕", t_rt))

    def rule_fn():
        def t():
            if not sj:
                raise NotMeasured("c1b_summary.json 不存在（38da／38db 還在跑或沒跑完）")
            out = []
            for m in MODELS:
                J = sj.get("models", {}).get(m, {}).get("T2-T4", {}).get("judge")
                if not J:
                    continue
                for meth, v in J.get("by_method", {}).items():
                    out.append([MODEL_LABEL[m], esc(meth), f"{ci_pp(v['a']['diff'])}；搬運 {fx(v['a']['ratio_moved'], 3)}",
                                f"{ci_pp(v['b']['diff'])}；搬運 {fx(v['b']['ratio_moved'], 3)}", ci_pp(v.get("b_minus_a")),
                                f"A {v['A']}／B {v['B']}／R {v['R']}", f"<b>{esc(v['verdict'])}</b>"])
                out.append([MODEL_LABEL[m], "<b>設計卡／可部署版</b>", "", "", f"STOP {esc(J.get('STOP'))}", f"n＝{J.get('n_items')}",
                            f"<b>{esc(J.get('verdict_design_card'))}</b>／<b>{esc(J.get('verdict_deployable'))}</b>"])
            return (["模型", "選法", "(a) k＝25%：對 FULL；搬運比", "(b) k＝25%：對 FULL；搬運比", "(b) − (a)", "成分", "類別"], out,
                    "讀 c1b_summary.json 的 judge（T2–T4）。")
        d.raw(add_table(ctx, sec, "C1b 的判定", t))
    exp_tail(d, ctx, e, spec, rule_fn)


# ---------------------------------------------------------------------------- 4.11 X4
X4_BITS = {"bf16": (C[0], "只降秩（BF16 係數）"), "int8": (C[2], "INT8 係數"), "int4": (C[7], "INT4 係數"), "k4v2": (C[1], "K4V2"),
           "k2v4": (C[6], "K2V4"), "int2": (C[3], "INT2 係數")}


def sec_x4(d: Doc, ctx: Ctx):
    e = EX("x4")
    sec = e["sec"]
    D_ = ctx.R("x_stack", "x4")
    rep = None
    try:
        rep = read_json(os.path.join(D_, "x4_report.json"))
        ctx.src(sec, os.path.join(D_, "x4_report.json"), [rep.get("run_id")] + list(rep.get("src_run_ids", [])))
    except NotMeasured:
        pass
    rt = (rep or {}).get("rule_text") or {}
    plan = ctx.D("x4_limit_plan.md")
    if rt:
        rule = "<br/>".join(f"<b>{esc(k)}</b>：{esc(v)}" for k, v in rt.items() if k in ("Q-A", "Q-B", "limit", "stack", "long", "multiturn", "both_refs"))
    else:
        try:
            rule = md(md_section(plan, r"^## 3\. "))
            ctx.src(sec, plan)
        except NotMeasured as ex:
            rule = f"<span class='nm'>NOT_MEASURED</span>：{esc(ex)}"
    spec = dict(
        q="把「多層低秩（xKV）＋量化」推到極限：Q-A 哪幾種方法疊在一起最划算（無損、輕度丟、係數位元、窗口 W、K／V 秩分配）？"
          "Q-B 在固定的目標倍數（8–32 倍）下最好的配比是什麼、極限在哪個倍數掉出 ε？長上下文（128K、Qwen 256K）與多輪（第 2 輪換問題）撐不撐得住？",
        method="X4a：每個模型約 70 個配置 × 96 題的 teacher-forcing 篩選（共用 SVD）；X4b：每個目標倍數帶最好的配置量生成答對率（192 題）；"
               "X4c：前緣 3 個配置在 128K（Qwen 另有 256K 切片）；X4d：前緣 2 個配置在 C1a 的兩輪題；無損：把前緣配置實際存下來的係數與基底用 zstd 壓（CPU）。"
               "設計見 docs/stage3_20261019/x4_limit_plan.md（看到數字前寫死）。",
        rule=rule,
        limits=["共用 SVD（每題每 K／V、W 一次，取前 r 欄；誤差 ≤ X1／X2 的每配置 SVD）；輕度丟一律 xl+drop。",
                "W 大（Llama 8、Qwen 7）因時間縮掉，W 只掃 2 與 4；X4d 前緣只放 2 個。",
                "第 2 輪的歷史固定成第 1 輪的標準答案（第二階段同一作法），不是模型自己的答案。",
                "256K 是第一次用 504 block 的題組；128K／256K 的位元組比用該長度的區實算（基底攤在更多 token 上）。"])
    exp_head(d, ctx, e, spec)

    def f_tf():
        plt = plt_setup(ctx)
        fig, axs = two_panels(plt, h=3.4, sharey=True)
        anyd = False
        for ax, m in zip(axs, MODELS):
            p = os.path.join(D_, f"x4a_configs_{m}.csv")
            if not os.path.exists(p):
                ax.text(0.5, 0.5, "NOT_MEASURED", transform=ax.transAxes, ha="center", color=C[7])
                continue
            rows = read_csv(p)
            ctx.src(sec, p, rids_of_rows(rows))
            anyd = True
            seen = set()
            for r in rows:
                x, y = ffloat(r["multiple"]), ffloat(r["d_gold_lp"])
                if x is None or y is None:
                    continue
                col, lab = X4_BITS.get(r.get("bits"), ("#777", r.get("bits") or "其他"))
                mk = {"2": "s", "4": "o"}.get(str(r.get("W")), "^")
                hollow = (ffloat(r.get("drop_frac")) or 0) > 0
                ax.scatter(x, y, color=col, marker=mk, s=16, facecolors="white" if hollow else col, edgecolors=col, linewidth=0.8,
                           label=lab if lab not in seen else None)
                seen.add(lab)
            for T in (rep or {}).get("targets", [8, 12, 16, 20, 24, 32]):
                ax.axvline(T, color="#cccccc", ls=":", lw=0.8)
            ax.set_xscale("log")
            nice_log(ax, "x")
            symlog_axis(ax, 1.0)
            ax.set_title(MODEL_LABEL[m])
            ax.set_xlabel("壓縮倍數（對數）")
        if not anyd:
            raise NotMeasured("x_stack/x4/x4a_configs_*.csv 不存在（X4a 還沒跑完或報告還沒出）")
        axs[0].set_ylabel("平均 Δgold_lp 對 E0（TF；symlog）")
        axs[1].legend(fontsize=6.2)
        fig.tight_layout()
        return fig, "顏色＝係數位元；形狀＝窗口 W（方塊 2、圓 4）；空心＝加了輕度丟；點虛線＝目標倍數。"
    d.raw(add_fig(ctx, sec, "X4a：倍數 vs teacher-forcing 品質", f_tf))

    def f_gen():
        if not rep:
            raise NotMeasured("x_stack/x4/x4_report.json 不存在（X4b 之後的報告還沒出）")
        plt = plt_setup(ctx)
        fig, axs = two_panels(plt, h=3.4, sharey=True)
        sty = {"32K": ("o", C[7]), "128K": ("s", C[0]), "256K": ("^", C[2])}
        anyd = False
        for ax, m in zip(axs, MODELS):
            G = rep.get("models", {}).get(m, {}).get("gen", {})
            for ctxl, v in G.items():
                mk, col = sty.get(ctxl, ("o", "#777"))
                first = True
                for s, c in sorted(v.get("configs", {}).items(), key=lambda kv: kv[1].get("multiple", 0)):
                    if s == "none" or not c.get("diff"):
                        continue
                    anyd = True
                    dd = c["diff"]
                    ax.errorbar(c["multiple"], 100 * dd["mean"], yerr=[[100 * (dd["mean"] - dd["lo"])], [100 * (dd["hi"] - dd["mean"])]], fmt=mk,
                                color=col, ms=4, capsize=1.5, lw=0.8, mfc=col if c.get("eps_pass") else "white",
                                label=f"{ctxl}（{v.get('n_items')} 題）" if first else None)
                    first = False
            ax.axhline(-100 * rep.get("eps", 0.02), color=C[7], ls="--", lw=1)
            ax.axhline(0, color="#666", lw=0.7)
            ax.set_xscale("log")
            nice_log(ax, "x")
            ax.set_title(MODEL_LABEL[m])
            ax.set_xlabel("壓縮倍數（對數）")
            ax.legend(fontsize=6.4)
        if not anyd:
            raise NotMeasured("x4_report.json 裡還沒有生成的配置（X4b／X4c 沒跑完）")
        axs[0].set_ylabel("生成答對率 − E0（百分點，95% CI）")
        fig.tight_layout()
        return fig, "實心＝過 ε、空心＝沒過；紅虛線＝−ε。讀 x4_report.json 的 models.*.gen。"
    d.raw(add_fig(ctx, sec, "X4b／X4c：生成答對率 vs 倍數（32K、128K、256K）", f_gen))

    def t_alloc():
        p = os.path.join(D_, "x4_best_allocation.csv")
        rows = read_csv(p)
        ctx.src(sec, p, rids_of_rows(rows))
        out = [[MODEL_LABEL.get(r["model"], r["model"]), esc(r["T"]), esc(r.get("tf_best") or "—"), fx(r.get("tf_multiple"), 1),
                esc(r.get("gen_best") or "—"), fx(r.get("gen_multiple"), 1),
                (f"{pm(100 * float(r['gen_diff']))} [{pm(100 * float(r['gen_diff_lo']))}, {pm(100 * float(r['gen_diff_hi']))}]"
                 if ffloat(r.get("gen_diff")) is not None else "—"),
                esc(r.get("eps") or "—"), esc(r.get("mlperf99") or "—"), fx(r.get("multiple_with_lossless"), 1)] for r in rows]
        return (["模型", "目標 T", "TF 最好（X4a）", "倍數", "生成最佳配比（X4b）", "倍數", "對 E0（百分點）", "ε", "≥99%", "×無損後倍數"], out,
                "讀 x_stack/x4/x4_best_allocation.csv。")
    d.raw(add_table(ctx, sec, "X4：目標倍數 → 最佳配比", t_alloc))

    def t_lim():
        if not rep:
            raise NotMeasured("x4_report.json 不存在")
        out = []
        for m in MODELS:
            for ctxl, v in rep.get("models", {}).get(m, {}).get("gen", {}).items():
                lm = v.get("limits", {})
                if "status" in lm:
                    out.append([MODEL_LABEL[m], esc(ctxl), esc(lm["status"]), "", "", ""])
                    continue

                def fmx(x):
                    return "沒有" if not x else f"{esc(x['spec'])}（{fx(x['multiple'], 1)}×）"
                out.append([MODEL_LABEL[m], esc(ctxl), fmx(lm.get("eps")), fmx(lm.get("mlperf99")), fmx(lm.get("mlperf999")),
                            esc(v.get("first_T_out_of_eps", "—"))])
        return (["模型", "長度", "B*_ε（ε 內最省）", "B*_99", "B*_99.9", "最佳配比第一個掉出 ε 的 T"], out, "題數 &lt; 48 → INCOMPLETE、不報倍數（事先寫死）。")
    d.raw(add_table(ctx, sec, "X4：極限", t_lim))

    def t_qa():
        if not rep:
            raise NotMeasured("x4_report.json 不存在")
        out = []
        for m in MODELS:
            for ax_, v in rep.get("models", {}).get(m, {}).get("x4a", {}).get("qa", {}).items():
                cells = []
                for T, x in v.get("by_T", {}).items():
                    if x.get("verdict"):
                        cells.append(f"T{T}：{'值得' if x['verdict'].startswith('值得') else '不值得'}")
                sm = v.get("summary", {})
                out.append([MODEL_LABEL[m], esc(v.get("desc", ax_)), "；".join(cells) or "—", f"{sm.get('worth', 0)}／{sm.get('n_compared', 0)}"])
        return (["模型", "軸（加在 xKV＋INT4 上）", "各目標倍數的判定（TF）", "值得／可比"], out,
                "「不值得」＝帶內有這一軸的最好配置對沒有的最好配置，配對差 95% CI 上界 &lt; 0（事先寫死）。")
    d.raw(add_table(ctx, sec, "X4 Q-A：哪一軸值得疊（TF）", t_qa))

    def t_turn():
        if not rep:
            raise NotMeasured("x4_report.json 不存在")
        out = []
        for m in MODELS:
            X = rep.get("models", {}).get(m, {}).get("x4d")
            if not X:
                continue
            t1, t2 = X.get("1", {}).get("configs", {}), X.get("2", {}).get("configs", {})
            for s, c in sorted(t2.items(), key=lambda kv: kv[1].get("multiple", 0)):
                out.append([MODEL_LABEL[m], esc(s), fx(c.get("multiple"), 1), ci_pp((t1.get(s) or {}).get("diff")), ci_pp(c.get("diff")),
                            esc(X.get("verdict_turn2", {}).get(s, "—")), ci_pp(X.get("2", {}).get("vs_drop", {}).get(s))])
        if not out:
            raise NotMeasured("x4_report.json 裡沒有 x4d（多輪還沒跑）")
        return (["模型", "配置", "倍數", "第 1 輪 對 E0", "第 2 輪 對 E0", "第 2 輪判定", "第 2 輪 對 drop 對照"], out, "百分點、95% CI。")
    d.raw(add_table(ctx, sec, "X4d：多輪（第 2 輪換問題）", t_turn))

    def t_ll():
        p = os.path.join(D_, "lossless_x4.json")
        j = read_json(p)
        ctx.src(sec, p, [j.get("run_id")])
        out = []
        for m, dd in j.get("summary", {}).items():
            for s, v in sorted(dd.items()):
                out.append([MODEL_LABEL.get(m, m), esc(s), "／".join(f"{k} {fx(x, 3)}" for k, x in v.get("ratio", {}).items()),
                            "值得" if v.get("worth_stacking") else "不值得", esc(json.dumps(v.get("decomp_MBps_median"), ensure_ascii=False))])
        return (["模型", "配置", "實際位元組的無損比", f"疊加判定（{esc(j.get('head_codec'))} ≥ {esc(j.get('worth'))}）", "解壓 MB/s（中位數）"], out,
                "讀 x_stack/x4/lossless_x4.json。")
    d.raw(add_table(ctx, sec, "X4：前緣配置的實際位元組再做無損壓縮", t_ll))
    exp_tail(d, ctx, e, spec, None)


# ---------------------------------------------------------------------------- 4.12 I3
I3_TAG = OrderedDict([("main", "主掃描（本地 SSD、CPU 大）"), ("busy_exo", "GPU 外加忙（duty 0.7）"), ("busy_off", "GPU 忙的效應關掉（對照）"),
                      ("cake", "Cake 式還原"), ("sigma2", "間隔 σ＝2"), ("ssd_nfs", "下層換 NFS"), ("cpu_small", "CPU 層小（0.25×HBM）"),
                      ("cpu_small_nfs", "CPU 層小＋NFS")])
I3_BASE = OrderedDict([("oracle_per_cell", "對每格最好的現成規則（神諭）"), ("fixed_vllm_tier", "對 vLLM 原生 CPU＋SSD 卸載（固定）"),
                       ("fixed_default_vllm", "對不卸載的 vLLM 預設")])


def sec_i3(d: Doc, ctx: Ctx):
    e = EX("i3")
    sec = e["sec"]
    D_ = ctx.R("i3_pause_sim")
    sm = None
    try:
        sm = read_json(os.path.join(D_, "summary.json"))
        ctx.src(sec, os.path.join(D_, "summary.json"), [sm.get("run_id")])
    except NotMeasured:
        pass
    spec = dict(
        q="agent 呼叫工具暫停時，KV 該留在 GPU、卸到 CPU、卸到 SSD，還是丟掉之後重算？依實測 κ（重算時間 ÷ 搬運時間）做的四選一，"
          "有沒有比現成的規則（vLLM 預設、vLLM 原生 CPU＋SSD、Continuum、Tokencake、InferCept）好？「最好的規則」會不會隨硬體翻轉？",
        method=("CPU 上的離散事件模擬（code/m16_pause_sim.py；"
                + (f"{sm.get('n_runs_ok')} 次、{sm.get('n_runs_not_ok')} 次失敗" if sm else "次數 NOT_MEASURED")
                + "）：MI300X 的實測成本常數（每個附 run_id，costs_sources.csv）、Mooncake toolagent 推得的 session（以 512-token block 換算並驗算）；"
                  "主掃描與 7 組敏感度；事前凍結的 κ 感知規則。10/20 另換三種基準重讀同一批格子（不重跑模擬）。"),
        rule="L3 缺口 1 事先寫死的證偽：κ 感知的四選一比最好的現成規則，回來那一輪 TTFT P95 好不到 5%（或兩平台選的動作完全相同）→ 缺口不成立，停（CLAUDE.md 規則 4）。",
        limits=["這是模擬：常數是實測的，排程與佇列是模型；GPU 驗證點沒有跑。",
                "成本常數是第二階段 harness 路徑；vLLM 實際 CPU 層的速度沒有代入。",
                "3090 的舊量測是 HEAVY 污染，只進靜態表（NOT_MEASURED）。",
                "事後加的 kappa_tier、kappa_end 是事後候選，不能下判定。"])
    exp_head(d, ctx, e, spec)

    def f_rr():
        p = os.path.join(D_, "reread_baselines.csv")
        rows = read_csv(p)
        ctx.src(sec, p, rids_of_rows(rows) + rids_of_rows(rows, "source_run_id"))
        R = [r for r in rows if r["candidate"] == "kappa" and r["baseline"] in I3_BASE]
        tags = [t for t in I3_TAG if any(r["tag"] == t for r in R)]
        plt = plt_setup(ctx)
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        h = 0.8 / len(I3_BASE)
        for k, (b, lab) in enumerate(I3_BASE.items()):
            for i, t in enumerate(tags):
                v = [r for r in R if r["tag"] == t and r["baseline"] == b]
                if not v:
                    continue
                med, p10, p90 = 100 * float(v[0]["median"]), 100 * float(v[0]["p10"]), 100 * float(v[0]["p90"])
                y = i + (k - 1) * h
                ax.barh(y, med, height=h * 0.9, color=C[k], label=lab if i == 0 else None)
                ax.plot([p10, p90], [y, y], color=INK, lw=0.8)
        ax.axvline(0, color="#444", lw=0.8)
        ax.axvline(5, color=C[7], ls="--", lw=0.9)
        ax.set_yticks(range(len(tags)))
        ax.set_yticklabels([I3_TAG[t] for t in tags], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(-110, 110)
        ax.set_xlabel("事前凍結的 κ 規則：1 − κ ÷ 基準（回來那一輪 TTFT P95，%；&gt;0＝κ 較好）".replace("&gt;", ">"))
        ax.legend(fontsize=6.4, loc="lower right")
        fig.tight_layout()
        return fig, "〔模擬〕長條＝格子的中位數、黑線＝第 10–90 百分位；紅虛線＝5% 門檻。超出 ±110% 的部分被截掉。出處：reread_baselines.csv。"
    d.raw(add_fig(ctx, sec, "I3〔模擬〕：κ 感知去留規則對三種基準", f_rr))

    def rule_fn():
        def t():
            if not sm:
                raise NotMeasured("i3_pause_sim/summary.json 不存在")
            out = []
            for tag, v in sm.get("gain_kappa", {}).items():
                x = v.get("ttft_ret_p95", {})
                out.append([esc(I3_TAG.get(tag, tag)), str(x.get("n_cells")), f"{x.get('n_ge_5pct')}／{x.get('n_cells')}", fpct(x.get("median")),
                            fpct(x.get("max")), "<span class='ok'>缺口不成立（停）</span>" if tag == "main" and x.get("n_ge_5pct") == 0 else
                            ("<b>主掃描成立</b>" if tag == "main" else "（敏感度，只報）")])
            return (["設定", "格數", "κ 好 ≥5% 的格子", "中位數", "最好", "判定"], out,
                    "gain＝1 − κ ÷ 每格最好的現成規則（回來那一輪 TTFT P95）；判定只看主掃描。讀 summary.json 的 gain_kappa。")
        d.raw(add_table(ctx, sec, "I3 的事先寫死條件（讀 summary.json）", t))
    exp_tail(d, ctx, e, spec, rule_fn)


# ---------------------------------------------------------------------------- 4.13 V0／F2／F1
def load_summaries(ctx: Ctx, sec: str, sub: str, prefix: str) -> list:
    out = []
    for p in sorted(glob.glob(ctx.R(sub, f"{prefix}_*.json"))):
        try:
            j = read_json(p)
        except Exception as ex:                  # noqa: BLE001
            ctx.warn(sec, f"讀不懂 {rel(p)}：{ex}")
            continue
        j["_path"] = p
        ctx.src(sec, p, [j.get("run_id")])
        out.append(j)
    return out


def sec_vllm(d: Doc, ctx: Ctx):
    e = EX("vllm")
    sec = e["sec"]
    spec = dict(
        q="V0：vLLM 0.28（ROCm）的前綴快取（APC）、CPU 卸載、tiering 檔案層在 MI300X 上能不能跑、命中與不命中的輸出是否逐 token 相同？"
          "F2：重現 vLLM #44294（命中的 chunk 正被別的請求載入 → 整個請求延後）：共用前綴的併發突發下，開 CPU 卸載會不會反而更慢？"
          "F1：前綴復用的收益在「重疊 × KV 容量 × 併發」的哪些區域趨近 0 或變負？平均與 P95／goodput 選出的最佳設定會不會翻轉？",
        method="V0：每個設定一個 server，亂數 token 與真文字各 3 個 prompt × 8K／16K／32K，miss → 命中 → 再 miss 比輸出（temperature 0）。"
               "F2：三段（1 個請求寫入前綴 → 擠出 GPU → N 個同時送），P＝10K／32K、N＝1／8／32／64，設定 A（APC 不卸載）、A′（APC＋TRITON_ATTN）、"
               "B（APC＋CPU 卸載）、C（不復用），每格 3 次。F1：重疊 0／50／100%、8K／32K、到達 burst 與兩個 Poisson 速率、"
               "設定 off／apc／apc＋cpu × 容量 ample／0.75／0.40 工作集；三個設定都用 TRITON_ATTN（卸載會退回這個 kernel）。",
        rule="V0：a／b／c 任一不過 → F1／F2 不能用 vLLM（改用自己的 Restorer）；d（ROCM_ATTN patch）、e（fs 層）不過只記下。"
             "F2：所有 N 下 (B) ≤ (A) → MI300X／0.28 不受 #44294 影響，停。F1：找到「收益 &lt;5%」或「變差」的區域，而且能用重疊、容量、併發三個變數說清楚邊界"
             " → 成功；全部區域收益都很大而且單調 → 停（只當 baseline）。Poisson 速率＝0.5／0.9×λ_sat、TTFT SLO＝1.5／5×單請求冷 prefill（f1_common.sh 事先寫死）。",
        limits=["ROCm 上接 KV connector 會退回 TRITON_ATTN；F1 三個設定統一用 TRITON_ATTN，F2 另有 A′ 做同 kernel 的對照。",
                "亂數 token 的 prompt 會有近似平手（輸出不同不一定是 KV 錯；看真文字那一輪與 first_diverge）。",
                "fs 層沒有容量上限；reset 不清 fs 層。",
                "02:00Z 的截止規則可能讓 F1 的部分 job 被跳過（.cutoff），表上會照實寫。"])
    exp_head(d, ctx, e, spec)

    def t_v0():
        S = load_summaries(ctx, sec, "v0_vllm_smoke", "v0check")
        if not S:
            raise NotMeasured("results/m16_stage3_mi300x/v0_vllm_smoke/v0check_*.json 不存在（40_v0 還沒跑）")
        out = []
        for j in sorted(S, key=lambda x: str((x.get("args") or {}).get("label"))):
            v = j.get("verdict", {})
            out.append([esc((j.get("args") or {}).get("label")), esc(j.get("mode")), f"<b>{esc(j.get('overall'))}</b>",
                        esc("；".join(f"{k} {x}" for k, x in v.items())), esc((j.get("guard") or {}).get("verdict")), esc(j.get("run_id"))])
        return (["設定（label）", "模式", "整體", "各項檢查", "gpu_guard", "run_id"], out,
                "PASS＝輸出 token 與 miss 完全相同、命中數夠、CPU／fs 真的有載入；NOT_AVAILABLE＝抓不到 metrics（不當成 PASS）。")
    d.raw(add_table(ctx, sec, "V0：vLLM 功能閘門", t_v0))

    def f2_rows():
        S = load_summaries(ctx, sec, "f2_shared_prefix_burst", "summary")
        if not S:
            raise NotMeasured("results/m16_stage3_mi300x/f2_shared_prefix_burst/summary_*.json 不存在（50_f2 還沒跑）")
        return S

    def f_f2():
        S = f2_rows()
        g = defaultdict(lambda: defaultdict(list))
        for j in S:
            m = re.match(r"^f2_(\w+?)_P(\d+)_N(\d+)_r\d+$", str((j.get("args") or {}).get("label")))
            if not m or not isinstance(j.get("ttft_s"), dict):
                continue
            cfg, P, N = m.group(1), int(m.group(2)), int(m.group(3))
            p95 = ffloat(j["ttft_s"].get("p95"))
            mk = ffloat((j.get("burst") or {}).get("makespan_s"))
            if p95 is not None:
                g[(cfg, P)][N].append((p95, mk))
        if not g:
            raise NotMeasured("F2 的 summary 沒有主格（f2_{設定}_P{P}_N{N}_r{次}）")
        plt = plt_setup(ctx)
        fig, axs = two_panels(plt, h=3.0, sharey=False)
        cl = {"A": (C[0], "A：APC 不卸載"), "Ap": (C[2], "A′：APC＋TRITON_ATTN"), "B": (C[7], "B：APC＋CPU 卸載"), "C": ("#777777", "C：不復用")}
        for ax, P in zip(axs, (10240, 32768)):
            for cfg, (col, lab) in cl.items():
                v = g.get((cfg, P))
                if not v:
                    continue
                Ns = sorted(v)
                ax.plot(Ns, [statistics.median(x[0] for x in v[n]) for n in Ns], "-o", color=col, ms=3.5, label=lab)
                mks = [statistics.median(x[1] for x in v[n] if x[1] is not None) if any(x[1] is not None for x in v[n]) else float("nan") for n in Ns]
                ax.plot(Ns, mks, ":", color=col, lw=1)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            nice_log(ax, "y")
            ax.set_xticks([1, 8, 32, 64])
            ax.set_xticklabels(["1", "8", "32", "64"])
            ax.set_title(f"共用前綴 P＝{P // 1024}K")
            ax.set_xlabel("同時請求數 N")
        axs[0].set_ylabel("burst 段 TTFT P95（實線）／makespan（點線）（s）")
        axs[1].legend(fontsize=6.4)
        fig.tight_layout()
        return fig, "每點＝3 次的中位數（只用乾淨的 run；污染的 run 不會寫進 results）。"
    d.raw(add_fig(ctx, sec, "F2：共用前綴的併發突發（#44294 重現）", f_f2))

    def t_f2probe():
        S = f2_rows()
        out = []
        for j in S:
            lab = str((j.get("args") or {}).get("label"))
            if "probe" not in lab:
                continue
            cl = (j.get("server") or {}).get("connector_log") or {}
            dl = cl.get("delay_chunks_being_loaded") if isinstance(cl, dict) else None
            out.append([esc(lab), fx((j.get("ttft_s") or {}).get("p50"), 3), fx((j.get("ttft_s") or {}).get("p95"), 3),
                        fx((j.get("burst") or {}).get("makespan_s"), 3), esc(json.dumps(dl, ensure_ascii=False) if dl else cl.get("status", "—")),
                        esc(j.get("run_id"))])
        if not out:
            raise NotMeasured("沒有 f2_B_probe_* 的 summary")
        return (["探測（B，P＝10K、N＝64）", "TTFT P50（s）", "P95（s）", "makespan（s）", "延後訊息（行數、請求數）", "run_id"], out,
                "evict＝reset／none／冷啟動三種，看哪一種重現 issue 的情形。")
    d.raw(add_table(ctx, sec, "F2：三種 evict 情形的探測", t_f2probe))

    def f_f1():
        S = load_summaries(ctx, sec, "f1_prefix_reuse", "summary")
        if not S:
            raise NotMeasured("results/m16_stage3_mi300x/f1_prefix_reuse/summary_*.json 不存在（51_*／52_* 還沒跑）")
        g = defaultdict(dict)
        for j in S:
            m = re.match(r"^f1_(\d+)_(\w+?)_(ample|c075|c040)_([\d.]+)_(burst|p1|p2)$", str((j.get("args") or {}).get("label")))
            if m and isinstance(j.get("ttft_s"), dict) and ffloat(j["ttft_s"].get("p95")) is not None:
                L, cfg, cap, ov, arr = m.groups()
                g[(int(L), arr)][(cfg, cap, float(ov))] = float(j["ttft_s"]["p95"])
        if not g:
            raise NotMeasured("F1 的 summary 沒有格子（f1_{L}_{設定}_{容量}_{重疊}_{到達}）")
        plt = plt_setup(ctx)
        Ls = sorted({k[0] for k in g})
        fig, axs = plt.subplots(1, len(Ls), figsize=(7.2, 3.1), squeeze=False)
        cfgc = {"off": C[0], "apc": C[2], "apccpu": C[7]}
        capls = {"ample": "-", "c075": "--", "c040": ":"}
        for ax, L in zip(axs[0], Ls):
            v = g.get((L, "burst"), {})
            for cfg, col in cfgc.items():
                for cap, ls in capls.items():
                    pts = sorted((ov, y) for (c1, c2, ov), y in v.items() if c1 == cfg and c2 == cap)
                    if pts:
                        ax.plot([x for x, _ in pts], [y for _, y in pts], ls, marker="o", ms=3, color=col, label=f"{cfg}・{cap}")
            ax.set_yscale("log")
            nice_log(ax, "y")
            ax.set_title(f"prompt {L // 1024}K、到達＝burst")
            ax.set_xlabel("共同前綴比例（重疊）")
        axs[0][0].set_ylabel("TTFT P95（s）")
        axs[0][-1].legend(fontsize=5.8, ncol=1)
        fig.tight_layout()
        return fig, "顏色＝設定（off 不復用、apc、apc＋cpu 卸載）；線型＝KV 容量（實線充足、虛線 0.75 工作集、點線 0.40）。Poisson 到達的格子見 summary 檔。"
    d.raw(add_fig(ctx, sec, "F1：前綴復用地圖（TTFT P95 vs 重疊）", f_f1))

    def t_f1cal():
        out = []
        for p in sorted(glob.glob(ctx.R("f1_prefix_reuse", "calib_*.json"))):
            j = read_json(p)
            ctx.src(sec, p, [j.get("sat_run")])
            out.append([esc(j.get("L")), fx(j.get("t1"), 3), fx(j.get("lambda_sat"), 3), esc(j.get("rates")), esc(j.get("slo_ttft")), esc(j.get("sat_run"))])
        if not out:
            raise NotMeasured("f1_prefix_reuse/calib_*.json 不存在")
        return (["prompt 長度", "單請求冷 prefill t1（s）", "λ_sat（req/s）", "Poisson 速率", "TTFT SLO（s）", "sat run"], out, "f1_common.sh 事先寫死的換算。")
    d.raw(add_table(ctx, sec, "F1：校準（λ_sat、SLO）", t_f1cal))
    exp_tail(d, ctx, e, spec, None)


# ============================================================================ 第 5 章：綜合
def sec_synthesis(d: Doc, ctx: Ctx):
    sec = "5"
    d.h(1, "ch5", "5　綜合")
    d.p("本章只把前面各節的結果檔並排（數字即時讀檔）；解讀由主 session 撰寫。")
    d.h(2, "ch5_limit", "5.1　壓縮疊加的極限與配比")

    def limits_rows():
        out = []   # (model, 來源, spec, multiple, ci, n)
        try:
            rows32, src = x2_rows_32k(ctx, sec)
            for m in MODELS:
                R = [r for r in rows32 if r["model"] == m and r["combo"] != "none" and ffloat(r.get("ratio"))]
                for lab, key in (("ε", "feasible"), ("MLPerf 99%", "mlperf99")):
                    ok = [r for r in R if str(r.get(key)) in ("1", "True")]
                    if ok:
                        b = min(ok, key=lambda r: float(r["ratio"]))
                        out.append((m, f"X2＋X2b 32K（{lab}；{src}）", b["combo"], 1 / float(b["ratio"]),
                                    f"{pm(100 * float(b['acc_delta']))} [{pm(100 * float(b['acc_delta_lo']))}, {pm(100 * float(b['acc_delta_hi']))}]",
                                    f"{b.get('n_items')}（{b.get('source', '')}）"))
                    else:
                        out.append((m, f"X2＋X2b 32K（{lab}；{src}）", "沒有配置過", None, "—", "—"))
        except NotMeasured as ex:
            out.append(("—", "X2＋X2b 32K", f"NOT_MEASURED：{ex}", None, "—", "—"))
        try:
            rep = read_json(ctx.R("x_stack", "x2_report.json"))
            for m in MODELS:
                B = rep.get("by_ctx", {}).get("128K", {}).get(m, {}).get("configs", {})
                for lab, key in (("ε", "eps_pass"), ("MLPerf 99%", "mlperf0.99")):
                    ok = [(s, v) for s, v in B.items() if v.get(key) and s != "none"]
                    if ok:
                        s, v = min(ok, key=lambda kv: kv[1]["ratio"])
                        out.append((m, f"X2 128K（{lab}；探索）", s, 1 / v["ratio"], ci_pp(v["diff"]), str(v.get("n"))))
                    else:
                        out.append((m, f"X2 128K（{lab}；探索）", "沒有配置過", None, "—", "—"))
        except NotMeasured as ex:
            out.append(("—", "X2 128K", f"NOT_MEASURED：{ex}", None, "—", "—"))
        try:
            r2 = read_json(ctx.R("x_stack", "v2", "xlcal_report.json"))
            for m in MODELS:
                Cf = r2.get("models", {}).get(m, {}).get("gen", {}).get("configs", {})
                ok = [(s, v) for s, v in Cf.items() if v.get("eps_pass")]
                if ok:
                    s, v = min(ok, key=lambda kv: kv[1]["ratio"])
                    out.append((m, "xKV 校準 38c（ε；每請求位元組，基底共用另計）", s, 1 / v["ratio"], ci_pp(v["diff"]), "96"))
                else:
                    out.append((m, "xKV 校準 38c（ε）", "沒有配置過", None, "—", "—"))
        except NotMeasured as ex:
            out.append(("—", "xKV 校準 38c", f"NOT_MEASURED：{ex}", None, "—", "—"))
        try:
            rep4 = read_json(ctx.R("x_stack", "x4", "x4_report.json"))
            for m in MODELS:
                for ctxl, v in rep4.get("models", {}).get(m, {}).get("gen", {}).items():
                    lm = v.get("limits", {})
                    for lab, key in (("ε", "eps"), ("MLPerf 99%", "mlperf99")):
                        x = lm.get(key) if "status" not in lm else None
                        if x:
                            out.append((m, f"X4 {ctxl}（{lab}）", x["spec"], x["multiple"], ci_pp(x.get("diff")), str(v.get("n_items"))))
                        else:
                            out.append((m, f"X4 {ctxl}（{lab}）", lm.get("status", "沒有配置過"), None, "—", str(v.get("n_items"))))
        except NotMeasured as ex:
            out.append(("—", "X4", f"NOT_MEASURED：{ex}", None, "—", "—"))
        return out

    def f_lim():
        rows = [r for r in limits_rows() if r[3] is not None]
        if not rows:
            raise NotMeasured("沒有任何來源有 ε 內的配置")
        plt = plt_setup(ctx)
        srcs = []
        for r in rows:
            if r[1] not in srcs:
                srcs.append(r[1])
        fig, ax = plt.subplots(figsize=(7.2, 0.9 + 0.32 * len(srcs)))
        for j, m in enumerate(MODELS):
            for i, s in enumerate(srcs):
                v = [r for r in rows if r[0] == m and r[1] == s]
                if v:
                    ax.barh(i + (j - 0.5) * 0.38, v[0][3], height=0.36, color=MODEL_COLOR[m], label=MODEL_LABEL[m] if i == 0 else None)
                    ax.text(v[0][3] * 1.02, i + (j - 0.5) * 0.38, f"{v[0][3]:.1f}×  {v[0][2]}", va="center", fontsize=5.6, color=INK2)
        ax.set_yticks(range(len(srcs)))
        ax.set_yticklabels(srcs, fontsize=6.6)
        ax.invert_yaxis()
        ax.set_xscale("log")
        mx = max(r[3] for r in rows)
        ax.set_xlim(1, mx * 6)
        nice_log(ax, "x")
        ax.set_xlabel("ε（或 MLPerf 門檻）內最省的配置的壓縮倍數（對數）")
        h_, l_ = ax.get_legend_handles_labels()
        fig.legend(h_, l_, loc="lower center", ncol=2)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        return fig, "每一列是一個結果檔的「門檻內位元組最省的配置」；不同列的題目、題數、長度不同，只並列、不直接比。"
    d.raw(add_fig(ctx, sec, "各實驗在門檻內能壓到的倍數", f_lim))

    def t_lim():
        rows = limits_rows()
        return (["模型", "來源（門檻）", "門檻內最省的配置", "倍數", "對 E0（百分點，95% CI）", "題數"],
                [[MODEL_LABEL.get(r[0], r[0]), esc(r[1]), esc(r[2]), fx(r[3], 1), r[4], esc(r[5])] for r in rows], "")
    d.raw(add_table(ctx, sec, "門檻內最省的配置（各來源並列）", t_lim))
    d.h(2, "ch5_cost", "5.2　加速與成本（寫入 SVD、GPU 忙時重建）")

    def t_cost():
        D2 = ctx.R("x3_time", "v2")
        wc = read_csv(os.path.join(D2, "write_cost.csv"))
        ctx.src(sec, os.path.join(D2, "write_cost.csv"), rids_of_rows(wc))
        try:
            bd = big_dir(os.path.join(D2, "BIG_FILES.md"), "x3_full")
            cells = read_csv(os.path.join(bd, "map_cells.csv"))
        except NotMeasured:
            cells = []
        T = {(r["model"], r["tier"], float(r["duty"]), r["action"]): ffloat(r["T_s"]) for r in cells if r.get("grid") == "named"}
        out = []
        for r in wc:
            m, seg = r["model"], r["segment"]
            du = float(r["duty"])
            rc = [T.get((m, t, 0.0, f"{seg}@oneshot")) for t in ("cpu_pinned", "ssd_meas", "nfs_param")]
            out.append([MODEL_LABEL.get(m, m), esc(seg), fx(du, 1), fx(r["t_total_s"], 2), fx(r["recompute_region_idle_s"], 2)]
                       + [fx(x * 1000, 1) if x else "—" for x in rc])
        return (["模型", "段", "寫入時 duty", "寫入（壓縮）一次（s）", "同區閒時重算（s）", "還原：CPU 層（ms）", "還原：本地 SSD（ms）", "還原：NFS 級（ms）"], out,
                "寫入成本讀 x3_time/v2/write_cost.csv；還原時間讀 map v2 的 map_cells.csv（GPU 閒、一次拆）〔模擬：零件實測〕。")
    d.raw(add_table(ctx, sec, "xKV 組合：寫入一次 vs 還原一次", t_cost))

    def t_busy():
        s = read_json(ctx.R("x3_time", "v2", "map_summary.json"))
        out = [[esc(k), fx(v.get("slow_lr"), 2) + "×", fx(v.get("slow_int4"), 2) + "×", fx(v.get("busy_recon_over_h2d"), 2), esc(v.get("verdict"))]
               for k, v in s.get("el3", {}).items()]
        ag = s.get("aggregates", {})
        for mode in ("L", "Q", "X"):
            a = ag.get(mode, {})
            out.append([f"聚合：模式 {mode}", "", "", "", f"A 對 D {fpct(a.get('vs_D', {}).get('gm_gain'))}；A 對 F {fpct(a.get('vs_F', {}).get('gm_gain'))}；"
                                                       f"A 對 S {fpct(a.get('vs_S', {}).get('gm_gain'))}"])
        return (["項目", "低秩重建 忙／閒", "INT4 忙／閒", "忙時重建 ÷ H2D", "判定／聚合"], out, "讀 x3_time/v2/map_summary.json（el3、aggregates）。")
    d.raw(add_table(ctx, sec, "GPU 忙時的重建成本與還原決策的總好處", t_busy))
    d.h3("綜合解讀（主 session 撰寫）")
    d.raw(narrative(ctx, "synthesis", "綜合：壓縮疊加的極限與配比、加速與成本"))


def sec_candidates(d: Doc, ctx: Ctx):
    d.h(1, "ch6", "6　第三階段候選題目排名")
    d.raw(narrative(ctx, "candidates", "第三階段候選題目排名（每個附：問題是否真的存在、前人做到哪、我們的訊號、下一步最該做的一個實驗）"))
    d.p("<span class='small'>參考：文獻缺口排序見第 3.4 節；每個實驗的判定見第 4 章各節的「事先寫死的判定與結果」。</span>")


# ============================================================================ 第 7 章：限制與誠實聲明
def sec_limits(d: Doc, ctx: Ctx):
    sec = "7"
    d.h(1, "ch7", "7　限制與誠實聲明")
    d.h(2, "ch7_sim", "7.1　模擬 vs 實測")
    rows = [[e["sec"], esc(e["title"]), esc(e["kind"])] for e in EXPS]
    d.raw(html_table(["節", "實驗", "性質"], rows))
    d.raw("<ul><li>〔模擬〕：I1 的事先預測、I3 全部、X3 的 map（零件實測、排程是模型）。</li>"
          "<li>〔計算〕：S1 的盈虧、C1b 的還原時間、各處的位元組比（位元組帳，GPU 路徑有 assert 實際 tensor 位元組＝預測）。</li>"
          "<li>實測：其餘；量時間的實測都有 gpu_guard，污染的 run 不進 results；品質的實測分數照用、時間欄作廢。</li></ul>")
    d.h(2, "ch7_review", "7.2　審查方式")
    d.p("這一輪所有的程式、判定與文獻整理都是 <b>self-review</b>，不是 cross-model review（這台機器沒有 Codex MCP；CLAUDE.md §5）。"
        "判定門檻都在看到數字之前寫進說明檔或程式常數；事後改動（例如 38k 只比兩對、X4 的取捨）都在報告與 experiment_log 寫明。")
    d.h(2, "ch7_power", "7.3　統計力")

    def t_power():
        out = []

        def row(lab, m, n, cf):
            hw = [(v["diff"]["hi"] - v["diff"]["lo"]) / 2 for v in cf if v.get("diff") and v["diff"].get("hi") is not None]
            ok = [(v["diff"]["hi"] - v["diff"]["lo"]) / 2 for v in cf if v.get("diff") and v["diff"].get("hi") is not None
                  and v["diff"]["mean"] > -0.10 and v.get("spec") not in ("none",)]
            if hw:
                out.append([lab, MODEL_LABEL.get(m, m), esc(n), fx(100 * statistics.median(hw), 1),
                            fx(100 * statistics.median(ok), 1) if ok else "—", str(len(ok)), fx(100 * min(hw), 1)])
        rep = read_json(ctx.R("x_stack", "x2_report.json"))
        for ctxl in ("32K", "128K"):
            for m, B in rep.get("by_ctx", {}).get(ctxl, {}).items():
                ns = sorted({v.get("n") for v in B["configs"].values()})
                row(f"X2 {ctxl}", m, "／".join(map(str, ns)), list(B["configs"].values()))
        try:
            r2 = read_json(ctx.R("x_stack", "v2", "x2b_report.json"))
            for m, M in r2.get("models", {}).items():
                row("X2＋X2b（D）", m, M.get("n_items", {}).get("D"), list(M.get("configs", {}).get("D", {}).values()))
        except NotMeasured:
            pass
        return (["資料", "模型", "題數", "CI 半寬中位數：全部配置", "CI 半寬中位數：沒崩的配置", "沒崩的配置數", "最窄"], out,
                "百分點。95% CI 的半寬＝(上界 − 下界)／2；「沒崩」＝對 E0 的差 &gt; −10 個百分點、不含 MASK0。ε＝2 個百分點：半寬接近或大於 ε 時，"
                "「沒過 ε」多半是統計力不夠。")
    d.raw(add_table(ctx, sec, "生成答對率的 CI 半寬（即時計算）", t_power))
    d.h(2, "ch7_fail", "7.4　失敗與修正清單（從 experiment_log 抽「失敗 N」）")

    def t_fail():
        p = ctx.D("experiment_log.md")
        fl = parse_failures(open(need(p), encoding="utf-8").read())
        ctx.src(sec, p)
        return (["#", "失敗與修正（原文）"], [[str(f["n"]), md(f["text"] + ("\n\n" + "\n\n".join(f["more"]) if f["more"] else ""))] for f in fl],
                "照 CLAUDE.md 規則 2：每次失敗都停下來、記錄、修好才重開。")
    d.raw(add_table(ctx, sec, "失敗清單", t_fail))
    d.h(2, "ch7_human", "7.5　人工處理紀錄（NEEDS_HUMAN）")

    def t_nh():
        out = []
        for p in sorted(glob.glob(os.path.join(ctx.jobs, "NEEDS_HUMAN*"))):
            t = open(p, encoding="utf-8", errors="replace").read().strip().splitlines()
            out.append([esc(os.path.basename(p)), "<br/>".join(esc(x[:220]) for x in t[:6]) + ("<br/>…" if len(t) > 6 else "")])
        if not out:
            raise NotMeasured(f"{ctx.jobs} 沒有 NEEDS_HUMAN 檔")
        return (["檔", "內容（前 6 行）"], out, "runner 遇到失敗或要人決定的事時寫 NEEDS_HUMAN；處理後改名 .handled_&lt;時間&gt;。")
    d.raw(add_table(ctx, sec, "NEEDS_HUMAN 檔", t_nh))
    d.h(2, "ch7_cont", "7.6　污染（同卡入侵者）紀錄")

    def t_cont():
        out = []
        for lab, p, key in (("X1", ("x_stack", "x1_report.json"), "contention"), ("X2b", ("x_stack", "v2", "x2b_report.json"), "contention_x2b"),
                            ("C1a", ("c1a_predictive", "c1a_summary.json"), "contention"), ("C1b", ("c1_selective_restore", "c1b_summary.json"), "contention"),
                            ("X4", ("x_stack", "x4", "x4_report.json"), "contention"), ("xKV 校準 38k", ("x_stack", "v3", "xlcal3_report.json"), "contention_new")):
            fp = ctx.R(*p)
            if not os.path.exists(fp):
                out.append([lab, "NOT_MEASURED", "", ""])
                continue
            c = read_json(fp).get(key) or {}
            out.append([lab, esc(c.get("rows_with_intruders")), esc(c.get("rows_not_checked", "—")), esc("、".join(c.get("host_levels", [])))])
        return (["實驗", "有入侵者的列", "沒檢查的列", "主機層級"], out,
                "品質實驗：同卡上自己的其他行程也算入侵者，所以列數多是預期的；分數照用、時間欄作廢（CLAUDE.md §3）。AMD 上主機層級是 UNOBSERVABLE_HOST。")
    d.raw(add_table(ctx, sec, "品質實驗的污染統計", t_cont))
    d.h3("限制與誠實聲明（主 session 撰寫）")
    d.raw(narrative(ctx, "limitations", "限制與誠實聲明（哪些結論不能下、哪些還要驗）"))


# ============================================================================ 第 8 章：附錄
def scan_run_ids(path: str) -> list:
    try:
        if path.endswith(".csv"):
            with open(path, newline="") as f:
                rd = csv.DictReader(f)
                if not rd.fieldnames:
                    return []
                cols = [c for c in rd.fieldnames if c in ("run_id", "run_id_map", "src_run_ids", "micro_run_id", "source_run_id")]
                out = set()
                for r in rd:
                    for c in cols:
                        out.update(split_rids(r.get(c)))
                return sorted(out)
        if path.endswith(".json"):
            j = json.load(open(path))
            out = set()
            if isinstance(j, dict):
                for k in ("run_id", "plan_run_id"):
                    out.update(split_rids(j.get(k)))
                for k in ("src_run_ids", "run_ids", "src_run_ids_x2b", "src_run_ids_new", "src_run_ids_gen"):
                    if isinstance(j.get(k), list):
                        for x in j[k]:
                            out.update(split_rids(x))
            return sorted(out)
    except Exception:                           # noqa: BLE001
        return []
    return []


def sec_appendix(d: Doc, ctx: Ctx):
    sec = "8"
    d.h(1, "ch8", "8　附錄")
    d.h(2, "ch8_idx", "8.1　run_id 索引")

    def t_idx():
        p = ctx.D("results.csv")
        rows = read_csv(p)
        ctx.src(sec, p, rids_of_rows(rows))
        g = OrderedDict()
        for r in rows:
            e = g.setdefault(r["exp"], dict(n=0, rids=set(), st=Counter(), kind=Counter()))
            e["n"] += 1
            e["rids"].update(split_rids(r["run_id"]))
            e["st"][r["status"]] += 1
            e["kind"][r["kind"]] += 1
        out = []
        for k, e in g.items():
            rr = sorted(e["rids"])
            out.append([esc(k), str(e["n"]), esc("、".join(f"{a} {b}" for a, b in e["st"].items())), esc("、".join(f"{a or '—'} {b}" for a, b in e["kind"].items())),
                        esc(", ".join(rr[:4]) + (f", …（共 {len(rr)} 個）" if len(rr) > 4 else ""))])
        mt = dt.datetime.fromtimestamp(os.path.getmtime(p), dt.timezone.utc)
        return (["實驗", "列數", "狀態", "性質", "run_id"], out,
                f"彙整 docs/stage3_20261019/results.csv（code/m16_index.py 產生；檔案時間 {mt:%m/%d %H:%M}Z）。results.csv 可能比結果檔舊，下一張表是直接掃描結果檔。")
    d.raw(add_table(ctx, sec, "results.csv 彙整", t_idx))

    def t_scan():
        g = OrderedDict()
        for p in sorted(glob.glob(os.path.join(ctx.res, "**", "*"), recursive=True)):
            if os.path.isdir(p) or not p.endswith((".csv", ".json")):
                continue
            sub = os.path.relpath(os.path.dirname(p), ctx.res)
            e = g.setdefault(sub, dict(files=0, rids=set()))
            e["files"] += 1
            e["rids"].update(scan_run_ids(p))
        out = []
        for sub, e in g.items():
            rr = sorted(e["rids"])
            out.append([esc(sub), str(e["files"]), str(len(rr)), esc(", ".join(rr[:3]) + (f", …" if len(rr) > 3 else "")),
                        esc(rr[-1] if rr else "—")])
        if not out:
            raise NotMeasured(f"{rel(ctx.res)} 沒有 CSV／JSON")
        return (["結果目錄（相對 results/m16_stage3_mi300x）", "CSV／JSON 檔數", "不同 run_id 數", "最早的幾個", "最新的"], out,
                "掃描每個 CSV 的 run_id 類欄位與 JSON 的 run_id／src_run_ids。原始 log：/mlsteam/data/tiara/runs/&lt;run_id&gt;/。")
    d.raw(add_table(ctx, sec, "結果目錄的 run_id 掃描", t_scan))
    d.h(2, "ch8_repro", "8.2　重現指令")
    rp = ctx.D("reproduction.md")
    try:
        d.raw(f"<p>完整的重現指令在 <b>{esc(rel(rp))}</b>；各實驗的作法與指令在對應的 howto（x_stack_howto.md、x3_time_howto.md、c1b_howto.md、"
              "c1a_s2_howto.md、k1_kvstruct_howto.md、v0_f1_i3_howto.md），每個 run 的實際指令在 run 目錄的 cmd.sh。以下轉錄 reproduction.md 的前三節。</p>")
        d.raw(md(md_section(rp, r"^## 0\. ", r"^## 3\. ")))
        ctx.src(sec, rp)
    except NotMeasured as ex:
        d.raw(f"<div class='box-nm'>NOT_MEASURED：{esc(ex)}</div>")
    d.raw("<p>這份報告：<code>source code/m7_env.sh; OMP_NUM_THREADS=4 nice -n 19 taskset -c 0-3 python code/m16_report_pdf.py build</code>"
          "（cpu 佇列的 job：code/m16_queue/jobs/90_report_pdf.sh；說明：docs/stage3_20261019/report/README.md）。</p>")
    d.h(2, "ch8_files", "8.3　檔案清單")

    def t_files():
        g = OrderedDict()
        for p in sorted(glob.glob(os.path.join(ctx.res, "**", "*"), recursive=True)):
            if os.path.isdir(p):
                continue
            sub = os.path.relpath(os.path.dirname(p), ctx.res)
            e = g.setdefault(sub, dict(n=0, size=0, mt=0, names=[]))
            e["n"] += 1
            e["size"] += os.path.getsize(p)
            e["mt"] = max(e["mt"], os.path.getmtime(p))
            e["names"].append(os.path.basename(p))
        out = []
        for sub, e in g.items():
            names = e["names"]
            shown = re.sub(r"_s\d+\.", "_s*.", "、".join(sorted(set(re.sub(r"_s\d+\.", "_s*.", n) for n in names))))
            out.append([esc(sub), str(e["n"]), f"{e['size'] / 1e6:.2f}", f"{dt.datetime.fromtimestamp(e['mt'], dt.timezone.utc):%m/%d %H:%M}Z",
                        esc(shown if len(shown) < 900 else shown[:897] + "…")])
        if not out:
            raise NotMeasured(f"{rel(ctx.res)} 是空的")
        return (["目錄", "檔數", "MB", "最新修改", "檔名（_s0、_s1… 合併成 _s*）"], out, "&gt;1 MB 的大檔在 run 目錄，見各目錄的 BIG_FILES.md。")
    d.raw(add_table(ctx, sec, "results/m16_stage3_mi300x 的檔案", t_files))
    d.h(2, "ch8_build", "8.4　這份報告怎麼產生的")
    d.raw("<!--BUILDINFO-->")


def build_info_html(ctx: Ctx) -> str:
    rows = [[esc(k), {"ok": "<span class='ok'>已撰寫</span>", "draft_only": "<span class='part'>只有草稿（報告不採用）</span>",
                      "missing": "<span class='nm'>待撰寫</span>"}[v]] for k, v in ctx.narr_status.items()]
    nm = [[esc(x["section"]), esc(x["item"]), esc(x["reason"])] for x in ctx.nm]
    parts = [f"<p>產生器 code/m16_report_pdf.py；git HEAD {esc(git_head())}；產生時間 {ctx.now:%Y-%m-%d %H:%M}Z。"
             f"圖 {len(ctx.figs)} 張（NOT_MEASURED {sum(f['status'] != 'ok' for f in ctx.figs)}）、表 {len(ctx.tables)} 張"
             f"（NOT_MEASURED {sum(t['status'] != 'ok' for t in ctx.tables)}）；讀檔警告 {len(ctx.warnings)} 則（全文在 build_log.json）。</p>",
             "<p><b>narrative 檔</b>（docs/stage3_20261019/report/narrative/&lt;節名&gt;.html）：</p>", html_table(["節名", "狀態"], rows)]
    if nm:
        parts += ["<p><b>NOT_MEASURED 的圖表</b>：</p>", html_table(["節", "項目", "原因"], nm)]
    return "".join(parts)


# ============================================================================ 字型：收字元 → 子集 → CFF 轉 TrueType
def font_cmap(path: str) -> set:
    from fontTools.ttLib import TTFont
    f = TTFont(path, lazy=True)
    return set(f.getBestCmap())


def sanitize_html(s: str, cmap: set) -> tuple:
    """文字節點：unescape → 字型沒有的字元換掉（CHAR_SUBST，否則 ?）→ escape。回傳 (html, Counter(換掉的字元))。"""
    repl = Counter()
    parts = re.split(r"(<[^>]*>)", s)
    for i in range(0, len(parts), 2):
        t = _html.unescape(parts[i])
        out = []
        for ch in t:
            o = ord(ch)
            if ch in "\n\r\t" or o in cmap:
                out.append(ch)
            elif ch in CHAR_SUBST:
                out.append(CHAR_SUBST[ch])
                repl[ch] += 1
            elif o < 32:
                continue
            else:
                out.append("?")
                repl[ch] += 1
        parts[i] = _html.escape("".join(out), quote=False)
    return "".join(parts), repl


def collect_chars(*texts) -> str:
    chars = set(chr(c) for c in range(32, 127))
    for t in texts:
        chars.update(_html.unescape(re.sub(r"<[^>]*>", " ", t)))
    chars.update("0123456789／第頁·　…•○◦■▪□–")      # 清單符號（MuPDF 自己產生，不在 HTML 文字裡）
    return "".join(sorted(c for c in chars if ord(c) >= 32))


def otf_to_ttf(font):
    """CFF 輪廓 → TrueType（Cu2QuPen＋TTGlyphPen 建 glyf／loca、刪 CFF／VORG、maxp 1.0、sfntVersion 0x00010000）。"""
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.ttLib import newTable
    if "CFF " not in font:
        return font
    order = font.getGlyphOrder()
    gs = font.getGlyphSet()
    glyphs = {}
    for g in order:
        tp = TTGlyphPen(gs)
        gs[g].draw(Cu2QuPen(tp, 1.0, reverse_direction=True))
        glyphs[g] = tp.glyph()
    font["loca"] = newTable("loca")
    glyf = font["glyf"] = newTable("glyf")
    glyf.glyphOrder = order
    glyf.glyphs = glyphs
    del font["CFF "]
    if "VORG" in font:
        del font["VORG"]
    glyf.compile(font)
    hm = font["hmtx"]
    for g in order:
        aw, _ = hm.metrics[g]
        hm.metrics[g] = (aw, getattr(glyf[g], "xMin", 0))
    maxp = font["maxp"] = newTable("maxp")
    maxp.tableVersion = 0x00010000
    for k in ("maxZones",):
        setattr(maxp, k, 1)
    for k in ("maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs", "maxStackElements",
              "maxSizeOfInstructions", "maxComponentElements"):
        setattr(maxp, k, 0)
    post = font["post"]
    post.formatType = 3.0
    font.sfntVersion = "\x00\x01\x00\x00"
    return font


def subset_font(src: str, dst: str, text: str) -> dict:
    from fontTools import subset
    from fontTools.ttLib import TTFont
    t0 = time.time()
    opts = subset.Options()
    opts.layout_features = []          # 不留 GSUB／GPOS：留著的話 MuPDF 會把部分數字換成替代字形，PDF 的文字擷取（ToUnicode）會壞
    opts.name_IDs = ["*"]
    opts.notdef_outline = True
    opts.desubroutinize = True
    opts.hinting = False
    f = TTFont(src)
    s = subset.Subsetter(opts)
    s.populate(text=text)
    s.subset(f)
    otf_to_ttf(f)
    f.save(dst)
    g = TTFont(dst)
    info = dict(src=src, dst=dst, bytes=os.path.getsize(dst), glyphs=len(g.getGlyphOrder()), sfnt=hex(int.from_bytes(g.reader.sfntVersion.encode("latin-1"), "big"))
                if isinstance(g.reader.sfntVersion, str) else str(g.reader.sfntVersion), has_glyf="glyf" in g, has_cff="CFF " in g,
                maxp=hex(g["maxp"].tableVersion), cmap_n=len(g.getBestCmap()), seconds=round(time.time() - t0, 2))
    return info


# ============================================================================ 排版（PyMuPDF Story）、驗收、預覽
def render_pdf(ctx: Ctx, body: str, css: str, out_pdf: str, cmap: set) -> dict:
    import pymupdf
    arch = pymupdf.Archive(ctx.out)
    MB = pymupdf.paper_rect("a4")
    BODY = MB + (50, 46, -50, -56)
    state = {"pos": [], "breaks": set(), "passes": 0}
    fig_ids = sorted(ctx.fig_w)

    def contentfn(positions):
        pages = {p.id: p.page_num for p in positions if getattr(p, "id", None)}
        state["pos"] = []
        state["passes"] += 1
        toc, _ = sanitize_html(toc_html(ctx, pages), cmap)
        b = body.replace("<!--TOC-->", toc)
        for fid in state["breaks"]:
            b = b.replace(f"<div class='fig' id='{fid}'>", f"<div class='fig' id='{fid}' style='page-break-before:always'>")
        return b

    def rectfn(rect_num, filled):
        return MB, BODY, None

    def positionfn(p):
        if getattr(p, "id", None) and getattr(p, "open_close", 1) & 1:
            state["pos"].append(dict(id=p.id, heading=p.heading, page=p.page_num, text=p.text))

    def shrunk_figs(doc):
        """實際排出來的圖（照文件順序）和要求的寬比；MuPDF 會把放不下剩餘空間的圖縮小。"""
        shown = [pymupdf.Rect(im["bbox"]).width for pg in doc for im in pg.get_image_info()]
        if len(shown) != len(fig_ids):
            ctx.warn("排版", f"PDF 裡的圖數 {len(shown)} ≠ 報告的圖數 {len(fig_ids)}；不檢查縮圖")
            return {}
        full = max(shown) if shown else 0
        out = {}
        for fid, dw in zip(fig_ids, shown):
            want = min(ctx.fig_w[fid], full)
            if want and dw < 0.95 * want:
                out[fid] = round(dw / want, 2)
        return out

    t0 = time.time()
    for _ in range(len(fig_ids) + 1):          # 貪婪分頁：每次只在文件順序上第一張被縮小的圖前面換頁，再重排
        doc = pymupdf.Story.write_stabilized_with_links(contentfn, rectfn, user_css=css, em=10, positionfn=positionfn,
                                                        archive=arch, add_header_ids=False)
        shr = shrunk_figs(doc)
        nxt = [f for f in sorted(shr) if f not in state["breaks"]]
        if not nxt:
            break
        state["breaks"].add(nxt[0])
        doc.close()
    if shr:
        ctx.warn("排版", f"這些圖仍被縮小（實際寬÷要求寬）：{shr}")
    total = doc.page_count
    for i, page in enumerate(doc):
        ft, _ = sanitize_html(FOOTER_TMPL.format(t=TITLE, n=i + 1, tot=total), cmap)
        page.insert_htmlbox(pymupdf.Rect(50, MB.y1 - 44, MB.x1 - 50, MB.y1 - 16), ft, css=css, archive=arch)
    ids = {hid: (lvl, text) for lvl, hid, text in ctx.headings}
    toc, seen = [], set()
    for p in state["pos"]:
        if p["id"] in ids and p["id"] not in seen and p["id"] != "toc" and ids[p["id"]][0] <= 2:
            seen.add(p["id"])
            toc.append([ids[p["id"]][0], ids[p["id"]][1], p["page"]])
    lvl_ok, prev = [], 0
    for t in toc:                                # set_toc 要求層級不跳
        if t[0] > prev + 1:
            t = [prev + 1, t[1], t[2]]
        lvl_ok.append(t)
        prev = t[0]
    if lvl_ok:
        doc.set_toc(lvl_ok)
    doc.set_metadata(dict(title=TITLE, author="Tiara 第三階段（自動產生：code/m16_report_pdf.py）", subject="第三階段前置探索報告",
                          creator="m16_report_pdf.py（PyMuPDF Story）", producer="PyMuPDF"))
    tmp = out_pdf + ".tmp"
    doc.save(tmp, garbage=4, deflate=True)
    doc.close()
    os.replace(tmp, out_pdf)
    return dict(pages=total, seconds=round(time.time() - t0, 1), passes=state["passes"], page_breaks_before_figs=sorted(state["breaks"]),
                figs_shrunk=shr, headings={p["id"]: p["page"] for p in state["pos"]})


def verify_pdf(path: str) -> dict:
    import pymupdf
    doc = pymupdf.open(path)
    fonts, bad = Counter(), set()
    foot_bad = []
    n = doc.page_count
    for i, p in enumerate(doc):
        for f in p.get_fonts(full=True):
            name = f[3]
            fonts[name] += 1
            base = name.split("+")[-1]
            if not base.startswith(FONT_FAMILY_PDF) or "Fallback" in base or "Droid" in base:
                bad.add(name)
        if f"第 {i + 1}／{n} 頁" not in p.get_text():
            foot_bad.append(i + 1)
    out = dict(pages=n, fonts=dict(fonts), bad_fonts=sorted(bad), footer_missing=foot_bad, toc_entries=len(doc.get_toc()),
               ok=not bad and not foot_bad and n > 0, bytes=os.path.getsize(path))
    doc.close()
    return out


def previews(path: str, outdir: str, pages: list, dpi=80) -> list:
    import pymupdf
    os.makedirs(outdir, exist_ok=True)
    for f in glob.glob(os.path.join(outdir, "page_*.png")):
        os.remove(f)
    doc = pymupdf.open(path)
    out = []
    for pg in sorted(set(p for p in pages if 1 <= p <= doc.page_count)):
        fp = os.path.join(outdir, f"page_{pg:03d}.png")
        doc[pg - 1].get_pixmap(dpi=dpi).save(fp)
        out.append(fp)
    doc.close()
    return out


# ============================================================================ build
SECTIONS = [sec_k1, sec_i4, sec_i1, sec_c1a, sec_s1, sec_x1, sec_x2, sec_xlcal, sec_x3, sec_c1b, sec_x4, sec_i3, sec_vllm]


def compose(ctx: Ctx) -> str:
    d = Doc(ctx)
    ch_cover(d, ctx)
    ch_summary(d, ctx)
    ch_toc(d, ctx)
    ch_background(d, ctx)
    ch_env(d, ctx)
    ch_lit(d, ctx)
    d.h(1, "ch4", "4　實驗結果")
    d.p("每一節的固定順序：狀態 → 要回答的問題 → 作法 → 主要圖表 → 事先寫死的判定與結果 → 解讀（主 session 撰寫）→ 限制 → run_id 與檔案出處。"
        "圖表的數字全部在產生報告時從結果檔讀；沒有結果的畫 NOT_MEASURED 佔位。〔模擬〕＝用實測常數算的、不是量測。")
    for fn in SECTIONS:
        try:
            fn(d, ctx)
        except Exception as ex:                  # noqa: BLE001
            ctx.warn(fn.__name__, f"整節產生失敗：{type(ex).__name__}: {ex}\n{traceback.format_exc(limit=4)}")
            d.raw(f"<div class='box-nm'><b>NOT_MEASURED（這一節產生失敗）</b>：{esc(type(ex).__name__)}: {esc(ex)}</div>")
    for fn in (sec_synthesis, sec_candidates, sec_limits, sec_appendix):
        try:
            fn(d, ctx)
        except Exception as ex:                  # noqa: BLE001
            ctx.warn(fn.__name__, f"整章產生失敗：{type(ex).__name__}: {ex}\n{traceback.format_exc(limit=4)}")
            d.raw(f"<div class='box-nm'><b>NOT_MEASURED（這一章產生失敗）</b>：{esc(type(ex).__name__)}: {esc(ex)}</div>")
    body = d.html().replace("<!--BUILDINFO-->", build_info_html(ctx))
    return body


def build(ctx: Ctx, make_pdf=True, preview_pages=None) -> dict:
    t0 = time.time()
    for f in glob.glob(os.path.join(ctx.figdir, "fig*.png")):
        os.remove(f)
    body = compose(ctx)
    reg = os.path.join(ctx.fonts_dir, FONT_FILES[0])
    bold = os.path.join(ctx.fonts_dir, FONT_FILES[1])
    cmap = font_cmap(reg) & font_cmap(bold)
    body, repl = sanitize_html(body, cmap)
    if repl:
        ctx.warn("字型", f"字型裡沒有的字元已替換：{dict((f'U+{ord(k):04X}', v) for k, v in repl.items())}")
    fdir = os.path.join(ctx.out, "fonts")
    os.makedirs(fdir, exist_ok=True)
    toc_probe, _ = sanitize_html(toc_html(ctx, {}), cmap)
    text = collect_chars(body, toc_probe, FOOTER_TMPL.format(t=TITLE, n=0, tot=0))
    fi = [subset_font(reg, os.path.join(fdir, "tc_regular.ttf"), text), subset_font(bold, os.path.join(fdir, "tc_bold.ttf"), text)]
    css = CSS_TMPL
    page_html = ("<!doctype html><html><head><meta charset='utf-8'><title>" + TITLE + "</title><style>" + css +
                 "</style></head><body>" + body + "</body></html>")
    hp = os.path.join(ctx.out, "stage3_report.html")
    open(hp, "w", encoding="utf-8").write(page_html)
    log = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=ctx.now.isoformat(timespec="seconds"), git=git_head(), out=ctx.out,
               res=ctx.res, html=hp, fonts=fi, chars=len(text), replaced_chars={f"U+{ord(k):04X}": v for k, v in repl.items()},
               figures=[{k: v for k, v in f.items() if k != "png"} for f in ctx.figs], tables=ctx.tables, not_measured=ctx.nm,
               warnings=ctx.warnings, narrative=dict(ctx.narr_status),
               sources={k: [dict(path=rel(p), run_ids=r) for p, r in v] for k, v in ctx.sources.items()})
    if make_pdf:
        pdf = os.path.join(ctx.out, "stage3_report.pdf")
        rinfo = render_pdf(ctx, body, css, pdf, cmap)
        ver = verify_pdf(pdf)
        log.update(pdf=pdf, render=rinfo, verify=ver)
        hp_ = rinfo["headings"]
        figpage = {f["title"]: hp_.get(f"fig{f['n']:02d}") for f in ctx.figs}
        want = [1, 2, hp_.get("toc"), hp_.get("exp_k1"), hp_.get("ch7"), ver["pages"]] + list(preview_pages or [])
        want += [figpage.get(t) for t in ("第三階段 job 的實際執行時間軸", "X2＋X2b：答對率對 E0 的差 vs 壓縮倍數（32K、128K）",
                                          "X3：各層連線下的 32K 還原時間（BF16、INT4、最好的組合；閒／忙）", "C1b：答對率 vs 搬運比例（各選法、k、補載與否）",
                                          "X4a：倍數 vs teacher-forcing 品質", "各實驗在門檻內能壓到的倍數")]
        log["previews"] = previews(pdf, os.path.join(ctx.out, "preview"), [w for w in want if w])
    log["seconds"] = round(time.time() - t0, 1)
    json.dump(log, open(os.path.join(ctx.out, "build_log.json"), "w"), indent=1, ensure_ascii=False, default=str)
    return log


def cmd_build(a) -> int:
    ctx = Ctx(a.res, a.docs, a.out, a.fonts, a.jobs, narr=a.narr, dpi=a.dpi)
    log = build(ctx, make_pdf=not a.no_pdf, preview_pages=a.preview)
    print("=" * 70)
    print(f"HTML：{log['html']}")
    if "pdf" in log:
        v = log["verify"]
        print(f"PDF：{log['pdf']}（{v['pages']} 頁、{v['bytes'] / 1e6:.2f} MB、書籤 {v['toc_entries']} 個）")
        print(f"字型：{sorted(v['fonts'])}；不該出現的字型：{v['bad_fonts'] or '無'}；頁碼缺的頁：{v['footer_missing'] or '無'}")
        print(f"預覽：{os.path.join(ctx.out, 'preview')}（{len(log['previews'])} 張）")
    nf = sum(f["status"] != "ok" for f in ctx.figs)
    nt = sum(t["status"] != "ok" for t in ctx.tables)
    print(f"圖 {len(ctx.figs)} 張（NOT_MEASURED／讀取失敗 {nf}）、表 {len(ctx.tables)} 張（{nt}）；讀檔警告 {len(ctx.warnings)} 則")
    miss = [k for k, v in ctx.narr_status.items() if v != "ok"]
    print(f"narrative 待撰寫（{len(miss)}）：{', '.join(miss) or '無'}")
    print(f"build_log：{os.path.join(ctx.out, 'build_log.json')}；花 {log['seconds']} 秒")
    if "pdf" in log and not log["verify"]["ok"]:
        print("驗收沒過（出現 fallback 字型或缺頁碼）→ exit 1（規則 2）", file=sys.stderr)
        return 1
    if a.strict and (ctx.warnings or nf or nt):
        return 2
    return 0


# ============================================================================ --self-test（CPU、合成資料）
def _synth_results(res: str):
    """最小的合成結果檔（欄位照真的格式；數字是假的，只驗流程）。"""
    def wcsv(p, rows):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    rid = "20990101-000000-selftest"
    rows = []
    for s in ("0", "1"):
        for g in ("idle", "busy2"):
            for lay in ("scatter", "contig"):
                for i, m in enumerate(("py_loop", "foreach", "coalesce", "stage", "zerocopy")):
                    rows.append(dict(GiBps=str(1.0 + 3 * i), busy_stats="", bytes="1", contaminated="False", cpu_s_med="1", gpu_state=g,
                                     host_level="X", kv_bad="", layout=lay, method=m, n_blocks="1", n_frag="1", reps="5", run_id=rid,
                                     scaled_from="", sdma=s, status="ok", t_med_s="1", t_q1="1", t_q3="1", ts="t", us_per_frag=str(10.0 / (i + 1))))
        wcsv(os.path.join(res, "i4_copy_batching", f"copybench_sdma{s}.csv"), [r for r in rows if r["sdma"] == s])
    fx_ = []
    for m in MODELS:
        for k, (spec, ratio, dlt) in enumerate((("q:int4_kivi", 0.3125, -0.01), ("xl:k22:v34+q:int4_kivi", 0.096, -0.012),
                                                 ("drop:KNORM:0.75", 0.254, -0.27), ("merge:4", 0.25, -0.7))):
            fx_.append(dict(model=m, x3_segment="", feasible="1" if dlt > -0.02 else "0", run_id=rid, ts="t", combo=spec, ratio=str(ratio),
                            acc="0.7", acc_e0="0.75", acc_delta=str(dlt), acc_delta_lo=str(dlt - 0.02), acc_delta_hi=str(dlt + 0.02), eps="0.02",
                            mlperf99="0", mlperf999="0", n_items="288", src_run_ids=rid, source="D"))
    wcsv(os.path.join(res, "x_stack", "v2", "x2_feasible.csv"), fx_[:4] + fx_[4:6])
    wcsv(os.path.join(res, "x_stack", "v2", "x2_unmapped.csv"), fx_[6:] + fx_[2:4])
    cells = []
    for m in MODELS:
        for meth in C1B_METHODS:
            for k in (10, 25, 50):
                for v in ("a", "b"):
                    cells.append(dict(run_id=rid, ts="t", model=m, group="T2-T4", cell=f"{meth}@{k}{v}", method=meth, k_pct=str(k), variant=v,
                                      kind="x", n="135", acc="0.7", diff_mean=str(-0.3 + k / 200), diff_lo=str(-0.35 + k / 200), diff_hi=str(-0.25 + k / 200),
                                      ratio_moved=str(k / 100 + 0.02), needle_cov_init="0.5", needle_cov_final="0.6", n_refill_blocks="1",
                                      oracle_selection="0"))
    wcsv(os.path.join(res, "c1_selective_restore", "c1b_cells.csv"), cells)
    # X4（格式照 code/m16_x4.py 的 cmd_report／cmd_lossless）
    x4 = os.path.join(res, "x_stack", "x4")
    for m in MODELS:
        wcsv(os.path.join(x4, f"x4a_configs_{m}.csv"),
             [dict(run_id=rid, ts="t", model=m, spec=f"xl:k{k}:v{int(k * 1.5)}+q:int4_kivi", roles="alloc:int4", W=w, rk=k, rv=int(k * 1.5), bits=b,
                   drop_method="", drop_frac=df, n=96, ratio=1 / mlt, multiple=mlt, d_gold_lp=-0.02 * mlt, d_gold_lp_lo=-0.03 * mlt, d_gold_lp_hi=0,
                   d_first_ok=0, d_first_ok_lo=0, d_first_ok_hi=0, kl_median=0.1, relerr_k=0.1, relerr_v=0.2)
              for k, w, b, df, mlt in ((8, 4, "int4", 0, 16.0), (11, 2, "int4", 0, 12.0), (16, 4, "bf16", 0, 8.0), (6, 4, "int2", 0.125, 30.0))])
    cfg = {"xl:k8:v12+q:int4_kivi": dict(multiple=16.0, diff=dict(mean=-0.01, lo=-0.03, hi=0.01), eps_pass=False),
           "q:int4_kivi": dict(multiple=3.2, diff=dict(mean=0.0, lo=-0.01, hi=0.01), eps_pass=True)}
    lim = dict(eps=dict(spec="q:int4_kivi", ratio=0.3125, multiple=3.2, acc=0.7, diff=dict(mean=0.0, lo=-0.01, hi=0.01)), mlperf99=None, mlperf999=None)
    M = dict(x4a=dict(n_items=96, configs={}, qa={"drop": dict(desc="輕度丟", by_T={"16": dict(verdict="值得疊（沒有比較差）")},
                                                         summary=dict(n_compared=1, worth=1, not_worth=0))}),
             gen={"32K": dict(n_items=192, configs=cfg, limits=lim, first_T_out_of_eps=16),
                  "128K": dict(n_items=40, configs=cfg, limits=dict(status="INCOMPLETE（40 題 < 48）"), holds={})},
             x4d={"1": dict(n_items=96, configs=cfg), "2": dict(n_items=96, configs=cfg, vs_drop={"q:int4_kivi": dict(mean=0.1, lo=0.05, hi=0.15)}),
                  "verdict_turn2": {"q:int4_kivi": "多輪不壞", "xl:k8:v12+q:int4_kivi": "壓縮傷多輪"}})
    os.makedirs(x4, exist_ok=True)
    json.dump(dict(run_id=rid, rule_text={"Q-A": "合成", "limit": "合成"}, eps=0.02, targets=[8, 12, 16], src_run_ids=[rid],
                   contention=dict(rows_with_intruders=0, host_levels=["X"]), models={m: M for m in MODELS}),
              open(os.path.join(x4, "x4_report.json"), "w"))
    wcsv(os.path.join(x4, "x4_best_allocation.csv"),
         [dict(run_id=rid, ts="t", model=m, T=T, tf_best="xl:k8:v12+q:int4_kivi", tf_multiple=T, tf_d_gold_lp=-0.1, gen_best="xl:k8:v12+q:int4_kivi",
               gen_multiple=T, gen_acc=0.7, gen_diff=-0.01, gen_diff_lo=-0.03, gen_diff_hi=0.01, eps=False, mlperf99=False, mlperf999=False,
               lossless_ratio="", multiple_with_lossless="") for m in MODELS for T in (8, 16)])
    json.dump(dict(run_id=rid, head_codec="zstd:3", worth=1.05, summary={m: {"xl:k8:v12+q:int4_kivi": dict(
        ratio={"zstd:3": 1.08}, worth_stacking=True, decomp_MBps_median={"zstd:3|t1": 900.0})} for m in MODELS}),
        open(os.path.join(x4, "lossless_x4.json"), "w"))
    # 38k（格式照 code/m16_xlcal3.py 的 cmd_report）
    v3 = os.path.join(res, "x_stack", "v3")
    os.makedirs(v3, exist_ok=True)
    pr = {"xl:k22:v34:cal+q:int4_kivi vs xl:k22:v34+q:int4_kivi": dict(cal_minus_ref=dict(mean=0.01, lo=-0.01, hi=0.03), verdict="CAN_PRECOMPUTE",
                                                                      amortized_ratio={"1": 0.096, "8": 0.072, "64": 0.069})}
    var = {k: dict(models={m: dict(n_items=192, pairs=pr, verdict="CAN_PRECOMPUTE") for m in MODELS}) for k in ("old96", "new192", "merged288")}
    json.dump(dict(run_id=rid, verdict="CAN_PRECOMPUTE", verdict_by_model={m: "CAN_PRECOMPUTE" for m in MODELS},
                   verdict_merged288_reference={m: "CAN_PRECOMPUTE" for m in MODELS}, rule_text={"caveat": "合成"}, src_run_ids_new=[rid],
                   variants=var, contention_new=dict(rows_with_intruders=0, host_levels=["X"])), open(os.path.join(v3, "xlcal3_report.json"), "w"))
    # X3 validate（格式照 code/m16_xtime.py 的 cmd_validate）
    vd = os.path.join(res, "x3_time", "v2", "validate")
    wcsv(os.path.join(vd, "validate_cells.csv"),
         [dict(run_id=rid, ts="t", cell=f"{m}|b32|0.0", mode="X", model=m, tier="b32", link_GiBps=1.5, duty=0.0, gpu_state="idle", action="raw",
               role="D", how="harness", pred_s=1.0 + i * 0.3, meas_s=1.05 + i * 0.3, rel_err=0.05, meas_detail="", src="", contaminated="False")
          for i, m in enumerate(MODELS)])
    json.dump(dict(run_id=rid, holds=False, bad_cells=[], gain_AF_measured=0.06, verdict="STOP-3 不成立", n_contaminated=0),
              open(os.path.join(vd, "validate_summary.json"), "w"))
    # V0、F1（格式照 code/m16_vllm_bench.py 的 v0check／run）
    os.makedirs(os.path.join(res, "v0_vllm_smoke"), exist_ok=True)
    json.dump(dict(run_id=rid, mode="apc", overall="PASS", verdict={"gpu_hit_match": "PASS"}, args=dict(label="v0b-rnd"), guard=dict(verdict="CLEAN")),
              open(os.path.join(res, "v0_vllm_smoke", f"v0check_{rid}.json"), "w"))
    f1 = os.path.join(res, "f1_prefix_reuse")
    os.makedirs(f1, exist_ok=True)
    for i, (cfg_, cap, ov) in enumerate((c, k, o) for c in ("off", "apc") for k in ("ample", "c040") for o in ("0", "0.5", "1.0")):
        json.dump(dict(run_id=f"{rid}f{i}", args=dict(label=f"f1_8192_{cfg_}_{cap}_{ov}_burst"), ttft_s=dict(p50=1.0, p95=2.0 - float(ov))),
                  open(os.path.join(f1, f"summary_{rid}f{i}.json"), "w"))
    json.dump(dict(L=8192, t1=0.5, lambda_sat=1.8, rates=[0.9, 1.62], slo_ttft=[0.75, 2.5], sat_run=rid), open(os.path.join(f1, "calib_8192.json"), "w"))
    os.makedirs(os.path.join(res, "f2_shared_prefix_burst"), exist_ok=True)
    for i, (cfg, P, N) in enumerate((c, p, n) for c in ("A", "B") for p in (10240, 32768) for n in (1, 8, 64)):
        json.dump(dict(run_id=f"{rid}{i}", args=dict(label=f"f2_{cfg}_P{P}_N{N}_r1"), ttft_s=dict(p50=0.1 * N, p95=0.2 * N),
                       burst=dict(makespan_s=0.3 * N), server={}), open(os.path.join(res, "f2_shared_prefix_burst", f"summary_{rid}{i}.json"), "w"))


def self_test() -> bool:
    ok_all = []

    def check(name, cond, **kw):
        ok_all.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name + ("  " + json.dumps(kw, ensure_ascii=False, default=str)[:260] if kw else ""), flush=True)

    # 0. 不碰 GPU
    try:
        __import__("torch")
        check("import_blocker_torch", False, why="torch 被 import 了")
    except ImportError as ex:
        check("import_blocker_torch", "不准 import" in str(ex), msg=str(ex)[:80])
    base = os.environ.get("M16_REPORT_SELFTEST_DIR") or os.path.join(DEFAULTS["out"], "selftest_" + now_utc().strftime("%Y%m%dT%H%M%SZ"))
    os.makedirs(base, exist_ok=True)
    reg = os.path.join(DEFAULTS["fonts"], FONT_FILES[0])
    # 1. 字元收集
    s = "<p title='zz'>A&lt;B &amp; 中文&#x2212;（全形）≤</p><td>κ</td>"
    cc = collect_chars(s)
    check("collect_chars_entities_cjk", all(c in cc for c in "<&中文−（）≤κ第頁／") and "\n" not in cc, n=len(cc))
    # 2. 字型沒有的字元
    cmap = font_cmap(reg)
    out, repl = sanitize_html("<b>a✗b​c😀d</b><!--TOC--><i>x&amp;y</i>", cmap)
    check("sanitize_missing_glyphs", "a×bc?d" in out and "<!--TOC-->" in out and "x&amp;y" in out and len(repl) == 3,
          out=out, repl={f"U+{ord(k):04X}": v for k, v in repl.items()})
    # 3. 子集＋CFF 轉 TrueType
    fdir = os.path.join(base, "fonts")
    os.makedirs(fdir, exist_ok=True)
    txt = collect_chars("測試報告：位元組 ≤ ≥ − × Δ ε κ 〔模擬〕NOT_MEASURED（待撰寫）第 1／2 頁")
    fi = subset_font(reg, os.path.join(fdir, "tc_regular.ttf"), txt)
    fb = subset_font(os.path.join(DEFAULTS["fonts"], FONT_FILES[1]), os.path.join(fdir, "tc_bold.ttf"), txt)
    sub_cmap = font_cmap(os.path.join(fdir, "tc_regular.ttf"))
    check("subset_ttf_structure", fi["has_glyf"] and not fi["has_cff"] and fi["maxp"] == "0x10000" and fb["has_glyf"] and not fb["has_cff"],
          reg=fi, bold={k: fb[k] for k in ("bytes", "glyphs")})
    check("subset_sfnt_truetype", open(os.path.join(fdir, "tc_regular.ttf"), "rb").read(4) == b"\x00\x01\x00\x00")
    check("subset_cmap_covers_text", all(ord(c) in sub_cmap for c in txt if ord(c) in cmap), missing=[c for c in txt if ord(c) in cmap and ord(c) not in sub_cmap])
    check("subset_small", fi["glyphs"] < len(txt) + 10 and fi["bytes"] < 2_000_000, glyphs=fi["glyphs"], bytes=fi["bytes"])
    # 4. Fallback 檢查（含反向對照：子集外的字一定要被抓到）
    import pymupdf
    css = CSS_TMPL

    def simple_pdf(html_, path):
        arch = pymupdf.Archive(base)
        st = pymupdf.Story(html=html_, user_css=css, archive=arch)
        w = pymupdf.DocumentWriter(path)
        more, n = 1, 0
        while more:
            dev = w.begin_page(pymupdf.paper_rect("a4"))
            more, _ = st.place(pymupdf.paper_rect("a4") + (50, 50, -50, -60))
            st.draw(dev)
            n += 1
            w.end_page()
        w.close()
        doc = pymupdf.open(path)
        tot = doc.page_count
        for i, p in enumerate(doc):
            p.insert_htmlbox(pymupdf.Rect(50, 790, 545, 830), FOOTER_TMPL.format(t="測試", n=i + 1, tot=tot), css=css, archive=arch)
        doc.save(path + ".2.pdf", garbage=4, deflate=True)
        doc.close()
        return verify_pdf(path + ".2.pdf")
    v1 = simple_pdf("<h3>測試報告</h3><p>位元組 ≤ ≥ − × Δ ε κ 〔模擬〕<b>NOT_MEASURED</b>（待撰寫）</p>", os.path.join(base, "t_ok.pdf"))
    check("pdf_only_our_fonts", v1["ok"] and not v1["bad_fonts"] and all(k.startswith(FONT_FAMILY_PDF) for k in v1["fonts"]), fonts=v1["fonts"])
    v2 = simple_pdf("<p>測試 龘 字</p>", os.path.join(base, "t_bad.pdf"))
    check("pdf_fallback_detected_negative_control", not v2["ok"] and v2["bad_fonts"], bad=v2["bad_fonts"])
    # 5. runner log 與失敗清單
    logs = os.path.join(base, "jobs")
    os.makedirs(logs, exist_ok=True)
    open(os.path.join(logs, "runner_timing.log"), "w").write(
        "[runner:s3:timing] START 00_smoke 獨佔兩張卡 need=both cards=1 need_gb=80 2026-10-20T13:59:24Z\n"
        "[runner:s3:timing] FAILED 00_smoke rc=1 2026-10-20T13:59:46Z\n"
        "[runner:s3:timing] START 00_smoke 獨佔兩張卡 need=both cards=1 need_gb=80 2026-10-20T14:00:51Z\n"
        "[runner:s3:timing] DONE 00_smoke 2026-10-20T14:05:19Z\n"
        "[runner:s3:timing] START 37a_x2_gen 獨佔兩張卡 need=one cards=0,1 need_gb=150 2026-10-20T21:41:38Z\n")
    open(os.path.join(logs, "runner_cpu.log"), "w").write(
        "[runner:s3:cpu] START 06_k1_kvstruct need=none 2026-10-20T14:28:05Z\n[runner:s3:cpu] DONE 06_k1_kvstruct 2026-10-20T14:39:01Z\n")
    js = parse_runner_logs([os.path.join(logs, "runner_timing.log"), os.path.join(logs, "runner_cpu.log")])
    st = sorted((j["name"], j["status"], j["kind"]) for j in js)
    check("runner_log_parse", st == sorted([("00_smoke", "失敗", "量時間（獨佔兩卡）"), ("00_smoke", "完成", "量時間（獨佔兩卡）"),
                                            ("37a_x2_gen", "跑中", "量品質（兩卡並行）"), ("06_k1_kvstruct", "完成", "CPU 分析")]), got=st)
    fl = parse_failures("- **失敗 1**：A 壞了\n  修法：改。\n- 其他\n- **失敗 2**（10/21）：B 壞了\n")
    check("parse_failures", [f["n"] for f in fl] == [1, 2] and fl[0]["more"] == ["修法：改。"], got=fl)
    # 6. 整份報告：合成結果＋空目錄 → NOT_MEASURED 佔位、待撰寫框、草稿提示、PDF 驗收
    res = os.path.join(base, "res")
    _synth_results(res)
    narr = os.path.join(base, "narr")
    os.makedirs(narr, exist_ok=True)
    open(os.path.join(narr, "summary.html"), "w").write("<p>自測摘要：這一段來自 narrative 檔。</p><style>p{color:red}</style>")
    open(os.path.join(narr, "k1.draft.html"), "w").write("<p>草稿</p>")
    ctx = Ctx(res, DEFAULTS["docs"], os.path.join(base, "out"), DEFAULTS["fonts"], logs, narr=narr, dpi=90)
    log = build(ctx, make_pdf=True)
    figst = {f["title"]: f["status"] for f in ctx.figs}
    errs = [f for f in ctx.figs if f["status"] == "ERROR"] + [t for t in ctx.tables if t["status"] == "ERROR"]
    check("build_no_read_errors", not errs and not [w for w in ctx.warnings if "整節" in w["msg"] or "整章" in w["msg"]],
          errors=[(e["title"], e.get("status")) for e in errs], warnings=[w["msg"][:120] for w in ctx.warnings][:5])
    want_ok = ["I4：各搬法的有效 GiB/s", "X2＋X2b：答對率對 E0 的差 vs 壓縮倍數（32K、128K）", "C1b：答對率 vs 搬運比例（各選法、k、補載與否）",
               "F2：共用前綴的併發突發（#44294 重現）", "第三階段 job 的實際執行時間軸", "X4a：倍數 vs teacher-forcing 品質",
               "X4b／X4c：生成答對率 vs 倍數（32K、128K、256K）", "X3 validate：還原時間的實測 vs 預測", "F1：前綴復用地圖（TTFT P95 vs 重疊）",
               "xKV 校準基底 − 每題 SVD 的配對差"]
    check("synthetic_figures_ok", all(figst.get(t) == "ok" for t in want_ok), got={t: figst.get(t) for t in want_ok})
    tabst = {t["title"]: t["status"] for t in ctx.tables}
    want_t = ["X4：目標倍數 → 最佳配比", "X4：極限", "X4 Q-A：哪一軸值得疊（TF）", "X4d：多輪（第 2 輪換問題）", "X4：前緣配置的實際位元組再做無損壓縮",
              "V0：vLLM 功能閘門", "F1：校準（λ_sat、SLO）", "xKV 校準基底的判定"]
    check("synthetic_tables_ok", all(tabst.get(t) == "ok" for t in want_t), got={t: tabst.get(t) for t in want_t})
    check("missing_results_not_measured", figst.get("K1：每（層, head）的低秩重建誤差 vs rank") == "NOT_MEASURED"
          and figst.get("S1：BF16 KV 的無損壓縮比") == "NOT_MEASURED" and tabst.get("F2：三種 evict 情形的探測") == "NOT_MEASURED"
          and len(ctx.nm) > 10, n_nm=len(ctx.nm))
    h = open(log["html"], encoding="utf-8").read()
    check("html_placeholders", "NOT_MEASURED" in h and "（待撰寫）" in h and "自測摘要" in h and "k1.draft.html" in h
          and "color:red" not in h, narr=log["narrative"])
    check("narrative_status", log["narrative"].get("summary") == "ok" and log["narrative"].get("k1") == "draft_only"
          and log["narrative"].get("x2") == "missing")
    v = log["verify"]
    check("report_pdf_verified", v["ok"] and v["pages"] > 10 and not v["bad_fonts"] and not v["footer_missing"] and v["toc_entries"] > 10,
          pages=v["pages"], fonts=v["fonts"], toc=v["toc_entries"])
    ok = all(ok_all)
    print(f"self-test：{sum(ok_all)}／{len(ok_all)} 項；{'ALL PASS' if ok else 'FAIL'}（輸出：{base}）")
    if ok and not os.environ.get("M16_REPORT_SELFTEST_KEEP"):
        shutil.rmtree(base, ignore_errors=True)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("build", help="產生全部圖表、HTML、PDF")
    b.add_argument("--out", default=DEFAULTS["out"])
    b.add_argument("--res", default=DEFAULTS["res"])
    b.add_argument("--docs", default=DEFAULTS["docs"])
    b.add_argument("--narr", default=None, help="narrative 目錄（預設 <docs>/report/narrative）")
    b.add_argument("--fonts", default=DEFAULTS["fonts"])
    b.add_argument("--jobs", default=DEFAULTS["jobs"])
    b.add_argument("--dpi", type=int, default=170)
    b.add_argument("--no-pdf", action="store_true", help="只出圖與 HTML")
    b.add_argument("--strict", action="store_true", help="有讀檔警告或 NOT_MEASURED 時 exit 2")
    b.add_argument("--preview", type=int, nargs="*", default=[], help="另外要轉成 PNG 的頁碼")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if a.cmd == "build":
        sys.exit(cmd_build(a))
    ap.print_help()
    sys.exit(2)


if __name__ == "__main__":
    main()
