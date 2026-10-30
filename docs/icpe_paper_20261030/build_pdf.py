#!/usr/bin/env python3
"""Render PAPER.md (Traditional Chinese, GitHub-flavoured markdown) to an A4 PDF.

Uses only what is already installed in the project conda env: markdown_it for
parsing, matplotlib (PDF backend) for drawing, fontTools (a matplotlib
dependency) for font subsetting, PIL for image downsampling.  No pandoc, LaTeX,
browser or extra packages.

    bash -c 'source code/env_h200.sh >/dev/null 2>&1; \
             python docs/icpe_paper_20261030/build_pdf.py'

Options:  --in PAPER.md  --out PAPER.pdf  --max-bytes 1000000
          --png-pages 1,3,7|all --png-dir DIR --png-dpi 110   (PNG previews of pages)

Pipeline: markdown -> block tree -> flat list of vertical boxes (text line,
table row, figure, math line, spacing) -> greedy pagination with
keep-with-next chains and repeated table headers -> one matplotlib Figure per
page -> PdfPages.  If the PDF is larger than --max-bytes, raster images are
downsampled / colour-quantised in memory (never on disk) and the PDF is rebuilt.

Fonts: Noto Sans TC is CFF-flavoured OpenType, which matplotlib can only embed
as Type 3 (large).  The used glyphs are therefore subset and converted to
TrueType in a temporary directory (deleted afterwards) and embedded as Type 42.
If that fails the original OTF is used with Type 3.

Math: $...$ and $$...$$ are pulled out before markdown parsing (no
mdit_py_plugins installed) and drawn with matplotlib mathtext; \\text -> \\mathrm,
bare CJK in math is wrapped in \\mathrm.  Expressions mathtext cannot parse are
drawn as raw TeX in monospace and listed in the report.

Rendering-only substitutions (the markdown is never edited): '⏳' -> '【待決】'
(no installed font has U+23F3); any other glyph missing from every font in
the chain is drawn as '[U+XXXX]' and reported.
"""
from __future__ import annotations

import argparse
import io
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib import _text_helpers  # noqa: E402
from matplotlib.ft2font import LoadFlags  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Rectangle, FancyBboxPatch, PathPatch, Polygon, Circle, Ellipse  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.mathtext import MathTextParser  # noqa: E402
from markdown_it import MarkdownIt  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------- page geometry (pt)
MM = 72.0 / 25.4
PAGE_W, PAGE_H = 210 * MM, 297 * MM
MARGIN = 18 * MM
TEXT_X = MARGIN
TEXT_W = PAGE_W - 2 * MARGIN
BODY_TOP = MARGIN
BODY_BOTTOM = PAGE_H - MARGIN
BODY_H = BODY_BOTTOM - BODY_TOP
FOOTER_Y = PAGE_H - 9 * MM

BODY_SIZE, BODY_LEAD = 9.5, 15.2
TABLE_SIZE, TABLE_LEAD = 8.0, 11.0
CAPTION_SIZE, CAPTION_LEAD = 9.0, 14.0
HEAD = {1: (16.0, 22.0, 0.0, 6.0), 2: (12.5, 17.0, 13.0, 4.0), 3: (10.5, 15.0, 8.0, 3.0)}
PARA_GAP, ITEM_GAP = 5.0, 1.5
FIG_MAX_H = 0.60 * BODY_H
MATH_BLOCK_SCALE = 1.2

COL_TEXT, COL_LINK, COL_CODE = "#1b1b1b", "#1d4f91", "#2b2f36"
CODE_BG, QUOTE_BG, QUOTE_BAR = "#eef0f3", "#f5f6f8", "#9aa4b2"
TH_BG, ZEBRA_BG, GRID, GRID_DARK = "#e6e9ee", "#fafbfc", "#a8adb5", "#5a5f66"

SUBST = [("⏳待決", "【待決】"), ("⏳", "【待決】")]  # rendering-only substitutions
EXTRA_CHARS = ("【待決】•◦▪[]U+0123456789ABCDEF/ ∑∏√∞≤≥≠±×÷−∈∉⊂⊆∪∩→←↔⇒·…′∂∇≈∝"
               "αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩ")
REPORT = {"subst": Counter(), "missing": Counter(), "svg_skipped": Counter(), "notes": [],
          "math_ok": 0, "math_failed": [], "math_glyph": Counter()}

CJK_FAMILY = "Noto Sans TC"
KNOWN_FAMILIES = set()
BASE_DIR = HERE


# ======================================================================= fonts
def _fc_files(family):
    try:
        out = subprocess.run(["fc-list", f":family={family}", "--format", "%{file}\n"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    return [p.strip() for p in out.split("\n") if p.strip() and os.path.exists(p.strip())]


def _otf_to_ttf(src, dst, chars, family, sub):
    """Subset `src` to `chars`, convert CFF outlines to TrueType, rename the family."""
    from fontTools.ttLib import TTFont, newTable
    from fontTools.subset import Subsetter, Options
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    f = TTFont(src)
    opt = Options()
    opt.hinting, opt.desubroutinize, opt.notdef_outline = False, True, True
    opt.name_IDs = ["*"]
    ss = Subsetter(opt)
    ss.populate(unicodes=sorted({ord(c) for c in chars}))
    ss.subset(f)
    if "CFF " in f:
        order = f.getGlyphOrder()
        gs = f.getGlyphSet()
        glyf = newTable("glyf")
        glyf.glyphOrder, glyf.glyphs = order, {}
        for n in order:
            pen = TTGlyphPen(gs)
            gs[n].draw(Cu2QuPen(pen, 1.0, reverse_direction=True))
            glyf.glyphs[n] = pen.glyph()
        f["loca"] = newTable("loca")
        f["glyf"] = glyf
        del f["CFF "]
        if "VORG" in f:
            del f["VORG"]
        glyf.compile(f)
        hm = f["hmtx"]
        for n, g in glyf.glyphs.items():
            if hasattr(g, "xMin"):
                hm[n] = (hm[n][0], g.xMin)
        mx = newTable("maxp")
        mx.tableVersion = 0x00010000
        for k in ("maxZones", "maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs",
                  "maxStackElements", "maxSizeOfInstructions", "maxComponentElements"):
            setattr(mx, k, 0)
        mx.maxZones = 1
        f["maxp"] = mx
        post = f["post"]
        post.formatType, post.extraNames, post.mapping, post.glyphOrder = 2.0, [], {}, order
        f["head"].glyphDataFormat = 0
        f.sfntVersion = "\x00\x01\x00\x00"
    nm = f["name"]
    for rec in list(nm.names):
        if rec.nameID in (1, 16):
            nm.setName(family, rec.nameID, rec.platformID, rec.platEncID, rec.langID)
        elif rec.nameID in (2, 17):
            nm.setName(sub, rec.nameID, rec.platformID, rec.platEncID, rec.langID)
        elif rec.nameID == 4:
            nm.setName(f"{family} {sub}", 4, rec.platformID, rec.platEncID, rec.langID)
        elif rec.nameID == 6:
            nm.setName(family.replace(" ", "") + "-" + sub, 6, rec.platformID, rec.platEncID, rec.langID)
    f.save(dst)


def setup_fonts(chars, tmpdir):
    """Register fonts with matplotlib; returns the pdf.fonttype to use."""
    global CJK_FAMILY, KNOWN_FAMILIES
    for fam in ("Noto Sans TC", "Liberation Sans", "Liberation Mono"):
        for p in _fc_files(fam):
            try:
                fm.fontManager.addfont(p)
            except Exception as e:  # pragma: no cover
                REPORT["notes"].append(f"addfont {p}: {e}")
    KNOWN_FAMILIES = {f.name for f in fm.fontManager.ttflist}
    if "Noto Sans TC" not in KNOWN_FAMILIES:
        sys.exit("ERROR: font family 'Noto Sans TC' not found (fc-list); Chinese would render as boxes")
    fonttype = 42
    try:
        for weight, sub in (("normal", "Regular"), ("bold", "Bold")):
            src = fm.findfont(fm.FontProperties(family="Noto Sans TC", weight=weight), fallback_to_default=False)
            dst = os.path.join(tmpdir, f"NotoSansTC-PDF-{sub}.ttf")
            _otf_to_ttf(src, dst, chars, "Noto Sans TC PDF", sub)
            fm.fontManager.addfont(dst)
        CJK_FAMILY = "Noto Sans TC PDF"
    except Exception as e:
        REPORT["notes"].append(f"TrueType conversion failed ({e!r}); embedding Noto Sans TC as Type 3")
        CJK_FAMILY, fonttype = "Noto Sans TC", 3
    KNOWN_FAMILIES = {f.name for f in fm.fontManager.ttflist}
    return fonttype


def _font_paths(prop):
    try:
        return list(fm.fontManager._find_fonts_by_props(prop))
    except AttributeError:  # private API changed: resolve one family at a time
        out = []
        for fam in prop.get_family():
            p = fm.FontProperties(family=fam, weight=prop.get_weight(), style=prop.get_style())
            f = fm.findfont(p, fallback_to_default=False)
            if f not in out:
                out.append(f)
        return out


class Face:
    """A font chain (primary + fallbacks) exactly as matplotlib will use it for drawing."""

    def __init__(self, families, weight="normal", style="normal"):
        self.families, self.weight, self.style = list(families), weight, style
        self.prop = fm.FontProperties(family=self.families, weight=weight, style=style)
        paths = _font_paths(self.prop)
        self.ft = fm.get_font(paths)
        self.cover = set()
        for p in paths:
            self.cover |= set(fm.get_font(p).get_charmap().keys())
        self._w = {}

    def kw(self):
        return dict(fontfamily=self.families, fontweight=self.weight, fontstyle=self.style)

    def width(self, s, size):
        key = (s, size)
        w = self._w.get(key)
        if w is None:
            ft = self.ft
            ft.set_size(size, 72)
            last = None
            for it in _text_helpers.layout(s, ft):
                last = it
            if last is None:
                w = 0.0
            else:
                g = last.ft_object.load_glyph(last.glyph_index, flags=LoadFlags.NO_HINTING)
                w = last.x + g.linearHoriAdvance / 65536.0
            self._w[key] = w
        return w


_FACE_CACHE = {}


def get_face(families, weight="normal", style="normal"):
    key = (tuple(families), weight, style)
    if key not in _FACE_CACHE:
        _FACE_CACHE[key] = Face(families, weight, style)
    return _FACE_CACHE[key]


def face(bold=False, italic=False, code=False):
    if code:
        fams = ["DejaVu Sans Mono", CJK_FAMILY, "DejaVu Sans"]
    elif italic:
        fams = (["Liberation Sans"] if "Liberation Sans" in KNOWN_FAMILIES else ["DejaVu Sans"]) + [CJK_FAMILY, "DejaVu Sans"]
    else:
        fams = [CJK_FAMILY, "DejaVu Sans"]
    return get_face(fams, "bold" if bold else "normal", "italic" if italic else "normal")


def render_text(s, fc):
    for a, b in SUBST:
        if a in s:
            REPORT["subst"][f"{a}->{b}"] += s.count(a)
            s = s.replace(a, b)
    out = []
    for ch in s:
        if ch in "\n\t " or ord(ch) in fc.cover:
            out.append(ch)
        else:
            REPORT["missing"][f"U+{ord(ch):04X}"] += 1
            out.append(f"[U+{ord(ch):04X}]")
    return "".join(out)


# ======================================================================= math
PH_IN = re.compile("\ue000(\\d+)\ue001")   # inline-math placeholder
PH_BL = re.compile("\ue002(\\d+)\ue003")   # display-math placeholder
MATH = {}                                  # id -> TeX source
_MT_PARSER = None


class _MathLog(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if "does not have a glyph" in msg or "No TeX to Unicode" in msg:
            REPORT["math_glyph"][msg[:120]] += 1


def _inline_math_line(line):
    """Replace $..$ / $$..$$ inside one line (outside code spans) with placeholders."""
    parts = re.split(r"(`+[^`]*`+)", line)
    for k in range(0, len(parts), 2):
        seg = parts[k]

        def sub(m):
            i = len(MATH)
            MATH[i] = m.group(1)
            return f"\ue000{i}\ue001"
        seg = re.sub(r"(?<!\\)\$\$(.+?)(?<!\\)\$\$", sub, seg)
        seg = re.sub(r"(?<![\\$])\$(?![\s$])([^$\n]*?[^\s\\$])\$(?![\d$])", sub, seg)
        parts[k] = seg
    return "".join(parts)


def extract_math(text):
    lines = text.split("\n")
    out, i, fence = [], 0, None
    while i < len(lines):
        ln = lines[i]
        mf = re.match(r"^\s*(?:>\s*)*(`{3,}|~{3,})", ln)
        if fence:
            out.append(ln)
            if mf and mf.group(1)[0] == fence[0] and len(mf.group(1)) >= len(fence):
                fence = None
            i += 1
            continue
        if mf:
            fence = mf.group(1)
            out.append(ln)
            i += 1
            continue
        m = re.match(r"^(\s*(?:>\s*)*)\$\$(.*)$", ln)
        if m:
            prefix, rest = m.group(1), m.group(2)
            body, j, closed = [], i, False
            if "$$" in rest:
                body, closed = [rest.split("$$", 1)[0]], True
            else:
                body = [rest]
                j = i + 1
                plen = len(prefix.rstrip())
                while j < len(lines):
                    l2 = lines[j]
                    l2 = l2[len(prefix):] if l2.startswith(prefix) else (l2[plen:] if l2[:plen].strip() == prefix.strip() else l2)
                    if "$$" in l2:
                        body.append(l2.split("$$", 1)[0])
                        closed = True
                        break
                    body.append(l2)
                    j += 1
            if closed:
                k = len(MATH)
                MATH[k] = "\n".join(body).strip()
                blank = prefix.rstrip() if prefix.strip() else ""
                out += [blank, f"{prefix}\ue002{k}\ue003", blank]
                i = j + 1
                continue
        out.append(_inline_math_line(ln))
        i += 1
    return "\n".join(out)


def _wrap_cjk(t):
    """Wrap bare CJK runs in \\mathrm{} (CJK inside \\mathrm/\\mathbf/... is left alone)."""
    out, pos = [], 0
    for m in re.finditer(r"\\math(?:rm|bf|sf|tt|it)\{[^{}]*\}", t):
        out.append(re.sub(r"([\u2e80-\u9fff\uff00-\uffef]+)", r"\\mathrm{\1}", t[pos:m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(re.sub(r"([\u2e80-\u9fff\uff00-\uffef]+)", r"\\mathrm{\1}", t[pos:]))
    return "".join(out)


def prep_tex(t):
    t = t.strip()
    sp = lambda s: re.sub(r"(?<!\\) ", r"\\ ", s)  # noqa: E731  (mathtext drops bare spaces)
    t = re.sub(r"\\(?:text|textrm|textnormal|mbox|operatorname\*?)\s*\{([^{}]*)\}", lambda m: r"\mathrm{" + sp(m.group(1)) + "}", t)
    t = re.sub(r"\\textbf\s*\{([^{}]*)\}", lambda m: r"\mathbf{" + sp(m.group(1)) + "}", t)
    t = re.sub(r"\\textit\s*\{([^{}]*)\}", lambda m: r"\mathit{" + sp(m.group(1)) + "}", t)
    t = re.sub(r"\\texttt\s*\{([^{}]*)\}", lambda m: r"\mathtt{" + sp(m.group(1)) + "}", t)
    t = re.sub(r"\\mathrm\{([^{}]*)\}", lambda m: r"\mathrm{" + sp(m.group(1)) + "}", t)
    for a, b in ((r"\le", r"\leq"), (r"\ge", r"\geq"), (r"\ne", r"\neq"), (r"\dfrac", r"\frac"), (r"\tfrac", r"\frac"),
                 (r"\lvert", "|"), (r"\rvert", "|"), (r"\lVert", r"\Vert"), (r"\rVert", r"\Vert"),
                 (r"\displaystyle", ""), (r"\textstyle", ""), (r"\limits", ""), (r"\nolimits", ""),
                 (r"\lbrace", r"\{"), (r"\rbrace", r"\}"), (r"\coloneqq", ":="), (r"\eqqcolon", "=:")):
        t = re.sub(re.escape(a) + r"(?![A-Za-z])", lambda m, b=b: b, t)
    t = re.sub(r"\\tag\*?\{[^{}]*\}", "", t)                  # display tags are drawn separately
    t = re.sub(r"\\[Bb]igg?[lrm]?(?![A-Za-z])", "", t)        # \big( \Big) ...: keep the delimiter only
    t = _wrap_cjk(t)
    return t


def math_metrics(tex, size):
    """(width, ascent, descent) of $tex$ at `size`, or None if mathtext cannot parse it."""
    global _MT_PARSER
    if _MT_PARSER is None:
        _MT_PARSER = MathTextParser("path")
    try:
        r = _MT_PARSER.parse(f"${tex}$", 72, fm.FontProperties(size=size))
    except Exception:
        return None
    return float(r.width), float(r.height - r.depth), float(r.depth)


def display_lines(src):
    t = re.sub(r"\\(?:begin|end)\{(?:aligned|align\*?|gather\*?|gathered|split|equation\*?|multline\*?|eqnarray\*?)\}", "", src)
    parts = [p.replace("&", "").strip() for p in re.split(r"\\\\(?:\[[^\]]*\])?", t)]
    return [p for p in parts if p]


# ======================================================================= inline -> atoms
def is_cjk(ch):
    o = ord(ch)
    return (0x2E80 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7AF or 0xF900 <= o <= 0xFAFF
            or 0xFE30 <= o <= 0xFE4F or 0xFF00 <= o <= 0xFFEF or 0x2460 <= o <= 0x24FF
            or o in (0x2014, 0x2026))


NO_START = set("，。、；：？！）」』】〕〉》’”．,.;:?!)]}／～…・｝％")
NO_END = set("（「『【〔〈《‘“([{｛")


class Atom:
    __slots__ = ("text", "kind", "face", "size", "color", "code", "w", "asc", "dsc")

    def __init__(self, text, kind, fc, size, color, code, metrics=None):
        self.text, self.kind, self.face, self.size, self.color, self.code = text, kind, fc, size, color, code
        self.asc = self.dsc = 0.0
        if kind == "math":
            self.w, self.asc, self.dsc = metrics
        else:
            self.w = fc.width(text, size) if kind != "br" else 0.0

    def skey(self):
        return (id(self.face), self.size, self.color, self.code)


def math_atoms(src, fc, size, color):
    """Atoms for one inline-math source: a math atom, or raw TeX in monospace if it fails."""
    tex = prep_tex(src)
    met = math_metrics(tex, size)
    if met is not None:
        REPORT["math_ok"] += 1
        return [Atom(tex, "math", fc, size, color, False, met)]
    REPORT["math_failed"].append(src)
    cf = face(code=True)
    return [Atom(c, "cjk", cf, size * 0.88, COL_CODE, True) for c in render_text(f"${src}$", cf)]


def inline_runs(children):
    """markdown_it inline children -> [[text, bold, italic, code, link]]"""
    runs, bold, ital, link = [], 0, 0, 0
    for t in children or []:
        tp = t.type
        if tp == "text":
            runs.append([t.content, bold > 0, ital > 0, False, link > 0])
        elif tp == "code_inline":
            # math placeholders cannot occur in code spans (extraction skips them)
            runs.append([t.content, bold > 0, ital > 0, True, link > 0])
        elif tp == "strong_open":
            bold += 1
        elif tp == "strong_close":
            bold -= 1
        elif tp == "em_open":
            ital += 1
        elif tp == "em_close":
            ital -= 1
        elif tp == "link_open":
            link += 1
        elif tp == "link_close":
            link -= 1
        elif tp == "softbreak":
            runs.append(["\x00SOFT", False, False, False, False])
        elif tp == "hardbreak":
            runs.append(["\n", bold > 0, ital > 0, False, link > 0])
        elif tp == "image":
            runs.append([f"[圖：{t.content or 'image'}]", bold > 0, ital > 0, False, link > 0])
        elif tp == "html_inline":
            runs.append([t.content, False, False, True, False])
        elif tp in ("s_open", "s_close"):
            pass
        else:
            REPORT["notes"].append(f"inline token {tp} rendered as text")
            runs.append([t.content or "", bold > 0, ital > 0, False, link > 0])
    for i, r in enumerate(runs):  # soft break: space between Latin words, nothing next to CJK
        if r[0] == "\x00SOFT":
            prev = runs[i - 1][0][-1:] if i > 0 and runs[i - 1][0] else ""
            nxt = runs[i + 1][0][:1] if i + 1 < len(runs) and runs[i + 1][0] else ""
            r[0] = "" if (prev and is_cjk(prev)) or (nxt and is_cjk(nxt)) else " "
            r[1:] = runs[i - 1][1:] if i > 0 else [False, False, False, False]
    return runs


def make_atoms(children, size, color=COL_TEXT, force_bold=False):
    atoms = []
    for text, bold, ital, code, link in inline_runs(children):
        fc = face(bold or force_bold, ital, code)
        sz = size * (0.88 if code else 1.0)
        col = COL_LINK if link else (COL_CODE if code else color)
        # split out math placeholders first
        pieces = re.split("(\ue000\\d+\ue001|\ue002\\d+\ue003)", text)
        for piece in pieces:
            m = re.fullmatch("[\ue000\ue002](\\d+)[\ue001\ue003]", piece)
            if m:
                atoms += math_atoms(MATH[int(m.group(1))], fc, size, col)
                continue
            piece = render_text(piece, fc)
            buf = ""

            def flush():
                nonlocal buf
                if buf:
                    atoms.append(Atom(buf, "word", fc, sz, col, code))
                    buf = ""
            for ch in piece:
                if ch == "\n":
                    flush()
                    atoms.append(Atom("", "br", fc, sz, col, code))
                elif ch in " \t":
                    flush()
                    if atoms and atoms[-1].kind == "space":
                        continue
                    atoms.append(Atom(" ", "space", fc, sz, col, code))
                elif is_cjk(ch):
                    flush()
                    atoms.append(Atom(ch, "cjk", fc, sz, col, code))
                else:
                    buf += ch
            flush()
    return atoms


def _can_break(a, b):
    if a.kind == "space" or b.kind == "space":
        return True
    if a.kind == "cjk" or b.kind == "cjk":
        if (b.kind == "cjk" and b.text[:1] in NO_START) or (a.kind == "cjk" and a.text[-1:] in NO_END):
            return False
        if a.text == "—" and b.text == "—":
            return False
        return True
    return False


def _chunks(atoms):
    out = []
    for a in atoms:
        if a.kind in ("space", "br"):
            out.append([])
        elif out and out[-1] and not _can_break(out[-1][-1], a):
            out[-1].append(a)
        else:
            out.append([a])
    return [c for c in out if c]


def break_lines(atoms, maxw):
    """Greedy line breaking.  Returns list of lines; a line is [(x, atom), ...]."""
    chunks = []
    for a in atoms:
        if a.kind in ("space", "br"):
            chunks.append([a])
        elif chunks and chunks[-1][-1].kind not in ("space", "br") and not _can_break(chunks[-1][-1], a):
            chunks[-1].append(a)
        else:
            chunks.append([a])
    lines, cur, curw, pend = [], [], 0.0, None

    def flush():
        nonlocal cur, curw, pend
        while cur and cur[-1].kind == "space":
            cur.pop()
        line, x = [], 0.0
        for a in cur:
            line.append((x, a))
            x += a.w
        lines.append(line)
        cur, curw, pend = [], 0.0, None

    def add_chars(chunk):  # hard-split an over-long chunk character by character
        nonlocal cur, curw
        for a in chunk:
            if a.kind == "math":
                b = a
                if a.w > maxw:  # shrink an over-wide formula to fit
                    tex, s2 = a.text, max(5.0, a.size * maxw / a.w)
                    met = math_metrics(tex, s2)
                    if met:
                        b = Atom(tex, "math", a.face, s2, a.color, False, met)
                if cur and curw + b.w > maxw + 0.3:
                    flush()
                cur.append(b)
                curw += b.w
                continue
            for ch in a.text:
                b = Atom(ch, "cjk", a.face, a.size, a.color, a.code)
                if cur and curw + b.w > maxw + 0.3:
                    flush()
                cur.append(b)
                curw += b.w

    for ch in chunks:
        k = ch[0].kind
        if k == "br":
            flush()
            continue
        if k == "space":
            if cur:
                pend = ch[0]
            continue
        cw = sum(a.w for a in ch)
        sw = pend.w if (pend is not None and cur) else 0.0
        if cur and curw + sw + cw <= maxw + 0.3:
            if pend is not None:
                cur.append(pend)
                curw += pend.w
            cur.extend(ch)
            curw += cw
        else:
            if cur:
                flush()
            if cw > maxw + 0.3:
                add_chars(ch)
            else:
                cur.extend(ch)
                curw = cw
        pend = None
    if cur or not lines:
        flush()
    return lines


def line_width(line):
    return (line[-1][0] + line[-1][1].w) if line else 0.0


def _baseline(lead, size):
    return lead / 2 + 0.36 * size


def line_metrics(line, size, lead):
    """(height, baseline offset) of a line; grows to fit tall inline math."""
    b0 = _baseline(lead, size)
    asc = max((a.asc for _, a in line if a.kind == "math"), default=0.0)
    dsc = max((a.dsc for _, a in line if a.kind == "math"), default=0.0)
    if asc <= b0 - 1.5 and dsc <= lead - b0 - 1.5:
        return lead, b0
    bl = max(b0, asc + 1.5)
    return max(lead, bl + max(lead - b0, dsc + 1.5)), bl


def line_ops(line, x0, baseline, w=None, align="left"):
    """Draw ops for one laid-out line; baseline is relative to the box top."""
    ops = []
    if not line:
        return ops
    lw = line_width(line)
    if align == "right" and w is not None:
        x0 += w - lw
    elif align == "center" and w is not None:
        x0 += (w - lw) / 2
    st = {"run": "", "rx": 0.0, "key": None, "proto": None}

    def emit():
        run, proto = st["run"], st["proto"]
        if proto is not None and run.strip():
            if proto.code:
                cw = proto.face.width(run, proto.size)
                ops.append(("rect", x0 + st["rx"] - 1.0, baseline - proto.size * 0.86, cw + 2.0,
                            proto.size * 1.18, CODE_BG, None, 0))
            ops.append(("text", x0 + st["rx"], baseline, run, proto.face, proto.size, proto.color))
        st.update(run="", key=None, proto=None)
    for x, a in line:
        if a.kind == "math":
            emit()
            ops.append(("math", x0 + x, baseline, a.text, a.size, a.color))
            continue
        k = a.skey()
        if k != st["key"]:
            emit()
            st.update(run=a.text, rx=x, key=k, proto=a)
        else:
            st["run"] += a.text
    emit()
    return ops


# ======================================================================= markdown -> blocks
def parse_blocks(tokens, i=0, end=None):
    blocks = []
    while i < len(tokens):
        t = tokens[i]
        tp = t.type
        if end and tp == end:
            return blocks, i + 1
        if tp == "heading_open":
            blocks.append(("heading", int(t.tag[1]), tokens[i + 1].children))
            i += 3
        elif tp == "paragraph_open":
            ch = tokens[i + 1].children or []
            raw = (tokens[i + 1].content or "").strip()
            imgs = [c for c in ch if c.type == "image"]
            rest = [c for c in ch if c.type not in ("image", "softbreak") and not (c.type == "text" and not c.content.strip())]
            mb = PH_BL.fullmatch(raw)
            if mb:
                blocks.append(("math", MATH[int(mb.group(1))]))
            elif imgs and not rest:
                for im in imgs:
                    blocks.append(("image", im.attrGet("src"), im.content))
            else:
                blocks.append(("para", ch))
            i += 3
        elif tp in ("bullet_list_open", "ordered_list_open"):
            ordered = tp.startswith("ordered")
            start = int(t.attrGet("start") or 1)
            close = tp.replace("_open", "_close")
            items = []
            i += 1
            while tokens[i].type != close:
                if tokens[i].type == "list_item_open":
                    sub, i = parse_blocks(tokens, i + 1, "list_item_close")
                    items.append(sub)
                else:
                    i += 1
            blocks.append(("list", ordered, start, items))
            i += 1
        elif tp == "blockquote_open":
            sub, i = parse_blocks(tokens, i + 1, "blockquote_close")
            blocks.append(("quote", sub))
        elif tp == "table_open":
            rows, in_head, row = [], False, None
            i += 1
            while tokens[i].type != "table_close":
                tt = tokens[i]
                if tt.type == "thead_open":
                    in_head = True
                elif tt.type == "thead_close":
                    in_head = False
                elif tt.type == "tr_open":
                    row = {"head": in_head, "cells": []}
                elif tt.type == "tr_close":
                    rows.append(row)
                elif tt.type in ("th_open", "td_open"):
                    st = tt.attrGet("style") or ""
                    al = "right" if "right" in st else ("center" if "center" in st else "left")
                    row["cells"].append((al, tokens[i + 1].children))
                i += 1
            if rows:
                blocks.append(("table", rows))
            i += 1
        elif tp == "hr":
            blocks.append(("hr",))
            i += 1
        elif tp in ("fence", "code_block"):
            blocks.append(("code", t.content))
            i += 1
        elif tp == "html_block":
            blocks.append(("code", t.content))
            REPORT["notes"].append("html_block rendered as preformatted text")
            i += 1
        else:
            i += 1
    return blocks, i


# ======================================================================= boxes
class Box:
    __slots__ = ("h", "ops", "keep", "glue", "header", "baseline", "decor")

    def __init__(self, h, ops=None, keep=False, glue=False, header=None, baseline=None):
        self.h, self.ops, self.keep, self.glue = h, list(ops or []), keep, glue
        self.header, self.baseline, self.decor = header, baseline, ()


class Flow:
    def __init__(self):
        self.boxes = []
        self.pending_keep = 0

    def add(self, box, decor):
        if decor:
            pre = []
            for (bx0, bx1, barx) in decor:
                pre.append(("rect", bx0, 0, bx1 - bx0, box.h, QUOTE_BG, None, 0))
                pre.append(("rect", barx, 0, 2.2, box.h, QUOTE_BAR, None, 0))
            box.ops = pre + box.ops
            box.decor = tuple(decor)
        if self.pending_keep and not box.glue:
            box.keep = True
            self.pending_keep -= 1
        self.boxes.append(box)
        return box

    def glue(self, h, decor):
        if self.boxes and self.boxes[-1].glue and self.boxes[-1].decor == tuple(decor):
            if h > self.boxes[-1].h:
                self.boxes.pop()
                self.add(Box(h, glue=True), decor)
            return
        self.add(Box(h, glue=True), decor)


def lay_lines(flow, atoms, x, w, size, lead, decor, align="left"):
    out = []
    for line in break_lines(atoms, w):
        h, b = line_metrics(line, size, lead)
        out.append(flow.add(Box(h, line_ops(line, x, b, w, align), baseline=b), decor))
    return out


def lay_raw(flow, text, x, w, decor, size=8.0, lead=11.0):
    """Preformatted text (code blocks, unparsable display math) in monospace on a grey band."""
    fc = face(code=True)
    for ln in text.rstrip("\n").split("\n"):
        atoms = [Atom(c, "cjk", fc, size, COL_CODE, False) for c in render_text(ln, fc)]
        for line in (break_lines(atoms, w - 8) if atoms else [[]]):
            b = _baseline(lead, size)
            flow.add(Box(lead, [("rect", x, 0, w, lead, CODE_BG, None, 0)] + line_ops(line, x + 4, b), baseline=b), decor)


def is_caption(children, word):
    ch = children or []
    return (len(ch) >= 2 and ch[0].type == "strong_open" and ch[1].type == "text"
            and re.match(rf"^{word}\s*\d", ch[1].content.strip() or "") is not None)


def lay_math_block(flow, src, x, w, decor):
    lines = display_lines(src)
    prepared = []
    fc = face()
    for ln in lines:
        tg = re.search(r"\\tag\*?\{([^{}]*)\}", ln)
        tag = render_text(f"({tg.group(1)})" if tg and "*" not in tg.group(0) else (tg.group(1) if tg else ""), fc)
        tw = fc.width(tag, BODY_SIZE) + 8.0 if tag else 0.0
        avail = w - 2 * tw  # keep the formula centred clear of the tag
        tex = prep_tex(ln)
        size = BODY_SIZE * MATH_BLOCK_SCALE
        met = math_metrics(tex, size)
        if met and met[0] > avail:
            size = max(5.0, size * avail / met[0])
            met = math_metrics(tex, size)
        if met is None:
            prepared = None
            break
        prepared.append((tex, size, met, tag, tw))
    if not prepared:
        REPORT["math_failed"].append(src)
        lay_raw(flow, "$$\n" + src + "\n$$", x, w, decor)
        return
    REPORT["math_ok"] += 1
    for tex, size, (mw, asc, dsc), tag, tw in prepared:
        h, bl = asc + dsc + 6.0, asc + 3.0
        ops = [("math", x + (w - mw) / 2, bl, tex, size, COL_TEXT)]
        if tag:
            ops.append(("text", x + w - (tw - 8.0), bl, tag, fc, BODY_SIZE, COL_TEXT))
        flow.add(Box(h, ops, baseline=bl), decor)


def lay_blocks(flow, blocks, x, w, decor, tight=False, depth=0):
    gap = ITEM_GAP if tight else PARA_GAP
    prev_para_boxes = None
    for bi, blk in enumerate(blocks):
        kind = blk[0]
        this_para = None
        if kind == "heading":
            lvl = blk[1]
            size, lead, before, after = HEAD.get(lvl, HEAD[3])
            if flow.boxes:
                flow.glue(before, decor)
            atoms = make_atoms(blk[2], size, force_bold=True)
            for b in lay_lines(flow, atoms, x, w, size, lead, decor, align="center" if lvl == 1 else "left"):
                b.keep = True
            flow.glue(after, decor)
            flow.pending_keep = 2
        elif kind == "para":
            ch = blk[1]
            cap = is_caption(ch, "圖") or is_caption(ch, "表")
            size, lead = (CAPTION_SIZE, CAPTION_LEAD) if cap else (BODY_SIZE, BODY_LEAD)
            bxs = lay_lines(flow, make_atoms(ch, size), x, w, size, lead, decor)
            if bxs and bi > 0 and blocks[bi - 1][0] == "image" and len(bxs) > 1:
                bxs[0].keep = True  # figure + first two caption lines stay together
            this_para = bxs
            flow.glue(gap, decor)
        elif kind == "math":
            flow.glue(2.0, decor)
            lay_math_block(flow, blk[1], x, w, decor)
            flow.glue(gap + 2.0, decor)
        elif kind == "list":
            ordered, start, items = blk[1], blk[2], blk[3]
            fc = face()
            labels = [render_text(f"{start + k}." if ordered else ("•", "◦", "▪")[min(depth, 2)], fc)
                      for k in range(len(items))]
            ind = max(13.0, max(fc.width(lb, BODY_SIZE) for lb in labels) + 5.0)
            for k, item in enumerate(items):
                n0 = len(flow.boxes)
                lay_blocks(flow, item, x + ind, w - ind, decor, tight=True, depth=depth + 1)
                first = next((b for b in flow.boxes[n0:] if not b.glue), None)
                if first is not None:
                    bl = first.baseline if first.baseline is not None else BODY_SIZE
                    first.ops.append(("text", x + ind - 4.0 - fc.width(labels[k], BODY_SIZE), bl, labels[k],
                                      fc, BODY_SIZE, COL_TEXT))
            flow.glue(gap, decor)
        elif kind == "quote":
            d2 = list(decor) + [(x, x + w, x)]
            flow.glue(gap, decor)
            flow.add(Box(5.0), d2)
            lay_blocks(flow, blk[1], x + 11.0, w - 17.0, d2, tight=False, depth=depth)
            while flow.boxes and flow.boxes[-1].glue and flow.boxes[-1].decor == tuple(d2):
                flow.boxes.pop()
            flow.add(Box(5.0), d2)
            flow.glue(PARA_GAP, decor)
        elif kind == "table":
            if prev_para_boxes:
                for b in prev_para_boxes[-2:]:
                    b.keep = True  # caption / lead-in stays with the header row
            lay_table(flow, blk[1], x, w, decor)
            flow.glue(PARA_GAP + 2, decor)
        elif kind == "image":
            lay_image(flow, blk[1], x, w, decor)
            flow.glue(4.0, decor)
        elif kind == "hr":
            flow.glue(4.0, decor)
            flow.add(Box(6.0, [("line", x, 3.0, x + w, 3.0, "#c3c7cd", 0.6)]), decor)
            flow.glue(4.0, decor)
        elif kind == "code":
            lay_raw(flow, blk[1], x, w, decor)
            flow.glue(PARA_GAP, decor)
        prev_para_boxes = this_para


def lay_table(flow, rows, x, w, decor):
    ncol = max(len(r["cells"]) for r in rows)
    size = TABLE_SIZE if ncol < 7 else 7.6
    lead = TABLE_LEAD if ncol < 7 else 10.6
    padx, pady = 3.2, 2.6
    cells = []
    for r in rows:
        row = []
        for c in range(ncol):
            al, ch = r["cells"][c] if c < len(r["cells"]) else ("left", [])
            row.append((al, make_atoms(ch, size, force_bold=r["head"])))
        cells.append(row)
    cmin, cmax = [0.0] * ncol, [0.0] * ncol
    for row in cells:
        for c, (al, atoms) in enumerate(row):
            words = [sum(a.w for a in chunk) for chunk in _chunks(atoms)]
            mn = (max(words) if words else 0.0) + 2 * padx
            mx = sum(a.w for a in atoms) + 2 * padx
            cmin[c] = max(cmin[c], min(mn, 0.30 * w))
            cmax[c] = max(cmax[c], mx)
    cmin = [max(mn, min(mx, 0.105 * w)) for mn, mx in zip(cmin, cmax)]
    if sum(cmax) <= w:
        widths = [m * w / sum(cmax) for m in cmax]
    elif sum(cmin) >= w:
        widths = [m * w / sum(cmin) for m in cmin]
    else:
        span = sum(mx - mn for mn, mx in zip(cmin, cmax)) or 1.0
        widths = [mn + (w - sum(cmin)) * (mx - mn) / span for mn, mx in zip(cmin, cmax)]
    xs = [x]
    for cw in widths:
        xs.append(xs[-1] + cw)
    header_box = None
    body_idx = 0
    for ri, (r, row) in enumerate(zip(rows, cells)):
        laid = []
        for c, (al, atoms) in enumerate(row):
            lines = break_lines(atoms, widths[c] - 2 * padx)
            laid.append([(ln, *line_metrics(ln, size, lead)) for ln in lines])
        h = max(sum(m[1] for m in cl) for cl in laid) + 2 * pady
        ops = []
        if r["head"]:
            ops.append(("rect", x, 0, w, h, TH_BG, None, 0))
        else:
            if body_idx % 2 == 1:
                ops.append(("rect", x, 0, w, h, ZEBRA_BG, None, 0))
            body_idx += 1
        for c, cl in enumerate(laid):
            yy = pady
            for ln, lh, bl in cl:
                ops += line_ops(ln, xs[c] + padx, yy + bl, widths[c] - 2 * padx, row[c][0])
                yy += lh
        last = ri == len(rows) - 1
        ops.append(("line", x, 0, x + w, 0, GRID_DARK if (ri == 0 or r["head"]) else GRID, 0.8 if ri == 0 else 0.4))
        ops.append(("line", x, h, x + w, h, GRID_DARK if (r["head"] or last) else GRID, 0.8 if (r["head"] or last) else 0.4))
        for xv in xs:
            ops.append(("line", xv, 0, xv, h, GRID, 0.4))
        if r["head"]:
            header_box = flow.add(Box(h, ops, keep=True), decor)
        else:
            flow.add(Box(h, ops, header=header_box), decor)


# ======================================================================= images
def lay_image(flow, src, x, w, decor):
    path = os.path.join(BASE_DIR, src or "")
    aspect = None
    if os.path.isfile(path):
        try:
            if path.lower().endswith(".svg"):
                vb = parse_svg(path)["vb"]
                aspect, kind = vb[3] / vb[2], "svg"
            else:
                with Image.open(path) as im:
                    pw, ph = im.size
                aspect, kind = ph / pw, "img"
        except Exception as e:
            REPORT["notes"].append(f"image {src} unreadable: {e!r}")
    if aspect is None:
        REPORT["notes"].append(f"image not found or unreadable: {src}")
        flow.add(Box(BODY_LEAD, [("text", x, _baseline(BODY_LEAD, BODY_SIZE), f"[missing image: {src}]",
                                  face(), BODY_SIZE, "#b00020")]), decor)
        return
    dw = w
    dh = dw * aspect
    if dh > FIG_MAX_H:
        dh = FIG_MAX_H
        dw = dh / aspect
    flow.add(Box(dh + 4.0, [(kind, path, x + (w - dw) / 2, 2.0, dw, dh)], keep=True), decor)


_IMG_CACHE = {}


def load_image(path, disp_w_pt, max_dpi, colors):
    key = (path, round(disp_w_pt, 1), max_dpi, colors)
    if key in _IMG_CACHE:
        return _IMG_CACHE[key]
    arr = mpimg.imread(path)  # PNG natively, other formats through PIL
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 4:  # flatten onto white so no soft mask is needed
        a = arr[:, :, 3:4].astype(np.float32) / 255.0
        arr = (arr[:, :, :3].astype(np.float32) * a + 255.0 * (1 - a) + 0.5).astype(np.uint8)
    im = Image.fromarray(np.ascontiguousarray(arr[:, :, :3]))
    target = int(round(disp_w_pt / 72.0 * max_dpi))
    if im.size[0] > target:
        im = im.resize((target, max(1, int(round(im.size[1] * target / im.size[0])))), Image.LANCZOS)
    if colors:
        im = im.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).convert("RGB")
    out = np.asarray(im)
    _IMG_CACHE[key] = out
    return out


# ======================================================================= SVG subset (fallback)
# Covers what .claude/skills/figure-spec/scripts/figure_renderer.py emits: rect (rx),
# path (M/L/H/V/C/S/Q/T/Z; arcs as lines), line, polygon/polyline, circle, ellipse,
# text (size/anchor/weight/fill), marker-end arrows, dasharray, opacity, g + transform.
def _num(v, d=0.0):
    if v is None:
        return d
    m = re.match(r"\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)", str(v))
    return float(m.group(1)) if m else d


def _color(v):
    if v is None or v in ("none", "transparent"):
        return None
    v = v.strip()
    m = re.match(r"rgba?\(([^)]*)\)", v)
    if m:
        parts = [p.strip() for p in m.group(1).split(",")]
        rgb = [float(p[:-1]) * 2.55 if p.endswith("%") else float(p) for p in parts[:3]]
        return tuple([c / 255.0 for c in rgb] + ([float(parts[3])] if len(parts) > 3 else [1.0]))
    if v == "currentColor":
        return (0, 0, 0, 1)
    try:
        return to_rgba(v)
    except ValueError:
        REPORT["svg_skipped"][f"color {v}"] += 1
        return (0, 0, 0, 1)


def _transform(s):
    M = np.eye(3)
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", s or ""):
        a = [float(t) for t in re.split(r"[\s,]+", args.strip()) if t]
        T = np.eye(3)
        if name == "translate":
            T[0, 2], T[1, 2] = a[0], (a[1] if len(a) > 1 else 0.0)
        elif name == "scale":
            T[0, 0], T[1, 1] = a[0], (a[1] if len(a) > 1 else a[0])
        elif name == "rotate":
            r = math.radians(a[0])
            R = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
            if len(a) == 3:
                T = np.array([[1, 0, a[1]], [0, 1, a[2]], [0, 0, 1]]) @ R @ np.array([[1, 0, -a[1]], [0, 1, -a[2]], [0, 0, 1]])
            else:
                T = R
        elif name == "matrix" and len(a) == 6:
            T = np.array([[a[0], a[2], a[4]], [a[1], a[3], a[5]], [0, 0, 1]])
        else:
            REPORT["svg_skipped"][f"transform {name}"] += 1
        M = M @ T
    return M


INHERIT = ("fill", "stroke", "stroke-width", "stroke-dasharray", "font-size", "font-family", "font-weight",
           "font-style", "text-anchor", "marker-end", "fill-opacity", "stroke-opacity")
SVG_HANDLED = {"rect", "line", "path", "polygon", "polyline", "circle", "ellipse", "text"}
_SVG_CACHE = {}


def parse_svg(path):
    if path in _SVG_CACHE:
        return _SVG_CACHE[path]
    root = ET.parse(path).getroot()
    tag = lambda e: e.tag.split("}")[-1]  # noqa: E731
    if root.get("viewBox"):
        vb = [float(v) for v in re.split(r"[\s,]+", root.get("viewBox").strip())]
    else:
        vb = [0.0, 0.0, _num(root.get("width"), 100), _num(root.get("height"), 100)]
    markers = {}
    for m in root.iter():
        if tag(m) == "marker":
            shape = next((c for c in m if tag(c) in ("polygon", "path")), None)
            if shape is None:
                REPORT["svg_skipped"]["marker without polygon/path"] += 1
                continue
            if tag(shape) == "polygon":
                nums = [float(t) for t in re.split(r"[\s,]+", shape.get("points").strip()) if t]
                pts = list(zip(nums[0::2], nums[1::2]))
            else:
                pts = [tuple(v) for v in _parse_path(shape.get("d"))[0]]
            markers[m.get("id")] = dict(pts=pts, fill=_color(shape.get("fill", "black")),
                                        rx=_num(m.get("refX")), ry=_num(m.get("refY")),
                                        units=m.get("markerUnits", "strokeWidth"), orient=m.get("orient", "0"))
    elems, skipped = [], Counter()

    def attrs_of(e, inh):
        a = dict(inh)
        for k in INHERIT:
            if e.get(k) is not None:
                a[k] = e.get(k)
        for decl in (e.get("style") or "").split(";"):
            if ":" in decl:
                k, v = decl.split(":", 1)
                a[k.strip()] = v.strip()
        a["opacity"] = float(inh.get("opacity", 1.0)) * _num(e.get("opacity"), 1.0)
        return a

    def walk(e, inh, M):
        for c in e:
            t = tag(c)
            if t in ("defs", "marker", "title", "desc", "metadata"):
                continue
            a = attrs_of(c, inh)
            M2 = M @ _transform(c.get("transform"))
            if t in ("g", "a", "svg"):
                walk(c, a, M2)
            elif t in SVG_HANDLED:
                elems.append((t, c, a, M2))
            else:
                skipped[t] += 1

    base = {"fill": "black", "stroke": "none", "stroke-width": "1", "font-size": "16",
            "font-family": root.get("font-family", "sans-serif"), "font-weight": "normal",
            "text-anchor": "start", "opacity": 1.0}
    base = attrs_of(root, base)
    base["opacity"] = 1.0
    walk(root, base, np.eye(3))
    for k, v in skipped.items():
        REPORT["svg_skipped"][f"{os.path.basename(path)}: <{k}> x{v}"] += 1
    data = dict(vb=vb, elems=elems, markers=markers)
    _SVG_CACHE[path] = data
    return data


_PATH_TOK = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _parse_path(d):
    """(vertices, codes, end_tangent) for the SVG path subset (arcs -> straight lines)."""
    toks = _PATH_TOK.findall(d or "")
    verts, codes = [], []
    cur = np.array([0.0, 0.0])
    start = cur.copy()
    last_ctrl, tangent = None, (1.0, 0.0)
    cmd, i = None, 0

    def nums(n):
        nonlocal i
        vals = [float(v) for v in toks[i:i + n]]
        i += n
        return vals
    while i < len(toks):
        if re.match(r"[A-Za-z]", toks[i]):
            cmd = toks[i]
            i += 1
            if cmd in "Zz":
                if verts:
                    verts.append(tuple(start))
                    codes.append(MPath.CLOSEPOLY)
                cur = start.copy()
                continue
        if cmd is None:
            break
        rel, C = cmd.islower(), cmd.upper()
        off = cur if rel else np.zeros(2)
        if C == "M":
            p = np.array(nums(2)) + off
            verts.append(tuple(p))
            codes.append(MPath.MOVETO)
            cur = start = p
            cmd = "l" if rel else "L"
            last_ctrl = None
        elif C in "LHV":
            if C == "L":
                p = np.array(nums(2)) + off
            elif C == "H":
                p = np.array([nums(1)[0] + (cur[0] if rel else 0), cur[1]])
            else:
                p = np.array([cur[0], nums(1)[0] + (cur[1] if rel else 0)])
            tangent = tuple(p - cur)
            verts.append(tuple(p))
            codes.append(MPath.LINETO)
            cur, last_ctrl = p, None
        elif C in "CS":
            if C == "C":
                v = nums(6)
                c1 = np.array(v[0:2]) + off
            else:
                v = [0, 0] + nums(4)
                c1 = 2 * cur - last_ctrl if last_ctrl is not None else cur.copy()
            c2, p = np.array(v[2:4]) + off, np.array(v[4:6]) + off
            verts += [tuple(c1), tuple(c2), tuple(p)]
            codes += [MPath.CURVE4] * 3
            tangent = tuple(p - c2)
            cur, last_ctrl = p, c2
        elif C in "QT":
            if C == "Q":
                v = nums(4)
                c1, p = np.array(v[0:2]) + off, np.array(v[2:4]) + off
            else:
                c1 = 2 * cur - last_ctrl if last_ctrl is not None else cur.copy()
                p = np.array(nums(2)) + off
            verts += [tuple(c1), tuple(p)]
            codes += [MPath.CURVE3] * 2
            tangent = tuple(p - c1)
            cur, last_ctrl = p, c1
        elif C == "A":
            p = np.array(nums(7)[5:7]) + off
            REPORT["svg_skipped"]["path arc drawn as straight line"] += 1
            tangent = tuple(p - cur)
            verts.append(tuple(p))
            codes.append(MPath.LINETO)
            cur, last_ctrl = p, None
        else:
            i += 1
    return verts, codes, tangent


def _apply(M, pts):
    P = np.asarray(pts, dtype=float).reshape(-1, 2)
    return (M[:2, :2] @ P.T).T + M[:2, 2]


def draw_svg(fig, path, x, y, w, h):
    svg = parse_svg(path)
    vb = svg["vb"]
    ax = fig.add_axes([x / PAGE_W, 1 - (y + h) / PAGE_H, w / PAGE_W, h / PAGE_H])
    ax.set_xlim(vb[0], vb[0] + vb[2])
    ax.set_ylim(vb[1] + vb[3], vb[1])
    ax.axis("off")
    k = w / vb[2]  # points per user unit
    z = 1
    for t, e, a, M in svg["elems"]:
        z += 1
        det = math.sqrt(abs(np.linalg.det(M[:2, :2]))) or 1.0
        alpha = a.get("opacity", 1.0)
        fill, stroke = _color(a.get("fill")), _color(a.get("stroke"))
        sw = _num(a.get("stroke-width"), 1.0)
        lw = sw * k * det if stroke else 0.0
        ls = "solid"
        if stroke and a.get("stroke-dasharray") not in (None, "none"):
            dl = [float(v) for v in re.split(r"[\s,]+", a["stroke-dasharray"].strip()) if v]
            ls = (0, tuple(v * k * det / max(lw, 1e-3) for v in dl))
        common = dict(facecolor=fill or "none", edgecolor=stroke or "none", linewidth=lw, linestyle=ls,
                      alpha=alpha if alpha < 1 else None, zorder=z)
        if t == "rect":
            rx_, ry_ = _num(e.get("x")), _num(e.get("y"))
            rw, rh = _num(e.get("width")), _num(e.get("height"))
            rr = _num(e.get("rx"), _num(e.get("ry"), 0.0))
            if abs(M[0, 1]) < 1e-9 and abs(M[1, 0]) < 1e-9:
                (x0, y0), (x1, y1) = _apply(M, [(rx_, ry_), (rx_ + rw, ry_ + rh)])
                x0, x1, y0, y1 = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
                if rr > 0:
                    rr = min(rr * abs(M[0, 0]), (x1 - x0) / 2, (y1 - y0) / 2)
                    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                                boxstyle=f"round,pad=0,rounding_size={rr}", **common))
                else:
                    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, **common))
            else:
                ax.add_patch(Polygon(_apply(M, [(rx_, ry_), (rx_ + rw, ry_), (rx_ + rw, ry_ + rh), (rx_, ry_ + rh)]),
                                     closed=True, **common))
        elif t in ("circle", "ellipse"):
            cx, cy = _apply(M, [(_num(e.get("cx")), _num(e.get("cy")))])[0]
            if t == "circle":
                ax.add_patch(Circle((cx, cy), _num(e.get("r")) * det, **common))
            else:
                ax.add_patch(Ellipse((cx, cy), 2 * _num(e.get("rx")) * det, 2 * _num(e.get("ry")) * det, **common))
        elif t in ("polygon", "polyline"):
            nums = [float(v) for v in re.split(r"[\s,]+", (e.get("points") or "").strip()) if v]
            ax.add_patch(Polygon(_apply(M, list(zip(nums[0::2], nums[1::2]))), closed=(t == "polygon"), **common))
            if len(nums) >= 4:
                _marker(ax, svg, a, M, (nums[-2], nums[-1]), (nums[-2] - nums[-4], nums[-1] - nums[-3]), sw, z)
        elif t in ("path", "line"):
            if t == "line":
                p0, p1 = (_num(e.get("x1")), _num(e.get("y1"))), (_num(e.get("x2")), _num(e.get("y2")))
                verts, codes, tang = [p0, p1], [MPath.MOVETO, MPath.LINETO], (p1[0] - p0[0], p1[1] - p0[1])
            else:
                verts, codes, tang = _parse_path(e.get("d"))
            if not verts:
                continue
            ax.add_patch(PathPatch(MPath(_apply(M, verts), codes), **common))
            _marker(ax, svg, a, M, verts[-1], tang, sw, z)
        elif t == "text":
            s = re.sub(r"\s+", " ", "".join(e.itertext())).strip()
            if not s:
                continue
            tx, ty = _apply(M, [(_num(e.get("x")), _num(e.get("y")))])[0]
            fams = [f.strip().strip("'\"") for f in a.get("font-family", "").split(",")]
            fams = [f for f in fams if f in KNOWN_FAMILIES] + [CJK_FAMILY, "DejaVu Sans"]
            wt = a.get("font-weight", "normal")
            bold = wt in ("bold", "bolder") or (wt.isdigit() and int(wt) >= 600)
            fc = get_face(fams, "bold" if bold else "normal", "italic" if a.get("font-style") == "italic" else "normal")
            ha = {"start": "left", "middle": "center", "end": "right"}.get(a.get("text-anchor", "start"), "left")
            ax.text(tx, ty, render_text(s, fc), fontsize=_num(a.get("font-size"), 16) * k * det, color=fill or "black",
                    ha=ha, va="baseline", rotation=-math.degrees(math.atan2(M[1, 0], M[0, 0])),
                    alpha=alpha if alpha < 1 else None, zorder=z, clip_on=False, parse_math=False, **fc.kw())


def _marker(ax, svg, a, M, end, tangent, sw, z):
    ref = a.get("marker-end")
    if not ref or ref == "none":
        return
    m = re.match(r"url\(#([^)]+)\)", ref)
    mk = svg["markers"].get(m.group(1)) if m else None
    if mk is None:
        REPORT["svg_skipped"][f"marker {ref}"] += 1
        return
    s = sw if mk["units"] == "strokeWidth" else 1.0
    ang = math.atan2(tangent[1], tangent[0]) if mk["orient"] in ("auto", "auto-start-reverse") else math.radians(_num(mk["orient"]))
    ca, sa = math.cos(ang), math.sin(ang)
    loc = [(end[0] + ca * (px - mk["rx"]) * s - sa * (py - mk["ry"]) * s,
            end[1] + sa * (px - mk["rx"]) * s + ca * (py - mk["ry"]) * s) for px, py in mk["pts"]]
    ax.add_patch(Polygon(_apply(M, loc), closed=True, facecolor=mk["fill"] or "black", edgecolor="none", zorder=z + 0.5))


# ======================================================================= pagination + drawing
def chain_height(boxes, i):
    need, j = 0.0, i
    while j < len(boxes):
        need += boxes[j].h
        if boxes[j].keep or boxes[j].glue:
            j += 1
            continue
        break
    return need


def paginate(boxes):
    pages, y = [[]], BODY_TOP
    for i, b in enumerate(boxes):
        at_top = not pages[-1]
        room = BODY_BOTTOM - y
        if b.glue:
            if at_top:
                continue
            if b.h > room:
                pages.append([])
                y = BODY_TOP
                continue
            pages[-1].append((y, b))
            y += b.h
            continue
        brk = False
        if not at_top and b.h > room + 0.01:
            brk = True
        elif not at_top and b.keep and room < chain_height(boxes, i) <= BODY_H * 0.9:
            brk = True
        if brk:
            while pages[-1] and pages[-1][-1][1].glue:
                pages[-1].pop()
            pages.append([])
            y = BODY_TOP
            if b.header is not None:
                pages[-1].append((y, b.header))
                y += b.header.h
        pages[-1].append((y, b))
        y += b.h
    return [p for p in pages if p]


def draw_page(page, number, total, img_params):
    fig = plt.figure(figsize=(PAGE_W / 72.0, PAGE_H / 72.0))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(PAGE_H, 0)
    ax.axis("off")
    for y, box in page:
        for op in box.ops:
            kind = op[0]
            if kind == "text":
                _, x, bl, s, fc, size, color = op
                ax.text(x, y + bl, s, fontsize=size, color=color, ha="left", va="baseline",
                        parse_math=False, clip_on=False, zorder=3, **fc.kw())
            elif kind == "math":
                _, x, bl, tex, size, color = op
                ax.text(x, y + bl, f"${tex}$", fontsize=size, color=color, ha="left", va="baseline",
                        parse_math=True, clip_on=False, zorder=3)
            elif kind == "rect":
                _, x, dy, w, h, fcol, ecol, lw = op
                ax.add_patch(Rectangle((x, y + dy), w, h, facecolor=fcol, edgecolor=ecol or "none",
                                       linewidth=lw, zorder=1))
            elif kind == "line":
                _, x0, y0, x1, y1, col, lw = op
                ax.plot([x0, x1], [y + y0, y + y1], color=col, linewidth=lw, solid_capstyle="butt", zorder=2)
            elif kind == "img":
                _, path, x, dy, w, h = op
                iax = fig.add_axes([x / PAGE_W, 1 - (y + dy + h) / PAGE_H, w / PAGE_W, h / PAGE_H])
                iax.imshow(load_image(path, w, *img_params), interpolation="none", aspect="auto")
                iax.axis("off")
            elif kind == "svg":
                _, path, x, dy, w, h = op
                draw_svg(fig, path, x, y + dy, w, h)
    ax.text(PAGE_W / 2, FOOTER_Y, f"{number} / {total}", fontsize=8, color="#6b7078", ha="center",
            va="baseline", parse_math=False, **face().kw())
    return fig


def build(md_path, out_path, max_bytes, png_pages, png_dir, png_dpi):
    global BASE_DIR
    BASE_DIR = os.path.dirname(os.path.abspath(md_path))
    text = open(md_path, encoding="utf-8").read()
    logging.getLogger("matplotlib.mathtext").addHandler(_MathLog())
    logging.getLogger("matplotlib._mathtext").addHandler(_MathLog())
    tmpdir = tempfile.mkdtemp(prefix="build_pdf_fonts_")
    try:
        chars = set(text) | {chr(c) for c in range(32, 127)} | set(EXTRA_CHARS)
        fonttype = setup_fonts(chars, tmpdir)
        matplotlib.rcParams.update({
            "pdf.fonttype": fonttype, "pdf.compression": 9, "axes.unicode_minus": False,
            # PNG previews: unhinted advances, i.e. the same glyph positions the PDF uses
            "text.hinting": "no_hinting", "text.hinting_factor": 1,
            "mathtext.fontset": "custom", "mathtext.rm": CJK_FAMILY, "mathtext.sf": CJK_FAMILY,
            "mathtext.bf": f"{CJK_FAMILY}:bold", "mathtext.it": "DejaVu Sans:italic",
            "mathtext.bfit": "DejaVu Sans:italic:bold", "mathtext.tt": "DejaVu Sans Mono",
            "mathtext.cal": "DejaVu Sans", "mathtext.fallback": "stixsans"})
        md = MarkdownIt("commonmark", {"html": False}).enable(["table", "strikethrough"])
        blocks, _ = parse_blocks(md.parse(extract_math(text)))
        title = next(("".join(c.content for c in b[2] if c.type == "text") for b in blocks if b[0] == "heading"), "PAPER")
        title = PH_IN.sub("", PH_BL.sub("", title))
        flow = Flow()
        lay_blocks(flow, blocks, TEXT_X, TEXT_W, [])
        pages = paginate(flow.boxes)
        total = len(pages)
        has_raster = any(op[0] == "img" for p in pages for _, b in p for op in b.ops)
        levels = [(200, 0), (200, 256), (150, 256), (150, 128), (120, 128), (100, 64)] if has_raster else [(200, 0)]
        data, used = None, None
        for lv in levels:
            buf = io.BytesIO()
            with PdfPages(buf, metadata={"Title": title, "Creator": "build_pdf.py (matplotlib)", "CreationDate": None}) as pdf:
                for n, page in enumerate(pages, 1):
                    fig = draw_page(page, n, total, lv)
                    pdf.savefig(fig)
                    plt.close(fig)
            data, used = buf.getvalue(), lv
            print(f"  images <= {lv[0]} dpi, {f'{lv[1]} colours' if lv[1] else 'full colour'}: {len(data):,} bytes")
            if len(data) < max_bytes:
                break
        with open(out_path, "wb") as f:
            f.write(data)
        if png_pages:
            os.makedirs(png_dir, exist_ok=True)
            for n in (range(1, total + 1) if png_pages == "all" else png_pages):
                if 1 <= n <= total:
                    fig = draw_page(pages[n - 1], n, total, used)
                    fig.savefig(os.path.join(png_dir, f"page_{n:02d}.png"), dpi=png_dpi)
                    plt.close(fig)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    ok = len(data) < max_bytes
    print(f"wrote {out_path}: {total} pages, {len(data):,} bytes, fonts Type {fonttype}"
          f"{'' if ok else '  (WARNING: over --max-bytes)'}")
    print(f"  math: {REPORT['math_ok']} rendered, {len(REPORT['math_failed'])} fell back to raw TeX")
    for src in REPORT["math_failed"]:
        print(f"    raw TeX: {src!r}")
    if REPORT["math_glyph"]:
        print(f"  mathtext glyph warnings: {dict(REPORT['math_glyph'])}")
    print(f"  substitutions (render only): {dict(REPORT['subst']) or 'none'}")
    print(f"  glyphs missing in every font: {dict(REPORT['missing']) or 'none'}")
    if REPORT["svg_skipped"]:
        print(f"  SVG elements skipped/approximated: {dict(REPORT['svg_skipped'])}")
    for n in sorted(set(REPORT["notes"])):
        print(f"  note: {n}")
    return 0 if ok else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="inp", default=os.path.join(HERE, "PAPER.md"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-bytes", type=int, default=1_000_000)
    ap.add_argument("--png-pages", default="", help="'all' or comma-separated 1-based page numbers to save as PNG")
    ap.add_argument("--png-dir", default=".")
    ap.add_argument("--png-dpi", type=int, default=110)
    a = ap.parse_args()
    out = a.out or os.path.splitext(a.inp)[0] + ".pdf"
    pp = "all" if a.png_pages.strip() == "all" else [int(v) for v in a.png_pages.split(",") if v.strip()]
    sys.exit(build(a.inp, out, a.max_bytes, pp, a.png_dir, a.png_dpi))


if __name__ == "__main__":
    main()
