#!/usr/bin/env python3
"""Fig 1 (system model) and Fig 2 (method) for the ICPE paper, drawn with matplotlib patches.

    python diagrams.py --lang zh|en|both      (default: both)

Outputs, next to this file:
    fig1_system_{zh,en}.{png,pdf}   full-width figure* : 7.0 in wide
    fig2_method_{zh,en}.{png,pdf}   single column      : 3.33 in wide
    fig1_system.png, fig2_method.png  (copies of the zh PNGs, embedded by PAPER.md)

Every coordinate is in inches at the final printed size, and every font size is in points,
so what you see in the PNG is what lands in the PDF column.  No text is smaller than 7 pt.
The figures carry no measured numbers except the preregistered thresholds (5 %, 15 %) and the
GPU-replay validation range (0.911-0.987, EVIDENCE.md row B6).  Bar lengths are schematic.
Design notes and the reference figures it borrows from: DIAGRAM_NOTES.md.
"""
import argparse
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- palette (grayscale- and CVD-safe)
INK, INK2, INK3 = "#1a1a1a", "#4d4d4d", "#8c8c8c"
LANE_FC, LANE_EC = "#f3f3f3", "#b3b3b3"
KV_FC, KV_EC = "#d4d4d4", "#4d4d4d"
COMP_FC, COMP_EC = "#f6c46f", "#a8650a"      # GPU recompute: orange + hatch
LOAD_FC, LOAD_EC = "#a3cdee", "#2a6da3"      # load from a tier: blue, solid
ACC, ACC_LT = "#a8155b", "#f5dbe7"           # the one accent = foresight
LEV = "#33414c"                              # levers that need no foresight

FS, FS_S, FS_B, FS_T = 7.5, 7.0, 8.0, 8.5     # pt; FS_S is the floor
MIN_PT = 7.0

# ---------------------------------------------------------------- labels
LABELS = {
    "en": {
        # fig 1
        "a_title": "(a) Where one session's KV lives between two requests",
        "b_title": "(b) The restore step",
        "h1_t": "Write time", "h1_d": "where and at what precision\nto keep each block, by position",
        "h2_t": "Return time", "h2_d": "prefetch, eviction order,\ndrop never-returners",
        "h3_t": "Restore time", "h3_d": "pick the restore method\nper request",
        "lane_gpu": "GPU HBM", "lane_gpu_s": "",
        "lane_cpu": "CPU tier", "lane_cpu_s": "DRAM, BF16",
        "lane_slow": "Slow tier, INT4", "lane_slow_s": "SSD class (throttled)\nNFS class (throttled)",
        "lane_drop": "Dropped",
        "kv": "session KV",
        "gpu_other": "GPU serves other requests",
        "mps": "MPS", "mps_t": "reserve SMs for restore",
        "write": "write all\nblocks",
        "demote": "CPU tier full:\ndemote or drop",
        "int4": "INT4", "int4_t": "fewer bytes",
        "drop_t": "recompute all on return",
        "prefetch": "prefetch",
        "restore_arrow": "restore",
        "restore_bar": "restore", "prefill_bar": "prefill",
        "ttft": "TTFT",
        "ax_k": "turn k", "ax_gap": "between requests", "ax_k1": "turn k+1",
        "blk_first": "first block", "blk_last": "last block", "meet": "meet",
        "recompute": "GPU recomputes", "load": "loads from the tier",
        "tb_head": "Restore time from the slow tier",
        "tb_load": "load all", "tb_rec": "recompute all", "tb_cake": "Cake",
        "tb_note": "shortest",
        "rule": "Base, no prediction:\nCPU tier: load all;  slow tier: Cake",
        "lg_hook": "foresight plug point (an oracle tests it)",
        "lg_lever": "lever that needs no foresight",
        "lg_act": "action that needs foresight",
        "lg_comp": "GPU recompute",
        "lg_load": "load from a tier (I/O)",
        # fig 2
        "tr_head": "Real traces",
        "tr1": "Mooncake\nconversation (chat)", "tr2": "Mooncake\ntoolagent (tools)", "tr3": "FreeInference\n(agent)",
        "sim_t": "Trace-driven simulator", "sim_s": "one request at a time",
        "c_h200": "CPU tier + GPU recompute:\nmeasured on H200",
        "c_slow": "SSD / NFS class: throttled\nemulation (constants from\nan earlier platform)",
        "sim_hook": "return-time foresight",
        "gpu_t": "Measured on the GPU", "gpu_s": "one H200, real runs",
        "gpu_const": "cost constants\nfrom microbenchmarks",
        "gpu_hooks": "write, restore time",
        "gpu_lev": "on vs. off",
        "gpu_val": "scaled-down replay:\nmeasured / simulated TTFT\n0.911–0.987, same ranking",
        "lad_head": "Per decision: four arms on the same trace",
        "lad_sub": "bar = mean TTFT of returning requests (schematic)",
        "arm_base": "base: restore with Cake", "arm_rival": "best rival, no foresight",
        "arm_free": "free signal (typing hint)", "arm_oracle": "oracle (perfect foresight)",
        "headroom": r"headroom $= 1 - T_{\mathrm{oracle}}\,/\,T_{\mathrm{rival}}$",
        "rul_head": "Verdict: the zone the headroom falls in",
        "z1": "dead end", "z2": "possible", "z3": "worth it",
        "rul_ax": "headroom",
        "rul_note": "thresholds preregistered before any result",
    },
    "zh": {
        "a_title": "(a) 一個 session 的 KV 在兩次請求之間停在哪",
        "b_title": "(b) 還原這一步",
        "h1_t": "寫入時", "h1_d": "依位置決定每塊\n放哪層、存什麼精度",
        "h2_t": "回來時", "h2_d": "回來前預取、淘汰順序、\n丟掉不再回來的 session",
        "h3_t": "還原時", "h3_d": "每個請求\n挑還原方式",
        "lane_gpu": "GPU HBM", "lane_gpu_s": "",
        "lane_cpu": "CPU 層", "lane_cpu_s": "DRAM，BF16",
        "lane_slow": "慢層，INT4", "lane_slow_s": "SSD 級（限速模擬）\nNFS 級（限速模擬）",
        "lane_drop": "丟掉",
        "kv": "session 的 KV",
        "gpu_other": "GPU 服務其他請求",
        "mps": "MPS", "mps_t": "替還原保留 SM",
        "write": "整段\n寫入",
        "demote": "CPU 層滿了：\n降層或丟掉",
        "int4": "INT4", "int4_t": "位元組變少",
        "drop_t": "回來時全部重算",
        "prefetch": "預取",
        "restore_arrow": "還原",
        "restore_bar": "還原", "prefill_bar": "prefill",
        "ttft": "TTFT",
        "ax_k": "第 k 回合", "ax_gap": "兩次請求之間", "ax_k1": "第 k+1 回合",
        "blk_first": "第一塊", "blk_last": "最後一塊", "meet": "會合",
        "recompute": "GPU 重算", "load": "儲存層載入",
        "tb_head": "從慢層還原的時間",
        "tb_load": "全部載入", "tb_rec": "全部重算", "tb_cake": "Cake",
        "tb_note": "最短",
        "rule": "底（不預測）：\nCPU 層全部載入；慢層用 Cake",
        "lg_hook": "預知的插入點（用 oracle 檢驗）",
        "lg_lever": "不用預知的旋鈕",
        "lg_act": "需要預知的動作",
        "lg_comp": "GPU 重算",
        "lg_load": "從儲存層載入（I/O）",
        "tr_head": "真實 trace",
        "tr1": "Mooncake\nconversation（chat）", "tr2": "Mooncake\ntoolagent（工具）", "tr3": "FreeInference\n（agent）",
        "sim_t": "trace 驅動的模擬器", "sim_s": "一次一個請求",
        "c_h200": "CPU 層＋GPU 重算：\nH200 實測",
        "c_slow": "SSD／NFS 級：限速模擬\n（常數沿用前一個平台）",
        "sim_hook": "回來時的預知",
        "gpu_t": "在 GPU 上實測", "gpu_s": "一張 H200，實際執行",
        "gpu_const": "微基準量出\n成本常數",
        "gpu_hooks": "寫入時、還原時",
        "gpu_lev": "開與關",
        "gpu_val": "縮小重播：\nTTFT 實測÷模擬\n0.911–0.987，排序相同",
        "lad_head": "每個決策：同一條 trace 跑四種臂",
        "lad_sub": "長條＝回來請求的平均 TTFT（示意）",
        "arm_base": "底：用 Cake 還原", "arm_rival": "最好的對手（不預知）",
        "arm_free": "免費訊號（打字提示）", "arm_oracle": "oracle（完美預知）",
        "headroom": r"headroom $= 1 - T_{\mathrm{oracle}}\,/\,T_{\mathrm{對手}}$",
        "rul_head": "判定：headroom 落在哪一區",
        "z1": "死路", "z2": "可能", "z3": "有看頭",
        "rul_ax": "headroom",
        "rul_note": "門檻在看到任何結果之前登記",
    },
}


# ---------------------------------------------------------------- fonts
def setup_fonts(lang):
    fdir = os.path.expanduser("~/.local/share/fonts")
    for f in ("NotoSansTC-Regular.otf", "NotoSansTC-Bold.otf"):
        p = os.path.join(fdir, f)
        if os.path.exists(p):
            fm.fontManager.addfont(p)
    fam = "Noto Sans TC" if lang == "zh" else "Liberation Sans"
    plt.rcParams.update({
        "font.family": [fam, "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": fam, "mathtext.it": fam + ":italic", "mathtext.bf": fam + ":bold",
        "pdf.fonttype": 42, "ps.fonttype": 42,     # TrueType/CID embedding, no Type 3
        "hatch.linewidth": 0.55,
        "savefig.facecolor": "white",
    })


# ---------------------------------------------------------------- drawing helpers
class Canvas:
    def __init__(self, w, h):
        self.W, self.H = w, h
        self.fig = plt.figure(figsize=(w, h))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(0, h)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self._r = None

    @property
    def renderer(self):
        if self._r is None:
            self._r = self.fig.canvas.get_renderer()
        return self._r

    def text(self, x, y, s, size=FS, color=INK, ha="left", va="center", weight="normal",
             style="normal", z=6, ls=1.15, **kw):
        assert size >= MIN_PT, (s, size)
        return self.ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight,
                            fontstyle=style, zorder=z, linespacing=ls, **kw)

    def text_w(self, t):
        """width and height of a drawn Text in inches"""
        bb = t.get_window_extent(self.renderer)
        return bb.width / self.fig.dpi, bb.height / self.fig.dpi

    def box(self, x, y, w, h, fc="white", ec=INK, lw=0.8, ls="-", r=0.035, z=2, hatch=None):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                           lw=lw, ls=ls, zorder=z, hatch=hatch)
        self.ax.add_patch(p)
        return p

    def rect(self, x, y, w, h, fc="white", ec=INK, lw=0.6, ls="-", z=3, hatch=None):
        p = Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, ls=ls, zorder=z, hatch=hatch)
        self.ax.add_patch(p)
        return p

    def line(self, xs, ys, color=INK, lw=0.8, ls="-", z=3):
        self.ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="butt")

    def arrow(self, p0, p1, color=INK, lw=0.9, ls="-", rad=0.0, z=5, ms=7, both=False, conn=None):
        style = "<|-|>" if both else "-|>"
        a = FancyArrowPatch(p0, p1, arrowstyle=f"{style},head_length=0.55,head_width=0.28",
                            connectionstyle=conn or f"arc3,rad={rad}", color=color, lw=lw, ls=ls,
                            mutation_scale=ms, zorder=z, shrinkA=0, shrinkB=0, capstyle="butt",
                            joinstyle="miter")
        self.ax.add_patch(a)
        return a

    def hook(self, x, y, n, r=0.072, z=9):
        self.ax.add_patch(Circle((x, y), r, fc=ACC, ec="white", lw=0.7, zorder=z))
        self.text(x, y - 0.004, str(n), size=FS, color="white", ha="center", va="center",
                  weight="bold", z=z + 1)

    def lever(self, x, y, s, h=0.14, z=9):
        """dark square-cornered tag with white text; left edge at x, centred at y. returns right edge"""
        t = self.text(x + 0.045, y - 0.003, s, size=FS_S, color="white", weight="bold", z=z + 1)
        w, _ = self.text_w(t)
        self.rect(x, y - h / 2, w + 0.09, h, fc=LEV, ec=LEV, lw=0.6, z=z)
        return x + w + 0.09

    def blocks(self, x, y, n, bw, h, fc=KV_FC, ec=KV_EC, lw=0.5, ls="-", hatch=None, z=4):
        """a bar of n KV blocks, lower-left at (x, y)"""
        for i in range(n):
            self.rect(x + i * bw, y, bw, h, fc=fc, ec=ec, lw=lw, ls=ls, hatch=hatch, z=z)

    def brace_h(self, x0, x1, y, h=0.05, color=INK2, lw=0.7):
        """bracket opening downwards: |‾‾‾‾|"""
        self.line([x0, x0, x1, x1], [y - h, y, y, y - h], color=color, lw=lw, z=6)

    # ---- quality checks: text overlaps, text outside the canvas, fonts below the floor
    def check(self, name):
        r = self.renderer
        fb = self.fig.bbox
        items = []
        for t in self.ax.texts:
            if not t.get_text().strip():
                continue
            bb = t.get_window_extent(r)
            items.append((t.get_text().replace("\n", " / ")[:40], bb))
            if bb.x0 < fb.x0 - 0.5 or bb.x1 > fb.x1 + 0.5 or bb.y0 < fb.y0 - 0.5 or bb.y1 > fb.y1 + 0.5:
                print(f"  [{name}] OUTSIDE canvas: {t.get_text()!r}")
            if t.get_fontsize() < MIN_PT:
                print(f"  [{name}] font < {MIN_PT}pt: {t.get_text()!r}")
        n = 0
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i][1], items[j][1]
                ox = min(a.x1, b.x1) - max(a.x0, b.x0)
                oy = min(a.y1, b.y1) - max(a.y0, b.y0)
                if ox > 1.0 and oy > 1.0:
                    n += 1
                    print(f"  [{name}] text overlap: {items[i][0]!r} <> {items[j][0]!r}")
        print(f"  [{name}] {len(items)} text items, {n} overlaps")

    def save(self, stem):
        png, pdf = stem + ".png", stem + ".pdf"
        self.fig.savefig(png, dpi=200)
        self.fig.savefig(pdf)
        plt.close(self.fig)
        return png, pdf


# ---------------------------------------------------------------- Fig 1
def fig1(L):
    W, H = 7.0, 3.05
    c = Canvas(W, H)
    # geometry ------------------------------------------------------------
    XL0, XL1 = 0.05, 4.56            # lanes, panel (a)
    XLAB = 0.98                      # end of the lane-label column
    XE, XA = 1.74, 3.28              # turn k ends / request k+1 arrives
    LANES = {                        # y0, y1
        "gpu": (1.74, 2.40), "cpu": (1.23, 1.66), "slow": (0.72, 1.15), "drop": (0.36, 0.64)}
    XB0 = 4.80                       # panel (b) left edge
    # panel titles
    c.text(0.06, H - 0.07, L["a_title"], size=FS_T, weight="bold", va="top")
    c.text(XB0, H - 0.07, L["b_title"], size=FS_T, weight="bold", va="top")
    c.line([4.68, 4.68], [0.06, H - 0.06], color="#c8c8c8", lw=0.8, ls=(0, (3, 3)), z=1)

    # lanes ---------------------------------------------------------------
    for k, (y0, y1) in LANES.items():
        dashed = k in ("slow", "drop")
        c.box(XL0, y0, XL1 - XL0, y1 - y0, fc="white" if k == "drop" else LANE_FC, ec=LANE_EC,
              lw=0.8, ls=(0, (3, 2)) if dashed else "-", r=0.03, z=1)
    ym = {k: (y0 + y1) / 2 for k, (y0, y1) in LANES.items()}
    c.text(0.11, ym["gpu"], L["lane_gpu"], size=FS_B, weight="bold")
    c.text(0.11, ym["cpu"] + 0.08, L["lane_cpu"], size=FS_B, weight="bold")
    c.text(0.11, ym["cpu"] - 0.08, L["lane_cpu_s"], size=FS_S, color=INK2)
    c.text(0.11, ym["slow"] + 0.125, L["lane_slow"], size=FS_B, weight="bold")
    c.text(0.11, ym["slow"] - 0.05, L["lane_slow_s"], size=FS_S, color=INK2, ls=1.1)
    c.text(0.11, ym["drop"], L["lane_drop"], size=FS_B, weight="bold")

    # event lines (time of the hooks) ----------------------------------------
    for x in (XE, XA):
        c.line([x, x], [0.30, 2.62], color="#9a9a9a", lw=0.7, ls=(0, (2, 2)), z=2)

    # bottom time axis
    c.arrow((1.0, 0.24), (4.50, 0.24), color=INK2, lw=0.7, ms=6, z=3)
    for x in (XE, XA):
        c.line([x, x], [0.20, 0.28], color=INK2, lw=0.7, z=3)
    c.text((1.0 + XE) / 2, 0.11, L["ax_k"], size=FS_S, color=INK2, ha="center")
    c.text((XE + XA) / 2, 0.11, L["ax_gap"], size=FS_S, color=INK2, ha="center")
    c.text((XA + 4.5) / 2, 0.11, L["ax_k1"], size=FS_S, color=INK2, ha="center")

    # GPU lane ---------------------------------------------------------------
    BW, BH = 0.068, 0.14             # one KV block (BF16)
    yb = 2.07                        # bar bottom in the GPU lane
    c.blocks(1.07, yb, 8, BW, BH)
    c.text(1.07 + 4 * BW, 1.93, L["kv"], size=FS_S, color=INK2, ha="center")
    # other requests during the gap
    c.rect(XE + 0.04, 1.80, XA - XE - 0.08, 0.54, fc="white", ec="#c9c9c9", lw=0.5, z=2, hatch="....")
    c.text((XE + XA) / 2, 2.18, L["gpu_other"], size=FS_S, color=INK2, ha="center",
           bbox=dict(fc="white", ec="none", pad=0.6))
    xm = c.lever(XE + 0.10, 1.93, L["mps"])
    c.text(xm + 0.05, 1.93, L["mps_t"], size=FS_S, color=INK, bbox=dict(fc="white", ec="none", pad=0.4))
    # restore (Cake) + prefill at turn k+1
    xr = XA + 0.07
    nc = 3
    c.blocks(xr, yb, nc, BW, BH, fc=COMP_FC, ec=COMP_EC, hatch="////")
    c.blocks(xr + nc * BW, yb, 8 - nc, BW, BH, fc=LOAD_FC, ec=LOAD_EC)
    c.rect(xr + nc * BW - 0.004, yb - 0.02, 0.008, BH + 0.04, fc=INK, ec=INK, lw=0, z=6)
    xp = xr + 8 * BW + 0.05
    c.rect(xp, yb, 0.26, BH, fc="#7f7f7f", ec=INK2, lw=0.5, z=4)
    xt = xp + 0.26
    c.ax.add_patch(Polygon([(xt + 0.02, yb - 0.01), (xt + 0.02, yb + BH + 0.01), (xt + 0.09, yb + BH / 2)],
                           closed=True, fc=INK, ec=INK, lw=0, zorder=6))
    c.text(xr + 4 * BW, 1.93, L["restore_bar"], size=FS_S, color=INK2, ha="center")
    c.text(xp + 0.13, 1.93, L["prefill_bar"], size=FS_S, color=INK2, ha="center")
    c.brace_h(XA, xt + 0.02, 2.29, h=0.05, color=INK)
    c.text((XA + xt) / 2, 2.30, L["ttft"], size=FS, weight="bold", ha="center", va="bottom")

    # CPU lane ---------------------------------------------------------------
    ycb = ym["cpu"] - BH / 2 + 0.02
    xc0 = XE + 0.08
    c.blocks(xc0, ycb, 8, BW, BH)
    c.arrow((1.07 + 8 * BW + 0.01, yb + BH / 2), (xc0 + 0.04, ycb + BH + 0.005), color=INK, lw=0.9,
            rad=-0.35)
    c.text(XE - 0.07, ym["cpu"] - 0.02, L["write"], size=FS_S, color=INK, ha="right", ls=1.05)

    # slow lane (INT4: fewer bytes -> thinner bar) --------------------------------
    xs0 = xc0 + 8 * BW + 0.10
    ysb = ym["slow"] + 0.05
    c.blocks(xs0, ysb, 8, BW, BH / 3.2, lw=0.4)
    c.arrow((xc0 + 8 * BW - 0.06, ycb), (xs0 + 0.03, ysb + BH / 3.2 + 0.005), color=INK, lw=0.9, rad=0.2)
    xi = c.lever(xs0 + 0.0, ym["slow"] - 0.10, L["int4"])
    c.text(xi + 0.05, ym["slow"] - 0.10, L["int4_t"], size=FS_S, color=INK)
    c.text(xc0 + 0.30, ym["slow"] + 0.02, L["demote"], size=FS_S, color=INK2, ha="right", ls=1.05)
    # drop
    xd = xc0 + 0.40
    c.arrow((xd, ycb), (xd, ym["drop"] + 0.07), color=INK2, lw=0.8, ls=(0, (3, 2)))
    c.text(xd, ym["drop"], "×", size=10, color=INK, ha="center", va="center", weight="bold")
    c.text(xd + 0.09, ym["drop"], L["drop_t"], size=FS_S, color=INK2)

    # prefetch: foresight action (dashed accent), slow -> CPU before the return
    xpf = xs0 + 8 * BW - 0.14
    c.arrow((xpf, ysb + BH / 3.2 + 0.01), (xpf + 0.04, ycb + 0.005), color=ACC, lw=1.0, ls=(0, (2.5, 1.6)),
            rad=-0.15)
    c.rect(xpf + 0.005, ycb, 0.16, BH, fc="white", ec=ACC, lw=0.8, ls=(0, (2, 1.5)), z=4)
    c.text(xpf - 0.04, ym["cpu"] - 0.165, L["prefetch"], size=FS_S, color=ACC, ha="right", style="italic")

    # restore arrow: slow tier -> GPU at the return
    c.arrow((xs0 + 8 * BW + 0.01, ysb + 0.02), (xr + 0.20, yb - 0.005), color=LOAD_EC, lw=1.1, rad=0.28)
    c.text(XA + 0.20, ym["cpu"] - 0.06, L["restore_arrow"], size=FS_S, color=LOAD_EC, ha="left",
           style="italic")

    # hooks row ---------------------------------------------------------------
    yt = 2.74                       # title line of the hooks
    c.hook(XE, yt, 1)
    c.text(XE - 0.11, yt, L["h1_t"], size=FS_B, weight="bold", color=ACC, ha="right")
    c.text(XE - 0.11, yt - 0.09, L["h1_d"], size=FS_S, ha="right", va="top", ls=1.1)
    xh2 = XE + 0.20
    c.hook(xh2, yt, 2)
    c.text(xh2 + 0.11, yt, L["h2_t"], size=FS_B, weight="bold", color=ACC)
    c.text(xh2 - 0.05, yt - 0.09, L["h2_d"], size=FS_S, va="top", ls=1.1)
    c.hook(XA, yt, 3)
    c.text(XA + 0.11, yt, L["h3_t"], size=FS_B, weight="bold", color=ACC)
    c.text(XA + 0.11, yt - 0.09, L["h3_d"], size=FS_S, va="top", ls=1.1)

    # panel (b): Cake ----------------------------------------------------------
    nb, bw = 12, 0.15
    xb = XB0 + 0.12
    ybb, bh = 2.30, 0.16
    ncb = 5
    c.blocks(xb, ybb, ncb, bw, bh, fc=COMP_FC, ec=COMP_EC, hatch="////", lw=0.6)
    c.blocks(xb + ncb * bw, ybb, nb - ncb, bw, bh, fc=LOAD_FC, ec=LOAD_EC, lw=0.6)
    xmeet = xb + ncb * bw
    c.rect(xmeet - 0.006, ybb - 0.04, 0.012, bh + 0.08, fc=INK, ec=INK, lw=0, z=6)
    ya = ybb + bh + 0.10
    c.arrow((xb, ya), (xmeet - 0.06, ya), color=COMP_EC, lw=1.2)
    c.arrow((xb + nb * bw, ya), (xmeet + 0.06, ya), color=LOAD_EC, lw=1.2)
    c.text(xb, ya + 0.09, L["recompute"], size=FS_S, color=INK)
    c.text(xb + nb * bw, ya + 0.09, L["load"], size=FS_S, color=INK, ha="right")
    c.text(xb, ybb - 0.10, L["blk_first"], size=FS_S, color=INK2)
    c.text(xb + nb * bw, ybb - 0.10, L["blk_last"], size=FS_S, color=INK2, ha="right")
    c.text(xmeet, ybb - 0.10, L["meet"], size=FS_S, color=INK, ha="center", weight="bold")

    # restore-time bars (schematic, slow tier)
    c.text(XB0, 1.94, L["tb_head"], size=FS, weight="bold")
    xl, x0 = XB0 + 0.78, XB0 + 0.84
    rows = [(1.75, "tb_load"), (1.53, "tb_rec"), (1.29, "tb_cake")]
    for y, key in rows:
        c.text(xl, y, L[key], size=FS_S, ha="right")
    c.rect(x0, 1.75 - 0.065, 1.24, 0.13, fc=LOAD_FC, ec=LOAD_EC, lw=0.6)
    c.rect(x0, 1.53 - 0.065, 1.02, 0.13, fc=COMP_FC, ec=COMP_EC, lw=0.6, hatch="////")
    xc = x0 + 0.52
    c.rect(x0, 1.29 + 0.005, xc - x0, 0.075, fc=COMP_FC, ec=COMP_EC, lw=0.6, hatch="////")
    c.rect(x0, 1.29 - 0.08, xc - x0, 0.075, fc=LOAD_FC, ec=LOAD_EC, lw=0.6)
    c.line([xc, xc], [1.15, 1.86], color=INK, lw=0.7, ls=(0, (2, 2)), z=6)
    c.text(xc + 0.06, 1.29, L["tb_note"], size=FS_S, color=INK2)
    c.line([x0, x0], [1.15, 1.86], color=INK2, lw=0.6, z=6)
    # base rule
    c.box(XB0, 0.79, 2.12, 0.30, fc="white", ec=INK2, lw=0.7, r=0.03)
    c.text(XB0 + 0.08, 0.94, L["rule"], size=FS_S, ls=1.15)

    # legend (bottom of panel b)
    yl = 0.62
    dy = 0.128
    c.hook(XB0 + 0.08, yl, 1, r=0.055)
    c.text(XB0 + 0.22, yl, L["lg_hook"], size=FS_S)
    c.rect(XB0 + 0.03, yl - dy - 0.05, 0.10, 0.10, fc=LEV, ec=LEV, z=9)
    c.text(XB0 + 0.22, yl - dy, L["lg_lever"], size=FS_S)
    c.arrow((XB0 + 0.00, yl - 2 * dy), (XB0 + 0.17, yl - 2 * dy), color=ACC, lw=1.0, ls=(0, (2.5, 1.6)))
    c.text(XB0 + 0.22, yl - 2 * dy, L["lg_act"], size=FS_S)
    c.rect(XB0 + 0.02, yl - 3 * dy - 0.05, 0.13, 0.10, fc=COMP_FC, ec=COMP_EC, hatch="////")
    c.text(XB0 + 0.22, yl - 3 * dy, L["lg_comp"], size=FS_S)
    c.rect(XB0 + 0.02, yl - 4 * dy - 0.05, 0.13, 0.10, fc=LOAD_FC, ec=LOAD_EC)
    c.text(XB0 + 0.22, yl - 4 * dy, L["lg_load"], size=FS_S)
    return c


# ---------------------------------------------------------------- Fig 2
def fig2(L):
    W, H = 3.33, 4.62
    c = Canvas(W, H)
    X0, X1 = 0.05, W - 0.05
    # traces --------------------------------------------------------------
    c.text(X0, H - 0.05, L["tr_head"], size=FS_B, weight="bold", va="top")
    cw, gap = (X1 - X0 - 2 * 0.05) / 3, 0.05
    ych0, ych1 = 4.04, 4.34
    cx = []
    for i, k in enumerate(("tr1", "tr2", "tr3")):
        x = X0 + i * (cw + gap)
        c.box(x, ych0, cw, ych1 - ych0, fc=LANE_FC, ec=INK2, lw=0.7, r=0.03)
        c.text(x + cw / 2, (ych0 + ych1) / 2, L[k], size=FS_S, ha="center", ls=1.1)
        cx.append(x + cw / 2)
    # bus
    yb = 3.95
    for x in cx:
        c.line([x, x], [ych0, yb], color=INK2, lw=0.8)
    c.line([cx[0], cx[2]], [yb, yb], color=INK2, lw=0.8)

    # two boxes --------------------------------------------------------------
    SX0, SX1 = X0, 1.58
    GX0, GX1 = 1.76, X1
    BY0, BY1 = 2.46, 3.84
    c.arrow(((SX0 + SX1) / 2, yb), ((SX0 + SX1) / 2, BY1), color=INK2, lw=0.8)
    c.arrow(((GX0 + GX1) / 2, yb), ((GX0 + GX1) / 2, BY1), color=INK2, lw=0.8)
    c.box(SX0, BY0, SX1 - SX0, BY1 - BY0, fc="white", ec=INK, lw=0.9, r=0.04)
    c.box(GX0, BY0, GX1 - GX0, BY1 - BY0, fc="white", ec=INK, lw=0.9, r=0.04)
    # simulator contents
    c.text(SX0 + 0.08, 3.73, L["sim_t"], size=FS_B, weight="bold")
    c.text(SX0 + 0.08, 3.585, L["sim_s"], size=FS_S, color=INK2)
    yc1 = 3.31
    c.box(SX0 + 0.07, yc1 - 0.14, SX1 - SX0 - 0.14, 0.28, fc=LANE_FC, ec=INK2, lw=0.7, r=0.025)
    c.text(SX0 + 0.12, yc1, L["c_h200"], size=FS_S, ls=1.1)
    yc2 = 2.905
    c.box(SX0 + 0.07, yc2 - 0.20, SX1 - SX0 - 0.14, 0.40, fc="white", ec=INK2, lw=0.7, r=0.025,
          ls=(0, (3, 2)))
    c.text(SX0 + 0.12, yc2, L["c_slow"], size=FS_S, ls=1.1)
    yh = 2.585
    c.hook(SX0 + 0.15, yh, 2)
    c.text(SX0 + 0.27, yh, L["sim_hook"], size=FS_S, color=ACC, weight="bold")
    # GPU contents
    c.text(GX0 + 0.08, 3.73, L["gpu_t"], size=FS_B, weight="bold")
    c.text(GX0 + 0.08, 3.585, L["gpu_s"], size=FS_S, color=INK2)
    c.arrow((GX0, yc1), (SX1 - 0.07, yc1), color=INK, lw=0.9)
    c.text(GX0 + 0.08, yc1, L["gpu_const"], size=FS_S)
    yg = 3.10
    c.hook(GX0 + 0.15, yg, 1)
    c.hook(GX0 + 0.31, yg, 3)
    c.text(GX0 + 0.43, yg, L["gpu_hooks"], size=FS_S, color=ACC, weight="bold")
    yl = 2.91
    xe = c.lever(GX0 + 0.08, yl, "INT4")
    xe = c.lever(xe + 0.05, yl, "MPS")
    c.text(xe + 0.06, yl, L["gpu_lev"], size=FS_S)
    yv = 2.64
    c.text(GX0 + 0.08, yv, L["gpu_val"], size=FS_S, ls=1.1)
    c.arrow((GX0, yh), (SX1, yh), color=INK, lw=0.9, both=True)

    # ladder -----------------------------------------------------------------
    c.arrow(((SX0 + SX1) / 2, BY0), ((SX0 + SX1) / 2, 2.27), color=INK2, lw=0.8)
    c.arrow(((GX0 + GX1) / 2, BY0), ((GX0 + GX1) / 2, 2.27), color=INK2, lw=0.8)
    c.text(X0, 2.22, L["lad_head"], size=FS_B, weight="bold", va="top")
    c.text(X0, 2.04, L["lad_sub"], size=FS_S, color=INK2, va="center")
    xlab, xb0 = 1.42, 1.48
    arms = [("arm_base", 3.20, "#c9c9c9", INK2, None),
            ("arm_rival", 2.86, "#8a8a8a", INK, None),
            ("arm_free", 2.56, ACC_LT, ACC, "////"),
            ("arm_oracle", 2.22, ACC, ACC, None)]
    ys = [1.86, 1.68, 1.50, 1.32]
    for (k, xend, fc, ec, hatch), y in zip(arms, ys):
        c.text(xlab, y, L[k], size=FS_S, ha="right")
        c.rect(xb0, y - 0.06, xend - xb0, 0.12, fc=fc, ec=ec, lw=0.6, hatch=hatch)
    xr, xo = arms[1][1], arms[3][1]
    yd = 1.155
    c.line([xr, xr], [ys[1] - 0.06, yd - 0.04], color=INK, lw=0.6, ls=(0, (2, 1.5)), z=6)
    c.line([xo, xo], [ys[3] - 0.06, yd - 0.04], color=INK, lw=0.6, ls=(0, (2, 1.5)), z=6)
    c.arrow((xo, yd), (xr, yd), color=INK, lw=0.8, both=True, ms=6)
    c.text(X1, yd - 0.14, L["headroom"], size=FS_S, ha="right", weight="bold")

    # ruler --------------------------------------------------------------------
    c.arrow(((SX0 + SX1) / 2, 1.09), ((SX0 + SX1) / 2, 0.93), color=INK2, lw=0.8)
    c.text(X0, 0.84, L["rul_head"], size=FS_B, weight="bold", va="center")
    rx0, rx1 = 0.22, 3.18
    ry0, ry1 = 0.43, 0.67
    hmax = 25.0

    def X(p):
        return rx0 + (rx1 - rx0) * p / hmax
    zones = [(0, 5, "white", None, "z1"), (5, 15, "#e3e3e3", "....", "z2"), (15, hmax, "#bdbdbd", "////", "z3")]
    for a, b, fc, hatch, k in zones:
        c.rect(X(a), ry0, X(b) - X(a), ry1 - ry0, fc=fc, ec=INK, lw=0.7, hatch=hatch, z=3)
        c.text((X(a) + X(b)) / 2, (ry0 + ry1) / 2, L[k], size=FS, weight="bold", ha="center",
               bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.9))
    c.arrow((rx1, ry0), (rx1 + 0.09, ry0), color=INK, lw=0.7, ms=5)
    for p in (5, 15):
        c.line([X(p), X(p)], [ry0 - 0.05, ry1], color=INK, lw=1.0, z=6)
        c.text(X(p), ry0 - 0.12, f"{p}%", size=FS, weight="bold", ha="center")
    c.text(rx1, ry0 - 0.12, L["rul_ax"], size=FS_S, color=INK2, ha="right")
    c.text(W / 2, 0.11, L["rul_note"], size=FS_S, color=INK2, ha="center", style="italic")
    return c


# ---------------------------------------------------------------- main
def build(lang, outdir):
    setup_fonts(lang)
    L = LABELS[lang]
    out = []
    for name, fn in (("fig1_system", fig1), ("fig2_method", fig2)):
        c = fn(L)
        c.check(f"{name}_{lang}")
        png, pdf = c.save(os.path.join(outdir, f"{name}_{lang}"))
        out += [png, pdf]
    if lang == "zh":
        for name in ("fig1_system", "fig2_method"):
            shutil.copyfile(os.path.join(outdir, f"{name}_zh.png"), os.path.join(outdir, f"{name}.png"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=("zh", "en", "both"), default="both")
    ap.add_argument("--outdir", default=HERE)
    a = ap.parse_args()
    for lang in (("zh", "en") if a.lang == "both" else (a.lang,)):
        for p in build(lang, a.outdir):
            print(f"{p}  {os.path.getsize(p) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
