#!/usr/bin/env python3
"""Fig 1 (system model + option space), Fig 2 (method) and Fig 11 (decision flow) for the ICPE paper,
drawn with matplotlib patches.

    python diagrams.py --lang zh|en|both      (default: both)

Outputs, next to this file:
    fig1_system_{zh,en}.{png,pdf}     full-width figure* : 7.0 in wide
    fig2_method_{zh,en}.{png,pdf}     single column      : 3.33 in wide
    fig11_decision_{zh,en}.{png,pdf}  full-width figure* : 7.0 in wide
    fig1_system.png, fig2_method.png, fig11_decision.png  (copies of the zh PNGs, embedded by PAPER.md)

Every coordinate is in inches at the final printed size, and every font size is in points,
so what you see in the PNG is what lands in the PDF column.  No text is smaller than 7 pt.
Fig 1 and Fig 2 carry no measured numbers except the preregistered thresholds (5 %, 15 %) and the
GPU-replay validation range (0.911-0.987, EVIDENCE.md row B6); their bar lengths are schematic.
Fig 11 uses only EVIDENCE.md section M ("組合地圖") plus rows B3, C2-GPU and G-M06, which M5 points to.
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
        # fig 1 (system model + option space)
        "a_title": "(a) One session's KV between two requests",
        "b_title": "(b) Options at the restore step",
        "lane_gpu": "GPU HBM", "lane_cpu": "CPU tier", "lane_cpu_s": "DRAM",
        "lane_slow": "Slow tier", "lane_slow_s": "SSD class (throttled)\nNFS class (throttled)",
        "lane_drop": "Dropped",
        "kv": "session KV", "gpu_other": "GPU serves\nother requests",
        "write": "write all\nblocks", "demote": "CPU tier full:\ndemote or drop",
        "drop_t": "recompute all on return", "prefetch": "prefetch",
        "restore_arrow": "restore", "restore_bar": "restore", "prefill_bar": "prefill", "ttft": "TTFT",
        "ev_end": "turn k ends", "ev_ret": "request returns",
        "pred_band": "prediction (add-on) acts here:\nprefetch, eviction order, drop",
        "ax_k": "turn k", "ax_gap": "between requests", "ax_k1": "turn k+1",
        "kn_dir": "direction", "kn_base": "in the base map", "kn_add": "add-ons",
        "d_reuse": "Reuse", "d_comp": "Compress", "d_focus": "Focus", "d_expand": "Expand",
        "d_gpu": "GPU share", "d_gpu_s": "(cross-\ncutting)",
        "k_load": "load all", "k_cake": "Cake", "k_rec": "recompute all", "k_approx": "approximate\nrecompute",
        "k_bf16": "BF16", "k_int4": "INT4", "k_lr": "low-rank+INT4",
        "k_prog": "progressive /\nselective restore",
        "k_cpu": "CPU tier", "k_slow": "slow tier",
        "k_pred": "prediction:\nprefetch, eviction\norder, drop", "k_path": "extra path via\nneighbour GPU",
        "k_stream": "stream", "k_mps": "MPS (reserve SMs)", "k_borrow": "borrowed\ncompute",
        "none": "\u2014",
        "lg_base": "option in the base map", "lg_add": "add-on outside the base",
        "lg_pred": "add-on that needs prediction (oracle study)",
        "lg_comp": "GPU recompute", "lg_load": "load (I/O)",
        # fig 11 (decision flow; numbers from EVIDENCE.md section M and B3/C2-GPU/G-M06)
        "q1": "Where is the KV?", "q1_s": "reuse + compress",
        "rq1": "Step 1: pick restore method and format", "rq2": "Step 2: share the GPU",
        "rq3": "Step 3: add prediction or a mechanism?",
        "q2": "Background work?", "q2_s": "GPU resources",
        "q3": "Add prediction?", "q3_s": "expand: simulate the oracle bound",
        "q4": "Neighbour idle?", "q4_s": "expand: extra path",
        "c1r1_c": "CPU tier", "c1r1_h": "INT4, by dequant code",
        "c1r1_d": "unpack once \u2192 load all\nper-block, unknown \u2192 Cake\n(always Cake: +6.7% avg,\n60.5% worst vs best)\nBF16 only: Cake when busy",
        "c1r2_c": "b32, b48,\nNFS class", "c1r2_h": "INT4 + Cake",
        "c1r2_d": "11\u201341% less time than\nthe next best; Qwen +\nbusy GPU: quality misses \u03b5",
        "c1r3_c": "b16", "c1r3_h": "INT4", "c1r3_d": "Cake and load all are close;\nload all may win",
        "c1r5_c": "low-rank\n+ INT4", "c1r5_h": "only if quality allows",
        "c1r5_d": "Llama, per-question SVD\nonly; idle GPU only",
        "c2r1_c": "no", "c2r1_h": "stream", "c2r1_d": "no reservation",
        "c2r2_c": "yes", "c2r2_h": "pick an MPS level",
        "c2r2_d": "vs stream, uncapped\nMPS: TTFT 31\u201363%\nlower, background\n0\u20136.5 pp lower\n(sim.; GPU check\nagrees in sign)",
        "c3r1_c": "bound high,\ne.g. Mooncake", "c3r1_h": "yes: prefetch",
        "c3r1_d": "with a free signal: the\ntyping hint \u22480.8 of the\noracle (GPU replay);\neviction order: no free\nsignal tested",
        "c3r2_c": "bound low, e.g.\nFreeInference", "c3r2_h": "don't add", "c3r2_d": "< 15% at every tested\nCPU size (sim.)",
        "c3r3_c": "write time", "c3r3_h": "no", "c3r3_d": "placement by position:\nno TTFT gain on GPU",
        "c4r1_c": "yes", "c4r1_h": "add a 2nd path",
        "c4r1_d": "via neighbour:\nH2D + NVLink,\n1.99\u00d7 faster\n(2 GPUs; its\nslowdown not\nmeasured)",
        "c4r2_c": "no", "c4r2_h": "one path",
        "stop_h": "Do not add (gate failed on H200):",
        "stop_d": "progressive restore \u00b7 approximate recompute \u00b7 borrowed compute \u00b7\nCPU-side compression at demotion \u00b7 online cost model or selector\n(use this static map) \u00b7 recompute all is never best at 32K",
        "nc_h": "Scope: one H200, 32K tokens. Not covered:",
        "nc_d": "SSD class and > 32K (not measured);\nselective loading, lookup decoding (no verdict);\ntyping-hint timing is synthetic",
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
        "gpu_lev": "on vs. off", "gpu_mps": "microbench + check",
        "sim_mps": "trade-off curve",
        "gpu_val": "scaled-down replay:\nmeasured / simulated TTFT\n0.911–0.987, same ranking",
        "lad_head": "Per decision: four arms on the same trace",
        "lad_sub": "bar = mean TTFT of returning requests (schematic)",
        "arm_base": "base: restore with Cake", "arm_rival": "best rival, no foresight",
        "arm_free": "free signal (typing hint)", "arm_oracle": "oracle (perfect foresight)",
        "headroom": r"headroom $= 1 - T_{\mathrm{oracle}}\,/\,T_{\mathrm{rival}}$",
        "thr_note": "Preregistered threshold: worth it if the bound is \u226515%",
    },
    "zh": {
        "a_title": "(a) 一個 session 的 KV 在兩次請求之間",
        "b_title": "(b) 還原這一步可以調的選項",
        "lane_gpu": "GPU HBM", "lane_cpu": "CPU 層", "lane_cpu_s": "DRAM",
        "lane_slow": "慢層", "lane_slow_s": "SSD 級（限速模擬）\nNFS 級（限速模擬）",
        "lane_drop": "丟掉",
        "kv": "session 的 KV", "gpu_other": "GPU 服務\n其他請求",
        "write": "整段\n寫入", "demote": "CPU 層滿了：\n降層或丟掉",
        "drop_t": "回來時全部重算", "prefetch": "預取",
        "restore_arrow": "還原", "restore_bar": "還原", "prefill_bar": "prefill", "ttft": "TTFT",
        "ev_end": "第 k 回合結束", "ev_ret": "請求回來",
        "pred_band": "預測（附加）在這裡起作用：\n預取、淘汰順序、丟掉",
        "ax_k": "第 k 回合", "ax_gap": "兩次請求之間", "ax_k1": "第 k+1 回合",
        "kn_dir": "方向", "kn_base": "基本組合（地圖內）", "kn_add": "附加",
        "d_reuse": "重用", "d_comp": "壓縮", "d_focus": "聚焦", "d_expand": "擴展",
        "d_gpu": "GPU 資源", "d_gpu_s": "（跨方向）",
        "k_load": "全部載入", "k_cake": "Cake", "k_rec": "全部重算", "k_approx": "近似重算",
        "k_bf16": "BF16", "k_int4": "INT4", "k_lr": "低秩＋INT4",
        "k_prog": "漸進／\n選擇性還原",
        "k_cpu": "CPU 層", "k_slow": "慢層",
        "k_pred": "預測：預取、\n淘汰順序、丟掉", "k_path": "借鄰卡多一條路徑",
        "k_stream": "stream", "k_mps": "MPS（保留 SM）", "k_borrow": "借鄰居的算力",
        "none": "\u2014",
        "lg_base": "基本組合裡的選項", "lg_add": "基本組合之外的附加",
        "lg_pred": "需要預測的附加（oracle 研究）",
        "lg_comp": "GPU 重算", "lg_load": "載入（I/O）",
        "q1": "KV 在哪？", "q1_s": "重用＋壓縮",
        "rq1": "第一步：選還原方式與格式", "rq2": "第二步：GPU 共用時分 SM",
        "rq3": "第三步：要不要再加預測或機制",
        "q2": "GPU 有背景工作？", "q2_s": "GPU 資源",
        "q3": "要不要加預測？", "q3_s": "擴展：先模擬自己 trace 的上界",
        "q4": "鄰卡閒著嗎？", "q4_s": "擴展：多一條路徑",
        "c1r1_c": "CPU 層", "c1r1_h": "INT4；方式看反量化實作",
        "c1r1_d": "一次拆完 \u2192 全部載入\n逐塊或不知道 \u2192 Cake\n（永遠 Cake：平均 +6.7%，\n最差 60.5%，對最好的）\n只准 BF16：GPU 忙時 Cake",
        "c1r2_c": "b32、\nb48、\nNFS 級", "c1r2_h": "INT4＋Cake",
        "c1r2_d": "比第二名快 11\u201341%\nQwen＋GPU 忙：品質沒過 \u03b5",
        "c1r3_c": "b16", "c1r3_h": "INT4", "c1r3_d": "Cake 與全部載入接近；\n全部載入可能贏",
        "c1r5_c": "低秩\n＋INT4", "c1r5_h": "品質允許才用",
        "c1r5_d": "只有 Llama 每題 SVD 過 \u03b5；\n只在 GPU 閒時",
        "c2r1_c": "否", "c2r1_h": "stream", "c2r1_d": "不保留 SM",
        "c2r2_c": "是", "c2r2_h": "挑 MPS 等級",
        "c2r2_d": "對 stream：不設上限\n的 MPS，TTFT 低\n31\u201363%、背景吞吐\n低 0\u20136.5 pp（模擬；\nGPU 檢查正負號同）",
        "c3r1_c": "上界高，如\nMooncake", "c3r1_h": "要：預取",
        "c3r1_d": "用免費訊號：打字\n提示拿到 oracle 的\n約 0.8（GPU 重播）；\n淘汰順序：沒有免費\n訊號比過",
        "c3r2_c": "上界低，如\nFreeInference", "c3r2_h": "不要加", "c3r2_d": "量過的每個 CPU 容量\n都 <15%（模擬）",
        "c3r3_c": "寫入時", "c3r3_h": "不要", "c3r3_d": "依位置放層級：\nGPU 上量不到 TTFT 收益",
        "c4r1_c": "是", "c4r1_h": "多一條還原路徑",
        "c4r1_d": "借鄰卡的 H2D\n＋NVLink：\n快 1.99 倍\n（2 卡；鄰卡\n變慢多少沒量）",
        "c4r2_c": "否", "c4r2_h": "一條路徑",
        "stop_h": "不要加（H200 上的閘沒過）：",
        "stop_d": "漸進還原 \u00b7 近似重算 \u00b7 借鄰居的算力 \u00b7 降層時在 CPU 壓縮 \u00b7\n線上成本模型或選擇器（用這張靜態的表）\u00b7\n32K 時全部重算從來不是最好",
        "nc_h": "範圍：一張 H200、32K token；沒涵蓋：",
        "nc_d": "SSD 級與 >32K 沒量；\n選擇性載入、查表解碼沒有判定；\n打字提示的時間是合成的",
        "tr_head": "真實 trace",
        "tr1": "Mooncake\nconversation（chat）", "tr2": "Mooncake\ntoolagent（工具）", "tr3": "FreeInference\n（agent）",
        "sim_t": "trace 驅動的模擬器", "sim_s": "一次一個請求",
        "c_h200": "CPU 層＋GPU 重算：\nH200 實測",
        "c_slow": "SSD／NFS 級：限速模擬\n（常數沿用前一個平台）",
        "sim_hook": "回來時的預知",
        "gpu_t": "在 GPU 上實測", "gpu_s": "一張 H200，實際執行",
        "gpu_const": "微基準量出\n成本常數",
        "gpu_hooks": "寫入時、還原時",
        "gpu_lev": "開與關", "gpu_mps": "微基準＋GPU 對照",
        "sim_mps": "取捨曲線",
        "gpu_val": "縮小重播：\nTTFT 實測÷模擬\n0.911–0.987，排序相同",
        "lad_head": "每個決策：同一條 trace 跑四種臂",
        "lad_sub": "長條＝回來請求的平均 TTFT（示意）",
        "arm_base": "底：用 Cake 還原", "arm_rival": "最好的對手（不預知）",
        "arm_free": "免費訊號（打字提示）", "arm_oracle": "oracle（完美預知）",
        "headroom": r"headroom $= 1 - T_{\mathrm{oracle}}\,/\,T_{\mathrm{對手}}$",
        "thr_note": "門檻（事先登記）：上界 \u226515% 才算值得",
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
    def __init__(self, w, h, y0=0.0):
        self.W, self.H = w, h
        self.fig = plt.figure(figsize=(w, h))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(y0, y0 + h)
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

    # ---- option chips (Fig 1b) and decision rows (Fig 11)
    def measure(self, s, size=FS_S, weight="normal", ls=1.12):
        t = self.text(0, 0, s, size=size, weight=weight, ls=ls)
        w, h = self.text_w(t)
        return t, w, h

    def _warn(self, what, need, have):
        if need > have + 1e-6:
            print(f"  WARN text wider than its box by {need - have:.3f} in: {what!r}")

    def chip(self, x, ytop, s, kind="base", glyph=None, maxw=None):
        """option chip with its top-left corner at (x, ytop). kind: base | add | pred. returns (w, h)"""
        t, w, h = self.measure(s)
        t.set_color(ACC if kind == "pred" else INK)
        gw = 0.105 if glyph else 0.0
        cw, ch = w + 0.09 + gw, h + 0.055
        t.set_position((x + 0.045 + gw, ytop - ch / 2))
        t.set_zorder(7)
        ec = {"base": INK2, "add": INK3, "pred": ACC}[kind]
        self.box(x, ytop - ch, cw, ch, fc="white", ec=ec, lw=0.8, ls="-" if kind == "base" else (0, (2.5, 1.5)),
                 r=0.025, z=4)
        if glyph:
            gx, gy, gh = x + 0.045, ytop - ch / 2 - 0.045, 0.09
            if glyph == "load":
                self.rect(gx, gy, 0.08, gh, fc=LOAD_FC, ec=LOAD_EC, lw=0.5, z=5)
            elif glyph == "comp":
                self.rect(gx, gy, 0.08, gh, fc=COMP_FC, ec=COMP_EC, lw=0.5, z=5, hatch="////")
            else:  # cake: recompute from the front, load from the back
                self.rect(gx, gy, 0.035, gh, fc=COMP_FC, ec=COMP_EC, lw=0.5, z=5, hatch="////")
                self.rect(gx + 0.035, gy, 0.045, gh, fc=LOAD_FC, ec=LOAD_EC, lw=0.5, z=5)
        if maxw is not None:
            self._warn(s, cw, maxw)
        return cw, ch

    def flow(self, items, x0, x1, ytop, gap=0.045, vgap=0.035):
        """lay chips left to right from (x0, ytop), wrapping at x1. returns the bottom y"""
        x, y, lh = x0, ytop, 0.0
        for s, kind, glyph in items:
            t, w, h = self.measure(s)
            t.remove()
            cw, ch = w + 0.09 + (0.105 if glyph else 0.0), h + 0.055
            if x > x0 and x + cw > x1 + 1e-6:
                x, y, lh = x0, y - lh - vgap, 0.0
            self.chip(x, y, s, kind, glyph, maxw=x1 - x0)
            x += cw + gap
            lh = max(lh, ch)
        return y - lh

    def drow(self, x, ytop, wc, wa, cond, head, detail=None, head_color=INK):
        """one decision row: grey condition cell | action cell (bold head + detail). returns bottom y"""
        pad = 0.045
        tc, wcw, hc = self.measure(cond, ls=1.08)
        th, wh, hh = self.measure(head, size=FS, weight="bold")
        th.set_color(head_color)
        th.set_va("top")
        hd, wd = 0.0, 0.0
        if detail:
            td, wd, hd = self.measure(detail, ls=1.1)
            td.set_va("top")
            td.set_color(INK2)
        h = max(hc, hh + (hd + 0.025 if detail else 0.0)) + 2 * pad
        tc.set_position((x + 0.05, ytop - h / 2))
        th.set_position((x + wc + 0.05, ytop - pad))
        if detail:
            td.set_position((x + wc + 0.05, ytop - pad - hh - 0.025))
        self.rect(x, ytop - h, wc, h, fc=LANE_FC, ec=INK3, lw=0.6, z=3)
        self.rect(x + wc, ytop - h, wa, h, fc="white", ec=INK3, lw=0.6, z=3)
        self._warn(cond, wcw + 0.08, wc)
        self._warn(head, wh + 0.08, wa)
        if detail:
            self._warn(detail, wd + 0.08, wa)
        return ytop - h

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
    """(a) a session's KV over time across tiers; (b) the option space at the restore step"""
    W, H = 7.0, 3.45
    c = Canvas(W, H)
    XL0, XL1 = 0.05, 4.08            # lanes, panel (a)
    XE, XA = 1.62, 2.98              # turn k ends / request k+1 returns
    LANES = {"gpu": (1.74, 2.40), "cpu": (1.23, 1.66), "slow": (0.72, 1.15), "drop": (0.36, 0.64)}
    XB0 = 4.30                       # panel (b) left edge
    c.text(0.06, H - 0.07, L["a_title"], size=FS_T, weight="bold", va="top")
    c.text(XB0, H - 0.07, L["b_title"], size=FS_T, weight="bold", va="top")
    c.line([4.19, 4.19], [0.06, H - 0.06], color="#c8c8c8", lw=0.8, ls=(0, (3, 3)), z=1)

    # (a) lanes ------------------------------------------------------------
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

    # event lines, prediction band, time axis
    for x in (XE, XA):
        c.line([x, x], [0.30, 2.56], color="#9a9a9a", lw=0.7, ls=(0, (2, 2)), z=2)
    c.text(XE, 2.62, L["ev_end"], size=FS_S, color=INK2, ha="center")
    c.text(XA, 2.62, L["ev_ret"], size=FS_S, color=INK2, ha="center")
    yp = 2.77
    c.line([XE + 0.03, XE + 0.03, XA - 0.03, XA - 0.03], [yp - 0.045, yp, yp, yp - 0.045], color=ACC, lw=0.9,
           ls=(0, (2.5, 1.5)), z=6)
    c.text((XE + XA) / 2, yp + 0.035, L["pred_band"], size=FS_S, color=ACC, ha="center", va="bottom", ls=1.1)
    c.arrow((0.95, 0.24), (4.02, 0.24), color=INK2, lw=0.7, ms=6, z=3)
    for x in (XE, XA):
        c.line([x, x], [0.20, 0.28], color=INK2, lw=0.7, z=3)
    c.text((0.95 + XE) / 2, 0.11, L["ax_k"], size=FS_S, color=INK2, ha="center")
    c.text((XE + XA) / 2, 0.11, L["ax_gap"], size=FS_S, color=INK2, ha="center")
    c.text((XA + 4.02) / 2, 0.11, L["ax_k1"], size=FS_S, color=INK2, ha="center")

    # GPU lane
    BW, BH = 0.068, 0.14
    yb = 2.07
    xk = 1.0
    c.blocks(xk, yb, 8, BW, BH)
    c.text(xk + 4 * BW, 1.93, L["kv"], size=FS_S, color=INK2, ha="center")
    c.rect(XE + 0.04, 1.80, XA - XE - 0.08, 0.54, fc="white", ec="#c9c9c9", lw=0.5, z=2, hatch="....")
    c.text((XE + XA) / 2, 2.07, L["gpu_other"], size=FS_S, color=INK2, ha="center", ls=1.1,
           bbox=dict(fc="white", ec="none", pad=0.6))
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
    c.text(xr + 2 * BW, 1.93, L["restore_bar"], size=FS_S, color=INK2, ha="center")
    c.text(xp + 0.13, 1.93, L["prefill_bar"], size=FS_S, color=INK2, ha="center")
    c.brace_h(XA, xt + 0.02, 2.29, h=0.05, color=INK)
    c.text((XA + xt) / 2, 2.30, L["ttft"], size=FS, weight="bold", ha="center", va="bottom")

    # CPU lane
    ycb = ym["cpu"] - BH / 2 + 0.02
    xc0 = XE + 0.08
    c.blocks(xc0, ycb, 8, BW, BH)
    c.arrow((xk + 8 * BW + 0.01, yb + BH / 2), (xc0 + 0.04, ycb + BH + 0.005), color=INK, lw=0.9, rad=-0.35)
    c.text(XE - 0.07, ym["cpu"] - 0.02, L["write"], size=FS_S, ha="right", ls=1.05)

    # slow lane (format is a knob in (b), so the bar keeps the same height)
    xs0 = xc0 + 8 * BW + 0.10
    ysb = ym["slow"] - BH / 2 + 0.03
    c.blocks(xs0, ysb, 8, BW, BH)
    c.arrow((xc0 + 8 * BW - 0.06, ycb), (xs0 + 0.03, ysb + BH + 0.005), color=INK, lw=0.9, rad=0.2)
    c.text(xc0 + 0.30, ym["slow"] + 0.0, L["demote"], size=FS_S, color=INK2, ha="right", ls=1.05)
    xd = xc0 + 0.40
    c.arrow((xd, ycb), (xd, ym["drop"] + 0.07), color=INK2, lw=0.8, ls=(0, (3, 2)))
    c.text(xd, ym["drop"], "×", size=10, color=INK, ha="center", va="center", weight="bold")
    c.text(xd + 0.09, ym["drop"], L["drop_t"], size=FS_S, color=INK2)
    # prediction add-on: prefetch (dashed accent), slow -> CPU before the return
    xpf = xs0 + 8 * BW - 0.14
    c.arrow((xpf, ysb + BH + 0.01), (xpf + 0.04, ycb + 0.005), color=ACC, lw=1.0, ls=(0, (2.5, 1.6)), rad=-0.15)
    c.rect(xpf + 0.005, ycb, 0.16, BH, fc="white", ec=ACC, lw=0.8, ls=(0, (2, 1.5)), z=4)
    c.text(xpf - 0.04, ym["cpu"] - 0.165, L["prefetch"], size=FS_S, color=ACC, ha="right", style="italic")
    # restore at the return
    c.arrow((xs0 + 8 * BW + 0.01, ysb + 0.05), (xr + 0.42, yb - 0.005), color=LOAD_EC, lw=1.1, rad=0.22)
    c.text(XA + 0.55, ym["cpu"] - 0.06, L["restore_arrow"], size=FS_S, color=LOAD_EC, style="italic")

    # (b) option space ---------------------------------------------------------
    XD0, XD1 = XB0, XB0 + 0.60
    XS0, XS1 = XD1 + 0.04, XD1 + 0.04 + 1.04
    XA0, XA1 = XS1 + 0.04, 6.96
    yh = 3.10
    c.text(XD0 + 0.02, yh, L["kn_dir"], size=FS_S, color=INK2, weight="bold")
    c.text(XS0, yh, L["kn_base"], size=FS_S, color=INK2, weight="bold")
    c.text(XA0, yh, L["kn_add"], size=FS_S, color=INK2, weight="bold")
    c.line([XB0, XA1], [yh - 0.075, yh - 0.075], color=INK2, lw=0.7)
    rows = [
        ("d_reuse", None, [("k_load", "base", "load"), ("k_cake", "base", "cake"), ("k_rec", "base", "comp")],
         [("k_approx", "add", None)]),
        ("d_comp", None, [("k_bf16", "base", None), ("k_int4", "base", None), ("k_lr", "base", None)], []),
        ("d_focus", None, [], [("k_prog", "add", None)]),
        ("d_expand", None, [("k_cpu", "base", None), ("k_slow", "base", None)],
         [("k_pred", "pred", None), ("k_path", "add", None)]),
        ("d_gpu", "d_gpu_s", [("k_stream", "base", None), ("k_mps", "base", None)], [("k_borrow", "add", None)]),
    ]
    y = yh - 0.075 - 0.05
    for i, (dk, dsub, base, add) in enumerate(rows):
        def col(items, x0, x1):
            if not items:
                c.text(x0 + 0.03, y - 0.08, L["none"], size=FS_S, color=INK3)
                return y - 0.16
            return c.flow([(L[k], kind, g) for k, kind, g in items], x0, x1, y)
        b1 = col(base, XS0, XS1)
        b2 = col(add, XA0, XA1)
        lab = L[dk] + ("\n" + L[dsub] if dsub else "")
        td, wd, hd = c.measure(lab, size=FS_S, weight="bold", ls=1.08)
        ybot = min(b1, b2, y - hd - 0.06)
        c.box(XD0, ybot, XD1 - XD0, y - ybot, fc=LANE_FC, ec=LANE_EC, lw=0.6, r=0.025, z=2)
        td.set_position((XD0 + 0.04, (y + ybot) / 2))
        c._warn(lab, wd + 0.06, XD1 - XD0)
        if i < len(rows) - 1:
            c.line([XS0, XA1], [ybot - 0.035, ybot - 0.035], color="#d0d0d0", lw=0.6, z=2)
        y = ybot - 0.06

    # legend (bottom of b)
    yl, dy = 0.47, 0.128
    x2 = XB0 + 1.50

    def mini(x, yc, kind):
        ec = {"base": INK2, "add": INK3, "pred": ACC}[kind]
        c.box(x, yc - 0.05, 0.22, 0.10, fc="white", ec=ec, lw=0.8, ls="-" if kind == "base" else (0, (2.5, 1.5)),
              r=0.02, z=4)
    mini(XB0, yl, "base")
    c.text(XB0 + 0.28, yl, L["lg_base"], size=FS_S)
    mini(XB0, yl - dy, "add")
    c.text(XB0 + 0.28, yl - dy, L["lg_add"], size=FS_S)
    mini(XB0, yl - 2 * dy, "pred")
    c.text(XB0 + 0.28, yl - 2 * dy, L["lg_pred"], size=FS_S, color=ACC)
    c.rect(x2, yl - 0.05, 0.14, 0.10, fc=COMP_FC, ec=COMP_EC, hatch="////")
    c.text(x2 + 0.20, yl, L["lg_comp"], size=FS_S)
    c.rect(x2, yl - dy - 0.05, 0.14, 0.10, fc=LOAD_FC, ec=LOAD_EC)
    c.text(x2 + 0.20, yl - dy, L["lg_load"], size=FS_S)
    if y < yl + 0.08:
        print(f"  WARN option rows run into the legend (bottom {y:.2f})")
    return c


# ---------------------------------------------------------------- Fig 2
def fig2(L):
    U = 0.22                        # extra height of the two boxes
    YC = 0.72                       # height cut from the bottom after the threshold ruler was dropped
    W, H = 3.33, 4.62 + U
    c = Canvas(W, H - YC, y0=YC)
    X0, X1 = 0.05, W - 0.05
    # traces --------------------------------------------------------------
    c.text(X0, H - 0.05, L["tr_head"], size=FS_B, weight="bold", va="top")
    cw, gap = (X1 - X0 - 2 * 0.05) / 3, 0.05
    ych0, ych1 = 4.04 + U, 4.34 + U
    cx = []
    for i, k in enumerate(("tr1", "tr2", "tr3")):
        x = X0 + i * (cw + gap)
        c.box(x, ych0, cw, ych1 - ych0, fc=LANE_FC, ec=INK2, lw=0.7, r=0.03)
        c.text(x + cw / 2, (ych0 + ych1) / 2, L[k], size=FS_S, ha="center", ls=1.1)
        cx.append(x + cw / 2)
    # bus
    yb = 3.95 + U
    for x in cx:
        c.line([x, x], [ych0, yb], color=INK2, lw=0.8)
    c.line([cx[0], cx[2]], [yb, yb], color=INK2, lw=0.8)

    # two boxes --------------------------------------------------------------
    SX0, SX1 = X0, 1.58
    GX0, GX1 = 1.76, X1
    BY0, BY1 = 2.44, 3.84 + U
    c.arrow(((SX0 + SX1) / 2, yb), ((SX0 + SX1) / 2, BY1), color=INK2, lw=0.8)
    c.arrow(((GX0 + GX1) / 2, yb), ((GX0 + GX1) / 2, BY1), color=INK2, lw=0.8)
    c.box(SX0, BY0, SX1 - SX0, BY1 - BY0, fc="white", ec=INK, lw=0.9, r=0.04)
    c.box(GX0, BY0, GX1 - GX0, BY1 - BY0, fc="white", ec=INK, lw=0.9, r=0.04)
    # simulator contents
    c.text(SX0 + 0.08, 3.73 + U, L["sim_t"], size=FS_B, weight="bold")
    c.text(SX0 + 0.08, 3.585 + U, L["sim_s"], size=FS_S, color=INK2)
    yc1 = 3.31 + U
    c.box(SX0 + 0.07, yc1 - 0.14, SX1 - SX0 - 0.14, 0.28, fc=LANE_FC, ec=INK2, lw=0.7, r=0.025)
    c.text(SX0 + 0.12, yc1, L["c_h200"], size=FS_S, ls=1.1)
    yc2 = 2.905 + U
    c.box(SX0 + 0.07, yc2 - 0.20, SX1 - SX0 - 0.14, 0.40, fc="white", ec=INK2, lw=0.7, r=0.025,
          ls=(0, (3, 2)))
    c.text(SX0 + 0.12, yc2, L["c_slow"], size=FS_S, ls=1.1)
    yh = 2.585 + U
    c.hook(SX0 + 0.15, yh, 2)
    c.text(SX0 + 0.27, yh, L["sim_hook"], size=FS_S, color=ACC, weight="bold")
    xe = c.lever(SX0 + 0.08, yh - 0.18, "MPS")
    c.text(xe + 0.06, yh - 0.18, L["sim_mps"], size=FS_S)
    # GPU contents
    c.text(GX0 + 0.08, 3.73 + U, L["gpu_t"], size=FS_B, weight="bold")
    c.text(GX0 + 0.08, 3.585 + U, L["gpu_s"], size=FS_S, color=INK2)
    c.arrow((GX0, yc1), (SX1 - 0.07, yc1), color=INK, lw=0.9)
    c.text(GX0 + 0.08, yc1, L["gpu_const"], size=FS_S)
    yg = 3.10 + U
    c.hook(GX0 + 0.15, yg, 1)
    c.hook(GX0 + 0.31, yg, 3)
    c.text(GX0 + 0.43, yg, L["gpu_hooks"], size=FS_S, color=ACC, weight="bold")
    yl = 2.91 + U
    xe = c.lever(GX0 + 0.08, yl, "INT4")
    c.text(xe + 0.06, yl, L["gpu_lev"], size=FS_S)
    xe = c.lever(GX0 + 0.08, yl - 0.175, "MPS")
    c.text(xe + 0.06, yl - 0.175, L["gpu_mps"], size=FS_S)
    yv = 2.68
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

    # the preregistered threshold is one line of text, not a figure element of its own
    c.text(X0, yd - 0.33, L["thr_note"], size=FS_S, color=INK2, style="italic")
    return c


# ---------------------------------------------------------------- Fig 11
def fig11(L):
    """decision flow for a returning request; every claim is from EVIDENCE.md section M (+ B3, C2-GPU, G-M06)"""
    W, H = 7.0, 3.75
    c = Canvas(W, H)
    # (x, width, width of the condition cell)
    cols = [(0.05, 1.98, 0.58), (2.17, 1.36, 0.30), (3.67, 1.99, 0.80), (5.80, 1.15, 0.30)]
    QY1 = H - 0.30
    QY0 = QY1 - 0.34
    qm = (QY0 + QY1) / 2
    # step names over the boxes: step 1 = box 1, step 2 = box 2, step 3 = boxes 3a and 3b
    yr = QY1 + 0.075
    for (i0, i1, key) in ((0, 0, "rq1"), (1, 1, "rq2"), (2, 3, "rq3")):
        x0, x1 = cols[i0][0], cols[i1][0] + cols[i1][1]
        c.line([x0 + 0.01, x0 + 0.01, x1 - 0.01, x1 - 0.01], [yr - 0.035, yr, yr, yr - 0.035], color=INK, lw=0.8, z=6)
        c.text(x0 + 0.02, yr + 0.035, L[key], size=FS, weight="bold", va="bottom")
    for i, (x, w, wc) in enumerate(cols):
        c.box(x, QY0, w, QY1 - QY0, fc=LANE_FC, ec=INK, lw=0.9, r=0.03, z=2)
        c.rect(x + 0.05, qm - 0.075, 0.17, 0.15, fc=LEV, ec=LEV, z=4)
        c.text(x + 0.135, qm - 0.003, ("1", "2", "3a", "3b")[i], size=FS_S, color="white", weight="bold",
               ha="center", z=7)
        c.text(x + 0.26, QY1 - 0.045, L[f"q{i + 1}"], size=FS, weight="bold", va="top")
        c.text(x + 0.26, QY0 + 0.04, L[f"q{i + 1}_s"], size=FS_S, color=INK2, va="bottom")
        if i < len(cols) - 1:
            c.arrow((x + w, qm), (cols[i + 1][0], qm), color=INK, lw=1.0, ms=6)
    rows = {
        0: [("c1r1", INK), ("c1r2", INK), ("c1r3", INK), ("c1r5", INK2)],
        1: [("c2r1", INK2), ("c2r2", INK)],
        2: [("c3r1", INK), ("c3r2", INK2), ("c3r3", INK2)],
        3: [("c4r1", INK), ("c4r2", INK2)],
    }
    ylow = H
    for i, (x, w, wc) in enumerate(cols):
        y = QY0 - 0.10
        c.arrow((x + wc / 2, QY0), (x + wc / 2, y), color=INK2, lw=0.8, ms=5)
        for key, hc in rows[i]:
            y = c.drow(x, y, wc, w - wc, L[key + "_c"], L[key + "_h"], L.get(key + "_d"), head_color=hc) - 0.04
        ylow = min(ylow, y)
    # bottom band
    yt = 0.64
    if ylow < yt + 0.03:
        print(f"  WARN decision rows run into the bottom band (lowest {ylow:.2f})")
    for (x0, x1, hk, dk) in ((0.05, 4.10, "stop_h", "stop_d"), (4.20, 6.95, "nc_h", "nc_d")):
        c.box(x0, 0.05, x1 - x0, yt - 0.05, fc="white", ec=INK3, lw=0.7, r=0.03, z=2)
        th = c.text(x0 + 0.07, yt - 0.05, L[hk], size=FS, weight="bold", va="top")
        _, hh = c.text_w(th)
        td = c.text(x0 + 0.07, yt - 0.05 - hh - 0.02, L[dk], size=FS_S, color=INK2, va="top", ls=1.12)
        wd, _ = c.text_w(td)
        c._warn(L[dk], wd + 0.10, x1 - x0)
    return c


# ---------------------------------------------------------------- main
def build(lang, outdir):
    setup_fonts(lang)
    L = LABELS[lang]
    out = []
    for name, fn in (("fig1_system", fig1), ("fig2_method", fig2), ("fig11_decision", fig11)):
        c = fn(L)
        c.check(f"{name}_{lang}")
        png, pdf = c.save(os.path.join(outdir, f"{name}_{lang}"))
        out += [png, pdf]
    if lang == "zh":
        for name in ("fig1_system", "fig2_method", "fig11_decision"):
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
