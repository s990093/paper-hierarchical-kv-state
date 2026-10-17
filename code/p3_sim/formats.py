"""formats.py — 格式表（大小、解壓時間、品質）與平台成本常數的載入。

格式 id（品質由好到壞，id 越大越省）：1=bf16、2=8bit（Llama＝FP8、Qwen＝INT8）、3=int4（KIVI）、4=int2（KIVI）。
0＝不在這一層。「只能往下壓」＝新格式的 id ≥ 舊的。

格式表的 JSON（placeholder 與正式表共用的結構；正式表若結構不同，改 load_format_table 的轉換，不改引擎）：
  {PLACEHOLDER: bool, len_bins_tokens: [邊界...], never_read_only: ["int2"],
   quality: {model: {fmt: ["pass"|"fail"|"unverified", ... 每個長度分箱一個]}},
   models: {model: {kv_bytes_per_token_bf16, block_tokens, formats: {fmt: {bytes_per_block, dequant_s_per_block: {platform: {idle, busy}}}}}}}
NOT_MEASURED 的值一律保留成字串；用到它的地方丟 NotMeasured（CLAUDE.md 規則 1、7）。
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(HERE, "config")
REPO = os.path.dirname(os.path.dirname(HERE))
OFFICIAL_FORMAT_TABLE = os.path.join(REPO, "results", "p3_formats", "format_table.json")
PLACEHOLDER_FORMAT_TABLE = os.path.join(CFG, "format_table.PLACEHOLDER.json")
FMT_NAMES = ["none", "bf16", "8bit", "int4", "int2"]
FID = {n: i for i, n in enumerate(FMT_NAMES)}
NOT = "NOT_MEASURED"


class NotMeasured(RuntimeError):
    pass


def _num(x, what):
    if isinstance(x, str) or x is None:
        raise NotMeasured(f"{what} = {x!r}")
    return x


class ModelFormats:
    """一個模型在一個平台上的格式常數。bytes[fid]（int）、dq[gpu][fid]（秒／block）、quality[fid][bin]。"""

    def __init__(self, model, platform, bytes_by_fid, dq_by_gpu, len_bins, quality, never_read_only, placeholder, src):
        self.model, self.platform = model, platform
        self.bytes = list(bytes_by_fid)          # index＝fid；NOT_MEASURED 的格式為 None
        self.dq = dq_by_gpu                      # {"idle": [..fid..], "busy": [...]}；None＝NOT_MEASURED
        self.len_bins = list(len_bins)
        self.quality = quality                   # {fid: [status per bin]}
        self.never_read_only = set(never_read_only)
        self.placeholder = bool(placeholder)
        self.src = src
        self.available = [f for f in range(1, len(FMT_NAMES)) if self.bytes[f] is not None]

    def bin_of(self, tokens):
        b = self.len_bins
        for i in range(len(b) - 1):
            if b[i] <= tokens < b[i + 1]:
                return i
        return len(b) - 2

    def status(self, fid, tokens):
        if fid == FID["bf16"]:
            return "pass"
        q = self.quality.get(fid)
        if q is None:
            return "unverified"
        return q[self.bin_of(tokens)]

    def nbytes(self, fid):
        b = self.bytes[fid]
        if b is None:
            raise NotMeasured(f"{self.model} {FMT_NAMES[fid]} bytes_per_block")
        return b

    def dq_list(self, gpu):
        return self.dq.get(gpu)

    def cheapest(self, fids):
        fids = [f for f in fids if self.bytes[f] is not None]
        return min(fids, key=lambda f: (self.bytes[f], f)) if fids else None


def sha256_file(path):
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_format_table(model, platform, path=None):
    """path=None → placeholder（明確，不會因為正式表出現就悄悄換掉；結果要標 placeholder）。
    正式表（results/p3_formats/format_table.json，另一個 agent）的結構是「每列＝模型×格式」，和這裡的不同：
    出來之後寫一個轉換函式接到這裡（不改引擎）；在那之前給正式表的路徑會直接丟錯。"""
    if path is None:
        path = PLACEHOLDER_FORMAT_TABLE
    d = json.load(open(path, encoding="utf-8"))
    if "models" not in d or "len_bins_tokens" not in d:
        raise NotImplementedError(f"{path}: 不是 p3_sim 的格式表結構（正式表要先寫轉換，見 06_simulator.md §3）")
    m = d["models"][model]
    by, dqi, dqb = [None] * len(FMT_NAMES), [None] * len(FMT_NAMES), [None] * len(FMT_NAMES)
    dq_ok = True
    for name, fe in m["formats"].items():
        f = FID[name]
        b = fe.get("bytes_per_block")
        by[f] = None if (b is None or isinstance(b, str)) else int(round(float(b)))
        dq = fe.get("dequant_s_per_block", {}).get(platform)
        if isinstance(dq, dict):
            dqi[f], dqb[f] = float(dq["idle"]), float(dq["busy"])
    q = {}
    for name, st in d.get("quality", {}).get(model, {}).items():
        q[FID[name]] = list(st)
    return ModelFormats(model, platform, by, {"idle": dqi, "busy": dqb}, d["len_bins_tokens"], q,
                        [FID[x] for x in d.get("never_read_only", [])], d.get("PLACEHOLDER", False),
                        dict(path=path, sha256=sha256_file(path), run_id=d.get("run_id")))


class Costs:
    """平台常數：f(i)（秒）、GPU 忙倍數、CPU 層、SSD 層。"""

    def __init__(self, f_ms, f_slope_ms, f_icpt_ms, busy_factor, cpu, ssd, platform="custom", src=None):
        self.f_ms, self.f_slope, self.f_icpt = f_ms, f_slope_ms, f_icpt_ms
        self.busy_factor = busy_factor
        self.cpu = cpu        # dict(r, w, k)  GiB/s
        self.ssd = ssd        # dict(r, w, k)
        self.platform = platform
        self.src = src

    @property
    def has_f(self):
        return not isinstance(self.f_ms, str)

    def f_array(self, nmax, gpu):
        """位置 0..nmax-1 的重算時間（秒）＝x1_sim.f_array 的同一個算式（逐位元）。"""
        if not self.has_f:
            raise NotMeasured(f"f(i) on {self.platform}: {self.f_ms}")
        fm = self.f_ms
        bf = _num(self.busy_factor, "busy_factor") if gpu == "busy" else 1.0
        out = []
        for i in range(nmax):
            ms = fm[i] if i < len(fm) else self.f_icpt + self.f_slope * i
            out.append(ms / 1e3 * (bf if gpu == "busy" else 1.0))
        return out


def load_costs(platform, model, ssd_profile="mi300x_local"):
    d = json.load(open(os.path.join(CFG, f"costs_{platform}.json"), encoding="utf-8"))
    m = d["models"][model]
    s = d["ssd_profiles"][ssd_profile]
    cpu = dict(r=m["cpu"]["read_GiBps"], w=m["cpu"]["write_GiBps"], k=m["cpu"]["k_read"])
    ssd = dict(r=s["read_GiBps"], w=s["write_GiBps"], k=s["k_read"])
    for k_, v in ssd.items():
        _num(v, f"ssd_profiles.{ssd_profile}.{k_}")
    return Costs(m["f_ms"], m.get("f_slope_ms"), m.get("f_icpt_ms"), m.get("busy_factor"), cpu, ssd, platform,
                 src=dict(costs_run=d["run_id"], f=m.get("f_src"), ssd=s.get("src"), cpu=m["cpu"].get("src"),
                          cpu_busy_h2d_factor=d.get("cpu_busy_h2d_factor")))


def cpu_busy_factor(platform):
    d = json.load(open(os.path.join(CFG, f"costs_{platform}.json"), encoding="utf-8"))
    return _num(d.get("cpu_busy_h2d_factor"), f"costs_{platform}.cpu_busy_h2d_factor")
