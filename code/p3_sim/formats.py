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

    def status_32k(self, fid):
        """32,768 token 那一點的狀態（O-寬鬆用：32K 過關＝任何長度都當過關）。"""
        return self.status(fid, QUALITY_MAX_TOKENS)

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


DEFAULT_DQ_SET = "cake_k0"


def split_format_path(path):
    """'official'／None → (正式表, 預設解壓欄)；'official:<欄組>' → (正式表, 欄組)；其他原樣 (path, None)。
    欄組見 DQ_SETS（06 §19）。placeholder／自己造的表沒有欄組。"""
    if path is None or path == "official":
        return OFFICIAL_FORMAT_TABLE, DEFAULT_DQ_SET
    if isinstance(path, str) and path.startswith("official:"):
        return OFFICIAL_FORMAT_TABLE, path.split(":", 1)[1]
    if path == "placeholder":
        return PLACEHOLDER_FORMAT_TABLE, None
    return path, None


def resolve_format_path(path):
    """None／'official'／'official:<欄組>' → 正式表；'placeholder' → placeholder；其他＝檔案路徑。
    2026-10-18（06 §11）：預設改成正式表（04 §9：地圖不准用 placeholder）。舊的冒煙／回歸要明確給 'placeholder'。"""
    return split_format_path(path)[0]


# ---- 正式表（results/p3_formats/format_table.json，「每列＝模型×格式」）→ 模擬器的 fid（06 §11）
OFFICIAL_CONCRETE = {"llama": {1: "bf16", 2: "fp8_token", 3: "int4_kivi", 4: "int2_kivi"},
                     "qwen": {1: "bf16", 2: "int8_token", 3: "int4_kivi", 4: "int2_kivi"}}   # Qwen 的 8-bit＝INT8（FP8 掉 32.8 分）
# 解壓常數：和 placeholder 同一個欄位語意（build_inputs.py）。MI300X＝X1 的規則（E1、load_only、CPU 層）；3090＝X7 的還原中（cake）
# 解壓常數的欄組（06 §19；04 v4 §9）：
#   cake_k0     ＝「還原中・Cake・K0」（05 §3.2 的建議，含和重算搶 GPU 的成本）；**v4 的預設**。
#   loadonly_k0 ＝「還原中・只載入・K0」（06 §11.1 的舊選擇：MI300X 一律用只載入欄）；還可以選，結果會不同。
# 3090 只有 cake_k0（X7 的還原中）；兩組在 3090 上相同。
DQ_SETS = {"cake_k0": {"mi300x": ("dq_mi300x_cake_k0_idle_ms", "dq_mi300x_cake_k0_busy_ms"),
                       "3090": ("dq_3090_cake_k0_idle_ms", "dq_3090_cake_k0_busy_ms")},
           "loadonly_k0": {"mi300x": ("dq_mi300x_loadonly_k0_idle_ms", "dq_mi300x_loadonly_k0_busy_ms"),
                           "3090": ("dq_3090_cake_k0_idle_ms", "dq_3090_cake_k0_busy_ms")}}
OFFICIAL_DQ_COLS = DQ_SETS[DEFAULT_DQ_SET]               # 舊名字（預設欄組）
GEN_SCENARIOS = ("A", "B_idle", "B_busy", "B_i2b")     # 生成品質的情境：全載入＋Cake b32 閒／忙／閒→忙
QUALITY_MAX_TOKENS = 32768                              # 品質只在 32K 實測；> 32,768 一律 unverified；32,768 本身算在實測的分箱
ARCH_ONLY = {"llama70b": 327680}                        # 正式表沒有的模型：只有 bf16（架構算的大小），其他 NOT_MEASURED


def _cell(row, k):
    c = row["cells"].get(k)
    return None if c is None else c.get("value")


def official_quality_32k(row):
    """一列（模型×格式）在 32K 的生成品質狀態。
    pass＝每個「有回報」的生成情境（A、B 閒／忙／閒→忙）的 95% CI 下界都 > −2 分；有任一個 ≤ −2 → fail；
    一個情境都沒有（只有 teacher forcing、PENDING、NOT_MEASURED）→ unverified。
    同時用表裡的 pass2 布林值驗算 CI 下界（規則 6：用 A 欄驗算 B 欄），對不上就丟錯。回傳 (狀態, 說明)。"""
    rep = []
    for s in GEN_SCENARIOS:
        lo = _cell(row, f"q_gen_{s}_ci_lo_pts")
        if isinstance(lo, bool) or not isinstance(lo, (int, float)):
            continue
        ok = float(lo) > -2.0
        p2 = _cell(row, f"q_gen_{s}_pass2")
        if isinstance(p2, bool) and p2 != ok:
            raise AssertionError(f"{row['model']} {row['fmt']} {s}: CI 下界 {lo} 和 pass2={p2} 對不上")
        rep.append((s, float(lo), ok))
    if not rep:
        return "unverified", "沒有生成品質（只有 teacher forcing／PENDING／NOT_MEASURED）"
    st = "pass" if all(x[2] for x in rep) else "fail"
    return st, ";".join(f"{s}:lo={lo:+.2f}" for s, lo, _ in rep)


def _load_official(d, path, model, platform, dq_set=DEFAULT_DQ_SET):
    """正式表 → ModelFormats。位元組＝bytes_per_block_code（同時斷言＝encode 的 nbytes＝實際存的大小、＝架構推的 BF16 大小）。"""
    sha = sha256_file(path)
    if model in ARCH_ONLY:
        bpt = ARCH_ONLY[model]
        by = [None, bpt * 512, None, None, None]
        q = {FID["bf16"]: ["pass", "pass"]}
        return ModelFormats(model, platform, by, {"idle": [None, 0.0, None, None, None], "busy": [None, 0.0, None, None, None]},
                            [0, QUALITY_MAX_TOKENS + 1, 1 << 40], q, [FID["int2"]], False,
                            dict(path=path, sha256=sha, run_id=d.get("run_id"), kind="official", dq_set=dq_set,
                                 note=f"{model} 不在正式表：bf16＝{bpt}×512（架構），其他 NOT_MEASURED"))
    if dq_set not in DQ_SETS:
        raise ValueError(f"解壓欄組 {dq_set!r} 不存在（{list(DQ_SETS)}）")
    if platform not in DQ_SETS[dq_set]:
        raise NotImplementedError(f"正式表沒有平台 {platform} 的解壓欄（{list(DQ_SETS[dq_set])}）")
    sh = d["shape"][model]
    bf_arch = sh["L"] * sh["H_kv"] * sh["D"] * 2 * 2 * sh["T_block"]       # K、V × BF16 2 bytes
    rows = {r["fmt"]: r for r in d["rows"] if r["model"] == model}
    by, dqi, dqb = [None] * len(FMT_NAMES), [None] * len(FMT_NAMES), [None] * len(FMT_NAMES)
    q, notes, conc = {}, {}, {}
    ci, cb = DQ_SETS[dq_set][platform]
    for fid, fmt in OFFICIAL_CONCRETE[model].items():
        r = rows[fmt]
        b = _cell(r, "bytes_per_block_code")
        be = _cell(r, "bytes_per_block_encode_cpu")
        agree = _cell(r, "bytes_measured_agree_with_code")
        kib = _cell(r, "kib_per_token")
        # 規則 6：位元組用三個來源互相驗算（公式＝encode＝實際存的；KiB/token 欄；BF16＝架構）
        assert isinstance(b, int) and b == be and agree is True, (model, fmt, b, be, agree)
        assert abs(b / sh["T_block"] / 1024 - float(kib)) < 1e-3, (model, fmt, b, kib)
        if fid == 1:
            assert b == bf_arch, (model, b, bf_arch)
        by[fid] = int(b)
        for col, arr in ((ci, dqi), (cb, dqb)):
            v = _cell(r, col)
            arr[fid] = None if (isinstance(v, str) or v is None) else float(v) / 1e3     # ms → 秒；NOT_MEASURED＝None
        st, why = ("pass", "bf16（參考）") if fid == 1 else official_quality_32k(r)
        q[fid] = [st, "unverified"]          # 分箱 0：≤ 32,768（含 32,768）＝32K 的狀態；分箱 1：> 32,768＝未驗證
        verdict = _cell(r, "quality_verdict_32k")
        expect = {"PASS": "pass", "FAIL": "fail", "UNVERIFIED_TF_ONLY": "unverified", "REFERENCE": "pass"}.get(verdict)
        notes[FMT_NAMES[fid]] = dict(concrete=fmt, status32=st, why=why, table_verdict=verdict,
                                     agrees_with_table_verdict=(expect == st))
        conc[fid] = fmt
    q[FID["bf16"]] = ["pass", "pass"]
    nro = [FID["int2"]] if q[FID["int2"]][0] != "pass" else []      # INT2 沒有生成品質過關 → 只准給不會再被讀的
    return ModelFormats(model, platform, by, {"idle": dqi, "busy": dqb}, [0, QUALITY_MAX_TOKENS + 1, 1 << 40], q, nro, False,
                        dict(path=path, sha256=sha, run_id=d.get("run_id"), kind="official", dq_set=dq_set, dq_cols=[ci, cb],
                             concrete={FMT_NAMES[k]: v for k, v in conc.items()}, quality=notes))


SCALED_70B = "llama70b_scaled"      # 04 v4 §5 的「70B 級，只算位元組」：Llama 的格式表，位元組×(327,680/131,072)；沒有解壓與 TTFT（06 §20）


def _scaled_70b(platform, path):
    """70B 級（約 320 KiB／token）的位元組：Llama 每個格式的位元組 × 2.5（每 token 的 BF16 大小比 327,680 / 131,072；其他格式假設和 BF16 的比例相同）。
    **這是假設，不是量測**：只給敏感度（只算位元組、不談品質）。解壓常數 NOT_MEASURED（BF16＝0）；品質狀態沿用 Llama 的（只拿來選格式）。"""
    base = load_format_table("llama", platform, path)
    per_tok = base.bytes[FID["bf16"]] // 512
    assert per_tok * 512 == base.bytes[FID["bf16"]]
    by = []
    for b in base.bytes:
        if b is None:
            by.append(None)
            continue
        assert (b * ARCH_ONLY["llama70b"]) % per_tok == 0, b
        by.append(b * ARCH_ONLY["llama70b"] // per_tok)
    dq = {g: [None, 0.0, None, None, None] for g in ("idle", "busy")}
    src = dict(base.src, kind="official+scaled70b", note=f"70B 級＝Llama 的位元組 × {ARCH_ONLY['llama70b']}/{per_tok}（假設）；解壓 NOT_MEASURED",
               scaled_from="llama")
    return ModelFormats(SCALED_70B, platform, by, dq, base.len_bins, dict(base.quality), base.never_read_only, False, src)


def load_format_table(model, platform, path=None):
    """path：None／'official'＝正式表 results/p3_formats/format_table.json（2026-10-18 起的預設，06 §11）；
    'placeholder'＝config/format_table.PLACEHOLDER.json（舊的冒煙與回歸用）；其他＝檔案路徑（測試自己造的表）。
    正式表的結構是「每列＝模型×格式」，由 _load_official 轉換（不改引擎）。"""
    if model == SCALED_70B:
        return _scaled_70b(platform, path)
    path, dq_set = split_format_path(path)
    d = json.load(open(path, encoding="utf-8"))
    if "rows" in d and "shape" in d and "models" not in d:
        return _load_official(d, path, model, platform, dq_set or DEFAULT_DQ_SET)
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
                        dict(path=path, sha256=sha256_file(path), run_id=d.get("run_id"),
                             kind="placeholder" if d.get("PLACEHOLDER") else "custom"))


class Costs:
    """平台常數：f(i)（秒）、GPU 忙倍數、CPU 層、SSD 層。"""

    def __init__(self, f_ms, f_slope_ms, f_icpt_ms, busy_factor, cpu, ssd, platform="custom", src=None, ssd_k_nm=False):
        self.ssd_k_nm = bool(ssd_k_nm)     # True：SSD 的 k_read（讀受寫拖慢的倍數）沒量；ssd["k"] 只是占位的 1.0，View 追蹤哪些請求的 TTFT 碰到它（06 §24）
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


SSD8_CONSTANTS = os.path.join(REPO, "results", "p3_calib_3090", "constants_3090_v2.json")


def _ssd8_profile():
    """3090 的 /ssd8（SATA QLC）：讀、寫頻寬＝results/p3_calib_3090 的實測中位數（run 20261018-001935-p3cal-ssd8bw）。
    讀受寫拖慢的倍數 k_read：/ssd8 **沒量**（disk_bw 只量單向）。04 v4 §5「只用量到的常數」→ **不借 MI300X 的 3.306**：
    k_read 在這裡是占位的 1.0（k_nm＝True），View 追蹤「TTFT 碰到 k_read 的請求」（讀被寫拖慢、排在被拖慢的讀後面、或時間由這種請求的 TTFT 推出）
    → 那些請求的 ttftq＝NOT_MEASURED；位元組不受影響。只有敏感度格子會用，src 欄寫明。"""
    d = json.load(open(SSD8_CONSTANTS, encoding="utf-8"))["new_in_v2"]["ssd8_disk"]
    return dict(read_GiBps=float(d["read_GiBps_median"]), write_GiBps=float(d["write_GiBps_median"]), k_read=1.0, k_nm=True,
                src=f"{SSD8_CONSTANTS}（run {d['run_id']}）讀 {d['read_GiBps_median']}、寫 {d['write_GiBps_median']} GiB/s；"
                    f"k_read＝NOT_MEASURED on /ssd8（不借別台機器的值；占位 1.0，碰到它的請求 ttftq＝NOT_MEASURED；06 §24）")


def cpu_factor_of(platform, cpu):
    """CPU 忙閒的 H2D 變慢倍數：'idle'→1；數字→照用（04 v4 §5 的敏感度 1.35、1.81）；'busy'→costs_<平台>.json（NOT_MEASURED 就丟錯）。"""
    if cpu == "idle":
        return 1.0
    if isinstance(cpu, (int, float)):
        return float(cpu)
    return cpu_busy_factor(platform)


def load_costs(platform, model, ssd_profile="mi300x_local"):
    d = json.load(open(os.path.join(CFG, f"costs_{platform}.json"), encoding="utf-8"))
    m = d["models"]["llama70b" if model == SCALED_70B else model]
    s = _ssd8_profile() if ssd_profile == "ssd8_3090_meas" else d["ssd_profiles"][ssd_profile]
    cpu = dict(r=m["cpu"]["read_GiBps"], w=m["cpu"]["write_GiBps"], k=m["cpu"]["k_read"])
    ssd = dict(r=s["read_GiBps"], w=s["write_GiBps"], k=s["k_read"])
    for k_, v in ssd.items():
        _num(v, f"ssd_profiles.{ssd_profile}.{k_}")
    return Costs(m["f_ms"], m.get("f_slope_ms"), m.get("f_icpt_ms"), m.get("busy_factor"), cpu, ssd, platform,
                 src=dict(costs_run=d["run_id"], f=m.get("f_src"), ssd=s.get("src"), cpu=m["cpu"].get("src"),
                          cpu_busy_h2d_factor=d.get("cpu_busy_h2d_factor")), ssd_k_nm=bool(s.get("k_nm")))


def cpu_busy_factor(platform):
    d = json.load(open(os.path.join(CFG, f"costs_{platform}.json"), encoding="utf-8"))
    return _num(d.get("cpu_busy_h2d_factor"), f"costs_{platform}.cpu_busy_h2d_factor")
