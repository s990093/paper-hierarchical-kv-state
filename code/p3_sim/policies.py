"""policies.py — 壓縮決定（我們唯一決定的東西，04 §1、§3）。

介面：每個 session 在一輪結束、把新 block 寫進 CPU 時，引擎呼叫一次 cpu_fid(ctx)；
外層把 block 搬去 SSD（降級、寫穿、第 2 次才寫）時呼叫 ssd_fid(ctx, cur)，回傳值必須 ≥ cur（只能往下壓）。

ctx（dict，引擎填）：
  r（請求 index）、s（session）、t（到達秒）、n_blocks（這一輪存的完整 block 數）、tokens（＝n_blocks×block_tokens）、
  n_new（這一輪新寫進 CPU 的 block 數）、nreq（這個 session 到這一輪為止的請求數）、kind（0 新、1 接續、2 分叉）、
  feats（trace 給的、不含任何未來資訊的特徵；可能是 None）、mf（ModelFormats）。
  只有 needs_oracle=True 的策略會另外拿到：next_read（之後第一次讀到這一輪新 block 的請求 index；不會再讀＝None）、
  max_read_tokens（之後讀到這些 block 的請求裡，存的前綴最長的 token 數；讀取長度的上界）、
  censored（決定時刻離觀察窗結尾 < H 秒：「之後有沒有被讀」不算分 → 一律當成會被讀）。
  另外（06 §18，嚴格讀取長度）：req_tokens（現在的上下文長度＝寫入的請求，或這個 session 最近一個請求的輸入長度；所有策略都有）、
  max_read_req_tokens（oracle：之後讀到這些 block 的請求裡，輸入長度最長的 token 數）。

學出來的預測器：PredictorPolicy(fn)，fn(state: dict, feats) → 格式名或 fid。state＝ctx 去掉 oracle 欄位與 mf。
這一步不訓練任何模型（CLAUDE.md §8）；介面只確保預測器拿不到標籤。

ssd_moment=True 的策略（2026-10-18，06 §11.2）：ssd_fid(ctx, cur) 的 ctx 是「外層要寫 SSD 的那一刻」的資訊（每個 block 一次）：
  s、r（決定時刻的請求 index：之後＝index > r）、t（決定時刻）、b（block）、tokens（這個 session 那一刻存的長度）、
  idle_s（離它上一個請求的到達多久）、cpu_ctx（寫進 CPU 時的 ctx）；needs_oracle 的另外有這個 block 在 r 之後的
  next_read、max_read_tokens、censored。ssd_moment=False 的策略拿到的是寫進 CPU 時的 ctx（c0c179b 的行為，C0–C4 不變）。
"""
from __future__ import annotations

from formats import FID, FMT_NAMES, NotMeasured


def read_len(ctx, rule):
    """長度門檻用的長度（04 v4 §3.1，第二輪審查 #8）。rule＝"stored"（c0c179b 的舊行為＝存的長度）｜"max"＝max(存的長度, 這個請求的輸入長度)
    （兩個都是決定當下就知道的；地圖用）。"""
    if rule == "max":
        return max(ctx["tokens"], ctx.get("req_tokens", 0) or 0)
    assert rule == "stored", rule
    return ctx["tokens"]


class CompressionPolicy:
    name = "base"
    needs_oracle = False
    ssd_moment = False      # True：ssd_fid 拿到「寫 SSD 那一刻」的 ctx（引擎的 _mctx）；False：寫進 CPU 時的 ctx（舊行為）

    def __init__(self, mf):
        self.mf = mf

    def _need(self, fname):
        f = FID[fname]
        if self.mf.bytes[f] is None:
            raise NotMeasured(f"{self.name}: {self.mf.model} {fname} 沒有大小（格式表 NOT_MEASURED）")
        return f

    def cpu_fid(self, ctx):
        raise NotImplementedError

    def ssd_fid(self, ctx, cur):
        return cur


class Fixed(CompressionPolicy):
    """C0／C2／C3：CPU 與 SSD 都同一個格式。"""

    def __init__(self, mf, fname, name):
        super().__init__(mf)
        self.name = name
        self.f = self._need(fname)

    def cpu_fid(self, ctx):
        return self.f


class C1(CompressionPolicy):
    """論文的做法：CPU 放 BF16，搬去 SSD 時壓 INT4。"""
    name = "C1"

    def __init__(self, mf):
        super().__init__(mf)
        self.fb, self.f4 = self._need("bf16"), self._need("int4")

    def cpu_fid(self, ctx):
        return self.fb

    def ssd_fid(self, ctx, cur):
        return max(cur, self.f4)


class C4(CompressionPolicy):
    """依長度：寫入時就知道的長度 ≤ 門檻 → BF16，否則 INT4。門檻是參數（預設 32,768 token＝品質實測的長度；要在登記時凍結）。"""
    name = "C4"

    def __init__(self, mf, thr_tokens=32768, len_rule="stored"):
        super().__init__(mf)
        self.thr, self.len_rule = thr_tokens, len_rule
        self.fb, self.f4 = self._need("bf16"), self._need("int4")

    def cpu_fid(self, ctx):
        return self.fb if read_len(ctx, self.len_rule) <= self.thr else self.f4


class Oracle(CompressionPolicy):
    """O：完美預測。之後會被讀（或落在截尾區、不知道）→ 品質過關、又最省（位元組最少）的格式；
    不會再被讀 → 最省的格式（含 never_read_only 的 INT2；沒有大小的格式跳過）。
    allow_unverified=False（預設）：『未驗證』不算過關。主指標是位元組，所以「最省」＝位元組最少；
    讀取時的層與 GPU 忙閒只影響 TTFT（報告指標），這版 O 不拿來決定。

    2026-10-18 的修改（06 §11.2）：ssd_oracle=True（預設）時，外層把 block 搬去 SSD（降級、寫穿、第 2 次才寫、閒置降級）的那一刻
    也用 oracle 再決定一次（每個 block）：那一刻之後不會再被讀（而且不在截尾區）→ 最省的格式（INT2）；
    否則＝「那一刻之後最長的讀取長度」下品質過關、最省的格式；一律不比 cur 好（只能往下壓）。
    ssd_oracle=False＝c0c179b 的舊 O（SSD 維持 CPU 的格式），只給回歸比對用。

    variant（06 §11.3）：
      O        ＝上面（判定用，O-嚴格）
      O-len    ＝只知道「之後最長讀到多長」，不知道會不會再被讀（每個 block 都當成會被讀；不用 INT2）
      O-reuse  ＝只知道「之後有沒有被讀」：不會再被讀 → INT2；會 → 用寫入當下的長度選過關、最省的（和非 oracle 的規則一樣）。
                 session 之後變長時會讀到不過關的格式：照實記在品質帳
      O-loose  ＝同 O，但 32K 生成品質過關的格式在任何長度都當成過關（只報告）；INT2 仍只給不會再被讀的（除非它有生成品質過關）。
                 和 allow_unverified=True 不同：後者把所有「未驗證」都當過關"""
    name = "O"
    needs_oracle = True
    ssd_moment = True
    VARIANTS = ("O", "O-len", "O-reuse", "O-loose")

    def __init__(self, mf, allow_unverified=False, variant="O", ssd_oracle=True, read_len="prefix"):
        super().__init__(mf)
        assert variant in self.VARIANTS, variant
        assert read_len in ("prefix", "request"), read_len
        self.read_len = read_len
        self.variant = variant
        self.name = variant if ssd_oracle else variant + "(cpu-only)"
        self.ssd_oracle = ssd_oracle
        self.ssd_moment = bool(ssd_oracle)
        self.allow_unverified = allow_unverified
        self.ok = ("pass", "unverified") if allow_unverified else ("pass",)
        self.never = mf.cheapest(mf.available)
        self._cache = {}

    def read_fid(self, tokens):
        mf = self.mf
        if self.variant == "O-loose":       # 32K 過關＝任何長度都過關
            f = self._cache.get("loose")
            if f is None:
                cand = [x for x in mf.available if x not in mf.never_read_only and mf.status_32k(x) == "pass"]
                f = self._cache["loose"] = mf.cheapest(cand)
            return f
        b = mf.bin_of(tokens)
        f = self._cache.get(b)
        if f is None:
            cand = [x for x in mf.available if x not in mf.never_read_only and mf.status(x, tokens) in self.ok]
            f = self._cache[b] = mf.cheapest(cand)
        return f

    def _pick_req(self, ctx):
        """read_len="request"（04 v4 §3.3 嚴格版）：讀取長度＝讀它的請求的輸入長度。有之後的讀取 → 對「之後讀到這個 block 的請求裡
        輸入最長的」過關（max_read_req_tokens，不再和寫入長度取 max）；沒有（截尾區、不知道之後）→ 至少用現在的長度（max(存的長度, 這個 session 最近請求的輸入長度)）。"""
        never_read = ctx["next_read"] is None and not ctx["censored"]
        v = self.variant
        cur = max(ctx["tokens"], ctx.get("req_tokens", 0))
        mr = ctx.get("max_read_req_tokens", 0) if ctx["next_read"] is not None else cur
        if v == "O-len":
            return self.read_fid(mr)
        if never_read:
            return self.never
        if v == "O-reuse":
            return self.read_fid(cur)
        return self.read_fid(mr)

    def _pick(self, ctx):
        if self.read_len == "request":
            return self._pick_req(ctx)
        never_read = ctx["next_read"] is None and not ctx["censored"]
        v = self.variant
        if v == "O-len":
            return self.read_fid(max(ctx["tokens"], ctx.get("max_read_tokens", 0)))
        if never_read:
            return self.never
        if v == "O-reuse":
            return self.read_fid(ctx["tokens"])
        # 格式只能往下壓 → 要對之後每一次讀都過關：用之後讀到這些 block 的最長請求的長度（截尾區不知道 → 至少用寫入時的長度）
        return self.read_fid(max(ctx["tokens"], ctx.get("max_read_tokens", 0)))

    def cpu_fid(self, ctx):
        return self._pick(ctx)

    def ssd_fid(self, ctx, cur):
        if not self.ssd_oracle:
            return cur
        f = self._pick(ctx)          # ctx＝寫 SSD 那一刻、這個 block 的 oracle 資訊（引擎的 _mctx）
        return f if f > cur else cur


class PredictorPolicy(CompressionPolicy):
    """學出來的預測器的掛鉤：fn(state, feats) → 格式名或 fid；ssd_fn 可省（省略＝維持 CPU 的格式）。
    回傳的格式必須有大小；若選了 never_read_only 的格式（INT2）而 session 之後被讀，品質帳會記下來（不擋）。
    ssd_fn(state, feats, cur)：外層要寫 SSD 的那一刻呼叫（06 §11.4）；state＝那一刻的 ctx（s、r、t、b、tokens、idle_s、cpu_state）
    去掉 oracle 欄位。回傳值和 cur 取較省的（只能往下壓）。"""

    ORACLE_KEYS = ("next_read", "censored", "max_read_tokens", "max_read_req_tokens", "mf", "feats", "cpu_ctx")

    def __init__(self, mf, fn, ssd_fn=None, name="P"):
        super().__init__(mf)
        self.fn, self.ssd_fn, self.name = fn, ssd_fn, name
        self.ssd_moment = ssd_fn is not None

    @classmethod
    def _state(cls, ctx):
        st = {k: v for k, v in ctx.items() if k not in cls.ORACLE_KEYS}
        if ctx.get("cpu_ctx") is not None:
            st["cpu_state"] = {k: v for k, v in ctx["cpu_ctx"].items() if k not in cls.ORACLE_KEYS}
        return st

    def _fid(self, x):
        f = FID[x] if isinstance(x, str) else int(x)
        if not (1 <= f < len(FMT_NAMES)) or self.mf.bytes[f] is None:
            raise NotMeasured(f"predictor 選了沒有大小的格式 {x!r}")
        return f

    def cpu_fid(self, ctx):
        return self._fid(self.fn(self._state(ctx), ctx.get("feats")))

    def ssd_fid(self, ctx, cur):
        if self.ssd_fn is None:
            return cur
        feats = ctx.get("feats")
        if feats is None and ctx.get("cpu_ctx") is not None:
            feats = ctx["cpu_ctx"].get("feats")
        return max(cur, self._fid(self.ssd_fn(self._state(ctx), feats, cur)))


# ---------------------------------------------------------------- 06 §11.4：更多非 oracle 的對照（參數是建構子的引數，值由 v4 凍結）
class TwoFormat(CompressionPolicy):
    """CPU 一個格式、搬去 SSD 時壓成另一個（不會比 CPU 好）。C1＝(bf16, int4)；C1-8＝(bf16, 8bit)；C6＝(8bit, int4)。"""

    def __init__(self, mf, cpu_fmt, ssd_fmt, name):
        super().__init__(mf)
        self.name = name
        self.fc, self.fs = self._need(cpu_fmt), self._need(ssd_fmt)

    def cpu_fid(self, ctx):
        return self.fc

    def ssd_fid(self, ctx, cur):
        return max(cur, self.fs)


class C4r(CompressionPolicy):
    """C4 反過來：寫入時的長度 ≤ 門檻 → INT4，否則 BF16（預設門檻 32,768 token）。SSD 維持 CPU 的格式。"""
    name = "C4r"

    def __init__(self, mf, thr_tokens=32768, len_rule="stored"):
        super().__init__(mf)
        self.thr, self.len_rule = thr_tokens, len_rule
        self.fb, self.f4 = self._need("bf16"), self._need("int4")

    def cpu_fid(self, ctx):
        return self.f4 if read_len(ctx, self.len_rule) <= self.thr else self.fb


class Deferred(CompressionPolicy):
    """延到搬去 SSD 那一刻才決定（只用那一刻拿得到的資訊；CPU 一律 BF16）。
      D-len ：那一刻這個 session 存的長度 ≤ thr → INT4，否則維持（預設 thr＝32,768 token）
      D-idle：那一刻離這個 session 上一個請求的到達 > tau 秒 → INT2，否則同 D-len（預設 tau＝1,800 s＝H、E5 的 30 分）。
              INT2 只有 teacher forcing：之後被讀就記在品質帳（或 int2_on_read=recompute 時改成重算）"""
    ssd_moment = True

    def __init__(self, mf, name, thr_tokens=32768, tau_s=None, len_rule="stored"):
        super().__init__(mf)
        self.name = name
        self.thr, self.tau, self.len_rule = thr_tokens, tau_s, len_rule
        self.fb, self.f4 = self._need("bf16"), self._need("int4")
        self.f2 = self._need("int2") if tau_s is not None else None

    def cpu_fid(self, ctx):
        return self.fb

    def ssd_fid(self, ctx, cur):
        if self.tau is not None and ctx["idle_s"] > self.tau:
            return max(cur, self.f2)
        return max(cur, self.f4) if read_len(ctx, self.len_rule) <= self.thr else cur


class R2(CompressionPolicy):
    """R2（04 v4 §3.1，第二輪審查 #1）：「先當成不會回來」。這個 session 到現在只有 1 個請求 → INT2（寫進 CPU 與搬去 SSD 都是）；
    來過 ≥2 個請求 → 照 D-len（CPU BF16；搬去 SSD 時長度 ≤ thr → INT4，否則維持；thr 預設 32,768）。
    INT2 沒有任何長度的生成品質驗證，所以單輪 session 回來時會重算（嚴格讀取規則）、重算出來的 KV 照當時的決定（這時 nreq≥2 → BF16）重存。
    nreq＝ctx["nreq"]：這個 session 到那一刻為止的請求數（引擎填；CPU 寫入＝含目前這個請求；SSD 寫入＝決定當下已經服務的請求數，寫穿時含目前這個）。"""
    name = "R2"
    ssd_moment = True

    def __init__(self, mf, thr_tokens=32768, len_rule="stored"):
        super().__init__(mf)
        self.thr, self.len_rule = thr_tokens, len_rule
        self.fb, self.f4, self.f2 = self._need("bf16"), self._need("int4"), self._need("int2")

    def cpu_fid(self, ctx):
        return self.f2 if ctx["nreq"] == 1 else self.fb

    def ssd_fid(self, ctx, cur):
        if ctx["nreq"] == 1:
            return max(cur, self.f2)
        return max(cur, self.f4) if read_len(ctx, self.len_rule) <= self.thr else cur


def make_policy(name, mf, **kw):
    if name == "C0":
        return Fixed(mf, "bf16", "C0")
    if name == "C1":
        return C1(mf)
    if name == "C2":
        return Fixed(mf, "int4", "C2")
    if name == "C3":
        return Fixed(mf, "8bit", "C3")
    lr = kw.get("len_rule", "stored")
    if name == "C4":
        return C4(mf, kw.get("c4_threshold_tokens", 32768), lr)
    if name in Oracle.VARIANTS:
        return Oracle(mf, kw.get("o_allow_unverified", False), variant=name, ssd_oracle=kw.get("o_ssd_oracle", True),
                      read_len=kw.get("o_read_len", "prefix"))
    if name == "C4r":
        return C4r(mf, kw.get("c4_threshold_tokens", 32768), lr)
    if name == "C1-8":
        return TwoFormat(mf, "bf16", "8bit", "C1-8")
    if name == "C6":
        return TwoFormat(mf, "8bit", "int4", "C6")
    if name == "C1-2":
        return TwoFormat(mf, "bf16", "int2", "C1-2")
    if name == "R2":
        return R2(mf, kw.get("d_threshold_tokens", 32768), lr)
    if name == "D-len":
        return Deferred(mf, "D-len", kw.get("d_threshold_tokens", 32768), None, lr)
    if name == "D-idle":
        return Deferred(mf, "D-idle", kw.get("d_threshold_tokens", 32768), kw.get("d_idle_tau_s", 1800.0), lr)
    raise ValueError(name)


POLICY_NAMES = ["C0", "C1", "C2", "C3", "C4", "O"]
# 06 §11：新加的（名字和預設參數見各 class；值由 v4 凍結）
POLICY_NAMES_V4 = POLICY_NAMES + ["O-len", "O-reuse", "O-loose", "C4r", "C1-8", "C6", "D-len", "D-idle"]
# 04 v4 第二輪審查之後：C1-2（CPU BF16、SSD INT2）、R2；長度規則 len_rule="max"（地圖用）
POLICY_NAMES_V4B = POLICY_NAMES_V4 + ["C1-2", "R2"]
