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

學出來的預測器：PredictorPolicy(fn)，fn(state: dict, feats) → 格式名或 fid。state＝ctx 去掉 oracle 欄位與 mf。
這一步不訓練任何模型（CLAUDE.md §8）；介面只確保預測器拿不到標籤。
"""
from __future__ import annotations

from formats import FID, FMT_NAMES, NotMeasured


class CompressionPolicy:
    name = "base"
    needs_oracle = False

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

    def __init__(self, mf, thr_tokens=32768):
        super().__init__(mf)
        self.thr = thr_tokens
        self.fb, self.f4 = self._need("bf16"), self._need("int4")

    def cpu_fid(self, ctx):
        return self.fb if ctx["tokens"] <= self.thr else self.f4


class Oracle(CompressionPolicy):
    """O：完美預測。之後會被讀（或落在截尾區、不知道）→ 品質過關、又最省（位元組最少）的格式；
    不會再被讀 → 最省的格式（含 never_read_only 的 INT2；沒有大小的格式跳過）。
    allow_unverified=False（預設）：『未驗證』不算過關。主指標是位元組，所以「最省」＝位元組最少；
    讀取時的層與 GPU 忙閒只影響 TTFT（報告指標），這版 O 不拿來決定。"""
    name = "O"
    needs_oracle = True

    def __init__(self, mf, allow_unverified=False):
        super().__init__(mf)
        self.allow_unverified = allow_unverified
        self.ok = ("pass", "unverified") if allow_unverified else ("pass",)
        self.never = mf.cheapest(mf.available)
        self._cache = {}

    def read_fid(self, tokens):
        b = self.mf.bin_of(tokens)
        f = self._cache.get(b)
        if f is None:
            cand = [x for x in self.mf.available if x not in self.mf.never_read_only and self.mf.status(x, tokens) in self.ok]
            f = self._cache[b] = self.mf.cheapest(cand)
        return f

    def cpu_fid(self, ctx):
        if ctx["next_read"] is None and not ctx["censored"]:
            return self.never
        # 格式只能往下壓 → 要對之後每一次讀都過關：用之後讀到這些 block 的最長請求的長度（截尾區不知道 → 至少用寫入時的長度）
        return self.read_fid(max(ctx["tokens"], ctx.get("max_read_tokens", 0)))


class PredictorPolicy(CompressionPolicy):
    """學出來的預測器的掛鉤：fn(state, feats) → 格式名或 fid；ssd_fn 可省（省略＝維持 CPU 的格式）。
    回傳的格式必須有大小；若選了 never_read_only 的格式（INT2）而 session 之後被讀，品質帳會記下來（不擋）。"""

    def __init__(self, mf, fn, ssd_fn=None, name="P"):
        super().__init__(mf)
        self.fn, self.ssd_fn, self.name = fn, ssd_fn, name

    @staticmethod
    def _state(ctx):
        return {k: v for k, v in ctx.items() if k not in ("next_read", "censored", "max_read_tokens", "mf", "feats")}

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
        return max(cur, self._fid(self.ssd_fn(self._state(ctx), ctx.get("feats"), cur)))


def make_policy(name, mf, **kw):
    if name == "C0":
        return Fixed(mf, "bf16", "C0")
    if name == "C1":
        return C1(mf)
    if name == "C2":
        return Fixed(mf, "int4", "C2")
    if name == "C3":
        return Fixed(mf, "8bit", "C3")
    if name == "C4":
        return C4(mf, kw.get("c4_threshold_tokens", 32768))
    if name == "O":
        return Oracle(mf, kw.get("o_allow_unverified", False))
    raise ValueError(name)


POLICY_NAMES = ["C0", "C1", "C2", "C3", "C4", "O"]
