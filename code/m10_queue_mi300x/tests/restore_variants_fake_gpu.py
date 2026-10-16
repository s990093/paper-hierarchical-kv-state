"""09 §6.4：還原器的方法變體（S 組、G 組、K 組）在 GPU 上用一個很小的「假模型」測正確性（只佔幾十 MB，品質段可以跑）。
每次還原後檢查：每個還原單位剛好出現一次；重算的＝參考 prefill、載 BF16 的＝參考、載低精度的＝原寫法 m10_codec.decode
（逐位元）；沒有寫到別的位置（還原前整塊清成 NaN 以外的固定值，結束後全部單位都要對）。
用法：python restore_variants_fake_gpu.py [--quick]
"""
import itertools
import os
import random
import sys
import time

sys.path.insert(0, "/mlsteam/workspace/paper-hierarchical-kv-state/code")
import torch                                    # noqa: E402
import m10_codec as C                           # noqa: E402
import m10_dequant_fast as DF                   # noqa: E402
import m7_restore_harness as RH                 # noqa: E402
from m7_model import CHUNK                      # noqa: E402

DEV = "cuda:0"


class FakeKM:
    """KVModel 的最小替身：forward_span 寫入由 (token, 位置, 層, head, channel) 決定的值；睡一下模擬計算。"""

    def __init__(self, n_blocks, L=4, H=2, D=128, sleep_per_tok=4e-6):
        self.L, self.H, self.D = L, H, D
        self.device = DEV
        self.max_len = n_blocks * CHUNK + CHUNK
        self.kv = torch.empty((L, 2, H, self.max_len, D), dtype=torch.bfloat16, device=DEV)
        self.chunk_bytes = L * 2 * H * CHUNK * D * 2
        self.sleep = sleep_per_tok
        li = torch.arange(L, device=DEV).view(L, 1, 1, 1, 1)
        kv = torch.arange(2, device=DEV).view(1, 2, 1, 1, 1)
        h = torch.arange(H, device=DEV).view(1, 1, H, 1, 1)
        d = torch.arange(D, device=DEV).view(1, 1, 1, 1, D)
        self._base = (li * 11 + kv * 17 + h * 5 + d).float()

    def chunk_view(self, i):
        return self.kv[:, :, :, i * CHUNK:(i + 1) * CHUNK, :]

    @torch.inference_mode()
    def forward_span(self, ids, start):
        n = ids.shape[-1]
        pos = torch.arange(start, start + n, device=DEV).view(1, 1, 1, n, 1).float()
        t = ids.view(1, 1, 1, n, 1).float()
        x = torch.remainder(self._base + t * 7 + pos * 3, 97.0) / 16.0 - 3.0
        x = x + (x.abs() > 2.5).float() * 40.0             # 一些大值（像 K 的離群通道）
        self.kv[:, :, :, start:start + n] = x.to(torch.bfloat16)
        time.sleep(self.sleep * n)


def build(km, ids, n, U, fmt):
    for s0 in range(0, n * CHUNK, U):
        km.forward_span(ids[s0:s0 + U], s0)
    torch.cuda.synchronize()
    ref = km.kv[:, :, :, :n * CHUNK].clone()
    k = CHUNK // U
    bf_u, lp_u = [], []
    for i in range(n):
        blk = km.chunk_view(i)
        for j in range(k):
            bf_u.append(blk[:, :, :, j * U:(j + 1) * U].contiguous().cpu().pin_memory())
        e = C.encode(blk.contiguous(), fmt).map(lambda t: t.cpu())
        lp_u.extend(x.pin() for x in (DF.split_enc(e, U) if U != CHUNK else [e]))
    return ref, bf_u, lp_u


def check(km, rs, rec, ref, lp_u, n_units, U):
    seen = [0] * n_units
    bad = 0
    for i, src, _, _ in rec["blocks"]:
        seen[i] += 1
        v = km.kv[:, :, :, i * U:(i + 1) * U]
        want = ref[:, :, :, i * U:(i + 1) * U] if src in ("recompute", "load_bf16") else C.decode(lp_u[i].to(DEV))
        bad += int(not torch.equal(v, want))
    return bad, seen.count(1) == n_units


def main():
    quick = "--quick" in sys.argv
    torch.manual_seed(0)
    n = 8 if quick else 12
    ids = torch.randint(1000, 30000, (n * CHUNK + 64,), device=DEV)
    fmts = ["fp8_token", "fp8_head", "int4_kivi"] if quick else ["fp8_token", "fp8_head", "fp8_channel", "int8_token",
                                                                  "int4_token", "int4_kivi"]
    variants = [dict(sched="serial"), dict(sched="pipe"), dict(sched="group", group=2), dict(sched="group", group=4),
                dict(sched="defer"), dict(sched="defer_meet"), dict(meet="estimate"), dict(compute_prio=True),
                dict(unit=256), dict(unit=128), dict(unit=256, sched="pipe"), dict(unit=128, sched="group", group=4),
                dict(sched="pipe", compute_prio=True), dict(sched="defer", meet="estimate")]
    kernels = ["k1", "k4", "k3:32:4"] if quick else ["k1", "k4", "k3:32:4", "k3:8:2", "k2"]
    tiers = [("cpu", RH.Tier("cpu_real", read_Bps=0, write_Bps=0)), ("thr", None)]
    ok = True
    t_all = time.time()
    # K 組：單 block 的 decode_variant 和原寫法逐位元相同（寫進大緩衝的視圖）
    x = (torch.randn((4, 2, 2, 512, 128), device=DEV) * 3).to(torch.bfloat16)
    buf = torch.zeros((4, 2, 2, 512 * 5, 128), dtype=torch.bfloat16, device=DEV)
    for fmt, kern in itertools.product(fmts + ["int4_token", "int8_token"], kernels):
        e = C.encode(x, fmt)
        buf.zero_()
        v = buf[:, :, :, 512 * 3:512 * 4]
        DF.decode_variant(e, v, kern)
        good = torch.equal(v, C.decode(e)) and not bool(buf[:, :, :, :512 * 3].any()) and not bool(buf[:, :, :, 512 * 4:].any())
        if kern == "k2" and not good:
            print(f"INFO kernel k2（torch.compile）{fmt} 不逐位元相同（只當對照）", flush=True)
            continue
        ok &= good
        print(("PASS" if good else "FAIL"), "kernel", fmt, kern, flush=True)
    for U in sorted({v.get("unit", CHUNK) for v in variants}, reverse=True):
        for fmt in fmts:
            km = FakeKM(n)
            ref, bf_u, lp_u = build(km, ids, n, U, fmt)
            k = CHUNK // U
            for var in [v for v in variants if v.get("unit", CHUNK) == U]:
                for kern in (kernels if var.get("sched") in ("group", "defer", "defer_meet") else ["k1"]):
                    DF.install(kern)
                    rs = RH.Restorer(km)
                    rs.time_h2d = True
                    rs.set_options(unit=U, sched=var.get("sched", "serial"), group=var.get("group", 1),
                                   compute_prio=var.get("compute_prio", False), meet=var.get("meet", "claim"))
                    for cfg in ("E0", "E1", "R25"):
                        if cfg == "E0":
                            mask = [False] * n
                        elif cfg == "E1":
                            mask = [True] * n
                        else:
                            mask = [False] * n
                            for i in random.Random(7).sample(range(1, n), n // 4):
                                mask[i] = True
                        host = [lp_u[u] if mask[u // k] else bf_u[u] for u in range(n * k)]
                        for tname, tier in tiers:
                            for mode in (("cake", "load_only", "compute_only") if U != CHUNK else ("cake",)):
                                if tier is None:
                                    tier_ = RH.Tier("ssd_thr", read_Bps=2 * RH.GiB, write_Bps=2 * RH.GiB)
                                else:
                                    tier_ = tier
                                km.kv.fill_(7.0)
                                torch.cuda.synchronize()
                                rec = rs.restore(ids, n * k, host, [tier_] * (n * k), mode)
                                bad, once = check(km, rs, rec, ref, lp_u, n * k, U)
                                good = bad == 0 and once
                                ok &= good
                                tag = f"U={U} {fmt} {var} kern={kern} {cfg} {tname} {mode}"
                                print(("PASS" if good else "FAIL"), tag, f"meet={rec['meet']} bad={bad} 每單位一次={once}"
                                      f" dq={rec['t_dequant'] * 1e3:.2f}ms h2d={rec['t_h2d'] * 1e3:.2f}ms", flush=True)
    DF.install("k1")
    print(f"耗時 {time.time() - t_all:.0f}s")
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
