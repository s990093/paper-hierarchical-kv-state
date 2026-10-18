"""13 號計畫（10/17）的新程式在 CPU 上的邏輯測試（不碰 GPU：GPU 正在跑別的計時）。
把 torch.cuda 的 Stream／Event／synchronize／pinned 記憶體換成 CPU 替身，用一個很小的假模型，測：
  1. 還原器：meet none（逐塊、8 塊一組）、g5（沒紀錄 → 照 Cake；profiler；重算很慢 → 全載入；限速很慢 → Cake）、
     nodq（只搬不拆：低精度 block 的 KV 不動、t_dequant＝0）、io_prio；預設路徑沒有 meet_rule_used。
  2. m10_speed 的 worker（假模型）：--extra-modes loadall claim、M10_MEET=g5 的 profiler、meet_rule_used 欄、
     nodq 的 kv_bad＝SKIP_nodq、舊欄位的 restores.csv 不能混寫。
  3. Busy：HSA_CU_MASK 有傳給背景 process、DONE 那行有讀進 stats（背景用假的 python 片段，不用 GPU）。
用法：TRITON_INTERPRET=1 python restore2_cpu_shim.py
"""
import contextlib
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
import types

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "/mlsteam/workspace/paper-hierarchical-kv-state/code")
import torch  # noqa: E402

# ---------------------------------------------------------------- torch.cuda 的 CPU 替身
class _Stream:
    def __init__(self, device=None, priority=0):
        self.priority = priority

    @staticmethod
    def priority_range():
        return (0, -1)

    def synchronize(self):
        pass

    def wait_event(self, e):
        pass

    def wait_stream(self, s):
        pass

    def query(self):
        return True


class _Event:
    def __init__(self, enable_timing=False):
        self.t = None

    def record(self, stream=None):
        self.t = time.perf_counter()

    def elapsed_time(self, other):
        return (other.t - self.t) * 1e3

    def synchronize(self):
        pass

    def query(self):
        return True


torch.cuda.Stream = _Stream
torch.cuda.Event = _Event
torch.cuda.stream = lambda s: contextlib.nullcontext()
torch.cuda.synchronize = lambda *a, **k: None
torch.cuda.current_stream = lambda *a, **k: _Stream()
_empty = torch.empty


def _empty_nopin(*a, **k):
    k.pop("pin_memory", None)
    return _empty(*a, **k)


torch.empty = _empty_nopin
torch.Tensor.pin_memory = lambda self, *a, **k: self

import m10_codec as C  # noqa: E402
import m7_restore_harness as RH  # noqa: E402
from m7_model import CHUNK  # noqa: E402

DEV = "cpu"
fails = []


def chk(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        fails.append(msg)


class FakeKM:
    """KVModel 的最小替身（CPU）：forward_span 寫入由 (token, 位置, 層, head, channel) 決定的值；睡一下模擬計算。"""

    def __init__(self, max_len, L=2, H=2, D=128, sleep_per_tok=2e-6):
        self.L, self.H, self.D = L, H, D
        self.device = DEV
        self.max_len = max_len
        self.kv = torch.zeros((L, 2, H, max_len, D), dtype=torch.bfloat16)
        self.chunk_bytes = L * 2 * H * CHUNK * D * 2
        self.sleep = sleep_per_tok
        self.cfg = types.SimpleNamespace(vocab_size=32000)
        li = torch.arange(L).view(L, 1, 1, 1, 1)
        kv = torch.arange(2).view(1, 2, 1, 1, 1)
        h = torch.arange(H).view(1, 1, H, 1, 1)
        d = torch.arange(D).view(1, 1, 1, 1, D)
        self._base = (li * 11 + kv * 17 + h * 5 + d).float()

    def chunk_view(self, i):
        return self.kv[:, :, :, i * CHUNK:(i + 1) * CHUNK, :]

    def host_chunk(self):
        return torch.empty((self.L, 2, self.H, CHUNK, self.D), dtype=torch.bfloat16)

    def forward_span(self, ids, start, want_logits=False, n_logits=1):
        n = ids.shape[-1]
        pos = torch.arange(start, start + n).view(1, 1, 1, n, 1).float()
        t = ids.view(1, 1, 1, n, 1).float()
        x = torch.remainder(self._base + t * 7 + pos * 3, 97.0) / 16.0 - 3.0
        x = x + (x.abs() > 2.5).float() * 40.0
        self.kv[:, :, :, start:start + n] = x.to(torch.bfloat16)
        time.sleep(self.sleep * n)
        return torch.zeros(16) if want_logits else None

    def prefill_chunked(self, ids, start=0):
        for s in range(0, ids.shape[-1], CHUNK):
            self.forward_span(ids[s:s + CHUNK], start + s)


def build(km, ids, n, fmt):
    km.prefill_chunked(ids[:n * CHUNK])
    ref = km.kv[:, :, :, :n * CHUNK].clone()
    bf = [km.chunk_view(i).contiguous().clone() for i in range(n)]
    lp = [C.encode(km.chunk_view(i).contiguous(), fmt) for i in range(n)]
    return ref, bf, lp


def ref32(e):
    """FP32 參考值（同 dq_multi_cpu_interp.py）：Triton 直譯器轉 BF16 是截斷，一次拆多塊（Triton）的結果要和它的截斷比。"""
    p, (L_, _, H_, T_, D_) = e.parts, e.shape
    if e.fmt.startswith("fp8_"):
        return p["q"].view(C.FP8).float() * p["scale"]
    G = C.KIVI_GROUP
    k = C._unpack4(p["k_q"]).float().view(L_, H_, T_ // G, G, D_) * p["k_scale"].float() + p["k_zero"].float()
    v = C._unpack4(p["v_q"]).float().view(L_, H_, T_, D_ // G, G) * p["v_scale"].float() + p["v_zero"].float()
    return torch.stack((k.reshape(L_, H_, T_, D_), v.reshape(L_, H_, T_, D_)), dim=1)


def trunc(y):
    return (y.contiguous().view(torch.int32) & -65536).view(torch.float32).to(torch.bfloat16)


def check_kv(km, rec, ref, lp, nodq=False, interp=False):
    bad, seen = 0, {}
    for i, src, _, _ in rec["blocks"]:
        seen[i] = seen.get(i, 0) + 1
        v = km.kv[:, :, :, i * CHUNK:(i + 1) * CHUNK]
        if src in ("recompute", "load_bf16"):
            want = ref[:, :, :, i * CHUNK:(i + 1) * CHUNK]
        elif nodq:
            want = torch.zeros_like(v)                  # 只搬不拆：KV 緩衝的那一段沒被寫
        elif interp:
            want = trunc(ref32(lp[i]))                  # 一次拆多塊走 Triton（CPU 直譯 → 截斷）；GPU 上和原寫法逐位元相同（09 §6.4.2）
        else:
            want = C.decode(lp[i])
        bad += int(not torch.equal(v, want))
    return bad, all(c == 1 for c in seen.values()) and len(seen) == len(rec["blocks"])


def restorer_tests():
    n = 8
    ids = torch.randint(1000, 30000, (n * CHUNK + 256,))
    for fmt in ("int4_kivi", "fp8_token"):
        km = FakeKM(n * CHUNK + 512)
        ref, bf, lp = build(km, ids, n, fmt)
        cpu = RH.Tier("cpu_real", read_Bps=0, write_Bps=0)
        for sched, group in (("serial", 1), ("group", 4)):
            for mix in ("E1", "E0", "half"):
                host = lp if mix == "E1" else bf if mix == "E0" else [lp[i] if i % 2 else bf[i] for i in range(n)]
                rs = RH.Restorer(km)
                rs.set_options(sched=sched, group=group, meet="none")
                km.kv.zero_()
                rec = rs.restore(ids, n, host, [cpu] * n, "cake")
                bad, once = check_kv(km, rec, ref, lp, interp=sched == "group")
                chk(rec["n_recompute"] == 0 and rec.get("meet_rule_used") == "none" and bad == 0 and once,
                    f"{fmt} {sched}{group} {mix}：none 不重算、每塊一次、KV 逐位元（bad={bad}, n_rec={rec['n_recompute']}）")
        # none 但有沒存的 block → 照 Cake
        rs = RH.Restorer(km)
        rs.set_options(meet="none")
        km.kv.zero_()
        rec = rs.restore(ids, n, lp, [cpu] * (n - 1) + [None], "cake")
        bad, once = check_kv(km, rec, ref, lp)
        chk(rec["meet_rule_used"] == "claim(unstored)" and rec["n_recompute"] >= 1 and bad == 0,
            f"{fmt}：none 遇到沒存的 block → claim(unstored)、KV 對")
        # g5：沒紀錄 → Cake；profiler 後依估計
        km.sleep = 2e-4                                    # 重算一塊 ≈ 0.1 s，遠比載入慢 → 應選全載入
        rs = RH.Restorer(km)
        rs.set_options(meet="g5")
        rs.g5_key = ("cpu", "idle")
        km.kv.zero_()
        rec = rs.restore(ids, n, lp, [cpu] * n, "cake")
        chk(rec["meet_rule_used"] == "claim(no_history)" and check_kv(km, rec, ref, lp)[0] == 0,
            f"{fmt}：g5 沒紀錄 → claim(no_history)")
        rs.g5_force_claim = True
        for host in (bf, lp):
            km.kv.zero_()
            rec = rs.restore(ids, n, host, [cpu] * n, "cake")
            chk(rec["meet_rule_used"] == "claim(profile)", f"{fmt}：g5 profiler 照 Cake")
        rs.g5_force_claim = False
        h = rs._g5_hist[("cpu", "idle")]
        chk(len(h["r0"]) == 3 and "load_bf16" in h["l"] and lp[0].kind and f"load_{lp[0].kind}" in h["l"],
            f"{fmt}：g5 紀錄有第 0 塊重算時間與兩種來源的載入時間（r0={len(h['r0'])}, l={sorted(h['l'])}）")
        km.kv.zero_()
        rec = rs.restore(ids, n, lp, [cpu] * n, "cake")
        chk(rec["meet_rule_used"] == "none" and rec["n_recompute"] == 0 and check_kv(km, rec, ref, lp)[0] == 0,
            f"{fmt}：g5 重算很慢 → 全載入（{rec['meet_rule_used']}）")
        # 限速很慢的層 → Cake
        slow = RH.Tier("ssd_slow", read_Bps=km.chunk_bytes / 0.05, write_Bps=km.chunk_bytes / 0.05)   # 一塊 BF16 50 ms
        km.sleep = 2e-5                                    # 重算一塊 ≈ 10 ms
        rs2 = RH.Restorer(km)
        rs2.set_options(meet="g5")
        rs2.g5_key = ("slow", "idle")
        rs2.g5_force_claim = True
        for host in (bf, lp):
            slow.reset_clock()
            km.kv.zero_()
            rs2.restore(ids, n, host, [slow] * n, "cake")
        rs2.g5_force_claim = False
        slow.reset_clock()
        km.kv.zero_()
        rec = rs2.restore(ids, n, bf, [slow] * n, "cake")
        chk(rec["meet_rule_used"] == "claim" and rec["n_recompute"] >= 1 and check_kv(km, rec, ref, lp)[0] == 0,
            f"{fmt}：g5 限速很慢 → 照 Cake（{rec['meet_rule_used']}, b={rec['n_recompute']}）")
        chk(("cpu", "idle") not in rs2._g5_hist, f"{fmt}：g5 的紀錄依 key 分開")
        # g5b：重算很慢 → 全載入；限速很慢 → Cake；「全部載完」比第 0 塊慢、但比 Cake 實際（多開一塊）快 → 全載入（g5 會選 Cake）
        for rule in ("g5b",):
            km.sleep = 2e-4
            rs6 = RH.Restorer(km)
            rs6.set_options(meet=rule)
            rs6.g5_key = ("cpu", "idle")
            rs6.g5_force_claim = True
            for host in (bf, lp):
                km.kv.zero_()
                rs6.restore(ids, n, host, [cpu] * n, "cake")
            rs6.g5_force_claim = False
            km.kv.zero_()
            rec = rs6.restore(ids, n, lp, [cpu] * n, "cake")
            chk(rec["meet_rule_used"] == "none" and check_kv(km, rec, ref, lp)[0] == 0, f"{fmt}：{rule} 重算很慢 → 全載入")
            km.sleep = 2e-5
            rs7 = RH.Restorer(km)
            rs7.set_options(meet=rule)
            rs7.g5_key = ("slow", "idle")
            rs7._g5_hist[("slow", "idle")] = {"l": {"load_bf16": [0.05]}, "r0": [0.01]}
            slow.reset_clock()
            km.kv.zero_()
            rec = rs7.restore(ids, n, bf, [slow] * n, "cake")
            chk(rec["meet_rule_used"] == "claim", f"{fmt}：{rule} 限速很慢 → 照 Cake")
        for rule, want in (("g5", "claim"), ("g5b", "none")):    # 合成紀錄：n＝8 塊，每塊載入 3、重算 14 → 全載入 24 > 14（g5 選 Cake），但模擬的 Cake 會開第 2 塊 → 28（g5b 選全載入）
            rs8 = RH.Restorer(km)
            rs8.set_options(meet=rule)
            rs8.g5_key = ("x", "idle")
            rs8._g5_hist[("x", "idle")] = {"l": {"load_bf16": [3.0]}, "r0": [14.0]}
            chk(rs8._g5_decide(n, bf) is (want == "none"), f"{fmt}：{rule} 的估計（載入 3×8、重算 14）→ {want}")
        # nodq
        km.sleep = 2e-6
        rs3 = RH.Restorer(km)
        rs3.set_options(sched="nodq", meet="none")
        km.kv.zero_()
        rec = rs3.restore(ids, n, lp, [cpu] * n, "cake")
        bad, once = check_kv(km, rec, ref, lp, nodq=True)
        chk(rec["n_recompute"] == 0 and rec["t_dequant"] == 0.0 and bad == 0 and once
            and all(s != "recompute" for _, s, _, _ in rec["blocks"]),
            f"{fmt}：nodq 只搬不拆（KV 沒動、t_dequant＝0、bad={bad}）")
        # 預設路徑：沒有 meet_rule_used，KV 對
        rs4 = RH.Restorer(km)
        km.kv.zero_()
        rec = rs4.restore(ids, n, lp, [cpu] * n, "cake")
        chk("meet_rule_used" not in rec and check_kv(km, rec, ref, lp)[0] == 0, f"{fmt}：預設 Cake 路徑不變")
        rs5 = RH.Restorer(km)
        rs5.set_options(io_prio=True)
        chk(rs5.io_prio and rs5.iostream.priority == -1 and rs5._h2d_s.priority == -1, f"{fmt}：io_prio 換成最高優先權的 stream")
        try:
            RH.Restorer(km).set_options(sched="defer", meet="g5")
            chk(False, "g5＋defer 應該被擋")
        except AssertionError:
            chk(True, "g5＋defer 被擋（載入時間不含拆）")


def worker_tests():
    import m7_model
    m7_model.KVModel = lambda max_len: FakeKM(max_len)
    import m10_speed as S
    out = tempfile.mkdtemp(prefix="restore2_shim_")
    base = dict(model="qwen", fmt="int4_kivi", new_tokens=64, lengths=[4096], main_length=4096,
                configs=["E0", "E1", "R25"], refs=[],
                tiers=[{"name": "cpu", "kind": "cpu"}, {"name": "b32", "kind": "throttle", "GiBps": 0.5}],
                gpu_states=[{"name": "idle", "duty": 0.0}], reps=2, verify=True, launch=300)
    # (a) g5＋extra modes
    os.environ["M10_MEET"] = "g5"
    d1 = os.path.join(out, "g5")
    S.worker(dict(base, out_dir=d1, blocks_out=os.path.join(d1, "blocks.csv"), extra_modes=["loadall", "claim"],
                  extra_tiers=["cpu"]))
    rows = list(csv.DictReader(open(os.path.join(d1, "restores.csv"))))
    modes = {(r["bw_name"], r["mode"]) for r in rows}
    chk("meet_rule_used" in rows[0], "worker：restores.csv 有 meet_rule_used")
    chk(modes == {("cpu", "cake"), ("cpu", "loadall"), ("cpu", "claim"), ("b32", "cake")},
        f"worker：extra modes 只在 cpu（{sorted(modes)}）")
    chk(len(rows) == 2 * (3 * 3 + 3), f"worker：列數 {len(rows)} == 24")
    chk(all(r["kv_bad"] in ("", "0") for r in rows), "worker：KV 逐位元（每格第一次）")
    chk(all(r["meet_rule_used"] == "none" for r in rows if r["mode"] == "loadall"), "worker：loadall 列＝none")
    chk(all(r["meet_rule_used"] == "claim" for r in rows if r["mode"] == "claim"), "worker：claim 列＝claim")
    g5 = [r["meet_rule_used"] for r in rows if r["mode"] == "cake"]
    chk(all(x in ("none", "claim") for x in g5), f"worker：g5 列都有決定（profiler 之後不會 no_history）：{sorted(set(g5))}")
    chk(all(r["n_recompute"] == "0" for r in rows if r["mode"] == "loadall"), "worker：loadall 不重算")
    # (b) 舊欄位的檔不能混寫
    os.environ["M10_MEET"] = "claim"
    try:
        S.worker(dict(base, out_dir=d1, blocks_out=os.path.join(d1, "blocks.csv")))
        chk(False, "worker：欄位不同應該拒絕")
    except SystemExit as ex:
        chk("欄位" in str(ex), "worker：欄位不同 → 拒絕混寫")
    # (c) nodq
    os.environ["M10_MEET"] = "none"
    os.environ["M10_DQ_SCHED"] = "nodq"
    d2 = os.path.join(out, "nodq")
    S.worker(dict(base, out_dir=d2, blocks_out=os.path.join(d2, "blocks.csv"), configs=["E0", "E1"]))
    rows = list(csv.DictReader(open(os.path.join(d2, "restores.csv"))))
    e1 = [r for r in rows if r["config"] == "E1"]
    chk(all(r["kv_bad"] in ("", "SKIP_nodq") for r in rows) and any(r["kv_bad"] == "SKIP_nodq" for r in e1),
        "worker：nodq 的 kv_bad＝SKIP_nodq")
    chk(all(float(r["t_dequant"]) == 0 and r["n_recompute"] == "0" for r in e1), "worker：nodq 的 E1 不拆、不重算")
    for k in ("M10_MEET", "M10_DQ_SCHED"):
        os.environ.pop(k)
    # (d) 預設：沒有 meet_rule_used 欄
    d3 = os.path.join(out, "default")
    S.worker(dict(base, out_dir=d3, blocks_out=os.path.join(d3, "blocks.csv"), configs=["E0", "E1"]))
    hdr = next(csv.reader(open(os.path.join(d3, "restores.csv"))))
    chk(hdr == S.RESTORE_FIELDS, "worker：預設欄位不變")
    print("輸出在", out)


def busy_tests():
    import m10_speed as S
    fake = ("import os,signal,sys,time\n"
            "print('READY matmul_ms=1.0 mask='+os.environ.get('HSA_CU_MASK',''),flush=True)\n"
            "stop=[0]\nsignal.signal(signal.SIGTERM,lambda *a: stop.__setitem__(0,1))\n"
            "n=0\nwhile not stop[0]:\n    time.sleep(0.01); n+=1\n"
            "print(f'DONE n={n} s=0.5',flush=True)\n")
    real = subprocess.Popen

    def popen(cmd, **k):
        if len(cmd) > 1 and cmd[1].endswith("m7_busy.py"):
            cmd = [sys.executable, "-c", fake]
        return real(cmd, **k)

    S.subprocess.Popen = popen
    os.environ["M10_BUSY_CU_MASK"] = "0:0-265"
    with S.Busy(0.7) as b:
        time.sleep(0.2)
    os.environ.pop("M10_BUSY_CU_MASK")
    st = b.stats
    chk(st["cu_mask"] == "0:0-265" and "mask=0:0-265" in st["ready"], f"Busy：HSA_CU_MASK 傳給背景（{st['ready']}）")
    chk(st.get("n_matmul", 0) > 0 and st.get("matmul_per_s") == round(st["n_matmul"] / 0.5, 3), f"Busy：DONE 讀進 stats（{st}）")
    with S.Busy(0.7) as b2:
        pass
    chk(b2.stats["cu_mask"] == "", "Busy：沒設遮罩時不傳")
    S.subprocess.Popen = real


restorer_tests()
worker_tests()
busy_tests()
print("ALL PASS" if not fails else f"{len(fails)} FAIL")
sys.exit(1 if fails else 0)
