"""12 §3：m10_tier2（兩層混合還原）的 worker＋summary，用假模型在 GPU 上跑一遍（只佔幾十 MB，品質段可以跑）。
真模型的冒煙在 stage 44 開頭（m10_tier2.py smoke）。

檢查（每一次還原）：
  1. KV 逐位元正確（kv_bad＝0：重算／BF16＝參考、INT4＝原寫法 decode）
  2. 不在慢層的配置（E0、E1）一個 block 都沒從慢層載；全部在慢層的（S-*）一個都沒從 CPU 載
  3. 從慢層載的 block 數 ≤ 慢層的 block 數；INT4 讀到的 ≤ INT4 存的；兩層合計＋重算＝n
  4. 「舊」：從慢層載到的 block 都在前 k 個；「新」：都在最後 k 個（逐 block 檔核對）
  5. 慢層越慢越久：S-bf16 在 nfs 的中位數 > ssd > cpu 的 E0
然後跑 summary（只檢查跑得完、比較有產生）。用法：python tier2_fake_gpu.py [--out /tmp/x]
"""
import argparse
import csv
import os
import statistics
import sys
import tempfile
import types

sys.path.insert(0, "/mlsteam/workspace/paper-hierarchical-kv-state/code")
sys.path.insert(0, "/mlsteam/workspace/paper-hierarchical-kv-state/code/m10_queue_mi300x/tests")
import torch                                    # noqa: E402
import m7_model                                 # noqa: E402
from restore_variants_fake_gpu import FakeKM    # noqa: E402


class FakeKVModel(FakeKM):
    def __init__(self, max_len):
        super().__init__(n_blocks=max_len // 512 - 1)
        self.cfg = types.SimpleNamespace(vocab_size=32000)

    def forward_span(self, ids, start, want_logits=False, n_logits=1):
        super().forward_span(ids, start)
        return torch.zeros(16, device=self.device) if want_logits else None

    def host_chunk(self):
        return torch.empty((self.L, 2, self.H, 512, self.D), dtype=torch.bfloat16, pin_memory=True)

    def prefill_chunked(self, ids, start=0):
        for s in range(0, ids.shape[-1], 512):
            self.forward_span(ids[s:s + 512], start + s)


m7_model.KVModel = FakeKVModel
import m10_tier2 as T                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out")
a = ap.parse_args()
out = a.out or tempfile.mkdtemp(prefix="tier2_fake_")
os.makedirs(out, exist_ok=True)
n = 16
# 假模型的區塊很小（4 層 × 2 head）：慢層頻寬照比例縮小，讓「慢層」真的比重算慢（否則全部被載入線吃掉）
tiers = {k: v / 64 for k, v in T.slow_tiers().items()}
spec = dict(model="qwen", length=n * 512, new_tokens=256, tiers=tiers, tier_names=["cpu", "ssd", "nfs", "w1", "w3", "w12", "w20"],
            gpu_states=[{"name": "idle", "duty": 0.0}], reps=2, out_dir=out, blocks_out=os.path.join(out, "blocks.csv"),
            verify=True, verify_all=True, launch=0)
T.worker(spec)

rows = list(csv.DictReader(open(os.path.join(out, "restores.csv"))))
blocks = list(csv.DictReader(open(os.path.join(out, "blocks.csv"))))
fails = []


def chk(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


want = sum(len(T.configs_for(t)) for t in spec["tier_names"]) * spec["reps"]
chk(len(rows) == want, f"還原次數 {len(rows)} == {want}")
chk(all(r["kv_bad"] == "0" for r in rows), f"KV 全部逐位元正確（kv_bad>0 的 {sum(r['kv_bad'] != '0' for r in rows)} 次）")
for r in rows:
    nr, nc, ns = int(r["n_recompute"]), int(r["n_load_cpu"]), int(r["n_load_ssd"])
    tag = f"{r['bw_name']} {r['config']} rep{r['rep']}"
    if r["config"] in ("E0", "E1"):
        chk(ns == 0, f"{tag}：沒從慢層載（{ns}）")
    if r["config"].startswith("S-"):
        chk(nc == 0, f"{tag}：沒從 CPU 載（{nc}）")
    chk(ns <= int(r["n_slow"]), f"{tag}：慢層載 {ns} ≤ 慢層有 {r['n_slow']}")
    chk(int(r["fp8_read"]) <= int(r["fp8_stored"]), f"{tag}：INT4 讀 {r['fp8_read']} ≤ 存 {r['fp8_stored']}")
    chk(nr + nc + ns == n, f"{tag}：重算 {nr}＋CPU {nc}＋慢層 {ns} == {n}")
k = {f: n * f // 100 for f in T.FRACS}
for b in blocks:
    if b["tier"] != "slow" or b["source"] == "recompute":
        continue
    cfg, i = b["config"], int(b["block"])
    if cfg.startswith("Told"):
        f = int(cfg[4:6])
        if not i < k[f]:
            fails.append(f"{cfg} 從慢層載到 block {i}（應 < {k[f]}）")
    if cfg.startswith("Tnew"):
        f = int(cfg[4:6])
        if not i >= n - k[f]:
            fails.append(f"{cfg} 從慢層載到 block {i}（應 ≥ {n - k[f]}）")
chk(not any("從慢層載到 block" in f for f in fails), "舊／新：從慢層載到的 block 都在該在的範圍")


def med(t, c):
    return statistics.median(float(r["t_restore"]) for r in rows if r["bw_name"] == t and r["config"] == c)


chk(med("nfs", "S-bf16") > med("ssd", "S-bf16") > med("cpu", "E0"),
    f"S-bf16：nfs {med('nfs', 'S-bf16'):.3f} > ssd {med('ssd', 'S-bf16'):.3f} > CPU E0 {med('cpu', 'E0'):.3f}")

noise = os.path.join(out, "noise.json")
open(noise, "w").write('{"delta_star": 0.05}')
# 假模型和其他 GPU 工作（例如 41）同時跑時，每列都會被 gpu_guard 標污染；summary 照規則不用這種列，所以這裡開測試用的開關
T.cmd_summary(argparse.Namespace(restores=os.path.join(out, "restores.csv"), noise=noise, out_dir=out, only_run=None,
                                 fig=True, include_contaminated=True))
chk(os.path.exists(os.path.join(out, "compare_qwen.json")), "summary 寫出 compare_qwen.json")
chk(os.path.exists(os.path.join(out, "figs", "B_qwen_tier2.png")), "summary 畫出圖 B")
print(f"輸出在 {out}")
print("ALL PASS" if not fails else f"{len(fails)} FAIL")
sys.exit(1 if fails else 0)
