"""09 §4.3：m10_confirm 的速度路徑（patch m10_speed 的 layout／Bench 後呼叫主線 worker）用假模型在 GPU 上跑一遍
（只佔幾十 MB，品質段可以跑；假模型沒有 attention，所以不測 ATT／AXE，真模型在 43 開頭的冒煙測）。
檢查：每個 SW- 配置存的低精度 block 數＝k_of(佔比)；KV 逐位元正確；meets.json 的鍵是「名稱@格式」；E1s、E0 照主線。"""
import argparse
import csv
import json
import os
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
os.environ["M10_CODEC_X"] = "1"
os.environ["RUN_ID"] = "fake-confirm-speed"
import m10_confirm as C                         # noqa: E402
import m10_speed as S                           # noqa: E402
import m10_sweep as SW                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out")
out = ap.parse_args().out or tempfile.mkdtemp(prefix="confirm_speed_fake_")
os.makedirs(out, exist_ok=True)
fmt = "int2h_g32"
cfgs = ["E0", "E1s", "SW-ERR-35", "SW-B-90", "SW-R0-35", "SW-M-55"]
spec = dict(model="qwen", fmt=fmt, new_tokens=256, lengths=[32768], main_length=32768, configs=cfgs, refs=[],
            tiers=[{"name": "cpu", "kind": "cpu"}, {"name": "b32", "kind": "throttle", "GiBps": 0.05}],
            gpu_states=[{"name": "idle", "duty": 0.0}], reps=2, out_dir=out, blocks_out=os.path.join(out, "blocks.csv"),
            verify=True, need_att=False, launch=0)
C._sworker(spec)
rows = list(csv.DictReader(open(os.path.join(out, "restores.csv"))))
fails = []


def chk(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


chk(len(rows) == len(cfgs) * 2 * 2, f"還原次數 {len(rows)} == {len(cfgs) * 4}")
chk(all(r["kv_bad"] in ("", "0") for r in rows), "KV 逐位元正確（每格第一次）")
for r in rows:
    c = r["config"]
    want = 0 if c == "E0" else 63 if c == "E1s" else SW.k_of(int(c.split("-")[2]) / 100)
    chk(int(r["fp8_stored"]) == want, f"{c}：存了 {r['fp8_stored']} 個低精度 block（應 {want}）")
rows2 = S._rows(os.path.join(out, "restores.csv"))
for r in rows2:                                 # 和 41 等工作同時跑時每列都被標污染，write_meets 照規則會全部丟掉；這裡只測鍵名
    r["contaminated_gpu"] = "False"
S.write_meets(argparse.Namespace(out_dir=out, model="qwen", main_length=32768, base_fmt="__none__"), rows2)
m = json.load(open(os.path.join(out, "meets.json")))["meets"]
chk(set(m) == {"E0"} | {f"{c}@{fmt}" for c in cfgs if c != "E0"}, f"meets.json 的鍵：{sorted(m)}")
print("輸出在", out)
print("ALL PASS" if not fails else f"{len(fails)} FAIL")
sys.exit(1 if fails else 0)
