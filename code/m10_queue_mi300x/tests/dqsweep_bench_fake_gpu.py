"""09 §6.4：m10_speed.Bench（還原單位 U、切段的主機副本、kv_bad）＋ m10_dqsweep smoke 的流程，用假模型在 GPU 上跑一遍
（只佔幾十 MB，品質段可以跑）。真模型的版本在 stage 39（m10_dqsweep.py smoke）。"""
import argparse
import sys
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
import m10_dqsweep as Q                         # noqa: E402

try:
    Q.cmd_smoke(argparse.Namespace(model="qwen", fmts=["int4_kivi", "fp8_head", "fp8_token"], sw="/nonexistent"))
except SystemExit as e:
    sys.exit(e.code)
