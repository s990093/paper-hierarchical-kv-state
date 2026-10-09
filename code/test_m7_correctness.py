"""test_m7_correctness.py — harness 正確性（phase1/05 §3 的三步）

1. 平台對照：HF 整段 prefill 跑兩次，最後 token 的 logits 是否位元相同。
2. 自管前向 vs HF：chunk 化 prefill 的最後 logits、argmax 與 HF 整段比較（報最大誤差）。
3. 重算 KV 的決定性：同樣切法重算兩次，KV 是否逐位元組相同；
   主機往返（GPU→pinned→GPU）是否逐位元組相同。
輸出 JSON 到 argv[1]。
"""
import json
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m7_model import CHUNK, KVModel  # noqa: E402


def main(out):
    L = 8192
    km = KVModel(max_len=L + 512)
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1000, 120000, (L,), generator=g).to(km.device)
    res = {"L": L}

    with torch.inference_mode():
        a = km.hf(ids.view(1, -1)).logits[0, -1].float()
        b = km.hf(ids.view(1, -1)).logits[0, -1].float()
    res["hf_twice_bitwise_equal"] = bool(torch.equal(a, b))
    res["hf_twice_maxabs"] = float((a - b).abs().max())

    # 自管 chunk 化前向：除了最後一個 chunk，其餘 prefill，最後一個 chunk 取 logits
    km.prefill_chunked(ids[:L - CHUNK])
    c = km.forward_span(ids[L - CHUNK:], L - CHUNK, want_logits=True).float()
    res["own_vs_hf_maxabs_logit"] = float((c - a).abs().max())
    res["own_vs_hf_argmax_equal"] = bool(int(c.argmax()) == int(a.argmax()))
    res["own_vs_hf_top5_overlap"] = len(set(c.topk(5).indices.tolist()) & set(a.topk(5).indices.tolist()))

    # HF 的 KV 對照：用 HF 的 past_key_values 取第 0 層和最後一層的 K
    with torch.inference_mode():
        out_hf = km.hf(ids.view(1, -1), use_cache=True)
    pkv = out_hf.past_key_values
    errs = []
    for li in (0, km.L // 2, km.L - 1):
        try:
            khf = pkv.layers[li].keys[0]
        except AttributeError:
            khf = pkv[li][0][0]
        kown = km.kv[li, 0, :, :L]
        errs.append(float((khf.float() - kown.float()).abs().max()))
    res["own_vs_hf_K_maxabs_layers_0_mid_last"] = errs

    ref = km.kv[:, :, :, :L].clone()
    # 重算決定性：同樣切法再跑一次
    km.kv.zero_()
    km.prefill_chunked(ids)
    res["recompute_same_chunking_bitwise_equal"] = bool(torch.equal(ref, km.kv[:, :, :, :L]))
    res["recompute_same_chunking_maxabs"] = float((ref.float() - km.kv[:, :, :, :L].float()).abs().max())

    # 主機往返
    h = km.host_chunk()
    t0 = time.perf_counter()
    h.copy_(km.chunk_view(3))
    torch.cuda.synchronize()
    km.chunk_view(3).zero_()
    km.chunk_view(3).copy_(h, non_blocking=True)
    torch.cuda.synchronize()
    res["host_roundtrip_bitwise_equal"] = bool(torch.equal(ref[:, :, :, 3 * CHUNK:4 * CHUNK], km.chunk_view(3)))
    res["chunk_bytes"] = km.chunk_bytes
    res["torch"] = torch.__version__
    res["hip"] = torch.version.hip
    print(json.dumps(res, indent=1))
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
