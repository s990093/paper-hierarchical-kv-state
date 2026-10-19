"""test_m7_correctness.py — harness 正確性（phase1/05 §3 的三步）

1. 平台對照：HF 整段 prefill 跑兩次，最後 token 的 logits 是否位元相同。
2. 自管前向 vs HF：chunk 化 prefill 的最後 logits、argmax 與 HF 整段比較（報最大誤差）。
3. 重算 KV 的決定性：同樣切法重算兩次，KV 是否逐位元組相同；
   主機往返（GPU→pinned→GPU）是否逐位元組相同。
輸出 JSON 到 argv[1]。

24 GB 卡（3090，2026-10-17）：
  - 單卡：用完的 HF 輸出（整段 logits，8K 約 2 GiB）比對完就釋放；maxabs 逐層算（逐層取 max 再取 max，
    和整塊一次算的值相同），不再一次轉出兩份 fp32 KV。原本 L=8192 在第 60 行 OOM（run 20261017-174343）。
  - M7_TEST_REF_DEVICE=cuda:1（兩張卡）：受測的自管前向與 KV 留在 cuda:0（和實驗同一條路徑），
    HF 對照模型、重算前的 KV 副本與所有比對放在第二張卡。不是 tensor parallel：模型沒有切開。
    這個模式下 HF 只算最後一個位置的 logits（logits_to_keep=1；32K 整段 logits 約 8 GiB），JSON 會記下來。
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m7_model import CHUNK, KVModel, model_path  # noqa: E402


def _maxabs(x, y):
    """逐層算 max|x−y|（fp32）；兩者形狀 [L, ...]，可以在不同裝置（y 搬到 x 的裝置）。"""
    m = 0.0
    for li in range(x.shape[0]):
        m = max(m, float((x[li].float() - y[li].to(x.device).float()).abs().max()))
    return m


def main(out):
    # 預設 8192；M7_TEST_L 可改（第 1 輪加的，沒有用到）
    L = int(os.environ.get("M7_TEST_L", 8192))
    km = KVModel(max_len=L + 512)
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1000, min(120000, km.cfg.vocab_size), (L,), generator=g).to(km.device)
    rdev = os.environ.get("M7_TEST_REF_DEVICE")
    if rdev:
        from transformers import AutoModelForCausalLM
        ref_dev = torch.device(rdev)
        assert ref_dev != km.device, f"M7_TEST_REF_DEVICE={rdev} 和受測模型同一張卡"
        hf_ref = AutoModelForCausalLM.from_pretrained(
            model_path(), dtype=torch.bfloat16, attn_implementation="sdpa").to(ref_dev).eval()
        hf_kw = {"logits_to_keep": 1}
        # 只取 logits 的兩次呼叫不建 HF 的 KV cache（32K 約 4 GiB；run 20261017-180212-…-llama-32k-2gpu 在這裡 OOM）
        logit_kw = dict(hf_kw, use_cache=False)
    else:
        ref_dev, hf_ref, hf_kw, logit_kw = km.device, km.hf, {}, {}
    ids_ref = ids.to(ref_dev)
    res = {"L": L, "dut_device": str(km.device), "ref_device": str(ref_dev),
           "hf_logits_to_keep": hf_kw.get("logits_to_keep"), "hf_logit_calls_use_cache": logit_kw.get("use_cache")}

    with torch.inference_mode():
        a = hf_ref(ids_ref.view(1, -1), **logit_kw).logits[0, -1].float()
        b = hf_ref(ids_ref.view(1, -1), **logit_kw).logits[0, -1].float()
    res["hf_twice_bitwise_equal"] = bool(torch.equal(a, b))
    res["hf_twice_maxabs"] = float((a - b).abs().max())

    # 自管 chunk 化前向：除了最後一個 chunk，其餘 prefill，最後一個 chunk 取 logits
    km.prefill_chunked(ids[:L - CHUNK])
    c = km.forward_span(ids[L - CHUNK:], L - CHUNK, want_logits=True).float().to(ref_dev)
    res["own_vs_hf_maxabs_logit"] = float((c - a).abs().max())
    res["own_vs_hf_argmax_equal"] = bool(int(c.argmax()) == int(a.argmax()))
    res["own_vs_hf_top5_overlap"] = len(set(c.topk(5).indices.tolist()) & set(a.topk(5).indices.tolist()))

    # HF 的 KV 對照：用 HF 的 past_key_values 取第 0 層和最後一層的 K
    with torch.inference_mode():
        out_hf = hf_ref(ids_ref.view(1, -1), use_cache=True, **hf_kw)
    pkv = out_hf.past_key_values
    errs = []
    for li in (0, km.L // 2, km.L - 1):
        try:
            khf = pkv.layers[li].keys[0]
        except AttributeError:
            khf = pkv[li][0][0]
        kown = km.kv[li, 0, :, :L].to(ref_dev)
        errs.append(float((khf.float() - kown.float()).abs().max()))
    res["own_vs_hf_K_maxabs_layers_0_mid_last"] = errs
    del out_hf, pkv, khf, kown          # 整段 logits 與 HF 的 KV 用完就放掉（24 GB 卡）
    torch.cuda.empty_cache()

    ref = km.kv[:, :, :, :L].to(ref_dev, copy=True)
    # 重算決定性：同樣切法再跑一次
    km.kv.zero_()
    km.prefill_chunked(ids)
    if ref_dev == km.device:
        res["recompute_same_chunking_bitwise_equal"] = bool(torch.equal(ref, km.kv[:, :, :, :L]))
    else:                               # 逐層搬到對照卡比（搬運是逐位元組的複製）
        res["recompute_same_chunking_bitwise_equal"] = all(
            torch.equal(ref[li], km.kv[li, :, :, :L].to(ref_dev)) for li in range(km.L))
    res["recompute_same_chunking_maxabs"] = _maxabs(ref, km.kv[:, :, :, :L])

    # 主機往返
    h = km.host_chunk()
    t0 = time.perf_counter()
    h.copy_(km.chunk_view(3))
    torch.cuda.synchronize()
    km.chunk_view(3).zero_()
    km.chunk_view(3).copy_(h, non_blocking=True)
    torch.cuda.synchronize()
    res["host_roundtrip_bitwise_equal"] = bool(torch.equal(ref[:, :, :, 3 * CHUNK:4 * CHUNK],
                                                           km.chunk_view(3).to(ref_dev)))
    res["chunk_bytes"] = km.chunk_bytes
    res["torch"] = torch.__version__
    res["hip"] = torch.version.hip
    print(json.dumps(res, indent=1))
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
