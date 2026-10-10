"""m9_f1_run.py — F1：開 CPU 卸載（OffloadingConnector）後 TTFT 變慢，重現＋找原因。

問題：D2（code/m8_vllm_drop.py）在 vLLM 0.28／MI300X 看到開卸載後 prefill 變慢約 2 倍。
  這支腳本重用 D2 的工作負載與 monkeypatch（import m8_vllm_drop），多做幾件事：
  (1) 一個行程跑多格（doc:1 doc:16 chat:1），seed 可選；
  (2) 記錄實際選到的 attention backend（vLLM 開 KV connector 時在 ROCm 上會換 backend）；
  (3) 每一步的牆鐘時間與排了多少 token（只記有排 token 的步）；
  (4) 每個請求輸出 token 的雜湊（拿來比對修法有沒有弄壞 KV）；
  (5) 選用：torch profiler 量一個 33K prefill 的 kernel 時間分布。

cfg：
  off            不卸載（vLLM 自動選 backend；ROCm 上是 ROCM_ATTN）
  off_triton     不卸載，但強制 TRITON_ATTN（隔離「backend 換掉」這一個因素）
  cpu50          CPU 卸載 50%（同 D2；vLLM 自動選 → TRITON_ATTN）
  cpu50_nostore  同 cpu50，但 store_threshold 設很大＝從不寫 CPU（隔離「搬資料」本身）
  cpu50_rocmfix  同 cpu50，但強制 ROCM_ATTN，並把每層 (2, num_blocks, ...) 的 KV 拆成 K、V 兩個
                 blocks-first 的 view 註冊給 connector（修法示範；只在本腳本 monkeypatch，不改 venv）
  cpu50_rocmnaive 強制 ROCM_ATTN 但不拆（上游擋掉的原因；負對照，預期 KV 搬錯）
  cpu50_rocmfix_nostore 同 cpu50_rocmfix，但永遠不存（看修法剩下的 excess 是不是搬資料造成的）

用法：python code/m9_f1_run.py --cfg cpu50 --seed 1 --cells doc:1 doc:16 chat:1 [--profile]
輸出（run 目錄）：f1_requests.csv、f1_cells.csv、f1_steps.csv、f1_outputs.csv、f1_meta.json、
  （--profile）f1_profile.csv、profile_*.json.gz
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import gzip
import hashlib
import json
import os
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m8_vllm_drop as m8  # noqa: E402  （import 時會 chdir 到 run 目錄、設 VLLM_ENABLE_V1_MULTIPROCESSING=0）

RUN_ID, RD, now_iso = m8.RUN_ID, m8.RD, m8.now_iso
BACKEND_SEEN: list = []
STEP = dict(tok=0, nreq=0)
CELL_KEY: list = [None]
OUTS: dict = {}


def install_extra_patches(cfg):
    # 記錄 ROCm 平台實際回傳的 attention backend
    from vllm.platforms import rocm as prm
    _g = prm.RocmPlatform.get_attn_backend_cls.__func__

    def g(cls, *a, **k):
        r = _g(cls, *a, **k)
        BACKEND_SEEN.append(str(r))
        return r
    prm.RocmPlatform.get_attn_backend_cls = classmethod(g)

    # 每一步排了多少 token
    from vllm.v1.core.sched import scheduler as vsch
    _sch = vsch.Scheduler.schedule

    def sch(self, *a, **k):
        out = _sch(self, *a, **k)
        STEP["tok"] += out.total_num_scheduled_tokens
        STEP["nreq"] = max(STEP["nreq"], len(out.num_scheduled_tokens))
        return out
    vsch.Scheduler.schedule = sch

    if cfg in ("cpu50_rocmfix", "cpu50_rocmnaive", "cpu50_rocmfix_nostore"):
        from vllm.v1.attention.backends import rocm_attn
        rocm_attn.RocmAttentionBackend.supports_kv_connector = classmethod(lambda cls: True)
    if cfg in ("cpu50_rocmfix", "cpu50_rocmfix_nostore"):
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading import worker as owk
        from vllm.v1.kv_offload.base import CanonicalKVCacheRef, CanonicalKVCaches, CanonicalKVCacheTensor
        _reg = owk.OffloadingConnectorWorker.register_kv_caches

        def reg(self, kv_caches):
            nb = self.kv_cache_config.num_blocks
            first = next(iter(kv_caches.values()))
            if not (isinstance(first, __import__("torch").Tensor) and first.dim() == 5
                    and first.shape[0] == 2 and first.shape[1] == nb):
                print("ROCMFIX: layout not KV-outer, fallback", tuple(first.shape), flush=True)
                return _reg(self, kv_caches)
            groups = self.kv_cache_config.kv_cache_groups
            assert len(groups) == 1, "rocmfix 只處理單一 KV group（Llama）"
            tensors, refs, seen = [], [], {}
            for ln in groups[0].layer_names:
                kv = kv_caches[ln]
                key = kv.data_ptr()
                if key in seen:   # 共用 KV 的層：指到同一組 tensor
                    refs += [CanonicalKVCacheRef(tensor_idx=i, page_size_bytes=tensors[i].page_size_bytes)
                             for i in seen[key]]
                    continue
                idx = []
                for half in (kv[0], kv[1]):        # K、V 各自是 (num_blocks, block, heads, dim)，連續
                    assert half.is_contiguous()
                    t = half.view(-1).view(__import__("torch").int8).view(nb, -1)
                    tensors.append(CanonicalKVCacheTensor(tensor=t, page_size_bytes=t.shape[1]))
                    idx.append(len(tensors) - 1)
                    refs.append(CanonicalKVCacheRef(tensor_idx=len(tensors) - 1, page_size_bytes=t.shape[1]))
                seen[key] = idx
            print(f"ROCMFIX: registered {len(tensors)} K/V tensors, page {tensors[0].page_size_bytes} B", flush=True)
            self._init_worker(CanonicalKVCaches(tensors=tensors, group_data_refs=[refs]))
        owk.OffloadingConnectorWorker.register_kv_caches = reg


COAL = collections.Counter()


def install_coalesce():
    """把位址相連的 copy 合併成一個（src、dst 都相連才合併）。只包住 vLLM 選出來的 swap 函式，不改 venv。
    需要 CPU 端一個 chunk 含多個 GPU block（kv_connector_extra_config 的 block_size > 16），CPU 位址才會相連。"""
    import numpy as np
    import torch
    from vllm.v1.kv_offload.cpu import gpu_worker as gw
    _sel = gw._select_swap_blocks_fn

    def sel(layer_refs_per_group, gpu_to_cpu):
        fn = _sel(layer_refs_per_group, gpu_to_cpu)

        def wrapped(src, dst, sizes, **kw):
            n = int(src.numel())
            if n > 1:
                s0, d0, z0 = src.numpy(), dst.numpy(), sizes.numpy()
                o = np.argsort(s0, kind="stable")
                s2, d2, z2 = s0[o], d0[o], z0[o]
                brk = np.ones(n, dtype=bool)
                brk[1:] = ~((s2[:-1] + z2[:-1] == s2[1:]) & (d2[:-1] + z2[:-1] == d2[1:]))
                st = np.flatnonzero(brk)
                src = torch.from_numpy(np.ascontiguousarray(s2[st]))
                dst = torch.from_numpy(np.ascontiguousarray(d2[st]))
                sizes = torch.from_numpy(np.ascontiguousarray(np.add.reduceat(z2, st)))
                COAL["d2h_in" if gpu_to_cpu else "h2d_in"] += n
                COAL["d2h_out" if gpu_to_cpu else "h2d_out"] += int(st.size)
            return fn(src, dst, sizes, **kw)
        return wrapped
    gw._select_swap_blocks_fn = sel


def wrap_step(eng, step_w, stats):
    _step = eng.step

    def step():
        STEP["tok"] = 0
        STEP["nreq"] = 0
        t0 = time.perf_counter()
        outs = _step()
        dt = time.perf_counter() - t0
        ck = CELL_KEY[0]
        if ck is not None:
            if STEP["tok"] > 0:
                stats[ck]["busy_steps"] += 1
                stats[ck]["busy_s"] += dt
                stats[ck]["sched_tok"] += STEP["tok"]
                step_w.writerow([RUN_ID, now_iso(), *ck, round(dt, 6), STEP["tok"], STEP["nreq"]])
            else:
                stats[ck]["idle_steps"] += 1
                stats[ck]["idle_s"] += dt
            for o in outs:
                if o.finished and o.outputs:
                    OUTS[(ck, o.request_id)] = list(o.outputs[0].token_ids)
        return outs
    eng.step = step


def profile_prefill(llm, cfg, seed, n_tok=33024):
    """一個新的 33K prompt（不會命中任何 cache），profile 到第一個 token 出來。"""
    import torch
    from torch.profiler import ProfilerActivity, profile
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    eng = llm.llm_engine
    rng = random.Random(f"prof-{seed}")
    prompt = [rng.randrange(1000, 120000) for _ in range(n_tok)]
    llm.reset_prefix_cache(reset_connector=cfg.startswith("cpu"))
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:   # 只收 GPU 活動（CPU 追蹤會讓牆鐘慢 40 倍，smoke run 實測）
        t0 = time.perf_counter()
        eng.add_request("prof0", TokensPrompt(prompt_token_ids=prompt), SamplingParams(max_tokens=1, temperature=0.0))
        done = False
        while not done:
            for o in eng.step():
                if o.request_id == "prof0" and o.finished:
                    done = True
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    # 再多跑幾步讓 store 收尾（不在 profile 裡）
    for _ in range(20):
        eng.step()
    rows = []
    for e in prof.key_averages():
        if "CUDA" not in str(getattr(e, "device_type", "")):   # 只算 GPU 上的 kernel／memcpy，不算 CPU op（會重複計）
            continue
        dev_us = getattr(e, "self_device_time_total", None)
        if dev_us is None:
            dev_us = getattr(e, "self_cuda_time_total", 0)
        if dev_us and dev_us > 0:
            rows.append((e.key, int(e.count), float(dev_us)))
    rows.sort(key=lambda r: -r[2])

    def cat(name):
        n = name.lower()
        if "memcpy" in n or "memset" in n or "copybuffer" in n or "copy_" in n and "kernel" not in n:
            return "memcpy"
        if "attention" in n or "attn" in n or "paged" in n or "context_" in n or n.startswith("_fwd_kernel"):
            return "attention"
        if "gemm" in n or "cijk" in n or "matmul" in n or "hipblaslt" in n or "_mm" in n:
            return "gemm"
        return "other"
    tot = collections.Counter()
    for k, c, us in rows:
        tot[cat(k)] += us
    trace = os.path.join(RD, f"profile_{cfg}_s{seed}.json")
    try:
        prof.export_chrome_trace(trace)
        with open(trace, "rb") as fi, gzip.open(trace + ".gz", "wb") as fo:
            shutil.copyfileobj(fi, fo)
        os.remove(trace)
    except Exception as ex:  # noqa: BLE001
        print("PROFILE_EXPORT_FAIL", repr(ex), flush=True)
    return wall, rows, tot


def _kv_zero(worker):
    import torch
    for t in worker.model_runner.kv_caches:
        t.zero_()
    torch.cuda.synchronize()
    return len(worker.model_runner.kv_caches)


def _kv_fingerprint(worker):
    """每個 GPU block、每層一個指紋：該 block 的原始位元組（當 int16）和固定隨機權重做內積（float64）。
    回傳 (num_blocks, num_layers) 的 CPU tensor。位元組完全相同 → 指紋完全相同。"""
    import torch
    mr = worker.model_runner
    nb = mr.kv_cache_config.num_blocks
    kvs = mr.kv_caches
    out = torch.zeros((nb, len(kvs)), dtype=torch.float64)
    g = torch.Generator(device="cpu").manual_seed(12345)
    w = None
    for li, t in enumerate(kvs):
        d = [i for i, sz in enumerate(t.shape) if sz == nb]
        assert len(d) == 1, (tuple(t.shape), nb)
        d = d[0]
        for i in range(0, nb, 1024):
            n = min(1024, nb - i)
            x = t.narrow(d, i, n).movedim(d, 0).reshape(n, -1)
            x = x.view(torch.int16).to(torch.float64)
            if w is None or w.numel() != x.shape[1]:
                w = torch.rand(x.shape[1], generator=g, dtype=torch.float64).to(x.device)
            out[i:i + n, li] = (x @ w).cpu()
    return out


def kv_roundtrip_check(llm, cfg, seed, n_tok=8192):
    """存 → 把 GPU KV 全部清 0 → 同一個 prompt 再跑一次（從 CPU 載入）→ 比對每個 block 的指紋。"""
    import torch
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    eng = llm.llm_engine
    rng = random.Random(f"kvcheck-{seed}")
    prompt = [rng.randrange(1000, 120000) for _ in range(n_tok)]
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    ok0 = llm.reset_prefix_cache(reset_connector=True)
    llm.collective_rpc(_kv_zero)

    def run_and_drain(tag):
        outs = llm.generate([TokensPrompt(prompt_token_ids=prompt)], sp, use_tqdm=False)
        t0 = time.perf_counter()
        for _ in range(200):   # 讓 store 做完
            if not m8.MANAGERS or m8.MANAGERS[-1]._num_write_pending_blocks == 0:
                break
            eng.step()
        pend = m8.MANAGERS[-1]._num_write_pending_blocks if m8.MANAGERS else None
        print("KVCHECK", tag, "cached", outs[0].num_cached_tokens, "pending_after_drain", pend,
              "drain_s", round(time.perf_counter() - t0, 3), flush=True)
        return outs[0]
    o1 = run_and_drain("pass1")
    fp1 = llm.collective_rpc(_kv_fingerprint)[0]
    ok1 = llm.reset_prefix_cache(reset_connector=False)   # 只清 GPU 的 prefix cache，CPU 留著
    llm.collective_rpc(_kv_zero)                           # GPU KV 全部清 0：之後有值的 block 只能來自 CPU 載入或重算
    o2 = run_and_drain("pass2")
    fp2 = llm.collective_rpc(_kv_fingerprint)[0]
    nz1 = (fp1 != 0).any(dim=1)
    nz2 = (fp2 != 0).any(dim=1)
    rows1 = {tuple(r.tolist()) for r in fp1[nz1]}
    rows2 = [tuple(r.tolist()) for r in fp2[nz2]]
    match = sum(r in rows1 for r in rows2)
    res = dict(run_id=RUN_ID, ts=now_iso(), cfg=cfg, seed=seed, prompt_tokens=n_tok,
               reset_ok=f"{ok0};{ok1}", pass1_cached=o1.num_cached_tokens, pass2_cached=o2.num_cached_tokens,
               pass2_loaded_blocks=(o2.num_cached_tokens or 0) // 16,
               blocks_nonzero_pass1=int(nz1.sum()), blocks_nonzero_pass2=int(nz2.sum()),
               pass2_blocks_matching_pass1=match,
               pass1_tok=o1.outputs[0].token_ids[0], pass2_tok=o2.outputs[0].token_ids[0])
    print("KVCHECK_RESULT", json.dumps(res), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True,
                    choices=["off", "off_triton", "cpu50", "cpu50_nostore", "cpu50_rocmfix", "cpu50_rocmnaive",
                             "cpu50_rocmfix_nostore"])
    ap.add_argument("--cells", nargs="+", default=["doc:1", "doc:16", "chat:1"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-sess", type=int, default=16)
    ap.add_argument("--out-tokens", type=int, default=32)
    ap.add_argument("--gpu-kv-gib", type=float, default=16.0)
    ap.add_argument("--profile", action="store_true", help="主格之前先 profile 一個 33K prefill")
    ap.add_argument("--profile-only", action="store_true")
    ap.add_argument("--kvcheck", action="store_true", help="只做 KV 存→載入的逐 block 位元組比對")
    ap.add_argument("--offload-block-size", type=int, default=None, help="卸載單位（token）；預設＝GPU block")
    ap.add_argument("--coalesce", action="store_true", help="合併位址相連的 copy（見 install_coalesce）")
    a = ap.parse_args()

    m8.install_patches()
    install_extra_patches(a.cfg)
    if a.coalesce:
        install_coalesce()
    label = a.cfg + (f"_bs{a.offload_block_size}" if a.offload_block_size else "") + ("_coal" if a.coalesce else "")
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.inputs import TokensPrompt

    ws_bytes = a.n_sess * 33 * 1024 * m8.KV_BYTES_PER_TOKEN   # 同 D2
    kw = dict(model=sorted(glob.glob(m8.MODEL_GLOB))[-1], max_model_len=40960,
              kv_cache_memory_bytes=int(a.gpu_kv_gib * m8.GIB), max_num_seqs=16, max_num_batched_tokens=8192,
              enable_prefix_caching=True, disable_log_stats=False, seed=a.seed)
    engine_id = f"f1-{RUN_ID}-{a.cfg}"
    if a.cfg == "off_triton":
        kw["attention_backend"] = "TRITON_ATTN"
    if a.cfg in ("cpu50_rocmfix", "cpu50_rocmnaive", "cpu50_rocmfix_nostore"):
        kw["attention_backend"] = "ROCM_ATTN"
    cpu_bytes = 0
    if a.cfg.startswith("cpu50"):
        cpu_bytes = int(ws_bytes * 0.5)
        extra = {"spec_name": "CPUOffloadingSpec", "cpu_bytes_to_use": cpu_bytes, "eviction_policy": "lru"}
        if a.cfg.endswith("nostore"):
            extra["store_threshold"] = 10 ** 9   # 每個 block 要被 lookup 10^9 次才存＝從不存
        if a.offload_block_size:
            extra["block_size"] = a.offload_block_size
        kw["kv_transfer_config"] = KVTransferConfig(kv_connector="OffloadingConnector", kv_role="kv_both",
                                                    engine_id=engine_id, kv_connector_extra_config=extra)
    meta = dict(run_id=RUN_ID, cfg=label, seed=a.seed, cells=a.cells, n_sess=a.n_sess, ws_bytes=ws_bytes,
                cpu_bytes=cpu_bytes, engine_id=engine_id,
                llm_kwargs={k: (str(v) if k == "kv_transfer_config" else v) for k, v in kw.items()})
    print("META", json.dumps(meta, default=str), flush=True)

    llm = LLM(**kw)
    meta["attn_backends_seen"] = sorted(set(BACKEND_SEEN))
    backend = ";".join(sorted({b.rsplit(".", 1)[-1] for b in BACKEND_SEEN}))
    print("BACKEND", backend, flush=True)
    json.dump(meta, open(os.path.join(RD, "f1_meta.json"), "w"), indent=1, default=str)

    def opener(name, cols):
        f = open(os.path.join(RD, name), "a", newline="")
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(cols)
        return f, w

    fr, rw_raw = opener("f1_requests.csv", m8.REQ_COLS + ["seed", "backend"])
    fsx, sw = opener("f1_steps.csv", ["run_id", "ts", "cfg", "workload", "conc", "dt_s", "sched_tokens", "n_reqs"])
    fo, ow = opener("f1_outputs.csv", ["run_id", "ts", "cfg", "seed", "workload", "conc", "req_id", "n_tok", "sha1", "tokens"])
    fc = open(os.path.join(RD, "f1_cells.csv"), "a", newline="")
    cw = None

    class RW:   # 在 m8 的 request 列後面補 seed、backend
        def writerow(self, r):
            rw_raw.writerow(list(r) + [a.seed, backend])

    stats = collections.defaultdict(lambda: collections.Counter())
    eng = llm.llm_engine
    try:
        llm.generate([TokensPrompt(prompt_token_ids=[random.Random(1).randrange(1000, 120000) for _ in range(512)])],
                     SamplingParams(max_tokens=1), use_tqdm=False)   # 暖機（不計）
        # 暖機 2：一個 8K prompt，讓 Triton 長序列的 kernel 先 JIT（不計）
        llm.generate([TokensPrompt(prompt_token_ids=[random.Random(2).randrange(1000, 120000) for _ in range(8192)])],
                     SamplingParams(max_tokens=2), use_tqdm=False)
        if a.profile or a.profile_only:
            for rep in range(2):   # 第 1 次可能有 JIT，取第 2 次
                wall, rows, tot = profile_prefill(llm, a.cfg, a.seed * 10 + rep)
                print("PROFILE", rep, round(wall, 4), dict(tot), flush=True)
            fp, pw = opener("f1_profile.csv", ["run_id", "ts", "cfg", "seed", "backend", "wall_s", "rank", "kernel",
                                                "count", "self_device_us", "category_totals_json"])
            for i, (k, c, us) in enumerate(rows[:60]):
                pw.writerow([RUN_ID, now_iso(), a.cfg, a.seed, backend, round(wall, 5), i, k[:200], c, round(us, 1),
                             json.dumps({kk: round(v, 1) for kk, v in tot.items()}) if i == 0 else ""])
            fp.close()
        if a.kvcheck:
            res = kv_roundtrip_check(llm, label, a.seed)
            res.update({f"coal_{k}": v for k, v in sorted(COAL.items())})
            fk, kw_ = opener("f1_kvcheck.csv", list(res.keys()))
            kw_.writerow(list(res.values()))
            fk.close()
        if not (a.profile_only or a.kvcheck):
            wrap_step(eng, sw, stats)
            for cell in a.cells:
                wl, conc = cell.split(":")
                conc = int(conc)
                llm.reset_prefix_cache(reset_connector=a.cfg.startswith("cpu"))
                CELL_KEY[0] = (label, wl, conc)
                COAL.clear()
                summ = m8.run_cell(llm, label, wl, conc, a.n_sess, a.out_tokens, a.seed, RW(), None)
                ck = CELL_KEY[0]
                CELL_KEY[0] = None
                st = stats[ck]
                summ.update(seed=a.seed, backend=backend, busy_steps=st["busy_steps"], busy_s=round(st["busy_s"], 4),
                            idle_steps=st["idle_steps"], idle_s=round(st["idle_s"], 4), sched_tok=st["sched_tok"],
                            coal_d2h_in=COAL["d2h_in"], coal_d2h_out=COAL["d2h_out"],
                            coal_h2d_in=COAL["h2d_in"], coal_h2d_out=COAL["h2d_out"])
                for (k2, rid), toks in sorted(OUTS.items()):
                    if k2 == ck:
                        ow.writerow([RUN_ID, now_iso(), label, a.seed, wl, conc, rid, len(toks),
                                     hashlib.sha1(json.dumps(toks).encode()).hexdigest()[:16],
                                     " ".join(map(str, toks))])
                if cw is None:
                    cw = csv.DictWriter(fc, fieldnames=list(summ.keys()))
                    if fc.tell() == 0:
                        cw.writeheader()
                cw.writerow(summ)
                for f in (fr, fc, fsx, fo):
                    f.flush()
                print("CELL", json.dumps({k: summ[k] for k in ("cfg", "workload", "conc", "seed", "backend",
                                                               "ttft_first_med", "ttft_ret_med", "wall_s",
                                                               "busy_s", "idle_s", "ret_cpu_hit", "ret_recomputed")}),
                      flush=True)
    finally:
        for f in (fr, fc, fsx, fo):
            f.close()
        try:
            del llm
        except Exception:  # noqa: BLE001
            pass
        for p in glob.glob(f"/dev/shm/vllm_offload_{engine_id}*"):   # 只刪自己的 mmap
            os.remove(p)
            print("removed", p, flush=True)


if __name__ == "__main__":
    main()
