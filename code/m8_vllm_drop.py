"""m8_vllm_drop.py — D2：真實 vLLM 0.28 的 CPU 卸載，「CPU 滿了→寫不進去／卡住」多常發生

問題：多個長 session 同時跑、會回來、沒有空檔時，OffloadingConnector 的 CPU 層
  (1) 有多少要存的 KV 因為 CPU 滿又沒有可淘汰的 block 而被跳過（prepare_store 回 None）；
  (2) 回來的請求有多少 token 從 CPU 載入、多少重算、重算的原因；
  (3) 卸載有沒有讓請求等（worker flush 等待、scheduler 延後、首輪 TTFT 對照 off）。

做法：同一行程裡跑引擎（VLLM_ENABLE_V1_MULTIPROCESSING=0），在自己這支腳本裡 monkeypatch
  CPUOffloadingManager / OffloadingConnectorScheduler / OffloadingConnectorWorker / Scheduler，
  不改 venv 的任何檔案。判準見 docs/research_20261010_directions/D2_vllm_drop.md §2。

輸出（run 目錄）：d2_requests.csv（每個請求一列）、d2_cells.csv（每格一列）、d2_steps.csv（每步 CPU 狀態取樣）。
用法：python code/m8_vllm_drop.py --cfg cpu50 --workload doc --conc 1 4 16
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone

RUNS = os.environ.get("TIARA_RUNS", "/mlsteam/data/tiara/runs")
RUN_ID = os.environ.get("RUN_ID", "adhoc-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
RD = os.path.join(RUNS, RUN_ID)
os.makedirs(RD, exist_ok=True)
os.chdir(RD)  # ROCm 當掉時 gpucore.* 會掉在 cwd：放在 run 目錄
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # 引擎在同一行程，patch 才有效

MODEL_GLOB = "/mlsteam/data/tiara/hf-cache/hub/models--unsloth--Llama-3.1-8B-Instruct/snapshots/*"
GIB = 2 ** 30
KV_BYTES_PER_TOKEN = 32 * 8 * 128 * 2 * 2  # Llama-3.1-8B bf16：128 KiB


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------------------- 計數器
class Cell:
    """一格（cfg × workload × conc）的所有計數。"""

    def __init__(self):
        self.offered: set = set()          # 進 prepare_store 時不在 CPU 的 key
        self.accepted: set = set()         # prepare_store 給了 CPU 空間的 key
        self.failed_once: set = set()      # 曾經在回 None 的呼叫裡的 key
        self.stored_ok: set = set()        # complete_store(success=True)
        self.store_failed: set = set()     # complete_store(success=False)
        self.evicted: set = set()          # 被 LRU 淘汰
        self.calls = 0                     # 有東西要存的 prepare_store 次數
        self.fail_calls = 0                # 回 None 的次數（＝vLLM ALLOCATION_FAILURE）
        self.fail_call_keys = 0            # 失敗呼叫裡的 key 數（含重試重複）
        self.evicted_n = 0
        self.loaded_keys = 0               # prepare_load 的 key 數
        self.flush_waits = 0
        self.flush_wait_s = 0.0
        self.handle_preempt_s = 0.0        # handle_preemptions 全部時間（含提交 store）
        self.start_xfer_s = 0.0            # start_kv_transfers 時間
        self.preemptions = 0
        self.deferrals = 0
        self.max_nonevict_frac = 0.0       # (寫入中＋載入中) / CPU 總 block
        self.max_write_pending_frac = 0.0
        self.max_used_frac = 0.0
        self.steps = 0


CUR: Cell | None = None
REQ = collections.defaultdict(dict)       # 外部 req id -> {local, ext, defer_n, defer_t0, defer_s, miss_reason}
MANAGERS: list = []
SCHED_CFG: dict = {}


def ext_id(request) -> str:
    e = getattr(request, "external_req_id", None)
    return e if e else request.request_id.split("-")[0]


def install_patches():
    from vllm.v1.kv_offload.cpu import manager as cm
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading import scheduler as osch
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading import worker as owk
    from vllm.v1.core.sched import scheduler as vsch

    M = cm.CPUOffloadingManager
    _init, _ps, _cs, _lk, _pl = M.__init__, M.prepare_store, M.complete_store, M.lookup, M.prepare_load

    def init(self, *a, **k):
        _init(self, *a, **k)
        MANAGERS.append(self)
    M.__init__ = init

    def prepare_store(self, keys, req_context):
        keys = list(keys)
        new = [k for k in keys if self._policy.get(k) is None]
        out = _ps(self, keys, req_context)
        c = CUR
        if c is not None and new:
            c.calls += 1
            c.offered.update(new)
            if out is None:
                c.fail_calls += 1
                c.fail_call_keys += len(new)
                c.failed_once.update(new)
            else:
                c.accepted.update(out.keys_to_store)
                c.evicted.update(out.evicted_keys)
                c.evicted_n += len(out.evicted_keys)
        return out
    M.prepare_store = prepare_store

    def complete_store(self, keys, req_context, success=True):
        keys = list(keys)
        c = CUR
        if c is not None:
            (c.stored_ok if success else c.store_failed).update(keys)
        return _cs(self, keys, req_context, success)
    M.complete_store = complete_store

    def lookup(self, key, req_context):
        r = _lk(self, key, req_context)
        c = CUR
        if c is not None and r.name == "MISS":
            rid = getattr(req_context, "req_id", None)
            # 最長前綴查詢遇到第一個 MISS 就停，所以這裡記的是「CPU 命中停在哪裡、為什麼」
            reason = ("evicted" if key in c.evicted
                      else "skipped" if (key in c.offered and key not in c.accepted)
                      else "never_offered")
            if rid is not None:
                d = REQ[str(rid).split("-")[0]]
                d.setdefault("miss_reason", reason)
        return r
    M.lookup = lookup

    def prepare_load(self, keys, req_context):
        keys = list(keys)
        if CUR is not None:
            CUR.loaded_keys += len(keys)
        return _pl(self, keys, req_context)
    M.prepare_load = prepare_load

    S = osch.OffloadingConnectorScheduler
    _gnm, _usa = S.get_num_new_matched_tokens, S.update_state_after_alloc

    def gnm(self, request, num_computed_tokens):
        if not SCHED_CFG:
            try:
                g = self.config.kv_group_configs[0]
                SCHED_CFG.update(tokens_per_chunk=g.tokens_per_chunk, blocks_per_chunk=self.config.blocks_per_chunk)
            except Exception as e:  # noqa: BLE001
                SCHED_CFG.update(err=repr(e))
        n, asy = _gnm(self, request, num_computed_tokens)
        d = REQ[ext_id(request)]
        d["local"] = num_computed_tokens
        if n is None:
            d["defer_n"] = d.get("defer_n", 0) + 1
            d.setdefault("defer_t0", time.perf_counter())
            if CUR is not None:
                CUR.deferrals += 1
        elif "defer_t0" in d and "defer_s" not in d:
            d["defer_s"] = time.perf_counter() - d["defer_t0"]
        return n, asy
    S.get_num_new_matched_tokens = gnm

    def usa(self, request, blocks, num_external_tokens):
        d = REQ[ext_id(request)]
        d.setdefault("ext", num_external_tokens)  # 第一次排程時的值（搶占後重排不覆蓋）
        d.setdefault("local_at_alloc", d.get("local", 0))
        return _usa(self, request, blocks, num_external_tokens)
    S.update_state_after_alloc = usa

    W = owk.OffloadingConnectorWorker
    _hp, _skt = W.handle_preemptions, W.start_kv_transfers

    def hp(self, meta):
        t0 = time.perf_counter()
        r = _hp(self, meta)
        dt = time.perf_counter() - t0
        if CUR is not None:
            CUR.handle_preempt_s += dt
            if meta.jobs_to_flush:
                CUR.flush_waits += 1
                CUR.flush_wait_s += dt
        return r
    W.handle_preemptions = hp

    def skt(self, meta):
        t0 = time.perf_counter()
        r = _skt(self, meta)
        if CUR is not None:
            CUR.start_xfer_s += time.perf_counter() - t0
        return r
    W.start_kv_transfers = skt

    VS = vsch.Scheduler
    _pr = VS._preempt_request

    def pr(self, *a, **k):
        if CUR is not None:
            CUR.preemptions += 1
        return _pr(self, *a, **k)
    VS._preempt_request = pr


def sample_cpu_state(writer, cell_key):
    c = CUR
    if c is None or not MANAGERS:
        return
    m = MANAGERS[-1]
    nb = m._num_blocks
    if nb <= 0:
        return
    used = m._num_allocated_blocks - len(m._free_list)          # 有資料（含寫入中、載入中）
    nonevict = used - m._num_evictable_cache_blocks              # 寫入中＋載入中（ref_cnt != 0）
    wp = m._num_write_pending_blocks
    c.max_nonevict_frac = max(c.max_nonevict_frac, nonevict / nb)
    c.max_write_pending_frac = max(c.max_write_pending_frac, wp / nb)
    c.max_used_frac = max(c.max_used_frac, used / nb)
    if writer is not None and c.steps % 5 == 0:
        writer.writerow([RUN_ID, now_iso(), *cell_key, c.steps, round(time.perf_counter(), 4), nb, used, nonevict, wp])


# ----------------------------------------------------------------------------- 工作負載
def make_sessions(workload, n_sess, rng):
    """每個 session：每一輪要新加的 token（list of lists）。輸出 token 由引擎產生後接上。"""
    tok = lambda n: [rng.randrange(1000, 120000) for _ in range(n)]  # noqa: E731
    out = []
    for _ in range(n_sess):
        if workload == "doc":
            turns = [tok(32768) + tok(256)] + [tok(256) for _ in range(3)]
        elif workload == "chat":
            turns = [tok(8192) for _ in range(4)]
        else:
            raise ValueError(workload)
        out.append(turns)
    return out


def run_cell(llm, cfg, workload, conc, n_sess, out_tokens, seed, req_w, step_w):
    global CUR
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    eng = llm.llm_engine
    rng = random.Random(f"{seed}-{workload}-{conc}")   # 同一格在不同 cfg 用同一批 token
    sessions = make_sessions(workload, n_sess, rng)
    ctx = [[] for _ in range(n_sess)]      # 到目前為止的完整 context（prompt＋輸出）
    turn = [0] * n_sess
    ready = collections.deque(range(n_sess))
    active = {}
    sp = SamplingParams(max_tokens=out_tokens, temperature=0.0, ignore_eos=True)
    cell_key = (cfg, workload, conc)
    CUR = Cell()
    REQ.clear()
    rows = []
    t_cell0 = time.perf_counter()
    while ready or active:
        while len(active) < conc and ready:
            s = ready.popleft()
            t = turn[s]
            prompt = ctx[s] + sessions[s][t]
            rid = f"c{conc}s{s:02d}t{t}"
            active[rid] = dict(s=s, t=t, prompt=prompt, t_add=time.perf_counter(), t_first=None)
            eng.add_request(rid, TokensPrompt(prompt_token_ids=prompt), sp)
        outs = eng.step()
        CUR.steps += 1
        sample_cpu_state(step_w, cell_key)
        tnow = time.perf_counter()
        for o in outs:
            a = active.get(o.request_id)
            if a is None:
                continue
            if a["t_first"] is None and o.outputs and len(o.outputs[0].token_ids) > 0:
                a["t_first"] = tnow
            if o.finished:
                s, t = a["s"], a["t"]
                gen = list(o.outputs[0].token_ids)
                ctx[s] = a["prompt"] + gen
                d = REQ.get(o.request_id, {})
                plen = len(a["prompt"])
                cached = o.num_cached_tokens or 0
                ext = d.get("ext", 0) or 0
                local = cached - ext if cfg != "off" else cached
                mt = o.metrics
                q_s = (mt.scheduled_ts - mt.queued_ts) if (mt and mt.scheduled_ts and mt.queued_ts) else ""
                pf_s = (mt.first_token_ts - mt.scheduled_ts) if (mt and mt.first_token_ts and mt.scheduled_ts) else ""
                rows.append([RUN_ID, now_iso(), cfg, workload, conc, s, t, plen, plen - len(sessions[s][t]),
                             cached, local, ext, plen - cached,
                             round(a["t_first"] - a["t_add"], 5) if a["t_first"] else "",
                             round(tnow - a["t_add"], 5),
                             round(q_s, 5) if q_s != "" else "", round(pf_s, 5) if pf_s != "" else "",
                             d.get("defer_n", 0), round(d.get("defer_s", 0.0), 5), d.get("miss_reason", "")])
                del active[o.request_id]
                turn[s] += 1
                if turn[s] < len(sessions[s]):
                    ready.insert(rng.randrange(len(ready) + 1), s)   # 修訂 1：插到隨機位置
    wall = time.perf_counter() - t_cell0
    # 收尾：讓還在飛的 store 做完（最多 50 步／30 s），沒做完的記下來
    t_d = time.perf_counter()
    pend_end = None
    for _ in range(50):
        if not MANAGERS or MANAGERS[-1]._num_write_pending_blocks == 0:
            break
        eng.step()
        sample_cpu_state(step_w, cell_key)
        if time.perf_counter() - t_d > 30:
            break
    if MANAGERS:
        pend_end = MANAGERS[-1]._num_write_pending_blocks
    for r in rows:
        req_w.writerow(r)
    c = CUR
    skipped = c.offered - c.accepted
    first = [r for r in rows if r[6] == 0]
    ret = [r for r in rows if r[6] > 0]
    med = lambda xs: round(statistics.median(xs), 5) if xs else ""  # noqa: E731
    ret_prev = sum(r[8] for r in ret)  # 回來請求的「上一輪就算過」的 token
    summary = dict(
        run_id=RUN_ID, ts=now_iso(), cfg=cfg, workload=workload, conc=conc, n_sess=n_sess,
        cpu_blocks=(MANAGERS[-1]._num_blocks if MANAGERS else 0),
        tokens_per_chunk=SCHED_CFG.get("tokens_per_chunk", ""),
        offered=len(c.offered), accepted=len(c.accepted), skipped=len(skipped),
        skip_ratio=(len(skipped) / len(c.offered)) if c.offered else "",
        alloc_fail_calls=c.fail_calls, store_calls=c.calls, fail_call_keys=c.fail_call_keys,
        failed_then_accepted=len(c.failed_once & c.accepted),
        lru_evicted=c.evicted_n, store_failed=len(c.store_failed), stored_ok=len(c.stored_ok),
        write_pending_at_end=pend_end, loaded_keys=c.loaded_keys,
        max_nonevict_frac=round(c.max_nonevict_frac, 4), max_write_pending_frac=round(c.max_write_pending_frac, 4),
        max_used_frac=round(c.max_used_frac, 4),
        flush_waits=c.flush_waits, flush_wait_s=round(c.flush_wait_s, 4),
        handle_preempt_s=round(c.handle_preempt_s, 4), start_xfer_s=round(c.start_xfer_s, 4),
        preemptions=c.preemptions, deferrals=c.deferrals, steps=c.steps, wall_s=round(wall, 3),
        n_first=len(first), n_ret=len(ret),
        ttft_first_med=med([r[13] for r in first if r[13] != ""]),
        ttft_ret_med=med([r[13] for r in ret if r[13] != ""]),
        ttft_ret_p90=(round(sorted(r[13] for r in ret if r[13] != "")[int(0.9 * (len(ret) - 1))], 5) if ret else ""),
        ret_prev_tokens=ret_prev,
        ret_gpu_hit=sum(r[10] for r in ret), ret_cpu_hit=sum(r[11] for r in ret),
        ret_recomputed=sum(r[12] for r in ret),
        ret_miss_reason=json.dumps(collections.Counter(r[19] for r in ret if r[19])),
    )
    CUR = None
    return summary


REQ_COLS = ["run_id", "ts", "cfg", "workload", "conc", "sess", "turn", "prompt_len", "prev_ctx_len",
            "cached_tokens", "gpu_hit", "cpu_hit", "recomputed", "ttft_s", "e2e_s", "queue_s", "prefill_s",
            "defer_n", "defer_s", "first_miss_reason"]
STEP_COLS = ["run_id", "ts", "cfg", "workload", "conc", "step", "t", "cpu_blocks", "used", "nonevict", "write_pending"]


def guard_status(rd):
    """回傳 (ok, note)。已知誤報：amd-smi 在自己行程結束時回報 pid −1（任務說明）。
    只有在「所有外來者都是 pid −1，而且第一次出現在最後一格寫完之後」才當成誤報。"""
    gp = os.path.join(rd, "gpu_guard.json")
    if not os.path.exists(gp):
        return False, "no_gpu_guard_json"
    g = json.load(open(gp))
    if not g.get("contaminated"):
        return True, "clean"
    if not g.get("started_clean", False):
        return False, "dirty_at_start"
    cells = os.path.join(rd, "d2_cells.csv")
    if not os.path.exists(cells):
        return False, "contaminated_no_cells"
    last_ts = max(datetime.fromisoformat(r["ts"]) for r in csv.DictReader(open(cells)))
    intr = g.get("intruders", [])
    if intr and all(i.get("pid") == -1 for i in intr) and \
            min(datetime.fromisoformat(i["first_seen"]) for i in intr) >= last_ts:
        return True, "pid-1_after_last_cell(known_false_positive)"
    return False, "contaminated"


def collect(run_dirs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cells, reqs, status = [], [], []
    for rd in run_dirs:
        ok, note = guard_status(rd)
        status.append(dict(run_id=os.path.basename(rd.rstrip("/")), ts=now_iso(), guard_ok=ok, guard_note=note))
        if not ok:
            continue
        cells += list(csv.DictReader(open(os.path.join(rd, "d2_cells.csv"))))
        reqs += list(csv.DictReader(open(os.path.join(rd, "d2_requests.csv"))))
    for name, rows in (("d2_cells.csv", cells), ("d2_requests.csv", reqs), ("d2_guard.csv", status)):
        with open(os.path.join(out_dir, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    # 事先寫好的主指標：cpu50，C∈{4,16}，doc＋chat，合併 skipped/offered
    real = [c for c in cells if c["cfg"] == "cpu50" and int(c["conc"]) in (4, 16)]
    summ = []
    for label, sel in (("preregistered_cpu50_c4c16", real),
                       ("stress_cpu25_all", [c for c in cells if c["cfg"] == "cpu25"]),
                       ("cpu50_all", [c for c in cells if c["cfg"] == "cpu50"]),
                       ("explore_E1_cpu4g", [c for c in cells if c["cfg"] == "cpu4g"]),
                       ("explore_E2_tier50", [c for c in cells if c["cfg"] == "tier50"]),
                       ("explore_E3_cpu50_bs512", [c for c in cells if c["cfg"] == "cpu50_bs512"])):
        off = sum(int(c["offered"]) for c in sel)
        sk = sum(int(c["skipped"]) for c in sel)
        summ.append(dict(run_id=";".join(sorted({c["run_id"] for c in sel})), ts=now_iso(), group=label,
                         n_cells=len(sel), offered=off, skipped=sk,
                         skip_ratio=(sk / off) if off else "",
                         alloc_fail_calls=sum(int(c["alloc_fail_calls"]) for c in sel),
                         lru_evicted=sum(int(c["lru_evicted"]) for c in sel)))
    with open(os.path.join(out_dir, "d2_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    # 每格的 TTFT 與回來請求的命中組成（給文件 §4 用；比值是〔算術〕）
    by = collections.defaultdict(list)
    for r in reqs:
        by[(r["workload"], int(r["conc"]), r["cfg"], int(r["turn"]) > 0)].append(r)
    trows = []
    for (wl, conc, cfg, is_ret) in sorted(k for k in by if not k[3]):
        f, rr = by[(wl, conc, cfg, False)], by.get((wl, conc, cfg, True), [])
        base = by.get((wl, conc, "off", False))
        t1 = sorted(float(x["ttft_s"]) for x in f)
        tR = sorted(float(x["ttft_s"]) for x in rr)
        prev = sum(int(x["prev_ctx_len"]) for x in rr)
        g = sum(int(x["gpu_hit"]) for x in rr)
        cp = sum(int(x["cpu_hit"]) for x in rr)
        b1 = statistics.median(float(x["ttft_s"]) for x in base) if base else None
        trows.append(dict(
            run_id=f[0]["run_id"], ts=now_iso(), workload=wl, conc=conc, cfg=cfg, n_first=len(t1), n_ret=len(tR),
            first_ttft_med=round(statistics.median(t1), 4), first_ttft_min=round(t1[0], 4), first_ttft_max=round(t1[-1], 4),
            first_ttft_ratio_vs_off=(round(statistics.median(t1) / b1, 4) if b1 else ""),
            ret_ttft_med=(round(statistics.median(tR), 4) if tR else ""),
            ret_ttft_p90=(round(tR[int(0.9 * (len(tR) - 1))], 4) if tR else ""),
            ret_ttft_max=(round(tR[-1], 4) if tR else ""),
            ret_prev_tokens=prev, ret_gpu_frac=(round(g / prev, 4) if prev else ""),
            ret_cpu_frac=(round(cp / prev, 4) if prev else ""),
            ret_prev_recomputed_frac=(round((prev - g - cp) / prev, 4) if prev else ""),
            ret_defer_s_max=(round(max(float(x["defer_s"]) for x in rr), 4) if rr else "")))
    with open(os.path.join(out_dir, "d2_ttft.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trows[0].keys()))
        w.writeheader()
        w.writerows(trows)
    for t in trows:
        print("TTFT", t)
    for s in status:
        print("GUARD", s)
    for s in summ:
        print("SUMMARY", s)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "collect":
        # 用法：python code/m8_vllm_drop.py collect <out_dir> <run_dir>...
        collect(sys.argv[3:], sys.argv[2])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, choices=["off", "cpu25", "cpu50", "cpu100", "cpu4g", "tier50"])
    ap.add_argument("--workload", required=True, choices=["doc", "chat"])
    ap.add_argument("--conc", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--n-sess", type=int, default=16)
    ap.add_argument("--out-tokens", type=int, default=32)
    ap.add_argument("--gpu-kv-gib", type=float, default=16.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="小規模：檢查能不能跑")
    ap.add_argument("--block-size", type=int, default=None, help="修訂 3 的 E3：卸載單位（token）；預設＝GPU block")
    a = ap.parse_args()

    install_patches()
    from vllm import LLM
    from vllm.config import KVTransferConfig

    n_sess = a.n_sess
    ws_tokens = n_sess * 33 * 1024   # 工作集約略：每 session 約 32K＋問題／輸出〔算術〕
    ws_bytes = ws_tokens * KV_BYTES_PER_TOKEN
    frac = {"cpu25": 0.25, "cpu50": 0.5, "cpu100": 1.0, "cpu4g": 4 * GIB / ws_bytes, "tier50": 0.5}.get(a.cfg)
    kw = dict(model=sorted(glob.glob(MODEL_GLOB))[-1], max_model_len=40960,
              kv_cache_memory_bytes=int(a.gpu_kv_gib * GIB), max_num_seqs=16, max_num_batched_tokens=8192,
              enable_prefix_caching=True, disable_log_stats=False, seed=a.seed)
    engine_id = f"d2-{RUN_ID}-{a.cfg}" + (f"-bs{a.block_size}" if a.block_size else "")
    if frac is not None:
        cpu_bytes = int(ws_bytes * frac)
        extra = {"spec_name": "CPUOffloadingSpec", "cpu_bytes_to_use": cpu_bytes, "eviction_policy": "lru"}
        if a.cfg == "tier50":   # 修訂 2 的 E2：CPU 下面接 fs 層（放在 run 目錄，NFS）
            os.makedirs(os.path.join(RD, "fs"), exist_ok=True)
            extra = {"spec_name": "TieringOffloadingSpec", "cpu_bytes_to_use": cpu_bytes, "eviction_policy": "lru",
                     "secondary_tiers": [{"type": "fs", "root_dir": os.path.join(RD, "fs")}]}
        if a.block_size:
            extra["block_size"] = a.block_size
        kw["kv_transfer_config"] = KVTransferConfig(
            kv_connector="OffloadingConnector", kv_role="kv_both", engine_id=engine_id,
            kv_connector_extra_config=extra)
    else:
        cpu_bytes = 0
    meta = dict(run_id=RUN_ID, cfg=a.cfg, workload=a.workload, conc=a.conc, n_sess=n_sess, ws_bytes=ws_bytes,
                cpu_bytes=cpu_bytes, gpu_kv_bytes=int(a.gpu_kv_gib * GIB), engine_id=engine_id,
                llm_kwargs={k: (str(v) if k == "kv_transfer_config" else v) for k, v in kw.items()})
    json.dump(meta, open("d2_meta.json", "w"), indent=1, default=str)
    print("META", json.dumps(meta, default=str), flush=True)

    llm = LLM(**kw)
    print("SCHED_CFG", SCHED_CFG, "managers", len(MANAGERS), flush=True)

    fr = open("d2_requests.csv", "a", newline="")
    fc = open("d2_cells.csv", "a", newline="")
    fs = open("d2_steps.csv", "a", newline="")
    rw, sw = csv.writer(fr), csv.writer(fs)
    if fr.tell() == 0:
        rw.writerow(REQ_COLS)
    if fs.tell() == 0:
        sw.writerow(STEP_COLS)
    cw = None
    try:
        # 暖機（不計）：一個 512 token 請求
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        llm.generate([TokensPrompt(prompt_token_ids=[random.Random(1).randrange(1000, 120000) for _ in range(512)])],
                     SamplingParams(max_tokens=1), use_tqdm=False)
        for conc in a.conc:
            llm.reset_prefix_cache(reset_connector=(a.cfg != "off"))
            if MANAGERS:
                print("after reset: allocated", MANAGERS[-1]._num_allocated_blocks, flush=True)
            label = a.cfg + (f"_bs{a.block_size}" if a.block_size else "")   # E3 不和主格的 cpu50 混在一起
            summ = run_cell(llm, label, a.workload, conc, n_sess if not a.smoke else 4,
                            a.out_tokens, a.seed, rw, sw)
            if cw is None:
                cw = csv.DictWriter(fc, fieldnames=list(summ.keys()))
                if fc.tell() == 0:
                    cw.writeheader()
            cw.writerow(summ)
            fr.flush(); fc.flush(); fs.flush()
            print("CELL", json.dumps(summ), flush=True)
    finally:
        fr.close(); fc.close(); fs.close()
        try:
            del llm
        except Exception:  # noqa: BLE001
            pass
        fsd = os.path.join(RD, "fs")
        if a.cfg == "tier50" and os.path.isdir(fsd):   # fs 層的 block 檔（暫存，NFS 上）：記大小後刪掉
            nbytes = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs_ in os.walk(fsd) for f in fs_)
            print("FS_TIER_BYTES", nbytes, flush=True)
            import shutil
            shutil.rmtree(fsd, ignore_errors=True)
        left = glob.glob(f"/dev/shm/vllm_offload_{engine_id}*")
        for p in left:   # 只刪自己這個 engine_id 的 mmap（RUNLOG_MI300X 發現 5）
            os.remove(p)
            print("removed", p, flush=True)


if __name__ == "__main__":
    main()
