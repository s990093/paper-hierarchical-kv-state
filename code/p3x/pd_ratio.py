"""prefill vs decode 的 token 比例（每個來源）：新算的 prefill token（input − cached，若有）、全部 input、output。"""
import glob, sys, json, datetime
import pyarrow.parquet as pq, numpy as np
root = sys.argv[1]; out = sys.argv[2]
rows = []
for src_variant in ["freeinference/week*", "bailian/traceA", "bailian/coder", "bailian/thinking", "wildchat/1m", "burstgpt/v3", "mooncake/conv-rule-strict", "cachewise/catraces"]:
    files = sorted(glob.glob(f"{root}/{src_variant}/turns.parquet"))
    if not files:
        continue
    cols = ["input_tokens", "output_tokens", "cached_tokens", "trigger"]
    tabs = []
    for f in files:
        have = [c for c in cols if c in pq.read_schema(f).names]
        tabs.append(pq.read_table(f, columns=have).to_pandas())
    import pandas as pd
    d = pd.concat(tabs, ignore_index=True)
    inp = d["input_tokens"].astype("float64"); outp = d["output_tokens"].astype("float64")
    cached = d["cached_tokens"].astype("float64").fillna(0) if "cached_tokens" in d else pd.Series(0.0, index=d.index)
    new_prefill = (inp - cached).clip(lower=0)
    r = dict(source=src_variant.split("/")[0], variant=src_variant.split("/")[1], n_rows=int(len(d)),
             input_med=float(np.nanmedian(inp)), output_med=float(np.nanmedian(outp)),
             sum_input=float(np.nansum(inp)), sum_output=float(np.nansum(outp)), sum_new_prefill=float(np.nansum(new_prefill)),
             cached_known=bool("cached_tokens" in d and d["cached_tokens"].notna().any()))
    r["output_over_input"] = r["sum_output"] / r["sum_input"] if r["sum_input"] else float("nan")
    r["output_over_new_prefill"] = r["sum_output"] / r["sum_new_prefill"] if r["sum_new_prefill"] else float("nan")
    rows.append(r)
json.dump(dict(ts=datetime.datetime.now().astimezone().isoformat(), rows=rows), open(out, "w"), indent=1)
for r in rows:
    print(f"{r['source']:13s} {r['variant']:18s} n={r['n_rows']:>9,d} in_med={r['input_med']:>9.0f} out_med={r['output_med']:>6.0f} "
          f"out/in={r['output_over_input']:.4f} out/new_prefill={r['output_over_new_prefill']:.4f} cached_known={r['cached_known']}")
