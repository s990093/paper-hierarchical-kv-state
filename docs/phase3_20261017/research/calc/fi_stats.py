import sys, pandas as pd, numpy as np
d = sys.argv[1]
r = pd.read_parquet(f'{d}/requests.parquet'); s = pd.read_parquet(f'{d}/sessions.parquet'); t = pd.read_parquet(f'{d}/tool_calls.parquet')
print('sessions', len(s), 'users', s.user_id.nunique(), 'subagent sessions', s.parent_session_id.notna().sum(), 'harness', s.harness.value_counts().to_dict())
print('requests', len(r), 'step_trigger', r.step_trigger.value_counts().to_dict())
print('waited_for', r.waited_for.value_counts(dropna=False).head(8).to_dict())
# unit check: len(block_ids) == floor(tiktoken input / 16)
tik = r.role_tokens.apply(lambda x: sum(v for v in x.values() if v is not None))
nb = r.block_ids.apply(len)
ok = (nb == tik // 16)
print('unit check len(block_ids)==floor(sum(role_tokens)/16):', ok.mean(), ' (ceil variant:', (nb == -(-tik // 16)).mean(), ')')
print('prompt_tokens(provider) pctl 50/90/99', np.percentile(r.prompt_tokens, [50,90,99]).round(0), '>=8K', (r.prompt_tokens>=8192).mean(), '>=32K', (r.prompt_tokens>=32768).mean(), '>=128K', (r.prompt_tokens>=131072).mean())
r = r.sort_values(['session_id','step'])
r['prev_end'] = r.groupby('session_id').end_ms.shift(1)
r['gap_s'] = (r.start_ms - r.prev_end)/1000
u = r[(r.step_trigger=='user') & r.prev_end.notna()]
tl = r[(r.step_trigger=='tool') & r.prev_end.notna()]
print('user-initiated (non-first) n', len(u), 'gap s pctl 10/50/90/99', np.percentile(u.gap_s.dropna(), [10,50,90,99]).round(1), 'neg', (u.gap_s<0).mean())
print('tool-triggered n', len(tl), 'gap s pctl 10/50/90/99', np.percentile(tl.gap_s.dropna(), [10,50,90,99]).round(1), 'neg', (tl.gap_s<0).mean())
print('tool latency_ms pctl 50/90/99', np.percentile(t.latency_ms.dropna(), [50,90,99]).round(0), 'nan frac', t.latency_ms.isna().mean())
first = r.groupby('session_id').step.min()
print('turns(user requests) per session pctl 50/90', np.percentile(r[r.step_trigger=='user'].groupby('session_id').size(), [50,90]))
