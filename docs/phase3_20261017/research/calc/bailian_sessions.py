import json, sys, math

def pct(a, qs):
    a=sorted(a); n=len(a); out=[]
    for q in qs:
        x=(n-1)*q/100; lo=int(x); hi=min(lo+1,n-1); out.append(round(a[lo]+(a[hi]-a[lo])*(x-lo),1))
    return out
def mean(a): a=list(a); return sum(a)/len(a) if a else float('nan')
path = sys.argv[1]
rows = [json.loads(l) for l in open(path)]
n = len(rows)
# unit assertion (CLAUDE.md rule 6): ceil(input/16)==len(hash_ids)
ok = sum(1 for r in rows if math.ceil(r['input_length']/16)==len(r['hash_ids']))
print('rows', n, 'unit_ok', ok, ok/n)
by = {r['chat_id']: r for r in rows}
assert len(by)==n
children = {}
for r in rows:
    p = r['parent_chat_id']
    if p != -1:
        children.setdefault(p, []).append(r['chat_id'])
roots = [r for r in rows if r['parent_chat_id']==-1]
missing_parent = sum(1 for r in rows if r['parent_chat_id']!=-1 and r['parent_chat_id'] not in by)
print('roots', len(roots), 'non-root', n-len(roots), 'parent_missing_in_file', missing_parent)
# turn consistency: child turn == parent turn + 1
tc = [(by[r['parent_chat_id']]['turn']+1==r['turn']) for r in rows if r['parent_chat_id'] in by]
print('turn==parent+1', sum(tc), len(tc))
multi_child = sum(1 for p,c in children.items() if len(c)>1)
print('parents with >1 child', multi_child, 'of', len(children))
ts=[r['timestamp'] for r in rows]; T=max(ts)
print('time span s', min(ts), T)
has_child=[r['chat_id'] in children for r in rows]
print('frac requests that get a next turn within trace', mean(has_child))
gaps=[by[c]['timestamp']-by[p]['timestamp'] for p,cs in children.items() if p in by for c in cs]
print('gap(child-parent ts) n', len(gaps), 'pctl 10/50/90/99', pct(gaps,[10,50,90,99]), 'min', min(gaps))
# censoring: among requests in first half vs last 10 min
print('has_child first half', mean(h for h,t in zip(has_child,ts) if t<T/2), 'last 600s', mean(h for h,t in zip(has_child,ts) if t>T-600))
types = {}
for r in rows: types[r['type']] = types.get(r['type'],0)+1
print('types', types)
inp=[r['input_length'] for r in rows]
print('input pctl 50/90/99/max', pct(inp,[50,90,99]), max(inp), '>=8K frac', mean(x>=8192 for x in inp), '>=32K', mean(x>=32768 for x in inp))
# prefix-chain check: child's hash_ids start with parent's hash_ids?
pref = 0; tot=0
for p,cs in children.items():
    if p not in by: continue
    ph = by[p]['hash_ids']
    for c in cs:
        ch = by[c]['hash_ids']; tot+=1
        k = len(ph)-1  # last block of parent may change (FAQ)
        if ch[:k]==ph[:k]: pref+=1
print('child shares parent prefix (all but last block)', pref, tot)
