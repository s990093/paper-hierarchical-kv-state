import json, sys, math
def pct(a, qs):
    a=sorted(a); n=len(a); out=[]
    for q in qs:
        x=(n-1)*q/100; lo=int(x); hi=min(lo+1,n-1); out.append(round(a[lo]+(a[hi]-a[lo])*(x-lo),1))
    return out
rows=[json.loads(l) for l in open(sys.argv[1])]
n=len(rows)
ok=sum(1 for r in rows if math.ceil(r['input_length']/512)==len(r['hash_ids']))
print('rows',n,'unit_ok(512)',ok/n)
rows.sort(key=lambda r:r['timestamp'])
T=rows[-1]['timestamp']/1000
# index by full-block prefix tuple of each request: key = tuple(hash_ids[:full])
full=[r['input_length']//512 for r in rows]
# child relation: later request whose hash_ids[:full_i] == hash_ids_i[:full_i] AND has more blocks than full_i
from collections import defaultdict
# map prefix tuple -> list of request idx having that exact full-prefix (as candidate parents)
cand=defaultdict(list)
for i,r in enumerate(rows):
    if full[i]>=1: cand[tuple(r['hash_ids'][:full[i]])].append(i)
# for each request j, find the longest proper prefix of its hash_ids that equals some earlier request's full prefix
first_child_gap=[None]*n; nparents=[]
for j,r in enumerate(rows):
    h=r['hash_ids']; best=None
    for L in range(len(h)-1,0,-1):   # proper prefix: child must extend
        ps=[i for i in cand.get(tuple(h[:L]),[]) if i<j]
        if ps:
            best=(L,ps); break
    if best:
        L,ps=best; nparents.append(len(ps))
        i=ps[-1]  # most recent candidate parent
        g=(r['timestamp']-rows[i]['timestamp'])/1000
        if first_child_gap[i] is None or g<first_child_gap[i]: first_child_gap[i]=g
has=[g is not None for g in first_child_gap]
print('frac requests with a later extending request', sum(has)/n)
gaps=[g for g in first_child_gap if g is not None]
print('gap s pctl 10/50/90/99', pct(gaps,[10,50,90,99]))
print('children with >1 candidate parent (same longest prefix)', sum(1 for k in nparents if k>1), 'of', len(nparents))
ts=[r['timestamp']/1000 for r in rows]
fh=[h for h,t in zip(has,ts) if t<T/2]; lt=[h for h,t in zip(has,ts) if t>T-600]
print('T s',T,'has-child first half',sum(fh)/len(fh),'last 600s',sum(lt)/len(lt))
roots=sum(1 for j in range(n) if True)
