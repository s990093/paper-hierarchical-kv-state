import json, sys
from collections import defaultdict, Counter
def pct(a, qs):
    a=sorted(a); n=len(a); out=[]
    for q in qs:
        x=(n-1)*q/100; lo=int(x); hi=min(lo+1,n-1); out.append(round(a[lo]+(a[hi]-a[lo])*(x-lo),1))
    return out
rows=sorted([json.loads(l) for l in open(sys.argv[1])], key=lambda r:r['timestamp'])
n=len(rows); T=rows[-1]['timestamp']/1000
print('timestamp mod 1000 == 0 frac', sum(1 for r in rows if r['timestamp']%1000==0)/n)
# how many requests share each block-prefix (popularity of prefix of length L)
pop=Counter()
for r in rows:
    h=r['hash_ids']
    for L in range(1,len(h)+1): pop[tuple(h[:L])]+=1
full=[r['input_length']//512 for r in rows]
cand=defaultdict(list)
for i,r in enumerate(rows):
    if full[i]>=1: cand[tuple(r['hash_ids'][:full[i]])].append(i)
strict=0; gaps=[]; amb=0; childof=[0]*n
for j,r in enumerate(rows):
    h=r['hash_ids']
    for L in range(len(h)-1,0,-1):
        ps=[i for i in cand.get(tuple(h[:L]),[]) if i<j and rows[i]['timestamp']<r['timestamp']]
        if ps:
            # require the matched prefix to be 'private': the parent's full prefix is shared by <=3 requests overall
            if pop[tuple(h[:L])]<=3 and len(ps)==1:
                i=ps[0]; strict+=1; childof[i]+=1
                gaps.append((r['timestamp']-rows[i]['timestamp'])/1000)
            elif len(ps)>1: amb+=1
            break
print('strict child links (private prefix, unique parent, gap>0)', strict, 'ambiguous', amb)
print('frac requests with strict child', sum(1 for c in childof if c>0)/n)
print('strict gap s pctl 10/50/90/99', pct(gaps,[10,50,90,99]))
