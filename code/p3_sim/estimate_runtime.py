"""estimate_runtime.py — 完整地圖要跑多久〔估計〕（06 §16）。只讀計時 run 的每請求秒數（run 20261018-102610-p3sim-timing3 的 stdout），線性外插。
不跑任何模擬。假設：FreeInference 每週的每請求成本＝week13（每請求 385 個 block）；WildChat 整份＝7 天窗的每請求成本；
non-oracle 策略用 C1 的成本、oracle 用 O 的成本；CPU 32／512 GiB 量到的取平均（FreeInference 只量 32）；E6 分 k 張 GPU 時總工作量不變。
  python code/p3_sim/estimate_runtime.py
"""
import ast, json
rows=[ast.literal_eval(l.strip()) for l in open('/ssd8/hungwei/paper-hkv/runs/20261018-102610-p3sim-timing3/stdout.log') if l.strip().startswith('{')]
cost={}
for r in rows:
    k=(r['source']+'/'+r['variant'], r['policy'])
    cost.setdefault(k,[]).append((r['session_s_per_req'], r['block_s_per_req']))
c={k:(sum(x[0] for x in v)/len(v), sum(x[1] for x in v)/len(v)) for k,v in cost.items()}
for k,v in sorted(c.items()): print(k, '%.3g %.3g'%v)
N={'bailian/coder':43011,'bailian/traceA':43058,'mooncake/conv-rule-strict':12031,'wildchat/1m':1960074,'freeinference/week13':None}
FI_A=282392; FI_ALL=1222199; FI_B_extra=FI_ALL-FI_A
def est(nonor, orc, outer_sess, cpus, views=8):
    # 每份資料：session 引擎 outer_sess×cpus×(nonor C1 成本 + orc O 成本)；block 引擎 2×cpus×(nonor+orc)×views
    out={}
    for d,n in list(N.items()):
        sc1, bc1 = c[(d,'C1')]; so, bo = c[(d,'O')]
        per_req = outer_sess*cpus*(nonor*sc1 + orc*so) + 2*cpus*views*(nonor*bc1 + orc*bo)
        sess = outer_sess*cpus*(nonor*sc1 + orc*so); blk = 2*cpus*views*(nonor*bc1 + orc*bo)
        out[d]=(per_req, sess, blk)
    return out
for tag,(nonor,orc,outer,cpus) in {'v3axes+new':(10,8,7,5),'v4axes':(19,8,7,8)}.items():
    e=est(nonor,orc,outer,cpus)
    def hours(d,n): return e[d][0]*n/3600, e[d][1]*n/3600, e[d][2]*n/3600
    A={d:hours(d,n) for d,n in N.items() if n}
    A['FI 01+05+09+13']=hours('freeinference/week13',FI_A)
    B16=hours('freeinference/week13',FI_ALL); B12=hours('freeinference/week13',FI_B_extra)
    tA=sum(v[0] for v in A.values())
    print('==',tag, 'placements/dataset session', outer*cpus*(nonor+orc), 'block runs/dataset', 2*cpus*8*(nonor+orc))
    for d,v in A.items(): print('  A', d, 'CPUh total %.1f (session %.1f, block %.1f)'%v)
    print('  A total CPUh %.1f -> 8 procs %.1f h'%(tA, tA/8))
    print('  B 16 weeks CPUh %.1f (session %.1f block %.1f) -> 8 procs %.1f h'%(B16[0],B16[1],B16[2],B16[0]/8))
    print('  B extra 12 weeks CPUh %.1f -> 8 procs %.1f h'%(B12[0],B12[0]/8))
# v4 主格（§6）：session 引擎 E2、E3、E4、E5(30 分) × 3 個記憶體點 × 27；block 引擎 E6＋E2 × 3 × 27 × 1 個時間設定（主視角）
def est_main(nonor=19, orc=8, outer=4, cpus=3, views=1):
    tot={}
    for d,n in list(N.items()):
        sc1, bc1 = c[(d,'C1')]; so, bo = c[(d,'O')]
        tot[d]=(outer*cpus*(nonor*sc1+orc*so) + 2*cpus*views*(nonor*bc1+orc*bo))
    return tot
e=est_main()
A=sum(e[d]*n for d,n in N.items() if n)+e['freeinference/week13']*FI_A
print('== v4 main cells only: A CPUh %.1f -> 8 procs %.1f h; B16 CPUh %.1f -> %.1f h; B extra12 CPUh %.1f -> %.1f h'%(A/3600, A/3600/8, e['freeinference/week13']*FI_ALL/3600, e['freeinference/week13']*FI_ALL/3600/8, e['freeinference/week13']*FI_B_extra/3600, e['freeinference/week13']*FI_B_extra/3600/8))
