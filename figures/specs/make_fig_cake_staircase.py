"""產生 figures/fig_cake_staircase.svg（學術風格）：六態動作空間＋逐位置決策＋寫入分界＝讀取會合線。
以 Cake Fig. 2 為底。所有數值與分界為示意，不是量測。"""
toks=['E','ating','cake','makes','me','happy',',','I','enjoy','eating','cakes','.']
N=12; MEET=6
# 每格狀態（示意）：0-5 不存、6-7 SSD、8 CPU、9 GPU-INT4、10 GPU-FP8、11 GPU-BF16
state=['DROP']*6+['SSD','SSD','CPU','INT4','FP8','BF16']
GPU0=9
W=1560; X0=300; CW=98; RH=54; GRID='#9CA3AF'; INK='#1F2937'; MUTED='#4B5563'
F='WenQuanYi Zen Hei, Noto Sans CJK TC, sans-serif'
BLUE='#0072B2'; VERM='#D55E00'; ORNG='#E69F00'; GREEN='#009E73'
C_COMP='#CCE3F0'; C_LOAD='#F6D5C2'; C_DONE='#F3F4F6'
FILL={'DROP':'url(#hatch)','SSD':'#FBE7C6','CPU':'#D6EAF8','INT4':'#E3F4EC','FP8':'#C3E8D5','BF16':'#9FD8BC'}
NAME={'DROP':'不存','SSD':'SSD','CPU':'CPU','INT4':'GPU·INT4','FP8':'GPU·FP8','BF16':'GPU·BF16'}
out=[]
def rect(x,y,w,h,fill,stroke=GRID,sw=1):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
def text(x,y,t,size=15,color=INK,anchor='middle',weight='normal',style='normal'):
    out.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}" font-style="{style}" dominant-baseline="middle">{t}</text>')
def poly(pts,color,sw=3,dash=None,marker=''):
    d=f' stroke-dasharray="{dash}"' if dash else ''
    p=' '.join(f'{x:.1f},{y:.1f}' for x,y in pts)
    out.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="miter"{d}{marker}/>')
cx=lambda i: X0+i*CW
defs=f'''<defs><pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
<rect width="8" height="8" fill="#FFFFFF"/><line x1="0" y1="0" x2="0" y2="8" stroke="#C7CBD1" stroke-width="3"/></pattern>
<marker id="ah" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs>'''
def panel(x,y,t): text(x,y,t,17,INK,'start','bold')
# ---------------- (a) 六態表 ----------------
ya=28; panel(20,ya,'(a) 每個 chunk 可選的 6 種狀態（動作空間）')
cols=[('狀態',150),('放在哪',130),('精度',110),('要用時的代價',380),('佔 GPU 空間',170),('品質',330)]
rows_a=[('BF16','GPU','16 bit','無（已在 GPU）','1','無損'),
        ('FP8','GPU','8 bit','讀取時反量化（小）','約 1/2','依模型 × 任務'),
        ('INT4','GPU','4 bit','讀取時反量化（小）','約 1/3.77','依模型 × 任務（Qwen2.5 會崩）'),
        ('CPU','CPU 記憶體','原樣','經 PCIe 載入（含每筆固定成本）','0','無損'),
        ('SSD','SSD','原樣','經 SSD 路徑載入（最慢）','0','無損'),
        ('DROP','—','—','重算 R(p)，隨位置變貴','0','無損')]
tx=20; ty=ya+22; rh=30
x=tx
for name,w in cols:
    rect(x,ty,w,rh,'#F9FAFB',GRID,1); text(x+w/2,ty+rh/2,name,14,INK,weight='bold'); x+=w
for r,row in enumerate(rows_a):
    x=tx; yy=ty+rh*(r+1)
    for c,(name,w) in enumerate(cols):
        f='#FFFFFF'
        if c==0: f=FILL[row[0]]
        rect(x,yy,w,rh,f,GRID,1)
        val=NAME[row[0]] if c==0 else row[c]
        text(x+w/2,yy+rh/2,val,13.5,INK if c==0 else MUTED); x+=w
text(tx,ty+rh*7+16,'註：INT8 可視為精度的另一階；前三種在 GPU 內換「空間 vs 品質」，後三種是無損但「要用時付代價」。',13,MUTED,'start')
# ---------------- (b) cost bars ----------------
yb0=ty+rh*7+60; panel(20,yb0,'(b) 同一條逐位置規則：比較重算成本 R(p) 與各層的載入成本')
Hmax=130; base=yb0+30+Hmax
poly([(X0-14,base),(cx(N)+20,base)],MUTED,1.2,marker=' marker-end="url(#ah)"')
poly([(X0-14,base),(X0-14,base-Hmax-12)],MUTED,1.2,marker=' marker-end="url(#ah)"')
text(X0-22,base-Hmax-4,'成本',13,MUTED,'end')
for i in range(N):
    h=Hmax*(i+1)/N; rect(cx(i)+24,base-h,CW-48,h,C_COMP,BLUE,1.2)
LSSD=base-Hmax*(MEET+0.5)/N
poly([(X0-14,LSSD),(cx(N)+4,LSSD)],VERM,2,'8 5')
text(X0+4,LSSD+16,'L：從 SSD 載入的成本（與位置無關）',13,VERM,'start')
text(X0+4,base-Hmax+6,'R(p) 隨位置線性增加',13,BLUE,'start')
ax=base+22
text(X0-14,ax,'位置：',14,MUTED,'end'); poly([(X0,ax),(cx(N),ax)],MUTED,1.2,marker=' marker-end="url(#ah)"'); text(X0+4,ax-11,'前',13,MUTED,'start'); text(cx(N)-6,ax-11,'後',13,MUTED,'end')
ax2=ax+28
text(X0-14,ax2,'重算：',14,MUTED,'end'); poly([(X0,ax2),(cx(N),ax2)],MUTED,1.2,marker=' marker-end="url(#ah)"'); text(X0+4,ax2-11,'便宜',13,MUTED,'start'); text(cx(N)-6,ax2-11,'貴',13,MUTED,'end')
yb=ax2+26
def bracket(i0,i1,label,color):
    x1=cx(i0)+4; x2=cx(i1)-4
    poly([(x1,yb-6),(x1,yb),(x2,yb),(x2,yb-6)],color,1.5); text((x1+x2)/2,yb+16,label,13,color)
bracket(0,MEET,'R(p) ＜ L：重算較便宜 → 不存',MUTED)
bracket(MEET,8,'存 SSD',ORNG)
bracket(8,GPU0,'CPU',BLUE)
bracket(GPU0,N,'留 GPU：精度依品質預算 ε',GREEN)
text(cx(N),yb+36,'越貴的放越快的層；層的容量有限，放不下就往下一層',12.5,MUTED,'end')
# ---------------- (c) write row ----------------
yw=yb+78; panel(20,yw-16,'(c) 寫入時（本提案新增）：每格選一種狀態')
rect(20,yw+4,260,RH+8,'#FFFFFF',INK,1); text(150,yw+4+(RH+8)/2,'放置決策',15)
for i,t in enumerate(toks):
    rect(cx(i),yw+4,CW,RH+8,FILL[state[i]],GRID,1)
    text(cx(i)+CW/2,yw+4+22,t,16); text(cx(i)+CW/2,yw+4+46,NAME[state[i]],12,MUTED)
# ---------------- (d) read ----------------
ys=yw+RH+8+64; panel(20,ys-20,'(d) 讀取時：雙向還原（同 Cake Fig. 2）；已在 GPU 的格子不必搬')
stages=[('Stage 0',0,GPU0),('Stage 1',2,8),('⋯',4,7),('Stage N−1',6,6),('Stage N',6,6)]
rows=[]
for r,(name,c,io) in enumerate(stages):
    yy=ys+r*RH; rows.append(yy)
    rect(20,yy,260,RH,'#FFFFFF',INK,1); text(150,yy+RH/2,name,15,style='italic' if name!='⋯' else 'normal')
    prev_io = stages[r-1][2] if r>0 else N
    for i,t in enumerate(toks):
        sub=None; f='#FFFFFF'; lab=t
        if i>=GPU0: f=FILL[state[i]]; sub=NAME[state[i]] if name=='Stage 0' else None
        elif name=='Stage N': f=C_DONE
        elif i<c: f=C_DONE
        elif i>=io: f=C_DONE
        if name!='Stage N' and name!='⋯':
            if c-2<=i<c and c>0: f=C_COMP
            if io<=i<prev_io and i<GPU0: f=C_LOAD; sub=state[i]
        if name=='Stage 0' and i<GPU0: f='#FFFFFF'
        if name=='⋯': lab='' ; f='#FFFFFF' if i<GPU0 else FILL[state[i]]
        rect(cx(i),yy,CW,RH,f,GRID,1)
        if sub: text(cx(i)+CW/2,yy+20,lab,15); text(cx(i)+CW/2,yy+40,'('+sub+')',11,VERM if f==C_LOAD else MUTED)
        else: text(cx(i)+CW/2,yy+RH/2,lab,15)
def stair(vals,color):
    pts=[]
    for r,v in enumerate(vals): x=cx(v); pts+=[(x,rows[r]),(x,rows[r]+RH)]
    poly(pts,color,3.5)
stair([s[1] for s in stages],BLUE); stair([s[2] for s in stages],VERM)
text(cx(2)+6,rows[1]+RH+14,'GPU 重算前緣 →',13,BLUE,'start')
text(cx(8)-6,rows[1]+RH+14,'← I/O 載入前緣',13,VERM,'end')
text(cx(GPU0)+3*CW/2,rows[2]+RH/2,'已在 GPU，直接可用',12.5,GREEN)
xm=cx(MEET); ytop=yw-2; ybot=rows[-1]+RH+10
poly([(xm,ytop),(xm,ybot)],'#B91C1C',3,'10 5')
text(xm+8,ybot+16,'寫入分界 ＝ 讀取會合線',14,'#B91C1C','start','bold')
# legend & caption
yl=ybot+50
items=[(C_COMP,BLUE,'GPU 重算'),(C_LOAD,VERM,'I/O 載入（括號：來源）'),(C_DONE,GRID,'已完成'),('url(#hatch)',GRID,'不存'),(FILL['SSD'],GRID,'SSD'),(FILL['CPU'],GRID,'CPU'),(FILL['INT4'],GRID,'GPU·INT4'),(FILL['FP8'],GRID,'GPU·FP8'),(FILL['BF16'],GRID,'GPU·BF16')]
x=20
for f,s,t in items:
    rect(x,yl-9,20,18,f,s,1.2); text(x+28,yl,t,13,MUTED,'start'); x+=28+len(t)*12+30
cap=['圖：以 Cake [ICML\'25] Fig. 2 為底的逐位置 KV 決策（示意，非量測）。(a) 每個 chunk 有 6 種狀態：GPU 內三種精度（換空間與品質），以及 CPU、SSD、',
     '不存三種無損但要付代價的狀態。(b) 重算成本 R(p) 隨位置增加，與各層載入成本比較；越貴的放越快的層，留在 GPU 的再依品質預算 ε 選精度。',
     '(c) 寫入時依此放置。(d) 讀取時與 Cake 相同：GPU 由前往後重算、I/O 由後往前載入；已在 GPU 的格子不必搬；兩條前緣會合的位置即 (c) 的寫入分界。',
     '分界隨實測成本移動，前段比例約 1/(1+κ)，κ＝重算成本 ÷ 傳輸成本；精度可否下降取決於模型與任務。']
for k,t in enumerate(cap): text(20,yl+40+k*24,t,13.5,INK,'start')
H=int(yl+40+len(cap)*24+16)
svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{F}">{defs}<rect width="{W}" height="{H}" fill="#FFFFFF"/>'+''.join(out)+'</svg>'
open('figures/fig_cake_staircase.svg','w').write(svg); print(H)
