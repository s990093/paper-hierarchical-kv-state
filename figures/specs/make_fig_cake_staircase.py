"""產生 figures/fig_cake_staircase.svg（學術風格）。
以 Cake Fig. 2 為底：(a) 逐位置的同一條規則、(b) 寫入時的放置、(c) 讀取時的雙向還原；
寫入分界與讀取會合線為同一條線。所有數值為示意，不是量測。"""
toks=['E','ating','cake','makes','me','happy',',','I','enjoy','eating','cakes','.']
N=12; MEET=6; CPU_FROM=10
W=1500; X0=290; CW=96; RH=54; GRID='#9CA3AF'; INK='#1F2937'; MUTED='#4B5563'
F='WenQuanYi Zen Hei, Noto Sans CJK TC, sans-serif'
# Okabe–Ito
BLUE='#0072B2'; VERM='#D55E00'; SKY='#56B4E9'; ORNG='#E69F00'
C_COMP='#CCE3F0'; C_LOAD='#F6D5C2'; C_DONE='#F3F4F6'; C_SSD='#FBE7C6'; C_CPU='#D6EAF8'
out=[]
def rect(x,y,w,h,fill,stroke=GRID,sw=1,extra=''):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>')
def text(x,y,t,size=15,color=INK,anchor='middle',weight='normal',style='normal'):
    out.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}" font-style="{style}" dominant-baseline="middle">{t}</text>')
def poly(pts,color,sw=3,dash=None,marker=''):
    d=f' stroke-dasharray="{dash}"' if dash else ''
    p=' '.join(f'{x:.1f},{y:.1f}' for x,y in pts)
    out.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="miter" stroke-linecap="butt"{d}{marker}/>')
cx=lambda i: X0+i*CW
defs=f'''<defs>
<pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
<rect width="8" height="8" fill="#FFFFFF"/><line x1="0" y1="0" x2="0" y2="8" stroke="#C7CBD1" stroke-width="3"/></pattern>
<marker id="ah" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker>
</defs>'''
def panel(x,y,t): text(x,y,t,17,INK,'start','bold')
# ---------------- (a) cost bars ----------------
ya=30; panel(20,ya,'(a) 逐位置的同一條規則')
text(40,ya+34,'比較每個 chunk 的',14,MUTED,'start'); text(40,ya+56,'重算成本 R(p)',14,BLUE,'start'); text(40,ya+78,'與載入成本 L',14,VERM,'start')
Hmax=140; base=ya+40+Hmax
# axes
poly([(X0-14,base),(cx(N)+20,base)],MUTED,1.2,marker=' marker-end="url(#ah)"')
poly([(X0-14,base),(X0-14,base-Hmax-12)],MUTED,1.2,marker=' marker-end="url(#ah)"')
text(X0-22,base-Hmax-4,'成本',13,MUTED,'end')
for i in range(N):
    h=Hmax*(i+1)/N
    rect(cx(i)+22,base-h,CW-44,h,C_COMP,BLUE,1.2)
LSSD=base-Hmax*(MEET+0.5)/N
poly([(X0-14,LSSD),(cx(N)+4,LSSD)],VERM,2,'8 5')
text(X0+4,LSSD+16,'L：從 SSD 載入的成本（與位置無關）',13,VERM,'start')
text(X0+4,base-Hmax+6,'R(p) 隨位置線性增加',13,BLUE,'start')
# axis captions (user-requested arrows)
ax=base+24
text(X0-14,ax,'位置：',14,MUTED,'end'); poly([(X0,ax),(cx(N),ax)],MUTED,1.2,marker=' marker-end="url(#ah)"'); text(X0+4,ax-12,'前',13,MUTED,'start'); text(cx(N)-6,ax-12,'後',13,MUTED,'end')
ax2=ax+30
text(X0-14,ax2,'重算：',14,MUTED,'end'); poly([(X0,ax2),(cx(N),ax2)],MUTED,1.2,marker=' marker-end="url(#ah)"'); text(X0+4,ax2-12,'便宜',13,MUTED,'start'); text(cx(N)-6,ax2-12,'貴',13,MUTED,'end')
# region brackets
yb=ax2+28
def bracket(i0,i1,label,color):
    x1=cx(i0)+4; x2=cx(i1)-4
    poly([(x1,yb-6),(x1,yb),(x2,yb),(x2,yb-6)],color,1.5)
    text((x1+x2)/2,yb+16,label,13,color)
bracket(0,MEET,'R(p) ＜ L：重算較便宜 → 不存',MUTED)
bracket(MEET,CPU_FROM,'R(p) ＞ L → 存 SSD',ORNG)
bracket(CPU_FROM,N,'最貴 → CPU（容量有限）',BLUE)
# ---------------- (b) write row ----------------
yw=yb+62; panel(20,yw-16,'(b) 寫入時（本提案新增）')
rect(20,yw+4,250,RH+8,'#FFFFFF',INK,1); text(145,yw+4+(RH+8)/2,'放置決策',15)
for i,t in enumerate(toks):
    if i<MEET: f,sub='url(#hatch)','不存'
    elif i>=CPU_FROM: f,sub=C_CPU,'CPU'
    else: f,sub=C_SSD,'SSD'
    rect(cx(i),yw+4,CW,RH+8,f,GRID,1)
    text(cx(i)+CW/2,yw+4+22,t,16); text(cx(i)+CW/2,yw+4+46,sub,12,MUTED)
# ---------------- (c) read stages ----------------
ys=yw+RH+8+62; panel(20,ys-20,'(c) 讀取時：雙向還原（同 Cake Fig. 2）')
stages=[('Stage 0',0,N),('Stage 1',2,10),('⋯',4,8),('Stage N−1',6,6),('Stage N',6,6)]
rows=[]
for r,(name,c,io) in enumerate(stages):
    yy=ys+r*RH; rows.append(yy)
    rect(20,yy,250,RH,'#FFFFFF',INK,1); text(145,yy+RH/2,name,15,style='italic' if name!='⋯' else 'normal')
    for i,t in enumerate(toks):
        sub=None; f='#FFFFFF'; lab=t
        if name=='Stage N': f=C_DONE
        elif name=='Stage N−1':
            f=C_COMP if i in(4,5) else (C_LOAD if i in(6,7) else C_DONE); sub='SSD' if i in(6,7) else None
        elif name=='Stage 1':
            f=C_COMP if i<2 else (C_LOAD if i>=10 else '#FFFFFF'); sub='CPU' if i>=10 else None
        elif name=='⋯': lab=''
        rect(cx(i),yy,CW,RH,f,GRID,1)
        if sub: text(cx(i)+CW/2,yy+20,lab,15); text(cx(i)+CW/2,yy+40,'('+sub+')',11,VERM)
        else: text(cx(i)+CW/2,yy+RH/2,lab,15)
def stair(vals,color):
    pts=[]
    for r,v in enumerate(vals):
        x=cx(v); pts+= [(x,rows[r]),(x,rows[r]+RH)]
    poly(pts,color,3.5)
stair([s[1] for s in stages],BLUE); stair([s[2] for s in stages],VERM)
text(cx(2)+6,rows[1]+RH+14,'GPU 重算前緣 →',13,BLUE,'start')
text(cx(10)-6,rows[1]+RH+14,'← I/O 載入前緣',13,VERM,'end')
# unified boundary line
xm=cx(MEET); ytop=yw-2; ybot=rows[-1]+RH+10
poly([(xm,ytop),(xm,ybot)],'#B91C1C',3,'10 5')
text(xm+8,ybot+16,'寫入分界 ＝ 讀取會合線',14,'#B91C1C','start','bold')
# legend
yl=ybot+52
items=[(C_COMP,BLUE,'GPU 重算'),(C_LOAD,VERM,'I/O 載入（括號：來源層）'),(C_DONE,GRID,'已完成'),('url(#hatch)',GRID,'不存'),(C_SSD,GRID,'存 SSD'),(C_CPU,GRID,'存 CPU')]
x=20
for f,s,t in items:
    rect(x,yl-9,20,18,f,s,1.2); text(x+28,yl,t,13,MUTED,'start'); x+=28+len(t)*14+34
# caption
cap=['圖：以 Cake [ICML\'25] Fig. 2 為底的逐位置 KV 決策（示意，非量測）。(a) 重算成本 R(p) 隨位置增加、載入成本 L 與位置無關，',
     '同一條規則把序列切成「不存／存 SSD／存 CPU」三段。(b) 寫入時依此放置。(c) 讀取時與 Cake 相同：GPU 由前往後重算、I/O 由後往前',
     '載入；兩條前緣最終會合的位置，正是 (b) 的寫入分界。分界隨實測成本移動，前段比例約 1/(1+κ)，κ＝重算成本 ÷ 傳輸成本。']
for k,t in enumerate(cap): text(20,yl+40+k*24,t,13.5,INK,'start')
H=int(yl+40+len(cap)*24+16)
svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{F}">{defs}<rect width="{W}" height="{H}" fill="#FFFFFF"/>'+''.join(out)+'</svg>'
open('figures/fig_cake_staircase.svg','w').write(svg); print(H)
