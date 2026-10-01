"""產生 figures/fig_cake_extension.svg：整合架構（寫入→六態儲存層→Cake 讀取）＋ chunk 對比。示意，非量測。"""
W=1480; F='WenQuanYi Zen Hei, Noto Sans CJK TC, sans-serif'
INK='#1F2937'; MUTED='#4B5563'; GRID='#9CA3AF'
PUR='#6D28D9'; PURF='#F3EEFF'; BLUE='#0072B2'; BLUEF='#E6F1F8'; VERM='#D55E00'; GREEN='#009E73'
FILL={'DROP':'url(#hatch)','SSD':'#FBE7C6','CPU':'#D6EAF8','INT4':'#E3F4EC','FP8':'#C3E8D5','BF16':'#9FD8BC'}
NAME={'DROP':'不存','SSD':'SSD','CPU':'CPU','INT4':'GPU·INT4','FP8':'GPU·FP8','BF16':'GPU·BF16'}
out=[]
def rect(x,y,w,h,fill,stroke=GRID,sw=1,rx=0,dash=None):
    d=f' stroke-dasharray="{dash}"' if dash else ''
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
def text(x,y,t,size=14,color=INK,anchor='middle',weight='normal'):
    out.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}" dominant-baseline="middle">{t}</text>')
def arrow(pts,color=MUTED,sw=1.8,dash=None):
    d=f' stroke-dasharray="{dash}"' if dash else ''
    p=' '.join(f'{x},{y}' for x,y in pts)
    mid={'#6D28D9':'ap','#0072B2':'ab','#4B5563':'am','#D55E00':'av','#009E73':'ag'}[color]
    out.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}"{d} marker-end="url(#{mid})"/>')
def box(x,y,w,h,title,sub=None,fill='#FFFFFF',stroke=INK,tag=None,tagc=None):
    rect(x,y,w,h,fill,stroke,1.6,6)
    if sub: text(x+w/2,y+h/2-9,title,15); text(x+w/2,y+h/2+12,sub,12,MUTED)
    else: text(x+w/2,y+h/2,title,15)
    if tag:
        tw=len(tag)*13+14; rect(x+w-tw-6,y-10,tw,20,'#FFFFFF',tagc,1.2,10); text(x+w-tw/2-6,y,tag,11.5,tagc)
mk=''.join(f'<marker id="{i}" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="{c}"/></marker>' for i,c in [('ap',PUR),('ab',BLUE),('am',MUTED),('av',VERM),('ag',GREEN)])
defs=f'<defs><pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="8" height="8" fill="#FFFFFF"/><line x1="0" y1="0" x2="0" y2="8" stroke="#C7CBD1" stroke-width="3"/></pattern>{mk}</defs>'
# ================= (A) =================
text(20,26,'(A) 整合後的系統：寫入（新增）→ 六態儲存層 → 讀取（Cake，改造後）',17,INK,'start','bold')
# column backgrounds
rect(20,50,370,500,PURF,'#C4B5FD',1,8); text(36,68,'寫入路徑（本提案新增）',13.5,PUR,'start','bold')
rect(470,50,330,500,'#F8FAFC','#CBD5E1',1,8); text(486,68,'儲存層：每個 chunk 一種狀態',13.5,MUTED,'start','bold')
rect(880,50,580,500,BLUEF,'#93C5FD',1,8); text(896,68,'讀取路徑（Cake 的雙向還原）',13.5,BLUE,'start','bold')
# stack rows
SX=500; SW=270
rows={'BF16':(120,40),'FP8':(165,40),'INT4':(210,40),'CPU':(290,50),'SSD':(360,50),'DROP':(450,50)}
rect(SX-14,96,SW+28,170,'#FFFFFF','#94A3B8',1.2,6); text(SX+SW/2,108,'GPU 記憶體（最快、容量最小）',12,MUTED)
for k,(y,h) in rows.items():
    rect(SX,y,SW,h,FILL[k],GRID,1.2,4)
    lab={'BF16':'BF16（無損）','FP8':'FP8（品質依模型×任務）','INT4':'INT4（品質依模型×任務）','CPU':'CPU 記憶體','SSD':'SSD／磁碟','DROP':'不存（新增）'}[k]
    sub={'CPU':'較慢、較大','SSD':'最慢、最大','DROP':'省空間，要用時重算'}.get(k)
    if sub: text(SX+SW/2,y+h/2-9,lab,14.5); text(SX+SW/2,y+h/2+12,sub,12,MUTED)
    else: text(SX+SW/2,y+h/2,lab,14)
cy={k:y+h/2 for k,(y,h) in rows.items()}
# write path
box(50,92,310,48,'prefill 完成，產生新的 KV',fill='#FFFFFF',stroke=PUR)
box(50,175,310,290,'',fill='#FFFFFF',stroke=PUR,tag='新增',tagc=PUR)
text(205,200,'① 逐位置決策',16,PUR,weight='bold')
for k,t in enumerate(['每個 chunk 比較：','重算成本 R(p)（隨位置變貴）','vs 各層的載入成本']):
    text(205,236+k*26,t,13.5,MUTED)
out.append(f'<line x1="90" y1="322" x2="320" y2="322" stroke="#DDD6FE" stroke-width="1.2"/>')
for k,t in enumerate(['重算較便宜 → 不存','越貴 → 放越快的層','留 GPU → 依品質預算 ε 選精度','逐出時從前段開始丟']):
    text(205,348+k*28,t,13.5,INK)
box(50,490,310,48,'實測成本常數','κ、每筆搬運固定成本',fill='#FFFFFF',stroke=PUR)
arrow([(205,140),(205,173)],PUR); arrow([(205,490),(205,467)],PUR,1.6,'6 4')
# decision -> tiers (horizontal-ish, orthogonal)
for k,lbl in [('BF16',''),('FP8','最貴：留 GPU'),('INT4',''),('CPU','後段'),('SSD','中段'),('DROP','前段')]:
    y=cy[k]
    if k in('BF16','FP8','INT4'):
        if k=='FP8': arrow([(360,y),(498,y)],PUR); text(400,y-11,lbl,12,PUR)
        continue
    arrow([(360,y),(498,y)],PUR); text(415,y-11,lbl,12,PUR)
# GPU bracket arrows to BF16 / INT4 from FP8 trunk
arrow([(445,cy['FP8']),(445,cy['BF16']),(498,cy['BF16'])],PUR,1.4)
arrow([(445,cy['FP8']),(445,cy['INT4']),(498,cy['INT4'])],PUR,1.4)
# read path boxes
RX=930; RW=270
box(RX,130,RW,90,'已在 GPU：直接可用','INT4／FP8 需反量化（小）',fill='#FFFFFF',stroke=GREEN)
box(RX,285,RW,135,'',fill='#FFFFFF',stroke=BLUE,tag='改造',tagc=PUR); text(RX+RW/2,328,'I/O 從後段往前載入',15)
text(RX+RW/2,368,'Cake 只從磁碟載；',12,MUTED); text(RX+RW/2,388,'改為從 CPU＋SSD 多層載',12,PUR)
box(RX,445,RW,60,'GPU 從前段往後重算','Cake 原有',fill='#FFFFFF',stroke=BLUE)
arrow([(SX+SW,cy['FP8']),(RX-2,cy['FP8'])],GREEN)
arrow([(SX+SW,cy['CPU']),(RX-2,cy['CPU'])],BLUE); arrow([(SX+SW,cy['SSD']),(RX-2,cy['SSD'])],BLUE)
text((SX+SW+RX)/2,cy['CPU']-11,'載入',12,BLUE); text((SX+SW+RX)/2,cy['SSD']-11,'載入',12,BLUE)
arrow([(SX+SW,cy['DROP']),(RX-2,475)],MUTED,1.6,'6 4'); text(828,cy['DROP']-12,'本來就重算',12,MUTED)
# merge: three read results -> one collector bus -> diamond
MX=1345; MY=330; BX=1225
for y in (175,352,475):
    out.append(f'<line x1="{RX+RW}" y1="{y}" x2="{BX}" y2="{y}" stroke="{BLUE}" stroke-width="1.6"/>')
out.append(f'<line x1="{BX}" y1="175" x2="{BX}" y2="475" stroke="{BLUE}" stroke-width="1.6"/>')
arrow([(BX,MY),(MX-92,MY)],BLUE,1.6)
out.append(f'<polygon points="{MX},{MY-55} {MX+90},{MY} {MX},{MY+55} {MX-90},{MY}" fill="#FFFFFF" stroke="{BLUE}" stroke-width="1.6"/>')
text(MX,MY-8,'會合',15); text(MX,MY+14,'→ 開始 decode',13,MUTED)
box(1255,450,180,70,'③ 退路（新增）','一邊明顯較快就只走那一邊',fill='#FFFFFF',stroke=PUR)
arrow([(MX,MY+55),(MX,448)],PUR,1.6,'6 4')
# legend A
text(20,575,'紫框＝本提案新增或修改　　藍框＝Cake 原有　　綠框＝已在 GPU　　Cake 原本：沒有寫入路徑，所有 KV 都存在磁碟，讀取時只從磁碟載、沒有退路。',13,MUTED,'start')
# ================= (B) chunk rows =================
toks=['E','ating','cake','makes','me','happy',',','I','enjoy','eating','cakes','.']
ours=['DROP']*6+['SSD','SSD','CPU','INT4','FP8','BF16']
X0=260; CW=96; RH=46; y0=650
text(20,y0-26,'(B) 同一段 context 的 12 個 chunk：Cake（全存磁碟）vs 本提案（六態）　（重繪自 Cake 論文 Fig. 2）',17,INK,'start','bold')
def row(y,title,cells):
    rect(20,y,220,RH,'#FFFFFF',INK,1); text(130,y+RH/2,title,14)
    for i,(t,f,sub,subc) in enumerate(cells):
        rect(X0+i*CW,y,CW,RH,f,GRID,1)
        if sub: text(X0+i*CW+CW/2,y+17,t,14); text(X0+i*CW+CW/2,y+35,sub,11.5,subc)
        else: text(X0+i*CW+CW/2,y+RH/2,t,14)
C_COMP='#CCE3F0'; C_LOAD='#F6D5C2'; DISK='#FBE7C6'
row(y0,'Cake 寫入時',[(t,DISK,'磁碟',MUTED) for t in toks])
row(y0+RH,'Cake 讀取時',[(t,C_COMP,'重算',BLUE) if i<6 else (t,C_LOAD,'從磁碟載',VERM) for i,t in enumerate(toks)])
y1=y0+2*RH+30
row(y1,'本提案 寫入時',[(t,FILL[s],NAME[s],MUTED) for t,s in zip(toks,ours)])
row(y1+RH,'本提案 讀取時',[(t,C_COMP,'重算',BLUE) if i<6 else ((t,C_LOAD,'從 '+ours[i]+' 載',VERM) if i<9 else (t,FILL[ours[i]],'直接可用',GREEN)) for i,t in enumerate(toks)])
xm=X0+6*CW
out.append(f'<line x1="{xm}" y1="{y0-8}" x2="{xm}" y2="{y1+2*RH+8}" stroke="#B91C1C" stroke-width="2.5" stroke-dasharray="9 5"/>')
text(xm,y1+2*RH+24,'寫入分界 ＝ 讀取會合線',13,'#B91C1C',weight='bold')
yc=y1+2*RH+58
cap=['讀取時兩者的會合點相同（前 6 格都由 GPU 重算），但本提案前段不存、省下的空間讓後段放到更快的層：CPU、甚至留在 GPU（依品質預算選精度），',
     '讀取時 I/O 要搬的格數變少、來源也更快。12 個 chunk 與分界位置為示意；實際分界由實測成本決定，精度能否下降取決於模型與任務。']
for k,t in enumerate(cap): text(20,yc+k*24,t,13.5,INK,'start')
H=yc+2*24+20
svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{F}">{defs}<rect width="{W}" height="{H}" fill="#FFFFFF"/>'+''.join(out)+'</svg>'
open('figures/fig_cake_extension.svg','w').write(svg); print(H)
