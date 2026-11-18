from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
except Exception:
    font = ImageFont.load_default(); small = font

def centered(draw, xy, text, f):
    box = draw.multiline_textbbox((0,0), text, font=f, align="center")
    draw.multiline_text((xy[0]-(box[2]-box[0])/2, xy[1]-(box[3]-box[1])/2), text, font=f, fill="#1f2933", align="center")

def diagram(name, boxes, links, note=None):
    im=Image.new("RGB",(1800,650),"white"); d=ImageDraw.Draw(im)
    for x,y,w,h,t,c in boxes:
        d.rounded_rectangle((x,y,x+w,y+h),radius=18,fill=c,outline="#315b7d",width=3); centered(d,(x+w/2,y+h/2),t,font)
    for a,b in links:
        x1,y1= a; x2,y2=b; d.line((x1,y1,x2,y2),fill="#4d4d4d",width=4)
        d.polygon([(x2,y2),(x2-14,y2-8),(x2-14,y2+8)],fill="#4d4d4d")
    if note: d.text((900,585),note,font=small,fill="#7a1f1f",anchor="mm")
    im.save(OUT/f"{name}.png",dpi=(300,300)); im.save(OUT/f"{name}.pdf",resolution=300)

diagram("fig1_architecture",[(30,250,240,100,"Question /\ncontext","#eaf2f8"),(350,250,270,100,"TinyLlama\n1.1B backbone","#dceef7"),(730,250,290,100,"Mature + premature\nhidden states","#eaf2f8"),(1130,190,270,100,"DoLa\nJSD + contrast","#e8f5e9"),(1130,350,270,100,"ENN\nuncertainty","#fff3cd"),(1510,250,250,120,"Answer score /\naccept or abstain","#f8d7da")], [((270,300),(350,300)),((620,300),(730,300)),((1020,300),(1130,240)),((1020,300),(1130,400)),((1400,240),(1510,290)),((1400,400),(1510,330))], "Archived answer-correctness ENN benchmark is reported separately; its source is absent from the active checkout.")
diagram("fig2_enn",[(30,250,250,110,"[h_m ; h_p]\n4096-D","#eaf2f8"),(380,250,250,110,"MLP\n4096→512","#eaf2f8"),(730,250,250,110,"MLP\n512→512","#eaf2f8"),(1080,250,280,110,"MLP\n512→2048×6","#eaf2f8"),(1430,140,250,100,"f_prior\n(frozen)","#e9ecef"),(1430,390,250,100,"f_learn\n(trainable)","#fff3cd")], [((280,305),(380,305)),((630,305),(730,305)),((980,305),(1080,305)),((1360,305),(1430,190)),((1360,305),(1430,440))], "Each branch contracts the 6-D epistemic index z; outputs are added before LM-head projection.")
diagram("fig3_dola",[(30,250,260,110,"Hidden states\nblocks 0…21","#eaf2f8"),(390,130,260,100,"Mature logits\nW h_21","#dceef7"),(390,390,260,100,"Candidate logits\nW·norm(h_j)","#dceef7"),(770,250,280,110,"JSD(mature,j)\nfor j∈{0,4,8,12,16,20}","#e8f5e9"),(1170,250,230,110,"j* = argmax\nJSD","#e8f5e9"),(1510,250,260,110,"l = l_m − α l_p\nsoftmax / decode","#fff3cd")], [((290,305),(390,180)),((290,305),(390,440)),((650,180),(770,280)),((650,440),(770,330)),((1050,305),(1170,305)),((1400,305),(1510,305))])
diagram("fig4_self_consistency",[(30,250,220,110,"Question","#eaf2f8"),(390,100,260,80,"DoLa sample 1","#e8f5e9"),(390,250,260,80,"DoLa sample 2","#e8f5e9"),(390,400,260,80,"DoLa sample N","#e8f5e9"),(820,250,280,110,"Normalize /\ncluster answers","#eaf2f8"),(1250,250,250,110,"Plurality or\nweighted vote","#fff3cd"),(1600,250,150,110,"Answer","#f8d7da")], [((250,305),(390,140)),((250,305),(390,290)),((250,305),(390,440)),((650,140),(820,280)),((650,290),(820,305)),((650,440),(820,330)),((1100,305),(1250,305)),((1500,305),(1600,305))])
diagram("fig5_detection_pipeline",[(30,250,200,110,"Generate","#eaf2f8"),(320,250,220,110,"Token\nuncertainty","#eaf2f8"),(630,250,220,110,"Answer\nfeatures","#eaf2f8"),(940,250,200,110,"ENN\nscore","#fff3cd"),(1230,250,250,110,"Threshold\n(selected on val)","#e8f5e9"),(1580,250,190,110,"Accept /\nreject","#f8d7da")], [((230,305),(320,305)),((540,305),(630,305)),((850,305),(940,305)),((1140,305),(1230,305)),((1480,305),(1580,305))], "Detection estimates reliability; mitigation changes decoding or aggregates candidates.")

def bar_chart(name, labels, vals, title, ylabel, color="#3182bd"):
    im=Image.new("RGB",(1800,700),"white"); d=ImageDraw.Draw(im)
    left,bottom,top,right=130,590,90,1740; maxv=max(vals)*1.2
    d.text((900,35),title,font=font,fill="#1f2933",anchor="mm")
    d.line((left,top,left,bottom),fill="#333",width=3); d.line((left,bottom,right,bottom),fill="#333",width=3)
    for k in range(6):
        y=bottom-(bottom-top)*k/5; v=maxv*k/5; d.line((left,y,right,y),fill="#e5e7eb",width=1); d.text((105,y),f"{v:.1f}",font=small,fill="#555",anchor="rm")
    gap=(right-left)/len(vals)
    for i,(lab,v) in enumerate(zip(labels,vals)):
        x=left+gap*i+gap*.18; w=gap*.64; y=bottom-(bottom-top)*v/maxv
        d.rectangle((x,y,x+w,bottom),fill=color,outline="#1f4e79",width=2)
        d.text((x+w/2,y-14),f"{v:.3f}",font=small,fill="#111",anchor="ms")
        d.text((x+w/2,bottom+18),lab,font=small,fill="#222",anchor="ma")
    d.text((35,340),ylabel,font=small,fill="#222",anchor="mm")
    im.save(OUT/f"{name}.png",dpi=(300,300)); im.save(OUT/f"{name}.pdf",resolution=300)

bar_chart("fig6_performance",["Base","DoLa","DoLa+SC","ENN-B","DoLa+ENN","Complete"],[.284,.299,.654,.782,.768,.725],"Archived held-out accuracy (n=211)","Accuracy", "#5b8db8")
bar_chart("fig7_ablation",["Rep","Token","SC","Rep+Token","+SC3","Full"],[.6681,.3664,.5758,.6695,.6694,.6713],"Archived feature ablation","AUROC", "#6a9f58")

im=Image.new("RGB",(900,800),"white"); d=ImageDraw.Draw(im); d.text((450,55),"DoLa + ENN-V2 + SC: archived confusion matrix",font=font,fill="#1f2933",anchor="mm")
x0,y0=240,170; s=240; cols=["#dbeafe","#93c5fd","#60a5fa","#2563eb"]; vals=[[11,43],[6,151]]
for i in range(2):
  for j in range(2):
    v=vals[i][j]; idx=0 if v<20 else 1 if v<50 else 2 if v<100 else 3
    d.rectangle((x0+j*s,y0+i*s,x0+(j+1)*s,y0+(i+1)*s),fill=cols[idx],outline="white",width=4)
    d.text((x0+(j+.5)*s,y0+(i+.5)*s),str(v),font=font,fill="white" if idx>=2 else "#111",anchor="mm")
d.text((x0+s,y0+2*s+50),"Predicted positive     Predicted negative",font=small,fill="#222",anchor="mm")
d.text((x0-70,y0+s),"True\nclass",font=small,fill="#222",anchor="mm")
d.text((x0-20,y0+s/2),"Positive",font=small,fill="#222",anchor="rm"); d.text((x0-20,y0+1.5*s),"Negative",font=small,fill="#222",anchor="rm")
im.save(OUT/"fig8_confusion.png",dpi=(300,300)); im.save(OUT/"fig8_confusion.pdf",resolution=300)
