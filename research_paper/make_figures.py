import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.titlesize": 9})

def save(name):
    plt.tight_layout()
    plt.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close()

def box(ax, xy, w, h, text, color="#eaf2f8"):
    p = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                       linewidth=0.8, edgecolor="#315b7d", facecolor=color)
    ax.add_patch(p); ax.text(xy[0]+w/2, xy[1]+h/2, text, ha="center", va="center", wrap=True)

def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10,
                                 linewidth=0.8, color="#4d4d4d"))

# Fig. 1: architecture
fig, ax = plt.subplots(figsize=(7.0, 2.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
box(ax,(0.2,2.55),1.25,0.65,"Question /\ncontext")
box(ax,(2.0,2.55),1.45,0.65,"TinyLlama\n1.1B backbone", "#dceef7")
box(ax,(4.0,2.55),1.55,0.65,"Mature + premature\nhidden states")
box(ax,(6.1,2.9),1.45,0.65,"DoLa\nJSD + contrast", "#e8f5e9")
box(ax,(6.1,1.95),1.45,0.65,"ENN\nuncertainty", "#fff3cd")
box(ax,(8.2,2.45),1.55,0.8,"Answer score /\naccept or abstain", "#f8d7da")
for a,b in [((1.45,2.88),(2,2.88)),((3.45,2.88),(4,2.88)),((5.55,2.88),(6.1,3.22)),((5.55,2.75),(6.1,2.28)),((7.55,3.22),(8.2,2.95)),((7.55,2.28),(8.2,2.75))]: arrow(ax,a,b)
ax.text(5,0.55,"Archived benchmark adds supervised answer-correctness ENN variants; source is not present in the active checkout.",ha="center",fontsize=7,color="#7a1f1f")
save("fig1_architecture")

# Fig. 2: ENN architecture
fig, ax = plt.subplots(figsize=(7.0, 2.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
box(ax,(0.3,1.1),1.35,0.7,"[h_m ; h_p]\n4096-D")
box(ax,(2.2,1.1),1.35,0.7,"MLP\n4096→512")
box(ax,(4.1,1.1),1.35,0.7,"MLP\n512→512")
box(ax,(6.0,1.1),1.35,0.7,"MLP\n512→2048×6")
box(ax,(8.0,1.85),1.35,0.7,"f_prior\n(frozen)","#e9ecef")
box(ax,(8.0,0.4),1.35,0.7,"f_learn\n(trainable)","#fff3cd")
for x in [1.65,3.55,5.45]: arrow(ax,(x,1.45),(x+0.55,1.45))
arrow(ax,(7.35,1.55),(8.0,2.2)); arrow(ax,(7.35,1.35),(8.0,0.75)); arrow(ax,(8.68,1.85),(9.25,1.55)); arrow(ax,(8.68,0.75),(9.25,1.55))
ax.text(9.25,1.25,"+ z∼N(0,I₆)\n→ ENN hidden residual",ha="center",va="top",fontsize=8)
save("fig2_enn")

# Fig. 3: DoLa workflow
fig, ax = plt.subplots(figsize=(7.0, 2.1)); ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
box(ax,(0.2,1.1),1.3,0.75,"Hidden states\nblocks 0…21")
box(ax,(2.0,1.65),1.5,0.7,"Mature logits\nW h₂₁")
box(ax,(2.0,0.55),1.5,0.7,"Candidate logits\nW·norm(h_j)")
box(ax,(4.1,1.1),1.55,0.75,"JSD(mature, j)\nfor j∈{0,4,8,12,16,20}")
box(ax,(6.25,1.1),1.45,0.75,"j* = argmax\nJSD")
box(ax,(8.25,1.1),1.45,0.75,"ℓ = ℓ_m − αℓ_p\nsoftmax / decode")
for a,b in [((1.5,1.48),(2,1.95)),((1.5,1.48),(2,0.9)),((3.5,1.95),(4.1,1.55)),((3.5,0.9),(4.1,1.4)),((5.65,1.48),(6.25,1.48)),((7.7,1.48),(8.25,1.48))]: arrow(ax,a,b)
save("fig3_dola")

# Fig. 4: self consistency
fig, ax = plt.subplots(figsize=(7.0, 2.0)); ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
box(ax,(0.3,1.05),1.35,0.8,"Question")
for i,y in enumerate([2.15,1.5,0.85]): box(ax,(2.2,y-0.25),1.55,0.5,f"DoLa sample {i+1}","#e8f5e9"); arrow(ax,(1.65,1.45),(2.2,y))
box(ax,(4.55,1.05),1.6,0.8,"Normalize /\ncluster answers")
box(ax,(7.0,1.05),1.45,0.8,"Plurality or\nweighted vote")
box(ax,(8.95,1.05),0.85,0.8,"Answer")
for y in [2.15,1.5,0.85]: arrow(ax,(3.75,y),(4.55,1.45))
arrow(ax,(6.15,1.45),(7,1.45)); arrow(ax,(8.45,1.45),(8.95,1.45))
save("fig4_self_consistency")

# Fig. 5: pipeline / detection decision
fig, ax = plt.subplots(figsize=(7.0, 2.0)); ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
for x,t in [(0.2,"Generate"),(1.9,"Token\nuncertainty"),(3.7,"Answer\nfeatures"),(5.5,"ENN\nscore"),(7.3,"Threshold\n(selected on val)"),(9.0,"Accept /\nreject")]: box(ax,(x,1.1),1.25,0.8,t)
for x in [1.45,3.25,5.05,6.85,8.55]: arrow(ax,(x,1.5),(x+0.45,1.5))
ax.text(5,0.45,"Detection estimates reliability; mitigation changes decoding or aggregates candidates.",ha="center",fontsize=8)
save("fig5_detection_pipeline")

# Fig. 6: actual comparison
metrics = json.load(open(ROOT/"results/audited_benchmark/comparative_metrics.json"))
names = ["Base", "DoLa", "DoLa+SC", "ENN-B\n2048-D", "DoLa+ENN\nSC", "Complete"]
keys = ["Base TinyLlama (E0)","TinyLlama + DoLa (E5)","TinyLlama + DoLa + SC (E6)","2048-D ENN Alone (E2_B)","DoLa + ENN-V2 + SC (E7)","Complete Combined Architecture"]
vals = [metrics[k]["accuracy"] for k in keys]
fig, ax = plt.subplots(figsize=(7.0, 2.7)); bars=ax.bar(names,vals,color=["#9ecae1","#6baed6","#74c476","#fd8d3c","#9e9ac8","#fdae6b"])
ax.set_ylim(0,0.9); ax.set_ylabel("Accuracy"); ax.set_title("Archived held-out benchmark (n=211)")
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+0.02,f"{v:.3f}",ha="center",fontsize=7)
save("fig6_performance")

# Fig. 7: ablation AUROC
ab = json.load(open(ROOT/"results/audited_benchmark/sc_ablation_results.json"))
labels=["Rep only","Token only","SC only","Rep+token","Rep+token+SC","Full"]
keys=[list(ab)[0],list(ab)[1],list(ab)[2],list(ab)[3],list(ab)[6],list(ab)[10]]
vals=[ab[k]["auroc"] for k in keys]
fig, ax=plt.subplots(figsize=(7.0,2.7)); bars=ax.bar(labels,vals,color="#3182bd"); ax.set_ylim(0,0.8); ax.set_ylabel("AUROC"); ax.set_title("Archived feature ablation")
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+0.015,f"{v:.3f}",ha="center",fontsize=7)
save("fig7_ablation")

# Fig. 8: confusion matrix for E7
cm=np.array([[11,43],[6,151]])
fig, ax=plt.subplots(figsize=(3.2,2.8)); im=ax.imshow(cm,cmap="Blues"); ax.set_xticks([0,1],labels=["Pred. positive","Pred. negative"]); ax.set_yticks([0,1],labels=["True positive","True negative"])
for i in range(2):
  for j in range(2): ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>80 else "black",fontsize=11)
ax.set_title("DoLa + ENN-V2 + SC\narchived confusion matrix")
save("fig8_confusion")
