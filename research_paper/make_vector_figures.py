from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

W, H = 1600, 520
FONT = "DejaVu Sans, Arial, sans-serif"
BLUE = "#e5f0f8"
GREEN = "#e5f4ea"
YELLOW = "#fff1c9"
GRAY = "#e8edf1"
BORDER = "#315b7d"
INK = "#1f2933"
ARROW = "#4b5563"

def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

def svg_start(title):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<title>{esc(title)}</title><defs>
<marker id="arrow" markerWidth="11" markerHeight="11" refX="9" refY="5.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L11,5.5 L0,11 z" fill="{ARROW}"/></marker>
</defs><rect width="100%" height="100%" fill="white"/>
'''

def box(x, y, w, h, lines, fill=BLUE, size=27):
    y0 = y + h/2 - (len(lines)-1)*size*0.58
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{fill}" stroke="{BORDER}" stroke-width="3"/>']
    for i, line in enumerate(lines):
        yy = y0 + i*size*1.16
        out.append(f'<text x="{x+w/2}" y="{yy}" text-anchor="middle" dominant-baseline="middle" font-family="{FONT}" font-size="{size}px" fill="{INK}">{esc(line)}</text>')
    return "\n".join(out)

def rich_box(x, y, w, h, lines, fill=BLUE, size=27):
    """Box whose lines contain trusted SVG tspan markup for subscripts."""
    y0 = y + h/2 - (len(lines)-1)*size*0.58
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{fill}" stroke="{BORDER}" stroke-width="3"/>']
    for i, line in enumerate(lines):
        yy = y0 + i*size*1.16
        out.append(f'<text x="{x+w/2}" y="{yy}" text-anchor="middle" dominant-baseline="middle" font-family="{FONT}" font-size="{size}px" fill="{INK}">{line}</text>')
    return "\n".join(out)

def arrow(x1, y1, x2, y2):
    return f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{ARROW}" stroke-width="3" marker-end="url(#arrow)"/>'

def write_render(name, svg):
    svg_path = OUT / f"{name}.svg"
    svg_path.write_text(svg, encoding="utf-8")
    subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(OUT/f"{name}.pdf"), str(svg_path)], check=True)
    subprocess.run(["rsvg-convert", "-f", "png", "-d", "600", "-p", "600", "-o", str(OUT/f"{name}.png"), str(svg_path)], check=True)

def fig2():
    s = svg_start("DoLa workflow implemented in the active decoder")
    s += box(35, 205, 225, 105, ["Hidden states", "blocks 0–21"])
    s += rich_box(365, 65, 270, 105, ["Mature logits", 'W h<tspan baseline-shift="sub" font-size="70%">21</tspan>'])
    s += rich_box(365, 345, 270, 105, ["Candidate logits", 'W<tspan baseline-shift="sub" font-size="70%">norm</tspan>(h<tspan baseline-shift="sub" font-size="70%">j</tspan>)'])
    s += box(730, 205, 350, 105, ["JSD(mature, j)", "j ∈ {0, 4, 8, 12, 16, 20}"], GREEN, 24)
    s += box(1160, 205, 230, 105, ["j* = argmax", "JSD"], GREEN)
    s += rich_box(1440, 205, 160, 105, ['ℓ = ℓ<tspan baseline-shift="sub" font-size="70%">m</tspan> − α ℓ<tspan baseline-shift="sub" font-size="70%">p</tspan>', "softmax / decode"], YELLOW, 20)
    s += arrow(260, 257, 365, 117) + arrow(260, 257, 365, 397)
    s += arrow(635, 117, 730, 245) + arrow(635, 397, 730, 270)
    s += arrow(1080, 257, 1160, 257) + arrow(1390, 257, 1440, 257)
    s += '</svg>'
    write_render("Fig_2_DoLa_Workflow", s)

def fig3():
    s = svg_start("Active C4-trained epinet")
    s += rich_box(35, 205, 235, 105, ['[h<tspan baseline-shift="sub" font-size="70%">m</tspan> ; h<tspan baseline-shift="sub" font-size="70%">p</tspan>]', "4096-D"])
    s += box(355, 205, 230, 105, ["MLP", "4096 → 512"])
    s += box(670, 205, 230, 105, ["MLP", "512 → 512"])
    s += box(985, 205, 270, 105, ["MLP", "512 → 2048 × 6"])
    s += rich_box(1370, 95, 190, 100, ['f<tspan baseline-shift="sub" font-size="70%">prior</tspan>', "(frozen)"], GRAY)
    s += rich_box(1370, 335, 190, 100, ['f<tspan baseline-shift="sub" font-size="70%">learn</tspan>', "(trainable)"], YELLOW)
    s += arrow(270, 257, 355, 257) + arrow(585, 257, 670, 257) + arrow(900, 257, 985, 257)
    s += arrow(1255, 257, 1370, 145) + arrow(1255, 257, 1370, 385)
    s += '<text x="800" y="485" text-anchor="middle" font-family="DejaVu Serif, Georgia, serif" font-size="22px" fill="#7a2d22">Each branch contracts the 6-D epistemic index z; outputs are added before LM-head projection.</text>'
    s += '</svg>'
    write_render("Fig_3_Active_Epinet", s)

def fig1():
    s = svg_start("Active repository pipeline")
    labels = [(35,205,190,105,["Question /", "context"],BLUE),(300,205,220,105,["TinyLlama", "1.1B backbone"],BLUE),(600,205,250,105,["Mature + premature", "hidden states"],BLUE),(960,135,205,105,["DoLa", "JSD + contrast"],GREEN),(960,295,205,105,["ENN", "uncertainty"],YELLOW),(1270,205,260,105,["Answer score /", "accept or abstain"],YELLOW)]
    for x,y,w,h,t,c in labels: s += box(x,y,w,h,t,c)
    s += arrow(225,257,300,257)+arrow(520,257,600,257)+arrow(850,257,960,187)+arrow(850,257,960,347)+arrow(1165,187,1270,245)+arrow(1165,347,1270,300)
    s += '<text x="800" y="485" text-anchor="middle" font-family="DejaVu Serif, Georgia, serif" font-size="20px" fill="#7a2d22">Archived answer-correctness ENN benchmark is reported separately from the active checkout.</text></svg>'
    write_render("Fig_1_Architecture", s)

def fig4():
    s = svg_start("Candidate generation and self-consistency aggregation")
    s += box(35,205,180,105,["Question"],BLUE)
    for y,n in [(75,"DoLa sample 1"),(205,"DoLa sample 2"),(335,"DoLa sample N")]: s += box(340,y,220,85,[n],GREEN)
    s += box(720,205,250,105,["Normalize /", "cluster answers"],BLUE)+box(1100,205,240,105,["Plurality or", "weighted vote"],YELLOW)+box(1460,205,110,105,["Answer"],YELLOW)
    s += arrow(215,257,340,117)+arrow(215,257,340,247)+arrow(215,257,340,377)+arrow(560,117,720,245)+arrow(560,247,720,257)+arrow(560,377,720,270)+arrow(970,257,1100,257)+arrow(1340,257,1460,257)+"</svg>"
    write_render("Fig_4_Self_Consistency", s)

def fig5():
    s = svg_start("Hallucination detection pipeline")
    xs=[35,300,585,870,1120,1410]; ts=[["Generate"],["Token", "uncertainty"],["Answer", "features"],["ENN", "score"],["Threshold", "selected on val"],["Accept /", "reject"]]; fills=[BLUE,BLUE,BLUE,YELLOW,GREEN,YELLOW]
    widths=[190,210,210,180,230,160]
    for x,t,f,w in zip(xs,ts,fills,widths): s += box(x,205,w,105,t,f,24)
    for i in range(len(xs)-1): s += arrow(xs[i]+widths[i],257,xs[i+1],257)
    s += '<text x="800" y="485" text-anchor="middle" font-family="DejaVu Serif, Georgia, serif" font-size="22px" fill="#7a2d22">Detection estimates reliability; mitigation changes decoding or aggregates candidates.</text></svg>'
    write_render("Fig_5_Detection_Pipeline", s)

if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
