"""Generate thesis figures as dependency-free SVG, driven by the actual demos.

Run: PYTHONPATH=../../src python gen_figures.py   (writes fig1..fig6 .svg here)
Figures:
  fig1  demo1 — RCC-8 collapses face/edge/point EC; the witness keeps them apart
  fig2  demo5 — the threshold is where the decision rights live (field -> rooms)
  fig3  demo7 — the geometry<->relation round-trip + the A-B(face)/B-C(edge) config
  fig4  architecture — carriers / connectors / single operational calculus
  fig5  fail-closed realizability ladder
  fig6  Deixis for Grasshopper — insertable relational control / re-synthesis stage
"""
from __future__ import annotations
import os, sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

# ---- minimal design system (muted, accessible, prints well) ----
INK = "#1b1f24"; MUTE = "#6b7480"; LINE = "#c7ccd2"; BG = "#ffffff"
A_FILL = "#dfe7f3"; A_STROKE = "#5b7fb4"      # region A / primary
B_FILL = "#f3e6de"; B_STROKE = "#b47f5b"      # region B / secondary
C_FILL = "#e3efe4"; C_STROKE = "#5ba36a"      # region C / tertiary
OK = "#2f8f5b"; BAD = "#c0472f"; WARN = "#c08a2f"
FONT = 'font-family="Helvetica,Arial,sans-serif"'


def svg(w, h, body, title=""):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" role="img" aria-label="{title}">'
            f'<rect width="{w}" height="{h}" fill="{BG}"/>{body}</svg>')

def T(x, y, s, size=13, col=INK, anchor="start", weight="400", style=""):
    return (f'<text x="{x}" y="{y}" {FONT} font-size="{size}" fill="{col}" '
            f'text-anchor="{anchor}" font-weight="{weight}" style="{style}">{s}</text>')

def rect(x, y, w, h, fill, stroke, sw=1.6, rx=2, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'

def line(x1, y1, x2, y2, col=LINE, sw=1.4, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="{sw}"{d}/>'

def arrow(x1, y1, x2, y2, col=MUTE, sw=1.6):
    return (f'<defs><marker id="ah{abs(hash((x1,y1,x2,y2)))%9999}" markerWidth="8" markerHeight="8" '
            f'refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="{col}"/></marker></defs>'
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="{sw}" '
            f'marker-end="url(#ah{abs(hash((x1,y1,x2,y2)))%9999})"/>')

def write(name, content):
    with open(os.path.join(HERE, name), "w") as f:
        f.write(content)
    print("wrote", name)


# ------------------------------------------------------------------ fig1
def fig1():
    from deixis.demos.demo1_contact_dimension import build_specs, run as _run
    specs = build_specs()  # (face, edge, point) specs
    labels = ["FACE (dim 2)", "EDGE (dim 1)", "POINT (dim 0)"]
    cols = [(A_FILL, A_STROKE), (B_FILL, B_STROKE), (C_FILL, C_STROKE)]
    W, H = 780, 320; body = [T(24, 34, "図1  同じ RCC-8 関係 EC ・ witness が面/辺/点を分ける", 16, INK, weight="600")]
    body.append(T(24, 56, "3 仕様とも RCC-8 マスクは EC（同一）。実現形は接触次元だけが違う。", 12, MUTE))
    # draw 3 realized configurations as 2 boxes each (project to 2D)
    for i, sp in enumerate(specs):
        real = _run(sp)  # demo1's default domain/bounds (exact)
        ox = 40 + i*250; oy = 90; sc = 26
        body.append(rect(ox-6, oy-6, 224, 190, "#fbfcfd", LINE, 1))
        body.append(T(ox+106, oy+14, labels[i], 12, INK, "middle", "600"))
        for j, box in enumerate(real.boxes[:2]):
            lo = [float(c) for c in box.lo]; hi = [float(c) for c in box.hi]
            x = ox + 30 + lo[0]*sc; y = oy + 150 - hi[1]*sc
            w = (hi[0]-lo[0])*sc; h = (hi[1]-lo[1])*sc
            f, s = cols[j]
            body.append(rect(x, y, max(w, 3), max(h, 3), f, s, 1.8))
            body.append(T(x+w/2, y+h/2+4, box.region_id, 12, INK, "middle", "600"))
        body.append(T(ox+106, oy+178, "RCC-8: EC", 11, MUTE, "middle"))
    write("fig1_contact_dimension.svg", svg(W, H, "".join(body), "demo1 contact dimension"))

# ------------------------------------------------------------------ fig2
def fig2():
    from deixis.field.scalar import ScalarField, threshold_ground
    F = Fraction
    vals = [F(1,5), F(1,2), F(3,5), F(9,10), F(1,5)]   # 'warm' field over 5 cells
    fld = ScalarField((5,), tuple(vals), (F(0),), F(1))
    taus = [F(1,2), F(4,5)]
    W, H = 760, 300; body = [T(24, 34, "図2  閾値は決定権が宿る場所（同じ場・違う τ で違う部屋）", 16, INK, weight="600")]
    body.append(T(24, 56, "連続な帰属場を未 ground のまま保持＝部屋に区切れない空間。閾値化＝GroundingDecision（場→部屋）。", 12, MUTE))
    ox, oy, cw, ch = 60, 210, 120, 120
    # field as bars
    for i, v in enumerate(vals):
        x = ox + i*cw; bh = float(v)*ch
        body.append(rect(x, oy-bh, cw-8, bh, A_FILL, A_STROKE, 1.4))
        body.append(T(x+(cw-8)/2, oy+18, f"c{i}", 11, MUTE, "middle"))
        body.append(T(x+(cw-8)/2, oy-bh-6, f"{v.numerator}/{v.denominator}", 10, MUTE, "middle"))
    # threshold lines + resulting rooms
    for k, tau in enumerate(taus):
        ty = oy - float(tau)*ch; col = [B_STROKE, C_STROKE][k]
        body.append(line(ox-10, ty, ox+5*cw-8, ty, col, 1.8, "5 3"))
        cells, gd = threshold_ground(fld, tau, f"room_{k}")
        got = sorted(c[0] for c in cells)
        body.append(T(ox+5*cw, ty+4, f"τ={tau.numerator}/{tau.denominator} → 部屋= cells {got}", 12, col, "start", "600"))
    body.append(T(60, 262, "各 τ は台帳に残る型付き GroundingDecision（誰が/なぜ）。argmax で全格子を分割すれば「部屋分割」＝特殊な ground。", 11, MUTE))
    write("fig2_threshold_grounding.svg", svg(W, H, "".join(body), "demo5 threshold grounding"))

# ------------------------------------------------------------------ fig3
def fig3():
    steps = ["関係仕様\n(RCC-8+witness)", "段階A\n記号整合", "grounding\n契約 I/G/P",
             "段階B 幾何\n(Z3 直交box)", "逆検証\n(幾何→関係)", "解族比較\n(不変核保存)"]
    W, H = 900, 300; body = [T(24, 34, "図3  一往復：関係→幾何→逆検証（demo7）", 16, INK, weight="600")]
    body.append(T(24, 56, "同一未 ground 仕様に grounding を差し替えると、不変核（A-B 面接触・B-C 辺接触）を保ったまま複数の形態族が出る。", 12, MUTE))
    bx, by, bw, bh, gap = 30, 90, 118, 56, 22
    for i, s in enumerate(steps):
        x = bx + i*(bw+gap)
        body.append(rect(x, by, bw, bh, "#f6f8fa", A_STROKE, 1.4))
        for li, ln in enumerate(s.split("\n")):
            body.append(T(x+bw/2, by+24+li*15, ln, 11, INK, "middle", "600" if li==0 else "400"))
        if i < len(steps)-1:
            body.append(arrow(x+bw, by+bh/2, x+bw+gap, by+bh/2))
    # two solution boxes below
    body.append(T(30, 190, "解族（grounded 部だけ差）:", 12, INK, "start", "600"))
    for k, (rel, col) in enumerate([("A-C = DC", B_STROKE), ("A-C = PO", C_STROKE)]):
        x = 40 + k*220; y = 205
        body.append(rect(x, y, 60, 40, A_FILL, A_STROKE)); body.append(T(x+30, y+24, "A", 12, INK, "middle", "600"))
        body.append(rect(x+58, y, 40, 40, B_FILL, B_STROKE)); body.append(T(x+78, y+24, "B", 11, INK, "middle", "600"))
        body.append(rect(x+120, y-20, 40, 40, C_FILL, C_STROKE)); body.append(T(x+140, y+4, "C", 11, INK, "middle", "600"))
        body.append(T(x+80, y+62, "s%d: %s (委譲部)" % (k, rel), 11, col, "middle", "600"))
        body.append(T(x+80, y+76, "A-B 面 / B-C 辺 は不変核として保存", 10, MUTE, "middle"))
    write("fig3_roundtrip.svg", svg(W, H, "".join(body), "demo7 round trip"))

# ------------------------------------------------------------------ fig4
def fig4():
    W, H = 820, 360; body = [T(24, 34, "図4  Witnessed Multi-Representation Spatial IR の構成", 16, INK, weight="600")]
    body.append(T(24, 54, "単一の witnessed cell complex K ＋単一 identity/provenance ストアを共有し、型の異なる carrier を明示 adapter で接続。", 12, MUTE))
    carriers = [("Geometry (Brep/box, Fraction)", A_FILL, A_STROKE),
                ("Relation-view: RCC-8 + IncidenceWitness + status", B_FILL, B_STROKE),
                ("Field-view: DEC 0/1/2-cochain, Scalar/Direction", C_FILL, C_STROKE),
                ("Overlay-view: signature poset", "#efe7f3", "#8a5bb4")]
    for i, (t, f, s) in enumerate(carriers):
        y = 78 + i*44; body.append(rect(40, y, 470, 34, f, s, 1.4)); body.append(T(52, y+22, t, 12, INK))
    # connectors
    body.append(rect(560, 78, 230, 166, "#fbfcfd", LINE, 1))
    body.append(T(675, 98, "connectors (lens 規律)", 12, INK, "middle", "600"))
    for i, c in enumerate(["Lift / Solve  (幾何↔関係)", "level-set 閾値 = grounding", "Poincaré ⋆ (多様体ガード)", "overlay signature"]):
        body.append(T(575, 122+i*26, "• " + c, 11, INK))
    # calculus bar
    body.append(rect(40, 268, 750, 40, "#f2f5f8", A_STROKE, 1.4))
    body.append(T(415, 285, "単一運用 calculus: status(invariant|grounded|delegated) / ground・unground / verify(exact+逆) / preserve(D(E(x))≅_O x)", 12, INK, "middle"))
    body.append(T(415, 300, "= 関係制約族と保存制約族の両方へ一様に効く（決定権配分の軸が「何を固定/幾何か関係か/部屋か連続場か」を統べる）", 11, MUTE, "middle"))
    write("fig4_architecture.svg", svg(W, H, "".join(body), "architecture"))

# ------------------------------------------------------------------ fig5
def fig5():
    rungs = [("abstract", "型として整合（未検査）", LINE),
             ("rcc_consistent", "RCC-8 記号整合（tractable fragment 内で健全かつ完全）", A_STROKE),
             ("realized_in_selected_domain_D", "選択ドメイン D（直交box）で Z3 実現＋逆検証", OK),
             ("fabrication_valid", "建築的許容述語（開口幅/階高/接地…）を満たす", C_STROKE)]
    W, H = 760, 300; body = [T(24, 34, "図5  fail-closed な realizability ラダー", 16, INK, weight="600")]
    body.append(T(24, 54, "下流出力（BOM/施工図/採用）は到達した最上位でのみ許可。unsat-in-D ≠ 不可能、unknown は不採用（fail-closed）。", 12, MUTE))
    for i, (name, desc, col) in enumerate(rungs):
        y = 240 - i*44
        body.append(rect(40, y, 320, 34, "#f6f8fa", col, 2)); body.append(T(52, y+22, name, 12, INK, "start", "600"))
        body.append(T(375, y+22, desc, 11, MUTE))
        if i < len(rungs)-1:
            body.append(arrow(200, y, 200, y-10, MUTE))
    write("fig5_ladder.svg", svg(W, H, "".join(body), "fail-closed ladder"))

# ------------------------------------------------------------------ fig6
def fig6():
    W, H = 900, 260; body = [T(24, 34, "図6  Deixis for Grasshopper：挿入可能な関係制御・再合成ステージ", 16, INK, weight="600")]
    body.append(T(24, 54, "既存プラグイン（Kangaroo/Wasp/手動）の幾何を Lift で関係層へ持ち上げ、関係/決定を編集し、Solve で幾何へ再合成して下流へ。", 12, MUTE))
    nodes = [("上流幾何\n(任意plugin)", A_FILL, A_STROKE), ("Lift\n(観測)", B_FILL, B_STROKE),
             ("Relate/Witness\nGround", "#efe7f3", "#8a5bb4"), ("Solve\n(再合成)", C_FILL, C_STROKE),
             ("下流\n(日照/構造…)", A_FILL, A_STROKE)]
    bx, by, bw, bh, gap = 30, 100, 140, 60, 40
    for i, (t, f, s) in enumerate(nodes):
        x = bx + i*(bw+gap)
        body.append(rect(x, by, bw, bh, f, s, 1.6))
        for li, ln in enumerate(t.split("\n")):
            body.append(T(x+bw/2, by+26+li*16, ln, 12, INK, "middle", "600" if li==0 else "400"))
        if i < len(nodes)-1:
            body.append(arrow(x+bw, by+bh/2, x+bw+gap, by+bh/2))
    # verify feedback
    vx = bx + 3*(bw+gap)
    body.append(rect(vx, by+110, bw, 36, "#f6f8fa", OK, 1.4)); body.append(T(vx+bw/2, by+133, "Verify (逆照合)", 12, OK, "middle", "600"))
    body.append(arrow(vx+bw/2, by+bh, vx+bw/2, by+110))
    body.append(T(30, 235, "spec は JSON 文字列でワイヤを流れる（level-2）。Solve は relation-preserving resynthesis（元トポロジー/部材は保たない）。", 11, MUTE))
    write("fig6_gh.svg", svg(W, H, "".join(body), "Deixis for Grasshopper"))


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print("done: 6 figures")
