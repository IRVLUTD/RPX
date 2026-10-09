#!/usr/bin/env python3
"""Render the RPX hero image (same scene, four views; per-phase quality; Phi-Jmin map).

Two variants: ``full`` (README/PyPI banner) and ``panel`` (docs home figure).

Numbers come from the paper's Tables III-VI and Fig. 4 (hero_data.json); the photos
are the scene012 guide assets. Run with Playwright + Chromium installed:
    python build_hero.py --data hero_data.json --assets <guides/assets> --out hero.png
"""
import argparse, base64, json, statistics
from pathlib import Path

TASKS = {  # key: (label, colour)
    "T1": ("Image depth", "#4F46E5"), "T2": ("Video depth", "#8B8FE8"), "T3": ("Tracking", "#0F9D58"),
    "T4": ("Camera pose", "#5BC48A"), "T5": ("VQA", "#C81E3A"), "T6": ("In-context VQA", "#EC0899"),
}
PHASES = [("clutter.jpg", "BEFORE", "Clutter"), ("interaction.jpg", "DURING", "Interaction"),
          ("clean.jpg", "AFTER", "Clean"), ("ego.jpg", "DURING", "Ego view")]
KEYS = ["clu", "int", "cln", "ego"]


def img64(p):
    return "data:image/jpeg;base64," + base64.b64encode(Path(p).read_bytes()).decode()


def scatter_svg(points, w=380, h=262):
    pad_l, pad_b, pad_t, pad_r = 52, 38, 26, 12
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    X = lambda v: pad_l + v * pw
    Y = lambda v: pad_t + (1 - v) * ph
    s = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">']
    for t in [0, .25, .5, .75, 1]:
        s.append(f'<line x1="{X(t)}" y1="{Y(0)}" x2="{X(t)}" y2="{Y(1)}" class="grid"/>'
                 f'<line x1="{X(0)}" y1="{Y(t)}" x2="{X(1)}" y2="{Y(t)}" class="grid"/>'
                 f'<text x="{X(t)}" y="{Y(0)+17}" class="tick" text-anchor="middle">{t:g}</text>'
                 f'<text x="{X(0)-8}" y="{Y(t)+4}" class="tick" text-anchor="end">{t:g}</text>')
    gx, gy = X(.75), Y(1)
    s.append(f'<rect x="{gx}" y="{gy}" width="{X(1)-gx}" height="{Y(.86)-gy}" class="gold"/>')
    s.append(f'<text x="{X(1)}" y="{gy-8}" class="goldlbl" text-anchor="end">golden zone: 𝒥min ≥ 0.75, Φ ≥ 0.86</text>'
             f'<text x="{(gx+X(1))/2}" y="{(gy+Y(.86))/2+6}" class="goldnum" text-anchor="middle">0 / 65</text>')
    for p in sorted(points, key=lambda p: p["task"]):
        s.append(f'<circle cx="{X(p["jmin"]):.1f}" cy="{Y(p["phi"]):.1f}" r="5.2" fill="{TASKS[p["task"]][1]}" class="pt"/>')
    s.append(f'<text x="{X(.5)}" y="{h-4}" class="axis" text-anchor="middle">worst-phase quality  𝒥<tspan baseline-shift="sub" font-size="11">min</tspan>  →</text>'
             f'<text transform="translate(12,{Y(.5)}) rotate(-90)" class="axis" text-anchor="middle">phase robustness  Φ  →</text>')
    s.append("</svg>")
    return "".join(s)


def bars(vals):
    """vals: list of (task, value or None) for one phase."""
    out = []
    for t, v in vals:
        lab, col = TASKS[t]
        if v is None:
            out.append(f'<div class="bar"><span class="bl">{lab}</span><span class="track na"></span><span class="bv na">n/a</span></div>')
        else:
            out.append(f'<div class="bar"><span class="bl">{lab}</span><span class="track"><i style="width:{v*100:.0f}%;background:{col}"></i></span><span class="bv">{v:.2f}</span></div>')
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--assets", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--html")
    ap.add_argument("--variant", choices=["full", "panel"], default="full",
                    help="panel: views, bars and map only (no title or install strip), for the docs site")
    a = ap.parse_args()
    H = 838
    D = json.loads(Path(a.data).read_text())
    mean = {t: {k: statistics.mean(r[k] for r in D["rows"][t]) for k in KEYS if k in D["rows"][t][0]} for t in ("T2", "T3", "T5")}
    cards = []
    for i, (f, when, name) in enumerate(PHASES):
        k = KEYS[i]
        vals = [(t, mean[t].get(k)) for t in ("T2", "T3", "T5")]
        hot = " hot" if k == "int" else ""
        cards.append(f'<figure class="ph{hot}"><img src="{img64(Path(a.assets)/f)}"><figcaption><b>{when}</b> {name}</figcaption>'
                     f'<div class="bars">{bars(vals)}</div></figure>')
    chips = "".join(f'<span class="chip"><i style="background:{c}"></i>{l}</span>' for l, c in TASKS.values())
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0}} body{{width:1200px;height:{H}px;font-family:Inter,system-ui,sans-serif;color:#0f172a;position:relative;overflow:hidden;
 background:radial-gradient(900px 420px at 10% -10%,#eef2ff 0%,transparent 60%),radial-gradient(700px 420px at 105% 105%,#fdf2f8 0%,transparent 60%),#fff}}
.wrap{{padding:30px 36px 0}} .top{{display:flex;align-items:center;justify-content:space-between}}
.head{{display:flex;gap:20px;align-items:center}} .logo{{font-weight:800;font-size:58px;letter-spacing:-2px;line-height:1}}
.logo .r{{color:#4F46E5}}.logo .p{{color:#DB2777}}.logo .x{{color:#C2410C}}
h1{{font-size:38px;font-weight:800;letter-spacing:-1px;line-height:1.05}} .sub{{font-size:17px;color:#475569;margin-top:6px}}
.stats{{display:flex;gap:22px;text-align:right}} .stats b{{display:block;font-size:28px;font-weight:800;letter-spacing:-.5px;line-height:1.1}}
.stats span{{font-size:12px;color:#64748b;text-transform:uppercase;letter-spacing:.06em}}
.row{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:22px}}
.ph{{background:#fff;border:1px solid #e2e8f0;border-radius:14px;overflow:hidden;box-shadow:0 6px 20px -12px rgba(15,23,42,.25)}}
.ph img{{width:100%;aspect-ratio:4/3;object-fit:cover;display:block}} .ph figcaption{{padding:10px 12px 2px;font-size:16px;color:#334155}}
.ph figcaption b{{font-size:12px;letter-spacing:.12em;color:#64748b;margin-right:6px}}
.ph.hot{{border-color:#fda4af;box-shadow:0 0 0 3px #ffe4e6,0 8px 24px -12px rgba(200,30,58,.45)}} .ph.hot figcaption{{color:#be123c;font-weight:700}}
.bars{{padding:6px 12px 12px}} .bar{{display:grid;grid-template-columns:92px 1fr 40px;align-items:center;gap:8px;height:23px}}
.bl{{font-size:13.5px;color:#475569}} .track{{height:9px;border-radius:5px;background:#f1f5f9;overflow:hidden}} .track i{{display:block;height:100%;border-radius:5px}}
.track.na{{background:repeating-linear-gradient(45deg,#f1f5f9 0 4px,#fff 4px 8px)}} .bv{{font:500 13.5px 'JetBrains Mono',monospace;text-align:right}} .bv.na{{color:#94a3b8}}
.lower{{display:flex;gap:18px;margin-top:18px;align-items:stretch}}
.story{{flex:1;display:flex;flex-direction:column;justify-content:center;gap:14px}}
.pill{{align-self:flex-start;background:#ffe4e6;color:#be123c;font-weight:800;border-radius:999px;padding:6px 14px;font-size:15px}}
.claim{{font-size:21px;line-height:1.35;color:#1e293b;font-weight:500}} .claim b{{color:#be123c}}
.note{{font-size:13.5px;color:#64748b}}
.chips{{display:flex;flex-wrap:wrap;gap:7px}} .chip{{font-size:13px;color:#334155;background:#f8fafc;border:1px solid #e2e8f0;border-radius:999px;padding:5px 11px}}
.chip i{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}}
.right{{width:400px;background:#fff;border:1px solid #e2e8f0;border-radius:16px;padding:12px 10px 4px;box-shadow:0 6px 20px -12px rgba(15,23,42,.25)}}
.right h2{{font-size:16.5px;font-weight:800;margin:0 0 0 8px}}
.grid{{stroke:#eef2f7;stroke-width:1}} .tick{{font:12px Inter;fill:#94a3b8}} .axis{{font:600 13px Inter;fill:#475569}}
.gold{{fill:#fef9c3;stroke:#d4a106;stroke-width:1.5;stroke-dasharray:5 4}} .goldlbl{{font:700 12px Inter;fill:#a16207}}
.goldnum{{font:800 18px Inter;fill:#a16207}} .pt{{stroke:#fff;stroke-width:1.2;opacity:.93}}
.foot{{position:absolute;left:0;right:0;bottom:0;height:78px;background:#0f172a;color:#e2e8f0;display:flex;align-items:center;gap:16px;padding:0 36px}}
.step{{display:flex;align-items:center;gap:10px}} .n{{width:28px;height:28px;border-radius:50%;background:#1e293b;color:#a5b4fc;display:grid;place-items:center;font-weight:700;font-size:14px}}
code{{font:500 16px 'JetBrains Mono',monospace;color:#fff}} .lab{{font-size:12.5px;color:#94a3b8;display:block}} .arrow{{color:#475569;font-size:20px}}
.res{{font-size:16px;font-weight:700;color:#fff}} .res sub{{font-size:10px}}
.panel .top,.panel .foot{{display:none}} .panel .row{{margin-top:0}} .panel .wrap{{padding:18px 18px 18px}}
</style></head><body class="{a.variant}"><div class="wrap">
<div class="top"><div class="head"><div class="logo"><span class="r">R</span><span class="p">P</span><span class="x">X</span></div>
<div><h1>Same scene. Different story.</h1><div class="sub">Test perception models before, during and after manipulation.</div></div></div>
<div class="stats"><div><b>100</b><span>real scenes</span></div><div><b>6</b><span>tasks</span></div><div><b>65</b><span>evaluations</span></div></div></div>
<div class="row">{''.join(cards)}</div>
<div class="lower"><div class="story"><span class="pill">Hands enter → quality drops</span>
<div class="claim">Interaction scores lowest for <b>all 11 trackers</b> and <b>34 of 55</b> evaluations, and <b>no model</b> is both accurate and robust.</div>
<div class="note">Bars: mean quality 𝒥 over all evaluated models, per view. Map: every evaluation in the paper.</div>
<div class="chips">{chips}</div></div>
<div class="right"><h2>0 of 65 reach the golden zone</h2>{scatter_svg(D['points'])}</div></div></div>
<div class="foot"><div class="step"><span class="n">1</span><div><span class="lab">install</span><code>pip install rpx-benchmark</code></div></div><span class="arrow">→</span>
<div class="step"><span class="n">2</span><div><span class="lab">wrap your model or API</span><code>rpx.run_monocular_depth(model)</code></div></div><span class="arrow">→</span>
<div class="step"><span class="n">3</span><div><span class="lab">get the paper's diagnosis</span><span class="res">Φ · 𝒥<sub>min</sub> · per phase & scene</span></div></div></div>
</body></html>"""
    if a.html: Path(a.html).write_text(html)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1200, "height": H}, device_scale_factor=2)
        pg.set_content(html, wait_until="networkidle"); pg.wait_for_timeout(400)
        if a.variant == "panel":
            pg.locator(".wrap").screenshot(path=a.out)
        else:
            pg.screenshot(path=a.out, full_page=False)
        b.close()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
