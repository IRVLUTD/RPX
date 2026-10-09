#!/usr/bin/env python3
"""Render the RPX hero image (same scene, four views; per-phase quality; Phi-Jmin map).

Numbers come from the paper's Tables III-VI and Fig. 4 (hero_data.json); the photos
are the scene012 guide assets. Variants: ``full`` (README/PyPI banner) and ``panel``
(docs home figure, no title or install strip). Themes: ``light`` and ``dark``.
The image has rounded corners on a transparent background so it sits as a card on
any page colour. Needs Playwright with Chromium:

    python build_hero.py --data hero_data.json --assets <guides/assets> \
        --variant full --theme light --out rpx-hero-light.webp
"""
import argparse, base64, json, statistics
from pathlib import Path

TASKS = {  # key: (label, light colour, dark colour)
    "T1": ("Image depth", "#4F46E5", "#818CF8"), "T2": ("Video depth", "#8B8FE8", "#A5B4FC"),
    "T3": ("Tracking", "#0F9D58", "#34D399"), "T4": ("Camera pose", "#5BC48A", "#86EFAC"),
    "T5": ("VQA", "#C81E3A", "#FB7185"), "T6": ("In-context VQA", "#EC0899", "#F472B6"),
}
PHASES = [("clutter.jpg", "BEFORE", "Clutter"), ("interaction.jpg", "DURING", "Interaction"),
          ("clean.jpg", "AFTER", "Clean"), ("ego.jpg", "DURING", "Ego view")]
KEYS = ["clu", "int", "cln", "ego"]
THEME = {
    "light": dict(r="#4F46E5", p="#DB2777", x="#C2410C", bg="#ffffff", glow1="#eef2ff", glow2="#fdf2f8", ink="#0f172a", body="#334155", muted="#64748b",
                  faint="#94a3b8", card="#ffffff", line="#e2e8f0", track="#eef2f7", grid="#eef2f7",
                  hot="#fda4af", hotglow="#ffe4e6", hotink="#be123c", pillbg="#ffe4e6", pillink="#be123c",
                  goldfill="#fef9c3", goldline="#ca8a04", goldink="#a16207", foot="#0f172a", footink="#e2e8f0",
                  footmuted="#94a3b8", num="#1e293b", numink="#a5b4fc", shadow="rgba(15,23,42,.18)", ptline="#ffffff"),
    "dark": dict(r="#818CF8", p="#F472B6", x="#FB923C", bg="#0b1120", glow1="rgba(79,70,229,.28)", glow2="rgba(219,39,119,.20)", ink="#f1f5f9", body="#cbd5e1",
                 muted="#94a3b8", faint="#64748b", card="#111a2e", line="#1e293b", track="#1e293b", grid="#1a2438",
                 hot="#e11d48", hotglow="rgba(225,29,72,.22)", hotink="#fb7185", pillbg="rgba(225,29,72,.18)",
                 pillink="#fda4af", goldfill="rgba(250,204,21,.10)", goldline="#facc15", goldink="#fde68a",
                 foot="#020617", footink="#e2e8f0", footmuted="#94a3b8", num="#1e293b", numink="#a5b4fc",
                 shadow="rgba(0,0,0,.45)", ptline="#0b1120"),
}


def img64(p):
    return "data:image/jpeg;base64," + base64.b64encode(Path(p).read_bytes()).decode()


def colour(task, theme):
    return TASKS[task][1 if theme == "light" else 2]


def scatter_svg(points, theme, w=378, h=282):
    pad_l, pad_b, pad_t, pad_r = 56, 44, 30, 12
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    X = lambda v: pad_l + v * pw
    Y = lambda v: pad_t + (1 - v) * ph
    s = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">']
    for t in [0, .25, .5, .75, 1]:
        s.append(f'<line x1="{X(t)}" y1="{Y(0)}" x2="{X(t)}" y2="{Y(1)}" class="grid"/>'
                 f'<line x1="{X(0)}" y1="{Y(t)}" x2="{X(1)}" y2="{Y(t)}" class="grid"/>'
                 f'<text x="{X(t)}" y="{Y(0)+19}" class="tick" text-anchor="middle">{t:g}</text>'
                 f'<text x="{X(0)-9}" y="{Y(t)+5}" class="tick" text-anchor="end">{t:g}</text>')
    gx, gy = X(.75), Y(1)
    s.append(f'<rect x="{gx}" y="{gy}" width="{X(1)-gx}" height="{Y(.86)-gy}" rx="3" class="gold"/>')
    s.append(f'<text x="{X(1)}" y="{gy-10}" class="goldlbl" text-anchor="end">golden zone</text>'
             f'<text x="{(gx+X(1))/2}" y="{(gy+Y(.86))/2+7}" class="goldnum" text-anchor="middle">0 / 65</text>')
    for p in sorted(points, key=lambda p: p["task"]):
        s.append(f'<circle cx="{X(p["jmin"]):.1f}" cy="{Y(p["phi"]):.1f}" r="5.6" fill="{colour(p["task"], theme)}" class="pt"/>')
    s.append(f'<text x="{X(.5)}" y="{h-3}" class="axis" text-anchor="middle">worst-phase quality 𝒥ₘᵢₙ →</text>'
             f'<text transform="translate(15,{Y(.5)}) rotate(-90)" class="axis" text-anchor="middle">phase robustness Φ →</text>')
    s.append("</svg>")
    return "".join(s)


def bars(vals, theme):
    out = []
    for t, v in vals:
        lab = TASKS[t][0]
        if v is None:
            out.append(f'<div class="bar"><span class="bl">{lab}</span><span class="track na"></span><span class="bv na">n/a</span></div>')
        else:
            out.append(f'<div class="bar"><span class="bl">{lab}</span><span class="track"><i style="width:{v*100:.0f}%;background:{colour(t, theme)}"></i></span><span class="bv">{v:.2f}</span></div>')
    return "".join(out)


def build_html(D, assets, variant, theme):
    c = THEME[theme]
    mean = {t: {k: statistics.mean(r[k] for r in D["rows"][t]) for k in KEYS if k in D["rows"][t][0]} for t in ("T2", "T3", "T5")}
    cards = []
    for i, (f, when, name) in enumerate(PHASES):
        k = KEYS[i]
        vals = [(t, mean[t].get(k)) for t in ("T2", "T3", "T5")]
        hot = " hot" if k == "int" else ""
        cards.append(f'<figure class="ph{hot}"><img src="{img64(Path(assets)/f)}"><figcaption><b>{when}</b>{name}</figcaption>'
                     f'<div class="bars">{bars(vals, theme)}</div></figure>')
    chips = "".join(f'<span class="chip"><i style="background:{colour(k, theme)}"></i>{v[0]}</span>' for k, v in TASKS.items())
    full = variant == "full"
    top = ("""<div class="top"><div class="head"><div class="logo"><span class="r">R</span><span class="p">P</span><span class="x">X</span></div>
<div><h1>Same scene. Different story.</h1><div class="sub">Test perception models before, during and after manipulation.</div></div></div>
<div class="stats"><div><b>100</b><span>real scenes</span></div><div><b>6</b><span>tasks</span></div><div><b>65</b><span>evaluations</span></div></div></div>""" if full else "")
    foot = ("""<div class="foot"><div class="step"><span class="n">1</span><div><span class="lab">install</span><code>pip install rpx-benchmark</code></div></div><span class="arrow">→</span>
<div class="step"><span class="n">2</span><div><span class="lab">wrap your model or API</span><code>rpx.run_monocular_depth(model)</code></div></div><span class="arrow">→</span>
<div class="step"><span class="n">3</span><div><span class="lab">get the paper's diagnosis</span><span class="res">Φ · 𝒥<sub>min</sub> · per phase</span></div></div></div>""" if full else "")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0}} html,body{{background:transparent}} body{{width:1200px;font-family:Inter,system-ui,sans-serif;color:{c['ink']};padding:0}}
.card{{position:relative;border-radius:28px;overflow:hidden;border:1px solid {c['line']};
 background:radial-gradient(820px 380px at 8% -12%,{c['glow1']} 0%,transparent 62%),radial-gradient(700px 420px at 104% 104%,{c['glow2']} 0%,transparent 60%),{c['bg']}}}
.wrap{{padding:{'32px 36px 26px' if full else '24px 24px 24px'}}}
.top{{display:flex;align-items:center;justify-content:space-between;margin-bottom:22px}}
.head{{display:flex;gap:20px;align-items:center}} .logo{{font-weight:800;font-size:64px;letter-spacing:-2.5px;line-height:1}}
.logo .r{{color:{c['r']}}}.logo .p{{color:{c['p']}}}.logo .x{{color:{c['x']}}}
h1{{font-size:42px;font-weight:800;letter-spacing:-1.2px;line-height:1.05}} .sub{{font-size:19px;color:{c['body']};margin-top:7px}}
.stats{{display:flex;gap:26px;text-align:right}} .stats b{{display:block;font-size:32px;font-weight:800;letter-spacing:-.6px;line-height:1.1}}
.stats span{{font-size:13.5px;color:{c['muted']};text-transform:uppercase;letter-spacing:.07em;font-weight:600}}
.row{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}
.ph{{background:{c['card']};border:1px solid {c['line']};border-radius:16px;overflow:hidden;box-shadow:0 10px 28px -16px {c['shadow']}}}
.ph img{{width:100%;aspect-ratio:4/3;object-fit:cover;display:block}}
.ph figcaption{{padding:11px 13px 3px;font-size:18px;font-weight:600;color:{c['ink']}}}
.ph figcaption b{{font-size:12.5px;letter-spacing:.13em;color:{c['muted']};margin-right:8px;font-weight:700}}
.ph.hot{{border-color:{c['hot']};box-shadow:0 0 0 3px {c['hotglow']},0 12px 30px -14px {c['shadow']}}} .ph.hot figcaption{{color:{c['hotink']}}}
.bars{{padding:6px 13px 13px}} .bar{{display:grid;grid-template-columns:100px 1fr 42px;align-items:center;gap:9px;height:27px}}
.bl{{font-size:15px;color:{c['body']}}} .track{{height:10px;border-radius:6px;background:{c['track']};overflow:hidden}} .track i{{display:block;height:100%;border-radius:6px}}
.track.na{{background:repeating-linear-gradient(45deg,{c['track']} 0 4px,transparent 4px 8px)}}
.bv{{font:600 15px 'JetBrains Mono',monospace;text-align:right;color:{c['ink']}}} .bv.na{{color:{c['faint']}}}
.lower{{display:flex;gap:18px;margin-top:18px;align-items:stretch}}
.story{{flex:1;display:flex;flex-direction:column;justify-content:center;gap:16px;padding:0 4px}}
.pill{{align-self:flex-start;background:{c['pillbg']};color:{c['pillink']};font-weight:800;border-radius:999px;padding:7px 16px;font-size:16px}}
.nw{{white-space:nowrap}} .claim{{font-size:26px;line-height:1.3;font-weight:600;letter-spacing:-.3px;color:{c['ink']}}} .claim b{{color:{c['hotink']};font-weight:800}}
.chips{{display:flex;flex-wrap:nowrap;gap:6px}} .chip{{white-space:nowrap;font-size:14px;color:{c['body']};background:{c['card']};border:1px solid {c['line']};border-radius:999px;padding:5px 10px}}
.chip i{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}}
.right{{width:404px;background:{c['card']};border:1px solid {c['line']};border-radius:16px;padding:14px 12px 6px;box-shadow:0 10px 28px -16px {c['shadow']}}}
.right h2{{font-size:18px;font-weight:800;margin:0 0 0 8px;color:{c['ink']}}}
.grid{{stroke:{c['grid']};stroke-width:1}} .tick{{font:500 13.5px Inter;fill:{c['faint']}}} .axis{{font:600 14.5px Inter;fill:{c['body']}}}
.gold{{fill:{c['goldfill']};stroke:{c['goldline']};stroke-width:1.6;stroke-dasharray:5 4}} .goldlbl{{font:700 13.5px Inter;fill:{c['goldink']}}}
.goldnum{{font:800 20px Inter;fill:{c['goldink']}}} .pt{{stroke:{c['ptline']};stroke-width:1.3;opacity:.95}}
.foot{{background:{c['foot']};color:{c['footink']};display:flex;align-items:center;gap:18px;padding:0 36px;height:86px}}
.step{{display:flex;align-items:center;gap:12px}} .n{{width:30px;height:30px;border-radius:50%;background:{c['num']};color:{c['numink']};display:grid;place-items:center;font-weight:800;font-size:15px}}
code{{font:600 18px 'JetBrains Mono',monospace;color:#fff}} .lab{{font-size:14px;color:{c['footmuted']};display:block;margin-bottom:1px}} .arrow{{color:#475569;font-size:22px}}
.res{{font-size:18px;font-weight:700;color:#fff}} .res sub{{font-size:11px}}
</style></head><body><div class="card"><div class="wrap">{top}
<div class="row">{''.join(cards)}</div>
<div class="lower"><div class="story"><span class="pill">Hands enter → quality drops</span>
<div class="claim">Interaction scores lowest for <b class="nw">all 11 trackers</b> and <b class="nw">34 of 55</b> evaluations, and <b class="nw">no model</b> is both accurate and robust.</div>
<div class="chips">{chips}</div></div>
<div class="right"><h2>0 of 65 reach the golden zone</h2>{scatter_svg(D['points'], theme)}</div></div></div>
{foot}</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--assets", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--html")
    ap.add_argument("--variant", choices=["full", "panel"], default="full")
    ap.add_argument("--theme", choices=["light", "dark"], default="light")
    ap.add_argument("--quality", type=int, default=90, help="WebP/JPEG quality")
    a = ap.parse_args()
    html = build_html(json.loads(Path(a.data).read_text()), a.assets, a.variant, a.theme)
    if a.html:
        Path(a.html).write_text(html)
    from playwright.sync_api import sync_playwright
    from PIL import Image
    tmp = Path(a.out).with_suffix(".tmp.png")
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1200, "height": 900}, device_scale_factor=2)
        pg.set_content(html, wait_until="networkidle"); pg.wait_for_timeout(500)
        pg.locator(".card").screenshot(path=str(tmp), omit_background=True); b.close()
    im = Image.open(tmp).convert("RGBA")
    out = Path(a.out)
    if out.suffix.lower() == ".webp":
        im.save(out, "WEBP", quality=a.quality, method=6)
    elif out.suffix.lower() in (".jpg", ".jpeg"):
        bg = Image.new("RGB", im.size, "white"); bg.paste(im, mask=im.split()[3]); bg.save(out, quality=a.quality, optimize=True, progressive=True)
    else:
        im.save(out, optimize=True)
    tmp.unlink()
    print("wrote", out, im.size)


if __name__ == "__main__":
    main()
