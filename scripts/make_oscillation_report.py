"""Render runs/rr_cycle_oscillation/results.json as a standalone HTML report.

Self-contained: inline CSS + a hand-drawn inline SVG temporal-profile chart, so the
file has no external image/JS dependencies. Matches the visual language of the other
reports in runs/reports/.

    python scripts/make_oscillation_report.py
    open runs/reports/relapse_cycle_oscillation.html
"""

from __future__ import annotations

import json
import os

CYCLE_ORDER = ["PLP CFA", "ONSET1", "ONSET2", "PEAK1", "REMISSION1",
               "PEAK2", "REMISSION2", "PEAK3"]
PEAKS = {"PEAK1", "PEAK2", "PEAK3"}
SHORT = {"PLP CFA": "CFA", "ONSET1": "ON1", "ONSET2": "ON2", "PEAK1": "PK1",
         "REMISSION1": "REM1", "PEAK2": "PK2", "REMISSION2": "REM2", "PEAK3": "PK3"}
COLORS = ["#3b5bdb", "#c0392b", "#2f9e44", "#e8590c", "#7048e8", "#0c8599"]

CSS = """
:root{--ink:#1a1a1a;--mut:#666;--line:#e6e6e6;--accent:#3b5bdb;--bg:#fafafa;
--up:#c0392b;--down:#2c6fbb;--good:#2f9e44;--warn:#e8590c}
*{box-sizing:border-box}body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
color:var(--ink);background:var(--bg);margin:0;line-height:1.55}
.wrap{max-width:920px;margin:0 auto;padding:48px 28px 80px}
h1{font-size:30px;margin:0 0 4px}h2{font-size:21px;margin:38px 0 10px;padding-bottom:6px;border-bottom:2px solid var(--line)}
h3{font-size:16px;margin:22px 0 6px;color:#333}
.sub{color:var(--mut);font-size:14px;margin:0 0 8px}
.card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:14px 0;
box-shadow:0 1px 2px rgba(0,0,0,.03)}
.kpis{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0}
.kpi{flex:1 1 150px;background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.kpi .v{font-size:26px;font-weight:650}.kpi .l{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:8px 0}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.up{color:var(--up);font-weight:600}.down{color:var(--down);font-weight:600}
.note{background:#fff8f0;border-left:3px solid var(--warn);padding:10px 14px;border-radius:0 8px 8px 0;font-size:13.5px;margin:12px 0}
.good{background:#f0faf2;border-left:3px solid var(--good);padding:10px 14px;border-radius:0 8px 8px 0;font-size:13.5px;margin:12px 0}
.gene{font-family:"SF Mono",Menlo,Consolas,monospace;font-size:12.5px}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;margin:6px 0 2px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.swatch{width:14px;height:3px;border-radius:2px;display:inline-block}
.cols{display:flex;gap:18px;flex-wrap:wrap}.cols>div{flex:1 1 360px}
.foot{color:var(--mut);font-size:12px;margin-top:28px;border-top:1px solid var(--line);padding-top:12px}
.pill{display:inline-block;font-size:11px;font-weight:600;padding:2px 8px;border-radius:10px;letter-spacing:.02em}
.pill.acute{background:#fde8e8;color:#c0392b}.pill.ratchet{background:#e7f0fb;color:#2c6fbb}
"""


def svg_profiles(profiles, w=860, h=320, pad=46):
    """Inline SVG multi-line chart of per-stage profiles, with peak stages shaded."""
    genes = list(profiles)
    xs = CYCLE_ORDER
    n = len(xs)
    # y-range across all plotted genes
    allv = [v for g in genes for v in profiles[g]["per_stage"].values() if v is not None]
    ymin, ymax = min(allv), max(allv)
    pad_y = (ymax - ymin) * 0.08 or 0.1
    ymin -= pad_y
    ymax += pad_y

    def X(i):
        return pad + i * (w - 2 * pad) / (n - 1)

    def Y(v):
        return h - pad - (v - ymin) / (ymax - ymin) * (h - 2 * pad)

    parts = [f"<svg viewBox='0 0 {w} {h}' width='100%' style='font:12px -apple-system,sans-serif'>"]
    # peak shading
    for i, s in enumerate(xs):
        if s in PEAKS:
            x0, x1 = X(i) - 26, X(i) + 26
            parts.append(f"<rect x='{x0:.1f}' y='{pad}' width='{x1-x0:.1f}' "
                         f"height='{h-2*pad}' fill='#f2c9c9' opacity='0.45'/>")
    # axes baseline + zero-ish gridline
    parts.append(f"<line x1='{pad}' y1='{h-pad}' x2='{w-pad}' y2='{h-pad}' stroke='#ccc'/>")
    for i, s in enumerate(xs):
        parts.append(f"<text x='{X(i):.1f}' y='{h-pad+18}' text-anchor='middle' "
                     f"fill='#555'>{SHORT[s]}</text>")
    # y ticks
    for k in range(4):
        v = ymin + (ymax - ymin) * k / 3
        y = Y(v)
        parts.append(f"<line x1='{pad}' y1='{y:.1f}' x2='{w-pad}' y2='{y:.1f}' "
                     f"stroke='#eee'/><text x='{pad-8}' y='{y+4:.1f}' text-anchor='end' "
                     f"fill='#999'>{v:.1f}</text>")
    # lines
    for gi, g in enumerate(genes):
        c = COLORS[gi % len(COLORS)]
        pts = []
        for i, s in enumerate(xs):
            v = profiles[g]["per_stage"].get(s)
            if v is None:
                continue
            pts.append((X(i), Y(v)))
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        parts.append(f"<path d='{d}' fill='none' stroke='{c}' stroke-width='2.4'/>")
        for x, y in pts:
            parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3.2' fill='{c}'/>")
    parts.append("</svg>")
    return "".join(parts)


def gene_legend(profiles):
    out = ["<div class='legend'>"]
    for gi, g in enumerate(profiles):
        c = COLORS[gi % len(COLORS)]
        cls = profiles[g]["class"]
        out.append(f"<span><i class='swatch' style='background:{c}'></i>"
                   f"<b class='gene'>{g}</b> <span style='color:#888'>{cls}</span></span>")
    out.append("</div>")
    return "".join(out)


def num(v, signed=True, prec=3):
    cls = "up" if v > 0 else "down" if v < 0 else ""
    s = f"{v:+.{prec}f}" if signed else f"{v:.{prec}f}"
    return f"<td class='num {cls}'>{s}</td>"


def acute_table(rows):
    h = ["<table><tr><th>gene</th><th>amplitude<br>(peak−rem)</th>"
         "<th>floor_drift<br>(REM2−REM1)</th><th>peak_drift<br>(PK3−PK1)</th><th>q</th></tr>"]
    for r in rows:
        h.append("<tr><td class='gene'>" + r["gene"] + "</td>"
                 + num(r["amplitude"]) + num(r["floor_drift"]) + num(r["peak_drift"])
                 + f"<td class='num'>{r['amp_q']:.2g}</td></tr>")
    return "".join(h) + "</table>"


def ratchet_table(rows):
    h = ["<table><tr><th>gene</th><th>floor_drift<br>(REM2−REM1)</th>"
         "<th>amplitude</th><th>mono_rho</th></tr>"]
    for r in rows:
        h.append("<tr><td class='gene'>" + r["gene"] + "</td>"
                 + num(r["floor_drift"]) + num(r["amplitude"])
                 + num(r["mono_rho"]) + "</tr>")
    return "".join(h) + "</table>"


def co_table(rows):
    h = ["<table><tr><th>gene</th><th>profile corr<br>with Hal</th>"
         "<th>amplitude</th><th>floor_drift</th></tr>"]
    for r in rows:
        h.append("<tr><td class='gene'>" + r["gene"] + "</td>"
                 + num(r["profile_corr_with_Hal"]) + num(r["amplitude"])
                 + num(r["floor_drift"]) + "</tr>")
    return "".join(h) + "</table>"


def profile_table(profiles):
    stages = [s for s in CYCLE_ORDER]
    h = ["<table><tr><th>gene</th>"
         + "".join(f"<th>{SHORT[s]}</th>" for s in stages)
         + "<th>class</th></tr>"]
    for g, pr in profiles.items():
        cells = []
        for s in stages:
            v = pr["per_stage"].get(s)
            cells.append(f"<td class='num'>{v:.2f}</td>" if v is not None else "<td>—</td>")
        pill = ("<span class='pill acute'>acute</span>" if "acute" in pr["class"]
                else "<span class='pill ratchet'>ratchet</span>" if "ratchet" in pr["class"]
                else pr["class"])
        h.append(f"<tr><td class='gene'>{g}</td>" + "".join(cells) + f"<td>{pill}</td></tr>")
    return "".join(h) + "</table>"


def main():
    src = "runs/rr_cycle_oscillation/results.json"
    out = "runs/reports/relapse_cycle_oscillation.html"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    r = json.load(open(src))
    nb = r["n_by_stage"]
    prof = r["gene_profiles"]
    hal = prof["Hal"]

    stage_chips = "  ".join(f"{SHORT[s]}={nb[s]}" for s in CYCLE_ORDER)

    H = [f"<!doctype html><html><head><meta charset='utf-8'>"
         f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>Immunoformer — relapse-cycle oscillation</title><style>{CSS}</style></head><body><div class='wrap'>"]
    H.append("<h1>Relapse-cycle temporal axis — oscillation vs ratchet</h1>")
    H.append("<p class='sub'>RRMAP2 spinal-cord EAE · within the RELAPSE-REMITTING cohort · "
             "animal-level pseudobulk (log CP10k). Cross-sectional: one stage per animal, "
             "stages are population snapshots across different mice, not a within-animal trajectory.</p>")

    # KPIs
    H.append("<div class='kpis'>")
    H.append(f"<div class='kpi'><div class='v'>33</div><div class='l'>RR animals</div></div>")
    H.append(f"<div class='kpi'><div class='v'>{r['n_peak_animals']} / {r['n_remission_animals']}</div>"
             f"<div class='l'>peak / remission animals</div></div>")
    H.append(f"<div class='kpi'><div class='v up'>+{hal['amplitude']:.2f}</div>"
             f"<div class='l'>Hal amplitude (peak−rem)</div></div>")
    H.append(f"<div class='kpi'><div class='v down'>{hal['floor_drift']:.2f}</div>"
             f"<div class='l'>Hal floor_drift (resets)</div></div>")
    H.append("</div>")

    H.append("<div class='good'><b>Headline.</b> Along the relapse cycle, genes split into two temporal "
             "classes: <b>acute / oscillating</b> (spike at every peak, fall back at remission, floor does "
             "not rise — resets each cycle) and <b>ratchet / cumulative</b> (remission floor rises cycle "
             "over cycle — survives clinical recovery). <span class='gene'>Hal</span> is a strong "
             "<b>acute oscillator</b>: it spikes at PK1/PK2/PK3 and fully resets, so it tracks "
             "<b>acute relapse activity</b>, not cumulative damage.</div>")

    # chart
    H.append("<h2>Temporal profiles</h2>")
    H.append(f"<p class='sub'>Stage order (n animals): {stage_chips}. Peak stages shaded. "
             "Cycle order has ONSET2 directly after ONSET1.</p>")
    H.append("<div class='card'>")
    H.append(gene_legend(prof))
    H.append(svg_profiles(prof))
    H.append("</div>")
    H.append(profile_table(prof))
    H.append(f"<div class='note'><b>Caveat — severity confound.</b> {r['note']} "
             "Peaks are also the high-clinical-score timepoints, so raw amplitude largely tracks acute "
             "severity; the severity-<i>independent</i> accumulation signal is floor_drift. Per-stage n is "
             "small (2–5 animals) and nothing clears FDR (Hal amplitude q≈0.06). Read as a strong "
             "descriptive pattern, not a significance claim.</div>")

    # two columns: acute vs ratchet
    H.append("<h2>The two temporal classes</h2>")
    H.append("<div class='cols'>")
    H.append("<div><h3>Acute / oscillating — resets each cycle</h3>"
             "<p class='sub'>Inflammatory + M2/repair + interferon macrophage program. "
             "<span class='gene'>Hal</span> sits here.</p>"
             + acute_table(r["top_acute_oscillating"]) + "</div>")
    H.append("<div><h3>Ratchet / cumulative — floor rises</h3>"
             "<p class='sub'>Lipid/foamy, ECM/scar, and homeostatic microglia "
             "(<span class='gene'>Fcrls</span>) — irreversible residue.</p>"
             + ratchet_table(r["top_ratchet_cumulative"]) + "</div>")
    H.append("</div>")

    # co-oscillators
    H.append("<h2>Hal co-oscillators</h2>")
    H.append("<p class='sub'>Genes whose 8-stage profile tracks Hal's (Pearson over stage means). "
             "These define the acute program Hal rides with — note the myeloid/infiltration character "
             "(<span class='gene'>Acp5</span>, <span class='gene'>Il7r</span>, "
             "<span class='gene'>Cxcr4</span>), consistent with Hal being an acute infiltrating-myeloid "
             "signature rather than a resident-microglial one (Fcrls is in the opposite ratchet class).</p>")
    H.append(co_table(r["hal_cooscillators"]))

    H.append("<div class='foot'>Immunoformer · RRMAP2 within-RR relapse cycle · generated by "
             "scripts/make_oscillation_report.py from runs/rr_cycle_oscillation/results.json. "
             "Companion: scripts/rr_cycle_oscillation.py, docs/hal-histidine-finding.md.</div>")
    H.append("</div></body></html>")

    with open(out, "w") as fh:
        fh.write("".join(H))
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
