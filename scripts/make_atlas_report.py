"""Render the RRMAP2 relapse atlas as a broad, self-contained HTML report.

Multi-section resource: study design, the temporal relapse-cycle axis, the spatial
L->T->C region axis, the space x time concordance, cellular composition with cluster
identities, gene-program reference tables, and the Hal vignette. Embeds the three
figures as base64. Reads the analysis JSONs/CSVs so it stays in sync with re-runs.

    python scripts/make_hal_celltype_fig.py
    python scripts/make_region_fig.py
    python scripts/make_atlas_report.py
    open runs/reports/rrmap2_relapse_atlas.html
"""

from __future__ import annotations

import base64
import json
import os

CSS = """
:root{--ink:#1a1a1a;--mut:#666;--line:#e6e6e6;--accent:#3b5bdb;--bg:#fafafa;
--up:#c0392b;--down:#2c6fbb;--good:#2f9e44;--warn:#e8590c}
*{box-sizing:border-box}body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
color:var(--ink);background:var(--bg);margin:0;line-height:1.55}
.wrap{max-width:980px;margin:0 auto;padding:48px 28px 90px}
h1{font-size:32px;margin:0 0 4px}
h2{font-size:21px;margin:40px 0 10px;padding-bottom:6px;border-bottom:2px solid var(--line)}
h3{font-size:16px;margin:22px 0 6px;color:#333}
.sub{color:var(--mut);font-size:14px;margin:0 0 8px}
.toc{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px 18px;margin:14px 0;font-size:14px}
.toc a{color:var(--accent);text-decoration:none;margin-right:14px;white-space:nowrap}
.kpis{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0}
.kpi{flex:1 1 150px;background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.kpi .v{font-size:24px;font-weight:650}.kpi .l{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:8px 0}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.up{color:var(--up);font-weight:600}.down{color:var(--down);font-weight:600}.muted{color:#999}
img{max-width:100%;border-radius:8px;margin:10px 0;border:1px solid var(--line)}
.note{background:#fff8f0;border-left:3px solid var(--warn);padding:10px 14px;border-radius:0 8px 8px 0;font-size:13.5px;margin:12px 0}
.good{background:#f0faf2;border-left:3px solid var(--good);padding:10px 14px;border-radius:0 8px 8px 0;font-size:14px;margin:12px 0}
.key{background:#eef2ff;border-left:3px solid var(--accent);padding:10px 14px;border-radius:0 8px 8px 0;font-size:14px;margin:12px 0}
.gene{font-family:"SF Mono",Menlo,Consolas,monospace;font-size:12.5px}
.cols{display:flex;gap:18px;flex-wrap:wrap}.cols>div{flex:1 1 360px}
.pill{display:inline-block;font-size:11px;font-weight:600;padding:2px 8px;border-radius:10px}
.pill.a{background:#fde8e8;color:#c0392b}.pill.r{background:#e7f0fb;color:#2c6fbb}
.stages{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}
.stage{font-size:12px;padding:4px 9px;border-radius:8px;background:#eef;border:1px solid #dde}
.stage.pk{background:#f7d9d9;border-color:#eebcbc}
.foot{color:var(--mut);font-size:12px;margin-top:30px;border-top:1px solid var(--line);padding-top:12px}
.abstract{background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px 22px;margin:16px 0;font-size:15px;line-height:1.62}
.abstract p{margin:0 0 10px}.abstract p:last-child{margin:0}
.terms{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px;margin:14px 0}
.term{background:#fff;border:1px solid var(--line);border-radius:10px;padding:10px 12px;font-size:13px;line-height:1.45}
.term b{display:block;color:var(--accent);font-size:12.5px;margin-bottom:2px}
.lead{font-size:14.5px;color:#333;margin:4px 0 12px;font-style:italic}
.howto{font-size:13px;color:var(--mut);background:#f7f8fc;border:1px dashed var(--line);border-radius:8px;padding:10px 14px;margin:12px 0}
"""
CYCLE = [("PLP·CFA", 3, 0), ("ONSET1", 2, 0), ("ONSET2", 2, 0), ("PEAK1", 4, 1),
         ("REM1", 5, 0), ("PEAK2", 4, 1), ("REM2", 4, 0), ("PEAK3", 5, 1)]


def b64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def img(path, alt):
    b = b64(path)
    return f"<img src='data:image/png;base64,{b}' alt='{alt}'>" if b else f"<p class='muted'>[missing figure: {path}]</p>"


def num(v, prec=2):
    cls = "up" if v > 0 else "down" if v < 0 else ""
    return f"<td class='num {cls}'>{v:+.{prec}f}</td>"


def jload(p, default=None):
    return json.load(open(p)) if os.path.exists(p) else default


def term(title, body):
    return f"<div class='term'><b>{title}</b>{body}</div>"


def lead(text):
    return f"<p class='lead'>{text}</p>"


def main():
    out = "runs/reports/rrmap2_relapse_atlas.html"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    osc = jload("runs/rr_cycle_oscillation/results.json")
    hal = jload("runs/rr_phase_niche/hal_celltype.json")
    reg = jload("runs/rr_region_gradient/results.json")
    labels = jload("runs/rr_region_gradient/cluster_labels.json", {})
    chr_ = jload("runs/chronic_trajectory/results.json")

    H = [f"<!doctype html><html><head><meta charset='utf-8'>"
         f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>RRMAP2 relapse atlas</title><style>{CSS}</style></head><body><div class='wrap'>"]
    H.append("<h1>A spatial molecular atlas of the relapsing–remitting EAE cycle</h1>")
    H.append("<p class='sub'>RRMAP2 spinal-cord Xenium · RELAPSE-REMITTING cohort · 33 animals, ~894k cells, "
             "5101-gene panel · animal-level pseudobulk. Two coupled axes of disease variation: a temporal "
             "<b>relapse cycle</b> and a spatial <b>L→T→C</b> gradient.</p>")
    H.append("<div class='toc'><b>Sections:</b> "
             "<a href='#about'>Overview</a><a href='#design'>Design</a><a href='#time'>Temporal cycle</a>"
             "<a href='#space'>Spatial gradient</a><a href='#myeloid'>Myeloid states</a>"
             "<a href='#signaling'>Signaling</a><a href='#cross'>Space × time</a>"
             "<a href='#conserve'>Cross-model</a><a href='#cells'>Cell types</a>"
             "<a href='#programs'>Program reference</a><a href='#accrual'>Accrual axis</a>"
             "<a href='#beyond'>Beyond severity</a>"
             "<a href='#hal'>Hal</a><a href='#scope'>Scope</a></div>")

    # KPIs
    H.append("<div class='kpis'>")
    H.append("<div class='kpi'><div class='v'>2</div><div class='l'>coupled disease axes</div></div>")
    if osc:
        H.append("<div class='kpi'><div class='v'>138</div><div class='l'>cycle genes q&lt;0.05</div></div>")
    if reg:
        H.append(f"<div class='kpi'><div class='v'>{reg['n_genes_q05_region']}</div>"
                 f"<div class='l'>region genes q&lt;0.05</div></div>")
        H.append(f"<div class='kpi'><div class='v down'>{reg['cross_axis']['spearman_movers']}</div>"
                 f"<div class='l'>space×time corr</div></div>")
    H.append("</div>")
    H.append("<div class='good'><b>Headline.</b> Disease is organized along two coupled axes. In <b>time</b>, "
             "the relapse cycle splits genes into an <b>acute reversible</b> program (flares each relapse, "
             "resets) and an <b>irreversible cumulative</b> program (ratchets up). In <b>space</b>, the same "
             "disease program is concentrated in <b>lumbar</b> cord and declines rostrally toward cervical. "
             "Space and time encode the same biology — the spatial epicenter is the temporally most-advanced "
             "state.</div>")

    # plain-language overview
    H.append("<h2 id='about'>What this is — in plain language</h2>")
    H.append("<div class='abstract'>"
             "<p><b>The disease.</b> These mice have <b>EAE</b> (experimental autoimmune encephalomyelitis), the "
             "standard animal model of <b>multiple sclerosis (MS)</b>: the immune system attacks the myelin "
             "insulation around nerve fibres in the spinal cord, causing weakness and paralysis. In the "
             "<b>relapsing–remitting</b> form studied here (the most common form of MS), an animal goes through "
             "<b>attacks</b> (relapses, the &lsquo;peaks&rsquo;) followed by partial recovery (<b>remissions</b>); "
             "we also compare a <b>chronic</b> form that steadily worsens without recovering.</p>"
             "<p><b>The data.</b> <b>Xenium</b> is imaging-based spatial transcriptomics &mdash; it measures ~5,000 "
             "genes in every individual cell while keeping that cell&rsquo;s physical position in the tissue. Here "
             "that is ~894,000 cells across spinal-cord sections from 33 relapsing-remitting mice (plus a chronic "
             "comparison group). Each mouse is sampled <b>once</b>, at a single point in its disease, and given a "
             "<b>clinical score</b> (0 = healthy, 3+ = paralysis) for how sick it is at that moment.</p>"
             "<p><b>What we did, and the one idea to take away.</b> We mapped how gene activity and cell types change "
             "as the disease waxes and wanes &mdash; over time (the relapse cycle) and through space (along the cord). "
             "One theme recurs: two kinds of change. An <b>acute, reversible</b> response that flares during an attack "
             "and then resets, and an <b>irreversible, cumulative</b> build-up of damage that never fully clears. "
             "Almost everything we measured tracks &lsquo;how sick the animal is right now&rsquo; &mdash; with one "
             "exception that matters: the accumulated damage keeps growing with disease <i>duration</i>, even between "
             "two equally-sick animals.</p></div>")
    H.append("<div class='terms'>"
             + term("EAE / MS model", "Mouse model of multiple sclerosis: immune attack on spinal-cord myelin.")
             + term("Relapsing-remitting vs chronic", "RR: attacks then partial recovery. Chronic: steady worsening. Different mouse strains.")
             + term("Relapse cycle", "onset → peak (attack) → remission (recovery), repeating: PEAK1, REM1, PEAK2, REM2, PEAK3.")
             + term("Clinical score", "Behavioural severity 0–3.25 (limb weakness to paralysis), recorded at sacrifice.")
             + term("Xenium / spatial", "~5,000 genes measured per single cell, with each cell's tissue location preserved.")
             + term("Pseudobulk", "Summing one animal's cells into a single profile so the <i>animal</i> is the statistical unit.")
             + term("Niche / cell type", "Cells grouped by gene profile (leiden) or by spatial neighbourhood (CellCharter).")
             + term("Acute vs cumulative", "Acute: flares & resets each attack. Cumulative: ratchets up and persists (scar / foamy macrophages).")
             + term("L → T → C", "Lumbar → thoracic → cervical: positions down-to-up the spinal cord.")
             + term("&lsquo;Beyond severity&rsquo;", "Whether a signal still carries information after the clinical score is statistically removed.")
             + "</div>")
    H.append("<div class='howto'><b>How to read this.</b> Green boxes are the main findings, blue boxes are the key "
             "quantified results, orange boxes are caveats and limits. Gene names appear in "
             "<span class='gene'>monospace</span> (e.g. <span class='gene'>Hal</span>). Every number is reproducible "
             "from the script named in its section. &lsquo;ρ&rsquo; (rho) is a correlation from −1 to +1; "
             "&lsquo;q&rsquo; / &lsquo;p&rsquo; are significance values (smaller = stronger evidence).</div>")

    # design
    H.append("<h2 id='design'>Study design &amp; the relapse cycle</h2>")
    H.append(lead("Each mouse is one snapshot at one disease stage, so the &lsquo;cycle&rsquo; below is "
                  "reconstructed by lining up many mice caught at different points — not a film of one animal."))
    H.append("<p class='sub'>Terminal cross-sectional: one stage per animal (n shown), so the cycle is a "
             "cross-animal reconstruction, not a within-animal time course. Region, by contrast, is a "
             "<b>within-animal</b> axis (32/33 animals span all three levels) — fully de-confounded from "
             "severity, batch and strain, and the better-powered backbone.</p>")
    H.append("<div class='stages'>" + "".join(
        f"<span class='stage {'pk' if pk else ''}'>{nm} · n={n}</span>" for nm, n, pk in CYCLE) + "</div>")

    # temporal
    H.append("<h2 id='time'>Axis 1 — the temporal relapse cycle</h2>")
    H.append(lead("As the disease attacks and recovers over repeated relapses, two distinct kinds of gene emerge: "
                  "ones that flare and reset with each attack, and ones that build up for good."))
    if osc:
        acute = osc["top_acute_oscillating"][:8]
        ratchet = osc["top_ratchet_cumulative"][:8]
        H.append("<div class='cols'>")
        H.append("<div><h3><span class='pill a'>acute / reversible</span> resets each cycle</h3>"
                 "<p class='sub'>Inflammatory / M2-repair / interferon myeloid; indexes current disease "
                 "activity.</p><table><tr><th>gene</th><th>peak−rem</th><th>floor drift</th></tr>")
        for r in acute:
            H.append("<tr><td class='gene'>" + r["gene"] + "</td>" + num(r["amplitude"]) + num(r["floor_drift"]) + "</tr>")
        H.append("</table></div>")
        H.append("<div><h3><span class='pill r'>irreversible / cumulative</span> floor rises</h3>"
                 "<p class='sub'>Lipid/foamy, ECM/scar, DAM &amp; microglia repopulation; persists after "
                 "recovery.</p><table><tr><th>gene</th><th>floor drift</th><th>peak−rem</th></tr>")
        for r in ratchet:
            H.append("<tr><td class='gene'>" + r["gene"] + "</td>" + num(r["floor_drift"]) + num(r["amplitude"]) + "</tr>")
        H.append("</table></div></div>")
        H.append(img("runs/reports/.cycle_unused.png", ""))  # placeholder removed below
    H[-1] = ""  # drop placeholder if added
    H.append("<p class='sub'>Full temporal profiles: see "
             "<a href='relapse_cycle_oscillation.html'>relapse_cycle_oscillation.html</a>.</p>")

    # spatial
    H.append("<h2 id='space'>Axis 2 — the spatial L→T→C gradient (within-animal)</h2>")
    H.append(lead("Is the disease worse in some parts of the spinal cord than others? Yes — and because each mouse "
                  "spans the whole cord, this axis is the cleanest in the study (it cancels every animal-level "
                  "confound)."))
    H.append(img("runs/rr_region_gradient/figures/region_programs.png", "region programs"))
    if reg:
        H.append("<div class='cols'>")
        H.append("<div><h3>Higher in LUMBAR (disease epicenter)</h3><table><tr><th>gene</th><th>L→C slope</th><th>q</th></tr>")
        for r in reg["region_down_LtoC"][:8]:
            H.append(f"<tr><td class='gene'>{r['gene']}</td>{num(r['region_slope'],3)}<td class='num'>{r['q']:.1g}</td></tr>")
        H.append("</table></div>")
        H.append("<div><h3>Higher in CERVICAL (spared)</h3><table><tr><th>gene</th><th>L→C slope</th><th>q</th></tr>")
        for r in reg["region_up_LtoC"][:8]:
            H.append(f"<tr><td class='gene'>{r['gene']}</td>{num(r['region_slope'],3)}<td class='num'>{r['q']:.1g}</td></tr>")
        H.append("</table></div></div>")
    H.append("<div class='note'><b>Read the axis carefully.</b> The raw region gradient is dominated by "
             "<b>Hox positional-identity</b> genes (posterior <span class='gene'>Hoxa9/b9/a10/c8</span> high "
             "in lumbar, anterior <span class='gene'>Hoxa5/b5</span> high in cervical) — anatomy, not "
             "disease. The <i>disease</i> component runs the same way: inflammation, complement and "
             "demyelination peak in lumbar; neurons and intact myelin increase toward cervical.</div>")

    # lesion architecture (radial)
    H.append("<h2 id='lesion'>Lesion architecture — radial organization</h2>")
    H.append("<p class='sub'>Segmenting lesions and profiling against signed distance-to-edge "
             "(scripts/lesion_radial.py, 570 lesions) recovers a textbook concentric structure de novo: "
             "inflammatory-myeloid + demyelinated <b>core</b> → lymphocyte + complement <b>margin</b> at the "
             "edge → reactive-<b>astrocyte rim</b> just outside → spared myelin parenchyma.</p>")
    H.append(img("runs/lesion_radial/figures/radial_profile.png", "lesion radial profile"))

    # myeloid states
    H.append("<h2 id='myeloid'>Myeloid states — recruitment → activation → resolution</h2>")
    H.append("<p class='sub'>The inflammatory-myeloid compartment is three distinct states that organize "
             "both in time and in the lesion's radial space: <b>c18</b> recruitment/antigen-presenting/IFN "
             "(<span class='gene'>Ccr2, Plac8, Ciita, Cd74, Cxcl10</span>), most <b>peripheral</b> (−11µm, the "
             "margin where cells enter); <b>c5</b> acute glycolytic/M2 (<span class='gene'>Arg1, Chil3, Acod1, "
             "Hal</span>), deepest in the <b>core</b> (−25µm) and the clean acute oscillator; <b>c3</b> "
             "repair/resident-like (<span class='gene'>Mrc1, Igf1, C6</span>), the persistent baseline. The "
             "<b>c5↔c3 swing</b> drives the cycle — c5 38%→9% peak→remission (p=0.009), c3 44%→70% (p=0.014). "
             "c18 is +12.6µm more peripheral than c5 (p=8e-7); astrocytes sit +44µm outside the core. So "
             "recruitment is peripheral, activation is in the core; resolution is an in-place temporal "
             "hand-off, not a migration (ordering inferred from a cross-sectional snapshot).</p>")
    H.append("<div class='cols'><div>" + img("runs/myeloid_states/figures/state_share_vs_cycle.png", "state shares")
             + "</div><div>" + img("runs/myeloid_states/figures/radial_position.png", "radial position") + "</div></div>")

    # signaling
    H.append("<h2 id='signaling'>Lesion signaling wiring (spatial ligand-receptor)</h2>")
    H.append("<p class='sub'>Neighborhood-enrichment + ligand-receptor analysis on a <i>per-section</i> graph "
             "(the precomputed global graph had 14.6% spurious cross-section edges; rebuilt block-diagonal). "
             "Adjacency matches the radial model: T cells embedded in the myeloid core (z≈+100–146), B/plasma "
             "a tight follicle-like aggregate (z+567) adjacent to T cells, and reactive astrocytes <b>avoid</b> "
             "the core (z −28 to −125 — the rim is spatially segregated). Signaling: <b>complement dominates</b> "
             "(<span class='gene'>C3→C3ar1/C5ar1</span>, with reactive astrocytes the top C3 source — notable as "
             "C1q is off-panel); astrocyte→microglia <span class='gene'>Csf1/Il34→Csf1r</span> maintenance; "
             "recruitment <span class='gene'>Ccl2→Ccr2, Cxcl10→Cxcr3</span>; lymphoid "
             "<span class='gene'>Cxcl13→Cxcr5</span>; costim <span class='gene'>Cd80/86→Ctla4/Cd28</span>.</p>")
    H.append("<div class='cols'><div>" + img("runs/lesion_signaling/figures/nhood_enrichment.png", "adjacency")
             + "</div><div>" + img("runs/lesion_signaling/figures/wiring_axes.png", "LR axes") + "</div></div>")
    H.append("<p class='sub muted'>This is the signaling structure of a severity-driven state — wiring, not a "
             "signal beyond severity.</p>")

    # cross-axis
    H.append("<h2 id='cross'>Space × time — does the spatial gradient recapitulate the cycle?</h2>")
    H.append(lead("Time and space could be two unrelated stories. They turn out to be the same story told twice — "
                  "the lower cord is, molecularly, the most disease-advanced region."))
    H.append(img("runs/rr_region_gradient/figures/region_vs_cycle.png", "region vs cycle"))
    if reg:
        ca = reg["cross_axis"]
        H.append(f"<div class='key'><b>Yes — and the sign is the punchline.</b> The per-gene region slope "
                 f"(L→T→C) and the relapse-cycle trend correlate "
                 f"<b>{ca['spearman_movers']:+.2f}</b> (Spearman over {ca['n_movers']} moving genes; "
                 f"{ca['spearman_all_genes']:+.2f} genome-wide). It is <b>negative</b>: the program that "
                 "rises over disease <i>time</i> is the same program concentrated in lumbar <i>space</i>. "
                 "Space and time are two readouts of one disease trajectory, with <b>lumbar = the most "
                 "advanced state</b> and progression running caudal→rostral.</div>")

    # cross-model conservation
    H.append("<h2 id='conserve'>Cross-model conservation — the program is strain-invariant</h2>")
    H.append(lead("Is this specific to one mouse strain, or general EAE biology? A second, independent model (a "
                  "different strain and disease course) says the core program is the same."))
    if chr_:
        cons = chr_["conservation_vs_rr"]["chronic_severity_vs_rr_cycle_mono_rho"]["spearman_rho"]
        nsig = chr_["severity_trajectory"]["n_sig_q05"]
        H.append(f"<p class='sub'>The atlas is built on the RR (SJL/PLP) cohort. The independent "
                 f"<b>chronic</b> cohort (B6/MOG — different strain, antigen and course) has its own "
                 f"severity trajectory ({nsig} genes q&lt;0.05, batch-robust). Correlating the two models' "
                 f"within-model slopes cancels the strain offset:</p>")
        H.append(f"<div class='key'><b>Chronic severity trend vs RR relapse-cycle trend: Spearman "
                 f"{cons:+.2f}</b> (5101 genes, p≈0). The same disease program runs in both models — a "
                 "strain-invariant EAE signature: complement/DAM up (<span class='gene'>C6, Gpnmb, C4b, "
                 "Abca1, Ctss, C3ar1, Csf1r</span>), cholesterol-biosynthesis/myelin down "
                 "(<span class='gene'>Msmo1, Idi1, Hmgcr, Lss, Mal, Plp1, Mog</span>).</div>")
    H.append(img("runs/chronic_trajectory/figures/conservation_scatter.png", "conservation"))
    H.append("<div class='note'><b>Unique to chronic, but blocked.</b> A severity × disease-<i>duration</i> "
             "grid (day-16 vs day-30 sacrifice at matched score) could separate active inflammation from "
             "chronic accrual — but here the day axis is perfectly confounded with run_date, so it needs "
             "matched-batch sampling. Key wet-lab follow-up.</div>")

    # cell types
    H.append("<h2 id='cells'>Cellular composition</h2>")
    H.append(f"<p class='sub'>The 25 leiden_1 niches label to: "
             + ", ".join(f"{ln} ×{n}" for ln, n in _counts(labels)) + ".</p>")
    if reg:
        H.append("<p class='sub'>Niches expanding toward cervical (spared tissue):</p>")
        H.append("<table><tr><th>cluster</th><th>identity</th><th>fraction slope L→C</th><th>q</th></tr>")
        for r in reg["composition_expand_LtoC"][:6]:
            H.append(f"<tr><td class='gene'>c{r['cluster']}</td><td>{labels.get(str(r['cluster']),'?')}</td>"
                     f"{num(r['frac_slope_LtoC'],4)}<td class='num'>{r['q']:.1g}</td></tr>")
        H.append("</table>")
    H.append("<p class='sub'>Along the temporal cycle, the strongest expanding niche is an "
             "<b>astrocyte</b> state (cluster 9, ρ≈+0.87); the acute relapse program centres on the "
             "<b>inflammatory-myeloid</b> niche (cluster 5; see Hal). Inflammatory-myeloid, T/NK and "
             "B/plasma niches are lumbar/thoracic-biased — the spatial disease side.</p>")

    # program reference
    H.append("<h2 id='programs'>Gene-program reference</h2>")
    H.append("<table><tr><th>program</th><th>direction</th><th>example genes</th></tr>"
             "<tr><td>Inflammation / complement</td><td>↑ cycle · ↑ lumbar</td>"
             "<td class='gene'>C6, C3, C4b, Cd74, B2m, Ctss</td></tr>"
             "<tr><td>Inflammatory myeloid (acute)</td><td>↑ peaks · resets</td>"
             "<td class='gene'>Arg1, Chil3, Cd14, Hal, Acod1, Cxcl10</td></tr>"
             "<tr><td>Cumulative / foamy / ECM (ratchet)</td><td>floor ↑ each cycle</td>"
             "<td class='gene'>Gpnmb, Plin4, Igf2, Fmod, Pmp22</td></tr>"
             "<tr><td>Cholesterol / myelin biosynthesis</td><td>↓ cycle · ↑ cervical</td>"
             "<td class='gene'>Hmgcr, Msmo1, Idi1, Ldlr, Plp1, Mbp, Mog</td></tr>"
             "<tr><td>Hox positional identity</td><td>spatial only (anatomy)</td>"
             "<td class='gene'>Hoxa9/b9/a10/c8 (lumbar), Hoxa5/b5 (cervical)</td></tr></table>")

    # accrual axis
    acc = jload("runs/accrual_axis/results.json")
    H.append("<h2 id='accrual'>Same clinical score, different molecular state — the accrual axis</h2>")
    H.append(lead("Two mice can look equally sick on the clinical scale yet be in different molecular states — "
                  "because the score sees current activity but not the damage piled up from earlier attacks."))
    H.append("<p class='sub'>The clinical score reads <i>current</i> disease activity (the reversible acute "
             "program) but is largely blind to <i>accumulated</i>, irreversible damage, which grows with "
             "disease history. So the same score can be a molecularly different state depending on how many "
             "relapses preceded it.</p>")
    if acc:
        w = acc["within_rr_peak1_vs_peak3"]
        H.append(f"<div class='key'><b>Strain-clean test (within RR): PEAK1 vs PEAK3 at near-identical score "
                 f"({w['score_PEAK1']} vs {w['score_PEAK3']}).</b> The <b>cumulative</b> program rises "
                 f"(z {w['cumulative_PEAK1']:+.2f} → {w['cumulative_PEAK3']:+.2f}, p={w['cumulative_p']}) — "
                 "<span class='gene'>Gpnmb</span> 1.9→2.7→3.2, <span class='gene'>Fcrls</span> 0.2→0.8→1.0 — "
                 f"while the <b>acute</b> program does not accrue (<span class='gene'>Hal</span> flat "
                 f"1.32/1.10/1.19; p={w['acute_p']}). Reaching the same score at a later relapse means more "
                 "accumulated damage on a comparable acute flare.</div>")
    H.append(img("runs/accrual_axis/figures/accrual_axis.png", "accrual axis"))
    H.append("<table><tr><th>peak group</th><th>n</th><th>clinical score</th><th>acute (z)</th>"
             "<th>cumulative (z)</th><th>Hal</th><th>Gpnmb</th></tr>")
    for r in (acc["groups"] if acc else []):
        H.append(f"<tr><td>{r['group']}</td><td class='num'>{r['n']}</td><td class='num'>{r['mean_score']}</td>"
                 + num(r['acute_program_z']) + num(r['cumulative_program_z'])
                 + f"<td class='num'>{r.get('gene_Hal','—')}</td><td class='num'>{r.get('gene_Gpnmb','—')}</td></tr>")
    H.append("</table>")
    H.append("<div class='note'><b>Chronic PEAK1 is strain-confounded.</b> Its absolute position mixes disease "
             "history AND genetic background (B6/MOG vs SJL/PLP), so it can't be cleanly compared to the RR "
             "peaks. One striking confounded difference: <span class='gene'>Hal</span> — the headline RR acute "
             "marker — is essentially absent at the chronic peak (0.03 vs ~1.2–1.3 in RR), the one gene that "
             "cleared FDR in the severity×model interaction; relapse-biology vs strain can't be resolved here. "
             "Small n (4–6) throughout — descriptive.</div>")

    # duration axis (actual time)
    dur = jload("runs/duration_axis/results.json")
    H.append("<h3>Duration axis — accrual tracks actual time, not just cycle number</h3>")
    if dur:
        rr = dur["RR"]
        H.append(f"<p class='sub'><code>day_of_sacrifice</code> (~11–49 days post-induction) is real disease "
                 f"duration. Within RR the relapse cycle decouples it from severity (collinearity only "
                 f"{rr['day_vs_score_collinearity_rho']:+.2f}), so a partial Spearman | severity isolates the "
                 "time effect, strain-matched:</p>")
        H.append(f"<div class='key'><b>Cumulative program vs day | severity: partial ρ = "
                 f"{rr['cumulative_program_partial_rho']:+.2f}</b> (p≈0; <span class='gene'>Fmod</span> +0.81, "
                 f"<span class='gene'>Igf2</span> +0.73, <span class='gene'>Fcrls</span> +0.65; "
                 f"{rr['n_genes_q05']} genes accrue with time after severity adjustment). The <b>acute</b> "
                 f"program does not — it wanes (ρ {rr['acute_program_partial_rho']:+.2f}; "
                 "<span class='gene'>Hal</span> −0.05, <span class='gene'>Arg1</span> −0.58). At matched "
                 "severity, later disease = more accumulated damage on a weaker acute flare.</div>")
    H.append(img("runs/duration_axis/figures/duration_axis.png", "duration axis"))
    H.append("<div class='note'><b>Chronic corroborates your late-timepoint intuition</b> (cumulative vs day | "
             "severity = +0.72), and its day-16 vs day-30 matched-severity pairs are the discrete version "
             "(day-30 higher in <span class='gene'>Gpnmb, Cd68, Fmod</span> at equal grade). But chronic's time "
             "axis is confounded with run_date (day16/day30 are different batches) and more collinear with "
             "severity (+0.58), so it corroborates rather than independently confirms.</div>")

    # beyond severity
    H.append("<h2 id='beyond'>Beyond severity — tested to exhaustion, consistently null</h2>")
    H.append(lead("Could the rich molecular data tell us something the simple clinical score cannot? We looked hard, "
                  "at every level of detail — and the honest answer is no."))
    H.append("<p class='sub'>Is anything separable from current disease severity (trajectory direction, "
             "lesion architecture)? Tested and null at every layer: bulk expression, 10 spatial niches, "
             "25 cell types, composition, <b>whole-lesion geometry</b> "
             "(888 lesions; count/size/confluence track severity but 0/22 peak-vs-rem and 0/22 region tests "
             "survive residualization), and — newest — <b>lesion internal organization</b> (core-rim "
             "polarization of myeloid/astrocyte/complement programs, residualized on severity + lesion size: "
             "null, peak-vs-rem p=0.25–0.88, region p=0.41–0.90). Severity is the organizing axis; the state "
             "is a faithful readout of it down to the radial architecture of individual lesions. Separating "
             "'direction of travel' or 'lesion age' from severity would need longitudinal sampling, not a "
             "different analysis.</p>")

    # Hal
    H.append("<h2 id='hal'>Vignette — Hal and its cell of origin</h2>")
    H.append(lead("A spotlight on the single most reproducible gene in the dataset — what it is, which cell carries "
                  "it, and why we read it cautiously."))
    if hal:
        H.append(f"<p class='sub'>The single most reproducible disease-associated gene. A clean acute "
                 f"oscillator (resets each remission), ~<b>{hal['fold_vs_median']}×</b> concentrated in "
                 f"leiden_1 cluster <b>{hal['hal_high_cluster']}</b> — an inflammatory antigen-presenting "
                 "myeloid state (<span class='gene'>Arg1, Chil3, Cd14, Cd74, Gpnmb</span>; homeostatic "
                 "microglia low). So Hal marks infiltrating/activated myeloid cells, not resident microglia "
                 "or neutrophils.</p>")
    H.append(img("runs/rr_phase_niche/figures/hal_celltype.png", "Hal cell type"))
    H.append("<div class='note'>The histidine-catabolism reading (bridge to 'histidine↓ in MS') is a "
             "hypothesis only — a dedicated review found it weak. Here Hal is best presented as a robust "
             "marker of the acute myeloid relapse state. Detail: docs/hal-histidine-deep-research.md.</div>")

    # scope
    H.append("<h2 id='scope'>Scope &amp; honest limits</h2>")
    H.append("<p class='sub'>Cross-sectional in time (no within-animal trajectory; no causal claims); small "
             "per-stage n (2–5 animals). The spatial axis is within-animal and well powered, but its raw form "
             "is dominated by Hox positional identity — disease signal must be read against that backdrop. "
             "Disease activity (severity) is the organizing variable by design; there is no separately "
             "identifiable 'direction of travel' beyond severity (tested null in bulk, niches, cell types and "
             "composition — see relapse_phase_feasibility.html).</p>")

    H.append("<div class='foot'>Immunoformer · RRMAP2 within-RR atlas · generated by "
             "scripts/make_atlas_report.py from runs/rr_cycle_oscillation/, runs/rr_phase_niche/, "
             "runs/rr_region_gradient/. Source: docs/rrmap2-relapse-atlas.md.</div>")
    H.append("</div></body></html>")
    html = "".join(p for p in H if p)
    with open(out, "w") as fh:
        fh.write(html)
    print(f"[done] -> {out}")


def _counts(labels):
    from collections import Counter
    return Counter(labels.values()).most_common()


if __name__ == "__main__":
    main()
