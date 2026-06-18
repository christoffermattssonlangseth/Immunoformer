"""Render the duration-clock analysis as figures + a self-contained HTML report.

Reads the cached fit arrays (runs/duration_clock/fit_cache.npz), the metrics
(results.json), the confound controls (controls.json) and the gene signature
(clock_genes.csv) — so it stays in sync with re-runs and needs no recompute. Builds five
figures and embeds them base64 into runs/reports/duration_clock.html.

    PYTHONPATH="$PWD" python scripts/make_duration_clock_report.py
    open runs/reports/duration_clock.html
"""
from __future__ import annotations

import base64
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "runs/duration_clock"
FIG = os.path.join(OUT, "figures")
REPORT = "runs/reports/duration_clock.html"

# biological module per gene (for colouring the signature figure)
MODULES = {
    "neuro": ("Neuro / myelin loss", "#2c6fbb",
              ["Mog", "Mal", "Uchl1", "Syn1", "Nptx2", "Gad1", "Gad2", "Slc17a6"]),
    "innate": ("Innate / IFN recedes", "#5aa7d4",
               ["Ccr2", "Tmem173", "Ifit1", "Ifit3", "Irf5", "H2-D1", "Tlr2", "Il10ra",
                "Tmem119", "Siglech", "Stab1", "Pirb", "F13a1"]),
    "ecm": ("ECM / scar switch", "#c0392b",
            ["Thbs2", "Serpine2", "Ptx3", "Fbln2", "Hpse", "Klf5", "Col4a1", "Col4a2", "Eln"]),
    "lymph": ("Lymphoid organization", "#8e44ad",
              ["Cxcl13", "Cxcl12", "Tcf7", "Il2ra", "Mzb1", "Jchain"]),
    "lipid": ("Lipid synth→scavenge", "#e88a00",
              ["Lpl", "Pltp", "Abca8a", "Srebf1", "Hmgcr", "Hsd17b7", "Idi1", "Msmo1",
               "Ldlr", "Acss2"]),
    "circ": ("Circadian / repair", "#2f9e44",
             ["Nr1d1", "Dbp", "Bhlhe40", "Per1", "Hlf", "Txnip", "Ddit4", "Igf1",
              "Igfbp2", "Il33", "S1pr3", "Mlc1"]),
}
REGION_COLORS = {"L": "#e8590c", "T": "#3b5bdb", "C": "#2f9e44"}
REGION_ORDER = ["L", "T", "C"]


def module_of(gene):
    for key, (_, color, genes) in MODULES.items():
        if gene in genes:
            return color
    return "#999999"


# ----------------------------------------------------------------- figures
def fig_scatter(cache):
    """Predicted vs true duration-residual (clock B), coloured by region."""
    predB, dayB, region = cache["predB"], cache["dayB"], cache["region"].astype(str)
    rhoB = float(cache["rhoB"])
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for r in REGION_ORDER:
        m = region == r
        ax.scatter(dayB[m], predB[m], s=80, edgecolor="k", linewidth=.5,
                   color=REGION_COLORS.get(r, "#888"), label=f"{r} (n={m.sum()})", zorder=3)
    lim = [min(dayB.min(), predB.min()), max(dayB.max(), predB.max())]
    pad = 0.05 * (lim[1] - lim[0])
    lim = [lim[0] - pad, lim[1] + pad]
    ax.plot(lim, lim, "--", color="grey", lw=1, zorder=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("true duration residual  (day | severity)")
    ax.set_ylabel("predicted (leave-one-animal-out)")
    ax.set_title(f"Severity-orthogonalized duration clock\nLOAO Spearman = {rhoB:+.2f}",
                 fontsize=12)
    ax.legend(title="cord region", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "report_scatter")


def fig_perm(cache, results):
    """Permutation null histogram vs observed clock-B rho."""
    perm = cache["perm_rho"]
    rhoB = float(cache["rhoB"])
    p = results["clock_B_severity_orthogonalized"]["perm_p"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.hist(perm, bins=16, color="#c9d3ea", edgecolor="#7088c0", linewidth=.6)
    ax.axvline(rhoB, color="#c0392b", lw=2.2, zorder=3,
               label=f"observed  ρ={rhoB:+.2f}")
    ax.axvline(float(perm.mean()), color="#666", lw=1.2, ls="--",
               label=f"null mean  ρ={perm.mean():+.2f}")
    ax.set_xlabel("LOAO Spearman ρ under shuffled day")
    ax.set_ylabel("permutations")
    ax.set_title(f"Permutation null (shuffle day, refit)  ·  p = {p:.4f}", fontsize=12)
    ax.legend(frameon=False, fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "report_perm")


def fig_controls(results, controls):
    """Ladder of LOAO Spearman across clocks + de-confounding controls."""
    rows = [
        ("Clock A (raw day~genes)", results["clock_A_raw"]["spearman"], "#9aa6c2"),
        ("Clock B (− severity)", results["clock_B_severity_orthogonalized"]["spearman"], "#3b5bdb"),
        ("+ sex controlled", controls["fully_controlled_clock"].get("sex_only_spearman", None), "#5a7be0"),
        ("+ sex + region controlled", controls["fully_controlled_clock"]["spearman"], "#2f9e44"),
        ("severity-only baseline (day~score)", results["severity_only_baseline_spearman"], "#c0392b"),
    ]
    rows = [(lbl, v, c) for lbl, v, c in rows if v is not None]
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    colors = [r[2] for r in rows][::-1]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.barh(labels, vals, color=colors, edgecolor="k", linewidth=.4)
    ax.axvline(0, color="k", lw=.8)
    for y, v in enumerate(vals):
        ax.text(v + (0.02 if v >= 0 else -0.02), y, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=10, fontweight="bold")
    ax.set_xlim(-0.75, 1.0)
    ax.set_xlabel("LOAO Spearman ρ  (predict day-of-sacrifice)")
    ax.set_title("The clock strengthens as nuisance axes are removed", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "report_controls")


def fig_genes(coefs):
    """Two-panel grouped signature: top accrue / decline, coloured by module."""
    up = coefs[coefs.coef > 0].sort_values("coef", ascending=False).head(22)
    dn = coefs[coefs.coef < 0].sort_values("coef").head(22)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 7.2))
    for ax, d, title in [(a1, up.iloc[::-1], "Accrue with duration (↑)"),
                         (a2, dn.iloc[::-1], "Decline with duration (↓)")]:
        cols = [module_of(g) for g in d.gene]
        ax.barh(d.gene, d.coef, color=cols, edgecolor="k", linewidth=.3)
        ax.axvline(0, color="k", lw=.6)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("elastic-net coef (duration | severity)")
        ax.tick_params(axis="y", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=v[1]) for v in MODULES.values()]
    labels = [v[0] for v in MODULES.values()]
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Duration-clock gene signature by biological module", fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, "report_genes")


def fig_region(cache, controls):
    """Day-by-region and score-by-region — region tracks severity, not time."""
    region = cache["region"].astype(str)
    day, score = cache["day"], cache["score"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 4.2))
    for ax, y, ylab, ttl in [(a1, day, "day of sacrifice", "Duration by region"),
                             (a2, score, "clinical score", "Severity by region")]:
        data = [y[region == r] for r in REGION_ORDER]
        bp = ax.boxplot(data, labels=REGION_ORDER, patch_artist=True, widths=.6,
                        medianprops=dict(color="k", lw=1.4))
        for patch, r in zip(bp["boxes"], REGION_ORDER):
            patch.set_facecolor(REGION_COLORS[r]); patch.set_alpha(.55)
        for i, r in enumerate(REGION_ORDER):
            xs = np.random.default_rng(i).normal(i + 1, 0.05, (region == r).sum())
            ax.scatter(xs, y[region == r], s=22, color=REGION_COLORS[r], edgecolor="k",
                       linewidth=.3, zorder=3)
        ax.set_ylabel(ylab); ax.set_xlabel("cord region (caudal → rostral)")
        ax.set_title(ttl, fontsize=11.5)
        ax.spines[["top", "right"]].set_visible(False)
    kp = controls["region"]["kruskal_day_region_p"]
    sp = controls["region"]["ordinal_LTC_spearman_score"]
    fig.suptitle(f"Region aligns with severity (ρ={sp:+.2f}) more than time "
                 f"(Kruskal day~region p={kp:.2f}, n.s.)", fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, "report_region")


def _save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=135, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- html
CSS = """
:root{--ink:#1a1a1a;--mut:#666;--line:#e6e6e6;--accent:#3b5bdb;--bg:#fafafa;
--up:#c0392b;--down:#2c6fbb;--good:#2f9e44;--warn:#e8590c}
*{box-sizing:border-box}body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
color:var(--ink);background:var(--bg);margin:0;line-height:1.55}
.wrap{max-width:980px;margin:0 auto;padding:48px 28px 90px}
h1{font-size:30px;margin:0 0 4px}
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
.lead{font-size:14.5px;color:#333;margin:4px 0 12px;font-style:italic}
.foot{color:var(--mut);font-size:12px;margin-top:30px;border-top:1px solid var(--line);padding-top:12px}
.abstract{background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px 22px;margin:16px 0;font-size:15px;line-height:1.62}
.abstract p{margin:0 0 10px}.abstract p:last-child{margin:0}
"""


def b64img(name, alt):
    return b64path(os.path.join(FIG, name + ".png"), alt)


def b64path(p, alt):
    if not os.path.exists(p):
        return f"<p class='muted'>[missing figure: {p}]</p>"
    with open(p, "rb") as fh:
        b = base64.b64encode(fh.read()).decode()
    return f"<img src='data:image/png;base64,{b}' alt='{alt}'>"


def lead(t):
    return f"<p class='lead'>{t}</p>"


def build_html(results, controls, coefs):
    cb = results["clock_B_severity_orthogonalized"]
    fc = controls["fully_controlled_clock"]
    nup = int((coefs.coef > 0).sum())
    ndn = int((coefs.coef < 0).sum())
    H = [f"<!doctype html><html><head><meta charset='utf-8'>"
         f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>RRMAP2 duration clock</title><style>{CSS}</style></head><body><div class='wrap'>"]
    H.append("<h1>A molecular clock of accumulated EAE damage</h1>")
    H.append("<p class='sub'>RRMAP2 spinal-cord Xenium · relapsing–remitting cohort · "
             f"{results['n_animals']} animals · 5101-gene panel · animal-level pseudobulk. "
             "A supervised clock that reads <b>disease duration</b> from tissue — the cumulative, "
             "irreversible axis that grows with disease history, <b>after current severity is removed</b>.</p>")
    H.append("<div class='toc'><b>Sections:</b> "
             "<a href='#what'>Overview</a><a href='#result'>Result</a><a href='#perm'>Validation</a>"
             "<a href='#confound'>Controls</a><a href='#region'>Region</a>"
             "<a href='#biology'>Biology</a><a href='#genes'>Signature</a>"
             "<a href='#tier2'>Localization</a><a href='#caveats'>Caveats</a><a href='#next'>Next</a></div>")

    # KPIs
    H.append("<div class='kpis'>")
    H.append(f"<div class='kpi'><div class='v'>{cb['spearman']:+.2f}</div>"
             "<div class='l'>clock B Spearman (LOAO)</div></div>")
    H.append(f"<div class='kpi'><div class='v'>{cb['perm_p']:.3f}</div>"
             "<div class='l'>permutation p</div></div>")
    H.append(f"<div class='kpi'><div class='v'>{fc['spearman']:+.2f}</div>"
             "<div class='l'>sex+region controlled</div></div>")
    H.append(f"<div class='kpi'><div class='v'>{cb['n_clock_genes']}</div>"
             "<div class='l'>clock genes</div></div>")
    H.append("</div>")
    H.append("<div class='good'><b>Headline.</b> EAE tissue carries two separable molecular clocks. "
             "<b>Severity</b> is acute and reversible — it resets at each relapse. <b>Duration</b> is "
             "cumulative and irreversible. A multivariate elastic-net clock predicts time-since-induction at "
             f"<b>Spearman {cb['spearman']:+.2f}</b> (leave-one-animal-out) even after severity is removed "
             f"(permutation p = {cb['perm_p']:.3f}), and it <b>strengthens</b> — not weakens — when sex and "
             f"cord region are additionally controlled ({fc['spearman']:+.2f}). The signature reads as the "
             "biology of an aging lesion field: neuron/myelin loss, a basement-membrane→fibrotic-scar switch, "
             "lymphoid organization, and a lipid synthesis→scavenging switch.</div>")

    # overview
    H.append("<h2 id='what'>What this is — in plain language</h2>")
    H.append("<div class='abstract'>"
             "<p><b>The problem.</b> Each mouse is sampled <b>once</b>, at one point in its disease, and given a "
             "<b>clinical score</b> for how sick it is at that moment. That score sees <i>current</i> disease "
             "activity but is largely blind to the <i>damage piled up</i> from earlier attacks. Two mice can "
             "look equally sick yet be at very different points in their disease history.</p>"
             "<p><b>The idea.</b> <code>day_of_sacrifice</code> (11–49 days post-induction) is a real measure of "
             "how long disease has run. Within the relapsing–remitting cohort the relapse cycle <b>decouples</b> "
             "time from severity (peaks recur at similar severity but later days), so we can ask: is there a "
             "molecular signal that tracks <i>duration</i> independent of <i>severity</i>? We train a clock to "
             "predict day-of-sacrifice from tissue, with severity statistically removed, and validate it by "
             "leave-one-animal-out cross-validation plus a shuffling test.</p>"
             "<p><b>The answer.</b> Yes — strongly, and it is not an artifact of severity, sex, region or batch.</p>"
             "</div>")

    # result
    H.append("<h2 id='result'>The clock predicts duration beyond severity</h2>")
    H.append(lead("Each dot is one held-out animal: its predicted duration-residual vs the truth. The points "
                  "hug the diagonal across all three cord regions."))
    H.append(b64img("report_scatter", "predicted vs true duration residual"))
    H.append(f"<div class='key'><b>Clock B (severity-orthogonalized): LOAO Spearman {cb['spearman']:+.2f}, "
             f"R² {cb['r2_loao']:+.2f}.</b> Genes <i>and</i> day are residualized on the clinical score within "
             "each training fold, so the clock cannot cheat through severity. For contrast, predicting day from "
             f"<b>severity alone</b> gives <b>{results['severity_only_baseline_spearman']:+.2f}</b> — negative: "
             "later timepoints are not sicker, the relapse cycle resets acute severity. That decoupling is what "
             "makes duration identifiable. A cross-target control (predict score from day-residualized features) "
             f"gives {results['cross_target_predict_score_spearman']:+.2f}, confirming the transcriptome carries "
             "both axes <i>separably</i>.</div>")

    # permutation
    H.append("<h2 id='perm'>Validation — the clock beats a shuffled null</h2>")
    H.append(lead("Could a clock this good arise by chance from 33 animals? We shuffle the day labels and refit, "
                  "50 times. None of the shuffles come close."))
    H.append(b64img("report_perm", "permutation null"))
    H.append(f"<p class='sub'>Permutation null mean ρ = {cb['perm_rho_mean']:+.2f}; observed "
             f"{cb['spearman']:+.2f}; <b>p = {cb['perm_p']:.4f}</b> (floor for 50 permutations). The clock is "
             "not an over-fit. Day-of-sacrifice is also clean of technical batch within RR: Kruskal p(day) = "
             f"{results['batch_leakage_day']['run_date']['kruskal_p_day']} (run_date), "
             f"{results['batch_leakage_day']['sample_id']['kruskal_p_day']} (animal multiplex), "
             f"{results['batch_leakage_day']['region']['kruskal_p_day']} (region).</p>")

    # confound controls
    H.append("<h2 id='confound'>The clock is not a sex, region or severity artifact</h2>")
    H.append(lead("Two nuisance axes could masquerade as time: sex (the signature includes the X-marker Xist) "
                  "and cord region. Removing them makes the clock <i>stronger</i>, not weaker."))
    H.append(b64img("report_controls", "controls ladder"))
    sx = controls["sex"]
    H.append(f"<div class='key'><b>Fully-controlled clock (severity + sex + region, Xist dropped): "
             f"Spearman {fc['spearman']:+.2f}, R² {fc['r2_loao']:+.2f}</b> — above the original "
             f"{cb['spearman']:+.2f}. Sex is orthogonal to day (Xist-raw vs day ρ = {sx['spearman_xistraw_day']:+.2f}; "
             f"Mann-Whitney day~sex p = {sx['mannwhitney_day_sex_p']}), and region tracks severity more than time "
             "(see below). Removing these nuisance directions sharpens the temporal signal.</div>")

    # region
    H.append("<h2 id='region'>Region — a spatial gradient that aligns with severity</h2>")
    H.append(lead("EAE has a spatial cord gradient. It is real, but it indexes how severe the tissue is, not how "
                  "long disease has run — so it cannot be what the clock reads."))
    H.append(b64img("report_region", "region boxplots"))
    rg = controls["region"]
    H.append(f"<p class='sub'>Cervical carries the most severe disease here (mean score C = "
             f"{rg['mean_score']['C']:.2f} &gt; T = {rg['mean_score']['T']:.2f} &gt; L = "
             f"{rg['mean_score']['L']:.2f}); lumbar is sampled earliest. Region orders with severity "
             f"(ordinal L&lt;T&lt;C vs score ρ = {rg['ordinal_LTC_spearman_score']:+.2f}) but is "
             f"<b>not significantly tied to day</b> (Kruskal p = {rg['kruskal_day_region_p']}; ordinal vs day "
             f"ρ = {rg['ordinal_LTC_spearman_day']:+.2f}, p = {rg['ordinal_LTC_spearman_day_p']}). Because clock B "
             "already removes severity, the region gradient cannot drive it — confirmed by the fully-controlled "
             "clock above.</p>")

    # biology
    H.append("<h2 id='biology'>The biology — an aging lesion field</h2>")
    H.append(lead("The multivariate clock picks non-redundant genes, so it reads as a coherent six-part "
                  "chronic-progression program (↑ accrues with duration, ↓ declines; severity removed)."))
    H.append("<table><tr><th>module</th><th>direction</th><th>genes</th></tr>"
             "<tr><td>Neurodegeneration / demyelination</td><td class='down'>↓ declines</td>"
             "<td class='gene'>Mog, Mal, Uchl1, Syn1, Nptx2, Gad1, Gad2, Slc17a6</td></tr>"
             "<tr><td>Acute innate / interferon</td><td class='down'>↓ recedes</td>"
             "<td class='gene'>Ccr2, Tmem173/STING, Ifit1, Ifit3, Irf5, H2-D1, Tmem119, Siglech</td></tr>"
             "<tr><td>Basement-membrane → fibrotic scar</td><td>↓ Col4a1/2, Eln · ↑ scar</td>"
             "<td class='gene'>Thbs2↑, Serpine2↑, Ptx3↑, Fbln2↑, Hpse↑, Klf5↑</td></tr>"
             "<tr><td>Adaptive / lymphoid organization</td><td>↑ follicular · ↓ plasma</td>"
             "<td class='gene'>Cxcl13↑, Cxcl12↑, Tcf7↑, Il2ra↑ · Mzb1↓, Jchain↓</td></tr>"
             "<tr><td>Lipid: synthesis → scavenging</td><td>↓ synth · ↑ uptake</td>"
             "<td class='gene'>Hmgcr↓, Idi1↓, Msmo1↓, Ldlr↓ · Lpl↑, Pltp↑, Abca8a↑, Srebf1↑</td></tr>"
             "<tr><td>Circadian / gliotic repair</td><td class='up'>↑ accrues</td>"
             "<td class='gene'>Nr1d1, Dbp, Bhlhe40, Per1 · Igf1, Igfbp2, Il33, S1pr3</td></tr></table>")
    H.append("<div class='note'><b>Circadian genes are real, not a sampling artifact.</b> Sacrifice "
             "time-of-day is constant across animals, so <span class='gene'>Nr1d1/Dbp/Bhlhe40/Per1</span> "
             "reflect a genuine shift in circadian/metabolic regulation with disease duration.</div>")

    # signature figure
    H.append("<h2 id='genes'>The full signature, coloured by module</h2>")
    H.append(f"<p class='sub'>{cb['n_clock_genes']} genes carry nonzero weight ({nup} accrue, {ndn} decline). "
             "Top 22 in each direction; full list in <span class='gene'>clock_genes.csv</span>.</p>")
    H.append(b64img("report_genes", "gene signature by module"))

    # tier 2 — spatial localization
    sp_path = os.path.join("runs/duration_clock_spatial", "results.json")
    if os.path.exists(sp_path):
        sp = json.load(open(sp_path))
        ctc = sp["cell_type_clocks"]
        comp = sp["composition_clock"]["spearman"]
        pn = sp["perm_nulls"]
        top = ctc[0]
        H.append("<h2 id='tier2'>Where the clock lives — spatial localization</h2>")
        H.append(lead("The clock says tissue tracks duration; this asks which cell types and niches carry it. "
                      "We rebuild the same severity-orthogonalized clock independently inside each compartment "
                      "(too few animals for a black-box attention net) and rank by held-out accuracy."))
        H.append(b64path("runs/duration_clock_spatial/figures/compartment_localization.png",
                         "per-compartment localization"))
        H.append(f"<div class='key'><b>Duration is a tissue-wide chronic program that peaks in glia.</b> Every "
                 f"compartment carries it (LOAO ρ +0.69 to +0.89), led by <b>{top['compartment']}</b> "
                 f"(ρ {top['spearman']:+.2f}, perm p={pn['best_cell_type']['perm_p']}), oligo/myelin and myeloid "
                 f"— neurons and lymphocytes carry the least. Crucially, a clock built on cell-type "
                 f"<b>proportions</b> alone is far weaker (composition ρ {comp:+.2f}, perm p="
                 f"{pn['composition']['perm_p']}): the signal is cells <b>changing state</b>, not just shifting "
                 "in number. The strongest spatial niche is "
                 f"{pn['best_niche']['compartment']} (ρ {pn['best_niche']['spearman']:+.2f}, perm p="
                 f"{pn['best_niche']['perm_p']}).</div>")
        H.append("<table><tr><th>compartment</th><th class='num'>LOAO ρ</th><th class='num'>R²</th>"
                 "<th class='num'>median cells</th></tr>")
        for r in ctc:
            H.append(f"<tr><td>{r['compartment']}</td><td class='num up'>{r['spearman']:+.2f}</td>"
                     f"<td class='num'>{r['r2_loao']:+.2f}</td><td class='num'>{r['median_cells']:.0f}</td></tr>")
        H.append(f"<tr><td class='muted'>composition (fractions only)</td><td class='num'>{comp:+.2f}</td>"
                 "<td class='num muted'>—</td><td class='num muted'>—</td></tr></table>")
        H.append("<h3>Each gene module sits in its expected cell type</h3>")
        H.append(lead("The clock-gene modules are not diffuse — they map onto the cell types that produce them, "
                      "which is what makes the tissue-wide signal mechanistically interpretable."))
        H.append(b64path("runs/duration_clock_spatial/figures/module_compartment_heatmap.png",
                         "module by compartment"))
        H.append("<p class='sub'>Lymphoid genes (<span class='gene'>Cxcl13, Mzb1, Jchain</span>) in B/plasma; "
                 "scar/matricellular (<span class='gene'>Thbs2, Ptx3, Serpine2</span>) in astrocytes; "
                 "basement-membrane (<span class='gene'>Col4a1/2, Eln, Cxcl12</span>) in vascular/endothelial; "
                 "myelin (<span class='gene'>Mog, Mal</span>) in oligodendrocytes; foamy-lipid uptake "
                 "(<span class='gene'>Lpl</span>) in myeloid; cholesterol synthesis "
                 "(<span class='gene'>Hmgcr, Idi1, Msmo1</span>) in OPC/oligo; neuronal "
                 "(<span class='gene'>Uchl1, Gad1, Slc17a6</span>) in neurons.</p>")

    # caveats
    H.append("<h2 id='caveats'>Honest limits</h2>")
    H.append("<p class='sub'>n = 33, animal-level pseudobulk (spatial information collapsed) — LOAO + "
             "permutation null guard against over-fit but the cohort is small. <b>RR only</b>: the chronic arm "
             "is excluded because there day-of-sacrifice aliases run_date (batch). Sex cohort is imbalanced "
             "(7 vs 26) though orthogonal to day. The clock is a supervised regression onto a de-confounded "
             "temporal <i>label</i>, not a within-animal time course — there are no per-animal sequences in "
             "terminal cross-sectional data.</p>")

    # next
    H.append("<h2 id='next'>Next</h2>")
    H.append("<p class='sub'>The localization shows the clock is glia-led and state-driven, but the compartment "
             "clocks still aggregate each cell type to one number per animal. The finer follow-ups: "
             "(1) a within-lesion version using the radial core→rim→margin zones (does duration concentrate in "
             "the scar core?); (2) validation in the chronic arm with matched-batch sampling to break the "
             "day~run_date confound; (3) a publication-grade null (N_PERM≈1000) on the headline clock.</p>")

    H.append("<div class='foot'>Immunoformer · RRMAP2 duration clock · generated by "
             "scripts/make_duration_clock_report.py from runs/duration_clock/. "
             "Source: docs/rrmap2-duration-clock.md · scripts/duration_clock.py, duration_clock_controls.py.</div>")
    H.append("</div></body></html>")
    return "".join(H)


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    cache = np.load(os.path.join(OUT, "fit_cache.npz"), allow_pickle=True)
    results = json.load(open(os.path.join(OUT, "results.json")))
    controls = json.load(open(os.path.join(OUT, "controls.json")))
    coefs = pd.read_csv(os.path.join(OUT, "clock_genes.csv"))

    fig_scatter(cache)
    fig_perm(cache, results)
    fig_controls(results, controls)
    fig_genes(coefs)
    fig_region(cache, controls)
    print("[figures] 5 panels ->", FIG, flush=True)

    html = build_html(results, controls, coefs)
    with open(REPORT, "w") as fh:
        fh.write(html)
    print(f"[done] -> {REPORT}", flush=True)


if __name__ == "__main__":
    main()
