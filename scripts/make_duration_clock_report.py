"""Render the duration-clock analysis as figures + a self-contained HTML report.

Reads the cached fit arrays (runs/duration_clock/fit_cache.npz), the metrics
(results.json), the confound controls (controls.json) and the gene signature
(clock_genes.csv) — so it stays in sync with re-runs and needs no recompute. Two further
sections render the two-clock split at the gene level: the continuous day-vs-severity
partial-correlation decomposition (runs/duration_gene_decomposition/) and, as model-free
corroboration, the discrete relapse-cycle oscillation/ratchet metrics
(runs/rr_cycle_oscillation/ + the cached RR pseudobulk). All sections degrade gracefully
if their inputs are absent. Figures are embedded base64 into runs/reports/duration_clock.html.

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

# ---- relapse-cycle oscillation context (the "other clock") ----
# Cached pseudobulk + metrics from scripts/rr_cycle_oscillation.py — lets us draw the
# severity (oscillating) axis next to the duration (ratchet) axis with no h5ad reload.
OSC_CACHE = "runs/rr_within_relapse/pseudobulk.npz"
OSC_METRICS = "runs/rr_cycle_oscillation/gene_oscillation_metrics.csv"
CYCLE_ORDER = ["PLP CFA", "ONSET1", "ONSET2", "PEAK1", "REMISSION1",
               "PEAK2", "REMISSION2", "PEAK3"]
CYCLE_PEAKS = {"PEAK1", "PEAK2", "PEAK3"}
CYCLE_SHORT = {"PLP CFA": "CFA", "ONSET1": "ON1", "ONSET2": "ON2", "PEAK1": "PK1",
               "REMISSION1": "REM1", "PEAK2": "PK2", "REMISSION2": "REM2", "PEAK3": "PK3"}
# representative profiles drawn as shape (z-scored): acute oscillators that reset,
# ratchet genes that climb past remission.
OSC_ACUTE_PROFILE = ["Arg1", "Chil3", "Timp1", "Acod1", "Hal"]
OSC_RATCHET_PROFILE = ["Igf2", "Fmod", "Fcrls"]
OSC_WARM = ["#c0392b", "#e8590c", "#d6336c", "#f08c00", "#a61e4d"]
OSC_COOL = ["#2c6fbb", "#1098ad", "#5f3dc4"]

# ---- gene-level two-clock decomposition on the continuous day axis ----
DECOMP_DIR = "runs/duration_gene_decomposition"
# ---- compositional (cell-number) vs cell-intrinsic accrual ----
COMPI_DIR = "runs/duration_composition_intrinsic"
# ---- chronic-arm replication + RR↔chronic cross-model relating ----
CHRONIC_DIR = "runs/duration_clock_chronic"
# ---- within-strain: monophasic vs relapsing course ----
MONO_DIR = "runs/monophasic_vs_relapsing"


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


def fig_oscillation(pb, genes, stage, metrics):
    """Two panels: z-scored relapse-cycle profiles (reset vs climb) + the
    amplitude-vs-floor_drift map that places every gene on the severity↔duration axes."""
    gi = {g: i for i, g in enumerate(genes)}
    present = [s for s in CYCLE_ORDER if (stage == s).sum() > 0]
    x = np.arange(len(present))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 5.4))

    # ---- panel A: temporal shape (z-scored so baselines don't matter) ----
    for s_i, s in enumerate(present):
        if s in CYCLE_PEAKS:
            a1.axvspan(s_i - 0.4, s_i + 0.4, color="#f2c9c9", alpha=.5, zorder=0)
    for k, g in enumerate([g for g in OSC_ACUTE_PROFILE if g in gi]):
        prof = np.array([pb[stage == s, gi[g]].mean() for s in present])
        z = (prof - prof.mean()) / (prof.std() + 1e-9)
        a1.plot(x, z, "-o", lw=2, ms=4, color=OSC_WARM[k % len(OSC_WARM)],
                label=f"{g}  (acute)", zorder=3)
    for k, g in enumerate([g for g in OSC_RATCHET_PROFILE if g in gi]):
        prof = np.array([pb[stage == s, gi[g]].mean() for s in present])
        z = (prof - prof.mean()) / (prof.std() + 1e-9)
        a1.plot(x, z, "--s", lw=2, ms=4, color=OSC_COOL[k % len(OSC_COOL)],
                label=f"{g}  (ratchet)", zorder=3)
    a1.axhline(0, color="#aaa", lw=.6)
    a1.set_xticks(x)
    a1.set_xticklabels([CYCLE_SHORT[s] for s in present], rotation=40, ha="right")
    a1.set_ylabel("z-scored pseudobulk  (shape only)")
    a1.set_title("Acute genes spike at each peak and reset;\nratchet genes climb past remission",
                 fontsize=11)
    a1.legend(fontsize=8, frameon=False, ncol=2)
    a1.spines[["top", "right"]].set_visible(False)

    # ---- panel B: the two-axis map ----
    a2.scatter(metrics.amplitude, metrics.floor_drift, s=6, c="#cfcfcf", zorder=1)
    acute = metrics[metrics["class"] == "acute_oscillating"]
    ratchet = metrics[metrics.floor_drift > 0.2]
    a2.scatter(acute.amplitude, acute.floor_drift, s=15, c="#c0392b", alpha=.7,
               zorder=2, label=f"acute oscillating (n={len(acute)})")
    a2.scatter(ratchet.amplitude, ratchet.floor_drift, s=15, c="#2c6fbb", alpha=.7,
               zorder=2, label=f"ratchet / cumulative (n={len(ratchet)})")
    lab = (set(acute.sort_values("amplitude", ascending=False).head(9).gene)
           | set(ratchet.sort_values("floor_drift", ascending=False).head(8).gene))
    for r in metrics[metrics.gene.isin(lab)].itertuples():
        a2.annotate(r.gene, (r.amplitude, r.floor_drift), fontsize=7.2,
                    xytext=(3, 2), textcoords="offset points")
    a2.axhline(0, color="#aaa", lw=.6)
    a2.axhline(0.2, color="#2c6fbb", lw=.6, ls="--")
    a2.axvline(0.3, color="#c0392b", lw=.6, ls="--")
    a2.set_xlabel("amplitude  (peak − remission)  →  oscillation / severity")
    a2.set_ylabel("floor_drift  (REM2 − REM1)  →  ratchet / duration")
    a2.set_title("Every gene placed by how much it oscillates\nvs how much it accrues",
                 fontsize=11)
    a2.legend(fontsize=8.5, frameon=False, loc="upper right")
    a2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "report_oscillation")


def osc_table(metrics, kind):
    """Expanded acute-oscillator / ratchet tables drawn from the full 5101-gene metrics."""
    if kind == "acute":
        d = (metrics[metrics["class"] == "acute_oscillating"]
             .sort_values("amplitude", ascending=False).head(20))
        head = ("<table><tr><th>gene</th><th class='num'>amplitude<br>(peak−rem)</th>"
                "<th class='num'>floor drift</th><th class='num'>peak drift</th>"
                "<th class='num'>q</th></tr>")
        rows = "".join(
            f"<tr><td class='gene'>{r.gene}</td><td class='num up'>{r.amplitude:+.2f}</td>"
            f"<td class='num'>{r.floor_drift:+.2f}</td><td class='num'>{r.peak_drift:+.2f}</td>"
            f"<td class='num'>{r.amp_q:.2g}</td></tr>" for r in d.itertuples())
    else:
        d = (metrics[metrics.floor_drift > 0]
             .sort_values("floor_drift", ascending=False).head(15))
        head = ("<table><tr><th>gene</th><th class='num'>floor drift<br>(REM2−REM1)</th>"
                "<th class='num'>amplitude</th><th class='num'>mono ρ</th></tr>")
        rows = "".join(
            f"<tr><td class='gene'>{r.gene}</td><td class='num up'>{r.floor_drift:+.2f}</td>"
            f"<td class='num'>{r.amplitude:+.2f}</td><td class='num'>{r.mono_rho:+.2f}</td></tr>"
            for r in d.itertuples())
    return head + rows + "</table>"


def decomp_table(records, axis):
    """Day-stratified gene table. axis 'up'/'down' = accrual; 'sev' = severity-oscillating."""
    def clk(r):
        return " <span class='gene' style='color:#2f9e44'>•clock</span>" if r.get("in_clock") else ""
    if axis in ("up", "down"):
        head = ("<table><tr><th>gene</th><th class='num'>r(day&nbsp;|&nbsp;score)</th>"
                "<th class='num'>r(score&nbsp;|&nbsp;day)</th><th class='num'>q</th></tr>")
        rows = "".join(
            f"<tr><td class='gene'>{r['gene']}{clk(r)}</td>"
            f"<td class='num {'up' if r['r_day_given_score'] > 0 else 'down'}'>"
            f"{r['r_day_given_score']:+.2f}</td>"
            f"<td class='num'>{r['r_score_given_day']:+.2f}</td>"
            f"<td class='num'>{r['q_day']:.1g}</td></tr>" for r in records)
    else:
        head = ("<table><tr><th>gene</th><th class='num'>r(score&nbsp;|&nbsp;day)</th>"
                "<th class='num'>r(day&nbsp;|&nbsp;score)</th><th class='num'>q</th></tr>")
        rows = "".join(
            f"<tr><td class='gene'>{r['gene']}{clk(r)}</td>"
            f"<td class='num {'up' if r['r_score_given_day'] > 0 else 'down'}'>"
            f"{r['r_score_given_day']:+.2f}</td>"
            f"<td class='num'>{r['r_day_given_score']:+.2f}</td>"
            f"<td class='num'>{r['q_score']:.1g}</td></tr>" for r in records)
    return head + rows + "</table>"


def _chronic_table(records, kind):
    """Conserved-gene table: RR vs chronic day-partial correlations side by side."""
    cls = "up" if kind == "accrue" else "down"
    head = ("<table><tr><th>gene</th><th class='num'>RR r(day|score)</th>"
            "<th class='num'>chronic r(day|score)</th></tr>")
    rows = "".join(
        f"<tr><td class='gene'>{r['gene']}</td>"
        f"<td class='num {cls}'>{r['rr_day']:+.2f}</td>"
        f"<td class='num {cls}'>{r['ch_day']:+.2f}</td></tr>" for r in records)
    return head + rows + "</table>"


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
.part{margin:58px 0 8px;border-bottom:3px solid var(--accent);padding-bottom:6px}
.part:first-of-type{margin-top:34px}
.part .pn{font-size:12px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:var(--accent)}
.part .pt{font-size:23px;font-weight:650;color:var(--ink);margin-top:1px}
.concept{background:#f5f8ff;border:1px solid #d7e0ff;border-left:4px solid var(--accent);
border-radius:0 10px 10px 0;padding:13px 17px;margin:14px 0;font-size:14.5px;line-height:1.62}
.concept .h{display:block;font-weight:700;color:var(--accent);font-size:12px;text-transform:uppercase;
letter-spacing:.05em;margin-bottom:4px}
.toc .grp{display:block;margin:3px 0}.toc .gl{color:#888;font-weight:600;margin-right:6px}
.side{display:none}
.side .t{font-size:13px;font-weight:700;margin:0 0 2px}
.side .s{font-size:11px;color:var(--mut);margin:0 0 12px}
.side .gl{display:block;font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
color:#aaa;margin:13px 0 3px}
.side a{display:block;color:#3a4250;text-decoration:none;padding:2px 0 2px 9px;line-height:1.3;
border-left:2px solid transparent}
.side a:hover{color:var(--accent);border-left-color:var(--accent)}
@media (min-width:1180px){
  body{padding-left:236px}
  .side{display:block;position:fixed;top:0;left:0;width:236px;height:100vh;overflow-y:auto;
    background:#fff;border-right:1px solid var(--line);padding:24px 18px 48px;box-sizing:border-box}
  .toc{display:none}
}
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


def part(num, title):
    return f"<div class='part'><div class='pn'>Part {num}</div><div class='pt'>{title}</div></div>"


def concept(text):
    return f"<div class='concept'><span class='h'>The idea</span>{text}</div>"


def methods_section(results):
    """Methodology block — placed up front, before the findings."""
    M = []
    M.append("<h2 id='methods'>Methods — data, models, and statistics</h2>")
    M.append(lead("Everything in this report is built from animal-level pseudobulk and cached per-cell-type "
                  "aggregates; no analysis below re-reads the raw cell-by-gene matrix. Each section names "
                  "the script that produces it."))
    M.append("<h3>Data &amp; cohort</h3>")
    M.append("<p class='sub'>RRMAP2 spinal-cord <b>Xenium</b>, a targeted <b>5101-gene</b> panel. "
             f"The headline clock is the relapsing–remitting cohort ({results['n_animals']} animals, "
             "SJL/PLP); the chronic arm (34 animals, B6/MOG) is analysed separately for cross-model "
             "replication. With corrected metadata the chronic <code>day_of_sacrifice</code> is continuous "
             "(8–50 dpi) and clean of <code>run_date</code> (Kruskal p≈0.57) — the earlier batch-alias "
             "concern was an artifact of the coarse day16/day30 stage grouping, not the underlying day. "
             "Each animal is a single <b>terminal</b> timepoint (cross-sectional — no within-animal time "
             "courses), labelled with <code>day_of_sacrifice</code> (11–49 days post-induction) and a "
             "clinical <code>score_sacrifice</code>. <b>Pseudobulk</b> = per-animal mean of log CP10k over "
             "all cells (<span class='gene'>runs/rr_within_relapse/pseudobulk.npz</span>); per-(animal × "
             "leiden-cluster × gene) means and cell counts come from "
             "<span class='gene'>runs/rr_phase_niche/agg_leiden_1.npz</span>.</p>")
    M.append("<h3>The duration clock (supervised regression)</h3>")
    M.append("<p class='sub'>Elastic-net (<span class='gene'>ElasticNetCV</span>, l1-ratio 0.5, 5-fold inner "
             "CV, 50-alpha path, features standardized) under <b>leave-one-animal-out</b> outer CV; accuracy "
             "is Spearman ρ and R² on the held-out predictions. A per-fold <b>top-1000-variance prefilter</b> "
             "(computed on the training fold only — no leakage) cuts 5101→1000 features. <b>Clock A</b> "
             "regresses day on genes directly. <b>Clock B</b> (headline) is <b>severity-orthogonalized</b>: "
             "within every training fold both the gene matrix and day are linearly residualized on "
             "<code>score_sacrifice</code>, so the clock predicts the part of duration that severity cannot "
             "explain; the evaluation target is the full-data day-residual. The signature is the nonzero "
             "coefficients of a full-data Clock-B fit. Source: <span class='gene'>scripts/duration_clock.py</span>.</p>")
    M.append("<p class='sub'><b>Adversarial controls.</b> (i) severity-only baseline — predict day from score "
             "alone; (ii) cross-target — run the same pipeline predicting score from day-residualized "
             "features (axes separable?); (iii) <b>permutation null</b> — shuffle day across animals and "
             "refit Clock B 50×, p = (1 + #{ρ<sub>perm</sub> ≥ ρ<sub>obs</sub>}) / (1 + 50); (iv) batch "
             "leakage — Kruskal–Wallis of day across run_date / animal-multiplex / region.</p>")
    M.append("<h3>Confound controls — sex &amp; region</h3>")
    M.append("<p class='sub'>The panel has <span class='gene'>Xist</span> but no Y-genes, so <b>sex</b> is "
             "called from Xist bimodality (high = female) and tested against day with Mann–Whitney and "
             "point-biserial correlation. <b>Region</b> (L/T/C, caudal→rostral) is tested with Kruskal–Wallis "
             "and an ordinal L&lt;T&lt;C Spearman against both day and score. The fully-controlled clock "
             "re-runs Clock B with <code>covar = [score, sex, region one-hot]</code> and Xist dropped from the "
             "features. Source: <span class='gene'>scripts/duration_clock_controls.py</span>.</p>")
    M.append("<h3>Gene-level day decomposition (partial correlation)</h3>")
    M.append("<p class='sub'>For each gene, two partial correlations over the 33 animals: "
             "r(gene, day | score) — the accrual axis — and r(gene, score | day) — the oscillation axis — "
             "via r<sub>xy·z</sub> = (r<sub>xy</sub> − r<sub>xz</sub>r<sub>yz</sub>) / "
             "√((1−r<sub>xz</sub>²)(1−r<sub>yz</sub>²)). Significance from t = r√(df/(1−r²)), df = n − 3, "
             "with Benjamini–Hochberg FDR across all 5101 genes. Genes are called accruing/declining at "
             "|r(day|score)| ≥ 0.30 and severity-oscillating at |r(score|day)| ≥ 0.30 with |r(day|score)| &lt; "
             "0.30. Source: <span class='gene'>scripts/duration_gene_decomposition.py</span>.</p>")
    M.append("<h3>Relapse-cycle oscillation (model-free, stage-stratified)</h3>")
    M.append("<p class='sub'>Within the relapse cycle CFA→ONSET1/2→PEAK1→REM1→PEAK2→REM2→PEAK3, per gene: "
             "<b>amplitude</b> = mean(peak animals) − mean(remission animals); <b>floor_drift</b> = "
             "mean(REMISSION2) − mean(REMISSION1); <b>peak_drift</b> = PEAK3 − PEAK1; mono_rho = Spearman vs "
             "cycle rank. Mann–Whitney (peaks vs remissions) with BH-FDR. Acute oscillating = amplitude &gt; "
             "0.3 with floor_drift &lt; 0.15 (resets); ratchet = floor_drift &gt; 0.2 (survives remission). "
             "Source: <span class='gene'>scripts/rr_cycle_oscillation.py</span>.</p>")
    M.append("<h3>Compositional vs cell-intrinsic accrual (shift-share)</h3>")
    M.append("<p class='sub'>25 leiden clusters are collapsed to 10 labelled cell types. Writing pseudobulk "
             "as pb(a) = Σ<sub>c</sub> f(a,c)·e(a,c) with cell-type fraction f and per-cell-type mean e, the "
             "Oaxaca/shift-share identity splits each animal's deviation into intrinsic Σf̄·δe, compositional "
             "Σē·δf, and interaction Σδf·δe. Because covariance is linear, the severity-removed day trend "
             "decomposes exactly: Cov(pb, day | score) = Cov<sub>intr</sub> + Cov<sub>comp</sub> + "
             "Cov<sub>inter</sub> (day residualized on score). Per-gene shares are reported over genes that "
             "accrue (r(day|score) ≥ 0.30 for the median; ≥ 0.40 for per-gene classification, where shares "
             "are stable). Cell-type fraction trends are partial correlations r(fraction, day | score). "
             "Source: <span class='gene'>scripts/duration_composition_intrinsic.py</span>.</p>")
    M.append("<h3>Spatial localization (Tier 2)</h3>")
    M.append("<p class='sub'>The severity-orthogonalized Clock B is rebuilt independently inside each cell "
             "type and each CellCharter niche from the cached aggregates (leaner: l1-ratio 0.5, 3-fold CV, "
             "30-alpha path, top-800 variance; an animal needs ≥20 cells in a compartment, a compartment "
             "≥30 usable animals). A composition clock predicts day from cell-type fractions alone. "
             "Permutation nulls (50×) on the best cell type, best niche, and the composition clock. "
             "Source: <span class='gene'>scripts/duration_clock_spatial.py</span>.</p>")
    M.append("<h3>Cross-model replication (chronic arm)</h3>")
    M.append("<p class='sub'>The identical Clock A/B pipeline and the gene-level day decomposition are run on "
             "the chronic pseudobulk (<span class='gene'>runs/chronic_trajectory/pseudobulk.npz</span>) with "
             "day/score from the corrected metadata. Because RR (SJL/PLP) and chronic (B6/MOG) differ in "
             "strain and slide, only within-model <i>slopes</i> are compared (per-gene day|score partial "
             "correlations), which cancels the cross-strain level offset: Spearman of the RR vs chronic "
             "day-axis over all 5101 genes, with the RR severity-oscillation axis as a negative-control "
             "contrast. Source: <span class='gene'>scripts/duration_clock_chronic.py</span>.</p>")
    M.append("<h3>Parameters at a glance</h3>")
    M.append("<table><tr><th>analysis</th><th>estimator / test</th><th>key settings</th></tr>"
             "<tr><td>Duration clock A/B</td><td>ElasticNetCV, LOAO</td>"
             "<td>l1 0.5 · inner cv 5 · 50 alphas · top-1000 var · perm 50×</td></tr>"
             "<tr><td>Confound refit</td><td>ElasticNetCV, LOAO</td>"
             "<td>covar [score, sex, region]; Xist dropped</td></tr>"
             "<tr><td>Gene day decomposition</td><td>partial correlation</td>"
             "<td>df = n−3 · BH-FDR · |r| ≥ 0.30</td></tr>"
             "<tr><td>Relapse-cycle split</td><td>Mann–Whitney + BH</td>"
             "<td>acute: amp&gt;0.3 &amp; floor&lt;0.15 · ratchet: floor&gt;0.2</td></tr>"
             "<tr><td>Composition vs intrinsic</td><td>shift-share (Oaxaca)</td>"
             "<td>10 cell types · Cov(pb, day | score) split</td></tr>"
             "<tr><td>Spatial localization</td><td>ElasticNetCV per compartment</td>"
             "<td>l1 0.5 · cv 3 · 30 alphas · top-800 · ≥20 cells / ≥30 animals</td></tr></table>")
    return M


def build_html(results, controls, coefs, osc=None, decomp=None, compi=None, chronic=None,
               mono=None):
    cb = results["clock_B_severity_orthogonalized"]
    fc = controls["fully_controlled_clock"]
    nup = int((coefs.coef > 0).sum())
    ndn = int((coefs.coef < 0).sum())

    # Part I content (relapse-cycle oscillation vs ratchet) is pre-built so it can lead the
    # narrative — the most intuitive, model-free demonstration of the two-axis idea.
    part1 = []
    if osc is not None:
        m = osc["metrics"]
        n_acute = int((m["class"] == "acute_oscillating").sum())
        n_ratchet = int((m.floor_drift > 0.2).sum())
        part1.append("<h2 id='cycle'>Two behaviours along the relapse cycle</h2>")
        part1.append(lead("Start with the simplest, model-free view. Track each gene across the relapse "
                          "cycle CFA→ONSET→PEAK1→REM1→PEAK2→REM2→PEAK3 and two distinct behaviours fall "
                          "out: genes that <b>spike at every attack and reset</b> in remission, and genes "
                          "whose remission <b>floor ratchets upward</b> cycle after cycle."))
        part1.append(b64img("report_oscillation", "oscillation vs ratchet across the relapse cycle"))
        part1.append("<div class='key'><b>Two measurable behaviours.</b> <b>amplitude</b> (peak − "
                     "remission) measures how hard a gene oscillates with each attack; <b>floor_drift</b> "
                     "(REMISSION2 − REMISSION1) measures whether it stays elevated after the animal has "
                     f"clinically recovered. {n_acute} genes are acute oscillators (spike, then reset), "
                     f"{n_ratchet} ratchet upward (survive remission). These are the two raw faces of the "
                     "<b>severity</b> axis (oscillates, reversible) and the <b>duration</b> axis "
                     "(accumulates, irreversible) — everything that follows makes the second one "
                     "quantitative.</div>")
        part1.append("<div class='cols'>")
        part1.append("<div><h3>Acute oscillators — reset each cycle (severity)</h3>"
                     "<p class='sub'>Inflammatory / M2-macrophage / interferon program: spikes at every "
                     "peak, floor does not rise.</p>" + osc_table(m, "acute") + "</div>")
        part1.append("<div><h3>Ratchet genes — floor rises (duration)</h3>"
                     "<p class='sub'>Lipid/foamy, ECM/scar, and homeostatic-microglia residue that "
                     "survives remission — and recurs in the duration clock and day-accrual genes later "
                     "(<span class='gene'>Fmod, Cemip, Crabp2, Igf2, Vtn</span>).</p>"
                     + osc_table(m, "ratchet") + "</div>")
        part1.append("</div>")
        part1.append("<div class='note'><b>Cross-sectional caveat.</b> One stage per animal (2–5 animals "
                     "per stage), so these are population snapshots across different mice, not within-animal "
                     "trajectories, and nothing clears FDR (best amplitude q≈0.06). Read it as the "
                     "descriptive motivation for the supervised clock that follows, not a significance "
                     "claim. Source: scripts/rr_cycle_oscillation.py.</div>")

    sidebar = (
        "<nav class='side'>"
        "<div class='t'>Duration clock</div><div class='s'>RRMAP2 · EAE spinal cord</div>"
        "<span class='gl'>Start</span>"
        "<a href='#what'>The big idea</a><a href='#approach'>How to read this</a>"
        "<span class='gl'>I · Two clocks</span>"
        "<a href='#cycle'>Two behaviours along the cycle</a>"
        "<span class='gl'>II · The clock</span>"
        "<a href='#result'>Reads duration</a><a href='#perm'>Validation</a>"
        "<a href='#confound'>Not a confound</a><a href='#region'>Region</a>"
        "<span class='gl'>III · What it reads</span>"
        "<a href='#oscillation'>Which genes</a><a href='#composition'>Real upregulation?</a>"
        "<a href='#biology'>Biology</a><a href='#genes'>Signature</a>"
        "<a href='#tier2'>Where it lives</a>"
        "<span class='gl'>IV · Meaning</span>"
        "<a href='#chronic'>Cross-model</a><a href='#course'>Relapse or not</a>"
        "<span class='gl'>Reference</span>"
        "<a href='#methods'>Methods</a><a href='#caveats'>Limits</a><a href='#next'>Next</a>"
        "</nav>")
    H = [f"<!doctype html><html><head><meta charset='utf-8'>"
         f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>RRMAP2 duration clock</title><style>{CSS}</style></head><body>"
         f"{sidebar}<div class='wrap'>"]
    H.append("<h1>A molecular clock of accumulated EAE damage</h1>")
    H.append("<p class='sub'>RRMAP2 spinal-cord Xenium · relapsing–remitting cohort · "
             f"{results['n_animals']} animals · 5101-gene panel · animal-level pseudobulk. "
             "A supervised clock that reads <b>disease duration</b> from tissue — the cumulative, "
             "irreversible axis that grows with disease history, <b>after current severity is removed</b>.</p>")
    H.append("<div class='toc'>"
             "<span class='grp'><span class='gl'>Start ·</span>"
             "<a href='#what'>The big idea</a><a href='#approach'>How to read this</a></span>"
             "<span class='grp'><span class='gl'>I ·</span>"
             "<a href='#cycle'>Two behaviours along the cycle</a></span>"
             "<span class='grp'><span class='gl'>II ·</span>"
             "<a href='#result'>The clock</a><a href='#perm'>Validation</a>"
             "<a href='#confound'>Not a confound</a><a href='#region'>Region</a></span>"
             "<span class='grp'><span class='gl'>III ·</span>"
             "<a href='#oscillation'>Which genes</a><a href='#composition'>Real upregulation?</a>"
             "<a href='#biology'>Biology</a><a href='#genes'>Signature</a>"
             "<a href='#tier2'>Where it lives</a></span>"
             "<span class='grp'><span class='gl'>IV ·</span>"
             "<a href='#chronic'>Cross-model</a><a href='#course'>What predisposes to relapse</a></span>"
             "<span class='grp'><span class='gl'>Ref ·</span>"
             "<a href='#methods'>Methods</a><a href='#caveats'>Limits</a><a href='#next'>Next</a></span></div>")

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
             "lymphoid organization, and a lipid synthesis→scavenging switch. A model-free pass over the "
             "relapse cycle recovers the same split independently — genes that oscillate with each attack "
             "and reset (severity) versus genes whose remission floor ratchets upward (duration).</div>")

    # ---------------- the big idea (plain-language primer) ----------------
    H.append("<h2 id='what'>The big idea</h2>")
    H.append("<div class='abstract'>"
             "<p><b>The setting.</b> EAE is the mouse model of multiple sclerosis: an autoimmune attack on the "
             "spinal cord. It comes in two courses — <b>relapsing–remitting</b> (attacks that flare and "
             "recede) and <b>chronic-progressive</b> (sustained worsening) — the same two courses seen in "
             "human MS. We profile spinal-cord tissue with spatial transcriptomics (Xenium, 5,101 genes) and "
             "ask what the tissue records about the disease.</p>"
             "<p><b>The problem.</b> Each mouse is sampled <b>once</b>, and given a <b>clinical score</b> for "
             "how sick it is at that moment. That score sees <i>current</i> disease activity but is largely "
             "blind to the <i>damage piled up</i> from earlier attacks. Two mice can look equally sick yet be "
             "at very different points in their disease history.</p>"
             "<p><b>The central claim.</b> Tissue carries <b>two separable molecular clocks</b>. "
             "<b>Severity</b> is acute and <i>reversible</i> — it rises in an attack and resets in remission. "
             "<b>Duration</b> is cumulative and <i>irreversible</i> — it grows with how long disease has run, "
             "regardless of how sick the animal looks today. Most of this report is about measuring the second "
             "one cleanly, separating it from the first and from technical confounds.</p>"
             "<p><b>The answer.</b> A molecular clock reads disease duration from tissue at "
             f"Spearman <b>{cb['spearman']:+.2f}</b> after severity is removed; it is not a sex/region/batch "
             "artifact; the accruing program is a glia-led, per-cell shift; and it replicates in a second EAE "
             "model and strain.</p>"
             "</div>")

    # ---------------- how to read this (conceptual approach, not parameters) ----------------
    H.append("<h2 id='approach'>How to read this — our approach</h2>")
    H.append("<div class='concept'><span class='h'>Five ideas that recur</span>"
             "<b>1 · The animal is the unit.</b> Cells within one mouse aren't independent, so every model is "
             "trained and tested with whole animals held out (no cell from a test animal is ever seen in "
             "training).<br>"
             "<b>2 · Pseudobulk.</b> We average each animal's cells into one expression profile per gene — a "
             "stable, animal-level summary.<br>"
             "<b>3 · Removing severity (“orthogonalization”).</b> To ask what tracks <i>duration</i> "
             "and not just <i>sickness</i>, we statistically subtract the clinical score from both the genes "
             "and the time label first — so nothing can score well just by reading how sick the animal is.<br>"
             "<b>4 · A clock.</b> A regression that predicts time-since-induction from the de-severitied "
             "expression. Good prediction on held-out animals = the tissue genuinely tracks duration.<br>"
             "<b>5 · Shuffle test.</b> We scramble the time labels and refit many times; if the real clock "
             "beats every shuffle, it isn't an accident of a small cohort.</div>")
    H.append("<p class='sub'>Exact estimators, parameters and tests are in <a href='#methods'>Methods</a> at "
             "the end. The findings below build in four steps: <b>I</b> the two behaviours in the raw data, "
             "<b>II</b> a clock that reads duration, <b>III</b> what that clock is actually reading, and "
             "<b>IV</b> whether it generalizes and what it means for disease course.</p>")

    # ============================= PART I =============================
    H.append(part("I", "Two clocks in one tissue"))
    H.append(concept("Before any model: do genes actually behave in two different ways over the relapse "
                     "cycle? Yes. Some flare with every attack and snap back to baseline in remission — the "
                     "<b>reversible severity</b> axis. Others never fully come back down; their baseline "
                     "creeps up attack after attack — the <b>irreversible duration</b> axis. This single "
                     "distinction is the backbone of everything that follows."))
    H.extend(part1)

    # ============================= PART II =============================
    H.append(part("II", "A clock that reads disease duration"))
    H.append(concept("Now we make the duration axis quantitative. We train a model to predict <i>how long "
                     "disease has run</i> from tissue — with current severity statistically removed — and "
                     "then spend three sections trying to break it: a shuffle test, and checks that it isn't "
                     "secretly reading sex, cord region, or batch."))

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

    # ============================= PART III =============================
    H.append(part("III", "What the clock is reading"))
    H.append(concept("The clock works — so what is it actually keying on? Four questions: <b>which genes</b> "
                     "carry the duration signal (vs the severity one), is the rise <b>real per-cell "
                     "upregulation or just more cells</b>, what is the <b>biology</b> of the accruing "
                     "program, and <b>where in the tissue</b> does it live."))

    # the two clocks at the gene level — primary view on the continuous day axis
    if decomp is not None:
        dr = decomp["results"]
        ca = dr["clock_agreement"]
        H.append("<h2 id='oscillation'>Which genes carry each clock — day vs severity</h2>")
        H.append(lead("The clock above is multivariate. This asks the same question one gene at a time, on "
                      "the same continuous axis the clock regresses on: for every gene we take its partial "
                      "correlation with <b>day-of-sacrifice with severity removed</b> (the accrual / "
                      "duration axis) and with <b>severity with day removed</b> (the oscillation / severity "
                      "axis). day and severity are near-orthogonal across animals (r = "
                      f"{dr['corr_day_score']:+.2f}), so a gene can load on either, both, or neither."))
        H.append(b64path(os.path.join(DECOMP_DIR, "figures", "day_decomposition.png"),
                         "gene-level day vs severity decomposition"))
        H.append("<div class='key'><b>The gene-level decomposition reproduces the clock and splits its "
                 "biology cleanly.</b> Of 5101 genes, "
                 f"<b>{dr['n_accrual_up']}</b> accrue with day and <b>{dr['n_accrual_down']}</b> decline "
                 f"(severity removed; {dr['n_day_sig_q05']} at day-FDR q&lt;0.05), while "
                 f"<b>{dr['n_severity_oscillating']}</b> track severity but are flat on day — they "
                 "oscillate with the relapse and reset. The day-partial correlations line up with the "
                 f"supervised clock coefficients (Spearman {ca['spearman_day_partial_vs_clock_coef']:+.2f}, "
                 f"sign-agreement {ca['sign_agreement_frac']:.0%} over {ca['n_clock_genes_overlap']} clock "
                 "genes), so the clock's weights are reading real per-gene temporal trends, not multivariate "
                 "artifacts. The declining arm is dominated by the interferon program "
                 "(<span class='gene'>Stat2, Usp18, Irgm1, Isg15, Oasl2</span>) — acute and severity-linked, "
                 "exactly the genes the clock sees receding with duration.</div>")
        H.append("<div class='cols'>")
        H.append("<div><h3>Accrue with day ↑ (severity removed)</h3>"
                 "<p class='sub'>Scar/ECM, lipid-scavenging and circadian genes that rise with elapsed time "
                 "independent of how sick the animal is. <span class='gene'>•clock</span> marks genes also "
                 "in the supervised signature.</p>"
                 + decomp_table(dr["top_accrual_up"][:18], "up") + "</div>")
        H.append("<div><h3>Decline with day ↓ (severity removed)</h3>"
                 "<p class='sub'>The acute interferon / innate response, receding as disease ages — high "
                 "early, lower late, with severity held fixed.</p>"
                 + decomp_table(dr["top_accrual_down"][:18], "down") + "</div>")
        H.append("</div>")
        H.append("<h3>Severity-oscillating — tracks the relapse, flat on day</h3>")
        H.append(lead("These genes swing with clinical severity but carry no net day trend — the gene-level "
                      "definition of the oscillating axis. Largely neuronal/synaptic genes that dip at "
                      "every peak and recover at remission, so they reset rather than accrue."))
        H.append(decomp_table(dr["top_severity_oscillating"][:14], "sev"))
        H.append("<p class='sub'>Full per-gene table (both axes, all 5101 genes): "
                 "<span class='gene'>runs/duration_gene_decomposition/gene_day_decomposition.csv</span>.</p>")

    # compositional vs cell-intrinsic accrual
    if compi is not None:
        ci = compi
        dc = ci["driver_counts"]
        exp = ci["expanding_with_duration"][:3]
        shr = ci["shrinking_with_duration"][:3]
        H.append("<h2 id='composition'>Is the accrual real upregulation, or just more cells?</h2>")
        H.append(lead("A pseudobulk gene can rise with duration for two very different reasons: the cell "
                      "type that makes it becomes more abundant (compositional — TLS B/T cells, foamy "
                      "macrophages, fibroblast-like cells) or each cell makes more of it (cell-intrinsic). "
                      "Using per-(animal × cell-type) means and cell counts, we split every accruing gene's "
                      "severity-removed day trend into the two — a shift-share decomposition that the "
                      "animal-level clock cannot do on its own."))
        H.append(b64path(os.path.join(COMPI_DIR, "figures", "composition_intrinsic.png"),
                         "compositional vs cell-intrinsic accrual"))
        H.append("<div class='good'><b>The accrual is overwhelmingly cell-intrinsic, not compositional.</b> "
                 f"Across {ci['n_accrual_genes_full']} accruing genes the median day trend is "
                 f"<b>{ci['median_intrinsic_share']:.0%} per-cell upregulation</b> vs only "
                 f"{ci['median_compositional_share']:.0%} cell-number; of the "
                 f"{ci['n_accrual_genes_classified']} strongly-accruing genes, "
                 f"<b>{dc['intrinsic']} are intrinsic-driven and just {dc['compositional']} compositional</b> "
                 f"({dc['mixed']} mixed). In the right panel almost every gene sits far below the diagonal "
                 "(intrinsic ≫ compositional). The duration clock is reading a genuine shift in cell "
                 "<i>state</i>, which matches the localization result that a clock built on cell-type "
                 "<i>proportions</i> alone is far weaker than one on expression.</div>")
        H.append("<div class='note'><b>The cell counts make the point sharper, not weaker.</b> With "
                 "severity removed, the compartments that <i>expand</i> with duration are "
                 f"{', '.join(f'{n} ({v:+.2f})' for n, v in exp)} — gliosis and vascular/fibrotic scar — "
                 f"while the inflammatory compartments <i>shrink</i> ({', '.join(f'{n} ({v:+.2f})' for n, v in shr)}). "
                 "Yet the foamy / scavenger-macrophage program (<span class='gene'>Mrc1, Folr2, Cd68, Ctsk, "
                 "Atp6v0d2</span>) still accrues strongly — so it is myeloid cells <b>changing state</b> "
                 "(becoming foamy/lipid-laden) as their numbers fall, not an influx of new ones. The few "
                 "compositional genes track the expanding astrocyte and vascular compartments.</div>")
        H.append("<h3>Top cell-intrinsic accrual genes (per-cell upregulation)</h3>")
        H.append("<table><tr><th>gene</th><th class='num'>intrinsic share</th>"
                 "<th class='num'>compositional share</th><th class='num'>r(day | score)</th></tr>")
        for r in ci["top_intrinsic_accrual"][:15]:
            H.append(f"<tr><td class='gene'>{r['gene']}</td>"
                     f"<td class='num up'>{r['intr_share']:+.2f}</td>"
                     f"<td class='num'>{r['comp_share']:+.2f}</td>"
                     f"<td class='num'>{r['r_day_total']:+.2f}</td></tr>")
        H.append("</table>")
        H.append("<p class='sub'>Shares are fractions of the severity-removed day covariance "
                 "(intrinsic + compositional + interaction = 1); a share above 1 means composition "
                 "actively <i>opposes</i> the trend — the gene accrues per-cell even as its cell type "
                 "becomes rarer. Full per-gene split: "
                 "<span class='gene'>runs/duration_composition_intrinsic/gene_composition_intrinsic.csv</span>. "
                 "Source: scripts/duration_composition_intrinsic.py.</p>")

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

    # ============================= PART IV =============================
    H.append(part("IV", "Does it generalize, and what does it mean?"))
    H.append(concept("Two payoff questions. <b>Generality:</b> is this duration program specific to one "
                     "model, or a shared feature of CNS autoimmune damage — does it replicate in a second "
                     "EAE model and strain? <b>Meaning:</b> what separates an animal that relapses from one "
                     "that doesn't — the disease-course question the whole project circles."))

    # cross-model — chronic-arm replication
    if chronic is not None:
        ccb = chronic["clock_B_severity_orthogonalized"]
        cm = chronic["cross_model"]
        dd = cm["rr_duration_vs_chronic_duration"]
        osc_c = cm.get("rr_oscillation_amp_vs_chronic_duration", {})
        H.append("<h2 id='chronic'>Cross-model — the clock replicates in the chronic arm</h2>")
        H.append(lead("The duration clock was built on RR because chronic day-of-sacrifice was thought to "
                      "alias batch. The corrected metadata shows otherwise: chronic has a <b>continuous</b> "
                      f"day axis ({chronic['day_range'][0]:.0f}–{chronic['day_range'][1]:.0f} dpi) that is "
                      "clean of run_date, so we can run the identical clock there and ask whether the "
                      "duration program is the same across two models and two strains "
                      "(RR = SJL/PLP, chronic = B6/MOG)."))
        H.append("<div class='kpis'>")
        H.append(f"<div class='kpi'><div class='v'>{ccb['spearman']:+.2f}</div>"
                 "<div class='l'>chronic clock B (LOAO)</div></div>")
        H.append(f"<div class='kpi'><div class='v'>{ccb['perm_p']:.3f}</div>"
                 "<div class='l'>chronic permutation p</div></div>")
        H.append(f"<div class='kpi'><div class='v'>{dd['spearman']:+.2f}</div>"
                 "<div class='l'>RR ↔ chronic duration axis</div></div>")
        H.append(f"<div class='kpi'><div class='v'>{chronic['batch_leakage_day_run_date_kruskal_p']:.2f}</div>"
                 "<div class='l'>day~run_date Kruskal p</div></div>")
        H.append("</div>")
        H.append(f"<div class='good'><b>The duration clock is not RR-specific.</b> Rebuilt identically on "
                 f"the chronic arm it reaches <b>Spearman {ccb['spearman']:+.2f}</b> (R² {ccb['r2_loao']:+.2f}, "
                 f"permutation p = {ccb['perm_p']:.3f}, {ccb['n_clock_genes']} genes), with day clean of batch "
                 f"(Kruskal day~run_date p = {chronic['batch_leakage_day_run_date_kruskal_p']:.2f}). And the "
                 f"two duration axes <b>agree</b>: the per-gene RR day-trend correlates with the chronic "
                 f"day-trend at <b>Spearman {dd['spearman']:+.2f}</b> ({dd['n_genes']} genes, "
                 f"{dd['sign_agreement']:.0%} sign-agreement) — across both strain and model. The agreement is "
                 "specific to the <i>duration</i> axis: the RR severity-oscillation axis does <b>not</b> "
                 f"transfer to chronic duration (Spearman {osc_c.get('spearman', float('nan')):+.2f}).</div>")
        H.append(b64path(os.path.join(CHRONIC_DIR, "figures", "rr_vs_chronic_duration.png"),
                         "RR vs chronic duration axis"))
        H.append(f"<div class='note'><b>One revealing difference.</b> In chronic, predicting day from "
                 f"<b>severity alone</b> is <i>positive</i> (Spearman {chronic['severity_only_baseline_spearman']:+.2f}) "
                 "— the disease is monotonically progressive, so later really is sicker. In RR the same "
                 f"baseline is <i>negative</i> ({results['severity_only_baseline_spearman']:+.2f}) because each "
                 "relapse resets acute severity. Same accruing program underneath; opposite severity–time "
                 "coupling on top — which is exactly the distinction between a relapsing and a progressive "
                 "course.</div>")
        H.append("<div class='cols'>")
        H.append("<div><h3>Conserved accrual — rises with duration in both</h3>"
                 f"<p class='sub'>{cm['n_conserved_accrue']} genes accrue with duration in both models "
                 "(r(day|score) ≥ 0.30 each). Scar/ECM and Wnt: "
                 "<span class='gene'>Igf2, Vtn, Wnt5a, Wnt6, Id4, Prelp</span>.</p>"
                 + _chronic_table(cm["top_conserved_accrue"][:15], "accrue") + "</div>")
        H.append("<div><h3>Conserved decline — falls with duration in both</h3>"
                 f"<p class='sub'>{cm['n_conserved_decline']} genes decline in both — led by the "
                 "cholesterol-synthesis program <span class='gene'>Hmgcr, Idi1, Msmo1, Lss, Hsd17b7, Ldlr</span>, "
                 "the same module the RR clock loses.</p>"
                 + _chronic_table(cm["top_conserved_decline"][:15], "decline") + "</div>")
        H.append("</div>")
        H.append("<p class='sub'><b>Model-specific accrual.</b> Chronic-only (B6/MOG): "
                 "<span class='gene'>" + ", ".join(g["gene"] for g in cm["chronic_specific_accrue"][:10])
                 + "</span> — more complement/inflammatory. RR-only (SJL/PLP): <span class='gene'>"
                 + ", ".join(g["gene"] for g in cm["rr_specific_accrue"][:10]) + "</span>. "
                 "Full tables: <span class='gene'>runs/duration_clock_chronic/</span>. "
                 "Source: scripts/duration_clock_chronic.py.</p>")

    # within-strain — what predisposes to relapse? (REM1-anchored trajectories)
    if mono is not None:
        t = mono["timing"]; ax = mono.get("axis_alignment", {})
        mt = mono["module_trajectories"]
        def mv(name, grp):
            return mt.get(name, {}).get(grp, float("nan"))
        H.append("<h2 id='course'>Within-strain — what predisposes to relapse?</h2>")
        H.append("<div class='note'><b>Exploratory, n = 5 / 4 / 4.</b> Conserved duration does <i>not</i> "
                 "mean there are no relapse-specific markers — and the cross-model contrast can't address "
                 "course (RR vs chronic is strain-confounded). The identifiable cut is <i>within</i> "
                 "SJL/PLP. By design <b>REMISSION1</b> (~day 22) branches into <b>MONOPHASIC</b> "
                 "(~day 33, did not relapse) and <b>PEAK2</b> (~day 32, relapsed): monophasic animals are "
                 "sampled at the same timepoint a relapser hits its second attack (day MW p = "
                 f"{t['MONO_vs_PEAK2_day_MW_p']:.2f}) but stay low-severity "
                 f"({t['MONO']['score_mean']:.2f} vs {t['PEAK2']['score_mean']:.2f}). So the time-matched "
                 "contrast is MONO vs PEAK2, and the relapse decision is the divergence of the two "
                 "trajectories from the shared REM1 origin. Cross-sectional (different animals; we cannot "
                 "label a REM1 animal's future), n tiny — effect sizes and direction only.</div>")
        H.append(b64path(os.path.join(MONO_DIR, "figures", "monophasic_vs_relapsing.png"),
                         "REM1 -> monophasic vs peak2 trajectories"))
        H.append("<div class='key'><b>Relapse re-ignites the acute program; not-relapsing resolves and "
                 "keeps accruing quietly.</b> From the REM1 origin, the <b>relapse</b> path (→PEAK2) turns "
                 f"the acute-oscillator module back on (z {mv('acute-oscillator','REM1'):+.2f}→"
                 f"{mv('acute-oscillator','PEAK2'):+.2f}), with innate/IFN and ECM/scar, and <b>loses "
                 f"neuron/myelin again</b> ({mv('neuro/myelin','REM1'):+.2f}→{mv('neuro/myelin','PEAK2'):+.2f}). "
                 "The <b>monophasic</b> path (→MONO) does the opposite: the acute oscillator stays off "
                 f"({mv('acute-oscillator','MONO'):+.2f}), <b>neuron/myelin recovers</b> "
                 f"(→{mv('neuro/myelin','MONO'):+.2f}), the glucocorticoid/stress tone resolves "
                 f"({mv('GC/stress','REM1'):+.2f}→{mv('GC/stress','MONO'):+.2f}), and the quiet "
                 "duration/repair accrual continues — the monophasic-specific genes are the conserved "
                 "accrual set <span class='gene'>Igf2, Fmod, Vtn</span> plus circadian <span class='gene'>"
                 "Nr1d1</span>. The relapse-specific genes are the acute myeloid/oscillator program "
                 "(<span class='gene'>Gpnmb, Chil1, Arg1, Timp1, Hal, Socs3</span>).</div>")
        if ax:
            H.append("<p class='sub'>The relapse-specific direction (PEAK2 − MONO) aligns with the acute "
                     f"<b>severity/oscillation</b> axis (Spearman {ax['divergence_vs_oscillation_amp']:+.2f} "
                     f"vs oscillation amplitude, {ax['divergence_vs_severity_axis']:+.2f} vs the score-axis) "
                     f"and runs <i>opposite</i> the duration axis ({ax['divergence_vs_duration_axis']:+.2f}). "
                     "So relapse = re-running the acute attack program; staying monophasic = resolving it "
                     "and continuing the slow accrual/repair trajectory.</p>")
        H.append("<p class='sub'><b>Honest reading.</b> PEAK2 is a severity peak by construction, so "
                 "“relapse-specific” genes are <i>expected</i> to be the acute severity program — "
                 "the informative half is the monophasic trajectory (myelin recovery + stress resolution + "
                 "quiet accrual). And because the data are terminal/cross-sectional, this is a "
                 "population-level trajectory, not proof that anything <i>at</i> REM1 decides the outcome; "
                 "that would need a longitudinal or pre-relapse design. Full table: "
                 "<span class='gene'>runs/monophasic_vs_relapsing/rem1_trajectory_genes.csv</span>. "
                 "Source: scripts/monophasic_vs_relapsing.py.</p>")

    # ============================= REFERENCE =============================
    H.append(part("Ref", "Methods, limits & next steps"))
    H.extend(methods_section(results))

    # caveats
    H.append("<h2 id='caveats'>Honest limits</h2>")
    H.append("<p class='sub'>Both cohorts are small (RR n = 33, chronic n = 34) animal-level pseudobulk "
             "(spatial information collapsed) — LOAO + permutation null guard against over-fit but the "
             "cohorts are small. The RR sex cohort is imbalanced (7 vs 26) though orthogonal to day. The "
             "clock is a supervised regression onto a de-confounded temporal <i>label</i>, not a "
             "within-animal time course — there are no per-animal sequences in terminal cross-sectional "
             "data. The RR↔chronic comparison is of within-model gene slopes (the cross-strain level "
             "offset is not identifiable); chronic day is clean of batch but the chronic cohort still spans "
             "two run_dates, controlled only by their both carrying the full day range.</p>")

    # next
    H.append("<h2 id='next'>Next</h2>")
    H.append("<p class='sub'>The localization shows the clock is glia-led and state-driven, but the compartment "
             "clocks still aggregate each cell type to one number per animal. The finer follow-ups: "
             "(1) a within-lesion version using the radial core→rim→margin zones (does duration concentrate in "
             "the scar core?); (2) validation in the chronic arm with matched-batch sampling to break the "
             "day~run_date confound; (3) a publication-grade null (N_PERM≈1000) on the headline clock.</p>")

    H.append("<div class='foot'>Immunoformer · RRMAP2 duration clock · generated by "
             "scripts/make_duration_clock_report.py from runs/duration_clock/, "
             "runs/duration_gene_decomposition/, runs/rr_cycle_oscillation/. "
             "Source: docs/rrmap2-duration-clock.md · scripts/duration_clock.py, "
             "duration_clock_controls.py, duration_gene_decomposition.py, "
             "duration_composition_intrinsic.py, duration_clock_chronic.py, "
             "monophasic_vs_relapsing.py, rr_cycle_oscillation.py.</div>")
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

    # optional relapse-cycle oscillation context (degrade gracefully if cache absent)
    osc = None
    if os.path.exists(OSC_CACHE) and os.path.exists(OSC_METRICS):
        oc = np.load(OSC_CACHE, allow_pickle=True)
        metrics = pd.read_csv(OSC_METRICS)
        fig_oscillation(oc["pb"], oc["genes"].astype(str), oc["stage"].astype(str), metrics)
        osc = {"metrics": metrics}
        print("[figures] 6 panels (incl. oscillation) ->", FIG, flush=True)
    else:
        print(f"[figures] 5 panels -> {FIG}  (oscillation cache absent, section skipped)",
              flush=True)

    # gene-level day-vs-severity decomposition (primary two-clock view)
    decomp = None
    dr_path = os.path.join(DECOMP_DIR, "results.json")
    if os.path.exists(dr_path):
        decomp = {"results": json.load(open(dr_path))}
        print("[decomp] day-stratified gene decomposition loaded", flush=True)
    else:
        print(f"[decomp] {dr_path} absent — run scripts/duration_gene_decomposition.py; "
              "section skipped", flush=True)

    # compositional vs cell-intrinsic accrual
    compi = None
    ci_path = os.path.join(COMPI_DIR, "results.json")
    if os.path.exists(ci_path):
        compi = json.load(open(ci_path))
        print("[compi] composition-vs-intrinsic decomposition loaded", flush=True)
    else:
        print(f"[compi] {ci_path} absent — run scripts/duration_composition_intrinsic.py; "
              "section skipped", flush=True)

    # chronic-arm replication + cross-model relating
    chronic = None
    ch_path = os.path.join(CHRONIC_DIR, "results.json")
    if os.path.exists(ch_path):
        chronic = json.load(open(ch_path))
        print("[chronic] chronic clock + cross-model relating loaded", flush=True)
    else:
        print(f"[chronic] {ch_path} absent — run scripts/duration_clock_chronic.py; "
              "section skipped", flush=True)

    # within-strain monophasic vs relapsing (course question)
    mono = None
    mo_path = os.path.join(MONO_DIR, "results.json")
    if os.path.exists(mo_path):
        mono = json.load(open(mo_path))
        print("[mono] monophasic-vs-relapsing loaded", flush=True)
    else:
        print(f"[mono] {mo_path} absent — run scripts/monophasic_vs_relapsing.py; section skipped",
              flush=True)

    html = build_html(results, controls, coefs, osc=osc, decomp=decomp, compi=compi,
                      chronic=chronic, mono=mono)
    with open(REPORT, "w") as fh:
        fh.write(html)
    print(f"[done] -> {REPORT}", flush=True)


if __name__ == "__main__":
    main()
