"""Build two self-contained HTML reports from the saved run outputs.

  runs/reports/performance.html  — Stage-1 baseline, transfer benchmark, RR-vs-chronic
  runs/reports/findings.html     — within-RR relapse-cycle biology (the main result)

Figures are regenerated as PNG and inlined as base64 (no external files), reading
only the small JSON results + the cached pseudobulk.npz — no h5ad reload.

    python scripts/make_report.py
"""

from __future__ import annotations

import base64
import io
import json
import os

import numpy as np
from scipy.stats import rankdata, spearmanr
from scipy.stats import t as tdist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "runs", "reports")
CYCLE_ORDER = ["PLP CFA", "ONSET1", "PEAK1", "REMISSION1",
               "ONSET2", "PEAK2", "REMISSION2", "PEAK3"]

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
img{max-width:100%;border-radius:8px;margin:8px 0}
.note{background:#fff8f0;border-left:3px solid var(--warn);padding:10px 14px;border-radius:0 8px 8px 0;font-size:13.5px;margin:12px 0}
.good{background:#f0faf2;border-left:3px solid var(--good);padding:10px 14px;border-radius:0 8px 8px 0;font-size:13.5px;margin:12px 0}
.gene{font-family:"SF Mono",Menlo,Consolas,monospace;font-size:12.5px}
.foot{color:var(--mut);font-size:12px;margin-top:40px;border-top:1px solid var(--line);padding-top:14px}
.pill{display:inline-block;background:#eef;color:#3b5bdb;border-radius:20px;padding:2px 10px;font-size:12px;font-weight:600}
"""


def img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return '<img src="data:image/png;base64,%s">' % base64.b64encode(buf.getvalue()).decode()


def load(*parts):
    p = os.path.join(ROOT, *parts)
    if not os.path.exists(p):
        return None
    if p.endswith(".json"):
        return json.load(open(p))
    if p.endswith(".npz"):
        return np.load(p, allow_pickle=True)
    return None


def page(title, body):
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{title}</title><style>{CSS}</style></head><body><div class='wrap'>{body}"
            f"<div class='foot'>Immunoformer · RRMAP2 spinal-cord EAE · generated from runs/ outputs. "
            f"Cross-sectional terminal data (one stage per animal); animal-level analyses.</div>"
            f"</div></body></html>")


def kpi(v, l):
    return f"<div class='kpi'><div class='v'>{v}</div><div class='l'>{l}</div></div>"


def _interaction_from_pb(npz):
    """Recompute the score:model interaction (RR_slope - chronic_slope) per gene."""
    pb, genes = npz["pb"], npz["genes"].astype(str)
    score, model = npz["score"].astype(float), npz["model"].astype(float)
    keep = ~np.isnan(score)
    Y, s, m = pb[keep], score[keep], model[keep]
    sc = s - s.mean()
    X = np.column_stack([np.ones_like(sc), sc, m, sc * m])
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ Y
    resid = Y - X @ beta
    sigma2 = (resid ** 2).sum(0) / (n - k)
    se = np.sqrt(np.outer(np.diag(XtXi), sigma2))
    pv = 2 * tdist.sf(np.abs(beta / se), n - k)[3]
    eff = beta[3]
    nonconst = Y.std(0) > 0
    q = np.full(len(genes), np.nan)
    pin = pv[nonconst]; o = np.argsort(pin); rk = np.empty(len(pin)); rk[o] = np.arange(1, len(pin) + 1)
    qq = pin * len(pin) / rk; qq[o] = np.minimum.accumulate(qq[o][::-1])[::-1]
    q[nonconst] = np.clip(qq, 0, 1)
    return genes, eff, q, nonconst


# ---------------------------------------------------------------- PERFORMANCE
def build_performance():
    stage1 = load("runs", "rrmap2_stage", "best_metrics.json")
    rrc = load("runs", "rr_vs_chronic", "results.json")
    transfer = load("runs", "transfer_experiment", "transfer_results.json")
    b = []
    b.append("<h1>Immunoformer — model performance</h1>")
    b.append("<p class='sub'>Attention-MIL on Xenium section bags. Three tasks: ordinal "
             "disease-stage regression, cross-etiology transfer, and RR-vs-chronic classification.</p>")

    # Stage-1 ordinal baseline
    b.append("<h2>1 · Stage-1 baseline — ordinal disease stage (held-out animals)</h2>")
    if stage1:
        b.append("<div class='kpis'>"
                 + kpi(f"{stage1['spearman']:.2f}", "Spearman ρ")
                 + kpi(f"{stage1['mae']:.2f}", "MAE (stage units)")
                 + kpi(f"{stage1['n']}", "held-out val bags")
                 + kpi("17", "ordinal stages") + "</div>")
        b.append("<p class='sub'>Leakage-free split over animals (<span class='gene'>sample_name</span>); "
                 "PCA(64) cell encoder + gated attention + CORAL head. A genuine within-domain stage signal.</p>")

    # RR vs chronic classifier
    b.append("<h2>2 · RR vs chronic classifier — and why it's a cautionary result</h2>")
    if rrc:
        r = rrc["results"]
        b.append("<div class='kpis'>"
                 + kpi(f"{r['auc_overall']:.3f}", "overall AUC")
                 + kpi(f"{r['auc_by_region'].get('L', float('nan')):.3f}", "lumbar-only AUC")
                 + kpi(f"{r['auc_region_only_baseline']:.2f}", "region-only baseline")
                 + kpi(f"{r['n_bags']}", "bags (animal-grouped CV)") + "</div>")
        # AUC by region figure
        regs = sorted(r["auc_by_region"])
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.bar(range(len(regs)), [r["auc_by_region"][x] for x in regs],
               color=["#3b5bdb" if x == "L" else "#c2c8d6" for x in regs])
        ax.axhline(r["auc_region_only_baseline"], color="#c0392b", ls=":", lw=1.5,
                   label=f"region-only {r['auc_region_only_baseline']:.2f}")
        ax.axhline(0.5, color="#999", lw=.8)
        ax.set_xticks(range(len(regs)))
        ax.set_xticklabels([f"{x}\nC{r['region_counts'][x]['chronic']}/R{r['region_counts'][x]['RR']}" for x in regs])
        ax.set_ylim(0, 1.05); ax.set_ylabel("OOF AUC"); ax.legend(fontsize=8)
        ax.set_title("RR vs chronic — AUC by spinal region")
        ax.spines[["top", "right"]].set_visible(False)
        b.append("<div class='card'>" + img(fig) + "</div>")
        b.append("<div class='good'><b>What it answers:</b> the separation is <b>not</b> the L→T→C "
                 "inflammation gradient — lumbar alone (balanced models) is still "
                 f"{r['auc_by_region'].get('L', float('nan')):.3f}, vs only "
                 f"{r['auc_region_only_baseline']:.2f} for a region-only model.</div>")
        b.append("<div class='note'><b>Why it's cautionary:</b> AUC ≈ 1.0 is a red flag. "
                 "All 54 Xenium slides are model-pure (RR and chronic never share a slide; ~3 run dates), "
                 "and the two models are different strains/antigens (SJL/PLP vs B6/MOG). So the label is "
                 "<b>100% confounded with batch + strain</b>. The classifier almost certainly reads "
                 "batch/strain identity, not relapse biology. We excluded the region confound; we cannot "
                 "exclude batch/strain. → The relapse question must be asked <b>within</b> the RR cohort.</div>")

    # Transfer benchmark
    b.append("<h2>3 · Cross-etiology transfer benchmark</h2>")
    if transfer:
        wd = stage1["spearman"] if stage1 else float("nan")
        rows = [("RRMAP2 (held-out, within-domain)", wd, stage1["n"] if stage1 else "", "reference")]
        for d in transfer["per_dataset"]:
            if d["tag"] == "rrmap2_val":
                continue
            rows.append((d["tag"], d["rho"], d["n_bags"], f"p={d['p_permutation']:.2f}"))
        th = "<table><tr><th>dataset</th><th>Spearman ρ</th><th>bags</th><th>note</th></tr>"
        for name, rho, n, note in rows:
            th += f"<tr><td>{name}</td><td class='num'>{rho:+.3f}</td><td class='num'>{n}</td><td class='sub'>{note}</td></tr>"
        th += "</table>"
        b.append("<div class='card'>" + th + "</div>")
        b.append("<div class='note'>Neither cross-etiology target transfers (optic-nerve ρ≈0, mtDNA-DSB ρ≈0.2, "
                 "n=12 — both underpowered/null). Within-domain shown is the corrected held-out ρ; the "
                 "transfer script's own within-domain eval was contaminated by training bags and has since "
                 "been fixed to score only held-out animals.</div>")
    return page("Immunoformer — performance", "\n".join(b))


# ---------------------------------------------------------------- FINDINGS
def _trends_from_pb(npz):
    """Recompute raw + severity-adjusted Spearman vs cycle rank from cached pseudobulk."""
    pb, genes = npz["pb"], npz["genes"].astype(str)
    rank, score = npz["cycle_rank"].astype(float), npz["score"].astype(float)
    m = (rank >= 0) & ~np.isnan(score)
    sub, r, s = pb[m], rankdata(rank[m]), rankdata(score[m])

    def zc(v):
        v = v - v.mean(); sd = v.std(); return v / sd if sd > 0 else v * 0
    zr, zs = zc(r), zc(s)
    r_rs = float(zr @ zs / len(r))
    gr = np.apply_along_axis(rankdata, 0, sub)
    zx = (gr - gr.mean(0)) / (gr.std(0) + 1e-12)
    raw = zx.T @ zr / len(r)
    r_xs = zx.T @ zs / len(r)
    partial = (raw - r_xs * r_rs) / np.sqrt(np.clip((1 - r_xs ** 2) * (1 - r_rs ** 2), 1e-12, None))
    nonconst = sub.std(0) > 0
    return genes, raw, partial, nonconst, r_rs, int(m.sum()), pb, npz["stage"].astype(str), rank


def build_findings():
    res = load("runs", "rr_within_relapse", "results.json")
    npz = load("runs", "rr_within_relapse", "pseudobulk.npz")
    b = []
    b.append("<h1>RRMAP2 — main biological findings</h1>")
    b.append("<p class='sub'>Animal-level, descriptive. Two questions: (A) what changes across the "
             "<b>relapse cycle</b> within the relapse-remitting cohort, and (B) how the <b>relapsing and "
             "chronic courses differ</b> in their disease dynamics. Both avoid the batch/strain confound that "
             "makes a naive RR-vs-chronic comparison uninterpretable.</p>")

    if not res:
        return page("Findings", "\n".join(b) + "<p>missing results</p>")
    b.append("<h2>A · Within the relapse-remitting cohort — the relapse cycle</h2>")
    b.append("<p class='sub'>Within RR, disease stage is de-confounded from slide (each stage on 9–12 slides), "
             "so these are not batch artifacts. The relapse cycle is read as a pseudo-trajectory across animals.</p>")

    sa = res["trajectory_severity_adjusted"]
    b.append("<div class='kpis'>"
             + kpi(f"{res['n_rr_animals']}", "RR animals")
             + kpi(f"{sa['n_sig_raw_q05']}", "cycle-trend genes (q<.05)")
             + kpi(f"{sa['n_sig_partial_q05']}", "survive severity adj.")
             + kpi(f"{sa['cycle_vs_severity_rho']:+.2f}", "cycle vs severity ρ") + "</div>")

    b.append("<div class='good'><b>Headline:</b> across the relapse cycle, a "
             "<b>disease-associated-microglia + complement</b> program rises while "
             "<b>cholesterol-biosynthesis / myelination</b> falls. After adjusting for clinical severity, "
             "the microglia/complement rise is the robust, cycle-<i>specific</i> signal (it even strengthens); "
             "the myelin decline is real but partly tracks severity.</div>")

    # figure 1: trajectory heatmap from pseudobulk
    genes, raw, partial, nonconst, r_rs, n_used, pb, stage, rank = _trends_from_pb(npz)
    order = np.argsort(raw)
    top_idx = np.concatenate([order[:15], order[-15:]])
    present = [s for s in CYCLE_ORDER if s in set(stage[rank >= 0])]
    M = np.zeros((len(top_idx), len(present)))
    for cj, st in enumerate(present):
        mm = stage == st
        M[:, cj] = pb[np.ix_(mm, top_idx)].mean(0) if mm.sum() else np.nan
    M = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(max(5, len(present) * .85), 7.5))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(present))); ax.set_xticklabels(present, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(top_idx))); ax.set_yticklabels(genes[top_idx], fontsize=7)
    ax.set_title("Top relapse-cycle genes (z-scored stage means)")
    fig.colorbar(im, ax=ax, fraction=.024, label="z")
    b.append("<h2>The relapse-cycle trajectory</h2>")
    b.append("<div class='card'>" + img(fig) + "</div>")

    # figure 2: raw vs severity-adjusted scatter
    df = n_used - 3
    tstat = partial * np.sqrt(df / np.clip(1 - partial ** 2, 1e-12, None))
    pval = 2 * tdist.sf(np.abs(tstat), df)
    from numpy import argsort
    q = np.full(len(pval), np.nan)
    idx = np.where(nonconst)[0]
    pv = pval[idx]; o = argsort(pv); ranks_ = np.empty(len(pv)); ranks_[o] = np.arange(1, len(pv) + 1)
    qv = pv * len(pv) / ranks_
    qv[o] = np.minimum.accumulate(qv[o][::-1])[::-1]
    q[idx] = np.clip(qv, 0, 1)
    surv = (q < 0.05)
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.scatter(raw[nonconst & ~surv], partial[nonconst & ~surv], s=6, c="#c8ccd4", label="ns after adj.")
    ax.scatter(raw[surv], partial[surv], s=11, c="#3b5bdb", label="cycle-specific (q<.05)")
    gi = {g: i for i, g in enumerate(genes)}
    for g in ["Hmgcr", "Msmo1", "Idi1", "Ldlr", "C6", "Igf1", "Fcrls", "Igkc", "Cd47"]:
        if g in gi:
            ax.annotate(g, (raw[gi[g]], partial[gi[g]]), fontsize=7.5)
    ax.plot([-1, 1], [-1, 1], "k--", lw=.8); ax.axhline(0, c="#bbb", lw=.5); ax.axvline(0, c="#bbb", lw=.5)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_xlabel("raw cycle ρ"); ax.set_ylabel("severity-adjusted partial ρ")
    ax.set_title("Raw vs severity-adjusted (off-diagonal = severity-driven)")
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    b.append("<h2>Severity adjustment — what's cycle-specific vs just 'sicker'</h2>")
    b.append("<div class='card'>" + img(fig) + "</div>")

    # gene tables (from results.json severity-adjusted)
    def gene_table(rows, col, label):
        h = f"<table><tr><th>gene</th><th>{label}</th><th>BH q</th></tr>"
        for r in rows[:10]:
            cls = "up" if r[col] > 0 else "down"
            h += (f"<tr><td class='gene'>{r['name']}</td>"
                  f"<td class='num {cls}'>{r[col]:+.3f}</td>"
                  f"<td class='num'>{r['q']:.2g}</td></tr>")
        return h + "</table>"

    b.append("<h3>Cycle-specific genes (survive severity adjustment)</h3>")
    b.append("<div style='display:flex;gap:18px;flex-wrap:wrap'>"
             "<div style='flex:1 1 320px'><b>Rising — microglia / complement</b>"
             + gene_table(sa["increasing_cycle_specific"], "partial_rho", "partial ρ") + "</div>"
             "<div style='flex:1 1 320px'><b>Falling</b>"
             + gene_table(sa["decreasing_cycle_specific"], "partial_rho", "partial ρ") + "</div></div>")

    # cholesterol raw->adjusted
    b.append("<h3>Cholesterol-biosynthesis genes: how much was just severity?</h3>")
    ch = "<table><tr><th>gene</th><th>raw ρ</th><th>raw q</th><th>partial ρ</th><th>partial q</th><th>verdict</th></tr>"
    for c in sa["cholesterol_genes"]:
        if c["raw_rho"] is None:
            continue
        survd = c["partial_q"] < 0.05
        v = "<span class='pill' style='background:#f0faf2;color:#2f9e44'>cycle-specific</span>" if survd \
            else "<span class='pill' style='background:#fff0e8;color:#e8590c'>severity-driven</span>"
        ch += (f"<tr><td class='gene'>{c['gene']}</td><td class='num down'>{c['raw_rho']:+.3f}</td>"
               f"<td class='num'>{c['raw_q']:.2g}</td><td class='num down'>{c['partial_rho']:+.3f}</td>"
               f"<td class='num'>{c['partial_q']:.2g}</td><td>{v}</td></tr>")
    ch += "</table>"
    b.append("<div class='card'>" + ch + "</div>")
    b.append("<div class='note'>The core sterol genes (Msmo1, Idi1, Ldlr) stay cycle-specific, but the "
             "rate-limiting <span class='gene'>Hmgcr</span> and <span class='gene'>Lss/Hsd17b7</span> drop below "
             "significance after adjustment — so the myelin-metabolism decline is partly a severity correlate, "
             "not pure cycle accumulation.</div>")

    # mono vs relapsed
    if "mono_vs_relapsed_genes" in res:
        mr = res["mono_vs_relapsed_genes"]
        b.append("<h2>Monophasic vs relapsing RR — exploratory</h2>")
        b.append("<p class='sub'>The cleanest 'relapse or not' contrast (same strain), but only 4 monophasic "
                 "vs 15 relapsed and monophasic animals sit on one run date — <b>0 genes survive FDR</b>. "
                 "Direction only, hypothesis-generating.</p>")
        def mr_list(rows):
            return ", ".join(f"<span class='gene'>{r['name']}</span>" for r in rows[:8])
        b.append("<div class='card'><b>Nominally up in relapsed:</b> " + mr_list(mr["up_in_relapsed"])
                 + " <span class='sub'>(foamy-macrophage / phagocyte)</span><br><br>"
                 "<b>Nominally up in monophasic:</b> " + mr_list(mr["up_in_monophasic"])
                 + " <span class='sub'>(myelin / neuronal — less damage)</span></div>")

    b.append("<div class='note'><b>Caveats:</b> cycle order is assumed "
             "(PLP CFA &lt; ONSET1 &lt; PEAK1 &lt; REMISSION1 &lt; ONSET2 &lt; PEAK2 &lt; REMISSION2 &lt; PEAK3 — "
             "confirm PEAK1/2/3 are sequential); cross-sectional pseudo-trajectory across animals, not "
             "within-animal progression; associational, not causal.</div>")

    # ---- Part B: RR vs chronic differential dynamics
    mdiff = load("runs", "model_difference", "results.json")
    mnpz = load("runs", "model_difference", "pseudobulk_both.npz")
    if mdiff and mnpz is not None:
        b.append("<h2>B · Relapsing vs chronic — how the two courses differ</h2>")
        b.append("<p class='sub'>A naive RR-vs-chronic comparison reads strain/batch (the two cohorts are "
                 "different strains on different slides). The de-confounded question is whether genes change "
                 "with severity (<span class='gene'>score_sacrifice</span>) <i>differently</i> between courses "
                 "— the score×model interaction, where a constant strain/batch offset cancels in the slope.</p>")
        b.append("<div class='kpis'>"
                 + kpi(f"{mdiff['model_main_sig_q05']:,}", "baseline DE genes (strain/batch)")
                 + kpi(f"{mdiff['interaction_sig_q05']}", "differential-dynamics genes")
                 + kpi(f"{mdiff['n_rr']}/{mdiff['n_chronic']}", "RR / chronic animals") + "</div>")
        b.append("<div class='good'>A naive comparison calls "
                 f"<b>{mdiff['model_main_sig_q05']:,} of 5,101 genes</b> 'different' — but that is the "
                 "strain/batch baseline (the same confound the classifier read). The interaction leaves only "
                 f"<b>{mdiff['interaction_sig_q05']} genes</b>: the two courses' disease <i>dynamics</i> are "
                 "largely shared; they diverge mostly in baseline (strain) identity.</div>")
        genes, eff, q, nonconst = _interaction_from_pb(mnpz)
        ql = -np.log10(np.clip(q, 1e-12, 1)); sig = q < 0.05
        fig, ax = plt.subplots(figsize=(5.8, 4.4))
        ax.scatter(eff[nonconst & ~sig], ql[nonconst & ~sig], s=6, c="#c8ccd4")
        ax.scatter(eff[sig], ql[sig], s=13, c="#3b5bdb")
        ax.axhline(-np.log10(0.05), color="#c0392b", ls=":", lw=1)
        gi = {g: i for i, g in enumerate(genes)}
        for g in ["Hal", "Cd74", "H2-Aa", "C4b", "Cx3cr1", "Cst7", "Arg1", "Chil3", "Gpnmb", "Lag3"]:
            if g in gi and not np.isnan(q[gi[g]]):
                ax.annotate(g, (eff[gi[g]], -np.log10(max(q[gi[g]], 1e-12))), fontsize=7.5)
        ax.set_xlabel("interaction effect   (← steeper in chronic    |    steeper in RR →)")
        ax.set_ylabel("-log10 BH q"); ax.set_title("Differential severity dynamics: RR vs chronic")
        ax.spines[["top", "right"]].set_visible(False)
        b.append("<div class='card'>" + img(fig) + "</div>")

        def gtab(rows, title):
            h = f"<div style='flex:1 1 330px'><b>{title}</b><table><tr><th>gene</th><th>RR−chr slope</th><th>q</th></tr>"
            for r in rows[:8]:
                cls = "up" if r["inter_effect"] > 0 else "down"
                h += (f"<tr><td class='gene'>{r['gene']}</td>"
                      f"<td class='num {cls}'>{r['inter_effect']:+.3f}</td>"
                      f"<td class='num'>{r['inter_q']:.2g}</td></tr>")
            return h + "</table></div>"
        b.append("<div style='display:flex;gap:18px;flex-wrap:wrap'>"
                 + gtab(mdiff["steeper_in_chronic"], "Steeper in CHRONIC — MHC-II / complement / DAM / T-cell")
                 + gtab(mdiff["steeper_in_RR"], "Steeper in RR — M2 / repair-macrophage / lipid") + "</div>")
        b.append("<div class='note'>Mostly sub-FDR individually (only <span class='gene'>Hal</span> clears it), so "
                 "read these as suggestive modules. The interaction cancels a <i>constant</i> strain/batch offset "
                 "but not batch that correlates with severity within a cohort; n=67, limited power. Notably this "
                 "<b>inverts the classifier's story</b>: near-identical to a classifier (batch/strain), yet disease "
                 "dynamics are mostly shared.</div>")

    return page("Immunoformer — findings", "\n".join(b))


def build_did():
    res = load("runs", "onset_peak_did", "results.json")
    npz = load("runs", "onset_peak_did", "did_full.npz")
    b = []
    b.append("<h1>RR vs chronic — onset→peak transition &amp; phase-dependence</h1>")
    b.append("<p class='sub'>Does the apparent model difference depend on which disease phase you look at? "
             "We contrast the two courses at matched phases via a difference-in-differences (DiD): "
             "<b>(RR<sub>peak</sub> − RR<sub>onset</sub>) − (chronic<sub>peak</sub> − chronic<sub>onset</sub>)</b>. "
             "Each within-model change is batch-clean, and subtracting them cancels the constant strain/batch "
             "offset — so this isolates how the onset→peak transition differs between courses.</p>")
    if not res or npz is None:
        return page("Phase-dependence", "\n".join(b) + "<p>missing results</p>")
    gp = res["groups"]
    b.append("<div class='kpis'>"
             + kpi(f"{res['peak1_single_contrast_sig_q05']}", "‘different’ at matched PEAK1 (the trap)")
             + kpi(f"{res['did_sig_q05']}", "DiD genes at FDR")
             + kpi(f"{gp['rr_onset']}", "RR onset animals (limit)")
             + kpi(f"{gp['chr_onset']}/{gp['chr_peak']}/{gp['rr_onset']}/{gp['rr_peak']}", "chrOn/chrPk/rrOn/rrPk") + "</div>")
    b.append("<div class='note'><b>The trap:</b> even a single <i>matched</i> PEAK1 RR-vs-chronic contrast calls "
             f"<b>{res['peak1_single_contrast_sig_q05']} genes</b> different — that is still strain/batch "
             "(same strains, different slides). Matching the phase does NOT de-confound; only the DiD does.</div>")
    b.append("<div class='note'><b>Underpowered:</b> the clean DiD finds "
             f"<b>{res['did_sig_q05']} genes at FDR</b> — not because there's no difference, but because RR onset "
             f"(ONSET1) is only <b>{gp['rr_onset']} animals</b>, which inflates the interaction's error. Everything "
             "below is <b>directional / hypothesis-generating only</b>.</div>")

    genes = npz["genes"].astype(str); did = npz["did"]; q = npz["q"]
    valid = ~np.isnan(q); ql = -np.log10(np.clip(q, 1e-12, 1)); sig = q < 0.05
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.scatter(did[valid & ~sig], ql[valid & ~sig], s=6, c="#c8ccd4")
    if sig.any():
        ax.scatter(did[sig], ql[sig], s=13, c="#3b5bdb")
    ax.axhline(-np.log10(0.05), color="#c0392b", ls=":", lw=1)
    gi = {g: i for i, g in enumerate(genes)}
    o = np.argsort(did[valid])
    for j in np.where(valid)[0][o][:6].tolist() + np.where(valid)[0][o][-6:].tolist():
        ax.annotate(genes[j], (did[j], ql[j]), fontsize=7)
    ax.set_xlabel("DiD   (← stronger onset→peak in chronic    |    stronger in RR →)")
    ax.set_ylabel("-log10 BH q"); ax.set_title("Onset→peak transition: RR vs chronic (DiD)")
    ax.spines[["top", "right"]].set_visible(False)
    b.append("<div class='card'>" + img(fig) + "</div>")

    # example onset->peak lines for top DiD genes (both directions)
    examples = [r["gene"] for r in res["stronger_onset_to_peak_in_RR"][:2]] + \
               [r["gene"] for r in res["stronger_onset_to_peak_in_chronic"][:2]]
    means = {k: npz[k] for k in ["chr_onset", "chr_peak", "rr_onset", "rr_peak"]}
    fig, axes = plt.subplots(1, len(examples), figsize=(3.2 * len(examples), 3.0))
    for ax, gene in zip(np.atleast_1d(axes), examples):
        j = gi[gene]
        ax.plot([0, 1], [means["chr_onset"][j], means["chr_peak"][j]], "-o", c="#e8590c", label="chronic")
        ax.plot([0, 1], [means["rr_onset"][j], means["rr_peak"][j]], "-o", c="#3b5bdb", label="RR")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["onset", "peak"])
        ax.set_title(gene, fontsize=10); ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("log CP10k"); np.atleast_1d(axes)[0].legend(fontsize=8)
    b.append("<div class='card'>" + img(fig) + "</div>")

    def gtab(rows, title):
        h = f"<div style='flex:1 1 330px'><b>{title}</b><table><tr><th>gene</th><th>DiD</th><th>RR Δ</th><th>chr Δ</th></tr>"
        for r in rows[:8]:
            cls = "up" if r["did"] > 0 else "down"
            h += (f"<tr><td class='gene'>{r['gene']}</td><td class='num {cls}'>{r['did']:+.2f}</td>"
                  f"<td class='num'>{r['rr_delta']:+.2f}</td><td class='num'>{r['chr_delta']:+.2f}</td></tr>")
        return h + "</table></div>"
    b.append("<h2>Onset→peak: which genes ramp differently</h2>")
    b.append("<div style='display:flex;gap:18px;flex-wrap:wrap'>"
             + gtab(res["stronger_onset_to_peak_in_RR"], "Ramps MORE in RR — MHC / antigen presentation")
             + gtab(res["stronger_onset_to_peak_in_chronic"], "Ramps MORE in chronic — Gpnmb / lipid") + "</div>")

    b.append("<h2>The point: the model difference is phase-dependent</h2>")
    b.append("<div class='good'>The MHC / antigen-presentation genes (<span class='gene'>Cd74</span>, "
             "<span class='gene'>H2-Aa</span>, <span class='gene'>B2m</span>, <span class='gene'>Tap1</span>) "
             "lean toward ramping <b>more in RR</b> across the onset→peak window — but the continuous "
             "severity-slope analysis (over the full disease range) found the same genes <b>steeper in chronic</b>. "
             "Same genes, opposite direction depending on the timepoint window. So yes: blending all timepoints can "
             "mask or flip a phase-specific signal. <span class='gene'>Hal</span> is the one gene robust to every "
             "cut.</div>")
    b.append("<div class='note'><b>Bottom line:</b> matching a stage doesn't de-confound (the 35-gene trap); the "
             "DiD is the right design but is gated by RR onset = 2 animals. To actually resolve the onset→peak "
             "divergence you'd need more RR ONSET1 samples.</div>")
    return page("Immunoformer — phase-dependence", "\n".join(b))


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, html in [("performance.html", build_performance()), ("findings.html", build_findings()),
                       ("phase_dependence.html", build_did())]:
        with open(os.path.join(OUT, name), "w") as fh:
            fh.write(html)
        print(f"[report] {os.path.join('runs/reports', name)}  ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
