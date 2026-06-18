"""Render the relapse-phase feasibility investigation as a standalone HTML report.

Reads the Phase 0 gate + niche-resolved follow-up JSONs and emits a single
self-contained HTML (inline CSS, no external deps) summarising the negative result
and the surviving decomposition. Matches runs/reports/ styling.

    python scripts/make_phase_report.py
    open runs/reports/relapse_phase_feasibility.html
"""

from __future__ import annotations

import json
import os

CSS = """
:root{--ink:#1a1a1a;--mut:#666;--line:#e6e6e6;--accent:#3b5bdb;--bg:#fafafa;
--up:#c0392b;--down:#2c6fbb;--good:#2f9e44;--warn:#e8590c;--stop:#c0392b}
*{box-sizing:border-box}body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
color:var(--ink);background:var(--bg);margin:0;line-height:1.55}
.wrap{max-width:920px;margin:0 auto;padding:48px 28px 80px}
h1{font-size:30px;margin:0 0 4px}h2{font-size:21px;margin:38px 0 10px;padding-bottom:6px;border-bottom:2px solid var(--line)}
h3{font-size:16px;margin:22px 0 6px;color:#333}
.sub{color:var(--mut);font-size:14px;margin:0 0 8px}
.card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:14px 0;box-shadow:0 1px 2px rgba(0,0,0,.03)}
.kpis{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0}
.kpi{flex:1 1 150px;background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.kpi .v{font-size:26px;font-weight:650}.kpi .l{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:8px 0}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.up{color:var(--up);font-weight:600}.down{color:var(--down);font-weight:600}.muted{color:#999}
.note{background:#fff8f0;border-left:3px solid var(--warn);padding:10px 14px;border-radius:0 8px 8px 0;font-size:13.5px;margin:12px 0}
.good{background:#f0faf2;border-left:3px solid var(--good);padding:10px 14px;border-radius:0 8px 8px 0;font-size:13.5px;margin:12px 0}
.stop{background:#fdecea;border-left:3px solid var(--stop);padding:12px 16px;border-radius:0 8px 8px 0;font-size:14px;margin:12px 0}
.gene{font-family:"SF Mono",Menlo,Consolas,monospace;font-size:12.5px}
.cols{display:flex;gap:18px;flex-wrap:wrap}.cols>div{flex:1 1 360px}
.foot{color:var(--mut);font-size:12px;margin-top:28px;border-top:1px solid var(--line);padding-top:12px}
.verdict{font-size:15px;font-weight:650;color:var(--stop)}
"""


def auc_cell(v):
    if v is None:
        return "<td class='num muted'>—</td>"
    cls = "up" if v > 0.65 else "muted"
    return f"<td class='num {cls}'>{v:.3f}</td>"


def niche_block(path, label):
    if not os.path.exists(path):
        return f"<p class='muted'>{label}: results not found ({path})</p>"
    r = json.load(open(path))
    tested = [x for x in r["per_niche"] if x.get("auc") is not None]
    aucs = [x["auc"] for x in tested]
    mx = max(aucs) if aucs else float("nan")
    comp = r["composition_test"]
    rows = "".join(
        f"<tr><td class='gene'>{x['niche']}</td><td class='num'>{x['n_peak']}/{x['n_rem']}</td>"
        + auc_cell(x["auc"])
        + (f"<td class='num'>{x['perm_p']:.3f}</td>" if x.get("perm_p") is not None
           else "<td class='num muted'>screened out</td>") + "</tr>"
        for x in sorted(tested, key=lambda z: -z["auc"])[:8])
    return (f"<h3>{label} — {r['n_tested']} testable of {r['n_niches']}</h3>"
            f"<p class='sub'>Best AUC <b>{mx:.3f}</b> (need &gt;{0.62} to even run the null; "
            f"Bonferroni α={r['bonferroni_alpha']}). Hits: "
            f"<b>{'none' if not r['hits'] else ', '.join(r['hits'])}</b>. "
            f"Composition (fractions) AUC {comp['residualized_auc']:.3f}, p={comp['perm_p']:.3f}.</p>"
            "<table><tr><th>niche</th><th>n pk/rem</th><th>resid AUC</th><th>perm p</th></tr>"
            + rows + "</table><p class='sub muted'>Top 8 by AUC shown; all are below chance.</p>")


def main():
    out = "runs/reports/relapse_phase_feasibility.html"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    g = json.load(open("runs/rr_phase_feasibility/results.json"))
    A, B = g["test_A"], g["test_B"]
    dec = g["decomposition_illustration"]
    hal = g["hal_residual"]

    H = [f"<!doctype html><html><head><meta charset='utf-8'>"
         f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>Immunoformer — relapse-phase feasibility</title><style>{CSS}</style></head><body><div class='wrap'>"]
    H.append("<h1>Is there a relapse <i>phase</i> to model? — feasibility gate</h1>")
    H.append("<p class='sub'>RRMAP2 within-RR · can a tissue snapshot's <b>direction of travel</b> "
             "(escalating vs recovering) be read molecularly, <b>beyond clinical severity</b>? "
             "Pre-registered gate in docs/relapse-phase-model-design.md. Animal-level, LOAO CV, "
             "severity-residualized features, permutation nulls.</p>")

    H.append("<div class='kpis'>")
    H.append(f"<div class='kpi'><div class='v down'>{A['residualized_molecular_auc']}</div>"
             f"<div class='l'>bulk phase AUC (chance=0.5)</div></div>")
    H.append(f"<div class='kpi'><div class='v'>{A['perm_p']}</div><div class='l'>perm p (bulk)</div></div>")
    H.append(f"<div class='kpi'><div class='v down'>{hal['resid_Hal_peak_minus_rem']:+.2f}</div>"
             f"<div class='l'>Hal peak−rem (severity-removed)</div></div>")
    H.append(f"<div class='kpi'><div class='v'>0</div><div class='l'>niches with phase signal</div></div>")
    H.append("</div>")

    H.append("<div class='stop'><span class='verdict'>VERDICT: STOP — abandon the phase model.</span> "
             "Phase-beyond-severity is null in bulk, in all 10 spatial niches, in all 25 cell types, and in "
             "cell-type composition. Trajectory direction is <b>not identifiable</b> in this terminal "
             "cross-sectional design — a study-design limit, not a modeling one.</div>")

    # gate
    H.append("<h2>The gate (bulk pseudobulk)</h2>")
    H.append("<table><tr><th>test</th><th>contrast</th><th>severity-only AUC</th>"
             "<th>severity-residualized AUC</th><th>perm p</th><th>read</th></tr>")
    H.append(f"<tr><td><b>A</b> phase-beyond-severity</td><td>peak (13) vs remission (9)</td>"
             f"<td class='num'>{A['severity_only_auc']}</td>{auc_cell(A['residualized_molecular_auc'])}"
             f"<td class='num'>{A['perm_p']}</td><td>null — peaks differ only by being severe</td></tr>")
    H.append(f"<tr><td><b>B</b> direction</td><td>onset (4) vs remission (9)</td>"
             f"<td class='num'>{B['severity_only_auc']}</td>{auc_cell(B['residualized_molecular_auc'])}"
             f"<td class='num'>{B['perm_p']}</td><td class='muted'>discard — see caveat</td></tr>")
    H.append("</table>")
    H.append("<div class='note'><b>Why TEST B is not a win.</b> It printed AUC 1.0, but (1) severity-only "
             "AUC = 0.0 means severity already separates onset from remission perfectly (onset is "
             "systematically <i>lower</i>-severity) — so 'matched severity' was false; and (2) AUC 1.0 on "
             "<b>n=4 onset animals</b> with 8 PCs is separable-by-anything (batch/region/overfit), not "
             "decoded direction. It is noise dressed as signal.</div>")
    H.append("<div class='note'><b>Hal is severity, not phase.</b> Once severity is removed, "
             f"<span class='gene'>Hal</span> peak−remission = <b>{hal['resid_Hal_peak_minus_rem']:+.2f}</b> "
             "(≈0). Hal's entire oscillation is explained by how sick the animal is — it carries no "
             "independent direction information.</div>")

    # niche
    H.append("<h2>The rescue attempt — niche-resolved (894k cells)</h2>")
    H.append("<p class='sub'>Bulk averages over cells, so phase could hide in one population. Repeat TEST A "
             "<i>inside</i> each niche / cell type. It does not hide there either:</p>")
    H.append("<div class='cols'>")
    H.append("<div>" + niche_block("runs/rr_phase_niche/results_CellCharter_10.json",
                                    "Spatial niches (CellCharter_10)") + "</div>")
    H.append("<div>" + niche_block("runs/rr_phase_niche/results_leiden_1.json",
                                    "Cell types (leiden_1)") + "</div>")
    H.append("</div>")
    H.append("<p class='sub muted'>Consistently sub-0.5 AUCs reflect LOAO small-n / class-imbalance bias, "
             "not inverse signal — the inferential point is only that nothing clears the null.</p>")

    # what survives
    H.append("<h2>What survives — the descriptive decomposition</h2>")
    H.append("<div class='good'>The relapse cycle is real but reduces to two axes, neither of which needs an "
             "AI model: an <b>acute / reversible severity</b> axis and an <b>irreversible cumulative accrual</b> "
             "axis. There is no third 'direction' axis to learn.</div>")
    H.append("<table><tr><th>program</th><th>peak − remission<br>(phase / acute)</th>"
             "<th>REM2 − REM1<br>(accrual)</th><th>interpretation</th></tr>")
    H.append(f"<tr><td><b>Oscillator</b> (Hal, Arg1, Chil3…)</td>"
             f"<td class='num up'>{dec['oscillator_peak_minus_rem']:+.2f}</td>"
             f"<td class='num'>{dec['oscillator_rem2_minus_rem1']:+.2f}</td>"
             f"<td>acute activity (mostly severity); resets</td></tr>")
    H.append(f"<tr><td><b>Ratchet</b> (Fcrls, Gpnmb…)</td>"
             f"<td class='num'>{dec['ratchet_peak_minus_rem']:+.2f}</td>"
             f"<td class='num up'>{dec['ratchet_rem2_minus_rem1']:+.2f}</td>"
             f"<td>irreversible accrual; floor rises each cycle</td></tr>")
    H.append("</table>")
    H.append("<p class='sub muted'>Decomposition is illustrative — gene lists derive from the same data; "
             "TEST A shows the oscillator 'phase' term is largely severity.</p>")

    H.append("<h2>What would actually change this</h2>")
    H.append("<p class='sub'>Not a better model — a different experiment. Trajectory direction needs "
             "<b>longitudinal sampling</b> (the same animal over time) or <b>many more onset sections</b> "
             "(ascending limb was n=4). Cross-sectional terminal data with severity-dominated axes cannot "
             "separate 'getting worse' from 'recovering'.</p>")

    H.append("<div class='foot'>Immunoformer · RRMAP2 within-RR · generated by scripts/make_phase_report.py "
             "from runs/rr_phase_feasibility/ + runs/rr_phase_niche/. "
             "Design + full result: docs/relapse-phase-model-design.md.</div>")
    H.append("</div></body></html>")
    with open(out, "w") as fh:
        fh.write("".join(H))
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
