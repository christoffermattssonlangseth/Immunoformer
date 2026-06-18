# `Hal` (histidine ammonia-lyase) — a recurrent, metabolically-coherent hit in RRMAP2

> **Update (2026-06-15):** a deep-research stress-test partially corrects this
> doc — see [`hal-histidine-deep-research.md`](hal-histidine-deep-research.md).
> Headline: the `Hal` finding is novel, but the *histidine-depletion bridge to
> systemic MS histidine↓* is weak (wrong cell type, histamine fork dominates,
> carnosine sink moves opposite, UCA failed its EAE test). Two claims below were
> refuted in verification (the MDPI/CSF paper does **not** pin histidine↓ to a
> fork; HAL kinetics are unresolved). Reframe toward "`Hal` = reproducible
> myeloid/neutrophil signature of the relapse cycle".

## Why it matters

`Hal` is the single most robust disease-associated gene across **every** RRMAP2
analysis we ran:

- **Within-RR relapse-cycle trajectory** — top increasing gene (Spearman vs cycle
  rank), survives severity adjustment.
- **RR-vs-chronic differential dynamics** (score×model interaction) — the only gene
  to clear FDR (q ≈ 4e-9).
- **Onset→peak difference-in-differences** — top gene leaning toward a steeper
  ramp in RR (directional; the DiD is underpowered overall).

`Hal` is robust to every cut, including the ones that flip other genes (e.g. the
MHC/antigen-presentation module changes direction between the full-severity and
onset→peak windows). That consistency is what makes it worth following up.

## What `Hal` is

**Histidine ammonia-lyase** (histidase; EC 4.3.1.3). A cytosolic enzyme that
catalyzes the committed first step of histidine catabolism:

> L-histidine → trans-urocanic acid + ammonia

It is a *metabolic* enzyme, not a canonical neuroinflammation marker — classically
expressed in liver and epidermis. That makes its strong, consistent disease
association here notable rather than expected.

## Literature context

**`Hal` itself is not a known EAE/MS gene.** No study directly ties the enzyme to
MS or EAE, so this is a relatively novel / underexplored observation in this
context.

**But the pathway it sits in is well-connected to MS, and our finding fits it:**

1. **Histidine is robustly *decreased* in MS.** Multiple metabolomic studies find
   lower histidine in CSF and serum of MS patients, correlating with higher
   disability (EDSS) and with fatigue. Histidine is both a histamine precursor and
   an ROS scavenger, so its depletion is read as a marker of inflammatory activity.

2. **`Hal` *consumes* histidine.** If `Hal` rises as disease progresses (what we
   observe), it would locally deplete histidine — offering a candidate tissue-level
   mechanism for the systemic "histidine ↓ in worse MS" finding. This is a testable
   hypothesis our data *generates*, not just recapitulates.

3. **The product side is equivocal.** Urocanic acid is a known immunosuppressant
   (cis-UCA especially), but the one direct EAE test found that raising cis-UCA did
   **not** reproduce UVB's protection against EAE — decoupling UCA from the effect.
   So the **histidine-depletion** reading is better supported than a simple
   "Hal↑ → UCA → immunosuppression" story.

For context, the better-studied branch of histidine metabolism in EAE is the other
fork — histidine → histamine via HDC, with the histamine H3 receptor (`Hrh3`) under
the EAE susceptibility locus *Eae8*. `Hal` diverts histidine *away* from that
histamine arm.

## Upshot / next checks

`Hal` is a novel, metabolically-coherent hit, not noise. The cleanest follow-up
framing is **histidine depletion**, not UCA immunosuppression. Natural next checks:

- Does the histamine-synthesis branch (`Hdc`) or do histidine transporters move in
  the *opposite* direction in the data?
- Which cell type is producing the `Hal` signal (targeted lookup, no full
  annotation needed)?

## Sources

- [Histidine ammonia-lyase — Wikipedia](https://en.wikipedia.org/wiki/Histidine_ammonia-lyase) (enzyme / biochemistry)
- [UCA does not mediate UVB suppression of EAE — BMC Neuroscience 2017 (PMC5217575)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5217575/)
- [CSF metabolomics in MS — histidine decreased (PMC9178205)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9178205/)
- [CSF amino/fatty acids in early MS — histidine ↓ correlates with EDSS (PMC10671192)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10671192/)
- [Lower serum histidine in fatigued women with MS — PubMed 32440368](https://pubmed.ncbi.nlm.nih.gov/32440368/)
- [Histamine H3R / Hrh3 and EAE susceptibility locus Eae8 (PMC3718788)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3718788/)
- [Histamine and neuroinflammation in EAE — Frontiers 2012](https://frontiersin.org/articles/10.3389/fnsys.2012.00032/full)

---
*Generated 2026-06-12 from the RRMAP2 within-RR analyses + a short literature search.
`Hal` cell-type source and `Hdc`/transporter direction are not yet checked.*
