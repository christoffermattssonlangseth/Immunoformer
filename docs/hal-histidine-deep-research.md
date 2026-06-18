# `Hal` / histidine story — deep-research stress-test

*Generated 2026-06-15 from a fan-out web-research pass (21 sources, 92 extracted
claims, top 25 adversarially verified — 21 confirmed, 4 killed). Companion to
[`hal-histidine-finding.md`](hal-histidine-finding.md), which it partially
corrects.*

## Question

Is HAL/histidase (EC 4.3.1.3) plausibly active and disease-relevant in the CNS
during neuroinflammation, and does a *"local HAL↑ → local histidine depletion →
systemic histidine↓ in MS"* chain hold up?

## Bottom line

The RRMAP2 `Hal` observation is **genuinely novel and worth pursuing**, but the
specific depletion bridge **does not hold up well**. It fails on four largely
independent grounds, and the human histidine-down signal it tries to explain is
weaker than it first appeared. One important escape hatch (inducible/lesional HAL
in healthy-tissue blind spot) keeps the hypothesis alive *as something to test in
tissue*, not to assert.

## The four problems with the bridge

1. **Wrong cell type.** In human single-cell data (Protein Atlas), HAL is
   essentially absent from all CNS parenchymal cells — microglia, astrocytes,
   neurons, oligodendrocytes all < 0.5 nCPM. It is "Group enriched
   (Neutrophils, Hepatocytes)", Tau 0.88; among immune cells **neutrophils
   dominate** (~340 nCPM vs ~3 in tissue macrophages). In the 2026 mouse+human
   microglia taxonomy no defining program (Surveillance, DAM/phagocytosis,
   Inflammation, IFN) contains `Hal`. So the likely myeloid source of a `Hal`
   signal is the **infiltrating neutrophil**, not the DAM/M2 microglial state the
   original framing leaned on.

2. **The documented CNS fork is histamine, not urocanate.** Inflammation-driven
   histidine catabolism in/near the CNS runs through HDC (histidine→histamine):
   ALS microglia express HDC/HNMT/DAO protein; inflamed psoriatic skin upregulates
   HDC. The HAL/urocanic-acid fork does not show up in neuroinflammation.

3. **The carnosine sink moves the opposite way.** CARNS1 (carnosine synthase,
   oligodendrocyte histidine-dipeptide sink) is **down** 12.6-fold in demyelinated
   MS lesions; `Carns1`-KO mice get **worse** EAE. The other major
   histidine-consuming branch is *decreasing* in lesions.

4. **The product failed its one direct EAE test.** Cis-urocanic acid (HAL's
   product) was systemically immunosuppressive (spleen weight −30%) yet did **not**
   change EAE incidence/onset/severity, while UVB itself dropped incidence
   100%→26% (PMC5217575 / BMC Neurosci 2017). Directly contradicts
   "Hal↑ → protective urocanate".

## The human signal is also weaker than assumed

- **CSF histidine down: real but modest.** ~27% reduction, p≈0.005–0.012;
  negatively correlates with EDSS at 1 yr (r=−0.42) and 2 yr (r=−0.43) but **not**
  at collection. Small attriting cohorts (40→31→21), borderline p, no
  multiple-comparison correction (would not survive Bonferroni).
- **Serum is discordant.** One NMR study found serum histidine **increased**
  1.9-fold in MS — opposite direction. CSF and serum do not move together.
- **Parsimonious competitor.** Low circulating histidine is a well-replicated
  marker of *whole-body* inflammation / protein-energy wasting (tracks CRP, IL-6,
  leukocytes across CKD, obesity, UK Biobank). No local catabolic enzyme required.

## Corrections to `hal-histidine-finding.md`

- **Killed claim (0–3):** the IJMS/MDPI CSF paper (PMC10671192) does **not**
  explicitly attribute decreased histidine to the histamine branch. It reports
  histidine down but does not pin it to a specific fork. The earlier doc's framing
  overstated this.
- **Killed claims (kinetics):** HAL Km ≈ 20 mM and "slow/incomplete in-vitro
  depletion" were both refuted. Raw biochemical plausibility of HAL meaningfully
  draining a local histidine pool is therefore **genuinely unresolved**, not
  settled in either direction.

## The caveat that rescues the hypothesis

All strong expression evidence is from **healthy** tissue. Protein Atlas cannot
see inflammation-*induced* HAL, or HAL carried by neutrophils/myeloid cells
infiltrating an EAE lesion — exactly the condition RRMAP2 captures. The evidence
argues against *constitutive* CNS HAL but **cannot exclude inducible/lesional
HAL**. The finding lives precisely in that blind spot.

## Recommended reframing

Move the story away from *"explains systemic histidine↓ in MS"* (weak bridge) and
toward: **`Hal` marks a novel, highly reproducible myeloid/neutrophil metabolic
signature of the EAE relapse cycle.** This is what the statistics actually support
and it depends on none of the four shaky links. Histidine depletion becomes a
*tissue-level hypothesis to test*, not a claim to assert.

## Decisive experiments

1. **Which cell carries `Hal`?** Spatial / single-cell + immunostaining in RRMAP2
   lesions — infiltrating neutrophils (`S100a8/9`, `Ly6g`, `Retnlg`), microglia/
   macrophages, or non-immune cells? Make-or-break; already the open item in the
   finding doc.
2. **Does `Hal`↑ lower *local* histidine and raise urocanate?** Direct lesion
   metabolomics for histidine, urocanate, carnosine, histamine together — flux and
   competing sinks at once.
3. **Is it causal?** `Hal` KO / inhibition or urocanic-acid manipulation → does
   EAE course or local/systemic histidine change? Driver vs. bystander marker of a
   neutrophil/myeloid state.

## Key sources

- [Protein Atlas — HAL single cell](https://www.proteinatlas.org/ENSG00000084110-HAL/single+cell) — HAL absent from CNS cells; neutrophil/hepatocyte enriched
- [2026 microglia taxonomy — Nat Immunol](https://www.nature.com/articles/s41590-026-02472-z) — no `Hal` in any microglial state program
- [Carnosine / CARNS1 in MS lesions & EAE — bioRxiv 2023](https://www.biorxiv.org/content/10.1101/2023.03.30.534899.full.pdf) — carnosine sink down in lesions; Carns1-KO worsens EAE
- [cis-UCA does not protect against EAE — BMC Neurosci 2017](https://bmcneurosci.biomedcentral.com/articles/10.1186/s12868-016-0323-2) (PMC5217575)
- [ALS microglia histamine enzymes — Front Immunol 2017](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2017.01689/full)
- [Early-MS CSF amino acids, histidine ↓ — IJMS 2023 (PMC10671192)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10671192/)
- [Pilot CSF metabolomics, histidine ↓ & EDSS — Front Neurol 2022](https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2022.874121/full)
- [Serum histidine fatigue in MS women — PubMed 32440368](https://pubmed.ncbi.nlm.nih.gov/32440368/)
- [Serum histidine INCREASED 1.9× in MS — PMC11208524](https://pmc.ncbi.nlm.nih.gov/articles/PMC11208524/)
- [Low plasma histidine tracks inflammation/PEW — PubMed 18541578](https://pubmed.ncbi.nlm.nih.gov/18541578/)
