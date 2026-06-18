# Grant proposal — datasets section (our data + literature)

*Draft addition for the "Spatial tri-omic atlas" proposal (boss's paragraph on
spatial ARP-seq / CTRP-seq, developing brain P0–P21 + LPC model, Zhang et al.,
Nature 2025). This adds detail on our Xenium single-cell spatial datasets and a
focused set of literature datasets. Optic-nerve data deliberately excluded.*

## Our datasets (spinal cord EAE and brain demyelination, Xenium single-cell spatial)

Complementing the spatial tri-omic resource, we have generated single-cell-resolution
spatial transcriptomic datasets on the 10x Genomics Xenium platform (full **5,101-gene
mouse panel**, single-cell segmentation with spatial coordinates and CellCharter niche
assignments) spanning two neuroinflammation and demyelination contexts.

**RRMAP2 — spinal-cord experimental autoimmune encephalomyelitis (EAE).** A spatial
atlas of the canonical mouse MS model comprising **~1.38 million cells across 158 tissue
sections from 67 animals**, covering two complementary disease courses on the same panel:

- a **relapsing–remitting cohort** (SJL/PLP; 33 animals, ~894,000 cells) sampled across
  the full relapse cycle (induction → onset → PEAK1 → REMISSION1 → PEAK2 → REMISSION2 →
  PEAK3), and
- a **chronic cohort** (C57BL/6/MOG; ~34 animals) sampled along a severity-graded
  trajectory (control → onset → mild → severe → peak).

Each cell carries clinical metadata — quantitative EAE score at sacrifice (0–3.25),
disease stage, model, anatomical region (lumbar/thoracic/cervical), day of sacrifice,
and sex — and 25 spatially-resolved cell-type niches (oligodendrocyte/myelin, neuronal,
vascular, three inflammatory-myeloid states, microglia, astrocyte, OPC, T/NK, B/plasma,
ependymal). The design is terminal/cross-sectional (one disease stage per animal),
enabling population-level reconstruction of the relapse cycle and de novo recovery of
lesion architecture.

**mtDNA-DSB — brain primary-oligodendrocyte demyelination.** A spatial dataset of a
cross-etiology CNS damage model (oligodendrocyte mitochondrial DNA double-strand breaks)
comprising **~980,000 cells from 12 animals** on the same 5,101-gene panel, annotated
with control-vs-lesion condition, three genotypes, two ages, and a deep cell-type
hierarchy (12 cell classes, 44 cell types, 12 brain regions). This provides a
non-autoimmune, brain demyelination counterpoint to the autoimmune spinal-cord EAE data.

Together with the developmental and LPC tri-omic resources, these Xenium datasets extend
the platform's coverage to single-cell-resolution spatial transcriptomics of both
autoimmune (EAE) and primary-oligodendrocyte (mtDNA-DSB) demyelination, across spinal
cord and brain.

## Datasets in the literature

The proposed work builds on, and is positioned against, a small set of landmark spatial
datasets:

- **Spatial atlases of MS/EAE neuropathology.** Kukanja, Langseth et al. (*Cell*, 2024)
  generated a single-cell spatial atlas of EAE and human MS spinal-cord tissue by in situ
  sequencing (Xenium), modeling spatio-temporal lesion evolution and disease-associated
  glia — directly comparable to, and a foundation for, our EAE resource. Lerma-Martin et
  al. (*Nat. Neurosci.*, 2024) paired Visium spatial transcriptomics with snRNA-seq across
  12 subcortical MS lesions, defining lesion-rim niches and APOE–TREM2 myeloid
  interactions. Absinta et al. (*Nature*, 2021) profiled chronic-active MS lesion edges by
  snRNA-seq, defining the "microglia inflamed in MS" (MIMS) state and a C1q-driven
  lymphocyte–microglia–astrocyte axis.
- **Spatial epigenome–transcriptome (tri-omic) methods and atlases.** Our tri-omic data
  extend the deterministic-cobarcoding lineage of Zhang et al. (*Nature*, 2023;
  spatial-ATAC-RNA-seq and spatial-CUT&Tag-RNA-seq, applied to embryonic and juvenile
  mouse brain) and its tri-omic successor, Zhang et al. (*Nature*, 2025), the
  spatiotemporal ATAC/CUT&Tag–RNA–protein atlas of mouse brain development (P0–P21) and
  neuroinflammation that anchors the present resource.
- **Reference brain atlases.** For developmental and anatomical referencing, the field's
  benchmarks are the whole adult mouse brain MERFISH atlases — Yao et al. (*Nature*, 2023;
  Allen Brain Cell Atlas) and Zhang et al. (*Nature*, 2023; ~9M cells, 1,122-gene panel) —
  together with emerging developmental spatial atlases spanning embryonic-to-postnatal
  stages (e.g., the whole-brain developmental spatial atlas, *Neuron*, 2025).

## References

- Kukanja, Langseth et al. *Cell* 2024 — Cellular architecture of evolving
  neuroinflammatory lesions and MS pathology.
  <https://www.cell.com/cell/fulltext/S0092-8674(24)00233-2>
- Lerma-Martin et al. *Nat. Neurosci.* 2024 — Cell type mapping reveals tissue niches and
  interactions in subcortical MS lesions.
  <https://www.nature.com/articles/s41593-024-01796-z>
- Absinta et al. *Nature* 2021 — A lymphocyte–microglia–astrocyte axis in chronic active
  MS. <https://www.nature.com/articles/s41586-021-03892-7>
- Zhang et al. *Nature* 2023 — Spatial epigenome–transcriptome co-profiling of mammalian
  tissues. <https://www.nature.com/articles/s41586-023-05795-1>
- Zhang et al. *Nature* 2025 — Spatial dynamics of brain development and neuroinflammation
  (boss's cited paper). <https://www.nature.com/articles/s41586-025-09663-y>
- Yao et al. *Nature* 2023 — A high-resolution transcriptomic and spatial atlas of cell
  types in the whole mouse brain. <https://www.nature.com/articles/s41586-023-06812-z>
- Zhang et al. *Nature* 2023 — Molecularly defined and spatially resolved cell atlas of
  the whole mouse brain. <https://www.nature.com/articles/s41586-023-06808-9>

## Open items / notes

- **Kukanja/Langseth 2024 is our own prior work** — placed in "literature" but framed as
  a foundation; could be moved into "our datasets" as prior published work.
- **Chronic-cohort animal count (~34)** is approximate (67 total − 33 RR). Exact RR/chronic
  split and per-stage n can be pulled from the h5ad / run outputs.
- Numbers sourced from `docs/dataset-hierarchy.md` and `docs/rrmap2-relapse-atlas.md`.
