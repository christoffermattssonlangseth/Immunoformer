# ImmunoTransformer — Design Note

> Maps the self-supervised pathology framework from Le Quesne et al.
> (bioRxiv 2025.11.12.688049 + Spatial Biology Congress 2026 talk) onto our
> Xenium spatial transcriptomics work in EAE / RR-EAE and related models,
> positioned against existing spatial single-cell foundation models.

## Sources

- **Paper:** Rakovic K, …, Yuan K, Le Quesne J. *"Self-supervised AI reveals a
  lethal discohesive phenotype in lung adenocarcinoma."* bioRxiv (2025).
  doi:10.1101/2025.11.12.688049. Covers the LUAD/discohesion result; does **not**
  describe the transformer.
- **Conference note:** `obsidian-vault/conferences/spatial biology congress 2026/day2/processed/John Le Quesne.md`.
  Source for the HPC pipeline and both transformer methods below.
- **Prior-art FMs:** Nicheformer (Nat Methods 2025), CellPLM, scGPT, Geneformer,
  scFoundation — see "Positioning" below.

## The framework (two layers)

**Layer 1 — SSL phenotype discovery (HPCs).** Self-supervised contrastive learning
over H&E image tiles, no labels. Clusters become **histomorphological phenotype
clusters (HPCs)** — an interpretable vocabulary of recurrent tissue appearances.
Each slide → **vector of HPC proportions**, regressed against ground truth. Payoff:
*epithelial discohesion is lethal, but only in immunologically cold stroma* — a
context-dependent interaction. Same shape as *CD8 predicts survival only when tumour
proliferation is low*.

**Layer 2 — the transformer (the "ImmunoTransformer" analogue).** Tokens in a
sequence, masked-token (BERT-style) pretraining.
- v1: HPC arrays as tokens → high tissue-type accuracy with a *fraction of the data*.
- v2 (in submission): **token = single cell** (marker expression + morphology),
  minimal spatial grid, same masked-LM training. Beats larger FMs on PD-L1,
  stage/grade, growth pattern; UMAP separates labels with little supervision.

### What transfers to our data vs. what must be substituted

- **HPCs do NOT transfer as-is** — they are H&E/morphology-defined and our data has
  no H&E. The transferable abstraction is "SSL discovery of recurrent phenotype
  clusters → proportion vector → outcome." Substrate swaps **pixel morphology →
  gene-expression niches** ("spatial-expression niche clusters"). Xenium morphology
  (area, shape, DAPI) can ride along but is optional.
- **Layer-2 v2 fits our data better than theirs** — it is already expression-based +
  spatial. Xenium is the same data shape as multiplex IF but a strict superset:
  5101 genes/cell vs. dozens of proteins. → this is the priority build.
- Caveat: H&E morphology is cheap and information-dense. Expression niches are
  noisier per feature, so niche discovery likely needs more spatial
  smoothing/neighbourhood aggregation than their tile clustering did.

## Our data

| Axis | Levels |
|---|---|
| Scale | ~3M cells, **5101-gene panel** (Xenium 5K, near-transcriptome/cell) |
| Models | Chronic EAE · standard EAE (staged) · relapsing EAE (~8 timepoints) · RR-EAE (relapse+remit) · **mtDNA double-strand-break in oligodendrocytes (brain)** · controls for all |
| EAE timecourse | pre-symptom-onset → onset → peak → 2 late timepoints |
| Spinal cord regions | lumbar · thoracic · cervical |
| Other tissue | optic nerve (EAE) · brain (mtDNA-DSB model) |

> Open check: "4 timepoints" was stated but five were listed (pre-onset, onset, peak,
> late-1, late-2). Treating as the full staged series — confirm.

Implications:
- **Foundation-model-scale corpus.** 3M cells is real domain-pretraining scale; SSL is
  not just a small-cohort trick here.
- **Ladder of weak labels:** disease model · **ordinal timepoint** (clean staged axis,
  better supervision than clinical score) · region · relapse/remit phase ·
  distance-from-control.
- **Timecourse + controls + multiple etiologies is the real asset** (see Applications).

## Selected priority & supervision

Decisions (2026-06-03):
- **Build first:** cell-token transformer (Layer-2 v2 recipe).
- **Ground truth:** clinical EAE score + relapse/remission phase, plus the metadata
  ladder above. Labels live at the **section/animal level** — weak supervision.

## Architecture: cell-token transformer

### Stage A — Self-supervised pretraining (no labels)

- **Token = one cell.** Features: 5101-gene expression vector + morphology
  (area, eccentricity, DAPI/total counts) + small local-neighbourhood summary.
- **Position:** continuous positional embedding of (x,y) centroid (Fourier/RFF),
  not a discrete grid — Xenium coordinates are real-valued.
- **Dual masking (enabled by the 5K panel — richer than the IF version could be):**
  1. **Intra-cell gene masking** — mask a subset of genes in a cell, reconstruct from
     the rest (cell-state / gene–gene coexpression; the scGPT/Geneformer axis).
  2. **Spatial cell masking** — mask whole cells, reconstruct from neighbours (niche
     context; the CellPLM / Le Quesne axis).
- **Reconstruction head:** negative-binomial / Poisson over counts, not MSE.
- **Read-depth / capture-efficiency conditioning:** add a read-depth recovery task
  (cf. scFoundation) and/or condition on total counts, so the model learns biology,
  not regional capture efficiency (grey vs. white matter differ).
- **Scale:** operate on local windows (tiles of N cells or k-NN neighbourhoods);
  full-section attention won't scale to 10^5–10^6 cells/section.

### Stage B — Weakly-supervised section head

- Pool cell embeddings → section embedding via **attention-based MIL** (gated
  attention pooling). Handles "label per-section, signal in a few cells"; attention
  weights expose which cells/niches drove the call — interpretable readout (the
  niche-proportion analogue of HPCs).
- Heads: **ordinal regression** for EAE score and for **timepoint** (both ordered);
  classification for relapse/remit phase, region, model.

### Validation order

1. **Linear-probe / fine-tune on the staged timepoint axis** — cleanest supervision.
   (NB: do *not* assume zero-shot UMAP separation — see Positioning caveat. Treat any
   unsupervised separation as a bonus, not the plan.)
2. **Weakly-supervised heads** for EAE score and relapse/remit.
3. **Attention attribution** → which cell populations drive predictions → cross-check
   vs. known EAE biology (perivascular T cells, DAM microglia, reactive astrocytes).

### Data pitfalls (ours specifically)

- **Animal-level leakage** — labels are per-animal; split by **animal**, not section
  or tile, or performance is massively overestimated.
- **Capture-efficiency confound** — see read-depth conditioning above.
- **Region/batch confound** — lumbar/thoracic/cervical + optic nerve + brain differ in
  baseline composition; include region as a covariate / stratify.

## Applications (headline biology)

1. **Pre-symptomatic niche detection.** We have *before-symptom-onset* tissue with
   matched controls. Can the model detect molecular niche changes that **precede
   clinical signs** (invisible to EAE score)? Direct analogue of Le Quesne's features
   "independent of current prognostic schema." Primary publishable hook.
2. **Cross-etiology transfer.** **mtDNA-DSB oligodendrocyte** model is a cell-intrinsic
   demyelinating insult, orthogonal to autoimmune EAE. Does a niche vocabulary learned
   on EAE transfer to primary oligo damage? Shared oligo-stress/demyelination niches =
   strong result; divergence equally interesting.
3. **Cross-tissue transfer.** Spinal cord → **optic nerve** (same panel, same disease)
   = near-ideal held-out generalization test. Also region transfer within cord.
4. **Relapse/remission dynamics.** Which niches track relapse; is remission lesion
   *resolution* (composition shift) or *silencing* (activity shift within a niche)?
5. **Context-dependent interaction hunting** (the conceptual import of the paper):
   - T-cell infiltration × microglial/astrocyte (DAM / reactive-astrocyte) state.
   - Remyelination × immune "coldness" (their lethality gate, inverted toward repair).
   Implemented as downstream readouts of the Stage-B attention layer.
6. **Atlas of relapse-/onset-driving niches** rather than per-animal models —
   Le Quesne's "TMA of lethal clusters." Score a new section by how much
   onset/relapse-driving tissue it contains.

## Positioning vs. existing spatial single-cell FMs

Masked-token pretraining on spatial single-cell data is **established**. Do not
rebuild the backbone — adopt/adapt and differentiate on biology + framing.

### Nicheformer (Theis lab, Nat Methods 2025) — gene-as-token, per-cell encoder

- **Tokenization:** gene-rank tokenizer (Geneformer-style); a cell = a set of gene
  tokens. **1,500-token context**.
- **Architecture:** 12 encoder layers, 16 heads, FFN 1,024, **512-dim embedding,
  49.3M params**. Learnable positional embeddings over the *token set* (NOT xy).
- **Objective:** MLM, **15% masking**, BERT 80/10/10. Metadata/context tokens =
  species, modality, assay.
- **Capture-efficiency handling:** technology-specific *nonzero-mean* vectors (not a
  global mean) → **directly solves our grey/white-matter capture confound.** Adopt this.
- **Spatial is NOT in pretraining.** Pretraining is per-cell; spatial signal comes
  from the 110M-cell corpus + downstream tasks (niche / region / density /
  composition), via linear-probe or fine-tune on frozen embeddings.
- **Corpus:** SpatialCorpus-110M (incl. Xenium). → robust per-cell *state* encoder.

### CellPLM (ICLR 2024) — cell-as-token, niche/neighbourhood encoder

- **Tokenization:** **cells are tokens, tissue = sentence** — models inter-cell
  relations *in pretraining* (the paradigm we want).
- **Four modules:** gene-expression embedder (extendable to arbitrary gene sets) →
  transformer encoder over cells → **Gaussian-mixture latent** (captures functional /
  niche-like cell groups) → batch-aware decoder.
- **Spatial:** **2D sinusoidal positional embedding on real (x,y)** for SRT cells
  (scRNA cells get a shared learned PE). H0 = expression-embed + positional-embed.
- **Objective:** cell-level MLM — mask cells, reconstruct from neighbours.
- **Corpus:** ~10M cells. Smaller/older, CNS coverage unknown.

### Others

- **scGPT / Geneformer** — gene-as-token MLM (scGPT reconstructs binned value;
  Geneformer reconstructs gene rank). The intra-cell-masking axis.
- **scFoundation** — added read-depth recovery task → second confirmation of the
  capture-efficiency conditioning above.
- **Honest caveat (zero-shot):** independent evaluations show these FMs can
  *underperform simple baselines* without task-specific fine-tuning, and that MLM
  pretraining doesn't guarantee useful zero-shot embeddings. → plan for fine-tuning;
  do not rely on zero-shot UMAP separation.

## Backbone decision (resolves the earlier open question)

The fork: **Nicheformer = per-cell state encoder, spatial bolted on downstream**;
**CellPLM = niche/neighbourhood encoder by design**. Our goals (niche discovery,
context-dependent interactions, pre-symptomatic spatial change) point at the
**CellPLM paradigm**, with Nicheformer's engineering grafted in.

**Recommended path (de-risk → specialize):**
1. **Cheap baseline first.** Use Nicheformer (or scGPT) as a *frozen per-cell encoder*
   → our attention-MIL spatial head (Stage B). Establishes a strong supervised
   baseline on the timepoint / EAE-score tasks with **zero pretraining cost**. If this
   already does well, we've saved months.
2. **Main model = ImmunoTransformer.** CellPLM paradigm (cell-as-token + 2D spatial PE
   + neighbourhood MLM + Gaussian-mixture latent) **plus** grafts from Nicheformer:
   (a) technology/read-depth **nonzero-mean normalization** for capture efficiency,
   (b) **metadata context tokens** = disease-model / region / ordinal-timepoint as
   conditioning, (c) the dual-masking from Stage A (intra-cell gene + spatial cell).
3. **From-scratch vs. init-from-CellPLM** is empirical. Start by **fine-tuning** to
   de-risk (CellPLM's gene embedder is explicitly designed to extend to new gene
   sets, so adapting to our mouse 5101-gene Xenium panel is feasible). Graduate to
   **from-scratch** only if the 5K-panel + CNS specialization beats the fine-tuned
   baseline on held-out animals. Note: 3M cells < CellPLM's 10M < Nicheformer's 110M,
   so from-scratch foundation pretraining is borderline on corpus size — specialization
   (one tissue system, one panel), not scale, is our edge.

**Our defensible contribution = (a) EAE/demyelination specialization at 3M cells &
5101 genes, (b) Le Quesne's weak section-level label + interpretable
niche-proportion + interaction-hunting framing, (c) the pre-symptomatic /
cross-etiology / cross-tissue biology** — not a from-scratch foundation model.

## Open questions

- Le Quesne v2 position encoding: discrete grid vs. continuous coords? (note says
  "tiny spatial grid, 1 cell per cell" — ambiguous). CellPLM resolves the analogous
  choice with 2D sinusoidal PE on real xy — sensible default for us.
- Le Quesne v2 masking fraction (Nicheformer/BERT default = 15%; reasonable default).
- Does CellPLM's pretrained gene embedder cover enough of our 5101-gene mouse CNS
  panel to make fine-tuning worthwhile, or is from-scratch cleaner? → small benchmark.
