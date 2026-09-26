# Phase 3 Literature Review: Representing Compounds and Dose as Model Inputs

Scope: how published perturbation-response models represent drugs/dose, and what that implies
for an 8 cell line × 92 drug × 3 dose slice of Tahoe-100M with a chem⊕cell ridge baseline vs a
conditional neural model. Budget: ~20 web searches/fetches used.

## Summary

- Every major single-cell/bulk perturbation-response model that conditions on chemical
  structure (CPA, chemCPA, PRnet) encodes the drug as a **fixed-size vector from a molecule
  encoder** — most commonly an RDKit/Morgan fingerprint or descriptor vector — fed through a
  small MLP into a shared latent space; chemCPA explicitly benchmarked fingerprint-based
  encoders against several pretrained graph/language embeddings (GROVER, MPNN, JT-VAE, Weave,
  seq2seq) [from abstract/secondary].
- Independent benchmarking work on small/medium QSAR datasets finds fixed fingerprints (ECFP)
  remain **competitive with or better than** pretrained embeddings (ChemBERTa, MolFormer,
  Uni-Mol, GROVER), with the gap favoring pretrained embeddings only once data is plentiful
  [verified: source read, arXiv:2508.06199].
- Bemis–Murcko scaffold splitting is known to be unreliable with small compound sets (many
  singleton scaffolds) and can *overestimate* performance by putting near-identical scaffolds on
  both sides of a naive split; recent work recommends fingerprint/Butina clustering splits as a
  harder, more realistic alternative, especially at small-to-medium scale [verified: source read,
  arXiv:2406.00873].
- Dose is handled either as a **scalar that scales the drug embedding** (CPA's learned
  per-drug "doser" nonlinearity; PRnet's log10-dose weighting of the fingerprint embedding) or
  as a simple log-dose feature concatenated to the input — no published model treats dose as a
  free one-hot without some monotonicity-preserving structure [from abstract/secondary +
  verified: source read for PRnet's RDKit-embedding × log-dose mechanism].
- A growing line of 2024–2026 benchmarking papers (Ahlmann-Eltze et al. 2025 for genetic
  perturbations; Agarwal & Bisht 2026 for drug-response transcriptomics) shows that simple
  linear/retrieval baselines on fixed representations are frequently competitive with, and by
  some metrics beat, deep conditional models — directly supporting this project's baseline
  ladder design [verified: source read for Agarwal & Bisht; from abstract/secondary for
  Ahlmann-Eltze et al.].

## 1. Drug representations used in published perturbation-response models

- **scGen** (Lotfollahi et al., 2019, *Nature Methods*) predicts perturbation response by latent
  vector arithmetic on expression alone and does **not** use any molecular/chemical
  representation of the perturbation — it cannot generalize to unseen drugs by structure. Useful
  as a contrast case (no chemistry input at all). [from abstract/secondary — general knowledge,
  not directly re-verified this session]
- **CPA** (Lotfollahi et al., *Compositional Perturbation Autoencoder*, bioRxiv
  2021.04.14.439903, later *Mol. Syst. Biol.* 2023) encodes each perturbation/drug as a learned
  embedding, additively combined with a basal cell state and covariate (cell-type) embeddings in
  a latent space, then decoded. Dose enters through a **per-drug nonlinear "doser" scalar** that
  scales the drug embedding magnitude, giving a learned dose-response curve rather than a fixed
  linear scaling. [from abstract/secondary]
- **chemCPA** (Hetzel et al., NeurIPS 2022, "Predicting Cellular Responses to Novel Drug
  Perturbations at a Single-Cell Resolution") extends CPA by replacing the learned
  (drug-ID-indexed) embedding with a **fixed-size embedding from an explicit molecule encoder**,
  so the model transfers to drugs unseen in the single-cell data. Their public repo
  (`theislab/chemCPA`) contains a dedicated `embeddings/` folder "one folder for each molecular
  embedding model we benchmarked," consistent with the paper's ablation across RDKit
  fingerprints/descriptors and several pretrained graph/sequence encoders (GROVER, MPNN, JT-VAE,
  Weave, seq2seq, sourced via DeepPurpose/TDC). The architecture surgery / transfer-learning
  result (pretraining on bulk RNA HTS, fine-tuning on single-cell) mattered more to their
  headline result than the specific choice of pretrained embedding. [verified: repo structure
  read; specific ablation numbers not independently confirmed this session —
  from abstract/secondary for the ranking of embeddings]
- **PRnet** (Qin et al., 2024, *Nature Communications*, "Predicting transcriptional responses to
  novel chemical perturbations using deep generative model for drug discovery") represents each
  compound as an **RDKit FCFP4 fingerprint embedding**, and folds dose in by **weighting the
  fingerprint embedding by the log10-scaled dose value** before it enters the Perturb-encoder.
  [from abstract/secondary]
- **biolord** (Piran et al., 2024, reported in *Nature Biotechnology*) disentangles expression
  into attribute-specific subspaces (perturbation, cell state, etc.) and, per secondary sources,
  trains cross-modal encoders linking molecular structure to transcriptomic profiles to
  generalize to unseen compounds; representation quality was assessed via how well a k-NN graph
  of the drug-embedding space recovers known pathway labels. [from abstract/secondary — could
  not fetch primary text this session]
- **PerturBench** (Wenteler et al., arXiv:2408.10609, NeurIPS 2024 track) is a benchmarking
  framework, not a representation paper, but its modular design explicitly separates the
  "chemical representation of the drug" as a pluggable component, and the authors report that
  representation choice interacts with **evaluation metric** (rank-based vs RMSE-based) more than
  any single representation dominating. [from abstract/secondary]
- **Tahoe/Arc "State"** model (Arc Institute, 2025) is trained across Tahoe-100M plus other
  atlases with a bidirectional-transformer architecture; public descriptions emphasize
  cell/perturbation-conditioning generally but did not surface, in this search budget, a
  specific published ablation of drug-representation choice — flagged as **not verified** rather
  than asserted. [unable to confirm — do not treat as evidence either way]

**Takeaway for ablations:** across CPA/chemCPA/PRnet, the representation is always a
fixed-size vector derived from structure (fingerprint or descriptor by default; pretrained graph
embedding as an alternative), never raw SMILES text or a one-hot drug ID when generalization to
unseen compounds is required. The transfer-learning/architecture choices tended to move the
needle more than the specific embedding family in chemCPA's reported story.

## 2. Fixed fingerprints vs pretrained molecular embeddings, small-data regime

- A dedicated 2025 benchmark of pretrained molecular embedding models (ChemBERTa, MolFormer,
  Uni-Mol, GROVER, and others) against classical descriptors (Praski, Adamczyk & Czech,
  "Benchmarking Pretrained Molecular Embedding Models For Molecular Representation Learning,"
  arXiv:2508.06199, 2025) found that pretrained embeddings win **only when there is enough data
  to exploit fine-tuning**; in small-data settings, **fixed fingerprints remain competitive**,
  and the authors explicitly caution against assuming learned representations are always better
  — recommending fingerprints as the pragmatic default for small/resource-constrained problems.
  [verified: source read]
- A second, independently-found benchmark (search snippet, MoleculeNet-style comparison) reported
  ECFP mean AUROC ≈ 79.9% essentially tied with ChemBERTa (≈ 80.0%) across a broad task panel,
  with only a handful of specialized fusion models (e.g., CLAMP, which itself incorporates
  fingerprints) beating plain ECFP outright. [from abstract/secondary]
- Architecturally, ChemBERTa needs on the order of tens of millions of training molecules
  (~77M in its pretraining corpus) to learn useful chemistry-agnostic SMILES-token
  representations, while chemistry-aware tokenizations (e.g. Morgan-fingerprint-informed) reach
  comparable performance with far less pretraining data — but neither pretraining corpus size is
  the bottleneck here; the bottleneck is **this project's own fine-tuning set (~92 drugs)**, far
  below where any of these pretrained encoders have been shown to add value over fingerprints.
  [from abstract/secondary]

**Implication:** with ~92 unique compounds, Morgan/ECFP (or ECFP + RDKit 2D descriptors) is the
literature-supported default; a pretrained embedding (ChemBERTa or MolFormer, being the easiest
to pull off-the-shelf) is a reasonable **ablation**, not a default, and should not be expected to
beat fingerprints at this scale per the cited benchmark.

## 3. Scaffold splitting: Bemis–Murcko vs generic scaffolds vs Butina/Tanimoto clustering

- Guo, Hernandez-Hernandez & Ballester, "Scaffold Splits Overestimate Virtual Screening
  Performance," arXiv:2406.00873 (2024), show that Bemis–Murcko scaffold splits can place
  near-identical scaffolds on both sides of the train/test boundary, inflating apparent
  generalization; they recommend **Butina clustering on fingerprints (Tanimoto distance)** as a
  harder, more realistic alternative, and note that "generic" (skeleton-only) scaffolds suffer
  the same fundamental bias as standard Bemis–Murcko scaffolds — they don't fix the problem.
  [verified: source read]
- Separately, general chemoinformatics practice notes (search-aggregated, e.g. Bemis–Murcko
  scaffold discussions and practical blog posts referenced in search results) confirm the
  well-known small-data failure mode: with only hundreds of compounds, a large fraction end up
  as **singleton scaffolds** (each compound is its own scaffold group), which makes a clean
  scaffold-grouped split either trivial (everything is "unseen") or unstable (which compounds
  land in test set swings results a lot across random seeds). [from abstract/secondary]

**Implication for this project (92 drugs):** Bemis–Murcko scaffold grouping, as specified in
the project brief, is reasonable and standard for the "unseen_drug" split, but expect many
singleton-scaffold groups; report split sizes/variance across multiple seeds rather than a
single split, and consider Butina/Tanimoto clustering as a secondary, harder split variant given
the cited evidence that scaffold splits alone can overstate performance.

## 4. Standardization

- RDKit's `Chem.MolStandardize` (and the newer C++-backed `rdMolStandardize`) module provides
  the standard building blocks: `Cleanup` (calls `RemoveHs`, `SanitizeMol`, `MetalDisconnector`,
  `Normalizer`, `Reionizer`, `AssignStereochemistry`), `MetalDisconnector` (breaks
  metal–organic bonds), `Normalizer` (applies SMARTS-based functional-group normalization
  rules), `Reionizer` (fixes charge states), `LargestFragmentChooser`/`FragmentParent` (salt and
  mixture stripping — keep the largest covalent fragment), `Uncharger` (neutralizes where
  possible), and `TautomerEnumerator`/`TautomerParent` (canonical tautomer). RDKit also defines a
  "super parent" (fragment + charge + isotope + stereo + tautomer parent) and a separate
  "stereo parent" that strips all tetrahedral/double-bond stereo — useful if stereo isn't
  resolvable from the source data. [verified: source read, RDKit docs]
- Recommended recipe order (standard practice, consistent with RDKit's own `Cleanup` pipeline):
  parse SMILES → sanitize → disconnect metals → strip salts/solvates (keep largest fragment) →
  normalize functional groups → reionize/neutralize → optionally canonicalize tautomers →
  decide stereo policy (keep if explicit and trustworthy, else strip to stereo-parent) →
  canonicalize SMILES for hashing/deduplication.
- **Compounds that cannot be featurized** (mixtures resolving to no organic fragment, purely
  inorganic entries, or SMILES RDKit fails to sanitize — the brief's own example, Talc, is
  typically deposited as an inorganic silicate fragment/mixture with no meaningful
  fingerprint/scaffold): the RDKit `PipelineResult`/`PipelineStatus` mechanism is built exactly
  for this — it logs parsing/validation failures per molecule so they can be filtered rather than
  silently zero-filled. Practical recommendation: **exclude** such compounds from chemistry-based
  splits and fingerprint features, flag them explicitly in the dataset manifest, and report the
  excluded count. [verified: source read for the Pipeline mechanism; the Talc-specific framing is
  this project's own observation, not sourced]

## 5. Dose encoding

- **CPA / chemCPA**: dose scales the drug embedding through a **learned nonlinear per-drug
  function of dose** ("doser"), producing a smooth, monotonic-ish dose–response curve in latent
  space rather than treating dose as an independent feature. [from abstract/secondary]
- **PRnet**: dose enters as a **log10-scaled scalar multiplying the fingerprint embedding**
  directly — i.e., `dose_weight(dose) × fingerprint_embedding`, no separate dose network.
  [from abstract/secondary, consistent across two independent search summaries]
- No source found in this search recommends per-dose one-hot encoding for a project with only 3
  dose levels per drug where doses are shared/aligned across drugs (0.05/0.5/5 µM) — one-hot
  would discard the known monotonic/ordinal structure of dose that both CPA and PRnet exploit.

**Implication:** the simplest defensible choice consistent with the literature is a **log10-dose
scalar**, either (a) concatenated to the chem⊕cell feature vector for the ridge baseline, and/or
(b) used to scale the chemistry embedding before concatenation for the neural model (chemCPA/PRnet-style).
Given only 3 discrete doses, log-dose-as-scalar is simpler to implement and interpret than a
learned doser network, and is a reasonable simplification at this data scale; per-dose one-hot is
not supported by any cited precedent and should be avoided as a default.

## 6. Evidence that simple baselines match deep models (bearing on representation choice)

- Ahlmann-Eltze, Huber & Anders, "Deep-learning-based gene perturbation effect prediction does
  not yet outperform simple linear baselines," bioRxiv 2024.09.16.613342 → *Nature Methods*
  (2025): across five foundation models (incl. scGPT, scFoundation) and GEARS, none beat a
  simple additive/mean-based linear baseline for single- and double-gene knockout prediction,
  including on unseen-perturbation splits. Note: this is **genetic** (CRISPR) perturbation, not
  chemical/drug — bears on the general "match the baseline ladder carefully" principle but not
  directly on chemical representation choice. [from abstract/secondary]
- Agarwal & Bisht, "The Metric Picks the Winner: Evaluation Choice Flips Model Rankings for
  Drug-Response Prediction in Unseen Chemistry," arXiv:2606.12639 (2026), is the closest
  analogue to this project's own Phase 4 design: on THP-1 DRUG-seq data (~14k training
  compounds) they build almost exactly this project's ladder — untreated/mean-response dumb
  baselines, **Tanimoto-weighted nearest-neighbor retrieval**, **ridge/regularized linear
  regression on Morgan fingerprints**, and deep models using ChemBERTa embeddings — under a
  **Bemis–Murcko scaffold split**. Their key finding: **model ranking inverts depending on the
  evaluation metric** — under an inverse-variance-style metric, ridge-on-fingerprints wins; under
  a "true active gene set" metric, the deep fusion model wins. This directly validates (a) this
  project's baseline ladder as a sound experimental design, and (b) the need to report results
  under more than one metric, since representation-choice conclusions ("fingerprints are enough"
  vs "you need embeddings") can flip with the metric alone. [verified: source read]
- Csendes et al., "Drug Response Prediction Provides a Biologically Relevant Benchmark for
  Perturbation Response Models," bioRxiv 2025.12.09.693213 (2025): per a secondary summary found
  via search, this work reports that **simple baselines outperformed foundation models** for
  post-perturbation gene expression prediction — consistent with the pattern above, but could not
  be fetched directly this session (rate-limited); treat as corroborating, not primary, evidence.
  [from abstract/secondary, unread primary source]

## Recommendation for this project

- **Default compound representation:** Morgan/ECFP fingerprint (radius 2–3, 1024–2048 bits,
  count-based rather than binary) computed from RDKit, optionally concatenated with RDKit 2D
  descriptors for the ridge baseline. *Justification:* the only representation shown, across
  chemCPA, PRnet, and two independent 2025 small-data benchmarks, to be competitive or default at
  this compound-count scale (~92 drugs) [verified: arXiv:2508.06199; from abstract/secondary: chemCPA/PRnet].
- **Dose encoding:** log10(dose) as a scalar — concatenated into the ridge feature vector, and
  used to scale the chemistry embedding for the neural model (chemCPA/PRnet pattern), rather than
  one-hot or a full learned doser network. *Justification:* matches the two published mechanisms
  found (CPA's doser, PRnet's log-dose weighting) while staying simple enough for 3 dose levels
  and a laptop-scale model [from abstract/secondary].
- **Scaffold rule:** Bemis–Murcko scaffold grouping for the primary `unseen_drug`/`both_unseen`
  splits (as already planned), but report scaffold-group size distribution (expect many
  singletons at n=92) and re-run with multiple seeds; add a Butina/Tanimoto-clustering split as a
  secondary, harder check. *Justification:* Bemis–Murcko is standard and matches the closest
  analogue benchmark (Agarwal & Bisht 2026), but is documented to overestimate performance versus
  fingerprint clustering, especially with few compounds [verified: arXiv:2406.00873].
- **Standardization recipe:** RDKit `Cleanup` → `MetalDisconnector` → `LargestFragmentChooser`
  (salts/mixtures) → `Normalizer`/`Reionizer` → canonical SMILES; drop and explicitly log any
  compound RDKit cannot sanitize or that resolves to no organic fragment (e.g. inorganic entries)
  rather than imputing a zero fingerprint. *Justification:* this is RDKit's own documented
  pipeline order, and silently zero-filling unfeaturizable compounds would bias the ridge/neural
  models toward treating "unparseable" as a meaningful chemical signal [verified: RDKit docs].
- **Ablations worth running (1–2, given laptop/€0 budget):**
  1. Fingerprint vs. one off-the-shelf pretrained embedding (ChemBERTa, via a cached/precomputed
     checkpoint — no local pretraining) on the same ridge/neural pipeline, to empirically check
     whether the small-data literature finding (fingerprints ≥ pretrained embeddings at ~100
     compounds) replicates on this slice [verified: arXiv:2508.06199].
  2. Bemis–Murcko split vs. Butina/Tanimoto-clustered split, holding the model fixed, to quantify
     how much the `unseen_drug` performance estimate depends on split method — directly following
     the documented overestimation risk [verified: arXiv:2406.00873].

## References

1. Lotfollahi, M. et al. (2019). *scGen predicts single-cell perturbation responses.* Nature
   Methods. [general knowledge; not re-verified this session]
2. Lotfollahi, M., Susmelj, A. K., et al. (2021/2023). *Compositional perturbation autoencoder
   for single-cell response modeling.* bioRxiv 2021.04.14.439903 (later Mol. Syst. Biol. 2023).
   https://www.biorxiv.org/content/10.1101/2021.04.14.439903
3. Hetzel, L. et al. (2022). *Predicting Cellular Responses to Novel Drug Perturbations at a
   Single-Cell Resolution* (chemCPA). NeurIPS 2022.
   https://proceedings.neurips.cc/paper_files/paper/2022/file/aa933b5abc1be30baece1d230ec575a7-Paper-Conference.pdf
   ; code: https://github.com/theislab/chemCPA
4. Qin, Y. et al. (2024). *Predicting transcriptional responses to novel chemical perturbations
   using deep generative model for drug discovery* (PRnet). Nature Communications.
   https://www.nature.com/articles/s41467-024-53457-1
5. Piran, Z. et al. (2024). *biolord* — disentangled representation learning for single-cell and
   spatial omics, incl. perturbation. Nature Biotechnology (secondary reference only; primary
   text not fetched this session).
6. Wenteler, A. et al. (2024). *PerturBench: Benchmarking Machine Learning Models for Cellular
   Perturbation Analysis.* arXiv:2408.10609. https://arxiv.org/abs/2408.10609
7. Praski, P., Adamczyk, J., Czech, W. (2025). *Benchmarking Pretrained Molecular Embedding
   Models For Molecular Representation Learning.* arXiv:2508.06199.
   https://arxiv.org/pdf/2508.06199
8. Guo, Q., Hernandez-Hernandez, S., Ballester, P. J. (2024). *Scaffold Splits Overestimate
   Virtual Screening Performance.* arXiv:2406.00873. https://arxiv.org/pdf/2406.00873
9. RDKit documentation. *rdkit.Chem.MolStandardize module.*
   http://www.rdkit.org/new_docs/source/rdkit.Chem.MolStandardize.html
10. Ahlmann-Eltze, C., Huber, W., Anders, S. (2024/2025). *Deep-learning-based gene perturbation
    effect prediction does not yet outperform simple linear baselines.* bioRxiv 2024.09.16.613342;
    Nature Methods (2025). https://www.biorxiv.org/content/10.1101/2024.09.16.613342v5 ;
    https://www.nature.com/articles/s41592-025-02772-6
11. Agarwal, D., Bisht, R. (2026). *The Metric Picks the Winner: Evaluation Choice Flips Model
    Rankings for Drug-Response Prediction in Unseen Chemistry.* arXiv:2606.12639.
    https://arxiv.org/abs/2606.12639
12. Csendes, G. et al. (2025). *Drug Response Prediction Provides a Biologically Relevant
    Benchmark for Perturbation Response Models.* bioRxiv 2025.12.09.693213 (secondary reference
    only; primary text not fetched this session — rate-limited).
    https://www.biorxiv.org/content/10.64898/2025.12.09.693213v1.full
13. Tahoe-100M: Zhang, et al. (2025). *Tahoe-100M: A Giga-Scale Single-Cell Perturbation Atlas
    for Context-Dependent Gene Function and Cellular Modeling.* bioRxiv 2025.02.20.639398.
    https://www.biorxiv.org/content/10.1101/2025.02.20.639398v1
