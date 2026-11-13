# Metacognitive Profile Atlas

**Domain-level metacognitive monitoring quality in 33 frontier LLMs.**

47,151 verbalized-confidence observations · 33 models · 8 families · 6 cognitive domains · 1,500 stratified MMLU items

[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset on HF](https://img.shields.io/badge/Dataset-HuggingFace-yellow.svg)](https://huggingface.co/datasets/synthiumjp/metacognitive-profile-atlas)
[![arXiv](https://img.shields.io/badge/arXiv-2605.06673-b31b1b.svg)](https://arxiv.org/abs/2605.06673)

## What this is

An atlas of Type-2 AUROC (confidence discriminating correct from incorrect responses) decomposed by cognitive domain for 33 frontier LLMs on MMLU. Aggregate AUROC obscures within-model variation; the atlas exposes it.

**Key findings:**

- Applied/Professional knowledge is reliably the easiest of these MMLU domains to monitor (mean AUROC .742, top-2 in 21/33 models). Formal Reasoning and Natural Science are the hardest (bottom-2 in 27/33). The ordering strengthens under precision weighting: excluding imprecise cells widens the extremum gap from .090 to .161 and raises Kendall's W from .164 to .317. Bootstrap rank stability over 500 item-level resamples puts Applied first in 99.8% of resamples, Science in the bottom two in 97.6% and Formal in 89.0%. Holm-corrected pairwise contrasts show only the four Applied-versus-other pairs survive correction: Applied is the one reliably distinct extremum, and Formal/Science form an undifferentiated bottom set.
- Within-family profile-shape clustering is significant at the population level (permutation test, p < .0001). Per-family tests with Holm correction support Anthropic (p_holm = .014); Google-Gemini and Qwen are nominally significant (p = .017) but do not survive correction and are reported as exploratory.
- Generational AUROC gains decompose into distinct mechanisms on paired items. Gemma 4 31B's +.202 over Gemma 3 27B is primarily error composition (it answers 82.6% of Gemma 3's high-confidence errors correctly, with confidence on residual errors unchanged). The Anthropic Opus trajectory shows genuine monitoring improvement (lower confidence on persisting errors, p < 1e-4).
- Three models classified Invalid on binary KEEP/WITHDRAW probes produce valid profiles under verbalized confidence (probe-format specificity).

**Scope.** All results are MMLU-specific; replication on an independent benchmark is untested. Individual model profiles have weak split-half reliability (median r = .167), so interpretation rests on the population-level extremum contrast, which is reliable (split-half r = .342, Spearman-Brown full-length .51; sign replicated in 50/50 splits). Per-model profiles and family clustering are released as exploratory structure. Type-2 AUROC measures discrimination, not calibration.

**Paper**: Cacioli, J. P. (2026). Domain-level metacognitive monitoring in frontier LLMs: A 33-model atlas. [arXiv:2605.06673](https://arxiv.org/abs/2605.06673).

## Quick start

```python
import pandas as pd
from datasets import load_dataset

# Load from HuggingFace
ds = load_dataset("synthiumjp/metacognitive-profile-atlas")

# Or load bootstrap CIs directly
cis = pd.read_csv("data/atlas_bootstrap_cis.csv")

# Domain-level AUROC for a specific model, with intervals
model_profile = cis[cis["model"] == "Opus 4.6"]
print(model_profile[["domain", "auroc", "ci_lo", "ci_hi", "ci_w"]])
```

Read cells against their intervals: 34% exceed .25 width, concentrated in high-accuracy models where few errors produce sparse contingency tables. Per-cell error counts are in `data/atlas_cell_error_counts.csv`.

## Evaluation workflow

1. **Screen**: run the portable validity screen (`pip install validity-screen`) on the model's aggregate confidence data. If Invalid, the signal is uninformative and profiling is pointless.
2. **Aggregate**: check overall Type-2 AUROC.
3. **Adjust**: apply the population-level finding — confidence discrimination is systematically weaker on Formal Reasoning and Natural Science content than on Applied/Professional content across nearly every model tested, so a domain-blind abstention threshold is miscalibrated in a predictable direction.

Step 3 is a directional correction supported by the cross-model contrast, not a lookup of any single cell. Individual model-domain cells are informative only where their intervals are tight.

## Repository structure

```
metacognitive-profile-atlas/
├── data/
│   ├── README.md                       # schema + canonical model-name mapping
│   ├── raw/                            # 33 per-model item-level CSVs
│   ├── atlas_bootstrap_cis.csv         # 198-row bootstrap CIs
│   ├── atlas_summary_matrix.csv        # 33×6 AUROC matrix
│   ├── atlas_cell_error_counts.csv     # per-cell n, errors, accuracy, AUROC, CI
│   ├── mmlu_item_locators.csv          # item_id → MMLU (subject, split, row_index)
│   ├── parsing_fallback_counts.csv     # per-model extraction fallback counts
│   └── thinking_conf_*.csv             # reasoning-block confidence discrepancies
├── docs/
│   └── parsing_protocol.md             # answer/confidence extraction rules
├── notebooks/
│   └── atlas_benchmark.py              # Kaggle benchmark task (item administration)
├── scripts/
│   ├── 01_load_data.py … 07_figures.py # analysis pipeline
│   ├── make_mmlu_locators.py           # regenerate + verify the item pool
│   ├── verify_parsing.py               # re-run extraction over raw outputs
│   ├── thinking_confidence_check.py    # reasoning-block extraction diagnostic
│   ├── patch_croissant.py              # metadata maintenance
│   └── rebuttal/                       # precision weighting, rank stability,
│                                       #   generational decomposition, per-family
│                                       #   permutation, pairwise contrasts,
│                                       #   reliability
├── figures/                            # 7 PDFs + 7 PNGs at 300 dpi
├── reproduce.py                        # single-command regeneration
├── croissant.json                      # Croissant metadata (Core + RAI)
└── LICENSE / LICENSE-DATA / CITATION.cff
```

Raw model outputs (complete conversation records, task manifests, per-run outputs as returned by the platform) are on HuggingFace under `raw_outputs/`. The benchmark task is public at [kaggle.com/benchmarks/tasks/jonpaulcacioli/metacognitive-profile-mmlu](https://www.kaggle.com/benchmarks/tasks/jonpaulcacioli/metacognitive-profile-mmlu/1).

## Reproducibility

Every step is auditable end to end:

- **Item pool**: `python scripts/make_mmlu_locators.py` reconstructs the seed-42 stratified sample from `cais/mmlu` and verifies it against the released CSVs (1,500 rows; 100% agreement on subject, domain, and question text). `data/mmlu_item_locators.csv` gives the canonical MMLU (subject, split, row_index) tuple for every item.
- **Parsing**: `python scripts/verify_parsing.py` re-runs the extraction rules over the raw outputs, regenerating 47,151 observations — matching the released corpus exactly. Rules and their per-model incidence are documented in `docs/parsing_protocol.md`.
- **Analysis**: `python reproduce.py` regenerates the matrix, CIs, and inferential tests. `scripts/rebuttal/` reproduces the precision-weighting, bootstrap rank-stability, generational-decomposition, per-family permutation, Holm-corrected pairwise-contrast, and reliability analyses.

**One parsing caveat.** The confidence extractor imputes 50 when no value parses. This fired on 985 of 47,151 responses (2.09%), of which 980 belong to GPT-oss-120B alone (65.3% of its items); five across the other 32 models combined. GPT-oss-120B has no response stating a confidence of 50, so every 50 in its series is imputed. Treat its confidence distribution as partly imputed.

**One reasoning-model detail.** DeepSeek-R1 is the only model returning an explicit `<think>` block, and states a confidence value inside it on 981 of 1,500 items. Because extraction takes the first match in the full response, the recorded value is the mid-reasoning one where present; it differs from the final-block value on 82 responses (5.5%), and substituting final-block values throughout moves R1's aggregate AUROC from .769 to .766. `scripts/thinking_confidence_check.py` reproduces this.

**One exclusion.** Gemma 4 26B A4B appears in the task's model list but never executed (queued, no start timestamp, no logs, no output). It is excluded because no data exists for it. No model was excluded on the basis of its parsed responses.

## Domain mapping

56 of the 57 MMLU test-split subjects mapped a priori to six cognitive-domain bins (250 items per domain, 1,500 total, seed = 42):

| Domain | Example subjects | Items sampled |
|--------|-----------------|---------------|
| Applied/Professional | professional_law, professional_medicine, clinical_knowledge | 250 |
| Factual Recall | high_school_european_history, world_religions, nutrition | 250 |
| Formal Reasoning | abstract_algebra, formal_logic, college_mathematics | 250 |
| Humanities/Comprehension | philosophy, high_school_psychology, human_sexuality | 250 |
| Natural Science | high_school_physics, college_chemistry, college_biology | 250 |
| Social/Moral | moral_scenarios, moral_disputes, sociology | 250 |

`college_medicine` was left unmapped as ambiguous between natural science and applied/professional knowledge, and was not sampled. Full mapping in `notebooks/atlas_benchmark.py`.

## Domain taxonomy: what it is and is not

The six-domain grouping is a pragmatic MMLU-subject taxonomy, not a validated latent cognitive construct. A subject-level coherence analysis (reported in the paper and reproduced by `reproduce.py`) gives a within-domain similarity ratio of 0.95: subjects inside a mapped domain are not empirically more similar to each other than to subjects in other domains. Treat the atlas as a benchmark-conditioned profile under a useful-but-unvalidated taxonomy, not as a map of latent metacognitive domains.

## Models

33 models from 8 families: Anthropic (8), DeepSeek (3), Google-Gemini (7), Google-Gemma (5), OpenAI (5), Qwen (4), Zhipu (1). Full list with canonical IDs in `data/README.md`.

## Related work (Classical Minds programme)

| Paper | arXiv | Topic |
|-------|-------|-------|
| P1: Signal Detectors | [2603.14893](https://arxiv.org/abs/2603.14893) | Type-2 SDT for LLM metacognition |
| P2: Domain-specific efficiency | [2603.25112](https://arxiv.org/abs/2603.25112) | Meta-d' and M-ratio |
| P3: Metacognitive Monitoring Battery | [2604.15702](https://arxiv.org/abs/2604.15702) | Cross-domain benchmark |
| P4a: Validity scaling | [2604.17707](https://arxiv.org/abs/2604.17707) | Six validity indices |
| P4b: Screen before you interpret | [2604.17714](https://arxiv.org/abs/2604.17714) | Portable validity protocol |
| P4c: Selective prediction | [2604.17716](https://arxiv.org/abs/2604.17716) | Concurrent criterion validation |
| **P5: Atlas** | **[2605.06673](https://arxiv.org/abs/2605.06673)** | **This paper** |

## Citation

```bibtex
@article{cacioli2026atlas,
  author  = {Cacioli, Jon-Paul},
  title   = {Domain-level metacognitive monitoring in frontier {LLMs}: {A} 33-model atlas},
  year    = {2026},
  journal = {arXiv preprint arXiv:2605.06673},
}
```

## License

- **Code**: MIT
- **Data**: CC-BY-4.0

## Contact

Jon-Paul Cacioli — synthium@hotmail.com — ORCID: [0009-0000-7054-2014](https://orcid.org/0009-0000-7054-2014)
