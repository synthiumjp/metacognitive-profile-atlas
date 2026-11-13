# Metacognitive Profile Atlas

**Domain-level confidence discrimination in 33 frontier LLMs on MMLU.**

46,157 scored verbalized-confidence observations · 33 models · 8 families · 6 subject-group bins · 1,500 stratified MMLU items

[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset on HF](https://img.shields.io/badge/Dataset-HuggingFace-yellow.svg)](https://huggingface.co/datasets/synthiumjp/metacognitive-profile-atlas)
[![arXiv](https://img.shields.io/badge/arXiv-2605.06673-b31b1b.svg)](https://arxiv.org/abs/2605.06673)

> ## Version 2 (November 2026)
>
> Version 2 of the data and paper (arXiv:2605.06673v2) corrects version 1. The main changes:
>
> - **GPT-oss-120B's anomalous profile was a parsing artefact.** When no confidence could be extracted, the parser substituted 50. This happened on 980 of GPT-oss-120B's 1,500 responses; the model never states 50 itself. Version 1 analysed these values and reported the model as having near-chance discrimination (AUROC .530) and the highest confidence variability in the atlas. On its 518 genuine responses its AUROC is .800. All parser-imputed confidences (985, including five in four other models) and the 13 unparsed answers are now excluded, and every number, table and figure is recomputed.
> - **Binary-probe validity results are corrected.** Version 1 relied on validity-screen classifications from arXiv:2604.17707 / 2604.17714v1, which contained a scoring error. Corrected classifications come from [arXiv:2604.17714v2](https://arxiv.org/abs/2604.17714). DeepSeek-R1 is not "inverted"; it withdraws 95% of its correct answers.
> - **Claims are narrowed** to the population-level domain ordering, which survives precision weighting, bootstrap rank analysis and Holm-corrected contrasts. Per-model profiles have weak split-half reliability and are released as exploratory.
> - **The excluded MMLU subject is `college_medicine`**, not `elementary_mathematics` as v1 stated (the analysis always used the correct mapping).
>
> Appendix A of the v2 paper lists every changed claim. The version 1 data are kept unchanged in `data/raw_v1/`; the v1 release is tagged [`v1.0`](https://github.com/synthiumjp/metacognitive-profile-atlas/tree/v1.0).

## What this is

Type-2 AUROC (how well confidence separates correct from incorrect answers) for 33 frontier LLMs, decomposed by MMLU subject-group bin. Aggregate AUROC can hide domains where confidence carries little information: Gemini 2.5 Pro has an aggregate AUROC of .712, yet discriminates at .889 on Applied items and at chance (.485) on Natural Science items.

**Findings (v2):**

- **Population-level ordering.** Confidence discriminates best on Applied/Professional items (mean AUROC .750) and worst on Formal Reasoning (.655) and Natural Science (.662) (Friedman χ²(5) = 28.26, p < .0001; Kendall's W = .171). Applied ranks first in 100% of 500 bootstrap item resamples; Formal and Science fall in the bottom two in 93% and 92%. After Holm correction Applied exceeds Formal, Science, Social and Humanities; no other pair survives, including Applied vs Factual. The ordering holds when wide-interval or few-error cells are excluded or down-weighted.
- **Reliability.** Aggregate AUROC is reproducible on half the items (r = .868). The Applied-minus-(Formal, Science) contrast has split-half r = .249 (Spearman–Brown .40), sign replicated in 50/50 splits. Full six-domain profiles within a model are not reliable (median split-half r = .157).
- **Families (exploratory).** Within-family profile-shape similarity exceeds between-family similarity (.406 vs .091; permutation p = .0001). After Holm correction only Anthropic is individually supported.
- **Generations.** Paired-item decomposition separates monitoring gains from error removal. Opus 4.1 → 4.5 lowers confidence on persisting errors (−7.9 points, p = 3e-7). Gemma 3 27B → Gemma 4 31B (+.206 AUROC) mainly removes confident errors (82.6% fixed) without lowering confidence on those that remain.
- **Probe format.** Battery AUROC under a binary KEEP/WITHDRAW probe does not predict atlas AUROC (ρ = .00). R1, Gemini 3.1 Pro and Qwen Think, which fail or cannot be certified on the binary probe, discriminate well here; Gemma 3 1B, 12B and 27B show the reverse.

**Scope.** All results are MMLU-specific and use one verbalized-confidence prompt with greedy decoding. Type-2 AUROC measures discrimination, not calibration. The six bins are a pragmatic grouping, not a validated construct: subjects within a bin are no more alike than subjects across bins (coherence ratio 1.00).

**Paper**: Cacioli, J. P. (2026). The Metacognitive Profile Atlas: Domain-level confidence discrimination in 33 frontier LLMs on MMLU. [arXiv:2605.06673v2](https://arxiv.org/abs/2605.06673). Source and PDF in [`paper/v2/`](paper/v2/).

## Quick start

```python
import pandas as pd

cis = pd.read_csv("data/atlas_bootstrap_cis.csv")
err = pd.read_csv("data/atlas_cell_error_counts.csv")

# Domain-level AUROC for one model, with intervals
print(cis[cis["model"] == "Opus 4.6"][["domain", "auroc", "ci_lo", "ci_hi", "ci_w"]])
```

Read cells against their intervals and error counts: the median cell has 20 errors, 28 cells have fewer than 10, and 36% of intervals are wider than .25 (mostly high-accuracy models).

## Using the atlas

1. **Screen** the confidence signal first (`pip install validity-screen`; [validity-scaling-llm](https://github.com/synthiumjp/validity-scaling-llm)).
2. **Check aggregate** Type-2 AUROC.
3. **Expect weaker discrimination on formal and scientific content** than on applied professional content. This is a population-level direction, not a per-model lookup: individual model-domain cells are informative only where their intervals are tight.

## Reproducing

```bash
pip install -r requirements.txt
python scripts/v2_clean.py        # data/raw_v1 -> data/raw (exclusions; writes data/v2_exclusions.json)
python reproduce.py               # matrix, bootstrap CIs, inferential tests, validation, figures
python scripts/v2_numbers.py      # every number quoted in the v2 paper -> data/v2_numbers.json
python scripts/v2_tables.py paper/v2
cd scripts/rebuttal && for s in analysis1_precision analysis2_generational analysis3_fast \
    analysis4_contrast final_robustness rank_stability; do python $s.py; done
```

Re-running the full chain from `data/raw_v1/` regenerates `data/raw/`, the bootstrap intervals and the summary matrix byte-for-byte. Bootstrap intervals are seeded per cell, so an interval depends only on its own data.

- **Item pool**: `python scripts/make_mmlu_locators.py` reconstructs the seed-42 stratified sample from `cais/mmlu` and verifies it against the released CSVs. `data/mmlu_item_locators.csv` gives the MMLU (subject, split, row_index) for every item.
- **Parsing**: `python scripts/verify_parsing.py` re-runs the extraction rules over the raw platform outputs (on Hugging Face under `raw_outputs/`) and reproduces the 47,151 v1 records. Rules and per-model fallback counts: `docs/parsing_protocol.md`, `data/parsing_fallback_counts.csv`.
- **GLM-5 retest**: GLM-5's original run stopped at 598 items. A full rerun three months later is in `data/glm5_rerun/`; `python scripts/glm5_matched_retest.py` compares the two on the shared items (domain ordering ρ = .77, correctness agreement 98.5%).

## Repository structure

```
metacognitive-profile-atlas/
├── data/
│   ├── raw/                            # v2 analysed corpus: 33 per-model item-level CSVs
│   ├── raw_v1/                         # v1 corpus as released (includes imputed values)
│   ├── glm5_rerun/                     # GLM-5 full rerun (July 2026)
│   ├── atlas_bootstrap_cis.csv         # 198 cells: AUROC, 95% bootstrap CI
│   ├── atlas_summary_matrix.csv        # 33×6 AUROC matrix
│   ├── atlas_cell_error_counts.csv     # per-cell n, errors, accuracy, AUROC, CI
│   ├── battery_v2.csv                  # corrected binary-probe battery results (20 models)
│   ├── v2_exclusions.json              # per-model excluded records
│   ├── v2_numbers.json                 # numbers quoted in the v2 paper
│   ├── mmlu_item_locators.csv          # item_id → MMLU (subject, split, row_index)
│   ├── parsing_fallback_counts.csv     # per-model extraction route counts
│   └── thinking_conf_*.csv             # DeepSeek-R1 reasoning-block discrepancies
├── docs/parsing_protocol.md            # extraction rules and their incidence
├── notebooks/atlas_benchmark.py        # Kaggle benchmark task (item administration)
├── scripts/                            # 01–07 pipeline, v2_*, verification, rebuttal/
├── figures/                            # 7 PDFs + 7 PNGs
├── paper/v2/                           # arXiv v2 LaTeX source and PDF
├── reproduce.py
├── croissant.json                      # Croissant metadata (Core + RAI)
└── LICENSE / LICENSE-DATA / CITATION.cff
```

Raw model outputs (conversation records, task manifests, per-run outputs) are on [Hugging Face](https://huggingface.co/datasets/synthiumjp/metacognitive-profile-atlas) under `raw_outputs/`. The benchmark task is public at [kaggle.com/benchmarks/tasks/jonpaulcacioli/metacognitive-profile-mmlu](https://www.kaggle.com/benchmarks/tasks/jonpaulcacioli/metacognitive-profile-mmlu/1).

## Data notes

- **Exclusions.** 13 unparsed answers and 985 parser-imputed confidences are excluded (`data/v2_exclusions.json`). In every model with imputed values no response genuinely states 50, so each 50 in those models is identifiable as a parser default. GPT-oss-120B retains 518 items; its imputation rate differs by domain (59–74%, p = .002), so its retained items are not a random sample and its domain profile should not be interpreted.
- **Partial runs.** Sixteen models have fewer than 1,500 scored items: twelve because of platform timeouts (minimum GLM-5, 598) and four only because of the exclusions.
- **DeepSeek-R1** states a confidence inside its reasoning block on many items, and the extractor takes the first match. On 82 items (5.5%) this differs from the final stated value; using the final values moves its AUROC from .769 to .766 (`scripts/thinking_confidence_check.py`).
- **Gemma 4 26B A4B** appears in the task's model list but never executed; no data exist for it.

## Domain mapping

56 of the 57 MMLU test-split subjects were mapped a priori to six bins (250 items per bin, 1,500 total, seed = 42):

| Domain | Example subjects | Items sampled |
|--------|-----------------|---------------|
| Applied/Professional | professional_law, professional_medicine, clinical_knowledge | 250 |
| Factual Recall | high_school_european_history, world_religions, nutrition | 250 |
| Formal Reasoning | abstract_algebra, formal_logic, elementary_mathematics | 250 |
| Humanities/Comprehension | philosophy, high_school_psychology, human_sexuality | 250 |
| Natural Science | high_school_physics, college_chemistry, college_biology | 250 |
| Social/Moral | moral_scenarios, moral_disputes, sociology | 250 |

`college_medicine` was left unmapped as ambiguous between Natural Science and Applied/Professional, and was not sampled. Full mapping in `notebooks/atlas_benchmark.py`.

## Models

33 models from 8 families: Anthropic (8), DeepSeek (3), Google-Gemini (7), Google-Gemma (5), OpenAI (5), Qwen (4), Zhipu (1). Canonical IDs in `data/README.md`.

## Related work (Classical Minds, Modern Machines)

| Paper | arXiv | Topic |
|-------|-------|-------|
| Signal Detectors | [2603.14893](https://arxiv.org/abs/2603.14893) | Type-2 SDT for LLM metacognition |
| Domain-specific efficiency | [2603.25112](https://arxiv.org/abs/2603.25112) | Meta-d' and M-ratio |
| Metacognitive Monitoring Battery | [2604.15702](https://arxiv.org/abs/2604.15702) | Cross-domain battery (its free-text track results are affected by the grading issue corrected in 2604.17714v2) |
| Screen Before You Interpret | [2604.17714v2](https://arxiv.org/abs/2604.17714) | Portable validity screen (corrected; supersedes 2604.17707 and 2604.17716) |
| **Metacognitive Profile Atlas** | **[2605.06673v2](https://arxiv.org/abs/2605.06673)** | **This dataset** |

## Citation

```bibtex
@article{cacioli2026atlas,
  author  = {Cacioli, Jon-Paul},
  title   = {The Metacognitive Profile Atlas: Domain-level confidence discrimination in 33 frontier {LLMs} on {MMLU}},
  year    = {2026},
  journal = {arXiv preprint arXiv:2605.06673v2},
}
```

## License

- **Code**: MIT
- **Data**: CC-BY-4.0

## Contact

Jon-Paul Cacioli — synthium@hotmail.com — ORCID: [0009-0000-7054-2014](https://orcid.org/0009-0000-7054-2014)
