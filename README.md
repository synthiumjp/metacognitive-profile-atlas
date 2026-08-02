# Metacognitive Profile Atlas

**Domain-level metacognitive monitoring quality in 33 frontier LLMs.**

47,151 verbalized-confidence observations · 33 models · 8 families · 6 cognitive domains · 1,500 stratified MMLU items

[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset on HF](https://img.shields.io/badge/Dataset-HuggingFace-yellow.svg)](https://huggingface.co/datasets/synthiumjp/metacognitive-profile-atlas)

## What this is

An atlas of Type-2 AUROC (confidence discriminating correct from incorrect responses) decomposed by cognitive domain for 33 frontier LLMs. Aggregate AUROC obscures within-model variation; the atlas reveals it.

**Key findings:**

- Applied/Professional knowledge is reliably the easiest domain to monitor (mean AUROC .742, top-2 in 21/33 models). Formal Reasoning and Natural Science are the hardest (bottom-2 in 27/33).
- Within-family profile-shape clustering is significant overall (permutation test, p < .0001), carried by Anthropic, Google-Gemini, and Qwen.
- Gemma 4 31B shows +.202 AUROC over Gemma 3 27B, the largest single-generation gain.
- Three models classified Invalid on binary probes produce valid profiles under verbalized confidence (probe-format specificity).

**Paper**: Cacioli, J. P. (2026). Domain-level metacognitive monitoring in frontier LLMs: A 33-model atlas. arXiv:[ID pending].

## Quick start

```python
import pandas as pd
from datasets import load_dataset

# Load from HuggingFace
ds = load_dataset("synthiumjp/metacognitive-profile-atlas")

# Or load bootstrap CIs directly
cis = pd.read_csv("data/atlas_bootstrap_cis.csv")

# Domain-level AUROC for a specific model
model_profile = cis[cis["model"] == "Opus 4.6"]
print(model_profile[["domain", "auroc", "ci_lo", "ci_hi"]])
```

## Three-step evaluation workflow

1. **Screen**: Run the portable validity screen (`pip install validity-screen`) on the model's aggregate confidence data. If Invalid, stop.
2. **Aggregate**: Check overall Type-2 AUROC.
3. **Profile**: Consult the domain-level AUROC for the domain relevant to the intended deployment.

The atlas provides step 3. Screening (step 1) is a prerequisite: see [validity-scaling-llm](https://github.com/synthiumjp/validity-scaling-llm).

## Repository structure

```
metacognitive-profile-atlas/
├── data/
│   ├── [model_name].csv          # 33 model CSVs (item-level results)
│   ├── atlas_bootstrap_cis.csv   # 198-row bootstrap CIs
│   └── atlas_summary_matrix.csv  # 33×6 AUROC matrix
├── notebooks/
│   └── atlas_analysis.ipynb      # Full analysis pipeline
├── figures/
│   ├── fig1_heatmap.pdf
│   ├── fig2_domain_hierarchy.pdf
│   ├── fig3_ipsative.pdf
│   ├── fig4_family_means.pdf
│   ├── fig5_generational.pdf
│   ├── fig6_splithalf.pdf
│   └── fig7_cross_benchmark.pdf
├── croissant.json                # Croissant metadata (Core + RAI)
├── README.md
└── LICENSE
```

## Domain mapping

56 of 57 MMLU subjects mapped a priori to six cognitive-domain bins (250 items per domain, 1,500 total, seed = 42):

| Domain | Example subjects | Items sampled |
|--------|-----------------|---------------|
| Applied/Professional | professional_law, professional_medicine, clinical_knowledge | 250 |
| Factual Recall | high_school_european_history, world_religions, nutrition | 250 |
| Formal Reasoning | abstract_algebra, formal_logic, college_mathematics | 250 |
| Humanities/Comprehension | philosophy, high_school_psychology, human_sexuality | 250 |
| Natural Science | high_school_physics, college_chemistry, college_biology | 250 |
| Social/Moral | moral_scenarios, moral_disputes, sociology | 250 |

`elementary_mathematics` excluded as ambiguous.

## Models

33 models from 8 families: Anthropic (8), DeepSeek (3), Google-Gemini (7), Google-Gemma (5), OpenAI (5), Qwen (4), Zhipu (1). Full list with canonical IDs in `data/README.md`.

## Related work (Classical Minds programme)

| Paper | arXiv | Topic |
|-------|-------|-------|
| P1: Signal Detectors | [2603.14893](https://arxiv.org/abs/2603.14893) | Type-2 SDT for LLM metacognition |
| P2: Domain-specific efficiency | [2603.25112](https://arxiv.org/abs/2603.25112) | Meta-d' and M-ratio |
| P3: Metacognitive Monitoring Battery | [2604.15702](https://arxiv.org/abs/2604.15702) | Cross-domain benchmark (M6) |
| P4a: Validity scaling | [2604.17707](https://arxiv.org/abs/2604.17707) | Six validity indices |
| P4b: Screen before you interpret | [2604.17714](https://arxiv.org/abs/2604.17714) | Portable validity protocol |
| P4c: Selective prediction | [2604.17716](https://arxiv.org/abs/2604.17716) | Concurrent criterion validation |
| **P5: Atlas** | **[ID pending]** | **This paper** |

## Citation

```bibtex
@article{cacioli2026atlas,
  author = {Cacioli, Jon-Paul},
  title = {Domain-level metacognitive monitoring in frontier {LLMs}: {A} 33-model atlas},
  year = {2026},
  journal = {arXiv preprint arXiv:[ID pending]},
}
```

## License

- **Code**: MIT
- **Data**: CC-BY-4.0

## Contact

Jon-Paul Cacioli — synthium@hotmail.com — ORCID: [0009-0000-7054-2014](https://orcid.org/0009-0000-7054-2014)
