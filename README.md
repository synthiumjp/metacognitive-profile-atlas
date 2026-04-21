# Metacognitive Profile Atlas

A domain-stratified benchmark for LLM metacognitive monitoring.

This repository accompanies the paper:

> Cacioli, J. P. (2026). Domain-level metacognitive monitoring in frontier LLMs: A 33-model atlas. arXiv preprint. [link pending]

## What is this?

Aggregate Type-2 AUROC reports one number per model. This hides within-model variation across cognitive demands. The atlas measures Type-2 AUROC per model-domain cell for 33 frontier LLMs across six MMLU-derived domain bins (Applied/Professional, Factual Recall, Formal Reasoning, Humanities, Natural Science, Social/Moral), using 1,500 MMLU items with verbalized confidence (0-100) and greedy decoding. Total: 47,151 observations.

The main findings:

- Applied/Professional knowledge is reliably the easiest benchmark domain to monitor (mean AUROC = .742)
- Formal Reasoning and Natural Science are reliably the hardest (.658 and .652)
- The middle three domains are statistically indistinguishable
- Family-level profile shape clusters significantly for Anthropic, Google-Gemini, and Qwen (permutation p < .0001)
- Three models classified Invalid under binary KEEP/WITHDRAW probes produce valid profiles under verbalized confidence — probe-format specificity

## Companion work

This benchmark is part of a programme applying clinical psychometric methodology to LLM evaluation.

| Paper | arXiv | Role |
|-------|-------|------|
| Before You Interpret the Profile | 2604.17707 | Validity scaling derivation (20 models, 524 items) |
| Screen Before You Interpret | 2604.17714 | Portable validity protocol |
| Concurrent Criterion Validation | 2604.17716 | Selective prediction validation |
| Metacognitive Monitoring Battery | 2604.15702 | Classical Minds 524-item battery |
| Metacognitive Profile Atlas | [this repo] | **This benchmark** |

The validity screening tool used for Stage A screening before AUROC analysis is available as a separate dependency:

```bash
pip install validity-screen
```

Source: https://github.com/synthiumjp/validity-scaling-llm

## Reproducibility

```bash
git clone https://github.com/synthiumjp/metacognitive-profile-atlas
cd metacognitive-profile-atlas
pip install -r requirements.txt
python reproduce.py
```

`reproduce.py` rebuilds the 33 × 6 AUROC matrix (Table 3), bootstrap 95% CIs for all 198 cells (Supplementary Table S1), the Friedman test on domain difficulty, the family permutation test (10,000 shuffles), split-half stability (aggregate and profile level), and all seven figures. Runtime: approximately 60 seconds on a laptop CPU.

## Layout

```
.
├── README.md                  # this file
├── LICENSE                    # MIT (code)
├── LICENSE-DATA               # CC-BY-4.0 (data)
├── CITATION.cff               # citation metadata
├── requirements.txt           # Python dependencies
├── reproduce.py               # single-command regeneration
├── data/                      # 33 model-level CSVs + aggregates
│   ├── README.md              # schema and canonical model names
│   ├── atlas_bootstrap_cis.csv
│   └── raw/                   # per-model item-level CSVs
├── figures/                   # 7 PDFs + 7 PNGs at 300 dpi
├── notebooks/                 # Kaggle benchmark notebook
└── scripts/                   # individual analysis scripts
```

## Dataset

- **Substrate**: 1,500 MMLU items (Hendrycks et al., 2021), 250 per domain, stratified sampling (seed=42)
- **Domain mapping**: 56 of 57 MMLU subjects mapped a priori to six cognitive domains. One subject excluded (173 items). Full mapping in the benchmark notebook.
- **Models**: 33 frontier LLMs from 8 families (Anthropic, DeepSeek, Google-Gemini, Google-Gemma, OpenAI, Qwen, Zhipu)
- **Elicitation**: verbalized confidence 0-100, greedy decoding, single-turn, independent context per item
- **Platform**: Kaggle Benchmarks (kbench SDK)
- **Total observations**: 47,151

## Leaderboard

A public leaderboard is maintained at the Kaggle Benchmarks platform:
https://kaggle.com/benchmarks/jonpaulcacioli/metacognitive-profile-atlas (link pending)

## Domain taxonomy: what it is and is not

The six-domain grouping is a pragmatic MMLU-subject taxonomy, not a validated latent cognitive construct. A subject-level coherence analysis (reported in §3.8 of the paper and reproduced by `reproduce.py`) shows within-domain similarity ratio = 0.97, meaning subjects inside a mapped domain are not empirically more similar to each other than to subjects in other domains. Readers should treat the atlas as a benchmark-conditioned profile under a useful-but-not-validated taxonomy, not as a map of latent metacognitive domains.

## License

Code: MIT
Data: CC-BY-4.0

See `LICENSE` and `LICENSE-DATA`.

## Citation

If you use this benchmark, please cite:

```bibtex
@misc{cacioli2026atlas,
  author = {Cacioli, Jon-Paul},
  title = {Domain-level metacognitive monitoring in frontier LLMs: A 33-model atlas},
  year = {2026},
  url = {https://github.com/synthiumjp/metacognitive-profile-atlas},
  note = {arXiv pending}
}
```

## Contact

Jon-Paul Cacioli (Melbourne, AU)
ORCID: 0009-0000-7054-2014

Issues and pull requests welcome.
