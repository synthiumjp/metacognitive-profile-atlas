# Data schema

## Item-level CSVs

One file per model in `data/raw/`, schema:

| column | type | description |
|--------|------|-------------|
| item_id | int | MMLU item index within the 1,500-item stratified sample |
| subject | str | MMLU subject (e.g., "college_biology") |
| domain | str | mapped cognitive domain (one of: applied_professional, factual_recall, formal_reasoning, humanities, natural_science, social_moral) |
| question | str | item text (truncated for storage) |
| answer | str | model's response letter (A/B/C/D) |
| correct_answer | str | ground-truth letter |
| is_correct | bool | whether answer == correct_answer |
| confidence | int | model's verbalized confidence, 0-100 |
| model | str | canonical model ID (see below) |

## Canonical model names

The 33 models span 8 families. The paper uses short display names (`Opus 4.6`); the raw CSVs use canonical IDs (`anthropic/claude-opus-4-6@default`). Mapping:

| Short name | Canonical ID | Family |
|---|---|---|
| Opus 4.6 | anthropic/claude-opus-4-6@default | Anthropic |
| Opus 4.5 | anthropic/claude-opus-4-5@20251101 | Anthropic |
| Opus 4.7 | anthropic/claude-opus-4-7@default | Anthropic |
| Opus 4.1 | anthropic/claude-opus-4-1@20250805 | Anthropic |
| Sonnet 4.6 | anthropic/claude-sonnet-4-6@default | Anthropic |
| Sonnet 4.5 | anthropic/claude-sonnet-4-5@20250929 | Anthropic |
| Sonnet 4 | anthropic/claude-sonnet-4@20250514 | Anthropic |
| Haiku 4.5 | anthropic/claude-haiku-4-5@20251001 | Anthropic |
| DeepSeek-R1 | deepseek-ai/deepseek-r1-0528 | DeepSeek |
| DeepSeek V3.2 | deepseek-ai/deepseek-v3.2 | DeepSeek |
| DeepSeek V3.1 | deepseek-ai/deepseek-v3.1 | DeepSeek |
| Gemini 3.1 Pro | google/gemini-3.1-pro-preview | G-Gemini |
| Gemini 3 Flash | google/gemini-3-flash-preview | G-Gemini |
| Gemini 2.5 Flash | google/gemini-2.5-flash | G-Gemini |
| Gemini 2.5 Pro | google/gemini-2.5-pro | G-Gemini |
| Gemini 3.1 FLite | google/gemini-3.1-flash-lite-preview | G-Gemini |
| Gemini 2.0 FLite | google/gemini-2.0-flash-lite | G-Gemini |
| Gemini 2.0 Flash | google/gemini-2.0-flash | G-Gemini |
| Gemma 4 31B | google/gemma-4-31b | G-Gemma |
| Gemma 3 27B | google/gemma-3-27b | G-Gemma |
| Gemma 3 12B | google/gemma-3-12b | G-Gemma |
| Gemma 3 4B | google/gemma-3-4b | G-Gemma |
| Gemma 3 1B | google/gemma-3-1b | G-Gemma |
| GPT-oss-20B | openai/gpt-oss-20b | OpenAI |
| GPT-oss-120B | openai/gpt-oss-120b | OpenAI |
| GPT-5.4 | openai/gpt-5.4-2026-03-05 | OpenAI |
| GPT-5.4 mini | openai/gpt-5.4-mini-2026-03-17 | OpenAI |
| GPT-5.4 nano | openai/gpt-5.4-nano-2026-03-17 | OpenAI |
| Qwen Think | qwen/qwen3-next-80b-a3b-thinking | Qwen |
| Qwen 80B Inst | qwen/qwen3-next-80b-a3b-instruct | Qwen |
| Qwen Coder | qwen/qwen3-coder-480b-a35b-instruct | Qwen |
| Qwen 235B | qwen/qwen3-235b-a22b-instruct-2507 | Qwen |
| GLM-5 | zai/glm-5 | Zhipu |

## Deduplication

The raw upload may contain duplicate item-model rows from overlapping benchmark runs. `reproduce.py` deduplicates with:

```python
data.drop_duplicates(subset=['model', 'item_id', 'domain'], keep='first')
```

After deduplication: 33 models × approximately 1,500 items = 47,151 observations (twelve models have partial runs; see paper §4.7).

## Bootstrap CIs

`atlas_bootstrap_cis.csv` contains bootstrap 95% confidence intervals for all 198 (33 × 6) model-domain AUROC cells.

| column | type | description |
|---|---|---|
| model | str | short display name |
| domain | str | one of Applied / Factual / Human. / Social / Formal / Science |
| n | int | number of items in this cell |
| auroc | float | point-estimate AUROC |
| ci_lo | float | bootstrap 2.5th percentile (1,000 resamples, seed=42) |
| ci_hi | float | bootstrap 97.5th percentile |
| ci_w | float | width (ci_hi - ci_lo) |

## Prompt

Every model saw the same elicitation prompt. See `notebooks/benchmark.ipynb` for the exact template.
