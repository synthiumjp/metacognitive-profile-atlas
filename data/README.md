# Data schema

## Item-level CSVs

`data/raw/` holds the version 2 analysed corpus (46,157 rows). `data/raw_v1/` holds the version 1 corpus exactly as first released (47,151 rows), including 13 unparsed answers (`answer` = X) and 985 confidence values of 50 that the parser imputed when no confidence could be extracted. `scripts/v2_clean.py` derives `raw/` from `raw_v1/` and records what it removed in `v2_exclusions.json`. Files are named by upload order, not by model; each contains one model. Schema:

| column | type | description |
|--------|------|-------------|
| item_id | int | MMLU item index within the 1,500-item stratified sample |
| subject | str | MMLU subject (e.g., "college_biology") |
| domain | str | mapped cognitive domain (one of: applied_professional, factual_recall, formal_reasoning, humanities, natural_science, social_moral) |
| question | str | item text (truncated for storage) |
| answer | str | extracted response letter (A/B/C/D; X = unparsed, v1 only) |
| correct_answer | str | ground-truth letter |
| is_correct | bool | whether answer == correct_answer |
| confidence | int | extracted verbalized confidence, 0-100 (v1 includes parser-imputed 50s) |
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

`reproduce.py` deduplicates defensively with:

```python
data.drop_duplicates(subset=['model', 'item_id', 'domain'], keep='first')
```

The v1 upload contained one byte-identical duplicate file for GLM-5, which is not carried into `raw_v1/`; neither directory now contains duplicate rows. Sixteen models have fewer than 1,500 rows in `raw/`: twelve partial runs because of platform timeouts and four that lost rows only to the v2 exclusions.

## Bootstrap CIs

`atlas_bootstrap_cis.csv` contains bootstrap 95% confidence intervals for all 198 (33 × 6) model-domain AUROC cells.

| column | type | description |
|---|---|---|
| model | str | short display name |
| domain | str | one of Applied / Factual / Human. / Social / Formal / Science |
| n | int | number of items in this cell |
| auroc | float | point-estimate AUROC |
| ci_lo | float | bootstrap 2.5th percentile (1,000 resamples; seed (42, crc32("model|domain")) per cell) |
| ci_hi | float | bootstrap 97.5th percentile |
| ci_w | float | width (ci_hi - ci_lo) |

## Prompt

Every model saw the same elicitation prompt. See `notebooks/atlas_benchmark.py` for the exact template and `docs/parsing_protocol.md` for the extraction rules.

## Other files

| file | description |
|---|---|
| atlas_summary_matrix.csv | 33 × 6 AUROC matrix plus aggregate |
| atlas_cell_error_counts.csv | per cell: n, errors, accuracy, AUROC, CI |
| battery_v2.csv | corrected binary-probe battery results for the 20 overlapping models (five keyed tracks; validity-screen 0.3.0; arXiv:2604.17714v2) |
| v2_exclusions.json | per model: rows, unparsed answers, imputed confidences, rows kept |
| v2_numbers.json | every number quoted in the v2 paper |
| glm5_rerun/ | GLM-5 full rerun (July 2026), same schema |
| mmlu_item_locators.csv | item_id → MMLU (subject, split, row_index) |
| parsing_fallback_counts.csv | per-model counts for each extraction route |
| thinking_conf_deepseek-r1-0528.csv | the 82 R1 items whose first-captured confidence differs from the final stated value |
