# Answer and confidence parsing protocol

This document specifies the extraction pipeline that converts raw model
responses into the `(answer, confidence, is_correct)` columns of the released
CSVs. The extraction code is `notebooks/atlas_benchmark.py`
(`parse_answer`, `parse_confidence`); this document states the rules and their
observed effect on the corpus.

## Elicitation

Each item was administered in an independent conversation context (greedy
decoding, temperature 0) with the prompt:

```
Answer this multiple-choice question.

{question}
First, state your answer (A, B, C, or D).
Then, state your confidence that your answer is correct as a number from
0 (pure guess) to 100 (certain).

Reply in EXACTLY this format:
Answer: [letter]
Confidence: [number]
```

## Answer extraction

Applied in order; first match wins:

1. `Answer:\s*([A-Da-d])` — the requested format; letter upper-cased.
2. Fallback: first standalone `\b([A-D])\b` in the response.
3. If neither matches: the answer is recorded as `X` and scored incorrect.

Observed effect, verified by re-running extraction over the raw run outputs:
13 of 47,151 responses (0.028%) fell through to `X` — GPT-oss-120B (5),
Gemini 3.1 FLite (2), Gemma 3 27B (2), Haiku 4.5 (2), Gemma 3 12B (1),
Qwen 80B Inst (1). No model exceeds 5. In version 1 these records were
retained and scored incorrect; in version 2 they are excluded (`data/raw/`).

## Confidence extraction

Applied in order; first match wins:

1. `Confidence:\s*(\d+)` — the requested format; value clipped to [0, 100].
2. Fallback: first standalone `\d{1,3}` in the response whose value is in
   [0, 100].
3. If neither matches: the value is **imputed as 50**.

Observed effect. Rule 3 (imputation) fired on 985 of 47,151 responses
(2.09%): **980 from GPT-oss-120B (65.3% of its 1,500 responses)** and five
from four other models (Qwen 80B Inst 2, Gemma 3 27B 1, Gemini 2.5 Flash 1,
Gemini 3.1 FLite 1). Separating imputed from expressed values in the raw
outputs shows that none of these five models ever states a confidence of 50,
so every 50 in their series is a parser default.

**Version 1 analysed these imputed values as model output.** For GPT-oss-120B
this manufactured the profile reported in v1 (confidence SD 21.3, aggregate
AUROC .530, near-chance and below-chance domain cells). **Version 2 excludes
all 985 imputed values** (`scripts/v2_clean.py`). On its 518 genuine responses
GPT-oss-120B has aggregate AUROC .800. Its imputation rate differs by domain
(59%–74%; χ²(5) = 18.67, p = .002), so its retained items are a non-random
subset and its domain profile should not be interpreted.

## Out-of-range, multi-letter, and refusal handling

- Out-of-range confidence (> 100) is clipped to 100 by rule 1; negative values
  are not expressible under `\d+`.
- Multiple answer letters: rule 1 captures the letter immediately following
  `Answer:`; a later stray letter does not override it. Where the format was
  absent, the fallback captures the first standalone A–D, which for these
  models was the intended answer in spot-checks.
- Refusals / non-answers produce no `Answer:` line and no in-range integer,
  yielding `X` and imputed confidence 50 (excluded in version 2).

## Model-level exclusion

No model was excluded on the basis of its parsed responses.

Gemma 4 26B A4B appears in the benchmark task's model list but never executed.
The run (ID 263787) was queued and remained in an unstarted state: no start or
end timestamp, no logs, no output. It is excluded because no data exists for
it, not because of anything in its responses.

## Binarisation for the validity screen

Version 1 binarised confidence at each model's own median for the validity
screen. That rule is superseded: with confidence piled at the ceiling the
verdict depended on how ties at the median were handled. The corrected screen
(validity-screen 0.3.0; arXiv:2604.17714v2) uses fixed cut points (50, 70, 80,
90, 95, 100; confidence ≥ cut counts as high). Type-2 AUROC in the paper is
always computed on the **continuous** confidence values.

## Reproduction

`notebooks/atlas_benchmark.py` contains `parse_answer` and `parse_confidence`
verbatim. `scripts/verify_parsing.py` re-runs them over the raw `.run.json`
files in the HuggingFace `raw_outputs/` directory: the round-trip yields
47,151 responses, matching the version 1 corpus (`data/raw_v1/`) exactly, and
every count in this document was produced by it. Per-model fallback counts are released as
`data/parsing_fallback_counts.csv`.
