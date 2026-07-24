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

Observed effect: 13 of 47,151 observations (0.028%) fell through to `X`,
distributed GPT-oss-120B (5), Gemini 3.1 FLite (2), Gemma 3 27B (2),
Haiku 4.5 (2), Gemma 3 12B (1), Qwen 80B Inst (1). No model exceeds 5 `X`
records; `X` records are retained and scored incorrect rather than dropped.

## Confidence extraction

Applied in order; first match wins:

1. `Confidence:\s*(\d+)` — the requested format; value clipped to [0, 100].
2. Fallback: first standalone `\d{1,3}` in the response whose value is in
   [0, 100].
3. If neither matches: the value is **imputed as 50**.

Observed effect and one material caveat: the imputed-50 fallback is
consequential for exactly one model. **GPT-oss-120B has confidence == 50 on
65.3% of items**; for every other model the rate is below 0.3%. GPT-oss-120B's
responses frequently omitted a parseable confidence statement, so its
distribution is dominated by the imputation value rather than by expressed
confidence. This is consistent with the model's anomalous profile reported in
the paper (confidence SD 21.3 driven by a bimodal expressed/imputed mixture,
aggregate AUROC .530). Readers should treat GPT-oss-120B's confidence
distribution as partly imputed; its cells are retained for completeness but the
imputation share is disclosed here and flagged in the model's results
discussion. No other model's confidence distribution is materially affected by
imputation.

## Out-of-range, multi-letter, and refusal handling

- Out-of-range confidence (> 100) is clipped to 100 by rule 1; negative values
  are not expressible under `\d+`.
- Multiple answer letters: rule 1 captures the letter immediately following
  `Answer:`; a later stray letter does not override it. Where the format was
  absent, the fallback captures the first standalone A–D, which for these
  models was the intended answer in spot-checks.
- Refusals / non-answers produce no `Answer:` line and no in-range integer,
  yielding `X` (incorrect) and imputed confidence 50.

## Model-level exclusion

Gemma 4 26B A4B was excluded entirely: the model repeatedly failed to emit
parseable output under the fixed template (API-side failures at the
item-response level), leaving too few valid items to screen. Its exclusion is
a parsing/availability decision, not a result-driven one.

## Median binarisation

The aggregate validity screen binarises confidence at each model's own median
before computing L and the KEEP/WITHDRAW-analogue indices. Type-2 AUROC in the
paper is computed on the **continuous** confidence values, not the binarised
series; binarisation affects only the screen, not the AUROC hierarchy.

## Reproduction

`notebooks/atlas_benchmark.py` contains `parse_answer` and `parse_confidence`
verbatim. Running them over the raw `.run.json` files in the HuggingFace
`raw_outputs/` directory regenerates the released CSV columns exactly; the
per-model `X` counts and confidence==50 rates above were computed by that
round-trip.
