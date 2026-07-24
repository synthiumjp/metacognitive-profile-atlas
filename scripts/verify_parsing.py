"""
Verify answer/confidence extraction fallback rates directly from the raw
Kaggle run.json outputs, rather than inferring them from the parsed CSVs.

Distinguishes genuine "Confidence: 50" responses from imputed 50s, and
produces the per-model discard/fallback counts cited in
docs/parsing_protocol.md.

RUN FROM THE REPO ROOT (after kaggle b t download):
    python scripts/verify_parsing.py
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path("raw_outputs/metacognitive-profile-mmlu")

# Extraction rules, verbatim from notebooks/atlas_benchmark.py
RE_ANS_PRIMARY = re.compile(r"[Aa]nswer:\s*([A-Da-d])")
RE_ANS_FALLBACK = re.compile(r"\b([A-D])\b")
RE_CONF_PRIMARY = re.compile(r"[Cc]onfidence:\s*(\d+)")
RE_CONF_FALLBACK = re.compile(r"\b(\d{1,3})\b")


def classify(text):
    """Return (answer_route, confidence_route, conf_value)."""
    if RE_ANS_PRIMARY.search(text):
        a_route = "primary"
    elif RE_ANS_FALLBACK.search(text):
        a_route = "fallback"
    else:
        a_route = "X"

    m = RE_CONF_PRIMARY.search(text)
    if m:
        return a_route, "primary", min(100, max(0, int(m.group(1))))
    for n in RE_CONF_FALLBACK.findall(text):
        if 0 <= int(n) <= 100:
            return a_route, "fallback", int(n)
    return a_route, "imputed", 50


def responses_from_run(path):
    """Yield model response text for each mmlu_* conversation in a run.json."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for conv in data.get("conversations", []):
        if not str(conv.get("id", "")).startswith("mmlu_"):
            continue
        try:
            parts = conv["requests"][0]["contents"][-1]["parts"]
            yield "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError):
            continue


rows = []
for run_json in sorted(ROOT.rglob("*.run.json")):
    model = run_json.parent.parent.name
    counts = {
        "ans_primary": 0, "ans_fallback": 0, "ans_X": 0,
        "conf_primary": 0, "conf_fallback": 0, "conf_imputed": 0,
        "conf_genuine_50": 0, "n": 0,
    }
    for text in responses_from_run(run_json):
        a, c, v = classify(text)
        counts["n"] += 1
        counts[f"ans_{a}"] += 1
        counts[f"conf_{c}"] += 1
        if c != "imputed" and v == 50:
            counts["conf_genuine_50"] += 1
    counts["model"] = model
    rows.append(counts)
    print(f"{model:32s} n={counts['n']:5d}  "
          f"X={counts['ans_X']:3d}  imputed={counts['conf_imputed']:5d} "
          f"({counts['conf_imputed'] / max(counts['n'], 1):6.1%})  "
          f"genuine50={counts['conf_genuine_50']:4d}")

df = pd.DataFrame(rows)[
    ["model", "n", "ans_primary", "ans_fallback", "ans_X",
     "conf_primary", "conf_fallback", "conf_imputed", "conf_genuine_50"]
]
df["imputed_rate"] = (df["conf_imputed"] / df["n"]).round(4)
df = df.sort_values("imputed_rate", ascending=False)

print("\n" + "=" * 70)
print("TOTALS")
print(f"  responses parsed:        {df['n'].sum():,}")
print(f"  answer fell through to X: {df['ans_X'].sum()} "
      f"({df['ans_X'].sum() / df['n'].sum():.3%})")
print(f"  confidence imputed:       {df['conf_imputed'].sum():,} "
      f"({df['conf_imputed'].sum() / df['n'].sum():.2%})")
print("\nTop 5 by imputation rate:")
print(df.head(5)[["model", "n", "conf_imputed", "imputed_rate",
                  "conf_genuine_50"]].to_string(index=False))

df.to_csv("data/parsing_fallback_counts.csv", index=False)
print("\nWrote data/parsing_fallback_counts.csv")
