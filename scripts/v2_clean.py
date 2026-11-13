"""
Version 2 cleaning: data/raw_v1/ (as released with v1) -> data/raw/.

Two kinds of record are removed before any analysis:
  1. Unparsed answers (answer == 'X').
  2. Parser-imputed confidences. When no confidence could be extracted, the
     v1 parser substituted 50. The CSVs do not mark these, but the per-model
     route counts recovered from the raw Kaggle outputs
     (data/parsing_fallback_counts.csv, scripts/verify_parsing.py) do. In every
     model with imputed values the count of *genuine* "Confidence: 50"
     responses is zero, so every 50 in those models is a parser default and is
     removed. Models with genuine 50s have no imputed values and are untouched.

Writes data/raw/*.csv (same file names) and data/v2_exclusions.json.

RUN FROM THE REPO ROOT:  python scripts/v2_clean.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw_v1"
DST = ROOT / "data" / "raw"
FB = pd.read_csv(ROOT / "data" / "parsing_fallback_counts.csv")

# Route counts use the bare model id; CSVs use "provider/model[@version]".
def bare(m):
    m = m.split("/", 1)[-1].split("@", 1)[0]
    return m

imputed_models = set(FB.loc[FB.conf_imputed > 0, "model"])
assert (FB.loc[FB.conf_imputed > 0, "conf_genuine_50"] == 0).all(), \
    "a model has both imputed and genuine 50s; rule cannot separate them"

def fb_key(csv_model):
    b = bare(csv_model)
    for k in imputed_models:
        # e.g. 'gemma-3-27b' (csv) vs 'gemma-3-27b-it' (route counts)
        if k == b or k.startswith(b + "-") or b.startswith(k):
            return k
    return None

DST.mkdir(exist_ok=True)
log = {}
for f in sorted(SRC.glob("*.csv")):
    d = pd.read_csv(f)
    keep = pd.Series(True, index=d.index)
    x = d["answer"].astype(str).str.upper().eq("X")
    keep &= ~x
    for m in d["model"].unique():
        if fb_key(m) is None:
            continue
        sel = d["model"].eq(m) & d["confidence"].eq(50)
        keep &= ~sel
    for m, g in d.groupby("model"):
        e = log.setdefault(m, {"rows": 0, "unparsed_answer": 0,
                               "imputed_conf": 0, "kept": 0})
        gi = g.index
        e["rows"] += len(g)
        e["unparsed_answer"] += int(x[gi].sum())
        if fb_key(m) is not None:
            e["imputed_conf"] += int((g["confidence"].eq(50) & ~x[gi]).sum())
        e["kept"] += int(keep[gi].sum())
    d[keep].to_csv(DST / f.name, index=False)

log = {m: v for m, v in log.items() if v["kept"] != v["rows"]}
tot = {k: sum(v[k] for v in log.values()) for k in
       ["rows", "unparsed_answer", "imputed_conf"]}
json.dump(log, open(ROOT / "data" / "v2_exclusions.json", "w"), indent=1)
print(json.dumps(log, indent=1))
print("removed: unparsed answers", tot["unparsed_answer"],
      "| imputed confidences (not also unparsed)", tot["imputed_conf"])
