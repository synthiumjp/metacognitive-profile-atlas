"""Diagnostic: how often does the extraction rule capture a confidence value
from INSIDE a reasoning block rather than from the final answer block?

The notebook rule is `re.search(r'[Cc]onfidence:\\s*(\\d+)', text)` over the full
response, first match wins. For models that emit <think> blocks and mention a
confidence value mid-reasoning, that first match can precede the final stated
confidence. This quantifies the incidence and the size of the discrepancy.

RUN FROM THE REPO ROOT:
    python scripts/thinking_confidence_check.py
"""
import json, glob, re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RE_CONF = re.compile(r"[Cc]onfidence:\s*(\d+)")
RE_ID = re.compile(r"mmlu_(\d+)")
THINK = re.compile(r"<think>(.*?)</think>", re.S)

out = []
for path in sorted(glob.glob("raw_outputs/**/*.run.json", recursive=True)):
    model = path.replace("\\", "/").split("/")[-3]
    n = n_think = n_conf_in_think = n_differ = 0
    diffs, recs = [], []
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        continue
    for conv in data.get("conversations", []):
        m = RE_ID.match(str(conv.get("id", "")))
        if not m:
            continue
        try:
            parts = conv["requests"][0]["contents"][-1]["parts"]
            text = "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError):
            continue
        n += 1
        t = THINK.search(text)
        if not t:
            continue
        n_think += 1
        first = RE_CONF.search(text)              # notebook rule
        after = RE_CONF.search(text[t.end():])    # final-answer-block value
        if first and t.start() <= first.start() < t.end():
            n_conf_in_think += 1
        if first and after and first.group(1) != after.group(1):
            n_differ += 1
            diffs.append(int(after.group(1)) - int(first.group(1)))
            recs.append((int(m.group(1)), int(first.group(1)), int(after.group(1))))
    if n_think:
        out.append(dict(model=model, n=n, n_think=n_think,
                        conf_inside_think=n_conf_in_think,
                        differ=n_differ,
                        differ_rate=n_differ / n if n else 0,
                        mean_shift=np.mean(diffs) if diffs else 0.0))
        if recs:
            pd.DataFrame(recs, columns=["item_id", "recorded", "final"]).to_csv(
                f"data/thinking_conf_{model}.csv", index=False)

df = pd.DataFrame(out).sort_values("differ_rate", ascending=False)
if df.empty:
    print("No <think> blocks found in any run.")
else:
    print("Models emitting reasoning blocks:")
    print(df.round(4).to_string(index=False))
    print("\nInterpretation: 'differ' counts responses where the recorded "
          "confidence (first match) differs from the value stated in the final "
          "answer block. mean_shift is (final - recorded).")

    # For affected models, does using the final value change AUROC?
    print("\n=== Does using the final-block value change the model's AUROC? ===")
    for _, r in df[df.differ > 0].iterrows():
        f = f"data/thinking_conf_{r['model']}.csv"
        try:
            fix = pd.read_csv(f)
        except FileNotFoundError:
            continue
        csvs = [x for x in glob.glob("data/raw/*.csv")]
        target = None
        for c in csvs:
            d = pd.read_csv(c)
            slug = str(d["model"].iloc[0]).split("/")[-1].replace("@", "-").lower()
            if r["model"].lower().replace("-", "") in slug.replace("-", "").replace(".", ""):
                target = d
                break
        if target is None:
            print(f"  {r['model']}: released CSV not matched, skipped")
            continue
        target["is_correct"] = target["is_correct"].astype(str).str.lower().eq("true")
        base = roc_auc_score(target["is_correct"], target["confidence"])
        merged = target.merge(fix, on="item_id", how="left")
        merged["conf_final"] = merged["final"].fillna(merged["confidence"])
        alt = roc_auc_score(merged["is_correct"], merged["conf_final"])
        print(f"  {r['model']}: AUROC as released {base:.3f} -> "
              f"using final-block values {alt:.3f} (delta {alt - base:+.3f}, "
              f"n changed = {int(r['differ'])})")
