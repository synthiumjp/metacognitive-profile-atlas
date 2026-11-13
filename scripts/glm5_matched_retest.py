"""Matched-item test-retest for GLM-5.

The April run covered 598 of the 1,500-item pool; the July run covered all
1,500. Comparing the two profiles as-is confounds occasion with item sampling.
This restricts BOTH occasions to the 598 items common to them, so the only
thing that differs is the occasion.

RUN FROM THE REPO ROOT:
    python scripts/glm5_matched_retest.py
"""
import glob

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

# April (released corpus)
april = None
for f in glob.glob("data/raw/*.csv"):
    d = pd.read_csv(f)
    if "glm-5" in str(d["model"].iloc[0]).lower():
        april = d
        break
assert april is not None, "April GLM-5 CSV not found in data/raw/"

# July (fresh run)
jf = glob.glob("data/glm5_rerun/metacognitive_profile_results.csv")
assert jf, "July GLM-5 rerun CSV not found at data/glm5_rerun/"
july = pd.read_csv(jf[0])

for d in (april, july):
    d["ok"] = d["is_correct"].astype(str).str.lower().eq("true")

shared = sorted(set(april["item_id"]) & set(july["item_id"]))
print(f"April items: {len(april)}  July items: {len(july)}  Shared: {len(shared)}")

a = april[april["item_id"].isin(shared)]
j = july[july["item_id"].isin(shared)]

def profile(df, label):
    out = {}
    print(f"\n{label} (matched items)")
    print(f"  {'domain':22s} {'n':>4s} {'err':>4s} {'AUROC':>7s}")
    for dom, g in df.groupby("domain"):
        e = int((~g["ok"]).sum())
        auc = roc_auc_score(g["ok"], g["confidence"]) if g["ok"].nunique() == 2 else np.nan
        out[dom] = auc
        print(f"  {dom:22s} {len(g):4d} {e:4d} {auc:7.3f}")
    agg = roc_auc_score(df["ok"], df["confidence"])
    print(f"  aggregate AUROC: {agg:.3f}   accuracy: {df['ok'].mean():.3f}")
    return out, agg

pa, agg_a = profile(a, "APRIL")
pj, agg_j = profile(j, "JULY")

doms = [d for d in pa if d in pj and not (np.isnan(pa[d]) or np.isnan(pj[d]))]
va, vj = [pa[d] for d in doms], [pj[d] for d in doms]
rho, p = spearmanr(va, vj)
r = np.corrcoef(va, vj)[0, 1]

print("\n=== MATCHED-ITEM TEST-RETEST ===")
print(f"Domains compared: {len(doms)}")
print(f"Spearman rho = {rho:+.3f} (p = {p:.3f});  Pearson r = {r:+.3f}")
print(f"Aggregate AUROC: April {agg_a:.3f} -> July {agg_j:.3f}")
print("\nPer-domain change:")
for d in doms:
    print(f"  {d:22s} {pa[d]:.3f} -> {pj[d]:.3f}  ({pj[d]-pa[d]:+.3f})")

# how much do item-level answers agree across occasions?
m = a[["item_id", "ok", "confidence"]].merge(
    j[["item_id", "ok", "confidence"]], on="item_id", suffixes=("_apr", "_jul"))
print(f"\nItem-level agreement on correctness: {(m['ok_apr'] == m['ok_jul']).mean():.1%}")
print(f"Confidence correlation across occasions: r = {np.corrcoef(m['confidence_apr'], m['confidence_jul'])[0,1]:+.3f}")
print("\nIf rho is still near zero here, the non-replication cannot be")
print("attributed to item sampling: same items, same model, different occasion.")
