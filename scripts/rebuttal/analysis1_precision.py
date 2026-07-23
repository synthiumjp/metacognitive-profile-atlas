"""Analysis 1: precision-weighted domain hierarchy + per-cell error counts."""
import glob, re
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
from sklearn.metrics import roc_auc_score

RAW = sorted(glob.glob("/home/claude/atlas/data/raw/*.csv"))
frames = [pd.read_csv(f) for f in RAW]
data = pd.concat(frames, ignore_index=True)
data = data.drop_duplicates(subset=["model", "item_id", "domain"], keep="first")
data["is_correct"] = data["is_correct"].astype(str).str.lower().eq("true")
data["confidence"] = pd.to_numeric(data["confidence"], errors="coerce")
print(f"Total observations after dedup: {len(data)}")
print(f"Models: {data['model'].nunique()}")

# short-name mapping from data/README.md
mapping = {}
for line in open("/home/claude/atlas/data/README.md"):
    m = re.match(r"\| (.+?) \| (.+?) \| (.+?) \|", line.strip())
    if m and "/" in m.group(2):
        mapping[m.group(2).strip()] = (m.group(1).strip(), m.group(3).strip())
data["short"] = data["model"].map(lambda x: mapping.get(x, (x, "?"))[0])
data["family"] = data["model"].map(lambda x: mapping.get(x, (x, "?"))[1])
unmapped = data[data["family"] == "?"]["model"].unique()
if len(unmapped):
    print("UNMAPPED:", unmapped)

DOM_SHORT = {"applied_professional": "Applied", "factual_recall": "Factual",
             "formal_reasoning": "Formal", "humanities": "Human.",
             "natural_science": "Science", "social_moral": "Social"}
data["dom"] = data["domain"].map(DOM_SHORT)

# per-cell stats: AUROC, n, errors
rows = []
for (short, dom), g in data.groupby(["short", "dom"]):
    n = len(g)
    errs = int((~g["is_correct"]).sum())
    if g["is_correct"].nunique() == 2:
        auroc = roc_auc_score(g["is_correct"], g["confidence"])
    else:
        auroc = np.nan
    rows.append(dict(short=short, dom=dom, n=n, errors=errs, acc=g["is_correct"].mean(), auroc=auroc))
cells = pd.DataFrame(rows)

# merge CI widths
cis = pd.read_csv("/home/claude/atlas/data/atlas_bootstrap_cis.csv")
cells = cells.merge(cis.rename(columns={"model": "short", "domain": "dom"})[["short", "dom", "auroc", "ci_lo", "ci_hi", "ci_w"]],
                    on=["short", "dom"], suffixes=("_recomp", ""), how="left")
print(f"\nCells: {len(cells)}; missing CI merge: {cells['ci_w'].isna().sum()}")
# verify recomputed AUROC matches published
diff = (cells["auroc_recomp"] - cells["auroc"]).abs()
print(f"Max |recomputed - published| AUROC: {diff.max():.4f}")

print("\n=== Per-cell error counts (summary) ===")
print(f"Median errors per cell: {cells['errors'].median():.0f}; min: {cells['errors'].min()}; cells with <10 errors: {(cells['errors']<10).sum()}; <5 errors: {(cells['errors']<5).sum()}")

def hierarchy(sub, label):
    means = sub.groupby("dom")["auroc"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    print(f"\n--- {label} ---")
    print(means.round(4))
    # Friedman on models with all 6 surviving cells
    wide = sub.pivot(index="short", columns="dom", values="auroc").dropna()
    if len(wide) >= 3 and wide.shape[1] == 6:
        stat, p = friedmanchisquare(*[wide[c] for c in wide.columns])
        k, nn = wide.shape[1], len(wide)
        W = stat / (nn * (k - 1))
        print(f"Friedman chi2(5) = {stat:.2f}, p = {p:.2e}, n_models = {nn}, Kendall's W = {W:.3f}")
    else:
        print(f"Friedman not run: {len(wide)} complete models")
    return means

# Baseline (all cells)
hierarchy(cells, "Baseline: all 198 cells")

# Exclusion at ci_w > .25
hierarchy(cells[cells["ci_w"] <= .25], "Excluding ci_w > .25 (n cells = %d)" % (cells["ci_w"] <= .25).sum())

# Exclusion at ci_w > .30
hierarchy(cells[cells["ci_w"] <= .30], "Excluding ci_w > .30 (n cells = %d)" % (cells["ci_w"] <= .30).sum())

# Minimum error count >= 10
hierarchy(cells[cells["errors"] >= 10], "Excluding cells with < 10 errors (n cells = %d)" % (cells["errors"] >= 10).sum())

# Inverse-variance weighted domain means
print("\n--- Inverse-variance weighted domain means (weight = 1/ci_w^2) ---")
c = cells.dropna(subset=["ci_w"]).copy()
c["w"] = 1 / c["ci_w"] ** 2
ivw = c.groupby("dom").apply(lambda g: np.average(g["auroc"], weights=g["w"])).sort_values(ascending=False)
print(ivw.round(4))

# rank stability under exclusion: is Applied top and Formal/Science bottom-2 in every variant?
cells.to_csv("/home/claude/cells_with_errors.csv", index=False)
print("\nSaved cells_with_errors.csv")
