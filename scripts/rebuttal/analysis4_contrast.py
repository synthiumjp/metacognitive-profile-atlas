import glob, re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(42)
RAW = sorted(glob.glob("../../data/raw/*.csv"))
data = pd.concat([pd.read_csv(f) for f in RAW], ignore_index=True)
data = data.drop_duplicates(subset=["model", "item_id", "domain"], keep="first")
data["is_correct"] = data["is_correct"].astype(str).str.lower().eq("true")
data["confidence"] = pd.to_numeric(data["confidence"], errors="coerce")
mapping = {}
for line in open("../../data/README.md"):
    m = re.match(r"\| (.+?) \| (.+?) \| (.+?) \|", line.strip())
    if m and "/" in m.group(2):
        mapping[m.group(2).strip()] = m.group(1).strip()
data["short"] = data["model"].map(mapping)
DOMS = ["applied_professional", "factual_recall", "formal_reasoning",
        "humanities", "natural_science", "social_moral"]
I_APP, I_FOR, I_SCI = 0, 2, 4
models = sorted(data["short"].unique())
N_SPLITS = 50

def auroc_safe(g):
    return roc_auc_score(g["is_correct"], g["confidence"]) if g["is_correct"].nunique() == 2 else np.nan

grouped = {(s, d): g for (s, d), g in data.groupby(["short", "domain"])}

contrast_rs, profile_med_rs, ext_agree = [], [], []
for it in range(N_SPLITS):
    ca, cb, prs = [], [], []
    agree_top, agree_bot, n_ok = 0, 0, 0
    for s in models:
        pa, pb = [], []
        for d in DOMS:
            g = grouped[(s, d)]
            mask = rng.integers(0, 2, len(g)).astype(bool)
            pa.append(auroc_safe(g[mask])); pb.append(auroc_safe(g[~mask]))
        pa, pb = np.array(pa), np.array(pb)
        if np.isnan(pa).any() or np.isnan(pb).any():
            continue
        n_ok += 1
        ca.append(pa[I_APP] - (pa[I_FOR] + pa[I_SCI]) / 2)
        cb.append(pb[I_APP] - (pb[I_FOR] + pb[I_SCI]) / 2)
        za, zb = pa - pa.mean(), pb - pb.mean()
        prs.append(np.corrcoef(za, zb)[0, 1])
    contrast_rs.append(np.corrcoef(ca, cb)[0, 1])
    profile_med_rs.append(np.median(prs))
    # population-level: does mean contrast stay positive in both halves?
    ext_agree.append(int(np.mean(ca) > 0 and np.mean(cb) > 0))

print(f"Models with complete split cells (last iter): {n_ok}/33")
print(f"Full 6-domain profile split-half: median-of-medians r = {np.median(profile_med_rs):.3f}")
print(f"Extremum contrast (Applied - mean(Formal,Science)) cross-model split-half r: "
      f"median {np.median(contrast_rs):.3f} [IQR {np.percentile(contrast_rs,25):.3f}, {np.percentile(contrast_rs,75):.3f}]")
print(f"Population mean contrast positive in both halves: {sum(ext_agree)}/{N_SPLITS} splits")
# also population-level contrast magnitude
print(f"Mean contrast across models (full data): "
      f"{np.mean([np.mean(ca)]):.3f} (half A, last split)")
