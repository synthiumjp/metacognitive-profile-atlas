import glob, re, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(42)
RAW = sorted(glob.glob("/home/claude/atlas/data/raw/*.csv"))
data = pd.concat([pd.read_csv(f) for f in RAW], ignore_index=True)
data = data.drop_duplicates(subset=["model", "item_id", "domain"], keep="first")
data["is_correct"] = data["is_correct"].astype(str).str.lower().eq("true")
data["confidence"] = pd.to_numeric(data["confidence"], errors="coerce")
mapping = {}
for line in open("/home/claude/atlas/data/README.md"):
    m = re.match(r"\| (.+?) \| (.+?) \| (.+?) \|", line.strip())
    if m and "/" in m.group(2):
        mapping[m.group(2).strip()] = (m.group(1).strip(), m.group(3).strip())
data["short"] = data["model"].map(lambda x: mapping[x][0])
data["family"] = data["model"].map(lambda x: mapping[x][1])
DOMS = ["applied_professional", "factual_recall", "formal_reasoning",
        "humanities", "natural_science", "social_moral"]

profs, shorts = [], []
for short, g in data.groupby("short"):
    row = [roc_auc_score(gd["is_correct"], gd["confidence"])
           for d in DOMS for gd in [g[g["domain"] == d]]]
    profs.append(row); shorts.append(short)
P = np.array(profs)
fam = data.groupby("short")["family"].first().loc[shorts].to_numpy(dtype=object)
C = P - P.mean(axis=1, keepdims=True)
Z = C / np.linalg.norm(C, axis=1, keepdims=True)
R = Z @ Z.T  # pairwise pearson r (since centered rows)
np.fill_diagonal(R, np.nan)
n = len(shorts)
iu = np.triu_indices(n, 1)

def wb_diff(labels):
    same = np.equal.outer(labels, labels)
    return np.nanmean(R[iu][same[iu]]) - np.nanmean(R[iu][~same[iu]])

obs_g = wb_diff(fam)
null_g = np.array([wb_diff(rng.permutation(fam)) for _ in range(10000)])
p_g = (np.sum(null_g >= obs_g) + 1) / 10001
print(f"Global: within-between diff = {obs_g:.3f}, p = {p_g:.4f}", flush=True)

fams, counts = np.unique(fam, return_counts=True)
res = []
for f, k in zip(fams, counts):
    if k < 3:
        print(f"{f}: n={k}, skipped (n<3)")
        continue
    idx = np.where(fam == f)[0]
    within = np.nanmean(R[np.ix_(idx, idx)][np.triu_indices(k, 1)])
    other = np.setdiff1d(np.arange(n), idx)
    between = np.nanmean(R[np.ix_(idx, other)])
    obs = within - between
    null = []
    for _ in range(10000):
        pseudo = rng.choice(n, size=k, replace=False)
        oth = np.setdiff1d(np.arange(n), pseudo)
        w = np.nanmean(R[np.ix_(pseudo, pseudo)][np.triu_indices(k, 1)])
        b = np.nanmean(R[np.ix_(pseudo, oth)])
        null.append(w - b)
    p = (np.sum(np.array(null) >= obs) + 1) / 10001
    res.append(dict(family=f, n=k, within_r=within, between_r=between, stat=obs, p=p))

res = pd.DataFrame(res).sort_values("p").reset_index(drop=True)
m = len(res)
holm, running = [], 0
for i, p in enumerate(res["p"]):
    running = max(running, min(1.0, (m - i) * p))
    holm.append(running)
res["p_holm"] = holm
print(res.round(4).to_string(index=False), flush=True)
res.to_csv("/home/claude/family_permutation_results.csv", index=False)
