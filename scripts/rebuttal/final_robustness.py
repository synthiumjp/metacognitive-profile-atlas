"""Four supplementary robustness analyses:
1. Subject-level gradient: does the Applied > Formal/Science ordering hold
   below the domain bins, at individual-subject level?
2. Holm-corrected pairwise domain contrasts (Wilcoxon signed-rank, 15 pairs).
3. Hierarchy excluding GPT-oss-120B (imputed-confidence model).
4. Spearman-Brown correction of split-half reliabilities."""
import glob, re
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
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
        mapping[m.group(2).strip()] = m.group(1).strip()
data["short"] = data["model"].map(mapping)
DOMS = ["applied_professional", "factual_recall", "formal_reasoning",
        "humanities", "natural_science", "social_moral"]
models = sorted(data["short"].unique())

# ---------- 1. Subject-level gradient ----------
# Per-subject AUROC pooled across models (confidence z-scored within model
# so pooling is scale-free), for subjects with enough pooled errors.
data["conf_z"] = data.groupby("short")["confidence"].transform(
    lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else 1))
rows = []
for (subj, dom), g in data.groupby(["subject", "domain"]):
    errs = int((~g["is_correct"]).sum())
    if errs >= 30 and g["is_correct"].nunique() == 2:
        rows.append(dict(subject=subj, domain=dom, n=len(g), errors=errs,
                         auroc=roc_auc_score(g["is_correct"], g["conf_z"])))
subj = pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True)
subj["rank"] = subj.index + 1
n_subj = len(subj)
print(f"=== 1. Subject-level gradient ({n_subj} subjects with >= 30 pooled errors) ===")
top10 = subj.head(10)
bot10 = subj.tail(10)
print("Top 10 subjects by AUROC:")
print(top10[["rank", "subject", "domain", "auroc"]].to_string(index=False))
print("Bottom 10:")
print(bot10[["rank", "subject", "domain", "auroc"]].to_string(index=False))
for d, lbl in [("applied_professional", "Applied"), ("formal_reasoning", "Formal"),
               ("natural_science", "Science")]:
    sd = subj[subj["domain"] == d]
    print(f"{lbl:8s}: n={len(sd):2d} subjects, mean rank {sd['rank'].mean():5.1f}/{n_subj}, "
          f"mean AUROC {sd['auroc'].mean():.3f}, in top half: {(sd['rank'] <= n_subj/2).mean():.0%}")
# Mann-Whitney: applied subjects vs formal+science subjects
from scipy.stats import mannwhitneyu
app = subj[subj["domain"] == "applied_professional"]["auroc"]
fs = subj[subj["domain"].isin(["formal_reasoning", "natural_science"])]["auroc"]
u, p = mannwhitneyu(app, fs, alternative="greater")
print(f"Mann-Whitney (Applied subjects > Formal/Science subjects): U = {u:.0f}, p = {p:.4g}")

# ---------- 2. Holm-corrected pairwise domain contrasts ----------
print("\n=== 2. Pairwise domain contrasts (Wilcoxon signed-rank over 33 models, Holm) ===")
prof = {}
for s in models:
    g = data[data["short"] == s]
    prof[s] = [roc_auc_score(gd["is_correct"], gd["confidence"])
               for d in DOMS for gd in [g[g["domain"] == d]]]
P = pd.DataFrame(prof, index=["Applied", "Factual", "Formal", "Human.", "Science", "Social"]).T
res = []
for a, b in combinations(P.columns, 2):
    stat, p = wilcoxon(P[a], P[b])
    res.append(dict(pair=f"{a} vs {b}", diff=(P[a] - P[b]).mean(), p=p))
res = pd.DataFrame(res).sort_values("p").reset_index(drop=True)
m = len(res)
holm, running = [], 0
for i, p in enumerate(res["p"]):
    running = max(running, min(1.0, (m - i) * p))
    holm.append(running)
res["p_holm"] = holm
res["sig"] = res["p_holm"] < .05
print(res.round(4).to_string(index=False))

# ---------- 3. Hierarchy excluding GPT-oss-120B ----------
print("\n=== 3. Domain means excluding GPT-oss-120B ===")
P32 = P.drop(index="GPT-oss-120B")
means = P32.mean().sort_values(ascending=False)
print(means.round(4).to_string())
from scipy.stats import friedmanchisquare
stat, p = friedmanchisquare(*[P32[c] for c in P32.columns])
print(f"Friedman chi2(5) = {stat:.2f}, p = {p:.2e} (n = {len(P32)})")

# ---------- 4. Spearman-Brown ----------
print("\n=== 4. Spearman-Brown full-length reliabilities ===")
for label, r in [("Extremum contrast (split-half r = .342)", .342),
                 ("Full 6-domain profile (split-half r = .167)", .167)]:
    sb = 2 * r / (1 + r)
    print(f"{label}: full-length reliability = {sb:.3f}")
