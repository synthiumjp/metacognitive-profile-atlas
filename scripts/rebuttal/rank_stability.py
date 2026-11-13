"""Bootstrap rank stability of the domain hierarchy (yhZy point 4) and
the correct->wrong regression cell for the generational decomposition."""
import glob, re
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
        mapping[m.group(2).strip()] = m.group(1).strip()
data["short"] = data["model"].map(mapping)
DOMS = ["applied_professional", "factual_recall", "formal_reasoning",
        "humanities", "natural_science", "social_moral"]
LBL = ["Applied", "Factual", "Formal", "Human.", "Science", "Social"]

cells = {(s, d): g[["is_correct", "confidence"]].to_numpy()
         for (s, d), g in data.groupby(["short", "domain"])}
models = sorted(data["short"].unique())

N_BOOT = 500
rank_first = np.zeros(6)
rank_bottom2 = np.zeros(6)
gap_applied_lowest = []
for b in range(N_BOOT):
    means = np.zeros(6)
    for j, d in enumerate(DOMS):
        aurocs = []
        for s in models:
            arr = cells[(s, d)]
            idx = rng.integers(0, len(arr), len(arr))
            y, c = arr[idx, 0].astype(bool), arr[idx, 1].astype(float)
            if 0 < y.sum() < len(y):
                aurocs.append(roc_auc_score(y, c))
        means[j] = np.mean(aurocs)
    order = np.argsort(-means)  # descending
    rank_first[order[0]] += 1
    rank_bottom2[order[-1]] += 1
    rank_bottom2[order[-2]] += 1
    gap_applied_lowest.append(means[0] - means.min())

print("Bootstrap rank stability (500 resamples of items within cells):")
for j, l in enumerate(LBL):
    print(f"  {l:8s}  P(rank 1) = {rank_first[j]/N_BOOT:.3f}   "
          f"P(bottom 2) = {rank_bottom2[j]/N_BOOT:.3f}")
g = np.array(gap_applied_lowest)
print(f"  Applied-minus-lowest gap: mean {g.mean():.3f}, 95% CI "
      f"[{np.percentile(g,2.5):.3f}, {np.percentile(g,97.5):.3f}], P(gap>0) = {(g>0).mean():.4f}")

# ---- generational 2x2 with regressions ----
print("\nGenerational 2x2 transitions (shared items):")
PAIRS = [("Gemma 3 27B", "Gemma 4 31B"), ("Opus 4.1", "Opus 4.7"),
         ("DeepSeek V3.2", "DeepSeek-R1")]
for e_s, l_s in PAIRS:
    e = data[data["short"] == e_s].set_index("item_id")
    l = data[data["short"] == l_s].set_index("item_id")
    shared = e.index.intersection(l.index)
    e, l = e.loc[shared], l.loc[shared]
    ww = ((~e.is_correct) & (~l.is_correct)).sum()
    wc = ((~e.is_correct) & (l.is_correct)).sum()
    cw = ((e.is_correct) & (~l.is_correct)).sum()
    cc = ((e.is_correct) & (l.is_correct)).sum()
    # confidence on regressions
    reg = shared[(e.is_correct) & (~l.is_correct)]
    reg_conf = l.loc[reg, "confidence"].mean() if len(reg) else float("nan")
    print(f"  {e_s} -> {l_s}: W->W {ww}, W->C {wc}, C->W {cw}, C->C {cc} "
          f"| later-model conf on regressions: {reg_conf:.1f}")
