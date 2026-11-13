"""Analysis 2: decompose generational AUROC gains on paired items.
For each (earlier, later) pair: subset items earlier got wrong with confidence >= 70.
Report: (a) fraction later answers correctly (error removal) and its confidence there;
(b) on items both get wrong, does later assign lower confidence (monitoring gain);
(c) per-item confidence correlation on shared items."""
import glob, re
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

RAW = sorted(glob.glob("../../data/raw/*.csv"))
data = pd.concat([pd.read_csv(f) for f in RAW], ignore_index=True)
data = data.drop_duplicates(subset=["model", "item_id", "domain"], keep="first")
data["is_correct"] = data["is_correct"].astype(str).str.lower().eq("true")
data["confidence"] = pd.to_numeric(data["confidence"], errors="coerce")

mapping = {}
for line in open("../../data/README.md"):
    m = re.match(r"\| (.+?) \| (.+?) \| (.+?) \|", line.strip())
    if m and "/" in m.group(2):
        mapping[m.group(1).strip()] = m.group(2).strip()

PAIRS = [
    ("Gemma 3 27B", "Gemma 4 31B"),
    ("Opus 4.1", "Opus 4.5"),
    ("Opus 4.5", "Opus 4.7"),
    ("Opus 4.1", "Opus 4.7"),
    ("DeepSeek V3.1", "DeepSeek V3.2"),
    ("DeepSeek V3.2", "DeepSeek-R1"),
]
THRESH = 70

for early_s, late_s in PAIRS:
    e = data[data["model"] == mapping[early_s]].set_index("item_id")
    l = data[data["model"] == mapping[late_s]].set_index("item_id")
    shared = e.index.intersection(l.index)
    e, l = e.loc[shared], l.loc[shared]
    print(f"\n=== {early_s} -> {late_s} (shared items: {len(shared)}) ===")
    # (c) confidence correlation on all shared items
    rho, p = spearmanr(e["confidence"], l["confidence"])
    print(f"Per-item confidence Spearman rho = {rho:.3f} (p = {p:.1e})")
    # target subset: earlier wrong with high confidence
    tgt = shared[(~e["is_correct"]) & (e["confidence"] >= THRESH)]
    print(f"Items {early_s} wrong with conf >= {THRESH}: {len(tgt)}")
    if len(tgt) == 0:
        continue
    lt = l.loc[tgt]
    et = e.loc[tgt]
    n_fixed = int(lt["is_correct"].sum())
    print(f"(a) Error removal: {late_s} answers {n_fixed}/{len(tgt)} correctly ({n_fixed/len(tgt):.1%})")
    fixed = lt[lt["is_correct"]]
    if len(fixed):
        d = fixed["confidence"] - et.loc[fixed.index, "confidence"]
        print(f"    On fixed items: mean conf change {d.mean():+.1f} (early {et.loc[fixed.index,'confidence'].mean():.1f} -> late {fixed['confidence'].mean():.1f})")
    # (b) both wrong: monitoring improvement?
    still_wrong = lt[~lt["is_correct"]]
    if len(still_wrong) >= 5:
        d = still_wrong["confidence"] - et.loc[still_wrong.index, "confidence"]
        try:
            stat, wp = wilcoxon(d)
        except ValueError:
            wp = np.nan
        print(f"(b) Monitoring: on {len(still_wrong)} still-wrong items, mean conf change {d.mean():+.1f} "
              f"(early {et.loc[still_wrong.index,'confidence'].mean():.1f} -> late {still_wrong['confidence'].mean():.1f}), Wilcoxon p = {wp:.3g}")
        print(f"    Items with lower conf: {(d<0).sum()}, unchanged: {(d==0).sum()}, higher: {(d>0).sum()}")
    else:
        print(f"(b) Only {len(still_wrong)} still-wrong items; too few for test")
    # (c2) items both get right: confidence comparison
    both_right = shared[(e["is_correct"]) & (l["is_correct"])]
    d = l.loc[both_right, "confidence"] - e.loc[both_right, "confidence"]
    print(f"(c) Both-correct items (n={len(both_right)}): mean conf change {d.mean():+.1f}")
