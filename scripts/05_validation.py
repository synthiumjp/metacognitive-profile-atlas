"""
Validation analyses (§3.8): aggregate split-half, profile-level split-half,
subject-level coherence ratio.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ---- Aggregate split-half across models ----
rng = np.random.default_rng(42)
sh_rows = []
for m in mat['model']:
    sub = data[data['model_short'] == m].reset_index(drop=True)
    n = len(sub)
    idx = np.arange(n)
    rng2 = np.random.default_rng(42)
    rng2.shuffle(idx)
    h = n // 2
    h1 = sub.iloc[idx[:h]]
    h2 = sub.iloc[idx[h:2*h]]
    try:
        a1 = roc_auc_score(h1['is_correct'], h1['confidence']) if h1['is_correct'].nunique() > 1 else np.nan
        a2 = roc_auc_score(h2['is_correct'], h2['confidence']) if h2['is_correct'].nunique() > 1 else np.nan
    except ValueError:
        a1 = a2 = np.nan
    sh_rows.append({'model': m, 'half1': a1, 'half2': a2})
sh = pd.DataFrame(sh_rows).dropna()
r_sh, p_sh = pearsonr(sh['half1'], sh['half2'])
print(f"  Aggregate split-half: r = {r_sh:.3f} across 33 models (p = {p_sh:.1e})")

# ---- Profile-level split-half (within-model, 100 splits for stability) ----
DOMAIN_LABELS = {'applied_professional': 'Applied', 'factual_recall': 'Factual',
                 'formal_reasoning': 'Formal', 'humanities': 'Human.',
                 'natural_science': 'Science', 'social_moral': 'Social'}

N_SPLITS = 100
per_model_med = []
for m in mat['model']:
    d = data[data['model_short'] == m]
    rs = []
    for split_i in range(N_SPLITS):
        v1, v2 = [], []
        for dom_raw, dom_short in DOMAIN_LABELS.items():
            dd = d[d['domain'] == dom_raw].reset_index(drop=True)
            nd = len(dd)
            if nd < 20:
                v1.append(np.nan); v2.append(np.nan); continue
            idx = np.arange(nd)
            np.random.default_rng(42 + split_i).shuffle(idx)
            h = nd // 2
            h1d = dd.iloc[idx[:h]]; h2d = dd.iloc[idx[h:2*h]]
            try:
                a1 = roc_auc_score(h1d['is_correct'], h1d['confidence']) if h1d['is_correct'].nunique() > 1 else np.nan
                a2 = roc_auc_score(h2d['is_correct'], h2d['confidence']) if h2d['is_correct'].nunique() > 1 else np.nan
            except ValueError:
                a1 = a2 = np.nan
            v1.append(a1); v2.append(a2)
        v1 = np.array(v1); v2 = np.array(v2)
        ok = ~(np.isnan(v1) | np.isnan(v2))
        if ok.sum() < 4 or np.std(v1[ok]) == 0 or np.std(v2[ok]) == 0:
            continue
        r, _ = pearsonr(v1[ok], v2[ok])
        rs.append(r)
    if rs:
        per_model_med.append(np.median(rs))

grand_med = np.median(per_model_med)
pct_pos = sum(1 for r in per_model_med if r > 0) / len(per_model_med) * 100
print(f"  Profile-level split-half: grand median r = {grand_med:.3f} (100 splits x {len(per_model_med)} models)")
print(f"  Positive median-r: {pct_pos:.0f}% of models")

# ---- Subject-level coherence ratio ----
# For each subject, compute the 33-vector of per-model AUROC on that subject.
# Then pairs of subjects: within-domain mean r vs between-domain mean r.
DOMAIN_MAP = {v: k for k, v in DOMAIN_LABELS.items()}
subject_aurocs = {}
for subj in data['subject'].unique():
    d = data[data['subject'] == subj]
    vec = {}
    for m in mat['model']:
        dd = d[d['model_short'] == m]
        if len(dd) > 5 and dd['is_correct'].nunique() > 1:
            try:
                vec[m] = roc_auc_score(dd['is_correct'], dd['confidence'])
            except ValueError:
                pass
    if len(vec) >= 20:
        subject_aurocs[subj] = vec
        # record which domain this subject belongs to
# Subject -> domain mapping (use first row)
subj_to_dom = data.drop_duplicates(subset='subject').set_index('subject')['domain'].to_dict()

# Pairwise similarity
subjects = list(subject_aurocs.keys())
within, between = [], []
for a, b in combinations(subjects, 2):
    keys = sorted(set(subject_aurocs[a].keys()) & set(subject_aurocs[b].keys()))
    if len(keys) < 10:
        continue
    va = np.array([subject_aurocs[a][k] for k in keys])
    vb = np.array([subject_aurocs[b][k] for k in keys])
    if np.std(va) == 0 or np.std(vb) == 0:
        continue
    r, _ = pearsonr(va, vb)
    if subj_to_dom.get(a) == subj_to_dom.get(b):
        within.append(r)
    else:
        between.append(r)

ratio = np.mean(within) / np.mean(between) if between else np.nan
print(f"  Subject coherence: within-domain mean r = {np.mean(within):.3f}, between = {np.mean(between):.3f}")
print(f"  Ratio = {ratio:.2f}  (1.0 = no domain cohesion)")
