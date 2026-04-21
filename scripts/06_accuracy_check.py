"""
Check whether domain AUROC differences track accuracy differences.
Reports the three critical statistics for §3.3.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

DOMAIN_LABELS = {'applied_professional': 'Applied', 'factual_recall': 'Factual',
                 'formal_reasoning': 'Formal', 'humanities': 'Human.',
                 'natural_science': 'Science', 'social_moral': 'Social'}

cells = []
for m in mat['model']:
    for dom_raw, dom in DOMAIN_LABELS.items():
        d = data[(data['model_short'] == m) & (data['domain'] == dom_raw)]
        if len(d) == 0:
            continue
        acc = d['is_correct'].mean()
        if d['is_correct'].nunique() > 1:
            try:
                auc = roc_auc_score(d['is_correct'], d['confidence'])
            except ValueError:
                continue
        else:
            continue
        cells.append({'model': m, 'domain': dom, 'acc': acc, 'auroc': auc})
cells = pd.DataFrame(cells)

r_all, p_all = pearsonr(cells['acc'], cells['auroc'])
print(f"  Per-cell accuracy vs AUROC (all 198 cells): r = {r_all:.3f}, p = {p_all:.2g}")

cells['acc_c'] = cells.groupby('model')['acc'].transform(lambda x: x - x.mean())
cells['auc_c'] = cells.groupby('model')['auroc'].transform(lambda x: x - x.mean())
r_within, p_within = pearsonr(cells['acc_c'], cells['auc_c'])
print(f"  Within-model (centered): r = {r_within:.3f}, p = {p_within:.2g}")

acc_by_dom = cells.groupby('domain')['acc'].mean()
auc_by_dom = cells.groupby('domain')['auroc'].mean()
rho_rank, p_rank = spearmanr(acc_by_dom, auc_by_dom)
print(f"  Domain-rank correlation (Spearman): rho = {rho_rank:.3f}")
print(f"    Highest-accuracy domain: {acc_by_dom.idxmax()} ({acc_by_dom.max():.3f})")
print(f"    Highest-AUROC domain:    {auc_by_dom.idxmax()} ({auc_by_dom.max():.3f})")
