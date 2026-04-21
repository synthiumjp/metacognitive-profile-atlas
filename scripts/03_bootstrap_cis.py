"""
Bootstrap 95% CIs for all 198 model-domain cells. 1,000 resamples, seed=42.
Writes DATA_OUT/atlas_bootstrap_cis.csv. Skips if already present (pre-cached)
since this step takes ~5 minutes; delete the file to force regeneration.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

OUT = DATA_OUT / "atlas_bootstrap_cis.csv"
DOMAIN_LABELS = {'applied_professional': 'Applied', 'factual_recall': 'Factual',
                 'formal_reasoning': 'Formal', 'humanities': 'Human.',
                 'natural_science': 'Science', 'social_moral': 'Social'}

if OUT.exists():
    bdf = pd.read_csv(OUT)
    med_w = bdf['ci_w'].median()
    frac_over_25 = (bdf['ci_w'] > 0.25).mean()
    print(f"  Using cached {OUT.name} ({len(bdf)} cells, median CI width {med_w:.3f})")
    print(f"  (Delete {OUT.name} to force regeneration; takes ~5 min)")
else:
    rng = np.random.default_rng(42)
    B = 1000
    rows = []
    for m in mat['model']:
        d = data[data['model_short'] == m]
        for dom_raw, dom_short in DOMAIN_LABELS.items():
            dd = d[d['domain'] == dom_raw]
            n = len(dd)
            if n == 0:
                continue
            y = dd['is_correct'].values
            c = dd['confidence'].values
            if len(np.unique(y)) < 2:
                continue
            pt = roc_auc_score(y, c)
            boot = []
            for _ in range(B):
                idx = rng.integers(0, n, n)
                yb, cb = y[idx], c[idx]
                if len(np.unique(yb)) < 2:
                    continue
                try:
                    boot.append(roc_auc_score(yb, cb))
                except ValueError:
                    continue
            if len(boot) < 100:
                lo, hi = np.nan, np.nan
            else:
                lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append({'model': m, 'domain': dom_short, 'n': n,
                         'auroc': round(pt, 4),
                         'ci_lo': round(lo, 4), 'ci_hi': round(hi, 4),
                         'ci_w': round(hi - lo, 4)})
    bdf = pd.DataFrame(rows)
    bdf.to_csv(OUT, index=False)
    med_w = bdf['ci_w'].median()
    frac_under_20 = (bdf['ci_w'] < 0.20).mean()
    frac_over_25 = (bdf['ci_w'] > 0.25).mean()
    print(f"  198 cells, median CI width = {med_w:.3f}")
    print(f"  {frac_under_20 * 100:.0f}% of cells have width < .20; {frac_over_25 * 100:.0f}% exceed .25")
    print(f"  Wrote {OUT.name}")
