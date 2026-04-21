"""
Compute the 33 x 6 AUROC matrix. Requires `data` from 01_load_data.py.
Writes DATA_OUT/atlas_summary_matrix.csv.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUT = DATA_OUT / "atlas_summary_matrix.csv"

DOMAIN_ORDER = ['applied_professional', 'factual_recall', 'humanities',
                'social_moral', 'formal_reasoning', 'natural_science']
DOMAIN_LABELS = {'applied_professional': 'Applied', 'factual_recall': 'Factual',
                 'formal_reasoning': 'Formal', 'humanities': 'Human.',
                 'natural_science': 'Science', 'social_moral': 'Social'}

MODEL_ORDER = [
    'Opus 4.6', 'Opus 4.5', 'Sonnet 4.5', 'Opus 4.7', 'Sonnet 4.6', 'Sonnet 4',
    'Haiku 4.5', 'Opus 4.1',
    'DeepSeek-R1', 'DeepSeek V3.2', 'DeepSeek V3.1',
    'Gemini 3.1 Pro', 'Gemini 3 Flash', 'Gemini 2.5 Flash', 'Gemini 2.5 Pro',
    'Gemini 3.1 FLite', 'Gemini 2.0 FLite', 'Gemini 2.0 Flash',
    'Gemma 4 31B', 'Gemma 3 27B', 'Gemma 3 4B', 'Gemma 3 12B', 'Gemma 3 1B',
    'GPT-oss-20B', 'GPT-5.4 mini', 'GPT-5.4', 'GPT-5.4 nano', 'GPT-oss-120B',
    'Qwen Think', 'Qwen Coder', 'Qwen 80B Inst', 'Qwen 235B',
    'GLM-5',
]

rows = []
for m in MODEL_ORDER:
    d = data[data['model_short'] == m]
    row = {'model': m, 'family': d['family'].iloc[0], 'n': len(d),
           'acc': d['is_correct'].mean()}
    row['aggregate'] = roc_auc_score(d['is_correct'], d['confidence']) if d['is_correct'].nunique() > 1 else np.nan
    for dom in DOMAIN_ORDER:
        dd = d[d['domain'] == dom]
        if len(dd) > 0 and dd['is_correct'].nunique() > 1:
            row[DOMAIN_LABELS[dom]] = roc_auc_score(dd['is_correct'], dd['confidence'])
        else:
            row[DOMAIN_LABELS[dom]] = np.nan
    rows.append(row)

mat = pd.DataFrame(rows)
DOMAINS = list(DOMAIN_LABELS.values())
mat['mean_dom'] = mat[DOMAINS].mean(axis=1)
mat['sd_dom'] = mat[DOMAINS].std(axis=1)
mat.to_csv(OUT, index=False)
print(f"  Wrote {OUT.name}  (33 models, domain AUROCs, aggregate, mean/SD)")
