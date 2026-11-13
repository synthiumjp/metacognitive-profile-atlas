"""Collect every number quoted in the atlas v2 text. Run from repo root after reproduce.py and the
rebuttal scripts:  python scripts/v2_numbers.py  -> data/v2_numbers.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import rankdata
ROOT = Path(__file__).resolve().parents[1]
ns = {'DATA_RAW': ROOT / 'data' / 'raw'}
exec(open(ROOT / 'scripts' / '01_load_data.py').read(), ns); data = ns['data']
mat = pd.read_csv(ROOT / 'data' / 'atlas_summary_matrix.csv')
cis = pd.read_csv(ROOT / 'data' / 'atlas_bootstrap_cis.csv')
D = ['Applied', 'Factual', 'Social', 'Human.', 'Formal', 'Science']
X = {}
X['n_obs'] = len(data); X['n_models'] = int(data.model_short.nunique())
X['items_range'] = [int(mat.n.min()), int(mat.n.max())]
X['partial_runs'] = int((mat.n < 1500).sum())
X['acc_range'] = [[r.model, round(r.acc, 3)] for r in mat.sort_values('acc').iloc[[0, -1]].itertuples()]
sd = data.groupby('model_short').confidence.std()
X['conf_sd_range'] = [[sd.idxmin(), round(sd.min(), 1)], [sd.idxmax(), round(sd.max(), 1)]]
X['agg_range'] = [[r.model, round(r.aggregate, 3)] for r in mat.sort_values('aggregate').iloc[[0, -1]].itertuples()]
w = cis.ci_w if 'ci_w' in cis else cis.ci_hi - cis.ci_lo
X['ci_median_width'] = round(w.median(), 3); X['ci_lt20'] = round((w < .20).mean(), 3); X['ci_gt25'] = round((w > .25).mean(), 3)
# Table 2
dm = mat[D].values
ranks = np.array([rankdata(-r) for r in dm])
t2 = []
for j, d in enumerate(D):
    t2.append(dict(domain=d, mean=round(dm[:, j].mean(), 3), sd=round(dm[:, j].std(ddof=1), 3),
                   mean_rank=round(ranks[:, j].mean(), 2), first=int((ranks[:, j] == 1).sum()),
                   last=int((ranks[:, j] == 6).sum())))
X['table2'] = sorted(t2, key=lambda r: -r['mean'])
chi2, p = stats.friedmanchisquare(*[dm[:, i] for i in range(6)])
X['friedman'] = [round(chi2, 2), float(f'{p:.2g}')]; X['kendall_W'] = round(chi2 / (33 * 5), 3)
X['applied_top2'] = int((ranks[:, 0] <= 2).sum()); X['formal_or_sci_bottom2'] = int(((ranks[:, 4] >= 5) | (ranks[:, 5] >= 5)).sum())
X['social_minus_human'] = round(dm[:, 2].mean() - dm[:, 3].mean(), 3)
acc_dom = data.groupby(data.domain).is_correct.mean().round(3).to_dict(); X['acc_by_domain'] = acc_dom
# families
fam = mat.groupby('family')['aggregate'].agg(['count', 'mean', 'min', 'max']).round(3)
X['families'] = fam.reset_index().values.tolist()
# selected cells with CIs
def cell(m, d):
    r = cis[(cis.model == m) & (cis.domain == d)].iloc[0]
    return [round(r.auroc, 3), round(r.ci_lo, 3), round(r.ci_hi, 3), int(r.n)]
X['cells'] = {f'{m}|{d}': cell(m, d) for m, d in [
    ('GPT-oss-120B', 'Applied'), ('GPT-oss-120B', 'Factual'), ('GPT-oss-120B', 'Formal'), ('GPT-oss-120B', 'Human.'),
    ('GPT-oss-120B', 'Science'), ('GPT-oss-120B', 'Social'), ('Gemini 2.5 Pro', 'Applied'), ('Gemini 2.5 Pro', 'Science'),
    ('Gemini 2.5 Pro', 'Formal'), ('GLM-5', 'Human.'), ('GLM-5', 'Science'), ('GPT-5.4', 'Formal'),
    ('DeepSeek-R1', 'Applied'), ('DeepSeek-R1', 'Social'), ('Gemma 4 31B', 'Applied'), ('Gemma 4 31B', 'Formal'),
    ('Gemma 4 31B', 'Social'), ('Sonnet 4.6', 'Social')]}
X['glm5_domains'] = mat.set_index('model').loc['GLM-5', D].round(3).to_dict()
X['agg'] = mat.set_index('model')['aggregate'].round(3).to_dict()
X['dom_mean'] = mat.set_index('model').mean_dom.round(3).to_dict()
X['dom_range'] = {m: [round(r[D].min(), 3), round(r[D].max(), 3)] for m, r in mat.set_index('model').iterrows()}
X['acc'] = mat.set_index('model').acc.round(3).to_dict()
X['gemma1b_ci_max_width'] = round(cis[cis.model == 'Gemma 3 1B'].pipe(lambda c: (c.ci_hi - c.ci_lo).max()), 3)
# GPT-oss-120B: imputation by domain
raw = pd.read_csv(sorted((ROOT / 'data' / 'raw_v1').glob('*.csv'))[0])  # placeholder to keep imports
g = pd.concat([pd.read_csv(f) for f in (ROOT / 'data' / 'raw_v1').glob('*.csv')])
g = g[g.model == 'openai/gpt-oss-120b']
g['imputed'] = g.confidence == 50
tab = pd.crosstab(g.domain, g.imputed)
X['gptoss_imputed_by_domain'] = (g.groupby('domain').imputed.mean().round(3)).to_dict()
X['gptoss_imputed_chi2'] = [round(stats.chi2_contingency(tab)[0], 2), float(f'{stats.chi2_contingency(tab)[1]:.2g}')]
X['gptoss_acc_imputed_vs_kept'] = [round(g[g.imputed].is_correct.astype(str).str.lower().eq('true').mean(), 3),
                                    # retained = the cleaned rows actually analysed (excludes unparsed answers)
                                    round(float(data[data.model_short == 'GPT-oss-120B'].is_correct.astype(float).mean()), 3)]
X['gptoss_kept_by_domain'] = data[data.model_short == 'GPT-oss-120B'].groupby('domain').size().to_dict()
# error counts per cell
ce = pd.read_csv(ROOT / 'scripts' / 'rebuttal' / 'cells_with_errors.csv') if (ROOT / 'scripts' / 'rebuttal' / 'cells_with_errors.csv').exists() else None
if ce is not None:
    e = ce['errors'] if 'errors' in ce else ce.iloc[:, -1]
    X['cell_errors'] = dict(median=float(e.median()), min=int(e.min()), lt10=int((e < 10).sum()), lt5=int((e < 5).sum()))
json.dump(X, open(ROOT / 'data' / 'v2_numbers.json', 'w'), indent=1, default=float)
print(json.dumps(X, indent=1, default=float))
