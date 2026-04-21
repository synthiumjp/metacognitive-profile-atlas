"""
Inferential tests for §3.3 (Friedman, Kendall's W) and §3.4 (family permutation).
Prints results; no file output.
"""
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, rankdata
import warnings
warnings.filterwarnings('ignore')

DOMAINS = ['Applied', 'Factual', 'Human.', 'Social', 'Formal', 'Science']

# ---- Friedman + Kendall's W over domains ----
data_mat = mat[DOMAINS].values
chi2, p = stats.friedmanchisquare(*[data_mat[:, i] for i in range(6)])
n = data_mat.shape[0]
k = data_mat.shape[1]
W = chi2 / (n * (k - 1))
print(f"  Friedman chi2(5) = {chi2:.3f}, p = {p:.4g}")
print(f"  Kendall's W = {W:.3f}")

# Within-model rank extremes
rank_mat = np.zeros_like(data_mat)
for i in range(n):
    rank_mat[i] = rankdata(-data_mat[i])
applied_top2 = (rank_mat[:, 0] <= 2).sum()
formal_sci_bot2 = ((rank_mat[:, 4] >= 5) | (rank_mat[:, 5] >= 5)).sum()
print(f"  Applied ranked top-2 in {applied_top2}/33 models")
print(f"  Formal or Science ranked bottom-2 in {formal_sci_bot2}/33 models")

# ---- Family permutation test on ipsative profiles ----
prof = mat[DOMAINS].values.astype(float)
prof_ipsative = prof - prof.mean(axis=1, keepdims=True)
families = mat['family'].values

R = np.full((n, n), np.nan)
for i in range(n):
    for j in range(i + 1, n):
        r, _ = pearsonr(prof_ipsative[i], prof_ipsative[j])
        R[i, j] = R[j, i] = r
iu = np.triu_indices(n, k=1)
r_flat = R[iu]

def wb(labels):
    same = labels[iu[0]] == labels[iu[1]]
    return r_flat[same].mean(), r_flat[~same].mean()

obs_w, obs_b = wb(families)
obs_diff = obs_w - obs_b

np.random.seed(42)
N_PERM = 10000
null_diffs = np.array([
    (lambda s: s[0] - s[1])(wb(np.random.permutation(families)))
    for _ in range(N_PERM)
])
p_perm = (null_diffs >= obs_diff).mean()
print(f"  Family permutation: within r = {obs_w:.3f}, between r = {obs_b:.3f}")
print(f"  Difference = {obs_diff:.3f}, permutation p = {p_perm:.5f}")
