"""
Build all 7 publication figures at 300 dpi (PDF + PNG).
Expects `mat`, `data`, and FIG_DIR in the shared namespace.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

FIG_DIR.mkdir(exist_ok=True)

mpl.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
})

DOMAINS = ['Applied', 'Factual', 'Human.', 'Social', 'Formal', 'Science']
FAMILY_COLORS = {
    'Anthropic': '#1f77b4', 'DeepSeek': '#ff7f0e', 'G-Gemini': '#2ca02c',
    'G-Gemma': '#d62728', 'OpenAI': '#9467bd', 'Qwen': '#8c564b', 'Zhipu': '#e377c2',
}

def save(name):
    plt.savefig(FIG_DIR / f"{name}.pdf", bbox_inches='tight')
    plt.savefig(FIG_DIR / f"{name}.png", bbox_inches='tight')
    plt.close()

# =============== F1: Heatmap ===============
heat = mat[DOMAINS].values
fig, ax = plt.subplots(figsize=(6.8, 10.0))
cmap = LinearSegmentedColormap.from_list(
    'auroc', [(0.0, '#b2182b'), (0.3, '#ef8a62'), (0.5, '#f7f7f7'),
             (0.7, '#67a9cf'), (1.0, '#2166ac')])
im = ax.imshow(heat, aspect='auto', cmap=cmap, vmin=0.4, vmax=0.9)
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        v = heat[i, j]
        color = 'white' if (v < 0.55 or v > 0.80) else 'black'
        ax.text(j, i, f'{v:.2f}'[1:], ha='center', va='center', fontsize=7.5, color=color)
ax.set_xticks(range(len(DOMAINS)))
ax.set_xticklabels(DOMAINS)
ax.set_yticks(range(len(mat)))
ax.set_yticklabels(mat['model'])
# family separators
prev = mat['family'].iloc[0]
for i, fam in enumerate(mat['family']):
    if fam != prev:
        ax.axhline(i - 0.5, color='black', linewidth=1.2, alpha=0.6)
        prev = fam
# family labels on right
start = 0
for i in range(1, len(mat) + 1):
    if i == len(mat) or mat['family'].iloc[i] != mat['family'].iloc[i - 1]:
        mid = (start + i - 1) / 2
        ax.text(6.05, mid, mat['family'].iloc[start], ha='left', va='center',
                fontsize=8, fontweight='bold', color='#444')
        if i < len(mat):
            start = i
ax.set_xlim(-0.5, 8.0)
ax.set_xlabel('Cognitive domain')
ax.set_title('Type-2 AUROC by model and domain', pad=12)
cbar = fig.colorbar(im, ax=ax, pad=0.18, shrink=0.5, aspect=20)
cbar.set_label('AUROC', fontsize=9)
save('fig1_heatmap')
print("  F1 heatmap")

# =============== F2: Domain difficulty bars ===============
means = mat[DOMAINS].mean().sort_values(ascending=False)
sds = mat[DOMAINS].std()
fig, ax = plt.subplots(figsize=(6.5, 3.5))
x = np.arange(len(means))
colors_bar = ['#2166ac', '#4393c3', '#92c5de', '#f4a582', '#d6604d', '#b2182b']
ax.bar(x, means.values, yerr=sds[means.index].values, capsize=4,
       color=colors_bar, edgecolor='black', linewidth=0.7, alpha=0.85,
       error_kw={'linewidth': 0.8, 'ecolor': '#333'})
ax.axhline(0.50, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
ax.text(len(means) - 0.4, 0.51, 'chance', fontsize=8, alpha=0.6, ha='right')
for i, (v, sd) in enumerate(zip(means.values, sds[means.index].values)):
    ax.text(i, v + sd + 0.01, f'{v:.3f}'[1:], ha='center', va='bottom', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(means.index)
ax.set_ylabel('Mean Type-2 AUROC (across 33 models)')
ax.set_ylim(0.45, 1.0)
ax.set_title('Domain difficulty hierarchy for metacognitive monitoring', pad=10)
ax.grid(axis='y', linestyle=':', alpha=0.4)
save('fig2_domain_hierarchy')
print("  F2 domain hierarchy")

# =============== F3: Ipsative profiles ===============
fig, ax = plt.subplots(figsize=(7.5, 4.5))
x = np.arange(len(DOMAINS))
for _, row in mat.iterrows():
    vals = np.array([row[d] for d in DOMAINS])
    ax.plot(x, vals - vals.mean(), color=FAMILY_COLORS[row['family']],
            alpha=0.45, linewidth=0.9)
for fam, color in FAMILY_COLORS.items():
    sub = mat[mat['family'] == fam]
    if len(sub) < 2:
        continue
    vals = sub[DOMAINS].values
    mean_prof = (vals - vals.mean(axis=1, keepdims=True)).mean(axis=0)
    ax.plot(x, mean_prof, color=color, linewidth=2.5,
            label=f"{fam} (n={len(sub)})", marker='o', markersize=5)
ax.axhline(0, color='black', linewidth=0.5, alpha=0.4, linestyle='--')
ax.set_xticks(x)
ax.set_xticklabels(DOMAINS)
ax.set_ylabel('AUROC − model mean (ipsative)')
ax.set_xlabel('Cognitive domain')
ax.set_title('Ipsative domain profiles: shape, not level', pad=10)
ax.grid(axis='y', linestyle=':', alpha=0.3)
ax.legend(loc='upper right', ncol=2, frameon=True, fontsize=7.5)
save('fig3_ipsative')
print("  F3 ipsative")

# =============== F4: Family means with range bars ===============
fam_stats = mat.groupby('family')['aggregate'].agg(['mean', 'min', 'max', 'count']).reset_index()
fam_stats = fam_stats.sort_values('mean', ascending=False)
fig, ax = plt.subplots(figsize=(6.5, 3.8))
y = np.arange(len(fam_stats))
for i, (_, r) in enumerate(fam_stats.iterrows()):
    ax.barh(i, r['mean'], color=FAMILY_COLORS[r['family']], alpha=0.85,
            edgecolor='black', linewidth=0.7, height=0.65)
    if r['count'] > 1:
        ax.plot([r['min'], r['max']], [i, i], color='black', linewidth=1.5, alpha=0.8)
        ax.plot([r['min'], r['min']], [i - 0.12, i + 0.12], color='black', linewidth=1.5)
        ax.plot([r['max'], r['max']], [i - 0.12, i + 0.12], color='black', linewidth=1.5)
ax.axvline(0.50, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.text(0.505, len(fam_stats) - 0.3, 'chance', fontsize=8, alpha=0.6)
for i, (_, r) in enumerate(fam_stats.iterrows()):
    label = f"{r['mean']:.3f} (n={r['count']}, range {r['min']:.3f}-{r['max']:.3f})" \
        if r['count'] > 1 else f"{r['mean']:.3f} (n=1)"
    ax.text(max(r['max'], r['mean']) + 0.005, i, label, va='center', fontsize=8)
ax.set_yticks(y)
ax.set_yticklabels(fam_stats['family'])
ax.set_xlabel('Aggregate Type-2 AUROC')
ax.set_xlim(0.45, 1.0)
ax.set_title('Family-level metacognitive quality', pad=10)
ax.invert_yaxis()
ax.grid(axis='x', linestyle=':', alpha=0.4)
save('fig4_family_means')
print("  F4 family means")

# =============== F5: Generational trajectories ===============
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)

def aurocs(names):
    return [mat.loc[mat['model'] == m, 'aggregate'].iloc[0] for m in names]

# Anthropic
ax = axes[0]
opus = ['Opus 4.1', 'Opus 4.5', 'Opus 4.6', 'Opus 4.7']
sonnet = ['Sonnet 4', 'Sonnet 4.5', 'Sonnet 4.6']
ax.plot(range(4), aurocs(opus), 'o-', color='#1f77b4', linewidth=2, markersize=7, label='Opus')
ax.plot([1, 2, 3], aurocs(sonnet), 's-', color='#4a9fd4', linewidth=2, markersize=7, label='Sonnet')
ax.plot([2], aurocs(['Haiku 4.5']), '^', color='#75b8e3', markersize=10, label='Haiku')
ax.set_xticks(range(4)); ax.set_xticklabels(['4.1', '4.5', '4.6', '4.7'])
ax.set_xlabel('Anthropic generation'); ax.set_ylabel('Aggregate Type-2 AUROC')
ax.set_title('Anthropic: plateau after 4.5')
ax.legend(loc='lower right', fontsize=8); ax.grid(axis='y', linestyle=':', alpha=0.4)
ax.axhline(0.50, color='black', linestyle='--', linewidth=0.5, alpha=0.4)

# Gemma
ax = axes[1]
g3 = ['Gemma 3 1B', 'Gemma 3 4B', 'Gemma 3 12B', 'Gemma 3 27B']
g3a = aurocs(g3); g4a = aurocs(['Gemma 4 31B'])
ax.plot(range(4), g3a, 'o-', color='#d62728', linewidth=2, markersize=7, label='Gemma 3')
ax.plot([3, 4], [g3a[-1], g4a[0]], '--', color='#d62728', alpha=0.5, linewidth=1.5)
ax.plot([4], g4a, '*', color='#ff4444', markersize=16, label='Gemma 4 31B')
ax.annotate(f'+{g4a[0] - g3a[-1]:.2f}', xy=(3.5, (g3a[-1] + g4a[0])/2 + 0.02),
            fontsize=10, ha='center', color='#d62728', fontweight='bold')
ax.set_xticks(range(5)); ax.set_xticklabels(['1B', '4B', '12B', '27B', '31B'])
ax.set_xlabel('Gemma model (size / generation)')
ax.set_title('Gemma: +.20 leap at Gen 4')
ax.legend(loc='upper left', fontsize=8); ax.grid(axis='y', linestyle=':', alpha=0.4)
ax.axhline(0.50, color='black', linestyle='--', linewidth=0.5, alpha=0.4)

# DeepSeek
ax = axes[2]
ds = ['DeepSeek V3.1', 'DeepSeek V3.2', 'DeepSeek-R1']
ax.plot(range(3), aurocs(ds), 'o-', color='#ff7f0e', linewidth=2, markersize=8)
ax.set_xticks(range(3)); ax.set_xticklabels(['V3.1', 'V3.2', 'R1'])
ax.set_xlabel('DeepSeek generation')
ax.set_title('DeepSeek: steady improvement')
ax.grid(axis='y', linestyle=':', alpha=0.4)
ax.axhline(0.50, color='black', linestyle='--', linewidth=0.5, alpha=0.4)

for ax in axes:
    ax.set_ylim(0.45, 0.85)

plt.suptitle('Generational trajectories in metacognitive monitoring', y=1.02, fontsize=12)
plt.tight_layout()
save('fig5_generational')
print("  F5 generational")

# =============== F6: Split-half aggregate scatter ===============
rng = np.random.default_rng(42)
sh_rows = []
for m in mat['model']:
    sub = data[data['model_short'] == m].reset_index(drop=True)
    n = len(sub)
    idx = np.arange(n)
    rng2 = np.random.default_rng(42)
    rng2.shuffle(idx)
    h = n // 2
    h1 = sub.iloc[idx[:h]]; h2 = sub.iloc[idx[h:2*h]]
    try:
        a1 = roc_auc_score(h1['is_correct'], h1['confidence']) if h1['is_correct'].nunique() > 1 else np.nan
        a2 = roc_auc_score(h2['is_correct'], h2['confidence']) if h2['is_correct'].nunique() > 1 else np.nan
    except ValueError:
        a1 = a2 = np.nan
    sh_rows.append({'model': m, 'family': mat.loc[mat['model'] == m, 'family'].iloc[0],
                    'half1': a1, 'half2': a2})
sh = pd.DataFrame(sh_rows).dropna()
r_sh, p_sh = pearsonr(sh['half1'], sh['half2'])

fig, ax = plt.subplots(figsize=(5.5, 5.5))
for fam, color in FAMILY_COLORS.items():
    s = sh[sh['family'] == fam]
    if len(s) == 0: continue
    ax.scatter(s['half1'], s['half2'], color=color, s=55,
               edgecolor='black', linewidth=0.6, label=fam, alpha=0.85)
lo, hi = 0.40, 0.90
ax.plot([lo, hi], [lo, hi], color='black', linewidth=0.8, alpha=0.4, linestyle='--')
ax.axhline(0.50, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
ax.axvline(0.50, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel('Half-1 AUROC'); ax.set_ylabel('Half-2 AUROC')
ax.set_title(f'Split-half aggregate stability (r = {r_sh:.3f}, p = {p_sh:.1e})', pad=10)
ax.legend(loc='upper left', fontsize=7.5, ncol=2, frameon=True)
ax.grid(linestyle=':', alpha=0.3)
# Label the 3 worst
sh['disag'] = (sh['half1'] - sh['half2']).abs()
for _, r in sh.nlargest(3, 'disag').iterrows():
    ax.annotate(r['model'], xy=(r['half1'], r['half2']),
                xytext=(5, -5), textcoords='offset points', fontsize=7.5, color='#444')
save('fig6_splithalf')
print("  F6 split-half")

# =============== F7: Cross-benchmark scatter ===============
# Battery values: five keyed tracks, validity-screen 0.3.0 (Cacioli, arXiv:2604.17714v2).
# Written to data/battery_v2.csv from that paper's released outputs.
bat = pd.read_csv(DATA_OUT / "battery_v2.csv")
cross = bat.merge(mat[['model', 'family', 'aggregate']], on='model')
cross = cross.rename(columns={'aggregate': 'mmlu_auroc'})
assert len(cross) == 20, len(cross)

fig, ax = plt.subplots(figsize=(7.2, 5.6))
TIER_M = {'Valid': 'o', 'Indeterminate': 's', 'Invalid': 'X'}
TIER_C = {'Valid': '#2ca02c', 'Indeterminate': '#ff7f0e', 'Invalid': '#d62728'}
for tier, mark in TIER_M.items():
    sub = cross[cross['battery_tier'] == tier]
    if len(sub) == 0: continue
    ax.scatter(sub['battery_auroc'], sub['mmlu_auroc'], marker=mark, color=TIER_C[tier], s=90,
               edgecolor='black', linewidth=0.8, alpha=0.85, label=f"Battery {tier} (n={len(sub)})")
ax.plot([0.3, 0.85], [0.3, 0.85], color='black', linewidth=0.8, alpha=0.4, linestyle='--')
ax.axhline(0.50, color='gray', linewidth=0.5, linestyle=':', alpha=0.6)
ax.axvline(0.50, color='gray', linewidth=0.5, linestyle=':', alpha=0.6)
ax.set_xlim(0.28, 0.80); ax.set_ylim(0.45, 0.85)
ax.set_xlabel('Battery Type-2 AUROC (binary KEEP/WITHDRAW + BET, five keyed tracks)')
ax.set_ylabel('Atlas MMLU AUROC (verbalized 0-100)')
for _, r in cross.iterrows():
    ax.annotate(r['model'], xy=(r['battery_auroc'], r['mmlu_auroc']), xytext=(5, -3),
                textcoords='offset points', fontsize=6.5, color='#333')
ax.legend(loc='lower left', fontsize=8, frameon=True)
ax.grid(linestyle=':', alpha=0.3)
from scipy.stats import spearmanr
rho, _ = spearmanr(cross['battery_auroc'], cross['mmlu_auroc'])
ax.set_title(f'Binary-probe battery vs verbalized-confidence atlas (n = 20); Spearman rho = {rho:+.2f}', fontsize=10)
save('fig7_cross_benchmark')
print(f"  F7 cross-benchmark (rho = {rho:+.3f})")
