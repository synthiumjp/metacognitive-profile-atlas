"""Write LaTeX Table A1 (33 x 6 matrix) and Table S1 (198 cells with error counts) for atlas v2.
Run from repo root after reproduce.py and scripts/rebuttal/analysis1_precision.py."""
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__import__('sys').argv[1]) if len(__import__('sys').argv) > 1 else ROOT / 'paper'
OUT.mkdir(exist_ok=True)
mat = pd.read_csv(ROOT / 'data' / 'atlas_summary_matrix.csv')
cells = pd.read_csv(ROOT / 'scripts' / 'rebuttal' / 'cells_with_errors.csv')
FAM = ['Anthropic', 'DeepSeek', 'G-Gemini', 'G-Gemma', 'OpenAI', 'Qwen', 'Zhipu']
COLS = ['Applied', 'Factual', 'Human.', 'Social', 'Formal', 'Science']
f3 = lambda x: f"{x:.3f}".replace('0.', '.', 1)
lines = [r"\begin{table}[!htbp]", r"\centering",
         r"\caption{Type-2 AUROC per model-domain cell, all 33 models. $n$ = scored items after excluding unparsed answers and parser-imputed confidences "
         r"(16 models have fewer than 1{,}500; GPT-oss-120B$^\dagger$ retains 518 after 982 exclusions and should not be interpreted at domain level). "
         r"Agg = AUROC over all of a model's items. Models grouped by family, sorted by Agg within family.}",
         r"\label{tab:matrix}", r"\small", r"\begin{tabular}{lcccccccc}", r"\toprule",
         r"\textbf{Model} & \textbf{$n$} & \textbf{Applied} & \textbf{Factual} & \textbf{Human.} & \textbf{Social} & "
         r"\textbf{Formal} & \textbf{Science} & \textbf{Agg} \\", r"\midrule"]
for i, fam in enumerate(FAM):
    sub = mat[mat.family == fam].sort_values('aggregate', ascending=False)
    if i: lines.append(r"\midrule")
    for _, r in sub.iterrows():
        name = r.model + (r"$^\dagger$" if r.model == 'GPT-oss-120B' else '')
        lines.append(" & ".join([name, str(int(r.n))] + [f3(r[c]) for c in COLS] + [f3(r["aggregate"])]) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(OUT / 'table_3.tex').write_text("\n".join(lines) + "\n")

DOM_ORDER = ['Applied', 'Factual', 'Human.', 'Social', 'Formal', 'Science']
cells['dom'] = pd.Categorical(cells.dom, DOM_ORDER, ordered=True)
order = {m: k for k, m in enumerate(pd.concat([mat[mat.family == f].sort_values('aggregate', ascending=False).model for f in FAM]))}
cells['o'] = cells.short.map(order)
cells = cells.sort_values(['o', 'dom'])
med = (cells.ci_hi - cells.ci_lo).median()
s = [r"\begin{longtable}{llrrrrr}",
     rf"\caption{{Bootstrap 95\% confidence intervals and error counts for all 198 model-domain AUROC cells (1{{,}}000 resamples per cell, seeded per cell). Median CI width = {f3(med)}. Archived as \texttt{{data/atlas\_bootstrap\_cis.csv}} and \texttt{{data/atlas\_cell\_error\_counts.csv}}.}} \label{{tab:s1}} \\",
     r"\toprule", r"\textbf{Model} & \textbf{Domain} & \textbf{n} & \textbf{Errors} & \textbf{AUROC} & \textbf{CI lo} & \textbf{CI hi} \\",
     r"\midrule", r"\endfirsthead", r"\multicolumn{7}{c}{\tablename\ \thetable\ -- continued} \\", r"\toprule",
     r"\textbf{Model} & \textbf{Domain} & \textbf{n} & \textbf{Errors} & \textbf{AUROC} & \textbf{CI lo} & \textbf{CI hi} \\",
     r"\midrule", r"\endhead", r"\midrule", r"\multicolumn{7}{r}{{Continued on next page}} \\", r"\endfoot",
     r"\bottomrule", r"\endlastfoot"]
for _, r in cells.iterrows():
    s.append(f"{r.short} & {r.dom} & {int(r.n)} & {int(r.errors)} & {r.auroc:.3f} & {r.ci_lo:.3f} & {r.ci_hi:.3f} \\\\")
s.append(r"\end{longtable}")
(OUT / 'table_s1.tex').write_text("\n".join(s) + "\n")
cells[['short', 'dom', 'n', 'errors', 'acc', 'auroc', 'ci_lo', 'ci_hi', 'ci_w']].to_csv(ROOT / 'data' / 'atlas_cell_error_counts.csv', index=False)
print("wrote", OUT / 'table_3.tex', OUT / 'table_s1.tex', "| cells:", len(cells), "| median CI width", round(med, 3))
