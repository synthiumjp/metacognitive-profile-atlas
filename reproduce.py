"""
Metacognitive Profile Atlas — single-command regeneration.

Usage:
    python reproduce.py

Reproduces every numerical claim and figure in the paper from the raw CSVs
in data/raw/. Runtime: ~60 seconds with cached bootstrap CIs (the default),
~5 minutes if bootstrap CIs need to be regenerated from scratch.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_OUT = ROOT / "data"
FIG_DIR = ROOT / "figures"
SCRIPTS = ROOT / "scripts"
FIG_DIR.mkdir(exist_ok=True)

if not DATA_RAW.exists() or not any(DATA_RAW.glob("*.csv")):
    print("ERROR: data/raw/ is empty. See data/README.md.")
    sys.exit(1)

t0 = time.time()
print("=" * 60)
print("Metacognitive Profile Atlas — reproducing paper results")
print("=" * 60)

# Shared namespace; scripts see ROOT, DATA_RAW, DATA_OUT, FIG_DIR.
ns = {'ROOT': ROOT, 'DATA_RAW': DATA_RAW, 'DATA_OUT': DATA_OUT, 'FIG_DIR': FIG_DIR}

steps = [
    ("Loading and deduplicating item-level data", "01_load_data.py"),
    ("Computing 33 x 6 AUROC matrix (Table 3)", "02_compute_matrix.py"),
    ("Computing bootstrap 95% CIs (Supp Table S1)", "03_bootstrap_cis.py"),
    ("Inferential tests (Friedman, Kendall W, family permutation)", "04_inferential.py"),
    ("Validation (split-half aggregate + profile, subject coherence)", "05_validation.py"),
    ("Accuracy confound control", "06_accuracy_check.py"),
    ("Building 7 figures at 300 dpi", "07_figures.py"),
]

for i, (desc, fname) in enumerate(steps, 1):
    print(f"\n[{i}/{len(steps)}] {desc}...")
    exec(compile(open(SCRIPTS / fname).read(), fname, 'exec'), ns)

print(f"\n{'=' * 60}")
print(f"Done in {time.time() - t0:.1f}s")
print(f"{'=' * 60}")
print(f"\nOutputs in: {DATA_OUT} and {FIG_DIR}")
