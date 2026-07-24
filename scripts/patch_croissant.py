"""
Patch croissant.json to match the revised paper.

Fixes:
  1. citeAs — add arXiv ID 2605.06673
  2. rai:dataCollection — the excluded subject is college_medicine, not
     elementary_mathematics (which is mapped to formal_reasoning and included)
  3. rai:dataLimitations — add profile-reliability, imputation, and
     discrimination-vs-calibration caveats
  4. distribution — register the raw_outputs fileset
  5. version bump

RUN FROM THE REPO ROOT:
    python scripts/patch_croissant.py
"""
import json
from pathlib import Path

P = Path("croissant.json")
c = json.loads(P.read_text(encoding="utf-8"))

# 1. Citation with arXiv ID
c["citeAs"] = ("Cacioli, J. P. (2026). Domain-level metacognitive monitoring in "
               "frontier LLMs: A 33-model atlas. arXiv:2605.06673.")

# 2. Correct the exclusion claim
c["rai:dataCollection"] = (
    "Items are 1,500 MMLU (Hendrycks et al., 2021) test-split questions drawn "
    "deterministically (seed=42) and stratified 250 per cognitive domain. 56 of the "
    "57 MMLU test-split subjects were mapped a priori to six domains; "
    "college_medicine was left unmapped as ambiguous between natural science and "
    "applied/professional knowledge, and was not sampled. elementary_mathematics is "
    "mapped to formal_reasoning and is included in the analysed corpus. Each model "
    "was prompted to answer (A-D) and state confidence (0-100) in a fixed template "
    "with independent conversation context per item. Greedy decoding (temperature 0). "
    "Data collected March-April 2026 via the Kaggle Benchmarks platform API. Raw "
    "model outputs are released; re-running the released extraction code over them "
    "regenerates the 47,151-observation corpus exactly."
)

# 3. Limitations aligned with the revision
c["rai:dataLimitations"] = (
    "All results are MMLU-specific; replication on an independent benchmark is "
    "untested. The a priori domain mapping is not factor-analytically validated "
    "(within-domain subject coherence ratio 0.95). Individual model profiles have "
    "weak split-half reliability (median r = .167); the reliable object is the "
    "population-level Applied-minus-(Formal, Science) contrast (split-half r = .342). "
    "Per-model profiles are exploratory. Type-2 AUROC measures discrimination, not "
    "calibration. Median bootstrap CI width .199; 34% of cells exceed .25, "
    "concentrated in high-accuracy models with sparse errors; per-cell error counts "
    "are released. 12 of 33 models have partial runs due to platform API instability "
    "(minimum 598 items for GLM-5), with missingness uniform across domains in every "
    "affected model. The confidence extractor imputes 50 when no value parses: this "
    "fired on 985 of 47,151 responses, of which 980 belong to GPT-oss-120B (65.3% of "
    "its items), so that model's confidence distribution is partly imputed. Gemma 4 "
    "26B A4B never executed and produced no data. Verbalized confidence only; greedy "
    "decoding only; English only. Snapshot from March-April 2026; model behaviour may "
    "have changed."
)

# 4. Register raw outputs
ids = {d.get("@id") for d in c.get("distribution", [])}
if "raw-outputs" not in ids:
    c["distribution"].append({
        "@type": "cr:FileSet",
        "@id": "raw-outputs",
        "name": "raw-outputs",
        "description": ("Raw model outputs for all 33 runs as returned by the Kaggle "
                        "Benchmarks platform: conversation records, task manifests, "
                        "and per-run outputs."),
        "containedIn": {"@id": "hf-repo"},
        "encodingFormat": "application/json",
        "includes": "raw_outputs/**",
    })

# 5. Version bump
c["version"] = "1.1.0"

P.write_text(json.dumps(c, indent=2, ensure_ascii=False), encoding="utf-8")

# Validate
reloaded = json.loads(P.read_text(encoding="utf-8"))
assert "2605.06673" in reloaded["citeAs"]
assert "elementary_mathematics excluded" not in reloaded["rai:dataCollection"]
assert "college_medicine" in reloaded["rai:dataCollection"]
print("croissant.json patched and validated")
print(f"  version:      {reloaded['version']}")
print(f"  citeAs:       {reloaded['citeAs']}")
print(f"  distribution: {[d['@id'] for d in reloaded['distribution']]}")
