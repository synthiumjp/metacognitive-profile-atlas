"""
Generate data/mmlu_item_locators.csv for the Metacognitive Profile Atlas.

Maps each internal item_id to the canonical MMLU (subject, split, row_index)
locator, so any user can verify or re-derive the exact item pool.

Reproduces the seed-42 stratified sample from notebooks/atlas_benchmark.py
and carries the per-subject row index through it, then verifies the result
against the released CSVs.

RUN FROM THE REPO ROOT:
    pip install datasets pandas
    python scripts/make_mmlu_locators.py
"""
import numpy as np
import pandas as pd
from datasets import load_dataset

DOMAIN_MAP = {
    # Domain 1: Factual Recall (history, geography, world knowledge)
    "high_school_european_history": "factual_recall",
    "high_school_us_history": "factual_recall",
    "high_school_world_history": "factual_recall",
    "prehistory": "factual_recall",
    "world_religions": "factual_recall",
    "high_school_geography": "factual_recall",
    "international_law": "factual_recall",
    "human_aging": "factual_recall",
    "nutrition": "factual_recall",
    "miscellaneous": "factual_recall",
    "global_facts": "factual_recall",
    "virology": "factual_recall",

    # Domain 2: Formal Reasoning (logic, math, CS, abstract)
    "abstract_algebra": "formal_reasoning",
    "formal_logic": "formal_reasoning",
    "logical_fallacies": "formal_reasoning",
    "college_mathematics": "formal_reasoning",
    "high_school_mathematics": "formal_reasoning",
    "elementary_mathematics": "formal_reasoning",
    "high_school_statistics": "formal_reasoning",
    "college_computer_science": "formal_reasoning",
    "high_school_computer_science": "formal_reasoning",
    "computer_security": "formal_reasoning",
    "machine_learning": "formal_reasoning",
    "electrical_engineering": "formal_reasoning",

    # Domain 3: Applied/Professional (law, medicine, business)
    "professional_law": "applied_professional",
    "professional_medicine": "applied_professional",
    "professional_accounting": "applied_professional",
    "clinical_knowledge": "applied_professional",
    "medical_genetics": "applied_professional",
    "management": "applied_professional",
    "marketing": "applied_professional",
    "business_ethics": "applied_professional",
    "professional_psychology": "applied_professional",
    "jurisprudence": "applied_professional",

    # Domain 4: Social/Moral (ethics, sociology, politics, psych)
    "moral_scenarios": "social_moral",
    "moral_disputes": "social_moral",
    "sociology": "social_moral",
    "high_school_government_and_politics": "social_moral",
    "us_foreign_policy": "social_moral",
    "public_relations": "social_moral",
    "security_studies": "social_moral",
    "econometrics": "social_moral",
    "high_school_macroeconomics": "social_moral",
    "high_school_microeconomics": "social_moral",

    # Domain 5: Natural Science (physics, chemistry, biology)
    "high_school_physics": "natural_science",
    "college_physics": "natural_science",
    "high_school_chemistry": "natural_science",
    "college_chemistry": "natural_science",
    "high_school_biology": "natural_science",
    "college_biology": "natural_science",
    "anatomy": "natural_science",
    "astronomy": "natural_science",
    "conceptual_physics": "natural_science",

    # Domain 6: Humanities/Comprehension (philosophy, literature)
    "philosophy": "humanities",
    "high_school_psychology": "humanities",
    "human_sexuality": "humanities",
}

ITEMS_PER_DOMAIN = 250

assert len(DOMAIN_MAP) == 56, f"expected 56 mapped subjects, got {len(DOMAIN_MAP)}"

# ── Load MMLU test split, tag canonical row index within each subject
ds = load_dataset("cais/mmlu", "all", split="test")
mmlu_df = pd.DataFrame(ds)
mmlu_df["row_index"] = mmlu_df.groupby("subject").cumcount()

all_subjects = set(mmlu_df["subject"].unique())
mapped_subjects = set(DOMAIN_MAP)
print(f"MMLU test-split subjects: {len(all_subjects)}")
print(f"Mapped subjects:          {len(mapped_subjects)}")
print(f"NOT MAPPED (excluded):    {sorted(all_subjects - mapped_subjects)}")
print(f"Mapped but absent:        {sorted(mapped_subjects - all_subjects)}")

# ── Replicate the sampling pipeline verbatim
mmlu_df["domain"] = mmlu_df["subject"].map(DOMAIN_MAP)
mapped = mmlu_df[mmlu_df["domain"].notna()].copy()

np.random.seed(42)
parts = []
for domain in sorted(set(DOMAIN_MAP.values())):
    items = mapped[mapped["domain"] == domain]
    parts.append(items.sample(n=min(ITEMS_PER_DOMAIN, len(items)), random_state=42))

result = pd.concat(parts).reset_index(drop=True)
result = result.sample(frac=1, random_state=42).reset_index(drop=True)
result["item_id"] = result.index

# ── Verify against a released model CSV
released = pd.read_csv("data/raw/metacognitive_profile_results.csv")
merged = released.merge(
    result[["item_id", "subject", "row_index", "domain", "question"]],
    on="item_id", suffixes=("_rel", "_loc"))

subject_ok = merged["subject_rel"] == merged["subject_loc"]
domain_ok = merged["domain_rel"] == merged["domain_loc"]
# released question text is truncated to 80 chars
question_ok = merged.apply(
    lambda r: r["question_loc"].startswith(str(r["question_rel"])[:60]), axis=1)

print()
print(f"Rows compared:  {len(merged)}")
print(f"Subject match:  {subject_ok.mean():.1%}")
print(f"Domain match:   {domain_ok.mean():.1%}")
print(f"Question match: {question_ok.mean():.1%}")

if not (subject_ok.all() and domain_ok.all() and question_ok.all()):
    bad = merged[~(subject_ok & domain_ok & question_ok)]
    print(f"\nMISMATCHES ({len(bad)}) - first 5:")
    print(bad[["item_id", "subject_rel", "subject_loc"]].head())
    raise SystemExit("Reconstruction does not match released data. Do not ship.")

out = result[["item_id", "subject", "row_index", "domain"]].copy()
out["split"] = "test"
out = out[["item_id", "subject", "split", "row_index", "domain"]]
out.to_csv("data/mmlu_item_locators.csv", index=False)
print(f"\nVERIFIED. Wrote data/mmlu_item_locators.csv ({len(out)} rows)")
