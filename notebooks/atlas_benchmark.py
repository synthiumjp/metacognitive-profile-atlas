"""
============================================================================
Benchmark: Metacognitive Profile — Domain-Stratified MMLU
Author: JP Cacioli (Classical Minds, Modern Machines)
============================================================================

WHAT THIS TASK MEASURES:
  Whether an LLM's verbalized confidence discriminates correct from
  incorrect responses DIFFERENTIALLY across cognitive domains. 

  1500 MMLU items are sampled across 6 cognitive domains (~250 each).
  Domain-level AUROC, validity screening, and selective prediction
  are computed per domain to construct a metacognitive profile.

WHY IT MATTERS:
  Aggregate validity does not guarantee domain-specific validity.
  A model classified Valid overall may show near-chance metacognitive
  monitoring in specific cognitive domains (Cacioli, 2026f, Table 2).
  Domain-level profiling reveals WHERE a model monitors well and
  where it does not.

DOMAINS (a priori mapping from MMLU subjects):
  1. Factual Recall — history, geography, world facts
  2. Formal Reasoning — logic, mathematics, CS theory
  3. Applied/Professional — law, medicine, engineering, business
  4. Social/Moral — ethics, sociology, psychology, politics
  5. Natural Science — physics, chemistry, biology, astronomy
  6. Humanities/Comprehension — philosophy, literature, languages

PROTOCOL:
  1. Present MMLU item (multiple choice)
  2. Model answers (A/B/C/D) and states confidence (0-100)
  3. After all items: binarise confidence at median
  4. Run validity_screen.screen() per domain
  5. Compute Type 2 AUROC per domain (bootstrap 95% CI)
  6. Report domain-level profile matrix

REFERENCE:
  Cacioli, J. P. (2026d). Before You Interpret the Profile.
  Cacioli, J. P. (2026e). Screen Before You Interpret.
  Cacioli, J. P. (2026f). Selective Prediction as External Criterion.
  Tool: pip install validity-screen
  Code: github.com/synthiumjp/validity-scaling-llm

============================================================================
"""

import kaggle_benchmarks as kbench
import re
import csv
import json
import numpy as np
import pandas as pd

# Install dependencies
import subprocess
subprocess.run(["pip", "install", "validity-screen", "datasets", "scikit-learn", "-q"], check=True)
from validity_screen import screen, binarise
from sklearn.metrics import roc_auc_score
import time


def prompt_with_retry(llm, prompt, max_retries=3, backoff=5):
    """Retry llm.prompt() on API failures. Returns None on total failure."""
    for attempt in range(max_retries):
        try:
            response = llm.prompt(prompt)
            if response is not None:
                return response
        except Exception as e:
            print(f"  API error (attempt {attempt+1}/{max_retries}): {e}")
        if attempt < max_retries - 1:
            wait = backoff * (2 ** attempt)
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)
    print(f"  All {max_retries} attempts failed. Skipping item.")
    return None


# ══════════════════════════════════════════════════════════════
# DOMAIN MAPPING — MMLU subjects to 6 cognitive domains
# ══════════════════════════════════════════════════════════════

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

DOMAIN_LABELS = {
    "factual_recall": "Factual",
    "formal_reasoning": "Formal",
    "applied_professional": "Applied",
    "social_moral": "Social",
    "natural_science": "Science",
    "humanities": "Human.",
}

ITEMS_PER_DOMAIN = 250


# ══════════════════════════════════════════════════════════════
# ITEM BANK — 1500 MMLU items, stratified by cognitive domain
# ══════════════════════════════════════════════════════════════

def load_mmlu_items():
    """Load MMLU items with domain-stratified sampling."""
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    mmlu_df = pd.DataFrame(ds)

    # Assign domains
    mmlu_df["domain"] = mmlu_df["subject"].map(DOMAIN_MAP)

    # Drop subjects not in our mapping
    mapped = mmlu_df[mmlu_df["domain"].notna()].copy()
    unmapped = mmlu_df[mmlu_df["domain"].isna()]
    if len(unmapped) > 0:
        unmapped_subjects = unmapped["subject"].unique()
        print(f"  {len(unmapped_subjects)} unmapped subjects excluded ({len(unmapped)} items)")

    # Stratified sample: ITEMS_PER_DOMAIN per domain
    np.random.seed(42)
    sampled_parts = []
    for domain in sorted(set(DOMAIN_MAP.values())):
        domain_items = mapped[mapped["domain"] == domain]
        n_available = len(domain_items)
        n_sample = min(ITEMS_PER_DOMAIN, n_available)
        sampled = domain_items.sample(n=n_sample, random_state=42)
        sampled_parts.append(sampled)
        print(f"  {DOMAIN_LABELS.get(domain, domain):<10}: {n_sample:>4} items sampled from {n_available:>5} available")

    result = pd.concat(sampled_parts).reset_index(drop=True)
    # Shuffle to avoid domain-clustering effects on API rate limits
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)

    return result


def format_question(row):
    """Format an MMLU item as a multiple-choice string."""
    choices = row["choices"]
    q = f"{row['question']}\n"
    for i, choice in enumerate(choices):
        q += f"  {chr(65+i)}) {choice}\n"
    return q


# ══════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════

ANSWER_AND_CONFIDENCE_PROMPT = (
    "Answer this multiple-choice question.\n\n"
    "{question}\n"
    "First, state your answer (A, B, C, or D).\n"
    "Then, state your confidence that your answer is correct "
    "as a number from 0 (pure guess) to 100 (certain).\n\n"
    "Reply in EXACTLY this format:\n"
    "Answer: [letter]\n"
    "Confidence: [number]"
)


# ══════════════════════════════════════════════════════════════
# PARSING
# ══════════════════════════════════════════════════════════════

def parse_answer(text):
    """Extract answer letter from response."""
    match = re.search(r'[Aa]nswer:\s*([A-Da-d])', text)
    if match:
        return match.group(1).upper()
    match = re.search(r'\b([A-D])\b', text)
    if match:
        return match.group(1)
    return "X"


def parse_confidence(text):
    """Extract confidence score from response."""
    match = re.search(r'[Cc]onfidence:\s*(\d+)', text)
    if match:
        return min(100, max(0, int(match.group(1))))
    nums = re.findall(r'\b(\d{1,3})\b', text)
    for n in nums:
        v = int(n)
        if 0 <= v <= 100:
            return v
    return 50


# ══════════════════════════════════════════════════════════════
# DOMAIN-LEVEL ANALYSIS
# ══════════════════════════════════════════════════════════════

def bootstrap_auroc(correct, confidence, n_boot=2000, seed=42):
    """Bootstrap 95% CI for AUROC."""
    rng = np.random.default_rng(seed)
    n = len(correct)
    if len(np.unique(correct)) < 2 or len(np.unique(confidence)) < 2:
        return np.nan, np.nan, np.nan
    point = roc_auc_score(correct.astype(int), confidence)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y, s = correct[idx], confidence[idx]
        if len(np.unique(y)) >= 2 and len(np.unique(s)) >= 2:
            boots.append(roc_auc_score(y.astype(int), s))
    if len(boots) < 100:
        return point, np.nan, np.nan
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


def run_domain_analysis(results_df):
    """Run per-domain validity screening and AUROC."""
    domains = sorted(results_df["domain"].unique())

    print("\n" + "=" * 70)
    print("DOMAIN-LEVEL METACOGNITIVE PROFILE")
    print("=" * 70)

    # Global median for binarisation (consistent threshold across domains)
    global_median = np.median(results_df["confidence"].values)
    print(f"Global confidence median: {global_median:.0f}")

    profile = []

    print(f"\n{'Domain':<12} {'n':>5} {'Acc':>6} {'AUROC':>7} {'[95% CI]':>18} {'Tier':>10} {'r':>7} {'ConfSD':>7}")
    print("-" * 80)

    for domain in domains:
        dm = results_df[results_df["domain"] == domain]
        n = len(dm)
        correct = dm["is_correct"].values.astype(bool)
        confidence = dm["confidence"].values.astype(float)
        acc = correct.mean()
        conf_sd = confidence.std()

        # AUROC on continuous confidence
        auroc, ci_lo, ci_hi = bootstrap_auroc(correct, confidence)

        # Binarise at global median for validity screen
        conf_binary = confidence >= global_median

        # Run validity screen
        try:
            sr = screen(
                correct, conf_binary,
                model_name="",
                benchmark_name=f"MMLU-{DOMAIN_LABELS.get(domain, domain)}",
            )
            tier = sr.tier
            r_val = sr.r_conf_correct.value if sr.r_conf_correct else np.nan
        except Exception:
            tier = "Error"
            r_val = np.nan

        profile.append({
            "domain": domain,
            "domain_label": DOMAIN_LABELS.get(domain, domain),
            "n": n,
            "accuracy": acc,
            "auroc": auroc,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "tier": tier,
            "r": r_val,
            "conf_sd": conf_sd,
        })

        ci_str = f"[{ci_lo:.3f}, {ci_hi:.3f}]" if not np.isnan(ci_lo) else "[—, —]"
        print(f"{DOMAIN_LABELS.get(domain, domain):<12} {n:>5} {acc:>6.3f} {auroc:>7.3f} {ci_str:>18} {tier:>10} {r_val:>7.3f} {conf_sd:>7.1f}")

    # Ipsative profile (deviation from own mean)
    aurocs = [p["auroc"] for p in profile if not np.isnan(p["auroc"])]
    if aurocs:
        mean_auroc = np.mean(aurocs)
        print(f"\nMean AUROC: {mean_auroc:.3f}")
        print(f"\nIpsative deviations (domain AUROC - model mean):")
        for p in profile:
            if not np.isnan(p["auroc"]):
                dev = p["auroc"] - mean_auroc
                bar = "+" * int(max(0, dev * 50)) + "-" * int(max(0, -dev * 50))
                print(f"  {p['domain_label']:<12} {dev:+.3f}  {bar}")

    return profile


# ══════════════════════════════════════════════════════════════
# TASK
# ══════════════════════════════════════════════════════════════

@kbench.task(
    name='metacognitive_profile_mmlu',
    description=(
        'Metacognitive profiling via domain-stratified MMLU. '
        'Tests whether verbalized confidence carries domain-specific '
        'item-level information. Reports per-domain AUROC, validity '
        'screening, and ipsative profile per Cacioli (2026e).'
    )
)
def metacognitive_profile_mmlu(llm) -> float:
    items = load_mmlu_items()
    print(f"\nLoaded {len(items)} MMLU items across {items['domain'].nunique()} domains")

    results_rows = []

    for idx, row in items.iterrows():
        question_text = format_question(row)
        correct_letter = chr(65 + row["answer"])

        with kbench.chats.new(f"mmlu_{idx:04d}"):
            prompt = ANSWER_AND_CONFIDENCE_PROMPT.format(question=question_text)
            response = prompt_with_retry(llm, prompt)

            if response is None:
                kbench.assertions.assert_true(
                    True,
                    expectation=(
                        f"[{row['subject'][:20]}|{DOMAIN_LABELS.get(row['domain'], '?')}] "
                        f"Ans=SKIP (exp={correct_letter}) "
                        f"Correct=False Conf=NA [API_FAILURE]"
                    )
                )
                continue

            answer = parse_answer(response)
            confidence = parse_confidence(response)
            is_correct = (answer == correct_letter)

            results_rows.append({
                'item_id': idx,
                'subject': row['subject'],
                'domain': row['domain'],
                'question': row['question'][:80],
                'answer': answer,
                'correct_answer': correct_letter,
                'is_correct': is_correct,
                'confidence': confidence,
            })

            kbench.assertions.assert_true(
                True,
                expectation=(
                    f"[{row['subject'][:20]}|{DOMAIN_LABELS.get(row['domain'], '?')}] "
                    f"Ans={answer} (exp={correct_letter}) "
                    f"Correct={is_correct} Conf={confidence}"
                )
            )

    # ══════════════════════════════════════════════════════════
    # POST-HOC ANALYSIS
    # ══════════════════════════════════════════════════════════

    results_df = pd.DataFrame(results_rows)
    model_name = llm.name if hasattr(llm, 'name') else "unknown"

    correct_arr = results_df['is_correct'].values.astype(bool)
    confidence_arr = results_df['confidence'].values.astype(float)

    # ── Aggregate screening ──
    median_conf = np.median(confidence_arr)
    conf_binary = confidence_arr >= median_conf

    sr = screen(
        correct_arr, conf_binary,
        model_name=model_name,
        benchmark_name="MMLU (1500-item domain-stratified)",
        elicitation_method="Verbalized confidence (0-100)",
        confidence_format="Continuous, binarised at median",
        binarisation_threshold=f"median={median_conf:.0f}",
        probe_timing="Concurrent",
    )

    if len(np.unique(correct_arr)) >= 2 and len(np.unique(confidence_arr)) >= 2:
        auroc = roc_auc_score(correct_arr.astype(int), confidence_arr)
    else:
        auroc = float("nan")

    # ── Print aggregate results ──
    print('\n' + '=' * 70)
    print('AGGREGATE VALIDITY SCREENING')
    print('=' * 70)
    print(f'Model:        {model_name}')
    n_skipped = len(items) - len(results_df)
    print(f'Items:        {len(correct_arr)} ({n_skipped} skipped)')
    print(f'Accuracy:     {correct_arr.mean():.3f}')
    print(f'Conf mean:    {confidence_arr.mean():.1f}')
    print(f'Conf SD:      {confidence_arr.std():.1f}')
    print(f'Conf median:  {median_conf:.0f}')
    print(f'AUROC:        {auroc:.3f}')
    print(f'Tier:         {sr.tier}')

    if sr.L:
        print(f'L:            {sr.L.value:.3f} [{sr.L.ci_lower:.3f}, {sr.L.ci_upper:.3f}]')
    if sr.r_conf_correct:
        print(f'r:            {sr.r_conf_correct.value:+.3f}')

    print('\n' + sr.vrs_table())

    # ── Domain-level profile ──
    domain_profile = run_domain_analysis(results_df)

    # ── Selective prediction (aggregate) ──
    print('\n' + '=' * 70)
    print('SELECTIVE PREDICTION (AGGREGATE)')
    print('=' * 70)
    baseline = correct_arr.mean()
    sel_df = results_df.sort_values('confidence', ascending=False)
    for cov in [1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1]:
        k = max(1, int(np.ceil(len(sel_df) * cov)))
        sel_acc = sel_df.iloc[:k]['is_correct'].mean()
        gain = sel_acc - baseline
        print(f'  coverage={cov:.0%}  acc={sel_acc:.3f}  gain={gain:+.3f}')

    # ══════════════════════════════════════════════════════════
    # SAVE CSVs
    # ══════════════════════════════════════════════════════════

    results_df['model'] = model_name
    results_df.to_csv(
        '/kaggle/working/metacognitive_profile_results.csv',
        index=False, quoting=csv.QUOTE_ALL
    )

    # Domain profile summary
    profile_df = pd.DataFrame(domain_profile)
    profile_df['model'] = model_name
    profile_df.to_csv(
        '/kaggle/working/metacognitive_profile_domains.csv',
        index=False, quoting=csv.QUOTE_ALL
    )

    # Aggregate summary
    summary_row = {
        'model': model_name,
        'tier': sr.tier,
        'n_items': len(correct_arr),
        'accuracy': round(float(correct_arr.mean()), 4),
        'auroc': round(float(auroc), 4),
        'conf_mean': round(float(confidence_arr.mean()), 1),
        'conf_sd': round(float(confidence_arr.std()), 1),
        'L': round(sr.L.value, 4) if sr.L else '',
        'r': round(sr.r_conf_correct.value, 4) if sr.r_conf_correct else '',
    }
    # Add domain AUROCs as columns
    for p in domain_profile:
        summary_row[f"auroc_{p['domain_label']}"] = round(p['auroc'], 4) if not np.isnan(p['auroc']) else ''

    pd.DataFrame([summary_row]).to_csv(
        '/kaggle/working/metacognitive_profile_summary.csv',
        index=False, quoting=csv.QUOTE_ALL
    )

    print('\n' + '=' * 70)
    print('CSV FILES SAVED')
    print('=' * 70)
    print('  metacognitive_profile_results.csv   (item-level)')
    print('  metacognitive_profile_domains.csv   (domain-level profile)')
    print('  metacognitive_profile_summary.csv   (aggregate + domain AUROCs)')
    print('=' * 70)

    return round(float(auroc), 4)


# ══════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════

metacognitive_profile_mmlu.run(llm=kbench.llm)

