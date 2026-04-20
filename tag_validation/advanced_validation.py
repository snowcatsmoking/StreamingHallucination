#!/usr/bin/env python3
"""Advanced statistical validation of hallucination tags"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from collections import Counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced tag validation")
    parser.add_argument("input", type=Path, help="Cleaned JSON dataset")
    return parser.parse_args()


def test_monotonicity_strength(dialogs: list) -> dict:
    """Test how strictly cumulative tags follow monotonic pattern"""
    results = {
        'strict_monotonic': 0,      # never T->F
        'with_recovery': 0,         # has T->F but reasonable
        'multiple_recovery': 0,     # multiple T->F transitions
        'recovery_histogram': [],
    }

    for dialog in dialogs:
        cum_halls = [s["cumulative_hallucination"] for s in dialog["steps_with_tags"]]

        # Count transitions
        transitions = []
        for i in range(1, len(cum_halls)):
            if cum_halls[i-1] != cum_halls[i]:
                transition_type = "T->F" if cum_halls[i-1] else "F->T"
                transitions.append(transition_type)

        recovery_count = transitions.count("T->F")
        results['recovery_histogram'].append(recovery_count)

        if recovery_count == 0:
            results['strict_monotonic'] += 1
        elif recovery_count == 1:
            results['with_recovery'] += 1
        else:
            results['multiple_recovery'] += 1

    return results


def test_step_cum_causality(dialogs: list) -> dict:
    """Test if step hallucinations cause cumulative hallucinations"""
    results = {
        'step_true_cum_true': 0,
        'step_true_cum_false': 0,
        'step_false_cum_true': 0,
        'step_false_cum_false': 0,
        'causality_violations': [],  # step=T but cum stays F afterward
    }

    for dialog in dialogs:
        steps = dialog["steps_with_tags"]
        step_halls = [s["step_hallucination"] for s in steps]
        cum_halls = [s["cumulative_hallucination"] for s in steps]

        for i, (step, cum) in enumerate(zip(step_halls, cum_halls)):
            # Count co-occurrence
            if step and cum:
                results['step_true_cum_true'] += 1
            elif step and not cum:
                results['step_true_cum_false'] += 1
            elif not step and cum:
                results['step_false_cum_true'] += 1
            else:
                results['step_false_cum_false'] += 1

            # Check causality: if step=T at position i, cum should be T for all j>=i
            if step:
                for j in range(i, len(cum_halls)):
                    if not cum_halls[j]:
                        results['causality_violations'].append({
                            'dialog_id': dialog.get('id'),
                            'step_pos': i,
                            'cum_false_pos': j
                        })
                        break  # only record first violation

    return results


def test_final_consistency(dialogs: list) -> dict:
    """Test if final cumulative tag matches is_correct"""
    results = {
        'total': 0,
        'consistent': 0,
        'final_cum_true_correct_false': 0,
        'final_cum_false_correct_true': 0,
    }

    for dialog in dialogs:
        if "is_correct" not in dialog:
            continue

        results['total'] += 1
        final_cum = dialog["steps_with_tags"][-1]["cumulative_hallucination"]
        is_correct = dialog["is_correct"]

        # Logic: final_cum=True means final answer is wrong, so is_correct should be False
        expected_correct = not final_cum

        if expected_correct == is_correct:
            results['consistent'] += 1
        else:
            if final_cum and is_correct:
                results['final_cum_true_correct_true'] += 1
            else:
                results['final_cum_false_correct_false'] += 1

    return results


def test_temporal_correlation(dialogs: list) -> dict:
    """Test if early hallucinations correlate with more total hallucinations"""
    results = {
        'early_vs_total_step': [],   # (first_pos, total_count)
        'early_vs_total_cum': [],
    }

    for dialog in dialogs:
        steps = dialog["steps_with_tags"]
        step_halls = [s["step_hallucination"] for s in steps]
        cum_halls = [s["cumulative_hallucination"] for s in steps]
        num_steps = len(steps)

        # Step hallucination
        if any(step_halls):
            first_step_pos = step_halls.index(True) / num_steps
            total_step_count = sum(step_halls) / num_steps
            results['early_vs_total_step'].append((first_step_pos, total_step_count))

        # Cumulative hallucination
        if any(cum_halls):
            first_cum_pos = cum_halls.index(True) / num_steps
            total_cum_count = sum(cum_halls) / num_steps
            results['early_vs_total_cum'].append((first_cum_pos, total_cum_count))

    return results


def test_recovery_reasonableness(dialogs: list) -> dict:
    """Test if recoveries happen in reasonable contexts"""
    results = {
        'recoveries': [],
        'recovery_gap_stats': [],  # time between entering hallucination and recovery
    }

    for dialog in dialogs:
        steps = dialog["steps_with_tags"]
        step_halls = [s["step_hallucination"] for s in steps]
        cum_halls = [s["cumulative_hallucination"] for s in steps]
        num_steps = len(steps)

        # Find all recoveries (T->F transitions)
        last_f_to_t = -1  # position of last F->T transition
        for i in range(1, num_steps):
            if not cum_halls[i-1] and cum_halls[i]:
                last_f_to_t = i
            elif cum_halls[i-1] and not cum_halls[i]:
                # Recovery found
                gap = i - last_f_to_t if last_f_to_t >= 0 else i
                results['recovery_gap_stats'].append(gap)

                results['recoveries'].append({
                    'dialog_id': dialog.get('id'),
                    'position': i,
                    'normalized_pos': i / num_steps,
                    'step_at_recovery': step_halls[i],
                    'gap_since_hallucination': gap,
                })

    return results


def compute_inter_annotator_metrics(dialogs: list) -> dict:
    """Compute metrics that would indicate annotation quality"""
    # Simulate what inter-annotator agreement would look like
    # by measuring internal consistency

    results = {
        'step_cum_agreement': 0,  # when step=T, how often is cum=T
        'cum_persistence': 0,     # once cum=T, how often does it stay T
    }

    total_step_true = 0
    step_true_cum_true = 0
    total_cum_true = 0
    cum_true_stays_true = 0

    for dialog in dialogs:
        steps = dialog["steps_with_tags"]
        step_halls = [s["step_hallucination"] for s in steps]
        cum_halls = [s["cumulative_hallucination"] for s in steps]

        for i, (step, cum) in enumerate(zip(step_halls, cum_halls)):
            if step:
                total_step_true += 1
                if cum:
                    step_true_cum_true += 1

            if i > 0 and cum_halls[i-1]:
                total_cum_true += 1
                if cum:
                    cum_true_stays_true += 1

    if total_step_true > 0:
        results['step_cum_agreement'] = step_true_cum_true / total_step_true

    if total_cum_true > 0:
        results['cum_persistence'] = cum_true_stays_true / total_cum_true

    return results


def main() -> None:
    args = parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        dialogs = json.load(f)

    print("=" * 70)
    print("Advanced Statistical Validation of Hallucination Tags")
    print("=" * 70)
    print(f"\nDataset: {args.input}")
    print(f"Total dialogues: {len(dialogs)}\n")

    # Test 1: Monotonicity strength
    print("【Test 1: Cumulative Monotonicity Strength】")
    mono_results = test_monotonicity_strength(dialogs)
    total = len(dialogs)
    print(f"  Strictly monotonic (no recovery): {mono_results['strict_monotonic']} ({mono_results['strict_monotonic']/total:.2%})")
    print(f"  With 1 recovery: {mono_results['with_recovery']} ({mono_results['with_recovery']/total:.2%})")
    print(f"  With 2+ recoveries: {mono_results['multiple_recovery']} ({mono_results['multiple_recovery']/total:.2%})")

    recovery_counter = Counter(mono_results['recovery_histogram'])
    print(f"  Recovery distribution: {dict(sorted(recovery_counter.items()))}")

    # Test 2: Step-Cum causality
    print("\n【Test 2: Step → Cumulative Causality】")
    causality = test_step_cum_causality(dialogs)
    total_steps = sum([causality['step_true_cum_true'], causality['step_true_cum_false'],
                       causality['step_false_cum_true'], causality['step_false_cum_false']])

    print(f"  Contingency table:")
    print(f"    step=T, cum=T: {causality['step_true_cum_true']} ({causality['step_true_cum_true']/total_steps:.2%})")
    print(f"    step=T, cum=F: {causality['step_true_cum_false']} ({causality['step_true_cum_false']/total_steps:.2%})")
    print(f"    step=F, cum=T: {causality['step_false_cum_true']} ({causality['step_false_cum_true']/total_steps:.2%})")
    print(f"    step=F, cum=F: {causality['step_false_cum_false']} ({causality['step_false_cum_false']/total_steps:.2%})")

    print(f"\n  Causality violations: {len(causality['causality_violations'])}")
    print(f"    (step=T but cum remains F in subsequent steps)")

    # Chi-square test for independence
    contingency = np.array([
        [causality['step_true_cum_true'], causality['step_true_cum_false']],
        [causality['step_false_cum_true'], causality['step_false_cum_false']]
    ])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    print(f"\n  Chi-square test (independence): χ²={chi2:.2f}, p-value={p_value:.2e}")
    print(f"    Interpretation: {'Dependent (good!)' if p_value < 0.001 else 'Independent (bad!)'}")

    # Test 3: Final consistency
    print("\n【Test 3: Final Cumulative ↔ is_correct Consistency】")
    final = test_final_consistency(dialogs)
    if final['total'] > 0:
        print(f"  Total tested: {final['total']}")
        print(f"  Consistent: {final['consistent']} ({final['consistent']/final['total']:.2%})")
        print(f"  Inconsistent: {final['total'] - final['consistent']}")

    # Test 4: Temporal correlation
    print("\n【Test 4: Early Hallucination vs Total Density】")
    temporal = test_temporal_correlation(dialogs)

    if temporal['early_vs_total_step']:
        early_step, total_step = zip(*temporal['early_vs_total_step'])
        corr_step, p_step = stats.pearsonr(early_step, total_step)
        print(f"  Step hallucination:")
        print(f"    Correlation (early position vs density): r={corr_step:.3f}, p={p_step:.2e}")
        print(f"    Interpretation: {'Earlier → more total' if corr_step < -0.1 else 'No strong pattern'}")

    if temporal['early_vs_total_cum']:
        early_cum, total_cum = zip(*temporal['early_vs_total_cum'])
        corr_cum, p_cum = stats.pearsonr(early_cum, total_cum)
        print(f"  Cumulative hallucination:")
        print(f"    Correlation (early position vs density): r={corr_cum:.3f}, p={p_cum:.2e}")

    # Test 5: Recovery reasonableness
    print("\n【Test 5: Recovery Pattern Analysis】")
    recovery = test_recovery_reasonableness(dialogs)
    if recovery['recovery_gap_stats']:
        gaps = recovery['recovery_gap_stats']
        print(f"  Total recoveries: {len(gaps)}")
        print(f"  Average gap (steps in hallucination): {np.mean(gaps):.1f}")
        print(f"  Median gap: {np.median(gaps):.1f}")
        print(f"  Gap range: [{min(gaps)}, {max(gaps)}]")

    # Test 6: Annotation quality proxy
    print("\n【Test 6: Internal Consistency (Annotation Quality Proxy)】")
    quality = compute_inter_annotator_metrics(dialogs)
    print(f"  Step-Cum agreement: {quality['step_cum_agreement']:.2%}")
    print(f"    (When step=T, how often cum=T)")
    print(f"  Cumulative persistence: {quality['cum_persistence']:.2%}")
    print(f"    (Once cum=T, how often it stays T)")

    # Overall assessment
    print("\n" + "=" * 70)
    print("【Overall Assessment】")

    score = 0
    max_score = 6

    # Criterion 1: High monotonicity
    if mono_results['strict_monotonic'] / total > 0.7:
        score += 1
        print("  ✓ Strong monotonicity (>70% strict monotonic)")
    else:
        print(f"  ✗ Weak monotonicity ({mono_results['strict_monotonic']/total:.1%} strict)")

    # Criterion 2: Strong step-cum dependency
    if p_value < 0.001:
        score += 1
        print("  ✓ Strong step-cumulative dependency (p<0.001)")

    # Criterion 3: Perfect final consistency
    if final['total'] > 0 and final['consistent'] / final['total'] > 0.99:
        score += 1
        print("  ✓ Near-perfect final consistency (>99%)")

    # Criterion 4: Reasonable causality violations
    if len(causality['causality_violations']) / total < 0.05:
        score += 1
        print(f"  ✓ Low causality violations (<5%)")

    # Criterion 5: High cumulative persistence
    if quality['cum_persistence'] > 0.95:
        score += 1
        print(f"  ✓ High cumulative persistence (>95%)")

    # Criterion 6: Temporal correlation makes sense
    if temporal['early_vs_total_step']:
        if corr_step < -0.1:  # earlier hallucination → more total
            score += 1
            print(f"  ✓ Logical temporal pattern (early → more total)")

    print(f"\n  Final Score: {score}/{max_score}")

    if score >= 5:
        print("  ★★★ EXCELLENT - Strong evidence of logical consistency")
    elif score >= 4:
        print("  ★★☆ GOOD - Reasonable internal logic with minor issues")
    else:
        print("  ★☆☆ FAIR - Significant inconsistencies detected")

    print("=" * 70)


if __name__ == "__main__":
    main()
