#!/usr/bin/env python3
"""
Verify internal logical consistency of hallucination tags.

Checks three properties of the step_hallucination / cumulative_hallucination labels:
  1. Final cumulative tag agrees with the top-level is_correct field.
  2. Epiphany anomalies: cumulative transitions T→F while step_hallucination is still True.
  3. Normal recovery: cumulative transitions T→F while step_hallucination is False.

Usage:
    python analyze_tags.py <dataset.json> [--verbose]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify internal consistency of hallucination tags")
    parser.add_argument("input", type=Path, help="Annotated JSON dataset")
    parser.add_argument("--verbose", action="store_true", help="Print individual conflict cases")
    return parser.parse_args()


def analyze_tag_consistency(dialogs: list) -> dict:
    stats = {
        "total_dialogs": len(dialogs),
        "total_steps": 0,
        "step_hall_count": 0,
        "cum_hall_count": 0,
    }

    consistency_checks = {
        "is_correct_conflicts": [],
        "epiphany_cases": [],
        "recovery_cases": [],
    }

    distributions = {
        "first_step_hall_positions": [],
        "first_cum_hall_positions": [],
        "cum_transitions": [],
        "steps_per_dialog": [],
        "step_hall_density": [],
        "cum_hall_density": [],
    }

    for dialog_idx, dialog in enumerate(dialogs):
        steps = dialog["steps_with_tags"]
        num_steps = len(steps)
        stats["total_steps"] += num_steps
        distributions["steps_per_dialog"].append(num_steps)

        step_halls = [s["step_hallucination"] for s in steps]
        cum_halls  = [s["cumulative_hallucination"] for s in steps]

        stats["step_hall_count"] += sum(step_halls)
        stats["cum_hall_count"]  += sum(cum_halls)

        distributions["step_hall_density"].append(sum(step_halls) / num_steps if num_steps else 0)
        distributions["cum_hall_density"].append(sum(cum_halls)  / num_steps if num_steps else 0)

        if "is_correct" in dialog:
            if (not cum_halls[-1]) != dialog["is_correct"]:
                consistency_checks["is_correct_conflicts"].append({
                    "dialog_id": dialog.get("id", dialog_idx),
                    "final_cum": cum_halls[-1],
                    "is_correct": dialog["is_correct"],
                })

        for i in range(1, num_steps):
            if cum_halls[i - 1] and not cum_halls[i]:
                first_true_idx = cum_halls.index(True) if True in cum_halls else -1
                all_true_before = all(cum_halls[first_true_idx:i]) if first_true_idx >= 0 else False
                if step_halls[i]:
                    consistency_checks["epiphany_cases"].append({
                        "dialog_id": dialog.get("id", dialog_idx),
                        "position": i,
                        "is_severe": all_true_before,
                        "is_last_step": (i == num_steps - 1),
                    })
                else:
                    consistency_checks["recovery_cases"].append({
                        "dialog_id": dialog.get("id", dialog_idx),
                        "position": i,
                        "normalized_pos": i / num_steps,
                        "is_last_step": (i == num_steps - 1),
                    })

        if any(step_halls):
            distributions["first_step_hall_positions"].append(step_halls.index(True) / num_steps)
        if any(cum_halls):
            distributions["first_cum_hall_positions"].append(cum_halls.index(True) / num_steps)

        for i in range(1, num_steps):
            if cum_halls[i - 1] != cum_halls[i]:
                distributions["cum_transitions"].append({
                    "dialog_id": dialog.get("id", dialog_idx),
                    "position": i,
                    "normalized_pos": i / num_steps,
                    "type": "T->F" if cum_halls[i - 1] else "F->T",
                    "step_hall_at_transition": step_halls[i],
                })

    return {"stats": stats, "consistency_checks": consistency_checks, "distributions": distributions}


def print_report(results: dict, verbose: bool = False) -> None:
    stats  = results["stats"]
    checks = results["consistency_checks"]
    dists  = results["distributions"]

    sep = "=" * 70
    print(sep)
    print("Hallucination Tag Consistency Report")
    print(sep)

    print(f"\n[1] Dataset statistics")
    print(f"  Dialogs: {stats['total_dialogs']}")
    print(f"  Steps:   {stats['total_steps']}  (avg {stats['total_steps']/stats['total_dialogs']:.1f} per dialog)")
    print(f"  step_hallucination=True:      {stats['step_hall_count']}/{stats['total_steps']} ({stats['step_hall_count']/stats['total_steps']:.2%})")
    print(f"  cumulative_hallucination=True: {stats['cum_hall_count']}/{stats['total_steps']} ({stats['cum_hall_count']/stats['total_steps']:.2%})")

    print(f"\n[2] is_correct / final cumulative conflicts: {len(checks['is_correct_conflicts'])}")
    if checks["is_correct_conflicts"] and verbose:
        for c in checks["is_correct_conflicts"]:
            print(f"    dialog {c['dialog_id']}: final_cum={c['final_cum']}, is_correct={c['is_correct']}")

    total_recovery = len(checks["recovery_cases"]) + len(checks["epiphany_cases"])
    dialogs_with_recovery = len(set(
        [r["dialog_id"] for r in checks["recovery_cases"]] +
        [e["dialog_id"] for e in checks["epiphany_cases"]]
    ))
    print(f"\n[3] Recovery analysis")
    print(f"  Dialogs with any recovery: {dialogs_with_recovery}/{stats['total_dialogs']} ({dialogs_with_recovery/stats['total_dialogs']:.2%})")
    print(f"  Normal recoveries (cum T->F, step=F): {len(checks['recovery_cases'])}")
    print(f"  Epiphany anomalies (cum T->F, step=T): {len(checks['epiphany_cases'])}")
    if checks["epiphany_cases"]:
        severe = sum(1 for e in checks["epiphany_cases"] if e["is_severe"])
        print(f"    Severe (all-True run before): {severe}")

    print(f"\n[4] Hallucination onset positions")
    if dists["first_step_hall_positions"]:
        print(f"  First step hallucination (mean normalized pos): {statistics.mean(dists['first_step_hall_positions']):.2%}")
    if dists["first_cum_hall_positions"]:
        print(f"  First cum hallucination  (mean normalized pos): {statistics.mean(dists['first_cum_hall_positions']):.2%}")

    transitions = dists["cum_transitions"]
    t_to_f = [t for t in transitions if t["type"] == "T->F"]
    f_to_t = [t for t in transitions if t["type"] == "F->T"]
    print(f"\n[5] Cumulative transition patterns")
    print(f"  F->T (onset):    {len(f_to_t)}")
    print(f"  T->F (recovery): {len(t_to_f)}")

    total_issues = len(checks["is_correct_conflicts"]) + len(checks["epiphany_cases"])
    print(f"\n{sep}")
    if total_issues == 0:
        print("Summary: no critical logical conflicts found.")
    else:
        print(f"Summary: {total_issues} issues detected  "
              f"(is_correct conflicts: {len(checks['is_correct_conflicts'])}, "
              f"epiphanies: {len(checks['epiphany_cases'])})")
    print(sep)


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8") as f:
        dialogs = json.load(f)

    results = analyze_tag_consistency(dialogs)
    print_report(results, args.verbose)

    output_file = args.input.parent / f"{args.input.stem}_consistency_report.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed report saved to: {output_file}")


if __name__ == "__main__":
    main()
