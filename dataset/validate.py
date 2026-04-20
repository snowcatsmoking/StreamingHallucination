"""
Validate hallucination tag quality for the CHAINED dataset.

Two checks are run, always together:

  1. Completeness  — every step has both step_hallucination and
                     cumulative_hallucination fields (no missing tags).

  2. Consistency   — internal logic of the tag sequence is sound:
       - is_correct vs. final cumulative_hallucination agree
       - "epiphany" anomalies (cum flips T->F but the step itself has
         step_hallucination=True) are flagged
       - normal recovery (cum flips T->F with step_hallucination=False)
         is reported as informational

Usage:
    python validate.py CHAINED_with_tag.json
    python validate.py CHAINED_with_tag.json --verbose
"""

import argparse
import json
import statistics
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Check 1: completeness
# ---------------------------------------------------------------------------

def check_completeness(data: list[dict]) -> list[dict]:
    """Return a list of {id, step_id, missing_keys} for every incomplete step."""
    issues = []
    for item in data:
        item_id = item.get("id", "?")
        for step in item.get("steps", []):
            missing = [
                k for k in ("step_hallucination", "cumulative_hallucination")
                if k not in step
            ]
            if missing:
                issues.append({
                    "id": item_id,
                    "step_id": step.get("step_id", "?"),
                    "missing_keys": missing,
                })
    return issues


# ---------------------------------------------------------------------------
# Check 2: consistency
# ---------------------------------------------------------------------------

def check_consistency(data: list[dict]) -> dict:
    stats = {"total_dialogs": len(data), "total_steps": 0,
             "step_hall_count": 0, "cum_hall_count": 0}

    is_correct_conflicts = []
    epiphany_cases = []
    recovery_cases = []

    first_step_positions = []
    first_cum_positions = []
    step_hall_densities = []
    cum_hall_densities = []
    cum_transitions = []
    steps_per_dialog = []

    for item in data:
        steps = item.get("steps", [])
        n = len(steps)
        if n == 0:
            continue

        stats["total_steps"] += n
        steps_per_dialog.append(n)

        step_flags = [bool(s.get("step_hallucination")) for s in steps]
        cum_flags  = [bool(s.get("cumulative_hallucination")) for s in steps]

        stats["step_hall_count"] += sum(step_flags)
        stats["cum_hall_count"]  += sum(cum_flags)

        step_hall_densities.append(sum(step_flags) / n)
        cum_hall_densities.append(sum(cum_flags) / n)

        # is_correct vs. final cumulative_hallucination
        if "is_correct" in item:
            if item["is_correct"] == cum_flags[-1]:
                is_correct_conflicts.append({
                    "id": item.get("id"),
                    "final_cum": cum_flags[-1],
                    "is_correct": item["is_correct"],
                })

        # Transition analysis
        for i in range(1, n):
            if cum_flags[i - 1] == cum_flags[i]:
                continue
            t_type = "T->F" if cum_flags[i - 1] else "F->T"
            cum_transitions.append({
                "id": item.get("id"), "position": i,
                "normalized_pos": i / n, "type": t_type,
                "step_hall_at_transition": step_flags[i],
            })
            if t_type == "T->F":
                entry = {"id": item.get("id"), "position": i,
                         "normalized_pos": i / n, "is_last": (i == n - 1)}
                if step_flags[i]:
                    first_true = cum_flags.index(True) if True in cum_flags else -1
                    severe = first_true >= 0 and all(cum_flags[first_true:i])
                    entry["is_severe"] = severe
                    epiphany_cases.append(entry)
                else:
                    recovery_cases.append(entry)

        # First-hallucination positions
        if any(step_flags):
            first_step_positions.append(step_flags.index(True) / n)
        if any(cum_flags):
            first_cum_positions.append(cum_flags.index(True) / n)

    return {
        "stats": stats,
        "is_correct_conflicts": is_correct_conflicts,
        "epiphany_cases": epiphany_cases,
        "recovery_cases": recovery_cases,
        "first_step_positions": first_step_positions,
        "first_cum_positions": first_cum_positions,
        "step_hall_densities": step_hall_densities,
        "cum_hall_densities": cum_hall_densities,
        "cum_transitions": cum_transitions,
        "steps_per_dialog": steps_per_dialog,
    }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def _mean(lst: list) -> float:
    return statistics.mean(lst) if lst else 0.0


def print_report(completeness_issues: list[dict], consistency: dict, verbose: bool):
    c = consistency
    stats = c["stats"]
    n_d = stats["total_dialogs"]
    n_s = stats["total_steps"]

    print("=" * 70)
    print("CHAINED Dataset Tag Validation Report")
    print("=" * 70)

    # --- Completeness ---
    print("\n[1] Completeness Check")
    if completeness_issues:
        print(f"  FAIL — {len(completeness_issues)} step(s) missing tags:")
        for issue in completeness_issues[:20]:
            print(f"    id={issue['id']} step_id={issue['step_id']} "
                  f"missing: {issue['missing_keys']}")
        if len(completeness_issues) > 20:
            print(f"    ... and {len(completeness_issues) - 20} more")
    else:
        print("  PASS — all steps have both hallucination tag fields")

    # --- Basic stats ---
    print(f"\n[2] Dataset Statistics")
    print(f"  Dialogs : {n_d}")
    print(f"  Steps   : {n_s}  (avg {n_s / n_d:.1f}/dialog)" if n_d else "  Steps: 0")
    if n_s:
        print(f"  step_hallucination=True  : {stats['step_hall_count']} "
              f"({stats['step_hall_count'] / n_s:.1%})")
        print(f"  cumulative_hallucination=True: {stats['cum_hall_count']} "
              f"({stats['cum_hall_count'] / n_s:.1%})")

    # --- is_correct conflicts ---
    print(f"\n[3] is_correct vs. Final Cumulative Consistency")
    conflicts = c["is_correct_conflicts"]
    if conflicts:
        print(f"  WARN — {len(conflicts)} conflict(s) ({len(conflicts) / n_d:.1%}):")
        for cf in conflicts[:10]:
            print(f"    id={cf['id']}  is_correct={cf['is_correct']}  "
                  f"final_cum={cf['final_cum']}")
        if len(conflicts) > 10:
            print(f"    ... and {len(conflicts) - 10} more")
    else:
        print("  PASS — no conflicts")

    # --- Recovery & epiphany ---
    recoveries = c["recovery_cases"]
    epiphanies = c["epiphany_cases"]
    total_transitions = len(recoveries) + len(epiphanies)

    print(f"\n[4] Recovery Analysis  (cum: True -> False)")
    print(f"  Normal recoveries (step=False): {len(recoveries)}")
    print(f"  Epiphany anomalies (step=True) : {len(epiphanies)}")
    if total_transitions:
        print(f"  Normal recovery rate: {len(recoveries) / total_transitions:.1%}")

    if epiphanies:
        severe = [e for e in epiphanies if e.get("is_severe")]
        print(f"  Severe epiphanies (all-True prefix): {len(severe)}")
        if verbose:
            for e in epiphanies[:10]:
                tag = "SEVERE" if e.get("is_severe") else "mild"
                print(f"    [{tag}] id={e['id']} pos={e['position']} last={e['is_last']}")
    if recoveries and verbose:
        print(f"  Recovery cases (first 10):")
        for r in recoveries[:10]:
            print(f"    id={r['id']} pos={r['position']} last={r['is_last']}")

    # --- Distribution ---
    print(f"\n[5] Hallucination Distribution")
    if c["first_step_positions"]:
        print(f"  Avg position of first step_hallucination : "
              f"{_mean(c['first_step_positions']):.1%}")
    if c["first_cum_positions"]:
        print(f"  Avg position of first cumulative_hallucination: "
              f"{_mean(c['first_cum_positions']):.1%}")
    if c["step_hall_densities"]:
        print(f"  Avg step_hallucination density/dialog    : "
              f"{_mean(c['step_hall_densities']):.1%}")
    if c["cum_hall_densities"]:
        print(f"  Avg cumulative_hallucination density/dialog: "
              f"{_mean(c['cum_hall_densities']):.1%}")

    # --- Transition summary ---
    transitions = c["cum_transitions"]
    f2t = [t for t in transitions if t["type"] == "F->T"]
    t2f = [t for t in transitions if t["type"] == "T->F"]
    print(f"\n[6] Cumulative Transition Patterns")
    print(f"  Total transitions : {len(transitions)}")
    print(f"  F->T (enter hallucination): {len(f2t)}"
          + (f"  avg pos {_mean([t['normalized_pos'] for t in f2t]):.1%}" if f2t else ""))
    print(f"  T->F (recovery)           : {len(t2f)}"
          + (f"  avg pos {_mean([t['normalized_pos'] for t in t2f]):.1%}" if t2f else ""))

    # --- Summary ---
    issues = len(completeness_issues) + len(conflicts) + len(epiphanies)
    print("\n" + "=" * 70)
    if issues == 0:
        print("RESULT: PASS — no issues found")
    else:
        print(f"RESULT: {issues} issue(s) found")
        print(f"  Missing tags      : {len(completeness_issues)}")
        print(f"  is_correct conflicts: {len(conflicts)}")
        print(f"  Epiphany anomalies: {len(epiphanies)}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate hallucination tag quality for CHAINED dataset"
    )
    parser.add_argument("input", type=Path, help="Tagged dataset JSON file")
    parser.add_argument("--verbose", action="store_true",
                        help="Print individual case details")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Error: expected a JSON array", file=sys.stderr)
        sys.exit(1)

    completeness_issues = check_completeness(data)
    consistency = check_consistency(data)
    print_report(completeness_issues, consistency, args.verbose)


if __name__ == "__main__":
    main()
