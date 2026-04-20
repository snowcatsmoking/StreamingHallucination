#!/usr/bin/env python3
"""
Compare two hallucination-annotated datasets and report agreement metrics.

Used to validate CHAINED tags by comparing the original Claude-annotated dataset
against a GPT-4o re-annotation of a 1000-sample subset.

Usage:
    python compare_datasets.py <new_dataset.json> <reference_dataset.json> \\
        [--output report.json] \\
        [--new-label GPT-4o] [--reference-label Claude]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two hallucination-annotated datasets")
    parser.add_argument("new_dataset",       type=Path, help="New annotation dataset (e.g. GPT re-annotation)")
    parser.add_argument("reference_dataset", type=Path, help="Reference dataset (original annotation)")
    parser.add_argument("--output",          type=Path, default=None, help="Save JSON report to this path")
    parser.add_argument("--max-examples",    type=int,  default=20,   help="Max disagreement examples per category")
    parser.add_argument("--new-label",       type=str,  default="New",       help="Label for new dataset in output")
    parser.add_argument("--reference-label", type=str,  default="Reference", help="Label for reference dataset in output")
    return parser.parse_args()


def _load(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _index(dialogs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(d["id"]): d for d in dialogs}


def compare_datasets(
    new_dialogs: list[dict[str, Any]],
    ref_dialogs: list[dict[str, Any]],
    max_examples: int = 20,
) -> dict[str, Any]:
    new_idx = _index(new_dialogs)
    ref_idx = _index(ref_dialogs)
    shared  = sorted(set(new_idx) & set(ref_idx))

    dialog_counts = {
        "new_total":       len(new_dialogs),
        "reference_total": len(ref_dialogs),
        "overlap":         len(shared),
        "only_in_new":     sorted(set(new_idx) - set(ref_idx)),
        "only_in_reference": sorted(set(ref_idx) - set(new_idx)),
    }

    step_counts = {
        "new_total_steps":           sum(len(d["steps_with_tags"]) for d in new_dialogs),
        "reference_total_steps":     sum(len(d["steps_with_tags"]) for d in ref_dialogs),
        "comparable_steps":          0,
        "steps_missing_in_new":      0,
        "steps_missing_in_reference": 0,
    }

    agreement = {
        "step_hallucination":       {"matches": 0, "total": 0},
        "cumulative_hallucination": {"matches": 0, "total": 0},
        "is_correct":               {"matches": 0, "total": len(shared)},
    }

    exact_match_dialogs = {
        "step_hallucination": 0,
        "cumulative_hallucination": 0,
        "all_tags": 0,
    }

    examples: dict[str, list] = {
        "step_hallucination": [],
        "cumulative_hallucination": [],
        "is_correct": [],
    }

    for did in shared:
        nd = new_idx[did]
        rd = ref_idx[did]

        new_steps = {s["step_id"]: s for s in nd["steps_with_tags"]}
        ref_steps = {s["step_id"]: s for s in rd["steps_with_tags"]}
        shared_steps = sorted(set(new_steps) & set(ref_steps))

        step_counts["steps_missing_in_new"]      += len(ref_steps) - len(shared_steps)
        step_counts["steps_missing_in_reference"] += len(new_steps) - len(shared_steps)
        step_counts["comparable_steps"]           += len(shared_steps)

        step_match = True
        cum_match  = True

        for sid in shared_steps:
            ns, rs = new_steps[sid], ref_steps[sid]

            agreement["step_hallucination"]["total"] += 1
            if ns["step_hallucination"] == rs["step_hallucination"]:
                agreement["step_hallucination"]["matches"] += 1
            else:
                step_match = False
                if len(examples["step_hallucination"]) < max_examples:
                    examples["step_hallucination"].append(
                        {"dialog_id": did, "step_id": sid,
                         "new": ns["step_hallucination"], "reference": rs["step_hallucination"]}
                    )

            agreement["cumulative_hallucination"]["total"] += 1
            if ns["cumulative_hallucination"] == rs["cumulative_hallucination"]:
                agreement["cumulative_hallucination"]["matches"] += 1
            else:
                cum_match = False
                if len(examples["cumulative_hallucination"]) < max_examples:
                    examples["cumulative_hallucination"].append(
                        {"dialog_id": did, "step_id": sid,
                         "new": ns["cumulative_hallucination"], "reference": rs["cumulative_hallucination"]}
                    )

        if step_match:
            exact_match_dialogs["step_hallucination"] += 1
        if cum_match:
            exact_match_dialogs["cumulative_hallucination"] += 1

        ic_match = nd.get("is_correct") == rd.get("is_correct")
        if ic_match:
            agreement["is_correct"]["matches"] += 1
        elif len(examples["is_correct"]) < max_examples:
            examples["is_correct"].append(
                {"dialog_id": did, "new": nd.get("is_correct"), "reference": rd.get("is_correct")}
            )

        if step_match and cum_match and ic_match:
            exact_match_dialogs["all_tags"] += 1

    for v in agreement.values():
        v["rate"] = (v["matches"] / v["total"]) if v["total"] else None

    return {
        "dialog_counts":        dialog_counts,
        "step_counts":          step_counts,
        "agreements":           agreement,
        "exact_match_dialogs":  exact_match_dialogs,
        "examples":             examples,
    }


def print_summary(report: dict[str, Any], new_label: str, ref_label: str) -> None:
    dc  = report["dialog_counts"]
    sc  = report["step_counts"]
    ag  = report["agreements"]
    ex  = report["exact_match_dialogs"]

    def fmt(item):
        if item["rate"] is None:
            return "n/a"
        return f"{item['rate']:.2%} ({item['matches']}/{item['total']})"

    print("=== Dialog coverage ===")
    print(f"  {ref_label}: {dc['reference_total']}  |  {new_label}: {dc['new_total']}  |  overlap: {dc['overlap']}")
    print(f"  Only in {ref_label}: {len(dc['only_in_reference'])}  |  only in {new_label}: {len(dc['only_in_new'])}")

    print("\n=== Step coverage ===")
    print(f"  Comparable steps: {sc['comparable_steps']}")
    print(f"  Missing in {new_label}: {sc['steps_missing_in_new']}  |  missing in {ref_label}: {sc['steps_missing_in_reference']}")

    print("\n=== Agreement rates ===")
    print(f"  step_hallucination:       {fmt(ag['step_hallucination'])}")
    print(f"  cumulative_hallucination: {fmt(ag['cumulative_hallucination'])}")
    print(f"  is_correct:               {fmt(ag['is_correct'])}")

    print("\n=== Exact-match dialogs ===")
    print(f"  All step tags match:       {ex['step_hallucination']}")
    print(f"  All cumulative tags match: {ex['cumulative_hallucination']}")
    print(f"  All tags match:            {ex['all_tags']}")


def main() -> None:
    args = parse_args()
    new_dialogs = _load(args.new_dataset)
    ref_dialogs = _load(args.reference_dataset)

    report = compare_datasets(new_dialogs, ref_dialogs, max_examples=args.max_examples)
    print_summary(report, args.new_label, args.reference_label)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
