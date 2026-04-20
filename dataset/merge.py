"""
Merge and post-process generated CoT data for CHAINED dataset construction.

This script handles two distinct stages of the pipeline via subcommands:

  merge   -- Combine multiple per-model JSON files (e.g. one for MuSiQue, one for
             BBH), filter out entries with empty answers, and re-index all IDs
             from 0.  Run this before tagging.

  inject  -- After tagging, inject step-level hallucination labels from the tag
             file back into the original data file, and drop any entries whose
             tagging failed (either flagged automatically or listed manually).

Usage:

  # Step 1: merge BBH + MuSiQue outputs, filter, re-index
  python merge.py merge \\
      --inputs bbh_output.json MuSiQue_output.json \\
      --output CHAINED_raw.json

  # Step 2: inject tags after running tag.py
  python merge.py inject \\
      --data    CHAINED_raw.json \\
      --tags    CHAINED_raw_tags.json \\
      --output  CHAINED_with_tag.json \\
      --remove  310 473 1333
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print(f"Error: expected a JSON array in {path}", file=sys.stderr)
        sys.exit(1)
    return data


def save_json(data: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} entries -> {path}")


# ---------------------------------------------------------------------------
# merge subcommand
# ---------------------------------------------------------------------------

def cmd_merge(args: argparse.Namespace) -> None:
    """Combine input files, drop empty-answer entries, re-index IDs."""
    combined: list[dict[str, Any]] = []

    for input_path in args.inputs:
        p = Path(input_path)
        data = load_json(p)
        combined.extend(data)
        print(f"Loaded {len(data):>5} entries from {p.name}")

    print(f"\nTotal before filtering: {len(combined)}")

    # Filter out entries where answer is missing or blank
    filtered = [
        item for item in combined
        if item.get("answer") and str(item["answer"]).strip()
    ]
    removed = len(combined) - len(filtered)
    print(f"Removed {removed} entries with empty/missing answer")
    print(f"Remaining: {len(filtered)}")

    # Re-index IDs starting from 0, keeping id as the first field
    reindexed = [{"id": i, **{k: v for k, v in item.items() if k != "id"}}
                 for i, item in enumerate(filtered)]

    save_json(reindexed, Path(args.output))


# ---------------------------------------------------------------------------
# inject subcommand
# ---------------------------------------------------------------------------

def _build_tag_map(
    tag_data: list[dict[str, Any]],
    ids_to_remove: list[int],
) -> dict[tuple[int, int], dict[str, bool]]:
    """
    Build a (item_id, step_id) -> {step_hallucination, cumulative_hallucination}
    lookup from the tag file.  Entries marked as tagging failures are added to
    ids_to_remove automatically.
    """
    tag_map: dict[tuple[int, int], dict[str, bool]] = {}
    auto_failed = 0

    for item in tag_data:
        item_id = item.get("id")
        if item.get("error") == "Tagging failed":
            if item_id is not None:
                ids_to_remove.append(int(item_id))
                auto_failed += 1
            continue

        for step in item.get("steps_with_tags", []):
            try:
                key = (int(item_id), int(step["step_id"]))
                tag_map[key] = {
                    "step_hallucination": step["step_hallucination"],
                    "cumulative_hallucination": step["cumulative_hallucination"],
                }
            except (KeyError, TypeError, ValueError):
                continue

    print(f"Tag map built: {len(tag_map)} step-level labels")
    if auto_failed:
        print(f"Auto-detected {auto_failed} tagging failures -> will be removed")
    return tag_map


def _inject_tags(
    data: list[dict[str, Any]],
    tag_map: dict[tuple[int, int], dict[str, bool]],
    ids_to_remove: set[int],
) -> list[dict[str, Any]]:
    """Inject hallucination tags into each step and drop excluded entries."""
    merged_steps = 0

    for item in data:
        try:
            item_id = int(item["id"])
        except (KeyError, TypeError, ValueError):
            continue

        for step in item.get("steps", []):
            try:
                key = (item_id, int(step["step_id"]))
            except (KeyError, TypeError, ValueError):
                continue

            if key in tag_map:
                step["step_hallucination"] = tag_map[key]["step_hallucination"]
                step["cumulative_hallucination"] = tag_map[key]["cumulative_hallucination"]
                merged_steps += 1

    print(f"Injected labels into {merged_steps} steps")

    before = len(data)
    data = [item for item in data if int(item.get("id", -1)) not in ids_to_remove]
    print(f"Removed {before - len(data)} entries (tagging failures + manual list)")
    return data


def cmd_inject(args: argparse.Namespace) -> None:
    """Inject hallucination tags from tag file into data file."""
    data = load_json(Path(args.data))
    tag_data = load_json(Path(args.tags))

    # Manual remove list from CLI, auto-failures appended inside _build_tag_map
    ids_to_remove: list[int] = list(args.remove) if args.remove else []
    tag_map = _build_tag_map(tag_data, ids_to_remove)

    result = _inject_tags(data, tag_map, set(ids_to_remove))
    save_json(result, Path(args.output))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge and post-process CHAINED dataset files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # -- merge --
    p_merge = sub.add_parser("merge", help="Combine input files, filter, re-index")
    p_merge.add_argument(
        "--inputs", nargs="+", required=True,
        help="One or more generated JSON files to combine (e.g. bbh.json musique.json)",
    )
    p_merge.add_argument("--output", required=True, help="Output merged JSON file")

    # -- inject --
    p_inject = sub.add_parser("inject", help="Inject hallucination tags into data file")
    p_inject.add_argument("--data",   required=True, help="Data JSON (output of 'merge')")
    p_inject.add_argument("--tags",   required=True, help="Tag JSON (output of tag.py)")
    p_inject.add_argument("--output", required=True, help="Output JSON with tags injected")
    p_inject.add_argument(
        "--remove", nargs="*", type=int, default=[],
        metavar="ID",
        help="Additional entry IDs to remove (on top of auto-detected failures)",
    )

    args = parser.parse_args()
    if args.cmd == "merge":
        cmd_merge(args)
    else:
        cmd_inject(args)


if __name__ == "__main__":
    main()
