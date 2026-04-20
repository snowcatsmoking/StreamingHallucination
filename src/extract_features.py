"""
Extract step-level feature vectors from processed JSONL files for hallucination detection.

Reads the JSONL output of process.py (one CoT entry per line).  For each step the
script computes five 4096-d pooling vectors over the step's token hidden states and
concatenates them into a single 20480-d feature vector:

    [0:4096]     local mean          — uniform mean over step tokens
    [4096:8192]  local time-exp      — exponentially time-weighted mean (e^3x, x in [0,1])
    [8192:12288] global mean         — cumulative uniform mean up to this step
    [12288:16384] global linear      — cumulative linearly-weighted mean (weight = token index)
    [16384:20480] global exp         — cumulative exp-weighted mean (w = exp(0.003 * index))

Global vectors are computed causally: only tokens seen so far (including the current
step) contribute, so the representation captures how reasoning has evolved up to each
decision point.

Output .pt files contain:
    features          [N, 20480]   float32
    labels            [N]          float32   (step_hallucination)
    cumulative_labels [N]          float32   (cumulative_hallucination)
    positions         [N]          float32   (relative step position in CoT, 0-1)

Usage:
    python extract_features.py \\
        --input  processed/Llama_train.jsonl \\
        --output features/Llama_train.pt

    # Process all splits at once:
    python extract_features.py \\
        --input  processed/Llama_train.jsonl processed/Llama_val.jsonl processed/Llama_test.jsonl \\
        --output features/Llama_train.pt     features/Llama_val.pt     features/Llama_test.pt
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import orjson
import torch

HIDDEN_DIM     = 4096
EXP_GROWTH_RATE = 0.003   # e^3 factor at token index ~1000


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _compute_step_features(
    hidden_states: torch.Tensor,
    global_mean:   torch.Tensor,
    global_linear: torch.Tensor,
    global_exp:    torch.Tensor,
) -> torch.Tensor:
    """
    Compute the 20480-d feature vector for one step.
    hidden_states: [L, H]
    global_*:      [H]  cumulative vectors up to and including this step
    """
    L = hidden_states.shape[0]
    device = hidden_states.device

    # 1. Local mean
    feat_local_mean = hidden_states.mean(dim=0)

    # 2. Local time-exp  (e^{3x}, x in [0,1])
    t = torch.linspace(0, 1, steps=L, device=device)
    w = torch.exp(3 * t)
    w = w / w.sum()
    feat_local_time_exp = (hidden_states * w.unsqueeze(1)).sum(dim=0)

    return torch.cat([
        feat_local_mean,    # 0       – 4096
        feat_local_time_exp,# 4096    – 8192
        global_mean,        # 8192    – 12288
        global_linear,      # 12288   – 16384
        global_exp,         # 16384   – 20480
    ])


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        print(f"[skip] not found: {input_path}", file=sys.stderr)
        return

    print(f"Processing {input_path.name} -> {output_path.name}")

    all_features:     list[torch.Tensor] = []
    all_step_labels:  list[float] = []
    all_cum_labels:   list[float] = []
    all_positions:    list[float] = []
    step_count = 0

    with open(input_path, "r", encoding="utf-8", buffering=10 * 1024 * 1024) as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data  = orjson.loads(line)
                steps = data.get("steps", [])
                n     = len(steps)
                if n == 0:
                    continue

                # Per-CoT causal accumulators
                run_sum_std = np.zeros(HIDDEN_DIM, dtype=np.float64)
                run_sum_lin = np.zeros(HIDDEN_DIM, dtype=np.float64)
                run_w_lin   = 0.0
                run_sum_exp = np.zeros(HIDDEN_DIM, dtype=np.float64)
                run_w_exp   = 0.0
                run_tokens  = 0

                for i, step in enumerate(steps):
                    raw_hidden    = step.get("token_hidden_states")
                    raw_step_lbl  = step.get("step_hallucination")
                    raw_cum_lbl   = step.get("cumulative_hallucination")

                    if not raw_hidden:
                        continue

                    L = len(raw_hidden)

                    # Normalise labels (accept bool or "true"/"false" string)
                    def _to_float(v):
                        if isinstance(v, bool):
                            return 1.0 if v else 0.0
                        if isinstance(v, str):
                            return 1.0 if v.lower() == "true" else 0.0
                        return 0.0

                    label_step = _to_float(raw_step_lbl)
                    label_cum  = _to_float(raw_cum_lbl)
                    rel_pos    = (i + 1) / n

                    hidden_np = np.array(raw_hidden, dtype=np.float32)  # [L, H]

                    # Update causal accumulators
                    run_sum_std += hidden_np.sum(axis=0)

                    global_idx = np.arange(run_tokens + 1, run_tokens + L + 1, dtype=np.float64)

                    w_lin = global_idx
                    run_sum_lin += (hidden_np * w_lin[:, None]).sum(axis=0)
                    run_w_lin   += w_lin.sum()

                    w_exp = np.exp(EXP_GROWTH_RATE * global_idx)
                    run_sum_exp += (hidden_np * w_exp[:, None]).sum(axis=0)
                    run_w_exp   += w_exp.sum()

                    run_tokens += L

                    # Instantaneous global vectors
                    g_mean   = (run_sum_std / run_tokens).astype(np.float32)
                    g_linear = (run_sum_lin / run_w_lin).astype(np.float32)
                    g_exp    = (run_sum_exp / run_w_exp).astype(np.float32)

                    feats = _compute_step_features(
                        torch.from_numpy(hidden_np),
                        torch.from_numpy(g_mean),
                        torch.from_numpy(g_linear),
                        torch.from_numpy(g_exp),
                    )

                    all_features.append(feats)
                    all_step_labels.append(label_step)
                    all_cum_labels.append(label_cum)
                    all_positions.append(rel_pos)
                    step_count += 1

            except Exception:
                continue

            if line_idx % 50 == 0:
                print(f"\r  lines={line_idx}  steps={step_count}", end="", flush=True)

    print()
    if step_count == 0:
        print(f"  No valid steps found in {input_path.name}", file=sys.stderr)
        return

    features_tensor  = torch.stack(all_features)
    step_lbl_tensor  = torch.tensor(all_step_labels,  dtype=torch.float32)
    cum_lbl_tensor   = torch.tensor(all_cum_labels,   dtype=torch.float32)
    positions_tensor = torch.tensor(all_positions,    dtype=torch.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features":          features_tensor,
            "labels":            step_lbl_tensor,
            "cumulative_labels": cum_lbl_tensor,
            "positions":         positions_tensor,
        },
        output_path,
    )

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  Saved {output_path}  shape={list(features_tensor.shape)}  "
          f"step_hall={step_lbl_tensor.mean():.3f}  "
          f"cum_hall={cum_lbl_tensor.mean():.3f}  "
          f"{size_mb:.1f} MB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract 20480-d step features from processed JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="One or more processed JSONL files (output of process.py)",
    )
    parser.add_argument(
        "--output", nargs="+", required=True,
        help="Corresponding output .pt file paths (must match --input count)",
    )
    args = parser.parse_args()

    if len(args.input) != len(args.output):
        print("Error: --input and --output must have the same number of entries", file=sys.stderr)
        sys.exit(1)

    for inp, out in zip(args.input, args.output):
        process_file(Path(inp), Path(out))


if __name__ == "__main__":
    main()
