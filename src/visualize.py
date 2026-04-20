"""
Visualize prefix-level hallucination probe predictions along reasoning trajectories.

Loads the teacher (step-level), baseline, and prefix probes and plots dual-panel
confidence curves for a sample of CoT sequences from the test set.  Each panel
overlays the student probe's cumulative estimate against the teacher's step-level
alarm, with ground-truth hallucination labels colour-coded on every marker.

Output: one PNG per selected CoT, saved to --output-dir.

Usage:
    python visualize.py \\
        --test      features/Llama_test.pt \\
        --teacher   models/Llama_teacher.pth \\
        --baseline  models/Llama_baseline.pth \\
        --prefix    models/Llama_prefix_probe.pth \\
        --output-dir results/viz
"""

import argparse
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import FancyArrowPatch, Patch
from sklearn.preprocessing import Normalizer

HIDDEN_DIM  = 4096
IDX_TEACHER = 1   # step_time_exp
IDX_STUDENT = 4   # global_exp  (must match train.py)


class LinearProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        return self.linear(x)


def _load_model(path: str, device) -> LinearProbe:
    if not os.path.exists(path):
        print(f"Error: model not found at {path}", file=sys.stderr)
        sys.exit(1)
    m = LinearProbe().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


class _CoT:
    def __init__(self, x_teacher, x_student, y_step, y_cum):
        self.x_teacher = x_teacher
        self.x_student = x_student
        self.y_step    = y_step
        self.y_cum     = y_cum
        self.length    = len(y_step)


def _load_cots(pt_path: str) -> list[_CoT]:
    data    = torch.load(pt_path, map_location="cpu")
    raw     = data["features"].numpy()
    y_step  = data["labels"].numpy()
    y_cum   = data["cumulative_labels"].numpy()
    pos     = data["positions"].numpy()

    scaler  = Normalizer(norm="l2")
    ft = scaler.fit_transform(raw[:, IDX_TEACHER * HIDDEN_DIM : (IDX_TEACHER + 1) * HIDDEN_DIM])
    fs = scaler.fit_transform(raw[:, IDX_STUDENT * HIDDEN_DIM : (IDX_STUDENT + 1) * HIDDEN_DIM])

    cots, s = [], 0
    for i in range(1, len(pos)):
        if pos[i] <= pos[i - 1]:
            cots.append(_CoT(
                torch.tensor(ft[s:i], dtype=torch.float32),
                torch.tensor(fs[s:i], dtype=torch.float32),
                y_step[s:i], y_cum[s:i],
            ))
            s = i
    cots.append(_CoT(
        torch.tensor(ft[s:], dtype=torch.float32),
        torch.tensor(fs[s:], dtype=torch.float32),
        y_step[s:], y_cum[s:],
    ))
    print(f"Loaded {len(cots)} CoT sequences from {pt_path}")
    return cots


def _plot_panel(ax, steps, p_student, p_teacher, student_label, gt_cum, gt_step, title):
    ax.plot(steps, p_student, marker="o", linewidth=2.5, markersize=8,
            label=student_label, color="#1f77b4", alpha=0.9)
    ax.plot(steps, p_teacher, marker="s", linewidth=2.0, markersize=6,
            label="Teacher (Step Signal)", color="#ff7f0e", alpha=0.6, linestyle="--")

    for i, s in enumerate(steps):
        ax.add_patch(FancyArrowPatch(
            (s, float(p_teacher[i])), (s, float(p_student[i])),
            arrowstyle="->", color="gray", alpha=0.3, linewidth=1, mutation_scale=10,
        ))

    for i, (s, lbl) in enumerate(zip(steps, gt_cum)):
        ax.scatter(s, p_student[i], c="#1f77b4", s=150,
                   edgecolors="red" if lbl else "green", linewidths=2.5, zorder=5, alpha=0.9)
    for i, (s, lbl) in enumerate(zip(steps, gt_step)):
        ax.scatter(s, p_teacher[i], c="#ff7f0e", s=100,
                   edgecolors="red" if lbl else "green", linewidths=2.0, zorder=4,
                   alpha=0.7, marker="s")

    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("Step Index")
    ax.set_ylabel("Probability")


def _visualize(cot: _CoT, teacher, baseline, prefix, save_path: str, idx: int):
    device = next(teacher.parameters()).device
    with torch.no_grad():
        p_t = torch.sigmoid(teacher(cot.x_teacher.to(device))).cpu().numpy().flatten()
        p_b = torch.sigmoid(baseline(cot.x_student.to(device))).cpu().numpy().flatten()
        p_p = torch.sigmoid(prefix(cot.x_student.to(device))).cpu().numpy().flatten()

    steps = np.arange(cot.length)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    _plot_panel(ax1, steps, p_b, p_t, "Baseline Probe (Cum)",
                cot.y_cum, cot.y_step,
                f"Case {idx}: Baseline vs Teacher")
    _plot_panel(ax2, steps, p_p, p_t, "Prefix Probe (Cum)",
                cot.y_cum, cot.y_step,
                f"Case {idx}: Prefix Probe vs Teacher")

    legend_elements = [
        Patch(facecolor="#1f77b4", label="Student Probe (Cum)"),
        Patch(facecolor="#ff7f0e", label="Teacher Probe (Step)"),
        Patch(facecolor="white", edgecolor="red",   linewidth=2, label="GT: Hallucination"),
        Patch(facecolor="white", edgecolor="green", linewidth=2, label="GT: Faithful"),
    ]
    fig.legend(handles=legend_elements, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize prefix probe predictions vs teacher along CoT trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--test",        required=True, help="Test .pt file (from extract_features.py)")
    parser.add_argument("--teacher",     required=True, help="Teacher probe .pth")
    parser.add_argument("--baseline",    required=True, help="Baseline probe .pth (saved during train.py)")
    parser.add_argument("--prefix",      required=True, help="Prefix probe .pth (saved by train.py)")
    parser.add_argument("--output-dir",  default="results/viz", help="Directory to save PNG files")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of CoTs to visualize")
    parser.add_argument("--min-length",  type=int, default=25, help="Minimum CoT length to consider")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher  = _load_model(args.teacher,  device)
    baseline = _load_model(args.baseline, device)
    prefix   = _load_model(args.prefix,   device)

    all_cots = _load_cots(args.test)

    interesting = [
        c for c in all_cots
        if c.length >= args.min_length and c.y_cum.sum() > 0 and c.y_cum[0] == 0
    ]
    if not interesting:
        print("No hallucination-onset CoTs found; falling back to any long CoT.")
        interesting = [c for c in all_cots if c.length >= args.min_length]
    if not interesting:
        print("No CoTs meet the minimum length requirement.", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    selected = random.sample(interesting, min(args.num_samples, len(interesting)))
    print(f"Plotting {len(selected)} CoTs ...")

    os.makedirs(args.output_dir, exist_ok=True)
    for i, cot in enumerate(selected):
        _visualize(cot, teacher, baseline, prefix,
                   os.path.join(args.output_dir, f"compare_case_{i}.png"), i)


if __name__ == "__main__":
    main()
