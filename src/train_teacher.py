"""
Train the step-level hallucination probe (Teacher).

Trains one LinearProbe per feature type using LBFGS, evaluates each on the test
set, and saves the step_time_exp probe as the teacher for the prefix-level stage.

The step_time_exp feature corresponds to the time-exponential weighted aggregation
of token hidden states within a step (Section 3 of the paper), which serves as the
local alarm signal c_t^step.

Usage:
    python train_teacher.py \\
        --train  features/Llama_train.pt \\
        --test   features/Llama_test.pt \\
        --output models/Llama_teacher.pth
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import Normalizer

HIDDEN_DIM = 4096
SEED = 42

FEATURE_MAP = {
    "step_mean":    0,
    "step_time_exp": 1,
    "global_mean":  2,
    "global_linear": 3,
    "global_exp":   4,
}


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StepDataset:
    def __init__(self, data: dict, feature_idx: int):
        feats  = data["features"]
        labels = data["labels"]
        start  = feature_idx * HIDDEN_DIM
        raw    = feats[:, start : start + HIDDEN_DIM].float().numpy()
        X      = torch.from_numpy(Normalizer(norm="l2").fit_transform(raw))
        self.X = X
        self.y = labels


class LinearProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        return self.linear(x)


def evaluate(model, X, y_true_np, device):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X.to(device))).cpu().numpy().flatten()
    preds = (probs > 0.5).astype(int)
    if len(np.unique(y_true_np)) < 2:
        return dict(auc=0., acc=0., f1=0., prec=0., rec=0.)
    return dict(
        auc  = roc_auc_score(y_true_np, probs),
        acc  = accuracy_score(y_true_np, preds),
        f1   = f1_score(y_true_np, preds, zero_division=0),
        prec = precision_score(y_true_np, preds, zero_division=0),
        rec  = recall_score(y_true_np, preds, zero_division=0),
    )


def train_probe(X_tr, y_tr, device):
    model = LinearProbe().to(device)
    X_tr  = X_tr.to(device)
    y_tr  = y_tr.to(device).unsqueeze(1)
    crit  = nn.BCEWithLogitsLoss()
    opt   = optim.LBFGS(model.parameters(), lr=1.0, max_iter=100,
                        line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss  = crit(model(X_tr), y_tr)
        loss += 1e-4 * sum(0.5 * torch.norm(p) ** 2 for p in model.parameters())
        loss.backward()
        return loss

    opt.step(closure)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train step-level hallucination probe (Teacher)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train",  required=True, help="Train .pt file")
    parser.add_argument("--test",   required=True, help="Test .pt file")
    parser.add_argument("--output", required=True, help="Path to save teacher .pth")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {args.train} ...")
    raw_train = torch.load(args.train, map_location="cpu")
    print(f"Loading {args.test} ...")
    raw_test  = torch.load(args.test,  map_location="cpu")

    pos_test   = raw_test.get("positions", torch.zeros(len(raw_test["labels"]))).numpy()
    mask_final = pos_test > 0.999
    print(f"Test steps: {len(pos_test)}  |  Final steps: {mask_final.sum()}")

    # Header
    sep = "=" * 220
    hdr = (f"{'Feature':<16} | {'Overall AUC/ACC/F1/P/R':<38} | "
           f"{'Final AUC/ACC/F1/P/R':<38} | "
           f"{'Early 1/3':<38} | {'Mid 1/3':<38} | {'Late 1/3':<38} | {'Saved'}")
    print(sep)
    print(hdr)
    print("-" * 220)

    fmt5 = "{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}"

    for feat_name in FEATURE_MAP:
        set_seed(SEED)
        feat_idx = FEATURE_MAP[feat_name]

        ds_tr = StepDataset(raw_train, feat_idx)
        ds_te = StepDataset(raw_test,  feat_idx)

        model   = train_probe(ds_tr.X, ds_tr.y, device)
        y_te_np = ds_te.y.numpy()
        X_te    = ds_te.X

        overall = evaluate(model, X_te, y_te_np, device)

        # Final-step metrics
        if mask_final.sum() > 0:
            final_m = evaluate(model, X_te[mask_final], y_te_np[mask_final], device)
        else:
            final_m = dict(auc=0., acc=0., f1=0., prec=0., rec=0.)

        # Position-segment metrics
        seg_masks = [
            pos_test < 0.334,
            (pos_test >= 0.334) & (pos_test < 0.667),
            pos_test >= 0.667,
        ]
        segs = []
        for m in seg_masks:
            if m.sum() > 1:
                segs.append(evaluate(model, X_te[m], y_te_np[m], device))
            else:
                segs.append(dict(auc=0., acc=0., f1=0., prec=0., rec=0.))

        # Save teacher
        saved = "—"
        if feat_name == "step_time_exp":
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            torch.save(model.state_dict(), args.output)
            saved = f"Saved -> {args.output}"

        def _fmt(d):
            return fmt5.format(d["auc"], d["acc"], d["f1"], d["prec"], d["rec"])

        print(f"{feat_name:<16} | {_fmt(overall):<38} | {_fmt(final_m):<38} | "
              f"{_fmt(segs[0]):<38} | {_fmt(segs[1]):<38} | {_fmt(segs[2]):<38} | {saved}")

    print("-" * 220)


if __name__ == "__main__":
    main()
