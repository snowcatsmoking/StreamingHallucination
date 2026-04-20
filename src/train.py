"""
Train the prefix-level hallucination probe (Student).

Implements the step-guided prefix-level estimation from Section 4 of the paper.
The model learns to estimate c_t^prefix using global hidden-state representations
(h_t), guided by a frozen step-level teacher probe (c_t^step).

Training objective (Eq. 11):
    L = L_anchor + lambda_sync * L_sync

    L_anchor: weighted BCE over all steps, with extra weight on the final step
              to enforce correct end-state prediction (Eq. 8).

    L_sync:   quadratic alarm synchronisation loss that penalises missed alarms
              when the teacher fires — enforces directional consistency without
              inducing monotonic accumulation (Eq. 9-10).

Must run train_teacher.py first to obtain the teacher checkpoint.

Usage:
    python train.py \\
        --train    features/Llama_train.pt \\
        --test     features/Llama_test.pt \\
        --teacher  models/Llama_teacher.pth \\
        --output   models/Llama_prefix_probe.pth
"""

import argparse
import copy
import gc
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import Normalizer

# ---------------------------------------------------------------------------
# Hyper-parameters (from paper / final experiments)
# ---------------------------------------------------------------------------
HIDDEN_DIM      = 4096
BATCH_SIZE_COT  = 64
EPOCHS          = 20
LR              = 1e-3
SEED            = 42
LAMBDA_FINAL    = 30.0   # anchor weight on the last step (Eq. 8)
LAMBDA_SYNC     = 120.0  # synchronisation loss coefficient (Eq. 11)

FEATURE_MAP = {
    "step_mean":     0,
    "step_time_exp": 1,
    "global_mean":   2,
    "global_linear": 3,
    "global_exp":    4,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CoTDataset:
    """
    Splits the flat feature tensor back into per-CoT sequences using the
    monotonically increasing positions field (position resets at each new CoT).

    x_glob : global_exp features  → input to the prefix probe
    x_step : step_time_exp features → input to the (frozen) teacher probe
    """

    def __init__(self, data: dict, global_feature: str = "global_exp"):
        feats  = data["features"]
        y_step = data["labels"]
        y_cum  = data["cumulative_labels"]
        pos    = data["positions"].numpy()

        scaler = Normalizer(norm="l2")

        g_idx  = FEATURE_MAP[global_feature] * HIDDEN_DIM
        X_glob = torch.from_numpy(
            scaler.fit_transform(feats[:, g_idx : g_idx + HIDDEN_DIM].float().numpy())
        )

        t_idx  = FEATURE_MAP["step_time_exp"] * HIDDEN_DIM
        X_step = torch.from_numpy(
            scaler.fit_transform(feats[:, t_idx : t_idx + HIDDEN_DIM].float().numpy())
        )

        self.cots: list[dict] = []
        start = 0
        for i in range(1, len(pos)):
            if pos[i] <= pos[i - 1]:
                self._add(X_glob, X_step, y_step, y_cum, start, i)
                start = i
        self._add(X_glob, X_step, y_step, y_cum, start, len(pos))
        print(f"  Parsed {len(self.cots)} CoTs from {len(pos)} steps.")

    def _add(self, Xg, Xs, Ys, Yc, s, e):
        if s >= e:
            return
        self.cots.append({
            "x_glob": Xg[s:e],
            "x_step": Xs[s:e],
            "y_step": Ys[s:e],
            "y_cum":  Yc[s:e],
        })

    def flat_baseline_data(self):
        return (
            torch.cat([c["x_glob"] for c in self.cots]),
            torch.cat([c["y_cum"]  for c in self.cots]),
        )


def _cot_batches(cots, batch_size):
    idx = list(range(len(cots)))
    random.shuffle(idx)
    for i in range(0, len(cots), batch_size):
        batch = [cots[j] for j in idx[i : i + batch_size]]
        yield {
            "x_glob":  torch.cat([c["x_glob"] for c in batch]),
            "x_step":  torch.cat([c["x_step"] for c in batch]),
            "y_cum":   torch.cat([c["y_cum"]  for c in batch]),
            "lengths": [len(c["x_glob"]) for c in batch],
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        return self.linear(x)


def load_teacher(path: str, device) -> LinearProbe:
    m = LinearProbe().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_baseline(dataset: CoTDataset, device) -> LinearProbe:
    """LBFGS baseline trained on flat global features with BCE."""
    model = LinearProbe().to(device)
    X, y  = dataset.flat_baseline_data()
    X, y  = X.to(device), y.to(device).unsqueeze(1)
    crit  = nn.BCEWithLogitsLoss()
    opt   = optim.LBFGS(model.parameters(), lr=1.0, max_iter=100)

    def closure():
        opt.zero_grad()
        loss  = crit(model(X), y)
        loss += 1e-4 * sum(0.5 * torch.norm(p) ** 2 for p in model.parameters())
        loss.backward()
        return loss

    opt.step(closure)
    return model


def train_prefix_probe(
    dataset: CoTDataset,
    baseline: LinearProbe,
    teacher: LinearProbe,
    device,
) -> LinearProbe:
    """
    AdamW fine-tuning with anchor loss + quadratic alarm sync loss.
    Initialised from the LBFGS baseline for stability.
    """
    model = copy.deepcopy(baseline)
    opt   = optim.AdamW(model.parameters(), lr=LR)
    crit  = nn.BCEWithLogitsLoss(reduction="none")

    for ep in range(EPOCHS):
        model.train()
        for batch in _cot_batches(dataset.cots, BATCH_SIZE_COT):
            xg   = batch["x_glob"].to(device)
            yc   = batch["y_cum"].to(device).unsqueeze(1)
            xs   = batch["x_step"].to(device)
            lens = batch["lengths"]

            # Up-weight the final step of each CoT (L_anchor, Eq. 8)
            end_idx = torch.tensor(np.cumsum(lens) - 1, device=device)
            weights = torch.ones_like(yc)
            weights[end_idx] = LAMBDA_FINAL

            opt.zero_grad()
            logits = model(xg)
            p_cum  = torch.sigmoid(logits)

            with torch.no_grad():
                p_step = torch.sigmoid(teacher(xs))

            # L_anchor
            loss_anchor = (crit(logits, yc) * weights).mean()

            # L_sync  (Eq. 9-10): one-way quadratic alarm synchronisation
            delta      = torch.relu(p_step - p_cum)          # missed alarms only
            w_alarm    = p_step.pow(2)                        # quadratic gate
            loss_sync  = (delta.pow(2) * w_alarm).sum() / sum(lens)

            loss = loss_anchor + LAMBDA_SYNC * loss_sync
            loss.backward()
            opt.step()

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _calc_dynamic_metrics(gt_cum_list, gt_step_list, pred_list):
    """
    Eight dynamic metrics characterising hallucination onset, recovery, and
    false-positive structure along the reasoning trajectory.
    """
    lag_steps, icr_hits, icr_total = [], 0, 0
    snap_sum, snap_n               = 0.0, 0
    brake_sum, brake_n             = 0.0, 0
    ling_times                     = []
    heal3_hits, heal_total         = 0, 0
    r_auc_preds                    = []
    fp_lengths                     = []

    for yc, ys, pp in zip(gt_cum_list, gt_step_list, pred_list):
        L = len(yc)

        # Reflex: lag & ICR
        if yc.sum() > 0:
            first = int(np.argmax(yc))
            icr_total += 1
            resp = pp[first:] > 0.5
            if resp.any():
                lag = int(np.argmax(resp))
                lag_steps.append(lag)
                if lag == 0:
                    icr_hits += 1
            else:
                lag_steps.append(L - first)
            if first > 0:
                snap_sum += float(pp[first] - pp[first - 1])
                snap_n   += 1

        # Recovery dynamics
        for t in range(1, L):
            if yc[t - 1] == 1 and yc[t] == 0:
                brake_sum += float(pp[t - 1] - pp[t])
                brake_n   += 1
                heal_total += 1
                healed = any(pp[k] < 0.5 for k in range(t, min(L, t + 3)))
                if healed:
                    heal3_hits += 1
                ling = sum(1 for k in range(t, L) if pp[k] > 0.5)
                ling_times.append(ling)

        # Recovery score
        in_hall = False
        for t in range(L):
            if yc[t] == 1:
                in_hall = True
            elif in_hall:
                r_auc_preds.append(float(pp[t]))

        # False-positive run length
        run = 0
        for t in range(L):
            if yc[t] == 0 and pp[t] > 0.5:
                run += 1
            else:
                if run:
                    fp_lengths.append(run)
                run = 0
        if run:
            fp_lengths.append(run)

    def _m(lst): return float(np.mean(lst)) if lst else 0.0
    def _r(n, d): return n / (d + 1e-9)

    return {
        "Lag":     _m(lag_steps),
        "ICR":     _r(icr_hits, icr_total),
        "Snap_M":  _r(snap_sum, snap_n),
        "Brake_S": _r(brake_sum, brake_n),
        "Ling_T":  _m(ling_times),
        "Heal_3":  _r(heal3_hits, heal_total),
        "R_Score": 1.0 - (float(np.mean(r_auc_preds)) if r_auc_preds else 0.5),
        "FP_Len":  _m(fp_lengths),
    }


def evaluate(model: LinearProbe, dataset: CoTDataset, device) -> dict:
    model.eval()
    all_gt_cum, all_gt_step, all_pred = [], [], []
    flat_gt, flat_pred                = [], []
    final_gt, final_pred              = [], []

    with torch.no_grad():
        for c in dataset.cots:
            pp     = torch.sigmoid(model(c["x_glob"].to(device))).cpu().numpy().flatten()
            yc     = c["y_cum"].numpy()
            ys     = c["y_step"].numpy()
            all_gt_cum.append(yc)
            all_gt_step.append(ys)
            all_pred.append(pp)
            flat_gt.extend(yc.tolist())
            flat_pred.extend(pp.tolist())
            if len(pp):
                final_gt.append(float(yc[-1]))
                final_pred.append(float(pp[-1]))

    flat_gt   = np.array(flat_gt)
    flat_pred = np.array(flat_pred)
    pred_bin  = (flat_pred > 0.5).astype(int)

    f_gt   = np.array(final_gt)
    f_pred = np.array(final_pred)
    f_bin  = (f_pred > 0.5).astype(int)

    dynamic = _calc_dynamic_metrics(all_gt_cum, all_gt_step, all_pred)

    return {
        **dynamic,
        "AUC":        roc_auc_score(flat_gt, flat_pred) if len(np.unique(flat_gt)) > 1 else 0.,
        "ACC":        accuracy_score(flat_gt, pred_bin),
        "F1":         f1_score(flat_gt, pred_bin, zero_division=0),
        "Final_AUC":  roc_auc_score(f_gt, f_pred) if len(np.unique(f_gt)) > 1 else 0.,
        "Final_ACC":  accuracy_score(f_gt, f_bin),
        "Final_F1":   f1_score(f_gt, f_bin, zero_division=0),
        "Final_Rec":  recall_score(f_gt, f_bin, zero_division=0),
        "Final_Prec": precision_score(f_gt, f_bin, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train prefix-level hallucination probe (Student)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train",   required=True, help="Train .pt file")
    parser.add_argument("--test",    required=True, help="Test .pt file")
    parser.add_argument("--teacher", required=True, help="Teacher .pth (from train_teacher.py)")
    parser.add_argument("--output",          required=True, help="Path to save prefix probe .pth")
    parser.add_argument("--baseline-output", default=None,  help="Optional path to save baseline probe .pth")
    parser.add_argument("--feature", default="global_exp",
                        choices=list(FEATURE_MAP.keys()),
                        help="Global feature for prefix probe input (default: global_exp)")
    args = parser.parse_args()

    if not os.path.exists(args.teacher):
        print(f"Error: teacher not found at {args.teacher}. Run train_teacher.py first.",
              file=sys.stderr)
        sys.exit(1)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\nLoading features...")
    print(f"  Train: {args.train}")
    ds_tr = CoTDataset(torch.load(args.train, map_location="cpu"), args.feature)
    print(f"  Test:  {args.test}")
    ds_te = CoTDataset(torch.load(args.test,  map_location="cpu"), args.feature)

    teacher = load_teacher(args.teacher, device)
    print(f"Teacher loaded from {args.teacher}")

    print("\n[1/3] Training LBFGS baseline ...")
    baseline = train_baseline(ds_tr, device)
    res_base = evaluate(baseline, ds_te, device)

    print("[2/3] Training prefix probe (anchor + sync loss) ...")
    prefix_probe = train_prefix_probe(ds_tr, baseline, teacher, device)
    res_prefix   = evaluate(prefix_probe, ds_te, device)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(prefix_probe.state_dict(), args.output)
    print(f"[3/3] Prefix probe saved -> {args.output}")
    if args.baseline_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.baseline_output)), exist_ok=True)
        torch.save(baseline.state_dict(), args.baseline_output)
        print(f"      Baseline probe saved -> {args.baseline_output}")

    # Results table
    lower_is_better = {"Lag", "Ling_T", "FP_Len"}
    specs = [
        ("Final",   "Final_AUC"),
        ("Final",   "Final_ACC"),
        ("Final",   "Final_F1"),
        ("Final",   "Final_Rec"),
        ("Final",   "Final_Prec"),
        ("Agility", "Brake_S"),
        ("Agility", "Ling_T"),
        ("Agility", "Heal_3"),
        ("Agility", "R_Score"),
        ("Reflex",  "Snap_M"),
        ("Reflex",  "Lag"),
        ("Reflex",  "ICR"),
        ("Struct",  "FP_Len"),
        ("Overall", "AUC"),
        ("Overall", "ACC"),
        ("Overall", "F1"),
    ]

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"{'Group':<10} {'Metric':<12} {'Baseline':<12} {'PrefixProbe':<12} {'Better'}")
    print("-" * 70)
    for group, key in specs:
        vb = res_base[key]
        vp = res_prefix[key]
        better = "Prefix" if (vp < vb if key in lower_is_better else vp > vb) else "Base"
        print(f"{group:<10} {key:<12} {vb:<12.4f} {vp:<12.4f} {better}")
    print(sep)
    print(f"lambda_final={LAMBDA_FINAL}  lambda_sync={LAMBDA_SYNC}  feature={args.feature}")

    del ds_tr, ds_te, baseline, prefix_probe
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
