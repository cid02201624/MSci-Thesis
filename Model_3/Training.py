"""
Author: Saskia Knight
Date: 26/02/2026
Description: Training loop for the two-head ResNet model, with time features. This is the main training loop 
that will be used to train the model on the dataset. It includes loading the data, setting up the model and 
optimiser, and running the training and evaluation loops for a specified number of epochs.
"""

import math, sys, os, json, time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torch.amp import autocast, GradScaler



from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)

import matplotlib
matplotlib.use("Agg")  # IMPORTANT on HPC / no display
import matplotlib.pyplot as plt


# Go up 2 levels: Model_2 → Data_Generation 
from pathlib import Path

# PROJECT_ROOT = Path.cwd().parents[1]
# MSCI_BACKUP = PROJECT_ROOT / "MSci-Backup"
# sys.path.insert(0, str(MSCI_BACKUP))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Personal Modules
# import Training_Data_Generation.Processing
# import importlib
# importlib.reload(Training_Data_Generation.Processing)

from Training_Data_Generation.Processing import PrecomputedPTShardDataset


# MODEL
class TwoHeadResNetWithTime(nn.Module):
    """
    Inputs:
      X:      [B, 2, F, T]    (H1/L1 q-transform images as 2 channels)
      t_feat: [B, 2]          (sin/cos of time-of-day; optionally add more)

    Outputs:
      logit_signal: [B]  (binary: noise vs signal)
      logit_gw:     [B]  (binary: glitch vs gw)  -- only meaningful when signal=1
    """
    def __init__(self, time_dim=2, backbone_out=512, time_hidden=32, head_hidden=128, pretrained=False):
        super().__init__()

        # ResNet18 backbone
        self.backbone = resnet18(weights=None if not pretrained else "DEFAULT")

        # Modify first conv to accept 2 channels instead of 3
        old_conv = self.backbone.conv1
        new_conv = nn.Conv2d(
            in_channels=2,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        # If using pretrained weights, initialise the 2-channel conv from pretrained conv1
        if pretrained:
            with torch.no_grad():
                # old_conv.weight shape: [out_c, 3, k, k]
                # Use mean over RGB and repeat to 2 channels
                w = old_conv.weight.data.mean(dim=1, keepdim=True)  # [out_c,1,k,k]
                new_conv.weight.copy_(w.repeat(1, 2, 1, 1))

        self.backbone.conv1 = new_conv

        # Replace classifier with identity, we’ll use pooled features
        self.backbone.fc = nn.Identity()  # output will be [B, 512]

        # Time embedding
        self.time_net = nn.Sequential(
            nn.Linear(time_dim, time_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(time_hidden, time_hidden),
            nn.ReLU(inplace=True),
        )

        feat_dim = backbone_out + time_hidden

        # Head A: noise vs signal
        self.head_signal = nn.Sequential(
            nn.Linear(feat_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(head_hidden, 1),   # logits
        )

        # Head B: glitch vs gw (conditioned on signal)
        self.head_gw = nn.Sequential(
            nn.Linear(feat_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(head_hidden, 1),   # logits
        )

    def forward(self, X, t_feat):
        img_feat = self.backbone(X)              # [B, 512]
        time_feat = self.time_net(t_feat)        # [B, time_hidden]
        feat = torch.cat([img_feat, time_feat], dim=1)

        logit_signal = self.head_signal(feat).squeeze(1)  # [B]
        logit_gw     = self.head_gw(feat).squeeze(1)      # [B]
        return logit_signal, logit_gw
    



# LOSS FUNCTIONS AND METRICS
def hierarchical_losses_and_metrics(
    logit_signal, logit_gw, y,
    lambda_gw=1.0,
    pos_weight_signal=None,
    pos_weight_gw=None
):
    """
    y: LongTensor [B] with values {0=noise, 1=glitch, 2=gw}
    """

    device = y.device

    # targets
    target_signal = (y != 0).float()   # 1 for {glitch,gw}, 0 for noise
    target_gw = (y == 2).float()       # 1 for gw, 0 for glitch (undefined for noise)

    # losses
    bce_signal = nn.BCEWithLogitsLoss(pos_weight=pos_weight_signal.to(device) if pos_weight_signal is not None else None)
    loss_signal = bce_signal(logit_signal, target_signal)

    # mask for signal examples only
    sig_mask = (y != 0)
    if sig_mask.any():
        bce_gw = nn.BCEWithLogitsLoss(pos_weight=pos_weight_gw.to(device) if pos_weight_gw is not None else None)
        loss_gw = bce_gw(logit_gw[sig_mask], target_gw[sig_mask])
    else:
        loss_gw = torch.zeros((), device=device)

    loss = loss_signal + lambda_gw * loss_gw

    # metrics
    with torch.no_grad():
        pred_signal = (torch.sigmoid(logit_signal) > 0.5).long()  # 0/1
        acc_signal = (pred_signal == target_signal.long()).float().mean()

        if sig_mask.any():
            pred_gw = (torch.sigmoid(logit_gw[sig_mask]) > 0.5).long()
            acc_gw = (pred_gw == target_gw[sig_mask].long()).float().mean()
        else:
            acc_gw = torch.tensor(float("nan"), device=device)

    metrics = {
        "loss_total": loss.detach(),
        "loss_signal": loss_signal.detach(),
        "loss_gw": loss_gw.detach(),
        "acc_signal": acc_signal.detach(),
        "acc_gw_on_signal": acc_gw.detach(),
        "n_signal": sig_mask.sum().detach(),
    }
    return loss, metrics


# TRAINING AND EVALUATION LOOPS

def seed_everything(seed: int = 1234):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for reproducible kernels (can reduce speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def three_class_from_heads(logit_signal, logit_gw, thresh_signal=0.5, thresh_gw=0.5):
    """
    Convert 2-head outputs to {0,1,2}:
      if signal_head says "noise" -> 0
      else -> glitch/gw based on gw_head
    """
    p_sig = torch.sigmoid(logit_signal)
    p_gw = torch.sigmoid(logit_gw)

    is_signal = (p_sig > thresh_signal)
    is_gw = (p_gw > thresh_gw)

    yhat = torch.zeros_like(p_sig, dtype=torch.long)
    yhat[is_signal & (~is_gw)] = 1
    yhat[is_signal & is_gw] = 2
    return yhat

def to_model_dtype(x, model, device, non_blocking=False):
    dtype = next(model.parameters()).dtype
    return x.to(device=device, dtype=dtype, non_blocking=non_blocking)


@torch.no_grad()
def evaluate(model, loader, device, return_outputs=False):
    model.eval()
    total_loss = 0.0
    n = 0

    correct3 = 0
    correct_signal = 0
    n_signal = 0
    correct_gw_on_signal = 0

    # Optional collections for plotting (used for final test only)
    if return_outputs:
        all_y = []
        all_yhat3 = []
        all_p_signal = []
        all_p_gw = []

    for X, t_feat, y in loader:
        # X = X.to(device)
        X = to_model_dtype(X, model, device)
        # t_feat = t_feat.to(device)
        t_feat = to_model_dtype(t_feat, model, device)
        y = y.to(device)

        logit_signal, logit_gw = model(X, t_feat)
        loss, metrics = hierarchical_losses_and_metrics(logit_signal, logit_gw, y)

        bs = X.size(0)
        total_loss += float(loss.item()) * bs
        n += bs

        yhat3 = three_class_from_heads(logit_signal, logit_gw)

        correct3 += int((yhat3 == y).sum().item())

        # head metrics
        target_signal = (y != 0)
        pred_signal = (torch.sigmoid(logit_signal) > 0.5)
        correct_signal += int((pred_signal == target_signal).sum().item())

        sig_mask = target_signal
        if sig_mask.any():
            pred_gw = (torch.sigmoid(logit_gw[sig_mask]) > 0.5)
            target_gw = (y[sig_mask] == 2)
            correct_gw_on_signal += int((pred_gw == target_gw).sum().item())
            n_signal += int(sig_mask.sum().item())

        if return_outputs:
            all_y.append(y.detach().cpu())
            all_yhat3.append(yhat3.detach().cpu())
            all_p_signal.append(torch.sigmoid(logit_signal).detach().cpu())
            all_p_gw.append(torch.sigmoid(logit_gw).detach().cpu())

    out = {
        "loss": total_loss / max(1, n),
        "acc_3class": correct3 / max(1, n),
        "acc_signal_head": correct_signal / max(1, n),
        "acc_gw_head_on_signal": (correct_gw_on_signal / max(1, n_signal)) if n_signal > 0 else float("nan"),
        "n": n,
        "n_signal": n_signal,
    }

    if return_outputs:
        out["outputs"] = {
            "y_true": torch.cat(all_y).numpy(),
            "y_pred3": torch.cat(all_yhat3).numpy(),
            "p_signal": torch.cat(all_p_signal).numpy(),
            "p_gw": torch.cat(all_p_gw).numpy(),
        }

    return out


def save_checkpoint(path, model, optimiser, epoch, best_val):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
        },
        path,
    )


def load_checkpoint(path, model, optimiser=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimiser is not None and "optimiser" in ckpt:
        optimiser.load_state_dict(ckpt["optimiser"])
    return ckpt


# -------------------
# Plotting / reporting helpers
# -------------------

CLASS_NAMES = ["noise", "glitch", "gw"]
PLOT_COLORS = ["cornflowerblue", "slateblue", "darkblue"]


def _to_python_scalar(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def save_history_json(history, out_path):
    serialisable = {}
    for k, v in history.items():
        serialisable[k] = [_to_python_scalar(x) for x in v]
    with open(out_path, "w") as f:
        json.dump(serialisable, f, indent=2)


def plot_val_metric_curves(history, out_dir):
    if len(history["epoch"]) == 0:
        return
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.plot(history["epoch"], history["val_acc3"], label="val acc (3-class)", color=PLOT_COLORS[0])
    plt.plot(history["epoch"], history["val_acc_signal"], label="val acc (signal head)", color=PLOT_COLORS[1])
    plt.plot(history["epoch"], history["val_acc_gw_on_sig"], label="val acc (gw head on signal)", color=PLOT_COLORS[2])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Curves")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "val_accuracy_curves.pdf"), bbox_inches='tight', transparent=True)
    plt.close()


def plot_epoch_time_curve(history, out_dir):
    if len(history["epoch"]) == 0 or "epoch_time_s" not in history:
        return
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.plot(history["epoch"], history["epoch_time_s"], marker="o", color=PLOT_COLORS[0])
    plt.xlabel("Epoch")
    plt.ylabel("Seconds")
    plt.title("Epoch Wall Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "epoch_time.pdf"), bbox_inches='tight', transparent=True)
    plt.close()


def confusion_matrix_counts(y_true, y_pred, n_classes=3):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def plot_confusion_matrix(cm, out_path, class_names=CLASS_NAMES, normalise=False):
    cm_plot = cm.astype(np.float64) if normalise else cm.astype(np.int64)
    title = "Confusion Matrix (Counts)"
    fmt = "d"

    if normalise:
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_plot = cm_plot / row_sums
        title = "Confusion Matrix (Row-Normalised)"
        fmt = ".2f"

    plt.figure(figsize=(6, 5))
    plt.imshow(cm_plot, interpolation="nearest", cmap="PuBu")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=20)
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    thresh = cm_plot.max() / 2.0 if cm_plot.size else 0.0
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            txt = format(cm_plot[i, j], fmt)
            plt.text(j, i, txt,
                     ha="center", va="center",
                     color="white" if cm_plot[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', transparent=True)
    plt.close()


def plot_head_score_histograms(y_true, p_signal, p_gw, out_dir):
    """
    p_signal: model sigmoid output from signal head (noise vs {glitch,gw})
    p_gw:     model sigmoid output from gw head (glitch vs gw), meaningful on signal examples
    """
    os.makedirs(out_dir, exist_ok=True)
    y_true = np.asarray(y_true)
    p_signal = np.asarray(p_signal)
    p_gw = np.asarray(p_gw)

    # Signal head: compare true noise vs true signal (glitch+gw)
    noise_mask = (y_true == 0)
    signal_mask = (y_true != 0)

    plt.figure(figsize=(7, 5))
    if noise_mask.any():
        plt.hist(p_signal[noise_mask], bins=40, alpha=0.6, density=True, label="true noise (y=0)", color=PLOT_COLORS[0])
    if signal_mask.any():
        plt.hist(p_signal[signal_mask], bins=40, alpha=0.6, density=True, label="true signal (y=1 or 2)", color=PLOT_COLORS[1])
    plt.xlabel("p_signal = sigmoid(logit_signal)")
    plt.ylabel("Density")
    plt.title("Signal Head Score Distribution (Test)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "test_signal_head_scores.pdf"), bbox_inches='tight', transparent=True)
    plt.close()

    # GW head: compare true glitch vs true gw, only on signal examples
    glitch_mask = (y_true == 1)
    gw_mask = (y_true == 2)

    plt.figure(figsize=(7, 5))
    if glitch_mask.any():
        plt.hist(p_gw[glitch_mask], bins=40, alpha=0.6, density=True, label="true glitch (y=1)", color=PLOT_COLORS[1])
    if gw_mask.any():
        plt.hist(p_gw[gw_mask], bins=40, alpha=0.6, density=True, label="true gw (y=2)", color=PLOT_COLORS[2])
    plt.xlabel("p_gw = sigmoid(logit_gw)")
    plt.ylabel("Density")
    plt.title("GW Head Score Distribution (Test; signal classes only)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "test_gw_head_scores.pdf"), bbox_inches='tight', transparent=True)
    plt.close()


def plot_per_class_accuracy_bar(cm, out_path, class_names=CLASS_NAMES):
    """
    Per-class accuracy = diag(cm) / row_sum(cm)
    (i.e., recall for each true class)
    """
    cm = np.asarray(cm, dtype=np.int64)
    row_sums = cm.sum(axis=1)
    correct = np.diag(cm)

    acc = np.zeros(len(class_names), dtype=np.float64)
    valid = row_sums > 0
    acc[valid] = correct[valid] / row_sums[valid]

    plt.figure(figsize=(7, 5))
    x = np.arange(len(class_names))
    bars = plt.bar(x, acc, color=PLOT_COLORS[:len(class_names)])

    plt.xticks(x, class_names)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy (Recall)")
    plt.title("Per-Class Accuracy (Test)")
    plt.grid(True, axis="y", alpha=0.3)

    # Annotate with acc and counts
    for i, b in enumerate(bars):
        txt = f"{acc[i]:.3f}\n({int(correct[i])}/{int(row_sums[i])})"
        plt.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.02,
            txt,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', transparent=True)
    plt.close()


def _plot_binary_roc_curve(y_true_bin, y_score, out_path, title):
    """
    y_true_bin: 0/1 array
    y_score: predicted probability for positive class
    """
    y_true_bin = np.asarray(y_true_bin).astype(np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)

    # Need both classes present for ROC/AUC
    if np.unique(y_true_bin).size < 2:
        print(f"[WARN] Skipping ROC plot ({title}) because only one class is present.")
        return None

    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    auc = roc_auc_score(y_true_bin, y_score)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", color=PLOT_COLORS[0])
    plt.plot([0, 1], [0, 1], linestyle="--", alpha=0.7, label="chance", color=PLOT_COLORS[1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', transparent=True)
    plt.close()

    return float(auc)


def _plot_binary_pr_curve(y_true_bin, y_score, out_path, title):
    """
    y_true_bin: 0/1 array
    y_score: predicted probability for positive class
    """
    y_true_bin = np.asarray(y_true_bin).astype(np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)

    # Need both classes present for PR/AP to be meaningful
    if np.unique(y_true_bin).size < 2:
        print(f"[WARN] Skipping PR plot ({title}) because only one class is present.")
        return None

    precision, recall, _ = precision_recall_curve(y_true_bin, y_score)
    ap = average_precision_score(y_true_bin, y_score)
    prevalence = float(y_true_bin.mean())  # chance baseline for PR

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}", color=PLOT_COLORS[0])
    plt.axhline(prevalence, linestyle="--", alpha=0.7, label=f"baseline = {prevalence:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', transparent=True)
    plt.close()

    return float(ap)

def plot_loss_curves(history, out_dir):
    if len(history["epoch"]) == 0:
        return
    os.makedirs(out_dir, exist_ok=True)

    epochs = np.asarray(history["epoch"], dtype=np.int64) + 1  # show 1-based epochs

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], label="train loss", color=PLOT_COLORS[0])
    plt.plot(epochs, history["val_loss"], label="val loss", color=PLOT_COLORS[1])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, "loss_curves.pdf"), bbox_inches="tight", transparent=True)
    plt.close()

def make_test_roc_pr_plots(y_true, p_signal, p_gw, out_dir):
    """
    Create ROC/PR plots for:
      1) signal head: positive = (y != 0)
      2) gw head: positive = (y == 2), evaluated only on signal examples y in {1,2}

    Returns a metrics dict with AUC/AP values where available.
    """
    os.makedirs(out_dir, exist_ok=True)

    y_true = np.asarray(y_true)
    p_signal = np.asarray(p_signal, dtype=np.float64)
    p_gw = np.asarray(p_gw, dtype=np.float64)

    rocpr_metrics = {}

    # -----------------------------------
    # Signal head: noise (0) vs signal (1/2)
    # -----------------------------------
    y_signal_bin = (y_true != 0).astype(np.uint8)

    auc_signal = _plot_binary_roc_curve(
        y_signal_bin,
        p_signal,
        os.path.join(out_dir, "test_signal_head_ROC.pdf"),
        "ROC: Signal Head (noise vs {glitch, gw})",
    )
    ap_signal = _plot_binary_pr_curve(
        y_signal_bin,
        p_signal,
        os.path.join(out_dir, "test_signal_head_PR.pdf"),
        "PR: Signal Head (noise vs {glitch, gw})",
    )

    rocpr_metrics["signal_head_roc_auc"] = auc_signal
    rocpr_metrics["signal_head_average_precision"] = ap_signal

    # -----------------------------------
    # GW head: glitch (1) vs gw (2), only on signal examples
    # -----------------------------------
    sig_mask = (y_true != 0)
    y_sig_only = y_true[sig_mask]
    p_gw_sig_only = p_gw[sig_mask]

    # Positive class = gw (y==2)
    y_gw_bin = (y_sig_only == 2).astype(np.uint8)

    auc_gw = _plot_binary_roc_curve(
        y_gw_bin,
        p_gw_sig_only,
        os.path.join(out_dir, "test_gw_head_on_signal_ROC.pdf"),
        "ROC: GW Head on Signal Examples (glitch vs gw)",
    )
    ap_gw = _plot_binary_pr_curve(
        y_gw_bin,
        p_gw_sig_only,
        os.path.join(out_dir, "test_gw_head_on_signal_PR.pdf"),
        "PR: GW Head on Signal Examples (glitch vs gw)",
    )

    rocpr_metrics["gw_head_on_signal_roc_auc"] = auc_gw
    rocpr_metrics["gw_head_on_signal_average_precision"] = ap_gw

    return rocpr_metrics

# -------------------
# Main training setup
# -------------------
seed_everything(2026)

device = "cuda" if torch.cuda.is_available() else "cpu"

use_amp = (device == "cuda")
scaler = GradScaler("cuda", enabled=use_amp)

# Keep X in saved dtype on CPU (likely float16); cast once per batch on GPU instead
train_ds = PrecomputedPTShardDataset("Training_Data_Generation/pt_dataset/train", cast_x_to_float32=False, max_samples=None)
val_ds   = PrecomputedPTShardDataset("Training_Data_Generation/pt_dataset/val", cast_x_to_float32=False, max_samples=None)
test_ds  = PrecomputedPTShardDataset("Training_Data_Generation/pt_dataset/test", cast_x_to_float32=False, max_samples=None)

# IMPORTANT for shard-backed datasets:
# shuffle=True causes random shard access and many torch.load() calls.
# Start with shuffle=False for speed. (If you later want randomness, do shard-level shuffling.)
train_loader = DataLoader(
    train_ds,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
    prefetch_factor=4,
)

val_loader = DataLoader(
    val_ds,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
    persistent_workers=True,
    prefetch_factor=4,
)

test_loader = DataLoader(
    test_ds,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
    persistent_workers=True,
    prefetch_factor=4,
)

model = TwoHeadResNetWithTime(pretrained=True).to(device)
optimiser = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

use_amp = (device == "cuda")
scaler = GradScaler(enabled=use_amp)

start_epoch = 0
best_val = float("inf")
ckpt_path = "checkpoints3/best.pt"
resume_path = None  # set to "checkpoints/best.pt" to resume

fig_dir = "training_figures3"
os.makedirs(fig_dir, exist_ok=True)

history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "val_acc3": [],
    "val_acc_signal": [],
    "val_acc_gw_on_sig": [],
    "epoch_time_s": [],
}

if resume_path and os.path.exists(resume_path):
    ckpt = load_checkpoint(resume_path, model, optimiser=optimiser, map_location=device)
    start_epoch = int(ckpt.get("epoch", 0) + 1)
    best_val = float(ckpt.get("best_val", best_val))

epochs = int(os.environ.get("EPOCHS", "30"))

for epoch in range(start_epoch, epochs):
    t_epoch0 = time.time()
    model.train()
    running = 0.0
    n = 0

    for X, t_feat, y in train_loader:
        print(f"Epoch {epoch:03d} | Batch {n//train_loader.batch_size:04d}/{len(train_loader):04d}", end="\r")
        X = to_model_dtype(X, model, device, non_blocking=True)
        t_feat = to_model_dtype(t_feat, model, device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=use_amp):
            logit_signal, logit_gw = model(X, t_feat)
            loss, _metrics = hierarchical_losses_and_metrics(logit_signal, logit_gw, y)

        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

        bs = X.size(0)
        running += float(loss.item()) * bs
        n += bs

    train_loss = running / max(1, n)
    val_stats = evaluate(model, val_loader, device)

    epoch_time_s = time.time() - t_epoch0

    history["epoch"].append(epoch)
    history["train_loss"].append(float(train_loss))
    history["val_loss"].append(float(val_stats["loss"]))
    history["val_acc3"].append(float(val_stats["acc_3class"]))
    history["val_acc_signal"].append(float(val_stats["acc_signal_head"]))
    history["val_acc_gw_on_sig"].append(float(val_stats["acc_gw_head_on_signal"]))
    history["epoch_time_s"].append(float(epoch_time_s))

    # overwrite plots each epoch (latest curves)
    plot_loss_curves(history, fig_dir)
    plot_val_metric_curves(history, fig_dir)
    plot_epoch_time_curve(history, fig_dir)
    save_history_json(history, os.path.join(fig_dir, "history.json"))

    print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_stats['loss']:.4f} "
          f"| val_acc3={val_stats['acc_3class']:.3f} | val_signal={val_stats['acc_signal_head']:.3f} "
          f"| val_gw_on_sig={val_stats['acc_gw_head_on_signal']:.3f}")

    # save best
    if val_stats["loss"] < best_val:
        best_val = val_stats["loss"]
        save_checkpoint(ckpt_path, model, optimiser, epoch, best_val)

# final test eval (best checkpoint)
ckpt = load_checkpoint(ckpt_path, model, optimiser=None, map_location=device)
test_stats = evaluate(model, test_loader, device, return_outputs=True)

# Print summary metrics only
test_summary = {k: v for k, v in test_stats.items() if k != "outputs"}
print("TEST:", test_summary)

# Save test metrics json
with open(os.path.join(fig_dir, "test_stats.json"), "w") as f:
    json.dump({k: _to_python_scalar(v) for k, v in test_summary.items()}, f, indent=2)

# Make test plots
outs = test_stats["outputs"]
y_true = outs["y_true"]
y_pred3 = outs["y_pred3"]
p_signal = outs["p_signal"]
p_gw = outs["p_gw"]


cm = confusion_matrix_counts(y_true, y_pred3, n_classes=3)

# Confusion matrices
plot_confusion_matrix(cm, os.path.join(fig_dir, "test_confusion_counts.pdf"), normalise=False)
plot_confusion_matrix(cm, os.path.join(fig_dir, "test_confusion_normalised.pdf"), normalise=True)

# Per-class accuracy bar chart (recall per true class)
plot_per_class_accuracy_bar(cm, os.path.join(fig_dir, "test_per_class_accuracy_bar.pdf"))

# Score histograms
plot_head_score_histograms(y_true, p_signal, p_gw, fig_dir)

# ROC / PR curves
rocpr_metrics = make_test_roc_pr_plots(y_true, p_signal, p_gw, fig_dir)

# Training curves (loss and val acc) over epochs
plot_loss_curves(history, fig_dir)

# Save ROC/PR summary metrics
with open(os.path.join(fig_dir, "test_rocpr_metrics.json"), "w") as f:
    json.dump({k: _to_python_scalar(v) for k, v in rocpr_metrics.items()}, f, indent=2)