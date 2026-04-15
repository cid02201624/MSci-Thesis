"""
Author: Saskia Knight
Date: 26/02/2026
"""

import math, sys, os, json, time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader
try:
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
except ImportError:
    from torchvision.models import convnext_tiny
    ConvNeXt_Tiny_Weights = None
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

from Training_Data_Generation.Processing2 import PrecomputedPTShardDataset

class JointConvNeXtGWWithTime(nn.Module):
    """
    Inputs:
      X:      [B, 2, F, T]   (H1 and L1 as two channels)
      t_feat: [B, 2]         (sin/cos of cyclic 24h GPS time)

    Outputs:
      logit_gw: [B]
    """
    def __init__(
        self,
        time_dim=2,
        time_hidden=64,
        head_hidden=256,
        dropout=0.2,
        pretrained=True,
        use_aux_3class=False,
        resize_hw=None,
    ):
        super().__init__()
        self.use_aux_3class = use_aux_3class
        self.resize_hw = resize_hw

        # IMPORTANT: now use 2 input channels
        self.encoder = ConvNeXtTinySingleViewEncoder(
            pretrained=pretrained,
            in_chans=2,
        )
        enc_dim = self.encoder.out_dim  # usually 768

        self.time_net = nn.Sequential(
            nn.Linear(time_dim, time_hidden),
            nn.GELU(),
            nn.LayerNorm(time_hidden),
            nn.Dropout(dropout),
            nn.Linear(time_hidden, time_hidden),
            nn.GELU(),
            nn.LayerNorm(time_hidden),
        )

        fusion_out_dim = enc_dim + time_hidden

        self.fusion_norm = nn.LayerNorm(fusion_out_dim)
        self.fusion_dropout = nn.Dropout(dropout)

        self.gw_head = nn.Sequential(
            nn.Linear(fusion_out_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

        if self.use_aux_3class:
            self.head_3class = nn.Sequential(
                nn.Linear(fusion_out_dim, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 3),
            )
        else:
            self.head_3class = None

    def forward(self, X, t_feat):
        # X: [B,2,F,T]
        if X.ndim != 4 or X.shape[1] != 2:
            raise ValueError(f"Expected X with shape [B,2,F,T], got {tuple(X.shape)}")

        if self.resize_hw is not None and (X.shape[-2:] != self.resize_hw):
            X = F.interpolate(X, size=self.resize_hw, mode="bilinear", align_corners=False)

        # Joint encoding of H1 and L1
        z = self.encoder(X)       # [B, enc_dim]

        # Time embedding
        t = self.time_net(t_feat) # [B, time_hidden]

        fused = torch.cat([z, t], dim=1)
        fused = self.fusion_norm(fused)
        fused = self.fusion_dropout(fused)

        logit_gw = self.gw_head(fused).squeeze(1)
        logits_3class = self.head_3class(fused) if self.head_3class is not None else None

        return {
            "logit_gw": logit_gw,
            "logits_3class": logits_3class,
            "attn_weights": None,
        }


# MODEL
class ConvNeXtTinySingleViewEncoder(nn.Module):
    """
    Wrap torchvision ConvNeXt-Tiny so it returns a feature vector (pre-classifier).
    Accepts 1-channel input by replacing the stem conv.
    """
    def __init__(self, pretrained=True, in_chans=1):
        super().__init__()

        if ConvNeXt_Tiny_Weights is not None:
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            base = convnext_tiny(weights=weights)
        else:
            # older torchvision fallback
            base = convnext_tiny(weights="DEFAULT" if pretrained else None)

        # torchvision convnext stem is usually base.features[0][0] = Conv2d(3, ...)
        old_conv = base.features[0][0]
        if not isinstance(old_conv, nn.Conv2d):
            raise RuntimeError("Unexpected ConvNeXt stem layout in torchvision version.")

        new_conv = nn.Conv2d(
            in_channels=in_chans,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

        with torch.no_grad():
            if pretrained and old_conv.weight.shape[1] == 3 and in_chans == 1:
                # average RGB -> grayscale
                w = old_conv.weight.data.mean(dim=1, keepdim=True)
                new_conv.weight.copy_(w)
                if old_conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias.data)
            elif pretrained and old_conv.weight.shape[1] == 3 and in_chans > 1:
                # initialize multi-channel input from averaged RGB weights
                w = old_conv.weight.data.mean(dim=1, keepdim=True)  # [out,1,k,k]
                w = w.repeat(1, in_chans, 1, 1) / float(in_chans)
                new_conv.weight.copy_(w)
                if old_conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias.data)
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
                if new_conv.bias is not None:
                    nn.init.zeros_(new_conv.bias)

        base.features[0][0] = new_conv

        self.features = base.features
        self.avgpool = base.avgpool

        # Keep everything except final Linear as pre-logits
        cls_layers = list(base.classifier.children())
        self.pre_logits = nn.Sequential(*cls_layers[:-1])
        self.out_dim = cls_layers[-1].in_features  # usually 768 for ConvNeXt-Tiny

    def forward(self, x):
        # x: [B,1,H,W]
        x = self.features(x)
        x = self.avgpool(x)
        x = self.pre_logits(x)  # [B, out_dim]
        return x


class JointConvNeXtGWNoTime(nn.Module):
    """
    Inputs:
      X: [B, 2, F, T]   (H1 and L1 as two channels)

    Outputs:
      logit_gw: [B]
    """
    def __init__(
        self,
        head_hidden=256,
        dropout=0.2,
        pretrained=True,
        use_aux_3class=False,
        resize_hw=None,
    ):
        super().__init__()
        self.use_aux_3class = use_aux_3class
        self.resize_hw = resize_hw

        # Same joint 2-detector encoder as before
        self.encoder = ConvNeXtTinySingleViewEncoder(
            pretrained=pretrained,
            in_chans=2,
        )
        enc_dim = self.encoder.out_dim  # usually 768

        # Minimal change: fused feature is now just encoder output
        fusion_out_dim = enc_dim

        self.fusion_norm = nn.LayerNorm(fusion_out_dim)
        self.fusion_dropout = nn.Dropout(dropout)

        self.gw_head = nn.Sequential(
            nn.Linear(fusion_out_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

        if self.use_aux_3class:
            self.head_3class = nn.Sequential(
                nn.Linear(fusion_out_dim, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 3),
            )
        else:
            self.head_3class = None

    def forward(self, X):
        # X: [B,2,F,T]
        if X.ndim != 4 or X.shape[1] != 2:
            raise ValueError(f"Expected X with shape [B,2,F,T], got {tuple(X.shape)}")

        if self.resize_hw is not None and (X.shape[-2:] != self.resize_hw):
            X = F.interpolate(X, size=self.resize_hw, mode="bilinear", align_corners=False)

        # Same two-detector joint encoding
        z = self.encoder(X)  # [B, enc_dim]

        fused = self.fusion_norm(z)
        fused = self.fusion_dropout(fused)

        logit_gw = self.gw_head(fused).squeeze(1)
        logits_3class = self.head_3class(fused) if self.head_3class is not None else None

        return {
            "logit_gw": logit_gw,
            "logits_3class": logits_3class,
            "attn_weights": None,
        }

class MultiViewConvNeXtGWWithTime(nn.Module):
    """
    Inputs:
      X:      [B, 2, F, T]   (two detector q-transforms: H1/L1 as two views)
      t_feat: [B, 2]         (sin/cos of cyclic 24h GPS time)

    Outputs (dict):
      logit_gw:      [B]         primary head, GW vs {noise, glitch}
      logits_3class: [B,3] or None  optional auxiliary head (noise/glitch/gw)
      attn_weights:  [B,2]       detector attention weights (H1/L1)
    """
    def __init__(
        self,
        time_dim=2,
        time_hidden=64,
        fuse_dim=384,
        head_hidden=256,
        dropout=0.2,
        pretrained=True,
        use_aux_3class=False,
        resize_hw=None,   # e.g. (224, 224) if you want forced resizing
    ):
        super().__init__()
        self.use_aux_3class = use_aux_3class
        self.resize_hw = resize_hw

        # Shared encoder for both detector views
        self.encoder = ConvNeXtTinySingleViewEncoder(pretrained=pretrained, in_chans=1)
        enc_dim = self.encoder.out_dim  # typically 768

        # Project encoder features into a fusion space
        self.view_proj = nn.Sequential(
            nn.Linear(enc_dim, fuse_dim),
            nn.GELU(),
            nn.LayerNorm(fuse_dim),
            nn.Dropout(dropout),
        )

        # Time embedding (cyclic 24h -> embedding)
        self.time_net = nn.Sequential(
            nn.Linear(time_dim, time_hidden),
            nn.GELU(),
            nn.LayerNorm(time_hidden),
            nn.Dropout(dropout),
            nn.Linear(time_hidden, time_hidden),
            nn.GELU(),
            nn.LayerNorm(time_hidden),
        )

        # Gated attention fusion over detector views, conditioned on time embedding
        self.attn_gate = nn.Sequential(
            nn.Linear(fuse_dim + time_hidden, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        # Final fused feature:
        #  - attention-weighted sum
        #  - mean over views
        #  - |H1-L1| coherence-ish morphology difference
        #  - time embedding
        fusion_out_dim = fuse_dim + fuse_dim + fuse_dim + time_hidden

        self.fusion_norm = nn.LayerNorm(fusion_out_dim)
        self.fusion_dropout = nn.Dropout(dropout)

        # Primary head: GW vs rest (the one you care about most)
        self.gw_head = nn.Sequential(
            nn.Linear(fusion_out_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

        # Optional auxiliary 3-class head
        if self.use_aux_3class:
            self.head_3class = nn.Sequential(
                nn.Linear(fusion_out_dim, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 3),  # logits for [noise, glitch, gw]
            )
        else:
            self.head_3class = None

    def forward(self, X, t_feat):
        # X: [B,2,F,T]
        B, V, H, W = X.shape
        if V != 2:
            raise ValueError(f"Expected exactly 2 detector views, got V={V}")

        # Encode both views in one pass through the shared encoder
        x = X.reshape(B * V, 1, H, W)

        if self.resize_hw is not None and (x.shape[-2:] != self.resize_hw):
            x = F.interpolate(x, size=self.resize_hw, mode="bilinear", align_corners=False)

        z = self.encoder(x)          # [B*2, enc_dim]
        z = self.view_proj(z)        # [B*2, fuse_dim]
        z = z.view(B, V, -1)         # [B,2,fuse_dim]

        # Time embedding
        t = self.time_net(t_feat)    # [B,time_hidden]

        # Attention scores per view, conditioned on time embedding
        t_rep = t.unsqueeze(1).expand(-1, V, -1)           # [B,2,time_hidden]
        gate_in = torch.cat([z, t_rep], dim=-1)            # [B,2,fuse_dim+time_hidden]
        attn_scores = self.attn_gate(gate_in).squeeze(-1)  # [B,2]
        attn_weights = torch.softmax(attn_scores, dim=1)   # [B,2]

        # Fused detector features
        z_attn = (attn_weights.unsqueeze(-1) * z).sum(dim=1)  # [B,fuse_dim]
        z_mean = z.mean(dim=1)                                  # [B,fuse_dim]
        z_absdiff = (z[:, 0] - z[:, 1]).abs()                  # [B,fuse_dim]

        fused = torch.cat([z_attn, z_mean, z_absdiff, t], dim=1)
        fused = self.fusion_norm(fused)
        fused = self.fusion_dropout(fused)

        logit_gw = self.gw_head(fused).squeeze(1)  # [B]
        logits_3class = self.head_3class(fused) if self.head_3class is not None else None

        return {
            "logit_gw": logit_gw,
            "logits_3class": logits_3class,
            "attn_weights": attn_weights,
        }
    



# LOSS FUNCTIONS AND METRICS
def focal_bce_with_logits(
    logits,
    targets,
    gamma=2.0,
    pos_weight=None,
    reduction="mean",
):
    """
    Binary focal loss built on BCEWithLogits.
    targets: float tensor in {0,1}
    """
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)  # prob of true class
    focal = (1.0 - p_t).pow(gamma)
    loss = focal * bce

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def multitask_gw_losses_and_metrics(
    outputs,
    y,
    lambda_aux=0.30,             # auxiliary 3-class loss weight
    gw_focal_gamma=2.0,          # make the primary head focus on hard examples
    pos_weight_gw=None,          # set >1 to push GW recall (e.g. 1.5~3.0)
    aux_label_smoothing=0.02,
    gw_threshold=0.5,
):
    """
    y: LongTensor [B] with values {0=noise, 1=glitch, 2=gw}
    Primary objective: GW vs rest
    Optional auxiliary: 3-class CE
    """
    device = y.device
    logit_gw = outputs["logit_gw"]                # [B]
    logits_3class = outputs["logits_3class"]      # [B,3] or None

    target_gw = (y == 2).float()  # 1 for gw, 0 otherwise

    pw = None
    if pos_weight_gw is not None:
        # allow float/int or tensor
        if not torch.is_tensor(pos_weight_gw):
            pos_weight_gw = torch.tensor([float(pos_weight_gw)], device=device, dtype=logit_gw.dtype)
        else:
            pos_weight_gw = pos_weight_gw.to(device=device, dtype=logit_gw.dtype)
        pw = pos_weight_gw

    loss_gw = focal_bce_with_logits(
        logit_gw, target_gw,
        gamma=gw_focal_gamma,
        pos_weight=pw,
        reduction="mean",
    )

    if logits_3class is not None and lambda_aux > 0.0:
        loss_aux = F.cross_entropy(
            logits_3class, y,
            label_smoothing=aux_label_smoothing,
        )
    else:
        loss_aux = torch.zeros((), device=device, dtype=logit_gw.dtype)

    loss = loss_gw + lambda_aux * loss_aux

    with torch.no_grad():
        p_gw = torch.sigmoid(logit_gw)
        pred_gw = (p_gw > gw_threshold)
        true_gw = (y == 2)

        tp = (pred_gw & true_gw).sum().float()
        fp = (pred_gw & (~true_gw)).sum().float()
        fn = ((~pred_gw) & true_gw).sum().float()

        gw_precision = tp / (tp + fp + 1e-8)
        gw_recall = tp / (tp + fn + 1e-8)
        gw_acc = (pred_gw == true_gw).float().mean()

        if logits_3class is not None:
            yhat3 = logits_3class.argmax(dim=1)
            acc3 = (yhat3 == y).float().mean()
        else:
            acc3 = torch.tensor(float("nan"), device=device)

    metrics = {
        "loss_total": loss.detach(),
        "loss_gw": loss_gw.detach(),
        "loss_aux3": loss_aux.detach(),
        "gw_acc_bin": gw_acc.detach(),
        "gw_precision": gw_precision.detach(),
        "gw_recall": gw_recall.detach(),
        "acc_3class_aux": acc3.detach(),
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


def three_class_from_aux_logits(logits_3class):
    return logits_3class.argmax(dim=1)

def to_model_dtype(x, model, device, non_blocking=False):
    dtype = next(model.parameters()).dtype
    return x.to(device=device, dtype=dtype, non_blocking=non_blocking)

def model_forward(model, X, t_feat=None):
    """
    Allows the same training/eval code to work for:
      - time-aware models: model(X, t_feat)
      - no-time models:    model(X)
    """
    if t_feat is None:
        return model(X)

    try:
        return model(X, t_feat)
    except TypeError:
        return model(X)

@torch.no_grad()
def evaluate(model, loader, device, return_outputs=False, gw_threshold=0.5, loss_cfg=None):
    model.eval()

    total_loss = 0.0
    n = 0

    # Binary GW-vs-rest metrics
    tp = fp = fn = tn = 0

    # Optional 3-class aux metrics
    correct3 = 0
    has_aux_head = None

    # Collect for ROC/PR and optional outputs
    all_y = []
    all_p_gw = []
    all_pred_gw = []
    all_yhat3 = []
    all_p3 = []
    all_attn = []

    # dataset-order sample indices
    all_sample_idx = []
    running_sample_idx = 0

    for X, t_feat, y in loader:
        X = to_model_dtype(X, model, device)
        y = y.to(device)

        # keep reading t_feat from dataset to avoid dataset changes,
        # but do not use it for no-time models
        t_feat = to_model_dtype(t_feat, model, device) if t_feat is not None else None

        outputs = model_forward(model, X, t_feat)
        loss_kwargs = {} if loss_cfg is None else dict(loss_cfg)
        loss, _ = multitask_gw_losses_and_metrics(outputs, y, gw_threshold=gw_threshold, **loss_kwargs)

        bs = X.size(0)
        total_loss += float(loss.item()) * bs
        n += bs

        logit_gw = outputs["logit_gw"]
        p_gw = torch.sigmoid(logit_gw)
        pred_gw = (p_gw > gw_threshold)
        true_gw = (y == 2)

        tp += int((pred_gw & true_gw).sum().item())
        fp += int((pred_gw & (~true_gw)).sum().item())
        fn += int(((~pred_gw) & true_gw).sum().item())
        tn += int(((~pred_gw) & (~true_gw)).sum().item())

        logits_3class = outputs["logits_3class"]
        if has_aux_head is None:
            has_aux_head = (logits_3class is not None)

        if logits_3class is not None:
            yhat3 = logits_3class.argmax(dim=1)
            correct3 += int((yhat3 == y).sum().item())
            all_yhat3.append(yhat3.detach().cpu())
            all_p3.append(torch.softmax(logits_3class, dim=1).detach().cpu())

        all_y.append(y.detach().cpu())
        all_p_gw.append(p_gw.detach().cpu())
        all_pred_gw.append(pred_gw.long().detach().cpu())

        if "attn_weights" in outputs and outputs["attn_weights"] is not None:
            all_attn.append(outputs["attn_weights"].detach().cpu())

        batch_indices = torch.arange(running_sample_idx, running_sample_idx + bs, dtype=torch.long)
        all_sample_idx.append(batch_indices)
        running_sample_idx += bs

    # Aggregate metrics
    gw_precision = tp / max(1, tp + fp)
    gw_recall = tp / max(1, tp + fn)
    gw_f1 = (2 * gw_precision * gw_recall) / max(1e-12, gw_precision + gw_recall)
    gw_acc_bin = (tp + tn) / max(1, tp + tn + fp + fn)

    y_true = torch.cat(all_y).numpy()
    p_gw_all = torch.cat(all_p_gw).numpy()
    pred_gw_all = torch.cat(all_pred_gw).numpy()

    y_true_gw_bin = (y_true == 2).astype(np.uint8)

    if np.unique(y_true_gw_bin).size >= 2:
        gw_roc_auc = float(roc_auc_score(y_true_gw_bin, p_gw_all))
        gw_ap = float(average_precision_score(y_true_gw_bin, p_gw_all))
    else:
        gw_roc_auc = float("nan")
        gw_ap = float("nan")

    out = {
        "loss": total_loss / max(1, n),
        "gw_acc_bin": gw_acc_bin,
        "gw_precision": gw_precision,
        "gw_recall": gw_recall,
        "gw_f1": gw_f1,
        "gw_roc_auc": gw_roc_auc,
        "gw_ap": gw_ap,
        "acc_3class": (correct3 / max(1, n)) if has_aux_head else float("nan"),
        "n": n,
    }

    if return_outputs:
        out_dict = {
            "sample_index": torch.cat(all_sample_idx).numpy(),
            "y_true": y_true,
            "y_true_gw_bin": y_true_gw_bin,
            "pred_gw": pred_gw_all,
            "p_gw": p_gw_all,
        }
        if len(all_yhat3) > 0:
            out_dict["y_pred3"] = torch.cat(all_yhat3).numpy()
        if len(all_p3) > 0:
            out_dict["p3"] = torch.cat(all_p3).numpy()
        if len(all_attn) > 0:
            out_dict["attn_weights"] = torch.cat(all_attn).numpy()
        out["outputs"] = out_dict

    return out


def binary_confusion_counts(y_true_bin, y_pred_bin):
    """
    Returns:
        cm = [[tn, fp],
              [fn, tp]]
    """
    y_true_bin = np.asarray(y_true_bin).astype(np.int64)
    y_pred_bin = np.asarray(y_pred_bin).astype(np.int64)

    tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
    fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
    fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))
    tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))

    return np.array([[tn, fp],
                     [fn, tp]], dtype=np.int64)

def plot_binary_confusion_matrix(cm, out_path, class_names=("Non-GW", "GW"), normalise=False):
    cm_plot = cm.astype(np.float64) if normalise else cm.astype(np.int64)
    title = "Binary Confusion Matrix (Counts)"
    fmt = "d"

    if normalise:
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_plot = cm_plot / row_sums
        title = "Binary Confusion Matrix (Row-Normalised)"
        fmt = ".2f"

    plt.figure(figsize=(5, 4))
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
            plt.text(
                j, i, txt,
                ha="center", va="center",
                color="white" if cm_plot[i, j] > thresh else "black"
            )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", transparent=True)
    plt.close()


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

CLASS_NAMES = ["Noise", "Glitch", "GW"]
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

    epochs = np.asarray(history["epoch"], dtype=np.int64) + 1

    plt.figure(figsize=(8, 5))
    # if "val_acc3" in history:
        # plt.plot(epochs, history["val_acc3"], label="Accuracy (3-Class Aux)")
    if "val_gw_precision" in history:
        plt.plot(epochs, history["val_gw_precision"], color=PLOT_COLORS[0], label="GW Precision")
    if "val_gw_recall" in history:
        plt.plot(epochs, history["val_gw_recall"], color=PLOT_COLORS[1], label="GW Recall")
    if "val_gw_ap" in history:
        plt.plot(epochs, history["val_gw_ap"], color=PLOT_COLORS[2], label="GW AP")

    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Validation Metrics (GW-focused)")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "val_metrics_curves.pdf"), bbox_inches='tight', transparent=True)
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
    cbar.ax.yaxis.get_offset_text().set_x(2)
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


def plot_gw_score_histograms(y_true, p_gw, out_dir):
    """
    p_gw: sigmoid output from primary GW-vs-rest head
    """
    os.makedirs(out_dir, exist_ok=True)
    y_true = np.asarray(y_true)
    p_gw = np.asarray(p_gw)

    gw_mask = (y_true == 2)
    non_gw_mask = (y_true != 2)

    plt.figure(figsize=(7, 5))
    if non_gw_mask.any():
        plt.hist(p_gw[non_gw_mask], bins=40, alpha=0.6, color="slateblue", density=True, label="True Non-GW (Noise+Glitch)")
    if gw_mask.any():
        plt.hist(p_gw[gw_mask], bins=40, alpha=0.6, color="cornflowerblue", density=True, label="True GW")
    plt.xlabel("Predicted Probability of GW")
    plt.ylabel("Density")
    plt.title("GW-vs-Rest Score Distribution (Test)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "test_gw_head_scores.pdf"), bbox_inches='tight', transparent=True)
    plt.close()


def make_test_gw_roc_pr_plots(y_true, p_gw, out_dir):
    """
    ROC/PR for primary head: positive = GW (y==2), negative = {noise, glitch}
    """
    os.makedirs(out_dir, exist_ok=True)

    y_true = np.asarray(y_true)
    p_gw = np.asarray(p_gw, dtype=np.float64)
    y_gw_bin = (y_true == 2).astype(np.uint8)

    metrics = {}
    metrics["gw_head_roc_auc"] = _plot_binary_roc_curve(
        y_gw_bin,
        p_gw,
        os.path.join(out_dir, "test_gw_head_ROC.pdf"),
        "ROC: GW Head (GW vs {Noise, Glitch})",
    )
    metrics["gw_head_average_precision"] = _plot_binary_pr_curve(
        y_gw_bin,
        p_gw,
        os.path.join(out_dir, "test_gw_head_PR.pdf"),
        "PR: GW Head (GW vs {Noise, Glitch})",
    )
    return metrics


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
    plt.title("Per-Class Accuracy (Test Data)")
    plt.grid(True, axis="y", alpha=0.3)

    # Annotate with acc and counts
    for i, b in enumerate(bars):
        # txt = f"{acc[i]:.3f}\n({int(correct[i])}/{int(row_sums[i])})"
        txt = f"{acc[i]*100:.1f}%\n({int(correct[i])}/{int(row_sums[i])})"
        plt.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() - 0.02,
            txt,
            ha="center",
            va="top",
            fontsize=9,
            color="white",
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
    plt.plot(epochs, history["train_loss"], label="Train Loss", color=PLOT_COLORS[0])
    plt.plot(epochs, history["val_loss"], label="Validation Loss", color=PLOT_COLORS[1])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, "loss_curves.pdf"), bbox_inches="tight", transparent=True)
    plt.close()


def main():
    # -------------------
    # Main training setup
    # -------------------
    seed_everything(2026)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    use_amp = (device == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)

    LOSS_CFG = dict(
        lambda_aux=0.30,
        gw_focal_gamma=2.0,
        pos_weight_gw=1.5,          # scalar here; tensor created on-device inside fn
        aux_label_smoothing=0.02,
    )

    # Keep X in saved dtype on CPU (likely float16); cast once per batch on GPU instead
    train_ds = PrecomputedPTShardDataset("Training_Data_Generation/pt_dataset/train", cast_x_to_float32=False, max_samples=None)  # use full 240k
    val_ds = PrecomputedPTShardDataset("Training_Data_Generation/pt_dataset/val", cast_x_to_float32=False, max_samples=None)  # use full 30k
    test_ds = PrecomputedPTShardDataset("Training_Data_Generation/pt_dataset/test", cast_x_to_float32=False, max_samples=None)  # use full 30k


    # IMPORTANT for shard-backed datasets:
    # shuffle=True causes random shard access and many torch.load() calls.
    # Start with shuffle=False for speed. (If you later want randomness, do shard-level shuffling.)
    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
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

    model = JointConvNeXtGWNoTime(
        pretrained=True,
        use_aux_3class=False,
        head_hidden=256,
        dropout=0.30,
    ).to(device)

    # for p in model.encoder.parameters():
    #     p.requires_grad = False

    # ConvNeXt usually likes a slightly smaller LR when fully fine-tuning
    param_groups = [
        {"params": model.encoder.parameters(), "lr": 1e-5},
        {"params": model.gw_head.parameters(), "lr": 1e-4},
    ]

    if model.head_3class is not None:
        param_groups.append({"params": model.head_3class.parameters(), "lr": 1e-4})

    optimiser = AdamW(param_groups, weight_decay=1.5e-2)

    start_epoch = 0
    best_val = -float("inf")
    early_stopping_patience = 5
    early_stopping_min_delta = 1e-4
    epochs_without_improvement = 0
    ckpt_path = "checkpoints12/best.pt"
    resume_path = None  # set to "checkpoints5/best.pt" to resume

    fig_dir = "training_figures12"
    os.makedirs(fig_dir, exist_ok=True)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc3": [],
        "val_gw_acc_bin": [],
        "val_gw_precision": [],
        "val_gw_recall": [],
        "val_gw_f1": [],
        "val_gw_roc_auc": [],
        "val_gw_ap": [],
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
        # if epoch == 3:
        #     for p in model.encoder.parameters():
        #         p.requires_grad = True
        
        # if epoch == 6:
        #     for p in model.encoder.parameters():
        #         p.requires_grad = False

        for X, t_feat, y in train_loader:
            print(f"Epoch {epoch:03d} | Batch {n//train_loader.batch_size:04d}/{len(train_loader):04d}", end="\r")
            X = to_model_dtype(X, model, device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # keep dataset unchanged; just ignore time in the no-time model
            t_feat = to_model_dtype(t_feat, model, device, non_blocking=True) if t_feat is not None else None

            optimiser.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=use_amp):
                outputs = model_forward(model, X, t_feat)

                # Example settings:
                # - pos_weight_gw > 1 nudges recall on GW class
                # - lambda_aux keeps 3-class separation as regulariser
                loss, _metrics = multitask_gw_losses_and_metrics(
                    outputs, y, gw_threshold=0.5, **LOSS_CFG
                )

            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            bs = X.size(0)
            running += float(loss.item()) * bs
            n += bs

        train_loss = running / max(1, n)
        val_stats = evaluate(model, val_loader, device, return_outputs=False, gw_threshold=0.5, loss_cfg=LOSS_CFG)

        epoch_time_s = time.time() - t_epoch0

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_stats["loss"]))
        history["val_acc3"].append(float(val_stats["acc_3class"]))
        history["val_gw_acc_bin"].append(float(val_stats["gw_acc_bin"]))
        history["val_gw_precision"].append(float(val_stats["gw_precision"]))
        history["val_gw_recall"].append(float(val_stats["gw_recall"]))
        history["val_gw_f1"].append(float(val_stats["gw_f1"]))
        history["val_gw_roc_auc"].append(float(val_stats["gw_roc_auc"]))
        history["val_gw_ap"].append(float(val_stats["gw_ap"]))
        history["epoch_time_s"].append(float(epoch_time_s))

        # overwrite plots each epoch (latest curves)
        plot_loss_curves(history, fig_dir)
        plot_val_metric_curves(history, fig_dir)  # replace this helper; see below
        plot_epoch_time_curve(history, fig_dir)
        save_history_json(history, os.path.join(fig_dir, "history.json"))

        msg = (
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_stats['loss']:.4f} "
            f"| GW(bin) P={val_stats['gw_precision']:.3f} "
            f"R={val_stats['gw_recall']:.3f} F1={val_stats['gw_f1']:.3f} "
            f"| val_GW_AP={val_stats['gw_ap']:.4f} | val_GW_AUC={val_stats['gw_roc_auc']:.4f}"
        )
        if np.isfinite(val_stats["acc_3class"]):
            msg += f" | val_acc3={val_stats['acc_3class']:.3f}"
        print(msg)

        # save best on GW average precision (primary objective)
        val_key = "gw_ap"
        val_metric = float(val_stats.get(val_key, float("nan")))

        if np.isfinite(val_metric) and (val_metric > best_val + early_stopping_min_delta):
            best_val = val_metric
            epochs_without_improvement = 0
            save_checkpoint(ckpt_path, model, optimiser, epoch, best_val)
            print(f"  -> New best {val_key}: {best_val:.6f}. Checkpoint saved.")
        else:
            epochs_without_improvement += 1
            print(
                f"  -> No significant improvement in {val_key} "
                f"for {epochs_without_improvement} epoch(s)."
            )

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping triggered after {early_stopping_patience} "
                f"epochs without improvement in {val_key}."
            )
            break

    # final test eval (best checkpoint)
    ckpt = load_checkpoint(ckpt_path, model, optimiser=None, map_location=device)
    test_stats = evaluate(model, test_loader, device, return_outputs=True, loss_cfg=LOSS_CFG)

    # Print summary metrics only
    test_summary = {k: v for k, v in test_stats.items() if k != "outputs"}
    print("TEST:", test_summary)

    # Save test metrics json
    with open(os.path.join(fig_dir, "test_stats.json"), "w") as f:
        json.dump({k: _to_python_scalar(v) for k, v in test_summary.items()}, f, indent=2)

    # Make test plots
    outs = test_stats["outputs"]
    y_true = outs["y_true"]
    y_true_gw_bin = outs["y_true_gw_bin"]
    pred_gw = outs["pred_gw"]
    p_gw = outs["p_gw"]
    sample_index = outs["sample_index"]

    pred_df = pd.DataFrame({
        "sample_index": sample_index.astype(int),
        "true_class": y_true.astype(int),                  # original 0/1/2 label
        "true_is_gw": y_true_gw_bin.astype(int),          # binary target
        "pred_is_gw": pred_gw.astype(int),                # binary prediction
        "p_gw": p_gw.astype(float),                       # score from primary head
        "correct_gw_binary": (y_true_gw_bin == pred_gw),  # binary correctness
    })

    if "y_pred3" in outs:
        pred_df["pred_class_aux"] = outs["y_pred3"].astype(int)

    pred_df.to_csv(os.path.join(fig_dir, "test_predictions_binary.csv"), index=False)

    cm_bin = binary_confusion_counts(y_true_gw_bin, pred_gw)

    plot_binary_confusion_matrix(
        cm_bin,
        os.path.join(fig_dir, "test_gw_confusion_counts.pdf"),
        class_names=("Non-GW", "GW"),
        normalise=False,
    )

    plot_binary_confusion_matrix(
        cm_bin,
        os.path.join(fig_dir, "test_gw_confusion_normalised.pdf"),
        class_names=("Non-GW", "GW"),
        normalise=True,
    )

    # GW score histogram (primary head)
    plot_gw_score_histograms(y_true, p_gw, fig_dir)

    # ROC / PR curves for GW-vs-rest primary head
    rocpr_metrics = make_test_gw_roc_pr_plots(y_true, p_gw, fig_dir)

    # Optional: save average detector attention weights by class for interpretability
    if "attn_weights" in outs:
        attn = outs["attn_weights"]  # shape [N,2] for [H1,L1]
        summary = {}
        for cls_id, cls_name in enumerate(CLASS_NAMES):
            m = (y_true == cls_id)
            if np.any(m):
                summary[cls_name] = {
                    "mean_attn_H1": float(attn[m, 0].mean()),
                    "mean_attn_L1": float(attn[m, 1].mean()),
                }
        with open(os.path.join(fig_dir, "test_attention_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    # Training curves (loss and val acc) over epochs
    plot_loss_curves(history, fig_dir)

    # Save ROC/PR summary metrics
    with open(os.path.join(fig_dir, "test_rocpr_metrics.json"), "w") as f:
        json.dump({k: _to_python_scalar(v) for k, v in rocpr_metrics.items()}, f, indent=2)


if __name__ == "__main__":
    main()