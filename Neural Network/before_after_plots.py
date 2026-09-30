"""
Before vs After Training — Matplotlib visualization module.

Generates comparison graphs from the evaluation results produced by
before_after_eval.py.  All values are from actual model outputs;
nothing is fabricated.

Output: PNG files saved to the specified output directory.
"""

import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

# Consistent color palette
COLOR_BEFORE = "#4A90D9"    # steel blue
COLOR_AFTER  = "#2ECC71"    # emerald green
COLOR_ACCENT = "#E74C3C"    # red for emphasis
COLOR_GRID   = "#ECECEC"

FIGSIZE_SINGLE = (10, 6)
FIGSIZE_WIDE   = (14, 6)
FIGSIZE_TALL   = (10, 8)
DPI = 150


def _style_ax(ax, title: str, ylabel: str = "", xlabel: str = ""):
    """Apply consistent styling to an axes object."""
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    ax.grid(axis="y", alpha=0.3, color=COLOR_GRID)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# 1. Accuracy bar chart
# ---------------------------------------------------------------------------

def plot_accuracy_comparison(
    before_agg: Dict[str, Any],
    after_agg: Dict[str, Any],
    output_dir: str,
) -> Optional[str]:
    """Bar chart: Before vs After MC accuracy (%)."""
    ba = before_agg.get("mc_accuracy")
    aa = after_agg.get("mc_accuracy")

    if ba is None or aa is None:
        print("[plots] Skipping accuracy chart — no MC accuracy data.")
        return None

    ba_pct = ba * 100
    aa_pct = aa * 100

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, dpi=DPI)

    bars = ax.bar(
        ["Before Training\n(DoLa only)", "After Training\n(DoLa + ENN)"],
        [ba_pct, aa_pct],
        color=[COLOR_BEFORE, COLOR_AFTER],
        width=0.5,
        edgecolor="white",
        linewidth=1.5,
    )

    # Value labels on bars
    for bar, val in zip(bars, [ba_pct, aa_pct]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.1f}%",
            ha="center", va="bottom",
            fontsize=14, fontweight="bold",
        )

    _style_ax(ax, "Before vs After Training — MC Accuracy", ylabel="Accuracy (%)")
    ax.set_ylim(0, max(ba_pct, aa_pct) * 1.2 + 5)

    # Improvement annotation
    diff = aa_pct - ba_pct
    color = COLOR_AFTER if diff >= 0 else COLOR_ACCENT
    ax.annotate(
        f"Change: {diff:+.1f}%",
        xy=(0.5, 0.95), xycoords="axes fraction",
        ha="center", fontsize=12, color=color,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.8),
    )

    path = os.path.join(output_dir, "accuracy_comparison.png")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# 2. Per-question log-likelihood comparison
# ---------------------------------------------------------------------------

def plot_log_likelihood_comparison(
    comparison: List[Dict[str, Any]],
    output_dir: str,
) -> str:
    """Grouped bar chart: log-likelihood per question, before vs after."""
    n = len(comparison)
    labels = [f"Q{r['id']}" for r in comparison]
    before_vals = [r["before_log_likelihood"] for r in comparison]
    after_vals = [r["after_log_likelihood"] for r in comparison]

    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE, dpi=DPI)

    ax.bar(x - width / 2, before_vals, width, label="Before (DoLa)",
           color=COLOR_BEFORE, edgecolor="white", linewidth=0.8)
    ax.bar(x + width / 2, after_vals, width, label="After (DoLa+ENN)",
           color=COLOR_AFTER, edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=10)

    _style_ax(ax, "Log-Likelihood per Question — Before vs After",
              ylabel="Mean Log-Likelihood", xlabel="Question")

    path = os.path.join(output_dir, "log_likelihood_comparison.png")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# 3. Per-question confidence comparison
# ---------------------------------------------------------------------------

def plot_confidence_comparison(
    comparison: List[Dict[str, Any]],
    output_dir: str,
) -> str:
    """Grouped bar chart: confidence per question."""
    n = len(comparison)
    labels = [f"Q{r['id']}" for r in comparison]
    before_vals = [r["before_confidence"] for r in comparison]
    after_vals = [r["after_confidence"] for r in comparison]

    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE, dpi=DPI)

    ax.bar(x - width / 2, before_vals, width, label="Before (DoLa)",
           color=COLOR_BEFORE, edgecolor="white", linewidth=0.8)
    ax.bar(x + width / 2, after_vals, width, label="After (DoLa+ENN)",
           color=COLOR_AFTER, edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=10)

    _style_ax(ax, "Confidence per Question — Before vs After",
              ylabel="Confidence (exp(log-likelihood))", xlabel="Question")

    path = os.path.join(output_dir, "confidence_comparison.png")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# 4. Per-question entropy comparison
# ---------------------------------------------------------------------------

def plot_entropy_comparison(
    comparison: List[Dict[str, Any]],
    output_dir: str,
) -> str:
    """Grouped bar chart: entropy per question."""
    n = len(comparison)
    labels = [f"Q{r['id']}" for r in comparison]
    before_vals = [r["before_entropy"] for r in comparison]
    after_vals = [r["after_entropy"] for r in comparison]

    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE, dpi=DPI)

    ax.bar(x - width / 2, before_vals, width, label="Before (DoLa)",
           color=COLOR_BEFORE, edgecolor="white", linewidth=0.8)
    ax.bar(x + width / 2, after_vals, width, label="After (DoLa+ENN)",
           color=COLOR_AFTER, edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=10)

    _style_ax(ax, "Output Entropy per Question — Before vs After",
              ylabel="Mean Entropy (nats)", xlabel="Question")

    path = os.path.join(output_dir, "entropy_comparison.png")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# 5. ENN uncertainty breakdown (after-training only)
# ---------------------------------------------------------------------------

def plot_uncertainty_breakdown(
    comparison: List[Dict[str, Any]],
    output_dir: str,
) -> Optional[str]:
    """Grouped bar chart: epistemic vs aleatoric vs predictive uncertainty."""
    # Filter to questions with uncertainty data
    with_unc = [r for r in comparison if r.get("after_epistemic") is not None]

    if not with_unc:
        print("[plots] Skipping uncertainty breakdown — no ENN uncertainty data.")
        return None

    n = len(with_unc)
    labels = [f"Q{r['id']}" for r in with_unc]

    epistemic = [r["after_epistemic"] for r in with_unc]
    aleatoric = [r["after_aleatoric"] for r in with_unc]
    predictive = [r["after_predictive"] for r in with_unc]

    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE, dpi=DPI)

    ax.bar(x - width, epistemic, width, label="Epistemic",
           color="#E74C3C", edgecolor="white", linewidth=0.8)
    ax.bar(x, aleatoric, width, label="Aleatoric",
           color="#F39C12", edgecolor="white", linewidth=0.8)
    ax.bar(x + width, predictive, width, label="Predictive",
           color="#9B59B6", edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=10)

    _style_ax(ax, "ENN Uncertainty Decomposition (After Training)",
              ylabel="Uncertainty (nats)", xlabel="Question")

    path = os.path.join(output_dir, "uncertainty_breakdown.png")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# 6. Correctness heatmap
# ---------------------------------------------------------------------------

def plot_correctness_heatmap(
    comparison: List[Dict[str, Any]],
    output_dir: str,
) -> Optional[str]:
    """Color-coded grid: correct/incorrect per question, before vs after."""
    mc_rows = [r for r in comparison if r.get("before_mc_correct") is not None]

    if not mc_rows:
        print("[plots] Skipping correctness heatmap — no MC data.")
        return None

    n = len(mc_rows)
    labels = [f"Q{r['id']}" for r in mc_rows]

    # Build 2×N grid: row 0 = before, row 1 = after
    grid = np.zeros((2, n))
    for j, r in enumerate(mc_rows):
        grid[0, j] = 1.0 if r["before_mc_correct"] else 0.0
        grid[1, j] = 1.0 if r["after_mc_correct"] else 0.0

    fig, ax = plt.subplots(figsize=(max(8, n * 1.2), 3), dpi=DPI)

    cmap = plt.cm.colors.ListedColormap(["#E74C3C", "#2ECC71"])
    ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Labels
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Before", "After"], fontsize=11)

    # Cell text
    for i in range(2):
        for j in range(n):
            val = "✓" if grid[i, j] == 1.0 else "✗"
            ax.text(j, i, val, ha="center", va="center",
                    fontsize=14, fontweight="bold", color="white")

    _style_ax(ax, "Per-Question Correctness — Before vs After")
    ax.grid(False)

    # Legend
    correct_patch = mpatches.Patch(color="#2ECC71", label="Correct")
    wrong_patch = mpatches.Patch(color="#E74C3C", label="Incorrect")
    ax.legend(handles=[correct_patch, wrong_patch], loc="upper right", fontsize=9)

    path = os.path.join(output_dir, "correctness_heatmap.png")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# 7. Training loss curve
# ---------------------------------------------------------------------------

def plot_training_loss(
    train_losses: List[float],
    val_losses: Optional[List[float]],
    output_dir: str,
) -> Optional[str]:
    """Line chart: ENN training and validation loss over epochs."""
    if not train_losses:
        print("[plots] Skipping training loss — no data.")
        return None

    epochs = list(range(1, len(train_losses) + 1))

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, dpi=DPI)

    ax.plot(epochs, train_losses, "o-", color=COLOR_BEFORE,
            label="Train Loss", linewidth=2, markersize=6)

    if val_losses and len(val_losses) == len(train_losses):
        ax.plot(epochs, val_losses, "s--", color=COLOR_AFTER,
                label="Val Loss", linewidth=2, markersize=6)

    ax.legend(fontsize=11)
    _style_ax(ax, "ENN Training Loss",
              ylabel="Cross-Entropy Loss", xlabel="Epoch")

    path = os.path.join(output_dir, "training_loss.png")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# 8. Summary dashboard
# ---------------------------------------------------------------------------

def plot_summary_dashboard(
    comparison: List[Dict[str, Any]],
    before_agg: Dict[str, Any],
    after_agg: Dict[str, Any],
    train_losses: Optional[List[float]],
    val_losses: Optional[List[float]],
    output_dir: str,
) -> str:
    """Multi-panel summary figure combining key metrics."""
    has_mc = before_agg.get("mc_accuracy") is not None
    has_unc = any(r.get("after_epistemic") is not None for r in comparison)
    has_loss = train_losses and len(train_losses) > 0

    # Determine grid
    n_panels = 2  # always: log-likelihood + entropy
    if has_mc:
        n_panels += 1
    if has_unc:
        n_panels += 1
    if has_loss:
        n_panels += 1

    n_cols = min(3, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(6 * n_cols, 5 * n_rows), dpi=DPI)
    if n_panels == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    panel = 0

    # Panel: Accuracy
    if has_mc:
        ax = axes[panel]
        ba = before_agg["mc_accuracy"] * 100
        aa = after_agg["mc_accuracy"] * 100
        bars = ax.bar(["Before", "After"], [ba, aa],
                       color=[COLOR_BEFORE, COLOR_AFTER], width=0.5)
        for bar, val in zip(bars, [ba, aa]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
        _style_ax(ax, "MC Accuracy (%)", ylabel="%")
        ax.set_ylim(0, 105)
        panel += 1

    # Panel: Mean Log-Likelihood
    ax = axes[panel]
    n = len(comparison)
    x = np.arange(n)
    width = 0.35
    labels = [f"Q{r['id']}" for r in comparison]
    ax.bar(x - width/2, [r["before_log_likelihood"] for r in comparison],
           width, color=COLOR_BEFORE, label="Before")
    ax.bar(x + width/2, [r["after_log_likelihood"] for r in comparison],
           width, color=COLOR_AFTER, label="After")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.legend(fontsize=8)
    _style_ax(ax, "Log-Likelihood", ylabel="Log-Prob")
    panel += 1

    # Panel: Entropy
    ax = axes[panel]
    ax.bar(x - width/2, [r["before_entropy"] for r in comparison],
           width, color=COLOR_BEFORE, label="Before")
    ax.bar(x + width/2, [r["after_entropy"] for r in comparison],
           width, color=COLOR_AFTER, label="After")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.legend(fontsize=8)
    _style_ax(ax, "Entropy", ylabel="Nats")
    panel += 1

    # Panel: Uncertainty
    if has_unc:
        ax = axes[panel]
        unc_data = [r for r in comparison if r.get("after_epistemic") is not None]
        unc_labels = [f"Q{r['id']}" for r in unc_data]
        xw = np.arange(len(unc_data))
        w = 0.25
        ax.bar(xw - w, [r["after_epistemic"] for r in unc_data], w,
               color="#E74C3C", label="Epistemic")
        ax.bar(xw, [r["after_aleatoric"] for r in unc_data], w,
               color="#F39C12", label="Aleatoric")
        ax.bar(xw + w, [r["after_predictive"] for r in unc_data], w,
               color="#9B59B6", label="Predictive")
        ax.set_xticks(xw)
        ax.set_xticklabels(unc_labels, fontsize=7)
        ax.legend(fontsize=8)
        _style_ax(ax, "ENN Uncertainty (After)", ylabel="Nats")
        panel += 1

    # Panel: Training loss
    if has_loss:
        ax = axes[panel]
        epochs = list(range(1, len(train_losses) + 1))
        ax.plot(epochs, train_losses, "o-", color=COLOR_BEFORE,
                label="Train", linewidth=2, markersize=5)
        if val_losses and len(val_losses) == len(train_losses):
            ax.plot(epochs, val_losses, "s--", color=COLOR_AFTER,
                    label="Val", linewidth=2, markersize=5)
        ax.legend(fontsize=8)
        _style_ax(ax, "ENN Training Loss", ylabel="Loss", xlabel="Epoch")
        panel += 1

    # Hide unused axes
    for i in range(panel, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Before vs After Training — Summary Dashboard",
                 fontsize=16, fontweight="bold", y=1.02)

    path = os.path.join(output_dir, "summary_dashboard.png")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_all_plots(
    comparison: List[Dict[str, Any]],
    before_agg: Dict[str, Any],
    after_agg: Dict[str, Any],
    train_losses: Optional[List[float]] = None,
    val_losses: Optional[List[float]] = None,
    output_dir: str = "output/before_after",
) -> List[str]:
    """Generate all comparison plots and return list of saved file paths."""
    os.makedirs(output_dir, exist_ok=True)

    saved = []

    print("\n[plots] Generating comparison graphs...")

    # 1. Accuracy bar chart
    p = plot_accuracy_comparison(before_agg, after_agg, output_dir)
    if p:
        saved.append(p)

    # 2. Log-likelihood
    p = plot_log_likelihood_comparison(comparison, output_dir)
    saved.append(p)

    # 3. Confidence
    p = plot_confidence_comparison(comparison, output_dir)
    saved.append(p)

    # 4. Entropy
    p = plot_entropy_comparison(comparison, output_dir)
    saved.append(p)

    # 5. Uncertainty breakdown
    p = plot_uncertainty_breakdown(comparison, output_dir)
    if p:
        saved.append(p)

    # 6. Correctness heatmap
    p = plot_correctness_heatmap(comparison, output_dir)
    if p:
        saved.append(p)

    # 7. Training loss
    p = plot_training_loss(train_losses, val_losses, output_dir)
    if p:
        saved.append(p)

    # 8. Summary dashboard
    p = plot_summary_dashboard(
        comparison, before_agg, after_agg,
        train_losses, val_losses, output_dir,
    )
    saved.append(p)

    print(f"[plots] Done — {len(saved)} graphs saved to {output_dir}/")

    return saved
