"""
C4 Token-Level Evaluation — Matplotlib visualization module.

Generates comparison graphs from the C4 evaluation results produced by
c4_eval.py.  All values come from actual model outputs.

Output: PNG files saved to the specified output directory.
"""

import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

COLOR_BEFORE = "#4A90D9"    # steel blue
COLOR_AFTER  = "#2ECC71"    # emerald green
COLOR_ACCENT = "#E74C3C"    # red for emphasis
COLOR_GRID   = "#ECECEC"
COLOR_EPIST  = "#9B59B6"    # purple
COLOR_ALEAT  = "#F39C12"    # orange
COLOR_PRED   = "#1ABC9C"    # teal

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
# 1. Perplexity comparison
# ---------------------------------------------------------------------------

def plot_perplexity_comparison(
    aggregate: Dict[str, Any],
    output_dir: str,
) -> Optional[str]:
    """Bar chart: Before vs After perplexity (lower is better)."""
    bp = aggregate.get("before_perplexity")
    ap = aggregate.get("after_perplexity")
    if bp is None:
        return None

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    after_label = aggregate.get("after_label", "DoLa + ENN")
    labels = ["Before\n(DoLa only)", f"After\n({after_label})"]
    values = [bp, ap if ap is not None else 0]
    colors = [COLOR_BEFORE, COLOR_AFTER]

    n_bars = 2 if ap is not None else 1
    bars = ax.bar(labels[:n_bars], values[:n_bars], color=colors[:n_bars],
                  width=0.5, edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, values[:n_bars]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=12)

    if ap is not None:
        change = ap - bp
        change_pct = (change / bp) * 100 if bp != 0 else 0
        ax.text(0.95, 0.95, f"Change: {change:+.2f} ({change_pct:+.1f}%)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=11, color=COLOR_AFTER if change < 0 else COLOR_ACCENT,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor=COLOR_GRID, alpha=0.9))

    _style_ax(ax, "Perplexity Comparison (↓ lower is better)",
              ylabel="Perplexity")
    plt.tight_layout()
    path = os.path.join(output_dir, "c4_perplexity_comparison.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# 2. Token accuracy comparison
# ---------------------------------------------------------------------------

def plot_accuracy_comparison(
    aggregate: Dict[str, Any],
    output_dir: str,
) -> Optional[str]:
    """Grouped bar chart: Top-1 and Top-5 accuracy, Before vs After."""
    bt1 = aggregate.get("before_top1_accuracy")
    bt5 = aggregate.get("before_top5_accuracy")
    if bt1 is None:
        return None

    at1 = aggregate.get("after_top1_accuracy")
    at5 = aggregate.get("after_top5_accuracy")

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    x = np.arange(2)
    width = 0.30

    before_vals = [bt1 * 100, bt5 * 100]
    bars1 = ax.bar(x - width/2, before_vals, width, label="Before (DoLa)",
                   color=COLOR_BEFORE, edgecolor="white", linewidth=1.5)

    if at1 is not None:
        after_vals = [at1 * 100, at5 * 100]
        after_label = aggregate.get("after_label", "DoLa + ENN")
        bars2 = ax.bar(x + width/2, after_vals, width, label=f"After ({after_label})",
                       color=COLOR_AFTER, edgecolor="white", linewidth=1.5)

        for bar, val in zip(bars2, after_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    for bar, val in zip(bars1, before_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(["Top-1 Accuracy", "Top-5 Accuracy"], fontsize=12)
    ax.legend(fontsize=11)
    n_texts = aggregate.get("n_texts")
    n_tokens = aggregate.get("n_tokens")
    if n_texts is not None and n_tokens is not None:
        title = f"Token Prediction Accuracy on Held-Out C4\n{n_texts} samples · {n_tokens:,} scored tokens"
    else:
        title = "Token Prediction Accuracy on Held-Out C4"
    _style_ax(ax, title, ylabel="Accuracy (%)")
    plt.tight_layout()
    path = os.path.join(output_dir, "c4_accuracy_comparison.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved → {path}")
    return path


def plot_dola_layer_selection(aggregate: Dict[str, Any], output_dir: str) -> Optional[str]:
    """Plot how often DoLa selected each premature layer across evaluated tokens."""
    counts = aggregate.get("dola_layer_counts") or {}
    if not counts:
        return None
    layers = sorted(counts, key=lambda layer: int(layer))
    values = [counts[layer] for layer in layers]
    total = sum(values)
    percentages = [value / total * 100 for value in values]
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    bars = ax.bar([f"Layer {layer}" for layer in layers], percentages,
                  color=COLOR_EPIST, edgecolor="white", linewidth=1.5)
    for bar, pct, count in zip(bars, percentages, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{pct:.1f}%\n(n={count})", ha="center", va="bottom", fontsize=9)
    _style_ax(ax, "DoLa Premature Layer Selection on Held-Out C4",
              ylabel="Selected token positions (%)", xlabel=f"Evaluated positions (n={total})")
    ax.set_ylim(0, max(percentages) * 1.2 + 1)
    plt.tight_layout()
    path = os.path.join(output_dir, "c4_dola_layer_selection.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# 3. Log-likelihood comparison
# ---------------------------------------------------------------------------

def plot_log_likelihood_comparison(
    aggregate: Dict[str, Any],
    output_dir: str,
) -> Optional[str]:
    """Bar chart: Mean log-likelihood Before vs After."""
    bll = aggregate.get("before_mean_ll")
    if bll is None:
        return None

    all_ = aggregate.get("after_mean_ll")

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    after_label = aggregate.get("after_label", "DoLa + ENN")
    labels = ["Before\n(DoLa only)", f"After\n({after_label})"]
    values = [bll, all_ if all_ is not None else 0]
    colors = [COLOR_BEFORE, COLOR_AFTER]

    n_bars = 2 if all_ is not None else 1
    bars = ax.bar(labels[:n_bars], values[:n_bars], color=colors[:n_bars],
                  width=0.5, edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, values[:n_bars]):
        if val < 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                    f"{val:.4f}", ha="center", va="center", color="white",
                    fontweight="bold", fontsize=11)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=11)

    if all_ is not None:
        change = all_ - bll
        ax.text(0.95, 0.05, f"Change: {change:+.4f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=11, color=COLOR_AFTER if change > 0 else COLOR_ACCENT,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor=COLOR_GRID, alpha=0.9))

    _style_ax(ax, "Mean Token Log-Likelihood on Held-Out C4 (↑ higher is better)",
              ylabel="Mean Log-Likelihood")
    plt.tight_layout()
    path = os.path.join(output_dir, "c4_log_likelihood_comparison.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# 4. Uncertainty calibration curve
# ---------------------------------------------------------------------------

def plot_uncertainty_calibration(
    aggregate: Dict[str, Any],
    output_dir: str,
    n_bins: int = 10,
) -> Optional[str]:
    """
    Calibration curve: bins tokens by ENN epistemic uncertainty
    and shows the actual error rate per bin.

    If ENN is well-calibrated, higher uncertainty → higher error rate.
    """
    epist = aggregate.get("_calibration_epistemic")
    errors = aggregate.get("_calibration_errors")
    if epist is None or errors is None:
        return None

    epist = np.array(epist)
    errors = np.array(errors)

    if len(epist) == 0:
        return None

    # Bin by epistemic uncertainty quantiles
    try:
        bin_edges = np.quantile(epist, np.linspace(0, 1, n_bins + 1))
        # Ensure unique edges
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 3:
            bin_edges = np.linspace(epist.min(), epist.max(), n_bins + 1)
    except Exception:
        bin_edges = np.linspace(epist.min(), epist.max(), n_bins + 1)

    bin_centers = []
    bin_error_rates = []
    bin_counts = []

    for i in range(len(bin_edges) - 1):
        if i == len(bin_edges) - 2:
            mask = (epist >= bin_edges[i]) & (epist <= bin_edges[i + 1])
        else:
            mask = (epist >= bin_edges[i]) & (epist < bin_edges[i + 1])

        if mask.sum() == 0:
            continue

        bin_centers.append(float(np.mean(epist[mask])))
        bin_error_rates.append(float(np.mean(errors[mask])))
        bin_counts.append(int(mask.sum()))

    if not bin_centers:
        return None

    fig, ax1 = plt.subplots(figsize=FIGSIZE_SINGLE)

    # Error rate line
    ax1.plot(bin_centers, bin_error_rates, "o-", color=COLOR_ACCENT,
             linewidth=2.5, markersize=8, label="Error Rate", zorder=3)
    ax1.fill_between(bin_centers, bin_error_rates, alpha=0.15, color=COLOR_ACCENT)

    _style_ax(ax1, "ENN Uncertainty Calibration\n(Higher uncertainty should → higher error rate)",
              ylabel="Token Error Rate", xlabel="Mean Epistemic Uncertainty (per bin)")
    ax1.set_ylim(bottom=0)

    # Token count on secondary axis
    ax2 = ax1.twinx()
    ax2.bar(bin_centers, bin_counts, width=(bin_centers[-1] - bin_centers[0]) / len(bin_centers) * 0.6,
            alpha=0.25, color=COLOR_BEFORE, label="Token Count", zorder=1)
    ax2.set_ylabel("Token Count", fontsize=11, color=COLOR_BEFORE)
    ax2.tick_params(axis="y", labelcolor=COLOR_BEFORE)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    plt.tight_layout()
    path = os.path.join(output_dir, "c4_uncertainty_calibration.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# 5. Uncertainty breakdown
# ---------------------------------------------------------------------------

def plot_uncertainty_breakdown(
    aggregate: Dict[str, Any],
    output_dir: str,
) -> Optional[str]:
    """Stacked bar showing epistemic vs aleatoric uncertainty."""
    me = aggregate.get("mean_epistemic")
    ma = aggregate.get("mean_aleatoric")
    mp = aggregate.get("mean_predictive")
    if me is None:
        return None

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    categories = ["Uncertainty\nDecomposition"]
    bars_epist = ax.bar(categories, [me], width=0.4,
                        color=COLOR_EPIST, label=f"Epistemic ({me:.4f})",
                        edgecolor="white", linewidth=1.5)
    bars_aleat = ax.bar(categories, [ma], width=0.4, bottom=[me],
                        color=COLOR_ALEAT, label=f"Aleatoric ({ma:.4f})",
                        edgecolor="white", linewidth=1.5)

    # Predictive total annotation
    ax.axhline(y=mp, color=COLOR_PRED, linestyle="--", linewidth=2,
               label=f"Predictive Total ({mp:.4f})")

    ax.legend(fontsize=11, loc="upper right")
    _style_ax(ax, "ENN Uncertainty Breakdown on Held-Out C4",
              ylabel="Uncertainty")
    plt.tight_layout()
    path = os.path.join(output_dir, "c4_uncertainty_breakdown.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# 6. Per-text perplexity scatter
# ---------------------------------------------------------------------------

def plot_per_text_perplexity_scatter(
    aggregate: Dict[str, Any],
    output_dir: str,
) -> Optional[str]:
    """Scatter plot: each text's perplexity Before (x) vs After (y)."""
    bp = aggregate.get("before_per_text_ppl")
    ap = aggregate.get("after_per_text_ppl")
    if bp is None or ap is None:
        return None

    bp = np.array(bp)
    ap = np.array(ap)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(bp, ap, alpha=0.6, color=COLOR_AFTER, edgecolors="white",
               linewidth=0.5, s=50, zorder=3)

    # Diagonal reference (no change line)
    lims = [min(bp.min(), ap.min()) * 0.9, max(bp.max(), ap.max()) * 1.1]
    ax.plot(lims, lims, "--", color="#888888", linewidth=1.5, alpha=0.7,
            label="No change line")

    # Count improved vs degraded
    n_improved = int(np.sum(ap < bp))
    n_degraded = int(np.sum(ap > bp))
    n_same = int(np.sum(ap == bp))

    ax.text(0.05, 0.95,
            f"Improved: {n_improved}  |  Degraded: {n_degraded}  |  Same: {n_same}",
            transform=ax.transAxes, ha="left", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=COLOR_GRID, alpha=0.9))

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.legend(fontsize=10)
    after_label = aggregate.get("after_label", "DoLa + ENN")
    _style_ax(ax, "Per-Text Perplexity: Before vs After",
              xlabel="Perplexity (Before — DoLa only)",
              ylabel=f"Perplexity (After — {after_label})")
    plt.tight_layout()
    path = os.path.join(output_dir, "c4_perplexity_scatter.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# 7. Summary dashboard
# ---------------------------------------------------------------------------

def plot_summary_dashboard(
    aggregate: Dict[str, Any],
    output_dir: str,
) -> Optional[str]:
    """Combined multi-panel summary figure."""
    has_after = aggregate.get("after_perplexity") is not None

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("C4 Token-Level Evaluation — Summary Dashboard",
                 fontsize=16, fontweight="bold", y=0.98)

    # Panel 1: Perplexity
    ax = axes[0, 0]
    labels = ["Before\n(DoLa)"]
    vals = [aggregate["before_perplexity"]]
    colors = [COLOR_BEFORE]
    if has_after:
        labels.append(f"After\n({aggregate.get('after_label', 'DoLa + ENN')})")
        vals.append(aggregate["after_perplexity"])
        colors.append(COLOR_AFTER)
    bars = ax.bar(labels, vals, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    _style_ax(ax, "Perplexity (↓ lower is better)", ylabel="Perplexity")

    # Panel 2: Token accuracy
    ax = axes[0, 1]
    x = np.arange(2)
    width = 0.30
    b_vals = [aggregate["before_top1_accuracy"] * 100,
              aggregate["before_top5_accuracy"] * 100]
    ax.bar(x - width/2, b_vals, width, label="Before",
           color=COLOR_BEFORE, edgecolor="white", linewidth=1.5)
    if has_after:
        a_vals = [aggregate["after_top1_accuracy"] * 100,
                  aggregate["after_top5_accuracy"] * 100]
        ax.bar(x + width/2, a_vals, width, label="After",
               color=COLOR_AFTER, edgecolor="white", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Top-1", "Top-5"], fontsize=11)
    ax.legend(fontsize=10)
    _style_ax(ax, "Token Prediction Accuracy (↑ higher is better)",
              ylabel="Accuracy (%)")

    # Panel 3: Log-likelihood
    ax = axes[1, 0]
    labels_ll = ["Before\n(DoLa)"]
    vals_ll = [aggregate["before_mean_ll"]]
    colors_ll = [COLOR_BEFORE]
    if has_after:
        labels_ll.append(f"After\n({aggregate.get('after_label', 'DoLa + ENN')})")
        vals_ll.append(aggregate["after_mean_ll"])
        colors_ll.append(COLOR_AFTER)
    bars = ax.bar(labels_ll, vals_ll, color=colors_ll, width=0.5,
                  edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, vals_ll):
        if val < 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                    f"{val:.4f}", ha="center", va="center", color="white",
                    fontweight="bold", fontsize=10)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    _style_ax(ax, "Mean Token Log-Likelihood (↑ higher is better)",
              ylabel="Log-Likelihood")

    # Panel 4: Uncertainty breakdown or stats
    ax = axes[1, 1]
    if has_after and aggregate.get("mean_epistemic") is not None:
        me = aggregate["mean_epistemic"]
        ma = aggregate["mean_aleatoric"]
        mp = aggregate["mean_predictive"]

        unc_labels = ["Epistemic", "Aleatoric", "Predictive"]
        unc_vals = [me, ma, mp]
        unc_colors = [COLOR_EPIST, COLOR_ALEAT, COLOR_PRED]
        bars = ax.bar(unc_labels, unc_vals, color=unc_colors, width=0.5,
                      edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, unc_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=10)
        _style_ax(ax, "ENN Uncertainty Breakdown", ylabel="Uncertainty")
    else:
        # Summary text panel
        ax.axis("off")
        summary_text = (
            f"Texts: {aggregate['n_texts']}\n"
            f"Tokens: {aggregate['n_tokens']}\n"
            f"PPL: {aggregate['before_perplexity']:.2f}"
        )
        if has_after:
            summary_text += f" → {aggregate['after_perplexity']:.2f}"
        ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
                ha="center", va="center", fontsize=14,
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=1", facecolor="#F8F8F8",
                          edgecolor=COLOR_GRID))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, "c4_summary_dashboard.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Public API: generate all plots
# ---------------------------------------------------------------------------

def generate_c4_plots(
    aggregate: Dict[str, Any],
    output_dir: str,
) -> List[str]:
    """
    Generate all C4 evaluation plots.

    Returns list of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[plots] Generating C4 evaluation graphs...")

    saved = []
    generators = [
        plot_perplexity_comparison,
        plot_accuracy_comparison,
        plot_dola_layer_selection,
        plot_log_likelihood_comparison,
        plot_uncertainty_calibration,
        plot_uncertainty_breakdown,
        plot_per_text_perplexity_scatter,
        plot_summary_dashboard,
    ]

    for gen_fn in generators:
        try:
            path = gen_fn(aggregate, output_dir)
            if path:
                saved.append(path)
        except Exception as e:
            print(f"[plots] Warning: {gen_fn.__name__} failed: {e}")

    print(f"[plots] Done — {len(saved)} graphs saved to {output_dir}/")
    return saved
