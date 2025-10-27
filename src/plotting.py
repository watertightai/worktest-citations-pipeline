"""
Plotting utilities for evaluation results.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_model_comparison(
    model_metrics: Dict[str, Dict],
    output_path: Optional[Path] = None,
    metrics_to_plot: List[str] = ["accuracy", "brier_score", "auroc"],
    title: str = "Model Comparison",
    baseline_accuracy: Optional[float] = None,
    confidence_intervals: Optional[Dict] = None,
    figsize: tuple = (12, 5),
):
    """
    Create comparison plot showing multiple metrics across models.

    Args:
        model_metrics: Dict mapping model_id -> metrics dict
        output_path: Where to save plot (if None, displays instead)
        metrics_to_plot: Which metrics to include
        title: Plot title
        baseline_accuracy: Optional baseline to show on accuracy plot
        confidence_intervals: Optional dict of confidence intervals per model
        figsize: Figure size (width, height)

    Example:
        >>> metrics = {
        ...     "claude-3-5-haiku": {"accuracy": 0.65, "brier_score": 0.24, "auroc": 0.71},
        ...     "claude-sonnet-4-5": {"accuracy": 0.78, "brier_score": 0.19, "auroc": 0.83}
        ... }
        >>> plot_model_comparison(metrics, Path("comparison.png"))
    """
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["font.size"] = 10

    # Determine number of subplots
    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    # Extract model names and shorten them for display
    model_names = list(model_metrics.keys())
    short_names = [_shorten_model_name(name) for name in model_names]

    # Plot each metric
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]

        # Extract values for this metric
        values = []
        lower_bounds = []
        upper_bounds = []

        for model_name in model_names:
            value = model_metrics[model_name].get(metric)
            values.append(value if value is not None else 0)

            # Add confidence intervals if available
            if confidence_intervals and model_name in confidence_intervals:
                intervals = confidence_intervals[model_name].get(metric, (None, None))
                if intervals[0] is not None:
                    lower_bounds.append(value - intervals[0])
                    upper_bounds.append(intervals[1] - value)
                else:
                    lower_bounds.append(0)
                    upper_bounds.append(0)
            else:
                lower_bounds.append(0)
                upper_bounds.append(0)

        # Create bar chart
        x_pos = np.arange(len(short_names))
        colors = sns.color_palette("husl", len(short_names))

        # Add error bars if we have confidence intervals
        if confidence_intervals:
            ax.bar(
                x_pos,
                values,
                color=colors,
                alpha=0.8,
                yerr=[lower_bounds, upper_bounds],
                capsize=5,
            )
        else:
            ax.bar(x_pos, values, color=colors, alpha=0.8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(short_names, rotation=45, ha="right")

        # Metric-specific formatting
        if metric == "accuracy":
            ax.set_ylabel("Accuracy")
            ax.set_ylim([0, 1])
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

            # Add baseline if provided
            if baseline_accuracy is not None:
                ax.axhline(
                    y=baseline_accuracy,
                    color="red",
                    linestyle="--",
                    alpha=0.5,
                    label=f"Baseline: {baseline_accuracy:.1%}",
                )
                ax.legend()

        elif metric == "brier_score":
            ax.set_ylabel("Brier Score")
            ax.set_ylim([0, max(values) * 1.2])
            ax.invert_yaxis()  # Lower is better

        elif metric == "auroc":
            ax.set_ylabel("AUROC")
            ax.set_ylim([0.5, 1.0])

        elif metric == "log_loss":
            ax.set_ylabel("Log Loss")
            ax.invert_yaxis()  # Lower is better

        # Add value labels on bars
        for i, (v, name) in enumerate(zip(values, short_names)):
            if metric in ["accuracy", "auroc"]:
                label = f"{v:.2%}" if v > 0 else "N/A"
            else:
                label = f"{v:.3f}" if v > 0 else "N/A"

            # Position label above bar
            y_pos = v if metric not in ["brier_score", "log_loss"] else v
            ax.text(i, y_pos, label, ha="center", va="bottom", fontsize=9)

        ax.set_title(metric.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.3)

    # Overall title
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    # Save or show
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def _shorten_model_name(name: str) -> str:
    """
    Shorten model names for cleaner display.

    Examples:
        claude-3-5-haiku-latest -> Haiku 3.5
        claude-3-7-sonnet-20250219 -> Sonnet 3.7
        claude-sonnet-4-5-20250929 -> Sonnet 4.5
    """
    name_lower = name.lower()

    # Claude models
    if "claude" in name_lower:
        if "haiku" in name_lower:
            if "3-5" in name or "3.5" in name:
                return "Haiku 3.5"
            elif "3" in name:
                return "Haiku 3"
        elif "sonnet" in name_lower:
            if "4-5" in name or "4.5" in name:
                return "Sonnet 4.5"
            elif "4-0" in name or "4.0" in name or "sonnet-4" in name_lower:
                return "Sonnet 4.0"
            elif "3-7" in name or "3.7" in name:
                return "Sonnet 3.7"
            elif "3-5" in name or "3.5" in name:
                return "Sonnet 3.5"
            elif "3" in name:
                return "Sonnet 3"
        elif "opus" in name_lower:
            if "4" in name:
                return "Opus 4"
            elif "3" in name:
                return "Opus 3"

    # GPT models
    if "gpt" in name_lower:
        if "4o" in name_lower:
            return "GPT-4o"
        elif "4" in name:
            if "turbo" in name_lower:
                return "GPT-4 Turbo"
            return "GPT-4"
        elif "3.5" in name or "3-5" in name:
            return "GPT-3.5"

    # Gemini models
    if "gemini" in name_lower:
        if "2.5" in name or "2-5" in name:
            if "pro" in name_lower:
                return "Gemini 2.5 Pro"
            return "Gemini 2.5 Flash"
        elif "2.0" in name or "2-0" in name:
            if "pro" in name_lower:
                return "Gemini 2.0 Pro"
            return "Gemini 2.0 Flash"
        elif "1.5" in name or "1-5" in name:
            if "pro" in name_lower:
                return "Gemini 1.5 Pro"
            return "Gemini 1.5 Flash"

    # Fallback: return original or truncated
    if len(name) > 20:
        return name[:20] + "..."
    return name