"""
Visualization utilities for generating plots and figures for research paper.

This module provides functions to create publication-quality plots comparing
the performance of different control methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import pandas as pd
from pathlib import Path


# Set publication-quality style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"


def plot_reward_comparison(
    results: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot box plot comparing episode rewards across methods.

    Args:
        results: Dictionary with results for each method
        save_path: Path to save figure
        show: Whether to display figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data
    data = []
    for method, method_results in results.items():
        rewards = method_results.get("episode_rewards", [])
        for reward in rewards:
            data.append({"Method": method, "Episode Reward": reward})

    df = pd.DataFrame(data)

    # Create box plot
    sns.boxplot(x="Method", y="Episode Reward", data=df, ax=ax, palette="Set2")
    sns.stripplot(
        x="Method",
        y="Episode Reward",
        data=df,
        ax=ax,
        color="black",
        alpha=0.3,
        size=3,
    )

    ax.set_title("Episode Reward Comparison", fontsize=16, fontweight="bold")
    ax.set_xlabel("Control Method", fontsize=14)
    ax.set_ylabel("Episode Reward", fontsize=14)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        print(f"Saved reward comparison to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_tracking_error_comparison(
    results: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot tracking error comparison across methods.

    Args:
        results: Dictionary with results for each method
        save_path: Path to save figure
        show: Whether to display figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data
    data = []
    for method, method_results in results.items():
        errors = method_results.get("tracking_errors", [])
        for error in errors:
            data.append({"Method": method, "Mean Tracking Error": error})

    df = pd.DataFrame(data)

    # Create box plot
    sns.boxplot(x="Method", y="Mean Tracking Error", data=df, ax=ax, palette="Set2")
    sns.stripplot(
        x="Method",
        y="Mean Tracking Error",
        data=df,
        ax=ax,
        color="black",
        alpha=0.3,
        size=3,
    )

    ax.set_title(
        "Tracking Error Comparison (Lower is Better)", fontsize=16, fontweight="bold"
    )
    ax.set_xlabel("Control Method", fontsize=14)
    ax.set_ylabel("Mean Distance to Target", fontsize=14)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        print(f"Saved tracking error comparison to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_success_rate_comparison(
    results: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot bar chart comparing success rates.

    Args:
        results: Dictionary with results for each method
        save_path: Path to save figure
        show: Whether to display figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(results.keys())
    success_rates = [
        results[method].get("mean_success_rate", 0) * 100 for method in methods
    ]
    success_stds = [
        results[method].get("std_success_rate", 0) * 100 for method in methods
    ]

    # Create bar plot
    bars = ax.bar(methods, success_rates, yerr=success_stds, capsize=10, palette="Set2")

    # Color bars
    colors = sns.color_palette("Set2", len(methods))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_title("Success Rate Comparison", fontsize=16, fontweight="bold")
    ax.set_xlabel("Control Method", fontsize=14)
    ax.set_ylabel("Success Rate (%)", fontsize=14)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, (bar, val, std) in enumerate(zip(bars, success_rates, success_stds)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std + 2,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    if save_path:
        plt.savefig(save_path)
        print(f"Saved success rate comparison to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_distance_over_time(
    results: Dict[str, Dict[str, Any]],
    episode_idx: int = 0,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot distance to target over time for a single episode.

    Args:
        results: Dictionary with results for each method
        episode_idx: Index of episode to plot
        save_path: Path to save figure
        show: Whether to display figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = sns.color_palette("Set2", len(results))

    for (method, method_results), color in zip(results.items(), colors):
        distances_over_time = method_results.get("distances_over_time", [])
        if episode_idx < len(distances_over_time):
            distances = distances_over_time[episode_idx]
            timesteps = np.arange(len(distances))
            ax.plot(timesteps, distances, label=method, linewidth=2, color=color)

    ax.set_title(
        f"Distance to Target Over Time (Episode {episode_idx})",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Timestep", fontsize=14)
    ax.set_ylabel("Distance to Target", fontsize=14)
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        print(f"Saved distance over time plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_learning_curve(
    tensorboard_log_dir: str,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot SAC learning curve from tensorboard logs.

    Args:
        tensorboard_log_dir: Path to tensorboard log directory
        save_path: Path to save figure
        show: Whether to display figure
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Load tensorboard data
        ea = event_accumulator.EventAccumulator(tensorboard_log_dir)
        ea.Reload()

        # Plot episode reward
        if "rollout/ep_rew_mean" in ea.Tags()["scalars"]:
            reward_data = ea.Scalars("rollout/ep_rew_mean")
            steps = [x.step for x in reward_data]
            rewards = [x.value for x in reward_data]
            ax1.plot(steps, rewards, linewidth=2, color="steelblue")
            ax1.set_title("SAC Learning Curve: Episode Reward", fontsize=16)
            ax1.set_xlabel("Timestep", fontsize=14)
            ax1.set_ylabel("Mean Episode Reward", fontsize=14)
            ax1.grid(True, alpha=0.3)

        # Plot episode length
        if "rollout/ep_len_mean" in ea.Tags()["scalars"]:
            length_data = ea.Scalars("rollout/ep_len_mean")
            steps = [x.step for x in length_data]
            lengths = [x.value for x in length_data]
            ax2.plot(steps, lengths, linewidth=2, color="coral")
            ax2.set_title("SAC Learning Curve: Episode Length", fontsize=16)
            ax2.set_xlabel("Timestep", fontsize=14)
            ax2.set_ylabel("Mean Episode Length", fontsize=14)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Saved learning curve to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    except Exception as e:
        print(f"Error plotting learning curve: {e}")


def generate_all_plots(
    results: Dict[str, Dict[str, Any]],
    output_dir: str = "results/figures",
    show: bool = False,
):
    """
    Generate all publication-ready plots.

    Args:
        results: Dictionary with results for each method
        output_dir: Directory to save figures
        show: Whether to display figures
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\nGenerating publication-ready figures...")

    # 1. Reward comparison
    plot_reward_comparison(
        results,
        save_path=f"{output_dir}/reward_comparison.png",
        show=show,
    )

    # 2. Tracking error comparison
    plot_tracking_error_comparison(
        results,
        save_path=f"{output_dir}/tracking_error_comparison.png",
        show=show,
    )

    # 3. Success rate comparison
    plot_success_rate_comparison(
        results,
        save_path=f"{output_dir}/success_rate_comparison.png",
        show=show,
    )

    # 4. Distance over time (first few episodes)
    for episode_idx in range(min(3, len(results[list(results.keys())[0]]["distances_over_time"]))):
        plot_distance_over_time(
            results,
            episode_idx=episode_idx,
            save_path=f"{output_dir}/distance_over_time_ep{episode_idx}.png",
            show=show,
        )

    print(f"\nAll figures saved to {output_dir}/")


def create_latex_table(results: Dict[str, Dict[str, Any]], output_path: str):
    """
    Create a LaTeX table for the research paper.

    Args:
        results: Dictionary with results for each method
        output_path: Path to save LaTeX table
    """
    latex_code = r"""\begin{table}[h]
\centering
\caption{Performance Comparison of Control Methods}
\label{tab:performance_comparison}
\begin{tabular}{lccc}
\toprule
\textbf{Metric} & \textbf{PID} & \textbf{Kalman-PID} & \textbf{SAC (Deep RL)} \\
\midrule
"""

    metrics = [
        ("Mean Reward", "mean_reward", "{:.2f}"),
        ("Tracking Error", "mean_tracking_error", "{:.3f}"),
        ("Success Rate (\\%)", "mean_success_rate", "{:.1%}"),
        ("Episode Length", "mean_length", "{:.1f}"),
    ]

    for metric_name, metric_key, fmt in metrics:
        line = metric_name
        for method in ["PID", "Kalman-PID", "SAC"]:
            if method in results:
                value = results[method].get(metric_key, 0)
                line += " & " + fmt.format(value)
            else:
                line += " & N/A"
        line += " \\\\\n"
        latex_code += line

    latex_code += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_path, "w") as f:
        f.write(latex_code)

    print(f"LaTeX table saved to {output_path}")
