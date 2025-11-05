"""
Comprehensive comparison script that evaluates all methods and generates plots.

This script:
1. Evaluates PID, Kalman-PID, and SAC agents
2. Computes all performance metrics
3. Generates publication-quality figures
4. Creates LaTeX tables for the paper
"""

import os
import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.evaluate import evaluate_all_methods, print_comparison_table, save_results
from utils.visualization import generate_all_plots, create_latex_table


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main comparison function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare all control methods and generate plots"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--sac-model",
        type=str,
        default="models/sac/final_model",
        help="Path to trained SAC model",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=100, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Output directory for results"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--show-plots", action="store_true", help="Show plots interactively"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("COMPREHENSIVE METHOD COMPARISON")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  - Config file: {args.config}")
    print(f"  - SAC model: {args.sac_model}")
    print(f"  - Evaluation episodes: {args.n_episodes}")
    print(f"  - Output directory: {args.output_dir}")
    print(f"  - Random seed: {args.seed}")
    print()

    # Load configuration
    config = load_config(args.config)

    # Evaluate all methods
    print("\n" + "=" * 80)
    print("STEP 1: EVALUATING ALL METHODS")
    print("=" * 80)
    results = evaluate_all_methods(
        config=config,
        sac_model_path=args.sac_model,
        n_episodes=args.n_episodes,
        render=False,
        seed=args.seed,
    )

    # Print comparison table
    print("\n" + "=" * 80)
    print("STEP 2: COMPARISON RESULTS")
    print("=" * 80)
    print_comparison_table(results)

    # Save results
    print("\n" + "=" * 80)
    print("STEP 3: SAVING RESULTS")
    print("=" * 80)
    save_results(results, output_dir=args.output_dir)

    # Generate plots
    print("\n" + "=" * 80)
    print("STEP 4: GENERATING PLOTS")
    print("=" * 80)
    figures_dir = os.path.join(args.output_dir, "figures")
    generate_all_plots(results, output_dir=figures_dir, show=args.show_plots)

    # Create LaTeX table
    print("\n" + "=" * 80)
    print("STEP 5: CREATING LATEX TABLE")
    print("=" * 80)
    latex_path = os.path.join(args.output_dir, "performance_table.tex")
    create_latex_table(results, latex_path)

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {args.output_dir}/")
    print(f"Figures available in: {figures_dir}/")
    print(f"LaTeX table: {latex_path}")
    print("\nReady for research paper integration!")


if __name__ == "__main__":
    main()
