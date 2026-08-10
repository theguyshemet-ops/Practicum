"""
main_experiment.py - Single-command entry point for the full experimental pipeline.

Chains:
1. run_experiments   -> Optimise patches + evaluate across bands/epsilons
2. aggregate_results -> Build comparison tables + LaTeX output
3. visualize_results -> Generate publication-quality figures

Usage:
    python -m src.experiments.main_experiment --data_dir data/nuscenes --results_dir results

    # Skip optimisation (use existing patches):
    python -m src.experiments.main_experiment --data_dir data/nuscenes --results_dir results --skip_optim

    # Quick smoke test (5 steps):
    python -m src.experiments.main_experiment --data_dir data/nuscenes --results_dir results --num_steps 5
"""

import argparse
import os
import sys
import time

# Ensure project root is on sys.path so `from src.xxx` imports work
# regardless of which directory the script is run from.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ViT vs CNN Adversarial Robustness - Full Experiment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full experiment (300 steps):
    python -m src.experiments.main_experiment --data_dir data/nuscenes --results_dir results

  Quick test (5 steps):
    python -m src.experiments.main_experiment --data_dir data/nuscenes --results_dir results --num_steps 5

  Skip optimisation, regenerate figures:
    python -m src.experiments.main_experiment --data_dir data/nuscenes --results_dir results --skip_optim
        """
    )

    parser.add_argument(
        "--data_dir", type=str, default="data/nuscenes",
        help="Path to nuScenes 2D data directory (default: data/nuscenes)"
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Output directory for all results (default: results)"
    )
    parser.add_argument(
        "--num_steps", type=int, default=300,
        help="PGD optimisation steps per campaign (default: 300)"
    )
    parser.add_argument(
        "--bands", type=str, default="low,high,full",
        help="Comma-separated frequency bands (default: low,high,full)"
    )
    parser.add_argument(
        "--epsilons", type=str, default="0.1,0.2,0.3,0.5",
        help="Comma-separated epsilon values for sweep (default: 0.1,0.2,0.3,0.5)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device: 'cuda' or 'cpu' (default: cuda)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2,
        help="Batch size for optimisation (default: 2)"
    )
    parser.add_argument(
        "--skip_optim", action="store_true",
        help="Skip optimisation, use existing patch files"
    )
    parser.add_argument(
        "--skip_eval", action="store_true",
        help="Skip evaluation, use existing JSON results"
    )
    parser.add_argument(
        "--skip_viz", action="store_true",
        help="Skip visualisation generation"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    bands = [b.strip() for b in args.bands.split(",")]
    epsilons = [float(e.strip()) for e in args.epsilons.split(",")]

    start_time = time.time()

    print("+" + "=" * 70 + "+")
    print("|  ViT vs CNN Adversarial Robustness - Main Experiment Pipeline       |")
    print("+" + "=" * 70 + "+")
    print(f"  Data dir:    {args.data_dir}")
    print(f"  Results dir: {args.results_dir}")
    print(f"  Steps:       {args.num_steps}")
    print(f"  Bands:       {bands}")
    print(f"  Epsilons:    {epsilons}")
    print(f"  Device:      {args.device}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Skip optim:  {args.skip_optim}")
    print(f"  Skip eval:   {args.skip_eval}")
    print(f"  Skip viz:    {args.skip_viz}")
    print()

    # ===================================================================
    # Stage 1: Run Experiments (Optimise + Evaluate)
    # ===================================================================
    print("\n" + "=" * 72)
    print("  STAGE 1: Experiment Execution")
    print("=" * 72)

    from src.experiments.run_experiments import run_full_experiment

    all_patches = run_full_experiment(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        num_steps=args.num_steps,
        bands=bands,
        epsilons=epsilons,
        device=args.device,
        batch_size=args.batch_size,
        skip_optim=args.skip_optim,
        skip_eval=args.skip_eval,
    )

    # ===================================================================
    # Stage 2: Aggregate Results
    # ===================================================================
    print("\n" + "=" * 72)
    print("  STAGE 2: Results Aggregation")
    print("=" * 72)

    from src.experiments.aggregate_results import aggregate_all

    summary = aggregate_all(args.results_dir)

    # ===================================================================
    # Stage 3: Generate Visualisations
    # ===================================================================
    if not args.skip_viz:
        print("\n" + "=" * 72)
        print("  STAGE 3: Visualisation Generation")
        print("=" * 72)

        try:
            from src.experiments.visualize_results import generate_all_figures

            figures_dir = os.path.join(args.results_dir, "figures")
            generate_all_figures(
                results_dir=args.results_dir,
                figures_dir=figures_dir,
            )
        except ImportError as e:
            print(f"\n  [WARN] Could not import visualize_results: {e}")
            print("  Skipping visualisation generation.")
        except Exception as e:
            print(f"\n  [ERROR] Visualisation failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n  [SKIP] Visualisation generation skipped (--skip_viz)")

    # ===================================================================
    # Done
    # ===================================================================
    elapsed = time.time() - start_time

    print("\n" + "+" + "=" * 70 + "+")
    print("|  Pipeline Complete!                                                  |")
    print("+" + "=" * 70 + "+")
    print(f"  Total elapsed:  {elapsed / 60:.1f} minutes ({elapsed:.0f} seconds)")
    print(f"  Results:        {os.path.abspath(args.results_dir)}")

    # List output files
    for dirpath, dirnames, filenames in os.walk(args.results_dir):
        for fname in sorted(filenames):
            fpath = os.path.join(dirpath, fname)
            size_kb = os.path.getsize(fpath) / 1024
            rel_path = os.path.relpath(fpath, args.results_dir)
            print(f"    {rel_path:<50} {size_kb:>8.1f} KB")

    print()


if __name__ == "__main__":
    main()
