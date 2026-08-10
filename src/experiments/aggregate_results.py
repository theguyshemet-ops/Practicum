"""
aggregate_results.py - Post-process experiment outputs into structured tables.

Reads JSON files from the evaluation/ directory and produces:
1. A unified summary table (JSON) across all experiments
2. Adversarial Delta analysis per band and epsilon
3. Pre-formatted LaTeX table strings for direct dissertation inclusion
"""

import json
import os
from typing import Dict, Optional

import numpy as np


def load_json(path: str) -> Optional[dict]:
    """Load a JSON file, returning None if not found."""
    if not os.path.exists(path):
        print(f"  [WARN] Not found: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


def aggregate_frequency_comparison(results_dir: str) -> Optional[dict]:
    """
    Aggregate the frequency comparison results into a summary table.

    Returns dict with structure:
    {
        "table": [
            {"band": "clean", "rcnn_mAP": ..., "yolos_mAP": ..., "adv_delta": ...},
            {"band": "low", "rcnn_mAP": ..., "yolos_mAP": ..., ...},
            ...
        ]
    }
    """
    freq_path = os.path.join(results_dir, "evaluation", "frequency_comparison.json")
    freq_data = load_json(freq_path)
    if freq_data is None:
        return None

    table = []
    for band in ["clean", "low", "high", "full"]:
        row = {"band": band}
        if band in freq_data.get("rcnn", {}):
            row["rcnn_mAP"] = freq_data["rcnn"][band].get("score", freq_data["rcnn"][band].get("mAP", 0.0))
            row["rcnn_mAP_drop"] = freq_data["rcnn"][band].get("score_drop", freq_data["rcnn"][band].get("mAP_drop", 0.0))
            row["rcnn_avg_det"] = freq_data["rcnn"][band].get("avg_detections", 0.0)
            row["rcnn_avg_conf"] = freq_data["rcnn"][band].get("avg_max_confidence", 0.0)
        if band in freq_data.get("yolos", {}):
            row["yolos_mAP"] = freq_data["yolos"][band].get("score", freq_data["yolos"][band].get("mAP", 0.0))
            row["yolos_mAP_drop"] = freq_data["yolos"][band].get("score_drop", freq_data["yolos"][band].get("mAP_drop", 0.0))
            row["yolos_avg_det"] = freq_data["yolos"][band].get("avg_detections", 0.0)
            row["yolos_avg_conf"] = freq_data["yolos"][band].get("avg_max_confidence", 0.0)

        # Compute Adversarial Delta
        rcnn_mAP = row.get("rcnn_mAP", 0.0)
        yolos_mAP = row.get("yolos_mAP", 0.0)
        row["adversarial_delta"] = yolos_mAP - rcnn_mAP

        table.append(row)

    return {"table": table}


def aggregate_rsf_data(results_dir: str) -> Optional[dict]:
    """Aggregate RSF curve data."""
    rsf_path = os.path.join(results_dir, "evaluation", "rsf_curves.json")
    return load_json(rsf_path)


def aggregate_epsilon_sweep(results_dir: str) -> Optional[dict]:
    """
    Aggregate epsilon sweep data into a structured table.

    Returns dict with structure:
    {
        "table": [
            {"epsilon": 0.1, "band": "low", "rcnn_mAP": ..., "yolos_mAP": ..., "adv_delta": ...},
            ...
        ],
        "by_epsilon": {
            "0.1": {"avg_rcnn_mAP": ..., "avg_yolos_mAP": ...},
            ...
        }
    }
    """
    sweep_path = os.path.join(results_dir, "evaluation", "epsilon_sweep.json")
    sweep_data = load_json(sweep_path)
    if sweep_data is None:
        return None

    table = []
    by_epsilon = {}

    for eps_str, bands in sorted(sweep_data.items(), key=lambda x: float(x[0])):
        eps_val = float(eps_str)
        rcnn_mAPs = []
        yolos_mAPs = []

        for band in ["low", "high", "full"]:
            if band in bands:
                row = {
                    "epsilon": eps_val,
                    "band": band,
                    "rcnn_mAP": bands[band].get("rcnn_score", bands[band].get("rcnn_mAP", 0.0)),
                    "yolos_mAP": bands[band].get("yolos_score", bands[band].get("yolos_mAP", 0.0)),
                    "adversarial_delta": bands[band].get("adversarial_delta", 0.0),
                }
                table.append(row)
                rcnn_mAPs.append(row["rcnn_mAP"])
                yolos_mAPs.append(row["yolos_mAP"])

        by_epsilon[eps_str] = {
            "avg_rcnn_mAP": np.mean(rcnn_mAPs) if rcnn_mAPs else 0.0,
            "avg_yolos_mAP": np.mean(yolos_mAPs) if yolos_mAPs else 0.0,
        }

    return {"table": table, "by_epsilon": by_epsilon}


# =======================================================================
# LaTeX Table Generation
# =======================================================================

def generate_latex_frequency_table(freq_summary: dict) -> str:
    """
    Generate a LaTeX table for the frequency band comparison.

    Output format:
    \\begin{table}[h]
    \\centering
    \\caption{...}
    \\begin{tabular}{l|cc|cc|c}
    Band & R-CNN mAP & R-CNN Drop & YOLOS mAP & YOLOS Drop & Adv. $\\Delta$ \\\\
    \\hline
    Clean & 0.452 & - & 0.389 & - & -0.063 \\\\
    ...
    \\end{tabular}
    \\end{table}
    """
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Frequency Band Comparison: mAP Under Adversarial Patches ($\varepsilon = 0.3$)}",
        r"\label{tab:freq_comparison}",
        r"\begin{tabular}{l|cc|cc|c}",
        r"\toprule",
        r"Band & R-CNN mAP & R-CNN $\Delta$mAP & YOLOS mAP & YOLOS $\Delta$mAP & Adv. $\Delta$ \\",
        r"\midrule",
    ]

    for row in freq_summary["table"]:
        band = row["band"].capitalize()
        rcnn_mAP = row.get("rcnn_mAP", 0.0)
        rcnn_drop = row.get("rcnn_mAP_drop", 0.0)
        yolos_mAP = row.get("yolos_mAP", 0.0)
        yolos_drop = row.get("yolos_mAP_drop", 0.0)
        adv_delta = row.get("adversarial_delta", 0.0)

        if band == "Clean":
            line = f"{band} & {rcnn_mAP:.4f} & -- & {yolos_mAP:.4f} & -- & {adv_delta:+.4f} \\\\"
        else:
            line = (f"{band} & {rcnn_mAP:.4f} & {rcnn_drop:+.4f} & "
                    f"{yolos_mAP:.4f} & {yolos_drop:+.4f} & {adv_delta:+.4f} \\\\")
        lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_latex_epsilon_table(sweep_summary: dict) -> str:
    """
    Generate a LaTeX table for the epsilon sweep results.
    """
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Epsilon Sweep: mAP Under Varying Perturbation Budgets}",
        r"\label{tab:epsilon_sweep}",
        r"\begin{tabular}{cc|cc|c}",
        r"\toprule",
        r"$\varepsilon$ & Band & R-CNN mAP & YOLOS mAP & Adv. $\Delta$ \\",
        r"\midrule",
    ]

    prev_eps = None
    for row in sweep_summary["table"]:
        eps = row["epsilon"]
        band = row["band"].capitalize()
        rcnn_mAP = row.get("rcnn_mAP", 0.0)
        yolos_mAP = row.get("yolos_mAP", 0.0)
        adv_delta = row.get("adversarial_delta", 0.0)

        eps_str = f"{eps:.1f}" if eps != prev_eps else ""
        if eps != prev_eps and prev_eps is not None:
            lines.append(r"\midrule")

        line = f"{eps_str} & {band} & {rcnn_mAP:.4f} & {yolos_mAP:.4f} & {adv_delta:+.4f} \\\\"
        lines.append(line)
        prev_eps = eps

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_latex_rsf_table(rsf_data: dict) -> str:
    """
    Generate a LaTeX table for RSF curve data.
    """
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Robustness Sensitivity Factor (RSF): mAP vs Patch Area Ratio}",
        r"\label{tab:rsf_curves}",
        r"\begin{tabular}{c|cc}",
        r"\toprule",
        r"Patch Ratio & R-CNN mAP & YOLOS mAP \\",
        r"\midrule",
    ]

    rcnn_ratios = rsf_data.get("rcnn", {}).get("ratios", [])
    rcnn_mAPs = rsf_data.get("rcnn", {}).get("mAPs", [])
    yolos_mAPs = rsf_data.get("yolos", {}).get("mAPs", [])

    for i, ratio in enumerate(rcnn_ratios):
        rcnn_mAP = rcnn_mAPs[i] if i < len(rcnn_mAPs) else 0.0
        yolos_mAP = yolos_mAPs[i] if i < len(yolos_mAPs) else 0.0
        lines.append(f"{ratio:.1f} & {rcnn_mAP:.4f} & {yolos_mAP:.4f} \\\\")

    rcnn_rsf = rsf_data.get("rcnn", {}).get("rsf", 0.0)
    yolos_rsf = rsf_data.get("yolos", {}).get("rsf", 0.0)
    lines.extend([
        r"\midrule",
        f"RSF (slope) & {rcnn_rsf:.4f} & {yolos_rsf:.4f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# =======================================================================
# CARLA-Specific Aggregation Functions
# =======================================================================

def aggregate_weather_ablation(results_dir: str) -> Optional[dict]:
    """Aggregate weather ablation results."""
    weather_path = os.path.join(results_dir, "evaluation", "weather_ablation.json")
    return load_json(weather_path)


def aggregate_distance_ablation(results_dir: str) -> Optional[dict]:
    """Aggregate distance ablation results."""
    dist_path = os.path.join(results_dir, "evaluation", "distance_ablation.json")
    return load_json(dist_path)


def aggregate_cross_domain(results_dir: str) -> Optional[dict]:
    """
    Compare cross-domain results (nuScenes vs CARLA).
    Assumes CARLA results are under results_dir (e.g. results/carla)
    and nuScenes results are under results/ (the parent directory).
    """
    carla_freq_path = os.path.join(results_dir, "evaluation", "frequency_comparison.json")
    carla_freq = load_json(carla_freq_path)
    
    parent_dir = os.path.dirname(os.path.abspath(results_dir))
    nuscenes_freq_path = os.path.join(parent_dir, "evaluation", "frequency_comparison.json")
    if not os.path.exists(nuscenes_freq_path):
        nuscenes_freq_path = os.path.join("results", "evaluation", "frequency_comparison.json")
        
    nuscenes_freq = load_json(nuscenes_freq_path)
    if carla_freq is None or nuscenes_freq is None:
        return None
        
    comparison = {"carla": {}, "nuscenes": {}}
    for domain, freq_data in [("carla", carla_freq), ("nuscenes", nuscenes_freq)]:
        for model in ["rcnn", "yolos"]:
            comparison[domain][model] = {}
            for band in ["clean", "low", "high", "full"]:
                if band in freq_data.get(model, {}):
                    # Carla uses score/score_drop, nuscenes uses mAP/mAP_drop
                    comparison[domain][model][band] = {
                        "mAP": freq_data[model][band].get("score", freq_data[model][band].get("mAP", 0.0)),
                        "mAP_drop": freq_data[model][band].get("score_drop", freq_data[model][band].get("mAP_drop", 0.0)),
                    }
    return comparison


def generate_latex_weather_table(weather_summary: dict) -> str:
    """Generate a LaTeX table for weather ablation."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Weather Ablation: mAP Under Clear-Optimised Patches Across Weather Conditions}",
        r"\label{tab:weather_ablation}",
        r"\begin{tabular}{l|cc|cc}",
        r"\toprule",
        r"Weather & R-CNN Clean & R-CNN Patched & YOLOS Clean & YOLOS Patched \\",
        r"\midrule",
    ]
    for weather in ["clear", "rain", "fog"]:
        if weather in weather_summary:
            rcnn_clean = weather_summary[weather]["rcnn"].get("clean_score", weather_summary[weather]["rcnn"].get("clean_mAP", 0.0))
            rcnn_patched = weather_summary[weather]["rcnn"].get("patched_score", weather_summary[weather]["rcnn"].get("patched_mAP", 0.0))
            yolos_clean = weather_summary[weather]["yolos"].get("clean_score", weather_summary[weather]["yolos"].get("clean_mAP", 0.0))
            yolos_patched = weather_summary[weather]["yolos"].get("patched_score", weather_summary[weather]["yolos"].get("patched_mAP", 0.0))
            lines.append(f"{weather.capitalize()} & {rcnn_clean:.4f} & {rcnn_patched:.4f} & {yolos_clean:.4f} & {yolos_patched:.4f} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def generate_latex_distance_table(dist_summary: dict) -> str:
    """Generate a LaTeX table for distance ablation."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Distance Ablation: mAP Under Adversarial Patches Across Distances}",
        r"\label{tab:distance_ablation}",
        r"\begin{tabular}{l|cc|cc}",
        r"\toprule",
        r"Distance & R-CNN Clean & R-CNN Patched & YOLOS Clean & YOLOS Patched \\",
        r"\midrule",
    ]
    sorted_dists = sorted(dist_summary.keys(), key=lambda x: int(x.replace('m', '')) if x.replace('m', '').isdigit() else 0)
    for dist in sorted_dists:
        rcnn_clean = dist_summary[dist]["rcnn"].get("clean_score", dist_summary[dist]["rcnn"].get("clean_mAP", 0.0))
        rcnn_patched = dist_summary[dist]["rcnn"].get("patched_score", dist_summary[dist]["rcnn"].get("patched_mAP", 0.0))
        yolos_clean = dist_summary[dist]["yolos"].get("clean_score", dist_summary[dist]["yolos"].get("clean_mAP", 0.0))
        yolos_patched = dist_summary[dist]["yolos"].get("patched_score", dist_summary[dist]["yolos"].get("patched_mAP", 0.0))
        lines.append(f"{dist} & {rcnn_clean:.4f} & {rcnn_patched:.4f} & {yolos_clean:.4f} & {yolos_patched:.4f} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


# =======================================================================
# Main Aggregation Pipeline
# =======================================================================

def aggregate_all(results_dir: str) -> dict:
    """
    Run the full aggregation pipeline.

    Reads evaluation JSONs and produces:
    1. summary_table.json - unified cross-experiment table
    2. latex_tables.txt - pre-formatted LaTeX tables
    """
    eval_dir = os.path.join(results_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    print("\n" + "=" * 72)
    print("  Results Aggregation")
    print("=" * 72)

    summary = {}

    # -- Frequency Comparison --------------------------------------------
    freq_summary = aggregate_frequency_comparison(results_dir)
    if freq_summary:
        summary["frequency_comparison"] = freq_summary

        print("\n  Frequency Comparison Table:")
        print(f"  {'Band':<10} {'R-CNN mAP':>10} {'R-CNN Drop':>12} {'YOLOS mAP':>10} {'YOLOS Drop':>12} {'Adv Delta':>8}")
        print(f"  {'-' * 62}")
        for row in freq_summary["table"]:
            band = row["band"].capitalize()
            rcnn = row.get("rcnn_mAP", 0.0)
            rcnn_d = row.get("rcnn_mAP_drop", 0.0)
            yolos = row.get("yolos_mAP", 0.0)
            yolos_d = row.get("yolos_mAP_drop", 0.0)
            delta = row.get("adversarial_delta", 0.0)
            print(f"  {band:<10} {rcnn:>10.4f} {rcnn_d:>+12.4f} {yolos:>10.4f} {yolos_d:>+12.4f} {delta:>+8.4f}")

    # -- RSF Data --------------------------------------------------------
    rsf_data = aggregate_rsf_data(results_dir)
    if rsf_data:
        summary["rsf_curves"] = rsf_data
        print(f"\n  RSF Slopes: R-CNN = {rsf_data.get('rcnn', {}).get('rsf', 0):.4f}, "
              f"YOLOS = {rsf_data.get('yolos', {}).get('rsf', 0):.4f}")

    # -- Epsilon Sweep ---------------------------------------------------
    sweep_summary = aggregate_epsilon_sweep(results_dir)
    if sweep_summary:
        summary["epsilon_sweep"] = sweep_summary

        print("\n  Epsilon Sweep Summary:")
        for eps_str, avg in sweep_summary.get("by_epsilon", {}).items():
            print(f"    epsilon={eps_str}: Avg R-CNN mAP={avg['avg_rcnn_mAP']:.4f}, "
                  f"Avg YOLOS mAP={avg['avg_yolos_mAP']:.4f}")

    # -- CARLA-Specific Ablations ----------------------------------------
    weather_summary = aggregate_weather_ablation(results_dir)
    if weather_summary:
        summary["weather_ablation"] = weather_summary
        print("\n  Weather Ablation Summary:")
        for weather, models in weather_summary.items():
            print(f"    {weather.capitalize()}:")
            for model_key, metrics in models.items():
                c_score = metrics.get('clean_score', metrics.get('clean_mAP', 0.0))
                p_score = metrics.get('patched_score', metrics.get('patched_mAP', 0.0))
                drop_score = metrics.get('score_drop', metrics.get('mAP_drop', 0.0))
                print(f"      {model_key.upper()}: Clean={c_score:.4f}, Patched={p_score:.4f}, Drop={drop_score:.4f}")

    dist_summary = aggregate_distance_ablation(results_dir)
    if dist_summary:
        summary["distance_ablation"] = dist_summary
        print("\n  Distance Ablation Summary:")
        for dist, models in sorted(dist_summary.items(), key=lambda x: int(x[0].replace('m', '')) if x[0].replace('m', '').isdigit() else 0):
            print(f"    {dist}:")
            for model_key, metrics in models.items():
                c_score = metrics.get('clean_score', metrics.get('clean_mAP', 0.0))
                p_score = metrics.get('patched_score', metrics.get('patched_mAP', 0.0))
                drop_score = metrics.get('score_drop', metrics.get('mAP_drop', 0.0))
                print(f"      {model_key.upper()}: Clean={c_score:.4f}, Patched={p_score:.4f}, Drop={drop_score:.4f}")

    cross_domain = aggregate_cross_domain(results_dir)
    if cross_domain:
        summary["cross_domain"] = cross_domain
        print("\n  Cross-Domain Transfer Summary (mAP Drop on Full-Band Patch):")
        for model in ["rcnn", "yolos"]:
            nuscenes_drop = cross_domain["nuscenes"].get(model, {}).get("full", {}).get("mAP_drop", 0.0)
            carla_drop = cross_domain["carla"].get(model, {}).get("full", {}).get("mAP_drop", 0.0)
            print(f"    {model.upper()}: nuScenes Drop={nuscenes_drop:.4f}, CARLA Drop={carla_drop:.4f}")

    # -- Save Summary ----------------------------------------------------
    summary_path = os.path.join(eval_dir, "summary_table.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\n  Saved: {summary_path}")

    # -- Generate LaTeX Tables -------------------------------------------
    latex_lines = []
    latex_lines.append("% ==========================================================")
    latex_lines.append("% Auto-generated LaTeX tables for ViT vs CNN Robustness Study")
    latex_lines.append("% ==========================================================")
    latex_lines.append("")

    if freq_summary:
        latex_lines.append("% --- Frequency Band Comparison ---")
        latex_lines.append(generate_latex_frequency_table(freq_summary))
        latex_lines.append("")

    if rsf_data:
        latex_lines.append("% --- RSF Curves ---")
        latex_lines.append(generate_latex_rsf_table(rsf_data))
        latex_lines.append("")

    if sweep_summary:
        latex_lines.append("% --- Epsilon Sweep ---")
        latex_lines.append(generate_latex_epsilon_table(sweep_summary))
        latex_lines.append("")

    if weather_summary:
        latex_lines.append("% --- Weather Ablation (CARLA) ---")
        latex_lines.append(generate_latex_weather_table(weather_summary))
        latex_lines.append("")

    if dist_summary:
        latex_lines.append("% --- Distance Ablation (CARLA) ---")
        latex_lines.append(generate_latex_distance_table(dist_summary))
        latex_lines.append("")

    latex_path = os.path.join(eval_dir, "latex_tables.txt")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))
    print(f"  Saved: {latex_path}")

    return summary


# =======================================================================
# CLI Entry Point
# =======================================================================
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Aggregate experiment results into summary tables and LaTeX output"
    )
    parser.add_argument(
        "--results_dir", type=str, default="results/carla",
        help="Path to results directory (default: results/carla)"
    )
    parser.add_argument(
        "--smoke_test", action="store_true",
        help="Run a smoke test with mock data instead of real aggregation"
    )
    args = parser.parse_args()

    if args.smoke_test:
        import tempfile

        print("=" * 72)
        print("  aggregate_results.py - Smoke Test (mock data)")
        print("=" * 72)

        # Create mock evaluation data
        with tempfile.TemporaryDirectory() as tmpdir:
            eval_dir = os.path.join(tmpdir, "evaluation")
            os.makedirs(eval_dir)

            # Mock frequency comparison
            freq = {
                "rcnn": {
                    "clean": {"mAP": 0.452},
                    "low": {"mAP": 0.410, "mAP_drop": 0.042, "avg_detections": 3.2, "avg_max_confidence": 0.78},
                    "high": {"mAP": 0.385, "mAP_drop": 0.067, "avg_detections": 2.8, "avg_max_confidence": 0.72},
                    "full": {"mAP": 0.340, "mAP_drop": 0.112, "avg_detections": 2.1, "avg_max_confidence": 0.65},
                },
                "yolos": {
                    "clean": {"mAP": 0.389},
                    "low": {"mAP": 0.320, "mAP_drop": 0.069, "avg_detections": 2.5, "avg_max_confidence": 0.70},
                    "high": {"mAP": 0.345, "mAP_drop": 0.044, "avg_detections": 2.9, "avg_max_confidence": 0.74},
                    "full": {"mAP": 0.280, "mAP_drop": 0.109, "avg_detections": 1.8, "avg_max_confidence": 0.60},
                },
            }
            with open(os.path.join(eval_dir, "frequency_comparison.json"), "w") as f:
                json.dump(freq, f)

            # Mock RSF curves
            rsf = {
                "rcnn": {"ratios": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], "mAPs": [0.452, 0.44, 0.41, 0.38, 0.34, 0.30], "rsf": 0.30},
                "yolos": {"ratios": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], "mAPs": [0.389, 0.37, 0.34, 0.30, 0.25, 0.20], "rsf": 0.38},
            }
            with open(os.path.join(eval_dir, "rsf_curves.json"), "w") as f:
                json.dump(rsf, f)

            # Mock epsilon sweep
            sweep = {
                "0.1": {"low": {"rcnn_mAP": 0.44, "yolos_mAP": 0.37, "adversarial_delta": -0.07}},
                "0.3": {"low": {"rcnn_mAP": 0.41, "yolos_mAP": 0.32, "adversarial_delta": -0.09}},
                "0.5": {"low": {"rcnn_mAP": 0.35, "yolos_mAP": 0.25, "adversarial_delta": -0.10}},
            }
            with open(os.path.join(eval_dir, "epsilon_sweep.json"), "w") as f:
                json.dump(sweep, f)

            # Run aggregation
            summary = aggregate_all(tmpdir)
            assert "frequency_comparison" in summary
            assert "rsf_curves" in summary
            assert "epsilon_sweep" in summary

            print("\n  ✅ Aggregation smoke test passed!")
    else:
        print("=" * 72)
        print(f"  aggregate_results.py - Aggregating results from: {args.results_dir}")
        print("=" * 72)
        aggregate_all(args.results_dir)

