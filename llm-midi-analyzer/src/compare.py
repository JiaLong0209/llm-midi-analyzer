"""
compare.py — Run both VQ-VAE variants sequentially and print a final report.

Usage:
  python3 src/compare.py --mode smoke   # quick comparison (default)
  python3 src/compare.py --mode full    # full LMD comparison
  python3 src/compare.py --save         # also dump configs to disk
"""
import sys
import os
import argparse
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from config import ExperimentConfig
from trainer.vqvae_trainer import train


def run_experiment(experiment: ExperimentConfig, save_configs: bool = False):
    results = []

    print(f"\n{'='*64}")
    print(f"  🧪 Experiment: {experiment.name}")
    print(f"  Runs: {len(experiment.runs)} variants")
    print(f"{'='*64}\n")

    if save_configs:
        experiment.save(f"{experiment.name}_config.json")

    for cfg in experiment.runs:
        variant = cfg.model.variant
        # Separate checkpoint dirs so runs don't overwrite each other
        cfg.training.checkpoint_dir = f"checkpoints/{experiment.name}/{variant}"
        os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)

        print(f"\n{'─'*64}")
        print(f"  ▶ Starting variant: [{variant.upper()}]")
        print(f"{'─'*64}")

        t0 = time.perf_counter()
        train(cfg)
        elapsed = time.perf_counter() - t0

        results.append({"variant": variant, "elapsed_s": elapsed})

    # ── Final Report ──────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  📊 Experiment Results: {experiment.name}")
    print(f"{'='*64}")
    print(f"  {'Variant':<20} {'Time (s)':>10}")
    print(f"  {'─'*30}")
    for r in results:
        print(f"  {r['variant']:<20} {r['elapsed_s']:>10.1f}s")
    print(f"{'='*64}")
    print("  ✅ See checkpoints/<experiment>/<variant>/ for saved models.")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniLLM-Muse: Compare VQ-VAE Variants")
    parser.add_argument("--mode", choices=["smoke", "full", "tokens"], default="smoke",
                        help="'smoke'=model variant compare | 'tokens'=5D vs 8D compare | 'full'=full run")
    parser.add_argument("--save", action="store_true", help="Dump experiment config to JSON")
    args = parser.parse_args()

    if args.mode == "full":
        experiment = ExperimentConfig.compare_full()
    elif args.mode == "tokens":
        experiment = ExperimentConfig.compare_tokens()
    else:
        experiment = ExperimentConfig.compare_smoke()

    run_experiment(experiment, save_configs=args.save)
