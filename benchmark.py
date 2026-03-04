#!/usr/bin/env python3
"""GHC Benchmark - Genuine Human Cognition Score

A benchmarking pipeline that measures how closely AI Chain of Thought 
matches real human thinking patterns.

Usage:
    python benchmark.py run --stage all          # Run full pipeline
    python benchmark.py run --stage generate      # Stage 1: Generate LLM samples
    python benchmark.py run --stage evaluate      # Stage 2: Run benchmarks
    python benchmark.py run --stage report        # Stage 3: Generate reports
"""

import os
import sys
import argparse

from src import config
from src.stage1_generate import run_generate
from src.stage2_evaluate import run_evaluate
from src.stage3_report import run_report


def main():
    parser = argparse.ArgumentParser(
        description="GHC Benchmark - Genuine Human Cognition Score",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py run --stage all              Run full pipeline
  python benchmark.py run --stage all --reset    Full reset and run
  python benchmark.py run --stage generate       Generate LLM samples only
  python benchmark.py run --stage evaluate       Run benchmarks only
  python benchmark.py run --stage report         Generate reports only
        """
    )
    
    parser.add_argument("command", nargs="?", default="run", help="Command to run (run)")
    parser.add_argument("--stage", choices=["generate", "evaluate", "report", "all"], 
                        default="all", help="Pipeline stage to run")
    parser.add_argument("--reset", action="store_true", 
                        help="Reset results and run from scratch")
    parser.add_argument("--models", nargs="+", 
                        help="Specific models to run (default: all)")
    
    args = parser.parse_args()
    
    if args.command != "run":
        print(f"Unknown command: {args.command}")
        print("Use 'run' as the command")
        sys.exit(1)
    
    os.makedirs("results", exist_ok=True)
    os.makedirs(config.CHARTS_DIR, exist_ok=True)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("GHC BENCHMARK - Genuine Human Cognition Score")
    print("=" * 60)
    
    if args.stage == "generate":
        print("\n>>> STAGE 1: Generating LLM samples...")
        run_generate(models=args.models, reset=args.reset)
    
    elif args.stage == "evaluate":
        print("\n>>> STAGE 2: Running evaluation benchmarks...")
        run_evaluate(models=args.models, reset=args.reset)
    
    elif args.stage == "report":
        print("\n>>> STAGE 3: Generating reports...")
        run_report()
    
    elif args.stage == "all":
        print("\n>>> STAGE 1: Generating LLM samples...")
        run_generate(models=args.models, reset=args.reset)
        
        print("\n>>> STAGE 2: Running evaluation benchmarks...")
        run_evaluate(models=args.models, reset=False)
        
        print("\n>>> STAGE 3: Generating reports...")
        run_report()
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"Charts: {config.CHARTS_DIR}/")
        print(f"Report: {config.REPORTS_DIR}/report.md")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
