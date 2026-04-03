"""CLI script to run the full evaluation pipeline.

Usage:
    # Run all agents on all questions
    python experiments/run_evaluation.py

    # Run specific agents
    python experiments/run_evaluation.py --agents single multi

    # Run on a specific dataset
    python experiments/run_evaluation.py --datasets arab_barometer

    # Run specific questions
    python experiments/run_evaluation.py --questions ab_01 ab_02 wvs_01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.evaluation.comparator import run_comparison, summarize_scores
from src.reporting.report import generate_report


def main():
    parser = argparse.ArgumentParser(description="Run full evaluation of agent architectures")
    parser.add_argument("--agents", nargs="+", help="Agent names (single, multi, rag)")
    parser.add_argument("--model", default="gpt-4o", help="Model name (e.g. gpt-4o, claude-sonnet-4-20250514)")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default=None,
                        help="LLM provider (auto-detected from model name if omitted)")
    parser.add_argument("--datasets", nargs="+", help="Filter to datasets (gss, arab_barometer, wvs)")
    parser.add_argument("--questions", nargs="+", help="Specific question IDs")
    parser.add_argument("--questions-file", default="experiments/questions.yaml")
    parser.add_argument("--output-dir", default="experiments/results")
    parser.add_argument("--report", default="experiments/results/report.md", help="Report output path")
    parser.add_argument("--no-cache", action="store_true", help="Bypass disk cache and re-run all agents")
    args = parser.parse_args()

    print(f"Starting evaluation with model={args.model}...\n")

    scores = run_comparison(
        agent_names=args.agents,
        question_ids=args.questions,
        datasets=args.datasets,
        model=args.model,
        provider=args.provider,
        questions_path=args.questions_file,
        output_dir=args.output_dir,
        no_cache=args.no_cache,
    )

    if not scores:
        print("No scores produced.")
        return

    # Print summary
    summary = summarize_scores(scores)
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for agent, stats in summary.items():
        print(f"\n  {agent}:")
        for k, v in stats.items():
            print(f"    {k}: {v}")

    # Generate report
    report = generate_report(scores, output_path=args.report)
    print(f"\nReport saved to {args.report}")


if __name__ == "__main__":
    main()
