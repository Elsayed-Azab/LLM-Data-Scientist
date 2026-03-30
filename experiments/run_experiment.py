"""CLI script to run analysis experiments.

Usage:
    python experiments/run_experiment.py --agent single --dataset wvs --question "What is the average age?"
    python experiments/run_experiment.py --agent single --dataset arab_barometer --questions experiments/questions.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.agents.single_agent import SingleAgent
from src.agents.multi_agent import MultiAgent
from src.agents.rag_agent import RAGAgent
from src.agents.base import AnalysisResult


AGENT_CLASSES = {
    "single": SingleAgent,
    "multi": MultiAgent,
    "rag": RAGAgent,
}


def run_single_question(agent_name: str, dataset: str, question: str,
                        model: str = "gpt-4o", provider: str | None = None) -> AnalysisResult:
    agent_cls = AGENT_CLASSES[agent_name]
    agent = agent_cls(model=model, provider=provider)
    return agent.analyze(question, dataset)


def main():
    parser = argparse.ArgumentParser(description="Run LLM data analysis experiments")
    parser.add_argument("--agent", choices=list(AGENT_CLASSES.keys()), default="single")
    parser.add_argument("--model", default="gpt-4o", help="Model name (e.g. gpt-4o, claude-sonnet-4-20250514)")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default=None,
                        help="LLM provider (auto-detected from model name if omitted)")
    parser.add_argument("--dataset", required=True, help="Dataset name (gss, arab_barometer, wvs)")
    parser.add_argument("--question", help="Single question to ask")
    parser.add_argument("--questions", help="Path to YAML file with multiple questions")
    parser.add_argument("--output", help="Path to save results JSON")
    args = parser.parse_args()

    questions: list[str] = []
    if args.question:
        questions.append(args.question)
    elif args.questions:
        import yaml
        with open(args.questions) as f:
            data = yaml.safe_load(f)
        questions = [q["question"] for q in data.get("questions", [])]
    else:
        parser.error("Provide --question or --questions")

    results = []
    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {q}")
        result = run_single_question(args.agent, args.dataset, q, model=args.model, provider=args.provider)
        print(f"  ✓ {result.execution_time_seconds}s")
        print(f"  Answer: {result.answer[:200]}...")
        results.append({
            "question": result.question,
            "dataset": result.dataset,
            "answer": result.answer,
            "code_executed": result.code_executed,
            "execution_time_seconds": result.execution_time_seconds,
            "errors": result.errors,
            "success": result.success,
        })

    # Save results
    out_path = args.output or f"experiments/results/{args.agent}_{args.dataset}_{datetime.now():%Y%m%d_%H%M%S}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
