"""Run all agent architectures on the same question set and compare results."""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.agents.base import AnalysisResult, BaseAgent
from src.agents.single_agent import SingleAgent
from src.agents.multi_agent import MultiAgent
from src.agents.rag_agent import RAGAgent
from src.data.registry import get_dataset_info
from src.evaluation.ground_truth import compute_ground_truth
from src.evaluation.metrics import (
    EvalScore,
    numeric_accuracy,
    categorical_accuracy,
    directional_accuracy,
    check_weight_usage,
    llm_judge,
)

# Delay between questions to avoid rate limits (seconds)
# gpt-4o-mini: 200K TPM, gpt-4o free-tier: 30K TPM.
# 5s is safe for gpt-4o-mini; increase to 30+ for gpt-4o free-tier.
INTER_QUESTION_DELAY = 5
# Max retries on rate-limit (429) errors
RATE_LIMIT_MAX_RETRIES = 3


AGENT_CLASSES: dict[str, type[BaseAgent]] = {
    "single": SingleAgent,
    "multi": MultiAgent,
    "rag": RAGAgent,
}


def load_questions(path: str | Path = "experiments/questions.yaml") -> list[dict]:
    """Load evaluation questions from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("questions", [])


def evaluate_answer(
    result: AnalysisResult,
    question_def: dict,
    ground_truth: dict,
    agent_name: str,
) -> EvalScore:
    """Score a single agent answer against ground truth."""
    info = get_dataset_info(question_def["dataset"])
    q_type = question_def["type"]

    score = EvalScore(
        question_id=question_def["id"],
        agent=agent_name,
        dataset=question_def["dataset"],
        execution_time=result.execution_time_seconds,
        had_error=not result.success,
        retries=result.retries,
        weight_used=check_weight_usage(result.code_executed, info.weight_column),
    )

    if not result.success or not result.answer:
        score.details = {
            "ground_truth": ground_truth,
            "errors": result.errors if result.errors else ["No answer produced"],
        }
        return score

    gt_value = ground_truth.get("value")

    if q_type == "numeric" and gt_value is not None:
        score.accuracy = numeric_accuracy(result.answer, float(gt_value))
        score.completeness = 1.0 if score.accuracy > 0 else 0.0

    elif q_type == "categorical" and gt_value is not None:
        aliases = ground_truth.get("aliases")
        score.accuracy = categorical_accuracy(result.answer, str(gt_value), aliases=aliases)
        score.completeness = 1.0 if score.accuracy > 0 else 0.0

    elif q_type == "directional" and gt_value is not None:
        correlation = ground_truth.get("correlation")
        score.accuracy = directional_accuracy(result.answer, str(gt_value), correlation=correlation)
        score.completeness = 1.0 if score.accuracy > 0 else 0.0

    elif q_type == "descriptive":
        judge_scores = llm_judge(
            question=question_def["question"],
            answer=result.answer,
            expected_description=str(gt_value) if gt_value else question_def["question"],
        )
        score.accuracy = judge_scores["accuracy"]
        score.completeness = judge_scores["completeness"]

    score.details = {
        "ground_truth": ground_truth,
        "answer_preview": result.answer[:300],
    }
    return score


def _agent_cache_dir() -> Path:
    """Return the agent results cache directory, creating it if needed."""
    d = Path("experiments/.cache/agent_results")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key_for(agent_name: str, model: str, question_id: str) -> str:
    """Build a safe filename key for the agent result cache."""
    safe_model = model.replace("/", "_").replace(":", "_")
    return f"{agent_name}_{safe_model}_{question_id}"


def _save_result_cache(agent_name: str, model: str, question_id: str, result: AnalysisResult) -> None:
    """Persist an AnalysisResult to disk as JSON."""
    key = _cache_key_for(agent_name, model, question_id)
    path = _agent_cache_dir() / f"{key}.json"
    with open(path, "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)


def _load_result_cache(agent_name: str, model: str, question_id: str) -> AnalysisResult | None:
    """Load a cached AnalysisResult from disk, or return None."""
    key = _cache_key_for(agent_name, model, question_id)
    path = _agent_cache_dir() / f"{key}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        # Backwards compat: fill in any missing fields with defaults
        defaults = {
            "question": "",
            "dataset": "",
            "answer": "",
            "code_executed": "",
            "raw_statistics": {},
            "execution_time_seconds": 0.0,
            "errors": [],
            "retries": 0,
        }
        for k, v in defaults.items():
            data.setdefault(k, v)
        return AnalysisResult(
            question=data["question"],
            dataset=data["dataset"],
            answer=data["answer"],
            code_executed=data["code_executed"],
            raw_statistics=data["raw_statistics"],
            execution_time_seconds=data["execution_time_seconds"],
            errors=data["errors"],
            retries=data["retries"],
        )
    except Exception:
        return None


def run_comparison(
    agent_names: list[str] | None = None,
    question_ids: list[str] | None = None,
    datasets: list[str] | None = None,
    model: str = "gpt-4o",
    provider: str | None = None,
    questions_path: str = "experiments/questions.yaml",
    output_dir: str = "experiments/results",
    no_cache: bool = False,
) -> list[EvalScore]:
    """Run specified agents on specified questions and return scored results.

    Args:
        agent_names: Agent architectures to test. Defaults to all.
        question_ids: Specific question IDs. Defaults to all.
        datasets: Filter to specific datasets. Defaults to all.
        model: Model name (e.g. "gpt-4o", "claude-sonnet-4-20250514").
        provider: LLM provider ("openai" or "anthropic"). Auto-detected if omitted.
        questions_path: Path to the questions YAML.
        output_dir: Directory to save results.
        no_cache: If True, bypass the disk cache and re-run all agents.

    Returns:
        List of EvalScore objects.
    """
    agents_to_run = agent_names or list(AGENT_CLASSES.keys())
    questions = load_questions(questions_path)

    # Filter questions
    if question_ids:
        questions = [q for q in questions if q["id"] in question_ids]
    if datasets:
        questions = [q for q in questions if q["dataset"] in datasets]

    if not questions:
        print("No matching questions found.")
        return []

    all_scores: list[EvalScore] = []

    for agent_name in agents_to_run:
        print(f"\n{'='*60}")
        print(f"Agent: {agent_name}")
        print(f"{'='*60}")

        agent = None  # lazy init only if needed

        for qi, q in enumerate(questions):
            print(f"\n  [{q['id']}] {q['question'][:80]}...")

            # Compute ground truth
            gt = compute_ground_truth(q["ground_truth_key"])

            # Check disk cache
            result = None
            if not no_cache:
                result = _load_result_cache(agent_name, model, q["id"])
                if result is not None:
                    print(f"  [cached] loaded from disk cache")

            # Run agent if no cached result
            if result is None:
                if agent is None:
                    agent = AGENT_CLASSES[agent_name](model=model, provider=provider)
                for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
                    try:
                        result = agent.analyze(q["question"], q["dataset"])
                        break
                    except Exception as e:
                        err_str = str(e)
                        if ("429" in err_str or "RateLimit" in err_str or "rate_limit" in err_str) and attempt < RATE_LIMIT_MAX_RETRIES:
                            wait = 30 * (2 ** attempt)  # 30s, 60s, 120s
                            print(f"    ⏳ Rate limited (attempt {attempt + 1}/{RATE_LIMIT_MAX_RETRIES + 1}), waiting {wait}s...")
                            time.sleep(wait)
                            continue
                        result = AnalysisResult(
                            question=q["question"],
                            dataset=q["dataset"],
                            answer="",
                            errors=[f"{type(e).__name__}: {e}\n{traceback.format_exc()}"],
                        )
                        break
                # Save to cache
                _save_result_cache(agent_name, model, q["id"], result)

            # Evaluate
            score = evaluate_answer(result, q, gt, agent_name)
            all_scores.append(score)

            status = "✓" if score.accuracy > 0.5 else "✗"
            print(f"  {status} accuracy={score.accuracy:.2f}  "
                  f"weights={'✓' if score.weight_used else '✗'}  "
                  f"time={score.execution_time:.1f}s")

            # Delay between questions to stay under TPM limits
            if qi < len(questions) - 1:
                time.sleep(INTER_QUESTION_DELAY)

    # Save raw results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir) / f"comparison_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump([asdict(s) for s in all_scores], f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return all_scores


def summarize_scores(scores: list[EvalScore]) -> dict[str, dict[str, float]]:
    """Produce per-agent summary statistics from a list of EvalScores."""
    from collections import defaultdict

    by_agent: dict[str, list[EvalScore]] = defaultdict(list)
    for s in scores:
        by_agent[s.agent].append(s)

    summary = {}
    for agent, agent_scores in by_agent.items():
        n = len(agent_scores)
        summary[agent] = {
            "n_questions": n,
            "avg_accuracy": round(sum(s.accuracy for s in agent_scores) / n, 3),
            "avg_completeness": round(sum(s.completeness for s in agent_scores) / n, 3),
            "weight_usage_pct": round(sum(1 for s in agent_scores if s.weight_used) / n * 100, 1),
            "error_rate_pct": round(sum(1 for s in agent_scores if s.had_error) / n * 100, 1),
            "avg_time_seconds": round(sum(s.execution_time for s in agent_scores) / n, 2),
            "avg_retries": round(sum(s.retries for s in agent_scores) / n, 2),
        }
    return summary
