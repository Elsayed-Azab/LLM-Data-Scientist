"""Generate markdown evaluation reports from comparison results."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.evaluation.metrics import EvalScore
from src.evaluation.comparator import summarize_scores


def generate_report(
    scores: list[EvalScore],
    output_path: str | Path = "experiments/results/report.md",
) -> str:
    """Generate a markdown report comparing agent architectures.

    Args:
        scores: List of EvalScore objects from a comparison run.
        output_path: Where to write the report.

    Returns:
        The markdown string.
    """
    summary = summarize_scores(scores)
    agents = sorted(summary.keys())

    lines = [
        "# LLM Data Scientist — Evaluation Report",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## Summary",
        "",
        _summary_table(summary, agents),
        "",
        "## Per-Question Results",
        "",
    ]

    # Group scores by question
    from collections import defaultdict
    by_question: dict[str, dict[str, EvalScore]] = defaultdict(dict)
    for s in scores:
        by_question[s.question_id][s.agent] = s

    for q_id in sorted(by_question.keys()):
        agent_scores = by_question[q_id]
        first = next(iter(agent_scores.values()))
        lines.append(f"### {q_id} ({first.dataset})")
        lines.append("")

        # Table header
        lines.append("| Agent | Accuracy | Completeness | Weights | Time (s) | Errors |")
        lines.append("|-------|----------|--------------|---------|----------|--------|")

        for agent in agents:
            s = agent_scores.get(agent)
            if s:
                w = "Yes" if s.weight_used else "No"
                err = "Yes" if s.had_error else "No"
                lines.append(
                    f"| {agent} | {s.accuracy:.2f} | {s.completeness:.2f} | "
                    f"{w} | {s.execution_time:.1f} | {err} |"
                )

        lines.append("")

    # Key findings
    lines.extend([
        "## Key Findings",
        "",
    ])

    if len(agents) >= 2:
        best_agent = max(agents, key=lambda a: summary[a]["avg_accuracy"])
        fastest_agent = min(agents, key=lambda a: summary[a]["avg_time_seconds"])
        best_weights = max(agents, key=lambda a: summary[a]["weight_usage_pct"])

        lines.append(f"- **Most accurate**: {best_agent} (avg accuracy: {summary[best_agent]['avg_accuracy']:.3f})")
        lines.append(f"- **Fastest**: {fastest_agent} (avg time: {summary[fastest_agent]['avg_time_seconds']:.1f}s)")
        lines.append(f"- **Best weight usage**: {best_weights} ({summary[best_weights]['weight_usage_pct']:.0f}%)")
        lines.append("")

    report = "\n".join(lines)

    # Write to file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    return report


def _summary_table(summary: dict, agents: list[str]) -> str:
    """Build a markdown summary table."""
    lines = [
        "| Metric | " + " | ".join(agents) + " |",
        "|--------| " + " | ".join("---" for _ in agents) + " |",
    ]

    metrics = [
        ("Avg Accuracy", "avg_accuracy"),
        ("Avg Completeness", "avg_completeness"),
        ("Weight Usage %", "weight_usage_pct"),
        ("Error Rate %", "error_rate_pct"),
        ("Avg Time (s)", "avg_time_seconds"),
        ("Avg Retries", "avg_retries"),
    ]

    for label, key in metrics:
        vals = " | ".join(str(summary[a][key]) for a in agents)
        lines.append(f"| {label} | {vals} |")

    return "\n".join(lines)
