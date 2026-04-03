"""Evaluation metrics for comparing agent outputs against ground truth."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


@dataclass
class EvalScore:
    """Score for a single question evaluation."""
    question_id: str
    agent: str
    dataset: str
    accuracy: float = 0.0       # 0-1 for numeric/categorical, LLM-judge for descriptive
    completeness: float = 0.0   # 0-1, did the answer address the full question?
    weight_used: bool = False   # did the code apply survey weights?
    execution_time: float = 0.0
    had_error: bool = False
    retries: int = 0
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Numeric accuracy
# ---------------------------------------------------------------------------

def numeric_accuracy(
    answer_text: str,
    expected: float,
    tolerance_pct: float = 5.0,
) -> float:
    """Extract a number from the answer and compare to expected.

    Returns 1.0 if within tolerance, partial credit scaled linearly,
    0.0 if no number found or way off.
    """
    # Strip commas from numbers like "15,627" before extracting
    cleaned = re.sub(r"(\d),(\d)", r"\1\2", answer_text)
    numbers = re.findall(r"[-+]?\d*\.?\d+", cleaned)
    if not numbers:
        return 0.0

    # Try each extracted number, take the best match
    best_score = 0.0
    for num_str in numbers:
        try:
            val = float(num_str)
        except ValueError:
            continue
        if expected == 0:
            score = 1.0 if abs(val) < 0.01 else 0.0
        else:
            pct_error = abs(val - expected) / abs(expected) * 100
            if pct_error <= tolerance_pct:
                score = 1.0
            elif pct_error <= tolerance_pct * 3:
                # Linear partial credit
                score = max(0.0, 1.0 - (pct_error - tolerance_pct) / (tolerance_pct * 2))
            else:
                score = 0.0
        best_score = max(best_score, score)

    return round(best_score, 3)


# ---------------------------------------------------------------------------
# Categorical accuracy
# ---------------------------------------------------------------------------

def categorical_accuracy(answer_text: str, expected: str, aliases: list[str] | None = None) -> float:
    """Check if the expected category appears in the answer (case-insensitive).

    Args:
        answer_text: The agent's answer text.
        expected: The expected category value (code or label).
        aliases: Optional list of alternative acceptable values (e.g. label
                 alternatives for a numeric code).

    Returns 1.0 if found, 0.0 otherwise.
    """
    text = answer_text.lower()
    candidates = [expected]
    if aliases:
        candidates.extend(aliases)
    return 1.0 if any(c.lower() in text for c in candidates) else 0.0


# ---------------------------------------------------------------------------
# Directional accuracy
# ---------------------------------------------------------------------------

def directional_accuracy(
    answer_text: str,
    expected_direction: str,
    correlation: float | None = None,
) -> float:
    """Check if the answer identifies the correct relationship direction.

    Args:
        expected_direction: One of 'positive', 'negative', 'none'.
        correlation: Optional actual correlation value. When the true
                     correlation is near zero (|r| < 0.1), reporting
                     either "weak positive/negative" or "none" is
                     considered acceptable (partial credit 0.5).

    Returns 1.0 if direction matches, 0.5 for borderline cases, 0.0 otherwise.
    """
    text = answer_text.lower()

    positive_signals = ["positive", "increases", "higher", "more likely", "associated with higher",
                        "positively correlated", "positive correlation", "positive relationship"]
    negative_signals = ["negative", "decreases", "lower", "less likely", "associated with lower",
                        "negatively correlated", "negative correlation", "inverse"]
    none_signals = ["no significant", "no relationship", "no association", "not significant",
                    "no correlation", "no clear", "negligible", "very weak", "weak"]

    detected_positive = any(s in text for s in positive_signals)
    detected_negative = any(s in text for s in negative_signals)
    detected_none = any(s in text for s in none_signals)

    # Borderline: if |correlation| < 0.1, both the expected direction and
    # a "weak/none" answer are defensible.
    is_borderline = correlation is not None and abs(correlation) < 0.1

    if expected_direction == "positive":
        if detected_positive:
            return 1.0
        if is_borderline and (detected_none or detected_negative):
            return 0.5
        return 0.0
    elif expected_direction == "negative":
        if detected_negative:
            return 1.0
        if is_borderline and (detected_none or detected_positive):
            return 0.5
        return 0.0
    elif expected_direction == "none":
        if detected_none:
            return 1.0
        if is_borderline and (detected_positive or detected_negative):
            return 0.5
        return 0.0
    return 0.0


# ---------------------------------------------------------------------------
# Weight usage check
# ---------------------------------------------------------------------------

def check_weight_usage(code_executed: str, weight_column: str) -> bool:
    """Check whether the executed code references the weight column."""
    if not code_executed:
        return False
    return weight_column.lower() in code_executed.lower()


# ---------------------------------------------------------------------------
# LLM-as-judge (for descriptive / complex answers)
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are an expert evaluator of statistical analyses on social science survey data.

Rate the following answer on two dimensions, each from 0 to 10:

1. **Accuracy**: Is the answer factually correct and statistically sound?
2. **Completeness**: Does the answer fully address the question?

Question: {question}
Expected answer characteristics: {expected}
Agent's answer: {answer}

Respond in EXACTLY this format (two lines, nothing else):
accuracy: <0-10>
completeness: <0-10>
"""


def llm_judge(
    question: str,
    answer: str,
    expected_description: str,
    model: str = "gpt-4o",
) -> dict[str, float]:
    """Use an LLM to judge accuracy and completeness of a descriptive answer.

    Returns dict with 'accuracy' and 'completeness' scores normalised to 0-1.
    """
    try:
        llm = ChatOpenAI(model=model, temperature=0.0)
        response = llm.invoke([
            SystemMessage(content="You are an expert evaluation judge."),
            HumanMessage(content=JUDGE_PROMPT.format(
                question=question,
                expected=expected_description,
                answer=answer,
            )),
        ])

        text = response.content.strip()
        scores = {}
        for line in text.split("\n"):
            line = line.strip().lower()
            if line.startswith("accuracy:"):
                scores["accuracy"] = min(float(re.findall(r"[\d.]+", line)[0]) / 10, 1.0)
            elif line.startswith("completeness:"):
                scores["completeness"] = min(float(re.findall(r"[\d.]+", line)[0]) / 10, 1.0)

        return {
            "accuracy": scores.get("accuracy", 0.0),
            "completeness": scores.get("completeness", 0.0),
        }
    except Exception:
        return {"accuracy": 0.0, "completeness": 0.0}
