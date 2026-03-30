"""Base agent interface that all architectures implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnalysisResult:
    """Standard output returned by every agent."""

    question: str
    dataset: str
    answer: str
    code_executed: str = ""
    raw_statistics: dict[str, Any] = field(default_factory=dict)
    execution_time_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    retries: int = 0

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


class BaseAgent(ABC):
    """Abstract base class for all agent architectures.

    Subclasses must implement ``analyze``, which takes a natural-language
    question and a dataset name, and returns an ``AnalysisResult``.
    """

    @abstractmethod
    def analyze(self, question: str, dataset_name: str) -> AnalysisResult:
        """Run an analysis and return structured results."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
