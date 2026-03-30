"""Sandboxed execution of LLM-generated Python code."""

from __future__ import annotations

import io
import threading
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.stats


@dataclass
class ExecutionResult:
    """Result of executing LLM-generated code."""

    success: bool
    output: str = ""
    error: str = ""
    return_value: object = None
    variables: dict = field(default_factory=dict)


class CodeExecutor:
    """Execute Python code in a restricted namespace.

    The namespace is pre-populated with pandas, numpy, scipy.stats,
    and an optional DataFrame named ``df``.
    """

    def __init__(self, timeout_seconds: int = 10):
        self.timeout = timeout_seconds

    def execute(
        self,
        code: str,
        df: pd.DataFrame | None = None,
        extra_globals: dict | None = None,
    ) -> ExecutionResult:
        """Run *code* and capture stdout, stderr, and namespace changes.

        Args:
            code: Python source to execute.
            df: Optional DataFrame injected as ``df`` in the namespace.
            extra_globals: Additional objects to inject.
        """
        namespace: dict = {
            "pd": pd,
            "np": np,
            "scipy": scipy.stats,
            "__builtins__": _safe_builtins(),
        }
        if df is not None:
            namespace["df"] = df
        if extra_globals:
            namespace.update(extra_globals)

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        result_holder: list[ExecutionResult] = []

        def _run():
            try:
                with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                    exec(code, namespace)  # noqa: S102

                user_vars = {
                    k: v
                    for k, v in namespace.items()
                    if not k.startswith("_") and k not in ("pd", "np", "scipy", "df")
                }

                result_holder.append(ExecutionResult(
                    success=True,
                    output=stdout_buf.getvalue(),
                    variables=user_vars,
                    return_value=namespace.get("result"),
                ))
            except Exception:
                result_holder.append(ExecutionResult(
                    success=False,
                    output=stdout_buf.getvalue(),
                    error=traceback.format_exc(),
                ))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout)

        if thread.is_alive():
            return ExecutionResult(
                success=False,
                error=f"Code execution exceeded {self.timeout}s limit",
            )

        if result_holder:
            return result_holder[0]

        return ExecutionResult(success=False, error="No result produced")


def _safe_builtins() -> dict:
    """Return a restricted set of builtins (no file/network access)."""
    import builtins

    allowed = [
        "abs", "all", "any", "bool", "dict", "enumerate", "filter",
        "float", "format", "frozenset", "getattr", "hasattr", "hash",
        "int", "isinstance", "issubclass", "iter", "len", "list",
        "map", "max", "min", "next", "print", "range", "repr",
        "reversed", "round", "set", "slice", "sorted", "str", "sum",
        "tuple", "type", "zip",
    ]
    return {name: getattr(builtins, name) for name in allowed}
