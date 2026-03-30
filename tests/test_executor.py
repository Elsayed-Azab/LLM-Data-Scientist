"""Tests for the sandboxed code executor."""

import pandas as pd
import pytest

from src.analysis.executor import CodeExecutor


@pytest.fixture
def executor():
    return CodeExecutor(timeout_seconds=5)


@pytest.fixture
def sample_df():
    return pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})


def test_simple_execution(executor):
    result = executor.execute("result = 1 + 1")
    assert result.success
    assert result.return_value == 2


def test_print_capture(executor):
    result = executor.execute("print('hello')")
    assert result.success
    assert "hello" in result.output


def test_pandas_available(executor, sample_df):
    result = executor.execute("result = df['x'].sum()", df=sample_df)
    assert result.success
    assert result.return_value == 6


def test_numpy_available(executor):
    result = executor.execute("result = np.mean([1, 2, 3])")
    assert result.success
    assert result.return_value == pytest.approx(2.0)


def test_syntax_error(executor):
    result = executor.execute("def bad(")
    assert not result.success
    assert "SyntaxError" in result.error


def test_runtime_error(executor):
    result = executor.execute("x = 1 / 0")
    assert not result.success
    assert "ZeroDivisionError" in result.error


def test_timeout(executor):
    executor.timeout = 1
    # Use a busy loop since import is blocked by safe builtins
    result = executor.execute("i = 0\nwhile i < 10**9: i += 1")
    assert not result.success
    assert "exceeded" in result.error.lower() or "timeout" in result.error.lower()


def test_restricted_builtins(executor):
    result = executor.execute("open('/etc/passwd')")
    assert not result.success


def test_extra_globals(executor):
    result = executor.execute("result = weight_column", extra_globals={"weight_column": "WT"})
    assert result.success
    assert result.return_value == "WT"


def test_variables_captured(executor):
    result = executor.execute("a = 10\nb = 20")
    assert result.success
    assert result.variables["a"] == 10
    assert result.variables["b"] == 20
