"""LangChain-compatible tools shared across all agent architectures."""

from __future__ import annotations

from typing import Any

import yaml
from langchain_core.tools import tool

from src.analysis.executor import CodeExecutor
from src.data.loader import DatasetLoader
from src.data.preprocessor import Preprocessor


def _load_config_timeout() -> int:
    """Read execution timeout from config/default.yaml."""
    try:
        with open("config/default.yaml") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("execution", {}).get("timeout_seconds", 120)
    except Exception:
        return 120


# Module-level singletons — initialised once per process.
_loader = DatasetLoader()
_executor = CodeExecutor(timeout_seconds=_load_config_timeout())
_preprocessor = Preprocessor()

# Cache loaded DataFrames to avoid re-reading large files.
_df_cache: dict[str, Any] = {}


@tool
def load_dataset(dataset_name: str, columns: list[str] | None = None) -> str:
    """Load a survey dataset and return its schema (columns, dtypes, sample rows).

    Args:
        dataset_name: One of 'gss', 'arab_barometer', 'wvs'.
        columns: Optional list of column names to load (recommended for large datasets).
    """
    cache_key = f"{dataset_name}:{','.join(columns) if columns else 'all'}"
    if cache_key not in _df_cache:
        df, meta = _loader.load(dataset_name, columns=columns)
        _df_cache[cache_key] = (df, meta)

    df, meta = _df_cache[cache_key]
    # Limit sample columns to avoid token bloat on wide datasets
    sample_df = df.head(3)
    if len(sample_df.columns) > 20:
        sample_df = sample_df.iloc[:, :20]
    sample = sample_df.to_string()
    dtype_items = list(df.dtypes.items())
    if len(dtype_items) > 100:
        dtypes = "\n".join(f"  {col}: {dtype}" for col, dtype in dtype_items[:100])
        dtypes += f"\n  ... and {len(dtype_items) - 100} more columns (use get_variable_info to inspect specific ones)"
    else:
        dtypes = "\n".join(f"  {col}: {dtype}" for col, dtype in dtype_items)
    return (
        f"Dataset: {meta.description}\n"
        f"Rows: {meta.num_rows}, Columns: {meta.num_columns}\n"
        f"Weight column: {meta.weight_column}\n\n"
        f"Dtypes:\n{dtypes}\n\n"
        f"Sample:\n{sample}"
    )


@tool
def get_dataset_schema(dataset_name: str) -> str:
    """Get lightweight schema info (columns, dtypes, sample) without loading the full dataset.

    Use this before load_dataset to inspect what columns are available.

    Args:
        dataset_name: One of 'gss', 'arab_barometer', 'wvs'.
    """
    schema = _loader.get_schema(dataset_name, sample_rows=3)
    all_cols = schema["columns"]
    if len(all_cols) > 100:
        col_lines = "\n".join(f"  {c}: {schema['dtypes'][c]}" for c in all_cols[:100])
        col_lines += f"\n  ... and {len(all_cols) - 100} more columns (use get_variable_info to inspect specific ones)"
    else:
        col_lines = "\n".join(f"  {c}: {schema['dtypes'][c]}" for c in all_cols)
    return f"Dataset: {dataset_name}\nColumns ({len(all_cols)}):\n{col_lines}"


@tool
def run_analysis_code(dataset_name: str, code: str) -> str:
    """Execute Python/pandas analysis code against a loaded dataset.

    The code runs in a sandboxed environment with ``df`` (the DataFrame),
    ``pd`` (pandas), ``np`` (numpy), and ``scipy`` (scipy.stats) available.

    Store final results in a variable called ``result`` so they are returned.
    Use ``print()`` for intermediate output.

    IMPORTANT: Always apply survey weights using the weight column.

    Args:
        dataset_name: The dataset to analyse.
        code: Python code to execute.
    """
    # Reuse any already-cached version of this dataset (prefer column subsets
    # that are already loaded over triggering a full load of a 567MB file).
    df, meta = None, None
    full_key = f"{dataset_name}:all"
    if full_key in _df_cache:
        df, meta = _df_cache[full_key]
    else:
        # Look for any cached subset for this dataset
        for key, val in _df_cache.items():
            if key.startswith(f"{dataset_name}:"):
                df, meta = val
                break
        # Nothing cached — load full dataset as last resort
        if df is None:
            df, meta = _loader.load(dataset_name)
            _df_cache[full_key] = (df, meta)

    extra = {"weight_column": meta.weight_column}

    exec_result = _executor.execute(code, df=df, extra_globals=extra)

    if not exec_result.success:
        return f"ERROR:\n{exec_result.error}"

    parts = []
    if exec_result.output:
        parts.append(f"Output:\n{exec_result.output}")
    if exec_result.return_value is not None:
        parts.append(f"Result:\n{exec_result.return_value}")
    return "\n".join(parts) if parts else "Code executed successfully (no output)."


@tool
def get_variable_info(dataset_name: str, variable_names: list[str]) -> str:
    """Get basic info about specific variables: dtype, unique values, missing count.

    Args:
        dataset_name: The dataset to inspect.
        variable_names: List of column names to describe.
    """
    schema = _loader.get_schema(dataset_name, sample_rows=0)
    available = set(schema["columns"])
    lines = []
    for var in variable_names:
        if var not in available:
            lines.append(f"{var}: NOT FOUND in dataset")
        else:
            lines.append(f"{var}: dtype={schema['dtypes'].get(var, '?')}")
    return "\n".join(lines)


@tool
def search_columns(dataset_name: str, keyword: str) -> str:
    """Search for column names containing a keyword. Essential for large datasets like GSS (6000+ columns).

    Use this BEFORE loading data to find the right column names.
    For example, search 'educ' to find education-related columns,
    'happy' for happiness, 'income' or 'realinc' for income, etc.

    Args:
        dataset_name: One of 'gss', 'arab_barometer', 'wvs'.
        keyword: Search term (case-insensitive).
    """
    schema = _loader.get_schema(dataset_name, sample_rows=1)
    keyword_lower = keyword.lower()
    matches = [c for c in schema["columns"] if keyword_lower in c.lower()]

    if not matches:
        return f"No columns matching '{keyword}' in {dataset_name}."

    lines = [f"Columns matching '{keyword}' in {dataset_name} ({len(matches)} found):"]
    for col in matches[:30]:
        lines.append(f"  {col}: {schema['dtypes'].get(col, '?')}")
    if len(matches) > 30:
        lines.append(f"  ... and {len(matches) - 30} more")
    return "\n".join(lines)


def get_all_tools() -> list:
    """Return all tools for agent construction."""
    return [load_dataset, get_dataset_schema, run_analysis_code, get_variable_info, search_columns]
