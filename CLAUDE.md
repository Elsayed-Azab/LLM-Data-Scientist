# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research platform evaluating how well LLMs perform as data analysts on social science survey data. Users ask analytical questions in natural language; the system generates code, executes analysis, and produces interpretations across three agent architectures (Single, Multi-Agent, RAG) for comparison.

## Setup & Commands

```bash
# Install dependencies (requires Python 3.10+, uv recommended)
uv venv .venv && uv pip install -e ".[dev]"

# Copy and fill in API keys
cp .env.example .env

# Run tests
.venv/bin/pytest

# Run a single analysis (agents: single, multi, rag)
.venv/bin/python experiments/run_experiment.py --agent single --dataset wvs --question "What is the average age?"

# Run a batch experiment
.venv/bin/python experiments/run_experiment.py --agent multi --dataset arab_barometer --questions experiments/questions.yaml

# Index codebooks for RAG agent (auto-indexes on first use too)
.venv/bin/python experiments/index_codebooks.py

# Run full evaluation (all agents × all questions → comparison report)
.venv/bin/python experiments/run_evaluation.py
.venv/bin/python experiments/run_evaluation.py --agents single rag --datasets wvs
```

## Architecture

Three agent architectures share a common base (`src/agents/base.py` → `BaseAgent.analyze(question, dataset_name) → AnalysisResult`):

1. **SingleAgent** (`src/agents/single_agent.py`) — ReAct agent via LangGraph. One LLM iteratively calls tools until it reaches an answer.
2. **MultiAgent** (`src/agents/multi_agent.py`) — Planner → Analyst → Reviewer pipeline via LangGraph state graph with retry loop (up to 2 retries).
3. **RAGAgent** (`src/agents/rag_agent.py`) — Retrieves codebook context from ChromaDB before analysis. Auto-indexes on first use.

### Key modules

- `src/data/loader.py` — `DatasetLoader` with `.load(name, columns=)` and `.get_schema(name)`. Always use column subsetting for GSS (567MB).
- `src/data/preprocessor.py` — Missing value handling, negative code recoding, column renaming.
- `src/data/registry.py` — Dataset metadata from `config/default.yaml`.
- `src/analysis/statistics.py` — Ground-truth weighted statistics (mean, std, median, frequency, crosstab).
- `src/analysis/executor.py` — `CodeExecutor` runs LLM-generated code in a sandboxed namespace with timeout.
- `src/agents/tools.py` — LangChain tools shared by all agents: `load_dataset`, `get_dataset_schema`, `run_analysis_code`, `get_variable_info`.
- `src/data/codebook.py` — PDF parsing for codebook variable sections.
- `src/rag/indexer.py` — `CodebookIndexer` chunks PDFs into ChromaDB (local all-MiniLM-L6-v2 embeddings).
- `src/rag/retriever.py` — `CodebookRetriever` queries the vector store with dataset-level filtering.
- `src/evaluation/metrics.py` — Scoring functions: numeric, categorical, directional accuracy + LLM-as-judge.
- `src/evaluation/ground_truth.py` — Reference answers computed directly via the statistics module.
- `src/evaluation/comparator.py` — Runs all agents on the question set and produces scored results.
- `src/reporting/report.py` — Generates markdown comparison reports.

## Datasets

All raw data lives in `Data/` (gitignored):

- **GSS** — `Data/GSS_stata/gss7224_r2.dta` (Stata, ~567MB). Weight: `WTSSPS`.
- **Arab Barometer VIII** — `Data/drive-download-20260327T185631Z-3-001/ArabBarometer_WaveVIII_English_v3.*` (CSV/DTA/SAV). Weight: `WT`.
- **World Values Survey 7** — `Data/WVS_Cross-National_Wave_7_csv_v6_0.csv` (~182MB). Weight: `W_WEIGHT`.

Survey columns use coded names (e.g. `Q1`, `Q201A_1`). Always apply survey weights in analysis.

## Project Documents

Reference docs in `archive/`:
- `Group B_ LLM Data Scientist.pdf` — project brief
- `How_good_are_LLMs_as_data_analysts_.pdf` — reference paper
- `System Design.docx` / `Project Planning & Requirements Analysis.docx`

## Config

`config/default.yaml` defines LLM settings, dataset paths/weights, execution timeouts, and RAG parameters. The dataset registry (`src/data/registry.py`) reads from this file.
