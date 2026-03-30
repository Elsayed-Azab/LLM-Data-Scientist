# LLM Data Scientist

A research platform evaluating how well LLMs perform as data analysts on social science survey data.

## Overview

Users ask analytical questions in natural language. The system generates Python code, executes it against real survey data, and produces statistical results with interpretations. Three agent architectures are compared:

- **Single Agent** — ReAct loop (reason + act iteratively)
- **Multi-Agent** — Planner → Analyst → Reviewer pipeline with retry
- **RAG Agent** — Retrieves codebook context from a vector store before analysis

Supports both **OpenAI** (gpt-4o, gpt-4o-mini) and **Anthropic** (Claude Sonnet, Opus, Haiku) models.

## Datasets

| Dataset | Format | Size | Weight Column |
|---------|--------|------|---------------|
| General Social Survey (GSS) 1972-2024 | Stata (.dta) | ~567 MB | `wtssps` |
| Arab Barometer Wave VIII | CSV/DTA/SAV | ~26 MB | `WT` |
| World Values Survey Wave 7 | CSV | ~182 MB | `W_WEIGHT` |

Place datasets in `Data/` (gitignored).

## Setup

```bash
# Create virtual environment and install
uv venv .venv
uv pip install -e ".[dev]"

# Configure API keys
cp .env.example .env
# Edit .env with your OpenAI and/or Anthropic keys
```

## Usage

### Web Dashboard (Flask)

```bash
.venv/bin/python web/app.py
# Open http://localhost:5001
```

Features:
- Chat-style interactive analysis with real-time status updates
- Agent comparison with charts and tabbed results
- Evaluation results browser with accuracy heatmaps

### CLI

```bash
# Single question
.venv/bin/python experiments/run_experiment.py \
    --agent single --model claude-sonnet-4-20250514 \
    --dataset arab_barometer \
    --question "What is the weighted average age?"

# Full evaluation (all agents x all questions)
.venv/bin/python experiments/run_evaluation.py \
    --model claude-sonnet-4-20250514 --agents single multi rag

# Index codebooks for RAG agent
.venv/bin/python experiments/index_codebooks.py
```

### Tests

```bash
.venv/bin/pytest tests/ -v
```

## Architecture

```
src/
├── agents/           # Agent implementations
│   ├── base.py       # BaseAgent + AnalysisResult interface
│   ├── single_agent.py
│   ├── multi_agent.py
│   ├── rag_agent.py
│   ├── llm_factory.py # Multi-provider LLM creation
│   └── tools.py      # LangChain tools (load, execute, inspect)
├── data/             # Data loading and preprocessing
│   ├── loader.py     # DatasetLoader (GSS/AB/WVS)
│   ├── preprocessor.py
│   ├── registry.py   # Dataset metadata from config
│   └── codebook.py   # PDF codebook parsing
├── analysis/         # Statistics and code execution
│   ├── statistics.py # Weighted mean/std/median/freq/crosstab
│   └── executor.py   # Sandboxed Python execution
├── rag/              # Retrieval-augmented generation
│   ├── indexer.py    # Codebook → ChromaDB
│   └── retriever.py  # Query vector store
├── evaluation/       # Evaluation framework
│   ├── metrics.py    # Scoring functions
│   ├── ground_truth.py
│   └── comparator.py # Run + compare agents
└── reporting/        # Report generation
    └── report.py

web/                  # Flask dashboard
├── app.py
├── templates/        # Jinja2 (home, analysis, comparison, results)
└── static/           # CSS + JS

experiments/          # CLI scripts + question definitions
├── run_experiment.py
├── run_evaluation.py
├── index_codebooks.py
└── questions.yaml
```

## Configuration

`config/default.yaml` defines LLM settings, dataset paths, execution timeouts, and RAG parameters.
